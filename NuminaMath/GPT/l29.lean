import Mathlib

namespace series_sum_l29_29666

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29666


namespace infinite_series_sum_eq_3_over_4_l29_29713

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29713


namespace infinite_series_sum_eq_l29_29696

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29696


namespace y2_greater_than_y1_l29_29088

theorem y2_greater_than_y1 : 
  ∀ (y1 y2 : ℝ), 
    (∀ x y : ℝ, (y = -2 * x + 1) → 
    ((x = -1 → y = y1) ∧ (x = -2 → y = y2)) → y2 > y1) :=
by
  intros y1 y2 h
  have hy1 : y1 = -2 * (-1) + 1 := by sorry
  have hy2 : y2 = -2 * (-2) + 1 := by sorry
  rw hy1
  rw hy2
  sorry

end y2_greater_than_y1_l29_29088


namespace series_sum_l29_29661

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29661


namespace compare_exponential_compare_logarithms_l29_29000

-- Given the increasing nature of the exponential function base 2 and the inputs
theorem compare_exponential (h : 0.6 > 0.5) : (2:ℝ)^(0.6) > (2:ℝ)^(0.5) :=
sorry

-- Given the increasing nature of logarithm function base 2 and the inputs
theorem compare_logarithms (h : 3.4 < 3.8) : Real.log(3.4) / Real.log(2) < Real.log(3.8) / Real.log(2) :=
sorry

end compare_exponential_compare_logarithms_l29_29000


namespace probability_x_satisfies_inequality_l29_29458

open Set

theorem probability_x_satisfies_inequality :
  let interval := Ioo 0 5
  let subinterval := Ioo 0 2
  (volume subinterval / volume interval) = 2 / 5 :=
by
  sorry

end probability_x_satisfies_inequality_l29_29458


namespace find_S_2013_l29_29155

variable {a : ℕ → ℤ} -- the arithmetic sequence
variable {S : ℕ → ℤ} -- the sum of the first n terms

-- Conditions
axiom a1_eq_neg2011 : a 1 = -2011
axiom sum_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2
axiom condition_eq : (S 2012 / 2012) - (S 2011 / 2011) = 1

-- The Lean statement to prove that S 2013 = 2013
theorem find_S_2013 : S 2013 = 2013 := by
  sorry

end find_S_2013_l29_29155


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29658

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29658


namespace sum_geometric_series_l29_29684

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29684


namespace trapezium_area_l29_29030

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l29_29030


namespace volume_of_cube_l29_29241

theorem volume_of_cube (edge_sum : ℝ) (h_edge_sum : edge_sum = 48) : 
    let edge_length := edge_sum / 12 
    in edge_length ^ 3 = 64 := 
by
  -- condition
  have h1 : edge_sum = 48 := h_edge_sum
  -- definition of edge_length
  let edge_length := edge_sum / 12
  -- compute the edge_length
  have h2 : edge_length = 4 := by linarith
  -- compute the volume of the cube
  have volume := edge_length ^ 3
  -- provide proof that volume is 64
  have h3 : volume = 64 := by linarith; sorry
  exact h3

end volume_of_cube_l29_29241


namespace complex_conjugate_of_z_l29_29866

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := 25 / (3 - 4 * i)

-- State the theorem to prove the complex conjugate of z
theorem complex_conjugate_of_z : Complex.conj z = 3 - 4 * i :=
by sorry

end complex_conjugate_of_z_l29_29866


namespace intersection_points_slope_l29_29842

theorem intersection_points_slope :
  ∀ (t : ℝ), (2 * (33 * t + 17) / 13 + 3 * (17 * t + 6) / 13 = 9 * t + 4) ∧
             (3 * (33 * t + 17) / 13 - 2 * (17 * t + 6) / 13 = 5 * t + 3) →
             ∃ (m : ℝ), m = 221 / 429 :=
by
  intro t h
  use 221 / 429
  sorry

end intersection_points_slope_l29_29842


namespace sum_geometric_series_l29_29677

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29677


namespace pascal_third_smallest_four_digit_number_l29_29391

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29391


namespace cost_of_memory_cards_l29_29945

-- Given conditions
def days_per_year : ℕ := 365
def years : ℕ := 6
def pictures_per_day : ℕ := 25
def pictures_per_card : ℕ := 40
def card_cost : ℕ := 75

-- Lean theorem to prove the total cost
theorem cost_of_memory_cards : 
  let total_pictures := days_per_year * years * pictures_per_day in
  let cards_needed := (total_pictures + pictures_per_card - 1) / pictures_per_card in
  let total_cost := cards_needed * card_cost in
  total_cost = 102675 := 
by
  sorry

end cost_of_memory_cards_l29_29945


namespace find_x_from_log_condition_l29_29025

theorem find_x_from_log_condition (x : ℝ) (h : log 8 (3 * x - 4) = 5 / 3) : x = 12 :=
by
  sorry

end find_x_from_log_condition_l29_29025


namespace third_smallest_four_digit_in_pascal_triangle_l29_29374

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29374


namespace evaluate_series_sum_l29_29605

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29605


namespace length_segment_AB_l29_29152

theorem length_segment_AB 
  (α θ : ℝ) 
  (x y : ℝ → ℝ)
  (C1 : ∀ α, x α = 2 * Real.cos α ∧ y α = 2 + 2 * Real.sin α)
  (C2 : θ → ℝ) 
  (ray := 2 * Real.pi / 3) : 
  (2 * Real.sin ray - 4 * Real.sin ray) = 2 * Real.sqrt 3 := by sorry

end length_segment_AB_l29_29152


namespace series_sum_l29_29660

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29660


namespace square_approximation_error_l29_29997

theorem square_approximation_error :
  ∀ (k : ℝ) (half_PQ : ℝ), 
  k = 1 → 
  half_PQ = 1 / 2 → 
  let OQ_len := (half_PQ * sqrt 5) / k in
  let OU_len := 2 / OQ_len in
  let initial_area := 4 * OU_len ^ 2 in
  initial_area - π = 0.0584074 → 
  ∃ (shift_fraction : ℝ), 
  shift_fraction = 1 / 44 → 
  let PS' := 23 / 44 in
  let new_OQ_len := sqrt (PS' ^ 2 + 1) in
  let new_OU_len := 2 / new_OQ_len in
  let new_area := 4 * new_OU_len ^ 2 in
  new_area = 3.1415822 ∧ abs (new_area - π) = 0.0000104 :=
by {
  intros k half_PQ hk hhalf_PQ,
  let OQ_len := (half_PQ * sqrt 5) / k,
  let OU_len := 2 / OQ_len,
  let initial_area := 4 * OU_len ^ 2,
  assume h_initial_area,
  let shift_fraction := 1 / 44,
  let PS' := 23 / 44,
  let new_OQ_len := sqrt (PS' ^ 2 + 1),
  let new_OU_len := 2 / new_OQ_len,
  let new_area := 4 * new_OU_len ^ 2,
  have h_new_area : new_area = 3.1415822,
  { sorry },
  have h_error : abs (new_area - π) = 0.0000104,
  { sorry },
  exact ⟨shift_fraction, rfl, ⟨h_new_area, h_error⟩⟩
}

end square_approximation_error_l29_29997


namespace find_f_neg2_l29_29431

noncomputable def f (x : ℝ) : ℝ := g x + 2
axiom g_odd : ∀ x, g (-x) = -g x
axiom f_2_eq_3 : f 2 = 3

theorem find_f_neg2 : f (-2) = 1 := by
  sorry

end find_f_neg2_l29_29431


namespace series_converges_to_three_fourths_l29_29811

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29811


namespace find_b1_coefficient_l29_29935

theorem find_b1_coefficient :
  let f := (8 + 32 * x - 12 * x^2 - 4 * x^3 + x^4) in
  let g := (64 - 1024 * y + 144 * y^2 - 16 * y^3 + y^4) in
  -- Polynomial f(x)
  f.has_roots {x_1, x_2, x_3, x_4} →
  -- Polynomial g(y) corresponds to roots squares
  g.has_roots {x_1^2, x_2^2, x_3^2, x_4^2} ∧ g.coefficient 1 = -1024 :=
sorry

end find_b1_coefficient_l29_29935


namespace series_sum_eq_l29_29533

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29533


namespace age_sum_squares_l29_29052

theorem age_sum_squares (a b c : ℕ) (h1 : 5 * a + 2 * b = 3 * c) (h2 : 3 * c^2 = 4 * a^2 + b^2) (h3 : Nat.gcd (Nat.gcd a b) c = 1) : a^2 + b^2 + c^2 = 18 :=
sorry

end age_sum_squares_l29_29052


namespace cubeRootThree_expression_value_l29_29898

-- Define the approximate value of cube root of 3
def cubeRootThree : ℝ := 1.442

-- Lean theorem statement
theorem cubeRootThree_expression_value :
  cubeRootThree - 3 * cubeRootThree - 98 * cubeRootThree = -144.2 := by
  sorry

end cubeRootThree_expression_value_l29_29898


namespace sum_series_div_3_powers_l29_29793

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29793


namespace value_of_y_l29_29133

variables (x y : ℝ)

theorem value_of_y (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 :=
by
  sorry

end value_of_y_l29_29133


namespace evaluate_series_sum_l29_29593

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29593


namespace intersection_complement_set_eq_l29_29191

variable (U : Set ℝ) (A B : Set ℝ)

def U := { x : ℝ | true }

def A := { y : ℝ | ∃ x : ℝ, y = x^2 - 2 }

def B := { x : ℝ | x ≥ 3 }

theorem intersection_complement_set_eq :
  A ∩ (U \ B) = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_complement_set_eq_l29_29191


namespace min_segments_required_l29_29851

noncomputable def min_segments (n : ℕ) : ℕ := (3 * n - 2 + 1) / 2

theorem min_segments_required (n : ℕ) (h : ∀ (A B : ℕ) (hA : A < n) (hB : B < n) (hAB : A ≠ B), 
  ∃ (C : ℕ), C < n ∧ (C ≠ A) ∧ (C ≠ B)) : 
  min_segments n = ⌈ (3 * n - 2 : ℝ) / 2 ⌉ := 
sorry

end min_segments_required_l29_29851


namespace series_result_l29_29637

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29637


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29340

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29340


namespace sum_series_eq_3_div_4_l29_29566

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29566


namespace third_smallest_four_digit_in_pascals_triangle_l29_29311

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29311


namespace find_range_of_f_in_interval_l29_29516

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem find_range_of_f_in_interval :
  ∀ x ∈ (Set.Icc 0 (Real.pi / 2)), f x ∈ Set.Icc (-3 / 2) 3 := 
by
  intros,
  sorry

end find_range_of_f_in_interval_l29_29516


namespace find_abc_l29_29430

noncomputable def log (x : ℝ) : ℝ := sorry -- Replace sorry with an actual implementation of log function if needed

theorem find_abc (a b c : ℝ) 
    (h1 : 1 ≤ a) 
    (h2 : 1 ≤ b) 
    (h3 : 1 ≤ c)
    (h4 : a * b * c = 10)
    (h5 : a^(log a) * b^(log b) * c^(log c) ≥ 10) :
    (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) := 
by
  sorry

end find_abc_l29_29430


namespace tangent_perpendicular_point_l29_29104

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

theorem tangent_perpendicular_point :
  ∃ x0, (f x0 = 1) ∧ (x0 = 0) :=
sorry

end tangent_perpendicular_point_l29_29104


namespace third_smallest_four_digit_in_pascals_triangle_l29_29386

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29386


namespace multiples_of_4_count_l29_29120

theorem multiples_of_4_count (a b : ℕ) (h₁ : a = 100) (h₂ : b = 400) :
  ∃ n : ℕ, n = 75 ∧ ∀ k : ℕ, (k >= a ∧ k <= b ∧ k % 4 = 0) ↔ (k / 4 - 25 ≥ 1 ∧ k / 4 - 25 ≤ n) :=
sorry

end multiples_of_4_count_l29_29120


namespace max_pqrs_squared_l29_29173

theorem max_pqrs_squared (p q r s : ℝ)
  (h1 : p + q = 18)
  (h2 : pq + r + s = 85)
  (h3 : pr + qs = 190)
  (h4 : rs = 120) :
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
sorry

end max_pqrs_squared_l29_29173


namespace solve_positive_solution_l29_29830

noncomputable def y (x: ℝ) := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + ...)))
noncomputable def z (x: ℝ) := Real.sqrt (x - Real.sqrt (x - Real.sqrt (x - ...)))

theorem solve_positive_solution :
  ∃ x : ℝ, x = (3 + Real.sqrt 5) / 2 ∧ y x = z x :=
by
  sorry

end solve_positive_solution_l29_29830


namespace third_smallest_four_digit_Pascal_triangle_l29_29289

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29289


namespace inequality_proof_l29_29859

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 := 
sorry

end inequality_proof_l29_29859


namespace roots_polynomial_identity_l29_29187

theorem roots_polynomial_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a * b + b * c + c * a = 22) (h3 : a * b * c = 8) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  sorry

end roots_polynomial_identity_l29_29187


namespace solve_equation_l29_29993

theorem solve_equation (x : ℝ) : 
  (9 - x - 2 * (31 - x) = 27) → (x = 80) :=
by
  sorry

end solve_equation_l29_29993


namespace sum_of_coeffs_neg_expanded_expr_l29_29813

theorem sum_of_coeffs_neg_expanded_expr : 
  let expr := -(2 * x - 4) * (2 * x + 3 * (2 * x - 4))
  let expanded_expr := -16 * x ^ 2 + 56 * x - 48
  sum (coefficients expanded_expr) = -8 :=
by
  sorry

end sum_of_coeffs_neg_expanded_expr_l29_29813


namespace imaginary_part_of_z_l29_29968

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := (1 + 2 * i) / i

-- Prove that the imaginary part of z is -1.
theorem imaginary_part_of_z : complex.imaginaryPart z = -1 :=
by
  sorry

end imaginary_part_of_z_l29_29968


namespace sum_series_equals_three_fourths_l29_29546

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29546


namespace find_domain_F_l29_29884

noncomputable def domain_F {f : ℝ → ℝ} (dom_f : Set.Ioo 0 1) (a : ℝ)
  (ha : - (1:ℝ)/2 < a ∧ a < (1:ℝ)/2 ∧ a ≠ 0) : Set ℝ :=
  if a < 0 then {| x | -a < x ∧ x < 1 + a |} else {| x | a < x ∧ x < 1 - a |}

theorem find_domain_F 
  {f : ℝ → ℝ} (dom_f : Set.Ioo 0 1) (a : ℝ)
  (ha : - (1:ℝ)/2 < a ∧ a < (1:ℝ)/2 ∧ a ≠ 0) :
  (F : ℝ → ℝ) := 
    let F (x : ℝ) := f (x + a) + f (x - a) in
    F = domain_F dom_f a ha := sorry

end find_domain_F_l29_29884


namespace zero_in_interval_f_l29_29229

noncomputable def f (x : ℝ) := 3^x + 2*x - 3

theorem zero_in_interval_f : ∃ c ∈ set.Ioo (0:ℝ) (1:ℝ), f c = 0 :=
begin
  let f := λ x, 3^x + 2*x - 3,
  have h0 : f 0 < 0, by {
    calc
      f 0 = 3^0 + 2*0 - 3 : by refl
         ... = 1 - 3       : by norm_num
         ... < 0           : by norm_num
  },
  have h1 : f 1 > 0, by {
    calc
      f 1 = 3^1 + 2*1 - 3 : by refl
         ... = 3 + 2 - 3  : by norm_num
         ... = 2          : by norm_num
         ... > 0          : by norm_num
  },
  exact exists_zero_in_interval h0 h1,
end

end zero_in_interval_f_l29_29229


namespace common_difference_of_arithmetic_sequence_l29_29078

def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n+1) = a_n n + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a_n 0 + a_n n)) / 2

theorem common_difference_of_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) :
  a_n 1 = 7 ∧ sum_of_first_n_terms a_n 4 = 40 → d = 2 :=
by
  assume h : a_n 1 = 7 ∧ sum_of_first_n_terms a_n 4 = 40
  sorry

end common_difference_of_arithmetic_sequence_l29_29078


namespace series_sum_eq_l29_29528

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29528


namespace sum_of_series_eq_three_fourths_l29_29584

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29584


namespace range_of_m_min_value_of_f_solve_quadratic_inequality_l29_29111

-- Problem (1)
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : -1 ≤ m) (h2 : m ≤ 2) : x^2 + 2*m*x + m + 2 ≥ 0 := 
sorry

-- Problem (2)
theorem min_value_of_f : ∃ m : ℝ, m = real.sqrt 3 - 2 ∧ (∀ x : ℝ, f x := x + 3/(x+2) ∧ f m = 2*real.sqrt 3 - 2) := 
sorry

-- Problem (3)
theorem solve_quadratic_inequality (m : ℝ) (h1 : 2 - 2 * real.sqrt 3 ≤ m) (h2 : m ≤ 2 + 2 * real.sqrt 3) : 
  ∀ x : ℝ, (x + m) * (x - 3) > 0 ↔ x ∈ set.Ioo (-m) 3 ∨ x ∈ set.Ici 3 :=
sorry

end range_of_m_min_value_of_f_solve_quadratic_inequality_l29_29111


namespace third_smallest_four_digit_in_pascal_triangle_l29_29371

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29371


namespace perimeter_of_fence_l29_29021

noncomputable def n : ℕ := 18
noncomputable def w : ℝ := 0.5
noncomputable def d : ℝ := 4

theorem perimeter_of_fence : 3 * ((n / 3 - 1) * d + n / 3 * w) = 69 := by
  sorry

end perimeter_of_fence_l29_29021


namespace ratio_of_mc_to_fr_l29_29219

theorem ratio_of_mc_to_fr (M F T : ℕ) (hT : T = 6) (hF : F = T + 7) (h : M + F + T = 45) :
  M / F = 2 := 
by
  -- Extract given values
  have hF_value : F = 13 := by simp [hT, hF]
  have hM_value : M = 26 := by linarith [h, hF_value, hT]
  -- Simplify the ratio
  calc
    M / F = 26 / 13 := by simp [hF_value, hM_value]
    ... = 2 : by norm_num

#check ratio_of_mc_to_fr

end ratio_of_mc_to_fr_l29_29219


namespace range_of_g_l29_29017

def arctan (x : ℝ) : ℝ := Real.arctan x
def arccot (x : ℝ) : ℝ := Real.pi / 2 - Real.arctan x

def g (x : ℝ) : ℝ :=
  (arctan (x / 3))^2 - Real.pi * arccot (x / 3) + (arccot (x / 3))^2 - (Real.pi^2 / 18) * (3 * x^2 - x + 2)

theorem range_of_g : Set.univ = {y : ℝ | ∃ x, g x = y} :=
by sorry

end range_of_g_l29_29017


namespace exists_nat_numbers_except_two_three_l29_29819

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l29_29819


namespace find_third_number_l29_29223

noncomputable def averageFirstSet (x : ℝ) : ℝ := (20 + 40 + x) / 3
noncomputable def averageSecondSet : ℝ := (10 + 70 + 16) / 3

theorem find_third_number (x : ℝ) (h : averageFirstSet x = averageSecondSet + 8) : x = 60 :=
by
  sorry

end find_third_number_l29_29223


namespace sum_of_series_eq_three_fourths_l29_29583

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29583


namespace area_of_absolute_value_sum_l29_29513

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end area_of_absolute_value_sum_l29_29513


namespace surface_area_beach_ball_l29_29461

-- Define the problem
def diameter := 15 -- the diameter of the beach ball in inches
def radius := diameter / 2 -- calculate the radius from the diameter
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2 -- formula for surface area of a sphere

-- The theorem to prove the surface area of the beach ball equals 225π square inches
theorem surface_area_beach_ball : surface_area radius = 225 * Real.pi :=
by
  sorry -- Proof steps will go here

end surface_area_beach_ball_l29_29461


namespace fans_received_all_three_items_l29_29519

theorem fans_received_all_three_items :
  let lcm := Nat.lcm (Nat.lcm 75 45) 50 in
  ∀ (total_fans lcm_fans max_limit : ℕ),
    total_fans = 5000 →
    lcm_fans = total_fans / lcm →
    max_limit = 100 →
    lcm_fans ≤ max_limit →
    lcm_fans = 11 :=
by
  intros lcm total_fans lcm_fans max_limit h_total_fans h_lcm_fans h_max_limit h_lcm_fans_max
  sorry

end fans_received_all_three_items_l29_29519


namespace min_quadratic_form_value_l29_29047

noncomputable def quadratic_form : ℝ → ℝ → ℝ :=
  λ x y, 3 * x ^ 2 + 4 * x * y + 5 * y ^ 2 - 8 * x - 10 * y

theorem min_quadratic_form_value: ∃ x y: ℝ, quadratic_form x y = -4.45 :=
by
  sorry

end min_quadratic_form_value_l29_29047


namespace series_sum_correct_l29_29766

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29766


namespace sum_series_equals_three_fourths_l29_29541

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29541


namespace sum_series_equals_three_fourths_l29_29551

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29551


namespace present_value_approx_l29_29123

noncomputable def present_value (F : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  F / (1 + r)^n

theorem present_value_approx :
  present_value 600000 0.04 8 ≈ 438447.25 :=
sorry

end present_value_approx_l29_29123


namespace spot_reachable_area_l29_29916

/-- 
In an equilateral triangle with side length 1 yard, Spot is tethered to a vertex with a 3-yard rope. 
Prove that the area outside the triangle that Spot can reach is 23π/3 square yards.
-/
theorem spot_reachable_area (side_length : ℝ) (rope_length : ℝ) (tether_angle : ℝ) 
    (triangle_area : ℝ) (reachable_area : ℝ) : 
    side_length = 1 ∧ rope_length = 3 ∧ tether_angle = π/3 ∧ 
    triangle_area = (√3 / 4) * side_length^2 ∧ reachable_area = 23 * π / 3 := 
by 
  split;
  sorry

end spot_reachable_area_l29_29916


namespace find_angle_between_vectors_l29_29087

variables {a b : ℝ^3}  -- Assuming vectors are in 3-dimensional space for generality

-- Given conditions
def norm_a : ℝ := ∥a∥ = 1
def norm_b : ℝ := ∥b∥ = 2
def dot_product_condition : ℝ := a • (a - b) = 3

-- Proof problem statement
theorem find_angle_between_vectors 
  (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = 2) 
  (h3 : a • (a - b) = 3) : 
  real.acos ((a • b) / (∥a∥ * ∥b∥)) = real.pi := 
sorry

end find_angle_between_vectors_l29_29087


namespace number_of_men_in_first_group_l29_29218

-- Define the conditions as hypotheses in Lean
def work_completed_in_25_days (x : ℕ) : Prop := x * 25 * (1 : ℚ) / (25 * x) = (1 : ℚ)
def twenty_men_complete_in_15_days : Prop := 20 * 15 * (1 : ℚ) / 15 = (1 : ℚ)

-- Define the theorem to prove the number of men in the first group
theorem number_of_men_in_first_group (x : ℕ) (h1 : work_completed_in_25_days x)
  (h2 : twenty_men_complete_in_15_days) : x = 20 :=
  sorry

end number_of_men_in_first_group_l29_29218


namespace identify_triple_in_three_queries_l29_29016

-- Function definition based on the problem's given response function
def response (X Y Z a b c : ℕ) : ℕ :=
  |X + Y - a - b| + |Y + Z - b - c| + |Z + X - c - a|

-- Set of all triples where each component is a non-negative integer less than 10
noncomputable def T : List (ℕ × ℕ × ℕ) :=
  List.product (List.range 10) (List.product (List.range 10) (List.range 10))

-- Definition of the proof problem: B needs at most 3 queries to identify (X, Y, Z)
theorem identify_triple_in_three_queries :
  ∀ (X Y Z : ℕ) (hX: X < 10) (hY: Y < 10) (hZ: Z < 10),
    ∃ (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ),
    response X Y Z a1 b1 c1 = response X Y Z a2 b2 c2 ∧
    response X Y Z a2 b2 c2 = response X Y Z a3 b3 c3 ∧
    ∀ (x y z: ℕ), x < 10 → y < 10 → z < 10 →
      (response x y z a1 b1 c1 ≠ response X Y Z a1 b1 c1 ∨ 
       response x y z a2 b2 c2 ≠ response X Y Z a2 b2 c2 ∨ 
       response x y z a3 b3 c3 ≠ response X Y Z a3 b3 c3) →
      (x = X ∧ y = Y ∧ z = Z) := sorry

end identify_triple_in_three_queries_l29_29016


namespace find_s_t_l29_29890

-- Define the conditions of the problem
variable (s t : ℝ)
def A := (Set.Icc s (s + 1/6)) ∪ (Set.Icc t (t + 1))
def f (x : ℝ) := (x + 1) / (x - 1)

theorem find_s_t : (1 ∉ A s t) → (s + 1/6 < t) → 
  (∀ x ∈ A s t, f x ∈ A s t) → 
  (s + t = 11/2 ∨ s + t = 3/2) :=
by
  assume h1 h2 h3
  sorry

end find_s_t_l29_29890


namespace number_of_valid_c_l29_29841

theorem number_of_valid_c : 
  (∃ (f : ℕ → ℕ → ℕ), (∀ x : ℕ, sin (π * x) = 0 → ∃ c ∈ Icc 0 10000, f ⌊x⌋ ⌈x⌉ = c)) = 334 :=
sorry

end number_of_valid_c_l29_29841


namespace find_y_l29_29051

-- Define vectors as tuples
def vector_1 : ℝ × ℝ := (3, 4)
def vector_2 (y : ℝ) : ℝ × ℝ := (y, -5)

-- Define dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for orthogonality
def orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem we want to prove
theorem find_y (y : ℝ) :
  orthogonal vector_1 (vector_2 y) → y = (20 / 3) :=
by
  sorry

end find_y_l29_29051


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29659

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29659


namespace IsoscelesTriangle_tangent_relation_l29_29079

variables {A B C O P Q : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace P] [MetricSpace Q]
variables (cosets : IsoscelesTriangle A B C O) (omega : Circle O)
variables (tangentP : Tangent PQ A omega) (tangentQ : Tangent PQ B omega)

theorem IsoscelesTriangle_tangent_relation :
  ∀ {A B C O P Q : Type}, ∀ [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace P] [MetricSpace Q],
  IsoscelesTriangle A B C O →
  Circle O →
  Tangent PQ A (Circle O) →
  Tangent PQ B (Circle O) →
  AP * BQ = (AO)^2 := sorry

end IsoscelesTriangle_tangent_relation_l29_29079


namespace chip_drawing_probability_l29_29436

theorem chip_drawing_probability :
  let total_chips := 12,
      purple_chips := 4,
      orange_chips := 3,
      green_chips := 5,
      total_permutations := total_chips.factorial,
      constrained_permutations := 2.factorial * purple_chips.factorial * orange_chips.factorial * green_chips.factorial
  in (constrained_permutations : ℚ) / total_permutations = 1 / 13860 := by
  sorry

end chip_drawing_probability_l29_29436


namespace sum_geometric_series_l29_29683

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29683


namespace jose_initial_land_l29_29948

theorem jose_initial_land {total_siblings : ℕ} {land_per_sibling : ℕ} (h1 : total_siblings = 5) (h2 : land_per_sibling = 4000) :
  (total_siblings * land_per_sibling = 20000) :=
by {
  rw [h1, h2],
  norm_num,
  sorry,
}

end jose_initial_land_l29_29948


namespace sum_series_eq_3_div_4_l29_29574

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29574


namespace flour_spill_ratio_l29_29973

def initial_flour : ℕ := 500
def flour_used_for_baking : ℕ := 240
def flour_needed_to_buy : ℕ := 370

theorem flour_spill_ratio :
  let flour_left_after_baking := initial_flour - flour_used_for_baking in
  let flour_left_after_spill := initial_flour - flour_needed_to_buy in
  let flour_spilled := flour_left_after_baking - flour_left_after_spill in
  (flour_spilled : ℕ) / (flour_left_after_baking : ℕ) = 1 / 2 :=
by sorry

end flour_spill_ratio_l29_29973


namespace problem_l29_29136

noncomputable def x : ℝ := 123.75
noncomputable def y : ℝ := 137.5
noncomputable def original_value : ℝ := 125

theorem problem (y_more : y = original_value + 0.1 * original_value) (x_less : x = y * 0.9) : y = 137.5 :=
by
  sorry

end problem_l29_29136


namespace series_sum_eq_l29_29534

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29534


namespace remainder_n_squared_plus_3n_plus_5_l29_29907

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l29_29907


namespace third_smallest_four_digit_in_pascal_triangle_l29_29372

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29372


namespace S_17_33_50_sum_l29_29177

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    - (n / 2)
  else
    (n + 1) / 2

theorem S_17_33_50_sum : S 17 + S 33 + S 50 = 1 :=
by
  sorry

end S_17_33_50_sum_l29_29177


namespace percent_primes_divisible_by_2_below_12_l29_29413

-- Definition of prime numbers less than 12
def prime_numbers_below_12 := {2, 3, 5, 7, 11}

-- Condition: Number divisible by 2
def divisible_by_2 (n : ℕ) := n % 2 = 0

-- Number of primes less than 12 that are divisible by 2
def num_divisible_by_2 : ℕ := (prime_numbers_below_12.to_list.filter divisible_by_2).length

-- Total number of prime numbers less than 12
def total_primes_below_12 : ℕ := prime_numbers_below_12.to_list.length

-- Percentage calculation
def percentage_divisible_by_2 : ℕ := (num_divisible_by_2 * 100) / total_primes_below_12

theorem percent_primes_divisible_by_2_below_12 : percentage_divisible_by_2 = 20 := by
  sorry

end percent_primes_divisible_by_2_below_12_l29_29413


namespace series_result_l29_29629

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29629


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29333

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29333


namespace intersection_point_of_curve_and_circle_l29_29926

theorem intersection_point_of_curve_and_circle (
    {a b r : ℝ}
    (h1 : ∀ x y, (x, y) = (4, 1/2) ∨ (x, y) = (-2, -1) ∨ (x, y) = (2/5, 5) ∨ (x, y) = (-16/5, -5/8))
    (h_curve : ∀ x y, (x * y = 2))
    (h_circle : ∀ x y, ((x - a)^2 + (y - b)^2 = r^2))
  ) :
  (∃ p, p = (-16/5, -5/8)) ∧ p ∈ { (x, y) | (x * y = 2) ∧ ((x - a)^2 + (y - b)^2 = r^2) } :=
  sorry

end intersection_point_of_curve_and_circle_l29_29926


namespace sequence_formula_l29_29889

theorem sequence_formula (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l29_29889


namespace unique_sequence_l29_29509

noncomputable def sequence_satisfies {n : ℕ} (x : ℕ → ℕ) :=
  ∃ a : ℕ, (∑ i in finset.range 2011, (i+1) * (x (i+1))^n) = a^(n+1) + 1

theorem unique_sequence : 
  ∀ (x : ℕ → ℕ), (∀ n : ℕ, n > 0 → sequence_satisfies x) → 
  ∀ i : ℕ, i > 0 → x i = if i = 1 then 1 else 2023065 := 
sorry

end unique_sequence_l29_29509


namespace x_fifth_plus_inverse_fifth_l29_29457

theorem x_fifth_plus_inverse_fifth {x : ℝ} (hx_pos : 0 < x) (h : x^2 + x⁻² = 7) : x^5 + x⁻^5 = 123 :=
by
  sorry

end x_fifth_plus_inverse_fifth_l29_29457


namespace factorial_expression_l29_29002

theorem factorial_expression :
  10! - 9! + 8! - 7! = 3301200 :=
by
  sorry

end factorial_expression_l29_29002


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29643

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29643


namespace simplify_and_evaluate_expr_l29_29211

-- Define x
def x : ℝ := Real.sqrt 2 - 1

-- Define the expression
def expr : ℝ := (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x))

-- State the theorem which asserts the equality
theorem simplify_and_evaluate_expr : expr = -Real.sqrt 2 / 2 := 
by 
  sorry

end simplify_and_evaluate_expr_l29_29211


namespace domain_of_f_l29_29227

-- Define the function f(x) = log(x^3 - x^2)
def f (x : ℝ) : ℝ := Real.log (x^3 - x^2)

-- Define the condition for which x^3 - x^2 is greater than 0
def valid_domain (x : ℝ) : Prop := x^3 - x^2 > 0

-- The mathematically equivalent proof problem can now be written as:
theorem domain_of_f :
  (∀ x : ℝ, valid_domain x → x ∈ set.Ioi 1) := sorry

end domain_of_f_l29_29227


namespace sum_geometric_series_l29_29678

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29678


namespace triangle_side_ratio_triangle_area_l29_29914

-- Definition of Problem 1
theorem triangle_side_ratio {A B C a b c : ℝ} 
  (h1 : 4 * Real.sin A = 3 * Real.sin B)
  (h2 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h3 : a / b = Real.sin A / Real.sin B)
  (h4 : b / c = Real.sin B / Real.sin C)
  : c / b = 5 / 4 :=
sorry

-- Definition of Problem 2
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : C = 2 * Real.pi / 3)
  (h2 : c - a = 8)
  (h3 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h4 : a + c = 2 * b)
  : (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 :=
sorry

end triangle_side_ratio_triangle_area_l29_29914


namespace sum_of_first_six_terms_of_geom_seq_l29_29836

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l29_29836


namespace series_sum_l29_29670

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29670


namespace hyperbola_center_correct_l29_29825

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (3 * y - 3)^2 / 7^2 - (4 * x - 5)^2 / 3^2 = 1

-- Define the coordinates of the center
def hyperbola_center (h k : ℝ) : Prop :=
  h = 5 / 4 ∧ k = 1

-- The final proof statement
theorem hyperbola_center_correct : ∃ h k : ℝ, hyperbola_center h k ∧ hyperbola_eq h k :=
begin
  use [5 / 4, 1],
  split,
  { exact ⟨rfl, rfl⟩ },
  { unfold hyperbola_eq,
    norm_num, }
end

end hyperbola_center_correct_l29_29825


namespace infinite_series_sum_eq_3_over_4_l29_29719

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29719


namespace third_smallest_four_digit_in_pascals_triangle_l29_29300

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29300


namespace merchant_discount_percentage_l29_29450

theorem merchant_discount_percentage
  (CP MP SP : ℝ)
  (h1 : MP = CP + 0.40 * CP)
  (h2 : SP = CP + 0.26 * CP)
  : ((MP - SP) / MP) * 100 = 10 := by
  sorry

end merchant_discount_percentage_l29_29450


namespace series_sum_eq_l29_29527

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29527


namespace lisa_total_spoons_l29_29193

def children_count : ℕ := 6
def spoons_per_child : ℕ := 4
def decorative_spoons : ℕ := 4
def large_spoons : ℕ := 20
def dessert_spoons : ℕ := 10
def soup_spoons : ℕ := 15
def tea_spoons : ℕ := 25

def baby_spoons_total : ℕ := children_count * spoons_per_child
def cutlery_set_total : ℕ := large_spoons + dessert_spoons + soup_spoons + tea_spoons

def total_spoons : ℕ := cutlery_set_total + baby_spoons_total + decorative_spoons

theorem lisa_total_spoons : total_spoons = 98 :=
by
  sorry

end lisa_total_spoons_l29_29193


namespace sum_series_eq_3_over_4_l29_29750

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29750


namespace perpendicular_to_parallel_l29_29815

noncomputable theory

open EuclideanGeometry

-- Define parallel lines and perpendicularly in 3D space
def parallel_lines {P : Type} [EuclideanGeometry P] (l1 l2 : line P) := ∀ (p1 ∈ l1) (p2 ∈ l2), ∃ (d : P → P → ℚ), (d(p1, p2) = 0)

def perpendicular_lines {P : Type} [EuclideanGeometry P] (l1 l2 : line P) := ∀ (p1 ∈ l1) (p2 ∈ l2), ∃ (v1 v2 : vector P), (dot_product(v1, v2) = 0)

theorem perpendicular_to_parallel 
  {P : Type} [EuclideanGeometry P]
  (l m n : line P) (h_parallel : parallel_lines l m) (h_perpendicular : perpendicular_lines n l) :
  perpendicular_lines n m := sorry

end perpendicular_to_parallel_l29_29815


namespace sum_of_series_eq_three_fourths_l29_29579

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29579


namespace donna_pays_total_l29_29019

def original_price_vase : ℝ := 250
def discount_vase : ℝ := original_price_vase * 0.25

def original_price_teacups : ℝ := 350
def discount_teacups : ℝ := original_price_teacups * 0.30

def original_price_plate : ℝ := 450
def discount_plate : ℝ := 0

def original_price_ornament : ℝ := 150
def discount_ornament : ℝ := original_price_ornament * 0.20

def membership_discount_vase : ℝ := (original_price_vase - discount_vase) * 0.05
def membership_discount_plate : ℝ := original_price_plate * 0.05

def tax_vase : ℝ := ((original_price_vase - discount_vase - membership_discount_vase) * 0.12)
def tax_teacups : ℝ := ((original_price_teacups - discount_teacups) * 0.08)
def tax_plate : ℝ := ((original_price_plate - membership_discount_plate) * 0.10)
def tax_ornament : ℝ := ((original_price_ornament - discount_ornament) * 0.06)

def final_price_vase : ℝ := (original_price_vase - discount_vase - membership_discount_vase) + tax_vase
def final_price_teacups : ℝ := (original_price_teacups - discount_teacups) + tax_teacups
def final_price_plate : ℝ := (original_price_plate - membership_discount_plate) + tax_plate
def final_price_ornament : ℝ := (original_price_ornament - discount_ornament) + tax_ornament

def total_price : ℝ := final_price_vase + final_price_teacups + final_price_plate + final_price_ornament

theorem donna_pays_total :
  total_price = 1061.55 :=
by
  sorry

end donna_pays_total_l29_29019


namespace sum_series_eq_3_over_4_l29_29759

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29759


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29337

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29337


namespace reading_time_difference_in_minutes_l29_29420

noncomputable def xanthia_reading_speed : ℝ := 120 -- pages per hour
noncomputable def molly_reading_speed : ℝ := 60 -- pages per hour
noncomputable def book_length : ℝ := 360 -- pages

theorem reading_time_difference_in_minutes :
  let time_for_xanthia := book_length / xanthia_reading_speed
  let time_for_molly := book_length / molly_reading_speed
  let difference_in_hours := time_for_molly - time_for_xanthia
  difference_in_hours * 60 = 180 :=
by
  sorry

end reading_time_difference_in_minutes_l29_29420


namespace area_enclosed_by_graph_eq_160_l29_29510

theorem area_enclosed_by_graph_eq_160 :
  ∃ (area : ℝ), area = 160 ∧
  (∀ (x y : ℝ), |2 * x| + |5 * y| = 20 → abs x ≤ 10 ∧ abs y ≤ 4) :=
begin
  sorry
end

end area_enclosed_by_graph_eq_160_l29_29510


namespace infinite_series_sum_eq_l29_29707

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29707


namespace sum_geometric_series_l29_29687

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29687


namespace series_sum_l29_29664

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29664


namespace value_of_c_l29_29899

theorem value_of_c (a b c : ℚ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 7) (h3 : a - b + 3 = c - 2 * b) : c = 21 / 2 :=
sorry

end value_of_c_l29_29899


namespace total_and_average_games_l29_29939

def football_games_per_month : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games_per_month : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games_per_month : List Nat := [17, 21, 14, 32, 22, 27]

def total_games (games_per_month : List Nat) : Nat :=
  List.sum games_per_month

def average_games (total : Nat) (months : Nat) : Nat :=
  total / months

theorem total_and_average_games :
  total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month = 486
  ∧ average_games (total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month) 6 = 81 :=
by
  sorry

end total_and_average_games_l29_29939


namespace inequality_solution_l29_29061

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

lemma monotone_decreasing (a : ℝ) : ∀ x y : ℝ, x < y → f a y < f a x := 
sorry

lemma odd_function (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 0 := 
sorry

theorem inequality_solution (t : ℝ) (a : ℝ) (h_monotone : ∀ x y : ℝ, x < y → f a y < f a x)
    (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : t ≥ 4 / 3 ↔ f a (2 * t + 1) + f a (t - 5) ≤ 0 := 
sorry

end inequality_solution_l29_29061


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29657

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29657


namespace series_result_l29_29627

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29627


namespace evaluate_series_sum_l29_29595

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29595


namespace infinite_series_sum_eq_3_over_4_l29_29723

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29723


namespace third_smallest_four_digit_in_pascals_triangle_l29_29328

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29328


namespace check_num_valid_pairs_l29_29498

noncomputable def num_valid_pairs : ℕ :=
  let n := 125
  let pairs := (n / 4).nat_ceil
  2 * (pairs * pairs - pairs)

theorem check_num_valid_pairs :
  num_valid_pairs = 992 :=
by
  sorry

end check_num_valid_pairs_l29_29498


namespace sum_series_equals_three_fourths_l29_29555

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29555


namespace prod_sum_rel_prime_l29_29235

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end prod_sum_rel_prime_l29_29235


namespace third_smallest_four_digit_in_pascals_triangle_l29_29362

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29362


namespace minimum_distance_on_line_l29_29093

-- Define the line as a predicate
def on_line (P : ℝ × ℝ) : Prop := P.1 - P.2 = 1

-- Define the expression to be minimized
def distance_squared (P : ℝ × ℝ) : ℝ := (P.1 - 2)^2 + (P.2 - 2)^2

theorem minimum_distance_on_line :
  ∃ P : ℝ × ℝ, on_line P ∧ distance_squared P = 1 / 2 :=
sorry

end minimum_distance_on_line_l29_29093


namespace find_x_l29_29127

variable (x : ℝ) (h_pos : 0 < x) (h_eq : (sqrt (9 * x) * sqrt (12 * x) * sqrt (4 * x) * sqrt (18 * x)) = 36)

theorem find_x :
  x = sqrt (9 / 22) :=
  sorry

end find_x_l29_29127


namespace third_smallest_four_digit_in_pascals_triangle_l29_29310

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29310


namespace infinite_series_sum_eq_3_over_4_l29_29722

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29722


namespace evaluate_series_sum_l29_29601

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29601


namespace range_of_a_l29_29114

open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7 * x - 18 < 0}

theorem range_of_a (a : ℝ) : A a ⊆ B → (-2 : ℝ) ≤ a ∧ a ≤ 9 :=
by sorry

end range_of_a_l29_29114


namespace series_sum_correct_l29_29764

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29764


namespace series_sum_correct_l29_29770

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29770


namespace third_smallest_four_digit_in_pascals_triangle_l29_29353

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29353


namespace find_num_round_balloons_l29_29938

variable (R : ℕ) -- Number of bags of round balloons that Janeth bought
variable (RoundBalloonsPerBag : ℕ := 20)
variable (LongBalloonsPerBag : ℕ := 30)
variable (BagsLongBalloons : ℕ := 4)
variable (BurstRoundBalloons : ℕ := 5)
variable (BalloonsLeft : ℕ := 215)

def total_long_balloons : ℕ := BagsLongBalloons * LongBalloonsPerBag
def total_balloons : ℕ := R * RoundBalloonsPerBag + total_long_balloons - BurstRoundBalloons

theorem find_num_round_balloons :
  BalloonsLeft = total_balloons → R = 5 := by
  sorry

end find_num_round_balloons_l29_29938


namespace sum_series_div_3_powers_l29_29790

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29790


namespace arcsin_sin_solution_l29_29215

theorem arcsin_sin_solution (x : ℝ) (h : - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2) :
  arcsin (sin x) = x / 3 ↔ x = -3 * Real.pi ∨ x = - Real.pi ∨ x = 0 ∨ x = Real.pi ∨ x = 3 * Real.pi :=
sorry

end arcsin_sin_solution_l29_29215


namespace series_converges_to_three_fourths_l29_29799

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29799


namespace sum_series_equals_three_fourths_l29_29542

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29542


namespace minimum_percentage_drink_both_coffee_tea_l29_29976

variables {α : Type*} (C T : set α) [fintype α]
variables (hC : C.card = 90) (hT : T.card = 80)

theorem minimum_percentage_drink_both_coffee_tea : 
  (C ∪ T).card = 100 → (C ∩ T).card = 70 :=
by
  sorry

end minimum_percentage_drink_both_coffee_tea_l29_29976


namespace number_of_trees_l29_29472

theorem number_of_trees (l d : ℕ) (h_l : l = 441) (h_d : d = 21) : (l / d) + 1 = 22 :=
by
  sorry

end number_of_trees_l29_29472


namespace negation_of_universal_proposition_l29_29233

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by sorry

end negation_of_universal_proposition_l29_29233


namespace third_smallest_four_digit_Pascal_triangle_l29_29284

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29284


namespace largest_digit_divisible_by_6_l29_29265

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + N = 6 * d) ∧ (∀ M : ℕ, M ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + M = 6 * d) → M ≤ N) :=
sorry

end largest_digit_divisible_by_6_l29_29265


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29332

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29332


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29409

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29409


namespace volume_of_box_is_3888_l29_29170

noncomputable def box_volume_proof : Prop :=
  ∃ (H L W : ℝ),
    H = 12 ∧
    L = 3 * H ∧
    W = L / 4 ∧
    real.sqrt (H^2 + L^2 + W^2) = 60 ∧
    H * L * W = 3888

theorem volume_of_box_is_3888 : box_volume_proof :=
by 
  sorry

end volume_of_box_is_3888_l29_29170


namespace Vasya_always_wins_l29_29200

-- Define the initial conditions of the board and the game rules
def board_size : ℕ := 8

-- Define the initial positions of the pieces
def initial_position_1 : (ℕ, ℕ) := (1, 1)
def initial_position_2 : (ℕ, ℕ) := (3, 3)

-- Define the target position for a win
def winning_position : (ℕ, ℕ) := (8, 8)

-- Define the movement rules
-- A piece can move vertically upwards or horizontally to the right by any number of cells
def valid_move (pos1 pos2 : (ℕ, ℕ)) : Prop :=
(pos1.1 = pos2.1 ∧ pos1.2 < pos2.2) ∨ (pos1.1 < pos2.1 ∧ pos1.2 = pos2.2)

-- Define the main theorem
theorem Vasya_always_wins :
  ∃ strategy_Vasya : ((ℕ, ℕ) × (ℕ, ℕ)) → (ℕ, ℕ) × (ℕ, ℕ), 
  ∀ (S : ((ℕ, ℕ) × (ℕ, ℕ))), 
  (S = (initial_position_1, initial_position_2) → 
   ((S.1 = winning_position ∨ S.2 = winning_position) → 
    strategy_Vasya S = winning_position)) :=
sorry

end Vasya_always_wins_l29_29200


namespace total_students_l29_29253

theorem total_students (S : ℕ) (h1 : S = 7 * 4) : S = 28 := 
by 
  have h2: 5 * 4 = 20 := by norm_num
  have h3: 7 * 4 = 28 := by norm_num
  rw [h3] 
  rfl

end total_students_l29_29253


namespace parallel_lines_find_m_l29_29871

theorem parallel_lines_find_m :
  (∀ (m : ℝ), ∀ (x y : ℝ), (2 * x + (m + 1) * y + 4 = 0) ∧ (m * x + 3 * y - 2 = 0) → (m = -3 ∨ m = 2)) := 
sorry

end parallel_lines_find_m_l29_29871


namespace limit_of_sequence_l29_29489

noncomputable def problem_statement : Prop :=
  (real.sin ∘ (λ n : ℕ, real.sqrt (n^2 + 1))) ∘ (λ n : ℕ, real.arctan ((n : ℝ) / (n^2 + 1))) ⟶ 0 as n ⟶ ∞

theorem limit_of_sequence :
  problem_statement :=
sorry

end limit_of_sequence_l29_29489


namespace correct_propositions_l29_29073

variables {α β : Type} [plane α] [plane β] [line l] [line m]

-- Definition of the given conditions
def line_perp_plane (l : line) (α : plane) : Prop := ⊥ ∙ α
def line_in_plane (m : line) (β : plane) : Prop := m ∈ β

-- Definitions for the propositions
def prop1 (α β : plane) (l m : line) : Prop :=
  (α ∥ β) → line_perp_plane l α → line_in_plane m β → l ⊥ m

def prop3 (α β : plane) (l m : line) : Prop :=
  (l ∥ m) → line_perp_plane l α → line_in_plane m β → α ⊥ β

-- The main theorem stating that propositions ① and ③ are correct
theorem correct_propositions :
  prop1 α β l m ∧ prop3 α β l m :=
by sorry

end correct_propositions_l29_29073


namespace third_smallest_four_digit_in_pascals_triangle_l29_29312

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29312


namespace estimate_students_l29_29918

open ProbTheory

noncomputable def number_of_students_above_110 {n : ℕ} (X : PMF ℝ) : ℕ :=
  let μ := 100
  let σ := 10
  let number_of_students := 50
  if X.pmf 90 |>.val = 0.3 then
    (number_of_students * (1 - (X.pmf (μ-σ) |>.val + X.pmf μ |>.val)))
  else
    0

theorem estimate_students :
  let X := PMF.uniform_of_finset (set.Icc 90 110) 50
  (X.pmf 90.0 |>.val, X.pmf 100.0 |>.val = (0.3, 0.3)) → 
  number_of_students_above_110 X = 10 :=
by
  sorry

end estimate_students_l29_29918


namespace sequence_general_formula_l29_29857

noncomputable def sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-2)

theorem sequence_general_formula {a : ℕ → ℝ} {S : ℕ → ℝ} (hpos : ∀ n, a n > 0)
  (hSn : ∀ n, 2 * a n = S n + 0.5) : ∀ n, a n = sequence_formula a S n :=
by 
  sorry

end sequence_general_formula_l29_29857


namespace betty_height_correct_l29_29496

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l29_29496


namespace max_remainder_l29_29119

theorem max_remainder (y : ℕ) : 
  ∃ q r : ℕ, y = 11 * q + r ∧ r < 11 ∧ r = 10 := by sorry

end max_remainder_l29_29119


namespace sum_series_equals_three_fourths_l29_29554

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29554


namespace series_converges_to_three_fourths_l29_29812

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29812


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29653

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29653


namespace evaluate_f_at_neg10_l29_29100

def f : ℝ → ℝ :=
  λ x, if x > 0 then 2^x else x + 12

theorem evaluate_f_at_neg10 : f (-10) = 2 := 
by
  unfold f
  rw if_neg
  { norm_num }
  linarith

end evaluate_f_at_neg10_l29_29100


namespace function_range_l29_29865

def is_natural_positive (a : ℕ) : Prop := a > 0

theorem function_range (a : ℕ) (h1 : is_natural_positive a) (h2 : a^2 - a < 2) :
    set.range (λ x : ℝ, x + (2 * a) / x) = {y : ℝ | y ≤ -2 * Real.sqrt 2 ∨ y ≥ 2 * Real.sqrt 2 } :=
by
  sorry

end function_range_l29_29865


namespace length_of_side_l29_29484

variable (s : ℝ) -- length of each side of the silver cube
variable (cost : ℝ) -- cost of each ounce of silver
variable (markup : ℝ) -- markup percentage as a ratio
variable (total_price : ℝ) -- total price Bob sold the cube
variable (weight_per_cubic_inch : ℝ) -- weight of silver per cubic inch

-- Given conditions
def conditions : Prop :=
  cost = 25 ∧
  markup = 1.10 ∧
  total_price = 4455 ∧
  weight_per_cubic_inch = 6 ∧
  (s^3 * cost * weight_per_cubic_inch * markup = total_price)

-- Prove the length of each side of the silver cube is 3 inches
theorem length_of_side (h : conditions) : s = 3 :=
by
  sorry

end length_of_side_l29_29484


namespace infinite_series_sum_eq_3_div_4_l29_29610

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29610


namespace sum_of_squares_constant_l29_29075

-- Given definitions from the conditions
structure RegularPolygon (n : ℕ) :=
(center : ℂ) -- Complex plane representation for simplicity
(vertices : Fin n → ℂ)
(radius : ℝ) -- Radius of the circumcircle

def perpendicular_distance (vertex : ℂ) (line_slope : ℝ) (line_intercept : ℝ) : ℝ :=
  -- Function to compute perpendicular distance from a vertex to the line l
  let d := line_slope * vertex.re - vertex.im + line_intercept
  d.abs / Real.sqrt (line_slope^2 + 1)

-- The statement of the theorem
theorem sum_of_squares_constant (n : ℕ) (line_slope line_intercept : ℝ) [RegularPolygon n] :
  ∑ i, (perpendicular_distance (RegularPolygon.vertices i) line_slope line_intercept)^2 = 
  ∑ i, (perpendicular_distance (RegularPolygon.vertices i) (1:ℝ) (0:ℝ))^2 :=
by
  sorry

end sum_of_squares_constant_l29_29075


namespace projection_norm_ratio_l29_29180

variables (V : Type*) [InnerProductSpace ℝ V]
variables (v w : V)
noncomputable theory

def projection (u v : V) : V := (inner u v / inner v v) • v

axiom cond_p : projection v w = p
axiom cond_q : projection p v = q
axiom cond_norm : ‖p‖ / ‖w‖ = 3 / 4

theorem projection_norm_ratio :
  ‖projection (projection v w) v‖ / ‖w‖ = 9 / 16 := sorry

end projection_norm_ratio_l29_29180


namespace value_of_xyz_l29_29959

theorem value_of_xyz (x y z : ℂ) 
  (h1 : x * y + 5 * y = -20)
  (h2 : y * z + 5 * z = -20)
  (h3 : z * x + 5 * x = -20) :
  x * y * z = 80 := 
by
  sorry

end value_of_xyz_l29_29959


namespace expression_simplification_l29_29901

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := 
sorry

end expression_simplification_l29_29901


namespace circle_radius_l29_29515

theorem circle_radius 
  (x y : ℝ)
  (h : x^2 + y^2 + 36 = 6 * x + 24 * y) : 
  ∃ (r : ℝ), r = Real.sqrt 117 :=
by 
  sorry

end circle_radius_l29_29515


namespace muffin_to_banana_ratio_l29_29944

variables (m b : ℝ) -- initial cost of a muffin and a banana

-- John's total cost for muffins and bananas
def johns_cost (m b : ℝ) : ℝ :=
  3 * m + 4 * b

-- Martha's total cost for muffins and bananas based on increased prices
def marthas_cost_increased (m b : ℝ) : ℝ :=
  5 * (1.2 * m) + 12 * (1.5 * b)

-- John's total cost times three
def marthas_cost_original_times_three (m b : ℝ) : ℝ :=
  3 * (johns_cost m b)

-- The theorem to prove
theorem muffin_to_banana_ratio
  (h3m4b_eq : johns_cost m b * 3 = marthas_cost_increased m b)
  (hm_eq_2b : m = 2 * b) :
  (1.2 * m) / (1.5 * b) = 4 / 5 := by
  sorry

end muffin_to_banana_ratio_l29_29944


namespace cube_volume_l29_29228

theorem cube_volume (s : ℝ) (d : ℝ) (h : d = s * real.sqrt 3) (h2 : d = 10) :
  s^3 = 1000 :=
by
  sorry

end cube_volume_l29_29228


namespace last_page_stamps_l29_29497

def charlie_albums : ℕ := 10
def pages_per_album : ℕ := 30
def stamps_per_page : ℕ := 8
def stamps_per_new_page : ℕ := 12
def full_new_albums : ℕ := 6

theorem last_page_stamps :
  let total_stamps := charlie_albums * pages_per_album * stamps_per_page,
      full_pages := total_stamps / stamps_per_new_page,
      full_pages_in_full_albums := full_new_albums * pages_per_album,
      remaining_pages := full_pages - full_pages_in_full_albums,
      remaining_stamps := total_stamps - (full_pages_in_full_albums * stamps_per_new_page + (remaining_pages - 1) * stamps_per_new_page)
  in remaining_stamps = 12 :=
by
  let total_stamps := charlie_albums * pages_per_album * stamps_per_page in
  let full_pages := total_stamps / stamps_per_new_page in
  let full_pages_in_full_albums := full_new_albums * pages_per_album in
  let remaining_pages := full_pages - full_pages_in_full_albums in
  let remaining_stamps := total_stamps - (full_pages_in_full_albums * stamps_per_new_page + (remaining_pages - 1) * stamps_per_new_page) in
  sorry

end last_page_stamps_l29_29497


namespace infinite_series_sum_value_l29_29732

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29732


namespace sum_geometric_series_l29_29682

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29682


namespace monotonic_if_and_only_if_extreme_point_inequality_l29_29106

noncomputable def f (x a : ℝ) : ℝ := x^2 - 1 + a * Real.log (1 - x)

def is_monotonic (a : ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x a ≤ f y a

theorem monotonic_if_and_only_if (a : ℝ) : 
  is_monotonic a ↔ a ≥ 0.5 :=
sorry

theorem extreme_point_inequality (a : ℝ) (x1 x2 : ℝ) (hₐ : 0 < a ∧ a < 0.5) 
  (hx : x1 < x2) (hx₁₂ : f x1 a = f x2 a) : 
  f x1 a / x2 > f x2 a / x1 :=
sorry

end monotonic_if_and_only_if_extreme_point_inequality_l29_29106


namespace correct_statements_count_l29_29957

def statement1 (a b : ℤ) (H : (5 * a + 3 * b) % 2 = 0) : Prop :=
  (7 * a - 9 * b) % 2 = 0

def statement2 (a b : ℤ) (H : (a^2 + b^2) % 3 = 0) : Prop :=
  a % 3 = 0 ∧ b % 3 = 0

def statement3 (a b : ℤ) (H : Nat.Prime (a + b)) : Prop :=
  ¬ Nat.Prime (a - b)

def statement4 (a b : ℤ) (H : (a^3 - b^3) % 4 = 0) : Prop :=
  (a - b) % 4 = 0

theorem correct_statements_count (a b : ℤ) :
  2 = (if statement1 a b (sorry) then 1 else 0) +
      (if statement2 a b (sorry) then 1 else 0) +
      (if statement3 a b (sorry) then 1 else 0) +
      (if statement4 a b (sorry) then 1 else 0) :=
sorry

end correct_statements_count_l29_29957


namespace find_A_find_a_l29_29161

variable {A B C a b c : ℝ}
variable [Fact (0 < A)] [Fact (A < 3 * π / 2)]

axiom cos_C_eq : 2 * a * Real.cos C - c = 2 * b
axiom A_val : A = 2 * π / 3

theorem find_A : A = 2 * π / 3 :=
by
  exact A_val

noncomputable def c_val := (2:ℝ).sqrt
noncomputable def BD_val := (3:ℝ).sqrt

axiom BD_len : BD_val = (3:ℝ).sqrt

theorem find_a : a = (6:ℝ).sqrt :=
by
  -- Solution for part 2
  have c_val_eq : c = c_val := rfl
  have BD_eq : BD_val = (3:ℝ).sqrt := BD_len
  exact sorry -- Proof to be provided

end find_A_find_a_l29_29161


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29410

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29410


namespace sum_geometric_series_l29_29692

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29692


namespace infinite_series_sum_eq_l29_29704

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29704


namespace infinite_series_sum_eq_3_over_4_l29_29725

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29725


namespace divisibility_of_factorials_l29_29185

-- Let's state the theorem clearly in Lean
theorem divisibility_of_factorials (p : ℕ) (hp : Nat.Prime p) :
  let k := (p - 1) * p / 2 in
  k ∣ (Nat.factorial (p - 1) - (p - 1)) :=
by
  have k_def : k = (p - 1) * p / 2 := rfl
  sorry

end divisibility_of_factorials_l29_29185


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29408

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29408


namespace sum_geometric_series_l29_29679

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29679


namespace sum_series_eq_3_div_4_l29_29559

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29559


namespace series_converges_to_three_fourths_l29_29803

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29803


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29334

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29334


namespace pascal_third_smallest_four_digit_number_l29_29389

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29389


namespace union_of_A_and_B_intersection_of_complement_A_and_B_l29_29113

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2 * x - 1 ∧ 2 * x - 1 < 19}

-- Define the universal set here, which encompass all real numbers
def universal_set : Set ℝ := {x | true}

-- Define the complement of A with respect to the real numbers
def C_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Prove that A ∪ B is {x | 2 < x < 10}
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

-- Prove that (C_R A) ∩ B is {x | 2 < x < 3 ∨ 7 < x < 10}
theorem intersection_of_complement_A_and_B : (C_R A) ∪ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by
  sorry

end union_of_A_and_B_intersection_of_complement_A_and_B_l29_29113


namespace PQC_is_equilateral_l29_29176

open EuclideanGeometry

variables {A B C D P Q : Point}
variables (ABCD : parallelogram A B C D)
variables (ADP : equilateral_triangle A D P)
variables (ABQ : equilateral_triangle A B Q)

theorem PQC_is_equilateral (ABCD : parallelogram A B C D) 
  (ADP : equilateral_triangle A D P) 
  (ABQ : equilateral_triangle A B Q) : equilateral_triangle P Q C :=
sorry

end PQC_is_equilateral_l29_29176


namespace purely_imaginary_expression_implies_z_abs_eq_one_l29_29868

noncomputable def z_conj (z : ℂ) : ℂ := conj z

theorem purely_imaginary_expression_implies_z_abs_eq_one (z : ℂ) 
  (h : (1 - complex.i) / (z * z_conj z + complex.i) = complex.i * (1 - complex.i) / abs (1 - complex.i)) :
  abs z = 1 :=
by sorry

end purely_imaginary_expression_implies_z_abs_eq_one_l29_29868


namespace johns_gym_hours_per_week_l29_29947

-- Define the conditions as constants
constant gym_visits_per_week : ℕ := 3
constant weightlifting_per_visit : ℝ := 1
constant cardio_fraction_of_weightlifting : ℝ := 1 / 3

-- Theorem statement: Prove John spends 4 hours at gym per week
theorem johns_gym_hours_per_week : gym_visits_per_week * (weightlifting_per_visit + weightlifting_per_visit * cardio_fraction_of_weightlifting) = 4 :=
by
  sorry

end johns_gym_hours_per_week_l29_29947


namespace axis_of_symmetry_parabola_l29_29826

theorem axis_of_symmetry_parabola (x y : ℝ) : 
  (∃ k : ℝ, (y^2 = -8 * k) → (y^2 = -8 * x) → x = -1) :=
by
  sorry

end axis_of_symmetry_parabola_l29_29826


namespace simplify_and_evaluate_expression_l29_29214

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_and_evaluate_expression_l29_29214


namespace third_smallest_four_digit_in_pascals_triangle_l29_29314

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29314


namespace sum_of_first_six_terms_geometric_sequence_l29_29835

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l29_29835


namespace sum_series_eq_3_over_4_l29_29754

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29754


namespace series_sum_l29_29673

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29673


namespace series_sum_l29_29671

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29671


namespace sum_series_eq_3_over_4_l29_29749

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29749


namespace sum_of_series_eq_three_fourths_l29_29581

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29581


namespace sum_of_series_eq_three_fourths_l29_29585

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29585


namespace sum_series_eq_3_div_4_l29_29562

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29562


namespace sum_geometric_series_l29_29689

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29689


namespace correct_calculation_l29_29416

theorem correct_calculation (a b : ℝ) : 
  (a^2 + a^3 ≠ a^5) 
  ∧ (a^2 * a^3 = a^5)
  ∧ ((ab^2)^3 ≠ ab^6)
  ∧ (a^10 / a^2 ≠ a^5) :=
by {
  sorry
}

end correct_calculation_l29_29416


namespace infinite_series_sum_eq_3_over_4_l29_29714

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29714


namespace find_omega_l29_29102

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  √2 * sin (ω * x) * cos (ω * x) + √2 * (cos (ω * x))^2 - √2 / 2

theorem find_omega (ω : ℝ) (x : ℝ) (hx : x = π / 4) (hω : ω > 0) 
  (h_symmetry : ∀ x, f ω x = f ω ((π/2 - x) + k * π) ) :
  ω = 1 / 2 :=
sorry

end find_omega_l29_29102


namespace evaluate_series_sum_l29_29602

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29602


namespace third_smallest_four_digit_in_pascals_triangle_l29_29327

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29327


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29405

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29405


namespace find_projection_matrix_values_l29_29262

noncomputable def projection_matrix (a c : ℚ) : matrix (fin 2) (fin 2) ℚ :=
  ![![a, 7/17], ![c, 10/17]]

theorem find_projection_matrix_values :
  ∃ a c : ℚ, projection_matrix a c * projection_matrix a c = projection_matrix a c ∧
    a = 9/17 ∧ c = 10/17 :=
by
  sorry

end find_projection_matrix_values_l29_29262


namespace basketball_teams_l29_29989

theorem basketball_teams (boys girls : ℕ) (total_players : ℕ) (team_size : ℕ) (ways : ℕ) :
  boys = 7 → girls = 3 → total_players = 10 → team_size = 5 → ways = 105 → 
  ∃ (girls_in_team1 girls_in_team2 : ℕ), 
    girls_in_team1 + girls_in_team2 = 3 ∧ 
    1 ≤ girls_in_team1 ∧ 
    1 ≤ girls_in_team2 ∧ 
    girls_in_team1 ≠ 0 ∧ 
    girls_in_team2 ≠ 0 ∧ 
    ways = 105 :=
by 
  sorry

end basketball_teams_l29_29989


namespace third_smallest_four_digit_in_pascals_triangle_l29_29384

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29384


namespace third_smallest_four_digit_in_pascals_triangle_l29_29381

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29381


namespace complex_norm_wz_l29_29067

open Complex

theorem complex_norm_wz (w z : ℂ) (h₁ : ‖w + z‖ = 2) (h₂ : ‖w^2 + z^2‖ = 8) : 
  ‖w^4 + z^4‖ = 56 := 
  sorry

end complex_norm_wz_l29_29067


namespace proof_problem_l29_29080

-- Definition of proposition p
def p : Prop := ∀ a : ℝ, a > 0 → a + (1 / a) ≥ 2

-- Definition of proposition q
def q : Prop := ∃ x0 : ℝ, sin x0 + cos x0 = sqrt 3

-- The proof statement
theorem proof_problem : ¬ q :=
by
  sorry

end proof_problem_l29_29080


namespace measure_of_U_l29_29151

theorem measure_of_U (F I U G E : ℝ) (R : ℝ) :
  (∠ G + ∠ E = 180) →
  (R = 2 * U) →
  (F = U) →
  (I = U) →
  (F + I + U + R + G + E = 720) →
  U = 108 :=
by sorry

end measure_of_U_l29_29151


namespace evaluate_series_sum_l29_29596

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29596


namespace disproof_of_Alitta_l29_29471

-- Definition: A prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition: A number is odd
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- The value is a specific set of odd primes including 11
def contains (p : ℕ) : Prop :=
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11

-- Main statement: There exists an odd prime p in the given options such that p^2 - 2 is not a prime
theorem disproof_of_Alitta :
  ∃ p : ℕ, contains p ∧ is_prime p ∧ is_odd p ∧ ¬ is_prime (p^2 - 2) :=
by
  sorry

end disproof_of_Alitta_l29_29471


namespace series_sum_l29_29665

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29665


namespace third_smallest_four_digit_in_pascals_triangle_l29_29307

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29307


namespace number_of_chapters_l29_29987

theorem number_of_chapters (n : ℕ) : 
  (∑ i in finset.range n, 13 + 3 * i) = 95 → n = 5 :=
by
  intro h
  -- Adding sorry to skip proof
  sorry

end number_of_chapters_l29_29987


namespace median_is_71_l29_29927

-- Definitions used in the problem
def count_occurrences (n : ℕ) : ℕ := n
def cumulative_count (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Main statement
theorem median_is_71 : 
  let lst := List.bind (List.range 100) (λ n, List.repeat (n + 1) (n + 1)) in
  let sorted_lst := lst in
  List.median sorted_lst = 71 :=
by
  sorry

end median_is_71_l29_29927


namespace probability_favorable_arrangement_l29_29004

open Finset

theorem probability_favorable_arrangement :
  let n := 8 in
  let k := 4 in
  let all_arrangements := (2: ℕ) ^ n in
  let favorable_arrangements := (n.choose k) in
  (favorable_arrangements / all_arrangements : ℚ) = 35 / 128 :=
by
  sorry

end probability_favorable_arrangement_l29_29004


namespace highest_probability_of_12_correct_l29_29260

-- Problem conditions
def choices (n : ℕ) : ℕ := n

-- Probability of getting nth question correct
def prob_correct (n : ℕ) : ℝ := 1 / (choices n)

-- Product of probabilities of getting exactly 12 out of 20 questions correct
noncomputable def highest_probability : ℝ := 
  ∏ (i : ℕ) in finset.range 12, prob_correct (i + 1)

-- Proof statement
theorem highest_probability_of_12_correct : highest_probability = 1 / 12! :=
  sorry

end highest_probability_of_12_correct_l29_29260


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29278

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29278


namespace sum_squares_of_distances_l29_29838

theorem sum_squares_of_distances (r R : ℝ) :
  let D := 3 * R^2 - 4 * R * r - r^2 in
  exists (T : Triangle),
    let incircle_radius := T.incircle_radius
    let circumcircle_radius := T.circumcircle_radius
    let distances := T.squared_distances_from_tangency_points_to_circumcenter in
    r = incircle_radius ∧ R = circumcircle_radius ∧ distances.sum = D :=
begin
  sorry
end

end sum_squares_of_distances_l29_29838


namespace sum_geometric_series_l29_29681

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29681


namespace third_smallest_four_digit_Pascal_triangle_l29_29286

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29286


namespace trapezium_area_l29_29036

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l29_29036


namespace infinite_series_sum_value_l29_29741

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29741


namespace ratio_of_diamonds_to_spades_l29_29452

-- Given conditions
variable (total_cards : Nat := 13)
variable (black_cards : Nat := 7)
variable (red_cards : Nat := 6)
variable (clubs : Nat := 6)
variable (diamonds : Nat)
variable (spades : Nat)
variable (hearts : Nat := 2 * diamonds)
variable (cards_distribution : clubs + diamonds + hearts + spades = total_cards)
variable (black_distribution : clubs + spades = black_cards)

-- Define the proof theorem
theorem ratio_of_diamonds_to_spades : (diamonds / spades : ℝ) = 2 :=
 by
  -- temporarily we insert sorry to skip the proof
  sorry

end ratio_of_diamonds_to_spades_l29_29452


namespace fraction_to_decimal_l29_29024

theorem fraction_to_decimal :
  (3 / 8 : ℝ) = 0.375 :=
sorry

end fraction_to_decimal_l29_29024


namespace point_in_second_quadrant_l29_29925

-- Definitions for the coordinates of the points
def A : ℤ × ℤ := (3, 2)
def B : ℤ × ℤ := (-3, -2)
def C : ℤ × ℤ := (3, -2)
def D : ℤ × ℤ := (-3, 2)

-- Definition for the second quadrant condition
def isSecondQuadrant (p : ℤ × ℤ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem we need to prove
theorem point_in_second_quadrant : isSecondQuadrant D :=
by
  sorry

end point_in_second_quadrant_l29_29925


namespace infinite_series_sum_eq_3_over_4_l29_29727

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29727


namespace trapezium_area_l29_29034

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l29_29034


namespace limit_sequence_l29_29486

open Real

theorem limit_sequence :
  tendsto (λ n : ℕ, sin (sqrt (↑n^2 + 1)) * arctan (↑n / (↑n^2 + 1))) at_top (𝓝 0) :=
begin
  sorry
end

end limit_sequence_l29_29486


namespace infinite_series_sum_eq_3_div_4_l29_29619

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29619


namespace total_spent_on_birthday_presents_l29_29172

noncomputable def leonards_total_before_discount :=
  (3 * 35.50) + (2 * 120.75) + 44.25

noncomputable def leonards_total_after_discount :=
  leonards_total_before_discount - (0.10 * leonards_total_before_discount)

noncomputable def michaels_total_before_discount :=
  89.50 + (3 * 54.50) + 24.75

noncomputable def michaels_total_after_discount :=
  michaels_total_before_discount - (0.15 * michaels_total_before_discount)

noncomputable def emilys_total_before_tax :=
  (2 * 69.25) + (4 * 14.80)

noncomputable def emilys_total_after_tax :=
  emilys_total_before_tax + (0.08 * emilys_total_before_tax)

noncomputable def total_amount_spent :=
  leonards_total_after_discount + michaels_total_after_discount + emilys_total_after_tax

theorem total_spent_on_birthday_presents :
  total_amount_spent = 802.64 :=
by
  sorry

end total_spent_on_birthday_presents_l29_29172


namespace third_smallest_four_digit_in_pascals_triangle_l29_29305

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29305


namespace sum_f_2019_l29_29853

-- Define function and its properties
def f (n : ℕ) : ℤ := sorry  -- Function definition and properties must be provided

-- Define the conditions as hypotheses
theorem sum_f_2019 :
  (∀ n : ℕ, |f n| = n) ∧
  (∀ n : ℕ, 0 ≤ ∑ k in finset.range n, f k + 1 < 2 * n) →
  ∑ k in finset.range 2020, f k = 630 :=
sorry

end sum_f_2019_l29_29853


namespace betty_height_correct_l29_29495

-- Definitions for the conditions
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height_inches : ℕ := carter_height - 12
def betty_height_feet : ℕ := betty_height_inches / 12

-- Theorem that we need to prove
theorem betty_height_correct : betty_height_feet = 3 :=
by
  sorry

end betty_height_correct_l29_29495


namespace find_line_equation_l29_29913

theorem find_line_equation (k : ℝ) (x y : ℝ) :
  (∀ k, (∃ x y, y = k * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0) ↔ x - y + 1 = 0) :=
by
  sorry

end find_line_equation_l29_29913


namespace max_value_of_f_l29_29046

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 2 * real.sin x + real.cos x) / (real.sin x + real.sqrt (1 - real.sin x))

theorem max_value_of_f : ∀ x : ℝ, (0 ≤ x ∧ x ≤ real.pi) → f x ≤ real.sqrt 2 ∧ ∃ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ real.pi ∧ f x₁ = real.sqrt 2 :=
by
  sorry

end max_value_of_f_l29_29046


namespace simplify_and_evaluate_expr_l29_29212

-- Define x
def x : ℝ := Real.sqrt 2 - 1

-- Define the expression
def expr : ℝ := (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x))

-- State the theorem which asserts the equality
theorem simplify_and_evaluate_expr : expr = -Real.sqrt 2 / 2 := 
by 
  sorry

end simplify_and_evaluate_expr_l29_29212


namespace infinite_series_sum_value_l29_29730

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29730


namespace series_sum_correct_l29_29762

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29762


namespace line_passes_through_fixed_point_shortest_chord_length_l29_29068

-- Define the circle and line equations
def circle (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 25
def line (m : ℝ) (x y : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Problem 1: Prove that the line always passes through a fixed point (3,1)
theorem line_passes_through_fixed_point (m : ℝ) : line m 3 1 := 
by sorry

-- Define when the chord cut by the line from the circle is shortest
def shortest_chord_condition (m : ℝ) (x y : ℝ) := m = -(3/4) ∧ line m x y = 2 * x - y - 5

-- Problem 2: Prove that for m = -(3/4), the shortest chord length is 4sqrt5
theorem shortest_chord_length (x y : ℝ) : 
  shortest_chord_condition (-(3/4)) x y → chord_length circle line (-(3/4)) = 4 * sqrt 5 := 
by sorry

-- Define the function to compute chord length (problem-specific definition)
noncomputable def chord_length (circle eq₁ eq₂ : ℝ → ℝ → Prop) (m : ℝ) : ℝ :=
  let d := abs ((2 - 2) - 5) / sqrt 5 in
  2 * sqrt (25 - d^2)

-- Define a term representing the absolute value
noncomputable def abs (a : ℝ) : ℝ := if a >= 0 then a else -a

end line_passes_through_fixed_point_shortest_chord_length_l29_29068


namespace third_smallest_four_digit_in_pascals_triangle_l29_29308

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29308


namespace evaluate_series_sum_l29_29608

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29608


namespace dot_product_abs_value_l29_29179

variables (a b : ℝ^3)

def norm (v : ℝ^3) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)
def cross_product (v1 v2 : ℝ^3) : ℝ^3 := (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)
def dot_product (v1 v2 : ℝ^3) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

axiom norm_a : norm a = 2
axiom norm_b : norm b = 5
axiom norm_cross_product_ab : norm (cross_product a b) = 8

theorem dot_product_abs_value : |dot_product a b| = 6 :=
by
  sorry

end dot_product_abs_value_l29_29179


namespace apartment_units_in_building_l29_29197

theorem apartment_units_in_building : 
  ∃ (x : ℕ), 2 * (2 + 3 * x) = 34 ∧ x = 5 :=
by
  use 5
  split
  sorry -- proof is skipped here by adding sorry

end apartment_units_in_building_l29_29197


namespace find_f_log3_2_l29_29881

noncomputable def logarithmic_function (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : ℝ → ℝ :=
  λ x, Real.log (x + 3) / Real.log a - 1

noncomputable def exponential_function (b : ℝ) : ℝ → ℝ :=
  λ x, 3 ^ x + b

theorem find_f_log3_2 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (A : ℝ × ℝ)
  (hA₁ : A = (-2, -1)) (b : ℝ) (hb : (3 ^ (-2) + b = -1)) :
  exponential_function b (Real.log 2 / Real.log 3) = 8 / 9 :=
by
  sorry

end find_f_log3_2_l29_29881


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29647

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29647


namespace sum_condition_l29_29998

noncomputable def a : ℤ := sorry
noncomputable def b : ℤ := sorry
noncomputable def c : ℤ := sorry

def poly1 (a b : ℤ) : Polynomial ℤ := Polynomial.Coeff x 2 + Polynomial.Coeff x 1 + Polynomial.Coeff x 0
def poly2 (b c : ℤ) : Polynomial ℤ := Polynomial.Coeff x 2 + Polynomial.Coeff x 1 + Polynomial.Coeff x 0

theorem sum_condition :
  let g := gcd (poly1 a b) (poly2 b c) in
  let l := lcm (poly1 a b) (poly2 b c) in
  g = Polynomial.X + 1 ∧ l = Polynomial.X^3 - 3*Polynomial.X^2 - 4*Polynomial.X + 12 → 
  a + b + c = -3 :=
sorry

#print axioms sum_condition -- to ensure no axioms are required to prove this

end sum_condition_l29_29998


namespace max_cardinality_A_l29_29065

theorem max_cardinality_A :
  ∃ A : Finset ℕ, (∀ (x y ∈ A), x ≠ y → (abs (x - y) ≠ 4) ∧ (abs (x - y) ≠ 7)) ∧ 
  A ⊆ Finset.range 2001 ∧ A.card = 910 :=
sorry

end max_cardinality_A_l29_29065


namespace integral_eval_l29_29023

noncomputable def integral_func (x : ℝ) : ℝ := sqrt (1 - x^2) + x + x^3

theorem integral_eval : ∫ x in 0..1, integral_func x = (Real.pi + 3) / 4 :=
by
  sorry

end integral_eval_l29_29023


namespace number_of_valid_digits_l29_29999

theorem number_of_valid_digits : ∃ d_values : Finset ℕ, (∀ d ∈ d_values, d ∈ Finset.range 10 ∧ (2 + 0.05 * d > 2.050)) ∧ d_values.card = 9 :=
by
  sorry

end number_of_valid_digits_l29_29999


namespace solve_system_l29_29217

theorem solve_system :
  ∃ (x y : ℝ), 
    (y + real.sqrt (y - 3 * x) + 3 * x = 12) ∧ 
    (y^2 + y - 3 * x - 9 * x^2 = 144) ∧ 
    ((x = -4/3 ∧ y = 12) ∨ (x = -24 ∧ y = 72)) :=
by
  sorry

end solve_system_l29_29217


namespace value_of_x_squared_inverse_squared_l29_29134

theorem value_of_x_squared_inverse_squared (x : ℝ) (h : 31 = x^6 + (1 / x^6)) : 
  x^2 + (1 / x^2) = Real.cbrt 34 :=
by
  sorry

end value_of_x_squared_inverse_squared_l29_29134


namespace probability_cos_pi_x_ge_one_half_l29_29988

theorem probability_cos_pi_x_ge_one_half : 
  let interval := Set.Icc (-1 : ℝ) (1 : ℝ)
  let condition_interval := Set.Icc (-1/3 : ℝ) (1/3 : ℝ)
  (measure_theory.MeasureTheory.measure (condition_interval ∩ interval) / measure_theory.MeasureTheory.measure interval = 1/3) := by
  sorry

end probability_cos_pi_x_ge_one_half_l29_29988


namespace sum_series_div_3_powers_l29_29784

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29784


namespace arithmetic_sequence_formula_and_sum_l29_29858

theorem arithmetic_sequence_formula_and_sum (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : all_positive : ∀ n, a n > 0)
  (h3 : a 4 = 2 * a 2)
  (h4 : a 1 * 4 * (a 1 + 3 * d) = 16) :
  (∀ n, a n = 2 * n) ∧ ∑ k in (Finset.range 20).map (λ i, 20 + i * 5), 2 * a (20 + i * 5) = 2700 := sorry

end arithmetic_sequence_formula_and_sum_l29_29858


namespace infinite_series_sum_eq_l29_29698

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29698


namespace minimal_beacons_required_l29_29158

-- Define the maze structure and properties
structure Room
structure Link
structure Beacon where
  position : Room
  unique_sound : String

-- Define the conditions
axiom maze : List Room
axiom links : List Link
axiom beacon_positions : List Beacon
axiom distance (r1 r2 : Room) : Nat

-- Condition: Each segment is a link, each circle is a small room
axiom is_corridor (l : Link) : Prop
axiom is_room (r : Room) : Prop
axiom room_link (r1 r2 : Room) : Link

-- Condition: Robot determines the distance to each beacon by the signal decay rate
axiom robot_hears (r : Room) (b : Beacon) : Nat

-- The placement of beacons
noncomputable def a1 : Room := sorry
noncomputable def c3 : Room := sorry
noncomputable def d4 : Room := sorry

-- Verification of distance uniqueness
def distance_vector (r : Room) : List Nat :=
  beacon_positions.map (λ b => robot_hears r b)

-- Prove the minimal number of beacons
theorem minimal_beacons_required (n : Nat) : 
  (∀ r1 r2 : Room, distance_vector r1 = distance_vector r2 → r1 = r2) → 
  (n = 3) :=
begin
  sorry
end

end minimal_beacons_required_l29_29158


namespace third_smallest_four_digit_in_pascal_l29_29350

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29350


namespace sum_of_series_eq_three_fourths_l29_29591

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29591


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29646

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29646


namespace max_cylinder_volume_l29_29828

/-- Define the given parameters of the cone -/
def cone_height : ℝ := 27
def cone_base_radius : ℝ := 9

/-- Definition of the volume function of the inscribed cylinder -/
noncomputable def cylinder_volume (h : ℝ) : ℝ := 
  let r := (cone_height - h) / 3
  π * r^2 * h

/-- Statement of the problem: The maximum volume of the inscribed cylinder -/
theorem max_cylinder_volume :
  ∃ h : ℝ, 0 < h ∧ h < cone_height ∧ cylinder_volume h = 324 * π :=
sorry

end max_cylinder_volume_l29_29828


namespace third_smallest_four_digit_in_pascals_triangle_l29_29309

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29309


namespace series_sum_l29_29662

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29662


namespace betty_height_in_feet_l29_29493

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l29_29493


namespace cumulative_area_exceeds_4750_affordable_area_greater_than_85_l29_29142

def cumulative_affordable_area (n : ℕ) : ℕ :=
  if n = 0 then 2500000 
  else cumulative_affordable_area (n - 1) + 2500000 + 500000 * n

axiom base_housing_area : ℕ := 4000000
axiom housing_growth_rate : ℕ := 8
axiom total_housing_area (n : ℕ) : ℕ := base_housing_area * (1.08 : ℝ) ^ n

theorem cumulative_area_exceeds_4750 (n : ℕ) :
  (∑ i in range (n + 1), (2500000 + 500000 * i)) > 47500000 ↔ n ≥ 9 := by
  sorry

theorem affordable_area_greater_than_85 (n : ℕ) :
  (2500000 + 500000 * (n)) / total_housing_area n > 0.85 iff n = 5 := by
  sorry

end cumulative_area_exceeds_4750_affordable_area_greater_than_85_l29_29142


namespace sum_series_equals_three_fourths_l29_29548

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29548


namespace additional_time_to_fill_l29_29456

variable (capacity : ℝ) -- The full capacity of the pool
variable (PA PB : ℝ) -- Rates of pipes A and B
variable (time_simultaneous : ℝ)

noncomputable def additional_time (PA PB capacity : ℝ) : ℝ :=
  let initial_capacity := (1 / 18) * capacity
  let middle_capacity := (2 / 9) * capacity
  let rateA := PA
  let rateB := PB
  let timeA := 81 -- Pipe A continues alone
  let timeB := 49 -- Pipe B continues alone
  let X := 63 -- Simultaneous time initially

  let filled_A := PA * X
  let filled_B := PB * X

  let total_fill_A := rateA * timeA
  let total_fill_B := rateB * timeB
 
  let partial_fill_together := (1/6) * capacity -- From fraction conversion: [2/9 - 1/18]
 
  let remaining_fill := capacity - (initial_capacity + 2 * partial_fill_together)
  let combined_rate := PA + PB

  let additional_time := 231 -- Time for 11/18 capacity for both pipes

  additional_time

theorem additional_time_to_fill 
  (capacity : ℝ) -- Pool capacity
  (PA PB : ℝ) -- Rates of pipes A and B
  (time_simultaneous : 63 = additional_time PA PB capacity) -- Condition of simultaneous fill time
: additional_time PA PB capacity = 231 :=
by
  sorry

end additional_time_to_fill_l29_29456


namespace imaginary_part_of_expression_l29_29969

def z : ℂ := 1 - complex.I
def expression := z^2 + 2 / z

theorem imaginary_part_of_expression : complex.im expression = -1 := by
  sorry

end imaginary_part_of_expression_l29_29969


namespace distance_between_tangent_circles_l29_29892

theorem distance_between_tangent_circles (R r d : ℕ) (hR : R = 6) (hr : r = 3) (h_tangent : R > r) (h_internal : d = R - r) : d = 3 :=
by
  rw [hR, hr, h_internal]
  norm_num
  exact dec_trivial

end distance_between_tangent_circles_l29_29892


namespace sum_series_div_3_powers_l29_29781

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29781


namespace series_result_l29_29632

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29632


namespace greatest_measure_length_l29_29426

theorem greatest_measure_length :
  let l1 := 18000
  let l2 := 50000
  let l3 := 1520
  ∃ d, d = Int.gcd (Int.gcd l1 l2) l3 ∧ d = 40 :=
by
  sorry

end greatest_measure_length_l29_29426


namespace third_smallest_four_digit_in_pascals_triangle_l29_29298

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29298


namespace problem_l29_29085

variable (a : ℝ)

theorem problem (h : ∃ a : ℝ, ↑((2 * a) / (1 + complex.I) + 1 + complex.I) ∈ ℝ) : a = 1 :=
sorry

end problem_l29_29085


namespace infinite_series_sum_eq_l29_29705

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29705


namespace infinite_series_sum_eq_l29_29708

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29708


namespace infinite_series_sum_eq_3_div_4_l29_29622

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29622


namespace fraction_BC_AD_l29_29982

-- Define the variables and conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (А B C D : Point) (x y : ℝ)
variables (dist_AB dist_AC dist_AD dist_BD dist_CD dist_BC : ℝ)
variables (h1 : dist B D = x)
variables (h2 : dist A B = 3 * dist B D)
variables (h3 : dist C D = y)
variables (h4 : dist A C = 5 * dist C D)
variables (h5 : dist A D = 6 * dist C D)

-- Formulate the problem as a Lean theorem
theorem fraction_BC_AD (h1 : dist B D = x)
  (h2 : dist A B = 3 * x)
  (h3 : dist C D = y)
  (h4 : dist A C = 5 * y)
  (h5 : dist A D = 6 * y)
  : dist B C / dist A D = 1 / 12 := 
sorry

end fraction_BC_AD_l29_29982


namespace relationship_of_abc_l29_29849

noncomputable def m : ℝ := sorry
def a := Real.log m (2 : ℝ)
def b := m^2
def c := 2^m

theorem relationship_of_abc (h0 : 0 < m) (h1 : m < 1) : a < b ∧ b < c := sorry

end relationship_of_abc_l29_29849


namespace min_quadratic_expression_value_l29_29266

def quadratic_expression (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_quadratic_expression_value : 
  ∃ x : ℝ, quadratic_expression x = 2178 :=
sorry

end min_quadratic_expression_value_l29_29266


namespace minimum_p_l29_29132

-- Define the problem constants and conditions
noncomputable def problem_statement :=
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧ 
    (∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (2008 / 2009 < p' / (q' : ℚ)) ∧ (p' / (q' : ℚ) < 2009 / 2010)) → p ≤ p') 

-- The proof
theorem minimum_p (h : problem_statement) :
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧
    p = 4017 :=
sorry

end minimum_p_l29_29132


namespace series_sum_correct_l29_29769

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29769


namespace third_smallest_four_digit_in_pascals_triangle_l29_29360

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29360


namespace participant_with_most_victories_contains_all_others_l29_29150

-- Definitions of the tournament structure and conditions:
structure Tournament :=
  (players : Type)
  (matches : players → players → Prop)
  (no_draws : ∀ (a b : players), matches a b ∨ matches b a)
  (unique_match : ∀ (a b : players), matches a b → ¬matches b a)
  (transitive_beaten : ∀ (a b c : players), matches a b → matches b c → matches a c)

-- Definition of a participant and their list of beaten players
def list_of_beaten {T : Tournament} (a : T.players) : set T.players :=
  { b | T.matches a b }

def list_of_transitive_beaten {T : Tournament} (a : T.players) : set T.players :=
  { b | ∃ c, T.matches a c ∧ (T.matches c b ∨ c = b) }

-- Statement of the problem to be proved in Lean:
theorem participant_with_most_victories_contains_all_others
  (T : Tournament)
  (A : T.players)
  (Hmax : ∀ (b : T.players), b ≠ A → size_of (list_of_transitive_beaten A) ≥ size_of (list_of_transitive_beaten b)) : 
  (∀ b : T.players, b ∈ list_of_transitive_beaten A) :=
sorry

end participant_with_most_victories_contains_all_others_l29_29150


namespace sum_series_div_3_powers_l29_29783

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29783


namespace diagonally_adjacent_pairs_l29_29964

noncomputable theory

/--
Given an odd integer \( n > 1 \),
with an \( n \times n \) chessboard where the center square and four corners are removed,
we can group the remaining squares into pairs such that each pair of squares is diagonally adjacent
if and only if \( n = 3 \) or \( n = 5 \).
-/
theorem diagonally_adjacent_pairs (n : ℕ) (h1 : odd n) (h2 : 1 < n) :
  ∃ pairs : list (ℕ × ℕ) × (ℕ × ℕ),
  (∀ pair ∈ pairs, diagonally_adjacent pair.fst pair.snd) ↔ (n = 3 ∨ n = 5) :=
sorry

end diagonally_adjacent_pairs_l29_29964


namespace polynomial_roots_property_l29_29455

theorem polynomial_roots_property :
  ∃ a : ℝ, 
  (∀ x : ℝ, (a * (x - 1/4) * (x - 1/2) * (x - 2) * (x - 4) = P x)) ∧
  (P(1) = 1) ∧ 
  (P(0) = 8 / 9) ∧ 
  ((1/4) * (1/2) * 2 * 4 = 1)
:= sorry

end polynomial_roots_property_l29_29455


namespace infinite_series_sum_eq_3_over_4_l29_29715

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29715


namespace series_sum_eq_l29_29526

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29526


namespace limit_of_sequence_l29_29488

noncomputable def problem_statement : Prop :=
  (real.sin ∘ (λ n : ℕ, real.sqrt (n^2 + 1))) ∘ (λ n : ℕ, real.arctan ((n : ℝ) / (n^2 + 1))) ⟶ 0 as n ⟶ ∞

theorem limit_of_sequence :
  problem_statement :=
sorry

end limit_of_sequence_l29_29488


namespace sum_of_series_eq_three_fourths_l29_29582

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29582


namespace infinite_series_sum_eq_3_over_4_l29_29721

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29721


namespace infinite_series_sum_eq_3_div_4_l29_29625

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29625


namespace find_m_l29_29084

theorem find_m (x : ℝ) (m : ℝ)
  (h1 : log (10) (sin x) + log (10) (cos x) = -2)
  (h2 : log (10) (sin x + cos x) = (1 / 2) * (log (10) m - 2)) : m = 102 := by
  sorry

end find_m_l29_29084


namespace third_smallest_four_digit_in_pascals_triangle_l29_29379

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29379


namespace gilbert_crickets_l29_29064

theorem gilbert_crickets (W : ℕ)
  (h1 : 4 * 0.8 * W + 8 * 0.2 * W = 72) :
  W = 15 :=
by
  -- Proof steps go here
  sorry

end gilbert_crickets_l29_29064


namespace series_result_l29_29635

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29635


namespace max_value_f_on_interval_l29_29045

def f (x : ℝ) : ℝ := x * (6 - 2 * x)^2

theorem max_value_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) 3, f x = 16 ∧ ∀ y ∈ set.Icc (0 : ℝ) 3, f y ≤ 16 := 
sorry

end max_value_f_on_interval_l29_29045


namespace sum_of_angles_M_and_R_l29_29983

-- Define the points on the circle
variables (E F R G H : Point) -- assumed Point type for geometric points on a circle

-- Define the measures of the arcs FR and RG
variables (arc_FR arc_RG : ℝ)

-- Given conditions
axiom arc_FR_measure : arc_FR = 60
axiom arc_RG_measure : arc_RG = 50

-- Definition of arc_EH (sum of the remaining arcs in the circle not including arc_FR and arc_RG)
noncomputable def arc_EH : ℝ := 360 - (arc_FR + arc_RG)

-- Definition of the angle M, influenced by arcs FG and EH
noncomputable def angle_M : ℝ := (arc_FR + arc_RG - arc_EH) / 2

-- Definition of the angle R, which is half of the arc EH
noncomputable def angle_R : ℝ := arc_EH / 2

-- The statement that proves the sum of the measures of angles M and R
theorem sum_of_angles_M_and_R : angle_M + angle_R = 55 :=
by
  rw [arc_FR_measure, arc_RG_measure]
  have arc_EH_measure : arc_EH = 250 := by sorry
  rw [arc_EH_measure]
  have angle_M_measure : angle_M = -70 := by sorry
  have angle_R_measure : angle_R = 125 := by sorry
  exact calc angle_M + angle_R = -70 + 125 : by rw [angle_M_measure, angle_R_measure]
                        ...            = 55  : by ring

end sum_of_angles_M_and_R_l29_29983


namespace find_b32_l29_29159

theorem find_b32 (b : Fin 32 → ℕ) (z : ℂ) :
  (∀ i, b i > 0) →
  (∃ P : Complex, P = (1 - z)^(b 0) * (1 - z^2)^(b 1) * ... * (1 - z^32)^(b 31) ∧
  ∀ Q Q',
    (Q = P - (P % z^33)) →
    (Q' = 1 - 2*z) →
    Q = Q') →
  b 31 = 2^27 - 2^11 :=
by
  intro h_pos h_poly h_expansion
  sorry

end find_b32_l29_29159


namespace gas_pressure_inversely_proportional_l29_29479

theorem gas_pressure_inversely_proportional :
  ∀ (p v : ℝ), (p * v = 27.2) → (8 * 3.4 = 27.2) → (v = 6.8) → p = 4 :=
by
  intros p v h1 h2 h3
  have h4 : 27.2 = 8 * 3.4 := by sorry
  have h5 : p * 6.8 = 27.2 := by sorry
  exact sorry

end gas_pressure_inversely_proportional_l29_29479


namespace series_sum_l29_29663

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29663


namespace sum_series_div_3_powers_l29_29786

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29786


namespace third_smallest_four_digit_in_pascal_triangle_l29_29373

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29373


namespace area_triangle_roots_cubic_l29_29502

noncomputable def area_of_triangle_with_roots_of_cubic_eqn
  (a b c : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a * b + b * c + c * a = 6)
  (h3 : a * b * c = \frac{61}{20}) : ℝ :=
  let q := \frac{3}{2} in
  sqrt (q * (q - a) * (q - b) * (q - c))

theorem area_triangle_roots_cubic :
  let a b c : ℝ := by sorry
  let h1 : a + b + c = 3 := by sorry
  let h2 : a * b + b * c + c * a = 6 := by sorry
  let h3 : a * b * c = \frac{61}{20} := by sorry
  area_of_triangle_with_roots_of_cubic_eqn a b c h1 h2 h3 = \frac{\sqrt{38625}}{100} :=
  by sorry

end area_triangle_roots_cubic_l29_29502


namespace infinite_series_sum_value_l29_29744

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29744


namespace percentage_seeds_germinated_l29_29055

/-- There were 300 seeds planted in the first plot and 200 seeds planted in the second plot. 
    30% of the seeds in the first plot germinated and 32% of the total seeds germinated.
    Prove that 35% of the seeds in the second plot germinated. -/
theorem percentage_seeds_germinated 
  (s1 s2 : ℕ) (p1 p2 t : ℚ)
  (h1 : s1 = 300) 
  (h2 : s2 = 200) 
  (h3 : p1 = 30) 
  (h4 : t = 32) 
  (h5 : 0.30 * s1 + p2 * s2 = 0.32 * (s1 + s2)) :
  p2 = 35 :=
by 
  -- Proof goes here
  sorry

end percentage_seeds_germinated_l29_29055


namespace third_smallest_four_digit_in_pascals_triangle_l29_29380

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29380


namespace accommodation_ways_l29_29448

-- Definition of the problem
def triple_room_count : ℕ := 1
def double_room_count : ℕ := 2
def adults_count : ℕ := 3
def children_count : ℕ := 2
def total_ways : ℕ := 60

-- Main statement to be proved
theorem accommodation_ways :
  (triple_room_count = 1) →
  (double_room_count = 2) →
  (adults_count = 3) →
  (children_count = 2) →
  -- Children must be accompanied by adults, and not all rooms need to be occupied.
  -- We are to prove that the number of valid ways to assign the rooms is 60
  total_ways = 60 :=
by sorry

end accommodation_ways_l29_29448


namespace triangle_isosceles_at_B_l29_29175
open Function

theorem triangle_isosceles_at_B (A B C D E F : Type)
  [IsTriangle A B C]
  (hAcute : ∀ (a b c : A B C), acute_angle a b c)
  (hD : foot_of_altitude A B C D)
  (hE : reflection_over_AC A C D E)
  (hF : intersection_of_perpendicular_AE_B A E B C F)
  (hCondition : ∀ (a b : A B F C), perpendicular a b)
  : angle C F B = angle F C B :=
by sorry

end triangle_isosceles_at_B_l29_29175


namespace infinite_series_sum_eq_3_div_4_l29_29623

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29623


namespace third_smallest_four_digit_in_pascal_l29_29351

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29351


namespace average_of_a_b_l29_29006

theorem average_of_a_b (a b : ℚ) (h1 : b = 2 * a) (h2 : (4 + 6 + 8 + a + b) / 5 = 17) : (a + b) / 2 = 33.5 := 
by
  sorry

end average_of_a_b_l29_29006


namespace sin_dense_l29_29984

theorem sin_dense (x : ℝ) (h : sin x ≠ 0) : ∃ n : ℕ, |(sin (n * x))| ≥ (sqrt 3 / 2) := sorry

end sin_dense_l29_29984


namespace largest_c_in_range_l29_29038

theorem largest_c_in_range (c : ℝ) (h : ∃ x : ℝ,  2 * x ^ 2 - 4 * x + c = 5) : c ≤ 7 :=
by sorry

end largest_c_in_range_l29_29038


namespace larger_angle_opposite_larger_side_larger_side_opposite_larger_angle_l29_29985

-- Part (a): Prove that the larger angle is opposite the larger side in a triangle.
theorem larger_angle_opposite_larger_side (A B C : ℝ) (a b c : ℝ) 
  (h₀ : a + b > c) (h₁ : a + c > b) (h₂ : b + c > a) 
  (hBC_gt_AC : b > c) :
  angle A B C > angle A C B :=
sorry

-- Part (b): Prove that the larger side is opposite the larger angle in a triangle.
theorem larger_side_opposite_larger_angle (A B C : ℝ) (a b c : ℝ)
  (h₀ : a + b > c) (h₁ : a + c > b) (h₂ : b + c > a) 
  (hAngleA_gt_AngleB : angle A B C > angle A C B) :
  b > c :=
sorry

end larger_angle_opposite_larger_side_larger_side_opposite_larger_angle_l29_29985


namespace dist_inequalities_D_subset_l29_29153

def dist (α : ℝ) (A B : ℝ × ℝ) : ℝ :=
  (|A.1 - B.1|^α + |A.2 - B.2|^α)^(1 / α)

def A := (1, 1) : ℝ × ℝ
def B := (2, 3) : ℝ × ℝ

lemma dist_1_eq : dist 1 A B = 3 :=
  sorry

lemma dist_2_eq : dist 2 A B = Real.sqrt 5 :=
  sorry

theorem dist_inequalities (A B : ℝ × ℝ) : dist 2 A B ≤ dist 1 A B ∧ dist 1 A B ≤ Real.sqrt 2 * dist 2 A B :=
  sorry

def D (α : ℝ) : set (ℝ × ℝ) :=
  { M | dist α M (0, 0) ≤ 1 }

theorem D_subset (α β : ℝ) (h : 0 < α ∧ α < β) : D α ⊆ D β :=
  sorry

end dist_inequalities_D_subset_l29_29153


namespace conditional_independence_law_equiv_l29_29178

noncomputable theory
open ProbabilityTheory

variables {α : Type*} {β : Type*} [MeasurableSpace α] [MeasurableSpace β]
variables (X : ℕ → α) (Y : ℕ → β) (n : ℕ)

def independence_pairwise (X : ℕ → α) (Y : ℕ → β) :=
  ∀ i j, i ≠ j → indep (λ ω, (X i ω, Y i ω)) (λ ω, (X j ω, Y j ω))

def sigma_algebra_Y (Y : ℕ → β) (n : ℕ) : MeasurableSpace α :=
  measurable_space.comap (λ x, \(n : ℕ), (Y n x)) measurable_space.{β}

theorem conditional_independence (h_indep : independence_pairwise X Y) :
  ∀ i, i ∈ finset.range n → indep_cond (λ ω, X ω) (λ ω, Y ω) :=
begin
  sorry
end

theorem law_equiv (h_indep : independence_pairwise X Y) :
  ∀ i, i ∈ finset.range n → 
  (∀ (B : set α), measurable_set B → 
  P[X i ∈ B | sigma_algebra_Y Y n] = P[X i ∈ B | Y i]) :=
begin
  sorry
end

end conditional_independence_law_equiv_l29_29178


namespace find_t_l29_29220

variables {a b c r s t : ℝ}

theorem find_t (h1 : a + b + c = -3)
             (h2 : a * b + b * c + c * a = 4)
             (h3 : a * b * c = -1)
             (h4 : ∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 → (x = a ∨ x = b ∨ x = c))
             (h5 : ∀ y, y^3 + r*y^2 + s*y + t = 0 → (y = a + b ∨ y = b + c ∨ y = c + a))
             : t = 11 :=
sorry

end find_t_l29_29220


namespace fred_owes_greg_accurate_l29_29520

-- Define the initial amounts each person has
def earl_initial : ℕ := 90
def fred_initial : ℕ := 48
def greg_initial : ℕ := 36

-- Define the debts
def earl_owes_fred : ℕ := 28
def greg_owes_earl : ℕ := 40

-- Define the total amount Greg and Earl have together after all debts are paid
def total_earl_greg_after_debts : ℕ := 130

-- Define the amount that Fred owes Greg
def fred_owes_greg : ℕ := 32

theorem fred_owes_greg_accurate :
  -- We need to show that Fred owes Greg $32 when the conditions are accounted for
  (earl_initial - earl_owes_fred + greg_owes_earl) + (greg_initial - greg_owes_earl + fred_owes_greg) = total_earl_greg_after_debts :=
begin
  sorry
end

end fred_owes_greg_accurate_l29_29520


namespace largest_square_area_l29_29140

theorem largest_square_area (total_string_length : ℕ) (h : total_string_length = 32) : ∃ (area : ℕ), area = 64 := 
  by
    sorry

end largest_square_area_l29_29140


namespace perpendicular_necessary_not_sufficient_l29_29910

variable (l m : Type) [LinearMap l] [LinearMap m] (α : Type) [Submodule α]

-- Conditions
variable (h₁ : m.is_perpendicular_to α)

-- Theorem Statement
theorem perpendicular_necessary_not_sufficient (h₁ : m.is_perpendicular_to α) :
  (l.is_perpendicular_to m) → (l.is_parallel_to α) ∧
  ¬((l.is_parallel_to α) → (l.is_perpendicular_to m)) :=
sorry

end perpendicular_necessary_not_sufficient_l29_29910


namespace option_C_does_not_satisfy_l29_29877

theorem option_C_does_not_satisfy (h1 : ∀ a b : ℝ, a = sqrt 2 ∧ b = 1) :
  let e := (sqrt ((sqrt 2)^2 - 1^2)) / (sqrt 2)
  in e ≠ (sqrt 3) / 2 :=
by
  intro h1
  let a := sqrt 2
  let b := 1
  let e := (sqrt (a^2 - b^2)) / a
  have : e = 1 / sqrt 2 := by sorry
  have : (sqrt 3) / 2 = e := by sorry
  contradiction

end option_C_does_not_satisfy_l29_29877


namespace hyperbola_line_segment_lengths_eq_l29_29888

theorem hyperbola_line_segment_lengths_eq
  {a b k m : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (A B C D : ℝ × ℝ),
    ∃ (l : ℝ → ℝ) (hyperbola : ℝ × ℝ → Prop),
    let hline := λ (p : ℝ × ℝ), ∃ x, p = (x, l x)
    in l = (λ x, k*x + m) ∧ hyperbola = (λ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 / b^2 = 1)
    ∧ hline A ∧ hline B ∧ hline C ∧ hline D
    ∧ hyperbola A ∧ hyperbola B ∧ hyperbola C ∧ hyperbola D) :
    let dist := λ (p1 p2 : ℝ × ℝ), real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
    in dist A B = dist C D := 
sorry

end hyperbola_line_segment_lengths_eq_l29_29888


namespace sum_of_letters_l29_29432

def A : ℕ := 0
def B : ℕ := 1
def C : ℕ := 2
def M : ℕ := 12

theorem sum_of_letters :
  A + B + M + C = 15 :=
by
  sorry

end sum_of_letters_l29_29432


namespace set_equality_l29_29112

theorem set_equality (a : ℤ) : 
  {z : ℤ | ∃ x : ℤ, (x - a = z ∧ a - 1 ≤ x ∧ x ≤ a + 1)} = {-1, 0, 1} :=
by {
  sorry
}

end set_equality_l29_29112


namespace series_sum_l29_29668

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29668


namespace unique_function_l29_29829

-- Define the function specification
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f(y) - 1) = x + y

-- The main theorem stating there is exactly one such function
theorem unique_function :
  ∃! f : ℝ → ℝ, satisfies_condition f ∧ (∀ x : ℝ, f x = x + 1 / 2) := 
by
  sorry

end unique_function_l29_29829


namespace plane_equation_of_points_l29_29013

theorem plane_equation_of_points :
  ∃ A B C D : ℤ, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  ∀ x y z : ℤ, (15 * x + 7 * y + 17 * z - 26 = 0) ↔
  (A * x + B * y + C * z + D = 0) :=
by
  sorry

end plane_equation_of_points_l29_29013


namespace infinite_series_sum_eq_l29_29706

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29706


namespace range_of_m_l29_29852

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 1 - |x|
else x^2 - 4 * x + 3

theorem range_of_m :
  {m : ℝ | f(f(m)) ≥ 0} = {m : ℝ | (-2 ≤ m) ∧ (m ≤ 2 + Real.sqrt 2) ∨ (4 ≤ m)} :=
by
  sorry

end range_of_m_l29_29852


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29271

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29271


namespace tangent_of_angle_sum_l29_29003

theorem tangent_of_angle_sum 
  {r x y : ℝ} 
  (h_r_eq_CD : (5 ^ 2 + (x + 2) ^ 2 = r ^ 2))
  (h_r_eq_AB : (4 ^ 2 + x ^ 2 = r ^ 2)) 
  (h_y_eq_CD_half : y = 5) 
  (h_x_cd_dist_eq_2 : x + 2 ≠ 0) :
  let tan_a : ℚ := (x / 5)
  let tan_b : ℚ := (x / 4)
  let tan_sum : ℚ := (tan_a + tan_b) / (1 - (tan_a * tan_b)) in
  tan_sum = 36 / 77 → 36 + 77 = 113 :=
by
  intros
  sorry

end tangent_of_angle_sum_l29_29003


namespace sum_series_equals_three_fourths_l29_29544

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29544


namespace radius_of_circle_l29_29441

theorem radius_of_circle (C : ℝ) (hC : C = 25.132741228718345) : ∃ r : ℝ, r = 4 :=
by
  have π := Real.pi
  have r := C / (2 * π)
  use r
  rw [hC]
  unfold Real.pi
  have approx_pi : π ≈ 3.141592653589793 -- use a more precise value of π
  linarith

end radius_of_circle_l29_29441


namespace third_smallest_four_digit_in_pascal_triangle_l29_29366

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29366


namespace largest_angle_of_parallelogram_l29_29095

theorem largest_angle_of_parallelogram (sum_angles : ℝ) :
  sum_angles = 70 → (∃ (a b : ℝ), a = 35 ∧ b = 35 ∧ a + b = sum_angles ∧ a + (180 - a) = 180) → 
  (∃ (c d : ℝ), 360 - 70 = c + d ∧ c = 145 ∧ d = 145 ∧ c = 180 - b) :=
by
  intro h1 h2
  use 70
  applyExists
  sorry

end largest_angle_of_parallelogram_l29_29095


namespace third_smallest_four_digit_in_pascal_l29_29348

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29348


namespace percent_pelicans_non_swans_l29_29171

noncomputable def percent_geese := 0.20
noncomputable def percent_swans := 0.30
noncomputable def percent_herons := 0.10
noncomputable def percent_ducks := 0.25
noncomputable def percent_pelicans := 0.15

theorem percent_pelicans_non_swans :
  (percent_pelicans / (1 - percent_swans)) * 100 = 21.43 := 
by 
  sorry

end percent_pelicans_non_swans_l29_29171


namespace condition_on_p_l29_29135

theorem condition_on_p (p q r M : ℝ) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : 0 < M) :
  p > (100 * (q + r)) / (100 - q - r) → 
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M :=
by
  intro h
  -- The proof will go here
  sorry

end condition_on_p_l29_29135


namespace series_sum_correct_l29_29763

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29763


namespace modulus_conjugate_l29_29096

def complex_z := (5 : ℂ) / (3 + 4 * Complex.I)

theorem modulus_conjugate (z : ℂ) (h : z = 5 / (3 + 4 * Complex.I)) : Complex.abs (Complex.conj z) = 1 := 
by
  sorry

end modulus_conjugate_l29_29096


namespace infinite_series_sum_eq_3_div_4_l29_29621

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29621


namespace milk_water_juice_final_ratio_l29_29143

noncomputable def initial_vol := 150
noncomputable def ratio_milk_water_juice := (7, 3, 5)
noncomputable def removed_vol_each := 10
noncomputable def final_solution_vol := 120
noncomputable def ratio_milk_water := (3, 1)
noncomputable def removed_portions_ratio := (7, 3, 5)

theorem milk_water_juice_final_ratio : 
  (initial_vol = 150) →
  (ratio_milk_water_juice = (7, 3, 5)) →
  (removed_vol_each = 10) →
  (final_solution_vol = 120) →
  (ratio_milk_water = (3, 1)) →
  (removed_portions_ratio = (7, 3, 5)) →
  ∃ x : ℕ, 
  let removed_vol := 30,
      ratio_combined := (1, 1, 1) in
  ((10 + x) = (10 + x)) ∧ 
  ((10 + x) = (10 + x)) ∧
  ((ratio_combined = (1, 1, 1))).
Proof
  sorry

end milk_water_juice_final_ratio_l29_29143


namespace cuts_for_20_pentagons_l29_29451

theorem cuts_for_20_pentagons (K : ℕ) : 20 * 540 + (K - 19) * 180 ≤ 360 * K + 540 ↔ K ≥ 38 :=
by
  sorry

end cuts_for_20_pentagons_l29_29451


namespace third_smallest_four_digit_in_pascals_triangle_l29_29303

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29303


namespace angles_of_ABC_k_cannot_be_3_l29_29936

-- Define the right-angled triangle and the required conditions
def right_angled_triangle (A B C : ℝ) (S S1 : ℝ) : Prop :=
  ∃ (a b c : ℝ), angle_ABC A B C = 90 ∧ 
  S = (1/2) * a * b ∧ 
  S1 = π * (c / 2)^2 ∧
  k = (S1 / S)

-- Part (a): Proof the angles are 75 and 15 degrees
theorem angles_of_ABC (A B C : ℝ) (S S1 : ℝ) (k : ℝ)
  (h1 : right_angled_triangle A B C S S1)
  (h2 : k = 2 * π) : 
  angle_ABC A B C = 75 ∧ angle_ABC B A C = 15 :=
sorry

-- Part (b): Proof k cannot be 3
theorem k_cannot_be_3 (A B C : ℝ) (S S1 : ℝ) (k : ℝ)
  (h1 : right_angled_triangle A B C S S1)
  (h2 : k = 3) : 
  false :=
sorry

end angles_of_ABC_k_cannot_be_3_l29_29936


namespace eval_sum_equal_smallest_sum_value_l29_29523

theorem eval_sum_equal (a b c : ℕ) (h₁ : a = 125001) (h₂ : b = 51) (h₃ : c = 0) :
  (∑ k in finset.range 50 + 1, (-1) ^ (k + 1) * (k ^ 3 + k ^ 2 + 1) / ((k + 1)!)) = (a / b! - c) :=
by sorry

theorem smallest_sum_value :
  let a : ℕ := 125001
          b : ℕ := 51
          c : ℕ := 0
  in a + b + c = 125052 :=
by  {
  let a : ℕ := 125001,
  let b : ℕ := 51,
  let c : ℕ := 0,
  have h₁ : a = 125001 by rfl,
  have h₂ : b = 51 by rfl,
  have h₃ : c = 0 by rfl,
  exact add_assoc _ _ _ ▸ add_comm _ _ ▸ (add_assoc _ _ _ ▸ rfl)
}

end eval_sum_equal_smallest_sum_value_l29_29523


namespace find_a1_l29_29076

noncomputable def a (n : ℕ) : ℤ := sorry -- the definition of sequence a_n is not computable without initial terms
noncomputable def S (n : ℕ) : ℤ := sorry -- similarly, the definition of S_n without initial terms isn't given

axiom recurrence_relation (n : ℕ) (h : n ≥ 3): 
  a (n) = a (n - 1) - a (n - 2)

axiom S9 : S 9 = 6
axiom S10 : S 10 = 5

theorem find_a1 : a 1 = 1 :=
by
  sorry

end find_a1_l29_29076


namespace smallest_m_to_make_fm_equal_to_3_l29_29009

def f (x : ℕ) : ℕ := (x^2) % 13

def iterate (n : ℕ) (f : ℕ → ℕ) (x : ℕ) : ℕ :=
match n with
| 0     => x
| (k+1) => f (iterate k f x)

theorem smallest_m_to_make_fm_equal_to_3 :
  ∃ m : ℕ, m > 0 ∧ iterate m f 3 = 3 ∧ ∀ k : ℕ, 0 < k < m → iterate k f 3 ≠ 3 :=
by
  sorry

end smallest_m_to_make_fm_equal_to_3_l29_29009


namespace youngest_sibling_age_l29_29425

theorem youngest_sibling_age
  (Y : ℕ)
  (h1 : Y + (Y + 3) + (Y + 6) + (Y + 7) = 120) :
  Y = 26 :=
by
  -- proof steps would be here 
  sorry

end youngest_sibling_age_l29_29425


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29276

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29276


namespace range_of_c_l29_29847

variable (c : ℝ)

def p : Prop := ∀ x : ℝ, x > 0 → c^x = c^(x+1) / c
def q : Prop := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → x + 1/x > 1/c

theorem range_of_c (h1 : c > 0) (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l29_29847


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29403

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29403


namespace sum_g_k_l29_29183

def g (n : ℕ) : ℕ := 
  let k := Real.toNat (Real.root (4.0) (n.toReal))
  if (Real.root (4.0) (n.toReal) - k.toReal).abs ≤ 0.5 then k else k + 1

theorem sum_g_k (h : ∀ k, 1 ≤ k ∧ k ≤ 4095) : 
  (∑ k in Finset.range 4096 \ {0}, 1 / (g k)) = 824 := 
by 
  sorry

end sum_g_k_l29_29183


namespace polar_to_rectangular_coords_l29_29504

theorem polar_to_rectangular_coords (r θ : ℝ) (x y : ℝ) 
  (hr : r = 5) (hθ : θ = 5 * Real.pi / 4)
  (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x = - (5 * Real.sqrt 2) / 2 ∧ y = - (5 * Real.sqrt 2) / 2 := 
by
  rw [hr, hθ] at hx hy
  simp [Real.cos, Real.sin] at hx hy
  rw [hx, hy]
  constructor
  . sorry
  . sorry

end polar_to_rectangular_coords_l29_29504


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29339

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29339


namespace length_of_train_l29_29466

theorem length_of_train :
  ∀ (L : ℝ) (V : ℝ),
  (∀ t p : ℝ, t = 14 → p = 535.7142857142857 → V = L / t) →
  (∀ t p : ℝ, t = 39 → p = 535.7142857142857 → V = (L + p) / t) →
  L = 300 :=
by
  sorry

end length_of_train_l29_29466


namespace possible_sets_l29_29951

structure Graph (V : Type) :=
(vertices : V → Prop)
(edges : V → V → Prop)
(connected : ∀ x y, exists path : list V, path.head = x ∧ path.last = y ∧ ∀ i, edges (path.nth i) (path.nth (i + 1)))

def shortest_path_length {V : Type} (G : Graph V) (x y : V) : ℕ := sorry

def S {V : Type} (G : Graph V) : set ℕ :=
{d | ∃ x y : V, shortest_path_length G x y = d}

theorem possible_sets (G : Graph ℤ)
  (H_connected : ∀ x y : ℤ, ∃ path : list ℤ, path.head = x ∧ path.last = y ∧ ∀ i, G.edges (path.nth i) (path.nth (i + 1)))
  (H_divisibility : ∀ x y : ℤ, shortest_path_length G x y ∣ (x - y)) :
  S G = set.univ ∨ S G = {0, 1} ∨ S G = {0, 1, 2} ∨ S G = {0, 1, 2, 3} :=
sorry

end possible_sets_l29_29951


namespace widely_spacy_subsets_T15_l29_29491

def T (n : ℕ) : set ℕ := { k | 1 ≤ k ∧ k ≤ n }

def is_widely_spacy (s : set ℕ) : Prop :=
  ∀ k, (k ∈ s) → (k + 1 ∉ s ∧ k + 2 ∉ s ∧ k + 3 ∉ s)

def d : ℕ → ℕ
| 0     := 1 -- The empty set is widely spacy
| 1     := 2
| 2     := 3
| 3     := 4
| 4     := 5
| (n+5) := d (n + 4) + d n

theorem widely_spacy_subsets_T15 : d 15 = 181 := by
  sorry

end widely_spacy_subsets_T15_l29_29491


namespace series_result_l29_29641

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29641


namespace series_sum_correct_l29_29772

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29772


namespace trapezium_area_l29_29028

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l29_29028


namespace total_dogs_l29_29481

variable (U : Type) [Fintype U]
variable (jump fetch shake : U → Prop)
variable [DecidablePred jump] [DecidablePred fetch] [DecidablePred shake]

theorem total_dogs (h_jump : Fintype.card {u | jump u} = 70)
  (h_jump_and_fetch : Fintype.card {u | jump u ∧ fetch u} = 30)
  (h_fetch : Fintype.card {u | fetch u} = 40)
  (h_fetch_and_shake : Fintype.card {u | fetch u ∧ shake u} = 20)
  (h_shake : Fintype.card {u | shake u} = 50)
  (h_jump_and_shake : Fintype.card {u | jump u ∧ shake u} = 25)
  (h_all_three : Fintype.card {u | jump u ∧ fetch u ∧ shake u} = 15)
  (h_none : Fintype.card {u | ¬jump u ∧ ¬fetch u ∧ ¬shake u} = 15) :
  Fintype.card U = 115 :=
by
  sorry

end total_dogs_l29_29481


namespace surface_area_of_circumscribed_sphere_l29_29517

/-- 
  Problem: Determine the surface area of the sphere circumscribed about a cube with edge length 2.

  Given:
  - The edge length of the cube is 2.
  - The space diagonal of a cube with edge length \(a\) is given by \(d = \sqrt{3} \cdot a\).
  - The diameter of the circumscribed sphere is equal to the space diagonal of the cube.
  - The surface area \(S\) of a sphere with radius \(R\) is given by \(S = 4\pi R^2\).

  To Prove:
  - The surface area of the sphere circumscribed about the cube is \(12\pi\).
-/
theorem surface_area_of_circumscribed_sphere (a : ℝ) (π : ℝ) (h1 : a = 2) 
  (h2 : ∀ a, d = Real.sqrt 3 * a) (h3 : ∀ d, R = d / 2) (h4 : ∀ R, S = 4 * π * R^2) : 
  S = 12 * π := 
by
  sorry

end surface_area_of_circumscribed_sphere_l29_29517


namespace part1_part2_l29_29105

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * abs (x^2 - a)

-- Define the two main proofs to be shown
theorem part1 (a : ℝ) (h : a = 1) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ I2 = Set.Icc (-1 + Real.sqrt 2) (1) ∧ 
  ∀ x ∈ I1 ∪ I2, ∀ y ∈ I1 ∪ I2, x ≤ y → f y 1 ≤ f x 1 :=
sorry

theorem part2 (a : ℝ) (h : a ≥ 0) (h_roots : ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = m) ∧ (∃ x : ℝ, x < 0 ∧ f x a = m)) : 
  ∃ m : ℝ, m = 4 / (Real.exp 2) :=
sorry

end part1_part2_l29_29105


namespace smallest_b_for_27_pow_b_gt_3_pow_24_l29_29427

theorem smallest_b_for_27_pow_b_gt_3_pow_24 :
  ∃ b : ℤ, 27 = 3^3 ∧ 27^b > 3^(24 : ℤ) ∧ (∀ b' : ℤ, 27 = 3^3 ∧ 27^b' > 3^24 → b' ≥ b) ∧ b = 9 := 
sorry

end smallest_b_for_27_pow_b_gt_3_pow_24_l29_29427


namespace value_of_k_l29_29192

theorem value_of_k :
  (∀ x, y, (x = 2 ∧ y = 13) → y = 5 * x + 3 ∧ y = k * x + 7) →
  k = 3 :=
by
  intro h
  apply funext
  intros
  sorry

end value_of_k_l29_29192


namespace multiple_of_k_l29_29130

theorem multiple_of_k (k : ℕ) (m : ℕ) (h₁ : 7 ^ k = 2) (h₂ : 7 ^ (m * k + 2) = 784) : m = 2 :=
sorry

end multiple_of_k_l29_29130


namespace sum_of_coordinates_is_ten_l29_29204

-- Definition of the coordinates of points C and D
def C_x : ℝ := 5
def C_y : ℝ
def D_x : ℝ := C_x
def D_y : ℝ := -C_y

-- The theorem stating the problem condition and proof goal
theorem sum_of_coordinates_is_ten : ∀ (y : ℝ), C_x + C_y + D_x + D_y = 10 := by
  intro y
  sorry

end sum_of_coordinates_is_ten_l29_29204


namespace hall_of_mirrors_area_l29_29165

def area_rectangular (length width : ℝ) : ℝ :=
length * width

def area_triangular (base height : ℝ) : ℝ :=
(1/2) * base * height

def area_trapezoidal (parallel_side1 parallel_side2 height : ℝ) : ℝ :=
(1/2) * (parallel_side1 + parallel_side2) * height

noncomputable def total_glass_area : ℝ :=
let rectangular_area := 2 * area_rectangular 30 12
let triangular_area := area_triangular 20 12
let trapezoidal_area1 := area_trapezoidal 20 15 12
let trapezoidal_area2 := area_trapezoidal 25 18 12
in rectangular_area + triangular_area + trapezoidal_area1 + trapezoidal_area2

theorem hall_of_mirrors_area : total_glass_area = 1308 := by
sorry

end hall_of_mirrors_area_l29_29165


namespace ratio_frogs_to_dogs_l29_29145

variable (D C F : ℕ)

-- Define the conditions as given in the problem statement
def cats_eq_dogs_implied : Prop := C = Nat.div (4 * D) 5
def frogs : Prop := F = 160
def total_animals : Prop := D + C + F = 304

-- Define the statement to be proved
theorem ratio_frogs_to_dogs (h1 : cats_eq_dogs_implied D C) (h2 : frogs F) (h3 : total_animals D C F) : F / D = 2 := by
  sorry

end ratio_frogs_to_dogs_l29_29145


namespace num_acute_triangles_with_side_15_20_l29_29840

def is_acute_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ 
  (a * a + b * b > c * c) ∧ (a * a + c * c > b * b) ∧ (b * b + c * c > a * a)

theorem num_acute_triangles_with_side_15_20 : 
  {x : ℕ | is_acute_triangle 15 20 x ∧ 5 < x ∧ x < 35}.to_finset.card = 11 :=
sorry

end num_acute_triangles_with_side_15_20_l29_29840


namespace third_smallest_four_digit_in_pascals_triangle_l29_29361

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29361


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29273

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29273


namespace series_sum_correct_l29_29767

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29767


namespace part1_part2_l29_29103

noncomputable def f (x a : ℝ) : ℝ := cos x ^ 2 + a * sin x + 2 * a - 1

-- Part 1: For a = 1, find the maximum and minimum of the function.
theorem part1 (x : ℝ) : (f x 1 ≤ 9 / 4) ∧ (0 ≤ f x 1) :=
  sorry

-- Part 2: Determine the range of values for a given that f(x) ≤ 5 for all x in [-π/2, π/2].
theorem part2 (a : ℝ) (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) : (f x a ≤ 5) → (a ≤ 2) :=
  sorry

end part1_part2_l29_29103


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29412

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29412


namespace number_of_sequences_l29_29980

theorem number_of_sequences (n : ℕ) : 
  ∀ (seq : fin n → ℕ), 3 ∈ seq → 
  (∀ i : fin (n-1), abs (seq i - seq (i + 1)) ≤ 1) →
  finset.card { s : fin n → ℕ | 3 ∈ s ∧ (∀ i : fin (n-1), abs (s i - s (i + 1)) ≤ 1) } = 3^n - 2^n :=
by
  sorry

end number_of_sequences_l29_29980


namespace sum_of_series_eq_three_fourths_l29_29578

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29578


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29336

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29336


namespace third_smallest_four_digit_in_pascals_triangle_l29_29306

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29306


namespace sum_series_eq_3_div_4_l29_29558

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29558


namespace third_smallest_four_digit_in_pascals_triangle_l29_29325

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29325


namespace sum_geometric_series_l29_29690

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29690


namespace sum_series_eq_3_over_4_l29_29760

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29760


namespace solution_set_inequality_l29_29833

theorem solution_set_inequality (x : ℝ) : (\(frac{1 - x}{2 + x} \geq 0\)) → (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solution_set_inequality_l29_29833


namespace remainder_n_squared_plus_3n_plus_5_l29_29906

theorem remainder_n_squared_plus_3n_plus_5 (n : ℕ) (h : n % 25 = 24) : (n^2 + 3 * n + 5) % 25 = 3 :=
by
  sorry

end remainder_n_squared_plus_3n_plus_5_l29_29906


namespace third_smallest_four_digit_Pascal_triangle_l29_29287

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29287


namespace rectangular_to_cylindrical_correct_l29_29008

noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ := 
  let r := real.sqrt (x*x + y*y)
  let θ := if y > 0 then real.pi - real.arctan (abs y / abs x) else 2 * real.pi - (real.pi - real.arctan (abs y / abs x))
  (r, θ, z)

theorem rectangular_to_cylindrical_correct :
  rectangular_to_cylindrical (-3) 4 5 = (5, real.pi - real.arctan(4 / 3), 5) := 
by
  sorry

end rectangular_to_cylindrical_correct_l29_29008


namespace third_smallest_four_digit_Pascal_triangle_l29_29281

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29281


namespace triangle_side_c_and_angle_C_l29_29092

theorem triangle_side_c_and_angle_C
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b + c = sqrt 2 + 1)
  (h2 : sin A + sin B = sqrt 2 * sin C)
  (h3 : (1 / 2) * a * b * sin C = 1 / 6 * sin C) :
  c = 1 ∧ C = π / 3 :=
by
  sorry

end triangle_side_c_and_angle_C_l29_29092


namespace count_unbounded_g1_sequences_l29_29506

def g1 (n : ℕ) : ℕ :=
  if h : n = 1 then 1
  else let ⟨factors, prod_eq_n⟩ := nat.prime_factorization_factors_unique n h
       in nat.prod (factors.to_finset.image (λ p, (p + 2)^(factors.count p - 1)))

def gm : ℕ → ℕ → ℕ
| 1 n := g1 n
| (m + 1) n := g1 (gm m n)

def is_unbounded_seq (f : ℕ → ℕ) (seq : ℕ → ℕ) :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ seq n > 0

def g_seq (m : ℕ) (n : ℕ) := gm m n

theorem count_unbounded_g1_sequences : 
  (finset.range 501).filter (λ n, is_unbounded_seq (gm 1 n) (gm 2 n)).card = 500 :=
  sorry

end count_unbounded_g1_sequences_l29_29506


namespace third_smallest_four_digit_in_pascal_triangle_l29_29367

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29367


namespace infinite_series_sum_eq_3_over_4_l29_29724

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29724


namespace xyz_problem_l29_29056

theorem xyz_problem (x y : ℝ) (h1 : x + y - x * y = 155) (h2 : x^2 + y^2 = 325) : |x^3 - y^3| = 4375 := by
  sorry

end xyz_problem_l29_29056


namespace work_together_days_l29_29422

-- Define the days it takes for A and B to complete the work individually.
def days_A : ℕ := 3
def days_B : ℕ := 6

-- Define the combined work rate.
def combined_work_rate : ℚ := (1 / days_A) + (1 / days_B)

-- State the theorem for the number of days A and B together can complete the work.
theorem work_together_days :
  1 / combined_work_rate = 2 := by
  sorry

end work_together_days_l29_29422


namespace minimum_distance_l29_29070

noncomputable def cube_edge : ℝ := 1

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def on_diagonal_BC1 (P : Point3D) : Prop :=
(P.x = 1 - P.z) ∧ (P.y = 1 - P.z)

def on_base_ABCD (Q : Point3D) : Prop :=
Q.z = 0

def distance (A B : Point3D) : ℝ :=
real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2 + (B.z - A.z)^2)

def point_C1 : Point3D := { x := 1, y := 1, z := 1 }
def point_D1 : Point3D := { x := 0, y := 0, z := 1 }
def point_B : Point3D := { x := 1, y := 1, z := 0 }

theorem minimum_distance (P Q : Point3D) (hP : on_diagonal_BC1 P) (hQ : on_base_ABCD Q) :
  (distance point_D1 P) + (distance P Q) = 1 + real.sqrt(2) / 2 := sorry

end minimum_distance_l29_29070


namespace huahuan_initial_cards_l29_29428

theorem huahuan_initial_cards
  (a b c : ℕ) -- let a, b, c be the initial number of cards Huahuan, Yingying, and Nini have
  (total : a + b + c = 2712)
  (condition_after_50_rounds : ∃ d, b = a + d ∧ c = a + 2 * d) -- after 50 rounds, form an arithmetic sequence
  : a = 754 := sorry

end huahuan_initial_cards_l29_29428


namespace max_slope_a_l29_29005

def is_nonprime (n : ℕ) : Prop :=
  ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def valid_slope (m : ℚ) : Prop :=
  1 / 3 < m ∧ m < 152 / 451 ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ Int.gcd p q = 1 ∧ q ≠ 1 ∧ is_nonprime q ∧ m = p / q)

theorem max_slope_a : (forall x, 0 < x ∧ x ≤ 150 → (∃ k, x = k * 451 → ¬ ∃ y, ⟨x, y⟩ ∈ (λ m x b, y = m * x + b) (152 / 451) b x)) :=
sorry

end max_slope_a_l29_29005


namespace wheel_diameter_l29_29467

variable (distance : ℝ) (revolutions : ℝ)

theorem wheel_diameter (h₁ : distance = 1056) (h₂ : revolutions = 6.005459508644222) :
  let π := Real.pi in
  let C := distance / revolutions in
  let D := C / π in
  D ≈ 56 :=
by
  sorry

end wheel_diameter_l29_29467


namespace third_smallest_four_digit_in_pascal_l29_29345

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29345


namespace part1_part2_l29_29880

noncomputable def f (x a : ℝ) : ℝ := a * x - 2 * log x + 2

theorem part1 (a : ℝ) : (∀ x, has_deriv_at (λ x, f x a) (a - 2 / x) 1) → a = 2 := 
sorry

theorem part2 (a : ℝ) : ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) → (0 < a ∧ a < 2 / real.exp 2)) :=
sorry

end part1_part2_l29_29880


namespace quadratic_distinct_real_roots_l29_29872

theorem quadratic_distinct_real_roots (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 = 0 → 
  (k ≠ 0 ∧ ((-2)^2 - 4 * k * (-1) > 0))) ↔ (k > -1 ∧ k ≠ 0) := 
sorry

end quadratic_distinct_real_roots_l29_29872


namespace third_smallest_four_digit_in_pascals_triangle_l29_29317

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29317


namespace third_smallest_four_digit_in_pascal_triangle_l29_29376

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29376


namespace sum_series_eq_3_div_4_l29_29560

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29560


namespace series_converges_to_three_fourths_l29_29801

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29801


namespace evaluate_series_sum_l29_29606

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29606


namespace sum_of_series_eq_three_fourths_l29_29587

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29587


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29280

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29280


namespace length_of_MN_l29_29854

variable (A B C D K L M N : Point)
variable (AB CD AK KL DN MN : ℝ)
variable (h1 : Rectangle A B C D)
variable (h2 : CircleIntersectAtPoints A B K L)
variable (h3 : CircleIntersectAtPoints C D M N)
variable (h4 : AK = 10)
variable (h5 : KL = 17)
variable (h6 : DN = 7)

theorem length_of_MN :
  MN = 23 := sorry

end length_of_MN_l29_29854


namespace series_result_l29_29626

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29626


namespace trapezium_area_l29_29032

-- Define the lengths of the parallel sides and the distance between them
def side_a : ℝ := 20
def side_b : ℝ := 18
def height : ℝ := 15

-- Define the formula for the area of a trapezium
def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem
theorem trapezium_area :
  area_trapezium side_a side_b height = 285 :=
by
  sorry

end trapezium_area_l29_29032


namespace infinite_series_sum_value_l29_29735

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29735


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29274

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29274


namespace sum_series_eq_3_div_4_l29_29570

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29570


namespace evaluate_series_sum_l29_29607

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29607


namespace total_games_attended_l29_29941

def games_in_months (this_month previous_month next_month following_month fifth_month : ℕ) : ℕ :=
  this_month + previous_month + next_month + following_month + fifth_month

theorem total_games_attended :
  games_in_months 24 32 29 19 34 = 138 :=
by
  -- Proof will be provided, but ignored for this problem
  sorry

end total_games_attended_l29_29941


namespace series_result_l29_29638

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29638


namespace third_smallest_four_digit_in_pascals_triangle_l29_29385

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29385


namespace pascal_third_smallest_four_digit_number_l29_29398

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29398


namespace infinite_series_sum_eq_3_over_4_l29_29720

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29720


namespace mod_exp_pattern_l29_29267

theorem mod_exp_pattern : (7 ^ 123) % 9 = 1 :=
by
  -- Acknowledge the recurring pattern that \(7^{3k} \equiv 1 \pmod{9}\)
  have h : (7 ^ 3) % 9 = 1 := by
    calc
      (7 ^ 3) % 9 = (343) % 9 : by norm_num
               ... = 1 : by norm_num
  -- Use the exponent rule that \( (a ^ b) ^ c = a ^ (b * c) \)
  calc
    (7 ^ 123) % 9 = ((7 ^ 3) ^ 41) % 9 : by rw [pow_mul]
               ... = (1 ^ 41) % 9    : by rw [h]
               ... = 1               : by norm_num

-- Sorry was added as placeholder to ensure the Lean proof checks successfully.
-- Remove the comments for detailed steps and execute each proof step sequentially to validate.

end mod_exp_pattern_l29_29267


namespace male_teacher_classes_per_month_l29_29232

theorem male_teacher_classes_per_month (x y a : ℕ) :
  (15 * x = 6 * (x + y)) ∧ (a * y = 6 * (x + y)) → a = 10 :=
by
  sorry

end male_teacher_classes_per_month_l29_29232


namespace jenny_total_wins_l29_29168

theorem jenny_total_wins :
  let games_against_friend := {
    Mark := 20,
    Jill := 20 * 2,
    Sarah := 15,
    Tom := 25
  }
  let win_percentage := {
    Mark := 0.75,
    Jill := 0.40,
    Sarah := 0.20,
    Tom := 0.60
  }
  let wins_against_friend (name : String) : ℕ :=
    (games_against_friend.find name).getD 0 * (win_percentage.find name).getD 0

  wins_against_friend "Mark" + wins_against_friend "Jill" +
  wins_against_friend "Sarah" + wins_against_friend "Tom" = 49 :=
by
  sorry

end jenny_total_wins_l29_29168


namespace sum_series_div_3_powers_l29_29787

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29787


namespace kitchen_width_l29_29194

theorem kitchen_width (length : ℕ) (height : ℕ) (rate : ℕ) (hours : ℕ) (coats : ℕ) 
  (total_painted : ℕ) (half_walls_area : ℕ) (total_walls_area : ℕ)
  (width : ℕ) : 
  length = 12 ∧ height = 10 ∧ rate = 40 ∧ hours = 42 ∧ coats = 3 ∧ 
  total_painted = rate * hours ∧ total_painted = coats * total_walls_area ∧
  half_walls_area = 2 * length * height ∧ total_walls_area = half_walls_area + 2 * width * height ∧
  2 * (total_walls_area - half_walls_area / 2) = 2 * width * height →
  width = 16 := 
by
  sorry

end kitchen_width_l29_29194


namespace sum_series_equals_three_fourths_l29_29556

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29556


namespace angle_B_value_value_of_k_l29_29160

variable {A B C a b c : ℝ}
variable {k : ℝ}
variable {m n : ℝ × ℝ}

theorem angle_B_value
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = Real.pi / 3 :=
by sorry

theorem value_of_k
  (hA : 0 < A ∧ A < 2 * Real.pi / 3)
  (hm : m = (Real.sin A, Real.cos (2 * A)))
  (hn : n = (4 * k, 1))
  (hM : 4 * k * Real.sin A + Real.cos (2 * A) = 7) :
  k = 2 :=
by sorry

end angle_B_value_value_of_k_l29_29160


namespace sum_series_eq_3_div_4_l29_29565

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29565


namespace xiao_ming_completion_days_l29_29438

/-
  Conditions:
  1. The total number of pages is 960.
  2. The planned number of days to finish the book is 20.
  3. Xiao Ming actually read 12 more pages per day than planned.

  Question:
  How many days did it actually take Xiao Ming to finish the book?

  Answer:
  The actual number of days to finish the book is 16 days.
-/

open Nat

theorem xiao_ming_completion_days :
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  actual_days = 16 :=
by
  let total_pages := 960
  let planned_days := 20
  let additional_pages_per_day := 12
  let planned_pages_per_day := total_pages / planned_days
  let actual_pages_per_day := planned_pages_per_day + additional_pages_per_day
  let actual_days := total_pages / actual_pages_per_day
  show actual_days = 16
  sorry

end xiao_ming_completion_days_l29_29438


namespace cube_volume_is_64_l29_29245

def cube_volume_from_edge_sum (edge_sum : ℝ) (num_edges : ℕ) : ℝ :=
  let edge_length := edge_sum / num_edges in
  edge_length ^ 3

theorem cube_volume_is_64 :
  cube_volume_from_edge_sum 48 12 = 64 := by
  sorry

end cube_volume_is_64_l29_29245


namespace sum_series_equals_three_fourths_l29_29553

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29553


namespace find_perpendicular_line_l29_29037

theorem find_perpendicular_line (x y : ℝ) (h₁ : y = (1/2) * x + 1)
    (h₂ : (x, y) = (2, 0)) : y = -2 * x + 4 :=
sorry

end find_perpendicular_line_l29_29037


namespace third_smallest_four_digit_in_pascal_l29_29342

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29342


namespace third_smallest_four_digit_in_pascals_triangle_l29_29326

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29326


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29269

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29269


namespace series_sum_l29_29675

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29675


namespace correct_equation_l29_29154

variable (x : ℕ)

def three_people_per_cart_and_two_empty_carts (x : ℕ) :=
  x / 3 + 2

def two_people_per_cart_and_nine_walking (x : ℕ) :=
  (x - 9) / 2

theorem correct_equation (x : ℕ) :
  three_people_per_cart_and_two_empty_carts x = two_people_per_cart_and_nine_walking x :=
by
  sorry

end correct_equation_l29_29154


namespace significant_figures_rounding_l29_29915

theorem significant_figures_rounding (n : ℕ) (h_n : n = 201949) :
  (∃ (a : ℝ) (k : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (a = 2.0) ∧ (k = 5) ∧ (h_n = a * 10^k)) :=
sorry

end significant_figures_rounding_l29_29915


namespace sum_series_equals_three_fourths_l29_29543

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29543


namespace increase_by_percentage_l29_29129

theorem increase_by_percentage (a b : ℝ) (percentage : ℝ) (final : ℝ) : b = a * percentage → final = a + b → final = 437.5 :=
by
  sorry

end increase_by_percentage_l29_29129


namespace infinite_series_sum_value_l29_29742

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29742


namespace smallest_n_divisible_sum_of_squares_l29_29832

theorem smallest_n_divisible_sum_of_squares :
  ∃ n : ℕ, 
    (∀ m : ℕ, 0 < m ∧ m < n → ¬(∃ k : ℕ, (m * (m + 1) * (2 * m + 1)) = 600 * k)) ∧
    (∃ k : ℕ, (n * (n + 1) * (2 * n + 1)) = 600 * k) :=
begin
  sorry
end

end smallest_n_divisible_sum_of_squares_l29_29832


namespace value_of_m_l29_29057

/-- 
For each integer n ≥ 5, let a_n denote the base-n number 0.144 with an infinite repeating cycle.
-/
def a_n (n : ℕ) (h : n ≥ 5) : ℚ :=
  (n^2 + 4*n + 4) / (n^3 - 1)

theorem value_of_m : (∏ n in finset.range' 5 96, a_n n (by linarith)) = 59180 / (nat.factorial 100) :=
sorry

end value_of_m_l29_29057


namespace infinite_series_sum_eq_3_over_4_l29_29712

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29712


namespace series_converges_to_three_fourths_l29_29796

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29796


namespace solution_set_of_f_l29_29090
noncomputable def f : ℝ → ℝ := sorry

axiom even_fn : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom monotonicity : ∀ x1 x2 : ℝ, x1 ∈ Iic 2 → x2 ∈ Iic 2 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0
axiom value_at_1 : f 1 = 0

theorem solution_set_of_f : ∀ x : ℝ, f x < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end solution_set_of_f_l29_29090


namespace no_two_or_three_digits_count_l29_29896

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 2 or 3 is 951. -/
theorem no_two_or_three_digits_count : 
  let digits := [0, 1, 4, 5, 6, 7, 8, 9]
  let valid_numbers := (1 : ℕ) to 2000 |>.filter (λ n, n.digits_dec.all (∈ digits))
  valid_numbers.length = 951 :=
by
  sorry

end no_two_or_three_digits_count_l29_29896


namespace series_sum_correct_l29_29778

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29778


namespace evaluate_series_sum_l29_29604

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29604


namespace max_value_b_l29_29189

noncomputable def f (x a : ℝ) := (3/2) * x^2 - 2 * a * x
noncomputable def g (x a b : ℝ) := a^2 * (Real.log x) + b

theorem max_value_b (a b : ℝ) (h_pos : a > 0) :
  (∃ (x0 : ℝ), (f x0 a = g x0 a b) ∧ (f.deriv x0 a = g.deriv x0 (a^2) b)) → b ≤ 1/(2 * Real.exp 2) :=
sorry

end max_value_b_l29_29189


namespace infinite_series_sum_value_l29_29729

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29729


namespace exists_nat_numbers_except_two_three_l29_29820

theorem exists_nat_numbers_except_two_three (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ (k ≠ 2 ∧ k ≠ 3) :=
by
  sorry

end exists_nat_numbers_except_two_three_l29_29820


namespace sum_series_div_3_powers_l29_29785

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29785


namespace series_sum_correct_l29_29775

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29775


namespace area_quadrilateral_WXYZ_l29_29421

-- Definitions related to the problem
def Rectangle (AB : ℝ) (BC : ℝ) := 
    { AB = 20, BC = 3 }

def Circle (center : ℝ × ℝ) (radius : ℝ) := 
    { center = (10, 3 / 2), -- Midpoint of DC is half of AB, and midpoint of BC
      radius = 5 }

-- The intersecting points W, X, Y, Z form a quadrilateral
-- whose area we need to find given the conditions above.

theorem area_quadrilateral_WXYZ
  (AB BC : ℝ)
  (rect : Rectangle AB BC)
  (o : ℝ × ℝ)
  (r : ℝ)
  (circ : Circle o r)
  (A B C D W X Y Z : ℝ × ℝ) 
  (intersects : (W ∈ Circle o r) ∧ (X ∈ Circle o r) ∧ (Y ∈ Circle o r) ∧ (Z ∈ Circle o r)) :
  let area := (1 / 2 : ℝ) * ((10 : ℝ) + 8) * 3 in
  area = 27 := 
begin
  sorry 
end

end area_quadrilateral_WXYZ_l29_29421


namespace repeating_fraction_simplification_l29_29264

theorem repeating_fraction_simplification :
  (0.727272727272... : ℝ) / (0.272727272727... : ℝ) = (8 : ℝ) / (3 : ℝ) :=
by
  sorry

end repeating_fraction_simplification_l29_29264


namespace third_smallest_four_digit_in_pascals_triangle_l29_29322

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29322


namespace problems_per_page_l29_29162

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def remaining_pages : ℕ := 5
def remaining_problems : ℕ := total_problems - finished_problems

theorem problems_per_page : remaining_problems / remaining_pages = 8 := 
by
  sorry

end problems_per_page_l29_29162


namespace infinite_series_sum_value_l29_29738

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29738


namespace infinite_series_sum_eq_l29_29700

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29700


namespace third_smallest_four_digit_in_pascal_triangle_l29_29370

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29370


namespace solve_system_eqns_l29_29994

noncomputable def eq1 (x y z : ℚ) : Prop := x^2 + 2 * y * z = x
noncomputable def eq2 (x y z : ℚ) : Prop := y^2 + 2 * z * x = y
noncomputable def eq3 (x y z : ℚ) : Prop := z^2 + 2 * x * y = z

theorem solve_system_eqns (x y z : ℚ) :
  (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ↔
  ((x, y, z) = (0, 0, 0) ∨
   (x, y, z) = (1/3, 1/3, 1/3) ∨
   (x, y, z) = (1, 0, 0) ∨
   (x, y, z) = (0, 1, 0) ∨
   (x, y, z) = (0, 0, 1) ∨
   (x, y, z) = (2/3, -1/3, -1/3) ∨
   (x, y, z) = (-1/3, 2/3, -1/3) ∨
   (x, y, z) = (-1/3, -1/3, 2/3)) :=
by sorry

end solve_system_eqns_l29_29994


namespace range_of_a_l29_29101

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) (h4 : ∀ x, g a x ≤ 0 → ¬(x < 0 ∧ g a x > 0)) :
  2 < a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l29_29101


namespace infinite_series_sum_value_l29_29734

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29734


namespace third_smallest_four_digit_in_pascal_l29_29346

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29346


namespace infinite_series_sum_eq_l29_29697

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29697


namespace tan_half_sum_l29_29184

variable (p q : ℝ)

-- Given conditions
def cos_condition : Prop := (Real.cos p + Real.cos q = 1 / 3)
def sin_condition : Prop := (Real.sin p + Real.sin q = 4 / 9)

-- Prove the target expression
theorem tan_half_sum (h1 : cos_condition p q) (h2 : sin_condition p q) : 
  Real.tan ((p + q) / 2) = 4 / 3 :=
sorry

-- For better readability, I included variable declarations and definitions separately

end tan_half_sum_l29_29184


namespace problem1_problem2_l29_29860

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + (a - 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem problem1 (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) → a = 2 ∨ a = 3 := sorry

theorem problem2 (m : ℝ) : (∀ x, x ∈ A → x ∈ C m) → m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) := sorry

end problem1_problem2_l29_29860


namespace even_numbers_greater_than_40000_l29_29256

theorem even_numbers_greater_than_40000 :
  let digits := {0, 1, 2, 3, 4, 5},
      five_digit_numbers := { n : ℕ | 40000 < n ∧ 
                                          digits ⊆ { (d : ℕ) | d ∈ digits }
                                          ∧ n % 2 = 0},
      unique_digits (n : ℕ) := list.nodup (nat.digits 10 n)
  in
  ∑ n in five_digit_numbers, (unique_digits n ∧ digits ⊆ (nat.digits 10 n)) = 120 := 
sorry

end even_numbers_greater_than_40000_l29_29256


namespace seating_arrangement_l29_29922

theorem seating_arrangement (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ n : ℕ, n = (boys.factorial * girls.factorial) + (girls.factorial * boys.factorial) ∧ n = 288) :=
by 
  sorry

end seating_arrangement_l29_29922


namespace smaller_part_volume_l29_29071

noncomputable def volume_of_smaller_part (a : ℝ) : ℝ :=
  (25 / 144) * (a^3)

theorem smaller_part_volume (a : ℝ) (h_pos : 0 < a) :
  ∃ v : ℝ, v = volume_of_smaller_part a :=
  sorry

end smaller_part_volume_l29_29071


namespace sample_mean_and_variance_l29_29460

def sample : List ℕ := [10, 12, 9, 14, 13]
def n : ℕ := 5

-- Definition of sample mean
noncomputable def sampleMean : ℝ := (sample.sum / n)

-- Definition of sample variance using population formula
noncomputable def sampleVariance : ℝ := (sample.map (λ x_i => (x_i - sampleMean)^2)).sum / n

theorem sample_mean_and_variance :
  sampleMean = 11.6 ∧ sampleVariance = 3.44 := by
  sorry

end sample_mean_and_variance_l29_29460


namespace cube_volume_l29_29239

theorem cube_volume : 
  (∃ (s : ℝ), 12 * s = 48) → 
  (∃ (V : ℝ), V = (4:ℝ)^3 ∧ V = 64) := 
by 
  intro h
  rcases h with ⟨s, hs⟩
  use s^3
  have hs_eq : s = 4 := by linarith
  rw [hs_eq, pow_succ, pow_succ, pow_zero, mul_assoc, mul_assoc, mul_one, pow_succ, pow_zero]
  norm_num
  exact ⟨rfl, by norm_num⟩

end cube_volume_l29_29239


namespace find_x_plus_2y_l29_29083

noncomputable def x : ℝ := Real.log 3 / Real.log 2
noncomputable def y : ℝ := Real.log (8 / 3) / Real.log 4

theorem find_x_plus_2y : x + 2 * y = 3 := by
  let x := x
  let y := y
  have h1 : 2 ^ x = 3 := by
    rw [Real.rpow_nat_cast, Real.rpow_mul, Real.log_def]
    exact rfl
  have h2 : y = Real.log (8 / 3) / Real.log 4 := by
    rw Real.log_div
    exact rfl
  sorry

end find_x_plus_2y_l29_29083


namespace find_ordered_pair_l29_29824

theorem find_ordered_pair : ∃ x y : ℝ, (2 * x - 3 * y = -5 ∧ 5 * x - 2 * y = 4) ∧ x = 2 ∧ y = 3 :=
by {
  have h1 : ∃ x y : ℝ, 2 * x - 3 * y = -5 ∧ 5 * x - 2 * y = 4, sorry,
  obtain ⟨x, y, h2, h3⟩ := h1,
  use [x, y],
  split,
  exact ⟨h2, h3⟩,
  sorry -- This is where the specific values are checked.
  }

end find_ordered_pair_l29_29824


namespace circle_tangency_intersections_l29_29098

theorem circle_tangency_intersections (a : ℝ) :
  let O1 := {x1 | x1.1^2 + x1.2^2 = 4}
  let O2 := {x2 | (x2.1 - a)^2 + x2.2^2 = 1}
  (∃ p : ℝ × ℝ, p ∈ O1 ∧ p ∈ O2 ∧ ∀ q ∈ O1, ∀ r ∈ O2, q = r → q=p) ↔ a ∈ {1, -1, 3, -3} :=
sorry

end circle_tangency_intersections_l29_29098


namespace series_sum_correct_l29_29774

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29774


namespace sum_series_eq_3_over_4_l29_29748

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29748


namespace tennis_balls_in_each_container_l29_29164

theorem tennis_balls_in_each_container :
  let total_balls := 100
  let given_away := total_balls / 2
  let remaining := total_balls - given_away
  let containers := 5
  remaining / containers = 10 := 
by
  sorry

end tennis_balls_in_each_container_l29_29164


namespace cos2_theta_plus_cos_theta_gt_one_l29_29844

variable (θ : ℝ)
variable (h1 : sin θ ^ 2 + sin θ = 1)
variable (h2 : 0 < θ ∧ θ < π / 2)

theorem cos2_theta_plus_cos_theta_gt_one (h1 : sin θ ^ 2 + sin θ = 1) (h2 : 0 < θ ∧ θ < π / 2) : 
  cos θ ^ 2 + cos θ > 1 :=
sorry

end cos2_theta_plus_cos_theta_gt_one_l29_29844


namespace infinite_series_sum_eq_l29_29703

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29703


namespace hyperbola_equation_l29_29870

theorem hyperbola_equation
  (C : ℝ → ℝ → Prop)
  (passes_through : C 3 (real.sqrt 2))
  (shares_asymptotes_with : ∀ x y, C x y ↔ ∃ λ ≠ 0, (x^2 / 6) - (y^2 / 2) = λ):
  ∀ x y, C x y ↔ (x^2 / 3) - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l29_29870


namespace JenniferSpentFractionOnSandwich_l29_29166

noncomputable def fractionSpentOnSandwich (x : ℝ) :=
  (x * 150 + (1 / 6) * 150 + (1 / 2) * 150 = 130) → (x = 1 / 5)

theorem JenniferSpentFractionOnSandwich :
  ∃ x : ℝ, fractionSpentOnSandwich x :=
begin
  use 1 / 5,
  intro h,
  rw [mul_assoc, mul_assoc, mul_assoc] at h,
  norm_num at h,
  sorry
end

end JenniferSpentFractionOnSandwich_l29_29166


namespace pascal_third_smallest_four_digit_number_l29_29393

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29393


namespace infinite_series_sum_eq_3_div_4_l29_29620

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29620


namespace angle_C_eq_2π_over_3_l29_29231

variables {a b c : ℝ}
variable (h : ({a + c, b} : ℝ × ℝ) ∥ ({b + a, c - a} : ℝ × ℝ))

theorem angle_C_eq_2π_over_3 (h : ({a + c, b} : ℝ × ℝ) ∥ ({b + a, c - a} : ℝ × ℝ)) : ∠ABC = 2 * π / 3 :=
sorry

end angle_C_eq_2π_over_3_l29_29231


namespace infinite_series_sum_value_l29_29736

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29736


namespace third_smallest_four_digit_in_pascals_triangle_l29_29364

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29364


namespace third_smallest_four_digit_in_pascals_triangle_l29_29315

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29315


namespace probability_of_selecting_specified_clothing_l29_29146

-- Define the number of each type of clothing
def shirts := 6
def shorts := 7
def socks := 8

-- Define the number of articles to be removed
def articles_to_remove := 5

-- Calculate the total number of articles
def total_articles := shirts + shorts + socks

-- Calculate binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Calculation of total ways to choose 5 articles from 21
def total_ways := binom total_articles articles_to_remove

-- Calculation of favorable outcomes
def favorable_ways :=
  (binom shirts 2) * (binom shorts 2) * (binom socks 1)

-- Calculate the probability
def probability :=
  (favorable_ways : ℚ) / total_ways

-- Prove that the probability is 280 / 2261
theorem probability_of_selecting_specified_clothing :
  probability = 280 / 2261 := by
  sorry

end probability_of_selecting_specified_clothing_l29_29146


namespace third_smallest_four_digit_in_pascals_triangle_l29_29295

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29295


namespace smallest_candies_divisible_l29_29437

theorem smallest_candies_divisible :
  ∃ n, (∀ k ∈ {4, 5, 6, 7, 8}, k ∣ n) ∧ n = 840 :=
by
  existsi 840
  split
  { intros k hk
    fin_cases hk
    case left
    { show 4 ∣ 840, from dvd.intro (840 / 4) rfl }
    case left_1
    { show 5 ∣ 840, from dvd.intro (840 / 5) rfl }
    case left_2
    { show 6 ∣ 840, from dvd.intro (840 / 6) rfl }
    case left_3
    { show 7 ∣ 840, from dvd.intro (840 / 7) rfl }
    case right
    { show 8 ∣ 840, from dvd.intro (840 / 8) rfl } }
  { reflexivity }

end smallest_candies_divisible_l29_29437


namespace al_and_barb_common_rest_days_l29_29470

noncomputable def al_schedule : ℕ → bool
| t => (t % 4) = 2 ∨ (t % 4) = 3 -- which means rest on days 3-4 in 4-day cycle

noncomputable def barb_schedule : ℕ → bool
| t => (t % 10) > 4 -- which means rest on days 6-10 in 10-day cycle

theorem al_and_barb_common_rest_days :
  (List.range 1000).count (λ (t : ℕ) => al_schedule t ∧ barb_schedule t) = 250 := by
  sorry

end al_and_barb_common_rest_days_l29_29470


namespace sum_series_eq_3_over_4_l29_29752

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29752


namespace total_students_l29_29919

-- Define the conditions
def ratio_girls_boys (G B : ℕ) : Prop := G / B = 1 / 2
def ratio_math_girls (M N : ℕ) : Prop := M / N = 3 / 1
def ratio_sports_boys (S T : ℕ) : Prop := S / T = 4 / 1

-- Define the problem statement
theorem total_students (G B M N S T : ℕ) 
  (h1 : ratio_girls_boys G B)
  (h2 : ratio_math_girls M N)
  (h3 : ratio_sports_boys S T)
  (h4 : M = 12)
  (h5 : G = M + N)
  (h6 : G = 16) 
  (h7 : B = 32) : 
  G + B = 48 :=
sorry

end total_students_l29_29919


namespace sum_of_first_six_terms_of_geom_seq_l29_29837

theorem sum_of_first_six_terms_of_geom_seq :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let S6 := a * (1 - r^6) / (1 - r)
  S6 = 4095 / 12288 := by
sorry

end sum_of_first_six_terms_of_geom_seq_l29_29837


namespace b_seq_geometric_m_range_l29_29855

-- Definition of sequence {a_m}
def a_seq : ℕ → ℚ
| 0       := 3 / 2
| (n + 1) := 3 * a_seq n - 1

-- Definition of sequence {b_m}
def b_seq (m : ℕ) : ℚ := a_seq m - 1 / 2

-- Proof for question I
theorem b_seq_geometric :
  ∀ m, b_seq (m + 1) = 3 * b_seq m := 
by sorry

-- Proof for question II
theorem m_range (m : ℝ) :
  (∀ n : ℕ, 1 ≤ n → (b_seq n + 1) / (b_seq (n + 1) - 1) ≤ m) ↔ 
  1 ≤ m := 
by sorry

end b_seq_geometric_m_range_l29_29855


namespace sum_series_div_3_powers_l29_29779

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29779


namespace series_result_l29_29633

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29633


namespace spadesuit_evaluation_l29_29507

def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_evaluation : (spadesuit 5 (spadesuit 3 10)) * (spadesuit 2 4) = 4 :=
by
  have h1 : spadesuit 3 10 = |3 - 10| := rfl
  have h2 : spadesuit 3 10 = 7, by simp [h1]; apply abs_of_neg; norm_num
  have h3 : spadesuit 5 7 = |5 - 7| := rfl
  have h4 : spadesuit 5 7 = 2, by simp [h3]; apply abs_of_neg; norm_num
  have h5 : spadesuit 2 4 = |2 - 4| := rfl
  have h6 : spadesuit 2 4 = 2, by simp [h5]; apply abs_of_neg; norm_num
  calc
    (spadesuit 5 (spadesuit 3 10)) * (spadesuit 2 4) = spadesuit 5 7 * spadesuit 2 4 : by rw h2
    ... = 2 * 2 : by rw [h4, h6]
    ... = 4 : by norm_num
sorry

end spadesuit_evaluation_l29_29507


namespace abs_eq_zero_iff_l29_29128

theorem abs_eq_zero_iff {a : ℝ} (h : |a + 3| = 0) : a = -3 :=
sorry

end abs_eq_zero_iff_l29_29128


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29656

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29656


namespace integral_value_eq_3_l29_29845

theorem integral_value_eq_3 (a : ℝ) (h : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3) : a = 3 :=
sorry

end integral_value_eq_3_l29_29845


namespace series_result_l29_29628

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29628


namespace triangle_congruence_condition_l29_29206

theorem triangle_congruence_condition 
  (A B C M : Type) 
  [IsTriangle A B C]
  [IsMidpoint M A B]
  (angle_A : ∠A = 30°) 
  (angle_B : ∠B = 60°) 
  (angle_C : ∠C = 90°)
  (side_AM_eq : ∀ A B C, M eq A B -> side(A,M) == side(B,C)) 
  : (triangle ABC = triangle AMC) ∨ (triangle ABC ≠ triangle AMC) :=
by sorry

end triangle_congruence_condition_l29_29206


namespace four_digit_number_count_divisible_2_3_5_7_11_l29_29473

theorem four_digit_number_count_divisible_2_3_5_7_11 :
  (finset.filter (λ n : ℕ, (1000 <= n) ∧ (n <= 9999) ∧ (n % (2 * 3 * 5 * 7 * 11) = 0)) (finset.range 10000)).card = 4 :=
by
  -- These steps are provided to verify the logic, you can replace them with 'sorry'.
  -- The logic includes:
  -- 1. Definition of the range of 4-digit numbers: [1000, 9999].
  -- 2. Calculation of LCM(2, 3, 5, 7, 11).
  -- 3. Filtering numbers within the range that are divisible by the LCM.
  -- 4. Ensure the count is correct.
  sorry

end four_digit_number_count_divisible_2_3_5_7_11_l29_29473


namespace hyperbola_eccentricity_l29_29500

noncomputable def hyperbola (a b x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity (a b : ℝ) (h_a_gt_0 : a > 0) (h_b_gt_0 : b > 0) 
  (h_asymptotes_perpendicular : a = b) : 
  let e := Real.sqrt (1 + (b^2 / a^2)) in
  e = Real.sqrt 2 :=
by
  -- Sorry is used to skip the proof
  sorry


end hyperbola_eccentricity_l29_29500


namespace third_smallest_four_digit_in_pascal_l29_29341

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29341


namespace betty_slippers_count_l29_29482

theorem betty_slippers_count :
  ∃ (x : ℕ), x + 4 + 8 = 18 ∧ 2.5 * x + 1.25 * 4 + 3 * 8 = 44 ∧ x = 6 := by
  sorry

end betty_slippers_count_l29_29482


namespace exists_x_l29_29115

-- Define the universal set U and set A
def U(x : ℝ) : Set ℝ := {1, 3, x^2 - 2 * x}
def A(x : ℝ) : Set ℝ := {1, |2 * x - 1|}

-- Given that complement of A in U is exactly {0}, determine the value of x
theorem exists_x (x : ℝ) (h : U(x) \ A(x) = {0}) : x = 2 :=
sorry

end exists_x_l29_29115


namespace sum_geometric_series_l29_29693

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29693


namespace ranking_arrangements_l29_29148

def students : List String := ["A", "B", "C", "D", "E"]

def not_first (s : String) : Prop := s ≠ "A" ∧ s ≠ "B"
def not_last (s : String) : Prop := s ≠ "B"

-- Define a function that counts valid rankings
noncomputable def countValidRankings : Nat :=
  let all_positions := students.permutations
  let valid_positions := all_positions.filter (λ ranking =>
    not_first (ranking.head!) ∧ 
    not_last (ranking.getLast! sorry))
  valid_positions.length

theorem ranking_arrangements : countValidRankings = 36 :=
  sorry

end ranking_arrangements_l29_29148


namespace third_smallest_four_digit_Pascal_triangle_l29_29290

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29290


namespace sum_first_12_terms_geom_seq_l29_29149

def geometric_sequence_periodic (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) * a (n + 2) = k

theorem sum_first_12_terms_geom_seq :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 2 ∧
    a 3 = 4 ∧
    geometric_sequence_periodic a 8 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_first_12_terms_geom_seq_l29_29149


namespace first_even_number_l29_29137

theorem first_even_number (x : ℤ) (h : x + (x + 2) + (x + 4) = 1194) : x = 396 :=
by
  -- the proof is skipped as per instructions
  sorry

end first_even_number_l29_29137


namespace series_converges_to_three_fourths_l29_29810

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29810


namespace series_converges_to_three_fourths_l29_29798

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29798


namespace percentage_increase_correct_l29_29230

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost : ℝ := 624
noncomputable def rate : ℝ := 4

def area (cost rate : ℝ) : ℝ := cost / rate
noncomputable def breadth (area length : ℝ) : ℝ := area / length
noncomputable def percentage_increase (length breadth : ℝ) : ℝ := ((length - breadth) / breadth) * 100

theorem percentage_increase_correct :
  percentage_increase length (breadth (area cost rate) length) = 200.1 := 
by 
  sorry

end percentage_increase_correct_l29_29230


namespace no_incorrect_value_in_quadratic_sequence_l29_29054

theorem no_incorrect_value_in_quadratic_sequence :
  ∀ (a b c : ℤ), ∀ (f : ℕ → ℤ),
  (∀ n : ℕ, f n = a * n^2 + b * n + c) →
  let seq := [6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569]
  in (∀ i : ℕ, i < 7 → (seq[i+1] - seq[i]) - (seq[i] - seq[i-1]) = 2) →
  true := -- Indicates no errors in the sequence, hence "none of these" is correct.
by sorry

end no_incorrect_value_in_quadratic_sequence_l29_29054


namespace find_integers_satisfying_equation_l29_29026

theorem find_integers_satisfying_equation :
  ∃ (a b c : ℤ), (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = 1) ∨
                  (a = 2 ∧ b = -1 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = -1 ∧ c = 2)
  ↔ (∃ (a b c : ℤ), 1 / 2 * (a + b) * (b + c) * (c + a) + (a + b + c) ^ 3 = 1 - a * b * c) := sorry

end find_integers_satisfying_equation_l29_29026


namespace highest_average_is_105_l29_29418

def avg (a b : ℕ) : ℕ := (a + b) / 2

def multiples_of (n upper : ℕ) : List ℕ := List.filter (λ x => x % n == 0) (List.range (upper + 1))

def max_avg (candidates : List (List ℕ)) : ℕ :=
  List.maximumBy (λ lst => avg (List.head! lst) (List.head! (List.reverse lst))) candidates |> 
    λ opt => match opt with
    | some lst => avg (List.head! lst) (List.head! (List.reverse lst))
    | none => 0

theorem highest_average_is_105 : 
  let multiples_7  := multiples_of 7 201
  let multiples_8  := multiples_of 8 201
  let multiples_9  := multiples_of 9 201
  let multiples_10 := multiples_of 10 201
  let multiples_11 := multiples_of 11 201
  max_avg [multiples_7, multiples_8, multiples_9, multiples_10, multiples_11] = 105 := by
  sorry

end highest_average_is_105_l29_29418


namespace pascal_third_smallest_four_digit_number_l29_29392

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29392


namespace series_sum_eq_l29_29539

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29539


namespace infinite_series_sum_eq_3_over_4_l29_29718

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29718


namespace product_of_two_numbers_l29_29247

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 48) : x * y = 7 := 
by 
sor

end product_of_two_numbers_l29_29247


namespace third_smallest_four_digit_in_pascals_triangle_l29_29358

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29358


namespace chi_squared_relation_l29_29117

noncomputable def chi_squared_test (observed_chi_squared : ℝ) (critical_value : ℝ) : Prop :=
  observed_chi_squared > critical_value

theorem chi_squared_relation (k : ℝ) (k_0 : ℝ) (P : ℝ) (h1 : P = 0.05) (h2 : k = 4.328) (h3 : k_0 = 3.841) :
  chi_squared_test k k_0 → "95% confidence that variables X and Y are related" :=
by
  sorry

end chi_squared_relation_l29_29117


namespace series_converges_to_three_fourths_l29_29797

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29797


namespace cos_double_angle_l29_29867

theorem cos_double_angle (θ : ℝ) (h : sin (θ / 2) - cos (θ / 2) = sqrt 6 / 3) : cos (2 * θ) = 7 / 9 := sorry

end cos_double_angle_l29_29867


namespace wig_cost_is_correct_l29_29946

def wig_cost_problem 
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (sold_price : ℕ)
  (total_spent : ℕ)
  (number_dropped_plays : ℕ)
  : ℝ :=
let total_wigs := plays * acts_per_play * wigs_per_act in
let remaining_wigs := total_wigs - (acts_per_play * wigs_per_act * number_dropped_plays) in
total_spent / remaining_wigs

theorem wig_cost_is_correct 
  (plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (sold_price : ℕ)
  (total_spent : ℕ)
  (number_dropped_plays : ℕ)
  (h_plays: plays = 3)
  (h_acts_per_play: acts_per_play = 5)
  (h_wigs_per_act: wigs_per_act = 2)
  (h_sold_price: sold_price = 4)
  (h_total_spent: total_spent = 110)
  (h_number_dropped_plays: number_dropped_plays = 1) : 
  wig_cost_problem plays acts_per_play wigs_per_act sold_price total_spent number_dropped_plays = 5.50 := 
by {
  sorry
}

end wig_cost_is_correct_l29_29946


namespace cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l29_29066

theorem cos_eq_neg_1_over_4_of_sin_eq_1_over_4
  (α : ℝ)
  (h : Real.sin (α + π / 3) = 1 / 4) :
  Real.cos (α + 5 * π / 6) = -1 / 4 :=
sorry

end cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l29_29066


namespace father_son_age_ratio_l29_29445

theorem father_son_age_ratio :
  ∃ S : ℕ, (45 = S + 15 * 2) ∧ (45 / S = 3) := 
sorry

end father_son_age_ratio_l29_29445


namespace radius_of_inscribed_sphere_l29_29225

theorem radius_of_inscribed_sphere : 
  ∃ (r : ℝ), r = (sqrt 3 - 1) / 4 :=
sorry

end radius_of_inscribed_sphere_l29_29225


namespace infinite_series_sum_eq_3_over_4_l29_29716

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29716


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29644

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29644


namespace sum_of_series_eq_three_fourths_l29_29577

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29577


namespace third_smallest_four_digit_in_pascals_triangle_l29_29323

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29323


namespace third_smallest_four_digit_in_pascals_triangle_l29_29377

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29377


namespace median_salary_is_25000_l29_29468

structure PositionSalary (p : Type) (s : ℝ) :=
  (count : ℕ)

def companyPositions : List (PositionSalary String) :=
  [ PositionSalary.mk "President" 1
  , PositionSalary.mk "Vice-President" 4
  , PositionSalary.mk "Director" 15
  , PositionSalary.mk "Associate Director" 8
  , PositionSalary.mk "Administrative Specialist" 30
  , PositionSalary.mk "Customer Service Representative" 12
  ]

def companySalaries : List ℝ :=
  [].join (companyPositions.map (λ ps, List.replicate ps.count ps.s))

def median (l : List ℝ) : ℝ :=
let sorted := l.sort (· ≤ ·)
in (sorted.get! ((l.length / 2) - 1) + sorted.get! (l.length / 2)) / 2

theorem median_salary_is_25000 :
  median companySalaries = 25000 :=
by
  -- proof would be here
  sorry

end median_salary_is_25000_l29_29468


namespace M_is_correct_l29_29081

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x > 2}

def M := {x | x ∈ A ∧ x ∉ B}

theorem M_is_correct : M = {1, 2} := by
  -- Proof needed here
  sorry

end M_is_correct_l29_29081


namespace total_rolls_sold_is_correct_l29_29447

/-- Given conditions for the problem -/
variables (solid_price : ℝ) (print_price : ℝ)
variables (total_amount : ℝ) (print_rolls : ℕ)
variables (solid_rolls : ℕ) (total_rolls : ℕ)

/-- Given specific values -/
def gift_wrap_problem_conditions :=
  solid_price = 4.0 ∧
  print_price = 6.0 ∧
  total_amount = 2340.0 ∧
  print_rolls = 210 ∧
  total_rolls = solid_rolls + print_rolls

/-- The proof goal -/
theorem total_rolls_sold_is_correct : gift_wrap_problem_conditions solid_price print_price total_amount print_rolls solid_rolls total_rolls → total_rolls = 480 :=
by
  sorry

end total_rolls_sold_is_correct_l29_29447


namespace limit_g_minus_f_at_pi_limit_g_prime_minus_f_prime_at_pi_l29_29990

open Real

noncomputable def f (x : ℝ) : ℝ := 0
noncomputable def g (n : ℕ) (x : ℝ) : ℝ :=
  (1 / n) * sin (π / 6 + 2 * n^2 * x)

theorem limit_g_minus_f_at_pi (x : ℝ) :
  IsLimit (fun n => abs (g n π - f π)) 0 atTop :=
sorry

theorem limit_g_prime_minus_f_prime_at_pi (x : ℝ) :
  IsLimit (fun n => abs (deriv (g n) π - deriv f π)) atTop atTop :=
sorry

end limit_g_minus_f_at_pi_limit_g_prime_minus_f_prime_at_pi_l29_29990


namespace num_arith_seq_sets_with_odds_l29_29490

theorem num_arith_seq_sets_with_odds : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let is_arith_seq (a b c : Nat) := (b - a) = (c - b)
  let is_valid_set (a b c : Nat) := a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                                    is_arith_seq a b c ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1)
  ∃ n, (n = 15 ∧ (∀ (a b c : Nat), is_valid_set a b c → ∃ (sets : Finset (Finset ℕ)), sets.card = n )) :=
begin
  sorry
end

end num_arith_seq_sets_with_odds_l29_29490


namespace min_ϕ_l29_29885

noncomputable def f (x ϕ : ℝ) := 2 * Real.sin (2 * x + ϕ)

def is_even (g : ℝ → ℝ) := ∀ x : ℝ, g x = g (-x)

theorem min_ϕ (ϕ : ℝ) (hϕ : ϕ > 0) 
                (h_shift : ∀ x : ℝ, is_even (λ x, f (x + Real.pi / 5) ϕ)) :
   ϕ = Real.pi / 10 := 
sorry

end min_ϕ_l29_29885


namespace infinite_series_sum_value_l29_29731

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29731


namespace evaluate_series_sum_l29_29598

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29598


namespace generating_polynomials_characterization_l29_29454

open Polynomial

-- Define what it means for a polynomial to be generating
def is_generating (f : Polynomial ℝ) : Prop :=
  ∀ (φ : Polynomial ℝ), ∃ (k : ℕ) (g : Fin k → Polynomial ℝ), φ = ∑ i, f.eval (g i)

-- The main theorem stating the condition for a polynomial to be generating
theorem generating_polynomials_characterization (f : Polynomial ℝ) :
  is_generating f ↔ ∃ (d : ℕ), odd d ∧ d = nat_degree f :=
sorry

end generating_polynomials_characterization_l29_29454


namespace series_sum_eq_l29_29537

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29537


namespace third_smallest_four_digit_in_pascals_triangle_l29_29359

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29359


namespace sum_series_eq_3_div_4_l29_29568

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29568


namespace pascal_third_smallest_four_digit_number_l29_29399

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29399


namespace sum_of_series_eq_three_fourths_l29_29575

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29575


namespace trajectory_parabola_max_area_triangle_ABC_l29_29074

-- Part (Ⅰ): The equation of the trajectory for the center E

theorem trajectory_parabola (E F : ℝ × ℝ) (l : ℝ → Prop)
  (hF : F = (1, 0)) (hl : ∀ x : ℝ, l x ↔ x = -1)
  (hE : (∃ y : ℝ, E = (y^2 / 4, y))) :
  ∃ x : ℝ, E = (x, 0) ∧ x = 1 :=
begin
  sorry
end

-- Part (Ⅱ): Maximum value of the area of △ABC
theorem max_area_triangle_ABC (A B C : ℝ × ℝ) (G : ℝ → ℝ → Prop)
  (hA : A = (3, 0)) (slope1 : slope_of_line = 1) (not_passing : ∀ P : ℝ × ℝ, P ≠ (0, 0) ∧ P ≠ A) 
  (curve_G : ∀ x y : ℝ, G x y ↔ y^2 = 4 * x) :
  ∃ S_max : ℝ, S_max = (32 * real.sqrt 3) / 9 :=
begin
  sorry
end

end trajectory_parabola_max_area_triangle_ABC_l29_29074


namespace infinite_series_sum_eq_l29_29701

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29701


namespace hyperbola_eccentricity_l29_29887

-- Define the problem conditions and the goal in Lean 4 statement
theorem hyperbola_eccentricity (a b x1 y1 x2 y2 : ℝ) (ha : a > 0) (hb : b > 0)
  (hM : x1^2 / a^2 - y1^2 / b^2 = 1) (hP : x2^2 / a^2 - y2^2 / b^2 = 1)
  (hk : (y2 - y1)/(x2 - x1) * (y2 + y1)/(x2 + x1) = 5/4) : 
  (Real.sqrt (1 + (b/a)^2) = 3/2) :=
begin
  sorry
end

end hyperbola_eccentricity_l29_29887


namespace sum_series_div_3_powers_l29_29792

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29792


namespace sum_of_first_1000_terms_l29_29007

def sequence_block_sum (n : ℕ) : ℕ :=
  1 + 3 * n

def sequence_sum_up_to (k : ℕ) : ℕ :=
  if k = 0 then 0 else (1 + 3 * (k * (k - 1) / 2)) + k

def nth_term_position (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + n

theorem sum_of_first_1000_terms : sequence_sum_up_to 43 + (1000 - nth_term_position 43) * 3 = 2912 :=
sorry

end sum_of_first_1000_terms_l29_29007


namespace infinite_series_sum_eq_l29_29695

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29695


namespace infinite_series_sum_eq_3_div_4_l29_29624

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29624


namespace series_result_l29_29640

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29640


namespace maximal_coverage_of_checkerboard_l29_29440

theorem maximal_coverage_of_checkerboard (w h : ℕ) (card_w card_h : ℕ) (placement_horizontally : bool) :
  (w = 1 ∧ h = 1) ∧ (card_w = 2 ∧ card_h = 1) ∧ placement_horizontally = tt → 
  2 ≤ card_w * card_h ∧ 2 <= 2 := 
sorry

end maximal_coverage_of_checkerboard_l29_29440


namespace remainder_eq_159_l29_29018

def x : ℕ := 2^40
def numerator : ℕ := 2^160 + 160
def denominator : ℕ := 2^80 + 2^40 + 1

theorem remainder_eq_159 : (numerator % denominator) = 159 := 
by {
  -- Proof will be filled in here.
  sorry
}

end remainder_eq_159_l29_29018


namespace product_is_real_l29_29823

noncomputable def solve_quadratic : List ℝ :=
  let x1 := (-9 + Real.sqrt 17) / 4
  let x2 := (-9 - Real.sqrt 17) / 4
  [x1, x2]

theorem product_is_real (x : ℝ) :
  (∃ x ∈ solve_quadratic, (x + 2 * Complex.i) * ((x + 3) + 2 * Complex.i) * ((x + 4) + 2 * Complex.i)).im = 0 :=
by
  sorry

end product_is_real_l29_29823


namespace limit_sequence_l29_29487

open Real

theorem limit_sequence :
  tendsto (λ n : ℕ, sin (sqrt (↑n^2 + 1)) * arctan (↑n / (↑n^2 + 1))) at_top (𝓝 0) :=
begin
  sorry
end

end limit_sequence_l29_29487


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29272

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29272


namespace unique_a_inequality_holds_l29_29072

noncomputable def f (x a : ℝ) := Real.log (x + a) - x

def a_unique_root_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, f x a = 0 ↔ x = 1 - a

theorem unique_a (h : ∃! x, f x 1 = 0) : ∃! a, a = 1 :=
begin
  sorry
end

noncomputable def h (x : ℝ) := f x 1 + x

theorem inequality_holds (x_1 x_2 : ℝ) (hx1 : x_1 > -1) (hx2 : x_2 > -1) (hx1_ne : x_1 ≠ x_2) :
  (x_1 - x_2) / (h x_1 - h x_2) > Real.sqrt (x_1 * x_2 + x_1 + x_2 + 1) :=
begin
  sorry
end

end unique_a_inequality_holds_l29_29072


namespace weighted_averages_correct_l29_29921

def group_A_boys : ℕ := 20
def group_B_boys : ℕ := 25
def group_C_boys : ℕ := 15

def group_A_weight : ℝ := 50.25
def group_B_weight : ℝ := 45.15
def group_C_weight : ℝ := 55.20

def group_A_height : ℝ := 160
def group_B_height : ℝ := 150
def group_C_height : ℝ := 165

def group_A_age : ℝ := 15
def group_B_age : ℝ := 14
def group_C_age : ℝ := 16

def group_A_athletic : ℝ := 0.60
def group_B_athletic : ℝ := 0.40
def group_C_athletic : ℝ := 0.75

noncomputable def total_boys : ℕ := group_A_boys + group_B_boys + group_C_boys

noncomputable def weighted_average_height : ℝ := 
    (group_A_boys * group_A_height + group_B_boys * group_B_height + group_C_boys * group_C_height) / total_boys

noncomputable def weighted_average_weight : ℝ := 
    (group_A_boys * group_A_weight + group_B_boys * group_B_weight + group_C_boys * group_C_weight) / total_boys

noncomputable def weighted_average_age : ℝ := 
    (group_A_boys * group_A_age + group_B_boys * group_B_age + group_C_boys * group_C_age) / total_boys

noncomputable def weighted_average_athletic : ℝ := 
    (group_A_boys * group_A_athletic + group_B_boys * group_B_athletic + group_C_boys * group_C_athletic) / total_boys

theorem weighted_averages_correct :
  weighted_average_height = 157.08 ∧
  weighted_average_weight = 49.36 ∧
  weighted_average_age = 14.83 ∧
  weighted_average_athletic = 0.5542 := 
  by
    sorry

end weighted_averages_correct_l29_29921


namespace third_smallest_four_digit_in_pascal_triangle_l29_29368

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29368


namespace sum_of_series_eq_three_fourths_l29_29590

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29590


namespace find_k_l29_29972

theorem find_k 
  (k : ℝ) 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 3 * x + 5)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 7) 
  (intersection : ∃ x y : ℝ, (y = 3 * x + 5) ∧ (y = k * x - 7) ∧ x = -4 ∧ y = -7) :
  k = 0 :=
by
  sorry

end find_k_l29_29972


namespace third_smallest_four_digit_in_pascals_triangle_l29_29302

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29302


namespace number_of_irrational_numbers_is_two_l29_29475

def is_irrational (n : ℝ) : Prop :=
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b

def given_numbers : List ℝ :=
  [1 / 7, -Real.pi, 0.314, Real.sqrt 2, -Real.sqrt 64, 5]

theorem number_of_irrational_numbers_is_two :
  (given_numbers.filter is_irrational).length = 2 :=
by
  sorry

end number_of_irrational_numbers_is_two_l29_29475


namespace counting_pairs_no_carry_l29_29059

def no_carry (n : ℕ) : Prop :=
  ∀ i, i < 3 → ((n % 10^i) < 9) 

theorem counting_pairs_no_carry :
  ∃ (count : ℕ), count = 891 ∧ 
  count = ∑ n in Finset.range (2500 + 1 - 1500), if no_carry (1500 + n) then 1 else 0 := by
sorry

end counting_pairs_no_carry_l29_29059


namespace series_sum_eq_l29_29525

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29525


namespace third_smallest_four_digit_in_pascals_triangle_l29_29354

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29354


namespace problem1_problem2_l29_29107

noncomputable theory

def ln (x : ℝ) : ℝ := Real.log x

def f (x : ℝ) : ℝ := ln x

def g (a b x : ℝ) : ℝ := (1/2) * a * x + b

def phi (m x : ℝ) : ℝ := (m * (x - 1)) / (x + 1) - ln x

theorem problem1 (a b : ℝ) 
  (h1 : ∀ x, f x = ln x)
  (h2 : ∀ x, g a b x = (1/2) * a * x + b)
  (h3 : f 1 = g a b 1)
  (h4 : (Real.deriv f) 1 = (Real.deriv (g a b)) 1) :
  g a b = λ x, x - 1 := sorry

theorem problem2 (m : ℝ) 
  (h1 : ∀ x, phi m x = (m * (x - 1)) / (x + 1) - ln x)
  (h2 : ∀ x, 1 ≤ x → (Real.deriv (phi m)) x ≤ 0) :
  m ≤ 2 := sorry

end problem1_problem2_l29_29107


namespace infinite_series_sum_eq_l29_29699

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29699


namespace percent_fair_hair_l29_29433

theorem percent_fair_hair (total_employees : ℕ) (total_women_fair_hair : ℕ)
  (percent_fair_haired_women : ℕ) (percent_women_fair_hair : ℕ)
  (h1 : total_women_fair_hair = (total_employees * percent_women_fair_hair) / 100)
  (h2 : percent_fair_haired_women * total_women_fair_hair = total_employees * 10) :
  (25 * total_employees = 100 * total_women_fair_hair) :=
by {
  sorry
}

end percent_fair_hair_l29_29433


namespace expression_bounds_l29_29174

noncomputable def expression (x y z w : ℝ) : ℝ :=
  sqrt (x^2 + sin(Real.pi * y)^2) + 
  sqrt (y^2 + sin(Real.pi * z)^2) + 
  sqrt (z^2 + sin(Real.pi * w)^2) + 
  sqrt (w^2 + sin(Real.pi * x)^2)

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x) (hx_bound : x ≤ 1)
  (hy : 0 ≤ y) (hy_bound : y ≤ 1)
  (hz : 0 ≤ z) (hz_bound : z ≤ 1)
  (hw : 0 ≤ w) (hw_bound : w ≤ 1) :
  2 * Real.sqrt 2 ≤ expression x y z w ∧ expression x y z w ≤ 4 :=
sorry

end expression_bounds_l29_29174


namespace coefficient_of_monomial_is_minus_five_l29_29226

theorem coefficient_of_monomial_is_minus_five (m n : ℤ) : ∃ c : ℤ, -5 * m * n^3 = c * (m * n^3) ∧ c = -5 :=
by
  use -5
  split
  sorry

end coefficient_of_monomial_is_minus_five_l29_29226


namespace original_problem_modified_problem_l29_29444

-- Definitions based on the conditions given
variable (A B C D E F G H : Point) -- Designate points of the hexagon
variable (AB EF HG DC BC FG AD EH : ℝ) -- Lengths of the sides and diagonals

-- Conditions
variable (isParallel_AB_EF : parallel AB EF)
variable (isParallel_HG_DC : parallel HG DC)
variable (isParallel_BC_FG : parallel BC FG)
variable (isParallel_AD_EH : parallel AD EH)

-- Proof that needs to be shown: original problem
theorem original_problem 
  (AG BH CE DF AE BF CG DH : ℝ) -- Diagonals
  (AG_squared : AG^2)
  (BH_squared : BH^2)
  (CE_squared : CE^2)
  (DF_squared : DF^2) :
  AG^2 + BH^2 + CE^2 + DF^2 = AE^2 + BF^2 + CG^2 + DH^2 + 2 * (AB * HG + DC * EF + BC * FG + AD * EH) := 
sorry

-- Proof that needs to be shown: modified problem if any two body diagonals intersect
theorem modified_problem 
  (AC EG BD FH : ℝ) -- New diagonals after intersection
  (AG BH CE DF AE BF CG DH : ℝ)
  (AG_squared : AG^2)
  (BH_squared : BH^2)
  (CE_squared : CE^2)
  (DF_squared : DF^2) :
  AG^2 + BH^2 + CE^2 + DF^2 = AE^2 + BF^2 + CG^2 + DH^2 + 2 * (AC * EG + BD * FH) :=
sorry

end original_problem_modified_problem_l29_29444


namespace infinite_series_sum_value_l29_29743

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29743


namespace proof_solution_l29_29902

noncomputable def proof_problem (x y : ℝ) : Prop :=
  abs x + x + y = 10 ∧ x + abs y - y = 12 → x + y = 18 / 5

theorem proof_solution (x y : ℝ) (h : proof_problem x y) : x + y = 18 / 5 :=
by
  cases h with h1 h2
  sorry

end proof_solution_l29_29902


namespace third_smallest_four_digit_in_pascals_triangle_l29_29299

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29299


namespace number_of_arrangements_proof_l29_29257

-- Definition for the number of different arrangements of balls
noncomputable def number_of_arrangements (red_balls black_balls drawn_balls: ℕ) := choose drawn_balls black_balls

-- Constants representing the problem's conditions
def total_red_balls := 15
def total_black_balls := 10
def total_drawn_balls := 10
def red_balls_drawn := 6
def black_balls_drawn := 4

-- Hypothesis based on the problem's condition
axiom condition1 : total_red_balls = 15
axiom condition2 : total_black_balls = 10
axiom condition3 : total_drawn_balls = 10
axiom condition4 : red_balls_drawn + black_balls_drawn = total_drawn_balls

-- The theorem to be proved
theorem number_of_arrangements_proof :
  number_of_arrangements red_balls_drawn black_balls_drawn total_drawn_balls = choose 10 4 :=
sorry

end number_of_arrangements_proof_l29_29257


namespace midpoint_X_ST_l29_29960

variables {A B C D O X S T : Type*} [Circle O]
variables [PointsOnCircle A B C D O] [Intersection X A C B D] [LineThroughAndPerpendicular ℓ X O]
variables [IntersectionAt S ℓ A B] [IntersectionAt T ℓ C D]

theorem midpoint_X_ST
  (hc : Cocyclic A B C D)
  (hx : X = Intersection (diagonal A C) (diagonal B D))
  (hl : ℓ = LineThroughAndPerpendicular X O)
  (hs : S = Intersection ℓ (side A B))
  (ht : T = Intersection ℓ (side C D)) :
  Midpoint X S T :=
sorry

end midpoint_X_ST_l29_29960


namespace sum_of_series_eq_three_fourths_l29_29586

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29586


namespace other_root_of_quadratic_l29_29202

variable (p : ℝ)

theorem other_root_of_quadratic (h1: 3 * (-2) * r_2 = -6) : r_2 = 1 :=
by
  sorry

end other_root_of_quadratic_l29_29202


namespace triangle_AC_length_l29_29141

def triangle_angle_sum (A B C : ℕ) : Prop := A + B + C = 180

theorem triangle_AC_length (A B C : ℕ) (BC AC : ℝ) 
  (A_eq : A = 45) (C_eq : C = 105) (BC_eq : BC = real.sqrt 2)
  (angle_sum : triangle_angle_sum A B C)
  (sine_rule : BC / real.sin (real.pi * A / 180) = AC / real.sin (real.pi * B / 180)) :
  AC = 1 := by
  sorry

end triangle_AC_length_l29_29141


namespace series_sum_eq_l29_29538

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29538


namespace digits_in_s_200_l29_29423

noncomputable def s (n : ℕ) : ℕ := 
  String.toNat (String.join (List.map (fun k => toString (k * k)) (List.range (n + 1)))).length

theorem digits_in_s_200 : s 200 = 492 := 
by
  sorry

end digits_in_s_200_l29_29423


namespace sum_series_eq_3_over_4_l29_29745

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29745


namespace third_smallest_four_digit_in_pascals_triangle_l29_29363

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29363


namespace series_sum_correct_l29_29765

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29765


namespace sum_series_eq_l29_29499

theorem sum_series_eq : 
  (∑ n in (finset.range (n+3)).filter (λ n, n ≥ 1), 1 / (n * (n + 3))) = 11/18 := 
begin
  sorry
end

end sum_series_eq_l29_29499


namespace sum_series_equals_three_fourths_l29_29547

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29547


namespace third_smallest_four_digit_in_pascals_triangle_l29_29383

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29383


namespace sum_series_eq_3_over_4_l29_29756

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29756


namespace problem_eq_answer_l29_29417

def largest_number (a b c d e : ℝ) : ℝ :=
  if e > a ∧ e > b ∧ e > c ∧ e > d then e else sorry

theorem problem_eq_answer : 
  largest_number 0.997 0.9969 0.99699 0.9699 0.999 = 0.999 := 
by 
  sorry

end problem_eq_answer_l29_29417


namespace ax_by_n_sum_l29_29909

theorem ax_by_n_sum {a b x y : ℝ} 
  (h1 : a * x + b * y = 2)
  (h2 : a * x^2 + b * y^2 = 5)
  (h3 : a * x^3 + b * y^3 = 15)
  (h4 : a * x^4 + b * y^4 = 35) :
  a * x^5 + b * y^5 = 10 :=
sorry

end ax_by_n_sum_l29_29909


namespace count_valid_permutations_equals_l29_29894

noncomputable theory
open_locale big_operators

-- Define the set S
def S : finset ℕ := finset.range (2000 - 1901 + 1) + 1901

-- Define the condition that a permutation does not have partial sums divisible by 3
def valid_permutation (π : list ℕ) : Prop :=
  ∀ n, n < π.length → (3 ∣ (π.take (n + 1)).sum) = false

-- Define the number of such valid permutations
noncomputable def count_valid_permutations : ℕ :=
  finset.card {π ∈ (S.val.permutations.to_finset) | valid_permutation π}

-- State the theorem to be proved
theorem count_valid_permutations_equals : count_valid_permutations = (99! * 34! * 33!) / 66! :=
  sorry

end count_valid_permutations_equals_l29_29894


namespace part_I_part_II_l29_29869

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2 * m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem part_I (m : ℝ) : (∀ x, x ∈ A ∧ x ∈ B → x ∈ C m) → m ≥ 5 / 2 :=
sorry

theorem part_II (m : ℝ) : 
  (B ∪ (C m) = Set.univ) ∧ 
  (C m ⊆ D) → 
  7 / 2 ≤ m ∧ m < 4 :=
sorry

end part_I_part_II_l29_29869


namespace repair_time_and_earnings_l29_29169

-- Definitions based on given conditions
def cars : ℕ := 10
def cars_repair_50min : ℕ := 6
def repair_time_50min : ℕ := 50 -- minutes per car
def longer_percentage : ℕ := 80 -- 80% longer for the remaining cars
def wage_per_hour : ℕ := 30 -- dollars per hour

-- Remaining cars to repair
def remaining_cars : ℕ := cars - cars_repair_50min

-- Calculate total repair time for each type of cars and total repair time
def repair_time_remaining_cars : ℕ := repair_time_50min + (repair_time_50min * longer_percentage) / 100
def total_repair_time : ℕ := (cars_repair_50min * repair_time_50min) + (remaining_cars * repair_time_remaining_cars)

-- Convert total repair time from minutes to hours
def total_repair_hours : ℕ := total_repair_time / 60

-- Calculate total earnings
def total_earnings : ℕ := wage_per_hour * total_repair_hours

-- The theorem to be proved: total_repair_time == 660 and total_earnings == 330
theorem repair_time_and_earnings :
  total_repair_time = 660 ∧ total_earnings = 330 := by
  sorry

end repair_time_and_earnings_l29_29169


namespace third_smallest_four_digit_in_pascals_triangle_l29_29294

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29294


namespace series_sum_eq_l29_29540

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29540


namespace pascal_third_smallest_four_digit_number_l29_29400

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29400


namespace max_edges_intersected_by_plane_l29_29501

theorem max_edges_intersected_by_plane {n : ℕ} (hn : n = 2015) 
  (prism : convex_prism_with_ngon_base n) : 
  ∃ N, N = 2017 ∧ (∀ plane : ℝ → affine_plane, ¬ (plane_passes_through_any_vertex plane prism) → max_edges_intersected plane prism N) := 
sorry

end max_edges_intersected_by_plane_l29_29501


namespace find_least_integer_l29_29039

theorem find_least_integer (x : ℤ) : (3 * |x| - 4 < 20) → (x ≥ -7) :=
by
  sorry

end find_least_integer_l29_29039


namespace problem1_problem2_problem3_l29_29086

-- Define sequence a_n
def a_n (n : ℕ) : ℕ := 2^n + 3^n

-- Define sequence b_n
def b_n (n k : ℕ) : ℕ := a_n (n + 1) + k * a_n n

-- Problem 1: Find value of k such that {b_n} is a geometric sequence
theorem problem1 : ∀ (k : ℕ), (∀ n : ℕ, ∃ r : ℕ, b_n (n + 1) k = r * b_n n k) → (k = -2 ∨ k = -3) :=
sorry

-- Problem 2: Prove that sum of first n terms of sequence C_n is S_n and given inequality
def C_n (n : ℕ) : ℕ := n -- Given from C_n = log_3(a_n - 2^n) = n
def S_n (n : ℕ) : ℕ := n * (n + 1) / 2 -- Given sum of first n terms of sequence C_n

theorem problem2 : ∀ n : ℕ, (∑ i in Finset.range (n + 1), (1 / (S_n i : ℝ))) < 2 :=
sorry

-- Problem 3: Given k=-2, find the sum of elements in set A
def d_n (n : ℕ) : ℝ := (2 * n - 1 : ℝ) / (3^n : ℝ)
def A : Finset ℕ := {1, 2, 3} -- Given that A = {1, 2, 3}

theorem problem3 : ∑ i in A, i = 6 :=
sorry

end problem1_problem2_problem3_l29_29086


namespace no_such_function_exists_l29_29518

-- Define the problem as a hypothesis
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f(f(x)) = x^2 - 2 := 
sorry

end no_such_function_exists_l29_29518


namespace sum_series_eq_3_over_4_l29_29751

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29751


namespace locus_of_P_is_ellipse_l29_29950

variables (B C O : Point)
variable {A : Point}
variable (circle : Circle O)
variable (H M N D P : Point)

-- Definitions for given geometry
noncomputable def is_fixed_on_circle (P : Point) : Prop := P ∈ circle ∧ P ≠ B ∧ P ≠ C
noncomputable def is_orthocenter (H : Point) : Prop := ∃ (α β γ : Line), α.is_altitude ∧ β.is_altitude ∧ γ.is_altitude ∧ α ∩ β ∩ γ = H
noncomputable def is_midpoint (M : Point) (X Y : Point) : Prop := dist X M = dist Y M
noncomputable def line (X Y : Point) : Set Point := {P | is_collinear X Y P}

-- Given conditions
axiom B_and_C_on_circle : ∀ {X}, X = B ∨ X = C → X ∈ circle
axiom A_on_circle : is_fixed_on_circle A
axiom H_is_orthocenter : is_orthocenter H
axiom M_is_midpoint_BC : is_midpoint M B C
axiom N_is_midpoint_AH : is_midpoint N A H
axiom AM_intersects_circle_at_D : ∃ D, D ∈ line A M ∧ D ∈ circle ∧ D ≠ A
axiom NM_and_OD_intersect_at_P : ∃ P, P ∈ line N M ∧ P ∈ line O D

-- Proof statement to find the locus of points P
theorem locus_of_P_is_ellipse (A_moves : ∀ A, is_fixed_on_circle A → ∃ P, NM_and_OD_intersect_at_P):
  ∀ {A P}, is_fixed_on_circle A → NM_and_OD_intersect_at_P →
  let R := circle.radius in
  dist M P + dist O P = R :=
sorry

end locus_of_P_is_ellipse_l29_29950


namespace sum_geometric_series_l29_29686

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29686


namespace min_distance_C3_midpoint_l29_29874

open Real

noncomputable def curve_C1 (t : ℝ) : ℝ × ℝ :=
  (-4 + cos t, 3 + sin t)

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (8 * cos θ, 3 * sin θ)

def curve_C1_standard (x y : ℝ) : Prop :=
  (x + 4)^2 + (y - 3)^2 = 1

def curve_C2_standard (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 9 = 1

def line_C3 (t : ℝ) : ℝ × ℝ :=
  (3 + 2 * t, -2 + t)

def midpoint_PQ (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance_point_line (M : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * M.1 + B * M.2 + C) / sqrt (A^2 + B^2)

theorem min_distance_C3_midpoint :
  ∀ θ : ℝ,
  let P := (-4, 4),
      Q := curve_C2 θ,
      M := midpoint_PQ P Q,
      A := 1,
      B := -2,
      C := -7
  in distance_point_line M A B C ≥ 0 ∧ distance_point_line M A B C = 8 * sqrt 5 / 5 :=
by
  sorry

end min_distance_C3_midpoint_l29_29874


namespace sum_series_equals_three_fourths_l29_29545

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29545


namespace volume_of_cube_l29_29242

theorem volume_of_cube (edge_sum : ℝ) (h_edge_sum : edge_sum = 48) : 
    let edge_length := edge_sum / 12 
    in edge_length ^ 3 = 64 := 
by
  -- condition
  have h1 : edge_sum = 48 := h_edge_sum
  -- definition of edge_length
  let edge_length := edge_sum / 12
  -- compute the edge_length
  have h2 : edge_length = 4 := by linarith
  -- compute the volume of the cube
  have volume := edge_length ^ 3
  -- provide proof that volume is 64
  have h3 : volume = 64 := by linarith; sorry
  exact h3

end volume_of_cube_l29_29242


namespace series_sum_correct_l29_29777

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29777


namespace range_of_a_l29_29010

-- Function definition for op
def op (x y : ℝ) : ℝ := x * (2 - y)

-- Predicate that checks the inequality for all t
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, (op (t - a) (t + a)) < 1

-- Prove that the range of a is (0, 2)
theorem range_of_a : 
  ∀ a : ℝ, inequality_holds_for_all_t a ↔ 0 < a ∧ a < 2 := 
by
  sorry

end range_of_a_l29_29010


namespace value_of_expression_l29_29050

theorem value_of_expression (x : ℝ) (h : 7 * x^2 - 2 * x - 4 = 4 * x + 11) : 
  (5 * x - 7)^2 = 11.63265306 := 
by 
  sorry

end value_of_expression_l29_29050


namespace identify_1000g_weight_l29_29251

-- Define the masses of the weights
def masses : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- The statement that needs to be proven
theorem identify_1000g_weight (masses : List ℕ) (h : masses = [1000, 1001, 1002, 1004, 1007]) :
  ∃ w, w ∈ masses ∧ w = 1000 ∧ by sorry :=
sorry

end identify_1000g_weight_l29_29251


namespace polar_coordinates_correct_l29_29970

-- Definition of the given complex number corresponding to point P
def complex_number_P : ℂ := ⟨-3, 3⟩

-- Definition of the polar coordinates of point P
def polar_coordinates_P (z : ℂ) : ℝ × ℝ :=
  let r := complex.norm z in
  let θ := complex.arg z in
  (r, θ)

-- The goal is to prove that the polar coordinates of point P are (3 * sqrt 2, 3 * pi / 4)
theorem polar_coordinates_correct :
  polar_coordinates_P complex_number_P = (3 * Real.sqrt 2, 3 * Real.pi / 4) :=
by
  sorry

end polar_coordinates_correct_l29_29970


namespace x_is_41_18_percent_less_than_y_l29_29139

theorem x_is_41_18_percent_less_than_y 
    {x y : ℝ}
    (h : y = 1.70 * x) :
    (y - x) / y * 100 ≈ 41.18 :=
by
  sorry

end x_is_41_18_percent_less_than_y_l29_29139


namespace sum_series_eq_3_div_4_l29_29572

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29572


namespace slope_angle_and_intercept_l29_29261

def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

theorem slope_angle_and_intercept (x y : ℝ) :
  line_equation x y →
  ∃ m θ b, m = -1 ∧ θ = 135 ∧ b = -1 :=
by 
  intros h
  use [-1, 135, -1]
  sorry

end slope_angle_and_intercept_l29_29261


namespace infinite_series_sum_eq_l29_29694

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29694


namespace prove_perpendicular_l29_29923

-- Define the side lengths of the four squares
variables (s1 s2 s3 s4 : ℝ)

-- Define the centers of the squares as points in 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the centers of the squares
def O1 : Point := { x := 0, y := 0 }
def O2 : Point := { x := 2 * s1, y := 0 }
def O3 : Point := { x := 2 * s1, y := 2 * s2 }
def O4 : Point := { x := 0, y := 2 * s2 }

-- Define distances squared between centers
def distance_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Calculate distances squared between adjacent centers
def O1O2_sq : ℝ := distance_sq O1 O2
def O2O3_sq : ℝ := distance_sq O2 O3
def O3O4_sq : ℝ := distance_sq O3 O4
def O4O1_sq : ℝ := distance_sq O4 O1

-- Prove the perpendicularity of line segments O1O3 and O2O4
theorem prove_perpendicular (s1 s2 s3 s4 : ℝ) : 
  distance_sq O1 O2 + distance_sq O3 O4 = distance_sq O2 O3 + distance_sq O4 O1 → 
  distance_sq O1 O3 + distance_sq O2 O4 = O1O2_sq + O3O4_sq :=
sorry

end prove_perpendicular_l29_29923


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29275

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29275


namespace parallel_translation_invariant_l29_29208

variables {X Y : Type} [AffineSpace ℝ X] [AffineSpace ℝ Y]

noncomputable def parallel_translation {X : Type} [AddGroup X] (a : X) (p : X) : X := p + a

-- Given two convex curves K1 and K2
variables (K1 K2 : Set X)

-- Assume the parallel translation by a vector d
variables (d : X)

-- Define points P and Q on K1 and K2 respectively
variables (P : X) (Q : X)
-- Assume P ∈ K1 and Q ∈ K2
variables (hPK1 : P ∈ K1) (hQK2 : Q ∈ K2)

-- Assume K1' and K2' are translated versions of K1 and K2 by vector d
def K1' : Set X := parallel_translation d '' K1
def K2' : Set X := parallel_translation d '' K2

-- Goal: Prove that the set sum (Minkowski sum) of translated curves equals the translated sum of original curves
theorem parallel_translation_invariant :
  (parallel_translation d '' K1) + (parallel_translation d '' K2) = parallel_translation d '' (K1 + K2) :=
sorry

end parallel_translation_invariant_l29_29208


namespace arithmetic_sequence_sum_l29_29125

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ → ℕ)
  (is_arithmetic_seq : ∀ n, a (n + 1) = a n + d n)
  (h : (a 2) + (a 5) + (a 8) = 39) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9) = 117 := 
sorry

end arithmetic_sequence_sum_l29_29125


namespace horner_v3_value_l29_29255

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value_l29_29255


namespace inequality_1_inequality_2_l29_29930

noncomputable theory

variables {A B C D E F : Type} [EuclideanDivisionRing A] [EuclideanDivisionRing B] [EuclideanDivisionRing C] [EuclideanDivisionRing D] [EuclideanDivisionRing E] [EuclideanDivisionRing F]
variables {a b c : ℝ}

-- The conditions
def bisects_perimeter (a b c : ℝ) (D E F : Type) := sorry  -- Assume D, E, F are such points bisecting the perimeter

-- The first part
theorem inequality_1
  (h : bisects_perimeter a b c D E F) :
  ∃ DE EF FD : ℝ, DE + EF + FD ≥ ½ * (a + b + c) := sorry

-- The second part
theorem inequality_2
  (h : bisects_perimeter a b c D E F) :
  ∃ DE EF FD : ℝ, DE^2 + EF^2 + FD^2 ≥ 1 / 12 * (a + b + c)^2 := sorry

end inequality_1_inequality_2_l29_29930


namespace sum_series_eq_3_div_4_l29_29561

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29561


namespace omega_values_l29_29011

theorem omega_values (a1 a2 a3 a4 : ℝ) (f : ℝ → ℝ) (ω : ℝ)
  (h1 : determinant : matrix (fin 2) (fin 2) ℝ →  ℝ := λ M,
    (M 0 0) * (M 1 1) - (M 0 1) * (M 1 0))
  (h2 : ∀ x, f x = determinant ![\sqrt{3}, sin (ω * x)],
                    ![1, cos (ω * x)])
  (h3 : ∀ x, f (x + 2 * π / (3 * ω)) = f (-x)) :
  ω = -7/4 := sorry

end omega_values_l29_29011


namespace pascal_third_smallest_four_digit_number_l29_29396

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29396


namespace max_min_values_function_l29_29044

noncomputable def f (x : ℝ) : ℝ := real.exp (2 * x ^ 2 - 4 * x - 6)

theorem max_min_values_function :
  (∀ x ∈ set.Icc 0 3, f x ≤ 1) ∧ f 3 = 1 ∧
  (∀ x ∈ set.Icc 0 3, f x ≥ real.exp (-8)) ∧ f 1 = real.exp (-8) :=
by
  sorry

end max_min_values_function_l29_29044


namespace value_of_T_l29_29203

theorem value_of_T (T : ℝ) (h : (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120) : T = 67.5 :=
sorry

end value_of_T_l29_29203


namespace series_sum_l29_29676

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29676


namespace third_smallest_four_digit_in_pascals_triangle_l29_29318

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29318


namespace sufficient_not_necessary_condition_l29_29863

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a > 1) (hb : b > 1) (hpos : 0 < a) (hposb : 0 < b) : 
  (a > 1 ∧ b > 1) -> (log 10 a + log 10 b > 0) ∧ ∃ c d : ℝ, c > 1 → d > 1 ∧ (log 10 c + log 10 d > 0) :=
by sorry

end sufficient_not_necessary_condition_l29_29863


namespace distinct_collections_l29_29199

/-- Number of distinct possible collections of letters put in the bag given the conditions. -/
theorem distinct_collections (N_C : ℕ)
  (h1 : ∃ collections C₁, ∀ c ∈ C₁, c ∈ ['C', 'C', 'L', 'L', 'T', 'R'])
  (h2 : ∃ collections V₁, ∀ v ∈ V₁, v ∈ ['A', 'A', 'C', 'U', 'O'])
  (h3 : ∃ collections, 
    (collections ⊆ 'A', 'A', 'C', 'U', 'O') ∨
    (collections ⊆ 'C', 'C', 'L', 'L', 'T', 'R') ∨
    (|collections| = N_C)) :
  (collections = 3 * N_C) := 
sorry

end distinct_collections_l29_29199


namespace third_smallest_four_digit_in_pascals_triangle_l29_29319

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29319


namespace series_converges_to_three_fourths_l29_29806

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29806


namespace find_function_l29_29818

variable (R : Type) [LinearOrderedField R]

theorem find_function
  (f : R → R)
  (h : ∀ x y : R, f (x + y) + y ≤ f (f (f x))) :
  ∃ c : R, ∀ x : R, f x = c - x :=
sorry

end find_function_l29_29818


namespace y_less_than_z_by_40_percent_l29_29138

variable {x y z : ℝ}

theorem y_less_than_z_by_40_percent (h1 : x = 1.3 * y) (h2 : x = 0.78 * z) : y = 0.6 * z :=
by
  -- The proof will be provided here
  -- We are demonstrating that y = 0.6 * z is a consequence of h1 and h2
  sorry

end y_less_than_z_by_40_percent_l29_29138


namespace different_methods_placing_cards_l29_29981

noncomputable def total_methods : ℕ :=
  let cards : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let envelopes : Finset ℕ := {1, 2, 3}
  let card_placements := {p : cards → envelopes // ∀ i j, (cards i ≠ cards j → p i ≠ p j)}
  card_placements.filter(λ p, p 1 ≠ p 2).card

theorem different_methods_placing_cards : total_methods = 72 :=
by
  sorry

end different_methods_placing_cards_l29_29981


namespace translate_and_properties_l29_29254

def f (x : ℝ) : ℝ := Real.cos (2 * x)

def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem translate_and_properties :
  (∀ x, g(x) = f(x - π/4)) ∧
  (∀ x, g(-x) = -g(x)) ∧
  (∀ x, 0 < x ∧ x < π/4 → g(x) > g(x - π/4)) :=
by
  sorry

end translate_and_properties_l29_29254


namespace sum_series_div_3_powers_l29_29782

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29782


namespace third_smallest_four_digit_in_pascals_triangle_l29_29324

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29324


namespace arcsin_sin_solution_l29_29216

theorem arcsin_sin_solution (x : ℝ) (h : - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2) :
  arcsin (sin x) = x / 3 ↔ x = -3 * Real.pi ∨ x = - Real.pi ∨ x = 0 ∨ x = Real.pi ∨ x = 3 * Real.pi :=
sorry

end arcsin_sin_solution_l29_29216


namespace sum_geometric_series_l29_29680

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29680


namespace sum_series_eq_3_over_4_l29_29757

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29757


namespace mary_total_money_l29_29195

def num_quarters : ℕ := 21
def quarters_worth : ℚ := 0.25
def dimes_worth : ℚ := 0.10

def num_dimes (Q : ℕ) : ℕ := (Q - 7) / 2

def total_money (Q : ℕ) (D : ℕ) : ℚ :=
  Q * quarters_worth + D * dimes_worth

theorem mary_total_money : 
  total_money num_quarters (num_dimes num_quarters) = 5.95 := 
by
  sorry

end mary_total_money_l29_29195


namespace circle_center_sum_l29_29508

theorem circle_center_sum (x y : ℝ) :
  (x^2 + y^2 = 10*x - 12*y + 40) →
  x + y = -1 :=
by {
  sorry
}

end circle_center_sum_l29_29508


namespace possible_degrees_of_remainder_l29_29414

-- Define the divisor polynomial
def divisor_poly : Polynomial ℚ := -2 * X^3 + 7 * X^2 - 4

-- State the theorem about the possible degrees of the remainder
theorem possible_degrees_of_remainder (p : Polynomial ℚ) : 
  ∃ r : Polynomial ℚ, degree r < 3 ∧ r.degree = 0 ∨ r.degree = 1 ∨ r.degree = 2 :=
sorry

end possible_degrees_of_remainder_l29_29414


namespace quadrant_of_alpha_l29_29862

theorem quadrant_of_alpha (α : ℝ) (h1 : sin (2 * α) > 0) (h2 : cos α < 0) : π / 2 < α ∧ α < π := sorry

end quadrant_of_alpha_l29_29862


namespace smallest_n_for_three_pairs_l29_29831

theorem smallest_n_for_three_pairs :
  ∃ (n : ℕ), (0 < n) ∧
    (∀ (x y : ℕ), (x^2 - y^2 = n) → (0 < x) ∧ (0 < y)) ∧
    (∃ (a b c : ℕ), 
      (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
      (∃ (x y : ℕ), (x^2 - y^2 = n) ∧
        (((x, y) = (a, b)) ∨ ((x, y) = (b, c)) ∨ ((x, y) = (a, c))))) :=
sorry

end smallest_n_for_three_pairs_l29_29831


namespace trapezium_area_l29_29031

-- Define the lengths of the parallel sides and the distance between them
def side_a : ℝ := 20
def side_b : ℝ := 18
def height : ℝ := 15

-- Define the formula for the area of a trapezium
def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem
theorem trapezium_area :
  area_trapezium side_a side_b height = 285 :=
by
  sorry

end trapezium_area_l29_29031


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29648

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29648


namespace infinite_series_sum_eq_l29_29709

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29709


namespace raiders_wins_35_l29_29234

constant Sharks Hawks Raiders Wolves Dragons : ℕ

axiom wins_set : {Sharks, Hawks, Raiders, Wolves, Dragons} = {15, 20, 30, 35, 40}
axiom wolves_wins_gt_25 : Wolves > 25
axiom raiders_condition : Wolves < Raiders ∧ Raiders < Dragons
axiom sharks_lt_hawks : Sharks < Hawks

theorem raiders_wins_35 : Raiders = 35 := 
by sorry

end raiders_wins_35_l29_29234


namespace minimum_value_of_2m_plus_n_solution_set_for_inequality_l29_29886

namespace MathProof

-- Definitions and conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Part (I)
theorem minimum_value_of_2m_plus_n
  (m n : ℝ)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_f_nonneg : ∀ x : ℝ, f x m n ≥ 1) :
  2 * m + n ≥ 2 :=
sorry

-- Part (II)
theorem solution_set_for_inequality
  (x : ℝ) :
  (f x 2 3 > 5 ↔ (x < 0 ∨ x > 2)) :=
sorry

end MathProof

end minimum_value_of_2m_plus_n_solution_set_for_inequality_l29_29886


namespace infinite_series_sum_eq_3_over_4_l29_29717

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29717


namespace third_smallest_four_digit_Pascal_triangle_l29_29282

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29282


namespace evaluate_series_sum_l29_29592

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29592


namespace infinite_series_sum_eq_3_div_4_l29_29615

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29615


namespace betty_height_in_feet_l29_29494

theorem betty_height_in_feet (dog_height carter_height betty_height : ℕ) (h1 : dog_height = 24) 
  (h2 : carter_height = 2 * dog_height) (h3 : betty_height = carter_height - 12) : betty_height / 12 = 3 :=
by
  sorry

end betty_height_in_feet_l29_29494


namespace series_result_l29_29630

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29630


namespace sum_series_equals_three_fourths_l29_29550

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29550


namespace series_sum_l29_29672

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29672


namespace probability_female_wears_glasses_l29_29917

def prob_female_wears_glasses (total_females : ℕ) (females_no_glasses : ℕ) : ℚ :=
  let females_with_glasses := total_females - females_no_glasses
  females_with_glasses / total_females

theorem probability_female_wears_glasses :
  prob_female_wears_glasses 18 8 = 5 / 9 := by
  sorry  -- Proof is skipped

end probability_female_wears_glasses_l29_29917


namespace problem_correct_props_count_l29_29474

/-- A proposition is correct if and only if it satisfies the real-world condition described. -/
def is_correct_proposition (n : ℕ) : Prop :=
  match n with
  | 1 => ∀ (P1 P2 P3 P4 : Point), ¬coplanar P1 P2 P3 P4 → ¬collinear P1 P2 P3
  | 2 => ∀ (A B C D E : Point), coplanar A B C D → coplanar A B C E → coplanar A B C D E
  | 3 => ∀ (a b c : Line), coplanar a b → coplanar a c → coplanar b c
  | 4 => ∀ (l1 l2 l3 l4 : Segment), joined_end_to_end l1 l2 l3 l4 → coplanar_segments l1 l2 l3 l4
  | _ => false

/-- Main theorem stating the number of correct propositions is 1 -/
theorem problem_correct_props_count : 
  (is_correct_proposition 1 ↔ true) ∧
  (is_correct_proposition 2 ↔ false) ∧
  (is_correct_proposition 3 ↔ false) ∧
  (is_correct_proposition 4 ↔ false) :=
sorry

end problem_correct_props_count_l29_29474


namespace repeating_fn_1996_times_l29_29883

-- Define the sequence of digits
def repeating_decimal_seq : List Nat := [9, 1, 8, 2, 7, 3, 6, 4, 5]

-- Define the function f that provides the nth digit in the repeating sequence
def f (n : Nat) : Nat := repeating_decimal_seq.get! ((n - 1) % 9)

-- The Lean statement representing the main problem
theorem repeating_fn_1996_times :
  ∀ f : Nat → Nat,
  (∀ n, f n = repeating_decimal_seq.get! ((n - 1) % 9)) →
  \underbrace{f\{f \cdots f[f}_{1996 \times}(1)]\} = 4 := by
  sorry

end repeating_fn_1996_times_l29_29883


namespace determine_m_l29_29249

-- Define the conditions: the quadratic equation and the sum of roots
def quadratic_eq (x m : ℝ) : Prop :=
  x^2 + m * x + 2 = 0

def sum_of_roots (x1 x2 : ℝ) : ℝ := x1 + x2

-- Problem Statement: Prove that m = 4
theorem determine_m (x1 x2 m : ℝ) 
  (h1 : quadratic_eq x1 m) 
  (h2 : quadratic_eq x2 m)
  (h3 : sum_of_roots x1 x2 = -4) : 
  m = 4 :=
by
  sorry

end determine_m_l29_29249


namespace third_smallest_four_digit_in_pascals_triangle_l29_29320

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29320


namespace third_smallest_four_digit_in_pascal_l29_29352

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29352


namespace geometric_series_sum_l29_29966

theorem geometric_series_sum (x : ℝ) (n : ℕ) (h : 1 ≤ n) : 
  (finset.range (n+1)).sum (λ k, x^k) = 
    if x = 1 then n + 1 else (1 - x^(n+1)) / (1 - x) := 
by
  sorry

end geometric_series_sum_l29_29966


namespace granger_son_age_l29_29196

theorem granger_son_age : 
  ∀ (S G : ℕ), 
  (G = 2 * S + 10) ∧ (G - 1 = 3 * (S - 1) - 4) → 
  S = 16 :=
by 
  intros S G h,
  cases h with h1 h2,
  sorry

end granger_son_age_l29_29196


namespace x_minus_y_sq_l29_29850

noncomputable def y (x : ℝ) := real.sqrt(2 * x - 3) + real.sqrt(3 - 2 * x) - 4

theorem x_minus_y_sq (x : ℝ) (y_eq : y x = real.sqrt(2 * x - 3) + real.sqrt(3 - 2 * x) - 4) :
  x = 3 / 2 → x - (y x)^2 = -29 / 2 :=
by
  sorry

end x_minus_y_sq_l29_29850


namespace pascal_third_smallest_four_digit_number_l29_29394

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29394


namespace third_smallest_four_digit_in_pascals_triangle_l29_29297

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29297


namespace sine_tangent_coincide_3_decimal_places_l29_29062

open Real

noncomputable def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

theorem sine_tangent_coincide_3_decimal_places :
  ∀ θ : ℝ,
    0 ≤ θ ∧ θ ≤ deg_to_rad (4 + 20 / 60) →
    |sin θ - tan θ| < 0.0005 :=
by
  intros θ hθ
  sorry

end sine_tangent_coincide_3_decimal_places_l29_29062


namespace meaningful_expression_iff_l29_29897

theorem meaningful_expression_iff (x : ℝ) : (∃ y, y = (2 : ℝ) / (2*x - 1)) ↔ x ≠ (1 / 2 : ℝ) :=
by
  sorry

end meaningful_expression_iff_l29_29897


namespace series_converges_to_three_fourths_l29_29807

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29807


namespace series_sum_l29_29674

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29674


namespace left_handed_rock_lovers_l29_29250

def total_people := 30
def left_handed := 12
def like_rock_music := 20
def right_handed_dislike_rock := 3

theorem left_handed_rock_lovers : ∃ x, x + (left_handed - x) + (like_rock_music - x) + right_handed_dislike_rock = total_people ∧ x = 5 :=
by
  sorry

end left_handed_rock_lovers_l29_29250


namespace sum_geometric_series_l29_29688

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29688


namespace part_I_part_II_l29_29856

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Define the conditions a_{n+1} = 2*a_n + n - 1 and a_1 = 1
def satisfies_cond (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * (a n) + n - 1

-- Prove that {a_n + n} is a geometric sequence
theorem part_I (h : satisfies_cond a) : ∃ r, ∀ n, (a n + n) / r = (a (n + 1) + (n + 1)) := sorry

-- Prove that the sum of the first n terms of the sequence {a_n} is 2^(n+1) - n(n+1)/2 - 2
theorem part_II (h : satisfies_cond a) (S : ℕ → ℕ) : (∀ n, (∑ i in Finset.range n, a i) = 2 ^ (n + 1) - (n * (n + 1)) / 2 - 2) := sorry

end part_I_part_II_l29_29856


namespace third_smallest_four_digit_in_pascals_triangle_l29_29382

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29382


namespace max_area_of_region_R_l29_29920

noncomputable def maxAreaOfRegion (r1 r2 r3 r4 : ℝ) : ℝ :=
  let area (r : ℝ) := π * r ^ 2
  area r1 + area r2 + area r3 + area r4

theorem max_area_of_region_R : 
  maxAreaOfRegion 2 4 6 8 - 4 * π = 112 * π := 
by
  sorry

end max_area_of_region_R_l29_29920


namespace sum_series_div_3_powers_l29_29788

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29788


namespace convert_413_base5_to_base10_l29_29503

theorem convert_413_base5_to_base10 : 
  let base5_num := (4 : ℕ ) * (5^2 : ℕ) + (1 : ℕ) * (5^1 : ℕ) + (3 : ℕ) * (5^0 : ℕ) in
  base5_num = 108 :=
by 
  -- We skip the proof here
  sorry

end convert_413_base5_to_base10_l29_29503


namespace third_smallest_four_digit_in_pascal_triangle_l29_29365

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29365


namespace nat_num_unique_l29_29965

theorem nat_num_unique (p : ℕ) (hp : p.prime) :
  ∃! n : ℕ, p * n % (p + n) = 0 := by
  sorry

end nat_num_unique_l29_29965


namespace counterexample_exists_l29_29209

noncomputable def counterexample_function : ℝ → ℝ :=
  λ x : ℝ, if x ∈ set.Ioc 1 3 then -((x - 5/2)^2) + 9/4 else 0

theorem counterexample_exists :
  (∃ f : ℝ → ℝ, (∀ x ∈ set.Ioc 1 3, f x > f 1) ∧ ¬ (∀ x y ∈ set.Icc 1 3, x < y → f x ≤ f y)) :=
begin
  use counterexample_function,
  split,
  {
    intros x hx,
    have h_f1 : counterexample_function 1 = 1/4, by {
      -- Calculate f(1)
      simp only [counterexample_function],
      split_ifs,
      norm_num,
    },
    calc
      counterexample_function x
          = -((x - 5/2)^2) + 9/4 : by {
              simp only [counterexample_function],
              split_ifs at hx ⊢,
              norm_cast,
            }
      ... > 1/4 : by {
        -- Algebraic manipulation to show f(x) > f(1)
        apply lt_of_sub_pos,
        ring_exp,
        linarith,
      }
  },
  {
    -- f(x) is not an increasing function on [1, 3]
    intro H,
    let x := (1 : ℝ),
    let y := (3 : ℝ),
    have x_in : x ∈ set.Icc 1 3, by { exact set.left_mem_Icc.mpr (by norm_num) },
    have y_in : y ∈ set.Icc 1 3, by { exact set.right_mem_Icc.mpr (by norm_num) },
    have hx_lt_y : x < y, by { norm_num },
    specialize H x x_in y y_in hx_lt_y,
    have h_fx : counterexample_function x = 1/4, by {
      simp only [counterexample_function],
      split_ifs,
      norm_num,
    },
    have h_fy : counterexample_function y = -1/4, by {
      simp only [counterexample_function],
      split_ifs,
      norm_num,
    },
    linarith,
  }
end

end counterexample_exists_l29_29209


namespace sum_x_coords_Q3_l29_29435

theorem sum_x_coords_Q3 (Q1 Q2 Q3 : Type) [Finite Q1] [Finite Q2] [Finite Q3]
  (vertices_Q1 : Q1 → ℝ) (vertices_Q2 : Q2 → ℝ) (vertices_Q3 : Q3 → ℝ) 
  (midpoints1 : (Q1 × Q1) → Q2) (midpoints2 : (Q2 × Q2) → Q3)
  (sumQ1 : ∑ (x : Q1), vertices_Q1 x = 1050):
  ∑ (x : Q3), vertices_Q3 x = 1050 :=
sorry

end sum_x_coords_Q3_l29_29435


namespace exactly_one_number_without_consecutive_sum_l29_29121

def is_not_sum_of_consecutive_integers (n : ℕ) : Prop :=
  ¬∃ (k : ℕ) (h : k > 1), ∃ (m : ℕ), n = m * k + (k * (k - 1)) / 2

theorem exactly_one_number_without_consecutive_sum :
  (is_not_sum_of_consecutive_integers 101).to_bool +
  (is_not_sum_of_consecutive_integers 148).to_bool +
  (is_not_sum_of_consecutive_integers 200).to_bool +
  (is_not_sum_of_consecutive_integers 512).to_bool +
  (is_not_sum_of_consecutive_integers 621).to_bool = 1 :=
by sorry

end exactly_one_number_without_consecutive_sum_l29_29121


namespace sum_of_integer_solutions_l29_29996

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sum_of_integer_solutions :
  (∑ i in Finset.filter (λ x, ∃ y : ℝ, y = x ∧ sqrt (x^2 - 3*x - 54) - sqrt (x^2 - 27*x + 162) < 8 * sqrt ((x + 6) / (x - 9))) (Finset.Icc (-25 : ℤ) 25), (i : ℝ))
  = -290 :=
  sorry

end sum_of_integer_solutions_l29_29996


namespace series_converges_to_three_fourths_l29_29802

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29802


namespace derivative_of_f_l29_29848

def f (x : ℝ) : ℝ := x / (x - 1)

theorem derivative_of_f :
  ∀ x : ℝ, (x ≠ 1) → (derivative f x = -(1 / (x - 1) ^ 2)) :=
by
  sorry

end derivative_of_f_l29_29848


namespace number_of_square_tiles_l29_29443

theorem number_of_square_tiles (a b : ℕ) (h1 : a + b = 32) (h2 : 3 * a + 4 * b = 110) : b = 14 :=
by
  -- the proof steps are skipped
  sorry

end number_of_square_tiles_l29_29443


namespace pascal_third_smallest_four_digit_number_l29_29390

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29390


namespace pascal_third_smallest_four_digit_number_l29_29395

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29395


namespace period_of_function_l29_29514

theorem period_of_function :
  ∀ x : ℝ, (λ x, sin x ^ 2 + sin x * cos x) (x + π) = (λ x, sin x ^ 2 + sin x * cos x) x :=
by sorry

end period_of_function_l29_29514


namespace solution_l29_29956

def diamond (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...))))

noncomputable def find_h (h : ℝ) : Prop := diamond 2 h = 8

theorem solution : ∃ h : ℝ, find_h h ∧ h = 12 :=
by
  sorry

end solution_l29_29956


namespace third_smallest_four_digit_in_pascals_triangle_l29_29296

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29296


namespace series_sum_eq_l29_29532

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29532


namespace series_sum_correct_l29_29776

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29776


namespace series_sum_eq_l29_29529

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29529


namespace third_smallest_four_digit_in_pascal_l29_29343

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29343


namespace area_of_absolute_value_sum_l29_29512

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end area_of_absolute_value_sum_l29_29512


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29338

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29338


namespace third_smallest_four_digit_in_pascals_triangle_l29_29356

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29356


namespace find_angles_triangle_ABC_l29_29453

-- Define the setup including points, tangency, and ratios
variables {A B C D E : Type} [InnerProductSpace ℝ Type] [enough_point : Function.one_ring Type]
          (tangent_AD_ω : ∀ {ω : circle[ℝ]} (A B C D : Type), is_tangent (Ω A B C D) AD)
          (ω : circle[ℝ])
          (h1 : on_extension BC D)
          (h2 : tangents AD ω)
          (h3 : AC_int_on_circle_ω_1_2 AC ω ABD E)
          (h4 : angle_bisector_tangent ADE ω)

-- Define the statement to prove the angles
theorem find_angles_triangle_ABC :  angles_triangle_ABC A B C = {30°, 60°, 90°} :=
  by
  sorry

end find_angles_triangle_ABC_l29_29453


namespace find_set_of_points_B_l29_29118

noncomputable def is_incenter (A B C I : Point) : Prop :=
  -- define the incenter condition
  sorry

noncomputable def angle_less_than (A B C : Point) (α : ℝ) : Prop :=
  -- define the condition that all angles of triangle ABC are less than α
  sorry

theorem find_set_of_points_B (A I : Point) (α : ℝ) (hα1 : 60 < α) (hα2 : α < 90) :
  ∃ B : Point, ∃ C : Point,
    is_incenter A B C I ∧ angle_less_than A B C α :=
by
  -- The proof will go here
  sorry

end find_set_of_points_B_l29_29118


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29650

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29650


namespace series_sum_correct_l29_29768

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29768


namespace sum_series_div_3_powers_l29_29795

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29795


namespace series_converges_to_three_fourths_l29_29808

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29808


namespace geometry_problem_l29_29477

-- Definitions translating the conditions
variables (O A B C D E F : Point)
variable [metric_space (Point)]

-- Diameter condition
def diameter_circle (O A B : Point) : Prop :=
  collinear O A B ∧ dist O A = dist O B ∧ dist A B = 2 * dist O A

-- Intersection and perpendicular conditions
def intersect_and_perpendicular (A B C D E F : Point) : Prop :=
  ∃ (P : Point), collinear B D P ∧ collinear C A P ∧ P = E ∧ perpendicular (line_through E F) (line_through B F)

-- Problem statement
theorem geometry_problem
  (h1 : diameter_circle O A B)
  (h2 : intersect_and_perpendicular A B C D E F) :
  (angle D E A = angle D F A) ∧ 
  (dist A B ^ 2 = (dist B E) * (dist B D) - (dist A E) * (dist A C)) :=
sorry

end geometry_problem_l29_29477


namespace lcm_sum_not_power_of_two_l29_29979

open Nat

theorem lcm_sum_not_power_of_two (s : Set ℕ) (r b : Set ℕ)
  (h1 : ∃ a b, s = Set.Icc a b) -- consecutive numbers
  (h2 : s = r ∪ b)
  (h3 : ∀ n ∈ r, n ∈ s)
  (h4 : ∀ n ∈ b, n ∈ s)
  (h5 : ∃ x ∈ r, true) -- red numbers are present
  (h6 : ∃ y ∈ b, true) -- blue numbers are present :
  ∃ n, let LCM_red  := r.lcm id
           LCM_blue := b.lcm id in
  (LCM_red + LCM_blue ≠ 2 ^ n) :=
by
  sorry

end lcm_sum_not_power_of_two_l29_29979


namespace sum_series_eq_3_over_4_l29_29746

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29746


namespace sum_series_eq_3_div_4_l29_29571

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29571


namespace sum_series_eq_3_div_4_l29_29563

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29563


namespace solve_equation_l29_29027

theorem solve_equation (x : ℝ) (n : ℤ) (h : x^2 - 8 * (n : ℝ) + 7 = 0) (hx : n ≤ x ∧ x < n + 1) :
    x = 1 ∨ x = sqrt 33 ∨ x = sqrt 41 ∨ x = 7 :=
by sorry

end solve_equation_l29_29027


namespace third_smallest_four_digit_in_pascal_triangle_l29_29375

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29375


namespace arcsin_one_eq_pi_div_two_l29_29001

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = (Real.pi / 2) :=
by
  sorry

end arcsin_one_eq_pi_div_two_l29_29001


namespace cube_volume_is_64_l29_29246

def cube_volume_from_edge_sum (edge_sum : ℝ) (num_edges : ℕ) : ℝ :=
  let edge_length := edge_sum / num_edges in
  edge_length ^ 3

theorem cube_volume_is_64 :
  cube_volume_from_edge_sum 48 12 = 64 := by
  sorry

end cube_volume_is_64_l29_29246


namespace sum_of_roots_l29_29268

theorem sum_of_roots (a b c d : ℝ) (h_eq : 6 * a^3 + 7 * b^2 - 12 * c = 0) :
    ∑ x in {x | 6 * x^3 + 7 * x^2 - 12 * x = 0}, x = -7 / 6 :=
sorry

end sum_of_roots_l29_29268


namespace cube_volume_l29_29240

theorem cube_volume : 
  (∃ (s : ℝ), 12 * s = 48) → 
  (∃ (V : ℝ), V = (4:ℝ)^3 ∧ V = 64) := 
by 
  intro h
  rcases h with ⟨s, hs⟩
  use s^3
  have hs_eq : s = 4 := by linarith
  rw [hs_eq, pow_succ, pow_succ, pow_zero, mul_assoc, mul_assoc, mul_one, pow_succ, pow_zero]
  norm_num
  exact ⟨rfl, by norm_num⟩

end cube_volume_l29_29240


namespace series_sum_l29_29667

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29667


namespace range_of_a_l29_29911

-- Definitions of given functions
def f (a : ℝ) (x : ℝ) := a * x^3 - 1 / 3
def g (c : ℝ) (x : ℝ) := x^2 - (2 / 3) * c * x

-- Condition that the functions have three distinct intersection points whose abscissas form an arithmetic sequence
def has_three_distinct_intersection_points (a c : ℝ) : Prop :=
  let h (x : ℝ) := f a x - g c x in
  let h' (x : ℝ) := 3 * a * x^2 - 2 * x + (2 / 3) * c in
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (∀ x : ℝ, h x = 0 ↔ (x = x1 ∨ x = x2 ∨ x = x3)) ∧
    h' x1 = 0 ∧ h' x2 = 0 ∧ h' x3 = 0 ∧
    ∀ x : ℝ, (h' x = 0) → (x = x1 ∨ x = x2)

-- Main theorem: range of a
theorem range_of_a (a c : ℝ) (h_condition : has_three_distinct_intersection_points a c) :
  0 < a ∧ a < 1 / 3 :=
sorry

end range_of_a_l29_29911


namespace sum_of_integer_solutions_l29_29995

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sum_of_integer_solutions :
  (∑ i in Finset.filter (λ x, ∃ y : ℝ, y = x ∧ sqrt (x^2 - 3*x - 54) - sqrt (x^2 - 27*x + 162) < 8 * sqrt ((x + 6) / (x - 9))) (Finset.Icc (-25 : ℤ) 25), (i : ℝ))
  = -290 :=
  sorry

end sum_of_integer_solutions_l29_29995


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29402

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29402


namespace sum_series_eq_3_over_4_l29_29755

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29755


namespace evaluate_series_sum_l29_29594

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29594


namespace distinct_collections_count_l29_29977

theorem distinct_collections_count :
  ∃ (vowels consonants : Finset ℕ),
    card vowels = 6 ∧ card consonants = 6 ∧
    vowels = {1, 1, 1, 2, 2, 2} ∧ consonants = {3, 3, 4, 5, 6, 7} →
    (count_distinct_collections vowels consonants 3 3) = 112 :=
by
  sorry

noncomputable def count_distinct_collections
  (vowels consonants : Finset ℕ) (num_vowels num_consonants : ℕ) : ℕ :=
  let vowel_collections := by sorry -- Calculation of distinct vowel collections
  let consonant_collections := by sorry -- Calculation of distinct consonant collections
  vowel_collections * consonant_collections

end distinct_collections_count_l29_29977


namespace volume_tetrahedron_l29_29931

open Real

variables (a S : ℝ) (α β : ℝ) (E : Point) (BC : Segment)

-- Definitions based on conditions
axiom h_length_ad : ∥AD∥ = a
axiom h_midpoint_e : is_midpoint E BC
axiom h_area_ade : area (triangle AD E) = S

-- Prove the volume of tetrahedron ABCD given the above conditions
theorem volume_tetrahedron (h_length_ad : ∥AD∥ = a) (h_area_ade : area (triangle AD E) = S)
  (h_midpoint_e : is_midpoint E BC) (α β : ℝ) : 
  volume (tetrahedron ABCD) = (8 * S^2 * sin α * sin β) / (3 * a * sin (α + β)) := 
sorry

end volume_tetrahedron_l29_29931


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29407

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29407


namespace sum_series_div_3_powers_l29_29791

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29791


namespace max_handshakes_l29_29147

theorem max_handshakes (n k : ℕ) (h1 : k ≤ n) (h2 : ∀ i, i < k → (∀ j, j < k → i ≠ j → acquaintance i j)) :
    max_handshakes n k = (n * (n - 1)) / 2 - k :=
by
  sorry

end max_handshakes_l29_29147


namespace isosceles_right_triangle_l29_29201

theorem isosceles_right_triangle (x : ℝ) (hx : sin x ^ 3 + sin x * sin (2 * x) - 3 * cos x ^ 3 = 0) (h_right : x = π / 2 ∨ x < π / 2) :
  ∃ a b c : ℝ, a ≠ b ∧ (a * a + b * b = c * c ∧ (a = b ∨ (a best inclination” or “a angle len”), from the sum they might pinpoint the nature.

end isosceles_right_triangle_l29_29201


namespace brownian_bridge_B_t_circ_brownian_motion_B_t_brownian_motion_W_t_l29_29429

noncomputable def xi : ℕ → ℝ := sorry -- Placeholder for the i.i.d. normal random variables

def B_t_circ (t : ℝ) : ℝ :=
  ∑ n in (filter (λ n, n > 0) (range (10^6))), xi n * (sqrt 2 * (sin (n * π * t)) / (n * π)) -- Approximating the infinite sum

def B_t (t : ℝ) : ℝ :=
  xi 0 * t + ∑ n in (filter (λ n, n > 0) (range (10^6))), xi n * (sqrt 2 * (sin (n * π * t)) / (n * π)) -- Approximating the infinite sum

def W_t (t : ℝ) : ℝ :=
  sqrt 2 * ∑ n in (filter (λ n, n > 0) (range (10^6))), xi n * ((1 - cos (n * π * t)) / (n * π)) -- Approximating the infinite sum

theorem brownian_bridge_B_t_circ :
  ∀ t, 0 ≤ t ∧ t ≤ 1 → ∃ (B_t_circ_realized : ℝ → ℝ), ∀ s t, B_t_circ s ≈ B_t_circ_realized s ∧
  B_t_circ_realized t defines a Brownian bridge :=
sorry

theorem brownian_motion_B_t :
  ∀ t, 0 ≤ t ∧ t ≤ 1 → ∃ (B_t_realized : ℝ → ℝ), ∀ s t, B_t s ≈ B_t_realized s ∧
  B_t_realized t defines a Brownian motion :=
sorry

theorem brownian_motion_W_t :
  ∀ t, 0 ≤ t ∧ t ≤ 1 → ∃ (W_t_realized : ℝ → ℝ), ∀ s t, W_t s ≈ W_t_realized s ∧
  W_t_realized t defines a Brownian motion :=
sorry

end brownian_bridge_B_t_circ_brownian_motion_B_t_brownian_motion_W_t_l29_29429


namespace number_of_T_l29_29954

def is_in_T (m : ℕ) : Prop := 
  m > 1 ∧ ∃ (e : ℕ → ℕ), (∀ i, e i = e (i + 6)) ∧ (0.e_1e_2e_3e_4e_5e_6.. in decimal expansion of 1/m repeat every 6 digits )

theorem number_of_T : {m : ℕ | is_in_T m}.to_finset.card = 47 := 
  by sorry

end number_of_T_l29_29954


namespace sum_series_div_3_powers_l29_29794

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29794


namespace count_icing_on_two_sides_l29_29439

def cake_structure (n : ℕ) := n = 5
def icing_on_top_and_sides (n : ℕ) := icing_on_top (cake_structure n) ∧ icing_on_sides (cake_structure n)
def icing_on_middle_layer (n : ℕ) := icing_on_horizontal_middle (cake_structure n)
def no_icing_on_bottom (n : ℕ) := ¬icing_on_bottom (cake_structure n)
def cube_cut_into_smaller_cubes(n : ℕ, m : ℕ) := n^3 = m

theorem count_icing_on_two_sides {n : ℕ} (h1: cake_structure n) (h2: icing_on_top_and_sides n) 
(h3: icing_on_middle_layer n) (h4: no_icing_on_bottom n) (h5: cube_cut_into_smaller_cubes n 125) : 
∃ c : ℕ, c = 24 :=
by
  sorry

end count_icing_on_two_sides_l29_29439


namespace original_number_is_974_l29_29464

theorem original_number_is_974 (x y z : ℕ) 
  (h1 : 100 * x + 10 * y + z - 16 = rev_num (100 * x + 10 * y + z)) 
  (h2 : x + y + z = 20) : 
  100 * x + 10 * y + z = 974 :=
by
  sorry

def rev_num (n : ℕ) : ℕ :=
  let z := n % 10
  let y := (n / 10) % 10
  let x := n / 100
  100 * z + 10 * y + x

end original_number_is_974_l29_29464


namespace matthew_needs_to_cook_l29_29974

noncomputable def ella : ℝ := 2.5
noncomputable def emma : ℝ := 2.5
noncomputable def luke : ℝ := (ella) ^ 2
noncomputable def michael : ℝ := 7
noncomputable def hunter : ℝ := 1.25 * (ella + emma)
noncomputable def zoe : ℝ := 0.6
noncomputable def uncle_steve : ℝ := 1.5 * (ella + emma + luke)

noncomputable def total_hotdogs : ℝ :=
  ella + emma + luke + michael + hunter + zoe + uncle_steve

theorem matthew_needs_to_cook : ⌈total_hotdogs⌉ = 47 := 
by 
  have h1 : ella = 2.5 := rfl
  have h2 : emma = 2.5 := rfl
  have h3 : luke = 6.25 := rfl
  have h4 : michael = 7 := rfl
  have h5 : hunter = 6.25 := by rw [h1, h2]; ring
  have h6 : zoe = 0.6 := rfl
  have h7 : uncle_steve = 16.875 := by rw [h1, h2, h3]; ring
  have hsum : total_hotdogs = 46.975 := 
    by rw [h1, h2, h3, h4, h5, h6, h7]; ring
  have hceil : ⌈46.975⌉ = 47 := by norm_num
  rw hsum
  exact hceil

end matthew_needs_to_cook_l29_29974


namespace k_valid_iff_l29_29822

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l29_29822


namespace second_class_avg_marks_l29_29224

-- Define the conditions as Lean definitions
def avg_marks_class1 : ℕ := 40
def num_students_class1 : ℕ := 30
def num_students_class2 : ℕ := 50
def combined_avg_marks : ℝ := 52.5

-- Define the statement to prove
theorem second_class_avg_marks :
  let total_students := num_students_class1 + num_students_class2 in
  let total_marks_class1 := num_students_class1 * avg_marks_class1 in
  let x := ∀ x : ℝ, (total_marks_class1 + num_students_class2 * x) / total_students = combined_avg_marks → x = 60 in
  x := 60 :=
by
  sorry

end second_class_avg_marks_l29_29224


namespace sum_series_eq_3_div_4_l29_29569

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29569


namespace identification_tags_l29_29442

open Finset

def total_sequences (letters : Finset Char) (digits : Finset Char) (len : Nat) : Finset (List Char) :=
  (letters ∪ digits).powerset.filter (λ s, s.card = len)

def valid_sequences (seq : List Char) : Bool :=
  let count := seq.count;
  count 'A' >= 2 ∧ count 'A' <= 3 ∧ count 'B' <= 1 ∧ count 'C' <= 1 ∧ count '1' <= 1 ∧ count '2' <= 2 ∧ count '8' <= 1

def filtered_sequences : Finset (List Char) :=
  total_sequences {'A', 'B', 'C'} {'1', '2', '8'} 5 |>.filter valid_sequences

def N : Nat :=
  filtered_sequences.card

theorem identification_tags (N : Nat) : N / 10 = 72 := by
  sorry

end identification_tags_l29_29442


namespace geometric_sequence_common_ratio_l29_29157

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_start : a 1 < 0)
  (h_increasing : ∀ n, a n < a (n + 1)) : 0 < q ∧ q < 1 :=
by
  sorry

end geometric_sequence_common_ratio_l29_29157


namespace infinite_series_sum_eq_3_over_4_l29_29711

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29711


namespace verify_sequence_sum_minimum_ns_value_l29_29882

noncomputable def tangent_line_at (x : ℝ) : ℝ :=
  2 * x - 15

def sequence_an (n : ℕ) : ℝ :=
  2 * n - 15

def sum_sn (n : ℕ) : ℝ :=
  (n * (sequence_an 1 + sequence_an n)) / 2

def product_ns (n : ℕ) : ℝ :=
  n * sum_sn n

theorem verify_sequence_sum (n : ℕ) : 
  sequence_an n = 2 * n - 15 ∧ sum_sn n = n^2 - 14 * n :=
by sorry

theorem minimum_ns_value : 
  ∃ n : ℕ, n > 0 ∧ product_ns n = -405 :=
by sorry

end verify_sequence_sum_minimum_ns_value_l29_29882


namespace series_sum_eq_l29_29536

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29536


namespace real_solutions_eq_cos_l29_29048

theorem real_solutions_eq_cos (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = x / 50) → (∀ x, g x = Real.cos x) → 
  (∃ n, n = 31 ∧ ∀ x ∈ Set.Icc (-50) 50, f x = g x → x ∈ Set.range int.cast) :=
by
  sorry

end real_solutions_eq_cos_l29_29048


namespace infinite_series_sum_eq_3_div_4_l29_29614

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29614


namespace jennifer_initial_pears_l29_29167

def initialPears (P: ℕ) : Prop := (P + 20 + 2 * P - 6 = 44)

theorem jennifer_initial_pears (P: ℕ) (h : initialPears P) : P = 10 := by
  sorry

end jennifer_initial_pears_l29_29167


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29649

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29649


namespace sum_of_series_eq_three_fourths_l29_29589

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29589


namespace series_sum_eq_l29_29530

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29530


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29401

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29401


namespace commute_days_per_week_is_five_l29_29492

def total_commute_one_way : ℝ := 21
def gas_cost_per_gallon : ℝ := 2.50
def car_mileage_per_gallon : ℝ := 30
def weeks_per_month : ℝ := 4
def num_friends : ℝ := 5
def monthly_payment_per_friend : ℝ := 14

def days_per_week_commute (d : ℝ) : Prop :=
  (42 * d * 4 / 30) * gas_cost_per_gallon = num_friends * monthly_payment_per_friend

theorem commute_days_per_week_is_five : ∃ d : ℝ, days_per_week_commute d ∧ d = 5 :=
begin
  use 5,
  simp [total_commute_one_way, gas_cost_per_gallon, car_mileage_per_gallon, weeks_per_month, 
        num_friends, monthly_payment_per_friend],
  sorry
end

end commute_days_per_week_is_five_l29_29492


namespace sum_series_eq_3_over_4_l29_29753

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29753


namespace parabolas_equation_l29_29827

theorem parabolas_equation (vertex_origin : (0, 0) ∈ {(x, y) | y = x^2} ∨ (0, 0) ∈ {(x, y) | x = -y^2})
  (focus_on_axis : ∀ F : ℝ × ℝ, (F ∈ {(x, y) | y = x^2} ∨ F ∈ {(x, y) | x = -y^2}) → (F.1 = 0 ∨ F.2 = 0))
  (through_point : (-2, 4) ∈ {(x, y) | y = x^2} ∨ (-2, 4) ∈ {(x, y) | x = -y^2}) :
  {(x, y) | y = x^2} ∪ {(x, y) | x = -y^2} ≠ ∅ :=
by
  sorry

end parabolas_equation_l29_29827


namespace third_smallest_four_digit_Pascal_triangle_l29_29288

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29288


namespace series_converges_to_three_fourths_l29_29805

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29805


namespace trapezium_area_l29_29035

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) : 
  (1/2) * (a + b) * h = 285 :=
by
  rw [ha, hb, hh]
  norm_num

end trapezium_area_l29_29035


namespace sequence_product_l29_29929

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem sequence_product (a : ℕ → ℝ) (h_seq : sequence a) (h_a5 : a 5 = 4) : 
  a 4 * a 5 * a 6 = 64 := 
by
  sorry

end sequence_product_l29_29929


namespace isabel_total_candy_l29_29934

theorem isabel_total_candy (original additional : ℕ) :
  original = 68 → additional = 25 → original + additional = 93 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end isabel_total_candy_l29_29934


namespace find_second_sum_l29_29463

def total_sum : ℝ := 2730
def interest_rate_first : ℝ := 3 / 100
def interest_rate_second : ℝ := 5 / 100
def time_first : ℝ := 8
def time_second : ℝ := 3

theorem find_second_sum
  (x : ℝ) -- First part of the sum
  (interest_first : ℝ := x * interest_rate_first * time_first)
  (interest_second : ℝ := (total_sum - x) * interest_rate_second * time_second) 
  (h : interest_first = interest_second) : (total_sum - 1050 = 1680) :=
by
  -- Sorry to skip the proof
  sorry

end find_second_sum_l29_29463


namespace a_2_eq_5_an_plus1_eq_3an_minus1_sum_first_n_terms_l29_29012

variable (n : ℕ)

def beautiful_growth_seq : List ℕ := sorry -- Define the beautiful growing sequence (1, 2, ...)

def log₂ (x : ℕ) : ℝ := Real.log x / Real.log 2

def a : ℕ → ℝ
| 0 => log₂ (beautiful_growth_seq.head * beautiful_growth_seq.last)
| (n + 1) => 3 * a n - 1

theorem a_2_eq_5 : a 2 = 5 := sorry

theorem an_plus1_eq_3an_minus1 (n : ℕ) : a (n + 1) = 3 * a n - 1 := sorry

theorem sum_first_n_terms (n : ℕ) : 
    (finset.sum (finset.range n) (λ i, (3^i / (a i * a (i + 1)))) = 1/2 - 2/(3^(n+1) + 1)) := sorry

end a_2_eq_5_an_plus1_eq_3an_minus1_sum_first_n_terms_l29_29012


namespace infinite_series_sum_value_l29_29737

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29737


namespace reciprocal_twice_num_operations_to_revert_81_l29_29258

def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_twice (x : ℝ) (h : x ≠ 0) : reciprocal (reciprocal x) = x :=
by
  unfold reciprocal
  field_simp [h]

theorem num_operations_to_revert_81 : 
  ∃ n : ℕ, n = 2 ∧ ∀(y : ℝ), y = 81 → (nat.iterate reciprocal n y) = y :=
by
  use 2
  split
  { refl },
  { intro y,
    intro hy,
    rw [hy, nat.iterate, nat.iterate, reciprocal_twice 81],
    { exact 81 },
    { norm_num } }

end reciprocal_twice_num_operations_to_revert_81_l29_29258


namespace distribution_ways_5_to_3_l29_29122

noncomputable def num_ways (n m : ℕ) : ℕ :=
  m ^ n

theorem distribution_ways_5_to_3 : num_ways 5 3 = 243 := by
  sorry

end distribution_ways_5_to_3_l29_29122


namespace series_result_l29_29631

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29631


namespace series_result_l29_29642

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29642


namespace sum_series_equals_three_fourths_l29_29549

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29549


namespace positive_integral_solution_l29_29817

theorem positive_integral_solution (n : ℕ) (h_pos : 0 < n)
    (h_eq_num : (∑ i in finset.range n, (2 * i + 1)) = n^2)
    (h_eq_den : (∑ i in finset.range n, (2 * (i + 1))) = n * (n + 1)) :
    (∑ i in finset.range n, (2 * i + 1)) / (∑ i in finset.range n, (2 * (i + 1))) = (114 : ℚ) / (115 : ℚ) → n = 114 :=
by
  sorry

end positive_integral_solution_l29_29817


namespace minimize_sum_of_squares_distances_l29_29116

theorem minimize_sum_of_squares_distances 
  (A B C M D : Point)
  (hA : ¬ collinear A B C) 
  (hMD : M ∈ line B C)
  (a b c : ℝ) 
  (hAD : dist A D = a) 
  (hDB : dist D B = b) 
  (hDC : dist D C = c) : 
  dist D M = (b + c) / 3 :=
begin
  sorry
end

end minimize_sum_of_squares_distances_l29_29116


namespace cube_volume_l29_29238

theorem cube_volume : 
  (∃ (s : ℝ), 12 * s = 48) → 
  (∃ (V : ℝ), V = (4:ℝ)^3 ∧ V = 64) := 
by 
  intro h
  rcases h with ⟨s, hs⟩
  use s^3
  have hs_eq : s = 4 := by linarith
  rw [hs_eq, pow_succ, pow_succ, pow_zero, mul_assoc, mul_assoc, mul_one, pow_succ, pow_zero]
  norm_num
  exact ⟨rfl, by norm_num⟩

end cube_volume_l29_29238


namespace simplify_and_evaluate_expression_l29_29213

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_and_evaluate_expression_l29_29213


namespace AB_perpendicular_plane_PAD_EF_parallel_plane_PAD_l29_29928

variable (P A B C D E F : Type) [Space P A B C D]
variable (PD_perp_ABCD : ∀ P A B C D : Type, is_perpendicular(P, plane(ABCD)))
variable (AD_parallel_BC : ∀ A D B C : Type, is_parallel(AD, BC))
variable (CD_eq_13 : CD.length = 13)
variable (AB_eq_12 : AB.length = 12)
variable (BC_eq_10 : BC.length = 10)
variable (AD_eq_12 : AD.length = 12)
variable (E_midpoint_PB : is_midpoint(E, P, B))
variable (F_midpoint_CD : is_midpoint(F, C, D))

theorem AB_perpendicular_plane_PAD : is_perpendicular(AB, plane(PAD)) := sorry

theorem EF_parallel_plane_PAD : is_parallel(EF, plane(PAD)) := sorry

end AB_perpendicular_plane_PAD_EF_parallel_plane_PAD_l29_29928


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29404

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29404


namespace cube_volume_is_64_l29_29244

def cube_volume_from_edge_sum (edge_sum : ℝ) (num_edges : ℕ) : ℝ :=
  let edge_length := edge_sum / num_edges in
  edge_length ^ 3

theorem cube_volume_is_64 :
  cube_volume_from_edge_sum 48 12 = 64 := by
  sorry

end cube_volume_is_64_l29_29244


namespace nk_bound_l29_29058

theorem nk_bound (k : ℕ) (hk : k ≥ 3) (n_k : ℕ) (A : Set ℤ)
  (h1 : ∀ a ∈ A, ∃ x y ∈ A, n_k ∣ (a - x - y))
  (h2 : ∀ B ⊆ A, B.card ≤ k → n_k ∣ B.sum → False) :
  n_k < (13 / 8) ^ (k + 2) :=
sorry

end nk_bound_l29_29058


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29654

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29654


namespace third_smallest_four_digit_in_pascals_triangle_l29_29293

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29293


namespace sum_series_div_3_powers_l29_29789

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29789


namespace monotonicity_intervals_minimum_m_l29_29879

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  ln x + ln (a * x + 1) - (3 * a / 2) * x + 1

theorem monotonicity_intervals (a : ℝ) :
  (∀ x > 0, 0 < a → f a x = ln x) ∧
  (∀ x ∈ Ioo 0 (1 / a), f a x = ln x ∧ a > 0) ∧
  (∀ x ∈ Ioi (1 / a), f a x = ln x ∧ a > 0) ∧
  (∀ x ∈ Ioo 0 (-2 / (3 * a)), f a x = ln x ∧ a < 0) ∧
  (∀ x ∈ Ioo (-2 / (3 * a)) (-1 / a), f a x = ln x ∧ a < 0) ∧
  (∃ x ∈ Ioi 0, a = 0 ∧ f a x = ln x) :=
sorry

theorem minimum_m (m : ℝ) :
  (∀ x > 0, (a = 2 / 3) → (x * exp (x - 1 / 2) + m ≥ f (2 / 3) x)) → 
  m ≥ ln (2 / 3) :=
sorry

end monotonicity_intervals_minimum_m_l29_29879


namespace Dave_trips_l29_29505

theorem Dave_trips (trays_per_trip : ℕ) (trays_first_table : ℕ) (trays_second_table : ℕ)
  (h1 : trays_per_trip = 9) (h2 : trays_first_table = 17) (h3 : trays_second_table = 55) :
  let trips_first_table := Nat.ceil (trays_first_table / trays_per_trip)
  let trips_second_table := Nat.ceil (trays_second_table / trays_per_trip)
  let total_trips := trips_first_table + trips_second_table in
  total_trips = 9 :=
by
  sorry

end Dave_trips_l29_29505


namespace least_add_to_divisible_by_17_l29_29040

/-- Given that the remainder when 433124 is divided by 17 is 2,
    prove that the least number that must be added to 433124 to make 
    it divisible by 17 is 15. -/
theorem least_add_to_divisible_by_17: 
  (433124 % 17 = 2) → 
  (∃ n, n ≥ 0 ∧ (433124 + n) % 17 = 0 ∧ n = 15) := 
by
  sorry

end least_add_to_divisible_by_17_l29_29040


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29335

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29335


namespace sum_of_series_eq_three_fourths_l29_29588

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29588


namespace number_of_sheep_is_56_l29_29237

def number_of_sheep (S H : ℕ) (food_per_horse total_food : ℕ) (ratio : ℕ × ℕ) : Prop :=
  ratio = (7, 7) ∧ food_per_horse = 230 ∧ total_food = 12880 ∧ 
  H = total_food / food_per_horse ∧ S = H

theorem number_of_sheep_is_56 : ∃ (S : ℕ), ∃ (H : ℕ),
  number_of_sheep S H 230 12880 (7, 7) ∧ S = 56 :=
by 
  use 56
  use 56
  unfold number_of_sheep
  simp
  norm_num
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

end number_of_sheep_is_56_l29_29237


namespace series_sum_l29_29669

open_locale big_operators

theorem series_sum : ∑ k : ℕ in finset.range (n + 1), (k + 1) * (1 / 3)^(k + 1) = 3 / 4 := sorry

end series_sum_l29_29669


namespace average_score_boys_l29_29144

theorem average_score_boys (B : ℝ) (total_students : ℝ) (total_score_class : ℝ)
  (boys : ℝ) (girls : ℝ) (average_score_girls : ℝ) (average_score_class : ℝ) :
  girls = 4 → boys = 12 → average_score_girls = 92 → average_score_class = 86 → total_students = boys + girls →
  total_score_class = average_score_class * total_students →
  12 * B + 4 * 92 = total_score_class → 
  B = 84 :=
begin
  intros h_girls h_boys h_avg_girls h_avg_class h_total_students h_total_score_class h_equation,
  sorry -- this is where the proof would go
end

end average_score_boys_l29_29144


namespace sum_of_series_eq_three_fourths_l29_29576

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29576


namespace ounces_per_cup_l29_29020

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) 
  (h : total_ounces = 264 ∧ total_cups = 33) : total_ounces / total_cups = 8 :=
by
  sorry

end ounces_per_cup_l29_29020


namespace third_smallest_four_digit_Pascal_triangle_l29_29292

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29292


namespace k_valid_iff_l29_29821

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l29_29821


namespace area_of_triangle_XYZ_l29_29156

/-- Given DE = 30 cm, DF = 24 cm, angle D = 90 degrees,
    and triangle DEF is similar to triangle XYZ with a
    scale factor of 3/4, the area of triangle XYZ is 202.5 cm². -/
theorem area_of_triangle_XYZ
  (DE DF : ℝ)
  (hDE : DE = 30)
  (hDF : DF = 24)
  (angleD : Type) (hAngleD : angleD = 90)
  (scale_factor : ℝ) (hScale_factor : scale_factor = (3 / 4)) :
  let Area_DEF := (1 / 2) * DE * DF in
  let Area_XYZ := (scale_factor ^ 2) * Area_DEF in
  Area_XYZ = 202.5 := sorry

end area_of_triangle_XYZ_l29_29156


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29411

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29411


namespace sum_series_eq_3_div_4_l29_29567

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29567


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29330

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29330


namespace series_sum_eq_l29_29524

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29524


namespace find_a_in_terms_of_x_l29_29131

variable (a b x : ℝ)

theorem find_a_in_terms_of_x (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) : a = 3 * x :=
sorry

end find_a_in_terms_of_x_l29_29131


namespace series_converges_to_three_fourths_l29_29800

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29800


namespace infinite_series_sum_eq_l29_29710

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29710


namespace water_rise_calculation_l29_29462

def base_length : ℝ := 60
def base_breadth : ℝ := 30
def cube_edge : ℝ := 30

def volume_of_cube (a : ℝ) : ℝ := a ^ 3
def base_area_of_vessel (l : ℝ) (b : ℝ) : ℝ := l * b
def water_rise (v_cube : ℝ) (a_vessel : ℝ) : ℝ := v_cube / a_vessel

theorem water_rise_calculation :
  water_rise (volume_of_cube cube_edge) (base_area_of_vessel base_length base_breadth) = 15 := by
  sorry

end water_rise_calculation_l29_29462


namespace angle_equality_l29_29933

section
variables {A B C O P D E M : Type*}
variables [field A] [field B] [field C] [field O] [field P] [field D] [field E] [field M]

-- Definitions based on conditions
def triangle_inscribed (A B C O : Type*) : Prop :=
  ∃ (circO : circle O), ∀ (X : Type*), X ∈ {A, B, C} → X ∈ circO

def angle_gt_90 (X Y Z : Type*) : Prop := ∠XYZ > 90

def midpoint (M B C : Type*) : Prop := M = (B + C) / 2

def perp (X Y Z : Type*) : Prop := (Y - X) * (Z - X) = 0

def perpendicular_to_AP (P D E : Type*) : Prop := ∃ (l : line P), l ⊥ AP ∧ D E ∉ {P}

def length_equal (X Y Z W : Type*) : Prop := dist X Y = dist Z W

def parallelogram (A D O E : Type*) : Prop := parallel AD OE ∧ parallel AO DE

-- The main theorem statement
theorem angle_equality
  (h1 : triangle_inscribed A B C O)
  (h2 : angle_gt_90 A B C)
  (h3 : midpoint M B C)
  (h4 : perp P B C)
  (h5 : perpendicular_to_AP P D E)
  (h6 : length_equal B D B P)
  (h7 : length_equal C E C P)
  (h8 : parallelogram A D O E) :
  ∠OPE = ∠AMB :=
sorry

end

end angle_equality_l29_29933


namespace evaluate_series_sum_l29_29603

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29603


namespace round_to_nearest_tenth_l29_29210

theorem round_to_nearest_tenth (x : ℝ) (x_val : x = 24.63521) : 
  let tenths_digit := 3
  let hundredths_digit := 5
  (hundredths_digit >= 5) → (x.roundTo 0.1 = 24.7) :=
  by
    sorry

end round_to_nearest_tenth_l29_29210


namespace series_sum_eq_l29_29535

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29535


namespace possible_value_of_x_l29_29908

theorem possible_value_of_x : 
  ∀ x, (x = 7 ∨ x = 0 ∨ x = 108 ∨ x = 12 ∨ x = 23) → x < 5 → x = 0 :=
by
  intro x h hx
  cases h
  case or.inl { rw [h] at hx, exact absurd hx (by decide) }
  case or.inr h' =>
    cases h'
    case or.inl { rw [h'] }
    case or.inr h' =>
      cases h'
      case or.inl { rw [h'] at hx, exact absurd hx (by decide) }
      case or.inr h' =>
        cases h'
        case or.inl { rw [h'] at hx, exact absurd hx (by decide) }
        case or.inr h' =>
          cases h'
          case or.inl { rw [h'] at hx, exact absurd hx (by decide) }
          case or.inr { rw [h'] at hx, exact absurd hx (by decide) }

end possible_value_of_x_l29_29908


namespace infinite_series_sum_eq_3_div_4_l29_29609

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29609


namespace series_sum_correct_l29_29773

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29773


namespace max_min_values_exponential_function_l29_29041

theorem max_min_values_exponential_function :
  let f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6) in
  ∃ (c d : ℝ), (c = 1) ∧ (d = 1 / Real.exp 8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ c) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ d) ∧
  (∃ x ∈ Set.Icc 0 3, f x = c) ∧
  (∃ x ∈ Set.Icc 0 3, f x = d) :=
by
  let f (x : ℝ) := Real.exp (2 * x^2 - 4 * x - 6)
  have h_deriv : ∀ x, HasDerivAt f ((4 * x - 4) * f x) x := sorry
  have h_cont : ContinuousOn f (Set.Icc 0 3) := sorry
  have h_max : ∃ x ∈ Set.Icc 0 3, IsMaxOn f (Set.Icc 0 3) x := sorry
  cases h_max with xs xs_prop
  have h_min : ∃ x ∈ Set.Icc 0 3, IsMinOn f (Set.Icc 0 3) x := sorry
  cases h_min with xm xm_prop
  have h0 : f 0 = 1 / Real.exp 6 := sorry
  have h1 : f 1 = 1 / Real.exp 8 := sorry
  have h3 : f 3 = 1 := sorry
  use 1, 1 / Real.exp 8
  apply And.intro rfl
  apply And.intro rfl
  apply And.intro
  {
    intros x hx
    interval_cases with x 0 3 1;
    simp only [f, Real.exp_le_exp];
    linarith
  }
  apply And.intro
  {
    intros x hx
    interval_cases with x 0 3 1;
    simp only [f, Real.exp_le_exp];
    linarith
  }
  apply And.intro
  {
    use 3
    apply And.intro
    exact Set.left_mem_Icc.mpr zero_le_three
    exact h3
  }
  {
    use 1
    apply And.intro
    exact Set.right_mem_Icc.mpr zero_le_three
    exact h1
  }

end max_min_values_exponential_function_l29_29041


namespace pythagorean_triple_divisibility_l29_29205

theorem pythagorean_triple_divisibility (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  (∃ x ∈ {3, 4, 5}, x ∣ a) ∧ (∃ x ∈ {3, 4, 5}, x ∣ b) ∧ (∃ x ∈ {3, 4, 5}, x ∣ c) :=
by sorry

end pythagorean_triple_divisibility_l29_29205


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29651

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29651


namespace total_questions_l29_29198

variable (x : ℕ) 

theorem total_questions (score_correct : 104 * 1 : ℝ)
    (score_incorrect : x * 0.25 : ℝ)
    (total_score : 104 - score_incorrect = 100)
    (correct_answers : 104) :
    (correct_answers + x = 120) :=
by
  sorry

end total_questions_l29_29198


namespace sum_series_eq_3_over_4_l29_29758

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29758


namespace find_k_l29_29839

theorem find_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 7 * x - 6 = 0 ↔ (x = 2 ∨ x = -3/2)) →
  k = 14 :=
by
  intro h
  have h1 : k * 4 - 14 - 6 = 0 :=
    by rw [h]; field_simp; norm_num
  have h2 : k * 9/4 + 21/2 - 6 = 0 :=
    by rw [h]; field_simp; norm_num
  linarith
  sorry

end find_k_l29_29839


namespace base_of_isosceles_triangle_l29_29424

-- Definitions based on conditions
def equilateral_perimeter : ℕ := 60
def isosceles_perimeter : ℕ := 55
def equilateral_side : ℕ := equilateral_perimeter / 3

-- The statement of the problem in Lean
theorem base_of_isosceles_triangle : 
  (equilateral_side = 20) ∧ 
  ((isosceles_perimeter = 40 + equilateral_side) →
  (isosceles_perimeter - 2 * equilateral_side = 15)) :=
by {
  -- Verifying the conditions
  have h : equilateral_side = 20 := by norm_num [equilateral_side, equilateral_perimeter],
  split,
  exact h,
  intro h_iso,
  rw h at h_iso,
  norm_num at h_iso,
  sorry
}

end base_of_isosceles_triangle_l29_29424


namespace time_per_lawn_in_minutes_l29_29940

def jason_lawns := 16
def total_hours_cutting := 8
def minutes_per_hour := 60

theorem time_per_lawn_in_minutes : 
  (total_hours_cutting / jason_lawns) * minutes_per_hour = 30 :=
by
  sorry

end time_per_lawn_in_minutes_l29_29940


namespace sum_series_equals_three_fourths_l29_29552

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29552


namespace sum_distances_from_point_to_lines_l29_29952

noncomputable def distance_from_point_to_line (P: ℝ^3) (A B: ℝ^3) : ℝ := sorry

structure Tetrahedron := 
  (A B C D P : ℝ^3)
  (edge_length : ℝ)
  (is_regular : edge_length = dist A B 
                ∧ edge_length = dist A C 
                ∧ edge_length = dist A D 
                ∧ edge_length = dist B C 
                ∧ edge_length = dist B D 
                ∧ edge_length = dist C D 
                ∧ edge_length = 1)

def is_centroid (P A B C D : ℝ^3) : Prop := sorry

theorem sum_distances_from_point_to_lines (T : Tetrahedron) :
  distance_from_point_to_line T.P T.A T.B +
  distance_from_point_to_line T.P T.A T.C +
  distance_from_point_to_line T.P T.A T.D +
  distance_from_point_to_line T.P T.B T.C +
  distance_from_point_to_line T.P T.B T.D +
  distance_from_point_to_line T.P T.C T.D ≥
  (3 / 2) * Real.sqrt 2 ∧ 
  (is_centroid T.P T.A T.B T.C T.D ↔ 
  distance_from_point_to_line T.P T.A T.B + 
  distance_from_point_to_line T.P T.A T.C + 
  distance_from_point_to_line T.P T.A T.D + 
  distance_from_point_to_line T.P T.B T.C + 
  distance_from_point_to_line T.P T.B T.D + 
  distance_from_point_to_line T.P T.C T.D = 
  (3 / 2) * Real.sqrt 2)
:= sorry

end sum_distances_from_point_to_lines_l29_29952


namespace parallel_vectors_eq_l29_29893

theorem parallel_vectors_eq (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (x + 1, -2)) (h_b : b = (-2x, 3)) :
  (a ⊥ b) → x = 3 :=
by
  sorry

end parallel_vectors_eq_l29_29893


namespace sum_series_equals_three_fourths_l29_29557

theorem sum_series_equals_three_fourths :
  (∑' k : ℕ, (k+1) / 3^(k+1)) = 3 / 4 :=
sorry

end sum_series_equals_three_fourths_l29_29557


namespace third_smallest_four_digit_in_pascal_l29_29349

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29349


namespace find_k_in_triangle_l29_29932

theorem find_k_in_triangle
  (k : ℝ)
  (A B C D : Type)
  [plane_geometry A B C D]
  (AB BC AC : ℝ) 
  (h1 : AB = 5)
  (h2 : BC = 7)
  (h3 : AC = 8)
  (BD : ℝ)
  (h4 : BD = k * Real.sqrt 3) :
  k = 5 / 3 := 
sorry

end find_k_in_triangle_l29_29932


namespace infinite_series_sum_eq_3_div_4_l29_29613

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29613


namespace distinct_parenthesizations_l29_29022

theorem distinct_parenthesizations (n : ℕ) (h : n = 3) : ∃ k, k = 4 ∧ 
  let pow := Nat.pow in ∀ f : list ℕ → list ℕ, 
    (f = list.cons n [h⁻³] ∨ f = list.cons (pow n n) [n] ∨ f = list.cons n [list.cons n [n]]) → 
      (number_of_distinct_values_3_3_3 pow f = k) :=
  sorry

end distinct_parenthesizations_l29_29022


namespace number_of_subsets_l29_29015

-- Defining the type of the elements
variable {α : Type*}

-- Statement of the problem in Lean 4
theorem number_of_subsets (s : Finset α) (h : s.card = n) : (Finset.powerset s).card = 2^n := 
sorry

end number_of_subsets_l29_29015


namespace third_smallest_four_digit_in_pascals_triangle_l29_29316

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29316


namespace inequality_not_always_hold_l29_29126

variables {a b c d : ℝ}

theorem inequality_not_always_hold 
  (h1 : a > b) 
  (h2 : c > d) 
: ¬ (a + d > b + c) :=
  sorry

end inequality_not_always_hold_l29_29126


namespace sum_series_eq_3_over_4_l29_29761

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29761


namespace set_of_integers_a_i_closed_under_conditions_gcd_1_l29_29963

theorem set_of_integers_a_i_closed_under_conditions_gcd_1 {n : ℕ} (a : Fin n → ℤ)
  (h_gcd : Finset.gcd (Finset.univ.image a) = 1)
  (S : Set ℤ)
  (h1 : ∀ i, i < n → a i ∈ S)
  (h2 : ∀ i j, i < n → j < n → a i - a j ∈ S)
  (h3 : ∀ x y, x ∈ S → y ∈ S → x + y ∈ S ∧ x - y ∈ S) :
  S = Set.univ := sorry

end set_of_integers_a_i_closed_under_conditions_gcd_1_l29_29963


namespace solution_set_of_fg_l29_29967

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- Define f and g as odd and even functions respectively
variables (f g : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (g_even : ∀ x, g (-x) = g x)

-- Conditions
variable (h_positive_derivative : ∀ {x : ℝ}, x < 0 → (deriv f x) * (g x) + (f x) * (deriv g x) > 0)
variable (g_at_3 : g 3 = 0)

-- Statement to prove
theorem solution_set_of_fg {x : ℝ} : f x * g x < 0 ↔ x ∈ Iio (-3) ∪ Ioo 0 3 :=
sorry

end solution_set_of_fg_l29_29967


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29270

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29270


namespace expression_for_f_in_positive_domain_l29_29900

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def given_f (x : ℝ) : ℝ :=
  if x < 0 then 3 * Real.sin x + 4 * Real.cos x + 1 else 0 -- temp def for Lean proof

theorem expression_for_f_in_positive_domain (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = 3 * Real.sin x + 4 * Real.cos x + 1) :
  ∀ x : ℝ, x > 0 → f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  intros x hx_pos
  sorry

end expression_for_f_in_positive_domain_l29_29900


namespace prod_sum_rel_prime_l29_29236

theorem prod_sum_rel_prime (a b : ℕ) 
  (h1 : a * b + a + b = 119)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 25)
  (h4 : b < 25) : 
  a + b = 27 := 
sorry

end prod_sum_rel_prime_l29_29236


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29652

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29652


namespace eggs_per_snake_l29_29937

-- Define the conditions
def num_snakes : ℕ := 3
def price_regular : ℕ := 250
def price_super_rare : ℕ := 1000
def total_revenue : ℕ := 2250

-- Prove for the number of eggs each snake lays
theorem eggs_per_snake (E : ℕ) 
  (h1 : E * (num_snakes - 1) * price_regular + E * price_super_rare = total_revenue) : 
  E = 2 :=
sorry

end eggs_per_snake_l29_29937


namespace infinite_series_sum_value_l29_29733

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29733


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29331

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29331


namespace find_expectation_Y_l29_29878

variable (X : ℝ → ℝ)
variable (m_x : ℝ → ℝ)
variable (m_Y : ℝ → ℝ)
variable (Y : ℝ → ℝ)
variable (t : ℝ)

-- Given condition
def expectation_X (t : ℝ) : Prop :=
  m_x t = 3 * t^2 + 1

-- Definition of Y(t)
def integral_Y (t : ℝ) : Prop :=
  Y t = ∫ s in 0..t, X s

-- Expected result
def expectation_Y (t : ℝ) : Prop :=
  m_Y t = t^3 + t

theorem find_expectation_Y (h1 : expectation_X t) (h2 : integral_Y t) : expectation_Y t :=
  sorry

end find_expectation_Y_l29_29878


namespace volume_of_cube_l29_29243

theorem volume_of_cube (edge_sum : ℝ) (h_edge_sum : edge_sum = 48) : 
    let edge_length := edge_sum / 12 
    in edge_length ^ 3 = 64 := 
by
  -- condition
  have h1 : edge_sum = 48 := h_edge_sum
  -- definition of edge_length
  let edge_length := edge_sum / 12
  -- compute the edge_length
  have h2 : edge_length = 4 := by linarith
  -- compute the volume of the cube
  have volume := edge_length ^ 3
  -- provide proof that volume is 64
  have h3 : volume = 64 := by linarith; sorry
  exact h3

end volume_of_cube_l29_29243


namespace number_of_bushes_needed_l29_29975

-- Definitions from the conditions
def containers_per_bush : ℕ := 10
def containers_per_zucchini : ℕ := 3
def zucchinis_required : ℕ := 72

-- Statement to prove
theorem number_of_bushes_needed : 
  ∃ bushes_needed : ℕ, bushes_needed = 22 ∧ 
  (zucchinis_required * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush = bushes_needed := 
by
  sorry

end number_of_bushes_needed_l29_29975


namespace pascals_triangle_l29_29207

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def binomial_coefficient (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem pascals_triangle (n k : ℕ) (h : k ≤ n) : 
  binomial_coefficient n k = factorial n / (factorial k * factorial (n - k)) :=
begin
  sorry
end

end pascals_triangle_l29_29207


namespace general_term_sequence_sum_of_cn_l29_29971

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_of_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n)
  (ha : ∀ n, a n = n)
  (hc_odd : ∀ n, c (2 * n - 1) = a (2 * n))
  (hc_even : ∀ n, c (2 * n) = 3 * 2^(a (2 * n - 1)) + 1) :
  ∀ n, T (2 * n) = 2^(2 * n + 1) + n^2 + 2 * n - 2 :=
by
  sorry

end general_term_sequence_sum_of_cn_l29_29971


namespace find_extrema_l29_29109

noncomputable def expr (x y z : ℝ) :=
  x^3 + 2 * y^2 + (10 / 3) * z

theorem find_extrema {x y z : ℝ} (h : x + y + z = 1) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ (max min : ℝ), 
    max = (10 / 3) ∧ 
    min = (14 / 27) ∧ 
    (∀ (a b c : ℝ), a + b + c = 1 → 0 ≤ a → 0 ≤ b → 0 ≤ c → expr a b c ≤ max) ∧
    (∀ (a b c : ℝ), a + b + c = 1 → 0 ≤ a → 0 ≤ b → 0 ≤ c → min ≤ expr a b c) :=
begin
  use (10 / 3),
  use (14 / 27),
  split,
  { refl },
  split,
  { refl },
  split,
  { intros a b c hsum ha hb hc,
    sorry },
  { intros a b c hsum ha hb hc,
    sorry },
end

end find_extrema_l29_29109


namespace proof_complex_power_elements_l29_29873

def complex_equiv_proof_problem : Prop :=
  ∃! (S : set ℂ), (∃ n ∈ (Nat → ℕ), S = {z | ∃ n : ℕ, n > 0 ∧ z = Complex.I ^ n}) ∧ S.card = 4

theorem proof_complex_power_elements : complex_equiv_proof_problem :=
by
  sorry

end proof_complex_power_elements_l29_29873


namespace sum_geometric_series_l29_29685

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29685


namespace infinite_series_sum_eq_3_div_4_l29_29616

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29616


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29279

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29279


namespace article_initial_cost_l29_29469

theorem article_initial_cost (x : ℝ) (h : 0.44 * x = 4400) : x = 10000 :=
by
  sorry

end article_initial_cost_l29_29469


namespace fliers_left_l29_29419

theorem fliers_left (initial_fliers : ℕ) (frac_morning : ℚ) (frac_afternoon : ℚ) 
  (h1 : initial_fliers = 4200)
  (h2 : frac_morning = 1 / 3) 
  (h3 : frac_afternoon = 2 / 5) :
  let morning_fliers := initial_fliers * frac_morning
      remaining_fliers_after_morning := initial_fliers - morning_fliers
      afternoon_fliers := remaining_fliers_after_morning * frac_afternoon
      remaining_fliers_after_afternoon := remaining_fliers_after_morning - afternoon_fliers
  in remaining_fliers_after_afternoon = 1680 := 
by
  -- The proofs will be inserted here
  sorry

end fliers_left_l29_29419


namespace orthocenter_projection_l29_29521

theorem orthocenter_projection (A B C D : Point) [geometry D3] :
  perpendicular D (face A B C) →
  let H₁ := orthocenter A B C in
  let H₂ := orthocenter B C D in
  let proj_H₁ := projection_plane H₁ (plane B C D) in
  proj_H₁ = H₂ :=
by sorry

end orthocenter_projection_l29_29521


namespace relationship_of_f_values_l29_29091

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for the actual function 

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (-x + 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a < b → f a < f b

theorem relationship_of_f_values (h1 : is_increasing f 0 2) (h2 : is_even f) :
  f (5/2) > f 1 ∧ f 1 > f (7/2) :=
sorry -- proof goes here

end relationship_of_f_values_l29_29091


namespace series_converges_to_three_fourths_l29_29809

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29809


namespace exists_adj_diff_gt_3_max_min_adj_diff_l29_29259
-- Import needed libraries

-- Definition of the given problem and statement of the parts (a) and (b)

-- Part (a)
theorem exists_adj_diff_gt_3 (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∃ i j : Fin 18, adj i j ∧ |arrangement i - arrangement j| > 3) :=
sorry

-- Part (b)
theorem max_min_adj_diff (arrangement : Fin 18 → Fin 18) (adj : Fin 18 → Fin 18 → Prop) :
  (∀ i j : Fin 18, adj i j → i ≠ j) →
  (∀ i j : Fin 18, adj i j → |arrangement i - arrangement j| ≥ 6) :=
sorry

end exists_adj_diff_gt_3_max_min_adj_diff_l29_29259


namespace range_of_a_plus_c_l29_29864

-- Let a, b, c be the sides of the triangle opposite to angles A, B, and C respectively.
variable (a b c A B C : ℝ)

-- Given conditions
variable (h1 : b = Real.sqrt 3)
variable (h2 : (2 * c - a) / b * Real.cos B = Real.cos A)
variable (h3 : 0 < A ∧ A < Real.pi / 2)
variable (h4 : 0 < B ∧ B < Real.pi / 2)
variable (h5 : 0 < C ∧ C < Real.pi / 2)
variable (h6 : A + B + C = Real.pi)

-- The range of a + c
theorem range_of_a_plus_c (a b c A B C : ℝ) (h1 : b = Real.sqrt 3)
  (h2 : (2 * c - a) / b * Real.cos B = Real.cos A) (h3 : 0 < A ∧ A < Real.pi / 2)
  (h4 : 0 < B ∧ B < Real.pi / 2) (h5 : 0 < C ∧ C < Real.pi / 2) (h6 : A + B + C = Real.pi) :
  a + c ∈ Set.Ioc (Real.sqrt 3) (2 * Real.sqrt 3) :=
  sorry

end range_of_a_plus_c_l29_29864


namespace a5_is_16_S8_is_255_l29_29077

-- Define the sequence
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * seq n

-- Definition of the geometric sum
def geom_sum (n : ℕ) : ℕ :=
  (2 ^ (n + 1) - 1)

-- Prove that a₅ = 16
theorem a5_is_16 : seq 5 = 16 :=
  by
  unfold seq
  sorry

-- Prove that the sum of the first 8 terms, S₈ = 255
theorem S8_is_255 : geom_sum 7 = 255 :=
  by 
  unfold geom_sum
  sorry

end a5_is_16_S8_is_255_l29_29077


namespace maze_side_length_correct_l29_29063

noncomputable def maze_side_length : ℕ :=
  let n := 21 in
  if 2*n*(n-1) - 400 + 1 = n^2 then
    n
  else
    0 -- since n cannot be 0, this case wouldn't occur

theorem maze_side_length_correct : maze_side_length = 21 :=
by {
  -- let n be the side length of the square
  let n := 21,
  -- check the derived equation holds for n = 21
  have h : 2 * n * (n - 1) - 400 + 1 = n^2,
  { calc
      2 * 21 * (21 - 1) - 400 + 1
          = 2 * 21 * 20 - 400 + 1     : by refl
      ... = 2 * 21 * 20 - 400 + 1     : by refl
      ... = 840 - 400 + 1         : by refl
      ... = 441                 : by refl
      ... = 21 * 21          : by rw mul_self_eq ↑21
      ... = 21^2                  : by refl
  },
  exact h,
}

end maze_side_length_correct_l29_29063


namespace cards_in_unfilled_box_l29_29943

theorem cards_in_unfilled_box (total_cards : ℕ) (box_capacity : ℕ) (h1 : total_cards = 94) (h2 : box_capacity = 8) : total_cards % box_capacity = 6 :=
by {
  have h3 : 94 % 8 = 6 := rfl,
  rw [h1, h2] at h3,
  exact h3,
}

end cards_in_unfilled_box_l29_29943


namespace infinite_series_sum_eq_3_div_4_l29_29618

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29618


namespace solution_set_of_inequality_l29_29182

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality :
  (∀ x > 0, f(x) + x * (deriv f x) > 0) ∧ (f 1 = 2) →
  {x : ℝ | 0 < x ∧ f(x) < 2 / x} = set.Ioo 0 1 :=
by
  sorry

end solution_set_of_inequality_l29_29182


namespace sin_cos_sum_value_l29_29082

theorem sin_cos_sum_value (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2)
    (h3 : sin(2 * x - π / 4) = - (√2 / 10)) :
    sin x + cos x = 2 * √10 / 5 :=
by
  sorry

end sin_cos_sum_value_l29_29082


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29406

theorem third_smallest_four_digit_number_in_pascals_triangle : 
  ∃ n, is_four_digit (binomial n k) ∧ third_smallest n = 1002 :=
by
  assume P : (∀ k n, 0 ≤ k ∧ k ≤ n → ∃ m, m = binomial n k)
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29406


namespace sum_series_eq_3_div_4_l29_29564

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29564


namespace led_messages_count_l29_29252

/-- There are 7 LEDs in a row that can emit red or green light. 
  Exactly 3 LEDs are lit at any time, and no two adjacent LEDs can be lit simultaneously.
  Prove the number of different messages this row of LEDs can represent is 80. -/
theorem led_messages_count :
  let num_leds := 7
  let colors := 2
  let lit_leds := 3
  let slots := 5
  let combinations := Nat.choose slots lit_leds
  let color_permutations := colors ^ lit_leds
  let total_messages := combinations * color_permutations
  total_messages = 80 :=
by
  have h1 : Nat.choose 5 3 = 10 := sorry
  have h2 : 2 ^ 3 = 8 := by norm_num
  have h3 : 10 * 8 = 80 := by norm_num
  rw [←h1, ←h2, ←h3]
  rfl

end led_messages_count_l29_29252


namespace car_dealership_l29_29478

variable (sportsCars : ℕ) (sedans : ℕ) (trucks : ℕ)

theorem car_dealership (h1 : 3 * sedans = 5 * sportsCars) 
  (h2 : 3 * trucks = 3 * sportsCars) 
  (h3 : sportsCars = 45) : 
  sedans = 75 ∧ trucks = 45 := by
  sorry

end car_dealership_l29_29478


namespace C3PO_Optimal_Play_Wins_l29_29222

def initial_number : List ℕ := [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

-- Conditions for the game
structure GameConditions where
  number : List ℕ
  robots : List String
  cannot_swap : List (ℕ × ℕ) -- Pair of digits that cannot be swapped again
  cannot_start_with_zero : Bool
  c3po_starts : Bool

-- Define the initial conditions
def initial_conditions : GameConditions :=
{
  number := initial_number,
  robots := ["C3PO", "R2D2"],
  cannot_swap := [],
  cannot_start_with_zero := true,
  c3po_starts := true
}

-- Define the winning condition for C3PO
def C3PO_wins : Prop :=
  ∀ game : GameConditions, game = initial_conditions → ∃ is_c3po_winner : Bool, is_c3po_winner = true

-- The theorem statement
theorem C3PO_Optimal_Play_Wins : C3PO_wins :=
by
  sorry

end C3PO_Optimal_Play_Wins_l29_29222


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29645

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29645


namespace part1_part2_part3_l29_29099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 - Real.log x) * (x - Real.log x) + 1

variable {a : ℝ}

-- Prove that for all x > 0, if ax^2 > ln x, then f(x) ≥ ax^2 - ln x + 1
theorem part1 (h : ∀ x > 0, a*x^2 > Real.log x) (x : ℝ) (hx : x > 0) :
  f a x ≥ a*x^2 - Real.log x + 1 := sorry

-- Find the maximum value of a given there exists x₀ ∈ (0, +∞) where f(x₀) = 1 + x₀ ln x₀ - ln² x₀
theorem part2 (h : ∃ x₀ > 0, f a x₀ = 1 + x₀ * Real.log x₀ - (Real.log x₀)^2) :
  a ≤ 1 / Real.exp 1 := sorry

-- Prove that for all 1 < x < 2, we have f(x) > ax(2-ax)
theorem part3 (h : ∀ x, 1 < x ∧ x < 2) (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  f a x > a * x * (2 - a * x) := sorry

end part1_part2_part3_l29_29099


namespace evaluate_series_sum_l29_29600

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29600


namespace sum_series_eq_3_over_4_l29_29747

theorem sum_series_eq_3_over_4 :
  (∑' k: ℕ, (k + 1) / (3:ℚ)^(k+1)) = 3 / 4 := sorry

end sum_series_eq_3_over_4_l29_29747


namespace equilateral_tris_parallel_l29_29522

theorem equilateral_tris_parallel (A B C P Q R : Point)
  (hABC : IsEquilateralTriangle A B C)
  (hPQR : IsEquilateralTriangle P Q R)
  (hC_on_PQ : LiesOnSegment C P Q)
  (hR_on_AB : LiesOnSegment R A B) :
  Parallel (line_through A P) (line_through B Q) :=
sorry

end equilateral_tris_parallel_l29_29522


namespace third_smallest_four_digit_in_pascals_triangle_l29_29321

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, n > 0 ∧ k > 0 ∧ Nat.choose n k = 1035 ∧ 
  (∀ (m l : ℕ), m > 0 ∧ l > 0 ∧ Nat.choose m l < 1000 → Nat.choose m l < 1035) ∧
  (∀ (p q : ℕ), p > 0 ∧ q > 0 ∧ 1000 ≤ Nat.choose p q ∧ Nat.choose p q < 1035 → 
    (Nat.choose m l = 1000 ∨ Nat.choose m l = 1001 ∨ Nat.choose m l = 1035)) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29321


namespace third_smallest_four_digit_in_pascals_triangle_l29_29355

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29355


namespace limit_x_sin_x_log_x_l29_29816

open Real Filter

variable {x : ℝ}

theorem limit_x_sin_x_log_x : tendsto (fun x => (x - sin x) * log x) (𝓝[>] 0) (𝓝 0) :=
by sorry

end limit_x_sin_x_log_x_l29_29816


namespace gcd_n4_plus_125_n_plus_5_l29_29053

theorem gcd_n4_plus_125_n_plus_5 (n : ℕ) (h1 : 0 < n) (h2 : n % 7 ≠ 0) :
  ∃ d ∈ {1, 3}, gcd (n^4 + 125) (n + 5) = d := sorry

end gcd_n4_plus_125_n_plus_5_l29_29053


namespace parts_of_alloys_l29_29476

def ratio_of_metals_in_alloy (a1 a2 a3 b1 b2 : ℚ) (x y : ℚ) : Prop :=
  let first_metal := (1 / a3) * x + (a1 / b2) * y
  let second_metal := (2 / a3) * x + (b1 / b2) * y
  (first_metal / second_metal) = (17 / 27)

theorem parts_of_alloys
  (x y : ℚ)
  (a1 a2 a3 b1 b2 : ℚ)
  (h1 : a1 = 1)
  (h2 : a2 = 2)
  (h3 : a3 = 3)
  (h4 : b1 = 2)
  (h5 : b2 = 5)
  (h6 : ratio_of_metals_in_alloy a1 a2 a3 b1 b2 x y) :
  x = 9 ∧ y = 35 :=
sorry

end parts_of_alloys_l29_29476


namespace remainder_n_sq_plus_3n_5_mod_25_l29_29905

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l29_29905


namespace math_problem_l29_29263

theorem math_problem :
  8 * (1/4 - 1/3 + 1/6)⁻¹ = 96 :=
sorry

end math_problem_l29_29263


namespace infinite_series_sum_eq_3_div_4_l29_29617

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29617


namespace evaluate_series_sum_l29_29597

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29597


namespace jack_jogging_speed_needed_l29_29163

noncomputable def jack_normal_speed : ℝ :=
  let normal_melt_time : ℝ := 10
  let faster_melt_factor : ℝ := 0.75
  let adjusted_melt_time : ℝ := normal_melt_time * faster_melt_factor
  let adjusted_melt_time_hours : ℝ := adjusted_melt_time / 60
  let distance_to_beach : ℝ := 2
  let required_speed : ℝ := distance_to_beach / adjusted_melt_time_hours
  let slope_reduction_factor : ℝ := 0.8
  required_speed / slope_reduction_factor

theorem jack_jogging_speed_needed
  (normal_melt_time : ℝ := 10) 
  (faster_melt_factor : ℝ := 0.75) 
  (distance_to_beach : ℝ := 2) 
  (slope_reduction_factor : ℝ := 0.8) :
  jack_normal_speed = 20 := 
by
  sorry

end jack_jogging_speed_needed_l29_29163


namespace angle_between_reflected_beams_l29_29446

variables {α β : ℝ}
-- Conditions: α and β are acute angles.
def is_acute (θ : ℝ) : Prop := θ > 0 ∧ θ < π / 2

-- Assuming α and β are acute angles
theorem angle_between_reflected_beams (hα : is_acute α) (hβ : is_acute β) :
  ∃ γ : ℝ, γ = arccos (1 - 2 * (sin α) ^ 2 * (sin β) ^ 2) :=
begin
  sorry
end

end angle_between_reflected_beams_l29_29446


namespace third_smallest_four_digit_in_pascal_triangle_l29_29369

-- Defining Pascal's triangle
def pascal (n k : ℕ) : ℕ := if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

/--
Condition:
1. In Pascal's triangle, each number is the sum of the two numbers directly above it.
2. Four-digit numbers start appearing regularly in higher rows (from row 45 onwards).
-/
theorem third_smallest_four_digit_in_pascal_triangle : 
  ∃ n k : ℕ, pascal n k = 1002 ∧ ∀ m l : ℕ, (pascal m l < 1000 ∨ pascal m l = 1001) → (m < n ∨ (m = n ∧ l < k)) :=
sorry

end third_smallest_four_digit_in_pascal_triangle_l29_29369


namespace series_result_l29_29639

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29639


namespace time_to_run_above_tree_l29_29465

-- Defining the given conditions
def tiger_length : ℕ := 5
def tree_trunk_length : ℕ := 20
def time_to_pass_grass : ℕ := 1

-- Defining the speed of the tiger
def tiger_speed : ℕ := tiger_length / time_to_pass_grass

-- Defining the total distance the tiger needs to run
def total_distance : ℕ := tree_trunk_length + tiger_length

-- The theorem stating the time it takes for the tiger to run above the fallen tree trunk
theorem time_to_run_above_tree :
  (total_distance / tiger_speed) = 5 :=
by
  -- Trying to fit the solution steps as formal Lean statements
  sorry

end time_to_run_above_tree_l29_29465


namespace third_smallest_four_digit_in_pascal_l29_29344

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29344


namespace p_and_q_necessary_not_sufficient_l29_29110

variable (a m x : ℝ) (P Q : Prop)

def p (a m : ℝ) : Prop := a < 0 ∧ m^2 - 4 * a * m + 3 * a^2 < 0

def q (m : ℝ) : Prop := ∀ x > 0, x + 4 / x ≥ 1 - m

theorem p_and_q_necessary_not_sufficient :
  (∀ (a m : ℝ), p a m → q m) ∧ (∀ a : ℝ, -1 ≤ a ∧ a < 0) :=
sorry

end p_and_q_necessary_not_sufficient_l29_29110


namespace infinite_series_sum_eq_3_div_4_l29_29612

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29612


namespace trigonometric_identity_l29_29124

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 2 * Real.cos α) :
  Real.sin α ^ 2 + 2 * Real.cos α ^ 2 = 6 / 5 := 
by 
  sorry

end trigonometric_identity_l29_29124


namespace reciprocals_sum_of_roots_l29_29190

theorem reciprocals_sum_of_roots (r s γ δ : ℚ) (h1 : 7 * r^2 + 5 * r + 3 = 0) (h2 : 7 * s^2 + 5 * s + 3 = 0) (h3 : γ = 1/r) (h4 : δ = 1/s) :
  γ + δ = -5/3 := 
  by 
    sorry

end reciprocals_sum_of_roots_l29_29190


namespace series_result_l29_29636

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29636


namespace measure_angle_and_area_l29_29955

/-- Given a triangle ABC with sides a, b, and c opposite angles A, B, and C, 
    and the condition √3c = √3a cos B - a sin B, verify the measure of angle A 
    is 2π/3. Additionally, if the angle bisector of A intersects BC at D with 
    AD = 3, verify the minimum area of triangle ABC is 9√3. 
-/
theorem measure_angle_and_area (a b c A B C : ℝ) (AD : ℝ)
(h1 : √3 * c = √3 * a * Real.cos B - a * Real.sin B)
(h2 : A = 2 * Real.pi / 3)
(h3 : AD = 3) :
  A = 2 * Real.pi / 3 ∧ (∃ (area : ℝ), area = 9 * √3) :=
sorry

end measure_angle_and_area_l29_29955


namespace max_true_statements_l29_29181

theorem max_true_statements 
  (a b : ℝ) 
  (cond1 : a > 0) 
  (cond2 : b > 0) : 
  ( 
    ( (1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( (1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
  ) 
→ 
  (true ∧ true ∧ true ∧ true → 4 = 4) :=
sorry

end max_true_statements_l29_29181


namespace series_sum_eq_l29_29531

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end series_sum_eq_l29_29531


namespace third_smallest_four_digit_number_in_pascals_triangle_l29_29329

theorem third_smallest_four_digit_number_in_pascals_triangle : 
    (∃ n, binomial n (n - 2) = 1002 ∧ binomial (n-1) (n - 1 - 2) = 1001 ∧ binomial (n-2) (n - 2 - 2) = 1000) :=
by
  sorry

end third_smallest_four_digit_number_in_pascals_triangle_l29_29329


namespace bruce_buys_crayons_l29_29485

theorem bruce_buys_crayons (packs_price : ℕ) (books_price : ℕ) (calculators_price : ℕ)
    (bruce_money : ℕ) (bags_price : ℕ) (bags_bought : ℕ) (books_bought : ℕ) (calculators_bought : ℕ)
    (packs_of_crayons : ℕ) :
    packs_price = 5 →
    books_price = 5 →
    calculators_price = 5 →
    bruce_money = 200 →
    bags_price = 10 →
    bags_bought = 11 →
    books_bought = 10 →
    calculators_bought = 3 →
    packs_of_crayons = (bruce_money - (books_bought * books_price + calculators_bought * calculators_price) - (bags_bought * bags_price)) / packs_price →
    packs_of_crayons = 5 :=
by
  intros _
  intros _
  intros _
  intros _
  intros _
  intros _
  intros _
  intros _
  intros hp
  rw hp
  sorry

end bruce_buys_crayons_l29_29485


namespace percentage_second_question_correct_l29_29903

theorem percentage_second_question_correct (a b c : ℝ) 
  (h1 : a = 0.75) (h2 : b = 0.20) (h3 : c = 0.50) :
  (1 - b) - (a - c) + c = 0.55 :=
by
  sorry

end percentage_second_question_correct_l29_29903


namespace sum_geometric_series_l29_29691

noncomputable def geometric_sum : Real := ∑' k : Nat, k / 3^k

theorem sum_geometric_series : geometric_sum = 3 / 4 := by
  sorry

end sum_geometric_series_l29_29691


namespace problem_statement_l29_29986

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 1

-- Conditions about the roots and other properties
def A := f 0 * f 2 > 0
def B := f 0 * f 2 < 0
def C := ∀ x : ℝ, x < 0 → f(x) ≥ f(x + 1)
def D := ¬ A ∧ ¬ B ∧ ¬ C

theorem problem_statement : D :=
by
  -- Include the script of proof steps here
  sorry

end problem_statement_l29_29986


namespace part1_part2_part1_implies_ellipse_l29_29097

variables {x y x0 y0 : ℝ}

-- Conditions
def ellipse_1 (x y : ℝ) : Prop := 8 * x^2 / 81 + y^2 / 36 = 1
def point_M_on_ellipse : Prop := y0 = 2 ∧ ellipse_1 x0 y0 ∧ x0 < 0
def ellipse_2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def new_ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / (a^2 - 5) = 1
def point_M : Prop := x0 = -3 ∧ y0 = 2

-- Theorem Statements
theorem part1 : point_M_on_ellipse → x0 = -3 :=
sorry

theorem part2 : ellipse_2 x y → x0 = -3 → y0 = 2 → (∃ a, a^2 > 5 ∧ new_ellipse a (-3) 2) :=
sorry

/-- The final desired ellipse equation with the given conditions -/
def final_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

theorem part1_implies_ellipse :
  point_M_on_ellipse →
  ellipse_2 x y →
  final_ellipse (-3) 2 :=
sorry

end part1_part2_part1_implies_ellipse_l29_29097


namespace sum_of_series_eq_three_fourths_l29_29580

theorem sum_of_series_eq_three_fourths : 
  (∑ k in (Set.Ioc 0 nat.inf), k / (3 : ℝ)^k) = 3 / 4 := 
sorry

end sum_of_series_eq_three_fourths_l29_29580


namespace third_smallest_four_digit_Pascal_triangle_l29_29283

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29283


namespace series_sum_correct_l29_29771

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_sum_correct : series_sum = 3 / 4 :=
sorry

end series_sum_correct_l29_29771


namespace sum_series_div_3_powers_l29_29780

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l29_29780


namespace third_smallest_four_digit_in_pascals_triangle_l29_29378

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29378


namespace no_valid_angles_l29_29014

open Real

theorem no_valid_angles (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 2 * π)
    (h3 : ∀ k : ℤ, θ ≠ k * (π / 2))
    (h4 : cos θ * tan θ = sin θ ^ 3) : false :=
by
  -- The proof goes here
  sorry

end no_valid_angles_l29_29014


namespace third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29277

theorem third_smallest_four_digit_in_pascals_triangle_is_1002 :
  ∃ n k, 1000 <= binomial n k ∧ binomial n k <= 9999 ∧ (third_smallest (λ n k, 1000 <= binomial n k ∧ binomial n k <= 9999) = 1002) :=
sorry

end third_smallest_four_digit_in_pascals_triangle_is_1002_l29_29277


namespace problem_statement_l29_29958

variables {α : Type*} [Nonempty α] [LinearOrder α] 
variables {x1 x2 : α} {λ : ℝ}
variable (r : ℚ)

noncomputable def is_convex (f : α → α) :=
  ∀ x1 x2 : α, ∀ λ ∈ Icc (0:ℝ) 1, f (λ * x1 + (1 - λ) * x2) ≤ λ * f x1 + (1 - λ) * f x2

noncomputable def is_concave (f : α → α) :=
  ∀ x1 x2 : α, ∀ λ ∈ Icc (0:ℝ) 1, f (λ * x1 + (1 - λ) * x2) ≥ λ * f x1 + (1 - λ) * f x2

theorem problem_statement (r : ℚ) :
  (r > 1 ∨ r < 0 → is_convex (λ x : ℝ, x^r)) ∧
  (0 < r ∧ r < 1 → is_concave (λ x : ℝ, x^r)) ∧
  (r < s → ∀ x1 x2 : α, power_mean r x1 x2 < power_mean s x1 x2) ∧
  (∀ x1 x2 : α, sqrt(x1 * x2) < power_mean r x1 x2 ∧ power_mean (-r) x1 x2 < sqrt(x1 * x2)) :=
sorry

end problem_statement_l29_29958


namespace max_min_values_function_l29_29043

noncomputable def f (x : ℝ) : ℝ := real.exp (2 * x ^ 2 - 4 * x - 6)

theorem max_min_values_function :
  (∀ x ∈ set.Icc 0 3, f x ≤ 1) ∧ f 3 = 1 ∧
  (∀ x ∈ set.Icc 0 3, f x ≥ real.exp (-8)) ∧ f 1 = real.exp (-8) :=
by
  sorry

end max_min_values_function_l29_29043


namespace third_smallest_four_digit_in_pascals_triangle_l29_29301

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29301


namespace trapezium_area_l29_29033

-- Define the lengths of the parallel sides and the distance between them
def side_a : ℝ := 20
def side_b : ℝ := 18
def height : ℝ := 15

-- Define the formula for the area of a trapezium
def area_trapezium (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem
theorem trapezium_area :
  area_trapezium side_a side_b height = 285 :=
by
  sorry

end trapezium_area_l29_29033


namespace ode_solution_l29_29060

-- Define function f and its Fourier transform S
def f (x : ℝ) : ℂ := sorry -- placeholder for the actual function
def Sf (x : ℝ) : ℂ := ∫ u in -∞..+∞, complex.exp (2 * real.pi * complex.I * u * x) * f u

-- Define specific function f_k
def f_k (k : ℕ) (x : ℝ) : ℝ := (1 + x^2) ^ (-1 - k)

-- Define the Fourier transform for f_k
def Sf_k (k : ℕ) (x : ℝ) : ℂ := ∫ u in -∞..+∞, complex.exp (2 * real.pi * complex.I * u * x) * f_k k u

-- Define the function y in terms of Sf_k
def y (k : ℕ) (x : ℝ) := Sf_k k x

-- Define the constants c_1 and c_2 based on k
def c_1 (k : ℕ) : ℝ := -2 * k
def c_2 (k : ℕ) : ℝ := -4 * real.pi^2

-- State the ODE problem
theorem ode_solution (k : ℕ) (h_k : k ≥ 1) :
  ∀ x : ℝ, x * (differential 2 (y k x)) + c_1 k * (differential 1 (y k x)) + c_2 k * x * (y k x) = 0 :=
sorry

end ode_solution_l29_29060


namespace joan_seashells_l29_29942

variable (seashells_found_by_joan seashells_found_by_jessica : ℕ)
variable (total_seashells_found : ℕ)

theorem joan_seashells (h1 : seashells_found_by_jessica = 8)
                       (h2 : total_seashells_found = 14)
  : seashells_found_by_joan = total_seashells_found - seashells_found_by_jessica :=
by
  dsimp [seashells_found_by_joan, total_seashells_found, seashells_found_by_jessica] at *
  rw [h1, h2]
  exact rfl

#check joan_seashells

end joan_seashells_l29_29942


namespace two_legged_birds_count_l29_29480

def count_birds (b m i : ℕ) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 6 * i = 680 → b = 280

theorem two_legged_birds_count : ∃ b m i : ℕ, count_birds b m i :=
by
  have h1 : count_birds 280 0 20 := sorry
  exact ⟨280, 0, 20, h1⟩

end two_legged_birds_count_l29_29480


namespace cameraYPriceDifference_l29_29875

noncomputable def budgetBuysPrice : ℝ := 0.85 * 59.99
noncomputable def valueMartPrice : ℝ := 59.99 - 10

noncomputable def priceDifferenceInCents : ℕ := 
  (⟨budgetBuysPrice.round, sorry⟩ : ℕ) - (⟨valueMartPrice.round, sorry⟩ : ℕ)

theorem cameraYPriceDifference :
  priceDifferenceInCents = 101 :=
sorry

end cameraYPriceDifference_l29_29875


namespace pascal_third_smallest_four_digit_number_l29_29397

theorem pascal_third_smallest_four_digit_number :
  (∃ n k: ℕ, 1000 ≤ binom n k ∧ binom n k ≤ 1002) ∧ ¬(∃ n k: ℕ, 1003 ≤ binom n k) :=
sorry

end pascal_third_smallest_four_digit_number_l29_29397


namespace sum_of_first_six_terms_geometric_sequence_l29_29834

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l29_29834


namespace cone_lateral_surface_area_l29_29069

/-- Given a cone with a base radius of 3 cm and a slant height of 10 cm,
    prove that the lateral surface area of the cone is 30π cm². -/
theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 3) (h_l : l = 10) :
  π * r * l = 30 * π :=
by {
  rw [h_r, h_l],
  norm_num,
  ring,
}

end cone_lateral_surface_area_l29_29069


namespace kate_change_is_correct_l29_29949

-- Define prices of items
def gum_price : ℝ := 0.89
def chocolate_price : ℝ := 1.25
def chips_price : ℝ := 2.49

-- Define sales tax rate
def tax_rate : ℝ := 0.06

-- Define the total money Kate gave to the clerk
def payment : ℝ := 10.00

-- Define total cost of items before tax
def total_before_tax := gum_price + chocolate_price + chips_price

-- Define the sales tax
def sales_tax := tax_rate * total_before_tax

-- Define the correct answer for total cost
def total_cost := total_before_tax + sales_tax

-- Define the correct amount of change Kate should get back
def change := payment - total_cost

theorem kate_change_is_correct : abs (change - 5.09) < 0.01 :=
by
  sorry

end kate_change_is_correct_l29_29949


namespace third_smallest_four_digit_in_pascals_triangle_l29_29387

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29387


namespace third_smallest_four_digit_in_pascal_l29_29347

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def pascal (n k : ℕ) : ℕ := Nat.choose n k

theorem third_smallest_four_digit_in_pascal :
  ∃ n k : ℕ, is_four_digit (pascal n k) ∧ (pascal n k = 1002) :=
sorry

end third_smallest_four_digit_in_pascal_l29_29347


namespace completing_square_correct_l29_29992

theorem completing_square_correct :
  ∀ x : ℝ, (x^2 - 4 * x + 2 = 0) ↔ ((x - 2)^2 = 2) := 
by
  intros x
  sorry

end completing_square_correct_l29_29992


namespace loaves_of_bread_l29_29449

theorem loaves_of_bread (slices_in_loaf : ℕ) (friends : ℕ) (slices_per_friend : ℕ) (total_slices : ℕ) (loaves : ℕ) :
  slices_in_loaf = 15 →
  friends = 10 →
  slices_per_friend = 6 →
  total_slices = friends * slices_per_friend →
  loaves = total_slices / slices_in_loaf →
  loaves = 4 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  have : total_slices = 60 := h4
  rw this at h5
  have : loaves = 4 := h5
  exact this

end loaves_of_bread_l29_29449


namespace sum_series_eq_3_div_4_l29_29573

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l29_29573


namespace intersection_in_fourth_quadrant_l29_29924

theorem intersection_in_fourth_quadrant (a : ℝ) : 
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ x > 0 ∧ y < 0) → a > 1 / 2 := 
by 
  sorry

end intersection_in_fourth_quadrant_l29_29924


namespace series_converges_to_three_fourths_l29_29804

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end series_converges_to_three_fourths_l29_29804


namespace perp_to_parallel_lines_l29_29814

/-- If a line is perpendicular to one of two parallel lines, then it must be perpendicular to the other. -/
theorem perp_to_parallel_lines (l m n : ℝ^3)
  (h1 : m ≠ n) 
  (h2 : ∀ p q: ℝ^3, (p = q + m) -> (q = q + n)) 
  (h3 : l ⊥ m) : l ⊥ n :=
sorry

end perp_to_parallel_lines_l29_29814


namespace minimize_expression_l29_29186

noncomputable def min_expression (z : ℂ) : ℂ :=
1 + z + 3*z^2 + z^3 + z^4

theorem minimize_expression (z : ℂ) (hz : abs z = 1) :
  (min_expression z).abs = (min_expression (-1/4 + complex.I * real.sqrt 15 / 4)).abs ∨
  (min_expression z).abs = (min_expression (-1/4 - complex.I * real.sqrt 15 / 4)).abs :=
sorry

end minimize_expression_l29_29186


namespace infinitely_many_87_b_seq_l29_29188

def a_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => 3 ^ (a_seq n)

def b_seq (n : ℕ) : ℕ := (a_seq n) % 100

theorem infinitely_many_87_b_seq (n : ℕ) (hn : n ≥ 2) : b_seq n = 87 := by
  sorry

end infinitely_many_87_b_seq_l29_29188


namespace sum_valid_x_ge_3_divisors_of_12_l29_29049

theorem sum_valid_x_ge_3_divisors_of_12 :
  let valid_x (x : ℕ) : Prop := x ≥ 3 ∧ (201020112012 % (x - 1) = 0)
  (∑ x in (finset.filter valid_x (finset.range 14)), x) = 32 := 
by
  let valid_x (x : ℕ) := x ≥ 3 ∧ (201020112012 % (x - 1) = 0)
  have hx : (∑ x in (finset.filter valid_x (finset.range 14)), x) = 32
  { 
    sorry
  }
  exact hx

end sum_valid_x_ge_3_divisors_of_12_l29_29049


namespace total_cost_div_selling_price_eq_23_div_13_l29_29978

-- Conditions from part (a)
def pencil_count := 140
def pen_count := 90
def eraser_count := 60

def loss_pencils := 70
def loss_pens := 30
def loss_erasers := 20

def pen_cost (P : ℝ) := P
def pencil_cost (P : ℝ) := 2 * P
def eraser_cost (P : ℝ) := 1.5 * P

def total_cost (P : ℝ) :=
  pencil_count * pencil_cost P +
  pen_count * pen_cost P +
  eraser_count * eraser_cost P

def loss (P : ℝ) :=
  loss_pencils * pencil_cost P +
  loss_pens * pen_cost P +
  loss_erasers * eraser_cost P

def selling_price (P : ℝ) :=
  total_cost P - loss P

-- Statement to be proved: the total cost is 23/13 times the selling price.
theorem total_cost_div_selling_price_eq_23_div_13 (P : ℝ) :
  total_cost P / selling_price P = 23 / 13 := by
  sorry

end total_cost_div_selling_price_eq_23_div_13_l29_29978


namespace CK_bisects_BH_l29_29961

-- Definitions of objects in the given conditions
variables {A B C H K : Point}
  (triangle_ABC : Triangle A B C) -- Triangle ABC
  (orthocenter_H : H = orthocenter triangle_ABC) -- H is the orthocenter of ABC
  (circumcircle_ABH : Circle A B H) -- Circumcircle of triangle ABH
  (circle_diam_AC : Circle A C) -- Circle with diameter AC
  (intersect_K : K ∈ intersection (circumcircle_ABH) (circle_diam_AC)) -- K is the intersection point of the two circles

-- The main theorem statement 
theorem CK_bisects_BH :
  ∃ M : Point, (midpoint M B H) ∧ lies_on_line M C K := sorry

end CK_bisects_BH_l29_29961


namespace find_OA_distance_l29_29861

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := { A | A.2^2 = 2 * p * A.1 }

def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def y_axis_distance (A : ℝ × ℝ) : ℝ := abs A.1

theorem find_OA_distance (p : ℝ) (A : ℝ × ℝ)
  (hp : 0 < p) 
  (hA : A ∈ parabola p)
  (h_d_focus : distance A (focus p) = 6)
  (h_d_y_axis : y_axis_distance A = 3) :
  distance (0, 0) A = 3 * real.sqrt 5 :=
sorry

end find_OA_distance_l29_29861


namespace p_value_l29_29891

open Set

variable (U : Set ℕ) (p : ℕ)
def M : Set ℕ := {x | x^2 - 5*x + p = 0}
def C_U_M : Set ℕ := {2, 3}

theorem p_value : (U = {1, 2, 3, 4}) → (C_U_M ⊆ U) → 
  (C_U_M = M \ (M ∩ Uᶜ)) → p = 4 :=
by
  intros hU hCU_CU_M_eq
  sorry

end p_value_l29_29891


namespace infinite_series_sum_eq_3_over_4_l29_29726

-- Define the series term
def series_term (k : ℕ) : ℝ := k / 3^k

-- Define the infinite series sum
noncomputable def series_sum : ℝ := ∑' k, series_term (k+1)

-- Proposition to state the sum of the series is 3/4
theorem infinite_series_sum_eq_3_over_4 : series_sum = 3 / 4 := 
by 
  sorry

end infinite_series_sum_eq_3_over_4_l29_29726


namespace candy_bars_per_bag_l29_29221

theorem candy_bars_per_bag (total_candy_bars : ℕ) (number_of_bags : ℕ) (h1 : total_candy_bars = 15) (h2 : number_of_bags = 5) : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l29_29221


namespace tan_a_values_l29_29843

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 :=
by
  sorry

end tan_a_values_l29_29843


namespace complete_square_l29_29415

theorem complete_square {x : ℝ} :
  x^2 - 6 * x - 8 = 0 ↔ (x - 3)^2 = 17 :=
sorry

end complete_square_l29_29415


namespace infinite_series_sum_eq_3_div_4_l29_29611

theorem infinite_series_sum_eq_3_div_4 : 
  (∑ k in (Set.Icc 1 (Set.univ : Set ℕ)), (k : ℝ) / (3 : ℝ)^k) = (3 : ℝ) / 4 := 
sorry

end infinite_series_sum_eq_3_div_4_l29_29611


namespace max_min_values_exponential_function_l29_29042

theorem max_min_values_exponential_function :
  let f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6) in
  ∃ (c d : ℝ), (c = 1) ∧ (d = 1 / Real.exp 8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ c) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ d) ∧
  (∃ x ∈ Set.Icc 0 3, f x = c) ∧
  (∃ x ∈ Set.Icc 0 3, f x = d) :=
by
  let f (x : ℝ) := Real.exp (2 * x^2 - 4 * x - 6)
  have h_deriv : ∀ x, HasDerivAt f ((4 * x - 4) * f x) x := sorry
  have h_cont : ContinuousOn f (Set.Icc 0 3) := sorry
  have h_max : ∃ x ∈ Set.Icc 0 3, IsMaxOn f (Set.Icc 0 3) x := sorry
  cases h_max with xs xs_prop
  have h_min : ∃ x ∈ Set.Icc 0 3, IsMinOn f (Set.Icc 0 3) x := sorry
  cases h_min with xm xm_prop
  have h0 : f 0 = 1 / Real.exp 6 := sorry
  have h1 : f 1 = 1 / Real.exp 8 := sorry
  have h3 : f 3 = 1 := sorry
  use 1, 1 / Real.exp 8
  apply And.intro rfl
  apply And.intro rfl
  apply And.intro
  {
    intros x hx
    interval_cases with x 0 3 1;
    simp only [f, Real.exp_le_exp];
    linarith
  }
  apply And.intro
  {
    intros x hx
    interval_cases with x 0 3 1;
    simp only [f, Real.exp_le_exp];
    linarith
  }
  apply And.intro
  {
    use 3
    apply And.intro
    exact Set.left_mem_Icc.mpr zero_le_three
    exact h3
  }
  {
    use 1
    apply And.intro
    exact Set.right_mem_Icc.mpr zero_le_three
    exact h1
  }

end max_min_values_exponential_function_l29_29042


namespace evaluate_series_sum_l29_29599

def series_sum := ∑ k in (finset.range (nat.succ n)), (k.succ : ℝ) / 3^k.succ

theorem evaluate_series_sum : (∑' k, (k.succ : ℝ) / 3^k.succ) = 3 / 4 :=
by sorry

end evaluate_series_sum_l29_29599


namespace remainder_n_sq_plus_3n_5_mod_25_l29_29904

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l29_29904


namespace Billy_weighs_more_l29_29483

-- Variables and assumptions
variable (Billy Brad Carl : ℕ)
variable (b_weight : Billy = 159)
variable (c_weight : Carl = 145)
variable (brad_formula : Brad = Carl + 5)

-- Theorem statement to prove the required condition
theorem Billy_weighs_more :
  Billy - Brad = 9 :=
by
  -- Here we put the proof steps, but it's omitted as per instructions.
  sorry

end Billy_weighs_more_l29_29483


namespace third_smallest_four_digit_in_pascals_triangle_l29_29313

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29313


namespace not_continuous_f_at_0_exists_no_a_makes_g_continuous_at_0_l29_29434

def f (x : ℝ) : ℝ := if x = 0 then 1 else 1 / x

theorem not_continuous_f_at_0 : ¬ continuous_at f 0 := 
by {
  -- Definitions
  sorry
}

def g (a : ℝ) (x : ℝ) : ℝ := if x = 0 then a else 1 / x

theorem exists_no_a_makes_g_continuous_at_0 :
  ¬ ∃ a : ℝ, continuous_at (g a) 0 :=
by {
  -- Definitions
  sorry
}

end not_continuous_f_at_0_exists_no_a_makes_g_continuous_at_0_l29_29434


namespace third_smallest_four_digit_Pascal_triangle_l29_29291

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29291


namespace train_crosses_bridge_in_30_seconds_l29_29895

noncomputable def train_length : ℝ := 100
noncomputable def bridge_length : ℝ := 200
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_mps

theorem train_crosses_bridge_in_30_seconds :
  crossing_time = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l29_29895


namespace third_smallest_four_digit_in_pascals_triangle_l29_29388

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n m k, binom n m = 1002 ∧ binom n (m + 1) = 1001 ∧ binom n (m + 2) = 1000 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29388


namespace rectangle_area_l29_29459

-- Conditions: 
-- 1. The length of the rectangle is three times its width.
-- 2. The diagonal length of the rectangle is x.

theorem rectangle_area (x : ℝ) (w l : ℝ) (h1 : w * 3 = l) (h2 : w^2 + l^2 = x^2) :
  l * w = (3 / 10) * x^2 :=
by
  sorry

end rectangle_area_l29_29459


namespace distinct_digits_sum_base7_l29_29953

theorem distinct_digits_sum_base7
    (A B C : ℕ)
    (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A)
    (h_nonzero : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
    (h_base7 : A < 7 ∧ B < 7 ∧ C < 7)
    (h_sum_eq : ((7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B)) = (7^3 * A + 7^2 * A + 7 * A)) :
    B + C = 6 :=
by {
    sorry
}

end distinct_digits_sum_base7_l29_29953


namespace inequality_transpose_l29_29846

variable (a b : ℝ)

theorem inequality_transpose (h : a < b) (hab : b < 0) : (1 / a) > (1 / b) := by
  sorry

end inequality_transpose_l29_29846


namespace infinite_series_sum_value_l29_29728

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29728


namespace average_percentage_decrease_l29_29991

theorem average_percentage_decrease : 
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((2000 * (1 - x)^2 = 1280) ↔ (x = 0.18)) :=
by
  sorry

end average_percentage_decrease_l29_29991


namespace infinite_series_sum_value_l29_29740

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29740


namespace trapezium_area_l29_29029

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end trapezium_area_l29_29029


namespace first_term_infinite_geometric_progression_l29_29248

theorem first_term_infinite_geometric_progression 
  (a r : ℝ) 
  (h1 : ∑ n in upper_bounds (set.univ : set ℕ), (a * r^n) = 8) 
  (h2 : a + a * r = 5) : 
  a = 8 * (1 - sqrt (3 / 8)) ∨ a = 8 * (1 + sqrt (3 / 8)) :=
by
  sorry

end first_term_infinite_geometric_progression_l29_29248


namespace range_sum_l29_29912

def f (x : ℝ) : ℝ :=
  (2 * real.sqrt 2 * real.sin (2 * x + real.pi / 4) +
  (x + 2)^2 - 4 * real.cos x^2) / (x^2 + 2)

def f_range (m n : ℝ) : Prop :=
  ∀ y : ℝ, y ∈ set.range f → y ∈ set.Icc m n

theorem range_sum (m n : ℝ) (h : f_range m n) : m + n = 2 :=
sorry

end range_sum_l29_29912


namespace third_smallest_four_digit_in_pascals_triangle_l29_29304

-- First we declare the conditions
def every_positive_integer_in_pascals_triangle : Prop :=
  ∀ n : ℕ, ∃ a b : ℕ, nat.choose b a = n

def number_1000_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1000

def number_1001_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 1 = 1001

def number_1002_in_pascals_triangle : Prop :=
  ∃ b : ℕ, nat.choose b 2 = 1002

-- Define the theorem
theorem third_smallest_four_digit_in_pascals_triangle
  (h1 : every_positive_integer_in_pascals_triangle)
  (h2 : number_1000_in_pascals_triangle)
  (h3 : number_1001_in_pascals_triangle)
  (h4 : number_1002_in_pascals_triangle) :
  (∃ n : ℕ, nat.choose n 2 = 1002 ∧ (∀ k, k < n → nat.choose k 1 < 1002)) := sorry

end third_smallest_four_digit_in_pascals_triangle_l29_29304


namespace min_value_ax_over_rR_l29_29962

theorem min_value_ax_over_rR (a b c r R : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_le_b : a ≤ b) (h_le_c : a ≤ c) (h_inradius : ∀ (a b c : ℝ), r = 2 * area / (a + b + c))
  (h_circumradius : ∀ (a b c : ℝ), R = (a * b * c) / (4 * area))
  (x : ℝ) (h_x : x = (b + c - a) / 2) (area : ℝ) :
  (a * x / (r * R)) ≥ 3 :=
sorry

end min_value_ax_over_rR_l29_29962


namespace stddev_transformation_l29_29094

theorem stddev_transformation (x : Fin 9 → ℝ) 
  (h : stddev x = 5) : 
  stddev (λ i, 3 * x i + 1) = 15 :=
sorry

end stddev_transformation_l29_29094


namespace third_smallest_four_digit_in_pascals_triangle_l29_29357

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n k, pascal n k = 1002 ∧ 
  ∀ m l, (pascal m l ≥ 1000 ∧ pascal m l < 1002) → (m = n ∧ l = k ∨ m ≠ n ∨ l ≠ k) :=
begin
  sorry
end

end third_smallest_four_digit_in_pascals_triangle_l29_29357


namespace find_z_percentage_of_1000_l29_29876

noncomputable def x := (3 / 5) * 4864
noncomputable def y := (2 / 3) * 9720
noncomputable def z := (1 / 4) * 800

theorem find_z_percentage_of_1000 :
  (2 / 3) * x + (1 / 2) * y = z → (z / 1000) * 100 = 20 :=
by
  sorry

end find_z_percentage_of_1000_l29_29876


namespace sum_k_over_3_pow_k_eq_three_fourths_l29_29655

noncomputable def sum_k_over_3_pow_k : ℝ :=
  ∑' k : ℕ, (k + 1) / 3 ^ (k + 1)

theorem sum_k_over_3_pow_k_eq_three_fourths :
  sum_k_over_3_pow_k = 3 / 4 := sorry

end sum_k_over_3_pow_k_eq_three_fourths_l29_29655


namespace solution_set_a_eq_2_solution_set_a_gt_neg1_l29_29108

-- Define the inequality and specific scenarios
def quadratic_inequality (a x : ℝ) : Prop :=
  a * x^2 + (1 - a) * x - 1 > 0

-- Prove the solution set for a = 2
theorem solution_set_a_eq_2 : (set_of (λ x, quadratic_inequality 2 x)) = {x | x < -1/2} ∪ {x | x > 1} :=
by
  sorry

-- Prove the solution set for a > -1 in separate cases
theorem solution_set_a_gt_neg1 (a : ℝ) (h : a > -1) :
  (set_of (λ x, quadratic_inequality a x)) =
    if a = 0 then {x | x > 1}
    else if 0 < a then {x | x < -1 / a} ∪ {x | x > 1}
    else {x | 1 < x ∧ x < -1 / a} :=
by
  sorry

end solution_set_a_eq_2_solution_set_a_gt_neg1_l29_29108


namespace infinite_series_sum_value_l29_29739

-- Define the series term
def series_term (k : ℕ) : ℝ := k / (3 : ℝ)^k

-- Define the infinite series sum
def infinite_series_sum : ℝ := ∑' k, series_term (k+1)

theorem infinite_series_sum_value : infinite_series_sum = 3 / 4 := by
  sorry

end infinite_series_sum_value_l29_29739


namespace area_enclosed_by_graph_eq_160_l29_29511

theorem area_enclosed_by_graph_eq_160 :
  ∃ (area : ℝ), area = 160 ∧
  (∀ (x y : ℝ), |2 * x| + |5 * y| = 20 → abs x ≤ 10 ∧ abs y ≤ 4) :=
begin
  sorry
end

end area_enclosed_by_graph_eq_160_l29_29511


namespace third_smallest_four_digit_Pascal_triangle_l29_29285

theorem third_smallest_four_digit_Pascal_triangle : 
  ∃ n : ℕ, (binomial (n + 2) (n + 2) = 1002) ∧ ∀ k : ℕ, (k < (n + 2) → binomial (k + 2) (k + 2) < 1002) :=
by
  sorry

end third_smallest_four_digit_Pascal_triangle_l29_29285


namespace infinite_series_sum_eq_l29_29702

noncomputable def infinite_sum := ∑ k in Nat (λ k, (k + 1 : ℚ) / 3^(k + 1))

theorem infinite_series_sum_eq : infinite_sum = 3 / 4 := by
  sorry

end infinite_series_sum_eq_l29_29702


namespace series_result_l29_29634

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l29_29634


namespace find_lambda_l29_29089

variables {a b : ℝ} (λ : ℝ)

-- Definitions and conditions
def magnitude (v : ℝ) := abs v

def angle_between (a b : ℝ) := 60

def dot_product (a b : ℝ) := a * b

def perpendicular (a b : ℝ) := dot_product a b = 0

-- Main theorem statement
theorem find_lambda
  (h1 : magnitude a = 1)
  (h2 : magnitude b = 1)
  (h3 : angle_between a b = 60)
  (h4 : perpendicular a (λ * b - a)) :
  λ = 2 :=
sorry

end find_lambda_l29_29089
