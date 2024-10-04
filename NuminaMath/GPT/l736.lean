import Mathlib

namespace solution_b_l736_736640

variable {f : ℝ → ℝ}
variable (h_increasing : ∀ x y, x ≤ y → f x ≤ f y)
variable (f_domain : ∀ x, x ≥ -4 → f x)

noncomputable def range_of_b : set ℝ := {b | ∀ x, -4 ≤ f (cos x - b^2) ∧ f (sin^2 x - b - 3) ≤ f (cos x - b^2)}
theorem solution_b (b : ℝ) : b ∈ range_of_b ↔ b ∈ set.Icc (1/2 - real.sqrt 2) 1 :=
by
  sorry

end solution_b_l736_736640


namespace intersecting_lines_ratio_l736_736840

variable (ABC : Type) [triangle ABC]
variable (A B C E D F : ABC)
variable (hE : E ∈ segment A B)
variable (hD : D ∈ segment B C)
variable (hRatio_AB : ratio (segment A E) (segment E B) = 1/3)
variable (hRatio_BC : ratio (segment C D) (segment D B) = 1/2)
variable (hIntersect_AD_CE: is_intersection (line A D) (line C E) F)

theorem intersecting_lines_ratio : 
  let EF := segment E F
  let FC := segment F C
  let AF := segment A F
  let FD := segment F D
  EF / FC + AF / FD = 3/2 :=
sorry

end intersecting_lines_ratio_l736_736840


namespace even_number_combinations_l736_736154

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736154


namespace prove_monotonicity_a_neg4_prove_range_a_l736_736802

-- Definitions
def f (x : ℝ) (a : ℝ) : ℝ := exp(2 * x) + a * exp(x)
def g (x : ℝ) (a : ℝ) : ℝ := exp(2 * x) + a * exp(x) - (a * a * x)

-- Proof of monotonicity of f when a = -4
theorem prove_monotonicity_a_neg4 : 
  ∀ x : ℝ, 
    if a = -4 then (f x a).derivative > 0 ↔ x > log 2 ∧ (f x a).derivative < 0 ↔ x < log 2 :=
sorry

-- Proof of range of a when f(x) ≥ a²x for all x
theorem prove_range_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ a^2 * x) ↔ -1 ≤ a ∧ a ≤ 2 * exp(3/4) :=
sorry

end prove_monotonicity_a_neg4_prove_range_a_l736_736802


namespace negation_of_existential_to_universal_l736_736666

variable (p : Prop)

noncomputable def negation_of_proposition (h : ∃ x : ℝ, sin x ≥ 1) : ¬ p :=
by
  sorry
  
theorem negation_of_existential_to_universal :
  (¬ ∃ x : ℝ, sin x ≥ 1) → (∀ x : ℝ, sin x < 1) :=
by
  intro h
  simp at h
  sorry

end negation_of_existential_to_universal_l736_736666


namespace area_of_triangle_ABC_l736_736331

section
variables (a b c : ℝ) (A B C : ℝ)
variables (sin cos : ℝ → ℝ)

hypothesis h1 : sqrt 3 * a * sin B - b * cos A = b
hypothesis h2 : b + c = 4
hypothesis h3 : A = π / 3

noncomputable def angle_A_at_A_min : ℝ := π / 3

theorem area_of_triangle_ABC :
  let abc_min := 2 in
  A = angle_A_at_A_min →
  (1 / 2) * 2 * 2 * (sin (π / 3)) = sqrt 3 :=
begin
  sorry
end
end

end area_of_triangle_ABC_l736_736331


namespace dot_product_result_l736_736308

noncomputable def vectorA : ℝ^3 := sorry
noncomputable def vectorB : ℝ^3 := sorry

def magnitude_one (a : ℝ^3) : Prop := real.sqrt (a.1^2 + a.2^2 + a.3^2) = 1

def angle_pi_div_two (a b : ℝ^3) : Prop := real_inner a b = 0

theorem dot_product_result :
  magnitude_one vectorA →
  angle_pi_div_two vectorA vectorB →
  real_inner vectorA (-6 • vectorA - vectorB) = -6 :=
by
  sorry

end dot_product_result_l736_736308


namespace evaluate_expression_l736_736209

theorem evaluate_expression : (64^((1:ℝ)/(3:ℝ)) - ((-2/3:ℝ)^0) + real.log (4) / real.log (2)) = 5 := by
  sorry

end evaluate_expression_l736_736209


namespace side_length_of_cube_l736_736030

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l736_736030


namespace continuity_and_discontinuity_of_f_l736_736781

noncomputable def f : ℝ → ℝ
| x := if x < 0 then exp (1 / x)
       else if x <= 1 then (x^2 + 7*x) / (x^2 + 19)
       else (4*x + 2) / (x^2 - 1)

theorem continuity_and_discontinuity_of_f :
  ContinuousAt f 0 ∧ ¬ContinuousAt f 1 := by
  sorry

end continuity_and_discontinuity_of_f_l736_736781


namespace find_unit_direction_vector_l736_736299

noncomputable def line_equation (x : ℝ) : ℝ := 2 * x + 2

def direction_vector : ℝ × ℝ := (1, 2)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

def unit_direction_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

theorem find_unit_direction_vector :
  unit_direction_vector direction_vector = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) ∨ 
    unit_direction_vector direction_vector = (-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5) :=
by
  sorry

end find_unit_direction_vector_l736_736299


namespace train_length_is_600_l736_736194

noncomputable def length_of_train (cross_platform_sec: ℕ) (cross_signal_sec: ℕ) (platform_length: ℕ) := 
  let V := cross_signal_sec
  let L := V * cross_signal_sec
  let total_length := L + platform_length
  L

theorem train_length_is_600 (cross_platform_sec: 39) (cross_signal_sec: 18) (platform_length: 700) : 
  length_of_train () cross_platform_sec cross_signal_sec platform_length = 600 := by
  sorry

end train_length_is_600_l736_736194


namespace z_conjugate_sum_l736_736718

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736718


namespace irreducible_geometric_series_l736_736390

open Polynomial

noncomputable def is_prime (p : ℕ) : Prop := sorry

noncomputable def is_irreducible (q : Polynomial ℚ) : Prop := sorry

theorem irreducible_geometric_series (p : ℕ) (hp : is_prime p) :
  is_irreducible (∑ i in Finset.range p, (Polynomial.C 1) * (Polynomial.X ^ i)) := sorry

end irreducible_geometric_series_l736_736390


namespace tens_digit_2015_pow_2016_minus_2017_l736_736909

theorem tens_digit_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 = 8 := 
sorry

end tens_digit_2015_pow_2016_minus_2017_l736_736909


namespace parabola_focus_distance_correct_l736_736643

noncomputable def parabola_focus_distance : ℝ :=
let p := 1 in
let A := (4, 2 * real.sqrt 4) in  -- A's y coordinate is derived from y^2 = 4*4
let focus := (1, 0) in            -- Focus of y^2 = 4x is at (1, 0)
real.sqrt ((A.1 - focus.1)^2 + A.2^2)

theorem parabola_focus_distance_correct : parabola_focus_distance = 5 :=
by sorry

end parabola_focus_distance_correct_l736_736643


namespace opposite_of_neg_three_l736_736450

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l736_736450


namespace planes_KnLnMn_concurrent_l736_736775

-- Definitions for the points and conditions in the tetrahedron
structure Tetrahedron (A B C D : Point) :=
(Point) (edge_length : Point × Point → ℝ)

variable {A B C D : Point}

-- Definitions for the sequence of points K_n, L_n, M_n
def K_n (n : ℕ) [n > 0] : Point := sorry -- Point on AB such that AB = n * AK_n
def L_n (n : ℕ) [n > 0] : Point := sorry -- Point on AC such that AC = (n+1) * AL_n
def M_n (n : ℕ) [n > 0] : Point := sorry -- Point on AD such that AD = (n+2) * AM_n

-- Define the plane created by points K_n, L_n, M_n
def Plane (a b c : Point) : Set Point := sorry -- the plane containing points a, b, and c

-- Define the concurrency condition for planes
def Concurrent (planes : Set (Set Point)) : Prop :=
  ∃ line : Set Point, ∀ plane ∈ planes, line ⊆ plane

-- The theorem to prove: all planes K_n L_n M_n are concurrent
theorem planes_KnLnMn_concurrent :
  Concurrent (λ n, Plane (K_n n) (L_n n) (M_n n)) :=
sorry

end planes_KnLnMn_concurrent_l736_736775


namespace fraction_equiv_l736_736481

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l736_736481


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736822

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736822


namespace z_conjugate_sum_l736_736700

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736700


namespace problem_l736_736426

theorem problem (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : (14 * y - 2)^2 = 258 := by
  sorry

end problem_l736_736426


namespace figure_100_squares_l736_736871

def nonoverlapping_unit_squares (n : ℕ) : ℕ := 4 * n ^ 2 + 1

theorem figure_100_squares : nonoverlapping_unit_squares 100 = 40001 :=
by {
  calc nonoverlapping_unit_squares 100
      = 4 * 100 ^ 2 + 1 : by refl
  ... = 40000 + 1 : by norm_num
  ... = 40001 : by norm_num
}

end figure_100_squares_l736_736871


namespace z_conjugate_sum_l736_736697

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736697


namespace max_scribable_pairs_l736_736568

noncomputable def scribable_pair (omega gamma : Circle) : Prop :=
  ∃ (T : Triangle), T.circumcircle = omega ∧ T.incircle = gamma

-- Maximum number of scribable pairs among n distinct circles
theorem max_scribable_pairs (n : ℕ) (circles : Fin n → Circle) :
  (∑ i j, if i ≠ j ∧ scribable_pair (circles i) (circles j) then 1 else 0) ≤ (n / 2) ^ 2 := 
sorry

end max_scribable_pairs_l736_736568


namespace min_score_needed_l736_736785

/-- 
Given the list of scores and the targeted increase in the average score,
ascertain that the minimum score required on the next test to achieve the
new average is 110.
 -/
theorem min_score_needed 
  (scores : List ℝ) 
  (target_increase : ℝ) 
  (new_score : ℝ) 
  (total_scores : ℝ)
  (current_average : ℝ) 
  (target_average : ℝ) 
  (needed_score : ℝ) :
  (total_scores = 86 + 92 + 75 + 68 + 88 + 84) ∧
  (current_average = total_scores / 6) ∧
  (target_average = current_average + target_increase) ∧
  (new_score = total_scores + needed_score) ∧
  (target_average = new_score / 7) ->
  needed_score = 110 :=
by
  sorry

end min_score_needed_l736_736785


namespace hilary_total_kernels_l736_736678

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end hilary_total_kernels_l736_736678


namespace product_of_first_30_terms_is_5_l736_736668

variable {a : ℕ → ℝ} (f : ℕ → ℝ) (log : ℝ → ℝ → ℝ)

-- Given the sequence defined as a_n = log_{(n+1)}(n+2)
def a (n : ℕ) : ℝ := log (n+1) (n+2)

-- Define the product of the first 30 terms
def product_of_first_30_terms : ℝ := (Finset.range 30).prod (λ n, a n)

-- The proof obligation
theorem product_of_first_30_terms_is_5 :
  product_of_first_30_terms = 5 :=
sorry

end product_of_first_30_terms_is_5_l736_736668


namespace max_distance_l736_736534

theorem max_distance (x y : ℝ) (u v w : ℝ)
  (h1 : u = Real.sqrt (x^2 + y^2))
  (h2 : v = Real.sqrt ((x - 1)^2 + y^2))
  (h3 : w = Real.sqrt ((x - 1)^2 + (y - 1)^2))
  (h4 : u^2 + v^2 = w^2) :
  ∃ (P : ℝ), P = 2 + Real.sqrt 2 :=
sorry

end max_distance_l736_736534


namespace fraction_calculation_l736_736246

theorem fraction_calculation :
  ( (3 / 7 + 5 / 8 + 1 / 3) / (5 / 12 + 2 / 9) = 2097 / 966 ) :=
by
  sorry

end fraction_calculation_l736_736246


namespace opposite_of_neg3_is_3_l736_736441

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l736_736441


namespace range_of_m_l736_736749

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x ^ 2 - 2 * (4 - m) * x + 1
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end range_of_m_l736_736749


namespace points_lie_on_y_eq_x_l736_736104

theorem points_lie_on_y_eq_x (x y : ℝ) (h : x = y) : y = x :=
by {
  exact h.symm,
}

end points_lie_on_y_eq_x_l736_736104


namespace gcd_8885_4514_5246_l736_736993

theorem gcd_8885_4514_5246 : Nat.gcd (Nat.gcd 8885 4514) 5246 = 1 :=
sorry

end gcd_8885_4514_5246_l736_736993


namespace train_to_bus_ratio_l736_736525

variable (total_distance : ℕ)
variable (distance_plane : ℕ)
variable (distance_bus : ℕ)
variable (distance_train : ℕ)
variable (ratio : ℚ)

def distances_workout : Prop := 
  total_distance = 900 ∧
  distance_plane = total_distance * 1 / 3 ∧
  distance_bus = 360 ∧
  distance_train = total_distance - (distance_plane + distance_bus) ∧
  ratio = distance_train / distance_bus

theorem train_to_bus_ratio (total_distance = 900) (distance_plane = total_distance * 1 / 3) (distance_bus = 360) : 
{ratio : ℚ // 
    distances_workout → 
    ratio = 2 / 3} :=
sorry

end train_to_bus_ratio_l736_736525


namespace log_bounded_sum_l736_736054

theorem log_bounded_sum :
  ∃ c d : ℤ, (c < real.log 1458 / real.log 10 ∧ real.log 1458 / real.log 10 < d) ∧ c + d = 7 :=
sorry

end log_bounded_sum_l736_736054


namespace complex_conjugate_sum_l736_736730

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736730


namespace flowers_count_l736_736380

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l736_736380


namespace salary_net_change_l736_736926

variable {S : ℝ}

theorem salary_net_change (S : ℝ) : (1.4 * S - 0.4 * (1.4 * S)) - S = -0.16 * S :=
by
  sorry

end salary_net_change_l736_736926


namespace rectangle_in_triangle_area_l736_736844

theorem rectangle_in_triangle_area
  (PR : ℝ) (h_PR : PR = 15)
  (Q_altitude : ℝ) (h_Q_altitude : Q_altitude = 9)
  (x : ℝ)
  (AD : ℝ) (h_AD : AD = x)
  (AB : ℝ) (h_AB : AB = x / 3) :
  (AB * AD = 675 / 64) :=
by
  sorry

end rectangle_in_triangle_area_l736_736844


namespace total_money_l736_736887

noncomputable theory

def money_distributions (a j : ℕ) : Prop :=
  let t := 48 in
  let t' := 3 * t in
  let j' := 3 * j in
  let a' := a - 2 * (t + j) in
  let t'' := 2 * t' in
  let a'' := 2 * a' in
  let j'' := j' - (a' + t') in
  let t''' := t'' - (a'' + j'') in
  t''' = 48

theorem total_money : ∀ (a j : ℕ), money_distributions a j → a + j + 48 = 528 :=
sorry

end total_money_l736_736887


namespace rowing_speed_in_still_water_l736_736523

theorem rowing_speed_in_still_water (c : ℝ) (h₀ : c = 1.2) (h₁ : ∃ (v t : ℝ), 2 * t * (v - c) = t * (v + c)) : 
  ∃ v : ℝ, v = 3.6 :=
by
  use 3.6
  sorry

end rowing_speed_in_still_water_l736_736523


namespace percentage_calculation_l736_736489

theorem percentage_calculation :
  ( (2 / 3 * 2432 / 3 + 1 / 6 * 3225) / 450 * 100 ) = 239.54 := 
sorry

end percentage_calculation_l736_736489


namespace axis_of_symmetry_translated_graph_l736_736320

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

theorem axis_of_symmetry_translated_graph (k : ℤ) :
 ∃ x : ℝ, 2 * x + π / 6 = k * π + π / 2 :=
sorry

end axis_of_symmetry_translated_graph_l736_736320


namespace no_solution_exists_l736_736581

theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ¬(2 / a + 2 / b = 1 / (a + b)) :=
sorry

end no_solution_exists_l736_736581


namespace molecular_weight_of_3_moles_caI2_l736_736094

namespace Chemistry

/-- Given the atomic weights of calcium and iodine, and the composition of CaI2, 
    prove that the molecular weight of 3 moles of CaI2 is 881.64 grams. -/
theorem molecular_weight_of_3_moles_caI2 
  (atomic_weight_ca : ℝ) (atomic_weight_I : ℝ) 
  (composition_I_in_caI2 : ℕ) 
  (h1 : atomic_weight_ca = 40.08) 
  (h2 : atomic_weight_I = 126.90) 
  (h3 : composition_I_in_caI2 = 2) 
  : 3 * (atomic_weight_ca + composition_I_in_caI2 * atomic_weight_I) = 881.64 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end Chemistry

end molecular_weight_of_3_moles_caI2_l736_736094


namespace midpoint_of_tangents_l736_736370

variables {Point : Type}
variables (A B P Q E F : Point) (S : set Point)

-- Define what it means to be an Apollonian circle.
def is_apollonian_circle (S : set Point) (A B : Point) : Prop :=
sorry  -- placeholder for the definition

-- Define what it means for tangents to be drawn from a point to a circle.
def tangents_from_point_to_circle (A : Point) (P Q : Point) (S : set Point) : Prop :=
sorry  -- placeholder for the definition

-- Define the condition that a point lies outside the circle.
def lies_outside (A : Point) (S : set Point) : Prop :=
sorry  -- placeholder for the definition

theorem midpoint_of_tangents 
  (h1 : is_apollonian_circle S A B)
  (h2 : lies_outside A S)
  (h3 : tangents_from_point_to_circle A P Q S) : 
  B = midpoint P Q :=
sorry  -- proof goes here

end midpoint_of_tangents_l736_736370


namespace playground_dimensions_l736_736967

theorem playground_dimensions 
  (a b : ℕ) 
  (h1 : (a - 2) * (b - 2) = 4) : a * b = 2 * a + 2 * b :=
by
  sorry

end playground_dimensions_l736_736967


namespace side_length_of_cube_l736_736031

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l736_736031


namespace complex_multiplication_l736_736994

theorem complex_multiplication :
  let i := Complex.I in
  i * (2 - i) = 1 + 2 * i :=
by
  intro i
  sorry

end complex_multiplication_l736_736994


namespace arithmetic_sequence_general_formula_sum_first_n_terms_of_c_l736_736279

theorem arithmetic_sequence_general_formula (a_n b_n : ℕ → ℕ) (h_arith : ∀ n, a_n = a_1 + (n-1) * 1)
  (h_geom : ∀ n, b_n = b_1 * 2 ^ (n-1)) (h_b2 : b_n 2 = 2) (h_b3 : b_n 3 = 4) (h_a1_b1 : a_n 1 = b_n 1) (h_a8_b4 : a_n 8 = b_n 4) :
  a_n n = n :=
sorry

theorem sum_first_n_terms_of_c (c_n a_n b_n : ℕ → ℕ) (h_arith : ∀ n, a_n = a_1 + (n-1) * 1)
  (h_geom : ∀ n, b_n = b_1 * 2 ^ (n-1)) (h_c_n : ∀ n, c_n = a_n + b_n) (h_b2 : b_n 2 = 2) (h_b3 : b_n 3 = 4) (h_a1_b1 : a_n 1 = b_n 1)
  (h_a8_b4 : a_n 8 = b_n 4) :
  ∀ n, (∑ k in range (n + 1), c_n k) = (n * (n + 1)) / 2 + 2^n - 1 :=
sorry

end arithmetic_sequence_general_formula_sum_first_n_terms_of_c_l736_736279


namespace relationship_abc_l736_736613

def a : ℝ := (-2) ^ 0
def b : ℝ := (1 / 2) ^ (-1)
def c : ℝ := (-3) ^ (-2)

theorem relationship_abc : a < b ∧ c < a := by
  sorry

end relationship_abc_l736_736613


namespace opposite_of_neg_three_l736_736432

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l736_736432


namespace elder_age_correct_l736_736417

noncomputable def elder_age (younger : ℕ) (age_diff : ℕ) (factor : ℕ) : ℕ :=
  let e := younger + age_diff
  have h : e - 10 = factor * (younger - 10),
    by sorry
  e

theorem elder_age_correct :
  ∀ (younger age_diff factor : ℕ), age_diff = 20 ∧ factor = 5 ∧ (elder_age younger 20 5 - 10 = 5 * (younger - 10)) →
  elder_age younger age_diff factor = 35 :=
begin
  intros,
  sorry,
end

end elder_age_correct_l736_736417


namespace number_of_even_results_l736_736145

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736145


namespace trains_distance_difference_l736_736902

theorem trains_distance_difference
  (v1 v2 : ℕ)    -- velocities in km/hr
  (d : ℕ)        -- distance between stations in km
  (h1 : v1 = 20)
  (h2 : v2 = 25)
  (h3 : d = 495) :
  let t := d / (v1 + v2) in
  let d1 := v1 * t in
  let d2 := v2 * t in
  d2 - d1 = 55 :=
by
  sorry

end trains_distance_difference_l736_736902


namespace circle_coordinates_l736_736556

theorem circle_coordinates (r : ℕ) (h : r ∈ {2, 4, 6, 8, 10}) :
  ∃ (D C A : ℝ), D = 2 * r ∧ C = 2 * real.pi * r ∧ A = real.pi * r^2 ∧
  ((D = 4 ∧ C = 4 * real.pi ∧ A = 4 * real.pi) ∨ 
   (D = 8 ∧ C = 8 * real.pi ∧ A = 16 * real.pi) ∨ 
   (D = 12 ∧ C = 12 * real.pi ∧ A = 36 * real.pi) ∨ 
   (D = 16 ∧ C = 16 * real.pi ∧ A = 64 * real.pi) ∨ 
   (D = 20 ∧ C = 20 * real.pi ∧ A = 100 * real.pi)) :=
by {
  use [2 * r, 2 * real.pi * r, real.pi * r^2],
  simp [set.mem_insert_iff, real.pi],
  sorry
}

end circle_coordinates_l736_736556


namespace cost_difference_l736_736413

def cost_cartons (qty : ℕ) (price : ℕ) : ℕ := qty * price

def discount (amount : ℕ) (percent : ℕ) : ℕ := amount * percent / 100

def apply_discount (amount : ℕ) (percent : ℕ) : ℕ := amount - (discount amount percent)

def tax (amount : ℕ) (percent : ℕ) : ℕ := amount * percent / 100

def apply_tax (amount : ℕ) (percent : ℕ) : ℕ := amount + (tax amount percent)

theorem cost_difference :
  let ice_cream_qty := 100
  let yoghurt_qty := 35
  let cheese_qty := 50
  let milk_qty := 20
  let ice_cream_price := 12
  let yoghurt_price := 3
  let cheese_price := 8
  let milk_price := 4
  let ice_cream_discount_percent := 5
  let yoghurt_tax_percent := 8
  let cheese_discount_percent := 10
  let returned_ice_cream_qty := 10
  let returned_cheese_qty := 5
  let total_cost_ice_cream := cost_cartons ice_cream_qty ice_cream_price
  let total_cost_yoghurt := cost_cartons yoghurt_qty yoghurt_price
  let total_cost_cheese := cost_cartons cheese_qty cheese_price
  let total_cost_milk := cost_cartons milk_qty milk_price
  let discounted_cost_ice_cream := apply_discount total_cost_ice_cream ice_cream_discount_percent
  let taxed_cost_yoghurt := apply_tax total_cost_yoghurt yoghurt_tax_percent
  let discounted_cost_cheese := apply_discount total_cost_cheese cheese_discount_percent
  let returned_cost_ice_cream := cost_cartons returned_ice_cream_qty ice_cream_price
  let returned_cost_cheese := cost_cartons returned_cheese_qty cheese_price
  let adjusted_cost_ice_cream := discounted_cost_ice_cream - returned_cost_ice_cream
  let adjusted_cost_cheese := discounted_cost_cheese - returned_cost_cheese
  let combined_cost_ice_cream_cheese := adjusted_cost_ice_cream + adjusted_cost_cheese
  let combined_cost_yoghurt_milk := taxed_cost_yoghurt + total_cost_milk
  combined_cost_ice_cream_cheese - combined_cost_yoghurt_milk = 1146.60 :=
by
  sorry

end cost_difference_l736_736413


namespace trihedral_sphere_radius_l736_736336

noncomputable def sphere_radius 
  (α r : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  : ℝ :=
r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3)

theorem trihedral_sphere_radius 
  (α r R : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  (hR : R = sphere_radius α r hα) 
  : R = r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3) :=
by
  sorry

end trihedral_sphere_radius_l736_736336


namespace pine_count_25_or_26_l736_736061

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l736_736061


namespace line_through_A_with_area_l736_736219

-- Define the condition of the point A
def point_A := (-2, 2)

-- Define function for computing the area of the triangle
def triangle_area (x_intercept y_intercept : ℝ) : ℝ :=
  (1 / 2) * x_intercept * y_intercept

-- Define the line equation in standard form
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 2 = 0

-- Define the given condition that the line passes through A and forms a triangle of area 1
theorem line_through_A_with_area :
  ∃ k : ℝ, 
  (-2, 2) = (λ (x : ℝ), (0, k * (0 + 2) + 2)) ∧
  triangle_area ((-(2 * k) - 2) / k) (2 * k + 2) = 1 ∧
  line_equation (-2) 2 :=
sorry

end line_through_A_with_area_l736_736219


namespace fraction_defined_range_l736_736322

theorem fraction_defined_range (x : ℝ) : 
  (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_defined_range_l736_736322


namespace largest_n_S_n_positive_l736_736313

-- We define the arithmetic sequence a_n.
def arith_seq (a_n : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Definitions for the conditions provided.
def first_term_positive (a_n : ℕ → ℝ) : Prop := 
  a_n 1 > 0

def term_sum_positive (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 + a_n 2017 > 0

def term_product_negative (a_n : ℕ → ℝ) : Prop :=
  a_n 2016 * a_n 2017 < 0

-- Sum of the first n terms of an arithmetic sequence
noncomputable def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a_n 1 + a_n n) / 2

-- Statement we want to prove in Lean 4.
theorem largest_n_S_n_positive (a_n : ℕ → ℝ) 
  (h_seq : arith_seq a_n) 
  (h1 : first_term_positive a_n) 
  (h2 : term_sum_positive a_n) 
  (h3 : term_product_negative a_n) : 
  ∀ n : ℕ, sum_first_n_terms a_n n > 0 → n ≤ 4032 := 
sorry

end largest_n_S_n_positive_l736_736313


namespace goods_train_passes_man_in_10_seconds_l736_736960

def goods_train_pass_time (man_speed_kmph goods_speed_kmph goods_length_m : ℕ) : ℕ :=
  let relative_speed_mps := (man_speed_kmph + goods_speed_kmph) * 1000 / 3600
  goods_length_m / relative_speed_mps

theorem goods_train_passes_man_in_10_seconds :
  goods_train_pass_time 55 60 320 = 10 := sorry

end goods_train_passes_man_in_10_seconds_l736_736960


namespace number_of_even_results_l736_736147

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736147


namespace middle_number_probability_l736_736868

noncomputable theory
open_locale classical

/-- 
  Define the problem conditions.
  - The numbers 1 to 11 are arranged in a line.
  - The middle number in the line is larger than exactly one number to its left.
-/
def middle_larger_left (l : list ℤ) : Prop :=
  l.length = 11 ∧ l.nth 5 > l.take 5.to_finset.count (< l.nth 5)

/--
  Define the probability calculation function for the given problem's conditions.
-/
def probability_larger_right (l : list ℤ) : ℚ :=
  if middle_larger_left l then
    let valid_arrangements := filter middle_larger_left (list.permutations (list.range' 1 11)) in
    (valid_arrangements.filter (λ m, m.nth 5 > m.drop 6.head!)).length / valid_arrangements.length
  else 0

/--
  Statement of the mathematical proof problem in Lean.
  - Prove the probability that the middle number is larger than exactly one number to its right is 10/33.
-/
theorem middle_number_probability :
  ∀ l : list ℤ, probability_larger_right l = 10 / 33 :=
by
  sorry

end middle_number_probability_l736_736868


namespace evaluate_g_at_2_l736_736373

def g (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem evaluate_g_at_2 : g 2 = 11 := by
  sorry

end evaluate_g_at_2_l736_736373


namespace z_conjugate_sum_l736_736723

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736723


namespace find_tangency_circle_radius_l736_736541

-- Conditions: a sphere of radius R and a central angle α in the axial section of the spherical sector.
variables {R α : ℝ}

-- Theorem: Finding the radius of the circle where the surfaces of the sphere and the sector touch
theorem find_tangency_circle_radius (hR : R > 0) (hα : 0 < α ∧ α < 2 * Mathlib.Real.pi) :
  let r := R * Mathlib.sin α / (4 * Mathlib.cos (Mathlib.Real.pi / 4 - α / 4) ^ 2) in
  r = R * Mathlib.sin α / (4 * Mathlib.cos (Mathlib.Real.pi / 4 - α / 4) ^ 2) :=
sorry

end find_tangency_circle_radius_l736_736541


namespace reduction_for_1750_yuan_max_daily_profit_not_1900_l736_736416

def average_shirts_per_day : ℕ := 40 
def profit_per_shirt_initial : ℕ := 40 
def price_reduction_increase_shirts (reduction : ℝ) : ℝ := reduction * 2 
def daily_profit (reduction : ℝ) : ℝ := (profit_per_shirt_initial - reduction) * (average_shirts_per_day + price_reduction_increase_shirts reduction)

-- Part 1: Proving the reduction that results in 1750 yuan profit
theorem reduction_for_1750_yuan : ∃ x : ℝ, daily_profit x = 1750 ∧ x = 15 := 
by {
  sorry
}

-- Part 2: Proving that the maximum cannot reach 1900 yuan
theorem max_daily_profit_not_1900 : ∀ x : ℝ, daily_profit x ≤ 1800 ∧ (∀ y : ℝ, y ≥ daily_profit x → y < 1900) :=
by {
  sorry
}

end reduction_for_1750_yuan_max_daily_profit_not_1900_l736_736416


namespace find_possible_values_of_y_l736_736799

theorem find_possible_values_of_y (x : ℝ) (h : x^2 + 9 * (3 * x / (x - 3))^2 = 90) :
  y = (x - 3)^3 * (x + 2) / (2 * x - 4) → y = 28 / 3 ∨ y = 169 :=
by
  sorry

end find_possible_values_of_y_l736_736799


namespace collinear_A_l736_736777

open EuclideanGeometry

-- Definitions:
variables {A B C A' B' C' D O F : Point}

-- Given Conditions:
def triangle (A B C : Point) : Prop := ¬collinear A B C
def midpoint (X Y Z : Point) : Prop := dist X Y = dist X Z
def tangent_point (circle_center : Point) (tangent_point : Point) (tangent_line : Line) : Prop :=
  tangent_line.contains tangent_point ∧ tangent_line ⊥ (Line.mk circle_center tangent_point)

-- Theorem to prove:
theorem collinear_A'_O_F
  (hABC : triangle A B C)
  (hA' : midpoint A' B C)
  (hB' : midpoint B' C A)
  (hC' : midpoint C' A B)
  (hO : incircle_center O (triangle_incircle A B C))
  (hD : tangent_point O D (Line.mk B C))
  (hF : intersection (Line.mk A D) (Line.mk B' C') = some F) :
  collinear A' O F :=
sorry

end collinear_A_l736_736777


namespace main_problem_l736_736016

noncomputable def statement1 := ∀ x : ℝ, f(x) = 4 * Real.cos (2 * x + Real.pi / 3) → ∃ y : ℝ, y = -5 * Real.pi / 12 ∧ f(y) = 0

noncomputable def statement2 := 
  ∀ (A B C D : EuclideanSpace ℝ 3), 
    (dist A B = 1) → 
    (dist A C = 3) → 
    (D = midpoint ℝ A C) →
    (A -ᵥ D) • (C -ᵥ B) = 4

noncomputable def statement3 := 
  ∀ A B : ℝ, 
    (A < B) ↔ (Real.cos (2 * A) > Real.cos (2 * B))

noncomputable def statement4 := 
  ∀ (x : ℝ), x ∈ ℝ →
    (Real.min (Real.sin x) (Real.cos x) ≤ Real.sqrt 2 / 2)

theorem main_problem (f : ℝ → ℝ) (A B C D : EuclideanSpace ℝ 3) (x : ℝ) :
  statement1 (f x) ∧ 
  statement2 A B C D ∧ 
  statement3 A B ∧ 
  statement4 x :=
sorry

end main_problem_l736_736016


namespace correct_propositions_l736_736218

-- Definitions and conditions for each proposition 
def prop1 := ∀ x, abs (sin x + 1 / 2) = abs (sin (x + 2 * π) + 1 / 2)
def prop2 := ∀ x, 4 * cos (2 * x - π / 6) = 4 * cos (2 * (-π / 6) - π / 6) - x
def prop5 := ∀ (A B : ℝ), A > B → sin A > sin B

-- Lean theorem stating the correctness of the propositions 1, 2, and 5
theorem correct_propositions : prop1 ∧ prop2 ∧ prop5 :=
by
  sorry

end correct_propositions_l736_736218


namespace partition_sum_abs_eq_n_squared_l736_736924

theorem partition_sum_abs_eq_n_squared (n : ℕ) (h_pos : 0 < n)
  (a b : Fin n → ℕ)
  (h_a_sorted : ∀ i j, i < j → a i < a j)
  (h_b_sorted : ∀ i j, i < j → b i > b j)
  (h_part : ∀ x ∈ (List.finRange n).map a ++ (List.finRange n).map b, x ∈ Finset.range (2 * n + 1))
  (h_disj : ∀ x, (x ∈ (List.finRange n).map a) ↔ ¬ (x ∈ (List.finRange n).map b)) :
  ∑ i, |a i - b i| = n * n := by
  sorry

end partition_sum_abs_eq_n_squared_l736_736924


namespace square_side_length_same_area_l736_736618

theorem square_side_length_same_area (length width : ℕ) (l_eq : length = 72) (w_eq : width = 18) : 
  ∃ side_length : ℕ, side_length * side_length = length * width ∧ side_length = 36 :=
by
  sorry

end square_side_length_same_area_l736_736618


namespace part1_part2_l736_736801

def z1 (a : ℝ) : Complex := Complex.mk 2 a
def z2 : Complex := Complex.mk 3 (-4)

-- Part 1: Prove that the product of z1 and z2 equals 10 - 5i when a = 1.
theorem part1 : z1 1 * z2 = Complex.mk 10 (-5) :=
by
  -- proof to be filled in
  sorry

-- Part 2: Prove that a = 4 when z1 + z2 is a real number.
theorem part2 (a : ℝ) (h : (z1 a + z2).im = 0) : a = 4 :=
by
  -- proof to be filled in
  sorry

end part1_part2_l736_736801


namespace sequence_explicit_formula_l736_736771

-- Define the sequence
def a : ℕ → ℤ
| 0     := 2
| (n+1) := a n - n - 1 + 3

-- Define the function to prove
def explicit_formula (n : ℕ) : ℤ := -(n * (n + 1)) / 2 + 3 * n + 2

-- The proof problem statement
theorem sequence_explicit_formula (n : ℕ) : a n = explicit_formula n :=
sorry

end sequence_explicit_formula_l736_736771


namespace part_I_part_II_l736_736297

noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem part_I (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ f x a = 0) → a ∈ Iio 0 := 
sorry

theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → f x a ≥ a) → a ∈ Iic 4 :=
sorry

end part_I_part_II_l736_736297


namespace total_marbles_l736_736516

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l736_736516


namespace new_foreign_students_l736_736561

theorem new_foreign_students 
  (total_students : ℕ)
  (percent_foreign : ℕ)
  (foreign_students_next_sem : ℕ)
  (current_foreign_students : ℕ := total_students * percent_foreign / 100) : 
  total_students = 1800 → 
  percent_foreign = 30 → 
  foreign_students_next_sem = 740 → 
  foreign_students_next_sem - current_foreign_students = 200 :=
by
  intros
  sorry

end new_foreign_students_l736_736561


namespace joint_purchases_popular_in_countries_joint_purchases_not_popular_in_building_l736_736935

-- Definitions using conditions from problem (a)
def cost_savings_info_sharing : Prop := 
  ∀ (groupBuy : Type), (significant_cost_savings : Prop) ∧ (accurate_product_evaluation : Prop)

-- Definitions using conditions from problem (b)
def transactional_costs_proximity : Prop := 
  ∀ (neighborhoodGroup : Type), 
    (diverse_products : Prop) ∧ 
    (significant_transactional_costs : Prop) ∧ 
    (organizational_burdens : Prop) ∧ 
    (proximity_to_stores : Prop)

-- Lean 4 statements to prove the questions
theorem joint_purchases_popular_in_countries :
  cost_savings_info_sharing → 
  (practice_of_joint_purchases_popular : Prop) := 
sorry

theorem joint_purchases_not_popular_in_building :
  transactional_costs_proximity → 
  (practice_of_joint_purchases_not_popular : Prop) :=
sorry

end joint_purchases_popular_in_countries_joint_purchases_not_popular_in_building_l736_736935


namespace find_sale_month_4_l736_736172

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4_l736_736172


namespace bricks_required_l736_736168

theorem bricks_required (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
  (brick_length_cm : ℕ) (brick_width_cm : ℕ)
  (h1 : courtyard_length_m = 30) (h2 : courtyard_width_m = 16)
  (h3 : brick_length_cm = 20) (h4 : brick_width_cm = 10) :
  (3000 * 1600) / (20 * 10) = 24000 :=
by sorry

end bricks_required_l736_736168


namespace tangent_line_equation_l736_736019

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P = (-4, -3)) :
  ∃ (a b c : ℝ), a * -4 + b * -3 + c = 0 ∧ a * a + b * b = (5:ℝ)^2 ∧ 
                 a = 4 ∧ b = 3 ∧ c = 25 := 
sorry

end tangent_line_equation_l736_736019


namespace line_intersects_plane_at_angle_l736_736283

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle_l736_736283


namespace exhibition_display_methods_l736_736461

theorem exhibition_display_methods :
  let booths := 7
  let exhibits := 3
  -- The function to count valid arrangements
  ∃ f : Fin (booths.choose exhibits), ∀ i j : Fin exhibits,
    i ≠ j → (2 ≤ |i.1 - j.1| ∧ |i.1 - j.1| ≤ 4) →
    (f (Fin.mk i.1 sorry) = f (Fin.mk j.1 sorry)) →
  cardinal.mk (set.univ : set (Fin booths)) = 42 :=
by
  sorry

end exhibition_display_methods_l736_736461


namespace opposite_of_neg_three_l736_736451

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l736_736451


namespace min_k_spherical_cap_cylinder_l736_736191

/-- Given a spherical cap and a cylinder sharing a common inscribed sphere with volumes V1 and V2 respectively,
we show that the minimum value of k such that V1 = k * V2 is 4/3. -/
theorem min_k_spherical_cap_cylinder (R : ℝ) (V1 V2 : ℝ) (h1 : V1 = (4/3) * π * R^3) 
(h2 : V2 = 2 * π * R^3) : 
∃ k : ℝ, V1 = k * V2 ∧ k = 4/3 := 
by 
  use (4/3)
  constructor
  . sorry
  . sorry

end min_k_spherical_cap_cylinder_l736_736191


namespace sum_of_six_selected_primes_is_even_l736_736407

noncomputable def prob_sum_even_when_selecting_six_primes : ℚ := 
  let first_twenty_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
  let num_ways_to_choose_6_without_even_sum := Nat.choose 19 6
  let total_num_ways_to_choose_6 := Nat.choose 20 6
  num_ways_to_choose_6_without_even_sum / total_num_ways_to_choose_6

theorem sum_of_six_selected_primes_is_even : 
  prob_sum_even_when_selecting_six_primes = 354 / 505 := 
sorry

end sum_of_six_selected_primes_is_even_l736_736407


namespace complex_conjugate_sum_l736_736710

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736710


namespace part_one_part_two_part_three_l736_736538

-- Part I
theorem part_one (a : ℕ → ℕ) (d : ℕ → ℕ) (h1 : d_n = a (n+2) + a n - 2 * a (n+1))
  (h2 : a 1 = 1) (h3 : ∀ n ≥ 1, d n = a n) (h4 : a 2 = 2) :
  ∀ n, a n = 2^(n-1) := sorry

-- Part II
theorem part_two (a : ℕ → ℤ) (d : ℕ → ℤ) (h1 : d_n = a (n+2) + a n - 2 * a (n+1))
  (h2 : a 1 = 1) (h3 : a 2 = -2) (h4 : ∀ n ≥ 1, d n ≥ 1) :
  ∀ n, a n ≥ -5 := sorry

-- Part III
theorem part_three (a : ℕ → ℤ) (d : ℕ → ℤ) (h1 : d_n = a (n+2) + a n - 2 * a (n+1))
  (h2 : a 1 = 1) (h3 : a 2 = 1) (h4 : ∀ n ≥ 1, |d n| = 1) (h5 : ∀ n ≥ 1, a (n+4) = a n) :
  ∃ sequences : ℕ → ℤ, sequences = [1,1,2,2] ∨ sequences = [1,1,0,0] := sorry

end part_one_part_two_part_three_l736_736538


namespace find_possible_values_l736_736374

noncomputable def complex_values (x y : ℂ) : Prop :=
  (x^2 + y^2) / (x + y) = 4 ∧ (x^4 + y^4) / (x^3 + y^3) = 2

theorem find_possible_values (x y : ℂ) (h : complex_values x y) :
  ∃ z : ℂ, z = (x^6 + y^6) / (x^5 + y^5) ∧ (z = 10 + 2 * Real.sqrt 17 ∨ z = 10 - 2 * Real.sqrt 17) :=
sorry

end find_possible_values_l736_736374


namespace x_squared_plus_y_squared_l736_736506

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := 
by
  sorry

end x_squared_plus_y_squared_l736_736506


namespace complex_conjugate_sum_l736_736733

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736733


namespace part_a_part_b_l736_736828

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l736_736828


namespace digit_difference_base2_150_950_l736_736914

def largest_power_of_2_lt (n : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ k, 2^k ≤ n).last' getLastRange

def base2_digits (n : ℕ) : ℕ := largest_power_of_2_lt n + 1

theorem digit_difference_base2_150_950 :
  base2_digits 950 - base2_digits 150 = 2 :=
by {
  sorry
}

end digit_difference_base2_150_950_l736_736914


namespace number_of_even_results_l736_736146

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736146


namespace fraction_equiv_l736_736480

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l736_736480


namespace britney_has_more_chickens_l736_736410

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end britney_has_more_chickens_l736_736410


namespace white_tiles_count_l736_736363

theorem white_tiles_count (total_tiles yellow_tiles purple_tiles : ℕ)
    (hy : yellow_tiles = 3)
    (hb : ∃ blue_tiles, blue_tiles = yellow_tiles + 1)
    (hp : purple_tiles = 6)
    (ht : total_tiles = 20) : 
    ∃ white_tiles, white_tiles = 7 :=
by
  obtain ⟨blue_tiles, hb_eq⟩ := hb
  let non_white_tiles := yellow_tiles + blue_tiles + purple_tiles
  have hnwt : non_white_tiles = 3 + (3 + 1) + 6,
  {
    rw [hy, hp, hb_eq],
    ring,
  }
  have hwt : total_tiles - non_white_tiles = 7,
  {
    rw ht,
    rw hnwt,
    norm_num,
  }
  use total_tiles - non_white_tiles,
  exact hwt,

end white_tiles_count_l736_736363


namespace find_interest_rate_l736_736419

theorem find_interest_rate
  (P : ℝ) (CI : ℝ) (T : ℝ) (n : ℕ)
  (comp_int_formula : CI = P * ((1 + (r / (n : ℝ))) ^ (n * T)) - P) :
  r = 0.099 :=
by
  have h : CI = 788.13 := sorry
  have hP : P = 5000 := sorry
  have hT : T = 1.5 := sorry
  have hn : (n : ℝ) = 2 := sorry
  sorry

end find_interest_rate_l736_736419


namespace result_prob_a_l736_736834

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l736_736834


namespace find_d_l736_736409

noncomputable def single_point_graph (d : ℝ) : Prop :=
  ∃ x y : ℝ, 3 * x^2 + 2 * y^2 + 9 * x - 14 * y + d = 0

theorem find_d : single_point_graph 31.25 :=
sorry

end find_d_l736_736409


namespace spring_chain_length_l736_736085

noncomputable def total_length_of_spring_chain (n : ℕ) (l0 : ℝ) (k : ℝ) (m : ℝ) (g : ℝ) : ℝ :=
  l0 * n + (∑ i in Finset.range (n + 1), (i * m * g) / k)

theorem spring_chain_length : total_length_of_spring_chain 10 0.5 200 2 9.8 = 10.39 :=
by
  sorry

end spring_chain_length_l736_736085


namespace q_evaluation_l736_736800

def q (x y : ℤ) : ℤ :=
if x ≥ 0 ∧ y ≤ 0 then x - y
else if x < 0 ∧ y > 0 then x + 3 * y
else 4 * x - 2 * y

theorem q_evaluation : q (q 2 (-3)) (q (-4) 1) = 6 :=
by
  sorry

end q_evaluation_l736_736800


namespace probability_real_part_greater_than_imaginary_part_l736_736128

theorem probability_real_part_greater_than_imaginary_part : 
  let dice_faces := {1, 2, 3, 4, 5, 6}
  let all_possible_outcomes := dice_faces.product dice_faces
  let favorable_outcomes := all_possible_outcomes.filter (λ (xy : ℕ × ℕ), xy.1 > xy.2)
  (favorable_outcomes.card / all_possible_outcomes.card : ℚ) = 5 / 12 :=
by
  have dice_faces_nonempty : dice_faces ≠ ∅ := by decide
  have all_possible_outcomes_card : all_possible_outcomes.card = 36 := by decide
  have favorable_outcomes_card : favorable_outcomes.card = 15 := by decide
  sorry

end probability_real_part_greater_than_imaginary_part_l736_736128


namespace white_tile_count_l736_736356

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l736_736356


namespace train_overtake_distance_l736_736083

/--
 Train A leaves the station traveling at 30 miles per hour.
 Two hours later, Train B leaves the same station traveling in the same direction at 42 miles per hour.
 Prove that Train A is overtaken by Train B 210 miles from the station.
-/
theorem train_overtake_distance
    (speed_A : ℕ) (speed_B : ℕ) (delay_B : ℕ)
    (hA : speed_A = 30)
    (hB : speed_B = 42)
    (hDelay : delay_B = 2) :
    ∃ d : ℕ, d = 210 ∧ ∀ t : ℕ, (speed_B * t = (speed_A * t + speed_A * delay_B) → d = speed_B * t) :=
by
  sorry

end train_overtake_distance_l736_736083


namespace problem_statement_l736_736788

-- Define the data and relationships:
variables {A B C D E F : Type} [Inhabited A]
variables (A B C D E F : A) 
variables (triangle : A → A → A → Prop)
variables (angle_right : A → A → A → Prop)

-- Define the conditions:
axiom angle_ACB_90 : angle_right A C B
axiom tangent_at_C : D -- tangent line at C intersecting line AB at D
axiom is_midpoint_E : E -- E is the midpoint of CD
axiom AF_parallel_CD : F -- AF is parallel to CD

-- State the proof problem:
theorem problem_statement : ∀ (A B C D E F : A), angle_right A C B → D → E → F → ∃ (perpendicular : Prop), perpendicular AB CF := 
by 
  sorry

end problem_statement_l736_736788


namespace number_of_even_results_l736_736149

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736149


namespace z_conjugate_sum_l736_736706

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736706


namespace find_fraction_l736_736490

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction_l736_736490


namespace evaluate_expression_at_3_l736_736225

theorem evaluate_expression_at_3 :
  ((3^(3^2))^(3^3)) = 3^(243) := 
by 
  sorry

end evaluate_expression_at_3_l736_736225


namespace wine_glass_movement_101_people_wine_glass_movement_100_people_l736_736787

variables (n : ℕ)

-- Part (a)
theorem wine_glass_movement_101_people : 
  ∀ (wine_colors : Fin 101 → Bool) 
    (h1 : ∃ i j, i ≠ j ∧ wine_colors i = true ∧ wine_colors j = false), 
  ∃ k, ∀ (glasses_post_midnight : Fin 101 → Bool), glasses_post_midnight k = false :=
begin
  intros wine_colors h1,
  -- Proof omitted for brevity
  sorry
end

-- Part (b)
theorem wine_glass_movement_100_people : 
  ∀ (wine_colors : Fin 100 → Bool) 
    (h1 : ∃ i j, i ≠ j ∧ wine_colors i = true ∧ wine_colors j = false), 
  ∃ k, ∀ (glasses_post_midnight : Fin 100 → Bool), glasses_post_midnight k = false :=
begin
  intros wine_colors h1,
  -- Proof omitted for brevity
  sorry
end

end wine_glass_movement_101_people_wine_glass_movement_100_people_l736_736787


namespace pine_count_25_or_26_l736_736063

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l736_736063


namespace even_combinations_result_in_486_l736_736167

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736167


namespace complex_conjugate_sum_l736_736729

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736729


namespace number_of_pines_possible_l736_736070

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l736_736070


namespace problem_conditions_l736_736329

variables {A B C : ℝ} {a b c : ℝ}

noncomputable def angle_B_solution :=
  (sin (2 * B) = sqrt 3 * sin B) →
  B = π / 6

noncomputable def triangle_shape_solution :=
  (a = 2 * b * cos C) →
  b = c

theorem problem_conditions
  (triangle_in_eq: angle_B_solution)
  (triangle_shape_eq: triangle_shape_solution)
  : triangle_in_eq ∧ triangle_shape_eq :=
by sorry

end problem_conditions_l736_736329


namespace find_m_l736_736289

noncomputable def complex_number := Complex ((4 : ℂ) + 2 * Complex.I) / ((1 + Complex.I) * (1 + Complex.I))

theorem find_m :
  let z : ℂ := complex_number in
  let x := z.re in
  let y := z.im in
  (x - 2 * y + m = 0) → m = -5 := by
  sorry

end find_m_l736_736289


namespace hexagon_diagonal_lt_two_l736_736877

theorem hexagon_diagonal_lt_two {A B C D E F : Type} 
  (hAB : dist A B < 1) 
  (hBC : dist B C < 1) 
  (hCD : dist C D < 1) 
  (hDE : dist D E < 1) 
  (hEF : dist E F < 1) 
  (hFA : dist F A < 1) 
  (hConvex : convex_hull (set.of {A, B, C, D, E, F}) == set.univ) : 
  (dist A D < 2) ∨ (dist B E < 2) ∨ (dist C F < 2) := 
sorry

end hexagon_diagonal_lt_two_l736_736877


namespace find_prob_X_greater_than_4_l736_736644

noncomputable def normal_distribution := sorry

variable {X : ℝ → ℝ}

axiom normal_X : X ~ normal_distribution(2, σ^2)
axiom prob_condition : P (0 < X ∧ X < 2) = 0.2

theorem find_prob_X_greater_than_4 : P (X > 4) = 0.3 :=
by
  sorry

end find_prob_X_greater_than_4_l736_736644


namespace find_principal_l736_736961

theorem find_principal (R : ℝ) (T : ℝ) (I : ℝ) (hR : R = 0.12) (hT : T = 1) (hI : I = 1500) :
  ∃ P : ℝ, I = P * R * T ∧ P = 12500 := 
by
  use 12500
  rw [hR, hT, hI]
  norm_num
  sorry

end find_principal_l736_736961


namespace find_m_value_l736_736861

theorem find_m_value (m : ℝ) :
  (∀ x y : ℝ, x^2 / 12 + y^2 / m = 1 ∧ (∀ a b c : ℝ, (a^2 = 12 ∧ b^2 = m → e = sqrt ((c^2) / 12) ∧ e = 1/2 → c^2 = 12 - m) ∨ 
  (a^2 = m ∧ b^2 = 12 → e = sqrt ((c^2) / m) ∧ e = 1/2 → c^2 = m - 12)) → m = 9 ∨ m = 16) :=
  sorry

end find_m_value_l736_736861


namespace negative_three_degrees_below_zero_l736_736343

-- Definitions based on conditions
def positive_temperature (t : ℤ) : Prop := t > 0
def negative_temperature (t : ℤ) : Prop := t < 0
def above_zero (t : ℤ) : Prop := positive_temperature t
def below_zero (t : ℤ) : Prop := negative_temperature t

-- Example given in conditions
def ten_degrees_above_zero := above_zero 10

-- Lean 4 statement for the proof
theorem negative_three_degrees_below_zero : below_zero (-3) :=
by
  sorry

end negative_three_degrees_below_zero_l736_736343


namespace probability_at_least_one_even_l736_736533

theorem probability_at_least_one_even 
  (cards : Finset ℕ) (h_card_count : cards.card = 9) (h_card_range : ∀ x ∈ cards, 1 ≤ x ∧ x ≤ 9) :
  let draws := cards.powerset.filter (λ s, s.card = 2)
  let evens := {x ∈ cards | x % 2 = 0}
  let favorable := draws.filter (λ s, ¬ s.disjoint evens)
  (favorable.card : ℚ) / (draws.card : ℚ) = 13/18 :=
sorry

end probability_at_least_one_even_l736_736533


namespace arithmetic_sequence_S9_l736_736760

-- Define the arithmetic sequence terms
def a (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1 : ℝ) * d
def S (n : ℕ) (a1 d : ℝ) : ℝ := n / 2 * (2 * a1 + (n - 1 : ℝ) * d)

-- Conditions given in the problem
def condition1 (a1 d : ℝ) : Prop := (a 4 a1 d) + (a 6 a1 d) = 12

-- Proof statement, where we prove the value of S_9 given the condition1
theorem arithmetic_sequence_S9 (a1 d : ℝ) (h : condition1 a1 d) : S 9 a1 d = 54 :=
by
  sorry

end arithmetic_sequence_S9_l736_736760


namespace min_value_expression_l736_736255

theorem min_value_expression (x : ℝ) : ∃ y, y = (x^2 + 7) / real.sqrt (x^2 + 3) ∧ y ≥ 4 :=
sorry

end min_value_expression_l736_736255


namespace hyperbola_properties_l736_736663

theorem hyperbola_properties :
  (∀ (m : ℝ), m > 0 → 
     (let C := ∀ (x y : ℝ), (x^2 / (m^2 + 3)) - (y^2 / m^2) = 1 in
      let asymptote := ∀ (x y : ℝ), y = (1/2) * x ∨ y = - (1/2) * x in
      
      -- Conditions
      asymptote (0, 0) ∧ asymptote (0, 0) →
      
      -- A: m = 1
      m = 1 ∧
      
      -- C: The curve y = ln(x-1) passes through a vertex of C
      ∃ (x : ℝ), C x (Real.ln (x-1)) ∧
      
      -- D: The hyperbola y^2 - (x^2 / 4) = 1 has the same asymptotes as C
      let D := ∀ (x y : ℝ), y^2 - (x^2 / 4) = 1 in
      (asymptote (0, 0) ∧ asymptote (0, 0))))
:= sorry

end hyperbola_properties_l736_736663


namespace mark_walks_distance_in_45_minutes_l736_736353

theorem mark_walks_distance_in_45_minutes :
  let rate := 1 / 20 -- Mark's walking rate in miles per minute
  let time := 45 -- Time in minutes
  let distance := time * rate -- Distance formula
  distance = 2.3 := 
by
  let rate := 1 / 20
  let time := 45
  let distance := time * rate
  have : distance = 45 * (1 / 20) := sorry
  have : distance = 2.25 := sorry
  have : distance_rounded := Real.round(distance * 10) / 10 := sorry
  have : distance_rounded = 2.3 := sorry
  sorry

end mark_walks_distance_in_45_minutes_l736_736353


namespace CircleF_tangent_to_B_C_D_l736_736592

theorem CircleF_tangent_to_B_C_D
  (A B C D F : Type)
  (rA : ℝ) (rA_eq : rA = 12)
  (rB : ℝ) (rB_eq : rB = 4)
  (rC : ℝ) (rC_eq : rC = 3)
  (rD : ℝ) (rD_eq : rD = 3)
  (rF : ℝ)
  (m n : ℕ)
  (rel_prime : nat.coprime m n)
  (rF_eq : rF = (m / n) ∧ n ≠ 0)
  (cond : circles_inscribed_and_tangent A B C D F P)
  : m + n = 111 := sorry

end CircleF_tangent_to_B_C_D_l736_736592


namespace count_prime_digits_with_prime_bars_l736_736684

def num_illuminated_bars (digit : ℕ) : ℕ :=
  match digit with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 3 => 5
  | 4 => 4
  | 5 => 5
  | 6 => 6
  | 7 => 3
  | 8 => 7
  | 9 => 6
  | _ => 0


def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_prime_digits_with_prime_bars :
  {n : ℕ | n < 10 ∧ is_prime n ∧ is_prime (num_illuminated_bars n)}.card = 4 :=
sorry

end count_prime_digits_with_prime_bars_l736_736684


namespace total_area_of_square_and_parallelogram_l736_736088

axiom area_square (a : ℕ) : ℕ
axiom area_parallelogram (b h : ℕ) : ℕ

theorem total_area_of_square_and_parallelogram : area_square 3 + area_parallelogram 3 2 = 15 := by
  have h1 : area_square 3 = 9 := sorry
  have h2 : area_parallelogram 3 2 = 6 := sorry
  rw [h1, h2]
  exact Nat.add_comm 9 6 ▸ rfl

end total_area_of_square_and_parallelogram_l736_736088


namespace find_a_and_decreasing_interval_l736_736293

def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2 + 1

theorem find_a_and_decreasing_interval (a : ℝ) :
  quadratic_function a 1 = 5 →
  (a = 2 ∨ a = -2) ∧ (∀ x : ℝ, x ≤ 0 → quadratic_function a x ≤ quadratic_function a 0) :=
by
  intro h,
  sorry

end find_a_and_decreasing_interval_l736_736293


namespace num_digits_difference_l736_736916

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end num_digits_difference_l736_736916


namespace minimum_handshakes_by_coaches_l736_736987

-- Define the initial conditions and the main goal
theorem minimum_handshakes_by_coaches (n m : ℕ) (h : n = 2 * m)
  (total_handshakes : ℕ) (handshake_eq : (m * (2 * m - 1) + 2 * m = 495)) :
  total_handshakes = 495 → ∃ k : ℕ, k = total_handshakes - (n * (n - 1) / 2) ∧ k = 60 :=
by
  -- Variables representing the number of players and teams
  have h1 : n = 2 * m := h
  have h2 : total_handshakes = 495 := by assumption
  -- Main equation for the handshakes
  have h3 := handshake_eq
  -- Ensuring the goal is correctly stated
  use (total_handshakes - n * (n - 1) / 2)
  sorry

end minimum_handshakes_by_coaches_l736_736987


namespace max_trading_cards_l736_736783

variable (money : ℝ) (cost_per_card : ℝ) (max_cards : ℕ)

theorem max_trading_cards (h_money : money = 9) (h_cost : cost_per_card = 1) : max_cards ≤ 9 :=
sorry

end max_trading_cards_l736_736783


namespace volleyball_team_selection_l736_736550

noncomputable def volleyball_squad_count (n m k : ℕ) : ℕ :=
  n * (Nat.choose m k)

theorem volleyball_team_selection :
  volleyball_squad_count 12 11 7 = 3960 :=
by
  sorry

end volleyball_team_selection_l736_736550


namespace base_of_parallelogram_l736_736252

theorem base_of_parallelogram (A h b : ℝ) (hA : A = 960) (hh : h = 16) :
  A = h * b → b = 60 :=
by
  sorry

end base_of_parallelogram_l736_736252


namespace bees_fly_one_km_l736_736919

theorem bees_fly_one_km (energy_per_liter : ℕ) (distance_per_bee_per_liter : ℕ) (h : distance_per_bee_per_liter = 7000) :
  let total_liters := 10 in
  let total_distance_for_all_bees := total_liters * distance_per_bee_per_liter in
  total_distance_for_all_bees / 1 = 70000 :=
by
  simp only [h]
  sorry

end bees_fly_one_km_l736_736919


namespace range_of_x_coordinate_of_A_l736_736277

theorem range_of_x_coordinate_of_A {A B : Type} [MetricSpace A]
  (O: A) (is_origin: O == (0, 0))
  (is_on_circle: ∀ (P : A), P ∈ circle -> dist O P = 1)
  (is_on_line: ∀ (Q : A), Q ∈ line -> 2 * Q.x - Q.y - 4 = 0)
  (exists_B: ∃ (B : A), B ∈ circle ∧ ∃ (A : A), A ∈ line ∧ 
    angle O A B == 30) : 
  ∃ (x : ℝ), x ∈ Icc (6/5) 2 :=
by
  sorry

end range_of_x_coordinate_of_A_l736_736277


namespace problem_part1_problem_part2_l736_736369

-- Definitions used in Lean 4

-- Representing N and its properties
def N (n : ℕ) (h : 2 ≤ n) := 2^n

-- Condition (1): For N = 16, x_7 should be at position 6 in P_2
theorem problem_part1 : ∀ (x : ℕ → ℕ), N 4 (by norm_num) = 16 → (NthPositionInP2 7 16 = 6) :=
sorry

-- Condition (2): For N = 2^n with n ≥ 8, x_173 should be at position 3 * 2^{n-4} + 11 in P_4
theorem problem_part2 : ∀ (n : ℕ) (x : ℕ → ℕ), 8 ≤ n → (NthPositionInP4 173 (2^n) = 3 * 2^(n-4) + 11) :=
sorry

-- Note: NthPositionInP2 and NthPositionInP4 are hypothetical functions
-- representing the transformations P_2 and P_4, respectively.

end problem_part1_problem_part2_l736_736369


namespace find_x_l736_736259

noncomputable def x : ℝ := 80 / 9

theorem find_x
  (hx_pos : 0 < x)
  (hx_condition : x * (⌊x⌋₊ : ℝ) = 80) :
  x = 80 / 9 :=
by
  sorry

end find_x_l736_736259


namespace max_value_of_expression_l736_736328

open Classical
open Real

theorem max_value_of_expression (a b : ℝ) (c : ℝ) (h1 : a^2 + b^2 = c^2 + ab) (h2 : c = 1) :
  ∃ x : ℝ, x = (1 / 2) * b + a ∧ x = (sqrt 21) / 3 := 
sorry

end max_value_of_expression_l736_736328


namespace min_value_frac_l736_736272

theorem min_value_frac (x y a b c d : ℝ) (hx : 0 < x) (hy : 0 < y)
  (harith : x + y = a + b) (hgeo : x * y = c * d) : (a + b) ^ 2 / (c * d) ≥ 4 := 
by sorry

end min_value_frac_l736_736272


namespace log_base_27_of_3_l736_736226

theorem log_base_27_of_3 : log 3 (27) (3) = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736226


namespace original_sum_of_money_l736_736193

theorem original_sum_of_money (P R : ℝ) 
  (h1 : 720 = P + (P * R * 2) / 100) 
  (h2 : 1020 = P + (P * R * 7) / 100) : 
  P = 600 := 
by sorry

end original_sum_of_money_l736_736193


namespace opposite_of_neg_three_l736_736448

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l736_736448


namespace pickup_carries_10_bags_per_trip_l736_736081

def total_weight : ℕ := 10000
def weight_one_bag : ℕ := 50
def number_of_trips : ℕ := 20
def total_bags : ℕ := total_weight / weight_one_bag
def bags_per_trip : ℕ := total_bags / number_of_trips

theorem pickup_carries_10_bags_per_trip : bags_per_trip = 10 := by
  sorry

end pickup_carries_10_bags_per_trip_l736_736081


namespace z_conjugate_sum_l736_736705

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736705


namespace find_CD_l736_736762

variable (A B C D : Type) [geometry_space A] [geometry_space B] [geometry_space C] [geometry_space D]
variable (AB BD AD CBD ABC : A → B → C → D → Type)

-- Given conditions
axiom ABC_eq_90 : ∀ (A B C : A), ∠BAC = 90
axiom CBD_eq_30 : ∀ (B C D : B), ∠BCD = 30
axiom AB_eq_BD : ∀ (A B : A), (AB A B) = 1 ∧ (BD B D) = 1
axiom C_on_AD : ∀ (A D : A), C = AD ∩ (segment A D)

theorem find_CD :
  let b := CD A D in b = 1 / sqrt(3) :=
by
  sorry

end find_CD_l736_736762


namespace composite_integer_expression_l736_736954

theorem composite_integer_expression (n : ℕ) (h : n > 1) (hn : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 :=
by
  sorry

end composite_integer_expression_l736_736954


namespace apples_in_boxes_l736_736109

theorem apples_in_boxes (apples_per_box : ℕ) (number_of_boxes : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_box = 12) (h2 : number_of_boxes = 90) : total_apples = 1080 :=
by
  sorry

end apples_in_boxes_l736_736109


namespace yen_received_correct_l736_736558

-- Exchange rate
def exchange_rate_AUD_to_JPY : ℝ := 8800 / 100

-- Transaction fee rate
def transaction_fee_rate : ℝ := 0.05

-- Amount of AUD to exchange
def amount_in_AUD : ℝ := 250

-- Function to calculate amount of JPY received after the transaction fee
def yen_received (aud: ℝ) (rate: ℝ) (fee: ℝ) : ℝ :=
  let raw_yen := aud * rate
  in raw_yen * (1 - fee)

-- Theorem to prove
theorem yen_received_correct : yen_received amount_in_AUD exchange_rate_AUD_to_JPY transaction_fee_rate = 20900 := 
by
  -- This part is intentionally left unproven
  sorry

end yen_received_correct_l736_736558


namespace ellipse_parameters_sum_l736_736862

theorem ellipse_parameters_sum 
  (h k a b : ℤ) 
  (h_def : h = 3) 
  (k_def : k = -5) 
  (a_def : a = 7) 
  (b_def : b = 2) : 
  h + k + a + b = 7 := 
by 
  -- definitions and sums will be handled by autogenerated proof
  sorry

end ellipse_parameters_sum_l736_736862


namespace factor_out_poly_l736_736982

open Polynomial

variable (x y : ℕ)

theorem factor_out_poly :
  (-6 * x^2 * y + 12 * x * y^2 - 3 * x * y) = -3 * x * y * (2 * x - 4 * y + 1) :=
by
  sorry

end factor_out_poly_l736_736982


namespace even_combinations_486_l736_736138

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736138


namespace scientific_notation_of_n_l736_736404

-- Define the given number
def n : ℝ := 0.000016

-- The theorem states that the given number in scientific notation is equal to another expression
theorem scientific_notation_of_n : n = 1.6 * 10^(-5) :=
by
  sorry

end scientific_notation_of_n_l736_736404


namespace z_conjugate_sum_l736_736694

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736694


namespace number_of_full_sequences_eq_factorial_l736_736189

-- Definition of a "full" sequence
def is_full_sequence {α : Type} [LinearOrder α] (seq : List α) : Prop :=
  ∀ k > 1, k ∈ seq → (seq.count (k-1) > seq.count k)

-- Function to count full sequences of a given length
noncomputable def count_full_sequences (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.factorial n

-- Main theorem stating that the number of full sequences of length n is n!
theorem number_of_full_sequences_eq_factorial 
  (n : ℕ) : 
  ∀ (seqs : List (List ℕ)), 
    (∀ seq ∈ seqs, seq.length = n ∧ is_full_sequence seq) → 
    seqs.length = Nat.factorial n :=
by
  sorry

end number_of_full_sequences_eq_factorial_l736_736189


namespace problem_statement_l736_736350

noncomputable def triangle_ratios (A B C a b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ a = Real.sqrt 3 * c ∧ b = c / (Real.sqrt 3)

theorem problem_statement : ∀ (A B C a b c : ℝ),
  triangle_ratios A B C a b c →
  a / b = Real.sqrt 3 :=
by
  intro A B C a b c h
  cases h with hA h1
  cases h1 with ha hb
  rw [ha, hb]
  sorry

end problem_statement_l736_736350


namespace geometry_problem_l736_736650

noncomputable def ellipse_c_eq(a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y : ℝ, (x / (real.sqrt a)) ^ 2 + (y / (real.sqrt b)) ^ 2 = 1

noncomputable def right_focus : Prop :=
  (2 : ℝ, 0 : ℝ)

noncomputable def point_p : Prop :=
  (2 : ℝ, (real.sqrt 6) / 3 : ℝ)

noncomputable def centroid_condition (k x y : ℝ) : Prop :=
  15 * k^4 + 2 * k^2 - 1 = 0

noncomputable def ellipse_c_specific : Prop :=
  ∀ x y : ℝ, (x/6) ^ 2 + (y/2) ^ 2 = 1

noncomputable def line_eq (k : ℝ) : Prop :=
  k = (real.sqrt 5) / 5 ∨ k = - (real.sqrt 5) / 5

theorem geometry_problem :
  ellipse_c_eq 6 2 ∧ point_p ∧ right_focus ∧
  ellipse_c_specific ∧ ∃ k, line_eq k :=
sorry

end geometry_problem_l736_736650


namespace volume_after_two_hours_l736_736981

-- Define constants and conditions
def ice_original_volume : ℝ := 6.4
def loss_fraction : ℝ := 3 / 4

-- Define the volume after each hour
def volume_after_first_hour : ℝ := (1 - loss_fraction) * ice_original_volume
def volume_after_second_hour : ℝ := (1 - loss_fraction) * volume_after_first_hour

-- Prove the final volume
theorem volume_after_two_hours : volume_after_second_hour = 0.4 :=
by
  sorry

end volume_after_two_hours_l736_736981


namespace min_value_polynomial_expression_at_k_eq_1_is_0_l736_736582

-- Definition of the polynomial expression
def polynomial_expression (k x y : ℝ) : ℝ :=
  3 * x^2 - 4 * k * x * y + (2 * k^2 + 1) * y^2 - 6 * x - 2 * y + 4

-- Proof statement
theorem min_value_polynomial_expression_at_k_eq_1_is_0 :
  (∀ x y : ℝ, polynomial_expression 1 x y ≥ 0) ∧ (∃ x y : ℝ, polynomial_expression 1 x y = 0) :=
by
  -- Expected proof here. For now, we indicate sorry to skip the proof.
  sorry

end min_value_polynomial_expression_at_k_eq_1_is_0_l736_736582


namespace euler_for_convex_polyhedron_l736_736498

variable {W : Type}
variables (𝒜 ℬ ℭ : ℕ)  -- ℱ = number of faces, ℬ = number of vertices, ℭ = number of edges (P)

theorem euler_for_convex_polyhedron (F B V : ℕ) : 
  W is_convex → 
  F - V + B = 2 :=
sorry

end euler_for_convex_polyhedron_l736_736498


namespace coins_collected_second_and_third_hours_l736_736244

-- Definitions
def coins_first_hour : ℕ := 15
def coins_fourth_hour_collected : ℕ := 50
def coins_given_to_coworker : ℕ := 15
def total_coins_after_fourth_hour : ℕ := 120

-- The proof goal
theorem coins_collected_second_and_third_hours (X : ℕ) :
  (15 + X + 50) - 15 = 120 → X = 70 := 
by
  intro h
  have h1 : 15 + X + 50 - 15 = X + 50 := by ring
  rw h1 at h
  linarith

end coins_collected_second_and_third_hours_l736_736244


namespace find_number_l736_736884

theorem find_number (x : ℤ) (h : 33 + 3 * x = 48) : x = 5 :=
by
  sorry

end find_number_l736_736884


namespace least_number_division_remainder_4_l736_736091

theorem least_number_division_remainder_4 : 
  ∃ n : Nat, (n % 6 = 4) ∧ (n % 130 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ∧ n = 2344 :=
by
  sorry

end least_number_division_remainder_4_l736_736091


namespace compute_expression_l736_736992

theorem compute_expression : 8^(-2 / 3) + log 10 (100 : ℝ) - (-7 / 8)^0 = 5 / 4 :=
  by
  sorry

end compute_expression_l736_736992


namespace value_range_x2_2x_minus_4_l736_736459

def eval_at (f : ℝ → ℝ) (x : ℝ) : ℝ := f x

noncomputable def function_range : set ℝ := {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) 2, y = (λ x, x^2 + 2 * x - 4) x}

theorem value_range_x2_2x_minus_4 :
  function_range = (set.Icc (-5 : ℝ) 4) :=
sorry

end value_range_x2_2x_minus_4_l736_736459


namespace red_paint_cans_needed_l736_736388

-- Definitions for the problem
def ratio_red_white : ℚ := 3 / 2
def total_cans : ℕ := 30

-- Theorem statement to prove the number of cans of red paint
theorem red_paint_cans_needed : total_cans * (3 / 5) = 18 := by 
  sorry

end red_paint_cans_needed_l736_736388


namespace right_triangle_area_l736_736647

theorem right_triangle_area (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) :
  (1/2) * (a : ℝ) * b = 30 :=
by
  sorry

end right_triangle_area_l736_736647


namespace probability_same_color_l736_736688

theorem probability_same_color (total_plates : ℕ) (red_plates : ℕ) (blue_plates : ℕ) (chosen_plates : ℕ) :
  total_plates = 11 → red_plates = 6 → blue_plates = 5 → chosen_plates = 3 →
  (nat.choose red_plates chosen_plates + nat.choose blue_plates chosen_plates) /
  nat.choose total_plates chosen_plates = 2 / 11 :=
by
  intro h_total h_red h_blue h_chosen
  rw [h_total, h_red, h_blue, h_chosen]
  sorry

end probability_same_color_l736_736688


namespace complex_conjugate_sum_l736_736715

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736715


namespace servant_leaves_salary_l736_736675

-- Definitions of the problem conditions
def one_year_cash_salary : ℝ := 90
def turban_value : ℝ := 110
def total_annual_salary : ℝ := one_year_cash_salary + turban_value
def months_worked : ℝ := 9
def total_months_in_year : ℝ := 12
def work_fraction := months_worked / total_months_in_year
def entitled_amount : ℝ := work_fraction * total_annual_salary
def turban_given := turban_value

-- Statement of the problem
theorem servant_leaves_salary :
  entitled_amount - turban_given = 40 := 
by
sory

end servant_leaves_salary_l736_736675


namespace solve_missing_figure_l736_736922

theorem solve_missing_figure (x : ℝ) (h : 0.25/100 * x = 0.04) : x = 16 :=
by
  sorry

end solve_missing_figure_l736_736922


namespace calculate_meals_l736_736812

-- Given conditions
def meal_cost : ℕ := 7
def total_spent : ℕ := 21

-- The expected number of meals Olivia's dad paid for
def expected_meals : ℕ := 3

-- Proof statement
theorem calculate_meals : total_spent / meal_cost = expected_meals :=
by
  sorry
  -- Proof can be completed using arithmetic simplification.

end calculate_meals_l736_736812


namespace abs_c_eq_181_l736_736008

theorem abs_c_eq_181
  (a b c : ℤ)
  (h_gcd : Int.gcd a (Int.gcd b c) = 1)
  (h_eq : a * (Complex.mk 3 2)^4 + b * (Complex.mk 3 2)^3 + c * (Complex.mk 3 2)^2 + b * (Complex.mk 3 2) + a = 0) :
  |c| = 181 :=
sorry

end abs_c_eq_181_l736_736008


namespace arithmetic_sequence_a5_l736_736646

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) = a n + 2

-- Statement of the theorem with conditions and conclusion
theorem arithmetic_sequence_a5 :
  ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ a 1 = 1 ∧ a 5 = 9 :=
by
  sorry

end arithmetic_sequence_a5_l736_736646


namespace car_speed_l736_736510

theorem car_speed (v t Δt : ℝ) (h1: 90 = v * t) (h2: 90 = (v + 30) * (t - Δt)) (h3: Δt = 0.5) : 
  ∃ v, 90 = v * t ∧ 90 = (v + 30) * (t - Δt) :=
by {
  sorry
}

end car_speed_l736_736510


namespace parabola_directrix_l736_736858

theorem parabola_directrix (y : ℝ) : (x : ℝ) (h : x^2 = 4 * y) → y = -1 :=
by
  sorry

end parabola_directrix_l736_736858


namespace rowing_speed_in_still_water_l736_736945

theorem rowing_speed_in_still_water
  (current_speed : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (row_distance_each_way : ℝ) :
  current_speed = 2 ∧ total_distance = 2 * row_distance_each_way ∧ row_distance_each_way = 3.5 ∧ total_time = 5 / 3 →
  ∃ x : ℝ, (x > 0) ∧ (row_distance_each_way / (x + current_speed) + row_distance_each_way / (x - current_speed) = total_time) ∧ x = 5 :=
by
  intro hyp
  have := and.elim_left hyp
  exact sorry

end rowing_speed_in_still_water_l736_736945


namespace lend_years_to_B_l736_736521

-- Define the principal amounts lent to B and C
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
-- Define the rate of interest per annum
def rate : ℝ := 13.75 / 100
-- Total interest received from B and C
def total_interest : ℝ := 2200
-- Number of years A lent to C
def time_C : ℝ := 4

-- Define the number of years A lent to B which we need to prove as 2
def time_B_to_prove : ℝ := 2

theorem lend_years_to_B :
  let interest_B := principal_B * rate * time_B_to_prove,
      interest_C := principal_C * rate * time_C in
  interest_B + interest_C = total_interest :=
by {
  sorry
}

end lend_years_to_B_l736_736521


namespace students_bought_tickets_l736_736080

theorem students_bought_tickets :
  ∃ S : ℕ, (6 * S + 8 * 12 = 216) ∧ (S = 20) :=
begin
  -- proof placeholder
  sorry
end

end students_bought_tickets_l736_736080


namespace female_employees_count_l736_736928

variable (E F M : ℕ)
variable (manager_fraction : ℚ)
variable (managers : ℕ)
variable (female_managers : ℕ)
variable (male_managers : ℕ)

axiom h1 : female_managers = 200
axiom h2 : manager_fraction = 2 / 5
axiom h3 : managers = manager_fraction * E
axiom h4 : male_managers = manager_fraction * M
axiom h5 : female_managers = F - male_managers
axiom h6 : E = F + M
axiom h7 : managers = female_managers + male_managers

theorem female_employees_count : F = 500 :=
by
sorries

end female_employees_count_l736_736928


namespace solve_basketball_court_dimensions_l736_736223

theorem solve_basketball_court_dimensions 
  (A B C D E F : ℕ) 
  (h1 : A - B = C) 
  (h2 : D = 2 * (A + B)) 
  (h3 : E = A * B) 
  (h4 : F = 3) : 
  A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 := 
by 
  sorry

end solve_basketball_court_dimensions_l736_736223


namespace percentage_of_number_l736_736527

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l736_736527


namespace maria_goal_l736_736988

theorem maria_goal (total_quizzes : ℕ) (required_percentage : ℚ) (initial_A : ℕ) (initial_quizzes : ℕ) : Prop :=
  let required_As := (required_percentage * total_quizzes).to_nat in
  let remaining_quizzes := total_quizzes - initial_quizzes in
  let additional_As_needed := required_As - initial_A in
  let non_A_remaining := remaining_quizzes - additional_As_needed in
  non_A_remaining = 11

#eval maria_goal 60 0.7 28 35 -- should return true

end maria_goal_l736_736988


namespace shane_current_age_l736_736894

theorem shane_current_age (Garret_age : ℕ) (h : Garret_age = 12) : 
  (let Shane_age_twenty_years_ago := 2 * Garret_age in
   let Shane_current := Shane_age_twenty_years_ago + 20 in
   Shane_current = 44) :=
by
  sorry

end shane_current_age_l736_736894


namespace tangent_line_at_pi_over_2_inequality_for_a_l736_736661

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem tangent_line_at_pi_over_2 :
  let π := Real.pi in
  ∀ (x : ℝ), f(π / 2) = π / 2 + 1 → 
  (∀ y, y = x + 1) ↔ y = Real.cos (π / 2) * (x - π / 2) + f(π / 2) :=
by
  sorry

theorem inequality_for_a
  (a : ℝ) : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ Real.pi / 2 →
  f(x) ≥ a * x * Real.cos x ↔ a ∈ Iic (2 : ℝ) :=
by
  sorry

end tangent_line_at_pi_over_2_inequality_for_a_l736_736661


namespace number_of_pines_possible_l736_736068

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l736_736068


namespace max_sum_sq_bound_l736_736274

theorem max_sum_sq_bound (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ k : Fin n, (a k / (a k + a (k + 1) % n))^2) ≥ if n = 3 then 3 / 4 else 1 :=
by
  sorry

end max_sum_sq_bound_l736_736274


namespace alice_palice_probability_l736_736199

theorem alice_palice_probability : 
  let alice_choices := [2, 4, 6, 8, 10]
  let palice_choices := [1, 3, 5, 7, 9]
  let total_outcomes := 25
  let favorable_outcomes := 14
  (∃ m n : ℕ, m + n = 39 ∧ Nat.gcd m n = 1 ∧ favorable_outcomes / total_outcomes = m / n) :=
begin
  sorry
end

end alice_palice_probability_l736_736199


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736826

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736826


namespace find_mean_of_two_l736_736574

-- Define the set of numbers
def numbers : List ℕ := [1879, 1997, 2023, 2029, 2113, 2125]

-- Define the mean of the four selected numbers
def mean_of_four : ℕ := 2018

-- Define the sum of all numbers
def total_sum : ℕ := numbers.sum

-- Define the sum of the four numbers with a given mean
def sum_of_four : ℕ := 4 * mean_of_four

-- Define the sum of the remaining two numbers
def sum_of_two (total sum_of_four : ℕ) : ℕ := total - sum_of_four

-- Define the mean of the remaining two numbers
def mean_of_two (sum_two : ℕ) : ℕ := sum_two / 2

-- Define the condition theorem to be proven
theorem find_mean_of_two : mean_of_two (sum_of_two total_sum sum_of_four) = 2047 := 
by
  sorry

end find_mean_of_two_l736_736574


namespace no_pairs_of_pyramids_l736_736586

theorem no_pairs_of_pyramids (n : ℕ) (hn : n ≥ 4) :
  ∀ (P₁ : Pyramid) (P₁.convex : Convex P₁) (P₂ : Pyramid) (h : P₂ = TriangularPyramid),
  ¬ ∃ (angles : Finset dihedral_angle) (h_angles : angles.card = 4),
    ∀ angle ∈ angles, angle ∈ P₁.dihedral_angles ∧ angle ∈ P₂.dihedral_angles := 
sorry

end no_pairs_of_pyramids_l736_736586


namespace number_of_students_taking_math_l736_736203

variable (totalPlayers physicsOnly physicsAndMath mathOnly : ℕ)
variable (h1 : totalPlayers = 15) (h2 : physicsOnly = 9) (h3 : physicsAndMath = 3)

theorem number_of_students_taking_math : mathOnly = 9 :=
by {
  sorry
}

end number_of_students_taking_math_l736_736203


namespace curvilinear_hexagon_decomposition_l736_736578

-- Mathematical definitions and conditions
def trapezoid (A B C D : Type) (P : A → B → C → D → Prop) : Prop :=
  P A B C D ∧ A ≠ B ∧ B ≠ C ∧ C ≠ D

def equilateral_triangle (A B C : Type) (Q : A → B → C → Prop) : Prop :=
  Q A B C ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Given conditions
def conditions (A B C D P Q : Type) : Prop :=
  trapezoid A B C D P ∧ 
  (length AB = 1) ∧ 
  (length BC = 1) ∧ 
  (length CD = 1) ∧ 
  (area_of P = area_of (equilateral_triangle A B C Q) (⟨A, B, C, Q⟩))

-- Proof statement
theorem curvilinear_hexagon_decomposition 
  (A B C D P Q : Type) 
  (H : conditions A B C D P Q) : 
  ∃ F : Type, is_convex_figure F ∧ area_of F = area_of (trapezoid A B C D P) :=
by sorry

end curvilinear_hexagon_decomposition_l736_736578


namespace sum_conjugate_eq_two_l736_736736

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736736


namespace min_value_3x_4y_l736_736850

open Real

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y = 5 :=
by
  sorry

end min_value_3x_4y_l736_736850


namespace horizontal_asymptote_l736_736220

def numerator := λ x : ℝ, 15 * x^5 + 7 * x^4 + 6 * x^3 + 2 * x^2 + x + 4
def denominator := λ x : ℝ, 4 * x^5 + 3 * x^4 + 9 * x^3 + 4 * x^2 + 2 * x + 1 
def rational_function := λ x : ℝ, numerator x / denominator x

theorem horizontal_asymptote : 
  ∃ L : ℝ, ∀ x : ℝ, x ≠ 0 → (tendsto (λ x, rational_function x) at_top (𝓝 L)) := 
begin
  use 15 / 4,
  sorry
end

end horizontal_asymptote_l736_736220


namespace ratio_perimeters_of_squares_l736_736901

theorem ratio_perimeters_of_squares (a b : ℝ) (h_diag : (a * Real.sqrt 2) / (b * Real.sqrt 2) = 2.5) : (4 * a) / (4 * b) = 10 :=
by
  sorry

end ratio_perimeters_of_squares_l736_736901


namespace shortest_edge_paths_count_l736_736998

-- Definitions from conditions
def is_rectangular_prism (length width height : ℕ) : Prop := 
  length > width ∧ width = height

variable (length width height : ℕ)
variable (A B : ℕ)

-- The proof problem statement
theorem shortest_edge_paths_count :
  is_rectangular_prism length width height ∧ length = 2 * width → 
  ∃ n : ℕ, n = 6 ∧ 
  (number_of_shortest_paths_from_A_to_B A B length width height = n) := sorry

end shortest_edge_paths_count_l736_736998


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736824

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736824


namespace race_permutations_l736_736554

-- Define the number of participants
def num_participants : ℕ := 4

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (n+1) => (n + 1) * factorial n

-- Theorem: Given 4 participants, the number of different possible orders they can finish the race is 24.
theorem race_permutations : factorial num_participants = 24 := by
  -- sorry added to skip the proof
  sorry

end race_permutations_l736_736554


namespace find_decreased_amount_l736_736125

variables (x y : ℝ)

axiom h1 : 0.20 * x - y = 6
axiom h2 : x = 50.0

theorem find_decreased_amount : y = 4 :=
by
  sorry

end find_decreased_amount_l736_736125


namespace consecutive_digits_product_square_l736_736180

theorem consecutive_digits_product_square (A : ℕ) (hA : 10^15 ≤ A ∧ A < 10^16) :
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ j ∧ j ≤ 16) ∧ (∃ (p : ℕ), digit_product A i j = p^2) :=
by sorry

/-- Helper function to extract the product of digits in a given range,
    assuming digits are indexed from 1 to 16 for a 16-digit number A -/
def digit_product (A : ℕ) (i j : ℕ) : ℕ :=
  -- Placeholder implementation; real implementation needed
  sorry

end consecutive_digits_product_square_l736_736180


namespace compare_neg_one_neg_sqrt_three_l736_736991

theorem compare_neg_one_neg_sqrt_three : -1 > -real.sqrt 3 :=
sorry

end compare_neg_one_neg_sqrt_three_l736_736991


namespace even_combinations_486_l736_736141

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736141


namespace reflected_ray_equation_l736_736185

-- Definitions for the given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Problem statement: proving equation of the reflected ray
theorem reflected_ray_equation : 
  ∀ x y : ℝ, incident_line x = y ∧ reflection_line x = y → x - 2*y - 1 = 0 :=
by
  sorry

end reflected_ray_equation_l736_736185


namespace cube_root_of_64_is_4_l736_736747

theorem cube_root_of_64_is_4 (x : ℝ) (h1 : 0 < x) (h2 : x^3 = 64) : x = 4 :=
by
  sorry

end cube_root_of_64_is_4_l736_736747


namespace find_interest_rate_l736_736497

theorem find_interest_rate (P r : ℝ) 
  (h1 : 460 = P * (1 + 3 * r)) 
  (h2 : 560 = P * (1 + 8 * r)) : 
  r = 0.05 :=
by
  sorry

end find_interest_rate_l736_736497


namespace two_pairs_equal_sum_l736_736605

theorem two_pairs_equal_sum (n : ℕ) (h1 : 2009 < n) : ∃ (k1 k2 k3 k4 : ℕ), 
  1 ≤ k1 ∧ k1 ≤ n ∧ 
  1 ≤ k2 ∧ k2 ≤ n ∧ 
  1 ≤ k3 ∧ k3 ≤ n ∧ 
  1 ≤ k4 ∧ k4 ≤ n ∧ 
  (k1 ≠ k2 ∧ k1 ≠ k3 ∧ k1 ≠ k4 ∧ k2 ≠ k3 ∧ k2 ≠ k4 ∧ k3 ≠ k4) ∧
  (\frac{k1}{n + 1 - k1} + \frac{k2}{n + 1 - k2}) = (\frac{k3}{n + 1 - k3} + \frac{k4}{n + 1 - k4}):= sorry

end two_pairs_equal_sum_l736_736605


namespace smallest_steps_l736_736577

def ceil : ℚ → ℤ := λ x, ⌈x⌉

theorem smallest_steps (n : ℕ) : 
  (ceil (n / 3) - ceil (n / 7) = 13) ∧ (11 ∣ n) → n = 69 :=
by
  sorry

end smallest_steps_l736_736577


namespace even_number_combinations_l736_736151

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736151


namespace greatest_common_divisor_l736_736899

theorem greatest_common_divisor (n : ℕ) (h1 : ∃ d : ℕ, d = gcd 180 n ∧ (∃ (l : List ℕ), l.length = 5 ∧ ∀ x : ℕ, x ∈ l → x ∣ d)) :
  ∃ x : ℕ, x = 27 :=
by
  sorry

end greatest_common_divisor_l736_736899


namespace bee_safe_flight_probability_l736_736958

noncomputable def probability_of_safe_flight (edge_length: ℝ) (safety_distance: ℝ) := 
  if safety_distance < edge_length/2 then 
    let safe_edge_length := edge_length - 2 * safety_distance in
    (safe_edge_length^3) / (edge_length^3)
  else 0

theorem bee_safe_flight_probability :
  probability_of_safe_flight 3 1 = 1 / 27 :=
by
  -- The detailed proof steps would be here.
  sorry

end bee_safe_flight_probability_l736_736958


namespace ratio_of_sides_l736_736780

theorem ratio_of_sides (a b : ℝ) (h1 : a + b = 3 * a) (h2 : a + b - Real.sqrt (a^2 + b^2) = (1 / 3) * b) : a / b = 1 / 2 :=
sorry

end ratio_of_sides_l736_736780


namespace range_of_a_l736_736664

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end range_of_a_l736_736664


namespace sum_series_decrease_l736_736918

theorem sum_series_decrease (n m : ℕ) (h1 : n = 100) (h2 : m = 200) :
  (∑ k in Finset.range (m - n + 1), (1 : ℚ) / (n + k)) > 
  (m - n + 1) * ((1 : ℚ) / 150) := 
by
  sorry

end sum_series_decrease_l736_736918


namespace coin_flip_probability_difference_l736_736095

theorem coin_flip_probability_difference :
  let p_4 := (Nat.choose 5 4) * (1/2:ℚ)^4 * (1/2:ℚ)^1 in
  let p_3 := (Nat.choose 5 3) * (1/2:ℚ)^3 * (1/2:ℚ)^2 in
  abs (p_4 - p_3) = 5 / 32 :=
by {
  let p_4 := (Nat.choose 5 4) * (1/2:ℚ)^4 * (1/2:ℚ)^1;
  let p_3 := (Nat.choose 5 3) * (1/2:ℚ)^3 * (1/2:ℚ)^2;
  show abs (p_4 - p_3) = 5 / 32,
  sorry
}

end coin_flip_probability_difference_l736_736095


namespace displacement_correct_l736_736264

-- Define the initial conditions of the problem
def init_north := 50
def init_east := 70
def init_south := 20
def init_west := 30

-- Define the net movements
def net_north := init_north - init_south
def net_east := init_east - init_west

-- Define the straight-line distance using the Pythagorean theorem
def displacement_AC := (net_north ^ 2 + net_east ^ 2).sqrt

theorem displacement_correct : displacement_AC = 50 := 
by sorry

end displacement_correct_l736_736264


namespace directrix_of_parabola_l736_736859

open Set

noncomputable theory

-- Define the problem and conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 4 * y ↔ y = x^2 / 4

def general_parabola (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 4 * p * y ↔ y = x^2 / (4 * p)

-- Define the Lean theorem statement
theorem directrix_of_parabola : parabola 1 → general_parabola 1 → ∃ y : ℝ, y = -1 :=
by
  sorry

end directrix_of_parabola_l736_736859


namespace opposite_of_neg3_is_3_l736_736438

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l736_736438


namespace opposite_of_neg_three_l736_736433

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l736_736433


namespace number_of_pines_l736_736075

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l736_736075


namespace number_of_pines_l736_736059

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l736_736059


namespace find_x_pos_int_l736_736743

theorem find_x_pos_int (x : ℕ) (h_pos : 0 < x) (h_eq : x! - (x-4)! = 120) : x = 5 :=
sorry

end find_x_pos_int_l736_736743


namespace book_order_l736_736406

def Book := Type

noncomputable def F : Book := sorry
noncomputable def B : Book := sorry
noncomputable def D : Book := sorry
noncomputable def C : Book := sorry
noncomputable def E : Book := sorry
noncomputable def A : Book := sorry

def on_top (x y : Book) : Prop := sorry -- x is directly on top of y
def covers (x y : Book) : Prop := sorry -- x covers y (i.e., x is above y, directly or indirectly)

axiom F_on_top : ∀ b : Book, on_top F b = false
axiom B_between_F_and_AC : on_top F B ∧ ¬ covers F B ∧ (covers B A ∧ covers B C)
axiom D_beneath_B_covers_E : covers B D ∧ (covers D E ∧ (covers E A))
axiom C_direct_beneath_F : on_top F C ∧ ¬ covers F C ∧ ¬ covers B C ∧ ¬ covers D C ∧ ¬ covers E C ∧ ¬ covers A C
axiom A_at_bottom : ∀ b : Book, b ≠ C → covers b A

theorem book_order : 
  ∃ order : List Book, order = [F, C, B, D, E, A] ∧ 
  (∀ (i : Nat) (b : Book), i < order.length - 1 → order.get i = b ↔ ∃ a : Book, on_top b a ∧ order.get (i+1) = a) := sorry

end book_order_l736_736406


namespace range_of_a_l736_736667

def quadratic_expression (a x : ℝ) : ℝ := a * x^2 - a * x - 2

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, quadratic_expression a x ≥ 0) ↔ (a ∈ set.Icc (-8 : ℝ) (0 : ℝ)) :=
sorry

end range_of_a_l736_736667


namespace even_combinations_486_l736_736143

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736143


namespace find_a_circle_line_intersection_l736_736291

theorem find_a_circle_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0)
  (h2 : ∀ x y : ℝ, x + 2 * y + 1 = 0) :
  a = 3 := 
sorry

end find_a_circle_line_intersection_l736_736291


namespace expected_winnings_l736_736974

noncomputable def probability_dist : Fin 6 → ℚ
| ⟨0, _⟩ := 1/3 -- 1 corresponds to 2 dollars
| ⟨1, _⟩ := 1/3 -- 2 corresponds to 2 dollars
| ⟨2, _⟩ := 1/3 -- 3 corresponds to 5 dollars
| ⟨3, _⟩ := 1/3 -- 4 corresponds to 5 dollars
| ⟨4, _⟩ := 1/6 -- 5 corresponds to -4 dollars
| ⟨5, _⟩ := 1/6 -- 6 corresponds to -2 dollars

noncomputable def winnings : Fin 6 → ℚ
| ⟨0, _⟩ := 2
| ⟨1, _⟩ := 2
| ⟨2, _⟩ := 5
| ⟨3, _⟩ := 5
| ⟨4, _⟩ := -4
| ⟨5, _⟩ := -2

noncomputable def expected_value (dist: Fin 6 → ℚ) (win: Fin 6 → ℚ) : ℚ :=
list.sum (list.map (λ i, dist i * win i) (list.fin_range 6))

theorem expected_winnings :
  expected_value probability_dist winnings = 4 / 3 :=
by sorry

end expected_winnings_l736_736974


namespace fewer_mpg_in_city_l736_736950

theorem fewer_mpg_in_city
  (highway_miles : ℕ)
  (city_miles : ℕ)
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (tank_size : ℝ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 32 →
  tank_size = 336 / 32 →
  highway_mpg = 462 / tank_size →
  (highway_mpg - city_mpg) = 12 :=
by
  intros h_highway_miles h_city_miles h_city_mpg h_tank_size h_highway_mpg
  sorry

end fewer_mpg_in_city_l736_736950


namespace sufficient_but_not_necessary_l736_736122

theorem sufficient_but_not_necessary (x : ℝ) :
  (x < -1 → x^2 - 1 > 0) ∧ (∃ x, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l736_736122


namespace vote_majority_is_160_l736_736337

-- Define the total number of votes polled
def total_votes : ℕ := 400

-- Define the percentage of votes polled by the winning candidate
def winning_percentage : ℝ := 0.70

-- Define the percentage of votes polled by the losing candidate
def losing_percentage : ℝ := 0.30

-- Define the number of votes gained by the winning candidate
def winning_votes := winning_percentage * total_votes

-- Define the number of votes gained by the losing candidate
def losing_votes := losing_percentage * total_votes

-- Define the vote majority
def vote_majority := winning_votes - losing_votes

-- Prove that the vote majority is 160 votes
theorem vote_majority_is_160 : vote_majority = 160 :=
sorry

end vote_majority_is_160_l736_736337


namespace triangle_area_l736_736424

theorem triangle_area (c : ℝ) :
  (∃ x h : ℝ, h = c * sqrt (sqrt 5 - 2) ∧ x = (c * (sqrt 5 - 1)) / 2 ∧ (c - x) / x = x / c) →
  let area := (c^2 * sqrt (sqrt 5 - 2)) / 2 in
  area = (c^2 * sqrt (sqrt 5 - 2)) / 2 :=
by
  intro h_def
  let area := (c^2 * sqrt (sqrt 5 - 2)) / 2
  have : area = (c^2 * sqrt (sqrt 5 - 2)) / 2 := sorry
  exact this

end triangle_area_l736_736424


namespace sam_initial_pennies_l736_736845

def initial_pennies_spent (spent: Nat) (left: Nat) : Nat :=
  spent + left

theorem sam_initial_pennies (spent: Nat) (left: Nat) : spent = 93 ∧ left = 5 → initial_pennies_spent spent left = 98 :=
by
  sorry

end sam_initial_pennies_l736_736845


namespace webinar_active_minutes_l736_736552

theorem webinar_active_minutes :
  let hours := 13
  let extra_minutes := 17
  let break_minutes := 22
  (hours * 60 + extra_minutes) - break_minutes = 775 := by
  sorry

end webinar_active_minutes_l736_736552


namespace lilly_can_buy_flowers_l736_736377

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l736_736377


namespace towels_maria_ended_up_with_l736_736108

theorem towels_maria_ended_up_with 
  (initial_green : ℕ) (initial_white : ℕ) (initial_blue : ℕ)
  (given_green : ℕ) (given_white : ℕ) (given_blue : ℕ)
  (initial_total : ℕ := initial_green + initial_white + initial_blue)
  (given_total : ℕ := given_green + given_white + given_blue)
  (remaining_total : ℕ := initial_total - given_total)
  (initial_green = 35) (initial_white = 21) (initial_blue = 15)
  (given_green = 22) (given_white = 14) (given_blue = 6) :
  remaining_total = 29 :=
by
  sorry

end towels_maria_ended_up_with_l736_736108


namespace number_of_true_propositions_is_one_l736_736462

def prism_with_parallelogram_base_is_parallelepiped : Prop := 
  ∀ (P : Prism), P.has_parallelogram_base → P.is_parallelepiped

def parallelepiped_with_rectangular_base_is_cuboid : Prop := 
  ∀ (Q : Parallelepiped), Q.has_rectangular_base → Q.is_cuboid

def right_prism_is_right_parallelepiped : Prop := 
  ∀ (R : RightPrism), R.is_right_parallelepiped

def equal_angles_pyramid_is_regular_pyramid : Prop := 
  ∀ (S : Pyramid), S.lateral_faces_have_equal_angles → S.is_regular

theorem number_of_true_propositions_is_one:
  (prism_with_parallelogram_base_is_parallelepiped ∧ 
  ¬parallelepiped_with_rectangular_base_is_cuboid ∧ 
  ¬right_prism_is_right_parallelepiped ∧ 
  ¬equal_angles_pyramid_is_regular_pyramid) →
  num_true_propositions = 1 :=
  sorry

end number_of_true_propositions_is_one_l736_736462


namespace twenty_four_is_75_percent_of_what_number_l736_736897

theorem twenty_four_is_75_percent_of_what_number :
  ∃ x : ℝ, 24 = (75 / 100) * x ∧ x = 32 :=
by {
  use 32,
  split,
  { norm_num },
  { norm_num }
} -- sorry

end twenty_four_is_75_percent_of_what_number_l736_736897


namespace boat_equation_l736_736341

-- Define the conditions given in the problem
def total_boats : ℕ := 8
def large_boat_capacity : ℕ := 6
def small_boat_capacity : ℕ := 4
def total_students : ℕ := 38

-- Define the theorem to be proven
theorem boat_equation (x : ℕ) (h0 : x ≤ total_boats) : 
  large_boat_capacity * (total_boats - x) + small_boat_capacity * x = total_students := by
  sorry

end boat_equation_l736_736341


namespace rowan_upstream_time_l736_736396

theorem rowan_upstream_time
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (still_water_speed : ℝ)
  (upstream_distance : ℝ)
  :
  downstream_distance = 26 ∧ downstream_time = 2 ∧ still_water_speed = 9.75 ∧ upstream_distance = 26 → 
  (upstream_distance / (still_water_speed - ((downstream_distance / downstream_time) - still_water_speed))) = 4 :=
by
  intros h
  cases h with d1 h; cases h with t1 h; cases h with s1 h; cases h with u1 h
  rw [d1, t1, s1, u1]
  sorry

end rowan_upstream_time_l736_736396


namespace cube_side_length_l736_736036

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l736_736036


namespace sampling_method_l736_736467

variables (high_income_families : ℕ) (middle_income_families : ℕ) (low_income_families : ℕ)
          (total_families_to_survey : ℕ) (sports_specialized_students : ℕ) (students_to_survey : ℕ)

/-- The sampling method for the given conditions -/
theorem sampling_method :
  high_income_families = 100 ∧ middle_income_families = 210 ∧ low_income_families = 90 ∧
  total_families_to_survey = 100 ∧ sports_specialized_students = 10 ∧ students_to_survey = 3 →
  "stratified sampling" ∧ "simple random sampling" :=
begin
  sorry
end

end sampling_method_l736_736467


namespace number_of_pines_possible_l736_736066

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l736_736066


namespace inverse_of_log2_l736_736025

theorem inverse_of_log2 (y x : ℝ) (h : x > 1) (hy : y = log 2 (x + 1)) : ∃ y, x = 2 ^ y - 1 :=
by {
  sorry
}

end inverse_of_log2_l736_736025


namespace log_base_27_of_3_l736_736227

theorem log_base_27_of_3 : log 3 (27) (3) = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736227


namespace z_conjugate_sum_l736_736701

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736701


namespace range_a_if_monotonically_decreasing_l736_736318

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

theorem range_a_if_monotonically_decreasing (a : ℝ) :
  (∀ x ∈ Icc (a-1) (a+1), f' x < 0) → (1 < a ∧ a ≤ 2)
:= sorry

end range_a_if_monotonically_decreasing_l736_736318


namespace segment_AB_division_ratio_l736_736472

noncomputable def ratio_of_AB_division (α : ℝ) : ℝ × ℝ × ℝ := sorry

theorem segment_AB_division_ratio (α : ℝ) (A B : ℝ) (C1 C2 : set ℝ) 
  (h1 : (C1 ∩ C2).nonempty) (h2 : C1 ∩ C2 = ∅)
  (h3 : ∃ r s : ℝ, r ∈ C1 ∧ s ∈ C2 ∧ dist r s = 2 * real.sqrt (real.cos α * real.sin α)) :
  ratio_of_AB_division α = (real.cos α ^ 2, real.sin α ^ 2, real.cos α ^ 2) := sorry

end segment_AB_division_ratio_l736_736472


namespace triangle_area_l736_736907

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (6, 3)
def C : ℝ × ℝ := (4, 9)

def base : ℝ := B.1 - A.1
def height : ℝ := C.2 - A.2

def area_of_triangle (a b h : ℝ) : ℝ := 0.5 * b * h

theorem triangle_area : area_of_triangle A B height = 15 := sorry

end triangle_area_l736_736907


namespace number_of_pines_possible_l736_736067

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l736_736067


namespace z_conjugate_sum_l736_736724

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736724


namespace monthly_salary_l736_736114

variables (S : ℕ) (h1 : S * 20 / 100 * 96 / 100 = 4 * 250)

theorem monthly_salary : S = 6250 :=
by sorry

end monthly_salary_l736_736114


namespace smallest_n_has_3000_solutions_l736_736573

def fractional_part (x : ℝ) : ℝ := x - Real.floor x

def f (x : ℝ) : ℝ := abs (3 * fractional_part x - 1.5)

def equation_n_solutions (n : ℕ) : Prop :=
  ∃ (x : ℝ), nf (f (x * f x)) = x

theorem smallest_n_has_3000_solutions :
  ∃ n : ℕ, (equation_n_solutions n) ∧ n = 1000 := sorry

end smallest_n_has_3000_solutions_l736_736573


namespace fireworks_set_off_l736_736355

-- Define initial quantities
def initial_firecrackers : ℕ := 48
def initial_sparklers : ℕ := 30

-- Define confiscation rates
def firecrackers_confiscated_rate : ℚ := 0.25
def sparklers_confiscated_rate : ℚ := 0.10

-- Define defect rates
def firecrackers_defect_rate : ℚ := 1/6
def sparklers_defect_rate : ℚ := 1/4

-- Define rates of setting off good fireworks
def firecrackers_set_off_rate : ℚ := 1/2
def sparklers_set_off_rate : ℚ := 2/3

theorem fireworks_set_off : 
  let initial_firecrackers := 48 in
  let initial_sparklers := 30 in
  let firecrackers_confiscated := initial_firecrackers * firecrackers_confiscated_rate in
  let sparklers_confiscated := initial_sparklers * sparklers_confiscated_rate in
  let remaining_firecrackers := initial_firecrackers - firecrackers_confiscated in
  let remaining_sparklers := initial_sparklers - sparklers_confiscated in
  let defective_firecrackers := remaining_firecrackers * firecrackers_defect_rate in
  let defective_sparklers := remaining_sparklers * sparklers_defect_rate in
  let good_firecrackers := remaining_firecrackers - defective_firecrackers in
  let good_sparklers := remaining_sparklers - defective_sparklers in
  let firecrackers_set_off := good_firecrackers * firecrackers_set_off_rate in
  let sparklers_set_off := good_sparklers * sparklers_set_off_rate in
  firecrackers_set_off + sparklers_set_off = 29 := 
by
  sorry

end fireworks_set_off_l736_736355


namespace min_coins_cover_99_l736_736903

def coin_values : List ℕ := [1, 5, 10, 25, 50]

noncomputable def min_coins_cover (n : ℕ) : ℕ := sorry

theorem min_coins_cover_99 : min_coins_cover 99 = 9 :=
  sorry

end min_coins_cover_99_l736_736903


namespace fraction_equivalent_l736_736482

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l736_736482


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736827

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736827


namespace value_of_expression_l736_736488

theorem value_of_expression :
  (3150 - 3030)^2 / 144 = 100 :=
by {
  -- This imported module allows us to use basic mathematical functions and properties
  sorry -- We use sorry to skip the actual proof
}

end value_of_expression_l736_736488


namespace find_m_for_q_find_m_for_pq_l736_736307

variable (m : ℝ)

-- Statement q: The equation represents a hyperbola if and only if m > 3
def q (m : ℝ) : Prop := m > 3

-- Statement p: The inequality holds if and only if m >= 1
def p (m : ℝ) : Prop := m ≥ 1

-- 1. If statement q is true, find the range of values for m.
theorem find_m_for_q (h : q m) : m > 3 := by
  exact h

-- 2. If (p ∨ q) is true and (p ∧ q) is false, find the range of values for m.
theorem find_m_for_pq (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end find_m_for_q_find_m_for_pq_l736_736307


namespace calculator_key_sequence_l736_736011

theorem calculator_key_sequence :
  ∀ x₀ : ℝ,
  (x₀ = 7) →
  (∀ (n : ℕ), ∃ x : ℝ, x = nat.iterate (λ x, 1 / (1 - x)) n x₀) →
  nat.iterate (λ x, 1 / (1 - x)) 102 x₀ = 7 :=
by
  intros x₀ h₀ h_iter
  sorry

end calculator_key_sequence_l736_736011


namespace area_of_semicircle_l736_736535

-- Defining given conditions
def rectangle_side_1 : ℝ := 1
def rectangle_side_2 : ℝ := 3
def diameter : ℝ := rectangle_side_2
def radius : ℝ := diameter / 2
def full_circle_area : ℝ := Real.pi * radius^2
def semicircle_area : ℝ := full_circle_area / 2

-- Theorem to be proved
theorem area_of_semicircle : 
  semicircle_area = (9 * Real.pi) / 2 := 
by
sory

end area_of_semicircle_l736_736535


namespace cube_side_length_l736_736035

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l736_736035


namespace minimum_toothpicks_to_remove_l736_736608

-- Definitions related to the problem statement
def total_toothpicks : Nat := 40
def initial_triangles : Nat := 36

-- Ensure that the minimal number of toothpicks to be removed to destroy all triangles is correct.
theorem minimum_toothpicks_to_remove : ∃ (n : Nat), n = 15 ∧ (∀ (t : Nat), t ≤ total_toothpicks - n → t = 0) :=
sorry

end minimum_toothpicks_to_remove_l736_736608


namespace tan_angle_addition_l736_736692

variable (α β : ℝ) (tan_alpha : ℝ) (tan_beta : ℝ)

def tan_add (α β : ℝ) := (Real.tan α + Real.tan β) / (1 - Real.tan α * Real.tan β)

theorem tan_angle_addition (h1: tan_alpha = 5) (h2: tan_beta = 3) : tan_add α β = -4 / 7 := 
by {
  -- Insert the conditions
  have h1' : Real.tan α = 5 := h1,
  have h2' : Real.tan β = 3 := h2,
  -- Define the tan addition
  let result := (5 + 3) / (1 - 5 * 3),
  -- Simplify the result
  have simplification : result = -4 / 7,
  sorry,
}

end tan_angle_addition_l736_736692


namespace c_plus_d_l736_736745

theorem c_plus_d (a b c d : ℝ) (h1 : a + b = 11) (h2 : b + c = 9) (h3 : a + d = 5) :
  c + d = 3 + b :=
by
  sorry

end c_plus_d_l736_736745


namespace diagonal_cubes_in_solid_l736_736508

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def count_diagonal_cubes (a b c : ℕ) : ℕ := 
  a + b + c 
  - gcd a b 
  - gcd b c 
  - gcd c a 
  + gcd (gcd a b) c

theorem diagonal_cubes_in_solid : count_diagonal_cubes 105 140 195 = 395 := 
by 
   sorry

end diagonal_cubes_in_solid_l736_736508


namespace amount_paid_for_grapes_l736_736200

-- Definitions based on the conditions
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := 2.23

-- The statement to be proved
theorem amount_paid_for_grapes : total_spent + refund_for_cherries = 12.08 := 
by 
  -- Here the specific mathematical proof would go, but is replaced by sorry as instructed
  sorry

end amount_paid_for_grapes_l736_736200


namespace max_red_line_intersections_l736_736570

-- Define the conditions
def is_perpendicular {A B C D : Type} (l₁ l₂ : A × B) (p₁ p₂ : C × D) : Prop := sorry
def is_parallel {A B C D : Type} (l₁ l₂ : A × B) (p₁ p₂ : C × D) : Prop := sorry

-- Definition stating there are five points in the plane
def points : Type := {X : Type // X = X_1 ∨ X = X_2 ∨ X = X_3 ∨ X = X_4 ∨ X = X_5}

-- Definition of blue lines such that no two are parallel or perpendicular
def blue_lines (p : points) (q : points) : Type := 
{l : (p × q) // ¬(is_parallel l l) ∧ ¬(is_perpendicular l l)}

-- Definition of red lines that are perpendicular to each blue line not passing through the point
def red_lines (p : points) : Type := 
  {r : (p × (Σ q : points, blue_lines p q)) // is_perpendicular r (p, _ )}

-- Theorem to prove
theorem max_red_line_intersections : 
  ∀ X_1 X_2 X_3 X_4 X_5 : Type, 
  ∀ red_lines : points → List (Σ p : points, red_lines p), 
  (Σ X: points, red_lines X) → 
  (Σ X: points, blue_lines X) →
  (Σ (X : points), ¬is_parallel X X) →
  (Σ (X : points), ¬is_perpendicular X X) →
  Σ red_lines, red_lines X → 315 :=
sorry

end max_red_line_intersections_l736_736570


namespace calculate_parallelepiped_properties_l736_736761

variables (projection_base : ℝ) (height : ℝ) (area_rhombus : ℝ) (diagonal_1 : ℝ)

def lateral_surface_area (projection_base height area_rhombus diagonal_1 : ℝ) : ℝ :=
  let lateral_edge := Real.sqrt (height^2 + projection_base^2) in
  let diagonal_2 := (2 * area_rhombus) / diagonal_1 in
  let side := Real.sqrt ((diagonal_1 / 2)^2 + (diagonal_2 / 2)^2) in
  let perimeter := 4 * side in
  perimeter * lateral_edge

def volume_parallelepiped (projection_base height area_rhombus diagonal_1 : ℝ) : ℝ :=
  let lateral_edge := Real.sqrt (height^2 + projection_base^2) in
  area_rhombus * lateral_edge

theorem calculate_parallelepiped_properties 
  (h_proj : projection_base = 5) 
  (h_height : height = 12) 
  (h_area : area_rhombus = 24) 
  (h_diag1 : diagonal_1 = 8) :
  lateral_surface_area projection_base height area_rhombus diagonal_1 = 260 ∧
  volume_parallelepiped projection_base height area_rhombus diagonal_1 = 312 :=
by 
  sorry

end calculate_parallelepiped_properties_l736_736761


namespace polynomial_division_remainder_correct_l736_736585

theorem polynomial_division_remainder_correct (k a : ℚ) 
  (h₁ : k = 14 / 3) 
  (h₂ : a = 94 / 9) :
  let f := (λ x : ℚ, x^4 - 8 * x^3 + 20 * x^2 - 28 * x + 12)
  let g := (λ x : ℚ, x^2 - 3 * x + k)
  let r := (λ x : ℚ, x + a)
  ∃ q : ℚ → ℚ, f = λ x, g x * q x + r x :=
sorry

end polynomial_division_remainder_correct_l736_736585


namespace prob_neither_A_nor_B_l736_736046

theorem prob_neither_A_nor_B
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ)
  (h1 : P_A = 0.25) (h2 : P_B = 0.30) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.60 :=
by
  sorry

end prob_neither_A_nor_B_l736_736046


namespace trapezoid_rectangle_area_ratio_l736_736332

theorem trapezoid_rectangle_area_ratio
  (b : ℝ) 
  (PQ_RS_SQ_decreasing : ∀ d : ℝ, 0 < d → 
    let SZ := Real.sqrt (b^2 - d^2);
    let QZ := Real.sqrt (b^2 - (2*d)^2);
    let area_trapezoid := d * (Real.sqrt (b^2 - d^2) + Real.sqrt (b^2 - 4 * d^2));
    let area_rectangle := 2 * d * Real.sqrt (b^2 - d^2);
    area_ratio := area_trapezoid / area_rectangle;
    has_limit (λ d, area_ratio) 0 1) 
: true :=
sorry

end trapezoid_rectangle_area_ratio_l736_736332


namespace log_base_27_of_3_l736_736229

theorem log_base_27_of_3 : log 3 (27) (3) = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736229


namespace sequence_explicit_formula_l736_736770

-- Define the sequence
def a : ℕ → ℤ
| 0     := 2
| (n+1) := a n - n - 1 + 3

-- Define the function to prove
def explicit_formula (n : ℕ) : ℤ := -(n * (n + 1)) / 2 + 3 * n + 2

-- The proof problem statement
theorem sequence_explicit_formula (n : ℕ) : a n = explicit_formula n :=
sorry

end sequence_explicit_formula_l736_736770


namespace min_value_of_f_l736_736866

noncomputable def f (x : ℝ) : ℝ :=
  1 / (Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_l736_736866


namespace median_of_consecutive_integers_l736_736052

theorem median_of_consecutive_integers (sum_total : ℤ) (n : ℤ) (sum_eq : sum_total = 5^5) (num_eq : n = 36) :
  let mean := sum_total / n in mean = 89 :=
by
  sorry

end median_of_consecutive_integers_l736_736052


namespace basketball_team_lineup_l736_736129

noncomputable def binom := Nat.choose

theorem basketball_team_lineup :
  let players := 15
  let quadruplets := ['Alex, 'Arthur, 'Anne, 'Amy]
  (card quadruplets = 4) ∧
  let starters := 6
  let remaining_players := players - card quadruplets
  (remaining_players = 11)
  let no_quadruplet_ways := binom remaining_players starters
  (no_quadruplet_ways = binom 11 6) ∧
  let one_quadruplet_ways := 4 * binom remaining_players (starters - 1)
  (one_quadruplet_ways = 4 * binom 11 5) ∧
  let two_quadruplet_ways := binom (card quadruplets) 2 * binom remaining_players (starters - 2)
  (two_quadruplet_ways = binom 4 2 * binom 11 4) ∧
  let total_ways := no_quadruplet_ways + one_quadruplet_ways + two_quadruplet_ways
  total_ways = 4290 := by
  sorry

end basketball_team_lineup_l736_736129


namespace curve_is_line_l736_736600

theorem curve_is_line (θ : ℝ) (hθ : θ = 5 * Real.pi / 6) : 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), r = 0 ↔
  (∃ p : ℝ × ℝ, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧
                p.1 * a + p.2 * b = 0) :=
sorry

end curve_is_line_l736_736600


namespace shares_of_valuable_stock_l736_736809

theorem shares_of_valuable_stock 
  (price_val : ℕ := 78)
  (price_oth : ℕ := 39)
  (shares_oth : ℕ := 26)
  (total_asset : ℕ := 2106)
  (x : ℕ) 
  (h_val_stock : total_asset = 78 * x + 39 * 26) : 
  x = 14 :=
by
  sorry

end shares_of_valuable_stock_l736_736809


namespace BF_eq_DG_l736_736366

-- Definitions for the problem
variables {A B C D O P F G : Type*}

-- Assuming A, B, C, D form a trapezoid and are inscribed in a circle with center O
noncomputable def is_trapezoid (A B C D : Type*) : Prop := sorry
noncomputable def inscribed_in_circle (A B C D : Type*) (O : Type*) : Prop := sorry
noncomputable def intersection (P : Type*) (BC AD : Type*) : Prop := sorry
noncomputable def circle_through (O P F G : Type*) : Prop := sorry

-- Additional assumptions
axiom trapezoid_ABCD : is_trapezoid A B C D
axiom inscribed_ABCD : inscribed_in_circle A B C D O
axiom intersection_P : intersection P (BC := B) (AD := A)
axiom circle_OP_intersects : circle_through O P F G

-- The goal to prove
theorem BF_eq_DG :
  B = F ∧ D = G → BF = DG := sorry

end BF_eq_DG_l736_736366


namespace opposite_of_neg3_l736_736444

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l736_736444


namespace opposite_of_neg3_is_3_l736_736440

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l736_736440


namespace z_conjugate_sum_l736_736722

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736722


namespace hyperbola_eccentricity_l736_736288

theorem hyperbola_eccentricity (m : ℝ) (h : m = Real.sqrt (2 * 8)) :
  let a := 1
  let b := Real.sqrt m
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  Real.sqrt (c ^ 2 / a ^ 2) = √5 :=
by
  sorry

end hyperbola_eccentricity_l736_736288


namespace domain_log_sin_sqrt_l736_736253

theorem domain_log_sin_sqrt (x : ℝ) : 
  (2 < x ∧ x < (5 * Real.pi) / 3) ↔ 
  (∃ k : ℤ, (Real.pi / 3) + (4 * k * Real.pi) < x ∧ x < (5 * Real.pi / 3) + (4 * k * Real.pi) ∧ 2 < x) :=
by
  sorry

end domain_log_sin_sqrt_l736_736253


namespace solution_form_l736_736119

noncomputable def required_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) ≤ (x * f y + y * f x) / 2

theorem solution_form (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 < x → 0 < f x) : required_function f → ∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 < x → f x = a * x :=
by
  intros
  sorry

end solution_form_l736_736119


namespace living_room_area_correct_minimum_tiles_needed_l736_736425

-- Problem definitions based on conditions
def length : ℝ := 5.2
def ratio : ℝ := 1.3
def width : ℝ := length / ratio

-- Define the area of the living room
def living_room_area : ℝ := length * width

def tile_side : ℝ := 0.4
def tile_area : ℝ := tile_side ^ 2
def num_tiles : ℝ := living_room_area / tile_area

-- Theorem statements
theorem living_room_area_correct : living_room_area = 20.8 := by sorry

theorem minimum_tiles_needed : num_tiles = 130 := by sorry

end living_room_area_correct_minimum_tiles_needed_l736_736425


namespace algebra_sum_l736_736026

-- Given conditions
def letterValue (ch : Char) : Int :=
  let pos := ch.toNat - 'a'.toNat + 1
  match pos % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 0 => -2
  | _ => 0  -- This case is actually unreachable.

def wordValue (w : List Char) : Int :=
  w.foldl (fun acc ch => acc + letterValue ch) 0

theorem algebra_sum : wordValue ['a', 'l', 'g', 'e', 'b', 'r', 'a'] = 0 :=
  sorry

end algebra_sum_l736_736026


namespace area_of_octagon_in_square_l736_736187

theorem area_of_octagon_in_square (perimeter : ℝ) (side_length : ℝ) (area_square : ℝ)
  (segment_length : ℝ) (area_triangle : ℝ) (total_area_triangles : ℝ) :
  perimeter = 144 →
  side_length = perimeter / 4 →
  segment_length = side_length / 3 →
  area_triangle = (segment_length * segment_length) / 2 →
  total_area_triangles = 4 * area_triangle →
  area_square = side_length * side_length →
  (area_square - total_area_triangles) = 1008 :=
by
  sorry

end area_of_octagon_in_square_l736_736187


namespace cubic_polynomial_root_l736_736854

theorem cubic_polynomial_root (p q r : ℕ) (h1 : 27 * x ^ 3 - 6 * x ^ 2 - 6 * x - 2 = 0) (h2 : x = (cbrt p + cbrt q + 2) / r) :
  p + q + r = 782 := 
sorry

end cubic_polynomial_root_l736_736854


namespace complement_intersection_l736_736304

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {3, 4, 5}) (hN : N = {2, 3}) :
  (U \ N) ∩ M = {4, 5} := by
  sorry

end complement_intersection_l736_736304


namespace k_value_and_set_exists_l736_736882

theorem k_value_and_set_exists
  (x1 x2 x3 x4 : ℚ)
  (h1 : (x1 + x2) / (x3 + x4) = -1)
  (h2 : (x1 + x3) / (x2 + x4) = -1)
  (h3 : (x1 + x4) / (x2 + x3) = -1)
  (hne : x1 ≠ x2 ∨ x1 ≠ x3 ∨ x1 ≠ x4 ∨ x2 ≠ x3 ∨ x2 ≠ x4 ∨ x3 ≠ x4) :
  ∃ (A B C : ℚ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ x1 = A ∧ x2 = B ∧ x3 = C ∧ x4 = -A - B - C := 
sorry

end k_value_and_set_exists_l736_736882


namespace sum_of_distinct_prime_divisors_1728_l736_736487

theorem sum_of_distinct_prime_divisors_1728 : 
  ∑ p in ((unique_factorization_monoid.factors 1728).to_finset).filter nat.prime, p = 5 :=
by
  sorry

end sum_of_distinct_prime_divisors_1728_l736_736487


namespace difference_in_interest_rates_l736_736975

-- Definitions
def Principal : ℝ := 2300
def Time : ℝ := 3
def ExtraInterest : ℝ := 69

-- The difference in rates
theorem difference_in_interest_rates (R dR : ℝ) (h : (Principal * (R + dR) * Time) / 100 =
    (Principal * R * Time) / 100 + ExtraInterest) : dR = 1 :=
  sorry

end difference_in_interest_rates_l736_736975


namespace solve_inequality_l736_736005

theorem solve_inequality (k x : ℝ) :
  (x^2 > (k + 1) * x - k) ↔ 
  (if k > 1 then (x < 1 ∨ x > k)
   else if k = 1 then (x ≠ 1)
   else (x < k ∨ x > 1)) :=
by
  sorry

end solve_inequality_l736_736005


namespace prob_A_is_3_over_19_l736_736020

variables (P : Set → ℝ) (A B : Set)

-- Conditions
axiom independent : ∀ {A B}, P (A ∩ B) = P A * P B
axiom prob_A_pos : 0 < P A
axiom prob_A_twice_B : P A = 2 * P B
axiom prob_union_is_18_times_inter : P (A ∪ B) = 18 * P (A ∩ B)

-- The theorem to prove
theorem prob_A_is_3_over_19 : P A = 3 / 19 :=
by sorry

end prob_A_is_3_over_19_l736_736020


namespace max_perimeter_triangle_l736_736546

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l736_736546


namespace find_area_of_overlapping_region_l736_736766

noncomputable def overlapping_region_area 
  (r R d : ℝ) : ℝ :=
  r^2 * real.arccos ((d^2 + r^2 - R^2) / (2 * d * r)) + 
  R^2 * real.arccos ((d^2 + R^2 - r^2) / (2 * d * R)) - 
  (1/2) * real.sqrt ((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))

theorem find_area_of_overlapping_region 
  (r R : ℝ) (h1 : r = 2) (h2 : R = 2) (d : ℝ) (h3 : d = 3) : 
  overlapping_region_area r R d ≈ (8 * real.pi / 3) - 3.968625 :=
by
  simp [overlapping_region_area, h1, h2, h3]
  sorry

end find_area_of_overlapping_region_l736_736766


namespace pyramid_cross_section_area_l736_736015

noncomputable def area_cross_section (a : ℝ) (h : ℝ) : ℝ :=
  (11 * real.sqrt 3) / 10

theorem pyramid_cross_section_area :
  ∀ a h : ℝ,
    a = 3 →
    h = real.sqrt 3 →
    area_cross_section a h = (11 * real.sqrt 3) / 10 :=
by
  intros a h ha hh
  rw [ha, hh]
  simp [area_cross_section]
  sorry

end pyramid_cross_section_area_l736_736015


namespace least_sum_of_exponents_l736_736316

theorem least_sum_of_exponents (a b c : ℕ) (ha : 2^a ∣ 520) (hb : 2^b ∣ 520) (hc : 2^c ∣ 520) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c = 12 :=
by
  sorry

end least_sum_of_exponents_l736_736316


namespace smallest_6_digit_div_by_111_l736_736923

theorem smallest_6_digit_div_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 := by
  sorry

end smallest_6_digit_div_by_111_l736_736923


namespace distinguish_truth_teller_and_liar_l736_736543

def Inhabitant := ℕ  -- Assuming arbitrary inhabitant IDs

inductive Behavior
| truth_teller 
| liar

def behavior_of (inhabitant : Inhabitant) : Behavior := 
sorry -- This would be some function or predicate defining the behavior of the inhabitant

-- The main statement
theorem distinguish_truth_teller_and_liar (inhabitant : Inhabitant) :
  (behavior_of inhabitant = Behavior.truth_teller ∧ 
   (inhabitant.surely_true := "If you were always telling the truth, how would you answer 'Are you a liar'?" = "No")) ∨ 
  (behavior_of inhabitant = Behavior.liar ∧ 
   (inhabitant.surely_false := "If you were always telling the truth, how would you answer 'Are you a liar'?" = "Yes")) :=
sorry

end distinguish_truth_teller_and_liar_l736_736543


namespace problem_27_integer_greater_than_B_over_pi_l736_736500

noncomputable def B : ℕ := 22

theorem problem_27_integer_greater_than_B_over_pi :
  Nat.ceil (B / Real.pi) = 8 := sorry

end problem_27_integer_greater_than_B_over_pi_l736_736500


namespace total_marbles_l736_736518

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l736_736518


namespace geometric_sequence_sum_q_value_l736_736347

theorem geometric_sequence_sum_q_value (q : ℝ) (a S : ℕ → ℝ) :
  a 1 = 4 →
  (∀ n, a (n+1) = a n * q ) →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, (S n + 2) = (S 1 + 2) * (q ^ (n - 1))) →
  q = 3
:= 
by
  sorry

end geometric_sequence_sum_q_value_l736_736347


namespace min_straight_line_cuts_l736_736682

theorem min_straight_line_cuts (can_overlap : Prop) : 
  ∃ (cuts : ℕ), cuts = 4 ∧ 
  (∀ (square : ℕ), square = 3 →
   ∀ (unit : ℕ), unit = 1 → 
   ∀ (divided : Prop), divided = True → 
   (unit * unit) * 9 = (square * square)) :=
by
  sorry

end min_straight_line_cuts_l736_736682


namespace classify_quadrilateral_as_kite_l736_736997

variable {V : Type} [InnerProductSpace ℝ V]

structure Quadrilateral (A B C D : V) :=
(diag_perpendicular : ∀ (O : V),
  (O = (1/2 • (A + C)) ∧ O = (1/2 • (B + D)) →
    ⟪A - C, B - D⟫ = 0)
(diag_length_ratio : ∃ (p : ℝ), p = 2 ∧ (∥A - C∥ = p * ∥B - D∥))

def is_kite {A B C D : V} (quad : Quadrilateral A B C D) : Prop :=
  (∥A - B∥ = ∥A - D∥ ∧ ∥C - B∥ = ∥C - D∥) ∨ (∥A - B∥ = ∥C - D∥ ∧ ∥A - D∥ = ∥C - B∥)

theorem classify_quadrilateral_as_kite
  {A B C D : V}
  (quad : Quadrilateral A B C D) :
  is_kite quad :=
by
  sorry

end classify_quadrilateral_as_kite_l736_736997


namespace z_conjugate_sum_l736_736699

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736699


namespace car_highway_mileage_l736_736130

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage_l736_736130


namespace sum_conjugate_eq_two_l736_736735

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736735


namespace radius_minimizes_perimeter_l736_736418

noncomputable def minimize_perimeter_radius (T : ℝ) (K : ℝ → ℝ) : ℝ :=
classical.some (exists_minimizer K (λ r, 0 < r))

theorem radius_minimizes_perimeter : minimize_perimeter_radius 100 (λ r, 2 * r + 200 / r) = 10 := 
by
  sorry

end radius_minimizes_perimeter_l736_736418


namespace subset_A_implies_a_eq_2_intersect_A_B_implies_a_eq_2_or_4_l736_736669

variable (A : Set ℝ) (B : Set ℝ) (a : ℝ)
variable hA : A = {-2, 3, 4, 6}
variable hB : B = {3, a, a^2}

theorem subset_A_implies_a_eq_2 :
  B ⊆ A → a = 2 :=
by
  intro h
  sorry

theorem intersect_A_B_implies_a_eq_2_or_4 :
  A ∩ B = {3, 4} → (a = 2 ∨ a = 4) :=
by
  intro h
  sorry

end subset_A_implies_a_eq_2_intersect_A_B_implies_a_eq_2_or_4_l736_736669


namespace factory_a_min_hours_l736_736511

theorem factory_a_min_hours (x : ℕ) :
  (550 * x + (700 - 55 * x) / 45 * 495 ≤ 7260) → (8 ≤ x) :=
by
  sorry

end factory_a_min_hours_l736_736511


namespace quadratic_function_properties_l736_736285

theorem quadratic_function_properties :
  ∃ a : ℝ, ∃ f : ℝ → ℝ,
    (∀ x : ℝ, f x = a * (x + 1) ^ 2 - 2) ∧
    (f 1 = 10) ∧
    (f (-1) = -2) ∧
    (∀ x : ℝ, x > -1 → f x ≥ f (-1))
:=
by
  sorry

end quadratic_function_properties_l736_736285


namespace determine_a_l736_736615

open Complex

noncomputable def complex_eq_real_im_part (a : ℝ) : Prop :=
  let z := (a - I) * (1 + I) / I
  (z.re, z.im) = ((a - 1 : ℝ), -(a + 1 : ℝ))

theorem determine_a (a : ℝ) (h : complex_eq_real_im_part a) : a = -1 :=
sorry

end determine_a_l736_736615


namespace min_visible_sum_of_cube_l736_736942

theorem min_visible_sum_of_cube
  (all_visible_faces_sum : ℕ)
  (num_dice : ℕ)
  (per_die_minimum_sum : ℕ)
  (face_values : ∀ (d : ℕ), d ∈ {1, 2, 3, 4, 5, 6} → d ≤ 6)
  (opposite_pairs : (ℕ × ℕ) → (ℕ × ℕ))
  (opposite_pairs_sums : ∀ (d1 d2 : ℕ), opposite_pairs (d1, d2) = (d1, d2) → d1 + d2 = 7)
  : all_visible_faces_sum = 48 := 
by
  -- This part can utilize the given conditions and challenge to show that the sum of visible faces is 48
  sorry

end min_visible_sum_of_cube_l736_736942


namespace A_walking_speed_l736_736551

theorem A_walking_speed
    (v : ℝ)
    (B_catch_up_distance : ℝ = 80)
    (A_start_advanced_time : ℝ = 4)
    (B_speed : ℝ = 20)
    (A_walking_distance : ℝ := 80)
    (A_walk_advanced_distance : ℝ := v * A_start_advanced_time):
    (20 * ((80 - 4 * v) / v) = 80) → (v = 10) :=
by
  intros h
  sorry

end A_walking_speed_l736_736551


namespace even_combinations_after_six_operations_l736_736159

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736159


namespace hyperbola_eccentricity_l736_736018

theorem hyperbola_eccentricity :
  (eccentricity_of_hyperbola (16 : ℝ) (9 : ℝ) = 5 / 4) := 
by
  -- Definition of the hyperbola equation coefficients
  def a_squared := 16
  def b_squared := 9
  def a := Real.sqrt a_squared
  def b := Real.sqrt b_squared
  
  -- Calculate c according to hyperbola properties
  def c := Real.sqrt (a_squared + b_squared)
  
  -- Define the eccentricity
  def eccentricity := c / a
  
  -- Prove the eccentricity is 5/4
  have h : eccentricity = 5 / 4,
  {
    unfold a_squared b_squared a b c eccentricity,
    exact sorry, -- proof omitted
  }
  exact h

end hyperbola_eccentricity_l736_736018


namespace number_of_lattice_points_in_intersection_l736_736471

def sphere1 (x y z : ℝ) : Prop :=
  x^2 + y^2 + (z - 21/2)^2 ≤ 6^2

def sphere2 (x y z : ℝ) : Prop :=
  x^2 + y^2 + (z - 1)^2 ≤ (9/2)^2

def is_lattice_point (x y z : ℤ) : Prop :=
  sphere1 x y z ∧ sphere2 x y z

theorem number_of_lattice_points_in_intersection : 
  (finset.card (finset.filter (λ p, is_lattice_point p.1 p.2 p.3) 
  (finset.fin_range (2 * int.of_nat 6 + 1).val * finset.fin_range (2 * int.of_nat 6 + 1).val * finset.fin_range (2 * int.of_nat 11 + 1).val))) = 9 := 
sorry

end number_of_lattice_points_in_intersection_l736_736471


namespace red_pigment_contribution_l736_736951

theorem red_pigment_contribution :
  ∀ (G : ℝ), (2 * G + G + 3 * G = 24) →
  (0.6 * (2 * G) + 0.5 * (3 * G) = 10.8) :=
by
  intro G
  intro h1
  sorry

end red_pigment_contribution_l736_736951


namespace total_marbles_l736_736515

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end total_marbles_l736_736515


namespace smallest_number_with_12_divisors_l736_736486

theorem smallest_number_with_12_divisors : ∃ n : ℕ, (n ≥ 1) ∧ (∀ d : ℕ, (d > 0 ∧ d | n → gcd d n = d) → divisors n = 12) ∧ (∀ m : ℕ, (m ≥ 1) ∧ (∀ d : ℕ, (d > 0 ∧ d | m → gcd d m = d) → divisors m = 12) → m ≥ n) := 
sorry

end smallest_number_with_12_divisors_l736_736486


namespace distance_from_Q_to_DE_l736_736889

-- Define the problem setup
variables (Q : Point) (D E F : Point) (h : ℝ)

-- Define the conditions
def conditions :=
  inside_triangle Q D E F ∧
  parallel_to_base_through_Q Q D E F ∧ 
  divides_triangle_area Q D E F (1/3) ∧
  altitude_length D E F 3

-- Define the statement to prove
theorem distance_from_Q_to_DE :
  conditions Q D E F h → h = 1 :=
begin
  sorry
end

end distance_from_Q_to_DE_l736_736889


namespace cello_viola_pairs_are_70_l736_736952

-- Given conditions
def cellos : ℕ := 800
def violas : ℕ := 600
def pair_probability : ℝ := 0.00014583333333333335

-- Theorem statement translating the mathematical problem
theorem cello_viola_pairs_are_70 (n : ℕ) (h1 : cellos = 800) (h2 : violas = 600) (h3 : pair_probability = 0.00014583333333333335) :
  n = 70 :=
sorry

end cello_viola_pairs_are_70_l736_736952


namespace remainder_of_3_pow_800_mod_17_l736_736102

theorem remainder_of_3_pow_800_mod_17 : (3^800) % 17 = 1 := by
  sorry

end remainder_of_3_pow_800_mod_17_l736_736102


namespace angle_between_adjacent_lateral_faces_l736_736853

theorem angle_between_adjacent_lateral_faces (m n : ℝ) (h_mn_positive : 0 < m ∧ 0 < n) :
  ∃ α : ℝ, α = π - real.arccos (n^2 / m^2) :=
by
  sorry

end angle_between_adjacent_lateral_faces_l736_736853


namespace samuel_time_l736_736846

/-- Sarah's time in hours -/
def sarah_time_hours : ℝ := 1.3

/-- Conversion from hours to minutes -/
def sarah_time_minutes : ℝ := sarah_time_hours * 60

/-- Time difference Samuel took faster than Sarah in minutes -/
def time_difference : ℝ := 48

/-- Proof that Samuel's time is 30 minutes -/
theorem samuel_time (sarah_time : ℝ) (time_diff : ℝ) (h_sarah : sarah_time = 78) (h_diff : time_diff = 48) :
  sarah_time - time_diff = 30 :=
by {
  rw [h_sarah, h_diff],
  norm_num,
  sorry
}

end samuel_time_l736_736846


namespace opposite_of_neg3_l736_736431

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l736_736431


namespace mangoes_purchased_l736_736201

theorem mangoes_purchased
    (kg_grapes : ℕ) (rate_grapes : ℕ) (total_paid : ℕ) (rate_mangoes : ℕ)
    (h1 : kg_grapes = 7)
    (h2 : rate_grapes = 68)
    (h3 : total_paid = 908)
    (h4 : rate_mangoes = 48) :
    ∃ kg_mangoes : ℕ, kg_mangoes = 9 :=
by
  -- Definitions derived from conditions
  let cost_grapes := kg_grapes * rate_grapes
  have h_cost_grapes : cost_grapes = 476, from calc
    kg_grapes * rate_grapes = 7 * 68 : by rw [h1, h2]
          ... = 476       : by norm_num,
  
  have amount_mangoes := total_paid - cost_grapes
  have h_amount_mangoes : amount_mangoes = 432, from calc
    total_paid - cost_grapes = 908 - 476 : by rw [h3, h_cost_grapes]
                     ... = 432  : by norm_num,

  let kg_mangoes := amount_mangoes / rate_mangoes
  have h_kg_mangoes : kg_mangoes = 432 / 48, from rfl
  use 9
  exact calc
    432 / 48 = 9 : by norm_num

end mangoes_purchased_l736_736201


namespace walnut_trees_l736_736784

theorem walnut_trees (logs_per_pine logs_per_maple logs_per_walnut pine_trees maple_trees total_logs walnut_trees : ℕ)
  (h1 : logs_per_pine = 80)
  (h2 : logs_per_maple = 60)
  (h3 : logs_per_walnut = 100)
  (h4 : pine_trees = 8)
  (h5 : maple_trees = 3)
  (h6 : total_logs = 1220)
  (h7 : total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut) :
  walnut_trees = 4 :=
by
  sorry

end walnut_trees_l736_736784


namespace soja_finished_fraction_l736_736848

def pages_finished (x pages_left total_pages : ℕ) : Prop :=
  x - pages_left = 100 ∧ x + pages_left = total_pages

noncomputable def fraction_finished (x total_pages : ℕ) : ℚ :=
  x / total_pages

theorem soja_finished_fraction (x : ℕ) (h1 : pages_finished x (x - 100) 300) :
  fraction_finished x 300 = 2 / 3 :=
by
  sorry

end soja_finished_fraction_l736_736848


namespace correct_dispersion_statements_l736_736557

def statement1 (make_use_of_data : Prop) : Prop :=
make_use_of_data = true

def statement2 (multi_numerical_values : Prop) : Prop :=
multi_numerical_values = true

def statement3 (dispersion_large_value_small : Prop) : Prop :=
dispersion_large_value_small = false

theorem correct_dispersion_statements
  (make_use_of_data : Prop)
  (multi_numerical_values : Prop)
  (dispersion_large_value_small : Prop)
  (h1 : statement1 make_use_of_data)
  (h2 : statement2 multi_numerical_values)
  (h3 : statement3 dispersion_large_value_small) :
  (make_use_of_data ∧ multi_numerical_values ∧ ¬ dispersion_large_value_small) = true :=
by
  sorry

end correct_dispersion_statements_l736_736557


namespace solve_for_x_l736_736910

theorem solve_for_x (x : ℚ) : 
  x + 5 / 6 = 11 / 18 - 2 / 9 → x = -4 / 9 := 
by
  intro h
  sorry

end solve_for_x_l736_736910


namespace cos_two_pi_over_three_eq_neg_one_half_l736_736124

theorem cos_two_pi_over_three_eq_neg_one_half :
  ∀ (x : ℝ), (cos (Real.pi - x) = - cos x) → (cos (Real.pi / 3) = 1 / 2) → cos (2 * Real.pi / 3) = - (1 / 2) :=
by
  intros x h_cos_pi_minus_x h_cos_pi_over_3
  sorry

end cos_two_pi_over_three_eq_neg_one_half_l736_736124


namespace z_conjugate_sum_l736_736696

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736696


namespace total_number_of_people_l736_736204

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people_l736_736204


namespace length_of_other_parallel_side_l736_736598

theorem length_of_other_parallel_side (a b h area : ℝ) 
  (h_area : area = 190) 
  (h_parallel1 : b = 18) 
  (h_height : h = 10) : 
  a = 20 :=
by
  sorry

end length_of_other_parallel_side_l736_736598


namespace checkerboard_sum_eq_2660_l736_736759

open Nat

theorem checkerboard_sum_eq_2660 (n : ℕ) (h : ∑ i in range n, ∑ j in range n, |i - j| = 2660) : n = 20 := by
  sorry

end checkerboard_sum_eq_2660_l736_736759


namespace number_of_pines_l736_736057

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l736_736057


namespace part_a_part_b_l736_736831

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l736_736831


namespace determine_k_l736_736658

noncomputable def f (x k : ℝ) : ℝ := -4 * x^3 + k * x

theorem determine_k : ∀ k : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x k ≤ 1) → k = 3 :=
by
  sorry

end determine_k_l736_736658


namespace pictures_deleted_l736_736107

theorem pictures_deleted (p_zoo p_museum p_left p_deleted : ℕ) 
  (H1 : p_zoo = 49)
  (H2 : p_museum = 8)
  (H3 : p_left = 19)
  (H4 : p_deleted = 57 - 19) : 
  p_deleted = 38 := by
  -- from the conditions
  rw [H1, H2, H3, H4]
  -- the expected result
  sorry

end pictures_deleted_l736_736107


namespace chessboard_grains_difference_l736_736959

theorem chessboard_grains_difference : 
  let grains_on_square : ℕ → ℕ := λ k, 3^k
  let first_ten_sum := ∑ k in finset.range 10, grains_on_square (k+1)
  let grains_on_12th := grains_on_square 12
  in grains_on_12th - first_ten_sum = 442869 :=
by
  let grains_on_square : ℕ → ℕ := λ k, 3^k
  let first_ten_sum := ∑ k in finset.range 10, grains_on_square (k+1)
  let grains_on_12th := grains_on_square 12
  sorry

end chessboard_grains_difference_l736_736959


namespace result_prob_a_l736_736833

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l736_736833


namespace jude_total_matchbox_vehicles_l736_736786

theorem jude_total_matchbox_vehicles :
  ∀ (price_car price_truck price_helicopter : ℕ)
    (initial_bottle_caps : ℕ)
    (trucks_bought : ℕ)
    (percent_remaining_on_cars : ℕ),
  price_car = 10 →
  price_truck = 15 →
  price_helicopter = 20 →
  initial_bottle_caps = 250 →
  trucks_bought = 5 →
  percent_remaining_on_cars = 60 →
  let bottle_caps_spent_on_trucks := trucks_bought * price_truck in
  let remaining_bottle_caps_after_trucks := initial_bottle_caps - bottle_caps_spent_on_trucks in
  let bottle_caps_spent_on_cars := percent_remaining_on_cars * remaining_bottle_caps_after_trucks / 100 in
  let cars_bought := bottle_caps_spent_on_cars / price_car in
  let remaining_bottle_caps_after_cars := remaining_bottle_caps_after_trucks - bottle_caps_spent_on_cars in
  let helicopters_bought := remaining_bottle_caps_after_cars / price_helicopter in
  trucks_bought + cars_bought + helicopters_bought = 18 :=
begin
  intros,
  sorry
end

end jude_total_matchbox_vehicles_l736_736786


namespace no_solution_frac_eq_l736_736750

theorem no_solution_frac_eq (k : ℝ) : (∀ x : ℝ, ¬(1 / (x + 1) = 3 * k / x)) ↔ (k = 0 ∨ k = 1 / 3) :=
by
  sorry

end no_solution_frac_eq_l736_736750


namespace find_diameter_l736_736599

noncomputable def cost_per_meter : ℝ := 2
noncomputable def total_cost : ℝ := 188.49555921538757
noncomputable def circumference (c : ℝ) (p : ℝ) : ℝ := c / p
noncomputable def diameter (c : ℝ) : ℝ := c / Real.pi

theorem find_diameter :
  diameter (circumference total_cost cost_per_meter) = 30 := by
  sorry

end find_diameter_l736_736599


namespace min_value_a_sum_l736_736414

noncomputable def p (a : ℕ) : ℚ :=
  (Nat.choose (32 - a) 2 + Nat.choose (a - 1) 2 : ℚ) / 703

theorem min_value_a_sum :
  ∃ (a : ℕ), p(a) ≥ 1/2 ∧ (let m := Nat.gcd (Numerator (p 33)) (Denominator (p 33)),
         Num := Numerator (p 33) / m, 
         Den := Denominator (p 33) / m in Num + Den = 1200) :=
begin
  sorry
end

end min_value_a_sum_l736_736414


namespace find_other_sides_of_triangle_l736_736623

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l736_736623


namespace columns_with_more_zero_rows_l736_736224

theorem columns_with_more_zero_rows
  (table : Fin 1001 → Fin 1001 → Bool)
  (h : ∀ j : Fin 1001, (∑ i, if table i j = false then 1 else 0) > (∑ i, if table i j = true then 1 else 0)) :
  ∃ (j1 j2 : Fin 1001), (∑ i, if table i j1 = false ∧ table i j2 = false then 1 else 0) > (∑ i, if table i j1 = true ∧ table i j2 = true then 1 else 0) := 
begin
  sorry
end

end columns_with_more_zero_rows_l736_736224


namespace auston_height_l736_736989

noncomputable def auston_height_in_meters (height_in_inches : ℝ) : ℝ :=
  let height_in_cm := height_in_inches * 2.54
  height_in_cm / 100

theorem auston_height : auston_height_in_meters 65 = 1.65 :=
by
  sorry

end auston_height_l736_736989


namespace num_solutions_l736_736263

def floor_sqrt_half (n : ℕ) : ℤ :=
  ⌊n + √n + 1 / 2⌋₊

theorem num_solutions (k : ℤ) (hk : k ≥ 1) :
  ∃ (s : finset ℕ), (∀ n ∈ s, floor_sqrt_half (floor_sqrt_half n) - floor_sqrt_half n = k) ∧ s.card = 2 * k - 1 :=
sorry

end num_solutions_l736_736263


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736818

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736818


namespace students_more_than_rabbits_l736_736202

theorem students_more_than_rabbits (students_per_classroom rabbits_per_classroom absent_rabbits count_classrooms : ℕ)
    (number_students := students_per_classroom * count_classrooms)
    (number_rabbits := rabbits_per_classroom * count_classrooms - absent_rabbits) :
    students_per_classroom = 24 →
    rabbits_per_classroom = 3 →
    absent_rabbits = -1 →
    count_classrooms = 5 →
    number_students - (number_rabbits + 1) = 105 :=
by
  -- Proof omitted
  sorry

end students_more_than_rabbits_l736_736202


namespace opposite_of_neg_three_l736_736436

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l736_736436


namespace min_value_one_over_a_plus_two_over_b_l736_736077

theorem min_value_one_over_a_plus_two_over_b :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 2) →
  ∃ (min_val : ℝ), min_val = (1 / a + 2 / b) ∧ min_val = 9 / 2 :=
by
  sorry

end min_value_one_over_a_plus_two_over_b_l736_736077


namespace remainder_when_b_div_11_l736_736790

theorem remainder_when_b_div_11 (n : ℕ) (h : 0 < n) :
  let b := ((5 ^ (2 * n)) + 7)⁻¹ % 11
  in b = 10 ∨ b = 9 ∨ b = 1 ∨ b = 7 :=
sorry

end remainder_when_b_div_11_l736_736790


namespace exists_root_in_interval_l736_736050

def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_root_in_interval : ∃ x ∈ Ioo 0 1, f x = 0 :=
sorry

end exists_root_in_interval_l736_736050


namespace proof_problem_l736_736456

variables {n : ℕ} {a : ℕ → ℤ} {S : ℕ → ℤ} {d a1 : ℤ}

-- Assume d is the common difference of the arithmetic sequence
-- Assume Sn is the sum of the first n terms of the sequence
axiom Sn_def : ∀ k, S k = k (2 * a 1 + (k - 1) * d) / 2

-- Assume the given conditions
axiom h1 : d ≠ 0
axiom h2 : (a 1 + d) < (a 1 + 2 * d)
axiom h3 : ((a 1 + d) * (a 1 + 5 * d)) = (a 1 + 2 * d) ^ 2

-- Prove the required conclusion
theorem proof_problem :
  (a 1) * d < 0 ∧ d * (S 3) > 0 :=
sorry

end proof_problem_l736_736456


namespace rowers_voting_for_coaches_l736_736758

theorem rowers_voting_for_coaches :
  ∀ (coaches rowers votes_per_coach: ℕ),
  coaches = 36 →
  votes_per_coach = 5 →
  rowers = 60 →
  (coaches * votes_per_coach / rowers) = 3 :=
by
  intros coaches rowers votes_per_coach h1 h2 h3
  rw [h1, h2, h3]
  sorry

end rowers_voting_for_coaches_l736_736758


namespace average_weight_a_b_l736_736014

theorem average_weight_a_b (A B C : ℝ) 
    (h1 : (A + B + C) / 3 = 45) 
    (h2 : (B + C) / 2 = 44) 
    (h3 : B = 33) : 
    (A + B) / 2 = 40 := 
by 
  sorry

end average_weight_a_b_l736_736014


namespace complex_multiplication_l736_736564

theorem complex_multiplication : (1 + complex.i) * (2 - complex.i) = 3 + complex.i := 
by
  sorry

end complex_multiplication_l736_736564


namespace largest_expression_l736_736371

-- Define the value y
def y : ℝ := 10 ^ (-2024)

-- Define the expressions
def expr_A : ℝ := 5 + y
def expr_B : ℝ := 5 - y
def expr_C : ℝ := 5 * y
def expr_D : ℝ := 5 / y
def expr_E : ℝ := y / 5

-- State the proof problem
theorem largest_expression : expr_D = 5 * 10 ^ 2024 :=
by {
  sorry -- Proof omitted as per instructions
}

end largest_expression_l736_736371


namespace multiplication_value_l736_736912

theorem multiplication_value (x : ℝ) (h : (2.25 / 3) * x = 9) : x = 12 :=
by
  sorry

end multiplication_value_l736_736912


namespace length_of_BD_l736_736878

theorem length_of_BD (a b c d : ℝ) (h₁ : {a, b, c, d, 42, 8, 14, 19, 28, 37} = {8, 14, 19, 28, 37, 42})
  (h₂ : a ≠ b) (h₃ : a ≠ c) (h₄ : a ≠ d) (h₅ : b ≠ c) (h₆ : b ≠ d) (h₇ : c ≠ d) (h_ac : c = 42) : 
  ∃ (bd : ℝ), bd = 28 :=
by
  use 28
  sorry

end length_of_BD_l736_736878


namespace Tim_took_out_11_rulers_l736_736076

-- Define the initial number of rulers
def initial_rulers := 14

-- Define the number of rulers left in the drawer
def rulers_left := 3

-- Define the number of rulers taken by Tim
def rulers_taken := initial_rulers - rulers_left

-- Statement to prove that the number of rulers taken by Tim is indeed 11
theorem Tim_took_out_11_rulers : rulers_taken = 11 := by
  sorry

end Tim_took_out_11_rulers_l736_736076


namespace convert_quadratic_form_l736_736576

def quadratic_function : ℝ → ℝ := λ x, x^2 - 4*x + 5

theorem convert_quadratic_form :
  quadratic_function = λ x, (x - 2)^2 + 1 :=
by sorry

end convert_quadratic_form_l736_736576


namespace binom_1450_2_eq_1050205_l736_736208

def binom_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_1450_2_eq_1050205 : binom_coefficient 1450 2 = 1050205 :=
by {
  sorry
}

end binom_1450_2_eq_1050205_l736_736208


namespace climbing_difference_l736_736354

theorem climbing_difference (rate_matt rate_jason time : ℕ) (h_rate_matt : rate_matt = 6) (h_rate_jason : rate_jason = 12) (h_time : time = 7) : 
  rate_jason * time - rate_matt * time = 42 :=
by
  sorry

end climbing_difference_l736_736354


namespace retailer_actual_profit_percentage_l736_736116

theorem retailer_actual_profit_percentage (cost_price : ℝ)
    (marked_up : cost_price * 1.40)
    (discounted : marked_up * 0.75):
    ((discounted - cost_price) / cost_price) * 100 = 5 := by
  sorry

end retailer_actual_profit_percentage_l736_736116


namespace fraction_equiv_l736_736479

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l736_736479


namespace angle_EMK_90_l736_736536

theorem angle_EMK_90 (A B C K N M E O : Point) 
  (h_triangle : is_right_triangle A B C)
  (h_angle_C : angle C = 90)
  (h_inscribed : is_inscribed_circle A B C O)
  (h_midpoint_arc : is_midpoint_of_arc K B C A O)
  (h_midpoint_AC : is_midpoint N A C)
  (h_intersection : M = intersection_point_of_ray_on_circle K N O (circle A B C O))
  (h_tangents_meet : meet_at E (tangent_circle_point A O) (tangent_circle_point C O)) :
  angle_bisector E M K = 90 := 
sorry

end angle_EMK_90_l736_736536


namespace average_score_calculation_l736_736415

theorem average_score_calculation :
  let total_students := 8 + 11 + 10 + 16 + 3 + 2
  let total_score := 90 * 8 + 83 * 11 + 74 * 10 + 65 * 16 + 56 * 3 + 49 * 2
  (total_score / total_students : ℝ) = 73.6 :=
by
  let total_students := 8 + 11 + 10 + 16 + 3 + 2
  let total_score := 90 * 8 + 83 * 11 + 74 * 10 + 65 * 16 + 56 * 3 + 49 * 2
  have h : (total_score / total_students : ℝ) = (5522 / 50 : ℝ) := by sorry
  have hd : (5522 / 50 : ℝ) = 73.6 := by sorry
  exact eq.trans h hd

end average_score_calculation_l736_736415


namespace total_nephews_proof_l736_736983

-- We declare the current number of nephews as unknown variables
variable (Alden_current Vihaan Shruti Nikhil : ℕ)

-- State the conditions as hypotheses
theorem total_nephews_proof
  (h1 : 70 = (1 / 3 : ℚ) * Alden_current)
  (h2 : Vihaan = Alden_current + 120)
  (h3 : Shruti = 2 * Vihaan)
  (h4 : Nikhil = Alden_current + Shruti - 40) :
  Alden_current + Vihaan + Shruti + Nikhil = 2030 := 
by
  sorry

end total_nephews_proof_l736_736983


namespace part1_monotonicity_part2_integer_k_l736_736298

noncomputable def f (x : ℝ) : ℝ := exp x - 1

def g (x : ℝ) (a : ℝ) : ℝ := f x - a * x

theorem part1_monotonicity (a : ℝ) : 
  (∀ x : ℝ, (if a ≤ 0 then (exp x - a) > 0 else 
  (if x < real.log a then (exp x - a) < 0 else (exp x - a) > 0))) := sorry

theorem part2_integer_k (k : ℤ) : 
  (∀ x : ℝ, x > 0 → (x - k - 1) * f x + x + 1 > 0) → k ≤ 1 := sorry

end part1_monotonicity_part2_integer_k_l736_736298


namespace log_base_27_3_l736_736240

-- Define the condition
lemma log_base_condition : 27 = 3^3 := rfl

-- State the theorem to be proven
theorem log_base_27_3 : log 27 3 = 1 / 3 :=
by 
  -- skip the proof for now
  sorry

end log_base_27_3_l736_736240


namespace geometric_problem_tangent_geometric_problem_intersect_geometric_problem_disjoint_l736_736929

variables {O : Type*} {m l : Set O} {A B C M P Q R : O}
variables {r x : ℝ}
variables {AM BM CM AB BC AC AP BQ CR : ℝ}
variables (t : ℝ)
variables [is_horizontal : horizontal m] [is_center : center O]
variables [is_perpendicular : perpendicular l m]
variables [line_a : line A M l] [line_b : line B M l] [line_c : line C M l]
variables [outside_a : outside_circle A O] [outside_b : outside_circle B O] [outside_c : outside_circle C O]
variables [tangent_ap : tangent l A P O] [tangent_bq : tangent l B Q O] [tangent_cr : tangent l C R O]
variables (a b c : ℝ)
variables (a_gt_b : a > b) (b_gt_c : b > c) (c_gt_0 : c > 0)
variables (radius_r : r = dist O P) (om_equals_x : x = dist O M)
variables (am_equals_a : a = dist A M) (bm_equals_b : b = dist B M) (cm_equals_c : c = dist C M)

theorem geometric_problem_tangent : (AB * CR + BC * AP = AC * BQ) :=
sorry

theorem geometric_problem_intersect (hx : 0 < x) (hx_lt_r : x < r) : (AB * CR + BC * AP < AC * BQ) :=
sorry

theorem geometric_problem_disjoint (hx : x > r) : (AB * CR + BC * AP > AC * BQ) :=
sorry

end geometric_problem_tangent_geometric_problem_intersect_geometric_problem_disjoint_l736_736929


namespace num_green_balls_l736_736944

theorem num_green_balls (G : ℕ) (h : (3 * 2 : ℚ) / ((5 + G) * (4 + G)) = 1/12) : G = 4 :=
by
  sorry

end num_green_balls_l736_736944


namespace n_th_equation_l736_736384

theorem n_th_equation (n : ℕ) : 
  (Finset.range n).sum (λ k, (-1:ℤ)^k * (k+1)^2) = ((-1:ℤ)^(n+1) * n * (n + 1) / 2) := 
sorry

end n_th_equation_l736_736384


namespace roots_disk_coverage_l736_736367

theorem roots_disk_coverage (P : Polynomial ℝ) (R : ℝ) (h : ∀ z, P.eval z = 0 → Complex.abs z ≤ R) :
  ∀ k : ℝ, ∀ z, (n : ℕ) := P.natDegree, Q : Polynomial ℝ := n • P - k • P.derivative, 
  roots Q z → Complex.abs z ≤ R + Complex.abs k :=
sorry

end roots_disk_coverage_l736_736367


namespace sum_conjugate_eq_two_l736_736739

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736739


namespace area_of_picture_l736_736182

theorem area_of_picture
  (paper_width : ℝ)
  (paper_height : ℝ)
  (left_margin : ℝ)
  (right_margin : ℝ)
  (top_margin_cm : ℝ)
  (bottom_margin_cm : ℝ)
  (cm_per_inch : ℝ)
  (converted_top_margin : ℝ := top_margin_cm * (1 / cm_per_inch))
  (converted_bottom_margin : ℝ := bottom_margin_cm * (1 / cm_per_inch))
  (picture_width : ℝ := paper_width - left_margin - right_margin)
  (picture_height : ℝ := paper_height - converted_top_margin - converted_bottom_margin)
  (area : ℝ := picture_width * picture_height)
  (h1 : paper_width = 8.5)
  (h2 : paper_height = 10)
  (h3 : left_margin = 1.5)
  (h4 : right_margin = 1.5)
  (h5 : top_margin_cm = 2)
  (h6 : bottom_margin_cm = 2.5)
  (h7 : cm_per_inch = 2.54)
  : area = 45.255925 :=
by sorry

end area_of_picture_l736_736182


namespace flowers_per_basket_l736_736217

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l736_736217


namespace min_A_plus_B_cardinality_l736_736282

-- Definitions of sets A and B with the given properties.
noncomputable 
def A : Set ℕ := sorry -- Assume we have a set A satisfying the conditions
def B : Set ℕ := sorry -- Assume we have a set B with |B| = 16 and contains positive integers

-- Conditions from the problem statement
def condition1 : (∀ a b m n ∈ A, a + b = m + n → ({a, b} = {m, n})) := sorry
def condition2 : (A.finite ∧ A.card = 20) := sorry
def condition3 : (B.finite ∧ B.card = 16) := sorry

-- The set A + B
def A_plus_B : Set ℕ := {ab | ∃ (a ∈ A) (b ∈ B), ab = a + b}

-- The statement to prove
theorem min_A_plus_B_cardinality : |A_plus_B| = 200 :=
  sorry

end min_A_plus_B_cardinality_l736_736282


namespace pdf_cos_transformation_l736_736184

noncomputable def uniform_pdf (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 * Real.pi then 1 / (2 * Real.pi) else 0

def transformed_pdf (y : ℝ) : ℝ :=
  if -1 < y ∧ y < 1 then 1 / (Real.pi * Real.sqrt (1 - y^2)) else 0

theorem pdf_cos_transformation : 
  ∀ y : ℝ, (∃ g_Y : ℝ → ℝ, (∀ y, g_Y y = transformed_pdf y)) := 
  by
  sorry

end pdf_cos_transformation_l736_736184


namespace marks_social_studies_val_l736_736402

variable (marks_math marks_science marks_english marks_biology marks_social total_avg_marks : ℕ) 

-- Given conditions from the problem
axiom marks_math_val : marks_math = 76
axiom marks_science_val : marks_science = 65
axiom marks_english_val : marks_english = 67
axiom marks_biology_val : marks_biology = 95
axiom total_avg_marks_val : total_avg_marks = 77

-- Define the total marks in all subjects
def total_marks_all : ℕ := total_avg_marks * 5

-- Define the total marks in known subjects
def total_marks_known : ℕ := marks_math + marks_science + marks_english + marks_biology

-- Prove Shekar's marks in social studies
theorem marks_social_studies_val : marks_social = total_marks_all - total_marks_known := 
by  simp [marks_math_val, marks_science_val, marks_english_val, marks_biology_val, total_avg_marks_val] ; sorry

end marks_social_studies_val_l736_736402


namespace ellipse_standard_equation_find_m_l736_736290

theorem ellipse_standard_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
(h3 : (sqrt 3 / 2) = ecc) (h4 : (0 : ℝ, 1 : ℝ) ∈ set_of (λ (p : ℝ × ℝ),
  (p.1^2 / a^2) + (p.2^2 / b^2) = 1)) (ecc : ℝ) : 
  (a = 2) ∧ (b = 1) → (∀ x y : ℝ, (x^2 / 4) + y^2 = 1) :=
sorry

theorem find_m (m : ℝ) (h1 : m > 0) : 
  let l := set_of (λ (p : ℝ × ℝ), p.2 = (k : ℝ) * p.1 + m) in
  ∀ (k : ℝ) (h2 : (l ∩ set_of (λ (p : ℝ × ℝ), (p.1^2 / 4) + p.2^2 = 1)).nonempty),
  (∀ p : ℝ × ℝ, p ∈ l → p ∈ set_of (λ (p : ℝ × ℝ), p.1^2 + p.2^2 = 5) →
     ∃ len : ℝ, len = 2 * sqrt 2) → 
  (m = 3) :=
sorry

end ellipse_standard_equation_find_m_l736_736290


namespace even_combinations_486_l736_736140

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736140


namespace find_sinA_find_lengthBC_l736_736351

variables {A B C : Type*} [decidable_eq A] [decidable_eq B] [decidable_eq C]

-- Given conditions
variables (cosB : ℝ) (cosC : ℝ) (sinA : ℝ) (areaABC : ℝ)
variables (hcosB : cosB = -5 / 13) (hcosC : cosC = 4 / 5)
variables (hareaABC : areaABC = 33 / 2) (hsinA : sinA = 33 / 65)

-- Prove the value of sin A
theorem find_sinA : sinA = 33 / 65 :=
by { rw hsinA, exact rfl }

-- Prove the length of BC
noncomputable def lengthBC (AB : ℝ) (AC : ℝ) : ℝ :=
  AB * sinA / (5 / 3)

theorem find_lengthBC (AB : ℝ) (AC : ℝ) (h1 : AB * AC = 65) (h2 : AC = 20 / 13 * AB) : lengthBC sinA (33 / 65) =  11 / 2 :=
sorry

end find_sinA_find_lengthBC_l736_736351


namespace solution_set_of_inequality_range_of_a_l736_736657

-- Question (1)
theorem solution_set_of_inequality (a : ℝ) (x : ℝ): 
  a = -2 → (|x - 1| + |x + a| < (1 / 2) * x + 3) ↔ (0 < x ∧ x < 4) := 
by
  sorry

-- Question (2)
theorem range_of_a (a x : ℝ) (h : a > -1) (hx : x ∈ set.Icc (-a) 1):
  (|x - 1| + |x + a| ≤ (1 / 2) * x + 3) →
  (-1 < a ∧ a ≤ 5/2) :=
by
  sorry

end solution_set_of_inequality_range_of_a_l736_736657


namespace determine_ab_l736_736044

noncomputable def f (a b : ℚ) := λ x : ℚ, a * x^3 - 7 * x^2 + b * x - 6

theorem determine_ab : ∃ a b : ℚ, 
  (f a b 2 = -8) ∧ (f a b (-1) = -18) ∧ 
  a = 2/3 ∧ b = 13/3 :=
by
  use 2/3, 13/3
  simp [f]
  sorry

end determine_ab_l736_736044


namespace women_fair_hair_percentage_l736_736953

variable (E : ℝ)  -- Total number of employees
variable (P_fair_hair : ℝ)  -- Percentage of employees who have fair hair
variable (P_women_given_fair : ℝ)  -- Percentage of fair-haired employees who are women

-- Conditions
def percentage_fair_hair (E : ℝ) : ℝ := 0.75 * E
def percentage_women_given_fair (E : ℝ) (P_fair_hair : ℝ) : ℝ := 0.40 * P_fair_hair

-- Question
def percentage_women_with_fair_hair (E : ℝ) (P_fair_hair : ℝ) (P_women_given_fair : ℝ) : ℝ :=
  (P_women_given_fair / E) * 100

theorem women_fair_hair_percentage : 
  P_fair_hair = 0.75 * E → 
  P_women_given_fair = 0.40 * (0.75 * E) → 
  percentage_women_with_fair_hair E P_fair_hair P_women_given_fair = 30 := by
  intros h1 h2
  sorry

end women_fair_hair_percentage_l736_736953


namespace exists_n_l736_736798

open Nat

theorem exists_n (p a k : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hk : p^a < k ∧ k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (binom n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
by
  sorry

end exists_n_l736_736798


namespace spherical_to_rectangular_conversion_l736_736183

open Real

theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
    ρ = 3 →
    θ = 3 * π / 4 →
    φ = π / 6 →
    x = ρ * sin φ * cos θ →
    y = ρ * sin φ * sin θ →
    z = ρ * cos φ →
    (sqrt (x^2 + (-y)^2 + z^2) = 3 ∧
     atan2 (-y) x = 5 * π / 4 ∧
     acos (z / 3) = π / 6) := 
begin
  intros ρ θ φ x y z hρ hθ hφ hx hy hz,
  -- Convert initial spherical to rectangular coordinates
  rw [hρ, hθ, hφ] at hx hy hz,
  have hx : x = -3 * sqrt 2 / 4 := by sorry,
  have hy : y = 3 * sqrt 2 / 4 := by sorry,
  have hz : z = 3 * sqrt 3 / 2 := by sorry,

  -- New rectangular coordinates for (x, -y, z)
  let x' := x,
  let y' := -y,
  let z' := z,

  -- Convert back to spherical coordinates
  have hρ' : sqrt (x'^2 + y'^2 + z'^2) = 3 := by sorry,
  have hθ' : atan2 y' x' = 5 * π / 4 := by sorry,
  have hφ' : acos (z' / 3) = π / 6 := by sorry,
  
  refine ⟨hρ', hθ', hφ'⟩,
end

end spherical_to_rectangular_conversion_l736_736183


namespace minimum_sum_of_x_and_y_l736_736635

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 4 * y = x * y

theorem minimum_sum_of_x_and_y (x y : ℝ) (h : conditions x y) : x + y ≥ 9 := by
  sorry

end minimum_sum_of_x_and_y_l736_736635


namespace opposite_of_neg3_l736_736429

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l736_736429


namespace simplify_expression_l736_736002

theorem simplify_expression (x : ℤ) : (3 * x) ^ 3 + (2 * x) * (x ^ 4) = 27 * x ^ 3 + 2 * x ^ 5 :=
by sorry

end simplify_expression_l736_736002


namespace infinite_solutions_to_congruence_l736_736391

theorem infinite_solutions_to_congruence :
  ∃ᶠ n in atTop, 3^((n-2)^(n-1)-1) ≡ 1 [MOD 17 * n^2] :=
by
  sorry

end infinite_solutions_to_congruence_l736_736391


namespace compute_result_l736_736572

-- Define the operations a # b and b # c
def operation (a b : ℤ) : ℤ := a * b - b + b^2

-- Define the expression for (3 # 8) # z given the operations
def evaluate (z : ℤ) : ℤ := operation (operation 3 8) z

-- Prove that (3 # 8) # z = 79z + z^2
theorem compute_result (z : ℤ) : evaluate z = 79 * z + z^2 := 
by
  sorry

end compute_result_l736_736572


namespace sum_of_first_9_primes_is_100_first_prime_number_is_2_l736_736457

def first_prime_numbers : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23]

theorem sum_of_first_9_primes_is_100 :
  first_prime_numbers.sum = 100 := by
  sorry

theorem first_prime_number_is_2 :
  first_prime_numbers.head = 2 := by
  sorry

end sum_of_first_9_primes_is_100_first_prime_number_is_2_l736_736457


namespace percentage_exceeds_l736_736528

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l736_736528


namespace feasible_measures_l736_736100

-- Conditions for the problem
def condition1 := "Replace iron filings with iron pieces"
def condition2 := "Use excess zinc pieces instead of iron pieces"
def condition3 := "Add a small amount of CuSO₄ solution to the dilute hydrochloric acid"
def condition4 := "Add CH₃COONa solid to the dilute hydrochloric acid"
def condition5 := "Add sulfuric acid of the same molar concentration to the dilute hydrochloric acid"
def condition6 := "Add potassium sulfate solution to the dilute hydrochloric acid"
def condition7 := "Slightly heat (without considering the volatilization of HCl)"
def condition8 := "Add NaNO₃ solid to the dilute hydrochloric acid"

-- The criteria for the problem
def isFeasible (cond : String) : Prop :=
  cond = condition1 ∨ cond = condition2 ∨ cond = condition3 ∨ cond = condition7

theorem feasible_measures :
  ∀ cond, 
  cond ≠ condition4 →
  cond ≠ condition5 →
  cond ≠ condition6 →
  cond ≠ condition8 →
  isFeasible cond :=
by
  intros
  sorry

end feasible_measures_l736_736100


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_for_max_range_l736_736930

-- Define the conditions
variables (a b : ℕ) (h : a > b)

-- Maximum possible range
theorem max_range_eq_a : max_range a b = a := 
sorry

-- Minimum possible range
theorem min_range_eq_a_minus_b : min_range a b = a - b := 
sorry

-- Number of sequences for maximum range
theorem num_sequences_for_max_range : num_sequences_for_max_range a b = nat.choose (a + b) a := 
sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_for_max_range_l736_736930


namespace new_rectangle_area_correct_l736_736619

variable {a b : ℝ} (h : a > b) 

def new_rectangle_base := a^2 + b^2 + a
def new_rectangle_height := a^2 + b^2 - b
def new_rectangle_area := new_rectangle_base * new_rectangle_height

theorem new_rectangle_area_correct :
  new_rectangle_area = a^4 + a^3 + 2*a^2*b^2 + a*b^3 - a*b + b^4 - b^3 - b^2 :=
by
  sorry

end new_rectangle_area_correct_l736_736619


namespace ratio_waiting_to_walking_l736_736807

theorem ratio_waiting_to_walking 
  (trip_time_hrs : ℕ) (bus_time_min : ℕ) (walk_time_min : ℕ) (train_time_hrs : ℕ)
  (h_trip : trip_time_hrs = 8) (h_bus : bus_time_min = 75) (h_walk : walk_time_min = 15) (h_train : train_time_hrs = 6) :
  (trip_time_hrs * 60 - (bus_time_min + walk_time_min + train_time_hrs * 60)) / walk_time_min = 2 := by
  -- Convert hour to minute for calculations
  let trip_time_min := trip_time_hrs * 60
  let train_time_min := train_time_hrs * 60
  -- Calculate the total trip time
  let total_active_time := bus_time_min + walk_time_min + train_time_min
  -- Calculate the waiting time
  let wait_time_min := trip_time_min - total_active_time
  -- Prove the ratio, given conditions
  have h_wait_time : wait_time_min = 30 := sorry
  have h_ratio := wait_time_min / walk_time_min
  show h_ratio = 2 from sorry

end ratio_waiting_to_walking_l736_736807


namespace geom_seq_product_a2_a3_l736_736767

theorem geom_seq_product_a2_a3 :
  ∃ (a_n : ℕ → ℝ), (a_n 1 * a_n 4 = -3) ∧ (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1) ^ (n - 1)) → a_n 2 * a_n 3 = -3 :=
by
  sorry

end geom_seq_product_a2_a3_l736_736767


namespace angle_of_inclination_l736_736642

theorem angle_of_inclination (a b c : ℝ) (h_symm : ∀ x, y = a * sin x + b * cos x → y = a * sin (π / 6 - x) + b * cos (π / 6 - x)) :
  Real.arctan (-a / b) = 5 * π / 6 := by
  sorry

end angle_of_inclination_l736_736642


namespace positive_products_count_l736_736079

theorem positive_products_count (a b c : ℤ) : 
  let p₁ := a * b,
      p₂ := b * c,
      p₃ := c * a in
  (p₁ > 0 ∧ p₂ > 0 ∧ p₃ > 0) ∨ 
  ((p₁ > 0 ∧ p₂ ≤ 0 ∧ p₃ ≤ 0) ∨ 
   (p₁ ≤ 0 ∧ p₂ > 0 ∧ p₃ ≤ 0) ∨ 
   (p₁ ≤ 0 ∧ p₂ ≤ 0 ∧ p₃ > 0)) :=
sorry

end positive_products_count_l736_736079


namespace evaluate_expression_l736_736611

theorem evaluate_expression (x : ℝ) (h : 3 * x^3 - x = 1) : 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := 
by
  sorry

end evaluate_expression_l736_736611


namespace value_of_a_plus_b_l736_736690

theorem value_of_a_plus_b (a b : ℝ) (h : (2 * a + 2 * b - 1) * (2 * a + 2 * b + 1) = 99) :
  a + b = 5 ∨ a + b = -5 :=
sorry

end value_of_a_plus_b_l736_736690


namespace cube_side_length_l736_736033

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l736_736033


namespace find_y_l736_736606

noncomputable def greatest_even_le (y : ℝ) : ℝ :=
  if h : ∃ n, n ≠ 0 ∧ ∃ m, y ≥ (2 * m : ℝ) ∧ (2 * m : ℝ) = n
  then nat.find h else 0

theorem find_y :
  (greatest_even_le 6.15 = 6) →
  (6.15 - 6 = 0.15000000000000036) →
  (∃ y : ℝ, y = 6.15) :=
by
  intros h1 h2
  use 6.15
  exact sorry

end find_y_l736_736606


namespace percentage_exceeds_l736_736529

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l736_736529


namespace number_of_even_results_l736_736144

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736144


namespace sin_inequality_holds_l736_736595

open Real

-- Definitions for the required conditions
def range0_2pi (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * pi

def sin_inequality (x y : ℝ) : Prop := sin (x + y) ≤ sin x + sin y

-- The statement
theorem sin_inequality_holds (x y : ℝ) (hx : range0_2pi x) (hy : range0_2pi y) :
  sin_inequality x y ↔ range0_2pi y ∧ y ≤ pi :=
sorry

end sin_inequality_holds_l736_736595


namespace z_conjugate_sum_l736_736702

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736702


namespace herons_formula_l736_736089

-- Define the context and the statement to be proven
theorem herons_formula (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2 in 
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in 
  area = 1 / 4 * Real.sqrt (4 * a ^ 2 * c ^ 2 - (a ^ 2 + c ^ 2 - b ^ 2) ^ 2) := 
begin
  -- The proof would go here
  sorry
end

end herons_formula_l736_736089


namespace exists_1000_digit_divisible_by_sum_of_digits_l736_736782

theorem exists_1000_digit_divisible_by_sum_of_digits :
  ∃ (n : ℕ), (nat.digits 10 n).length = 1000 ∧ (∀ d ∈ nat.digits 10 n, d ≠ 0) ∧ n % (nat.digits 10 n).sum = 0 := 
sorry

end exists_1000_digit_divisible_by_sum_of_digits_l736_736782


namespace problem_statement_l736_736691

-- Define a : ℝ such that (a + 1/a)^3 = 7
variables (a : ℝ) (h : (a + 1/a)^3 = 7)

-- Goal: Prove that a^4 + 1/a^4 = 1519/81
theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 7) : a^4 + 1/a^4 = 1519 / 81 := 
sorry

end problem_statement_l736_736691


namespace polynomial_zero_l736_736251

theorem polynomial_zero (a b c : ℝ) (h : ∛a + ∛b = ∛c) : 
  27 * a * b * c - (c - a - b)^3 = 0 := 
by 
  sorry

end polynomial_zero_l736_736251


namespace complex_conjugate_sum_l736_736717

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736717


namespace number_of_even_results_l736_736134

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736134


namespace f_expression_f_monotonicity_solve_inequality_l736_736295

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (a * x + b) / (x^2 + 1)

axiom f_odd : ∀ (x : ℝ) (a b : ℝ), f (-x) a b = -f x a b 
axiom f_at_1_2 : ∀ (a b : ℝ), f (1 / 2) a b = 2 / 5

theorem f_expression (a b : ℝ) : f (1 / 2) 1 0 = 2 / 5 → f (x : ℝ) 1 0 = x / (x^2 + 1) := 
begin
  sorry
end

theorem f_monotonicity (a b : ℝ) : ∀ (x1 x2 : ℝ), -1 < x1 → x1 < x2 → x2 < 1 → f x1 1 0 < f x2 1 0 :=
begin
  sorry
end

theorem solve_inequality (a b : ℝ) : ∀ (x : ℝ), f (2 * x - 1) 1 0 + f x 1 0 < 0 → 0 < x → x < 1 / 3 :=
begin
  sorry
end

end f_expression_f_monotonicity_solve_inequality_l736_736295


namespace fraction_equivalent_l736_736483

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l736_736483


namespace area_of_special_triangle_l736_736544

-- Definitions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def triangle_incenter (a b c : ℕ) (ax ay bx by cx cy : ℤ) : ℤ × ℤ :=
  let px := (a * bx + b * cx + c * ax) / (a + b + c),
      py := (a * by + b * cy + c * ay) / (a + b + c)
  (px, py)

def triangle_circumcenter (ax ay bx by cx cy : ℤ) : ℤ × ℤ :=
  let mx := (ax + bx) / 2,
      my := (ay + by) / 2
  (mx, my)

def triangle_centroid (ax ay bx by cx cy : ℤ) : ℤ × ℤ :=
  let gx := (ax + bx + cx) / 3,
      gy := (ay + by + cy) / 3
  (gx, gy)

-- Main theorem
theorem area_of_special_triangle :
  ∀ (a b c : ℕ) (ax ay bx by cx cy : ℤ),
  a = 18 → b = 24 → c = 30 →
  ax = 0 → ay = 0 →
  bx = 24 → by = 0 →
  cx = 0 → cy = 18 →
  is_right_triangle a b c →
  let i := triangle_incenter a b c ax ay bx by cx cy,
      o := triangle_circumcenter ax ay bx by cx cy,
      g := triangle_centroid ax ay bx by cx cy in
  let area : ℤ := abs ((i.fst - g.fst) * (o.snd - g.snd) - (i.snd - g.snd) * (o.fst - g.fst)) / 2 in
  area = 6 := by
  intros a b c ax ay bx by cx cy ha hb hc hax hay hbx hby hcx hcy hr h i o g area
  sorry

end area_of_special_triangle_l736_736544


namespace fraction_simplification_l736_736995

theorem fraction_simplification :
  (3 / (2 - (3 / 4))) = 12 / 5 := 
by
  sorry

end fraction_simplification_l736_736995


namespace probability_of_rectangle_area_greater_than_32_l736_736386

-- Definitions representing the problem conditions
def segment_length : ℝ := 12
def point_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ segment_length
def rectangle_area (x : ℝ) : ℝ := x * (segment_length - x)

-- The probability we need to prove. 
noncomputable def desired_probability : ℝ := 1 / 3

theorem probability_of_rectangle_area_greater_than_32 :
  (∀ x, point_C x → rectangle_area x > 32) → (desired_probability = 1 / 3) :=
by
  sorry

end probability_of_rectangle_area_greater_than_32_l736_736386


namespace function_odd_and_decreasing_l736_736296

noncomputable def f (a x : ℝ) : ℝ := (1 / a) ^ x - a ^ x

theorem function_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -f a x) ∧ (∀ x y, x < y → f a x > f a y) :=
by
  sorry

end function_odd_and_decreasing_l736_736296


namespace probability_of_two_accurate_forecasts_l736_736874

noncomputable def event_A : Type := {forecast : ℕ | forecast = 1}

def prob_A : ℝ := 0.9
def prob_A' : ℝ := 1 - prob_A

-- Define that there are 3 independent trials
def num_forecasts : ℕ := 3

-- Given
def probability_two_accurate (x : ℕ) : ℝ :=
if x = 2 then 3 * (prob_A^2 * prob_A') else 0

-- Statement to be proved
theorem probability_of_two_accurate_forecasts : probability_two_accurate 2 = 0.243 := by
  -- Proof will go here
  sorry

end probability_of_two_accurate_forecasts_l736_736874


namespace min_value_of_f_l736_736601

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  (∀ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) → f x y ≥ 12 / 35) ∧
  ∃ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) ∧ f x y = 12 / 35 :=
by
  sorry

end min_value_of_f_l736_736601


namespace identify_non_standard_medal_in_three_weighings_l736_736604

-- Definitions for the conditions
def GMedal := {g : ℕ // g < 7}
def SMedal := {s : ℕ // s < 7}
def BMedal := {b : ℕ // b < 7}

structure Medals :=
  (gold: fin 7 → ℕ)
  (silver: fin 7 → ℕ)
  (bronze: fin 7 → ℕ)
  (non_standard: ℕ × char)

-- Conditions
axiom gold_non_standard_lighter (m : Medals) (i : fin 7): (m.non_standard.2 = 'G') → (m.gold i) > (m.gold (⟨ m.non_standard.1, h_proof ⟩))
axiom bronze_non_standard_heavier (m : Medals) (i : fin 7): (m.non_standard.2 = 'B') → (m.bronze i) < (m.bronze (⟨ m.non_standard.1, h_proof ⟩))
axiom silver_non_standard_diff (m : Medals) (i : fin 7): (m.non_standard.2 = 'S') → (m.silver i) ≠ (m.silver (⟨ m.non_standard.1, h_proof ⟩))

-- Theorem statement
theorem identify_non_standard_medal_in_three_weighings (m : Medals) :
  ∃ (weighings : list (fin 7 × char × bool)), 
    (list.length weighings ≤ 3) ∧ 
    ∀ (g : fin 7), ∀ (s : fin 7), ∀ (b : fin 7), 
      (list.nth weighings g.snd = m.non_standard → g.fst = m.non_standard.1) ∧
      (list.nth weighings s.snd = m.non_standard → s.fst = m.non_standard.1) ∧
      (list.nth weighings b.snd = m.non_standard → b.fst = m.non_standard.1) :=
sorry

end identify_non_standard_medal_in_three_weighings_l736_736604


namespace sum_of_x_satisfying_equations_l736_736803

theorem sum_of_x_satisfying_equations:
  let S := { (x : ℂ, y : ℂ, z : ℂ) | x + y * z = 9 ∧ y + x * z = 13 ∧ z + x * y = 13 } in
  (∑ p in S, p.1) = 8 :=
by
  sorry

end sum_of_x_satisfying_equations_l736_736803


namespace phase_shift_of_cosine_function_l736_736602

theorem phase_shift_of_cosine_function :
  ∀ (x : ℝ), 4 * cos (x + π / 3) = 4 * cos (x - (-π / 3)) :=
by
  intros x
  sorry

end phase_shift_of_cosine_function_l736_736602


namespace volume_of_region_l736_736099

open MeasureTheory Set Filter

noncomputable theory
open_locale classical

def regionVolume : ℝ := 8

theorem volume_of_region :
  let S := {p : ℝ³ | |p.1 + p.2| ≤ 1 ∧ |p.1 + p.2 + p.3 - 2| ≤ 2} in
  volume S = regionVolume :=
sorry

end volume_of_region_l736_736099


namespace region_probability_l736_736560

def x_set := {-1, 1}
def y_set := {-2, 0, 2}

def region (x y : Int) : Prop := x + 2 * y ≥ 1

def satisfying_points : Finset (Int × Int) :=
  Finset.filter (λ p, region p.1 p.2) (x_set.product y_set)

def total_points : Finset (Int × Int) := x_set.product y_set

def probability : ℚ := (satisfying_points.card : ℚ) / (total_points.card : ℚ)

theorem region_probability :
  probability = 1 / 2 :=
by
  sorry

end region_probability_l736_736560


namespace sum_of_x_satisfying_equations_l736_736804

theorem sum_of_x_satisfying_equations:
  let S := { (x : ℂ, y : ℂ, z : ℂ) | x + y * z = 9 ∧ y + x * z = 13 ∧ z + x * y = 13 } in
  (∑ p in S, p.1) = 8 :=
by
  sorry

end sum_of_x_satisfying_equations_l736_736804


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l736_736269

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l736_736269


namespace base3_to_base10_equiv_l736_736575

theorem base3_to_base10_equiv : 
  let repr := 1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0
  repr = 142 :=
by
  sorry

end base3_to_base10_equiv_l736_736575


namespace problem1_problem2_l736_736503

-- Problem (1)
theorem problem1 (a b : ℝ) (ha: a > 0) (hb: b > 0) (hab: a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem (2)
theorem problem2 (a : ℕ → ℝ) (h : ∀ n, a n > 0) (h_eq : ∀ n, ((List.sum (List.map a (List.range n)))^2 = (List.sum (List.map (λ x, (a x)^3) (List.range n)))) ) : 
  ∀ n, a n = n :=
sorry

end problem1_problem2_l736_736503


namespace distance_traveled_correct_modulus_delta_velocity_correct_l736_736876

open real

noncomputable def robot_position (t : ℝ) : ℝ × ℝ :=
if t ≤ 7 then (t - 6)^2 else ((t - 6)^2, (t - 7)^2)

noncomputable def distance_traveled_in_7_minutes (t : ℝ) : ℝ :=
if t ≤ 7 then abs ((0 - 6)^2 - (7 - 6)^2) else 0

noncomputable def velocity_components (t : ℝ) : ℝ × ℝ :=
if 0 ≤ t ∧ t < 7 then (2 * (t - 6), 0)
else if t ≥ 7 then (2 * (t - 6), 2 * (t - 7))
else (0, 0)

noncomputable def delta_velocity (t : ℝ) : ℝ × ℝ :=
if t = 8 then
  let v7 := velocity_components 7 in
  let v8 := velocity_components 8 in
  (v8.1 - v7.1, v8.2 - v7.2)
else (0, 0)

noncomputable def modulus_delta_velocity (t : ℝ) : ℝ :=
if t = 8 then
  let dv := delta_velocity t in
  real.sqrt (dv.1^2 + dv.2^2)
else 0

theorem distance_traveled_correct : distance_traveled_in_7_minutes 7 = 37 :=
by sorry

theorem modulus_delta_velocity_correct : modulus_delta_velocity 8 = 2 * sqrt 2 :=
by sorry

end distance_traveled_correct_modulus_delta_velocity_correct_l736_736876


namespace square_side_length_l736_736096

theorem square_side_length (A : ℝ) (h : A = 25) : ∃ s : ℝ, s * s = A ∧ s = 5 :=
by
  sorry

end square_side_length_l736_736096


namespace least_possible_a_plus_b_l736_736027

-- Define the conditions
def is_valid_base (n : ℕ) := n > 1

-- Define the conversion from a numeral to its decimal representation
def base_conversion (num : ℕ) (base : ℕ) : ℕ :=
  num.digit_sum base

-- State the theorem
theorem least_possible_a_plus_b (a b : ℕ) 
  (h₁ : is_valid_base a) 
  (h₂ : is_valid_base b) 
  (h₃ : base_conversion 78 a = base_conversion 87 b) : 
  a + b = 17 :=
sorry

end least_possible_a_plus_b_l736_736027


namespace maria_cookies_left_l736_736376

theorem maria_cookies_left
    (total_cookies : ℕ) -- Maria has 60 cookies
    (friend_share : ℕ) -- 20% of the initial cookies goes to the friend
    (family_share : ℕ) -- 1/3 of the remaining cookies goes to the family
    (eaten_cookies : ℕ) -- Maria eats 4 cookies
    (neighbor_share : ℕ) -- Maria gives 1/6 of the remaining cookies to neighbor
    (initial_cookies : total_cookies = 60)
    (friend_fraction : friend_share = total_cookies * 20 / 100)
    (remaining_after_friend : ℕ := total_cookies - friend_share)
    (family_fraction : family_share = remaining_after_friend / 3)
    (remaining_after_family : ℕ := remaining_after_friend - family_share)
    (eaten : eaten_cookies = 4)
    (remaining_after_eating : ℕ := remaining_after_family - eaten_cookies)
    (neighbor_fraction : neighbor_share = remaining_after_eating / 6)
    (neighbor_integerized : neighbor_share = 4) -- assumed whole number for neighbor's share
    (remaining_after_neighbor : ℕ := remaining_after_eating - neighbor_share) : 
    remaining_after_neighbor = 24 :=
sorry  -- The statement matches the problem, proof is left out

end maria_cookies_left_l736_736376


namespace perfect_squares_good_l736_736192

def is_good (A : set ℕ) := ∀ (n : ℕ), n > 0 → ∃! p : ℕ, prime p ∧ n - p ∈ A

theorem perfect_squares_good : is_good { n | ∃ k : ℕ, n = k * k } :=
sorry

end perfect_squares_good_l736_736192


namespace cutting_square_into_8_pieces_l736_736687

theorem cutting_square_into_8_pieces :
  ∃ n : ℕ, n = 54 ∧ 
  ∀ P : set (set (ℝ × ℝ)), 
    (∀ p ∈ P, (p = congruent_polygon ∧
               (∀ angle ∈ interior_angles p, angle = 45 ∨ angle = 90))) ∧
    (∀ distinct1 distinct2 ∈ P, distinct1 ≠ distinct2 → different_cutting_locations distinct1 distinct2) ∧
   (consider_rotations_reflections distinct1 distinct2) :=
sorry

end cutting_square_into_8_pieces_l736_736687


namespace complex_conjugate_sum_l736_736732

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736732


namespace exists_m_integer_l736_736795

def f (r : ℝ) : ℝ := r * (⌈r⌉ : ℝ)

def f_iter (r : ℝ) (m : ℕ) : ℝ :=
  nat.rec_on m r (λ m' f_m', f f_m')

def v2 (k : ℕ) : ℕ :=
  nat.find_greatest (λ n, 2^n ∣ k) k

theorem exists_m_integer (k : ℕ) (h_pos : 0 < k) :
  ∃ m : ℕ, f_iter (k + 0.5) m ∈ ℤ :=
  sorry

end exists_m_integer_l736_736795


namespace eval_integral_F_l736_736242

def F (x : ℝ) : ℝ := x^2 * sin x + sqrt (4 - x^2)

theorem eval_integral_F : ∫ x in (-2 : ℝ)..(2 : ℝ), F x = 2 * Real.pi :=
by
  sorry

end eval_integral_F_l736_736242


namespace fraction_equivalent_l736_736484

theorem fraction_equivalent (x : ℝ) : x = 433 / 990 ↔ x = 0.4 + 37 / 990 * 10 ^ -2 :=
by
  sorry

end fraction_equivalent_l736_736484


namespace max_binomial_coeff_max_term_coeff_l736_736292

theorem max_binomial_coeff (x : ℝ) :
  let n := 7
  let T4 := (nat.choose n 3) * (2 * real.sqrt x)^3
  let T5 := (nat.choose n 4) * (2 * real.sqrt x)^4
  T4 = 280 * x^((3:ℝ)/2) ∧ T5 = 560 * x^2 :=
by
  sorry

theorem max_term_coeff (x : ℝ) :
  let n := 7
  let r := 5
  let coeff := (nat.choose n r) * (2 * real.sqrt x) ^ r
  coeff = 672 * x^((5:ℝ)/2) :=
by
  sorry

end max_binomial_coeff_max_term_coeff_l736_736292


namespace f_minimum_at_l736_736294

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem f_minimum_at : ∀ x : ℝ, x = -Real.log 2 → (∀ y : ℝ, f y ≥ f x) :=
by
  sorry

end f_minimum_at_l736_736294


namespace even_combinations_486_l736_736142

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736142


namespace octal_742_is_482_in_decimal_l736_736197

def octal_to_decimal (n : ℕ) : ℕ := 
  match n with
  | 742 => 2 * 8^0 + 4 * 8^1 + 7 * 8^2
  | _ => 0 -- We only define behavior for 742; other cases return 0

theorem octal_742_is_482_in_decimal : octal_to_decimal 742 = 482 := by
  unfold octal_to_decimal
  simp
  sorry

end octal_742_is_482_in_decimal_l736_736197


namespace opposite_of_neg3_l736_736427

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l736_736427


namespace result_prob_a_l736_736835

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l736_736835


namespace equal_at_three_l736_736580

def g (x : ℝ) : ℝ := 5 * x - 12
def g_inv (x : ℝ) : ℝ := (x + 12) / 5

theorem equal_at_three (x : ℝ) : g x = g_inv x → x = 3 :=
by
  sorry

end equal_at_three_l736_736580


namespace bodies_meet_in_time_distance_by_A_distance_by_B_l736_736898

noncomputable theory
open_locale classical

-- Define initial conditions
def initial_distance : ℝ := 343
def initial_velocity_A : ℝ := 3
def acceleration_A : ℝ := 5
def initial_velocity_B : ℝ := 4
def acceleration_B : ℝ := 7

-- Function to calculate the time t at which two bodies meet
def meet_time (d v_a a_a v_b a_b : ℝ) : ℝ :=
(let t := (-v_a + -v_b + real.sqrt ((v_a + v_b)^2 + 4 * a_a * d + 4 * a_b * d)) / (2 * (a_a + a_b)) in 
if t ≥ 0 then t else (-v_a + -v_b - real.sqrt ((v_a + v_b)^2 + 4 * a_a * d + 4 * a_b * d)) / (2 * (a_a + a_b)))

-- Function to calculate the distance travelled by a body given time
def distance_travelled (v a t : ℝ) : ℝ :=
v * t + 0.5 * a * t^2

-- Lean statement to prove the mathematically equivalent problem
theorem bodies_meet_in_time :
  meet_time initial_distance initial_velocity_A acceleration_A initial_velocity_B acceleration_B = 7 :=
by sorry

theorem distance_by_A :
  distance_travelled initial_velocity_A acceleration_A 7 = 143.5 :=
by sorry

theorem distance_by_B :
  distance_travelled initial_velocity_B acceleration_B 7 = 199.5 :=
by sorry

end bodies_meet_in_time_distance_by_A_distance_by_B_l736_736898


namespace log27_3_eq_one_third_l736_736235

theorem log27_3_eq_one_third : ∃ x : ℝ, (27 = 3^3) ∧ (27^x = 3) ∧ (x = 1/3) :=
by {
  have h1: 27 = 3^3 := by norm_num,
  have h2: 27^(1/3) = 3 := by {
    rw [h1, ← real.rpow_mul (real.rpow_pos_of_pos (by norm_num) (3: ℝ))],
    norm_num
  },
  exact ⟨1/3, h1, h2⟩
}

end log27_3_eq_one_third_l736_736235


namespace hilary_total_kernels_l736_736679

-- Define the conditions given in the problem
def ears_per_stalk : ℕ := 4
def total_stalks : ℕ := 108
def kernels_per_ear_first_half : ℕ := 500
def additional_kernels_second_half : ℕ := 100

-- Express the main problem as a theorem in Lean
theorem hilary_total_kernels : 
  let total_ears := ears_per_stalk * total_stalks
  let half_ears := total_ears / 2
  let kernels_first_half := half_ears * kernels_per_ear_first_half
  let kernels_per_ear_second_half := kernels_per_ear_first_half + additional_kernels_second_half
  let kernels_second_half := half_ears * kernels_per_ear_second_half
  kernels_first_half + kernels_second_half = 237600 :=
by
  sorry

end hilary_total_kernels_l736_736679


namespace compare_powers_l736_736268

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l736_736268


namespace part_a_part_b_l736_736832

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l736_736832


namespace sum_of_triples_l736_736806

theorem sum_of_triples : 
  ∀ (x y z : ℂ), 
    (x + y * z = 9) ∧ 
    (y + x * z = 13) ∧ 
    (z + x * y = 13) → 
    let S := { (x', y', z') : ℂ × ℂ × ℂ | 
                x' + y' * z' = 9 ∧ 
                y' + x' * z' = 13 ∧ 
                z' + x' * y' = 13 } in 
    (Σ t : S, t.1.1) = 9 := 
sorry

end sum_of_triples_l736_736806


namespace joint_purchases_popular_in_countries_joint_purchases_not_popular_in_building_l736_736936

-- Definitions using conditions from problem (a)
def cost_savings_info_sharing : Prop := 
  ∀ (groupBuy : Type), (significant_cost_savings : Prop) ∧ (accurate_product_evaluation : Prop)

-- Definitions using conditions from problem (b)
def transactional_costs_proximity : Prop := 
  ∀ (neighborhoodGroup : Type), 
    (diverse_products : Prop) ∧ 
    (significant_transactional_costs : Prop) ∧ 
    (organizational_burdens : Prop) ∧ 
    (proximity_to_stores : Prop)

-- Lean 4 statements to prove the questions
theorem joint_purchases_popular_in_countries :
  cost_savings_info_sharing → 
  (practice_of_joint_purchases_popular : Prop) := 
sorry

theorem joint_purchases_not_popular_in_building :
  transactional_costs_proximity → 
  (practice_of_joint_purchases_not_popular : Prop) :=
sorry

end joint_purchases_popular_in_countries_joint_purchases_not_popular_in_building_l736_736936


namespace determine_abs_d_l736_736009

noncomputable def polynomial_condition (a b c d : ℤ) : Prop :=
  let z : ℂ := 3 + complex.i in
  a * z^5 + b * z^4 + c * z^3 + d * z^2 + c * z + b + a = 0

theorem determine_abs_d (a b c d : ℤ) (h : gcd (gcd (gcd a b) c) d = 1) :
    polynomial_condition a b c d → |d| = 16 :=
by
  intro h_poly_cond
  sorry

end determine_abs_d_l736_736009


namespace expected_candies_per_block_l736_736593

-- Define the probability distributions and expected candies
def house1_expectation : ℝ :=
  (4 * 0.25) + (5 * 0.25) + (6 * 0.25) + (7 * 0.25)

def house2_expectation : ℝ :=
  (5 * 0.20) + (6 * 0.20) + (7 * 0.20) + (8 * 0.20)

def house3_expectation : ℝ :=
  (6 * 0.25) + (7 * 0.25) + (8 * 0.25) + (9 * 0.25)

def house4_expectation : ℝ :=
  (7 + 8 + 9) / 3

def house5_expectation : ℝ :=
  (10 * 0.50) + (11 * 0.50)

-- Total expected candies per block
def total_expectation : ℝ :=
  house1_expectation + house2_expectation + house3_expectation + house4_expectation + house5_expectation

theorem expected_candies_per_block : total_expectation = 36.7 :=
by
  -- Lean expects a proof here, so we use sorry to skip the actual proof.
  sorry

end expected_candies_per_block_l736_736593


namespace result_prob_a_l736_736836

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l736_736836


namespace flowers_in_each_basket_l736_736214

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end flowers_in_each_basket_l736_736214


namespace z_conjugate_sum_l736_736709

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736709


namespace hyperbola_eccentricity_proof_l736_736637

open Real

noncomputable def focus_of_parabola (p : ℝ) : (ℝ × ℝ) :=
  (p / 2, 0)

noncomputable def hyperbola_eccentricity (a b c e : ℝ) : Prop :=
  ∃ (p : ℝ), p > 0 ∧ a > 0 ∧ b > 0 ∧
    y^2 = 2 * p * x ∧
    (x / a^2) - (y / b^2) = 1 ∧
    p / 2 = c ∧
    c^2 = a^2 + b^2 ∧
    ((p^2 / 4) / a^2) - (p^2 / b^2) = 1 ∧
    e^2 = 3 + 2 * sqrt(2)

theorem hyperbola_eccentricity_proof :
  ∀ (a b c e : ℝ), hyperbola_eccentricity a b c e → e = 1 + sqrt(2) :=
by
  intros a b c e h
  sorry

end hyperbola_eccentricity_proof_l736_736637


namespace triangle_property_l736_736778

/-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
    Vectors m = (c, a - b) and n = (sin B - sin C, sin A + sin B) are perpendicular.
    D is a point on AC such that AD = BD and BC = 3. -/
theorem triangle_property 
  (a b c : ℝ) (A B C : ℝ) (D : ℝ)
  (h1 : A + B + C = π)
  (h2 : c * (sin B - sin C) + (a - b) * (sin A + sin B) = 0)
  (h3 : a / sin A = b / sin B)
  (h4 : a / sin A = c / sin C)
  (h5 : A ∈ (0, π))
  (h6 : BC = 3)
  (h7 : AD = BD): 
  (A = π / 3) ∧ (area_of_triangle B C D = 3*sqrt 3 / 4) := sorry

end triangle_property_l736_736778


namespace compute_exponent_multiplication_l736_736569

theorem compute_exponent_multiplication : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end compute_exponent_multiplication_l736_736569


namespace unique_sequence_length_l736_736463

theorem unique_sequence_length :
  ∃ (b : ℕ → ℕ) (m : ℕ), (∀ i j, i < j → b i < b j) ∧
  (2^{225} + 1) / (2^{15} + 1) = (finset.range(m + 1)).sum (λ i, 2^(b i)) ∧
  m = 8 :=
sorry

end unique_sequence_length_l736_736463


namespace problem_statement_l736_736281

variable (f : ℝ → ℝ) [Differentiable ℝ f]
variable (h : ∀ x : ℝ, deriv f x < f x)

theorem problem_statement : f 1 < Real.exp 1 * f 0 ∧ f 2014 < Real.exp 2014 * f 0 := 
sorry

end problem_statement_l736_736281


namespace speed_of_man_in_still_water_l736_736524

theorem speed_of_man_in_still_water :
  ∀ (v_m v_s : ℝ),
    (v_m + v_s = 9) ∧ (v_m - v_s = 5) → v_m = 7 :=
by
  intros v_m v_s h
  cases h with h1 h2
  sorry

end speed_of_man_in_still_water_l736_736524


namespace arithmetic_sequence_solution_geometric_sequence_solution_l736_736507

-- Problem 1: Arithmetic sequence
noncomputable def arithmetic_general_term (n : ℕ) : ℕ := 30 - 3 * n
noncomputable def arithmetic_sum_terms (n : ℕ) : ℝ := -1.5 * n^2 + 28.5 * n

theorem arithmetic_sequence_solution (n : ℕ) (a8 a10 : ℕ) (sequence : ℕ → ℝ) :
  a8 = 6 → a10 = 0 → (sequence n = arithmetic_general_term n) ∧ (sequence n = arithmetic_sum_terms n) ∧ (n = 9 ∨ n = 10) := 
sorry

-- Problem 2: Geometric sequence
noncomputable def geometric_general_term (n : ℕ) : ℝ := 2^(n-2)
noncomputable def geometric_sum_terms (n : ℕ) : ℝ := 2^(n-1) - 0.5

theorem geometric_sequence_solution (n : ℕ) (a1 a4 : ℝ) (sequence : ℕ → ℝ):
  a1 = 0.5 → a4 = 4 → (sequence n = geometric_general_term n) ∧ (sequence n = geometric_sum_terms n) := 
sorry

end arithmetic_sequence_solution_geometric_sequence_solution_l736_736507


namespace sum_conjugate_eq_two_l736_736741

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736741


namespace sum_of_solutions_eq_five_l736_736098

theorem sum_of_solutions_eq_five :
  let a := 2
  let b := -10
  let c := 3
  let equation := a*x^2 + b*x + c = 0
  ∃ r s : ℝ, (equation.HasRoot r ∧ equation.HasRoot s) ∧ r + s = 5 :=
begin
  sorry
end

end sum_of_solutions_eq_five_l736_736098


namespace rectangle_horizontal_length_l736_736383

variable (squareside rectheight : ℕ)

-- Condition: side of the square is 80 cm, vertical side length of the rectangle is 100 cm
def square_side_length := 80
def rect_vertical_length := 100

-- Question: Calculate the horizontal length of the rectangle
theorem rectangle_horizontal_length :
  (4 * square_side_length) = (2 * rect_vertical_length + 2 * rect_horizontal_length) -> rect_horizontal_length = 60 := by
  sorry

end rectangle_horizontal_length_l736_736383


namespace product_of_real_roots_eq_one_l736_736047

theorem product_of_real_roots_eq_one :
  (∏ x in ({y : ℝ | y ^ Real.log y = 10} : Set ℝ), y) = 1 :=
sorry

end product_of_real_roots_eq_one_l736_736047


namespace even_painted_faces_smaller_blocks_l736_736980

-- Definition of the original block
structure Block :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)
  (painted_faces : ℕ) -- to represent faces painted

-- The original block definition
def original_block : Block :=
  { length := 6, width := 6, height := 2, painted_faces := 6 }

-- Definition of the smaller block
structure SmallBlock :=
  (side : ℕ)
  (painted_faces : ℕ) -- to represent faces painted

-- The smaller block has side 2 inches
def small_block : SmallBlock :=
  { side := 2, painted_faces := 0 }

theorem even_painted_faces_smaller_blocks : 
  (∀ b : SmallBlock, b.painted_faces % 2 = 0) → 14 := 
by 
  sorry

end even_painted_faces_smaller_blocks_l736_736980


namespace repeating_decimals_addition_l736_736939

noncomputable def repeating_decimal_to_fraction (r : ℕ) : ℚ :=
  r / 999

def lean_problem_statement : ℚ :=
  repeating_decimal_to_fraction 567 + repeating_decimal_to_fraction 345 - repeating_decimal_to_fraction 234

theorem repeating_decimals_addition : lean_problem_statement = 226 / 333 := 
  sorry

end repeating_decimals_addition_l736_736939


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736817

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736817


namespace compare_powers_l736_736267

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l736_736267


namespace intersect_at_0_intersect_at_180_intersect_at_90_l736_736474

-- Define radii R and r, and the distance c
variables {R r c : ℝ}

-- Formalize the conditions and corresponding angles
theorem intersect_at_0 (h : c = R - r) : True := 
sorry

theorem intersect_at_180 (h : c = R + r) : True := 
sorry

theorem intersect_at_90 (h : c = Real.sqrt (R^2 + r^2)) : True := 
sorry

end intersect_at_0_intersect_at_180_intersect_at_90_l736_736474


namespace even_number_combinations_l736_736150

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736150


namespace correlation_statements_l736_736883

variables {x y : ℝ}
variables (r : ℝ) (h1 : r > 0) (h2 : r = 1) (h3 : r = -1)

theorem correlation_statements :
  (r > 0 → (∀ x y, x > 0 → y > 0)) ∧
  (r = 1 ∨ r = -1 → (∀ x y, ∃ m b : ℝ, y = m * x + b)) :=
sorry

end correlation_statements_l736_736883


namespace limit_is_minus_three_l736_736120

noncomputable def sequence_limit : ℝ := 
  lim (fun n : ℕ => 
    (n : ℝ) * (n : ℝ)^(1/5) - (27 * (n : ℝ)^6 + (n : ℝ)^2)^(1/3)) / 
    ((n : ℝ) + (n : ℝ)^(1/4) * (9 + (n : ℝ)^2)^(1/2))

theorem limit_is_minus_three :
  sequence_limit = -3 :=
by
  sorry

end limit_is_minus_three_l736_736120


namespace linear_function_no_second_quadrant_l736_736023

theorem linear_function_no_second_quadrant (x y : ℝ) (h : y = 2 * x - 3) :
  ¬ ((x < 0) ∧ (y > 0)) :=
by {
  sorry
}

end linear_function_no_second_quadrant_l736_736023


namespace team_division_cooperative_property_l736_736464

/-
Theorem: There exists a way to divide 24 students into 4 teams of 6
such that for any given division method, either exactly three teams are cooperative
or exactly one team is cooperative, and both situations will occur.
-/
theorem team_division_cooperative_property :
  ∃ division : fin 6 → fin 4 → Prop, 
  (∀ division_set : fin 4 → fin 6, 
    (∃ U V W X : fin 6 → fin 6, 
    (∀ i : fin 24, (division_set divs $ i) == division_set (U (divs) $ i) == division_set (V (divs) $ i) == division_set (W (divs) $ i) == division_set (X (divs) $ i)))-/
 sorry
 
end team_division_cooperative_property_l736_736464


namespace fibonacci_gcd_l736_736648

-- Definition of Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

-- Prove that gcd of F_{2017} and (F_{99} * F_{101} + 1) is 1
theorem fibonacci_gcd : gcd (fibonacci 2017) (fibonacci 99 * fibonacci 101 + 1) = 1 := 
sorry

end fibonacci_gcd_l736_736648


namespace number_of_pines_l736_736074

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l736_736074


namespace even_combinations_after_six_operations_l736_736156

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736156


namespace tan_theta_eq_neg_one_max_magnitude_sum_vecs_l736_736672

variables {θ : ℝ}

def vector_a : ℝ × ℝ := (Real.sin θ, 1)
def vector_b : ℝ × ℝ := (1, Real.cos θ)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem tan_theta_eq_neg_one (h : perp vector_a vector_b) : Real.tan θ = -1 := by
  sorry

theorem max_magnitude_sum_vecs : 
  ∀ θ : ℝ, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 → 
  ∃ max_val, max_val = 1 + Real.sqrt 2 ∧ 
    ∀ x, x = (Real.sqrt((Real.sin θ + 1)^2 + (1 + Real.cos θ)^2)) → x ≤ max_val := by
  sorry

end tan_theta_eq_neg_one_max_magnitude_sum_vecs_l736_736672


namespace hilary_corn_shucking_l736_736680

theorem hilary_corn_shucking : 
    (total_ears : ℕ) (total_stalks : ℕ) (half_ears_kernels : ℕ) (other_half_ears_kernels : ℕ) 
    (ears_per_stalk : ℕ) (stalks : ℕ) 
    (h1 : ears_per_stalk = 4) 
    (h2 : stalks = 108) 
    (h3 : half_ears_kernels = 500) 
    (h4 : other_half_ears_kernels = 600) : 
    let total_ears := stalks * ears_per_stalk
    let half_ears := total_ears / 2 in
    total_ears * half_ears_kernels / 2 + total_ears * other_half_ears_kernels / 2 = 237600 :=
by 
    intros
    rw [h1, h2, h3, h4]
    sorry

end hilary_corn_shucking_l736_736680


namespace cube_side_length_l736_736042

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l736_736042


namespace flowers_count_l736_736379

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l736_736379


namespace log27_3_eq_one_third_l736_736237

theorem log27_3_eq_one_third : ∃ x : ℝ, (27 = 3^3) ∧ (27^x = 3) ∧ (x = 1/3) :=
by {
  have h1: 27 = 3^3 := by norm_num,
  have h2: 27^(1/3) = 3 := by {
    rw [h1, ← real.rpow_mul (real.rpow_pos_of_pos (by norm_num) (3: ℝ))],
    norm_num
  },
  exact ⟨1/3, h1, h2⟩
}

end log27_3_eq_one_third_l736_736237


namespace log_base_27_3_l736_736239

-- Define the condition
lemma log_base_condition : 27 = 3^3 := rfl

-- State the theorem to be proven
theorem log_base_27_3 : log 27 3 = 1 / 3 :=
by 
  -- skip the proof for now
  sorry

end log_base_27_3_l736_736239


namespace dust_storm_acres_l736_736520

def total_acres : ℕ := 64013
def untouched_acres : ℕ := 522
def dust_storm_covered : ℕ := total_acres - untouched_acres

theorem dust_storm_acres :
  dust_storm_covered = 63491 := by
  sorry

end dust_storm_acres_l736_736520


namespace white_tile_count_l736_736358

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l736_736358


namespace increasing_function_range_a_l736_736652

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x - 3 * a
  else Real.log x / Real.log a

theorem increasing_function_range_a :
  ∀ (a : ℝ), (∀ x y : ℝ, x < y → f a x ≤ f a y) → 1 < a ∧ a < 2 :=
begin
  assume a h_increasing,
  split,
  {
    sorry, -- Proof for the condition 1 < a
  },
  {
    sorry, -- Proof for the condition a < 2
  }
end

end increasing_function_range_a_l736_736652


namespace opposite_of_neg_three_l736_736435

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l736_736435


namespace total_deck_expense_is_correct_l736_736473

variable (deck_cost : ℕ)
variable (victor_decks : ℕ)
variable (friend_decks : ℕ)
variable (total_spent : ℕ)

theorem total_deck_expense_is_correct :
  deck_cost = 8 →
  victor_decks = 6 →
  friend_decks = 2 →
  total_spent = (victor_decks * deck_cost) + (friend_decks * deck_cost) →
  total_spent = 64 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end total_deck_expense_is_correct_l736_736473


namespace necessary_sufficient_AD_AE_AP_AQ_l736_736345

-- Define necessary components of the geometric setup
structure Triangle := (A B C : Point)
structure Circle := (O : Point) (circumference : Set Point)
structure Line := (P Q : Point)

variables {A B C D E F G P Q : Point}
variables (circumcircle : Circle)
variables (l : Line) (EF : Line)

/- Define conditions -/
def is_tangent (l : Line) (A : Point) (circumcircle : Circle) : Prop := 
  -- tangency condition here
  sorry

def point_on_extension (E D A : Point) : Prop :=
  -- E is on the extension of DA
  sorry

def point_on_minor_arc (F B C : Point) (circumcircle : Circle) : Prop := 
  -- F is on the minor arc BC of the circumcircle
  sorry

def line_intersects_arc (EF : Line) (arc : Set Point) (G : Point) : Prop := 
  -- EF intersects minor arc AB at G
  sorry

def lines_intersect (l : Line) (FB GC : Line) (P Q : Point) : Prop :=
  -- FB and GC intersect line l at points P and Q respectively
  sorry

/- Main statement -/
theorem necessary_sufficient_AD_AE_AP_AQ
  (acute_triangle : Triangle)
  (circumcircle : Circle)
  (tangent : is_tangent l A circumcircle)
  (D_on_BC : D ≠ A ∧ ∃ B C, Line.mk B C = Line.mk B C)
  (E_extension : point_on_extension E D A)
  (F_minor_arc : point_on_minor_arc F B C circumcircle)
  (G_arc_intersect : line_intersects_arc EF {A | True} G)
  (PQ_intersect : lines_intersect l (Line.mk F B) (Line.mk G C) P Q)
  : AP = AQ ↔ AD = AE :=
sorry

end necessary_sufficient_AD_AE_AP_AQ_l736_736345


namespace monthly_charge_for_motel_l736_736207

/-- The monthly charge for the motel is $1000 given the conditions detailed. -/
theorem monthly_charge_for_motel : 
  let weekly_charge := 280 in
  let weeks_per_month := 4 in
  let months_staying := 3 in
  let savings_when_monthly := 360 in
  let total_weeks_charges := monthly_charge * weeks_per_month * months_staying in
  let monthly_charge := (total_weeks_charges - savings_when_monthly) / months_staying in
  monthly_charge = 1000 :=
by
  sorry

end monthly_charge_for_motel_l736_736207


namespace simplify_expression_l736_736851

variables {a b c : ℝ}
variable (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
variable (h₃ : b - 2 / c ≠ 0)

theorem simplify_expression :
  (a - 2 / b) / (b - 2 / c) = c / b :=
sorry

end simplify_expression_l736_736851


namespace bob_mean_score_l736_736847

-- Conditions
def scores : List ℝ := [68, 72, 76, 80, 85, 90]
def alice_scores (a1 a2 a3 : ℝ) : Prop := a1 < a2 ∧ a2 < a3 ∧ a1 + a2 + a3 = 225
def bob_scores (b1 b2 b3 : ℝ) : Prop := b1 + b2 + b3 = 246

-- Theorem statement proving Bob's mean score
theorem bob_mean_score (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1 ∈ scores) (h2 : a2 ∈ scores) (h3 : a3 ∈ scores)
  (h4 : b1 ∈ scores) (h5 : b2 ∈ scores) (h6 : b3 ∈ scores)
  (h7 : alice_scores a1 a2 a3)
  (h8 : bob_scores b1 b2 b3)
  (h9 : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3)
  : (b1 + b2 + b3) / 3 = 82 :=
sorry

end bob_mean_score_l736_736847


namespace line_AC_equation_l736_736327

/-- Given a triangle ABC with vertices A(-1, 5), B(0, -1), and the equation of the angle bisector 
of ∠C being x + y - 2 = 0, prove that the equation of the line AC is 3x + 4y - 17 = 0. -/
theorem line_AC_equation :
  let A := (-1 : ℝ, 5 : ℝ)
  let B := (0 : ℝ, -1 : ℝ)
  let bisector_C := {p : ℝ × ℝ | p.1 + p.2 - 2 = 0}
  line_eqn A B bisector_C = "3x + 4y - 17 = 0" :=
sorry

end line_AC_equation_l736_736327


namespace sum_of_triples_l736_736805

theorem sum_of_triples : 
  ∀ (x y z : ℂ), 
    (x + y * z = 9) ∧ 
    (y + x * z = 13) ∧ 
    (z + x * y = 13) → 
    let S := { (x', y', z') : ℂ × ℂ × ℂ | 
                x' + y' * z' = 9 ∧ 
                y' + x' * z' = 13 ∧ 
                z' + x' * y' = 13 } in 
    (Σ t : S, t.1.1) = 9 := 
sorry

end sum_of_triples_l736_736805


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736825

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736825


namespace points_lie_on_same_sphere_l736_736972

-- Given definitions
variables (S A B C A1 B1 C1 A2 B2 C2 : Type*)
-- Conditions
def sphere_omega_passes_through_S_and_intersects_edges_at_points : Prop :=
  ∃ S A B C A1 B1 C1 : Type*, 
  (intersects (ω) V (SA, A1) ∧ intersects (ω) V (SB, B1) ∧ intersects (ω) V (SC, C1))

def sphere_Omega_circumscribed_and_intersects_omega_in_plane_parallel_to_ABC : Prop :=
  ∃ Σ (circum (Ω, (S, A, B, C))), intersects (Ω) (circum ((SABC), λ ω (plane_parallel (ABC))))

def points_symmetric_to_midpoints_of_edges : Prop :=
  ∀(M_X: Type*), midpoint A1 A2 = midpoint M_X ∧ midpoint B1 B2 = midpoint M_X ∧ midpoint C1 C2 = midpoint M_X

-- Proof statement
theorem points_lie_on_same_sphere :
  sphere_omega_passes_through_S_and_intersects_edges_at_points S A B C A1 B1 C1 →
  sphere_Omega_circumscribed_and_intersects_omega_in_plane_parallel_to_ABC S A B C A1 B1 C1 A2 B2 C2 →
  points_symmetric_to_midpoints_of_edges A B C A1 B1 C1 A2 B2 C2 →
  lie_on_same_sphere {A, B, C, A2, B2, C2} :=
by 
  sorry

end points_lie_on_same_sphere_l736_736972


namespace pencil_rows_l736_736247

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 35) (h2 : pencils_per_row = 5) : (total_pencils / pencils_per_row) = 7 :=
by
  sorry

end pencil_rows_l736_736247


namespace find_m_for_chord_of_length_l736_736616

def line (m : ℝ) := { p : ℝ × ℝ | p.1 - p.2 + m = 0 }

def circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

theorem find_m_for_chord_of_length :
  ∃ m : ℝ, (∀ p : ℝ × ℝ, p ∈ line m ∧ p ∈ circle → 
           ∃ q : ℝ × ℝ, q ∈ line m ∧ q ∈ circle ∧
           (abs ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 12)) →
           (m = sqrt 2 ∨ m = -sqrt 2) :=
sorry

end find_m_for_chord_of_length_l736_736616


namespace cost_per_sq_foot_insulation_l736_736186

-- Dimensions of the tank
def tank_length : ℝ := 4
def tank_width : ℝ := 5
def tank_height : ℝ := 2

-- Total cost to cover the surface
def total_cost : ℝ := 1520

-- Surface area of the tank
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Cost per square foot of insulation
def cost_per_sq_foot (total_cost surface_area : ℝ) : ℝ := total_cost / surface_area

-- The theorem to prove
theorem cost_per_sq_foot_insulation :
  cost_per_sq_foot total_cost (surface_area tank_length tank_width tank_height) = 20 := by
  sorry

end cost_per_sq_foot_insulation_l736_736186


namespace part_a_part_b_l736_736830

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l736_736830


namespace minimum_shots_to_destroy_tank_l736_736955

theorem minimum_shots_to_destroy_tank :
  let n := 41 in
  let cells := n*n in
  let white_cells := (n*n + 1) / 2 in
  let black_cells := n*n - white_cells in
  (2 * black_cells + white_cells) = 3 * cells - 1 → (3 * cells - 1) / 2 = 2521 :=
by
  -- definitions:
  let n := 41
  let cells := n * n
  let white_cells := (cells + 1) / 2
  let black_cells := cells - white_cells
  -- calculation:
  have hx : (2 * black_cells + white_cells) = 3 * cells - 1 := sorry
  have hy : (3 * cells - 1) / 2 = 2521 := sorry
  exact hy


end minimum_shots_to_destroy_tank_l736_736955


namespace length_KL_l736_736779

-- Define the triangle ABC, with K and L as midpoints of CB and CA respectively
def Triangle (α β γ : Type _) := 
  (A B C : α)
  (K : β)
  (L : γ)

variable {α β γ : Type _} [MetricSpace α] [MetricSpace β] [MetricSpace γ]

-- Define the conditions given in the problem
def is_midpoint (x y z : α) (m : β) : Prop := dist x m = dist y m 
def perimeter_qABKL (AB BK KL LA : ℝ) : ℝ := AB + BK + KL + LA
def perimeter_tKLC (CK KL LC : ℝ) : ℝ := CK + KL + LC

-- Length of side KL is to be determined
theorem length_KL {A B C K L : α} 
  (hK_mid : is_midpoint C B K)
  (hL_mid : is_midpoint A C L)
  (h_perim_qABKL : perimeter_qABKL (dist A B) (dist B K) (dist K L) (dist L A) = 10)
  (h_perim_tKLC : perimeter_tKLC (dist C K) (dist K L) (dist L C) = 6) :
  dist K L = 2 := 
sorry

end length_KL_l736_736779


namespace sum_series_1_to_60_l736_736565

-- Define what it means to be the sum of the first n natural numbers
def sum_n (n : Nat) : Nat := n * (n + 1) / 2

theorem sum_series_1_to_60 : sum_n 60 = 1830 :=
by
  sorry

end sum_series_1_to_60_l736_736565


namespace sum_of_middle_elements_l736_736212

theorem sum_of_middle_elements (n : ℕ) (h : n ≥ 3) : 
  let S (n : ℕ) := ∑ b in Finset.range (n-2+1), b * (b + 1) * (n - b)
  in S n = (n-2)*(n-1)*n*(n+1) / 12 :=
by
  sorry

end sum_of_middle_elements_l736_736212


namespace tickets_to_buy_l736_736110

theorem tickets_to_buy
  (ferris_wheel_cost : Float := 2.0)
  (roller_coaster_cost : Float := 7.0)
  (multiple_rides_discount : Float := 1.0)
  (newspaper_coupon : Float := 1.0) :
  (ferris_wheel_cost + roller_coaster_cost - multiple_rides_discount - newspaper_coupon = 7.0) :=
by
  sorry

end tickets_to_buy_l736_736110


namespace ice_cream_cones_sixth_day_l736_736562

theorem ice_cream_cones_sixth_day (cones_day1 cones_day2 cones_day3 cones_day4 cones_day5 cones_day7 : ℝ)
  (mean : ℝ) (h1 : cones_day1 = 100) (h2 : cones_day2 = 92) 
  (h3 : cones_day3 = 109) (h4 : cones_day4 = 96) 
  (h5 : cones_day5 = 103) (h7 : cones_day7 = 105) 
  (h_mean : mean = 100.1) : 
  ∃ cones_day6 : ℝ, cones_day6 = 95.7 :=
by 
  sorry

end ice_cream_cones_sixth_day_l736_736562


namespace intersection_of_sets_l736_736300

def SetA : Set ℝ := { x | |x| ≤ 1 }
def SetB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_of_sets : (SetA ∩ SetB) = { x | 0 ≤ x ∧ x ≤ 1 } := 
by
  sorry

end intersection_of_sets_l736_736300


namespace number_of_correct_statements_l736_736284

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then exp x * (x + 1) else -exp (-x) * (-x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_two_zeros (f : ℝ → ℝ) : Prop :=
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧ ∀ x₁ x₂ x₃, f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 → x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃

def solution_set (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x > 0 ↔ (x > -1 ∧ x < 0) ∨ x > 1)

def difference_bound (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| < 2

theorem number_of_correct_statements :
  is_odd_function f ∧ ¬has_two_zeros f ∧ solution_set f ∧ difference_bound f → 2 = 2 := 
by
  sorry

end number_of_correct_statements_l736_736284


namespace even_combinations_after_six_operations_l736_736158

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736158


namespace length_of_diagonal_AC_l736_736776

variable {Point : Type}
variables {A B C D E F : Point}
variables {AD BC AB CD AC : ℝ}

-- Conditions given in the problem
variables (h_parallel_AD_BC : AD ∥ BC) (h_equal_AB_CD : AB = CD)
          (h_BC : BC = 9) (h_AD : AD = 21) (h_perimeter : AB + BC + CD + AD = 50)

-- To prove
theorem length_of_diagonal_AC (h_congruent_triangles : ∀ (E F : Point), BE ⊥ AD ∧ CF ⊥ AD ∧ AB = CD → AC = 17) : AC = 17 :=
by
  sorry

end length_of_diagonal_AC_l736_736776


namespace question1_question2_l736_736656

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Problem 1: Prove the valid solution of x when f(x) = 3 and x ∈ [0, 4]
theorem question1 (h₀ : 0 ≤ 3) (h₁ : 4 ≥ 3) : 
  ∃ (x : ℝ), (f x = 3 ∧ 0 ≤ x ∧ x ≤ 4) → x = 3 :=
by
  sorry

-- Problem 2: Prove the range of f(x) when x ∈ [0, 4]
theorem question2 : 
  ∃ (a b : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 4 → a ≤ f x ∧ f x ≤ b) → a = -1 ∧ b = 8 :=
by
  sorry

end question1_question2_l736_736656


namespace value_of_b_minus_a_l736_736744

open Real

def condition (a b : ℝ) : Prop := 
  abs a = 3 ∧ abs b = 2 ∧ a + b > 0

theorem value_of_b_minus_a (a b : ℝ) (h : condition a b) :
  b - a = -1 ∨ b - a = -5 :=
  sorry

end value_of_b_minus_a_l736_736744


namespace even_combinations_486_l736_736139

def operation := 
  | inc2  -- increase by 2
  | inc3  -- increase by 3
  | mul2  -- multiply by 2

def apply_operation (n : ℕ) (op : operation) : ℕ :=
  match op with
  | operation.inc2 => n + 2
  | operation.inc3 => n + 3
  | operation.mul2 => n * 2

def apply_operations (n : ℕ) (ops : List operation) : ℕ :=
  List.foldl apply_operation n ops

theorem even_combinations_486 : 
  let initial_n := 1
  let possible_operations := [operation.inc2, operation.inc3, operation.mul2]
  let all_combinations := List.replicate 6 possible_operations -- List of length 6 with all possible operations
  let even_count := all_combinations.filter (fun ops => (apply_operations initial_n ops % 2 = 0)).length
  even_count = 486 := by
    sorry

end even_combinations_486_l736_736139


namespace length_of_side_in_triangle_l736_736549

theorem length_of_side_in_triangle :
  ∀ (x: ℝ),
  (sin 30 = 1/2) →
  (sin 60 = √3 / 2) →
  ∃ t : triangle ℝ, 
  (t.angleA = 30) ∧ (t.angleB = 60) ∧ (t.sideA = 2) ∧ (t.sideB = x) →
  x = 2 * √3 :=
begin
  sorry
end

end length_of_side_in_triangle_l736_736549


namespace average_speed_trip_l736_736222

theorem average_speed_trip :
  let distance_1 := 65
  let distance_2 := 45
  let distance_3 := 55
  let distance_4 := 70
  let distance_5 := 60
  let total_time := 5
  let total_distance := distance_1 + distance_2 + distance_3 + distance_4 + distance_5
  let average_speed := total_distance / total_time
  average_speed = 59 :=
by
  sorry

end average_speed_trip_l736_736222


namespace find_other_sides_of_triangle_l736_736624

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end find_other_sides_of_triangle_l736_736624


namespace initial_percentage_concentrated_kola_9_l736_736127

/-- 
Given:
  - A 340-liter solution of kola is made from 64% water, some percent concentrated kola, and the rest is made from sugar.
  - 3.2 liters of sugar, 8 liters of water, and 6.8 liters of concentrated kola were added to the solution.
  - After the addition, 26.536312849162012% of the solution is made from sugar.
  Prove that the initial percentage of concentrated kola in the solution was 9%.
-/
theorem initial_percentage_concentrated_kola_9 
  (total_initial_volume : ℝ)
  (percent_water_initial : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_percent_sugar : ℝ) :
  total_initial_volume = 340 ∧
  percent_water_initial = 0.64 ∧
  added_sugar = 3.2 ∧
  added_water = 8 ∧
  added_kola = 6.8 ∧
  final_percent_sugar = 0.26536312849162012 →
  let initial_volume_sugar := (100 - 64 - C) * 340 / 100
      initial_volume_concentrated_kola := C * 340 / 100 in
  C = 9 :=
by
  intros h
  sorry

end initial_percentage_concentrated_kola_9_l736_736127


namespace no_primes_divisible_by_60_l736_736686

theorem no_primes_divisible_by_60 (p : ℕ) (prime_p : Nat.Prime p) : ¬ (60 ∣ p) :=
by
  sorry

end no_primes_divisible_by_60_l736_736686


namespace sin_pi_minus_alpha_eq_3_over_5_l736_736286

theorem sin_pi_minus_alpha_eq_3_over_5 
  (x y r α : ℝ) 
  (h_point_on_terminal_side : x = -4 ∧ y = 3)
  (h_r : r = Real.sqrt (x^2 + y^2))
  (h_sin_alpha : real.sin α = y / r) :
  real.sin (π - α) = 3 / 5 :=
by
  have h_x : x = -4 := h_point_on_terminal_side.1
  have h_y : y = 3 := h_point_on_terminal_side.2
  have h_r_val : r = real.sqrt (x^2 + y^2) := h_r
  have h_r_val_calc : r = 5 := by
    -- Using the Pythagorean theorem to demonstrate r = 5
    sorry
  have h_sin_alpha_val : real.sin α = 3 / 5 := h_sin_alpha
  -- Using the property of sine function sin(π - α) = sin(α)
  show real.sin (π - α) = 3 / 5 from
    by
    rw <-h_sin_alpha_val
    -- since sin(π - α) = sin(α)
    sorry

end sin_pi_minus_alpha_eq_3_over_5_l736_736286


namespace L_shaped_region_perimeter_l736_736048

-- Definitions for the L-shaped configuration and conditions
def area_of_L_shaped_region : ℝ := 392
def number_of_squares : ℕ := 8
def squares_in_top_row : ℕ := 3
def squares_in_bottom_row : ℕ := 5
def side_length_of_square (area_of_square : ℝ) : ℝ := Real.sqrt area_of_square

-- Area condition
def area_of_each_square : ℝ := area_of_L_shaped_region / number_of_squares

-- Side length derived from the area of each square
def side_length (area : ℝ) : ℝ := side_length_of_square area

-- Perimeter calculation based on given configuration
def perimeter_of_L_shaped_region (side_len : ℝ) : ℝ :=
  (3 * side_len) + (5 * side_len) + (2 * side_len) + side_len + (2 * side_len)

-- Proof statement: Given the conditions, the perimeter is 91 cm
theorem L_shaped_region_perimeter : perimeter_of_L_shaped_region (side_length area_of_each_square) = 91 :=
by sorry

end L_shaped_region_perimeter_l736_736048


namespace sum_of_four_digit_integers_up_to_4999_l736_736097

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999_l736_736097


namespace infinite_divisors_l736_736000

theorem infinite_divisors : ∃ᶠ n in at_top, n ∈ ℕ ∧ n ∣ 2^n + 2 := sorry

end infinite_divisors_l736_736000


namespace num_digits_difference_l736_736915

-- Define the two base-10 integers
def n1 : ℕ := 150
def n2 : ℕ := 950

-- Find the number of digits in the base-2 representation of these numbers.
def num_digits_base2 (n : ℕ) : ℕ :=
  Nat.log2 n + 1

-- State the theorem
theorem num_digits_difference :
  num_digits_base2 n2 - num_digits_base2 n1 = 2 :=
by
  sorry

end num_digits_difference_l736_736915


namespace equivalence_condition_l736_736121

def fractional_part (z : ℝ) : ℝ := z - floor z

def system_equivalence (a : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 = a ∧ x + y = 0) ↔ (x^2 + y^2 = a ∧ fractional_part (x + y) = 0)

theorem equivalence_condition (a : ℝ) : system_equivalence a → a < 1 / 2 :=
sorry

end equivalence_condition_l736_736121


namespace arrange_in_descending_order_l736_736271

theorem arrange_in_descending_order (a b c : ℝ) (h₁ : a = log 6 (0.2)) (h₂ : b = 6^(0.2)) (h₃ : c = 0.2^6) :
  b > c ∧ c > a := 
by
  sorry

end arrange_in_descending_order_l736_736271


namespace sum_conjugate_eq_two_l736_736738

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736738


namespace potatoes_weight_150kg_l736_736754

theorem potatoes_weight_150kg (
  conversion_factor : ℝ := 0.4536
  ) :
  (150 / conversion_factor).round = 331 :=
by
  sorry

end potatoes_weight_150kg_l736_736754


namespace even_number_combinations_l736_736152

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736152


namespace leap_day_2040_is_friday_l736_736365

def leap_day_day_of_week (start_year : ℕ) (start_day : ℕ) (end_year : ℕ) : ℕ :=
  let num_years := end_year - start_year
  let num_leap_years := (num_years + 4) / 4 -- number of leap years including start and end year
  let total_days := 365 * (num_years - num_leap_years) + 366 * num_leap_years
  let day_of_week := (total_days % 7 + start_day) % 7
  day_of_week

theorem leap_day_2040_is_friday :
  leap_day_day_of_week 2008 5 2040 = 5 := 
  sorry

end leap_day_2040_is_friday_l736_736365


namespace friction_coefficient_l736_736940

open Real

/-- Constants of the problem -/
def angle := 30 * (π / 180)
def g := 9.8 -- gravitational constant

/-- Given conditions on accelerations -/
def a_up := -2
def a_down := -2 / 3

/-- Sine and cosine values for the given angle 30 degrees -/
def sin_30 := 1 / 2
def cos_30 := sqrt 3 / 2

theorem friction_coefficient (μ : ℝ) : 
  (-g * sin_30 - μ * g * cos_30 = a_up) ∧ (g * sin_30 - μ * g * cos_30 = a_down) → μ = 0.29 := sorry

end friction_coefficient_l736_736940


namespace parrots_per_cage_l736_736965

theorem parrots_per_cage (P : ℕ) (parakeets_per_cage : ℕ) (cages : ℕ) (total_birds : ℕ) 
    (h1 : parakeets_per_cage = 7) (h2 : cages = 8) (h3 : total_birds = 72) 
    (h4 : total_birds = cages * P + cages * parakeets_per_cage) : 
    P = 2 :=
by
  sorry

end parrots_per_cage_l736_736965


namespace card_distribution_l736_736389

theorem card_distribution (n : ℕ) (h : n > 0) : 
  (∑ k in Finset.range (n+1), Nat.choose n k) - 2 = 2 * (2^(n-1) - 1) :=
by
  sorry

end card_distribution_l736_736389


namespace A_B_distance_l736_736469

noncomputable def distance_between_A_and_B 
  (vA: ℕ) (vB: ℕ) (vA_after_return: ℕ) 
  (meet_distance: ℕ) : ℚ := sorry

theorem A_B_distance (distance: ℚ) 
  (hA: vA = 40) (hB: vB = 60) 
  (hA_after_return: vA_after_return = 60) 
  (hmeet: meet_distance = 50) : 
  distance_between_A_and_B vA vB vA_after_return meet_distance = 1000 / 7 := sorry

end A_B_distance_l736_736469


namespace chain_of_inequalities_l736_736403

theorem chain_of_inequalities (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  9 / (a + b + c) ≤ (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ∧ 
  (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ≤ (1 / a + 1 / b + 1 / c) := 
by 
  sorry

end chain_of_inequalities_l736_736403


namespace ratio_of_areas_l736_736257

variables {A B D Y : Type} [point A] [point B] [point D] [point Y]

-- Distances between points
variable (BD AD : ℕ)
-- Set the values based on problem conditions
axiom h_BD : BD = 35
axiom h_AD : AD = 40

-- Given that DY bisects angle ADB
axiom angle_bisector : bisects_angle D Y A B

-- Prove the ratio of areas
theorem ratio_of_areas : area_ratio (triangle B D Y) (triangle A D Y) = 7 / 8 :=
by
  -- skip the proof
  sorry

end ratio_of_areas_l736_736257


namespace shirt_final_price_including_taxes_l736_736539

noncomputable def final_price_percentage (P0 : ℝ) : ℝ :=
  let P1 := P0 * 0.5
  let P2 := P1 * 0.9
  let P3 := P2 * 0.8
  let Pf := P3 * 1.08
  (Pf / P0) * 100
  
theorem shirt_final_price_including_taxes (P0 : ℝ) : final_price_percentage P0 = 38.88 :=
by
  have step1 : P1 = P0 * 0.5 := rfl
  have step2 : P2 = P1 * 0.9 := rfl
  have step3 : P3 = P2 * 0.8 := rfl
  have final : Pf = P3 * 1.08 := rfl
  sorry

end shirt_final_price_including_taxes_l736_736539


namespace britney_has_more_chickens_l736_736411

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end britney_has_more_chickens_l736_736411


namespace terror_permutations_count_l736_736584

noncomputable def num_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  Nat.factorial n / counts.foldl (λ (acc : ℕ) (x : ℕ), acc * Nat.factorial x) 1

theorem terror_permutations_count : num_permutations 6 [2, 2] = 180 := by
  sorry

end terror_permutations_count_l736_736584


namespace cube_side_length_l736_736038

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l736_736038


namespace product_b1_bn_l736_736051

variable {b : ℕ → ℝ} (n : ℕ)

def Sn (n : ℕ) : ℝ := ∑ i in Finset.range n, b i
def Sn_reciprocal (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / b i

axiom condition1 : Sn n = (1 / 6) * Sn_reciprocal n

theorem product_b1_bn (n : ℕ) : b 0 * b (n - 1) = 1 / 6 :=
by
  -- proof placeholder
  sorry

end product_b1_bn_l736_736051


namespace cube_side_length_l736_736041

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l736_736041


namespace fred_baseball_cards_l736_736609

variable (initial_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_baseball_cards (h1 : initial_cards = 5) (h2 : bought_cards = 3) : initial_cards - bought_cards = 2 := by
  sorry

end fred_baseball_cards_l736_736609


namespace angle_FCG_eq_67_l736_736385

open_locale affine

noncomputable def point := ℝ × ℝ

noncomputable def circle (O : point) (r : ℝ) := { P : point | sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = r }

variables {A B C D E F G : point}
variables (O : point) (r : ℝ) [h_circle : circle O r]

-- Conditions
variable (diameter_AE : A.1 - E.1 = 0 ∧ sqrt((A.1 - O.1)^2 + (A.2 - O.2)^2) = r ∧ sqrt((E.1 - O.1)^2 + (E.2 - O.2)^2) = r)
variable (angle_ABF_81 : ∠A B F = 81)
variable (angle_EDG_76 : ∠E D G = 76)

-- Theorem statement
theorem angle_FCG_eq_67 : ∠F C G = 67 :=
sorry

end angle_FCG_eq_67_l736_736385


namespace opposite_of_neg3_is_3_l736_736439

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l736_736439


namespace problem_statement_l736_736634

-- Define the function and its properties
variable (f : ℝ → ℝ)
variable (h_even : ∀ x ∈ set.Icc (-5 : ℝ) 5, f x = f (-x))
variable (h_cond : f 3 > f 1)

-- The goal
theorem problem_statement : f (-1) < f 3 :=
sorry

end problem_statement_l736_736634


namespace hilary_corn_shucking_l736_736681

theorem hilary_corn_shucking : 
    (total_ears : ℕ) (total_stalks : ℕ) (half_ears_kernels : ℕ) (other_half_ears_kernels : ℕ) 
    (ears_per_stalk : ℕ) (stalks : ℕ) 
    (h1 : ears_per_stalk = 4) 
    (h2 : stalks = 108) 
    (h3 : half_ears_kernels = 500) 
    (h4 : other_half_ears_kernels = 600) : 
    let total_ears := stalks * ears_per_stalk
    let half_ears := total_ears / 2 in
    total_ears * half_ears_kernels / 2 + total_ears * other_half_ears_kernels / 2 = 237600 :=
by 
    intros
    rw [h1, h2, h3, h4]
    sorry

end hilary_corn_shucking_l736_736681


namespace bacteria_population_growth_l736_736045

noncomputable def final_population (initial_population : ℕ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_population * growth_rate^time

theorem bacteria_population_growth :
  final_population 1000 2 8.965784284662087 ≈ 495033 :=
by
  sorry

end bacteria_population_growth_l736_736045


namespace cube_side_length_l736_736032

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l736_736032


namespace sequence_odd_l736_736793

def sequence (n : Nat) : Int :=
  match n with
  | 0 => 2
  | 1 => 7
  | _ => sequence (n - 1) * 3 - sequence (n - 2) * 2

theorem sequence_odd (n : Nat) (h : n ≥ 2) : sequence n % 2 = 1 := 
by
  sorry

end sequence_odd_l736_736793


namespace movie_ticket_cost_l736_736886

theorem movie_ticket_cost :
  ∃ (x : ℝ),
    let popcorn_cost := 2 * 1.5,
        milk_tea_cost := 3 * 3,
        total_contribution := 3 * 11,
        total_cost := 3 * x + popcorn_cost + milk_tea_cost in
    total_cost = total_contribution ∧ x = 7 :=
by
  use 7
  let popcorn_cost := 2 * 1.5
  let milk_tea_cost := 3 * 3
  let total_contribution := 3 * 11
  let total_cost := 3 * 7 + popcorn_cost + milk_tea_cost
  have : popcorn_cost = 3 := by norm_num
  have : milk_tea_cost = 9 := by norm_num
  have : total_cost = 33 := by norm_num
  have : total_contribution = 33 := by norm_num
  exact ⟨this, rfl⟩

end movie_ticket_cost_l736_736886


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736816

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736816


namespace z_conjugate_sum_l736_736698

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736698


namespace inequality_f_l736_736794

variable (x y : ℝ)

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem inequality_f (x y : ℝ) :
  (f x - f y ≥ 0) ↔ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  intro h
  sorry

end inequality_f_l736_736794


namespace inequality_solution_l736_736612

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then 
    {x : ℝ | 1 < x}
  else if 0 < a ∧ a < 2 then 
    {x : ℝ | 1 < x ∧ x < (2 / a)}
  else if a = 2 then 
    ∅
  else if a > 2 then 
    {x : ℝ | (2 / a) < x ∧ x < 1}
  else 
    {x : ℝ | x < (2 / a)} ∪ {x : ℝ | 1 < x}

theorem inequality_solution (a : ℝ) :
  ∀ x : ℝ, (ax^2 - (a + 2) * x + 2 < 0) ↔ (x ∈ solve_inequality a) :=
sorry

end inequality_solution_l736_736612


namespace chocolate_bar_cost_l736_736589

theorem chocolate_bar_cost (num_bars : ℕ) (sold_bars : ℕ) (total_amount : ℕ) 
  (h₀ : num_bars = 13) (h₁ : num_bars - 6 = sold_bars) (h₂ : sold_bars = 7) (h₃ : total_amount = 42) :
  (total_amount / sold_bars : ℕ) = 6 :=
by
  rw [←h₂, h₃]
  norm_num
  sorry

end chocolate_bar_cost_l736_736589


namespace number_of_even_results_l736_736137

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736137


namespace range_of_c_l736_736751

theorem range_of_c (c : ℝ) :
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + c ≥ 0) ↔ c ∈ set.Ici (real.sqrt 2 - 1) := 
sorry

end range_of_c_l736_736751


namespace cube_side_length_l736_736039

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l736_736039


namespace exists_subset_with_three_colors_used_l736_736344

noncomputable def lean_proof_statement : Type :=
  sorry

theorem exists_subset_with_three_colors_used
  (V : Type) [Fintype V] [DecidableEq V]
  (E : V → V → Prop) (color : V → V → ℕ)
  (h_complete : ∀ (u v : V), u ≠ v → E u v)
  (h_colors : ∀ (u v : V), color u v ∈ {1, 2, 3, 4})
  (h_at_least_one_color : ∀ c ∈ {1, 2, 3, 4}, ∃ (u v : V), color u v = c ∧ E u v)
  (h_card_V : Fintype.card V = 2004) :
  ∃ (u v w : V), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
    (∃ c1 c2 c3 ∈ {1, 2, 3, 4}, 
      {color u v, color u w, color v w} = {c1, c2, c3}) :=
by
  sorry

end exists_subset_with_three_colors_used_l736_736344


namespace digit_sum_eq_four_l736_736053

open Nat

theorem digit_sum_eq_four (n : ℕ) (h : 0 < n) : 
  digit_sum ((10 ^ (4 * n ^ 2 + 8) + 1) ^ 2) = 4 := 
by sorry

end digit_sum_eq_four_l736_736053


namespace sum_conjugate_eq_two_l736_736734

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736734


namespace log_base_27_of_3_l736_736228

theorem log_base_27_of_3 : log 3 (27) (3) = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736228


namespace arithmetic_sequence_sum_l736_736342

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ)

def S₁₀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ) : ℕ :=
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀

theorem arithmetic_sequence_sum (h : S₁₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ = 120) :
  a₁ + a₁₀ = 24 :=
by
  sorry

end arithmetic_sequence_sum_l736_736342


namespace Shane_current_age_44_l736_736892

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end Shane_current_age_44_l736_736892


namespace digit_difference_base2_150_950_l736_736913

def largest_power_of_2_lt (n : ℕ) : ℕ :=
  (List.range (n+1)).filter (λ k, 2^k ≤ n).last' getLastRange

def base2_digits (n : ℕ) : ℕ := largest_power_of_2_lt n + 1

theorem digit_difference_base2_150_950 :
  base2_digits 950 - base2_digits 150 = 2 :=
by {
  sorry
}

end digit_difference_base2_150_950_l736_736913


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736819

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736819


namespace sequence_explicit_formula_l736_736772

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end sequence_explicit_formula_l736_736772


namespace log_base_two_identity_l736_736491

theorem log_base_two_identity (x : ℝ) : log 2 (2^x) = x := by sorry

end log_base_two_identity_l736_736491


namespace dealership_sales_l736_736563

theorem dealership_sales (sports_cars sedans suvs : ℕ) (h_sc : sports_cars = 35)
  (h_ratio_sedans : 5 * sedans = 8 * sports_cars) 
  (h_ratio_suvs : 5 * suvs = 3 * sports_cars) : 
  sedans = 56 ∧ suvs = 21 := by
  sorry

#print dealership_sales

end dealership_sales_l736_736563


namespace linear_regression_intercept_l736_736055

open Real

def groups : List (ℝ × ℝ) := [(1, 2), (2, 4), (3, 4), (4, 7), (5, 8)]

def x_mean : ℝ := (1 + 2 + 3 + 4 + 5) / 5

def y_mean : ℝ := (2 + 4 + 4 + 7 + 8) / 5

theorem linear_regression_intercept :
  ∃  (a : ℝ), (∀ (x y : ℝ), (x, y) ∈ groups → y = 0.7 * x + a) ∧ a = 2.9 := by
  sorry

end linear_regression_intercept_l736_736055


namespace ticket_values_equiv_l736_736977

theorem ticket_values_equiv :
  let x : ℕ := 30 in
  (Nat.divisors x).length = 8 :=
by {
  let x : ℕ := Nat.gcd 90 150
  have h_gcd : x = 30 := by norm_num,
  rw h_gcd,
  show (Nat.divisors x).length = 8,
  sorry,
}

end ticket_values_equiv_l736_736977


namespace symmetric_rays_parallel_l736_736394

theorem symmetric_rays_parallel (A B C M : Point) (circumcircle : Circle)
  (h₁ : OnCircle M circumcircle)
  (h₂ : Triangle A B C)
  (h₃ : OnCircle A circumcircle)
  (h₄ : OnCircle B circumcircle)
  (h₅ : OnCircle C circumcircle)
  (angle_bisector_A : Line)
  (angle_bisector_B : Line)
  (angle_bisector_C : Line)
  (h₆ : AngleBisector angle_bisector_A (∠ BAC))
  (h₇ : AngleBisector angle_bisector_B (∠ ABC))
  (h₈ : AngleBisector angle_bisector_C (∠ ACB)) :
  (ReflectedRay AM angle_bisector_A ∥ ReflectedRay BM angle_bisector_B ∥ ReflectedRay CM angle_bisector_C) ↔
  OnCircle M circumcircle := sorry

end symmetric_rays_parallel_l736_736394


namespace train_passing_time_l736_736542

-- Define the conditions given in the problem
def train_length_m := 150 -- Length in meters
def train_speed_km_per_hr := 54 -- Speed in km/hr

-- Convert speed to m/s
def train_speed_m_per_s : ℝ := (train_speed_km_per_hr : ℝ) * 1000 / 3600

-- State the theorem to be proved
theorem train_passing_time : train_length_m / train_speed_m_per_s = 10 :=
by
  -- Explicitly specify the values and simplify the expression to verify the proof
  have h : train_speed_m_per_s = 15 := by
    unfold train_speed_m_per_s
    norm_num
  rw h
  norm_num
  sorry

end train_passing_time_l736_736542


namespace red_yellow_flowers_l736_736555

theorem red_yellow_flowers
  (total : ℕ)
  (yellow_white : ℕ)
  (red_white : ℕ)
  (extra_red_over_white : ℕ)
  (H1 : total = 44)
  (H2 : yellow_white = 13)
  (H3 : red_white = 14)
  (H4 : extra_red_over_white = 4) :
  ∃ (red_yellow : ℕ), red_yellow = 17 := by
  sorry

end red_yellow_flowers_l736_736555


namespace solve_for_x_l736_736278

theorem solve_for_x (x : ℝ) : 3 * 2^x + 2 * 2^(x + 1) = 2048 → x = 11 - Real.logb 2 7 :=
by
  sorry

end solve_for_x_l736_736278


namespace second_player_wins_l736_736900

-- Define the game conditions
def game_conditions (peasant : Nat) (move: Nat → Nat → Prop) : Prop :=
  ∀ (n : Nat), ¬move n n ∧ move n 0 → False

-- Define the win condition
def win_condition (N : Nat) : Prop :=
  ∀ (move: Nat → Nat → Prop), -- The move function relations
    (∀ (peasant: Nat), game_conditions peasant move) →
    (∃ second_player_win_strategy : (Nat → Nat → Nat),
      ∀ (move: Nat → Nat → Prop), strategy_correct second_player_win_strategy move N)

-- Declare the main theorem
theorem second_player_wins (N : Nat) : win_condition N :=
  sorry

end second_player_wins_l736_736900


namespace circumscribed_triangle_angle_l736_736333

theorem circumscribed_triangle_angle
  (O : Point)
  (A B C : Point)
  (h_circum : circumscribed_triangle O A B C)
  (h_BOC : angle B O C = 110)
  (h_AOB : angle A O B = 150) :
  angle A B C = 50 :=
begin
  sorry
end

end circumscribed_triangle_angle_l736_736333


namespace correct_answers_l736_736639

-- Given conditions
def parabola_E := { p : ℝ × ℝ // p.2 ^ 2 = 4 * p.1 }
def parabola_focus : ℝ × ℝ := (1, 0)
def circle_F (r : ℝ) (h : 0 < r ∧ r < 1) := { p : ℝ × ℝ // (p.1 - 1) ^ 2 + p.2 ^ 2 = r ^ 2 }
def line_l0 (t : ℝ) := { p : ℝ × ℝ // p.1 = t * p.2 + 1 }
def point_A (t : ℝ) := { p : ℝ × ℝ // p ∈ parabola_E ∧ p.1 = (line_l0 t).val.1 ∧ p.2 = (line_l0 t).val.2 }
def point_B (t : ℝ) := { p : ℝ × ℝ // p ∈ parabola_E ∧ p.1 = (line_l0 t).val.1 ∧ p.2 = (line_l0 t).val.2 }
def point_M (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def point_T : ℝ × ℝ := (0, 1)
def point_G (t : ℝ) := (0, -1 / t)

-- Questions to prove
theorem correct_answers (r : ℝ) (hr : 0 < r ∧ r < 1) (t : ℝ) (A B M G : ℝ × ℝ) :
  (A = point_A t) → (B = point_B t) → (M = point_M A B) → (G = point_G t) →
    (1 / (A.2) + 1 / (B.2) = 1 / (G.2)) ∧
    (∃ p : ℝ × ℝ, p ∈ parabola_E ∧ M = p) ∧
    (∀ O : ℝ × ℝ, ∇(O.1 - 1, O.2) * ∇(M.1, M.2) ≠ -1) ∧
    (∀ l0 : ℝ, ¬(r ∈ (1 / 2) ∧ (|A.1 - C.1|, |C.1 - D.1|, |D.1 - B.1|) forms_AR))
:=
sorry

end correct_answers_l736_736639


namespace number_of_books_l736_736811

theorem number_of_books (original_books new_books : ℕ) (h1 : original_books = 35) (h2 : new_books = 56) : 
  original_books + new_books = 91 :=
by {
  -- the proof will go here, but is not required for the statement
  sorry
}

end number_of_books_l736_736811


namespace sum_conjugate_eq_two_l736_736737

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736737


namespace smallest_positive_integer_l736_736092

theorem smallest_positive_integer (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 2)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 4) :
  a = 59 :=
sorry

end smallest_positive_integer_l736_736092


namespace probability_of_getting_exactly_9_heads_in_12_flips_of_biased_coin_l736_736904

noncomputable def probability_getting_heads := (3 : ℝ) / 4
noncomputable def probability_getting_tails := 1 - probability_getting_heads
noncomputable def num_flips := 12
noncomputable def num_heads := 9
noncomputable def binomial_coeff := (finset.range (num_flips - num_heads + 1)).prod (λ k, (num_flips - k)) / (finset.range (num_heads + 1)).prod id
noncomputable def prob_exact_heads := binomial_coeff * (probability_getting_heads ^ num_heads) * (probability_getting_tails ^ (num_flips - num_heads))

theorem probability_of_getting_exactly_9_heads_in_12_flips_of_biased_coin :
  prob_exact_heads = (4330260 : ℝ) / 16777216 := sorry

end probability_of_getting_exactly_9_heads_in_12_flips_of_biased_coin_l736_736904


namespace smallest_sum_of_squares_l736_736422

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 231) :
  x^2 + y^2 ≥ 281 :=
sorry

end smallest_sum_of_squares_l736_736422


namespace y_coord_third_vertex_eq_l736_736979

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def altitude_of_equilateral (s : ℝ) : ℝ :=
  (s * real.sqrt 3) / 2

theorem y_coord_third_vertex_eq (H : (0, 7) < p ∧ p < (10, 7)) :
  ∃ p : ℝ × ℝ, (p.1, p.2 + altitude_of_equilateral 10) = (0, 7) ∧ (10, 7) := 
sorry

end y_coord_third_vertex_eq_l736_736979


namespace nat_divisibility_l736_736086

theorem nat_divisibility {n : ℕ} : (n + 1 ∣ n^2 + 1) ↔ (n = 0 ∨ n = 1) := 
sorry

end nat_divisibility_l736_736086


namespace find_parallel_side_length_l736_736254

-- Define the given conditions
def is_parallel_side (x : ℝ) : Prop :=
  (1 / 2) * (x + 20) * 21 = 504

theorem find_parallel_side_length :
  ∃ x : ℝ, is_parallel_side x ∧ x = 28 :=
by
  existsi 28
  split
  · unfold is_parallel_side
    sorry
  · refl

#check find_parallel_side_length

end find_parallel_side_length_l736_736254


namespace logarithm_identity_l736_736405

theorem logarithm_identity :
  1 / (Real.log 3 / Real.log 8 + 1) + 
  1 / (Real.log 2 / Real.log 12 + 1) + 
  1 / (Real.log 4 / Real.log 9 + 1) = 3 := 
by
  sorry

end logarithm_identity_l736_736405


namespace slope_product_l736_736470

theorem slope_product (m n : ℝ) (θ₂ : ℝ) (h₁ : m = 3 * n) (h₂ : m = Math.tan (3 * θ₂)) (h₃ : n = Math.tan θ₂) :
  m * n = 9 / 4 :=
by
  -- Proof will go here
  sorry

end slope_product_l736_736470


namespace max_ab_bc_cd_da_l736_736869

theorem max_ab_bc_cd_da {a b c d : ℕ} (h : a ∈ {2, 3, 5, 6} ∧ b ∈ {2, 3, 5, 6} ∧ 
                              c ∈ {2, 3, 5, 6} ∧ d ∈ {2, 3, 5, 6} ∧ 
                              a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ab + bc + cd + da ≤ 64 :=
by {
  sorry
}

end max_ab_bc_cd_da_l736_736869


namespace z_conjugate_sum_l736_736707

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736707


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736815

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736815


namespace even_combinations_result_in_486_l736_736162

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736162


namespace prop1_false_prop2_true_l736_736636

def S : Set ℕ := {a | (∃ m n ∈ S, m ≠ n ∧ a = m + n) ∨ (∃ p q ∉ S, p ≠ q ∧ a = p + q)}

theorem prop1_false : ¬ (4 ∈ S) :=
sorry

theorem prop2_true : ∀ x, (∃ n : ℕ, x = 3 * n + 5) → x ∈ S :=
sorry

end prop1_false_prop2_true_l736_736636


namespace complex_number_transformation_l736_736012

theorem complex_number_transformation :
  let z := -4 * complex.I in
  let rotation := (1 / 2) + (real.sqrt 3) * complex.I / 2 in
  let dilation := 2 in
  let transformation := rotation * dilation in
  z * transformation = (4 * real.sqrt 3) - 4 * complex.I :=
by
  let z := -4 * complex.I
  let rotation := (1 / 2) + (real.sqrt 3) * complex.I / 2
  let dilation := 2
  let transformation := rotation * dilation
  calc
    z * transformation = -4 * complex.I * ((1 / 2) + (real.sqrt 3) * complex.I / 2) * 2 : by rw [mul_assoc]
       ... = -4 * complex.I * (1 + (real.sqrt 3) * complex.I) : by sorry
       ... = 4 * real.sqrt 3 - 4 * complex.I : by sorry

end complex_number_transformation_l736_736012


namespace largest_divisor_of_342_and_285_l736_736485

-- Define the problem's main entities
def divisors (n : ℕ) : set ℕ := { d | d ∣ n }

-- Define the largest common divisor function
def largest_common_divisor (a b : ℕ) : ℕ :=
  Finset.max' (Finset.filter (λ x, x ∣ b) (Finset.filter (λ x, x ∣ a) (Finset.range (a + 1)))) (by sorry)

-- The main theorem statement
theorem largest_divisor_of_342_and_285 :
  largest_common_divisor 342 285 = 57 :=
by sorry

end largest_divisor_of_342_and_285_l736_736485


namespace expand_simplify_expr_l736_736245

theorem expand_simplify_expr (x : ℤ) : 
  (1 + x^3) * (1 - x^4) * (1 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 - x^9 - x^{12} := 
by
  sorry

end expand_simplify_expr_l736_736245


namespace flower_bed_area_l736_736468

theorem flower_bed_area (r : ℝ) (w : ℝ) (h : w = 28) (r_val : r = 7) : 
  ∃ A : ℝ, abs (A - 49.67) < 0.01 ∧ 
  let θ := (360 / Real.pi : ℝ) in
  let arc_len := w - (2 * r) in
  let C := 2 * Real.pi * r in
  arc_len = (θ / 360 * C) ∧
  A = (θ / 360) * Real.pi * r^2 :=
sorry

end flower_bed_area_l736_736468


namespace total_chairs_l736_736949

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l736_736949


namespace number_of_valid_3_digit_integers_l736_736683

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def is_valid_triplet (a b c : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ (a * b * c = 30)

theorem number_of_valid_3_digit_integers : 
  (card { (a, b, c) : ℕ × ℕ × ℕ | 100 ≤ a * 100 + b * 10 + c ∧ is_valid_triplet a b c}) = 12 :=
by
  sorry

end number_of_valid_3_digit_integers_l736_736683


namespace number_of_cows_l736_736334

variable (x y z : ℕ)

theorem number_of_cows (h1 : 4 * x + 2 * y + 2 * z = 24 + 2 * (x + y + z)) (h2 : z = y / 2) : x = 12 := 
sorry

end number_of_cows_l736_736334


namespace complement_intersection_l736_736303

section SetTheory

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 3}) (hB : B = {2, 3, 4}) : 
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end SetTheory

end complement_intersection_l736_736303


namespace grasshopper_cannot_return_to_start_l736_736171

theorem grasshopper_cannot_return_to_start :
  let jumps := (finset.range 1986).sum id  -- Sum of jumps from 1 to 1985
  jumps % 2 ≠ 0 :=     -- Prove that sum is odd
by
  sorry

end grasshopper_cannot_return_to_start_l736_736171


namespace difference_non_negative_l736_736856

theorem difference_non_negative (x : ℝ) (h : x - 8 ≥ 0) : True :=
begin
  exact True.intro
end

end difference_non_negative_l736_736856


namespace required_folders_l736_736532

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_count : ℕ := 24
def total_cost : ℝ := 30

theorem required_folders : ∃ (folders : ℕ), folders = 20 ∧ 
  (pencil_count * pencil_cost + folders * folder_cost = total_cost) :=
sorry

end required_folders_l736_736532


namespace livestock_min_count_l736_736522

/-- A livestock trader bought some horses at $344 each and some oxen at $265 each. 
The total cost of all the horses was $33 more than the total cost of all the oxen. 
Prove that the minimum number of horses and oxen he could have bought under these conditions 
are x = 36 and y = 25. -/
theorem livestock_min_count 
    (x y: ℤ) (horses_cost oxen_cost : ℤ) (price_diff : ℤ)
    (h_horses_cost : horses_cost = 344) (h_oxen_cost : oxen_cost = 265) (h_price_diff : price_diff = 33) 
    (h_eq: horses_cost * x = oxen_cost * y + price_diff): 
    (x = 36) ∧ (y = 25) :=
by
    sorry

end livestock_min_count_l736_736522


namespace sum_of_roots_is_zero_l736_736641

variable {R : Type*} [LinearOrderedField R]

-- Define the function f : R -> R and its properties
variable (f : R → R)
variable (even_f : ∀ x, f x = f (-x))
variable (roots_f : Finset R)
variable (roots_f_four : roots_f.card = 4)
variable (roots_f_set : ∀ x, x ∈ roots_f → f x = 0)

theorem sum_of_roots_is_zero : (roots_f.sum id) = 0 := 
sorry

end sum_of_roots_is_zero_l736_736641


namespace natural_number_property_l736_736596

theorem natural_number_property (N k : ℕ) (hk : k > 0)
    (h1 : 10^(k-1) ≤ N) (h2 : N < 10^k) (h3 : N * 10^(k-1) ≤ N^2) (h4 : N^2 ≤ N * 10^k) :
    N = 10^(k-1) := 
sorry

end natural_number_property_l736_736596


namespace arrange_descending_order_l736_736633

noncomputable def a := (3 / 5) ^ (-1 / 3)
noncomputable def b := (4 / 3) ^ (-1 / 2)
noncomputable def c := Real.log (3 / 5)

theorem arrange_descending_order : a > b ∧ b > c := by
  sorry

end arrange_descending_order_l736_736633


namespace max_size_of_valid_subset_A_l736_736627

-- Problem conditions
variables (m n : ℕ) (hm : 2 ≤ m) (hn : 3 ≤ n)

-- Define the set S
def S := { p : ℕ × ℕ | p.1 ∈ (fin m).to_set ∧ p.2 ∈ (fin n).to_set }

-- Define subset A with the given condition
def valid_subset_A (A : Set (ℕ × ℕ)) : Prop :=
  A ⊆ S ∧ ∀ (x1 x2 y1 y2 y3 : ℕ), x1 < x2 → y1 < y2 → y2 < y3 → 
  (x1, y1) ∈ A → (x1, y2) ∈ A → (x1, y3) ∈ A → (x2, y2) ∉ A

-- Prove that the maximum size of a valid subset A is 2m + n - 2
theorem max_size_of_valid_subset_A (A : Set (ℕ × ℕ)) 
  (hA : valid_subset_A m n A) : A.card ≤ 2 * m + n - 2 := sorry

end max_size_of_valid_subset_A_l736_736627


namespace exists_four_digit_number_divisible_by_101_l736_736352

theorem exists_four_digit_number_divisible_by_101 :
  ∃ (a b c d : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    b ≠ c ∧ b ≠ d ∧
    c ≠ d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) % 101 = 0 := 
by
  -- To be proven
  sorry

end exists_four_digit_number_divisible_by_101_l736_736352


namespace total_marbles_l736_736517

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l736_736517


namespace arrival_probability_l736_736579

-- Definitions for arrival times and conditions
variables {t1 t2 t3 : ℝ} -- The arrival times
variables (ht1 : t1 < t2) -- Given condition: t1 < t2

-- Assume arrival times are uniformly random and independent
-- Note: Uniform distribution over a session duration can be conceptualized but is not explicitly defined here

noncomputable def prob_t1_lt_t3_given_t1_lt_t2 : ℝ := 
real.to_nnreal (2 / 3)

-- Theorem statement: prove the probability calculation
theorem arrival_probability : prob_t1_lt_t3_given_t1_lt_t2 = 2 / 3 := sorry

end arrival_probability_l736_736579


namespace sufficient_condition_for_x_square_l736_736453

theorem sufficient_condition_for_x_square (a : ℝ) (h : a ≥ 5) : ∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - a ≤ 0 :=
by
  intros x hx
  rw [set.mem_Icc] at hx
  cases hx with h1 h2
  have h3 : x^2 ≤ 4 :=
    by 
      have hx_max : (2:ℝ) = 2 := rfl
      apply pow_le_pow_of_le_left zero_le_one h2 hx_max,
  linarith [h, h3]

end sufficient_condition_for_x_square_l736_736453


namespace bankers_discount_is_correct_l736_736458

-- Definitions for the problem conditions
def FV : ℝ := 74500
def TD : ℝ := 11175
def R : ℝ := 15
def T : ℝ := 1

-- Expected banker's discount
def expected_BD : ℝ := 74500 * 15 / 100

-- Statement to prove
theorem bankers_discount_is_correct : (FV * R * T) / 100 = expected_BD := by
  sorry

end bankers_discount_is_correct_l736_736458


namespace equal_elements_l736_736454

theorem equal_elements (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h_perm : ∃ (σ : Equiv.Perm (Fin 2011)), ∀ i, x' i = x (σ i))
  (h_eq : ∀ i : Fin 2011, x i + x ((i + 1) % 2011) = 2 * x' i) :
  ∀ i j : Fin 2011, x i = x j :=
by
  sorry

end equal_elements_l736_736454


namespace simplify_expression_l736_736412

variable (b : ℝ)

theorem simplify_expression :
  (2 * b + 6 - 5 * b) / 2 = -3 / 2 * b + 3 :=
sorry

end simplify_expression_l736_736412


namespace z_conjugate_sum_l736_736695

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 :=
sorry

end z_conjugate_sum_l736_736695


namespace A_walking_speed_l736_736196

theorem A_walking_speed (v : ℝ) (t : ℝ) (distance_A distance_B : ℝ) :
  (distance_B = 20 * t) →
  (distance_A = v * (t + 3)) →
  (distance_A = distance_B) →
  (v * (t + 3) = 60) →
  v = 10 :=
by
  intros h1 h2 h3 h4
  rw [← h3] at h2
  sorry

end A_walking_speed_l736_736196


namespace nested_fraction_value_l736_736567

theorem nested_fraction_value : 
  let expr := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))
  expr = 21 / 55 :=
by 
  sorry

end nested_fraction_value_l736_736567


namespace joint_purchases_popular_in_countries_joint_purchases_not_popular_among_neighbours_l736_736938

theorem joint_purchases_popular_in_countries 
    (risks : Prop) 
    (cost_savings : Prop) 
    (info_sharing : Prop)
    (quality_assessment : Prop)
    (willingness_to_share : Prop)
    : (cost_savings ∧ info_sharing) → risks → ∀ country, practice_of_joint_purchases_popular country :=
by
  intros h1 h2 country
  -- Proof required here.
  sorry

theorem joint_purchases_not_popular_among_neighbours 
    (transactional_costs : Prop) 
    (coordination_challenges : Prop) 
    (necessary_compensation : Prop)
    (proximity_to_stores : Prop)
    (disputes : Prop)
    : (transactional_costs ∧ coordination_challenges ∧ necessary_compensation ∧ proximity_to_stores ∧ disputes) 
    → ∀ (neighbours : Type), ¬ practice_of_joint_purchases_popular_for groceries neighbours :=
by
  intros h1 neighbours
  -- Proof required here.
  sorry

end joint_purchases_popular_in_countries_joint_purchases_not_popular_among_neighbours_l736_736938


namespace log_base_27_3_l736_736241

-- Define the condition
lemma log_base_condition : 27 = 3^3 := rfl

-- State the theorem to be proven
theorem log_base_27_3 : log 27 3 = 1 / 3 :=
by 
  -- skip the proof for now
  sorry

end log_base_27_3_l736_736241


namespace correct_answers_l736_736638

-- Given conditions
def parabola_E := { p : ℝ × ℝ // p.2 ^ 2 = 4 * p.1 }
def parabola_focus : ℝ × ℝ := (1, 0)
def circle_F (r : ℝ) (h : 0 < r ∧ r < 1) := { p : ℝ × ℝ // (p.1 - 1) ^ 2 + p.2 ^ 2 = r ^ 2 }
def line_l0 (t : ℝ) := { p : ℝ × ℝ // p.1 = t * p.2 + 1 }
def point_A (t : ℝ) := { p : ℝ × ℝ // p ∈ parabola_E ∧ p.1 = (line_l0 t).val.1 ∧ p.2 = (line_l0 t).val.2 }
def point_B (t : ℝ) := { p : ℝ × ℝ // p ∈ parabola_E ∧ p.1 = (line_l0 t).val.1 ∧ p.2 = (line_l0 t).val.2 }
def point_M (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def point_T : ℝ × ℝ := (0, 1)
def point_G (t : ℝ) := (0, -1 / t)

-- Questions to prove
theorem correct_answers (r : ℝ) (hr : 0 < r ∧ r < 1) (t : ℝ) (A B M G : ℝ × ℝ) :
  (A = point_A t) → (B = point_B t) → (M = point_M A B) → (G = point_G t) →
    (1 / (A.2) + 1 / (B.2) = 1 / (G.2)) ∧
    (∃ p : ℝ × ℝ, p ∈ parabola_E ∧ M = p) ∧
    (∀ O : ℝ × ℝ, ∇(O.1 - 1, O.2) * ∇(M.1, M.2) ≠ -1) ∧
    (∀ l0 : ℝ, ¬(r ∈ (1 / 2) ∧ (|A.1 - C.1|, |C.1 - D.1|, |D.1 - B.1|) forms_AR))
:=
sorry

end correct_answers_l736_736638


namespace not_inequality_neg_l736_736314

theorem not_inequality_neg (x y : ℝ) (h : x > y) : ¬ (-x > -y) :=
by {
  sorry
}

end not_inequality_neg_l736_736314


namespace euclid1976_partb_problem2_l736_736849

theorem euclid1976_partb_problem2
  (x y : ℝ)
  (geo_prog : y^2 = 2 * x)
  (arith_prog : 2 / y = 1 / x + 9 / x^2) :
  x * y = 27 / 2 := by 
  sorry

end euclid1976_partb_problem2_l736_736849


namespace rational_roots_count_l736_736273

-- Define the polynomial with integer coefficients
def polynomial (b₄ b₃ b₂ b₁ : ℤ) : Polynomial ℤ :=
  16 * (Polynomial.X ^ 5) + b₄ * (Polynomial.X ^ 4) + b₃ * (Polynomial.X ^ 3) + b₂ * (Polynomial.X ^ 2) + b₁ * Polynomial.X + 24

-- The theorem to prove the number of different possible rational roots
theorem rational_roots_count (b₄ b₃ b₂ b₁ : ℤ) :
  ∃ roots : Finset ℚ, roots.card = 32 ∧
    ∀ root ∈ roots, IsRoot (polynomial b₄ b₃ b₂ b₁) root :=
by
  sorry

end rational_roots_count_l736_736273


namespace amount_cloth_on_second_day_l736_736765

-- Definitions based on conditions
def geometric_sequence (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a1 * q ^ n

-- Proven theorem statement
theorem amount_cloth_on_second_day 
  (a1 : ℝ) 
  (q : ℝ) 
  (a_sum : ℝ) 
  (h1 : q = 2) 
  (h2 : geom_sum : (5 : ℝ) = a1 * (1 - q ^ 5) / (1 - q)) :
  geometric_sequence a1 q 1 = 10 / 31 := 
by
  -- skip the proof for now
  sorry

end amount_cloth_on_second_day_l736_736765


namespace angle_in_third_quadrant_l736_736629

def quadrant_of_angle (θ : ℝ) : ℕ :=
if π < θ % (2 * π) ∧ θ % (2 * π) < 3 * π / 2 then 3 else 0

theorem angle_in_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : quadrant_of_angle θ = 3 :=
by {
    sorry
}

end angle_in_third_quadrant_l736_736629


namespace meeting_time_is_23_percent_l736_736808

/-- Makarla's work time in minutes -/
def total_work_time_min : ℕ := 10 * 60

/-- Time spent in meetings in minutes -/
def total_meeting_time_min : ℕ := 35 + 105

/-- Function to calculate the percentage of time spent in meetings -/
noncomputable def meeting_time_percentage : ℚ := (total_meeting_time_min.to_rat / total_work_time_min.to_rat) * 100

/-- Theorem to prove that Makarla spent 23% of her work day in meetings-/
theorem meeting_time_is_23_percent :
  meeting_time_percentage = 23 := by
  sorry

end meeting_time_is_23_percent_l736_736808


namespace closest_integer_to_sum_a_closest_integer_to_sum_b_closest_integer_to_sum_c_l736_736583

-- Part (a)
theorem closest_integer_to_sum_a : (⌊(19 / 15 + 19 / 3) + 0.5⌋ = 8) :=
by
  -- Define the problem
  let sum := (19 / 15 + 19 / 3)
  have h : (⌊sum + 0.5⌋ = 8) := sorry
  exact h

-- Part (b)
theorem closest_integer_to_sum_b : (⌊(85 / 42 + 43 / 21 + 29 / 14 + 15 / 7) + 0.5⌋ = 8) :=
by
  -- Define the problem
  let sum := (85 / 42 + 43 / 21 + 29 / 14 + 15 / 7)
  have h : (⌊sum + 0.5⌋ = 8) := sorry
  exact h

-- Part (c)
theorem closest_integer_to_sum_c : (⌊(-11 / 10 - 1 / 2 - 7 / 5 + 2 / 3) + 0.5⌋ = -2) :=
by
  -- Define the problem
  let sum := (-11 / 10 - 1 / 2 - 7 / 5 + 2 / 3)
  have h : (⌊sum + 0.5⌋ = -2) := sorry
  exact h

end closest_integer_to_sum_a_closest_integer_to_sum_b_closest_integer_to_sum_c_l736_736583


namespace number_of_pines_l736_736072

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l736_736072


namespace jordyn_total_cost_l736_736323

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end jordyn_total_cost_l736_736323


namespace seats_per_table_l736_736590

-- Definitions based on conditions
def tables := 4
def total_people := 32

-- Statement to prove
theorem seats_per_table : (total_people / tables) = 8 :=
by 
  sorry

end seats_per_table_l736_736590


namespace count_H_functions_l736_736170

def H_function (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

def f1 (x : ℝ) : ℝ := -x^2 + x + 1
def f2 (x : ℝ) : ℝ := 3 * x - 2 * (sin x - cos x)
def f3 (x : ℝ) : ℝ := exp x + 1
def f4 (x : ℝ) : ℝ := if x ≠ 0 then abs (log x) else 0

theorem count_H_functions : ([
    f1,
    f2,
    f3,
    f4
  ].filter H_function).length = 2 :=
sorry

end count_H_functions_l736_736170


namespace percent_increase_l736_736921

open Real

theorem percent_increase (p : ℝ) : 
  p > 0 → 
  (round (((p - 0.87 * p) / (0.87 * p)) * 100) : ℤ) = 15 := 
by
  intro h
  have h1 : ((p - 0.87 * p) / (0.87 * p)) * 100 = (13 / 87) * 100 := by
    calc
    ((p - 0.87 * p) / (0.87 * p)) * 100 = ((1 - 0.87) * p / (0.87 * p)) * 100 := by ring
    _ = ((0.13 * p) / (0.87 * p)) * 100 := by ring
    _ = (0.13 / 0.87) * 100 := by rw [mul_div_cancel_left _ h]
    _ = (13 / 87) * 100 := by norm_num
  rw h1
  norm_num
  exact rfl

end percent_increase_l736_736921


namespace number_of_pines_l736_736073

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l736_736073


namespace total_candles_used_l736_736007

def cakes_baked : ℕ := 8
def cakes_given_away : ℕ := 2
def remaining_cakes : ℕ := cakes_baked - cakes_given_away
def candles_per_cake : ℕ := 6

theorem total_candles_used : remaining_cakes * candles_per_cake = 36 :=
by
  -- proof omitted
  sorry

end total_candles_used_l736_736007


namespace jelly_beans_percentage_l736_736174

noncomputable def initial_beans : ℕ := 100
noncomputable def initial_red_beans : ℕ := 54
noncomputable def initial_green_beans : ℕ := 30
noncomputable def initial_blue_beans : ℕ := 16

def removed_beans (x : ℕ) : ℕ := x
def beans_remaining (x : ℕ) : ℕ := initial_beans - 2 * x
def blue_beans_remaining : ℕ := initial_blue_beans
def percent_blue_beans (x : ℕ) : ℕ := 20

theorem jelly_beans_percentage (x : ℕ) (h1 : 0.2 * (initial_beans - 2 * x) = blue_beans_remaining) : 
    let red_beans_now := initial_red_beans - removed_beans x in
    let total_beans_now := beans_remaining x in
    100 * red_beans_now / total_beans_now = 55 :=
sorry

end jelly_beans_percentage_l736_736174


namespace eval_expression_l736_736501

theorem eval_expression : 1999^2 - 1998 * 2002 = -3991 := 
by
  sorry

end eval_expression_l736_736501


namespace team_not_losing_probability_l736_736013

theorem team_not_losing_probability
  (p_center_forward : ℝ) (p_winger : ℝ) (p_attacking_midfielder : ℝ)
  (rate_center_forward : ℝ) (rate_winger : ℝ) (rate_attacking_midfielder : ℝ)
  (h_center_forward : p_center_forward = 0.2) (h_winger : p_winger = 0.5) (h_attacking_midfielder : p_attacking_midfielder = 0.3)
  (h_rate_center_forward : rate_center_forward = 0.4) (h_rate_winger : rate_winger = 0.2) (h_rate_attacking_midfielder : rate_attacking_midfielder = 0.2) :
  (p_center_forward * (1 - rate_center_forward) + p_winger * (1 - rate_winger) + p_attacking_midfielder * (1 - rate_attacking_midfielder)) = 0.76 :=
by
  sorry

end team_not_losing_probability_l736_736013


namespace number_of_even_results_l736_736136

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736136


namespace investment_growth_theorem_l736_736891

variable (x : ℝ)

-- Defining the initial and final investments
def initial_investment : ℝ := 800
def final_investment : ℝ := 960

-- Defining the growth equation
def growth_equation (x : ℝ) : Prop := initial_investment * (1 + x) ^ 2 = final_investment

-- The theorem statement that needs to be proven
theorem investment_growth_theorem : growth_equation x := sorry

end investment_growth_theorem_l736_736891


namespace z_conjugate_sum_l736_736720

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736720


namespace area_AMP_is_correct_l736_736519

-- Define the coordinates for points A, M, and P
def pointA : ℝ × ℝ × ℝ := (0,0,0)
def pointM : ℝ × ℝ × ℝ := (1,0,0)
def pointP : ℝ × ℝ × ℝ := (1,1,2)

-- Define vectors AM and AP
def vecAM : ℝ × ℝ × ℝ := (1 - 0, 0 - 0, 0 - 0)
def vecAP : ℝ × ℝ × ℝ := (1 - 0, 1 - 0, 2 - 0)

-- Define cross product of two vectors
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2.2 * v.2.1 - u.2.1 * v.2.2, u.1 * v.2.2 - u.2.2 * v.1, u.2.1 * v.1 - u.1 * v.2.1)

-- Calculate cross product of vecAM and vecAP
def cross_prod_AM_AP : ℝ × ℝ × ℝ := cross_product vecAM vecAP

-- Calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2.1 ^ 2 + v.2.2 ^ 2)

-- Area of the triangle
noncomputable def area_triangle_AMP : ℝ :=
1 / 2 * magnitude cross_prod_AM_AP

-- The final theorem stating the area of triangle AMP
theorem area_AMP_is_correct : area_triangle_AMP = Real.sqrt 5 / 2 := 
by
  sorry

end area_AMP_is_correct_l736_736519


namespace ellipse_equation_max_area_triangle_l736_736768

open Real

def ellipse (a b : ℝ) (a_gt_b : a > b) (b_pos : b > 0) : Prop :=
  ∃ x y, x = 2 ∧ y = 1 ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1

def eccentricity (a b e : ℝ) : Prop :=
  e = (sqrt (a^2 - b^2)) / a

def line (m : ℝ) : ℝ × ℝ → Prop :=
  λ P, P.snd = (1 / 2) * P.fst + m

def distance_to_line (P : ℝ × ℝ) (m : ℝ) : ℝ :=
  abs(2 * m) / (sqrt 5)

def area_of_triangle (P A B : ℝ × ℝ) : ℝ :=
  let AB := sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) in
  let d := distance_to_line P (A.snd - (1/2) * A.fst) in
  1/2 * AB * d

theorem ellipse_equation (P : ℝ × ℝ) (e : ℝ) (a b : ℝ) (ha: a > b) (hb: b > 0)
  (he: e = sqrt(3) / 2) (hP: P = (2, 1)) :
  (∃ a b, ellipse a b ha hb ∧ eccentricity a b e) →
  (∃ x y, (x^2) / 8 + (y^2) / 2 = 1) :=
sorry

theorem max_area_triangle (P : ℝ × ℝ) (e : ℝ) (a b : ℝ) (m : ℝ)
  (ha: a > b) (hb: b > 0) (he: e = sqrt(3) / 2) (hP: P = (2, 1))
  (h_line: ∀ P : ℝ × ℝ, line m P) :
  ∃ S : ℝ, 
    (∀ A B, ∃ A B, (A ≠ B) ∧ 
                 let S := area_of_triangle P A B in
                 S ≤ 2) :=
sorry

end ellipse_equation_max_area_triangle_l736_736768


namespace highest_probability_event_l736_736986

variables {Ω : Type} [MeasurableSpace Ω] (P : MeasureTheory.ProbabilityMeasure Ω)

def EventA : Set Ω := {ω | True}  -- Anya waits for the bus for at least one minute
def EventB : Set Ω := {ω | True}  -- Anya waits for the bus for at least two minutes
def EventC : Set Ω := {ω | True}  -- Anya waits for the bus for at least five minutes

-- Given relationships: 
axiom inclusion_C_B : EventC ⊆ EventB
axiom inclusion_B_A : EventB ⊆ EventA

-- Prove that P(A) >= P(B) >= P(C) and that the event A has the highest probability
theorem highest_probability_event : 
  P (EventA) ≥ P (EventB) ∧ P (EventB) ≥ P (EventC) ∧ P (EventA) = max (P (EventA)) (max (P (EventB)) (P (EventC))) :=
by {
  sorry
}

end highest_probability_event_l736_736986


namespace problem_1_problem_2_l736_736266

def set_A := { y : ℝ | 2 < y ∧ y < 3 }
def set_B := { x : ℝ | x > 1 ∨ x < -1 }

theorem problem_1 : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

def set_C := { x : ℝ | x ∈ set_B ∧ ¬(x ∈ set_A) }

theorem problem_2 : set_C = { x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end problem_1_problem_2_l736_736266


namespace find_n_l736_736863

noncomputable def sequence_term (n : ℕ) : ℝ := 1 / (Real.sqrt (n + 1) + Real.sqrt n)

noncomputable def sequence_sum (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, sequence_term (i + 1))

theorem find_n (n : ℕ) (h : sequence_sum n = 9) : n = 99 :=
by 
  sorry

end find_n_l736_736863


namespace no_partition_convex_polygon_into_nonconvex_quadrilaterals_l736_736588

theorem no_partition_convex_polygon_into_nonconvex_quadrilaterals (P : set (set point)) 
  (hP : convex_polygon P) : ¬ (∃ quads : list (set point), (∀ q ∈ quads, non_convex_quadrilateral q) ∧ (⋃ q ∈ quads, q) = P) :=
sorry

end no_partition_convex_polygon_into_nonconvex_quadrilaterals_l736_736588


namespace number_of_even_results_l736_736132

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736132


namespace third_person_profit_share_l736_736260

noncomputable def investment_first : ℤ := 9000
noncomputable def investment_second : ℤ := investment_first + 2000
noncomputable def investment_third : ℤ := investment_second - 3000
noncomputable def investment_fourth : ℤ := 2 * investment_third
noncomputable def investment_fifth : ℤ := investment_fourth + 4000
noncomputable def total_investment : ℤ := investment_first + investment_second + investment_third + investment_fourth + investment_fifth

noncomputable def total_profit : ℤ := 25000
noncomputable def third_person_share : ℚ := (investment_third : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem third_person_profit_share :
  third_person_share = 3076.92 := sorry

end third_person_profit_share_l736_736260


namespace integer_solutions_of_equation_l736_736250

theorem integer_solutions_of_equation :
  {p : ℤ × ℤ // 7 * p.1 ^ 2 - 40 * p.1 * p.2 + 7 * p.2 ^ 2 = (|p.1 - p.2| + 2) ^ 3} =
  {⟨2, -2⟩, ⟨-2, 2⟩} :=
by
  sorry

end integer_solutions_of_equation_l736_736250


namespace skee_ball_tickets_l736_736494

-- Represent the given conditions as Lean definitions
def whack_a_mole_tickets : ℕ := 33
def candy_cost_per_piece : ℕ := 6
def candies_bought : ℕ := 7
def total_candy_tickets : ℕ := candies_bought * candy_cost_per_piece

-- Goal: Prove the number of tickets won playing 'skee ball'
theorem skee_ball_tickets (h : 42 = total_candy_tickets): whack_a_mole_tickets + 9 = total_candy_tickets :=
by {
  sorry
}

end skee_ball_tickets_l736_736494


namespace BoyleMcCrinkProof_l736_736287

noncomputable def BoyleMcCrinkTheorem : Prop :=
  ∃ n : ℕ, 
    (binomial_coeff n 2 * (-1) ^ 2) / (binomial_coeff n 4 * (-1) ^ 4) = (3 : ℝ) / 14 ∧
    (n = 10) ∧ 
    (∃ T : ℝ, 
      T = binomial_coeff 10 8 * (-1) ^ 8 ∧ 
      T = 45)

theorem BoyleMcCrinkProof : BoyleMcCrinkTheorem := by
  sorry

end BoyleMcCrinkProof_l736_736287


namespace segment_length_l736_736755

theorem segment_length (XY XZ YZ WT: ℝ) (hXY : XY = 13) (hXZ : XZ = 12) (hYZ : YZ = 5)
    (hCirc: Circle S is the smallest radius circle passing through Z
           and tangent to XY at its midpoint) :
    WT = 13 :=
  sorry

end segment_length_l736_736755


namespace yellow_prob_l736_736513

variable (red orange yellow : ℝ)

axiom red_prob : red = 0.25
axiom orange_prob : orange = 0.35
axiom total_prob : red + orange + yellow = 1

theorem yellow_prob : yellow = 1 - (red + orange) := by
  rw [red_prob, orange_prob]
  sorry

end yellow_prob_l736_736513


namespace derivative_of_f_l736_736421

def f (x : ℝ) : ℝ := x^3 / 3 + 1 / x

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : 
  (derivative f x) = x^2 - 1 / x^2 :=
by
  sorry

end derivative_of_f_l736_736421


namespace ratio_a_b_l736_736665

theorem ratio_a_b (a b c : ℝ) (h1 : a * (-1) ^ 2 + b * (-1) + c = 1) (h2 : a * 3 ^ 2 + b * 3 + c = 1) : 
  a / b = -2 :=
by 
  sorry

end ratio_a_b_l736_736665


namespace trig_identity_l736_736499

theorem trig_identity (α : ℝ) : 
  (sin (5 * α) - sin (6 * α) - sin (7 * α) + sin (8 * α)) = 
  -4 * sin (α / 2) * sin α * sin (13 * α / 2) :=
by
  sorry

end trig_identity_l736_736499


namespace parabola_line_intersect_at_one_point_l736_736870

noncomputable def c_value := 
  let y_parabola (x : ℝ) (c : ℝ) := x^2 + 2 * x + c + 1
  let y_line : ℝ := 1
  ∃ (x : ℝ), y_parabola x 1 = y_line ∧ ∀ z : ℝ, y_parabola z 1 = y_line → z = x

theorem parabola_line_intersect_at_one_point : c_value = 1 :=
sorry

end parabola_line_intersect_at_one_point_l736_736870


namespace maximize_sum_first_n_terms_l736_736276

-- Define the proof problem, using sigmoid type definition for the sequence and conditions
noncomputable def arithmetic_sequence (a₁ d : ℝ) : ℕ → ℝ :=
λ n, a₁ + n * d

-- State the problem statement in Lean
theorem maximize_sum_first_n_terms (a₁ d : ℝ)
  (h1 : abs (arithmetic_sequence a₁ d 2) = abs (arithmetic_sequence a₁ d 8))
  (h2 : d < 0) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, (1 ≤ m ∧ m ≠ 5 ∧ m ≠ 6) → 
          sum (seq (arithmetic_sequence a₁ d) m) < sum (seq (arithmetic_sequence a₁ d) n)) :=
by
  sorry

end maximize_sum_first_n_terms_l736_736276


namespace problem1_problem2_l736_736123
noncomputable theory

-- Problem 1
theorem problem1 : (-3)^2 + (π - 1 / 2)^0 - |(-4)| = 6 := 
  by sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a ≠ -1) : ((1 - 1 / (a + 1)) * (a^2 + 2 * a + 1) / a) = (a + 1) := 
  by sorry

end problem1_problem2_l736_736123


namespace log_base_27_of_3_l736_736233

theorem log_base_27_of_3 (h : 27 = 3 ^ 3) : log 27 3 = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736233


namespace nonneg_triple_inequality_l736_736792

theorem nonneg_triple_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) :=
by
  sorry

end nonneg_triple_inequality_l736_736792


namespace larger_segment_length_l736_736211

theorem larger_segment_length {a b c : ℝ} (h_tri : a = 20 ∧ b = 48 ∧ c = 52) (h_c : true) :
  ∃ x : ℝ, x = 52 - 7.69 ∧ x ≈ 44.31 :=
by sorry

end larger_segment_length_l736_736211


namespace star_polygon_internal_angles_sum_l736_736973

-- Define the core aspects of the problem using type defintions and axioms.
def n_star_polygon_total_internal_angle_sum (n : ℕ) : ℝ :=
  180 * (n - 4)

theorem star_polygon_internal_angles_sum (n : ℕ) (h : n ≥ 6) :
  n_star_polygon_total_internal_angle_sum n = 180 * (n - 4) :=
by
  -- This step would involve the formal proof using Lean
  sorry

end star_polygon_internal_angles_sum_l736_736973


namespace min_omega_bound_l736_736319

theorem min_omega_bound (ω : ℝ) (hω : ω > 0)
  (h : ∀ f : ℝ → ℝ, (∀ x, f x = sin (ω * π * x) ^ 2) →
    (∃ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1/2 ∧ 0 ≤ x2 ∧ x2 ≤ 1/2 ∧ 
    (f x1 = 1) ∧ (f x2 = 0) ∧ x1 ≠ x2 ∧ (∃ x3 x4 : ℝ, 0 ≤ x3 ∧ x3 ≤ 1/2 ∧ 0 ≤ x4 ∧ x4 ≤ 1/2 ∧ 
    (f x3 = 1) ∧ (f x4 = 0) ∧ x3 ≠ x4))) :
  ω ≥ 3 := 
sorry

end min_omega_bound_l736_736319


namespace percentage_of_number_l736_736526

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l736_736526


namespace finish_remaining_work_l736_736495

theorem finish_remaining_work (A B : ℕ) (work_remaining : ℚ) : 
  (A = 21) → (B = 15) → (work_remaining = 7) → 
  let b_rate := 1 / (B : ℚ)
  let work_done_by_b := b_rate * 10
  let a_rate := 1 / (A : ℚ)
  let remaining_work := 1 - work_done_by_b
  let time_a_to_finish := remaining_work / a_rate
  time_a_to_finish = work_remaining := by {
  intros,
  -- Proof goes here
  sorry
}

end finish_remaining_work_l736_736495


namespace lilly_can_buy_flowers_l736_736378

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end lilly_can_buy_flowers_l736_736378


namespace solution_set_of_inequality_l736_736879

variable (a b c : ℝ)

theorem solution_set_of_inequality 
  (h1 : a < 0)
  (h2 : b = a)
  (h3 : c = -2 * a)
  (h4 : ∀ x : ℝ, -2 < x ∧ x < 1 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, (x ≤ -1 / 2 ∨ x ≥ 1) ↔ cx^2 + ax + b ≥ 0 :=
sorry

end solution_set_of_inequality_l736_736879


namespace number_of_pines_possible_l736_736069

-- Definitions based on conditions in the problem
def total_trees : ℕ := 101
def at_least_one_between_poplars (poplars : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (poplars[i] - poplars[j]) > 1
def at_least_two_between_birches (birches : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (birches[i] - birches[j]) > 2
def at_least_three_between_pines (pines : List ℕ) : Prop :=
  ∀ i j, i ≠ j → abs (pines[i] - pines[j]) > 3

-- Proving the number of pines planted is either 25 or 26
theorem number_of_pines_possible (poplars birches pines : List ℕ)
  (h1 : length (poplars ++ birches ++ pines) = total_trees)
  (h2 : at_least_one_between_poplars poplars)
  (h3 : at_least_two_between_birches birches)
  (h4 : at_least_three_between_pines pines) :
  length pines = 25 ∨ length pines = 26 :=
sorry

end number_of_pines_possible_l736_736069


namespace sum_of_squares_mod_7_l736_736093

-- Define the series of squares
def sum_of_squares (n : ℕ) : ℕ := (∑ i in Finset.range (n + 1), i^2)

-- Main theorem statement
theorem sum_of_squares_mod_7 : (sum_of_squares 144) % 7 = 2 :=
by
  -- Adding a placeholder for the proof
  sorry

end sum_of_squares_mod_7_l736_736093


namespace total_chairs_l736_736948

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l736_736948


namespace number_of_pines_l736_736058

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l736_736058


namespace moles_of_tetrachloromethane_l736_736256

noncomputable def balanced_reaction (ch4 cl2 ccl4 hcl : ℕ) : Prop :=
  ch4 = 1 ∧ cl2 = 4 ∧ ccl4 = 1 ∧ hcl = 4

theorem moles_of_tetrachloromethane :
  ∀ ch4 cl2 ccl4 hcl : ℕ,
  (ch4 = 1) → (cl2 = 4) → (balanced_reaction ch4 cl2 ccl4 hcl) → ccl4 = 1 := 
begin
  intros,
  sorry
end

end moles_of_tetrachloromethane_l736_736256


namespace complex_conjugate_sum_l736_736716

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736716


namespace size_relationship_l736_736875

/-- Define the numbers a, b, and c -/
def a : ℝ := (-0.3)^0
def b : ℝ := 0.3^2
def c : ℝ := 2^0.3

/-- Theorem stating the size relationship -/
theorem size_relationship : b < a ∧ a < c :=
by
  -- Proof will be provided here
  sorry

end size_relationship_l736_736875


namespace recurring_decimal_to_fraction_l736_736477

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l736_736477


namespace harold_car_payment_l736_736310

variables (C : ℝ)

noncomputable def harold_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def groceries : ℝ := 50
noncomputable def remaining_after_retirement : ℝ := 1300

-- Harold's utility cost is half his car payment
noncomputable def utilities (C : ℝ) : ℝ := C / 2

-- Harold's total expenses.
noncomputable def total_expenses (C : ℝ) : ℝ := rent + C + utilities C + groceries

-- Proving that Harold’s car payment \(C\) can be calculated with the remaining money
theorem harold_car_payment : (2500 - total_expenses C = 1300) → (C = 300) :=
by 
  sorry

end harold_car_payment_l736_736310


namespace commission_is_31_25_percent_l736_736117

def saleswoman_commission (total_sale : ℝ) (threshold : ℝ) (rate1 rate2 : ℝ) : ℝ :=
  if total_sale <= threshold 
  then rate1 * total_sale 
  else rate1 * threshold + rate2 * (total_sale - threshold)

theorem commission_is_31_25_percent (total_sale : ℝ) (commission : ℝ) (percent_commission : ℝ) : 
  total_sale = 800 ∧ commission = saleswoman_commission total_sale 500 0.20 0.50 ∧ percent_commission = (commission / total_sale) * 100 → 
  percent_commission = 31.25 :=
by
  assume h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  sorry

end commission_is_31_25_percent_l736_736117


namespace largest_integer_value_x_l736_736090

theorem largest_integer_value_x : ∀ (x : ℤ), (5 - 4 * x > 17) → x ≤ -4 := sorry

end largest_integer_value_x_l736_736090


namespace shortest_part_is_15_l736_736943

namespace ProofProblem

def rope_length : ℕ := 60
def ratio_part1 : ℕ := 3
def ratio_part2 : ℕ := 4
def ratio_part3 : ℕ := 5

def total_parts := ratio_part1 + ratio_part2 + ratio_part3
def length_per_part := rope_length / total_parts
def shortest_part_length := ratio_part1 * length_per_part

theorem shortest_part_is_15 :
  shortest_part_length = 15 := by
  sorry

end ProofProblem

end shortest_part_is_15_l736_736943


namespace isosceles_triangle_sides_l736_736625

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l736_736625


namespace sum_evaluation_l736_736243

noncomputable def T : ℝ := ∑' k : ℕ, (2*k+1) / 5^(k+1)

theorem sum_evaluation : T = 5 / 16 := sorry

end sum_evaluation_l736_736243


namespace smallest_m_plus_n_l736_736796

theorem smallest_m_plus_n (m n : ℕ) (hmn : m > n) (hid : (2012^m : ℕ) % 1000 = (2012^n) % 1000) : m + n = 104 :=
sorry

end smallest_m_plus_n_l736_736796


namespace relation_among_abc_l736_736632

noncomputable def a := Classical.some (Real.exists_of_analytic_eq_zero (λ x, 2^x + x) (by sorry))
noncomputable def b := Classical.some (Real.exists_of_analytic_eq_zero (λ x, log 2 x = 2) (by sorry))
noncomputable def c := Classical.some (Real.exists_of_analytic_eq_zero (λ x, log (1/2) x = x) (by sorry))

theorem relation_among_abc
  (ha : ∀ x, 2^x + x = 0 → a = x)
  (hb : ∀ x, log (by sorry) x = x → b = x)
  (hc : ∀ x, log 2 x = 2 → c = x) :
  a < b ∧ b < c :=
by
  obtain ⟨a, ha⟩ := exists_real_root_of_eq_zero (by sorry)
  obtain ⟨b, hb⟩ := exists_real_root_of_eq_zero (by sorry)
  obtain ⟨c, hc⟩ := exists_real_root_of_eq_zero (by sorry)
  sorry

end relation_among_abc_l736_736632


namespace z_conjugate_sum_l736_736704

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736704


namespace log_decreasing_interval_l736_736867

noncomputable def decreasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := 
∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x ≥ f y
    
theorem log_decreasing_interval : 
  ∀ x : ℝ, -x^2 + x + 6 > 0 → (∃ a b : ℝ, a = 0.5 ∧ b = 3 ∧ decreasing_interval (λ x, Real.log (-x^2 + x + 6)) a b) :=
by
  sorry

end log_decreasing_interval_l736_736867


namespace sum_of_max_values_of_f_l736_736660

-- Define the function
def f (x : ℝ) : ℝ := exp x * (sin x - cos x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2015 * real.pi

-- Statement to prove
theorem sum_of_max_values_of_f :
  (∑ k in finset.range 1008, f ((2 * k + 1) * real.pi)) = 
  (exp real.pi * (1 - exp (2014 * real.pi))) / (1 - exp (2 * real.pi)) :=
sorry

end sum_of_max_values_of_f_l736_736660


namespace fraction_multiplication_l736_736087

theorem fraction_multiplication (x : ℚ) (h : x = 236 / 100) : x * 3 = 177 / 25 :=
by
  sorry

end fraction_multiplication_l736_736087


namespace white_tiles_count_l736_736359

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l736_736359


namespace polynomial_zero_pairs_l736_736597

theorem polynomial_zero_pairs (r s : ℝ) :
  (∀ x : ℝ, (x = 0 ∨ x = 0) ↔ x^2 - 2 * r * x + r = 0) ∧
  (∀ x : ℝ, (x = 0 ∨ x = 0 ∨ x = 0) ↔ 27 * x^3 - 27 * r * x^2 + s * x - r^6 = 0) → 
  (r, s) = (0, 0) ∨ (r, s) = (1, 9) :=
by
  sorry

end polynomial_zero_pairs_l736_736597


namespace twenty_four_is_75_percent_of_what_number_l736_736896

theorem twenty_four_is_75_percent_of_what_number :
  ∃ x : ℝ, 24 = (75 / 100) * x ∧ x = 32 :=
by {
  use 32,
  split,
  { norm_num },
  { norm_num }
} -- sorry

end twenty_four_is_75_percent_of_what_number_l736_736896


namespace z_conjugate_sum_l736_736725

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736725


namespace sequence_sum_2011_l736_736348

theorem sequence_sum_2011 : 
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n > 0 → a (n + 1) + a n = 1) → 
  (finset.sum (finset.range 2011) a) = 1007 := 
by
  sorry

end sequence_sum_2011_l736_736348


namespace flowers_in_each_basket_l736_736215

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end flowers_in_each_basket_l736_736215


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736821

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736821


namespace parabola_line_intersection_length_l736_736177

open Real

theorem parabola_line_intersection_length
  (F : Point ℝ := ⟨1, 0⟩)
  (A B : Point ℝ)
  (parabola : ∀ (x y : ℝ), y^2 = 4 * x)
  (line : ∀ (x y : ℝ), passes_through F x y)
  (h1 : parabola A.1 A.2)
  (h2 : parabola B.1 B.2)
  (h3 : line A.1 A.2)
  (h4 : line B.1 B.2)
  (h_sum : A.1 + B.1 = 3) :
  dist A B = 5 :=
sorry

end parabola_line_intersection_length_l736_736177


namespace even_combinations_result_in_486_l736_736164

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736164


namespace z_conjugate_sum_l736_736703

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736703


namespace cosine_angle_CA_CB_l736_736774

open ComplexConjugate

def point := ℝ × ℝ × ℝ

def vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def dot_product (v1 v2 : point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : point) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def cosine_angle (v1 v2 : point) : ℝ :=
  (dot_product v1 v2) / (magnitude v1 * magnitude v2)

def A : point := (1, 2, -1)
def B : point := (2, 0, 0)
def C : point := (0, 1, 3)

theorem cosine_angle_CA_CB :
  cosine_angle (vector C A) (vector C B) = 13 * Real.sqrt 7 / 42 := by
  sorry

end cosine_angle_CA_CB_l736_736774


namespace work_completed_in_11_days_l736_736113

theorem work_completed_in_11_days (W : ℝ) :
  let rate_a := W / 24
      rate_b := W / 30
      rate_c := W / 40
      rate_abc := rate_a + rate_b + rate_c
      rate_ab := rate_a + rate_b in
  ∀ (D : ℝ), (D - 4) * rate_abc + 4 * rate_ab = W → D = 11 :=
by
  sorry

end work_completed_in_11_days_l736_736113


namespace opposite_of_neg3_l736_736428

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l736_736428


namespace lucy_jump_count_l736_736375

-- Duration of each song in seconds
def songDurations : List ℕ := [210, 150, 180, 240, 228, 132, 192, 270, 162, 234]

-- Jump rates (times per second) corresponding to slow, medium, and fast songs
def jumpRates : List ℚ := [1.2, 2.0, 1.5, 1.5, 1.2, 2.0, 1.2, 1.5, 2.0, 1.2]

-- Calculate jumps for each song
def jumps_per_song (durations : List ℕ) (rates : List ℚ) : List ℚ :=
  durations.zip rates |>.map (λ (d, r) => d * r)

-- Sum of all jumps, rounding each value to the nearest integer (down)
noncomputable def totalJumps : ℕ :=
  (jumps_per_song songDurations jumpRates).map (λ j => ⌊j⌋).sum

-- The main theorem: Lucy will jump exactly 2958 times
theorem lucy_jump_count : totalJumps = 2958 := by
  sorry

end lucy_jump_count_l736_736375


namespace value_of_a2023_plus_b2023_l736_736614

theorem value_of_a2023_plus_b2023 (a b : ℝ)
  (h1 : {a, b / a, 1} = {a^2, a + b, 0}): a^2023 + b^2023 = -1 :=
by
  sorry

end value_of_a2023_plus_b2023_l736_736614


namespace evaluate_f_640_minus_f_320_l736_736571

def σ (n : ℕ) : ℕ :=
  (Nat.divisors n).sum

def f (n : ℕ) : ℚ :=
  σ n / n

theorem evaluate_f_640_minus_f_320 :
  (f 640) - (f 320) = (3 : ℚ) / 320 :=
by {
  -- The proof will be placed here.
  sorry
}

end evaluate_f_640_minus_f_320_l736_736571


namespace cuboctahedron_max_sides_l736_736966

theorem cuboctahedron_max_sides (cube_edge_length : ℝ) (h_cube_edge_length : cube_edge_length = 1) :
    ∃ (n : ℕ), n = 8 ∧ ∀ polygon (is_cross_section : polygon.regular ∧ polygon.is_cross_section_of_cuboctahedron),
    polygon.num_sides ≤ n :=
by
  sorry

end cuboctahedron_max_sides_l736_736966


namespace percentage_sold_is_40_l736_736956

variables (O R S : ℝ)

def original_apples : ℝ := 699.9998833333527
def remaining_apples : ℝ := 420
def apples_sold : ℝ := original_apples - remaining_apples
def percentage_apples_sold : ℝ := (apples_sold / original_apples) * 100

theorem percentage_sold_is_40 : percentage_apples_sold ≈ 40 := 
  by 
    sorry

end percentage_sold_is_40_l736_736956


namespace eight_people_permutations_l736_736339

theorem eight_people_permutations : ∃ (n : ℕ), (n = 8!) ∧ (n = 40320) :=
by
  use 8!
  split
  { rfl }
  { sorry }

end eight_people_permutations_l736_736339


namespace average_fuel_efficiency_round_trip_l736_736173

noncomputable def average_fuel_efficiency (d1 d2 mpg1 mpg2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let fuel_used := (d1 / mpg1) + (d2 / mpg2)
  total_distance / fuel_used

theorem average_fuel_efficiency_round_trip :
  average_fuel_efficiency 180 180 36 24 = 28.8 :=
by 
  sorry

end average_fuel_efficiency_round_trip_l736_736173


namespace find_principal_l736_736962

theorem find_principal (R : ℝ) (T : ℝ) (I : ℝ) (hR : R = 0.12) (hT : T = 1) (hI : I = 1500) :
  ∃ P : ℝ, I = P * R * T ∧ P = 12500 := 
by
  use 12500
  rw [hR, hT, hI]
  norm_num
  sorry

end find_principal_l736_736962


namespace complex_conjugate_sum_l736_736731

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736731


namespace sum_of_consecutive_odd_numbers_l736_736880

theorem sum_of_consecutive_odd_numbers (n : ℕ) : (∑ i in finset.range (n + 1), (2 * i + 1)) = (n + 1) ^ 2 :=
sorry

end sum_of_consecutive_odd_numbers_l736_736880


namespace square_side_length_square_area_l736_736017

theorem square_side_length 
  (d : ℝ := 4) : (s : ℝ) = 2 * Real.sqrt 2 :=
  sorry

theorem square_area 
  (s : ℝ := 2 * Real.sqrt 2) : (A : ℝ) = 8 :=
  sorry

end square_side_length_square_area_l736_736017


namespace find_x_l736_736674

  -- Definition of the vectors
  def a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
  def b : ℝ × ℝ := (2, 1)

  -- Condition that vectors are parallel
  def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

  -- Theorem statement
  theorem find_x (x : ℝ) (h : are_parallel (a x) b) : x = 5 :=
  sorry
  
end find_x_l736_736674


namespace side_length_of_cube_l736_736029

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l736_736029


namespace sequence_sum_l736_736188

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 2 else 
  if n = 2 then 3 else 
  (1 / 2) * sequence (n - 1) + (1 / 3) * sequence (n - 2)

theorem sequence_sum : (∑' n, sequence n) = 24 :=
  sorry

end sequence_sum_l736_736188


namespace trigonometric_identity_l736_736001

theorem trigonometric_identity (x : ℝ) :
  (tan x + 4 * tan (3 * x) + 9 * tan (9 * x) + 27 * cot (27 * x)) = cot x :=
begin
  sorry
end

end trigonometric_identity_l736_736001


namespace cyclists_meet_again_l736_736885

-- Definitions of speeds and track length
def track_length : ℝ := 600
def speed1 : ℝ := 3.6
def speed2 : ℝ := 3.9
def speed3 : ℝ := 4.2

-- The time at which they meet again
def time_seconds := 2000

-- The Lean statement to prove
theorem cyclists_meet_again :
  ∃ t : ℕ, 
  ⟪ (speed2 - speed1) * t % track_length = 0 ⟫ ∧
  ⟪ (speed3 - speed2) * t % track_length = 0 ⟫ ∧
  ⟪ (speed3 - speed1) * t % track_length = 0 ⟫ ∧
  t = time_seconds := 
sorry

end cyclists_meet_again_l736_736885


namespace a_10_contains_more_than_1000_nines_l736_736049

noncomputable def a : ℕ → ℕ
| 0       := 9
| (n + 1) := 3 * a n ^ 4 + 4 * a n ^ 3

theorem a_10_contains_more_than_1000_nines : 
  ∃ k ≥ 1000, ∀ m, 9 * 10^m + k = a 10 :=
sorry

end a_10_contains_more_than_1000_nines_l736_736049


namespace triangle_mnk_obtuse_l736_736021

-- Define the elements of the problem
variables {P : Type*} [EuclideanGeometry P]

open EuclideanGeometry

-- Define the quadrilateral ABCD with given properties
variables (A B C D K M N : P)
variables (AD BC : ℝ)

-- Assume the conditions
variables (h0 : convex_quadrilateral A B C D)
variables (h1 : collinear [A, B, K])
variables (h2 : collinear [C, D, K])
variables (h3 : dist A D = dist B C)
variables (hm : midpoint M A B)
variables (hn : midpoint N C D)

-- Prove that triangle MNK is obtuse
theorem triangle_mnk_obtuse
  (h0 : convex_quadrilateral A B C D)
  (h1 : collinear [A, B, K])
  (h2 : collinear [C, D, K])
  (h3 : dist A D = dist B C)
  (hm : midpoint M A B)
  (hn : midpoint N C D) :
  obtuse_triangle M N K :=
sorry

end triangle_mnk_obtuse_l736_736021


namespace isosceles_triangle_sides_l736_736626

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l736_736626


namespace valid_pairs_count_l736_736010

def triangle_area (P Q O : (ℤ × ℤ)) : ℤ :=
  let (Px, Py) := P in
  let (Qx, Qy) := Q in
  let (Ox, Oy) := O in
  abs ((Px - Ox) * (Qy - Oy) - (Py - Oy) * (Qx - Ox))

def valid_triangle (m n : ℤ) : Prop :=
  0 < m ∧ m < n ∧ triangle_area (m, n) (n, m) (0, 0) = 2024

def count_valid_pairs : ℕ := 
  let pairs := [(m, n) | m n, m n ∈ ℕ, valid_triangle m n] in
  pairs.length

theorem valid_pairs_count : count_valid_pairs = 6 :=
  sorry

end valid_pairs_count_l736_736010


namespace quadrilateral_AB_eq_BP_iff_angle_MXB_60_l736_736340

variables {A B C D M P X : Type}
variables [has_angle A B D] [has_angle A D C]
variables [midpoint M A D] 
variables [parallel (line_through M P) (line_through C D)] 
variables [on_line X (line_through C D)]
variables [equal_length B X M X]

theorem quadrilateral_AB_eq_BP_iff_angle_MXB_60 :
  (angle B = 60 ∧ angle D = 60 ∧ AB = BP) ↔ (angle (line_through M X) B = 60) :=
sorry

end quadrilateral_AB_eq_BP_iff_angle_MXB_60_l736_736340


namespace jordyn_total_cost_l736_736325

theorem jordyn_total_cost (
  price_cherries : ℕ := 5,
  price_olives : ℕ := 7,
  discount : ℕ := 10,
  quantity : ℕ := 50
) : (50 * (price_cherries - (price_cherries * discount / 100)) + 50 * (price_olives - (price_olives * discount / 100))) = 540 :=
by
  sorry

end jordyn_total_cost_l736_736325


namespace cos_identity_l736_736630

theorem cos_identity (α : ℝ) (h : sin α - cos α = 1/3) : cos (π/2 - 2*α) = 8/9 :=
by
  sorry

end cos_identity_l736_736630


namespace pine_count_25_or_26_l736_736065

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l736_736065


namespace complex_conjugate_sum_l736_736726

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736726


namespace opposite_of_neg_three_l736_736447

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l736_736447


namespace flowers_per_basket_l736_736216

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l736_736216


namespace bills_fraction_l736_736111
-- Lean 4 code

theorem bills_fraction (total_stickers : ℕ) (andrews_fraction : ℚ) (total_given_away : ℕ)
  (andrews_stickers : ℕ) (remaining_stickers : ℕ)
  (bills_stickers : ℕ) :
  total_stickers = 100 →
  andrews_fraction = 1/5 →
  andrews_stickers = 1/5 * 100 →
  total_given_away = 44 →
  andrews_stickers = 20 →
  remaining_stickers = total_stickers - andrews_stickers →
  bills_stickers = total_given_away - andrews_stickers →
  bills_stickers = 24 →
  bills_stickers / remaining_stickers = 3 / 10 :=
begin
  sorry
end

end bills_fraction_l736_736111


namespace find_ellipse_eq_find_range_t_l736_736649

noncomputable theory

-- Define the properties of the ellipse and circle
def ellipse_eq (a b x y : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity_eq (e a b : ℝ) : Prop := 
  e = (Real.sqrt 2) / 2 ∧ e^2 = (a^2 - b^2) / a^2 ∧ e^2 = 1 / 2

def tangent_line_circle (b x y : ℝ) : Prop := 
  b = Real.sqrt 2 / Real.sqrt (1+1) ∧ b = 1 ∧ (x - y + Real.sqrt 2 = 0)

-- Define the conditions and proof goals
def condition_one (x y a b : ℝ) : Prop := 
  ellipse_eq a b x y ∧ eccentricity_eq ((Real.sqrt 2) / 2) a b ∧ tangent_line_circle b x y

theorem find_ellipse_eq (a b : ℝ) : 
  condition_one 0 0 a b → (a^2 = 2 ∧ b = 1) → (∀ x y, ellipse_eq a b x y = (x^2 / 2 + y^2 = 1)) :=
sorry

def line_AB (k x y : ℝ) : Prop := y = k * (x - 2)

def condition_two (x1 y1 x2 y2 k a b t : ℝ) : Prop := 
  line_AB k x1 y1 ∧ line_AB k x2 y2 ∧ ellipse_eq a b x1 y1 ∧ ellipse_eq a b x2 y2 ∧ 
  (x1 + x2, y1 + y2) = t * (x, y) ∧ 
  (1 + k^2) * ((x1 + x2)^2 - 4 * x1 * x2) < 20 / 9 ∧ 
  16 * k^2 = t^2 * (1 + 2 * k^2)

theorem find_range_t (t : ℝ) : 
  (∃ x1 y1 x2 y2 k a b, condition_two x1 y1 x2 y2 k a b t) → 
  t ∈ set.Ioo (-2) (-((2 : ℝ) * Real.sqrt 6) / 3) ∪ set.Ioo ((2 : ℝ) * Real.sqrt 6 / 3) 2 :=
sorry

end find_ellipse_eq_find_range_t_l736_736649


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736813

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736813


namespace max_triangle_perimeter_l736_736548

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l736_736548


namespace part_a_part_b_l736_736829

-- Define the conditions
def initial_pills_in_bottles : ℕ := 10
def start_date : ℕ := 1 -- Representing March 1 as day 1
def check_date : ℕ := 14 -- Representing March 14 as day 14
def total_days : ℕ := 13

-- Define the probability of finding an empty bottle on day 14 as a proof
def probability_find_empty_bottle_on_day_14 : ℚ :=
  286 * (1 / 2 ^ 13)

-- The expected value calculation for pills taken before finding an empty bottle
def expected_value_pills_taken : ℚ :=
  21 * (1 - (1 / (real.sqrt (10 * real.pi))))

-- Assertions to be proven
theorem part_a : probability_find_empty_bottle_on_day_14 = 143 / 4096 := sorry
theorem part_b : expected_value_pills_taken ≈ 17.3 := sorry

end part_a_part_b_l736_736829


namespace f_at_minus_one_l736_736693

variable {ℝ : Type*} [Real ℝ]

-- Define the differentiable function f
def f (x : ℝ) := x^2 + 2 * deriv f 2 * x + 3

-- State the goal
theorem f_at_minus_one : diffeq_on(ℝ, f) → f(-1) = 12 := by
  sorry

end f_at_minus_one_l736_736693


namespace min_val_f_ab_l736_736653

def f : ℝ → ℝ 
| x :=
  if 1 ≤ x then (Real.log x)
  else if 0 < x then -(Real.log x)
  else 0 -- This case should never arise due to the constraints 0 < x.

theorem min_val_f_ab : 
  ∀ (a b : ℝ), 0 < a ∧ a < b ∧ f a = f b ∧ a * b = 1 ∧ (b = 4 * a) → 
  f (a + b) = Real.log 5 - Real.log 4 :=
by
  intros a b h,
  sorry

end min_val_f_ab_l736_736653


namespace pine_count_25_or_26_l736_736064

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l736_736064


namespace sin_angle_BAE_l736_736502

-- Introducing the conditions as Lean 4 definitions
variables (s : ℝ) (A B E : EuclideanSpace ℝ (Fin 3))
variable (h_cube : ∀ i j, A i = 0 → E j = A j + s)
variables (h_AB : dist A B = s) (h_AE : dist A E = s)
variables (right_angle_A : angle A B E = π / 2)

-- Statement to prove
theorem sin_angle_BAE : sin (angle B A E) = 1 / Real.sqrt 2 :=
sorry

end sin_angle_BAE_l736_736502


namespace mary_total_cost_is_48_l736_736382

structure Prices where
  apple : ℕ
  orange : ℕ
  banana : ℕ
  peach : ℕ
  grape : ℕ

structure Discounts where
  per_five_fruits : ℕ
  peach_grape : ℕ
  orange : Float
  banana_percent : Float

structure Purchased where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ
  peaches : ℕ
  grapes : ℕ

noncomputable def calculate_total_cost 
  (prices : Prices)
  (discounts : Discounts)
  (purchased : Purchased) : ℕ :=
let total_fruits := purchased.apples + purchased.oranges + purchased.bananas + purchased.peaches + purchased.grapes
let initial_cost := (purchased.apples * prices.apple) + 
                    (purchased.oranges * prices.orange) + 
                    (purchased.bananas * prices.banana) + 
                    (purchased.peaches * prices.peach) + 
                    (purchased.grapes * prices.grape)
let discount_for_five_fruits := (total_fruits / 5) * discounts.per_five_fruits
let discount_for_peach_grape := ((purchased.peaches / 3) * (purchased.grapes / 2)) * discounts.peach_grape
let effective_orange_cost := (purchased.oranges / 2 + purchased.oranges % 2) * prices.orange
let effective_banana_cost := (purchased.bananas * prices.banana * (1 - discounts.banana_percent))
let total_cost := initial_cost - discount_for_five_fruits - discount_for_peach_grape + 
                effective_orange_cost + effective_banana_cost - (purchased.oranges * prices.orange)
(total_cost : ℕ)

theorem mary_total_cost_is_48 : 
  calculate_total_cost 
    { apple := 1, orange := 2, banana := 3, peach := 4, grape := 5 }
    { per_five_fruits := 1, peach_grape := 3, orange := 1, banana_percent := 0.25 }
    { apples := 5, oranges := 6, bananas := 4, peaches := 6, grapes := 4 } = 48 :=
sorry

end mary_total_cost_is_48_l736_736382


namespace number_of_students_scored_no_more_than_70_l736_736957

open MeasureTheory

noncomputable def student_scores : ℝ → MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.normal 90 σ²

theorem number_of_students_scored_no_more_than_70 :
  (∫ x in set.Ioo 70 110, (student_scores x)).to_real = 0.7 →
  ∫ x in set.Iic 70, (student_scores x) * 1000 = 150 :=
by
  sorry

end number_of_students_scored_no_more_than_70_l736_736957


namespace value_of_reciprocal_l736_736317

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : 2 + real.log2 a = 3 + real.log 3 b
axiom h4 : 3 + real.log 3 b = real.log 6 (a + b)

-- Theorem to prove
theorem value_of_reciprocal (h1 : 0 < a) (h2 : 0 < b) 
                            (h3 : 2 + real.log2 a = 3 + real.log 3 b)
                            (h4 : 3 + real.log 3 b = real.log 6 (a + b)) : 
                            (1 / a + 1 / b) = 108 :=
sorry

end value_of_reciprocal_l736_736317


namespace find_2001st_removed_sq_term_find_position_of_2001_l736_736475

def natural_number (n : Nat) : Prop := true

def is_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def without_square_numbers (n : Nat) : Nat :=
  n + Nat.sqrt n

theorem find_2001st_removed_sq_term :
  without_square_numbers 2000 = 2046 := sorry

theorem find_position_of_2001 :
  ∃ k : Nat, without_square_numbers k = 2001 :=
  ∃ k : Nat, k + Nat.sqrt k = 2001 ∧ k = 1957 := sorry

end find_2001st_removed_sq_term_find_position_of_2001_l736_736475


namespace probability_square_area_between_25_and_49_l736_736839

theorem probability_square_area_between_25_and_49 :
  let length_AB := 10
  let favorable_length := 2
  let total_length := 10
  probability_square_area_between_25_and_49 = favorable_length / total_length := sorry

end probability_square_area_between_25_and_49_l736_736839


namespace log_base_27_3_l736_736238

-- Define the condition
lemma log_base_condition : 27 = 3^3 := rfl

-- State the theorem to be proven
theorem log_base_27_3 : log 27 3 = 1 / 3 :=
by 
  -- skip the proof for now
  sorry

end log_base_27_3_l736_736238


namespace mean_score_74_9_l736_736757

/-- 
In a class of 100 students, the score distribution is as follows:
- 10 students scored 100%
- 15 students scored 90%
- 20 students scored 80%
- 30 students scored 70%
- 20 students scored 60%
- 4 students scored 50%
- 1 student scored 40%

Prove that the mean percentage score of the class is 74.9.
-/
theorem mean_score_74_9 : 
  let scores := [100, 90, 80, 70, 60, 50, 40]
  let counts := [10, 15, 20, 30, 20, 4, 1]
  let total_students := 100
  let total_score := 1000 + 1350 + 1600 + 2100 + 1200 + 200 + 40
  (total_score / total_students : ℝ) = 74.9 :=
by {
  -- The detailed proof steps are omitted with sorry.
  sorry
}

end mean_score_74_9_l736_736757


namespace ray_DY_bisects_angle_ZDB_l736_736505

/-- Let ABCD be a quadrilateral inscribed in a circle such that BC = CD. The diagonals AC and BD intersect at X. 
Let AD < AB. The circumcircle of triangle BCX intersects the segment AB at Y ≠ B. 
Ray CY meets the circle Ω again at Z ≠ C. Prove that ray DY bisects angle ZDB. -/
theorem ray_DY_bisects_angle_ZDB 
  (A B C D X Y Z : Point)
  (circle_Omega : Circle ℝ)
  (h1 : InscribedQuadrilateral A B C D circle_Omega)
  (h2 : BC = CD)
  (h3 : Intersects (Line AC) (Line BD) at X)
  (h4 : AD < AB)
  (h5 : IntersectsCircumcircleOfBCX (Triangle B C X) (Line AB) at Y ≠ B)
  (h6 : Meets (Ray CY) circle_Omega at Z ≠ C)
  : Bisects (Ray DY) (angle Z D B) :=
sorry

end ray_DY_bisects_angle_ZDB_l736_736505


namespace number_of_good_sets_l736_736301

def is_good_set (C : set (ℝ × ℝ)) : Prop :=
∀ (x1 y1 : ℝ), (x1, y1) ∈ C → ∃ (x2 y2 : ℝ), (x2, y2) ∈ C ∧ x1 * x2 + y1 * y2 = 0

def C1 : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 9 }
def C2 : set (ℝ × ℝ) := { p | p.1 ^ 2 - p.2 ^ 2 = 9 }
def C3 : set (ℝ × ℝ) := { p | 2 * p.1 ^ 2 + p.2 ^ 2 = 9 }
def C4 : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 = 9 }

theorem number_of_good_sets : 
  (if is_good_set C1 then 1 else 0) +
  (if is_good_set C2 then 1 else 0) +
  (if is_good_set C3 then 1 else 0) +
  (if is_good_set C4 then 1 else 0) = 3 := sorry

end number_of_good_sets_l736_736301


namespace shortest_distance_from_W_to_V_through_cylinder_l736_736395

def shortest_distance (WX WZ ZV : ℕ) : ℕ :=
  let π := Real.pi
  let triangle := WY * WY + 2 * (2 / π) ^ 2
  let square := 9 * π * π + 8
  sqrt (square / π * π)
  
theorem shortest_distance_from_W_to_V_through_cylinder :
  shortest_distance 4 3 3 = 18 :=
  sorry

end shortest_distance_from_W_to_V_through_cylinder_l736_736395


namespace z_conjugate_sum_l736_736721

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736721


namespace shane_current_age_l736_736895

theorem shane_current_age (Garret_age : ℕ) (h : Garret_age = 12) : 
  (let Shane_age_twenty_years_ago := 2 * Garret_age in
   let Shane_current := Shane_age_twenty_years_ago + 20 in
   Shane_current = 44) :=
by
  sorry

end shane_current_age_l736_736895


namespace cube_surface_area_l736_736927

theorem cube_surface_area (V : ℝ) (hV : V = 3375) : 
  let s := real.cbrt V 
  in 6 * s^2 = 1350 :=
by
  sorry

end cube_surface_area_l736_736927


namespace circles_intersect_at_two_points_l736_736312

theorem circles_intersect_at_two_points : 
  let C1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
  let C2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 36}
  ∃ pts : Finset (ℝ × ℝ), pts.card = 2 ∧ ∀ p ∈ pts, p ∈ C1 ∧ p ∈ C2 := 
sorry

end circles_intersect_at_two_points_l736_736312


namespace baker_ingredients_l736_736509

theorem baker_ingredients (flour : ℚ) (eggs_per_flour : ℚ) (milk_per_flour : ℚ) (sugar_per_flour : ℚ) (n_flour : ℚ)
  (h_eggs_per_flour : eggs_per_flour = 3 / 2)
  (h_milk_per_flour : milk_per_flour = 1 / 4)
  (h_sugar_per_flour : sugar_per_flour = 6 / 5)
  (h_n_flour : n_flour = 24) :
  let n_eggs := eggs_per_flour * n_flour,
      n_milk := milk_per_flour * n_flour,
      n_sugar := sugar_per_flour * n_flour in
  n_eggs = 36 ∧ n_milk = 6 ∧ n_sugar ≈ 29 :=
by {
  sorry
}

end baker_ingredients_l736_736509


namespace shooter_probability_l736_736540

theorem shooter_probability (hit_prob : ℝ) (n : ℕ) (k : ℕ) (hit_prob_condition : hit_prob = 0.8) (n_condition : n = 5) (k_condition : k = 2) :
  (probability (at_least_k_hits n hit_prob k) = 0.9929) :=
sorry

end shooter_probability_l736_736540


namespace tetrahedron_angle_sum_constant_l736_736346

theorem tetrahedron_angle_sum_constant
  (A B C D E F : Type*)
  [regular_tetrahedron A B C D]
  [on_edge E A B]
  [on_edge F C D]
  (λ : ℝ) (hλ : 0 < λ)
  (hE : (AE / EB) = λ)
  (hF : (CF / FD) = λ) :
  let α := angle EF AC,
      β := angle EF BD in
  α + β = 90 :=
by sorry

end tetrahedron_angle_sum_constant_l736_736346


namespace distribution_table_and_expected_value_conditional_probability_l736_736452

-- Definitions and conditions
def seating_arrangement := {A B C D E : Type}
def X (S : seating_arrangement) : ℕ := sorry -- Define X as the number of experts in correct seats

-- Distribution probabilities
def P_X (X_val : ℕ) : ℚ :=
  match X_val with
  | 0     => 11 / 30
  | 1     => 3 / 8
  | 2     => 1 / 6
  | 3     => 1 / 12
  | 5     => 1 / 120
  | _     => 0 -- X can only be 0, 1, 2, 3, 5

-- Expected value
def E_X : ℚ := 0 * (11 / 30) + 1 * (3 / 8) + 2 * (1 / 6) + 3 * (1 / 12) + 5 * (1 / 120)

-- Conditional probability
def three_wrong : Prop := sorry -- Define the condition that three are in the wrong seats
def two_correct : Prop := sorry -- Define the condition that exactly two are in the correct seats
def P_conditional : ℚ := (1 / 6) / ((1 - (1 / 12) - (1 / 120)))

-- Proof statements
theorem distribution_table_and_expected_value :
  (∀ j, (j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 5 → P_X j = P_X j) ∧
  E_X = 1) := by sorry

theorem conditional_probability :
  (three_wrong → two_correct → P_conditional = 20 / 109) := by sorry

end distribution_table_and_expected_value_conditional_probability_l736_736452


namespace max_quarters_is_13_l736_736400

noncomputable def number_of_quarters (total_value : ℝ) (quarters nickels dimes : ℝ) : Prop :=
  total_value = 4.55 ∧
  quarters = nickels ∧
  dimes = quarters / 2 ∧
  (0.25 * quarters + 0.05 * nickels + 0.05 * quarters / 2 = 4.55)

theorem max_quarters_is_13 : ∃ q : ℝ, number_of_quarters 4.55 q q (q / 2) ∧ q = 13 :=
by
  sorry

end max_quarters_is_13_l736_736400


namespace two_digit_integers_count_l736_736651

theorem two_digit_integers_count : 
  let digits : Set ℕ := {2, 4, 7, 8} in
  (∃ (nums : ℕ → Set ℕ), 
    (∀ n, n ∈ nums n ↔ (n / 10 ∈ digits ∧ n % 10 ∈ digits ∧ n / 10 ≠ n % 10)) ∧
    nums 2 = {20, 21, 23, 24}) → 
  card (nums 2) = 12 :=
by
  sorry

end two_digit_integers_count_l736_736651


namespace visibility_set_convex_polygon_l736_736175

theorem visibility_set_convex_polygon (n : ℕ) (P : Polygon) (hP : P.is_non_convex) :
  ∃ (m : ℕ) (T : ConvexPolygon), m ≤ n ∧ T.verts ⊆ P.inner_verts ∧ ∀ p ∈ T.points, ∀ v ∈ P.verts, is_visible_from p v :=
sorry

end visibility_set_convex_polygon_l736_736175


namespace binomial_coefficient_sum_l736_736566

theorem binomial_coefficient_sum (n : ℕ) :
    (∑ k in Finset.range n, 2^k * Nat.choose n (k + 1)) = (3^n - 1) / 2 := by
  sorry

end binomial_coefficient_sum_l736_736566


namespace find_f_2_l736_736655

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 3

theorem find_f_2 (a b : ℝ) (hf_neg2 : f a b (-2) = 7) : f a b 2 = -13 :=
by
  sorry

end find_f_2_l736_736655


namespace pascal_triangle_sum_difference_l736_736999

theorem pascal_triangle_sum_difference :
  (∑ i in Finset.range 101, (Nat.choose (101) i) / (Nat.choose (102) i)) - 
  (∑ i in Finset.range 100, (Nat.choose (100) i) / (Nat.choose (101) i)) = 1 / 2 := 
sorry

end pascal_triangle_sum_difference_l736_736999


namespace opposite_of_neg3_l736_736442

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l736_736442


namespace cafeteria_ground_mince_total_l736_736537

theorem cafeteria_ground_mince_total :
  let L := 100 -- number of lasagnas
  let P := 100 -- number of cottage pies
  let mince_per_lasagna := 2 -- pounds per lasagna
  let mince_per_pie := 3 -- pounds per pie
  L * mince_per_lasagna + P * mince_per_pie = 500 :=
by
  have l_mince : L * mince_per_lasagna = 200 := rfl
  have p_mince : P * mince_per_pie = 300 := rfl
  show L * mince_per_lasagna + P * mince_per_pie = 500, by
    rw [l_mince, p_mince]
    exact rfl

end cafeteria_ground_mince_total_l736_736537


namespace rectangle_area_l736_736321

theorem rectangle_area (a b : ℝ) (h : 2 * a^2 - 11 * a + 5 = 0) (hb : 2 * b^2 - 11 * b + 5 = 0) : a * b = 5 / 2 :=
sorry

end rectangle_area_l736_736321


namespace number_of_pines_l736_736060

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l736_736060


namespace sum_conjugate_eq_two_l736_736740

variable {ℂ : Type} [ComplexField ℂ]

theorem sum_conjugate_eq_two (z : ℂ) (h : complex.I * (1 - z) = 1) : z + conj(z) = 2 :=
by
  sorry

end sum_conjugate_eq_two_l736_736740


namespace area_ratio_equilateral_triangle_l736_736372

theorem area_ratio_equilateral_triangle (XYZ : Triangle) (K L M : Point) 
(h1 : XYZ.is_equilateral) 
(h2 : point_on_side K XYZ.XY ∧ ratio_of_division K XYZ.XY = B) 
(h3 : point_on_side L XYZ.YZ ∧ ratio_of_division L XYZ.YZ = (1/C)) 
(h4 : point_on_side M XYZ.ZX ∧ ratio_of_division M XYZ.ZX = 1) : 
area_ratio K L M XYZ = 1/5 := sorry

end area_ratio_equilateral_triangle_l736_736372


namespace acrobat_count_l736_736493

theorem acrobat_count (a e c : ℕ) (h1 : 2 * a + 4 * e + 2 * c = 88) (h2 : a + e + c = 30) : a = 2 :=
by
  sorry

end acrobat_count_l736_736493


namespace sum_of_solutions_eq_0_l736_736258

-- Define the conditions
def y : ℝ := 6
def main_eq (x : ℝ) : Prop := x^2 + y^2 = 145

-- State the theorem
theorem sum_of_solutions_eq_0 : 
  let x1 := Real.sqrt 109
  let x2 := -Real.sqrt 109
  x1 + x2 = 0 :=
by {
  sorry
}

end sum_of_solutions_eq_0_l736_736258


namespace log_base_27_of_3_l736_736230

theorem log_base_27_of_3 (h : 27 = 3 ^ 3) : log 27 3 = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736230


namespace even_number_combinations_l736_736155

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736155


namespace dog_food_cans_l736_736198

theorem dog_food_cans 
  (packages_cat_food : ℕ)
  (cans_per_package_cat_food : ℕ)
  (packages_dog_food : ℕ)
  (additional_cans_cat_food : ℕ)
  (total_cans_cat_food : ℕ)
  (total_cans_dog_food : ℕ)
  (num_cans_dog_food_package : ℕ) :
  packages_cat_food = 9 →
  cans_per_package_cat_food = 10 →
  packages_dog_food = 7 →
  additional_cans_cat_food = 55 →
  total_cans_cat_food = packages_cat_food * cans_per_package_cat_food →
  total_cans_dog_food = packages_dog_food * num_cans_dog_food_package →
  total_cans_cat_food = total_cans_dog_food + additional_cans_cat_food →
  num_cans_dog_food_package = 5 :=
by
  sorry

end dog_food_cans_l736_736198


namespace length_QR_l736_736330

theorem length_QR {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (triangle_ABC : Triangle A B C)
    (hAB : dist A B = 10)
    (hAC : dist A C = 8)
    (hBC : dist B C = 6)
    (P : Circle A) 
    (hPC : P.C ∈ P) 
    (hP_tangent : P.Tangent AB) 
    (Q R : Point)
    (hQ : Q ∉ {A B C})
    (hR : R ∉ {A B C}) 
    (hPQintAC : P.Int (Line A C) Q)
    (hPRintBC : P.Int (Line B C) R) :
  dist Q R = 9.6 :=
sorry

end length_QR_l736_736330


namespace angle_PCB_in_rectangle_equilateral_triangle_l736_736112

-- Define the main Lean statement
theorem angle_PCB_in_rectangle_equilateral_triangle : 
  ∀ (A B C D M P : Point) (h_rectangle : is_rectangle A B C D)
    (h_AB_CD : dist A B = 1 ∧ dist C D = 1) 
    (h_BC_DA : dist B C = 2 ∧ dist D A = 2)
    (h_M_mid_AD : midpoint M A D) 
    (h_P_cond : ∃ Q : Point, Q ≠ A ∧ equilateral_triangle M B P),
  angle P C B = 30 :=
by
  sorry

end angle_PCB_in_rectangle_equilateral_triangle_l736_736112


namespace marked_price_percent_l736_736179

variable {L : ℝ} -- List price
variable {P : ℝ} -- Purchase price
variable {M : ℝ} -- Marked price
variable {S : ℝ} -- Selling price
variable {T : ℝ} -- Actual amount received after tax
variable {Profit : ℝ} -- Profit

-- Given conditions
def list_price (L : ℝ) := L = 100
def purchase_price (P L : ℝ) := P = L * 0.70
def discount_selling_price (S M : ℝ) := S = M * 0.75
def amount_after_tax (T S : ℝ) := T = S * 0.95
def profit_condition (T P : ℝ) := T - P = 0.30 * T

-- Prove that marked price should be 140% of the list price
theorem marked_price_percent (L M : ℝ) 
    (h1: list_price L) 
    (h2: purchase_price P L) 
    (h3: discount_selling_price S M) 
    (h4: amount_after_tax T S) 
    (h5: profit_condition T P) : 
    M = 1.40 * L := 
by
-- Leave the actual proof as an exercise
sory

end marked_price_percent_l736_736179


namespace rachel_older_than_leah_l736_736393

theorem rachel_older_than_leah (rachel_age leah_age : ℕ) (h1 : rachel_age = 19) (h2 : rachel_age + leah_age = 34) :
  rachel_age - leah_age = 4 :=
by sorry

end rachel_older_than_leah_l736_736393


namespace range_of_m_trigonometric_set_nonempty_l736_736753

noncomputable def trigonometric_set_nonempty (m : ℝ) : Prop :=
  ∃ x : ℝ, cos x ^ 2 + sin x + m = 0

theorem range_of_m_trigonometric_set_nonempty (m : ℝ) :
  trigonometric_set_nonempty m ↔ - (5 / 4) ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_trigonometric_set_nonempty_l736_736753


namespace total_miles_walked_l736_736872

-- Definition of the conditions
def num_islands : ℕ := 4
def miles_per_day_island1 : ℕ := 20
def miles_per_day_island2 : ℕ := 25
def days_per_island : ℚ := 1.5

-- Mathematically Equivalent Proof Problem
theorem total_miles_walked :
  let total_miles_island1 := 2 * (miles_per_day_island1 * days_per_island)
  let total_miles_island2 := 2 * (miles_per_day_island2 * days_per_island)
  total_miles_island1 + total_miles_island2 = 135 := by
  sorry

end total_miles_walked_l736_736872


namespace acute_angle_cos_30_l736_736748

theorem acute_angle_cos_30 (A : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : real.cos A = sqrt 3 / 2) : A = π / 6 :=
by
  sorry

end acute_angle_cos_30_l736_736748


namespace log27_3_eq_one_third_l736_736234

theorem log27_3_eq_one_third : ∃ x : ℝ, (27 = 3^3) ∧ (27^x = 3) ∧ (x = 1/3) :=
by {
  have h1: 27 = 3^3 := by norm_num,
  have h2: 27^(1/3) = 3 := by {
    rw [h1, ← real.rpow_mul (real.rpow_pos_of_pos (by norm_num) (3: ℝ))],
    norm_num
  },
  exact ⟨1/3, h1, h2⟩
}

end log27_3_eq_one_third_l736_736234


namespace correct_statement_parallel_lines_l736_736917

-- Lean 4 statement representing the problem
theorem correct_statement_parallel_lines
  (P1 : ∀ l₁ l₂ l₃ : Line, intersect l₁ l₃ ∧ intersect l₂ l₃ → supplementary_angles l₁ l₂)
  (P2 : ∀ (Δ : Triangle), ∀ θ₁ θ₂, exterior_angle Δ θ₁ = θ₁ + θ₂)
  (P3 : ∀ l₁ l₂ l₃ : Line, parallel l₁ l₂ ∧ parallel l₂ l₃ → parallel l₁ l₃)
  (A_eq_3 : average A = 3)
  (B_eq_3 : average B = 3)
  (S_A_sq : variance A = 0.8)
  (S_B_sq : variance B = 1.4)
  : ∀ (s : String), s = "Two lines parallel to the same line are parallel to each other" → s := 
by sorry

end correct_statement_parallel_lines_l736_736917


namespace complex_conjugate_sum_l736_736713

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736713


namespace value_of_expression_l736_736621

variable {a : Nat → Int}

def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n m : Nat, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_expression
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 :=
  sorry

end value_of_expression_l736_736621


namespace dimes_left_l736_736397

-- Definitions based on the conditions
def Initial_dimes : ℕ := 8
def Sister_borrowed : ℕ := 4
def Friend_borrowed : ℕ := 2

-- The proof problem statement (without the proof)
theorem dimes_left (Initial_dimes Sister_borrowed Friend_borrowed : ℕ) : 
  Initial_dimes = 8 → Sister_borrowed = 4 → Friend_borrowed = 2 →
  Initial_dimes - (Sister_borrowed + Friend_borrowed) = 2 :=
by
  intros
  sorry

end dimes_left_l736_736397


namespace less_than_its_reciprocal_l736_736492

-- Define the numbers as constants
def a := -1/3
def b := -3/2
def c := 1/4
def d := 3/4
def e := 4/3 

-- Define the proposition that needs to be proved
theorem less_than_its_reciprocal (n : ℚ) :
  (n = -3/2 ∨ n = 1/4) ↔ (n < 1/n) :=
by
  sorry

end less_than_its_reciprocal_l736_736492


namespace opposite_of_neg3_l736_736443

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l736_736443


namespace max_xy_value_l736_736315

theorem max_xy_value (x y : ℕ) (h1 : 7 * x + 5 * y = 140) : 
  ∃ (x_max y_max : ℕ), (7 * x_max + 5 * y_max = 140) ∧ (x_max * y_max = 140) := 
begin
  sorry
end

end max_xy_value_l736_736315


namespace train_tunnel_length_l736_736978

theorem train_tunnel_length 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (time_for_tail_to_exit : ℝ) 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 90) 
  (h_time_for_tail_to_exit : time_for_tail_to_exit = 2 / 60) :
  ∃ tunnel_length : ℝ, tunnel_length = 1 := 
by
  sorry

end train_tunnel_length_l736_736978


namespace cube_side_length_l736_736034

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l736_736034


namespace nails_sum_is_correct_l736_736205

-- Define the fractions for sizes 2d, 3d, 5d, and 8d
def fraction_2d : ℚ := 1 / 6
def fraction_3d : ℚ := 2 / 15
def fraction_5d : ℚ := 1 / 10
def fraction_8d : ℚ := 1 / 8

-- Define the expected answer
def expected_fraction : ℚ := 21 / 40

-- The theorem to prove
theorem nails_sum_is_correct : fraction_2d + fraction_3d + fraction_5d + fraction_8d = expected_fraction :=
by
  -- The proof is not required as per the instructions
  sorry

end nails_sum_is_correct_l736_736205


namespace new_average_after_changes_l736_736401

theorem new_average_after_changes (s : Fin 10 → ℝ)
  (h_avg : (∑ i, s i) / 10 = 6.2)
  (h_modifications : ∀ i, i = 2 → s i + 4 |
                                i = 6 → s i + 2 |
                                i = 8 → s i + 5 |
                                i ≠ 2 ∧ i ≠ 6 ∧ i ≠ 8 → s i)
  : (∑ i, if i = 2 then s i + 4 else if i = 6 then s i + 2 else if i = 8 then s i + 5 else s i) / 10 = 7.3 :=
sorry

end new_average_after_changes_l736_736401


namespace maximize_profit_l736_736103

noncomputable def profit (x : ℝ) : ℝ :=
  let s_price := 50 + x
  let volume := 500 - 10 * x
  (s_price * volume) - (40 * volume)

theorem maximize_profit : ∃ x : ℝ, profit x = 9000 ∧ (50 + x = 70) :=
by {
  use 20,
  dsimp [profit],
  split,
    norm_num,
    norm_num,
  sorry
}

end maximize_profit_l736_736103


namespace z_conjugate_sum_l736_736708

theorem z_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by
  sorry

end z_conjugate_sum_l736_736708


namespace book_arrangement_l736_736976

theorem book_arrangement : 
  ∀ (G C : ℕ), G = 5 → C = 3 → (nat.choose (G + C) G) = 56 :=
by
  intros G C hG hC
  rw [hG, hC]
  apply nat.choose_eq_factorial_div_factorial
  sorry

end book_arrangement_l736_736976


namespace number_of_even_results_l736_736135

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736135


namespace ethan_presents_l736_736594

variable (A E : ℝ)

theorem ethan_presents (h1 : A = 9) (h2 : A = E - 22.0) : E = 31 := 
by
  sorry

end ethan_presents_l736_736594


namespace combined_resistance_parallel_l736_736338

theorem combined_resistance_parallel (R1 R2 : ℝ) (r : ℝ) 
  (hR1 : R1 = 8) (hR2 : R2 = 9) (h_parallel : (1 / r) = (1 / R1) + (1 / R2)) : 
  r = 72 / 17 :=
by
  sorry

end combined_resistance_parallel_l736_736338


namespace coworkers_count_l736_736990

def main_meal_cost : ℝ := 12.0
def appetizer_count : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0
def total_cost : ℝ := 77.0

theorem coworkers_count (x : ℝ) : 
  let meal_cost := main_meal_cost * x
      appetizer_total := appetizer_count * appetizer_cost
      subtotal := meal_cost + appetizer_total
      tip := tip_rate * subtotal
      total := subtotal + tip + rush_order_fee
  in total_cost = total → x - 1 = 3 :=
by
  sorry

end coworkers_count_l736_736990


namespace cut_short_consumption_l736_736925

theorem cut_short_consumption (N P : ℝ) (hN : N > 0) (hP : P > 0) :
  let newN := 0.9 * N,
      newP := 1.2 * P,
      C := (N * P) / (newN * newP) in
  ((C : ℝ) ≈ (0.9259 : ℝ)) → ((100 - C * 100) ≈ (7.41 : ℝ)) := by
  intros newN newP C hC
  sorry

end cut_short_consumption_l736_736925


namespace quality_related_production_line_l736_736131

theorem quality_related_production_line : 
  let a := 40
  let b := 80
  let c := 80
  let d := 100
  let n := a + b + c + d
  let K_squared := (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))
  in K_squared > 2.706 :=
by
  let a := 40
  let b := 80
  let c := 80
  let d := 100
  let n := a + b + c + d
  let K_squared := (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))
  sorry

end quality_related_production_line_l736_736131


namespace cube_side_length_l736_736043

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l736_736043


namespace symmetric_pentominoes_count_l736_736210

-- Assume we have exactly fifteen pentominoes
def num_pentominoes : ℕ := 15

-- Define the number of pentominoes with particular symmetrical properties
def num_reflectional_symmetry : ℕ := 8
def num_rotational_symmetry : ℕ := 3
def num_both_symmetries : ℕ := 2

-- The theorem we wish to prove
theorem symmetric_pentominoes_count 
  (n_p : ℕ) (n_r : ℕ) (n_b : ℕ) (n_tot : ℕ)
  (h1 : n_p = num_pentominoes)
  (h2 : n_r = num_reflectional_symmetry)
  (h3 : n_b = num_both_symmetries)
  (h4 : n_tot = n_r + num_rotational_symmetry - n_b) :
  n_tot = 9 := 
sorry

end symmetric_pentominoes_count_l736_736210


namespace cheap_feed_amount_l736_736890

theorem cheap_feed_amount (x y : ℝ) (h1 : x + y = 27) (h2 : 0.17 * x + 0.36 * y = 7.02) : 
  x = 14.21 :=
sorry

end cheap_feed_amount_l736_736890


namespace cubic_term_in_line_l736_736387

-- Define the equation of the line.
def line_equation (x : ℝ) : ℝ :=
  x^2 - x^3

-- Define the statement that needs to be proved.
theorem cubic_term_in_line :
  (∃ c : ℝ, c = -1 ∧ ∀ x, line_equation x = x^2 + c * x^3) :=
by
  exists -1
  split
  . refl
  . intro x
    sorry

end cubic_term_in_line_l736_736387


namespace cube_side_length_l736_736037

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l736_736037


namespace recurring_decimal_to_fraction_l736_736478

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l736_736478


namespace bacteria_seventh_generation_l736_736841

/-- Represents the effective multiplication factor per generation --/
def effective_mult_factor : ℕ := 4

/-- The number of bacteria in the first generation --/
def first_generation : ℕ := 1

/-- A helper function to compute the number of bacteria in the nth generation --/
def bacteria_count (n : ℕ) : ℕ :=
  first_generation * effective_mult_factor ^ n

/-- The number of bacteria in the seventh generation --/
theorem bacteria_seventh_generation : bacteria_count 7 = 4096 := by
  sorry

end bacteria_seventh_generation_l736_736841


namespace lambda_values_general_term_sum_inequality_l736_736275

def sequence (a : ℕ → ℤ) : Prop :=
  a(1) = 2 ∧ a(2) = 10 ∧ ∀ n : ℕ, n > 0 → a(n + 2) = 2 * a(n + 1) + 3 * a(n)

theorem lambda_values (a : ℕ → ℤ) (λ : ℤ) (geo_seq : ∀ (n : ℕ), n > 0 → a(n + 1) + λ * a(n) = (a(n + 2) + λ * a(n + 1)) / (a(n + 1) + λ * a(n))) :
  λ = 1 ∨ λ = -3 :=
  sorry

theorem general_term (a : ℕ → ℤ) (h : sequence a) : ∀ n : ℕ, a n = 3^n + (-1)^n :=
  sorry

theorem sum_inequality (a : ℕ → ℤ) (h : sequence a) : ∀ n : ℕ, n > 0 → (∑ k in finset.range(n + 1), 1 / (a (k + 1) : ℚ)) < 2 / 3 :=
  sorry

end lambda_values_general_term_sum_inequality_l736_736275


namespace white_tile_count_l736_736357

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end white_tile_count_l736_736357


namespace sara_payment_l736_736399

noncomputable def calculate_total (balloons_cost tablecloths_cost streamers_cost banners_cost confetti_cost discount_rate sales_tax_rate amount_received_back) : ℝ :=
  let total_cost := balloons_cost + tablecloths_cost + streamers_cost + banners_cost + confetti_cost
  let discount := (balloons_cost + tablecloths_cost) * discount_rate
  let discounted_cost := total_cost - discount
  let tax := discounted_cost * sales_tax_rate
  let final_amount := discounted_cost + tax
  final_amount + amount_received_back

theorem sara_payment :
  let balloons_cost := 3.50
  let tablecloths_cost := 18.25
  let streamers_cost := 9.10
  let banners_cost := 14.65
  let confetti_cost := 7.40
  let discount_rate := 0.10
  let sales_tax_rate := 0.05
  let amount_received_back := 6.38
  calculate_total balloons_cost tablecloths_cost streamers_cost banners_cost confetti_cost discount_rate sales_tax_rate amount_received_back = 59.64 :=
by
  sorry

end sara_payment_l736_736399


namespace even_combinations_after_six_operations_l736_736157

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736157


namespace polynomial_remainder_problem_l736_736742

theorem polynomial_remainder_problem:
  let f := λ (x : ℤ) => x^7 + x^5,
      x0 := 1 / 3,
      p1 := polynomial.div_by_x_minus_c f x0,
      s1 := polynomial.rem_by_x_minus_c f x0,
      p2 := polynomial.div_by_x_minus_c p1 x0
  in polynomial.rem_by_x_minus_c p2 x0 = 0 :=
by
  sorry

end polynomial_remainder_problem_l736_736742


namespace exists_subset_sum_2n_l736_736368

theorem exists_subset_sum_2n (n : ℕ) (h : n > 3) (s : Finset ℕ)
  (hs : ∀ x ∈ s, x < 2 * n) (hs_card : s.card = 2 * n)
  (hs_sum : s.sum id = 4 * n) :
  ∃ t ⊆ s, t.sum id = 2 * n :=
by sorry

end exists_subset_sum_2n_l736_736368


namespace sequence_explicit_formula_l736_736773

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end sequence_explicit_formula_l736_736773


namespace case1_case2_case3_l736_736408

-- Define the inequality problem as a predicate
def inequality (a x : ℝ) := a * x * x + (a - 2) * x - 2 >= 0

-- Prove statement for the three cases
theorem case1 (a : ℝ) (h : -2 < a ∧ a < 0) : ∀ x : ℝ, ¬inequality a x :=
by sorry

theorem case2 (a : ℝ) (h : a = -2) : ∃ x : ℝ, equality (x = -1) ∧ inequality a x :=
by sorry

theorem case3 (a : ℝ) (h : a < -2) : ∀ x : ℝ, ¬inequality a x :=
by sorry

end case1_case2_case3_l736_736408


namespace total_symmetric_scanning_codes_l736_736970

def symmetric_scanning_codes (G : Matrix (Fin 5) (Fin 5) Bool) : Prop :=
  ∃b w, b ≠ w ∧
  (∀ i j, G i j = G j (4 - i)) ∧ 
  (∀ i j, G i j = G (4 - i) j) ∧
  (∀ i j, G i j = G (4 - j) (4 - i)) ∧
  (∀ i j, G i j = G (4 - j) (4 - i))

theorem total_symmetric_scanning_codes :
  ∃ (G : Matrix (Fin 5) (Fin 5) Bool),
    symmetric_scanning_codes G = 30 :=
sorry

end total_symmetric_scanning_codes_l736_736970


namespace suff_and_not_necessary_l736_736631

theorem suff_and_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) :
  (|a| > |b|) ∧ (¬(∀ x y : ℝ, (|x| > |y|) → (x > y ∧ y > 0))) :=
by
  sorry

end suff_and_not_necessary_l736_736631


namespace remainder_div_3973_28_l736_736908

theorem remainder_div_3973_28 : (3973 % 28) = 9 := by
  sorry

end remainder_div_3973_28_l736_736908


namespace partition_convex_hulls_equal_vertices_l736_736620

def is_even (n : Nat) : Prop := n % 2 = 0

def no_three_collinear (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p3 ∈ S →
    (p1.1 - p2.1) * (p1.2 - p3.2) ≠ (p1.2 - p2.2) * (p1.1 - p3.1)

theorem partition_convex_hulls_equal_vertices
  (S : Set (ℝ × ℝ))
  (h_even : is_even (S.card))
  (h_no_three_collinear : no_three_collinear S) :
  ∃ (X Y : Set (ℝ × ℝ)), 
  (X ∪ Y = S) ∧ (X ∩ Y = ∅) ∧
  (convexHull ℝ X).card = (convexHull ℝ Y).card :=
sorry

end partition_convex_hulls_equal_vertices_l736_736620


namespace intersection_point_l736_736176

noncomputable def line1 (t : ℝ) : AffineSpace.Point ℝ ℝ :=
  ⟨(2 + 3 * t), (3 + 4 * t)⟩

noncomputable def line2 (u : ℝ) : AffineSpace.Point ℝ ℝ :=
  ⟨(6 + 5 * u), (1 - u)⟩

theorem intersection_point : ∃ t u : ℝ, line1 t = line2 u :=
begin
  use [-6/23, -22/23],
  sorry
end

example : line1 (-6/23) = line2 (-22/23) := by
  have h := intersection_point,
  exact (Classical.choose_spec h)

end intersection_point_l736_736176


namespace hunter_B_more_success_prob_l736_736084

open ProbabilityTheory

noncomputable def binomial_distribution := BernoulliProcess 0.5

def probability_hunter_Bs_catch_exceeds_A : ℕ → ℕ → ℚ := 
  fun (nA nB : ℕ) =>
    ∑ kA in finset.range (nA + 1), 
    ∑ kB in finset.range (nB + 1), 
    if kB > kA then 
      (binomial_distribution.prob nA kA) * (binomial_distribution.prob nB kB) else 0

theorem hunter_B_more_success_prob : probability_hunter_Bs_catch_exceeds_A 50 51 = 1 / 2 :=
begin
  sorry
end

end hunter_B_more_success_prob_l736_736084


namespace largest_num_is_num2_smallest_num_is_num5_l736_736985

def num1 := 3 ^ 0.4
def num2 := (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180))
def num3 := Real.log 2 3 * Real.log 9 8
def num4 := 5 ^ (-0.2)
def num5 := (-3) ^ (1 / 3)

theorem largest_num_is_num2 : num2 = max num1 (max num2 (max num3 (max num4 num5))) := 
sorry

theorem smallest_num_is_num5 : num5 = min num1 (min num2 (min num3 (min num4 num5))) := 
sorry

end largest_num_is_num2_smallest_num_is_num5_l736_736985


namespace complex_conjugate_sum_l736_736714

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736714


namespace guess_hidden_number_l736_736466

-- Definitions based on conditions
def total_sum := 110

def matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![[10, 23, 16, 29, 32],
    [27, 15, 28, 31, 9],
    [14, 32, 30, 8, 26],
    [36, 24, 12, 25, 13],
    [23, 16, 24, 17, 30]]

-- The main Lean 4 statement based on the identified question and correct answer
theorem guess_hidden_number (r c : Fin 5) (sum_remaining : ℕ) :
  (matrix.row r).sum = total_sum ∧ (matrix.column c).sum = total_sum →
  ((matrix.row r).sum - sum_remaining = matrix r c) ∨ ((matrix.column c).sum - sum_remaining = matrix r c) :=
by
  intros h;
  sorry  -- Proof to be filled in later

end guess_hidden_number_l736_736466


namespace cauchy_bunyakovsky_l736_736392

theorem cauchy_bunyakovsky (n : ℕ) (a b : Fin n → ℝ) (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  ∑ i, a i * b i ≤ Real.sqrt (∑ i, (a i)^2) * Real.sqrt (∑ i, (b i)^2) ∧ 
  (∑ i, a i * b i = Real.sqrt (∑ i, (a i)^2) * Real.sqrt (∑ i, (b i)^2) ↔ ∃ k : ℝ, ∀ i, b i = k * a i) :=
sorry

end cauchy_bunyakovsky_l736_736392


namespace exponential_function_passes_through_point_l736_736911

noncomputable def f (a : ℝ) (x : ℝ) := a^(x-1) + 3

theorem exponential_function_passes_through_point (a : ℝ) (h1: a > 0) (h2: a ≠ 1) : f a 1 = 4 :=
by {
  -- proof will be written here
  sorry,
}

end exponential_function_passes_through_point_l736_736911


namespace probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736820

-- Question (a) - Probability of discovering the empty bottle on March 14
theorem probability_of_empty_bottle_on_march_14 :
  let ways_to_choose_10_out_of_13 := Nat.choose 13 10
  ∧ let probability_sequence := (1/2) ^ 13
  ∧ let probability_pick_empty_on_day_14 := 1 / 2
  in 
  (286 / 8192 = 0.035) := sorry

-- Question (b) - Expected number of pills taken by the time of first discovery
theorem expected_number_of_pills_first_discovery :
  let expected_value := Σ k in 10..20, (k * (Nat.choose (k-1) 3) * (1/2) ^ k)
  ≈ 17.3 := sorry

end probability_of_empty_bottle_on_march_14_expected_number_of_pills_first_discovery_l736_736820


namespace gas_price_increase_l736_736873

noncomputable def additional_increase (initial_increase first_second_third_increase total_year_increase : ℚ) : ℚ :=
  (total_year_increase / initial_increase) - 1

theorem gas_price_increase :
  let initial : ℚ := 1
  let first_increase : ℚ := 1 + 5 / 100
  let second_increase : ℚ := first_increase * (1 + 6 / 100)
  let third_increase : ℚ := second_increase * (1 + 10 / 100)
  let total_increase : ℚ := 4 / 3
  show additional_increase third_increase total_increase ≈ 8.91 / 100 :=
by sorry

end gas_price_increase_l736_736873


namespace triplet_sum_not_one_l736_736106

def sum_triplet (a b c : ℝ) : ℝ := a + b + c

def triplet_A : (ℝ × ℝ × ℝ) := (1/4, 2/4, 1/4)
def triplet_B : (ℝ × ℝ × ℝ) := (-3, 5, -1)
def triplet_C : (ℝ × ℝ × ℝ) := (0.2, 0.4, 0.4)
def triplet_D : (ℝ × ℝ × ℝ) := (0.9, -0.5, 0.6)

theorem triplet_sum_not_one :
  ¬ (sum_triplet triplet_D.1 triplet_D.2 triplet_D.3 = 1) :=
by 
  sorry

end triplet_sum_not_one_l736_736106


namespace least_clock_equivalent_l736_736838

-- Define the predicate to check the "clock equivalent" condition
def clock_equivalent (h : ℕ) : Prop := (h * h) % 24 = h % 24

theorem least_clock_equivalent :
  ∃ h, h > 2 ∧ clock_equivalent h ∧ (∀ k, k > 2 ∧ clock_equivalent k → h ≤ k) :=
begin
  use 5,
  split,
  { norm_num, },
  split,
  { unfold clock_equivalent,
    norm_num, },
  { intros k hk1 hk2,
    cases k,
    { norm_num at hk1, },
    { cases k,
      { norm_num at hk1, },
      { cases k,
        { norm_num at hk1, },
        { cases k,
          { norm_num at hk1, },
          { cases k,
            { norm_num at hk1, },
            { cases k,
              { norm_num at hk1, },
              { cases k,
                { norm_num at hk1, },
                { cases k,
                  { norm_num at hk1, },
                  { cases k,
                    { norm_num at hk1, },
                    { cases k,
                      { norm_num at hk1, },
                      { cases k,
                        { norm_num at hk1, },
                        { cases k,
                          { norm_num at hk1, },
                          { cases k,
                            { norm_num at hk1, },
                            { cases k,
                              { norm_num at hk1, },
                              { cases k,
                                { norm_num at hk1, },
                                { cases k,
                                  { norm_num at hk1, },
                                  { cases k,
                                    { norm_num at hk1, },
                                    { cases k,
                                      { norm_num at hk1, },
                                      { cases k,
                                        { norm_num at hk1, },
                                        { norm_num,
                                          sorry, } } } } } } } } } } } } } } } } }
end

end least_clock_equivalent_l736_736838


namespace cards_sum_divisible_by_3_l736_736881

theorem cards_sum_divisible_by_3 :
  let numbers := (Finset.range 140).image (λ n => 4 * (n + 1))
  (Finset.card (numbers.powerset.filter (λ s => s.card = 3 ∧ s.sum % 3 = 0)) = 149224) :=
by
  let numbers := (Finset.range 140).image (λ n => 4 * (n + 1))
  have h : Finset.card (numbers.powerset.filter (λ s => s.card = 3 ∧ s.sum % 3 = 0)) = 149224 := sorry
  exact h

end cards_sum_divisible_by_3_l736_736881


namespace net_rate_per_hour_l736_736514

-- Definitions and conditions given in the problem statement
def travel_time : ℝ := 3 -- hours
def speed : ℝ := 65 -- miles per hour
def gas_efficiency : ℝ := 28 -- miles per gallon
def payment_per_mile : ℝ := 0.55 -- dollars per mile
def gas_cost_per_gallon : ℝ := 2.5 -- dollars per gallon

-- The theorem to be proven
theorem net_rate_per_hour : 
  let total_distance := speed * travel_time in
  let total_gasoline_used := total_distance / gas_efficiency in
  let total_earnings := payment_per_mile * total_distance in
  let total_gas_cost := gas_cost_per_gallon * total_gasoline_used in
  let net_earnings := total_earnings - total_gas_cost in
  let net_rate_per_hour := net_earnings / travel_time in
  net_rate_per_hour = 30 := 
by 
  sorry

end net_rate_per_hour_l736_736514


namespace nearest_integer_to_expansion_l736_736221

theorem nearest_integer_to_expansion : 
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  abs (a^4 - 1090) < 1 :=
by
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  sorry

end nearest_integer_to_expansion_l736_736221


namespace number_of_friends_l736_736335

theorem number_of_friends (pairs_of_shoes : ℕ) (victoria_shoes : ℕ) (shoes_per_person : ℕ) (h_pairs : pairs_of_shoes = 36) (h_victoria : victoria_shoes = 2) (h_shoes_per_person : shoes_per_person = 2) : 
  let total_shoes := pairs_of_shoes * 2 in
  let shoes_for_friends := total_shoes - victoria_shoes in
  let friends := shoes_for_friends / shoes_per_person in
  friends = 35 :=
by
  sorry

end number_of_friends_l736_736335


namespace max_triangle_perimeter_l736_736547

theorem max_triangle_perimeter (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 7 + 9 + y ≤ 31 :=
by
  -- proof goes here
  sorry

end max_triangle_perimeter_l736_736547


namespace circumradius_of_given_triangle_equals_sum_of_smaller_circumradii_l736_736888

variable (a b c : ℝ) (R₁ R₂ R₃ : ℝ)

def semi_perimeter : ℝ := (a + b + c) / 2

def circumradius_of_triangle : ℝ :=
R₁ + R₂ + R₃

theorem circumradius_of_given_triangle_equals_sum_of_smaller_circumradii
  (R : ℝ)
  (hR : R = circumradius_of_triangle R₁ R₂ R₃) :
  R = R₁ + R₂ + R₃ :=
by
  rw [hR]
  sorry

end circumradius_of_given_triangle_equals_sum_of_smaller_circumradii_l736_736888


namespace milk_replacement_l736_736195

theorem milk_replacement : 
  ∀ (x : ℕ), 
  (let initial_milk := 30 in 
   let final_milk := 14.7 in 
   let milk_left_after_first := initial_milk - x in
   let milk_removed_second := (x * milk_left_after_first) / initial_milk in
   let milk_final_quantity := initial_milk - x - milk_removed_second in
   milk_final_quantity = final_milk) → x = 9 :=
by sorry

end milk_replacement_l736_736195


namespace opposite_of_neg_three_l736_736449

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l736_736449


namespace max_n_for_positive_sum_l736_736645

theorem max_n_for_positive_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith: ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2)
  (h_cond1: a 9 + a 12 < 0)
  (h_cond2: a 10 * a 11 < 0)
  : ∀ n, S 19 > 0 ∧ S 20 < 0 → n ≤ 19 :=
begin
  sorry
end

end max_n_for_positive_sum_l736_736645


namespace find_k_l736_736671

def system_of_equations (x y k : ℝ) : Prop :=
  x - y = k - 3 ∧
  3 * x + 5 * y = 2 * k + 8 ∧
  x + y = 2

theorem find_k (x y k : ℝ) (h : system_of_equations x y k) : k = 1 := 
sorry

end find_k_l736_736671


namespace complex_conjugate_sum_l736_736712

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736712


namespace proof_system_solution_l736_736004

noncomputable def solve_system_of_equations : Prop :=
  ∃ (x y z : ℝ), 
    √(x^2 - 2*x + 6) * Real.logBase 3 (6 - y) = x ∧
    √(y^2 - 2*y + 6) * Real.logBase 3 (6 - z) = y ∧
    √(z^2 - 2*z + 6) * Real.logBase 3 (6 - x) = z ∧
    x = y ∧ y = z ∧ x ≈ 2.5

theorem proof_system_solution : solve_system_of_equations :=
  sorry

end proof_system_solution_l736_736004


namespace person_died_at_33_l736_736941

-- Define the conditions and constants
def start_age : ℕ := 25
def insurance_payment : ℕ := 10000
def premium : ℕ := 450
def loss : ℕ := 1000
def annual_interest_rate : ℝ := 0.05
def half_year_factor : ℝ := 1.025 -- half-yearly compounded interest factor

-- Calculate the number of premium periods (as an integer)
def n := 16 -- (derived from the calculations in the given solution)

-- Define the final age based on the number of premium periods
def final_age : ℕ := start_age + (n / 2)

-- The proof statement
theorem person_died_at_33 : final_age = 33 := by
  sorry

end person_died_at_33_l736_736941


namespace bounded_set_condition_l736_736607

variables {α : Type*} [linear_ordered_field α]

/-- Prove that for k > 1, there exists a bounded set of positive real numbers S with at least 
3 elements such that for all a, b in S with a > b, k(a - b) ∈ S if and only if k = 2 
or k = (1 + sqrt(5))/2. -/
theorem bounded_set_condition (k : α) (h_k : k > 1) :
  (∃ S : set α, (∃ M > 0, ∀ x ∈ S, x < M) ∧ (S ⊆ set_of (λ x, 0 < x)) ∧ (S.card ≥ 3) ∧ 
  ∀ a b ∈ S, a > b → k * (a - b) ∈ S) ↔ (k = 2 ∨ k = (1 + real.sqrt 5) / 2) :=
sorry

end bounded_set_condition_l736_736607


namespace problem_statement_l736_736455

noncomputable def sequence (n : ℕ) : ℚ := 
  if n = 0 then 1
  else sorry -- since the precise definition is recursive and complex as per problem conditions

def least_integer_k (a : ℕ → ℚ) : ℕ :=
  Nat.find (λ k, k > 1 ∧ ∃ (m : ℤ), a k = m)

theorem problem_statement : least_integer_k sequence = 41 := 
sorry

end problem_statement_l736_736455


namespace gwen_earned_points_l736_736677

-- Define the points for the first bag and the percentage increase
def points_per_bag : ℕ := 8
def percentage_increase : ℝ := 0.10

-- Calculate the points for the second bag
noncomputable def points_second_bag : ℕ :=
  nat.ceil ((points_per_bag : ℝ) * (1 + percentage_increase))

-- Calculate the total points for two bags
noncomputable def total_points (n : ℕ) : ℕ :=
  if n = 1 then points_per_bag
  else points_per_bag + points_second_bag

-- Given the conditions
def bags_recycled : ℕ := 2

-- The theorem to prove
theorem gwen_earned_points : total_points bags_recycled = 17 :=
by sorry

end gwen_earned_points_l736_736677


namespace math_problem_modulo_l736_736206

theorem math_problem_modulo :
    (245 * 15 - 20 * 8 + 5) % 17 = 1 := 
by
  sorry

end math_problem_modulo_l736_736206


namespace proof_number_of_correct_statements_l736_736689

noncomputable def number_of_correct_statements (α β : Real) (h : α + β = Real.pi) : Nat := 
  if (Real.sin α = Real.sin β) ∧ 
     (Real.sin α ≠ -Real.sin β) ∧ 
     (Real.cos α = -Real.cos β) ∧ 
     (Real.cos α ≠ Real.cos β) α 
     (Real.tan α = -Real.tan β) then 3
  else 0

theorem proof_number_of_correct_statements (α β : Real) (h : α + β = Real.pi) : 
  number_of_correct_statements α β h = 3 :=
sorry

end proof_number_of_correct_statements_l736_736689


namespace courtyard_length_l736_736685

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

end courtyard_length_l736_736685


namespace percentage_equivalence_l736_736746

-- Define the conditions given in the problem
def seventy_five_percent_of_six_hundred := 600 * 0.75
def percentage_of_nine_hundred (x : ℝ) := (x / 900) * 100

-- State the problem as a Lean theorem
theorem percentage_equivalence : percentage_of_nine_hundred seventy_five_percent_of_six_hundred = 50 := 
sorry

end percentage_equivalence_l736_736746


namespace joint_purchases_popular_in_countries_joint_purchases_not_popular_among_neighbours_l736_736937

theorem joint_purchases_popular_in_countries 
    (risks : Prop) 
    (cost_savings : Prop) 
    (info_sharing : Prop)
    (quality_assessment : Prop)
    (willingness_to_share : Prop)
    : (cost_savings ∧ info_sharing) → risks → ∀ country, practice_of_joint_purchases_popular country :=
by
  intros h1 h2 country
  -- Proof required here.
  sorry

theorem joint_purchases_not_popular_among_neighbours 
    (transactional_costs : Prop) 
    (coordination_challenges : Prop) 
    (necessary_compensation : Prop)
    (proximity_to_stores : Prop)
    (disputes : Prop)
    : (transactional_costs ∧ coordination_challenges ∧ necessary_compensation ∧ proximity_to_stores ∧ disputes) 
    → ∀ (neighbours : Type), ¬ practice_of_joint_purchases_popular_for groceries neighbours :=
by
  intros h1 neighbours
  -- Proof required here.
  sorry

end joint_purchases_popular_in_countries_joint_purchases_not_popular_among_neighbours_l736_736937


namespace product_of_undefined_x_l736_736603

theorem product_of_undefined_x : 
  let f := λ x : ℝ, (x^3 + 3*x^2 + 3*x + 1) / (x^2 + 3*x - 4) in
  ∃ x1 x2 : ℝ, (x^2 + 3*x - 4 = 0) ∧ (x1 ≠ x2) ∧ (-4) = x1 * x2 :=
by 
  sorry

end product_of_undefined_x_l736_736603


namespace hex_351_is_849_l736_736190

noncomputable def hex_to_decimal : ℕ := 1 * 16^0 + 5 * 16^1 + 3 * 16^2

-- The following statement is the core of the proof problem
theorem hex_351_is_849 : hex_to_decimal = 849 := by
  -- Here the proof steps would normally go
  sorry

end hex_351_is_849_l736_736190


namespace number_of_people_l736_736842

theorem number_of_people (total_eggs : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : 
  total_eggs = 36 → eggs_per_omelet = 4 → omelets_per_person = 3 → 
  (total_eggs / eggs_per_omelet) / omelets_per_person = 3 :=
by
  intros h1 h2 h3
  sorry

end number_of_people_l736_736842


namespace dice_probability_l736_736169

theorem dice_probability :
  let die1_faces := (List.range 19).map (λ x => x + 1) ++ [0] in
  let die2_faces := (List.range 7).map (λ x => x + 1) ++ (List.range 13).map (λ x => x + 9) ++ [0] in
  let valid_pairs := List.filter (λ p : ℕ × ℕ => p.1 + p.2 = 26) (List.product die1_faces die2_faces) in
  (valid_pairs.length : ℚ) / (die1_faces.length * die2_faces.length : ℚ) = 13 / 400 :=
by
  sorry

end dice_probability_l736_736169


namespace coin_flip_sequences_l736_736512

theorem coin_flip_sequences : 
  let flips := 10
  let choices := 2
  let total_sequences := choices ^ flips
  total_sequences = 1024 :=
by
  sorry

end coin_flip_sequences_l736_736512


namespace area_of_triangle_WRX_l736_736764

variables {W X Y Z P Q R S : Type*}
variables [normedAddCommGroup W] [normedAddCommGroup X] [normedAddCommGroup Y]
variables (WZ XY YP QZ PQ RS : ℝ)

-- Definitions from conditions
def Rectangle_WXYZ := (WZ = 7) ∧ (XY = 4)
def Points_PQ_on_YZ := (YP = 2) ∧ (QZ = 3)
def Lines_WP_XQ_intersect_R := true  -- Placeholder, specific intersection details omitted

-- The theorem stating the problem
theorem area_of_triangle_WRX :
  Rectangle_WXYZ WZ XY →
  Points_PQ_on_YZ YP QZ →
  Lines_WP_XQ_intersect_R →
  (RS = 8 / 5) →
  (1 / 2) * ((RS + 4) * 7) = 98 / 5 := 
sorry

end area_of_triangle_WRX_l736_736764


namespace number_of_even_results_l736_736133

def valid_operations : List (ℤ → ℤ) := [λ x => x + 2, λ x => x + 3, λ x => x * 2]

def apply_operations (start : ℤ) (ops : List (ℤ → ℤ)) : ℤ :=
  ops.foldl (λ x op => op x) start

def is_even (n : ℤ) : Prop := n % 2 = 0

theorem number_of_even_results :
  (Finset.card (Finset.filter (λ ops => is_even (apply_operations 1 ops))
    (Finset.univ.image (λ f : Fin 6 → Fin 3 => List.map (λ i => valid_operations.get i.val) (List.of_fn f))))) = 486 := by
  sorry

end number_of_even_results_l736_736133


namespace problem_statement_l736_736791

variables {p q r s : ℝ}

theorem problem_statement 
  (h : (p - q) * (r - s) / (q - r) * (s - p) = 3 / 7) : 
  (p - r) * (q - s) / (p - q) * (r - s) = -4 / 3 :=
by sorry

end problem_statement_l736_736791


namespace number_of_fish_that_die_each_year_l736_736082

variable (d : ℕ)

theorem number_of_fish_that_die_each_year :
  let fish_after_n_years (n : ℕ) := 2 + n * 2 - n * d in
  fish_after_n_years 5 = 7 → d = 1 :=
by sorry

end number_of_fish_that_die_each_year_l736_736082


namespace even_number_combinations_l736_736153

def operation (x : ℕ) (op : ℕ) : ℕ :=
  match op with
  | 0 => x + 2
  | 1 => x + 3
  | 2 => x * 2
  | _ => x

def apply_operations (x : ℕ) (ops : List ℕ) : ℕ :=
  ops.foldl operation x

def is_even (n : ℕ) : Bool :=
  n % 2 = 0

def count_even_results (num_ops : ℕ) : ℕ :=
  let ops := [0, 1, 2]
  let all_combos := List.replicateM num_ops ops
  all_combos.count (λ combo => is_even (apply_operations 1 combo))

theorem even_number_combinations : count_even_results 6 = 486 :=
  by sorry

end even_number_combinations_l736_736153


namespace recurring_decimal_to_fraction_l736_736476

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l736_736476


namespace part_1_trajectory_eqn_part_2_range_k_over_MN_l736_736617

--(1) Equation of the trajectory C of point P
theorem part_1_trajectory_eqn (P F : ℝ × ℝ) (l_1 : ℝ → ℝ) (hF : F = (1, 0)) (hl1 : ∀ x y : ℝ, l_1 x = -1) (hP : ∀ x y : ℝ, ∃ P, (dist P F = dist (fst P) -1)) :
  ∃ C, (C = y^2 = 4 * x) :=
by
  sorry

--(2) Range of values for |k| / |MN|
theorem part_2_range_k_over_MN (P M N : ℝ × ℝ) (hM : M.1 = -1) (hN : N.1 = -1) (inscribed_circle_eq : ∃ r, r^2 = 1) (k : ℝ) :
  0 < abs k / dist M N ∧ abs k / dist M N < 1/2 :=
by
  sorry

end part_1_trajectory_eqn_part_2_range_k_over_MN_l736_736617


namespace campers_difference_l736_736126

theorem campers_difference:
  let morning_campers := 33 in
  let afternoon_campers := 34 in
  let evening_campers := 10 in
  afternoon_campers - evening_campers = 24 :=
by
  sorry

end campers_difference_l736_736126


namespace rational_exponent_simplification_l736_736265

theorem rational_exponent_simplification (x : ℝ) (hx : x ≠ 0) (h : x^(1 / 2) + x^(-1 / 2) = 3) :
  (x + x^(-1) + 3) / (x^2 + x^(-2) - 2) = 2 / 9 :=
by
  sorry

end rational_exponent_simplification_l736_736265


namespace part1_part2_l736_736302

-- Definition of the system of linear equations
variables (m x y : ℝ)

-- Define the conditions
def eq1 := x + y = 3 * m
def eq2 := 2 * x - 3 * y = m + 5

-- Define the first proof problem (solutions are positive)
theorem part1 (h : eq1 ∧ eq2) : x > 0 ∧ y > 0 → m > 1 := by
  sorry

-- Define the second proof problem (x - y is not less than 0)
theorem part2 (h : eq1 ∧ eq2) : x - y ≥ 0 → m ≥ -2 := by
  sorry

end part1_part2_l736_736302


namespace max_value_of_quadratic_in_interval_l736_736504

theorem max_value_of_quadratic_in_interval :
  ∃ x ∈ Icc (-1 : ℝ) 1, ∀ y ∈ Icc (-1 : ℝ) 1, f y ≤ f x ∧ f x = 4 := by
  let f := λ x : ℝ, x^2 + 4 * x + 1
  have h : ∀ x ∈ Icc (-1 : ℝ) 1, f x = 4 → x = 1 ∨ x = -1 := sorry
  use 1
  split
  { exact ⟨le_of_lt zero_lt_one, le_refl 1⟩ }
  { intros y hy
    have h' : y ∈ Icc (-1 : ℝ) 1 := hy
    sorry }


end max_value_of_quadratic_in_interval_l736_736504


namespace opposite_of_neg3_l736_736445

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l736_736445


namespace fixed_point_of_log_function_l736_736864

theorem fixed_point_of_log_function (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ 
  ∀ x : ℝ, f x = log a (2 * x - 3) + 1 :=
begin
  let P := (2 : ℝ, 1 : ℝ),
  use P,
  split,
  { refl },
  { intros x,
    sorry
  }
end

end fixed_point_of_log_function_l736_736864


namespace find_x_l736_736309

theorem find_x (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (3, x)) (h : (a.fst * b.fst + a.snd * b.snd) = 3) : x = 3 :=
by
  sorry

end find_x_l736_736309


namespace triangle_area_small_enough_l736_736905

theorem triangle_area_small_enough (points : Fin 201 → (ℝ × ℝ)) 
  (h_in_grid : ∀ i, (0 ≤ (points i).1 ∧ (points i).1 ≤ 10) ∧ 
                    (0 ≤ (points i).2 ∧ (points i).2 ≤ 10)) 
  : ∃ (i j k : Fin 201), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    (let (x1, y1) := points i in
     let (x2, y2) := points j in
     let (x3, y3) := points k in
     abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2 < 0.5) :=
sorry

end triangle_area_small_enough_l736_736905


namespace probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736814

-- Define the conditions
def initial_pills_per_bottle := 10
def starting_day := 1
def check_day := 14

-- Part (a) Theorem
/-- The probability that on March 14, the Mathematician finds an empty bottle for the first time is 143/4096. -/
noncomputable def probability_empty_bottle_on_march_14 (initial_pills: ℕ) (check_day: ℕ) : ℝ := 
  if check_day = 14 then
    let C := (fact 13) / ((fact 10) * (fact 3)); 
    2 * C * (1 / (2^13)) * (1 / 2)
  else
    0

-- Proof for Part (a)
theorem probability_empty_bottle_march_14 : probability_empty_bottle_on_march_14 initial_pills_per_bottle check_day = 143 / 4096 :=
  by sorry

-- Part (b) Theorem
/-- The expected number of pills taken by the Mathematician by the time he finds an empty bottle is 17.3. -/
noncomputable def expected_pills_taken (initial_pills: ℕ) (total_days: ℕ) : ℝ :=
  ∑ k in (finset.range (20 + 1)).filter (λ k, k ≥ 10), 
  k * ((nat.choose k (k - 10)) * (1 / 2^k))

-- Proof for Part (b)
theorem expected_pills_taken_by_empty_bottle : expected_pills_taken initial_pills_per_bottle (check_day + 7) = 17.3 :=
  by sorry

end probability_empty_bottle_march_14_expected_pills_taken_by_empty_bottle_l736_736814


namespace ellen_legos_l736_736591

theorem ellen_legos : ∀ (orig lost remaining : ℕ), orig = 2080 → lost = 17 → remaining = orig - lost → remaining = 2063 :=
by
  intros orig lost remaining h_orig h_lost h_remaining
  have h_result : orig - lost = 2080 - 17 := by 
    rw [h_orig, h_lost]
  rw h_remaining at h_result
  exact h_result

end ellen_legos_l736_736591


namespace correct_propositions_l736_736306

-- Define objects: lines and planes.
variables (m n : Type) (alpha beta : Type)

-- Define relations: parallel and perpendicular.
variables (parallel : m → n → Prop) (perpendicular : m → alpha → Prop)
variables (parallel_planes : alpha → beta → Prop) (subset : m → alpha → Prop)

-- Define propositions
def prop1 : Prop :=
  parallel m n ∧ (perpendicular m alpha → perpendicular n alpha)

def prop2 : Prop :=
  parallel_planes alpha beta ∧ subset m alpha ∧ subset n beta → parallel m n

def prop3 : Prop :=
  parallel m n ∧ (parallel m alpha → parallel n alpha)

def prop4 : Prop :=
  parallel_planes alpha beta ∧ parallel m n ∧ perpendicular m alpha → perpendicular n beta

-- Prove propositions ① and ④ are true
theorem correct_propositions (parallel_mn : parallel m n) 
                            (perpendicular_ma : perpendicular m alpha) :
                            -- Proposition ①
                            prop1 ∧ 
                            -- Proposition ④
                            prop4 :=
sorry

end correct_propositions_l736_736306


namespace pine_count_25_or_26_l736_736062

-- Define the total number of trees
def total_trees : ℕ := 101

-- Define the constraints
def poplar_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 1 → t.nth i = some 1 → t.nth (i + 1) ≠ some 1

def birch_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 2 → t.nth i = some 2 → t.nth (i + 1) ≠ some 2 ∧ t.nth (i + 2) ≠ some 2

def pine_spacing (t : List ℕ) : Prop := 
  ∀ i, i < total_trees - 3 → t.nth i = some 3 → t.nth (i + 1) ≠ some 3 ∧ t.nth (i + 2) ≠ some 3 ∧ t.nth (i + 3) ≠ some 3

-- Define the number of pines
def number_of_pines (t : List ℕ) : ℕ := t.countp (λ x, x = 3)

-- The main theorem asserting the number of pines possible
theorem pine_count_25_or_26 (t : List ℕ) (htotal : t.length = total_trees) 
    (hpoplar : poplar_spacing t) (hbirch : birch_spacing t) (hpine : pine_spacing t) :
  number_of_pines t = 25 ∨ number_of_pines t = 26 := 
sorry

end pine_count_25_or_26_l736_736062


namespace mutually_exclusive_not_complementary_l736_736610

-- Define the events
def event_one_black_ball : set (finset (fin 4)) :=
  {s | s.card = 2 ∧ s.filter (λ x, x < 2).card = 1}

def event_two_black_balls : set (finset (fin 4)) :=
  {s | s.card = 2 ∧ s.filter (λ x, x < 2).card = 2}

-- Prove that the events are mutually exclusive but not complementary
theorem mutually_exclusive_not_complementary :
  (∀ s, s ∈ event_one_black_ball → ¬(s ∈ event_two_black_balls)) ∧
  (∃ s, ¬(s ∈ event_one_black_ball) ∧ ¬(s ∈ event_two_black_balls)) :=
by {
  sorry
}

end mutually_exclusive_not_complementary_l736_736610


namespace midpoint_of_BF_l736_736559

theorem midpoint_of_BF {O A B P K D E F : ℝ} 
  (h1 : tangent P A O) 
  (h2 : tangent P B O)
  (h3 : point_on_circle K O) 
  (h4 : perp BD OK D)
  (h5 : intersect_at BD KP E)
  (h6 : intersect_at BD KA F) : midpoint E B F :=
sorry

end midpoint_of_BF_l736_736559


namespace exists_2009_integers_with_gcd_condition_l736_736587

theorem exists_2009_integers_with_gcd_condition : 
  ∃ (S : Finset ℕ), S.card = 2009 ∧ (∀ x ∈ S, ∀ y ∈ S, x ≠ y → |x - y| = Nat.gcd x y) :=
sorry

end exists_2009_integers_with_gcd_condition_l736_736587


namespace z_conjugate_sum_l736_736719

-- Definitions based on the condition from the problem
def z : ℂ := 1 + Complex.i

-- Proof statement
theorem z_conjugate_sum (z : ℂ) (h : Complex.i * (1 - z) = 1) : z + Complex.conj z = 2 :=
sorry

end z_conjugate_sum_l736_736719


namespace solve_trig_eq_l736_736003

theorem solve_trig_eq (x : ℝ) : 
  (∃ k : ℤ, x = (π / 10) + (k * π / 5))
  ↔ 
  (cos (2 * x) - 3 * cos (4 * x))^2 = 16 + (cos (5 * x))^2 :=
begin
  sorry
end

end solve_trig_eq_l736_736003


namespace spending_during_last_quarter_l736_736022

variable total_spent_by_end_September : ℝ
variable total_spent_by_end_December : ℝ

theorem spending_during_last_quarter 
  (h1 : total_spent_by_end_September = 3.8) 
  (h2 : total_spent_by_end_December = 5) : 
  total_spent_by_end_December - total_spent_by_end_September = 1.2 := 
by
  sorry

end spending_during_last_quarter_l736_736022


namespace remainder_when_five_times_plus_nine_divided_by_seven_l736_736181

theorem remainder_when_five_times_plus_nine_divided_by_seven (n : ℤ) (h : n ≡ 2 [MOD 7]) : (5 * n + 9) ≡ 5 [MOD 7] :=
by sorry

end remainder_when_five_times_plus_nine_divided_by_seven_l736_736181


namespace b2_possible_values_l736_736971

noncomputable def sequence (b : ℕ → ℕ) : Prop := ∀ n, b (n + 2) = (b (n + 1) - b n).natAbs

theorem b2_possible_values :
  ∃ (b : ℕ → ℕ), sequence b ∧ b 1 = 1024 ∧ b 2 < 1024 ∧ b 2010 = 2 ∧ 
  (∑ x in (finset.range 1024).filter (λ x, x % 2 = 0 ∧ (∃ n, n * 2 + 2 = x) ∧ nat.gcd 1024 x = 2), 1) = 256 := 
sorry

end b2_possible_values_l736_736971


namespace cats_remaining_proof_l736_736963

def initial_siamese : ℕ := 38
def initial_house : ℕ := 25
def sold_cats : ℕ := 45

def total_cats (s : ℕ) (h : ℕ) : ℕ := s + h
def remaining_cats (total : ℕ) (sold : ℕ) : ℕ := total - sold

theorem cats_remaining_proof : remaining_cats (total_cats initial_siamese initial_house) sold_cats = 18 :=
by
  sorry

end cats_remaining_proof_l736_736963


namespace find_number_of_x_l736_736968

def count_valid_x_values : ℕ :=
  let condition1 (x : ℕ) := (x + 3) * (x - 4) * (x^2 + 16) < 800
  let condition2 (x : ℕ) := x^2 + 16 > 30 in
  (List.range (800 + 1)).count (λ x => x > 0 ∧ condition1 x ∧ condition2 x)

theorem find_number_of_x :
  count_valid_x_values = 2 :=
sorry

end find_number_of_x_l736_736968


namespace max_perimeter_triangle_l736_736545

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l736_736545


namespace number_of_n_values_l736_736262

-- Definition of sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- The main statement to prove
theorem number_of_n_values : 
  ∃ M, M = 8 ∧ ∀ n : ℕ, (n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2010) → M = 8 :=
by
  sorry

end number_of_n_values_l736_736262


namespace unique_continuous_f_l736_736249

theorem unique_continuous_f {f : ℝ → ℝ} 
  (h1 : ∀ x > 0, 1 ≤ f x) 
  (h2 : continuous f) 
  (h3 : ∀ (n : ℕ) (x : ℝ), (0 < x) → (∏ k in finset.range (n + 1), f (k * x) < 2010 * n ^ 2010)) :
  (∀ x > 0, f x = 1) :=
sorry

end unique_continuous_f_l736_736249


namespace b_investment_l736_736553

theorem b_investment (A_invest C_invest total_profit A_profit x : ℝ) 
(h1 : A_invest = 2400) 
(h2 : C_invest = 9600) 
(h3 : total_profit = 9000) 
(h4 : A_profit = 1125)
(h5 : x = (8100000 / 1125)) : 
x = 7200 := by
  rw [h5]
  sorry

end b_investment_l736_736553


namespace largest_hole_leakage_rate_l736_736101

theorem largest_hole_leakage_rate (L : ℝ) (h1 : 600 = (L + L / 2 + L / 6) * 120) : 
  L = 3 :=
sorry

end largest_hole_leakage_rate_l736_736101


namespace jordyn_total_cost_l736_736324

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end jordyn_total_cost_l736_736324


namespace particle_after_150_moves_l736_736531

noncomputable def omega : ℂ := complex.cis (real.pi / 4)
noncomputable def z_n (n : ℕ) : ℂ :=
  if n = 0 then 5 else omega * z_n (n - 1) + 10

theorem particle_after_150_moves :
  z_n 150 = complex.mk (-5 * real.sqrt 2) (5 + 5 * real.sqrt 2) :=
sorry

end particle_after_150_moves_l736_736531


namespace opposite_of_neg3_l736_736446

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l736_736446


namespace parakeets_per_cage_l736_736964

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (hcages : num_cages = 6) 
  (hparrots : parrots_per_cage = 6) 
  (htotal : total_birds = 48) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := 
  by
  sorry

end parakeets_per_cage_l736_736964


namespace n_gt_2_pow_k_iff_monochromatic_eight_l736_736996

def collinear_points (n : ℕ) : Prop := ∃ P : fin n → ℝ × ℝ, ∀ i j : fin n, i ≠ j → (P i).1 = (P j).1

def circles_with_diameters (n : ℕ) : fin n → fin n → set ℝ × ℝ :=
λ i j, if i < j then { (x, y) | dist (P i) (P j) = 2 * sqrt ((x - (P i).1)^2 + (y - (P i).2)^2) }
             else ∅

def colored_circles (n k : ℕ) : Type := fin (n * (n - 1) / 2) → fin k

def solid_eight (n k : ℕ) (jumble : colored_circles n k) : Prop := 
∃ c : fin k, ∃ i j, ∃ k l, i < j ∧ k < l ∧ jumble ⟨i, j⟩ = c ∧ jumble ⟨k, l⟩ = c ∧
  circles_with_diameters n i j ∩ circles_with_diameters n k l ≠ ∅

theorem n_gt_2_pow_k_iff_monochromatic_eight (n k : ℕ) (hn : collinear_points n) :
  (∃ jumble : colored_circles n k, ¬solid_eight n k jumble) ↔ n ≤ 2^k := sorry

end n_gt_2_pow_k_iff_monochromatic_eight_l736_736996


namespace length_FJ_isosceles_right_triangle_l736_736622

variable (D E F J : Type)

theorem length_FJ_isosceles_right_triangle
  (isosceles_right_triangle : IsIsoscelesRightTriangle D E F ∧ isRightAngle ∠DEF)
  (EF_length : distance E F = 6)
  (incenter_J : isIncenter J (triangle D E F)) : distance F J = 6 - 3 * Real.sqrt 2 :=
  sorry

end length_FJ_isosceles_right_triangle_l736_736622


namespace number_of_pines_l736_736056

theorem number_of_pines (trees : List Nat) :
  (∑ t in trees, 1) = 101 ∧
  (∀ i, ∀ j, trees[i] = trees[j] → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 1))) ∧
  (∀ i, ∀ j, trees[i] = 2 → trees[j] = 2 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 2))) ∧
  (∀ i, ∀ j, trees[i] = 3 → trees[j] = 3 → (i ≠ j) → (∀ k, (i < k ∧ k < j → trees[k] ≠ 3))) →
  (∑ t in trees, if t = 3 then 1 else 0) = 25 ∨ (∑ t in trees, if t = 3 then 1 else 0) = 26 :=
by
  sorry

end number_of_pines_l736_736056


namespace parabola_directrix_l736_736857

theorem parabola_directrix (y : ℝ) : (x : ℝ) (h : x^2 = 4 * y) → y = -1 :=
by
  sorry

end parabola_directrix_l736_736857


namespace smallest_x_with_20_factors_and_factors_18_and_24_exists_l736_736024

theorem smallest_x_with_20_factors_and_factors_18_and_24_exists : 
  ∃ x : ℕ, (20 = (factors x).length) ∧ (18 ∣ x) ∧ (24 ∣ x) ∧ ∀ y : ℕ, (20 = (factors y).length) ∧ (18 ∣ y) ∧ (24 ∣ y) → (y ≥ 480)
:= 
sorry

end smallest_x_with_20_factors_and_factors_18_and_24_exists_l736_736024


namespace sum_of_reciprocals_squares_inequality_l736_736843

theorem sum_of_reciprocals_squares_inequality (n : ℕ) (h : n ≥ 2) :
  (1 / 2) - (1 / (n + 1)) < (Finset.range (n - 1)).sum (λ k, 1 / (k + 2)^2) 
    ∧ (Finset.range (n - 1)).sum (λ k, 1 / (k + 2)^2) < (n - 1) / n :=
by
  sorry

end sum_of_reciprocals_squares_inequality_l736_736843


namespace even_combinations_result_in_486_l736_736166

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736166


namespace log_base_27_of_3_l736_736231

theorem log_base_27_of_3 (h : 27 = 3 ^ 3) : log 27 3 = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736231


namespace complex_conjugate_sum_l736_736728

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736728


namespace compare_a_b_c_compare_explicitly_defined_a_b_c_l736_736270

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end compare_a_b_c_compare_explicitly_defined_a_b_c_l736_736270


namespace range_of_a_l736_736670

theorem range_of_a (A B C : Set ℝ) (a : ℝ) :
  A = { x | -1 < x ∧ x < 4 } →
  B = { x | -5 < x ∧ x < (3 / 2) } →
  C = { x | (1 - 2 * a) < x ∧ x < (2 * a) } →
  (C ⊆ (A ∩ B)) →
  a ≤ (3 / 4) :=
by
  intros hA hB hC hSubset
  sorry

end range_of_a_l736_736670


namespace log_base_27_of_3_l736_736232

theorem log_base_27_of_3 (h : 27 = 3 ^ 3) : log 27 3 = 1 / 3 :=
by
  sorry

end log_base_27_of_3_l736_736232


namespace trains_crossing_time_l736_736118

open Real

-- Define the speed in km/h
def speed_kmh := 80

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℝ) : ℝ :=
  v * 1000 / 3600

-- Define lengths of the trains in meters
def length_train := 100

-- Define the relative speed in m/s
def relative_speed := kmh_to_ms (speed_kmh + speed_kmh)

-- Define the total distance to be covered
def total_distance := 2 * length_train

-- Define the time taken to cross each other completely
def crossing_time : ℝ :=
  total_distance / relative_speed

-- Prove that the crossing time is approximately 4.5 seconds
theorem trains_crossing_time :
  |crossing_time - 4.5| < 0.01 := 
by 
  unfold crossing_time relative_speed total_distance length_train kmh_to_ms speed_kmh
  simp only [*, add_mul, div_mul_div_comm, Real.add, add_self_eq_zero, sub_self]
  norm_num
  sorry

end trains_crossing_time_l736_736118


namespace find_g_of_nine_l736_736423

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_nine (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x) : g 9 = 2 :=
by
  sorry

end find_g_of_nine_l736_736423


namespace rational_sum_zero_l736_736628

theorem rational_sum_zero (x1 x2 x3 x4 : ℚ)
  (h1 : x1 = x2 + x3 + x4)
  (h2 : x2 = x1 + x3 + x4)
  (h3 : x3 = x1 + x2 + x4)
  (h4 : x4 = x1 + x2 + x3) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 := 
sorry

end rational_sum_zero_l736_736628


namespace opposite_of_neg3_is_3_l736_736437

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg3_is_3_l736_736437


namespace opposite_of_neg3_l736_736430

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
sor

end opposite_of_neg3_l736_736430


namespace final_price_l736_736530

def initial_price : ℝ := 200
def discount_morning : ℝ := 0.40
def increase_noon : ℝ := 0.25
def discount_afternoon : ℝ := 0.20

theorem final_price : 
  let price_after_morning := initial_price * (1 - discount_morning)
  let price_after_noon := price_after_morning * (1 + increase_noon)
  let final_price := price_after_noon * (1 - discount_afternoon)
  final_price = 120 := 
by
  sorry

end final_price_l736_736530


namespace probability_second_third_different_colors_l736_736460

def probability_different_colors (blue_chips : ℕ) (red_chips : ℕ) (yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + red_chips + yellow_chips
  let prob_diff :=
    ((blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips)) +
    ((red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips)) +
    ((yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips))
  prob_diff

theorem probability_second_third_different_colors :
  probability_different_colors 7 6 5 = 107 / 162 :=
by
  sorry

end probability_second_third_different_colors_l736_736460


namespace anika_sequence_correct_l736_736006

noncomputable def anika_sequence : ℚ :=
  let s0 := 1458
  let s1 := s0 * 3
  let s2 := s1 / 2
  let s3 := s2 * 3
  let s4 := s3 / 2
  let s5 := s4 * 3
  s5

theorem anika_sequence_correct :
  anika_sequence = (3^9 : ℚ) / 2 := by
  sorry

end anika_sequence_correct_l736_736006


namespace number_of_pines_l736_736071

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l736_736071


namespace find_a_l736_736280

noncomputable def solve_a (a : ℝ) : Prop :=
  let z := 1 / (a - complex.i)
  let p := complex.re z
  let q := complex.im z
  (q = 2 * p) → a = 1 / 2

theorem find_a : solve_a (1 / 2) :=
by sorry

end find_a_l736_736280


namespace cos_B_value_triangle_area_l736_736305

-- Step (1) Defining the variables and constants:
variables {A B C : ℝ} {a b c : ℝ}
axiom triangle_condition : (a - c) ^ 2 = b ^ 2 - (3 / 4) * a * c
axiom b_value : b = Real.sqrt 13
axiom sin_arith_seq : 2 * Real.sin B = Real.sin A + Real.sin C

-- Step (2) Lean statement for problem part (1):
theorem cos_B_value : triangle_condition → cos B = 5 / 8 :=
by sorry

-- Step (3) Lean statement for problem part (2):
theorem triangle_area : triangle_condition → b_value → sin_arith_seq → 
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 39 / 4 :=
by sorry

end cos_B_value_triangle_area_l736_736305


namespace complex_conjugate_sum_l736_736727

theorem complex_conjugate_sum (z : ℂ) (h : complex.I * (1 - z) = 1) : z + complex.conj(z) = 2 :=
sorry

end complex_conjugate_sum_l736_736727


namespace sample_size_l736_736763

theorem sample_size 
  (n : ℕ)
  (ratios : list ℕ := [2, 3, 5, 2, 6]) 
  (largest_ratio : ℕ := 6)
  (sum_ratios : ℕ := 18)
  (contribution : ℕ := 100)
  (h_ratios_sum : ratios.sum = sum_ratios)
  (h_ratio_largest : largest_ratio = ratios.max)
  (h_contribution_eq : largest_ratio * n / sum_ratios = contribution) : 
  n = 300 := 
sorry

end sample_size_l736_736763


namespace white_tiles_count_l736_736362

theorem white_tiles_count (total_tiles yellow_tiles purple_tiles : ℕ)
    (hy : yellow_tiles = 3)
    (hb : ∃ blue_tiles, blue_tiles = yellow_tiles + 1)
    (hp : purple_tiles = 6)
    (ht : total_tiles = 20) : 
    ∃ white_tiles, white_tiles = 7 :=
by
  obtain ⟨blue_tiles, hb_eq⟩ := hb
  let non_white_tiles := yellow_tiles + blue_tiles + purple_tiles
  have hnwt : non_white_tiles = 3 + (3 + 1) + 6,
  {
    rw [hy, hp, hb_eq],
    ring,
  }
  have hwt : total_tiles - non_white_tiles = 7,
  {
    rw ht,
    rw hnwt,
    norm_num,
  }
  use total_tiles - non_white_tiles,
  exact hwt,

end white_tiles_count_l736_736362


namespace sum_of_S_odd_is_n_pow_four_l736_736676

-- Define the sequence of sums 

noncomputable def S : ℕ+ → ℕ
| 1 := 1
| 2 := 5
| 3 := 15
| 4 := 34
| 5 := 65
| 6 := 111
| 7 := 175
-- Continue the pattern...

-- Define the groups (optional, not necessary for the proof)

-- The theorem to prove
theorem sum_of_S_odd_is_n_pow_four (n : ℕ+) : 
  (∑ i in Finset.range n, S (2 * i + 1)) = n^4 := sorry

end sum_of_S_odd_is_n_pow_four_l736_736676


namespace no_common_point_in_all_circles_l736_736769

variable {Point : Type}
variable {Circle : Type}
variable (center : Circle → Point)
variable (contains : Circle → Point → Prop)

-- Given six circles in the plane
variables (C1 C2 C3 C4 C5 C6 : Circle)

-- Condition: None of the circles contain the center of any other circle
axiom condition_1 : ∀ (C D : Circle), C ≠ D → ¬ contains C (center D)

-- Question: Prove that there does not exist a point P that lies in all six circles
theorem no_common_point_in_all_circles : 
  ¬ ∃ (P : Point), (contains C1 P) ∧ (contains C2 P) ∧ (contains C3 P) ∧ (contains C4 P) ∧ (contains C5 P) ∧ (contains C6 P) :=
sorry

end no_common_point_in_all_circles_l736_736769


namespace parts_from_9_blanks_parts_from_14_blanks_blanks_needed_for_40_parts_l736_736756

theorem parts_from_9_blanks : 
  ∀ (initial_blanks : ℕ) (details_from_blanks : ℕ → ℕ) (new_blanks_from_shavings : ℕ → ℕ),
  initial_blanks = 9 →
  (∀ (b : ℕ), details_from_blanks b = b) →
  (∀ (b : ℕ), new_blanks_from_shavings b = b / 3) →
  details_from_blanks 9 + details_from_blanks (new_blanks_from_shavings 9) + details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 9)) = 13 :=
by
  intros initial_blanks details_from_blanks new_blanks_from_shavings h1 h2 h3
  have h4 : details_from_blanks 9 = 9, from h2 9
  have h5 : details_from_blanks (new_blanks_from_shavings 9) = details_from_blanks 3, from congr_arg details_from_blanks (h3 9)
  have h6 : new_blanks_from_shavings 9 = 3, from h3 9
  have h7 : details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 9)) = details_from_blanks 1, from congr_arg details_from_blanks (h3 3)
  have h8 : new_blanks_from_shavings 3 = 1, from h3 3
  have h9 : details_from_blanks 1 = 1, from h2 1
  rw [h4, h6, h5, h8, h7, h9]
  exact rfl

theorem parts_from_14_blanks : 
  ∀ (initial_blanks : ℕ) (details_from_blanks : ℕ → ℕ) (new_blanks_from_shavings : ℕ → ℕ),
  initial_blanks = 14 →
  (∀ (b : ℕ), details_from_blanks b = b) →
  (∀ (b : ℕ), new_blanks_from_shavings b = b / 3) →
  details_from_blanks 12 + details_from_blanks (new_blanks_from_shavings 12) + details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 12)) = 20 :=
by
  intros initial_blanks details_from_blanks new_blanks_from_shavings h1 h2 h3
  have h4 : details_from_blanks 12 = 12, from h2 12
  have h5 : details_from_blanks (new_blanks_from_shavings 12) = details_from_blanks 4, from congr_arg details_from_blanks (h3 12)
  have h6 : new_blanks_from_shavings 12 = 4, from h3 12
  have h7 : details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 12)) = details_from_blanks 1, from congr_arg details_from_blanks (h3 4)
  have h8 : new_blanks_from_shavings 4 = 1, from h3 4
  have h9 : details_from_blanks 2 = details_from_blanks 2, from h2 2
  rw [h4, h6, h5, h8, h7, h9]
  exact rfl

theorem blanks_needed_for_40_parts : 
  ∀ (details_needed : ℕ) (details_from_blanks : ℕ → ℕ) (new_blanks_from_shavings : ℕ → ℕ),
  details_needed = 40 →
  (∀ (b : ℕ), details_from_blanks b = b) →
  (∀ (b : ℕ), new_blanks_from_shavings b = b / 3) →
  details_from_blanks 27 + details_from_blanks (new_blanks_from_shavings 27) + details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 27)) = 40 :=
by
  intros details_needed details_from_blanks new_blanks_from_shavings h1 h2 h3
  have h4 : details_from_blanks 27 = 27, from h2 27
  have h5 : details_from_blanks (new_blanks_from_shavings 27) = details_from_blanks 9, from congr_arg details_from_blanks (h3 27)
  have h6 : new_blanks_from_shavings 27 = 9, from h3 27
  have h7 : details_from_blanks (new_blanks_from_shavings (new_blanks_from_shavings 27)) = details_from_blanks 3, from congr_arg details_from_blanks (h3 9)
  have h8 : new_blanks_from_shavings 9 = 3, from h3 9
  have h9 : details_from_blanks 1 = 1, from h2 1
  rw [h4, h6, h5, h8, h7, h9]
  exact rfl

end parts_from_9_blanks_parts_from_14_blanks_blanks_needed_for_40_parts_l736_736756


namespace even_combinations_after_six_operations_l736_736161

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736161


namespace white_tiles_count_l736_736364

theorem white_tiles_count (total_tiles yellow_tiles purple_tiles : ℕ)
    (hy : yellow_tiles = 3)
    (hb : ∃ blue_tiles, blue_tiles = yellow_tiles + 1)
    (hp : purple_tiles = 6)
    (ht : total_tiles = 20) : 
    ∃ white_tiles, white_tiles = 7 :=
by
  obtain ⟨blue_tiles, hb_eq⟩ := hb
  let non_white_tiles := yellow_tiles + blue_tiles + purple_tiles
  have hnwt : non_white_tiles = 3 + (3 + 1) + 6,
  {
    rw [hy, hp, hb_eq],
    ring,
  }
  have hwt : total_tiles - non_white_tiles = 7,
  {
    rw ht,
    rw hnwt,
    norm_num,
  }
  use total_tiles - non_white_tiles,
  exact hwt,

end white_tiles_count_l736_736364


namespace cube_side_length_l736_736040

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end cube_side_length_l736_736040


namespace probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736823

-- Define the given problem as Lean statements

-- Mathematician starts taking pills from March 1
def start_date : ℕ := 1
-- Number of pills per bottle
def pills_per_bottle : ℕ := 10
-- Total bottles
def total_bottles : ℕ := 2
-- Number of days until March 14
def days_till_march_14 : ℕ := 14

-- Define the probability of choosing an empty bottle on March 14
def probability_empty_bottle : ℝ := (286 : ℝ) / 8192

theorem probability_find_empty_bottle_march_14 : 
  probability_empty_bottle = 143 / 4096 :=
sorry

-- Define the expected number of pills taken by the time of discovering an empty bottle
def expected_pills_taken : ℝ := 17.3

theorem expected_pills_taken_when_empty_bottle_discovered : 
  expected_pills_taken = 17.3 :=
sorry

end probability_find_empty_bottle_march_14_expected_pills_taken_when_empty_bottle_discovered_l736_736823


namespace max_angle_on_circumference_l736_736865

theorem max_angle_on_circumference (O A B C : Point) (hO : Center O) (hA : InsideCircle A O) (hB : InsideCircle B O) (hC : OnCircumference C O) :
  (∀ C' : Point, OnCircumference C' O → ∠ ABC ≥ ∠ ABC') ↔ ∠ ACB = 90 :=
sorry

end max_angle_on_circumference_l736_736865


namespace directrix_of_parabola_l736_736860

open Set

noncomputable theory

-- Define the problem and conditions
def parabola (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 4 * y ↔ y = x^2 / 4

def general_parabola (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 4 * p * y ↔ y = x^2 / (4 * p)

-- Define the Lean theorem statement
theorem directrix_of_parabola : parabola 1 → general_parabola 1 → ∃ y : ℝ, y = -1 :=
by
  sorry

end directrix_of_parabola_l736_736860


namespace sum_of_lengths_of_edges_l736_736969

theorem sum_of_lengths_of_edges (s h : ℝ) 
(volume_eq : s^2 * h = 576) 
(surface_area_eq : 4 * s * h = 384) : 
8 * s + 4 * h = 112 := 
by
  sorry

end sum_of_lengths_of_edges_l736_736969


namespace sam_final_amount_is_59616_l736_736398

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def final_amount_after_investment : ℝ :=
  let initial_investment := 10000
  let first_period := 3
  let first_interest_rate := 0.20
  let first_compounded := compound_interest initial_investment first_interest_rate 1 first_period
  let tripled_amount := 3 * first_compounded
  let second_interest_rate := 0.15
  compound_interest tripled_amount second_interest_rate 1 1

theorem sam_final_amount_is_59616 : final_amount_after_investment = 59616 :=
by
  sorry

end sam_final_amount_is_59616_l736_736398


namespace same_even_odd_property_and_monotonicity_l736_736984

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def f (x : ℝ) : ℝ := -2^(abs x)
def g (x : ℝ) : ℝ := 1 - x^2

theorem same_even_odd_property_and_monotonicity : 
  is_even_function f ∧ is_increasing_on f {x | x < 0} → 
  is_even_function g ∧ is_increasing_on g {x | x < 0} :=
by
  sorry

end same_even_odd_property_and_monotonicity_l736_736984


namespace inequality_l736_736932

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) : 
  (1 / (8 * a^2 - 18 * a + 11)) + (1 / (8 * b^2 - 18 * b + 11)) + (1 / (8 * c^2 - 18 * c + 11)) ≤ 3 := 
sorry

end inequality_l736_736932


namespace hyperbola_eccentricity_l736_736662

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = real.sqrt (a^2 + b^2)) 
  (h_eq : ∀ x y, (x = c / real.sqrt 2) → (y = c * real.sqrt a / (real.sqrt 2 * b)) → (y = b / a * x)) 
  (h_AF : ∀ x y, y = -at.div a b (x - c)) 
  (h_B : ∀ y, y = a * c / b → ∃ x, x = 0) :
  ∃ e, e = c / a ∧ e = real.sqrt (1 + b^2 / a^2) ∧ e = real.sqrt 6 / 2 :=
begin
  sorry
end

end hyperbola_eccentricity_l736_736662


namespace represent_2021_as_squares_l736_736931

theorem represent_2021_as_squares :
  ∃ n : ℕ, n = 505 → 2021 = (n + 1)^2 - (n - 1)^2 + 1^2 :=
by
  sorry

end represent_2021_as_squares_l736_736931


namespace complex_conjugate_sum_l736_736711

theorem complex_conjugate_sum (z : ℂ) (h : i * (1 - z) = 1) : z + conj z = 2 := 
by 
  sorry

end complex_conjugate_sum_l736_736711


namespace solution_set_quadratic_inequality_l736_736261

/--
For a real number x, [x] = n if and only if n ≤ x < n + 1 (n ∈ ℕ∗), 
then the solution set of the inequality 4[x]^2 - 36[x] + 45 < 0 for x is [2,8).
-/
theorem solution_set_quadratic_inequality (x : ℝ) (n : ℕ) (hn : n > 0) (h : n ≤ x ∧ x < n + 1 ∧ (4 * (n: ℝ)^2 - 36 * n + 45 < 0)) :
  2 ≤ x ∧ x < 8 :=
sorry

end solution_set_quadratic_inequality_l736_736261


namespace find_m_l736_736752

noncomputable def f (m : ℝ) (x : ℝ) := (x^2 + m * x) * Real.exp x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m 
  (a b : ℝ) 
  (h_interval : a = -3/2 ∧ b = 1)
  (h_decreasing : is_monotonically_decreasing_on_interval (f m) a b) :
  m = -3/2 := 
sorry

end find_m_l736_736752


namespace solve_system_of_equations_l736_736213

theorem solve_system_of_equations (t : ℝ) (k u v : ℝ) 
  (h1 : t = 75) 
  (h2 : t = 5 / 9 * (k - 32)) 
  (h3 : u = t^2 + 2 * t + 5) 
  (h4 : v = log 3 (u - 9)) : 
  k = 167 ∧ u = 5780 ∧ v ≈ 7.882 :=
by {
  sorry
}

end solve_system_of_equations_l736_736213


namespace analytical_expression_of_f_area_of_triangle_ABC_l736_736659

theorem analytical_expression_of_f :
  ∀ (A ω ϕ : ℝ) (f : ℝ → ℝ), 
    (A > 0) → (ω > 0) → (0 < ϕ) → (ϕ < π) → (∀ x : ℝ, f x = A * sin (ω * x + ϕ)) →
    (∀ x : ℝ, x ∈ ℝ) → 
    (∀ x : ℝ, f x ≤ 1) → 
    (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) → 
    (f 0 = 1) → 
    f = λ x, cos x :=
by sorry

theorem area_of_triangle_ABC :
  ∀ (A B C : ℝ) (a : ℝ), 
    (0 < A) → (A < π) →
    (0 < B) → (B < π) →
    (0 < C) → (C < π) →
    (cos A = 3 / 5) → 
    (cos B = 5 / 13) → 
    (a = 13) → 
    1 / 2 * a * (a * sin (A + B) / sin A) * sin B = 84 :=
by sorry

end analytical_expression_of_f_area_of_triangle_ABC_l736_736659


namespace correct_statement_is_D_l736_736105

axiom three_points_determine_plane : Prop
axiom line_and_point_determine_plane : Prop
axiom quadrilateral_is_planar_figure : Prop
axiom two_intersecting_lines_determine_plane : Prop

theorem correct_statement_is_D : two_intersecting_lines_determine_plane = True := 
by sorry

end correct_statement_is_D_l736_736105


namespace interior_angles_of_n_plus_4_sided_polygon_l736_736855

theorem interior_angles_of_n_plus_4_sided_polygon (n : ℕ) (hn : 180 * (n - 2) = 1800) : 
  180 * (n + 4 - 2) = 2520 :=
by sorry

end interior_angles_of_n_plus_4_sided_polygon_l736_736855


namespace inverse_of_3_mod_37_l736_736248

theorem inverse_of_3_mod_37 : ∃ x, 3 * x % 37 = 1 ∧ 0 ≤ x ∧ x < 37 :=
by
  use 25
  split
  { show 3 * 25 % 37 = 1, sorry }
  { split
    { show 0 ≤ 25, linarith }
    { show 25 < 37, linarith }
  }

end inverse_of_3_mod_37_l736_736248


namespace base_conversion_proof_l736_736906

theorem base_conversion_proof :
  let b7_2456 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 2 * 7^3,
      b3_101 := 1 * 3^0 + 0 * 3^1 + 1 * 3^2,
      b5_1234 := 4 * 5^0 + 3 * 5^1 + 2 * 5^2 + 1 * 5^3,
      b7_6789 := 9 * 7^0 + 8 * 7^1 + 7 * 7^2 + 6 * 7^3 in
  (b7_2456 / b3_101) * b5_1234 - b7_6789 = 15420.2 :=
by
  let b7_2456 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 2 * 7^3
  let b3_101 := 1 * 3^0 + 0 * 3^1 + 1 * 3^2
  let b5_1234 := 4 * 5^0 + 3 * 5^1 + 2 * 5^2 + 1 * 5^3
  let b7_6789 := 9 * 7^0 + 8 * 7^1 + 7 * 7^2 + 6 * 7^3
  calc (b7_2456 / b3_101) * b5_1234 - b7_6789 = 92.3 * 194 - 2466 := by sorry
  ... = 15420.2 := by sorry

end base_conversion_proof_l736_736906


namespace even_combinations_result_in_486_l736_736163

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736163


namespace max_total_trains_l736_736810

def trains_birthday (years: ℕ) := years
def trains_christmas (years: ℕ) := 2 * years
def trains_easter (years: ℕ) := 3 * years
def trains_special (years: ℕ) := 4 * years
def total_celebrations_trains (years: ℕ) := 
  trains_birthday years + trains_christmas years + trains_easter years + trains_special years

def trains_uncle (years: ℕ) : ℕ := 
  finset.range years.succ.sum (λ i, nat.floor (real.sqrt (i: ℝ)))

def total_trains (years: ℕ) : ℕ := 
  let initial_trains := total_celebrations_trains years + trains_uncle years - 3 in
  let bonus_trains := initial_trains / 2 in
  initial_trains + bonus_trains + (3 * (initial_trains + bonus_trains))

theorem max_total_trains : total_trains 9 = 636 := by 
  sorry

end max_total_trains_l736_736810


namespace square_geometry_l736_736789

theorem square_geometry 
  (A B C D E F G N P T : Type)
  [Square ABCD]
  (hE : E ∈ segment A B)
  (hF : F ∈ segment B C)
  (hBE : dist B E = dist B F)
  (hN : foot N E B C)
  (hG : ∃ X, G = line_through A D ∩ X ∧ X = extension N perpendicular E B)
  (hP : P = intersection_line EC FG)
  (hT : T = intersection_line NF DC):
  is_perpendicular (line_through D P) (line_through B T) :=
sorry

end square_geometry_l736_736789


namespace triangle_cosine_l736_736349

theorem triangle_cosine (LM : ℝ) (cos_N : ℝ) (LN : ℝ) (h1 : LM = 20) (h2 : cos_N = 3/5) :
  LM / LN = cos_N → LN = 100 / 3 :=
by
  intro h3
  sorry

end triangle_cosine_l736_736349


namespace side_length_of_cube_l736_736028

theorem side_length_of_cube (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 :=
by
  sorry

end side_length_of_cube_l736_736028


namespace cups_of_flour_put_in_l736_736381

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in_l736_736381


namespace exists_unique_adjacent_sums_in_circle_l736_736920

theorem exists_unique_adjacent_sums_in_circle :
  ∃ (f : Fin 10 → Fin 11),
    (∀ (i j : Fin 10), i ≠ j → (f i + f (i + 1)) % 11 ≠ (f j + f (j + 1)) % 11) :=
sorry

end exists_unique_adjacent_sums_in_circle_l736_736920


namespace total_time_spent_l736_736078

/-- 
Three industrial machines were working together to produce shirts. 
Machine A produced 360 shirts at a rate of 4 shirts per minute,
Machine B produced 480 shirts at a rate of 5 shirts per minute,
Machine C produced 300 shirts at a rate of 3 shirts per minute.
There were two machine malfunctions during the process that halted production on all machines.
Each malfunction took 5 minutes to fix. 
Prove the total time spent on producing the shirts and fixing the malfunctions.
-/
theorem total_time_spent :
  let time_A := 360 / 4,
      time_B := 480 / 5,
      time_C := 300 / 3,
      production_time := max (max time_A time_B) time_C,
      malfunction_time := 2 * 5,
      total_time := production_time + malfunction_time
  in total_time = 110 :=
by
  sorry

end total_time_spent_l736_736078


namespace swap_values_l736_736852

theorem swap_values (a b : ℕ) (h₁ : a = 8) (h₂ : b = 7) (c : ℕ) : 
  (let c := b in let b := a in let a := c in (a = 7 ∧ b = 8)) :=
by
  sorry

end swap_values_l736_736852


namespace total_chairs_calculation_l736_736947

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l736_736947


namespace number_of_even_results_l736_736148

def initial_number : ℕ := 1

def operations (x : ℕ) : List (ℕ → ℕ) :=
  [x + 2, x + 3, x * 2]

def apply_operations (x : ℕ) (ops : List (ℕ → ℕ)) : ℕ :=
  List.foldl (fun acc op => op acc) x ops

def even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_results : 
  ∃ (ops : List (ℕ → ℕ) → List (ℕ → ℕ)), List.length ops = 6 → 
  ∑ (ops_comb : List (List (ℕ → ℕ))) in (List.replicate 6 (operations initial_number)).list_prod,
    if even (apply_operations initial_number ops_comb) then 1 else 0 = 486 := 
sorry

end number_of_even_results_l736_736148


namespace omega_in_range_l736_736654

theorem omega_in_range (ω : ℝ) (h1 : 0 < ω) 
  (h2 : ∀ x y, - (2 * Real.pi) / 3 ≤ x ∧ x ≤ y ∧ y ≤ (Real.pi) / 3 → f x ≤ f y) 
  (h3 : ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (ω * x)) → ∃! x, 0 ≤ x ∧ x ≤ Real.pi ∧ abs (f x) = 1) :
  1 / 2 ≤ ω ∧ ω ≤ 3 / 4 :=
by sorry

end omega_in_range_l736_736654


namespace min_M_value_l736_736797

theorem min_M_value (n : ℕ) (a : Fin n -> ℝ) (h_pos : ∀ i, 0 < a i) (h_sum : (Finset.univ.sum a) = 2023) :
  ∃ M, M = 1 - 1 / real.sqrt (2 ^ (1 / (n : ℝ))) ∧ 
  M = Finset.univ.sup (λ i : Fin n, ∑ j in Finset.range (i + 1), a j / (2023 + ∑ k in Finset.range (i + 1), a k)) := 
sorry

end min_M_value_l736_736797


namespace total_chairs_calculation_l736_736946

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l736_736946


namespace number_of_pupils_l736_736496

theorem number_of_pupils (n : ℕ) : (83 - 63) / n = 1 / 2 → n = 40 :=
by
  intro h
  -- This is where the proof would go.
  sorry

end number_of_pupils_l736_736496


namespace retail_price_of_machine_l736_736115

theorem retail_price_of_machine (R : ℝ) (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  wholesale_price = 108 → 
  discount_rate = 0.10 → 
  profit_rate = 0.20 → 
  R * (1 - discount_rate) = wholesale_price * (1 + profit_rate) → 
  R = 144 := 
by 
  assume h₁ : wholesale_price = 108 
  assume h₂ : discount_rate = 0.10 
  assume h₃ : profit_rate = 0.20 
  assume h₄ : R * (1 - discount_rate) = wholesale_price * (1 + profit_rate)
  -- Substitute values to show that R = 144 after simplification
  sorry

end retail_price_of_machine_l736_736115


namespace initial_children_count_l736_736465

theorem initial_children_count (passed retake : ℝ) (h_passed : passed = 105.0) (h_retake : retake = 593) : 
    passed + retake = 698 := 
by
  sorry

end initial_children_count_l736_736465


namespace white_tiles_count_l736_736361

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l736_736361


namespace angle_in_fourth_quadrant_l736_736933

theorem angle_in_fourth_quadrant (θ : ℝ) (h : θ = -1445) : (θ % 360) > 270 ∧ (θ % 360) < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l736_736933


namespace man_alone_days_l736_736178

-- Conditions from the problem
variables (M : ℕ) (h1 : (1 / (↑M : ℝ)) + (1 / 12) = 1 / 3)  -- Combined work rate condition

-- The proof statement we need to show
theorem man_alone_days : M = 4 :=
by {
  sorry
}

end man_alone_days_l736_736178


namespace log27_3_eq_one_third_l736_736236

theorem log27_3_eq_one_third : ∃ x : ℝ, (27 = 3^3) ∧ (27^x = 3) ∧ (x = 1/3) :=
by {
  have h1: 27 = 3^3 := by norm_num,
  have h2: 27^(1/3) = 3 := by {
    rw [h1, ← real.rpow_mul (real.rpow_pos_of_pos (by norm_num) (3: ℝ))],
    norm_num
  },
  exact ⟨1/3, h1, h2⟩
}

end log27_3_eq_one_third_l736_736236


namespace number_of_factors_of_48_multiples_of_8_l736_736311

theorem number_of_factors_of_48_multiples_of_8 : 
  {n : ℕ | n > 0 ∧ n ∣ 48 ∧ 8 ∣ n}.to_finset.card = 4 := 
by sorry

end number_of_factors_of_48_multiples_of_8_l736_736311


namespace white_tiles_count_l736_736360

-- Definitions from conditions
def total_tiles : ℕ := 20
def yellow_tiles : ℕ := 3
def blue_tiles : ℕ := yellow_tiles + 1
def purple_tiles : ℕ := 6

-- We need to prove that number of white tiles is 7
theorem white_tiles_count : total_tiles - (yellow_tiles + blue_tiles + purple_tiles) = 7 := by
  -- Placeholder for the actual proof
  sorry

end white_tiles_count_l736_736360


namespace opposite_of_neg_three_l736_736434

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end opposite_of_neg_three_l736_736434


namespace result_prob_a_l736_736837

open Classical

noncomputable def prob_a : ℚ := 143 / 4096

theorem result_prob_a (k : ℚ) (h : k = prob_a) : k ≈ 0.035 := by
  sorry

end result_prob_a_l736_736837


namespace even_combinations_after_six_operations_l736_736160

def operation1 := (x : ℕ) => x + 2
def operation2 := (x : ℕ) => x + 3
def operation3 := (x : ℕ) => x * 2

def operations := [operation1, operation2, operation3]

def apply_operations (ops : List (ℕ → ℕ)) (x : ℕ) : ℕ :=
ops.foldl (λ acc f => f acc) x

def even_number_combinations (n : ℕ) : ℕ :=
(List.replicateM n operations).count (λ ops => (apply_operations ops 1) % 2 = 0)

theorem even_combinations_after_six_operations : even_number_combinations 6 = 486 :=
by
  sorry

end even_combinations_after_six_operations_l736_736160


namespace part_a_part_b_l736_736934

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem part_a :
  ¬∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^3) :=
sorry

theorem part_b :
  ∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^4) :=
sorry

end part_a_part_b_l736_736934


namespace jordyn_total_cost_l736_736326

theorem jordyn_total_cost (
  price_cherries : ℕ := 5,
  price_olives : ℕ := 7,
  discount : ℕ := 10,
  quantity : ℕ := 50
) : (50 * (price_cherries - (price_cherries * discount / 100)) + 50 * (price_olives - (price_olives * discount / 100))) = 540 :=
by
  sorry

end jordyn_total_cost_l736_736326


namespace even_combinations_result_in_486_l736_736165

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end even_combinations_result_in_486_l736_736165


namespace option_C_correct_option_D_correct_l736_736673

open Real

def a : ℝ × ℝ × ℝ := (1, -1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 1)
def c : ℝ × ℝ × ℝ := (2, -3, 1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def add_vectors (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def scalar_mult (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

theorem option_C_correct : dot_product (add_vectors a (scalar_mult 5 b)) c = 0 :=
  sorry

theorem option_D_correct : ∃ k : ℝ, k ≠ 0 ∧ ∀ i, (a -ₐ *(b - c)) = (k * a) :=
  sorry

end option_C_correct_option_D_correct_l736_736673


namespace Shane_current_age_44_l736_736893

-- Declaring the known conditions and definitions
variable (Garret_present_age : ℕ) (Shane_past_age : ℕ) (Shane_present_age : ℕ)
variable (h1 : Garret_present_age = 12)
variable (h2 : Shane_past_age = 2 * Garret_present_age)
variable (h3 : Shane_present_age = Shane_past_age + 20)

theorem Shane_current_age_44 : Shane_present_age = 44 :=
by
  -- Proof to be filled here
  sorry

end Shane_current_age_44_l736_736893


namespace find_original_price_l736_736420

theorem find_original_price (x y : ℝ) 
  (h1 : 60 * x + 75 * y = 2700)
  (h2 : 60 * 0.85 * x + 75 * 0.90 * y = 2370) : 
  x = 20 ∧ y = 20 :=
sorry

end find_original_price_l736_736420
