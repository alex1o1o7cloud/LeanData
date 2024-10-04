import Mathlib

namespace range_of_m_l37_37097

noncomputable def f (x : ℝ) : ℝ := ite (x ≥ 0) (x - sin x) (-(x - sin (-x)))

theorem range_of_m
  (hf_odd : ∀ x : ℝ, f(-x) = -f(x))
  (hf_nonneg : ∀ x : ℝ, 0 ≤ x → f(x) = x - sin x)
  (hf_ineq : ∀ t : ℝ, f(-4 * t) > f(2 * m * t^2 + m)) :
  m ∈ Set.Iio (-Real.sqrt 2) :=
sorry

end range_of_m_l37_37097


namespace intersection_of_M_and_complement_N_l37_37529

-- Define the sets M and N over the reals
def M : Set ℝ := {x | 0 < x ∧ x < 27}
def N : Set ℝ := {x | x < -1 ∨ x > 5}

-- Goal: Prove that M ∩ (complement_R N) is (0, 5]
theorem intersection_of_M_and_complement_N :
  M ∩ {x | -1 ≤ x ∧ x ≤ 5} = {x | 0 < x ∧ x ≤ 5} :=
by
  sorry -- Proof is omitted

end intersection_of_M_and_complement_N_l37_37529


namespace test_takers_percent_correct_l37_37976

theorem test_takers_percent_correct 
  (n : Set ℕ → ℝ) 
  (A B : Set ℕ) 
  (hB : n B = 0.75) 
  (hAB : n (A ∩ B) = 0.60) 
  (hneither : n (Set.univ \ (A ∪ B)) = 0.05) 
  : n A = 0.80 := by
  sorry

end test_takers_percent_correct_l37_37976


namespace train_crossing_time_l37_37333

-- Define the conditions
def length_of_train : ℝ := 100
def speed_kmh : ℝ := 360

-- Conversion from km/hr to m/s
def speed_mps : ℝ := speed_kmh * (1000 / 3600)

-- Time to cross the pole calculated using the formula time = distance / speed
theorem train_crossing_time : 
  (length_of_train / speed_mps) = 1 :=
by
  -- Given conditions and definitions, we need to verify the time calculated
  sorry

end train_crossing_time_l37_37333


namespace planes_perpendicular_l37_37077

variables {m n : Type} [line m] [line n]
variables {α β γ : Type} [plane α] [plane β] [plane γ]

theorem planes_perpendicular (hm_beta : m ⊥ β) (hm_alpha : m ∥ α) : α ⊥ β :=
sorry

end planes_perpendicular_l37_37077


namespace natural_numbers_solution_l37_37864

theorem natural_numbers_solution (a : ℕ) :
  ∃ k n : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1 ∧ (7 * k + 15 * n - 1) % (3 * k + 4 * n) = 0 :=
sorry

end natural_numbers_solution_l37_37864


namespace min_value_of_f_l37_37913

def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2 * y + x * y^2 - 3 * (x^2 + y^2 + x * y) + 3 * (x + y)

theorem min_value_of_f : ∀ x y : ℝ, x ≥ 1/2 → y ≥ 1/2 → f x y ≥ 1
    := by
      intros x y hx hy
      -- Rest of the proof would go here
      sorry

end min_value_of_f_l37_37913


namespace equilateral_triangle_side_length_l37_37299
noncomputable def equilateral_triangle_side (r R : ℝ) (h : R > r) : ℝ :=
  r * R * Real.sqrt 3 / (Real.sqrt (r ^ 2 - r * R + R ^ 2))

theorem equilateral_triangle_side_length
  (r R : ℝ) (hRgr : R > r) :
  ∃ a, a = equilateral_triangle_side r R hRgr :=
sorry

end equilateral_triangle_side_length_l37_37299


namespace parabola_directrix_is_x_eq_1_l37_37888

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37888


namespace problem1_problem2_l37_37128

-- Definition of the function f
def f (a x : ℝ) := Real.exp x - a * x - 1

-- Definition of the series sum we need to bound
def series_sum (n : ℕ) : ℝ := (Finset.range n).sum (λ k, ((k + 1 : ℝ) / n) ^ n)

theorem problem1 (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : a = 1 :=
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : series_sum n < Real.exp 1 / (Real.exp 1 - 1) :=
sorry

end problem1_problem2_l37_37128


namespace LaKeisha_needs_more_mowing_l37_37209

noncomputable def LaKeisha_additional_mowing (lawn_charge hedge_charge rake_charge : ℝ)
                                             (book_cost : ℝ)
                                             (num_lawns lawn_area : ℕ)
                                             (hedge_length : ℕ)
                                             (leaves_area : ℕ)
: ℕ :=
  let earned_mowing := lawn_charge * (num_lawns * lawn_area)
  let earned_trimming := hedge_charge * hedge_length
  let earned_raking := rake_charge * leaves_area
  let total_earned := earned_mowing + earned_trimming + earned_raking
  let remaining_needed := book_cost - total_earned
  in remaining_needed / lawn_charge

theorem LaKeisha_needs_more_mowing :
  LaKeisha_additional_mowing 0.10 0.05 0.02 375 5 600 100 500 = 600 :=
by
  sorry

end LaKeisha_needs_more_mowing_l37_37209


namespace min_product_non_neg_reals_l37_37928

variable {α : Type*} [OrderedCommGroup α]

theorem min_product_non_neg_reals (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) 
    (h_sum : (∑ i, x i) ≤ 1/2) : 
    (∏ i, (1 - x i)) ≥ 1/2 := 
sorry

end min_product_non_neg_reals_l37_37928


namespace sequence_classification_l37_37423

noncomputable def sequence_property (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 1 ≤ n →
  (a (n + 2) % a n = 0) ∧
  (| s (n + 1) - (n + 1) * a n | = 1)
  where s m := ∑ i in range m, (-1)^i * a i

theorem sequence_classification (a : ℕ → ℕ) :
  sequence_property a → a 0 ≥ 2015 →
  (∃ c : ℕ, c ≥ 2014 ∧ a 0 = c + 1 ∧ (∀ n, 1 ≤ n → a n = c * (n + 2) * n)) ∨
  (∃ c : ℕ, c ≥ 2016 ∧ a 0 = c - 1 ∧ (∀ n, 1 ≤ n → a n = c * (n + 2) * n)) :=
sorry

end sequence_classification_l37_37423


namespace chi_squared_significance_probability_B_second_day_l37_37429

-- Defining the chi-squared test problem
def data := {a : Nat := 40, b : Nat := 10, c : Nat := 20, d : Nat := 30, n : Nat := 100}

def chi_squared(data : {a : Nat, b : Nat, c : Nat, d : Nat, n : Nat}) : Float :=
  let {a, b, c, d, n} := data
  Float.ofNat(n) * (Float.ofNat(a * d - b * c))^2 / 
  (Float.ofNat(a + b) * Float.ofNat(c + d) * Float.ofNat(a + c) * Float.ofNat(b + d))

def chi_squared_critical_value : Float := 10.828

theorem chi_squared_significance :
  chi_squared(data) > chi_squared_critical_value :=
by 
  sorry

-- Defining the probability calculation problem
def P_A : Float := 0.5
def P_B_given_A : Float := 0.7
def P_B_given_not_A : Float := 0.8

def P_B : Float := P_A * P_B_given_A + (1 - P_A) * P_B_given_not_A

def P_not_B : Float := 1 - P_B

theorem probability_B_second_day :
  P_not_B = 0.25 :=
by 
  sorry

end chi_squared_significance_probability_B_second_day_l37_37429


namespace inequality_proof_l37_37106

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l37_37106


namespace tangent_of_parabola_axis_equals_tangent_to_circle_l37_37139

theorem tangent_of_parabola_axis_equals_tangent_to_circle (p : ℝ) (hp : 0 < p) :
  (∀ x y, y^2 = 2 * p * x) ∧
  (∀ x y, x^2 + y^2 - 6 * x - 7 = 0) ∧
  (∀ x1 y1, 
    √(1^2 + 0^2) * 4 = |1 * 3 + 0 * 0 + p / 2| →
    √((x1 - 3)^2 + y1^2) = 4) → p = 2 :=
sorry

end tangent_of_parabola_axis_equals_tangent_to_circle_l37_37139


namespace positive_value_of_m_l37_37507

theorem positive_value_of_m (α m : ℝ) 
  (h1 : sin α * cos α = -7 / 16) 
  (h2 : α ∈ Ioo (π / 2) π) 
  (h3 : m * cos (2 * α) = sin (π / 4 - α)) : 
  m = 2 :=
sorry

end positive_value_of_m_l37_37507


namespace total_weight_proof_l37_37599
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l37_37599


namespace y_intercept_of_line_l37_37314

theorem y_intercept_of_line (x y : ℝ) (h : 3 * x - 2 * y + 7 = 0) (hx : x = 0) : y = 7 / 2 :=
by
  sorry

end y_intercept_of_line_l37_37314


namespace john_must_work_10_more_days_l37_37207

theorem john_must_work_10_more_days
  (total_days : ℕ)
  (total_earnings : ℝ)
  (daily_earnings : ℝ)
  (target_earnings : ℝ)
  (additional_days : ℕ) :
  total_days = 10 →
  total_earnings = 250 →
  daily_earnings = total_earnings / total_days →
  target_earnings = 2 * total_earnings →
  additional_days = (target_earnings - total_earnings) / daily_earnings →
  additional_days = 10 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end john_must_work_10_more_days_l37_37207


namespace binary_156_y_minus_x_is_zero_l37_37847

def binary_representation (n : ℕ) : list ℕ :=
  if n = 0 then [0] else
  let rec helper n :=
    if n = 0 then [] else (n % 2) :: helper (n / 2)
  helper n

def count_zeros_ones (l : list ℕ) : ℕ × ℕ :=
  l.foldr (λ b ⟨zeros, ones⟩, if b = 0 then (zeros + 1, ones) else (zeros, ones + 1)) (0, 0)

def y_minus_x (n : ℕ) : ℤ :=
  let (x, y) := count_zeros_ones (binary_representation n)
  y - x

theorem binary_156_y_minus_x_is_zero : y_minus_x 156 = 0 := by
  sorry

end binary_156_y_minus_x_is_zero_l37_37847


namespace man_l37_37364

variable (v : ℝ)  -- Man's speed in still water
variable (c : ℝ)  -- Speed of current
variable (s : ℝ)  -- Man's speed against the current
variable (u : ℝ)  -- Man's speed with the current

-- Given Conditions
axiom current_speed : c = 2.5
axiom against_current_speed : s = 10

-- Question to be proved
theorem man's_speed_with_current : u = 15 :=
by
  -- Define the problem as per the given conditions
  have v := s + c
  have u := v + c
  rw [current_speed, against_current_speed] at u
  exact u

end man_l37_37364


namespace quadratic_two_distinct_real_roots_l37_37518

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k > 0) :
    (∃ x : ℝ, x^2 - 2*x + 1 - k = 0 ∧ x ≠ (2 ± sqrt(4 - 4*(1-k)))/2) :=
by
  sorry

end quadratic_two_distinct_real_roots_l37_37518


namespace vector_length_sum_l37_37962

variables (a b : ℝ^3) -- Assuming 3-dimensional vectors for the context of this problem

-- Vector norms
def norm (v : ℝ^3) : ℝ := real.sqrt (v.dot v)

-- Given conditions
axiom norm_a : norm a = 1
axiom norm_b : norm b = 2
axiom dot_eq_zero : a.dot (a - 2 • b) = 0

-- The proof problem
theorem vector_length_sum : norm (a + b) = real.sqrt 6 :=
by
  sorry

end vector_length_sum_l37_37962


namespace find_parabola_directrix_l37_37140

open Real

noncomputable def parabola_directrix (p : ℝ) (h : p > 0) : Prop :=
  ∃ (x_mid : ℝ), (x_mid = 3) →
    let focus := (p / 2, 0) in
    let line_eq := λ (x : ℝ), 2 * (x - p / 2) in
    let parabola_eq := λ (x y : ℝ), y^2 = 2 * p * x in
    (∀ x y, line_eq(x) = y → parabola_eq(x, y)) →
    (∃ (directrix : ℝ), directrix = -2)

theorem find_parabola_directrix : ∀ (p : ℝ) (h : p > 0),
  parabola_directrix p h :=
begin
  intros p h,
  use 3,
  intros h_mid,
  let focus := (p / 2, 0),
  let line_eq := λ (x : ℝ), 2 * (x - p / 2),
  let parabola_eq := λ (x y : ℝ), y^2 = 2 * p * x,
  intro line_parabola_intersect,
  use -2,
  sorry,
end

end find_parabola_directrix_l37_37140


namespace tan_geometric_seq_l37_37941

theorem tan_geometric_seq (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a(n+1) = a(n) * r) 
  (h_eqn : a 1 * a 13 + 2 * (a 7)^2 = 4 * real.pi) : 
  real.tan (a 2 * a 12) = real.sqrt 3 := 
by {
  -- We would have the proof here
  sorry
}

end tan_geometric_seq_l37_37941


namespace min_distance_to_line_l37_37058

theorem min_distance_to_line : 
  ∃ (x y : ℤ), ∀ (a b : ℤ), 
    let d := (|25 * a - 15 * b + 12| / (5 * sqrt 34)) in 
    d ≥ (1 / sqrt 85) := sorry

end min_distance_to_line_l37_37058


namespace ball_drawing_ways_l37_37762

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37762


namespace omega_values_l37_37162

noncomputable def omega_range : set ℝ :=
  {ω | ∃ t ∈ set.Ioo 0 (Real.pi / 2), (∃ k : ℤ, t = (2 * Real.pi / 3 + k * Real.pi) / ω) ∧ (0 < ω)}

theorem omega_values :
  omega_range = {ω | ω ∈ set.Ioo (4 / 3) (10 / 3) ∪ {10 / 3}} :=
by sorry

end omega_values_l37_37162


namespace number_of_true_propositions_l37_37220

-- Definitions for lines, planes, and their relationships
variables {Line Plane : Type}
  [linear_ordered_field Plane]
  (lines_parallel : Line → Plane → Prop)
  (plane_intersect : Line → Plane → Prop)
  (skew_lines : Line → Line → Prop)
  (plane_parallel : Plane → Plane → Prop)
  (l m n : Line)
  (α β γ : Plane)

-- Propositions
def proposition1 : Prop := (skew_lines l m) ∧ (lines_parallel l α) ∧ (lines_parallel m β) → plane_parallel α β
def proposition2 : Prop := (plane_parallel α β) ∧ (lines_parallel l α) ∧ (lines_parallel m β) → lines_parallel l m
def proposition3 : Prop := (plane_intersect l α) ∧ (plane_intersect m β) ∧ (plane_intersect n γ) ∧ (lines_parallel l n) → lines_parallel m n

-- Statement to prove the number of true propositions
theorem number_of_true_propositions : (¬ proposition1) ∧ (¬ proposition2) ∧ proposition3 ∧ (1 = 1) :=
by 
  sorry

end number_of_true_propositions_l37_37220


namespace square_area_eq_42_25_l37_37381

theorem square_area_eq_42_25:
  let a := 7.5 in
  let b := 8.5 in
  let c := 10 in
  let perimeter_triangle := a + b + c in
  let side_square := perimeter_triangle / 4 in
  let area_square := side_square ^ 2 in
  area_square = 42.25 :=
by
  sorry

end square_area_eq_42_25_l37_37381


namespace find_angle_B_find_sin_C_l37_37560

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, C respectively

theorem find_angle_B
  (h₁ : 0 < A ∧ A < π)
  (h₂ : 0 < B ∧ B < π)
  (h₃ : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = π / 3 :=
sorry

theorem find_sin_C
  (a : ℝ := 2)
  (c : ℝ := 3)
  (B : ℝ := π / 3)
  (h₁ : noncomputable)
  (b : ℝ := Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)) :
  Real.sin C = (3 * Real.sqrt 21) / 14 :=
sorry

end find_angle_B_find_sin_C_l37_37560


namespace hyperbola_tangent_condition_l37_37262

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through (E : ℝ × ℝ) (m : ℝ) : set (ℝ × ℝ) := 
  {P | ∃ x, P = (x, m * (x - E.1))}

theorem hyperbola_tangent_condition (A B C D E : ℝ × ℝ) (e : set (ℝ × ℝ)) (m : ℝ) :
  (A = (1, -1)) ∧ (B = (1, 1)) ∧ (C = (-1, 1)) ∧ (D = (-1, -1)) ∧ 
  (E = (real.sqrt 2, 0)) ∧ 
  (∃ P Q, line_through E m P ∧ line_through E m Q ∧ 
          P.1 = 1 ∧ Q.1 = -1 ∧ ∃ K, midpoint P Q = K ∧ 
          hyperbola_with_asymptotes_parallel_to_AC_BD_centered_at_K_PQ_tangent_to_circle K P Q A B C D) →
  |m| ≤ real.sqrt 2 ∧ |m| ≠ 1 := 
by 
  intros h,
  sorry

end hyperbola_tangent_condition_l37_37262


namespace cosine_largest_angle_l37_37088

variables {a b c : ℝ}
variables {A B C : ℝ}

theorem cosine_largest_angle (h1: a^2 - a - real.sqrt 3 * b - real.sqrt 3 * c = 0)
  (h2: a + real.sqrt 3 * b - real.sqrt 3 * c + 2 = 0) :
  cos C = -real.sqrt 3 / 3 :=
sorry

end cosine_largest_angle_l37_37088


namespace hyperbola_eccentricity_l37_37137

-- Define the given hyperbola and its properties
variables {a b x y : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0)

def hyperbola (x y : ℝ) : Prop := (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Define the foci
noncomputable def c : ℝ := Real.sqrt (a^2 + b^2)
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)

-- Line passing through F2 with inclination π/4
def line_through_F2 (x : ℝ) : ℝ := x - c

-- Intersection point A
def A : ℝ × ℝ := 
  let x := Classical.some 
    (Exists.intro (c + sqrt (a^2 - b^2)) (hyperbola (c + sqrt (a^2 - b^2)) (line_through_F2 c (c + sqrt (a^2 - b^2)))))
  (x, line_through_F2 c x)

-- Define the isosceles right triangle condition
def is_isosceles_right_triangle (A F1 F2 : ℝ × ℝ) : Prop :=
  let d_F1_A := Real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2)
  let d_F2_A := Real.sqrt ((A.1 - F2.1)^2 + (A.2 - F2.2)^2)
  (d_F1_A = d_F2_A) ∧ (d_F1_A * d_F1_A = 2 * ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2))

-- Proof of eccentricity
theorem hyperbola_eccentricity
  (h : hyperbola (A.1) (A.2))
  (h_isosceles : is_isosceles_right_triangle A (F1) (F2))
  : Real.sqrt (1 + b^2 / a^2) = sqrt 2 + 1 :=
sorry

end hyperbola_eccentricity_l37_37137


namespace min_period_of_function_l37_37275

theorem min_period_of_function : 
  ∀ x : ℝ, (2 * sin (3 * x + π / 6) = 2 * sin (3 * (x + (2 * π / 3)) + π / 6)) :=
by sorry

end min_period_of_function_l37_37275


namespace count_solutions_congruence_l37_37111

theorem count_solutions_congruence : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, x + 20 ≡ 75 [MOD 45] ∧ x < 150 :=
sorry

end count_solutions_congruence_l37_37111


namespace relationship_between_a_b_c_l37_37474

noncomputable def a : ℝ := Real.log 11 / Real.log 5
noncomputable def b : ℝ := Real.log 8 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt Real.exp 1

theorem relationship_between_a_b_c : a < b ∧ b < c := by
  sorry

end relationship_between_a_b_c_l37_37474


namespace complex_number_coordinate_l37_37481

noncomputable def complex_coord : ℂ := (i^2015) / (i - 2)

theorem complex_number_coordinate :
  (complex_coord.re, complex_coord.im) = (-1/5, 2/5) := 
sorry

end complex_number_coordinate_l37_37481


namespace heptagon_isosceles_same_color_l37_37430

theorem heptagon_isosceles_same_color 
  (color : Fin 7 → Prop) (red blue : Prop)
  (h_heptagon : ∀ i : Fin 7, color i = red ∨ color i = blue) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ color i = color j ∧ color j = color k ∧ ((i + j) % 7 = k ∨ (j + k) % 7 = i ∨ (k + i) % 7 = j) :=
sorry

end heptagon_isosceles_same_color_l37_37430


namespace room_ratio_l37_37377

theorem room_ratio (length : ℝ) (width_inches : ℝ) 
  (length_eq : length = 25) 
  (width_eq : width_inches = 150) : (1 : ℝ) = 1 / 3 :=
by
  -- Define constants
  let width_feet : ℝ := 12.5
  have width_conversion : width_feet = width_inches / 12 := by sorry
  
  -- Define the perimeter
  let perimeter : ℝ := 75
  have perimeter_def : perimeter = 2 * (length + width_feet) := by sorry
  
  -- Define the length to perimeter ratio
  let ratio : ℝ := length / perimeter
  have ratio_def : ratio = 1 / 3 := by sorry
  
  -- Conclude the proof
  exact ratio_def

end room_ratio_l37_37377


namespace directrix_of_given_parabola_l37_37882

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37882


namespace rank_identity_l37_37604

theorem rank_identity (n p : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) 
  (h1: 2 ≤ n) (h2: 2 ≤ p) (h3: A^(p+1) = A) : 
  Matrix.rank A + Matrix.rank (1 - A^p) = n := 
  sorry

end rank_identity_l37_37604


namespace carly_lollipops_total_l37_37418

theorem carly_lollipops_total (C : ℕ) (h1 : C / 2 = cherry_lollipops)
  (h2 : C / 2 = 3 * 7) : C = 42 :=
by
  sorry

end carly_lollipops_total_l37_37418


namespace parabola_vertex_l37_37268

noncomputable def is_vertex (x y : ℝ) : Prop :=
  y^2 + 8 * y + 4 * x + 5 = 0 ∧ (∀ y₀, y₀^2 + 8 * y₀ + 4 * x + 5 ≥ 0)

theorem parabola_vertex : is_vertex (11 / 4) (-4) :=
by
  sorry

end parabola_vertex_l37_37268


namespace min_area_triangle_AOB_l37_37513

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := -1/x

theorem min_area_triangle_AOB :
  (∃ u v : ℝ, u > 0 ∧ v > 0 ∧
               ∠ ((0, 0): ℝ × ℝ) (u, f u) (v, g v) = real.pi / 3 ∧
               S (u, v) = 1.1465) := 
begin
  -- Definitions to compute area
  let A := λ u : ℝ, (u, f u),
  let B := λ v : ℝ, (v, g v),
  let S := λ (u v : ℝ), (1 / 2) * abs (u * (-1 / v) - v * u^2),
  
  -- Real part of the problem statement, including the proof that will derive the answer
  use 0.411797,
  -- Some example value for u and v that you would use to test/debug
  use (u : ℝ),
  use (v : ℝ),
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { 
    have h_area : S 0.411797 (some_v_value) = 1.1465, 
    { sorry },
    rw h_area,
  }
end

end min_area_triangle_AOB_l37_37513


namespace problem_1_problem_2_l37_37127

noncomputable def f (ω x : ℝ) : ℝ :=
  Real.sin (ω * x + (Real.pi / 4))

theorem problem_1 (ω : ℝ) (hω : ω > 0) : f ω 0 = Real.sqrt 2 / 2 :=
by
  unfold f
  simp [Real.sin_pi_div_four]

theorem problem_2 : 
  ∃ x : ℝ, 
  0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi / 2 → f 2 y ≤ f 2 x) ∧ 
  f 2 x = 1 :=
by
  sorry

end problem_1_problem_2_l37_37127


namespace complex_conjugate_of_z_l37_37796

variable (z : ℂ) (h : z * (1 - 2 * complex.I) = 3 + 2 * complex.I)

theorem complex_conjugate_of_z : z.conj = 1 + 8 * complex.I := 
sorry

end complex_conjugate_of_z_l37_37796


namespace find_a_from_geometric_sequence_l37_37181

-- Definition of the geometric sequence and the sum of the first n terms
variables (a : ℝ) (S : ℕ → ℝ)
def geometric_sequence (n : ℕ) : ℝ := S n
def sum_first_n_terms (n : ℕ) : ℝ := 5^(n+1) + a

-- Problem statement: Given S_n = 5^(n+1) + a, prove that a = -5
theorem find_a_from_geometric_sequence (h : ∀ n, geometric_sequence a S n = sum_first_n_terms a n) : a = -5 :=
by
  -- Proof goes here
  sorry

end find_a_from_geometric_sequence_l37_37181


namespace ellipse_properties_l37_37099

noncomputable def ellipse_equation (a b : ℝ) (a_gt_b : a > b) (eccentricity : ℝ) (e_cond : eccentricity = (Real.sqrt 2) / 2) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (eccentricity = (Real.sqrt 2) / 2) ∧ (a = 4 * Real.sqrt 2) ∧ (b = 4) ∧
  (∀ x y : ℝ, (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 32 + p.2^2 / 16 = 1 })

noncomputable def possible_k_values (k : ℝ) : Prop :=
  k ∈ Ioo (-Real.sqrt 94 / 2) 0 ∪ Ioo 0 (Real.sqrt 94 / 2)

-- Lean 4 statement encapsulating the proof problems
theorem ellipse_properties (a b : ℝ) (a_gt_b : a > b) (eccentricity : ℝ) (k : ℝ) :
  (ellipse_equation a b a_gt_b eccentricity (Real.sqrt 2 / 2)) ∧ (possible_k_values k) :=
begin
  sorry
end

end ellipse_properties_l37_37099


namespace abc_le_sqrt2_div_4_l37_37515

variable {a b c : ℝ}
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variable (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1)

theorem abc_le_sqrt2_div_4 (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1) :
  a * b * c ≤ (Real.sqrt 2) / 4 := 
sorry

end abc_le_sqrt2_div_4_l37_37515


namespace graph_intersection_unique_l37_37844

theorem graph_intersection_unique :
  ∃! x : ℝ, 3 * log 3 x = log 3 (9 * x) :=
begin
  sorry
end

end graph_intersection_unique_l37_37844


namespace john_work_more_days_l37_37201

theorem john_work_more_days (days_worked : ℕ) (amount_made : ℕ) (daily_earnings : ℕ) (h1 : days_worked = 10) (h2 : amount_made = 250) (h3 : daily_earnings = amount_made / days_worked) : 
  ∃ more_days : ℕ, more_days = (2 * amount_made / daily_earnings) - days_worked := 
by
  have h4 : daily_earnings = 25 := by {
    rw [h1, h2],
    norm_num,
  }
  have h5 : 2 * amount_made / daily_earnings = 20 := by {
    rw [h2, h4],
    norm_num,
  }
  use 10
  rw [h1, h5]
  norm_num

end john_work_more_days_l37_37201


namespace probability_at_least_one_two_given_different_l37_37719

theorem probability_at_least_one_two_given_different :
  let Ω := { (d1, d2) : ℕ × ℕ | 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ d1 ≠ d2 }
  let favorable := { (d1, d2) : ℕ × ℕ | (d1 = 2 ∨ d2 = 2) ∧ d1 ≠ d2 }
  (finset.card favorable : ℚ) / (finset.card Ω : ℚ) = 1 / 3 :=
by
  -- Define the sample space Ω
  let Ω := { (d1, d2) : ℕ × ℕ | 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ d1 ≠ d2 }
  -- Define the set of favorable outcomes 
  let favorable := { (d1, d2) : ℕ × ℕ | (d1 = 2 ∨ d2 = 2) ∧ d1 ≠ d2 }
  -- Total number of distinct outcomes (30) and number of favorable outcomes (10)
  have HΩ: finset.card Ω = 30, sorry
  have Hfav: finset.card favorable = 10, sorry
  -- Use these results to compute the probability
  calc (finset.card favorable : ℚ) / (finset.card Ω : ℚ)
       = (10 : ℚ) / (30 : ℚ) : by rw [HΩ, Hfav]
       ... = 1 / 3 : by norm_num
  sorry

end probability_at_least_one_two_given_different_l37_37719


namespace correct_operations_l37_37827

theorem correct_operations : 
  (∀ x y : ℝ, x^2 + x^4 ≠ x^6) ∧
  (∀ x y : ℝ, 2*x + 4*y ≠ 6*x*y) ∧
  (∀ x : ℝ, x^6 / x^3 = x^3) ∧
  (∀ x : ℝ, (x^3)^2 = x^6) :=
by 
  sorry

end correct_operations_l37_37827


namespace train_cross_time_l37_37334

-- Definitions from conditions
def train_length : ℝ := 400   -- length in meters
def train_speed_kmh : ℝ := 144 -- speed in km/hr

-- Convert train speed from km/hr to m/s
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

-- Time to cross an electric pole
theorem train_cross_time : (train_length / train_speed_ms) = 10 := by
  sorry

end train_cross_time_l37_37334


namespace stratified_sampling_l37_37795

theorem stratified_sampling :
  let total_employees := 150
  let middle_managers := 30
  let senior_managers := 10
  let selected_employees := 30
  let selection_probability := selected_employees / total_employees
  let selected_middle_managers := middle_managers * selection_probability
  let selected_senior_managers := senior_managers * selection_probability
  selected_middle_managers = 6 ∧ selected_senior_managers = 2 :=
by
  sorry

end stratified_sampling_l37_37795


namespace f_increasing_f_t_range_l37_37488

noncomputable def f : Real → Real :=
  sorry

axiom f_prop1 : f 2 = 1
axiom f_prop2 : ∀ x, x > 1 → f x > 0
axiom f_prop3 : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

theorem f_increasing (x1 x2 : Real) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) : f x1 < f x2 := by
  sorry

theorem f_t_range (t : Real) (ht : t > 0) (ht3 : t - 3 > 0) (hf : f t + f (t - 3) ≤ 2) : 3 < t ∧ t ≤ 4 := by
  sorry

end f_increasing_f_t_range_l37_37488


namespace remaining_pencils_l37_37195

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l37_37195


namespace ratio_of_areas_of_inscribed_figures_l37_37349

theorem ratio_of_areas_of_inscribed_figures
  (r : ℝ) 
  (a : ℝ) 
  (b : ℝ)
  (h_triangle : a = 2 * sqrt 3 * r)
  (h_square : b = sqrt 2 * r) :
  (sqrt 3 * a^2 / 4) / (b^2) = 3 * sqrt 3 / 2 := by
  sorry

end ratio_of_areas_of_inscribed_figures_l37_37349


namespace sum_S5_l37_37686

-- Geometric sequence definitions and conditions
noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def sum_of_geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions translated into Lean:
-- a2 * a3 = 2 * a1
def condition1 := (geometric_sequence a r 1) * (geometric_sequence a r 2) = 2 * a

-- Arithmetic mean of a4 and 2 * a7 is 5/4
def condition2 := (geometric_sequence a r 3 + 2 * geometric_sequence a r 6) / 2 = 5 / 4

-- The final goal proving that S5 = 31
theorem sum_S5 (h1 : condition1 a r) (h2 : condition2 a r) : sum_of_geometric_sequence a r 5 = 31 := by
  apply sorry

end sum_S5_l37_37686


namespace price_of_Roger_cookie_is_13_33_l37_37464

variables (radius count_A count_R : ℕ) (total_sales per_person_sales price : ℝ)
variables (area_A_1 total_area_A : ℝ)

-- Define the given conditions
def Art_cookies_radius : radius = 2 := rfl
def Art_cookies_count : count_A = 10 := rfl
def Roger_cookies_count : count_R = 15 := rfl
def Total_sales : total_sales = 800 := rfl
def Equal_sales_per_person : per_person_sales = total_sales / 4 := rfl

-- Calculate area of one of Art's cookies
def Area_Art_cookie : area_A_1 = Real.pi * radius^2 := rfl
def Total_area_Art_cookies : total_area_A = area_A_1 * count_A := rfl

-- Express price per Roger's cookie
def Roger_cookie_price : price = per_person_sales / count_R := rfl

-- Prove the correct answer
theorem price_of_Roger_cookie_is_13_33 : price = 200 / 15 := by sorry

end price_of_Roger_cookie_is_13_33_l37_37464


namespace smallest_k_even_rightmost_nonzero_digit_l37_37221

noncomputable def b (n : ℕ) : ℕ :=
  if n > 0 then (Nat.factorial (n + 14)) / (Nat.factorial (n - 1)) else 0

def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  if n = 0 then 0 else (List.find! (λ d, d ≠ 0) (n.digits 10)).getD 0

theorem smallest_k_even_rightmost_nonzero_digit :
  ∃ k : ℕ, k > 0 ∧ rightmost_nonzero_digit (b k) % 2 = 0 :=
by
  use 1
  sorry

end smallest_k_even_rightmost_nonzero_digit_l37_37221


namespace circle_problem_l37_37920

noncomputable def circle_equation (a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_problem :
  ∃ a b r : ℝ,
    (b = 2 * a) ∧
    (abs (4 * a - 3 * b - 11) / 5 = r) ∧
    ((2 - a) ^ 2 + (-1 - b) ^ 2 = r ^ 2) ∧
    (circle_equation a b r = circle_equation (2 / 11) (4 / 11) (25 / 11)) :=
sorry

end circle_problem_l37_37920


namespace annes_initial_bottle_caps_l37_37010

-- Define the conditions
def albert_bottle_caps : ℕ := 9
def annes_added_bottle_caps : ℕ := 5
def annes_total_bottle_caps : ℕ := 15

-- Question (to prove)
theorem annes_initial_bottle_caps :
  annes_total_bottle_caps - annes_added_bottle_caps = 10 :=
by sorry

end annes_initial_bottle_caps_l37_37010


namespace volume_not_occupied_l37_37300

-- Definitions of the conditions
def radius_of_cones_and_sphere := 10
def height_of_cones := 15
def height_of_cylinder := 30

def volume_of_cylinder : ℝ :=
  π * radius_of_cones_and_sphere ^ 2 * height_of_cylinder

def volume_of_one_cone : ℝ :=
  (1 / 3) * π * radius_of_cones_and_sphere ^ 2 * height_of_cones

def total_volume_of_cones : ℝ :=
  2 * volume_of_one_cone

def radius_of_sphere : ℝ :=
  radius_of_cones_and_sphere / 2

def volume_of_sphere : ℝ :=
  (4 / 3) * π * radius_of_sphere ^ 3

-- Statement to be proved
theorem volume_not_occupied : volume_of_cylinder - total_volume_of_cones - volume_of_sphere = (5500 / 3) * π :=
by sorry

end volume_not_occupied_l37_37300


namespace trig_identity_sum_product_l37_37035

theorem trig_identity_sum_product :
  ∀ (x : ℝ), cos x + cos (5 * x) + cos (11 * x) + cos (15 * x) = 4 * cos (8 * x) * cos (5 * x) * cos (4 * x) :=
by
  intros x
  sorry

end trig_identity_sum_product_l37_37035


namespace real_solution_l37_37434

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end real_solution_l37_37434


namespace smallest_square_area_with_three_interior_lattice_points_l37_37382

theorem smallest_square_area_with_three_interior_lattice_points :
  ∃ (s : ℝ), (∃ (a : ℝ), s = a * Math.sqrt 2 ∧ 2 * a * a = 3) ∧ s^2 = 8 :=
by
  sorry

end smallest_square_area_with_three_interior_lattice_points_l37_37382


namespace count_three_digit_multiples_of_35_l37_37150

theorem count_three_digit_multiples_of_35 : 
  ∃ n : ℕ, n = 26 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → (x % 35 = 0 → x = 35 * (3 + ((x / 35) - 3))) := 
sorry

end count_three_digit_multiples_of_35_l37_37150


namespace points_on_same_circle_l37_37981
open Real

theorem points_on_same_circle (m : ℝ) :
  ∃ D E F, 
  (2^2 + 1^2 + 2 * D + 1 * E + F = 0) ∧
  (4^2 + 2^2 + 4 * D + 2 * E + F = 0) ∧
  (3^2 + 4^2 + 3 * D + 4 * E + F = 0) ∧
  (1^2 + m^2 + 1 * D + m * E + F = 0) →
  (m = 2 ∨ m = 3) := 
sorry

end points_on_same_circle_l37_37981


namespace mike_oranges_l37_37403

-- Definitions and conditions
variables (O A B : ℕ)
def condition1 := A = 2 * O
def condition2 := B = O + A
def condition3 := O + A + B = 18

-- Theorem to prove that Mike received 3 oranges
theorem mike_oranges (h1 : condition1 O A) (h2 : condition2 O A B) (h3 : condition3 O A B) : 
  O = 3 := 
by 
  sorry

end mike_oranges_l37_37403


namespace product_roots_evaluation_l37_37420

noncomputable def P (x : ℂ) : ℂ :=
  ∏ k in finset.range 1 8 \setminus {1}, (x - complex.exp (2 * real.pi * complex.I * k / 8))

theorem product_roots_evaluation :
  ∏ j in finset.range 1 9, ∏ k in finset.range 1 8, (complex.exp (2 * real.pi * complex.I * j / 9) - complex.exp (2 * real.pi * complex.I * k / 8)) = 1 :=
by
  sorry

end product_roots_evaluation_l37_37420


namespace area_bounded_curves_l37_37867

open Real

theorem area_bounded_curves :
  ∫ x in 0..1, (2 * x - x) + ∫ x in 1..2, (2 * x - x^2) = 7 / 6 :=
by
  sorry

end area_bounded_curves_l37_37867


namespace unique_polynomial_satisfying_condition_l37_37680

noncomputable def num_polynomials_satisfying_condition : ℕ :=
  if ∃ (f : polynomial ℂ), (∀ (x : ℂ), f(x^2) = (f(x))^2 ∧ f(x^2) = f(f(x))) 
  then 1 else 0

theorem unique_polynomial_satisfying_condition :
  num_polynomials_satisfying_condition = 1 :=
sorry

end unique_polynomial_satisfying_condition_l37_37680


namespace man_l37_37362

-- Defining the conditions
def man's_speed_in_still_water := 12.5
def current_speed := 2.5
def man's_speed_against_current := 10

-- Theorem statement
theorem man's_speed_with_current :
  (man's_speed_in_still_water - current_speed = man's_speed_against_current) →
  (man's_speed_in_still_water + current_speed = 15) :=
by
  -- conditions
  intro h1,
  -- the statement follows from the conditions
  sorry

end man_l37_37362


namespace b_is_arithmetic_a_general_formula_l37_37582

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n + 2 ^ n

def b (n : ℕ) : ℕ := a n / 2^(n-1)

theorem b_is_arithmetic :
  ∀ n : ℕ, b (n + 1) = b n + 1 := 
by sorry

theorem a_general_formula :
  ∀ n : ℕ, a n = n * 2^(n-1) := 
by sorry

end b_is_arithmetic_a_general_formula_l37_37582


namespace smallest_k_inequality_l37_37852

theorem smallest_k_inequality (y1 y2 y3 A : ℝ) (h_sum : y1 + y2 + y3 = 0) (h_max : A = max (|y1|) (max (|y2|) (|y3|))) :
  ∃ k : ℝ, k = 1.5 ∧ ∀ (y1 y2 y3 : ℝ), y1 + y2 + y3 = 0 → A = max (|y1|) (max (|y2|) (|y3|)) → y1^2 + y2^2 + y3^2 ≥ k * A^2 := 
begin
  use 1.5,
  intros y1 y2 y3 h_sum h_max,
  sorry
end

end smallest_k_inequality_l37_37852


namespace number_of_ways_to_draw_4_from_15_l37_37750

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37750


namespace number_of_sandwiches_l37_37663

-- Definitions based on conditions
def breads : Nat := 5
def meats : Nat := 7
def cheeses : Nat := 6
def total_sandwiches : Nat := breads * meats * cheeses
def turkey_mozzarella_exclusions : Nat := breads
def rye_beef_exclusions : Nat := cheeses

-- The proof problem statement
theorem number_of_sandwiches (total_sandwiches := 210) 
  (turkey_mozzarella_exclusions := 5) 
  (rye_beef_exclusions := 6) : 
  total_sandwiches - turkey_mozzarella_exclusions - rye_beef_exclusions = 199 := 
by sorry

end number_of_sandwiches_l37_37663


namespace similar_triangles_PQ_length_l37_37297

theorem similar_triangles_PQ_length (XY YZ QR : ℝ) (hXY : XY = 8) (hYZ : YZ = 16) (hQR : QR = 24)
  (hSimilar : ∃ (k : ℝ), XY = k * 8 ∧ YZ = k * 16 ∧ QR = k * 24) : (∃ (PQ : ℝ), PQ = 12) :=
by 
  -- Here we need to prove the theorem using similarity and given equalities
  sorry

end similar_triangles_PQ_length_l37_37297


namespace slowerPainterDuration_l37_37380

def slowerPainterStartTime : ℝ := 14 -- 2:00 PM in 24-hour format
def fasterPainterStartTime : ℝ := slowerPainterStartTime + 3 -- 3 hours later
def finishTime : ℝ := 24.6 -- 0.6 hours past midnight

theorem slowerPainterDuration :
  finishTime - slowerPainterStartTime = 10.6 :=
by
  sorry

end slowerPainterDuration_l37_37380


namespace maximum_value_of_magnitude_l37_37855

-- Define Euler's formula
def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

-- Define the magnitude of a complex number
def magnitude (z : ℂ) : ℝ := complex.abs z

-- The mathematical statement to be proved
theorem maximum_value_of_magnitude : 
  ∃ x : ℝ, (magnitude (euler_formula x - 2) = 3) :=
sorry

end maximum_value_of_magnitude_l37_37855


namespace directrix_of_parabola_l37_37894

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37894


namespace smallest_product_l37_37451

theorem smallest_product : 
  ∃ (x y ∈ ({-10, -3, 0, 2, 6} : Set ℤ)), x * y = -60 ∧ 
  (∀ (a b ∈ ({-10, -3, 0, 2, 6} : Set ℤ)), a * b ≥ x * y) :=
sorry

end smallest_product_l37_37451


namespace ratio_of_sums_l37_37926

variables {α : Type*} [LinearOrderedField α] 
variables {a1 d : α}

def a (n : ℕ) : α := a1 + (n - 1) * d
def S (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

theorem ratio_of_sums (h : (a 2) / (a 3) = 5 / 2) : (S 3) / (S 5) = 3 / 2 :=
by
  sorry

end ratio_of_sums_l37_37926


namespace cos_sin_identity_l37_37460

theorem cos_sin_identity (n : ℕ) (t : ℝ) (hn : 1 ≤ n ∧ n ≤ 500) :
  (complex.cos t - complex.i * complex.sin t) ^ n = complex.cos (n * t) - complex.i * complex.sin (n * t) := 
sorry

end cos_sin_identity_l37_37460


namespace john_must_work_10_more_days_l37_37203

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l37_37203


namespace complex_number_z_l37_37124

variable (x a b : ℝ)

-- Conditions
axiom eq_real_root : x^2 + (4 + complex.i) * x + 4 + a * complex.i = 0
axiom a_real : a ∈ ℝ
axiom z_def : z = a + b * complex.i

-- Expected Result
theorem complex_number_z (x a b : ℝ) (h₁ : x^2 + (4 + complex.i) * x + 4 + a * complex.i = 0) (h₂ : a ∈ ℝ) (h₃ : z = a + b * complex.i) : z = 2 - 2 * complex.i := 
sorry

end complex_number_z_l37_37124


namespace odd_segments_diff_types_l37_37144

noncomputable theory

open Int Real

def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

def irrational (x : ℝ) : Prop := ¬ rational x

def segment_type (x y : ℝ) :=
  if rational x ∧ irrational y ∨ irrational x ∧ rational y
  then -1
  else 1

theorem odd_segments_diff_types {n : ℕ} (points : Fin (n+2) → ℝ)
  (points_sorted : ∀ i j : Fin (n+2), i < j → points i < points j)
  (h0 : points 0 = 1)
  (hn : points (Fin.ofNat (n+1)) = sqrt 2)
  : ∃ k : ℕ, (∏ i in Finset.range (n + 1), segment_type (points i) (points (i+1))) = (-1)^k ∧ k % 2 = 1 := 
  sorry

end odd_segments_diff_types_l37_37144


namespace pumps_empty_pool_together_in_144_minutes_l37_37735

theorem pumps_empty_pool_together_in_144_minutes : 
  (∀ t: ℝ, (0 < t →  t = 4) → t ≥ 0 → (∀ t2: ℝ, (0 < t2 → t2 = 6) → t2 ≥ 0) → ∀ t1 t2 t3: ℝ, t3 = (1 / ((1/4) + (1/6)) * 60) → t3 = 144  := 
by sorry

end pumps_empty_pool_together_in_144_minutes_l37_37735


namespace directrix_eqn_of_parabola_l37_37874

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37874


namespace cannot_return_l37_37177

def valid_moves (p : (ℝ × ℝ)) : set (ℝ × ℝ) :=
  {q | (q.1 = p.1 ∧ (q.2 = p.2 + 2 * p.1 ∨ q.2 = p.2 - 2 * p.1)) ∨ 
       (q.2 = p.2 ∧ (q.1 = p.1 + 2 * p.2 ∨ q.1 = p.1 - 2 * p.2))}

def no_immediate_return (p q r : (ℝ × ℝ)) : Prop :=
  ¬(q = r) ∨ ¬(q = p)

def step (i : ℕ) (start : ℝ × ℝ) (path: (fin i) → (ℝ × ℝ)) : Prop :=
  path 0 = start ∧
  (∀ j : fin i, path j ∈ valid_moves (path (j - 1)) ∧ no_immediate_return (path (j - 2)) (path (j - 1)) (path j))

theorem cannot_return (i : ℕ) : ∀ (path : (fin i) → (ℝ × ℝ)),
  step i (1, real.sqrt 2) path → path (i-1) ≠ (1, real.sqrt 2) :=
sorry

end cannot_return_l37_37177


namespace determine_m_l37_37215

theorem determine_m (a b c m : ℤ) 
  (h1 : c = -4 * a - 2 * b)
  (h2 : 70 < 4 * (8 * a + b) ∧ 4 * (8 * a + b) < 80)
  (h3 : 110 < 5 * (9 * a + b) ∧ 5 * (9 * a + b) < 120)
  (h4 : 2000 * m < (2500 * a + 50 * b + c) ∧ (2500 * a + 50 * b + c) < 2000 * (m + 1)) :
  m = 5 := sorry

end determine_m_l37_37215


namespace sum_of_areas_lt_side_length_square_l37_37122

variable (n : ℕ) (a : ℝ)
variable (S : Fin n → ℝ) (d : Fin n → ℝ)

-- Conditions
axiom areas_le_one : ∀ i, S i ≤ 1
axiom sum_d_le_a : (Finset.univ).sum d ≤ a
axiom areas_less_than_diameters : ∀ i, S i < d i

-- Theorem Statement
theorem sum_of_areas_lt_side_length_square :
  ((Finset.univ : Finset (Fin n)).sum S) < a :=
sorry

end sum_of_areas_lt_side_length_square_l37_37122


namespace michael_passes_donovan_l37_37332

noncomputable def track_length : ℕ := 400 -- in meters
noncomputable def donovan_lap_time : ℕ := 48 -- in seconds
noncomputable def michael_lap_time : ℕ := 40 -- in seconds

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ) :
  let donovan_speed := track_length / donovan_lap_time,
      michael_speed := track_length / michael_lap_time,
      speed_difference := michael_speed - donovan_speed,
      time_to_gain := track_length / speed_difference,
      laps_completed := time_to_gain / michael_lap_time in
  laps_completed.ceil = 6 :=
by
  sorry

end michael_passes_donovan_l37_37332


namespace draw_4_balls_in_order_ways_l37_37773

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37773


namespace find_f_of_3_l37_37477

def f : ℝ → ℝ 
| x := if x >= 6 then x - 4 else f (x + 2)

theorem find_f_of_3 : f 3 = 3 := sorry

end find_f_of_3_l37_37477


namespace problem_statement_l37_37312

noncomputable def f : ℚ := 5 - (-3:ℚ)^(-3)

theorem problem_statement : f = 136 / 27 := 
by 
  sorry

end problem_statement_l37_37312


namespace frederick_final_amount_l37_37315

-- Definitions of conditions
def P : ℝ := 2000
def r : ℝ := 0.05
def n : ℕ := 18

-- Define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Theorem stating the question's answer
theorem frederick_final_amount : compound_interest P r n = 4813.24 :=
by
  sorry

end frederick_final_amount_l37_37315


namespace even_terms_in_binomial_expansion_l37_37972

-- Define m to be an even integer
def is_even (m : ℤ) : Prop := ∃ k : ℤ, m = 2 * k

-- Define n to be an odd integer
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Lean statement of the proof problem
theorem even_terms_in_binomial_expansion (m n : ℤ) (hm : is_even m) (hn : is_odd n) : 
  count_even_terms_in_expansion ((m + n)^8) = 8 :=
sorry

end even_terms_in_binomial_expansion_l37_37972


namespace ball_drawing_ways_l37_37764

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37764


namespace max_zeros_sum_end_l37_37466

/-- 
  From the digits {1, 2, 3, 4, 5, 6, 7, 8, 9}, nine 9-digit numbers (not necessarily different) are 
  formed with each digit used exactly once in each number. We need to prove that the largest number 
  of zeros that the sum of these nine numbers can end with is 8.
-/
theorem max_zeros_sum_end (numbers : Fin 9 → Fin 10⁹) 
(h_valid_nums : ∀ i, is_valid_number (numbers i))
(h_divisible_by_9 : ∀ i, numbers i % 9 = 0) :
  ∃ k, (∑ i, numbers i) = k * 10^8 ∧ (∑ i, numbers i) % 10^(8 + 1) = 0 := sorry

/-
 - Definition to assert the is_valid_number criteria. The digit constraints ensure every digit 
   from [1..9] appears exactly once per number formed.
-/
def is_valid_number (n : Fin 10⁹) : Prop :=
  let digits := List.range 1 10 in
  ∃ (permutation : List ℕ), permutation ~ digits ∧
  (∀ (d ∈ permutation), n.digits Nat.pred d) -- ensuring each digit once.

-- Note: Detailed digit validation can be added if necessary within this constraint scope.

end max_zeros_sum_end_l37_37466


namespace tangent_line_eqn_l37_37691

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derive (x : ℝ) : ℝ := (1 / x) + 1

theorem tangent_line_eqn :
  (∀ m x₀ y₀ : ℝ, (derive x₀ = m) → (y₀ = curve x₀) → (∀ x : ℝ, y₀ + m * (x - x₀) = 2 * x)) :=
by
  sorry

end tangent_line_eqn_l37_37691


namespace sum_groups_l37_37482

variables {n : ℕ} (a : Fin n → ℝ)

def non_decreasing (a : Fin n → ℝ) : Prop :=
  ∀ i j, i ≤ j → a i ≤ a j

theorem sum_groups (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_nondec : non_decreasing a):
    ∃ groups : List (List (Fin n → ℝ)), 
      (∀ g ∈ groups, ∀ x y ∈ g, x ≠ y → 1 ≤ y / x ∧ y / x ≤ 2) ∧ 
      (∃ partition : List (List (List (Fin n → ℝ))),
        partition.length = n ∧ 
        ∀ i < partition.length, 
          ∀ g ∈ partition.get! i, 
            groups.contains g) := 
sorry

end sum_groups_l37_37482


namespace courtyard_brick_problem_l37_37329

noncomputable def area_courtyard (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_brick (length width : ℝ) : ℝ :=
  length * width

noncomputable def total_bricks_required (court_area brick_area : ℝ) : ℝ :=
  court_area / brick_area

theorem courtyard_brick_problem 
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ)
  (H1 : courtyard_length = 18)
  (H2 : courtyard_width = 12)
  (H3 : brick_length = 15 / 100)
  (H4 : brick_width = 13 / 100) :
  
  total_bricks_required (area_courtyard courtyard_length courtyard_width * 10000) 
                        (area_brick brick_length brick_width) 
  = 11077 :=
by
  sorry

end courtyard_brick_problem_l37_37329


namespace problem_statement_l37_37145

noncomputable def e1 : ℝ^3 := ⟨1, 0, 0⟩
noncomputable def e2 : ℝ^3 := ⟨0, 1, 0⟩
noncomputable def e3 : ℝ^3 := ⟨0, 0, 1⟩

noncomputable def a : ℝ^3 := e1 + e2 + 2 • e3
noncomputable def b : ℝ^3 := e1 + e2 - e3
noncomputable def c : ℝ^3 := e1 + e2 - 2 • e3

theorem problem_statement :
  (inner a b = 0) ∧ 
  (∃ k : ℝ, (2 • c + a) = k • b) → False ∧
  (cos_angle b c = (2 * sqrt 2) / 3) ∧ 
  (∃ λ μ : ℝ, a = λ • b + μ • c ∧ b = μ • c + λ • a ∧ c = λ • a + μ • b) :=
by 
  sorry 

end problem_statement_l37_37145


namespace cyclic_quadrilateral_solution_l37_37681

noncomputable def cyclic_quadrilateral_problem 
  (A B C D P : Type)
  [CirclePoints A B C D P] -- Assumption that A, B, C, D, P lie on a circle
  (AngleEqual1 : angle A P B = angle B P C)
  (AngleEqual2 : angle B P C = angle C P D)
  (a b c d : ℝ) -- lengths of segments
  (H1 : dist A B = a)
  (H2 : dist B C = b)
  (H3 : dist C D = c)
  (H4 : dist D A = d)
  : Prop :=
  (a + c) / (b + d) = b / c

theorem cyclic_quadrilateral_solution 
  {A B C D P : Type}
  [CirclePoints A B C D P]
  {AngleEqual1 : angle A P B = angle B P C}
  {AngleEqual2 : angle B P C = angle C P D}
  {a b c d : ℝ}
  {H1 : dist A B = a}
  {H2 : dist B C = b}
  {H3 : dist C D = c}
  {H4 : dist D A = d} :
  (cyclic_quadrilateral_problem A B C D P AngleEqual1 AngleEqual2 a b c d H1 H2 H3 H4) := sorry

end cyclic_quadrilateral_solution_l37_37681


namespace draw_4_balls_in_order_l37_37778

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37778


namespace total_time_equiv_7_75_l37_37596

def acclimation_period : ℝ := 1
def learning_basics : ℝ := 2
def research_time_without_sabbatical : ℝ := learning_basics + 0.75 * learning_basics
def sabbatical : ℝ := 0.5
def research_time_with_sabbatical : ℝ := research_time_without_sabbatical + sabbatical
def dissertation_without_conference : ℝ := 0.5 * acclimation_period
def conference : ℝ := 0.25
def dissertation_with_conference : ℝ := dissertation_without_conference + conference
def total_time : ℝ := acclimation_period + learning_basics + research_time_with_sabbatical + dissertation_with_conference

theorem total_time_equiv_7_75 : total_time = 7.75 := by
  sorry

end total_time_equiv_7_75_l37_37596


namespace power_addition_l37_37551

theorem power_addition {x : ℝ} (h : 3^x = 5) : 3^(x + 2) = 45 :=
by sorry

end power_addition_l37_37551


namespace draw_4_balls_in_order_ways_l37_37772

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37772


namespace john_work_more_days_l37_37199

theorem john_work_more_days (days_worked : ℕ) (amount_made : ℕ) (daily_earnings : ℕ) (h1 : days_worked = 10) (h2 : amount_made = 250) (h3 : daily_earnings = amount_made / days_worked) : 
  ∃ more_days : ℕ, more_days = (2 * amount_made / daily_earnings) - days_worked := 
by
  have h4 : daily_earnings = 25 := by {
    rw [h1, h2],
    norm_num,
  }
  have h5 : 2 * amount_made / daily_earnings = 20 := by {
    rw [h2, h4],
    norm_num,
  }
  use 10
  rw [h1, h5]
  norm_num

end john_work_more_days_l37_37199


namespace f_neg_l37_37113

-- Define f(x) for x > 0
def f_pos (x : ℝ) (hx : 0 < x) : ℝ := x^2 + x + 1

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem f_neg (f : ℝ → ℝ)
  (hodd : odd_function f)
  (hfx_pos : ∀ x, 0 < x → f(x) = f_pos x hx)
  (x : ℝ) (hx : x < 0) : f(x) = -x^2 + x - 1 := 
sorry

end f_neg_l37_37113


namespace intersection_point_for_m_l37_37927

variable (n : ℕ) (x_0 y_0 : ℕ)
variable (h₁ : n ≥ 2)
variable (h₂ : y_0 ^ 2 = n * x_0 - 1)
variable (h₃ : y_0 = x_0)

theorem intersection_point_for_m (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k ≥ 2 ∧ (y_0 ^ m = x_0 ^ m) ∧ (y_0 ^ m) ^ 2 = k * (x_0 ^ m) - 1 :=
by
  sorry

end intersection_point_for_m_l37_37927


namespace intersection_complement_l37_37628

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l37_37628


namespace geom_sequence_eq_l37_37083

theorem geom_sequence_eq :
  ∀ {a : ℕ → ℝ} {q : ℝ}, (∀ n, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by
  intro a q hgeom hsum hsum_sq
  sorry

end geom_sequence_eq_l37_37083


namespace problem_statement_l37_37475

open Real

noncomputable def a : ℝ := log 11 / log 5
noncomputable def b : ℝ := (log 8 / log 2) / 2
noncomputable def c : ℝ := sqrt real.e

theorem problem_statement : a < b ∧ b < c :=
by {
  -- Definitions
  have ha : a = log 11 / log 5 := by rfl,
  have hb : b = (log 8 / log 2) / 2 := by rfl,
  have hc : c = sqrt real.e := by rfl,

  -- Proof
  sorry
}

end problem_statement_l37_37475


namespace class_tree_total_l37_37745

theorem class_tree_total
  (trees_A : ℕ)
  (trees_B : ℕ)
  (hA : trees_A = 8)
  (hB : trees_B = 7)
  : trees_A + trees_B = 15 := 
by
  sorry

end class_tree_total_l37_37745


namespace permits_stamped_l37_37399

def appointments : ℕ := 2
def hours_per_appointment : ℕ := 3
def workday_hours : ℕ := 8
def stamps_per_hour : ℕ := 50

theorem permits_stamped :
  let total_appointment_hours := appointments * hours_per_appointment in
  let stamping_hours := workday_hours - total_appointment_hours in
  let total_permits := stamping_hours * stamps_per_hour in
  total_permits = 100 :=
by
  sorry

end permits_stamped_l37_37399


namespace tangent_line_equation_l37_37694

noncomputable def curve (x : ℝ) : ℝ :=
  Real.log x + x + 1

theorem tangent_line_equation : ∃ x y : ℝ, derivative curve x = 2 ∧ curve x = y ∧ y = 2 * x := 
begin
  sorry
end

end tangent_line_equation_l37_37694


namespace cos_alpha_eq_sqrt_ten_div_ten_l37_37573

noncomputable def point_a : ℝ × ℝ := (2, 1)

noncomputable def rotation (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := 
((p.1 * Real.cos θ - p.2 * Real.sin θ),
 (p.1 * Real.sin θ + p.2 * Real.cos θ))

noncomputable def point_b : ℝ × ℝ := rotation (Real.pi / 4) point_a

noncomputable def angle_of_inclination (p : ℝ × ℝ) : ℝ :=
Real.atan (p.2 / p.1)

noncomputable def alpha : ℝ := angle_of_inclination point_b

theorem cos_alpha_eq_sqrt_ten_div_ten : Real.cos alpha = ↑(Real.sqrt 10) / 10 := 
sorry

end cos_alpha_eq_sqrt_ten_div_ten_l37_37573


namespace find_m_l37_37147

noncomputable def a : (ℝ × ℝ) := (1, m)
noncomputable def b : (ℝ × ℝ) := (3, -2)
noncomputable def perp_condition := (4, m - 2).fst * (3) + (4, m - 2).snd * (-2) = 0

theorem find_m (m : ℝ) : perp_condition → m = 8 :=
by
  -- Insert proof here
  sorry

end find_m_l37_37147


namespace problem_l37_37969

theorem problem (a : ℝ) (h : a^2 - 2 * a - 1 = 0) : -3 * a^2 + 6 * a + 5 = 2 := by
  sorry

end problem_l37_37969


namespace remainder_of_2_pow_2018_plus_2019_mod_5_l37_37449

theorem remainder_of_2_pow_2018_plus_2019_mod_5 : (2^2018 + 2019) % 5 = 3 :=
by
  -- Euler's Theorem condition
  have euler : ∀ a n, nat.gcd a n = 1 → ∃ k, a^k % n = 1 := sorry

  -- φ(5) = 4
  have phi5 : nat.totient 5 = 4 := sorry

  -- 2^4 ≡ 1 (mod 5)
  have pow4mod5 : 2^4 % 5 = 1 := sorry

  -- 2018 = 4 * 504 + 2
  have eq2018 : 2018 = 4 * 504 + 2 := by
    norm_num

  -- Prove 2^2018 % 5 = 4
  have pow2018mod5 : 2^2018 % 5 = 4 := sorry

  -- Prove 2019 % 5 = 4
  have rem2019mod5 : 2019 % 5 = 4 := by
    norm_num
  
  -- Combine both results
  have result : ((2^2018 % 5) + (2019 % 5)) % 5 = (4 + 4) % 5 := by
    rw [pow2018mod5, rem2019mod5]
    norm_num

  -- Prove final result
  exact result

end remainder_of_2_pow_2018_plus_2019_mod_5_l37_37449


namespace probability_third_shot_l37_37002

-- Definition of probability of hitting the target and number of shots
def p := 0.9
def n := 4

-- Define the condition for independent shots
def shots_independent (hits : Finₓ n → Prop) : Prop := 
  ∀ i j, i ≠ j → hits i ↔ hits j

-- Define the condition for hitting the target
def hit_target (i : Finₓ n) : Prop := Prob (i = 3) = p

-- Lean statement of the proof problem
theorem probability_third_shot (h : shots_independent hit_target) : 
  Prob (hit_target (Finₓ.mk 2 sorry)) = p :=
sorry

end probability_third_shot_l37_37002


namespace problem_l37_37257

theorem problem (b : ℝ) (h : 5 = b + b⁻¹) : b^6 + b⁻⁶ = 12098 := by
  sorry

end problem_l37_37257


namespace total_weight_moved_l37_37598

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l37_37598


namespace solve_sum_of_digits_eq_2018_l37_37622

def s (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem solve_sum_of_digits_eq_2018 : ∃ n : ℕ, n + s n = 2018 := by
  sorry

end solve_sum_of_digits_eq_2018_l37_37622


namespace ratio_of_3_numbers_l37_37683

variable (A B C : ℕ)
variable (k : ℕ)

theorem ratio_of_3_numbers (h₁ : A = 5 * k) (h₂ : B = k) (h₃ : C = 4 * k) (h_sum : A + B + C = 1000) : C = 400 :=
  sorry

end ratio_of_3_numbers_l37_37683


namespace problem_statement_l37_37567

-- Definitions
variables {A B C D E F I M : Point}
variables {Ω : Circle}

-- Conditions
def acute_triangle (ABC : Triangle) : Prop :=
  ABC.angle_ABC < 90 ∧ ABC.angle_ACB < 90 ∧ ABC.angle_BAC < 90

def AB_gt_AC (ABC : Triangle) : Prop :=
  ABC.side_AB > ABC.side_AC

def incenter {A B C : Point} (I : Point) : Prop :=
  -- I is the incenter of triangle ABC

def circumcircle (ABC : Triangle) (Ω : Circle) : Prop :=
  -- Ω is the circumcircle of triangle ABC

def perp_foot (A B C D : Point) : Prop :=
  -- D is the foot of the perpendicular from A to the line BC

def AI_intersects (A I Ω M : Point) : Prop :=
  -- AI intersects Γ at M, M != A

def perp_line_intersects (M A D E : Point) : Prop :=
  -- The line through M perpendicular to AM intersects AD at E

def perp_from_incenter (I A D F : Point) : Prop :=
  -- F is the foot of the perpendicular from I to AD

axiom acute_triangle_ABC : acute_triangle ABC
axiom AB_gt_AC_ABC : AB_gt_AC ABC
axiom incenter_exists : incenter I
axiom circumcircle_exists : circumcircle ABC Ω
axiom perp_foot_exists : perp_foot A B C D
axiom AI_intersects_circumcircle : AI_intersects A I Ω M
axiom perp_line_intersects_AD : perp_line_intersects M A D E
axiom perp_from_incenter_AD : perp_from_incenter I A D F

theorem problem_statement :
  ID * AM = IE * AF :=
sorry


end problem_statement_l37_37567


namespace draw_4_balls_ordered_l37_37759

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37759


namespace ratio_F_J_l37_37021

noncomputable def O : ℕ := 5
noncomputable def J : ℕ := 2 * O + 2
noncomputable def F : ℕ := 20 - O - J

theorem ratio_F_J : F / J = 1 / 4 :=
by
  let O := 5
  let J := 2 * O + 2
  let F := 20 - O - J
  have hO : O = 5 := rfl
  have hJ : J = 12 := by simp [J, hO]
  have hF : F = 3 := by simp [F, hO, hJ]
  calc
    F / J = 3 / 12 := by simp [F, J, hO, hJ, hF]
    ...   = 1 / 4  := by norm_num
  sorry

end ratio_F_J_l37_37021


namespace probability_two_balls_same_color_l37_37343

theorem probability_two_balls_same_color:
  let total_balls := 10 in
  let blue_balls := 5 in
  let yellow_balls := 5 in
  let prob_two_blues := (blue_balls / total_balls) * (blue_balls / total_balls) in
  let prob_two_yellows := (yellow_balls / total_balls) * (yellow_balls / total_balls) in
  let prob_same_color := prob_two_blues + prob_two_yellows in
  prob_same_color = 1 / 2 :=
by
  sorry

end probability_two_balls_same_color_l37_37343


namespace length_of_bridge_l37_37385

/-
Problem: A train 100 meters long completely crosses a bridge of a certain length in 24 seconds. The speed of the train is 60 km/h. How long is the bridge?

Conditions:
1. The train is 100 meters long.
2. The train completely crosses the bridge in 24 seconds.
3. The speed of the train is 60 km/h, which is approximately 16.67 m/s.

Proof that the length of the bridge is 300.08 meters.
-/

theorem length_of_bridge (train_length : ℝ) (time_cross : ℝ) (train_speed_kmh : ℝ) :
  let train_speed := train_speed_kmh / 3.6,
      total_distance := train_speed * time_cross,
      bridge_length := total_distance - train_length in
  train_length = 100 ∧
  time_cross = 24 ∧
  train_speed_kmh = 60 →
  bridge_length = 300.08 :=
by
  intros
  sorry

end length_of_bridge_l37_37385


namespace sum_first_15_terms_bn_eq_135_l37_37942

theorem sum_first_15_terms_bn_eq_135
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1_a3 : a 1 + a 3 = 30)
  (h_S4  : S 4 = 120)
  (h_b_def: ∀ n, b n = 1 + Real.logb 3 (a n)) :
  ∑ n in Finset.range 15, b (n + 1) = 135 :=
sorry

end sum_first_15_terms_bn_eq_135_l37_37942


namespace directrix_of_given_parabola_l37_37887

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37887


namespace number_of_ways_to_draw_balls_l37_37789

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37789


namespace square_area_on_circle_with_tangent_l37_37303

/-- 
Given:
- The radius of a circle is 5 cm.
- Two vertices of a square lie on this circle.
- The other two vertices lie on a tangent to the circle.

Prove the area of the square is 64 cm^2.
-/
theorem square_area_on_circle_with_tangent
  (R : ℝ) (R_eq_5 : R = 5) (square_side_length : ℝ) 
  (vertices_conditions : 
    (2 * vertices_conditions / 2 = R ∧ 
    ( ∀x y ∈ ℝ, x ≠ y ⊃ distance (x,R) = R ∨ tangent x ∉ distance same tangent)
  ) : square_side_length = (8) : square_area equations :
  pythagorean_theorem square:
) :
  let side := 8 in side ^ 2 = 64 :=
sorry

end square_area_on_circle_with_tangent_l37_37303


namespace eccentricity_of_hyperbola_l37_37952

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0) : ℝ :=
  let C := ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1
  let P := (2 * a, sqrt 3 * b)
  let F1 := (-sqrt (a^2 + b^2), 0)
  let F2 := (sqrt (a^2 + b^2), 0)
  let PF1 := sqrt ((P.1 + F1.1)^2 + F1.2^2)
  let PF2 := sqrt ((P.1 - F2.1)^2 + F2.2^2)
  if PF1 = 2 * PF2 then 3 / 2 else 0

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1)
  (intersectP : (2 * a, sqrt 3 * b))
  (focus_condition : |(√((intersectP.1 + sqrt(a^2 + b^2) )^2 + intersectP.2^2))| = 2 * |(√((intersectP.1 - sqrt(a^2 + b^2) )^2 + intersectP.2^2))|) :
  hyperbola_eccentricity  a b ha hb = 3 / 2 :=
  sorry

end eccentricity_of_hyperbola_l37_37952


namespace parabola_directrix_is_x_eq_1_l37_37889

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37889


namespace work_problem_l37_37737

theorem work_problem (W : ℝ) (d : ℝ) :
  (1 / 40) * d * W + (28 / 35) * W = W → d = 8 :=
by
  intro h
  sorry

end work_problem_l37_37737


namespace min_age_of_youngest_person_l37_37699

theorem min_age_of_youngest_person
  {a b c d e : ℕ}
  (h_sum : a + b + c + d + e = 256)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_diff : 2 ≤ (b - a) ∧ (b - a) ≤ 10 ∧ 
            2 ≤ (c - b) ∧ (c - b) ≤ 10 ∧ 
            2 ≤ (d - c) ∧ (d - c) ≤ 10 ∧ 
            2 ≤ (e - d) ∧ (e - d) ≤ 10) : 
  a = 32 :=
sorry

end min_age_of_youngest_person_l37_37699


namespace borrowed_amount_correct_l37_37805

def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ := principal * rate * time / 100

def total_gain_per_year : ℚ := 80

axiom borrowed_amount : ℚ

axiom condition1 : simple_interest borrowed_amount 4 2
axiom condition2 : simple_interest borrowed_amount 6 2

theorem borrowed_amount_correct :
  let interest_paid := simple_interest borrowed_amount 4 2,
      interest_earned := simple_interest borrowed_amount 6 2,
      total_gain := total_gain_per_year * 2 in
  interest_earned - interest_paid = total_gain →
  borrowed_amount = 2000 := sorry

end borrowed_amount_correct_l37_37805


namespace relationship_among_a_b_c_l37_37042

open Real

-- Define the terms
def a : ℝ := 0.3^2
def b : ℝ := log 0.3 / log 2 -- Using change of base formula for logarithms
def c : ℝ := 2^0.3

-- State the theorem
theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  -- sorry to skip the proof
  sorry

end relationship_among_a_b_c_l37_37042


namespace transformation_impossible_l37_37830

-- Define the value function for a word represented as a list of 0s and 1s
def value_function (A : List ℕ) : ℕ :=
  A.foldr (λ (a : ℕ) (acc : ℕ × ℕ), (acc.1 + a * acc.2, acc.2 + 1)) (0, 1)).1

-- The main theorem statement which states the problem in Lean
theorem transformation_impossible : 
  ∀ (A B : List ℕ),
  -- Conditions for operations
  ((∀ C : List ℕ, A = C ++ repeat C 3 ++ B) ∨ (∀ C : List ℕ, A = B ++ repeat C 3 ++ C)) →
  -- Target words
  A = [1, 0] → B = [0, 1] → 
  -- Conclusion that transformation is not possible
  value_function A % 3 ≠ value_function B % 3 :=
begin
  intros A B operations A_def B_def,
  rw [A_def, B_def],
  change value_function [1, 0] % 3 ≠ value_function [0, 1] % 3,
  simp [value_function],
  norm_num,
  sorry
end

end transformation_impossible_l37_37830


namespace explicit_form_of_f_zeros_of_f_l37_37489

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x - 3 / x + 2
else if x = 0 then 0
else x - 3 / x - 2

theorem explicit_form_of_f :
  ∀ x : ℝ, f(x) =
    if x < 0 then x - 3 / x + 2
    else if x = 0 then 0
    else x - 3 / x - 2 := by
  sorry

theorem zeros_of_f :
  {x : ℝ | f(x) = 0} = {-3, 0, 3} := by
  sorry

end explicit_form_of_f_zeros_of_f_l37_37489


namespace bonnie_roark_wire_ratio_l37_37414

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_piece_length := 8 in
  let bonnie_wire_pieces := 12 in
  let bonnie_wire_total_length := bonnie_wire_pieces * bonnie_wire_piece_length in
  let bonnie_cube_volume := bonnie_wire_piece_length ^ 3 in

  let roark_cube_side_length := 0.5 in
  let roark_cube_volume := (roark_cube_side_length) ^ 3 in
  let roark_cubes_needed := bonnie_cube_volume / roark_cube_volume in
  let roark_wire_piece_length := roark_cube_side_length in
  let roark_wire_pieces := 12 in
  let roark_wire_total_length := roark_cubes_needed * (roark_wire_pieces * roark_wire_piece_length) in

  (bonnie_wire_total_length / roark_wire_total_length) = (1/256) :=
by
  sorry

end bonnie_roark_wire_ratio_l37_37414


namespace directrix_eqn_of_parabola_l37_37871

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37871


namespace _l37_37216

noncomputable def standard_eqn_of_ellipse (x y : ℝ) (a b : ℝ) : Prop := 
  (x^2 / a) + (y^2 / b) = 1

@[simp] theorem find_standard_eqn_of_ellipse (P : ℝ → ℝ → Prop) (a b : ℝ) (h1 : a > b) (h2 : b > 0) (F1 F2 : ℝ → Prop)
  (eccentricity : ℝ) (h3 : P 1 0) (h4 : P F1 2) (h5 : P F2 2) (h6 : |F1 0| + |F2 0| = 4) (h7 : eccentricity = (sqrt 3) / 2) :
  standard_eqn_of_ellipse 4 1 4 1 :=
by
  sorry

noncomputable def value_of_m (x₁ x₂ y₁ y₂ : ℝ) : ℝ × ℝ := (-5/3, x₁)

@[simp] theorem find_value_of_m (P : ℝ → ℝ → Prop) (M N : ℝ × ℝ) (a b : ℝ) (h1 : P 1 0) (h2 : a > b) (h3 : b > 0) (F1 F2 : ℝ → Prop)
  (eccentricity : ℝ) (h4 : |F1 1| + |F2 1| = 4) (h5 : eccentricity = (sqrt 3) / 2) (h6 : P (1/2) (x₁ + x₂)) 
  (h7 : P (1/2) (y₁ + y₂)) :
  value_of_m x₁ x₂ y₁ y₂ = -5/3 :=
by
  sorry

end _l37_37216


namespace problem1_problem2_l37_37947

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

def is_increasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ < f x₂

theorem problem1 : is_increasing_on f {x | 1 ≤ x} := 
by sorry

def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂

theorem problem2 (g : ℝ → ℝ) (h_decreasing : is_decreasing g)
  (h_inequality : ∀ x : ℝ, 1 ≤ x → g (x^3 + 2) < g ((a^2 - 2 * a) * x)) :
  -1 < a ∧ a < 3 :=
by sorry

end problem1_problem2_l37_37947


namespace draw_4_balls_in_order_l37_37781

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37781


namespace sum_of_angles_S_R_l37_37640

-- Given conditions
variables {A B R E C : Type}
variable (circ : Set Point)
variable (BR RE : Angle)
variable (measures_BR measures_RE : ℝ)
variable (arc_BE : ℝ := measures_BR + measures_RE)
variable [circle : is_circle points := {A, B, R, E, C} ∈ circ]
variable [circ_basic : circle_arc BR]
variable (value_BR : circle_arc.measure BR = 48)
variable (value_RE : circle_arc.measure RE = 34)
variable [S R : Angle]
variables (AC : ℝ)

-- Problem: Prove that the sum of the measures of angles \( S \) and \( R \) is \( 41^\circ \)
theorem sum_of_angles_S_R : 
  let θ_S := (arc_BE - AC) / 2,
  let θ_R := AC / 2 in
  (θ_S + θ_R) = 41 :=
by
  sorry

end sum_of_angles_S_R_l37_37640


namespace unique_root_of_increasing_l37_37120

variable {R : Type} [LinearOrderedField R] [DecidableEq R]

def increasing (f : R → R) : Prop :=
  ∀ x1 x2 : R, x1 < x2 → f x1 < f x2

theorem unique_root_of_increasing (f : R → R)
  (h_inc : increasing f) :
  ∃! x : R, f x = 0 :=
sorry

end unique_root_of_increasing_l37_37120


namespace cone_volume_proof_l37_37455

noncomputable def cone_volume (a α β : ℝ) : ℝ :=
  (Math.pi * a^3 * Real.cot β) / (24 * (Real.sin (α / 2))^3)

theorem cone_volume_proof (a α β : ℝ) :
  ∀ (V : ℝ), V = cone_volume a α β :=
by
  intro V
  rw [cone_volume]
  sorry

end cone_volume_proof_l37_37455


namespace symmetric_line_circle_O2_equation_l37_37919

open Real

noncomputable def circle1_center : point := (0, -1)
noncomputable def point_A : point := (1, -2)
noncomputable def line_symmetric := λ P Q l: String, (P ≠ Q) ∧ (l contains point_A)

def equation_of_l (center_P Q : point) : String :=
  "x + y + 1 = 0"

def symmetric_point_O2 (sym_line : String) : point :=
  (2, 1)

def equation_of_circle_O2 (center_O2 : point) : String :=
  "((x - 2)² + (y + 1)² = 20) ∨ ((x - 2)² + (y + 1)² = 4)"

theorem symmetric_line (P Q : point) (l : String) :
  line_symmetric P Q l → equation_of_l circle1_center point_A = "x + y + 1 = 0" :=
sorry

theorem circle_O2_equation (O1 O2 : String) :
  circle1_center = (0, -1) → point_A = (1, -2) →
  symmetric_point_O2 "x + 3y = 0" = (2, 1) →
  equation_of_circle_O2 (2, 1) = "((x - 2)² + (y + 1)² = 20) ∨ ((x - 2)² + (y + 1)² = 4)" :=
sorry

end symmetric_line_circle_O2_equation_l37_37919


namespace zoo_escaped_lions_l37_37254

theorem zoo_escaped_lions (total_time : ℕ) (time_per_animal : ℕ) (rhino_count : ℕ) :
  total_time = 10 → time_per_animal = 2 → rhino_count = 2 → 
  (total_time - rhino_count * time_per_animal) / time_per_animal = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end zoo_escaped_lions_l37_37254


namespace z_conj_in_second_quadrant_l37_37519

noncomputable def z : ℂ := 2 / (complex.I - 1)
noncomputable def z_conj : ℂ := complex.conj z

theorem z_conj_in_second_quadrant (hz : z = 2 / (complex.I - 1)) :
  z_conj.re < 0 ∧ z_conj.im > 0 :=
by
  -- Proof is not required
  sorry

end z_conj_in_second_quadrant_l37_37519


namespace coinsSold_l37_37834

-- Given conditions
def initialCoins : Nat := 250
def additionalCoins : Nat := 75
def coinsToKeep : Nat := 135

-- Theorem to prove
theorem coinsSold : (initialCoins + additionalCoins - coinsToKeep) = 190 := 
by
  -- Proof omitted 
  sorry

end coinsSold_l37_37834


namespace abs_f_x_minus_f_a_lt_l37_37626

variable {R : Type*} [LinearOrderedField R]

def f (x : R) (c : R) := x ^ 2 - x + c

theorem abs_f_x_minus_f_a_lt (x a c : R) (h : abs (x - a) < 1) : 
  abs (f x c - f a c) < 2 * (abs a + 1) :=
by
  sorry

end abs_f_x_minus_f_a_lt_l37_37626


namespace three_digit_numbers_with_repeated_digits_l37_37713

theorem three_digit_numbers_with_repeated_digits :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in 
  let total_three_digit_numbers := 900 in
  let three_digit_numbers_without_repeated_digits := 9 * 9 * 8 in
  total_three_digit_numbers - three_digit_numbers_without_repeated_digits = 252 := 
by 
  sorry

end three_digit_numbers_with_repeated_digits_l37_37713


namespace part1_part2_l37_37176

noncomputable def ellipse_equation : ℝ × ℝ → Prop :=
  λ (x y : ℝ), (x^2 / 4) + y^2 = 1

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c = sqrt 3) (area : b * sqrt 3 = sqrt 3) :
  a^2 = 4 ∧ b = 1 :=
sorry

theorem part2 (l : ℝ → ℝ → Prop) (exist_Q : ∃ Qx Qy, ellipse_equation Qx Qy ∧ (Qx, Qy) = (8 * sqrt 3 / (4 + 8), -(2 * sqrt 3 * (2 * sqrt 2) / (4 + 8)))) :
  ∃ m : ℝ, (m = 2 * sqrt 2 ∨ m = -2 * sqrt 2) ∧ (∀ x y, l x y ↔ x = 2 * sqrt 2 * y + sqrt 3 ∨ x = -2 * sqrt 2 * y + sqrt 3) :=
sorry

end part1_part2_l37_37176


namespace diane_stamp_combinations_l37_37043

theorem diane_stamp_combinations :
  let stamps := [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
  in stamps.count_combinations(15) = 168 :=
sorry

end diane_stamp_combinations_l37_37043


namespace theorem1_theorem2_theorem3_l37_37219

-- Given conditions as definitions
variables {x y p q : ℝ}

-- Condition definitions
def condition1 : x + y = -p := sorry
def condition2 : x * y = q := sorry

-- Theorems to be proved
theorem theorem1 (h1 : x + y = -p) (h2 : x * y = q) : x^2 + y^2 = p^2 - 2 * q := sorry

theorem theorem2 (h1 : x + y = -p) (h2 : x * y = q) : x^3 + y^3 = -p^3 + 3 * p * q := sorry

theorem theorem3 (h1 : x + y = -p) (h2 : x * y = q) : x^4 + y^4 = p^4 - 4 * p^2 * q + 2 * q^2 := sorry

end theorem1_theorem2_theorem3_l37_37219


namespace diploma_count_l37_37994

variable (students : Fin 13)

noncomputable def received_diploma (s : Fin 13) : Prop := sorry

theorem diploma_count :
  (∑ s, if received_diploma s then 1 else 0) = 2 :=
sorry

end diploma_count_l37_37994


namespace find_d_h_l37_37289

theorem find_d_h (a b c d g h : ℂ) (h1 : b = 4) (h2 : g = -a - c) (h3 : a + c + g = 0) (h4 : b + d + h = 3) : 
  d + h = -1 := 
by
  sorry

end find_d_h_l37_37289


namespace john_uniform_number_13_l37_37072

open Nat

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Prime n

theorem john_uniform_number_13 :
  ∃ t m j d,
    is_two_digit_prime t ∧
    is_two_digit_prime m ∧
    is_two_digit_prime j ∧
    is_two_digit_prime d ∧
    t + j = 26 ∧
    m + d = 32 ∧
    j + d = 34 ∧
    t + d = 36 ∧
    j = 13 :=
by
  sorry

end john_uniform_number_13_l37_37072


namespace M_is_subset_of_N_l37_37959

theorem M_is_subset_of_N : 
  ∀ (x y : ℝ), (|x| + |y| < 1) → 
    (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by
  intro x y h
  sorry

end M_is_subset_of_N_l37_37959


namespace number_of_ways_to_draw_4_from_15_l37_37753

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37753


namespace finite_completely_symmetric_set_classification_l37_37001

-- Define the conditions for a completely symmetric set
structure CompletelySymmetricSet (S : Set Point) : Prop :=
  (card_gt_two : 2 < S.card)
  (symmetry_condition : ∀ ⦃A B : Point⦄, A ∈ S → B ∈ S → A ≠ B → 
                      SymmetryPlane (PerpendicularBisector A B) S)

-- Main proof statement
theorem finite_completely_symmetric_set_classification 
  (S : Set Point) (h_finite : S.Finite) (h_sym : CompletelySymmetricSet S) :
  ∃ (n : ℕ), (n = 3 ∧ ∃ t : RegularPolygon n, S = t.vertices) ∨
  (∃ t : RegularTetrahedron, S = t.vertices) ∨
  (∃ t : RegularOctahedron, S = t.vertices) :=
sorry

end finite_completely_symmetric_set_classification_l37_37001


namespace smallest_n_with_triplet_l37_37450

theorem smallest_n_with_triplet :
  ∃ n : ℕ, (∀ (A B : set ℕ), A ∪ B = {i | i ≤ n} → A ∩ B = ∅ →
  (∃ a b c ∈ A, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)
  ∨ (∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)) ∧ n = 96 :=
sorry

end smallest_n_with_triplet_l37_37450


namespace product_terms_l37_37579

variable (a_n : ℕ → ℝ)
variable (r : ℝ)

-- a1 = 1 and a10 = 3
axiom geom_seq  (h : ∀ n, a_n (n + 1) = r * a_n n) : a_n 1 = 1 → a_n 10 = 3

theorem product_terms :
  (∀ n, a_n (n + 1) = r * a_n n) → a_n 1 = 1 → a_n 10 = 3 → 
  a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 = 81 :=
by
  intros h1 h2 h3
  sorry

end product_terms_l37_37579


namespace find_number_l37_37803

theorem find_number {x : ℤ} (h : x + 5 = 6) : x = 1 :=
sorry

end find_number_l37_37803


namespace remaining_pencils_check_l37_37192

variables (Jeff_initial : ℕ) (Jeff_donation_percentage : ℚ) (Vicki_ratio : ℚ) (Vicki_donation_fraction : ℚ)

def Jeff_donated_pencils := (Jeff_donation_percentage * Jeff_initial).toNat
def Jeff_remaining_pencils := Jeff_initial - Jeff_donated_pencils

def Vicki_initial_pencils := (Vicki_ratio * Jeff_initial).toNat
def Vicki_donated_pencils := (Vicki_donation_fraction * Vicki_initial_pencils).toNat
def Vicki_remaining_pencils := Vicki_initial_pencils - Vicki_donated_pencils

def total_remaining_pencils := Jeff_remaining_pencils + Vicki_remaining_pencils

theorem remaining_pencils_check
    (Jeff_initial : ℕ := 300)
    (Jeff_donation_percentage : ℚ := 0.3)
    (Vicki_ratio : ℚ := 2)
    (Vicki_donation_fraction : ℚ := 0.75) :
    total_remaining_pencils Jeff_initial Jeff_donation_percentage Vicki_ratio Vicki_donation_fraction = 360 :=
by
  sorry

end remaining_pencils_check_l37_37192


namespace possible_cycle_1990_committees_11_countries_l37_37485

theorem possible_cycle_1990_committees_11_countries :
  let n : ℕ := 11,
  let m : ℕ := 1990,
  exists_cycle (n m) := 
  ∀ (A : ℕ → fin n → fin 3), 
  (∀ i : fin m+1, set.univ (fin n) = {j : fin n | A i j} ∧ 
    (∀ i1 i2, i1 ≠ i2 → {j : fin n | A i1 j} ≠ {j : fin n | A i2 j}) ∧
    (∀ i, disjoint {j | A i j} {j | A (i + 1) % m j}) ∧
    (∀ i j m k, 1 < |i - j| < m - 1 → {j | A i j} ∩ {j | A j j} ≠ ∅)) :=
by
  let n := 11;
  let m := 1990;
  exists_cycle (n m) sorry

end possible_cycle_1990_committees_11_countries_l37_37485


namespace parabola_directrix_l37_37880

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37880


namespace diff_mean_median_l37_37231

-- Define the percentage of students scoring specific points
def students_scoring (s : Nat) : ℝ :=
  if s = 60 then 0.15 * 100 else
  if s = 75 then 0.20 * 100 else
  if s = 85 then 0.25 * 100 else
  if s = 92 then 0.10 * 100 else
  if s = 100 then 0.30 * 100 else 0

-- Calculate the median score
def median_score : ℝ :=
  85

-- Calculate the mean score
def mean_score : ℝ :=
  (60 * 0.15 + 75 * 0.20 + 85 * 0.25 + 92 * 0.10 + 100 * 0.30) * 100 / 100

-- Define the proof statement
theorem diff_mean_median : (median_score - mean_score) = 0.55 := by
  sorry

end diff_mean_median_l37_37231


namespace packet_weight_l37_37232

theorem packet_weight
  (tons_to_pounds : ℕ := 2600) -- 1 ton = 2600 pounds
  (total_tons : ℕ := 13)       -- Total capacity in tons
  (num_packets : ℕ := 2080)    -- Number of packets
  (expected_weight_per_packet : ℚ := 16.25) : 
  total_tons * tons_to_pounds / num_packets = expected_weight_per_packet := 
sorry

end packet_weight_l37_37232


namespace parabola_ab_length_l37_37845

noncomputable def parabola_length (p : ℝ) (xA xB : ℝ) : ℝ :=
  let mf : ℝ := (-9 : ℝ)
  in let midpoint_condition : Prop := ((xA + xB) / 2 = mf)
  in let sum_x_coords := (-18 : ℝ)
  in let derived_midpoint := (xA + xB = sum_x_coords)
  in xA + xB + p

theorem parabola_ab_length : parabola_length 6 (-27) 9 = 24 := by
  sorry

end parabola_ab_length_l37_37845


namespace gen_term_seq_l37_37922

open Nat

def seq (a : ℕ → ℕ) : Prop := 
a 1 = 1 ∧ (∀ n : ℕ, n ≠ 0 → a (n + 1) = 2 * a n - 3)

theorem gen_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end gen_term_seq_l37_37922


namespace similarity_of_triangles_l37_37912

/-
  Statement of the problem:
  Given an acute-angled triangle ABC with altitude CH from vertex C to side AB.
  From point H, perpendiculars HM and HN are dropped onto sides BC and AC respectively.
  Prove that triangles MNC and ABC are similar.
-/

open EuclideanGeometry

variables {A B C M N H : Point}
variables {CH HM HN : Line}
variables {triangle_ABC : Triangle}
variables {acute_triangle : isAcuteTriangle triangle_ABC}
variables {altitude_CH : isAltitude C H CH AB}
variables {perpendicular_HM : isPerpendicular HM BC}
variables {perpendicular_HN : isPerpendicular HN AC}

theorem similarity_of_triangles :
  similarity (triangle.mk M N C) (triangle.mk A B C) :=
  sorry

end similarity_of_triangles_l37_37912


namespace robie_initial_cards_l37_37241

-- Definitions of the problem conditions
def each_box_cards : ℕ := 25
def extra_cards : ℕ := 11
def given_away_boxes : ℕ := 6
def remaining_boxes : ℕ := 12

-- The final theorem we need to prove
theorem robie_initial_cards : 
  (given_away_boxes + remaining_boxes) * each_box_cards + extra_cards = 461 :=
by
  sorry

end robie_initial_cards_l37_37241


namespace length_of_bridge_l37_37802

theorem length_of_bridge
    (speed_kmh : Real)
    (time_minutes : Real)
    (speed_cond : speed_kmh = 5)
    (time_cond : time_minutes = 15) :
    let speed_mmin := speed_kmh * 1000 / 60
    let distance_m := speed_mmin * time_minutes
    distance_m = 1250 :=
by
    sorry

end length_of_bridge_l37_37802


namespace base_of_parallelogram_l37_37438

-- Definitions according to the conditions in the problem
def area : ℝ := 231
def height : ℝ := 11

-- Definition of the base we need to prove
def base : ℝ := area / height

-- Theorem we need to prove
theorem base_of_parallelogram : base = 21 := 
by 
  -- This is where the actual proof would go
  sorry

end base_of_parallelogram_l37_37438


namespace sum_of_odd_divisors_of_450_l37_37718

theorem sum_of_odd_divisors_of_450 : ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 450)), d = 403 :=
by {
  sorry
}

end sum_of_odd_divisors_of_450_l37_37718


namespace second_storm_duration_l37_37301

theorem second_storm_duration (x y : ℕ) 
  (h1 : x + y = 45) 
  (h2 : 30 * x + 15 * y = 975) : 
  y = 25 :=
by
  sorry

end second_storm_duration_l37_37301


namespace large_number_of_divisors_l37_37217

theorem large_number_of_divisors (n : ℕ) (h : n ≥ 1) (p : Fin n → ℕ) 
  (ppos : ∀ i, p i ≥ 5) (pprime : ∀ i, Prime (p i)) :
  ∃ k, k ≥ 2^(2^n) ∧ (2^(∏ i, p i) + 1) ∣ k :=
sorry

end large_number_of_divisors_l37_37217


namespace algebraic_expression_value_l37_37916

noncomputable def a : ℝ := 1 + Real.sqrt 2
noncomputable def b : ℝ := 1 - Real.sqrt 2

theorem algebraic_expression_value :
  let a := 1 + Real.sqrt 2
  let b := 1 - Real.sqrt 2
  a^2 - a * b + b^2 = 7 := by
  sorry

end algebraic_expression_value_l37_37916


namespace find_f_three_halves_l37_37522

noncomputable def f : ℝ → ℝ 
| x => if 0 < x ∧ x < 1 then 
          1 - 4 * x
       else if x ≥ 1 then 
          2 * f (x / 2)
       else 
          0 -- Since we do not have the definition for x ≤ 0 

theorem find_f_three_halves : f (3 / 2) = -4 :=
by
  sorry

end find_f_three_halves_l37_37522


namespace sum_of_odd_divisors_of_450_l37_37717

theorem sum_of_odd_divisors_of_450 : ∑ d in (finset.filter (λ x, x % 2 = 1) (finset.divisors 450)), d = 403 :=
by {
  sorry
}

end sum_of_odd_divisors_of_450_l37_37717


namespace number_of_valid_six_digit_numbers_l37_37151

theorem number_of_valid_six_digit_numbers : 
  let total := 2^6 in
  let excluded := 2 in
  let valid_numbers := total - excluded in
  valid_numbers = 62 :=
by
  let total := 64
  let excluded := 2
  let valid_numbers := total - excluded
  have h : valid_numbers = 62 := by
    calc
      valid_numbers = 64 - 2 : by rfl
                   ... = 62   : by rfl
  exact h

end number_of_valid_six_digit_numbers_l37_37151


namespace stella_monthly_income_proof_l37_37256

variable (annual_income : ℕ) (months_worked : ℕ)

def stella_monthly_income (annual_income : ℕ) (months_worked : ℕ) : ℕ :=
  annual_income / months_worked

theorem stella_monthly_income_proof : 
  stella_monthly_income 49190 10 = 4919 :=
by
  unfold stella_monthly_income
  simp
  sorry

end stella_monthly_income_proof_l37_37256


namespace minimum_quotient_of_digits_l37_37679

-- Definitions
def is_non_zero_digit (x : ℕ) : Prop := x > 0 ∧ x < 10
def all_different (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Statement of the problem
theorem minimum_quotient_of_digits : 
  ∀ (a b c : ℕ), 
  is_non_zero_digit a → is_non_zero_digit b → is_non_zero_digit c →
  all_different a b c →
  (∀ x y z, is_non_zero_digit x → is_non_zero_digit y → is_non_zero_digit z → all_different x y z → 
            (100 * x + 10 * y + z) / (x + y + z) ≥ 10.5) :=
  sorry

end minimum_quotient_of_digits_l37_37679


namespace square_of_any_real_number_not_always_greater_than_zero_l37_37808

theorem square_of_any_real_number_not_always_greater_than_zero (a : ℝ) : 
    (∀ x : ℝ, x^2 ≥ 0) ∧ (exists x : ℝ, x = 0 ∧ x^2 = 0) :=
by {
  sorry
}

end square_of_any_real_number_not_always_greater_than_zero_l37_37808


namespace initial_investment_calculation_l37_37398

-- Define the conditions
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 2
def A : ℝ := 6050.000000000001
def one : ℝ := 1

-- The goal is to prove that the initial principal P is 5000 under these conditions
theorem initial_investment_calculation (P : ℝ) : P = 5000 :=
by
  have interest_compounded : ℝ := (one + r / n) ^ (n * t)
  have total_amount : ℝ := P * interest_compounded
  sorry

end initial_investment_calculation_l37_37398


namespace sum_of_solutions_l37_37066

def f (x : ℝ) : ℝ := 2^(|x|) + 4 * |x|

theorem sum_of_solutions : (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0 ∧ x₁ ≠ x₂) →
  (∑ x in {solution : ℝ | f solution = 20}.to_finset, x = 0) :=
by
  sorry

end sum_of_solutions_l37_37066


namespace modulus_of_exp_pi_div_3_i_l37_37856

def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem modulus_of_exp_pi_div_3_i :
  complex.abs (euler_formula (real.pi / 3)) = 1 := by
-- sorry is used as a placeholder for the actual proof
sorry

end modulus_of_exp_pi_div_3_i_l37_37856


namespace carousel_rotation_time_l37_37800

-- Definitions and Conditions
variables (a v U x : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (U * a - v * a = 2 * Real.pi)
def condition2 : Prop := (v * a = U * (x - a / 2))

-- Statement to prove
theorem carousel_rotation_time :
  condition1 a v U ∧ condition2 a v U x → x = 2 * a / 3 :=
by
  intro h
  have c1 := h.1
  have c2 := h.2
  sorry

end carousel_rotation_time_l37_37800


namespace first_number_in_each_fraction_is_one_l37_37698

theorem first_number_in_each_fraction_is_one :
  let sum := (1 / (1 * 2.3) + 1 / (2 * 3.4) + 1 / (3 * 4.5) + 1 / (4 * 5.6))
  in sum = 0.23333333333333334 →
  ∀ n : ℕ, n ∈ {1, 2, 3, 4} → (1 / (n * (n.succ.succ.toReal))) = (1 / (n * some_const)) := 
by
  sorry

end first_number_in_each_fraction_is_one_l37_37698


namespace odds_against_C_undefined_l37_37996

-- Defining the probabilities based on the conditions from the problem
def prob_A : ℝ := 1 / 3
def prob_B : ℝ := 2 / 3
def prob_C : ℝ := 1 - prob_A - prob_B

-- Theorem stating that the odds against horse C winning are undefined
theorem odds_against_C_undefined
  (h1 : prob_A = 1 / 3)
  (h2 : prob_B = 2 / 3)
  (h3 : ¬(ties_possible : Bool := false))
  : prob_C = 0 → false := sorry

end odds_against_C_undefined_l37_37996


namespace polynomial_solution_l37_37608

noncomputable def P (x : ℝ) : ℝ := x^2 - x - 1

theorem polynomial_solution (x : ℝ) : 
  (∀ x, P x = P 0 + P 1 * x + P 2 * x^2) ∧ (P (-1) = 1) → 
  P(x) = x^2 - x - 1 := 
by
  sorry

end polynomial_solution_l37_37608


namespace equal_contribution_expense_split_l37_37908

theorem equal_contribution_expense_split (Mitch_expense Jam_expense Jay_expense Jordan_expense total_expense each_contribution : ℕ)
  (hmitch : Mitch_expense = 4 * 7)
  (hjam : Jam_expense = (2 * 15) / 10 + 4) -- note: 1.5 dollar per box interpreted as 15/10 to avoid float in Lean
  (hjay : Jay_expense = 3 * 3)
  (hjordan : Jordan_expense = 4 * 2)
  (htotal : total_expense = Mitch_expense + Jam_expense + Jay_expense + Jordan_expense)
  (hequal_split : each_contribution = total_expense / 4) :
  each_contribution = 13 :=
by
  sorry

end equal_contribution_expense_split_l37_37908


namespace longest_segment_in_cylinder_l37_37355

theorem longest_segment_in_cylinder 
  (r h : ℝ) 
  (r_eq : r = 5) 
  (h_eq : h = 12) : 
  ∃ l, l = Real.sqrt ((2 * r)^2 + h^2) ∧ l = Real.sqrt 244 :=
by
  use Real.sqrt ((2 * r)^2 + h^2)
  split
  . rfl
  sorry

end longest_segment_in_cylinder_l37_37355


namespace remaining_pencils_l37_37197

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l37_37197


namespace smallest_positive_period_of_f_max_min_values_of_f_on_interval_l37_37126

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + π) = f x :=
by sorry

theorem max_min_values_of_f_on_interval : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧
  f x₁ = 0 ∧ f x₂ = 1 + Real.sqrt 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_on_interval_l37_37126


namespace major_premise_wrong_l37_37807

-- Definitions of the given conditions in Lean
def is_parallel_to_plane (line : Type) (plane : Type) : Prop := sorry -- Provide an appropriate definition
def contains_line (plane : Type) (line : Type) : Prop := sorry -- Provide an appropriate definition
def is_parallel_to_line (line1 : Type) (line2 : Type) : Prop := sorry -- Provide an appropriate definition

-- Given conditions
variables (b α a : Type)
variable (H1 : ¬ contains_line α b)  -- Line b is not contained in plane α
variable (H2 : contains_line α a)    -- Line a is contained in plane α
variable (H3 : is_parallel_to_plane b α) -- Line b is parallel to plane α

-- Proposition to prove: The major premise is wrong
theorem major_premise_wrong : ¬(∀ (a b : Type), is_parallel_to_plane b α → contains_line α a → is_parallel_to_line b a) :=
by
  sorry

end major_premise_wrong_l37_37807


namespace volume_proof_l37_37456

-- Define the required conditions for the variables
def region (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 120 * fract x ≥ floor x + floor y ∧ z ≤ y

-- Define the volume calculation result
def volume_result : ℝ := 9680.20

-- State the theorem that given the conditions, the volume is 9680.20
theorem volume_proof :
  (∫ x in 0..1, ∫ y in 0..1, ∫ z in 0..1, indicator (region x y z)) = volume_result :=
by
  sorry

end volume_proof_l37_37456


namespace problem_1_problem_2_l37_37133

-- Problem 1: Range of the function f(x) = x^2 - 2x + 1 on [0, 3]
theorem problem_1 :
  (∀ x ∈ set.Icc 0 3, (x^2 - 2 * x + 1) ∈ set.Icc 0 4) :=
by
  sorry

-- Problem 2: Existence and Uniqueness of a such that the domain of f(x) = x^2 - 2ax + a is [-1, 1] and the range is [-2, 2]
theorem problem_2 :
  ∃! a : ℝ, (∀ x ∈ set.Icc (-1 : ℝ) 1, (x^2 - 2 * a * x + a) ∈ set.Icc (-2 : ℝ) 2) :=
by
  sorry

end problem_1_problem_2_l37_37133


namespace student_average_always_greater_l37_37383

theorem student_average_always_greater (x y z : ℝ) (h1 : x < z) (h2 : z < y) :
  (B = (x + z + 2 * y) / 4) > (A = (x + y + z) / 3) := by
  sorry

end student_average_always_greater_l37_37383


namespace bill_health_insurance_cost_l37_37412

noncomputable def calculate_health_insurance_cost : ℕ := 3000

theorem bill_health_insurance_cost
  (normal_monthly_price : ℕ := 500)
  (gov_pay_less_than_10000 : ℕ := 90) -- 90%
  (gov_pay_between_10001_and_40000 : ℕ := 50) -- 50%
  (gov_pay_more_than_50000 : ℕ := 20) -- 20%
  (hourly_wage : ℕ := 25)
  (weekly_hours : ℕ := 30)
  (weeks_per_month : ℕ := 4)
  (months_per_year : ℕ := 12)
  (income_between_10001_and_40000 : Prop := (hourly_wage * weekly_hours * weeks_per_month * months_per_year) >= 10001 ∧ (hourly_wage * weekly_hours * weeks_per_month * months_per_year) <= 40000):
  (calculate_health_insurance_cost = 3000) :=
by
sry


end bill_health_insurance_cost_l37_37412


namespace maximum_area_triangle_l37_37638

-- Definitions based on conditions
def a : ℝ := 75
def b (x : ℝ) : ℝ := x
def c (x : ℝ) : ℝ := 2 * x

-- Semi-perimeter function
def s (x : ℝ) : ℝ := (a + b x + c x) / 2

-- Area formula using Heron's formula
def area (x : ℝ) : ℝ := 
  Real.sqrt (s x * (s x - a) * (s x - b x) * (s x - c x)) / 4

-- The proposition we want to prove
theorem maximum_area_triangle : ∃ x : ℝ, x > 0 ∧ area x = 1125 := 
by
  sorry

end maximum_area_triangle_l37_37638


namespace gcd_g_150_151_l37_37614

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 :=
  by
  sorry

end gcd_g_150_151_l37_37614


namespace count_packages_needing_extra_postage_l37_37797

structure Package where
  length : ℕ
  width : ℕ
  weight : ℕ

def needs_extra_postage (p : Package) : Prop :=
  (p.length / p.width < 3 ∧ p.length / p.width > 1.5) ∨ (p.weight > 5)

def Package_X : Package := { length := 8, width := 6, weight := 4 }
def Package_Y : Package := { length := 12, width := 4, weight := 6 }
def Package_Z : Package := { length := 7, width := 7, weight := 5 }
def Package_W : Package := { length := 14, width := 4, weight := 3 }

theorem count_packages_needing_extra_postage :
  [Package_X, Package_Y, Package_Z, Package_W].count needs_extra_postage = 4 := by
  sorry

end count_packages_needing_extra_postage_l37_37797


namespace find_time_when_velocity_is_zero_l37_37953

noncomputable def motion_law (t : ℝ) : ℝ := t^3 - 6 * t^2 + 5

theorem find_time_when_velocity_is_zero : ∃ t > 0, derivative (motion_law t) = 0 ∧ t = 4 := 
by 
  sorry

end find_time_when_velocity_is_zero_l37_37953


namespace arithmetic_sequence_30th_term_l37_37313

theorem arithmetic_sequence_30th_term :
  let a := 3
  let d := 7 - 3
  ∀ n, (n = 30) → (a + (n - 1) * d) = 119 := by
  sorry

end arithmetic_sequence_30th_term_l37_37313


namespace ellipse_equation_line_passes_fixed_point_l37_37092

variables {k : ℝ} (a b : ℝ) (x y : ℝ) (A B P Q : ℝ × ℝ)
hypothesis minor_axis_length : b = sqrt 3
hypothesis point_on_ellipse : ∃ a : ℝ, a > b ∧ (1^2 / a^2) + ((-3/2)^2 / b^2) = 1
hypothesis vertices_of_ellipse : A = (-2, 0) ∧ B = (2, 0)
hypothesis distinct_points_on_ellipse : P ≠ A ∧ P ≠ B ∧ Q ≠ A ∧ Q ≠ B ∧ P ≠ Q
hypothesis slopes : ∃ k : ℝ, (P.2 - B.2) / (P.1 - B.1) = k ∧ (Q.2 - A.2) / (Q.1 - A.1) = 2 * k

theorem ellipse_equation :
  ∃ a : ℝ, a = 2 ∧ b = sqrt 3 ∧ ∀ x y : ℝ, (x^2 / (2^2)) + (y^2 / (sqrt 3 ^ 2)) = 1 :=
begin
  sorry
end

theorem line_passes_fixed_point :
  ∀ P Q : ℝ × ℝ, distinct_points_on_ellipse → slopes →
  ∃ M : ℝ × ℝ, M = (-2/3, 0) ∧ 
  ((P.2 - M.2) / (P.1 - M.1)) = ((Q.2 - M.2) / (Q.1 - M.1)) :=
begin
  sorry
end

end ellipse_equation_line_passes_fixed_point_l37_37092


namespace drama_club_students_neither_math_nor_physics_l37_37634

theorem drama_club_students_neither_math_nor_physics
  (total_students : ℕ)
  (math_students : ℕ)
  (physics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 50)
  (h2 : math_students = 36)
  (h3 : physics_students = 27)
  (h4 : both_students = 20) :
  ∃ n : ℕ, n = total_students - (math_students - both_students + physics_students - both_students + both_students) ∧ n = 7 :=
by
  use total_students - (math_students - both_students + physics_students - both_students + both_students)
  split
  sorry
  sorry

end drama_club_students_neither_math_nor_physics_l37_37634


namespace sequence_solution_exists_l37_37618

noncomputable def math_problem (a : ℕ → ℝ) : Prop :=
  ∀ n < 1990, a n > 0 ∧ a 1990 < 0

theorem sequence_solution_exists {a0 c : ℝ} (h_a0 : a0 > 0) (h_c : c > 0) :
  ∃ (a : ℕ → ℝ),
    a 0 = a0 ∧
    (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
    math_problem a :=
by
  sorry

end sequence_solution_exists_l37_37618


namespace no_grasshopper_lands_on_fourth_vertex_l37_37706

-- Define the initial positions of the grasshoppers
def initial_positions : set (ℤ × ℤ) :=
  {(0, 0), (1, 0), (0, 1)}

-- Define the leapfrog move
def leapfrog_move (A B : ℤ × ℤ) : ℤ × ℤ :=
  (2 * B.1 - A.1, 2 * B.2 - A.2)

-- Define the main theorem
theorem no_grasshopper_lands_on_fourth_vertex :
  ∀ (positions : set (ℤ × ℤ)),
  positions = initial_positions ∨
  (∃ A B ∈ positions, positions' = positions ∪ {leapfrog_move A B}) →
  ¬ ∃ pos ∈ positions, pos = (1, 1) :=
sorry

end no_grasshopper_lands_on_fourth_vertex_l37_37706


namespace product_of_common_roots_l37_37669

theorem product_of_common_roots (p q r t : ℝ) (C D : ℝ) :
  (p, q, r are roots of x^3 + 3x^2 + Cx + 15 = 0) →
  (p, q, t are roots of x^3 + Dx^2 + 70 = 0) →
  ∃ (a b c : ℕ), (pq = a * (c ^ (1 / b : ℝ))) ∧ (a + b + c = 20) :=
by
  sorry

end product_of_common_roots_l37_37669


namespace trapezoid_shaded_area_fraction_l37_37388

-- Define a structure for the trapezoid
structure Trapezoid (A : Type) :=
(strips : list A)
(equal_width : ∀ i j, i ≠ j ∧ i ∈ strips ∧ j ∈ strips → width i = width j)
(shaded_strips : list A)
(shaded : ∀ s, s ∈ shaded_strips ↔ s ∈ strips ∧ is_shaded s)

-- Define the predicate to check if a strip is shaded
def is_shaded (s : Strip) : Prop := s ∈ shaded_strips

-- Define the problem as a theorem in Lean
theorem trapezoid_shaded_area_fraction
  (T : Trapezoid Strip)
  (h_strips : length T.strips = 7)
  (h_shaded : length T.shaded_strips = 4)
  : fraction_shaded_area T = 4 / 7 :=
begin
  sorry
end

end trapezoid_shaded_area_fraction_l37_37388


namespace smallest_positive_period_interval_of_monotonic_decrease_range_of_sum_b_c_l37_37132

open Real

/-- Part 1:
    Prove that the smallest positive period of 
    f(x) = sqrt(3) * sin(x) ^ 2 + sin(x) * cos(x) - sqrt(3) / 2 
    is pi.
-/
theorem smallest_positive_period : 
  ∀ x, sqrt 3 * (sin x) ^ 2 + sin x * cos x - (sqrt 3) / 2 = f (x + π) := sorry

/-- Part 2:
    Prove that the interval where 
    f(x) = sqrt(3) * sin(x) ^ 2 + sin(x) * cos(x) - sqrt(3) / 2 
    is monotonically decreasing is 
    [5π/12 + kπ, 11π/12 + kπ], k ∈ ℤ.
-/
theorem interval_of_monotonic_decrease : 
  ∀ k : ℤ, ∀ x, (x ∈ Icc (5 * π / 12 + k * π) (11 * π / 12 + k * π)) → 
  deriv (λ x, sqrt 3 * sin x ^ 2 + sin x * cos x - sqrt 3 / 2) x < 0 := sorry

/-- Part 3:
    Given f(A/2 + π/4) = 1 and a = 2 in triangle ∆ABC,
    prove that the range of values for b + c is (2, 4].
-/
theorem range_of_sum_b_c (A B C a b c : ℝ) :
  A = π / 3 → f (A / 2 + π / 4) = 1 → a = 2 →  
  ∃ s : Set ℝ, s = Set.Ioc 2 4 ∧ b + c ∈ s := sorry

end smallest_positive_period_interval_of_monotonic_decrease_range_of_sum_b_c_l37_37132


namespace expected_rolls_for_2010_l37_37356

noncomputable def expected_rolls_to_reach (goal : ℕ) : ℝ := 
  sorry

theorem expected_rolls_for_2010 :
  expected_rolls_to_reach 2010 ≈ 574.761904 :=
sorry

end expected_rolls_for_2010_l37_37356


namespace second_candidate_votes_l37_37569

theorem second_candidate_votes
  (total_votes : ℕ)
  (first_candidate_percentage : ℕ)
  (second_candidate_percentage : ℕ)
  (first_candidate_votes : ℕ)
  : total_votes = 800 ∧ first_candidate_percentage = 70 ∧ second_candidate_percentage = 30 ∧ first_candidate_votes = 560
  → second_candidate_votes = 240 :=
by
  intro h,
  cases h with h_total h,
  cases h with h_first_pct h,
  cases h with h_second_pct h_first_votes,
  have h_first : first_candidate_votes = (first_candidate_percentage * total_votes) / 100 := by sorry,
  rw [h_first_pct, h_total] at h_first,
  norm_num at h_first,
  have h_second : second_candidate_votes = (second_candidate_percentage * total_votes) / 100 := by sorry,
  rw [h_second_pct, h_total],
  norm_num,
  exact h_second

end second_candidate_votes_l37_37569


namespace permits_stamped_l37_37400

def appointments : ℕ := 2
def hours_per_appointment : ℕ := 3
def workday_hours : ℕ := 8
def stamps_per_hour : ℕ := 50

theorem permits_stamped :
  let total_appointment_hours := appointments * hours_per_appointment in
  let stamping_hours := workday_hours - total_appointment_hours in
  let total_permits := stamping_hours * stamps_per_hour in
  total_permits = 100 :=
by
  sorry

end permits_stamped_l37_37400


namespace max_subsets_of_five_elements_S_l37_37528

open Finset

noncomputable def max_5element_subsets (S : Finset ℕ) (n : ℕ) : ℕ :=
  if h : S.card = n then
    classical.some (exists_maximum (λ (k : ℕ), ∀ (A : Finset (Finset S)) 
      (h₁ : ∀ T ∈ A, T.card = 5) 
      (h₂ : ∀ (x y : ℕ), {x, y} ⊆ S → ∃ U V ∈ A, {U, V}.card ≤ 2),
        A.card ≤ k))
  else 0

theorem max_subsets_of_five_elements_S :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  max_5element_subsets S 10 = 8 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  drop sorry

end max_subsets_of_five_elements_S_l37_37528


namespace radical_axis_of_circum_and_nine_point_circle_l37_37846

variables {A B C D E F O N : Point}
variables {Ω : Circle}
variables {nine_point_circle : Circle}

-- Definitions of the conditions
def triangle_ABC (A B C : Point) : Triangle := Triangle.mk A B C
def circumcircle (t : Triangle) : Circle := Circle.circumcircle t
def nine_point_circle (t : Triangle) : Circle := Circle.ninePointCircle t

-- Question (goal) definition
def radical_axis (Ω : Circle) (nine_point_circle : Circle) : Line :=
  Line.radicalAxis Ω nine_point_circle

-- Problem
theorem radical_axis_of_circum_and_nine_point_circle (A B C D E F O N : Point)
  (h_triangle : Triangle A B C)
  (h_circumcircle : circumcircle h_triangle = Ω)
  (h_nine_point : nine_point_circle h_triangle = nine_point_circle)
  (h_feet_altitudes : AltitudeFeet A B C D E F)
  (h_circumcenter : Circumcenter h_triangle O)
  (h_nine_point_center : NinePointCenter h_triangle N) :
  ∃ radical_axis,
  radical_axis.perpendicular (Line_through O N) ∧
  radical_axis = Line.radicalAxis Ω nine_point_circle := 
sorry -- Proof not required.

end radical_axis_of_circum_and_nine_point_circle_l37_37846


namespace bill_annual_healthcare_cost_l37_37409

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end bill_annual_healthcare_cost_l37_37409


namespace find_a_l37_37612

theorem find_a (a b c : ℂ) (ha : a.re = a) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 6) : a = 1 :=
by
  sorry

end find_a_l37_37612


namespace distance_AB_l37_37182

def point := ℝ × ℝ × ℝ

def distance (A B : point) : ℝ := 
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2)

def A : point := (1, 1, 1)
def B : point := (-3, -3, -3)

theorem distance_AB : distance A B = 4 * real.sqrt 3 := 
  sorry

end distance_AB_l37_37182


namespace sam_last_30_minutes_speed_l37_37648

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end sam_last_30_minutes_speed_l37_37648


namespace center_of_circle_l37_37067

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 - 10 * x + 4 * y = -40) : 
  x + y = 3 := 
sorry

end center_of_circle_l37_37067


namespace base4_div_quotient_l37_37862

def base4_to_base10 (n : ℕ) : ℕ :=
  let digits := List.ofFn (λ i => ((n / (4^i)) % 4)) (Nat.log n / Nat.log 4 + 1)
  digits.foldl (λ acc x => acc * 4 + x) 0

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n : ℕ) (acc : ℕ) (pos : ℕ) : ℕ :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * (10^pos)) (pos + 1)
  convert n 0 0

def base4_div (num1 num2 : ℕ) : ℕ :=
  let num1_base10 := base4_to_base10 num1
  let num2_base10 := base4_to_base10 num2
  let quotient_base10 := num1_base10 / num2_base10
  base10_to_base4 quotient_base10

theorem base4_div_quotient : base4_div 1332 13 = 102 :=
by
  unfold base4_to_base10
  unfold base10_to_base4
  unfold base4_div
  sorry

end base4_div_quotient_l37_37862


namespace possible_values_of_ceil_x_sq_l37_37974

-- Definitions based on conditions
def is_real_number (x : ℝ) : Prop := true

-- Define the ceil function.
def ceil (x : ℝ) : ℤ := ⌈x⌉

-- Main theorem statement
theorem possible_values_of_ceil_x_sq (x : ℝ) (h : ceil x = 11) :
  (∃ n, 101 ≤ n ∧ n ≤ 121 ∧ ∀ m, 101 ≤ m ∧ m ≤ 121 → ceil (x * x) = m) ↔ (finset.card (finset.range 21) = 21) :=
by
  -- Proof would go here, but it is elided for the sake of this task
  sorry

end possible_values_of_ceil_x_sq_l37_37974


namespace exists_unobserved_planet_l37_37564

-- Given conditions and definitions
variables {Planet : Type} [fintype Planet]

-- Assume there is a function that gives the distance between any two planets.
variable (distance : Planet → Planet → ℝ)

-- Assume there is an odd number of planets
variable (h_odd : fintype.card Planet % 2 = 1)

-- Assume all distances are distinct
variable (h_distinct : ∀ (p1 p2 p3 p4 : Planet), p1 ≠ p2 ∧ p3 ≠ p4 → distance p1 p2 ≠ distance p3 p4)

-- Function to get the closest planet for a given planet
variable (closest_planet : Planet → Planet)

-- Each planet has one astronomer who studies the planet closest to their own
variable (h_closest : ∀ p : Planet, p ≠ closest_planet p ∧ (∀ q : Planet, q ≠ p → distance p (closest_planet p) ≤ distance p q))

-- Formal statement of the theorem
theorem exists_unobserved_planet : ∃ p : Planet, ∀ q : Planet, closest_planet q ≠ p := by
  sorry

end exists_unobserved_planet_l37_37564


namespace volume_of_smaller_solid_l37_37037

-- Definition of points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Cube vertices
def A : Point3D := {x := 0, y := 0, z := 0}
def E : Point3D := {x := 0, y := 2, z := 0}
def F : Point3D := {x := 2, y := 2, z := 0}
def G : Point3D := {x := 2, y := 2, z := 2}
def H : Point3D := {x := 0, y := 2, z := 2}

-- Midpoints of EF and GH
def P : Point3D := {x := 1, y := 2, z := 0}
def Q : Point3D := {x := 1, y := 2, z := 2}

-- Volume of a pyramid given its base area and height
def pyramid_volume (base_area height : ℝ) : ℝ := (1 / 3) * base_area * height

def correct_volume : ℝ := 0.5

-- Lean theorem statement asserting the volume of the smaller solid
theorem volume_of_smaller_solid :
  ∃ (v : ℝ), v = correct_volume ∧
  let cube_total_volume := 8 in
  let volume_removed := 2 * (pyramid_volume (1 / 2) (1.5)) in
  let smaller_volume := cube_total_volume - volume_removed in
  v = smaller_volume := by
  sorry

end volume_of_smaller_solid_l37_37037


namespace problem_I_problem_II_l37_37524

noncomputable def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 4) * x - Real.pi / 6)

theorem problem_I (ω : ℝ) (φ : ℝ)
    (hω : ω > 0)
    (hφ1 : |φ| ≤ Real.pi / 2)
    (h_highest : ∃ (x : ℝ), x = 8 / 3 ∧ f x = 1)
    (h_zero_neigh : ∃ (x : ℝ), x = 14 / 3 ∧ f x = 0) :
    f = (λ x, Real.sin ((Real.pi / 4) * x - Real.pi / 6)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f x - 2 * (Real.cos ((Real.pi / 8) * x)) ^ 2 + 1

theorem problem_II : 
  ∀ x ∈ Set.Icc (2 / 3) 2, 
  - Real.sqrt 3 / 2 ≤ g x ∧ g x ≤ Real.sqrt 3 / 2 :=
sorry

end problem_I_problem_II_l37_37524


namespace total_logs_in_stack_l37_37004

theorem total_logs_in_stack : 
  ∀ (a_1 a_n : ℕ) (n : ℕ), 
  a_1 = 5 → a_n = 15 → n = a_n - a_1 + 1 → 
  (a_1 + a_n) * n / 2 = 110 :=
by
  intros a_1 a_n n h1 h2 h3
  sorry

end total_logs_in_stack_l37_37004


namespace problem_statement_l37_37655

-- Define a multiple of 6 and a multiple of 9
variables (a b : ℤ)
variable (ha : ∃ k, a = 6 * k)
variable (hb : ∃ k, b = 9 * k)

-- Prove that a + b is a multiple of 3
theorem problem_statement : 
  (∃ k, a + b = 3 * k) ∧ 
  ¬((∀ m n, a = 6 * m ∧ b = 9 * n → (a + b = odd))) ∧ 
  ¬(∃ k, a + b = 6 * k) ∧ 
  ¬(∃ k, a + b = 9 * k) :=
by
  sorry

end problem_statement_l37_37655


namespace correlated_relationships_l37_37395

-- Defining the conditions as relations
def heightWeight (h w : ℝ) : Prop := sorry  -- Placeholder definition
def distanceSpeedTime (d s t : ℝ) : Prop := d = s * t
def heightEyesight (h e : ℝ) : Prop := sorry  -- Placeholder definition
def volumeEdgeLength (v e : ℝ) : Prop := v = e^3

-- Defining what it means for a relationship to be correlated
def isCorrelated {α β : Type} (relation : α → β → Prop) : Prop := sorry  -- Placeholder definition

axiom heightWeight_correlated : isCorrelated heightWeight
axiom distanceSpeedTime_correlated : isCorrelated distanceSpeedTime
axiom heightEyesight_not_correlated : ¬ isCorrelated heightEyesight
axiom volumeEdgeLength_correlated : isCorrelated volumeEdgeLength

-- Main theorem stating which relationships are correlated
theorem correlated_relationships :
  {heightWeight, distanceSpeedTime, volumeEdgeLength} = {heightWeight, distanceSpeedTime, volumeEdgeLength} :=
by
  sorry

end correlated_relationships_l37_37395


namespace maximum_k_value_l37_37273

open Nat

-- We define the sequence a_n and the sequence of partial sums S_n
variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition 1: The infinite sequence consists of k different numbers
def distinct_elements (k : ℕ) : Prop :=
  ∃ (s : Finset ℤ), s.card = k ∧ ∀ n, a n ∈ s

-- Condition 2: S_n is the sum of the first n terms of a_n
def sum_of_terms : Prop :=
  ∀ n, S n = (Finset.range (n + 1)).sum (λ i, a i)

-- Condition 3: For any n in ℕ*, S_n is in {1, 3}
def valid_partial_sum : Prop :=
  ∀ n, n > 0 → ((S n = 1) ∨ (S n = 3))

-- The theorem we need to prove
theorem maximum_k_value (a : ℕ → ℤ) (S : ℕ → ℤ) :
  distinct_elements a 4 →
  sum_of_terms a S →
  valid_partial_sum S →
  ∃ k, k ≤ 4 ∧ distinct_elements a k := by
    sorry

end maximum_k_value_l37_37273


namespace hyperbola_eccentricity_l37_37951

-- Define the hyperbola conditions
variables (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2) (h₄ : c = 3)
-- Prove the relationship for a in the hyperbola equation
def hyperbola_eqn := b^2 = c^2 - a^2
-- Define eccentricity e
def eccentricity (c a : ℝ) := c / a

-- Axiom stating that the hyperbola equation holds
axiom hyperbola_eqn_axiom : hyperbola_eqn a b c

-- Main theorem statement to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity c  (Real.sqrt 5) = 3 * Real.sqrt 5 / 5 := 
by
  -- The proof goes here
  sorry

end hyperbola_eccentricity_l37_37951


namespace flies_will_collide_l37_37636

theorem flies_will_collide (P : Type) [convex_polyhedron P] (n : ℕ) (flies : P → list Fly)
  (h_speeds : ∀ (p : P) (f : Fly), f ∈ flies p → f.speed >= 1) :
  ∃ (f1 f2 : Fly) (p1 p2 : P), f1 ∈ flies p1 ∧ f2 ∈ flies p2 ∧ f1 ≠ f2 ∧ collides f1 f2 := sorry

end flies_will_collide_l37_37636


namespace angle_BCA_measure_l37_37178

theorem angle_BCA_measure
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_BAC : ℝ)
  (h1 : angle_ABC = 90)
  (h2 : angle_BAC = 2 * angle_BCA) :
  angle_BCA = 30 :=
by
  sorry

end angle_BCA_measure_l37_37178


namespace circle1_standard_form_circle2_standard_form_l37_37036

-- Define the first circle equation and its corresponding answer in standard form
theorem circle1_standard_form :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 4*y - 4 = 0) ↔ ((x + 1)^2 + (y + 2)^2 = 9) :=
by
  intro x y
  sorry

-- Define the second circle equation and its corresponding answer in standard form
theorem circle2_standard_form :
  ∀ x y : ℝ, (3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0) ↔ ((x + 1)^2 + (y + 1/2)^2 = 25/4) :=
by
  intro x y
  sorry

end circle1_standard_form_circle2_standard_form_l37_37036


namespace ellipse_equation_line_passes_fixed_point_l37_37091

variables {k : ℝ} (a b : ℝ) (x y : ℝ) (A B P Q : ℝ × ℝ)
hypothesis minor_axis_length : b = sqrt 3
hypothesis point_on_ellipse : ∃ a : ℝ, a > b ∧ (1^2 / a^2) + ((-3/2)^2 / b^2) = 1
hypothesis vertices_of_ellipse : A = (-2, 0) ∧ B = (2, 0)
hypothesis distinct_points_on_ellipse : P ≠ A ∧ P ≠ B ∧ Q ≠ A ∧ Q ≠ B ∧ P ≠ Q
hypothesis slopes : ∃ k : ℝ, (P.2 - B.2) / (P.1 - B.1) = k ∧ (Q.2 - A.2) / (Q.1 - A.1) = 2 * k

theorem ellipse_equation :
  ∃ a : ℝ, a = 2 ∧ b = sqrt 3 ∧ ∀ x y : ℝ, (x^2 / (2^2)) + (y^2 / (sqrt 3 ^ 2)) = 1 :=
begin
  sorry
end

theorem line_passes_fixed_point :
  ∀ P Q : ℝ × ℝ, distinct_points_on_ellipse → slopes →
  ∃ M : ℝ × ℝ, M = (-2/3, 0) ∧ 
  ((P.2 - M.2) / (P.1 - M.1)) = ((Q.2 - M.2) / (Q.1 - M.1)) :=
begin
  sorry
end

end ellipse_equation_line_passes_fixed_point_l37_37091


namespace rearrange_students_l37_37393

theorem rearrange_students (n : ℕ) (students : Fin n.succ × Fin n.succ → Prop) :
  (∀ i j, students⟨i, j⟩ → ∃ k l, students⟨k, l⟩ ∧
    ((i = k ∧ j = l + 1) ∨ (i = k ∧ j + 1 = l) ∨ (i = k + 1 ∧ j = l) ∨ (i + 1 = k ∧ j = l))) ↔ ¬ (Odd n) :=
begin
  sorry,
end

end rearrange_students_l37_37393


namespace pebbles_difference_l37_37247

theorem pebbles_difference :
  ∃ (blue yellow : ℕ), 
    (∃ total : ℕ, total = 40) ∧
    (∃ red : ℕ, red = 9) ∧
    (∃ blue : ℕ, blue = 13) ∧ 
    (∃ remaining : ℕ, remaining = total - red - blue) ∧
    (∃ groups : ℕ, groups = 3) ∧
    (∃ yellow : ℕ, yellow = remaining / groups) →
  blue - yellow = 7 :=
begin
  sorry
end

end pebbles_difference_l37_37247


namespace sixth_grade_boys_l37_37003

theorem sixth_grade_boys (x : ℕ) :
    (1 / 11) * x + (147 - x) = 147 - x → 
    (152 - (x - (1 / 11) * x + (147 - x) - (152 - x - 5))) = x
    → x = 77 :=
by
  intros h1 h2
  sorry

end sixth_grade_boys_l37_37003


namespace math_problem_l37_37417

noncomputable def expr : ℝ :=
  (Real.pi - 4)^0 + |3 - Real.tan (Float.pi / 3)| - (1 / 2)^(-2 : ℤ) + Real.sqrt 27

theorem math_problem :
  expr = 2 * Real.sqrt 3 :=
sorry

end math_problem_l37_37417


namespace BC_l37_37019

/-
  Points A, B, C are on the x-axis (line l)
  Points A', B', C' are on the y-axis (line l')
  Lines AB' and A'B are parallel, implying a * a' = b * b'
  Lines AC' and A'C are parallel, implying a * a' = c * c'
  Necessarily, b * b' = c * c'
  Prove: Lines BC' and B'C are parallel
-/

variables {k : Type*} [field k]

-- Define coordinates of the points on line l
variables (a b c a' b' c' : k)

-- Conditions given in the problem
axiom AB'_parallel_A'B : a * a' = b * b'
axiom AC'_parallel_A'C : a * a' = c * c'

theorem BC'_parallel_B'C (h1 : a * a' = b * b') (h2 : a * a' = c * c') : 
  b * c' = c * b' := 
sorry

end BC_l37_37019


namespace sine_sum_square_greater_l37_37212

variable {α β : Real} (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1)

theorem sine_sum_square_greater (α β : Real) (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
  (h : Real.sin α ^ 2 + Real.sin β ^ 2 < 1) : 
  Real.sin (α + β) ^ 2 > Real.sin α ^ 2 + Real.sin β ^ 2 :=
sorry

end sine_sum_square_greater_l37_37212


namespace draw_4_balls_ordered_l37_37758

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37758


namespace max_solution_inequality_l37_37440

noncomputable def t (x : ℝ) : ℝ := 80 - 2 * x * real.sqrt (30 - 2 * x)
noncomputable def numerator (x : ℝ) : ℝ := - real.logb 3 (t x) ^ 2 + (abs (real.logb 3 (t x) - 3 * real.logb 3 (x ^ 2 - 2 * x + 29)))
noncomputable def denominator (x : ℝ) : ℝ := 7 * real.logb 7 (65 - 2 * x * real.sqrt (30 - 2 * x)) - 4 * real.logb 3 (t x)
noncomputable def inequality_expr (x : ℝ) : ℝ := numerator x / denominator x

theorem max_solution_inequality : 
  ∃ x, (x = 8 - real.sqrt 13) ∧ (inequality_expr x ≥ 0) := 
begin
  use 8 - real.sqrt 13,
  split,
  { refl },
  { sorry }
end

end max_solution_inequality_l37_37440


namespace solve_inequality_l37_37435

theorem solve_inequality (x : ℝ) :
  (x ≠ 3) → (x * (x + 2) / (x - 3)^2 ≥ 8) ↔ (x ∈ set.Iic (18/7) ∪ set.Ioi 4) :=
by
  sorry

end solve_inequality_l37_37435


namespace problem_statement_l37_37483

theorem problem_statement (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end problem_statement_l37_37483


namespace number_of_cars_l37_37168

theorem number_of_cars (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 68) (h2 : wheels_per_car = 4) : total_wheels / wheels_per_car = 17 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end number_of_cars_l37_37168


namespace common_ratio_geometric_sequence_l37_37497

-- Define the arithmetic sequence.
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Given conditions:
-- The sequence is arithmetic with a non-zero common difference.
variables (a d : ℤ) (h : d ≠ 0)

-- The 2nd, 3rd, and 6th terms form a geometric sequence.
theorem common_ratio_geometric_sequence (h : d ≠ 0) :
  let a2 := arithmetic_seq a d 1 in
  let a3 := arithmetic_seq a d 2 in
  let a6 := arithmetic_seq a d 5 in
  (a2 * a6 = a3 * a3) → (a3 / a2 = 3) :=
by
  -- Define the terms.
  let a2 := arithmetic_seq a d 1
  let a3 := arithmetic_seq a d 2
  let a6 := arithmetic_seq a d 5
  assume h_geometric : a2 * a6 = a3 * a3
  -- We proceed to show the common ratio is 3.
  sorry

end common_ratio_geometric_sequence_l37_37497


namespace sum_binomial_identity_l37_37068

theorem sum_binomial_identity (n : ℕ) (h : 0 < n) :
  ∑ k in Finset.range (n + 1), (nat.choose n k * (-1)^k / (n + 1 - k) ^ 2 - (-1)^n / (k + 1) / (n + 1)) = 0 := 
sorry

end sum_binomial_identity_l37_37068


namespace game_must_end_l37_37011

-- Define the necessary conditions for the game
structure GameState where
  segments : List (ℝ × ℝ × ℝ × ℝ)  -- List of segments starting and ending points
  current_player : Bool -- true for Alice, false for Bob

-- Define the game rules
def valid_segment (a b : ℝ × ℝ) (segments : List (ℝ × ℝ × ℝ × ℝ)) : Prop :=
  ∀ s ∈ segments, ¬ line_segment_intersect_except_at (a, b) s

-- Function to draw a segment ensuring game rules are satisfied
def draw_segment (state : GameState) (start end_: ℝ × ℝ) : Prop :=
  valid_segment start end_ state.segments ∧
  (dist start end_ = 1) ∧
  (state.segments = [] → state.current_player = true)

-- Define the theorem statement
theorem game_must_end (state : GameState) :
  (∃ n, n ≤ 65 ∧ (state.segments.length = n → end_game state)) :=
sorry

end game_must_end_l37_37011


namespace average_snowfall_per_minute_l37_37561

def total_snowfall := 550
def days_in_december := 31
def hours_per_day := 24
def minutes_per_hour := 60

theorem average_snowfall_per_minute :
  (total_snowfall : ℝ) / (days_in_december * hours_per_day * minutes_per_hour) = 550 / (31 * 24 * 60) :=
by
  sorry

end average_snowfall_per_minute_l37_37561


namespace angle_bisector_proof_l37_37724

noncomputable def triangle (A B C : Type*) := sorry

variables {Point : Type*}
variables (S Q T M O : Point)

variables [triangle S Q T]

variable (is_angle_bisector : ∀ A B C D : Point, Prop)

variable (angle_sum_condition : ∀ P1 P2 P3 : Point, Prop)

theorem angle_bisector_proof
  (h1 : is_angle_bisector S M Q T)
  (h2 : angle_sum_condition O Q T (O Q T)) :
  is_angle_bisector O M Q T :=
sorry

end angle_bisector_proof_l37_37724


namespace range_of_b_in_acute_triangle_l37_37175

variable {a b c : ℝ}

theorem range_of_b_in_acute_triangle (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_acute : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2))
  (h_arith_seq : ∃ d : ℝ, 0 ≤ d ∧ a = b - d ∧ c = b + d)
  (h_sum_squares : a^2 + b^2 + c^2 = 21) :
  (2 * Real.sqrt 42) / 5 < b ∧ b ≤ Real.sqrt 7 :=
sorry

end range_of_b_in_acute_triangle_l37_37175


namespace intersection_point_l37_37040

-- Mathematical problem translated to Lean 4 statement

theorem intersection_point : 
  ∃ x y : ℝ, y = -3 * x + 1 ∧ y + 1 = 15 * x ∧ x = 1 / 9 ∧ y = 2 / 3 := 
by
  sorry

end intersection_point_l37_37040


namespace pebbles_difference_l37_37246

theorem pebbles_difference :
  ∃ (blue yellow : ℕ), 
    (∃ total : ℕ, total = 40) ∧
    (∃ red : ℕ, red = 9) ∧
    (∃ blue : ℕ, blue = 13) ∧ 
    (∃ remaining : ℕ, remaining = total - red - blue) ∧
    (∃ groups : ℕ, groups = 3) ∧
    (∃ yellow : ℕ, yellow = remaining / groups) →
  blue - yellow = 7 :=
begin
  sorry
end

end pebbles_difference_l37_37246


namespace hexagon_vectors_l37_37085

variable (a b : Vector ℝ 2)
variable (A B C D E F M O N : Point ℝ 2)
variable (h1 : RegularHexagon A B C D E F)
variable (h2 : B = A + a)
variable (h3 : F = A + b)
variable (M_midpoint : Midpoint M E F)

theorem hexagon_vectors
  (h : RegularHexagon A B C D E F)
  (hab : \overrightarrow{A B} = a)
  (haf : \overrightarrow{A F} = b) :
  ( \overrightarrow{A D} = 2 * a + 2 * b) ∧
  ( \overrightarrow{B D} = a + 2 * b) ∧
  ( \overrightarrow{F D} = 2 * a + b) ∧
  ( \overrightarrow{B M} = -\frac{1}{2} * a + \frac{3}{2} * b) := sorry

end hexagon_vectors_l37_37085


namespace min_length_AB_l37_37174

-- Definition of points A and B
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := { x := 1, y := 0, z := 2 }
def B (t : ℝ) : Point3D := { x := t, y := 2, z := -1 }

-- Distance function for two points in 3D space
def dist (P1 P2 : Point3D) : ℝ :=
  real.sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2 + (P2.z - P1.z)^2)

-- Minimum length of the line segment AB
theorem min_length_AB : ∃ t, dist A (B t) = real.sqrt 13 := 
by
  use 1
  unfold dist A B
  simp
  sorry

end min_length_AB_l37_37174


namespace ratio_of_x_and_y_l37_37025

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 0.25 :=
by
  sorry

end ratio_of_x_and_y_l37_37025


namespace max_solution_inequality_l37_37442

def t (x : ℝ) : ℝ :=
  80 - 2 * x * real.sqrt (30 - 2 * x)

noncomputable def inequality_expr (x : ℝ) : ℝ :=
  let numerator := -real.log 3 (t x)^2 + abs (real.log 3 (t x) - 3 * real.log 3 ((x^2 - 2*x + 29)^3))
  let denominator := 7 * real.log 7 (65 - 2 * x * real.sqrt (30 - 2 * x)) - 4 * real.log 3 (t x)
  numerator / denominator

theorem max_solution_inequality : ∃ x, (inequality_expr x ≥ 0) ∧ (x = 8 - real.sqrt 13) :=
by
  sorry

end max_solution_inequality_l37_37442


namespace irrational_sqrt2_l37_37828

theorem irrational_sqrt2 : irrational (real.sqrt 2) 
:= sorry

end irrational_sqrt2_l37_37828


namespace group_of_five_l37_37565

theorem group_of_five (G : Finset ℕ) (hG : G.card = 9) (h_pairwise : ∀ (a b c : ℕ), a ∈ G → b ∈ G → c ∈ G → (a ≠ b ∧ b ≠ c ∧ a ≠ c) → (a ∈ G ∩ b's friends ∨ a ∈ G ∩ c's friends ∨ b ∈ G ∩ c's friends)) :
  ∃ (H : Finset ℕ), H ⊆ G ∧ H.card = 5 ∧ ∀ x ∈ H, (H ∩ x's friends).card ≥ 4 :=
sorry

end group_of_five_l37_37565


namespace prob_score_5_points_is_three_over_eight_l37_37990

noncomputable def probability_of_scoring_5_points : ℚ :=
  let total_events := 2^3
  let favorable_events := 3 -- Calculated from combinatorial logic.
  favorable_events / total_events

theorem prob_score_5_points_is_three_over_eight :
  probability_of_scoring_5_points = 3 / 8 :=
by
  sorry

end prob_score_5_points_is_three_over_eight_l37_37990


namespace overall_sale_price_per_kg_l37_37820

-- Defining the quantities and prices
def tea_A_quantity : ℝ := 80
def tea_A_cost_per_kg : ℝ := 15
def tea_B_quantity : ℝ := 20
def tea_B_cost_per_kg : ℝ := 20
def tea_C_quantity : ℝ := 50
def tea_C_cost_per_kg : ℝ := 25
def tea_D_quantity : ℝ := 40
def tea_D_cost_per_kg : ℝ := 30

-- Defining the profit percentages
def tea_A_profit_percentage : ℝ := 0.30
def tea_B_profit_percentage : ℝ := 0.25
def tea_C_profit_percentage : ℝ := 0.20
def tea_D_profit_percentage : ℝ := 0.15

-- Desired sale price per kg
theorem overall_sale_price_per_kg : 
  (tea_A_quantity * tea_A_cost_per_kg * (1 + tea_A_profit_percentage) +
   tea_B_quantity * tea_B_cost_per_kg * (1 + tea_B_profit_percentage) +
   tea_C_quantity * tea_C_cost_per_kg * (1 + tea_C_profit_percentage) +
   tea_D_quantity * tea_D_cost_per_kg * (1 + tea_D_profit_percentage)) / 
  (tea_A_quantity + tea_B_quantity + tea_C_quantity + tea_D_quantity) = 26 := 
by
  sorry

end overall_sale_price_per_kg_l37_37820


namespace tan_2α_plus_pi_over_4_l37_37105

noncomputable def α : ℝ := sorry
noncomputable def sin_α : ℝ := 3 / 5
noncomputable def quadrant : ∀ k : ℤ, (kπ < α ∧ α < k*π + π/2) := sorry

theorem tan_2α_plus_pi_over_4 :
  sin α = 3 / 5 → (∀ k : ℤ, kπ < α ∧ α < k*π + π/2) → tan (2*α + π/4) = -17 / 31 :=
by
  intros hsin hquad
  sorry

end tan_2α_plus_pi_over_4_l37_37105


namespace final_cost_correct_l37_37819

noncomputable theory

-- Define the conditions
def original_price (tax_inclusive_price : ℝ) := tax_inclusive_price / 1.40

def price_after_tax {P : ℝ} := 1.40 * P

def discount {price : ℝ} := 0.20 * price

def cashback {price : ℝ} := 0.10 * price

def final_cost (tax_inclusive_price : ℝ) := (tax_inclusive_price - (discount tax_inclusive_price)) - cashback (tax_inclusive_price - discount tax_inclusive_price)

-- The final theorem to be proved
theorem final_cost_correct : final_cost 1680 = 1209.60 :=
by
    sorry

end final_cost_correct_l37_37819


namespace parabola_directrix_is_x_eq_1_l37_37891

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37891


namespace find_digit_A_l37_37155

theorem find_digit_A :
  ∃ A : ℕ, 
    2 * 10^6 + A * 10^5 + 9 * 10^4 + 9 * 10^3 + 5 * 10^2 + 6 * 10^1 + 1 = (3 * (523 + A)) ^ 2 
    ∧ A = 4 :=
by
  sorry

end find_digit_A_l37_37155


namespace trig_identity_example_l37_37026

theorem trig_identity_example :
  sin(65 * (Real.pi / 180)) * sin(115 * (Real.pi / 180)) + cos(65 * (Real.pi / 180)) * sin(25 * (Real.pi / 180)) = 1 :=
by
  sorry

end trig_identity_example_l37_37026


namespace meaningful_expression_l37_37707

theorem meaningful_expression (x : ℝ) : (∃ y, y = sqrt (3 - x) + 1 / (3 * x - 1)) ↔ (x ≤ 3 ∧ x ≠ 1 / 3) :=
by sorry

end meaningful_expression_l37_37707


namespace max_cubes_submerged_l37_37716

noncomputable def cylinder_radius (diameter: ℝ) : ℝ := diameter / 2

noncomputable def water_volume (radius height: ℝ) : ℝ := Real.pi * radius^2 * height

noncomputable def cube_volume (edge: ℝ) : ℝ := edge^3

noncomputable def height_of_cubes (edge n: ℝ) : ℝ := edge * n

theorem max_cubes_submerged (diameter height water_height edge: ℝ) 
  (h1: diameter = 2.9)
  (h2: water_height = 4)
  (h3: edge = 2):
  ∃ max_n: ℝ, max_n = 5 := 
  sorry

end max_cubes_submerged_l37_37716


namespace trapezoid_perimeter_l37_37570

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD BC DA : ℝ)
  (AB_parallel_CD : AB = CD)
  (BC_eq_DA : BC = 13)
  (DA_eq_BC : DA = 13)
  (sum_AB_CD : AB + CD = 24)

-- Define the problem's conditions as Lean definitions
def trapezoidABCD : IsoscelesTrapezoid ℝ ℝ ℝ ℝ :=
{
  AB := 12,
  CD := 12,
  BC := 13,
  DA := 13,
  AB_parallel_CD := by sorry,
  BC_eq_DA := by sorry,
  DA_eq_BC := by sorry,
  sum_AB_CD := by sorry,
}

-- State the theorem we want to prove
theorem trapezoid_perimeter (trapezoid : IsoscelesTrapezoid ℝ ℝ ℝ ℝ) : 
  trapezoid.AB + trapezoid.BC + trapezoid.CD + trapezoid.DA = 50 :=
by sorry

end trapezoid_perimeter_l37_37570


namespace number_of_ways_to_draw_4_from_15_l37_37748

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37748


namespace triangle_longest_side_l37_37677

theorem triangle_longest_side (y : ℝ) (h_perimeter : 10 + (y + 6) + (3y + 2) = 45) :
  max 10 (max (y + 6) (3y + 2)) = 22.25 :=
sorry

end triangle_longest_side_l37_37677


namespace find_25_percent_l37_37266

theorem find_25_percent (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 :=
by
  sorry

end find_25_percent_l37_37266


namespace incorrect_statement_D_l37_37964

variables {α β : Type*} {l m n : Type*} 
variable [plane α] [plane β] [line l] [line m] [line n]

def is_perpendicular (x y : Type*) [plane x] [plane y] : Prop := sorry
def is_parallel (x y : Type*) [line x] [line y] : Prop := sorry
def is_subset (x y : Type*) [line x] [plane y] : Prop := sorry
def is_not_subset (x y : Type*) [line x] [plane y] : Prop := sorry
def intersection (x y : Type*) [plane x] [plane y] [line l] : Prop := sorry

theorem incorrect_statement_D :
  is_perpendicular α β →
  is_perpendicular m α →
  is_perpendicular n β →
  ¬ is_perpendicular m n :=
by
  assume h1 h2 h3,
  sorry

end incorrect_statement_D_l37_37964


namespace compare_fractions_sqrt_inequality_l37_37078

-- Conditions
variables {x y m : ℝ}

-- Problem 1
theorem compare_fractions (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0) : 
  (y / x) < ((y + m) / (x + m)) :=
  sorry

-- Problem 2
theorem sqrt_inequality (hx : x > 0) (hy : y > 0) : 
  (√(x * y) * (2 - √(x * y))) ≤ 1 :=
  sorry

end compare_fractions_sqrt_inequality_l37_37078


namespace compound_interest_l37_37457

variable (a r : ℝ) (x y : ℝ)

theorem compound_interest :
  a = 1000 → r = 0.0225 → x = 4 → y = a * (1 + r) ^ x → y ≈ 1093.08 :=
by
  intros ha hr hx hy
  rw [ha, hr, hx] at hy
  sorry

end compound_interest_l37_37457


namespace winning_vertex_l37_37831

noncomputable def winning_vertex_exists_for_pentagon (n: ℕ) : Prop :=
  ∀ (initial : Fin 5 → ℤ), 
  (∑ i, initial i = n) →
  ∃! (v : Fin 5), ∃ (turns : ℕ → (Fin 5) × ℤ),  
    (∀ t, turns t.snd → (∑ i, (if i = v then 2 * turns t.snd + initial i else initial i - turns t.snd) = n)) ∧ 
    (∀ i ≠ v, (if i ≠ v then initial i - turns t.snd = 0 else initial i - turns t.snd ≠ 0))

-- Given condition: sum of integers at vertices equals 2011
theorem winning_vertex (initial : Fin 5 → ℤ) (h : (∑ i, initial i) = 2011) :
  winning_vertex_exists_for_pentagon 2011
:=
sorry

end winning_vertex_l37_37831


namespace packets_of_chips_l37_37271

theorem packets_of_chips (x : ℕ) 
  (h1 : ∀ x, 2 * (x : ℝ) + 1.5 * (10 : ℝ) = 45) : 
  x = 15 := 
by 
  sorry

end packets_of_chips_l37_37271


namespace equilateral_triangle_side_length_on_parabola_l37_37955

theorem equilateral_triangle_side_length_on_parabola :
  ∀ (a : ℝ), 
  a ≠ 0 → 
  let parabola (x y : ℝ) := y^2 = a * x,
      symmetry_axis (x : ℝ) := x = -3,
      is_equilateral_triangle_on_parabola (A B C : ℝ × ℝ) := 
        ∀ (O : ℝ × ℝ), (O = (0, 0)) →
        A = O ∧ 
        B.fst = C.fst ∧ B.snd = -C.snd ∧ 
        parabola A.fst A.snd ∧ 
        parabola B.fst B.snd ∧ 
        parabola C.fst C.snd ∧ 
        (dist B C = dist A B ∧ dist A B = dist A C)
  in 
    ∀ (A B C : ℝ × ℝ),
      is_equilateral_triangle_on_parabola A B C →
      let side_length_square (A B : ℝ × ℝ) := 
        (B.fst - A.fst)^2 + (B.snd - A.snd)^2
      in 
        (side_length_square A B) = (24 * (sqrt 3))^2 :=
sorry

end equilateral_triangle_side_length_on_parabola_l37_37955


namespace heat_to_increase_temperature_l37_37096

noncomputable def heat_required 
  (V : ℝ) -- Volume of the container
  (M : ℝ) -- Molar mass of the gas
  (c_p : ℝ) -- Specific heat at constant pressure
  (p : ℝ := 101325) -- External pressure in Pascals (1 atm = 101325 Pa)
  (R : ℝ := 8.3145) -- Universal gas constant
  (T0 : ℝ) -- Initial temperature
  (T1 : ℝ) -- Final temperature
  : ℝ :=
  c_p * (p * V * M / R) * Real.log (T1 / T0)

theorem heat_to_increase_temperature
  (V : ℝ) 
  (M : ℝ) 
  (c_p : ℝ) 
  (T0 : ℝ)
  (T1 : ℝ) 
  (p : ℝ := 101325) 
  (R : ℝ := 8.3145) 
  : heat_required V M c_p T0 T1 = c_p * (p * V * M / R) * Real.log (T1 / T0) := 
by sorry

end heat_to_increase_temperature_l37_37096


namespace rational_root_of_polynomial_with_rational_coefficients_l37_37279

-- Question and conditions as definitions
variables (a b c : ℚ)
def poly (x : ℂ) : ℂ := x^3 + (a : ℂ) * x^2 + (b : ℂ) * x + (c : ℂ)

theorem rational_root_of_polynomial_with_rational_coefficients
  (h_root1 : poly a b c (3 + complex.sqrt 5) = 0)
  (h_root2 : poly a b c (3 - complex.sqrt 5) = 0)
  (h_rational_root_exists : ∃ r : ℚ, poly a b c r = 0):
  ∃ r : ℚ, r = -6 :=
by
  sorry

end rational_root_of_polynomial_with_rational_coefficients_l37_37279


namespace ten_percent_markup_and_markdown_l37_37809

theorem ten_percent_markup_and_markdown (x : ℝ) (hx : x > 0) : 0.99 * x < x :=
by 
  sorry

end ten_percent_markup_and_markdown_l37_37809


namespace smallest_n_l37_37905

noncomputable def f (p q r : ℕ) : ℕ :=
  (nat.factorial p) ^ p * (nat.factorial q) ^ q * (nat.factorial r) ^ r

theorem smallest_n (n : ℕ) :
  (∀ (a b c x y z : ℕ), a + b + c = 2020 → x + y + z = n → f x y z % f a b c = 0) ↔ n = 6052 := 
by
  sorry

end smallest_n_l37_37905


namespace min_value_of_expression_l37_37508

open Real

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (minval : ℝ), minval = 4 ∧ ∀ a b > 0, (1 / a + 1 / b + 2 * sqrt (a * b)) ≥ minval :=
begin
  use 4,
  split,
  { refl },
  { intros a b ha hb,
    sorry }
end

end min_value_of_expression_l37_37508


namespace infinite_series_problem_l37_37419

noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (2 * (n + 1)^2 - 3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))

theorem infinite_series_problem :
  infinite_series_sum = -4 :=
by sorry

end infinite_series_problem_l37_37419


namespace initial_money_Yanni_l37_37722

variable (X : ℝ)
variable (h_mother : Yanni_initial + 0.40 = X + 0.40)
variable (h_found : Yanni_initial + 0.40 + 0.50 = X + 0.40 + 0.50)
variable (h_toy : Yanni_initial + 0.40 + 0.50 - 1.60 = X + 0.15)

theorem initial_money_Yanni :
  X = 0.85 :=
begin
  sorry
end

end initial_money_Yanni_l37_37722


namespace tangent_line_cannot_be_four_l37_37625

/-
  Prove that for the function 
  \( f(x) = \sqrt{3} \sin \left( 2x + \frac{\pi}{3} \right) \), 
  the derivative \( f'(x) \) can never be equal to \( 4 \) for any \( x \).
-/

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 3) * Real.sin (2 * x + Real.pi / 3)

theorem tangent_line_cannot_be_four (x : ℝ) : 
  let f' := fun x => 2 * (Real.sqrt 3) * Real.cos (2 * x + Real.pi / 3) in
  f' x ≠ 4 := by
  let f' := fun x => 2 * (Real.sqrt 3) * Real.cos (2 * x + Real.pi / 3)
  sorry

end tangent_line_cannot_be_four_l37_37625


namespace prove_correct_y_l37_37728

noncomputable def find_larger_y (x y : ℕ) : Prop :=
  y - x = 1365 ∧ y = 6 * x + 15

noncomputable def correct_y : ℕ := 1635

theorem prove_correct_y (x y : ℕ) (h : find_larger_y x y) : y = correct_y :=
by
  sorry

end prove_correct_y_l37_37728


namespace directrix_of_parabola_l37_37895

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37895


namespace problem_A_problem_B_problem_C_problem_D_l37_37161

-- Define the sequence and properties
namespace Sequences

def is_seq (a : Fin 5 → ℕ) : Prop :=
  (Set.range a).toFinset = {1, 2, 3, 4, 5}.toFinset

-- Define the conditions
def cond_A (a : Fin 5 → ℕ) : Prop :=
  a 2 = 3 ∧ (a 0 + a 1 < a 3 + a 4)

def cond_B (a : Fin 5 → ℕ) : Prop :=
  (a 0 % 2 ≠ a 1 % 2) ∧
  (a 1 % 2 ≠ a 2 % 2) ∧ 
  (a 2 % 2 ≠ a 3 % 2) ∧
  (a 3 % 2 ≠ a 4 % 2)

def cond_C (a : Fin 5 → ℕ) : Prop :=
  ∃ i : ℕ, 1 ≤ i ∧ i < 5 ∧ ∀ j < i, a j > a (j + 1) ∧ ∀ j ≥ i, a j < a (j + 1)

def cond_D (a : Fin 5 → ℕ) : Prop :=
  a 0 < a 1 ∧ a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 < a 4

-- Problem statements
theorem problem_A : ∀ a : Fin 5 → ℕ, is_seq a → cond_A a → False := 
by
  sorry

theorem problem_B : ∃ n, ∀ a : Fin 5 → ℕ, is_seq a → cond_B a → n = 12 := 
by 
  sorry

theorem problem_C : ∃ n, ∀ a : Fin 5 → ℕ, is_seq a → cond_C a → n = 14 := 
by 
  sorry

theorem problem_D : ∃ n, ∀ a : Fin 5 → ℕ, is_seq a → cond_D a → n = 11 := 
by 
  sorry

end Sequences

end problem_A_problem_B_problem_C_problem_D_l37_37161


namespace problem_1_problem_2_l37_37961

open Set

variables {U : Type*} [TopologicalSpace U] (a x : ℝ)

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def N (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a + 1 }

noncomputable def complement_N (a : ℝ) : Set ℝ := { x | x < a + 1 ∨ 2 * a + 1 < x }

theorem problem_1 (h : a = 2) :
  M ∩ (complement_N a) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

theorem problem_2 (h : M ∪ N a = M) :
  a ≤ 2 :=
sorry

end problem_1_problem_2_l37_37961


namespace stable_shape_l37_37318

-- Define the shapes
inductive Shape
| Parallelogram
| Square
| Rectangle
| RightTriangle

open Shape

-- State the problem in Lean
theorem stable_shape (s : Shape) : s = RightTriangle :=
by
  cases s
  sorry

end stable_shape_l37_37318


namespace number_of_zeros_l37_37945

-- Define the piecewise function f
def f (k : ℝ) (x : ℝ) : ℝ :=
  if x <= 2 then k * x + 2 else Real.log x

-- State the problem as a theorem
theorem number_of_zeros (k : ℝ) (hk : k > 0) :
  ∃ x1 x2 x3 x4 : ℝ, |f k x1| - 1 = 0 ∧ |f k x2| - 1 = 0 ∧ |f k x3| - 1 = 0 ∧ |f k x4| - 1 = 0 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 :=
by
  sorry

end number_of_zeros_l37_37945


namespace find_k_l37_37210

def f (x k : ℝ) : ℝ := (3 * x^2 + 2 * x + 1) / (k * x^2 + 2 * x - 3)

def is_inverse (x k : ℝ) := f (f x k) k = x

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x ∈ set.Ioo (9 * k + 1) (9 * k + 2) -> is_inverse x k) ↔
  k ∈ (set.Ioo (-∞ : ℝ) (-2 / 9) ∪ set.Ioo (-2 / 9) (∞ : ℝ)) :=
sorry

end find_k_l37_37210


namespace range_of_m_l37_37504

variable (x m : ℝ)

def p := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q := x^2 - 2 * x + (1 - m^2) ≤ 0 ∧ m > 0

theorem range_of_m (h : ¬p → ¬q) : 9 ≤ m :=
by {
  sorry
}

end range_of_m_l37_37504


namespace no_tiling_9x9_with_dominoes_l37_37186

def domino_tiling_problem : Prop :=
  ∀ k : ℕ, 9 * 9 ≠ 2 * k

theorem no_tiling_9x9_with_dominoes : domino_tiling_problem :=
by
  intro k
  have h81 : 81 = 9 * 9 := rfl
  have hodd : odd 81 := by
    apply odd.intro
    use 40
    norm_num
  have h2k : even (2 * k) := by
    apply even_mul
    norm_num
  intro h
  rw [h] at hodd
  exact absurd hodd h2k

end no_tiling_9x9_with_dominoes_l37_37186


namespace exist_rectangle_same_color_l37_37278

-- Define the colors.
inductive Color
| red
| green
| blue

open Color

-- Define the point and the plane.
structure Point :=
(x : ℝ) (y : ℝ)

-- Assume a coloring function that assigns colors to points on the plane.
def coloring : Point → Color := sorry

-- The theorem stating the existence of a rectangle with vertices of the same color.
theorem exist_rectangle_same_color :
  ∃ (A B C D : Point), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  coloring A = coloring B ∧ coloring B = coloring C ∧ coloring C = coloring D :=
sorry

end exist_rectangle_same_color_l37_37278


namespace curve_transformation_l37_37965

noncomputable def curve_type (α : ℝ) : Type :=
  if α = 0 then UnitCircle
  else if 0 < α ∧ α < π / 2 then Ellipse
  else if α = π / 2 then ParallelLines
  else if π / 2 < α ∧ α < π then Hyperbola
  else RectangularHyperbola

theorem curve_transformation :
  (∀ α, 0 ≤ α ∧ α ≤ π →
    (curve_type α = UnitCircle ↔ α = 0) ∧
    (curve_type α = Ellipse ↔ 0 < α ∧ α < π / 2) ∧
    (curve_type α = ParallelLines ↔ α = π / 2) ∧
    (curve_type α = Hyperbola ↔ π / 2 < α ∧ α < π) ∧
    (curve_type α = RectangularHyperbola ↔ α = π)) :=
by { sorry }

end curve_transformation_l37_37965


namespace part1_part2_maxmin_l37_37146

variables (x : ℝ)

def a := (Real.cos x, Real.sin x)
def b := (3 : ℝ, -Real.sqrt 3)
def f := (Real.cos x, Real.sin x).1 * 3 + (Real.cos x, Real.sin x).2 * -Real.sqrt 3

-- Problem (1) statement
theorem part1 (hx : x ∈ set.Icc 0 Real.pi) (hperp : a x.1 * (b x).1 + a x.2 * (b x).2 = 0) :
  x = (5 * Real.pi) / 6 := sorry

-- Problem (2) statement
theorem part2_maxmin (hx : x ∈ set.Icc 0 Real.pi) :
  (f x).supset = (3, 0) ∧ (f x).infsupset = (-2*Real.sqrt 3, (5 * Real.pi) / 6) := sorry

end part1_part2_maxmin_l37_37146


namespace number_of_odd_decreasing_three_digit_integers_l37_37545

theorem number_of_odd_decreasing_three_digit_integers : 
  let N := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ (n % 2 = 1) ∧ 
                   (∃ a b c : ℕ, a > b ∧ b > c ∧ n = 100 * a + 10 * b + c)} in
  fintype.card N = 50 :=
by
  sorry

end number_of_odd_decreasing_three_digit_integers_l37_37545


namespace find_m_l37_37123

theorem find_m (
  x : ℚ 
) (m : ℚ) 
  (h1 : 4 * x + 2 * m = 3 * x + 1) 
  (h2 : 3 * x + 2 * m = 6 * x + 1) 
: m = 1/2 := 
  sorry

end find_m_l37_37123


namespace div_transitivity_l37_37153

theorem div_transitivity (a b c : ℚ) : 
  (a / b = 3) → (b / c = 2 / 5) → (c / a = 5 / 6) :=
by 
  intros h1 h2
  have : c / a = (c / b) * (b / a),
  { field_simp, }
  rw h1 at this,
  rw h2 at this,
  field_simp at this,
  exact this

end div_transitivity_l37_37153


namespace ball_drawing_ways_l37_37763

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37763


namespace range_of_a_l37_37269

-- Define the function f(x)
noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the derivative of the function
noncomputable def f' (a x : ℝ) : ℝ := 2*x - 2*a

-- Define the decreasing condition for the interval (-∞, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 2 → f'(a, x) < 0

-- State the theorem that must be proven
theorem range_of_a (a : ℝ) :
  is_decreasing_in_interval a ↔ 2 ≤ a := 
sorry

end range_of_a_l37_37269


namespace geometric_sum_S6_l37_37517

theorem geometric_sum_S6 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0) (h_q : q > 1) (h_a3_a5 : a 3 + a 5 = 20) (h_a2_a6 : a 2 * a 6 = 64) :
  S 6 = a 1 * (1 - q^6) / (1 - q) :=
begin
  sorry
end

end geometric_sum_S6_l37_37517


namespace domain_of_g_l37_37850

def g (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x ^ 2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | x^2 - 5 * x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_g_l37_37850


namespace cos_sine_identity_l37_37459

theorem cos_sine_identity (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) :
  ∀ t : ℝ, (complex.ofReal (cos t) - complex.I * complex.ofReal (sin t))^n = complex.ofReal (cos (n * t)) - complex.I * complex.ofReal (sin (n * t)) :=
sorry

end cos_sine_identity_l37_37459


namespace probability_of_positive_difference_ge_three_l37_37294

def select_three_numbers_probability : ℚ := 1 / 28

theorem probability_of_positive_difference_ge_three :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ (A : Finset ℕ), A ⊆ S ∧ A.card = 3 ∧ 
  (∀ (x y ∈ A), x ≠ y → abs (x - y) ≥ 3) → 
  fintype.card (set.subset_subfinset S {A | A.card = 3 ∧ ∀ (x y ∈ A), x ≠ y → abs (x - y) ≥ 3 }) /
  fintype.card (set.subset_subfinset S {A | A.card = 3}) = select_three_numbers_probability := by
  sorry

end probability_of_positive_difference_ge_three_l37_37294


namespace greatest_possible_lower_bound_of_sum_sq_l37_37032

-- Define the polynomial with given conditions
variables {n : ℕ} {a : ℕ → ℝ}

-- Assume the conditions given in the problem
def poly_monic (p : polynomial ℝ) : Prop :=
  p.coeff (p.nat_degree) = 1

def poly_degree_n (p : polynomial ℝ) (n : ℕ) : Prop :=
  p.nat_degree = n

def a_rel (a_(n_1) a_(n_2) : ℝ) : Prop :=
  a_(n_1) = 2 * a_(n_2)

noncomputable def sum_sq_roots (a_(n_1) a_(n_2) : ℝ) : ℝ :=
  4 * a_(n_2)^2 - 2 * a_(n_2)

-- Statement to be proved
theorem greatest_possible_lower_bound_of_sum_sq {p : polynomial ℝ} 
  (monic_p : poly_monic p) (deg_p : poly_degree_n p n) 
  (rel : a_rel (p.coeff (n-1)) (p.coeff (n-2))): 
  ∃ lb, lb = 1 / 4 ∧ 
  lb ≤ abs (sum_sq_roots (p.coeff (n-1)) (p.coeff (n-2))) :=
sorry

end greatest_possible_lower_bound_of_sum_sq_l37_37032


namespace man_l37_37361

-- Defining the conditions
def man's_speed_in_still_water := 12.5
def current_speed := 2.5
def man's_speed_against_current := 10

-- Theorem statement
theorem man's_speed_with_current :
  (man's_speed_in_still_water - current_speed = man's_speed_against_current) →
  (man's_speed_in_still_water + current_speed = 15) :=
by
  -- conditions
  intro h1,
  -- the statement follows from the conditions
  sorry

end man_l37_37361


namespace scaling_transformation_l37_37711

theorem scaling_transformation : 
  ∀ (P : ℝ × ℝ), P = (1, -2) →
  let P' := (2 * P.1, (1/2) * P.2) in 
  P' = (2, -1) := by 
  intros P hP
  let P' := (2 * P.1, (1/2) * P.2)
  rw hP
  simp
  sorry

end scaling_transformation_l37_37711


namespace A_work_rate_correct_A_compute_work_days_l37_37345

noncomputable def A_completion_days : ℝ := by sorry

def B_work_rate_per_day: ℝ := 1 / 14
def C_work_rate_per_day: ℝ := 1 / 16
def combined_work_rate: ℝ := 1 / 4.977777777777778

theorem A_work_rate_correct : (1 / A_completion_days) + B_work_rate_per_day + C_work_rate_per_day = combined_work_rate := by sorry

theorem A_compute_work_days : A_completion_days ≈ 5.027 := by sorry

end A_work_rate_correct_A_compute_work_days_l37_37345


namespace min_value_f_l37_37917

noncomputable def f (x a b c : ℝ) : ℝ :=
  max (abs (x^3 - a * x^2 - b * x - c))

theorem min_value_f : ∃ a b c : ℝ, 
  (∀ x ∈ set.Icc 1 3, f x a b c ≥ (1 / 4)) ∧
  ∃ a b c : ℝ, ∀ x ∈ set.Icc 1 3, f x a b c = (1 / 4) :=
sorry

end min_value_f_l37_37917


namespace train_crossing_bridge_time_l37_37730

theorem train_crossing_bridge_time (train_length bridge_length : ℕ) (train_speed_kmph : ℕ) :
  train_length = 100 →
  bridge_length = 150 →
  train_speed_kmph = 72 →
  let train_speed_mps := (train_speed_kmph * 1000) / 3600 in
  let total_distance := train_length + bridge_length in
  (total_distance : ℚ) / train_speed_mps = 12.5 :=
by
  intros h_train_length h_bridge_length h_train_speed_kmph
  let train_speed_mps := (train_speed_kmph * 1000) / 3600
  let total_distance := train_length + bridge_length
  sorry

end train_crossing_bridge_time_l37_37730


namespace parabola_directrix_l37_37876

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37876


namespace maximum_value_such_that_factorial_has_2013_trailing_zeros_l37_37307

def count_factors_of_five (n : ℕ) : ℕ :=
  let rec aux (n : ℕ) (acc : ℕ) (p : ℕ) : ℕ :=
    if p > n then acc
    else aux n (acc + n / p) (p * 5)
  in aux n 0 5

theorem maximum_value_such_that_factorial_has_2013_trailing_zeros :
  ∃ (N : ℕ), count_factors_of_five N = 2013 ∧ 
  ∀ (M : ℕ), count_factors_of_five M = 2013 → M ≤ N :=
begin
  use 8069,
  split,
  { 
    -- Proof that count_factors_of_five 8069 = 2013
    sorry 
  },
  { 
    -- Proof that for any M, if count_factors_of_five M = 2013 then M ≤ 8069
    sorry 
  }
end

end maximum_value_such_that_factorial_has_2013_trailing_zeros_l37_37307


namespace f_at_neg_one_l37_37672

def f (x : ℝ) : ℝ := 2^x + 3

theorem f_at_neg_one : f (-1) = 7 / 2 := 
by 
  sorry

end f_at_neg_one_l37_37672


namespace evaluate_expression_l37_37744

theorem evaluate_expression :
  -1^2020 + (∛(-27)) + (sqrt ((-2)^2)) + abs (sqrt 3 - 2) = - sqrt 3 :=
by 
  sorry

end evaluate_expression_l37_37744


namespace initially_planned_days_l37_37341

theorem initially_planned_days (D : ℕ) (h1 : 6 * 3 + 10 * 3 = 6 * D) : D = 8 := by
  sorry

end initially_planned_days_l37_37341


namespace find_divisor_l37_37305

theorem find_divisor : 
  ∀ (dividend quotient remainder divisor : ℕ), 
    dividend = 140 →
    quotient = 9 →
    remainder = 5 →
    dividend = (divisor * quotient) + remainder →
    divisor = 15 :=
by
  intros dividend quotient remainder divisor hd hq hr hdiv
  sorry

end find_divisor_l37_37305


namespace tangent_line_equation_l37_37696

noncomputable def curve (x : ℝ) : ℝ :=
  Real.log x + x + 1

theorem tangent_line_equation : ∃ x y : ℝ, derivative curve x = 2 ∧ curve x = y ∧ y = 2 * x := 
begin
  sorry
end

end tangent_line_equation_l37_37696


namespace average_speed_last_segment_l37_37645

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time_minutes : ℕ)
  (avg_speed_first_segment : ℕ)
  (avg_speed_second_segment : ℕ)
  (expected_avg_speed_last_segment : ℕ) :
  total_distance = 96 →
  total_time_minutes = 90 →
  avg_speed_first_segment = 60 →
  avg_speed_second_segment = 65 →
  expected_avg_speed_last_segment = 67 →
  (3 * (avg_speed_first_segment + avg_speed_second_segment + expected_avg_speed_last_segment) = (total_distance * 2)) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have total_time_hours := 1.5
  have overall_avg_speed := 96 / 1.5
  have overall_avg_speed_value : overall_avg_speed = 64 := by linarith [total_time_hours]
  have avg_calc : (60 + 65 + 67) / 3 = 64 := by linarith
  sorry

end average_speed_last_segment_l37_37645


namespace roses_after_trading_equals_36_l37_37208

-- Definitions of the given conditions
def initial_roses_given : ℕ := 24
def roses_after_trade (n : ℕ) : ℕ := n
def remaining_roses_after_first_wilt (roses : ℕ) : ℕ := roses / 2
def remaining_roses_after_second_wilt (roses : ℕ) : ℕ := roses / 2
def roses_remaining_second_day : ℕ := 9

-- The statement we want to prove
theorem roses_after_trading_equals_36 (n : ℕ) (h : roses_remaining_second_day = 9) :
  ( ∃ x, roses_after_trade x = n ∧ remaining_roses_after_first_wilt (remaining_roses_after_first_wilt x) = roses_remaining_second_day ) →
  n = 36 :=
by
  sorry

end roses_after_trading_equals_36_l37_37208


namespace solve_fraction_eq_l37_37454

theorem solve_fraction_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : (1 / (x - 1) = 3 / (x - 3)) ↔ x = 0 :=
by {
  sorry
}

end solve_fraction_eq_l37_37454


namespace find_altitude_l37_37568

-- Assumptions: the conditions of the problem
variables (A B : ℝ)
axiom sin_A_plus_B : sin (A + B) = 3 / 5
axiom sin_A_minus_B : sin (A - B) = 1 / 5
constant AB : ℝ := 3

-- Definition and theorem to prove
noncomputable def altitude_C_to_AB : ℝ :=
  AB * (2 + real.sqrt 6)

theorem find_altitude :
  altitude_C_to_AB = 6 + 3 * real.sqrt 6 :=
sorry

end find_altitude_l37_37568


namespace length_of_FD_l37_37578

def square_side : ℝ := 10
def point_E_distance_from_A : ℝ := square_side / 3
def point_E_distance_from_D : ℝ := square_side - point_E_distance_from_A

theorem length_of_FD :
  ∃ (x : ℝ), (10 - x)^2 = x^2 + (point_E_distance_from_D)^2 ∧ x = 25 / 9 :=
by
  sorry

end length_of_FD_l37_37578


namespace f_2008_equals_3_l37_37076

theorem f_2008_equals_3
  (a b α β : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : α ≠ 0) (h4 : β ≠ 0)
  (h_f : ∀ x : ℝ, f x = a * sin (π * x + α) + b * cos (π * x + β) + 4)
  (h_f1988 : f 1988 = 3) : f 2008 = 3 :=
sorry

end f_2008_equals_3_l37_37076


namespace infinite_product_converges_to_zero_l37_37378

def sequence_a : ℕ → ℝ
| 0    := 1 / 3
| (n+1) := 2 - (sequence_a n - 2)^3

theorem infinite_product_converges_to_zero :
  (∀ a: (ℕ → ℝ),  a = sequence_a → (∏ i, a i) = 0) :=
sorry

end infinite_product_converges_to_zero_l37_37378


namespace rectangular_prism_volume_l37_37553

theorem rectangular_prism_volume :
  ∀ (l w h : ℕ), 
  l = 2 * w → 
  w = 2 * h → 
  4 * (l + w + h) = 56 → 
  l * w * h = 64 := 
by
  intros l w h h_l_eq_2w h_w_eq_2h h_edge_len_eq_56
  sorry -- proof not provided

end rectangular_prism_volume_l37_37553


namespace find_sample_size_l37_37815

theorem find_sample_size
  (teachers : ℕ := 200)
  (male_students : ℕ := 1200)
  (female_students : ℕ := 1000)
  (sampled_females : ℕ := 80)
  (total_people := teachers + male_students + female_students)
  (ratio : sampled_females / female_students = n / total_people)
  : n = 192 := 
by
  sorry

end find_sample_size_l37_37815


namespace evaluate_s_squared_plus_c_squared_l37_37616

variable {x y : ℝ}

theorem evaluate_s_squared_plus_c_squared (r : ℝ) (h_r_def : r = Real.sqrt (x^2 + y^2))
                                          (s : ℝ) (h_s_def : s = y / r)
                                          (c : ℝ) (h_c_def : c = x / r) :
  s^2 + c^2 = 1 :=
sorry

end evaluate_s_squared_plus_c_squared_l37_37616


namespace ant_probability_on_cubic_grid_l37_37397

-- Representing the problem in Lean 4 statement
theorem ant_probability_on_cubic_grid :
  ∀ (A B : ℤ × ℤ × ℤ), 
  A = (0, 0, 0) → 
  B = (0, 0, 1) → 
  (∃ n : ℕ, n = 4) →
  (∃ p : ℚ, p = 1 / 1296) →
  ∃ p : ℚ, p = probability_ant_at_point A B 4 :=
sorry

end ant_probability_on_cubic_grid_l37_37397


namespace least_negative_b_l37_37267

theorem least_negative_b (x b : ℤ) (h1 : x^2 + b * x = 22) (h2 : b < 0) : b = -21 :=
sorry

end least_negative_b_l37_37267


namespace min_value_x_y_l37_37623

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 :=
by 
  sorry

end min_value_x_y_l37_37623


namespace tan_cot_solution_count_l37_37061

theorem tan_cot_solution_count :
  ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → 
  (∃! θ, ∀ θ, tan(3 * π * cos θ) = cot(3 * π * sin θ)) ∧ 16 :=
by
  sorry

end tan_cot_solution_count_l37_37061


namespace part1_part2_part3_l37_37317

-- Part 1: There exists a real number a such that a + 1/a ≤ 2
theorem part1 : ∃ a : ℝ, a + 1/a ≤ 2 := sorry

-- Part 2: For all positive real numbers a and b, b/a + a/b ≥ 2
theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : b / a + a / b ≥ 2 := sorry

-- Part 3: For positive real numbers x and y such that x + 2y = 1, then 2/x + 1/y ≥ 8
theorem part3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 2 / x + 1 / y ≥ 8 := sorry

end part1_part2_part3_l37_37317


namespace base_form_exists_l37_37587

-- Definitions for three-digit number and its reverse in base g
def N (a b c g : ℕ) : ℕ := a * g^2 + b * g + c
def N_reverse (a b c g : ℕ) : ℕ := c * g^2 + b * g + a

-- The problem statement in Lean
theorem base_form_exists (a b c g : ℕ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 < g)
    (h₅ : N a b c g = 2 * N_reverse a b c g) : ∃ k : ℕ, g = 3 * k + 2 ∧ k > 0 :=
by
  sorry

end base_form_exists_l37_37587


namespace meaningful_range_l37_37159

   noncomputable def isMeaningful (x : ℝ) : Prop :=
     (3 - x ≥ 0) ∧ (x + 1 ≠ 0)

   theorem meaningful_range :
     ∀ x : ℝ, isMeaningful x ↔ (x ≤ 3 ∧ x ≠ -1) :=
   by
     sorry
   
end meaningful_range_l37_37159


namespace sum_of_roots_quadratic_eq_l37_37853

theorem sum_of_roots_quadratic_eq (a b c : ℝ) (h_eq : a = 1) (h_b : b = -6) (h_c : c = 8) :
  let s := ( -b / a) in
  s = 6 :=
by
  sorry

end sum_of_roots_quadratic_eq_l37_37853


namespace license_plate_count_correct_l37_37658

def rotokas_letters : Finset Char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U'}

def valid_license_plate_count : ℕ :=
  let first_letter_choices := 2 -- Letters A or E
  let last_letter_fixed := 1 -- Fixed as P
  let remaining_letters := rotokas_letters.erase 'V' -- Exclude V
  let second_letter_choices := (remaining_letters.erase 'P').card - 1 -- Exclude P and first letter
  let third_letter_choices := second_letter_choices - 1
  let fourth_letter_choices := third_letter_choices - 1
  2 * 9 * 8 * 7

theorem license_plate_count_correct :
  valid_license_plate_count = 1008 := by
  sorry

end license_plate_count_correct_l37_37658


namespace driving_hours_requirements_l37_37590

theorem driving_hours_requirements
  (trip_time : ℕ) (driving_days : ℕ) (trips_per_day : ℕ) (minutes_per_hour : ℕ)
  (trip_time_eq : trip_time = 20)
  (driving_days_eq : driving_days = 75)
  (trips_per_day_eq : trips_per_day = 2)
  (minutes_per_hour_eq : minutes_per_hour = 60) :
  (2 * trips_per_day * trip_time / minutes_per_hour) * driving_days = 50 :=
by {
  rw [trip_time_eq, driving_days_eq, trips_per_day_eq, minutes_per_hour_eq],
  norm_num,
  refl,
} 

end driving_hours_requirements_l37_37590


namespace find_x_plus_y_l37_37112

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 1003 :=
sorry

end find_x_plus_y_l37_37112


namespace no_int_solutions_l37_37642

theorem no_int_solutions (c x y : ℤ) (h1 : 0 < c) (h2 : c % 2 = 1) : x ^ 2 - y ^ 3 ≠ (2 * c) ^ 3 - 1 :=
sorry

end no_int_solutions_l37_37642


namespace weekly_loss_due_to_production_shortfall_l37_37592

-- Defining the conditions
def daily_production : ℕ := 1000
def cost_per_tire : ℕ := 250
def selling_price_factor : ℕ := 3 / 2
def potential_daily_sales : ℕ := 1200

-- Defining the problem
theorem weekly_loss_due_to_production_shortfall :
  (let selling_price_per_tire := cost_per_tire * selling_price_factor in
   let profit_per_tire := selling_price_per_tire - cost_per_tire in
   let daily_missing_tires := potential_daily_sales - daily_production in
   let daily_loss := daily_missing_tires * profit_per_tire in
   let weekly_loss := daily_loss * 7 in
   weekly_loss = 175000) := 
sorry

end weekly_loss_due_to_production_shortfall_l37_37592


namespace intersect_planes_parallel_int_line_to_l_l37_37510

-- Definitions of the conditions
variables {m n l : Line} {α β : Plane}

-- Assuming the conditions
axiom skew_lines : m ≠ n ∧ ¬ ∃ p q : Point, m.contains p ∧ m.contains q ∧ n.contains p ∧ n.contains q
axiom perp_m_alpha : m ⊥ α
axiom perp_n_beta : n ⊥ β
axiom line_l_conditions : (l ⊥ m) ∧ (l ⊥ n) ∧ (¬ α.contains l) ∧ (¬ β.contains l)

-- The theorem statement based on the correct answer (option B)
theorem intersect_planes_parallel_int_line_to_l : (∃ (line_intersection : Line), α ∩ β = some line_intersection ∧ line_intersection ∥ l) :=
sorry

end intersect_planes_parallel_int_line_to_l_l37_37510


namespace cost_of_3000_pencils_l37_37344

theorem cost_of_3000_pencils (pencils_per_box : ℕ) (cost_per_box : ℝ) (pencils_needed : ℕ) (unit_cost : ℝ): 
  pencils_per_box = 120 → cost_per_box = 36 → pencils_needed = 3000 → unit_cost = 0.30 →
  (pencils_needed * unit_cost = (3000 : ℝ) * 0.30) :=
by
  intros _ _ _ _
  sorry

end cost_of_3000_pencils_l37_37344


namespace factorization_correct_l37_37030

def factor_expression (x : ℝ) : ℝ :=
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x)

theorem factorization_correct (x : ℝ) : 
  factor_expression x = 3 * x * (5 * x^3 - 7 * x^2 + 12) :=
by
  sorry

end factorization_correct_l37_37030


namespace problem1_problem2_l37_37047

theorem problem1 : log 2 (sqrt 2) + log 9 27 + 3 ^ log 3 16 = 18 := 
by 
-- We use sorry to indicate that the proof is not provided.
sorry

theorem problem2 : 0.25 ^ (-2) + (8 / 27) ^ (-1 / 3) - (1 / 2) * log 10 16 - 2 * log 10 5 + (1 / 2) ^ 0 = 33 / 2 := 
by 
-- We use sorry to indicate that the proof is not provided.
sorry

end problem1_problem2_l37_37047


namespace intersection_M_N_l37_37531

def M (x : ℝ) : Prop := Real.log x / Real.log 2 ≥ 0
def N (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x | N x} = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l37_37531


namespace highest_y_coordinate_l37_37668

theorem highest_y_coordinate (x y : ℝ) (h : (x^2 / 49 + (y-3)^2 / 25 = 0)) : y = 3 :=
by
  sorry

end highest_y_coordinate_l37_37668


namespace range_of_a_l37_37986

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l37_37986


namespace right_triangle_area_is_integer_l37_37676

theorem right_triangle_area_is_integer (a b : ℕ) (h1 : ∃ (A : ℕ), A = (1 / 2 : ℚ) * ↑a * ↑b) : (a % 2 = 0) ∨ (b % 2 = 0) :=
sorry

end right_triangle_area_is_integer_l37_37676


namespace find_f_minus_2016_l37_37944

def f : ℝ → ℝ
| x := if x > 0 then log x / log 2 + 2017 else -f (x + 2)

theorem find_f_minus_2016 : f (-2016) = -2018 :=
by {
  /- The detailed proof is omitted, but we assume here this part would correctly validate -2018 as the result -/
  sorry
}

end find_f_minus_2016_l37_37944


namespace draw_4_balls_ordered_l37_37755

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37755


namespace parabola_directrix_is_x_eq_1_l37_37893

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37893


namespace log_base_32_of_16_is_4_over_5_l37_37274

theorem log_base_32_of_16_is_4_over_5 : ∃ y : ℝ, log 32 16 = y ∧ y = 4 / 5 :=
by
  use 4 / 5
  split
  {
    apply log_of_elements 32 16 4 / 5
    sorry
  }
  {
    refl
  }

end log_base_32_of_16_is_4_over_5_l37_37274


namespace number_subtracted_l37_37979

theorem number_subtracted (t k x : ℝ) (h1 : t = (5 / 9) * (k - x)) (h2 : t = 105) (h3 : k = 221) : x = 32 :=
by
  sorry

end number_subtracted_l37_37979


namespace second_next_perfect_square_l37_37062

noncomputable def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, x = k^2

theorem second_next_perfect_square (x : ℕ) (hx : is_perfect_square x) :
  ∃ y : ℕ, y = x + 4 * nat.sqrt x + 4 :=
begin
  sorry
end

end second_next_perfect_square_l37_37062


namespace goshawk_eurasian_reserve_l37_37563

theorem goshawk_eurasian_reserve (B : ℕ) (hB: B = 100):
  let hawks := 30,
      non_hawks := B - hawks,
      paddyfield_warblers := 0.4 * non_hawks,
      kingfishers := 0.25 * paddyfield_warblers,
      white_storks := kingfishers + 0.1 * kingfishers,
      remaining_birds := B - (hawks + paddyfield_warblers + kingfishers + white_storks)
  in remaining_birds / B * 100 = 27 :=
by
  let hawks := 30,
      non_hawks := B - hawks,
      paddyfield_warblers := 0.4 * non_hawks,
      kingfishers := 0.25 * paddyfield_warblers,
      white_storks := kingfishers + 0.1 * kingfishers,
      remaining_birds := B - (hawks + paddyfield_warblers + kingfishers + white_storks)
  have h1 : B = 100 := hB
  sorry

end goshawk_eurasian_reserve_l37_37563


namespace draw_4_balls_in_order_ways_l37_37770

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37770


namespace number_of_new_books_l37_37023

-- Defining the given conditions
def adventure_books : ℕ := 24
def mystery_books : ℕ := 37
def used_books : ℕ := 18

-- Defining the total books and new books
def total_books : ℕ := adventure_books + mystery_books
def new_books : ℕ := total_books - used_books

-- Proving the number of new books
theorem number_of_new_books : new_books = 43 := by
  -- Here we need to show that the calculated number of new books equals 43
  sorry

end number_of_new_books_l37_37023


namespace corresponding_angles_not_always_equal_l37_37240

theorem corresponding_angles_not_always_equal 
  (A B : Type) [Geometry] (angle1 angle2 : Angle A) 
  (h : CorrespondingAngles angle1 angle2) : 
  ¬ (∀ a1 a2 : Angle A, CorrespondingAngles a1 a2 → a1 = a2) :=
sorry

end corresponding_angles_not_always_equal_l37_37240


namespace cos_sin_identity_l37_37461

theorem cos_sin_identity (n : ℕ) (t : ℝ) (hn : 1 ≤ n ∧ n ≤ 500) :
  (complex.cos t - complex.i * complex.sin t) ^ n = complex.cos (n * t) - complex.i * complex.sin (n * t) := 
sorry

end cos_sin_identity_l37_37461


namespace regression_line_decrease_l37_37980

theorem regression_line_decrease :
  ∀ (x : ℝ), (let y1 := 2 - 1.5 * x,
                  y2 := 2 - 1.5 * (x + 1) in
                y2 - y1 = -1.5) := by
  intro x
  let y1 := 2 - 1.5 * x
  let y2 := 2 - 1.5 * (x + 1)
  exact calc
    y2 - y1 = (2 - 1.5 * (x + 1)) - (2 - 1.5 * x) : by rfl
         ... = (2 - 1.5 * x - 1.5) - (2 - 1.5 * x) : by rw mul_add
         ... = -1.5 : by ring

end regression_line_decrease_l37_37980


namespace isosceles_triangle_possible_x_l37_37283

namespace IsoscelesTriangleProof

-- Define the conditions and the problem
theorem isosceles_triangle_possible_x (x : ℝ) : 
  (0 < x ∧ x < 90) → -- x is acute
  (sin x = sin x ∧ sin x = sin 9 * x ∧ angle = 3 * x) → -- side and vertex angle conditions
  (x = 30) := 
by
  sorry

end IsoscelesTriangleProof

end isosceles_triangle_possible_x_l37_37283


namespace sequence_formulas_T_formula_find_m_l37_37923

def seq_sum (a : ℕ → ℕ) (n : ℕ) : ℕ := (finset.range n).sum a -- S_n is the sum of first n terms

def a_seq (n : ℕ) : ℕ := n + 5
def b_seq : ℕ → ℕ
| 1 := sorry
| 2 := sorry
| n := 3 * n + 2   -- derived that b_n = 3n + 2
  
def sum_b : ℕ := (finset.range 9).sum b_seq

theorem sequence_formulas :
  (∀ n, seq_sum a_seq n = n*(n+11)/2) ∧
  (∀ n, b_seq n = 3*n + 2) ∧
  (sum_b = 153) :=
begin
  split,
  { sorry },
  { sorry },
  { sorry },
end

def T : ℕ → ℕ
| n := 32 * (n-1) * 2^(n+1) + 64   -- T_n for the series (a_n - 5) * 2^(a_n) = 32n * 2^n

theorem T_formula (n : ℕ) : T n = 32*(n-1)*2^(n+1) + 64 :=
sorry

def f : ℕ → ℕ
| n := if (n % 2 = 1) then a_seq n else b_seq n

theorem find_m (m : ℕ) :
  (f (m+15) = 5*f m) → m = 11 :=
begin
  sorry,
end

end sequence_formulas_T_formula_find_m_l37_37923


namespace age_difference_l37_37336

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l37_37336


namespace x_squared_minus_y_squared_l37_37975

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9 / 13) (h2 : x - y = 5 / 13) : x^2 - y^2 = 45 / 169 := 
by 
  -- proof omitted 
  sorry

end x_squared_minus_y_squared_l37_37975


namespace eccentricity_of_hyperbola_l37_37667

def a : ℝ := 3
def b : ℝ := 2
def c : ℝ := real.sqrt (a^2 + b^2)
def e : ℝ := c / a

theorem eccentricity_of_hyperbola : e = real.sqrt 13 / 3 := by
  sorry

end eccentricity_of_hyperbola_l37_37667


namespace bill_annual_healthcare_cost_l37_37410

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end bill_annual_healthcare_cost_l37_37410


namespace cross_product_scaled_v_and_w_l37_37869

-- Assume the vectors and their scalar multiple
def v : ℝ × ℝ × ℝ := (3, 1, 4)
def w : ℝ × ℝ × ℝ := (-2, 2, -3)
def v_scaled : ℝ × ℝ × ℝ := (6, 2, 8)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.1 * b.2.2 - a.2.2 * b.1,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_scaled_v_and_w :
  cross_product v_scaled w = (-22, -2, 16) :=
by
  sorry

end cross_product_scaled_v_and_w_l37_37869


namespace sequence_a1_sequence_general_sequence_b_sum_l37_37611

theorem sequence_a1 (S1 : ℕ) (hS1 : 2 * S1^2 - 4 * S1 - 6 = 0) : S1 = 3 := sorry

theorem sequence_general (a : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, 2 * (∑ i in finset.range n, a i)^2 - 
                     (3 * n^2 + 3 * n - 2) * (∑ i in finset.range n, a i) - 
                     3 * (n^2 + n) = 0) :
  ∀ n : ℕ, n > 0 → a n = 3 * n := sorry

theorem sequence_b_sum (a : ℕ → ℕ) (b n : ℕ → ℝ) (h_bn : ∀ n : ℕ, b n = (a n) / (3^(n+1))) :
  ∑ i in finset.range n, b i = (3 / 4) - ((2 * n + 3) / (4 * (3^n))) := sorry

end sequence_a1_sequence_general_sequence_b_sum_l37_37611


namespace find_initial_cards_l37_37644

noncomputable def initial_cards (x : ℕ) : Prop := x + 41 + 20 = 88

theorem find_initial_cards : ∃ x : ℕ, initial_cards x :=
by
  exists 27
  show initial_cards 27
  sorry

end find_initial_cards_l37_37644


namespace A_union_B_l37_37103

noncomputable def A : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - 2^x) ∧ x < 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2 ∧ x > 0}
noncomputable def union_set : Set ℝ := {x | x < 0 ∨ x > 0}

theorem A_union_B :
  A ∪ B = union_set :=
by
  sorry

end A_union_B_l37_37103


namespace congruence_is_sufficient_but_not_necessary_for_equal_area_l37_37930

-- Definition of conditions
def Congruent (Δ1 Δ2 : Type) : Prop := sorry -- Definition of congruent triangles
def EqualArea (Δ1 Δ2 : Type) : Prop := sorry -- Definition of triangles with equal area

-- Theorem statement
theorem congruence_is_sufficient_but_not_necessary_for_equal_area 
  (Δ1 Δ2 : Type) :
  (Congruent Δ1 Δ2 → EqualArea Δ1 Δ2) ∧ (¬ (EqualArea Δ1 Δ2 → Congruent Δ1 Δ2)) :=
sorry

end congruence_is_sufficient_but_not_necessary_for_equal_area_l37_37930


namespace coin_flip_probability_l37_37258

/-- Calculates the probability that the penny, nickel, and half dollar all come up heads when flipping five coins simultaneously. -/
theorem coin_flip_probability :
  (let total_outcomes := 2^5 in
   let successful_outcomes := 1 * 1 * 1 * 2 * 2 in
   successful_outcomes / total_outcomes = 1 / 8) :=
by {
  let total_outcomes := 2^5,
  let successful_outcomes := 1 * 1 * 1 * 2 * 2,
  exact (div_eq_iff (by norm_num : total_outcomes ≠ 0)).mpr rfl
}

end coin_flip_probability_l37_37258


namespace find_a4_l37_37141

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 2 (fun k a_k => a_k + k)

theorem find_a4 : sequence 4 = 8 :=
by
  sorry

end find_a4_l37_37141


namespace nonzero_terms_count_l37_37544

noncomputable def expand_poly (x : ℝ) : ℝ :=
  (x^2 + 5) * (3*x^3 + 2*x^2 + 6) - 4*(x^4 - 3*x^3 + 8*x^2 + 1) + 2*x^3

-- Define the final polynomial for comparison.
noncomputable def final_poly (x : ℝ) : ℝ :=
  3*x^5 - 2*x^4 + 20*x^3 + 8*x^2 + 29

-- Statement of the theorem
theorem nonzero_terms_count :
  ∀ x : ℝ, expand_poly x = final_poly x ∧
           (3*x^5 ≠ 0 ∧ -2*x^4 ≠ 0 ∧ 20*x^3 ≠ 0 ∧ 8*x^2 ≠ 0 ∧ 29 ≠ 0) :=
begin
  intros,
  sorry
end

end nonzero_terms_count_l37_37544


namespace range_of_a_l37_37982

theorem range_of_a (a : ℝ) (e : ℝ) : (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
  2 * e^2 - a * x₁^2 + (a - 2 * e) * x₁ = 0 ∧ 
  2 * e^2 - a * x₂^2 + (a - 2 * e) * x₂ = 0 ∧ 
  2 * e^2 - a * x₃^2 + (a - 2 * e) * x₃ = 0) ↔ a ∈ set.Ioi 0 :=
by
  sorry

end range_of_a_l37_37982


namespace correct_operation_l37_37316

theorem correct_operation (a : ℝ) :
  (2 * a^2) * a = 2 * a^3 :=
by sorry

end correct_operation_l37_37316


namespace draw_4_balls_in_order_ways_l37_37769

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37769


namespace correct_quotient_l37_37991

theorem correct_quotient (x : ℕ) (hx1: x = 12 * 63)
                          (hx2 : x = 18 * 112) 
                          (hx3 : x = 24 * 84)
                          (hdiv21 : x % 21 = 0)
                          (hdiv27 : x % 27 = 0)
                          (hdiv36 : x % 36 = 0) :
    x / 21 = 96 :=
by sorry

end correct_quotient_l37_37991


namespace general_solution_l37_37901

noncomputable def solve_diff_eq (x : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) :=
  ∀ (c : ℝ), x * y' x - (y x / real.log x) = 0 → y x = c * real.log x

theorem general_solution (x : ℝ) (y : ℝ → ℝ) (y' : ℝ → ℝ) : solve_diff_eq x y y' :=
by
  sorry

end general_solution_l37_37901


namespace sum_of_factors_eq_12_l37_37700

-- Define the polynomial for n = 1
def poly (x : ℤ) : ℤ := x^5 + x + 1

-- Define the two factors when x = 2
def factor1 (x : ℤ) : ℤ := x^3 - x^2 + 1
def factor2 (x : ℤ) : ℤ := x^2 + x + 1

-- State the sum of the two factors at x = 2 equals 12
theorem sum_of_factors_eq_12 (x : ℤ) (h : x = 2) : factor1 x + factor2 x = 12 :=
by {
  sorry
}

end sum_of_factors_eq_12_l37_37700


namespace greatest_integer_less_AD_eq_149_l37_37572

theorem greatest_integer_less_AD_eq_149
  (A B C D E : ℝ → ℝ → Prop)
  (rectangle : is_rectangle A B C D)
  (AB_eq_150 : distance A B = 150)
  (E_mid_AD : midpoint A D E)
  (perpendicular_AC_BE : is_perpendicular (line_through A C) (line_through B E))
  (isosceles_BCD : is_isosceles B C D) :
  greatest_integer_less_than (distance A D) = 149 :=
begin
  sorry
end

end greatest_integer_less_AD_eq_149_l37_37572


namespace chocolate_ratio_l37_37227

theorem chocolate_ratio (N A : ℕ) (h1 : N = 10) (h2 : A - 5 = N + 15) : A / N = 3 :=
by {
  sorry
}

end chocolate_ratio_l37_37227


namespace inequality_system_solution_set_l37_37286

theorem inequality_system_solution_set (m : ℝ) : 
  (∀ x : ℝ, 3 * x - 9 > 0 → x > m → x > 3) → m ≤ 3 :=
by
  intro h
  specialize h 3.1
  have h1 : 3 * 3.1 - 9 > 0 := by norm_num
  have h2 : 3.1 > m := by linarith
  specialize h h1 h2
  linarith

end inequality_system_solution_set_l37_37286


namespace ariana_total_owe_l37_37404

-- Definitions based on the conditions
def first_bill_principal : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_overdue_months : ℕ := 2

def second_bill_principal : ℕ := 130
def second_bill_late_fee : ℕ := 50
def second_bill_overdue_months : ℕ := 6

def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80

-- Theorem
theorem ariana_total_owe : 
  first_bill_principal + 
    (first_bill_principal : ℝ) * first_bill_interest_rate * (first_bill_overdue_months : ℝ) +
    second_bill_principal + 
    second_bill_late_fee * second_bill_overdue_months + 
    third_bill_first_month_fee + 
    third_bill_second_month_fee = 790 := 
by 
  sorry

end ariana_total_owe_l37_37404


namespace domain_of_composite_function_l37_37950

theorem domain_of_composite_function :
  ∀ (f : ℝ → ℝ), (∀ x, -1 ≤ x ∧ x ≤ 3 → ∃ y, f x = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f (2*x - 1) = y) :=
by
  intros f domain_f x hx
  sorry

end domain_of_composite_function_l37_37950


namespace max_solution_inequality_l37_37443

def t (x : ℝ) : ℝ :=
  80 - 2 * x * real.sqrt (30 - 2 * x)

noncomputable def inequality_expr (x : ℝ) : ℝ :=
  let numerator := -real.log 3 (t x)^2 + abs (real.log 3 (t x) - 3 * real.log 3 ((x^2 - 2*x + 29)^3))
  let denominator := 7 * real.log 7 (65 - 2 * x * real.sqrt (30 - 2 * x)) - 4 * real.log 3 (t x)
  numerator / denominator

theorem max_solution_inequality : ∃ x, (inequality_expr x ≥ 0) ∧ (x = 8 - real.sqrt 13) :=
by
  sorry

end max_solution_inequality_l37_37443


namespace find_first_discount_percentage_l37_37685

theorem find_first_discount_percentage :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 298 * (0.85 - x / 100 * 0.85) = 222.904 :=
begin
  use 12,  -- We know x ∼ 12 from the solution
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end find_first_discount_percentage_l37_37685


namespace slope_angle_of_tangent_line_l37_37284

theorem slope_angle_of_tangent_line :
  (∃ (x : ℝ), x = 1 ∧ y = x^3 - 2*x + 4) ∧ (∂y/∂x | x = 1 = 1) ∧ (angle = 45°) :=
by
  sorry

end slope_angle_of_tangent_line_l37_37284


namespace max_value_neg1_interval_l37_37118

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f (x)

def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 4 then x^2 - 4*x + 5 else
  if -4 ≤ x ∧ x ≤ -1 then (-x)^2 - 4*(-x) + 5 else 0

theorem max_value_neg1_interval :
  (odd_function f) →
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = x^2 - 4*x + 5) →
  ∃ x, -4 ≤ x ∧ x ≤ -1 ∧ ∀ y, -4 ≤ y ∧ y ≤ -1 → f y ≤ f x ∧ f x = -1 :=
by
  sorry

end max_value_neg1_interval_l37_37118


namespace Ned_not_washed_shirts_l37_37226

theorem Ned_not_washed_shirts 
  (short_sleeve_shirts : ℕ) 
  (long_sleeve_shirts : ℕ) 
  (washed_shirts : ℕ) 
  (total_shirts := short_sleeve_shirts + long_sleeve_shirts) :
  (short_sleeve_shirts = 9) → 
  (long_sleeve_shirts = 21) → 
  (washed_shirts = 29) → 
  (total_shirts - washed_shirts = 1) :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  show (9 + 21 - 29 = 1)
  rw [add_comm 9 21]
  rw [add_sub_assoc 21 9 29]
  rw [add_comm 29 (21 - 29)]
  show (9 + 0 = 1)
  rw [zero_add]
  sorry

end Ned_not_washed_shirts_l37_37226


namespace function_form_for_condition_l37_37503

theorem function_form_for_condition (k : ℕ) (l : ℕ) (f : ℕ → ℕ) :
  (∀ m n : ℕ, (f(m) + f(n)) ∣ (m + n + l) ^ k) → 
  (∃ c : ℕ, l = 2 * c ∧ ∀ n, f(n) = n + c) ∨
  (∃ odd_l : l % 2 = 1, ∀ f : ℕ → ℕ, ¬ (∀ m n : ℕ, (f(m) + f(n)) ∣ (m + n + l) ^ k)) :=
by
  sorry

end function_form_for_condition_l37_37503


namespace trapezoid_shaded_fraction_l37_37387

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end trapezoid_shaded_fraction_l37_37387


namespace solve_inequality_l37_37436

theorem solve_inequality (x : ℝ) :
  (x ≠ 3) → (x * (x + 2) / (x - 3)^2 ≥ 8) ↔ (x ∈ set.Iic (18/7) ∪ set.Ioi 4) :=
by
  sorry

end solve_inequality_l37_37436


namespace problem1_problem2_l37_37958

def A := { x : ℝ | -2 < x ∧ x ≤ 4 }
def B := { x : ℝ | 2 - x < 1 }
def U := ℝ
def complement_B := { x : ℝ | x ≤ 1 }

theorem problem1 : { x : ℝ | 1 < x ∧ x ≤ 4 } = { x : ℝ | x ∈ A ∧ x ∈ B } := 
by sorry

theorem problem2 : { x : ℝ | x ≤ 4 } = { x : ℝ | x ∈ A ∨ x ∈ complement_B } := 
by sorry

end problem1_problem2_l37_37958


namespace clea_escalator_time_standing_l37_37840

noncomputable def escalator_time (c : ℕ) : ℝ :=
  let s := (7 * c) / 5
  let d := 72 * c
  let t := d / s
  t

theorem clea_escalator_time_standing (c : ℕ) (h1 : 72 * c = 72 * c) (h2 : 30 * (c + (7 * c) / 5) = 72 * c): escalator_time c = 51 :=
by
  sorry

end clea_escalator_time_standing_l37_37840


namespace option_d_true_l37_37069

-- Define the conditions
variables (a b : ℝ)
-- Assume a > b
variables (hab : a > b)

-- State the theorem to be proved
theorem option_d_true : a ^ 2023 > b ^ 2023 :=
begin
  sorry
end

end option_d_true_l37_37069


namespace worker_savings_multiple_l37_37391

theorem worker_savings_multiple 
  (P : ℝ)
  (P_gt_zero : P > 0)
  (save_fraction : ℝ := 1/3)
  (not_saved_fraction : ℝ := 2/3)
  (total_saved : ℝ := 12 * (save_fraction * P)) :
  ∃ multiple : ℝ, total_saved = multiple * (not_saved_fraction * P) ∧ multiple = 6 := 
by 
  sorry

end worker_savings_multiple_l37_37391


namespace min_questions_l37_37997

theorem min_questions (people : Finset ℕ) (questions : Finset ℕ)
    (answered : ∀ q ∈ questions, ∃ p ⊆ people, p.card = 5 ∧ ∀ x ∈ p, answers x q)
    (not_answered_by_four : ∀ q ∈ questions, ∀ p ⊆ people, p.card = 4 → ¬ ∀ x ∈ p, answers x q) :
  questions.card = 210 := 
sorry

end min_questions_l37_37997


namespace drawn_from_grade12_correct_l37_37799

-- Variables for the conditions
variable (total_students : ℕ) (sample_size : ℕ) (grade10_students : ℕ) 
          (grade11_students : ℕ) (grade12_students : ℕ) (drawn_from_grade12 : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 2400 ∧
  sample_size = 120 ∧
  grade10_students = 820 ∧
  grade11_students = 780 ∧
  grade12_students = total_students - grade10_students - grade11_students ∧
  drawn_from_grade12 = (grade12_students * sample_size) / total_students

-- Theorem to prove
theorem drawn_from_grade12_correct : conditions total_students sample_size grade10_students grade11_students grade12_students drawn_from_grade12 → drawn_from_grade12 = 40 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end drawn_from_grade12_correct_l37_37799


namespace relationship_abc_l37_37933

theorem relationship_abc (a b c : ℝ)
  (ha : a = 0.6 ^ 0.6)
  (hb : b = 0.6 ^ 1.5)
  (hc : c = 1.5 ^ 0.6) :
  b < a ∧ a < c :=
  sorry

end relationship_abc_l37_37933


namespace perpendicular_vecs_l37_37538

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

end perpendicular_vecs_l37_37538


namespace alice_initial_quarters_l37_37825

def quarters_initial (total_value_dollars : ℝ) (iron_nickel_percent : ℝ) (iron_nickel_value : ℝ) (regular_nickel_value : ℝ) : ℝ :=
  let N := total_value_dollars / (iron_nickel_percent * iron_nickel_value + (1 - iron_nickel_percent) * regular_nickel_value)
  N / 5

theorem alice_initial_quarters :
  quarters_initial 64 0.20 3 0.05 = 20 := 
by {
  -- Proof here, but we are not providing it as per instructions
  sorry
}

end alice_initial_quarters_l37_37825


namespace small_trucks_needed_l37_37044

-- Defining the problem's conditions
def total_flour : ℝ := 500
def large_truck_capacity : ℝ := 9.6
def num_large_trucks : ℝ := 40
def small_truck_capacity : ℝ := 4

-- Theorem statement to find the number of small trucks needed
theorem small_trucks_needed : (total_flour - (num_large_trucks * large_truck_capacity)) / small_truck_capacity = (500 - (40 * 9.6)) / 4 :=
by
  sorry

end small_trucks_needed_l37_37044


namespace count_rational_roots_l37_37424

theorem count_rational_roots : 
  let p_factors := [1, 2, 3, 6, 9, 18];
  let q_factors := [1, 2, 4];
  let possible_roots := (p_factors.product q_factors).map (λ (p_q : ℕ × ℕ), [p_q.1 / p_q.2, -(p_q.1 / p_q.2)]).join in
  (possible_roots.to_finset).card = 20 :=
by
  let p_factors := [1, 2, 3, 6, 9, 18]
  let q_factors := [1, 2, 4]
  let possible_roots := (p_factors.product q_factors).map (λ (p_q : ℕ × ℕ), [p_q.1 / p_q.2, -(p_q.1 / p_q.2)]).join
  have : (possible_roots.to_finset).card = 20 := sorry
  exact this

end count_rational_roots_l37_37424


namespace travel_time_reduction_l37_37187

theorem travel_time_reduction : 
  let t_initial := 19.5
  let factor_1998 := 1.30
  let factor_1999 := 1.25
  let factor_2000 := 1.20
  t_initial / factor_1998 / factor_1999 / factor_2000 = 10 := by
  sorry

end travel_time_reduction_l37_37187


namespace equilateral_triangle_area_correct_l37_37015

noncomputable def equilateral_triangle_area (s : ℕ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem equilateral_triangle_area_correct :
  ∀ (s : ℕ), (s = 29) → (equilateral_triangle_area s = 841 * sqrt 3 / 4) :=
by
  intros s hs
  -- Proof required here, but it is omitted.
  sorry

end equilateral_triangle_area_correct_l37_37015


namespace find_f_36_l37_37082

-- Define the function f with the given property
def f : ℕ → ℝ := sorry

-- Given conditions
axiom f_mul (a b : ℕ) : f (a * b) = f a + f b
axiom f_two : f 2 = p
axiom f_three : f 3 = q

-- The main theorem to prove
theorem find_f_36 (p q : ℝ) : f 36 = 2 * (p + q) :=
by
  -- The proof is omitted because it's not required
  sorry

end find_f_36_l37_37082


namespace intersection_sums_eq_5_2_l37_37673

noncomputable def findIntersectionSums : ℝ × ℝ :=
let f := λ x : ℝ, x^3 - 5*x^2 + 8*x - 4
let g := λ y : ℝ, (5 - y) / 5
let roots := { x | f x = g x }
let x1_based_sums := roots.to_finset.sum (λ x, x)
let y_based_sums := roots.to_finset.sum (λ x, (5 - x) / 5)
(x1_based_sums, y_based_sums)

theorem intersection_sums_eq_5_2 : findIntersectionSums = (5, 2) :=
sorry

end intersection_sums_eq_5_2_l37_37673


namespace problem1_problem2_problem3_problem4_l37_37921

section
variables (y : ℝ → ℝ) (m : ℝ) (x : ℝ)

-- Given a linear function y = (2m+1)x + m - 2
def linear_function := (2 * m + 1) * x + m - 2

-- Problem (1): If the graph of the function passes through the origin, find the value of m.
def passes_through_origin := linear_function y m 0 = 0

-- Problem (2): If the graph of the function has a y-intercept of -3, find the value of m.
def y_intercept := linear_function y m 0 = -3

-- Problem (3): If the graph of the function is parallel to the line y = x + 1, find the value of m.
def parallel_to_line := (2 * m + 1) = 1

-- Problem (4): If the graph of the function does not pass through the second quadrant, find the range of m.
def not_in_second_quadrant := (2 * m + 1 > 0) ∧ (m - 2 < 0)

-- Proof that m = 2 given the graph passes through the origin
theorem problem1 : passes_through_origin y m ↔ m = 2 := 
sorry

-- Proof that m = -1 given the graph has a y-intercept of -3
theorem problem2 : y_intercept y m ↔ m = -1 := 
sorry

-- Proof that m = 0 given the graph is parallel to y = x + 1
theorem problem3 : parallel_to_line m ↔ m = 0 := 
sorry

-- Proof that -1/2 < m < 2 given the graph does not pass through the second quadrant
theorem problem4 : not_in_second_quadrant m ↔ (-1/2 < m ∧ m < 2) := 
sorry

end

end problem1_problem2_problem3_problem4_l37_37921


namespace rectangle_area_proof_l37_37999

-- Given Conditions
variables (BE AF : ℝ) (E : Point) (F : Point) (C : ∠)
variables (BE_length : BE = 8) (AF_length : AF = 4)
variables (trisected_C : angle_trisection ∠C E F)

-- Rectangle definition
structure Rectangle (A B C D : Point) :=
(angle_A : ∠A = 90)
(angle_B : ∠B = 90)
(angle_C : ∠C = 90)
(angle_D : ∠D = 90)
(length_BC : ℝ)
(length_CD : ℝ)

-- Area calculation
def area_rect {A B C D : Point} (rect : Rectangle A B C D) : ℝ :=
  rect.length_BC * rect.length_CD

-- Proof statement for the given problem
theorem rectangle_area_proof {A B C D : Point} (rect : Rectangle A B C D) :
  (rect.length_BC = 8 * sqrt 3) →
  (rect.length_CD = 24 - 4 * sqrt 3) →
  area_rect rect = 192 * sqrt 3 - 96 :=
by 
  intros hBC hCD
  rw [area_rect, hBC, hCD]
  sorry

end rectangle_area_proof_l37_37999


namespace range_of_fx_ge_1_l37_37130

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then abs (3 * x - 4)
  else 2 / (x - 1)

theorem range_of_fx_ge_1 :
  { x : ℝ | f x ≥ 1 } = { x : ℝ | x ≤ 1 } ∪ { x : ℝ | 5 / 3 ≤ x ∧ x ≤ 3 } :=
begin
  sorry
end

end range_of_fx_ge_1_l37_37130


namespace valid_triples_are_zero_l37_37027

def is_prime_covering (P : ℤ → ℤ) : Prop :=
  ∀ (p : ℕ), p.prime → ∃ n : ℤ, p ∣ P n

def polynomial (a b c : ℤ) (x : ℤ) : ℤ :=
  (x^2 - a) * (x^2 - b) * (x^2 - c)

def number_of_valid_triples : ℕ :=
  (Finset.range 25).filter (λ a, 1 ≤ a).sublist 3 |>.filter
    (λ abc, let ⟨a, b, c⟩ := abc in (1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25) ∧
                              is_prime_covering (polynomial a b c)).size

theorem valid_triples_are_zero : number_of_valid_triples = 0 :=
  sorry

end valid_triples_are_zero_l37_37027


namespace positive_t_solution_l37_37906

theorem positive_t_solution (t : ℝ) (ht : 0 < t) (h : complex.abs (8 + complex.I * 2 * t) = 14) : t = real.sqrt 33 :=
sorry

end positive_t_solution_l37_37906


namespace john_hot_chocolate_l37_37635

theorem john_hot_chocolate (h1 : ∀ t : ℕ, t = 5 → 60 * t = 300) 
                           (h2 : ∀ t : ℕ, 300 = 20 * t → t = 15) :
  ∃ c : ℕ, (300 = 20 * c) ∧ c = 15 := 
by
  have H1 := h1 5 rfl
  have H2 := h2 15 rfl
  exact ⟨15, ⟨H1.symm.trans H2, rfl⟩⟩

end john_hot_chocolate_l37_37635


namespace arithmetic_sequence_sum_l37_37549

theorem arithmetic_sequence_sum (a b c : ℤ)
  (h1 : ∃ d : ℤ, a = 3 + d)
  (h2 : ∃ d : ℤ, b = 3 + 2 * d)
  (h3 : ∃ d : ℤ, c = 3 + 3 * d)
  (h4 : 3 + 3 * (c - 3) = 15) : a + b + c = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l37_37549


namespace inequality_proof_l37_37080

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end inequality_proof_l37_37080


namespace order_magnitudes_l37_37125

noncomputable def a := Real.log (2 / 3)
noncomputable def b := - Real.logBase 3 (3 / 2)
noncomputable def c := (2 / 3) ^ (1 / 3 : ℝ)

theorem order_magnitudes : c > a ∧ a > b := by
  sorry

end order_magnitudes_l37_37125


namespace num_men_scenario1_is_15_l37_37158

-- Definitions based on the conditions
def hours_per_day_scenario1 : ℕ := 9
def days_scenario1 : ℕ := 16
def men_scenario2 : ℕ := 18
def hours_per_day_scenario2 : ℕ := 8
def days_scenario2 : ℕ := 15
def total_work_done : ℕ := men_scenario2 * hours_per_day_scenario2 * days_scenario2

-- Definition of the number of men M in the first scenario
noncomputable def men_scenario1 : ℕ := total_work_done / (hours_per_day_scenario1 * days_scenario1)

-- Statement of desired proof: prove that the number of men in the first scenario is 15
theorem num_men_scenario1_is_15 :
  men_scenario1 = 15 := by
  sorry

end num_men_scenario1_is_15_l37_37158


namespace minimum_value_condition_l37_37653

noncomputable def minimum_value (a : ℝ) (h : a ≥ 1) : ℝ :=
  let m := a - real.sqrt (a ^ 2 - 1)
  let n := a + real.sqrt (a ^ 2 - 1)
  (m - 1) ^ 2 + (n - 1) ^ 2

theorem minimum_value_condition (a : ℝ) (h : a ≥ 1) : 
  minimum_value a h = if a = 1 then 0 else 4 * (a - 0.5) ^ 2 := by 
  sorry

end minimum_value_condition_l37_37653


namespace first_terrific_tuesday_after_start_l37_37571

theorem first_terrific_tuesday_after_start (start_date : ℕ) (is_monday : start_date = 2) 
  (has_31_days : ∀ (month : ℕ), month = 10 → 31)
  (fifth_tuesday_declaration : ∀ (month : ℕ), month = 10 → count_tuesdays 10 = 5 → is_terrific 31) : 
  ∃ (terrific_tuesday : ℕ), terrific_tuesday = 31 :=
by
  sorry

def count_tuesdays (month : ℕ) : ℕ :=
  if month = 10 then 5 else 0

def is_terrific (date : ℕ) : Prop :=
  date = 31

end first_terrific_tuesday_after_start_l37_37571


namespace unique_integer_n_l37_37462

theorem unique_integer_n
  (n : ℕ)
  (sum_b : ℝ)
  (b : Fin n → ℝ)
  (H_sum : sum (λ i, b i) = 23)
  (H_Tn : ∀ n, (T_n:ℝ) = sqrt (n^4 + 4 * n^2 + 530)) :
  ∃! n, T_n ∈ ℝ ∧ ∀ m, T_m = sqrt (10^4 + 4 * 10^2 + 530) :=
by
  sorry

end unique_integer_n_l37_37462


namespace work_completion_days_l37_37823

theorem work_completion_days (D : ℕ) (W : ℕ) :
  (D : ℕ) = 6 :=
by 
  -- define constants and given conditions
  let original_men := 10
  let additional_men := 10
  let early_days := 3

  -- define the premise
  -- work done with original men in original days
  have work_done_original : W = (original_men * D) := sorry
  -- work done with additional men in reduced days
  have work_done_with_additional : W = ((original_men + additional_men) * (D - early_days)) := sorry

  -- prove the equality from the condition
  have eq : original_men * D = (original_men + additional_men) * (D - early_days) := sorry

  -- simplify to solve for D
  have solution : D = 6 := sorry

  exact solution

end work_completion_days_l37_37823


namespace count_terms_in_simplified_expression_l37_37034

theorem count_terms_in_simplified_expression :
  let x y z w : ℝ := 0;
  let f := (x + y + z + w)^2010 + (x - y - z - w)^2010;
  ∃ n : ℕ, n = ∑ a in Finset.range' 0 2011, if a % 2 = 0 then Nat.choose (2010 - a + 2) 2 else 0 :=
sorry

end count_terms_in_simplified_expression_l37_37034


namespace proof_m_plus_n_l37_37914

variable (m n : ℚ) -- Defining m and n as rational numbers (ℚ)
-- Conditions from the problem:
axiom condition1 : 2 * m + 5 * n + 8 = 1
axiom condition2 : m - n - 3 = 1

-- Proof statement (theorem) that needs to be established:
theorem proof_m_plus_n : m + n = -2/7 :=
by
-- Since the proof is not required, we use "sorry" to placeholder the proof.
sorry

end proof_m_plus_n_l37_37914


namespace relationship_between_a_b_c_l37_37473

noncomputable def a : ℝ := Real.log 11 / Real.log 5
noncomputable def b : ℝ := Real.log 8 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt Real.exp 1

theorem relationship_between_a_b_c : a < b ∧ b < c := by
  sorry

end relationship_between_a_b_c_l37_37473


namespace georgie_ghost_enter_exit_diff_window_l37_37358

theorem georgie_ghost_enter_exit_diff_window (n : ℕ) (h : n = 8) :
    (∃ enter exit, enter ≠ exit ∧ 1 ≤ enter ∧ enter ≤ n ∧ 1 ≤ exit ∧ exit ≤ n) ∧
    (∃ W : ℕ, W = (n * (n - 1))) :=
sorry

end georgie_ghost_enter_exit_diff_window_l37_37358


namespace floor_sqrt_33_squared_eq_25_l37_37046

theorem floor_sqrt_33_squared_eq_25:
  (⌊real.sqrt 33⌋ : ℕ)^2 = 25 :=
by
  sorry

end floor_sqrt_33_squared_eq_25_l37_37046


namespace minimum_circumcircle_radius_l37_37235

variables (a b c : ℝ) (abc_triangle : Triangle ABC) (AM : Point) (BM : Point) (CM : Point)
           (on_AB : M ∈ segment A B)
           (AM_val : dist A M = a)
           (BM_val : dist B M = b)
           (CM_val : dist C M = c)
           (condition1 : c < a)
           (condition2 : c < b)

theorem minimum_circumcircle_radius : 
  ∃ R, R = (1 / 2) * ((a * b / c) + c) ∧ is_circumcircle_radius abc_triangle R :=
by sorry

end minimum_circumcircle_radius_l37_37235


namespace find_share_of_b_l37_37330

variable (a b c : ℕ)
axiom h1 : a = 3 * b
axiom h2 : b = c + 25
axiom h3 : a + b + c = 645

theorem find_share_of_b : b = 134 := by
  sorry

end find_share_of_b_l37_37330


namespace centroid_inequality_l37_37075

-- Define the triangle with vertices A, B, C
variables {A B C G E F : Type} 
variables [geometry A B C G E F]

-- Define G as the centroid of triangle ABC
def is_centroid (G : Type) (A B C : Type) : Prop :=
  ∃ (G : Type), G.centroid_of_triangle A B C

-- Define that line through G intersects AB at E and AC at F
def intersects (E F G : Type) (A B C : Type) : Prop :=
  ∃ (E F : Type), E ∈ line_of G A ∧ E ∈ segment_of A B ∧ F ∈ line_of G A ∧ F ∈ segment_of A C

-- Main theorem to prove
theorem centroid_inequality (hG : is_centroid G A B C) (hEF : intersects E F G A B C) : 
  distance E G ≤ 2 * distance G F :=
sorry

end centroid_inequality_l37_37075


namespace mans_rate_in_still_water_l37_37331

theorem mans_rate_in_still_water (with_stream against_stream : ℝ) 
  (h1 : with_stream = 25) (h2 : against_stream = 13) : 
  (with_stream + against_stream) / 2 = 19 :=
by
  rw [h1, h2]
  norm_num
  sorry

end mans_rate_in_still_water_l37_37331


namespace intersecting_lines_common_point_or_plane_intersecting_circles_common_points_or_sphere_l37_37738

-- Part (a)
theorem intersecting_lines_common_point_or_plane 
  (L : Set (ℝ × ℝ → ℝ))
  (h : ∀ (l1 l2 : (ℝ × ℝ → ℝ)), l1 ∈ L → l2 ∈ L → l1 ≠ l2 → ∃ (p : ℝ × ℝ), l1 p = 0 ∧ l2 p = 0) :
  (∃ (p : ℝ × ℝ), ∀ l ∈ L, l p = 0) ∨ (∃ (π : ℝ → ℝ), ∀ l ∈ L, ∃ (a b c : ℝ), l = λ p, a * p.1 + b * p.2 + c ∧ a * π 1 + b * π 2 + c = 0) :=
sorry

-- Part (b)
theorem intersecting_circles_common_points_or_sphere 
  (C : Set (ℝ × ℝ → ℝ → ℝ))
  (h : ∀ (c1 c2 : ℝ × ℝ → ℝ → ℝ), c1 ∈ C → c2 ∈ C → c1 ≠ c2 → ∃ (p1 p2 : ℝ × ℝ), c1 p1 = 0 ∧ c2 p1 = 0 ∧ c1 p2 = 0 ∧ c2 p2 = 0) :
  (∃ (p1 p2 : ℝ × ℝ), ∀ c ∈ C, c p1 = 0 ∧ c p2 = 0) ∨ (∃ (σ : Set (ℝ × ℝ → ℝ)), ∀ c ∈ C, ∃ (a b r : ℝ), c = λ (p : ℝ × ℝ), (p.1 - a) ^ 2 + (p.2 - b) ^ 2 - r ^ 2 = 0 ∧ ∀ (p1 p2 : ℝ × ℝ), c p1 = 0 ∧ c p2 = 0 → (σ p1 = 0 ∧ σ p2 = 0)) :=
sorry

end intersecting_lines_common_point_or_plane_intersecting_circles_common_points_or_sphere_l37_37738


namespace min_value_S_minus_AB_l37_37138

-- Define the conditions and restate the theorem in Lean 4
theorem min_value_S_minus_AB (k : ℝ) (x₁ x₂ y₁ y₂ x₀ y₀ S : ℝ) 
  (h1 : y₁ = k * x₁ + 1) 
  (h2 : y₂ = k * x₂ + 1) 
  (h3 : x₁^2 = 4 * y₁)
  (h4 : x₂^2 = 4 * y₂) 
  (h5 : y₀ = k * 2 * x₀)
  (h6 : x₀ = 2 * k)
  (h7 : y₀ = k^2)
  (h8 : S = (1 / 2) * (y₁ + y₂ + 4) * sqrt(k^2 + 1)) : 
  let AB := y₁ + y₂ + 4 in
  let d := sqrt(k^2 + 1) in
  S - AB = - (64 / 27) :=
by
  sorry

end min_value_S_minus_AB_l37_37138


namespace sequence_a_n_eq_T_n_formula_C_n_formula_l37_37493

noncomputable def sequence_S (n : ℕ) : ℕ := n * (2 * n - 1)

def arithmetic_seq (n : ℕ) : ℚ := 2 * n - 1

def a_n (n : ℕ) : ℤ := 4 * n - 3

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * n + 1)

def c_n (n : ℕ) : ℚ := 3^(n - 1)

def C_n (n : ℕ) : ℚ := (3^n - 1) / 2

theorem sequence_a_n_eq (n : ℕ) : a_n n = 4 * n - 3 := by sorry

theorem T_n_formula (n : ℕ) : T_n n = (n : ℚ) / (4 * n + 1) := by sorry

theorem C_n_formula (n : ℕ) : C_n n = (3^n - 1) / 2 := by sorry

end sequence_a_n_eq_T_n_formula_C_n_formula_l37_37493


namespace ellipse_proof_l37_37500

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (ecc : ℝ) (ecc_h : ecc = (Real.sqrt 3) / 2) 
  (dist_h : 2 = a) : Prop :=
  let c := ecc * a in
  (c = Real.sqrt 3) ∧
  (a^2 = b^2 + c^2) ∧
  (b = 1) ∧
  (b^2 = 1) ∧
  (a = 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1)) 

theorem ellipse_proof : ellipse_equation 2 1 
  (by linarith) (by linarith) 
  ((Real.sqrt 3) / 2) (by norm_num : (Real.sqrt 3) / 2 = (Real.sqrt 3) / 2) 
  (by norm_num : 2 = 2) :=
  sorry

end ellipse_proof_l37_37500


namespace odd_fraction_l37_37171

theorem odd_fraction (n : ℕ) (n_pos : n = 15) : 
  let odd_count := (list.range ((n + 1) - 1)).countp (λ x, x % 2 = 1)
  odd_count^2 = (64 : ℕ) :=
begin
  sorry
end

example : ((64 : ℕ)/(225 : ℕ)) = (64 / 225 : ℝ) := 
begin
  norm_num,
end

end odd_fraction_l37_37171


namespace coprime_set_contains_prime_l37_37654

def is_coprime_pair (m n : ℕ) : Prop := Nat.gcd m n = 1

def contains_prime (S : Finset ℕ) : Prop :=
  ∃ a ∈ S, Nat.prime a

theorem coprime_set_contains_prime :
  ∀ (S : Finset ℕ),
  (S.card = 15) →
  (∀ a ∈ S, a ≥ 2 ∧ a ≤ 2012) →
  (∀ a b ∈ S, a ≠ b → is_coprime_pair a b) →
  contains_prime S :=
sorry

end coprime_set_contains_prime_l37_37654


namespace part1_part2_l37_37960

-- Definition of the system of linear equations
def system_of_equations (m : ℝ) (x y : ℝ) : Prop :=
  x + y = -4 * m ∧ 2 * x + y = 2 * m + 1

-- Definition of the solution in terms of m
def solution (m : ℝ) : ℝ × ℝ :=
  (6 * m + 1, -10 * m - 1)

-- First part: verifying that the proposed solution satisfies the system of equations
theorem part1 (m : ℝ) : 
  system_of_equations m (fst (solution m)) (snd (solution m)) := 
sorry

-- Definition of the additional condition
def solution_satisfies_condition (x y : ℝ) : Prop :=
  x - y = 10

-- Second part: finding the value of m that satisfies the additional condition
theorem part2 : 
  ∀ m, solution_satisfies_condition (fst (solution m)) (snd (solution m)) → m = 1/2 := 
sorry

end part1_part2_l37_37960


namespace ball_drawing_ways_l37_37768

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37768


namespace Carl_l37_37838

noncomputable def garden_area_proof (posts total_posts : ℕ) (distance_between_posts : ℕ) (shorter_ratio longer_ratio : ℕ) 
    (a b : ℕ) (shorter_posts longer_posts : ℕ) : Prop :=
  let shorter_length := distance_between_posts * (shorter_posts - 1)
  let longer_length := distance_between_posts * (longer_posts - 1)
  let area := shorter_length * longer_length
  (total_posts = 2 * (a + b)) ∧
  (a = shorter_ratio * (longer_ratio + 1) - 1) ∧
  (b = 3 * (a + 1) - 1) ∧
  (area = 2016)

theorem Carl's_garden_area :
  ∃ (a b : ℕ), garden_area_proof 36 36 6 1 3 a b (a + 1) (b + 1) :=
begin
  existsi 4,
  existsi 14,
  unfold garden_area_proof,
  simp [/* Simplify further if needed */],
  sorry --proof omitted
end

end Carl_l37_37838


namespace sum_of_common_divisors_is_10_l37_37290

-- Define the list of numbers
def numbers : List ℤ := [42, 84, -14, 126, 210]

-- Define the common divisors
def common_divisors : List ℕ := [1, 2, 7]

-- Define the function that checks if a number is a common divisor of all numbers in the list
def is_common_divisor (d : ℕ) : Prop :=
  ∀ n ∈ numbers, (d : ℤ) ∣ n

-- Specify the sum of the common divisors
def sum_common_divisors : ℕ := common_divisors.sum

-- State the theorem to be proved
theorem sum_of_common_divisors_is_10 : 
  (∀ d ∈ common_divisors, is_common_divisor d) → 
  sum_common_divisors = 10 := 
by
  sorry

end sum_of_common_divisors_is_10_l37_37290


namespace single_discount_equivalence_l37_37804

noncomputable def original_price : ℝ := 50
noncomputable def discount1 : ℝ := 0.15
noncomputable def discount2 : ℝ := 0.10
noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)
noncomputable def effective_discount_price := 
  apply_discount (apply_discount original_price discount1) discount2
noncomputable def effective_discount :=
  (original_price - effective_discount_price) / original_price

theorem single_discount_equivalence :
  effective_discount = 0.235 := by
  sorry

end single_discount_equivalence_l37_37804


namespace fraction_greater_than_decimal_l37_37861

theorem fraction_greater_than_decimal :
  (1 / 4 : ℝ) > (24999999 / (10^8 : ℝ)) + (1 / (4 * (10^8 : ℝ))) :=
by
  sorry

end fraction_greater_than_decimal_l37_37861


namespace correct_statement_D_l37_37321

-- Definitions based on the conditions in the problem
def precision_thousand (x : ℕ) : Prop :=
  x % 1000 = 0

def precision_tenth (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n / 10

def rounded_to_nearest_thousand (x y : ℕ) : Prop :=
  abs (x - y) < 500

def representation_ten_thousand (x : ℕ) (y : ℚ) : Prop :=
  y * 10000 = x

-- Theorem stating that D is the correct statement given the conditions
theorem correct_statement_D :
  rounded_to_nearest_thousand 317500 318000 ∧ representation_ten_thousand 318000 31.8 :=
by
  sorry

end correct_statement_D_l37_37321


namespace length_of_goods_train_l37_37008

/-- 
Define the problem's conditions and use them to prove that the length of the goods train is 299.976 meters.
-/
theorem length_of_goods_train (woman_train_speed : ℝ) (goods_train_speed : ℝ) (time_seconds : ℝ) (length_meters : ℝ) : 
  woman_train_speed = 20 ∧ 
  goods_train_speed = 51.99424046076314 ∧ 
  time_seconds = 15 -> 
  length_meters = 299.976 :=
by {
  intro h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry
}

end length_of_goods_train_l37_37008


namespace sum_consecutive_nat_numbers_l37_37416

theorem sum_consecutive_nat_numbers (k : ℕ) : 
  let start := k^2 + 1
  let seq := list.range (2*k+2) |>.map (λ i, start + i)
  let S := seq.sum
  S = (k + 1)^3 + k^3 := 
  sorry

end sum_consecutive_nat_numbers_l37_37416


namespace set_of_points_A_l37_37710

noncomputable def S (A B C : Point) : Set Point :=
{ A | ∃ (h : ℝ), 1/2 * (dist B C) * h = 4 }

theorem set_of_points_A (B C : Point) (BC_nonzero: dist B C ≠ 0) :
  ∃ l₁ l₂ : Line, (∀ A ∈ S A B C, A ∈ l₁ ∨ A ∈ l₂) ∧
  parallel l₁ l₂ ∧
  dist_line_point l₁ B = dist_line_point l₂ B ∧
  dist_line_point l₁ B = 8 / dist B C := sorry

end set_of_points_A_l37_37710


namespace percent_increase_area_eq_491_9_l37_37014

theorem percent_increase_area_eq_491_9 :
  let s1 := 4
  let s5 := s1 * 1.25^4
  let area (s : ℝ) := (real.sqrt 3 / 4) * s^2
  let A1 := area s1
  let A5 := area s5
  let percent_increase := (A5 - A1) / A1 * 100
  A1 = 4 * real.sqrt 3 →
  A5 = 23.6746826171875 * real.sqrt 3 →
  percent_increase ≈ 491.9 :=
by
  intros
  have A1_eq : A1 = 4 * real.sqrt 3 := by assumption
  have A5_eq : A5 = 23.6746826171875 * real.sqrt 3 := by assumption
  sorry

end percent_increase_area_eq_491_9_l37_37014


namespace parabola_directrix_l37_37879

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37879


namespace sin_sum_of_roots_l37_37934

theorem sin_sum_of_roots (x1 x2 m : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ π) (hx2 : 0 ≤ x2 ∧ x2 ≤ π)
    (hroot1 : 2 * Real.sin x1 + Real.cos x1 = m) (hroot2 : 2 * Real.sin x2 + Real.cos x2 = m) :
    Real.sin (x1 + x2) = 4 / 5 := 
sorry

end sin_sum_of_roots_l37_37934


namespace find_A_find_b_c_l37_37987

noncomputable def triangle_A (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = (sqrt 3) * a * sin C - c * cos A

noncomputable def triangle_b_c (a b c : ℝ) (A : ℝ) (area : ℝ) : Prop :=
  a = 2 ∧ area = sqrt 3 ∧ b * c = 4 ∧ b + c = 4

theorem find_A :
  ∀ (a b c A B C : ℝ), triangle_A a b c A B C → A = π / 3 :=
begin
  intros a b c A B C hC,
  sorry
end

theorem find_b_c :
  ∀ (a b c A : ℝ), a = 2 ∧ a = 2 ∧ a * c * sin (π / 3) = sqrt 3 → triangle_b_c a b c A (sqrt 3) → b = 2 ∧ c = 2 :=
begin
  intros a b c A h_ac_area h_triangle,
  sorry
end

end find_A_find_b_c_l37_37987


namespace Ahmed_has_more_trees_l37_37824

theorem Ahmed_has_more_trees : 
  ∀(ao lo aa ha ho hl ha : ℕ) (h1 : ao = 8) (h2 : lo = 6) (h3 : ha = 1) (h4 : ho = 2) (h5 : hl = 5) (h6 : ha = 3) (h_apple : aa = 4 * ha),
  (ao + lo + aa - (ha + ho + hl + ha)) = 7 :=
by
  intros
  rw [h1, h2, h3, h4, h5, h6, h_apple]
  -- Now the expression should be "8 + 6 + 4 - (1 + 2 + 5 + 3) = 7"
  norm_num
  sorry

end Ahmed_has_more_trees_l37_37824


namespace axis_of_symmetry_l37_37109

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f(x + 2) = f(2 - x)) :
  ∃! c, ∀ x, f(2 * x) = f (2 * (2 * c - x)) :=
by
  sorry

end axis_of_symmetry_l37_37109


namespace isosceles_triangle_if_trig_cond_l37_37988

theorem isosceles_triangle_if_trig_cond (A B C : ℝ) (h : sin A * sin B - 1 = -(sin (C / 2))^2) : 
  A = B ∨ A = C ∨ B = C := 
sorry

end isosceles_triangle_if_trig_cond_l37_37988


namespace hyperbola_asymptote_slope_l37_37272

theorem hyperbola_asymptote_slope :
  (∃ m : ℚ, m > 0 ∧ ∀ x : ℚ, ∀ y : ℚ, ((x*x/16 - y*y/25 = 1) → (y = m * x ∨ y = -m * x))) → m = 5/4 :=
sorry

end hyperbola_asymptote_slope_l37_37272


namespace coeff_x_squared_l37_37265

theorem coeff_x_squared (x : ℝ) (h : x ≠ 0) :
  (let f := (1/x - 2) * (x + 1)^5 in
   @Monomial.coeff _ _ _ 2 f = -30) :=
sorry

end coeff_x_squared_l37_37265


namespace sam_last_30_minutes_speed_l37_37647

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end sam_last_30_minutes_speed_l37_37647


namespace mod_inverse_7_31_l37_37059

theorem mod_inverse_7_31 : ∃ a : ℕ, (7 * a) % 31 = 1 ∧ a = 9 :=
by
  use 9
  split
  by norm_num
  sorry

end mod_inverse_7_31_l37_37059


namespace determine_h_l37_37854

theorem determine_h : 
  ∃ k : ℝ, ∀ x : ℝ, (3 * x^2 + 9 * x + 20) = 3 * (x + (3/2))^2 + k :=
begin
  sorry
end


end determine_h_l37_37854


namespace part_I_part_II_l37_37214

-- Define the function f
def f (n : ℝ) (x : ℝ) : ℝ := (1 / n) * x + (1 / x) - (1 / 2)

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1 / 2) * x^2 - a * x - (1 / 2) * a^2

-- Part I: Prove |f(x)| ≥ - (x-1)^2 + 1/2 for x > 0
theorem part_I (n : ℝ) (x : ℝ) (h : x > 0) : abs (f n x) ≥ -(x - 1)^2 + (1 / 2) := sorry

-- Part II: Find the range of a such that g(x1) ≥ ⌊f(x2)⌋ for any x1 ≥ 0 and some x2 > 0
theorem part_II (a : ℝ) :
  (∀ x1 : ℝ, x1 ≥ 0 → ∃ x2 : ℝ, x2 > 0 ∧ g a x1 ≥ Real.floor (f 1 x2)) ↔ (-Real.sqrt 2 ≤ a ∧ a ≤ 2 - Real.log 2) := sorry

end part_I_part_II_l37_37214


namespace ryan_distance_correct_l37_37413

-- Definitions of the conditions
def billy_distance : ℝ := 30
def madison_distance : ℝ := billy_distance * 1.2
def ryan_distance : ℝ := madison_distance * 0.5

-- Statement to prove
theorem ryan_distance_correct : ryan_distance = 18 := by
  sorry

end ryan_distance_correct_l37_37413


namespace cone_radius_l37_37682

noncomputable def radius_of_cone (V : ℝ) (h : ℝ) : ℝ := 
  3 / Real.sqrt (Real.pi)

theorem cone_radius :
  ∀ (V h : ℝ), V = 12 → h = 4 → radius_of_cone V h = 3 / Real.sqrt (Real.pi) :=
by
  intros V h hV hv
  sorry

end cone_radius_l37_37682


namespace pentagon_C_y_coordinate_l37_37639

theorem pentagon_C_y_coordinate :
  ∃ (C_y : ℝ), let C := (3, C_y) in
  let A := (0, 0) in
  let B := (0, 6) in
  let E := (6, 0) in
  let D := (6, 6) in
  let pentagon_area := 72 in
  (1 / 2) * 6 * (C.2 - 6) = 36 ∧ pentagon_area = 72
  → C_y = 18 :=
by
  sorry

end pentagon_C_y_coordinate_l37_37639


namespace even_perfect_last_digits_l37_37053

-- Define the conditions under which n is a perfect number
def is_perfect (n : ℕ) : Prop :=
  nat.sigma n = 2 * n

-- Define the condition that an even perfect number can be written as 2^(k-1) * (2^k - 1)
def is_even_perfect (n : ℕ) : Prop :=
  ∃ k : ℕ, nat.prime (2^k - 1) ∧ n = 2^(k-1) * (2^k - 1)

-- The last digit cycles for powers of 2
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the last digit of (2^k - 1) based on k mod 4
def last_digit_of_mersenne_prime (k : ℕ) : ℕ :=
  ((2^k - 1) % 10)

-- Prove that the last digits of even perfect numbers are 6 and 8
theorem even_perfect_last_digits :
  ∀ n : ℕ, is_even_perfect n → (last_digit n = 6 ∨ last_digit n = 8) :=
by sorry

end even_perfect_last_digits_l37_37053


namespace draw_4_balls_ordered_l37_37756

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37756


namespace smallest_positive_integer_term_decimal_and_contains_digit_9_l37_37308

theorem smallest_positive_integer_term_decimal_and_contains_digit_9 :
  ∃ (n : ℕ), n > 0 ∧
    (∀ (p : ℕ), nat.prime p → p ∣ n → p = 2 ∨ p = 5) ∧
    (∃ k : ℕ, n.to_digits 10 = k :: n.to_digits 10 ∧ k = 9) ∧
    n = 2560 :=
by sorry

end smallest_positive_integer_term_decimal_and_contains_digit_9_l37_37308


namespace train_a_distance_at_meeting_l37_37302

-- Define the problem conditions as constants
def distance := 75 -- distance between start points of Train A and B
def timeA := 3 -- time taken by Train A to complete the trip in hours
def timeB := 2 -- time taken by Train B to complete the trip in hours

-- Calculate the speeds
def speedA := distance / timeA -- speed of Train A in miles per hour
def speedB := distance / timeB -- speed of Train B in miles per hour

-- Calculate the combined speed and time to meet
def combinedSpeed := speedA + speedB
def timeToMeet := distance / combinedSpeed

-- Define the distance traveled by Train A at the time of meeting
def distanceTraveledByTrainA := speedA * timeToMeet

-- Theorem stating Train A has traveled 30 miles when it met Train B
theorem train_a_distance_at_meeting : distanceTraveledByTrainA = 30 := by
  sorry

end train_a_distance_at_meeting_l37_37302


namespace dilation_center_1_plus_2i_scale_2_l37_37665

theorem dilation_center_1_plus_2i_scale_2 :
  ∀ (z w : ℂ), (w - (1 + 2 * complex.I) = 2 * ((z - (1 + 2 * complex.I))) → z = (2 + complex.I) → w = 3) :=
by
  intros z w h1 h2
  sorry

end dilation_center_1_plus_2i_scale_2_l37_37665


namespace mark_spends_47_l37_37705

def apple_price : ℕ := 2
def apple_quantity : ℕ := 4
def bread_price : ℕ := 3
def bread_quantity : ℕ := 5
def cheese_price : ℕ := 6
def cheese_quantity : ℕ := 3
def cereal_price : ℕ := 5
def cereal_quantity : ℕ := 4
def coupon : ℕ := 10

def calculate_total_cost (apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon : ℕ) : ℕ :=
  let apples_cost := apple_price * (apple_quantity / 2)  -- Apply buy-one-get-one-free
  let bread_cost := bread_price * bread_quantity
  let cheese_cost := cheese_price * cheese_quantity
  let cereal_cost := cereal_price * cereal_quantity
  let subtotal := apples_cost + bread_cost + cheese_cost + cereal_cost
  let total_cost := if subtotal > 50 then subtotal - coupon else subtotal
  total_cost

theorem mark_spends_47 : calculate_total_cost apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon = 47 :=
  sorry

end mark_spends_47_l37_37705


namespace tan_theta_cos_sin_id_l37_37918

theorem tan_theta_cos_sin_id (θ : ℝ) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) =
  (17 * (Real.sqrt 10 + 1)) / 24 :=
by
  sorry

end tan_theta_cos_sin_id_l37_37918


namespace highest_score_of_batsman_l37_37733

theorem highest_score_of_batsman
  (total_innings : ℕ)
  (avg_innings : ℕ → ℚ)
  (highest_lowest_diff : ℚ := 150)
  (avg_excluding_highest_lowest : ℚ := 58)
  (total_runs : ℚ := avg_innings total_innings * total_innings)
  (total_runs_excluding_hl : ℚ := avg_excluding_highest_lowest * (total_innings - 2)) :
  (total_innings = 46) →
  (avg_innings 46 = 59) →
  total_runs = 2704 →
  total_runs_excluding_hl = 2552 →
  ∃ H L : ℚ, (H - L = highest_lowest_diff) ∧ (H + L = total_runs - total_runs_excluding_hl) ∧ H = 151 :=
by
  intros h_total_innings h_avg_innings h_total_runs h_total_runs_excluding
  let H := ((highest_lowest_diff + (total_runs - total_runs_excluding_hl)) / 2)
  have h_H : H = 151 := by sorry
  existsi H, H - highest_lowest_diff
  split
  · have h_diff : H - (H - highest_lowest_diff) = highest_lowest_diff := by sorry
    exact h_diff
  split
  · have h_sum : H + (H - highest_lowest_diff) = total_runs - total_runs_excluding_hl := by sorry
    exact h_sum
  · exact h_H

end highest_score_of_batsman_l37_37733


namespace max_choir_members_l37_37348

theorem max_choir_members (n : ℕ) (x y : ℕ) : 
  n = x^2 + 11 ∧ n = y * (y + 3) → n = 54 :=
by
  sorry

end max_choir_members_l37_37348


namespace number_of_sets_count_number_of_sets_l37_37039

theorem number_of_sets (P : Set ℕ) :
  ({1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) → (P = {1, 2} ∨ P = {1, 2, 3} ∨ P = {1, 2, 4}) :=
sorry

theorem count_number_of_sets :
  ∃ (Ps : Finset (Set ℕ)), 
  (∀ P ∈ Ps, {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) ∧ Ps.card = 3 :=
sorry

end number_of_sets_count_number_of_sets_l37_37039


namespace regular_octagon_area_l37_37813

noncomputable def area_of_regular_octagon (r : ℝ) : ℝ :=
  8 * (1/2 * r * r * real.sin (real.pi / 4))

theorem regular_octagon_area (r : ℝ) (h : r = 3) : 
  area_of_regular_octagon r = 18 * real.sqrt 2 := by
  rw [h, area_of_regular_octagon]
  sorry

end regular_octagon_area_l37_37813


namespace pradeep_max_marks_l37_37236

-- conditions
variables (M : ℝ)
variable (h1 : 0.40 * M = 220)

-- question and answer
theorem pradeep_max_marks : M = 550 :=
by
  sorry

end pradeep_max_marks_l37_37236


namespace largest_triangle_angle_l37_37678

theorem largest_triangle_angle (y : ℝ) (h1 : 45 + 60 + y = 180) : y = 75 :=
by { sorry }

end largest_triangle_angle_l37_37678


namespace banana_to_orange_equiv_l37_37020

variable (Banana Apple Orange : Type)
variable (cost : Banana → ℝ)
variable (cost : Apple → ℝ)
variable (cost : Orange → ℝ)

-- Conditions
axiom h1 : cost (5 * Banana) = cost (3 * Apple)
axiom h2 : cost (9 * Apple) = cost (6 * Orange)

-- The target statement
theorem banana_to_orange_equiv :
  cost (30 * Banana) = cost (12 * Orange) := 
sorry

end banana_to_orange_equiv_l37_37020


namespace relationship_among_a_b_c_l37_37527

noncomputable def a : ℝ := 5^(-1 / 2)
noncomputable def b : ℝ := Real.exp 1
def f (x : ℝ) := 3 * x^3 + x - 1

theorem relationship_among_a_b_c (aPos : a = 5^(-1 / 2))
  (bPos : ∃ b, Real.log b = 1)
  (cExists : ∃ c, f c = 0 ∧ c > 0 ∧ c < 1) :
  ∃ a b c, a = 5^(-1/2) ∧ Real.log b = 1 ∧ 3 * c^3 + c = 1 ∧ b > c ∧ c > a :=
by
  sorry

end relationship_among_a_b_c_l37_37527


namespace max_sum_product_l37_37613

theorem max_sum_product (a b c d : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum: a + b + c + d = 200) : 
  ab + bc + cd + da ≤ 10000 := 
sorry

end max_sum_product_l37_37613


namespace second_alloy_amount_l37_37998

theorem second_alloy_amount (x : ℝ) : 
  (0.10 * 15 + 0.08 * x = 0.086 * (15 + x)) → 
  x = 35 := by 
sorry

end second_alloy_amount_l37_37998


namespace right_triangle_legs_l37_37993

theorem right_triangle_legs (c : ℝ) (x y z : ℝ) (h1 : x^2 + y^2 = c^2)
  (h2 : x^2 + z^2 = c^2 / 3) (h3 : z = (y * x) / (c + x)) :
  x = 0.5 * c ∧ y = (sqrt 3 * c) / 2 := by
  sorry

end right_triangle_legs_l37_37993


namespace polynomial_functional_equation_l37_37425

theorem polynomial_functional_equation :
  ∀ P : Polynomial ℝ, 
    (∃ n : ℤ, 0 < n ∧ ∀ x ∈ ℚ, P.eval (x + (1 / n : ℝ)) + P.eval (x - (1 / n : ℝ)) = 2 * P.eval x) →
    (∃ a b : ℝ, ∀ x : ℝ, P.eval x = a * x + b) :=
by
  sorry

end polynomial_functional_equation_l37_37425


namespace draw_4_balls_ordered_l37_37760

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37760


namespace palindromes_between_300_800_l37_37543

def palindrome_count (l u : ℕ) : ℕ :=
  (u / 100 - l / 100 + 1) * 10

theorem palindromes_between_300_800 : palindrome_count 300 800 = 50 :=
by
  sorry

end palindromes_between_300_800_l37_37543


namespace matrix_equation_l37_37842

open Matrix
variable (a b c : ℝ)
noncomputable def d := a + b
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -b], 
    ![-d, 0, a], 
    ![b, -a, 0]]

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2 * a^2, a * b, a * c], 
    ![a * b, 2 * b^2, b * c], 
    ![a * c, b * c, 2 * c^2]]

def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_equation : A ⬝ B + 2 • I = ![![2, 0, 0], ![0, 2, 0], ![0, 0, 2]] :=
by
  sorry

end matrix_equation_l37_37842


namespace number_of_ways_to_draw_4_from_15_l37_37752

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37752


namespace linear_function_solutions_l37_37936

theorem linear_function_solutions (f : ℝ → ℝ) (h₀ : ∃ (a b : ℝ), ∀ x, f(x) = a * x + b)
  (h₁ : ∀ x, f(f(x)) = 16 * x - 15) :
  (∀ x, f(x) = 4 * x - 3) ∨ (∀ x, f(x) = -4 * x + 5) :=
sorry

end linear_function_solutions_l37_37936


namespace collinearity_proof_l37_37185

variable {A B C F E M X H : Type*}

noncomputable def problem_statement 
  (triangle : A × B × C)
  (angle_bisectors : (B × F) × (C × E))
  (midpoint : M = (B + C) / 2)
  (incircles : (ω_1 : Type*) × (ω_2 : Type*))
  (tangent_points : F × E)
  (intersection : X = intersect_internal_tangents(ω_1, ω_2)) : Prop :=
  collinear X M H

theorem collinearity_proof 
  (triangle : A × B × C)
  (angle_bisectors : (B × F) × (C × E))
  (midpoint : M = (B + C) / 2)
  (incircles : (ω_1 : Type*) × (ω_2 : Type*))
  (tangent_points : F × E)
  (intersection : X = intersect_internal_tangents(ω_1, ω_2)) : 
  collinear X M H :=
begin
  -- Proof goes here
  sorry
end

end collinearity_proof_l37_37185


namespace exists_similar_orthic_obtuse_triangle_l37_37248

noncomputable def is_orthic_triangle (A B C A' B' C' : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A'] [AddGroup B'] [AddGroup C'] [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (hA : A = lin_ord.add (pi / 7 : A) (0 : A)) 
  (hB : B = lin_ord.add (2 * pi / 7 : B) (0 : B)) 
  (hC : C = lin_ord.add (4 * pi / 7 : C) (0 : C)) 
  : Prop := 
  ∀ (A B C : Angle), 
  (180 - A - B - C = 0) ∧ 
  (A < B) ∧ (B < C) ∧ (A > pi / 2) ∧ 
  (orthic_triangle A B C A' B' C')

theorem exists_similar_orthic_obtuse_triangle : 
  ∃ (A B C : Angle), 
  is_obtuse A B C ∧ 
  is_orthic_triangle A B C :=
begin
  use (pi / 7, 2 * pi / 7, 4 * pi / 7),
  split,
  { /- is_obtuse part proof -/ sorry },
  { /- is_orthic_triangle part proof -/ sorry }
end

end exists_similar_orthic_obtuse_triangle_l37_37248


namespace identify_curve_is_cardioid_l37_37055

theorem identify_curve_is_cardioid (θ : ℝ) : (∃ r : ℝ, r = 1 + 2 * sin θ) :=
sorry

end identify_curve_is_cardioid_l37_37055


namespace initial_distance_correct_l37_37255

noncomputable def initial_distance_between_stacy_and_heather
  (H : ℝ := 5) -- Heather's walking rate
  (S : ℝ := 6) -- Stacy's walking rate
  (delay : ℝ := 0.4) -- Heather's delay in hours
  (heather_distance : ℝ := 10.272727272727273) -- Distance walked by Heather when they meet
  : ℝ :=
    let t := heather_distance / H in
    let stacy_distance := S * (t + delay) in
    heather_distance + stacy_distance

theorem initial_distance_correct : initial_distance_between_stacy_and_heather = 25 := by
  sorry

end initial_distance_correct_l37_37255


namespace right_triangle_hypotenuse_l37_37375

noncomputable def hypotenuse_of_rotated_cones 
  (x y : ℝ) 
  (h1: (1 / 3 * Real.pi * y ^ 2 * x = 1500 * Real.pi))
  (h2: (1 / 3 * Real.pi * x ^ 2 * y = 540 * Real.pi))
  : ℝ :=
  let h := Real.sqrt (x ^ 2 + y ^ 2) in h

theorem right_triangle_hypotenuse : 
  ∃ x y : ℝ, 
    (1 / 3 * Real.pi * y ^ 2 * x = 1500 * Real.pi) ∧ 
    (1 / 3 * Real.pi * x ^ 2 * y = 540 * Real.pi) ∧ 
    (Real.sqrt (x ^ 2 + y ^ 2) = 5 * Real.sqrt 34) :=
by
  sorry

end right_triangle_hypotenuse_l37_37375


namespace LCM_activities_l37_37022

theorem LCM_activities :
  ∃ (d : ℕ), d = Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) ∧ d = 48 :=
by
  sorry

end LCM_activities_l37_37022


namespace price_decrease_fourth_month_l37_37833

theorem price_decrease_fourth_month (P0 : ℝ) (x : ℝ) :
  P0 = 100 →
  let P1 := P0 * 1.15 in
  let P2 := P1 * 0.90 in
  let P3 := P2 * 1.30 in
  P3 * (1 - x) = P0 → 
  x = 0.26 :=
by
  intros hP0 --
  rw [hP0]
  let P1 := (100 : ℝ) * 1.15
  let P2 := P1 * 0.90
  let P3 := P2 * 1.30
  sorry

end price_decrease_fourth_month_l37_37833


namespace no_real_roots_range_l37_37160

theorem no_real_roots_range (a : ℝ) : ∀ (f : ℝ → ℝ), (∀ x, f x = x^2 + 2*x + a) → (∀ x, f x ≠ 0) → a > 1 :=
by
  intros f h₁ h₂
  have h₃ : ∀ x, 4 - 4 * a < 0 := sorry
  exact sorry

end no_real_roots_range_l37_37160


namespace fabric_softener_price_increase_l37_37281

variable (initial_price : ℝ) (new_price : ℝ) (initial_usage_per_15_liters : ℕ) 
variable (new_usage_per_8_liters : ℕ)

def price_increase_percentage (initial_price new_price : ℝ) (initial_usage new_usage : ℕ) : ℝ :=
  (((new_price / initial_price) * (new_usage / initial_usage) - 1) * 100)

theorem fabric_softener_price_increase :
  initial_price = 13.70 →
  new_price = 49 →
  initial_usage_per_15_liters = 2 →  -- Half capful per 15 liters means 1 capful per 30
  new_usage_per_8_liters = 15 / 8 →  -- 1 capful per 8 liters; multiplied by 15/8 to match capfuls with usages
  round (price_increase_percentage 13.70 49 2 (15 / 8)) = 1240 :=
by
  sorry

end fabric_softener_price_increase_l37_37281


namespace tan_K_in_right_triangle_l37_37173

theorem tan_K_in_right_triangle (KL JL : ℝ) (hkl : KL = 20) (hjl : JL = 12)
  (h_right : ∃ (J K L : Type) (s : fintype J) (a : angtype s), σ₁ = angle_axiom J K L a 90) : 
  real.tan (real.arctan (16 / 12)) = 4 / 3 := by
  -- This uses the right-angle and provided triangle properties
  sorry

end tan_K_in_right_triangle_l37_37173


namespace infer_correct_l37_37943

theorem infer_correct (a b c : ℝ) (h1: c < b) (h2: b < a) (h3: a + b + c = 0) :
  (c * b^2 ≤ ab^2) ∧ (ab > ac) :=
by
  sorry

end infer_correct_l37_37943


namespace abs_inequality_solution_l37_37650

theorem abs_inequality_solution :
  {x : ℝ | |x - 2| + |x + 3| < 7} = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end abs_inequality_solution_l37_37650


namespace correct_statements_count_l37_37826

theorem correct_statements_count :
  let s1 := (deriv sin x) ≠ -cos x
  let s2 := (deriv (deriv (1/x)) x) ≠ 1/(x^2)
  let s3 := (deriv (log x / log 3) x) ≠ 1/(3 * (log x))
  let s4 := (deriv (log x) x) = 1/x
  s1 ∧ s2 ∧ s3 ∧ s4 → 1 :=
by
  intro s1 s2 s3 s4
  sorry

end correct_statements_count_l37_37826


namespace select_100_numbers_no_infinite_subsequence_l37_37836

section part_a
def sequence : ℕ → ℚ
| 0 := 1
| (n + 1) := 1 / (n + 2)

theorem select_100_numbers :
  ∃ (seq : ℕ → ℚ), (∀ n, seq n = sequence n) ∧ (∀ m n : ℕ, m < 100 → n < 100 → m < n → seq m < seq n) :=
sorry
end part_a

section part_b
def infinite_sequence_property (seq : ℕ → ℚ) :=
∀ n ≥ 2, seq n = seq (n - 2) - seq (n - 1)

theorem no_infinite_subsequence :
  ¬∃ (seq : ℕ → ℚ), (∀ n, seq n ∈ { sequence k | k : ℕ }) ∧ infinite_sequence_property seq :=
sorry
end part_b

end select_100_numbers_no_infinite_subsequence_l37_37836


namespace max_x_real_nums_l37_37617

theorem max_x_real_nums (x y z : ℝ) (h₁ : x + y + z = 6) (h₂ : x * y + x * z + y * z = 10) : x ≤ 2 :=
sorry

end max_x_real_nums_l37_37617


namespace range_of_a_l37_37558

theorem range_of_a (a : ℝ) (h : a - 2 * 1 + 4 > 0) : a > -2 :=
by
  -- proof is not required
  sorry

end range_of_a_l37_37558


namespace directrix_of_parabola_l37_37899

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37899


namespace correct_statements_l37_37184

variables {A B C : Type}
variables (a b c : ℝ) (α β γ : ℝ)

noncomputable def triangle_obtuse (a b c : ℝ) (h1 : a^2 + b^2 < c^2) : Prop :=
  ∃ C : Type, ∠C > π / 2 ∧ a^2 + b^2 < c^2

noncomputable def sin_law_greater (a b : ℝ) (α β : ℝ) (h2 : sin α > sin β) : Prop :=
  a > b

theorem correct_statements (a b c α β γ : ℝ) :
  (triangle_obtuse a b c (by linarith) ∧ sin_law_greater a b α β (by simp))?
  sorry

end correct_statements_l37_37184


namespace area_of_circle_section_l37_37304

theorem area_of_circle_section 
  (h₁ : ∀ x y : ℝ, x^2 - 16x + y^2 = 48 → (x - 8)^2 + y^2 = 112) 
  (h₂ : ∀ x y : ℝ, y ≥ 0)
  (h₃ : ∀ x y : ℝ, y ≤ 7 - x) :
  ∃ (A : ℝ), A = 48 * Real.pi :=
begin
  sorry
end

end area_of_circle_section_l37_37304


namespace polynomial_expansion_correct_l37_37050

open Polynomial

noncomputable def poly1 : Polynomial ℤ := X^2 + 3 * X - 4
noncomputable def poly2 : Polynomial ℤ := 2 * X^2 - X + 5
noncomputable def expected : Polynomial ℤ := 2 * X^4 + 5 * X^3 - 6 * X^2 + 19 * X - 20

theorem polynomial_expansion_correct :
  poly1 * poly2 = expected :=
sorry

end polynomial_expansion_correct_l37_37050


namespace min_value_reciprocal_add_b_not_min_value_square_terms_max_value_a_over_b_min_value_half_b_minus_a_l37_37471

-- Problem 1: Prove that the minimum value of \( \frac{1}{a} + b \) is 4 given \( a > 0 \), \( b > 0 \), and \( a + \frac{1}{b} = 1 \).
theorem min_value_reciprocal_add_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 1/b = 1) : 
  (1/a) + b ≥ 4 := by
  sorry

-- Problem 2: Prove that \( a^2 + \frac{1}{b^2} \neq \frac{1}{4} \) given \( a > 0 \), \( b > 0 \), and \( a + \frac{1}{b} = 1 \).
theorem not_min_value_square_terms (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 1/b = 1) : 
  a^2 + (1/b^2) ≠ 1/4 := by
  sorry

-- Problem 3: Prove that the maximum value of \( \frac{a}{b} \) is \( \frac{1}{4} \) given \( a > 0 \), \( b > 0 \), and \( a + \frac{1}{b} = 1 \).
theorem max_value_a_over_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 1/b = 1) : 
  (a/b) ≤ 1/4 := by
  sorry

-- Problem 4: Prove that the minimum value of \( \frac{1}{2}b - a \) is \( \sqrt{2} - 1 \) given \( a > 0 \), \( b > 0 \), and \( a + \frac{1}{b} = 1 \).
theorem min_value_half_b_minus_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 1/b = 1) : 
  ((1/2)*b) - a ≥ sqrt 2 - 1 := by
  sorry

end min_value_reciprocal_add_b_not_min_value_square_terms_max_value_a_over_b_min_value_half_b_minus_a_l37_37471


namespace draw_4_balls_in_order_ways_l37_37774

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37774


namespace doctors_to_nurses_ratio_l37_37288

theorem doctors_to_nurses_ratio (total_people : ℕ) (number_of_nurses : ℕ) (number_of_doctors : ℕ)
    (H1 : total_people = 280) (H2 : number_of_nurses = 180) (H3 : number_of_doctors = total_people - number_of_nurses) :
    number_of_doctors * 9 = number_of_nurses * 5 :=
by
  rw [H1, H2] at H3
  rw [← H3]
  norm_num
  sorry

end doctors_to_nurses_ratio_l37_37288


namespace measure_of_angle_B_l37_37585

theorem measure_of_angle_B (a b c R : ℝ) (A B C : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C)
  (h4 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = Real.pi / 4 :=
by
  sorry

end measure_of_angle_B_l37_37585


namespace correct_solution_l37_37114

variable (n : ℕ)
variable (lambda : fin n → ℝ)
variable (x : fin n → ℝ)

-- Specifying the conditions
def positive_reals (l : fin n → ℝ) : Prop :=
  ∀ i, 0 < l i

axiom lambda_positive : positive_reals n lambda
axiom x_positive : positive_reals n x

-- Problem Definition
def satisfies_equation (x : fin n → ℝ) (lambda : fin n → ℝ) : Prop :=
  ∑ i, (x i + lambda i) ^ 2 / x ((i + 1) % n) = 4 * ∑ i, lambda i

-- The expected solution
def solution (lambda : fin n → ℝ) (k : fin n) : ℝ :=
  (1 / (2 ^ n - 1)) * ∑ i, 2 ^ i * lambda ((k.val + i - 1) % n)

-- The Proof problem to state in Lean
theorem correct_solution : 
  satisfies_equation n (solution n lambda) lambda :=
by 
  sorry

end correct_solution_l37_37114


namespace quadratic_coefficients_l37_37463

/-- Given the quadratic equation 5x² + 2x - 1 = 0, 
    the coefficients of the quadratic term, linear term, and constant term are 5, 2, and -1 respectively. -/
theorem quadratic_coefficients :
  ∀ (x : ℝ), 5 * x^2 + 2 * x - 1 = 0 → (5, 2, -1) :=
by
  intros
  sorry

end quadratic_coefficients_l37_37463


namespace polynomial_coefficients_sum_l37_37656

theorem polynomial_coefficients_sum :
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  10 * a + 5 * b + 2 * c + d = 60 :=
by
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  sorry

end polynomial_coefficients_sum_l37_37656


namespace object_reaches_maximum_height_at_t2_l37_37017

theorem object_reaches_maximum_height_at_t2 :
  ∀ t : ℝ, (h : ℝ) (h = -15 * (t - 2)^2 + 150) → ∃ t_max : ℝ, (h_max = -15 * (t_max - 2)^2 + 150) ∧ t_max = 2 :=
by
  sorry

end object_reaches_maximum_height_at_t2_l37_37017


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l37_37534

open Set -- Open the Set namespace for convenience

-- Define the universal set U, and sets A and B
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof statements
theorem complement_U_A : U \ A = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 3} :=
by sorry

theorem complement_U_intersection_A_B : U \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem complement_A_intersection_B : (U \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l37_37534


namespace Joe_total_income_l37_37198

theorem Joe_total_income : 
  (∃ I : ℝ, 0.1 * 1000 + 0.2 * 3000 + 0.3 * (I - 500 - 4000) = 848 ∧ I - 500 > 4000) → I = 4993.33 :=
by
  sorry

end Joe_total_income_l37_37198


namespace tan_alpha_plus_pi_div_4_l37_37932

theorem tan_alpha_plus_pi_div_4 (α : ℝ) (hcos : Real.cos α = 3 / 5) (h0 : 0 < α) (hpi : α < Real.pi) :
  Real.tan (α + Real.pi / 4) = -7 :=
by
  sorry

end tan_alpha_plus_pi_div_4_l37_37932


namespace triangular_weight_l37_37687

theorem triangular_weight (c t : ℝ) (h1 : c + t = 3 * c) (h2 : 4 * c + t = t + c + 90) : t = 60 := 
by sorry

end triangular_weight_l37_37687


namespace ellipse_equation_and_perimeter_constancy_l37_37516

theorem ellipse_equation_and_perimeter_constancy
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (E : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (A : -2 = -2 ∧ ∀ y : ℝ, E (-2 y) = 0)
  (B : 2 = 2 ∧ ∀ y : ℝ, E (2 y) = 0)
  (C : 1 = 1 ∧ ∀ y : ℝ, E (1 y) = 3 / 2)
  (F1 F2 : ℝ × ℝ)
  (L_intersects_E_at_M_N : ∃ M N : ℝ × ℝ, (E M.1 M.2 = 1) ∧ (E N.1 N.2 = 1))
  : (a^2 = 4 ∧ b^2 = 3) ∧ ∀ M N : ℝ × ℝ, E M.1 M.2 = 1 ∧ E N.1 N.2 = 1 → 2 * a + 2 * a = 8 := 
sorry

end ellipse_equation_and_perimeter_constancy_l37_37516


namespace smallest_digit_sum_nat_num_l37_37285

theorem smallest_digit_sum_nat_num : ∃ (n : ℕ), 
  (n = 25) ∧ 
  (∃ (x : ℕ), x < 10^n ∧ (nat_sum x = 218)) ∧ 
  (∀ m, (∃ (y : ℕ), y < 10^m ∧ (nat_sum y = 218)) → m ≥ n) :=
by 
  sorry


end smallest_digit_sum_nat_num_l37_37285


namespace sector_sin_alpha_l37_37492

theorem sector_sin_alpha : 
  ∀ (r : ℝ) (s : ℝ) (α : ℝ), r = 2 → s = (8 * real.pi) / 3 → α = s / r → 
  real.sin α = - (real.sqrt 3) / 2 :=
by
  intros r s α hr hs hα
  subst hr
  subst hs
  subst hα
  sorry

end sector_sin_alpha_l37_37492


namespace find_smallest_n_l37_37031

def smallest_n (n : ℕ) : Prop :=
  ∑ k in finset.range (n + 1), real.logb 3 (1 + 1 / 3^(2^k)) ≥ 1 + real.logb 3 (3000 / 3001)

theorem find_smallest_n : ∃ n, smallest_n n ∧ ∀ m, m < n → ¬smallest_n m :=
begin
  sorry,
end

end find_smallest_n_l37_37031


namespace sum_of_digit_products_l37_37234

theorem sum_of_digit_products : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7}
  ∑ d1 in digits, ∑ d2 in digits, ∑ d3 in digits, ∑ d4 in digits, ∑ d5 in (digits \ {0}), d1 * d2 * d3 * d4 * d5 = 17210368 := 
by
  sorry

end sum_of_digit_products_l37_37234


namespace smallest_positive_period_of_f_max_and_min_values_of_f_l37_37525

noncomputable def f (x : ℝ) : ℝ := 
  sin (2 * x + π / 3) + sin (2 * x - π / 3) + 2 * cos x ^ 2 - 1

theorem smallest_positive_period_of_f : (∀ x ∈ ℝ, f (x + π) = f x) ∧ 
                                      (∀ T > 0, (∀ x ∈ ℝ, f (x + T) = f x) → T ≥ π) :=
sorry

theorem max_and_min_values_of_f : 
  ∃ (max min : ℝ), (∀ x ∈ set.Icc (-π/4) (π/4), f x ≤ max ∧ f x ≥ min) ∧ 
                    max = sqrt 2 ∧ min = -1 :=
sorry

end smallest_positive_period_of_f_max_and_min_values_of_f_l37_37525


namespace find_f5_and_f_prime5_l37_37949

def tangent_line_at_point (f : ℝ → ℝ) (x : ℝ) : ℝ → ℝ := λ t, -t + 8

theorem find_f5_and_f_prime5 (f : ℝ → ℝ) (hf : ∀ x, deriv f x = -1) (hx : f 5 = 3) :
  f 5 + deriv f 5 = 2 :=
by
  simp [hx]
  rw [hf 5]
  norm_num
  sorry

end find_f5_and_f_prime5_l37_37949


namespace find_range_a_l37_37512

-- Define the proposition p
def p (m : ℝ) : Prop :=
1 < m ∧ m < 3 / 2

-- Define the proposition q
def q (m a : ℝ) : Prop :=
(m - a) * (m - (a + 1)) < 0

-- Define the sufficient but not necessary condition
def sufficient (a : ℝ) : Prop :=
(a ≤ 1) ∧ (3 / 2 ≤ a + 1)

theorem find_range_a (a : ℝ) :
  (∀ m, p m → q m a) → sufficient a → (1 / 2 ≤ a ∧ a ≤ 1) :=
sorry

end find_range_a_l37_37512


namespace complex_division_l37_37509

-- Define i as the imaginary unit
def i : Complex := Complex.I

-- Define the problem statement to prove that 2i / (1 - i) equals -1 + i
theorem complex_division : (2 * i) / (1 - i) = -1 + i :=
by
  -- Since we are focusing on the statement, we use sorry to skip the proof
  sorry

end complex_division_l37_37509


namespace increasing_function_conditions_l37_37131

theorem increasing_function_conditions (a b : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x < y → x < 4 → y < 4 → f(x) < f(y)) :
  4 ≤ a ∧ a < b :=
by
  -- Given: f(x) = (x - b) / (x - a)
  let f := λ x : ℝ, (x - b) / (x - a)
  -- f is increasing on the interval (-∞, 4)
  sorry

end increasing_function_conditions_l37_37131


namespace parallelogram_area_correct_l37_37697

noncomputable def area_of_parallelogram : ℂ :=
  let z1 := Complex.sqrtComplex (9 + 9 * sqrt 7 * Complex.I)
  let z2 := Complex.sqrtComplex (5 + 10 * sqrt 2 * Complex.I)
  let vertices := {z1, -z1, z2, -z2} in
  let [v1, v2] := vertices.to_list in
  Complex.abs (Complex.imag (v1 * Complex.conj v2))

theorem parallelogram_area_correct :
  area_of_parallelogram = 8 * sqrt 7 + 8 * sqrt 3 + 7 * sqrt 30 :=
by
  sorry

end parallelogram_area_correct_l37_37697


namespace population_net_increase_l37_37172

-- Defining the birth rates and death rates for the given time periods
def birth_rate : ℕ → ℕ 
| 0 := 4
| 1 := 8
| 2 := 10
| 3 := 6

def death_rate : ℕ → ℕ
| 0 := 3
| 1 := 3
| 2 := 4
| 3 := 2

-- Each period has 10800 two-second intervals
def intervals_per_period : ℕ := 10800

-- Net increase over one period
def net_increase_per_period (period : ℕ) : ℕ :=
  (birth_rate period - death_rate period) * intervals_per_period

-- Total net increase over all periods
def total_net_increase_in_one_day : ℕ :=
  (net_increase_per_period 0) +
  (net_increase_per_period 1) +
  (net_increase_per_period 2) +
  (net_increase_per_period 3)

-- The theorem to prove
theorem population_net_increase : total_net_increase_in_one_day = 172800 :=
  by
    sorry

end population_net_increase_l37_37172


namespace y_in_terms_of_x_l37_37079

theorem y_in_terms_of_x (m : ℤ) (x y : ℤ) 
  (h1 : x = 2^m + 1) (h2 : y = 3 + 2^(m+1)) : 
  y = 2 * x + 1 := 
by 
  sorry

end y_in_terms_of_x_l37_37079


namespace inequality_B_l37_37110

variable {x y : ℝ}

theorem inequality_B (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : x + 1 / (2 * y) > y + 1 / x :=
sorry

end inequality_B_l37_37110


namespace subset_condition_l37_37957

def A : Set ℝ := {x | x ≠ 1 ∧ x ≠ -1}
def B (a : ℝ) : Set ℝ := {x | x = a}

theorem subset_condition (a : ℝ) (h : B a ⊆ A) : a ≠ 1 ∧ a ≠ -1 :=
by
  unfold B at h
  unfold A at h
  exact h {x := a}

end subset_condition_l37_37957


namespace smallest_solution_abs_eq_20_l37_37426

theorem smallest_solution_abs_eq_20 : ∃ x : ℝ, x = -7 ∧ |4 * x + 8| = 20 ∧ (∀ y : ℝ, |4 * y + 8| = 20 → x ≤ y) :=
by
  sorry

end smallest_solution_abs_eq_20_l37_37426


namespace intersection_M_N_is_neq_neg1_0_1_l37_37222

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N_is_neq_neg1_0_1 :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_M_N_is_neq_neg1_0_1_l37_37222


namespace problem_ellipse_problem_triangle_area_l37_37499

-- Define the initial conditions and problem statement
theorem problem_ellipse (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3: a = Real.sqrt (2) * b) (h4 : a^2 - b^2 = 1) :
  (c : ℝ) (hc : c = 1) :
  ∃ (equation : ∀ x y : ℝ, x^2 / 2 + y^2 = 1),
  equation := 
begin
  use λ x y, x^2 / 2 + y^2,
  sorry
end

-- Define problem part 2
theorem problem_triangle_area (k : ℝ) :
  (∃ A B: ℝ × ℝ, ∃ P : ℝ × ℝ, 
  (A ≠ B) ∧ 
  P = (A + B) / 2 ∧ 
  (A_X := A.fst, A_Y := A.snd) (B_X := B.fst, B_Y := B.snd)  :
  a^2 = 2 / (b * Real.sqrt5)
    proof := 
begin
  use ⟨0,1⟩, ⟨-4 * k/(1 + 2 * k^2), 1-2k^2 /(1 + 2 * k^2),
  sorry
end

end problem_ellipse_problem_triangle_area_l37_37499


namespace cross_fills_space_without_gaps_l37_37074

structure Cube :=
(x : ℤ)
(y : ℤ)
(z : ℤ)

structure Cross :=
(center : Cube)
(adjacent : List Cube)

def is_adjacent (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ abs (c1.z - c2.z) = 1) ∨
  (c1.x = c2.x ∧ abs (c1.y - c2.y) = 1 ∧ c1.z = c2.z) ∨
  (abs (c1.x - c2.x) = 1 ∧ c1.y = c2.y ∧ c1.z = c2.z)

def valid_cross (c : Cross) : Prop :=
  ∀ (adj : Cube), adj ∈ c.adjacent → is_adjacent c.center adj

def fills_space (crosses : List Cross) : Prop :=
  ∀ (pos : Cube), ∃ (c : Cross), c ∈ crosses ∧ 
    (pos = c.center ∨ pos ∈ c.adjacent)

theorem cross_fills_space_without_gaps 
  (crosses : List Cross) 
  (Hcross : ∀ c ∈ crosses, valid_cross c) : 
  fills_space crosses :=
sorry

end cross_fills_space_without_gaps_l37_37074


namespace union_sets_l37_37505

-- Define the sets A and B based on the given conditions
def set_A : Set ℝ := {x | abs (x - 1) < 2}
def set_B : Set ℝ := {x | Real.log x / Real.log 2 < 3}

-- Problem statement: Prove that the union of sets A and B is {x | -1 < x < 9}
theorem union_sets : (set_A ∪ set_B) = {x | -1 < x ∧ x < 9} :=
by
  sorry

end union_sets_l37_37505


namespace area_triangle_PAB_range_lambda_l37_37536

noncomputable def pointA : ℝ × ℝ := (1, -3 / 2)
noncomputable def pointB : ℝ × ℝ := (-2, 0)
noncomputable def is_on_parabola (P : ℝ × ℝ) : Prop := P.snd = P.fst ^ 2
noncomputable def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.sqrt 3

theorem area_triangle_PAB {x_0 : ℝ} (h1 : in_interval x_0) (h2 : is_on_parabola (x_0, x_0^2)) :
    x_0 = 1 / 2 → 1 / 2 * (3 / 2) * 3 = 9 / 4 :=
  sorry

theorem range_lambda {x_0 : ℝ} (h1 : in_interval x_0) (h2 : is_on_parabola (x_0, x_0^2))
    (h3 : ∀ λ : ℝ, ⟪(pointB.fst - pointA.fst, pointB.snd - pointA.snd),
      ((x_0 - pointA.fst, x_0^2 - pointA.snd) + λ * (x_0 - pointB.fst, x_0^2 - pointB.snd))⟫ = 0) :
    1 / 2 ≤ λ ∧ λ ≤ 7 / 8 :=
  sorry

end area_triangle_PAB_range_lambda_l37_37536


namespace example_problem_l37_37470

def diamond (a b : ℕ) : ℕ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem example_problem : diamond 3 2 = 125 := by
  sorry

end example_problem_l37_37470


namespace evaluate_expression_l37_37521

def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then 2^(x - 1) else 2 - Real.log2 (4 - x)

theorem evaluate_expression :
  f (Real.log2 14) + f (-4) = 6 := by
  sorry

end evaluate_expression_l37_37521


namespace hyperbola_focus_distance_l37_37490
open Real

theorem hyperbola_focus_distance
  (a b : ℝ)
  (ha : a = 5)
  (hb : b = 3)
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (∃ M : ℝ × ℝ, M = (x, y)))
  (M : ℝ × ℝ)
  (hM_on_hyperbola : ∃ x y : ℝ, M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1)
  (F1_pos : ℝ)
  (h_dist_F1 : dist M (F1_pos, 0) = 18) :
  (∃ (F2_dist : ℝ), (F2_dist = 8 ∨ F2_dist = 28) ∧ dist M (F2_dist, 0) = F2_dist) := 
sorry

end hyperbola_focus_distance_l37_37490


namespace speed_of_the_stream_l37_37701

theorem speed_of_the_stream (d v_s : ℝ) :
  (∀ (t_up t_down : ℝ), t_up = d / (57 - v_s) ∧ t_down = d / (57 + v_s) ∧ t_up = 2 * t_down) →
  v_s = 19 := by
  sorry

end speed_of_the_stream_l37_37701


namespace relationship_between_a_b_c_l37_37939

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(x) = f(-x)

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f(x1) > f(x2)

def a : ℝ := f (Real.log 5 / Real.log 2)
def b : ℝ := f (2 ^ 0.2)
def c : ℝ := f (0.2 ^ 0.3)

theorem relationship_between_a_b_c
  (even_f : is_even f)
  (mono_dec_f : is_monotonically_decreasing f)
  (a_def : a = f (Real.log 5 / Real.log 2)) -- same as f (log2 0.2)
  (b_def : b = f (2 ^ 0.2))
  (c_def : c = f (0.2 ^ 0.3))
  (order : Real.log 5 / Real.log 2 > 2 ^ 0.2 ∧ 2 ^ 0.2 > 0.2 ^ 0.3) :
  a < b ∧ b < c := sorry

end relationship_between_a_b_c_l37_37939


namespace train_stops_12_minutes_per_hour_l37_37727

-- Definitions of the conditions
def speed_without_stoppages : ℝ := 45 -- in kmph
def speed_with_stoppages : ℝ := 36 -- in kmph

-- Prove the train stops for 12 minutes per hour
theorem train_stops_12_minutes_per_hour :
  ∃ (t : ℕ), t = 12 ∧ (t = 60 * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) :=
by
  have t := (60 * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages)
  use t.natAbs
  split
  case h₁ => sorry -- prove t.natAbs = 12
  case h₂ => sorry -- ensure t = 60 * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages

end train_stops_12_minutes_per_hour_l37_37727


namespace parabola_area_l37_37954

noncomputable def parabola_area_problem (x y : ℝ) (P : ℝ × ℝ) (O F : ℝ × ℝ) : Prop :=
  (O = (0, 0)) ∧
  (F = (2, 0)) ∧
  (P.1 = 3) ∧ 
  (P.2 = 2 * Real.sqrt 6) ∧
  (x^2 = 8 * y) ∧
  (Real.sqrt ((x - F.1) ^ 2 + (y - F.2) ^ 2) = 5) →
  (1/2 * Real.abs (F.1 * P.2) = 2 * Real.sqrt 6)

theorem parabola_area :
  ∃ x y : ℝ, ∃ P : ℝ × ℝ, ∃ O F : ℝ × ℝ,
    parabola_area_problem x y P O F :=
sorry

end parabola_area_l37_37954


namespace angle_congruence_parallelogram_l37_37739

theorem angle_congruence_parallelogram
  (A B C D O : Type)
  [parallelogram A B C D]
  (hO_inside: inside O A B C D)
  (h_angle_sum : ∠AOB + ∠DOC = π) :
  ∠CBO = ∠CDO :=
sorry

end angle_congruence_parallelogram_l37_37739


namespace pentagon_vertex_C_y_coordinate_l37_37233

-- Define the vertices of the pentagon
def vertex_A := (0, 0)
def vertex_B := (0, 5)
def vertex_D := (5, 5)
def vertex_E := (5, 0)

-- Define the area of a triangle given the base and height
def triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

-- Given conditions and concluding the proof
theorem pentagon_vertex_C_y_coordinate :
  ∃ y : ℝ, triangle_area 5 (y - 5) = 25 ∧ y = 15 :=
by
  -- By direct calculation
  use 15
  split
  any_goals sorry

end pentagon_vertex_C_y_coordinate_l37_37233


namespace distinct_real_roots_l37_37365

def operation (a b : ℝ) : ℝ := a^2 - a * b + b

theorem distinct_real_roots {x : ℝ} : 
  (operation x 3 = 5) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation x1 3 = 5 ∧ operation x2 3 = 5) :=
by 
  -- Add your proof here
  sorry

end distinct_real_roots_l37_37365


namespace train_length_l37_37822

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (conversion_factor : ℚ) :
  speed_kmh = 95 → time_s = 13 → conversion_factor = 5 / 18 →
  let speed_ms := speed_kmh * conversion_factor in
  let length := speed_ms * time_s in
  length = 343.07 := 
by
  intros h_speed h_time h_conversion
  -- Here would go the proof steps, but this is not required.
  sorry

end train_length_l37_37822


namespace find_platform_length_l37_37007

open Real

def speed_kmph_to_mps (v: ℝ) : ℝ := v * (1000 / 3600)

def total_distance (v: ℝ) (t: ℝ) : ℝ := v * t

def platform_length (total_dist: ℝ) (train_length: ℝ) : ℝ := total_dist - train_length

theorem find_platform_length (L_train: ℝ) (v_train: ℝ) (t: ℝ) (v_train_mps: ℝ := speed_kmph_to_mps v_train)
  (total_dist: ℝ := total_distance v_train_mps t) (L_platform: ℝ := platform_length total_dist L_train) :
  L_platform = 323.2 :=
by
  sorry

end find_platform_length_l37_37007


namespace intersection_points_eq_one_l37_37851

-- Definitions for the equations of the circles
def circle1 (x y : ℝ) : ℝ := x^2 + (y - 3)^2
def circle2 (x y : ℝ) : ℝ := x^2 + (y + 2)^2

-- The proof problem statement
theorem intersection_points_eq_one : 
  ∃ p : ℝ × ℝ, (circle1 p.1 p.2 = 9) ∧ (circle2 p.1 p.2 = 4) ∧
  (∀ q : ℝ × ℝ, (circle1 q.1 q.2 = 9) ∧ (circle2 q.1 q.2 = 4) → q = p) :=
sorry

end intersection_points_eq_one_l37_37851


namespace friends_signed_up_first_day_l37_37602

theorem friends_signed_up_first_day (F : ℕ) : 
  (5 + 10 * (F + 7) = 125) → F = 5 :=
by 
  intro h,
  sorry

end friends_signed_up_first_day_l37_37602


namespace area_of_rectangle_l37_37054

def length : ℕ := 4
def width : ℕ := 2

theorem area_of_rectangle : length * width = 8 :=
by
  sorry

end area_of_rectangle_l37_37054


namespace sum_of_roots_l37_37311

theorem sum_of_roots :
  let a := (-1 : ℝ)
      b := (-18 : ℝ)
      c := (36 : ℝ) in
  (a * (roots : ℝ) * (roots : ℝ) + b * (roots : ℝ) - c = 0) →
  let r := (-b / a) in
  r = 18 := sorry

end sum_of_roots_l37_37311


namespace part1_part2_l37_37581

noncomputable def a_n (n : ℕ) : ℕ := 1 - n

def S_n : ℕ → ℕ 
| 0     := 0
| (n+1) := S_n n + a_n n

theorem part1 (n : ℕ) (S_n : ℕ → ℕ) (h₀ : ∀ n, 2 * S_n n = n - n^2) :
    a_n n = 1 - n :=
sorry

noncomputable def b_n (n : ℕ) : ℝ :=
if even n then
  1 / (n^2 + 2 * n : ℝ)
else
  n * 2^(1 - n)

def T_2n (n : ℕ) : ℝ := 
  (finset.range (2 * n)).sum (λ x, b_n (x + 1))

theorem part2 (n : ℕ) :
  T_2n n = (20 / 9 : ℝ) - (24 * n + 20 : ℝ) / (9 * 2^(2 * n : ℝ)) + (n / (4 * (n + 1) : ℝ)) :=
sorry

end part1_part2_l37_37581


namespace centroid_of_Harry_Sandy_Luna_l37_37149

-- Define the coordinates of Harry, Sandy, and Luna.
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 5)
def Luna : ℝ × ℝ := (-2, 9)

-- Define the centroid function for three given points.
def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- State that the centroid of the coordinates is \((10/3, 11/3)\).
theorem centroid_of_Harry_Sandy_Luna :
  centroid Harry Sandy Luna = (10 / 3, 11 / 3) :=
by
  sorry

end centroid_of_Harry_Sandy_Luna_l37_37149


namespace draw_4_balls_in_order_ways_l37_37775

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37775


namespace percent_research_and_development_is_9_l37_37793

-- Define given percentages
def percent_transportation := 20
def percent_utilities := 5
def percent_equipment := 4
def percent_supplies := 2

-- Define degree representation and calculate percent for salaries
def degrees_in_circle := 360
def degrees_salaries := 216
def percent_salaries := (degrees_salaries * 100) / degrees_in_circle

-- Define the total percentage representation
def total_percent := 100
def known_percent := percent_transportation + percent_utilities + percent_equipment + percent_supplies + percent_salaries

-- Calculate the percent for research and development
def percent_research_and_development := total_percent - known_percent

-- Theorem statement
theorem percent_research_and_development_is_9 : percent_research_and_development = 9 :=
by 
  -- Placeholder for actual proof
  sorry

end percent_research_and_development_is_9_l37_37793


namespace no_valid_partition_exists_l37_37589

theorem no_valid_partition_exists :
  ¬ ∃ (n : ℕ), n > 1 ∧ ∃ (A : ℕ → Set ℕ), 
    (∀ i, i < n → ∃ x, x ∈ A i) ∧ 
    ∀ (i : ℕ) (h : i < n), ∃ (B : Fin (n-1) → ℕ), 
      (∀ j, B j ∈ A (if j.val < i then j.val else j.val + 1)) ∧ 
      (∑ j, B j) ∈ A i := 
sorry

end no_valid_partition_exists_l37_37589


namespace number_of_teachers_in_school_l37_37995

-- Definitions based on provided conditions
def number_of_girls : ℕ := 315
def number_of_boys : ℕ := 309
def total_number_of_people : ℕ := 1396

-- Proof goal: Number of teachers in the school
theorem number_of_teachers_in_school : 
  total_number_of_people - (number_of_girls + number_of_boys) = 772 :=
by
  sorry

end number_of_teachers_in_school_l37_37995


namespace concert_ticket_price_l37_37243

theorem concert_ticket_price (x : ℝ) :
  let ticket_cost := 2 * x
  let processing_fee := 0.30 * x
  let parking_fee := 10
  let entrance_fee := 2 * 5
  (ticket_cost + processing_fee + parking_fee + entrance_fee = 135) → x = 50 :=
by
  intro h
  have h1 : 2 * x + 0.30 * x + 10 + 10 = 135 := h
  have h2 : 2.30 * x + 20 = 135 := by ring_nf at h1; exact h1
  have h3 : 2.30 * x = 115 := by linarith
  have h4 : x = 50 := by field_simp [h3, mul_comm 2.30 50, div_self]
  exact h4

end concert_ticket_price_l37_37243


namespace min_cards_for_certain_event_l37_37910

-- Let's define the deck configuration
structure DeckConfig where
  spades : ℕ
  clubs : ℕ
  hearts : ℕ
  total : ℕ

-- Define the given condition of the deck
def givenDeck : DeckConfig := { spades := 5, clubs := 4, hearts := 6, total := 15 }

-- Predicate to check if m cards drawn guarantees all three suits are present
def is_certain_event (m : ℕ) (deck : DeckConfig) : Prop :=
  m >= deck.spades + deck.hearts + 1

-- The main theorem to prove the minimum number of cards m
theorem min_cards_for_certain_event : ∀ m, is_certain_event m givenDeck ↔ m = 12 :=
by
  sorry

end min_cards_for_certain_event_l37_37910


namespace combination_simplify_l37_37714

theorem combination_simplify (n : ℕ) (h : 0 < n) :
    (finset.range (n + 1)).sum (λ k, (C(n, k) * (1 / (k + 1) * (1 / 5) ^ (k + 1)))) = 
    (1 / (n + 1)) * ((6 / 5) ^ (n + 1) - 1) := 
sorry

end combination_simplify_l37_37714


namespace solution_correct_l37_37907

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  x^2 - 36 * x + 320 ≤ 16

theorem solution_correct (x : ℝ) : quadratic_inequality_solution x ↔ 16 ≤ x ∧ x ≤ 19 :=
by sorry

end solution_correct_l37_37907


namespace smallest_base_for_150_l37_37309

theorem smallest_base_for_150 : ∃ b : ℕ, (b^2 ≤ 150 ∧ 150 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 150 ∧ 150 < n^3) → n ≥ b :=
by
  let b := 6
  have h1 : b^2 ≤ 150 := by norm_num
  have h2 : 150 < b^3 := by norm_num
  have h3 : ∀ n : ℕ, n^2 ≤ 150 → 150 < n^3 → n ≥ b :=
    by
      intro n hn1 hn2
      have : n ≥ 6 := by sorry  -- Detailed proof omitted
      exact this
  exact ⟨b, ⟨h1, h2⟩, h3⟩

end smallest_base_for_150_l37_37309


namespace simplify_expression_l37_37731

theorem simplify_expression :
  (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + x = 12 → x = 63 :=
by
  intro h
  sorry

end simplify_expression_l37_37731


namespace range_of_x_if_cos2_gt_sin2_l37_37968

theorem range_of_x_if_cos2_gt_sin2 (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) (h2 : Real.cos x ^ 2 > Real.sin x ^ 2) :
  x ∈ Set.Ico 0 (Real.pi / 4) ∪ Set.Ioc (3 * Real.pi / 4) Real.pi :=
by
  sorry

end range_of_x_if_cos2_gt_sin2_l37_37968


namespace ratio_of_initial_games_to_gifts_l37_37239

-- Define the total number of gifts
def gifts : ℕ := 12 + 8

-- Define the total number of games Radhika owns
def total_games : ℕ := 30

-- Define the number of games Radhika initially owned
def initial_games := total_games - gifts

-- Prove the ratio of initial games to the number of gifts is 1:2
theorem ratio_of_initial_games_to_gifts : initial_games * 2 = gifts := by
  calc
    initial_games = total_games - gifts : rfl
    _ = 30 - 20 : rfl
    _ = 10 : rfl
    initial_games * 2 = 10 * 2 : rfl
    _ = 20 : rfl
    gifts = 20 : rfl
    sorry -- The proof continues here

#reduce gifts -- Outputs 20
#reduce total_games -- Outputs 30
#reduce initial_games -- Outputs 10

end ratio_of_initial_games_to_gifts_l37_37239


namespace tangent_line_at_slope_two_l37_37690

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l37_37690


namespace ramesh_share_correct_l37_37657

-- Define basic conditions
def suresh_investment := 24000
def ramesh_investment := 40000
def total_profit := 19000

-- Define Ramesh's share calculation
def ramesh_share : ℤ :=
  let ratio_ramesh := ramesh_investment / (suresh_investment + ramesh_investment)
  ratio_ramesh * total_profit

-- Proof statement
theorem ramesh_share_correct : ramesh_share = 11875 := by
  sorry

end ramesh_share_correct_l37_37657


namespace correct_statement_D_l37_37320

-- Definitions based on the conditions in the problem
def precision_thousand (x : ℕ) : Prop :=
  x % 1000 = 0

def precision_tenth (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n / 10

def rounded_to_nearest_thousand (x y : ℕ) : Prop :=
  abs (x - y) < 500

def representation_ten_thousand (x : ℕ) (y : ℚ) : Prop :=
  y * 10000 = x

-- Theorem stating that D is the correct statement given the conditions
theorem correct_statement_D :
  rounded_to_nearest_thousand 317500 318000 ∧ representation_ten_thousand 318000 31.8 :=
by
  sorry

end correct_statement_D_l37_37320


namespace harmonic_log_inequality_l37_37741

theorem harmonic_log_inequality (n : ℕ) (h : n ≥ 2) :
  let S := ∑ i in finset.range (n-1), (1 : ℝ) / (i + 1)
  let Sn := ∑ i in finset.range n, (1 : ℝ) / (i + 1)
  S * real.log (n + 1) > Sn * real.log n :=
by
  sorry

end harmonic_log_inequality_l37_37741


namespace value_of_k_l37_37556

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l37_37556


namespace marco_strawberries_weight_l37_37629

theorem marco_strawberries_weight 
  (m : ℕ) 
  (total_weight : ℕ := 40) 
  (dad_weight : ℕ := 32) 
  (h : total_weight = m + dad_weight) : 
  m = 8 := 
sorry

end marco_strawberries_weight_l37_37629


namespace extreme_points_of_f_max_min_f_on_interval_l37_37946

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x^2 + 5 * x - 2 * Real.log x

theorem extreme_points_of_f :
  (∀ x > 0, f' x = x^2 - 4 * x + 5 - 2 / x → (x = 2) ∧ ∀ y ≠ x, f y > f x) :=
begin
  -- Proof that the function has a local minimum at x=2 and no local maximum
  sorry
end

theorem max_min_f_on_interval :
  (∀ x ∈ ℝ, 1 ≤ x ∧ x ≤ 3 → f x ≥ f 2 ∨ f x ≤ f 2) ∧
  (f 1 = 10 / 3) ∧ (f 3 = 6 - 2 * Real.log 3) ∧ (f 2 = 14 / 3 - 2 * Real.log 2) ∧
  (f 3 > f 1) :=
begin
  -- Proof of the maximum and minimum values on the interval [1, 3]
  sorry
end

end extreme_points_of_f_max_min_f_on_interval_l37_37946


namespace triangle_proof_l37_37183

-- Declare a structure for a triangle with given conditions
structure TriangleABC :=
  (a b c : ℝ) -- sides opposite to angles A, B, and C
  (A B C : ℝ) -- angles A, B, and C
  (R : ℝ) -- circumcircle radius
  (r : ℝ := 3) -- inradius is given as 3
  (area : ℝ := 6) -- area of the triangle is 6
  (h1 : a * Real.cos A + b * Real.cos B + c * Real.cos C = R / 3) -- given condition
  (h2 : ∀ a b c A B C, a * Real.sin A + b * Real.sin B + c * Real.sin C = 2 * area / (a+b+c)) -- implied area condition

-- Define the theorem using the above conditions
theorem triangle_proof (t : TriangleABC) :
  t.a + t.b + t.c = 4 ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C)) = 1/3 ∧
  t.R = 6 :=
by
  sorry

end triangle_proof_l37_37183


namespace trapezoid_angles_equal_l37_37790

-- Defining trapezoid and properties
structure Trapezoid (A B C D : Type) :=
  (is_good : Bool)
  (is_cyclic : Bool)
  (AB CD : ℝ)
  (AD : ℝ)
  (par_AB_CD : AB > CD)
  (angle_BAD : ℝ)

-- Defining tangents and intersection points
def tangents {A B C D : Type} (t : Trapezoid A B C D) (S E F : Type) : Prop := sorry

-- Main theorem stating the equivalent condition
theorem trapezoid_angles_equal 
  {A B C D S E F : Type} 
  (t : Trapezoid A B C D)
  (hgood_trapezoid : t.is_good)
  (hcyclic : t.is_cyclic)
  (htangents : tangents t S E F)
  (hpar_AB_CD : t.par_AB_CD):
  (∠ BSE = ∠ FSC) ↔ (t.angle_BAD = 60.0 ∨ t.AD = t.AB) := 
sorry

end trapezoid_angles_equal_l37_37790


namespace sin_double_angle_tan_angle_subtraction_l37_37506

-- Defining the conditions and translating to Lean definitions
variables {α β : Real} {sin cos : Real → Real}
variables (hcos_alpha : cos α = -3/5) (h_alpha_range : π < α ∧ α < 2 * π)
variables (h_point_beta : ∃ (x y : Real), x = 3 ∧ y = -1 ∧ tan β = y / x)

-- Statement of the proof
theorem sin_double_angle : sin (2 * α) = 24 / 25 :=
sorry

theorem tan_angle_subtraction : tan (α - β) = 3 :=
sorry

end sin_double_angle_tan_angle_subtraction_l37_37506


namespace find_length_of_b_l37_37584

noncomputable def length_of_b {a c : ℝ} (angleB : ℝ) (b : ℝ) : Prop :=
  (a = 3 * Real.sqrt 3) ∧ (c = 2) ∧ (angleB = 150) ∧ (b = Real.sqrt ((a^2) + (c^2) - 2 * a * c * (Real.cos (angleB * Real.pi / 180))))

theorem find_length_of_b : ∃ b : ℝ, length_of_b 3 * Real.sqrt 3 2 150 b ∧ b = 7 :=
by
  sorry

end find_length_of_b_l37_37584


namespace train_pass_time_correct_l37_37821

def train_length : ℕ := 375 -- in meters
def train_speed : ℕ := 72   -- in kmph
def man_speed : ℕ := 12     -- in kmph
def time_to_pass (train_length : ℕ) (train_speed : ℕ) (man_speed : ℕ) : ℕ := 22.5 -- in seconds

theorem train_pass_time_correct :
  ∀ (train_length train_speed man_speed : ℕ),
    train_speed = 72 → man_speed = 12 → train_length = 375 →
    time_to_pass train_length train_speed man_speed = 22.5 :=
by {
  intros,
  sorry
}

end train_pass_time_correct_l37_37821


namespace directrix_eqn_of_parabola_l37_37875

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37875


namespace max_good_operations_min_good_operations_l37_37428

/-- Define the maximum number of good operations given initial rope length 2018 -/
theorem max_good_operations (initial_length : ℕ) 
    (h : initial_length = 2018) : 
    ∃ k : ℕ, 2 * k - 2 = 2016 := 
sorry

/-- Define the minimum number of good operations and prove number of different lengths are the same -/
theorem min_good_operations (initial_length : ℕ) 
    (h : initial_length = 2018) 
    : ∃ (configs : list ℕ), 
        (∀ k : ℕ, 1 ≤ k → k ≤ length configs → ∃ s : set ℕ, s = { x ∈ configs | x = length k}) := 
sorry

end max_good_operations_min_good_operations_l37_37428


namespace car_speed_graph_condition_l37_37028

-- Definitions based on conditions
constant M_speed : ℝ
constant N_speed : ℝ := 3 * M_speed
constant M_start_time : ℝ := 0
constant N_start_time : ℝ := 2
constant d : ℝ := M_speed * 3 -- since Car M travels for 3 hours

-- Condition:
-- Car M travels at a constant speed M_speed for 3 hours
constant M_distance_travelled : ∀ (t : ℝ), t >= 0 → t <= 3 → M_speed * t = d
-- Car N travels at a constant speed N_speed for 1 hour, starting at time 2 hours
constant N_distance_travelled : ∀ (t : ℝ), t >= 2 → t <= 3 → N_speed * (t - 2) = d

-- Prove that Car N's speed versus time graph starts later and is three times higher than Car M's
theorem car_speed_graph_condition : 
  ∀ (t : ℝ),
    (M_start_time ≤ t ∧ t ≤ 3 → M_distance_travelled t) ∧
    (N_start_time ≤ t ∧ t ≤ 3 → N_distance_travelled t) →
    N_start_time > M_start_time ∧ N_speed = 3 * M_speed := 
by {
  sorry
}

end car_speed_graph_condition_l37_37028


namespace sahil_transportation_charges_l37_37242

def transportation_charge (purchase_price repair_cost selling_price profit_rate : ℝ) : ℝ :=
  let total_cost_before_transportation := purchase_price + repair_cost
  let cost_price := selling_price / (1 + profit_rate)
  cost_price - total_cost_before_transportation

theorem sahil_transportation_charges :
  transportation_charge 10000 5000 24000 0.5 = 1000 := by
  sorry

end sahil_transportation_charges_l37_37242


namespace max_solution_inequality_l37_37441

noncomputable def t (x : ℝ) : ℝ := 80 - 2 * x * real.sqrt (30 - 2 * x)
noncomputable def numerator (x : ℝ) : ℝ := - real.logb 3 (t x) ^ 2 + (abs (real.logb 3 (t x) - 3 * real.logb 3 (x ^ 2 - 2 * x + 29)))
noncomputable def denominator (x : ℝ) : ℝ := 7 * real.logb 7 (65 - 2 * x * real.sqrt (30 - 2 * x)) - 4 * real.logb 3 (t x)
noncomputable def inequality_expr (x : ℝ) : ℝ := numerator x / denominator x

theorem max_solution_inequality : 
  ∃ x, (x = 8 - real.sqrt 13) ∧ (inequality_expr x ≥ 0) := 
begin
  use 8 - real.sqrt 13,
  split,
  { refl },
  { sorry }
end

end max_solution_inequality_l37_37441


namespace collinear_X_Y_Z_l37_37674

variables {A B C D E F P Q R X Y Z : Type}
variables [Incircle : circle ABC] [tangent_D : tangent circle ABC D]
variables [tangent_E : tangent circle ABC E] [tangent_F : tangent circle ABC F]
variables [circumcircle_AEF : circle (set.triangle A E F)]
variables [tangent_P : tangent circumcircle_AEF P]
variables [concurrent_AX : concurrent {EF, DP, AX}]
variables [concurrent_BY : concurrent {EF, EQ, BY}]
variables [concurrent_CZ : concurrent {EF, FR, CZ}]

-- statement to prove X, Y, Z are collinear
theorem collinear_X_Y_Z : collinear {X, Y, Z} := 
sorry

end collinear_X_Y_Z_l37_37674


namespace quadratic_equation_with_given_root_l37_37841

theorem quadratic_equation_with_given_root : 
  ∃ p q : ℤ, (∀ x : ℝ, x^2 + (p : ℝ) * x + (q : ℝ) = 0 ↔ x = 2 - Real.sqrt 7 ∨ x = 2 + Real.sqrt 7) 
  ∧ (p = -4) ∧ (q = -3) :=
by
  sorry

end quadratic_equation_with_given_root_l37_37841


namespace price_increase_percentage_l37_37601

variables
  (coffees_daily_before : ℕ := 4)
  (price_per_coffee_before : ℝ := 2)
  (coffees_daily_after : ℕ := 2)
  (price_increase_savings : ℝ := 2)
  (spending_before := coffees_daily_before * price_per_coffee_before)
  (spending_after := spending_before - price_increase_savings)
  (price_per_coffee_after := spending_after / coffees_daily_after)

theorem price_increase_percentage :
  ((price_per_coffee_after - price_per_coffee_before) / price_per_coffee_before) * 100 = 50 :=
by
  sorry

end price_increase_percentage_l37_37601


namespace number_of_ways_to_draw_balls_l37_37783

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37783


namespace probability_point_P_in_small_spheres_l37_37372

noncomputable def regular_tetrahedron_probability (a : ℝ) : ℝ :=
  let R1 := (ℝ.sqrt 6) / 12 * a
  let R2 := (ℝ.sqrt 6) / 4 * a
  let r := (R2 - R1) / 2
  let V_inner := (4 / 3) * ℝ.pi * R1^3
  let V_cap := (4 / 3) * ℝ.pi * r^3
  let V_total := V_inner + 4 * V_cap
  let V_outer := (4 / 3) * ℝ.pi * R2^3
  (V_total / V_outer)

theorem probability_point_P_in_small_spheres (a : ℝ) : 
  0 ≤ a → regular_tetrahedron_probability a = 5 / 27 :=
by
  intros
  sorry

end probability_point_P_in_small_spheres_l37_37372


namespace evaluate_expression_correct_l37_37860

noncomputable def evaluate_expression : ℚ :=
  (List.prod (List.map (λ n, 1 - (1 / n)) (List.range' 2 100))) * 2

theorem evaluate_expression_correct :
  evaluate_expression = 1 / 50 :=
by
  sorry

end evaluate_expression_correct_l37_37860


namespace prove_alpha_n_lt_three_sixteenth_prove_alpha_n_gt_seven_forty_l37_37740

theorem prove_alpha_n_lt_three_sixteenth (α : ℝ) (hα : 0 < α ∧ α < 1 / 2 ∧ irrational α) :
  ∃ n : ℕ, let α_n := (nat.iterate (λ x, min (2 * x) (1 - 2 * x)) n α)
  in α_n < 3 / 16 :=
sorry

theorem prove_alpha_n_gt_seven_forty (α : ℝ) (hα : 0 < α ∧ α < 1 / 2 ∧ irrational α) :
  (∀ n : ℕ, let α_n := (nat.iterate (λ x, min (2 * x) (1 - 2 * x)) n α)
  in α_n > 7 / 40) → false :=
sorry

end prove_alpha_n_lt_three_sixteenth_prove_alpha_n_gt_seven_forty_l37_37740


namespace fraction_of_earth_surface_humans_can_inhabit_l37_37977

theorem fraction_of_earth_surface_humans_can_inhabit :
  (1 / 3) * (2 / 3) = (2 / 9) :=
by
  sorry

end fraction_of_earth_surface_humans_can_inhabit_l37_37977


namespace find_reflection_point_l37_37811

noncomputable theory

variables (A C B : ℝ × ℝ × ℝ)
variable (n : ℝ × ℝ × ℝ := (1, 1, 1))

-- Define the points A and C
def point_A : ℝ × ℝ × ℝ := (-2, 8, 10)
def point_C : ℝ × ℝ × ℝ := (2, 4, 8)

-- Define the plane equation
def plane_eq (P : ℝ × ℝ × ℝ) : Prop := P.1 + P.2 + P.3 = 10

-- Define the point B that needs to be proven
def point_B : ℝ × ℝ × ℝ := (10/13, 60/13, 100/13)

-- The proof goal
theorem find_reflection_point (hPlaneA : plane_eq A) (hPlaneC : plane_eq C) :
  B = point_B := by
  sorry

end find_reflection_point_l37_37811


namespace ninety_third_term_is_2_13_l37_37229

noncomputable def sequence_term (m : ℕ) (n : ℕ) : (ℕ × ℕ) :=
  if h1 : m > 0 ∧ n > 0 ∧ n ≤ m then (n, m - n + 1) else (1, 1)

noncomputable def find_group (k : ℕ) : ℕ × ℕ :=
  let n := Nat.find (λ n, k ≤ (n * (n + 1)) / 2)
  let term_pos := k - (n * (n - 1)) / 2
  (n, term_pos)

theorem ninety_third_term_is_2_13 :
  let (group, pos) := find_group 93 in
  (sequence_term group pos) = (2, 13) :=
by
  let (group, pos) := (find_group 93)
  have h_group : group = 14 := sorry
  have h_pos : pos = 2 := sorry
  rw [h_group, h_pos]
  exact rfl

end ninety_third_term_is_2_13_l37_37229


namespace identify_counterfeit_coin_l37_37909

theorem identify_counterfeit_coin (coins : Fin 15 → ℝ) (h1 : ∃ i, coins i ≠ 1) :
  ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (2 weighings) :=
by
  -- Proof omitted
  sorry

end identify_counterfeit_coin_l37_37909


namespace find_equation_of_ellipse_fixed_point_exists_l37_37090

-- Definitions and conditions for part (1)
def ellipse_center_origin : Prop :=
  (0, 0) = (0, 0) -- Center at origin

def ellipse_eccentricity : Prop :=
  ∃ a b : ℝ, a = sqrt 2 ∧ b = 1 ∧ (sqrt (a^2 - b^2) / a) = sqrt 2 / 2 -- Eccentricity condition

def ellipse_focus_condition : Prop :=
  ∃ c a : ℝ, c = 1 ∧ a = sqrt 2 -- Focus condition

-- Statement for part (1)
theorem find_equation_of_ellipse :
  ellipse_center_origin ∧ ellipse_eccentricity ∧ ellipse_focus_condition →
  ∃ a b : ℝ, a = sqrt 2 ∧ b = 1 ∧ (∀ x y : ℝ, (y^2 / a^2) + (x^2 / b^2) = 1) :=
by sorry

-- Definitions and conditions for part (2)
def line_through_point_S : Prop :=
  ∃ l : ℝ → ℝ, l (-1/3) = 0 -- Line l passing through S

def intersect_ellipse_at_AB : Prop :=
  ∃ A B : ℝ × ℝ, ∀ k : ℝ, l (-1/3) = 0 → (y = k * (x + 1/3)) ∧ ((y^2 / 2) + x^2 = 1) -- Intersection condition

-- Statement for part (2)
theorem fixed_point_exists :
  line_through_point_S ∧ intersect_ellipse_at_AB →
  ∃ T : ℝ × ℝ, T = (1, 0) ∧ (∀ A B : ℝ × ℝ, circle_diameter_AB A B = true) :=
by sorry

end find_equation_of_ellipse_fixed_point_exists_l37_37090


namespace parabola_directrix_l37_37881

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37881


namespace draw_4_balls_in_order_l37_37779

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37779


namespace number_of_ways_to_draw_balls_l37_37787

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37787


namespace no_negative_range_l37_37287

-- Defining the function and conditions
def f (x a : ℝ) : ℝ := x^2 - 2 * x + a

theorem no_negative_range (a : ℝ) :
  (∃ y, ∀ x, y = log (f x a) → y ∈ (-∞, 0]) → false :=
by
  sorry

end no_negative_range_l37_37287


namespace zero_of_f_in_interval_l37_37675

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.exp x

theorem zero_of_f_in_interval : 
  ∃ c ∈ Ioo (0 : ℝ) (1/e), f c = 0 :=
by
  sorry

end zero_of_f_in_interval_l37_37675


namespace solve_for_k_l37_37554

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l37_37554


namespace find_x_l37_37530

theorem find_x (x : ℤ) (A : Set ℤ) (B : Set ℤ) (hA : A = {1, 4, x}) (hB : B = {1, 2 * x, x ^ 2}) (hinter : A ∩ B = {4, 1}) : x = -2 :=
sorry

end find_x_l37_37530


namespace vector_combination_l37_37223

variable (A B Q : Type) [AddCommGroup A]
variable (toVec : B → A)
variable (linear_comb : A → B → B → B)

def BQ_AB_ratio : Prop := ∀ (b a : B), linear_comb (toVec Q - toVec b) (by of_rat 7) (toVec a - toVec b) (by of_rat (-2))

theorem vector_combination (A B Q : Type) [AddCommGroup A]
    (toVec : B → A) (linear_comb : A → B → B → B)
    [h : BQ_AB_ratio A B Q toVec linear_comb]
    : toVec Q = - (7 / 2) • toVec A + (9 / 2) • toVec B :=
sorry

end vector_combination_l37_37223


namespace tan_double_angle_l37_37468

theorem tan_double_angle (α : ℝ) (h1 : Real.cos (Real.pi - α) = 4 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_l37_37468


namespace gcd_245_1001_l37_37306

-- Definitions based on the given conditions

def fact245 : ℕ := 5 * 7^2
def fact1001 : ℕ := 7 * 11 * 13

-- Lean 4 statement of the proof problem
theorem gcd_245_1001 : Nat.gcd fact245 fact1001 = 7 :=
by
  -- Add the prime factorizations as assumptions
  have h1: fact245 = 245 := by sorry
  have h2: fact1001 = 1001 := by sorry
  -- The goal is to prove the GCD
  sorry

end gcd_245_1001_l37_37306


namespace value_of_J_l37_37931

-- Given conditions
variables (Y J : ℤ)

-- Condition definitions
axiom condition1 : 150 < Y ∧ Y < 300
axiom condition2 : Y = J^2 * J^3
axiom condition3 : ∃ n : ℤ, Y = n^3

-- Goal: Value of J
theorem value_of_J : J = 3 :=
by { sorry }  -- Proof omitted

end value_of_J_l37_37931


namespace square_pizza_area_larger_by_27_percent_l37_37351

theorem square_pizza_area_larger_by_27_percent :
  let r := 5
  let A_circle := Real.pi * r^2
  let s := 2 * r
  let A_square := s^2
  let delta_A := A_square - A_circle
  let percent_increase := (delta_A / A_circle) * 100
  Int.floor (percent_increase + 0.5) = 27 :=
by
  sorry

end square_pizza_area_larger_by_27_percent_l37_37351


namespace number_of_ways_to_draw_balls_l37_37786

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37786


namespace multiple_of_2_and_3_is_divisible_by_6_l37_37467

theorem multiple_of_2_and_3_is_divisible_by_6 (n : ℤ) (h1 : n % 2 = 0) (h2 : n % 3 = 0) : n % 6 = 0 :=
sorry

end multiple_of_2_and_3_is_divisible_by_6_l37_37467


namespace n_greater_than_1788_l37_37621

theorem n_greater_than_1788 (n : ℕ) (h1 : 0 < n) 
  (h2 : ∀ (S : Finset ℕ), S.card = n → (S ∈  Finset.powerset_univ _ → ∃ (a d : ℕ), a + 28 * d < 1988 ∧ ∀ i ∈ S, (i = a + d * i)) : n > 1788 := 
sorry

end n_greater_than_1788_l37_37621


namespace proof_a6_bounds_l37_37070

theorem proof_a6_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 :=
by
  sorry

end proof_a6_bounds_l37_37070


namespace problem1_problem2_problem3_l37_37129

-- Condition: Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (2 * x + (π / 3)) + 1

-- Problem 1: Symmetry about the line x = t, minimum value of t
theorem problem1 (t : ℝ) (ht : t > 0) :
  (∀ x, f (2 * t - x) = f x) → t = π / 12 :=
by sorry

-- Problem 2: The equation mf(x_0) - 2 = 0 having two solutions in the given interval
theorem problem2 (m : ℝ) :
  (∃ x₀₁ x₀₂ ∈ set.Icc (-π / 12) (π / 6), x₀₁ ≠ x₀₂ ∧ m * f x₀₁ - 2 = 0 ∧ m * f x₀₂ - 2 = 0) →
  (2 / 3 < m ∧ m ≤ sqrt 3 - 1) :=
by sorry

-- Problem 3: Interval [a, b] having at least 6 zeros of y = f(x), minimum value of b - a
theorem problem3 (a b : ℝ) (ha : a < b) :
  (∃ xs : list ℝ, list.length xs ≥ 6 ∧ (∀ x ∈ xs, a ≤ x ∧ x ≤ b ∧ f x = 0)) →
  b - a ≥ 7 * π / 3 :=
by sorry

end problem1_problem2_problem3_l37_37129


namespace no_odd_vertices_closed_polygon_l37_37641

theorem no_odd_vertices_closed_polygon (
  (vertices : List (ℚ × ℚ))
  (h_odd_number : vertices.length % 2 = 1)
  (h_closed : vertices.head = vertices.last)
  (h_unit_length : ∀ (i j : ℕ), i < vertices.length → j < vertices.length → (vertices.get_or_else i (0,0) - vertices.get_or_else j (0,0)).norm = 1)
) : false :=
sorry

end no_odd_vertices_closed_polygon_l37_37641


namespace band_males_not_in_orchestra_l37_37259

theorem band_males_not_in_orchestra :
  ∀ (F_band F_orchestra M_band M_orchestra F_both total_students : ℕ),
    F_band = 100 →
    M_band = 80 →
    F_orchestra = 80 →
    M_orchestra = 100 →
    F_both = 60 →
    total_students = 230 →
    let F_either := (F_band + F_orchestra - F_both),
        M_either := (total_students - F_either),
        M_both := (M_band + M_orchestra - M_either)
    in (M_band - M_both) = 10 :=
by
  intros F_band F_orchestra M_band M_orchestra F_both total_students
    hF_band hM_band hF_orchestra hM_orchestra hF_both htotal_students
  let F_either := (F_band + F_orchestra - F_both)
  let M_either := (total_students - F_either)
  let M_both := (M_band + M_orchestra - M_either)
  exact Eq.refl (M_band - M_both)

end band_males_not_in_orchestra_l37_37259


namespace transformed_stats_l37_37924

variables {α : Type*} [field α]

-- Assume an arbitrary data set (x_1, ..., x_5) over type α with mean and variance conditions
variables (x1 x2 x3 x4 x5 : α)

-- Definitions for mean and variance
def mean (x1 x2 x3 x4 x5 : α) : α := (x1 + x2 + x3 + x4 + x5) / 5
def variance (x1 x2 x3 x4 x5 : α) : α := (1 / 5) * ((x1 - mean x1 x2 x3 x4 x5) ^ 2 + (x2 - mean x1 x2 x3 x4 x5) ^ 2 + (x3 - mean x1 x2 x3 x4 x5) ^ 2 + (x4 - mean x1 x2 x3 x4 x5) ^ 2 + (x5 - mean x1 x2 x3 x4 x5) ^ 2)

-- Assumptions for conditions
def original_mean (x1 x2 x3 x4 x5 : α) : Prop := mean x1 x2 x3 x4 x5 = 2
def original_variance (x1 x2 x3 x4 x5 : α) : Prop := variance x1 x2 x3 x4 x5 = 1 / 3

-- Transform the original data
def transform (x : α) : α := 3 * x - 2

-- Prove the mean and variance of the transformed data set
theorem transformed_stats (x1 x2 x3 x4 x5 : α) (h_mean : original_mean x1 x2 x3 x4 x5) (h_variance : original_variance x1 x2 x3 x4 x5) :
  mean (transform x1) (transform x2) (transform x3) (transform x4) (transform x5) = 4 ∧ variance (transform x1) (transform x2) (transform x3) (transform x4) (transform x5) = 3 :=
by sorry

end transformed_stats_l37_37924


namespace circumference_of_course_l37_37009

-- Define the speeds of Ajith and Rana, and the time taken to meet again
def Ajith_speed : ℝ := 4
def Rana_speed : ℝ := 5
def meet_time : ℝ := 115

-- Define the relationship to prove
theorem circumference_of_course :
  let relative_speed := Rana_speed - Ajith_speed in
  let distance_gained := relative_speed * meet_time in
  distance_gained = 115 := by
  sorry

end circumference_of_course_l37_37009


namespace ball_drawing_ways_l37_37767

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37767


namespace sum_of_modified_numbers_l37_37798

variable (n : ℕ) (s : ℕ)

theorem sum_of_modified_numbers (n : ℕ) (s : ℕ) (original_set : Fin n → ℕ) 
  (h_sum : (∑ i : Fin n, original_set i) = s) :
  (∑ i : Fin n, (3 * (original_set i + 10) - 15)) = 3 * s + 15 * n :=
by
  sorry

end sum_of_modified_numbers_l37_37798


namespace proof_equivalent_problem_l37_37937

variable (A : ℝ × ℝ) (B : ℝ × ℝ)
variable (l : Line)
variable (C : ℝ × ℝ)
variable (area_ABC : ℝ)

def is_reflection_point (A B C : ℝ × ℝ) (l : Line) : Prop :=
  -- This indicates if C is the reflection point of A on line l and passes through B.
  ∃ (eq1 eq2 : ℝ), l = (eq1, eq2 , 1) ∧
  A = (-3, -1) ∧
  B = (-4, 4) ∧
  C = (-1, -2)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs (((C.1 - A.1) * (B.2 - A.2)) - ((B.1 - A.1) * (C.2 - A.2))) / 2

theorem proof_equivalent_problem : 
  is_reflection_point A B C l → triangle_area A B C = 9 / 2 := 
by
  intros h
  sorry

end proof_equivalent_problem_l37_37937


namespace total_students_in_high_school_l37_37169

theorem total_students_in_high_school 
  (num_freshmen : ℕ)
  (num_sample : ℕ) 
  (num_sophomores : ℕ)
  (num_seniors : ℕ)
  (freshmen_drawn : ℕ)
  (sampling_ratio : ℕ)
  (total_students : ℕ)
  (h1 : num_freshmen = 600)
  (h2 : num_sample = 45)
  (h3 : num_sophomores = 20)
  (h4 : num_seniors = 10)
  (h5 : freshmen_drawn = 15)
  (h6 : sampling_ratio = 40)
  (h7 : freshmen_drawn * sampling_ratio = num_freshmen)
  : total_students = 1800 :=
sorry

end total_students_in_high_school_l37_37169


namespace value_of_x1_plus_x2_l37_37967

variable {X : Type}
variable [Discrete X]
variable {x1 x2 : ℝ}

def P (X : ℝ) : ℝ := sorry
def E (X : ℝ) : ℝ := x1 + x2
def D (X : ℝ) : ℝ := 2 * x1^2 + 2 * x2^2

theorem value_of_x1_plus_x2 
  (hP : P(X=x1) = P(X=x2)) 
  (hx1_lt_x2 : x1 < x2)
  (hE : E(X) = 4)
  (hD : D(X) = sorry): 
  x1 + x2 = 3 :=
sorry

end value_of_x1_plus_x2_l37_37967


namespace product_is_even_l37_37238

theorem product_is_even (a : Fin 7 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j) 
                        (perm : Multiset.card (Multiset.map a (Multiset.univ 7)) = 7) :
  Even ((a 0 - 1) * (a 1 - 2) * (a 2 - 3) * (a 3 - 4) * (a 4 - 5) * (a 5 - 6) * (a 6 - 7)) :=
sorry

end product_is_even_l37_37238


namespace sequence_an_bn_sum_Tn_l37_37282

open Nat

theorem sequence_an_bn {n : ℕ} (hn : n ≠ 0) (S : ℕ → ℕ) (a b : ℕ → ℕ):
  (∀ n, S n = 2 * n^2 + n) →
  (∀ n, a n = 4 * Int.log2 (b n) + 3) →
  (∀ n, n ≠ 0 → a n = S n - S (n - 1)) →
  ∀ n, n ≠ 0 → a n = 4 * n - 1 ∧ b n = 2^(n - 1) := 
sorry

theorem sum_Tn {n : ℕ} (hn : n ≠ 0) (a b : ℕ → ℕ):
  (∀ n, n ≠ 0 → a n = 4 * n - 1) →
  (∀ n, n ≠ 0 → b n = 2^(n - 1)) →
  let t (m : ℕ) := a m * b m in 
  (T : ℕ → ℕ) →
  (∀ n, T n = (Σ m in range n, t (m + 1))) →
  T n = (4 * n - 5) * 2^n + 5 :=
sorry

end sequence_an_bn_sum_Tn_l37_37282


namespace dates_relation_l37_37631

def melanie_data_set : set ℕ :=
  { x | (x >= 1 ∧ x <= 28) ∨ (x = 29 ∧ x <= 29) ∨ (x = 30 ∧ x <= 30) ∨ (x = 31 ∧ x <= 31)}

noncomputable def median (s : set ℕ) : ℝ := sorry -- Median calculation
noncomputable def mean (s : set ℕ) : ℝ := sorry -- Mean calculation
noncomputable def modes_median (s : set ℕ) : ℝ := sorry -- Median of modes calculation

theorem dates_relation : 
  let s := melanie_data_set in
  d < mean s < median s :=
sorry

end dates_relation_l37_37631


namespace solve_arctan_eq_pi_over_3_l37_37251

open Real

theorem solve_arctan_eq_pi_over_3 (x : ℝ) :
  arctan (1 / x) + arctan (1 / x^2) = π / 3 ↔ 
  x = (1 + sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) ∨
  x = (1 - sqrt (13 + 4 * sqrt 3)) / (2 * sqrt 3) :=
by
  sorry

end solve_arctan_eq_pi_over_3_l37_37251


namespace find_a2_l37_37609

def arithmetic_sequence (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n + d 

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a2 (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a a1 d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : a1 = -2010)
  (h4 : (S 2010) / 2010 - (S 2008) / 2008 = 2) :
  a 2 = -2008 :=
sorry

end find_a2_l37_37609


namespace john_work_more_days_l37_37200

theorem john_work_more_days (days_worked : ℕ) (amount_made : ℕ) (daily_earnings : ℕ) (h1 : days_worked = 10) (h2 : amount_made = 250) (h3 : daily_earnings = amount_made / days_worked) : 
  ∃ more_days : ℕ, more_days = (2 * amount_made / daily_earnings) - days_worked := 
by
  have h4 : daily_earnings = 25 := by {
    rw [h1, h2],
    norm_num,
  }
  have h5 : 2 * amount_made / daily_earnings = 20 := by {
    rw [h2, h4],
    norm_num,
  }
  use 10
  rw [h1, h5]
  norm_num

end john_work_more_days_l37_37200


namespace last_digit_to_appear_mod7_l37_37421

theorem last_digit_to_appear_mod7 (n : ℕ) : ∃ N, ∀ d ∈ {0, 1, 2, 3, 4, 5, 6}, 
  ∃ m ≤ N, nat.fib m % 7 = d ∧ ∀ k < m, nat.fib k % 7 ≠ d :=
sorry

end last_digit_to_appear_mod7_l37_37421


namespace find_m_l37_37360

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (p q : V)
variables (m : ℝ)

-- Define the condition that the vector m p + (5/8) q lies on the line passing through p and q
def lies_on_line (m : ℝ) : Prop := ∃ s : ℝ, m * p + (5 / 8) * q = p + s * (q - p)

-- The main statement to be proved
theorem find_m :
  lies_on_line p q (3 / 8) :=
sorry

end find_m_l37_37360


namespace problem1_proof_l37_37335

noncomputable def amoeba_bacteria_ratio (a₁ b₁ : ℕ) : Prop :=
2^99 * (b₁ - a₁) = 0 → a₁ = b₁

theorem problem1_proof (a₁ b₁ : ℕ) : amoeba_bacteria_ratio a₁ b₁ :=
begin
  intros h,
  have h1 : 2^99 ≠ 0 := by norm_num,
  have h2 : b₁ - a₁ = 0 := by sorry,
  show a₁ = b₁,
  exact eq_of_sub_eq_zero h2,
end

end problem1_proof_l37_37335


namespace directrix_of_parabola_l37_37896

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37896


namespace perpendicular_CK_A_l37_37337

open EuclideanGeometry

variables {A B C A' B' I K : Point}

theorem perpendicular_CK_A'B' 
  (h_triangle : Triangle A B C)
  (h_angle_C : ∠ A B C = 60)
  (h_AA'_bisector : AngleBisector A A' B)
  (h_BB'_bisector : AngleBisector B B' A)
  (h_I_incenter : InCenter I A B C)
  (h_K_symmetry : Symmetric K I A B) :
  Perpendicular (Line_through C K) (Line_through A' B') := 
sorry

end perpendicular_CK_A_l37_37337


namespace number_of_zeros_f_in_interval_0_2012_l37_37671

def f : ℝ → ℝ := sorry

theorem number_of_zeros_f_in_interval_0_2012 :
  (∀ x, f (x + 2) = -f x) →
  (∀ x, x ∈ Ioc (-2:ℝ) 2 → f x = abs x - 1) →
  (finset.card (finset.filter (λ x, f x = 0) (finset.range 2013).map (λ n, 4 * n)) = 1006) :=
sorry

end number_of_zeros_f_in_interval_0_2012_l37_37671


namespace inequality_proof_l37_37107

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end inequality_proof_l37_37107


namespace minimum_value_2a7_a11_l37_37496

variable {a_n : ℕ → ℝ}
variable (h_arith : ∀ n, 0 < a_n)
variable (h_geom_mean : a_n 4 * a_n 14 = 8)

theorem minimum_value_2a7_a11 :
  ∃ C, (∀ n, (a_n n ∈ { a_7, a_11 })) ∧ 2 * a_n 7 + a_n 11 ≥ C := 
sorry

end minimum_value_2a7_a11_l37_37496


namespace smallest_n_for_mod_20_l37_37452

theorem smallest_n_for_mod_20 (n : ℕ) (hn : n ≥ 9)
  (h : ∀ (S : Finset ℤ), S.card = n → ∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b) % 20 = (c + d) % 20) : True :=
by
  sorry

end smallest_n_for_mod_20_l37_37452


namespace sum_third_three_l37_37576

variables {a : ℕ → ℤ}

-- Define the properties of the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

-- Given conditions
axiom sum_first_three : a 1 + a 2 + a 3 = 9
axiom sum_second_three : a 4 + a 5 + a 6 = 27
axiom arithmetic_seq : is_arithmetic_sequence a

-- The proof goal
theorem sum_third_three : a 7 + a 8 + a 9 = 45 :=
by
  sorry  -- Proof is omitted here

end sum_third_three_l37_37576


namespace hyperbola_asymptote_value_of_b_l37_37938

theorem hyperbola_asymptote_value_of_b (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 :=
begin
  intros h,
  -- Proof is not required
  sorry
end

end hyperbola_asymptote_value_of_b_l37_37938


namespace enclose_by_polygons_l37_37371

theorem enclose_by_polygons (n : ℕ) 
  (h1 : regular_polygon 15)
  (h2 : ∀ (k : ℕ), k < 15 → encloses_polygons n 15)
  : n = 15 :=
sorry

end enclose_by_polygons_l37_37371


namespace cylinder_curved_surface_area_l37_37374

theorem cylinder_curved_surface_area {r h : ℝ} (hr: r = 2) (hh: h = 5) :  2 * Real.pi * r * h = 20 * Real.pi :=
by
  rw [hr, hh]
  sorry

end cylinder_curved_surface_area_l37_37374


namespace solution_set_f_gt_zero_l37_37119

variable (f : ℝ → ℝ)
variable (h_decreasing : ∀ x y : ℝ, x < y → f x > f y)
variable (h_f0 : f 0 = 1)
variable (h_f1 : f 1 = 0)

theorem solution_set_f_gt_zero :
  { x : ℝ | f x > 0 } = Iio 1 := sorry

end solution_set_f_gt_zero_l37_37119


namespace parabola_directrix_l37_37877

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37877


namespace cats_weigh_more_than_puppies_l37_37540

noncomputable def weight_puppy_A : ℝ := 6.5
noncomputable def weight_puppy_B : ℝ := 7.2
noncomputable def weight_puppy_C : ℝ := 8
noncomputable def weight_puppy_D : ℝ := 9.5
noncomputable def weight_cat : ℝ := 2.8
noncomputable def num_cats : ℕ := 16

theorem cats_weigh_more_than_puppies :
  (num_cats * weight_cat) - (weight_puppy_A + weight_puppy_B + weight_puppy_C + weight_puppy_D) = 13.6 :=
by
  sorry

end cats_weigh_more_than_puppies_l37_37540


namespace bill_health_insurance_cost_l37_37411

noncomputable def calculate_health_insurance_cost : ℕ := 3000

theorem bill_health_insurance_cost
  (normal_monthly_price : ℕ := 500)
  (gov_pay_less_than_10000 : ℕ := 90) -- 90%
  (gov_pay_between_10001_and_40000 : ℕ := 50) -- 50%
  (gov_pay_more_than_50000 : ℕ := 20) -- 20%
  (hourly_wage : ℕ := 25)
  (weekly_hours : ℕ := 30)
  (weeks_per_month : ℕ := 4)
  (months_per_year : ℕ := 12)
  (income_between_10001_and_40000 : Prop := (hourly_wage * weekly_hours * weeks_per_month * months_per_year) >= 10001 ∧ (hourly_wage * weekly_hours * weeks_per_month * months_per_year) <= 40000):
  (calculate_health_insurance_cost = 3000) :=
by
sry


end bill_health_insurance_cost_l37_37411


namespace min_sum_of_areas_l37_37296

-- Defining the structures and necessary conditions

def is_trapezoid (A B C D : Point) : Prop :=
  Parallel AD BC ∧ ¬ Collinear A B C ∧ ¬ Collinear A C D

def divides_into_triangles (A B C D : Point) (AC : Line) : Prop :=
  ¬ Collinear A B C ∧ ¬ Collinear A C D

def line_passing_through_intersection_point (A B C D : Point) (P : Point) : Prop :=
  ∃ AC BD : Line, Intersects AC BD = P ∧ Intersection_Parallel_to_Base A D AD BC

-- The main statement to prove
theorem min_sum_of_areas (A B C D P : Point) 
  (h_trapezoid : is_trapezoid A B C D)
  (h_divides : divides_into_triangles A B C D (Line AC))
  (h_passes : line_passing_through_intersection_point A B C D P) :
  ∀ l: Line, Parallel l AD → (sum_area_of_resulting_triangles l) ≥ (sum_area_of_resulting_triangles (line_through P AD)) := 
sorry

end min_sum_of_areas_l37_37296


namespace cos_sum_formula_l37_37469

theorem cos_sum_formula (x m : ℝ) (h : cos (x - π / 6) = m) : 
  cos x + cos (x - π / 3) = √3 * m :=
by
  sorry

end cos_sum_formula_l37_37469


namespace trigonometric_inequality_l37_37624

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hyz : y < z) 
  (hz : z < (π / 2)) : 
  (π / 2) + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) := 
sorry

end trigonometric_inequality_l37_37624


namespace ratio_of_circular_section_to_sphere_l37_37368

theorem ratio_of_circular_section_to_sphere
  (R : ℝ)
  (hR : R > 0) :
  let r := sqrt(3/4) * R in
  (π * r^2) / (4 * π * R^2) = 3 / 16 := by
  sorry

end ratio_of_circular_section_to_sphere_l37_37368


namespace number_of_ways_to_draw_4_from_15_l37_37754

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37754


namespace max_diagonals_no_intersection_l37_37829

theorem max_diagonals_no_intersection (chessboard : Matrix (Fin 8) (Fin 8) ℕ) :
  (∀ i j, 1 ≤ chessboard i j ∧ chessboard i j ≤ 2) →
  (∃ (diagonals : Fin 64 → Fin 2), 
     (∀ (i j : Fin 64 × Fin 2), i ≠ j → diagonals i ≠ diagonals j) ∧
     (diagonals.card = 36)) :=
sorry

end max_diagonals_no_intersection_l37_37829


namespace difference_blue_yellow_l37_37245

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end difference_blue_yellow_l37_37245


namespace man_l37_37363

variable (v : ℝ)  -- Man's speed in still water
variable (c : ℝ)  -- Speed of current
variable (s : ℝ)  -- Man's speed against the current
variable (u : ℝ)  -- Man's speed with the current

-- Given Conditions
axiom current_speed : c = 2.5
axiom against_current_speed : s = 10

-- Question to be proved
theorem man's_speed_with_current : u = 15 :=
by
  -- Define the problem as per the given conditions
  have v := s + c
  have u := v + c
  rw [current_speed, against_current_speed] at u
  exact u

end man_l37_37363


namespace sum_of_solutions_l37_37065

def f (x : ℝ) : ℝ := 2^(|x|) + 4 * |x|

theorem sum_of_solutions : (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0 ∧ x₁ ≠ x₂) →
  (∑ x in {solution : ℝ | f solution = 20}.to_finset, x = 0) :=
by
  sorry

end sum_of_solutions_l37_37065


namespace minimize_sum_of_distances_l37_37448

-- Defining the vertex coordinates of triangle ABC
variables {A B C : Point}

-- A arbitrary point inside the triangle 
variable (O : Point)

-- Definition of the sum of distances from a point to the vertices of the triangle
def sumOfDistances (P : Point) : ℝ :=
  dist P A + dist P B + dist P C

-- The statement of the theorem
theorem minimize_sum_of_distances (H : is_triangle A B C) : 
  ∃ O : Point, O ∈ triangle A B C ∧ 
    (∀ P ∈ triangle A B C, sumOfDistances O ≤ sumOfDistances P) :=
sorry

end minimize_sum_of_distances_l37_37448


namespace symmetry_axis_of_g_l37_37136

open Real

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * sin (ω * x + φ)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 12) 2 (π / 3)

theorem symmetry_axis_of_g :
  ∃ (k : ℤ), (k = 3) → (∀ x, g x = g (2π - x)) :=
sorry

end symmetry_axis_of_g_l37_37136


namespace directrix_eqn_of_parabola_l37_37873

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37873


namespace solve_for_m_l37_37835

theorem solve_for_m :
  ∃ (m : ℝ), (m ∈ ℝ) ∧ (∀ (i : ℂ), i = complex.I → (2 / (1 + i)) = (1 + m * i)) ↔ m = -1 :=
by
  sorry

end solve_for_m_l37_37835


namespace frac_eq_three_l37_37539

theorem frac_eq_three (a b c : ℝ) 
  (h₁ : a / b = 4 / 3) (h₂ : (a + c) / (b - c) = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
  sorry

end frac_eq_three_l37_37539


namespace number_of_ways_to_draw_balls_l37_37785

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37785


namespace minimum_t_value_l37_37708

noncomputable def f (x : ℝ) : ℝ :=
| 1, sin (2 * x) |
| sqrt 3, cos (2 * x) |

theorem minimum_t_value
  (t_pos : t > 0)
  (h_f : ∀ x, f x = det (matrix ([[1, sin (2 * x)], [sqrt 3, cos (2 * x)]])) ) :
  ∃ t, t > 0 ∧ f (x - t) = -f (x) ∧ t = π / 12 :=
by
  repeat { sorry }

end minimum_t_value_l37_37708


namespace vector_c_identity_l37_37535

variables (a b c : Vector ℝ 2)

def vector_a := ![1, 1]
def vector_b := ![2, -1]
def vector_c := ![-1, 2]

theorem vector_c_identity :
  c = vector_a - vector_b :=
  sorry

end vector_c_identity_l37_37535


namespace person_speed_correct_l37_37367

-- Define the conditions as hypotheses
def distance_to_bus_stop := 5 -- in km

def time_missed_by := 10 / 60 -- in hours
def time_before_arrival := 5 / 60 -- in hours

def speed_slow := 4 -- in km/h

-- Define the goal to be proved
theorem person_speed_correct :
  (distance_to_bus_stop / speed_slow) - time_missed_by = (distance_to_bus_stop / (distance_to_bus_stop / (1 - time_before_arrival)))) :=
begin
  sorry
end

end person_speed_correct_l37_37367


namespace inverse_of_g_at_84_l37_37948

theorem inverse_of_g_at_84:
  let g (x : ℝ) := 3 * x ^ 3 + 3 in
  g 3 = 84 ↔ ∃ y : ℝ, g y = 84 ∧ y = 3 :=
by
  sorry

end inverse_of_g_at_84_l37_37948


namespace number_of_students_who_bought_2_pencils_l37_37170

variable (a b c : ℕ)     -- a is the number of students buying 1 pencil, b is the number of students buying 2 pencils, c is the number of students buying 3 pencils.
variable (total_students total_pencils : ℕ) -- total_students is 36, total_pencils is 50
variable (students_condition1 students_condition2 : ℕ) -- conditions: students_condition1 for the sum of the students, students_condition2 for the sum of the pencils

theorem number_of_students_who_bought_2_pencils :
  total_students = 36 ∧
  total_pencils = 50 ∧
  total_students = a + b + c ∧
  total_pencils = a * 1 + b * 2 + c * 3 ∧
  a = 2 * (b + c) → 
  b = 10 :=
by sorry

end number_of_students_who_bought_2_pencils_l37_37170


namespace solve_n_l37_37152

-- Define the value of n given the condition
def n : ℕ := 55

-- Provide the condition as an assumption
axiom condition : sqrt (9 + n) = 8

-- Write the theorem to prove n equals 55 given the condition
theorem solve_n : n = 55 :=
by
  -- proof can be provided here later
  sorry

end solve_n_l37_37152


namespace minimize_sum_distances_l37_37832

theorem minimize_sum_distances (A B C : Point) (hA : acute (< A B C)) :
  ∃ P : Point, (P ∈ line_segment A B ∨ P ∈ line_segment B C ∨ P ∈ line_segment C A) ∧ 
  (∀ Q : Point, (Q ∈ line_segment A B ∨ Q ∈ line_segment B C ∨ Q ∈ line_segment C A) →
               PA + PB + PC ≥ PA + PB + PC) :=
sorry

end minimize_sum_distances_l37_37832


namespace marble_probability_l37_37292

theorem marble_probability (W G R B : ℕ) (h_total : W + G + R + B = 84) 
  (h_white : W / 84 = 1 / 4) (h_green : G / 84 = 1 / 7) :
  (R + B) / 84 = 17 / 28 :=
by
  sorry

end marble_probability_l37_37292


namespace andrew_stamps_permits_l37_37402

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end andrew_stamps_permits_l37_37402


namespace andrew_stamps_permits_l37_37401

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end andrew_stamps_permits_l37_37401


namespace tangent_line_at_slope_two_l37_37689

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l37_37689


namespace constant_length_EF_l37_37298

theorem constant_length_EF
  (ω₁ ω₂ : Circle)
  (O₁ O₂ A B C D E F : Point)
  (h_intersect : ω₁ ∩ ω₂ = {A, B})
  (h_ell : Line → ∃ (C D : Point), (C, B, D) ∈ ω₁ ∧ Line B ∩ ω₁ = {C} ∧ Line B ∩ ω₂ = {D})
  (h_tangent_C : Tangent(ω₁, C))
  (h_tangent_D : Tangent(ω₂, D))
  (h_tangent_intersect : ∃ E : Point, Tangent(ω₁, C) ∩ Tangent(ω₂, D) = {E})
  (h_AE_F : ∃ F : Point, Line A E ∩ Circumcircle(Δ AO₁O₂) = {A, F})
  : ∀ 𝓁 : Line, Segment E F = ConstantLength :=
sorry

end constant_length_EF_l37_37298


namespace maximum_contribution_l37_37627

def sum_contributions (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range n, x i

noncomputable def k (n : ℕ) (T : ℝ) (x : ℕ → ℝ) : ℝ :=
  Finset.sup (Finset.range n) x

theorem maximum_contribution 
  (n : ℕ) (T : ℝ)
  (x : ℕ → ℝ)
  (hn : n = 30)
  (hT : T = 60)
  (h_sum : sum_contributions n x = T)
  (h_n_le_T : ↑n ≤ T)
  (h_min_contrib : ∀ i, i < n → x i ≥ 1)
  : k n T x = 31 := 
sorry

end maximum_contribution_l37_37627


namespace average_speed_last_segment_l37_37646

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time_minutes : ℕ)
  (avg_speed_first_segment : ℕ)
  (avg_speed_second_segment : ℕ)
  (expected_avg_speed_last_segment : ℕ) :
  total_distance = 96 →
  total_time_minutes = 90 →
  avg_speed_first_segment = 60 →
  avg_speed_second_segment = 65 →
  expected_avg_speed_last_segment = 67 →
  (3 * (avg_speed_first_segment + avg_speed_second_segment + expected_avg_speed_last_segment) = (total_distance * 2)) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  have total_time_hours := 1.5
  have overall_avg_speed := 96 / 1.5
  have overall_avg_speed_value : overall_avg_speed = 64 := by linarith [total_time_hours]
  have avg_calc : (60 + 65 + 67) / 3 = 64 := by linarith
  sorry

end average_speed_last_segment_l37_37646


namespace minimum_tangent_distance_correct_l37_37384

noncomputable def minimum_tangent_distance : ℝ :=
  let center_circle := (3 : ℝ, 0 : ℝ)
  let point_on_line (x : ℝ) := (x, x + 1)
  let distance_from_center_to_line := 2 * Real.sqrt 2
  let tangent_length := Real.sqrt 7
  tangent_length

theorem minimum_tangent_distance_correct : minimum_tangent_distance = Real.sqrt 7 :=
  sorry

end minimum_tangent_distance_correct_l37_37384


namespace my_proof_l37_37915

noncomputable def proof_problem (α : ℝ) : Prop :=
  α ∈ Ioo (π / 2) π ∧ 
  sin (α + π / 4) = sqrt 2 / 10 → 
  cos α = -3 / 5 ∧ 
  sin (2 * α - π / 4) = -17 * sqrt 2 / 50

-- not the proof, just the statement of the proof problem
theorem my_proof (α : ℝ) : proof_problem α :=
  by { 
    -- Assuming lemma proofs, which would be filled in where required
    intros h,
    split,
    sorry,
    sorry
  }

end my_proof_l37_37915


namespace area_of_lit_plot_l37_37726

noncomputable def litArea (r : ℝ) : ℝ := (π * r^2) / 4

theorem area_of_lit_plot :
  let radius := 21 in
  litArea 21 = 110.25 * π :=
by
  sorry

end area_of_lit_plot_l37_37726


namespace discount_price_equation_correct_l37_37346

def original_price := 200
def final_price := 148
variable (a : ℝ) -- assuming a is a real number representing the percentage discount

theorem discount_price_equation_correct :
  original_price * (1 - a / 100) ^ 2 = final_price :=
sorry

end discount_price_equation_correct_l37_37346


namespace num_ways_disperse_students_l37_37406

/-- 
Theorem: The number of ways to assign 4 students to 3 classes (A, B, C) such that each class contains at least one student is 36.
-/
theorem num_ways_disperse_students : 
  let num_students := 4
  let num_classes := 3
  let ways :=
    (nat.choose num_students 2) * 
    (nat.factorial num_classes)
  ways = 36 := 
begin
  sorry
end

end num_ways_disperse_students_l37_37406


namespace cos_sum_l37_37340

theorem cos_sum (cos_105 : Real := Real.cos (105 * Real.pi / 180))
                  (cos_45 : Real := Real.cos (45 * Real.pi / 180))
                  (sin_105 : Real := Real.sin (105 * Real.pi / 180))
                  (sin_45 : Real := Real.sin (45 * Real.pi / 180)) :
  cos_105 * cos_45 + sin_105 * sin_45 = 1 / 2 := 
by
  -- the proof logic goes here
  sorry

end cos_sum_l37_37340


namespace modulus_of_complex_number_l37_37857

theorem modulus_of_complex_number : abs (3 - 4 * complex.i) = 5 := 
by
  sorry

end modulus_of_complex_number_l37_37857


namespace sum_of_solutions_eq_zero_l37_37064

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l37_37064


namespace centroid_of_triangle_l37_37868

theorem centroid_of_triangle :
  let A := (2, 8)
  let B := (6, 2)
  let C := (0, 4)
  let centroid := ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )
  centroid = (8 / 3, 14 / 3) := 
by
  sorry

end centroid_of_triangle_l37_37868


namespace arithmetic_sequence_property_l37_37089

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : (∑ i in (Finset.range 101).map (Function.Embedding.coe_ofStrictMono Nat.lt_succ_self), a (i + 1)) = 0) : a 3 + a 99 = 0 :=
sorry

end arithmetic_sequence_property_l37_37089


namespace draw_4_balls_in_order_l37_37777

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37777


namespace minimum_period_of_sin_squared_l37_37983

theorem minimum_period_of_sin_squared :
  ∃ C > 0, (∀ x : ℝ, sin x ^ 2 = sin (x + C) ^ 2) ∧ (∀ C' > 0, (∀ x : ℝ, sin x ^ 2 = sin (x + C') ^ 2) → C' ≥ C) ∧ C = π :=
sorry

end minimum_period_of_sin_squared_l37_37983


namespace tangent_line_eqn_l37_37692

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derive (x : ℝ) : ℝ := (1 / x) + 1

theorem tangent_line_eqn :
  (∀ m x₀ y₀ : ℝ, (derive x₀ = m) → (y₀ = curve x₀) → (∀ x : ℝ, y₀ + m * (x - x₀) = 2 * x)) :=
by
  sorry

end tangent_line_eqn_l37_37692


namespace cells_remain_unchanged_l37_37723

-- We have a square S divided into 4 parts p1, p2, p3, and p4
constant S : Type
constant p1 p2 p3 p4 : S

-- Cells of S are 64
def cells (s : S) : ℕ := 64

-- Condition: initially, the square has 64 cells.
axiom h0 : cells S = 64

-- Rearranging is defined as a function that reconfigures the parts
def rearrange (s : S) (p1 p2 p3 p4 : S) : S := sorry

-- We need to prove that the number of cells remains unchanged after rearrangement
theorem cells_remain_unchanged (s : S) (p1 p2 p3 p4 : S) :
  cells (rearrange s p1 p2 p3 p4) = cells s := by
  sorry

end cells_remain_unchanged_l37_37723


namespace odd_population_days_l37_37342

theorem odd_population_days 
  (born : ℕ) 
  (born_odd : odd born) 
  (lifespan : ∀ m, m ∈ born → m + 100 ∈ born → m ∉ born) : 
  ∃ (days_when_odd : set ℕ), days_when_odd.card ≥ 100 :=
by sorry

end odd_population_days_l37_37342


namespace draw_4_balls_ordered_l37_37761

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37761


namespace directrix_eqn_of_parabola_l37_37872

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37872


namespace intersection_A_B_l37_37101

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_A_B : A ∩ B = {-1, 2} :=
by
  sorry

end intersection_A_B_l37_37101


namespace problem_solution_l37_37550

theorem problem_solution (x : ℝ) (N : ℝ) (h1 : 625 ^ (-x) + N ^ (-2 * x) + 5 ^ (-4 * x) = 11) (h2 : x = 0.25) :
  N = 25 / 2809 :=
by
  sorry

end problem_solution_l37_37550


namespace tangent_line_eqn_l37_37693

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derive (x : ℝ) : ℝ := (1 / x) + 1

theorem tangent_line_eqn :
  (∀ m x₀ y₀ : ℝ, (derive x₀ = m) → (y₀ = curve x₀) → (∀ x : ℝ, y₀ + m * (x - x₀) = 2 * x)) :=
by
  sorry

end tangent_line_eqn_l37_37693


namespace directrix_of_parabola_l37_37898

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37898


namespace pencils_total_l37_37191

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l37_37191


namespace find_transform_matrix_l37_37445

variable {R : Type*} [CommRing R]

def transform_matrix (N M : Matrix (Fin 2) (Fin 2) R) : Matrix (Fin 2) (Fin 2) R :=
  N * M

theorem find_transform_matrix (a b c d : R) :
  ∃ (N : Matrix (Fin 2) (Fin 2) R),
    transform_matrix N (Matrix.vecCons (Matrix.vecCons a b) (Matrix.vecCons c d)) =
    Matrix.vecCons (Matrix.vecCons (4 * a) (4 * b)) (Matrix.vecCons (2 * c) (2 * d)) :=
  by
    let N := ![![4, 0], ![0, 2]]
    use N
    sorry

end find_transform_matrix_l37_37445


namespace regular_polygon_sides_l37_37814

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (i : ℕ), i < n → exterior_angle (polygon n) i = 10) :
  n = 36 :=
by
  -- The sum of all exterior angles of any polygon is 360 degrees.
  let sum_exterior_angles : ℕ := n * 10
  have key : sum_exterior_angles = 360,
  sorry   -- continue proof

end regular_polygon_sides_l37_37814


namespace john_weekly_loss_is_correct_l37_37594

-- Definitions based on the conditions
def daily_production_capacity := 1000
def production_cost_per_tire := 250
def selling_price_multiplier := 1.5
def potential_daily_sales := 1200

-- Function to calculate the additional revenue lost per week
def weekly_revenue_lost : ℕ :=
  let additional_tires := potential_daily_sales - daily_production_capacity
  let daily_revenue_lost := additional_tires * (production_cost_per_tire * selling_price_multiplier)
  daily_revenue_lost * 7

-- The theorem we need to prove
theorem john_weekly_loss_is_correct : weekly_revenue_lost = 525000 := by
  sorry

end john_weekly_loss_is_correct_l37_37594


namespace problem1_problem2_l37_37526

-- Define the functions f(x) and g(x)
def f (x : ℝ) : ℝ := (Real.exp x) / x
def g (x : ℝ) : ℝ := 2 * (x - Real.log x)

-- Problem (I): Prove that f(x) > g(x) when x > 0
theorem problem1 (x : ℝ) (hx : 0 < x) : f x > g x := sorry

-- Define points P and Q and function h(x)
def P (x : ℝ) : ℝ × ℝ := (x, x * f x)
def Q (x : ℝ) : ℝ × ℝ := (-Real.sin x, Real.cos x)
def h (x : ℝ) : ℝ := (P x).1 * (Q x).1 + (P x).2 * (Q x).2

-- Problem (II): Determine the number of zeros of h(x) in the specified range
theorem problem2 : ∃! x : ℝ, x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ h x = 0 := sorry

end problem1_problem2_l37_37526


namespace range_of_b_l37_37478

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ) (hodd : ∀ x, f (-x) = -f x)
  (hf_def : ∀ x, f x = (2^x - a) / (2^x + 1))
  (hg_def : ∀ x, g x = log (x^2 - b))
  (hineq : ∀ x1 x2 : ℝ, f x1 ≤ g x2) : b ≤ -real.exp 1 := 
sorry

end range_of_b_l37_37478


namespace front_view_correct_l37_37086

section stack_problem

def column1 : List ℕ := [3, 2]
def column2 : List ℕ := [1, 4, 2]
def column3 : List ℕ := [5]
def column4 : List ℕ := [2, 1]

def tallest (l : List ℕ) : ℕ := l.foldr max 0

theorem front_view_correct :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 4, 5, 2] :=
sorry

end stack_problem

end front_view_correct_l37_37086


namespace average_score_girls_proof_l37_37392

noncomputable def average_score_girls_all_schools (A a B b C c : ℕ)
  (adams_boys : ℕ) (adams_girls : ℕ) (adams_comb : ℕ)
  (baker_boys : ℕ) (baker_girls : ℕ) (baker_comb : ℕ)
  (carter_boys : ℕ) (carter_girls : ℕ) (carter_comb : ℕ)
  (all_boys_comb : ℕ) : ℕ :=
  -- Assume number of boys and girls per school A, B, C (boys) and a, b, c (girls)
  if (adams_boys * A + adams_girls * a) / (A + a) = adams_comb ∧
     (baker_boys * B + baker_girls * b) / (B + b) = baker_comb ∧
     (carter_boys * C + carter_girls * c) / (C + c) = carter_comb ∧
     (adams_boys * A + baker_boys * B + carter_boys * C) / (A + B + C) = all_boys_comb
  then (85 * a + 92 * b + 80 * c) / (a + b + c) else 0

theorem average_score_girls_proof (A a B b C c : ℕ)
  (adams_boys : ℕ := 82) (adams_girls : ℕ := 85) (adams_comb : ℕ := 83)
  (baker_boys : ℕ := 87) (baker_girls : ℕ := 92) (baker_comb : ℕ := 91)
  (carter_boys : ℕ := 78) (carter_girls : ℕ := 80) (carter_comb : ℕ := 80)
  (all_boys_comb : ℕ := 84) :
  average_score_girls_all_schools A a B b C c adams_boys adams_girls adams_comb baker_boys baker_girls baker_comb carter_boys carter_girls carter_comb all_boys_comb = 85 :=
by
  sorry

end average_score_girls_proof_l37_37392


namespace number_of_integer_solutions_for_f_geq_0_when_x_leq_0_l37_37940

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else -(x^2 - 2*(-x)- 3)

theorem number_of_integer_solutions_for_f_geq_0_when_x_leq_0 : ∃! (n : ℕ), n = 4 ∧ (∀ x : ℤ, x ≤ 0 → f x ≥ 0 → x ∈ { -3, -2, -1, 0 }) :=
by
  sorry

end number_of_integer_solutions_for_f_geq_0_when_x_leq_0_l37_37940


namespace linear_inequality_condition_l37_37966

theorem linear_inequality_condition (m : ℤ) (x : ℝ) : 
  (3m - 5 * x ^ (3 + m) > 4) → (3 + m = 1) → (m = -2) :=
by
  sorry

end linear_inequality_condition_l37_37966


namespace white_pairs_coincide_l37_37045

theorem white_pairs_coincide :
  ∀ (red_triangles blue_triangles white_triangles : ℕ)
    (red_pairs blue_pairs red_blue_pairs : ℕ),
  red_triangles = 4 →
  blue_triangles = 4 →
  white_triangles = 6 →
  red_pairs = 3 →
  blue_pairs = 2 →
  red_blue_pairs = 1 →
  (2 * white_triangles - red_triangles - blue_triangles - red_blue_pairs) = white_triangles →
  6 = white_triangles :=
by
  intros red_triangles blue_triangles white_triangles
         red_pairs blue_pairs red_blue_pairs
         H_red H_blue H_white
         H_red_pairs H_blue_pairs H_red_blue_pairs
         H_pairs
  sorry

end white_pairs_coincide_l37_37045


namespace frost_time_with_sprained_wrist_l37_37228

-- Definitions
def normal_time_per_cake : ℕ := 5
def additional_time_for_10_cakes : ℕ := 30
def normal_time_for_10_cakes : ℕ := 10 * normal_time_per_cake
def sprained_time_for_10_cakes : ℕ := normal_time_for_10_cakes + additional_time_for_10_cakes

-- Theorems
theorem frost_time_with_sprained_wrist : ∀ x : ℕ, 
  (10 * x = sprained_time_for_10_cakes) ↔ (x = 8) := 
sorry

end frost_time_with_sprained_wrist_l37_37228


namespace graduation_photo_arrangement_l37_37408

theorem graduation_photo_arrangement (teachers middle_positions other_students : Finset ℕ) (A B : ℕ) :
  teachers.card = 2 ∧ middle_positions.card = 2 ∧ 
  (other_students ∪ {A, B}).card = 4 ∧ ∀ t ∈ teachers, t ∈ middle_positions →
  ∃ arrangements : ℕ, arrangements = 8 :=
by
  sorry

end graduation_photo_arrangement_l37_37408


namespace directrix_eqn_of_parabola_l37_37870

theorem directrix_eqn_of_parabola : 
  ∀ y : ℝ, x = - (1 / 4 : ℝ) * y ^ 2 → x = 1 :=
by
  sorry

end directrix_eqn_of_parabola_l37_37870


namespace quadratic_has_negative_root_iff_l37_37276

theorem quadratic_has_negative_root_iff (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by
  sorry

end quadratic_has_negative_root_iff_l37_37276


namespace number_of_ways_to_draw_balls_l37_37788

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37788


namespace not_like_terms_l37_37720

theorem not_like_terms (x y a b c m : ℝ) :
  ¬ (2 * x^5 * y = -1 / 2 * x^5 * y) ∧
  ¬ (-2.5 = abs (-2)) ∧
  ¬ (abc / 3 = abc) ∧
  (m^2 ≠ 2 * m) := by 
  sorry

end not_like_terms_l37_37720


namespace perpendicular_vectors_l37_37963

/-- If vectors a = (1, 2) and b = (x, 4) are perpendicular, then x = -8. -/
theorem perpendicular_vectors (x : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (x, 4)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : x = -8 :=
by {
  sorry
}

end perpendicular_vectors_l37_37963


namespace g_675_eq_42_l37_37620

theorem g_675_eq_42 
  (g : ℕ → ℕ) 
  (h_mul : ∀ x y : ℕ, x > 0 → y > 0 → g (x * y) = g x + g y) 
  (h_g15 : g 15 = 18) 
  (h_g45 : g 45 = 24) : g 675 = 42 :=
by
  sorry

end g_675_eq_42_l37_37620


namespace petya_can_win_l37_37715

theorem petya_can_win :
  ∃ (turns : list ℕ), 
  (length turns = 55 ∧
   ∀ t ∈ turns, 1 ≤ t ∧ t ≤ 55 ∧ 
   (∃ (i : ℕ), i ≤ 55 ∧ ∑ j in (finset.range i), (turns.nth j).get_or_else 0 = 50)) :=
sorry

end petya_can_win_l37_37715


namespace garden_problem_l37_37812

theorem garden_problem (c d : ℕ) (hc : c > 0) (hd : d > 0) (hcd : d > c) 
  (hb : (c - 4) * (d - 4) = 2 / 3 * c * d) : 
  ({(c, d) : ℕ × ℕ | c > 0 ∧ d > 0 ∧ d > c ∧ (c - 4) * (d - 4) = 2 / 3 * c * d}).card = 4 := 
sorry

end garden_problem_l37_37812


namespace fraction_identity_l37_37049

variables {a b : ℝ}

theorem fraction_identity (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) :=
by sorry

end fraction_identity_l37_37049


namespace range_of_a_for_increasing_function_l37_37670

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a ^ x

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3/2 ≤ a ∧ a < 6) := sorry

end range_of_a_for_increasing_function_l37_37670


namespace directrix_of_given_parabola_l37_37884

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37884


namespace geometric_sequence_q_value_l37_37610

theorem geometric_sequence_q_value:
  ∃ (a_n : ℕ → ℝ) (q : ℝ),
  (∀ n : ℕ, a_n (n + 1) = q * a_n n) ∧
  |q| > 1 ∧
  ({a_n 0, a_n 1, a_n 2, a_n 3} = { -72, -32, 48, 108 }) ∧
  2 * q = -3 :=
begin
  sorry
end

end geometric_sequence_q_value_l37_37610


namespace sum_opposite_sign_zero_l37_37325

def opposite_sign (a b : ℝ) : Prop :=
(a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem sum_opposite_sign_zero {a b : ℝ} (h : opposite_sign a b) : a + b = 0 :=
sorry

end sum_opposite_sign_zero_l37_37325


namespace integral_equality_l37_37415

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..1, (real.exp x + 2 * x)

theorem integral_equality : definite_integral = real.exp 1 :=
by
  sorry

end integral_equality_l37_37415


namespace stickers_distribution_l37_37541

theorem stickers_distribution : 
  ∃ (ways : ℕ), ways = 7 ∧ 
  (∀ (sticker sheet : ℕ), sticker = 10 ∧ sheet = 5 → 
  (∃! (f : fin sheet → ℕ), (∀ j, 1 ≤ f(j)) ∧ (∑ j, f(j) = sticker)) → ways = 7) := 
sorry

end stickers_distribution_l37_37541


namespace find_x_y_l37_37218

noncomputable def x_y_sum (x y : ℝ) : Prop :=
  (4^x = 16^(y+2)) ∧ (25^y = 5^(x-15)) ∧ (x + y = -12.5)

theorem find_x_y (x y : ℝ) : x_y_sum x y :=
by {
  sorry
}

end find_x_y_l37_37218


namespace total_cost_verification_l37_37562

-- Conditions given in the problem
def holstein_cost : ℕ := 260
def jersey_cost : ℕ := 170
def num_hearts_on_card : ℕ := 4
def num_cards_in_deck : ℕ := 52
def cow_ratio_holstein : ℕ := 3
def cow_ratio_jersey : ℕ := 2
def sales_tax : ℝ := 0.05
def transport_cost_per_cow : ℕ := 20

def num_hearts_in_deck := num_cards_in_deck
def total_num_cows := 2 * num_hearts_in_deck
def total_parts_ratio := cow_ratio_holstein + cow_ratio_jersey

-- Total number of cows calculated 
def num_holstein_cows : ℕ := (cow_ratio_holstein * total_num_cows) / total_parts_ratio
def num_jersey_cows : ℕ := (cow_ratio_jersey * total_num_cows) / total_parts_ratio

-- Cost calculations
def holstein_total_cost := num_holstein_cows * holstein_cost
def jersey_total_cost := num_jersey_cows * jersey_cost
def total_cost_before_tax_and_transport := holstein_total_cost + jersey_total_cost
def total_sales_tax := total_cost_before_tax_and_transport * sales_tax
def total_transport_cost := total_num_cows * transport_cost_per_cow
def final_total_cost := total_cost_before_tax_and_transport + total_sales_tax + total_transport_cost

-- Lean statement to prove the result
theorem total_cost_verification : final_total_cost = 26324.50 := by sorry

end total_cost_verification_l37_37562


namespace overall_net_gain_percent_approx_l37_37792

noncomputable theory
open Real

def cost_price_A : ℝ := 100
def selling_price_A : ℝ := 120

def cost_price_B : ℝ := 150
def discount_B : ℝ := 0.1
def actual_cost_price_B : ℝ := cost_price_B * (1 - discount_B)
def selling_price_B : ℝ := 180

def cost_price_C : ℝ := 200
def tax_C : ℝ := 0.05
def actual_cost_price_C : ℝ := cost_price_C * (1 + tax_C)
def selling_price_C : ℝ := 210

def total_cost_price : ℝ := cost_price_A + actual_cost_price_B + actual_cost_price_C
def total_selling_price : ℝ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℝ := total_selling_price - total_cost_price

def net_gain_percent : ℝ := (total_profit / total_cost_price) * 100

theorem overall_net_gain_percent_approx : abs (net_gain_percent - 14.61) < 0.01 :=
by
  -- Proof is omitted
  sorry

end overall_net_gain_percent_approx_l37_37792


namespace maximum_height_l37_37810

def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem maximum_height : ∃ t₀ : ℝ, ∀ t : ℝ, height t ≤ height t₀ ∧ height t₀ = 161 :=
by
  let t₀ : ℝ := 5 / 2
  existsi t₀
  sorry

end maximum_height_l37_37810


namespace new_set_mean_variance_l37_37494

variables (x : ℕ → ℝ) (n : ℕ)

noncomputable def mean (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, x i) / n

noncomputable def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, (x i - mean x n) ^ 2) / n

theorem new_set_mean_variance (mean_old variance_old : ℝ)
  (hmean_old : mean_old = 8) (hvariance_old : variance_old = 2) :
  let y i := 4 * x i + 1 in
  mean y n = 33 ∧ variance y n = 32 :=
by
  -- Conditions given
  simp [hmean_old, hvariance_old],
  sorry

end new_set_mean_variance_l37_37494


namespace incorrect_intervals_l37_37324

theorem incorrect_intervals :
  ¬ (3.4 ∈ set.Ico 2 3.4) ∧ ¬ (3.4 ∈ set.Icc 2 3) :=
by
  sorry

end incorrect_intervals_l37_37324


namespace find_tangent_line_l37_37115

theorem find_tangent_line 
  (a b : ℝ)
  (L1 : ∀ x y : ℝ, a * x + b * y - 3 = 0)
  (C1 : ∀ x y : ℝ, x^2 + y^2 + 4 * x - 1 = 0)
  (P_on_line : L1 (-1) 2)
  (L1_tangent_C1_at_P : ∀ x y : ℝ, L1 x y = 0 ∧ C1 x y = 0 → (x, y) = (-1, 2)) :
  a = 1 ∧ b = 2 :=
by
  sorry

end find_tangent_line_l37_37115


namespace solve_m_l37_37237

noncomputable def value_of_m (x y m : ℤ) : Prop :=
  (3 * x + 7 * y = 5 * m - 3) ∧ (2 * x + 3 * y = 8) ∧ (x + 2 * y = 5)

theorem solve_m : ∃ m : ℤ, ∀ x y : ℤ, value_of_m x y m → m = 4 :=
by
  intros
  existsi 4
  intros x y h
  rw [value_of_m] at h
  sorry

end solve_m_l37_37237


namespace sqrt_pattern_l37_37633

theorem sqrt_pattern (n : ℕ) : 
  √(n - n / (n^2 + 1)) = n * √(n / (n^2 + 1)) :=
  sorry

end sqrt_pattern_l37_37633


namespace solve_for_k_l37_37555

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end solve_for_k_l37_37555


namespace compare_abc_l37_37511

theorem compare_abc (m : ℝ) (h : 0 < m ∧ m < 1) : 
  let a := 3 ^ m
  let b := Real.logBase 3 m
  let c := m ^ 3
  a > c ∧ c > b :=
by
  sorry

end compare_abc_l37_37511


namespace intersection_M_N_eq_segment_l37_37102

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq_segment : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_segment_l37_37102


namespace friend_who_earned_35_gives_15_71_l37_37649

noncomputable def totalEarnings : ℝ := 10 + 15 + 20 + 25 + 30 + 35 + 0
noncomputable def eachFriendShare : ℝ := totalEarnings / 7
noncomputable def amountToRedistribute : ℝ := 35 - eachFriendShare 

theorem friend_who_earned_35_gives_15_71 :
  eachFriendShare = 135 / 7 →
  amountToRedistribute = 15.71 :=
by
  -- Define the total earnings
  have h1 : totalEarnings = 135 := by sorry
  -- Define each friend's share
  have h2 : eachFriendShare = 135 / 7 := by sorry
  -- Therefore,
  show amountToRedistribute = 15.71, from sorry

end friend_who_earned_35_gives_15_71_l37_37649


namespace simplest_square_root_l37_37319

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_simplest_root (s : ℝ) : Prop :=
  ∃ n : ℕ, s = real.sqrt n ∧ is_prime n

theorem simplest_square_root :
  is_simplest_root (real.sqrt 11) ∧
  ¬ is_simplest_root (real.sqrt 8) ∧
  ¬ is_simplest_root (real.sqrt 12) ∧
  ¬ is_simplest_root (real.sqrt 36) :=
sorry

end simplest_square_root_l37_37319


namespace largest_set_S_size_l37_37379

open Set

-- Define the conditions for the triangles in set S
def is_valid_triangle (a b c : ℕ) : Prop := a > b ∧ b ≥ c ∧ b + c > a ∧ a < 6 ∧ b < 6 ∧ c < 6

-- Define the set S consisting of valid triangles
def S : Set (ℕ × ℕ × ℕ) := { t | ∃ a b c, t = (a, b, c) ∧ is_valid_triangle a b c }

-- Define the property of not being congruent or similar
def not_similar (t1 t2 : ℕ × ℕ × ℕ) : Prop :=
  let (a1, b1, c1) := t1
  let (a2, b2, c2) := t2
  ¬ (a1 / b1 = a2 / b2 ∧ b1 / c1 = b2 / c2 ∧ a1 / c1 = a2 / c2)

-- Define the function to check uniqueness based on similarity
def is_unique_set (S : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ t1 t2 ∈ S, t1 ≠ t2 → not_similar t1 t2

-- The main theorem
theorem largest_set_S_size : ∃ (n : ℕ), is_unique_set S ∧ n = 8 := by
  sorry

end largest_set_S_size_l37_37379


namespace tangent_line_equation_l37_37695

noncomputable def curve (x : ℝ) : ℝ :=
  Real.log x + x + 1

theorem tangent_line_equation : ∃ x y : ℝ, derivative curve x = 2 ∧ curve x = y ∧ y = 2 * x := 
begin
  sorry
end

end tangent_line_equation_l37_37695


namespace draw_4_balls_ordered_l37_37757

theorem draw_4_balls_ordered : 
  ∀ (n m : ℕ), (n = 15) ∧ (m = 4) → (∏ i in finset.range(m), n - i) = 32760 :=
by
  intros n m h
  rcases h with ⟨hn, hm⟩
  rw [hn, hm]
  norm_num
  sorry

end draw_4_balls_ordered_l37_37757


namespace roots_in_ap_difference_one_l37_37684

theorem roots_in_ap_difference_one :
  ∀ (r1 r2 r3 : ℝ), 
    64 * r1^3 - 144 * r1^2 + 92 * r1 - 15 = 0 ∧
    64 * r2^3 - 144 * r2^2 + 92 * r2 - 15 = 0 ∧
    64 * r3^3 - 144 * r3^2 + 92 * r3 - 15 = 0 ∧
    (r2 - r1 = r3 - r2) →
    max (max r1 r2) r3 - min (min r1 r2) r3 = 1 := 
by
  intros r1 r2 r3 h
  sorry

end roots_in_ap_difference_one_l37_37684


namespace max_product_of_removed_numbers_l37_37911

theorem max_product_of_removed_numbers :
  ∃ (S : Finset ℕ), 
  S.card = 10 ∧
  S.sum = 55 ∧ 
  (∃ (S₃ S₇ : Finset ℕ),
    S₃.card = 3 ∧ 
    S₇.card = 7 ∧
    S₃ ∪ S₇ = S ∧ 
    S₃ ∩ S₇ = ∅ ∧ 
    S₇.sum = (55 * 7) / 11 ∧ 
    (∃ a b c : ℕ, a ∈ S₃ ∧ b ∈ S₃ ∧ c ∈ S₃ ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      a + b + c = 20 ∧
      a * b * c = 280)) :=
sorry

end max_product_of_removed_numbers_l37_37911


namespace ball_drawing_ways_l37_37766

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37766


namespace remaining_pencils_l37_37196

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l37_37196


namespace find_a_if_parallel_l37_37502

-- Given conditions
variables {A B : Type} [normed_add_torsor ℝ B] (a : ℝ)

def point_A : B := ⟨1, 2⟩
def point_B : B := ⟨a, 4⟩
def vector_m : B := ⟨2, 1⟩

-- Ensure parallelism between vectors AB and m
def is_parallel (v1 v2 : B) : Prop :=
∃ k : ℝ, v1 = k • v2

-- Calculate AB vector from point coordinates
def vector_AB : B := ⟨a - 1, 2⟩

-- Main theorem: Prove if AB is parallel to m then a = 5
theorem find_a_if_parallel (h : is_parallel vector_AB vector_m) : a = 5 := by
  sorry

end find_a_if_parallel_l37_37502


namespace difference_blue_yellow_l37_37244

def total_pebbles : ℕ := 40
def red_pebbles : ℕ := 9
def blue_pebbles : ℕ := 13
def remaining_pebbles : ℕ := total_pebbles - red_pebbles - blue_pebbles
def groups : ℕ := 3
def pebbles_per_group : ℕ := remaining_pebbles / groups
def yellow_pebbles : ℕ := pebbles_per_group

theorem difference_blue_yellow : blue_pebbles - yellow_pebbles = 7 :=
by
  unfold blue_pebbles yellow_pebbles pebbles_per_group remaining_pebbles total_pebbles red_pebbles
  sorry

end difference_blue_yellow_l37_37244


namespace find_a_l37_37559

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

theorem find_a : ∃ (a : ℝ), (a = 1) ∧
  (let f' := fun x => Real.log x + 1 in
  let tangent_line := fun x => x - 1 + a in
  tangent_line 2 = 2) :=
by
  use 1
  simp [f]
  sorry

end find_a_l37_37559


namespace largest_integer_satisfying_sin_cos_condition_proof_l37_37057

noncomputable def largest_integer_satisfying_sin_cos_condition :=
  ∀ (x : ℝ) (n : ℕ), (∀ (n' : ℕ), (∀ x : ℝ, (Real.sin x ^ n' + Real.cos x ^ n' ≥ 2 / n') → n ≤ n')) → n = 4

theorem largest_integer_satisfying_sin_cos_condition_proof :
  largest_integer_satisfying_sin_cos_condition :=
by
  sorry

end largest_integer_satisfying_sin_cos_condition_proof_l37_37057


namespace price_of_third_variety_l37_37732

theorem price_of_third_variety :
  ∃ P : ℝ,
    ((1 * 126 + 1 * 135 + 2 * P) / 4 = 154) →
    P = 177.5 :=
begin
  use 177.5,
  intro h,
  have : 1 * 126 + 1 * 135 + 2 * 177.5 = 4 * 154,
  { norm_num },
  exact (div_eq_iff (by norm_num : (4:ℝ) ≠ 0)).mpr this,
  sorry
end

end price_of_third_variety_l37_37732


namespace area_enclosed_by_curves_l37_37849

noncomputable def f1 := fun x : ℝ => Real.sin x
noncomputable def f2 := fun x : ℝ => (4 / Real.pi) ^ 2 * Real.sin ((Real.pi / 4) * x ^ 2)

theorem area_enclosed_by_curves :
  ∫ (x : ℝ) in 0..(Real.pi / 4), (f1 x - f2 x) = 1 - (Real.sqrt 2 / 2) * (1 + Real.pi / 12) :=
by
  sorry

end area_enclosed_by_curves_l37_37849


namespace find_length_AX_l37_37843

theorem find_length_AX 
    (O A B C D X : ℝ^2) 
    (hO : dist O A = 1)
    (hO2 : dist O B = 1)
    (hO3 : dist O C = 1)
    (hO4 : dist O D = 1)
    (hAOB : angle O A B = 120)
    (hBOC : angle O B C = 60)
    (hCOD : angle O C D = 180)
    (hAXC : X ∈ segment (arc_minor (circle O 1) A C))
    (hAXB : angle A X B = 90) : 
    dist A X = sqrt 3 :=
sorry

end find_length_AX_l37_37843


namespace find_height_of_trapezoid_trapezoid_has_one_pair_parallel_bases_l37_37376

variables {A B C D : Type}
variables (trapezoid : A)
variables (largerBase smallerBase : B)
variables (nonParallelSide1 nonParallelSide2 : C)
variables (height : D)

constants (AB CD : ℝ) (midSegmentLength : ℝ := 3)
constants (angleA angleB : ℝ := 30) (angleC angleD : ℝ := 60)

-- Assuming conditions
axiom (trapezoid_has_one_pair_parallel_bases : true)
axiom (angles_at_ends_of_larger_base : angleA = 30 ∧ angleB = 60)
axiom (segment_connecting_midpoints_is_3 : midSegmentLength = 3)

-- To prove
theorem find_height_of_trapezoid_trapezoid_has_one_pair_parallel_bases:
  height = (3 * real.sqrt 3) / 2 := 
sorry

end find_height_of_trapezoid_trapezoid_has_one_pair_parallel_bases_l37_37376


namespace max_intersections_l37_37863

-- Define the conditions
def num_points_x : ℕ := 15
def num_points_y : ℕ := 10

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the problem statement
theorem max_intersections (I : ℕ) :
  (15 : ℕ) == num_points_x →
  (10 : ℕ) == num_points_y →
  (I = binom 15 2 * binom 10 2) →
  I = 4725 := by
  -- We add sorry to skip the proof
  sorry

end max_intersections_l37_37863


namespace smallest_base_for_150_l37_37310

theorem smallest_base_for_150 : ∃ b : ℕ, (b^2 ≤ 150 ∧ 150 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 150 ∧ 150 < n^3) → n ≥ b :=
by
  let b := 6
  have h1 : b^2 ≤ 150 := by norm_num
  have h2 : 150 < b^3 := by norm_num
  have h3 : ∀ n : ℕ, n^2 ≤ 150 → 150 < n^3 → n ≥ b :=
    by
      intro n hn1 hn2
      have : n ≥ 6 := by sorry  -- Detailed proof omitted
      exact this
  exact ⟨b, ⟨h1, h2⟩, h3⟩

end smallest_base_for_150_l37_37310


namespace compare_exponents_compare_2004_2005_l37_37837

theorem compare_exponents (n : ℕ) (h : n > 3) : n^(n + 1) > (n + 1)^n :=
sorry

theorem compare_2004_2005 : 2004^2005 > 2005^2004 :=
begin
  apply compare_exponents 2004,
  norm_num,
end

end compare_exponents_compare_2004_2005_l37_37837


namespace symmetric_points_x_axis_l37_37100

theorem symmetric_points_x_axis (a b : ℝ) (h_a : a = -2) (h_b : b = -1) : a + b = -3 :=
by
  -- Skipping the proof steps and adding sorry
  sorry

end symmetric_points_x_axis_l37_37100


namespace contractor_fine_per_absent_day_l37_37353

theorem contractor_fine_per_absent_day :
  ∀ (total_days absent_days wage_per_day total_receipt fine_per_absent_day : ℝ),
    total_days = 30 →
    wage_per_day = 25 →
    absent_days = 4 →
    total_receipt = 620 →
    (total_days - absent_days) * wage_per_day - absent_days * fine_per_absent_day = total_receipt →
    fine_per_absent_day = 7.50 :=
by
  intros total_days absent_days wage_per_day total_receipt fine_per_absent_day
  intro h1 h2 h3 h4 h5
  sorry

end contractor_fine_per_absent_day_l37_37353


namespace div_transitivity_l37_37154

theorem div_transitivity (a b c : ℚ) : 
  (a / b = 3) → (b / c = 2 / 5) → (c / a = 5 / 6) :=
by 
  intros h1 h2
  have : c / a = (c / b) * (b / a),
  { field_simp, }
  rw h1 at this,
  rw h2 at this,
  field_simp at this,
  exact this

end div_transitivity_l37_37154


namespace directrix_of_parabola_l37_37897

-- Define the problem conditions
def parabola (x y : ℝ) : Prop :=
  x = -1/4 * y^2

-- State the theorem to prove that the directrix of the given parabola is x = 1
theorem directrix_of_parabola :
  (∀ (x y : ℝ), parabola x y → true) → 
  let d := 1 in
  ∀ (f : ℝ), (f = -d ∧ f^2 = d^2 ∧ d - f = 2) → true :=
by
  sorry

end directrix_of_parabola_l37_37897


namespace intersection_M_N_l37_37606

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {x : ℤ | x * (x - 1) = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := sorry

end intersection_M_N_l37_37606


namespace even_terms_in_binomial_expansion_l37_37973

-- Define m to be an even integer
def is_even (m : ℤ) : Prop := ∃ k : ℤ, m = 2 * k

-- Define n to be an odd integer
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Lean statement of the proof problem
theorem even_terms_in_binomial_expansion (m n : ℤ) (hm : is_even m) (hn : is_odd n) : 
  count_even_terms_in_expansion ((m + n)^8) = 8 :=
sorry

end even_terms_in_binomial_expansion_l37_37973


namespace valid_4digit_count_l37_37542

def is_valid_digit (d : ℕ) : Prop := d ≠ 5
def is_valid_A (A : ℕ) : Prop := A ≠ 0 ∧ A ≠ 5
def count_valid_4digit_numbers : ℕ :=
  let valid_A := {d | 1 ≤ d ∧ d ≤ 9 ∧ is_valid_A d}
  let valid_BC := {d | 0 ≤ d ∧ d ≤ 9 ∧ is_valid_digit d}
  let valid_D := {d | d = 0}
  card valid_A * card valid_BC * card valid_BC * card valid_D

theorem valid_4digit_count : count_valid_4digit_numbers = 648 := by
  sorry

end valid_4digit_count_l37_37542


namespace kaydence_father_age_l37_37167

-- Defining the conditions
def Kaydence_family_total_age := 200 
def Kaydence_sister_age := 40 
def Kaydence_age := 12 

-- Defining the ages of the family members in terms of father's age F
def mother_age (F : ℝ) := F - 2
def brother_age (F : ℝ) := F / 2

-- Stating the theorem
theorem kaydence_father_age : ∃ F : ℝ, F + mother_age F + brother_age F + Kaydence_sister_age + Kaydence_age = Kaydence_family_total_age ∧ F = 60 :=
by
  use 60
  simp [mother_age, brother_age, Kaydence_sister_age, Kaydence_age, Kaydence_family_total_age]
  calc
    60 + (60 - 2) + (60 / 2) + 40 + 12 = 60 + 58 + 30 + 40 + 12 : by norm_num
    ... = 200 : by norm_num
  done


end kaydence_father_age_l37_37167


namespace license_plate_combinations_l37_37024

theorem license_plate_combinations : 
  let letters := 26 in
  let positions1 := 5 in
  let positions2 := 5 - 2 in  -- since 2 positions are taken by the first letter
  let remaining_positions := 3 in -- since 3 positions are left after placing two letters twice
  let remaining_letters := 24 in  -- since 2 letters are already used
  let digits := 9 in
  (Nat.choose letters 2) * (Nat.choose positions1 2) * 
  (Nat.choose remaining_positions 2) * remaining_letters * digits = 210600 :=
by 
  sorry

end license_plate_combinations_l37_37024


namespace collinear_O1O2_through_N_l37_37087

theorem collinear_O1O2_through_N (A B C D M N O1 O2: Point)
  (hTrapezoid: AD_parallel_BC AD BC)
  (hAngle: Angle ABC > 90)
  (hPointM: OnLateralSide M AB)
  (hCircumcenter1: O1 = circumcenter (Triangle MAD))
  (hCircumcenter2: O2 = circumcenter (Triangle MBC))
  (hCircumcircle1: N ∈ circumcircle (Triangle MO1D))
  (hCircumcircle2: N ∈ circumcircle (Triangle MO2C)):
  collinear [O1, O2, N] :=
sorry

end collinear_O1O2_through_N_l37_37087


namespace weekly_loss_due_to_production_shortfall_l37_37593

-- Defining the conditions
def daily_production : ℕ := 1000
def cost_per_tire : ℕ := 250
def selling_price_factor : ℕ := 3 / 2
def potential_daily_sales : ℕ := 1200

-- Defining the problem
theorem weekly_loss_due_to_production_shortfall :
  (let selling_price_per_tire := cost_per_tire * selling_price_factor in
   let profit_per_tire := selling_price_per_tire - cost_per_tire in
   let daily_missing_tires := potential_daily_sales - daily_production in
   let daily_loss := daily_missing_tires * profit_per_tire in
   let weekly_loss := daily_loss * 7 in
   weekly_loss = 175000) := 
sorry

end weekly_loss_due_to_production_shortfall_l37_37593


namespace length_of_MN_is_zero_l37_37165

variables (A B C M N : Type) [EuclideanGeometry A B C M N]

/-- 
In a triangle ABC, M is the midpoint of BC, AN is the median to BC and AN 
is also perpendicular to BC. Given sides AB and AC have lengths 15 and 20, 
respectively, show that the length of MN is 0.
--/
theorem length_of_MN_is_zero 
  (triangle_ABC : Triangle A B C) 
  (midpoint_M : midpoint B C M)
  (median_TO_BC_AN : median A N B C)
  (perpendicular_AN_BC : perpendicular A N B C)
  (length_AB : length A B = 15)
  (length_AC : length A C = 20) : length M N = 0 :=
sorry

end length_of_MN_is_zero_l37_37165


namespace cylinder_radius_and_remaining_space_l37_37373

theorem cylinder_radius_and_remaining_space 
  (cone_radius : ℝ) (cone_height : ℝ) 
  (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cone_radius = 8 →
  cone_height = 20 →
  cylinder_height = 2 * cylinder_radius →
  (20 - 2 * cylinder_radius) / cylinder_radius = 20 / 8 →
  (cylinder_radius = 40 / 9 ∧ (cone_height - cylinder_height) = 100 / 9) :=
by
  intros cone_radius_8 cone_height_20 cylinder_height_def similarity_eq
  sorry

end cylinder_radius_and_remaining_space_l37_37373


namespace primes_in_7_pow_7_pow_n_add_1_l37_37747

theorem primes_in_7_pow_7_pow_n_add_1 (n : ℕ) : 
  ∃ ps : multiset ℕ, (∀ p ∈ ps, nat.prime p) ∧ 7^(7^n) + 1 = (ps.map coe).prod ∧ ps.card ≥ 2 * n + 3 := 
sorry

end primes_in_7_pow_7_pow_n_add_1_l37_37747


namespace trapezoid_shaded_area_fraction_l37_37389

-- Define a structure for the trapezoid
structure Trapezoid (A : Type) :=
(strips : list A)
(equal_width : ∀ i j, i ≠ j ∧ i ∈ strips ∧ j ∈ strips → width i = width j)
(shaded_strips : list A)
(shaded : ∀ s, s ∈ shaded_strips ↔ s ∈ strips ∧ is_shaded s)

-- Define the predicate to check if a strip is shaded
def is_shaded (s : Strip) : Prop := s ∈ shaded_strips

-- Define the problem as a theorem in Lean
theorem trapezoid_shaded_area_fraction
  (T : Trapezoid Strip)
  (h_strips : length T.strips = 7)
  (h_shaded : length T.shaded_strips = 4)
  : fraction_shaded_area T = 4 / 7 :=
begin
  sorry
end

end trapezoid_shaded_area_fraction_l37_37389


namespace coefficient_of_x4_y3_l37_37179

theorem coefficient_of_x4_y3 :
  ∀ (x y : ℚ), (coeff x^4 y^3 in ((1/y + x) * (x + 3*y)^6)) = 540 :=
by
  sorry

end coefficient_of_x4_y3_l37_37179


namespace least_possible_sum_of_bases_l37_37444

theorem least_possible_sum_of_bases : 
  ∃ (c d : ℕ), (2 * c + 9 = 9 * d + 2) ∧ (c + d = 13) :=
by
  sorry

end least_possible_sum_of_bases_l37_37444


namespace minimum_value_range_l37_37984

noncomputable def f (a x : ℝ) : ℝ := abs (3 * x - 1) + a * x + 2

theorem minimum_value_range (a : ℝ) :
  (-3 ≤ a ∧ a ≤ 3) ↔ ∃ m, ∀ x, f a x ≥ m := sorry

end minimum_value_range_l37_37984


namespace ellipse_standard_equation_and_fixed_point_l37_37093

theorem ellipse_standard_equation_and_fixed_point :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ b = sqrt 3 ∧ ∃ x y : ℝ,
  (x = 1 ∧ y = -3/2 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ a^2 = 4 ∧ b^2 = 3 ∧
  (∀ P Q : ℝ × ℝ, 
    (P ≠ (-2, 0) ∧ P ≠ (2, 0) ∧ 
    Q ≠ (-2, 0) ∧ Q ≠ (2, 0) ∧ 
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ 
    Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧ 
    ∃ k : ℝ, k ≠ 0 ∧ 
    ((P.2 = k * (P.1 - 2)) ∧ (Q.2 = 2 * k * (Q.1 + 2)) ∧ 
    ∃ M : ℝ × ℝ, M = (-2/3, 0) ∧
    ∃ mPQ : ℝ, mPQ = (P.2 - Q.2) / (P.1 - Q.1) ∧
    M.2 = mPQ * (M.1 - Q.1) + Q.2)))) := sorry

end ellipse_standard_equation_and_fixed_point_l37_37093


namespace find_cost_of_chocolate_l37_37643

theorem find_cost_of_chocolate
  (C : ℕ)
  (h1 : 5 * C + 10 = 90 - 55)
  (h2 : 5 * 2 = 10)
  (h3 : 55 = 90 - (5 * C + 10)):
  C = 5 :=
by
  sorry

end find_cost_of_chocolate_l37_37643


namespace find_natural_numbers_l37_37052

noncomputable def valid_n (n : ℕ) : Prop :=
  2 ^ n % 7 = n ^ 2 % 7

theorem find_natural_numbers :
  {n : ℕ | valid_n n} = {n : ℕ | n % 21 = 2 ∨ n % 21 = 4 ∨ n % 21 = 5 ∨ n % 21 = 6 ∨ n % 21 = 10 ∨ n % 21 = 15} :=
sorry

end find_natural_numbers_l37_37052


namespace closest_integer_to_sum_l37_37588

theorem closest_integer_to_sum :
  (⌊(\sqrt 2013 - 1) / sqrt 2 + 0.5⌋ : ℤ) = 31 :=
sorry

end closest_integer_to_sum_l37_37588


namespace parabola_directrix_l37_37878

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l37_37878


namespace size_of_bigger_sail_is_30_l37_37848

noncomputable def size_of_bigger_sail (speed_small_sail speed_big_sail size_small_sail : ℕ) : ℕ :=
let distance := 200 in
let time_small_sail := distance / speed_small_sail in
let time_big_sail := distance / speed_big_sail in
if time_small_sail - time_big_sail = 6 then
  let proportion := (speed_small_sail : ℚ) / (speed_big_sail : ℚ) = (size_small_sail : ℚ) / S in
  ((speed_small_sail : ℚ) / (speed_big_sail : ℚ)) * (S : ℚ) = size_small_sail
else
  sorry

theorem size_of_bigger_sail_is_30 : size_of_bigger_sail 20 50 12 = 30 :=
sorry

end size_of_bigger_sail_is_30_l37_37848


namespace point_symmetric_to_line_l37_37660

-- Define the problem statement
theorem point_symmetric_to_line (M : ℝ × ℝ) (l : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (1, 4) →
  l = (1, -1) →
  (∃ a b, N = (a, b) ∧ a + b = 5 ∧ a - b = 1) →
  N = (3, 2) :=
by
  sorry

end point_symmetric_to_line_l37_37660


namespace john_must_work_10_more_days_l37_37206

theorem john_must_work_10_more_days
  (total_days : ℕ)
  (total_earnings : ℝ)
  (daily_earnings : ℝ)
  (target_earnings : ℝ)
  (additional_days : ℕ) :
  total_days = 10 →
  total_earnings = 250 →
  daily_earnings = total_earnings / total_days →
  target_earnings = 2 * total_earnings →
  additional_days = (target_earnings - total_earnings) / daily_earnings →
  additional_days = 10 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end john_must_work_10_more_days_l37_37206


namespace real_solution_approximately_l37_37453

noncomputable def exists_real_solution (x : ℝ) : Prop :=
x = 1 - x^2 + x^4 - x^6 + x^8 - x^10 + ...

theorem real_solution_approximately :
  ∃ x : ℝ, exists_real_solution x ∧ x ≈ 0.6823 := 
sorry

end real_solution_approximately_l37_37453


namespace directrix_of_given_parabola_l37_37885

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37885


namespace minimum_chess_pieces_l37_37326

open Nat

def chess_grid (grid : list (list ℕ)) : Prop :=
  let rows := grid.map (λ r => r.sum)
  let cols := (List.range 3).map (λ c => grid.map (λ r => r[c]).sum)
  rows.length = 3 ∧ cols.length = 3 ∧
  (rows ++ cols).nodup

theorem minimum_chess_pieces : ∃ (grid : list (list ℕ)), chess_grid grid ∧ grid.sum (λ r => r.sum) = 8 :=
sorry

end minimum_chess_pieces_l37_37326


namespace determine_Q_l37_37142

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem determine_Q : Q = {2, 3, 4} :=
by
  sorry

end determine_Q_l37_37142


namespace true_statements_l37_37012

theorem true_statements :
  (∀ x : ℝ, (√(x + 1) * (2 * x - 1) ≥ 0) → x ≥ 1 / 2) ∧ 
  (∀ (x y : ℝ), (x > 1 ∧ y > 2) → (x + y > 3) ∧ ¬(x + y > 3 → x > 1 ∧ y > 2)) ∧ 
  (∀ x : ℝ, ∃ y, y = √(x ^ 2 + 2) + 1 / √(x ^ 2 + 2) ∧ y ≥ 2) →
  (∀ x : ℝ, x ^ 2 + x + 1 ≥ 0) → 
  True := 
by
  sorry

end true_statements_l37_37012


namespace inverse_graph_point_passes_through_l37_37157

theorem inverse_graph_point_passes_through :
  (∀ (f : ℝ → ℝ) (x : ℝ), (∃ (y : ℝ), y = f x) ∧ (∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x))) →
  (∃ (x y : ℝ), y = x - f x ∧ (x, y) = (1, 2)) →
  (∃ x y : ℝ, y = inverse f x - x ∧ (x, y) = (-1, 2)) :=
by
  intros hf hg
  sorry

end inverse_graph_point_passes_through_l37_37157


namespace problem_statement_l37_37108

def f (x : ℝ) : ℝ := 
  if x > 0 then x^2 - x - 1 
  else if x < 0 then -x^2 + x + 1 
  else 0  -- for x = 0, though this case is irrelevant to the problem

theorem problem_statement (x : ℝ) (h1 : f x = x^2 - x - 1) (h2 : x < 0) : f x = -x^2 + x + 1 := by
  sorry

end problem_statement_l37_37108


namespace function_domain_l37_37666

def domain_of_function (x : ℝ) : Set ℝ := 
    {y : ℝ | (1 - 2 * x > 0) ∧ (2 * x + 1 ≠ 0)}

theorem function_domain (x : ℝ) : 
    domain_of_function x = (Set.Ioo (-(real.of_rat 1 / 2)) (real.of_rat 1 / 2)) :=
by
    sorry

end function_domain_l37_37666


namespace dates_relation_l37_37630

def melanie_data_set : set ℕ :=
  { x | (x >= 1 ∧ x <= 28) ∨ (x = 29 ∧ x <= 29) ∨ (x = 30 ∧ x <= 30) ∨ (x = 31 ∧ x <= 31)}

noncomputable def median (s : set ℕ) : ℝ := sorry -- Median calculation
noncomputable def mean (s : set ℕ) : ℝ := sorry -- Mean calculation
noncomputable def modes_median (s : set ℕ) : ℝ := sorry -- Median of modes calculation

theorem dates_relation : 
  let s := melanie_data_set in
  d < mean s < median s :=
sorry

end dates_relation_l37_37630


namespace solve_inequality_l37_37651

noncomputable def inequality_solution (a x : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : Prop :=
  if h1 : 1 > a then x > 0
  else if h2 : a > 1 then (x > a^a) ∨ (0 < x ∧ x < a^(2 - a))
  else False

theorem solve_inequality (a x : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  | real.log x / real.log a - 1 | > a - 1 ↔ inequality_solution a x ha_pos ha_ne_one :=
by
  sorry

end solve_inequality_l37_37651


namespace shirts_not_washed_l37_37743

variable (short_sleeve_shirts : Nat)
variable (long_sleeve_shirts : Nat)
variable (washed_shirts : Nat)

theorem shirts_not_washed (h1 : short_sleeve_shirts = 40) 
                          (h2 : long_sleeve_shirts = 23) 
                          (h3 : washed_shirts = 29) :
    short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 34 :=
  by
  rw [h1, h2, h3]
  exact Nat.add_sub_cancel _ _

  -- By rewrite (rw) and exact commands to handle the given hypothesis

end shirts_not_washed_l37_37743


namespace solution_range_of_a_l37_37533

theorem solution_range_of_a (a : ℝ) (x y : ℝ) :
  3 * x + y = 1 + a → x + 3 * y = 3 → x + y < 2 → a < 4 :=
by
  sorry

end solution_range_of_a_l37_37533


namespace maya_lift_increase_l37_37225

def initial_lift_America : ℕ := 240
def peak_lift_America : ℕ := 300

def initial_lift_Maya (a_lift : ℕ) : ℕ := a_lift / 4
def peak_lift_Maya (p_lift : ℕ) : ℕ := p_lift / 2

def lift_difference (initial_lift : ℕ) (peak_lift : ℕ) : ℕ := peak_lift - initial_lift

theorem maya_lift_increase :
  lift_difference (initial_lift_Maya initial_lift_America) (peak_lift_Maya peak_lift_America) = 90 :=
by
  -- Proof is skipped with sorry
  sorry

end maya_lift_increase_l37_37225


namespace max_license_plates_l37_37005

theorem max_license_plates (n : ℕ) (h_n : n = 6) : 
  (∃ P : ℕ, P = 10^(n-1) ∧ ∀ x y : ℕ, x ≠ y → count_different_positions x y ≥ 2) :=
begin
  -- Given that n = 6, we need to show the existence of P = 10^(n-1)
  sorry
end

-- Define count_different_positions to clarify the requirement:
def count_different_positions (x y : ℕ) : ℕ :=
  -- Calculate the number of differing positions in the decimal representation of x and y
  sorry

end max_license_plates_l37_37005


namespace popton_school_bus_total_toes_l37_37637

-- Define the number of toes per hand for each race
def toes_per_hand_hoopit : ℕ := 3
def toes_per_hand_neglart : ℕ := 2
def toes_per_hand_zentorian : ℕ := 4

-- Define the number of hands for each race
def hands_per_hoopit : ℕ := 4
def hands_per_neglart : ℕ := 5
def hands_per_zentorian : ℕ := 6

-- Define the number of students from each race on the bus
def num_hoopits : ℕ := 7
def num_neglarts : ℕ := 8
def num_zentorians : ℕ := 5

-- Calculate the total number of toes on the bus
def total_toes_on_bus : ℕ :=
  num_hoopits * (toes_per_hand_hoopit * hands_per_hoopit) +
  num_neglarts * (toes_per_hand_neglart * hands_per_neglart) +
  num_zentorians * (toes_per_hand_zentorian * hands_per_zentorian)

-- Theorem stating the number of toes on the bus
theorem popton_school_bus_total_toes : total_toes_on_bus = 284 :=
by
  sorry

end popton_school_bus_total_toes_l37_37637


namespace count_ordered_pairs_l37_37904

theorem count_ordered_pairs : 
  ∃ n : ℕ, n = 136 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℤ, a * x + b * y = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) → n = 136 := 
sorry

end count_ordered_pairs_l37_37904


namespace parabola_standard_equation_l37_37427

theorem parabola_standard_equation (x y : ℝ) 
  (focus_condition : 3 * x - 4 * y - 12 = 0) :
  ∃ p : ℝ, (p = 3 / 2) ∧
           (let eqn := (x ^ 2 = 4 * p * y) in eqn = x^2 = 6 * y) ∧
           (let directrix := (y = p * 2) in directrix = y = 3):=
by
  sorry

end parabola_standard_equation_l37_37427


namespace arianna_sleep_hours_l37_37405

-- Defining the given conditions
def total_hours_in_a_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_in_class : ℕ := 3
def hours_at_gym : ℕ := 2
def hours_on_chores : ℕ := 5

-- Formulating the total hours spent on activities
def total_hours_on_activities := hours_at_work + hours_in_class + hours_at_gym + hours_on_chores

-- Proving Arianna's sleep hours
theorem arianna_sleep_hours : total_hours_in_a_day - total_hours_on_activities = 8 :=
by
  -- Direct proof placeholder, to be filled in with actual proof steps or tactic
  sorry

end arianna_sleep_hours_l37_37405


namespace limit_zero_l37_37095

variable (a : ℕ → ℝ)

theorem limit_zero (h : Tendsto (λ n, a (n + 1) - (1 / 2) * a n) atTop (nhds 0)) : 
  Tendsto a atTop (nhds 0) :=
sorry

end limit_zero_l37_37095


namespace four_digit_sum_divisible_l37_37357

theorem four_digit_sum_divisible (A B C D : ℕ) :
  (10 * A + B + 10 * C + D = 94) ∧ (1000 * A + 100 * B + 10 * C + D % 94 = 0) →
  false :=
by
  sorry

end four_digit_sum_divisible_l37_37357


namespace polynomial_simplification_l37_37903

theorem polynomial_simplification (x : ℝ) : 
  (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2 * x^3 - 4 * x^2 + 10 * x + 1 :=
by
  sorry

end polynomial_simplification_l37_37903


namespace purse_with_less_than_M_dollars_l37_37742

variables (M N : ℕ)
variables (c s : ℕ → ℕ)
variables (j : ℕ) (hM : 0 < M) (hN : 0 < N)
variables (h_sum_c : ∑ j in finset.range M, c j = N)
variables (h_sum_s : ∑ j in finset.range M, s j = 2003)
variables (hN_gt_sj : ∀ j < M, N > s j)

theorem purse_with_less_than_M_dollars :
  ∃ i < N, c i < M :=
sorry

end purse_with_less_than_M_dollars_l37_37742


namespace solve_system_of_equations_l37_37532

theorem solve_system_of_equations : ∃ (x y : ℝ), 
  y * x^(Real.log x / Real.log y) = x^2.5 ∧ 
  (Real.log 3 y) * (Real.log y (y - 2 * x)) = 1 ∧ 
  x = 3 ∧ y = 9 :=
by
  existsi (3 : ℝ), (9 : ℝ)
  split
  sorry

end solve_system_of_equations_l37_37532


namespace rounding_to_thousand_representation_correct_l37_37323

theorem rounding_to_thousand_representation_correct:
  (approximation_precision_5000 : (∃ p : ℕ, 5000 = p)) →
  (approximation_precision_5_thousand : (∃ p : ℕ, 5 * 1000 = p)) →
  (precision_8_4 : ∃ t : ℕ, t / 10 = 8.4) →
  (precision_0_7 : ∃ t : ℕ, t / 10 = 0.7) →
  (precision_2_46_ten_thousand : ∃ p : ℕ, (2.46 * 10000) = p) →
  (∃ k : ℕ, 317500 / 1000 = k) →
  (∃ p : ℕ, (318000 / 10000) = 31.8 * 10000 / 10000) :=
by
  intros
  sorry

end rounding_to_thousand_representation_correct_l37_37323


namespace manuscript_fee_tax_l37_37583

theorem manuscript_fee_tax :
  ∀ (M : ℕ), (M ≤ 800 → 0 = 420) ∧ 
             (800 < M ∧  M ≤ 4000 → 0.14 * (M - 800) = 420) ∧ 
             (M > 4000 → 0.11 * M = 420) → 
             M = 3800 :=
by
  sorry

end manuscript_fee_tax_l37_37583


namespace projection_matrix_exists_l37_37447

noncomputable def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, (20 : ℚ) / 49], ![c, (29 : ℚ) / 49]]

theorem projection_matrix_exists :
  ∃ (a c : ℚ), P a c * P a c = P a c ∧ a = (20 : ℚ) / 49 ∧ c = (29 : ℚ) / 49 := 
by
  use ((20 : ℚ) / 49), ((29 : ℚ) / 49)
  simp [P]
  sorry

end projection_matrix_exists_l37_37447


namespace approx_participants_l37_37575

def is_near (x y: ℕ) (precision: ℕ): Prop :=
  abs (x - y) ≤ precision

theorem approx_participants:
  let participants := 21780
  let expected_approx := 22000
  is_near participants expected_approx 1000 :=
sorry

end approx_participants_l37_37575


namespace odd_number_divides_3n_plus_1_l37_37865

theorem odd_number_divides_3n_plus_1 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n ∣ 3^n + 1) : n = 1 :=
by
  sorry

end odd_number_divides_3n_plus_1_l37_37865


namespace remaining_pencils_check_l37_37193

variables (Jeff_initial : ℕ) (Jeff_donation_percentage : ℚ) (Vicki_ratio : ℚ) (Vicki_donation_fraction : ℚ)

def Jeff_donated_pencils := (Jeff_donation_percentage * Jeff_initial).toNat
def Jeff_remaining_pencils := Jeff_initial - Jeff_donated_pencils

def Vicki_initial_pencils := (Vicki_ratio * Jeff_initial).toNat
def Vicki_donated_pencils := (Vicki_donation_fraction * Vicki_initial_pencils).toNat
def Vicki_remaining_pencils := Vicki_initial_pencils - Vicki_donated_pencils

def total_remaining_pencils := Jeff_remaining_pencils + Vicki_remaining_pencils

theorem remaining_pencils_check
    (Jeff_initial : ℕ := 300)
    (Jeff_donation_percentage : ℚ := 0.3)
    (Vicki_ratio : ℚ := 2)
    (Vicki_donation_fraction : ℚ := 0.75) :
    total_remaining_pencils Jeff_initial Jeff_donation_percentage Vicki_ratio Vicki_donation_fraction = 360 :=
by
  sorry

end remaining_pencils_check_l37_37193


namespace email_sending_ways_l37_37652

theorem email_sending_ways :
  let email_addresses := 3
  let emails := 5
  email_addresses ^ emails = 243 :=
begin
  sorry
end

end email_sending_ways_l37_37652


namespace sum_of_squares_inequality_l37_37935

noncomputable theory

variables {n : ℕ} (x : ℕ → ℝ)

theorem sum_of_squares_inequality (hpos : ∀ i, 1 ≤ i → i ≤ n → 0 < x i) :
  (∑ i in finset.range n, (x i)^2 / (x ((i + 1) % n))) ≥ (∑ i in finset.range n, x i) :=
sorry

end sum_of_squares_inequality_l37_37935


namespace measure_of_angle_C_l37_37018

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 12 * D) : C = 2160 / 13 :=
by {
  sorry
}

end measure_of_angle_C_l37_37018


namespace find_b_l37_37547

noncomputable def a_and_b_integers_and_factor (a b : ℤ) : Prop :=
  ∀ (x : ℝ), (x^2 - x - 1) * (a*x^3 + b*x^2 - x + 1) = 0

theorem find_b (a b : ℤ) (h : a_and_b_integers_and_factor a b) : b = -1 :=
by 
  sorry

end find_b_l37_37547


namespace operation_on_b_l37_37661

theorem operation_on_b (t b0 b1 : ℝ) (h : t * b1^4 = 16 * t * b0^4) : b1 = 2 * b0 :=
by
  sorry

end operation_on_b_l37_37661


namespace RS_segment_length_l37_37051

theorem RS_segment_length (P Q R S : ℝ) (r1 r2 : ℝ) (hP : P = 0) (hQ : Q = 10) (rP : r1 = 6) (rQ : r2 = 4) :
    (∃ PR QR SR : ℝ, PR = 6 ∧ QR = 4 ∧ SR = 6) → (R - S = 12) :=
by
  sorry

end RS_segment_length_l37_37051


namespace direction_cosines_l37_37056

theorem direction_cosines (x y z : ℝ) (α β γ : ℝ)
  (h1 : 2 * x - 3 * y - 3 * z - 9 = 0)
  (h2 : x - 2 * y + z + 3 = 0) :
  α = 9 / Real.sqrt 107 ∧ β = 5 / Real.sqrt 107 ∧ γ = 1 / Real.sqrt 107 :=
by
  -- Here, we will sketch out the proof to establish that these values for α, β, and γ hold.
  sorry

end direction_cosines_l37_37056


namespace freddy_is_7_l37_37073

noncomputable theory

def freddy_age (job_age : ℕ) (oliver_age : ℕ) (stephanie_age : ℕ) (tim_age : ℕ) (tina_age : ℕ) (freddy_age : ℕ) : Prop :=
  (freddy_age = tina_age + 2) ∧ 
  (tina_age = oliver_age / 3) ∧ 
  (tim_age = oliver_age / 2) ∧ 
  (3 * stephanie_age = job_age + tim_age) ∧ 
  (freddy_age = stephanie_age - 2.5) ∧ 
  (job_age = 5) ∧ 
  (oliver_age = job_age + 10)

theorem freddy_is_7 :
  ∃ (freddy_age : ℕ), ∃ (job_age oliver_age stephanie_age tim_age tina_age : ℕ),
    freddy_age (job_age) (oliver_age) (stephanie_age) (tim_age) (tina_age) freddy_age ∧ freddy_age = 7 :=
begin
  use 7,
  use 5, -- job_age
  use 15, -- oliver_age
  use 125 / 30, -- stephanie_age in decimal form
  use 15 / 2, -- tim_age in decimal form
  use 15 / 3, -- tina_age
  split,
  { sorry }, -- the conditions will be verified here
  { refl } -- proving freddy_age = 7
end

end freddy_is_7_l37_37073


namespace prob_b_score_10_distribution_X_eq_expectation_X_l37_37702

-- Define the probabilities
def p_a_correct := 2 / 3
def p_b_correct := 4 / 5

-- Define the score calculation mechanism
def score_correct := 10
def score_other_correct := 0
def score_other_incorrect := 5

-- Probability that player B's total score is 10 points
theorem prob_b_score_10 : 
  let p := 2 * (1/2) * p_b_correct * (1/2) * (1 - p_b_correct) + 
           2 * (1/2) * p_b_correct * (1/2) * p_a_correct + 
           (1/2) * (1 - p_a_correct) * (1/2) * (1 - p_a_correct) in
  p = 337 / 900 :=
sorry

-- Distribution and expectation for player A's total score X
def prob_X_0 := (1/2) * (1 - p_a_correct) * (1/2) * (1 - p_a_correct) + 
                2 * (1/2) * (1 - p_a_correct) * (1/2) * p_b_correct + 
                (1/2) * p_b_correct * (1/2) * p_b_correct

def prob_X_5 := 2 * (1/2) * (1 - p_b_correct) * (1/2) * (1 - p_a_correct) + 
                2 * (1/2) * p_b_correct * (1/2) * (1 - p_b_correct)

def prob_X_10 := 2 * (1/2) * p_a_correct * (1/2) * (1 - p_a_correct) +
                 (1/2) * (1 - p_b_correct) * (1/2) * (1 - p_b_correct) +
                 2 * (1/2) * p_a_correct * (1/2) * p_b_correct

def prob_X_15 := 2 * (1/2) * (1 - p_b_correct) * (1/2) * p_a_correct

def prob_X_20 := (1/2) * p_a_correct * (1/2) * p_a_correct

def distribution_X (X : ℕ) : ℚ :=
  match X with
  | 0  => prob_X_0
  | 5  => prob_X_5
  | 10 => prob_X_10
  | 15 => prob_X_15
  | 20 => prob_X_20
  | _  => 0

theorem distribution_X_eq :
  distribution_X 0 = 289 / 900 ∧
  distribution_X 5 = 17 / 150 ∧
  distribution_X 10 = 349 / 900 ∧
  distribution_X 15 = 1 / 15 ∧
  distribution_X 20 = 1 / 9 :=
sorry

theorem expectation_X :
  let E := 0 * (289 / 900) + 5 * (17 / 150) + 10 * (349 / 900) + 15 * (1 / 15) + 20 * (1 / 9) in
  E = 23 / 3 :=
sorry

end prob_b_score_10_distribution_X_eq_expectation_X_l37_37702


namespace area_of_parallelogram_l37_37291

theorem area_of_parallelogram (base : ℝ) (height : ℝ)
  (h1 : base = 3.6)
  (h2 : height = 2.5 * base) :
  base * height = 32.4 :=
by
  sorry

end area_of_parallelogram_l37_37291


namespace total_weight_proof_l37_37600
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end total_weight_proof_l37_37600


namespace sum_of_derivatives_at_2_and_neg_2_l37_37270

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + m * x^2 + 2 * x + 5

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := 3 * x^2 + 2 * m * x + 2

theorem sum_of_derivatives_at_2_and_neg_2 (m : ℝ) : f'(2, m) + f'(-2, m) = 28 := 
by 
  simp [f', pow_two, mul_assoc, mul_add, add_assoc]
  sorry

end sum_of_derivatives_at_2_and_neg_2_l37_37270


namespace infinite_or_no_parallelograms_l37_37098

/-- There are infinitely many parallelograms with vertices on four mutually skew lines
    if the perpendicular bisector planes of the segments joining points on each pair
    of lines intersect. Otherwise, no parallelogram can be formed if these planes are parallel and separate. -/
theorem infinite_or_no_parallelograms {a b c d : Line} 
  (skew_a_b : Skew a b) (skew_a_c : Skew a c) (skew_a_d : Skew a d) 
  (skew_b_c : Skew b c) (skew_b_d : Skew b d) (skew_c_d : Skew c d) :
  (∃ σ_ab σ_cd : Plane, (perpendicular_bisector_plane a b σ_ab) ∧ (perpendicular_bisector_plane c d σ_cd) ∧ (σ_ab ∩ σ_cd ≠ ∅)) →
  ∃ infinie_parallelgrams_lines (a b c d),
  ∃ P Q R S, (P ∈ a ∧ Q ∈ b ∧ R ∈ c ∧ S ∈ d) ∧ parallelogram P Q R S :=
sorry

end infinite_or_no_parallelograms_l37_37098


namespace range_of_m_l37_37929

noncomputable def problem (m : ℝ) : Prop :=
  let p := ∀ x : ℝ, |x| + |x - 1| > m
  let q := ∀ x1 x2 : ℝ, x1 < x2 → -(7 - 3 * m) ^ x1 > -(7 - 3 * m) ^ x2
  (p ∧ ¬q) ∨ (¬p ∧ q)

theorem range_of_m :
  ∀ m : ℝ, (problem m ↔ 1 ≤ m ∧ m < 2) :=
begin
  intro m,
  split,
  {
    intro H,
    split,
    {
      sorry,
    },
    {
      sorry,
    }
  },
  {
    intro H,
    cases H with H1 H2,
    -- Check exactly one of p or q is true
    unfold problem,
    rw [and_comm (¬q), or_comm],
    {
      sorry,
    }
  }
end

end range_of_m_l37_37929


namespace quadrilateral_is_rectangle_l37_37432

-- Given a quadrilateral ABCD
variables (A B C D : Type) [AffineSpace ℝ A]

-- Define similarity condition for triangles
def triangles_similar (A B C D : A) :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (triangle.side_length A B = k * triangle.side_length C D) ∧
    (triangle.angle A B C = triangle.angle C D A)

-- State the main proposition
theorem quadrilateral_is_rectangle 
  (A B C D : A)
  (H1 : triangles_similar A D B C)
  (H2 : triangles_similar C D A B)
  (H3 : triangles_similar B C D A)
  (H4 : triangles_similar D A B C) : 
  is_rectangle A B C D := 
  sorry

end quadrilateral_is_rectangle_l37_37432


namespace optimal_investment_plan_l37_37818

variables {R : Type*} [linear_ordered_field R] [topological_space R] [opens_measurable_space R]
  [derivable_space R] [order_topology R] [density_space R]

noncomputable def profit_A (x : R) (a : R) := a * (x - 1) + 2
noncomputable def profit_B (x : R) (b : R) := 6 * log (x + b)

def find_a_b (h₀ : profit_A 0 a = 0) (h₁ : profit_B 0 b = 0) (ha : a > 0) (hb : b > 0) :
  a = 2 ∧ b = 1 :=
by sorry

def total_profit (x : R) := 6 * log (x + 1) - 2 * x + 10

noncomputable def maximum_profit (total_investment : R) (h : 0 < total_investment) :
  ∃ x, 0 < x ∧ x ≤ total_investment ∧ (∀ y, 0 < y → y ≤ total_investment → total_profit y ≤ total_profit x) :=
by sorry

theorem optimal_investment_plan :
  ∃ (xa xb : R), xa = 3 ∧ xb = 2 ∧ total_profit 2 = 6 * log (2 + 1) + 6 :=
by sorry

end optimal_investment_plan_l37_37818


namespace pencils_total_l37_37189

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l37_37189


namespace sum_of_solutions_eq_zero_l37_37063

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) + 4 * abs x

theorem sum_of_solutions_eq_zero : 
  (∃ x : ℝ, f x = 20) ∧ (∃ y : ℝ, f y = 20 ∧ x = -y) → 
  x + y = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l37_37063


namespace coin_flip_probability_l37_37156

theorem coin_flip_probability :
  (∀ (p : ℝ), p = 0.5 → (choose 3 2) * p^2 * (1 - p)^(3 - 2) = 0.375) → p = 0.5 :=
by
  intro h
  specialize h 0.5 rfl
  have h0 : (3:ℝ).choose 2 = 3 := by norm_num
  simp at h
  linarith

end coin_flip_probability_l37_37156


namespace ellipse_standard_equation_and_fixed_point_l37_37094

theorem ellipse_standard_equation_and_fixed_point :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ b = sqrt 3 ∧ ∃ x y : ℝ,
  (x = 1 ∧ y = -3/2 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ a^2 = 4 ∧ b^2 = 3 ∧
  (∀ P Q : ℝ × ℝ, 
    (P ≠ (-2, 0) ∧ P ≠ (2, 0) ∧ 
    Q ≠ (-2, 0) ∧ Q ≠ (2, 0) ∧ 
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ 
    Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧ 
    ∃ k : ℝ, k ≠ 0 ∧ 
    ((P.2 = k * (P.1 - 2)) ∧ (Q.2 = 2 * k * (Q.1 + 2)) ∧ 
    ∃ M : ℝ × ℝ, M = (-2/3, 0) ∧
    ∃ mPQ : ℝ, mPQ = (P.2 - Q.2) / (P.1 - Q.1) ∧
    M.2 = mPQ * (M.1 - Q.1) + Q.2)))) := sorry

end ellipse_standard_equation_and_fixed_point_l37_37094


namespace rectangle_area_l37_37369

theorem rectangle_area (x : ℝ) (h1 : (x^2 + (3*x)^2) = (15*Real.sqrt 2)^2) :
  (x * (3 * x)) = 135 := 
by
  sorry

end rectangle_area_l37_37369


namespace compare_negatives_l37_37029

theorem compare_negatives : -0.5 > -0.7 := 
by 
  exact sorry 

end compare_negatives_l37_37029


namespace draw_4_balls_in_order_l37_37776

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37776


namespace jerry_needs_money_l37_37338

theorem jerry_needs_money
  (jerry_has : ℕ := 7)
  (total_needed : ℕ := 16)
  (cost_per_figure : ℕ := 8) :
  (total_needed - jerry_has) * cost_per_figure = 72 :=
by
  sorry

end jerry_needs_money_l37_37338


namespace induction_term_added_l37_37712

theorem induction_term_added (k : ℕ) (h : 0 < k) :
  (k+2)*(k+3)*...*(2*k+1)*(2*k+2) / (k+1) = 2 * (2 * k + 1) :=
sorry

end induction_term_added_l37_37712


namespace simplify_fraction_l37_37249

theorem simplify_fraction (x : ℝ) :
  ((x + 2) / 4) + ((3 - 4 * x) / 3) = (18 - 13 * x) / 12 := by
  sorry

end simplify_fraction_l37_37249


namespace probability_fourth_roll_six_l37_37839

-- We assume a fair die and a biased die with the given probabilities
def fair_die_probability (side : ℕ) : ℚ :=
  if side = 6 then 1 / 6 else 1 / 6

def biased_die_probability (side : ℕ) : ℚ :=
  if side = 6 then 3 / 4 else 1 / 20

-- Given the conditions described in the problem
def conditions (first_three_rolls_six : Prop) (p q : ℕ) : Prop :=
  -- "Charles randomly selects one of the two dice and rolls it four times"
  (first_three_rolls_six → p + q = 1607)

-- We stipulate that the first three rolls are sixes
axiom first_three_rolls_six : Prop := sorry

-- Now we state the main proof problem:
theorem probability_fourth_roll_six (p q : ℕ) : conditions first_three_rolls_six p q :=
by
  sorry

end probability_fourth_roll_six_l37_37839


namespace find_n_l37_37725

theorem find_n : ∃ n : ℕ, n > 0 ∧ (let a := (7 + 14 + 21 + 28 + 35 + 42 + 49) / 7 in
                                  let b := 2 * n in
                                  a^2 - b^2 = 0 ∧ n = 14) :=
by
  sorry

end find_n_l37_37725


namespace total_shoes_l37_37188

theorem total_shoes :
  let daniel_shoes := 15 in
  let christopher_shoes := ∑ (one_half := 2.5 * daniel_shoes).toNat (2 in
  let brian_shoes := christopher_shoes + 5 in
  let edward_shoes := 3.5 * brian_shoes in
  let jacob_shoes := (2 / 3 : ℝ) * edward_shoes in
  daniel_shoes + christopher_shoes + brian_shoes + edward_shoes + jacob_shoes = 339 :=
by
  sorry

end total_shoes_l37_37188


namespace gcd_divisors_leaving_remainders_l37_37729

theorem gcd_divisors_leaving_remainders :
  let greatestDivisor := Int.gcd (3815 - 31) (4521 - 33)
  in greatestDivisor = 64 := by
    sorry

end gcd_divisors_leaving_remainders_l37_37729


namespace smallest_perfect_cube_divisor_l37_37615

-- Conditions
variables {p q r s : ℕ}
variables [nat.prime p] [nat.prime q] [nat.prime r] [nat.prime s]
variables (n : ℕ)
hypothesis h1 : p ≠ q
hypothesis h2 : p ≠ r
hypothesis h3 : p ≠ s
hypothesis h4 : q ≠ r
hypothesis h5 : q ≠ s
hypothesis h6 : r ≠ s
hypothesis h7 : n = p^2 * q^2 * r^3 * s

-- Statement
theorem smallest_perfect_cube_divisor : ∃ m : ℕ, m = (p * q * r^3 * s)^3 ∧ n ∣ m :=
by {
  -- Proof would go here
  sorry
}

end smallest_perfect_cube_divisor_l37_37615


namespace pencils_total_l37_37190

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l37_37190


namespace poly_coeff_ratio_l37_37619

theorem poly_coeff_ratio (a_n a_n_minus1 ... a_0 c_n_plus1 c_n ... c_0 : ℝ)
  (h₁ : ∃ (γ : ℝ), g(x) = (x + γ) * f(x))
  (h₂ : ∀ i, 0 ≤ i → i ≤ n → |c_i| ≤ c)
  (h₃ : ∀ i, 0 ≤ i → i ≤ n-1 → |a_i| ≤ a)
  (h₄ : g(x) = c_n_plus1 * x^(n+1) + c_n * x^n + ... + c_0)
  (h₅ : f(x) = a_n * x^n + a_n_minus1 * x^(n-1) + ... + a_0) :
  a / c ≤ n + 1 := by
  sorry

end poly_coeff_ratio_l37_37619


namespace find_actual_time_when_office_clock_shows_3pm_l37_37263

noncomputable def faulty_clock (
  same_time_initial: ℝ,
  wristwatch_end: ℝ,
  office_end: ℝ,
  office_shows_when_return: ℝ
): Prop :=
  same_time_initial = 8 ∧
  wristwatch_end = 9 ∧
  office_end = 9 + 10/60 ∧ -- 9:10 AM
  (same_time_initial + 6 = 14) ∧ -- 14:00 or 2:00 PM,
  office_shows_when_return = 15 ∧
  (let rate := office_end / wristwatch_end in
   let actual_time := office_shows_when_return / rate in
   actual_time = 14) -- equivalent to 2:00 PM
theorem find_actual_time_when_office_clock_shows_3pm :
  faulty_clock 8 9 9.166666666666666 15 := -- 9.166666... is 9 + 10/60 and 15 is 3:00 PM
sorry

end find_actual_time_when_office_clock_shows_3pm_l37_37263


namespace sum_f_1_to_2017_l37_37523

noncomputable def f (x : ℕ) := Real.cos (Real.pi / 3 * x)

theorem sum_f_1_to_2017 : (∑ k in Finset.range 2017, f (k + 1)) = (1 / 2) := by
  sorry

end sum_f_1_to_2017_l37_37523


namespace solve_tan_eq_l37_37253

noncomputable theory

open Real

theorem solve_tan_eq (z : ℝ) (k n : ℤ) :
  (tan (4 * z) / tan (2 * z) + tan (2 * z) / tan (4 * z) + 5 / 2 = 0) ∧ 
  (tan (2 * z) ≠ 0) ∧ (tan (4 * z) ≠ 0) ∧ (cos (2 * z) ≠ 0) ∧ (cos (4 * z) ≠ 0) → 
  (z = (1 / 2 : ℝ) * arctan (sqrt 5) + (π * k) / 2) ∨ (z = -(1 / 2 : ℝ) * arctan (sqrt 5) + (π * k) / 2) ∨ 
  (z = (1 / 2 : ℝ) * arctan (sqrt 2) + (π * n) / 2) ∨ (z = -(1 / 2 : ℝ) * arctan (sqrt 2) + (π * n) / 2) :=
sorry

end solve_tan_eq_l37_37253


namespace school_raised_amount_correct_l37_37709

def school_fundraising : Prop :=
  let mrsJohnson := 2300
  let mrsSutton := mrsJohnson / 2
  let missRollin := mrsSutton * 8
  let topThreeTotal := missRollin * 3
  let mrEdward := missRollin * 0.75
  let msAndrea := mrEdward * 1.5
  let totalRaised := mrsJohnson + mrsSutton + missRollin + mrEdward + msAndrea
  let adminFee := totalRaised * 0.02
  let maintenanceExpense := totalRaised * 0.05
  let totalDeductions := adminFee + maintenanceExpense
  let finalAmount := totalRaised - totalDeductions
  finalAmount = 28737

theorem school_raised_amount_correct : school_fundraising := 
by 
  sorry

end school_raised_amount_correct_l37_37709


namespace john_must_work_10_more_days_l37_37202

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l37_37202


namespace real_solution_l37_37433

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end real_solution_l37_37433


namespace coefficient_x3_in_expansion_l37_37439

theorem coefficient_x3_in_expansion : 
  (let T (r : ℕ) := Nat.choose 5 r * (-2)^r;
  let coeff_x2 := T 3 * 1;
  let coeff_x3 := T 2 * 1;
  coeff_x2 + coeff_x3 = -40) :=
sorry

end coefficient_x3_in_expansion_l37_37439


namespace translation_correctness_and_characteristics_summary_l37_37339

/--
Given the classical Chinese sentences: 
  - '之'
  - '绝出'
  - '以'
  - '籍'
  - '挠'
  - '谁何'
Prove their translations are:
  - When he took the exam, one of the examiners got his paper and thought it was outstanding, so he showed it to other examiners.
  - The family of Yang Wenmin has been noble and prominent for generations, causing trouble in the administration, and no previous officials could do anything about it.
Additionally, given the context of actions, behaviors, propositions, emotions, thoughts, morals, personality, aspirations, intelligence, and the comprehensive nature, prove that Shen Jun's characteristics in office can be summarized as:
  - upright and outspoken
  - disdainful of sycophancy
  - fair and lawful
  - considerate of the people.
-/
theorem translation_correctness_and_characteristics_summary :
  (translate "之" == "When he took the exam, one of the examiners got his paper and thought it was outstanding, so he showed it to other examiners.") ∧
  (translate "绝出" == "The family of Yang Wenmin has been noble and prominent for generations, causing trouble in the administration, and no previous officials could do anything about it.") ∧
  (summarize Shen_Jun_characteristics "third_paragraph" == "upright and outspoken, disdainful of sycophancy, fair and lawful, and considerate of the people.")
:=
sorry

end translation_correctness_and_characteristics_summary_l37_37339


namespace no_five_congruent_connected_regions_l37_37703

-- Define Mars as a sphere
def mars_sphere : Type := sorry

-- Define a research lab
structure ResearchLab :=
  (location : mars_sphere)

-- Define the condition of having exactly five research labs
def has_five_labs (labs : list ResearchLab) : Prop :=
  labs.length = 5

-- Define what it means for regions to be connected and cover Mars
def connected_cover (regions : list (set mars_sphere)) : Prop :=
  (∀ r ∈ regions, r.nonempty) ∧
  (∀ r₁ r₂ ∈ regions, r₁ ≠ r₂ → disjoint r₁ r₂) ∧
  (⋃₀ regions = set.univ)

-- Define what it means for regions to be congruent
def congruent_regions (regions : list (set mars_sphere)) : Prop :=
  ∀ r₁ r₂ ∈ list.to_finset regions, is_congruent r₁ r₂

-- Define is_congruent as a placeholder, usually involving isometry properties
def is_congruent (r₁ r₂ : set mars_sphere) : Prop := sorry

-- The main theorem statement
theorem no_five_congruent_connected_regions
  (laboratories : list ResearchLab)
  (h_five_labs : has_five_labs laboratories) :
  ¬ ∃ (regions : list (set mars_sphere)), connected_cover regions ∧
  congruent_regions regions ∧
  (∀ (lab : ResearchLab) (r ∈ regions), lab.location ∈ r) :=
by sorry

end no_five_congruent_connected_regions_l37_37703


namespace value_of_k_l37_37557

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l37_37557


namespace Int_G_is_group_l37_37746

variables (G : Type) [group G]

def Int_G : set (G →* G) :=
  { f | ∃ x : G, f = λ g, x * g * x⁻¹ }

theorem Int_G_is_group : is_group (Int_G G) :=
by {
  sorry
}

end Int_G_is_group_l37_37746


namespace range_of_g_l37_37041

open Real

noncomputable def g (x : ℝ) : ℝ := x / (x^2 + x + 1)

theorem range_of_g : (set.range g) = set.Icc (-1 : ℝ) (1/3) := 
sorry

end range_of_g_l37_37041


namespace arithmetic_sequence_common_difference_l37_37116

theorem arithmetic_sequence_common_difference (a_1 a_5 d : ℝ) 
  (h1 : a_5 = a_1 + 4 * d) 
  (h2 : a_1 + (a_1 + d) + (a_1 + 2 * d) = 6) : 
  d = 2 := 
  sorry

end arithmetic_sequence_common_difference_l37_37116


namespace base_conversion_l37_37437

theorem base_conversion (k : ℕ) : (5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k + 4) → k = 7 :=
by 
  let x := 5 * 8^2 + 2 * 8^1 + 4 * 8^0
  have h : x = 340 := by sorry
  have hk : 6 * k^2 + 6 * k + 4 = 340 := by sorry
  sorry

end base_conversion_l37_37437


namespace total_biscuits_needed_l37_37632

   theorem total_biscuits_needed (num_dogs : ℕ) (biscuits_per_dog : ℕ) (h1 : num_dogs = 2) (h2 : biscuits_per_dog = 3) : num_dogs * biscuits_per_dog = 6 :=
   by
     rw [h1, h2]
     norm_num
   
end total_biscuits_needed_l37_37632


namespace number_of_ways_to_draw_balls_l37_37784

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l37_37784


namespace students_eat_in_cafeteria_l37_37794

variable (C : ℕ)
variable h_total_students : 60 = C + 3 * C + 20

theorem students_eat_in_cafeteria (C : ℕ) (h : 60 = C + 3 * C + 20) : C = 10 :=
sorry

end students_eat_in_cafeteria_l37_37794


namespace expected_participants_1999_l37_37180

variable (P : ℕ → ℕ) -- number of participants as function of year

-- Conditions
def initial_participation := P 1996 = 800
def annual_increase := ∀ n, P (n + 1) = P n * 3 / 2

-- Conclusion to prove
theorem expected_participants_1999 (h₀ : initial_participation) (h₁ : annual_increase) : P 1999 = 2700 :=
by
  sorry

end expected_participants_1999_l37_37180


namespace length_of_BD_l37_37166

theorem length_of_BD {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (AC BC : ℝ) (AC_eq : AC = 10) (BC_eq : BC = 10)
  (AB : ℝ) (AB_eq : AB = 6)
  (D_on_AB : D ∈ line_segment (A : point) (B : point))
  (B_between_AD : between_r A B D)
  (CD : ℝ) (CD_eq : CD = 12) :
  ∃ BD : ℝ, BD = real.sqrt 53 - 3 :=
by
  sorry

end length_of_BD_l37_37166


namespace smaller_molds_radius_l37_37359

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  (2/3) * real.pi * r^3

theorem smaller_molds_radius :
  ∀ (r_large r_small : ℝ), 
  (r_large = 2) →
  (volume_of_hemisphere r_large = 64 * volume_of_hemisphere r_small) →
  r_small = 1 / 2 :=
by
  intros r_large r_small h_large h_volume
  sorry

end smaller_molds_radius_l37_37359


namespace exists_tangent_line_through_L_l37_37925

noncomputable def triangle_perimeter {B Q P L A C : Point} (p : ℝ) :=
  ∠QBP ∧ L ∉ InteriorAngle QBP ∧ LineThrough L A ∧ LineThrough L C ∧
  Meets BQ A ∧ Meets BP C ∧ perimeter_triangle A B C = 2 * p

theorem exists_tangent_line_through_L 
  (B Q P L : Point) (p : ℝ) :
  L ∉ InteriorAngle QBP →
  ∃ (A C : Point),
    triangle_perimeter p :=
by
  sorry

end exists_tangent_line_through_L_l37_37925


namespace no_real_roots_ff_eq_x_l37_37135

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_ff_eq_x (a b c : ℝ)
  (h : a ≠ 0)
  (discriminant_condition : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x := 
by 
  sorry

end no_real_roots_ff_eq_x_l37_37135


namespace number_of_sets_M_l37_37607

open Set

variable (a1 a2 a3 a4 : Type)

theorem number_of_sets_M :
  (∀ M : Set a1,
    M ⊆ ({a1, a2, a3, a4} : Set a1) →
    M ∩ ({a1, a2, a3} : Set a1) = ({a1, a2, a3} : Set a1) →
    M = ({a1, a2, a3} : Set a1) ∨ M = ({a1, a2, a3, a4} : Set a1)) →
  ∃ num : ℕ, (num = 2) :=
sorry

end number_of_sets_M_l37_37607


namespace confidence_relation_l37_37163

/-- If there is a 95% confidence that events A and B are related, then the specific calculated data satisfies K^2 > 3.841 -/
theorem confidence_relation (A B : Prop) (K_squared : ℝ) (h_confidence : 0.95 = 95%) : 
  (∃ (related : Prop), related ↔ K_squared > 3.841) :=
by 
  sorry

end confidence_relation_l37_37163


namespace polygon_sequence_area_l37_37000

theorem polygon_sequence_area (P : ℕ → Polygon) (area : Polygon → ℝ) 
  (h_initial : area (P 0) = 1)
  (h_transform : ∀ n, is_regular_hexagon (P n) → P (n+1) = join_midpoints_and_cut_corner (P n))
  :
  ∀ n, area (P n) > 1 / 3 :=
sorry

end polygon_sequence_area_l37_37000


namespace uneaten_pancakes_time_l37_37422

theorem uneaten_pancakes_time:
  ∀ (production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya : ℕ) (k : ℕ),
    production_rate_dad = 70 →
    production_rate_mom = 100 →
    consumption_rate_petya = 10 * 4 → -- 10 pancakes in 15 minutes -> (10/15) * 60 = 40 per hour
    consumption_rate_vasya = 2 * consumption_rate_petya →
    k * ((production_rate_dad + production_rate_mom) / 60 - (consumption_rate_petya + consumption_rate_vasya) / 60) ≥ 20 →
    k ≥ 24 := 
by
  intros production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya k
  sorry

end uneaten_pancakes_time_l37_37422


namespace arithmetic_expression_value_l37_37902

theorem arithmetic_expression_value :
  (19 + 43 / 151) * 151 = 2910 :=
by {
  sorry
}

end arithmetic_expression_value_l37_37902


namespace smallest_real_solution_l37_37252

theorem smallest_real_solution (x : ℝ) : 
  (x * |x| = 3 * x + 4) → x = 4 :=
by {
  sorry -- Proof omitted as per the instructions
}

end smallest_real_solution_l37_37252


namespace minimize_expression_l37_37605

theorem minimize_expression (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ)
  (hx_nonneg : ∀ i, 0 ≤ x i)
  (hx_sum : (Finset.univ : Finset (Fin n)).sum x = n) :
  (n-1) * ((Finset.univ : Finset (Fin n)).sum (λ i, (x i)^2)) + n * (Finset.univ : Finset (Fin n)).prod x ≥ n^2 :=
sorry

end minimize_expression_l37_37605


namespace arrange_students_l37_37250

def four_boys_and_two_girls : Prop :=
∃ (students : Finset String) (schools : Finset String), 
  students = {"boy1", "boy2", "boy3", "boyA", "girl1", "girl2"} ∧
  schools = {"A", "B", "C"} ∧
  ∀ s ∈ students, s ≠ "C" ∧
  ∀ (a b : String), a ≠ b → a ≠ "C" ∧ b ≠ "C" →
  (∃ (sch : String), sch = "A" ∧ sch ≠ a ∧ sch ≠ b) →
  (∃ (sch : String), sch = "B" ∧ sch ≠ a ∧ sch ≠ b) →
  ∀ student, student = "boyA" → student ≠ "A" →

  -- Complete the proof with the total number of different arrangements
  -- Providing an exact number count here
  (number_of_arrangements 4 2 ({"A", "B", "C"} \ {"A"} {"B"}) = 18)

theorem arrange_students :
  four_boys_and_two_girls :=
sorry

end arrange_students_l37_37250


namespace directrix_of_given_parabola_l37_37883

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37883


namespace m_greater_n_and_p_l37_37484

variable (a : ℝ)
variable (ha : a < -3)

def m : ℝ := (a+2) / (a+3)
def n : ℝ := (a+1) / (a+2)
def p : ℝ := a / (a+1)

theorem m_greater_n_and_p (h : a < -3) : m a > n a ∧ m a > p a :=
by
  sorry

end m_greater_n_and_p_l37_37484


namespace tom_tim_typing_ratio_l37_37736

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
by
  sorry

end tom_tim_typing_ratio_l37_37736


namespace probability_of_event_l37_37293

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def box_A : finset ℕ := finset.Icc 1 30
def box_B : finset ℕ := finset.Icc 15 44

def condition_A (n : ℕ) : Prop := n < 20 ∨ is_prime n
def condition_B (n : ℕ) : Prop := n % 2 = 1 ∨ n > 40

noncomputable def probability_A : ℚ := (finset.filter condition_A box_A).card / box_A.card
noncomputable def probability_B : ℚ := (finset.filter condition_B box_B).card / box_B.card

theorem probability_of_event :
  probability_A * probability_B = 323 / 900 :=
by sorry

end probability_of_event_l37_37293


namespace even_terms_in_expansion_of_m_plus_n_to_eight_l37_37970

variable (m n : ℤ)

def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem even_terms_in_expansion_of_m_plus_n_to_eight
  (hm : is_even m) (hn : is_odd n) :
  (count_even_terms : (m + n) ^ 8).nat = 8 :=
sorry

end even_terms_in_expansion_of_m_plus_n_to_eight_l37_37970


namespace cube_greater_l37_37472

theorem cube_greater (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end cube_greater_l37_37472


namespace cylinder_height_max_volume_l37_37491

noncomputable def height_of_cylinder_max_volume (r : ℝ) (h a : ℝ) : Prop :=
  (∀ h a, 0 < a ∧ a = sqrt(2) ∧ h = 2 * sqrt(1 - (a^2 / 3)) → h = 2 * sqrt(1 - (2 / 3)))

theorem cylinder_height_max_volume (r : ℝ) :
  height_of_cylinder_max_volume r (2 * sqrt(1 - (sqrt(2)^2 / 3))) sqrt(2) :=
by sorry

end cylinder_height_max_volume_l37_37491


namespace integer_values_between_fractions_l37_37514

theorem integer_values_between_fractions :
  let a := 4 / (Real.sqrt 3 + Real.sqrt 2)
  let b := 4 / (Real.sqrt 5 - Real.sqrt 3)
  ((⌊b⌋ - ⌈a⌉) + 1) = 6 :=
by sorry

end integer_values_between_fractions_l37_37514


namespace cos_of_angle_alpha_l37_37121

theorem cos_of_angle_alpha {α : ℝ} 
  (h1 : ∀ P : ℝ × ℝ, P = (-3 / 5, 4 / 5)) :
  cos α = -3 / 5 := 
sorry -- This is where the actual proof would go, which is omitted

end cos_of_angle_alpha_l37_37121


namespace internet_bill_is_100_l37_37230

theorem internet_bill_is_100 (initial_amount rent paycheck electricity_bill phone_bill final_amount internet_bill : ℝ)
  (h1 : initial_amount = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity_bill = 117)
  (h5 : phone_bill = 70)
  (h6 : final_amount = 1563)
  (h7 : initial_amount - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_amount) :
  internet_bill = 100 :=
by
  sorry

end internet_bill_is_100_l37_37230


namespace problem_statement_l37_37476

open Real

noncomputable def a : ℝ := log 11 / log 5
noncomputable def b : ℝ := (log 8 / log 2) / 2
noncomputable def c : ℝ := sqrt real.e

theorem problem_statement : a < b ∧ b < c :=
by {
  -- Definitions
  have ha : a = log 11 / log 5 := by rfl,
  have hb : b = (log 8 / log 2) / 2 := by rfl,
  have hc : c = sqrt real.e := by rfl,

  -- Proof
  sorry
}

end problem_statement_l37_37476


namespace number_of_pen_refills_l37_37295

-- Conditions
variable (k : ℕ) (x : ℕ) (hk : k > 0) (hx : (4 + k) * x = 6)

-- Question and conclusion as a theorem statement
theorem number_of_pen_refills (hk : k > 0) (hx : (4 + k) * x = 6) : 2 * x = 2 :=
sorry

end number_of_pen_refills_l37_37295


namespace ratio_of_james_to_jacob_l37_37224

noncomputable def MarkJumpHeight : ℕ := 6
noncomputable def LisaJumpHeight : ℕ := 2 * MarkJumpHeight
noncomputable def JacobJumpHeight : ℕ := 2 * LisaJumpHeight
noncomputable def JamesJumpHeight : ℕ := 16

theorem ratio_of_james_to_jacob : (JamesJumpHeight : ℚ) / (JacobJumpHeight : ℚ) = 2 / 3 :=
by
  sorry

end ratio_of_james_to_jacob_l37_37224


namespace john_must_work_10_more_days_l37_37205

theorem john_must_work_10_more_days
  (total_days : ℕ)
  (total_earnings : ℝ)
  (daily_earnings : ℝ)
  (target_earnings : ℝ)
  (additional_days : ℕ) :
  total_days = 10 →
  total_earnings = 250 →
  daily_earnings = total_earnings / total_days →
  target_earnings = 2 * total_earnings →
  additional_days = (target_earnings - total_earnings) / daily_earnings →
  additional_days = 10 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end john_must_work_10_more_days_l37_37205


namespace simplify_fraction_l37_37548

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)
variable (h3 : x - y^2 ≠ 0)

theorem simplify_fraction :
  (y^2 - 1/x) / (x - y^2) = (x * y^2 - 1) / (x^2 - x * y^2) :=
by
  sorry

end simplify_fraction_l37_37548


namespace y1_greater_than_y2_l37_37985

-- Define the function and points
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m

-- Define the points A and B on the parabola
def A_y1 (m : ℝ) : ℝ := parabola 0 m
def B_y2 (m : ℝ) : ℝ := parabola 1 m

-- Theorem statement
theorem y1_greater_than_y2 (m : ℝ) : A_y1 m > B_y2 m := 
  sorry

end y1_greater_than_y2_l37_37985


namespace parabola_directrix_is_x_eq_1_l37_37892

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37892


namespace king_henry_hogs_l37_37704

theorem king_henry_hogs (C H : ℕ) (h1 : H = 3 * C) (h2 : 0.60 * C - 5 = 10) : H = 75 := by
  sorry

end king_henry_hogs_l37_37704


namespace not_square_of_natural_l37_37817

theorem not_square_of_natural (N : ℕ) 
  (h1 : nat.digits 10 N = replicate 30 1 ++ replicate 30 0 ∨ nat.digits 10 N = replicate 30 0 ++ replicate 30 1)
  (h2 : 30 = (nat.digits 10 N).sum) : 
  ¬∃ n : ℕ, N = n ^ 2 := 
by
  sorry

end not_square_of_natural_l37_37817


namespace coeff_x3_in_expansion_l37_37264

-- Define the polynomial expressions
def poly1 (x : ℝ) := x^2 + 1
def poly2 (x : ℝ) := (2 * x + 1)^6

-- Define the full expression
def expression (x : ℝ) := poly1 x * poly2 x

-- The goal is to find the coefficient of x^3 in the expansion of (x^2 + 1)(2x + 1)^6
theorem coeff_x3_in_expansion : coeff (expression x) 3 = 184 := 
sorry

end coeff_x3_in_expansion_l37_37264


namespace number_of_ways_to_draw_4_from_15_l37_37749

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37749


namespace identical_solutions_k_value_l37_37071

theorem identical_solutions_k_value (k : ℝ) :
  (∀ (x y : ℝ), y = x^2 ∧ y = 4 * x + k → (x - 2)^2 = 0) → k = -4 :=
by
  sorry

end identical_solutions_k_value_l37_37071


namespace g_at_2_eq_9_l37_37038

def g (x : ℝ) : ℝ := x^2 + 3 * x - 1

theorem g_at_2_eq_9 : g 2 = 9 := by
  sorry

end g_at_2_eq_9_l37_37038


namespace rounding_to_thousand_representation_correct_l37_37322

theorem rounding_to_thousand_representation_correct:
  (approximation_precision_5000 : (∃ p : ℕ, 5000 = p)) →
  (approximation_precision_5_thousand : (∃ p : ℕ, 5 * 1000 = p)) →
  (precision_8_4 : ∃ t : ℕ, t / 10 = 8.4) →
  (precision_0_7 : ∃ t : ℕ, t / 10 = 0.7) →
  (precision_2_46_ten_thousand : ∃ p : ℕ, (2.46 * 10000) = p) →
  (∃ k : ℕ, 317500 / 1000 = k) →
  (∃ p : ℕ, (318000 / 10000) = 31.8 * 10000 / 10000) :=
by
  intros
  sorry

end rounding_to_thousand_representation_correct_l37_37322


namespace min_boxes_eliminated_l37_37566

theorem min_boxes_eliminated :
  ∃ (x : ℕ), 30 - x ≥ 0 ∧
  (7 : ℚ) / (30 - x : ℚ) ≥ 2 / 3 ∧
  x ≥ 20 := by
sorry

end min_boxes_eliminated_l37_37566


namespace draw_4_balls_in_order_ways_l37_37771

theorem draw_4_balls_in_order_ways : 
  ∀ (balls : Finset ℕ), 
  balls.card = 15 → 
  finset.permutations (balls).card 4 = 32760 := 
by 
  sorry

end draw_4_balls_in_order_ways_l37_37771


namespace exercise_data_l37_37347

open Real

theorem exercise_data (total_surveyed_students : ℕ) 
  (num_60_to_70 : ℕ) (perc_60_to_70 : ℝ)
  (num_70_to_80 : ℕ) (num_80_to_90 : ℕ) (perc_80_to_90 : ℝ)
  (perc_ge_90 : ℝ) (total_school_students : ℕ) :
  total_surveyed_students = 100 →
  num_60_to_70 = 14 → perc_60_to_70 = 0.14 →
  num_70_to_80 = 40 →
  num_80_to_90 = 35 → perc_80_to_90 = 0.35 →
  perc_ge_90 = 0.11 → 

  (let m := 40
  let n := 11

  m = 40 ∧ n = 11 ∧
  let estimated_students := total_school_students * (perc_80_to_90 + perc_ge_90)
  estimated_students = 460 ∧
  (80, [80, 81, 81, 81, 82, 82, 83, 83, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 86, 87, 87, 87, 87, 87, 88, 88, 88, 89, 89, 89, 89, 89]):
  p = 86) :=
by
  intros
  sorry

end exercise_data_l37_37347


namespace probability_of_forming_triangle_l37_37465

def segment_lengths : List ℕ := [1, 3, 5, 7, 9]
def valid_combinations : List (ℕ × ℕ × ℕ) := [(3, 5, 7), (3, 7, 9), (5, 7, 9)]
def total_combinations := Nat.choose 5 3

theorem probability_of_forming_triangle :
  (valid_combinations.length : ℚ) / total_combinations = 3 / 10 := 
by
  sorry

end probability_of_forming_triangle_l37_37465


namespace proof_clients_using_magazines_l37_37396

open Set

variables (T R M : Finset ℕ)
variables (nT nR nTR nTM nRM nTRM : ℕ)

noncomputable def number_of_clients_using_magazines (total_clients nT nR nTR nTM nRM nTRM : ℕ) :=
  total_clients = nT + nR + (M.card) - nTR - nTM - nRM + nTRM

theorem proof_clients_using_magazines :
  number_of_clients_using_magazines 180 115 110 75 85 95 80 = 130 :=
by {
  sorry
}

end proof_clients_using_magazines_l37_37396


namespace proof_part1_proof_part2_l37_37081

theorem proof_part1 (k : ℝ) (hk : k ≠ 0) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = k * x) (hf_eq : ∀ x, f (x + 1) * f x = x^2 + x) : ∀ x, f x = x ∨ f x = -x :=
begin
  sorry
end

theorem proof_part2 {f : ℝ → ℝ} (hf : ∀ x, f x = x) (h : ℝ → ℝ) 
  (hh : ∀ x, h x = (f x + 1) / (f x - 1)) (domain : ∀ x, f x ≠ 1) :
  (∃ m : ℝ, (∀ x, m ≤ x ∧ x ≤ m + 1 → m ≤ h x ∧ h x ≤ m + 1) ∧ (h m = m + 1) ∧ (h (m + 1) = m)) :=
begin
  sorry
end

end proof_part1_proof_part2_l37_37081


namespace cos_sine_identity_l37_37458

theorem cos_sine_identity (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500) :
  ∀ t : ℝ, (complex.ofReal (cos t) - complex.I * complex.ofReal (sin t))^n = complex.ofReal (cos (n * t)) - complex.I * complex.ofReal (sin (n * t)) :=
sorry

end cos_sine_identity_l37_37458


namespace perpendicular_lines_eqn_l37_37900

theorem perpendicular_lines_eqn
  (a : Fin (n+1) → ℝ)
  (n : ℕ) :
  ∀ x y : ℝ,
    a 0 * x ^ n + a 1 * x ^ (n - 1) * y + 
    a 2 * x ^ (n - 2) * y ^ 2 + ... + 
    a n * y ^ n = 0 ↔ 
    a 0 * y ^ n - a 1 * y ^ (n - 1) * x + 
    a 2 * y ^ (n - 2) * x ^ 2 - ... + 
    (-1)^n * a n * x ^ n = 0 :=
by sorry

end perpendicular_lines_eqn_l37_37900


namespace problem_solution_l37_37486

noncomputable def problem_statement : Prop :=
  let A B P : ℝ^3
  let AB := B - A
  let PA := A - P
  let PB := B - P
  (|AB| = 10) ∧ 
  (∀ t : ℝ, |PA - t • AB| ≥ 3) →
  (∃ P, ⟪PA, PB⟫ = -16 ∧ |PA + PB| = 6)

theorem problem_solution : problem_statement :=
by
  -- Here would go the proof
  sorry

end problem_solution_l37_37486


namespace ones_divisible_by_7_is_divisible_by_13_l37_37280

theorem ones_divisible_by_7_is_divisible_by_13 (n : ℕ)
  (hN : n = (((10 ^ 6) - 1) / 9) * (10 ^ (6 * t - 6) + 10 ^ (6 * t - 12) + ... + 10 ^ 6 + 1))
  (hN_div_7 : 7 ∣ n) : 13 ∣ n :=
sorry

end ones_divisible_by_7_is_divisible_by_13_l37_37280


namespace trapezoid_shaded_fraction_l37_37386

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end trapezoid_shaded_fraction_l37_37386


namespace trivia_team_total_score_l37_37390

theorem trivia_team_total_score 
  (scores : List ℕ)
  (present_members : List ℕ)
  (H_score : scores = [4, 6, 2, 8, 3, 5, 10, 3, 7])
  (H_present : present_members = scores) :
  List.sum present_members = 48 := 
by
  sorry

end trivia_team_total_score_l37_37390


namespace max_x2_y2_on_circle_l37_37084

noncomputable def max_value_on_circle : ℝ :=
  12 + 8 * Real.sqrt 2

theorem max_x2_y2_on_circle (x y : ℝ) (h : x^2 - 4 * x - 4 + y^2 = 0) : 
  x^2 + y^2 ≤ max_value_on_circle := 
by
  sorry

end max_x2_y2_on_circle_l37_37084


namespace part_I_part_II_l37_37501

-- Given a geometric sequence $a_n$ with $a_1 = 1$ and $a_1$, $a_2$, $a_3 - 1$ forming an arithmetic sequence.
def geometric_sequence (n : ℕ) : ℕ := 2 ^ (n - 1)

-- Given $b_n = 2n - 1 + a_n$
def b (n : ℕ) : ℕ := 2 * n - 1 + geometric_sequence n

-- Sum of the first n terms of the sequence $b_n$
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, b (i + 1)

theorem part_I (n : ℕ) (hn : n > 0) : geometric_sequence n = 2 ^ (n - 1) :=
  sorry

theorem part_II (n : ℕ) (hn : n > 0) : S n < n^2 + 2^n :=
  sorry

end part_I_part_II_l37_37501


namespace draw_4_balls_in_order_l37_37782

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37782


namespace non_overlapping_areas_difference_l37_37350

noncomputable def square (side : ℝ) : ℝ := side ^ 2
noncomputable def circle (radius : ℝ) : ℝ := π * radius ^ 2

theorem non_overlapping_areas_difference :
  let s := square 2
  let c := circle 3
  ∃ x : ℝ, (c - x) - (s - x) = 9 * π - 4 :=
by {
  intro s,
  intro c,
  use 0, -- using x = 0 for simplicity, proof should be detailed if actual overlap area needed
  simp [square, circle, s, c],
  sorry
}

end non_overlapping_areas_difference_l37_37350


namespace roots_form_parallelogram_l37_37866

theorem roots_form_parallelogram (b : ℝ) :
  (∀ z : ℂ, z ∈ (roots (λ z : ℂ, z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 4*b - 4)*z + 9)) →
    (complex.conjugate z + z = 4)) →
  b = 7/3 ∨ b = 2 :=
by
  sorry

end roots_form_parallelogram_l37_37866


namespace percent_errors_l37_37791

theorem percent_errors (S : ℝ) (hS : S > 0) (Sm : ℝ) (hSm : Sm = 1.25 * S) :
  let P := 4 * S
  let Pm := 4 * Sm
  let A := S^2
  let Am := Sm^2
  let D := S * Real.sqrt 2
  let Dm := Sm * Real.sqrt 2
  let E_P := ((Pm - P) / P) * 100
  let E_A := ((Am - A) / A) * 100
  let E_D := ((Dm - D) / D) * 100
  E_P = 25 ∧ E_A = 56.25 ∧ E_D = 25 :=
by
  sorry

end percent_errors_l37_37791


namespace geometric_sequence_log_sum_l37_37580

noncomputable def log_base_three (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h3 : a 6 * a 7 = 9) :
  log_base_three (a 1) + log_base_three (a 2) + log_base_three (a 3) +
  log_base_three (a 4) + log_base_three (a 5) + log_base_three (a 6) +
  log_base_three (a 7) + log_base_three (a 8) + log_base_three (a 9) +
  log_base_three (a 10) + log_base_three (a 11) + log_base_three (a 12) = 12 :=
  sorry

end geometric_sequence_log_sum_l37_37580


namespace point_Q_coordinates_l37_37574

-- Define the points O and P
def O := (0 : ℝ, 0 : ℝ)
def P := (4 : ℝ, 3 : ℝ)

-- Define the rotation function
def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x * Real.cos θ - y * Real.sin θ,
   x * Real.sin θ + y * Real.cos θ)

-- Define the angle θ to be -2π/3 for clockwise rotation
def θ := -(2 * Real.pi / 3)

-- Rotate point P around point O by -2π/3 to get point Q
def Q := rotate θ P

theorem point_Q_coordinates :
  Q = (-(4 + 3 * Real.sqrt 3) / 2, (-3 + 4 * Real.sqrt 3) / 2) :=
sorry

end point_Q_coordinates_l37_37574


namespace distance_from_point_to_focus_of_parabola_l37_37978

theorem distance_from_point_to_focus_of_parabola (x₀ y p : ℝ) (H : y^2 = 4 * x) (H₁ : y = 2) (H₂ : x = 1):
  ∃ d : ℝ, d = 2 ∧ (x₀, y) = (1, 2) →
    distance (x₀ - 1, y - 0) = 2 := 
by {
  intro H,
  intros H₁ H₂,
  have x₀ := 1,
  have y := 2,
  have focus := (1, 0),
  show distance (1, 2) (1, 0) = 2,
  sorry 
}

end distance_from_point_to_focus_of_parabola_l37_37978


namespace harper_water_duration_l37_37148

theorem harper_water_duration
  (half_bottle_per_day : ℝ)
  (bottles_per_case : ℕ)
  (cost_per_case : ℝ)
  (total_spending : ℝ)
  (cases_bought : ℕ)
  (days_per_case : ℕ)
  (total_days : ℕ) :
  half_bottle_per_day = 1/2 →
  bottles_per_case = 24 →
  cost_per_case = 12 →
  total_spending = 60 →
  cases_bought = total_spending / cost_per_case →
  days_per_case = bottles_per_case * 2 →
  total_days = days_per_case * cases_bought →
  total_days = 240 :=
by
  -- The proof will be added here
  sorry

end harper_water_duration_l37_37148


namespace distance_between_consecutive_trees_l37_37992

-- Define the conditions as separate definitions
def num_trees : ℕ := 57
def yard_length : ℝ := 720
def spaces_between_trees := num_trees - 1

-- Define the target statement to prove
theorem distance_between_consecutive_trees :
  yard_length / spaces_between_trees = 12.857142857 := sorry

end distance_between_consecutive_trees_l37_37992


namespace amanda_family_painting_theorem_l37_37394

theorem amanda_family_painting_theorem
  (rooms_with_4_walls : ℕ)
  (walls_per_room_with_4_walls : ℕ)
  (rooms_with_5_walls : ℕ)
  (walls_per_room_with_5_walls : ℕ)
  (walls_per_person : ℕ)
  (total_rooms : ℕ)
  (h1 : rooms_with_4_walls = 5)
  (h2 : walls_per_room_with_4_walls = 4)
  (h3 : rooms_with_5_walls = 4)
  (h4 : walls_per_room_with_5_walls = 5)
  (h5 : walls_per_person = 8)
  (h6 : total_rooms = 9)
  : rooms_with_4_walls * walls_per_room_with_4_walls +
    rooms_with_5_walls * walls_per_room_with_5_walls =
    5 * walls_per_person :=
by
  sorry

end amanda_family_painting_theorem_l37_37394


namespace modulus_of_complex_number_l37_37858

theorem modulus_of_complex_number : abs (3 - 4 * complex.i) = 5 := 
by
  sorry

end modulus_of_complex_number_l37_37858


namespace smallest_y_for_perfect_cube_l37_37366

-- Define the given conditions
def x : ℕ := 5 * 24 * 36

-- State the theorem to prove
theorem smallest_y_for_perfect_cube (y : ℕ) (h : y = 50) : 
  ∃ y, (x * y) % (y * y * y) = 0 :=
by
  sorry

end smallest_y_for_perfect_cube_l37_37366


namespace quadratic_equation_root_and_coef_l37_37552

theorem quadratic_equation_root_and_coef (k x : ℤ) (h1 : x^2 - 3 * x + k = 0)
  (root4 : x = 4) : (x = 4 ∧ k = -4 ∧ ∀ y, y ≠ 4 → y^2 - 3 * y + k = 0 → y = -1) :=
by {
  sorry
}

end quadratic_equation_root_and_coef_l37_37552


namespace directrix_of_given_parabola_l37_37886

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l37_37886


namespace parabola_directrix_is_x_eq_1_l37_37890

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l37_37890


namespace remaining_pencils_check_l37_37194

variables (Jeff_initial : ℕ) (Jeff_donation_percentage : ℚ) (Vicki_ratio : ℚ) (Vicki_donation_fraction : ℚ)

def Jeff_donated_pencils := (Jeff_donation_percentage * Jeff_initial).toNat
def Jeff_remaining_pencils := Jeff_initial - Jeff_donated_pencils

def Vicki_initial_pencils := (Vicki_ratio * Jeff_initial).toNat
def Vicki_donated_pencils := (Vicki_donation_fraction * Vicki_initial_pencils).toNat
def Vicki_remaining_pencils := Vicki_initial_pencils - Vicki_donated_pencils

def total_remaining_pencils := Jeff_remaining_pencils + Vicki_remaining_pencils

theorem remaining_pencils_check
    (Jeff_initial : ℕ := 300)
    (Jeff_donation_percentage : ℚ := 0.3)
    (Vicki_ratio : ℚ := 2)
    (Vicki_donation_fraction : ℚ := 0.75) :
    total_remaining_pencils Jeff_initial Jeff_donation_percentage Vicki_ratio Vicki_donation_fraction = 360 :=
by
  sorry

end remaining_pencils_check_l37_37194


namespace not_all_from_same_city_fourth_traveler_is_knight_l37_37407

inductive City
| knights
| liars

open City

variable (A B C D : City)
variable (A_statement B_statement C_statement : Prop)

/-- Traveler $A$ says: "Besides me, there is exactly one resident of my city here." --/
def A_statement : Prop := 
  (if A = knights then (B = knights ∨ C = knights ∨ D = knights) ∧ 
    (B ≠ knights ∧ C ≠ knights ∧ D ≠ knights).count_true = 1 else 
    (B = liars ∨ C = liars ∨ D = liars) ∧ 
    (B ≠ liars ∧ C ≠ liars ∧ D ≠ liars).count_true ≠ 1)

/-- Traveler $B$ says: "I am the only one from my city." --/
def B_statement : Prop := 
  (if B = knights then A ≠ knights ∧ C ≠ knights ∧ D ≠ knights else 
    A = liars ∧ C = liars ∧ D = liars)

/-- Traveler $C$ says: "You are right." (Confirming $B$'s statement) --/
def C_statement : Prop := 
  (if C = knights then B_statement else ¬B_statement)

/-- Not all travelers are from the same city --/
theorem not_all_from_same_city : A ≠ B ∨ A ≠ C ∨ A ≠ D ∨ B ≠ C ∨ B ≠ D ∨ C ≠ D := sorry

theorem fourth_traveler_is_knight : D = knights := sorry

end not_all_from_same_city_fourth_traveler_is_knight_l37_37407


namespace john_must_work_10_more_days_l37_37204

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end john_must_work_10_more_days_l37_37204


namespace draw_4_balls_in_order_l37_37780

theorem draw_4_balls_in_order :
  let choices : list ℕ := [15, 14, 13, 12] in
  (choices.foldr (λ x acc => x * acc) 1) = 32760 :=
by
  sorry

end draw_4_balls_in_order_l37_37780


namespace correct_calculation_result_l37_37006

theorem correct_calculation_result 
  (P : Polynomial ℝ := -x^2 + x - 1) :
  (P + -3 * x) = (-x^2 - 2 * x - 1) :=
by
  -- Since this is just the proof statement, sorry is used to skip the proof.
  sorry

end correct_calculation_result_l37_37006


namespace rectangle_area_l37_37734

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 40) : l * b = 75 := by
  sorry

end rectangle_area_l37_37734


namespace find_angle_bxy_l37_37577

-- Conditions Definitions
variables {A B C D X Y : Type}
variables (a b c d x y : ℝ)

-- Angles and Parallel Definition
noncomputable def axe := (x = 4 * (y) - 120)
noncomputable def parallel_ab_cd := (a // b // x // y ∧ c // d // x // y)

-- Statement of the theorem
theorem find_angle_bxy (h1 : parallel_ab_cd a b c d x y) (h2 : axe x y) : x = 40 :=
by
  sorry

end find_angle_bxy_l37_37577


namespace proper_subset_count_l37_37277

open Finset

noncomputable def setA : Finset ℕ := {1, 2, 3}

theorem proper_subset_count (A : Finset ℕ) (h : A = setA) : ((A.powerset.filter (λ s, s ≠ A)).card = 7) :=
by
  rw h
  sorry

end proper_subset_count_l37_37277


namespace trajectory_of_P_l37_37143

theorem trajectory_of_P (x y : ℝ) :
  let A := (-2, 0) in
  let B := (1, 0) in
  (∀ P : ℝ × ℝ, P = (x, y) → (Real.sqrt ((P.fst + 2)^2 + P.snd^2)) = 2 * (Real.sqrt ((P.fst - 1)^2 + P.snd^2))) →
  (x - 2)^2 + y^2 = 4 :=
by
  intros A B P hP
  sorry

end trajectory_of_P_l37_37143


namespace multiplication_result_l37_37261

theorem multiplication_result : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end multiplication_result_l37_37261


namespace shaded_region_area_is_correct_l37_37816

noncomputable def area_of_shaded_region (PQ S R Q P T U : Type) [MetricSpace PQ]
  (center_R : S) (radius_semicircle : ℝ) 
  (RS_perpendicular_PQ : PQ) (radius_arcs : ℝ) (center_P : P) (center_Q : Q) (center_S : S) : ℝ :=
  have h1 : radius_semicircle = 2 := by sorry,
  have h2 : radius_arcs = 3 := by sorry,
  have h3 : RS_perpendicular_PQ := by sorry,
  -- Compute sectors PTU, QSU and semicircle PQS areas
  let area_PTU := (1/2) * radius_arcs^2 * (π / 4),
  let area_QSU := (1/2) * radius_arcs^2 * (π / 4),
  let area_PQS := (1/2) * π * radius_semicircle^2,
  -- Combine the areas for total area
  let total_area := area_PTU + area_QSU + area_PQS,
  total_area

theorem shaded_region_area_is_correct {PQ S R Q P T U : Type} [MetricSpace PQ] 
  (center_R : S) (radius_semicircle : ℝ) 
  (RS_perpendicular_PQ : PQ) (radius_arcs : ℝ) (center_P : P) (center_Q : Q) (center_S : S) :
  area_of_shaded_region PQ S R Q P T U center_R radius_semicircle RS_perpendicular_PQ radius_arcs center_P center_Q center_S = (17 * π) / 4 :=
by
  sorry 

end shaded_region_area_is_correct_l37_37816


namespace find_N_l37_37164

theorem find_N (x N : ℝ) (h1 : x + 1 / x = N) (h2 : x^2 + 1 / x^2 = 2) : N = 2 :=
sorry

end find_N_l37_37164


namespace count_ordered_19_tuples_l37_37446

theorem count_ordered_19_tuples :
  (∃ (a : Fin 19 → ℤ), (∀ i : Fin 19, a i ^ 2 = (∑ j : Fin 19, a j) - a i) 
   ∧ 54264 = Fintype.card {a : Fin 19 → ℤ // ∀ i : Fin 19, a i ^ 2 = (∑ j : Fin 19, a j) - a i }) :=
sorry

end count_ordered_19_tuples_l37_37446


namespace interval_of_increase_l37_37013

noncomputable def f (x : ℝ) : ℝ :=
  -abs x

theorem interval_of_increase :
  ∀ x, f x ≤ f (x + 1) ↔ x ≤ 0 := by
  sorry

end interval_of_increase_l37_37013


namespace john_weekly_loss_is_correct_l37_37595

-- Definitions based on the conditions
def daily_production_capacity := 1000
def production_cost_per_tire := 250
def selling_price_multiplier := 1.5
def potential_daily_sales := 1200

-- Function to calculate the additional revenue lost per week
def weekly_revenue_lost : ℕ :=
  let additional_tires := potential_daily_sales - daily_production_capacity
  let daily_revenue_lost := additional_tires * (production_cost_per_tire * selling_price_multiplier)
  daily_revenue_lost * 7

-- The theorem we need to prove
theorem john_weekly_loss_is_correct : weekly_revenue_lost = 525000 := by
  sorry

end john_weekly_loss_is_correct_l37_37595


namespace degree_monomial_equal_four_l37_37662

def degree_of_monomial (a b : ℝ) := 
  (3 + 1)

theorem degree_monomial_equal_four (a b : ℝ) 
  (h : a^3 * b = (2/3) * a^3 * b) : 
  degree_of_monomial a b = 4 :=
by sorry

end degree_monomial_equal_four_l37_37662


namespace max_cards_possible_l37_37806

-- Define the dimensions for the cardboard and the card.
def cardboard_length : ℕ := 48
def cardboard_width : ℕ := 36
def card_length : ℕ := 16
def card_width : ℕ := 12

-- State the theorem to prove the maximum number of cards.
theorem max_cards_possible : (cardboard_length / card_length) * (cardboard_width / card_width) = 9 :=
by
  sorry -- Skip the proof, as only the statement is required.

end max_cards_possible_l37_37806


namespace determine_a_b_l37_37033

-- Define the polynomial function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + 3 * x^3 - 5 * x^2 + b * x - 7

-- Problem statement: prove the values of a and b
theorem determine_a_b :
  ∃ (a b : ℝ), f a b 2 = 9 ∧ f a b (-1) = -4 ∧ a = 7/9 ∧ b = -2/9 :=
by 
  -- Our variables a and b
  let a := 7/9
  let b := -2/9 
  -- Check the conditions
  use a, b
  split; sorry
  split; sorry
  split; sorry
  split; sorry

#check determine_a_b

end determine_a_b_l37_37033


namespace r_and_d_investment_expression_and_year_exceeding_two_million_l37_37352

theorem r_and_d_investment_expression_and_year_exceeding_two_million (x : ℕ) :
  ∃ n : ℕ, y = 1.3 * (1 + 0.12)^x ∧ (1.3 * (1 + 0.12) ^ n > 2 ∧ 2015 + n = 2019) :=
by
  sorry

end r_and_d_investment_expression_and_year_exceeding_two_million_l37_37352


namespace problem_1_problem_2_l37_37498

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def ellipse_condition (a b : ℝ) : Prop :=
  a = sqrt 2 ∧ b = 1

theorem problem_1 
  (a b : ℝ) 
  (h : a > b ∧ b > 0) 
  (perimeter_condition : a * 4 = 4 * sqrt 2 ∧ b = sqrt (a^2 - b^2)) :
  ellipse_equation a b h :=
begin
  use [a, b],
  split,
  { exact sqrt 2 },
  { exact 1 },
  sorry
end

noncomputable def line_intersection_condition (m : ℝ) (h : m > 0) : Prop :=
  ∀ (k : ℝ), m^2 < (k^2 + 1) / (k^2 + 3)

theorem problem_2 (m : ℝ) (h : m > 0) :
  D_inside_circle_diameter_EF_condition m h :=
begin
  split,
  { linarith },
  { exact sqrt (3) / 3 },
  sorry
end

end problem_1_problem_2_l37_37498


namespace total_weight_moved_l37_37597

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end total_weight_moved_l37_37597


namespace number_of_solutions_l37_37546

-- Define the relevant trigonometric equation
def trig_equation (x : ℝ) : Prop := (Real.cos x)^2 + 3 * (Real.sin x)^2 = 1

-- Define the range for x
def in_range (x : ℝ) : Prop := -20 < x ∧ x < 100

-- Define the predicate that x satisfies both the trig equation and the range condition
def satisfies_conditions (x : ℝ) : Prop := trig_equation x ∧ in_range x

-- The final theorem statement (proof is omitted)
theorem number_of_solutions : 
  ∃ (count : ℕ), count = 38 ∧ ∀ (x : ℝ), satisfies_conditions x ↔ x = k * Real.pi ∧ -20 < k * Real.pi ∧ k * Real.pi < 100 := sorry

end number_of_solutions_l37_37546


namespace angle_NHC_l37_37495

-- Definitions
variables {Point : Type} [metric_space Point] [inner_product_space ℝ Point]

def square (A B C D : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  dist A C = dist B D ∧ ∠ A B C = 90 ∧ ∠ B C D = 90 ∧ ∠ C D A = 90 ∧ ∠ D A B = 90

def equilateral_triangle (B C S : Point) : Prop := 
  dist B C = dist B S ∧ dist B S = dist S C ∧ ∠ B C S = 60 ∧ ∠ C S B = 60 ∧ ∠ S B C = 60

def midpoint (M A B : Point) : Prop := 
  dist A M = dist M B

-- Main Problem
theorem angle_NHC (A B C D S N H : Point)
  (sq : square A B C D)
  (eq_tri : equilateral_triangle B C S)
  (mid_AS : midpoint N A S)
  (mid_CD : midpoint H C D) :
  ∠ N H C = 60 :=
sorry

end angle_NHC_l37_37495


namespace tangent_line_at_slope_two_l37_37688

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l37_37688


namespace ball_drawing_ways_l37_37765

theorem ball_drawing_ways :
  let n := 15
  let k := 4
  ∑(n-k+1 to n) = 32760 :=
by
  sorry

end ball_drawing_ways_l37_37765


namespace number_of_ordered_pairs_l37_37060

theorem number_of_ordered_pairs :
  {na : ℕ // na = 6} = 
  {na : ℕ // na = (Finset.filter (λ p : ℤ × ℤ, (4 - 4 * p.1 * p.2) ≥ 0) 
  {p | p.1 ∈ {-1, 1, 2} ∧ p.2 ∈ {-1, 1, 2}}).card} :=
by
  sorry

end number_of_ordered_pairs_l37_37060


namespace number_of_ways_to_draw_4_from_15_l37_37751

theorem number_of_ways_to_draw_4_from_15 : 
  let n := 4
  let k := 15
  (∏ i in finset.range n, (k - i)) = 32760 := 
by
  sorry

end number_of_ways_to_draw_4_from_15_l37_37751


namespace maximum_value_l37_37479

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value (x₀ : ℝ) (h₁ : f(x₀) = x₀) (h₂ : ∀ x : ℝ, f(x) ≤ f(x₀)) :
  x₀ > 1 ∧ f(x₀) > (1 : ℝ) / 9 :=
sorry

end maximum_value_l37_37479


namespace range_of_ab_l37_37213

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |2 - a^2| = |2 - b^2|) : 0 < a * b ∧ a * b < 2 := by
  sorry

end range_of_ab_l37_37213


namespace pie_eating_fraction_l37_37328

theorem pie_eating_fraction :
  (1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4 + 1 / 3^5 + 1 / 3^6 + 1 / 3^7) = 1093 / 2187 := 
sorry

end pie_eating_fraction_l37_37328


namespace OK_eq_OL_l37_37664

open Locale Classical

variables {A B C D O K L : Type} [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry D]
variable [O : Barycentric O A B]

-- Conditions of the problem
def is_cyclic_quad (A B C D : Type) [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry D] := 
  ∃ circ : Circle, circ.contains A ∧ circ.contains B ∧ circ.contains C ∧ circ.contains D

def diagonals_intersect (A C B D O : Type) [PlaneGeometry A] [PlaneGeometry C] [PlaneGeometry B] [PlaneGeometry D] [Barycentric O A C] [Barycentric O B D] : Prop := 
  ∃ O, Line A C.lines ∩ Line B D.lines = {O}

def intersection_with_circumcircle (A B O D AD_lines BC_lines K L : Type) [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry O] [PlaneGeometry D] [Line AD_lines] [Line BC_lines] [Point K] [Point L] : Prop :=
  ∃ K L, (Line AD_lines).intersects (circumcircle A O B) = {K} ∧ (Line BC_lines).intersects (circumcircle A O B) = {L}

def given_angle (A B C D : Type) [PlaneGeometry A] [PlaneGeometry B] [PlaneGeometry C] [PlaneGeometry D] :=
  ∠ B C A = ∠ B D C

-- The goal of the proof
theorem OK_eq_OL
  (h1 : is_cyclic_quad A B C D)
  (h2 : diagonals_intersect A C B D O)
  (h3 : intersection_with_circumcircle A B O D AD BC K L)
  (h4 : given_angle A B C D) :
  OK = OL := sorry

end OK_eq_OL_l37_37664


namespace monotonicity_and_range_of_a_l37_37134

noncomputable def f (a x : ℝ) := Real.exp(2 * x) + (2 - a) * Real.exp(x) - a * x + (a * Real.exp(1)) / 2

theorem monotonicity_and_range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 2 * Real.exp 1) :=
sorry

end monotonicity_and_range_of_a_l37_37134


namespace even_terms_in_expansion_of_m_plus_n_to_eight_l37_37971

variable (m n : ℤ)

def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem even_terms_in_expansion_of_m_plus_n_to_eight
  (hm : is_even m) (hn : is_odd n) :
  (count_even_terms : (m + n) ^ 8).nat = 8 :=
sorry

end even_terms_in_expansion_of_m_plus_n_to_eight_l37_37971


namespace find_x_l37_37956

variables {a b : EuclideanSpace ℝ (Fin 2)} {x : ℝ}

theorem find_x (h1 : ‖a + b‖ = 1) (h2 : ‖a - b‖ = x) (h3 : inner a b = -(3 / 8) * x) : x = 2 ∨ x = -(1 / 2) :=
sorry

end find_x_l37_37956


namespace Hamiltonian_cycle_exists_iff_odd_least_turning_points_l37_37989

theorem Hamiltonian_cycle_exists_iff_odd (m n : ℕ) :
  ((∃ (route : list (ℕ × ℕ)), (route.head = some (0, 0)) ∧ (route.ilast = some (0, 0)) ∧
                             (route.nodup) ∧ (route.length = m * n + 1) ∧ 
                             (∀ i ∈ route, (0 ≤ (fst i) < m) ∧ (0 ≤ (snd i) < n))) ↔ (odd m ∨ odd n)) :=
sorry

theorem least_turning_points (m n : ℕ) (exists_route : (∃ (route : list (ℕ × ℕ)), (route.head = some (0, 0)) ∧ (route.ilast = some (0, 0)) ∧
                                                           (route.nodup) ∧ (route.length = m * n + 1) ∧ 
                                                           (∀ i ∈ route, (0 ≤ (fst i) < m) ∧ (0 ≤ (snd i) < n)))) :
  (∃ min_turning_points : ℕ, (if even (min m n) then min_turning_points = 2 * (min m n) + 1 else min_turning_points = 2 * (max m n) + 1)) :=
sorry

end Hamiltonian_cycle_exists_iff_odd_least_turning_points_l37_37989


namespace perp_bisector_bisects_AD_l37_37603

variables {A B C M B' C' D P : Point}
variables {BC : Segment}
variables (ABC : Triangle A B C)
variables (M_mid : Midpoint M BC)
variables (circ : Circle M (dist M A))
variables (B'_on_AB : B' ∈ LineThrough A B)
variables (C'_on_AC : C' ∈ LineThrough A C)
variables (B'_on_circ : OnCircle B' circ)
variables (C'_on_circ : OnCircle C' circ)
variables (tangent_B' : Tangent B' D circ)
variables (tangent_C' : Tangent C' D circ)

def perpendicular_bisector (BC : Segment) : Line := sorry -- Assume an implementation for the perpendicular bisector

noncomputable def bisects (line : Line) (segment : Segment 1) (segment 2) : Prop := sorry --Assume an implementation that checks if the line bisects segment2

theorem perp_bisector_bisects_AD : 
  bisects (perpendicular_bisector BC) (Segment.mk A D) (Segment.mk D A) := 
sorry

end perp_bisector_bisects_AD_l37_37603


namespace common_chord_and_length_l37_37117

-- Define the two circles
def circle1 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y - 5 = 0
def circle2 (x y : ℝ) := x^2 + y^2 + 2*x - 1 = 0

-- The theorem statement with the conditions and expected solutions
theorem common_chord_and_length :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → y = -1)
  ∧
  (∃ A B : (ℝ × ℝ), (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
                    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
                    (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4)) :=
by
  sorry

end common_chord_and_length_l37_37117


namespace john_total_spend_l37_37591

-- Define the conditions
def amount_silver := 1.5
def amount_gold := 2 * amount_silver
def cost_per_ounce_silver := 20
def cost_per_ounce_gold := 50 * cost_per_ounce_silver

-- Define the cost calculations
def cost_silver := amount_silver * cost_per_ounce_silver
def cost_gold := amount_gold * cost_per_ounce_gold

-- Define the theorem
theorem john_total_spend : cost_silver + cost_gold = 3030 := 
by
  sorry

end john_total_spend_l37_37591


namespace range_of_t_l37_37104

-- Given Definitions
def Sn (a : ℕ → ℚ) (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), a i

axiom a_1_eq_1 : a 1 = 1

axiom Sn_formula (a : ℕ → ℚ) (n : ℕ) : 2 * Sn a n = (n + 1) * a n

axiom inequality_has_integer_solutions (a : ℕ → ℚ) (t : ℚ) : 
  (∃ n ∈ finset.range 1000, n > 0 ∧ n ≠ 1 ∧ a n^2 - t * a n ≤ 2 * t^2)

-- The problem is to prove that the range of t is exactly [1, 3/2)
theorem range_of_t (t : ℚ) : 
  (1 ≤ t ∧ t < 3/2) ↔ inequality_has_integer_solutions (λ n, n) t :=
sorry

end range_of_t_l37_37104


namespace euler_formula_for_triangle_l37_37211

noncomputable def power_of_I_with_respect_to_circumscribed_circle (ABC : Triangle) (O I : Point) (R r : ℝ) 
  (H1 : O = circumcenter ABC)
  (H2 : I = incenter ABC)
  (H3 : circumscribed_radius ABC = R)
  (H4 : inscribed_radius ABC = r) : Prop :=
  pow (distance O I) 2 = pow R 2 - 2 * R * r

theorem euler_formula_for_triangle (ABC : Triangle) (O I : Point) (R r : ℝ) 
  (H1 : O = circumcenter ABC)
  (H2 : I = incenter ABC)
  (H3 : circumscribed_radius ABC = R)
  (H4 : inscribed_radius ABC = r) 
  : power_of_I_with_respect_to_circumscribed_circle ABC O I R r H1 H2 H3 H4:=
  sorry

end euler_formula_for_triangle_l37_37211


namespace max_marked_segments_no_complete_triangle_l37_37016

theorem max_marked_segments_no_complete_triangle (n : ℕ) : 
  let total_segments := n * (n + 1)
  (∀ (i j k : ℕ), i < j → j < k → ¬(marked i j ∧ marked j k ∧ marked k i)) → 
  total_segments = n * (n + 1) :=
sorry

end max_marked_segments_no_complete_triangle_l37_37016


namespace polygon_D_has_largest_area_l37_37327

noncomputable def area_A := 4 * 1 + 2 * (1 / 2) -- 5
noncomputable def area_B := 2 * 1 + 2 * (1 / 2) + Real.pi / 4 -- ≈ 3.785
noncomputable def area_C := 3 * 1 + 3 * (1 / 2) -- 4.5
noncomputable def area_D := 3 * 1 + 1 * (1 / 2) + 2 * (Real.pi / 4) -- ≈ 5.07
noncomputable def area_E := 1 * 1 + 3 * (1 / 2) + 3 * (Real.pi / 4) -- ≈ 4.855

theorem polygon_D_has_largest_area :
  area_D > area_A ∧
  area_D > area_B ∧
  area_D > area_C ∧
  area_D > area_E :=
by
  sorry

end polygon_D_has_largest_area_l37_37327


namespace four_consecutive_vertices_in_same_heptagon_l37_37354

theorem four_consecutive_vertices_in_same_heptagon
    (N : ℕ)
    (polygon : Type)
    (heptagon_division : polygon → set polygon)
    (is_convex : ∀ p : polygon, convex polygon)
    (is_heptagon : ∀ p : polygon, size p = 7)
    (all_sides_in_heptagon : ∀ s : side (polygon), ∃ h : polygon, s ∈ sides h)
    (N_polygon : specific_polygon N): 
    ∃ h : polygon, ∃ (v1 v2 v3 v4 : vertex polygon), 
        consecutive_vertices N_polygon v1 v2 v3 v4 ∧ 
        v1 ∈ vertices h ∧ v2 ∈ vertices h ∧ v3 ∈ vertices h ∧ v4 ∈ vertices h :=
by
  sorry

end four_consecutive_vertices_in_same_heptagon_l37_37354


namespace machine_value_after_2_years_l37_37801

section
def initial_value : ℝ := 1200
def depreciation_rate_year1 : ℝ := 0.10
def depreciation_rate_year2 : ℝ := 0.12
def repair_rate : ℝ := 0.03
def major_overhaul_rate : ℝ := 0.15

theorem machine_value_after_2_years :
  let value_after_repairs_2 := (initial_value * (1 - depreciation_rate_year1) + initial_value * repair_rate) * (1 - depreciation_rate_year2 + repair_rate)
  (value_after_repairs_2 * (1 - major_overhaul_rate)) = 863.23 := 
by
  -- proof here
  sorry
end

end machine_value_after_2_years_l37_37801


namespace intersection_eq_set_0_lt_x_lt_1_l37_37487

def f (x : ℝ) : ℝ := x^2 - 2*x
def A : Set ℝ := {x | f x < 0}
def B : Set ℝ := {x | f'' x < 0} -- Note: Usually f'' would be defined by differentiating f twice manually in Lean. 

theorem intersection_eq_set_0_lt_x_lt_1 : (A ∩ B) = {x : ℝ | 0 < x ∧ x < 1} :=
by
  unfold A B f -- Here, additional steps would typically be to compute f'' in Lean.
  sorry

end intersection_eq_set_0_lt_x_lt_1_l37_37487


namespace integral_evaluation_l37_37859

theorem integral_evaluation :
  (∫ x in 1..e, 1 / x) + (∫ x in -2..2, Real.sqrt (4 - x ^ 2)) = 2 * Real.pi + 1 := 
by
  have I1 : (∫ x in 1..e, 1 / x) = 1 := sorry
  have I2 : (∫ x in -2..2, Real.sqrt (4 - x ^ 2)) = 2 * Real.pi := sorry
  rw [I1, I2]
  norm_num

end integral_evaluation_l37_37859


namespace tan_G_in_right_triangle_l37_37431

theorem tan_G_in_right_triangle 
  (GH FG : ℝ) 
  (GH_is_20 : GH = 20) 
  (FG_is_25 : FG = 25) 
  (is_right_triangle : GH^2 + HF^2 = FG^2) 
  (HF : ℝ) 
  (HF_is_15 : HF = 15) :
  tan G = 3 / 4 := 
sorry

end tan_G_in_right_triangle_l37_37431


namespace secretary_participation_l37_37721

-- Definitions of the parameters
def small_emails (M : ℕ) : ℕ := 42 * M
def large_emails (B : ℕ) : ℕ := 210 * B
def secretary_emails_small (m : ℕ) : ℕ := 6 * m
def secretary_emails_large (b : ℕ) : ℕ := 14 * b

-- The condition of the total number of emails
def total_emails (M B m b : ℕ) : Prop :=
  small_emails M + large_emails B = 1994 + secretary_emails_small m + secretary_emails_large b ∧ m + b ≤ 10

-- The Lean statement to be proven
theorem secretary_participation :
  ∃ (m b : ℕ), total_emails M B m b :=
by { existsi (6 : ℕ), existsi (2 : ℕ), sorry }

end secretary_participation_l37_37721


namespace smallest_sum_of_digits_l37_37520

def is_valid_digit_set (s : List ℕ) : Prop :=
  s = [3, 7, 2, 9, 5]

def use_each_digit_exactly_once (a b : ℕ) (digits : List ℕ) : Prop :=
  let digits_a := a.digits 10
  let digits_b := b.digits 10
  digits_a ++ digits_b = digits

theorem smallest_sum_of_digits (a b : ℕ) (digits : List ℕ)
  (h_digits : is_valid_digit_set digits)
  (h_use_exactly_once : use_each_digit_exactly_once a b digits)
  (h_four_digit_a : 1000 ≤ a ∧ a < 10000)
  (h_three_digit_b : 100 ≤ b ∧ b < 1000)
  : a + b = 3257 :=
begin
  sorry
end

end smallest_sum_of_digits_l37_37520


namespace abs_diff_x_y_is_392_l37_37659

noncomputable def arithmetic_mean (x y : ℕ) := (x + y) / 2
noncomputable def geometric_mean (x y : ℕ) := Real.sqrt (x * y)

theorem abs_diff_x_y_is_392 
  (x y : ℕ)
  (h1 : arithmetic_mean x y = 450)
  (h2 : geometric_mean x y = 405)
  (h3 : x + y = 900) :
  |x - y| = 392 :=
by
  sorry

end abs_diff_x_y_is_392_l37_37659


namespace project_total_time_l37_37260

def total_hours (x : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : ratio1 = 3) (h2 : ratio2 = 5) (h3 : ratio3 = 6) (h4 : ratio3 * x = ratio1 * x + 30) : ℕ := 
  ratio1 * x + ratio2 * x + ratio3 * x

theorem project_total_time : ∃ x, x = 30 ∧ total_hours x 3 5 6 rfl rfl rfl (by linarith) = 140 := 
by
  use 30
  split
  · rfl
  · sorry

end project_total_time_l37_37260


namespace symmetry_implies_a_eq_1_l37_37480

-- Define the function f(x) and the symmetry condition
def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

-- Given condition: The graph of f(x) is symmetric about the line x = 1
def is_symmetry_about_x1 (f : ℝ → ℝ) : Prop :=
∀ (x : ℝ), f (1 + x) = f (1 - x)

-- Conclusion: The value of the real number a
theorem symmetry_implies_a_eq_1 (a : ℝ) (H : is_symmetry_about_x1 (f a)) : a = 1 :=
sorry

end symmetry_implies_a_eq_1_l37_37480


namespace evaluate_expression_l37_37048

noncomputable def x : ℚ := 4 / 8
noncomputable def y : ℚ := 5 / 6

theorem evaluate_expression : (8 * x + 6 * y) / (72 * x * y) = 3 / 10 :=
by
  sorry

end evaluate_expression_l37_37048


namespace magnitude_of_sum_l37_37537

noncomputable def vector_magnitude (a b : ℝ³) (θ : ℝ) : ℝ :=
  real.sqrt ((2^2 + (2 * 1 * real.cos θ * 2) + (2 * 1)^2))

theorem magnitude_of_sum:
  let a b : ℝ³ 
  let |a| := 2
  let |b| := 1
  let θ := 60 * π / 180
  in vector_magnitude a b θ = 2 * real.sqrt 3 :=
by sorry

end magnitude_of_sum_l37_37537


namespace real_is_special_case_of_complex_l37_37586

-- Definitions based on conditions
structure Complex where
  real_part : ℝ
  imaginary_part : ℝ

-- Definition of a real number in the context of complex numbers
def is_real_as_complex (x : ℝ) : Complex := 
  { real_part := x, imaginary_part := 0 }

-- The theorem statement
theorem real_is_special_case_of_complex (x : ℝ) : ∃ z : Complex, z.real_part = x ∧ z.imaginary_part = 0 :=
  ⟨is_real_as_complex x, by simp [is_real_as_complex]⟩

end real_is_special_case_of_complex_l37_37586


namespace area_of_triangle_DEF_l37_37370

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 8)
  let base1 := 6
  let height1 := 2
  let base2 := 3
  let height2 := 8
  let base3 := 3
  let height3 := 6
  let area_triangle_DE := 1 / 2 * (base1 * height1)
  let area_triangle_EF := 1 / 2 * (base2 * height2)
  let area_triangle_FD := 1 / 2 * (base3 * height3)
  let area_rectangle := 6 * 8
  ∃ area_def_triangle, 
  area_def_triangle = area_rectangle - (area_triangle_DE + area_triangle_EF + area_triangle_FD) 
  ∧ area_def_triangle = 21 :=
by 
  sorry

end area_of_triangle_DEF_l37_37370
