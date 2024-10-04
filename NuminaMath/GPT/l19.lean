import Mathlib

namespace marco_paints_8_15_in_32_minutes_l19_19624

-- Define the rates at which Marco and Carla paint
def marco_rate : ℚ := 1 / 60
def combined_rate : ℚ := 1 / 40

-- Define the function to calculate the fraction of the room painted by Marco alone in a given time
def fraction_painted_by_marco (time: ℚ) : ℚ := time * marco_rate

-- State the theorem to prove
theorem marco_paints_8_15_in_32_minutes :
  (marco_rate + (combined_rate - marco_rate) = combined_rate) →
  fraction_painted_by_marco 32 = 8 / 15 := by
  sorry

end marco_paints_8_15_in_32_minutes_l19_19624


namespace total_money_proof_l19_19859

noncomputable def totalMoney (r_d r_e total_d_e : ℕ) : ℕ :=
  let ratio_units := 15
  let value_of_one_part := total_d_e / (r_d + r_e)
  ratio_units * value_of_one_part

theorem total_money_proof (r_a r_b r_c r_d r_e : ℕ) (total_d_e : ℕ) :
  r_a = 2 → r_b = 4 → r_c = 3 → r_d = 1 → r_e = 5 → total_d_e = 4800 →
  totalMoney r_d r_e total_d_e = 12000 :=
by
  intros h_a h_b h_c h_d h_e h_de
  rw [h_a, h_b, h_c, h_d, h_e, h_de]
  sorry

end total_money_proof_l19_19859


namespace lowest_numbered_true_statement_is_204_l19_19811

-- The conditions given in the problem
axiom S206_is_true : 1 + 1 = 2
axiom S201_def : Prop -- Statement 201: "Statement 203 is true".
axiom S202_def : Prop -- Statement 202: "Statement 201 is true".
axiom S203_def : ¬S206_is_true -- Statement 203: "Statement 206 is false".
axiom S204_def : ¬S202_def -- Statement 204: "Statement 202 is false".
axiom S205_def : ¬(S201_def ∨ S202_def ∨ S203_def ∨ S204_def) -- Statement 205: "None of the statements 201, 202, 203 or 204 are true".

-- Prove that the lowest numbered true statement is 204
theorem lowest_numbered_true_statement_is_204 : S204_def :=
by
  sorry

end lowest_numbered_true_statement_is_204_l19_19811


namespace more_girls_than_boys_l19_19244

theorem more_girls_than_boys (girls boys total_pupils : ℕ) (h1 : girls = 692) (h2 : total_pupils = 926) (h3 : boys = total_pupils - girls) : girls - boys = 458 :=
by
  sorry

end more_girls_than_boys_l19_19244


namespace decimal_to_fraction_l19_19786

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l19_19786


namespace average_weight_of_children_l19_19749

theorem average_weight_of_children (avg_weight_boys avg_weight_girls : ℕ)
                                   (num_boys num_girls : ℕ)
                                   (h1 : avg_weight_boys = 160)
                                   (h2 : avg_weight_girls = 110)
                                   (h3 : num_boys = 8)
                                   (h4 : num_girls = 5) :
                                   (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 141 :=
by
    sorry

end average_weight_of_children_l19_19749


namespace analogous_proposition_in_solid_geometry_true_l19_19560

axiom parallel_line_segments_equal (l₁ l₂ : 𝔼) (p₁ p₂ : ℝ) : 
  are_parallel l₁ l₂ ∧ between l₁ l₂ p₁ p₂ → segment_equal p₁ p₂

noncomputable def parallel_plane_segments_equal (l₃ l₄ : Plain) (q₁ q₂ : ℝ) : 
  Prop := 
  are_parallel l₃ l₄ ∧ between l₃ l₄ q₁ q₂ → segment_equal q₁ q₂

theorem analogous_proposition_in_solid_geometry_true (l₃ l₄ : Plain) (q₁ q₂ : ℝ) :
  are_parallel l₃ l₄ ∧ between l₃ l₄ q₁ q₂ → segment_equal q₁ q₂ :=
sorry

end analogous_proposition_in_solid_geometry_true_l19_19560


namespace triangle_XYZ_XY2_XZ2_difference_l19_19675

-- Define the problem parameters and conditions
def YZ : ℝ := 10
def XM : ℝ := 6
def midpoint_YZ (M : ℝ) := 2 * M = YZ

-- The main theorem to be proved
theorem triangle_XYZ_XY2_XZ2_difference :
  ∀ (XY XZ : ℝ), 
  (∀ (M : ℝ), midpoint_YZ M) →
  ((∃ (x : ℝ), (0 ≤ x ∧ x ≤ 10) ∧ XY^2 + XZ^2 = 2 * x^2 - 20 * x + 2 * (11 * x - x^2 - 11) + 100)) →
  (120 - 100 = 20) :=
by
  sorry

end triangle_XYZ_XY2_XZ2_difference_l19_19675


namespace cone_volume_divided_by_pi_l19_19443

noncomputable def volume_of_cone_divided_by_pi (r : ℝ) (angle : ℝ) : ℝ :=
  if angle = 270 ∧ r = 20 then
    let base_circumference := 30 * Real.pi in
    let base_radius := 15 in
    let slant_height := r in
    let height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
    let volume := (1 / 3) * Real.pi * base_radius ^ 2 * height in
    volume / Real.pi
  else 0

theorem cone_volume_divided_by_pi : 
  volume_of_cone_divided_by_pi 20 270 = 375 * Real.sqrt 7 :=
by
  sorry

end cone_volume_divided_by_pi_l19_19443


namespace ellipse_equation_line_equation_through_F2_l19_19610

theorem ellipse_equation (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) : 
  F1 = (-1, 0) ∧ F2 = (1, 0) ∧
  let f : ℝ × ℝ → ℝ := λ q, (q.1 + 1)^2 + q.2^2 in
  let e : ℝ × ℝ → ℝ := λ q, (q.1 - 1)^2 + q.2^2 in
  (f P + e P = 4*(F2.1 - F1.1)^2) →
  (exists a b : ℝ, a > b ∧ b > 0 ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)) →
  (a = 2) ∧ (b^2 = 3) :=
sorry

theorem line_equation_through_F2 (F1 F2 : ℝ × ℝ) (P Q : ℝ × ℝ) :
  F1 = (-1, 0) ∧ F2 = (1, 0) ∧
  (let dist (q r: ℝ × ℝ) := (q.1 - r.1)^2 + (q.2-r.2)^2 in
    let k := (Q.2 - P.2)/(Q.1 - P.1) in
    dist P Q^2 = dist P F1^2 + dist Q F1^2 ∧ 
    F2.1 ∈ [P.1, Q.1] ∧
    (exists k : ℝ, k = ± (3 * Real.sqrt 7 / 7)) →
    (Q.2 = ± (3 * Real.sqrt 7 /7)*(Q.1 - 1))) :=
sorry

end ellipse_equation_line_equation_through_F2_l19_19610


namespace line_passes_through_fixed_point_l19_19970

theorem line_passes_through_fixed_point
  (O A B C : Type)
  (m λ μ : ℝ)
  (h1 : λ = 1 / 3)
  (h2 : μ = 2 / 3)
  (h3 : ∀ (x y : ℝ), (m + λ) * x + (μ - 2 * m) * y + 3 * m = 0)
  (h4 : λ ∈ ℝ)
  (h5 : μ ∈ ℝ)
  (hOC : ∀ (OC OA OB : Type), OC = λ * OA + μ * OB) :
  ∃ (x y : ℝ), (x, y) = (-3 / 2, 3 / 4) :=
by
  sorry

end line_passes_through_fixed_point_l19_19970


namespace same_color_edges_l19_19978

structure Prism :=
  (vertices_upper : Fin 5 → V)
  (vertices_lower : Fin 5 → V)
  (color_edge : V → V → Prop)

def edges_colored_diff : Prop :=
  ∀ (i j : Fin 5), i ≠ j →
  ((color_edge (vertices_upper i) (vertices_upper j) ∧ ¬color_edge (vertices_upper (i + 1)mod 5) (vertices_lower i)) ∨
   (¬color_edge (vertices_upper i) (vertices_upper j) ∧ color_edge (vertices_upper (i + 1)mod 5) (vertices_lower i)))

theorem same_color_edges (P : Prism)
  (h_color : edges_colored_diff P) :
  (∀ i j, P.color_edge (P.vertices_upper i) (P.vertices_upper j)) ↔
  (∀ i j, P.color_edge (P.vertices_lower i) (P.vertices_lower j)) :=
sorry

end same_color_edges_l19_19978


namespace penalty_kicks_count_l19_19238

theorem penalty_kicks_count (players goalies : ℕ) (h1 : players = 18) (h2 : goalies = 4) : (goalies * (players - 1)) = 68 :=
by
  have h3: players - 1 = 17 := by
    simp [h1]
  simp [h2, h3]
  sorry

end penalty_kicks_count_l19_19238


namespace geometric_sequence_sum_l19_19977

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r)
    (h2 : r = 2) (h3 : a 1 * 2 + a 3 * 8 + a 5 * 32 = 3) :
    a 4 * 16 + a 6 * 64 + a 8 * 256 = 24 :=
sorry

end geometric_sequence_sum_l19_19977


namespace second_root_of_system_l19_19416

def system_of_equations (x y : ℝ) : Prop :=
  (2 * x^2 + 3 * x * y + y^2 = 70) ∧ (6 * x^2 + x * y - y^2 = 50)

theorem second_root_of_system :
  system_of_equations 3 4 →
  system_of_equations (-3) (-4) :=
by
  sorry

end second_root_of_system_l19_19416


namespace tobias_distance_swum_l19_19027

def swimming_pool_duration : Nat := 3 * 60 -- Tobias at the pool for 3 hours in minutes.

def swimming_duration : Nat := 5  -- Time taken to swim 100 meters in minutes.

def pause_duration : Nat := 5  -- Pause duration after every 25 minutes of swimming.

def swim_interval : Nat := 25  -- Swimming interval before each pause in minutes.

def total_pause_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  (total_minutes / (interval + pause)) * pause

def total_swimming_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  total_minutes - total_pause_time(total_minutes, interval, pause)

def distance_swum (swimming_minutes: Nat) (swim_duration: Nat) : Nat :=
  (swimming_minutes / swim_duration) * 100

theorem tobias_distance_swum :
  distance_swum (total_swimming_time swimming_pool_duration swim_interval pause_duration) swimming_duration = 3000 :=
by
  -- perform the calculation steps indicated in the solution
  sorry

end tobias_distance_swum_l19_19027


namespace exists_xy_l19_19272

-- Given conditions from the problem
variables (m x0 y0 : ℕ)
-- Integers x0 and y0 are relatively prime
variables (rel_prim : Nat.gcd x0 y0 = 1)
-- y0 divides x0^2 + m
variables (div_y0 : y0 ∣ x0^2 + m)
-- x0 divides y0^2 + m
variables (div_x0 : x0 ∣ y0^2 + m)

-- Main theorem statement
theorem exists_xy 
  (hm : m > 0) 
  (hx0 : x0 > 0) 
  (hy0 : y0 > 0) 
  (rel_prim : Nat.gcd x0 y0 = 1) 
  (div_y0 : y0 ∣ x0^2 + m) 
  (div_x0 : x0 ∣ y0^2 + m) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ Nat.gcd x y = 1 ∧ y ∣ x^2 + m ∧ x ∣ y^2 + m ∧ x + y ≤ m + 1 := 
sorry

end exists_xy_l19_19272


namespace min_guests_l19_19758

theorem min_guests (total_food : ℕ) (max_food : ℝ) 
  (H1 : total_food = 337) 
  (H2 : max_food = 2) : 
  ∃ n : ℕ, n = ⌈total_food / max_food⌉ ∧ n = 169 :=
by
  sorry

end min_guests_l19_19758


namespace triangle_right_angle_l19_19626

theorem triangle_right_angle (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : γ = α + β) : γ = 90 :=
by
  sorry

end triangle_right_angle_l19_19626


namespace number_of_days_woman_weaves_l19_19741

theorem number_of_days_woman_weaves
  (a_1 : ℝ) (a_n : ℝ) (S_n : ℝ) (n : ℝ)
  (h1 : a_1 = 5)
  (h2 : a_n = 1)
  (h3 : S_n = 90)
  (h4 : S_n = n * (a_1 + a_n) / 2) :
  n = 30 :=
by
  rw [h1, h2, h3] at h4
  sorry

end number_of_days_woman_weaves_l19_19741


namespace matrices_equal_l19_19700

open Matrix

variables {n : ℕ}
variables {x y : Fin n → ℝ}
variables (A B : Matrix (Fin n) (Fin n) ℝ)

-- Definition of matrix A based on the given conditions
def matrixA (i j : Fin n) : ℝ :=
  if x i + y j >= 0 then 1 else 0

-- Definition of matrix B satisfying the given conditions
def matrixB (B : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, B i j = 0 ∨ B i j = 1) ∧
  (∀ i, (∑ j, B i j) = (∑ j, matrixA x y i j)) ∧
  (∀ j, (∑ i, B i j) = (∑ i, matrixA x y i j))

theorem matrices_equal
  (hB : matrixB B) :
  A = B := sorry

end matrices_equal_l19_19700


namespace train_speed_proof_l19_19470

-- Define all necessary conditions
def length_of_train : ℕ := 200 -- length in meters
def time_to_cross_man : ℕ := 4 -- time in seconds
def speed_of_man_kmh : ℕ := 8 -- speed of man in km/h

-- Convert speed of man from km/h to m/s for calculations
def speed_of_man_ms : ℝ := (speed_of_man_kmh * 1000) / 3600

-- Condition: Correct Answer is 172 km/h
def correct_speed_of_train_kmh : ℝ := 172

-- The proof statement
theorem train_speed_proof :
  let relative_speed_ms := length_of_train / time_to_cross_man in
  let train_speed_ms := relative_speed_ms - speed_of_man_ms in
  let train_speed_kmh := (train_speed_ms * 3600) / 1000 in
  train_speed_kmh = correct_speed_of_train_kmh :=
by
  sorry

end train_speed_proof_l19_19470


namespace radius_of_circle_B_l19_19901

noncomputable def radius_of_B (rA : ℝ) (rD : ℝ) (rC : ℝ) : ℝ := 
  (2 + 4/3)^2 = (2 + 4/3)^2 + (16/9)^2 ∧ 
  (4 - 16/9)^2 = (4/3)^2 + (16/9)^2 → 
  rD = 4 → 
  rA = 2 → 
  (rC = rB) → 
  (B : circle) rB = 16/9

theorem radius_of_circle_B (A B C D : circle) (rA rB √rD rC : ℝ) (E H F G : point) : 
  (∀ P : point, tangent_point A B P) → externally_tangent A B →
  (∀ P : point, tangent_point A C P) → externally_tangent A C →
  (∀ P : point, tangent_point B C P) → externally_tangent B C →
  internally_tangent A D →
  internally_tangent B D →
  internally_tangent C D →
  congruent B C →
  radius A = 2 →
  passes_through_center A D → 
  radius D = 2 * radius A →
  radius_of_B 2 4 (radius C) = 16 / 9 :=
sorry  

end radius_of_circle_B_l19_19901


namespace max_a_range_m_min_a_exists_l19_19694

-- Problem 1
theorem max_a (a : ℝ) (h_pos : a > 0) (h_geq : ∀ x, e^x - a * (x + 1) ≥ 0) : a ≤ 1 := sorry

-- Problem 2
theorem range_m (m : ℝ) (a : ℝ) (h_leq_neg1 : a ≤ -1) (h_ge_slope : ∀ x, e^x - a / e^x - a ≥ m) : m ≤ 3 := sorry

-- Problem 3
theorem min_a_exists : ∃ a : ℕ, (∀ n : ℕ, n > 0 → (∑ (i : ℕ) in finset.range (2 * n), (if i % 2 = 1 then (i^n) else 0)) < (sqrt e / (e - 1) * (a * n)^n)) ∧ a = 2 := sorry

end max_a_range_m_min_a_exists_l19_19694


namespace ratio_x_y_l19_19123

noncomputable def right_triangle_ratio : ℚ :=
  let triangle := (5, 12, 13) in
  let x := 144 / 17 in
  let y := 169 / 30 in
  x / y

theorem ratio_x_y : right_triangle_ratio = 4320 / 2873 := by
  sorry

end ratio_x_y_l19_19123


namespace find_m_l19_19276

noncomputable def U := ℝ
def A : set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (m : ℝ) : set ℝ := {x | x^2 + (m+1) * x + m = 0}
def C_U (A : set ℝ) : set ℝ := {x | ¬ (x ∈ A)}

theorem find_m (m : ℝ) : (C_U A ∩ B m = ∅) ↔ (m = 1 ∨ m = 2) :=
sorry

end find_m_l19_19276


namespace domain_h_l19_19132

def h (x : ℝ) : ℝ := (5 * x - 2) / (x^2 + 2 * x - 15)

theorem domain_h : 
  (∀ x : ℝ, h x = (5 * x - 2) / (x^2 + 2 * x - 15) → x ≠ -5 ∧ x ≠ 3) →
  ∀ x, (x ≠ -5 ∧ x ≠ 3) ↔ (h x ≠ 0) :=
by
  intros
  split
  -- proof omitted
  sorry

end domain_h_l19_19132


namespace angle_sum_acutes_l19_19186

theorem angle_sum_acutes (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_condition : |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0) : 
  α + β = π * 5/12 :=
by sorry

end angle_sum_acutes_l19_19186


namespace cameron_answers_l19_19893

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end cameron_answers_l19_19893


namespace tangent_line_at_a_1_monotonic_intervals_intersection_g_for_a_minus_2_l19_19597

-- Define the function f(x) for all a in R
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 2 * x - 2) * Real.exp x
-- Define the function g(x) for part (3)
noncomputable def g (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * x^2

-- Part (1): Prove the tangent line equation
theorem tangent_line_at_a_1 : 
  let a := 1 in 
  let f_x := f a 1 in 
  let f_prime_1 := (f'(λ x, (1 * x^2 + 2 * x - 2) * Real.exp x)) 1 in 
  let point := (1, f_x) in 
  ((4 * Real.exp 1) * (1 - 1)) + f_x - (4 * Real.exp 1) * 1 = 3 * Real.exp 1 - y := sorry

-- Part (2): Prove the monotonic intervals
theorem monotonic_intervals (a : ℝ) (h: a < 0) : 
  (if a = -1/2 then 
    ∀ x : ℝ, derivative (λ x, f a x) x ≤ 0
  else if a < -1/2 
    then 
      ∀ x : ℝ, (x < -2 - 1 / a ∨ 0 < x) → derivative (λ x, f a x) x < 0
    else 
      ∀ x : ℝ, (x < 0 ∨ x > -2 - 1 / a) → derivative (λ x, f a x) x < 0  
  ) := sorry

-- Part (3): Prove the intersection with g(x) implies range for m
theorem intersection_g_for_a_minus_2 (m : ℝ): 
  let a := -2 in 
  let f_x := f a in 
  let F := λ x, (f_x x) - g x in 
  -- f(x) intersects g(x) at 3 points implies the range for m
  (∀ x : ℝ, F(-1) < m < F(0)) → 
  m ∈ (- (4 / Real.exp 1) - (1 / 6), -2) := sorry

end tangent_line_at_a_1_monotonic_intervals_intersection_g_for_a_minus_2_l19_19597


namespace math_problem_MATHEMATICS_l19_19006

theorem math_problem_MATHEMATICS : 
  let vowels := ['A', 'A', 'E', 'I'],
      consonants := ['M', 'T', 'H', 'M', 'T', 'C', 'S'],
      n_vowels := 4,
      n_consonants := 7,
      unique_vowel_perms := (Fact.factorial n_vowels) / (Fact.factorial 2),
      unique_consonant_perms := (Fact.factorial n_consonants) / (Fact.factorial 2 * Fact.factorial 2),
      total_permutations := unique_vowel_perms * unique_consonant_perms
  in total_permutations = 15120 :=
by
  sorry

end math_problem_MATHEMATICS_l19_19006


namespace exists_above_product_l19_19684

-- We define the existence of a doubly infinite array of positive integers 
-- where each positive integer appears exactly 8 times

def doubly_infinite_array (a : ℕ × ℕ → ℕ) : Prop :=
  ∀ k: ℕ, 8 = (finset.univ.image (λ mn, a mn)).count k

-- The theorem to prove the existence of (m, n) such that a_{m, n} > m * n
theorem exists_above_product (a : ℕ × ℕ → ℕ) (ha : doubly_infinite_array a) :
  ∃ (m n : ℕ), a (m, n) > m * n :=
begin
  sorry
end

end exists_above_product_l19_19684


namespace cos_angle_of_vectors_l19_19195

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem cos_angle_of_vectors (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = 2) :
  (inner a b) / (‖a‖ * ‖b‖) = 1/4 :=
by
  sorry

end cos_angle_of_vectors_l19_19195


namespace max_principals_in_10_years_l19_19489

theorem max_principals_in_10_years (term_length : ℕ) (period_length : ℕ) (max_principals : ℕ)
  (term_length_eq : term_length = 4) (period_length_eq : period_length = 10) :
  max_principals = 4 :=
by
  sorry

end max_principals_in_10_years_l19_19489


namespace angle_BAC_is_55_degrees_l19_19669

theorem angle_BAC_is_55_degrees
  (AB DC : Line)
  [parallel : Parallel AB DC]
  (A B C D E : Point)
  (h1 : angle ADE = 180)
  (h2 : angle ABC = 85)
  (h3 : angle ADC = 125)
  (h4 : ADE_is_straight : StraightLine ADE) :
  angle BAC = 55 := 
sorry

end angle_BAC_is_55_degrees_l19_19669


namespace percentage_of_primes_divisible_by_2_l19_19053

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def divisible_by (n k : ℕ) : Prop := k ∣ n

def percentage_divisible_by (k : ℕ) (lst : List ℕ) : ℚ :=
  (lst.filter (divisible_by k)).length / lst.length * 100

theorem percentage_of_primes_divisible_by_2 : 
  percentage_divisible_by 2 primes_less_than_20 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_2_l19_19053


namespace gray_region_area_l19_19900

-- Definitions based on given conditions
noncomputable def Circle (center : ℝ × ℝ) (radius : ℝ) : Type := sorry

def CircleA : Type := Circle (4, 5) 5
def CircleB : Type := Circle (12, 5) 5

-- Statement asserting the area of the gray region
theorem gray_region_area (A B : Type) (R : ℝ) : 
  A = Circle (4, 5) 5 →
  B = Circle (12, 5) 5 →
  R = 5 →
  let rect_area := 8 * R,
      semi_circ_area := (1 / 2) * Real.pi * R^2 in
  rect_area - 2 * semi_circ_area = 40 - 25 * Real.pi :=
by
  intros hA hB hR,
  -- Defer proof
  sorry

end gray_region_area_l19_19900


namespace emma_bank_account_balance_l19_19533

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l19_19533


namespace tangent_line_integer_intersections_count_l19_19715

def num_points_with_integer_tangent_intercepts : ℕ := 40

theorem tangent_line_integer_intersections_count :
  ∃ (f : ℝ → ℝ) (n : ℕ), (∀ x : ℝ, f x = 2020 / x) ∧ n = 40 := 
by 
  use (λ x, 2020 / x)
  use 40
  sorry

end tangent_line_integer_intersections_count_l19_19715


namespace minimum_product_of_eccentricities_l19_19997

-- Define the proof problem with the given conditions
theorem minimum_product_of_eccentricities :
  ∀ (F1 F2 P : Type) (angleF1PF2 : ℝ),
  is_foci_of_ellipse_and_hyperbola F1 F2 →
  is_common_point_of_ellipse_and_hyperbola P →
  angleF1PF2 = 60 * (Mathlib.pi / 180) → 
  ∃ (e1 e2 : ℝ), e1 * e2 = (Mathlib.sqrt 3) / 2 := sorry

end minimum_product_of_eccentricities_l19_19997


namespace find_additional_payment_l19_19839

-- Definitions used from the conditions
def total_payments : ℕ := 52
def first_partial_payments : ℕ := 25
def second_partial_payments : ℕ := total_payments - first_partial_payments
def first_payment_amount : ℝ := 500
def average_payment : ℝ := 551.9230769230769

-- Condition in Lean
theorem find_additional_payment :
  let total_amount := average_payment * total_payments
  let first_payment_total := first_partial_payments * first_payment_amount
  ∃ x : ℝ, total_amount = first_payment_total + second_partial_payments * (first_payment_amount + x) → x = 100 :=
by
  sorry

end find_additional_payment_l19_19839


namespace part_a_part_b_part_c_l19_19847

-- Part (a)
theorem part_a (Q_A: list ℕ) (Q_B: list ℕ) (emigrant: ℕ) (h1: emigrant ∈ Q_A)
                (h2: Q_A ≠ []) (h3: Q_B ≠ []) :
                (average_without Q_A emigrant < average Q_A) →
                (average_with Q_B emigrant > average Q_B) :
  ∃ Q_A' Q_B', average Q_A' > average Q_A ∧ average Q_B' > average Q_B := sorry

-- Part (b)
theorem part_b (Q_A: list ℕ) (Q_B: list ℕ) (emigrant: ℕ) (h: emigrant ∈ Q_B) 
               (hA: Q_A ≠ []) (hB: Q_B ≠ []):
               average Q_A < average Q_B →
               average_without Q_B emigrant > average Q_B →
               average_with Q_A emigrant < average Q_A : 
               false := sorry

-- Part (c)
theorem part_c (Q_A Q_B Q_C: list ℕ) 
               (emigrants_AB: list ℕ) 
               (emigrants_BC: list ℕ) 
               (emigrants_CB: list ℕ)
               (emigrants_BA: list ℕ)
               (Q_AB: list ℕ) (Q_BC: list ℕ) (Q_CA: list ℕ)
               (mA: Q_A ≠ []) (mB: Q_B ≠ []) (mC: Q_C ≠ []):
               average (Q_A \ Q_AB) < average (Q_A \ Q_AB ∪ Q_BA) ∧ 
               average (Q_B \ Q_BC) < average (Q_B \ Q_BC ∪ Q_CB) ∧ 
               average (Q_C \ Q_CA) < average (Q_C \ Q_CA ∪ Q_AB) → 
               (average Q_A' > average Q_A ∧ average Q_B' > average Q_B ∧ average Q_C' > average Q_C) :=
  sorry

end part_a_part_b_part_c_l19_19847


namespace probability_region_omega_l19_19100

noncomputable def region_area : ℝ :=
  -∫ x in 0..(Real.pi/2), Real.cos x

noncomputable def rectangle_area : ℝ :=
  (Real.pi / 2) * 1

theorem probability_region_omega :
  let omega_area := abs region_area;
  let rectangle_area := rectangle_area in
  omega_area / rectangle_area = 2 / Real.pi :=
by
  sorry

end probability_region_omega_l19_19100


namespace joyce_gave_oranges_l19_19903

variable o1 o2 a : ℕ

theorem joyce_gave_oranges (h1 : o1 = 5) (h2 : o2 = 8) (h3 : o2 = o1 + a) : a = 3 := 
by
  sorry

end joyce_gave_oranges_l19_19903


namespace jerry_remaining_debt_l19_19262

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l19_19262


namespace area_square_of_triangle_l19_19126

open Real

-- Definition of the problem
theorem area_square_of_triangle {ABCD : Type} [convex_quadrilateral ABCD]
  (angle_B : angle B = π/2) (angle_C : angle C = π/2)
  (BC : length BC = 20) (AD : length AD = 30)
  (perpendicular_diagonals : perpendicular (diagonal AC) (diagonal BD)) :
  let CD := side length of segment CD,
      AB := side length of segment AB in
  ((herons_formula CD AD AB)^2 = 30000) :=
sorry

-- Heron's formula definition
noncomputable def herons_formula (a b c: ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  sqrt (s * (s - a) * (s - b) * (s - c))

end area_square_of_triangle_l19_19126


namespace hyperbola_focus_distance_l19_19185

theorem hyperbola_focus_distance :
  ∀ (F1 F2 P : ℝ × ℝ),
  ((∀ x y, x^2 / 16 - y^2 / 20 = 1) →  -- Hyperbola condition
  (P.fst^2 / 16 - P.snd^2 / 20 = 1) →  -- Point P is on the hyperbola
  (dist P F1 = 9) →                     -- Distance from P to F1
  abs (dist P F1 - dist P F2) = 8 →   -- Hyperbola focus property
  dist P F2 = 17) := 
begin
  sorry
end

end hyperbola_focus_distance_l19_19185


namespace container_fullness_calc_l19_19834

theorem container_fullness_calc (initial_percent : ℝ) (added_water : ℝ) (total_capacity : ℝ) (result_fraction : ℝ) :
  initial_percent = 0.3 →
  added_water = 27 →
  total_capacity = 60 →
  result_fraction = 3/4 →
  ((initial_percent * total_capacity + added_water) / total_capacity) = result_fraction :=
by
  intros h1 h2 h3 h4
  sorry

end container_fullness_calc_l19_19834


namespace smallest_pieces_left_l19_19577

theorem smallest_pieces_left (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) : 
    ∃ k, (k = 2 ∧ (m * n) % 3 = 0) ∨ (k = 1 ∧ (m * n) % 3 ≠ 0) :=
by
    sorry

end smallest_pieces_left_l19_19577


namespace total_percentage_of_failed_candidates_is_correct_l19_19246

def total_candidates : ℕ := 2000
def number_of_girls : ℕ := 900
def number_of_boys : ℕ := total_candidates - number_of_girls
def percentage_boys_passed : ℚ := 38 / 100
def percentage_girls_passed : ℚ := 32 / 100
def number_of_boys_passed : ℚ := percentage_boys_passed * number_of_boys
def number_of_girls_passed : ℚ := percentage_girls_passed * number_of_girls
def total_candidates_passed : ℚ := number_of_boys_passed + number_of_girls_passed
def total_candidates_failed : ℚ := total_candidates - total_candidates_passed
def total_percentage_failed : ℚ := (total_candidates_failed / total_candidates) * 100

theorem total_percentage_of_failed_candidates_is_correct :
  total_percentage_failed = 64.7 := by
  sorry

end total_percentage_of_failed_candidates_is_correct_l19_19246


namespace largest_possible_integer_l19_19851

theorem largest_possible_integer {l : List ℕ} (h_length : l.length = 5)
  (h_pos : ∀ n ∈ l, 0 < n)
  (h_repeat : l.count 7 > 1 ∧ ∀ n ≠ 7, l.count n ≤ 1)
  (h_median : l.sorted = true ∧ l.nth_le 2 (by linarith) = 10)
  (h_mean : l.sum = 5 * 12) : 
  ∃ n ∈ l, n = 25 := 
sorry

end largest_possible_integer_l19_19851


namespace paper_folds_to_pentagon_l19_19232

noncomputable def fold_paper_to_pentagon (sq : Square) : Pentagon := sorry

theorem paper_folds_to_pentagon (sq : Square) : is_regular_pentagon (fold_paper_to_pentagon sq) :=
sorry

end paper_folds_to_pentagon_l19_19232


namespace statement2_statement3_correct_statements_l19_19593

-- Definitions for the conditions
def condition1 (a : ℝ) (h : a < 0) : (a ^ 2) ^ (3 / 2) = a ^ 3 := sorry

def condition2 (n : ℕ) (a : ℝ) (h1 : 1 < n) (h2 : n % 2 = 0) : n * a ^ n = |a| := 
  by sorry

def func_domain (x : ℝ) : Prop :=
  x >= 2 ∧ x ≠ 7 / 3

def f (x : ℝ) : ℝ := (x - 2) ^ (1 / 2) - (3 * x - 7) ^ 0

-- Prove statements
theorem statement2 (n : ℕ) (a : ℝ) (h1 : 1 < n) (h2 : n % 2 = 0) : n * a ^ n = |a| := 
  condition2 n a h1 h2

theorem statement3 : ∀ x, func_domain x → (f x = (x - 2) ^ (1 / 2) - 1) :=
  by sorry

-- Main theorem: Verify the correct statements
theorem correct_statements : statement2 ∧ statement3 := by
  sorry

end statement2_statement3_correct_statements_l19_19593


namespace no_prime_pairs_sum_53_l19_19661

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l19_19661


namespace ratio_girls_to_boys_l19_19641

theorem ratio_girls_to_boys (total_students : ℕ) (more_girls_than_boys : ℕ) (h_total : total_students = 30) (h_difference : more_girls_than_boys = 5) :
  let g := (total_students + more_girls_than_boys) / 2,
      b := (total_students - more_girls_than_boys) / 2 in
  b ≠ 0 ∧ g / b = 3 / 2 :=
by
  sorry

end ratio_girls_to_boys_l19_19641


namespace trapezoid_EFGH_area_l19_19056

-- Define the vertices of the trapezoid
structure Point where
  x : ℝ
  y : ℝ

def E : Point := {x := 2, y := 0}
def F : Point := {x := 2, y := 3}
def G : Point := {x := 8, y := 5}
def H : Point := {x := 8, y := -1}

-- Define the lengths of the bases and height of the trapezoid
def length_EF : ℝ := F.y - E.y
def length_GH : ℝ := G.y - H.y
def height : ℝ := G.x - E.x

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 h : ℝ) : ℝ := 0.5 * (b1 + b2) * h

-- Theorem stating the area of the trapezoid EFGH
theorem trapezoid_EFGH_area : trapezoid_area length_EF length_GH height = 27 := by
  -- This is where the proof would go
  sorry

end trapezoid_EFGH_area_l19_19056


namespace cuboid_surface_area_correct_l19_19346

-- Define the dimensions of the cuboid
def l : ℕ := 4
def w : ℕ := 5
def h : ℕ := 6

-- Define the function to calculate the surface area of the cuboid
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

-- The theorem stating that the surface area of the cuboid is 148 cm²
theorem cuboid_surface_area_correct : surface_area l w h = 148 := by
  sorry

end cuboid_surface_area_correct_l19_19346


namespace projection_of_AB_onto_CD_l19_19182

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (1, 2)
noncomputable def C : ℝ × ℝ := (-2, -1)
noncomputable def D : ℝ × ℝ := (3, 4)

noncomputable def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_AB_onto_CD :
  let AB := vector_sub A B
  let CD := vector_sub C D
  (magnitude AB) * (dot_product AB CD) / (magnitude CD) ^ 2 = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end projection_of_AB_onto_CD_l19_19182


namespace no_prime_pair_summing_to_53_l19_19656

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l19_19656


namespace correct_answer_l19_19309

-- Define the propositions p and q
def prop_p (a : Line) (α β : Plane) : Prop :=
  (a ⊆ α) → (a ⊥ β) → (α ⊥ β)

def prop_q : Prop :=
  ∀ P : Polyhedron, P.has_two_parallel_faces_and_all_other_faces_trapezoids → ¬P.is_prism

-- The main statement
theorem correct_answer (a : Line) (α β : Plane) (P : Polyhedron) :
  prop_p a α β ∧ (¬ prop_q) :=
by
  -- Placeholder for proof
  sorry

end correct_answer_l19_19309


namespace min_value_Q_l19_19557

/-- For every odd integer \( k \) within the range \( 1 \leq k \leq 49 \), let \( Q(k) \) represent 
the probability that
\[
\left\lfloor \frac{n}{k} \right\rfloor + \left\lfloor \frac{80 - n}{k} \right\rfloor = 
\left\lfloor \frac{80}{k} \right\rfloor
\]
holds true for an integer \( n \) randomly selected from the set \{1, 2, ..., 50\}. 
Determine the minimum possible value of \( Q(k) \) over the specified range of \( k \). 
-/


theorem min_value_Q (k : ℕ) (hk : odd k ∧ 1 ≤ k ∧ k ≤ 49) :
  (∃ (Q : ℕ → ℚ), Q(k) = (1 / 2)) :=
by
  sorry

end min_value_Q_l19_19557


namespace trapezium_shorter_side_l19_19146

theorem trapezium_shorter_side (a b h : ℝ) (H1 : a = 10) (H2 : b = 18) (H3 : h = 10.00001) : a = 10 :=
by
  sorry

end trapezium_shorter_side_l19_19146


namespace johns_average_speed_l19_19267

theorem johns_average_speed :
  let distance1 := 20
  let speed1 := 10
  let distance2 := 30
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 14.29 :=
by
  sorry

end johns_average_speed_l19_19267


namespace ratio_of_fractions_proof_l19_19817

noncomputable def ratio_of_fractions (x y : ℝ) : Prop :=
  (5 * x = 6 * y) → (x ≠ 0 ∧ y ≠ 0) → ((1/3) * x / ((1/5) * y) = 2)

theorem ratio_of_fractions_proof (x y : ℝ) (hx: 5 * x = 6 * y) (hnz: x ≠ 0 ∧ y ≠ 0) : ((1/3) * x / ((1/5) * y) = 2) :=
  by 
  sorry

end ratio_of_fractions_proof_l19_19817


namespace inverse_g_eval_l19_19167

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 5

theorem inverse_g_eval :
  (∃ x : ℝ, g x = 3 / 1240) → ∃ x : ℝ, x = (1255 / 1240)^(1 / 7) :=
begin
  intro h,
  cases h with x hx,
  use (1255 / 1240)^(1 / 7),
  sorry
end

end inverse_g_eval_l19_19167


namespace simplify_exponent_l19_19421

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l19_19421


namespace julia_game_difference_l19_19269

theorem julia_game_difference :
  let tag_monday := 28
  let hide_seek_monday := 15
  let tag_tuesday := 33
  let hide_seek_tuesday := 21
  let total_monday := tag_monday + hide_seek_monday
  let total_tuesday := tag_tuesday + hide_seek_tuesday
  let difference := total_tuesday - total_monday
  difference = 11 := by
  sorry

end julia_game_difference_l19_19269


namespace min_tangent_length_l19_19631

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end min_tangent_length_l19_19631


namespace solve_h_eq_5_l19_19289

def h (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else 3 * x - 18

theorem solve_h_eq_5 (x : ℝ) : h(x) = 5 ↔ (x = -1 ∨ x = 23 / 3) :=
by
  sorry

end solve_h_eq_5_l19_19289


namespace determinant_expression_l19_19913

variables {ℝ : Type*} [inner_product_space ℝ ℝ^3]

variables (u v w x : ℝ^3)
variable (E : ℝ)
variable (F : ℝ)

def determinant_matrix_with_columns (a b c : ℝ^3) : ℝ :=
  a • (b × c)

noncomputable def E' (u v w x : ℝ^3) :=
  determinant_matrix_with_columns (u × v) (v × w) (w × x)

theorem determinant_expression (u v w x : ℝ^3) (E : ℝ) (F : ℝ) 
  (h1 : E = (u • (v × w))) 
  (h2 : F = (w • (u × x))) 
  (h3 : determinant_matrix_with_columns u v w = E) :
  E' u v w x = F * E :=
sorry

end determinant_expression_l19_19913


namespace problem_1_l19_19428

theorem problem_1 (a : ℝ) : (nat.choose 10 3) * a^3 = 15 → a = 1/2 :=
by sorry

end problem_1_l19_19428


namespace equilateral_triangle_inscribed_circle_area_correct_l19_19369

noncomputable def equilateral_triangle_inscribed_circle_area : ℝ :=
  let s := 6
  let A := (0, 0)
  let B := (6, 0)
  let C := (3, 3 * Real.sqrt 3)
  let incenter := (3, Real.sqrt 3)
  let radius := Real.sqrt 3 * 2 -- Distance from incenter to vertex
  π * radius^2

theorem equilateral_triangle_inscribed_circle_area_correct :
  equilateral_triangle_inscribed_circle_area = 12 * π :=
by
  sorry

end equilateral_triangle_inscribed_circle_area_correct_l19_19369


namespace trains_cross_time_l19_19431

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time_l19_19431


namespace earnings_from_jam_l19_19493

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l19_19493


namespace length_eq_l19_19397

namespace vector_proposition

variable {A B : ℝ^3} 

theorem length_eq (A B : ℝ^3) : 
  (dist A B) = (dist B A) :=
by
  sorry

end vector_proposition

end length_eq_l19_19397


namespace max_on_edge_l19_19304

/- Define the grid structure and the properties of the arithmetic mean -/
structure Grid (α : Type) :=
  (data : ℕ × ℕ → α)
  (mean_property : ∀ i j : ℕ, 1 ≤ i ∧ 1 ≤ j →
    data (i, j) = (data (i-1, j) + data (i+1, j) + data (i, j-1) + data (i, j+1)) / 4)

variables {α : Type} [LinearOrder α] [Add α] [Div α α]
/- A selected portion of the grid -/
structure SelectedPortion (α : Type) :=
  (region : set (ℕ × ℕ))
  (grid : Grid α)
  (mem_region : ∀ i j : ℕ, region (i, j) → 1 ≤ i ∧ 1 ≤ j)

noncomputable def is_edge (portion : SelectedPortion α) (i j : ℕ) : Prop :=
  !portion.region (i - 1, j) ∨ !portion.region (i + 1, j) ∨ !portion.region (i, j - 1) ∨ !portion.region (i, j + 1)

theorem max_on_edge (portion : SelectedPortion α) (i j : ℕ) (h : ∀ (i' j' : ℕ), portion.region (i', j') → portion.grid.data (i, j) > portion.grid.data (i', j')) :
  is_edge portion i j :=
sorry

end max_on_edge_l19_19304


namespace cos_x_minus_pi_over_3_l19_19621

theorem cos_x_minus_pi_over_3 (x : ℝ) (h : Real.sin (x + π / 6) = 4 / 5) :
  Real.cos (x - π / 3) = 4 / 5 :=
sorry

end cos_x_minus_pi_over_3_l19_19621


namespace simplify_expression_l19_19734

noncomputable def original_expression (x : ℝ) : ℝ :=
(x - 3 * x / (x + 1)) / ((x - 2) / (x^2 + 2 * x + 1))

theorem simplify_expression:
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 2 → 
  (original_expression x = x^2 + x) ∧ 
  ((x = 1 → original_expression x = 2) ∧ (x = 0 → original_expression x = 0)) :=
by
  intros
  sorry

end simplify_expression_l19_19734


namespace coreys_candies_l19_19740

variable (C : ℝ) (Tapanga Corey : ℝ)

theorem coreys_candies :
  Tapanga + Corey = 66.5 ∧ Tapanga = Corey + 8.25 → Corey = 29.125 :=
by
  intro h
  cases h with hT hCorey
  -- Add missing assumptions
  have h1 : 2 * Corey + 8.25 = 66.5, { sorry }
  have h2 : 2 * Corey = 58.25, { sorry }
  sorry

end coreys_candies_l19_19740


namespace A_squared_infinite_possible_l19_19691

variables {A : Matrix (Fin 2) (Fin 2) ℝ}

theorem A_squared_infinite_possible (h : A^4 = 0) : ∃ b : ℝ, ∃ c : ℝ, ∃ d : ℝ, ∃ (A_squared : Matrix (Fin 2) (Fin 2) ℝ), A_squared = Matrix.vecCons (λ i, Matrix.vecCons (λ j, if (i = 1 ∧ j = 1) then b else (if (i = 2 ∧ j =2) then c else 0))) (λ i, Matrix.vecCons (λ j, if (i = 1 ∧ j = 1) then d else if (i = 2 ∧ j = 2) then b else 0)) :=
sorry

end A_squared_infinite_possible_l19_19691


namespace largest_angle_in_triangle_l19_19361

theorem largest_angle_in_triangle (a b : ℝ) (h1 : a + b = 126) (h2 : a + 40 = b) : 
  ∃ c, c ∈ ({a, b, c} : set ℝ) ∧ c = 83 := by
  sorry

end largest_angle_in_triangle_l19_19361


namespace real_solutions_of_polynomial_l19_19520

theorem real_solutions_of_polynomial :
  ∀ x : ℝ, x^4 - 3 * x^3 + x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end real_solutions_of_polynomial_l19_19520


namespace f_neg_2_not_defined_l19_19001

variable (a : ℝ) (f : ℝ → ℝ)
hypothesis h_a_pos : a > 0
hypothesis h_a_ne_one : a ≠ 1
hypothesis h_f_2_eq_3 : f 2 = log a 2 = 3

theorem f_neg_2_not_defined : ¬(∃ y : ℝ, f (-2) = y) :=
by
  sorry

end f_neg_2_not_defined_l19_19001


namespace min_value_expression_l19_19383

theorem min_value_expression : ∃ x y : ℝ, (x^2 * y - 2)^2 + (x^2 + y)^2 = 4 :=
by
  use [0, 0]
  simp
  sorry

end min_value_expression_l19_19383


namespace sufficient_condition_l19_19426

theorem sufficient_condition (x y : ℤ) (h : x + y ≠ 2) : x ≠ 1 ∧ y ≠ 1 := 
sorry

end sufficient_condition_l19_19426


namespace length_of_DP_l19_19250

theorem length_of_DP
  (ABCD_square : ∀ (A B C D: ℝ), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ (A - B)^2 + (B - C)^2 + (C - D)^2 + (D - A)^2 = 4 * 8^2)
  (WXYZ_rectangle : ∀ (W X Y Z: ℝ), (Z - Y) = 12 ∧ (X - Y) = 4 ∧ (A - W) ∣ (B - X))
  (perpendicular_AD_WX : ∀ (A D W X: ℝ), (A - D) ⊥ (W - X))
  (shaded_area_condition : ∀ (area_WXYZ: ℝ), area_WXYZ = 48 → 36 = 3 / 4 * area_WXYZ):
  DP = 4.5 :=
by sorry

end length_of_DP_l19_19250


namespace max_y_on_graph_l19_19929

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l19_19929


namespace cookies_per_kid_l19_19709

theorem cookies_per_kid (total_calories_per_lunch : ℕ) (burger_calories : ℕ) (carrot_calories_per_stick : ℕ) (num_carrot_sticks : ℕ) (cookie_calories : ℕ) (num_cookies : ℕ) : 
  total_calories_per_lunch = 750 →
  burger_calories = 400 →
  carrot_calories_per_stick = 20 →
  num_carrot_sticks = 5 →
  cookie_calories = 50 →
  num_cookies = (total_calories_per_lunch - (burger_calories + num_carrot_sticks * carrot_calories_per_stick)) / cookie_calories →
  num_cookies = 5 :=
by
  sorry

end cookies_per_kid_l19_19709


namespace sqrt_meaningful_range_l19_19230

theorem sqrt_meaningful_range (x : ℝ) : (∃ y, y = sqrt (x - 1) ∧ 0 ≤ y) ↔ (x ≥ 1) :=
by
  sorry

end sqrt_meaningful_range_l19_19230


namespace perpendicular_lines_slope_product_l19_19181

theorem perpendicular_lines_slope_product (a : ℝ) (x y : ℝ) :
  let l1 := ax + y + 2 = 0
  let l2 := x + y = 0
  ( -a * -1 = -1 ) -> a = -1 :=
sorry

end perpendicular_lines_slope_product_l19_19181


namespace fraction_addition_l19_19566

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4)

theorem fraction_addition (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
  sorry

end fraction_addition_l19_19566


namespace smallest_positive_integer_l19_19456

-- Definitions of the conditions
def condition1 (k : ℕ) : Prop := k % 10 = 9
def condition2 (k : ℕ) : Prop := k % 9 = 8
def condition3 (k : ℕ) : Prop := k % 8 = 7
def condition4 (k : ℕ) : Prop := k % 7 = 6
def condition5 (k : ℕ) : Prop := k % 6 = 5
def condition6 (k : ℕ) : Prop := k % 5 = 4
def condition7 (k : ℕ) : Prop := k % 4 = 3
def condition8 (k : ℕ) : Prop := k % 3 = 2
def condition9 (k : ℕ) : Prop := k % 2 = 1

-- Statement of the problem
theorem smallest_positive_integer : ∃ k : ℕ, 
  k > 0 ∧
  condition1 k ∧ 
  condition2 k ∧ 
  condition3 k ∧ 
  condition4 k ∧ 
  condition5 k ∧ 
  condition6 k ∧ 
  condition7 k ∧ 
  condition8 k ∧ 
  condition9 k ∧
  k = 2519 := 
sorry

end smallest_positive_integer_l19_19456


namespace factorization_of_polynomial_l19_19539

theorem factorization_of_polynomial (x : ℤ) :
  3 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x ^ 2 =
  (3 * x ^ 2 + 58 * x + 231) * (x + 7) * (x + 11) :=
begin
  sorry
end

end factorization_of_polynomial_l19_19539


namespace probability_blue_or_green_face_l19_19842

def cube_faces: ℕ := 6
def blue_faces: ℕ := 3
def red_faces: ℕ := 2
def green_faces: ℕ := 1

theorem probability_blue_or_green_face (h1: blue_faces + red_faces + green_faces = cube_faces):
  (3 + 1) / 6 = 2 / 3 :=
by
  sorry

end probability_blue_or_green_face_l19_19842


namespace sheela_monthly_income_l19_19412

variable (deposits : ℝ) (percentage : ℝ) (monthly_income : ℝ)

-- Conditions
axiom deposit_condition : deposits = 3400
axiom percentage_condition : percentage = 0.15
axiom income_condition : deposits = percentage * monthly_income

-- Proof goal
theorem sheela_monthly_income :
  monthly_income = 3400 / 0.15 :=
sorry

end sheela_monthly_income_l19_19412


namespace prob_ace_then_king_l19_19373

theorem prob_ace_then_king :
  let total_cards := 52
  let total_aces := 4
  let total_kings := 4
  let prob_first_ace := total_aces / total_cards
  let prob_second_king := total_kings / (total_cards - 1)
  prob_first_ace * prob_second_king = 4 / 663 := by
{
  -- Definitions
  let total_cards := 52
  let total_aces := 4
  let total_kings := 4

  -- Calculation of probability
  let prob_first_ace := total_aces / total_cards
  let prob_second_king := total_kings / (total_cards - 1)
  have h : prob_first_ace * prob_second_king = 4 / 663 := by sorry,

  -- Return the result
  h,
}

end prob_ace_then_king_l19_19373


namespace find_value_expression_l19_19286

theorem find_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 4)
  (h3 : z^2 + x * z + x^2 = 79) :
  x * y + y * z + x * z = 20 := 
sorry

end find_value_expression_l19_19286


namespace course_selection_l19_19862

theorem course_selection (A B : ℕ) (cA cB : ℕ) (total : ℕ) :
  cA = 3 → cB = 4 → total = 3 →
  ∑ i in finset.range (total + 1), 
    (if i > 0 ∧ total - i > 0 then 
      (nat.choose cA i * nat.choose cB (total - i)) 
    else 0) = 30 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end course_selection_l19_19862


namespace solve_inequality_l19_19355

-- Define the odd and monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Assume the given conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom decreasing_f : ∀ x y, x < y → y < 0 → f x > f y
axiom f_at_2 : f 2 = 0

-- The proof statement
theorem solve_inequality (x : ℝ) : (x - 1) * f (x + 1) > 0 ↔ -3 < x ∧ x < -1 :=
by
  -- Proof omitted
  sorry

end solve_inequality_l19_19355


namespace chairs_per_table_l19_19750

theorem chairs_per_table (x y : ℕ) (total_chairs : ℕ) 
  (tables_indoor tables_outdoor : ℕ)
  (h_tables_indoor : tables_indoor = 8)
  (h_tables_outdoor : tables_outdoor = 12)
  (h_total_chairs : total_chairs = 60)
  (h_chairs_indoor : tables_indoor * x = 8 * x)
  (h_chairs_outdoor : tables_outdoor * y = 12 * y)
  (h_chairs_equal : x = y) :
  x = 3 :=
by
  have h : 8 * x + 12 * y = 60, from sorry,
  have h' : 20 * x = 60, from sorry,
  have h'' : x = 3, from sorry,
  exact h''

end chairs_per_table_l19_19750


namespace nine_chapters_problem_l19_19744

variable (m n : ℕ)

def horses_condition_1 : Prop := m + n = 100
def horses_condition_2 : Prop := 3 * m + n / 3 = 100

theorem nine_chapters_problem (h1 : horses_condition_1 m n) (h2 : horses_condition_2 m n) :
  (m + n = 100 ∧ 3 * m + n / 3 = 100) :=
by
  exact ⟨h1, h2⟩

end nine_chapters_problem_l19_19744


namespace correct_function_l19_19482

def f1 (x : ℝ) : ℝ := x + x⁻¹
def f2 (x : ℝ) : ℝ := x^2 + x⁻²
def f3 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)
def f4 (x : ℝ) : ℝ := 2^(-x) - 2^x

theorem correct_function :
  (f3 (-x) = f3 x) ∧ (∀ x < 0, (f3 'x < 0)) := 
sorry

end correct_function_l19_19482


namespace number_of_girls_l19_19358

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end number_of_girls_l19_19358


namespace max_y_coordinate_l19_19952

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l19_19952


namespace two_pt_seven_five_as_fraction_l19_19796

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l19_19796


namespace determine_by_median_l19_19020

-- Define the conditions
def num_students : ℕ := 19
def top_students : ℕ := 10

-- Define the property about the median helping in decision making
def can_determine_finalists (scores : List ℕ) (student_score : ℕ) : Prop :=
  scores.nth (num_students / 2) ≤ student_score

-- Lean theorem statement
theorem determine_by_median (scores : List ℕ) (student_score : ℕ) (h : scores.length = num_students) (h_distinct : scores.nodup):
  can_determine_finalists scores student_score ↔ ∃ k, k < top_students ∧ 
    ∃ xs ys zs, xs.length = k ∧ ys.length = 1 ∧ zs.length = num_students - (k + 1) ∧
    (xs ++ ys ++ zs = scores) ∧ (ys.nth 0 = some student_score) :=
sorry

end determine_by_median_l19_19020


namespace PQ_passes_through_circumcenter_l19_19177

theorem PQ_passes_through_circumcenter
  (A B C P Q : Point)
  (hABC : Triangle A B C)
  (hProjectionsSimilar : ∀ (X Y Z : Point), ProjectionsSimilar A B C P Q X Y Z) :
  PassesThroughCircumcenter A B C P Q :=
sorry

end PQ_passes_through_circumcenter_l19_19177


namespace emma_bank_account_balance_l19_19534

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l19_19534


namespace portfolio_weighted_average_yield_l19_19401

-- Define the conditions
def yield_stock_A : ℝ := 0.21
def yield_stock_B : ℝ := 0.15
def investment_A : ℝ := 10000
def investment_B : ℝ := 15000

-- Define the yields from each stock
def yield_from_A : ℝ := investment_A * yield_stock_A
def yield_from_B : ℝ := investment_B * yield_stock_B

-- Define total yield and total investment
def total_yield : ℝ := yield_from_A + yield_from_B
def total_investment : ℝ := investment_A + investment_B

-- Define the weighted average yield
def weighted_average_yield : ℝ := total_yield / total_investment

-- The theorem to be proved: the weighted average yield is 0.174 (or 17.4%)
theorem portfolio_weighted_average_yield : weighted_average_yield = 0.174 := by
  sorry

end portfolio_weighted_average_yield_l19_19401


namespace cubes_with_one_colored_face_l19_19350

theorem cubes_with_one_colored_face (n : ℕ) (c1 : ℕ) (c2 : ℕ) :
  (n = 64) ∧ (c1 = 4) ∧ (c2 = 4) → ((4 * n) * 2) / n = 32 :=
by 
  sorry

end cubes_with_one_colored_face_l19_19350


namespace equilateral_triangle_side_length_l19_19307

theorem equilateral_triangle_side_length (P Q R S : Point) (A B C : Point) (x s : ℝ)
  (h1 : P ∈ triangle ABC)
  (h2 : is_foot_perpendicular P Q (segment AB))
  (h3 : is_foot_perpendicular P R (segment BC))
  (h4 : is_foot_perpendicular P S (segment CA))
  (h5 : distance P Q = x)
  (h6 : distance P R = 2 * x)
  (h7 : distance P S = 3 * x)
  (h8 : is_equilateral_triangle A B C)
  : s = 4 * sqrt 3 * x := sorry

end equilateral_triangle_side_length_l19_19307


namespace smaller_variance_A_l19_19363

-- Definitions based on given conditions
def isConcentrated (data : List ℝ) (lower upper : ℝ) : Prop :=
  ∀ x ∈ data, lower ≤ x ∧ x ≤ upper

def isDispersed (data : List ℝ) : Prop :=
  ¬ isConcentrated data 0.06 0.07

-- Given data for locations A and B
def dataA : List ℝ := [0.061, 0.062, 0.063, 0.065, 0.067, 0.069]
def dataB : List ℝ := [0.05, 0.09, 0.07, 0.08, 0.04]

-- Given conditions
axiom dataA_concentrated : isConcentrated dataA 0.06 0.07
axiom dataB_dispersed : isDispersed dataB

-- Statement to prove
theorem smaller_variance_A :
  (∀ x ∈ dataA, 0.06 ≤ x ∧ x ≤ 0.07) ∧ (¬ ∀ x ∈ dataB, 0.06 ≤ x ∧ x ≤ 0.07) →
  variance dataA < variance dataB :=
sorry

end smaller_variance_A_l19_19363


namespace number_of_sides_of_polygon_l19_19754

theorem number_of_sides_of_polygon : 
  ∀ (n : ℕ), let d := n * (n - 3) / 2 in d - n = 7 → n = 7 :=
by
  intros n d h
  sorry

end number_of_sides_of_polygon_l19_19754


namespace log_power_relationship_l19_19618

theorem log_power_relationship (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (hm : m = Real.log c / Real.log a)
  (hn : n = Real.log c / Real.log b)
  (hr : r = a^c) :
  r > m ∧ m > n :=
sorry

end log_power_relationship_l19_19618


namespace tickets_difference_is_cost_l19_19490

def tickets_won : ℝ := 48.5
def yoyo_cost : ℝ := 11.7
def tickets_left (w : ℝ) (c : ℝ) : ℝ := w - c
def difference (w : ℝ) (l : ℝ) : ℝ := w - l

theorem tickets_difference_is_cost :
  difference tickets_won (tickets_left tickets_won yoyo_cost) = yoyo_cost :=
by
  -- Proof will be written here
  sorry

end tickets_difference_is_cost_l19_19490


namespace count_trapezoids_in_22_gon_l19_19379

-- Definition of the problem
def regular_polygon (n : ℕ) := n ≥ 3

def trapezoid (n : ℕ) := 
  ∃ (a b c d : ℕ), 
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ n ∧ 
  (∃ (p q : ℕ), 
    p ≠ q ∧ p ∈ (set.range (λ k, (a + k) % n)) ∧ 
    q ∈ (set.range (λ k, (b + k) % n)) )

-- Theorem statement
theorem count_trapezoids_in_22_gon : 
  count_trapezoids 22 = 990 := 
sorry

end count_trapezoids_in_22_gon_l19_19379


namespace round_to_nearest_hundredth_l19_19731

theorem round_to_nearest_hundredth (x : ℝ) (digits : ℕ) (ht : x = 24.6374) (ht_digits : digits = 2) :
  Real.round_to x digits = 24.64 :=
by
  sorry

end round_to_nearest_hundredth_l19_19731


namespace blue_water_bottles_initial_count_l19_19772

theorem blue_water_bottles_initial_count
    (red : ℕ) (black : ℕ) (taken_out : ℕ) (left : ℕ) (initial_blue : ℕ) :
    red = 2 →
    black = 3 →
    taken_out = 5 →
    left = 4 →
    initial_blue + red + black = taken_out + left →
    initial_blue = 4 := by
  intros
  sorry

end blue_water_bottles_initial_count_l19_19772


namespace believe_more_blue_l19_19074

-- Define the conditions
def total_people : ℕ := 150
def more_green : ℕ := 90
def both_more_green_and_more_blue : ℕ := 40
def neither : ℕ := 20

-- Theorem statement: Prove that the number of people who believe teal is "more blue" is 80
theorem believe_more_blue : 
  total_people - neither - (more_green - both_more_green_and_more_blue) = 80 :=
by
  sorry

end believe_more_blue_l19_19074


namespace ratio_of_sums_l19_19690

-- Define the arithmetic sequence and the condition
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables {a₁ a₂ d : ℝ}

-- Define the initial conditions and properties
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a₁ + n * d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

def condition (a : ℕ → ℝ) : Prop :=
  (a 6) / (a 3) = 7 / 13

-- The main theorem statement
theorem ratio_of_sums (a: ℕ → ℝ) (S: ℕ → ℝ) (h_seq : arithmetic_sequence a) 
  (h_sum_seq : sum_arithmetic_sequence S a) (h_cond: condition a) :
  (S 13) / (S 7) = 1 :=
sorry

end ratio_of_sums_l19_19690


namespace distance_between_parallel_lines_l19_19345

theorem distance_between_parallel_lines (x y : ℝ) : 
  ( ∀ x y, x + y - 1 = 0 → 2 * x + 2 * y + 1 ≠ 0 ) → 
  ∃ d : ℝ, 
    d = abs (2 * 1 + 2 * 0 + 1) / (real.sqrt (2^2 + 2^2)) ∧
    d = (3 * real.sqrt 2) / 4 :=
by 
  intro h
  use (3 * real.sqrt 2) / 4
  split
  { 
    field_simp,
    ring, 
  }
  { 
    sorry 
  }

end distance_between_parallel_lines_l19_19345


namespace water_in_tank_after_25_days_l19_19101

theorem water_in_tank_after_25_days (initial_water : ℕ) (evaporation_rate : ℕ) (days1 : ℕ) (added_water : ℕ) (days2 : ℕ) :
  initial_water = 500 →
  evaporation_rate = 2 →
  days1 = 15 →
  added_water = 100 →
  days2 = 25 →
  initial_water - (evaporation_rate * days1) + added_water - (evaporation_rate * days2) = 520 := 
by intros initial_water_eq evaporation_rate_eq days1_eq added_water_eq days2_eq
   rw [initial_water_eq, evaporation_rate_eq, days1_eq, added_water_eq, days2_eq]
   simp
   rfl  -- Resulting computation will affirm that the final water amount is 520 liters

end water_in_tank_after_25_days_l19_19101


namespace complement_of_A_l19_19161

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (C_UA : Set ℕ) :
  U = {2, 3, 4} →
  A = {x | (x - 1) * (x - 4) < 0 ∧ x ∈ Set.univ} →
  C_UA = {x ∈ U | x ∉ A} →
  C_UA = {4} :=
by
  intros hU hA hCUA
  -- proof omitted, sorry placeholder
  sorry

end complement_of_A_l19_19161


namespace arseniy_can_cut_two_matching_squares_l19_19751

-- Define the problem conditions in Lean 4
def grid : Type := Fin 8 → Fin 8 → Bool -- representing an 8 x 8 grid with 2 colors

-- The main theorem to prove
theorem arseniy_can_cut_two_matching_squares (G : grid) : 
  ∃ (s1 s2 : Fin 6 × Fin 6), s1 ≠ s2 ∧ 
  let p1 := (λ i j, G (s1.1 + i) (s1.2 + j)),
      p2 := (λ i j, G (s2.1 + i) (s2.2 + j)) in
  p1 = p2 :=
sorry

end arseniy_can_cut_two_matching_squares_l19_19751


namespace gcd_squares_example_l19_19043

noncomputable def gcd_of_squares : ℕ :=
  Nat.gcd (101 ^ 2 + 202 ^ 2 + 303 ^ 2) (100 ^ 2 + 201 ^ 2 + 304 ^ 2)

theorem gcd_squares_example : gcd_of_squares = 3 :=
by
  sorry

end gcd_squares_example_l19_19043


namespace shelves_used_l19_19417

def initial_books : ℕ := 86
def books_sold : ℕ := 37
def books_per_shelf : ℕ := 7
def remaining_books : ℕ := initial_books - books_sold
def shelves : ℕ := remaining_books / books_per_shelf

theorem shelves_used : shelves = 7 := by
  -- proof will go here
  sorry

end shelves_used_l19_19417


namespace cone_volume_divided_by_pi_l19_19445

noncomputable def volume_of_cone_divided_by_pi (r : ℝ) (angle : ℝ) : ℝ :=
  if angle = 270 ∧ r = 20 then
    let base_circumference := 30 * Real.pi in
    let base_radius := 15 in
    let slant_height := r in
    let height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
    let volume := (1 / 3) * Real.pi * base_radius ^ 2 * height in
    volume / Real.pi
  else 0

theorem cone_volume_divided_by_pi : 
  volume_of_cone_divided_by_pi 20 270 = 375 * Real.sqrt 7 :=
by
  sorry

end cone_volume_divided_by_pi_l19_19445


namespace probability_of_k_balls_in_ith_box_and_l_balls_in_first_i_minus_1_boxes_l19_19019

noncomputable def event_probability (n r i k l : ℕ) [fact (k + l ≤ r)] : ℚ :=
  if i > 0 ∧ i ≤ n then
    (nat.choose r k * nat.choose (r - k) l * (i - 1)^l * (n - i)^(r - k - l) : ℚ)
    / (n^r : ℚ)
  else 0

theorem probability_of_k_balls_in_ith_box_and_l_balls_in_first_i_minus_1_boxes
  (n r i k l : ℕ) (h : k + l ≤ r)
  (P : ℚ) :
  P = event_probability n r i k l :=
sorry

end probability_of_k_balls_in_ith_box_and_l_balls_in_first_i_minus_1_boxes_l19_19019


namespace negation_proposition_l19_19763

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) :=
by 
  sorry

end negation_proposition_l19_19763


namespace cos_two_x_zero_l19_19585

theorem cos_two_x_zero (x_0 : ℝ) (h : sin x_0 - 2 * cos x_0 = 0) : cos (2 * x_0) = -3 / 5 :=
sorry

end cos_two_x_zero_l19_19585


namespace algebraic_expression_l19_19128

def ast (n : ℕ) : ℕ := sorry

axiom condition_1 : ast 1 = 1
axiom condition_2 : ∀ (n : ℕ), ast (n + 1) = 3 * ast n

theorem algebraic_expression (n : ℕ) :
  n > 0 → ast n = 3^(n - 1) :=
by
  -- Proof to be completed
  sorry

end algebraic_expression_l19_19128


namespace sum_of_x_and_y_l19_19356

theorem sum_of_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
    (hx : ∃ (a : ℕ), 720 * x = a^2)
    (hy : ∃ (b : ℕ), 720 * y = b^4) :
    x + y = 1130 :=
sorry

end sum_of_x_and_y_l19_19356


namespace ratio_area_of_doubled_square_l19_19464

theorem ratio_area_of_doubled_square (s : ℝ) : 
  let original_area := s^2 in
  let enlarged_area := (2 * s)^2 in
  original_area / enlarged_area = 1 / 4 :=
by
  let original_area := s^2
  let enlarged_area := (2 * s)^2
  have h : original_area / enlarged_area = (s^2) / (4 * s^2) := by rfl
  have h1 : (s^2) / (4 * s^2) = 1 / 4 := by
    field_simp [ne_of_gt (sq_pos_of_pos (2 * s))]
    exact eq.div_self (four_ne_zero)
  exact eq.trans h h1

end ratio_area_of_doubled_square_l19_19464


namespace mixed_fruit_litres_opened_l19_19761

def superfruit_cost_per_litre : ℝ := 1399.45
def mixed_fruit_cost_per_litre : ℝ := 262.85
def acai_berry_cost_per_litre : ℝ := 3104.35
def acai_berry_litres : ℝ := 24.666666666666668

theorem mixed_fruit_litres_opened :
  let total_cost := mixed_fruit_cost_per_litre * x + acai_berry_cost_per_litre * acai_berry_litres
  let expected_total_cost := superfruit_cost_per_litre * (x + acai_berry_litres)
  total_cost = expected_total_cost → x ≈ 37.02 :=
begin
  sorry
end

end mixed_fruit_litres_opened_l19_19761


namespace perfect_square_of_sequence_l19_19205

def a : ℕ → ℤ 
| 1       := 1
| 2       := -1
| (n + 3) := -a (n + 2) - 2 * a (n + 1)

theorem perfect_square_of_sequence (n : ℕ) : 
  ∃ k : ℤ, 2 ^ (n + 2) - 7 * (a n) ^ 2 = k ^ 2 :=
sorry

end perfect_square_of_sequence_l19_19205


namespace skiing_speed_at_t_3_l19_19236

def l (t : ℝ) : ℝ := 2 * t^2 + 1.5 * t

theorem skiing_speed_at_t_3 :
  (derivative l 3) = 13.5 := by
  sorry

end skiing_speed_at_t_3_l19_19236


namespace find_other_number_l19_19824

-- Definitions based on the given conditions
def lcm (a b : ℕ) : ℕ := (a * b) / (gcd a b)
def hcf (a b : ℕ) : ℕ := gcd a b

-- Given conditions
axiom lcm_eq : lcm 44530 B = 9699690
axiom hcf_eq : hcf 44530 B = 385

-- Statement to prove the other number B given the conditions
theorem find_other_number : B = 83891 := by
  sorry

end find_other_number_l19_19824


namespace euro_share_and_change_l19_19523

theorem euro_share_and_change (initial_funds total_funds : ℝ) 
  (usd_funds : ℝ) (other_currencies : list ℝ) 
  (initial_euro_share : ℝ) (expected_euro_share : ℝ) 
  (expected_change : ℝ) :
  ∀ (f : ℝ), 
  f = initial_funds - usd_funds - other_currencies.sum →
  (f / total_funds) * 100 = expected_euro_share →
  expected_euro_share - initial_euro_share = expected_change →
  expected_euro_share = 4.37 ∧ expected_change = -38 :=
by
  intro f h1 h2 h3
  sorry

end euro_share_and_change_l19_19523


namespace number_of_adults_had_meal_l19_19846

theorem number_of_adults_had_meal (A : ℝ) :
  let num_children_food : ℝ := 63
  let food_for_adults : ℝ := 70
  let food_for_children : ℝ := 90
  (food_for_children - A * (food_for_children / food_for_adults) = num_children_food) →
  A = 21 :=
by
  intros num_children_food food_for_adults food_for_children h
  have h2 : 90 - A * (90 / 70) = 63 := h
  sorry

end number_of_adults_had_meal_l19_19846


namespace total_marbles_proof_l19_19515

def dan_violet_marbles : Nat := 64
def mary_red_marbles : Nat := 14
def john_blue_marbles (x : Nat) : Nat := x

def total_marble (x : Nat) : Nat := dan_violet_marbles + mary_red_marbles + john_blue_marbles x

theorem total_marbles_proof (x : Nat) : total_marble x = 78 + x := by
  sorry

end total_marbles_proof_l19_19515


namespace average_of_remaining_two_numbers_l19_19748

theorem average_of_remaining_two_numbers (S₆ avg₆ avg₂₁ avg₂₂ : ℚ)
    (h₁ : avg₆ = 2.80)
    (h₂ : avg₂₁ = 2.4)
    (h₃ : avg₂₂ = 2.3)
    (h₄ : S₆ = 6 * avg₆)
    (h₅ : ∑ i in (finset.range 2), avg₂₁ = 2 * avg₂₁)
    (h₆ : ∑ i in (finset.range 2), avg₂₂ = 2 * avg₂₂) :
  (S₆ - (2 * avg₂₁ + 2 * avg₂₂)) / 2 = 3.7 :=
by
  sorry

end average_of_remaining_two_numbers_l19_19748


namespace triangle_sin_A_l19_19640

theorem triangle_sin_A (A B C : ℝ) (p : ℝ) (hB : ∠B = p) (hC : ∠C = p) : sin (180 - 2 * p) = 1 / 2 := 
by 
  sorry

end triangle_sin_A_l19_19640


namespace general_term_geometric_sequence_sum_l19_19986

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

// Given conditions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) :=
  (a 1 = a1) ∧ (∀ n, a (n + 1) = a n + d) 

def conditions_for_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) :=
  a 3 * a 4 = 48 ∧ a 3 + a 4 = 14 ∧ d > 0

// General term proof
theorem general_term (a : ℕ → ℝ) (d a1 : ℝ) :
  conditions_for_arithmetic_sequence a d a1 →
  ∀ n, a n = 2 * n :=
sorry

// Sum of geometric sequence b_n
theorem geometric_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, b n = 2 ^ n) →
  ∀ n, T n = 2 ^ (n + 1) - 2 :=
sorry

end general_term_geometric_sequence_sum_l19_19986


namespace gerald_toy_cars_l19_19564

theorem gerald_toy_cars :
  let initial_cars := 20
  let donated_fraction := 2 / 5
  let new_cars := 10
  let donated_cars := (donated_fraction * initial_cars).toNat
  let remaining_cars := initial_cars - donated_cars
  remaining_cars + new_cars = 22 :=
by
  -- Definitions
  let initial_cars := 20
  let donated_fraction := 2 / 5
  let new_cars := 10
  let donated_cars := (donated_fraction * initial_cars).toNat
  let remaining_cars := initial_cars - donated_cars
  
  -- Proof
  have h_donated_cars : donated_cars = 8 := by ... -- Calculation: 2/5 * 20 = 8
  have h_remaining_cars : remaining_cars = 12 := by ... -- Calculation: 20 - 8 = 12
  have h_final_cars : remaining_cars + new_cars = 22 := by ... -- Calculation: 12 + 10 = 22
  
  exact h_final_cars -- Assert the final number of cars is 22
  sorry -- the actual calculation proof

end gerald_toy_cars_l19_19564


namespace cost_of_15_brown_socks_is_3_dollars_l19_19376

def price_of_brown_sock (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) : ℚ :=
  (price_white_socks - price_white_more_than_brown) / 2

def cost_of_15_brown_socks (price_brown_sock : ℚ) : ℚ :=
  15 * price_brown_sock

theorem cost_of_15_brown_socks_is_3_dollars
  (price_white_socks : ℚ) (price_white_more_than_brown : ℚ) 
  (h1 : price_white_socks = 0.45) (h2 : price_white_more_than_brown = 0.25) :
  cost_of_15_brown_socks (price_of_brown_sock price_white_socks price_white_more_than_brown) = 3 := 
by
  sorry

end cost_of_15_brown_socks_is_3_dollars_l19_19376


namespace cone_volume_270_degree_sector_l19_19439

noncomputable def coneVolumeDividedByPi (R θ: ℝ) (r h: ℝ) (circumf sector_height: ℝ) : ℝ := 
  if R = 20 
  ∧ θ = 270 / 360 
  ∧ 2 * Mathlib.pi * 20 = 40 * Mathlib.pi 
  ∧ circumf = 30 * Mathlib.pi
  ∧ 2 * Mathlib.pi * r = circumf
  ∧ r = 15
  ∧ sector_height = R
  ∧ r^2 + h^2 = sector_height^2 
  then (1/3) * Mathlib.pi * r^2 * h / Mathlib.pi 
  else 0

theorem cone_volume_270_degree_sector : coneVolumeDividedByPi 20 (270 / 360) 15 (5 * Real.sqrt 7) (30 * Mathlib.pi) 20 = 1125 * Real.sqrt 7 := 
by {
  -- This is where the proof would go
  sorry
}

end cone_volume_270_degree_sector_l19_19439


namespace binomial_coeff_ratio_l19_19000

open Nat

theorem binomial_coeff_ratio (n k : ℕ) (h1 : choose n k * 3 = choose n (k + 1))
                             (h2 : choose n (k + 1) * 2 = choose n (k + 2)) :
                             n + k = 13 := by
  sorry

end binomial_coeff_ratio_l19_19000


namespace three_hundredth_term_l19_19039

-- Let's define necessary sequences and conditions first:

/-- Define the sequence of positive integers omitting perfect squares -/
def not_perfect_square (n : ℕ) : Prop := ∀ m : ℕ, m^2 ≠ n

/-- Define the sequence of positive integers omitting multiples of 3 -/
def not_multiple_of_3 (n : ℕ) : Prop := n % 3 ≠ 0

/-- Define the sequence formed by omitting both perfect squares and multiples of 3 -/
def filtered_seq (n : ℕ) : Prop := not_perfect_square n ∧ not_multiple_of_3 n

/-- Define a sequence index function, where index k means the k-th number in the sequence -/
def sequence_index (k : ℕ) : ℕ :=
  if h : k > 0 then
    Nat.find_greatest (λ n, (filtered_seq n) ∧ (Card {m : ℕ | filtered_seq m ∧ m ≤ n}) = k) k
  else
    0  -- this won't be used since k > 0

/-- The proof statement -/
theorem three_hundredth_term : sequence_index 300 = 450 :=
  sorry

end three_hundredth_term_l19_19039


namespace simplified_expression_evaluation_l19_19331

-- Problem and conditions
def x := Real.sqrt 5 - 1

-- Statement of the proof problem
theorem simplified_expression_evaluation : 
  ( (x / (x - 1) - 1) / (x^2 - 1) / (x^2 - 2 * x + 1) ) = Real.sqrt 5 / 5 :=
sorry

end simplified_expression_evaluation_l19_19331


namespace girls_in_class_l19_19359

theorem girls_in_class (total_students boys_ratio girls_ratio : ℕ) (h_total : total_students = 260)
  (h_ratio_boys : boys_ratio = 5) (h_ratio_girls : girls_ratio = 8) : 
  let total_ratio := boys_ratio + girls_ratio in
  let boys_fraction := boys_ratio / total_ratio in
  let boys := total_students * boys_fraction in
  let girls := total_students - boys in
  girls = 160 := 
by 
  sorry

end girls_in_class_l19_19359


namespace max_y_coordinate_l19_19951

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l19_19951


namespace round_to_nearest_hundredth_l19_19730

theorem round_to_nearest_hundredth (x : ℝ) (digits : ℕ) (ht : x = 24.6374) (ht_digits : digits = 2) :
  Real.round_to x digits = 24.64 :=
by
  sorry

end round_to_nearest_hundredth_l19_19730


namespace consecutive_seating_probability_l19_19371

theorem consecutive_seating_probability :
  let total_ways := Nat.factorial 11
  let favorable_ways := 
    choose 12 5 * Nat.factorial 4 * 
    choose 7 4 * Nat.factorial 3 * 
    Nat.factorial 3 
  favorable_ways / total_ways = 1 / 10 := 
by
  sorry

end consecutive_seating_probability_l19_19371


namespace find_f_neg2_l19_19974

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a * f b
axiom f_pos (x : ℝ) : f x > 0
axiom f_one : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 :=
by {
  sorry
}

end find_f_neg2_l19_19974


namespace girls_in_class_l19_19360

theorem girls_in_class (total_students boys_ratio girls_ratio : ℕ) (h_total : total_students = 260)
  (h_ratio_boys : boys_ratio = 5) (h_ratio_girls : girls_ratio = 8) : 
  let total_ratio := boys_ratio + girls_ratio in
  let boys_fraction := boys_ratio / total_ratio in
  let boys := total_students * boys_fraction in
  let girls := total_students - boys in
  girls = 160 := 
by 
  sorry

end girls_in_class_l19_19360


namespace bank_account_balance_l19_19529

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l19_19529


namespace abc_unique_l19_19284

theorem abc_unique (n : ℕ) (hn : 0 < n) (p : ℕ) (hp : Nat.Prime p) 
                   (a b c : ℤ) 
                   (h : a^n + p * b = b^n + p * c ∧ b^n + p * c = c^n + p * a) 
                   : a = b ∧ b = c :=
by
  sorry

end abc_unique_l19_19284


namespace parallelogram_area_l19_19543

theorem parallelogram_area (a b : ℝ) (d₁ d₂ : ℝ) (h₁ : a = 51) (h₂ : d₁ = 40) (h₃ : d₂ = 74) : 
  let AO := d₁ / 2,
      OD := d₂ / 2,
      s := (a + AO + OD) / 2,
      area_triangle := Real.sqrt (s * (s - a) * (s - AO) * (s - OD)),
      area_parallelogram := 4 * area_triangle in
  area_parallelogram = 1224 := by
  sorry

end parallelogram_area_l19_19543


namespace largest_prime_factor_of_sum_of_divisors_of_200_l19_19274

-- Define the number 200
def n : ℕ := 200

-- Define the prime factorization of 200
def pfactors : (ℕ × ℕ) × (ℕ × ℕ) := ((2, 3), (5, 2))

-- Calculate the sum of the divisors of 200 using the prime factorization
def sumOfDivisors (n : ℕ) : ℕ :=
  let (p1, e1) := (pfactors.1.1, pfactors.1.2)
  let (p2, e2) := (pfactors.2.1, pfactors.2.2)
  (List.sum (List.range (e1 + 1))).map (λ i, p1 ^ i) * 
  (List.sum (List.range (e2 + 1)).map (λ i, p2 ^ i))
 
-- Define N to be the sum of divisors of 200
def N : ℕ := sumOfDivisors n

-- The statement to prove: The largest prime factor of N is 31
theorem largest_prime_factor_of_sum_of_divisors_of_200 : 
  Nat.prime 31 ∧ Nat.largestPrimeFactor N = 31 := 
  sorry

end largest_prime_factor_of_sum_of_divisors_of_200_l19_19274


namespace second_largest_example_second_largest_example_l19_19044
open List 

theorem second_largest_example :
  (secondLargest [5, 8, 4, 3, 7] = 7) := by
  sorry

-- We need additional definitions and helper lemmas to make it working code:

def secondLargest (l : List ℕ) : ℕ :=
  (l.erase (l.maximum? (by aesop))).maximum? (by aesop) |> Option.get_or_else 0

theorem second_largest_example :
  secondLargest [5, 8, 4, 3, 7] = 7 := by
  sorry

end second_largest_example_second_largest_example_l19_19044


namespace family_total_weight_gain_l19_19491

def orlando_gain : ℕ := 5
def jose_gain : ℕ := 2 * orlando_gain + 2
def fernando_gain : ℕ := (jose_gain / 2) - 3
def total_weight_gain : ℕ := orlando_gain + jose_gain + fernando_gain

theorem family_total_weight_gain : total_weight_gain = 20 := by
  -- proof omitted
  sorry

end family_total_weight_gain_l19_19491


namespace minimize_q_neg_1_l19_19098

noncomputable def respectful_polynomial (a b : ℝ) : (ℝ → ℝ) := λ x : ℝ, x^2 + a * x + b

theorem minimize_q_neg_1 (a b : ℝ) (h1 : b = -a - 1) (h2 : ∀ x, ((respectful_polynomial a b) ((respectful_polynomial a b) x) = 0) → (x^2 + (2 * a) * x + (a^2 + 2 * b + 1) * x + (2 * a * b + a) + (b^2 + a * b + b) = 0)) :
  (respectful_polynomial a b) (-1) = 0 :=
by {
  sorry
}

end minimize_q_neg_1_l19_19098


namespace minimum_value_of_polynomial_l19_19544

-- Define the polynomial expression
def polynomial_expr (x : ℝ) : ℝ := (8 - x) * (6 - x) * (8 + x) * (6 + x)

-- State the theorem with the minimum value
theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial_expr x = -196 := by
  sorry

end minimum_value_of_polynomial_l19_19544


namespace radius_inscribed_circle_ABC_l19_19920

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_inscribed_circle_ABC (hAB : AB = 18) (hAC : AC = 18) (hBC : BC = 24) :
  radius_of_inscribed_circle 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end radius_inscribed_circle_ABC_l19_19920


namespace min_distance_from_point_M_to_line_l19_19582

-- Define the circle and the line as functions
def circle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 4
def line (x y : ℝ) : Prop := 4 * x + 3 * y - 4 = 0

-- Calculate the distance from a point to a line
def point_to_line_dist (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

-- Define the center of the circle
def center := (5 : ℝ, 3 : ℝ)

-- Calculate the radius of the circle
def radius := (2 : ℝ)

-- Define the minimum distance from any point on the circle to the line
def min_distance_from_circle_to_line : ℝ :=
  point_to_line_dist (center.1) (center.2) 4 3 (-4) - radius

theorem min_distance_from_point_M_to_line :
  ∀ M : ℝ × ℝ, circle M.1 M.2 → min_distance_from_circle_to_line = 3 :=
by
  sorry

end min_distance_from_point_M_to_line_l19_19582


namespace standard_circle_eq_l19_19551

noncomputable def circle_equation : String :=
  "The standard equation of the circle whose center lies on the line y = -4x and is tangent to the line x + y - 1 = 0 at point P(3, -2) is (x - 1)^2 + (y + 4)^2 = 8"

theorem standard_circle_eq
  (center_x : ℝ)
  (center_y : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (point : ℝ × ℝ)
  (eqn_line : ∀ x y, tangent_line x y ↔ x + y - 1 = 0)
  (center_on_line : ∀ x y, y = -4 * x → center_y = y)
  (point_on_tangent : point = (3, -2))
  (tangent_at_point : tangent_line (point.1) (point.2)) :
  (center_x = 1 ∧ center_y = -4 ∧ (∃ r : ℝ, r = 2 * Real.sqrt 2)) →
  (∀ x y, (x - 1)^2 + (y + 4)^2 = 8) := by
  sorry

end standard_circle_eq_l19_19551


namespace angle_sum_around_point_l19_19252

theorem angle_sum_around_point (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) : 
    x + y + 130 = 360 → x + y = 230 := by
  sorry

end angle_sum_around_point_l19_19252


namespace money_made_is_40_l19_19494

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l19_19494


namespace b_investment_l19_19408

noncomputable def B_share := 880
noncomputable def A_share := 560
noncomputable def A_investment := 7000
noncomputable def C_investment := 18000
noncomputable def total_investment (B: ℝ) := A_investment + B + C_investment

theorem b_investment (B : ℝ) (P : ℝ)
    (h1 : 7000 / total_investment B * P = A_share)
    (h2 : B / total_investment B * P = B_share) : B = 8000 :=
by
  sorry

end b_investment_l19_19408


namespace emma_bank_account_balance_l19_19532

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l19_19532


namespace find_t_l19_19568

open Real

def vec2 (x y : ℝ) : (ℝ × ℝ) := (x, y)

def dotProduct (u v : (ℝ × ℝ)) : ℝ :=
u.1 * v.1 + u.2 * v.2

def magnitude (u : (ℝ × ℝ)) : ℝ :=
Real.sqrt (u.1^2 + u.2^2)

def projection (u v : (ℝ × ℝ)) : ℝ :=
(dotProduct u v) / (magnitude u)

theorem find_t :
  let a := vec2 3 (-4)
  let b := vec2 3 t
  projection a b = -3 → t = 6 := 
by 
  intros
  sorry

end find_t_l19_19568


namespace ellipse_eq_1_ellipse_eq_2_l19_19150

/-- To prove the standard equation of the ellipse with given foci and point passes through. -/
theorem ellipse_eq_1 (focal_distance : ℝ) (a_squared : ℝ) (b_squared : ℝ) :
  (focal_distance = 4) →
  (3^2 / a_squared + (-2*sqrt 6)^2 / b_squared = 1) →
  (a_squared - b_squared = 4) →
  (a_squared = 36) ∧ (b_squared = 32) :=
by  
  sorry

/-- To prove the standard equation of the ellipse with given focal distance and eccentricity. -/
theorem ellipse_eq_2 (focal_distance : ℝ) (eccentricity : ℝ) (a_squared : ℝ) (b_squared : ℝ) :
  (focal_distance = 8) →
  (eccentricity = 0.8) →
  ((a_squared, b_squared) = (25, 9) ∨ (a_squared, b_squared) = (9, 25)) :=
by
  sorry

end ellipse_eq_1_ellipse_eq_2_l19_19150


namespace cameron_answers_l19_19891

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end cameron_answers_l19_19891


namespace equal_values_of_means_l19_19113

theorem equal_values_of_means (f : ℤ × ℤ → ℤ) 
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ p, f p = (f (p.1 + 1, p.2) + f (p.1 - 1, p.2) + f (p.1, p.2 + 1) + f (p.1, p.2 - 1)) / 4):
  ∃ m : ℤ, ∀ p, f p = m := sorry

end equal_values_of_means_l19_19113


namespace initial_amount_is_1875_l19_19138

-- Defining the conditions as given in the problem
def initial_amount : ℝ := sorry
def spent_on_clothes : ℝ := 250
def spent_on_food (remaining : ℝ) : ℝ := 0.35 * remaining
def spent_on_electronics (remaining : ℝ) : ℝ := 0.50 * remaining

-- Given conditions
axiom condition1 : initial_amount - spent_on_clothes = sorry
axiom condition2 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) = sorry
axiom condition3 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) - spent_on_electronics (initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes)) = 200

-- Prove that initial amount is $1875
theorem initial_amount_is_1875 : initial_amount = 1875 :=
sorry

end initial_amount_is_1875_l19_19138


namespace find_x_in_equation_l19_19065

theorem find_x_in_equation :
  ∃ x : ℝ, x / 18 * (x / 162) = 1 ∧ x = 54 :=
by
  sorry

end find_x_in_equation_l19_19065


namespace final_sign_is_minus_l19_19415

theorem final_sign_is_minus 
  (plus_count : ℕ) 
  (minus_count : ℕ) 
  (h_plus : plus_count = 2004) 
  (h_minus : minus_count = 2005) 
  (transform : (ℕ → ℕ → ℕ × ℕ) → Prop) :
  transform (fun plus minus =>
    if plus >= 2 then (plus - 1, minus)
    else if minus >= 2 then (plus, minus - 1)
    else if plus > 0 && minus > 0 then (plus - 1, minus - 1)
    else (0, 0)) →
  (plus_count = 0 ∧ minus_count = 1) := sorry

end final_sign_is_minus_l19_19415


namespace compute_a1_b1_l19_19102

def sequence_satisfies (a b : ℕ → ℝ) :=
  ∀ n, (a (n + 1) = 2 * a n - b n) ∧ (b (n + 1) = 2 * b n + a n)

def initial_condition (a b : ℕ → ℝ) :=
  (a 50 = 3) ∧ (b 50 = 5)

theorem compute_a1_b1 (a b : ℕ → ℝ) [sequence_satisfies a b] [initial_condition a b] :
  ∃ k : ℝ, a 1 + b 1 = k := sorry

end compute_a1_b1_l19_19102


namespace simplify_and_rationalize_l19_19319

theorem simplify_and_rationalize :
  (∀ (a b c d e f : ℝ), a = real.sqrt 3 → b = real.sqrt 4 → 
                        c = real.sqrt 5 → d = real.sqrt 6 → 
                        e = real.sqrt 8 → f = real.sqrt 9 → 
   ((a / b) * (c / d) * (e / f) = real.sqrt 15 / 9)) :=
by
  intros a b c d e f ha hb hc hd he hf,
  -- Starting with the original problem
  have h1 : a = real.sqrt 3 := ha,
  have h2 : b = real.sqrt 4 := hb,
  have h3 : c = real.sqrt 5 := hc,
  have h4 : d = real.sqrt 6 := hd,
  have h5 : e = real.sqrt 8 := he,
  have h6 : f = real.sqrt 9 := hf,
  -- result to be filled with proof steps
  sorry

end simplify_and_rationalize_l19_19319


namespace parabola_directrix_line_intersects_parabola_l19_19667

/-- 
  Given a parabola y^2 = 2 * p * x and directrix x = -1/2,
  prove that p = 1 
-/
theorem parabola_directrix (p : ℝ) (h : ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -1/2) : p = 1 :=
sorry

/-- 
  Given p = 1 and the line y = x + t (t ≠ 0),
  prove that the length of the line segment AB is 2√10
-/
theorem line_intersects_parabola (t : ℝ) (h₀ : t ≠ 0) :
  ∀ A B : ℝ × ℝ,
  (y^2 = 2 - y^2 = 2 * x) -- parabola equation
  (y = x + t) -- line equation
  (A.1 * B.1 + A.2 * B.2 = 0) -- OA ⊥ OB
  (distance A B = 2 * real.sqrt 10) :=
sorry

end parabola_directrix_line_intersects_parabola_l19_19667


namespace complex_number_solution_l19_19292

theorem complex_number_solution (z : ℂ) (h : complex.I * (z + 1) = 1 + 2 * complex.I) : z = 1 - complex.I := 
by 
  sorry

end complex_number_solution_l19_19292


namespace max_y_coordinate_l19_19949

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l19_19949


namespace consecutive_seating_probability_l19_19372

theorem consecutive_seating_probability :
  let total_ways := Nat.factorial 11
  let favorable_ways := 
    choose 12 5 * Nat.factorial 4 * 
    choose 7 4 * Nat.factorial 3 * 
    Nat.factorial 3 
  favorable_ways / total_ways = 1 / 10 := 
by
  sorry

end consecutive_seating_probability_l19_19372


namespace arc_RS_variation_l19_19510

theorem arc_RS_variation
  (DEF : Triangle)
  (isosceles : DEF.is_isosceles)
  (DE EF DF h r : ℝ)
  (altitude_eq_base : h = DF)
  (radius_half_altitude : r = h / 2)
  (circle_rolls_along_DF : ∀ P : Point, circle_tangent_at P rolls_along DF)
  (intersect_R : ∀ P : Point, R = circle.intersect(DE))
  (intersect_S : ∀ P : Point, S = circle.intersect(EF)) :
  ∃ θ : Set ℝ, θ = {x | 90 ≤ x ∧ x ≤ 180} :=
sorry

end arc_RS_variation_l19_19510


namespace cos_2C_of_triangle_conditions_l19_19255

theorem cos_2C_of_triangle_conditions (A B C : Type)
    [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
    (BC AC : ℝ)
    (hBC : BC = 8)
    (hAC : AC = 5)
    (S : ℝ)
    (h_area : S = 12)
    (h_area_formula : S = 1 / 2 * BC * AC * sin C) :
    cos (2 * C) = -(7/25) :=
by sorry

end cos_2C_of_triangle_conditions_l19_19255


namespace intersection_complement_l19_19295

def U : Set ℤ := Set.univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}
def CUM : Set ℤ := {x : ℤ | x ∉ M}

theorem intersection_complement :
  P ∩ CUM = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l19_19295


namespace T_shape_in_grid_l19_19159

noncomputable def T_shape_exists (grid_size : ℕ) (num_removed_dominoes : ℕ) : Prop :=
  let total_cells := grid_size * grid_size in
  let total_removed_cells := num_removed_dominoes * 2 in
  let remaining_cells := total_cells - total_removed_cells in
  grid_size = 100 ∧ num_removed_dominoes = 1950 → ∃ T_shape, T_shape ⊆ ℤ × ℤ ∧
    -- T_shape is a four-cell figure in the shape of a T
    (∃ a b, a ≠ b ∧ ∀ x y, (x, y) ∈ T_shape → x ∈ {a, a + 1, a - 1} ∧ y = b ∨ x = a ∧ y = b + 1 ∨ y = b - 1)

theorem T_shape_in_grid: T_shape_exists 100 1950 :=
by {
  -- proof goes here
  sorry
}

end T_shape_in_grid_l19_19159


namespace geometric_product_formula_l19_19509

theorem geometric_product_formula (b : ℕ → ℝ) (n : ℕ) (hpos : ∀ k, b k > 0) :
  ∏ i in Finset.range(n), b (i + 1) = (b 1 * b n)^(n / 2 : ℕ) :=
sorry

end geometric_product_formula_l19_19509


namespace sum_of_roots_of_equation_l19_19388

theorem sum_of_roots_of_equation : 
  (∀ x, 5 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ x1 x2, (5 = x1) ∧ (5 = x2) ∧ (x1 + x2 = 4)) := 
by
  sorry

end sum_of_roots_of_equation_l19_19388


namespace no_integer_solutions_for_sum_of_squares_l19_19927

theorem no_integer_solutions_for_sum_of_squares :
  ∀ a b c : ℤ, a^2 + b^2 + c^2 ≠ 20122012 := 
by sorry

end no_integer_solutions_for_sum_of_squares_l19_19927


namespace cosine_opposite_values_l19_19636

theorem cosine_opposite_values (θ : ℝ) (k : ℝ) (h_sin : sin θ = k) :
  cos θ = sqrt (1 - k^2) ∨ cos θ = -sqrt (1 - k^2) :=
by
  sorry

end cosine_opposite_values_l19_19636


namespace largest_mu_inequality_l19_19149

theorem largest_mu_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ (μ : ℝ), μ = (2 * real.sqrt 6) / 5 ∧ 
  (a^2 + b^2 + 2 * c^2 + 2 * d^2 >= μ * a * b + 2 * b * c + 2 * μ * c * d) := 
begin 
  use (2 * real.sqrt 6) / 5,
  sorry
end

end largest_mu_inequality_l19_19149


namespace number_of_valid_ks_l19_19524

theorem number_of_valid_ks : 
  (finset.card (finset.filter (λ k, (20 % k = 0)) (finset.range 21))) = 6 := 
by sorry

end number_of_valid_ks_l19_19524


namespace time_difference_between_joshua_and_malcolm_l19_19298

noncomputable def time_to_complete (speed_per_mile : ℝ) (distance : ℝ) : ℝ :=
  speed_per_mile * distance

theorem time_difference_between_joshua_and_malcolm :
  let malcolm_speed := 7
  let joshua_speed := 8
  let race_distance := 12
  let malcolm_time := time_to_complete malcolm_speed race_distance
  let joshua_time := time_to_complete joshua_speed race_distance
  joshua_time - malcolm_time = 12 :=
by
  let malcolm_speed := 7
  let joshua_speed := 8
  let race_distance := 12
  let malcolm_time := time_to_complete malcolm_speed race_distance
  let joshua_time := time_to_complete joshua_speed race_distance
  have h1: malcolm_time = malcolm_speed * race_distance := rfl
  have h2: joshua_time = joshua_speed * race_distance := rfl
  calc
    joshua_time - malcolm_time = (8 * 12) - (7 * 12) : by rw [h1, h2]
                       ...     = 96 - 84            : by norm_num
                       ...     = 12                 : by norm_num
  sorry

end time_difference_between_joshua_and_malcolm_l19_19298


namespace general_formula_a_n_sum_T_n_formula_l19_19602

-- Define the sequences and the conditions given in the problem
def seq_a (n : ℕ) := if n = 0 then 1/2 else (a_n : ℕ) := (1 / 2) ^ n
def sum_to_n (f : ℕ → ℝ) (n : ℕ) := ∑ i in range (n + 1), f i

theorem general_formula_a_n (n : ℕ) (hn : n > 0) :
  let a_n := (1 / 2) ^ n in
  sum_to_n (λ k, a_n) n = 1 - a_n :=
sorry

theorem sum_T_n_formula (n : ℕ) (hn : n > 0) :
  let a_n := (1 / 2) ^ n in
  let b_n := 2^n * a_n in
  let T_n := sum_to_n b_n n in
  T_n = (n - 1) * 2 ^ (n + 1) + 2 :=
sorry

end general_formula_a_n_sum_T_n_formula_l19_19602


namespace tobias_swimming_distance_l19_19030

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end tobias_swimming_distance_l19_19030


namespace find_number_l19_19840

variable (number x : ℝ)

theorem find_number (h1 : number * x = 1600) (h2 : x = -8) : number = -200 := by
  sorry

end find_number_l19_19840


namespace heartsuit_sum_l19_19556

def heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

theorem heartsuit_sum : heartsuit 1 + heartsuit 2 + heartsuit 4 = 33.666666666666664 := 
  by
  -- Proof can be filled in here
  sorry

end heartsuit_sum_l19_19556


namespace total_germs_l19_19668

theorem total_germs 
  (total_petri_dishes : ℝ) 
  (germs_per_dish : ℝ) 
  (h1 : total_petri_dishes = 18000 * 10^(-3)) 
  (h2 : germs_per_dish = 199.99999999999997) 
  : total_petri_dishes * germs_per_dish = 3600 :=
by
  sorry

end total_germs_l19_19668


namespace maximize_distance_difference_l19_19367

def line_eq (P : ℝ × ℝ) : Prop := 2 * P.1 - P.2 = 4

def dist (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

def A : ℝ × ℝ := (4, -1)
def B : ℝ × ℝ := (3, 4)
def P : ℝ × ℝ := (5, 6)

theorem maximize_distance_difference : 
  line_eq P ∧ ∀ Q : ℝ × ℝ, line_eq Q → dist Q A - dist Q B ≤ dist P A - dist P B :=
by
  sorry

end maximize_distance_difference_l19_19367


namespace evaluate_expression_l19_19143

theorem evaluate_expression : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end evaluate_expression_l19_19143


namespace simplify_and_rationalize_l19_19320

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 8 / sqrt 9) = sqrt 15 / 9 :=
by
  -- Solve the problem here.
  sorry

end simplify_and_rationalize_l19_19320


namespace f_neg_a_l19_19002

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_neg_a_l19_19002


namespace first_term_of_geometric_series_l19_19015

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l19_19015


namespace angle_sum_eq_l19_19983

structure Quadrilateral (α : Type _) :=
(E L M I : α)

variables {α : Type _} [EuclideanGeometry α]

-- Definition of the points E, L, M, I
variables (ELMI : Quadrilateral α)
variables {E L M I : α} [pointOnLine L M] [pointOnLine M I]

-- Condition 1: quadrilateral ELMI defined
def ELMI_exists : Prop := true

-- Condition 2: The sum of angles ∠LME and ∠MEI is 180 degrees
def sum_angles_eq_180 (L M E I : α) [IsAngle (LME : EUclideanAngle)] [IsAngle (MEI : EUclideanAngle)] : Prop :=
  ∠ LME + ∠ MEI = 180

-- Condition 3: EL = EI + LM
def length_rel (L M E I : α) : Prop :=
  dist E L = dist E I + dist L M

-- Proof statement: Prove that the sum of angles ∠LEM and ∠EMI is ∠MIE
theorem angle_sum_eq (α : Type _) [EuclideanGeometry α] (E L M I : α)
                      [IsAngle (LEM : EuclideanAngle)] [IsAngle (EMI : EuclideanAngle)] [IsAngle (MIE : EuclideanAngle)]
                      (sum_angles_180 : sum_angles_eq_180 L M E I) (length_cond : length_rel L M E I) :
  ∠ LEM + ∠ EMI = ∠ MIE := sorry

end angle_sum_eq_l19_19983


namespace percentage_primes_divisible_by_2_l19_19050

theorem percentage_primes_divisible_by_2 :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  (100 * primes.filter (fun n => n % 2 = 0).card / primes.card) = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  have h1 : primes.filter (fun n => n % 2 = 0).card = 1 := sorry
  have h2 : primes.card = 8 := sorry
  have h3 : (100 * 1 / 8 : ℝ) = 12.5 := by norm_num
  exact h3

end percentage_primes_divisible_by_2_l19_19050


namespace cameron_total_questions_answered_l19_19895

def questions_per_tourist : ℕ := 2
def group1_size : ℕ := 6
def group2_size : ℕ := 11
def group3_size_regular : ℕ := 7
def group3_inquisitive_size : ℕ := 1
def group4_size : ℕ := 7

theorem cameron_total_questions_answered :
  let group1_questions := questions_per_tourist * group1_size in
  let group2_questions := questions_per_tourist * group2_size in
  let group3_regular_questions := questions_per_tourist * group3_size_regular in
  let group3_inquisitive_questions := group3_inquisitive_size * (questions_per_tourist * 3) in
  let group3_questions := group3_regular_questions + group3_inquisitive_questions in
  let group4_questions := questions_per_tourist * group4_size in
  group1_questions + group2_questions + group3_questions + group4_questions = 68 :=
by
  sorry

end cameron_total_questions_answered_l19_19895


namespace sylvia_fraction_of_incorrect_answers_l19_19338

variable (questions : ℕ) (sergio_mistakes : ℕ) (sergio_more_correct_than_sylvia : ℕ)

theorem sylvia_fraction_of_incorrect_answers
  (h1 : questions = 50)
  (h2 : sergio_mistakes = 4)
  (h3 : sergio_more_correct_than_sylvia = 6) :
  let sylvia_incorrect := questions - (questions - sergio_mistakes - sergio_more_correct_than_sylvia)
  in sylvia_incorrect / questions = 1 / 5 := 
by
  sorry

end sylvia_fraction_of_incorrect_answers_l19_19338


namespace radius_of_inscribed_circle_l19_19745

theorem radius_of_inscribed_circle 
  (a b c h S r : ℝ)
  (h_iso : a + b = 2 * c)
  (h_height : h = (a + b) / 4)
  (h_area : S = (a + b) * h / 2)
  (h_radius : h = 2 * r)
  (angle_base : ∀ θ : ℝ, θ = 30 → sin θ = 1 / 2) :
  r = (Real.sqrt (2 * S)) / 4 :=
by
  -- Proof goes here
  sorry

end radius_of_inscribed_circle_l19_19745


namespace shaded_area_between_circles_l19_19782

theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5)
  (tangent : True) -- This represents that the circles are externally tangent
  (circumscribed : True) -- This represents the third circle circumscribing the two circles
  : ∃ r3 : ℝ, r3 = 9 ∧ π * r3^2 - (π * r1^2 + π * r2^2) = 40 * π :=
  sorry

end shaded_area_between_circles_l19_19782


namespace cameron_total_questions_answered_l19_19896

def questions_per_tourist : ℕ := 2
def group1_size : ℕ := 6
def group2_size : ℕ := 11
def group3_size_regular : ℕ := 7
def group3_inquisitive_size : ℕ := 1
def group4_size : ℕ := 7

theorem cameron_total_questions_answered :
  let group1_questions := questions_per_tourist * group1_size in
  let group2_questions := questions_per_tourist * group2_size in
  let group3_regular_questions := questions_per_tourist * group3_size_regular in
  let group3_inquisitive_questions := group3_inquisitive_size * (questions_per_tourist * 3) in
  let group3_questions := group3_regular_questions + group3_inquisitive_questions in
  let group4_questions := questions_per_tourist * group4_size in
  group1_questions + group2_questions + group3_questions + group4_questions = 68 :=
by
  sorry

end cameron_total_questions_answered_l19_19896


namespace geometric_series_first_term_l19_19014

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l19_19014


namespace only_odd_integer_option_l19_19399

theorem only_odd_integer_option : 
  (6 ^ 2 = 36 ∧ Even 36) ∧ 
  (23 - 17 = 6 ∧ Even 6) ∧ 
  (9 * 24 = 216 ∧ Even 216) ∧ 
  (96 / 8 = 12 ∧ Even 12) ∧ 
  (9 * 41 = 369 ∧ Odd 369)
:= by
  sorry

end only_odd_integer_option_l19_19399


namespace wholesale_price_l19_19459

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end wholesale_price_l19_19459


namespace max_y_coordinate_l19_19944

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l19_19944


namespace scientific_notation_of_21500000_l19_19639

theorem scientific_notation_of_21500000 :
  21500000 = 2.15 * 10^7 :=
by
  sorry

end scientific_notation_of_21500000_l19_19639


namespace elastic_ellipse_sum_l19_19282

noncomputable def h : ℝ := 7
noncomputable def k : ℝ := 2
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4

def F1 : ℝ × ℝ := (4, 2)
def F2 : ℝ × ℝ := (10, 2)

theorem elastic_ellipse_sum : 
  (PF₁ + PF₂ = 10) → 
  ∀ P : ℝ × ℝ, let Q := ((P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2) in
  (Q = 1) → (h + k + a + b = 18) := sorry

end elastic_ellipse_sum_l19_19282


namespace group_acquaintances_l19_19644

/-- In a group of 60 people who do not initially know each other, 
it is always possible for some of them to get to know each other 
such that in any subset of 3 people, the number of acquaintances 
they each know within this group is not all equal. 
-/
theorem group_acquaintances (n : ℕ) (h : n = 60) :
  ∃ (P : Finset (Fin 60)), ∀ (x y z : Fin 60), x ∈ P → y ∈ P → z ∈ P → 
  (finset.card (finset.filter (λ w, w ∈ P) (finset.range 60)) ≠ 
   finset.card (finset.filter (λ t, t ∈ P) (finset.range 60))) ∨
  (finset.card (finset.filter (λ u, u ∈ P) (finset.range 60)) ≠ 
   finset.card (finset.filter (λ v, v ∈ P) (finset.range 60))) :=
sorry

end group_acquaintances_l19_19644


namespace sum_of_angles_is_equal_l19_19981

variable {Point : Type}
variable {Angle : Type}
variable [AddGroup Angle] [CommGroup Angle] {deg : Angle → Lean.expr} {rad : Angle → Lean.expr}

variables (E L M I : Point)
variables (∠ : Point → Point → Point → Angle)
variable (deg180 : 180)

variables 
  (h1 : ∠ E L M + ∠ E M I = deg180)
  (h2 : dist E L = dist E I + dist L M)

theorem sum_of_angles_is_equal 
  (E L M I : Point)
  (∠ : Point → Point → Point → Angle)
  (deg180 : 180)
  (h1 : ∠ E L M + ∠ E M I = deg180)
  (h2 : dist E L = dist E I + dist L M) :
  ∠ L E M + ∠ E M I = ∠ M I E := 
sorry

end sum_of_angles_is_equal_l19_19981


namespace volleyball_substitutions_mod_1000_l19_19474

theorem volleyball_substitutions_mod_1000 :
  (let a0 := 1
       a1 := 6 * 13 * a0
       a2 := 6 * 12 * a1
       a3 := 6 * 11 * a2
       a4 := 6 * 10 * a3
       a5 := 6 * 9 * a4
       n := a0 + a1 + a2 + a3 + a4 + a5
   in n % 1000) = 271 := by
  sorry

end volleyball_substitutions_mod_1000_l19_19474


namespace probability_red_green_blue_l19_19773

theorem probability_red_green_blue :
  let S : Finset (Fin 12) := Finset.univ, -- set of 12 shoes
  let red := {i | i < 5}, -- red shoes are represented as the first 5 indices
  let green := {i | i < 9 ∧ i ≥ 5}, -- green shoes are next 4 indices
  let blue := {i | i ≥ 9}, -- blue shoes are last 3 indices
  (∀ (s₁ s₂ s₃ : Fin 12), s₁ ∈ red ∧ s₂ ∈ green ∧ s₃ ∈ blue → 
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ → 
    ((S.card / 12 : ℚ) * ((S.card - 1) / 11) * ((S.card - 2) / 10) = 1 / 22)
  sorry

end probability_red_green_blue_l19_19773


namespace right_triangle_hypotenuse_l19_19649

theorem right_triangle_hypotenuse (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 2500)
  (h2 : c - a = 10)
  (h3 : c^2 = a^2 + b^2) :
  c = 25 * sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l19_19649


namespace k_range_l19_19956

def k (x : ℝ) : ℝ := (2 * x + 7) / (x - 3)

theorem k_range : 
  set.range k = {y : ℝ | y ≠ 2} :=
sorry

end k_range_l19_19956


namespace connor_password_last_digit_l19_19121

def is_digit_4 (n : ℕ) : Prop := n % 10 = 4

def is_three_digit_square (n : ℕ) : Prop := 
  n = 64 ∨ n = 81 ∨ n = 100

def is_three_digit_even_square (n : ℕ) : Prop := 
  (n = 64 ∨ n = 100) ∧ n % 2 = 0

def is_tens_digit_6 (n : ℕ) : Prop :=
  (n / 10) % 10 = 6

def is_valid_password (n : ℕ) : Prop :=
  n ≥ 50 ∧ n ≤ 100 ∧ 
  (is_three_digit_square n ∧ is_three_digit_even_square n ∧ is_tens_digit_6 n)

theorem connor_password_last_digit (n : ℕ) :
  is_valid_password n → is_digit_4 n :=
begin
  sorry
end

end connor_password_last_digit_l19_19121


namespace mark_collect_money_l19_19299

theorem mark_collect_money (d h : ℕ) (hh : 20) (dd : 5) (f : ℕ) (hf : f = 2) (w : ℕ) (hw: w = 20) :
  (20 * 5 / 2) * (2 * 20) = 2000 :=
by
  sorry

end mark_collect_money_l19_19299


namespace nested_fraction_expression_l19_19389

theorem nested_fraction_expression : 
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := 
by sorry

end nested_fraction_expression_l19_19389


namespace length_of_train_is_500_l19_19471

noncomputable def speed_kmh_to_ms (v : ℕ) : ℕ :=
  v * 1000 / 3600

def length_of_train (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem length_of_train_is_500 (speed_kmh time : ℕ) (h_speed : speed_kmh = 90) (h_time : time = 20) :
  length_of_train (speed_kmh_to_ms speed_kmh) time = 500 :=
by
  have h1 : speed_kmh_to_ms 90 = 25 := by
    simp [speed_kmh_to_ms]
  rw [h_speed, h1, h_time]
  simp [length_of_train]
  sorry

end length_of_train_is_500_l19_19471


namespace hyperbola_asymptotes_angle_45_deg_l19_19153

theorem hyperbola_asymptotes_angle_45_deg (a b : ℝ) 
  (h : a > b) 
  (h' : ∀ θ, tan θ = b / a → θ = π / 4) :
  a / b = 1 :=
by sorry

end hyperbola_asymptotes_angle_45_deg_l19_19153


namespace proportion_x_l19_19816

theorem proportion_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
sorry

end proportion_x_l19_19816


namespace algebraic_expression_evaluation_l19_19583

theorem algebraic_expression_evaluation (a b : ℝ) (h : 1 / a + 1 / (2 * b) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1 / 2 := 
by
  sorry

end algebraic_expression_evaluation_l19_19583


namespace geometric_series_first_term_l19_19013

theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 18) 
  (h2 : a^2 / (1 - r^2) = 72) : 
  a = 7.2 :=
by
  sorry

end geometric_series_first_term_l19_19013


namespace fraction_div_add_result_l19_19620

theorem fraction_div_add_result : 
  (2 / 3) / (4 / 5) + (1 / 2) = (4 / 3) := 
by 
  sorry

end fraction_div_add_result_l19_19620


namespace maximum_angle_AFB_l19_19169

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.2 ^ 2 = 2 * p * xy.1}

variables {p x1 x2 y1 y2 : ℝ}
variables A B : ℝ × ℝ
variable F : ℝ × ℝ

def on_parabola (pt : ℝ × ℝ) (p : ℝ) : Prop :=
  pt.2 ^ 2 = 2 * p * pt.1

def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def angle_AFB (A B F : ℝ × ℝ) : ℝ := sorry -- This would require more auxiliary definitions

theorem maximum_angle_AFB : 
  on_parabola A p ∧
  on_parabola B p ∧
  (A.1 + B.1 + p = (2 * real.sqrt 3 / 3) * distance A B) →
  angle_AFB A B F = (2 * real.pi / 3) :=
sorry

end maximum_angle_AFB_l19_19169


namespace proof_problem_l19_19921

noncomputable def smallest_positive_integer (n : ℕ) (a b : ℝ) (h0 : 0 < a) (h1 : 0 < b) : Prop :=
  (a + 3 * b * Complex.I) ^ n = (a - 3 * b * Complex.I) ^ n

noncomputable def ratio_of_b_to_a (a b : ℝ) (h0 : 0 < a) (h1 : 0 < b) : ℝ :=
  b / a

theorem proof_problem :
  ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ smallest_positive_integer 3 a b ∧ ratio_of_b_to_a a b = Real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l19_19921


namespace triangle_area_l19_19674

open Real

theorem triangle_area
  (AB BC : ℝ)
  (BD : ℝ)
  (AC : ℝ)
  (F : ℝ)
  (h1 : AB = BC)
  (h2 : BD > 0)
  (h3 : F - 8 = 0)
  (h4 : ∃ (α β : ℝ), tan(α - β) * tan(α + β) = tan(α) * tan(α) ∧ 
       ((tan(α + β) / tan(α - β)) ^ 2 = 1 ∧ 
        β = arccos((1 + tan(α + β) * cot(α - β) - tan(α + β) / cot(α + β)) / 2)))
  (h5 : ∃ (β : ℝ), 1 - (BD / AC) * tan(π / 4 - β) = 2 - BD / AC)
  (h6 : BD / AC = 3) :
  area_ABC(AB BC BD AC F) = 32 / 3 := sorry

end triangle_area_l19_19674


namespace xy_y_sq_eq_y_sq_3y_12_l19_19622

variable (x y : ℝ)

theorem xy_y_sq_eq_y_sq_3y_12 (h : x * (x + y) = x^2 + 3 * y + 12) : 
  x * y + y^2 = y^2 + 3 * y + 12 := 
sorry

end xy_y_sq_eq_y_sq_3y_12_l19_19622


namespace marathon_time_l19_19091

theorem marathon_time
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_time : ℝ)
  (remaining_pace_factor : ℝ)
  (remaining_distance := total_distance - first_part_distance)
  (first_pace := first_part_distance / first_part_time)
  (remaining_pace := remaining_pace_factor * first_pace)
  (remaining_time := remaining_distance / remaining_pace)
  (total_time := first_part_time + remaining_time) :
  total_distance = 26 → first_part_distance = 10 → first_part_time = 1 →
  remaining_pace_factor = 0.8 → total_time = 3 :=
by
  intros h_total_distance h_first_part_distance h_first_part_time h_remaining_pace_factor
  simp [total_distance, first_part_distance, first_part_time, remaining_pace_factor] at *
  sorry

end marathon_time_l19_19091


namespace jessica_domino_arrangements_l19_19677

theorem jessica_domino_arrangements (n m : ℕ) (h₁ : n = 4) (h₂ : m = 5) :
  ∃ (k : ℕ), k = ((n + m - 2).choose (n - 1)) ∧ k = 35 :=
by
  have h : (n + m - 2).choose (n - 1) = 35, from sorry,
  use ((n + m - 2).choose (n - 1)),
  simp [h],
  exact h

end jessica_domino_arrangements_l19_19677


namespace factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l19_19540

-- Given condition and question, prove equality for the first expression
theorem factorize_x4_minus_16y4 (x y : ℝ) :
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by sorry

-- Given condition and question, prove equality for the second expression
theorem factorize_minus_2a3_plus_12a2_minus_16a (a : ℝ) :
  -2 * a^3 + 12 * a^2 - 16 * a = -2 * a * (a - 2) * (a - 4) := 
by sorry

end factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l19_19540


namespace binary_1011_is_11_decimal_124_is_174_l19_19125

-- Define the conversion from binary to decimal
def binaryToDecimal (n : Nat) : Nat :=
  (n % 10) * 2^0 + ((n / 10) % 10) * 2^1 + ((n / 100) % 10) * 2^2 + ((n / 1000) % 10) * 2^3

-- Define the conversion from decimal to octal through division and remainder
noncomputable def decimalToOctal (n : Nat) : String := 
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  (aux n []).foldr (fun d s => s ++ d.repr) ""

-- Prove that the binary number 1011 (base 2) equals the decimal number 11
theorem binary_1011_is_11 : binaryToDecimal 1011 = 11 := by
  sorry

-- Prove that the decimal number 124 equals the octal number 174 (base 8)
theorem decimal_124_is_174 : decimalToOctal 124 = "174" := by
  sorry

end binary_1011_is_11_decimal_124_is_174_l19_19125


namespace brick_height_l19_19836

theorem brick_height (length width : ℕ) (num_bricks : ℕ) (wall_length wall_width wall_height : ℕ) (h : ℕ) :
  length = 20 ∧ width = 10 ∧ num_bricks = 25000 ∧ wall_length = 2500 ∧ wall_width = 200 ∧ wall_height = 75 ∧
  ( 20 * 10 * h = (wall_length * wall_width * wall_height) / 25000 ) -> 
  h = 75 :=
by
  sorry

end brick_height_l19_19836


namespace solve_system_of_equations_l19_19334

theorem solve_system_of_equations :
    ∃ x y : ℝ, (x * Real.log 3 / Real.log 2 + y = Real.log 18 / Real.log 2) ∧ (5 ^ x = 25 ^ y) ∧ (x = 2) ∧ (y = 1) :=
by {
    have log2_3 := Real.log 3 / Real.log 2,
    have log2_18 := Real.log 18 / Real.log 2,
    have eq1 : x * log2_3 + y = log2_18 := sorry,
    have eq2 : 5 ^ x = 25 ^ y := sorry,
    use [2, 1],
    split,
    { exact eq1 },
    split,
    { exact eq2 },
    split,
    { refl },
    { refl },
    sorry
}

end solve_system_of_equations_l19_19334


namespace image_of_1_neg2_preimages_of_1_neg2_l19_19695

variable {A B: Type} [Field A] [Field B]

def f (p : A × A) : A × A :=
  (p.1 + p.2, p.1 * p.2)

theorem image_of_1_neg2 : f (1, -2) = (-1, -2) := 
  sorry

theorem preimages_of_1_neg2 (p : A × A) : 
  f p = (1, -2) ↔ p = (2, -1) ∨ p = (-1, 2) := 
  sorry

end image_of_1_neg2_preimages_of_1_neg2_l19_19695


namespace triangle_acute_angle_contradiction_l19_19377

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end triangle_acute_angle_contradiction_l19_19377


namespace s_formula_functional_eq_l19_19288

noncomputable def a : ℕ → ℕ
| 0       := 0
| (2 * n) := n
| (2 * n + 1) := n

noncomputable def s (n : ℕ) : ℕ := (List.range (n + 1)).map a).sum

theorem s_formula (n : ℕ) : s(n) = (n^2) / 4 := by
  sorry

theorem functional_eq (m n : ℕ) (h : m > n) : s(m + n) = m * n + s(m - n) := by
  sorry

end s_formula_functional_eq_l19_19288


namespace decimal_to_fraction_l19_19792

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l19_19792


namespace find_k_l19_19698

theorem find_k (k : ℝ) (h1 : k > 1) (h2 : ∑' n : ℕ in (List.range (n+1)).at (n.sum 1), (7 * n - 2) / k ^ n = 3) : 
  k = (21 + Real.sqrt 477) / 6 := by
{
  sorry
}

end find_k_l19_19698


namespace positive_root_exists_for_all_permutations_l19_19608

theorem positive_root_exists_for_all_permutations 
  (a b c : ℝ) 
  (h1 : 0 ≠ a) (h2 : 0 ≠ b) (h3 : 0 ≠ c) 
  (h_real_root : ∀ (p : Fin 3 → ℝ),
    (p !0 * x^2 + p !1 * x + p !2 = 0) → 
    (∃ x : ℝ, p !0 * x^2 + p !1 * x + p !2 = 0)) :
  ∀ (p : Fin 3 → ℝ), 
    (p !0 * x^2 + p !1 * x + p !2 = 0) →
    (∃ x : ℝ, 0 < x ∧ p !0 * x^2 + p !1 * x + p !2 = 0) := 
sorry

end positive_root_exists_for_all_permutations_l19_19608


namespace nice_numbers_sum_first_ten_l19_19496

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ n = ∏ i in (finset.filter (is_proper_divisor n) (finset.range n)), i

def first_ten_nice_numbers : list ℕ :=
  [8, 27, 125, 343, 6, 10, 14, 15, 21, 35]

def sum_of_first_ten_nice_numbers : ℕ :=
  list.sum first_ten_nice_numbers

theorem nice_numbers_sum_first_ten : sum_of_first_ten_nice_numbers = 604 := 
by 
  rw [sum_of_first_ten_nice_numbers] 
  sorry -- The proof is omitted according to the problem requirements.

end nice_numbers_sum_first_ten_l19_19496


namespace evaluate_expression_solve_fractional_equation_l19_19830

section problem1

def expression := sqrt 8 + (1 / 2)⁻¹ - 2 * real.sin (real.pi / 4) - abs (1 - sqrt 2)

theorem evaluate_expression : expression = 3 := by
  sorry

end problem1

section problem2

variable {x : ℝ}

theorem solve_fractional_equation (h_eq : (1 - x) / (x - 3) = 1 / (3 - x) - 2) : x = -4 :=
by 
  sorry

end problem2

end evaluate_expression_solve_fractional_equation_l19_19830


namespace arithmetic_sequence_sum_l19_19807

theorem arithmetic_sequence_sum :
  let a₁ := -5
  let aₙ := 40
  let n := 10
  (n : ℝ) = 10 →
  (a₁ : ℝ) = -5 →
  (aₙ : ℝ) = 40 →
  ∑ i in finset.range n, (a₁ + i * ((aₙ - a₁) / (n - 1))) = 175 :=
by
  intros
  sorry

end arithmetic_sequence_sum_l19_19807


namespace simplify_and_rationalize_l19_19321

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 8 / sqrt 9) = sqrt 15 / 9 :=
by
  -- Solve the problem here.
  sorry

end simplify_and_rationalize_l19_19321


namespace proper_fraction_decomposition_l19_19111

theorem proper_fraction_decomposition
  (n : ℕ)
  (c : ℤ)
  (m : Fin n → ℤ)
  (h_coprime : ∀ i j, i ≠ j → Nat.coprime (m i) (m j))
  (h_nonzero : ∀ i, m i ≠ 0):
  ∃ n_coeff : Fin n → ℤ, (c : ℚ) / m.prod → ∑ i, (n_coeff i : ℚ) / m i := 
by
  sorry

end proper_fraction_decomposition_l19_19111


namespace book_pages_count_l19_19296

theorem book_pages_count :
  (∀ n : ℕ, n = 4 → 42 * n = 168) ∧
  (∀ n : ℕ, n = 2 → 50 * n = 100) ∧
  (∀ p1 p2 : ℕ, p1 = 168 ∧ p2 = 100 → p1 + p2 = 268) ∧
  (∀ p : ℕ, p = 268 → p + 30 = 298) →
  298 = 298 := by
  sorry

end book_pages_count_l19_19296


namespace range_f_l19_19971

open Real

noncomputable def f (x: ℝ) : ℝ := sin x - cos x

theorem range_f (x : ℝ) (hx : x ∈ Icc (pi / 2) (3 * pi / 4)) : 
  (f x) ∈ Icc 0 (sqrt 2) := 
sorry

end range_f_l19_19971


namespace man_rate_in_still_water_l19_19061

theorem man_rate_in_still_water (Vm Vs : ℝ) :
  Vm + Vs = 20 ∧ Vm - Vs = 8 → Vm = 14 :=
by
  sorry

end man_rate_in_still_water_l19_19061


namespace no_prime_pairs_sum_53_l19_19664

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l19_19664


namespace circle_radius_tangents_l19_19882

theorem circle_radius_tangents
  (AB CD EF r : ℝ)
  (circle_tangent_AB : AB = 5)
  (circle_tangent_CD : CD = 11)
  (circle_tangent_EF : EF = 15) :
  r = 2.5 := by
  sorry

end circle_radius_tangents_l19_19882


namespace plates_not_adj_l19_19853

def num_ways_arrange_plates (blue red green orange : ℕ) (no_adj : Bool) : ℕ :=
  -- assuming this function calculates the desired number of arrangements
  sorry

theorem plates_not_adj (h : num_ways_arrange_plates 6 2 2 1 true = 1568) : 
  num_ways_arrange_plates 6 2 2 1 true = 1568 :=
  by exact h -- using the hypothesis directly for the theorem statement

end plates_not_adj_l19_19853


namespace find_m_l19_19553

theorem find_m (n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by
  sorry

end find_m_l19_19553


namespace points_per_touchdown_l19_19340

theorem points_per_touchdown (number_of_touchdowns : ℕ) (total_points : ℕ) (h1 : number_of_touchdowns = 3) (h2 : total_points = 21) : (total_points / number_of_touchdowns) = 7 :=
by
  sorry

end points_per_touchdown_l19_19340


namespace tobias_swimming_distance_l19_19029

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end tobias_swimming_distance_l19_19029


namespace democrats_ratio_l19_19022

theorem democrats_ratio (F M: ℕ) 
  (h_total_participants : F + M = 810)
  (h_female_democrats : 135 * 2 = F)
  (h_male_democrats : (1 / 4) * M = 135) : 
  (270 / 810 = 1 / 3) :=
by 
  sorry

end democrats_ratio_l19_19022


namespace find_b_value_l19_19151

theorem find_b_value
  (b : ℝ) :
  (∃ x y : ℝ, x = 3 ∧ y = -5 ∧ b * x + (b + 2) * y = b - 1) → b = -3 :=
by
  sorry

end find_b_value_l19_19151


namespace tangent_parallel_points_l19_19637

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∃ (x0 y0 : ℝ), (curve x0 = y0) ∧ 
                 (deriv curve x0 = 4) ∧
                 ((x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -4)) :=
by
  sorry

end tangent_parallel_points_l19_19637


namespace proof_y_minus_x_l19_19017

theorem proof_y_minus_x (x y : ℤ) (h1 : x + y = 540) (h2 : x = (4 * y) / 5) : y - x = 60 :=
sorry

end proof_y_minus_x_l19_19017


namespace cone_volume_270_degree_sector_l19_19442

noncomputable def coneVolumeDividedByPi (R θ: ℝ) (r h: ℝ) (circumf sector_height: ℝ) : ℝ := 
  if R = 20 
  ∧ θ = 270 / 360 
  ∧ 2 * Mathlib.pi * 20 = 40 * Mathlib.pi 
  ∧ circumf = 30 * Mathlib.pi
  ∧ 2 * Mathlib.pi * r = circumf
  ∧ r = 15
  ∧ sector_height = R
  ∧ r^2 + h^2 = sector_height^2 
  then (1/3) * Mathlib.pi * r^2 * h / Mathlib.pi 
  else 0

theorem cone_volume_270_degree_sector : coneVolumeDividedByPi 20 (270 / 360) 15 (5 * Real.sqrt 7) (30 * Mathlib.pi) 20 = 1125 * Real.sqrt 7 := 
by {
  -- This is where the proof would go
  sorry
}

end cone_volume_270_degree_sector_l19_19442


namespace cone_volume_divided_by_pi_l19_19438

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l19_19438


namespace solve_for_s_l19_19554

theorem solve_for_s (s : ℝ) :
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 14) = (s^2 - 3 * s - 18) / (s^2 - 2 * s - 24) →
  s = -5 / 4 :=
by {
  sorry
}

end solve_for_s_l19_19554


namespace find_coeffs_l19_19414

structure Coeffs :=
(a : Fin 2022 → ℝ)
(b : Fin 2021 → ℝ)

noncomputable def f (x : ℝ) (c : Coeffs) : ℝ :=
  c.a 0 + ∑ k in Finset.range 2021, 
    c.a (k + 1) * Real.cos (2 * (k + 1) * Real.pi * x) + 
    c.b (k + 1) * Real.sin (2 * (k + 1) * Real.pi * x)

theorem find_coeffs (c : Coeffs) : 
  (∀ x : ℝ, f x c + f (x + 1/2) c = f (2 * x) c) 
  ∧ (∀ x : ℝ, ∀ c' : Coeffs, (f x c = f x c' → c = c')) 
  → c = ⟨λ _, 0, λ _, 0⟩ :=
by
  sorry

end find_coeffs_l19_19414


namespace symmetric_origin_coordinates_l19_19382

def symmetric_coordinates (x y : ℚ) (x_line y_line : ℚ) : Prop :=
  x_line - 2 * y_line + 2 = 0 ∧ y_line = -2 * x_line ∧ x = -4/5 ∧ y = 8/5

theorem symmetric_origin_coordinates :
  ∃ (x_0 y_0 : ℚ), symmetric_coordinates x_0 y_0 (-4/5) (8/5) :=
by
  use -4/5, 8/5
  sorry

end symmetric_origin_coordinates_l19_19382


namespace distance_range_l19_19479

-- Defining the conditions as assumptions
variables (d : ℝ)
variables (H1 : d < 8)          -- Alice's statement is false
variables (H2 : d > 7)          -- Bob's statement is false
variables (H3 : d > 6)          -- Charlie's statement is false
variables (H4 : d ≠ 5)          -- Dana's statement is false

-- Problem statement in Lean 4
theorem distance_range (d : ℝ) (H1 : d < 8) (H2 : d > 7) (H3 : d > 6) (H4 : d ≠ 5) : 7 < d ∧ d < 8 :=
begin
  split,
  exact H2,
  exact H1,
end

end distance_range_l19_19479


namespace cyclic_quadrilateral_area_l19_19724

theorem cyclic_quadrilateral_area (A B C D : Point) (R : ℝ) (φ : ℝ) 
  (inscribed : inscribed_in_circle R A B C D) 
  (diagonal_angle : angle_between_diagonals φ A C B D) :
  area_of_quadrilateral A B C D = 2 * R^2 * sin (angle A) * sin (angle B) * sin φ :=
sorry

end cyclic_quadrilateral_area_l19_19724


namespace total_path_length_l19_19110

-- Define the conditions
def equilateral_triangle_side := 3 -- inches
def square_side := 6 -- inches
def rotation_angle := 60 -- degrees
def rotation_angle_rad : ℝ := rotation_angle * real.pi / 180 -- converting to radians
def arc_length_per_rotation := equilateral_triangle_side * rotation_angle_rad

-- Define the question translated as a theorem
theorem total_path_length :
  let number_of_sides := 4 in
  let number_of_vertices_per_side := 3 in
  let total_arc_length : ℝ := number_of_sides * number_of_vertices_per_side * arc_length_per_rotation in
  total_arc_length = 12 * real.pi :=
begin
  sorry
end

end total_path_length_l19_19110


namespace second_place_prize_l19_19097

theorem second_place_prize (total_prize : ℕ) (first_prize : ℕ) (third_prize : ℕ) (fourth_to_eighteenth_prize: ℕ) (number_of_winners: ℕ) (per_person_prize: ℕ)
  (h1 : total_prize = 800)
  (h2 : number_of_winners = 18)
  (h3 : first_prize = 200)
  (h4 : third_prize = 120)
  (h5 : per_person_prize = 22)
  (h6 : fourth_to_eighteenth_prize = (number_of_winners - 3) * per_person_prize) :
  let second_prize := total_prize - first_prize - third_prize - fourth_to_eighteenth_prize in
  second_prize = 150 :=
by
  have h7 : fourth_to_eighteenth_prize = 330 := by sorry  -- directly from the solution steps 1 and 2
  have h8 : first_prize + third_prize = 320 := by sorry  -- directly from solution step 3
  have second_prize := total_prize - (first_prize + third_prize + fourth_to_eighteenth_prize)
  show second_prize = 150 from by sorry

end second_place_prize_l19_19097


namespace intersection_M_N_l19_19133

def log_base_10 (x : ℝ) : ℝ := log x / log 10

def set_M : set ℝ := {y | ∃ x : ℝ, y = log_base_10 (x^2 + 1)}
def set_N : set ℝ := {x | 4^x > 4}

theorem intersection_M_N :
  (set_M ∩ set_N) = {z | z ∈ (1 : ℝ, ∞)} := 
by
  sorry

end intersection_M_N_l19_19133


namespace f_at_6_l19_19085

noncomputable def f (u : ℝ) : ℝ :=
  let x := (u - 2) / 4 in
  x^2 - x + 1

theorem f_at_6 : f 6 = 1 / 2 := by
  sorry

end f_at_6_l19_19085


namespace polynomial_relationship_l19_19204

theorem polynomial_relationship :
  (∀ x y, ((x = 0 ∧ y = 200) ∨ 
           (x = 1 ∧ y = 150) ∨ 
           (x = 2 ∧ y = 80) ∨ 
           (x = 3 ∧ y = 0) ∨ 
           (x = 4 ∧ y = -140)) →
  (y = (-10) * x ^ 3 + 20 * x ^ 2 - 60 * x + 200)) :=
by
  intros x y h
  cases h
  case inl h0 { rw [h0.1, h0.2] }
  case inr { cases h
    case inl h1 { rw [h1.1, h1.2] }
    case inr { cases h
      case inl h2 { rw [h2.1, h2.2] }
      case inr { cases h
        case inl h3 { rw [h3.1, h3.2] }
        case inr h4 { rw [h4.1, h4.2] }
      }
    }
  }
  done

end polynomial_relationship_l19_19204


namespace area_triangle_ABC_eq_3sqrt7div2_l19_19574

def point (x y : ℝ) := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.2 - A.2) * (C.1 - A.1)

theorem area_triangle_ABC_eq_3sqrt7div2 :
  let A := point 0 0,
      B := point 0 3,
      C := point (Real.sqrt 7) 0 in
  C.1^2 + (C.2 - B.2)^2 = 16 ∧ C.2 = 0 →
  area_of_triangle A B C = (3 * Real.sqrt 7) / 2 :=
by intros A B C cond; sorry

end area_triangle_ABC_eq_3sqrt7div2_l19_19574


namespace fifth_graders_more_than_seventh_l19_19239

theorem fifth_graders_more_than_seventh (price_per_pencil : ℕ) (price_per_pencil_pos : price_per_pencil > 0)
    (total_cents_7th : ℕ) (total_cents_7th_val : total_cents_7th = 201)
    (total_cents_5th : ℕ) (total_cents_5th_val : total_cents_5th = 243)
    (pencil_price_div_7th : total_cents_7th % price_per_pencil = 0)
    (pencil_price_div_5th : total_cents_5th % price_per_pencil = 0) :
    (total_cents_5th / price_per_pencil - total_cents_7th / price_per_pencil = 14) := 
by
    sorry

end fifth_graders_more_than_seventh_l19_19239


namespace arccos_sin_three_l19_19506

theorem arccos_sin_three : arccos (sin 3) = 3 - (π / 2) :=
by sorry

end arccos_sin_three_l19_19506


namespace transformed_polynomial_roots_l19_19281

theorem transformed_polynomial_roots :
  (∀ (r : ℝ), (r^3 - 3 * r^2 + 13 = 0) ↔ (r = 3 * r₁ ∨ r = 3 * r₂ ∨ r = 3 * r₃)) →
  ∀ (p : Polynomial ℝ),
    (p = Polynomial.Coeff (Polynomial.monic (Polynomial.mk [0, 0, -9, 351]))) →
    Polynomial_roots p = [3 * r₁, 3 * r₂, 3 * r₃] :=
by {
  sorry
}

end transformed_polynomial_roots_l19_19281


namespace nonzero_tricky_teeny_polynomials_l19_19096

theorem nonzero_tricky_teeny_polynomials :
  ∃ P : ℤ[X] → Prop, 
    (∀ (p : ℤ[X]), P p ↔ (
      (∃ (a b : ℤ), 
        (p = polynomial.C a + polynomial.X * polynomial.C b 
        ∧ a ≠ 0
        ∧ b = 4
        ∧ -7 ≤ a ∧ a ≤ 7 
        ∧ -7 ≤ b ∧ b ≤ 7
        ∧ p.eval 4 = 0))) 
    ∧ (∃ p1 p2 : ℤ[X], P p1 ∧ P p2 ∧ p1 ≠ p2 ∧ ¬(∃ p3 : ℤ[X], P p3 ∧ p3 ≠ p1 ∧ p3 ≠ p2))) :=
sorry

end nonzero_tricky_teeny_polynomials_l19_19096


namespace circle_radius_chords_l19_19032

open Real

/-- 
Given two chords AB and AC from a single point on a circle with lengths 9 and 17, 
and the distance between midpoints of these chords as 5, 
prove that the radius of the circle is 85/8.
-/
theorem circle_radius_chords 
  (AB AC : ℝ) (d : ℝ) 
  (hAB : AB = 9) 
  (hAC : AC = 17) 
  (hd : d = 5) :
  ∃ R : ℝ, R = 85 / 8 :=
by
  use 85 / 8
  sorry

end circle_radius_chords_l19_19032


namespace leg_equals_sum_of_radii_l19_19310

variables {A B C : Type}
          [euclidean_space A]
          (triangle_abc : triangle A B C)
          (r R : ℝ) -- radii for inscribed and excircle

-- Assume the conditions given
axiom right_angle_at_C : triangle_abc.∠C = 90
axiom incenter : incenter A B C = O
axiom excircle_center_opposite_to_A : excenter A B C opposite_to_A = O₁
axiom tangency_inscribed_bc : leg_tangent BC = P
axiom tangency_excircle_bc : leg_tangent BC = Q
axiom radius_inscribed_circle : radius (inscribed_circle A B C) = r
axiom radius_excircle_bc : radius (excircle A B C opposite_to_A) = R

-- Prove the desired equality
theorem leg_equals_sum_of_radii :
  ∀ (triangle_abc : triangle ℝ) (r R : ℝ), right_angle_at_C -> incenter -> excircle_center_opposite_to_A ->
  tangency_inscribed_bc -> tangency_excircle_bc -> radius_inscribed_circle -> radius_excircle_bc ->
  BC = R + r :=
by
  sorry

end leg_equals_sum_of_radii_l19_19310


namespace emma_final_balance_correct_l19_19537

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l19_19537


namespace find_number_of_female_students_l19_19822

noncomputable def number_of_female_students
  (avg_all : ℕ)
  (num_males : ℕ)
  (avg_males : ℕ)
  (avg_females : ℕ)
  (total_avg : ℕ) : ℕ :=
  let total_male_score := num_males * avg_males
  let total_score x := total_male_score + x * avg_females
  let total_students x := num_males + x
  Classical.choose (exists_eq (fun x => (total_score x) / (total_students x) = total_avg))

theorem find_number_of_female_students
  (avg_all : ℕ)
  (num_males : ℕ)
  (avg_males : ℕ)
  (avg_females : ℕ)
  (total_avg : ℕ)
  (h_avg_all : avg_all = 90)
  (h_num_males : num_males = 8)
  (h_avg_males : avg_males = 84)
  (h_avg_females : avg_females = 92)
  (h_total_avg : total_avg = 90)
  : number_of_female_students avg_all num_males avg_males avg_females total_avg = 24 :=
by
  simp [number_of_female_students, h_avg_all, h_num_males, h_avg_males, h_avg_females, h_total_avg]
  sorry

end find_number_of_female_students_l19_19822


namespace train_length_correct_l19_19870

noncomputable def length_of_train (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let total_distance := speed_m_per_s * time_s
  total_distance - platform_length_m

theorem train_length_correct :
  length_of_train 55 520 43.196544276457885 = 140 :=
by
  unfold length_of_train
  -- The conversion and calculations would be verified here
  sorry

end train_length_correct_l19_19870


namespace distance_between_trees_l19_19240

-- Definitions based on conditions
def yard_length : ℝ := 360
def number_of_trees : ℕ := 31
def number_of_gaps : ℕ := number_of_trees - 1

-- The proposition to prove
theorem distance_between_trees : yard_length / number_of_gaps = 12 := sorry

end distance_between_trees_l19_19240


namespace number_of_special_numbers_l19_19122

-- Definitions derived from the problem conditions.
def is_special (d1 d2 d3 d4 d5 : ℕ) : Prop :=
  (d1 * 10 + d2 = d3 * 10 + d4) ∨ 
  (d1 * 10 + d2 = d4 * 10 + d5)

def decimal_digit (d : ℕ) : Prop := 0 ≤ d ∧ d ≤ 9

-- Lean statement of the problem.
theorem number_of_special_numbers : 
  (finset.univ.filter (λ n : ℕ × ℕ × ℕ × ℕ × ℕ,
      let ⟨d1, d2, d3, d4, d5⟩ := n in
      decimal_digit d1 ∧ decimal_digit d2 ∧ decimal_digit d3 ∧ decimal_digit d4 ∧ decimal_digit d5 ∧ is_special d1 d2 d3 d4 d5)).card = 1990 :=
sorry

end number_of_special_numbers_l19_19122


namespace solution_set_of_inequality_l19_19972

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Problem conditions
theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x, x < 0 → f x = x + 2) :
  { x : ℝ | 2 * f x - 1 < 0 } = { x : ℝ | x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) } :=
by
  sorry

end solution_set_of_inequality_l19_19972


namespace tenth_pair_in_twentieth_row_l19_19605

noncomputable def pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if k = 0 ∨ k > n then (0, 0) else (k, n + 1 - k)

theorem tenth_pair_in_twentieth_row : pair_in_row 20 10 = (10, 11) := by
  sorry

end tenth_pair_in_twentieth_row_l19_19605


namespace range_of_f_when_a_half_increasing_values_of_a_l19_19201

section
  variable (x : ℝ) (a : ℝ)
  -- Conditions
  def f (x : ℝ) := Real.logBase a (a*x^2 - x + 1)
  def cond (a : ℝ) := (a > 0 : Prop) ∧ (a ≠ 1 : Prop)

  theorem range_of_f_when_a_half (h : a = 1 / 2) (c : cond a) : 
    (f x ≤ 1 : Prop) ∧ ∀ y, y < 1 → ∃ x, f x = y := 
  sorry

  theorem increasing_values_of_a (c : cond a) 
    (h_inc : ∀ x ∈ Icc (1/4 : ℝ) (3/2 : ℝ), MonotonicOn (Real.logBase a) Icc (1/4 : ℝ) (3/2 : ℝ)) :
    (a ∈ Icc (2/9 : ℝ) (1/3 : ℝ) ∨ Ici (2 : ℝ)) :=
  sorry
end

end range_of_f_when_a_half_increasing_values_of_a_l19_19201


namespace typeB_lines_l19_19209

noncomputable def isTypeBLine (line : Real → Real) : Prop :=
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧ (Real.sqrt ((P.1 + 5)^2 + P.2^2) - Real.sqrt ((P.1 - 5)^2 + P.2^2) = 6)

theorem typeB_lines :
  isTypeBLine (fun x => x + 1) ∧ isTypeBLine (fun x => 2) :=
by sorry

end typeB_lines_l19_19209


namespace student_marks_l19_19866

variable (max_marks : ℕ) (pass_percent : ℕ) (fail_by : ℕ)

theorem student_marks
  (h_max : max_marks = 400)
  (h_pass : pass_percent = 35)
  (h_fail : fail_by = 40)
  : max_marks * pass_percent / 100 - fail_by = 100 :=
by
  sorry

end student_marks_l19_19866


namespace solve_for_x_l19_19594

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1
  else if x > 1 then 4 - x^2
  else 0   -- Note: this line covers the undefined region between 0 and 1 for clarity.

theorem solve_for_x (x : ℝ) (h : f x = -1) : x = -2 ∨ x = Real.sqrt 5 :=
  sorry

end solve_for_x_l19_19594


namespace correct_representation_of_3_minus_10_minus_7_as_a_sum_l19_19394

theorem correct_representation_of_3_minus_10_minus_7_as_a_sum :
  (3 - 10 - 7 = 3 + (-10) + (-7)) :=
begin
  sorry
end

end correct_representation_of_3_minus_10_minus_7_as_a_sum_l19_19394


namespace find_S2_l19_19689

-- Define the sequence and the cumulative sum as specified in the problem

def sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a n = 4 * S n - 3

def cumulative_sum (a S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = (finset.range (n + 1)).sum a

theorem find_S2 (a S : ℕ → ℚ)
  (h1 : sequence a S)
  (h2 : cumulative_sum a S)
  (h3 : S 1 = a 1) :
  S 2 = 4 / 3 :=
by
  sorry

end find_S2_l19_19689


namespace conic_section_is_ellipse_l19_19813

theorem conic_section_is_ellipse : 
  (∀ x y : ℝ, (Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12) → (conic_section_type (x, y) = "E")) := 
by
  intro x y
  sorry

end conic_section_is_ellipse_l19_19813


namespace total_points_correct_l19_19316

def points_from_two_pointers (t : ℕ) : ℕ := 2 * t
def points_from_three_pointers (th : ℕ) : ℕ := 3 * th
def points_from_free_throws (f : ℕ) : ℕ := f

def total_points (two_points three_points free_throws : ℕ) : ℕ :=
  points_from_two_pointers two_points + points_from_three_pointers three_points + points_from_free_throws free_throws

def sam_points : ℕ := total_points 20 5 10
def alex_points : ℕ := total_points 15 6 8
def jake_points : ℕ := total_points 10 8 5
def lily_points : ℕ := total_points 12 3 16

def game_total_points : ℕ := sam_points + alex_points + jake_points + lily_points

theorem total_points_correct : game_total_points = 219 :=
by
  sorry

end total_points_correct_l19_19316


namespace symmetric_point_coordinates_l19_19771

structure Point : Type where
  x : ℝ
  y : ℝ

def symmetric_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def P : Point := { x := -10, y := -1 }

def P1 : Point := symmetric_y P

def P2 : Point := symmetric_x P1

theorem symmetric_point_coordinates :
  P2 = { x := 10, y := 1 } := by
  sorry

end symmetric_point_coordinates_l19_19771


namespace projection_magnitude_l19_19587

-- Define the coordinates of point A
def A : ℝ × ℝ × ℝ := (3, -1, 2)

-- Define the coordinates of point B, which is the projection of A onto the Oxy plane
def B : ℝ × ℝ × ℝ := (A.1, A.2, 0)

-- Define the magnitude of the vector OB
def magnitude_OB : ℝ := Real.sqrt (B.1 ^ 2 + B.2 ^ 2 + B.3 ^ 2)

-- State the theorem
theorem projection_magnitude : magnitude_OB = Real.sqrt 10 := by
  -- The proof will be filled in here
  sorry

end projection_magnitude_l19_19587


namespace bridge_length_l19_19107

def train_length : ℝ := 360
def train_speed_kmh : ℝ := 52
def time_to_pass_bridge : ℝ := 34.61538461538461
def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600

theorem bridge_length : 
  let total_distance := train_speed_ms * time_to_pass_bridge in
  let bridge_length := total_distance - train_length in
  bridge_length = 140 :=
by
  sorry

end bridge_length_l19_19107


namespace decimal_to_fraction_l19_19788

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l19_19788


namespace sum_of_first_five_terms_l19_19189

variable (n : ℕ) (a : ℕ → ℚ)

-- Definition of the geometric sequence with first term 1
def geom_seq (a : ℕ → ℚ) (q : ℚ) := a 0 = 1 ∧ ∀ n, a (n + 1) = q * a n

-- Definition of the sum of the first n terms of a geometric sequence
def geom_sum (a : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n, S n = ∑ i in Finset.range (n + 1), a i

-- Conditions stated in the problem
theorem sum_of_first_five_terms :
  ∃ q : ℚ, q ≠ 1 ∧
    geom_seq a q ∧ 
    geom_sum a (λ n, ∑ i in Finset.range (n + 1), a i) ∧
    9 * ∑ i in Finset.range 3, a i = ∑ i in Finset.range 6, a i ∧
    ∑ i in Finset.range 5, (1 / a i) = 31 / 16 :=
sorry

end sum_of_first_five_terms_l19_19189


namespace max_y_coordinate_l19_19946

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l19_19946


namespace arc_measure_is_100_degrees_l19_19424

theorem arc_measure_is_100_degrees
  (A B C M D : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited D]
  (triangle_is_isosceles : ∀ (a b c : Type), b = c)
  (M_is_midpoint_BD : ∀ (b d : Type), ∃ (m : Type), m = d)
  (M_radius_MD : ∀ (m d : Type), ∃ (r : ℝ), r = 1)
  (angle_BAC : ℝ)
  (vertex_angle : angle_BAC = 65) :
  ∃ (central_angle : ℝ), central_angle = 100 := by
  sorry

end arc_measure_is_100_degrees_l19_19424


namespace sum_q_t_at_8_l19_19283

noncomputable def T : set (fin 8 → bool) := { t | true }

def q_t (t : fin 8 → bool) : polynomial ℚ :=
  polynomial.interpolate (finset.univ.image (λ n : fin 8, (n, if t n then 1 else 0)))

def q (x : ℚ) : ℚ :=
  ∑ t in T.to_finset, polynomial.eval x (q_t t)

theorem sum_q_t_at_8 : q 8 = 128 :=
by sorry

end sum_q_t_at_8_l19_19283


namespace proof_problem_l19_19607

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 > 4}
def B := {x : ℝ | 2^x > 1}

-- Defining the complement of A in the universal set R
def complement_A := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- The intersection of the complement of A and B
def result_set := {x : ℝ | 0 < x ∧ x ≤ 2}

-- The theorem statement that needs to be proved
theorem proof_problem : (complement_A ∩ B) = result_set :=
by sorry

end proof_problem_l19_19607


namespace solution_to_F_ss2_eq_1000_l19_19516

def F (a b c : ℝ) : ℝ := a * b^(c + 1)

theorem solution_to_F_ss2_eq_1000 : ∃ s > 0, F s s 2 = 1000 ∧ s = 10^0.75 :=
by
  sorry

end solution_to_F_ss2_eq_1000_l19_19516


namespace find_circle_area_l19_19672

noncomputable def circle_area : Prop :=
  ∃ (R : ℝ) (A B C D : ℝ),
    ∠ A K B = 60 ∧
    (tangent_to_angle_side A B) ∧
    (intersects_other_side A B C D) ∧
    (intersects_bisector C D) ∧
    (AB = sqrt 6) ∧
    (CD = sqrt 6) ∧
    (π * R^2 = π * sqrt 3)

theorem find_circle_area : circle_area :=
sorry

end find_circle_area_l19_19672


namespace Compute_fraction_power_l19_19905

theorem Compute_fraction_power :
  (81081 / 27027) ^ 4 = 81 :=
by
  -- We provide the specific condition as part of the proof statement
  have h : 27027 * 3 = 81081 := by norm_num
  sorry

end Compute_fraction_power_l19_19905


namespace fractional_eq_no_real_roots_l19_19228

theorem fractional_eq_no_real_roots (k : ℝ) :
  (∀ x : ℝ, (x - 1) ≠ 0 → (k / (x - 1) + 3 ≠ x / (1 - x))) → k = -1 :=
by
  sorry

end fractional_eq_no_real_roots_l19_19228


namespace meal_cost_l19_19843

/-- 
    Define the cost of a meal consisting of one sandwich, one cup of coffee, and one piece of pie 
    given the costs of two different meals.
-/
theorem meal_cost (s c p : ℝ) (h1 : 2 * s + 5 * c + p = 5) (h2 : 3 * s + 8 * c + p = 7) :
    s + c + p = 3 :=
by
  sorry

end meal_cost_l19_19843


namespace positive_divisors_of_7_factorial_l19_19890

theorem positive_divisors_of_7_factorial : 
  let n := 7!
  let prime_factors := [(2, 4), (3, 2), (5, 1), (7, 1)]
  number_of_divisors n = 60 :=
sorry

end positive_divisors_of_7_factorial_l19_19890


namespace calculate_principal_sum_simple_interest_l19_19767

theorem calculate_principal_sum_simple_interest :
  let P := 1750 in
  let r1 := 8 in
  let t1 := 3 in
  let r2 := 10 in
  let t2 := 2 in
  let principal_ci := 4000 in
  let A := principal_ci * (1 + r2 / 100) ^ t2 in
  let CI := A - principal_ci in
  let SI := CI / 2 in
  SI = (P * r1 * t1) / 100 :=
by
  let P := 1750
  let r1 := 8
  let t1 := 3
  let r2 := 10
  let t2 := 2
  let principal_ci := 4000
  let A := principal_ci * (1 + r2 / 100) ^ t2
  let CI := A - principal_ci
  let SI := CI / 2
  sorry

end calculate_principal_sum_simple_interest_l19_19767


namespace find_angle_C_find_area_triangle_l19_19208

-- Define the given conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (triangle_ABC : Triangle A B C a b c) -- Denote that it's triangle ABC

-- First Part
theorem find_angle_C (h1 : (1 - tan A) * (1 - tan B) = 2) : C = π / 4 :=
sorry

-- Second Part
theorem find_area_triangle (C : ℝ) (h1 : C = π / 4) (h2 : b = 2 * sqrt 2) 
  (h3 : c = 4) : area_triangle a b c = 2 * sqrt 3 + 2 :=
sorry

end find_angle_C_find_area_triangle_l19_19208


namespace tim_number_of_goats_l19_19775

variable (G L : ℕ)
variable (cost_goat cost_llama total_cost : ℕ)
variable (ratio_llama_goat llama_extra_cost : ℚ)

noncomputable def cost_goat := 400
noncomputable def ratio_llama_goat := 2
noncomputable def llama_extra_cost := 0.5
noncomputable def cost_llama := cost_goat + (cost_goat * llama_extra_cost).toInt
noncomputable def total_cost := 4800

theorem tim_number_of_goats (h1 : L = ratio_llama_goat * G)
                           (h2 : cost_llama = cost_goat + (cost_goat * llama_extra_cost).toInt)
                           (h3 : total_cost = cost_goat * G + cost_llama * L) :
  G = 3 :=
by 
  -- proof omitted
  sorry

end tim_number_of_goats_l19_19775


namespace smallest_num_rectangles_l19_19045

theorem smallest_num_rectangles (a b : ℕ) (h_dim : a = 3 ∧ b = 4) : 
  ∃ n : ℕ, (∃ side : ℕ, side * side = n * (a * b)) ∧ n = 12 :=
by
  use 12
  have h : 12 * 12 = 12 * (a * b) := 
    have ha : a = 3 := h_dim.1
    have hb : b = 4 := h_dim.2
    calc
      12 * 12 = 12 * 12 : rfl
           ... = 12 * (3 * 4) : by rw [ha, hb]
           ... = 144 : rfl
  exact ⟨144, ⟨12, h⟩, rfl⟩

end smallest_num_rectangles_l19_19045


namespace corrected_mean_l19_19826

theorem corrected_mean (mean : ℝ) (num_observations : ℕ) 
  (incorrect_observation correct_observation : ℝ)
  (h_mean : mean = 36) (h_num_observations : num_observations = 50)
  (h_incorrect_observation : incorrect_observation = 23) 
  (h_correct_observation : correct_observation = 44)
  : (mean * num_observations + (correct_observation - incorrect_observation)) / num_observations = 36.42 := 
by
  sorry

end corrected_mean_l19_19826


namespace distinct_negative_real_roots_l19_19926

def poly (p : ℝ) (x : ℝ) : ℝ := x^4 + 2*p*x^3 + x^2 + 2*p*x + 1

theorem distinct_negative_real_roots (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly p x1 = 0 ∧ poly p x2 = 0) ↔ p > 3/4 :=
sorry

end distinct_negative_real_roots_l19_19926


namespace area_triangle_l19_19119

-- Definition of the problem's conditions
def radius : ℝ := 5
def ω1 : Type := Unit -- Dummy type representing circle ω1
def ω2 : Type := Unit -- Dummy type representing circle ω2
def ω3 : Type := Unit -- Dummy type representing circle ω3

-- Position Points
axiom Q1_on_ω1 : ω1
axiom Q2_on_ω2 : ω2
axiom Q3_on_ω3 : ω3

-- Tangency conditions and equilateral property
axiom Q1Q2_eq_Q2Q3_eq_Q3Q1 : Q1_on_ω1 ≠ Q2_on_ω2 ∧ Q2_on_ω2 ≠ Q3_on_ω3 ∧ Q1_on_ω1 ≠ Q3_on_ω3
axiom tangent1 : tangent Q1_on_ω1 Q2_on_ω2
axiom tangent2 : tangent Q2_on_ω2 Q3_on_ω3
axiom tangent3 : tangent Q3_on_ω3 Q1_on_ω1

-- The main theorem to prove 
theorem area_triangle : 
  ∃ area : ℝ, area = 143.75 + 25 * Real.sqrt 19 := 
sorry

end area_triangle_l19_19119


namespace smallest_next_divisor_of_221_l19_19680

noncomputable def is_divisor (a b : ℕ) := b % a = 0

theorem smallest_next_divisor_of_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m < 10000) (h2 : m % 2 = 0) (h3 : is_divisor 221 m) :
  ∃ n, n > 221 ∧ is_divisor n m ∧ n = 238 :=
by
  use 238
  split
  · -- Proof that 238 > 221
    sorry
  split
  · -- Proof that 238 is a divisor of m
    sorry
  · -- Proof that 238 is the smallest next divisor
    sorry

end smallest_next_divisor_of_221_l19_19680


namespace digit_one_not_in_mean_l19_19513

def seq : List ℕ := [5, 55, 555, 5555, 55555, 555555, 5555555, 55555555, 555555555]

noncomputable def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem digit_one_not_in_mean :
  ¬(∃ d, d ∈ (arithmetic_mean seq).digits 10 ∧ d = 1) :=
sorry

end digit_one_not_in_mean_l19_19513


namespace acute_triangle_angle_A_is_60_degrees_l19_19018

open Real

variables {A B C : ℝ} -- Assume A, B, C are reals representing the angles of the triangle

theorem acute_triangle_angle_A_is_60_degrees
  (h_acute : A < 90 ∧ B < 90 ∧ C < 90)
  (h_eq_dist : dist A O = dist A H) : A = 60 :=
  sorry

end acute_triangle_angle_A_is_60_degrees_l19_19018


namespace angles_equal_l19_19308

-- Define the geometry and related properties
variables {A B C D E F M G : Type} 
variables [euclidean_geometry A B C D E F M G]

noncomputable def midpoint (X Y Z : Type) [euclidean_geometry X Y Z] : Prop :=
  dist X Y = dist Y Z

variables (ABCD : parallelogram A B C D) 
          (EFcoord : collinear B C E) (EF2coord : collinear A D F) 
          (EF_eq_ED : dist E F = dist E D) (ED_eq_DC : dist E D = dist D C) 
          (M_midpoint_BE : midpoint B E M)
          (G_on_MD : lies_on_line M D G) (G_on_EF : lies_on_line E F G)

-- Prove the desired angle equality
theorem angles_equal :
  ∡ E A C = ∡ G B D :=
sorry

end angles_equal_l19_19308


namespace simplify_expression_eval_l19_19326

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l19_19326


namespace simplify_expression_eval_l19_19328

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l19_19328


namespace cone_volume_270_degree_sector_l19_19441

noncomputable def coneVolumeDividedByPi (R θ: ℝ) (r h: ℝ) (circumf sector_height: ℝ) : ℝ := 
  if R = 20 
  ∧ θ = 270 / 360 
  ∧ 2 * Mathlib.pi * 20 = 40 * Mathlib.pi 
  ∧ circumf = 30 * Mathlib.pi
  ∧ 2 * Mathlib.pi * r = circumf
  ∧ r = 15
  ∧ sector_height = R
  ∧ r^2 + h^2 = sector_height^2 
  then (1/3) * Mathlib.pi * r^2 * h / Mathlib.pi 
  else 0

theorem cone_volume_270_degree_sector : coneVolumeDividedByPi 20 (270 / 360) 15 (5 * Real.sqrt 7) (30 * Mathlib.pi) 20 = 1125 * Real.sqrt 7 := 
by {
  -- This is where the proof would go
  sorry
}

end cone_volume_270_degree_sector_l19_19441


namespace proving_inequality_l19_19699

open Euler

def sequence_a (φ : ℕ → ℕ) (n : ℕ) : ℕ → ℕ
| 1       := 2
| (m + 1) := φ (sequence_a φ m)

theorem proving_inequality (φ : ℕ → ℕ) (Hφ : ∀ k, φ k ≤ k) (n : ℕ) :
  (∀ m, 1 ≤ m → m ≤ n - 1 → sequence_a φ n m = φ (sequence_a φ n (m + 1))) →
  sequence_a φ n n ≥ 2^(n-1) :=
begin
  sorry
end

end proving_inequality_l19_19699


namespace angle_projection_line_l19_19451

noncomputable def angle_between_projection_and_line (angle_e_S angle_e_f : ℝ) : ℝ :=
  Real.arccos ((Real.cos angle_e_f) / (Real.cos angle_e_S))

theorem angle_projection_line (h1 : angle_e_S = 40)
                             (h2 : angle_e_f = 49) :
                             angle_between_projection_and_line 40 49 ≈ 31 + 6/60 :=
  sorry

end angle_projection_line_l19_19451


namespace sine_shift_l19_19777

theorem sine_shift (x : ℝ) : 
  sin (2 * (x - (π / 12))) = sin (2 * x - π / 6) := 
by
  sorry

end sine_shift_l19_19777


namespace centroid_circular_path_l19_19609

def triangle := {A B C : ℝ × ℝ}
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Given triangle ABC
def ABC_fixed_base (A B : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) (C_moves_on_circle: dist (midpoint A B) C = r): Prop :=
  let M := midpoint A B
  ∃ G : ℝ × ℝ, (dist G M = dist C M / 3)

-- Statement that needs to be proved
theorem centroid_circular_path
  (A B : ℝ × ℝ) (r : ℝ) 
  (C_moves_on_circle: ∀ C, dist (midpoint A B) C = r):
  ∃ G : ℝ × ℝ, dist (midpoint A B) G = r / 3 :=
by sorry

end centroid_circular_path_l19_19609


namespace all_terms_are_integers_l19_19066

   noncomputable def a : ℕ → ℤ
   | 0 => 1
   | 1 => 1
   | 2 => 997
   | n + 3 => (1993 + a (n + 2) * a (n + 1)) / a n

   theorem all_terms_are_integers : ∀ n : ℕ, ∃ (a : ℕ → ℤ), 
     (a 1 = 1) ∧ 
     (a 2 = 1) ∧ 
     (a 3 = 997) ∧ 
     (∀ n : ℕ, a (n + 3) = (1993 + a (n + 2) * a (n + 1)) / a n) → 
     (∀ n : ℕ, ∃ k : ℤ, a n = k) := 
   by 
     sorry
   
end all_terms_are_integers_l19_19066


namespace binom_10_8_equals_45_l19_19507

theorem binom_10_8_equals_45 : Nat.choose 10 8 = 45 := 
by
  sorry

end binom_10_8_equals_45_l19_19507


namespace time_to_pass_correct_l19_19850

-- Define the jogger's speed in m/s
def jogger_speed_kmph : ℝ := 9
def jogger_speed_ms := jogger_speed_kmph * (1000 / 3600)

-- Define the train's speed in m/s
def train_speed_kmph : ℝ := 60
def train_speed_ms := train_speed_kmph * (1000 / 3600)

-- Define the incline reduction factor (5%)
def incline_factor : ℝ := 0.05

-- Calculate the reduced speeds due to incline
def jogger_reduced_speed := jogger_speed_ms * (1 - incline_factor)
def train_reduced_speed := train_speed_ms * (1 - incline_factor)

-- Calculate the relative speed
def relative_speed := train_reduced_speed - jogger_reduced_speed

-- Define the distances
def jogger_lead : ℝ := 420
def train_length : ℝ := 200
def total_distance := jogger_lead + train_length

-- Assume total time to pass the jogger in seconds
def time_to_pass := total_distance / relative_speed

-- Prove the time to pass is approximately 46.05 seconds
theorem time_to_pass_correct : abs (time_to_pass - 46.05) < 0.01 := by 
  sorry

end time_to_pass_correct_l19_19850


namespace quadratic_range_l19_19561

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range_l19_19561


namespace percentage_of_loss_is_10_l19_19090

-- Definitions based on conditions
def cost_price : ℝ := 1800
def selling_price : ℝ := 1620
def loss : ℝ := cost_price - selling_price

-- The goal: prove the percentage of loss equals 10%
theorem percentage_of_loss_is_10 :
  (loss / cost_price) * 100 = 10 := by
  sorry

end percentage_of_loss_is_10_l19_19090


namespace volcano_intact_l19_19856

theorem volcano_intact (initial_count : ℕ)
                       (perc_2months : ℝ)
                       (perc_halfyear : ℝ)
                       (perc_yearend : ℝ)
                       (exploded_2months: initial_count * perc_2months / 100)
                       (remaining_after_2months: initial_count - exploded_2months)
                       (exploded_halfyear: remaining_after_2months * perc_halfyear / 100)
                       (remaining_after_halfyear: remaining_after_2months - exploded_halfyear)
                       (exploded_yearend: remaining_after_halfyear * perc_yearend / 100)
                       (remaining_after_yearend: remaining_after_halfyear - exploded_yearend) :
  let results : Nat := remaining_after_yearend in
  initial_count = 200 ∧ perc_2months = 20 ∧ perc_halfyear = 40 ∧ perc_yearend = 50 → results = 48 :=
begin
  sorry
end

end volcano_intact_l19_19856


namespace cube_problem_l19_19108

theorem cube_problem (n : ℕ) (V : ℕ) 
  (painted_faces : 4)
  (unit_cubes : n^3)
  (total_faces : 6 * n^3) :
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 2 ∧ V = 8 :=
by
  sorry

end cube_problem_l19_19108


namespace area_of_trapezoid_EFGH_l19_19147

-- Define the conditions: coordinates of the vertices
def E := (0, 0)
def F := (0, 4)
def G := (3, 4)
def H := (3, -1)

-- Define the lengths of the bases and the height of the trapezoid
def EF : ℝ := 4
def GH : ℝ := 5
def Height : ℝ := 3

-- Define the formula for the area of the trapezoid
def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  (1/2) * (base1 + base2) * height

-- Prove the statement
theorem area_of_trapezoid_EFGH : 
  trapezoid_area EF GH Height = 13.5 := 
  by sorry

end area_of_trapezoid_EFGH_l19_19147


namespace total_perimeter_approx_l19_19873

-- Conditions
def base1 : ℝ := 5
def base2 : ℝ := 7
def side1 : ℝ := 3
def side2 : ℝ := 4
def radius : ℝ := 3.1
def pi_approx : ℝ := 3.14

-- Perimeter of trapezoid excluding the longer base
def trapezoid_perimeter := base1 + side1 + side2

-- Circumference of the semicircle
def semicircle_circumference := pi_approx * radius

-- Total perimeter of the combined shape
def total_perimeter := trapezoid_perimeter + semicircle_circumference

-- Proof that the total perimeter is approximately 21.734 cm
theorem total_perimeter_approx : total_perimeter ≈ 21.734 := by 
  sorry

end total_perimeter_approx_l19_19873


namespace count_squares_in_region_l19_19058

theorem count_squares_in_region : 
  let bounded_region (x y : ℕ) := y ≤ 3 * x ∧ y ≥ -1 ∧ x ≤ 3
  ∀ (n : ℕ), 
  let count_squares (n : ℕ) := if n = 1 then 19 else if n = 2 then 14 else if n = 3 then 1 else 0
  ∑ i in finset.range 4, count_squares i = 34 :=
by sorry

end count_squares_in_region_l19_19058


namespace smallest_n_integer_l19_19906

noncomputable def fourthRoot (x : ℝ) : ℝ :=
  x^(1 / 4)

noncomputable def y (n : ℕ) : ℝ :=
  if n = 1 then fourthRoot 4 else
  let y_prev := y (n - 1)
  in y_prev ^ fourthRoot 4

theorem smallest_n_integer : ∃ n : ℕ, y n ∈ set.Ioo 0 4 ∧ y n ∈ set.Ioo (sqrt 2) (2 ^ 2) ∧ y n = 4 ∧ ∀ m : ℕ, m < n → (y m ∉ set.range floor) :=
sorry

end smallest_n_integer_l19_19906


namespace percentage_of_primes_divisible_by_2_l19_19049

open_locale classical
noncomputable theory

def prime_numbers_less_than_twenty := {p : ℕ | nat.prime p ∧ p < 20}

theorem percentage_of_primes_divisible_by_2 : 
  (card {p ∈ prime_numbers_less_than_twenty | 2 ∣ p}).to_real / (card prime_numbers_less_than_twenty).to_real * 100 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_2_l19_19049


namespace problem_solution_l19_19985

def a_sequence : ℕ → ℝ
| 0     := 15
| (n+1) := a_sequence n - 2 / 3

theorem problem_solution :
  ∃ k, a_sequence k * a_sequence (k + 1) < 0 ∧ k = 23 := by
  sorry

end problem_solution_l19_19985


namespace convert_radians_to_degrees_convert_degrees_to_radians_l19_19514

-- Definitions for conversion formulas
def radians_to_degrees (r : ℝ) : ℝ := r * (180 / Real.pi)
def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Problem 1: Convert -5/3 pi to degrees
theorem convert_radians_to_degrees : radians_to_degrees (- (5 / 3) * Real.pi) = -300 :=
by sorry

-- Problem 2: Convert -135 degrees to radians
theorem convert_degrees_to_radians : degrees_to_radians (-135) = - (3 / 4) * Real.pi :=
by sorry

end convert_radians_to_degrees_convert_degrees_to_radians_l19_19514


namespace bridge_length_l19_19832

noncomputable def speed (distance time : ℝ) : ℝ := distance / time

theorem bridge_length (train_length time_post time_bridge : ℝ) :
    time_post = 40 → 
    time_bridge = 600 → 
    speed train_length time_post = 15 →
    ∃ bridge_length, speed (train_length + bridge_length) time_bridge = 15 ∧ bridge_length = 8400 :=
by
  intros h1 h2 h3
  use 8400
  rw [h1, h2, h3]
  sorry

end bridge_length_l19_19832


namespace subtract_30_divisible_l19_19828

theorem subtract_30_divisible (n : ℕ) (d : ℕ) (r : ℕ) 
  (h1 : n = 13602) (h2 : d = 87) (h3 : r = 30) 
  (h4 : n % d = r) : (n - r) % d = 0 :=
by
  -- Skipping the proof as it's not required
  sorry

end subtract_30_divisible_l19_19828


namespace distance_to_AB_equal_l19_19336

noncomputable def rotation_about_point (X : ℂ) (theta : ℝ) (Z : ℂ) : ℂ :=
let rotation := complex.exp (complex.i * theta) in
rotation * (Z - X) + X

def distance_from_point_to_line (P : ℂ) : ℝ :=
complex.abs P.im

theorem distance_to_AB_equal
  (A B : ℂ) (AB_unit_len : complex.abs (B - A) = 1)
  (f g : ℂ → ℂ) (P : ℂ)
  (h_f : ∀ X, f X = rotation_about_point A (real.pi / 3) X)
  (h_g : ∀ X, g X = rotation_about_point B (-real.pi / 2) X)
  (h_P : g (f P) = P) :
  distance_from_point_to_line P = (1 + real.sqrt 3) / 2 := by
  sorry

end distance_to_AB_equal_l19_19336


namespace moving_circle_trajectory_eq_l19_19589

noncomputable def is_trajectory_equation (x y : ℝ) : Prop :=
  (x^2 / 36) + (y^2 / 27) = 1

theorem moving_circle_trajectory_eq :
  ∀ P : ℝ × ℝ,
    ((∃ R : ℝ, ( (P.1 - 3)^2 + P.2^2 = (2 + R)^2 )
      ∧ ( (P.1 + 3)^2 + P.2^2 = (10 - R)^2 )) → is_trajectory_equation P.1 P.2) :=
begin
  sorry
end

end moving_circle_trajectory_eq_l19_19589


namespace decimal_to_fraction_l19_19793

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l19_19793


namespace probability_two_green_two_red_l19_19076

theorem probability_two_green_two_red (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) :
  total_balls = 30 → green_balls = 9 → white_balls = 14 → red_balls = 7 →
  (draws : ℕ) = 4 →
  -- Probability calculation
  (P(2 Green and 2 Red) = (green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1)) *
                         (red_balls / (total_balls - 2)) * ((red_balls - 1) / (total_balls - 3))) → 
  P(2 Green and 2 Red) = 21 / 435 := 
sorry

end probability_two_green_two_red_l19_19076


namespace circle_equation_equivalence_l19_19916

theorem circle_equation_equivalence 
    (x y : ℝ) : 
    x^2 + y^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 :=
sorry

end circle_equation_equivalence_l19_19916


namespace hexagon_circumcircle_distance_l19_19425

variable (P A B C D E F : Type) 
variable [AddGroup P] [AddGroup A] [AddGroup B] [AddGroup C]
variable [AddGroup D] [AddGroup E] [AddGroup F] 
variable (circumcircle : Set (P × P))

open Real Complex

-- Assume vertices of the regular hexagon in the coordinate plane (as complex numbers)
noncomputable def vertex_A := (1 : ℂ)
noncomputable def vertex_B := Complex.exp (Complex.i * π / 3)
noncomputable def vertex_C := Complex.exp (Complex.i * 2 * π / 3)
noncomputable def vertex_D := -1
noncomputable def vertex_E := -Complex.exp (Complex.i * π / 3)
noncomputable def vertex_F := -Complex.exp (Complex.i * 2 * π / 3)

def point_on_circumcircle (p : ℂ) : Prop := p ∈ circumcircle

def opposite_sides (P A E D : ℂ) : Prop := ¬ (SameSide P A ED)

axiom circumcircle_property_of_regular_hexagon :
  ∀ (P : ℂ), point_on_circumcircle P → opposite_sides P vertex_A vertex_E vertex_D →
  Complex.abs (P - vertex_A) + Complex.abs (P - vertex_B) = 
  Complex.abs (P - vertex_C) + Complex.abs (P - vertex_D) + 
  Complex.abs (P - vertex_E) + Complex.abs (P - vertex_F)
  
theorem hexagon_circumcircle_distance :
  (point_on_circumcircle P) →
  (opposite_sides P vertex_A vertex_E vertex_D) →
  Complex.abs (P - vertex_A) + Complex.abs (P - vertex_B) =
  Complex.abs (P - vertex_C) + Complex.abs (P - vertex_D) +
  Complex.abs (P - vertex_E) + Complex.abs (P - vertex_F) :=
  by sorry
  

end hexagon_circumcircle_distance_l19_19425


namespace hyperbola_eccentricity_l19_19005

theorem hyperbola_eccentricity 
  (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0)
  (h : 9 * (a ^ 2) + a ^ 2 = 4 * (c ^ 2)) :
  let e := c / a in
  e = Real.sqrt 10 / 2 :=
by
  exist (a_pos) (b_pos) (c_pos) (h) λέ a b c e h sorry

end hyperbola_eccentricity_l19_19005


namespace max_y_coordinate_l19_19936

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l19_19936


namespace period_extrema_symmetry_l19_19199

noncomputable def f (x : ℝ) : ℝ :=
  5 * sin x * cos x - 5 * real.sqrt 3 * (cos x)^2 + (5 * real.sqrt 3) / 2

theorem period_extrema_symmetry :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧  -- Period of f(x)
  (∃ max_val min_val : ℝ, max_val = 5 ∧ min_val = -5 ∧  -- Extrema values
     ∀ x : ℝ, (∃ k : ℤ, (x = (5 * real.pi / 12 + k * real.pi) ∧ f x = max_val) 
                      ∨ (x = (k * real.pi - real.pi / 12) ∧ f x = min_val))) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ set.Icc (k * real.pi - real.pi / 12) (5 * real.pi / 12 + k * real.pi) 
                → ∃ n : ℕ, f x.succ > f x) ∧  -- Increasing intervals
  (∀ k : ℤ, ∃ x : ℝ, x = 5 * real.pi / 12 + k * real.pi / 2) ∧  -- Axis of symmetry
  (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (real.pi / 6 + k * real.pi / 2, 0)) -- Center of symmetry
:= by sorry

end period_extrema_symmetry_l19_19199


namespace posters_count_l19_19261

-- Define the regular price per poster
def regular_price : ℕ := 4

-- Jeremy can buy 24 posters at regular price
def posters_at_regular_price : ℕ := 24

-- Total money Jeremy has is equal to the money needed to buy 24 posters
def total_money : ℕ := posters_at_regular_price * regular_price

-- The special deal: buy one get the second at half price
def cost_of_two_posters : ℕ := regular_price + regular_price / 2

-- Number of pairs Jeremy can buy with his total money
def number_of_pairs : ℕ := total_money / cost_of_two_posters

-- Total number of posters Jeremy can buy under the sale
def total_posters := number_of_pairs * 2

-- Prove that the total posters is 32
theorem posters_count : total_posters = 32 := by
  sorry

end posters_count_l19_19261


namespace proteges_57_l19_19083

def divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d => n % d = 0)

def units_digit (n : ℕ) : ℕ := n % 10

def proteges (n : ℕ) : List ℕ := (divisors n).map units_digit

theorem proteges_57 : proteges 57 = [1, 3, 9, 7] :=
sorry

end proteges_57_l19_19083


namespace distance_between_C_and_D_is_1_l19_19703

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def vertex_of_parabola (a b c : ℝ) : ℝ × ℝ := 
  let x := -b / (2 * a)
  let y := a * x^2 + b * x + c
  (x, y)

def vertex_C := vertex_of_parabola 2 (-8) 18 -- vertex of y = 2x^2 - 8x + 18
def vertex_D := vertex_of_parabola (-3) 6 7 -- vertex of y = -3x^2 + 6x + 7

theorem distance_between_C_and_D_is_1 :
  distance vertex_C vertex_D = 1 :=
  sorry

end distance_between_C_and_D_is_1_l19_19703


namespace min_value_fraction_l19_19998

theorem min_value_fraction {a b x y : ℝ} {n : ℕ} 
  (a_pos : 0 < a) (b_pos : 0 < b) (x_pos : 0 < x) (y_pos : 0 < y) 
  (x_y_sum_one : x + y = 1) (n_pos : 0 < n):
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y = 1) ∧ (∃ n : ℕ, 0 < n) → 
  (∃ min_val : ℝ, min_val = (Real.sqrt(n+1) a + Real.sqrt(n+1) b)^(n+1) ∧ 
  ∀ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (x + y = 1) → ((a / x^n) + (b / y^n) ≥ min_val))) :=
sorry

end min_value_fraction_l19_19998


namespace lightyear_digit_count_l19_19089

theorem lightyear_digit_count :
  let ly_distance := 9.46 * 10^12
  (ly_distance : ℝ).digits = 13 :=
by
  /-
  Here you would construct the necessary proof.
  By simplification, proving the digit count of 9460000000000 is 13.
  -/
  sorry

end lightyear_digit_count_l19_19089


namespace air_quality_conditional_prob_l19_19876

theorem air_quality_conditional_prob :
  let p1 := 0.8
  let p2 := 0.68
  let p := p2 / p1
  p = 0.85 :=
by
  sorry

end air_quality_conditional_prob_l19_19876


namespace equal_sets_d_l19_19398

theorem equal_sets_d : 
  (let M := {x | x^2 - 3*x + 2 = 0}
   let N := {1, 2}
   M = N) :=
by 
  sorry

end equal_sets_d_l19_19398


namespace solution_correctness_l19_19542

theorem solution_correctness (x r : ℝ) (h_r : 4 < r ∧ r < 5) :
  (x ≠ 4) →
  ((x^2 * (x + 1) / (x - 4)^2) ≥ 20) ↔ (x ≥ r) :=
begin
  -- Proof skipped
  sorry
end

end solution_correctness_l19_19542


namespace Mary_work_days_l19_19300

theorem Mary_work_days :
  ∀ (M : ℝ), (∀ R : ℝ, R = M / 1.30) → (R = 20) → M = 26 :=
by
  intros M h1 h2
  sorry

end Mary_work_days_l19_19300


namespace max_distance_product_l19_19290

theorem max_distance_product (P : ℂ) (A B : ℂ) (hA : A = -2) (hB : B = 2)
  (hP_unit : complex.abs P = 1) : 
  (∃ PA PB : ℝ, PA = complex.abs (P + 2) ∧ PB = complex.abs (P - 2) ∧ PA * PB = 5) := sorry

end max_distance_product_l19_19290


namespace greatest_sum_solution_l19_19365

theorem greatest_sum_solution (x y : ℤ) (h : x^2 + y^2 = 20) : 
  x + y ≤ 6 :=
sorry

end greatest_sum_solution_l19_19365


namespace max_y_coordinate_l19_19945

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l19_19945


namespace number_of_girls_l19_19357

theorem number_of_girls
  (B G : ℕ)
  (ratio_condition : B * 8 = 5 * G)
  (total_condition : B + G = 260) :
  G = 160 :=
by
  sorry

end number_of_girls_l19_19357


namespace arithmetic_sequence_sum_l19_19248

noncomputable theory
open_locale classical

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (h : arithmetic_sequence a) (h5 : a 5 = 21) : 
  a 4 + a 5 + a 6 = 63 :=
by
  sorry

end arithmetic_sequence_sum_l19_19248


namespace true_propositions_l19_19484

theorem true_propositions :
  (∀ x y, (x * y = 1 → x * y = (x * y))) ∧
  (¬ (∀ (a b : ℝ), (∀ (A B : ℝ), a = b → A = B) ∧ (A = B → a ≠ b))) ∧
  (∀ m : ℝ, (m ≤ 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0)) ↔
    (true ∧ true ∧ true) :=
by sorry

end true_propositions_l19_19484


namespace intersecting_circle_radius_l19_19785

-- Definitions representing the conditions
def non_intersecting_circles (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) : Prop :=
  ∀ i j, i ≠ j → dist (O_i i) (O_i j) ≥ r_i i + r_i j

def min_radius_one (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) := 
  ∀ i, r_i i ≥ 1

-- The main theorem stating the proof goal
theorem intersecting_circle_radius 
  (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) (O : ℕ) (r : ℝ)
  (h_non_intersecting : non_intersecting_circles O_i r_i)
  (h_min_radius : min_radius_one O_i r_i)
  (h_intersecting : ∀ i, dist O (O_i i) ≤ r + r_i i) :
  r ≥ 1 := 
sorry

end intersecting_circle_radius_l19_19785


namespace transformed_sin_to_cos_l19_19317

noncomputable def transformed_graph (x : ℝ) : ℝ := sin (2 * (x + π / 4))

theorem transformed_sin_to_cos (x : ℝ) : transformed_graph x = cos (2 * x) :=
by
  exact sorry

end transformed_sin_to_cos_l19_19317


namespace simplify_expression_l19_19563

theorem simplify_expression :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end simplify_expression_l19_19563


namespace find_r_l19_19270

theorem find_r (r : ℚ) (h1 : 16 = 2^(7 * r - 3)) : r = 1 :=
sorry

end find_r_l19_19270


namespace path_count_4_4_equals_37_l19_19211

def numberOfPaths (start : (ℕ × ℕ)) (end : (ℕ × ℕ)) : ℕ :=
  sorry

theorem path_count_4_4_equals_37 :
  numberOfPaths (0, 0) (4, 4) = 37 :=
by
  sorry

end path_count_4_4_equals_37_l19_19211


namespace rahim_books_payment_l19_19725

theorem rahim_books_payment (books_first_shop : ℕ) (books_second_shop : ℕ) 
(amount_second_shop : ℝ) (avg_price_per_book : ℝ) :
books_first_shop = 42 →
books_second_shop = 22 →
amount_second_shop = 248 →
avg_price_per_book = 12 →
(books_first_shop + books_second_shop) * avg_price_per_book - amount_second_shop = 520 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    (42 + 22) * 12 - 248 = 64 * 12 - 248 : by rw [add_mul]
                   ... = 768 - 248     : by norm_num
                   ... = 520           : by norm_num
  sorry

end rahim_books_payment_l19_19725


namespace max_y_coordinate_l19_19950

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l19_19950


namespace sum_of_factors_2002_value_of_b_value_of_c_value_of_d_l19_19625

-- G4.1: Sum of all positive factors of 2002
theorem sum_of_factors_2002 : 
  let a := (3 * 8 * 12 * 14 : ℕ) in a = 4032 :=
by
  let a := 2002
  sorry

-- G4.2: Solving the given condition with x > 0 and y > 0
theorem value_of_b :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (sqrt x * (sqrt x + sqrt y) = 3 * sqrt y * (sqrt x + 5 * sqrt y)) →
  let b := (2 * x + sqrt (x * y) + 3 * y) / (x + sqrt (x * y) - y) in b = 2 :=
by
  intros x y x_pos y_pos h
  let b := (2 * x + sqrt (x * y) + 3 * y) / (x + sqrt (x * y) - y)
  sorry

-- G4.3: Given the equation ||x-2|-1|=c has only 3 integral solutions
theorem value_of_c :
  ∃ c : ℝ, ∀ x : ℤ, (abs (abs (x - 2) - 1) = c) →
  (number of integral solutions) = 3 → c = 1 :=
by
  intros c x h_num_sol
  sorry

-- G4.4: Positive real root of the given equation
theorem value_of_d :
  let f (x : ℝ) := 0.5 * (0.5 * (0.5 * (0.5 * x^2 + 2) + 2) + 2) in
  ∃ d : ℝ, f d = 2 ∧ d > 0 → d = 2 :=
by
  intros d h
  let f := fun x => 0.5 * (0.5 * (0.5 * (0.5 * x^2 + 2) + 2) + 2)
  sorry

end sum_of_factors_2002_value_of_b_value_of_c_value_of_d_l19_19625


namespace H_on_angle_bisector_l19_19752

variables {α : Type*} [ordered_field α] {A B C A1 B1 C1 H : α}

-- Conditions 
def is_right_triangle (A B C : α) : Prop := sorry

def inscribed_circle_touches (A B C A1 B1 C1 : α) : Prop := sorry

def altitude_in_triangle (A1 B1 C1 B1h : α) : Prop := sorry

-- Statement of the theorem
theorem H_on_angle_bisector (h_triangle : is_right_triangle A B C) 
  (h_inscribed : inscribed_circle_touches A B C A1 B1 C1)
  (h_altitude : altitude_in_triangle A1 B1 C1 H) :
  lies_on_angle_bisector H (angle_bisector (∠ A B C)) :=
sorry

end H_on_angle_bisector_l19_19752


namespace cos_C_in_triangle_eq_one_fifth_l19_19257

theorem cos_C_in_triangle_eq_one_fifth
  {A B C : ℝ}
  {a b c : ℝ}
  (h1 : a = sin A)
  (h2 : b = sin B)
  (h3 : c = sin C)
  (h4 : a * cos B + b * cos A = 5 * c * cos C) :
  cos C = 1 / 5 := 
sorry

end cos_C_in_triangle_eq_one_fifth_l19_19257


namespace meadow_to_campsite_distance_l19_19676

variable (d1 d2 d_total d_meadow_to_campsite : ℝ)

theorem meadow_to_campsite_distance
  (h1 : d1 = 0.2)
  (h2 : d2 = 0.4)
  (h_total : d_total = 0.7)
  (h_before_meadow : d_before_meadow = d1 + d2)
  (h_distance : d_meadow_to_campsite = d_total - d_before_meadow) :
  d_meadow_to_campsite = 0.1 :=
by 
  sorry

end meadow_to_campsite_distance_l19_19676


namespace chord_length_intercepted_by_line_l19_19351

theorem chord_length_intercepted_by_line
  (a : ℝ)
  (h : (∀ x y : ℝ, x^2 + y^2 - 2*a*x + a = 0 → ax + y + 1 = 0 → 2 = 2)) :
  a = -2 := by
  have simplified_eq : (∀ x y : ℝ, (x - a)^2 + y^2 = a^2 - a) := sorry
  have center_r : ((a, 0), Math.sqrt (a^2 - a)) := sorry
  have distance_from_center_to_line : Math.sqrt (a^2 + 1) := sorry
  have chord_length : a^2 + 1 + 1 = a^2 - a := sorry
  exact sorry

end chord_length_intercepted_by_line_l19_19351


namespace simplify_expression_eval_l19_19327

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  ((x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2 * x + 1)))

theorem simplify_expression_eval : simplify_expression (Real.sqrt 5 - 1) = (Real.sqrt 5) / 5 :=
by
  sorry

end simplify_expression_eval_l19_19327


namespace points_on_circle_distance_l19_19009

noncomputable def distance (x y : ℝ) : ℝ := abs ((3 : ℝ) * x + 4 * y - 11) / real.sqrt ((3 : ℝ) ^ 2 + (4 : ℝ) ^ 2)

theorem points_on_circle_distance :
  let circle_eq := ((λ (x y : ℝ), (x - 3) ^ 2 + (y -3) ^ 2 = 9) : ℝ → ℝ → Prop),
      line_eq := (λ (x y : ℝ), 3 * x + 4 * y - 11 = 0),
      dist (x y : ℝ) := abs (3 * x + 4 * y - 11) / (real.sqrt (3 ^ 2 + 4 ^ 2))
  in (∀ x y : ℝ, circle_eq x y → dist x y = 1) → ∃! p q : ℝ, circle_eq p q ∧ dist p q = 1 :=
begin
  sorry
end

end points_on_circle_distance_l19_19009


namespace robert_ate_more_chocolates_l19_19821

theorem robert_ate_more_chocolates :
  (let robert_chocolates := 7) →
  (let nickel_chocolates := 3) →
  robert_chocolates - nickel_chocolates = 4 :=
by
  intros robert_chocolates nickel_chocolates
  sorry

end robert_ate_more_chocolates_l19_19821


namespace canteen_is_equidistant_l19_19086

noncomputable def distance {α : Type*} [metric_space α] (a b : α) := dist a b

structure CampsDistance :=
  (AG BG : ℝ)

def girls_camp_distance {d : CampsDistance} : ℝ := d.AG
def boys_camp_distance {d : CampsDistance} : ℝ := d.BG

/-- 
  Function that checks if a canteen equidistant from both camps exists 
  and returns the distance if such canteen exists.
-/
def canteen_distance (d : CampsDistance) : ℝ :=
  let AG := girls_camp_distance d
  let BG := boys_camp_distance d
  (BG^2 + AG^2) / (2 * BG)

/-- 
  Theorem: Given a girls' camp located 400 rods from the road and 
  a boys' camp located 600 rods along the road from the girls' camp,
  there exists a canteen on the road equidistant from both camps, 
  and this distance is 410 rods.
-/
theorem canteen_is_equidistant : 
  ∀ d : CampsDistance, d.AG = 400 → d.BG = 500 → canteen_distance d = 410 :=
by
  intros d h1 h2
  sorry

end canteen_is_equidistant_l19_19086


namespace original_radius_of_cylinder_in_inches_l19_19224

theorem original_radius_of_cylinder_in_inches
  (r : ℝ) (h : ℝ) (V : ℝ → ℝ → ℝ → ℝ) 
  (h_increased_radius : V (r + 4) h π = V r (h + 4) π) 
  (h_original_height : h = 3) :
  r = 8 :=
by
  sorry

end original_radius_of_cylinder_in_inches_l19_19224


namespace passengers_taken_second_station_l19_19872

def initial_passengers : ℕ := 288
def passengers_dropped_first_station : ℕ := initial_passengers / 3
def passengers_after_first_station : ℕ := initial_passengers - passengers_dropped_first_station
def passengers_taken_first_station : ℕ := 280
def total_passengers_after_first_station : ℕ := passengers_after_first_station + passengers_taken_first_station
def passengers_dropped_second_station : ℕ := total_passengers_after_first_station / 2
def passengers_left_after_second_station : ℕ := total_passengers_after_first_station - passengers_dropped_second_station
def passengers_at_third_station : ℕ := 248

theorem passengers_taken_second_station : 
  ∃ (x : ℕ), passengers_left_after_second_station + x = passengers_at_third_station ∧ x = 12 :=
by 
  sorry

end passengers_taken_second_station_l19_19872


namespace find_area_of_triangle_l19_19755

noncomputable def area_of_equilateral_triangle (P A B C : Point) 
  (PA PB PC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) 
  (is_equilateral : equilateral_triangle A B C) : ℝ :=
  9 + (25 * Real.sqrt 3) / 4

theorem find_area_of_triangle (P A B C : Point) 
  (PA PB PC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) 
  (is_equilateral : equilateral_triangle A B C) :
  area_of_equilateral_triangle P A B C PA PB PC hPA hPB hPC is_equilateral = 
    9 + (25 * Real.sqrt 3) / 4 :=
sorry

-- Definitions for the required structures
structure Point := 
mk :: (x : ℝ) (y : ℝ)

def equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

end find_area_of_triangle_l19_19755


namespace manufacturer_price_l19_19455

theorem manufacturer_price (M : ℝ) (h₀ : 0 < M)
  (h₁ : ∃ d₁ : ℝ, d₁ ∈ (0.10:ℝ) .. 0.30 ∧ ∃ d₂ : ℝ, d₂ = 0.20)
  (h₂ : ∃ P : ℝ, P = 22.40) :
  ∃ M : ℝ, P = 0.56 * M ↔ M = 40 :=
by
  sorry

end manufacturer_price_l19_19455


namespace divisor_count_leq_2_sqrt_l19_19722

theorem divisor_count_leq_2_sqrt {n : ℕ} (h_pos : 0 < n) :
  (∀ d : ℕ, d ∣ n → ∃ e : ℕ, e ∣ n ∧ d * e = n) ∧ 
  (∀ d : ℕ, d ∣ n → d ≤ sqrt n ∨ sqrt n < d) → 
  (nat.totient n ≤ 2 * sqrt n) :=
sorry

end divisor_count_leq_2_sqrt_l19_19722


namespace cos_double_angle_transform_l19_19163

theorem cos_double_angle_transform (α : ℝ) (h : sin (α - π / 3) = 2 / 3) : cos (2 * α + π / 3) = -1 / 9 := 
sorry

end cos_double_angle_transform_l19_19163


namespace evaluate_log_limit_l19_19220

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem evaluate_log_limit (x : ℝ) (hx : 0 < x) : 
  filter.tendsto (λ x, log2 (5 * x + 2) - log2 (x + 4)) filter.at_top (𝓝 (log2 5)) :=
sorry

end evaluate_log_limit_l19_19220


namespace sum_of_angles_is_equal_l19_19980

variable {Point : Type}
variable {Angle : Type}
variable [AddGroup Angle] [CommGroup Angle] {deg : Angle → Lean.expr} {rad : Angle → Lean.expr}

variables (E L M I : Point)
variables (∠ : Point → Point → Point → Angle)
variable (deg180 : 180)

variables 
  (h1 : ∠ E L M + ∠ E M I = deg180)
  (h2 : dist E L = dist E I + dist L M)

theorem sum_of_angles_is_equal 
  (E L M I : Point)
  (∠ : Point → Point → Point → Angle)
  (deg180 : 180)
  (h1 : ∠ E L M + ∠ E M I = deg180)
  (h2 : dist E L = dist E I + dist L M) :
  ∠ L E M + ∠ E M I = ∠ M I E := 
sorry

end sum_of_angles_is_equal_l19_19980


namespace bisect_segment_XY_l19_19488

noncomputable def midpoint_of_arc (A B C : Point) (w : Circle) : Point := sorry
noncomputable def intersection_of_line_and_tangent (D A : Point) (w : Circle) (tangent_B tangent_C : Line) : (Point × Point) := sorry
noncomputable def intersection_of_two_lines (l1 l2 : Line) : Point := sorry

-- Define the geometric setup
variable (A B C D P Q X Y T : Point)
variable (w : Circle)
variable (tangent_B tangent_C : Line)

-- Assume given conditions
axiom H1 : D = midpoint_of_arc A B C w
axiom H2 : (P, Q) = intersection_of_line_and_tangent D A w tangent_B tangent_C
axiom H3 : X = intersection_of_two_lines (line_from_points B Q) (line_from_points A C)
axiom H4 : Y = intersection_of_two_lines (line_from_points C P) (line_from_points A B)
axiom H5 : T = intersection_of_two_lines (line_from_points B Q) (line_from_points C P)

-- The theorem to be proven
theorem bisect_segment_XY : bisects (line_from_points A T) (segment_from_points X Y) :=
sorry

end bisect_segment_XY_l19_19488


namespace part_a_part_b_l19_19337

open Real

-- Define the conditions for the set of real numbers x, y, z
variables (x y z : ℝ)
hypothesis (h_sum : x + y + z = 12)
hypothesis (h_squares : x^2 + y^2 + z^2 = 54)

-- Part (a): Prove that each of the products xy, yz, zx are between 9 and 25 inclusive.
theorem part_a : (9 ≤ x * y ∧ x * y ≤ 25) ∧ (9 ≤ y * z ∧ y * z ≤ 25) ∧ (9 ≤ z * x ∧ z * x ≤ 25) := 
by
s 
sorry

-- Part (b): Prove that one of the numbers x, y, z is at most 3 and another one is at least 5.
theorem part_b : ∃ e m ∈ ({x, y, z} : set ℝ), e ≤ 3 ∧ m ≥ 5 :=
by
s 
sorry

end part_a_part_b_l19_19337


namespace complement_of_A_l19_19606

open Set

variable (U : Set ℝ) (A : Set ℝ)

-- Conditions
def universal_set : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Required proof: Complement of A in U
theorem complement_of_A :
  compl A = {x | x ≤ -2} ∪ {x | x > 1} :=
sorry

end complement_of_A_l19_19606


namespace percentage_of_primes_divisible_by_2_l19_19054

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def divisible_by (n k : ℕ) : Prop := k ∣ n

def percentage_divisible_by (k : ℕ) (lst : List ℕ) : ℚ :=
  (lst.filter (divisible_by k)).length / lst.length * 100

theorem percentage_of_primes_divisible_by_2 : 
  percentage_divisible_by 2 primes_less_than_20 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_2_l19_19054


namespace round_24_6374_to_nearest_hundredth_l19_19728

noncomputable def round_to_hundredths (x : ℝ) : ℝ :=
  let scaled := x * 100 in
  if scaled - scaled.floor ≥ 0.5 then (scaled.floor + 1) / 100 else scaled.floor / 100

theorem round_24_6374_to_nearest_hundredth :
  round_to_hundredths 24.6374 = 24.64 :=
by
  sorry

end round_24_6374_to_nearest_hundredth_l19_19728


namespace min_tangent_length_is_4_l19_19629

-- Define the circle and symmetry conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0
def symmetry_condition (a b : ℝ) : Prop := 2*a*(-1) + b*2 + 6 = 0

-- Define the length of the tangent line from (a, b) to the circle center (-1, 2)
def min_tangent_length (a b : ℝ) : ℝ :=
  let d := Real.sqrt ((a + 1)^2 + (b - 2)^2) in
  d - Real.sqrt 2

-- Prove that the minimum tangent length is 4 given the conditions
theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_condition a b →
  ∃ (min_len : ℝ), min_len = min_tangent_length a b ∧ min_len = 4 :=
by
  sorry

end min_tangent_length_is_4_l19_19629


namespace min_value_sigma_range_k_value_l19_19599

-- Part (1)
theorem min_value_sigma :
  (∃ x > 0, (∀ y > 0, y ≠ x → (\ln y + (exp y / y) - y - (e - 1) ≥ 0))) ∧ 
  (∀ x > 0, (\ln x + (exp x / x) - x < e - 1)) :=
sorry

-- Part (2)
theorem range_k_value :
  ∀ (a b : ℝ), (1 / 2 ≤ a) → (a ≤ b) → 
  ((∀ x ∈ set.Icc a b, (x ^ 2 - x * log x + 2 ∈ { k * (x + 2) | k ∈ set.Icc 1 ((9 + 2 * log 2) / 10) })) :=
sorry

end min_value_sigma_range_k_value_l19_19599


namespace prove_ab_l19_19217

theorem prove_ab 
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 :=
by
  sorry

end prove_ab_l19_19217


namespace smallest_x_value_l19_19550

theorem smallest_x_value (x : ℝ) (h : |4 * x + 9| = 37) : x = -11.5 :=
sorry

end smallest_x_value_l19_19550


namespace whale_plankton_feeding_frenzy_l19_19475

theorem whale_plankton_feeding_frenzy
  (x y : ℕ)
  (h1 : x + 5 * y = 54)
  (h2 : 9 * x + 36 * y = 450) :
  y = 4 :=
sorry

end whale_plankton_feeding_frenzy_l19_19475


namespace simplify_exponent_l19_19422

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l19_19422


namespace determine_original_pen_colors_l19_19737

-- Define the types and passengers
inductive Passenger
| Barna
| Fehér
| Fekete
| Kékesi
| Piroska
| Zöld

open Passenger

-- Define the colors
inductive Color
| zöld
| fekete
| fehér
| barna
| kék
| piros

open Color

-- A function that assigns the original pen color to each passenger
constant originalPenColor : Passenger → Color

-- Conditions
axiom no_repeated_initials : True
axiom no_pen_matches_last_name : ∀ p : Passenger, not (originalPenColor p = match p with
  | Barna => zöld
  | Fehér => fehér
  | Fekete => fekete
  | Kékesi => kék
  | Piroska => piros
  | Zöld => zöld)

-- Theorem to be proven
theorem determine_original_pen_colors :
  originalPenColor Barna = zöld ∧
  originalPenColor Fehér = fekete ∧
  originalPenColor Fekete = fehér ∧
  originalPenColor Kékesi = barna ∧
  originalPenColor Piroska = kék ∧
  originalPenColor Zöld = piros :=
sorry

end determine_original_pen_colors_l19_19737


namespace value_of_x_plus_2y_l19_19193

theorem value_of_x_plus_2y (x y : ℝ) (h1 : (x + y) / 3 = 1.6666666666666667) (h2 : 2 * x + y = 7) : x + 2 * y = 8 := by
  sorry

end value_of_x_plus_2y_l19_19193


namespace students_no_A_l19_19241

theorem students_no_A (T AH AM AHAM : ℕ) (h1 : T = 35) (h2 : AH = 10) (h3 : AM = 15) (h4 : AHAM = 5) :
  T - (AH + AM - AHAM) = 15 :=
by
  sorry

end students_no_A_l19_19241


namespace base_number_is_4_l19_19623

theorem base_number_is_4 (some_number : ℕ) (h : 16^8 = some_number^16) : some_number = 4 :=
sorry

end base_number_is_4_l19_19623


namespace number_of_true_propositions_l19_19136

theorem number_of_true_propositions : 
  let original_p := ∀ (a : ℝ), a > -1 → a > -2
  let converse_p := ∀ (a : ℝ), a > -2 → a > -1
  let inverse_p := ∀ (a : ℝ), a ≤ -1 → a ≤ -2
  let contrapositive_p := ∀ (a : ℝ), a ≤ -2 → a ≤ -1
  (original_p ∧ contrapositive_p ∧ ¬converse_p ∧ ¬inverse_p) → (2 = 2) :=
by
  intros
  sorry

end number_of_true_propositions_l19_19136


namespace no_prime_pairs_sum_53_l19_19662

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l19_19662


namespace second_sample_number_is_057_l19_19026

-- Defining the rows of the random number table.
def row_7 : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76]
def row_8 : List ℕ := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

-- Defining the function to extract the second sample number.
def second_sample_number (row: List ℕ) (start : ℕ) : ℕ :=
  (row.drop (start - 1)).nth 1 |> Option.getD 0

-- The theorem we want to prove.
theorem second_sample_number_is_057 : second_sample_number row_7 5 = 57 := sorry

end second_sample_number_is_057_l19_19026


namespace minimum_g_of_tetrahedron_l19_19704

theorem minimum_g_of_tetrahedron :
  ∀ (X : Point),
  (AD = 30 ∧ BC = 30 ∧ AC = 40 ∧ BD = 40 ∧ AB = 50 ∧ CD = 50) →
  let g := λ (X : Point), (distance A X + distance B X + distance C X + distance D X) in
  ∃ (p q : ℕ), g(X) = p * real.sqrt q ∧ q ≠ 0 ∧ (∀ r : ℕ, r*r ≠ q) ∧ p + q = 81 :=
begin
  sorry
end

end minimum_g_of_tetrahedron_l19_19704


namespace no_prime_pairs_sum_53_l19_19658

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l19_19658


namespace circle_numbers_l19_19024

open Function

theorem circle_numbers (a : ℕ → ℕ)
    (h1 : ∀ i : ℕ, 1 ≤ i → i ≤ 30 → a i = |a (i - 1) % 30 - a (i + 1) % 30|)
    (h2 : (Finset.range 30).sum a = 2000) :
    ∀ i, 1 ≤ i → i ≤ 30 → a i = 100 ∨ a i = 0 :=
by sorry

end circle_numbers_l19_19024


namespace sum_of_roots_l19_19218

theorem sum_of_roots (a β : ℝ) 
  (h1 : a^2 - 2 * a = 1) 
  (h2 : β^2 - 2 * β - 1 = 0) 
  (hne : a ≠ β) 
  : a + β = 2 := 
sorry

end sum_of_roots_l19_19218


namespace ab_value_l19_19166

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 :=
by
  sorry

end ab_value_l19_19166


namespace product_slope_y_intercept_l19_19452

theorem product_slope_y_intercept :
  ∀ (m b : ℚ), b = 3 → m = -3 / 2 → m * b = -9 / 2 :=
by
  intros m b h_b h_m
  rw [h_b, h_m]
  norm_num
  sorry

end product_slope_y_intercept_l19_19452


namespace number_of_negative_numbers_in_set_l19_19879

theorem number_of_negative_numbers_in_set :
  ∀ (S : Set ℤ), S = {-3, -2, 0, 5} → S.filter (λ x, x < 0).card = 2 :=
by
  intros
  sorry

end number_of_negative_numbers_in_set_l19_19879


namespace polynomial_roots_bounded_by_degree_l19_19719

theorem polynomial_roots_bounded_by_degree (n : ℕ) (p : polynomial ℂ) (deg_p : p.degree = n) :
  ∃ (s : finset ℂ), (s.card ≤ n) ∧ ∀ z : ℂ, z ∈ s ↔ p.eval z = 0 :=
by
  sorry

end polynomial_roots_bounded_by_degree_l19_19719


namespace simplify_and_evaluate_expr_l19_19325

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l19_19325


namespace max_distance_traveled_l19_19743

def distance_traveled (t : ℝ) : ℝ := 15 * t - 6 * t^2

theorem max_distance_traveled : ∃ t : ℝ, distance_traveled t = 75 / 8 :=
by
  sorry

end max_distance_traveled_l19_19743


namespace decreasing_interval_l19_19760

theorem decreasing_interval:
  ∀ x : ℝ, 0 < x ∧ x < 2  →  deriv (λ x : ℝ, x^3 - 3x^2 + 1) x < 0  :=
by
  intro x hx
  have h_deriv : deriv (λ x : ℝ, x^3 - 3x^2 + 1) x = 3*x^2 - 6*x := by
    calc deriv (λ x : ℝ, x^3 - 3x^2 + 1) x = 3*x^2 - 6*x := by sorry  -- Add computation of the derivative here
  have h_ineq : 3*x^2 - 6*x < 0 := by
    calc 3*x^2 - 6*x < 0 := by sorry  -- Add solution to the inequality here
  exact h_ineq

end decreasing_interval_l19_19760


namespace max_coloring_distance_eq_l19_19989

-- Define the structure of the problem
structure Gon (m n : ℕ) :=
(black_points : fin (2 * m))
(white_points : fin (2 * n))
(coloring_distance_black : (black_points → black_points → ℕ))
(coloring_distance_white : (white_points → white_points → ℕ))
(B_matching : list (fin m × fin m))
(W_matching : list (fin n × fin n))
(PB : ∀ (B : B_matching), Σ i, coloring_distance_black B.1 B.2)
(PW : ∀ (W : W_matching), Σ j, coloring_distance_white W.1 W.2)

-- The statement of the problem
theorem max_coloring_distance_eq (m n : ℕ) (g : Gon m n) : 
  g.PB = g.PW :=
sorry

end max_coloring_distance_eq_l19_19989


namespace cameron_total_questions_l19_19899

theorem cameron_total_questions :
  let questions_per_tourist := 2
  let first_group := 6
  let second_group := 11
  let third_group := 8
  let third_group_special_tourist := 1
  let third_group_special_questions := 3 * questions_per_tourist
  let fourth_group := 7
  let first_group_total_questions := first_group * questions_per_tourist
  let second_group_total_questions := second_group * questions_per_tourist
  let third_group_total_questions := (third_group - third_group_special_tourist) * questions_per_tourist + third_group_special_questions
  let fourth_group_total_questions := fourth_group * questions_per_tourist
  in first_group_total_questions + second_group_total_questions + third_group_total_questions + fourth_group_total_questions = 68 := by
  sorry

end cameron_total_questions_l19_19899


namespace bounded_function_inequality_bounded_function_equality_l19_19287

theorem bounded_function_inequality (f : ℝ → ℝ) (D : set ℝ) (M : ℝ) (hM : M ∈ ℝ+)
  (hf : ∀ x ∈ D, |f x| ≤ M) (n : ℕ) (h_n : n ≥ 1) (x : fin n → ℝ) (hx : ∀ i, x i ∈ D) :
  (n - 1) * M^n + ∏ i, f(x i) ≥ M^(n - 1) * ∑ i, f(x i) :=
sorry

theorem bounded_function_equality (f : ℝ → ℝ) (D : set ℝ) (M : ℝ) (hM : M ∈ ℝ+)
  (hf : ∀ x ∈ D, |f x| ≤ M) (n : ℕ) (h_n : n ≥ 1) (x : fin n → ℝ) (hx : ∀ i, x i ∈ D) :
  ((n - 1) * M^n + ∏ i, f(x i) = M^(n - 1) * ∑ i, f(x i)) ↔ (∀ i, f (x i) = M) :=
sorry

end bounded_function_inequality_bounded_function_equality_l19_19287


namespace cone_volume_divided_by_pi_l19_19446

noncomputable def volume_of_cone_divided_by_pi (r : ℝ) (angle : ℝ) : ℝ :=
  if angle = 270 ∧ r = 20 then
    let base_circumference := 30 * Real.pi in
    let base_radius := 15 in
    let slant_height := r in
    let height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
    let volume := (1 / 3) * Real.pi * base_radius ^ 2 * height in
    volume / Real.pi
  else 0

theorem cone_volume_divided_by_pi : 
  volume_of_cone_divided_by_pi 20 270 = 375 * Real.sqrt 7 :=
by
  sorry

end cone_volume_divided_by_pi_l19_19446


namespace sqrt_expression_eq_1720_l19_19120

theorem sqrt_expression_eq_1720 : Real.sqrt ((43 * 42 * 41 * 40) + 1) = 1720 := by
  sorry

end sqrt_expression_eq_1720_l19_19120


namespace cameron_answers_l19_19892

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end cameron_answers_l19_19892


namespace lower_circle_radius_153_l19_19067

theorem lower_circle_radius_153 :
  ∀ (dist_parallel_lines square_side_length upper_circle_radius lower_circle_radius : ℝ),
  dist_parallel_lines = 400 →
  square_side_length = 279 →
  upper_circle_radius = 65 →
  -- Assume the geometric constraints as detailed.
  (lower_circle_radius = 153) :=
by {
  intros,
  sorry -- proof to be completed
}

end lower_circle_radius_153_l19_19067


namespace expression_A_expression_B_expression_C_expression_D_l19_19481

theorem expression_A :
  (Real.sin (7 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) + 
   Real.sin (83 * Real.pi / 180) * Real.cos (67 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_B :
  (2 * Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_C :
  (Real.sqrt 3 * Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) / 
   Real.sin (50 * Real.pi / 180) ≠ 1 / 2 :=
sorry

theorem expression_D :
  (1 / ((1 + Real.tan (27 * Real.pi / 180)) * (1 + Real.tan (18 * Real.pi / 180)))) = 1 / 2 :=
sorry

end expression_A_expression_B_expression_C_expression_D_l19_19481


namespace factorial_division_l19_19508

theorem factorial_division : 11! / 10! = 11 := 
by sorry

end factorial_division_l19_19508


namespace least_divisible_perfect_square_l19_19863

def divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem least_divisible_perfect_square : ∃ n : ℕ,
  divisible_by n 4 ∧
  divisible_by n 5 ∧
  divisible_by n 7 ∧
  divisible_by n 8 ∧
  is_perfect_square n ∧
  n = 19600 :=
begin
  sorry
end

end least_divisible_perfect_square_l19_19863


namespace initial_average_daily_production_l19_19965

theorem initial_average_daily_production
  (n : ℕ)
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (initial_avg_daily_production : ℕ) :
  n = 14 →
  today_production = 90 →
  new_avg_production = 62 →
  14 * initial_avg_daily_production + today_production = 15 * new_avg_production →
  initial_avg_daily_production = 60 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end initial_average_daily_production_l19_19965


namespace counting_correct_statements_l19_19984

variable {α : Type*}

-- Definition for sequence and its sum.
def seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = ∑ i in finset.range (n + 1), a i

-- The main theorem stating the number of correct statements.
theorem counting_correct_statements (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (inc_a : ∀ n, a (n + 1) ≥ a n) 
  (arith_seq : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d) 
  (geom_seq: ∃ r, ∀ n, a (n + 1) = a n * r) :
  seq a S →
  let st1 := ¬ ( ∀ n, (a (n + 1) ≥ a n) → (S (n + 1) ≥ S n) )
  let st2 := ¬ ( (∀ n, S (n + 1) ≥ S n) ↔ (∀ n, a n > 0) )
  let st3 := ¬ ( arith_seq → ( (S 1 * S 2 * S 3 * S 4 * S k = 0) ↔ (a 1 * a 2 * a 3 * a 4 * a k = 0)) )
  let st4 := ( geom_seq → ( (S 1 * S 2 * S 3 * S 4 * k = 0) ↔ (a n + a (n + 1) = 0)) ) in
  1 = [st1, st2, st3, st4].count(true) :=
sorry

end counting_correct_statements_l19_19984


namespace original_wattage_l19_19088

theorem original_wattage (W : ℝ) (new_W : ℝ) (h1 : new_W = 1.25 * W) (h2 : new_W = 100) : W = 80 :=
by
  sorry

end original_wattage_l19_19088


namespace complex_number_real_l19_19342

theorem complex_number_real (a : ℝ) : 
  let Z := (1 : ℂ) + complex.I * (1 - a) in
  (imag_part Z = 0) → a = 1 :=
by
  -- Definitions and conditions
  let Z := (1 : ℂ) + complex.I * (1 - a)
  intro h

  -- Proof (skipped)
  sorry

end complex_number_real_l19_19342


namespace average_death_rate_l19_19648

-- Definitions and given conditions
def birth_rate_per_two_seconds := 6
def net_increase_per_day := 172800

-- Calculate number of seconds in a day as a constant
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the net increase per second
def net_increase_per_second : ℕ := net_increase_per_day / seconds_per_day

-- Define the birth rate per second
def birth_rate_per_second : ℕ := birth_rate_per_two_seconds / 2

-- The final proof statement
theorem average_death_rate : 
  ∃ (death_rate_per_two_seconds : ℕ), 
    death_rate_per_two_seconds = birth_rate_per_two_seconds - 2 * net_increase_per_second := 
by 
  -- We are required to prove this statement
  use (birth_rate_per_second - net_increase_per_second) * 2
  sorry

end average_death_rate_l19_19648


namespace part_a_part_b_part_c_part_d_part_e_part_f_part_g_l19_19114

-- Defining the Bernoulli random walk and necessary conditions
variable {n N : ℕ}
variable {S : ℕ → ℤ}

-- Bernoulli random walk conditions
axiom bernoulli_walk : ∀ k, S 0 = 0 ∧ (S k = ∑ i in range k, if i ≤ n then 1 else -1)

-- Theorems about probabilities of the Bernoulli walk
theorem part_a : 
  (p {k | 1 ≤ k ∧ k ≤ n ∧ S k ≥ N ∧ S n < N}) = (p {S n > N}) := sorry

theorem part_b : 
  (p {k | 1 ≤ k ∧ k ≤ n ∧ S k ≥ N}) = 2 * (p {S n ≥ N}) - (p {S n = N}) := sorry

theorem part_c : 
  (p {k | 1 ≤ k ∧ k ≤ n ∧ S k = N}) = 2^(-n) * binom n ((n + N + 1) / 2) := sorry

theorem part_d : 
  (p {k | 1 ≤ k ∧ k ≤ n ∧ S k ≤ 0}) = 2^(-n) * binom n (n / 2) := sorry

theorem part_e : 
  (p {k | 1 ≤ k ∧ k < n ∧ S k ≤ 0 ∧ S n > 0}) = (p {S k ≠ 0 ∧ S (k + 1) = 0}) := sorry

theorem part_f : 
  (p {k | 1 ≤ k ∧ k < 2*n ∧ S k > 0 ∧ S (2*n) = 0}) = (1 / n) * 2^(-2*n) * binom (2*n-2) (n-1) := sorry

theorem part_g : 
  (p {k | 1 ≤ k ∧ k < 2*n ∧ S k ≥ 0 ∧ S (2*n) = 0}) = (1 / (n+1)) * 2^(-2*n) * binom (2*n) n := sorry

end part_a_part_b_part_c_part_d_part_e_part_f_part_g_l19_19114


namespace pythagorean_triplets_l19_19129

theorem pythagorean_triplets (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ d p q : ℤ, a = 2 * d * p * q ∧ b = d * (q^2 - p^2) ∧ c = d * (p^2 + q^2) := sorry

end pythagorean_triplets_l19_19129


namespace monotonic_decreasing_interval_l19_19007

-- Define the quadratic function t
def t (x : ℝ) := x^2 - 3 * x + 2

-- Define the logarithmic function with base 1/2
def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Define the composite function y
def y (x : ℝ) := log_half (t x)

-- Stating the proof problem
theorem monotonic_decreasing_interval :
  ∀ x, (2 < x → -∞ < x → y x) = (t x) < 0  :=
by 
  sorry

end monotonic_decreasing_interval_l19_19007


namespace no_prime_pair_summing_to_53_l19_19657

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l19_19657


namespace exists_club_with_two_thirds_l19_19242

-- Assuming the class has 'n' students and denoting the set of students by 'Students'
variable (Students : Type) [Fintype Students] (n : ℕ) [Fintype.card Students = n]

-- Condition 1: Any two students share at least one common club.
variable (clubs : Students → Finset (Fin 3)) -- each student can be in up to 3 clubs (at most two).

-- Helper function to check membership.
def in_common_club (s1 s2 : Students) : Prop :=
  ∃ c, c ∈ clubs s1 ∧ c ∈ clubs s2

-- Condition 2: Each student is a member of at most two clubs.
variable (h_club_limit : ∀ s, (clubs s).card ≤ 2)

-- Prove that there exists an extracurricular activity attended by at least (2/3) of the class.
theorem exists_club_with_two_thirds :
  ∃ c, (∃ t, t ∈ clubs) ∧ (Finset.card (Finset.filter (λ s => c ∈ clubs s) Finset.univ) ≥ n * 2 / 3) :=
sorry

end exists_club_with_two_thirds_l19_19242


namespace lambda_value_l19_19188

-- Given definitions
def vecAB : ℝ × ℝ := (3, 1)
def veca (λ : ℝ) : ℝ × ℝ := (2, λ)

-- Condition of parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Theorem to prove
theorem lambda_value (λ : ℝ) (h : parallel (veca λ) vecAB) : λ = 2 / 3 :=
by
  sorry

end lambda_value_l19_19188


namespace cost_of_paving_l19_19063

theorem cost_of_paving (L W R : ℝ) (hL : L = 6.5) (hW : W = 2.75) (hR : R = 600) : 
  L * W * R = 10725 := by
  rw [hL, hW, hR]
  -- To solve the theorem successively
  -- we would need to verify the product of the values
  -- given by the conditions.
  sorry

end cost_of_paving_l19_19063


namespace pages_left_after_all_projects_l19_19354

-- Definitions based on conditions
def initial_pages : ℕ := 120
def pages_for_science : ℕ := (initial_pages * 25) / 100
def pages_for_math : ℕ := 10
def pages_after_science_and_math : ℕ := initial_pages - pages_for_science - pages_for_math
def pages_for_history : ℕ := (initial_pages * 15) / 100
def pages_after_history : ℕ := pages_after_science_and_math - pages_for_history
def remaining_pages : ℕ := pages_after_history / 2

theorem pages_left_after_all_projects :
  remaining_pages = 31 :=
  by
  sorry

end pages_left_after_all_projects_l19_19354


namespace cone_volume_divided_by_pi_l19_19436

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l19_19436


namespace exists_positive_C_l19_19723

theorem exists_positive_C (C : ℝ) :
  ∃ C > 0, ∀ (n : ℤ) (X : Set ℤ), n ≥ 2 ∧ (|X| ≥ 2) → 
  let α := (|X| / n : ℝ)
  in ∀ x y z w ∈ X, 0 < |(x * y) - (z * w)| < C * α ^ (-4) :=
sorry

end exists_positive_C_l19_19723


namespace trigonometric_identity_l19_19713

open Real

theorem trigonometric_identity (α : ℝ) : 
  sin α * sin α + cos (π / 6 + α) * cos (π / 6 + α) + sin α * cos (π / 6 + α) = 3 / 4 :=
sorry

end trigonometric_identity_l19_19713


namespace infinite_zeros_of_sin_exp_l19_19521

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.exp x)

theorem infinite_zeros_of_sin_exp :
  ∀ x : ℝ, -∞ < x → x < 0 → ∃ (a : ℕ), a > 0 ∧ f x = 0 :=
by
  sorry

end infinite_zeros_of_sin_exp_l19_19521


namespace coeff_x4_in_expansion_l19_19670

theorem coeff_x4_in_expansion : 
  (∀ (x : ℝ), (x^2 + (2/x))^5 = (∑ r in finset.range 6, (binomial 5 r) * 2^r * x^(10 - 3*r)) → 
    ∃ c : ℝ, (x^4 = c * x^4) ∧ c = 40) :=
by 
  sorry

end coeff_x4_in_expansion_l19_19670


namespace largest_prime_factor_of_85_l19_19812

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_85 :
  let a := 65
  let b := 85
  let c := 91
  let d := 143
  let e := 169
  largest_prime_factor b 17 :=
by
  sorry

end largest_prime_factor_of_85_l19_19812


namespace problem_convex_quadrilateral_l19_19653

theorem problem_convex_quadrilateral (A B C D : Point)
  (h1 : ∠A = ∠C)
  (h2 : dist A B = 200)
  (h3 : dist C D = 200)
  (h4 : dist A D ≠ dist B C)
  (h5 : dist A B + dist B C + dist C D + dist D A = 700) :
  ⌊1000 * (cos ∠C)⌋ = 750 := 
sorry

end problem_convex_quadrilateral_l19_19653


namespace price_decrease_is_correct_l19_19885

-- Define the initial conditions
def original_price_per_pack : ℝ := 9 / 6
def new_price_per_pack : ℝ := 10 / 8

-- Calculate the percent decrease
def percent_decrease : ℝ :=
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100

-- The main theorem stating that the percent decrease is 16.67%
theorem price_decrease_is_correct :
  percent_decrease = 16.67 :=
by
  -- Here the calculations would be shown in a full proof, but we skip it with sorry
  sorry

end price_decrease_is_correct_l19_19885


namespace smallest_5digit_number_multiple_of_2014_with_cde_has_16_divisors_l19_19831

/-- Defining a condition that a number has exactly 16 divisors -/
def has_exactly_16_divisors (n : ℕ) : Prop :=
  (finset.divisors n).card = 16

/-- Defining the main theorem that establishes the stated conditions and conclusion -/
theorem smallest_5digit_number_multiple_of_2014_with_cde_has_16_divisors :
  ∃ ABCDE : ℕ, ABCDE < 100000 ∧ 10000 ≤ ABCDE ∧ 
  ABCDE % 2014 = 0 ∧ has_exactly_16_divisors (ABCDE % 1000) ∧ ABCDE = 24168 :=
sorry

end smallest_5digit_number_multiple_of_2014_with_cde_has_16_divisors_l19_19831


namespace rational_sum_inequality_l19_19285

noncomputable theory

open_locale classical

variables {α : Type*} [linear_ordered_field α]

theorem rational_sum_inequality (x y : ℚ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1)
  (n : ℕ) (a b : fin n → α) (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (∑ i, (a i) ^ x * (b i) ^ y) ≤ (∑ i, a i) ^ x * (∑ i, b i) ^ y :=
sorry

end rational_sum_inequality_l19_19285


namespace sin_double_angle_plus_π_six_l19_19184

theorem sin_double_angle_plus_π_six 
  (α : ℝ)
  (h : sin (α - π / 3) + √3 * cos α = 1 / 3) :
  sin (2 * α + π / 6) = -7 / 9 := 
sorry

end sin_double_angle_plus_π_six_l19_19184


namespace oa_perp_ob_l19_19168

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def curve (x y : ℝ) : Prop := x + 3 * y^2 = 4

theorem oa_perp_ob :
  (∃ A B : ℝ × ℝ, curve A.1 A.2 ∧ curve B.1 B.2 ∧ A ≠ B ∧ 
  ∃ l : ℝ → ℝ × ℝ, ∀ t : ℝ, l t = A ∨ l t = B) →
  (O : ℝ × ℝ) →
  circle O.1 O.2 →
  ∀ A B : ℝ × ℝ, (A ∨ B) ∧ (A ∨ B) → (A - O) ⬝ (B - O) = 0 :=
sorry

end oa_perp_ob_l19_19168


namespace smallest_number_l19_19386

theorem smallest_number (n : ℕ) : (∀ d ∈ [8, 14, 26, 28], (n - 18) % d = 0) → n = 746 := by
  sorry

end smallest_number_l19_19386


namespace distance_between_planes_l19_19131

-- Define planes
def plane1 (x y z : ℝ) : Prop := 3 * x + 2 * y - 6 * z = 12
def plane2 (x y z : ℝ) : Prop := 6 * x + 4 * y - 12 * z = 18

-- Define normal vector for the planes
def n : (ℝ × ℝ × ℝ) := (3, 2, -6)

-- Define a point on the first plane
def point_on_plane1 : ℝ × ℝ × ℝ := (0, 0, -2)

-- Calculate the distance between the point and the second plane
noncomputable def distance_from_point_to_plane : ℝ :=
  (abs (3 * 0 + 2 * 0 - 6 * (-2) - 9)) / (real.sqrt (3^2 + 2^2 + (-6)^2))

-- Statement to prove the distance
theorem distance_between_planes :
  distance_from_point_to_plane = 3 / 7 := by
  sorry

end distance_between_planes_l19_19131


namespace complex_magnitude_conjugate_l19_19573

theorem complex_magnitude_conjugate
    (z : ℂ) (h : z = (3 + I) / (1 - I)) :
    (complex.abs (complex.conj z + 3 * I) = sqrt 2) :=
by
  sorry

end complex_magnitude_conjugate_l19_19573


namespace num_ways_to_select_valid_points_l19_19922

-- Define the properties required for the proof problem
def circle_points : list ℕ := list.range 24

def is_valid_selection (selection : list ℕ) : Prop :=
  selection.length = 8 ∧
  ∀ (x y : ℕ), x ≠ y → x ∈ selection → y ∈ selection → 
    let diff := (y - x + 24) % 24 in 
    diff ≠ 3 ∧ diff ≠ 8

-- Define the main statement for the proof problem
theorem num_ways_to_select_valid_points : ∃ (n : ℕ), n = 6561 ∧
  (∃ selection : list ℕ, is_valid_selection selection) :=
by
  sorry

end num_ways_to_select_valid_points_l19_19922


namespace unique_fraction_l19_19145

theorem unique_fraction : ∃! (x y : ℕ), Nat.Coprime x y ∧ 0 < x ∧ 0 < y ∧ (x + 1) * y = 1.05 * x * (y + 1) := sorry

end unique_fraction_l19_19145


namespace travel_cost_AB_l19_19716

theorem travel_cost_AB
  (distance_AB : ℕ)
  (booking_fee : ℕ)
  (cost_per_km_flight : ℝ)
  (correct_total_cost : ℝ)
  (h1 : distance_AB = 4000)
  (h2 : booking_fee = 150)
  (h3 : cost_per_km_flight = 0.12) :
  correct_total_cost = 630 :=
by
  sorry

end travel_cost_AB_l19_19716


namespace series_converges_to_limit_l19_19124

theorem series_converges_to_limit :
  let a := 3
  let r := (1 / 3 : ℝ)
  has_sum (λ n : ℕ, a * r^n) (9 / 2) :=
begin
  sorry
end

end series_converges_to_limit_l19_19124


namespace convert_1623_to_base7_l19_19914

theorem convert_1623_to_base7 :
  ∃ a b c d : ℕ, 1623 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
  a = 4 ∧ b = 5 ∧ c = 0 ∧ d = 6 :=
by
  sorry

end convert_1623_to_base7_l19_19914


namespace george_elaine_ratio_l19_19886

-- Define the conditions
def time_jerry := 3
def time_elaine := 2 * time_jerry
def time_kramer := 0
def total_time := 11

-- Define George's time based on the given total time condition
def time_george := total_time - (time_jerry + time_elaine + time_kramer)

-- Prove the ratio of George's time to Elaine's time is 1:3
theorem george_elaine_ratio : time_george / time_elaine = 1 / 3 :=
by
  -- Lean proof would go here
  sorry

end george_elaine_ratio_l19_19886


namespace min_varphi_symmetry_l19_19003

theorem min_varphi_symmetry (ϕ : ℝ) (hϕ : ϕ > 0) :
  (∃ k : ℤ, ϕ = (4 * Real.pi) / 3 - k * Real.pi ∧ ϕ > 0 ∧ (∀ x : ℝ, Real.cos (x - ϕ + (4 * Real.pi) / 3) = Real.cos (-x - ϕ + (4 * Real.pi) / 3))) 
  → ϕ = Real.pi / 3 :=
sorry

end min_varphi_symmetry_l19_19003


namespace no_three_diagonals_intersect_at_point_l19_19259

theorem no_three_diagonals_intersect_at_point 
  (H : is_regular_heptagon H) : 
  ¬ ∃ point, (at_least_three_diagonals_intersect_at H point) :=
sorry

end no_three_diagonals_intersect_at_point_l19_19259


namespace no_prime_pair_summing_to_53_l19_19655

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l19_19655


namespace remainder_when_Xn_divided_by_X2_minus_3X_plus_2_l19_19973

theorem remainder_when_Xn_divided_by_X2_minus_3X_plus_2 (n : ℕ) (hn : n ≥ 2) :
  ∃ (R : polynomial ℝ), degree R < 2 ∧ 
  ∀ (X : polynomial ℝ), X^n % (X^2 - 3*X + 2) = R :=
by
  let R := (2^n - 1) * X + (2 - 2^n)
  have hdeg : degree R < 2 := sorry
  use R
  split
  · exact hdeg
  · intro X
    sorry

end remainder_when_Xn_divided_by_X2_minus_3X_plus_2_l19_19973


namespace exists_coprime_prime_l19_19183

open Nat

theorem exists_coprime_prime (a : List ℕ) (h : List.gcd a ≠ 1) :
  ∃ p : ℕ, Nat.prime p ∧ ∀ x ∈ a, Nat.gcd x p = 1 := by
  sorry

end exists_coprime_prime_l19_19183


namespace romeo_profit_l19_19314

def total_revenue : ℕ := 340
def cost_purchasing : ℕ := 175
def cost_packaging : ℕ := 60
def cost_advertising : ℕ := 20

def total_costs : ℕ := cost_purchasing + cost_packaging + cost_advertising
def profit : ℕ := total_revenue - total_costs

theorem romeo_profit : profit = 85 := by
  unfold profit
  unfold total_costs
  simp
  sorry

end romeo_profit_l19_19314


namespace length_AF_four_l19_19707

theorem length_AF_four (O : Circle) (A B C D F : Point) 
(h1 : InscribedTrapezoid A B C D O)
(h2 : Parallel AB CD)
(h3 : TangentAt D O F)
(h4 : Intersects AC F)
(h5 : Parallel DF BC)
(h6 : Length CA = 5)
(h7 : Length BC = 4) :
Length AF = 4 :=
sorry

end length_AF_four_l19_19707


namespace rows_of_triangle_with_cans_l19_19079

theorem rows_of_triangle_with_cans :
  ∃ (n : ℕ), ∑ i in range (n + 1), i = 465 ∧ 480 - 15 = 465 ∧ n = 30 := 
by
  sorry

end rows_of_triangle_with_cans_l19_19079


namespace determine_a_l19_19918

theorem determine_a (p a b c : ℤ) :
  (∃ a b c : ℤ, (λ x : ℤ, (x - a)*(x - 15) + 1) = (λ x : ℤ, (x + b)*(x + c)) ) ↔ a = 13 ∨ a = 17 := 
sorry

end determine_a_l19_19918


namespace cos_alpha_value_l19_19162

noncomputable def cos_alpha (α : ℝ) : ℝ :=
  (3 - 4 * Real.sqrt 3) / 10

theorem cos_alpha_value (α : ℝ) (h1 : Real.sin (Real.pi / 6 + α) = 3 / 5) (h2 : Real.pi / 3 < α ∧ α < 5 * Real.pi / 6) :
  Real.cos α = cos_alpha α :=
by
sorry

end cos_alpha_value_l19_19162


namespace proof_problem_l19_19071

noncomputable def percent_to_decimal := 2.58
def my_value := 1265
def intermediate_result := percent_to_decimal * my_value
def final_result := intermediate_result / 6

theorem proof_problem : final_result ≈ 544.28 := 
by 
  sorry

end proof_problem_l19_19071


namespace max_y_coordinate_l19_19947

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l19_19947


namespace coins_difference_l19_19141

theorem coins_difference (p n d : ℕ) (h1 : p + n + d = 3030)
  (h2 : 1 ≤ p) (h3 : 1 ≤ n) (h4 : 1 ≤ d) (h5 : p ≤ 3029) (h6 : n ≤ 3029) (h7 : d ≤ 3029) :
  (max (p + 5 * n + 10 * d) (max (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) - 
  (min (p + 5 * n + 10 * d) (min (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) = 27243 := 
sorry

end coins_difference_l19_19141


namespace mean_greater_than_median_by_l19_19559

-- Define the data: number of students missing specific days
def studentsMissingDays := [3, 1, 4, 1, 1, 5] -- corresponding to 0, 1, 2, 3, 4, 5 days missed

-- Total number of students
def totalStudents := 15

-- Function to calculate the sum of missed days weighted by the number of students
def totalMissedDays := (0 * 3) + (1 * 1) + (2 * 4) + (3 * 1) + (4 * 1) + (5 * 5)

-- Calculate the mean number of missed days
def meanDaysMissed := totalMissedDays / totalStudents

-- Select the median number of missed days (8th student) from the ordered list
def medianDaysMissed := 2

-- Calculate the difference between the mean and median
def difference := meanDaysMissed - medianDaysMissed

-- Define the proof problem statement
theorem mean_greater_than_median_by : 
  difference = 11 / 15 :=
by
  -- This is where the actual proof would be written
  sorry

end mean_greater_than_median_by_l19_19559


namespace find_r_l19_19277

theorem find_r (a b m p r : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b = 4)
  (h4 : ∀ x : ℚ, x^2 - m * x + 4 = (x - a) * (x - b)) :
  (a - 1 / b) * (b - 1 / a) = 9 / 4 := by
  sorry

end find_r_l19_19277


namespace induction_sequence_step_l19_19037

theorem induction_sequence_step (k : ℕ) :
  1^2 + 2^2 + ... + (k-1)^2 + k^2 + (k-1)^2 + ... + 2^2 + 1^2 + (k+1)^2 + k^2 =
  1^2 + 2^2 + ... + (k-1)^2 + k^2 + (k-1)^2 + ... + 2^2 + 1^2 + ((k + 1)^2 + k^2)
  :=
  sorry

end induction_sequence_step_l19_19037


namespace gear_q_revolutions_per_minute_is_40_l19_19902

-- Definitions corresponding to conditions
def gear_p_revolutions_per_minute : ℕ := 10
def gear_q_revolutions_per_minute (r : ℕ) : Prop :=
  ∃ (r : ℕ), (r * 20 / 60) - (10 * 20 / 60) = 10

-- Statement we need to prove
theorem gear_q_revolutions_per_minute_is_40 :
  gear_q_revolutions_per_minute 40 :=
sorry

end gear_q_revolutions_per_minute_is_40_l19_19902


namespace product_of_roots_l19_19278

theorem product_of_roots :
  let a := 3
  let b := -8
  let c := 5
  let d := -9
  (∀ x : ℝ, 3 * x^3 - 8 * x^2 + 5 * x - 9 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (root_product : -((d : ℝ) / (a : ℝ)) = 3) :=
by
  intro a b c d habc
  have prod_roots := by sorry
  exact prod_roots.habc 

end product_of_roots_l19_19278


namespace smallest_x_solution_l19_19046

theorem smallest_x_solution :
  (∃ x : ℝ, (3 * x^2 + 36 * x - 90 = x * (x + 15)) ∧
              (∀ y : ℝ, (3 * y^2 + 36 * y - 90 = y * (y + 15)) → (-15 ≤ y))) :=
begin
  sorry
end

end smallest_x_solution_l19_19046


namespace fran_speed_l19_19266

-- Definitions for conditions
def joann_speed : ℝ := 15 -- in miles per hour
def joann_time : ℝ := 4 -- in hours
def fran_time : ℝ := 2 -- in hours
def joann_distance : ℝ := joann_speed * joann_time -- distance Joann traveled

-- Proof Goal Statement
theorem fran_speed (hf: fran_time ≠ 0) : (joann_speed * joann_time) / fran_time = 30 :=
by
  -- Sorry placeholder skips the proof steps
  sorry

end fran_speed_l19_19266


namespace earnings_from_jam_l19_19492

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l19_19492


namespace right_triangle_perimeter_l19_19861

theorem right_triangle_perimeter (n : ℕ) (hn : Nat.Prime n) (x y : ℕ) 
  (h1 : y^2 = x^2 + n^2) : n + x + y = n + n^2 := by
  sorry

end right_triangle_perimeter_l19_19861


namespace percentage_of_primes_divisible_by_2_l19_19047

open_locale classical
noncomputable theory

def prime_numbers_less_than_twenty := {p : ℕ | nat.prime p ∧ p < 20}

theorem percentage_of_primes_divisible_by_2 : 
  (card {p ∈ prime_numbers_less_than_twenty | 2 ∣ p}).to_real / (card prime_numbers_less_than_twenty).to_real * 100 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_2_l19_19047


namespace A_squared_infinite_possible_l19_19692

variables {A : Matrix (Fin 2) (Fin 2) ℝ}

theorem A_squared_infinite_possible (h : A^4 = 0) : ∃ b : ℝ, ∃ c : ℝ, ∃ d : ℝ, ∃ (A_squared : Matrix (Fin 2) (Fin 2) ℝ), A_squared = Matrix.vecCons (λ i, Matrix.vecCons (λ j, if (i = 1 ∧ j = 1) then b else (if (i = 2 ∧ j =2) then c else 0))) (λ i, Matrix.vecCons (λ j, if (i = 1 ∧ j = 1) then d else if (i = 2 ∧ j = 2) then b else 0)) :=
sorry

end A_squared_infinite_possible_l19_19692


namespace no_three_digits_all_prime_l19_19710

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that forms a three-digit number from digits a, b, c
def form_three_digit (a b c : ℕ) : ℕ :=
100 * a + 10 * b + c

-- Define a function to check if all permutations of three digits form prime numbers
def all_permutations_prime (a b c : ℕ) : Prop :=
is_prime (form_three_digit a b c) ∧
is_prime (form_three_digit a c b) ∧
is_prime (form_three_digit b a c) ∧
is_prime (form_three_digit b c a) ∧
is_prime (form_three_digit c a b) ∧
is_prime (form_three_digit c b a)

-- The main theorem stating that there are no three distinct digits making all permutations prime
theorem no_three_digits_all_prime : ¬∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  all_permutations_prime a b c :=
sorry

end no_three_digits_all_prime_l19_19710


namespace cone_volume_divided_by_pi_l19_19444

noncomputable def volume_of_cone_divided_by_pi (r : ℝ) (angle : ℝ) : ℝ :=
  if angle = 270 ∧ r = 20 then
    let base_circumference := 30 * Real.pi in
    let base_radius := 15 in
    let slant_height := r in
    let height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
    let volume := (1 / 3) * Real.pi * base_radius ^ 2 * height in
    volume / Real.pi
  else 0

theorem cone_volume_divided_by_pi : 
  volume_of_cone_divided_by_pi 20 270 = 375 * Real.sqrt 7 :=
by
  sorry

end cone_volume_divided_by_pi_l19_19444


namespace regression_line_correct_l19_19578

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def regression_slope (points : List (ℝ × ℝ)) : ℝ :=
  let n := points.length
  let x_bar := mean (points.map Prod.fst)
  let y_bar := mean (points.map Prod.snd)
  let numerator := points.map (fun (x, y) => (x - x_bar) * (y - y_bar)).sum
  let denominator := points.map (fun (x, _) => (x - x_bar)^2).sum
  numerator / denominator

noncomputable def regression_intercept (points : List (ℝ × ℝ)) : ℝ :=
  let x_bar := mean (points.map Prod.fst)
  let y_bar := mean (points.map Prod.snd)
  y_bar - (regression_slope points) * x_bar

noncomputable def regression_line (points : List (ℝ × ℝ)) : ℝ → ℝ :=
  let m := regression_slope points
  let b := regression_intercept points
  fun x => m * x + b

theorem regression_line_correct :
  let points := [(1, 3), (2, 3.8), (3, 5.2), (4, 6)]
  regression_line points = fun x => 1.04 * x + 1.9 :=
by
  let points := [(1, 3), (2, 3.8), (3, 5.2), (4, 6)]
  funext
  sorry

end regression_line_correct_l19_19578


namespace major_axis_length_l19_19095

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.8 * minor_axis) :
  major_axis = 7.2 :=
sorry

end major_axis_length_l19_19095


namespace john_full_steps_l19_19681

theorem john_full_steps : 
  ∃ n : ℕ, (∑ k in Finset.range (n+1), (2 * k + 1)) = 255 ∧ n = 15 :=
begin
  sorry
end

end john_full_steps_l19_19681


namespace jerry_remaining_debt_l19_19264

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l19_19264


namespace liu_xiang_hurdle_distance_and_best_time_l19_19478

noncomputable def total_distance : ℝ := 110
noncomputable def distance_to_first_hurdle : ℝ := 13.72
noncomputable def distance_from_last_hurdle_to_finish : ℝ := 14.02
noncomputable def num_hurdles : ℝ := 10
noncomputable def num_hurdle_cycles : ℝ := num_hurdles - 1  -- 9 spaces between 10 hurdles
noncomputable def best_time_first_segment : ℝ := 2.5
noncomputable def best_time_last_segment : ℝ := 1.4
noncomputable def best_hurdle_cycle_time : ℝ := 0.96

/-- Prove that the distance between two consecutive hurdles is 9.14 meters 
    and the theoretically best time Liu Xiang could achieve in the 110m hurdles is 12.54 seconds 
    given the following conditions. -/
theorem liu_xiang_hurdle_distance_and_best_time :
  let total_hurdle_distance := total_distance - distance_to_first_hurdle - distance_from_last_hurdle_to_finish in
  let distance_between_hurdles := total_hurdle_distance / num_hurdle_cycles in
  let theoretical_best_time := best_time_first_segment + (num_hurdle_cycles * best_hurdle_cycle_time) + best_time_last_segment in
  distance_between_hurdles = 9.14 ∧ theoretical_best_time = 12.54 :=
by {
  -- Definitions
  let total_hurdle_distance :=  total_distance - distance_to_first_hurdle - distance_from_last_hurdle_to_finish,
  let distance_between_hurdles := total_hurdle_distance / num_hurdle_cycles,
  let theoretical_best_time := best_time_first_segment + (num_hurdle_cycles * best_hurdle_cycle_time) + best_time_last_segment,
  
  -- Distances and Times
  have h1 : total_hurdle_distance = 110 - 13.72 - 14.02 := rfl,
  have h2 : distance_between_hurdles = total_hurdle_distance / (10 - 1),
  have h3 : theoretical_best_time = 2.5 + (9 * 0.96) + 1.4 := rfl,
  have h4 : total_hurdle_distance = 82.26 := by norm_num,
  have h5 : distance_between_hurdles = 9.14 := by norm_num,
  have h6 : theoretical_best_time = 12.54 := by norm_num,
  split; assumption
}

end liu_xiang_hurdle_distance_and_best_time_l19_19478


namespace find_PA_values_l19_19727

theorem find_PA_values :
  ∃ P A : ℕ, 10 ≤ P * 10 + A ∧ P * 10 + A < 100 ∧
            (P * 10 + A) ^ 2 / 1000 = P ∧ (P * 10 + A) ^ 2 % 10 = A ∧
            ((P = 9 ∧ A = 5) ∨ (P = 9 ∧ A = 6)) := by
  sorry

end find_PA_values_l19_19727


namespace wholesale_price_l19_19461

theorem wholesale_price (W R SP : ℝ) (h1 : R = 120) (h2 : SP = R - 0.10 * R) (h3 : SP = W + 0.20 * W) : 
  W = 90 :=
by
  -- Given conditions
  have hR : R = 120 := h1
  have hSP_def : SP = 108 := by
    rw [h1] at h2
    norm_num at h2
  -- Solving for W
  sorry

end wholesale_price_l19_19461


namespace part1_part2_l19_19172

noncomputable def a : ℕ+ → ℕ
| ⟨1, _⟩ => 1
| ⟨n+1, _⟩ => 2 * a ⟨n, Nat.succ_pos n⟩ + 3

def b (n : ℕ+) : ℕ := a n + 3

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => a ⟨i+1, Nat.succ_pos i⟩)

theorem part1 (n : ℕ+) : (b (n + 1) = 2 * b n) :=
by
  sorry

theorem part2 (n : ℕ) : S n = 2^n + n*2^n - 2^(n+1) :=
by
  sorry

end part1_part2_l19_19172


namespace quadratic_two_distinct_real_roots_l19_19203

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  2 * k ≠ 0 → (8 * k + 1)^2 - 64 * k^2 > 0 → k > -1 / 16 ∧ k ≠ 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l19_19203


namespace twelve_consecutive_not_square_eleven_consecutive_square_example_l19_19721

-- Definitions related to the first proof problem
def sum_consecutive_12 (k : ℤ) : ℤ :=
  (12 * k) + 78

-- Definitions related to the second example problem
def sum_consecutive_11 (k : ℤ) : ℤ :=
  (11 * k) + 66

-- Proof statement for the first problem: sum of 12 consecutive integers is not a square
theorem twelve_consecutive_not_square (k : ℤ) : 
  ¬ ∃ n : ℤ, sum_consecutive_12 k = n * n :=
by { sorry }

-- Example statement for the second problem: sum of 11 consecutive integers is a perfect square for k = 5
theorem eleven_consecutive_square_example : 
  ∃ n : ℤ, sum_consecutive_11 5 = n * n :=
by { use 11, norm_num }

end twelve_consecutive_not_square_eleven_consecutive_square_example_l19_19721


namespace probability_of_drawing_black_balls_l19_19078

def combination (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_drawing_black_balls :
  let total_balls := 7 + 8 in
  let total_combinations := combination total_balls 2 in
  let black_balls := 8 in
  let black_combinations := combination black_balls 2 in
  (black_combinations : ℚ) / (total_combinations : ℚ) = 4 / 15 :=
by
  sorry

end probability_of_drawing_black_balls_l19_19078


namespace max_y_coordinate_is_three_fourths_l19_19940

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l19_19940


namespace contractor_laborers_l19_19060

theorem contractor_laborers (x : ℕ) (h1 : 15 * x = 20 * (x - 5)) : x = 20 :=
by sorry

end contractor_laborers_l19_19060


namespace marts_income_percentage_of_juans_income_l19_19820

variables (M T J : ℝ)

def M_eq_1_3_T := M = 1.30 * T
def T_eq_0_6_J := T = 0.60 * J

theorem marts_income_percentage_of_juans_income
  (h1 : M_eq_1_3_T M T)
  (h2 : T_eq_0_6_J T J) :
  M = 0.78 * J :=
by
  rw [M_eq_1_3_T, T_eq_0_6_J] at h1 h2
  sorry

end marts_income_percentage_of_juans_income_l19_19820


namespace remainder_of_expression_l19_19803

theorem remainder_of_expression :
  let a := 2^206 + 206
  let b := 2^103 + 2^53 + 1
  a % b = 205 := 
sorry

end remainder_of_expression_l19_19803


namespace equivalent_proof_problem_l19_19889

def base7_to_int (n : ℕ) : ℕ :=
  let digits := [6, 8, 5, 1]  -- digits of 1586_7 in reverse order
  digits.enum_from 0 |>.foldl
    (λ acc ⟨i, x⟩, acc + x * 7 ^ i) 0

def base5_to_int (n : ℕ) : ℕ :=
  let digits := [1, 3, 1]  -- digits of 131_5 in reverse order
  digits.enum_from 0 |>.foldl
    (λ acc ⟨i, x⟩, acc + x * 5 ^ i) 0

def base6_to_int (n : ℕ) : ℕ :=
  let digits := [1, 5, 4, 3]  -- digits of 3451_6 in reverse order
  digits.enum_from 0 |>.foldl
    (λ acc ⟨i, x⟩, acc + x * 6 ^ i) 0

def base7_to_int2 (n : ℕ) : ℕ :=
  let digits := [7, 8, 8, 2]  -- digits of 2887_7 in reverse order
  digits.enum_from 0 |>.foldl
    (λ acc ⟨i, x⟩, acc + x * 7 ^ i) 0

theorem equivalent_proof_problem :
  (base7_to_int 1586) / (base5_to_int 131) - (base6_to_int 3451) + (base7_to_int2 2887) = 334 :=
by
  sorry

end equivalent_proof_problem_l19_19889


namespace perimeter_of_photo_l19_19094

theorem perimeter_of_photo 
  (frame_width : ℕ)
  (frame_area : ℕ)
  (outer_edge_length : ℕ)
  (photo_perimeter : ℕ) :
  frame_width = 2 → 
  frame_area = 48 → 
  outer_edge_length = 10 →
  photo_perimeter = 16 :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end perimeter_of_photo_l19_19094


namespace incorrect_expression_among_options_l19_19395

theorem incorrect_expression_among_options :
  ¬(0.75 ^ (-0.3) < 0.75 ^ (0.1)) :=
by
  sorry

end incorrect_expression_among_options_l19_19395


namespace difference_in_average_speed_l19_19033

theorem difference_in_average_speed:
  ∀ (t : ℕ), 
  (distance : ℕ) (d_TeamA : ℕ) (speed_TeamW : ℕ) (speed_TeamA : ℕ),
  distance = 300 →
  speed_TeamW = 20 →
  distance = speed_TeamW * t →
  d_TeamA = t - 3 →
  speed_TeamA = distance / d_TeamA →
  speed_TeamA - speed_TeamW = 5 :=
by
  sorry

end difference_in_average_speed_l19_19033


namespace duty_pairing_impossible_l19_19966

theorem duty_pairing_impossible :
  ∀ (m n : ℕ), 29 * m + 32 * n ≠ 29 * 32 := 
by 
  sorry

end duty_pairing_impossible_l19_19966


namespace firing_sequence_hits_submarine_l19_19739

theorem firing_sequence_hits_submarine (a b : ℕ) (hb : b > 0) : ∃ n : ℕ, (∃ (an bn : ℕ), (an + bn * n) = a + n * b) :=
sorry

end firing_sequence_hits_submarine_l19_19739


namespace max_y_coordinate_is_three_fourths_l19_19942

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l19_19942


namespace domino_perfect_play_winner_l19_19245

theorem domino_perfect_play_winner :
  ∀ {PlayerI PlayerII : Type} 
    (legal_move : PlayerI → PlayerII → Prop)
    (initial_move : PlayerI → Prop)
    (next_moves : PlayerII → PlayerI → PlayerII → Prop),
    (∀ pI pII, legal_move pI pII) → 
    (∃ m, initial_move m) → 
    (∀ mI mII, next_moves mII mI mII) → 
    ∃ winner, winner = PlayerI :=
by
  sorry

end domino_perfect_play_winner_l19_19245


namespace range_of_a_l19_19214

theorem range_of_a (a : ℝ) (h : sqrt ((1 - 2 * a)^2) = 2 * a - 1) : a ≥ 1 / 2 := sorry

end range_of_a_l19_19214


namespace num_functions_fixed_point_eq_sum_l19_19575

noncomputable def num_functions_fixed_point (m n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), k^(n - k) * nat.choose n k * A m k

theorem num_functions_fixed_point_eq_sum (m n : ℕ) :
  B m n = num_functions_fixed_point m n :=
by sorry

end num_functions_fixed_point_eq_sum_l19_19575


namespace find_k_l19_19206

section linear_system

variables {x y k : ℝ}

-- Given conditions
def eq1 := 2 * x + 3 * y = k
def eq2 := x + 4 * y = k - 16
def condition := x + y = 8

-- The proof goal
theorem find_k (h1 : eq1) (h2 : eq2) (h3 : condition) : k = 12 :=
sorry

end linear_system

end find_k_l19_19206


namespace incenter_on_line_KL_l19_19638

variables {A B C D E F G H J K L O I : Point}
variables {Γ : Circle}

-- Assume the given conditions
axiom triangle_ABC (hABC : Triangle A B C) (Γ : Circle) 
  (tangent_Gamma_AB : Tangent Γ A B D) 
  (tangent_Gamma_AC : Tangent Γ A C E) 
  (BD_plus_CE_less_BC : Length (Segment B D) + Length (Segment C E) < Length (Segment B C)) 
  (BF_equals_BD : Length (Segment B F) = Length (Segment B D)) 
  (CE_equals_CG : Length (Segment C E) = Length (Segment C G))
  (EF_intersects_DG : Intersects (Line E F) (Line D G) K)
  (L_on_arc_DE : OnMinorArc Γ D E L)
  (tangent_L_parallel_BC : Parallel (TangentAt Γ L) (Line B C)).

-- Prove that the incenter I lies on the line KL
theorem incenter_on_line_KL : OnLine (incenter A B C) (Line K L) :=
sorry

end incenter_on_line_KL_l19_19638


namespace probability_at_least_one_vowel_l19_19732

def set1 : Set Char := {'a', 'b', 'c', 'd', 'e'}
def set2 : Set Char := {'k', 'l', 'm', 'n', 'o', 'p'}
def set3 : Set Char := {'r', 's', 't', 'u', 'v'}
def set4 : Set Char := {'w', 'x', 'y', 'z', 'i'}

def vowels (s : Set Char) : Set Char := s.filter (λ c, c ∈ {'a', 'e', 'i', 'o', 'u'})

def isValidVowelCombination (c1 c2 c3 c4 : Char) : Prop :=
  (c1 ∉ {'a', 'e'} ∧ c3 ∉ {'u'} ∧ c4 ∉ {'i'}) ∨
  (c1 = 'a' ∧ c3 = 'u' ∧ c4 ∉ {'i'}) ∨
  (c1 = 'e' ∧ c4 = 'i' ∧ c3 ∉ {'u'})

noncomputable def probabilityOfAtLeastOneVowel : ℚ := 
  let totalCases := 5 * 6 * 5 * 5
  let case1 := 3 * 6 * 4 * 4
  let case2 := 1 * 5 * 1 * 4
  let case3 := 1 * 5 * 4 * 1
  (58 / 125 : ℚ)

theorem probability_at_least_one_vowel :
  (3 * 4 * 4 + 1 * 1 + 1 * 1) / 5 / 5 / 5 = (58 / 125 : ℚ) :=
by sorry

end probability_at_least_one_vowel_l19_19732


namespace cricket_run_rate_l19_19642

theorem cricket_run_rate (x : ℝ) (hx : 3.2 * x + 6.25 * 40 = 282) : x = 10 :=
by sorry

end cricket_run_rate_l19_19642


namespace part1_part2_part3_l19_19726

-- Part 1
theorem part1 (a b : ℝ) : 3*(a-b)^2 - 6*(a-b)^2 + 2*(a-b)^2 = -(a-b)^2 :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x^2 - 2*y = 4) : 3*x^2 - 6*y - 21 = -9 :=
sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) : 
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 :=
sorry

end part1_part2_part3_l19_19726


namespace decimal_to_fraction_l19_19790

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l19_19790


namespace mean_of_squares_eq_l19_19498

noncomputable def sum_of_squares (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def arithmetic_mean_of_squares (n : ℕ) : ℚ := sum_of_squares n / n

theorem mean_of_squares_eq (n : ℕ) (h : n ≠ 0) : arithmetic_mean_of_squares n = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end mean_of_squares_eq_l19_19498


namespace smallest_rectangle_area_l19_19041

theorem smallest_rectangle_area (r : ℕ) (h : r = 6) : 
  let diameter := 2 * r 
  in let length := diameter 
  in let width := 3 * r 
  in (length * width = 216) :=
by
  sorry

end smallest_rectangle_area_l19_19041


namespace volcano_intact_l19_19857

theorem volcano_intact (initial_count : ℕ)
                       (perc_2months : ℝ)
                       (perc_halfyear : ℝ)
                       (perc_yearend : ℝ)
                       (exploded_2months: initial_count * perc_2months / 100)
                       (remaining_after_2months: initial_count - exploded_2months)
                       (exploded_halfyear: remaining_after_2months * perc_halfyear / 100)
                       (remaining_after_halfyear: remaining_after_2months - exploded_halfyear)
                       (exploded_yearend: remaining_after_halfyear * perc_yearend / 100)
                       (remaining_after_yearend: remaining_after_halfyear - exploded_yearend) :
  let results : Nat := remaining_after_yearend in
  initial_count = 200 ∧ perc_2months = 20 ∧ perc_halfyear = 40 ∧ perc_yearend = 50 → results = 48 :=
begin
  sorry
end

end volcano_intact_l19_19857


namespace longer_leg_smallest_triangle_l19_19527

noncomputable def length_of_longer_leg_of_smallest_triangle (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then a 
  else if n = 1 then (a / 2) * Real.sqrt 3
  else if n = 2 then ((a / 2) * Real.sqrt 3 / 2) * Real.sqrt 3
  else ((a / 2) * Real.sqrt 3 / 2 * Real.sqrt 3 / 2) * Real.sqrt 3

theorem longer_leg_smallest_triangle : 
  length_of_longer_leg_of_smallest_triangle 3 10 = 45 / 8 := 
sorry

end longer_leg_smallest_triangle_l19_19527


namespace recurrence_relation_l19_19517

def f : ℕ → ℕ
| 0     := 0
| (n+1) := f n + Nat.factorial (n + 1)

theorem recurrence_relation (n : ℕ) :
  f (n + 2) = (n + 3) * f (n + 1) - (n + 2) * f n :=
sorry

end recurrence_relation_l19_19517


namespace searchlight_revolutions_l19_19463

theorem searchlight_revolutions (p : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : p = 0.6666666666666667) 
  (h2 : t = 10) 
  (h3 : p = (60 / r - t) / (60 / r)) : 
  r = 2 :=
by sorry

end searchlight_revolutions_l19_19463


namespace spheres_tangent_l19_19059

open_locale big_operators

variables {n : ℕ}
variables (S : fin n → ℝ → Prop) (P : ℝ → ℝ)
variables (x y : fin n → ℝ)

def sphere (S : fin n → ℝ → Prop) (i : fin n) : Prop := 
  ∃ r, r = 1 ∧ S i r

def tangent_point (S : fin n → ℝ → Prop) (P : ℝ → ℝ) (i : fin n) (xi yi : ℝ) : Prop :=
  ∃ x y, x = xi ∧ y = yi ∧ x ≥ 0 ∧ y ≥ 0

theorem spheres_tangent (S : fin n → ℝ → Prop) (P : ℝ → ℝ) 
  (x y : fin n → ℝ)
  (h1 : ∀ i, sphere S i) 
  (h2 : ∏ i, x i = ∏ i, y i) :
  ∏ i, x i ≥ ∏ i, y i :=
sorry

end spheres_tangent_l19_19059


namespace possible_values_l19_19919

def expression (m n : ℕ) : ℤ :=
  (m^2 + m * n + n^2) / (m * n - 1)

theorem possible_values (m n : ℕ) (h : m * n ≠ 1) : 
  ∃ (N : ℤ), N = expression m n → N = 0 ∨ N = 4 ∨ N = 7 :=
by
  sorry

end possible_values_l19_19919


namespace minimum_positive_period_cos2_sin2_l19_19762

def function_y (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.sin x) ^ 2

theorem minimum_positive_period_cos2_sin2 : ∃ T > 0, ∀ x, function_y (x + T) = function_y x ∧ (∀ T' > 0, (∀ x, function_y (x + T') = function_y x) → T' ≥ T) :=
by
  have h : function_y = λ x, Real.cos (2 * x), from sorry
  use π
  split
  · apply Real.pi_pos
  · split
    · intros x
      rw h
      apply Real.periodic_cos (2 * x)
      · exact 2
    · intros T' T'_pos all_periodic
      sorry

end minimum_positive_period_cos2_sin2_l19_19762


namespace polynomial_solution_l19_19969

noncomputable def p1 (x : ℝ) := (x - 1) * (x + 2)
noncomputable def p2 (x : ℝ) := (x^2 - 4) * (x - 1) * (x + 2)
noncomputable def c := -4

theorem polynomial_solution (p1 p2 : ℝ → ℝ) (c : ℝ) 
(h1 : ∀ x, (deriv ((p2 x) / (p1 x))) = 2 * x) 
(h2 : ∀ x, p2 x = (x^2 + c) * (p1 x)) 
(h3 : ∀ x, (p1 x + p2 x) = (x - 1) * (x + 2) * (x^2 - 3)) :
  p1 = λ x, (x - 1) * (x + 2) ∧ p2 = λ x, (x^2 - 4) * (x - 1) * (x + 2) :=
by
  intro p1 p2 c h1 h2 h3
  sorry

end polynomial_solution_l19_19969


namespace binary_111_to_decimal_l19_19915

-- Define a function to convert binary list to decimal
def binaryToDecimal (bin : List ℕ) : ℕ :=
  bin.reverse.enumFrom 0 |>.foldl (λ acc ⟨i, b⟩ => acc + b * (2 ^ i)) 0

-- Assert the equivalence between the binary number [1, 1, 1] and its decimal representation 7
theorem binary_111_to_decimal : binaryToDecimal [1, 1, 1] = 7 :=
  by
  sorry

end binary_111_to_decimal_l19_19915


namespace cone_volume_divided_by_pi_l19_19437

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l19_19437


namespace k_lt_half_plus_sqrt_two_n_l19_19685

variable (n k : ℕ) (S : Finset (ℝ × ℝ))

-- Conditions
def no_three_collinear (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 p3 : ℝ × ℝ), (p1 ∈ S) → (p2 ∈ S) → (p3 ∈ S) → 
  ¬collinear p1 p2 p3

def at_least_k_equidistant (S : Finset (ℝ × ℝ)) (k : ℕ) : Prop :=
  ∀ (P : ℝ × ℝ), P ∈ S → 
  (S.filter (λ Q, dist P Q = dist P (S.to_list.head))).card ≥ k

-- The main theorem
theorem k_lt_half_plus_sqrt_two_n : 
  no_three_collinear S →
  at_least_k_equidistant S k →
  k < 1 / 2 + real.sqrt (2 * n) :=
by
  sorry

end k_lt_half_plus_sqrt_two_n_l19_19685


namespace three_digit_integers_with_product_30_count_l19_19614

theorem three_digit_integers_with_product_30_count :
  {n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ( let digits := (n / 100) :: (n / 10 % 10) :: (n % 10) :: [] 
                                    in (digits.foldl (λ x y => x * y) 1 = 30) 
                                      ∧ (∀ d ∈ digits, 1 ≤ d ∧ d ≤ 9)) }.to_finset.card = 12 :=
sorry

end three_digit_integers_with_product_30_count_l19_19614


namespace log_4_of_one_over_2_sqrt2_l19_19924

theorem log_4_of_one_over_2_sqrt2 : log 4 (1 / (2 * sqrt 2)) = -3 / 4 :=
by
  sorry

end log_4_of_one_over_2_sqrt2_l19_19924


namespace total_time_hover_layover_two_days_l19_19880

theorem total_time_hover_layover_two_days 
    (hover_pacific_day1 : ℝ)
    (hover_mountain_day1 : ℝ)
    (hover_central_day1 : ℝ)
    (hover_eastern_day1 : ℝ)
    (layover_time : ℝ)
    (speed_increase : ℝ)
    (time_decrease : ℝ) :
    hover_pacific_day1 = 2 →
    hover_mountain_day1 = 3 →
    hover_central_day1 = 4 →
    hover_eastern_day1 = 3 →
    layover_time = 1.5 →
    speed_increase = 0.2 →
    time_decrease = 1.6 →
    hover_pacific_day1 + hover_mountain_day1 + hover_central_day1 + hover_eastern_day1 + 4 * layover_time 
      + (hover_eastern_day1 - (speed_increase * hover_eastern_day1) + hover_central_day1 - (speed_increase * hover_central_day1) 
         + hover_mountain_day1 - (speed_increase * hover_mountain_day1) + hover_pacific_day1 - (speed_increase * hover_pacific_day1)) 
      + 4 * layover_time = 33.6 := 
by
  intros
  sorry

end total_time_hover_layover_two_days_l19_19880


namespace fare_calculation_l19_19362

-- Definitions for given conditions
def initial_mile_fare : ℝ := 3.00
def additional_rate : ℝ := 0.30
def initial_miles : ℝ := 0.5
def available_fare : ℝ := 15 - 3  -- Total minus tip

-- Proof statement
theorem fare_calculation (miles : ℝ) : initial_mile_fare + additional_rate * (miles - initial_miles) / 0.10 = available_fare ↔ miles = 3.5 :=
by
  sorry

end fare_calculation_l19_19362


namespace hexagon_perimeter_l19_19073

theorem hexagon_perimeter (AB BC CD DE EF FA : ℝ) (hAB : AB = 1) (hBC : BC = 1)
  (hCD : CD = 1) (hDE : DE = 2) (hEF : EF = 1) (hFA : FA = 2 * Real.sqrt 2) :
  AB + BC + CD + DE + EF + FA = 6 + 2 * Real.sqrt 2 :=
by {
  rw [hAB, hBC, hCD, hDE, hEF, hFA],
  norm_num,
}

end hexagon_perimeter_l19_19073


namespace no_third_place_QT_l19_19646

def Runner := ℕ -- Define a type for runners as natural numbers for simplicity

variable (Q P R S T : Runner)

variable 
  (cond1 : Q > P)      -- Q beats P
  (cond2 : Q > R)      -- Q beats R
  (cond3 : P > S)      -- P beats S
  (cond4 : Q < T ∧ T < P) -- T finishes after Q but before P
  (cond5 : S > T)      -- S beats T

theorem no_third_place_QT :
  ∀ (third_place : Runner), (third_place ≠ Q) ∧ (third_place ≠ T) :=
by
  sorry

end no_third_place_QT_l19_19646


namespace quadratic_distinct_real_roots_l19_19757

theorem quadratic_distinct_real_roots (a b t l : ℝ) (h_tl : -1 < t ∧ t < 0) (h_vieta1 : t + l = -a) (h_vieta2 : t * l = b) :
  let a' := a + t
  let b' := b + t
  let g := λ x, x^2 + a'*x + b'
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g(x₁) = 0 ∧ g(x₂) = 0 := 
by
  let a' := a + t
  let b' := b + t
  -- Calculating the discriminant of g(x)
  have discriminant : (a' - 2 * t)^2 - 4 * 1 * (b' + t) > 0 :=
    sorry
  
  -- Since the discriminant is positive, there exist two distinct real roots.
  use (1 : ℝ) -- example values, the exact roots are not required for the theorem statement
  use (-1 : ℝ)
  split
  · sorry
  split
  · have g_x1 := g 1
    exact g_x1 -- this is just a placeholder
  sorry

end quadratic_distinct_real_roots_l19_19757


namespace problem_statement_l19_19294

def f (x : ℝ) : ℝ := Real.exp x + x - 2
def g (x : ℝ) : ℝ := Real.log x + x ^ 2 - 3

theorem problem_statement {a b : ℝ} (h₁ : f a = 0) (h₂ : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end problem_statement_l19_19294


namespace correct_propositions_l19_19180

-- Definitions
axiom line (l : Type) : Prop
axiom plane (α : Type) : Prop
axiom perp (l : Type) (α : Type) : Prop
axiom parallel (α β : Type) : Prop
axiom lies_in (m : Type) (β : Type) : Prop

variables (l m : Type) (α β : Type)

-- Conditions
axiom l_perpendicular_to_alpha : perp l α
axiom m_lies_in_beta : lies_in m β

-- Propositions to prove
theorem correct_propositions :
  (parallel α β → perp l m) ∧
  (parallel l m → perp α β) :=
begin
  sorry
end

end correct_propositions_l19_19180


namespace grape_juice_solution_l19_19627

noncomputable def grape_juice_problem : Prop :=
  ∃ (x : ℝ), 
    (40 * 0.10 = 4) ∧
    ((4 + x) / (40 + x) = 0.28) ∧ 
    (x = 10)

theorem grape_juice_solution : grape_juice_problem :=
begin
  use 10,
  split, { norm_num, },
  split, 
  { norm_num, 
    field_simp [by norm_cast : (0.28:ℝ) = 28 / 100], 
    simp [mul_comm], 
    ring, },
  { ring, }
end

end grape_juice_solution_l19_19627


namespace point_P_quadrant_IV_l19_19068

theorem point_P_quadrant_IV (x y : ℝ) (h1 : x > 0) (h2 : y < 0) : x > 0 ∧ y < 0 :=
by
  sorry

end point_P_quadrant_IV_l19_19068


namespace quadrilateral_area_sum_l19_19099

theorem quadrilateral_area_sum :
  let a b c d : ℕ := (4, 6, 8, 10)
  let s := (a + b + c + d) / 2
  let brahmagupta_formula : ℝ := (s.toReal - a.toReal) * (s.toReal - b.toReal) * (s.toReal - c.toReal) * (s.toReal - d.toReal)
  let r4 := 16
  let n3 := 21
  let r5 := 0
  let n4 := 0
  let r6 := 0
  Int.floor (r4 + r5 + r6 + n3 + n4 : ℝ) = 37 :=
by
  sorry

end quadrilateral_area_sum_l19_19099


namespace number_of_triangles_number_of_cuts_l19_19176

-- Definitions
def M (points : ℕ) : Set ℕ := {points | points ≤ 2005} ∪ {0, 1, 2, 3}

-- Main Theorems
theorem number_of_triangles (points : ℕ) (h_points : points = 2005) (h_collinear : ∀ x y z ∈ (M points), ¬ collinear x y z) :
  let total_vertices := 2005 + 4 in
  ∑ i in (M points), internal_angle_sum i = 2006 * 360 :=
  180 * 4012 :=
begin
  sorry -- The proof is skipped.
end

theorem number_of_cuts (points : ℕ) (h_points : points = 2005) (h_collinear : ∀ x y z ∈ (M points), ¬ collinear x y z) :
  let total_triangles := 4012 in
  let cuts y := y * 2 + 4 = total_triangles * 3 in
  2 * 6016 + 4 = 12036 :=
begin
  sorry -- The proof is skipped.
end

end number_of_triangles_number_of_cuts_l19_19176


namespace period_of_fraction_l19_19705

-- Define gcd function for Lean environment
def gcd (a b : Nat) : Nat :=
  if b = 0 then a else gcd b (a % b)

-- Define the conditions as Lean definitions
variables (a b m1 : Nat)
variable (h_gcd : gcd 10 m1 = 1)
def m : Nat := 2^a * 5^b * m1
def k : Nat := max a b

-- Statement of the theorem
theorem period_of_fraction (h1 : m = 2^a * 5^b * m1) (h2 : gcd 10 m1 = 1) :
  (start_period : Nat) × (length_period : Nat) :=
  start_period = k + 1 ∧ length_period = period_length (1 / m1) :=
by
  sorry

end period_of_fraction_l19_19705


namespace equilateral_triangle_distinct_lines_l19_19134

theorem equilateral_triangle_distinct_lines (T : Triangle) (h1 : T.is_equilateral) : 
  ∃! l : Line, (∀ V : Vertex, l.is_altitude V T ∧ l.is_median V T ∧ l.is_angle_bisector V T) :=
by
  sorry

end equilateral_triangle_distinct_lines_l19_19134


namespace megan_homework_problems_l19_19711

theorem megan_homework_problems
  (finished_problems : ℕ)
  (pages_remaining : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ) :
  finished_problems = 26 →
  pages_remaining = 2 →
  problems_per_page = 7 →
  total_problems = finished_problems + (pages_remaining * problems_per_page) →
  total_problems = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end megan_homework_problems_l19_19711


namespace intersection_eq_l19_19603

def A : Set ℝ := { x | abs x ≤ 2 }
def B : Set ℝ := { x | 3 * x - 2 ≥ 1 }

theorem intersection_eq :
  A ∩ B = { x | 1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_eq_l19_19603


namespace percentage_of_primes_divisible_by_2_l19_19048

open_locale classical
noncomputable theory

def prime_numbers_less_than_twenty := {p : ℕ | nat.prime p ∧ p < 20}

theorem percentage_of_primes_divisible_by_2 : 
  (card {p ∈ prime_numbers_less_than_twenty | 2 ∣ p}).to_real / (card prime_numbers_less_than_twenty).to_real * 100 = 12.5 := 
by
  sorry

end percentage_of_primes_divisible_by_2_l19_19048


namespace bank_account_balance_l19_19531

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l19_19531


namespace max_y_coordinate_is_three_fourths_l19_19943

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l19_19943


namespace youngest_age_is_29_l19_19652

-- Define that the ages form an arithmetic sequence
def arithmetic_sequence (a1 a2 a3 a4 : ℕ) : Prop :=
  ∃ (d : ℕ), a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d

-- Define the problem statement
theorem youngest_age_is_29 (a1 a2 a3 a4 : ℕ) (h_seq : arithmetic_sequence a1 a2 a3 a4) (h_oldest : a4 = 50) (h_sum : a1 + a2 + a3 + a4 = 158) :
  a1 = 29 :=
by
  sorry

end youngest_age_is_29_l19_19652


namespace problem_solution_l19_19910

theorem problem_solution (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : abs (Real.logb 5 k - Real.logb 5 (k^2 + 3)) = 0.6) : 
  a + b = 15 :=
sorry

end problem_solution_l19_19910


namespace no_prime_pairs_sum_53_l19_19659

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l19_19659


namespace spadesuit_proof_l19_19518

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem spadesuit_proof : 
  spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6)) = 3 :=
by
  sorry

end spadesuit_proof_l19_19518


namespace arithmetic_progression_correct_l19_19545

noncomputable def nth_term_arithmetic_progression (n : ℕ) : ℝ :=
  4.2 * n + 9.3

def recursive_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  a 1 = 13.5 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + 4.2

theorem arithmetic_progression_correct (n : ℕ) :
  (nth_term_arithmetic_progression n = 4.2 * n + 9.3) ∧
  ∀ (a : ℕ → ℝ), recursive_arithmetic_progression a → a n = 4.2 * n + 9.3 :=
by
  sorry

end arithmetic_progression_correct_l19_19545


namespace mountains_still_intact_at_end_of_year_l19_19855

theorem mountains_still_intact_at_end_of_year
  (initial_volcanoes : ℕ)
  (percent_erupted_first_two_months : ℕ)
  (percent_erupted_halfway : ℕ)
  (percent_erupted_end_year : ℕ)
  (initial_volcanoes = 200)
  (percent_erupted_first_two_months = 20)
  (percent_erupted_halfway = 40)
  (percent_erupted_end_year = 50) :
  let volcanoes_after_two_months := initial_volcanoes - (initial_volcanoes * percent_erupted_first_two_months / 100)
  let volcanoes_after_half_year := volcanoes_after_two_months - (volcanoes_after_two_months * percent_erupted_halfway / 100)
  let volcanoes_end_year := volcanoes_after_half_year - (volcanoes_after_half_year * percent_erupted_end_year / 100)
  volcanoes_end_year = 48 :=
by
  sorry

end mountains_still_intact_at_end_of_year_l19_19855


namespace percentage_primes_divisible_by_2_l19_19051

theorem percentage_primes_divisible_by_2 :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  (100 * primes.filter (fun n => n % 2 = 0).card / primes.card) = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  have h1 : primes.filter (fun n => n % 2 = 0).card = 1 := sorry
  have h2 : primes.card = 8 := sorry
  have h3 : (100 * 1 / 8 : ℝ) = 12.5 := by norm_num
  exact h3

end percentage_primes_divisible_by_2_l19_19051


namespace value_of_k_l19_19781

noncomputable def radius_of_larger_circle : ℝ :=
  let P : ℝ × ℝ := (9, 12) in
  real.sqrt (P.1^2 + P.2^2)

lemma radius_larger_circle_is_15 : radius_of_larger_circle = 15 := by 
  -- calculation skipped
  sorry

noncomputable def radius_of_smaller_circle : ℝ :=
  radius_of_larger_circle - 5

lemma radius_smaller_circle_is_10 : radius_of_smaller_circle = 10 := by 
  -- calculation skipped
  sorry

noncomputable def k : ℝ :=
  let S : ℝ × ℝ := (0, radius_of_smaller_circle) in
  S.2

theorem value_of_k : k = 10 := by
  -- calculation skipped
  sorry

end value_of_k_l19_19781


namespace trig_identity_l19_19959

theorem trig_identity :
  let sin := Real.sin;
      cos := Real.cos;
      deg_to_rad := λ deg : ℝ => deg * (π / 180)
  in sin (deg_to_rad 15) * cos (deg_to_rad 75) + cos (deg_to_rad 15) * sin (deg_to_rad 105) = 1 :=
by
  sorry

end trig_identity_l19_19959


namespace f_2016_l19_19975

noncomputable def f : ℤ → ℝ 
| 1 := 2
| (x + 1) := (1 + f x) / (1 - f x)
| x := sorry -- Define handling for other x as needed

theorem f_2016 : f 2016 = 1/3 :=
by sorry

end f_2016_l19_19975


namespace incorrect_D_l19_19393

-- Definitions according to conditions
variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b : α)
variables (AB AC : α)
variables (θ₁ θ₂ : ℝ)

-- Conditions from the problem
def condition_A (a b : α) : Prop := a + b = 0 → a = -b ∧ a ∥ b
def condition_B (AB AC : α) : Prop := AB ≠ AC → True -- Just stating points B and C do not coincide
def condition_C (θ₁ θ₂ : ℝ) : Prop := θ₁ = 70 ∧ θ₂ = 20 → True -- Just stating vectors are collinear
def condition_D (a b : α) : Prop := a ∥ b → ∥a∥ = ∥b∥

-- Incorrectness of Statement D
theorem incorrect_D : ¬ condition_D a b :=
begin
 sorry -- The proof is not required, only the statement
end

end incorrect_D_l19_19393


namespace range_of_m_l19_19634

def f (m x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def f_derivative_nonnegative_on_interval (m : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0

theorem range_of_m (m : ℝ) : f_derivative_nonnegative_on_interval m ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l19_19634


namespace simplify_and_evaluate_expression_l19_19322

variable (x y : ℚ)

theorem simplify_and_evaluate_expression (hx : x = 1) (hy : y = 1 / 2) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x - y) ^ 2 = 31 / 4 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l19_19322


namespace new_ortan_license_plates_l19_19874

/-- 
A valid license plate in New Ortan consists of three letters followed by four digits.
Prove that the total number of valid license plates possible in New Ortan is 175,760,000.
-/
theorem new_ortan_license_plates : 
  let letters := 26
  let digits := 10
  let num_letters := 3
  let num_digits := 4
  letters ^ num_letters * digits ^ num_digits = 175760000 :=
by
  let letters := 26
  let digits := 10
  let num_letters := 3
  let num_digits := 4
  calc
    letters ^ num_letters * digits ^ num_digits = 26 ^ 3 * 10 ^ 4 : by rfl
    ... = 175760000 : by norm_num

end new_ortan_license_plates_l19_19874


namespace circles_intersect_l19_19764

open real

def Circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 1
def Circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y - 1 = 0

theorem circles_intersect :
  (∃ x y : ℝ, Circle1 x y ∧ Circle2 x y) :=
sorry

end circles_intersect_l19_19764


namespace F_sequence_example_1_F_sequence_increasing_F_sequence_sum_zero_l19_19291

-- Define F sequence property
def F_sequence (a : Nat → Int) (d : Int) (n : Nat) : Prop :=
  ∀ k, 1 ≤ k ∧ k < n → abs (a (k+1) - a k) = d

-- Problem 1: F sequences A_5 with a_1 = a_5 = 0
theorem F_sequence_example_1 (d : Int) (a : Nat → Int) (h_d_pos : d > 0) :
  F_sequence a d 5 → a 1 = 0 → a 5 = 0 → 
  a = (λ x, match x with 
            | 1 => 0 
            | 2 => d 
            | 3 => 0 
            | 4 => d 
            | 5 => 0 
            | _ => 0 
           end) ∨ 
  a = (λ x, match x with 
            | 1 => 0 
            | 2 => -d 
            | 3 => 0 
            | 4 => -d 
            | 5 => 0 
            | _ => 0 
           end) :=
by sorry

-- Problem 2: F sequence is increasing iff a_{2016} = 2016
theorem F_sequence_increasing (a : Nat → Int) :
  F_sequence a 1 2016 → a 1 = 1 → a 2016 = 2016 ↔ (∀ k, 1 ≤ k ∧ k < 2016 → a k < a (k + 1)) :=
by sorry

-- Problem 3: S(A_n) = 0 iff n=4k
theorem F_sequence_sum_zero (a : Nat → Int) (d n : Nat) (h_d_pos : d > 0) :
  a 1 = 0 → F_sequence a d n → 
  let S := (∑ i in Finset.range n, a (i + 1)) in
  (S = 0 ↔ ∃ k : Nat, n = 4 * k) :=
by sorry

end F_sequence_example_1_F_sequence_increasing_F_sequence_sum_zero_l19_19291


namespace sum_first_12_terms_l19_19651

variable (a_n : ℕ → ℝ)
variable (d : ℝ)

-- Arithmetic sequence, general term a_n = a_1 + (n-1)d
def a (n : ℕ) : ℝ := a_n 1 + (n - 1) * d

-- Given condition
axiom h1 : (a 1 + a 4 + a 7) + 3 * a 9 = 15

-- S₁₂ is the sum of the first 12 terms of the arithmetic sequence
def S₁₂ := ∑ i in finset.range 12, a (i + 1)

-- Theorem to prove
theorem sum_first_12_terms : S₁₂ = 30 :=
sorry

end sum_first_12_terms_l19_19651


namespace tetrahedron_edge_length_l19_19157

noncomputable def length_of_tetrahedron_edge (r : ℝ) : ℝ := 
  let l := (2 * r * Math.sqrt 61) / 3
  l

theorem tetrahedron_edge_length
  (r : ℝ)
  (h_pos : r = 2) :
  length_of_tetrahedron_edge r = (2 * Real.sqrt 61) / 3 :=
by
  -- Here we provide the definitions and constraints from the problem
  sorry

end tetrahedron_edge_length_l19_19157


namespace expected_value_of_winnings_is_4_l19_19844

noncomputable def expected_value_of_winnings : ℕ := 
  let outcomes := [7, 6, 5, 4, 4, 3, 2, 1]
  let total_winnings := outcomes.sum
  total_winnings / 8

theorem expected_value_of_winnings_is_4 :
  expected_value_of_winnings = 4 :=
by
  sorry

end expected_value_of_winnings_is_4_l19_19844


namespace symmetric_circle_equation_l19_19769

-- Define the original circle and line
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = 1

-- State the problem of finding the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, (circle x y ↔ (x - 1)^2 + (y - 1)^2 = 1)) :=
sorry

end symmetric_circle_equation_l19_19769


namespace initial_donuts_correct_l19_19887

-- Definition: Initial number of donuts
def initial_donuts (D : ℕ) : Prop :=
  let after_bill := D - 2 in
  let after_secretary := after_bill - 4 in
  let after_coworkers := after_secretary / 2 in
  after_coworkers = 22

-- Proof that the initial number of donuts was 50
theorem initial_donuts_correct : initial_donuts 50 :=
by 
  let after_bill := 50 - 2
  let after_secretary := after_bill - 4
  let after_coworkers := after_secretary / 2
  have h : after_coworkers = 22 := by 
    simp [after_bill, after_secretary, after_coworkers]
    sorry
  show initial_donuts 50 from h

end initial_donuts_correct_l19_19887


namespace asymptotically_stable_at_origin_l19_19258

def system (t : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (y - x^3, -x - 3 * y^3)
  
def V (x y : ℝ) : ℝ :=
  x^2 + y^2

theorem asymptotically_stable_at_origin :
  ∀ t : ℝ, ∀ x y : ℝ,
  (differential at (t, V x y) by system t x y) ≤ 0 :=
sorry

end asymptotically_stable_at_origin_l19_19258


namespace power_ranger_stickers_difference_l19_19315

theorem power_ranger_stickers_difference :
  ∃ (d : ℤ), 
    let a₁ := 30 in
    let a₂ := a₁ + d in
    let a₃ := a₂ + d in
    let a₄ := a₃ + d in
    let a₅ := a₄ + d in
    a₁ + a₂ + a₃ + a₄ + a₅ = 250 ∧ (a₅ - a₁) = 40 :=
begin
  sorry
end

end power_ranger_stickers_difference_l19_19315


namespace sequence_initial_term_l19_19650

theorem sequence_initial_term (a : ℕ) :
  let a_1 := a
  let a_2 := 2
  let a_3 := a_1 + a_2
  let a_4 := a_1 + a_2 + a_3
  let a_5 := a_1 + a_2 + a_3 + a_4
  let a_6 := a_1 + a_2 + a_3 + a_4 + a_5
  a_6 = 56 → a = 5 :=
by
  intros h
  sorry

end sequence_initial_term_l19_19650


namespace emily_three_blue_marbles_probability_l19_19140

noncomputable def probability_exactly_three_blue_marbles : ℝ :=
  let blue_marble_prob := (8:ℕ) / (14:ℕ)
  let red_marble_prob := (6:ℕ) / (14:ℕ)
  let n := 6
  let k := 3
  let comb := (nat.choose n k)
  in comb * (blue_marble_prob ^ k) * (red_marble_prob ^ (n - k))

theorem emily_three_blue_marbles_probability :
  probability_exactly_three_blue_marbles = 34560 / 117649 := by
  sorry

end emily_three_blue_marbles_probability_l19_19140


namespace trapezoid_area_l19_19254

-- Define the conditions of the problem

structure TrapezoidData where
  A B C D E : Type
  area_ADE : ℝ
  area_BCE : ℝ
  parallel_AD_BC : Prop
  intersect_diagonals_at_E : Prop

-- Assuming specific values for the areas
example : TrapezoidData := {
  A := ℝ,
  B := ℝ,
  C := ℝ,
  D := ℝ,
  E := ℝ,
  area_ADE := 12,
  area_BCE := 3,
  parallel_AD_BC := sorry,          -- AD || BC
  intersect_diagonals_at_E := sorry  -- Diagonals intersect at E
}

-- Theorem statement that needs to be proven.
theorem trapezoid_area (data : TrapezoidData) (h₁ : data.parallel_AD_BC) (h₂ : data.intersect_diagonals_at_E)
  (h₃ : data.area_ADE = 12) (h₄ : data.area_BCE = 3) : 
  let area_ABCD := data.area_ADE + data.area_BCE + 2 * data.area_BCE + 2 * data.area_BCE ==> 27 := 
sorry

end trapezoid_area_l19_19254


namespace abs_neg_sub_three_eq_zero_l19_19502

theorem abs_neg_sub_three_eq_zero : |(-3 : ℤ)| - 3 = 0 :=
by sorry

end abs_neg_sub_three_eq_zero_l19_19502


namespace opposite_of_neg_6_l19_19010

theorem opposite_of_neg_6 : ∀ (n : ℤ), n = -6 → -n = 6 :=
by
  intro n h
  rw [h]
  sorry

end opposite_of_neg_6_l19_19010


namespace domain_of_f_l19_19227

theorem domain_of_f : 
  (∀ x : ℝ, -1 < x ∧ x < 0 → f(2*x + 1)) → (∀ x : ℝ, -1 < x ∧ x < 1 → f(x)) := 
sorry

end domain_of_f_l19_19227


namespace no_prime_pair_summing_to_53_l19_19654

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l19_19654


namespace Mickey_less_than_twice_Minnie_l19_19127

def Minnie_horses_per_day : ℕ := 10
def Mickey_horses_per_day : ℕ := 14

theorem Mickey_less_than_twice_Minnie :
  2 * Minnie_horses_per_day - Mickey_horses_per_day = 6 := by
  sorry

end Mickey_less_than_twice_Minnie_l19_19127


namespace min_colors_required_l19_19008

def color_grid (grid : Matrix Fin 3 (Fin 3) ℕ) : Prop :=
  (∀ i : Fin 3, ∀ j1 j2 : Fin 3, j1 ≠ j2 → grid i j1 ≠ grid i j2) ∧ -- Unique colors in each row
  (∀ j : Fin 3, ∀ i1 i2 : Fin 3, i1 ≠ i2 → grid i1 j ≠ grid i2 j) ∧ -- Unique colors in each column
  (∀ d1 d2 : Fin 3, d1 ≠ d2 → grid d1 d1 ≠ grid d2 d2) ∧          -- Unique colors in the main diagonal
  (∀ d1 d2 : Fin 3, d1 ≠ d2 → grid d1 (2 - d1) ≠ grid d2 (2 - d2)) -- Unique colors in the anti-diagonal

theorem min_colors_required (grid : Matrix Fin 3 (Fin 3) ℕ) (c : ℕ) :
  color_grid grid → (∀ i j : Fin 3, grid i j < c) → c ≥ 5 := 
sorry

end min_colors_required_l19_19008


namespace wholesale_price_l19_19458

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end wholesale_price_l19_19458


namespace traveler_is_lying_l19_19469

/-- A carpet is a pair of positive real numbers (a, b). --/
def Carpet := ℝ × ℝ

/-- Determines if a carpet is "large", i.e., both sides are greater than 1. --/
def is_large (c : Carpet) : Prop := c.1 > 1 ∧ c.2 > 1

/-- Determines if a carpet has one side longer than 1 and one side shorter than 1. --/
def one_side_longer (c : Carpet) : Prop := (c.1 > 1 ∧ c.2 < 1) ∨ (c.1 < 1 ∧ c.2 > 1)

/-- Determines if a carpet is "small", i.e., both sides are less than 1. --/
def is_small (c : Carpet) : Prop := c.1 < 1 ∧ c.2 < 1

/-- Represents the first type of exchange. --/
def exchange1 (c : Carpet) : Carpet := (1 / c.1, 1 / c.2)

/-- Represents the second type of exchange. --/
def exchange2 (c : Carpet) (k : ℝ) : Carpet × Carpet := ((k, c.2), (c.1 / k, c.2))

/-- The initial condition: a carpet with both sides greater than 1. --/
def initial_carpet (c : Carpet) : Prop := is_large c

/-- Inductive definition to simulate the sequence of allowed exchanges.
The idea is to simulate exchanges of carpets, ensuring that the nature of carpets remain intact.
--/
inductive Exchanges : Carpet → Prop
| initial (c : Carpet) (h : initial_carpet c) : Exchanges c
| exchange1 (c : Carpet) (h : Exchanges c) : Exchanges (exchange1 c)
| exchange2 (c : Carpet) (k : ℝ) (h : Exchanges c) : Exchanges (exchange2 c k).1 ∧ Exchanges (exchange2 c k).2

/-- Main theorem stating the impossibility of ending up with a set of carpets each having one side longer than 1 and the other side shorter than 1 given the conditions. --/
theorem traveler_is_lying (c : Carpet) (h : Exchanges c) : ¬ one_side_longer c := sorry

end traveler_is_lying_l19_19469


namespace no_prime_pairs_sum_53_l19_19660

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l19_19660


namespace simplified_expression_evaluation_l19_19329

-- Problem and conditions
def x := Real.sqrt 5 - 1

-- Statement of the proof problem
theorem simplified_expression_evaluation : 
  ( (x / (x - 1) - 1) / (x^2 - 1) / (x^2 - 2 * x + 1) ) = Real.sqrt 5 / 5 :=
sorry

end simplified_expression_evaluation_l19_19329


namespace length_of_train_l19_19825

-- Definitions based on conditions
def length_train_eq_length_platform (l_train l_platform : ℝ) : Prop := l_train = l_platform

def speed_train := 144 * 1000 / 3600  -- Speed in m/s, convert 144 km/hr to m/s

def time_crossing := 60  -- Train crossing time in seconds

-- Prove the length of the train is 1200
theorem length_of_train (l_train l_platform : ℝ) (h_eq : length_train_eq_length_platform l_train l_platform) (h_speed : speed_train = 40) (h_time : time_crossing = 60) :
  l_train = 1200 :=
by
  -- Calculate distance
  let distance := speed_train * time_crossing
  have h_distance : distance = 2400 := by calc
    distance = 40 * 60 : by rw [h_speed, h_time]
    ... = 2400 : by norm_num
  -- Use length_train_eq_length_platform
  have h_sum := h_eq
  rw h_eq at h_sum
  -- Prove train length is 1200
  sorry

end length_of_train_l19_19825


namespace value_of_a_if_lines_are_parallel_l19_19352

theorem value_of_a_if_lines_are_parallel (a : ℝ) :
  (∀ (x y : ℝ), x + a*y - 7 = 0 → (a+1)*x + 2*y - 14 = 0) → a = -2 :=
sorry

end value_of_a_if_lines_are_parallel_l19_19352


namespace convex_polygon_perpendicular_foot_extension_l19_19243

theorem convex_polygon_perpendicular_foot_extension {n : ℕ} (h : 4 ≤ n) :
  ∀ (P : Fin n → ℝ × ℝ),
  (∀ i : Fin n, Convex Polygon P) →
  (∀ i j : Fin n, ¬Adjacent P i j → Foot_of_Perpendicular_Lands_on_Extension P i j) →
  False :=
by intros;
   sorry

end convex_polygon_perpendicular_foot_extension_l19_19243


namespace median_length_l19_19784

def isosceles_triangle (X Y Z M : Type) [metric_space X] [metric_space Y] [metric_space Z] [metric_space M] :=
  ∃ (XY XZ YZ YM XM : ℝ), XY = 10 ∧ XZ = 10 ∧ YZ = 12 ∧ YM = YM ∧ YZ = 2 * YM ∧ XM = sqrt (XY^2 - YM^2)

theorem median_length (X Y Z M : Type) [metric_space X] [metric_space Y] [metric_space Z] [metric_space M]
  (h : isosceles_triangle X Y Z M) : ∃ (XM : ℝ), XM = 8 := 
by
  sorry

end median_length_l19_19784


namespace express_y_in_terms_of_x_l19_19571

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 6) : y = 2 * x + 6 :=
by
  sorry

end express_y_in_terms_of_x_l19_19571


namespace cupcakes_gluten_nut_nonvegan_l19_19454

-- Definitions based on conditions
def total_cupcakes := 120
def gluten_free_cupcakes := total_cupcakes / 3
def vegan_cupcakes := total_cupcakes / 4
def nut_free_cupcakes := total_cupcakes / 5
def gluten_and_vegan_cupcakes := 15
def vegan_and_nut_free_cupcakes := 10

-- Defining the theorem to prove the main question
theorem cupcakes_gluten_nut_nonvegan : 
  total_cupcakes - ((gluten_free_cupcakes + (vegan_cupcakes - gluten_and_vegan_cupcakes)) - vegan_and_nut_free_cupcakes) = 65 :=
by sorry

end cupcakes_gluten_nut_nonvegan_l19_19454


namespace max_y_coordinate_is_three_fourths_l19_19941

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l19_19941


namespace supermarket_A_is_more_cost_effective_l19_19305

def price_A (kg : ℕ) : ℕ :=
  if kg <= 4 then kg * 10
  else 4 * 10 + (kg - 4) * 6

def price_B (kg : ℕ) : ℕ :=
  kg * 10 * 8 / 10

theorem supermarket_A_is_more_cost_effective :
  price_A 3 = 30 ∧ 
  price_A 5 = 46 ∧ 
  ∀ (x : ℕ), (x > 4) → price_A x = 6 * x + 16 ∧ 
  price_A 10 < price_B 10 :=
by 
  sorry

end supermarket_A_is_more_cost_effective_l19_19305


namespace trisected_right_triangle_product_l19_19312

theorem trisected_right_triangle_product :
  ∀ (XYZ : Triangle) (X Y Z P Q : Point),
    XY = 228 → YZ = 2004 →
    is_right_angle (∠XYZ) → 
    trisects Y (∠XYZ) P Q →
    product_eq_1370736 ((PY + YZ) * (QY + XY)) :=
begin
  sorry
end

end trisected_right_triangle_product_l19_19312


namespace sector_central_angle_l19_19632

theorem sector_central_angle (r α: ℝ) (hC: 4 * r = 2 * r + α * r): α = 2 :=
by
  -- Proof is to be filled in
  sorry

end sector_central_angle_l19_19632


namespace modulus_of_z_l19_19293

theorem modulus_of_z (z : ℂ) (h : (2017 * z - 25) / (z - 2017) = 3 + 4 * complex.I) :
  complex.abs z = 5 := 
sorry

end modulus_of_z_l19_19293


namespace area_of_regionB_l19_19912

open Complex

noncomputable def regionB_area : ℝ := 2500 - 625 * Real.pi

theorem area_of_regionB :
  let B := {z : ℂ | (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1) ∧ (0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
                      ∧ (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1)
                      ∧ (0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1) } in
  measure_theory.measure.comap Complex.add B 1 = regionB_area := by sorry

end area_of_regionB_l19_19912


namespace regression_estimate_y_at_3_l19_19562

variables {x y : Type} [linear_ordered_field x] [linear_ordered_field y]

-- Defining the observations and regression line
noncomputable def observations (x_i y_i : ℕ → x × y) : Prop :=
  ∃ (xs ys : list x), (xs.length = 8 ∧ ys.length = 8) ∧
  (sum xs = 8 ∧ sum ys = 4)

def regression_line (y_hat : x → y) : Prop :=
  ∀ x, y_hat x = (1/3 : x) * x + (1/6 : y)

-- The main statement to be proved
theorem regression_estimate_y_at_3
  (x_i y_i : ℕ → x × y)
  (h_obs : observations x_i y_i)
  (y_hat : x → y)
  (h_reg : regression_line y_hat):
  y_hat 3 = (7/6 : y) :=
sorry

end regression_estimate_y_at_3_l19_19562


namespace ivanov_voted_against_kuznetsov_l19_19835

theorem ivanov_voted_against_kuznetsov
    (members : List String)
    (vote : String → String)
    (majority_dismissed : (String × Nat))
    (petrov_statement : String)
    (ivanov_concluded : Bool) :
  members = ["Ivanov", "Petrov", "Sidorov", "Kuznetsov"] →
  (∀ x ∈ members, vote x ∈ members ∧ vote x ≠ x) →
  majority_dismissed = ("Ivanov", 3) →
  petrov_statement = "Petrov voted against Kuznetsov" →
  ivanov_concluded = True →
  vote "Ivanov" = "Kuznetsov" :=
by
  intros members_cond vote_cond majority_cond petrov_cond ivanov_cond
  sorry

end ivanov_voted_against_kuznetsov_l19_19835


namespace max_y_on_graph_l19_19933

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l19_19933


namespace log_4_of_one_over_2_sqrt2_l19_19923

theorem log_4_of_one_over_2_sqrt2 : log 4 (1 / (2 * sqrt 2)) = -3 / 4 :=
by
  sorry

end log_4_of_one_over_2_sqrt2_l19_19923


namespace find_a_l19_19410

variable (a : ℝ)

def average_condition (a : ℝ) : Prop :=
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74

theorem find_a (h: average_condition a) : a = 28 :=
  sorry

end find_a_l19_19410


namespace concave_number_probability_l19_19105

def is_concave_number (a b c : ℕ) := a > b ∧ b < c

def num_possibilities : ℕ := 24 
def num_concave : ℕ := 8 

theorem concave_number_probability :
  let choices := {1, 2, 3, 4}
  let total_possibilities := num_possibilities
  let concave_possibilities := num_concave
  concave_possibilities / total_possibilities = 1 / 3 :=
by
  sorry

end concave_number_probability_l19_19105


namespace a_2010_eq_9_l19_19253

namespace sequence_problem

def units_digit (x : ℕ) : ℕ := x % 10

def seq : ℕ → ℕ
| 0     := 3
| 1     := 7
| (n+2) := units_digit (seq n * seq (n+1))

theorem a_2010_eq_9 : seq 2009 = 9 := sorry

end sequence_problem

end a_2010_eq_9_l19_19253


namespace area_triangle_OAB_is_constant_find_circle_equation_l19_19592

open Real

-- Definition of the circle's center
def center (t : ℝ) : ℝ × ℝ := (t, 2/t)

-- (1) Prove that the area of triangle OAB is constant.
theorem area_triangle_OAB_is_constant (t : ℝ) (ht : t ≠ 0) :
  let C : ℝ × ℝ := center t in
  let A : ℝ × ℝ := (2 * t, 0) in
  let B : ℝ × ℝ := (0, 4 / t) in
  let O : ℝ × ℝ := (0, 0) in
  1 / 2 * abs (O.1 - A.1) * abs (O.2 - B.2) = 4 := sorry

-- (2) Given the line y = -2x + 4 intersects the circle C at points M and N,
--     and OM = ON, find the equation of the circle.
theorem find_circle_equation (t : ℝ) (ht : t = 2 ∨ t = -2) :
  let C : ℝ × ℝ := center t in
  let d := √5 in
  C = (2, 1) ∧ d = sqrt (t^2 + (2/t)^2) ∧
  (OM = ON → (x - 2)^2 + (y - 1)^2 = 5) := sorry

end area_triangle_OAB_is_constant_find_circle_equation_l19_19592


namespace bridge_length_l19_19075

theorem bridge_length
  (train_length : ℝ) (train_speed : ℝ) (time_taken : ℝ)
  (h_train_length : train_length = 280)
  (h_train_speed : train_speed = 18)
  (h_time_taken : time_taken = 20) : ∃ L : ℝ, L = 80 :=
by
  let distance_covered := train_speed * time_taken
  have h_distance_covered : distance_covered = 360 := by sorry
  let bridge_length := distance_covered - train_length
  have h_bridge_length : bridge_length = 80 := by sorry
  existsi bridge_length
  exact h_bridge_length

end bridge_length_l19_19075


namespace parabola_directrix_l19_19197

theorem parabola_directrix (vertex_origin : ∀ (x y : ℝ), x = 0 ∧ y = 0)
    (directrix : ∀ (y : ℝ), y = 4) : ∃ p, x^2 = -2 * p * y ∧ p = 8 ∧ x^2 = -16 * y := 
sorry

end parabola_directrix_l19_19197


namespace cos_A_value_l19_19576

theorem cos_A_value (A : ℝ) (h : tan A + sec A = 3) :
  cos A = 3 / 5 :=
by
  sorry

end cos_A_value_l19_19576


namespace find_m_n_find_y_range_l19_19567

-- Define the conditions for the solutions
variable (m n : ℚ)
variable (x y : ℚ)

-- Define the two solutions and the equation
def solution1 := (x = 2 ∧ y = -3)
def solution2 := (x = 4 ∧ y = 1)
def equation := (m * x - 3 * n * y = 5)

-- Define the part 1 proof: Prove the values of m and n
theorem find_m_n :
  (equation → solution1) ∧ (equation → solution2) → (m = 10 / 7 ∧ n = 5 / 21) := by
  sorry

-- Define the part 2 proof: Prove the range of y given x < -2
theorem find_y_range (h : m = 10 / 7) (h_n : n = 5 / 21) (h_x : x < -2) :
  equation → solution1 ∧ equation → solution2 → (y < -11) := by
  sorry

end find_m_n_find_y_range_l19_19567


namespace area_common_part_l19_19860

theorem area_common_part (R : ℝ) :
  let S := (R^2 * (8 * Real.sqrt 3 - 9)) / 4 in
  true := sorry

end area_common_part_l19_19860


namespace sum_m_Cnm_eq_n_pow2n_1_sum_m2_Cnm_eq_n_n1_pow2n_2_l19_19718

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that ∑ m=1 to n of m * C(n, m) = n * 2^(n-1)
theorem sum_m_Cnm_eq_n_pow2n_1 (n : ℕ) : 
  (∑ m in Finset.range (n + 1), m * binomial n m) = n * 2^(n-1) :=
sorry

-- Problem 2: Prove that ∑ m=1 to n of m² * C(n, m) = n (n + 1) * 2^(n-2)
theorem sum_m2_Cnm_eq_n_n1_pow2n_2 (n : ℕ) : 
  (∑ m in Finset.range (n + 1), m^2 * binomial n m) = n * (n + 1) * 2^(n-2) :=
sorry

end sum_m_Cnm_eq_n_pow2n_1_sum_m2_Cnm_eq_n_n1_pow2n_2_l19_19718


namespace solve_system_of_equations_l19_19335

theorem solve_system_of_equations : 
  ∃ (x y : ℤ), 2 * x + 5 * y = 8 ∧ 3 * x - 5 * y = -13 ∧ x = -1 ∧ y = 2 :=
by
  sorry

end solve_system_of_equations_l19_19335


namespace cameron_total_questions_l19_19898

theorem cameron_total_questions :
  let questions_per_tourist := 2
  let first_group := 6
  let second_group := 11
  let third_group := 8
  let third_group_special_tourist := 1
  let third_group_special_questions := 3 * questions_per_tourist
  let fourth_group := 7
  let first_group_total_questions := first_group * questions_per_tourist
  let second_group_total_questions := second_group * questions_per_tourist
  let third_group_total_questions := (third_group - third_group_special_tourist) * questions_per_tourist + third_group_special_questions
  let fourth_group_total_questions := fourth_group * questions_per_tourist
  in first_group_total_questions + second_group_total_questions + third_group_total_questions + fourth_group_total_questions = 68 := by
  sorry

end cameron_total_questions_l19_19898


namespace sum_of_areas_of_circles_l19_19765

noncomputable def radius (n : ℕ) : ℝ :=
  3 / 3^n

noncomputable def area (n : ℕ) : ℝ :=
  Real.pi * (radius n)^2

noncomputable def total_area : ℝ :=
  ∑' n, area n

theorem sum_of_areas_of_circles:
  total_area = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_of_circles_l19_19765


namespace general_term_of_sequence_l19_19196

/--
Given a sequence \{a_n\} with the sum of the first n terms S_n, and a_1 = 2S_n + 4n + 2,
prove that the general term of the sequence is a_n = 2^n.
-/
theorem general_term_of_sequence
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, a 1 = 2 * S n + 4 * n + 2)
  (h2 : ∀ n, S n = ∑ i in finset.range (n + 1), a i) :
  ∀ n, a n = 2 ^ n :=
by
  sorry

end general_term_of_sequence_l19_19196


namespace work_rate_problem_l19_19837

theorem work_rate_problem
  (W : ℕ) -- total work
  (A_rate : ℕ) -- A's work rate in days
  (B_rate : ℕ) -- B's work rate in days
  (x : ℕ) -- days A worked alone
  (total_days : ℕ) -- days A and B worked together
  (hA : A_rate = 12) -- A can do the work in 12 days
  (hB : B_rate = 6) -- B can do the work in 6 days
  (hx : total_days = 3) -- remaining days they together work
  : x = 3 := 
by
  sorry

end work_rate_problem_l19_19837


namespace bus_stoppage_time_l19_19409

noncomputable def travel_time_without_stoppages (v1 D : ℝ) : ℝ :=
  D / v1

noncomputable def travel_time_with_stoppages (v2 D : ℝ) : ℝ :=
  D / v2

def stoppage_time_per_hour (t1 t2 : ℝ) : ℝ :=
  (t2 - t1) * 60 / t2

theorem bus_stoppage_time (v1 v2 D : ℝ) (h1 : v1 = 60) (h2 : v2 = 20) (hD : D = 60) :
  stoppage_time_per_hour (travel_time_without_stoppages v1 D) (travel_time_with_stoppages v2 D) = 40 :=
by
  sorry

end bus_stoppage_time_l19_19409


namespace remainder_a3_mod_15_l19_19280

theorem remainder_a3_mod_15 
  (a : ℤ) 
  (h : a * a ≡ 1 [MOD 15]) : a ^ 3 ≡ 1 [MOD 15] :=
by
  sorry

end remainder_a3_mod_15_l19_19280


namespace region_perimeter_l19_19249

/-- Define the geometric conditions and perimeter conclusion based on given problem -/
theorem region_perimeter
  (right_angles : ∀ a b c d : ℝ, ∠ a b c d = π / 2)
  (length_ticks : ∀ s, s ∈ (finset.univ : finset ℝ) → s = 1)
  (total_area : ℝ = 72)
  : ∃ p : ℝ, p = 42.25 :=
by
  sorry

end region_perimeter_l19_19249


namespace jerry_remaining_debt_l19_19265

theorem jerry_remaining_debt :
  ∀ (paid_two_months_ago paid_last_month total_debt: ℕ),
  paid_two_months_ago = 12 →
  paid_last_month = paid_two_months_ago + 3 →
  total_debt = 50 →
  total_debt - (paid_two_months_ago + paid_last_month) = 23 :=
by
  intros paid_two_months_ago paid_last_month total_debt h1 h2 h3
  sorry

end jerry_remaining_debt_l19_19265


namespace range_of_log2_sqrt_sin_l19_19384

theorem range_of_log2_sqrt_sin (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 180) : 
  set.Icc 0 180 ⊆ (set.image (λ x : ℝ, real.log 2 (real.sqrt (real.sin (x * (real.pi / 180))))) (set.Icc 0 180)) :=
sorry

end range_of_log2_sqrt_sin_l19_19384


namespace decimal_to_fraction_l19_19787

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l19_19787


namespace jordan_bike_lock_l19_19268

-- Define the set of digits and subsets of odd and even digits
def S := {1, 2, 3, 4, 5, 6}
def O := {1, 3, 5}
def E := {2, 4, 6}

-- Define the alternating sequence property
def alternates (lst : List ℕ) : Prop :=
  lst.length = 6 ∧
  (lst.enum.map (λ (idx, val), if (idx % 2 = 0) then val ∈ O else val ∈ E)).and

-- The final theorem statement
theorem jordan_bike_lock (lst : List ℕ) :
  (∀ lst, alternates lst → std.set I (lst.elements = 6) → 1458)

end jordan_bike_lock_l19_19268


namespace arctan_series_sum_l19_19962

open Real

theorem arctan_series_sum (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range n, arctan (1 / (2 * (k + 1) ^ 2))) = arctan (n / (n + 1)) :=
sorry

end arctan_series_sum_l19_19962


namespace fixed_point_and_angle_of_line_l19_19198

noncomputable def fixed_point_and_angle_inclination (x y : ℝ) (α : ℝ) : Prop :=
  ∀ (m : ℝ), (y - 3 = m * (x - 4)) → (x = 4) ∧ (y = 3) ∧ (m = sqrt 3) ∧ (α = Real.arctan (sqrt 3))

theorem fixed_point_and_angle_of_line :
  fixed_point_and_angle_inclination 4 3 (π / 3) :=
begin
  sorry
end

end fixed_point_and_angle_of_line_l19_19198


namespace smallest_n_l19_19957

theorem smallest_n :
  ∃ (n : ℕ), n = 13 ∧ (∀ (a : ℕ → ℤ), (∀ i j, i ≤ j → a i = a j) →
  ∃ (s : finset ℕ), s.card = 9 ∧ (∃ (b : fin 9 → ℤ), (∀ i, b i ∈ {4, 7}) ∧ 9 ∣ finset.univ.sum (λ i : fin 9, b i * a (s.val i)))) :=
sorry

end smallest_n_l19_19957


namespace determine_k_l19_19155

theorem determine_k (k : ℝ) (h : 2 - 2^2 = k * (2)^2 + 1) : k = -3/4 :=
by
  sorry

end determine_k_l19_19155


namespace time_to_cross_same_direction_l19_19036

-- Defining the conditions
def speed_train1 : ℝ := 60 -- kmph
def speed_train2 : ℝ := 40 -- kmph
def time_opposite_directions : ℝ := 10.000000000000002 -- seconds 
def relative_speed_opposite_directions : ℝ := speed_train1 + speed_train2 -- 100 kmph
def relative_speed_same_direction : ℝ := speed_train1 - speed_train2 -- 20 kmph

-- Defining the proof statement
theorem time_to_cross_same_direction : 
  (time_opposite_directions * (relative_speed_opposite_directions / relative_speed_same_direction)) = 50 :=
by
  sorry

end time_to_cross_same_direction_l19_19036


namespace find_E_l19_19996

theorem find_E (A H S M E : ℕ) (h1 : A ≠ 0) (h2 : H ≠ 0) (h3 : S ≠ 0) (h4 : M ≠ 0) (h5 : E ≠ 0) 
  (cond1 : A + H = E)
  (cond2 : S + M = E)
  (cond3 : E = (A * M - S * H) / (M - H)) : 
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end find_E_l19_19996


namespace problem_statement_l19_19070

-- Define the given conditions
def varies_directly_inversely (R S T : ℝ) (c : ℝ) : Prop :=
  R = c * (S / (T * T))

noncomputable def find_s (R_goal T_goal : ℝ) : ℝ :=
  let R := 3
  let T := 2
  let S := 16
  let c := R * (T * T) / S in
  let S_goal := (R_goal * T_goal * T_goal) / c in
  S_goal

theorem problem_statement : find_s 50 5 = 5000 / 3 :=
by
  simp [find_s, varies_directly_inversely]
  sorry

end problem_statement_l19_19070


namespace min_sum_abc_l19_19011

theorem min_sum_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hprod : a * b * c = 2550) : a + b + c ≥ 48 :=
by sorry

end min_sum_abc_l19_19011


namespace arithmetic_sequence_sum_l19_19806

theorem arithmetic_sequence_sum :
  let a₁ := -5
  let aₙ := 40
  let n := 10
  (n : ℝ) = 10 →
  (a₁ : ℝ) = -5 →
  (aₙ : ℝ) = 40 →
  ∑ i in finset.range n, (a₁ + i * ((aₙ - a₁) / (n - 1))) = 175 :=
by
  intros
  sorry

end arithmetic_sequence_sum_l19_19806


namespace lcm_gcd_product_15_10_l19_19549

theorem lcm_gcd_product_15_10 : 
  let a := 15 in let b := 10 in 
  a = 3 * 5 ∧ b = 2 * 5 →
  Nat.lcm a b * Nat.gcd a b = 150 :=
by
  intros
  sorry

end lcm_gcd_product_15_10_l19_19549


namespace function_inequality_condition_l19_19190

open Real

theorem function_inequality_condition
  (a : ℝ)
  (h : a > 0)
  (f : ℝ → ℝ := λ x, x + a^2 / x)
  (g : ℝ → ℝ := λ x, x - log x)
  (H : ∀ (x1 : ℝ) (hx1: 0 < x1), ∀ (x2 : ℝ) (hx2: 1 ≤ x2 ∧ x2 ≤ exp 1), f x1 ≥ g x2) :
  a ≥ sqrt (exp 1 - 2) := 
sorry

end function_inequality_condition_l19_19190


namespace marble_choice_l19_19031

def numDifferentGroupsOfTwoMarbles (red green blue : ℕ) (yellow : ℕ) (orange : ℕ) : ℕ :=
  if (red = 1 ∧ green = 1 ∧ blue = 1 ∧ yellow = 2 ∧ orange = 2) then 12 else 0

theorem marble_choice:
  let red := 1
  let green := 1
  let blue := 1
  let yellow := 2
  let orange := 2
  numDifferentGroupsOfTwoMarbles red green blue yellow orange = 12 :=
by
  dsimp[numDifferentGroupsOfTwoMarbles]
  split_ifs
  · rfl
  · sorry

-- Ensure the theorem type matches the expected Lean 4 structure.
#print marble_choice

end marble_choice_l19_19031


namespace calculation_correct_l19_19497

theorem calculation_correct : (3.456 - 1.234) * 0.5 = 1.111 :=
by
  sorry

end calculation_correct_l19_19497


namespace two_point_seven_five_as_fraction_l19_19801

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l19_19801


namespace unique_positive_integer_triples_l19_19135

theorem unique_positive_integer_triples (a b c : ℕ) (h1 : ab + 3 * b * c = 63) (h2 : ac + 3 * b * c = 39) : 
∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + 3 * b * c = 63 ∧ ac + 3 * b * c = 39 :=
by sorry

end unique_positive_integer_triples_l19_19135


namespace angle_sum_eq_l19_19982

structure Quadrilateral (α : Type _) :=
(E L M I : α)

variables {α : Type _} [EuclideanGeometry α]

-- Definition of the points E, L, M, I
variables (ELMI : Quadrilateral α)
variables {E L M I : α} [pointOnLine L M] [pointOnLine M I]

-- Condition 1: quadrilateral ELMI defined
def ELMI_exists : Prop := true

-- Condition 2: The sum of angles ∠LME and ∠MEI is 180 degrees
def sum_angles_eq_180 (L M E I : α) [IsAngle (LME : EUclideanAngle)] [IsAngle (MEI : EUclideanAngle)] : Prop :=
  ∠ LME + ∠ MEI = 180

-- Condition 3: EL = EI + LM
def length_rel (L M E I : α) : Prop :=
  dist E L = dist E I + dist L M

-- Proof statement: Prove that the sum of angles ∠LEM and ∠EMI is ∠MIE
theorem angle_sum_eq (α : Type _) [EuclideanGeometry α] (E L M I : α)
                      [IsAngle (LEM : EuclideanAngle)] [IsAngle (EMI : EuclideanAngle)] [IsAngle (MIE : EuclideanAngle)]
                      (sum_angles_180 : sum_angles_eq_180 L M E I) (length_cond : length_rel L M E I) :
  ∠ LEM + ∠ EMI = ∠ MIE := sorry

end angle_sum_eq_l19_19982


namespace train_passes_man_in_time_l19_19869

noncomputable def train_pass_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let train_speed_mps := train_speed * (5 / 18)
  let man_speed_mps := man_speed * (5 / 18)
  let relative_speed := train_speed_mps + man_speed_mps
  train_length / relative_speed

theorem train_passes_man_in_time :
  train_pass_time 110 56 6 ≈ 6.38 := sorry

end train_passes_man_in_time_l19_19869


namespace integral_solutions_l19_19541

theorem integral_solutions :
  ∀ (m n : ℤ), (m^2 + n) * (m + n^2) = (m + n)^3 →
    (m = 0 ∨
     n = 0 ∨
     (m, n) = (-5, 2) ∨
     (m, n) = (-1, 1) ∨
     (m, n) = (1, -1) ∨
     (m, n) = (2, -5) ∨
     (m, n) = (4, 11) ∨
     (m, n) = (5, 7) ∨
     (m, n) = (7, 5) ∨
     (m, n) = (11, 4)) :=
begin
  intros m n h,
  sorry
end

end integral_solutions_l19_19541


namespace min_square_area_for_rectangles_l19_19783

theorem min_square_area_for_rectangles :
  ∀ (a b c d : ℕ), 
    (a = 2 ∧ b = 4 ∧ c = 3 ∧ d = 5) →
    ∃ s : ℕ, (∀ (x y : ℕ), ((x = a ∨ x = b) ∧ (y = c ∨ y = d)) → x + y ≤ s) ∧ s^2 = 25 :=
by
  { intros a b c d h,
    rcases h with ⟨ha, hb, hc, hd⟩,
    use 5,
    split,
    intros x y hx hy,
    rcases hx with ⟨hx1 | hx2⟩; rcases hy with ⟨hy1 | hy2⟩;
    (try {simp [hx1, ha, hb]}), (try {simp [hx2, ha, hb]}), (try {simp [hy1, hc, hd]}), (try {simp [hy2, hc, hd]}),
    -- other cases
    sorry } 

end min_square_area_for_rectangles_l19_19783


namespace trajectory_and_min_area_l19_19179

theorem trajectory_and_min_area (C : ℝ → ℝ → Prop) (P : ℝ × ℝ → Prop)
  (l : ℝ → ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (k : ℝ) : 
  (∀ x y, P (x, y) ↔ x ^ 2 = 4 * y) → 
  P (0, 1) →
  (∀ y, l y = -1) →
  F = (0, 1) →
  (∀ x1 y1 x2 y2, x1 + x2 = 4 * k → x1 * x2 = -4 →
    M (x1, y1) (x2, y2) = (2 * k, -1)) →
  (min_area : ℝ) → 
  min_area = 4 :=
by
  intros
  sorry

end trajectory_and_min_area_l19_19179


namespace product_sum_estimation_l19_19207

theorem product_sum_estimation (
    x y z : ℝ,
    h1 : x = 1.6,
    h2 : y = 9.2,
    h3 : z = 3.1
  ):
  let exact_value := x * y + z in
  let rounded_value_E := (⟶ (floor x).to_real * (floor y).to_real + (ceil z).to_real) in
  rounded_value_E < exact_value :=
  sorry

end product_sum_estimation_l19_19207


namespace solutionA_solutionB_solutionC_solutionD_l19_19057

-- Define the inequalities
def inequalityA (x : ℝ) : Prop := x^2 - 12 * x + 20 > 0
def inequalityB (x : ℝ) : Prop := x^2 - 5 * x + 6 < 0
def inequalityC (x : ℝ) : Prop := 9 * x^2 - 6 * x + 1 > 0
def inequalityD (x : ℝ) : Prop := -2 * x^2 + 2 * x - 3 > 0

-- Statements of the mathematical problems
theorem solutionA : ∀ x : ℝ, inequalityA x ↔ (x < 2 ∨ x > 10) := 
by sorry

theorem solutionB : ∀ x : ℝ, inequalityB x ↔ (2 < x ∧ x < 3) :=
by sorry

theorem solutionC : ∀ x : ℝ, (inequalityC x → x ≠ 1 / 3) ∧ (∃ x, x ∈ set.Ioo (-∞) (1 / 3) ∨ x ∈ set.Ioo (1 / 3) ∞) :=
by sorry

theorem solutionD : ∀ x : ℝ, ¬inequalityD x :=
by sorry

end solutionA_solutionB_solutionC_solutionD_l19_19057


namespace Yi_wins_strategy_l19_19678

theorem Yi_wins_strategy 
  (a : Fin 13 → ℕ)
  (h1 : ∀ i : Fin 12, 100 ≤ a i ∧ a i < 1000 ∧ a i < a (Fin.succ i))
  : ∃ (i j k : Fin 8), 3 < (a (Fin.succ i)) / (a i) + (a (Fin.succ (Fin.succ i))) / (a (Fin.succ i)) + (a (Fin.succ (Fin.succ (Fin.succ i)))) / (a (Fin.succ (Fin.succ i))) < 4 :=
sorry

end Yi_wins_strategy_l19_19678


namespace mountains_still_intact_at_end_of_year_l19_19854

theorem mountains_still_intact_at_end_of_year
  (initial_volcanoes : ℕ)
  (percent_erupted_first_two_months : ℕ)
  (percent_erupted_halfway : ℕ)
  (percent_erupted_end_year : ℕ)
  (initial_volcanoes = 200)
  (percent_erupted_first_two_months = 20)
  (percent_erupted_halfway = 40)
  (percent_erupted_end_year = 50) :
  let volcanoes_after_two_months := initial_volcanoes - (initial_volcanoes * percent_erupted_first_two_months / 100)
  let volcanoes_after_half_year := volcanoes_after_two_months - (volcanoes_after_two_months * percent_erupted_halfway / 100)
  let volcanoes_end_year := volcanoes_after_half_year - (volcanoes_after_half_year * percent_erupted_end_year / 100)
  volcanoes_end_year = 48 :=
by
  sorry

end mountains_still_intact_at_end_of_year_l19_19854


namespace value_of_x_l19_19223

theorem value_of_x (x : ℝ) : 16^(x + 2) = 1000 + 16^x → x = 0.5 :=
by
  intro h
  sorry

end value_of_x_l19_19223


namespace number_of_real_numbers_eq_91_l19_19963

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem number_of_real_numbers_eq_91 :
  (∃ count : ℕ, 
    count = finset.card { x | 1 ≤ x ∧ x ≤ 10 ∧ (fractional_part x)^2 = fractional_part (x^2)}) 
  ∧ count = 91 :=
begin
  -- The proof would go here
  sorry,
end

end number_of_real_numbers_eq_91_l19_19963


namespace radius_of_circle_is_2_l19_19486

-- Define the circle, points and the properties given in the problem
def circle (O : Type) : Type := sorry
def point (O : Type) : Type := sorry
def diameter (A B : point O) (c : circle O) : Prop := sorry
def chord (C D : point O) (c : circle O) : Prop := sorry
def intersect_at (A B C D P : point O) : Prop := sorry
def angle_is (P C D : point O) (deg : ℝ) : Prop := sorry
def sum_of_squares (x y : ℝ) (sum : ℝ) : Prop := sorry

-- Define the radius of the circle
def radius (c : circle O) : ℝ := sorry

-- Given conditions as Lean hypotheses
variables (O : Type) (c : circle O)
variables (A B C D P : point O)

-- Diameter and chord conditions
hypothesis (h1 : diameter A B c)
hypothesis (h2 : chord C D c)

-- Intersect condition
hypothesis (h3 : intersect_at A B C D P)

-- Angle condition
hypothesis (h4 : angle_is A P D 45)

-- Sum of squares condition
hypothesis (h5 : sum_of_squares (dist P C) (dist P D) 8)

-- The theorem stating the radius of the circle
theorem radius_of_circle_is_2 : radius c = 2 :=
sorry

end radius_of_circle_is_2_l19_19486


namespace less_likely_outcome_l19_19968

noncomputable def red_balls := 8
noncomputable def white_balls := 2
noncomputable def total_balls := red_balls + white_balls

theorem less_likely_outcome :
  (white_balls / total_balls) < (red_balls / total_balls) :=
by
  calc
    white_balls / total_balls = 2 / 10 : by rw [white_balls, total_balls]
    ... < 8 / 10 : by norm_num
    ... = red_balls / total_balls : by rw [red_balls, total_balls]

end less_likely_outcome_l19_19968


namespace two_pt_seven_five_as_fraction_l19_19797

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l19_19797


namespace find_a_minus_c_l19_19411

section
variables (a b c : ℝ)
variables (h₁ : (a + b) / 2 = 110) (h₂ : (b + c) / 2 = 170)

theorem find_a_minus_c : a - c = -120 :=
by
  sorry
end

end find_a_minus_c_l19_19411


namespace percentage_of_primes_divisible_by_2_l19_19055

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def divisible_by (n k : ℕ) : Prop := k ∣ n

def percentage_divisible_by (k : ℕ) (lst : List ℕ) : ℚ :=
  (lst.filter (divisible_by k)).length / lst.length * 100

theorem percentage_of_primes_divisible_by_2 : 
  percentage_divisible_by 2 primes_less_than_20 = 12.5 := by
  sorry

end percentage_of_primes_divisible_by_2_l19_19055


namespace ratio_platform_to_train_length_l19_19871

variable (L P t : ℝ)

-- Definitions based on conditions
def train_has_length (L : ℝ) : Prop := true
def train_constant_velocity : Prop := true
def train_passes_pole_in_t_seconds (L t : ℝ) : Prop := L / t = L
def train_passes_platform_in_4t_seconds (L P t : ℝ) : Prop := L / t = (L + P) / (4 * t)

-- Theorem statement: ratio of the length of the platform to the length of the train is 3:1
theorem ratio_platform_to_train_length (h1 : train_has_length L) 
                                      (h2 : train_constant_velocity) 
                                      (h3 : train_passes_pole_in_t_seconds L t)
                                      (h4 : train_passes_platform_in_4t_seconds L P t) :
  P / L = 3 := 
by sorry

end ratio_platform_to_train_length_l19_19871


namespace area_closed_figure_l19_19746

def f (x : ℝ) : ℝ := 2 / (x^2 - 1)

theorem area_closed_figure :
  ∫ x in 2..3, f x = Real.log (3/2) := by
  sorry

end area_closed_figure_l19_19746


namespace y_intercept_line_b_l19_19297

-- Definitions of the lines and points as per given conditions
def line_a (x : ℝ) : ℝ := -3 * x + 6
def point_b : ℝ × ℝ := (4, -1)

-- Definition of line_b, which shares the same slope as line_a and passes through point_b
noncomputable def line_b (x : ℝ) : ℝ := -3 * x + (let c := (-1 + 3 * 4) in c)

-- Theorem stating that the y-intercept of line_b is 11
theorem y_intercept_line_b : line_b 0 = 11 :=
by
  -- To be filled with proof
  sorry

end y_intercept_line_b_l19_19297


namespace log_base_a_of_b_l19_19433

-- Define the variables (including noncomputable due to logarithms)
noncomputable theory

variables {a b : ℝ}

-- Define the conditions
def diameter (a : ℝ) := log 10 (a^3)
def circumference (b : ℝ) := log 10 (b^6)

-- Statement of the theorem
theorem log_base_a_of_b (a b : ℝ) (h1 : diameter a = log 10 (a^3)) (h2 : circumference b = log 10 (b^6)) :
  log a b = π / 2 := 
by 
  sorry

end log_base_a_of_b_l19_19433


namespace largest_divisor_of_m_l19_19225

theorem largest_divisor_of_m (m : ℤ) (hm_pos : 0 < m) (h : 33 ∣ m^2) : 33 ∣ m :=
sorry

end largest_divisor_of_m_l19_19225


namespace D_n_formula_l19_19702

-- Defining the problem conditions
def is_permutation (π : List ℕ) (n : ℕ) : Prop :=
  π.perm (List.range n)

def no_number_retains_position (π : List ℕ) : Prop :=
  ∀ i, i < π.length → π.nthLe i sorry ≠ i  -- nthLe requires proof of bounds

-- Defining D_n
def D_n (n : ℕ) : ℕ :=
  List.permutations (List.range n)
  |>.filter no_number_retains_position
  |>.length

-- The theorem to be proved
theorem D_n_formula (n : ℕ) :
  D_n n = n.factorial * (List.range (n + 1)).sum (λ k => (-1 : ℤ)^k / k.factorial) :=
sorry

end D_n_formula_l19_19702


namespace ceil_sqrt_sum_proof_l19_19538

noncomputable def ceil_sqrt_sum : ℝ :=
  (Real.ceil (sqrt 5)) + (Real.ceil (sqrt 6)) +
  (Real.ceil (sqrt 7)) + (Real.ceil (sqrt 8)) + (Real.ceil (sqrt 9)) +
  (Real.ceil (sqrt 10)) + (Real.ceil (sqrt 11)) + (Real.ceil (sqrt 12)) +
  (Real.ceil (sqrt 13)) + (Real.ceil (sqrt 14)) + (Real.ceil (sqrt 15)) +
  (Real.ceil (sqrt 16)) + (Real.ceil (sqrt 17)) + (Real.ceil (sqrt 18)) +
  (Real.ceil (sqrt 19)) + (Real.ceil (sqrt 20)) + (Real.ceil (sqrt 21)) +
  (Real.ceil (sqrt 22)) + (Real.ceil (sqrt 23)) + (Real.ceil (sqrt 24)) +
  (Real.ceil (sqrt 25)) + (Real.ceil (sqrt 26)) + (Real.ceil (sqrt 27)) +
  (Real.ceil (sqrt 28)) + (Real.ceil (sqrt 29)) + (Real.ceil (sqrt 30)) +
  (Real.ceil (sqrt 31)) + (Real.ceil (sqrt 32)) + (Real.ceil (sqrt 33)) +
  (Real.ceil (sqrt 34)) + (Real.ceil (sqrt 35)) + (Real.ceil (sqrt 36))

theorem ceil_sqrt_sum_proof : ceil_sqrt_sum = 154 :=
by {
  sorry
}

end ceil_sqrt_sum_proof_l19_19538


namespace max_teams_tournament_l19_19468

theorem max_teams_tournament (n : ℕ) :
  (∀ i j, (i ≠ j → ¬(S_i ⊆ S_j ∨ S_j ⊆ S_i))) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → S_i ≠ ∅ ∧ S_i ⊂ {1, 2, 3, 4}) → 
  n ≤ 6 :=
by
  sorry

end max_teams_tournament_l19_19468


namespace two_point_seven_five_as_fraction_l19_19799

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l19_19799


namespace average_speed_l19_19062

-- Define the problem conditions and provide the proof statement
theorem average_speed (D : ℝ) (hD0 : D > 0) : 
  let speed_1 := 80
  let speed_2 := 24
  let speed_3 := 60
  let time_1 := (D / 3) / speed_1
  let time_2 := (D / 3) / speed_2
  let time_3 := (D / 3) / speed_3
  let total_time := time_1 + time_2 + time_3
  let average_speed := D / total_time
  average_speed = 720 / 17 := 
by
  sorry

end average_speed_l19_19062


namespace max_sides_convex_polygon_l19_19447

-- Define the problem conditions in Lean
def isValidPolygon (n : ℕ) (polygon : Type) :=
  (polygon.hasSideOfLength1) ∧ (∀ (i j : ℕ), 1 ≤ i < j ≤ n → (polygon.diagonalLength i j).isInteger) 

-- Lean statement for the maximum number of sides given the conditions
theorem max_sides_convex_polygon : ∀ (polygon : Type), isValidPolygon n polygon →
  (∃ (m : ℕ), m ≤ 5 ∧ isValidPolygon m polygon) :=
sorry

end max_sides_convex_polygon_l19_19447


namespace triangle_area_l19_19988

theorem triangle_area (A B C : Type)
  [metric_space A] [metric_space B] [metric_space C]
  (AB BC CA : Real)
  (h_AB : AB = sqrt 3)
  (h_BC : BC = 1)
  (h_sin_cos : sin C = sqrt 3 * cos C) :
  let area := 0.5 * (AC * BC * sin C)
  in area = sqrt 3 / 2 :=
sorry

end triangle_area_l19_19988


namespace shortest_side_of_triangle_l19_19473

noncomputable def triangle_with_integer_sides (a b c : ℕ) :=
  a = 15 ∧ a + b + c = 40 ∧ 
  ∃ (k : ℕ), k^2 = 20 * (20 - 15) * (20 - b) * (20 - (25 - b))

theorem shortest_side_of_triangle : ∃ (b : ℕ), 
  triangle_with_integer_sides 15 b (25 - b) ∧ 
  (∀ (b' : ℕ), triangle_with_integer_sides 15 b' (25 - b') → b ≤ b') :=
begin
  sorry
end

end shortest_side_of_triangle_l19_19473


namespace x_finishes_work_alone_in_18_days_l19_19413

theorem x_finishes_work_alone_in_18_days
  (y_days : ℕ) (y_worked : ℕ) (x_remaining_days : ℝ)
  (hy : y_days = 15) (hy_worked : y_worked = 10) 
  (hx_remaining : x_remaining_days = 6.000000000000001) :
  ∃ (x_days : ℝ), x_days = 18 :=
by 
  sorry

end x_finishes_work_alone_in_18_days_l19_19413


namespace range_of_f_interval_of_monotonicity_l19_19200

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x - 1

def domain : Set ℝ := Set.Icc 0 (Real.pi / 2)

theorem range_of_f : Set.range (λ x, f x) = Set.Icc 0 (1 / 4) :=
by
  sorry

theorem interval_of_monotonicity :
  ∀ (x₁ x₂ : ℝ), x₁ ∈ domain → x₂ ∈ domain → x₁ ≤ x₂ → (f x₁ ≤ f x₂ ↔ x₂ ≤ Real.pi / 6) :=
by
  sorry

end range_of_f_interval_of_monotonicity_l19_19200


namespace middle_seat_occupied_by_David_l19_19152

-- Define the given conditions
def friends : Type := {Becca, David, Norah, Olivia, Rick : String}
def seats : Type := Fin 5

-- Proving that the middle seat is occupied by a specific friend given the constraints
theorem middle_seat_occupied_by_David :
  ∃ (seat : seats) (in_first : Olivia = seats.val 1) 
    (david_in_front_norah : ∃ d n : seats, David = d ∧ Norah = n ∧ (d.val + 1) = n.val)
    (rick_behind_david : ∃ r d : seats, Rick = r ∧ David = d ∧ d.val < r.val)
    (at_least_two_between : ∃ b r : seats, Becca = b ∧ Rick = r ∧ (abs (b.val - r.val) ≥ 2)), 
  David = seats.val 2 :=
by
  sorry

end middle_seat_occupied_by_David_l19_19152


namespace count_numbers_in_range_l19_19480

def does_not_contain_3 (n : ℕ) : Prop :=
  ¬(3 ∈ (n.digits 10))

def count_numbers_without_3 (n : ℕ) : ℕ :=
  (List.range n).filter does_not_contain_3 |>.length

theorem count_numbers_in_range :
  count_numbers_without_3 300 = 242 :=
by
  sorry

end count_numbers_in_range_l19_19480


namespace length_of_BC_l19_19434

theorem length_of_BC
  (O A B C : EuclideanGeometry.Point)
  (circle : EuclideanGeometry.Circle O 8)
  (triangleABC : EuclideanGeometry.Triangle A B C)
  (h1 : triangleABC.is_isosceles (A = B))
  (h2 : EuclideanGeometry.PerpendicularBisector O A B (line BC))
  (O_inside_ABC : O ∈ interior (triangleABC))
  (area_circle : EuclideanGeometry.area circle = 64 * π)
  (OA_distance : dist O A = 8) :
  dist B C = 8 * sqrt 2 :=
by
  sorry

end length_of_BC_l19_19434


namespace sequenceOfFractions_exists_l19_19210

theorem sequenceOfFractions_exists (n : ℕ) (h : n > 0) :
  ∃ (seq : list ℚ), 
    (seq = (list.range (n+1)).map (λ i, (i : ℚ)) ∧ seq.length = n) ∧
    ∀ x ∈ seq, ∃ d, d ≠ 0 ∧ x = (1 : ℚ) / d := 
by sorry

end sequenceOfFractions_exists_l19_19210


namespace correct_option_l19_19392

variable {α : Type*} [CommSemiring α] (a b : α)

theorem correct_option : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  calc
    (2 * a * b^2)^3 = (2^3) * (a^3) * (b^2)^3 : by rw [mul_pow, mul_pow, pow_mul]
    ... = 8 * a^3 * b^6 : by norm_num

end correct_option_l19_19392


namespace math_problem_l19_19117

theorem math_problem :
  (π - 2)^0 + (-1 / 2)^(-2) - 2 * Real.sin (Real.pi / 3) = 5 - Real.sqrt 3 :=
by
  have h1 : (π - 2)^0 = 1 := by
    -- (π - 2)^0 = 1 because any number to the power of 0 is 1
    sorry
  have h2 : (-1 / 2)^(-2) = 4 := by
    -- (-1 / 2)^(-2) = 4 because reciprocal and squaring of (-1 / 2)
    sorry
  have h3 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := by
    -- The sine of 60 degrees (pi/3 radians) equals sqrt(3)/2
    sorry
  rw [h1, h2, h3]
  -- Substitute h1, h2, and h3 into the original goal
  sorry

end math_problem_l19_19117


namespace repeated_root_cubic_l19_19960

theorem repeated_root_cubic (p : ℝ) :
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ (9 * x^2 - 2 * (p + 1) * x + 4 = 0)) →
  (p = 5 ∨ p = -7) :=
by
  sorry

end repeated_root_cubic_l19_19960


namespace ratio_of_integers_l19_19809

theorem ratio_of_integers (a b : ℤ) (h : 1996 * a + b / 96 = a + b) : a / b = 1 / 2016 ∨ b / a = 2016 :=
by
  sorry

end ratio_of_integers_l19_19809


namespace area_triangle_MP_l19_19979

-- Definitions based on the conditions
def parabola (x y : ℝ) := y^2 = 4 * x

def directrix (x y : ℝ) := x = -1

def focus := (1 : ℝ, 0 : ℝ)

def perpendicular_distance (P F : ℝ × ℝ) : ℝ :=
  (P.1 - F.1)^2 + (P.2 - F.2)^2

-- The theorem stating the result
theorem area_triangle_MP (P : ℝ × ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) (hP : parabola P.1 P.2) (hM : directrix M.1 M.2) (hPF : perpendicular_distance P F = 25) : 
  1/2 * real.sqrt (P.1 - M.1) * abs(P.2) = 10 := 
  sorry

end area_triangle_MP_l19_19979


namespace find_volume_l19_19907

-- Definition of input conditions
def s : ℝ := 4
def h : ℝ := 9

-- Definition of areas of hexagon base and upper face
def A_base : ℝ := (3 * Real.sqrt 3 / 2) * s^2
def A_upper : ℝ := (3 * Real.sqrt 3 / 2) * (1.5 * s)^2

-- Average area definition
def A_avg : ℝ := (A_base + A_upper) / 2

-- Volume definition
def volume : ℝ := A_avg * h

-- Theorem stating the volume of the solid
theorem find_volume : volume = 351 * Real.sqrt 3 := 
by {
  -- We skip the proof here as instructed
  sorry
}

end find_volume_l19_19907


namespace range_of_f_l19_19191

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc (1 : ℝ) (Real.sqrt 2) := 
by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc 1 (Real.sqrt 2) := sorry
  exact this x hx

end range_of_f_l19_19191


namespace range_of_a_l19_19581

def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a - 1) * x0 + 1 < 0

theorem range_of_a (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : 
  -1 ≤ a ∧ a ≤ 1 ∨ 3 < a :=
sorry

end range_of_a_l19_19581


namespace beetle_speed_is_correct_l19_19881

open Real

noncomputable def ant_distance : ℝ := 600
noncomputable def ant_time_min : ℝ := 12
noncomputable def beetle_distance := ant_distance * (1 - 0.15)
noncomputable def beetle_distance_km := beetle_distance / 1000
noncomputable def ant_time_hr := ant_time_min / 60

theorem beetle_speed_is_correct :
  let beetle_speed := beetle_distance_km / ant_time_hr in
  beetle_speed = 2.55 := by
  sorry

end beetle_speed_is_correct_l19_19881


namespace polynomial_remainder_l19_19688

-- Definitions of the polynomial Q and its conditions
variable {R : Type*} [CommRing R]
variable {Q : R → R}

-- Conditions given in the problem
def cond1 : Prop := Q 10 = 50
def cond2 : Prop := Q 50 = 10

-- Statement to prove the remainder is -x + 60
theorem polynomial_remainder (h1 : cond1) (h2 : cond2) : ∃ a b : R, (a = -1) ∧ (b = 60) ∧ (∀ x, Q x % ((x - 10) * (x - 50)) = a * x + b) :=
sorry

end polynomial_remainder_l19_19688


namespace quadrilateral_is_rhombus_l19_19883

theorem quadrilateral_is_rhombus
  (A B C D P Q R S : Type)
  [convex_quadrilateral A B C D]
  [isosceles_triangle A P B]
  [isosceles_triangle B Q C]
  [isosceles_triangle C R D]
  [isosceles_triangle D S A]
  [rectangle P Q R S]
  (PQ_ne_QR : P ≠ Q) : is_rhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l19_19883


namespace two_point_seven_five_as_fraction_l19_19800

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l19_19800


namespace area_Q1Q2Q3Q4_is_8_l19_19171

-- Definition of a regular octagon with apothem
def regular_octagon (apothem : ℝ) (sides : ℕ := 8) :=
  ∀ i : fin sides, distance (center i) (center (i+1)) = apothem

-- Midpoint function
def midpoint (P1 P2 : ℝ×ℝ) : ℝ×ℝ := ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2)

-- Definition of the midpoints Qi
def Q_i (P : ℕ → ℝ×ℝ) (i : ℕ) : ℝ×ℝ := midpoint (P (2*i-1)) (P (2*i))

noncomputable def area_of_quadrilateral (Q1 Q2 Q3 Q4 : ℝ×ℝ) : ℝ := sorry

theorem area_Q1Q2Q3Q4_is_8 (P : ℕ → ℝ×ℝ) (apothem : ℝ) (h : regular_octagon apothem P) :
  area_of_quadrilateral (Q_i P 1) (Q_i P 2) (Q_i P 3) (Q_i P 4) = 8 :=
by
  sorry

end area_Q1Q2Q3Q4_is_8_l19_19171


namespace money_made_is_40_l19_19495

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l19_19495


namespace _l19_19679

def Joey_chloe_theorem 
  (Chloe_age : ℕ)
  (Joey_age : ℕ)
  (Max_age : ℕ)
  (birthday_count : ℕ): 
  Prop :=
  ∃ (n : ℕ), Max_age = 2 ∧ Joey_age = Chloe_age + 2 
   ∧ birthday_count = 6 
   ∧ (Joey_age + n) % (Max_age + n) == 0 
   ∧ ∃ (next_n : ℕ), (Chloe_age + next_n) % (Max_age + next_n) == 0 
   ∧ (Chloe_age + next_n).digits.sum = 10

end _l19_19679


namespace volume_of_open_box_from_sheet_l19_19092

def volume_of_box (length width height : ℕ) := length * width * height

theorem volume_of_open_box_from_sheet :
  ∀ (original_length original_width cut_length : ℕ),
    original_length = 48 ∧ original_width = 38 ∧ cut_length = 8 →
    volume_of_box (original_length - 2 * cut_length) (original_width - 2 * cut_length) cut_length = 5632 :=
by
  intros original_length original_width cut_length h
  cases h with hl hw
  cases hw with hw hc
  simp [hl, hw, hc, volume_of_box]
  sorry

end volume_of_open_box_from_sheet_l19_19092


namespace simplify_and_rationalize_l19_19318

theorem simplify_and_rationalize :
  (∀ (a b c d e f : ℝ), a = real.sqrt 3 → b = real.sqrt 4 → 
                        c = real.sqrt 5 → d = real.sqrt 6 → 
                        e = real.sqrt 8 → f = real.sqrt 9 → 
   ((a / b) * (c / d) * (e / f) = real.sqrt 15 / 9)) :=
by
  intros a b c d e f ha hb hc hd he hf,
  -- Starting with the original problem
  have h1 : a = real.sqrt 3 := ha,
  have h2 : b = real.sqrt 4 := hb,
  have h3 : c = real.sqrt 5 := hc,
  have h4 : d = real.sqrt 6 := hd,
  have h5 : e = real.sqrt 8 := he,
  have h6 : f = real.sqrt 9 := hf,
  -- result to be filled with proof steps
  sorry

end simplify_and_rationalize_l19_19318


namespace cube_divides_space_l19_19449

-- Defining the problem: A cube divides the space into parts
def divides_space_into_parts (n : ℕ) : Prop :=
  let cube_planes_division := 27
  n = cube_planes_division

theorem cube_divides_space : divides_space_into_parts 27 :=
begin
  sorry
end

end cube_divides_space_l19_19449


namespace computer_room_arrangements_l19_19774

theorem computer_room_arrangements (n : ℕ) (h : n = 6) : 
  ∃ arrangements : ℕ, arrangements = 2^6 - (1 + 6) ∧ arrangements = 57 := 
by 
  use 57
  simp [h]
  sorry

end computer_room_arrangements_l19_19774


namespace madeline_biked_more_l19_19708

def madeline_speed : ℕ := 12
def madeline_time : ℕ := 3
def max_speed : ℕ := 15
def max_time : ℕ := 2

theorem madeline_biked_more : (madeline_speed * madeline_time) - (max_speed * max_time) = 6 := 
by 
  sorry

end madeline_biked_more_l19_19708


namespace cannot_be_written_as_sum_of_two_elements_of_A_l19_19565

open Finset

def A : Finset ℕ := {1, 2, 3, 5, 8, 13, 21, 34, 55}

theorem cannot_be_written_as_sum_of_two_elements_of_A :
  let range_numbers := Finset.filter (λ n, 3 ≤ n ∧ n ≤ 89) (Finset.range 90)
  let sums_of_A := (A.product A).image (λ (p : ℕ × ℕ), p.1 + p.2)
  (range_numbers.card - sums_of_A.card) = 51 :=
by sorry

end cannot_be_written_as_sum_of_two_elements_of_A_l19_19565


namespace distance_to_line_eq_l19_19148

variable (P : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ)

def line (A B : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (A.1 + t * B.1, A.2 + t * B.2, A.3 + t * B.3)

def distance (P₁ P₂ : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P₁.1 - P₂.1) ^ 2 + (P₂.2 - P₂.2) ^ 2 + (P₁.3 - P₂.3) ^ 2)

theorem distance_to_line_eq : 
  distance (2, 3, 4) (line (5, 6, 8) (4, 3, -3) (-9 / 34)) = Real.sqrt 1501 / 17 :=
by
  sorry

end distance_to_line_eq_l19_19148


namespace part1_part2_part3_l19_19569

variables {α β k : ℝ} (h : k > 0)

/-- Define vectors a and b given angles alpha and beta. --/
def vec_a := (Complex.exp (Complex.I * α)).re_im
def vec_b := (Complex.exp (Complex.I * β)).re_im

/-- Condition given in the problem. --/
def condition := ((k * vec_a + vec_b).norm = √3 * (vec_a - k * vec_b).norm)

/-- Goal: Expressing dot product in terms of alpha and beta. --/
theorem part1 (h_cond : condition) : dot_product vec_a vec_b = Real.cos (α - β) :=
sorry

/-- Goal: Expressing dot product in terms of k. --/
theorem part2 (h_cond : condition) : dot_product vec_a vec_b = (k^2 + 1) / (4 * k) :=
sorry

/-- Goal: Find the minimum value of the dot product and the corresponding angle theta. --/
theorem part3 (h_cond : condition) : 
  ∃ θ (hθ : 0 ≤ θ ∧ θ ≤ Real.pi), 
  dot_product vec_a vec_b = 1 / 2 ∧ Real.cos θ = 1 / 2 ∧ θ = Real.pi / 3 :=
sorry

end part1_part2_part3_l19_19569


namespace total_sales_correct_l19_19116

-- Define constants representing sales for different months.
variable (January February March April : ℕ)

-- Define the conditions given in the problem.
axiom jan_sales_low : January
axiom feb_sales : February = 2 * January
axiom mar_sales : March = (5 / 4) * February
axiom apr_sales : April = March - (1 / 10) * March
axiom mar_sales_value : March = 12100

-- Define the total sales.
def total_sales : ℕ := January + February + March + April

-- Prove that the total sales from January to April is equal to 37510.
theorem total_sales_correct : total_sales January February March April = 37510 := by
  sorry

end total_sales_correct_l19_19116


namespace find_m_n_l19_19429

noncomputable def f (a b c d z: ℂ) : ℂ := (a * z + b) / (c * z + d)

theorem find_m_n (a b c d : ℂ)
  (h1 : f a b c d 1 = complex.I)
  (h2 : f a b c d 2 = -1)
  (h3 : f a b c d 3 = -complex.I)
  (h_relatively_prime : ∀ m n : ℕ, m > 0 → n > 0 → nat.gcd m n = 1 → f a b c d 4.re = (m : ℝ)/(n : ℝ)) :
  m^2 + n^2 = 34 := sorry

end find_m_n_l19_19429


namespace locus_of_M_is_circle_l19_19487

-- Define the data structure for points on the parabola
structure PointOnParabola :=
  (x : ℝ)
  (y : ℝ)
  (h : y^2 = 4 * p * x)

noncomputable def parabola_locus (p : ℝ) (h : 0 < p) :=
  ∃ M : PointOnParabola, 
    ∀ A B : PointOnParabola,
    A.x ≠ 0 → B.x ≠ 0 → 
    (∃ M : ℝ × ℝ, 
      (A.x * B.x) + A.y * B.y = 0 ∧
      (M.1 * (B.y - A.y) + M.2 * ((A.y^2 - B.y^2) / (4 * p))) = 0 →
       M.1^2 + M.2^2 = 4 * p * M.1)

-- The main theorem statement
theorem locus_of_M_is_circle : ∀ (p : ℝ) (h : 0 < p),
  parabola_locus p h :=
begin
  intros p h,
  -- Proof will be provided here
  sorry
end

end locus_of_M_is_circle_l19_19487


namespace circle_area_given_conditions_l19_19671

noncomputable def circle_area (r : ℝ) : ℝ := π * r^2

theorem circle_area_given_conditions (A B C D E F O : ℝ) (h1 : ∀ P Q : ℝ, P ≠ Q → ∃! R : ℝ, dist P R = dist R Q)
  (h2 : E - D = 3) (h3 : F - E = 9) : circle_area 12 = 144 * π :=
by
  sorry

end circle_area_given_conditions_l19_19671


namespace cos_sum_is_one_or_cos_2a_l19_19187

open Real

theorem cos_sum_is_one_or_cos_2a (a b : ℝ) (h : ∫ x in a..b, sin x = 0) : cos (a + b) = 1 ∨ cos (a + b) = cos (2 * a) :=
  sorry

end cos_sum_is_one_or_cos_2a_l19_19187


namespace equation_of_line_l19_19042

noncomputable def line_equation_parallel (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (3 * x - 6 * y = 9) ∧ (m = 1/2)

theorem equation_of_line (m : ℝ) (b : ℝ) :
  line_equation_parallel 3 9 →
  (m = 1/2) →
  (∀ (x y : ℝ), (y = m * x + b) ↔ (y - (-1) = m * (x - 2))) →
  b = -2 :=
by
  intros h_eq h_m h_line
  sorry

end equation_of_line_l19_19042


namespace product_x_z_l19_19697

-- Defining the variables x, y, z as positive integers and the given conditions.
theorem product_x_z (x y z : ℕ) (h1 : x = 4 * y) (h2 : z = 2 * x) (h3 : x + y + z = 3 * y ^ 2) : 
    x * z = 5408 / 9 := 
  sorry

end product_x_z_l19_19697


namespace max_y_coordinate_l19_19953

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l19_19953


namespace cost_of_eight_CDs_l19_19374

theorem cost_of_eight_CDs (cost_of_two_CDs : ℕ) (h : cost_of_two_CDs = 36) : 8 * (cost_of_two_CDs / 2) = 144 := by
  sorry

end cost_of_eight_CDs_l19_19374


namespace relationship_x_y_l19_19247

variables {A B C : ℝ}

-- Definitions and conditions
def right_triangle (A B C: ℝ) : Prop :=
  A + B = π / 2 -- Sum of angles in a right triangle where angle C is 90 degrees

def x (A : ℝ) : ℝ := Real.sin A + Real.cos A
def y (B : ℝ) : ℝ := Real.sin B + Real.cos B

theorem relationship_x_y (h : right_triangle A B C) : x A = y B :=
by
  -- Proof goes here (not required for this problem)
  sorry

end relationship_x_y_l19_19247


namespace max_ages_acceptable_within_one_std_dev_l19_19823

theorem max_ages_acceptable_within_one_std_dev
  (average_age : ℤ)
  (std_deviation : ℤ)
  (acceptable_range_lower : ℤ)
  (acceptable_range_upper : ℤ)
  (h1 : average_age = 31)
  (h2 : std_deviation = 5)
  (h3 : acceptable_range_lower = average_age - std_deviation)
  (h4 : acceptable_range_upper = average_age + std_deviation) :
  ∃ n : ℕ, n = acceptable_range_upper - acceptable_range_lower + 1 ∧ n = 11 :=
by
  sorry

end max_ages_acceptable_within_one_std_dev_l19_19823


namespace range_of_a_l19_19004

theorem range_of_a :
  (∀ t : ℝ, 0 < t ∧ t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) →
  (2 / 13 ≤ a ∧ a ≤ 1) :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end range_of_a_l19_19004


namespace two_pt_seven_five_as_fraction_l19_19794

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l19_19794


namespace find_focus_with_larger_x_coordinate_l19_19909

noncomputable def focus_of_hyperbola_with_larger_x_coordinate : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 7
  let b := 9
  let c := Real.sqrt (a^2 + b^2)
  (h + c, k)

theorem find_focus_with_larger_x_coordinate :
  focus_of_hyperbola_with_larger_x_coordinate = (5 + Real.sqrt 130, 20) := by
  sorry

end find_focus_with_larger_x_coordinate_l19_19909


namespace range_of_m_for_line_intersecting_ellipse_l19_19588

theorem range_of_m_for_line_intersecting_ellipse :
  ∀ m : ℝ, 
  (∃ x y : ℝ, y = x + m ∧ (x^2 / 4 + y^2 / 3 = 1)) ↔ m ∈ Icc (-Real.sqrt 7) (Real.sqrt 7) :=
by
  sorry

end range_of_m_for_line_intersecting_ellipse_l19_19588


namespace max_y_coordinate_l19_19948

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l19_19948


namespace tan_add_pi_over_3_l19_19216

theorem tan_add_pi_over_3 (x : ℝ) (h : tan x = 1 / 2) :
  tan (x + π / 3) = 7 + 4 * Real.sqrt 3 :=
by
  sorry

end tan_add_pi_over_3_l19_19216


namespace segments_relationships_l19_19701

variables (A B C D M N P Q R S : Type)
variable [parallelogram A B C D]
variable (MN_AB_parallel : parallel MN AB)
variable (P_on_MN_BD : P ∈ line MN ∩ line BD)
variable (Q_on_AP_CD : Q ∈ line (A, P) ∩ line (C, D))
variable (R_on_Q_parallel_BC : R ∈ line (Q, ↑(parallel_to BC)) ∩ line MN)
variable (S_on_Q_parallel_BC : S ∈ line (Q, ↑(parallel_to BC)) ∩ line BD)

theorem segments_relationships :
  1 / segment_length AB = 1 / segment_length MP - 1 / segment_length MR ∧
  1 / segment_length BC = 1 / segment_length QR - 1 / segment_length QS := 
sorry

end segments_relationships_l19_19701


namespace skew_intersecting_pos_rel_l19_19219

noncomputable def posRelSkewIntersecting (l₁ l₂ l₃ : Type*) [skew_lines l₁ l₂] [parallel_lines l₃ l₁] : Prop :=
  ∃ (relationship : Type), (relationship = Skew ∨ relationship = Intersecting)

theorem skew_intersecting_pos_rel {l₁ l₂ l₃ : Type*} [skew_lines l₁ l₂] [parallel_lines l₃ l₁] :
  posRelSkewIntersecting l₁ l₂ l₃ :=
sorry

end skew_intersecting_pos_rel_l19_19219


namespace tomato_red_flesh_probability_l19_19778

theorem tomato_red_flesh_probability :
  (P_yellow_skin : ℝ) = 3 / 8 →
  (P_red_flesh_given_yellow_skin : ℝ) = 8 / 15 →
  (P_yellow_skin_given_not_red_flesh : ℝ) = 7 / 30 →
  (P_red_flesh : ℝ) = 1 / 4 := 
by
  intros h1 h2 h3
  sorry

end tomato_red_flesh_probability_l19_19778


namespace disrespectful_polynomial_evaluation_at_2_l19_19911

noncomputable def disrespectful_polynomial : ℝ → ℝ :=
λ x, x^2 - (r + s) * x + r * s

theorem disrespectful_polynomial_evaluation_at_2 :
  ∃ r s : ℝ, disrespectful_polynomial (disrespectful_polynomial 2) = 0 → disrespectful_polynomial 2 = 45 / 16 :=
by
  sorry

end disrespectful_polynomial_evaluation_at_2_l19_19911


namespace quadrilateral_is_rectangle_quadrilateral_is_isosceles_trapezoid_l19_19645

-- Definitions for the vectors involved
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)
variables (m n : ℝ)

-- Problem 1: Prove that ABCD is a rectangle
theorem quadrilateral_is_rectangle (h1 : a + b + c + d = 0)
  (h2 : ⟪a, b⟫ = m) (h3 : ⟪b, c⟫ = m) (h4 : ⟪c, d⟫ = m) (h5 : ⟪d, a⟫ = m)
  : (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) → (is_rectangle a b c d) :=
begin
  sorry
end

-- Problem 2: Prove that ABCD is an isosceles trapezoid
theorem quadrilateral_is_isosceles_trapezoid (h1 : a + b + c + d = 0)
  (h2 : ⟪a, b⟫ = m) (h3 : ⟪b, c⟫ = m) (h4 : ⟪c, d⟫ = n) (h5 : ⟪d, a⟫ = n) (h6 : m ≠ n)
  : (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) → (is_isosceles_trapezoid a b c d) :=
begin
  sorry
end

end quadrilateral_is_rectangle_quadrilateral_is_isosceles_trapezoid_l19_19645


namespace max_y_coordinate_l19_19934

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l19_19934


namespace proof_problem_l19_19586

variables {a : ℝ × ℝ} {b : ℝ × ℝ} {k : ℝ}

-- Given conditions
def vec_a_magnitude : Prop := (a.1 ^ 2 + a.2 ^ 2) = 4
def vec_b_value : Prop := b = (-1/2, real.sqrt 3 / 2)
def angle_ab : Prop := b.1 * a.1 + b.2 * a.2 = -1

-- Questions rephrased as proofs
def question1 : Prop := (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 4

def question2 (k : ℝ) : Prop := ∃ k, (a.1 + k * b.1) * (2 * b.1 - a.1) + (a.2 + k * b.2) * (2 * b.2 - a.2) = 0 → k = 2

-- Main theorem combining the conditions and questions
theorem proof_problem :
  vec_a_magnitude → vec_b_value → angle_ab → question1 ∧ question2 :=
begin
  intros h1 h2 h3,
  split,
  { sorry },
  { sorry }
end

end proof_problem_l19_19586


namespace conditional_probability_example_l19_19087

theorem conditional_probability_example
  (P : Set ℕ → ℚ)
  (A B : Set ℕ)
  (hPB : P B = 4/9)
  (hPAB : P (A ∩ B) = 1/9) :
  P (A ∩ B) / P B = 1/4 :=
by
  rw [hPB, hPAB]
  norm_num
  sorry

end conditional_probability_example_l19_19087


namespace alice_alex_probability_same_number_l19_19878

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

/-- Alice and Alex each choosing a positive integer less than 300,
    where Alice's number is a multiple of 20 and Alex's number is a multiple of 36. -/
theorem alice_alex_probability_same_number :
  let N := 300
  let alices_multiples := 20
  let alexs_multiples := 36
  let common_multiples := lcm alices_multiples alexs_multiples
  let number_of_common_multiples := Nat.floor (N / common_multiples)
  let number_of_alices_multiples := Nat.floor (N / alices_multiples)
  let number_of_alexs_multiples := Nat.floor (N / alexs_multiples)
  let total_combinations := number_of_alices_multiples * number_of_alexs_multiples
  (number_of_common_multiples * 1) / total_combinations = 1 / 120 :=
by
  sorry

end alice_alex_probability_same_number_l19_19878


namespace arrangement_valid_count_l19_19430

open Nat

theorem arrangement_valid_count :
  let n := 6
  let total_permutations := fact n
  let invalid_A_left := fact 5
  let invalid_B_right := fact 5
  let invalid_both := fact 4
  total_permutations - invalid_A_left - invalid_B_right + invalid_both = 504 :=
by
  let n := 6
  let total_permutations := fact n
  let invalid_A_left := fact 5
  let invalid_B_right := fact 5
  let invalid_both := fact 4
  calc
    total_permutations - invalid_A_left - invalid_B_right + invalid_both
      = 720 - 120 - 120 + 24 : by rw [fact, fact, fact, fact]
      = 504 : by norm_num

end arrangement_valid_count_l19_19430


namespace funnel_height_is_9_l19_19082

noncomputable def funnel_height
  (r : ℝ) (V : ℝ) : ℝ :=
  (3 * V) / (π * r ^ 2)

theorem funnel_height_is_9 :
  funnel_height 4 150 = 9 := by
  sorry

end funnel_height_is_9_l19_19082


namespace solution_set_of_inequality_l19_19213

variable (a x : ℝ)

theorem solution_set_of_inequality (h : 0 < a ∧ a < 1) :
  (a - x) * (x - (1/a)) > 0 ↔ a < x ∧ x < 1/a :=
sorry

end solution_set_of_inequality_l19_19213


namespace N_subset_M_l19_19604

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x - 2 = 0}

theorem N_subset_M : N ⊆ M := sorry

end N_subset_M_l19_19604


namespace analytical_expression_and_monotonicity_inequality_solution_l19_19994

-- Define the domain condition for f
def domain (x : ℝ) := x ≠ 0

-- Define that f(x) is odd
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Define the function f for x > 0
def f_pos (x : ℝ) := 1 + 1 / x

-- Define the function f for x < 0
def f_neg (x : ℝ) := -1 + 1 / x

-- Combine both cases
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then f_pos x
  else f_neg x

-- Define the intervals of monotonicity
def intervals_of_monotonicity := {I : set ℝ | I = Iio 0 ∨ I = Ioi 0 }

-- Analytical expression and interval monotonicity theorem
theorem analytical_expression_and_monotonicity : 
  (∀ x : ℝ, x > 0 → f(x) = 1 + 1 / x) ∧ 
  (∀ x : ℝ, x < 0 → f(x) = -1 + 1 / x) ∧ 
  (intervals_of_monotonicity = {I : set ℝ | I = Iio 0 ∨ I = Ioi 0 }) := sorry

-- Define the inequality function
def inequality (x : ℝ) := f(2 * x + 1) + 2

-- Solution set of the inequality
def solution_set := { x : ℝ | x ≤ -1 ∨ x > -1/2 }

-- Inequality solution theorem
theorem inequality_solution : ∀ x : ℝ, inequality x ≥ 0 ↔ x ∈ solution_set := sorry

end analytical_expression_and_monotonicity_inequality_solution_l19_19994


namespace arithmetic_sequence_sum_l19_19804

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end arithmetic_sequence_sum_l19_19804


namespace circle_tangent_sum_l19_19504

/-- Given a circle ω with radius 6 centered at O, and a point A outside ω such that OA = 15. 
    Tangents from A to ω intersect the circle at points B and C, and line BC is tangent to ω, 
    with ω outside of triangle ABC. If BC = 10, then AB + AC equals 6 * real.sqrt 21 - 10. -/
theorem circle_tangent_sum {O A B C : Point}
  (ω : Circle) (h_ω_radius : ω.radius = 6) (h_O_center : ω.center = O)
  (h_A_outside : dist O A = 15) (h_tangents : Tangents ω A B C)
  (h_BC_tangent : TangentLine ω B C) (h_BC_length : dist B C = 10) :
  dist A B + dist A C = 6 * real.sqrt 21 - 10 :=
by
  sorry

end circle_tangent_sum_l19_19504


namespace car_dealership_theorem_l19_19838

def car_dealership_problem : Prop :=
  let initial_cars := 100
  let new_shipment := 150
  let initial_silver_percentage := 0.20
  let new_silver_percentage := 0.40
  let initial_silver := initial_silver_percentage * initial_cars
  let new_silver := new_silver_percentage * new_shipment
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_shipment
  let silver_percentage := (total_silver / total_cars) * 100
  silver_percentage = 32

theorem car_dealership_theorem : car_dealership_problem :=
by {
  sorry
}

end car_dealership_theorem_l19_19838


namespace standard_deviation_best_reflects_dispersion_l19_19391

/-- Problem Statement:
Standard deviation is the measure that best reflects the degree of dispersion of a dataset.
-/
theorem standard_deviation_best_reflects_dispersion :
  ∀ (data : list ℝ), ∃ (std_dev : ℝ), std_dev = standard_deviation data :=
by
  sorry

end standard_deviation_best_reflects_dispersion_l19_19391


namespace additivity_of_f_l19_19271

noncomputable def f : ℝ → ℝ := sorry

theorem additivity_of_f (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f(x + y + x * y) = f(x) + f(y) + f(x * y)) : 
  ∀ x y : ℝ, f(x + y) = f(x) + f(y) :=
sorry

end additivity_of_f_l19_19271


namespace max_y_on_graph_l19_19930

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l19_19930


namespace smallest_positive_solutions_sum_l19_19958

noncomputable def floor_function (x : ℝ) : ℤ := int.floor x

noncomputable def equation (x : ℝ) : ℝ := x - int.floor x - (2 : ℝ) / int.floor x

theorem smallest_positive_solutions_sum :
  let sol1 := 3
  let sol2 := 3 + 2 / 3
  sol1 + sol2 = 6 + 2 / 3 :=
by
  let sol1 := (3 : ℝ)
  let sol2 := (3 + 2 / 3 : ℝ)
  calc
    sol1 + sol2 = (3 : ℝ) + (3 + 2 / 3) : rfl
           ...  = 6 + 2 / 3 : by ring

#check smallest_positive_solutions_sum

end smallest_positive_solutions_sum_l19_19958


namespace max_y_coordinate_l19_19935

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l19_19935


namespace necessary_but_not_sufficient_l19_19418

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + a < 0) → (a < 11) ∧ ¬((a < 11) → (∃ x : ℝ, x^2 - 2*x + a < 0)) :=
by
  -- Sorry to bypass proof below, which is correct as per the problem statement requirements.
  sorry

end necessary_but_not_sufficient_l19_19418


namespace simplify_exponent_expression_l19_19419

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l19_19419


namespace complex_modulus_circle_l19_19633

noncomputable def is_circle (z : ℂ) : Prop :=
  |z + 1 - 3 * complex.I| = 2

theorem complex_modulus_circle :
  ∀ (z : ℂ), is_circle z → ∃ (c : ℂ) (r : ℝ), c = -1 + 3 * complex.I ∧ r = 2 :=
by
  intro z h
  use [-1 + 3 * complex.I, 2]
  split
  sorry
  sorry

end complex_modulus_circle_l19_19633


namespace application_methods_l19_19423

theorem application_methods (n m : ℕ) (h_n : n = 5) (h_m : m = 3) : m ^ n = 3 ^ 5 := by
  rw [h_n, h_m]
  rfl

end application_methods_l19_19423


namespace exchange_5_rubles_l19_19503

theorem exchange_5_rubles :
  ¬ ∃ n : ℕ, 1 * n + 2 * n + 3 * n + 5 * n = 500 :=
by 
  sorry

end exchange_5_rubles_l19_19503


namespace minimize_max_modulus_on_interval_l19_19955

noncomputable def minimal_max_modulus_polynomial (a b : ℝ) :=
  x^2 + a * x + b

theorem minimize_max_modulus_on_interval :
  ∃ (a b : ℝ), (minimal_max_modulus_polynomial a b = λ x, x^2 - 1 / 2) ∧
  (∀ (P : ℝ → ℝ), (P = minimal_max_modulus_polynomial a b) →
  ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), |P x| ≤ 1 / 2) :=
begin
  sorry
end

end minimize_max_modulus_on_interval_l19_19955


namespace tangent_line_slope_cannot_be_neg_two_tangent_line_slope_can_be_three_one_tangent_line_through_point_two_tangent_lines_through_point_l19_19596

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_slope_cannot_be_neg_two :
  ¬(∃ x : ℝ, deriv f x = -2) :=
sorry

theorem tangent_line_slope_can_be_three :
  ∃ x : ℝ, deriv f x = 3 :=
sorry

theorem one_tangent_line_through_point :
  ∃! x : ℝ, (λ t : ℝ, f t + (t - x) * deriv f x = 2) 0 :=
sorry

theorem two_tangent_lines_through_point :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (λ t : ℝ, f t + (t - x1) * deriv f x1 = 4) 1 ∧ (λ t : ℝ, f t + (t - x2) * deriv f x2 = 4) 1 :=
sorry

end tangent_line_slope_cannot_be_neg_two_tangent_line_slope_can_be_three_one_tangent_line_through_point_two_tangent_lines_through_point_l19_19596


namespace max_f_value_g_decreasing_intervals_l19_19598
noncomputable theory

-- Define the given functions and conditions
def f (ω x : ℝ) : ℝ := 2 * sin(ω * x + π / 4) ^ 2 + 2 * cos(ω * x) ^ 2 

-- Assume the distance between two adjacent lowest points on its graph
axiom distance_between_lowest_points (ω : ℝ) (hω : ω > 0) : 2 * π / (3 * ω)

-- Define the maximum value of the function
def max_f (ω : ℝ) := 2 + sqrt 2

-- Prove the maximum value of f(x) given ω = 3/2
theorem max_f_value (x : ℝ) (k : ℤ) (ω : ℝ) (hω : ω > 0) :
  distance_between_lowest_points ω hω = π / 3 →
  f (3/2) x = 2 + sqrt 2 * sin(3 * x + π / 4) →
  max_f ω = 2 + sqrt 2 :=
by
  sorry

-- Define the transformed function g
def g (x : ℝ) : ℝ := 2 + sqrt 2 * sin(-3 * x - π / 8)

-- Prove the interval(s) where g(x) is decreasing
theorem g_decreasing_intervals (x : ℝ) (k : ℤ) :
  ∀ k : ℤ, 
  (2/3 : ℝ) * k * π - (5/24 : ℝ) * π ≤ x ∧ 
  x ≤ (2/3 : ℝ) * k * π + (π / 8 : ℝ) →
  deriv (λ x, g x) x < 0 :=
by
  sorry

end max_f_value_g_decreasing_intervals_l19_19598


namespace number_of_shoes_outside_library_l19_19450

-- Define the conditions
def number_of_people : ℕ := 10
def shoes_per_person : ℕ := 2

-- Define the proof that the number of shoes kept outside the library is 20.
theorem number_of_shoes_outside_library : number_of_people * shoes_per_person = 20 :=
by
  -- Proof left as sorry because the proof steps are not required
  sorry

end number_of_shoes_outside_library_l19_19450


namespace decimal_to_fraction_l19_19791

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l19_19791


namespace problem_proof_l19_19990

def p : Prop := ∀ x : ℝ, 2^x > 0
def q : Prop := ∃ x : ℝ, sin x + cos x > sqrt 2

theorem problem_proof : p ∧ ¬q :=
by {
  sorry
}

end problem_proof_l19_19990


namespace number_of_four_digit_integers_l19_19615

theorem number_of_four_digit_integers :  
  {n : ℕ // n % 7 = 1 ∧ n % 10 = 3 ∧ n % 13 = 6 ∧ 1000 ≤ n ∧ n < 10000}.card = 10 :=
  by sorry

end number_of_four_digit_integers_l19_19615


namespace polynomial_roots_l19_19512

theorem polynomial_roots :
  ∀ x : ℝ, (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
by 
  intro x
  split
  { intro h
    have h1 : (x - 2) * (x + 2) = x^2 - 4 := by ring,
    rw h1 at h,
    cases eq_zero_or_eq_zero_of_mul_eq_zero h
    { left
      assumption
    }
    { right
      assumption }
  }
  { intro h,
    cases h
    { rw h
      ring }
    { rw h
      ring }
  }

end polynomial_roots_l19_19512


namespace necessary_but_not_sufficient_l19_19753

-- Define the condition within the proof problem
def elliptical_condition (k : ℝ) : Prop :=
  4 < k ∧ k < 6

-- Define the equation being considered
def represents_ellipse (k : ℝ) : Prop :=
  6 - k > 0 ∧ k - 4 > 0 ∧ 6 - k ≠ k - 4

-- Main theorem
theorem necessary_but_not_sufficient (k : ℝ) :
  elliptical_condition k ↔ represents_ellipse k :=
begin
  sorry
end

end necessary_but_not_sufficient_l19_19753


namespace emma_final_balance_correct_l19_19536

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l19_19536


namespace calculation_equivalence_l19_19501

theorem calculation_equivalence : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := 
by
  sorry

end calculation_equivalence_l19_19501


namespace not_perfect_square_T_l19_19558

noncomputable def operation (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom associative {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  operation x (operation y z) = operation (operation x y) z

noncomputable def T (n : ℕ) : ℝ :=
  if h : n ≥ 4 then
    (List.range (n - 2)).foldr (λ x acc => operation (x + 3) acc) 3
  else 0

theorem not_perfect_square_T (n : ℕ) (h : n ≥ 4) :
  ¬ (∃ k : ℕ, (96 / (T n - 2) : ℝ) = k ^ 2) :=
sorry

end not_perfect_square_T_l19_19558


namespace least_possible_perimeter_l19_19233

noncomputable def cos_a : ℝ := 3 / 5
noncomputable def cos_b : ℝ := 1 / 3
noncomputable def cos_c : ℝ := -1 / 5

theorem least_possible_perimeter 
  (A B C : Type) 
  [linear_ordered_field A]
  [linear_ordered_field B]
  [linear_ordered_field C]
  (a b c : A) 
  (ha : cos_a = 3 / 5) 
  (hb : cos_b = 1 / 3) 
  (hc : cos_c = -1 / 5)
  (triangle_abc : a * a + b * b - 2 * a * b * cos_a = 0 
                ∧ b * b + c * c - 2 * b * c * cos_b = 0 
                ∧ c * c + a * a - 2 * c * a * cos_c = 0) 
  (ha' : a ≠ 0) 
  (hb' : b ≠ 0) 
  (hc' : c ≠ 0) 
  (h1 : ∀ x y z : ℝ, x + y + z = a + b + c) :
  a + b + c = 98 :=
  sorry

end least_possible_perimeter_l19_19233


namespace part1_part2_l19_19173

-- Conditions
def S_n (n : ℕ) : ℕ := n^2 + n
def a_n (n : ℕ) : ℕ := if n = 1 then 2 else S_n n - S_n (n - 1)
def b_n (n : ℕ) : ℝ := Real.sqrt (2^a_n n)
def c_n (n : ℕ) : ℝ := a_n n * b_n n
def T_n (n : ℕ) : ℝ := (n-1) * 2^(n+2) + 4

-- Proof statement for part 1
theorem part1 (n : ℕ) (h : 0 < n) : b_n n = 2^n := sorry

-- Proof statement for part 2
theorem part2 (n : ℕ) (h : 0 < n) : ∑ k in Finset.range n, c_n (k + 1) = (n-1) * 2^(n+2) + 4 := sorry

end part1_part2_l19_19173


namespace find_ratio_of_AE_AB_l19_19237

theorem find_ratio_of_AE_AB (ABCD : Type) [square : square ABCD]
  (E : Point) (A B : Point) (angle_EAB : ∠EAB = 30 : Real)
  (area_square : ℝ) (area_triangle : ℝ)
  (h : ℝ) (a : ℝ) :
  (AB = 1) →
  (AD = 1) →
  (area_square = 1) →
  (area_triangle = (h / 2)) →
  (area_square = 6 * area_triangle) →
  (sin (30) = h / a) →
  AE / AB = 2 / 3 :=
by
  sorry

end find_ratio_of_AE_AB_l19_19237


namespace cone_lateral_area_l19_19590

noncomputable def lateral_area (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r ^ 2 + h ^ 2)
  in π * r * l

theorem cone_lateral_area : lateral_area 1 (Real.sqrt 3) = 2 * π :=
by
  sorry

end cone_lateral_area_l19_19590


namespace car_mpg_difference_l19_19081

variable (T mpg_highway mpg_city : ℝ)

-- conditions
def condition1 : Prop := mpg_highway = 560 / T
def condition2 : Prop := T = 336 / 9
def condition3 : Prop := mpg_city = 9

-- target
def target_statement : Prop := mpg_highway - mpg_city = 6

-- Theorem statement
theorem car_mpg_difference (h1 : condition1 T mpg_highway) 
                           (h2 : condition2 T) 
                           (h3 : condition3 mpg_city) : 
                           target_statement T mpg_highway mpg_city :=
sorry

end car_mpg_difference_l19_19081


namespace machines_needed_l19_19221

theorem machines_needed 
    (machines : ℕ) 
    (cell_phones : ℕ) 
    (cell_phones_per_minute : ℕ) 
    (rate : cell_phones_per_minute = cell_phones / machines) 
    (desired_cell_phones : ℕ) 
    (desired_cell_phones = 50)
    (initial_machines = 2)
    (initial_cell_phones = 10)
    (initial_rate : cell_phones_per_minute = 10 / 2)
    (required_machines : ℕ) : 
    required_machines = 10 := sorry

end machines_needed_l19_19221


namespace number_of_female_students_is_30_l19_19643

variable (total_students : ℕ) (female_students : ℕ)
variable (equal_chance : total_students = 63)
variable (prob_condition : (female_students / total_students.to_rat) = 10 / 11 * ((total_students - female_students) / total_students.to_rat))

theorem number_of_female_students_is_30 :
  female_students = 30 :=
by
  sorry

end number_of_female_students_is_30_l19_19643


namespace coloring_2n_gon_l19_19617

theorem coloring_2n_gon (n : ℕ) :
  let colors := {R, G, B},
      vertices := Finset.range (2 * n) in
  let valid_coloring_count := 3^n + (-2)^(n+1) - 1 in
  ∀ (coloring : vertices → colors),
  -- Condition: No two adjacent vertices share the same color
  (∀ i : Finset.range 2*n, coloring i ≠ coloring ((i + 1) % (2*n))) →
  -- Condition: No two vertices directly across from each other share the same color
  (∀ i : Finset.range n, coloring i ≠ coloring (i + n)) →
  -- Correct answer: number of valid colorings
  valid_coloring_count = 3^n + (-2)^(n+1) - 1 :=
by sorry

end coloring_2n_gon_l19_19617


namespace diagonals_in_eight_sided_polygon_l19_19448

-- Definitions based on the conditions
def n := 8  -- Number of sides
def right_angles := 2  -- Number of right angles

-- Calculating the number of diagonals using the formula
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Lean statement for the problem
theorem diagonals_in_eight_sided_polygon : num_diagonals n = 20 :=
by
  -- Substitute n = 8 into the formula and simplify
  sorry

end diagonals_in_eight_sided_polygon_l19_19448


namespace player_A_winning_strategy_l19_19069

theorem player_A_winning_strategy :
  ∃ strategy : (fin 5 → ℕ) × (list ℕ → list ℕ → bool),
    strategy.1 = [1, 2^1995, 2^1996, 2^1997, 2^1998] ∧
    ∀ (remaining : list ℕ), 
      ∀ (turn : ℕ), 
        turn % 2 = 0 → -- indicating it's A's turn
          ∀ (b_choices : fin 5 → ℕ),
            all (λ x, x ≥ 0) b_choices → -- ensuring B's choices don't introduce negative
              let a_choices := strategy.2 remaining b_choices in
              all (λ x, x ≥ 0) a_choices ∧
              -- defining remaining set after choices made
              let new_remaining := (remaining.diff b_choices.toList).diff a_choices.toList in
              (turn < 2000) → 
                (new_remaining.all (λ x, x - 1 ≥ 0)) -- remaining elements allowing game continuation
:= 
sorry

end player_A_winning_strategy_l19_19069


namespace find_m_l19_19353

theorem find_m (m : ℝ) : (m + 2) * (m - 2) + 3 * m * (m + 2) = 0 ↔ m = 1/2 ∨ m = -2 :=
by
  sorry

end find_m_l19_19353


namespace geometry_problem_l19_19841

theorem geometry_problem 
  (A B C D E F M : Type) 
  (h1 : circle_passing_through A B (triangle A B C))
  (h2 : circle_intersecting_segments D E A B C)
  (h3 : lines_intersect_at F (line_through B A) (line_through E D))
  (h4 : lines_intersect_at M (line_through B D) (line_through C F)) :
  (MF = MC ↔ MB * MD = MC^2) := 
sorry

end geometry_problem_l19_19841


namespace prop1_prop2_prop3_prop4_l19_19023

-- Define what it means for a function to be odd.
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

-- Define what it means for a function to be even.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of range being zero.
def range_is_zero (f : ℝ → ℝ) : Prop := ∀ y, y ∈ set.range f → y = 0

-- Proposition 1: If f is both odd and even, its range is {0}.
theorem prop1 (f : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even f) : range_is_zero f :=
sorry

-- Proposition 2: If f is even, then f(|x|) = f(x).
theorem prop2 (f : ℝ → ℝ) (h_even : is_even f) : ∀ x, f (|x|) = f x :=
sorry

-- Proposition 3: It's not necessarily true that a non-monotonic function cannot have an inverse.
theorem prop3 : ¬ ∀ f : ℝ → ℝ, (¬ strict_mono_incr_on f set.univ ∨ ¬ strict_mono_decr_on f set.univ) → ¬ function.has_left_inverse f :=
sorry

-- Proposition 4: It's not necessarily true that intersection points of f and f⁻¹ must lie on y = x if f⁻¹ is non-identical.
theorem prop4 {f g : ℝ → ℝ} (h_inv : function.left_inverse g f) :
  ¬ (∀ x, (f x = g x ∨ (∃ y, f y = x ∧ g y = x) ∧ x ≠ g x) → y = x) :=
sorry

end prop1_prop2_prop3_prop4_l19_19023


namespace sqrt_patterns_and_sequence_sum_l19_19303

theorem sqrt_patterns_and_sequence_sum :
  (sqrt (5 * 7 + 1) = 6) ∧ 
  (sqrt (26 * 28 + 1) = 27) ∧
  (∀ n : ℕ, n ≥ 1 → sqrt (n * (n + 2) + 1) = n + 1) ∧
  (∀ n : ℕ, n ≥ 2 → sqrt ((n - 1) * (n + 1) + 1) = n) ∧
  (∑ i in finset.range 1011, (-1 : ℤ) ^ (i + 1) * sqrt ((2 * i + 1) * (2 * i + 3) + 1) = -1010) :=
by sorry

end sqrt_patterns_and_sequence_sum_l19_19303


namespace find_point_P_distance_of_P_to_l3_line_passing_through_P_parallel_to_l4_line_passing_through_P_perpendicular_to_l4_l19_19579

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def l3 (x y : ℝ) := 4 * x - 3 * y - 6 = 0
def l4 (x y : ℝ) := 3 * x - y + 1 = 0

def point_P := (-2 : ℝ, 2 : ℝ)

theorem find_point_P:
  l1 point_P.1 point_P.2 ∧ l2 point_P.1 point_P.2 :=
sorry

theorem distance_of_P_to_l3:
  let P := point_P in
  let x1 := P.1 in
  let y1 := P.2 in
  let d := abs (4 * x1 - 3 * y1 - 6) / real.sqrt (4 ^ 2 + (-3) ^ 2) in
  d = 4 :=
sorry
  
theorem line_passing_through_P_parallel_to_l4:
  let slope := 3 in
  let (x1, y1) := point_P in
  3 * x - y + 8 = 0 :=
sorry
  
theorem line_passing_through_P_perpendicular_to_l4:
  let slope_perp := -1/3 in
  let (x1, y1) := point_P in
  x + 3 * y - 4 = 0 :=
sorry

end find_point_P_distance_of_P_to_l3_line_passing_through_P_parallel_to_l4_line_passing_through_P_perpendicular_to_l4_l19_19579


namespace seating_arrangements_chairs_3_people_next_to_each_other_l19_19021

def total_seating_arrangements : ℕ := 12

theorem seating_arrangements_chairs_3_people_next_to_each_other :
  ∀ (P : Type) [fintype P], (fintype.card P = 3) →
  ∀ (C : Type) [fintype C], (fintype.card C = 5) →
  (∀ A B : P, ∃ (adjacent : finset (C × C)), (∀ (c1 c2 : C), (c1, c2) ∈ adjacent → c1 = c2 + 1 ∨ c2 = c1 + 1) ∧
   (∀ C1 C2 C3 : C, C1 ≠ C2 ∧ C2 ≠ C3 ∧ C1 ≠ C3) ∧
   ∃ count : ℕ, count = total_seating_arrangements) :=
begin
  sorry
end

end seating_arrangements_chairs_3_people_next_to_each_other_l19_19021


namespace solution_l19_19555

theorem solution (y : ℚ) : 
  16 ^ (-3 : ℚ) = 4 ^ (60 / y) / (4 ^ (32 / y) * 16 ^ (24 / y)) → 
  y = 10 / 3 :=
by
  intro h
  sorry

end solution_l19_19555


namespace bally_subset_count_l19_19093

-- Define what it means for a set to be Bally
def is_bally_set (S : Set ℕ) : Prop :=
  ∀ m ∈ S, (S.filter (< m)).card < m / 2

-- The explicit set we are considering
def big_set : Set ℕ := {i | 1 ≤ i ∧ i ≤ 2020}

-- The main theorem stating the number of Bally subsets
theorem bally_subset_count :
  {T : Set ℕ | T ⊆ big_set ∧ is_bally_set T}.card = binom 2021 1010 - 1 :=
by
  sorry

end bally_subset_count_l19_19093


namespace value_of_t_l19_19226

theorem value_of_t (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 122) : t = 50 :=
by
  have : t = 5 / 9 * (122 - 32), from (congr_arg (λ x, 5 / 9 * (x - 32)) h2),
  rw [h1, this],
  sorry

end value_of_t_l19_19226


namespace moles_of_ammonium_nitrate_formed_l19_19547

def ammonia := ℝ
def nitric_acid := ℝ
def ammonium_nitrate := ℝ

-- Define the stoichiometric coefficients from the balanced equation.
def stoichiometric_ratio_ammonia : ℝ := 1
def stoichiometric_ratio_nitric_acid : ℝ := 1
def stoichiometric_ratio_ammonium_nitrate : ℝ := 1

-- Define the initial moles of reactants.
def initial_moles_ammonia (moles : ℝ) : Prop := moles = 3
def initial_moles_nitric_acid (moles : ℝ) : Prop := moles = 3

-- The reaction goes to completion as all reactants are used:
theorem moles_of_ammonium_nitrate_formed :
  ∀ (moles_ammonia moles_nitric_acid : ℝ),
    initial_moles_ammonia moles_ammonia →
    initial_moles_nitric_acid moles_nitric_acid →
    (moles_ammonia / stoichiometric_ratio_ammonia) = 
    (moles_nitric_acid / stoichiometric_ratio_nitric_acid) →
    (moles_ammonia / stoichiometric_ratio_ammonia) * stoichiometric_ratio_ammonium_nitrate = 3 :=
by
  intros moles_ammonia moles_nitric_acid h_ammonia h_nitric_acid h_ratio
  rw [h_ammonia, h_nitric_acid] at *
  simp only [stoichiometric_ratio_ammonia, stoichiometric_ratio_nitric_acid, stoichiometric_ratio_ammonium_nitrate] at *
  sorry

end moles_of_ammonium_nitrate_formed_l19_19547


namespace Euclid_Middle_School_AMC8_contest_l19_19112

theorem Euclid_Middle_School_AMC8_contest (students_Germain students_Newton students_Young : ℕ)
       (hG : students_Germain = 11) 
       (hN : students_Newton = 8) 
       (hY : students_Young = 9) : 
       students_Germain + students_Newton + students_Young = 28 :=
by
  sorry

end Euclid_Middle_School_AMC8_contest_l19_19112


namespace simplify_and_evaluate_expr_l19_19324

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l19_19324


namespace wages_sum_days_l19_19868

/-- Given conditions: S = 24P and S = 40Q, 
    prove that the sum of money S is sufficient
    to pay the wages of both p and q together for 15 days.
-/
theorem wages_sum_days (S P Q : ℝ) (h1 : S = 24 * P) (h2 : S = 40 * Q) : 
  ∃ D : ℕ, D = 15 :=
by
  use 15
  sorry

end wages_sum_days_l19_19868


namespace general_term_formula_sum_first_n_terms_l19_19591

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, a n = 3^(n-2) := 
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, S n = (3^(n-2)) / 2 - 1 / 18 := 
by
  sorry

end general_term_formula_sum_first_n_terms_l19_19591


namespace problem_l19_19818

theorem problem (x : ℝ) (h : x + 1 / x = 5) : x ^ 2 + (1 / x) ^ 2 = 23 := 
sorry

end problem_l19_19818


namespace group_c_questions_l19_19819

theorem group_c_questions:
  (B = 23) →
  (A + B + C = 100) →
  (A ≥ (0.6 * (A + 2 * B + 3 * C))) →
  ∃ C : ℕ, C = 1 := 
by 
  intros B_value eq1 ge_condition
  have hc1 : A + C = 77 := 
    by sorry
  have hc2 : A ≥ 69 + 4.5 * C := 
    by sorry
  use 1
  --We need further reasoning that C = 1 is the only solution given all the conditions.
  --This will involve combining hc1 and hc2 to arrive at a conclusion.
  sorry

end group_c_questions_l19_19819


namespace exists_quadratic_polynomials_Q_l19_19275

noncomputable def P : Polynomial ℝ := (X - 4) * (X - 5) * (X - 6)

theorem exists_quadratic_polynomials_Q :
  ∃ (Q : Polynomial ℝ), degree Q = 2 ∧
    ∃ (R : Polynomial ℝ), degree R = 3 ∧ R.eval 1 = 3 ∧
    (P.comp Q) = (P * R) :=
sorry

end exists_quadratic_polynomials_Q_l19_19275


namespace arrangement_count_l19_19156

theorem arrangement_count :
  ∃! (arrangements : ℕ), arrangements = 216 ∧ 
  (∀ grid : Array (Array (Option Char)), 
  -- Ensuring grid is a 4x4 grid
  grid.size = 4 ∧ (∀ row, row ∈ grid -> row.size = 4) ∧ 
  -- Each row has distinct A, B, C, D
  (∀ row : Array (Option Char), row ∈ grid → (∃ perm : List Char, row.to_list = perm ∧ perm.nodup ∧
    perm.perm [some 'A', some 'B', some 'C', some 'D'])) ∧
  -- Each column has distinct A, B, C, D
  (∀ col_idx : Fin 4, (∃ perm : List Char, (Array.map (λ row, row[col_idx]) grid).to_list = perm ∧ perm.nodup ∧
    perm.perm [some 'A', some 'B', some 'C', some 'D'])) ∧ 
  -- A is placed in the upper right corner
  grid[0][3] = some 'A') :=
begin
  -- number of arrangements
  -- proves there are exactly 216 ways
  use 216,
  split,
  -- show arrangements = 216 by solving the math problem
  sorry,
  -- show that the grid follows all given conditions
  sorry,
end

end arrangement_count_l19_19156


namespace subset_weight_range_l19_19175

theorem subset_weight_range (n : ℕ) (A : finset ℝ) (hA_card : A.card = n) (hA_weight : ∀ a ∈ A, 1 ≤ a) (hA_sum : A.sum id = 2 * n) (r : ℝ) (h_r : 0 ≤ r ∧ r ≤ 2 * n - 2) : 
  ∃ B ⊆ A, r ≤ B.sum id ∧ B.sum id ≤ r + 2 :=
by 
  sorry

end subset_weight_range_l19_19175


namespace zhou_yu_age_at_death_l19_19311

theorem zhou_yu_age_at_death (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 9)
    (h₂ : ∃ age : ℕ, age = 10 * (x - 3) + x)
    (h₃ : x^2 = 10 * (x - 3) + x) :
    x^2 = 10 * (x - 3) + x :=
by
  sorry

end zhou_yu_age_at_death_l19_19311


namespace shoe_company_current_monthly_earnings_l19_19864

variables (X : ℕ) (annual_goal monthly_additional : ℕ)

theorem shoe_company_current_monthly_earnings
    (annual_goal_eq : annual_goal = 60000)
    (monthly_additional_eq : monthly_additional = 1000)
    (monthly_goal : ℕ := annual_goal / 12) :
  X = monthly_goal - monthly_additional :=
begin
  sorry
end

end shoe_company_current_monthly_earnings_l19_19864


namespace speed_in_still_water_l19_19406

-- Defining the terms as given conditions in the problem
def speed_downstream (v_m v_s : ℝ) : ℝ := v_m + v_s
def speed_upstream (v_m v_s : ℝ) : ℝ := v_m - v_s

-- Given conditions translated into Lean definitions
def downstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_downstream v_m v_s = 7

def upstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_upstream v_m v_s = 4

-- The problem statement to prove
theorem speed_in_still_water : 
  downstream_condition ∧ upstream_condition → ∃ v_m : ℝ, v_m = 5.5 :=
by 
  intros
  sorry

end speed_in_still_water_l19_19406


namespace valid_triples_l19_19130

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1) ↔ (x, y, z) = (1, 1, 1) ∨ 
                                                      (x, y, z) = (1, 1, 2) ∨ 
                                                      (x, y, z) = (1, 3, 2) ∨ 
                                                      (x, y, z) = (3, 5, 4) :=
by
  sorry

end valid_triples_l19_19130


namespace simplify_and_evaluate_expr_l19_19323

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end simplify_and_evaluate_expr_l19_19323


namespace fractional_inequality_solution_l19_19768

theorem fractional_inequality_solution :
  {x : ℝ | (2 * x - 1) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 1 / 2} := 
by
  sorry

end fractional_inequality_solution_l19_19768


namespace work_rate_B_l19_19080

theorem work_rate_B :
  (∀ A B : ℝ, A = 30 → (1 / A + 1 / B = 1 / 19.411764705882355) → B = 55) := by 
    intro A B A_cond combined_rate
    have hA : A = 30 := A_cond
    rw [hA] at combined_rate
    sorry

end work_rate_B_l19_19080


namespace monotonic_intervals_a_quarter_l19_19595

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2 - x

theorem monotonic_intervals_a_quarter :
  let a := (1 / 4 : ℝ)
  ∀ x : ℝ, (-1 < x ∧ x < 0) ∨ (1 < x) → (1 < x) ∨ (x < 0 ∧ x < 0) ∨ (0 < x ∧ x < 1) → 
    (Real.log (x + 1) + (1 / 4 : ℝ) * x ^ 2 - x = 0 → 
      (∀ y : ℝ, f_prime y > 0 ∧ (x != y) → forall x y))
  sorry

end monotonic_intervals_a_quarter_l19_19595


namespace angle_B_is_pi_over_3_range_of_expression_l19_19235

variable {A B C a b c : ℝ}

-- Conditions
def sides_opposite_angles (A B C : ℝ) (a b c : ℝ): Prop :=
  (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Part 1: Prove B = π/3
theorem angle_B_is_pi_over_3 (h : sides_opposite_angles A B C a b c) : 
    B = Real.pi / 3 := 
  sorry

-- Part 2: Prove the range of sqrt(3) * (sin A + sin(C - π/6)) is (1, 2]
theorem range_of_expression (h : 0 < A ∧ A < 2 * Real.pi / 3) : 
    (1:ℝ) < Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) 
    ∧ Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) ≤ 2 := 
  sorry

end angle_B_is_pi_over_3_range_of_expression_l19_19235


namespace difference_before_flipping_l19_19366

-- Definitions based on the conditions:
variables (Y G : ℕ) -- Number of yellow and green papers

-- Condition: flipping 16 yellow papers changes counts as described
def papers_after_flipping (Y G : ℕ) : Prop :=
  Y - 16 = G + 16

-- Condition: after flipping, there are 64 more green papers than yellow papers.
def green_more_than_yellow_after_flipping (G Y : ℕ) : Prop :=
  G + 16 = (Y - 16) + 64

-- Statement: Prove the difference in the number of green and yellow papers before flipping was 32.
theorem difference_before_flipping (Y G : ℕ) (h1 : papers_after_flipping Y G) (h2 : green_more_than_yellow_after_flipping G Y) :
  (Y - G) = 32 :=
by
  sorry

end difference_before_flipping_l19_19366


namespace JackEmails_l19_19260

theorem JackEmails (E : ℕ) (h1 : 10 = E + 7) : E = 3 :=
by
  sorry

end JackEmails_l19_19260


namespace simplify_exponent_expression_l19_19420

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l19_19420


namespace zeros_after_decimal_in_1_div_40_pow_10_l19_19343

theorem zeros_after_decimal_in_1_div_40_pow_10 : 
  let n := 10 in
  let x := 40^n in 
  let y := 1/x in 
  (string.length (string.take_while (λ c => c = '0') (string.drop 2 (to_string y)))) = 16 := 
by 
  sorry

end zeros_after_decimal_in_1_div_40_pow_10_l19_19343


namespace hyperbola_asymptotes_and_eccentricity_l19_19347

theorem hyperbola_asymptotes_and_eccentricity :
  (∀ x y : ℝ, x^2 - y^2 / 2 = 1 → y = sqrt 2 * x ∨ y = -sqrt 2 * x) ∧
  (∃ e : ℝ, e = sqrt 3 ∧ ∀ x y : ℝ, x^2 - y^2 / 2 = 1 → e = sqrt (1 + (2 / 1))) :=
by
  sorry

end hyperbola_asymptotes_and_eccentricity_l19_19347


namespace first_term_of_geometric_series_l19_19016

theorem first_term_of_geometric_series (a r : ℝ) 
    (h1 : a / (1 - r) = 18) 
    (h2 : a^2 / (1 - r^2) = 72) : 
    a = 72 / 11 := 
  sorry

end first_term_of_geometric_series_l19_19016


namespace co_president_probability_l19_19364

-- Definition of students and clubs
def club_size (c : ℕ) : ℕ :=
  if c = 1 then 5 else if c = 2 then 7 else 8

def club_prob (c n : ℕ) : ℚ :=
  if c = 1 then 3 / 10 else if c = 2 then 1 / 7 else 3 / 28

-- Probability that the selected members include two co-presidents
def total_prob : ℚ :=
  (1 / 3) * (3 / 10 + 1 / 7 + 3 / 28)

-- Theorem that states the probability condition
theorem co_president_probability :
  total_prob = 11 / 60 :=
begin
  sorry
end

end co_president_probability_l19_19364


namespace original_cost_air_conditioning_l19_19485

variable {P : ℝ} -- original cost of the air-conditioning unit

theorem original_cost_air_conditioning 
  (h_discount : 0.84 * P = P - 0.16 * P)
  (h_increase : 0.9408 * P = (0.84 * P) + 0.12 * (0.84 * P))
  (h_final_price : 0.9408 * P = 442.18) :
  P ≈ 469.99 := by
  sorry

end original_cost_air_conditioning_l19_19485


namespace probability_eq_no_real_roots_l19_19034

noncomputable
def probability_no_real_roots : ℚ := 17 / 36

theorem probability_eq_no_real_roots :
  let outcomes := (finset.product (finset.range 1 7) (finset.range 1 7)) in
  let pairs := finset.filter (λ (x : ℕ × ℕ), (x.1 * x.1 < 4 * x.2)) outcomes in
  (pairs.card = 17 → real.to_rat (pairs.card / outcomes.card) = probability_no_real_roots) :=
by
  sorry

end probability_eq_no_real_roots_l19_19034


namespace determine_symmetric_circle_equation_l19_19999

-- Definition of the initial circle's equation
def initial_circle_equation : Prop := ∀ x y : ℝ, x^2 + y^2 + 2x = 0

-- Definition of the line about which to reflect
def symmetry_line_equation : Prop := ∀ x y : ℝ, x + y - 1 = 0

-- Definition of the equation of the resulting symmetric circle
def symmetric_circle_equation (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = r^2

-- The radius and center of the initial circle
def initial_circle_center : ℝ × ℝ := (-1, 0)
def initial_circle_radius : ℝ := 1

-- The center of the symmetric circle obtained by symmetry
def symmetric_circle_center : ℝ × ℝ := (1, 2)

-- Lean Theorem to prove the equation of the symmetric circle
theorem determine_symmetric_circle_equation : 
  initial_circle_equation →
  symmetry_line_equation →
  symmetric_circle_center = (1, 2) →
  symmetric_circle_equation 1 2 1 :=
by
  intros
  sorry

end determine_symmetric_circle_equation_l19_19999


namespace inverse_function_value_l19_19165

-- Define the function f
def f (x : ℝ) := x^2 + 2

-- Define the inverse function f_inv
noncomputable def f_inv (y : ℝ) : ℝ := 
  if h : ∃ x : ℝ, x^2 + 2 = y ∧ x ≤ 0 
  then classical.some h 
  else 0

-- The inverse function condition
axiom f_inv_correct {x : ℝ} (hx : x^2 + 2 = 3) (hx_neg : x ≤ 0) : f_inv 3 = x

-- Theorem statement
theorem inverse_function_value :
  f_inv 3 = -1 := 
sorry

end inverse_function_value_l19_19165


namespace probability_point_between_X_and_Z_l19_19717

theorem probability_point_between_X_and_Z (XW XZ YW : ℝ) (h1 : XW = 4 * XZ) (h2 : XW = 8 * YW) :
  (XZ / XW) = 1 / 4 := by
  sorry

end probability_point_between_X_and_Z_l19_19717


namespace area_quadrilateral_APBC_l19_19370

-- Define the points and distances as given in the problem
variables {A P Q B C : Type} 
variables [metric_space : metric_space P]

-- Define the distances according to the problem statement
def AP : ℝ := 15
def PB : ℝ := 20
def PC : ℝ := 25

-- Define the condition of right triangles
def is_right_triangle {X Y Z : P} (XY : ℝ) (YZ : ℝ) (XZ : ℝ) : Prop :=
  XY^2 + YZ^2 = XZ^2

-- Assert the conditions of the problem
axiom h1 : is_right_triangle AP PB PC
axiom h2 : is_right_triangle PAQ PB PC

-- The theorem that needs to be proved
theorem area_quadrilateral_APBC : 
  area (triangle APQ) + area (triangle PBC) = 300 := 
sorry

end area_quadrilateral_APBC_l19_19370


namespace melted_mixture_weight_l19_19403

/-- 
If the ratio of zinc to copper is 9:11 and 27 kg of zinc has been consumed, then the total weight of the melted mixture is 60 kg.
-/
theorem melted_mixture_weight (zinc_weight : ℕ) (ratio_zinc_to_copper : ℕ → ℕ → Prop)
  (h_ratio : ratio_zinc_to_copper 9 11) (h_zinc : zinc_weight = 27) :
  ∃ (total_weight : ℕ), total_weight = 60 :=
by
  sorry

end melted_mixture_weight_l19_19403


namespace sin_alpha_plus_beta_zero_l19_19992

theorem sin_alpha_plus_beta_zero (α β : ℝ) (h : cos α * cos β = -1) : sin (α + β) = 0 :=
by
  sorry

end sin_alpha_plus_beta_zero_l19_19992


namespace distribution_schemes_of_6_interns_to_3_schools_is_540_l19_19525

theorem distribution_schemes_of_6_interns_to_3_schools_is_540 :
  ∃ (distribution_schemes : ℕ), distribution_schemes = 540 ∧
  (∀ (interns : ℕ) (schools : ℕ) (at_least_one_per_school : Prop) (one_school_per_intern : Prop),
    interns = 6 → schools = 3 → at_least_one_per_school → one_school_per_intern →
    distribution_schemes = distribution_schemes) :=
begin
  sorry
end

end distribution_schemes_of_6_interns_to_3_schools_is_540_l19_19525


namespace tetrahedron_volume_l19_19647

theorem tetrahedron_volume (a b c d i: Point)
  (h_regular_tetrahedron: regular_tetrahedron d a b c)
  (h_side_length_base: length a b = 6 ∧ length b c = 6 ∧ length c a = 6)
  (h_lateral_edge_length: (length d a = 5) ∧ (length d b = 5) ∧ (length d c = 5))
  (h_incenter: incenter i d a b)
  : volume i a b c = (9 / 8) * sqrt 39 := 
sorry

end tetrahedron_volume_l19_19647


namespace remainder_sum_l19_19808

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7) = 7 := by
  sorry

end remainder_sum_l19_19808


namespace derivative_y_l19_19499

noncomputable def y (x : ℝ) : ℝ := (1 + cos (2 * x)) ^ 3

theorem derivative_y (x : ℝ) : deriv y x = -48 * (cos x) ^ 5 * sin x := by
  sorry

end derivative_y_l19_19499


namespace max_marks_l19_19380

theorem max_marks (M : ℝ) (h : 0.80 * M = 240) : M = 300 :=
sorry

end max_marks_l19_19380


namespace dice_probability_l19_19137

/-- Probability that there is at least one pair but not a four-of-a-kind (which also avoids the scenario of a three-of-a-kind) when six standard six-sided dice are rolled. -/
theorem dice_probability :
  let total_outcomes := 6^6,
      successful_outcomes := 6 * 15 * 120 + 15 * 90 * 12 + 20 * 90 in
  (successful_outcomes / total_outcomes: ℚ) = 25 / 81 :=
by
  let total_outcomes := 6^6
  let successful_outcomes := 6 * 15 * 120 + 15 * 90 * 12 + 20 * 90
  have h1 : total_outcomes = 46656 := by norm_num
  have h2 : successful_outcomes = 28800 := by norm_num
  show (successful_outcomes: ℚ) / total_outcomes = 25 / 81
  calc (28800: ℚ) / 46656 = 25 / 81 : by norm_num

sorry

end dice_probability_l19_19137


namespace overlap_length_l19_19766

noncomputable def length_of_all_red_segments := 98 -- in cm
noncomputable def total_length := 83 -- in cm
noncomputable def number_of_overlaps := 6 -- count

theorem overlap_length :
  ∃ (x : ℝ), length_of_all_red_segments - total_length = number_of_overlaps * x ∧ x = 2.5 := by
  sorry

end overlap_length_l19_19766


namespace power_of_a_point_l19_19693

theorem power_of_a_point
  (ω : Circle)
  (A B C D : ω.points)
  (P : Point)
  (hAB : Line_through_points A B P)
  (hCD : Line_through_points C D P) :
  distance P A * distance P B = distance P C * distance P D :=
sorry

end power_of_a_point_l19_19693


namespace jerry_remaining_debt_l19_19263

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jerry_remaining_debt_l19_19263


namespace parabolas_intersect_on_circle_l19_19511

theorem parabolas_intersect_on_circle :
  let parabola1 (x y : ℝ) := y = (x - 2)^2
  let parabola2 (x y : ℝ) := x + 6 = (y + 1)^2
  ∃ (cx cy r : ℝ), ∀ (x y : ℝ), (parabola1 x y ∧ parabola2 x y) → (x - cx)^2 + (y - cy)^2 = r^2 ∧ r^2 = 33/2 :=
by
  sorry

end parabolas_intersect_on_circle_l19_19511


namespace concert_ticket_problem_l19_19467

theorem concert_ticket_problem :
  ∃ (x : ℕ → Bool), 
    (∀ n, x n = (n ∣ 36 ∧ n ∣ 54)) ∧ 
    (finset.card (finset.filter (λ n, x n) (finset.range 37)) = 6) :=
by
  sorry

end concert_ticket_problem_l19_19467


namespace simplify_expression_l19_19118

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1 / 3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 :=
by
  -- The proof is omitted as per the instructions
  sorry

end simplify_expression_l19_19118


namespace conclusion_1_conclusion_3_l19_19917

def tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem conclusion_1 : tensor 2 (-2) = 6 :=
by sorry

theorem conclusion_3 (a b : ℝ) (h : a + b = 0) : tensor a a + tensor b b = 2 * a * b :=
by sorry

end conclusion_1_conclusion_3_l19_19917


namespace handshake_count_l19_19736

theorem handshake_count (couples : ℕ) (people : ℕ) (total_handshakes : ℕ) :
  couples = 6 →
  people = 2 * couples →
  total_handshakes = (people * (people - 1)) / 2 - couples →
  total_handshakes = 60 :=
by
  intros h_couples h_people h_handshakes
  sorry

end handshake_count_l19_19736


namespace ratio_of_final_to_true_average_l19_19104

theorem ratio_of_final_to_true_average {scores : List ℝ} (h : scores.length = 50)
  (A : ℝ) (hA : A = (scores.sum / 50)) :
  let new_scores := scores ++ [A, A] in
  (new_scores.sum / 52) = A :=
by
  sorry

end ratio_of_final_to_true_average_l19_19104


namespace emma_final_balance_correct_l19_19535

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l19_19535


namespace probability_x_plus_y_multiple_of_5_probability_x_minus_y_multiple_of_3_probability_one_of_x_or_y_is_5_or_6_l19_19077

theorem probability_x_plus_y_multiple_of_5 :
  let pairs := [(1,4), (4,1), (2,3), (3,2), (5,5), (4,6), (6,4)]
  let total_pairs := 6 * 6
  let favorable_outcomes := length pairs
  (favorable_outcomes : ℚ) / total_pairs = 7 / 36 :=
by
  let pairs := [(1,4), (4,1), (2,3), (3,2), (5,5), (4,6), (6,4)]
  let total_pairs := 6 * 6
  let favorable_outcomes := pairs.length
  show (favorable_outcomes : ℚ) / total_pairs = 7 / 36
  sorry

theorem probability_x_minus_y_multiple_of_3 :
  let pairs := [(4,1), (5,2), (6,3), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (1,4), (2,5), (3,6)]
  let total_pairs := 6 * 6
  let favorable_outcomes := length pairs
  (favorable_outcomes : ℚ) / total_pairs = 1 / 3 :=
by
  let pairs := [(4,1), (5,2), (6,3), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (1,4), (2,5), (3,6)]
  let total_pairs := 6 * 6
  let favorable_outcomes := pairs.length
  show (favorable_outcomes : ℚ) / total_pairs = 1 / 3
  sorry

theorem probability_one_of_x_or_y_is_5_or_6 :
  let pairs := [(1,5), (5,1), (2,5), (5,2), (3,5), (5,3), (4,5), (5,4), (5,5), (5,6), (6,5), (6,4), (4,6), (3,6), (6,3), (2,6), (6,2), (1,6), (6,1), (6,6)]
  let total_pairs := 6 * 6
  let favorable_outcomes := length pairs
  (favorable_outcomes : ℚ) / total_pairs = 5 / 9 :=
by
  let pairs := [(1,5), (5,1), (2,5), (5,2), (3,5), (5,3), (4,5), (5,4), (5,5), (5,6), (6,5), (6,4), (4,6), (3,6), (6,3), (2,6), (6,2), (1,6), (6,1), (6,6)]
  let total_pairs := 6 * 6
  let favorable_outcomes := pairs.length
  show (favorable_outcomes : ℚ) / total_pairs = 5 / 9
  sorry

end probability_x_plus_y_multiple_of_5_probability_x_minus_y_multiple_of_3_probability_one_of_x_or_y_is_5_or_6_l19_19077


namespace student_marks_l19_19465

theorem student_marks 
  (percentage : ℝ := 0.33)
  (failed_by : ℕ := 56)
  (max_marks : ℕ := 700) :
  let P := percentage * max_marks in
  let M := P - failed_by in
  M = 175 :=
by
  let P := percentage * max_marks
  let M := P - failed_by
  have hP : P = 231 := by norm_num [percentage, max_marks] 
  have hM : M = 231 - 56 := by congr; exact hP; 
  norm_num at hM; exact hM

end student_marks_l19_19465


namespace equal_probability_of_selection_l19_19160

-- Define a structure representing the scenario of the problem.
structure SamplingProblem :=
  (total_students : ℕ)
  (eliminated_students : ℕ)
  (remaining_students : ℕ)
  (selection_size : ℕ)
  (systematic_step : ℕ)

-- Instantiate the specific problem.
def problem_instance : SamplingProblem :=
  { total_students := 3001
  , eliminated_students := 1
  , remaining_students := 3000
  , selection_size := 50
  , systematic_step := 60 }

-- Define the main theorem to be proven.
theorem equal_probability_of_selection (prob : SamplingProblem) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ prob.remaining_students → 
  (prob.remaining_students - prob.systematic_step * ((i - 1) / prob.systematic_step) = i) :=
sorry

end equal_probability_of_selection_l19_19160


namespace minimize_transportation_cost_l19_19742

noncomputable def transportation_cost (v : ℝ) : ℝ :=
  166 * (0.02 * v + 200 / v)

theorem minimize_transportation_cost :
  ∀ v : ℝ, 60 ≤ v ∧ v ≤ 120 → 
    transportation_cost v ≥ 664 ∧ 
    transportation_cost 100 = 664 :=
begin
  sorry
end

end minimize_transportation_cost_l19_19742


namespace max_sum_two_digit_primes_l19_19109

theorem max_sum_two_digit_primes : (89 + 97) = 186 := 
by
  sorry

end max_sum_two_digit_primes_l19_19109


namespace max_value_reached_l19_19279

noncomputable def max_value_problem (a b c d : ℝ) : ℝ :=
  a + b^2 + c^3 + d^4

theorem max_value_reached (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 2) :
  max_value_problem a b c d ≤ 2 :=
begin
  sorry
end

end max_value_reached_l19_19279


namespace wholesale_price_l19_19460

theorem wholesale_price (W R SP : ℝ) (h1 : R = 120) (h2 : SP = R - 0.10 * R) (h3 : SP = W + 0.20 * W) : 
  W = 90 :=
by
  -- Given conditions
  have hR : R = 120 := h1
  have hSP_def : SP = 108 := by
    rw [h1] at h2
    norm_num at h2
  -- Solving for W
  sorry

end wholesale_price_l19_19460


namespace non_empty_proper_subsets_count_l19_19072

theorem non_empty_proper_subsets_count (A : set ℕ) (hA : A = {0, 1, 2, 3}) : 
  ∃ n : ℕ, n = (2 ^ 4 - 1) - 1 ∧ n = 14 :=
by {
  sorry
}

end non_empty_proper_subsets_count_l19_19072


namespace average_growth_rate_eq_l19_19967

theorem average_growth_rate_eq :
  ∀ (x : ℝ),
  let initial_income := 5.76 in
  let final_income := 6.58 in
  initial_income * (1 + x)^2 = final_income :=
by
  sorry

end average_growth_rate_eq_l19_19967


namespace dave_fifth_store_car_count_l19_19400

theorem dave_fifth_store_car_count :
  let cars_first_store := 30
  let cars_second_store := 14
  let cars_third_store := 14
  let cars_fourth_store := 21
  let mean := 20.8
  let total_cars := mean * 5
  let total_cars_first_four := cars_first_store + cars_second_store + cars_third_store + cars_fourth_store
  total_cars - total_cars_first_four = 25 := by
sorry

end dave_fifth_store_car_count_l19_19400


namespace max_y_coordinate_l19_19938

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l19_19938


namespace sin_sum_to_fraction_l19_19904

theorem sin_sum_to_fraction :
  ∑ k in finset.range (10 + 1), (Real.sin (k * (Math.pi / 18)))^6 = 53 / 16 :=
by
  sorry

end sin_sum_to_fraction_l19_19904


namespace r_plus_s_l19_19696

/-- Given x is a real value such that ∛x + ∛(30 - x) = 3,
    and the smaller of the two real values of x can be expressed as r - √s where r and s are integers.
    Prove that r + s = 14. -/
theorem r_plus_s (x r s : ℝ) (h1 : ∧ (∃ t1 t2 : ℝ, t1 = real.cbrt x ∧ t2 = real.cbrt (30 - x) ∧ t1 + t2 = 3))
  (h2 : ∀ u v : ℝ, (u = real.cbrt x ∧ v = real.cbrt (30 - x)) -> (u+v=3)) 
  :
  (r = 3 ∧ s = 11) -> r + s = 14 :=
by
  sorry

end r_plus_s_l19_19696


namespace geometric_sequence_xz_eq_three_l19_19164

theorem geometric_sequence_xz_eq_three 
  (x y z : ℝ)
  (h1 : ∃ r : ℝ, x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * z = 3 :=
by
  -- skip the proof
  sorry

end geometric_sequence_xz_eq_three_l19_19164


namespace harmonic_mean_closest_to_6_l19_19759

open Real

def harmonic_mean (a b : ℕ) : ℚ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_6 : harmonic_mean 3 504 ≈ 6 := by
  unfold harmonic_mean
  simp [div_eq_inv_mul]
  norm_num
  sorry

end harmonic_mean_closest_to_6_l19_19759


namespace radii_touching_circles_l19_19025

noncomputable def radius_of_circles_touching_unit_circles 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (centerA centerB centerC : A) 
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius) 
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius) 
  : Prop :=
  ∃ r₁ r₂ : ℝ, r₁ = 1/3 ∧ r₂ = 7/3

theorem radii_touching_circles (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (centerA centerB centerC : A)
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius)
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius)
  : radius_of_circles_touching_unit_circles A B C centerA centerB centerC unit_radius h1 h2 h3 :=
sorry

end radii_touching_circles_l19_19025


namespace bikers_passes_l19_19780

theorem bikers_passes (b_travels : ℕ) (s_travels : ℕ) (h_b : b_travels = 11) (h_s : s_travels = 7) :
  ∃ passes : ℕ, passes = 7 + 1 :=
by
  use 8
  sorry

end bikers_passes_l19_19780


namespace complex_numbers_addition_l19_19993

theorem complex_numbers_addition (a b : ℝ) (h : (1 + 2 * Complex.i) / (a + b * Complex.i) = 1 + Complex.i) : a + b = 2 := 
sorry

end complex_numbers_addition_l19_19993


namespace intersection_A_B_l19_19273

def A : Set ℝ := {x | x^2 = 1}
def B : Set ℝ := {x | x^2 - 2x - 3 = 0}

theorem intersection_A_B :
  A ∩ B = {-1} :=
by sorry

end intersection_A_B_l19_19273


namespace two_point_seven_five_as_fraction_l19_19798

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l19_19798


namespace circumcircle_points_l19_19686

theorem circumcircle_points (A B C P : Point)
  (h_distinct: P ≠ A ∧ P ≠ B ∧ P ≠ C):
  (∃ R : ℝ, circumcircle_radius A B P = R ∧ circumcircle_radius B C P = R ∧ circumcircle_radius C A P = R) ↔
  (is_on_circumcircle A B C P ∨ is_orthocenter A B C P) :=
sorry

end circumcircle_points_l19_19686


namespace ellipse_min_area_contains_circles_l19_19908

-- Define the ellipse and circles
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def circle1 (x y : ℝ) := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) := ((x + 2)^2 + y^2 = 4)

-- Proof statement: The smallest possible area of the ellipse containing the circles
theorem ellipse_min_area_contains_circles : 
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), 
    (circle1 x y → ellipse x y) ∧ 
    (circle2 x y → ellipse x y)) ∧
  (k = 12) := 
sorry

end ellipse_min_area_contains_circles_l19_19908


namespace total_number_of_meetings_proof_l19_19035

-- Define the conditions in Lean
variable (A B : Type)
variable (starting_time : ℕ)
variable (location_A location_B : A × B)

-- Define speeds
variable (speed_A speed_B : ℕ)

-- Define meeting counts
variable (total_meetings : ℕ)

-- Define A reaches point B 2015 times
variable (A_reaches_B_2015 : Prop)

-- Define that B travels twice as fast as A
axiom speed_ratio : speed_B = 2 * speed_A

-- Define that A reaches point B for the 5th time when B reaches it for the 9th time
axiom meeting_times : A_reaches_B_2015 → (total_meetings = 6044)

-- The Lean statement to prove
theorem total_number_of_meetings_proof : A_reaches_B_2015 → total_meetings = 6044 := by
  sorry

end total_number_of_meetings_proof_l19_19035


namespace line_segment_proportionality_l19_19483

theorem line_segment_proportionality :
  (∀ (a b c d : ℝ), a = 3 ∧ b = 6 ∧ c = 2 ∧ d = 4 → a / b = c / d) ∧
  (∀ (a b c d : ℝ), a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 4 → a / b = c / d) ∧
  (∀ (a b c d : ℝ), a = 4 ∧ b = 6 ∧ c = 5 ∧ d = 10 → a / b ≠ c / d) ∧
  (∀ (a b c d : ℝ), a = 1 ∧ b = 1/2 ∧ c = 1/6 ∧ d = 1/3 → b / a = c / d) ∧
  (∀ (a b c d : ℝ), a = 4 ∧ b = 6 ∧ c = 5 ∧ d = 10 → a / b ≠ c / d) :=
begin
  sorry
end

end line_segment_proportionality_l19_19483


namespace slope_of_given_line_is_30_degrees_l19_19012
noncomputable def line_angle (x y : ℝ) : ℝ :=
  let m := -1 / real.sqrt 3 in
  real.arctan m

theorem slope_of_given_line_is_30_degrees :
  line_angle 1 (-real.sqrt 3) = real.pi / 6 :=
by
  -- defining the line equation x - sqrt(3)y + 1 = 0
  let line := λ (x y : ℝ), x - real.sqrt 3 * y + 1
  sorry

end slope_of_given_line_is_30_degrees_l19_19012


namespace find_equation_of_curve_find_fixed_point_l19_19432

theorem find_equation_of_curve 
  (x y : ℝ)
  (O : x^2 + y^2 = 4) 
  (F : (1, 0)) 
  (P : ℝ × ℝ)
  (tangent : ∃ P, circle_diameter_fp_tangent_FO x y O F P) :
  ∃ C, ∀ P ∈ C, ∃ S, is_midpoint S (F, P) ∧ ∃ F', is_symmetric F F' ∧ ellipse_with_foci_FF' C S :=
sorry

theorem find_fixed_point 
  (M N : ℝ × ℝ)
  (hM : M ∈ curve_C)
  (hN : N ∈ curve_C)
  (line_MN : ∃ k : ℝ, ∀ x y, y = k * x + 1/2)
  (Q : ℝ × ℝ)
  (point_0_1_2 : is_point_on_line (0, 1/2) line_MN)
  (angle_MQO_NQO : are_angles_equal (MQO M Q O) (NQO N Q O)) : 
  Q = (0, 6) :=
sorry

end find_equation_of_curve_find_fixed_point_l19_19432


namespace find_length_BC_l19_19306

noncomputable def circle_length_BC (r : ℝ) (alpha : ℝ) : ℝ :=
  let OB := r
  let OC := r
  let cos_alpha := ℂ.cos(alpha)
  2 * OB * cos_alpha

theorem find_length_BC : 
  (∀ (O A M B C : Type) (r : ℝ) (alpha : ℝ), 
   -- Conditions
   r = 12 ∧ cos(alpha) = 1 / 4 ∧
   (M : Type) ∧
   (B : Type) ∧
   (C : Type) ∧
   angle A M B = alpha ∧ angle O M C = alpha 
   -- Result
   → circle_length_BC r alpha = 6) :=
sorry

end find_length_BC_l19_19306


namespace possible_sums_in_circle_l19_19526

noncomputable def possible_circle_sums {a b c d : ℕ} (h : {a, b, c, d} = {2, 4, 6, 8} ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : ℕ := 
  let prod1 := a * b in 
  let prod2 := a * c in 
  let prod3 := b * d in 
  let prod4 := c * d in 
  prod1 + prod2 + prod3 + prod4

theorem possible_sums_in_circle :
  ∀ (a b c d : ℕ),
    {a, b, c, d} = {2, 4, 6, 8} ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
    → possible_circle_sums {a, b, c, d} = 84 ∨ possible_circle_sums {a, b, c, d} = 96 ∨ possible_circle_sums {a, b, c, d} = 100 :=
by
  intros a b c d h,
  sorry

end possible_sums_in_circle_l19_19526


namespace circles_are_separated_l19_19987

noncomputable def circle_center_and_radius (h : ℝ × ℝ → Prop) : ℝ × ℝ × ℝ :=
  let ⟨cx, cy, r, h_eq⟩ := classical.some_spec (exists_quadratic_eq_of_circle_eq h)
  (cx, cy, r)

def O1 := circle_center_and_radius (λ p, p.1^2 + p.2^2 = 1)
def O2 := circle_center_and_radius (λ p, (p.1 - 3)^2 + (p.2 + 4)^2 = 9)

theorem circles_are_separated :
  let d := dist (prod.fst O1, prod.snd O1) (prod.fst O2, prod.snd O2)
  d > (O1.2 + O2.2) :=
by {
  sorry
}

end circles_are_separated_l19_19987


namespace biking_distance_l19_19476

/-- Mathematical equivalent proof problem for the distance biked -/
theorem biking_distance
  (x t d : ℕ)
  (h1 : d = (x + 1) * (3 * t / 4))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 36 :=
by
  -- The proof goes here
  sorry

end biking_distance_l19_19476


namespace min_tangent_length_is_4_l19_19628

-- Define the circle and symmetry conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0
def symmetry_condition (a b : ℝ) : Prop := 2*a*(-1) + b*2 + 6 = 0

-- Define the length of the tangent line from (a, b) to the circle center (-1, 2)
def min_tangent_length (a b : ℝ) : ℝ :=
  let d := Real.sqrt ((a + 1)^2 + (b - 2)^2) in
  d - Real.sqrt 2

-- Prove that the minimum tangent length is 4 given the conditions
theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_condition a b →
  ∃ (min_len : ℝ), min_len = min_tangent_length a b ∧ min_len = 4 :=
by
  sorry

end min_tangent_length_is_4_l19_19628


namespace unique_solution_l19_19519

theorem unique_solution (a n : ℕ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 3^n = a^2 - 16) : a = 5 ∧ n = 2 :=
by
sorry

end unique_solution_l19_19519


namespace hypotenuse_length_l19_19462

-- Assume we have real numbers a and b such that the data constraints are met
variables (a b : ℝ)

-- First condition: volume of the first cone
def first_cone_volume : Prop := (1 / 3) * π * b^2 * a = 900 * π

-- Second condition: volume of the second cone
def second_cone_volume : Prop := (1 / 3) * π * a^2 * b = 1800 * π

-- Conclusion: hypotenuse length of the right-angle triangle
theorem hypotenuse_length (h1 : first_cone_volume a b) (h2 : second_cone_volume a b) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c = Real.sqrt 605 :=
by sorry

end hypotenuse_length_l19_19462


namespace cone_volume_270_degree_sector_l19_19440

noncomputable def coneVolumeDividedByPi (R θ: ℝ) (r h: ℝ) (circumf sector_height: ℝ) : ℝ := 
  if R = 20 
  ∧ θ = 270 / 360 
  ∧ 2 * Mathlib.pi * 20 = 40 * Mathlib.pi 
  ∧ circumf = 30 * Mathlib.pi
  ∧ 2 * Mathlib.pi * r = circumf
  ∧ r = 15
  ∧ sector_height = R
  ∧ r^2 + h^2 = sector_height^2 
  then (1/3) * Mathlib.pi * r^2 * h / Mathlib.pi 
  else 0

theorem cone_volume_270_degree_sector : coneVolumeDividedByPi 20 (270 / 360) 15 (5 * Real.sqrt 7) (30 * Mathlib.pi) 20 = 1125 * Real.sqrt 7 := 
by {
  -- This is where the proof would go
  sorry
}

end cone_volume_270_degree_sector_l19_19440


namespace number_of_right_triangles_l19_19348

-- Definitions for the points.
variables (A P B C Q D : Type) [point A] [point P] [point B] [point C] [point Q] [point D]

-- Rectangle and congruent squares condition.
variables (rect : rectangle ABCD) (sq_div : divides_into_congruent_squares rect PQ)

-- Main theorem statement.
theorem number_of_right_triangles :
  count_right_triangles {A, P, B, C, Q, D}  = 14 :=
sorry

end number_of_right_triangles_l19_19348


namespace smallest_integer_form_l19_19387

theorem smallest_integer_form (m n : ℤ) : ∃ (a : ℤ), a = 2011 * m + 55555 * n ∧ a > 0 → a = 1 :=
by
  sorry

end smallest_integer_form_l19_19387


namespace son_l19_19368

variable (M S : ℕ)

theorem son's_age (h1 : M = 4 * S) (h2 : (M - 3) + (S - 3) = 49) : S = 11 :=
by
  sorry

end son_l19_19368


namespace intersection_M_N_l19_19706

def M : Set ℝ := {y | ∃ x, x ∈ Set.Icc (-5) 5 ∧ y = 2 * Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N : {x | 1 < x ∧ x ≤ 2} = {x | x ∈ M ∩ N} :=
by sorry

end intersection_M_N_l19_19706


namespace find_k_l19_19472

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (Real.sqrt p.1, Real.sqrt p.2)

-- Define the vertices of the quadrilateral
def A := (900, 300 : ℝ × ℝ)
def B := (1800, 600 : ℝ × ℝ)
def C := (600, 1800 : ℝ × ℝ)
def D := (300, 900 : ℝ × ℝ)

-- Define the transformed vertices
def A' := transform A
def B' := transform B
def C' := transform C
def D' := transform D

-- Define the main theorem
theorem find_k : ∃ k : ℝ, k ≤ 100 * Real.pi ∧ k = 314 :=
by
  sorry

end find_k_l19_19472


namespace equivalent_proof_problem_l19_19572

noncomputable def proof_problem : Prop :=
  ∀ (a b x : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    ( (sin x) ^ 4 / a ^ 2 + (cos x) ^ 4 / b ^ 2 = 1 / (a ^2 + b ^ 2) ) →
    ( (sin x) ^ 100 / a ^ 100 + (cos x) ^ 100 / b ^ 100 = 2 / (a ^2 + b ^ 2) ^ 100 )

theorem equivalent_proof_problem : proof_problem := 
  by 
  sorry

end equivalent_proof_problem_l19_19572


namespace sequence_a2_l19_19673

noncomputable def sequence (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    nat.rec_on (n / 2) 0 (λ n' an', 2 * an' + 1)
  else
    0

theorem sequence_a2 :
  (∀ n, sequence (2 * n) = 2 * sequence (2 * (n - 1)) + 1) →
  sequence 16 = 127 →
  sequence 2 = 0 :=
by
  intros h_seq h_16
  sorry

end sequence_a2_l19_19673


namespace find_number_l19_19848

-- Define the condition
def condition (x : ℝ) : Prop :=
  x + 0.35 * x = x + 150

-- State the theorem to prove the correct answer
theorem find_number : ∃ x : ℝ, condition x ∧ x = 428.57 :=
by
  use 428.57
  unfold condition
  split
  · sorry
  · sorry

end find_number_l19_19848


namespace least_subtracted_number_l19_19390

theorem least_subtracted_number (a b c d e : ℕ) 
  (h₁ : a = 2590) 
  (h₂ : b = 9) 
  (h₃ : c = 11) 
  (h₄ : d = 13) 
  (h₅ : e = 6) 
  : ∃ (x : ℕ), a - x % b = e ∧ a - x % c = e ∧ a - x % d = e := by
  sorry

end least_subtracted_number_l19_19390


namespace greatest_divisor_l19_19928

theorem greatest_divisor (n : ℕ) (h1 : 3461 % n = 23) (h2 : 4783 % n = 41) : n = 2 := by {
  sorry
}

end greatest_divisor_l19_19928


namespace max_y_coordinate_is_three_fourths_l19_19939

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l19_19939


namespace simplified_expression_evaluation_l19_19330

-- Problem and conditions
def x := Real.sqrt 5 - 1

-- Statement of the proof problem
theorem simplified_expression_evaluation : 
  ( (x / (x - 1) - 1) / (x^2 - 1) / (x^2 - 2 * x + 1) ) = Real.sqrt 5 / 5 :=
sorry

end simplified_expression_evaluation_l19_19330


namespace exclude_three_digit_patterns_l19_19616

theorem exclude_three_digit_patterns : 
  let num3d := finset.range 900 \ finset.range 100,
      aba := {n : ℕ | ∃ a b : ℕ, a ≠ b ∧ n = 100 * a + 10 * b + a},
      aca := {n : ℕ | ∃ a c : ℕ, a ≠ c ∧ n = 100 * a + 10 * c + a}
  in num3d.card - (aba ∩ num3d).card - (aca ∩ num3d).card = 738 :=
  by sorry

end exclude_three_digit_patterns_l19_19616


namespace sum_of_squares_l19_19770

-- Define the main theorem statement
theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + a * c = 131) : 
  a^2 + b^2 + c^2 = 62 :=
begin
  sorry
end

end sum_of_squares_l19_19770


namespace rank_from_right_l19_19867

theorem rank_from_right (n total rank_left : ℕ) (h1 : rank_left = 5) (h2 : total = 21) : n = total - (rank_left - 1) :=
by {
  sorry
}

end rank_from_right_l19_19867


namespace power_function_even_m_eq_neg1_l19_19635

-- Define the power function
def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - m - 1) * x^(1 - m)

-- Define the property of being an even function
def is_even_function (f : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- The theorem stating the problem and answer
theorem power_function_even_m_eq_neg1 :
  (∀ x : ℝ, power_function m x = power_function m (-x)) ↔ m = -1 :=
by
  sorry  -- Proof omitted

end power_function_even_m_eq_neg1_l19_19635


namespace find_initial_velocity_l19_19683

noncomputable def distance := 30
noncomputable def g := 9.8
noncomputable def angle := Real.pi / 6 -- 30 degrees in radians

theorem find_initial_velocity (v : ℝ) :
  let d := distance
  let g := g
  let θ := angle
  d = (v^2 * Real.sqrt 3) / (2 * g) → 
  v = 19 :=
by
  assume d g θ h
  have h₁ : d = 30 := by sorry
  have h₂ : g = 9.8 := by sorry
  have h₃ : θ = Real.pi / 6 := by sorry
  have h₄ : (v^2 * Real.sqrt 3) / (2 * g) = 19 * 19 := by sorry
  sorry

end find_initial_velocity_l19_19683


namespace cost_price_cupboard_l19_19381

theorem cost_price_cupboard (C : ℝ) 
  (sell_below_cost : 0.84 * C)
  (sell_profit : 1.16 * C)
  (profit_diff : 1.16 * C - 0.84 * C = 1800) : 
  C = 5625 :=
by
  sorry

end cost_price_cupboard_l19_19381


namespace modulus_sum_of_two_complex_numbers_l19_19192

theorem modulus_sum_of_two_complex_numbers
  (z1 z2 : ℂ)
  (h1 : ∥z1∥ = 1)
  (h2 : ∥z2∥ = 1)
  (h3 : ∥z1 - z2∥ = 1) :
  ∥z1 + z2∥ = Real.sqrt 3 :=
by
  sorry

end modulus_sum_of_two_complex_numbers_l19_19192


namespace worker_b_days_l19_19405

variables (W_A W_B W : ℝ)
variables (h1 : W_A = 2 * W_B)
variables (h2 : (W_A + W_B) * 10 = W)
variables (h3 : W = 30 * W_B)

theorem worker_b_days : ∃ days : ℝ, days = 30 :=
by
  sorry

end worker_b_days_l19_19405


namespace two_pt_seven_five_as_fraction_l19_19795

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l19_19795


namespace handshake_problem_l19_19528

theorem handshake_problem :
  ∃ (n : ℕ), n = 8 → 
  let people := 2 * n in
  let handshakes_per_person := people - 2 in
  let total_handshakes := people * handshakes_per_person / 2 in
  total_handshakes = 112 :=
by
  intro n hn
  let people := 2 * n
  let handshakes_per_person := people - 2
  let total_handshakes := people * handshakes_per_person / 2
  sorry

end handshake_problem_l19_19528


namespace find_range_of_a_l19_19611

-- Define the propositions p and q
def p (a : ℝ) : Prop := a > 1
def q (a : ℝ) : Prop := -2 < a ∧ a < 2

-- Proposition proving that (p ∨ q) ∧ ¬(p ∧ q) implies a ∈ (-2, 1] ∪ [2, +∞)
theorem find_range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : a ∈ set.Icc (-2 : ℝ) 1 ∪ set.Ici 2 :=
sorry

end find_range_of_a_l19_19611


namespace complex_number_quadrant_l19_19570

variable (i : ℂ)
variable (z : ℂ)

noncomputable def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant (i_re_im : i = complex.I) (z_val : z = -3 - 2*complex.I) : 
  is_in_third_quadrant (z) :=
by {
  have h1: z.re = -3, by sorry,
  have h2: z.im = -2, by sorry,
  exact ⟨h1.le, h2.le⟩, 
  sorry
}

end complex_number_quadrant_l19_19570


namespace range_of_m_l19_19154

theorem range_of_m (m : ℤ) :
  (∀ x : ℤ, 2 < 2 * x - m ∧ 2 * x - m < 8) → (∑ x in set_of (λ x, 2 < 2 * x - m ∧ 2 * x - m < 8), x = 0) ↔ -6 < m ∧ m < -4 := 
sorry

end range_of_m_l19_19154


namespace interest_rate_calculation_l19_19852

variables (face_value dividend_percentage market_value : ℝ)

def dividend_per_share : ℝ := (dividend_percentage / 100) * face_value

def interest_rate : ℝ := (dividend_per_share / market_value) * 100

theorem interest_rate_calculation 
  (h_face_value : face_value = 60)
  (h_dividend_percentage : dividend_percentage = 9)
  (h_market_value : market_value = 45) :
  interest_rate face_value dividend_percentage market_value = 12 :=
by
  -- Proof omitted
  sorry

end interest_rate_calculation_l19_19852


namespace theta_in_fourth_quadrant_l19_19427

theorem theta_in_fourth_quadrant 
    (θ : ℝ) 
    (h1 : cos θ > 0) 
    (h2 : sin (2 * θ) < 0) : 
    3 * π / 2 < θ ∧ θ < 2 * π := 
sorry

end theta_in_fourth_quadrant_l19_19427


namespace fraction_simplification_l19_19735

theorem fraction_simplification :
  (36 / 19) * (57 / 40) * (95 / 171) = (3 / 2) :=
by
  sorry

end fraction_simplification_l19_19735


namespace cone_volume_divided_by_pi_l19_19435

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l19_19435


namespace total_albums_l19_19712

variables {Adele Bridget Katrina Miriam : ℕ}

-- Define the conditions as Lean statements
def condition_adele (A : ℕ) := A = 30
def condition_bridget (B A : ℕ) := B = A - 15
def condition_katrina (K B : ℕ) := K = 6 * B
def condition_miriam (M K : ℕ) := M = 5 * K

theorem total_albums : ∀ (A B K M : ℕ),
  condition_adele A →
  condition_bridget B A →
  condition_katrina K B →
  condition_miriam M K →
  A + B + K + M = 585 :=
by {
  intros,
  sorry
}

end total_albums_l19_19712


namespace cameron_total_questions_l19_19897

theorem cameron_total_questions :
  let questions_per_tourist := 2
  let first_group := 6
  let second_group := 11
  let third_group := 8
  let third_group_special_tourist := 1
  let third_group_special_questions := 3 * questions_per_tourist
  let fourth_group := 7
  let first_group_total_questions := first_group * questions_per_tourist
  let second_group_total_questions := second_group * questions_per_tourist
  let third_group_total_questions := (third_group - third_group_special_tourist) * questions_per_tourist + third_group_special_questions
  let fourth_group_total_questions := fourth_group * questions_per_tourist
  in first_group_total_questions + second_group_total_questions + third_group_total_questions + fourth_group_total_questions = 68 := by
  sorry

end cameron_total_questions_l19_19897


namespace equal_perimeter_triangle_side_length_l19_19457

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length_l19_19457


namespace expressible_in_first_1000_positive_integers_l19_19212

def floor (x : ℝ) : ℕ := ⌊x⌋
def f (x : ℝ) : ℕ := floor (2 * x) + floor (4 * x) + floor (6 * x) + floor (8 * x)

theorem expressible_in_first_1000_positive_integers : 
  (∃ n : ℕ, n ≤ 1000 ∧ (∃ x : ℝ, f x = n)) ↔ 
  (∃ Y : List ℕ, Y.length = 600 ∧ Y.all (λ n, n ≤ 1000 ∧ ∃ x : ℝ, f x = n)) := 
sorry

end expressible_in_first_1000_positive_integers_l19_19212


namespace milk_problem_l19_19814

theorem milk_problem (x : ℕ) (hx : 0 < x)
    (total_cost_wednesday : 10 = x * (10 / x))
    (price_reduced : ∀ x, 0.5 = (10 / x - (10 / x) + 0.5))
    (extra_bags : 2 = (x + 2) - x)
    (extra_cost : 2 + 10 = x * (10 / x) + 2) :
    x^2 + 6 * x - 40 = 0 := by
  sorry

end milk_problem_l19_19814


namespace arithmetic_sequence_sum_l19_19805

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end arithmetic_sequence_sum_l19_19805


namespace average_of_11_results_l19_19747

theorem average_of_11_results (a b c : ℝ) (avg_first_6 avg_last_6 sixth_result avg_all_11 : ℝ)
  (h1 : avg_first_6 = 58)
  (h2 : avg_last_6 = 63)
  (h3 : sixth_result = 66) :
  avg_all_11 = 60 :=
by
  sorry

end average_of_11_results_l19_19747


namespace five_digit_numbers_equality_l19_19810

theorem five_digit_numbers_equality : 
  let count_not_divisible_by_5 := 9 * 10^3 * 8,
      count_first_two_digits_not_five := 8 * 9 * 10^3 
  in count_not_divisible_by_5 = 72000 ∧ count_first_two_digits_not_five = 72000 ∧ count_not_divisible_by_5 = count_first_two_digits_not_five :=
by
  sorry

end five_digit_numbers_equality_l19_19810


namespace max_y_coordinate_l19_19937

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l19_19937


namespace area_enclosed_l19_19040

/-- 
  Given the condition that the region is defined by the equation x^2 + y^2 - 4x + 2y = -8,
  prove that the area enclosed by this region is 3π.
-/
theorem area_enclosed (x y : ℝ) (h : x^2 + y^2 - 4x + 2y = -8) : 
  ∃ a : ℝ, a = 3 * Real.pi := by 
  sorry

end area_enclosed_l19_19040


namespace vector_coordinates_l19_19584

-- Given a unit orthogonal basis {i, j, k} of a vector space
variables (i j k : ℝ^3)
variables (ui : i = (1, 0, 0)) (uj : j = (0, 1, 0)) (uk : k = (0, 0, 1))
variables (orthogonal_basis : orthonormal_basis (fin 3) ℝ^3 ![i, j, k])

-- The vector b = -5i + 2k
def b : ℝ^3 := -5 • i + 2 • k

-- Prove that b can be represented in coordinate form as (-5, 0, 2)
theorem vector_coordinates : b = (-5, 0, 2) :=
by
  -- Skipping the detailed proof steps
  sorry

end vector_coordinates_l19_19584


namespace additional_people_needed_l19_19139

theorem additional_people_needed (k m : ℕ) (h1 : 8 * 3 = k) (h2 : m * 2 = k) : (m - 8) = 4 :=
by
  sorry

end additional_people_needed_l19_19139


namespace cos_neg_pi_over_3_l19_19144

theorem cos_neg_pi_over_3 : Real.cos (-π / 3) = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l19_19144


namespace projection_is_b_diff_is_perpendicular_l19_19613

noncomputable def vector_a : ℝ × ℝ := (2, 0)
noncomputable def vector_b : ℝ × ℝ := (1, 1)

def projection_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let dot_product := a.1 * b.1 + a.2 * b.2 
  let scalar := dot_product / b_norm_sq
  (scalar * b.1, scalar * b.2)

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2 = 0)

theorem projection_is_b : projection_vector vector_a vector_b = vector_b :=
  sorry

theorem diff_is_perpendicular : is_perpendicular (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) vector_b :=
  sorry

end projection_is_b_diff_is_perpendicular_l19_19613


namespace vector_addition_correct_l19_19612

def a : ℝ × ℝ := (-1, 6)
def b : ℝ × ℝ := (3, -2)
def c : ℝ × ℝ := (2, 4)

theorem vector_addition_correct : a + b = c := by
  sorry

end vector_addition_correct_l19_19612


namespace A_share_of_annual_gain_l19_19404

variable (x : ℝ) (annual_gain : ℝ) (six_months : ℝ := 6) (eight_months : ℝ := 8) (twelve_months : ℝ := 12)
variable (B_investment : ℝ := 2 * x) (C_investment : ℝ := 3 * x)

theorem A_share_of_annual_gain :
  annual_gain = 12000 →
  (1 / 3) * annual_gain = 4000 :=
by
  intros hg
  rw [hg]
  norm_num
  sorry

end A_share_of_annual_gain_l19_19404


namespace segment_ratios_l19_19301

theorem segment_ratios 
  (AB_parts BC_parts : ℝ) 
  (hAB: AB_parts = 3) 
  (hBC: BC_parts = 4) 
  : AB_parts / (AB_parts + BC_parts) = 3 / 7 ∧ BC_parts / (AB_parts + BC_parts) = 4 / 7 := 
  sorry

end segment_ratios_l19_19301


namespace primes_finite_l19_19174

theorem primes_finite (p : ℕ → ℕ) (h1 : ∀ n, Prime (p n))
  (h2 : ∀ n, p (n+2) = (p n + p (n+1) + 2018).natAbs.primeFactors.max') :
  ∃ N, ∀ n, n > N → p n ≤ k * Nat.factorial 2021 + 1 :=
sorry

end primes_finite_l19_19174


namespace five_digit_numbers_l19_19349

def divisible_by_4_and_9 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0)

def is_candidate (n : ℕ) : Prop :=
  ∃ a b, n = 10000 * a + 1000 + 200 + 30 + b ∧ a < 10 ∧ b < 10

theorem five_digit_numbers :
  ∀ (n : ℕ), is_candidate n → divisible_by_4_and_9 n → n = 11232 ∨ n = 61236 :=
by
  sorry

end five_digit_numbers_l19_19349


namespace equivalent_multipliers_l19_19222

variable (a b c : ℝ)

theorem equivalent_multipliers :
  (a - 0.07 * a + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c :=
sorry

end equivalent_multipliers_l19_19222


namespace math_proof_problem_l19_19666

noncomputable def problem_statement : Prop :=
  ∃ (P : ℝ × ℝ), P = (-1, 1) ∧
    ∀ (A B : ℝ × ℝ), 
      (A ≠ B ∧ 
       (A.1^2 + A.2^2 = 4) ∧ 
       (B.1^2 + B.2^2 = 4) ∧ 
       (∃ (l : ℝ → ℝ → Prop), l P.1 P.2 ∧ l A.1 A.2 ∧ l B.1 B.2)) →
      dist A B = 2 * real.sqrt 3 →
        (l (1, 0) → l (-1, 1) = l (-1, 1) ∨ l (0, 1) → l (0, -1) = l (1, 1)) ∧
        (∃ M : ℝ × ℝ, (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) ∧
                     ((M.1 + 1/2)^2 + (M.2 - 1/2)^2 = 1/2))

theorem math_proof_problem : problem_statement := sorry

end math_proof_problem_l19_19666


namespace robin_total_spending_l19_19313

def jessica_bracelets := 7
def tori_bracelets := 4
def lily_bracelets := 4
def patrice_metal_bracelets := 3
def patrice_beaded_bracelets := 4

def plastic_bracelet_cost := 2
def metal_bracelet_cost := 3
def beaded_bracelet_cost := 5

def total_bracelets := jessica_bracelets + tori_bracelets + lily_bracelets + patrice_metal_bracelets + patrice_beaded_bracelets

def total_cost_before_discount := (jessica_bracelets * plastic_bracelet_cost) + 
                                  (tori_bracelets * metal_bracelet_cost) + 
                                  (lily_bracelets * beaded_bracelet_cost) + 
                                  (patrice_metal_bracelets * metal_bracelet_cost) + 
                                  (patrice_beaded_bracelets * beaded_bracelet_cost)

def discount := if total_bracelets >= 10 then total_cost_before_discount * 0.10 else 0
def discounted_total := total_cost_before_discount - discount

def sales_tax := discounted_total * 0.07
def final_total := discounted_total + sales_tax

theorem robin_total_spending : final_total = 72.23 :=
by
  sorry

end robin_total_spending_l19_19313


namespace triangle_identity_l19_19256

variables (a b c h_a h_b h_c x y z : ℝ)

-- Define the given conditions
def condition1 := a / h_a = x
def condition2 := b / h_b = y
def condition3 := c / h_c = z

-- Statement of the theorem to be proved
theorem triangle_identity 
  (h1 : condition1 a h_a x) 
  (h2 : condition2 b h_b y) 
  (h3 : condition3 c h_c z) : 
  x^2 + y^2 + z^2 - 2 * x * y - 2 * y * z - 2 * z * x + 4 = 0 := 
  by 
    sorry

end triangle_identity_l19_19256


namespace one_correct_prop_l19_19580

section Propositions

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions for perpendicular and parallel of lines and planes
def perp (x : Line) (y : Plane) : Prop := sorry -- replace with actual definition
def parallel (x : Line) (y : Plane) : Prop := sorry -- replace with actual definition
def perp_planes (x y : Plane) : Prop := sorry -- replace with actual definition
def parallel_planes (x y : Plane) : Prop := sorry -- replace with actual definition

-- Propositions
def Prop1 : Prop := (perp m α) ∧ (perp n β) ∧ (perp m n) → perp_planes α β
def Prop2 : Prop := (parallel m α) ∧ (parallel n β) ∧ (parallel m n) → parallel_planes α β
def Prop3 : Prop := (perp m α) ∧ (parallel n β) ∧ (perp m n) → perp_planes α β
def Prop4 : Prop := (perp m α) ∧ (parallel n β) ∧ (parallel m n) → parallel_planes α β

-- Correctness of each proposition
def correct_prop1 : Prop := sorry -- proof that Prop1 is correct
def correct_prop2 : Prop := sorry -- proof that Prop2 is incorrect
def correct_prop3 : Prop := sorry -- proof that Prop3 is incorrect
def correct_prop4 : Prop := sorry -- proof that Prop4 is incorrect

-- The main statement: exactly one proposition is correct
theorem one_correct_prop : (∃! p, p = Prop1 ∧ correct_prop1 p) := sorry

end Propositions

end one_correct_prop_l19_19580


namespace vector_projection_eq_minus_four_l19_19995

variables (a b : EuclideanSpace ℝ (Fin 3)) -- Euclidean space for vectors a and b
variables (ha : ∥a∥ = 5)
variables (hb : ∥b∥ = 3)
variables (hdot : inner a b = -12)

theorem vector_projection_eq_minus_four :
  (5 * (inner a b / (∥a∥ * ∥b∥)) = -4) :=
by
  have h_cos_theta : (inner a b / (∥a∥ * ∥b∥)) = -4/5,
  { 
    -- This step shows the calculation of cos theta.
    sorry, 
  },
  -- Projection calculation using the previous cos theta result.
  calc 
    5 * (inner a b / (∥a∥ * ∥b∥)) = 5 * (-4 / 5) : by rw [h_cos_theta]
                                 ... = -4 : by norm_num
  -- The final answer is -4.
  sorry

end vector_projection_eq_minus_four_l19_19995


namespace initial_quantity_of_milk_l19_19815

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk_l19_19815


namespace determinant_zero_implies_values_l19_19964

-- Define real numbers a, b, c and scalar λ
variables (a b c λ : ℝ)

-- Define the matrix
def M := matrix (fin 3) (fin 3) ℝ :=
  ![![a + λ, b, c],
    ![b, c + λ, a],
    ![c, a, b + λ]]

-- State the problem conditions and result
theorem determinant_zero_implies_values (ha : a + b + c + 3 * λ = 0 ∨ a = b ∧ b = c) :
  (∃ (x : ℝ), x = (a / (b + c) + b / (a + c) + c / (a + b)) ∧ (x = -1 ∨ x = 3 / 2)) :=
by {
  -- This proof is omitted for now
  sorry
}

end determinant_zero_implies_values_l19_19964


namespace geometric_sequence_sixth_term_l19_19756

theorem geometric_sequence_sixth_term (a r : ℕ) (h₁ : a = 8) (h₂ : a * r ^ 3 = 64) : a * r ^ 5 = 256 :=
by
  -- to be filled (proof skipped)
  sorry

end geometric_sequence_sixth_term_l19_19756


namespace middle_pile_cards_l19_19829

theorem middle_pile_cards (x : Nat) (h : x ≥ 2) : 
    let left := x - 2
    let middle := x + 2
    let right := x
    let middle_after_step3 := middle + 1
    let final_middle := middle_after_step3 - left
    final_middle = 5 := 
by
  sorry

end middle_pile_cards_l19_19829


namespace sum_of_first_20_triangular_numbers_l19_19500

theorem sum_of_first_20_triangular_numbers :
  ∑ n in Finset.range 20, (n + 1) * (n + 2) / 2 = 1540 :=
by
  sorry

end sum_of_first_20_triangular_numbers_l19_19500


namespace midpoint_one_sixth_one_ninth_l19_19546

theorem midpoint_one_sixth_one_ninth : (1 / 6 + 1 / 9) / 2 = 5 / 36 := by
  sorry

end midpoint_one_sixth_one_ninth_l19_19546


namespace find_horizontal_length_l19_19402

def rectangle_horizontal_length (P : ℕ) (d : ℕ) (h : ℕ) (v : ℕ) : Prop :=
  P = 2 * v + 2 * h ∧ h = v + d

theorem find_horizontal_length :
  ∀ (P d : ℕ), P = 54 → d = 3 → ∃ h v, rectangle_horizontal_length P d h v ∧ h = 15 := 
by
  intros P d h v
  intro h₃
  intro h
  use v sorry

end find_horizontal_length_l19_19402


namespace number_of_mappings_l19_19961

def A : Set Int := {-1, 0}
def B : Set Int := {1, 2}
def mappings (A B : Type) := A → B

theorem number_of_mappings : ∃ num_mappings, num_mappings = 4 ∧ ∀ (f : mappings A B), true :=
by
  use 4
  split
  exact rfl
  intros f
  trivial
  sorry

end number_of_mappings_l19_19961


namespace remaining_work_nonpositive_l19_19466

theorem remaining_work_nonpositive 
  (num_welders : ℕ) 
  (work_A_per_day : ℕ -> ℝ) 
  (initial_work_A : ℝ) (initial_workers_on_A : ℕ) 
  (order_A_man_days : ℝ) 
  (workers_reassigned: ℕ) : 
  (num_welders = 24) 
  → (work_A_per_day = λ n, if n < initial_workers_on_A then 1.5 else 0) 
  → (initial_workers_on_A = num_welders) 
  → (order_A_man_days = 30) 
  → (initial_work_A = (initial_workers_on_A * work_A_per_day 0)) 
  → (workers_reassigned = 10) 
  → (initial_work_A > order_A_man_days) 
  → ((order_A_man_days - initial_work_A) ≤ 0) :=
begin
  intros num_welders_eq work_A_per_day_eq initial_workers_on_A_eq
         order_A_man_days_eq initial_work_A_eq workers_reassigned_eq
         initial_work_A_gt,

  -- This is the part where we would perform our proof, which we'll skip:
  sorry
end

end remaining_work_nonpositive_l19_19466


namespace part_one_part_two_l19_19600

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x - sqrt 3 * sin x)

theorem part_one : f (π / 6) = 0 :=
by
  unfold f
  have := sin_pi
  sorry

theorem part_two :
  ∀ x (hx : 0 ≤ x ∧ x ≤ π / 2), -sqrt 3 ≤ f x ∧ f x ≤ 1 - sqrt 3 / 2 :=
by
  unfold f
  sorry

end part_one_part_two_l19_19600


namespace max_weight_automobile_l19_19845

theorem max_weight_automobile (W : ℕ) (ferry_capacity_tons : ℕ) 
  (pounds_per_ton : ℕ) (max_automobiles_float : ℚ) :
  ferry_capacity_tons = 50 →
  pounds_per_ton = 2000 →
  max_automobiles_float = 62.5 →
  W = (ferry_capacity_tons * pounds_per_ton) / max_automobiles_float :=
begin
  sorry
end

end max_weight_automobile_l19_19845


namespace train_speed_is_54_kmph_l19_19106

-- Definitions of the conditions
def train_length : ℝ := 165
def bridge_length : ℝ := 625
def time_to_cross : ℝ := 52.66245367037304

-- The total distance covered by the train in crossing the bridge
def total_distance : ℝ := train_length + bridge_length

-- Speed of the train in meters per second
def speed_m_per_s : ℝ := total_distance / time_to_cross

-- Speed of the train in kilometers per hour
def speed_km_per_h : ℝ := speed_m_per_s * 3.6

-- Theorem statement
theorem train_speed_is_54_kmph : speed_km_per_h = 54.000 :=
sorry

end train_speed_is_54_kmph_l19_19106


namespace sin_2alpha_pos_of_tan_alpha_pos_l19_19215

theorem sin_2alpha_pos_of_tan_alpha_pos (α : Real) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_of_tan_alpha_pos_l19_19215


namespace minimum_selling_price_l19_19453

theorem minimum_selling_price (total_cost : ℝ) (total_fruit : ℝ) (spoilage : ℝ) (min_price : ℝ) :
  total_cost = 760 ∧ total_fruit = 80 ∧ spoilage = 0.05 ∧ min_price = 10 → 
  ∀ price : ℝ, (price * total_fruit * (1 - spoilage) >= total_cost) → price >= min_price :=
by
  intros h price hp
  rcases h with ⟨hc, hf, hs, hm⟩
  sorry

end minimum_selling_price_l19_19453


namespace special_sum_correct_l19_19505

noncomputable def special_sum : ℝ :=
  ∑ n in finset.range 98, (λ n, 1 / (n + 3 : ℝ) * (real.sqrt ((n + 3) ^ 2 - 9) + 9 * real.sqrt (n + 3)))

theorem special_sum_correct : special_sum = (real.sqrt 103 - 1) / real.sqrt 103 :=
by
  sorry

end special_sum_correct_l19_19505


namespace positive_difference_perimeters_l19_19375

def shape1_length := 7
def shape1_height := 3

def shape2_total_length := 6
def shape2_total_height := 2
def shape2_internal_divisions := 5
def internal_division_contribution := 2

def perimeter_shape1 := 2 * (shape1_length + shape1_height)
def perimeter_shape2 := 2 * (shape2_total_length + shape2_total_height) + shape2_internal_divisions * internal_division_contribution

theorem positive_difference_perimeters :
  abs (perimeter_shape1 - perimeter_shape2) = 6 :=
by
  sorry

end positive_difference_perimeters_l19_19375


namespace irreducible_polynomial_condition_l19_19522

noncomputable def polynomial_is_irreducible (a b c : ℤ) : Prop :=
  irreducible (Polynomial.X * (Polynomial.X - a) * (Polynomial.X - b) * (Polynomial.X - c) + 1)

theorem irreducible_polynomial_condition {a b c : ℤ} 
  (h_cond : 0 < |c| ∧ |c| < |b| ∧ |b| < |a|)
  (h_diff : (a, b, c) ≠ (1, 2, 3) ∧ (a, b, c) ≠ (-1, -2, -3)) : 
  polynomial_is_irreducible a b c :=
by 
  sorry

end irreducible_polynomial_condition_l19_19522


namespace same_letter_probability_l19_19827

theorem same_letter_probability : 
  let johnson_letters := {('J', 1), ('O', 1), ('H', 1), ('N', 2), ('S', 1)} in
  let jones_letters := {('J', 1), ('O', 1), ('N', 1), ('E', 1), ('S', 1)} in
  let total_johnson := 7 in
  let total_jones := 5 in
  let prob_J := ((johnson_letters.lookup 'J') * (jones_letters.lookup 'J')) in
  let prob_O := ((johnson_letters.lookup 'O') * (jones_letters.lookup 'O')) in
  let prob_N := ((johnson_letters.lookup 'N') * (jones_letters.lookup 'N')) in
  let prob_S := ((johnson_letters.lookup 'S') * (jones_letters.lookup 'S')) in
  let total_prob := (prob_J + prob_O + prob_N + prob_S) in
  (total_prob / (total_johnson * total_jones) = (1 / 7)) :=
by
  -- Here, we will use the given frequencies to prove the probability
  sorry

end same_letter_probability_l19_19827


namespace ratio_of_ap_l19_19231

theorem ratio_of_ap (a d : ℕ) (h : 30 * a + 435 * d = 3 * (15 * a + 105 * d)) : a = 8 * d :=
by
  sorry

end ratio_of_ap_l19_19231


namespace variance_of_yield_l19_19084

/-- Given a data set representing annual average yields,
    prove that the variance of this data set is approximately 171. --/
theorem variance_of_yield {yields : List ℝ} 
  (h_yields : yields = [450, 430, 460, 440, 450, 440, 470, 460]) :
  let mean := (yields.sum / yields.length : ℝ)
  let squared_diffs := (yields.map (fun x => (x - mean)^2))
  let variance := (squared_diffs.sum / (yields.length - 1 : ℝ))
  abs (variance - 171) < 1 :=
by
  sorry

end variance_of_yield_l19_19084


namespace perp_bisector_eq_l19_19341

/-- The circles x^2+y^2=4 and x^2+y^2-4x+6y=0 intersect at points A and B. 
Find the equation of the perpendicular bisector of line segment AB. -/

theorem perp_bisector_eq : 
  let C1 := (0, 0)
  let C2 := (2, -3)
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = 0 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
by
  sorry

end perp_bisector_eq_l19_19341


namespace bank_account_balance_l19_19530

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l19_19530


namespace min_tangent_length_l19_19630

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end min_tangent_length_l19_19630


namespace grid_with_value_exists_possible_values_smallest_possible_value_l19_19142

open Nat

def isGridValuesP (P : ℕ) (a b c d e f g h i : ℕ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = P) ∧ (d * e * f = P) ∧
  (g * h * i = P) ∧ (a * d * g = P) ∧
  (b * e * h = P) ∧ (c * f * i = P)

theorem grid_with_value_exists (P : ℕ) :
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem possible_values (P : ℕ) :
  P ∈ [1992, 1995] ↔ 
  ∃ a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i :=
sorry

theorem smallest_possible_value : 
  ∃ P a b c d e f g h i : ℕ, isGridValuesP P a b c d e f g h i ∧ 
  ∀ Q, (∃ w x y z u v s t q : ℕ, isGridValuesP Q w x y z u v s t q) → Q ≥ 120 :=
sorry

end grid_with_value_exists_possible_values_smallest_possible_value_l19_19142


namespace total_pages_in_book_l19_19158

theorem total_pages_in_book (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 22) (h2 : days = 569) : total_pages = 12518 :=
by
  sorry

end total_pages_in_book_l19_19158


namespace arrangeable_sequence_l19_19038

theorem arrangeable_sequence (n : Fin 2017 → ℤ) :
  (∀ i : Fin 2017, ∃ (perm : Fin 5 → Fin 5),
    let a := n ((i + perm 0) % 2017)
    let b := n ((i + perm 1) % 2017)
    let c := n ((i + perm 2) % 2017)
    let d := n ((i + perm 3) % 2017)
    let e := n ((i + perm 4) % 2017)
    a - b + c - d + e = 29) →
  (∀ i : Fin 2017, n i = 29) :=
by
  sorry

end arrangeable_sequence_l19_19038


namespace area_of_circumcircle_l19_19234

theorem area_of_circumcircle 
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  let a := Real.sin α
  let b := Real.sin β
  let c := Real.sin (α + β)
  area_of_circumcircle (Triangle.mk a b c) = π / 4 :=
by
  sorry

end area_of_circumcircle_l19_19234


namespace round_24_6374_to_nearest_hundredth_l19_19729

noncomputable def round_to_hundredths (x : ℝ) : ℝ :=
  let scaled := x * 100 in
  if scaled - scaled.floor ≥ 0.5 then (scaled.floor + 1) / 100 else scaled.floor / 100

theorem round_24_6374_to_nearest_hundredth :
  round_to_hundredths 24.6374 = 24.64 :=
by
  sorry

end round_24_6374_to_nearest_hundredth_l19_19729


namespace min_expression_value_l19_19976

variable {a : ℕ → ℝ}
variable (m n : ℕ)
variable (q : ℝ)

axiom pos_seq (n : ℕ) : a n > 0
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom seq_condition : a 7 = a 6 + 2 * a 5
axiom exists_terms :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1)

theorem min_expression_value : 
  (∃m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1) ∧ 
  a 7 = a 6 + 2 * a 5 ∧ 
  (∀ n, a n > 0 ∧ a (n + 1) = q * a n)) → 
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_expression_value_l19_19976


namespace exists_triangle_with_area_at_least_three_l19_19714

theorem exists_triangle_with_area_at_least_three 
    (A B C X Y : Point)
    (h : ∀ P Q R ∈ {A, B, C, X, Y}, area_triangle P Q R ≥ 2) :
    ∃ P Q R ∈ {A, B, C, X, Y}, area_triangle P Q R ≥ 3 :=
by
  sorry

end exists_triangle_with_area_at_least_three_l19_19714


namespace tobias_distance_swum_l19_19028

def swimming_pool_duration : Nat := 3 * 60 -- Tobias at the pool for 3 hours in minutes.

def swimming_duration : Nat := 5  -- Time taken to swim 100 meters in minutes.

def pause_duration : Nat := 5  -- Pause duration after every 25 minutes of swimming.

def swim_interval : Nat := 25  -- Swimming interval before each pause in minutes.

def total_pause_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  (total_minutes / (interval + pause)) * pause

def total_swimming_time (total_minutes: Nat) (interval: Nat) (pause: Nat) : Nat :=
  total_minutes - total_pause_time(total_minutes, interval, pause)

def distance_swum (swimming_minutes: Nat) (swim_duration: Nat) : Nat :=
  (swimming_minutes / swim_duration) * 100

theorem tobias_distance_swum :
  distance_swum (total_swimming_time swimming_pool_duration swim_interval pause_duration) swimming_duration = 3000 :=
by
  -- perform the calculation steps indicated in the solution
  sorry

end tobias_distance_swum_l19_19028


namespace polynomial_identity_l19_19738

theorem polynomial_identity (x : ℝ) (h1 : x ^ 2019 - 3 * x + 2 = 0) (h2 : x ≠ 1) : 
  x ^ 2018 + x ^ 2017 + ∑ i in finset.range 2017, x ^ i + 1 = 3 :=
by
  -- solution proof
  sorry

end polynomial_identity_l19_19738


namespace no_prime_pairs_sum_53_l19_19663

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l19_19663


namespace sum_six_digits_correct_l19_19333

noncomputable def sum_of_six_digits : ℕ :=
let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  ∃ (a b c d f g : ℕ), 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    d ≠ f ∧ f ≠ g ∧ g ≠ d ∧ 
    a ≠ d ∧ a ≠ f ∧ a ≠ g ∧ b ≠ d ∧ b ≠ f ∧ b ≠ g ∧ c ≠ d ∧ c ≠ f ∧ c ≠ g ∧
    d ∈ S ∧ f ∈ S ∧ g ∈ S ∧
    a, b, c ∈ S ∧
    a + b + c = 24 ∧ 
    d + b + f + g = 14 ∧
    b ∈ S ∧
    {a, b, c, d, f, g}.card = 6 ∧
    a + b + c + d + f + g = 30

theorem sum_six_digits_correct : sum_of_six_digits = 30 :=
sorry

end sum_six_digits_correct_l19_19333


namespace min_time_to_return_to_A_l19_19858

-- Definitions based on the given conditions
def track_circumference : ℕ := 400
def walking_speed_m_per_min : ℕ := 100
def walking_pattern : list (ℤ × ℕ) := [(1, 1), (-1, 3), (1, 5)]

-- The problem statement to prove
theorem min_time_to_return_to_A (circuit_length : ℕ) (speed : ℕ) (pattern : list (ℤ × ℕ)) : 
  circuit_length = 400 → 
  speed = 100 →
  pattern = [(1, 1), (-1, 3), (1, 5)] →
  ∃ t : ℕ, t = 1 :=
by
  sorry

end min_time_to_return_to_A_l19_19858


namespace how_many_did_not_play_l19_19865

def initial_players : ℕ := 40
def first_half_starters : ℕ := 11
def first_half_substitutions : ℕ := 4
def second_half_extra_substitutions : ℕ := (first_half_substitutions * 3) / 4 -- 75% more substitutions
def injury_substitution : ℕ := 1
def total_second_half_substitutions : ℕ := first_half_substitutions + second_half_extra_substitutions + injury_substitution
def total_players_played : ℕ := first_half_starters + first_half_substitutions + total_second_half_substitutions
def players_did_not_play : ℕ := initial_players - total_players_played

theorem how_many_did_not_play : players_did_not_play = 17 := by
  sorry

end how_many_did_not_play_l19_19865


namespace remainder_of_3045_div_32_l19_19385

theorem remainder_of_3045_div_32 : 3045 % 32 = 5 :=
by sorry

end remainder_of_3045_div_32_l19_19385


namespace megan_numbers_difference_l19_19302

theorem megan_numbers_difference 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_mean3 : (x1 + x2 + x3) / 3 = -3)
  (h_mean4 : (x1 + x2 + x3 + x4) / 4 = 4)
  (h_mean5 : (x1 + x2 + x3 + x4 + x5) / 5 = -5) :
  x4 - x5 = 66 :=
by
  sorry

end megan_numbers_difference_l19_19302


namespace count_unique_sums_l19_19251

def valid_digits : List ℕ := [1, 3, 5, 7]

theorem count_unique_sums :
  ∃ (A : set ℕ), (∀ (a b c d : ℕ), a ∈ valid_digits → b ∈ valid_digits → c ∈ valid_digits → d ∈ valid_digits →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * b + c * d) ∈ A) ∧ A.card = 3 :=
by
  sorry

end count_unique_sums_l19_19251


namespace total_samples_l19_19339

theorem total_samples (total_counties : ℕ) (samples_jiujiang : ℕ) (counties_jiujiang : ℕ) (total_prefectures : ℕ)
  (h1 : total_counties = 20)
  (h2 : samples_jiujiang = 2)
  (h3 : counties_jiujiang = 8)
  (h4 : total_prefectures = 9) :
  let sampling_fraction := samples_jiujiang * total_counties / counties_jiujiang in
  total_samples = 5 := 
by {
  sorry
}

end total_samples_l19_19339


namespace hexagon_area_sum_l19_19849

theorem hexagon_area_sum (p q : ℕ) (h : (∃ (hex : Hexagon), hex.sides = 12 ∧ hex.side_length = 3 ∧ hex.area = sqrt p + sqrt q)) : p + q = 810 :=
sorry

end hexagon_area_sum_l19_19849


namespace no_prime_pairs_sum_53_l19_19665

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l19_19665


namespace time_taken_to_cross_l19_19064

def length_train_A : ℝ := 125
def length_train_B : ℝ := 150

def speed_train_A_kmh : ℝ := 54
def speed_train_B_kmh : ℝ := 36
def speed_convert_kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

def speed_train_A : ℝ := speed_convert_kmph_to_mps speed_train_A_kmh
def speed_train_B : ℝ := speed_convert_kmph_to_mps speed_train_B_kmh

def relative_speed : ℝ := speed_train_A + speed_train_B
def total_distance : ℝ := length_train_A + length_train_B

def time_to_cross : ℝ := total_distance / relative_speed

theorem time_taken_to_cross : time_to_cross = 11 := by
  sorry

end time_taken_to_cross_l19_19064


namespace triangle_abo_perimeter_12_l19_19344

-- Definitions from the problem
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), (O = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧ (O = ((B.1 + D.1) / 2, (B.2 + D.2) / 2))

def intersect_at (A B C D O : ℝ × ℝ) : Prop :=
  A.1 + C.1 = B.1 + D.1 ∧ A.2 + C.2 = B.2 + D.2 ∧ O = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

def magnitude (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def angle (A O D : ℝ × ℝ) : ℝ :=
  real.arccos (((A.1 - O.1) * (D.1 - O.1) + (A.2 - O.2) * (D.2 - O.2)) /
  (magnitude A O * magnitude D O))

-- Given conditions
variables (A B C D O : ℝ × ℝ)
variable (h_rect : is_rectangle A B C D)
variable (h_intersect : intersect_at A B C D O)
variable (h_angle : angle A O D = 2 * π / 3)
variable (h_ac : magnitude A C = 8)

theorem triangle_abo_perimeter_12 :
  (magnitude A O) + (magnitude B O) + (magnitude A B) = 12 :=
sorry

end triangle_abo_perimeter_12_l19_19344


namespace product_bounds_1999_l19_19733

noncomputable def seq_product : ℕ → ℝ
| 0       := 1/2
| (n + 1) := (2 * (n + 1) - 1)/(2 * (n + 1)) * seq_product n

theorem product_bounds_1999 :
  let k := seq_product 1998 in
  1/1999 < k ∧ k < 1/44 :=
by { 
  let k := seq_product 1998, 
  have h₁ : k = ∏ i in finset.range 1998, (2*i+1)/(2*(i+1)) := sorry,
  have h₂ : 1/1999 < k := sorry,
  have h₃ : k < 1/44 := sorry,
  exact ⟨h₂, h₃⟩
}

end product_bounds_1999_l19_19733


namespace cost_price_of_bicycle_l19_19407

variables {CP_A SP_AB SP_BC : ℝ}

theorem cost_price_of_bicycle (h1 : SP_AB = CP_A * 1.2)
                             (h2 : SP_BC = SP_AB * 1.25)
                             (h3 : SP_BC = 225) :
                             CP_A = 150 :=
by sorry

end cost_price_of_bicycle_l19_19407


namespace difference_divisible_l19_19720

theorem difference_divisible {a_k a_k_minus1 a_k_plus1 m n : ℤ}
  (h1 : a_k = x * m^n)
  (h2 : a_k_plus1 = a_k_minus1 + a_k)
  (h3 : a_k_plus1^m - a_k_minus1^m ≡ 0 [MOD m^(n+1)]) :
  a_k_plus1^m - a_k_minus1^m ≡ 0 [MOD m^(n+1)] := by
  sorry

end difference_divisible_l19_19720


namespace polynomial_divisibility_l19_19619

theorem polynomial_divisibility (p : ℝ) :
  (4 * (2 : ℝ)^3 - 12 * (2 : ℝ)^2 + p * (2 : ℝ) - 16 = 0) →
  (4 * (4 : ℝ)^3 - 12 * (4 : ℝ)^2 + p * (4 : ℝ) - 16 = 0) :=
by
  intro h
  have hp : p = 16 := by
    have : 4 * (2 : ℝ)^3 - 12 * (2 : ℝ)^2 + p * (2 : ℝ) - 16 = -32 + 2 * p
    simp
    linarith
  rw [hp]
  simp
  sorry

end polynomial_divisibility_l19_19619


namespace range_of_t_l19_19202

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

noncomputable def g (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2

theorem range_of_t (t : ℝ) : 
  (∃ a : ℝ, f' a * (2 - a) = t + 6) ↔ -6 < t ∧ t < 2 :=
by
  sorry

end range_of_t_l19_19202


namespace perpendicular_lines_l19_19229

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x - a * y - 1 = 0 → x + 2 * y = 0) →
  (a = 1) :=
by
  sorry

end perpendicular_lines_l19_19229


namespace ferris_wheel_travel_time_l19_19833

theorem ferris_wheel_travel_time :
  ∀ (r t : ℝ), r = 30 ∧ t = 90 → ∃ (time : ℝ), 30 * cos ((2 * real.pi / t) * time) + 30 = 15 ∧ time = 30 :=
by
  intro r t h
  obtain ⟨hr, ht⟩ := h
  use 30
  rw [hr, ht]
  split
  · sorry
  · rfl

end ferris_wheel_travel_time_l19_19833


namespace center_of_circle_sum_l19_19552

open Real

theorem center_of_circle_sum (x y : ℝ) (h k : ℝ) :
  (x - h)^2 + (y - k)^2 = 2 → (h = 3) → (k = 4) → h + k = 7 :=
by
  intro h_eq k_eq
  sorry

end center_of_circle_sum_l19_19552


namespace sin_690_degree_l19_19332

theorem sin_690_degree : Real.sin (690 * Real.pi / 180) = -1/2 :=
by
  sorry

end sin_690_degree_l19_19332


namespace polynomial_roots_arithmetic_progression_not_all_real_l19_19925

theorem polynomial_roots_arithmetic_progression_not_all_real :
  ∀ (a : ℝ), (∃ r d : ℂ, r - d ≠ r ∧ r ≠ r + d ∧ r - d + r + (r + d) = 9 ∧ (r - d) * r + (r - d) * (r + d) + r * (r + d) = 33 ∧ d ≠ 0) →
  a = -45 :=
by
  sorry

end polynomial_roots_arithmetic_progression_not_all_real_l19_19925


namespace max_y_on_graph_l19_19932

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l19_19932


namespace problem_statement_l19_19178

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Definition of the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^n

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℕ := a n + b n

-- Sum of first n terms of the sequence {c_n}
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 2^(n + 1) - 2

-- Prove the problem statement
theorem problem_statement :
  (a 1 + a 2 = 3) ∧
  (a 4 - a 3 = 1) ∧
  (b 2 = a 4) ∧
  (b 3 = a 8) ∧
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2) :=
by {
  sorry -- Proof goes here
}

end problem_statement_l19_19178


namespace trip_times_comparison_l19_19682

theorem trip_times_comparison (v : ℝ) (h₁ : v > 0) :
  let t1 := 40 / v
  let t3 := 480 / (4 * v)
  t3 = 3 * t1 :=
by
  let t1 := 40 / v
  let t3 := 480 / (4 * v)
  have h3 : t3 = 120 / v := rfl
  have h1 : t1 = 40 / v := rfl
  calc
    t3 = 120 / v  : h3
    ... = 3 * (40 / v) : by ring
    ... = 3 * t1 : by rw h1

end trip_times_comparison_l19_19682


namespace function_above_x_axis_l19_19601

noncomputable def quadratic_function (a x : ℝ) := (a^2 - 3 * a + 2) * x^2 + (a - 1) * x + 2

theorem function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x > 0) ↔ (a > 15 / 7 ∨ a ≤ 1) :=
by {
  sorry
}

end function_above_x_axis_l19_19601


namespace min_f_value_l19_19954

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / x + 1 / (2 * x + 1 / x)

theorem min_f_value : (∃ x : ℝ, 0 < x ∧ ∀ y : ℝ, 0 < y → f y ≥ f x) ∧ f (1 / sqrt 2) = 5 * sqrt 2 / 3 :=
by
  refine ⟨⟨1 / sqrt 2, _, _⟩, _⟩
  { exact div_pos zero_lt_one (sqrt_pos.2 zero_lt_two) }
  { intro y hy
    sorry }
  { sorry }

end min_f_value_l19_19954


namespace frustum_volume_computation_l19_19875

def volume_of_frustum (a h b k : ℝ) : ℝ :=
  let V_original := (1/3) * (a^2) * h
  let V_smaller := (1/3) * (b^2) * k
  V_original - V_smaller

theorem frustum_volume_computation :
  volume_of_frustum 15 10 9 6 = 588 := 
by
  sorry

end frustum_volume_computation_l19_19875


namespace find_m_l19_19194

theorem find_m (m : ℝ) :
  (∀ x : ℝ, ((m - 2) * x^2 + 3 * x - m^2 - m + 6 = 0) → x = 0) → m ≠ 2 → m = -3 :=
by
  assume h h0,
  sorry

end find_m_l19_19194


namespace value_of_expression_l19_19884

variables {x1 x2 x3 x4 x5 x6 : ℝ}

theorem value_of_expression
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 = 1)
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 = 14)
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 = 135) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 = 832 :=
by
  sorry

end value_of_expression_l19_19884


namespace least_multiple_of_36_with_product_of_digits_multiple_of_36_l19_19802

def product_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).prod

theorem least_multiple_of_36_with_product_of_digits_multiple_of_36 :
  ∀ n : ℕ, n % 36 = 0 → (∀ k : ℕ, k % 36 = 0 → product_of_digits k % 36 ≠ 0 → k ≥ 1296) ∧ product_of_digits 1296 % 36 = 0 :=
  sorry

end least_multiple_of_36_with_product_of_digits_multiple_of_36_l19_19802


namespace find_positive_integer_n_l19_19548

theorem find_positive_integer_n :
  ∃ n : ℕ, 0 < n ∧ ((n + 1)! + (n + 3)!) = n! * 1320 ∧ n = 11 :=
begin
  sorry
end

end find_positive_integer_n_l19_19548


namespace decryption_correct_l19_19776

theorem decryption_correct (a b : ℤ) (h1 : a - 2 * b = 1) (h2 : 2 * a + b = 7) : a = 3 ∧ b = 1 :=
by
  sorry

end decryption_correct_l19_19776


namespace max_y_on_graph_l19_19931

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l19_19931


namespace percentage_primes_divisible_by_2_l19_19052

theorem percentage_primes_divisible_by_2 :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  (100 * primes.filter (fun n => n % 2 = 0).card / primes.card) = 12.5 :=
by
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  have h1 : primes.filter (fun n => n % 2 = 0).card = 1 := sorry
  have h2 : primes.card = 8 := sorry
  have h3 : (100 * 1 / 8 : ℝ) = 12.5 := by norm_num
  exact h3

end percentage_primes_divisible_by_2_l19_19052


namespace find_theta_l19_19378

def rectangle : Type := sorry
def angle (α : ℝ) : Prop := 0 ≤ α ∧ α < 180

-- Given conditions in the problem
variables {α β γ δ θ : ℝ}

axiom angle_10 : angle 10
axiom angle_14 : angle 14
axiom angle_33 : angle 33
axiom angle_26 : angle 26

axiom zig_zag_angles (a b c d e f : ℝ) :
  a = 26 ∧ f = 10 ∧
  26 + b = 33 ∧ b = 7 ∧
  e + 10 = 14 ∧ e = 4 ∧
  c = b ∧ d = e ∧
  θ = c + d

theorem find_theta : θ = 11 :=
sorry

end find_theta_l19_19378


namespace verify_first_rope_length_l19_19779

def length_first_rope : ℝ :=
  let rope1_len := 20
  let rope2_len := 2
  let rope3_len := 2
  let rope4_len := 2
  let rope5_len := 7
  let knots := 4
  let knot_loss := 1.2
  let total_len := 35
  rope1_len

theorem verify_first_rope_length : length_first_rope = 20 := by
  sorry

end verify_first_rope_length_l19_19779


namespace percentage_attending_college_l19_19877

variables (boys girls : ℕ)
variables (boys_attending girls_total girls_not_attending total_attending total_students : ℝ)

-- Conditions
def num_boys := boys = 160
def num_girls := girls = 200
def percent_boys_attending := boys_attending = 0.75 * boys
def percent_girls_not_attending := girls_not_attending = 0.40 * girls

def total_girls := girls_total = girls
def total_girls_attending := total_girls - girls_not_attending
def total_attending_students := total_attending = boys_attending + total_girls_attending
def total_students_sum := total_students = boys + girls

-- Proof statement
theorem percentage_attending_college
    (num_boys : num_boys)
    (num_girls : num_girls)
    (percent_boys_attending : percent_boys_attending)
    (percent_girls_not_attending : percent_girls_not_attending)
    (total_attending_students : total_attending_students)
    (total_students_sum : total_students_sum) :
    (total_attending / total_students) * 100 = 66.67 := 
sorry

end percentage_attending_college_l19_19877


namespace prove_disjunction_l19_19991

def proposition_p := "A line has an infinite number of direction vectors"
def proposition_q := "A plane has only one normal vector"

theorem prove_disjunction :
  (let p := true in
   let q := false in
   (¬p) ∨ (¬q)) :=
by
  let p := true
  let q := false
  show (¬p) ∨ (¬q)
  sorry

end prove_disjunction_l19_19991


namespace cameron_total_questions_answered_l19_19894

def questions_per_tourist : ℕ := 2
def group1_size : ℕ := 6
def group2_size : ℕ := 11
def group3_size_regular : ℕ := 7
def group3_inquisitive_size : ℕ := 1
def group4_size : ℕ := 7

theorem cameron_total_questions_answered :
  let group1_questions := questions_per_tourist * group1_size in
  let group2_questions := questions_per_tourist * group2_size in
  let group3_regular_questions := questions_per_tourist * group3_size_regular in
  let group3_inquisitive_questions := group3_inquisitive_size * (questions_per_tourist * 3) in
  let group3_questions := group3_regular_questions + group3_inquisitive_questions in
  let group4_questions := questions_per_tourist * group4_size in
  group1_questions + group2_questions + group3_questions + group4_questions = 68 :=
by
  sorry

end cameron_total_questions_answered_l19_19894


namespace hexagon_area_formula_l19_19687

theorem hexagon_area_formula (x : ℝ) (b c : ℝ) (h₀ : 0 < x) (h₁ : x < 0.5) :
  (∀ (A B C A₁ A₂ : Type) (hA₁ : A₁ = x) (hA₂ : A₂ = x) (T_A : Set A) (T_B : Set B) (T_C : Set C),
   area (hexagon T_A T_B T_C) = (8 * x^2 - b * x + c) / ((2 - x) * (x + 1)) * (sqrt 3 / 4)) →
  b = 8 ∧ c = 2 := 
by
  sorry

end hexagon_area_formula_l19_19687


namespace sequence_bound_l19_19170

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end sequence_bound_l19_19170


namespace angle_relationship_l19_19477

-- define points and quadrilateral
variables {A B C D X Y : Type}
variables (ABCD_convex: ∀ (p : A) (q : B) (r : C) (s : D), convex_quad p q r s)

-- define perpendicular bisectors
variables (AB_perp : ∀ {p q : A}, ⟨⟨p ≠ q⟩ , ∃ m, m * xy p q = -1⟩)
variables (CD_perp : ∀ {p q : D}, ⟨⟨p ≠ q⟩ , ∃ m, m * xy p q = -1⟩)

-- Y is the intersection of the perpendicular bisectors of AB and CD
variables (Y_inter : intersection AB_perp CD_perp = Y)

-- Define angle properties of point X
variables (angle_ADX_BCX : ∀ {p q r s : X}, p ∠ q r = s ∠ r q)
variables (angle_DAX_CBX : ∀ {p q r s : X}, p ∠ q r = s ∠ r q)

theorem angle_relationship (A B C D X Y : Type) 
    (ABCD_convex : ∀ (p : A) (q : B) (r : C) (s : D), convex_quad p q r s)
    (AB_perp : ∀ {p q : A}, ⟨⟨p ≠ q⟩ , ∃ m, m * xy p q = -1⟩)
    (CD_perp : ∀ {p q : D}, ⟨⟨p ≠ q⟩ , ∃ m, m * xy p q = -1⟩)
    (Y_inter : intersection AB_perp CD_perp = Y)
    (angle_ADX_BCX : ∀ {p q r s : X}, p ∠ q r = s ∠ r q)
    (angle_DAX_CBX : ∀ {p q r s : X}, p ∠ q r = s ∠ r q)
    : (X ∠ Y B = 2 * (A ∠ D X)) :=
sorry

end angle_relationship_l19_19477


namespace focus_coordinates_l19_19115

noncomputable def semi_major_axis_length := 3.5
noncomputable def semi_minor_axis_length := 2.5
noncomputable def center := (3.5, 0)
noncomputable def distance_between_foci := Real.sqrt (semi_major_axis_length ^ 2 - semi_minor_axis_length ^ 2) / 2
noncomputable def focus_with_greater_x := (center.1 + distance_between_foci, 0)

theorem focus_coordinates :
  focus_with_greater_x = (3.5 + Real.sqrt 6 / 2, 0) :=
sorry

end focus_coordinates_l19_19115


namespace irrational_sqrt_5_among_options_l19_19396

theorem irrational_sqrt_5_among_options :
    (∀ x, x ∈ ({0.618, (22 : ℚ) / 7, (∛ (-27 : ℤ) : ℚ), real.sqrt 5} : set ℝ) →
     x = real.sqrt 5 → irrational x) :=
begin
  sorry
end

end irrational_sqrt_5_among_options_l19_19396


namespace decimal_to_fraction_l19_19789

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l19_19789


namespace range_at_least_20_l19_19103

def set_x (x : Set ℤ) :=
  ∃ (n : ℕ) (median : ℤ) (max_elem : ℤ),
    x.card = 10 ∧
    (median ∈ x ∨ 
    (∃ a b, a ∈ x ∧ b ∈ x ∧ (a + b) / 2 = 30 ∧ 
      x = (x.erase a).erase b ∧ 
      (insert ((a + b) / 2) (insert (a + b / 2) x) = x)) ) ∧
    max_elem = 50

theorem range_at_least_20 (x : Set ℤ) (h : set_x x) : 
  (∃ ymin ymax, ymin ∈ x ∧ ymax ∈ x ∧ ymax = 50 ∧ 50 - 30 = 20 ∧  ∀ z ∈ x, z ≥ ymin) := 
sorry

end range_at_least_20_l19_19103


namespace calc_expression_l19_19888

theorem calc_expression :
  15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 :=
by
  sorry

end calc_expression_l19_19888
