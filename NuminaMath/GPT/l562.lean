import Mathlib

namespace negation_of_exists_proposition_l562_562251

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 * x > 2) ↔ (∀ x : ℝ, x^2 - 2 * x ≤ 2) :=
by
  sorry

end negation_of_exists_proposition_l562_562251


namespace variance_averaged_function_l562_562734

noncomputable def varphi : ℝ → ℝ := sorry
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x, 1 / (b - a)

theorem variance_averaged_function (a b : ℝ) (h : a < b) :
  let I := ∫ x in a..b, varphi x in
  let I1_star := (b - a) * (∑ x_i in range n, varphi x_i) / n in
  let σ2 := (b - a) * ∫ x in a..b, (varphi x)^2 - (∫ x in a..b, varphi x * (f a b) x)^2 in
  σ2 = (b - a) * ∫ x in a..b, (varphi x)^2 - (∫ x in a..b, varphi x * (f a b) x)^2 :=
begin
  sorry
end

end variance_averaged_function_l562_562734


namespace intersection_products_l562_562538

noncomputable def line_l_parametric (t : ℝ) : ℝ × ℝ := 
  (3 + (real.sqrt 2) / 2 * t, 4 + (real.sqrt 2) / 2 * t)

def circle_C_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y = 0

theorem intersection_products (M A B : ℝ × ℝ) 
  (h_M : M = (3, 4)) 
  (h_l_inter_A : ∃ t1, line_l_parametric t1 = A ∧ circle_C_cartesian (prod.fst A) (prod.snd A))
  (h_l_inter_B : ∃ t2, line_l_parametric t2 = B ∧ circle_C_cartesian (prod.fst B) (prod.snd B)) :
  |dist M A| * |dist M B| = 9 :=
sorry

end intersection_products_l562_562538


namespace tangent_line_equation_l562_562239

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562239


namespace tangent_line_to_curve_l562_562231

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562231


namespace mean_transformed_data_stddev_transformed_data_l562_562625

variable {a : ℕ → ℝ} {n : ℕ} (k b : ℝ) (μ σ : ℝ)

-- Assuming variance and mean of the original data
axiom original_variance : ∀ (a : ℕ → ℝ) (n : ℕ), (∑ i in finset.range n, (a i - μ)^2) / n = σ^2
axiom original_mean : ∀ (a : ℕ → ℝ) (n : ℕ), (∑ i in finset.range n, a i) / n = μ

-- Declaring the transformed data
noncomputable def transformed_data (a : ℕ → ℝ) (k b : ℝ) : ℕ → ℝ := λ i, k * a i + b

-- Lean 4 statement that proves the mean of the transformed data is kμ + b
theorem mean_transformed_data (a : ℕ → ℝ) (n : ℕ) : (∑ i in finset.range n, transformed_data a k b i) / n = k * μ + b :=
by
  sorry

-- Lean 4 statement that proves the standard deviation of the transformed data is |k| * σ
theorem stddev_transformed_data (a : ℕ → ℝ) (n : ℕ) : 
  sqrt ((∑ i in finset.range n, (transformed_data a k b i - (k * μ + b))^2) / n) = |k| * σ :=
by
  sorry

end mean_transformed_data_stddev_transformed_data_l562_562625


namespace max_newsstands_six_corridors_l562_562882

def number_of_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem max_newsstands_six_corridors : number_of_intersections 6 = 15 := 
by sorry

end max_newsstands_six_corridors_l562_562882


namespace solution_set_of_inequality_system_l562_562262

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l562_562262


namespace find_a7_l562_562465

-- Definitions based on given conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * (a 1 - a 0)

-- Given condition in Lean statement
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 11 = 22

-- Proof problem
theorem find_a7 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : a 7 = 11 := 
  sorry

end find_a7_l562_562465


namespace arvin_total_distance_week_l562_562390

def arvin_distance_day (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | _ => arvin_distance_day (n - 1) + 1

theorem arvin_total_distance_week : 
  ∑ n in (Finset.range 5).map (λ i, i + 1), arvin_distance_day n = 20 := by
  sorry

end arvin_total_distance_week_l562_562390


namespace pencils_lost_l562_562503

theorem pencils_lost (bought_pencils remaining_pencils lost_pencils : ℕ)
                     (h1 : bought_pencils = 16)
                     (h2 : remaining_pencils = 8)
                     (h3 : lost_pencils = bought_pencils - remaining_pencils) :
                     lost_pencils = 8 :=
by {
  sorry
}

end pencils_lost_l562_562503


namespace sum_of_distances_l562_562150

noncomputable def point_on_sphere (O : Type) [metric_space O] (A B C D : O) (r : ℝ) : Prop :=
  dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

theorem sum_of_distances (O : Type) [metric_space O] (A B C D : O) (r : ℝ)
  (h : point_on_sphere O A B C D r) (hO : r = 1)
  (hVec : (A - O) + (B - O) + (C - O) = (0 : O)) :
  |dist A D + dist B D + dist C D| ≠ 3 :=
begin
  sorry
end

end sum_of_distances_l562_562150


namespace octagon_area_l562_562301

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562301


namespace smallest_rational_bound_l562_562416

theorem smallest_rational_bound :
  ∃ (r s : ℚ), r / s = 41 / 42 ∧ ∀ (k m n : ℕ), (1 / k + 1 / m + 1 / n < 1) → (1 / k + 1 / m + 1 / n ≤ r / s) :=
begin
  sorry
end

end smallest_rational_bound_l562_562416


namespace sin_shift_eq_cos_l562_562853

theorem sin_shift_eq_cos (α : ℝ) : sin (5 * Real.pi / 2 + α) = cos α :=
sorry

end sin_shift_eq_cos_l562_562853


namespace train_speed_l562_562379

theorem train_speed
  (train_length : ℝ) (platform_length : ℝ) (time_seconds : ℝ)
  (h_train_length : train_length = 450)
  (h_platform_length : platform_length = 300.06)
  (h_time : time_seconds = 25) :
  (train_length + platform_length) / time_seconds * 3.6 = 108.01 :=
by
  -- skipping the proof with sorry
  sorry

end train_speed_l562_562379


namespace find_k_l562_562020

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end find_k_l562_562020


namespace average_weight_of_whole_class_l562_562663

theorem average_weight_of_whole_class :
  let students_A := 26
  let students_B := 34
  let avg_weight_A := 50
  let avg_weight_B := 30
  let total_weight_A := avg_weight_A * students_A
  let total_weight_B := avg_weight_B * students_B
  let total_weight_class := total_weight_A + total_weight_B
  let total_students_class := students_A + students_B
  let avg_weight_class := total_weight_class / total_students_class
  avg_weight_class = 38.67 :=
by {
  sorry -- Proof is not required as per instructions
}

end average_weight_of_whole_class_l562_562663


namespace sum_of_squares_of_roots_l562_562434

theorem sum_of_squares_of_roots :
  ∀ r1 r2 : ℝ, (r1 + r2 = 14) ∧ (r1 * r2 = 8) → (r1^2 + r2^2 = 180) := by
  sorry

end sum_of_squares_of_roots_l562_562434


namespace octagon_area_inscribed_in_circle_l562_562312

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562312


namespace length_of_AB_in_right_triangle_l562_562133

def right_triangle (X Y Z : Type) [metric_space X] (A B C : X) := 
  angle A C B = 90

variables (A B C : Type) [metric_space A]

def AC : ℝ := 6
def BC : ℝ := 2
def AB : ℝ := 2 * real.sqrt 10

theorem length_of_AB_in_right_triangle :
  right_triangle A B C A B C → AB = real.sqrt (AC^2 + BC^2) :=
by sorry

end length_of_AB_in_right_triangle_l562_562133


namespace monotonically_decreasing_interval_l562_562611

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem monotonically_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | ∃ (s : ℝ), f'(s) < 0 } :=
sorry

end monotonically_decreasing_interval_l562_562611


namespace find_angle_A_find_length_b_l562_562812

variables (A B C a b c : ℝ)
variables (m n : ℝ × ℝ)
variables (triangle : Prop)

-- Conditions
def m_vector : ℝ × ℝ := (sqrt 3, cos A + 1)
def n_vector : ℝ × ℝ := (sin A, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def orthogonal_vectors (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

def triangle_ABC : Prop :=
  A + B + C = π

def side_lengths_positive : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_equation : Prop :=
  orthogonal_vectors m_vector n_vector

-- Problem Statement
theorem find_angle_A (h1 : angle_A_equation) : A = π / 3 :=
sorry

variables (a_value : ℝ := 2) (cos_B_value : ℝ := sqrt 3 / 3)

def sin_B_value : ℝ := sqrt 6 / 3

def sine_law (a b A B : ℝ) : Prop :=
  a / sin A = b / sin B

-- Problem Statement
theorem find_length_b (h2 : A = π / 3) (h3 : a = a_value) (h4 : cos B = cos_B_value) :
  b = 4 * sqrt 2 / 3 :=
sorry

end find_angle_A_find_length_b_l562_562812


namespace regular_octagon_area_l562_562294

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562294


namespace integers_between_2000_and_3000_divisible_by_12_18_24_l562_562500

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem integers_between_2000_and_3000_divisible_by_12_18_24 :
  ∃ n : ℕ, (2000 ≤ n ∧ n ≤ 3000 ∧ is_divisible n 72).count = 14 :=
by
  sorry

end integers_between_2000_and_3000_divisible_by_12_18_24_l562_562500


namespace regular_octagon_area_l562_562317

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562317


namespace basis_of_long_jump_measurement_is_D_l562_562536

def measurement_condition_A := "Vertical definition"
def measurement_condition_B := "Shortest line segment between two points"
def measurement_condition_C := "Two points determine a straight line"
def measurement_condition_D := "Shortest perpendicular line segment"

theorem basis_of_long_jump_measurement_is_D :
  ∀ (A B C D : String), 
  (D = "Shortest perpendicular line segment") → 
  (A ≠ measurement_condition_D) ∧ 
  (B ≠ measurement_condition_D) ∧ 
  (C ≠ measurement_condition_D) →
  ∃ basis, basis = D := 
by
  intros A B C D h1 h2
  use D
  exact h1

end basis_of_long_jump_measurement_is_D_l562_562536


namespace marika_father_age_twice_l562_562939

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l562_562939


namespace reciprocals_not_arithmetic_sequence_l562_562807

theorem reciprocals_not_arithmetic_sequence 
  (a b c : ℝ) (h : 2 * b = a + c) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ¬ (1 / a + 1 / c = 2 / b) :=
by
  sorry

end reciprocals_not_arithmetic_sequence_l562_562807


namespace tan_pi_seventh_product_tan_pi_seventh_square_sum_l562_562188

theorem tan_pi_seventh_product :
  tan (Real.pi / 7) * tan (2 * Real.pi / 7) * tan (3 * Real.pi / 7) = Real.sqrt 7 :=
sorry

theorem tan_pi_seventh_square_sum :
  tan (Real.pi / 7) ^ 2 + tan (2 * Real.pi / 7) ^ 2 + tan (3 * Real.pi / 7) ^ 2 = 21 :=
sorry

end tan_pi_seventh_product_tan_pi_seventh_square_sum_l562_562188


namespace lcm_factor_l562_562987

/-- The h.c.f. of two numbers is 10, and the larger of the two numbers is 150. 
The other two factors of their l.c.m. are 11 and a certain value. 
Prove that the second of the other two factors of their l.c.m. is 15. -/
theorem lcm_factor {A B : ℕ} (hcf_A_B : Nat.gcd A B = 10) (larger_A : A = 150) (lcm_other_factor : ∃ X, Nat.lcm A B = 10 * 11 * X) :
  ∃ X, X = 15 :=
by 
  have h₀ : exists multiple of 10 that is 150 := sorry
  have h₁ : Nat.lcm A B = 150 := sorry
  use 15
  exact sorry

end lcm_factor_l562_562987


namespace max_value_of_P_l562_562164

noncomputable def P (a b c : ℝ) : ℝ :=
  (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1))

theorem max_value_of_P (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c + a + c = b) :
  ∃ x, x = 1 ∧ ∀ y, (y = P a b c) → y ≤ x :=
sorry

end max_value_of_P_l562_562164


namespace sandwich_count_l562_562384

theorem sandwich_count (lunch_meat: ℕ) (cheese: ℕ) (choose_meat: lunch_meat = 12) (choose_cheese: cheese = 11) : (Nat.choose 12 1) * (Nat.choose 11 3) = 1980 :=
by
  rw [choose_meat, choose_cheese]
  calc
    (Nat.choose 12 1) * (Nat.choose 11 3)
        = 12 * 165 : by rw [Nat.choose_one_right, Nat.choose_five]
    ... = 1980 : by norm_num

example : sandwich_count 12 11 rfl rfl := sorry

end sandwich_count_l562_562384


namespace obtuse_triangles_in_convex_quad_l562_562799

variables (A B C D : Type)
variables (is_convex_quad : ConvexQuadrilateral A B C D)
variables (is_obtuse_angle_D : IsObtuseAngle (angle D))
variables (triangles : Finset (Triangle A B C D))
variables (are_obtuse_triangles : ∀ t ∈ triangles, IsObtuseTriangle t)
variables (disjoint : ∀ t1 t2 ∈ triangles, t1 ≠ t2 → Intersection t1 t2 = ∅)
variables (boundary_condition : ∀ t ∈ triangles, ∃ v1 v2 v3, Vertices t = {v1, v2, v3} ∧ v1, v2, v3 ∈ {A, B, C, D})

theorem obtuse_triangles_in_convex_quad (n : ℕ) (h : triangles.card = n) : n ≥ 4 :=
sorry

end obtuse_triangles_in_convex_quad_l562_562799


namespace distance_P2017_P2018_l562_562574

-- Define a point type, possibly using real numbers for coordinates.
structure Point :=
(x : ℝ) (y : ℝ)

-- Define function to calculate distance between points
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define reflection functions based on given conditions
def reflect_with_respect_to (A B : Point) : Point :=
  ⟨2 * A.x - B.x, 2 * A.y - B.y⟩

noncomputable def P : ℕ → Point
| 0           := ...  -- initial point, should be defined
| (nat.succ n) :=
    if n % 2 = 0
    then reflect_with_respect_to P1 (P n)
    else reflect_with_respect_to P2 (P n)

-- Main statement to prove
theorem distance_P2017_P2018 :
  ∀ P1 P2 : Point,
    dist P1 P2 = 1 →
    dist (P 2017) (P 2018) = 1 :=
by sorry

end distance_P2017_P2018_l562_562574


namespace S_2017_eq_18134_l562_562155

noncomputable def S (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), Nat.floor (Real.log k / Real.log 2)

theorem S_2017_eq_18134 : S 2017 = 18134 :=
by
  sorry

end S_2017_eq_18134_l562_562155


namespace angle_DEC_is_120_l562_562116

noncomputable def line (α β: Type) [CommRing α] := α → α
variables (α β γ : Type) [CommRing α] [CommRing β] [CommRing γ]

theorem angle_DEC_is_120
  (AB: line α α)             -- Line AB is a straight line.
  (C: α)                     -- Point C lies on line AB.
  (D: α)                     -- Point D
  (E: α)                     -- Point E
  (h1 : ∠DCE = 115)          -- Angle DCE=115°.
  (h2 : ∠ECD = 35)           -- Angle ECD=35°.
  (h3 : ∠ACD = 90)           -- Angle ACD=90° (perpendicular)
  (h4 : ∠BCD = 90)           -- Angle BCD=90° (perpendicular)
  : ∠DEC = 120 :=            -- Conclusion: angle DEC = 120°.
by
  sorry

end angle_DEC_is_120_l562_562116


namespace trader_gain_percentage_l562_562659

theorem trader_gain_percentage (C : ℝ) (hC : 0 < C) : 
  let cost_of_pens := 88 * C
  let gain := 22 * C
  let selling_price := cost_of_pens + gain
  (gain / cost_of_pens) * 100 = 25 := 
by
  let cost_of_pens := 88 * C
  let gain := 22 * C
  let gain_percentage := (gain / cost_of_pens) * 100
  have h1 : gain_percentage = ((22 * C) / (88 * C)) * 100 := by rfl
  have h2 : gain_percentage = (22 / 88) * 100 := by
    simp [mul_comm, mul_div_assoc, div_self hC]
  have h3 : gain_percentage = (1 / 4) * 100 := by
    norm_num [div_eq_mul_inv]
  have h4 : gain_percentage = 25 := by
    exact_mod_cast (1 / 4 * 100)
  exact h4

end trader_gain_percentage_l562_562659


namespace charlie_roaming_area_l562_562406

-- Definitions of the conditions
def shed_length : ℝ := 4
def shed_width : ℝ := 3
def leash_length : ℝ := 4
def total_area : ℝ := (3 / 4) * real.pi * (leash_length ^ 2) + (1 / 2) * real.pi * (1 ^ 2)

-- Proof statement
theorem charlie_roaming_area :
  total_area = 12.5 * real.pi :=
sorry

end charlie_roaming_area_l562_562406


namespace monotone_intervals_and_extreme_value_min_difference_extreme_points_l562_562839

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem monotone_intervals_and_extreme_value :
  (∀ x, f x = x^2 + x - Real.log x) →
  (∀ x, (1 / 2 < x → deriv f x > 0) ∧ (0 < x ∧ x < 1 / 2 → deriv f x < 0)) ∧
  (f (1 / 2) = 3 / 4 + Real.log 2) :=
begin
  intros,
  sorry
end

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := x^2 - (2 * b - 2) * x + 2 * Real.log x

theorem min_difference_extreme_points {b : ℝ} (h : b ≥ 1 + 4 * Real.sqrt 3 / 3) :
  ∃ (x1 x2 : ℝ), x1 < x2 ∧
  (deriv (g b) x1 = 0 ∧ deriv (g b) x2 = 0) →
  (g x1 b - g x2 b) = 8 / 3 - 2 * Real.log 3 :=
begin
  intros,
  sorry
end

end monotone_intervals_and_extreme_value_min_difference_extreme_points_l562_562839


namespace binomial_expansion_x10_minus_x5_coeff_a5_l562_562860

theorem binomial_expansion_x10_minus_x5_coeff_a5 :
  let f := λ x : ℝ, x^10 - x^5,
      expansion := ∑ i in Finset.range 11, (binomial 10 i) * (x-1)^i - ∑ j in Finset.range 6, (binomial 5 j) * (x-1)^j,
      a5 := (binomial 10 5) - (binomial 5 0)
  in  true :=
by {
  sorry
}

end binomial_expansion_x10_minus_x5_coeff_a5_l562_562860


namespace part1_part2_period_part2_monotonic_intervals_increasing_part2_monotonic_intervals_decreasing_part3_l562_562082

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * (sin x)^2 + 1

theorem part1 : f (5 * real.pi / 4) = sqrt 3 := sorry

theorem part2_period : ∃ T > 0, ∀ x, f (x + T) = f x := 
  ⟨real.pi, by sorry⟩

theorem part2_monotonic_intervals_increasing : 
  ∀ k : ℤ, ∀ x, -real.pi / 3 + k * real.pi ≤ x ∧ x ≤ real.pi / 6 + k * real.pi → function.monotone (λ y, f y ) :=
sorry

theorem part2_monotonic_intervals_decreasing : 
  ∀ k : ℤ, ∀ x, real.pi / 6 + k * real.pi ≤ x ∧ x ≤ 2 * real.pi / 3 + k * real.pi → function.antitone (λ y, f y ) :=
sorry

theorem part3 : f (-real.pi / 5) < f (7 * real.pi / 8) := sorry

end part1_part2_period_part2_monotonic_intervals_increasing_part2_monotonic_intervals_decreasing_part3_l562_562082


namespace solution_set_l562_562070

-- Define the function f with given properties
axiom f : ℝ → ℝ

axiom domain_f : ∀ x : ℝ, f x = f x
axiom f_pos : ∀ x > 0, f x > 1
axiom f_mul : ∀ x y : ℝ, f (x + y) = f x * f y

-- Define the function log base 1/2
noncomputable def log_half (x : ℝ) : ℝ := log x / log (1/2)

-- The goal is to prove the inequality solution set
theorem solution_set : { x : ℝ | f (log_half x) ≤ 1 / f (log_half x + 1) } = { x : ℝ | x ≥ 4 } :=
by
  sorry

end solution_set_l562_562070


namespace find_ratio_b_over_a_l562_562085

theorem find_ratio_b_over_a (a b : ℝ)
  (h1 : ∀ x, deriv (fun x => a * x^2 + b) x = 2 * a * x)
  (h2 : deriv (fun x => a * x^2 + b) 1 = 2)
  (h3 : a * 1^2 + b = 3) : b / a = 2 := 
sorry

end find_ratio_b_over_a_l562_562085


namespace tangent_line_eq_l562_562240

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562240


namespace distance_of_intersection_points_l562_562829

def C1 (x y : ℝ) : Prop := x - y + 4 = 0
def C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

theorem distance_of_intersection_points {A B : ℝ × ℝ} (hA1 : C1 A.fst A.snd) (hA2 : C2 A.fst A.snd)
  (hB1 : C1 B.fst B.snd) (hB2 : C2 B.fst B.snd) : dist A B = Real.sqrt 2 := by
  sorry

end distance_of_intersection_points_l562_562829


namespace opposite_of_neg_3_is_3_l562_562995

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l562_562995


namespace marikas_father_twice_her_age_l562_562944

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l562_562944


namespace military_unit_soldiers_l562_562713

theorem military_unit_soldiers:
  ∃ (x N : ℕ), 
      (N = x * (x + 5)) ∧
      (N = 5 * (x + 845)) ∧
      N = 4550 :=
by
  sorry

end military_unit_soldiers_l562_562713


namespace problem_area_of_centroid_quadrilateral_l562_562119

noncomputable def area_of_quadrilateral_centroids (side_length : ℝ) (EQ FQ : ℝ) : ℝ :=
  let diag_1 := (2 / 3) * side_length
  let diag_2 := (2 / 3) * side_length
  (1 / 2) * diag_1 * diag_2

theorem problem_area_of_centroid_quadrilateral :
  let side_length := 40
  let EQ := 16
  let FQ := 34
  area_of_quadrilateral_centroids side_length EQ FQ = 6400 / 9 :=
by
  intro side_length EQ FQ
  simp [area_of_quadrilateral_centroids]
  norm_num

end problem_area_of_centroid_quadrilateral_l562_562119


namespace max_people_transition_l562_562446

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end max_people_transition_l562_562446


namespace percentage_less_than_l562_562866

variable {w x y z : ℝ}

theorem percentage_less_than (h₁ : w = 0.60 * x) (h₂ : z = 0.54 * y) (h₃ : z = 1.50 * w) : (y - x) / y = 0.40 :=
by
  have h₄ : 1.50 * 0.60 * x = 0.90 * x := by norm_num
  rw [←h₃, h₁] at h₂
  have h₅ : 1.50 * 0.60 * x = 0.54 * y := by linarith
  rw [h₄] at h₅
  field_simp at h₅
  rw [←mul_inv] at h₅
  have h₆ : x = 0.60 * y := by linarith
  have h₇ : y - x = y - 0.60 * y := by rw [h₆]
  rw [sub_mul, one_mul] at h₇
  field_simp at h₇
  exact h₇

end percentage_less_than_l562_562866


namespace mutually_exclusive_not_contradictory_draw_l562_562040

def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)
def not_contradictory (A B : Prop) : Prop := ∃ X, X = A ∨ X = B ∨ (X ≠ A ∧ X ≠ B)

variable {bag : set string}
variable {drawn : set string}

axiom h1 : bag = {"red1", "red2", "black1", "black2"}
axiom h2 : ∃ drawn, drawn ⊆ bag ∧ drawn.card = 2

def exactly_one_black_ball (drawn : set string) : Prop :=
  drawn.count("black1") + drawn.count("black2") = 1

def exactly_two_black_balls (drawn : set string) : Prop :=
  drawn.count("black1") + drawn.count("black2") = 2

theorem mutually_exclusive_not_contradictory_draw 
  (bag_condition : bag = {"red1", "red2", "black1", "black2"})
  (draw_condition : ∃ drawn, drawn ⊆ bag ∧ drawn.card = 2) :
  let E1 := exactly_one_black_ball drawn,
      E2 := exactly_two_black_balls drawn
  in mutually_exclusive E1 E2 ∧ not_contradictory E1 E2 :=
by
  sorry

end mutually_exclusive_not_contradictory_draw_l562_562040


namespace sequence_properties_l562_562463

noncomputable def a (n : ℕ) : ℕ := 2 + (n - 1) * 2

noncomputable def b (n : ℕ) : ℕ := 2 * (2 ^ (n - 1))

theorem sequence_properties :
  (a 1 = 2) ∧ (a 2 = 4) ∧
  (b 1 = 2) ∧ (b 2 = 4) ∧
  (∀ n : ℕ, a n = 2 * n ∧ b n = 2 ^ n) ∧
  (∀ n : ℕ, n ≥ 3 → a n < b n) :=
by
  split; sorry

end sequence_properties_l562_562463


namespace smallest_sum_a_b_l562_562255

theorem smallest_sum_a_b :
  ∃ (a b : ℕ), (7 * b - 4 * a = 3) ∧ a > 7 ∧ b > 7 ∧ a + b = 24 :=
by
  sorry

end smallest_sum_a_b_l562_562255


namespace bee_travel_distance_l562_562936

-- Define the constants
def distance_between_people : ℝ := 120
def speed_mr_A : ℝ := 30
def speed_mrs_A : ℝ := 10
def speed_bee : ℝ := 60

-- Define the time taken for Mr. A and Mrs. A to meet
def time_to_meet : ℝ := distance_between_people / (speed_mr_A + speed_mrs_A)

-- Theorem to prove the distance the bee travels
theorem bee_travel_distance : (speed_bee * time_to_meet) = 180 := by
  sorry

end bee_travel_distance_l562_562936


namespace sum_of_k_for_distinct_integer_roots_l562_562328

theorem sum_of_k_for_distinct_integer_roots :
  (∑ k in {k : ℤ | ∃ p q : ℤ, p ≠ q ∧ (2 * (p^2 + q^2) - k * (p + q) + 12 = 0)}, k) = 0 := by
  sorry

end sum_of_k_for_distinct_integer_roots_l562_562328


namespace area_of_inscribed_regular_octagon_l562_562304

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562304


namespace geometric_series_sum_l562_562421

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 6
  let sum := (∑ i in Finset.range n, a * r^i)
  sum = 4095 / 12288 :=
by
  sorry

end geometric_series_sum_l562_562421


namespace coefficient_x8_in_expansion_l562_562425

theorem coefficient_x8_in_expansion :
  coeff (expand (1 - x + 3 * x^2)^5) x^8 = 270 := 
sorry

end coefficient_x8_in_expansion_l562_562425


namespace value_of_inverse_product_l562_562482

theorem value_of_inverse_product (x y : ℝ) (h1 : x * y > 0) (h2 : 1/x + 1/y = 15) (h3 : (x + y) / 5 = 0.6) :
  1 / (x * y) = 5 :=
by 
  sorry

end value_of_inverse_product_l562_562482


namespace bill_experience_now_l562_562745

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l562_562745


namespace profit_percent_calculation_l562_562861

variable (SP : ℝ) (CP : ℝ) (Profit : ℝ) (ProfitPercent : ℝ)
variable (h1 : CP = 0.75 * SP)
variable (h2 : Profit = SP - CP)
variable (h3 : ProfitPercent = (Profit / CP) * 100)

theorem profit_percent_calculation : ProfitPercent = 33.33 := 
sorry

end profit_percent_calculation_l562_562861


namespace regular_octagon_area_l562_562319

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562319


namespace positive_difference_between_a_values_l562_562898

def f (n : ℤ) : ℤ :=
  if n < 0 then n^2 + 3*n + 2 else 3*n - 15

noncomputable def a_values_difference : ℚ :=
  let f_neg3 := (f (-3) : ℤ)
  let f_3 := (f 3 : ℤ)
  have h_eq : f_neg3 + f_3 + f (a: ℤ) = 0 → f a = 4 := 
    sorry
  let a1 := -2 
  let a2 := 19 / 3
  (19 / 3 : ℚ) - (-2 : ℚ)

theorem positive_difference_between_a_values : a_values_difference = 25 / 3 :=
  sorry

end positive_difference_between_a_values_l562_562898


namespace polynomial_div_simplify_l562_562667

theorem polynomial_div_simplify (x : ℝ) (hx : x ≠ 0) :
  (6 * x ^ 4 - 4 * x ^ 3 + 2 * x ^ 2) / (2 * x ^ 2) = 3 * x ^ 2 - 2 * x + 1 :=
by sorry

end polynomial_div_simplify_l562_562667


namespace arithmetic_progressions_sum_l562_562159

theorem arithmetic_progressions_sum 
  (c : ℕ → ℝ) (d : ℕ → ℝ)
  (hc1 : c 1 = 30)
  (hd1 : d 1 = 90)
  (h100 : c 100 + d 100 = 300) : 
  (Finset.range 100).sum (λ n, c (n + 1) + d (n + 1)) = 21000 := 
sorry

end arithmetic_progressions_sum_l562_562159


namespace moles_of_NH4Cl_formed_l562_562025

-- Definitions that directly appear in the conditions
def NH3 : Type := ...
def HCl : Type := ...
def NH4Cl : Type := ...

-- Stoichiometric coefficient in a 1:1:1 ratio (these values are affirmed by the balanced equation)
def coeff_NH3 : ℕ := 1
def coeff_HCl : ℕ := 1
def coeff_NH4Cl : ℕ := 1

-- Given initial moles of NH3 and HCl
def initial_moles_NH3 : ℕ := 1
def initial_moles_HCl : ℕ := 1

-- The proof goal: Given the reaction and initial conditions, prove the moles of NH4Cl formed is 1 mole.
theorem moles_of_NH4Cl_formed :
  initial_moles_NH3 = coeff_NH3 →
  initial_moles_HCl = coeff_HCl →
  (coeff_NH3 * initial_moles_NH3) / coeff_NH3 = coeff_NH4Cl :=
by
  intros h1 h2
  sorry

end moles_of_NH4Cl_formed_l562_562025


namespace chi_square_test_probability_distribution_expectation_value_l562_562272

variables (a b c d : ℕ)
variables (n : ℕ := a + b + c + d)

-- Chi-Square Calculation
noncomputable def K_squared : ℝ :=
  (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

def crit_value_85 : ℝ := 2.072

theorem chi_square_test (a b c d : ℕ) (h1 : a = 12) (h2 : b = 4)
  (h3 : c = 9) (h4 : d = 5) (K_sq : ℝ := K_squared a b c d)
  (crit_val : ℝ := crit_value_85) :
  K_sq < crit_val := by
  sorry

-- Probability Distribution and Mathematical Expectation
def prob (n k : ℕ) : ℚ := nat.choose n k / nat.choose 9 3

noncomputable def expectation : ℚ :=
  (0 * prob 5 3) + (1 * prob 4 1 * prob 5 2) + (2 * prob 4 2 * prob 5 1) + (3 * prob 4 3)

theorem probability_distribution (h : 4 = 4) :
  prob 5 3 = 5 / 42 ∧ prob 4 1 * prob 5 2 = 10 / 21 ∧ prob 4 2 * prob 5 1 = 5 / 14 ∧ prob 4 3 = 1 / 21 := by
  sorry

theorem expectation_value : expectation = 4 / 3 := by
  sorry

end chi_square_test_probability_distribution_expectation_value_l562_562272


namespace reciprocal_sqrt5_minus_2_l562_562617

theorem reciprocal_sqrt5_minus_2 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := 
by
  sorry

end reciprocal_sqrt5_minus_2_l562_562617


namespace calculate_length_of_train_l562_562380

noncomputable def length_of_train (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := (relative_speed_kmh : ℝ) * 1000 / 3600
  relative_speed_ms * time_seconds

theorem calculate_length_of_train :
  length_of_train 50 5 7.2 = 110 := by
  -- This is where the actual proof would go, but it's omitted for now as per instructions.
  sorry

end calculate_length_of_train_l562_562380


namespace inscribed_circle_diameter_of_right_triangle_l562_562345

theorem inscribed_circle_diameter_of_right_triangle (a b : ℕ) (hc : a = 8) (hb : b = 15) :
  2 * (60 / (a + b + Int.sqrt (a ^ 2 + b ^ 2))) = 6 :=
by
  sorry

end inscribed_circle_diameter_of_right_triangle_l562_562345


namespace tangent_line_eq_l562_562245

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562245


namespace min_abs_phi_l562_562864

theorem min_abs_phi {ϕ : ℝ} (h_symm : ∀ x, 3 * cos (2 * x + ϕ) = 3 * cos(2 * (4 * π / 3 - x) + ϕ)) : |ϕ| = π / 6 := by
  sorry

end min_abs_phi_l562_562864


namespace minimum_framing_feet_needed_l562_562675

theorem minimum_framing_feet_needed :
  let original_width := 4
  let original_height := 6
  let enlarged_width := 4 * original_width
  let enlarged_height := 4 * original_height
  let border := 3
  let total_width := enlarged_width + 2 * border
  let total_height := enlarged_height + 2 * border
  let perimeter := 2 * (total_width + total_height)
  let framing_feet := (perimeter / 12).ceil
  framing_feet = 9 := by
  -- The theorem statement translates given conditions into definitions and finally asserts the result.
  sorry

end minimum_framing_feet_needed_l562_562675


namespace intersection_of_g_and_g_inv_l562_562899

noncomputable def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 9 * x - 5
noncomputable def g_inv (y : ℝ) : Set ℝ := {x : ℝ | g x = y}

theorem intersection_of_g_and_g_inv :
  ∃ c d : ℝ, (c = d) ∧ g c = d ∧ (∀ x y : ℝ, (g x = y ∧ y = g_inv x.set_classical.choice) → x = c ∧ y = d) :=
sorry

end intersection_of_g_and_g_inv_l562_562899


namespace elaine_meals_count_l562_562737

theorem elaine_meals_count:
  let entrees := 4 in
  let drinks := 3 in
  let desserts := 2 in
  entrees * drinks * desserts = 24 :=
by
  sorry

end elaine_meals_count_l562_562737


namespace find_a_eq_0_range_of_a_l562_562454

-- First part
theorem find_a_eq_0 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = |x + a|) (h2 : ∀ x, f(x) ≥ |2 * x + 3| → x ∈ Icc (-3 : ℝ) (-1 : ℝ))
: a = 0 :=
sorry

-- Second part
theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = |x + a|) (h2 : ∀ x, f(x) + |x - a| ≥ a^2 - 2 * a)
: 0 ≤ a ∧ a ≤ 4 :=
sorry

end find_a_eq_0_range_of_a_l562_562454


namespace unique_diagonal_products_l562_562884

def vertices := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem unique_diagonal_products : 
  ∃ (arrangement : Fin 9 → ℕ), 
  (∀ i, arrangement i ∈ vertices) ∧ 
  (∀ i j k l, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l 
  → (arrangement i) * (arrangement j) ≠ (arrangement k) * (arrangement l)) :=
sorry

end unique_diagonal_products_l562_562884


namespace problem_1_problem_2_problem_3_l562_562187

noncomputable def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C
noncomputable def area_quadrilateral (e f φ : ℝ) : ℝ := (1/2) * e * f * Real.sin φ

theorem problem_1 (a b C : ℝ) (hC : Real.sin C ≤ 1) : 
  area_triangle a b C ≤ (a^2 + b^2) / 4 :=
sorry

theorem problem_2 (e f φ : ℝ) (hφ : Real.sin φ ≤ 1) : 
  area_quadrilateral e f φ ≤ (e^2 + f^2) / 4 :=
sorry

theorem problem_3 (a b C c d D : ℝ) 
  (hC : Real.sin C ≤ 1) 
  (hD : Real.sin D ≤ 1) :
  area_triangle a b C + area_triangle c d D ≤ (a^2 + b^2 + c^2 + d^2) / 4 :=
sorry

end problem_1_problem_2_problem_3_l562_562187


namespace smallest_d_for_inverse_l562_562575

def g (x : ℝ) : ℝ := (x + 1)^2 - 6

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x₁ x₂ ≥ d, g x₁ = g x₂ → x₁ = x₂) ∧ (∀ y : ℝ, ∃ x : ℝ, x ≥ d ∧ g x = y) ∧ d = -1 :=
by
  sorry

end smallest_d_for_inverse_l562_562575


namespace beginner_trigonometry_probability_l562_562526

-- Definitions for the conditions
variable (C : ℝ) -- Number of calculus students
variable (T : ℝ) -- Total number of students
variable (beginner_calculus : ℝ) -- Number of beginner calculus students
variable (beginner_total : ℝ) -- Number of beginners in total

-- Expressions for values derived from given conditions
def trigonometry_students := 1.5 * C
def T := C + trigonometry_students
def beginner_calculus := 0.7 * C
def beginner_total := (4 / 5) * T

-- Derived calculation for number of beginner trigonometry students
def beginner_trigonometry := beginner_total - beginner_calculus

-- Probability calculation
def probability_beginner_trigonometry := beginner_trigonometry / T

-- Theorem statement
theorem beginner_trigonometry_probability :
  probability_beginner_trigonometry = 13 / 25 :=
by
  -- Definitions and steps here would follow the logic to prove the statement
  sorry

end beginner_trigonometry_probability_l562_562526


namespace eggs_needed_per_month_l562_562177

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l562_562177


namespace f_neg_def_l562_562760

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def f_pos_def (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x = x * (1 - x)

theorem f_neg_def (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f_pos_def f) :
  ∀ x : ℝ, x < 0 → f x = x * (1 + x) :=
by
  sorry

end f_neg_def_l562_562760


namespace polynomial_coeffs_l562_562090

theorem polynomial_coeffs :
  ( ∃ (a1 a2 a3 a4 a5 : ℕ), (∀ (x : ℝ), (x + 1) ^ 3 * (x + 2) ^ 2 = x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5) ∧ a4 = 16 ∧ a5 = 4) := 
by
  sorry

end polynomial_coeffs_l562_562090


namespace opposite_number_subtraction_l562_562653

variable (a b : ℝ)

theorem opposite_number_subtraction : -(a - b) = b - a := 
sorry

end opposite_number_subtraction_l562_562653


namespace max_distance_point_to_line_l562_562089

theorem max_distance_point_to_line :
  let line_param (t : ℝ) := (4 - 2 * t, t - 2)
  let ellipse (x y : ℝ) := x^2 / 4 + y^2 = 1
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
  ∃ t : ℝ, line_param t = P ∨ distance line_param t P = (2 * (10:ℝ).sqrt) / 5 :=
begin
  sorry
end

end max_distance_point_to_line_l562_562089


namespace fraction_value_l562_562451

theorem fraction_value
  (a b c d : ℚ)
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b ≠ 0)
  (h4 : d ≠ 0)
  (h5 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 :=
sorry

end fraction_value_l562_562451


namespace part2_part3_l562_562053

-- Define the arithmetic and geometric sequence properties.
def arithmetic_sequence (a b : ℤ) (n : ℤ) : ℤ := a + (n - 1) * b
def geometric_sequence (a r : ℤ) (n : ℤ) : ℤ := a * r^(n-1)

-- Define an r-related sequence
def r_related_sequence (a : ℕ → ℤ) (r : ℕ) := 
  (∀ n : ℕ, n < r → a n = arithmetic_sequence (a 1) 1 n) ∧ 
  (∀ n : ℕ, n ≥ (r-1) → a n = geometric_sequence (a (r-1)) 2 (n - (r-1)))

-- General formula for a "6-related sequence"
def a_n (n : ℕ) : ℤ := if n ≤ 6 then n - 4 else 2^(n - 5)

-- Sum of the first n terms S_n
def S_n (n : ℕ) : ℤ := ∑ i in finset.range (n+1), a_n i

-- Prove that a_n S_n ≥ a_6 S_6 for any n ∈ N^*
theorem part2 (n : ℕ) : n > 0 → (a_n n) * (S_n n) ≥ (a_n 6) * (S_n 6) := sorry

-- Given r-related sequence and a1 = -10, find pairs (k, m)
def k_m_pairs (r : ℕ) (k m : ℕ) : Prop := 
  r_related_sequence a_n r ∧ a_n 1 = -10 ∧ k < m ∧
  (∑ i in finset.range (k+1), a_n i) = (∑ i in finset.range (m+1), a_n i)

def valid_pairs := [(5, 15), (8, 13), (9, 12), (10, 11)]

theorem part3 (k m : ℕ) : k < m → k_m_pairs 13 k m ↔ (k, m) ∈ valid_pairs := sorry

end part2_part3_l562_562053


namespace exists_circle_integral_zero_l562_562563

noncomputable def twice_continuously_differentiable {X : Type*} [TopologicalSpace X] (f : X → ℝ) : Prop :=
∀ (x : X), Continuous ∘ D f x ∧ ∀ (s : ℝ), Smooth s (D f x)

def rho (h : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
y * (deriv h x) - x * (deriv h y)

theorem exists_circle_integral_zero (h : ℝ → ℝ → ℝ) (d r : ℝ) (hd : d > r) (H : twice_continuously_differentiable (λ p : ℝ × ℝ, h p.1 p.2)) :
  ∃ (p : ℝ × ℝ) (R : ℝ), R = r ∧ (dist p (0, 0) = d) ∧ (∫∫ 0 ≤ theta ≤ 2 * π, rho h (R * cos theta) (R * sin theta)) = 0 :=
sorry

end exists_circle_integral_zero_l562_562563


namespace part1_part2_part3_l562_562055

def A (x y : ℝ) := 2*x^2 + 3*x*y + 2*y
def B (x y : ℝ) := x^2 - x*y + x

theorem part1 (x y : ℝ) : A x y - 2 * B x y = 5*x*y - 2*x + 2*y := by
  sorry

theorem part2 (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y = 28 ∨ A x y - 2 * B x y = -40 ∨ A x y - 2 * B x y = -20 ∨ A x y - 2 * B x y = 32 := by
  sorry

theorem part3 (y : ℝ) : (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by
  sorry

end part1_part2_part3_l562_562055


namespace ratio_kara_amanda_l562_562405

-- Definitions of the conditions in the proof problem

def Candice_read_books := 18 -- Candice read 18 books
def factor_candice_amanda := 3 -- Candice read 3 times as many books as Amanda
def Amanda_read_books := Candice_read_books / factor_candice_amanda -- Amanda read books

def fraction_kara_amanda (x : ℝ) := x * Amanda_read_books -- Kara read a fraction x of Amanda's books

-- The ratio of the number of books Kara read to the number of books Amanda read is x
theorem ratio_kara_amanda (x : ℝ) : fraction_kara_amanda x / Amanda_read_books = x := by
  unfold fraction_kara_amanda Amanda_read_books Candice_read_books
  field_simp
  sorry

end ratio_kara_amanda_l562_562405


namespace students_distribute_l562_562359

theorem students_distribute (x y : ℕ) (h₁ : x + y = 4200)
        (h₂ : x * 108 / 100 + y * 111 / 100 = 4620) :
    x = 1400 ∧ y = 2800 :=
by
  sorry

end students_distribute_l562_562359


namespace father_twice_marika_age_in_2036_l562_562947

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l562_562947


namespace arithmetic_sequence_general_formula_is_not_term_l562_562121

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) (h17 : a 17 = 66) :
  ∀ n : ℕ, a n = 4 * n - 2 := sorry

theorem is_not_term (a : ℕ → ℤ) 
  (ha : ∀ n : ℕ, a n = 4 * n - 2) :
  ∀ k : ℤ, k = 88 → ¬ ∃ n : ℕ, a n = k := sorry

end arithmetic_sequence_general_formula_is_not_term_l562_562121


namespace largest_whole_number_l562_562428

theorem largest_whole_number (x : ℕ) : 8 * x < 120 → x ≤ 14 :=
by
  intro h
  -- prove the main statement here
  sorry

end largest_whole_number_l562_562428


namespace rate_of_barbed_wire_correct_l562_562980

noncomputable def rate_of_barbed_wire
  (area_of_square : ℝ) 
  (additional_barbed_wire : ℝ) 
  (gate_width : ℝ) 
  (num_gates : ℕ) 
  (total_cost : ℝ) : ℝ :=
  let side_length := real.sqrt area_of_square
  let perimeter := 4 * side_length
  let adjusted_perimeter := perimeter - num_gates * gate_width
  let total_length := adjusted_perimeter + additional_barbed_wire
  total_cost / total_length

-- Given conditions
def area_of_square := 3136
def additional_barbed_wire := 3
def gate_width := 1
def num_gates := 2
def total_cost := 732.6

theorem rate_of_barbed_wire_correct :
  rate_of_barbed_wire area_of_square additional_barbed_wire gate_width num_gates total_cost = 3.256 := by
  sorry  -- proof to be filled in

end rate_of_barbed_wire_correct_l562_562980


namespace probability_sum_seven_l562_562275

theorem probability_sum_seven :
  let faces := fin 6
  ∃ (dice₁ dice₂ : faces → ℕ),
  (∀ f ∈ faces, (1 ≤ dice₁ f ∧ dice₁ f ≤ 6) ∧ (1 ≤ dice₂ f ∧ dice₂ f ≤ 6)) →
  (∀ x ∈ (finset.univ : finset (faces × faces)), 
    (dice₁ x.1 + dice₂ x.2 = 7)) →
  ∑ x in finset.univ, (if dice₁ x.1 + dice₂ x.2 = 7 then 1 else 0) / 
  finset.card (finset.univ : finset (faces × faces)) = 1 / 6 := sorry

end probability_sum_seven_l562_562275


namespace cube_partition_l562_562705

/-- 
Let there be a cube of edge length 4 cm. This cube is cut into smaller cubes with edge lengths 
of whole numbers of centimeters and these cubes are not uniformly sized. The total number of 
smaller cubes after partitioning is 57.
-/
theorem cube_partition (N : ℕ) : 
  (∃ l₁ l₂ : ℕ, l₁ ≠ l₂ ∧ l₁ ∣ 4 ∧ l₂ ∣ 4 ∧
  let total_volume := 4^3 in
  let vol_l1 := l₁^3 in
  let vol_l2 := l₂^3 in
  let count_l1 := total_volume / vol_l1 in
  let volume_remaining := total_volume % vol_l1 in
  let count_l2 := volume_remaining / vol_l2 in
  count_l1 + count_l2 = N ->
  N = 57) :=
sorry

end cube_partition_l562_562705


namespace congruence_determinant_l562_562637

theorem congruence_determinant :
  ∀ (AAS SAS AAS' AAA : Prop),
  (AAS = "One side and any two angles") →
  (SAS = "Two sides and the angle between them") →
  (AAS' = "Two angles and the side opposite one of the angles") →
  (AAA = "Three corresponding angles equal") →
  (AAA → ¬(∀ (tri1 tri2 : Triangle), tri1 ≅ tri2)) :=
by intros AAS SAS AAS' AAA hAAS hSAS hAAS' hAAA H; sorry

end congruence_determinant_l562_562637


namespace probability_abs_le_one_in_interval_l562_562596

theorem probability_abs_le_one_in_interval (x : ℝ) (h : x ∈ set.Icc (-2 : ℝ) 4) :
  ∃ (p : ℚ), p = 1 / 3 := 
sorry

end probability_abs_le_one_in_interval_l562_562596


namespace four_letter_product_eq_l562_562769

open Nat

def letter_value (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26 | _ => 0

theorem four_letter_product_eq (list1 : List Char) (list2 : List Char) :
  (list1 = ['T', 'U', 'W', 'Y']) →
  (list1.foldr (fun c acc => acc * (letter_value c)) 1 = list2.foldr (fun c acc => acc * (letter_value c)) 1) →
  (list2 = ['N', 'O', 'W', 'Y']) :=
by
  intros h h_eq
  -- Proof steps here
  sorry

end four_letter_product_eq_l562_562769


namespace max_sequence_gt_44_l562_562579

def sequence (a : ℕ → ℕ → ℝ) : ℕ → ℕ → ℝ
| 0, k => a 0 k
| (n + 1), k => if k < 2016 then a n k + 1 / (2 * a n (k + 1))
                else a n 2016 + 1 / (2 * a n 1)

theorem max_sequence_gt_44 
  (a : ℕ → ℕ → ℝ)
  (h : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 2016 → 0 < a 0 k)
  : (∃ k, 1 ≤ k ∧ k ≤ 2016 ∧ sequence a 2016 k > 44) :=
sorry

end max_sequence_gt_44_l562_562579


namespace exists_student_attended_at_least_8_sessions_l562_562527

theorem exists_student_attended_at_least_8_sessions :
  ∀ (students : Finset ℕ) (sessions : Finset (Finset ℕ)),
  students.card = 50 →
  (∀ (s t : ℕ), s ∈ students → t ∈ students → s ≠ t → ∃! (sess : Finset ℕ), {s, t} ⊆ sess ∧ sess ∈ sessions) →
  ¬ ∃ (sess : Finset ℕ), students ⊆ sess →
  ∃ (s ∈ students), (∃ session ∈ sessions, s ∈ session) ∧ (Finset.filter (λ (session : Finset ℕ), s ∈ session) sessions).card ≥ 8 :=
sorry

end exists_student_attended_at_least_8_sessions_l562_562527


namespace solve_values_of_x_l562_562568

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 1

theorem solve_values_of_x (x : ℝ) :
  f(f(x)) = f(x) ↔ 
  (x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 ∨
   x = (11 + Real.sqrt 101) / 2 ∨ x = (11 - Real.sqrt 101) / 2) := 
sorry

end solve_values_of_x_l562_562568


namespace amount_returned_l562_562553

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l562_562553


namespace marikas_father_age_twice_in_2036_l562_562951

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l562_562951


namespace total_population_of_stratified_sampling_l562_562716

noncomputable def total_population (sample_size : ℕ) (selection_probability : ℚ) : ℕ :=
  sample_size / selection_probability

theorem total_population_of_stratified_sampling :
  ∀ (sample_size : ℕ) (selection_probability : ℚ), 
  selection_probability = 1/12 → 
  sample_size = 10 → 
  total_population sample_size selection_probability = 120 :=
by
  intros sample_size selection_probability hp hs
  rw [hp, hs]
  -- normally you would provide the detailed proof here
  -- but as per instructions, not needed and so we use sorry
  sorry

end total_population_of_stratified_sampling_l562_562716


namespace amount_returned_l562_562552

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l562_562552


namespace reflection_of_lines_over_perpendiculars_l562_562919

variables {A B C D O Q : Type*}
variables [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] [euclidean_geometry D] [euclidean_geometry O] [euclidean_geometry Q]

theorem reflection_of_lines_over_perpendiculars
  (hAB_perpendicular_CD : is_perpendicular AB CD)
  (hO_intersection : intersects AB CD O)
  (hQ_second_intersection : second_intersection_point (circumcircle A O C) (circumcircle B O D) Q) :
  are_reflections_over OP OQ AB ∧ are_reflections_over OP OQ CD :=
sorry

end reflection_of_lines_over_perpendiculars_l562_562919


namespace marians_new_balance_l562_562926

theorem marians_new_balance :
  ∀ (initial_balance grocery_cost return_amount : ℝ),
    initial_balance = 126 →
    grocery_cost = 60 →
    return_amount = 45 →
    let gas_cost := grocery_cost / 2 in
    let new_balance_before_returns := initial_balance + grocery_cost + gas_cost in
    new_balance_before_returns - return_amount = 171 :=
begin
  intros initial_balance grocery_cost return_amount h_init h_grocery h_return,
  let gas_cost := grocery_cost / 2,
  let new_balance_before_returns := initial_balance + grocery_cost + gas_cost,
  have h_gas : gas_cost = 30,
  { 
    rw h_grocery, 
    norm_num },
  have h_new_balance : new_balance_before_returns = 216,
  {
    rw [h_init, h_grocery, h_gas],
    norm_num },
  rw [h_new_balance, h_return],
  norm_num,
end

end marians_new_balance_l562_562926


namespace tangent_line_eq_l562_562242

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562242


namespace distance_DC_l562_562619

def side_small_square := 8 / 4
def side_large_square := Real.sqrt 25
def horizontal_dist := side_small_square + side_large_square
def vertical_dist := side_large_square - side_small_square

theorem distance_DC :
  Real.sqrt (horizontal_dist^2 + vertical_dist^2) = 7.6 :=
sorry

end distance_DC_l562_562619


namespace absolute_value_inequality_solution_l562_562198

theorem absolute_value_inequality_solution (x : ℝ) :
  abs ((3 * x + 2) / (x + 2)) > 3 ↔ (x < -2) ∨ (-2 < x ∧ x < -4 / 3) :=
by
  sorry

end absolute_value_inequality_solution_l562_562198


namespace solution_set_of_inequality_system_l562_562261

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l562_562261


namespace will_money_left_l562_562654

def initial_money : ℝ := 74
def sweater_cost : ℝ := 9
def tshirt_cost : ℝ := 11
def shoes_cost : ℝ := 30
def hat_cost : ℝ := 5
def socks_cost : ℝ := 4
def refund_percentage : ℝ := 0.85
def discount_percentage : ℝ := 0.1
def tax_percentage : ℝ := 0.05

-- Total cost before returns and discounts
def total_cost_before : ℝ := 
  sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost

-- Refund for shoes
def shoes_refund : ℝ := refund_percentage * shoes_cost

-- New total cost after refund
def total_cost_after_refund : ℝ := total_cost_before - shoes_refund

-- Total cost of remaining items (excluding shoes)
def remaining_items_cost : ℝ := total_cost_before - shoes_cost

-- Discount on remaining items
def discount : ℝ := discount_percentage * remaining_items_cost

-- New total cost after discount
def total_cost_after_discount : ℝ := total_cost_after_refund - discount

-- Sales tax on the final purchase amount
def sales_tax : ℝ := tax_percentage * total_cost_after_discount

-- Final purchase amount with tax
def final_purchase_amount : ℝ := total_cost_after_discount + sales_tax

-- Money left after the final purchase
def money_left : ℝ := initial_money - final_purchase_amount

theorem will_money_left : money_left = 41.87 := by 
  sorry

end will_money_left_l562_562654


namespace marika_father_age_twice_l562_562941

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l562_562941


namespace ellipse_foci_distance_l562_562830

theorem ellipse_foci_distance 
  (x y : ℝ)
  (h : (x^2 / 16) + (y^2 / 9) = 1)
  (F1 F2 A B : ℝ × ℝ)
  (h_f1_foci : F1 = (4, 0))
  (h_f2_foci : F2 = (-4, 0))
  (h_AB_line : (A.1 - F1.1) * (B.2 - F1.2) = (B.1 - F1.1) * (A.2 - F1.2))
  (h_AB_dist : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36) :
  (real.sqrt ((A.1 + 4)^2 + A.2^2) + real.sqrt ((B.1 + 4)^2 + B.2^2)) = 10 :=
sorry

end ellipse_foci_distance_l562_562830


namespace regular_octagon_area_l562_562293

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562293


namespace symmedian_ratio_l562_562710

-- Definitions
structure Triangle (α : Type*) [AddGroup α] :=
(A B C : α)

def symmedian_from_B_to_K (α : Type*) [AddGroup α] (T : Triangle α) : α :=
sorry -- Definition of K, where the symmedian from vertex B intersects AC

-- Theorem statement
theorem symmedian_ratio (α : Type*) [AddGroup α] (T : Triangle α)
  (K : α) (h_K : K = symmedian_from_B_to_K α T) :
  let AK := sorry, -- length of segment AK
      KC := sorry, -- length of segment KC
      AB := sorry, -- length of segment AB
      BC := sorry  -- length of segment BC
  in |AK| / |KC| = |AB|^2 / |BC|^2 :=
sorry

end symmedian_ratio_l562_562710


namespace distinct_z_values_l562_562810

def is_reversal (x y : ℕ) : Prop :=
  let a := x / 100
  let b := (x % 100) / 10
  let c := x % 10
  let x' := 100 * a + 10 * b + c
  let y' := 100 * c + 10 * b + a
  (x' = x ∧ y' = y) ∨ (x' = y ∧ y' = x)

def z (x y : ℕ) : ℕ :=
  if h : x >= y then x - y else y - x

theorem distinct_z_values : ∃! n, 
  n = 9 ∧ 
  ∀ (x y : ℕ), (100 ≤ x ∧ x ≤ 999 ∧ 100 ≤ y ∧ y ≤ 999 ∧ is_reversal x y) → 
    ∃ (zs : Finset ℕ), 
      (∀ (a c : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ c ∧ c ≤ 9) → zs = 
        { 99 * (a - c).natAbs }) 
        ∧ zs.card = n :=
sorry

end distinct_z_values_l562_562810


namespace minimum_framing_feet_needed_l562_562673

theorem minimum_framing_feet_needed :
  let original_width := 4
  let original_height := 6
  let enlarged_width := 4 * original_width
  let enlarged_height := 4 * original_height
  let border := 3
  let total_width := enlarged_width + 2 * border
  let total_height := enlarged_height + 2 * border
  let perimeter := 2 * (total_width + total_height)
  let framing_feet := (perimeter / 12).ceil
  framing_feet = 9 := by
  -- The theorem statement translates given conditions into definitions and finally asserts the result.
  sorry

end minimum_framing_feet_needed_l562_562673


namespace max_area_of_region_S_l562_562532

noncomputable def max_possible_area (radii : List ℝ) : ℝ :=
  let areas := radii.map (λ r, Real.pi * r^2)
  let total_area := (areas.sum)
  -- Subtract overlapped area (area of circle with radius 3 in this case)
  let overlapped_area := Real.pi * 3^2
  total_area - overlapped_area

theorem max_area_of_region_S : max_possible_area [1, 3, 5, 7] = 65 * Real.pi := by
  sorry

end max_area_of_region_S_l562_562532


namespace no_simplified_solution_exists_l562_562815

theorem no_simplified_solution_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : sqrt (x^2 + sqrt (y / z)) = y^2 * sqrt (x / z)) :
  ¬∃ (c : ℕ), some_condition c :=
by sorry

end no_simplified_solution_exists_l562_562815


namespace problem1_problem2_l562_562462

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ) -- S for the area of the triangle

noncomputable def angleC (a b c S : ℝ) (h : a^2 + b^2 - c^2 = 4 * S) : Prop :=
  C = π / 4

noncomputable def valueRange (A B : ℝ) (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 4 * 1/2 * a * b * sin (π / 4)) (h2 : c = sqrt 2) : Prop :=
  -1 < sqrt 2 * sin (A - π / 4) ∧ sqrt 2 * sin (A - π / 4) < sqrt 2

theorem problem1 (h : a^2 + b^2 - c^2 = 4 * S) : angleC a b c S h :=
by
  sorry

theorem problem2 (h1 : a^2 + b^2 - c^2 = 4 * 1/2 * a * b * sin (π / 4)) (h2 : c = sqrt 2) : valueRange A B a b c h1 h2 :=
by
  sorry

end problem1_problem2_l562_562462


namespace value_of_m_l562_562104

theorem value_of_m (m : ℤ) : 
  (∃ f : ℤ → ℤ, ∀ x : ℤ, x^2 - (m+1)*x + 1 = (f x)^2) → (m = 1 ∨ m = -3) := 
by
  sorry

end value_of_m_l562_562104


namespace minimum_value_is_3_l562_562483

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  1 / a + 2 / b

theorem minimum_value_is_3 :
  ∃ (a b : ℝ), (x^2 + y^2 - 2 * x - 4 * y + 3 = 0) ∧ (a > 0) ∧ (b > 0) ∧ (a + 2 * b = 3) ∧ (minimum_value a b = 3) :=
begin
  sorry
end

end minimum_value_is_3_l562_562483


namespace maximal_acute_triangles_l562_562666

theorem maximal_acute_triangles (n : ℕ) : 
  ∀ lines (h : set (Π (l : ℕ), l ≤ 2*n + 1)),
  (∀ {a b c : ℕ} (ha : a ∈ h) (hb : b ∈ h) (hc : c ∈ h), a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬ (right_angle a b c)) →
  (2*n + 1 > 0) →
  count_acute_triangles(lines) = (binom (2*n + 2) 3) / 4 :=
sorry

end maximal_acute_triangles_l562_562666


namespace find_k_l562_562019

open Real

noncomputable def k_value (θ : ℝ) : ℝ :=
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 - 2 * (tan θ ^ 2 + 1 / tan θ ^ 2) 

theorem find_k (θ : ℝ) (h : θ ≠ 0 ∧ θ ≠ π / 2 ∧ θ ≠ π) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k_value θ → k_value θ = 6 :=
by
  sorry

end find_k_l562_562019


namespace required_framing_feet_l562_562683

-- Definition of the original picture dimensions
def original_width : ℕ := 4
def original_height : ℕ := 6

-- Definition of the enlargement factor
def enlargement_factor : ℕ := 4

-- Definition of the border width
def border_width : ℕ := 3

-- Given the enlarged and bordered dimensions, calculate the required framing in feet
theorem required_framing_feet : 
  let enlarged_width := enlargement_factor * original_width
  let enlarged_height := enlargement_factor * original_height
  let bordered_width := enlarged_width + 2 * border_width
  let bordered_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (bordered_width + bordered_height)
  let perimeter_feet := (perimeter_inches + 11) / 12
  perimeter_feet = 9 :=
begin
  sorry
end

end required_framing_feet_l562_562683


namespace degrees_to_minutes_l562_562348

theorem degrees_to_minutes (d : ℚ) (fractional_part : ℚ) (whole_part : ℤ) :
  1 ≤ d ∧ d = fractional_part + whole_part ∧ fractional_part = 0.45 ∧ whole_part = 1 →
  (whole_part + fractional_part) * 60 = 1 * 60 + 27 :=
by { sorry }

end degrees_to_minutes_l562_562348


namespace total_distance_l562_562336

noncomputable def walk_time : ℝ := 60 / 60
noncomputable def walking_speed : ℝ := 3.5
noncomputable def run_time : ℝ := 45 / 60
noncomputable def running_speed : ℝ := 8
noncomputable def break_time : ℝ := 15 / 60
noncomputable def distance_walked : ℝ := walking_speed * walk_time
noncomputable def distance_run : ℝ := running_speed * run_time

theorem total_distance (wt : walk_time = 1) (ws : walking_speed = 3.5) (rt : run_time = 0.75) (rs : running_speed = 8) (bt : break_time = 0.25) : 
  distance_walked + distance_run = 9.5 :=
by
  rw [wt, ws, rt, rs]
  simp [distance_walked, distance_run]
  repeat {rw mul_comm}
  simp [walk_time, run_time, walking_speed, running_speed]
  norm_num
  sorry

end total_distance_l562_562336


namespace father_twice_marika_age_in_2036_l562_562949

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l562_562949


namespace remainder_division_l562_562327

-- Definition of the number in terms of its components
def num : ℤ := 98 * 10^6 + 76 * 10^4 + 54 * 10^2 + 32

-- The modulus
def m : ℤ := 25

-- The given problem restated as a hypothesis and goal
theorem remainder_division : num % m = 7 :=
by
  sorry

end remainder_division_l562_562327


namespace sum_first_2n_terms_of_cn_l562_562542

theorem sum_first_2n_terms_of_cn :
  ∀ (a_n b_n c_n : ℕ → ℤ) (q : ℤ) (n : ℕ),
    (∀ n : ℕ, a_n = 3^n) →
    (∀ n : ℕ, b_n = 2 * n + 1) →
    (∀ n : ℕ, c_n = (-1)^n * b_n + a_n) →
    q ≠ 1 →
    b_1 = a_1 →
    b_4 = a_2 →
    b_{13} = a_3 →
    (∑ k in finset.range (2 * n + 1), c_k) = 2 * n + (3^(2 * n + 1) - 3) / 2 :=
by
  sorry

end sum_first_2n_terms_of_cn_l562_562542


namespace exists_quadrilaterals_l562_562418

/-- Exist two quadrilaterals such that the sides of the first are smaller than the corresponding sides of the second, 
    but the corresponding diagonals are larger. -/
theorem exists_quadrilaterals :
  ∃ (Q₁ Q₂ : Quadrilateral),
    (∀ i, side_length Q₁ i < side_length Q₂ i) ∧
    (∀ j, diagonal_length Q₁ j > diagonal_length Q₂ j) :=
sorry

end exists_quadrilaterals_l562_562418


namespace maximum_marks_l562_562338

theorem maximum_marks (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (passing_marks : ℝ) (total_marks : ℝ) :
  passing_percentage = 0.5 → marks_obtained = 50 → marks_failed_by = 10 → passing_marks = marks_obtained + marks_failed_by ∧ passing_marks / passing_percentage = total_marks → total_marks = 120 := 
by 
  intros h1 h2 h3 ⟨h4, h5⟩
  sorry

end maximum_marks_l562_562338


namespace prove_temperature_on_Thursday_l562_562742

def temperature_on_Thursday 
  (temps : List ℝ)   -- List of temperatures for 6 days.
  (avg : ℝ)          -- Average temperature for the week.
  (sum_six_days : ℝ) -- Sum of temperature readings for 6 days.
  (days : ℕ := 7)    -- Number of days in the week.
  (missing_day : ℕ := 1)  -- One missing day (Thursday).
  (thurs_temp : ℝ := 99.8) -- Temperature on Thursday to be proved.
: Prop := (avg * days) - sum_six_days = thurs_temp

theorem prove_temperature_on_Thursday 
  : temperature_on_Thursday [99.1, 98.2, 98.7, 99.3, 99, 98.9] 99 593.2 :=
by
  sorry

end prove_temperature_on_Thursday_l562_562742


namespace isosceles_triangle_perimeter_l562_562805

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : a = 3 ∨ a = 6)
  (h3 : b = 3 ∨ b = 6) (h4 : c = 3 ∨ c = 6) (h5 : a + b + c = 15) : a + b + c = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l562_562805


namespace union_of_M_and_N_l562_562586

def M : set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : set ℝ := {x | x * (x - 1) ≤ 0}

theorem union_of_M_and_N : M ∪ N = {x | -1/2 < x ∧ x ≤ 1} := 
by sorry

end union_of_M_and_N_l562_562586


namespace valid_x_for_expression_l562_562518

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end valid_x_for_expression_l562_562518


namespace two_digit_factors_count_l562_562501

-- Definition of the expression 10^8 - 1
def expr : ℕ := 10^8 - 1

-- Factorization of 10^8 - 1
def factored_expr : List ℕ := [73, 137, 101, 11, 3^2]

-- Define the condition for being a two-digit factor
def is_two_digit (n : ℕ) : Bool := n > 9 ∧ n < 100

-- Count the number of positive two-digit factors in the factorization of 10^8 - 1
def num_two_digit_factors : ℕ := List.length (factored_expr.filter is_two_digit)

-- The theorem stating our proof problem
theorem two_digit_factors_count : num_two_digit_factors = 2 := by
  sorry

end two_digit_factors_count_l562_562501


namespace max_sum_at_n_is_6_l562_562464

-- Assuming an arithmetic sequence a_n where a_1 = 4 and d = -5/7
def arithmetic_seq (n : ℕ) : ℚ := (33 / 7) - (5 / 7) * n

-- Sum of the first n terms (S_n) of the arithmetic sequence {a_n}
def sum_arithmetic_seq (n : ℕ) : ℚ := (n / 2) * (2 * (arithmetic_seq 1) + (n - 1) * (-5 / 7))

theorem max_sum_at_n_is_6 
  (a_1 : ℚ) (d : ℚ) (h1 : a_1 = 4) (h2 : d = -5/7) :
  ∀ n : ℕ, sum_arithmetic_seq n ≤ sum_arithmetic_seq 6 :=
by
  sorry

end max_sum_at_n_is_6_l562_562464


namespace problem_I5_1_l562_562102

theorem problem_I5_1 (a : ℝ) (h : a^2 - 8^2 = 12^2 + 9^2) : a = 17 := 
sorry

end problem_I5_1_l562_562102


namespace solution_set_of_inequality_system_l562_562260

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l562_562260


namespace z_is_7i_if_purely_imaginary_l562_562049

-- Define the complex number z as in the given problem
noncomputable def z (b : ℝ) : ℂ := complex.I * (1 + b * complex.I) + 2 + 3 * b * complex.I

-- Define the predicate for purely imaginary numbers
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Given the condition on b
variable (b : ℝ)

-- The theorem statement
theorem z_is_7i_if_purely_imaginary (h : is_purely_imaginary (z b)) : z b = 7 * complex.I := by
  sorry

end z_is_7i_if_purely_imaginary_l562_562049


namespace range_of_t_l562_562476

noncomputable def f (x t : ℝ) : ℝ := x^3 - t * x^2 + 3 * x

theorem range_of_t 
  (h₀ : ∀ x ∈ set.Icc (1 : ℝ) (4 : ℝ), deriv (f x t) ≤ 0) : t ∈ set.Ici (51 / 8) :=
by 
  sorry

end range_of_t_l562_562476


namespace complex_eq_solution_l562_562491

theorem complex_eq_solution (x y : ℝ) (h : (x + y) + (y - 1) * complex.I = (2 * x + 3 * y) + (2 * y + 1) * complex.I) : 
  x + y = 2 :=
begin
  sorry
end

end complex_eq_solution_l562_562491


namespace find_general_term_l562_562825

theorem find_general_term (p : ℤ) (a_n : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n^2 + p * n) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  let a_2 := a_n 2 in let a_5 := a_n 5 in let a_10 := a_n 10 in
  (a_2 * a_2 = a_5 * a_10) →
  a_n = λ n, 2 * n + 5 :=
by
  sorry

end find_general_term_l562_562825


namespace find_quadratic_coefficients_l562_562614
noncomputable def quadratic_coefficients (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), (4 * x^2 + 1 - 6 * x = a * x^2 + b * x + c)

theorem find_quadratic_coefficients :
  quadratic_coefficients 4 (-6) 1 :=
by
-- Proof to be provided
sory

end find_quadratic_coefficients_l562_562614


namespace martha_apples_l562_562171

theorem martha_apples (martha_initial_apples : ℕ) (jane_apples : ℕ) 
  (james_additional_apples : ℕ) (target_martha_apples : ℕ) :
  martha_initial_apples = 20 →
  jane_apples = 5 →
  james_additional_apples = 2 →
  target_martha_apples = 4 →
  (let james_apples := jane_apples + james_additional_apples in
   let martha_remaining_apples := martha_initial_apples - jane_apples - james_apples in
   martha_remaining_apples - target_martha_apples = 4) :=
begin
  sorry
end

end martha_apples_l562_562171


namespace tangent_line_eq_l562_562243

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562243


namespace parallelogram_area_l562_562157

open Matrix

def v : ℝ × ℝ := (7, -4)
def w : ℝ × ℝ := (3, 1)
def u : ℝ × ℝ := (-1, 5)
def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem parallelogram_area : 
  let v_plus_u := vector_add v u in
  abs (v_plus_u.1 * w.2 - v_plus_u.2 * w.1) = 3 :=
by
  let v_plus_u := vector_add v u
  sorry

end parallelogram_area_l562_562157


namespace inequality_abc_l562_562963

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^3) / (a^2 + a * b + b^2) + (b^3) / (b^2 + b * c + c^2) + (c^3) / (c^2 + c * a + a^2) ≥ (a + b + c) / 3 := 
by
    sorry

end inequality_abc_l562_562963


namespace problem_I_problem_II_i_problem_II_ii_l562_562832

-- Problem I
theorem problem_I (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = (a * log x - b * exp x) / x) (h1 : a ≠ 0)
  (h2 : b = 0) (h3 : ∀ x, 0 < x → deriv f x = (a * (1 - log x)) / x^2)
  (h4 : ∀ x ∈ Ioo (0 : ℝ) e, deriv f x > 0 ∧ ∀ x ∈ Ioo e (⊤ : ℝ), deriv f x < 0) :
  a < 0 := sorry

-- Problem II(i)
theorem problem_II_i (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  ∀ x > 0, x * ((a * log x - b * exp x) / x) + 2 < 0 := sorry

-- Problem II(ii)
theorem problem_II_ii (a b : ℝ) (h1 : a = 1) (h2 : b = -1) :
  ∃ m, (∀ x > 1, x * ((a * log x - b * exp x) / x) > e + m * (x - 1)) ↔ m ≤ 1 + exp 1 := sorry

end problem_I_problem_II_i_problem_II_ii_l562_562832


namespace range_of_a_l562_562862

def f (a x : ℝ) := a * (x - 2) * Real.exp x + Real.log x - x

theorem range_of_a 
  (h_unique_extremum : ∀ a x, differentiable (f a) x → (f a)' x = 0 → (x = 1)) 
  (h_extremum_lt_0 : ∀ a, f a 1 < 0) : 
  ∀ a, a ∈ set.Icc (-1 / Real.exp 1) 0 := by
sorry

end range_of_a_l562_562862


namespace complex_pow_plus_inv_l562_562473

-- Define the problem conditions and theorem
theorem complex_pow_plus_inv (z : ℂ) (h : z + 1/z = 2 * Complex.cos(Real.pi / 36)) :
  z^600 + 1/z^600 = -1 :=
sorry

end complex_pow_plus_inv_l562_562473


namespace acute_angles_sum_pi_over_2_l562_562813

theorem acute_angles_sum_pi_over_2 {α β : ℝ} 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h : (sin α)^4 / (cos β)^2 + (cos α)^4 / (sin β)^2 = 1) : 
  α + β = π / 2 :=
sorry

end acute_angles_sum_pi_over_2_l562_562813


namespace rectangle_area_with_circles_l562_562718

theorem rectangle_area_with_circles (r : ℝ) (h_pos : r = 3) :
  let d := 2 * r,
      length := 2 * d,
      width := d,
      area := length * width
  in area = 72 :=
by
  let d := 2 * r
  let length := 2 * d
  let width := d
  let area := length * width
  have length_def : length = 12 := by
    calc length = 2 * d : by rfl
           ... = 2 * (2 * r) : by rfl
           ... = 2 * 2 * r : by rfl
           ... = 4 * r : by ring
           ... = 4 * 3 : by simp [←h_pos]
           ... = 12 : by linarith
  have width_def : width = 6 := by
    calc width = d : by rfl
          ... = 2 * r : by rfl
          ... = 2 * 3 : by simp [←h_pos]
          ... = 6 : by linarith
  have area_def : area = 72 := by
    calc area = length * width : by rfl
          ... = 12 * 6 : by simp [length_def, width_def]
          ... = 72 : by norm_num
  exact area_def

end rectangle_area_with_circles_l562_562718


namespace calculate_expression_l562_562757

theorem calculate_expression :
  36 + (150 / 15) + (12 ^ 2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end calculate_expression_l562_562757


namespace z_unique_m_range_l562_562797

noncomputable def complex_z := {z : ℂ // (z + 2*ℂ.I).im = 0 ∧ (z / (2 - ℂ.I)).im = 0}
noncomputable def z_value : ℂ := 4 - 2*ℂ.I

theorem z_unique (h : complex_z) : h.val = z_value :=
sorry

noncomputable def fourth_quadrant_point (m : ℝ) : Prop :=
  let z := 4 - 2*ℂ.I in
  let z1 := complex.conj z + (1 / (m - 1)) - (7 / (m + 2)) * ℂ.I in
  z1.re > 0 ∧ z1.im < 0

theorem m_range : { m : ℝ // fourth_quadrant_point m } = 
  { x : ℝ // (-2 < x ∧ x < (3 / 4)) ∨ ((1 < x) ∧ (x < (3 / 2))) } :=
sorry

end z_unique_m_range_l562_562797


namespace angle_in_second_quadrant_l562_562110

theorem angle_in_second_quadrant (θ : ℝ) (h1 : sin (2 * θ) < 0) (h2 : 2 * cos θ < 0) : 
  (π / 2 < θ ∧ θ < π) :=
by
  sorry

end angle_in_second_quadrant_l562_562110


namespace framing_required_l562_562679

theorem framing_required
  (initial_width : ℕ)
  (initial_height : ℕ)
  (scale_factor : ℕ)
  (border_width : ℕ)
  (increments : ℕ)
  (initial_width_def : initial_width = 4)
  (initial_height_def : initial_height = 6)
  (scale_factor_def : scale_factor = 4)
  (border_width_def : border_width = 3)
  (increments_def : increments = 12) :
  Nat.ceil ((2 * (4 * scale_factor  + 2 * border_width + 6 * scale_factor + 2 * border_width).toReal) / increments) = 9 := by
  sorry

end framing_required_l562_562679


namespace tangent_line_to_curve_l562_562230

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562230


namespace sphere_volume_correct_l562_562806

noncomputable def sphereVolume (A B : ℝ) (O : point) (AB : ℝ) (angleAOB : ℝ) : ℝ := 
  if AB = 3 ∧ angleAOB = 120 then 4 * π * sqrt(3) 
  else 0

/--
  Given points A and B on the surface of a sphere, with O as the center of the sphere,
  AB = 3, and ∠AOB = 120°, then the volume of the sphere is 4π√3.
-/
theorem sphere_volume_correct :
  ∀ (A B O : point),
  ∀ (AB : ℝ),
  ∀ (angleAOB : ℝ),
  AB = 3 →
  angleAOB = 120 →
  sphereVolume A B O AB angleAOB = 4 * π * sqrt(3) :=
by
  intros
  -- Proof goes here
  sorry

end sphere_volume_correct_l562_562806


namespace log_equivalence_l562_562855

theorem log_equivalence (x : ℝ) (h : log 16 (x - 3) = 1 / 4) : log 256 x = (1 / 8) * log 2 x :=
by
  sorry

end log_equivalence_l562_562855


namespace math_problem_l562_562624

theorem math_problem
  (z : ℝ)
  (hz : z = 80)
  (y : ℝ)
  (hy : y = (1/4) * z)
  (x : ℝ)
  (hx : x = (1/3) * y)
  (w : ℝ)
  (hw : w = x + y + z) :
  x = 20 / 3 ∧ w = 320 / 3 :=
by
  sorry

end math_problem_l562_562624


namespace tangent_line_equation_l562_562227

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562227


namespace solve_for_w_l562_562196

theorem solve_for_w (w : ℂ) (i : ℂ) (i_squared : i^2 = -1) 
  (h : 3 - i * w = 1 + 2 * i * w) : 
  w = -2 * i / 3 := 
sorry

end solve_for_w_l562_562196


namespace opposite_of_neg3_l562_562998

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l562_562998


namespace tan_double_angle_l562_562818

theorem tan_double_angle (θ : ℝ) 
  (h1 : θ ≠ π / 2)
  (h2 : θ ≠ -π / 2)
  (h3 : ∃ (x : ℝ), x ≠ 0 ∧ tan θ = 1 / 2) : 
  tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l562_562818


namespace union_result_l562_562153

-- Define the problem context and variables
variable (a b : ℝ)

def M := {a, b}
def N := {a + 1, 3}

-- Problem condition
axiom h_inter : M ∩ N = {2}

-- Prove the desired union result
theorem union_result : M ∪ N = {1, 2, 3} := by
  sorry

end union_result_l562_562153


namespace length_of_bridge_proof_l562_562727

-- Definitions for the conditions
def length_of_train : ℝ := 170
def speed_kmh : ℝ := 45
def time_seconds : ℝ := 30
def length_of_bridge : ℝ := 205

-- Conversion from km/h to m/s
def speed_ms := (speed_kmh * 1000) / 3600

-- Distance calculation using speed and time
def total_distance := speed_ms * time_seconds

-- Proof goal: length of the bridge
theorem length_of_bridge_proof :
  length_of_bridge = total_distance - length_of_train :=
  by
  sorry

end length_of_bridge_proof_l562_562727


namespace calculate_XY_l562_562132

-- Definition of the problem conditions
def is_right_triangle (X Y Z : ℝ) := (X^2 + Y^2 = Z^2)
def angle_X_is_right (X : ℝ) := (X = 90)
def triangle_tan_Z (XY XZ : ℝ) (YZ: ℝ) := (XY / XZ)
def triangle_sin_Y (XY YZ : ℝ) := (XY / YZ)

-- Main theorem
theorem calculate_XY {XY XZ YZ : ℝ} 
  (h1 : angle_X_is_right 90)
  (h2 : YZ = 20)
  (h3 : triangle_tan_Z XY XZ YZ = 3 * triangle_sin_Y XY YZ) :
  XY = (40 * (real.sqrt 2) / 3) :=
by
  sorry

end calculate_XY_l562_562132


namespace area_of_inscribed_regular_octagon_l562_562302

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562302


namespace sum_of_f_values_l562_562466

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem sum_of_f_values 
  (f : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hf_periodic : ∀ x, f (2 - x) = f x)
  (hf_neg_one : f (-1) = 1) :
  f 1 + f 2 + f 3 + f 4 + (502 * (f 1 + f 2 + f 3 + f 4)) = -1 := 
sorry

end sum_of_f_values_l562_562466


namespace opposite_of_neg3_l562_562997

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l562_562997


namespace regular_octagon_area_l562_562320

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562320


namespace find_a_l562_562823

variable (a b c : ℤ)

theorem find_a (h1 : a + b = 2) (h2 : b + c = 0) (h3 : |c| = 1) : a = 3 ∨ a = 1 := 
sorry

end find_a_l562_562823


namespace max_sin_2x_sin_x_cos_x_l562_562782

open Real

theorem max_sin_2x_sin_x_cos_x : 
  (∀ x ∈ Icc (0 : ℝ) (π / 2), sin (2 * x) + sin x - cos x ≤ 5 / 4) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (π / 2), sin (2 * x) + sin x - cos x = 5 / 4) :=
by
  sorry

end max_sin_2x_sin_x_cos_x_l562_562782


namespace infinite_sum_evaluates_to_constant_l562_562441

noncomputable def euler_totient_function (n : ℕ) : ℕ :=
  -- Definition of Euler totient function
  nat.totient n

theorem infinite_sum_evaluates_to_constant : 
  (∑ n in (set.Ici 1), (euler_totient_function n * 4^n) / (7^n - 4^n)) = (28 / 9) :=
by
  sorry

end infinite_sum_evaluates_to_constant_l562_562441


namespace solution_set_l562_562080

noncomputable def f : ℝ → ℝ
| x := if x < 0 then (2 + x) / x else log 2 (1 / x)

theorem solution_set (x : ℝ) : f x + 2 ≤ 0 ↔ (-2/3 ≤ x ∧ x < 0) ∨ (4 ≤ x) := 
by {
  sorry -- The proof is left as an exercise
}

end solution_set_l562_562080


namespace rearrangement_impossible_l562_562888

-- Define the primary problem conditions and goal
theorem rearrangement_impossible :
  ¬ ∃ (f : Fin 100 → Fin 51), 
    (∀ k : Fin 51, ∃ i j : Fin 100, 
      f i = k ∧ f j = k ∧ (i < j ∧ j.val - i.val = k.val + 1)) :=
sorry

end rearrangement_impossible_l562_562888


namespace problem_solution_l562_562645

theorem problem_solution :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 :=
by sorry

end problem_solution_l562_562645


namespace bond_selling_price_l562_562559

theorem bond_selling_price:
  ∀ (face_value interest_rate selling_rate : ℝ),
    face_value = 5000 →
    interest_rate = 0.06 →
    selling_rate = 0.065 →
    let interest := face_value * interest_rate in
    let selling_price := interest / selling_rate in
    selling_price ≈ 4615.38 := 
by {
  sorry
}

end bond_selling_price_l562_562559


namespace negation_proposition_l562_562250

open Set

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 + 2 * x + 5 > 0) → (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) :=
sorry

end negation_proposition_l562_562250


namespace max_monotonic_a_3_l562_562453

noncomputable def maxMonotonicA (a : ℝ) (h : a > 0) : ℝ :=
  if ∀ x : ℝ, x ≥ 1 → (3 * x^2 - a) ≥ 0 then a else 0

theorem max_monotonic_a_3 : maxMonotonicA 3 (by norm_num) = 3 := sorry

end max_monotonic_a_3_l562_562453


namespace boys_playing_both_l562_562529

theorem boys_playing_both (total B F N BF : ℕ)
    (h_total : total = 22)
    (h_B : B = 13)
    (h_F : F = 15)
    (h_N : N = 3)
    (h_total_eq : total = B + F - BF + N) :
    BF = 9 :=
by
  have h := h_total_eq
  rw [h_total, h_B, h_F, h_N] at h
  linarith

end boys_playing_both_l562_562529


namespace constantin_mother_deposit_return_l562_562557

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l562_562557


namespace trigonometric_identity_proof_l562_562571

variable (x y : ℝ)

theorem trigonometric_identity_proof 
  (h1 : sin x / sin y = 4) 
  (h2 : cos x / cos y = 1 / 2) 
  (h3 : x = 2 * y) : 
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 15 / 14 := 
by
  sorry

end trigonometric_identity_proof_l562_562571


namespace bret_time_reading_book_l562_562754

def total_train_ride_time : ℕ := 9
def time_spent_eating_dinner : ℕ := 1
def time_spent_watching_movies : ℕ := 3
def time_planned_for_nap : ℕ := 3

def time_spent_reading_book := total_train_ride_time - (time_spent_eating_dinner + time_spent_watching_movies + time_planned_for_nap)

theorem bret_time_reading_book : time_spent_reading_book = 2 :=
by 
  -- Calculate the time spent reading the book
  calc time_spent_reading_book
     = 9 - (1 + 3 + 3) : rfl
 ... = 9 - 7         : by norm_num
 ... = 2             : by norm_num

end bret_time_reading_book_l562_562754


namespace find_a_eq_sqrt2_l562_562009

theorem find_a_eq_sqrt2 :
  ∃ a : ℝ, (∫ x in 0..(π / 4), (sin x - a * cos x)) = - (sqrt 2) / 2 ∧ a = sqrt 2 :=
by
  sorry

end find_a_eq_sqrt2_l562_562009


namespace spent_on_puzzle_l562_562772

-- Defining all given conditions
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def final_amount : ℕ := 1

-- Define the total money before spending on the puzzle
def total_before_puzzle := initial_money + saved_money - spent_on_comic

-- Prove that the amount spent on the puzzle is $18
theorem spent_on_puzzle : (total_before_puzzle - final_amount) = 18 := 
by {
  sorry
}

end spent_on_puzzle_l562_562772


namespace number_of_people_per_taxi_l562_562253

def num_people_in_each_taxi (x : ℕ) (cars taxis vans total : ℕ) : Prop :=
  (cars = 3 * 4) ∧ (vans = 2 * 5) ∧ (total = 58) ∧ (taxis = 6 * x) ∧ (cars + vans + taxis = total)

theorem number_of_people_per_taxi
  (x cars taxis vans total : ℕ)
  (h1 : cars = 3 * 4)
  (h2 : vans = 2 * 5)
  (h3 : total = 58)
  (h4 : taxis = 6 * x)
  (h5 : cars + vans + taxis = total) :
  x = 6 :=
by
  sorry

end number_of_people_per_taxi_l562_562253


namespace number_of_men_in_first_group_l562_562199

theorem number_of_men_in_first_group
    (M : ℕ) 
    (work_completion_in_30_days : M * 30 = 15 * 36) :
    M = 18 :=
begin
  sorry -- proof will be added later
end

end number_of_men_in_first_group_l562_562199


namespace problem_l562_562488

def f (x : ℝ) : ℝ := (1 / (Real.exp x + 1)) - 1 / 2

theorem problem (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = (1 / (Real.exp x + 1)) - 1 / 2) →
  (∀ x : ℝ, f (-x) = -f(x)) ∧ (∀ x y : ℝ, x < y → f x > f y) :=
by
  sorry

end problem_l562_562488


namespace awards_distribution_count_l562_562031

-- Define the problem conditions
def num_awards : Nat := 5
def num_students : Nat := 3

-- Verify each student gets at least one award
def each_student_gets_at_least_one (distributions : List (List Nat)) : Prop :=
  ∀ (dist : List Nat), dist ∈ distributions → (∀ (d : Nat), d > 0)

-- Define the main theorem to be proved
theorem awards_distribution_count :
  ∃ (distributions : List (List Nat)), each_student_gets_at_least_one distributions ∧ distributions.length = 150 :=
sorry

end awards_distribution_count_l562_562031


namespace max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l562_562448

theorem max_transition_channel_BC (hB: (1500: ℝ) * 0.042 = 63) (hC1: (4536: ℝ) / 72 = 63):
  max 63 63 = (63: ℝ) :=
by {
  simp [*, max_def];
}

theorem lowest_cost_per_transition (hA: (3417: ℝ) / 51 = 67):
  (67: ℝ) ≤ 78 :=
by {
  linarith,
}

theorem highest_profit_from_channel_C (hC_sales: (63: ℝ) * 0.05 = 3.15) (rounded_sales_C: ⌊3.15⌋ = 3) 
  (sale_revenue_C: 3 * 2500 = 7500) (total_cost_C: 4536):
  7500 - 4536 = (2964: ℝ) :=
by {
  norm_num,
}

end max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l562_562448


namespace jill_clothing_percentage_l562_562183

variable (T : ℝ) -- Total amount spent
variable (C : ℝ) -- Percentage spent on clothing, in decimal
variable (food : ℝ := 0.25 * T) -- Amount spent on food
variable (other : ℝ := 0.25 * T) -- Amount spent on other items
variable (total_tax : ℝ := 0.1 * T) -- Total tax paid
variable (tax_clothes : ℝ := 0.1 * (C * T)) -- Tax on clothing
variable (tax_food : ℝ := 0) -- Tax on food
variable (tax_other : ℝ := 0.2 * (0.25 * T)) -- Tax on other items

theorem jill_clothing_percentage :
  0.1 * T = 0.1 * (C * T) + 0 + 0.05 * T → C = 0.5 :=
by
  intros h
  have : 0.1 * T = 0.1 * (C * T) + 0.05 * T := by exact h
  have : 0.05 * T = 0.1 * (C * T) := by linarith
  have : 0.05 = 0.1 * C := by linarith
  have : C = 0.5 := by linarith
  exact this

end jill_clothing_percentage_l562_562183


namespace general_term_formula_sum_first_n_terms_l562_562056

-- Given conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℤ} (h1 : a 2 = 0) (h2 : a 6 + a 8 = -10)
  (ha : arithmetic_sequence a)

-- Proving the general term formula matches
theorem general_term_formula (n : ℕ) :
  (∃ a₀ d : ℤ, (a 1 = a₀ + d ∧ a 2 = a₀ + 2 * d ∧ -- initial conditions
                a (n + 1) = 2 - n)) :=
sorry

-- Proving the sum of the first n terms of sequence {an * 3^(n-1)}
theorem sum_first_n_terms (n : ℕ) :
  let a_n := λ (n : ℕ), 2 - n in
  (∑ i in Finset.range n, (a_n i) * 3 ^ i =
   (5 / 4 - n / 2) * 3 ^ n - 5 / 4) :=
sorry

end general_term_formula_sum_first_n_terms_l562_562056


namespace binary_sum_correct_l562_562728

-- Definitions of the binary numbers
def bin1 : ℕ := 0b1011
def bin2 : ℕ := 0b101
def bin3 : ℕ := 0b11001
def bin4 : ℕ := 0b1110
def bin5 : ℕ := 0b100101

-- The statement to prove
theorem binary_sum_correct : bin1 + bin2 + bin3 + bin4 + bin5 = 0b1111010 := by
  sorry

end binary_sum_correct_l562_562728


namespace photo_gallery_total_l562_562924

theorem photo_gallery_total (initial_photos: ℕ) (first_day_photos: ℕ) (second_day_photos: ℕ)
  (h_initial: initial_photos = 400) 
  (h_first_day: first_day_photos = initial_photos / 2)
  (h_second_day: second_day_photos = first_day_photos + 120) : 
  initial_photos + first_day_photos + second_day_photos = 920 :=
by
  sorry

end photo_gallery_total_l562_562924


namespace count_score_scenarios_l562_562547

-- Define the rules and conditions as premises
structure Game :=
(score_jia : ℕ)
(score_yi : ℕ)
(valid : (score_jia = 11 ∧ score_yi < 10) ∨ (score_yi = 11 ∧ score_jia < 10)
          ∨ (score_jia ≥ 10 ∧ score_yi ≥ 10 ∧ |score_jia - score_yi| ≥ 2))

def valid_set_of_games (total_points : ℕ) (num_games : ℕ) (points_to_win : ℕ) : Set (List Game) :=
{ g | length g = num_games ∧ (∀ game, game ∈ g → Game.valid game) ∧ (sum (map Game.score_jia g) + sum (map Game.score_yi g) = total_points) }

-- State the theorem
theorem count_score_scenarios : 
  ∃ s, s ∈ valid_set_of_games 30 3 11 ∧ s.card = 8 :=
sorry

end count_score_scenarios_l562_562547


namespace fixed_monthly_fee_l562_562192

theorem fixed_monthly_fee (x y : ℝ)
  (h1 : x + y = 15.80)
  (h2 : x + 3 * y = 28.62) :
  x = 9.39 :=
sorry

end fixed_monthly_fee_l562_562192


namespace fractional_part_wall_in_12_minutes_l562_562107

-- Definitions based on given conditions
def time_to_paint_wall : ℕ := 60
def time_spent_painting : ℕ := 12

-- The goal is to prove that the fraction of the wall Mark can paint in 12 minutes is 1/5
theorem fractional_part_wall_in_12_minutes (t_pw: ℕ) (t_sp: ℕ) (h1: t_pw = 60) (h2: t_sp = 12) : 
  (t_sp : ℚ) / (t_pw : ℚ) = 1 / 5 :=
by 
  sorry

end fractional_part_wall_in_12_minutes_l562_562107


namespace exists_zero_point_f_max_value_k_l562_562060

-- Define the given functions
def f (x : ℝ) := x - Real.log x - 2
def g (x : ℝ) := x * Real.log x + x

-- 1. Prove that f(x) has a zero in the interval (3, 4)
theorem exists_zero_point_f : ∃ c ∈ Set.Ioo 3 4, f c = 0 := sorry

-- 2. Find the maximum value of k ∈ ℤ such that ∀ x > 1, g(x) > k * (x - 1)
theorem max_value_k : ∀ x > 1, g x > (3 : ℤ) * (x - 1) := sorry

end exists_zero_point_f_max_value_k_l562_562060


namespace probability_prime_sum_correct_l562_562690

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_ways_to_form_sum (s : ℕ) : ℕ :=
  ∑ i in finset.Icc 1 6, (if 1 ≤ s - i ∧ s - i ≤ 8 then 1 else 0)

def number_of_ways_to_get_prime_sum : ℕ :=
  ∑ p in finset.filter is_prime (finset.Icc 2 14), count_ways_to_form_sum p

def total_outcomes : ℕ := 6 * 8

def probability_prime_sum : ℚ :=
  number_of_ways_to_get_prime_sum / total_outcomes

-- Theorem that needs proving
theorem probability_prime_sum_correct :
  probability_prime_sum = 11 / 24 :=
sorry

end probability_prime_sum_correct_l562_562690


namespace triangle_bc_correct_l562_562886

noncomputable def triangle_side_length_bc : Prop :=
  ∀ (A B C D : Type) (AB AC AD BC : ℝ) 
    (h1 : AB = 6) 
    (h2 : AC = 9) 
    (h3 : AD = 5) 
    (h4 : D = (B + C) / 2)
    (h5 : AD = 1 / 2 * sqrt (2 * AB^2 + 2 * AC^2 - BC^2)), 
  BC = sqrt 134

theorem triangle_bc_correct : triangle_side_length_bc :=
by {
  intros A B C D AB AC AD BC h1 h2 h3 h4 h5,
  have : 10 = sqrt (234 - BC^2), from by {
    rw [h1, h2, h3, h5],
    norm_num,
  },
  have : BC^2 = 134, by {
    rw this,
    norm_num,
  },
  exact eq_of_sq_eq_sq this,
  sorry
}

end triangle_bc_correct_l562_562886


namespace count_distinct_rat_k_l562_562435

theorem count_distinct_rat_k : 
  (∃ N : ℕ, N = 108 ∧ ∀ k : ℚ, abs k < 300 → (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0) →
  (∃! k, abs k < 300 ∧ (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0))) :=
sorry

end count_distinct_rat_k_l562_562435


namespace required_framing_feet_l562_562682

-- Definition of the original picture dimensions
def original_width : ℕ := 4
def original_height : ℕ := 6

-- Definition of the enlargement factor
def enlargement_factor : ℕ := 4

-- Definition of the border width
def border_width : ℕ := 3

-- Given the enlarged and bordered dimensions, calculate the required framing in feet
theorem required_framing_feet : 
  let enlarged_width := enlargement_factor * original_width
  let enlarged_height := enlargement_factor * original_height
  let bordered_width := enlarged_width + 2 * border_width
  let bordered_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (bordered_width + bordered_height)
  let perimeter_feet := (perimeter_inches + 11) / 12
  perimeter_feet = 9 :=
begin
  sorry
end

end required_framing_feet_l562_562682


namespace area_regular_octagon_in_circle_l562_562281

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562281


namespace count_complex_numbers_l562_562580

noncomputable def f (z : ℂ) : ℂ := z^2 - 2 * (complex.I * z) + 2

theorem count_complex_numbers :
  let s := {z : ℂ | 0 < z.im ∧ (∃ (a b : ℤ), abs a ≤ 10 ∧ abs b ≤ 10 ∧ f(z).re = a ∧ f(z).im = b)} in
  s.to_finset.card = 251 :=
begin
  sorry
end

end count_complex_numbers_l562_562580


namespace negation_of_universal_statement_l562_562166

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 ≤ 0 :=
sorry

end negation_of_universal_statement_l562_562166


namespace eq_different_solution_l562_562331

theorem eq_different_solution (x : ℝ) :
  let base_eq : x - 3 = 3x + 4,
      sol := -7 / 2
  in (7 * x - 4) * (x - 1) = (5 * x - 11) * (x - 1) →
     ∃ x₀ : ℝ, x₀ ≠ sol ∧ (7 * x₀ - 4) * ( x₀ - 1) = (5 * x₀ - 11) * ( x₀ - 1) :=
by
  intro base_eq sol eq_D
  sorry

end eq_different_solution_l562_562331


namespace probability_prime_sum_correct_l562_562689

-- Definitions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def count_ways_to_form_sum (s : ℕ) : ℕ :=
  ∑ i in finset.Icc 1 6, (if 1 ≤ s - i ∧ s - i ≤ 8 then 1 else 0)

def number_of_ways_to_get_prime_sum : ℕ :=
  ∑ p in finset.filter is_prime (finset.Icc 2 14), count_ways_to_form_sum p

def total_outcomes : ℕ := 6 * 8

def probability_prime_sum : ℚ :=
  number_of_ways_to_get_prime_sum / total_outcomes

-- Theorem that needs proving
theorem probability_prime_sum_correct :
  probability_prime_sum = 11 / 24 :=
sorry

end probability_prime_sum_correct_l562_562689


namespace vector_conditions_l562_562565

def vector := ℝ × ℝ × ℝ

def cross_prod (u v : vector) : vector :=
  (u.2.2 * v.3 - u.3 * v.2.2,
   u.3 * v.1 - u.1 * v.3,
   u.1 * v.2.2 - u.2.2 * v.1)

def a : vector := (2, 3, 1)
def b : vector := (-1, 1, 2)
def v : vector := (0, 5, 5)

theorem vector_conditions :
  (cross_prod v a = (2:ℝ) • (cross_prod b a)) ∧
  (cross_prod v b = cross_prod a b) :=
by
  sorry

end vector_conditions_l562_562565


namespace loss_representation_l562_562407

-- Define the condition for profit representation
def profit_representation (amount : Int) : Int :=
  amount

-- State the theorem for loss representation
theorem loss_representation (loss_amount : Int) (profit_amount : Int) : 
  profit_representation profit_amount = profit_amount → 
  loss_amount = -30 → 
  loss_amount = -profit_representation 30 := 
by
  intro h_profit h_loss
  rw [←h_loss]
  exact h_profit

end loss_representation_l562_562407


namespace martha_apples_l562_562174

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l562_562174


namespace train_length_l562_562381

/-- 
Given the speed of the train is 50 km/hr and it crosses a pole in 18 seconds, 
prove that the length of the train is 250 meters.
-/
theorem train_length :
  ∀ (speed_km_hr : ℝ) (time_sec : ℝ), 
  speed_km_hr = 50 → time_sec = 18 →
  let speed_m_s := (speed_km_hr * 1000) / 3600 in
  let length := speed_m_s * time_sec in
  length = 250 :=
by
  intros speed_km_hr time_sec hs ht
  rw [hs, ht]
  have speed_m_s := (50 * 1000) / 3600
  have length := speed_m_s * 18
  have h_speed : speed_m_s = 50 * 1000 / 3600 := rfl
  rw h_speed at length
  have h_length : length = (50 * 1000 / 3600) * 18 := rfl
  have length_val : 50 * 1000 / 3600 * 18 = 250 := sorry   -- skipping the actual arithmetic proof
  rw ← h_length at length_val
  exact length_val

end train_length_l562_562381


namespace find_lambda_l562_562474

-- Definitions and conditions
variables {α : Type*} [linear_ordered_field α]
variables {P A B C : α → α}
variables {λ : α}
variables (triangle : α → α → α × α × α)
variables (tan : α → α)
variables (circumcenter : α × α × α → α)
variables (midpoint : α × α → α)
variables (vector_add : α → α → α)
variables (vector_eq : α → α → α → Prop)

-- Given Conditions
def P_is_circumcenter (triangle : α → α → α × α × α) (P : α → α) : Prop := 
  P = circumcenter (triangle 0 0)

def vectors_add_up (PA PB PC : α) (λ : α) : Prop := 
  vector_eq (vector_add PA PB) λ PC

def tan_C_condition (P A B C : α → α) : Prop := 
  tan C = 12 / 5

-- Lean Statement
theorem find_lambda (P A B C : α → α) (λ : α)
  (h1 : P_is_circumcenter triangle P)
  (h2 : vectors_add_up (P A) (P B) (P C) λ)
  (h3 : tan_C_condition P A B C) :
  λ = -10 / 13 :=
sorry

end find_lambda_l562_562474


namespace sum_of_surface_areas_l562_562761

noncomputable def cube_sequence : ℕ → ℝ 
| 0       := 6 * 1^2
| (n + 1) := 6 * (1/3 ^ (n+1))^2

noncomputable def octa_sequence : ℕ → ℝ 
| 0       := real.sqrt 3
| (n + 1) := real.sqrt 3 * (1/3 ^ (n+1))^2

noncomputable def S_cubes : ℝ :=
 ∑' n, cube_sequence n

noncomputable def S_octahedrons : ℝ :=
 ∑' n, octa_sequence n

noncomputable def S_total : ℝ :=
 S_cubes + S_octahedrons

theorem sum_of_surface_areas : S_total = (54 + 9 * real.sqrt 3) / 8 :=
by 
  sorry

end sum_of_surface_areas_l562_562761


namespace deadlift_weight_loss_is_200_l562_562894

def initial_squat : ℕ := 700
def initial_bench : ℕ := 400
def initial_deadlift : ℕ := 800
def lost_squat_percent : ℕ := 30
def new_total : ℕ := 1490

theorem deadlift_weight_loss_is_200 : initial_deadlift - (new_total - ((initial_squat * (100 - lost_squat_percent)) / 100 + initial_bench)) = 200 :=
by
  sorry

end deadlift_weight_loss_is_200_l562_562894


namespace f_270_l562_562908

noncomputable def f : ℕ → ℕ :=
sorry -- We assume the existence of such a function f for now.

axiom f_property : ∀ x y : ℕ, f(x * y) = f(x) + f(y)
axiom f_30 : f(30) = 21
axiom f_90 : f(90) = 27

theorem f_270 : f(270) = 33 :=
by {
  -- We state the theorem and provide a placeholder proof.
  sorry
}

end f_270_l562_562908


namespace Marian_credit_card_balance_l562_562928

theorem Marian_credit_card_balance :
  let initial_balance := 126.00 in
  let groceries := 60.00 in
  let gas := groceries / 2 in
  let returned := 45.00 in
  initial_balance + groceries + gas - returned = 171.00 :=
by
  let initial_balance := 126.00
  let groceries := 60.00
  let gas := groceries / 2
  let returned := 45.00
  calc
    126.00 + 60.00 + 30.00 - 45.00 = 216.00 - 45.00 : by congr
    ... = 171.00 : by norm_num

#suppressAllProofSteps

end Marian_credit_card_balance_l562_562928


namespace smallest_n_for_perfect_square_subsets_l562_562920
open Set Int

def S : Set ℤ := {m | m > 0 ∧ ∀ p, prime p → p ∣ m → p < 10}

theorem smallest_n_for_perfect_square_subsets :
  ∃ n, ∀ A ⊆ S, (A.card ≥ n) → ∃ x y z w ∈ A, (((x * y * z * w) ^ (2:ℤ)) = (x * y * z * w)) ∧ x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ n = 9 :=
sorry

end smallest_n_for_perfect_square_subsets_l562_562920


namespace acute_angle_between_diagonals_leq_60_l562_562369

theorem acute_angle_between_diagonals_leq_60 
  (A B C D : Type) [affine_space A] [vector_space ℝ A]
  (points : A → A → Prop) (bisect : A → A → A)
  (isosceles_trapezoid : A → A → A → A → Prop)
  (AB AD BC CD : ℝ):
  isosceles_trapezoid A B C D ∧ (AD = AB + CD) →
  acute_angle_between_diagonals A B C D ≤ 60 :=
begin
  sorry
end

end acute_angle_between_diagonals_leq_60_l562_562369


namespace find_n_l562_562767

noncomputable def is_prime (p : ℕ) : Prop := nat.prime p

theorem find_n :
  ∃ n : ℕ, 
    let p1 := 2,
        p2 := 5 in
    (p1 * p1 != 0) ∧ (p2 * p2 != 0) ∧
    is_prime p1 ∧ is_prime p2 ∧
    (2 * p1 * p1 - 8 * n * p1 + 10 * p1 - n^2 + 35 * n - 76 = 0) ∧
    (2 * p2 * p2 - 8 * n * p2 + 10 * p2 - n^2 + 35 * n - 76 = 0) ∧
    p1 + p2 = 4 * n - 5 ∧
    n = 3 :=
by
  sorry

end find_n_l562_562767


namespace ratio_of_speeds_l562_562005

def distance_AB := 570 -- distance between city A and city B in kms
def distance_AC := 300 -- distance between city A and city C in kms
def time_Eddy := 3 -- time taken by Eddy to travel in hours
def time_Freddy := 4 -- time taken by Freddy to travel in hours

theorem ratio_of_speeds :
  let speed_Eddy := distance_AB / time_Eddy in
  let speed_Freddy := distance_AC / time_Freddy in
  let ratio := speed_Eddy / speed_Freddy in
  ratio = 38 / 15 := 
  by sorry

end ratio_of_speeds_l562_562005


namespace hexagon_coloring_l562_562771

-- Definition of the problem conditions and the required proof.
theorem hexagon_coloring:
  let colors := {1, 2, 3, 4, 5, 6} in
  let vertices := {A, B, C, D, E, F} in
  (∀ (A B : vertices), A ≠ B → ∀ (c : colors), ∃ (cA cB : colors), cA ≠ cB) →
  (∀ (x y : vertices), adjacent x y → x ≠ y → ∀ (c : colors), ∃ (cx cy : colors), cx ≠ cy) →
  (∀ (u v : vertices), diagonal u v → u ≠ v → ∀ (cu cv : colors), cu ≠ cv) →
  ∃ (count : ℕ), count = 9600 := sorry

end hexagon_coloring_l562_562771


namespace additional_rows_added_l562_562374

theorem additional_rows_added
  (initial_tiles : ℕ) (initial_rows : ℕ) (initial_columns : ℕ) (new_columns : ℕ) (new_rows : ℕ)
  (h1 : initial_tiles = 48)
  (h2 : initial_rows = 6)
  (h3 : initial_columns = initial_tiles / initial_rows)
  (h4 : new_columns = initial_columns - 2)
  (h5 : new_rows = initial_tiles / new_columns) :
  new_rows - initial_rows = 2 := by sorry

end additional_rows_added_l562_562374


namespace tangent_line_equation_at_point_l562_562211

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562211


namespace largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l562_562022

theorem largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative :
  ∃ (n : ℤ), (4 < n) ∧ (n < 7) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_such_that_n_squared_minus_11n_plus_28_is_negative_l562_562022


namespace volleyball_team_selection_l562_562594

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

end volleyball_team_selection_l562_562594


namespace prove_k_eq_5_l562_562585

variable (a b k : ℕ)

theorem prove_k_eq_5 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (a^2 - 1 - b^2) / (a * b - 1) = k) : k = 5 :=
sorry

end prove_k_eq_5_l562_562585


namespace discriminant_sufficient_not_necessary_real_roots_l562_562582

variable (a b c : ℝ) (h : a ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem discriminant_sufficient_not_necessary_real_roots :
  (discriminant a b c > 0) → (∃ (x : ℝ), a * x^2 + b * x + c = 0) ∧ ¬ (∃ (x : ℝ), a * x^2 + b * x + c = 0 → discriminant a b c ≤ 0) := 
begin
  sorry
end

end discriminant_sufficient_not_necessary_real_roots_l562_562582


namespace cake_volume_icing_area_sum_l562_562704

-- Define the conditions based on the problem description
def cube_edge_length : ℕ := 4
def volume_of_piece := 16
def icing_area := 12

-- Define the statements to be proven
theorem cake_volume_icing_area_sum : 
  volume_of_piece + icing_area = 28 := 
sorry

end cake_volume_icing_area_sum_l562_562704


namespace capital_payment_l562_562358

theorem capital_payment (m : ℕ) (hm : m ≥ 3) : 
  ∃ d : ℕ, d = (1000 * (3^m - 2^(m-1))) / (3^m - 2^m) 
  ∧ (∃ a : ℕ, a = 4000 ∧ a = ((3/2)^(m-1) * (3000 - 3 * d) + 2 * d)) := 
by
  sorry

end capital_payment_l562_562358


namespace part_I_part_II_l562_562836

-- Define the function f
def f (a x : ℝ) : ℝ := a * (Real.log x - 1) + 1 / x

-- Define the function g
def g (b x : ℝ) : ℝ := (b - 1) * Real.logb b x - (x^2 - 1) / 2

-- Conditions
variable (a x : ℝ)
variable (hx0 : f a x = 0) (hx1 : f' a x = 0)

-- Theorem statement for part (I)
theorem part_I (a x : ℝ) : f a x ≤ (x - 1)^2 / x := 
sorry

-- Conditions for part (II)
variable (b : ℝ)
variable (hx_in : 1 < x ∧ x < Real.sqrt b)

-- Theorem statement for part (II)
theorem part_II (b x : ℝ) (hx_in : 1 < x ∧ x < Real.sqrt b) : 0 < g b x ∧ g b x < (b - 1)^2 / 2 :=
sorry

end part_I_part_II_l562_562836


namespace person_in_middle_car_is_sharon_l562_562383

-- Define the seats
inductive Seat : Type
| car1 | car2 | car3 | car4 | car5

-- Define the people
inductive Person : Type
| Aaron | Darren | Karen | Maren | Sharon

open Seat Person

-- Conditions as hypotheses
variables {pos : Person → Seat}
hypothesis h1 : pos Maren = car5
hypothesis h2 : ∃ s : Seat, pos Aaron = s ∧ pos Sharon = Seat.car (s + 1)
hypothesis h3 : ∃ d : Seat, pos Darren = d ∧ d < (choose s, pos Aaron = s)
hypothesis h4 : ∃ k m : Seat, pos Karen = k ∧ pos Darren = m ∧ abs (k - m) ≥ 1

-- Proof statement
theorem person_in_middle_car_is_sharon : pos Sharon = car3 :=
sorry

end person_in_middle_car_is_sharon_l562_562383


namespace credit_card_balance_l562_562932

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l562_562932


namespace concyclic_points_l562_562129

variables {A B C H P Q : Type} [EuclideanGeometry A B C H P Q]

-- Let's define the triangles, circles, and points
variables (triangle_ABC : RightTriangle A B C)
variables (altitude_CH : Altitude C H (line AB))
variables (incircle_O1 : Incircle (Triangle.mk A H C))
variables (incircle_O2 : Incircle (Triangle.mk B H C))
variables (common_tangent_P : CommonTangent incircle_O1 incircle_O2 (line AC) P)
variables (common_tangent_Q : CommonTangent incircle_O1 incircle_O2 (line BC) Q)

-- The proof problem in Lean statement
theorem concyclic_points (h : ∀ {X}, IsRightAngle (angle C X B)) :
  Concyclic P A B Q :=
sorry

end concyclic_points_l562_562129


namespace concentration_after_removal_l562_562366

/-- 
Given:
1. A container has 27 liters of 40% acidic liquid.
2. 9 liters of water is removed from this container.

Prove that the concentration of the acidic liquid in the container after removal is 60%.
-/
theorem concentration_after_removal :
  let initial_volume := 27
  let initial_concentration := 0.4
  let water_removed := 9
  let pure_acid := initial_concentration * initial_volume
  let new_volume := initial_volume - water_removed
  let final_concentration := (pure_acid / new_volume) * 100
  final_concentration = 60 :=
by {
  sorry
}

end concentration_after_removal_l562_562366


namespace points_can_be_arranged_l562_562048

variables (n : ℕ) (A : fin n → Point)
  (h_n : n ≥ 3)
  (h_conditions : ∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k → ∃ l, angle A i A j A k = l ∧ l ≥ 120)

theorem points_can_be_arranged :
  ∃ (f : fin n → Point), ∀ (i j k : fin n), i < j ∧ j < k → angle (f i) (f j) (f k) ≥ 120 :=
sorry

end points_can_be_arranged_l562_562048


namespace min_days_to_plant_trees_l562_562720

theorem min_days_to_plant_trees (n : ℕ) (h : 2 ≤ n) :
  (2 ^ (n + 1) - 2 ≥ 1000) ↔ (n ≥ 9) :=
by sorry

end min_days_to_plant_trees_l562_562720


namespace no_real_roots_range_of_a_l562_562088

theorem no_real_roots_range_of_a (a : ℝ) : ¬ ∃ x : ℝ, a * x^2 + 6 * x + 1 = 0 ↔ a > 9 :=
by
  sorry

end no_real_roots_range_of_a_l562_562088


namespace probability_heads_greater_tails_l562_562706

noncomputable def probability_more_heads_than_tails : ℚ :=
  (4.choose 3) * (1/2)^4 + (4.choose 4) * (1/2)^4

theorem probability_heads_greater_tails : 
  probability_more_heads_than_tails = 5 / 16 :=
by
  sorry

end probability_heads_greater_tails_l562_562706


namespace range_of_t_l562_562520

-- Define the points A and B
def A (t : ℝ) := (1 - t, 1 + t)
def B (t : ℝ) := (3, 2 * t)

-- Define the slope of line AB
def slope_AB (t : ℝ) := ((1 + t) - (2 * t)) / ((1 - t) - 3)

-- Lean statement for the problem
theorem range_of_t (t : ℝ) : (slope_AB t < 0) → -2 < t ∧ t < 1 :=
  sorry

end range_of_t_l562_562520


namespace line_circle_interaction_l562_562111

theorem line_circle_interaction (a : ℝ) :
  let r := 10
  let d := |a| / 5
  let intersects := -50 < a ∧ a < 50 
  let tangent := a = 50 ∨ a = -50 
  let separate := a < -50 ∨ a > 50 
  (d < r ↔ intersects) ∧ (d = r ↔ tangent) ∧ (d > r ↔ separate) :=
by sorry

end line_circle_interaction_l562_562111


namespace sum_of_all_distinct_products_l562_562252

theorem sum_of_all_distinct_products :
  ∃ G H : ℕ, (G < 10) ∧ (H < 10) ∧ (54 * 10^6 + G * 10^5 + 1 * 10^4 + 5 * 10^3 + 0 * 10^2 + H * 10 + 726) % 72 = 0 →
  (G + H) = 6 ∧ (finset.image (λ p : ℕ × ℕ, p.1 * p.2) { (G, H) ∈ finset.filter (λ p, p.1 + p.2 = 6) (finset.univ.product finset.univ) }).sum = 22 := by
  sorry

end sum_of_all_distinct_products_l562_562252


namespace number_of_integers_satisfying_condition_l562_562783

theorem number_of_integers_satisfying_condition :
  {n : ℕ // n > 0 ∧ (⌊(2012 : ℝ) / n⌋ - ⌊(2012 : ℝ) / (n + 1)⌋ = 1)}.subtype.card = 52 :=
sorry

end number_of_integers_satisfying_condition_l562_562783


namespace recurring_decimal_fraction_l562_562011

theorem recurring_decimal_fraction :
  let a := 0.714714714...
  let b := 2.857857857...
  (a / b) = (119 / 476) :=
by
  let a := (714 / (999 : ℝ))
  let b := (2856 / (999 : ℝ))
  sorry

end recurring_decimal_fraction_l562_562011


namespace calculate_integral_cos8_l562_562396

noncomputable def integral_cos8 : ℝ :=
  ∫ x in (Real.pi / 2)..(2 * Real.pi), 2^8 * (Real.cos x)^8

theorem calculate_integral_cos8 :
  integral_cos8 = 219 * Real.pi :=
by
  sorry

end calculate_integral_cos8_l562_562396


namespace min_value_of_f_at_sqrt2_l562_562764

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x)))

theorem min_value_of_f_at_sqrt2 :
  f (Real.sqrt 2) = (11 * Real.sqrt 2) / 6 :=
sorry

end min_value_of_f_at_sqrt2_l562_562764


namespace find_a_l562_562519

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ (x : ℝ), f(x) = (1/3) * x^3 - a * x^2 + 1) 
  (h_max : ∀ (x : ℝ), ∃ c, c = -4 → f(-4) ≥ f(x)) :
  a = -2 :=
sorry

end find_a_l562_562519


namespace minimum_value_l562_562062

noncomputable def min_value (a b c d : ℝ) : ℝ :=
(a - c) ^ 2 + (b - d) ^ 2

theorem minimum_value (a b c d : ℝ) (hab : a * b = 3) (hcd : c + 3 * d = 0) :
  min_value a b c d ≥ (18 / 5) :=
by
  sorry

end minimum_value_l562_562062


namespace sin_of_point_on_circle_l562_562074

theorem sin_of_point_on_circle :
  let α : ℝ in
  let P : ℝ × ℝ := (-1 / 2, sqrt 3 / 2) in
  let y := P.2 in
  let r := sqrt ((P.1)^2 + (P.2)^2) in
  y / r = sqrt 3 / 2 :=
by
  sorry

end sin_of_point_on_circle_l562_562074


namespace billy_reads_60_pages_per_hour_l562_562752

theorem billy_reads_60_pages_per_hour
  (free_time_per_day : ℕ)
  (days : ℕ)
  (video_games_time_percentage : ℝ)
  (books : ℕ)
  (pages_per_book : ℕ)
  (remaining_time_percentage : ℝ)
  (total_free_time := free_time_per_day * days)
  (time_playing_video_games := video_games_time_percentage * total_free_time)
  (time_reading := remaining_time_percentage * total_free_time)
  (total_pages := books * pages_per_book)
  (pages_per_hour := total_pages / time_reading) :
  free_time_per_day = 8 →
  days = 2 →
  video_games_time_percentage = 0.75 →
  remaining_time_percentage = 0.25 →
  books = 3 →
  pages_per_book = 80 →
  pages_per_hour = 60 :=
by
  intros
  sorry

end billy_reads_60_pages_per_hour_l562_562752


namespace original_expenditure_l562_562340

theorem original_expenditure (initial_students new_students : ℕ) (increment_expense : ℝ) (decrement_avg_expense : ℝ) (original_avg_expense : ℝ) (new_avg_expense : ℝ) 
  (total_initial_expense original_expenditure : ℝ)
  (h1 : initial_students = 35) 
  (h2 : new_students = 7) 
  (h3 : increment_expense = 42)
  (h4 : decrement_avg_expense = 1)
  (h5 : new_avg_expense = original_avg_expense - decrement_avg_expense)
  (h6 : total_initial_expense = initial_students * original_avg_expense)
  (h7 : original_expenditure = total_initial_expense)
  (h8 : 42 * new_avg_expense - original_students * original_avg_expense = increment_expense) :
  original_expenditure = 420 := 
by
  sorry

end original_expenditure_l562_562340


namespace maximum_distance_l562_562126

-- Define the circle and line equations in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop := ρ^2 + 2 * ρ * Real.cos θ - 3 = 0
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ + ρ * Real.sin θ - 7 = 0

-- Convert the circle's polar equation to Cartesian coordinates
def cartesian_circle (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 3 = 0

-- Convert the line's polar equation to Cartesian coordinates
def cartesian_line (x y : ℝ) : Prop := x + y - 7 = 0

-- Describe the center and radius of the circle
def circle_center : (ℝ × ℝ) := (-1, 0)
def circle_radius : ℝ := 2

-- Define the distance formula from a point to a line in Cartesian coordinates
def point_to_line_distance (x0 y0 a b c : ℝ) : ℝ := abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

-- Describe the maximum distance from a point on the circle to the line
def max_distance_from_circle_to_line : ℝ := 4 * sqrt 2 + 2

-- Prove the maximum distance from a point on the circle to the line
theorem maximum_distance :
  @max_distance_from_circle_to_line = distance_from_point_to_line + circle_radius :=
by {
  -- Define the line equation in Cartesian form
  let a := (1 : ℝ),
  let b := (1 : ℝ),
  let c := -7,

  -- Distance from the center of circle to the line
  let distance_center_to_line := point_to_line_distance (-1) 0 a b c,

  -- The maximum distance is the sum of the center-line distance and the circle's radius
  have center_line_distance : distance_center_to_line = 4 * sqrt 2,
  have circle_radius : circle_radius = 2,
  have total_distance := distance_center_to_line + circle_radius,
  
  -- Prove the maximum distance equals 4 * sqrt 2 + 2
  have max_distance := 4 * sqrt 2 + circle_radius,
  sorry
}

end maximum_distance_l562_562126


namespace correct_proposition_l562_562061

variables {m n l : Line} {α β γ : Plane}

def proposition_1 := (α ⊥ β ∧ α ⊥ γ) → (β ∥ γ)
def proposition_2 := (α ∥ β ∧ α ∥ γ) → (β ∥ γ)
def proposition_3 := (l ⊥ m ∧ l ⊥ n) → (m ∥ n)
def proposition_4 := (m ∥ α ∧ m ∥ n) → (n ∥ α)

theorem correct_proposition : proposition_2 :=
by sorry

end correct_proposition_l562_562061


namespace soccer_balls_percentage_holes_l562_562591

variable (x : ℕ)

theorem soccer_balls_percentage_holes 
    (h1 : ∃ x, 0 ≤ x ∧ x ≤ 100)
    (h2 : 48 = 80 * (100 - x) / 100) : 
  x = 40 := sorry

end soccer_balls_percentage_holes_l562_562591


namespace taxi_ride_cost_l562_562376

noncomputable def base_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def discount : ℝ := 1.00
noncomputable def miles : ℝ := 12.00

def total_cost (base_cost cost_per_mile discount miles : ℝ) : ℝ :=
  let cost := base_cost + miles * cost_per_mile
  if miles > 10 then cost - discount else cost

theorem taxi_ride_cost :
  total_cost base_cost cost_per_mile discount miles = 4.60 := by
  sorry

end taxi_ride_cost_l562_562376


namespace sin_cos_eq_one_l562_562777

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h1 : x < 2 * Real.pi) :
  sin x + cos x = 1 → x = 0 ∨ x = Real.pi / 2 :=
by
  sorry

end sin_cos_eq_one_l562_562777


namespace length_of_floor_correct_l562_562988

noncomputable def length_of_floor (total_cost : ℝ) (rate_per_sq_m : ℝ) (breadth : ℝ) : ℝ :=
  let l := 3 * breadth in
  l

theorem length_of_floor_correct (breadth : ℝ) (total_cost : ℝ) (rate_per_sq_m : ℝ) :
  (3 * breadth * breadth = total_cost / rate_per_sq_m) →
  length_of_floor total_cost rate_per_sq_m breadth = 15.489 :=
by
  sorry

end length_of_floor_correct_l562_562988


namespace probability_FG_l562_562707

def probability_of_D : ℚ := 1 / 4
def probability_of_E : ℚ := 1 / 3
def total_probability : ℚ := 1

theorem probability_FG (P(F) P(G) : ℚ) (h : probability_of_D + probability_of_E + P(F) + P(G) = total_probability) : 
  P(F) + P(G) = 5 / 12 :=
sorry

end probability_FG_l562_562707


namespace set_sum_difference_l562_562339

def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define sets A and B
def setA := {n | 2 ≤ n ∧ n ≤ 50 ∧ is_even n}
def setB := {n | 102 ≤ n ∧ n ≤ 150 ∧ is_even n}

-- Define function to sum elements of a set
def sum_of_set (s : set ℕ) : ℕ := s.to_finset.sum id

-- Define the problem statement
theorem set_sum_difference :
  (sum_of_set setB) - (sum_of_set setA) = 2500 := 
sorry

end set_sum_difference_l562_562339


namespace find_a_for_parallel_lines_l562_562028

def direction_vector_1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a, 3, 2)

def direction_vector_2 : ℝ × ℝ × ℝ :=
  (2, 3, 2)

theorem find_a_for_parallel_lines : ∃ a : ℝ, direction_vector_1 a = direction_vector_2 :=
by
  use 1
  unfold direction_vector_1
  sorry  -- proof omitted

end find_a_for_parallel_lines_l562_562028


namespace sufficient_condition_implies_true_l562_562859

variable {p q : Prop}

theorem sufficient_condition_implies_true (h : p → q) : (p → q) = true :=
by
  sorry

end sufficient_condition_implies_true_l562_562859


namespace part_I_monotonicity_and_extremum_part_II_range_of_a_l562_562834

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x

theorem part_I_monotonicity_and_extremum (a : ℝ) (h : a = -2 * Real.exp 1) : 
  (∀ x : ℝ, 0 < x → x < Real.sqrt (Real.exp 1) → deriv (λ x, f x a) x < 0) ∧ 
  (∀ x : ℝ, Real.sqrt (Real.exp 1) < x → deriv (λ x, f x a) x > 0) ∧ 
  (∃ x : ℝ, x = Real.sqrt (Real.exp 1) ∧ f x a = 0) :=
  sorry

theorem part_II_range_of_a :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → deriv (λ x, f x a) x ≤ 0) → a ≤ -32 :=
  sorry

end part_I_monotonicity_and_extremum_part_II_range_of_a_l562_562834


namespace solve_system_l562_562656

theorem solve_system :
  ∃ (x y : ℝ), 
  (x + y > 0) ∧
  (x + 2y > 0) ∧
  ((0.48^(x^2 + 2))^(2x - y) = 1) ∧
  (log 10 (x + y) - 1 = log 10 6 - log 10 (x + 2y)) ∧
  (x = 2 ∧ y = 4) :=
begin
  sorry
end

end solve_system_l562_562656


namespace mortgage_loan_amount_l562_562643

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end mortgage_loan_amount_l562_562643


namespace cosA_value_value_of_a_l562_562064

variable {a b c A B C : ℝ}

/- Condition 1: a, b, and c are the sides opposite to angles A, B, and C respectively in triangle ABC. -/
constant side_opposite_angles : Prop

/- Condition 2: a cos C + (c - 3b) cos A = 0 -/
def equation1 : Prop := a * Real.cos C + (c - 3 * b) * Real.cos A = 0

/- Question 1: Prove cos A = 1/3 -/
theorem cosA_value : equation1 → Real.cos A = 1/3 := by
  sorry

/- Additional conditions for question 2:
     The area of triangle ABC is sqrt{2}
     b - c = 2
-/
variable (sqrt2_area : Prop) (diff_bc : b - c = 2)

def area : Prop := 1/2 * b * c * Real.sin A = Real.sqrt 2

/- Question 2: Prove a = 2 * sqrt 2 -/
theorem value_of_a : equation1 → sqrt2_area → diff_bc → a = 2 * Real.sqrt 2 := by
  sorry

end cosA_value_value_of_a_l562_562064


namespace f_2015_l562_562916

def f : ℤ → ℤ := sorry

axiom f1 : f 1 = 1
axiom f2 : f 2 = 0
axiom functional_eq (x y : ℤ) : f (x + y) = f x * f (1 - y) + f (1 - x) * f y

theorem f_2015 : f 2015 = 1 ∨ f 2015 = -1 :=
sorry

end f_2015_l562_562916


namespace regular_octagon_area_l562_562291

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562291


namespace total_chrome_parts_l562_562770

theorem total_chrome_parts (a b : ℕ) 
  (h1 : a + b = 21) 
  (h2 : 3 * a + 2 * b = 50) : 2 * a + 4 * b = 68 := 
sorry

end total_chrome_parts_l562_562770


namespace AK_perpendicular_to_BC_l562_562147

variables {A B C M E F K : ℝ}
variables (hABC_angle : ∀ {A B C : Type}, 0 < ∠ABC ∧ ∠ABC < π/2)
variables (hM_midpoint : M = (B + C) / 2)
variables (hE_foot : E = foot B C A)
variables (hF_foot : F = foot C B A)
variables (hK_external_tangents : K ∈ external_tangents (circumcircle B M E) (circumcircle C M F))
variables (hK_on_circumcircle : K ∈ circumcircle A B C)

theorem AK_perpendicular_to_BC :
  ⟦AK⟧ ⊥ ⟦BC⟧
:= sorry

end AK_perpendicular_to_BC_l562_562147


namespace business_value_after_three_months_l562_562712

-- Definitions based on conditions
def owned_fraction : ℝ := 1 / 3
def sold_fraction : ℝ := 3 / 5
def sale_amount : ℝ := 15000
def conversion_rate : ℝ := 74
def monthly_fluctuation : ℝ := 0.05
def months : ℕ := 3

/-- The final value in USD after fluctuations and conversion -/
def final_value_usd : ℝ :=
  let original_value_inr := (sale_amount / sold_fraction) / owned_fraction
  let total_business_value_inr := original_value_inr * 3
  let final_value_inr := total_business_value_inr * (1 + monthly_fluctuation)^months
  final_value_inr / conversion_rate

theorem business_value_after_three_months :
  final_value_usd ≈ 1173.54 :=
sorry

end business_value_after_three_months_l562_562712


namespace sin_values_of_B_l562_562506

open Real

theorem sin_values_of_B (B : ℝ) (h : tan B + csc B = 3) :
  sin B = sqrt ((1 + sqrt 0.6) / 2) ∨ sin B = sqrt ((1 - sqrt 0.6) / 2) :=
by
  sorry

end sin_values_of_B_l562_562506


namespace number_of_values_1_to_50_with_prime_sum_of_divisors_l562_562918

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, 1 < m → m < p → p % m ≠ 0

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).sum id

theorem number_of_values_1_to_50_with_prime_sum_of_divisors :
  (finset.filter (λ n, is_prime (sum_of_divisors n)) (finset.range 51)).card = 9 :=
sorry

end number_of_values_1_to_50_with_prime_sum_of_divisors_l562_562918


namespace unique_handshakes_l562_562349

theorem unique_handshakes (n : ℕ) (k : ℕ) (h_n : n = 12) (h_k : k = 6) : 
  (n * k) / 2 = 36 :=
by 
  rw [h_n, h_k] 
  norm_num

end unique_handshakes_l562_562349


namespace two_digit_numbers_with_perfect_square_sums_l562_562502

theorem two_digit_numbers_with_perfect_square_sums :
  let perfect_square_sums := [1, 4, 9, 16]
  in (finset.filter (λn, ((n / 10) + (n % 10) ∈ perfect_square_sums)) (finset.range (100) \ finset.range (10))).card = 17 :=
by
  let perfect_square_sums := [1, 4, 9, 16]
  have H : (finset.filter (λn, ((n / 10) + (n % 10) ∈ perfect_square_sums)) (finset.range (100) \ finset.range (10))).card = 17
  { sorry, }
  exact H

end two_digit_numbers_with_perfect_square_sums_l562_562502


namespace johnsonville_max_members_l562_562268

theorem johnsonville_max_members 
  (n : ℤ) 
  (h1 : 15 * n % 30 = 6) 
  (h2 : 15 * n < 900) 
  : 15 * n ≤ 810 :=
sorry

end johnsonville_max_members_l562_562268


namespace orchard_trees_l562_562874

theorem orchard_trees (x p : ℕ) (h : x + p = 480) (h2 : p = 3 * x) : x = 120 ∧ p = 360 :=
by
  sorry

end orchard_trees_l562_562874


namespace binomial_distribution_parameters_l562_562073

theorem binomial_distribution_parameters (n p : ℝ) (ξ : ℝ) 
  (h1 : ξ = n * p) (h2 : ξ * (1 - p) = 1.28) : 
  ξ = 1.6 ∧ n = 8 ∧ p = 0.2 :=
by
  have h3 : ξ = 1.6 := h1
  have h4 : p = 0.2 := by sorry
  have h5 : n = 8 := by sorry
  exact ⟨h3, h5, h4⟩

end binomial_distribution_parameters_l562_562073


namespace line_through_points_l562_562029

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : x1 ≠ x2) (hx1 : x1 = -3) (hy1 : y1 = 1) (hx2 : x2 = 1) (hy2 : y2 = 5) :
  ∃ (m b : ℝ), (m + b = 5) ∧ (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) :=
by
  sorry

end line_through_points_l562_562029


namespace triangle_angle_A_triangle_angle_B_l562_562524

-- Given conditions for triangle ABC
variables {A B C : ℝ} {a b c : ℝ}
hypothesis h1 : ∀ {A B C : ℝ}, a = A * cos B
hypothesis h2 : ∀ {A B C : ℝ}, 2 * a * cos B = 2 * c - b

-- Proof 1: Prove the magnitude of angle A is π / 3
theorem triangle_angle_A (h1 : 2 * a * cos B = 2 * c - b) : A = π / 3 :=
sorry

-- Given additional condition for proof 2
def c_eq_2b : Prop := c = 2 * b

-- Proof 2: Prove the magnitude of angle B is π / 6
theorem triangle_angle_B (h1 : 2 * a * cos B = 2 * c - b) (h3 : c_eq_2b) : B = π / 6 :=
sorry

end triangle_angle_A_triangle_angle_B_l562_562524


namespace social_media_usage_in_week_l562_562775

def days_in_week : ℕ := 7
def daily_phone_usage : ℕ := 16
def daily_social_media_usage : ℕ := daily_phone_usage / 2

theorem social_media_usage_in_week :
  daily_social_media_usage * days_in_week = 56 :=
by
  sorry

end social_media_usage_in_week_l562_562775


namespace greatest_divisor_of_arithmetic_sequence_l562_562324

theorem greatest_divisor_of_arithmetic_sequence :
  ∀ (x c : ℤ), (∀ n, 0 ≤ n ∧ n < 15 → x + n * c > 0) →
  15 ∣ ∑ n in finset.range 15, (x + n * c) :=
begin
  intros x c h,
  sorry
end

end greatest_divisor_of_arithmetic_sequence_l562_562324


namespace volume_of_cuboid_l562_562511

theorem volume_of_cuboid (l w h : ℝ) (hlw: l * w = 120) (hwh: w * h = 72) (hhl: h * l = 60) : l * w * h = 720 :=
  sorry

end volume_of_cuboid_l562_562511


namespace sum_of_digits_smallest_n_l562_562161

def gcd (a b : ℕ) : ℕ := a.gcd b

def smallest_n : ℕ :=
  Inf {n : ℕ | n > 200 ∧ gcd 70 (n + 150) = 35 ∧ gcd (n + 70) 150 = 75}

def sum_digits (n : ℕ) : ℕ :=
  n.to_string.foldl (λ acc c => acc + c.to_digit) 0

theorem sum_of_digits_smallest_n : sum_digits smallest_n = 8 := 
by
  sorry

end sum_of_digits_smallest_n_l562_562161


namespace required_framing_feet_l562_562684

-- Definition of the original picture dimensions
def original_width : ℕ := 4
def original_height : ℕ := 6

-- Definition of the enlargement factor
def enlargement_factor : ℕ := 4

-- Definition of the border width
def border_width : ℕ := 3

-- Given the enlarged and bordered dimensions, calculate the required framing in feet
theorem required_framing_feet : 
  let enlarged_width := enlargement_factor * original_width
  let enlarged_height := enlargement_factor * original_height
  let bordered_width := enlarged_width + 2 * border_width
  let bordered_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (bordered_width + bordered_height)
  let perimeter_feet := (perimeter_inches + 11) / 12
  perimeter_feet = 9 :=
begin
  sorry
end

end required_framing_feet_l562_562684


namespace largest_by_changing_first_digit_l562_562652

def value_with_digit_changed (d : Nat) : Float :=
  match d with
  | 1 => 0.86123
  | 2 => 0.78123
  | 3 => 0.76823
  | 4 => 0.76183
  | 5 => 0.76128
  | _ => 0.76123 -- default case

theorem largest_by_changing_first_digit :
  ∀ d : Nat, d ∈ [1, 2, 3, 4, 5] → value_with_digit_changed 1 ≥ value_with_digit_changed d :=
by
  intro d hd_list
  sorry

end largest_by_changing_first_digit_l562_562652


namespace angle_A_60_l562_562137

-- Define the triangle and the given conditions
axiom TriangleABC (A B C : Type) (angle : A → B → C → ℝ) (cos_angle : B → ℝ) :
  angle A B C = 90 ∧ cos_angle B = (sqrt 3) / 2

-- The degree of ∠A
def degree_A : ℝ := 60

-- The proof statement that ∠A = 60°
theorem angle_A_60 (A B C : Type) (angle : A → B → C → ℝ) (cos_angle : B → ℝ) 
  (h : angle A B C = 90) (hc : cos_angle B = (sqrt 3) / 2) : 
  angle A B C + cos_angle B + degree_A = 180 :=
by 
  sorry

end angle_A_60_l562_562137


namespace probability_of_exactly_two_dice_showing_one_l562_562006

theorem probability_of_exactly_two_dice_showing_one :
  (∃ p : ℚ, p = 28 * (1 / 36) * (15625 / 46656) ∧ p ≈ 0.780) :=
sorry

end probability_of_exactly_two_dice_showing_one_l562_562006


namespace elementary_schools_in_Lansing_l562_562558

theorem elementary_schools_in_Lansing (total_students : ℕ) (students_per_school : ℕ) (h1 : total_students = 6175) (h2 : students_per_school = 247) : total_students / students_per_school = 25 := 
by sorry

end elementary_schools_in_Lansing_l562_562558


namespace line_circle_intersection_diff_l562_562052

theorem line_circle_intersection_diff {t : ℝ}
  (param_line : ∀ t, (1 + t * Real.sqrt 3, 1 + t))
  (circle_eq : ∀ x y, x^2 + y^2 = 4) :
  let A := (1 + t₁ * Real.sqrt 3, 1 + t₁),
      B := (1 + t₂ * Real.sqrt 3, 1 + t₂)
  in (Real.dist (1, 1) A - Real.dist (1, 1) B) = Real.sqrt 3 + 1 :=
sorry

end line_circle_intersection_diff_l562_562052


namespace document_review_order_count_l562_562739

theorem document_review_order_count : 
  ∀ {S : set ℕ}, S ⊆ {1, 2, 3, 4, 5, 6, 7, 8} → (∃ n ∈ S, n = 9) → 
  ∃ n ∈ S, n = 10 ∨ n ∉ S → 
  ∑ k in (finset.range 9), finset.card (finset.powerset_len k {1, 2, 3, 4, 5, 6, 7, 8}) * (k + 2) = 1440 :=
sorry

end document_review_order_count_l562_562739


namespace solve_inequality_l562_562415

def p (x : ℝ) : ℝ := x^2 - 5*x + 3

theorem solve_inequality (x : ℝ) : 
  abs (p x) < 9 ↔ (-1 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) :=
sorry

end solve_inequality_l562_562415


namespace max_speeds_l562_562634

-- Define the given masses
def m1 : ℝ := 150 / 1000 -- in kg
def m2 : ℝ := 1 / 1000 -- in kg
def m3 : ℝ := 30 / 1000 -- in kg

-- Initial speed of the small bead
def V : ℝ := 10 -- in m/s

-- No friction and perfectly elastic collisions
-- Use conservation of momentum and kinetic energy equations

theorem max_speeds (V_1 V_3 : ℝ) : 
  (m1 = 150 / 1000) →
  (m2 = 1 / 1000) →
  (m3 = 30 / 1000) →
  (V = 10) →
  (-m1 * V_1 + m3 * V_3 = m2 * V) →
  (m1 * V_1^2 + m3 * V_3^2 = m2 * V^2) →
  (V_1 = 0.28) ∧ (V_3 = 1.71) := sorry

end max_speeds_l562_562634


namespace min_intersection_l562_562789

open Finset

-- Definition of subset count function
def n (S : Finset ℕ) : ℕ :=
  2 ^ S.card

theorem min_intersection {A B C : Finset ℕ} (hA : A.card = 100) (hB : B.card = 100) 
  (h_subsets : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≥ 97 := by
  sorry

end min_intersection_l562_562789


namespace dark_squares_exceed_light_squares_by_one_l562_562672

theorem dark_squares_exceed_light_squares_by_one :
  let dark_squares := 25
  let light_squares := 24
  dark_squares - light_squares = 1 :=
by
  sorry

end dark_squares_exceed_light_squares_by_one_l562_562672


namespace roots_real_and_equal_l562_562765

theorem roots_real_and_equal :
  ∀ x : ℝ,
  (x^2 - 4 * x * Real.sqrt 5 + 20 = 0) →
  (Real.sqrt ((-4 * Real.sqrt 5)^2 - 4 * 1 * 20) = 0) →
  (∃ r : ℝ, x = r ∧ x = r) :=
by
  intro x h_eq h_discriminant
  sorry

end roots_real_and_equal_l562_562765


namespace eventually_all_cards_face_up_l562_562528

theorem eventually_all_cards_face_up
  (cards : List ℕ)
  (h : ∀ d ∈ cards, d = 1 ∨ d = 2)
  (flip : List ℕ → List ℕ)
  (flip_condition : ∀ card : List ℕ, card ≠ [] → card.head! = 2 → card.last! = 2 → 
                    flip card = card.map (λ d, if d = 2 then 1 else d))
  (decrease : ∀ card : List ℕ, card ≠ flip card → card.map (λ d, if d = 2 then 1 else d).toNat < card.toNat)
  (bounded : ∀ card : List ℕ, card.toNat ≥ 0) :
  ∃ n, ∀ k ≥ n, (cards' = flip^[k] cards → ∀ d ∈ cards', d = 1) :=
by
  sorry

end eventually_all_cards_face_up_l562_562528


namespace tangent_line_to_curve_l562_562229

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562229


namespace solution_l562_562854

def problem (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (x + a) * (x - 3) = x^2 + 2 * x - b

theorem solution (a b : ℝ) (h : problem a b) : a - b = -10 :=
  sorry

end solution_l562_562854


namespace range_of_function_l562_562616

theorem range_of_function : ∀ x : ℝ, (2^(-(abs x)) ∈ set.Ioo 0 1 ∨ 2^(-(abs x)) = 1) :=
by
  sorry

end range_of_function_l562_562616


namespace negative_double_inequality_l562_562332

theorem negative_double_inequality (a : ℝ) (h : a < 0) : 2 * a < a :=
by { sorry }

end negative_double_inequality_l562_562332


namespace product_of_roots_l562_562756

theorem product_of_roots : (Real.sqrt 9) * (Real.cbrt 4) = 3 * 2^(2/3) := by
  sorry

end product_of_roots_l562_562756


namespace inequality_abc_lt_l562_562046

variable (a b c : ℝ)

theorem inequality_abc_lt:
  c > b → b > a → a^2 * b + b^2 * c + c^2 * a < a * b^2 + b * c^2 + c * a^2 :=
by
  intros h1 h2
  sorry

end inequality_abc_lt_l562_562046


namespace perpendicular_line_plane_pairs_count_l562_562109

-- Define the concept of a perpendicular line-plane pair.
structure PerpendicularLinePlanePair (V : Type) [EuclideanSpace V] :=
(line : Line V)
(plane : Plane V)
(is_perpendicular : Perpendicular line plane)

-- Define the cube structure and its properties.
structure Cube (V : Type) [EuclideanSpace V] :=
(vertices : Finset V)
(edges : Finset (Line V))
(faces : Finset (Plane V))
(vertices_count : vertices.card = 8)
(edges_count : edges.card = 12)
(faces_count : faces.card = 6)
(vertices_plane_condition : ∀ (plane : Plane V), plane ∈ faces → plane.support.card = 4)
(edges_perpendicular_plane_condition : ∀ (edge : Line V), edge ∈ edges → ∃ (plane1 plane2 : Plane V), plane1 ∈ faces ∧ plane2 ∈ faces ∧ Perpendicular edge plane1 ∧ Perpendicular edge plane2)

-- The theorem to prove the number of perpendicular line-plane pairs in a cube.
theorem perpendicular_line_plane_pairs_count (V : Type) [EuclideanSpace V] (cube : Cube V) : 
  (Finset.filter (λ pair : PerpendicularLinePlanePair V, pair ∈ cube.edges ×ˢ cube.faces) Finset.univ).card = 36 := 
sorry

end perpendicular_line_plane_pairs_count_l562_562109


namespace max_people_transition_l562_562445

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end max_people_transition_l562_562445


namespace solve_inequality_l562_562197

theorem solve_inequality (x : ℝ) (hx : x > 0) : 
  (log x)^2 - 3 * log x + 3) / (log x - 1) < 1 ↔ 0 < x ∧ x < 10 := 
  sorry

end solve_inequality_l562_562197


namespace polynomial_coefficient_values_l562_562504

theorem polynomial_coefficient_values :
  let a : ℕ → ℝ := λ n, (1 - 2 * polynomial.x) ^ 7.coeff n in
  (a 0 = 1) ∧ (∑ i in finset.range 8, a i = -1) := sorry

end polynomial_coefficient_values_l562_562504


namespace tan_ratio_identity_l562_562793

variable {x y : ℝ}

theorem tan_ratio_identity (h1 : (sin x / cos y) - (sin y / cos x) = 2) (h2 : (cos x / sin y) - (cos y / sin x) = 3) :
  (tan x / tan y) - (tan y / tan x) = -1 / 5 :=
by
  sorry

end tan_ratio_identity_l562_562793


namespace length_of_AB_l562_562135

theorem length_of_AB (AC BC : ℝ) (h1 : AC = 6) (h2 : BC = 2) :
  AB = 2 * real.sqrt 10 :=
by sorry

end length_of_AB_l562_562135


namespace area_of_inscribed_regular_octagon_l562_562305

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562305


namespace regular_octagon_area_l562_562318

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562318


namespace determine_discrete_random_variable_l562_562417

def is_discrete_random_variable (X : Type) [fintype X] : Prop := true

def number_of_pages_received_by_pager : Type := ℕ
def random_number_in_interval : Type := Icc (0 : ℝ) 1
def number_of_customers_in_supermarket : Type := ℕ

theorem determine_discrete_random_variable :
  is_discrete_random_variable number_of_pages_received_by_pager ∧
  ¬ is_discrete_random_variable random_number_in_interval ∧
  is_discrete_random_variable number_of_customers_in_supermarket :=
by {
  sorry
}

end determine_discrete_random_variable_l562_562417


namespace saturday_to_wednesday_ratio_is_1_to_2_l562_562895

-- Define the conditions
def num_rabbits : ℕ := 16
def toys_monday : ℕ := 6
def toys_wednesday : ℕ := 2 * toys_monday -- = 12
def toys_friday : ℕ := 4 * toys_monday -- = 24
def total_toys : ℕ := num_rabbits * 3 -- 48

-- Calculate the number of toys bought on Saturday
def toys_bought_from_mon_to_fri : ℕ := toys_monday + toys_wednesday + toys_friday -- = 42
def toys_saturday : ℕ := total_toys - toys_bought_from_mon_to_fri -- = 6

-- Define the ratio of toys bought on Saturday to Wednesday
def ratio_saturday_to_wednesday : ℕ × ℕ := (toys_saturday, toys_wednesday)

-- Theorem to prove
theorem saturday_to_wednesday_ratio_is_1_to_2 : ratio_saturday_to_wednesday = (1, 2) :=
by 
  have h1 : toys_wednesday = 12 := by sorry
  have h2 : toys_saturday = 6 := by sorry
  rw [h1, h2]
  exact rfl

end saturday_to_wednesday_ratio_is_1_to_2_l562_562895


namespace number_of_sets_M_l562_562669

def a1 := "a1"
def a2 := "a2"
def a3 := "a3"
def a4 := "a4"
def U := {a1, a2, a3, a4}
def cond := {a1, a2, a3}
def M_set := {M : set String | M ⊆ U ∧ M ∩ cond = {a1, a2}}

theorem number_of_sets_M : 
    (M_set.card = 4) := 
by 
  sorry

end number_of_sets_M_l562_562669


namespace V3_at_neg4_eq_neg57_l562_562401

-- Polynomial definition and value at x = -4
def poly (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Define the value of V3 using Horner's Rule process
def V3 (x : ℤ) : ℤ :=
  let v0 := 3 in
  let v1 := v0 * x + 5 in
  let v2 := v1 * x + 6 in
  let v3 := v2 * x + 79 in
  let v4 := v3 * x - 8 in
  let v5 := v4 * x + 35 in
  let v6 := v5 * x + 12 in
  v6

-- Prove that V3(-4) = -57
theorem V3_at_neg4_eq_neg57 : V3 (-4) = -57 := by
  sorry

end V3_at_neg4_eq_neg57_l562_562401


namespace range_of_a_l562_562081

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (x^2 + a * x + b) * (Real.exp x - Real.exp 1)

theorem range_of_a (a b : ℝ) (h₁ : ∀ x > 0, f x a b ≥ 0) : a ≥ -1 :=
by
  have : b = -(a + 1),
  sorry

end range_of_a_l562_562081


namespace x_value_l562_562001

variables (w v u x : ℤ)

-- Define the conditions
def condition1 := x = 2 * u + 12
def condition2 := u = v - 15
def condition3 := v = 3 * w + 30
def condition4 := w = 50

-- Statement of the problem
theorem x_value (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : x = 342 :=
by sorry

end x_value_l562_562001


namespace bob_total_investment_l562_562392

variable (x : ℝ) -- the amount invested at 14%

noncomputable def total_investment_amount : ℝ :=
  let interest18 := 7000 * 0.18
  let interest14 := x * 0.14
  let total_interest := 3360
  let total_investment := 7000 + x
  total_investment

theorem bob_total_investment (h : 7000 * 0.18 + x * 0.14 = 3360) :
  total_investment_amount x = 22000 := by
  sorry

end bob_total_investment_l562_562392


namespace total_worksheets_l562_562377

/--
A teacher had a certain number of worksheets to grade, each with 3 problems on it.
She had already graded 7 of them and has 24 more problems to grade.
How many worksheets does she have in total?
-/
theorem total_worksheets (graded_worksheets : ℕ) (problems_per_worksheet : ℕ) (remaining_problems : ℕ) :
  graded_worksheets = 7 → problems_per_worksheet = 3 → remaining_problems = 24 →
  (graded_worksheets + remaining_problems / problems_per_worksheet) = 15 :=
begin
  intros h1 h2 h3,
  sorry
end

end total_worksheets_l562_562377


namespace triangle_angle_sum_l562_562543

theorem triangle_angle_sum (x : ℝ) :
  let a := 40
  let b := 60
  let sum_of_angles := 180
  a + b + x = sum_of_angles → x = 80 :=
by
  intros
  sorry

end triangle_angle_sum_l562_562543


namespace find_f_of_negative_four_l562_562165

def f (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 4 else 6 - 3 * x

theorem find_f_of_negative_four : f (-4) = -8 :=
by
  sorry

end find_f_of_negative_four_l562_562165


namespace remaining_card_is_nine_l562_562627

theorem remaining_card_is_nine
  (cards : Finset ℕ)
  (A_cards B_cards C_cards : Finset ℕ)
  (hA_sum : ∑ n in (Permutations 3 A_cards).toFinset, to_three_digit_num n = 1554)
  (hB_sum : ∑ n in (Permutations 3 B_cards).toFinset, to_three_digit_num n = 1688)
  (hC_sum : ∑ n in (Permutations 3 C_cards).toFinset, to_three_digit_num n = 4662)
  (hCard_disjoint : cards = Finset.range 10)
  (hABC_disjoint : A_cards ∪ B_cards ∪ C_cards ⊂ cards)
  : ∃ x, x ∉ A_cards ∧ x ∉ B_cards ∧ x ∉ C_cards ∧ x ∈ cards ∧ x = 9 :=
by
  sorry

noncomputable def to_three_digit_num (cards : Finset ℕ) : ℕ :=
  sorry

noncomputable def Permutations (n : ℕ) (s : Finset ℕ) : List (Finset ℕ) :=
  sorry

end remaining_card_is_nine_l562_562627


namespace simplify_expression_l562_562182

theorem simplify_expression (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a⁻² * b⁻²) / (a⁻⁴ + b⁻⁴) = (a⁴ * b⁴) / (a⁴ + b⁴) :=
by
  -- skipping proof as per instructions
  sorry

end simplify_expression_l562_562182


namespace smallest_positive_period_intervals_of_monotonic_decrease_range_of_g_on_interval_l562_562584

noncomputable def f (x : ℝ) : ℝ := cos (2 * x + (2 * π) / 3) + 2 * (cos x) ^ 2
noncomputable def g (x : ℝ) : ℝ := cos (2 * x - π / 3) + 1

theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ ∀ x ∈ ℝ, f (x + T) = f x := 
sorry

theorem intervals_of_monotonic_decrease (k : ℤ) :
  ∃ (a b : ℝ), a = k * π - π / 6 ∧ b = k * π + π / 3 ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
sorry

theorem range_of_g_on_interval :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), g x ∈ Icc (1 / 2 : ℝ) 2 :=
sorry

end smallest_positive_period_intervals_of_monotonic_decrease_range_of_g_on_interval_l562_562584


namespace problem_solution_l562_562093

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2}

-- Define the complement function specific to our universal set U
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Lean theorem to prove the given problem's correctness
theorem problem_solution : complement U (A ∪ B) = {3, 4} :=
by
  sorry -- Proof is omitted as per the instructions

end problem_solution_l562_562093


namespace max_candies_per_student_l562_562723

theorem max_candies_per_student (n_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) (max_candies : ℕ) :
  n_students = 50 ∧
  mean_candies = 7 ∧
  min_candies = 1 ∧
  max_candies = 20 →
  ∃ m : ℕ, m ≤ max_candies :=
by
  intro h
  use 20
  sorry

end max_candies_per_student_l562_562723


namespace total_visitors_over_the_weekend_l562_562954

theorem total_visitors_over_the_weekend (morning_sat afternoon_sat evening_sat morning_inc afternoon_inc evening_inc : ℕ) :
  morning_sat = 60 →
  afternoon_sat = 70 →
  evening_sat = 70 →
  morning_inc = 20 →
  afternoon_inc = 30 →
  evening_inc = 50 →
  (let morning_sun := morning_sat + (morning_inc * morning_sat / 100),
       afternoon_sun := afternoon_sat + (afternoon_inc * afternoon_sat / 100),
       evening_sun := evening_sat + (evening_inc * evening_sat / 100),
       total_sat := morning_sat + afternoon_sat + evening_sat,
       total_sun := morning_sun + afternoon_sun + evening_sun
   in total_sat + total_sun = 468) :=
by 
  intros,
  let morning_sun := morning_sat + (morning_inc * morning_sat / 100),
  let afternoon_sun := afternoon_sat + (afternoon_inc * afternoon_sat / 100),
  let evening_sun := evening_sat + (evening_inc * evening_sat / 100),
  let total_sat := morning_sat + afternoon_sat + evening_sat,
  let total_sun := morning_sun + afternoon_sun + evening_sun,
  sorry

end total_visitors_over_the_weekend_l562_562954


namespace negation_universal_proposition_l562_562992

theorem negation_universal_proposition :
  ¬ (∀ x : ℝ, |x| + x^4 ≥ 0) ↔ ∃ x₀ : ℝ, |x₀| + x₀^4 < 0 :=
by
  sorry

end negation_universal_proposition_l562_562992


namespace octagon_area_l562_562295

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562295


namespace domain_width_of_k_l562_562160

def j (x : ℝ) : ℝ → Prop := x ∈ Set.Icc (-10) 10

def k (x : ℝ) : Prop := j (x / 3)

theorem domain_width_of_k :
  (∃ a b : ℝ, (∀ x, k x ↔ x ∈ Set.Icc a b) ∧ b - a = 60) :=
sorry

end domain_width_of_k_l562_562160


namespace range_of_a_1_range_of_a_2_l562_562086
noncomputable theory

def g (x a : ℝ) : ℝ := x^2 + (a - 1)*x + a - 2*a^2

def h (x : ℝ) : ℝ := (x - 1)^2

def A (a : ℝ) : set ℝ := { x | g x a > 0 }

def B : set ℝ := { x | 0 < x ∧ x < 2 }

variable (a : ℝ)

theorem range_of_a_1 :
  (∃ x, x ∈ A a ∧ x ∈ B) →
  (1 / 3 < a ∧ a < 2 ∨ -1 / 2 < a ∧ a < 1 / 3) :=
sorry

def f (x a : ℝ) : ℝ := x * g x a

def C (a : ℝ) : set ℝ := { x | f x a > 0 }

theorem range_of_a_2 :
  (∃ x, x ∈ C a ∧ x ∈ B) →
  (1 / 3 < a ∧ a < 2 ∨ -1 / 2 < a ∧ a < 1 / 3) :=
sorry

end range_of_a_1_range_of_a_2_l562_562086


namespace octagon_area_l562_562296

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562296


namespace number_of_students_in_range_l562_562768

-- Define the basic variables and conditions
variable (a b : ℝ) -- Heights of the rectangles in the histogram

-- Define the total number of surveyed students
def total_students : ℝ := 1500

-- Define the width of each histogram group
def group_width : ℝ := 5

-- State the theorem with the conditions and the expected result
theorem number_of_students_in_range (a b : ℝ) :
    5 * (a + b) * total_students = 7500 * (a + b) :=
by
  -- Proof will be added here
  sorry

end number_of_students_in_range_l562_562768


namespace number_of_triangles_l562_562146

theorem number_of_triangles (n : ℕ) (hn : 0 < n) :
  ∃ t, t = (n + 2) ^ 2 - 2 * (⌊ (n : ℝ) / 2 ⌋) / 4 :=
by
  sorry

end number_of_triangles_l562_562146


namespace days_to_complete_work_l562_562356

theorem days_to_complete_work {D : ℝ} (h1 : D > 0)
  (h2 : (1 / D) + (2 / D) = 0.3) :
  D = 10 :=
sorry

end days_to_complete_work_l562_562356


namespace inequality_correctness_l562_562452

variable (a b : ℝ)
variable (h1 : a < b) (h2 : b < 0)

theorem inequality_correctness : a^2 > ab ∧ ab > b^2 := by
  sorry

end inequality_correctness_l562_562452


namespace angle_B_l562_562867

theorem angle_B (A B C a b c : ℝ) (h : 2 * b * (Real.cos A) = 2 * c - Real.sqrt 3 * a) :
  B = Real.pi / 6 :=
sorry

end angle_B_l562_562867


namespace uber_vs_lyft_cost_l562_562274

-- Let x be the original cost of the taxi ride.
variable (x : ℚ)

-- Define the 20% tip condition.
def tip_condition (x : ℚ) : ℚ := 1.20 * x

-- Define the total cost of the taxi ride being $18
def total_taxi_cost := tip_condition x = 18

-- Define the Lyft ride cost in terms of the taxi ride cost.
def lyft_cost := x + 4

-- Define the Uber ride cost.
def uber_cost := 22

-- The goal is to prove that the Uber ride costs $3 more than the Lyft ride.
theorem uber_vs_lyft_cost (h : total_taxi_cost) : (uber_cost - lyft_cost) = 3 :=
by
  -- Introduce the hypothesis
  have hyp := h
  -- Sorry to skip the proof steps
  sorry

end uber_vs_lyft_cost_l562_562274


namespace tangent_line_eq_l562_562244

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562244


namespace count_valid_numbers_l562_562499

noncomputable def satisfies_conditions (N : ℕ) : Prop :=
  (5000 ≤ N) ∧ (N < 8000) ∧ 
  (N % 5 = 0) ∧ 
  (N % 3 = 0) ∧ 
  let digits := Nat.digits 10 N in
  (digits.length = 4) ∧ 
  (3 ≤ digits.nth 1.getD 0) ∧ 
  (digits.nth 1.getD 0 < digits.nth 2.getD 0) ∧ 
  (digits.nth 2.getD 0 ≤ 6)

theorem count_valid_numbers : 
  (#{ N : ℕ | satisfies_conditions N }.toFinset.card = 8) :=
  sorry

end count_valid_numbers_l562_562499


namespace angle_PCA_67_5_l562_562735

/-- Given:
1. AB is the diameter of circle O.
2. PD is tangent to circle O at point C.
3. PD intersects the extension line of AB at D.
4. CO = CD.
Prove:
∠PCA = 67.5°
--/
theorem angle_PCA_67_5 
  (O A B P D C : Point)
  (h1 : diameter O A B)
  (h2 : tangent PD O C)
  (h3 : intersects_extension PD (extension_line A B) D)
  (h4 : distance O C = distance C D) : 
  angle P C A = 67.5 :=
sorry

end angle_PCA_67_5_l562_562735


namespace shirts_and_pants_neither_plaid_nor_purple_l562_562976

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l562_562976


namespace find_k_l562_562017

theorem find_k (θ : ℝ) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k + 2 * (tan θ^2 + (1 / tan θ)^2) →
  k = 5 := by
  sorry

end find_k_l562_562017


namespace tangent_line_equation_l562_562234

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562234


namespace valid_outfit_choices_l562_562848

def shirts := 6
def pants := 6
def hats := 12
def patterned_hats := 6

theorem valid_outfit_choices : 
  (shirts * pants * hats) - shirts - (patterned_hats * shirts * (pants - 1)) = 246 := by
  sorry

end valid_outfit_choices_l562_562848


namespace distinct_labels_count_eq_catalan_distinct_labels_count_eq_cn_l562_562626

-- Define the conditions given in the problem
def regular_2n_plus_1_gon (n : Nat) : Type := Finset (Fin (2 * n + 1))
def total_zeros_ones (n : Nat) (labels : regular_2n_plus_1_gon n) : Prop :=
  labels.card = 2 * n + 1 ∧ (labels.filter (λ x => x = 0)).card = n + 1 ∧ (labels.filter (λ x => x = 1)).card = n

-- Define the function that counts number of distinct sets of labels considering rotations
def distinct_labels_count (n : Nat) : Nat := (Nat.choose (2 * n + 1) n) / (2 * n + 1)

-- Define the Catalan number
def catalan (n : Nat) : Nat := Nat.choose (2 * n) n / (n + 1)

-- Prove that the number of distinct sets of labels is equal to given expressions
theorem distinct_labels_count_eq_catalan (n : Nat) :
  distinct_labels_count n = catalan n := 
  sorry

-- Prove that the number of distinct sets of labels following the conditions is c_n (Catalan number)
theorem distinct_labels_count_eq_cn (n : Nat) :
  ∃ (cn : Nat), distinct_labels_count n = cn := 
  sorry

end distinct_labels_count_eq_catalan_distinct_labels_count_eq_cn_l562_562626


namespace max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l562_562447

theorem max_transition_channel_BC (hB: (1500: ℝ) * 0.042 = 63) (hC1: (4536: ℝ) / 72 = 63):
  max 63 63 = (63: ℝ) :=
by {
  simp [*, max_def];
}

theorem lowest_cost_per_transition (hA: (3417: ℝ) / 51 = 67):
  (67: ℝ) ≤ 78 :=
by {
  linarith,
}

theorem highest_profit_from_channel_C (hC_sales: (63: ℝ) * 0.05 = 3.15) (rounded_sales_C: ⌊3.15⌋ = 3) 
  (sale_revenue_C: 3 * 2500 = 7500) (total_cost_C: 4536):
  7500 - 4536 = (2964: ℝ) :=
by {
  norm_num,
}

end max_transition_channel_BC_lowest_cost_per_transition_highest_profit_from_channel_C_l562_562447


namespace spring_flu_infection_l562_562130

theorem spring_flu_infection (initially_infected : ℕ) (total_after_two_rounds : ℕ) (average_infected_per_round : ℕ) :
  initially_infected = 1 → total_after_two_rounds = 81 → average_infected_per_round = 8 →
  average_infected_per_round * (average_infected_per_round * initially_infected + initially_infected) + 
  initially_infected = 81 ∧ (total_after_two_rounds * average_infected_per_round + total_after_two_rounds = 729) :=
by
  intros h_initial h_total h_average
  split
  { rw [h_initial, h_total, h_average]
    norm_num }
  { rw [h_total, h_average]
    norm_num }
  sorry

end spring_flu_infection_l562_562130


namespace max_marks_l562_562961

theorem max_marks (marks_obtained failed_by : ℝ) (passing_percentage : ℝ) (M : ℝ) : 
  marks_obtained = 180 ∧ failed_by = 40 ∧ passing_percentage = 0.45 ∧ (marks_obtained + failed_by = passing_percentage * M) → M = 489 :=
by 
  sorry

end max_marks_l562_562961


namespace total_gain_l562_562144

theorem total_gain (Investment_N Time_N : ℝ)
  (h1 : ∃ (Investment_K : ℝ), Investment_K = 4 * Investment_N)
  (h2 : ∃ (Time_K : ℝ), Time_K = 3 * Time_N)
  (h3 : Investment_N * Time_N = 2000) :
  let Gain_N := Investment_N * Time_N
  let Gain_K := Investment_N * 4 * (Time_N * 3)
  (Gain_N + Gain_K) = 26000 :=
by
  let Gain_N := Investment_N * Time_N
  let Gain_K := Investment_N * 4 * (Time_N * 3)
  calc
    Gain_N + Gain_K
      = Investment_N * Time_N + Investment_N * 4 * (Time_N * 3) : sorry
      = Investment_N * Time_N + 12 * Investment_N * Time_N : sorry
      = (1 + 12) * (Investment_N * Time_N) : sorry
      = 13 * 2000 : by rw [h3]
      = 26000 : by norm_num

end total_gain_l562_562144


namespace intersection_on_AB_l562_562128

-- Define a right triangle ABC with a right angle at C
variable (A B C H O₁ O₂ P₁ P₂: Type)
variable [real.triangle ABC]
variable (is_right_angle : ∀ (C: point), right_angle (triangle_angle ABC C) = true)
variable (height_CH : ∀ (H: point), height (triangle_height ABC C H) = true)

-- Define inscribed circles in triangles ACH and BCH
variable (inscribed_circle_AH : ∀ (O₁: point), center (circle_inscribed ACH O₁) = O₁ ∧ point_of_tangency (circle_inscribed ACH O₁ AC) = P₁)
variable (inscribed_circle_BH : ∀ (O₂: point), center (circle_inscribed BCH O₂) = O₂ ∧ point_of_tangency (circle_inscribed BCH O₂ BC) = P₂)

-- Prove that lines O₁P₁ and O₂P₂ intersect on AB
theorem intersection_on_AB :
  ∃ K: point, lies_on (line_intersection O₁P₁ O₂P₂ K) ∧ lies_on (line_intersection AB K) :=
sorry

end intersection_on_AB_l562_562128


namespace circle_equation_circle_center_trajectory_point_N_and_ratio_l562_562828

-- Define the curve G and the given conditions
def G (x a : ℝ) : ℝ := (x^2) / 2 + (a / 2) * x - a^2

-- 1. Prove the circle equation
theorem circle_equation (a : ℝ) (h : a ≠ 0) :
  ∃ D E F, 
    (∀ x y, y = G x a → x^2 + y^2 + D * x + E * y + F = 0) ∧
    D = a ∧ E = a^2 - 2 ∧ F = -2 * a^2 :=
sorry

-- 2. Prove the trajectory equation of the circle center
theorem circle_center_trajectory (a : ℝ) (h : a ≠ 0) :
  let C := (-a / 2, (2 - a^2) / 2) in
  ∀ Cx Cy, (Cx, Cy) = C → Cy = 1 - 2 * Cx^2 ∧ Cx ≠ 0 :=
sorry

-- 3. Prove coordinates of point N and the constant ratio
theorem point_N_and_ratio :
  ∃ N : ℝ × ℝ,
    N = (0, 3 / 2) ∧
    (∀ P : ℝ × ℝ, (P.1^2 + (P.2 - 1)^2 = 1 → 
      let M := (0, 3) in
      ∃ k, k = 1/2 ∧ dist P N / dist P M = k)) :=
sorry

end circle_equation_circle_center_trajectory_point_N_and_ratio_l562_562828


namespace evaluate_expression_l562_562422

theorem evaluate_expression :
  - (18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l562_562422


namespace probability_fewer_heads_than_tails_flipping_8_coins_l562_562648

theorem probability_fewer_heads_than_tails_flipping_8_coins : 
  (let total_outcomes := 256
       heads_eq_tails := 70 / total_outcomes
       fewer_heads := (1 - heads_eq_tails) / 2 
   in fewer_heads = 93 / 256) :=
by
  let total_outcomes := 256
  let heads_eq_tails := 70 / total_outcomes
  let fewer_heads := (1 - heads_eq_tails) / 2
  have : fewer_heads = 93 / 256 := sorry
  exact this

end probability_fewer_heads_than_tails_flipping_8_coins_l562_562648


namespace log_base_4_of_16_l562_562774

theorem log_base_4_of_16 : log 4 16 = 2 :=
by sorry

end log_base_4_of_16_l562_562774


namespace count_distinct_integer_a_l562_562036

-- Define the quadratic equation property
def has_integer_solutions (a : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -a ∧ x1 * x2 = 9 * a

-- State the problem to prove the number of distinct integer values of a
theorem count_distinct_integer_a :
  {a : ℤ | has_integer_solutions a}.to_finset.card = 5 :=
by sorry

end count_distinct_integer_a_l562_562036


namespace value_of_star_l562_562796

theorem value_of_star (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : (a + b) % 4 = 0) : a^2 + 2*a*b + b^2 = 64 :=
by
  sorry

end value_of_star_l562_562796


namespace angle_between_vectors_l562_562521

-- Definitions of vectors and conditions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)

-- Conditions from the problem
def condition1 : Prop := (a - b) ⬝ (a + b) = 0
def condition2 : Prop := ∥a + b∥ = real.sqrt 3 * ∥a∥

-- Statement to prove (angle between a and b is π / 3)
theorem angle_between_vectors (h1 : condition1 a b) (h2 : condition2 a b) : 
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_l562_562521


namespace max_three_digit_multiples_of_4_l562_562458

theorem max_three_digit_multiples_of_4 
  (n : ℕ) (hn : n ≥ 3)
  (a : fin (n+1) → ℕ)
  (h_increasing : ∀ k : fin n, a k < a (k + 1))
  (h_relation : ∀ k : fin (n - 1), a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
  (h_contains_2022 : ∃ k : fin (n + 1), a k = 2022) : 
  3 ≤ n ∧ (Σ x, x < n ∧ a x ≠ 0 ∧
    100 ≤ a x ∧ a x ≤ 999 ∧ a x % 4 = 0) = 225 :=
  sorry

end max_three_digit_multiples_of_4_l562_562458


namespace hyperbola_parabola_focus_l562_562087

theorem hyperbola_parabola_focus (a : ℝ) (h₁ : a > 0)
  (h₂ : let focus := -real.sqrt(a^2 + 1) in focus = -4) : a = real.sqrt(15) :=
sorry

end hyperbola_parabola_focus_l562_562087


namespace sum_of_digits_l562_562649

theorem sum_of_digits (n : ℕ) : 
  n = 2 ^ 2007 * 5 ^ 2005 * 7 → 
  digits_sum n = 10 := by
  intro h
  sorry

end sum_of_digits_l562_562649


namespace find_k_l562_562018

theorem find_k (θ : ℝ) :
  (sin θ + 1 / sin θ)^2 + (cos θ + 1 / cos θ)^2 = k + 2 * (tan θ^2 + (1 / tan θ)^2) →
  k = 5 := by
  sorry

end find_k_l562_562018


namespace total_price_of_mountain_bike_l562_562588

theorem total_price_of_mountain_bike (upfront_percent : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percent = 0.20 → upfront_payment = 240 → total_price = 1200 :=
begin
  intros h1 h2,
  have h3 : 1% = 12, from calc
    (1 / 20) * upfront_payment : sorry,
  have h4 : total_price = 12 * 100, from calc
    12 * 100 : sorry,
  rw [h3, h4],
end

end total_price_of_mountain_bike_l562_562588


namespace problem_solution_l562_562077

noncomputable def f (x a : ℝ) : ℝ := ln x - 2 * a / (x - 1) - a

theorem problem_solution (a x₁ x₂ : ℝ) (hx₁ : f x₁ a = 0) (hx₂ : f x₂ a = 0) (h_distinct: x₁ ≠ x₂) (ha : 0 < a) 
  (h_domain_x₁: x₁ ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_domain_x₂: x₂ ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  (1 / (ln x₁ + a) + 1 / (ln x₂ + a)) < 0 :=
sorry

end problem_solution_l562_562077


namespace coefficient_x3_in_expansion_l562_562763

theorem coefficient_x3_in_expansion :
  ∃ c : ℤ, c = -2 ∧ (x : ℤ) → 
  (coeff (x^3) ((x + 1) * (x - 1)^3) = c) :=
by {
  sorry
}

end coefficient_x3_in_expansion_l562_562763


namespace point_intersection_constant_l562_562120

-- Definitions and conditions based on the problem
def circleM (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36
def curveC (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1
def pointF : ℝ × ℝ := (1, 0)

-- Main theorem statement
theorem point_intersection_constant
  (A P : ℝ × ℝ)
  (hA_curveC : curveC A.1 A.2)
  (hP_curveC : curveC P.1 P.2)
  (B : ℝ × ℝ := (A.1, -A.2))
  (hA_ne_P : A ≠ P) :
  let S := (A.1 * P.2 - P.1 * A.2) / (A.2 - P.2), 0
  let T := (A.1 * P.2 + P.1 * A.2) / (A.2 + P.2), 0
  in |S| * |T| = 9 := sorry

end point_intersection_constant_l562_562120


namespace solution_set_inequality_l562_562209

theorem solution_set_inequality (f : ℝ → ℝ) (hf : ∀ x1 x2, x1 < x2 → (f x1 - f x2) / (x1 - x2) > -1) (h1 : f 1 = 1) :
  {x : ℝ | f (|x - 1|) < 2 - |x - 1|} = set.Ioo 0 2 :=
sorry

end solution_set_inequality_l562_562209


namespace min_framing_required_l562_562687

-- Define the original dimensions
def original_length := 4 -- inches
def original_width := 6 -- inches

-- Define the scale factor
def scale_factor := 4

-- Define the border width
def border_width := 3 -- inches

-- Define the function to compute the new dimensions after enlarging
def enlarged_dimensions (length width : ℕ) : ℕ × ℕ :=
  (length * scale_factor, width * scale_factor)

-- Define the function to compute the dimensions after adding the border
def dimensions_with_border (length width : ℕ) : ℕ × ℕ :=
  (length + 2 * border_width, width + 2 * border_width)

-- Prove that the minimum number of linear feet of framing required is 9
theorem min_framing_required :
  let (enlarged_length, enlarged_width) := enlarged_dimensions original_length original_width
      (final_length, final_width) := dimensions_with_border enlarged_length enlarged_width
      perimeter := 2 * (final_length + final_width)
      perimeter_in_feet := (perimeter + 11) / 12 -- add 11 to effectively round up to the next foot
  in perimeter_in_feet = 9 :=
by
  -- Skipping proof, focusing only on statement structure.
  sorry

end min_framing_required_l562_562687


namespace polynomial_irreducible_l562_562562

def is_irreducible_in_Q (p : Polynomial ℚ) : Prop :=
  irreducible p

theorem polynomial_irreducible {a : ℚ} {n : ℕ} (h_pos : n > 0) :
  is_irreducible_in_Q (X ^ (2 ^ n) * (X + Polynomial.C a) ^ (2 ^ n) + 1) :=
by
  sorry

end polynomial_irreducible_l562_562562


namespace tangent_line_to_curve_l562_562228

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562228


namespace exist_right_triangle_with_trisectable_angles_l562_562781

theorem exist_right_triangle_with_trisectable_angles :
  ∃ (α : ℚ) (A B C : ℤ), 
    6 * α < 90 ∧ 
    tan α ∈ ℚ ∧ 
    ∃ (A B C : ℕ), -- Triangle side lengths
      A * A + B * B = C * C ∧ -- Pythagorean theorem
      (let θ := 6 * α in 
       ∃ (α : ℚ),
         ∃ (tan α = 1/4) /\
         rational (cos θ) ∧ rational (sin θ) ∧
         rational (cos (2*α)) ∧ rational (sin (2*α))) ∧
         ∃ (x : ℚ ∈ [0,90], tan(3x))) ∧
              cos(6α)/cos(2α)=1 ∧
              sin(6α)=1 ∧
              ∙ α + 1 =0.
  sorry

end exist_right_triangle_with_trisectable_angles_l562_562781


namespace health_scores_proof_l562_562204

theorem health_scores_proof 
  (scores : List ℝ)
  (num_students : ℝ)
  (sample_size : ℝ)
  (excellent_threshold : ℝ)
  (excellent_count : ℕ)
  (median_score : ℝ)
  (mode_score : ℝ)
  (xiao_huang_score : ℝ) :
  scores = [55, 65, 71, 73, 78, 82, 85, 85, 86, 86, 86, 88, 92, 92, 93, 94, 96, 97, 99, 100] →
  num_students = 600 →
  sample_size = 20 →
  excellent_threshold = 90 →
  excellent_count = 8 →
  median_score = 86 →
  mode_score = 86 →
  xiao_huang_score = 89 →
  (median scores) = 86 ∧
  (mode scores) = 86 ∧
  (length (filter (λ x, x ≥ excellent_threshold) scores)) = excellent_count ∧
  ((num_students * (excellent_count / sample_size)) = 240) ∧
  (xiao_huang_score > median_score) :=
by 
  sorry

end health_scores_proof_l562_562204


namespace number_of_gigs_played_l562_562694

-- Definitions based on given conditions
def earnings_per_member : ℕ := 20
def number_of_members : ℕ := 4
def total_earnings : ℕ := 400

-- Proof statement in Lean 4
theorem number_of_gigs_played : (total_earnings / (earnings_per_member * number_of_members)) = 5 :=
by
  sorry

end number_of_gigs_played_l562_562694


namespace sequence_sum_formula_l562_562623

noncomputable def sequence_term (k : ℕ) : ℝ :=
  (k + 1) / 2^k

noncomputable def sequence_sum (n : ℕ) : ℝ := 
  (Finset.range n).sum (λ k, sequence_term (k+1))

theorem sequence_sum_formula (n : ℕ) : sequence_sum n = 3 - (n+3) / 2^n :=
by
  sorry

end sequence_sum_formula_l562_562623


namespace frank_final_payment_l562_562039

noncomputable def cost_of_buns (buns_count : ℕ) (bun_price : ℝ) : ℝ :=
  buns_count * bun_price

noncomputable def cost_of_milk (milk_count : ℕ) (milk_price : ℝ) : ℝ :=
  milk_count * milk_price

noncomputable def cost_of_eggs (egg_count : ℕ) (milk_price : ℝ) : ℝ :=
  egg_count * (4 * milk_price)

noncomputable def total_cost_before_discount (buns_cost milk_cost eggs_cost jam_cost bacon_cost : ℝ) : ℝ :=
  buns_cost + milk_cost + eggs_cost + jam_cost + bacon_cost

noncomputable def discount (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

noncomputable def apply_discount (amount discount : ℝ) : ℝ :=
  amount - discount

noncomputable def apply_tax (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

noncomputable def final_amount (total_amount tax : ℝ) : ℝ :=
  total_amount + tax

theorem frank_final_payment :
  let buns_count := 12
  let bun_price := 0.15
  let milk_count := 3
  let milk_price := 2.25
  let egg_count := 1
  let jam_cost := 3.50
  let bacon_cost := 4.75
  let discount_percentage := 0.10
  let tax_percentage := 0.05
  
  let buns_cost := cost_of_buns buns_count bun_price
  let milk_cost := cost_of_milk milk_count milk_price
  let eggs_cost := cost_of_eggs egg_count milk_price
  let total_before_discount := total_cost_before_discount buns_cost milk_cost eggs_cost jam_cost bacon_cost
  
  let discount_amount := discount discount_percentage buns_cost
  let discounted_buns_cost := apply_discount buns_cost discount_amount
  
  let total_after_discount := discounted_buns_cost + milk_cost + eggs_cost + jam_cost + bacon_cost
  
  let tax_amount := apply_tax tax_percentage total_after_discount
  let final_payment := final_amount total_after_discount tax_amount
  
  final_payment.round = 26.90
  := by
  sorry

end frank_final_payment_l562_562039


namespace stone_hits_ground_l562_562985

noncomputable def time_to_hit_ground : ℝ :=
-1 + Real.sqrt 21

theorem stone_hits_ground :
  let t := time_to_hit_ground
  (y : ℝ) = -4*t^2 - 8*t + 80,
  y = 0 → t ≈ 3.58 :=
by
  intro t y ht
  have := calc
    y = -4*t^2 - 8*t + 80 : by sorry
    0 = y : ht
    0 = -4*t^2 - 8*t + 80 : by sorry
    t^2 + 2*t - 20 = 0 : by sorry
    t = -1 + Real.sqrt 21 : by sorry
    t ≈ 3.58 : by sorry
  sorry

end stone_hits_ground_l562_562985


namespace max_f_exists_max_f_l562_562989

open Set

variable {α : Type*} [LinearOrderedField α] 

noncomputable def f (x : α) : α := 5 * real.sqrt (x - 1) + real.sqrt (10 - 2 * x)

theorem max_f : ∀ x ∈ Icc (1 : α) 5, f x ≤ 6 * real.sqrt 3 :=
begin
  sorry
end

theorem exists_max_f : ∃ x ∈ Icc (1 : α) 5, f x = 6 * real.sqrt 3 :=
begin
  use (127 / 27 : α),
  split,
  { split; linarith },
  { sorry }
end

end max_f_exists_max_f_l562_562989


namespace larry_win_probability_correct_l562_562896

/-- Define the probabilities of knocking off the bottle for both players in the first four turns. -/
structure GameProb (turns : ℕ) :=
  (larry_prob : ℚ)
  (julius_prob : ℚ)

/-- Define the probabilities of knocking off the bottle for both players from the fifth turn onwards. -/
def subsequent_turns_prob : ℚ := 1 / 2
/-- Initial probabilities for the first four turns -/
def initial_prob : GameProb 4 := { larry_prob := 2 / 3, julius_prob := 1 / 3 }
/-- The probability that Larry wins the game -/
def larry_wins (prob : GameProb 4) (subsequent_prob : ℚ) : ℚ :=
  -- Calculation logic goes here resulting in the final probability
  379 / 648

theorem larry_win_probability_correct :
  larry_wins initial_prob subsequent_turns_prob = 379 / 648 :=
sorry

end larry_win_probability_correct_l562_562896


namespace fraction_simplification_l562_562015

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l562_562015


namespace remaining_distance_l562_562605

theorem remaining_distance (total_depth distance_traveled remaining_distance : ℕ) (h_total_depth : total_depth = 1218) 
  (h_distance_traveled : distance_traveled = 849) : remaining_distance = total_depth - distance_traveled := 
by
  sorry

end remaining_distance_l562_562605


namespace magnitude_vector_sum_l562_562095

open Real EuclideanGeometry 

variable (a b : EuclideanSpace ℝ (Fin 2))

def a_def : a = ![2, 0] := rfl
def b_norm : ‖b‖ = 1 := sorry
def a_perp_b : ∀ x y : EuclideanSpace ℝ (Fin 2), orthogonal x y := sorry

theorem magnitude_vector_sum : 
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := 
by
  have a : a = ![2, 0] := a_def
  have ha := a
  have hb : ‖b‖ = 1 := b_norm
  have hab : orthogonal a b := a_perp_b a b
  sorry

end magnitude_vector_sum_l562_562095


namespace constantin_mother_deposit_return_l562_562556

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l562_562556


namespace dice_same_color_probability_l562_562849

def probability_same_color := 
  (∑ (color: Fin 5), match color with 
    | 0 => (5 / 24) * (5 / 24)
    | 1 => (6 / 24) * (6 / 24)
    | 2 => (8 / 24) * (8 / 24)
    | 3 => (4 / 24) * (4 / 24)
    | 4 => (1 / 24) * (1 / 24)
  ) = 71 / 288

theorem dice_same_color_probability :
  probability_same_color := by
  sorry

end dice_same_color_probability_l562_562849


namespace inequality_solution_l562_562042

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem inequality_solution (k : ℝ) (h_pos : 0 < k) :
  (0 < k ∧ k < 1 ∧ (1 : ℝ) < x ∧ x < (1 / k)) ∨
  (k = 1 ∧ False) ∨
  (1 < k ∧ (1 / k) < x ∧ x < 1)
  ∨ False :=
sorry

end inequality_solution_l562_562042


namespace january_revenue_fraction_l562_562361

theorem january_revenue_fraction (N D J : ℚ) 
  (h1 : N = (3 / 5) * D)
  (h2 : D = (20 / 7) * (N + J) / 2) :
  J / N = 1 / 6 :=
sorry

end january_revenue_fraction_l562_562361


namespace evaluate_expression_l562_562008

theorem evaluate_expression (c : ℕ) (hc : c = 4) : 
  ((c^c - 2 * c * (c-2)^c + c^2)^c) = 431441456 :=
by
  rw [hc]
  sorry

end evaluate_expression_l562_562008


namespace part_one_part_two_a_part_two_b_l562_562911

variable (a : ℝ) (x : ℝ)

def y1 := a^(3 * x + 5)
def y2 := a^(-2 * x)

-- Condition: 0 < a and a ≠ 1
axiom ha_pos : 0 < a
axiom ha_ne_one : a ≠ 1

-- Proof statements
theorem part_one (hy1_eq_y2 : y1 = y2) : x = -1 := by 
  -- Placeholder for the proof
  sorry

theorem part_two_a (hy1_gt_y2 : y1 > y2) (ha_gt_one : a > 1) : x > -1 := by
  -- Placeholder for the proof
  sorry

theorem part_two_b (hy1_gt_y2 : y1 > y2) (ha_lt_one : 0 < a ∧ a < 1) : x < -1 := by
  -- Placeholder for the proof
  sorry

end part_one_part_two_a_part_two_b_l562_562911


namespace locust_of_midpoints_l562_562564

-- Define Circle K and point P on the circle (not the center)
variables {K : Circle} (r : ℝ) (O : Point) (P : Point)
hypothesis (P_on_K : OnCircle P K)
hypothesis (P_not_center : P ≠ O)

-- Define Q as an external point to the circle
variable (Q : Point)
hypothesis (Q_outside_K : ¬OnCircle Q K)

-- Define secant passing through points Q and P
def secant (Q P : Point) : Line := Line.through Q P

-- Define the midpoint of the secant PQ
noncomputable def midpoint (Q P : Point) : Point :=
  Point.midpoint Q P

-- Statement of the theorem
theorem locust_of_midpoints :
  ∀ Q, ¬OnCircle Q K → OnLine (midpoint Q P) (Line.extends_diameter K P) :=
sorry

end locust_of_midpoints_l562_562564


namespace tangent_line_at_point_l562_562216

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562216


namespace regular_octagon_area_l562_562322

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562322


namespace area_of_inscribed_regular_octagon_l562_562303

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562303


namespace not_all_positive_l562_562163

theorem not_all_positive (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a^2 + b^2 + c^2 = 12) (h3 : a * b * c = 1) : a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0 :=
sorry

end not_all_positive_l562_562163


namespace number_of_arrangements_l562_562969

theorem number_of_arrangements (n : ℕ) (h : n = 7) :
  ∃ (arrangements : ℕ), arrangements = 20 :=
by
  sorry

end number_of_arrangements_l562_562969


namespace total_number_of_animals_l562_562277

-- Prove that the total number of animals is 300 given the conditions described.
theorem total_number_of_animals (A : ℕ) (H₁ : 4 * (A / 3) = 400) : A = 300 :=
sorry

end total_number_of_animals_l562_562277


namespace correct_conclusions_l562_562248

def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)

theorem correct_conclusions :
  (∀ x, f (2 * (Real.pi / 6) - x) = f x) ∧
  (∀ x, f (2 * (2 * (Real.pi / 3)) - x) = f x) ∧
  (∀ x, x ∈ set.Icc (Real.pi / 3) (5 * (Real.pi / 6)) → 
       deriv (Real.sin (x - Real.pi / 3)) > 0) :=
by sorry

end correct_conclusions_l562_562248


namespace roy_total_pens_l562_562968

def number_of_pens (blue black red green purple : ℕ) : ℕ :=
  blue + black + red + green + purple

theorem roy_total_pens (blue black red green purple : ℕ)
  (h1 : blue = 8)
  (h2 : black = 4 * blue)
  (h3 : red = blue + black - 5)
  (h4 : green = red / 2)
  (h5 : purple = blue + green - 3) :
  number_of_pens blue black red green purple = 114 := by
  sorry

end roy_total_pens_l562_562968


namespace imaginary_part_of_z_l562_562076

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : z = 2 / (-1 + I)) : z.im = -1 := 
by
  sorry

end imaginary_part_of_z_l562_562076


namespace marie_profit_l562_562934

-- Define constants and conditions
def loaves_baked : ℕ := 60
def morning_price : ℝ := 3.00
def discount : ℝ := 0.25
def afternoon_price : ℝ := morning_price * (1 - discount)
def cost_per_loaf : ℝ := 1.00
def donated_loaves : ℕ := 5

-- Define the number of loaves sold and revenue
def morning_loaves : ℕ := loaves_baked / 3
def morning_revenue : ℝ := morning_loaves * morning_price

def remaining_after_morning : ℕ := loaves_baked - morning_loaves
def afternoon_loaves : ℕ := remaining_after_morning / 2
def afternoon_revenue : ℝ := afternoon_loaves * afternoon_price

def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_loaves
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

-- Define the total revenue and cost
def total_revenue : ℝ := morning_revenue + afternoon_revenue
def total_cost : ℝ := loaves_baked * cost_per_loaf

-- Define the profit
def profit : ℝ := total_revenue - total_cost

-- State the proof problem
theorem marie_profit : profit = 45 := by
  sorry

end marie_profit_l562_562934


namespace T_n_eq_max_m_l562_562480

-- Define the sequence a_n satisfying the given condition
def a_n (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n terms of a_n
def S_n (n : ℕ) : ℕ := n * (2 * n - 1) / 2

-- Given condition
axiom SumCondition (n : ℕ) : 4 * S_n n = (a_n n + 1) ^ 2

-- b_n definition
def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

-- T_n definition
def T_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k => b_n (k + 1))

-- Prove that T_n = n / (2n + 1)
theorem T_n_eq (n : ℕ) : T_n n = n / (2 * n + 1) :=
  sorry

-- Prove that if T_n > m / 23, then m <= 7
theorem max_m (n : ℕ) (m : ℕ) (h : T_n n > m / 23) : m ≤ 7 :=
  sorry

end T_n_eq_max_m_l562_562480


namespace first_movie_length_is_90_l562_562142

-- Define the parameters and conditions
def length_first_movie (x : ℕ) : Prop :=
  let length_second_movie := x + 30 in
  let time_popcorn := 10 in
  let time_fries := 2 * time_popcorn in
  let total_cooking_time := time_popcorn + time_fries in
  let total_time := 240 in -- 4 hours in minutes
  x + length_second_movie + total_cooking_time = total_time

-- State the theorem
theorem first_movie_length_is_90 : length_first_movie 90 :=
by
  -- This is where you would provide the formal proof
  sorry

end first_movie_length_is_90_l562_562142


namespace student_marks_l562_562732

variable (x : ℕ)
variable (passing_marks : ℕ)
variable (max_marks : ℕ := 400)
variable (fail_by : ℕ := 14)

theorem student_marks :
  (passing_marks = 36 * max_marks / 100) →
  (x + fail_by = passing_marks) →
  x = 130 :=
by sorry

end student_marks_l562_562732


namespace increasing_exponential_function_l562_562609

theorem increasing_exponential_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → (a ^ x) < (a ^ y)) → (1 < a) :=
by
  sorry

end increasing_exponential_function_l562_562609


namespace Konstantin_mother_returns_amount_l562_562551

theorem Konstantin_mother_returns_amount
  (deposit_usd : ℝ)
  (exchange_rate : ℝ)
  (equivalent_rubles : ℝ)
  (h_deposit_usd : deposit_usd = 10000)
  (h_exchange_rate : exchange_rate = 58.15)
  (h_equivalent_rubles : equivalent_rubles = deposit_usd * exchange_rate) :
  equivalent_rubles = 581500 :=
by {
  rw [h_deposit_usd, h_exchange_rate] at h_equivalent_rubles,
  exact h_equivalent_rubles
}

end Konstantin_mother_returns_amount_l562_562551


namespace largest_natural_number_divisible_by_square_l562_562427

theorem largest_natural_number_divisible_by_square :
  ∃ n : ℕ, 
    (∀ k : ℕ, k ≤ 20 → 
      (∏ i in Finset.range 21, (n + i)) % (n + k)^2 = 0) ∧ 
    n = 2^18 * 3^8 * 5^4 * 7^2 * 11 * 13 * 17 * 19 - 20 :=
sorry

end largest_natural_number_divisible_by_square_l562_562427


namespace problem_tiles_count_l562_562273

theorem problem_tiles_count (T B : ℕ) (h: 2 * T + 3 * B = 301) (hB: B = 3) : T = 146 := 
by
  sorry

end problem_tiles_count_l562_562273


namespace compute_fraction_l562_562410

theorem compute_fraction : (2015^2) / (2014^2 + 2016^2 - 2) = 1 / 2 := by
  let a := 2015
  let b := 2014
  let c := 2016
  have h1 : b = a - 1 := by rfl
  have h2 : c = a + 1 := by rfl
  rw [h1, h2]
  sorry

end compute_fraction_l562_562410


namespace vector_equation_l562_562840

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -1)

-- State the theorem we want to prove
theorem vector_equation : (1/3) • a - (4/3) • b = (-1, 2) :=
by sorry

end vector_equation_l562_562840


namespace carbon_emission_l562_562334

theorem carbon_emission (x y : ℕ) (h1 : x + y = 70) (h2 : x = 5 * y - 8) : y = 13 ∧ x = 57 := by
  sorry

end carbon_emission_l562_562334


namespace orthocenter_invariance_l562_562902

open EuclideanGeometry

/-- Let ω be a circle with center O and A, C be two different points on ω. For any third point P of the circle,
    let X and Y be the midpoints of the segments AP and CP. Finally, let H be the orthocenter of the triangle OXY.
    Prove that the position of the point H does not depend on the choice of P. -/
theorem orthocenter_invariance
  (ω : circle) (O A C : Point) (P : Point) (hA : A ∈ ω) (hC : C ∈ ω) (hP : P ∈ ω)
  (X Y M H : Point) (hX : midpoint X A P) (hY : midpoint Y C P) (hM : midpoint M A C)
  (hH : orthocenter H O X Y) :
  H = M :=
sorry

end orthocenter_invariance_l562_562902


namespace tangent_line_to_curve_l562_562233

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562233


namespace fill_time_correct_l562_562388

-- Define conditions
def tank_volume : ℕ := 8000
def initial_volume : ℕ := tank_volume / 2
def inflow_rate : ℝ := 1 / 2  -- kiloliters per minute
def outflow_rate1 : ℝ := 1 / 4  -- kiloliters per minute
def outflow_rate2 : ℝ := 1 / 6  -- kiloliters per minute

-- Define the net inflow rate
def net_inflow_rate : ℝ := inflow_rate - (outflow_rate1 + outflow_rate2)

-- The remaining volume to be filled
def remaining_volume : ℝ := (tank_volume / 2) / 1000  -- kiloliters

-- Define the time to fill the tank
def time_to_fill : ℝ := remaining_volume / net_inflow_rate

-- The theorem to prove
theorem fill_time_correct : time_to_fill ≈ 48 := by
  sorry

end fill_time_correct_l562_562388


namespace probability_two_acceptable_cans_l562_562357

variable {total_cans : ℕ} {unacceptable : ℕ}

-- Conditions
def total_cans : ℕ := 6
def unacceptable : ℕ := 2
def acceptable : ℕ := total_cans - unacceptable
def total_ways_to_choose_two := Nat.choose total_cans 2
def acceptable_ways_to_choose_two := Nat.choose acceptable 2

-- Statement to prove the probability
theorem probability_two_acceptable_cans : 
  (acceptable_ways_to_choose_two : ℚ) / (total_ways_to_choose_two : ℚ) = 2 / 5 := 
by
  sorry

end probability_two_acceptable_cans_l562_562357


namespace dot_product_a_b_l562_562494

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (4, -3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Statement of the theorem to prove
theorem dot_product_a_b : dot_product vector_a vector_b = -1 := 
by sorry

end dot_product_a_b_l562_562494


namespace largest_n_is_253_l562_562375

-- Define the triangle property for a set
def triangle_property (s : Set ℕ) : Prop :=
∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a < b → b < c → c < a + b

-- Define the problem statement
def largest_possible_n (n : ℕ) : Prop :=
∀ (s : Finset ℕ), (∀ (x : ℕ), x ∈ s → 4 ≤ x ∧ x ≤ n) → (s.card = 10 → triangle_property s)

-- The given proof problem
theorem largest_n_is_253 : largest_possible_n 253 :=
by
  sorry

end largest_n_is_253_l562_562375


namespace find_PB_l562_562904

noncomputable def PA : ℝ := 5
noncomputable def PT (AB : ℝ) : ℝ := 2 * (AB - PA) + 1
noncomputable def PB (AB : ℝ) : ℝ := PA + AB

theorem find_PB (AB : ℝ) (AB_condition : AB = PB AB - PA) :
  PB AB = (81 + Real.sqrt 5117) / 8 :=
by
  sorry

end find_PB_l562_562904


namespace intersection_sets_l562_562092

def set_A : Set ℝ := { x : ℝ | | x - 2 | ≥ 1 }
def set_B : Set ℝ := { x : ℝ | 1 / x < 1 }
def set_C : Set ℝ := { x : ℝ | x < 0 } ∪ { x : ℝ | x ≥ 3 }

theorem intersection_sets (A := set_A) (B := set_B) : A ∩ B = set_C := 
sorry

end intersection_sets_l562_562092


namespace count_odd_subsets_of_set_A_l562_562587

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_odd_subset (S : Finset ℕ) : Prop := is_odd (S.sum id)

def set_A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def odd_subsets_count (A : Finset ℕ) : ℕ :=
  (A.powerset.filter is_odd_subset).card

theorem count_odd_subsets_of_set_A : odd_subsets_count set_A = 256 := by
  sorry

end count_odd_subsets_of_set_A_l562_562587


namespace number_of_valid_arrangements_l562_562632

-- Define the problem conditions
def attendees := ("Martians", "Venusians", "Earthlings")
def num_martians := 4
def num_venusians := 6
def num_earthlings := 5
def total_chairs := 15
def fixed_martian_position := 1

-- Define constraints
def constraint1 := ∀ (n ≥ 2), chair_occupied_by n ≠ "Martian" ∨ chair_occupied_by (n-1) ≠ "Martian"
def constraint2 := ∀ (n ≥ 2), chair_occupied_by n ≠ "Earthling" ∨ chair_occupied_by (n-1) ≠ "Venusian"
def constraint3 := chair_occupied_by 1 = "Martian"
def constraint4 := ∀ (n ≥ 2), if chair_occupied_by n = "Martian" then chair_occupied_by (n-1) = "Venusian" ∨ chair_occupied_by (n-1) = "Earthling"

-- Define the statement to find the total number of valid seating arrangements
theorem number_of_valid_arrangements :
  ∃ N : ℕ, satisfies_constraints(N, num_martians, num_venusians, num_earthlings, total_chairs, fixed_martian_position, constraint1, constraint2, constraint3, constraint4) := sorry

end number_of_valid_arrangements_l562_562632


namespace vectors_parallel_x_value_l562_562097

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), let a := (2, 5) in
             let b := (x, -2) in
             (∃ k : ℝ, b = (k * 2, k * 5)) → x = -4/5 :=
by
  intro x a b h
  sorry

end vectors_parallel_x_value_l562_562097


namespace area_of_field_l562_562373

-- Given data
def L : ℕ := 30
def total_fencing : ℕ := 78

-- Condition to be proved
theorem area_of_field : ∃ W : ℕ, (2 * W + L = total_fencing) ∧ (L * W = 720) := by
  use 24
  split
  · -- Prove that 2 * W + L = total_fencing
    sorry
  · -- Prove that L * W = 720
    sorry

end area_of_field_l562_562373


namespace find_f2014_l562_562485

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ x : ℝ, f (x + 6) + f x = 0
axiom f_property2 : symmetric_about f 1 0 -- This needs a proper definition for "symmetric about (1,0)"
axiom f_property3 : f 2 = 4

theorem find_f2014 : f 2014 = -4 :=
by
  sorry

end find_f2014_l562_562485


namespace trapezoid_AD_length_l562_562885

/-
Given:
1. Trapezoid ABCD with AB ∥ CD
2. AD ⊥ BC
3. BC = CD = 39
4. O is the intersection of AC and BD
5. P is the midpoint of BC
6. OP = 10

Prove that length of AD is 5 * sqrt 76
-/

theorem trapezoid_AD_length 
  (A B C D O P : Point)
  (h_trapezoid : Trapezoid ABCD)
  (h_parallel : Parallel AB CD)
  (h_perpendicular : Perpendicular AD BC)
  (h_length_BC : length BC = 39)
  (h_length_CD : length CD = 39)
  (h_inter_diag : Intersection AC BD O)
  (h_midpoint : Midpoint P BC)
  (h_OP : distance O P = 10) : 
  ∃ m n, length AD = m * Real.sqrt n ∧ m = 5 ∧ n = 76 :=
by
  sorry

end trapezoid_AD_length_l562_562885


namespace find_m_n_sum_l562_562438

theorem find_m_n_sum :
  ∃ (m n : ℕ), (∀ p : ℕ, prime p → ¬(p^2 ∣ m)) ∧
    (m + n = 19) ∧
    (∃ (x1 x2 : ℝ), (4 * x1^2 - 12 * x1 - 9 = 0) ∧
                    (4 * x2^2 - 12 * x2 - 9 = 0) ∧
                    (abs (x1 - x2) = real.sqrt m / n)) :=
sorry

end find_m_n_sum_l562_562438


namespace lowest_students_l562_562657

theorem lowest_students (n : ℕ) (h₁ : ∃ k₁ : ℕ, n = 15 * k₁) (h₂ : ∃ k₂ : ℕ, n = 24 * k₂) : n = Nat.lcm 15 24 := by sorry

end lowest_students_l562_562657


namespace largest_C_n_exists_l562_562034

theorem largest_C_n_exists (n : ℕ) (h : 0 < n) :
  ∃ C_n : ℝ,
  (∀ (f : ℕ → (ℝ → ℝ)),
    (∀ i, ∀ x ∈ Icc (0 : ℝ) 1, f i x ∈ Icc (0 : ℝ) 1) →
    ∃ (x : ℕ → ℝ), 
      (∀ i, x i ∈ Icc (0 : ℝ) 1) ∧ 
      abs ((∑ i in Finset.range n, f i (x i)) - (∏ i in Finset.range n, x i)) ≥ C_n) ∧
    C_n = (n - 1) / (2 * n) := by
  sorry

end largest_C_n_exists_l562_562034


namespace extreme_points_sum_l562_562068

theorem extreme_points_sum (p q : ℝ) :
  (∀ x, fderiv ℝ (λ x : ℝ, x^3 + p * x^2 + q * x) x = 3 * x^2 + 2 * p * x + q) ∧
  (3 * 2^2 + 2 * p * 2 + q = 0) ∧
  (3 * (-4)^2 + 2 * p * (-4) + q = 0) →
  p + q = -21 := 
sorry

end extreme_points_sum_l562_562068


namespace solve_system_of_equations_l562_562971

theorem solve_system_of_equations :
  {n : ℕ × ℕ × ℕ × ℕ // (n.1 + n.2.1 = n.2.2 * n.2.2.1) ∧ (n.2.2 + n.2.2.1 = n.1 * n.2.1)} =
  { (1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2), (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1), (2, 2, 2, 2) } := 
by
  sorry

end solve_system_of_equations_l562_562971


namespace second_white_probability_l562_562693

-- Define the conditions
structure Bag :=
  (white_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : white_balls + black_balls)

-- Define the problem with the provided conditions and the question to be proven
def probability_second_white (bag : Bag) (first_white : Prop) : Prop :=
  bag.white_balls = 3 ∧
  bag.black_balls = 2 ∧
  first_white →
  ∃ p : ℚ, p = 1 / 2

-- State the theorem to be proven
theorem second_white_probability (bag : Bag) :
  probability_second_white bag (bag.white_balls > 0) :=
sorry

end second_white_probability_l562_562693


namespace find_range_of_a_l562_562486

noncomputable def f (a x : ℝ) : ℝ := (a / x) + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - 5
noncomputable def h (x : ℝ) : ℝ := x - x^2 * Real.log x

theorem find_range_of_a :
  (∀ x₁ x₂ ∈ Set.Icc (1 / 2 : ℝ) 2, f a x₁ - g x₂ ≥ 2) → (a ≥ 1) := sorry

end find_range_of_a_l562_562486


namespace total_handshakes_l562_562106

theorem total_handshakes (n : ℕ) (h : n = 10) : 
  let total = n * (n - 1) / 2 
  in total - 2 = 43 :=
by
  intro h,
  simp [h],
  let total := 10 * (10 - 1) / 2,
  have eq1 : total = 45 := by 
    rw [Nat.mul_sub, Nat.div],
    simp,
  rw [eq1],
  exact rfl

end total_handshakes_l562_562106


namespace fraction_upgraded_sensors_l562_562658

theorem fraction_upgraded_sensors:
  ∃ (N U : ℕ), (N = U / 6) ∧ (24 * N + U > 0) → (U / (24 * N + U) = 1 / 5) :=
begin
  sorry
end

end fraction_upgraded_sensors_l562_562658


namespace mortgage_loan_amount_l562_562644

theorem mortgage_loan_amount (C : ℝ) (hC : C = 8000000) : 0.75 * C = 6000000 :=
by
  sorry

end mortgage_loan_amount_l562_562644


namespace length_of_AD_l562_562545

theorem length_of_AD (A B C D : ℝ) (h h_sq : ℝ) (AB AC BD CD : ℝ) 
  (h1 : AB = 13) (h2 : AC = 17) (h3 : BD/CD = 3/4) (h4 : BD^2 = 169 - h^2) 
  (h5 : CD^2 = 289 - h^2) (h6 : 16 * (169 - h^2) = 9 * (289 - h^2)) : 
  h = Real.sqrt (103 / 7) :=
begin
  sorry
end

end length_of_AD_l562_562545


namespace solve_for_x_l562_562650

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 = 6 - x) : x = 3 :=
by
  sorry

end solve_for_x_l562_562650


namespace kids_stay_home_lawrence_county_l562_562004

def total_kids_lawrence_county : ℕ := 1201565
def kids_camp_lawrence_county : ℕ := 610769

theorem kids_stay_home_lawrence_county : total_kids_lawrence_county - kids_camp_lawrence_county = 590796 := by
  sorry

end kids_stay_home_lawrence_county_l562_562004


namespace min_framing_required_l562_562686

-- Define the original dimensions
def original_length := 4 -- inches
def original_width := 6 -- inches

-- Define the scale factor
def scale_factor := 4

-- Define the border width
def border_width := 3 -- inches

-- Define the function to compute the new dimensions after enlarging
def enlarged_dimensions (length width : ℕ) : ℕ × ℕ :=
  (length * scale_factor, width * scale_factor)

-- Define the function to compute the dimensions after adding the border
def dimensions_with_border (length width : ℕ) : ℕ × ℕ :=
  (length + 2 * border_width, width + 2 * border_width)

-- Prove that the minimum number of linear feet of framing required is 9
theorem min_framing_required :
  let (enlarged_length, enlarged_width) := enlarged_dimensions original_length original_width
      (final_length, final_width) := dimensions_with_border enlarged_length enlarged_width
      perimeter := 2 * (final_length + final_width)
      perimeter_in_feet := (perimeter + 11) / 12 -- add 11 to effectively round up to the next foot
  in perimeter_in_feet = 9 :=
by
  -- Skipping proof, focusing only on statement structure.
  sorry

end min_framing_required_l562_562686


namespace angle_CEF_eq_2_angle_AGF_l562_562736

variables {O : Type*} [metric_space O] [empirical_space O] -- Circle O
variables {A B C D E F G : O} -- Points on the circle and other defined points

def diameter (A B : O) (O : empirical_space O) : Prop :=
  ∃ midpoint O (A ≠ B), O = midpoint A B

def on_circle (X : O) (O : empirical_space O) : Prop :=
  ∃ (r : ℝ), dist X O = r

def tangent_to_circle_at (X E : O) (O : empirical_space O) : Prop :=
  ∀ Y, dist X O = dist Y O → dist X Y ≤ dist E Y

def intersect_lines (X Y Z : O) : Prop :=
  ∃ W, lies_on_line W X Y ∧ lies_on_line W X Z

theorem angle_CEF_eq_2_angle_AGF 
  (O : empirical_space O) (A B C D E F G : O)
  (h_diameter_AB : diameter A B O)
  (h_on_circle_C : on_circle C O)
  (h_on_circle_D : on_circle D O)
  (h_same_side_AB : ∀ P Q ε, lies_on_line P A B ∧ lies_on_line Q A B → dist ε P Q < dist P Q)
  (h_tangents_intersect_E : tangent_to_circle_at C E O ∧ tangent_to_circle_at D E O)
  (h_lines_intersect_F : intersect_lines B C F ∧ intersect_lines A D F)
  (h_EB_intersects_at_G : lies_on_line G E B ∧ on_circle G O) :
  ∠ CEF = 2 * ∠ AGF :=
sorry -- Proof goes here

end angle_CEF_eq_2_angle_AGF_l562_562736


namespace eggs_needed_per_month_l562_562180

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l562_562180


namespace proof_problem_l562_562333

noncomputable def main_theorem : Nat := 
  let product := ∏ k in Finset.range 45, Real.sec ((4 * (k + 1) - 3) * Real.pi / 180) ^ 2
  let m := 2
  let n := 360
  m + n

theorem proof_problem : main_theorem = 364 :=
  sorry

end proof_problem_l562_562333


namespace percent_half_dollars_correct_l562_562655

open Rat

noncomputable def value_nickels (n : ℕ) : ℚ := n * 5
noncomputable def value_half_dollars (h : ℕ) : ℚ := h * 50
noncomputable def total_value (n h : ℕ) : ℚ := value_nickels n + value_half_dollars h
noncomputable def percent_half_dollars (n h : ℕ) : ℚ :=
  value_half_dollars h / total_value n h * 100

theorem percent_half_dollars_correct :
  percent_half_dollars 80 40 = 83.33 := by
  sorry

end percent_half_dollars_correct_l562_562655


namespace sum_reciprocal_sin_l562_562962

theorem sum_reciprocal_sin (n : ℕ) (x : ℝ) (hx : ∀ t k, (t ∈ Finset.range (n + 1)) → x ≠ (k * π) / (2^t)) :
  (∑ i in Finset.range n, (1 / Real.sin (2^(i+1) * x))) = Real.cot x - Real.cot (2^n * x) :=
by
  sorry

end sum_reciprocal_sin_l562_562962


namespace find_DC_l562_562540

open Real

/-- In the diagram below, AB = 36, and ∠ADB = 90°. If sin(A) = 4/5 and sin(C) = 1/5, then DC = 4 * √12441.6 -/
theorem find_DC 
  (A B C D : ℝ)
  (hAB: A = 0 ∧ B = 36)
  (hAngle: ∠ A D B = 90)
  (hSinA : sin A = 4/5)
  (hSinC : sin C = 1/5) :
  ∥D - C∥ = 4 * sqrt 12441.6 :=
sorry

end find_DC_l562_562540


namespace polygon_area_after_rotation_l562_562798

-- Definitions from Conditions
variable {n : ℕ} -- n-gon
variable {A : ℝ} -- Area of the n-gon
variable {x : ℝ} -- Rotation angle
variable {A_i : Fin n → ℝ × ℝ} -- Vertices of the n-gon
variable {P : ℝ × ℝ} -- Point P

-- Statement of the theorem
theorem polygon_area_after_rotation (A : ℝ) (x : ℝ) : ℝ :=
4 * (Real.sin (x / 2))^2 * A

-- Placeholder proof to ensure the statement builds successfully
begin
  sorry
end

end polygon_area_after_rotation_l562_562798


namespace quadratic_roots_and_expression_value_l562_562411

theorem quadratic_roots_and_expression_value :
  let a := 3 + Real.sqrt 21
  let b := 3 - Real.sqrt 21
  (a ≥ b) →
  (∃ x : ℝ, x^2 - 6 * x + 11 = 23) →
  3 * a + 2 * b = 15 + Real.sqrt 21 :=
by
  intros a b h1 h2
  sorry

end quadratic_roots_and_expression_value_l562_562411


namespace smallest_number_condition_l562_562620

theorem smallest_number_condition :
  ∃ n : ℕ, (n + 1) % 12 = 0 ∧
           (n + 1) % 18 = 0 ∧
           (n + 1) % 24 = 0 ∧
           (n + 1) % 32 = 0 ∧
           (n + 1) % 40 = 0 ∧
           n = 2879 :=
sorry

end smallest_number_condition_l562_562620


namespace recurring_decimal_fraction_l562_562012

theorem recurring_decimal_fraction :
  let a := 0.714714714...
  let b := 2.857857857...
  (a / b) = (119 / 476) :=
by
  let a := (714 / (999 : ℝ))
  let b := (2856 / (999 : ℝ))
  sorry

end recurring_decimal_fraction_l562_562012


namespace framing_required_l562_562678

theorem framing_required
  (initial_width : ℕ)
  (initial_height : ℕ)
  (scale_factor : ℕ)
  (border_width : ℕ)
  (increments : ℕ)
  (initial_width_def : initial_width = 4)
  (initial_height_def : initial_height = 6)
  (scale_factor_def : scale_factor = 4)
  (border_width_def : border_width = 3)
  (increments_def : increments = 12) :
  Nat.ceil ((2 * (4 * scale_factor  + 2 * border_width + 6 * scale_factor + 2 * border_width).toReal) / increments) = 9 := by
  sorry

end framing_required_l562_562678


namespace qualified_product_probability_l562_562114

theorem qualified_product_probability 
  (pA : ℝ)
  (pB : ℝ)
  (qA : ℝ)
  (qB : ℝ)
  (h1: pA = 0.60)
  (h2: pB = 0.40)
  (h3: qA = 0.95)
  (h4: qB = 0.90)
  : pA * qA + pB * qB = 0.93 :=
by {
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end qualified_product_probability_l562_562114


namespace tangent_line_at_point_l562_562221

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562221


namespace margie_distance_l562_562590

theorem margie_distance (mpg cost_per_gallon money distance : ℕ) (h1 : mpg = 40) (h2 : cost_per_gallon = 5) (h3 : money = 25) (h4 : distance = 200) : 
distance = (money / cost_per_gallon) * mpg :=
by 
  rw [h1, h2, h3, h4]
  sorry

end margie_distance_l562_562590


namespace hyperbola_proof_l562_562606

noncomputable def hyperbola_equation : Bool :=
  ∃ (a b : ℝ),
  ((let c := (2 * Real.sqrt 3) / 3 in
  let asymptotes := λ x : ℝ, (Real.sqrt 3 * x, -Real.sqrt 3 * x) in
  c^2 = a^2 + b^2 ∧ (b / a = Real.sqrt 3) ∧
   (3*x^2 - y^2 = 1))) ∧
   ∀ x y, ∀ P : ℝ * ℝ, let m := ∣ Real.sqrt 3 * P.1 - P.2 ∣ / 2 in 
   let n := ∣ Real.sqrt 3 * P.1 + P.2 ∣ / 2 in 
   m * n = 1 / 4

-- Placeholder for proof
theorem hyperbola_proof : hyperbola_equation :=
by sorry

end hyperbola_proof_l562_562606


namespace min_toothpicks_to_remove_l562_562037

theorem min_toothpicks_to_remove (n_toothpicks : ℕ) (n_triangles : ℕ) (n_squares : ℕ)
  (h1 : n_toothpicks = 40) (h2 : n_triangles > 20) (h3 : n_squares = 10) :
  ∃ k : ℕ, k = 20 ∧ ∀ (k' : ℕ), (k' < k) → (∃ (fp_toothpicks : ℕ), fp_toothpicks = (n_toothpicks - k') 
  → ¬ (∃ (t : ℕ), t ≤ n_triangles ∧ t > 0) ∨ ¬ (∃ (s : ℕ), s ≤ n_squares ∧ s > 0)) :=
begin
  sorry
end

end min_toothpicks_to_remove_l562_562037


namespace randy_trip_total_distance_l562_562189

-- Definition of the problem condition
def randy_trip_length (x : ℝ) : Prop :=
  x / 3 + 20 + x / 5 = x

-- The total length of Randy's trip
theorem randy_trip_total_distance : ∃ x : ℝ, randy_trip_length x ∧ x = 300 / 7 :=
by
  sorry

end randy_trip_total_distance_l562_562189


namespace original_combined_price_l562_562787

theorem original_combined_price (C S : ℝ)
  (hC_new : (C + 0.25 * C) = 12.5)
  (hS_new : (S + 0.50 * S) = 13.5) :
  (C + S) = 19 := by
  -- sorry makes sure to skip the proof
  sorry

end original_combined_price_l562_562787


namespace find_a_l562_562044

theorem find_a (A B : Real) (b a : Real) (hA : A = 45) (hB : B = 60) (hb : b = Real.sqrt 3) : 
  a = Real.sqrt 2 :=
sorry

end find_a_l562_562044


namespace octagon_area_inscribed_in_circle_l562_562313

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562313


namespace certain_amount_added_l562_562714

theorem certain_amount_added {x y : ℕ} 
    (h₁ : x = 15) 
    (h₂ : 3 * (2 * x + y) = 105) : y = 5 :=
by
  sorry

end certain_amount_added_l562_562714


namespace percentage_loss_l562_562711

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end percentage_loss_l562_562711


namespace math_problem_l562_562403

-- Problem (1)
def problem1 : Prop :=
  sqrt 27 + sqrt (1 / 3) - sqrt 12 = 4 * sqrt 3 / 3

-- Problem (2)
def problem2 : Prop :=
  (sqrt 2 + 1) ^ 2 + 2 * sqrt 2 * (sqrt 2 - 1) = 7

-- Statement that needs to be proven
theorem math_problem : problem1 ∧ problem2 := by
  sorry

end math_problem_l562_562403


namespace necessary_and_sufficient_condition_l562_562914

-- Variables and conditions
variables (a : ℕ) (A B : ℝ)
variable (positive_a : 0 < a)

-- System of equations
def system_has_positive_integer_solutions (x y z : ℕ) : Prop :=
  (x^2 + y^2 + z^2 = (13 * a)^2) ∧ 
  (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
    (1 / 4) * (2 * A + B) * (13 * a)^4)

-- Statement of the theorem
theorem necessary_and_sufficient_condition:
  (∃ (x y z : ℕ), system_has_positive_integer_solutions a A B x y z) ↔ B = 2 * A :=
sorry

end necessary_and_sufficient_condition_l562_562914


namespace grocer_profit_is_six_l562_562708

-- Define the initial conditions
def cost_per_pound : ℝ := 0.5 / 3
def selling_price_per_pound : ℝ := 1 / 4
def total_weight : ℝ := 72

-- Define the cost, revenue, and then the profit
def cost : ℝ := total_weight * cost_per_pound
def revenue : ℝ := total_weight * selling_price_per_pound
def profit : ℝ := revenue - cost

-- Lean statement that asserts the profit is $6
theorem grocer_profit_is_six :
  profit = 6 := by
  sorry

end grocer_profit_is_six_l562_562708


namespace log_10_7_eqn_l562_562508

variables (p q : ℝ)
noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_10_7_eqn (h1 : log_base 4 5 = p) (h2 : log_base 5 7 = q) : 
  log_base 10 7 = (2 * p * q) / (2 * p + 1) :=
by 
  sorry

end log_10_7_eqn_l562_562508


namespace credit_card_balance_l562_562933

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l562_562933


namespace cameron_shots_l562_562404

theorem cameron_shots (initial_shots additional_shots made_init_percent made_final_percent : ℕ) (h_init_shots : initial_shots = 30) (h_additional_shots : additional_shots = 10) (h_made_init_percent : made_init_percent = 60) (h_made_final_percent : made_final_percent = 62) :
  let made_initial := (initial_shots * made_init_percent) / 100,
      made_total := ((initial_shots + additional_shots) * made_final_percent) / 100
  in made_total - made_initial = 7 := 
by
  sorry

end cameron_shots_l562_562404


namespace bill_experience_l562_562748

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l562_562748


namespace circumscribed_sphere_surface_area_of_cube_l562_562513

theorem circumscribed_sphere_surface_area_of_cube (a : ℝ) (h : a = sqrt 3) :
  let d := sqrt (a^2 + a^2 + a^2) in
  let R := d / 2 in
  let S := 4 * Real.pi * R^2 in
  S = 9 * Real.pi :=
by
  have a_def : a = sqrt 3 := h
  let d := sqrt (a^2 + a^2 + a^2)
  have d_def : d = 3 := by sorry
  let R := d / 2
  have R_def : R = 3 / 2 := by sorry
  let S := 4 * Real.pi * R^2
  have S_def : S = 9 * Real.pi := by sorry
  exact S_def

end circumscribed_sphere_surface_area_of_cube_l562_562513


namespace total_rainfall_l562_562393

theorem total_rainfall :
  let monday := 0.12962962962962962
  let tuesday := 0.35185185185185186
  let wednesday := 0.09259259259259259
  let thursday := 0.25925925925925924
  let friday := 0.48148148148148145
  let saturday := 0.2222222222222222
  let sunday := 0.4444444444444444
  (monday + tuesday + wednesday + thursday + friday + saturday + sunday) = 1.9814814814814815 :=
by
  -- proof to be filled here
  sorry

end total_rainfall_l562_562393


namespace cannot_transform_l562_562639

theorem cannot_transform 
    (a b c : ℝ)
    (h1 : a = 1 / Real.sqrt 2)
    (h2 : b = Real.sqrt 2)
    (h3 : c = 2) : 
    ¬∃ x y z : ℝ, 
        (∀ u v : ℝ, u = x ∨ u = y ∨ u = z → v = x ∨ v = y ∨ v = z → 
         ∃ p q : ℝ, p = (u + v) / Real.sqrt 2 ∧ q = (u - v) / Real.sqrt 2) ∧ 
        x = 1 ∧ y = Real.sqrt 2 ∧ z = 1 + Real.sqrt 2 :=
begin
  sorry
end

end cannot_transform_l562_562639


namespace neg_mul_neg_pos_mul_neg_neg_l562_562398

theorem neg_mul_neg_pos (a b : Int) (ha : a < 0) (hb : b < 0) : a * b > 0 :=
sorry

theorem mul_neg_neg : (-1) * (-3) = 3 := 
by
  have h1 : -1 < 0 := by norm_num
  have h2 : -3 < 0 := by norm_num
  have h_pos := neg_mul_neg_pos (-1) (-3) h1 h2
  linarith

end neg_mul_neg_pos_mul_neg_neg_l562_562398


namespace quadratic_inequality_solution_l562_562600

theorem quadratic_inequality_solution :
  ∀ x, 9 * x^2 - 6 * x + 1 > 0 ↔ x ∈ set.Ioo (-∞) (1/3 : ℝ) ∪ set.Ioo (1/3 : ℝ) (∞) := by
  sorry

end quadratic_inequality_solution_l562_562600


namespace difference_of_sorted_numbers_l562_562323

def largest_number (digits : list ℕ) : ℕ :=
  (list.sort (≤) digits).reverse.foldl (λ acc n, acc * 10 + n) 0

def smallest_number (digits : list ℕ) : ℕ :=
  (list.sort (≤) digits).foldl (λ acc n, acc * 10 + n) 0

theorem difference_of_sorted_numbers :
  let digits := [7, 3, 1, 4] in
  (largest_number digits - smallest_number digits) = 6084 :=
by
  sorry

end difference_of_sorted_numbers_l562_562323


namespace find_function_perfect_square_condition_l562_562021

theorem find_function_perfect_square_condition (g : ℕ → ℕ)
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end find_function_perfect_square_condition_l562_562021


namespace perimeter_ratio_l562_562264

theorem perimeter_ratio (a : ℝ) (h : 0 < a) :
  let p1 := (a, 2 * a),
      p2 := (-a, -2 * a),
      L1 := 3 * a,
      L2 := 2 * a * Real.sqrt 5,
      L3 := a
  in (L1 + L2 + L3) / a = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end perimeter_ratio_l562_562264


namespace minimal_triangle_areas_l562_562561

theorem minimal_triangle_areas (A B C D : Point) (r : Line)
 (h_rect : is_rectangle A B C D)
 (h_r_parallel : parallel_to r AB)
 (h_r_intersects : intersects r AC)
 (h_mid_AD : midpoint_of r AD) :
  let triangle_area := ∀ r, ∃ Δ₁ Δ₂ : Triangle, Δ₁ ∧ Δ₂ 
  triangle_area = (1 / 2) * (b * a) :=
by sorry

end minimal_triangle_areas_l562_562561


namespace planes_perpendicular_of_line_conditions_l562_562190

variables (a b l : Line) (M N : Plane)

-- Definitions of lines and planes and their relations
def parallel_to_plane (a : Line) (M : Plane) : Prop := sorry
def perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry
def subset_of_plane (a : Line) (M : Plane) : Prop := sorry

-- Statement of the main theorem to be proved
theorem planes_perpendicular_of_line_conditions (a b l : Line) (M N : Plane) :
  (perpendicular_to_plane a M) → (parallel_to_plane a N) → (perpendicular_to_plane N M) :=
  by
  sorry

end planes_perpendicular_of_line_conditions_l562_562190


namespace octagon_area_inscribed_in_circle_l562_562314

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562314


namespace light_beam_distance_l562_562355

noncomputable theory

def point := ℝ × ℝ × ℝ

def distance (p q : point) : ℝ := 
  real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

theorem light_beam_distance :
  let P : point := (1, 2, 3)
  let Q : point := (4, 4, 4)
  let Q' : point := (4, 4, -4)
  distance P Q' = real.sqrt 62 := 
by 
  intros P Q Q' 
  unfold distance
  -- Simplify step by step
  have h1: (Q'.1 - P.1) ^ 2 = (4 - 1) ^ 2, by norm_num,
  have h2: (Q'.2 - P.2) ^ 2 = (4 - 2) ^ 2, by norm_num,
  have h3: (Q'.3 - P.3) ^ 2 = (-4 - 3) ^ 2, by norm_num,
  rw [h1, h2, h3],
  norm_num,
  rw [real.sqrt_eq_rpow, ← real.rpow_nat_cast (show 3, from ⟨9⟩), ← real.rpow_nat_cast, ← real.rpow_nat_cast],
  refl,
sorry

end light_beam_distance_l562_562355


namespace emily_candy_per_day_l562_562788

theorem emily_candy_per_day :
  let totalCandy := 5 + 13 in
  totalCandy / 2 = 9 :=
by
  have h1 : totalCandy = 18 := by rfl
  have h2 : totalCandy / 2 = 18 / 2 := by rfl
  have h3 : 18 / 2 = 9 := by rfl
  sorry

end emily_candy_per_day_l562_562788


namespace x_minus_y_eq_14_l562_562966

theorem x_minus_y_eq_14 (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 100) : x - y = 14 :=
sorry

end x_minus_y_eq_14_l562_562966


namespace angle_C_not_right_lambda_range_l562_562795

-- Define the basic trigonometric and geometric conditions
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle opposite to angles A, B, and C
variable {λ : ℝ} -- Lambda

-- Condition that the triangle satisfies
axiom triangle_condition : 4 * sin B * sin (C - A) + 1 = 4 * sin A * sin B + cos (2 * C)
-- Acute triangle and sine relation
axiom acute_triangle (A_lt_90 : A < real.pi / 2) (B_lt_90 : B < real.pi / 2) (C_lt_90 : C < real.pi / 2) : 
  0 < sin A ∧ 0 < sin B ∧ 0 < sin C
axiom sine_relation : sin B = λ * sin A

-- Problem 1: Prove that angle C cannot be a right angle
theorem angle_C_not_right : ¬ (C = real.pi / 2) := sorry

-- Problem 2: Prove the range of λ
theorem lambda_range (acute_triangle) (sine_relation) : 1/3 < λ ∧ λ < 5/3 := sorry

end angle_C_not_right_lambda_range_l562_562795


namespace sequence_x2015_l562_562258

theorem sequence_x2015 (x : ℕ → ℝ) (h1 : x 1 = 3) 
  (h2 : ∀ n, n ≥ 2 → (∑ i in Finset.range (n - 1), x (i + 1)) + (4 / 3) * x n = 4) :
  x 2015 = 3 / (4 ^ 2014) :=
sorry

end sequence_x2015_l562_562258


namespace alternate_sum_of_areas_l562_562959

-- Define a point being inside an equilateral triangle
variable (A B C P D E F : Point)
variable (triangle_eq : EquilateralTriangle A B C)
variable (perpendiculars : 
  Perpendicular P D (Line B C) ∧
  Perpendicular P E (Line C A) ∧
  Perpendicular P F (Line A B))

theorem alternate_sum_of_areas :
  Area (Triangle.mk P A D) + Area (Triangle.mk P B E) + Area (Triangle.mk P C F) =
  Area (Triangle.mk P D B) + Area (Triangle.mk P E C) + Area (Triangle.mk P F A) :=
by sorry

end alternate_sum_of_areas_l562_562959


namespace area_ECODF_shaded_l562_562759

noncomputable def circle_centered (center : ℝ × ℝ) (radius : ℝ) := 
  { P : ℝ × ℝ | (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2 }

variables (A B O C D E F : ℝ × ℝ)

def distances := 
  (dist A B = 6 ∧ dist O A = 3 * (real.sqrt 2) ∧ 
   dist A E = 3 ∧ dist B F = 3 ∧ 
   right_triangle A C O ∧ right_triangle B D O ∧ 
   is_tangent OC (circle_centered A 3) ∧ 
   is_tangent OD (circle_centered B 3) ∧ 
   common_tangent_of_circles_higher EF (circle_centered A 3) (circle_centered B 3))

theorem area_ECODF_shaded :
  distances →
  area (region_ECODF) = 18 * real.sqrt 2 - 9 - (9 * real.pi) / 4 :=
by
  sorry

end area_ECODF_shaded_l562_562759


namespace cube_octahedron_l562_562964

theorem cube_octahedron (A B C D A1 B1 C1 D1 P Q R P' Q' R' : Point) 
  (hA : distance A B = distance A D ∧ distance A D = distance A A1 ∧ distance A1 B1 = distance A1 D1 ∧ distance A1 D1 = distance A1 C1 ∧ distance C1 B1 = distance C1 D1 ∧ distance B B1 = distance C C1)
  (hP : distance A P = (3/4) * distance A B)
  (hQ : distance A Q = (3/4) * distance A D)
  (hR : distance A R = (3/4) * distance A A1)
  (hP' : distance C1 P' = (3/4) * distance C1 B1)
  (hQ' : distance C1 Q' = (3/4) * distance C1 D1)
  (hR' : distance C1 R' = (3/4) * distance C1 C) :
  is_octahedron {P, Q, R, P', Q', R'} := sorry

end cube_octahedron_l562_562964


namespace no_such_two_digit_number_exists_l562_562002

theorem no_such_two_digit_number_exists :
  ¬ ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
                 (10 * x + y = 2 * (x^2 + y^2) + 6) ∧
                 (10 * x + y = 4 * (x * y) + 6) := by
  -- We need to prove that no two-digit number satisfies
  -- both conditions.
  sorry

end no_such_two_digit_number_exists_l562_562002


namespace wheel_radius_l562_562615

theorem wheel_radius (distance_covered : ℝ) (revolutions : ℕ) (h : distance_covered = 563.2) (h' : revolutions = 400) :
  let C := distance_covered / revolutions in
  let r := C / (2 * Real.pi) in
  r = 0.224 :=
by
  let C := distance_covered / revolutions
  let r := C / (2 * Real.pi)
  sorry

end wheel_radius_l562_562615


namespace hundred_pow_neg_y_l562_562851

theorem hundred_pow_neg_y (y : ℝ) (h : 100^y = 16) : 100^(-y) = 1 / 6.31 :=
  sorry

end hundred_pow_neg_y_l562_562851


namespace b_negative_l562_562065

variable {R : Type*} [LinearOrderedField R]

theorem b_negative (a b : R) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : R, 0 ≤ x → (x - a) * (x - b) * (x - (2*a + b)) ≥ 0) : b < 0 := 
sorry

end b_negative_l562_562065


namespace minimum_value_inequality_l562_562566

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2) >= 3 := 
begin
  sorry
end

end minimum_value_inequality_l562_562566


namespace remaining_garden_area_not_covered_l562_562363

/-- Given a circular garden with radius 7 feet and a 4-feet wide arc-shaped path passing through the center,
    prove that the remaining area of the garden not covered by the path is 29π square feet -/
theorem remaining_garden_area_not_covered (r : ℝ) (w : ℝ) (h₀ : r = 7) (h₁ : w = 4) :
  let total_area := π * r^2,
      inner_radius := r - w,
      inner_area := π * inner_radius^2,
      path_area := total_area - inner_area,
      remaining_area := total_area - path_area in
  remaining_area = 29 * π :=
by { sorry }

end remaining_garden_area_not_covered_l562_562363


namespace evaluate_g_l562_562762

def g : ℝ → ℝ := λ x, if x > 3 then x^2 else if x < 3 then 2*x else 3*x

theorem evaluate_g (x : ℝ) (h : x = 1) : g (g (g x)) = 16 := by
  unfold g
  -- Using sorry here as placeholder for the actual proof
  sorry

end evaluate_g_l562_562762


namespace octagon_area_inscribed_in_circle_l562_562310

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562310


namespace part1_part2_l562_562193

theorem part1 (n : ℕ) (h_prime : Nat.prime (2 * n - 1)) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ (i j : Fin n), i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) :=
sorry

theorem part2 (n : ℕ) (h_composite : ¬Nat.prime (2 * n - 1)) :
  ∃ (a : Fin n → ℕ), Function.Injective a ∧ ∀ (i j : Fin n), i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1) :=
sorry

end part1_part2_l562_562193


namespace minimum_framing_feet_needed_l562_562676

theorem minimum_framing_feet_needed :
  let original_width := 4
  let original_height := 6
  let enlarged_width := 4 * original_width
  let enlarged_height := 4 * original_height
  let border := 3
  let total_width := enlarged_width + 2 * border
  let total_height := enlarged_height + 2 * border
  let perimeter := 2 * (total_width + total_height)
  let framing_feet := (perimeter / 12).ceil
  framing_feet = 9 := by
  -- The theorem statement translates given conditions into definitions and finally asserts the result.
  sorry

end minimum_framing_feet_needed_l562_562676


namespace trigonometric_values_and_identity_l562_562481

theorem trigonometric_values_and_identity 
  (θ : ℝ) 
  (h : (3, -4) ∈ { p : ℝ × ℝ | p = (3, -4) }) :
  sin θ = -(4/5) ∧ cos θ = 3/5 ∧ tan θ = -(4/3) ∧ 
  (cos (3 * π - θ) + cos (3 * π / 2 + θ)) / (sin (π / 2 - θ) + tan (π + θ)) = 21/11 :=
by
  sorry

end trigonometric_values_and_identity_l562_562481


namespace constantin_mother_deposit_return_l562_562555

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end constantin_mother_deposit_return_l562_562555


namespace solve_equation_l562_562766

theorem solve_equation (x : ℝ) (h_eq : 1 / (x - 2) = 3 / (x - 5)) : 
  x = 1 / 2 :=
  sorry

end solve_equation_l562_562766


namespace Marian_credit_card_balance_l562_562930

theorem Marian_credit_card_balance :
  let initial_balance := 126.00 in
  let groceries := 60.00 in
  let gas := groceries / 2 in
  let returned := 45.00 in
  initial_balance + groceries + gas - returned = 171.00 :=
by
  let initial_balance := 126.00
  let groceries := 60.00
  let gas := groceries / 2
  let returned := 45.00
  calc
    126.00 + 60.00 + 30.00 - 45.00 = 216.00 - 45.00 : by congr
    ... = 171.00 : by norm_num

#suppressAllProofSteps

end Marian_credit_card_balance_l562_562930


namespace solution_set_of_inequality_system_l562_562263

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l562_562263


namespace ellen_smoothie_l562_562007

theorem ellen_smoothie :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries_used := total_ingredients - (yogurt + orange_juice)
  strawberries_used = 0.2 := by
  sorry

end ellen_smoothie_l562_562007


namespace parallel_lines_slope_l562_562329

theorem parallel_lines_slope (b : ℝ) 
  (h₁ : ∀ x y : ℝ, 3 * y - 3 * b = 9 * x → (b = 3 - 9)) 
  (h₂ : ∀ x y : ℝ, y + 2 = (b + 9) * x → (b = 3 - 9)) : b = -6 :=
by
  sorry

end parallel_lines_slope_l562_562329


namespace tangent_line_at_point_l562_562219

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562219


namespace altitude_length_of_isosceles_triangle_GEM_l562_562601

noncomputable def length_of_altitude (side_length : ℝ) (common_area : ℝ) : ℝ :=
  if side_length = 10 ∧ common_area = 80
  then 25
  else 0

theorem altitude_length_of_isosceles_triangle_GEM :
  ∀ (AIME : Type) [has_side : Π (s : AIME), ℝ] (EM : ℝ) (G : AIME) (alt : ℝ),
  has_side AIME = 10 ∧
  is_isosceles_triangle AIME EM G ∧
  common_area_triangle_and_square AIME EM G = 80 →
  alt = 25 :=
by
  intros,
  sorry

end altitude_length_of_isosceles_triangle_GEM_l562_562601


namespace find_cost_price_of_radio_l562_562721

def cost_price_of_radio
  (profit_percent: ℝ) (overhead_expenses: ℝ) (selling_price: ℝ) (C: ℝ) : Prop :=
  profit_percent = ((selling_price - (C + overhead_expenses)) / C) * 100

theorem find_cost_price_of_radio :
  cost_price_of_radio 21.457489878542503 15 300 234.65 :=
by
  sorry

end find_cost_price_of_radio_l562_562721


namespace terminating_decimals_count_l562_562442

theorem terminating_decimals_count : 
  {n : ℕ | 1 ≤ n ∧  n ≤ 1000 ∧ ∃ k : ℕ, n = 21 * k }.finite.card = 47 := 
by 
  sorry

end terminating_decimals_count_l562_562442


namespace xiao_ming_final_score_correct_l562_562335

/-- Xiao Ming's scores in image, content, and effect are 9, 8, and 8 points, respectively.
    The weights (ratios) for these scores are 3:4:3.
    Prove that Xiao Ming's final competition score is 8.3 points. -/
def xiao_ming_final_score : Prop :=
  let image_score := 9
  let content_score := 8
  let effect_score := 8
  let image_weight := 3
  let content_weight := 4
  let effect_weight := 3
  let total_weight := image_weight + content_weight + effect_weight
  let weighted_score := (image_score * image_weight) + (content_score * content_weight) + (effect_score * effect_weight)
  weighted_score / total_weight = 8.3

theorem xiao_ming_final_score_correct : xiao_ming_final_score := by
  sorry

end xiao_ming_final_score_correct_l562_562335


namespace max_players_with_at_least_54_points_l562_562631

theorem max_players_with_at_least_54_points (n : ℕ) (players : fin n → ℕ) :
  n = 90 →
  (∀ i j, i ≠ j → (players i + players j = 1 ∨ players i + players j = 0.5)) →
  (∃ (k : ℕ), k ≤ 71 ∧ (∀ i : fin n, players i ≥ 54 → i ∈ {i : fin n | i < k})) :=
by
  intros n_eq ninety_players
  have total_games : 4005 := by
    rw n_eq
    apply finset.card
    exact finset.triangle_count
  have total_points : 4005 := total_games
  sorry

end max_players_with_at_least_54_points_l562_562631


namespace reciprocal_of_G_is_S_l562_562539

noncomputable def G : ℂ := -0.6 + 0.8 * complex.i
noncomputable def S : ℂ := -0.8 - 0.6 * complex.i

theorem reciprocal_of_G_is_S : (1 / G) = S := 
by
  sorry

end reciprocal_of_G_is_S_l562_562539


namespace ones_digit_17_exp_l562_562026

theorem ones_digit_17_exp (n : ℕ) : Nat.digit 10 (17 ^ (17 * (13 ^ 13))) = 7 := by
  -- Define the necessary conditions
  have h1 : ∀ m : ℕ, Nat.digit 10 (17 ^ m) = Nat.digit 10 (7 ^ m) := by -- condition 1
    sorry

  have h2 : ∀ m : ℕ, Nat.digit 10 (7 ^ (m + 4)) = Nat.digit 10 (7 ^ m) := by -- condition 2 (pattern every 4)
    sorry

  -- Use the established conditions to reach the conclusion
  sorry

end ones_digit_17_exp_l562_562026


namespace average_distance_is_six_l562_562717

def side_length : ℝ := 12
def diagonal_hop : ℝ := 7.2
def right_turn_hop : ℝ := 3

def calculate_average_distance (x y : ℝ) : ℝ :=
  let left_side_dist := x
  let bottom_side_dist := y
  let right_side_dist := side_length - x
  let top_side_dist := side_length - y
  (left_side_dist + bottom_side_dist + right_side_dist + top_side_dist) / 4

theorem average_distance_is_six :
  let final_x := 8.1
  let final_y := 5.1
  calculate_average_distance final_x final_y = 6 :=
by
  sorry

end average_distance_is_six_l562_562717


namespace value_of_4a_plus_b_l562_562544

theorem value_of_4a_plus_b (a b : ℝ) (E F G : ℝ × ℝ) (hE : E = (a-1, a))
  (hF : F = (b, a-b)) (hG : G = ((E.1 + F.1) / 2, (E.2 + F.2) / 2)) 
  (hG_on_y_axis : G.1 = 0) (hG_distance : abs G.2 = 1) :
  4 * a + b = 4 ∨ 4 * a + b = 0 :=
begin
  sorry
end

end value_of_4a_plus_b_l562_562544


namespace mat_radius_increase_l562_562364

theorem mat_radius_increase (C1 C2 : ℝ) (h1 : C1 = 40) (h2 : C2 = 50) :
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  (r2 - r1) = 5 / Real.pi := by
  sorry

end mat_radius_increase_l562_562364


namespace vertex_of_parabola_l562_562983

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = (x - 6)^2 + 3 ↔ (x = 6 ∧ y = 3)) :=
sorry

end vertex_of_parabola_l562_562983


namespace find_function_l562_562776

-- Define the function f : ℝ → ℝ
variable {f : ℝ → ℝ}

-- Define the functional equation as a condition
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f(x^2 - y^2) = (x - y) * (f(x) + f(y))

-- Define the proof problem statement
theorem find_function (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f(x) = k * x :=
sorry

end find_function_l562_562776


namespace parallel_lines_equal_slopes_l562_562819

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end parallel_lines_equal_slopes_l562_562819


namespace quadratic_eq_equal_roots_l562_562801

theorem quadratic_eq_equal_roots (m x : ℝ) (h : (x^2 - m * x + m - 1 = 0) ∧ ((x - 1)^2 = 0)) : 
    m = 2 ∧ ((x = 1 ∧ x = 1)) :=
by
  sorry

end quadratic_eq_equal_roots_l562_562801


namespace product_of_numbers_l562_562412

noncomputable def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem product_of_numbers {a b c : ℕ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_lcm : lcm (lcm a b) c = 1800) (h_hcf : Nat.gcd (Nat.gcd a b) c = 12) :
  a * b * c = 21600 :=
sorry

end product_of_numbers_l562_562412


namespace Giselle_yellow_paint_ratio_l562_562792

theorem Giselle_yellow_paint_ratio (R Y B : ℕ) (h_ratio : R: Y : B = 5: 3 : 7)
(h_blue : B = 21) : Y = 9 := by
  -- Definitions and conditions from part a)
  have h_total_ratio : 5 + 3 + 7 = 15 := by norm_num
  have h_blue_ratio : 7 * (21 / 7) = 21 := by {
    have h_div : 21 / 7 = 3 := by norm_num,
    rw ← h_div,
    norm_num
  }
  -- Calculation of yellow paint based on the given ratio and amount of blue paint
  have h_yellow_calc : Y = 3 * (21 / 7) := by {
    rw h_blue_ratio,
    norm_num
  }
  -- Final amount of yellow paint
  exact h_yellow_calc.trans (by norm_num)

end Giselle_yellow_paint_ratio_l562_562792


namespace problem_equivalence_l562_562879

noncomputable def semicircle_parametric_eq : Prop :=
  ∀ θ ∈ set.Icc 0 (Real.pi / 2), 
  let ρ := 2 * Real.cos θ in
  ρ ^ 2 = 2 * ρ * Real.cos θ

noncomputable def point_D_coordinates : Prop :=
  let α := Real.pi / 3 in
  ∃ (x y : ℝ),
    (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 - 1) ^ 2 + p.2 ^ 2 = 1)
    ∧ ( ∃ (y' : ℝ),
        y' = Real.sqrt 3 * x + 2
        ∧ x = 1 + Real.cos α
        ∧ y = Real.sin α )
    ∧ x = 3 / 2 ∧ y = Real.sqrt 3 / 2

theorem problem_equivalence : semicircle_parametric_eq ∧ point_D_coordinates :=
  sorry

end problem_equivalence_l562_562879


namespace EHFO_is_parallelogram_l562_562573

variables {A B C E F H O : Type*} -- Variables for points
variables [triangle : Triangle A B C] -- A triangle A B C
variables (E : FootOfAltitude B A C) -- Foot of the altitude from B to AC
variables (F : FootOfAltitude C A B) -- Foot of the altitude from C to AB
variables (H : Intersect BE FC) -- Intersection point of lines BE and FC
variables (O : Circumcenter A B C) -- Circumcenter of triangle ABC

-- Assuming AF = FC
def isosceles_right_triangle_AFC := 
  ∀ {A B C : Type*} (A F C : Triangle A B C), A F = F C

-- Prove EHFO is a parallelogram
theorem EHFO_is_parallelogram 
  (triangle_ABC : Triangle A B C)
  (altitude_B_E : FootOfAltitude B A C)
  (altitude_C_F : FootOfAltitude C A B)
  (intersection_H : Intersect BE FC)
  (circumcenter_O : Circumcenter A B C)
  (isosceles_AF_FC : isosceles_right_triangle_AFC):
  Parallelogram EHFO := 
sorry

end EHFO_is_parallelogram_l562_562573


namespace martha_apples_l562_562176

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l562_562176


namespace domain_fraction_function_l562_562817

theorem domain_fraction_function (f : ℝ → ℝ):
  (∀ x : ℝ, -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 0) →
  (∀ x : ℝ, x ≠ 0 → -2 ≤ x ∧ x < 0) →
  (∀ x, (2^x - 1) ≠ 0) →
  true := sorry

end domain_fraction_function_l562_562817


namespace chord_difference_l562_562583

noncomputable def parabola_focus : (p : ℝ) → ℝ × ℝ := λ p, (p / 2, 0)
noncomputable def parabola_directrix : (p : ℝ) → Set (ℝ × ℝ) := λ p, {pt | pt.1 = -p / 2}

theorem chord_difference (p : ℝ) (hp : p > 0) (A B F : ℝ × ℝ) (C : ℝ × ℝ)
  (h_parabola : (A.2)^2 = 2 * p * A.1 ∧ (B.2)^2 = 2 * p * B.1) 
  (h_focus : F = parabola_focus p)
  (h_directrix : C ∈ parabola_directrix p)
  (h_chord : A ≠ B ∧ ∀ t, (A.1 - F.1) * t + F.1 = B.1 ∧ (A.2 - F.2) * t + F.2 = B.2 )
  (h_angle : ∠CBF = π / 2) :
  (Real.dist A.1 F.1 - Real.dist B.1 F.1) = 2 * p := sorry

end chord_difference_l562_562583


namespace triangle_perimeter_ellipse_l562_562468

-- Define what it means to be a focus of the ellipse
def is_focus (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (sqrt (a^2 - b^2), 0) ∨ P = (-sqrt (a^2 - b^2), 0)

-- Define the ellipse equation
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Given ellipse parameters
def a : ℝ := 2
def b : ℝ := sqrt 3

-- Define the points M and N being intersection points on the ellipse
def is_intersection (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ ellipse x y a b

-- The main theorem to prove
theorem triangle_perimeter_ellipse (F1 F2 M N : ℝ × ℝ) :
  is_focus F1 a b ∧ is_focus F2 a b ∧ is_intersection M a b ∧ is_intersection N a b ∧ M ≠ N ∧
  (∃ l, l ≠ y=0 ∧
    l = λ x : ℝ, (F1.2 / F1.1) * x + (F1.2 - (F1.2 / F1.1) * F1.1) ∧
    ∃ x1 y1 x2 y2, M = (x1, y1) ∧ N = (x2, y2) ∧
    ellipse x1 y1 a b ∧ ellipse x2 y2 a b) →
  euclidean_dist M F1 + euclidean_dist M F2 + euclidean_dist N F1 + euclidean_dist N F2 = 8 :=
begin
  sorry
end

end triangle_perimeter_ellipse_l562_562468


namespace distance_on_number_line_2_neg5_distance_on_number_line_neg2_neg5_distance_on_number_line_1_neg3_distance_between_x_neg1_eq_2_min_value_range_x_l562_562664

theorem distance_on_number_line_2_neg5 : abs (2 - (-5)) = 7 := sorry

theorem distance_on_number_line_neg2_neg5 : abs (-2 - (-5)) = 3 := sorry

theorem distance_on_number_line_1_neg3 : abs (1 - (-3)) = 4 := sorry

theorem distance_between_x_neg1_eq_2 (x : ℝ) : abs (x + 1) = 2 → x = 1 ∨ x = -3 := sorry

theorem min_value_range_x (x : ℝ) : -1 ≤ x ∧ x ≤ 2 ↔ |x + 1| + |x - 2| = min (|x + 1| + |x - 2|) := sorry

end distance_on_number_line_2_neg5_distance_on_number_line_neg2_neg5_distance_on_number_line_1_neg3_distance_between_x_neg1_eq_2_min_value_range_x_l562_562664


namespace sum_of_positive_differences_eq_787484_l562_562154

open Finset

def S : Finset ℕ := (range 11).image (λ x, 3^x)

theorem sum_of_positive_differences_eq_787484 : 
  let differences := S.product S.filter (λ p, p.1 < p.2) in
  let positive_diffs := differences.map (λ p, p.2 - p.1) in
  positive_diffs.sum = 787484 :=
by
  sorry

end sum_of_positive_differences_eq_787484_l562_562154


namespace sin_cos_eq_one_sol_set_l562_562779

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_cos_eq_one_sol_set_l562_562779


namespace tangent_line_equation_l562_562222

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562222


namespace price_of_one_shirt_l562_562633

variable (P : ℝ)

-- Conditions
def cost_two_shirts := 1.5 * P
def cost_three_shirts := 1.9 * P 
def full_price_three_shirts := 3 * P
def savings := full_price_three_shirts - cost_three_shirts

-- Correct answer
theorem price_of_one_shirt (hs : savings = 11) : P = 10 :=
by
  sorry

end price_of_one_shirt_l562_562633


namespace problem_1_problem_2_l562_562835

def f (x : ℝ) : ℝ := (sin x - cos x) * sin (2 * x) / sin x

theorem problem_1:
  (∀ x, f x = 2 * (sin x - cos x) * cos x) →
  (∀ x, x ∉ {k * π | k : ℤ}) →
  (∀ x, (x ∉ {k * π | k : ℤ}) ↔ (x ∉ {k * π | k : ℤ})) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) :=
sorry

theorem problem_2:
  (∀ x, f x = 2 * (sin x - cos x) * cos x) →
  (∀ x, x ∉ {k * π | k : ℤ}) →
  (∀ k ∈ ℤ, 
    ((k * π - π / 8 < x ∧ x < k * π) ∨ 
     (k * π < x ∧ x < k * π + 3 * π / 8) → 
     (∃ δ > 0, ∀ y, 
       y ∈ { t | t ∈ ((k * π - π / 8), k * π) ∪ (k * π, k * π + 3 * π / 8)} 
      ∧ abs (y - x) < δ → f y > f x))) :=
sorry

end problem_1_problem_2_l562_562835


namespace ratio_of_areas_l562_562903

noncomputable def area_of_equilateral_triangle (x : ℝ) : ℝ :=
  (sqrt 3 / 4) * x^2

theorem ratio_of_areas (x : ℝ) (h : x > 0) :
  let ABC_area := area_of_equilateral_triangle x in
  let A'B'C'_area := area_of_equilateral_triangle (2 * x) in
  A'B'C'_area / ABC_area = 4 := 
by 
  sorry

end ratio_of_areas_l562_562903


namespace bird_counting_problem_l562_562382

namespace BirdCounting

-- Define the initial conditions
def initial_blackbirds : ℕ := 3 * 7
def initial_magpies : ℕ := 13
def initial_bluejays : ℕ := 2 * 5
def initial_robins : ℕ := 4

-- Define the changes
def blackbirds_after_change : ℕ := initial_blackbirds - 6
def magpies_after_change : ℕ := initial_magpies + 8
def bluejays_after_change : ℕ := initial_bluejays + 3
def robins_after_change : ℕ := initial_robins

-- Calculate the total number of birds after the changes
def total_birds_after_change : ℕ :=
  blackbirds_after_change + magpies_after_change + bluejays_after_change + robins_after_change

-- Calculate the percentage changes
def perc_change_blackbirds : ℚ :=
  (blackbirds_after_change - initial_blackbirds : ℚ) / initial_blackbirds * 100

def perc_change_magpies : ℚ :=
  (magpies_after_change - initial_magpies : ℚ) / initial_magpies * 100

def perc_change_bluejays : ℚ :=
  (bluejays_after_change - initial_bluejays : ℚ) / initial_bluejays * 100

def perc_change_robins : ℚ :=
  (robins_after_change - initial_robins : ℚ) / initial_robins * 100

-- The final ratio of the birds (blackbirds : magpies : bluejays : robins)
def bird_ratio : List ℕ := [blackbirds_after_change, magpies_after_change, bluejays_after_change, robins_after_change]

theorem bird_counting_problem :
  total_birds_after_change = 53 ∧
  perc_change_blackbirds ≈ -28.57 ∧
  perc_change_magpies ≈ 61.54 ∧
  perc_change_bluejays = 30 ∧
  perc_change_robins = 0 ∧
  bird_ratio = [15, 21, 13, 4] :=
by
  sorry -- Proof steps would go here
end BirdCounting

end bird_counting_problem_l562_562382


namespace tangent_line_equation_at_point_l562_562214

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562214


namespace octagon_area_l562_562299

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562299


namespace infinite_sum_of_angles_l562_562185

noncomputable def infinite_sum_angle (A B : ℕ → ℝ) (α : ℝ) : ℝ :=
∑' i, -

theorem infinite_sum_of_angles (A B : ℕ → ℝ) (α : ℝ) (distA : ∀ i : ℕ, |A i - A (i+1)| = 1)
(distB : ∀ i : ℕ, |B i - B (i+1)| = 2) 
(angle_cond : ∀ i : ℕ, ∠(A 0) (A 1) (B 0) = α) :
∑' i, ∠(A i) (B i) (A (i + 1)) = 180 - α :=
sorry

end infinite_sum_of_angles_l562_562185


namespace roller_coaster_costs_4_l562_562546

-- Definitions from conditions
def tickets_initial: ℕ := 5                     -- Jeanne initially has 5 tickets
def tickets_to_buy: ℕ := 8                      -- Jeanne needs to buy 8 more tickets
def total_tickets_needed: ℕ := tickets_initial + tickets_to_buy -- Total tickets needed
def tickets_ferris_wheel: ℕ := 5                -- Ferris wheel costs 5 tickets
def tickets_total_after_ferris_wheel: ℕ := total_tickets_needed - tickets_ferris_wheel -- Remaining tickets after Ferris wheel

-- Definition to be proved (question = answer)
def cost_roller_coaster_bumper_cars: ℕ := tickets_total_after_ferris_wheel / 2 -- Each of roller coaster and bumper cars cost

-- The theorem that corresponds to the solution
theorem roller_coaster_costs_4 :
  cost_roller_coaster_bumper_cars = 4 :=
by
  sorry

end roller_coaster_costs_4_l562_562546


namespace max_value_f_k_range_l562_562570

noncomputable def f (x : ℝ) : ℝ := (x / Real.exp x) + Real.log x - x

/-
Statement 1:
Prove that the maximum value of f(x) on (0, ∞) is f(1) = 1/e - 1
-/
theorem max_value_f :
  ∃ x ∈ set.Ioi 0, ∀ y ∈ set.Ioi 0, f y ≤ f x :=
begin
  use 1,
  split,
  { exact one_pos },
  { intro y,
    assume hy : y > 0,
    -- proof steps to be filled in
    sorry }
end

/-
Statement 2:
Given f(x₁) = f(x₂) for x₁ < x₂, and k x₁ + x₂ having a minimum value, prove that k ∈ (1, ∞)
-/
theorem k_range (x₁ x₂ : ℝ) (h₁ : 0 < x₁ < x₂) (h₂ : f x₁ = f x₂) :
  ∃ k : ℝ, 1 < k ∧ (∀ y < k, k * x₁ + x₂ < y * x₁ + x₂) :=
begin
  -- proof steps to be filled in
  sorry
end

end max_value_f_k_range_l562_562570


namespace octagon_area_l562_562298

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562298


namespace min_framing_required_l562_562685

-- Define the original dimensions
def original_length := 4 -- inches
def original_width := 6 -- inches

-- Define the scale factor
def scale_factor := 4

-- Define the border width
def border_width := 3 -- inches

-- Define the function to compute the new dimensions after enlarging
def enlarged_dimensions (length width : ℕ) : ℕ × ℕ :=
  (length * scale_factor, width * scale_factor)

-- Define the function to compute the dimensions after adding the border
def dimensions_with_border (length width : ℕ) : ℕ × ℕ :=
  (length + 2 * border_width, width + 2 * border_width)

-- Prove that the minimum number of linear feet of framing required is 9
theorem min_framing_required :
  let (enlarged_length, enlarged_width) := enlarged_dimensions original_length original_width
      (final_length, final_width) := dimensions_with_border enlarged_length enlarged_width
      perimeter := 2 * (final_length + final_width)
      perimeter_in_feet := (perimeter + 11) / 12 -- add 11 to effectively round up to the next foot
  in perimeter_in_feet = 9 :=
by
  -- Skipping proof, focusing only on statement structure.
  sorry

end min_framing_required_l562_562685


namespace father_twice_marika_age_in_2036_l562_562946

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l562_562946


namespace regular_octagon_area_l562_562321

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562321


namespace tangent_line_equation_l562_562223

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562223


namespace smallest_positive_n_l562_562069

theorem smallest_positive_n (n : ℕ) (z : ℂ) (h : z = (complex.mk (sqrt 3) (-3))^n) (hz_real : z.im = 0) : n = 3 :=
sorry

end smallest_positive_n_l562_562069


namespace marksman_target_breaking_orders_l562_562531

theorem marksman_target_breaking_orders : 
  let targets := 10
  let columnA := 4
  let columnB := 3
  let columnC := 3
  targets = columnA + columnB + columnC →
  ∃ n, nat.perm (columnA + columnB + columnC) columnA columnB columnC = n ∧ n = 4200 := 
by
  sorry

end marksman_target_breaking_orders_l562_562531


namespace tan_double_alpha_l562_562071

theorem tan_double_alpha (α : ℝ) (h : ∀ x : ℝ, (3 * Real.sin x + Real.cos x) ≤ (3 * Real.sin α + Real.cos α)) :
  Real.tan (2 * α) = -3 / 4 :=
sorry

end tan_double_alpha_l562_562071


namespace problem_statement_l562_562467

-- Definitions for the propositions p and q
def p : Prop := ∃ x : ℝ, x^2 + 1 < 2 * x
def q : Prop := ∀ m : ℝ, (∀ x : ℝ, mx^2 - mx + 1 > 0) → (0 < m ∧ m < 4)

-- The proof statement asserting that p ∨ q is false
theorem problem_statement : ¬ (p ∨ q) :=
by
  sorry

end problem_statement_l562_562467


namespace probability_prime_sum_l562_562692

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end probability_prime_sum_l562_562692


namespace tent_fabric_area_tent_volume_calc_tent_avg_height_calc_l562_562604

variable (a : ℝ) (h_a : a = 15)

def tent_conditions :=
  (a = 15) ∧
  (true) ∧ -- Boundary condition about arcs and segments (not explicitly used later)
  (true) ∧ -- Arc centers and radius
  (true) ∧ -- Straight segment condition (external common tangents)
  (true) ∧ -- Columns at vertices with specific heights
  (true) ∧ -- Conical surfaces and planar quadrangles

noncomputable def tent_area := (3 * Real.sqrt 2 * (Float.pi + Real.sqrt 3) + 2 * Real.sqrt 5) * a^2 / 4
noncomputable def tent_volume := (99 + 21 * Real.sqrt 3 + 17 * Float.pi) * a^3 / 72
noncomputable def tent_avg_height := tent_volume / (5 * a^2 / 2)

theorem tent_fabric_area : tent_area a = 1892 :=
  by
  unfold tent_area
  sorry

theorem tent_volume_calc : tent_volume a = 8850 :=
  by
  unfold tent_volume
  sorry

theorem tent_avg_height_calc : tent_avg_height a = 6.40 :=
  by
  unfold tent_avg_height
  sorry

end tent_fabric_area_tent_volume_calc_tent_avg_height_calc_l562_562604


namespace room_length_correct_l562_562113

def Room (length breadth height : ℝ) := length * breadth * height

variables (L B H A : ℝ)
variable numberOfBricksPerSquareMeter : ℝ 
variable totalNumberOfBricks : ℝ 

def Area (length breadth : ℝ) := length * breadth

noncomputable def lengthOfRoom :=
  if H = 2 ∧ B = 5 
     ∧ numberOfBricksPerSquareMeter = 17 
     ∧ totalNumberOfBricks = 340
     then (totalNumberOfBricks / numberOfBricksPerSquareMeter) / B
     else 0

theorem room_length_correct : 
  ∀ (B H numberOfBricksPerSquareMeter totalNumberOfBricks : ℝ), 
  B = 5 ∧ H = 2 ∧ numberOfBricksPerSquareMeter = 17 ∧ totalNumberOfBricks = 340 -> lengthOfRoom = 4 := by
  sorry

end room_length_correct_l562_562113


namespace tangent_line_at_point_l562_562218

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562218


namespace largest_possible_n_l562_562592

/-- Given a set of distinct integers with specified properties, 
    this theorem asserts the largest possible cardinality of such a set. -/
theorem largest_possible_n (s : Set ℤ) (h1 : ∀ a ∈ s, ∀ b ∈ s, a ≤ b → b = a ∨ b = a + 1)
                           (h2 : (∀ a ∈ s, ∀ b ∈ s, (b = a + 1) → a * b = 77))
                           (h3 : (∀ a ∈ s, ∀ b ∈ s, (b = a - 1) → a * b = 77)) :
  (s.card ≤ 17) :=
sorry

end largest_possible_n_l562_562592


namespace min_framing_required_l562_562688

-- Define the original dimensions
def original_length := 4 -- inches
def original_width := 6 -- inches

-- Define the scale factor
def scale_factor := 4

-- Define the border width
def border_width := 3 -- inches

-- Define the function to compute the new dimensions after enlarging
def enlarged_dimensions (length width : ℕ) : ℕ × ℕ :=
  (length * scale_factor, width * scale_factor)

-- Define the function to compute the dimensions after adding the border
def dimensions_with_border (length width : ℕ) : ℕ × ℕ :=
  (length + 2 * border_width, width + 2 * border_width)

-- Prove that the minimum number of linear feet of framing required is 9
theorem min_framing_required :
  let (enlarged_length, enlarged_width) := enlarged_dimensions original_length original_width
      (final_length, final_width) := dimensions_with_border enlarged_length enlarged_width
      perimeter := 2 * (final_length + final_width)
      perimeter_in_feet := (perimeter + 11) / 12 -- add 11 to effectively round up to the next foot
  in perimeter_in_feet = 9 :=
by
  -- Skipping proof, focusing only on statement structure.
  sorry

end min_framing_required_l562_562688


namespace probability_even_sum_l562_562276

-- Definitions of probabilities for the first wheel
def P_even1 : ℚ := 1 / 2
def P_odd1 : ℚ := 1 / 2

-- Definitions of probabilities for the second wheel
def P_even2 : ℚ := 1 / 5
def P_odd2 : ℚ := 4 / 5

-- Probability that the sum of numbers from both wheels is even
def P_even_sum : ℚ := P_even1 * P_even2 + P_odd1 * P_odd2

-- Theorem statement
theorem probability_even_sum : P_even_sum = 1 / 2 :=
by {
  sorry -- The proof is not required
}

end probability_even_sum_l562_562276


namespace correct_statements_l562_562124

def circle_equation : Prop :=
  ∀ C D : ℝ × ℝ,
  C.1^2 + C.2^2 = 4 ∧ D.1^2 + D.2^2 = 4 →
  ∃ A B O' : ℝ × ℝ,
  A = (-3, 0) ∧ B = (-1, 2) ∧ O' = (-3, 3) ∧
  ∀ x y : ℝ, (x + 3)^2 + (y - 3)^2 = 4

def parallelogram_min_area : Prop :=
  ∀ C D : ℝ × ℝ,
  C.1^2 + C.2^2 = 4 ∧ D.1^2 + D.2^2 = 4 →
  ∃ A B : ℝ × ℝ,
  A = (-3, 0) ∧ B = (-1, 2) ∧
  let AB := (2 : ℝ) in
  let CD := (2 * (2 : ℝ) ^ (1 / 2)) in
  let abcd_is_parallelogram :=
    (B.1 - A.1) = (D.1 - C.1) ∧ (B.2 - A.2) = (D.2 - C.2) ∧
    (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 8 ∧
    (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = 8 ∧ (A.1 = B.1 → A.2 - B.2 ≠ 0) in
  abcd_is_parallelogram →
  let area := 2 in
  area = 2

theorem correct_statements :
  circle_equation ∧ parallelogram_min_area :=
by
  split
  all_goals { sorry }

end correct_statements_l562_562124


namespace annual_rent_per_sqft_l562_562990

theorem annual_rent_per_sqft
  (length width monthly_rent : ℕ)
  (H_length : length = 10)
  (H_width : width = 8)
  (H_monthly_rent : monthly_rent = 2400) :
  (12 * monthly_rent) / (length * width) = 360 := by
  sorry

end annual_rent_per_sqft_l562_562990


namespace inequality_must_hold_l562_562863

variable {R : Type*} [LinearOrderedField R] {f : R → R} {a b : R}

theorem inequality_must_hold (h1 : ∀ (x : R), x * (f' x) > -f x) (h2 : a > b) : a * f(a) > b * f(b) :=
sorry

end inequality_must_hold_l562_562863


namespace collinear_M_H_K_l562_562151

-- Assume the context of the given problem.

variables {A B C H H_B H_C K M : Type} [metric_space A] [metric_space B] 
[metric_space C] [metric_space H] [metric_space H_B] 
[metric_space H_C] [metric_space K] [metric_space M]

-- Define the necessary points and circles.
variable [triangle ABC]
variable [orthocenter H ABC]
variable [circumcircle Γ ABC]
variable [foot H_B B H]
variable [foot H_C C H]
variable [midpoint M B C]
variable [second_intersection_point K Γ (circumcircle AH_BH_C)]

-- State the theorem.
theorem collinear_M_H_K : collinear {M, H, K} :=
sorry

end collinear_M_H_K_l562_562151


namespace f_g_of_4_eq_18_sqrt_21_div_7_l562_562569

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x - 3

theorem f_g_of_4_eq_18_sqrt_21_div_7 : f (g 4) = (18 * Real.sqrt 21) / 7 := by
  sorry

end f_g_of_4_eq_18_sqrt_21_div_7_l562_562569


namespace fish_caught_in_second_catch_l562_562869

theorem fish_caught_in_second_catch
  (H50_tagged: ∀ N : ℕ, 50 tagged and returned to the pond)
  (H10_tagged_in_second_catch: 10 fish were tagged and caught again)
  (Hratio_approx: ∀ X N : ℕ, percent of tagged fish in the second catch approximates the percent in the pond)
  (Happrox_N: N ≈ 250) : X = 50 :=
by
  sorry

end fish_caught_in_second_catch_l562_562869


namespace no_polynomial_factorization_l562_562108

noncomputable def polynomial (R : Type*) [CommRing R] := R[X]

theorem no_polynomial_factorization
  (n : ℕ)
  (a : Fin n → ℤ) -- a_i's are the inputs as a finite function of distinct integers
  (distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j) -- condition for distinctness
  :
  ¬ ∃ (f g : polynomial ℤ),
    f.degree ≥ 1 ∧ g.degree ≥ 1 ∧
    (∀ (x : ℤ), f.eval x = 0) ∧ 
    (∀ (x : ℤ), g.eval x = 0) ∧ 
    (∀ (x : ℤ), (eval₂ (RingHom.id ℤ) x ((X - C (a 0)) * (X - C (a 1)) * ... * (X - C (a (Fin.last n))) - 1)) = f.eval x * g.eval x) :=
sorry

end no_polynomial_factorization_l562_562108


namespace problem_propositions_correct_l562_562471

variables {L : Type} [LinearOrderedField L]
variables (a b : Line L) (α β γ : Plane L)

-- Conditions 
def non_coincident_lines (a b : Line L) : Prop := a ≠ b
def pairwise_non_coincident_planes (α β γ : Plane L) : Prop := α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Propositions to Prove
def proposition_1 (a : Line L) (α β : Plane L) : Prop :=
  (a.perpendicular_to α ∧ a.perpendicular_to β) → (α.parallel_to β)

def proposition_4 (α β γ : Plane L) (a b : Line L) : Prop :=
  (α.parallel_to β ∧ (α.intersect γ = a) ∧ (β.intersect γ = b)) → (a.parallel_to b)

-- Theorem to Establish Propositions Correctness
theorem problem_propositions_correct (a b : Line L) (α β γ : Plane L) :
  non_coincident_lines a b →
  pairwise_non_coincident_planes α β γ →
  proposition_1 a α β ∧ proposition_4 α β γ a b :=
by
  sorry

end problem_propositions_correct_l562_562471


namespace pet_preferences_l562_562715

/-- A store has several types of pets: 20 puppies, 10 kittens, 8 hamsters, and 5 birds.
Alice, Bob, Charlie, and David each want a different kind of pet, with the following preferences:
- Alice does not want a bird.
- Bob does not want a hamster.
- Charlie does not want a kitten.
- David does not want a puppy.
Prove that the number of ways they can choose different types of pets satisfying
their preferences is 791440. -/
theorem pet_preferences :
  let P := 20    -- Number of puppies
  let K := 10    -- Number of kittens
  let H := 8     -- Number of hamsters
  let B := 5     -- Number of birds
  let Alice_options := P + K + H -- Alice does not want a bird
  let Bob_options := P + K + B   -- Bob does not want a hamster
  let Charlie_options := P + H + B -- Charlie does not want a kitten
  let David_options := K + H + B   -- David does not want a puppy
  let Alice_pick := Alice_options
  let Bob_pick := Bob_options - 1
  let Charlie_pick := Charlie_options - 2
  let David_pick := David_options - 3
  Alice_pick * Bob_pick * Charlie_pick * David_pick = 791440 :=
by
  sorry

end pet_preferences_l562_562715


namespace second_to_last_digit_of_n_squared_plus_2n_l562_562612
open Nat

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) (h : (n^2 + 2 * n) % 10 = 4) : ((n^2 + 2 * n) / 10) % 10 = 2 :=
  sorry

end second_to_last_digit_of_n_squared_plus_2n_l562_562612


namespace domain_of_f_l562_562426

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | x^2 - 4 >= 0 ∧ x^2 - 4 ≠ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end domain_of_f_l562_562426


namespace area_regular_octagon_in_circle_l562_562287

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562287


namespace charles_speed_without_music_l562_562758

-- Define speed with music
def speed_with_music : ℝ := 6

-- Define duration of the album in hours
def album_duration : ℝ := 40 / 60

-- Define total distance
def total_distance : ℝ := 6

-- Define total time in hours
def total_time : ℝ := 70 / 60

-- Define the target speed without music
def target_speed_without_music : ℝ := 4

-- The proof statement
theorem charles_speed_without_music :
  (total_distance - speed_with_music * album_duration) / (total_time - album_duration) = target_speed_without_music :=
by
  sorry

end charles_speed_without_music_l562_562758


namespace aunt_may_morning_milk_l562_562740

-- Defining the known quantities as variables
def evening_milk : ℕ := 380
def sold_milk : ℕ := 612
def leftover_milk : ℕ := 15
def milk_left : ℕ := 148

-- Main statement to be proven
theorem aunt_may_morning_milk (M : ℕ) :
  M + evening_milk + leftover_milk - sold_milk = milk_left → M = 365 := 
by {
  -- Skipping the proof
  sorry
}

end aunt_may_morning_milk_l562_562740


namespace credit_card_balance_l562_562931

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l562_562931


namespace contrapositive_true_l562_562808

theorem contrapositive_true (x : ℝ) : (x^2 - 2*x - 8 ≤ 0 → x ≥ -3) :=
by
  -- Proof omitted
  sorry

end contrapositive_true_l562_562808


namespace sum_of_digits_of_y_l562_562354

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_y :
  ∃ (y : ℕ), 1000 ≤ y ∧ y ≤ 9999 ∧ is_palindrome y ∧ is_palindrome (y + 10) ∧ sum_of_digits y = 28 :=
by
  use 9991
  split
  { linarith }
  split
  { linarith }
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }
  sorry

end sum_of_digits_of_y_l562_562354


namespace a_general_form_b_n_inequality_l562_562492

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := 2
| (n+1) := (sqrt 2 - 1) * (a n + 2)

-- Define the general form of a_n
def a_general (n : ℕ) : ℝ :=
  sqrt 2 * (sqrt 2 - 1) ^ n + sqrt 2

-- Prove that the general form of a_n is correct
theorem a_general_form (n : ℕ) : a n = a_general n :=
by sorry

-- Define the sequence b_n
def b : ℕ → ℝ
| 0     := 2
| (n+1) := (3 * b n + 4) / (2 * b n + 3)

-- Prove the inequality sqrt(2) < b_n <= a_(4n-3)
theorem b_n_inequality (n : ℕ) : sqrt 2 < b n ∧ b n ≤ a (4 * n - 3) :=
by sorry

end a_general_form_b_n_inequality_l562_562492


namespace minimum_value_of_f_l562_562430

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (4 * x) + 6 * Real.cos (3 * x) + 17 * Real.cos (2 * x) + 30 * Real.cos x

theorem minimum_value_of_f : ∃ x ∈ set.univ, f x = -18 :=
by
  sorry

end minimum_value_of_f_l562_562430


namespace overall_profit_percentage_l562_562372

-- Definitions based on the given conditions
def CP1 := 60  -- Cost price of the first book
def SP1 := 78  -- Selling price of the first book
def CP2 := 45  -- Cost price of the second book
def SP2 := 54  -- Selling price of the second book
def CP3 := 70  -- Cost price of the third book
def SP3 := 84  -- Selling price of the third book

-- Profit for each book
def P1 := SP1 - CP1
def P2 := SP2 - CP2
def P3 := SP3 - CP3

-- Total cost price
def TCP := CP1 + CP2 + CP3

-- Total selling price
def TSP := SP1 + SP2 + SP3

-- Total profit
def TP := P1 + P2 + P3

-- Profit percentage calculation
def profit_percentage : Float := (TP.toFloat / TCP.toFloat) * 100

-- The final theorem to prove the profit percentage is approximately 23.43%
theorem overall_profit_percentage : profit_percentage ≈ 23.43 :=
by 
  sorry

end overall_profit_percentage_l562_562372


namespace arithmetic_sequence_contains_term_l562_562247

theorem arithmetic_sequence_contains_term (a1 : ℤ) (d : ℤ) (k : ℕ) (h1 : a1 = 3) (h2 : d = 9) :
  ∃ n : ℕ, (a1 + (n - 1) * d) = 3 * 4 ^ k := by
  sorry

end arithmetic_sequence_contains_term_l562_562247


namespace max_min_values_triangle_inequality_k_l562_562487

def f (x k : ℝ) : ℝ := (x^4 + k * x^2 + 1) / (x^4 + x^2 + 1)

theorem max_min_values (k : ℝ) : 
  (k ≥ 1 → ∀ x : ℝ, 1 ≤ f x k ∧ f x k ≤ (k+2)/3) ∧ 
  (k < 1 → ∀ x : ℝ, (k+2)/3 ≤ f x k ∧ f x k ≤ 1) :=
sorry

theorem triangle_inequality_k (k : ℝ) : 
  (-1/2 < k ∧ k < 4) ↔ ∀ a b c : ℝ, 
    (f a k + f b k > f c k) ∧ 
    (f a k + f c k > f b k) ∧ 
    (f b k + f c k > f a k) :=
sorry

end max_min_values_triangle_inequality_k_l562_562487


namespace tangent_line_eq_l562_562241

theorem tangent_line_eq {x y : ℝ} (h : y = (2 * x - 1) / (x + 2)) (hx1 : x = -1) (hy1 : y = -3) : 
  ∃ m b, 5 * x - y + 2 = 0 :=
by
  sorry

end tangent_line_eq_l562_562241


namespace valid_x_for_expression_l562_562517

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end valid_x_for_expression_l562_562517


namespace general_formula_is_correct_find_k_value_l562_562804

variable {a_n : ℕ → ℤ} {a_1 a_3 S_k : ℤ} (k : ℕ)

-- Conditions
axiom h1 : a_1 = 1
axiom h2 : a_3 = -3
axiom h3 : S_k = -35

-- General formula and sum of arithmetic sequence
def arithmetic_sequence := ∀ n : ℕ, a_n = 1 + (n - 1) * (a_3 - a_1) / 2

def sum_of_sequence := ∀ (n : ℕ), S_k = n * (2 * a_1 + (n - 1) * (a_3 - a_1)) / 2

-- Goals
theorem general_formula_is_correct : arithmetic_sequence ∧ sum_of_sequence → 
                                     ∃ d : ℤ, d = -2 ∧ ∃ f : ℕ → ℤ, ∀ n, a_n = 3 - 2 * n := 
sorry

theorem find_k_value : arithmetic_sequence ∧ sum_of_sequence → k = 7 := 
sorry

end general_formula_is_correct_find_k_value_l562_562804


namespace pond_depth_range_l562_562635

theorem pond_depth_range (d : ℝ) (adam_false : d < 10) (ben_false : d > 8) (carla_false : d ≠ 7) : 
    8 < d ∧ d < 10 :=
by
  sorry

end pond_depth_range_l562_562635


namespace angle_AOC_is_45_or_15_l562_562786

theorem angle_AOC_is_45_or_15 (A O B C : Type) (α β γ : ℝ) 
  (h1 : α = 30) (h2 : β = 15) : γ = 45 ∨ γ = 15 :=
sorry

end angle_AOC_is_45_or_15_l562_562786


namespace sin_cos_eq_one_sol_set_l562_562780

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_cos_eq_one_sol_set_l562_562780


namespace total_money_raised_l562_562033

-- Define the conditions as stated in the problem
def num_students_brownies := 50
def brownies_per_student := 20
def num_students_cookies := 30
def cookies_per_student := 36
def num_students_donuts := 25
def donuts_per_student := 18

def price_per_brownie := 1.50
def price_per_cookie := 2.25
def price_per_donut := 3.00

-- Calculations for the total number of each baked good
def total_brownies := num_students_brownies * brownies_per_student
def total_cookies := num_students_cookies * cookies_per_student
def total_donuts := num_students_donuts * donuts_per_student

-- Calculations for the total amount of money raised from each type of baked good
def money_from_brownies := total_brownies * price_per_brownie
def money_from_cookies := total_cookies * price_per_cookie
def money_from_donuts := total_donuts * price_per_donut

-- The final proof statement
theorem total_money_raised : 
  money_from_brownies + money_from_cookies + money_from_donuts = 5280 := by
  sorry

end total_money_raised_l562_562033


namespace angle_comparison_l562_562827

theorem angle_comparison :
  let A := 60.4
  let B := 60.24
  let C := 60.24
  A > B ∧ B = C :=
by
  sorry

end angle_comparison_l562_562827


namespace length_of_AB_l562_562136

theorem length_of_AB (AC BC : ℝ) (h1 : AC = 6) (h2 : BC = 2) :
  AB = 2 * real.sqrt 10 :=
by sorry

end length_of_AB_l562_562136


namespace find_point_A_l562_562877

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end find_point_A_l562_562877


namespace number_of_functions_l562_562912

open Nat

noncomputable def count_functions (n k : ℕ) (hk : 1 < k ∧ k ≤ n) : ℕ := 
  binomial n k * binomial (n-1) (k-1)

theorem number_of_functions (n k : ℕ) (hn : 1 < n) :
  n > 1 → 1 < k ∧ k ≤ n →
  count_functions n k (by omega) = binomial n k * binomial (n-1) (k-1) :=
by
  intros
  sorry

end number_of_functions_l562_562912


namespace regression_line_correct_l562_562826

-- Define the necessary conditions
def positively_correlated (x y : ℝ) := sorry
def sample_means (x_mean y_mean : ℝ) := (x_mean, y_mean) = (3, 3.5)

-- Define the linear regression equation
def regression_equation (x y : ℝ) := y = 0.4 * x + 2.3

-- The theorem statement
theorem regression_line_correct : ∀ (x y : ℝ),
  positively_correlated x y →
  sample_means (3 : ℝ) (3.5 : ℝ) →
  regression_equation x y :=
by sorry

end regression_line_correct_l562_562826


namespace determine_set_f_values_l562_562909

-- Definitions of the conditions
def f (n : ℕ) (hn : n > 0) : ℚ :=
  (∑ k in Finset.range n, if even k then -1 else 1) / n

theorem determine_set_f_values :
  ∀ (n : ℕ) (hn : n > 0), f n hn ∈ ({0, 1 / n} : Set ℚ) :=
begin
  sorry
end

end determine_set_f_values_l562_562909


namespace mixture_ratio_l562_562662

noncomputable section

variable (P Q : ℝ)

def milk_in_p := (5 / 8) * P
def water_in_p := (3 / 8) * P
def milk_in_q := (1 / 4) * Q
def water_in_q := (3 / 4) * Q

theorem mixture_ratio (h : milk_in_p P + milk_in_q Q = water_in_p P + water_in_q Q) :
    P = 2 * Q :=
begin
  -- proof goes here
  sorry
end

end mixture_ratio_l562_562662


namespace tangent_line_to_curve_l562_562232

noncomputable def tangent_line_eq_at_point : Prop :=
  let f := λ (x : ℝ), (2 * x - 1) / (x + 2)
  tangent_line_eq f (-1 : ℝ, -3 : ℝ) = 5 * (λ (x : ℝ), x) - λ (y : ℝ), y + 2

theorem tangent_line_to_curve :
  tangent_line_eq_at_point :=
begin
  sorry
end

end tangent_line_to_curve_l562_562232


namespace area_of_acute_triangle_l562_562083

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) :=
  (1/2) * b * c * Real.sin A

theorem area_of_acute_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = Real.sqrt 19) (h2 : b = 5) (h3 : f A = 0) (h4 : f x = Real.cos (2 * x) + 1/2)
  (h_monotonic : ∀ {x : ℝ}, x ∈ set.Ioo (π / 2) π → f x = Real.cos (2 * x) + 1/2) :
  area_of_triangle A B C a b c = 15 * Real.sqrt 3 / 4 := 
begin
  sorry 
end

end area_of_acute_triangle_l562_562083


namespace find_x_l562_562493

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Lean statement asserting that if a is parallel to b for some x, then x = 2
theorem find_x (x : ℝ) (h : parallel a (b x)) : x = 2 := 
by sorry

end find_x_l562_562493


namespace periodic_sum_at_pi_quarter_l562_562455

noncomputable def f1(x : ℝ) : ℝ := sin x + cos x
noncomputable def f2(x : ℝ) : ℝ := deriv (λ x, f1 x) x
noncomputable def f3(x : ℝ) : ℝ := deriv (λ x, f2 x) x
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
if h : n > 0 then (finset.range (n - 1)).allowed_sum (fun k => deriv^[k] (λ x, f1 x) x)
else 0

theorem periodic_sum_at_pi_quarter : 
  f1 (π / 4) + f2 (π / 4) + ∑ i in finset.range 2016, (fn i.succ (π / 4)) = 0 :=
sorry

end periodic_sum_at_pi_quarter_l562_562455


namespace policeman_catches_thief_l562_562724

theorem policeman_catches_thief :
  (∃ (strategy : ℕ → ℝ), ∀ t : ℕ, strategy (t+1) ∈ {-2, 4} ∧ strategy 0 = 0) →
  (∀ thief_direction : ℕ → ℝ, 
   (thief_direction 0 = 0) ∧ 
   (∀ t : ℕ, thief_direction (t+1) = thief_direction t + d) ∧ 
   (d = 1 ∨ d = -1) →
   ∃ n : ℕ, strategy n = thief_direction n) := by
  sorry

end policeman_catches_thief_l562_562724


namespace josie_money_left_l562_562143

theorem josie_money_left :
  let gift_amount : ℤ := 150
  let cost_cassette_tapes : ℤ := 5 * 18
  let cost_headphone_sets : ℤ := 2 * 45
  let cost_vinyl_records : ℤ := 3 * 22
  let cost_magazines : ℤ := 4 * 7
  gift_amount - (cost_cassette_tapes + cost_headphone_sets + cost_vinyl_records + cost_magazines) = -124 :=
by {
  let gift_amount : ℤ := 150,
  let cost_cassette_tapes : ℤ := 5 * 18,
  let cost_headphone_sets : ℤ := 2 * 45,
  let cost_vinyl_records : ℤ := 3 * 22,
  let cost_magazines : ℤ := 4 * 7,
  have h : gift_amount - (cost_cassette_tapes + cost_headphone_sets + cost_vinyl_records + cost_magazines) = -124 := sorry,
  exact h,
}

end josie_money_left_l562_562143


namespace problem_1_problem_2a_problem_2b_l562_562496

noncomputable def v_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def v_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (v_a x).1 * (v_b).1 + (v_a x).2 * (v_b).2

theorem problem_1 (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) : 
  (v_a x).1 * (v_b).2 = (v_a x).2 * (v_b).1 → x = (5 * Real.pi / 6) :=
by
  sorry

theorem problem_2a : 
  ∃ x ∈ Set.Icc 0 Real.pi, f x = 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≤ 3 :=
by
  sorry

theorem problem_2b :
  ∃ x ∈ Set.Icc 0 Real.pi, f x = -2 * Real.sqrt 3 ∧ ∀ y ∈ Set.Icc 0 Real.pi, f y ≥ -2 * Real.sqrt 3 :=
by
  sorry

end problem_1_problem_2a_problem_2b_l562_562496


namespace regular_octagon_area_l562_562316

theorem regular_octagon_area (r : ℝ) (h₁ : π * r^2 = 256 * π)
  (h₂ : 8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2) :
  8 * (1 / 2 * r^2 * sin (π / 8) * cos (π / 8)) = 512 * real.sqrt 2 := 
sorry

end regular_octagon_area_l562_562316


namespace intensity_of_replacement_paint_l562_562972

theorem intensity_of_replacement_paint (f : ℚ) (I_new : ℚ) (I_orig : ℚ) (I_repl : ℚ) :
  f = 2/3 → I_new = 40 → I_orig = 60 → I_repl = (40 - 1/3 * 60) * (3/2) := by
  sorry

end intensity_of_replacement_paint_l562_562972


namespace correct_statements_l562_562059

noncomputable def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def circle_O1 (x y r : ℝ) (hr : r > 0) : Prop := (x - 2)^2 + y^2 = r^2

theorem correct_statements (r : ℝ) (hr : r > 0) :
  (∀ m : ℝ, ∃ x y : ℝ, circle_O1 x y r hr ∧ (m * x + y = m + 1) ∧ (m * x + y ≠ m + 1)) → r > sqrt 2 ∧
  (∀ A B : ℝ × ℝ, circle_O A.1 A.2 ∧ circle_O B.1 B.2 → (A.1 = 0 ∧ A.2 = 2) ∧ (B.1 = 0 ∧ B.2 = 2) → 
    dist A B = sqrt 3) :=
by
  sorry

end correct_statements_l562_562059


namespace sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l562_562437

theorem sin_30_eq_one_half : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

theorem cos_11pi_over_4_eq_neg_sqrt2_over_2 : Real.cos (11 * Real.pi / 4) = - Real.sqrt 2 / 2 :=
by 
  -- This is the statement only, the proof will be here
  sorry

end sin_30_eq_one_half_cos_11pi_over_4_eq_neg_sqrt2_over_2_l562_562437


namespace intersection_M_N_l562_562167

def M (x : ℝ) : Prop := (x + 1) / (x - 2) ≤ 0

def N (x : ℝ) : Prop := 2^x > 1/2 

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end intersection_M_N_l562_562167


namespace cubic_three_distinct_roots_in_interval_l562_562414

theorem cubic_three_distinct_roots_in_interval (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ x ∈ {x1, x2, x3}, x ∈ set.Ioo (-2) 4 ∧ x^3 + p * x + q = 0) ↔
  4 * p^3 + 27 * q^2 < 0 ∧ -4 * p - 64 < q ∧ q < 2 * p + 8 := sorry

end cubic_three_distinct_roots_in_interval_l562_562414


namespace regular_octagon_area_l562_562288

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562288


namespace log_system_base_log_values_l562_562525

theorem log_system_base (x : ℝ) (h : log x 450 - log x 40 = 1.1639) : x = 8 :=
by
  sorry

theorem log_values (x : ℝ) (hx : x = 8) : 
  (log x 40 ≈ 1.774) ∧ (log x 450 ≈ 2.9379) :=
by
  sorry

end log_system_base_log_values_l562_562525


namespace Konstantin_mother_returns_amount_l562_562549

theorem Konstantin_mother_returns_amount
  (deposit_usd : ℝ)
  (exchange_rate : ℝ)
  (equivalent_rubles : ℝ)
  (h_deposit_usd : deposit_usd = 10000)
  (h_exchange_rate : exchange_rate = 58.15)
  (h_equivalent_rubles : equivalent_rubles = deposit_usd * exchange_rate) :
  equivalent_rubles = 581500 :=
by {
  rw [h_deposit_usd, h_exchange_rate] at h_equivalent_rubles,
  exact h_equivalent_rubles
}

end Konstantin_mother_returns_amount_l562_562549


namespace martha_apples_l562_562172

theorem martha_apples (martha_initial_apples : ℕ) (jane_apples : ℕ) 
  (james_additional_apples : ℕ) (target_martha_apples : ℕ) :
  martha_initial_apples = 20 →
  jane_apples = 5 →
  james_additional_apples = 2 →
  target_martha_apples = 4 →
  (let james_apples := jane_apples + james_additional_apples in
   let martha_remaining_apples := martha_initial_apples - jane_apples - james_apples in
   martha_remaining_apples - target_martha_apples = 4) :=
begin
  sorry
end

end martha_apples_l562_562172


namespace find_equation_of_line_l_l562_562816

-- Define the conditions
def point_P : ℝ × ℝ := (2, 3)

noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

def intercept_condition (a b : ℝ) : Prop := a + b = 0

-- The proof statement
theorem find_equation_of_line_l :
  ∃ (k : ℝ), k = Real.tan angle_of_inclination ∧
  ∃ (C : ℝ), ∀ (x y : ℝ), (y - 3 = k * (x - 2)) ∧ C = (3 + 2 * (Real.sqrt 3)) ∨ 
             (intercept_condition (x / point_P.1) (y / point_P.2) ∧ C = 1) ∨ 
             -- The standard forms of the line equation
             ((Real.sqrt 3 * x + y - C = 0) ∨ (x - y + 1 = 0)) :=
sorry

end find_equation_of_line_l_l562_562816


namespace q_at_2_equals_9_l562_562741

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

-- Define the function q(x)
noncomputable def q (x : ℝ) : ℝ :=
sgn (3 * x - 1) * |3 * x - 1| ^ (1/2) +
3 * sgn (3 * x - 1) * |3 * x - 1| ^ (1/3) +
|3 * x - 1| ^ (1/4)

-- The theorem stating that q(2) equals 9
theorem q_at_2_equals_9 : q 2 = 9 :=
by sorry

end q_at_2_equals_9_l562_562741


namespace problem1_problem2_problem3_l562_562402

-- 1. Prove that (3ab³)² = 9a²b⁶
theorem problem1 (a b : ℝ) : (3 * a * b^3)^2 = 9 * a^2 * b^6 :=
by sorry

-- 2. Prove that x ⋅ x³ + x² ⋅ x² = 2x⁴
theorem problem2 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 :=
by sorry

-- 3. Prove that (12x⁴ - 6x³) ÷ 3x² = 4x² - 2x
theorem problem3 (x : ℝ) : (12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x :=
by sorry

end problem1_problem2_problem3_l562_562402


namespace framing_required_l562_562677

theorem framing_required
  (initial_width : ℕ)
  (initial_height : ℕ)
  (scale_factor : ℕ)
  (border_width : ℕ)
  (increments : ℕ)
  (initial_width_def : initial_width = 4)
  (initial_height_def : initial_height = 6)
  (scale_factor_def : scale_factor = 4)
  (border_width_def : border_width = 3)
  (increments_def : increments = 12) :
  Nat.ceil ((2 * (4 * scale_factor  + 2 * border_width + 6 * scale_factor + 2 * border_width).toReal) / increments) = 9 := by
  sorry

end framing_required_l562_562677


namespace tangent_line_equation_at_point_l562_562215

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562215


namespace total_protest_days_l562_562139

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l562_562139


namespace recurring_decimal_fraction_l562_562013

theorem recurring_decimal_fraction :
  let a := 0.714714714...
  let b := 2.857857857...
  (a / b) = (119 / 476) :=
by
  let a := (714 / (999 : ℝ))
  let b := (2856 / (999 : ℝ))
  sorry

end recurring_decimal_fraction_l562_562013


namespace regular_octagon_area_l562_562292

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562292


namespace Cara_skated_distance_l562_562636

noncomputable def Cara_and_Danny_distance : ℝ :=
  let d_cd := 120
  let v_cara := 8
  let v_danny := 7
  let theta := 45 * (Real.pi / 180) -- Convert 45 degrees to radians
  let consine_theta := Real.sqrt 2 / 2
  let a := v_cara ^ 2 - Real.pow v_danny 2
  let b := -2 * v_cara * d_cd * consine_theta
  let c := d_cd ^ 2
  let delta := b ^ 2 - 4 * a * c
  let t := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) -- quadratic root applying the (-b - sqrt(...)) / 2a form
  v_cara * t

theorem Cara_skated_distance : Cara_and_Danny_distance = 179.2 := by
  sorry

end Cara_skated_distance_l562_562636


namespace centaur_not_able_to_complete_tour_l562_562206

/-- The centaur moves alternately like a knight and a white pawn (strictly one square upward).
    The chessboard is colored in a standard alternating black and white pattern, starting from a white square.
    We want to prove that the centaur cannot visit all 64 squares, visiting each square exactly once, if its first move is like a pawn.
    The starting square is considered visited.
-/
def centaur_tour_impossible : Prop :=
  let board_size := 8
  let start_position := (0, 0) -- assuming top left corner is (0, 0) in a 0-indexed coordinate
  let pawn_move := (1, 0) -- move strictly one square upward
  let knight_moves := [(2, 1), (1, 2), (-1, 2), (-2, 1), 
                       (-2, -1), (-1, -2), (1, -2), (2, -1)] -- possible knight moves
  let color (r c : ℕ) := (r + c) % 2 == 0

  ∃ P : ℕ × ℕ → Prop, -- P is a proposition indicating visited squares
  P start_position ∧ -- starting square is visited
  ∀ (r c : ℕ), P (r, c) → -- if a square is visited,
    (∃ (move : ℕ × ℕ), -- there exists a move the centaur can make 
      (let (dr, dc) := move in 
       (P (r + dr, c + dc)))) ∧ ¬(∀ i j, (P (i, j))) -- ensures not all squares can be visited

theorem centaur_not_able_to_complete_tour : centaur_tour_impossible :=
by
  sorry

end centaur_not_able_to_complete_tour_l562_562206


namespace fraction_of_shaded_area_l562_562279

theorem fraction_of_shaded_area (total_length total_width : ℕ) (total_area : ℕ)
  (quarter_fraction half_fraction : ℚ)
  (h1 : total_length = 15) 
  (h2 : total_width = 20)
  (h3 : total_area = total_length * total_width)
  (h4 : quarter_fraction = 1 / 4)
  (h5 : half_fraction = 1 / 2) :
  (half_fraction * quarter_fraction * total_area) / total_area = 1 / 8 :=
by
  sorry

end fraction_of_shaded_area_l562_562279


namespace octagon_area_inscribed_in_circle_l562_562311

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562311


namespace solve_swim_problem_l562_562371

/-- A man swims downstream 36 km and upstream some distance taking 3 hours each time. 
The speed of the man in still water is 9 km/h. -/
def swim_problem : Prop :=
  ∃ (v : ℝ) (d : ℝ),
    (9 + v) * 3 = 36 ∧ -- effective downstream speed and distance condition
    (9 - v) * 3 = d ∧ -- effective upstream speed and distance relation
    d = 18            -- required distance upstream is 18 km

theorem solve_swim_problem : swim_problem :=
  sorry

end solve_swim_problem_l562_562371


namespace isosceles_triangle_angle_properties_l562_562883

theorem isosceles_triangle_angle_properties
  (ABC : Triangle)
  (isosceles_ABC : AB = AC)
  (D : Point)
  (E : Point)
  (K : Point)
  (in_bisector_CAB : bisects C A B D)
  (in_bisector_ABC : bisects A B C E)
  (incircle_center_ADC : incenter_of ADC K)
  (angle_BEK_45 : ∠BEK = 45°) :
  ∠CAB = 60° ∨ ∠CAB = 90° :=
  sorry

end isosceles_triangle_angle_properties_l562_562883


namespace tangent_line_equation_l562_562235

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562235


namespace child_with_two_green_neighbors_l562_562967

variables (children : ℕ)
variables (r b : ℕ)
variables (red blue green : children -> Prop) 

-- Conditions
def RedWithGreenNeighbor (red blue green : children -> Prop) : Prop := r = 20
def BlueWithGreenNeighbor (red blue green : children -> Prop) : Prop := b = 25

-- Statement to prove
theorem child_with_two_green_neighbors :
  (RedWithGreenNeighbor red blue green) ∧ (BlueWithGreenNeighbor red blue green) →
  ∃ (child : children), (red child ∨ blue child) ∧ green (child - 1) ∧ green (child + 1) := 
by
  sorry

end child_with_two_green_neighbors_l562_562967


namespace cost_price_per_metre_l562_562378

theorem cost_price_per_metre
  (num_metres : ℕ)
  (selling_price : ℝ)
  (profit_per_metre : ℝ) :
  (num_metres = 30) →
  (selling_price = 4500) →
  (profit_per_metre = 10) →
  (selling_price - profit_per_metre * num_metres) / num_metres = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry -- Proof steps can be filled here

end cost_price_per_metre_l562_562378


namespace determine_c_l562_562203

theorem determine_c {f : ℝ → ℝ} (c : ℝ) (h : ∀ x, f x = 2 / (3 * x + c))
  (hf_inv : ∀ x, (f⁻¹ x) = (3 - 6 * x) / x) : c = 18 :=
by sorry

end determine_c_l562_562203


namespace origin_outside_circle_l562_562078

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle_eq := λ x y : ℝ, x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0 in
  ¬ circle_eq 0 0 :=
by
  let circle_eq := λ x y : ℝ, x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0
  have h₁ : circle_eq 0 0 = (a-1)^2 := by
    simp [circle_eq]
  have h₂ : (a-1)^2 > 0 := by
    nlinarith [h.1, h.2]
  rw h₁
  exact h₂

end origin_outside_circle_l562_562078


namespace f_at_6_l562_562058

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f(x) = y

axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)

axiom f_periodicity : ∀ x : ℝ, f(x + 2) = -f(x)

theorem f_at_6 : f(6) = 0 := sorry

end f_at_6_l562_562058


namespace text_messages_ratio_l562_562890

theorem text_messages_ratio :
  ∀ (T : ℕ),
    (220 + T + 3 * 50 = 5 * 96) →
    T = 110 →
    (T : ℚ) / 220 = 1 / 2 :=
by
  intros T h1 h2
  rw [←Rat.div_self (by norm_cast; linarith [220] : 220 ≠ 0 : ℚ)]
  have h3 : (110 : ℚ) = T :=
    by exact_mod_cast h2
  rw [←h3, Rat.div_eq_div_iff] -- some rewriting with rational equality
  norm_cast
  sorry

end text_messages_ratio_l562_562890


namespace bill_experience_l562_562746

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l562_562746


namespace three_digit_solutions_count_l562_562099

theorem three_digit_solutions_count : 
  ∃ (n : ℕ), n = 31 ∧ 
    ∀ x : ℤ, (4253 * x + 527 ≡ 1388 [MOD 29]) →
             (100 ≤ x) ∧ (x ≤ 999) ↔ 
             ∃ k : ℤ, (x ≡ 9 + 29 * k [MOD 29]) ∧
                      (4 ≤ k) ∧ (k ≤ 34) := 
by sorry

end three_digit_solutions_count_l562_562099


namespace length_of_major_axis_l562_562831

theorem length_of_major_axis (x y : ℝ) (h : (x^2 / 25) + (y^2 / 16) = 1) : 10 = 10 :=
by
  sorry

end length_of_major_axis_l562_562831


namespace TileC_in_Rectangle3_l562_562271

structure Tile :=
(top : ℕ)
(right : ℕ)
(bottom : ℕ)
(left : ℕ)

def Tile_A : Tile := {top := 6, right := 1, bottom := 3, left := 2}
def Tile_B : Tile := {top := 3, right := 6, bottom := 2, left := 0}
def Tile_C : Tile := {top := 4, right := 0, bottom := 5, left := 6}
def Tile_D : Tile := {top := 2, right := 5, bottom := 1, left := 4}

def Rectangle := ℕ -- 1, 2, 3, 4

noncomputable def tile_at_rectangle : Rectangle → Tile
| 1 := sorry
| 2 := sorry
| 3 := Tile_C
| 4 := Tile_B
| _ := sorry

theorem TileC_in_Rectangle3 : tile_at_rectangle 3 = Tile_C :=
by
  -- Proof steps omitted
  sorry

end TileC_in_Rectangle3_l562_562271


namespace polynomial_irreducible_l562_562665

noncomputable def f (a : ℕ → ℤ) (p : ℕ) (m n : ℕ) : Polynomial ℤ := 
  ∑ i in Finset.range (n+1), polynomial.C (a i) * polynomial.X ^ (n - i)

theorem polynomial_irreducible (a : ℕ → ℤ) (p : ℕ) (m n : ℕ)
  (h1 : 0 < a 0) (h2 : ∀ i (hi : i < n), a i < a (i + 1))
  (h3 : a n = p^m) (h4 : Nat.Prime p) (h5 : ¬ (p ∣ a (n - 1))) :
  irreducible (f a p m n) :=
sorry

end polynomial_irreducible_l562_562665


namespace albert_earnings_l562_562730

theorem albert_earnings (E P : ℝ) 
  (h1 : E * 1.20 = 660) 
  (h2 : E * (1 + P) = 693) : 
  P = 0.26 :=
sorry

end albert_earnings_l562_562730


namespace circle_chord_symmetry_l562_562593

open EuclideanGeometry

variables {P : Type} [MetricSpace P]

theorem circle_chord_symmetry 
  (S : set P) [is_circle S]
  (O : P) [center S = O]
  (A B C : P) 
  (hAB : A ∈ S ∧ B ∈ S) 
  (hC : C ∈ line_segment A B)
  (D : P) 
  (hD : D ∈ (circumcircle_triang A C O ∩ S) ∧ D ≠ A) :
  dist C D = dist C B :=
sorry

end circle_chord_symmetry_l562_562593


namespace line_equation_l562_562460

-- Given conditions
variable (P : ℝ × ℝ) (x_intercept y_intercept : ℝ)
hypothesis hp : P = (1, -2)
hypothesis hxy : x_intercept = -y_intercept

-- Prove that the equation of line l
theorem line_equation (l : ℝ → ℝ → Prop) :
  (∀ x y, l x y ↔ 2 * x + y = 0) ∨ (∀ x y, l x y ↔ x - y - 3 = 0) :=
sorry

end line_equation_l562_562460


namespace probability_of_both_even_l562_562670

def totalBalls : ℕ := 17
def evenBalls : ℕ := 8
def firstDrawProb : ℚ := evenBalls / totalBalls
def secondDrawEvenBalls : ℕ := evenBalls - 1
def totalRemainingBalls : ℕ := totalBalls - 1
def secondDrawProb : ℚ := secondDrawEvenBalls / totalRemainingBalls

theorem probability_of_both_even : firstDrawProb * secondDrawProb = 7 / 34 :=
by
  -- proof to be filled
  sorry

end probability_of_both_even_l562_562670


namespace ratio_of_times_l562_562697

-- Given conditions as definitions
def distance : ℕ := 630 -- distance in km
def previous_time : ℕ := 6 -- time in hours
def new_speed : ℕ := 70 -- speed in km/h

-- Calculation of times
def previous_speed : ℕ := distance / previous_time

def new_time : ℕ := distance / new_speed

-- Main theorem statement
theorem ratio_of_times :
  (new_time : ℚ) / (previous_time : ℚ) = 3 / 2 :=
  sorry

end ratio_of_times_l562_562697


namespace find_negative_m_l562_562837

theorem find_negative_m :
  ∃ m : ℝ, (∃ a b : ℝ, a ≠ b ∧ (a^3 - (3/2)*a^2 - m = 0) ∧ (b^3 - (3/2)*b^2 - m = 0)) ∧ 
          (a^3 - (3/2)*a^2 - m = 0 → m = -1/2) :=
begin
  sorry
end

end find_negative_m_l562_562837


namespace find_2023rd_letter_in_sequence_l562_562646

def repeating_sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F', 'E', 'D', 'C', 'B', 'A']

def nth_in_repeating_sequence (n : ℕ) : Char :=
  repeating_sequence.get! (n % 13)

theorem find_2023rd_letter_in_sequence :
  nth_in_repeating_sequence 2023 = 'H' :=
by
  sorry

end find_2023rd_letter_in_sequence_l562_562646


namespace gas_cost_correct_l562_562038

-- Definitions for initial setup
def initial_cost_per_person (x : ℝ) : ℝ := x / 4
def updated_cost_per_person (x : ℝ) : ℝ := x / 7
def decrease_in_cost (x : ℝ) : ℝ := initial_cost_per_person x - updated_cost_per_person x

-- Condition for the problem
def condition_decrease (x : ℝ) : Prop := decrease_in_cost x = 8

-- Theorem to prove
theorem gas_cost_correct : ∃ x : ℝ, condition_decrease x ∧ x = 74.67 := by
  sorry

end gas_cost_correct_l562_562038


namespace neither_plaid_nor_purple_l562_562975

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l562_562975


namespace no_integers_a_b_for_equation_l562_562389

noncomputable theory

open Int

theorem no_integers_a_b_for_equation :
  ¬ ∃ a b : ℤ, a^2 = b^15 + 1004 := 
sorry

end no_integers_a_b_for_equation_l562_562389


namespace quadratic_inequality_solution_set_l562_562858

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) : 
  ∀ x : ℝ, (x^2 - (m + 1/m) * x + 1 < 0) ↔ m < x ∧ x < 1/m :=
by
  sorry

end quadratic_inequality_solution_set_l562_562858


namespace find_quadratic_polynomial_l562_562433

noncomputable def quadratic_polynomial (a b c : ℚ) : ℚ → ℚ :=
  λ x, a * x^2 + b * x + c

theorem find_quadratic_polynomial :
  ∃ a b c : ℚ, quadratic_polynomial a b c (-2) = 7 ∧
               quadratic_polynomial a b c 1 = 2 ∧
               quadratic_polynomial a b c 3 = 10 :=
by
  have h0 : quadratic_polynomial (17/15) (-8/15) (7/5) (-2) = 7 := sorry
  have h1 : quadratic_polynomial (17/15) (-8/15) (7/5) 1 = 2 := sorry
  have h2 : quadratic_polynomial (17/15) (-8/15) (7/5) 3 = 10 := sorry
  use [17/15, -8/15, 7/5]
  exact ⟨h0, h1, h2⟩

end find_quadratic_polynomial_l562_562433


namespace machine_work_rate_l562_562270

theorem machine_work_rate (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -6 ∧ x ≠ -1) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end machine_work_rate_l562_562270


namespace tangent_line_equation_l562_562238

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562238


namespace incorrect_inequality_transformation_l562_562856

theorem incorrect_inequality_transformation 
    (a b : ℝ) 
    (h : a > b) 
    : ¬(1 - a > 1 - b) := 
by {
  sorry 
}

end incorrect_inequality_transformation_l562_562856


namespace maximum_chips_on_chessboard_l562_562647

theorem maximum_chips_on_chessboard (w b : ℕ → ℕ) (h1 : ∀ i, w i = 2 * b i) (i j : ℕ) (hrow : i < 8) (hcol : j < 8) :
  (Σ k, w k + Σ k, b k)  ≤ 48 :=
begin
  sorry,
end

end maximum_chips_on_chessboard_l562_562647


namespace no_real_solution_for_equation_l562_562000

theorem no_real_solution_for_equation
  (x : ℝ) :
  ¬ (sqrt (4 * x + 2) + 1) / sqrt (8 * x + 10) = 2 / sqrt 5 :=
by
  sorry

end no_real_solution_for_equation_l562_562000


namespace tan_of_tan_squared_2025_deg_l562_562409

noncomputable def tan_squared (x : ℝ) : ℝ := (Real.tan x) ^ 2

theorem tan_of_tan_squared_2025_deg : 
  Real.tan (tan_squared (2025 * Real.pi / 180)) = Real.tan (Real.pi / 180) :=
by
  sorry

end tan_of_tan_squared_2025_deg_l562_562409


namespace martha_apples_l562_562173

theorem martha_apples (martha_initial_apples : ℕ) (jane_apples : ℕ) 
  (james_additional_apples : ℕ) (target_martha_apples : ℕ) :
  martha_initial_apples = 20 →
  jane_apples = 5 →
  james_additional_apples = 2 →
  target_martha_apples = 4 →
  (let james_apples := jane_apples + james_additional_apples in
   let martha_remaining_apples := martha_initial_apples - jane_apples - james_apples in
   martha_remaining_apples - target_martha_apples = 4) :=
begin
  sorry
end

end martha_apples_l562_562173


namespace solution_1_solution_2_l562_562495

variable {n : ℕ}

def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * a (n - 1)

def b (n : ℕ) : ℕ :=
  n

def a_n_times_b_n (n : ℕ) : ℕ :=
  n * 2^(n-1)

def S (n : ℕ) : ℕ :=
  ∑ i in finset.range n, (i + 1) * 2^(i)

theorem solution_1 (n : ℕ) (h : n ≥ 1) : a n = 2^(n-1) := by
  sorry

theorem solution_2 (n : ℕ) (h : n ≥ 1) : S n = 1 + (n-1) * 2^n := by
  sorry

end solution_1_solution_2_l562_562495


namespace count_positive_integers_satisfying_inequality_l562_562845

theorem count_positive_integers_satisfying_inequality :
  let count := (2 to 15).count (λ n, (n + 9) * (n - 2) * (n - 15) < 0)
  count = 12 := sorry

end count_positive_integers_satisfying_inequality_l562_562845


namespace even_and_period_pi_l562_562079

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem even_and_period_pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
by
  -- First, prove that f(x) is an even function: ∀ x, f(-x) = f(x)
  -- Next, find the smallest positive period T: ∃ T > 0, ∀ x, f(x + T) = f(x)
  -- Finally, show that this period is pi: T = π
  sorry

end even_and_period_pi_l562_562079


namespace tangent_line_equation_at_point_l562_562210

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562210


namespace locus_of_cone_centers_l562_562699

noncomputable
def geometric_locus (S : Circle) (O : Point) (P : Plane) : Set Point := sorry

theorem locus_of_cone_centers (S : Circle) (O : Point) (h1 : O ≠ S.center) :
  ∃ P : Plane, P.contains S ∧ P.contains O ∧
  (∀ X : Point, (X ∈ geometric_locus S O P) ↔ (X ≠ O ∧ ∃ A : Point, 
    inversion_point S.center S.radius O A ∧ (distance A O)^2 = S.radius^2 ∧ P.contains A 
    ∧ (right_angle X A O) ∧ X ∈ Circle (midpoint O A) (distance O A / 2))) :=
sorry

def inversion_point (C : Point) (r : ℝ) (O : Point) (A : Point) : Prop :=
distance C O * distance C A = r^2

def right_angle (X A O : Point) : Prop :=
angle X A O = 90°

end locus_of_cone_centers_l562_562699


namespace lisa_candy_count_l562_562170

theorem lisa_candy_count :
  (∃ (candy : ℕ), candy = 36) →
  (∀ (d : ℕ), d = 2 ∨ d = 3 → (d = 2 → (candies_eaten d = 2))) →
  (∀ (d : ℕ), d ≠ 2 ∧ d ≠ 3 → candies_eaten d = 1) →
  (total_candies_eaten (4 * 7) = 36) →
  (candies_eaten 2 = 2 ∧ candies_eaten 3 = 2) :=
sorry

end lisa_candy_count_l562_562170


namespace remainder_sand_amount_l562_562628

def total_sand : ℝ := 2548726
def bag_capacity : ℝ := 85741.2
def full_bags : ℝ := 29
def not_full_bag_sand : ℝ := 62231.2

theorem remainder_sand_amount :
  total_sand - (full_bags * bag_capacity) = not_full_bag_sand :=
by
  sorry

end remainder_sand_amount_l562_562628


namespace octagon_area_inscribed_in_circle_l562_562315

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562315


namespace angle_range_l562_562497

noncomputable def vector_magnitudes (a b : ℝ) : Prop := |a| = 2 * |b| ∧ |a| ≠ 0

noncomputable def function_monotonic_increasing (a b : ℝ → ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem angle_range (a b : ℝ) (f : ℝ → ℝ) (theta : ℝ) :
  vector_magnitudes a b →
  function_monotonic_increasing a b (λ x => 2*x^3 + 3*|a|*x^2 + 6*(a • b)*x + 5) →
  0 ≤ theta ∧ theta ≤ π / 3 :=
sorry

end angle_range_l562_562497


namespace knight_moves_equal_n_seven_l562_562881

def knight_moves (n : ℕ) : ℕ := sorry -- Function to calculate the minimum number of moves for a knight.

theorem knight_moves_equal_n_seven :
  ∀ {n : ℕ}, n = 7 →
    knight_moves n = knight_moves n := by
  -- Conditions: Position on standard checkerboard 
  -- and the knight moves described above.
  sorry

end knight_moves_equal_n_seven_l562_562881


namespace length_of_AB_in_right_triangle_l562_562134

def right_triangle (X Y Z : Type) [metric_space X] (A B C : X) := 
  angle A C B = 90

variables (A B C : Type) [metric_space A]

def AC : ℝ := 6
def BC : ℝ := 2
def AB : ℝ := 2 * real.sqrt 10

theorem length_of_AB_in_right_triangle :
  right_triangle A B C A B C → AB = real.sqrt (AC^2 + BC^2) :=
by sorry

end length_of_AB_in_right_triangle_l562_562134


namespace opposite_of_neg_3_is_3_l562_562994

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l562_562994


namespace part1_part2_l562_562360

-- Definitions for prices and cashback percentages
def priceA := 90
def priceB := 100
def cashbackA := 30 / 100
def cashbackB := 15 / 100
def cashbackScheme2 := 20 / 100

-- Proposition part 1: Scheme 2 is more cost-effective for 30 pieces of A and 90 pieces of B
def cost_scheme1 (numA numB : ℕ) := 
  numA * priceA * (1 - cashbackA) + numB * priceB * (1 - cashbackB)

def cost_scheme2 (numA numB : ℕ) := 
  (numA * priceA + numB * priceB) * (1 - cashbackScheme2)

theorem part1 : cost_scheme1 30 90 > cost_scheme2 30 90 := 
by 
  let diff := cost_scheme1 30 90 - cost_scheme2 30 90
  have h : diff = 180 := 
    by 
      rw [cost_scheme1, cost_scheme2]
      norm_num
  exact lt_of_eq_of_lt h (by norm_num)
  done

-- Proposition part 2: Scheme 2 offers greater savings when x ≥ 33
def piecesB (x : ℕ) := 2 * x + 1

theorem part2 (x : ℕ) (hx : x ≥ 33) : 
  cost_scheme1 x (piecesB x) > cost_scheme2 x (piecesB x) := 
by 
  let numA := x
  let numB := piecesB x
  let diff := cost_scheme1 numA numB - cost_scheme2 numA numB
  have h : diff = x + 5 := 
    by 
      rw [cost_scheme1, cost_scheme2]
      have ha : numB = 2 * x + 1 := rfl
      rw [ha]
      norm_num
  exact lt_add_of_pos_right _ (by norm_num)
  done

end part1_part2_l562_562360


namespace dissection_least_rectangles_l562_562148

theorem dissection_least_rectangles (n : ℕ) (points : Fin n → Point)
  (R : Rectangle) (Hno_parallel : ∀ i j, i ≠ j → ¬∃ l, IsLineParallelToSidesOfR l ∧ 
  (points i ∈ l ∧ points j ∈ l)) :
  ∃ rectangles : List Rectangle, 
  length rectangles ≥ n + 1 ∧ 
  (∃ divisions : ∀ r ∈ rectangles, r ∈ DissectedRectangles, 
  ∀ i j, i ≠ j → ∀ r ∈ rectangles, points i ∉ r.interior ∧ points j ∉ r.interior) 
:=
sorry

end dissection_least_rectangles_l562_562148


namespace eq_product_l562_562432

def line_equation_condition (z : ℂ) : Prop := 
  (6 - 5i) * z + (-6 - 5i) * conj(z) = 16

theorem eq_product :
  ∃ (a b : ℂ), (a = 6 - 5i) ∧ (b = -6 - 5i) ∧ a * b = 61 :=
by {
  use [6 - 5i, -6 - 5i],
  split; try {refl},
  split; try {refl},
  sorry
}

end eq_product_l562_562432


namespace sec_7pi_over_4_eq_sqrt_2_l562_562423

theorem sec_7pi_over_4_eq_sqrt_2 : Real.sec (7 * Real.pi / 4) = Real.sqrt 2 := 
by 
  sorry

end sec_7pi_over_4_eq_sqrt_2_l562_562423


namespace find_d_over_a1_l562_562790

noncomputable theory

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem find_d_over_a1
  (a : ℕ → ℝ) (d a_1 : ℝ)
  (ha : a 0 = a_1)
  (h_arith : arithmetic_sequence a d)
  (h_pos : ∀ n, a n > 0)
  (S : ℕ → ℝ)
  (hS : ∀ n, S n = (n : ℝ) / 2 * (2 * a_1 + (n - 1) * d))
  (h_geom : geometric_sequence (a 1) (S 2) (a 1 + S 4)) :
  d / a_1 = 3 / 2 :=
by sorry

end find_d_over_a1_l562_562790


namespace four_real_roots_iff_l562_562084

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)

theorem four_real_roots_iff (ω : ℝ) (hω : ω > 0) : 
  (∃ (roots : fin 4 → ℝ), ∀ i, 0 < roots i ∧ roots i < π ∧ f ω (roots i) = -1) ↔ (7 / 2 < ω ∧ ω ≤ 25 / 6) :=
by
  sorry

end four_real_roots_iff_l562_562084


namespace expand_product_l562_562010

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := 
by
  sorry

end expand_product_l562_562010


namespace perimeter_of_triangle_abe_is_42_l562_562205

theorem perimeter_of_triangle_abe_is_42 {ABCD : Type} (A B C D E : ABCD) 
  (H_area : area_square ABCD = 196) (H_equal_distances : dist D E = dist C E) 
  (H_angle_DEC : angle D E C = 150) : 
  perimeter (triangle A B E) = 42 :=
sorry

end perimeter_of_triangle_abe_is_42_l562_562205


namespace geometric_progression_digits_l562_562246

theorem geometric_progression_digits (a b N : ℕ) (q : ℚ)
  (h1 : b = a * q)
  (h2 : 10 * a + b = 3 * (a * q^2))
  (h3 : 10 ≤ N) (h4 : N ≤ 99) :
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 :=
by
  sorry

end geometric_progression_digits_l562_562246


namespace EF_parallel_BD_G_midpoint_BD_area_triangle_EFG_l562_562200

variables {P : Type*} [EuclideanGeometry P]
variables {A B C D E F G : P}
variables (area : P → P → P → ℝ)
variables (parallels : P → P → P → P → Prop)

-- Assume the given conditions
variables (h1 : convex_quad A B C D)
variables (h2 : ∃ E, collinear A B E ∧ collinear C D E)
variables (h3 : ∃ F, collinear D A F ∧ collinear C B F)
variables (h4 : ∃ G, intersection_point A C B D G)
variables (h5 : area D B F = area D B E)
variables (h_abd : area A B D = 4)
variables (h_cbd : area C B D = 6)

-- Proving parallelism of EF and BD
theorem EF_parallel_BD (h5 : area D B F = area D B E) : parallels E F B D := 
sorry

-- Proving G is the midpoint of BD
theorem G_midpoint_BD (h5 : area D B F = area D B E) (h4 : ∃ G, intersection_point A C B D G) : 
  midpoint G B D := sorry

-- Proving the area of triangle EFG
theorem area_triangle_EFG (h5 : area D B F = area D B E) (h_abd : area A B D = 4) 
  (h_cbd : area C B D = 6) : area E F G = 2 := 
sorry

end EF_parallel_BD_G_midpoint_BD_area_triangle_EFG_l562_562200


namespace smallest_odd_prime_divisor_m_n_sq_l562_562910

theorem smallest_odd_prime_divisor_m_n_sq (m n: ℕ) (hm: m = 3 ∨ m = 5 ∨ m = 7) (hn: n = 3 ∨ n = 5 ∨ n = 7) (h_lt: n < m) : 
  ∃ p: ℕ, prime p ∧ (∀ k, k = m^2 - n^2 → p | k) ∧ p = 3 :=
by 
  sorry

end smallest_odd_prime_divisor_m_n_sq_l562_562910


namespace slope_of_line_l562_562820

theorem slope_of_line (x1 x2 y1 y2 : ℝ) (h1 : 1 = (x1 + x2) / 2) (h2 : 1 = (y1 + y2) / 2) 
                      (h3 : (x1^2 / 36) + (y1^2 / 9) = 1) (h4 : (x2^2 / 36) + (y2^2 / 9) = 1) :
  (y2 - y1) / (x2 - x1) = -1 / 4 :=
by
  sorry

end slope_of_line_l562_562820


namespace area_regular_octagon_in_circle_l562_562282

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562282


namespace min_difference_of_factors_of_1794_l562_562505

theorem min_difference_of_factors_of_1794 : ∃ a b : ℕ, a * b = 1794 ∧ a ≠ 0 ∧ b ≠ 0 ∧ (∀ a' b' : ℕ, a' * b' = 1794 ∧ a' ≠ 0 ∧ b' ≠ 0 → |a - b| ≤ |a' - b'|) ∧ |a - b| = 7 :=
begin
  sorry
end

end min_difference_of_factors_of_1794_l562_562505


namespace max_value_of_x_plus_2y_l562_562922

theorem max_value_of_x_plus_2y {x y : ℝ} (h : |x| + |y| ≤ 1) : (x + 2 * y) ≤ 2 :=
sorry

end max_value_of_x_plus_2y_l562_562922


namespace ratio_bananas_apples_is_3_to_1_l562_562630

def ratio_of_bananas_to_apples (oranges apples bananas peaches total_fruit : ℕ) : ℚ :=
if oranges = 6 ∧ apples = oranges - 2 ∧ peaches = bananas / 2 ∧ total_fruit = 28
   ∧ 6 + apples + bananas + peaches = total_fruit then
    bananas / apples
else 0

theorem ratio_bananas_apples_is_3_to_1 : ratio_of_bananas_to_apples 6 4 12 6 28 = 3 := by
sorry

end ratio_bananas_apples_is_3_to_1_l562_562630


namespace men_required_l562_562510

variable (m w : ℝ) -- Work done by one man and one woman in one day respectively
variable (x : ℝ) -- Number of men

-- Conditions from the problem
def condition1 (m w : ℝ) (x : ℝ) : Prop :=
  x * m = 12 * w

def condition2 (m w : ℝ) : Prop :=
  (6 * m + 11 * w) * 12 = 1

-- Proving that the number of men required to do the work in 20 days is x
theorem men_required (m w : ℝ) (x : ℝ) (h1 : condition1 m w x) (h2 : condition2 m w) : 
  (∃ x, condition1 m w x ∧ condition2 m w) := 
sorry

end men_required_l562_562510


namespace shirts_and_pants_neither_plaid_nor_purple_l562_562977

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l562_562977


namespace common_ratio_of_geometric_sequence_l562_562123

theorem common_ratio_of_geometric_sequence (a_1 a_2 a_3 a_4 q : ℝ)
  (h1 : a_1 * a_2 * a_3 = 27)
  (h2 : a_2 + a_4 = 30)
  (geometric_sequence : a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3) :
  q = 3 ∨ q = -3 :=
sorry

end common_ratio_of_geometric_sequence_l562_562123


namespace max_possible_stacks_l562_562695

-- Defining the conditions
def valid_stack (x y : ℕ) : Prop :=
  2 * x + 5 * y = 1000 / 10 ∧ x ≥ 1 ∧ y ≥ 1

def unique_x (s : Finset (ℕ × ℕ)) : Prop :=
  (∀ a b ∈ s, a.1 ≠ b.1 → a ≠ b)

-- The main theorem stating the conclusion
theorem max_possible_stacks : ∃ s : Finset (ℕ × ℕ), unique_x s ∧ (∀ (x y : ℕ), (x, y) ∈ s ↔ valid_stack x y) ∧ s.card = 9 :=
by {
  -- Proof placeholder
  sorry 
}

end max_possible_stacks_l562_562695


namespace area_is_18_sqrt_21_l562_562534

variables {A B C M N : Type*}
variables [IsoscelesTriangle ABC] (M N : Point ABC)
variables (AM AN CM CN : ℝ)
variables (h_AM : AM = 5) (h_AN : AN = 2 * Real.sqrt 37)
variables (h_CM : CM = 11) (h_CN : CN = 10)

def area_triangle (A B C : Point) : ℝ := sorry

theorem area_is_18_sqrt_21 :
  area_triangle A B C = 18 * Real.sqrt 21 :=
sorry

end area_is_18_sqrt_21_l562_562534


namespace smallest_m_property_l562_562145

def smallest_m (S : Set ℕ) : ℕ := 243

theorem smallest_m_property :
  ∀ (M : ℕ) (H : M ≥ 3) (S : Set ℕ) (A B : Set ℕ),
    S = {n | 3 ≤ n ∧ n ≤ M} →
    A ∩ B = ∅ →
    A ∪ B = S →
    (∃ a b c ∈ A, a * b = c) ∨ (∃ a b c ∈ B, a * b = c) :=
begin
  intros M H S A B HS HInt HUnion,
  sorry,
end

end smallest_m_property_l562_562145


namespace tangent_segment_lengths_sum_equals_side_lengths_sum_l562_562578

variable (P A1 A2 A3 T1 T2 T3 : Point)
variable (PA1 PA2 PA3 : Segment)
variable (Δ : Triangle)
variable [InsideTriangle P Δ] [AcuteTriangle Δ]
variable [TangentSegment A1 T1 (Circle (Diameter (Segment P A2)))]
variable [TangentSegment A2 T2 (Circle (Diameter (Segment P A3)))]
variable [TangentSegment A3 T3 (Circle (Diameter (Segment P A1)))]

theorem tangent_segment_lengths_sum_equals_side_lengths_sum :
  2 * (A1.T1.length^2 + A2.T2.length^2 + A3.T3.length^2) =
  A1.A2.length^2 + A2.A3.length^2 + A3.A1.length^2 :=
sorry

end tangent_segment_lengths_sum_equals_side_lengths_sum_l562_562578


namespace sum_of_real_roots_eq_five_pi_l562_562785

theorem sum_of_real_roots_eq_five_pi :
  ∀ x : ℝ, 0 < x ∧ x < 2 * Real.pi → 3 * (Real.tan x)^2 + 8 * Real.tan x + 3 = 0 → x = 5 * Real.pi :=
begin
  sorry
end

end sum_of_real_roots_eq_five_pi_l562_562785


namespace length_GH_l562_562537

theorem length_GH (AB BC : ℝ) (hAB : AB = 10) (hBC : BC = 5) (DG DH GH : ℝ)
  (hDG : DG = DH) (hArea_DGH : 1 / 2 * DG * DH = 1 / 5 * (AB * BC)) :
  GH = 2 * Real.sqrt 10 :=
by
  sorry

end length_GH_l562_562537


namespace calculate_product_l562_562395

theorem calculate_product : 6^6 * 3^6 = 34012224 := by
  sorry

end calculate_product_l562_562395


namespace norm_of_vector_sum_l562_562098

theorem norm_of_vector_sum (a b : ℝ × ℝ) (θ : ℝ) 
    (ha : a = (3, -4)) 
    (hb : ∥b∥ = 2) 
    (h_angle : θ = 2 * Real.pi / 3) :
    ∥(2 : ℝ) • a + b∥ = 2 * Real.sqrt 21 :=
by
  sorry

end norm_of_vector_sum_l562_562098


namespace find_AE_l562_562413

variables {AB CD AC AE : ℝ}
variables {A B C D E : Type*}
variables [ConvexQuadrilateral A B C D]
variables [DiagonalsIntersect AC BD E]
variables [EqArea AED BEC]

def sides : AB = 8 ∧ CD = 10 ∧ AC = 15 := sorry

theorem find_AE (h : sides) : AE = 20 / 3 := 
by sorry

end find_AE_l562_562413


namespace necessary_but_not_sufficient_condition_l562_562479

variable (a1 : ℝ) (q : ℝ) (S : ℕ → ℝ)

def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a1
  else a1 * (1 - q ^ n) / (1 - q)

theorem necessary_but_not_sufficient_condition (a1_pos : a1 > 0) (q_pos : ℝ) (n : ℕ) (S_pos : S n > 0) :
  (∃ n : ℕ, (S n > 0 → q > 0 ∧ ∀ q ≤ 0, S n > 0)) :=
  sorry

end necessary_but_not_sufficient_condition_l562_562479


namespace mass_percentage_H_in_C4H8O2_l562_562023

theorem mass_percentage_H_in_C4H8O2 (molar_mass_C : Real := 12.01) 
                                    (molar_mass_H : Real := 1.008) 
                                    (molar_mass_O : Real := 16.00) 
                                    (num_C_atoms : Nat := 4)
                                    (num_H_atoms : Nat := 8)
                                    (num_O_atoms : Nat := 2) :
    (num_H_atoms * molar_mass_H) / ((num_C_atoms * molar_mass_C) + (num_H_atoms * molar_mass_H) + (num_O_atoms * molar_mass_O)) * 100 = 9.15 :=
by
  sorry

end mass_percentage_H_in_C4H8O2_l562_562023


namespace third_friday_diff_l562_562871

def dates_in_month (days_since: Nat -> String) : Prop :=
    (days_since 5 = "Tuesday")

def third_friday_date_after_given_day (days_since: Nat -> String) (day: Nat) (friday_count: Nat) : Nat :=
    if friday_count = 3 then day else third_friday_date_after_given_day days_since (day + 1) (if days_since (day + 1) = "Friday" then friday_count + 1 else friday_count)

theorem third_friday_diff
  (days_since: Nat -> String)
  (h: dates_in_month days_since) :
  third_friday_date_after_given_day days_since 1 1 - 18 = 4 :=
sorry

end third_friday_diff_l562_562871


namespace warehouse_inventory_l562_562278

theorem warehouse_inventory (x : ℝ) :
  (1200 * (2 / 3) + 0.1 * x * (8 / 15) = x * (8 / 15) * 0.9) → x = 1875 :=
by
  intro h
  have h1 : x * (8 / 15) * 0.9 = (8 * x) / 15 * 0.9 := by ring
  rw ← h1 at h
  have h2 : 0.1 * x * (8 / 15) = (4 * x) / 75 := by ring
  rw ← h2 at h
  have h3 : 1200 * (2 / 3) = 800 := by norm_num
  rw h3 at h
  sorry -- Proof steps will be written here

end warehouse_inventory_l562_562278


namespace parabola_x_intercepts_l562_562843

theorem parabola_x_intercepts :
  ∃! (x : ℝ), ∃ (y : ℝ), y = 0 ∧ x = -2 * y^2 + y + 1 :=
sorry

end parabola_x_intercepts_l562_562843


namespace cyclic_quadrilateral_angle_equality_l562_562444

-- Define the points and the conditions
def PointsOnCircle (C P Q D : Type) : Prop :=
∃ (circle : Type), (C ∈ circle) ∧ (P ∈ circle) ∧ (Q ∈ circle) ∧ (D ∈ circle)

-- Define the parallelogram condition
def Parallelogram (B C Q P : Type) : Prop :=
∃ (parallelogram : Type), (B ∈ parallelogram) ∧ (C ∈ parallelogram) ∧ (Q ∈ parallelogram) ∧ (P ∈ parallelogram)

-- Define inscribed angle equality in cyclic quadrilateral
def InscribedAngle (A B C D : Type) : Prop :=
∃ (E : Type), (A ∈ E) ∧ (B ∈ E) ∧ (C ∈ E) ∧ (D ∈ E) ∧ (InscribedAngle E A B = InscribedAngle E C D)

-- State the problem
theorem cyclic_quadrilateral_angle_equality {C P Q D B : Type} (h₁ : PointsOnCircle C P Q D) (h₂ : Parallelogram B C Q P) :
  ∃ (ABC : Type), (InscribedAngle C Q D B = InscribedAngle P Q D B) ∧ (InscribedAngle C D Q B = InscribedAngle P D Q B)
:= sorry

end cyclic_quadrilateral_angle_equality_l562_562444


namespace num_n_le_100_g50_eq_18_l562_562035

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def g1 (n : ℕ) : ℕ :=
  3 * num_divisors n

def g (j : ℕ) (n : ℕ) : ℕ :=
  nat.rec_on j g1 (λ j ih, g1(ih n))

theorem num_n_le_100_g50_eq_18 : (finset.range 101).filter (λ n, g 50 n = 18).card = 6 :=
sorry

end num_n_le_100_g50_eq_18_l562_562035


namespace octagon_area_inscribed_in_circle_l562_562309

-- Define the area of the circle as given
def circle_area : ℝ := 256 * Real.pi

-- Define the radius of the circle derived from the area
def radius (A : ℝ) : ℝ := Real.sqrt (A / Real.pi)

-- Define the side length of the inscribed octagon
def octagon_side_length (r : ℝ) : ℝ := r * Real.sqrt (2 - Real.sqrt 2)

-- Define the formula for the area of a regular octagon given its side length
def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

-- Final theorem to prove that the area of the octagon is 512 * sqrt 2
theorem octagon_area_inscribed_in_circle : 
  octagon_area (octagon_side_length (radius circle_area)) = 512 * Real.sqrt 2 :=
sorry

end octagon_area_inscribed_in_circle_l562_562309


namespace score_three_std_devs_above_mean_l562_562032

-- Define what it means to be 2 standard deviations below the mean
def two_std_devs_below_mean (mean std_dev score : ℝ) : Prop :=
  score = mean - 2 * std_dev

-- Define what it means to be 3 standard deviations above the mean
def three_std_devs_above_mean (mean std_dev score : ℝ) : Prop :=
  score = mean + 3 * std_dev

-- Constants based on the conditions
axiom mean_score : ℝ := 76
axiom score_60_below : ℝ := 60

-- Noncomputable def for the standard deviation (standard deviation is derived)
noncomputable def std_dev : ℝ := (mean_score - score_60_below) / 2

-- The statement we want to prove
theorem score_three_std_devs_above_mean : 
  ∃ (score : ℝ), three_std_devs_above_mean mean_score std_dev score ∧ score = 100 :=
by
  -- We'll skip the proof for now
  sorry

end score_three_std_devs_above_mean_l562_562032


namespace proof_problem_l562_562045

/-- Define the constants a, b, and c as per the problem conditions.-/
def a := Real.logBase 0.7 0.3
def b := Real.logBase 0.3 0.7
def c := 0.5

/-- Prove mathematically equivalent proof problem. -/
theorem proof_problem : b < c ∧ c < a :=
by
  sorry -- Proof to be provided

end proof_problem_l562_562045


namespace amount_returned_l562_562554

theorem amount_returned (deposit_usd : ℝ) (exchange_rate : ℝ) (h1 : deposit_usd = 10000) (h2 : exchange_rate = 58.15) : 
  deposit_usd * exchange_rate = 581500 := 
by 
  sorry

end amount_returned_l562_562554


namespace length_of_bridge_l562_562725

/-- 
Given:
- A train is 100 meters long.
- The train's speed is 80 km/h.
- It takes the train 10.889128869690424 seconds to cross a bridge.
Prove:
The length of the bridge is 142 meters.
-/
theorem length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_seconds : ℝ) 
(h1 : length_of_train = 100) 
(h2 : speed_kmh = 80) 
(h3 : time_seconds = 10.889128869690424) :
  let speed_mps := speed_kmh * (1000 / 3600) in
  let total_distance := speed_mps * time_seconds in
  let length_of_bridge := total_distance - length_of_train in
  length_of_bridge = 142 := 
by {
  sorry
}

end length_of_bridge_l562_562725


namespace percentage_local_science_students_l562_562115

variable (local_arts local_commerce total_locals num_science : ℕ) (x : ℕ)

def condition1 : local_arts = 200 := 
by rfl

def condition2 : local_commerce = 102 := 
by rfl

def condition3 : total_locals = 327 := 
by rfl

def condition4 : num_science = 100 := 
by rfl

theorem percentage_local_science_students :
  local_arts + local_commerce + (x * num_science / 100) = total_locals → x = 25 :=
by
  sorry

end percentage_local_science_students_l562_562115


namespace bill_experience_now_l562_562744

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l562_562744


namespace george_exchange_amount_l562_562041

def george_received_bills : ℕ := 10
def spent_percentage : ℝ := 0.20
def exchange_rate : ℝ := 1.5

theorem george_exchange_amount : 
  let total_bills := george_received_bills in
  let spent_bills := total_bills * spent_percentage in
  let remaining_bills := total_bills - spent_bills in
  let total_amount := remaining_bills * exchange_rate in
  total_amount = 12 := by
  sorry

end george_exchange_amount_l562_562041


namespace root_triple_condition_l562_562791

theorem root_triple_condition (a b c α β : ℝ)
  (h_eq : a * α^2 + b * α + c = 0)
  (h_β_eq : β = 3 * α)
  (h_vieta_sum : α + β = -b / a)
  (h_vieta_product : α * β = c / a) :
  3 * b^2 = 16 * a * c :=
by
  sorry

end root_triple_condition_l562_562791


namespace alpha_bound_l562_562803

noncomputable def f : ℕ × ℕ → ℝ → ℝ
| (0, 0) α := 1
| (m, 0) α := 0
| (0, n) α := 0
| (m, n) α := α * f (m, n-1) α + (1 - α) * f (m-1, n-1) α

theorem alpha_bound {α : ℝ} :
  (∀ (m n : ℕ), |f (m, n) α| < 1989) ↔ (0 < α ∧ α < 1) :=
sorry

end alpha_bound_l562_562803


namespace hyperbola_intersection_lines_l562_562709

theorem hyperbola_intersection_lines : 
  let h : ∀ (x y : ℝ), 2 * x^2 - y^2 = 2,
      F : (ℝ × ℝ) := (1, 0) in  
  ∃ᶠ lines passing through F, ∃ A B, h A.1 A.2 ∧ h B.1 B.2 ∧ |A.1 - B.1| = 4,
  number of such lines = 3 :=
by
  sorry

end hyperbola_intersection_lines_l562_562709


namespace original_cube_volume_proof_l562_562703

noncomputable def original_cube_volume (s : ℝ) : ℝ :=
  s^3

theorem original_cube_volume_proof (s : ℝ) (hs : (2 * s)^3 = 1728) : original_cube_volume s = 216 :=
by
  have h1 : 8 * s^3 = 1728 := by rw [pow_mul, pow_two, mul_pow]; exact hs
  have h2 : s^3 = 216 := by linarith
  exact h2

end original_cube_volume_proof_l562_562703


namespace necessary_but_not_sufficient_condition_l562_562923

-- Let \vec{a} and \vec{b} be non-zero vectors in space.
variables (a b : ℝ)

-- Assume that \vec{a} \cdot \vec{b} < 0 (Proposition p)
def p := a * b < 0

-- Assume there exists a negative number \lambda such that \vec{b} = \lambda \vec{a} (Proposition q)
def q := ∃ λ : ℝ, λ < 0 ∧ b = λ * a

-- Proving that p is a necessary but not sufficient condition for q
theorem necessary_but_not_sufficient_condition (a b : ℝ) (p : a * b < 0) :
  (∀ λ : ℝ, λ < 0 → b = λ * a → a * b < 0) ∧
  (¬ (a * b < 0 → ∃ λ : ℝ, λ < 0 ∧ b = λ * a)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l562_562923


namespace minute_hand_gains_per_hour_l562_562700

theorem minute_hand_gains_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_by_6pm : total_gain = 63) (hours_from_9_to_6 : total_hours = 9) : (total_gain / total_hours) = 7 :=
by
  -- The proof is not required as per instruction.
  sorry

end minute_hand_gains_per_hour_l562_562700


namespace percentage_of_total_l562_562350

theorem percentage_of_total (total part : ℕ) (h₁ : total = 100) (h₂ : part = 30):
  (part / total) * 100 = 30 := by
  sorry

end percentage_of_total_l562_562350


namespace tangent_line_equation_l562_562226

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562226


namespace find_a8_l562_562168

noncomputable def arithmetic_sequence : Type := sorry

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (n d : ℝ)
variables (h1 : a 6 = 12) (h2 : S 3 = 12)

theorem find_a8 : 
  let d := 2,
  let a1 := 2 in
  (a 8 = a1 + 7 * d) →
  a 8 = 16 :=
by
  sorry

end find_a8_l562_562168


namespace arithmetic_sequence_sum_l562_562873

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a₁ d, ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  arithmetic_sequence a →
  a 3 + a 4 + a 10 + a 11 = 2002 →
  a 1 + a 5 + a 7 + a 9 + a 13 = 2502.5 :=
by
  intros has_prop condition,
  sorry

end arithmetic_sequence_sum_l562_562873


namespace balls_in_boxes_l562_562101

-- Define the main problem
theorem balls_in_boxes : 
  ∃ ways : ℕ, 
    ways = 495 ∧ 
    (∀ (balls boxes : ℕ), 
      balls = 7 ∧ 
      boxes = 4 →
      ways = 
        1 + 7 + 21 + 21 + 35 + 35 + 35 + 35 + 105 + 35 + 105) :=
begin
  -- Using dummy exist statement to satisfy the need of the example
  use 495,
  split,
  { refl, },
  { intros balls boxes,
    rintro ⟨rfl, rfl⟩,
    sorry,  -- The full proof computation would be here
  }
end

end balls_in_boxes_l562_562101


namespace propB_correct_l562_562047

variable {V : Type} [RealVectorSpace V]

variables (m n : V) (α β : ℝ → Prop)

-- Assumptions
variable (hmn : ∀ t, α t → β t → (m ≠ n))
variable (ha : ∀ t, α t → β t → (β t = α t ∧ α t))
variable (h₁: ∀ t, α t → (m t ∧ n t))
variable (h₂: ∀ v, m = m ∧ n = n)

-- Conditions: The conditions reflecting the problem statement.
def is_parallel_to (l1 l2 : V) := ∀ t, l1 t = l2 t
def is_perpendicular_to (l : V) (p : ℝ → Prop) := ∀ t, p t → l t = ⊥ 

-- The correct answer statement: Proposition B.
theorem propB_correct (h0 : is_perpendicular_to m α) (h1 : is_parallel_to m n) :
  is_perpendicular_to n α :=
sorry

end propB_correct_l562_562047


namespace binom_18_6_eq_4767_l562_562408

theorem binom_18_6_eq_4767 : Nat.binom 18 6 = 4767 :=
by
  sorry

end binom_18_6_eq_4767_l562_562408


namespace average_age_when_youngest_born_l562_562269

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_y : ℕ) (total_yr : ℕ) (reduction_yr yr_older : ℕ) (avg_age_older : ℕ) 
  (h1 : n = 7)
  (h2 : avg_age = 30)
  (h3 : current_y = 7)
  (h4 : total_yr = n * avg_age)
  (h5 : reduction_yr = (n - 1) * current_y)
  (h6 : yr_older = total_yr - reduction_yr)
  (h7 : avg_age_older = yr_older / (n - 1)) :
  avg_age_older = 28 :=
by 
  sorry

end average_age_when_youngest_born_l562_562269


namespace symmetric_line_l562_562986

theorem symmetric_line (x y : ℝ) :
  ∀ (l : ℝ → ℝ), (∀ x, l x = 3 * x + 3) →
  (∃ p:ℝ × ℝ, p = (3, 2)) →
  (∀ x y, (6-x, 4-y) ∈ set_of (λ p:ℝ × ℝ, y = l x)) →
  y = 3 * x - 17 :=
by
  sorry

end symmetric_line_l562_562986


namespace range_of_a_l562_562514

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ x^2 - a * x + a^2 - 3 = 0) ↔ (-real.sqrt 3 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l562_562514


namespace sequence_general_formula_valid_m_l562_562057

noncomputable def a_n : ℕ → ℤ 
| n := if n ≤ 4 then n - 4 else 2^(n - 5)

theorem sequence_general_formula : 
  ∀ n, a_n n = if n ≤ 4 then n - 4 else 2^(n - 5) :=
sorry

theorem valid_m : 
  {m // a_n m + a_n (m+1) + a_n (m+2) = a_n m * a_n (m+1) * a_n (m+2)} = {1, 3} :=
sorry

end sequence_general_formula_valid_m_l562_562057


namespace max_s_squared_l562_562668

open Real

theorem max_s_squared (r : ℝ) (h : 0 ≤ r) :
  ∃ (θ : ℝ), s = r * (sec θ + csc θ) ∧ s^2 ≤ 8 * r^2 :=
by
  sorry

end max_s_squared_l562_562668


namespace modules_count_l562_562337

theorem modules_count (x y: ℤ) (hx: 10 * x + 35 * y = 450) (hy: x + y = 11) : y = 10 :=
by
  sorry

end modules_count_l562_562337


namespace sum_n_k_l562_562608

theorem sum_n_k (n k : ℕ) (h₁ : (x+1)^n = 2 * x^k + 3 * x^(k+1) + 4 * x^(k+2)) (h₂ : 3 * k + 3 = 2 * n - 2 * k)
  (h₃ : 4 * k + 8 = 3 * n - 3 * k - 3) : n + k = 47 := 
sorry

end sum_n_k_l562_562608


namespace find_angles_l562_562138

open EuclideanGeometry

noncomputable def triangle : Type := 
  {A B C : Point}

noncomputable def center_of_excircle (A B C : Point) (J : Point) : Prop := 
  excircle_center A B C J

noncomputable def tangent_point (x y Z: Point) (A1 B1 C1 : Point) (J : Point) : Prop := 
  is_tangent_point Z x J A1 ∧ is_tangent_point B1 y J A ∧ is_tangent_point C1 Z J B

noncomputable def perpendicular_from_point (P Q : Point) (L : Line) : Point :=
  foot_of_perp P L

theorem find_angles (A B C A1 B1 C1 D E J: Point)
  (h1 : center_of_excircle A B C J)
  (h2 : tangent_point A B C A1 B1 C1 J)
  (h3 : is_perpendicular (line_through A1 B1) (line_through A B))
  (h4 : intersect (line_through A1 B1) (line_through A B) = D)
  (h5 : perpendicular_from_point C1 D (line_through DJ) = E) :
  angle B E A1 = 90 ∧ angle A E B1 = 90 := 
sorry

end find_angles_l562_562138


namespace symmetric_point_polar_coord_l562_562127

open Real

theorem symmetric_point_polar_coord :
  ∀ (r θ : ℝ), (r, θ) = (1, π / 3) → (r, -θ) = (1, -π / 3) :=
by
  intro r θ h
  rw [h]
  exact rfl

end symmetric_point_polar_coord_l562_562127


namespace sin_cos_eq_one_l562_562778

theorem sin_cos_eq_one (x : ℝ) (h0 : 0 ≤ x) (h1 : x < 2 * Real.pi) :
  sin x + cos x = 1 → x = 0 ∨ x = Real.pi / 2 :=
by
  sorry

end sin_cos_eq_one_l562_562778


namespace AJ_has_370_l562_562394

variable {K CJ AJ : ℕ}

-- Defining the conditions
def condition1 := CJ = 2 * K + 5
def condition2 := K = AJ / 2
def condition3 := CJ + K + AJ = 930

-- Stated goal
theorem AJ_has_370 (h1 : condition1) (h2 : condition2) (h3 : condition3) : AJ = 370 :=
by
  sorry

end AJ_has_370_l562_562394


namespace limit_of_function_l562_562342

theorem limit_of_function :
  filter.tendsto (λ x: ℝ, (sqrt (1 - 2 * x + 3 * x ^ 2) - (1 + x)) / (x ^ (1 / 3))) (nhds 0) (nhds 0) :=
by
  sorry

end limit_of_function_l562_562342


namespace max_ratio_lemma_l562_562461

theorem max_ratio_lemma (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn : ∀ n, S n = (n + 1) / 2 * a n)
  (hSn_minus_one : ∀ n, S (n - 1) = n / 2 * a (n - 1)) :
  ∀ n > 1, (a n / a (n - 1) ≤ 2) ∧ (a 2 / a 1 = 2) := sorry

end max_ratio_lemma_l562_562461


namespace minimum_framing_feet_needed_l562_562674

theorem minimum_framing_feet_needed :
  let original_width := 4
  let original_height := 6
  let enlarged_width := 4 * original_width
  let enlarged_height := 4 * original_height
  let border := 3
  let total_width := enlarged_width + 2 * border
  let total_height := enlarged_height + 2 * border
  let perimeter := 2 * (total_width + total_height)
  let framing_feet := (perimeter / 12).ceil
  framing_feet = 9 := by
  -- The theorem statement translates given conditions into definitions and finally asserts the result.
  sorry

end minimum_framing_feet_needed_l562_562674


namespace find_common_ratio_l562_562880

noncomputable def common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 = 7 ∧ (a 1 + a (1 * q) + a (1 * q * q) = 21) ∧ 
  (a 1 * q * q = 7 ∧ a 1 * (1 + q + q * q) = 21)

theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 3 = 7) (h₂ : a 1 + a (1 * q) + a (1 * q * q) = 21) :
  q = 1 ∨ q = -1/2 :=
by {
  sorry
}

end find_common_ratio_l562_562880


namespace area_regular_octagon_in_circle_l562_562284

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562284


namespace rice_grains_difference_l562_562370

theorem rice_grains_difference :
  let square := λ n : ℕ, 2^n in
  let grains_on_12th_square := square 12 in
  let sum_of_first_10_squares := (2 * (2^10 - 1)) in
  grains_on_12th_square - sum_of_first_10_squares = 2050 :=
by
  sorry

end rice_grains_difference_l562_562370


namespace fixed_point_passage_shortest_chord_l562_562457

-- Define the circle and line
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def line (k x y : ℝ) : Prop := y = k * x - 3 * k + 1

-- First part: proving the line passes through the fixed point (3,1)
theorem fixed_point_passage (k : ℝ) : line k 3 1 :=
by
  -- The proof goes here
  sorry

-- Second part: finding the value of k and the length of the shortest chord
theorem shortest_chord (k : ℝ) (h : k = -3) : 
  ∃ len : ℝ, len = 2 * real.sqrt 6 :=
by
  -- The proof goes here
  sorry

end fixed_point_passage_shortest_chord_l562_562457


namespace number_of_solutions_cos_sin_l562_562100

theorem number_of_solutions_cos_sin : 
  let k_bound := (Int.floor (150 / Real.pi), Int.ceil (-25 / Real.pi))
  let int_range := Finset.range (k_bound.1 - k_bound.2 + 1)
  int_range.card = 55 := 
by {
  let k_bound := (Int.floor (150 / Real.pi), Int.ceil (-25 / Real.pi))
  let int_range := Finset.range (k_bound.1 - k_bound.2 + 1)
  sorry
}

end number_of_solutions_cos_sin_l562_562100


namespace bryden_receives_amount_correct_l562_562701

theorem bryden_receives_amount_correct :
  ∀ (multiplier : ℝ) (num_quarters : ℕ) (face_value : ℝ),
    multiplier = 25 →
    num_quarters = 7 →
    face_value = 0.25 →
    (multiplier * (num_quarters * face_value) = 43.75) :=
by
  intros multiplier num_quarters face_value hmultip hnquarters hface
  rw [hmultip, hnquarters, hface]
  norm_num
  exact sorry

end bryden_receives_amount_correct_l562_562701


namespace parallel_lines_slope_eq_l562_562477

theorem parallel_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 ↔ 6 * x + m * y + 11 = 0) → m = 8 :=
by
  sorry

end parallel_lines_slope_eq_l562_562477


namespace solve_quadratic_inequality_l562_562821

def quadratic_inequality_solution (a : ℝ) (x : ℝ) : Prop :=
  3 * x^2 + 2 * x + 2 * a < 0

theorem solve_quadratic_inequality (b : ℝ) (solution_set2 : set ℝ) :
  (∀ x, (2 * a * x^2 - 2 * x + 3 < 0) ↔ (2 < x ∧ x < b)) →
  a = 1 / 8 →
  solution_set2 = {x : ℝ | -1/2 < x ∧ x < -1/6} →
  ∀ x, quadratic_inequality_solution a x ↔ x ∈ solution_set2 :=
by
  intros h1 h2 h3 x
  have ha : a = 1 / 8 := h2
  have solution_correct : (quadratic_inequality_solution (1 / 8) x ↔ x ∈ {x : ℝ | -1 / 2 < x ∧ x < -1 / 6}) := sorry
  exact solution_correct

end solve_quadratic_inequality_l562_562821


namespace problem_a_eq_0_or_3_l562_562809

open Set 

theorem problem_a_eq_0_or_3 {a : ℝ} (h : {3, real.sqrt a} ∩ {1, a} = {a}) : a = 0 ∨ a = 3 := 
sorry

end problem_a_eq_0_or_3_l562_562809


namespace candidate_X_wins_by_20_percent_l562_562870

-- Definitions for the conditions
def ratio_republicans_democrats (R D : ℕ) : Prop := 3 * D = 2 * R

def total_voters (R D : ℕ) : ℕ := R + D

def votes_for_X (R D : ℕ) : ℕ := 9 * R / 10 + 15 * D / 100

def votes_for_Y (R D : ℕ) : ℕ := R - votes_for_X R D + D - votes_for_X R D

def percent_win (R D : ℕ) : ℝ := (votes_for_X R D - votes_for_Y R D) / total_voters R D * 100

-- Lean 4 statement to prove candidate X wins by 20%
theorem candidate_X_wins_by_20_percent (R D : ℕ) (h_ratio : ratio_republicans_democrats R D) :
  percent_win R D = 20 := by
sorry

end candidate_X_wins_by_20_percent_l562_562870


namespace minimum_operations_for_n_triangles_l562_562050

theorem minimum_operations_for_n_triangles (n : ℕ) :
  ∃ (N : ℕ), N = (9 * n^2 - 7 * n) / 2 :=
begin
  sorry
end

end minimum_operations_for_n_triangles_l562_562050


namespace circle_radius_l562_562733

/-- Let a circle have a maximum distance of 11 cm and a minimum distance of 5 cm from a point P.
Prove that the radius of the circle can be either 3 cm or 8 cm. -/
theorem circle_radius (max_dist min_dist : ℕ) (h_max : max_dist = 11) (h_min : min_dist = 5) :
  (∃ r : ℕ, r = 3 ∨ r = 8) :=
by
  sorry

end circle_radius_l562_562733


namespace exists_countably_generated_algebra_l562_562577

variables (E : Type*) (𝓔 : set (set E)) (φ : E → ℝ)

-- Condition: (E, 𝓔) is a Borel space with an injective mapping φ
def is_injective (φ : E → ℝ) : Prop :=
  ∀ {x y : E}, φ x = φ y → x = y

def is_borel_space (E : Type*) (𝓔 : set (set E)) (φ : E → ℝ) : Prop :=
  is_injective φ ∧ 
  (∃ B : set ℝ, B ∈ measurable_set ℝ ∧ ∀ (A : set E), A ∈ 𝓔 ↔ B ∩ φ '' A ∈ measurable_set ℝ)

-- The goal:
theorem exists_countably_generated_algebra (h : is_borel_space E 𝓔 φ) : 
  ∃ (𝓐 : set (set E)), countable 𝓐 ∧ algebra.set_algebra 𝓐 ∧ measurable_set 𝓐 := 
sorry

end exists_countably_generated_algebra_l562_562577


namespace find_probability_l562_562051

noncomputable def xi := @MeasureTheory.Measure_theory.Measure.stdNormal 1.5 sigma^2

theorem find_probability (xi : MeasureTheory.Measure ℝ)
  (H1 : xi ≤ᵐ xi (measure_le_measure 2.5) = 0.78) : 
  xi ≤ᵐ xi (measure_le_measure 0.5) = 0.22 :=
sorry

end find_probability_l562_562051


namespace bill_experience_now_l562_562743

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l562_562743


namespace train_stops_time_l562_562660

theorem train_stops_time 
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 20 := 
by
  sorry

end train_stops_time_l562_562660


namespace find_coordinates_of_A_l562_562876

-- Definition of points in Cartesian coordinate system
structure Point where
  x : Int
  y : Int

-- Definitions for translations
def translate_left (A : Point) (distance : Int) : Point :=
  { x := A.x - distance, y := A.y }

def translate_up (A : Point) (distance : Int) : Point :=
  { x := A.x, y := A.y + distance }

-- Given conditions
variables A : Point
variables h1 : ∃ d, translate_left A d = { x := 1, y := 2 }
variables h2 : ∃ d, translate_up A d = { x := 3, y := 4 }

-- Proof statement
theorem find_coordinates_of_A : A = { x := 3, y := 2 } :=
sorry

end find_coordinates_of_A_l562_562876


namespace marikas_father_age_twice_in_2036_l562_562952

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l562_562952


namespace max_value_of_expression_l562_562907

theorem max_value_of_expression 
  (a b c : ℝ)
  (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  6 * a + 3 * b + 10 * c ≤ 3.2 :=
sorry

end max_value_of_expression_l562_562907


namespace train_crossing_time_l562_562847

-- Define the given conditions
def length_of_train : ℝ := 220 -- meters
def speed_of_man : ℝ := 8 * 1000 / 3600 -- converting 8 km/hr to m/s
def speed_of_train : ℝ := 80 * 1000 / 3600 -- converting 80 km/hr to m/s

-- Define the relative speed of the train with respect to the man
def relative_speed : ℝ := speed_of_train - speed_of_man

-- The theorem stating the problem and the expected answer
theorem train_crossing_time : 
  (length_of_train / relative_speed) = 11 := 
by
  sorry

end train_crossing_time_l562_562847


namespace marika_father_age_twice_l562_562940

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l562_562940


namespace area_of_triangle_F1PF2_l562_562489

def area_of_triangle (a b c : ℝ) : ℝ :=
  (1 / 2) * (a + b + c) -- Placeholder definition

theorem area_of_triangle_F1PF2 :
  ∃ (P : ℝ × ℝ),
    let a := 3 in
    let b := 4 in
    let c := 5 in
    let F1 := (-c, 0 : ℝ) in
    let F2 := (c, 0 : ℝ) in
    ∃ (m n : ℝ),
      (m ∈ ℝ) ∧ (n ∈ ℝ) ∧ 
      ( ∃ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1 ∧ P = (x, y) )
      ∧ (abs (m - n) = 6)
      ∧ (m^2 + n^2 = 100)
      ∧ (m * n = 32)
      ∧ (area_of_triangle m n 0 = 16) :=
sorry

end area_of_triangle_F1PF2_l562_562489


namespace shaded_area_FIGH_l562_562118

variables {E F G H I : Type} 
variables (Area_EFGH : ℝ) (EI IH EH : ℝ) (shaded_area : ℝ)

-- Given conditions
-- 1. Point I divides side EH such that EI = 3 * IH 
def condition1 : EI = 3 * IH := sorry

-- 2. The area of the parallelogram EFGH is 120 square units
def condition2 : Area_EFGH = 120 := sorry

-- 3. EH = 15 units
def condition3 : EH = 15 := sorry

-- Prove that the area of the shaded region FIGH is 84 square units
theorem shaded_area_FIGH : shaded_area = 84 :=
by
  -- Include the given conditions
  apply condition1,
  apply condition2,
  apply condition3,
  sorry

end shaded_area_FIGH_l562_562118


namespace function_is_odd_l562_562459

theorem function_is_odd (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f(x + y) = f(x) + f(y)) (h_nonzero : ¬ (∀ x, f(x) = 0)) : ∀ x : ℝ, f(-x) = -f(x) :=
by
  sorry

end function_is_odd_l562_562459


namespace find_point_A_l562_562878

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end find_point_A_l562_562878


namespace num_representable_integers_l562_562431

theorem num_representable_integers : 
  ∃ (n : ℕ), n ≤ 2000 ∧ (∃ (x : ℝ), ⌊x⌋ + ⌊4 * x⌋ + ⌊5 * x⌋ = n) ∧ 
  (797 = Finset.card (Finset.filter (λ n, ∃ (x : ℝ), ⌊x⌋ + ⌊4 * x⌋ + ⌊5 * x⌋ = n) 
    (Finset.range 2001))) := sorry

end num_representable_integers_l562_562431


namespace nested_sqrt_equals_2_l562_562979

-- Define the condition of an infinite nested radical expression
def nested_sqrt (x : ℝ) : ℝ := sqrt (2 + x)

-- Define the equation to solve based on the problem description
def equation (m : ℝ) : Prop := 2 + m = m * m

-- The statement we want to prove
theorem nested_sqrt_equals_2 : {m : ℝ // equation m} := 
    begin
        use 2, -- construct exists instance with m = 2
        show equation 2,
        calc
        2 + 2 = 4 : by rfl
        ... = 2 * 2 : by rfl
    end

end nested_sqrt_equals_2_l562_562979


namespace fixed_point_of_transformed_logarithmic_function_l562_562937

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_a (a : ℝ) (x : ℝ) : ℝ := 1 + log_a a (x - 1)

theorem fixed_point_of_transformed_logarithmic_function
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1) : f_a a 2 = 1 :=
by
  -- Prove the theorem using given conditions
  sorry

end fixed_point_of_transformed_logarithmic_function_l562_562937


namespace four_digit_numbers_count_l562_562449

theorem four_digit_numbers_count : 
  let digits := {0, 1, 2, 3, 4, 5} in
  let odd_numbers := {1, 3, 5} in
  let even_numbers := {0, 2, 4} in
  (∃ (a b : 1 \le odd_numbers.length := 2) (c d : 1 \le even_numbers.length := 2), 
    (a ≠ b) → (c ≠ d)) ∧ (∀ x ∈ {a, b, c, d}, 
    (x ≠ 0 ∨ x ≠ 1 \lor x ≠ 2 \lor x ≠ 3) → 
    (exists_perm : {x} = {0, 1, 2, 3}) ∧ (¬(x = 0) → x ∈ digits))
    → 72 + 108 = 180 :=
by
  sorry

end four_digit_numbers_count_l562_562449


namespace f_C_is_even_l562_562385

def f_A (x : ℝ) : ℝ := x + 1
def f_B (x : ℝ) : ℝ := 1 / x
def f_C (x : ℝ) : ℝ := x^4
def f_D (x : ℝ) : ℝ := x^5

-- Prove that f_C(x) is an even function, i.e., ∀ x, f_C(x) = f_C(-x)
theorem f_C_is_even : ∀ x : ℝ, f_C x = f_C (-x) :=
by
  intro x
  calc
    f_C x = x^4 : rfl
    ... = (-x)^4 : by rw [←neg_mul_neg x, pow_mul]
    ... = f_C (-x) : rfl

end f_C_is_even_l562_562385


namespace marika_father_age_twice_l562_562938

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l562_562938


namespace min_value_f_l562_562024

noncomputable def f : ℝ → ℝ := λ x, (x^2 - x + 3) / (x - 1)

theorem min_value_f (x : ℝ) (hx : 3 ≤ x) : 
 ∃ (min_val : ℝ), 
  (∀y ∈ set.Ici 3, f y ≥ min_val) ∧ min_val = 9 / 2 :=
begin
  use 9 / 2,
  split,
  { sorry },
  { refl }
end

end min_value_f_l562_562024


namespace number_of_pairs_l562_562844

theorem number_of_pairs :
  let valid_pair (a b : ℕ) := (0 < a ∧ 0 < b ∧
    a + b ≤ 100 ∧
    (a + (1 / b : ℚ)) / ((1 / a : ℚ) + b) = 13)
  in (Finset.filter (λ ⟨a, b⟩, valid_pair a b) ((Finset.range 101).product (Finset.range 101))).card = 7 :=
by {
  -- Define all conditions and the result expectation here
  sorry
}

end number_of_pairs_l562_562844


namespace intersection_distance_l562_562254

-- The two parabolas' equations
def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x ^ 2 - 3 * x + 5

-- Points of intersection (p, q) and (r, s) with conditions r >= p
def p : ℝ := 0
def r : ℝ := 3 / 5

-- The proof statement
theorem intersection_distance : r - p = 3 / 5 := by
  sorry

end intersection_distance_l562_562254


namespace quadratic_equation_correct_l562_562027

noncomputable def quadratic_equation : ℝ → ℝ → Prop := 
  λ x y, x^2 - (x * 12) + 11 = 0

theorem quadratic_equation_correct {x y : ℝ}
  (h₁ : x + y = 12)
  (h₂ : |x - y| = 10) :
  quadratic_equation x y :=
by
  sorry

end quadratic_equation_correct_l562_562027


namespace problem1_part1_problem1_part2_l562_562833

theorem problem1_part1 (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 ≤ φ ∧ φ ≤ π)
    (h3 : ∀ x : ℝ, sin (ω * (-x) + φ) = sin (ω * x + φ)) 
    (h4 : ∃ x y : ℝ, local_maximum (sin (ω * x + φ)) ∧ local_maximum (sin (ω * y + φ)) 
    ∧ abs (y - x) = sqrt (4 + π^2)) :
    f(x) = cos(x) :=
sorry

theorem problem1_part2 (α : ℝ) (h5 : sin α + cos α = 2 / 3) :
    (√2 * sin (2 * α - π / 4) + 1) / (1 + tan α) = -5/9 :=
sorry

end problem1_part1_problem1_part2_l562_562833


namespace guess_all_points_l562_562957

theorem guess_all_points (n : ℕ) (rect_sheet : Type) (circle : rect_sheet) (points : set rect_sheet)
  (h_circle : ∀ p ∈ points, p ∈ circle) : (∃ k < (n+1)^2, ∀ p ∈ points, Kolya_guessed(p)) :=
by
  sorry

end guess_all_points_l562_562957


namespace product_of_successive_numbers_l562_562256

theorem product_of_successive_numbers (n : ℝ) (h : n = 64.4980619863884) :
  n * (n + 1) ≈ 4225 :=
by
  sorry

end product_of_successive_numbers_l562_562256


namespace coprime_odd_sum_of_floors_l562_562576

theorem coprime_odd_sum_of_floors (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h_coprime : Nat.gcd p q = 1) : 
  (List.sum (List.map (λ i => Nat.floor ((i • q : ℚ) / p)) ((List.range (p / 2 + 1)).tail)) +
   List.sum (List.map (λ i => Nat.floor ((i • p : ℚ) / q)) ((List.range (q / 2 + 1)).tail))) =
  (p - 1) * (q - 1) / 4 :=
by
  sorry

end coprime_odd_sum_of_floors_l562_562576


namespace tan_alpha_value_sin_cos_2alpha_value_l562_562794

noncomputable def theta_condition (alpha : ℝ) : Prop :=
  (cos (alpha - (π / 2)))^2 / (sin ((5 * π / 2) + alpha) * sin (π + alpha)) = 1 / 2

theorem tan_alpha_value (alpha : ℝ) (h : theta_condition alpha) : 
  tan alpha = -1 / 2 := 
  sorry

theorem sin_cos_2alpha_value (alpha : ℝ) (h : theta_condition alpha) :
  sin (2 * alpha) + cos (2 * alpha) = -1 / 5 := 
  sorry

end tan_alpha_value_sin_cos_2alpha_value_l562_562794


namespace log_a_greater_log_b_c_l562_562043

variable {a b c : ℝ}

theorem log_a_greater_log_b_c (h : 0 < a ∧ a < b ∧ b < 1 ∧ 1 < c) : 
  log a c > log b c := 
sorry

end log_a_greater_log_b_c_l562_562043


namespace region_difference_l562_562618

theorem region_difference (Y : ℕ) (total : ℕ) (difference : ℕ) 
  (h1 : Y = 200) 
  (h2 : total = 350) 
  (h3 : difference = Y - (total - Y)) : 
  difference = 50 := 
begin
  sorry
end

end region_difference_l562_562618


namespace clock_triangle_area_l562_562365

open Real

/-- Define the angular velocities of hour and minute hands -/
def ω_OA : ℝ := -π / 6
def ω_OB : ℝ := -2 * π

/-- Define the lengths of hour and minute hands -/
def a : ℝ := 3
def b : ℝ := 4

/-- Define the angle between the two hands -/
def angle_AOB (t : ℝ) : ℝ := (ω_OA - ω_OB) * t

/-- Define the expression for the area of triangle OAB -/
def S (t : ℝ) : ℝ := 6 * |sin (angle_AOB t)|

/-- Within a 24-hour period, the number of times the area of triangle OAB reaches its maximum value -/
def max_occurrences : ℝ := 24 / (6 / 11)

/-- The main proof problem statement -/
theorem clock_triangle_area : 
    (∀ t : ℝ, 0 ≤ t ∧ t ≤ 24 → S t = 6 * |sin ((11 * π / 6) * t)) ∧ max_occurrences = 44 :=
by
  sorry

end clock_triangle_area_l562_562365


namespace tan_half_sum_l562_562158

theorem tan_half_sum (a b : ℝ) (h1 : cos a + cos b = 3/5) (h2 : sin a + sin b = 5/13) :
  tan ((a + b) / 2) = 25 / 39 :=
by
  sorry

end tan_half_sum_l562_562158


namespace meaningful_expr_x_value_l562_562516

theorem meaningful_expr_x_value (x : ℝ) :
  (10 - x >= 0) ∧ (x ≠ 4) ↔ (x = 8) :=
begin
  -- proof omitted
  sorry
end

end meaningful_expr_x_value_l562_562516


namespace final_number_on_board_l562_562640

theorem final_number_on_board :
  let S := {a | ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 2011 ∧ a = 1 / n},
      invariant_product := ∏ a in S, ((1 / a) - 1)
    in
    (∀ x y ∈ S, let z := (x * y) / (x * y + (1 - x) * (1 - y))
                in S = (S \ {x, y}) ∪ {z}) → 
    invariant_product = 2010! →
    ∃ v, S = {v} ∧ v = 1 / (2010! + 1) :=
begin
  sorry
end

end final_number_on_board_l562_562640


namespace smallest_base_power_l562_562572

theorem smallest_base_power (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h_log_eq : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ Real.log y / Real.log 3 = Real.log z / Real.log 5) :
  z ^ (1 / 5) < x ^ (1 / 2) ∧ z ^ (1 / 5) < y ^ (1 / 3) :=
by
  -- required proof here
  sorry

end smallest_base_power_l562_562572


namespace square_side_length_l562_562541

-- Define the given dimensions and total length
def rectangle_width : ℕ := 2
def total_length : ℕ := 7

-- Define the unknown side length of the square
variable (Y : ℕ)

-- State the problem and provide the conclusion
theorem square_side_length : Y + rectangle_width = total_length -> Y = 5 :=
by 
  sorry

end square_side_length_l562_562541


namespace set_inter_complement_l562_562094

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {0, 2, 3}
def compl_U := {1}
def result : Set Nat := {1}

theorem set_inter_complement :
  (M ∩ compl_U) = result :=
by
  unfold compl_U
  simp only [U, M, N, set_compl_def, set_inter_def]
  sorry

end set_inter_complement_l562_562094


namespace framing_required_l562_562680

theorem framing_required
  (initial_width : ℕ)
  (initial_height : ℕ)
  (scale_factor : ℕ)
  (border_width : ℕ)
  (increments : ℕ)
  (initial_width_def : initial_width = 4)
  (initial_height_def : initial_height = 6)
  (scale_factor_def : scale_factor = 4)
  (border_width_def : border_width = 3)
  (increments_def : increments = 12) :
  Nat.ceil ((2 * (4 * scale_factor  + 2 * border_width + 6 * scale_factor + 2 * border_width).toReal) / increments) = 9 := by
  sorry

end framing_required_l562_562680


namespace find_parabola_coeffs_l562_562602

def parabola_vertex_form (a b c : ℝ) : Prop :=
  ∃ k:ℝ, k = c - b^2 / (4*a) ∧ k = 3

def parabola_through_point (a b c : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, x = 0 ∧ y = 1 ∧  y = a * x^2 + b * x + c

theorem find_parabola_coeffs :
  ∃ a b c : ℝ, parabola_vertex_form a b c ∧ parabola_through_point a b c ∧
  a = -1/2 ∧ b = 2 ∧ c = 1 :=
by
  sorry

end find_parabola_coeffs_l562_562602


namespace constant_term_expansion_is_minus_20_l562_562207

noncomputable def constant_term_in_expansion : ℤ :=
  let x := (arbitrary ℝ) in 
  let expr := (|x| + 1 / |x| - 2) in
  expanding_term expr

theorem constant_term_expansion_is_minus_20 : constant_term_in_expansion = -20 :=
  sorry

end constant_term_expansion_is_minus_20_l562_562207


namespace proof_statements_l562_562469

open Point
open Real

variable (a : ℝ) (h_a : a ≠ 0) (p : ℝ) (h_p : p > 0)
variable (O : Point := (0, 0))
variable (A : Point := (2 * a, 0))
variable (B : Point := (2 * a, 2 * a^2))
variable (C : Point → Prop := λ (P : Point), P.1^2 = 2 * p * P.2)
variable (M : Point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
variable (ON : Line := line_through O B)
variable (N : Point := intersection ON C)

theorem proof_statements :
  (¬ directrix C = λ (y : ℝ), y = -1 / 2) ∧
  (B = midpoint O N) ∧
  (tangent_line C A N tangent C) ∧
  (parallel (tangent_line C M) ON) :=
sorry

end proof_statements_l562_562469


namespace bill_experience_l562_562749

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l562_562749


namespace elaine_spent_on_rent_last_year_l562_562897

variable (E : ℝ) (P : ℝ)

-- Conditions
def last_year_rent (E P : ℝ) : ℝ := (P / 100) * E
def this_year_earnings (E : ℝ) : ℝ := 1.20 * E
def this_year_rent (E : ℝ) : ℝ := 0.30 * 1.20 * E

-- Given that the amount spent on rent this year is 180% of the amount spent last year:
theorem elaine_spent_on_rent_last_year :
  0.36 * E = 1.80 * (P / 100) * E → P = 20 :=
by
  intro h
  sorry

end elaine_spent_on_rent_last_year_l562_562897


namespace marikas_father_age_twice_in_2036_l562_562950

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l562_562950


namespace fraction_simplification_l562_562014

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l562_562014


namespace compute_fraction_l562_562567

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem compute_fraction : (f (g (f 1))) / (g (f (g 1))) = 6801 / 281 := 
by 
  sorry

end compute_fraction_l562_562567


namespace ellipse_property_l562_562913

open Real

-- Definitions based on the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 9) = 1

def foci_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

-- The mathematically equivalent proof problem statement
theorem ellipse_property (P : ℝ × ℝ) (hP : ellipse P.1 P.2) : 
  ∃ F1 F2 : ℝ × ℝ, let |PF1| := dist P F1 in let |PF2| := dist P F2 in 
  F1 = (- Real.sqrt 7, 0) ∧ F2 = (Real.sqrt 7, 0) ∧ 
  let |OP| := dist P (0, 0) in 
    |PF1| * |PF2| + |OP|^2 = 25 :=
by
  sorry

end ellipse_property_l562_562913


namespace find_matrix_N_l562_562429

theorem find_matrix_N (N : Matrix (Fin 4) (Fin 4) ℤ)
  (hi : N.mulVec ![1, 0, 0, 0] = ![3, 4, -9, 1])
  (hj : N.mulVec ![0, 1, 0, 0] = ![-1, 6, -3, 2])
  (hk : N.mulVec ![0, 0, 1, 0] = ![8, -2, 5, 0])
  (hl : N.mulVec ![0, 0, 0, 1] = ![1, 0, 7, -1]) :
  N = ![![3, -1, 8, 1],
         ![4, 6, -2, 0],
         ![-9, -3, 5, 7],
         ![1, 2, 0, -1]] := by
  sorry

end find_matrix_N_l562_562429


namespace unit_vector_in_same_direction_as_a_sub_b_l562_562841

def a : ℝ × ℝ := (2, -9)
def b : ℝ × ℝ := (-3, 3)
def desired_unit_vector : ℝ × ℝ := (5 / 13, -12 / 13)

theorem unit_vector_in_same_direction_as_a_sub_b :
  let v := (a.1 - b.1, a.2 - b.2) in 
  let norm_v := real.sqrt (v.1^2 + v.2^2) in
  (v.1 / norm_v, v.2 / norm_v) = desired_unit_vector := 
by
  sorry

end unit_vector_in_same_direction_as_a_sub_b_l562_562841


namespace floor_inequality_floor_difference_l562_562347

theorem floor_inequality (x : ℝ) : x - 1 < ⌊x⌋ ∧ ⌊x⌋ ≤ x := by
  sorry

theorem floor_difference (x : ℝ) : ⌊2 * x⌋ - 2 * ⌊x⌋ ∈ {0, 1} := by
  sorry

end floor_inequality_floor_difference_l562_562347


namespace average_cookies_per_package_l562_562533

def cookies_per_package : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]

theorem average_cookies_per_package :
  (cookies_per_package.sum : ℚ) / cookies_per_package.length = 13.5 := by
  sorry

end average_cookies_per_package_l562_562533


namespace sum_sequence_l562_562509

/--
  Consider a sequence \( (x_k) \) defined by the recurrence relation:
  \( x_{k+1} = x_k + \frac{1}{2} \) where \( k = 1, 2, \ldots, n-1 \) and the initial condition is \( x_1 = 1 \).
  Prove that the sum of the first n terms of this sequence is \( \frac{n^2 + 3n}{4} \).
-/
theorem sum_sequence (n : ℕ) (hpos : 0 < n) : 
  let x : ℕ → ℚ := λ k, if k = 1 then 1 else x (k - 1) + 1/2 in
  (Finset.sum (Finset.range n) (λ k, x (k + 1))) = (n^2 + 3 * n) / 4 :=
by
  sorry

end sum_sequence_l562_562509


namespace number_of_towers_l562_562362

theorem number_of_towers (
  red_cubes : ℕ
  blue_cubes : ℕ
  green_cubes : ℕ
  total_cubes : ℕ
  tower_height : ℕ
) : 
  red_cubes = 3 → 
  blue_cubes = 2 → 
  green_cubes = 5 → 
  total_cubes = 10 → 
  tower_height = 7 → 
  (nat.choose total_cubes tower_height * nat.perms tower_height) / (nat.factorial red_cubes * nat.factorial blue_cubes * nat.factorial (tower_height - (red_cubes + blue_cubes))) = 210 :=
by
  intros,
  sorry

end number_of_towers_l562_562362


namespace problem_1_problem_2_l562_562346

-- Problem 1: Solve the system of inequalities
theorem problem_1 (x : ℝ) : (10 - 3 * x < -5) ∧ ((x / 3) ≥ 4 - ((x - 2) / 2)) ↔ x ≥ 6 :=
begin
  sorry
end

-- Problem 2: Simplify the expression
theorem problem_2 (a : ℝ) (ha₁ : a ≠ -1) (ha₂ : a ≠ 0) (ha₃ : a ≠ 1) : 
  (2 / (a + 1) - (a - 2) / (a^2 - 1) / (a * (a - 2) / (a^2 - 2 * a + 1))) = 1 / a :=
begin
  sorry
end

end problem_1_problem_2_l562_562346


namespace symbols_invariance_l562_562955

def final_symbol_invariant (symbols : List Char) : Prop :=
  ∀ (erase : List Char → List Char), 
  (∀ (l : List Char), 
    (erase l = List.cons '+' (List.tail (List.tail l)) ∨ 
    erase l = List.cons '-' (List.tail (List.tail l))) → 
    erase (erase l) = List.cons '+' (List.tail (List.tail (erase l))) ∨ 
    erase (erase l) = List.cons '-' (List.tail (List.tail (erase l)))) →
  (symbols = []) ∨ (symbols = ['+']) ∨ (symbols = ['-'])

theorem symbols_invariance (symbols : List Char) (h : final_symbol_invariant symbols) : 
  ∃ (s : Char), s = '+' ∨ s = '-' :=
  sorry

end symbols_invariance_l562_562955


namespace chessboard_coloring_l562_562622

theorem chessboard_coloring (color : Fin 4 → Fin 7 → Bool) :
  ∃ (r1 r2 c1 c2 : ℕ), r1 < r2 ∧ c1 < c2 ∧
  color ⟨r1, by simp⟩ ⟨c1, by simp⟩ = color ⟨r1, by simp⟩ ⟨c2, by simp⟩ ∧
  color ⟨r1, by simp⟩ ⟨c2, by simp⟩ = color ⟨r2, by simp⟩ ⟨c2, by simp⟩ ∧
  color ⟨r2, by simp⟩ ⟨c2, by simp⟩ = color ⟨r2, by simp⟩ ⟨c1, by simp⟩ ∧
  color ⟨r2, by simp⟩ ⟨c1, by simp⟩ = color ⟨r1, by simp⟩ ⟨c1, by simp⟩ :=
by
  sorry

end chessboard_coloring_l562_562622


namespace find_multiple_of_q_l562_562522

theorem find_multiple_of_q
  (q : ℕ)
  (x : ℕ := 55 + 2 * q)
  (y : ℕ)
  (m : ℕ)
  (h1 : y = m * q + 41)
  (h2 : x = y)
  (h3 : q = 7) : m = 4 :=
by
  sorry

end find_multiple_of_q_l562_562522


namespace exp_arbitrarily_large_l562_562162

theorem exp_arbitrarily_large (a : ℝ) (h : a > 1) : ∀ y > 0, ∃ x > 0, a^x > y := by
  sorry

end exp_arbitrarily_large_l562_562162


namespace bill_experience_l562_562747

theorem bill_experience (j b : ℕ) 
  (h₁ : j - 5 = 3 * (b - 5)) 
  (h₂ : j = 2 * b) : b = 10 :=
sorry

end bill_experience_l562_562747


namespace book_cost_l562_562978

theorem book_cost (x : ℕ) (hx1 : 10 * x ≤ 1100) (hx2 : 11 * x > 1200) : x = 110 := 
by
  sorry

end book_cost_l562_562978


namespace marians_new_balance_l562_562927

theorem marians_new_balance :
  ∀ (initial_balance grocery_cost return_amount : ℝ),
    initial_balance = 126 →
    grocery_cost = 60 →
    return_amount = 45 →
    let gas_cost := grocery_cost / 2 in
    let new_balance_before_returns := initial_balance + grocery_cost + gas_cost in
    new_balance_before_returns - return_amount = 171 :=
begin
  intros initial_balance grocery_cost return_amount h_init h_grocery h_return,
  let gas_cost := grocery_cost / 2,
  let new_balance_before_returns := initial_balance + grocery_cost + gas_cost,
  have h_gas : gas_cost = 30,
  { 
    rw h_grocery, 
    norm_num },
  have h_new_balance : new_balance_before_returns = 216,
  {
    rw [h_init, h_grocery, h_gas],
    norm_num },
  rw [h_new_balance, h_return],
  norm_num,
end

end marians_new_balance_l562_562927


namespace most_marbles_l562_562629

def total_marbles := 24
def red_marble_fraction := 1 / 4
def red_marbles := red_marble_fraction * total_marbles
def blue_marbles := red_marbles + 6
def yellow_marbles := total_marbles - red_marbles - blue_marbles

theorem most_marbles : blue_marbles > red_marbles ∧ blue_marbles > yellow_marbles :=
by
  sorry

end most_marbles_l562_562629


namespace complement_union_equals_l562_562921

variable I : Set ι
variable A : Set ι
variable B : Set ι

theorem complement_union_equals {I : Set ℕ} {A B : Set ℕ} 
  (hI : I = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) 
  (hB : B = {2, 3, 4}) : 
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by
  sorry

end complement_union_equals_l562_562921


namespace find_sale_in_second_month_l562_562368

variable (sale_month2 sale_month1 sale_month3 sale_month4 sale_month5 sale_month6 total_sales : ℝ)

def average_sale (months : ℝ) (total_sales : ℝ) : ℝ := total_sales / months

theorem find_sale_in_second_month
  (h1 : sale_month1 = 3435)
  (h3 : sale_month3 = 3855)
  (h4 : sale_month4 = 4230)
  (h5 : sale_month5 = 3560)
  (h6 : sale_month6 = 2000)
  (avg : average_sale 6 total_sales = 3500) :
  sale_month2 = 3920 := by
  sorry

end find_sale_in_second_month_l562_562368


namespace angle_A_thirty_deg_l562_562507

theorem angle_A_thirty_deg 
  (G : Point) (A B C : Point)
  (a b c : ℝ)
  (h1 : is_centroid G A B C)
  (h2 : a • vector_from G A + b • vector_from G B + (sqrt 3 / 3) * c • vector_from G C = vector_zero) :
  ∡ A = 30 :=
sorry

end angle_A_thirty_deg_l562_562507


namespace train_crosses_pole_in_40_seconds_l562_562726

theorem train_crosses_pole_in_40_seconds
  (length : ℕ) (speed_km_per_h : ℕ)
  (h_length : length = 1600)
  (h_speed : speed_km_per_h = 144) :
  (1600 : ℕ) / (let speed_m_per_s := speed_km_per_h * 1000 / 3600 in speed_m_per_s) = 40 :=
by
  have h_length_1600 : length = 1600 := h_length
  have h_speed_144 : speed_km_per_h = 144 := h_speed
  let speed_m_per_s := speed_km_per_h * 1000 / 3600
  have h_speed_m_per_s : speed_m_per_s = 40 := by
    -- speed_m_per_s calculation proof.
    sorry
  show (length / speed_m_per_s = 40)
  sorry


end train_crosses_pole_in_40_seconds_l562_562726


namespace sequence_infinite_even_odd_l562_562965

noncomputable def a_n (n : ℕ) : ℤ :=
  ⌊↑n * real.sqrt 2⌋ + ⌊↑n * real.sqrt 3⌋

theorem sequence_infinite_even_odd:
  (∀ (a_n : ℕ → ℤ), ∃ infinite_even_infinitely : ℕ, (a_n infinite_even_infinitely % 2 = 0) ∧ (∃ infinite_odd_infinitely : ℕ, (a_n infinite_odd_infinitely % 2 = 1))) :=
by
  sorry

end sequence_infinite_even_odd_l562_562965


namespace no_nine_white_balls_l562_562443

theorem no_nine_white_balls :
  let P_initial := (1 : ℤ) * 1 * 1 * 1 * (-1) * (-1) * (-1) * (-1) * (-1),
      black_ball_value := (1 : ℤ),
      white_ball_value := (-1 : ℤ),
      transform : ℤ → ℤ := λ P, P * P
  in ∀ k ≥ 0, (P_initial = -1) → (∀ k ≥ 2, transform (-1) k = 1) → false :=
by
  -- The proof steps will go here
  sorry

end no_nine_white_balls_l562_562443


namespace find_n_l562_562651

theorem find_n (n k : ℕ) (a b : ℝ) (h_pos : k > 0) (h_n : n ≥ 2) (h_ab_neq : a ≠ 0 ∧ b ≠ 0) (h_a : a = (k + 1) * b) : n = 2 * k + 2 :=
by sorry

end find_n_l562_562651


namespace max_distance_curve_C_prime_to_line_l_l562_562478

-- Define curve C and line l in parametric form
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4
def line_l (t x y : ℝ) : Prop := x = √3 - (1/2) * t ∧ y = 1 + (√3/2) * t

-- Define the transformation to curve C'
def transformation (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y
def curve_C_prime (x' y' : ℝ) : Prop := (x'^2) / 4 + (y'^2) / 16 = 1

-- Statement for the maximum distance from any point on curve C' to line l
theorem max_distance_curve_C_prime_to_line_l (M : Π (x' y' : ℝ), Prop) (x0 y0 : ℝ) :
  (curve_C_prime x0 y0) →
  ∃ x y t, transformation x y x0 y0 ∧ line_l t x y ∧ 
    M x0 y0 = (2 + √3 : ℝ) :=
by sorry

end max_distance_curve_C_prime_to_line_l_l562_562478


namespace molecular_weight_al_fluoride_l562_562326

/-- Proving the molecular weight of Aluminum fluoride calculation -/
theorem molecular_weight_al_fluoride (x : ℕ) (h : 10 * x = 840) : x = 84 :=
by sorry

end molecular_weight_al_fluoride_l562_562326


namespace average_six_conseq_ints_l562_562599

theorem average_six_conseq_ints (c d : ℝ) (h₁ : d = c + 2.5) :
  (d - 2 + d - 1 + d + d + 1 + d + 2 + d + 3) / 6 = c + 3 :=
by
  sorry

end average_six_conseq_ints_l562_562599


namespace find_complex_Z_l562_562075

open Complex

theorem find_complex_Z (Z : ℂ) (h : (2 + 4 * I) / Z = 1 - I) : 
  Z = -1 + 3 * I :=
by
  sorry

end find_complex_Z_l562_562075


namespace range_of_a_l562_562865

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 < log a x) ↔ (1/16 < a ∧ a < 1) :=
by 
  sorry

end range_of_a_l562_562865


namespace max_area_rectangle_product_of_distances_l562_562490

-- Given conditions:
def parametric_line (m t : ℝ) : ℝ × ℝ :=
  (m + 1/2 * t, (√3) / 2 * t)

def polar_ellipse (ρ θ : ℝ) : Prop :=
  5 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2 = 45 / ρ ^ 2

def ellipse_in_cartesian (x y : ℝ) : Prop :=
  (x ^ 2) / 9 + (y ^ 2) / 5 = 1

def is_focus (x : ℝ) : Prop := 
  x = 2

-- Problems to prove
theorem max_area_rectangle (m t : ℝ) (x y : ℝ) : 
  ellipse_in_cartesian x y → 
  ∃ α : ℝ, x = 3 * Real.cos α ∧ y = √5 * Real.sin α → 
  6 * √5 ≥ |4 * 3 * Real.cos α * √5 * Real.sin α| := 
sorry

theorem product_of_distances (m t : ℝ) : 
  parametric_line 2 t ∈ ellipse_in_cartesian →
  -25 / 8 * (parameters_of_root ...) = 25 / 8 := 
sorry

end max_area_rectangle_product_of_distances_l562_562490


namespace tangent_line_equation_l562_562236

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562236


namespace perimeter_of_region_l562_562719

noncomputable def side_length : ℝ := 2 / Real.pi

noncomputable def semicircle_perimeter : ℝ := 2

theorem perimeter_of_region (s : ℝ) (p : ℝ) (h1 : s = 2 / Real.pi) (h2 : p = 2) :
  4 * (p / 2) = 4 :=
by
  sorry

end perimeter_of_region_l562_562719


namespace count_zero_terms_l562_562852

open Real

def S (n : Nat) : Real :=
  ∑ k in Finset.range (n + 1), sin ((k + 1) * π / 7)

theorem count_zero_terms : Finset.filter (λ n, S n = 0) (Finset.range 2018).card = 288 := by
  sorry

end count_zero_terms_l562_562852


namespace neither_plaid_nor_purple_l562_562974

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l562_562974


namespace B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l562_562822

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x < 1 ∨ x > 2}
def setB (m : ℝ) : Set ℝ := 
  if m = 0 then {x : ℝ | x > 1} 
  else if m < 0 then {x : ℝ | x > 1 ∨ x < (2/m)}
  else if 0 < m ∧ m < 2 then {x : ℝ | 1 < x ∧ x < (2/m)}
  else if m = 2 then ∅
  else {x : ℝ | (2/m) < x ∧ x < 1}

-- Complement of set A
def complementA : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

-- Proposition: if B subset of complement of A
theorem B_subset_complementA (m : ℝ) : setB m ⊆ complementA ↔ 1 ≤ m ∧ m ≤ 2 := by
  sorry

-- Similarly, we can define the other two propositions
theorem A_intersection_B_nonempty (m : ℝ) : (setA ∩ setB m).Nonempty ↔ m < 1 ∨ m > 2 := by
  sorry

theorem A_union_B_eq_A (m : ℝ) : setA ∪ setB m = setA ↔ m ≥ 2 := by
  sorry

end B_subset_complementA_A_intersection_B_nonempty_A_union_B_eq_A_l562_562822


namespace required_framing_feet_l562_562681

-- Definition of the original picture dimensions
def original_width : ℕ := 4
def original_height : ℕ := 6

-- Definition of the enlargement factor
def enlargement_factor : ℕ := 4

-- Definition of the border width
def border_width : ℕ := 3

-- Given the enlarged and bordered dimensions, calculate the required framing in feet
theorem required_framing_feet : 
  let enlarged_width := enlargement_factor * original_width
  let enlarged_height := enlargement_factor * original_height
  let bordered_width := enlarged_width + 2 * border_width
  let bordered_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (bordered_width + bordered_height)
  let perimeter_feet := (perimeter_inches + 11) / 12
  perimeter_feet = 9 :=
begin
  sorry
end

end required_framing_feet_l562_562681


namespace prove_number_of_irrationals_l562_562386

-- Definitions for each number
def numbers : List ℚ := [3/8, 2, 0.22222222222] -- Rational numbers in the list
def real_numbers : List ℝ := [(-Real.sqrt 2), -Real.pi, Real.cbrt 9] -- Irrational and real numbers in the list

-- The statement to be proved
theorem prove_number_of_irrationals (list_rationals : List ℚ) (list_reals : List ℝ) :
  (list_reals.filter (λ x, ¬ ∃ q : ℚ, (q : ℝ) = x)).length = 3 :=
by
  sorry

end prove_number_of_irrationals_l562_562386


namespace centroids_triangle_obtuse_l562_562956

def is_centroid (P : Point) (A B C : Point) : Prop := 
  (vector.from P (vector.midpoint A B)) = (1/3) • (vector.from P C) + (1/3) • (vector.from P B)

def radius := Real -- Define radius as real numbers since we are performing geometric calculations

theorem centroids_triangle_obtuse (A B C D E : Point) (r : radius)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A)
  (h_circle : ∀ P ∈ {A, B, C, D, E}, norm (vector.from O P) = r)
  (h_geq : dist A B = dist C D ∧ dist C D = dist D E ∧ dist A B > r) :
  ∃ G1 G2 G3 : Point, 
    is_centroid G1 A B D ∧ 
    is_centroid G2 B C D ∧ 
    is_centroid G3 A D E ∧ 
    ∠ G2 G1 G3 > 90 :=
begin
  -- Proof goes here
  sorry
end

end centroids_triangle_obtuse_l562_562956


namespace depth_of_second_hole_l562_562671

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let total_man_hours1 := workers1 * hours1
  let rate_of_work := depth1 / total_man_hours1
  let workers2 := 45 + 45
  let hours2 := 6
  let total_man_hours2 := workers2 * hours2
  let depth2 := rate_of_work * total_man_hours2
  depth2 = 45 := by
    sorry

end depth_of_second_hole_l562_562671


namespace sin_cos_sum_inequality_l562_562887

theorem sin_cos_sum_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := 
sorry

end sin_cos_sum_inequality_l562_562887


namespace discriminant_non_negative_real_roots_l562_562857

theorem discriminant_non_negative_real_roots (a d : ℝ) :
  let b := a + d
  let c := a + 2 * d
  let Δ := (4 * b)^2 - 4 * a * c 
  Δ = (4 * (3 * a + 4 * d))^2 :=
by
  let b := a + d
  let c := a + 2 * d
  let Δ := (4 * b)^2 - 4 * a * c 
  have h1 : 4 * b = 4 * (a + d) := by ring
  rw [h1]
  have h2 : Δ = 16 * (a + d)^2 - 4 * a * c := by ring_exp
  rw [h2]
  have h3 : 16 * (a + d)^2 = 16 * (a^2 + 2 * a * d + d^2) := by ring
  rw [h3]
  have h4 : 4 * a * c = 4 * a * (a + 2 * d) := by ring
  rw [h4]
  have h5 : 4 * a * (a + 2 * d) = 4 * a^2 + 8 * a * d := by ring
  rw [h5]
  have h6 : 16 * (a^2 + 2 * a * d + d^2) - 4 * a ^2 - 8 * a * d = 12 * a^2 + 16 * a * d + 16 * d^2 := by ring
  rw [h6]
  have h7 : 12 * a^2 + 16 * a * d + 16 * d^2 = 4 * (3 * a + 4 * d)^2 := by ring
  rw [h7]
  exact Eq.refl _

end discriminant_non_negative_real_roots_l562_562857


namespace product_of_symmetric_complex_numbers_l562_562581

def z1 : ℂ := 1 + 2 * Complex.I

def z2 : ℂ := -1 + 2 * Complex.I

theorem product_of_symmetric_complex_numbers :
  z1 * z2 = -5 :=
by 
  sorry

end product_of_symmetric_complex_numbers_l562_562581


namespace sin_of_difference_l562_562450

-- Define the conditions
variables {α : ℝ} (h1 : cos (α + π) = (sqrt 3) / 2) (h2 : π < α ∧ α < 3 * π / 2)

-- Define the theorem to be proved
theorem sin_of_difference (h1 : cos (α + π) = (sqrt 3) / 2) (h2 : π < α ∧ α < 3 * π / 2) : 
  sin (2 * π - α) = 1 / 2 := 
sorry

end sin_of_difference_l562_562450


namespace annie_ride_miles_l562_562935

noncomputable def annie_ride_distance : ℕ := 14

theorem annie_ride_miles
  (mike_base_rate : ℝ := 2.5)
  (mike_per_mile_rate : ℝ := 0.25)
  (mike_miles : ℕ := 34)
  (annie_base_rate : ℝ := 2.5)
  (annie_bridge_toll : ℝ := 5.0)
  (annie_per_mile_rate : ℝ := 0.25)
  (annie_miles : ℕ := annie_ride_distance)
  (mike_cost : ℝ := mike_base_rate + mike_per_mile_rate * mike_miles)
  (annie_cost : ℝ := annie_base_rate + annie_bridge_toll + annie_per_mile_rate * annie_miles) :
  mike_cost = annie_cost → annie_miles = 14 := 
by
  sorry

end annie_ride_miles_l562_562935


namespace jim_miles_remaining_l562_562892

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (h_total : total_miles = 1200) (h_driven : miles_driven = 384) 
: total_miles - miles_driven = 816 := 
by 
  subst h_total 
  subst h_driven 
  exact Nat.sub_eq_succ.pred_eq 816

end jim_miles_remaining_l562_562892


namespace tangent_line_equation_l562_562237

noncomputable def curve (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

def point := (-1 : ℝ, -3 : ℝ)

theorem tangent_line_equation :
  ∃ m b : ℝ, (m = 5 ∧ b = 2) ∧ (∀ x : ℝ, ∀ y : ℝ, (y = curve x → point = (x, y) → 5 * x - y + 2 = 0)) :=
sorry

end tangent_line_equation_l562_562237


namespace bill_experience_l562_562750

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l562_562750


namespace water_transfer_difference_l562_562266

theorem water_transfer_difference
  (initial_small_bucket : ℕ)
  (initial_large_bucket : ℕ)
  (transferred_amount : ℕ)
  (final_small_bucket : ℕ)
  (final_large_bucket : ℕ)
  (initial_small_bucket = 7)
  (initial_large_bucket = 5)
  (transferred_amount = 2)
  (final_small_bucket = initial_small_bucket + transferred_amount)
  (final_large_bucket = initial_large_bucket - transferred_amount) :
  final_small_bucket - final_large_bucket = 6 :=
by
  sorry

end water_transfer_difference_l562_562266


namespace unbalanced_primitive_integer_squares_infinite_l562_562900

theorem unbalanced_primitive_integer_squares_infinite :
  ∃ (B D : ℤ × ℤ × ℤ), 
  (gcd B.1.1 (gcd B.1.2 B.2) = 1 ∧ gcd D.1.1 (gcd D.1.2 D.2) = 1) ∧
  (abs B.1.1 + abs B.1.2 + abs B.2 ≠ abs D.1.1 + abs D.1.2 + abs D.2) ∧
  ∀ t : ℤ, B.1.1 ^ 2 + B.1.2 ^ 2 + B.2 ^ 2 = t ^ 2 ∧ D.1.1 ^ 2 + D.1.2 ^ 2 + D.2 ^ 2 = t ^ 2 ∧
  B.1.1 * D.1.1 + B.1.2 * D.1.2 + B.2 * D.2 = 0 ∧
  (∀ c : ℤ, c ≠ 0 → B ≠ c • D) :=
sorry

end unbalanced_primitive_integer_squares_infinite_l562_562900


namespace piano_harmonies_count_l562_562456

theorem piano_harmonies_count :
  (nat.choose 7 3) + (nat.choose 7 4) + (nat.choose 7 5) + (nat.choose 7 6) + (nat.choose 7 7) = 99 :=
by
  sorry

end piano_harmonies_count_l562_562456


namespace original_number_is_1212_or_2121_l562_562610

theorem original_number_is_1212_or_2121 (x y z t : ℕ) (h₁ : t ≠ 0)
  (h₂ : 1000 * x + 100 * y + 10 * z + t + 1000 * t + 100 * x + 10 * y + z = 3333) : 
  (1000 * x + 100 * y + 10 * z + t = 1212) ∨ (1000 * x + 100 * y + 10 * z + t = 2121) :=
sorry

end original_number_is_1212_or_2121_l562_562610


namespace tangent_line_equation_at_point_l562_562213

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562213


namespace second_third_parts_length_l562_562970

variable (total_length : ℝ) (first_part : ℝ) (last_part : ℝ)
variable (second_third_part_length : ℝ)

def is_equal_length (x y : ℝ) := x = y

theorem second_third_parts_length :
  total_length = 74.5 ∧ first_part = 15.5 ∧ last_part = 16 → 
  is_equal_length (second_third_part_length) 21.5 :=
by
  intros h
  let remaining_distance := total_length - first_part - last_part
  let second_third_part_length := remaining_distance / 2
  sorry

end second_third_parts_length_l562_562970


namespace marians_new_balance_l562_562925

theorem marians_new_balance :
  ∀ (initial_balance grocery_cost return_amount : ℝ),
    initial_balance = 126 →
    grocery_cost = 60 →
    return_amount = 45 →
    let gas_cost := grocery_cost / 2 in
    let new_balance_before_returns := initial_balance + grocery_cost + gas_cost in
    new_balance_before_returns - return_amount = 171 :=
begin
  intros initial_balance grocery_cost return_amount h_init h_grocery h_return,
  let gas_cost := grocery_cost / 2,
  let new_balance_before_returns := initial_balance + grocery_cost + gas_cost,
  have h_gas : gas_cost = 30,
  { 
    rw h_grocery, 
    norm_num },
  have h_new_balance : new_balance_before_returns = 216,
  {
    rw [h_init, h_grocery, h_gas],
    norm_num },
  rw [h_new_balance, h_return],
  norm_num,
end

end marians_new_balance_l562_562925


namespace simplify_fraction_l562_562103

theorem simplify_fraction (a b : ℕ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b :=
sorry

end simplify_fraction_l562_562103


namespace cost_of_fence_l562_562341

theorem cost_of_fence (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 59) : 
  let side_length := Real.sqrt area in
  let perimeter := 4 * side_length in
  let cost := perimeter * price_per_foot in
  cost = 4012 :=
by
  sorry

end cost_of_fence_l562_562341


namespace fraction_simplification_l562_562016

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end fraction_simplification_l562_562016


namespace used_mystery_books_l562_562391

theorem used_mystery_books (total_books used_adventure_books new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13.0)
  (h3 : new_crime_books = 15.0) :
  total_books - (used_adventure_books + new_crime_books) = 17.0 := by
  sorry

end used_mystery_books_l562_562391


namespace number_of_balls_sold_l562_562184

theorem number_of_balls_sold 
  (selling_price : ℤ) (loss_per_5_balls : ℤ) (cost_price_per_ball : ℤ) (n : ℤ) 
  (h1 : selling_price = 720)
  (h2 : loss_per_5_balls = 5 * cost_price_per_ball)
  (h3 : cost_price_per_ball = 48)
  (h4 : (n * cost_price_per_ball) - selling_price = loss_per_5_balls) :
  n = 20 := 
by
  sorry

end number_of_balls_sold_l562_562184


namespace Konstantin_mother_returns_amount_l562_562550

theorem Konstantin_mother_returns_amount
  (deposit_usd : ℝ)
  (exchange_rate : ℝ)
  (equivalent_rubles : ℝ)
  (h_deposit_usd : deposit_usd = 10000)
  (h_exchange_rate : exchange_rate = 58.15)
  (h_equivalent_rubles : equivalent_rubles = deposit_usd * exchange_rate) :
  equivalent_rubles = 581500 :=
by {
  rw [h_deposit_usd, h_exchange_rate] at h_equivalent_rubles,
  exact h_equivalent_rubles
}

end Konstantin_mother_returns_amount_l562_562550


namespace problem1_problem2_problem3_problem4_l562_562755

theorem problem1 : 9 - 5 - (-4) + 2 = 10 := by
  sorry

theorem problem2 : (- (3 / 4) + 7 / 12 - 5 / 9) / (-(1 / 36)) = 26 := by
  sorry

theorem problem3 : -2^4 - ((-5) + 1 / 2) * (4 / 11) + (-2)^3 / (abs (-3^2 + 1)) = -15 := by
  sorry

theorem problem4 : (100 - 1 / 72) * (-36) = -(3600) + (1 / 2) := by
  sorry

end problem1_problem2_problem3_problem4_l562_562755


namespace octagon_area_l562_562300

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562300


namespace octagon_area_l562_562297

noncomputable def area_of_circle := 256 * Real.pi
def radius_of_circle : ℝ := Real.sqrt (area_of_circle / Real.pi)
def theta : ℝ := Real.pi / 4 -- 45 degrees in radians

theorem octagon_area (h : radius_of_circle = 16): 
  let r := radius_of_circle in
  let area_of_one_triangle := (1 / 2) * r^2 * Real.sin theta in
  let total_area := 8 * area_of_one_triangle in
  total_area = 512 * Real.sqrt 2 := 
  by
    sorry

end octagon_area_l562_562297


namespace simplify_rationalize_l562_562195

theorem simplify_rationalize :
  (1 / (1 + (1 / (real.cbrt 3 + 1)))) = 9 / 13 :=
by
  sorry

end simplify_rationalize_l562_562195


namespace percent_of_75_of_125_l562_562351

theorem percent_of_75_of_125 : (75 / 125) * 100 = 60 := by
  sorry

end percent_of_75_of_125_l562_562351


namespace proof_problem_l562_562067

-- Define the necessary conditions from the problem
variables (A B C D E : Type) [linear_ordered_field A]
variables (m_angle : A → A → A → A) (BC EC k p q : A)
variables (perp : A → A → Prop) (angle_sum : A → A → A → A)

-- The conditions 
variable (angle_A_eq : m_angle A B C = 60)
variable (BC_eq : BC = 15)
variable (BD_perp_AC : perp B D ∧ perp D C)
variable (CE_perp_AB : perp C E ∧ perp E B)
variable (angle_DBC_5angle_ECB : m_angle D B C = 5 * m_angle E C B)

-- Goal: prove k + p + q = 18
theorem proof_problem :
  ∃ k p q : ℕ, BC = 15 ∧
  m_angle A B C = 60 ∧
  perp B D ∧ perp D C ∧
  perp C E ∧ perp E B ∧
  m_angle D B C = 5 * m_angle E C B ∧
  EC = k * real.sqrt p + real.sqrt q ∧
  p % 4 ≠ 0 ∧
  q % 4 ≠ 0 ∧
  k + p + q = 18 :=
  sorry

end proof_problem_l562_562067


namespace basketball_win_percentage_l562_562696

theorem basketball_win_percentage :
  ∀ (won_initial : ℕ) (played_initial : ℕ) (remaining_games : ℕ) (required_wins : ℕ),
    won_initial = 35 →
    played_initial = 50 →
    remaining_games = 25 →
    required_wins = 13 →
    (won_initial + required_wins) * 100 / (played_initial + remaining_games) = 64 := by
  intros won_initial played_initial remaining_games required_wins
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end basketball_win_percentage_l562_562696


namespace sufficient_but_not_necessary_for_increasing_l562_562981

variable {a : ℝ}

def f (x : ℝ) : ℝ := x^2 - 4*a*x + 3

theorem sufficient_but_not_necessary_for_increasing :
  (∀ x : ℝ, x ≥ 2 → deriv (f x) ≥ 0) ↔ a = 1 := by
  sorry

end sufficient_but_not_necessary_for_increasing_l562_562981


namespace opposite_of_neg_3_is_3_l562_562993

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l562_562993


namespace trigonometric_identity_l562_562436

theorem trigonometric_identity :
  4 * Real.cos (10 * (Real.pi / 180)) - Real.tan (80 * (Real.pi / 180)) = -Real.sqrt 3 := 
by 
  sorry

end trigonometric_identity_l562_562436


namespace water_fall_amount_l562_562548

theorem water_fall_amount (M_before J_before M_after J_after n : ℕ) 
  (h1 : M_before = 48) 
  (h2 : M_before = J_before + 32)
  (h3 : M_after = M_before + n) 
  (h4 : J_after = J_before + n)
  (h5 : M_after = 2 * J_after) : 
  n = 16 :=
by 
  -- proof omitted
  sorry

end water_fall_amount_l562_562548


namespace find_coordinates_of_A_l562_562875

-- Definition of points in Cartesian coordinate system
structure Point where
  x : Int
  y : Int

-- Definitions for translations
def translate_left (A : Point) (distance : Int) : Point :=
  { x := A.x - distance, y := A.y }

def translate_up (A : Point) (distance : Int) : Point :=
  { x := A.x, y := A.y + distance }

-- Given conditions
variables A : Point
variables h1 : ∃ d, translate_left A d = { x := 1, y := 2 }
variables h2 : ∃ d, translate_up A d = { x := 3, y := 4 }

-- Proof statement
theorem find_coordinates_of_A : A = { x := 3, y := 2 } :=
sorry

end find_coordinates_of_A_l562_562875


namespace probability_gt1_probability_more_1cm_l562_562722

-- Definitions for the scores coming from two different intervals.
def scoresA (x : ℝ) : Prop := 7.5 ≤ x ∧ x < 8.5
def scoresB (x : ℝ) : Prop := 9.5 ≤ x ∧ x < 10.5

-- The six scores denoted as x1, x2, x3 for the first three shots and y1, y2, y3 for the next three shots.
variables (x1 x2 x3 y1 y2 y3 : ℝ)

-- Assumptions about the score's intervals.
axiom h1 : scoresA x1
axiom h2 : scoresA x2
axiom h3 : scoresA x3
axiom h4 : scoresB y1
axiom h5 : scoresB y2
axiom h6 : scoresB y3

-- The proof for the probability |a - b| > 1
theorem probability_gt1 : 
  let pairs := [(x1, x2), (x1, x3), (x2, x3), (y1, y2), (y1, y3), (y2, y3), (x1, y1), (x1, y2), (x1, y3), (x2, y1), (x2, y2), (x2, y3), (x3, y1), (x3, y2), (x3, y3)] in
  let cond := λ (a b : ℝ), |a - b| > 1 in
  probability_gt1 := 9 / 15 :=
sorry

-- Reference values for geometric probabilities
def prop_more_than_1cm (ABC_area more_1cm_area : ℝ) : ℝ := (ABC_area - 3 * more_1cm_area) / ABC_area

-- Areas calculated for triangle and regions within 1cm of the vertices.
axiom ABC_area : ℝ := (9 * Real.sqrt 3) / 4
axiom more_1cm_area : ℝ := 1/2 * 1^2 * Real.pi / 3

-- The proof for the probability of landing point more than 1cm away from A, B, C
theorem probability_more_1cm : prop_more_than_1cm ABC_area more_1cm_area = 1 - (2 * Real.sqrt 3 * Real.pi / 27) :=
sorry

end probability_gt1_probability_more_1cm_l562_562722


namespace largest_divisor_of_m_l562_562661

-- Definitions
def positive_integer (m : ℕ) : Prop := m > 0
def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement
theorem largest_divisor_of_m (m : ℕ) (h1 : positive_integer m) (h2 : divisible_by (m^2) 54) : ∃ k : ℕ, k = 9 ∧ k ∣ m := 
sorry

end largest_divisor_of_m_l562_562661


namespace average_of_roots_l562_562800

theorem average_of_roots (a b c : ℚ) (h : a ≠ 0) (h_eq : 3 * a = 9 ∧ 4 * a = 36 / 9 ∧ -8 * a = -24) : 
  average_of_roots (Poly.of_coeffs![3, 4, -8]) = -2/3 :=
sorry

end average_of_roots_l562_562800


namespace exist_r_nonzero_l562_562901

open Complex BigOperators

noncomputable def f (A : set (zmod n)) (r : zmod n) : ℂ :=
  ∑ s in A, exp(2 * real.pi * I * (r * s).val / n)

theorem exist_r_nonzero (A : set (zmod n)) (n : ℕ) (hA_card : |A| ≤ real.log n / 100) :
  ∃ r : zmod n, r ≠ 0 ∧ complex.abs (f A r) ≥ |A| / 2 := 
sorry

end exist_r_nonzero_l562_562901


namespace marikas_father_twice_her_age_l562_562945

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l562_562945


namespace mortgage_loan_l562_562641

theorem mortgage_loan (D : ℝ) (hD : D = 2000000) : 
  ∃ C : ℝ, (C = D + 0.75 * C) ∧ (0.75 * C = 6000000) :=
by
  -- (Optional) Set up the problem with condition D = 2,000,000
  use 8000000  -- From the solution steps, we found C = 8000000
  split
  · -- Show that the equation C = D + 0.75 * C is satisfied
    rw [hD]
    linarith
  · -- Show the mortgage loan amount is 6,000,000
    linarith

end mortgage_loan_l562_562641


namespace minimum_moves_to_determine_polynomial_l562_562960

-- Define quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define conditions as per the given problem
variables {f g : ℝ → ℝ}
def is_quadratic (p : ℝ → ℝ) := ∃ a b c : ℝ, ∀ x : ℝ, p x = quadratic a b c x

axiom f_is_quadratic : is_quadratic f
axiom g_is_quadratic : is_quadratic g

-- Define the main problem statement
theorem minimum_moves_to_determine_polynomial (n : ℕ) :
  (∀ (t : ℕ → ℝ), (∀ m ≤ n, (f (t m) = g (t m)) ∨ (f (t m) ≠ g (t m))) →
  (∃ a b c: ℝ, ∀ x: ℝ, f x = quadratic a b c x ∨ g x = quadratic a b c x)) ↔ n = 8 :=
sorry -- Proof is omitted

end minimum_moves_to_determine_polynomial_l562_562960


namespace range_x_sub_cos_y_l562_562343

theorem range_x_sub_cos_y (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) : 
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 :=
sorry

end range_x_sub_cos_y_l562_562343


namespace max_L_shaped_figures_in_5x7_rectangle_l562_562325

def L_shaped_figure : Type := ℕ

def rectangle_area := 5 * 7

def l_shape_area := 3

def max_l_shapes_in_rectangle (rect_area : ℕ) (l_area : ℕ) : ℕ := rect_area / l_area

theorem max_L_shaped_figures_in_5x7_rectangle : max_l_shapes_in_rectangle rectangle_area l_shape_area = 11 :=
by
  sorry

end max_L_shaped_figures_in_5x7_rectangle_l562_562325


namespace complex_magnitude_sqrt_2_l562_562512

theorem complex_magnitude_sqrt_2 (z : ℂ) (h : (3 + 2 * complex.i) * z = 5 - complex.i) : complex.abs z = real.sqrt 2 := by
  sorry

end complex_magnitude_sqrt_2_l562_562512


namespace volume_of_pyramid_l562_562440

noncomputable def volume_of_regular_triangular_pyramid (h R : ℝ) : ℝ :=
  (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4

theorem volume_of_pyramid (h R : ℝ) : volume_of_regular_triangular_pyramid h R = (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4 :=
  by sorry

end volume_of_pyramid_l562_562440


namespace find_m_plus_n_l562_562773

noncomputable def expected_value (r c : ℕ) : ℚ :=
if (r, c) = (1, 1) ∨ (r, c) = (5, 5) then 1 else
if (r, c) = (1, 5) ∨ (r, c) = (5, 1) then 5 else
let adj_vals := [expected_value (r-1) c, expected_value (r+1) c, expected_value r (c-1), expected_value r (c+1)] in
(sum adj_vals) / 4

theorem find_m_plus_n :
  let m := 13
  let n := 5
  expected_value 2 2 = 13 / 5 ∧ m + n = 18 :=
by sorry

end find_m_plus_n_l562_562773


namespace coeff_x4_expansion_l562_562607

theorem coeff_x4_expansion : 
  (∑ k in Finset.range 11, (Nat.choose 10 k) * (1 ^ (10 - k)) * (x ^ k)) = 210 :=
sorry

end coeff_x4_expansion_l562_562607


namespace percentage_blue_shirts_l562_562117

theorem percentage_blue_shirts (total_students := 600) 
 (percent_red := 23)
 (percent_green := 15)
 (students_other := 102)
 : (100 - (percent_red + percent_green + (students_other / total_students) * 100)) = 45 := by
  sorry

end percentage_blue_shirts_l562_562117


namespace length_of_AD_in_triangle_ABC_l562_562523

noncomputable def length_AD (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (AB BC AC BD DC : ℝ) : ℝ :=
  let AD := Real.sqrt 507 in
  AD

theorem length_of_AD_in_triangle_ABC :
  ∀ (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
    (AB BC AC : ℝ)
    (cond1 : AB = 26) (cond2 : BC = 26) (cond3 : AC = 24)
    (D_midpoint_BC : BD = DC)
    (BD : BD = 13) (DC : DC = 13),
  (AD = Real.sqrt 507) :=
by
  sorry

end length_of_AD_in_triangle_ABC_l562_562523


namespace rectangle_area_l562_562530

theorem rectangle_area (x : ℝ) (h1 : 10 * x = 160) : 4 * (4 * x * x) = 1024 := by
  have h2 : x = 16 := by
    linarith [h1]
  rw [h2]
  ring

end rectangle_area_l562_562530


namespace barbed_wire_rate_l562_562603

noncomputable def side_length_of_square (area : ℝ) : ℝ := real.sqrt area

noncomputable def perimeter_with_gates (side_length gate_width : ℝ) : ℝ :=
  4 * side_length - 2 * gate_width

noncomputable def rate_per_meter (total_cost length_of_wire : ℝ) : ℝ :=
  total_cost / length_of_wire

theorem barbed_wire_rate :
  let area := 3136
  let gate_width := 1
  let total_cost := 932.40
  let side_length := side_length_of_square area
  let perimeter := perimeter_with_gates side_length gate_width
  let length_of_wire := perimeter
  in rate_per_meter total_cost length_of_wire = 4.2 :=
by
  sorry

end barbed_wire_rate_l562_562603


namespace remaining_meat_ounces_l562_562191

noncomputable def initial_beef_ounces := 5 * 16
noncomputable def initial_pork_ounces := 8 * 16
noncomputable def initial_chicken_ounces := 4 * 16

noncomputable def beef_per_link := initial_beef_ounces / 20
noncomputable def pork_per_link := initial_pork_ounces / 25
noncomputable def chicken_per_link := initial_chicken_ounces / 15

noncomputable def beef_eaten := 8 * beef_per_link
noncomputable def pork_eaten := 6 * pork_per_link
noncomputable def chicken_eaten := 4 * chicken_per_link

noncomputable def remaining_beef_ounces := initial_beef_ounces - beef_eaten
noncomputable def remaining_pork_ounces := initial_pork_ounces - pork_eaten
noncomputable def remaining_chicken_ounces := initial_chicken_ounces - chicken_eaten

theorem remaining_meat_ounces :
  remaining_beef_ounces = 48 ∧
  remaining_pork_ounces = 97.28 ∧
  remaining_chicken_ounces = 46.92 :=
by
  sorry

end remaining_meat_ounces_l562_562191


namespace percentage_of_sikh_boys_l562_562872

-- Define the conditions
def total_boys : ℕ := 850
def percentage_muslim_boys : ℝ := 0.46
def percentage_hindu_boys : ℝ := 0.28
def boys_other_communities : ℕ := 136

-- Theorem to prove the percentage of Sikh boys is 10%
theorem percentage_of_sikh_boys : 
  (((total_boys - 
      (percentage_muslim_boys * total_boys + 
       percentage_hindu_boys * total_boys + 
       boys_other_communities))
    / total_boys) * 100 = 10) :=
by
  -- sorry prevents the need to provide proof here
  sorry

end percentage_of_sikh_boys_l562_562872


namespace opposite_of_neg_3_is_3_l562_562996

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l562_562996


namespace regular_octagon_area_l562_562290

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562290


namespace mean_equality_l562_562249

-- Define average calculation function
def average (a b c : ℕ) : ℕ :=
  (a + b + c) / 3

def average_two (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem mean_equality (x : ℕ) 
  (h : average 8 16 24 = average_two 10 x) : 
  x = 22 :=
by {
  -- The actual proof is here
  sorry
}

end mean_equality_l562_562249


namespace meaningful_expr_x_value_l562_562515

theorem meaningful_expr_x_value (x : ℝ) :
  (10 - x >= 0) ∧ (x ≠ 4) ↔ (x = 8) :=
begin
  -- proof omitted
  sorry
end

end meaningful_expr_x_value_l562_562515


namespace add_and_round_nearest_tenth_l562_562729

theorem add_and_round_nearest_tenth (a b : ℝ) (h₁ : a = 95.32) (h₂ : b = 47.268) :
  Float.round_nearest_ten (a + b) = 142.6 := 
by
  sorry

end add_and_round_nearest_tenth_l562_562729


namespace fraction_of_students_getting_F_l562_562868

theorem fraction_of_students_getting_F
  (students_A students_B students_C students_D passing_fraction : ℚ) 
  (hA : students_A = 1/4)
  (hB : students_B = 1/2)
  (hC : students_C = 1/8)
  (hD : students_D = 1/12)
  (hPassing : passing_fraction = 0.875) :
  (1 - (students_A + students_B + students_C + students_D)) = 1/24 :=
by
  sorry

end fraction_of_students_getting_F_l562_562868


namespace area_regular_octagon_in_circle_l562_562285

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562285


namespace find_c_value_l562_562030

theorem find_c_value : (8^3 * 9^3) / 679 ≈ 550 := sorry

end find_c_value_l562_562030


namespace Tonya_initial_stamps_l562_562893

theorem Tonya_initial_stamps :
  ∀ (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (jimmy_matchbooks : ℕ) (tonya_remaining_stamps : ℕ),
  stamps_per_match = 12 →
  matches_per_matchbook = 24 →
  jimmy_matchbooks = 5 →
  tonya_remaining_stamps = 3 →
  tonya_remaining_stamps + (jimmy_matchbooks * matches_per_matchbook) / stamps_per_match = 13 := 
by
  intros stamps_per_match matches_per_matchbook jimmy_matchbooks tonya_remaining_stamps
  sorry

end Tonya_initial_stamps_l562_562893


namespace minimal_marked_points_2018_l562_562125

theorem minimal_marked_points_2018 :
  ∀ (points : list (Real × Real)), points.length = 2018 →
  (∀ (p1 p2 : (Real × Real)), p1 ≠ p2 → dist p1 p2 ≠ dist p2 p1) →
  ∃ m : Nat, minimal_number_of_marked_points points = 2 * m + 1 :=
by
  -- Definitions and additional constraints would be inserted here.
  sorry

-- Assuming we have a function that calculates the minimal number of marked points.
def minimal_number_of_marked_points (points : list (Real × Real)) : Nat := sorry

end minimal_marked_points_2018_l562_562125


namespace pentagon_area_correct_l562_562208

noncomputable def area_pentagon (EA AB BC CD DE : ℝ) (angleA angleB angleC : ℝ) : ℝ :=
  let area_EAB := sqrt 3  -- Area of the equilateral triangle EAB
  let area_BCD := (1 / 2) * BC * CD  -- Area of the right-angle triangle BCD
  let area_ACDE := 6  -- Area of the rectangle ACDE
  area_EAB + area_BCD + area_ACDE

theorem pentagon_area_correct : 
  area_pentagon 2 2 2 3 3 120 120 90 = 9 + sqrt 3 := 
by sorry

end pentagon_area_correct_l562_562208


namespace rationalize_denominator_l562_562597

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2) :=
by
  sorry

end rationalize_denominator_l562_562597


namespace shortened_commercial_length_l562_562638

theorem shortened_commercial_length
  (original_length : ℕ := 30)
  (percentage_shortening : ℝ := 0.30) :
  let amount_shortened := (percentage_shortening * original_length : ℝ) in
  (original_length - amount_shortened.floor) = 21 :=
begin
  sorry
end

end shortened_commercial_length_l562_562638


namespace area_regular_octagon_in_circle_l562_562283

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562283


namespace set_equals_interval_l562_562259

theorem set_equals_interval : {x : ℝ | x ≥ 2} = set.Ici 2 := 
sorry

end set_equals_interval_l562_562259


namespace sum_of_angles_l562_562958

theorem sum_of_angles (ACB CAD : ℝ) (h1 : ACB = 60) (h2 : CAD = 50) : ACB + CAD = 70 :=
by
  -- Definitions and conditions
  have ACB_eq : ACB = 60 := h1
  have CAD_eq : CAD = 50 := h2
  -- Result
  calc
    (ACB + CAD) = 60 + 50 : by rw [ACB_eq, CAD_eq]
            ... = 70 : by norm_num

end sum_of_angles_l562_562958


namespace train_cross_bridge_in_30_seconds_l562_562498

def train_length : ℝ := 110 -- in meters
def train_speed_kmph : ℝ := 60 -- in kilometers per hour
def bridge_length : ℝ := 390 -- in meters

def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

def total_distance : ℝ := train_length + bridge_length

def time_taken : ℝ := total_distance / train_speed_mps

theorem train_cross_bridge_in_30_seconds :
  time_taken ≈ 30 :=
by sorry

end train_cross_bridge_in_30_seconds_l562_562498


namespace regular_octagon_area_l562_562289

open Real 

theorem regular_octagon_area (r : ℝ) (A : ℝ) (hA : A = 256 * π) (hr : r = 16) :
  let octagon_area : ℝ := 8 * (1 / 2 * r^2 * sin (π / 4))
  octagon_area = 512 * sqrt 2 :=
by
  have hA_eq : π * r^2 = 256 * π, from hA,
  have hr_eq : r = 16, by { sorry }, -- This follows directly from hA_eq
  have octagon_area_def : octagon_area = 8 * (1 / 2 * 16^2 * sin (π / 4)), by { sorry }, -- Plugging in r = 16
  have sin_π_4 : sin (π / 4) = sqrt 2 / 2, by { sorry }, -- Known value of sin(45°)
  have oct_area_calc : octagon_area = 8 * (1 / 2 * 16^2 * (sqrt 2 / 2)), by { sorry }, -- Substituting sin(π / 4)
  have oct_area_simpl : octagon_area = 512 * sqrt 2, by { sorry }, -- Simplifying the calculation
  exact oct_area_simpl. -- Concluding the proof

end regular_octagon_area_l562_562289


namespace quadratic_expression_value_l562_562811

variable (x y : ℝ)

theorem quadratic_expression_value (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 100 := 
by 
  sorry

end quadratic_expression_value_l562_562811


namespace max_value_and_permutations_l562_562152

def max_value_permutation (xs : List ℕ) : ℕ :=
  xs.head! * xs.get! 1 + xs.get! 1 * xs.get! 2 + xs.get! 2 * xs.get! 3 + 
  xs.get! 3 * xs.get! 4 + xs.get! 4 * xs.get! 5 + xs.get! 5 * xs.head!

theorem max_value_and_permutations:
  let S := {xs : List ℕ | xs ~ [1, 2, 3, 4, 5, 6]} in
  ( ∃ xs ∈ S, max_value_permutation xs = 79 ) ∧
  ( ∀ xs ∈ S, max_value_permutation xs ≤ 79 ) ∧
  ( Multiset.card (Multiset.filter (λ xs, max_value_permutation xs = 79) S.toMultiset) = 6 ) →
  79 + 6 = 85 :=
by
  sorry

end max_value_and_permutations_l562_562152


namespace complement_intersection_l562_562156

open Set Real

def U : Set Real := univ
def A : Set Real := {x : Real | x^2 - 4 ≥ 0}
def B : Set Real := {x : Real | 2^x ≥ 2}

theorem complement_intersection (x : Real) : x ∈ ((U \ A) ∩ B) ↔ 1 ≤ x ∧ x < 2 :=
by
  sorry

end complement_intersection_l562_562156


namespace number_of_pieces_of_paper_used_l562_562891

theorem number_of_pieces_of_paper_used
  (P : ℕ)
  (h1 : 1 / 5 > 0)
  (h2 : 2 / 5 > 0)
  (h3 : 1 < (P : ℝ) * (1 / 5) + 2 / 5 ∧ (P : ℝ) * (1 / 5) + 2 / 5 ≤ 2) : 
  P = 8 :=
sorry

end number_of_pieces_of_paper_used_l562_562891


namespace bill_experience_l562_562751

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l562_562751


namespace negation_of_proposition_l562_562991

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l562_562991


namespace center_of_circle_l562_562984

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (3, 8)) (h2 : (x2, y2) = (11, -4)) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 2) := by
  sorry

end center_of_circle_l562_562984


namespace sequence_arithmetic_sum_b_n_l562_562824

variable {n : ℕ}
variable {S : ℕ → ℕ} -- S represents the sum of the sequence a_n
variable {a : ℕ → ℕ}
variable {b : ℕ → ℝ} -- using ℝ as the division is involved

-- Part 1
theorem sequence_arithmetic (hS : ∀ n, S n = n^2 + 2 * n)
: ∃ d : ℕ, (∀ n ≥ 2, a n - a (n - 1) = d) ∧ a 1 = S 1 := sorry

-- Part 2
theorem sum_b_n (ha : ∀ n, a n = 2 * n + 1)
: (∀ k ≥ 1, b k = 1 / (a k * a (k + 1))) 
  → ∑ k in Finset.range n, b (k + 1) 
  = n / ↑3 / (2 * n + 3) := sorry

end sequence_arithmetic_sum_b_n_l562_562824


namespace f_is_prime_prime_representation_l562_562906

-- Definitions for a(n) and b(n)
def a (n : ℕ) : ℕ := ((n - 1)!)^2 % n
def b (n : ℕ) : ℕ := ((n - 1)! + 1)^2 % n

-- Definition of the function f(n)
def f (n : ℕ) : ℕ := n * a(n) + 2 * b(n)

-- Theorems to prove
theorem f_is_prime (n : ℕ) : Prime (f n) := sorry

theorem prime_representation (p : ℕ) (h : Prime p) : ∃ n : ℕ, f n = p := sorry

end f_is_prime_prime_representation_l562_562906


namespace orphanage_build_year_l562_562112

noncomputable def time_to_build_orphanage (P₀ : ℝ) (r : ℝ) (C : ℝ) (Yp : ℝ) (Mo : ℝ) (No : ℝ) : ℝ :=
  let total_maintenance_cost := Yp + Mo * No
  let total_principal_needed := total_maintenance_cost / r
  let total_required := C + total_principal_needed
  let t := Real.log (total_required / P₀) / Real.log (1 + r)
  t

theorem orphanage_build_year :
  time_to_build_orphanage 100000 0.05 100000 3960 200 100 ≈ 36 :=
by
  sorry

end orphanage_build_year_l562_562112


namespace zeros_of_g_l562_562066

def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 - 3 * x else -(x^2 - 3 * x)

def g (x : ℝ) : ℝ := f x - x + 3

theorem zeros_of_g :
  { x : ℝ | g x = 0 } = {-1, 1, 3} :=
by
  sorry

end zeros_of_g_l562_562066


namespace nine_by_nine_chessboard_dark_light_excess_l562_562353

theorem nine_by_nine_chessboard_dark_light_excess :
  let board_size := 9
  let odd_row_dark := 5
  let odd_row_light := 4
  let even_row_dark := 4
  let even_row_light := 5
  let num_odd_rows := (board_size + 1) / 2
  let num_even_rows := board_size / 2
  let total_dark_squares := (odd_row_dark * num_odd_rows) + (even_row_dark * num_even_rows)
  let total_light_squares := (odd_row_light * num_odd_rows) + (even_row_light * num_even_rows)
  total_dark_squares - total_light_squares = 1 :=
by {
  sorry
}

end nine_by_nine_chessboard_dark_light_excess_l562_562353


namespace eggs_needed_per_month_l562_562178

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l562_562178


namespace tangent_line_at_point_l562_562217

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562217


namespace coins_on_circle_l562_562267

theorem coins_on_circle (n : ℕ) (h : n ≥ 2) : 
  (∃ arrangement : List ℕ, (∀ i ∈ arrangement, i ∈ List.range n) ∧ (∀ i j ∈ arrangement, i ≠ j) ∧ 
    ∀ k < n, arrangement[(k + arrangement[k]) % n] = arrangement[k + 1]) ↔ n % 2 = 0 :=
by
  sorry

end coins_on_circle_l562_562267


namespace mortgage_loan_l562_562642

theorem mortgage_loan (D : ℝ) (hD : D = 2000000) : 
  ∃ C : ℝ, (C = D + 0.75 * C) ∧ (0.75 * C = 6000000) :=
by
  -- (Optional) Set up the problem with condition D = 2,000,000
  use 8000000  -- From the solution steps, we found C = 8000000
  split
  · -- Show that the equation C = D + 0.75 * C is satisfied
    rw [hD]
    linarith
  · -- Show the mortgage loan amount is 6,000,000
    linarith

end mortgage_loan_l562_562642


namespace no_equal_area_chord_l562_562731

def chord (p : Polygon → ℝ²) (c : ℝ² → ℝ² → ℝ²) := 
  ∀ A B ∈ boundary p, (⟦A,B⟧ ⊆ p)

theorem no_equal_area_chord (p : Polygon) :
  ¬ (∃ c, chord p c ∧ divides_into_equal_parts p c) := 
sorry

end no_equal_area_chord_l562_562731


namespace find_k_l562_562475

def x : ℝ := ∑ n in Finset.range 2007, ((-1) ^ (n + 1)) / (n + 1)
def y : ℝ := ∑ n in Finset.range (2007 - 1005 + 1), (1005 + n)⁻¹

theorem find_k : ∃ k : ℕ, x = y + (k : ℝ)⁻¹ :=
begin
  use 2008,
  sorry
end

end find_k_l562_562475


namespace num_primes_between_50_and_100_count_primes_between_50_and_100_l562_562846

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem num_primes_between_50_and_100 : primes_between 50 100 = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97] :=
by
  sorry

theorem count_primes_between_50_and_100 : 
  List.length (primes_between 50 100) = 10 :=
by
  have h : primes_between 50 100 = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97] := num_primes_between_50_and_100
  rw [h]
  norm_num

end num_primes_between_50_and_100_count_primes_between_50_and_100_l562_562846


namespace card_restacking_l562_562054

theorem card_restacking (n : ℕ) (h1 : even (2 * n)) (h2 : odd 80) (h3 : ∀ i ≤ 2 * n, (i ≤ n → i = i ∨ i+n = i ∨ i = i-n) ∧ (n < i ≤ 2 * n → i = i ∨ i-n = i ∨ i = i+n)) : 2 * n = 240 :=
sorry

end card_restacking_l562_562054


namespace area_of_inscribed_regular_octagon_l562_562307

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562307


namespace price_of_pants_l562_562560

theorem price_of_pants (P : ℝ) (h1 : 4 * 33 = 132) (h2 : 2 * P + 132 = 240) : P = 54 :=
sorry

end price_of_pants_l562_562560


namespace greatest_number_of_quarters_l562_562598

-- Define the conditions
def total_value := 4.80
def three_times_nickels (q : ℕ) := 3 * q

-- Lean statement declaring the proof problem
theorem greatest_number_of_quarters (q : ℕ) (h1 : 0.40 * q = total_value) :
  q = 12 :=
by sorry

end greatest_number_of_quarters_l562_562598


namespace sum_of_ratios_l562_562131

-- Define the conditions and the target proof as a Lean statement.
theorem sum_of_ratios (A B C D E F : Type) 
  (B_is_mid_AC : B = midpoint A C)
  (BD_DC_ratio : divides_line_segment_with_ratio D B C 2 1)
  (AE_EB_ratio : divides_line_segment_with_ratio E A B 1 3) : 
  (EF_over_FC F E C + AF_over_FD F A D) = 13 / 4 :=
sorry

end sum_of_ratios_l562_562131


namespace probability_one_after_six_l562_562141

theorem probability_one_after_six : 
  (∀ i, i ≤ 5 → roll i = 6) →
  (is_fair_die D) →
  (independent_rolls D) →
  (prob_roll D 1 = 1/6) := 
by
  sorry

end probability_one_after_six_l562_562141


namespace tangent_line_equation_l562_562224

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562224


namespace probability_prime_sum_l562_562691

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end probability_prime_sum_l562_562691


namespace find_phi_l562_562484

theorem find_phi (phi : ℝ) (hphi : |phi| < Real.pi / 2)
                (h_period : ∀ x, sin ((1 / 3) * x + phi) = sin (1 / 3 * (x - 2 * Real.pi / 3))) :
                phi = 2 * Real.pi / 9 :=
by 
  sorry

end find_phi_l562_562484


namespace perpendicular_lines_l562_562096

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2a = 0) ∧ (∃ x y : ℝ, (2a - 1)x + ay = 0) ∧ 
  (∀ (m1 m2 : ℝ), m1 = a ∧ m2 = -((2a - 1) / a) → m1 * m2 = -1) → 
  a = 0 ∨ a = 1 :=
by 
  -- We provide the structure for the proof
  sorry

end perpendicular_lines_l562_562096


namespace opposite_of_neg3_l562_562999

theorem opposite_of_neg3 : -(-3) = 3 := 
by 
  sorry

end opposite_of_neg3_l562_562999


namespace cyclic_inequality_l562_562917

theorem cyclic_inequality
  (n : ℕ) (h_n : n ≥ 3)
  (a : Fin n → ℝ) (h_ai : ∀ i, 2 ≤ a i ∧ a i ≤ 3)
  (S : ℝ) (h_S : S = ∑ i, a i) :
  (∑ i : Fin n, (a i)^2 + (a (i + 1))^2 - (a (i + 2))^2) / (a i + a (i + 1) - a (i + 2)) ≤ 2 * S - 2 * n :=
by
  sorry

end cyclic_inequality_l562_562917


namespace intersection_l562_562091

def setA : Set ℝ := { x : ℝ | x^2 - 2*x - 3 < 0 }
def setB : Set ℝ := { x : ℝ | x > 1 }

theorem intersection (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 1 < x ∧ x < 3 := by
  sorry

end intersection_l562_562091


namespace count_irrational_numbers_l562_562387

open Real

/--
Among the real numbers $-\sqrt{5}$, $0.\dot 42\dot 1$, $3.14$, $0$, $\frac{\pi}{2}$, $\frac{}
{22}{7}$, $\sqrt{81}$, $0.1616616661\ldots$ (with one more $6$ between each pair of $1$s),
the number of irrational numbers is $3$.
-/
theorem count_irrational_numbers :
  let L := [ -sqrt 5, 0.421421421421..., 3.14, 0, pi / 2, 22 / 7, sqrt 81, 
             ∑' n : ℕ, (10 ^ n + 1) * (10 : ℝ) ^ (-n - 1) ]
  in (L.filter (λ x, ¬ ∃ q:ℚ, (q:ℝ) = x)).length = 3 :=
by
  sorry

end count_irrational_numbers_l562_562387


namespace area_triangle_equation_l562_562842

-- Define points and their relationships
variables {A B C D H P : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space H] [metric_space P]
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]

-- Given conditions
variables (AB BC CA AD : ℝ) (h1 : AB = BC) (h2 : BC = CA) (h3 : CA = AD)
variables (AH : ℝ) (CD : ℝ) (h4 : orthogonal (A -ᵥ H) (C -ᵥ D))
variables (CP : ℝ) (BP : ℝ) (h5 : orthogonal (C -ᵥ P) (B -ᵥ C)) (h6 : intersect (A -ᵥ H) (C -ᵥ P) = P)

-- Prove the statement
theorem area_triangle_equation
  (S : ℝ)
  (hS : S = (sqrt 3 / 4) * (AP * BD)) :
  S = area_triangle A B C :=
sorry

end area_triangle_equation_l562_562842


namespace smallest_number_divisible_by_set_l562_562784

theorem smallest_number_divisible_by_set : ∃ x : ℕ, (∀ d ∈ [12, 24, 36, 48, 56, 72, 84], (x - 24) % d = 0) ∧ x = 1032 := 
by {
  sorry
}

end smallest_number_divisible_by_set_l562_562784


namespace father_twice_marika_age_in_2036_l562_562948

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l562_562948


namespace balls_boxes_exactly_3_non_matching_l562_562397

/-- The number of ways to place 10 balls labeled 1 to 10 into 10 boxes 
such that each box contains one ball, and exactly 3 balls are placed in boxes 
with non-matching labels, is 360. -/
theorem balls_boxes_exactly_3_non_matching : 
  ∃ (perm : Equiv.Perm (Fin 10)), (∃ (non_matching : Finset (Fin 10)), 
    non_matching.card = 3 ∧ 
    ∀ x ∈ non_matching, perm x ≠ x) → 
    (∃ (matching : Finset (Fin 10)), 
    matching.card = 7 ∧ 
    ∀ x ∈ matching, perm x = x) :=
sorry

end balls_boxes_exactly_3_non_matching_l562_562397


namespace simple_interest_initial_amount_l562_562738

theorem simple_interest_initial_amount :
  ∃ P : ℝ, (P + P * 0.04 * 5 = 900) ∧ P = 750 :=
by
  sorry

end simple_interest_initial_amount_l562_562738


namespace fraction_to_local_crisis_fund_equal_to_3_over_8_l562_562589

-- Definitions
def annual_donation : ℕ := 240
def community_pantry_fraction : ℚ := 1 / 3
def livelihood_fraction : ℚ := 1 / 4
def contingency_fund : ℕ := 30

-- Theorem statement
theorem fraction_to_local_crisis_fund_equal_to_3_over_8 :
  let remaining_after_pantry := annual_donation - (community_pantry_fraction * annual_donation).natAbs
  let livelihood_project_funds := (livelihood_fraction * remaining_after_pantry).natAbs
  let combined_crisis_and_contingency := remaining_after_pantry - livelihood_project_funds
  let local_crisis_fund := combined_crisis_and_contingency - contingency_fund
  local_crisis_fund / annual_donation.toNatRat = 3 / 8 := by
  sorry

end fraction_to_local_crisis_fund_equal_to_3_over_8_l562_562589


namespace river_depth_in_mid_may_l562_562535

variable (D : ℕ)
variable (h1 : D + 10 - 5 + 8 = 45)

theorem river_depth_in_mid_may (h1 : D + 13 = 45) : D = 32 := by
  sorry

end river_depth_in_mid_may_l562_562535


namespace convex_polygon_contains_center_l562_562702

theorem convex_polygon_contains_center
  (P : Set ℝ) (hP_convex : Convex ℝ P)
  (hP_area : area P = 7)
  (C : ℝ)
  (hC_radius : radius C = 2)
  (hP_in_C : P ⊆ C) :
  center C ∈ P :=
by
  sorry

end convex_polygon_contains_center_l562_562702


namespace repeating_decimal_fraction_l562_562265

theorem repeating_decimal_fraction : (0.363636363636 : ℚ) = 4 / 11 := 
sorry

end repeating_decimal_fraction_l562_562265


namespace car_travel_distance_l562_562698

theorem car_travel_distance (distance : ℝ) 
  (speed1 : ℝ := 80) 
  (speed2 : ℝ := 76.59574468085106) 
  (time_difference : ℝ := 2 / 3600) : 
  (distance / speed2 = distance / speed1 + time_difference) → 
  distance = 0.998177 :=
by
  -- assuming the above equation holds, we need to conclude the distance
  sorry

end car_travel_distance_l562_562698


namespace simplify_expression_l562_562194

theorem simplify_expression (x : ℝ) :
  (sin (3 * x) = 3 * sin x - 4 * (sin x) ^ 3) →
  (cos (3 * x) = 4 * (cos x) ^ 3 - 3 * cos x) →
  (1 + sin (3 * x) - cos (3 * x)) / (1 + sin (3 * x) + cos (3 * x)) =
  (1 + 3 * (sin x + cos x) - 4 * (sin x) ^ 3 - 4 * (cos x) ^ 3) / 
  (1 + 3 * (sin x - cos x) - 4 * (sin x) ^ 3 + 4 * (cos x) ^ 3) :=
by
  intros hs hc
  sorry

end simplify_expression_l562_562194


namespace alcohol_percentage_l562_562352

theorem alcohol_percentage (x : ℝ)
  (h1 : 8 * x / 100 + 2 * 12 / 100 = 22.4 * 10 / 100) : x = 25 :=
by
  -- skip the proof
  sorry

end alcohol_percentage_l562_562352


namespace polygon_area_is_nine_l562_562003

-- Definitions of vertices and coordinates.
def vertexA := (0, 0)
def vertexD := (3, 0)
def vertexP := (3, 3)
def vertexM := (0, 3)

-- Area of the polygon formed by the vertices A, D, P, M.
def polygonArea (A D P M : ℕ × ℕ) : ℕ :=
  (D.1 - A.1) * (P.2 - A.2)

-- Statement of the theorem.
theorem polygon_area_is_nine : polygonArea vertexA vertexD vertexP vertexM = 9 := by
  sorry

end polygon_area_is_nine_l562_562003


namespace Ms_Rush_Speed_to_be_on_time_l562_562181

noncomputable def required_speed (d t r : ℝ) :=
  d = 50 * (t + 1/12) ∧ 
  d = 70 * (t - 1/9) →
  r = d / t →
  r = 74

theorem Ms_Rush_Speed_to_be_on_time 
  (d t r : ℝ) 
  (h1 : d = 50 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/9)) 
  (h3 : r = d / t) : 
  r = 74 :=
sorry

end Ms_Rush_Speed_to_be_on_time_l562_562181


namespace factory_output_decrease_l562_562613

theorem factory_output_decrease (original_output : ℝ) 
  (first_increase_percent : ℝ)
  (second_increase_percent : ℝ)
  (final_output : ℝ) : 
  original_output = 100 →
  first_increase_percent = 10 / 100 →
  second_increase_percent = 60 / 100 →
  final_output = original_output * (1 + first_increase_percent) * (1 + second_increase_percent) →
  ((final_output - original_output) / final_output) * 100 ≈ 43.18 :=
by
  intros h_orig h_first h_second h_final
  sorry

end factory_output_decrease_l562_562613


namespace problem_statement_l562_562905

noncomputable def S : Type := sorry
variable [nonempty S]
variable (star : S → S → S)
variable (h_star : ∀ a b : S, star a (star b a) = b)

theorem problem_statement (a b : S) : star (star a b) a ≠ a := sorry

end problem_statement_l562_562905


namespace find_a_when_ab_is_24_l562_562621

theorem find_a_when_ab_is_24 (a b : ℝ) (k : ℝ) (h1 : ∀ a b, a^2 * b.sqrt = k)
  (h2 : a = 3) (h3 : b = 16) (h4 : a * b = 24) : 
  a = 3 * (2)^(1/3) :=
by
  sorry

end find_a_when_ab_is_24_l562_562621


namespace rational_sum_rational_l562_562982

theorem rational_sum_rational {a b : ℚ} (ha : is_rational a) (hb : is_rational b) : is_rational (a + b) :=
sorry

end rational_sum_rational_l562_562982


namespace ellipse_points_intersect_minimum_area_quad_l562_562149

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 9) + (p.2 ^ 2 / 4) = 1}

noncomputable def slope_line_through (x1 y1: ℝ): ℝ := -4 * x1 / (9 * y1)

theorem ellipse_points_intersect
  (x1 y1: ℝ)
  (h : (x1, y1) ∈ ellipse)
  (l := slope_line_through x1 y1)
  (A := (3, 0))
  (A' := (-3, 0))
  (M := (3, (12 - 4 * x1) / (3 * y1)))
  (M' := (-3, (12 + 4 * x1) / (3 * y1)))
  : | 3 / ( (12 - 4 * x1) / y1 ) | * | -3 / ( (12 + 4 * x1) / y1 ) | = 4 :=
sorry

theorem minimum_area_quad
  (x1 y1: ℝ)
  (h : (x1, y1) ∈ ellipse)
  (A := (3, 0))
  (A' := (-3, 0))
  (M := (3, (12 - 4 * x1) / (3 * y1)))
  (M' := (-3, (12 + 4 * x1) / (3 * y1)))
  : ∃ y1_max, y1_max = 2 ∧ (3 * |3| / y1_max) = 12 :=
sorry

end ellipse_points_intersect_minimum_area_quad_l562_562149


namespace marikas_father_twice_her_age_l562_562942

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l562_562942


namespace tangent_line_at_point_l562_562220

noncomputable def f : ℝ → ℝ := λ x, (2 * x - 1) / (x + 2)

def point : ℝ × ℝ := (-1, -3)

def tangent_line_eq (m x₁ y₁ : ℝ) := 
  λ x y, y - y₁ = m * (x - x₁)

theorem tangent_line_at_point :
  let slope_at_point := 5 in
  let x₀ := point.fst in
  let y₀ := point.snd in
  ∀ x y, (tangent_line_eq slope_at_point x₀ y₀ x y) ↔ (5 * x - y + 2 = 0) := 
by {
  sorry
}

end tangent_line_at_point_l562_562220


namespace unique_k_value_l562_562753
noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m ∧ m ∣ n → m = n

theorem unique_k_value :
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 74 ∧ p * q = 213) ∧
  ∀ (p₁ q₁ k₁ p₂ q₂ k₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ p₁ + q₁ = 74 ∧ p₁ * q₁ = k₁ ∧
    is_prime p₂ ∧ is_prime q₂ ∧ p₂ + q₂ = 74 ∧ p₂ * q₂ = k₂ →
    k₁ = k₂ :=
by
  sorry

end unique_k_value_l562_562753


namespace marikas_father_twice_her_age_l562_562943

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l562_562943


namespace sum_first_five_terms_l562_562814

theorem sum_first_five_terms (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (h1 : ∀ n : ℕ, a_n = a_1 + n * 1/2)
    (h2 : a_2, a_6, a_14 are in geometric sequence) : 
    S_5 = 25/2 := by
  sorry

end sum_first_five_terms_l562_562814


namespace find_window_width_on_second_wall_l562_562169

noncomputable def total_wall_area (width length height: ℝ) : ℝ :=
  4 * width * height

noncomputable def doorway_area (width height : ℝ) : ℝ :=
  width * height

noncomputable def window_area (width height : ℝ) : ℝ :=
  width * height

theorem find_window_width_on_second_wall :
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  total_area - first_doorway - second_doorway - window_area w window_height = area_to_paint
  → w = 6 :=
by
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  sorry

end find_window_width_on_second_wall_l562_562169


namespace marikas_father_age_twice_in_2036_l562_562953

theorem marikas_father_age_twice_in_2036 :
  ∃ (x : ℕ), (10 + x = 2006 + x) ∧ (50 + x = 2 * (10 + x)) ∧ (2006 + x = 2036) :=
by
  sorry

end marikas_father_age_twice_in_2036_l562_562953


namespace eggs_needed_per_month_l562_562179

def weekly_eggs_needed : ℕ := 10 + 14 + (14 / 2)

def weeks_in_month : ℕ := 4

def monthly_eggs_needed (weekly_eggs : ℕ) (weeks : ℕ) : ℕ :=
  weekly_eggs * weeks

theorem eggs_needed_per_month : 
  monthly_eggs_needed weekly_eggs_needed weeks_in_month = 124 :=
by {
  -- calculation details go here, but we leave it as sorry
  sorry
}

end eggs_needed_per_month_l562_562179


namespace Marian_credit_card_balance_l562_562929

theorem Marian_credit_card_balance :
  let initial_balance := 126.00 in
  let groceries := 60.00 in
  let gas := groceries / 2 in
  let returned := 45.00 in
  initial_balance + groceries + gas - returned = 171.00 :=
by
  let initial_balance := 126.00
  let groceries := 60.00
  let gas := groceries / 2
  let returned := 45.00
  calc
    126.00 + 60.00 + 30.00 - 45.00 = 216.00 - 45.00 : by congr
    ... = 171.00 : by norm_num

#suppressAllProofSteps

end Marian_credit_card_balance_l562_562929


namespace permutation_product_even_l562_562915

theorem permutation_product_even (a : Fin 7 → ℕ) (h_perm : Multiset.sort (a '' (Finset.univ : Finset (Fin 7))) = [1, 2, 3, 4, 5, 6, 7]) : 
  Even ((a 1 - 1) * (a 2 - 2) * (a 3 - 3) * (a 4 - 4) * (a 5 - 5) * (a 6 - 6) * (a 7 - 7)) :=
by
-- We provide the statement only, no proof here.
sorry

end permutation_product_even_l562_562915


namespace find_ax5_plus_by5_l562_562202

theorem find_ax5_plus_by5 (a b x y : ℝ)
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_plus_by5_l562_562202


namespace geometric_mean_eq_6_l562_562072

theorem geometric_mean_eq_6 (b c : ℝ) (hb : b = 3) (hc : c = 12) :
  (b * c) ^ (1/2 : ℝ) = 6 := 
by
  sorry

end geometric_mean_eq_6_l562_562072


namespace two_hundredth_digit_of_three_seventh_l562_562280

noncomputable def decimal_representation_of_three_seveth := "428571..."

def repeating_sequence_3_7 : string := "428571"

def digit_at_position (s: string) (pos : ℕ) : char :=
s[pos % s.length]

theorem two_hundredth_digit_of_three_seventh :
  digit_at_position repeating_sequence_3_7 200 = '2' :=
by
  sorry

end two_hundredth_digit_of_three_seventh_l562_562280


namespace expression_value_l562_562439

theorem expression_value (a b c : ℝ) (h : a + b + c = 0) : (a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b)) = 3 := 
by 
  sorry

end expression_value_l562_562439


namespace shortest_path_on_cube_surface_l562_562367

theorem shortest_path_on_cube_surface {a : ℝ} (h_a : a > 0) :
  ∃ P : ℝ^3, ∀ P ∈ edge_of_cube(not_vertex) a,
  shortest_path(P, traversing_6_faces_and_back) = 3 * a * real.sqrt 2 :=
sorry

end shortest_path_on_cube_surface_l562_562367


namespace postage_cost_correct_l562_562973

-- Define the cost function based on the given conditions
noncomputable def postageCost (W : ℝ) : ℝ :=
  min (8 * Real.ceil W) 80

-- State the theorem
theorem postage_cost_correct (W : ℝ) : postageCost W = min (8 * Real.ceil W) 80 :=
by
  -- Proof is omitted as per instructions
  sorry

end postage_cost_correct_l562_562973


namespace area_of_inscribed_regular_octagon_l562_562306

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562306


namespace total_protest_days_l562_562140

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end total_protest_days_l562_562140


namespace tangent_line_equation_l562_562225

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l562_562225


namespace acute_angle_not_greater_than_45_l562_562330

theorem acute_angle_not_greater_than_45 (A B C : Type) [triangle A B C] (right_triangle : right_triangle A B C) :
  (acute_angle A B C < 45 ∧ acute_angle B C A > 45) ∨ (acute_angle A B C > 45 ∧ acute_angle B C A < 45) :=
sorry

end acute_angle_not_greater_than_45_l562_562330


namespace quadratic_function_analytical_expression_l562_562802

noncomputable def f (x : ℝ) : ℝ := - (9 / 4) * (x - 1 / 2) ^ 2 + 8

theorem quadratic_function_analytical_expression :
  (f 2 = -1) ∧ (f (-1) = -1) ∧ (∀ x, f x ≤ 8) ∧ 
  (∀ m, m < 3 → 
    (m ≤ -1 / 2 → ∃ min_val, min_val = max (- (9 / 4) * m ^ 2 + (9 / 2) * m + (23 / 4)) (-33 / 4)) ∧
    (-1 / 2 < m → m ≤ 3 → f 3 = -33 / 4)) :=
begin
  sorry
end

end quadratic_function_analytical_expression_l562_562802


namespace ellen_smoothies_total_cups_l562_562419

structure SmoothieIngredients where
  strawberries : ℝ
  yogurt       : ℝ
  orange_juice : ℝ
  honey        : ℝ
  chia_seeds   : ℝ
  spinach      : ℝ

def ounces_to_cups (ounces : ℝ) : ℝ := ounces * 0.125
def tablespoons_to_cups (tablespoons : ℝ) : ℝ := tablespoons * 0.0625

noncomputable def total_cups (ing : SmoothieIngredients) : ℝ :=
  ing.strawberries +
  ing.yogurt +
  ing.orange_juice +
  ounces_to_cups (ing.honey) +
  tablespoons_to_cups (ing.chia_seeds) +
  ing.spinach

theorem ellen_smoothies_total_cups :
  total_cups {
    strawberries := 0.2,
    yogurt := 0.1,
    orange_juice := 0.2,
    honey := 1.0,
    chia_seeds := 2.0,
    spinach := 0.5
  } = 1.25 := by
  sorry

end ellen_smoothies_total_cups_l562_562419


namespace martha_apples_l562_562175

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l562_562175


namespace identify_parrots_l562_562186

-- Definitions of parrots
inductive Parrot
| gosha : Parrot
| kesha : Parrot
| roma : Parrot

open Parrot

-- Properties of each parrot
def always_honest (p : Parrot) : Prop :=
  p = gosha

def always_liar (p : Parrot) : Prop :=
  p = kesha

def sometimes_honest (p : Parrot) : Prop :=
  p = roma

-- Statements given by each parrot
def Gosha_statement : Prop :=
  always_liar kesha

def Kesha_statement : Prop :=
  sometimes_honest kesha

def Roma_statement : Prop :=
  always_honest kesha

-- Final statement to prove the identities
theorem identify_parrots (p : Parrot) :
  Gosha_statement ∧ Kesha_statement ∧ Roma_statement → (always_liar Parrot.kesha ∧ sometimes_honest Parrot.roma) :=
by
  intro h
  exact sorry

end identify_parrots_l562_562186


namespace calculate_x_l562_562850

theorem calculate_x :
  (∑ n in Finset.range 1992, n * (1993 - n)) = 1992 * 996 * 665 :=
by
  sorry

end calculate_x_l562_562850


namespace problem_statement_l562_562063

noncomputable def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def B : Set ℝ := {x | x^2 - 4 * x < 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem problem_statement (m : ℝ) :
    (A ∩ B = {x | 2 < x ∧ x < 4}) ∧
    (¬(A ∪ B) = {x | x ≤ 0 ∨ x > 6}) ∧
    (C m ⊆ B → m ∈ Set.Iic (5/2)) := 
by
  sorry

end problem_statement_l562_562063


namespace base_of_fourth_exponent_l562_562105

theorem base_of_fourth_exponent (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a = 7) (h4 : (18 ^ a) * 9 ^ (3 * a - 1) = 2 ^ 7 * some_number ^ b) : ∃ c : ℕ, c = 3 :=
by
  have h5 : (18 ^ 7) * 9 ^ 20 = 2 ^ 7 * some_number ^ b := by
    rw [h3, mul_sub_right_distrib, pow_add, pow_sub]
    sorry   -- Skip full proof for rewriting
  have h6 : 18 = 2 * 9 := by sorry
  have h7 : 9 = 3 ^ 2 := by sorry

  sorry   -- Proof steps to conclude c = 3.
   
  exact ⟨3, rfl⟩

end base_of_fourth_exponent_l562_562105


namespace tangent_line_equation_at_point_l562_562212

-- Define the function y = f(x) = (2x - 1) / (x + 2)
def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

-- Define the point at which the tangent is evaluated
def point : ℝ × ℝ := (-1, -3)

-- Proof statement for the equation of the tangent line at the given point
theorem tangent_line_equation_at_point : 
  (∃ (a b c : ℝ), a * point.1 + b * point.2 + c = 0 ∧ a = 5 ∧ b = -1 ∧ c = 2) :=
sorry

end tangent_line_equation_at_point_l562_562212


namespace arithmetic_series_sum_l562_562399

theorem arithmetic_series_sum :
  ∑ k in (Finset.range 64).map (⟨λ i, 25 + i * (2/5 : ℝ), sorry⟩ : ℕ → ℝ) = 2400 := 
by
  sorry

end arithmetic_series_sum_l562_562399


namespace product_of_possible_b_div_a2_l562_562595

theorem product_of_possible_b_div_a2 (a : ℕ) (b : ℕ) (h1 : a > 0) (h2 : b = a * (10^(String.length (Nat.toDigits 10 a)) + 1)) (hb : b % (a^2) = 0) :
  ∃ c : ℕ, c = 77 := 
sorry

end product_of_possible_b_div_a2_l562_562595


namespace quadrant_of_product_conjugate_l562_562472

-- Definitions of the conditions
def z1 : ℂ := 1 + complex.i
def z2 : ℂ := 2 * complex.i - 1

-- The proof goal expressing the correct answer
theorem quadrant_of_product_conjugate :
  let product := (conj z1) * z2 in
    (0 < product.re) ∧ (0 < product.im) :=
  by
    sorry

end quadrant_of_product_conjugate_l562_562472


namespace trig_proof_l562_562470

-- Define the condition
def trig_condition (α : ℝ) : Prop :=
  sin (α + π / 6) - cos α = 1 / 3

-- Define the statement to be proved
def trig_statement (α : ℝ) : Prop :=
  cos (2 * α - π / 3) = 7 / 9

-- The final theorem statement combining the condition and the proof goal
theorem trig_proof (α : ℝ) (h : trig_condition α) : trig_statement α :=
by sorry

end trig_proof_l562_562470


namespace necessary_but_not_sufficient_l562_562122

-- Introduce the necessary definitions and hypotheses
variables {α : ℝ}

-- Define the condition
def sin_eq_cos : Prop := sin α = cos α

-- State the necessary but not sufficient condition
theorem necessary_but_not_sufficient (h : sin_eq_cos) : 
  (α = π / 4) → (sin_eq_cos ∧ ¬ (sin_eq_cos → α = π / 4)) :=
sorry

end necessary_but_not_sufficient_l562_562122


namespace jacob_river_water_collection_l562_562889

/-- Definitions: 
1. Capacity of the tank in milliliters
2. Daily water collected from the rain in milliliters
3. Number of days to fill the tank
4. To be proved: Daily water collected from the river in milliliters
-/
def tank_capacity_ml : Int := 50000
def daily_rain_ml : Int := 800
def days_to_fill : Int := 20
def daily_river_ml : Int := 1700

/-- Prove that the amount of water Jacob collects from the river every day equals 1700 milliliters.
-/
theorem jacob_river_water_collection (total_water: Int) 
  (rain_water: Int) (days: Int) (correct_river_water: Int) : 
  total_water = tank_capacity_ml → 
  rain_water = daily_rain_ml → 
  days = days_to_fill → 
  correct_river_water = daily_river_ml → 
  (total_water - rain_water * days) / days = correct_river_water := 
by 
  intros; 
  sorry

end jacob_river_water_collection_l562_562889


namespace angle_CAD_is_24_degrees_l562_562420

-- Definitions related to the conditions
def is_equilateral_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  ∀ a b c : A × B × C, dist a b = dist b c ∧ dist b c = dist c a

def is_regular_pentagon (B C D E G : Type) [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace G] :=
  ∀ b c d e g : B × C × D × E × G, dist b c = dist c d ∧ dist c d = dist d e ∧ dist d e = dist e g ∧
                                    dist e g = dist g b ∧
                                    (∠ b c d = 108° ∧ ∠ c d e = 108° ∧ ∠ d e g = 108° ∧ ∠ e g b = 108° ∧ ∠ g b c = 108°)

-- Theorem that needs to be proved
theorem angle_CAD_is_24_degrees (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h1 : is_equilateral_triangle A B C)
  (h2 : is_regular_pentagon B C D E G) :
  ∠ A C D = 24° :=
  sorry

end angle_CAD_is_24_degrees_l562_562420


namespace range_of_a_l562_562838

noncomputable def quadratic_expression (a x : ℝ) : ℝ :=
  a * x^2 + a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_expression a x > 0) → (0 ≤ a ∧ a < 4) :=
begin
  sorry
end

end range_of_a_l562_562838


namespace expression_simplification_l562_562400

    variable (a : ℝ)

    theorem expression_simplification : a^4 - a^(-4) = (a + (1 / a))^2 - 2 * (a - (1 / a))^2 :=
    sorry
    
end expression_simplification_l562_562400


namespace bn_arithmetic_sequence_an_general_term_sn_sum_bn_sequence_l562_562257

-- Definitions based on provided conditions
def sequence_a : ℕ → ℕ
| 0     := 1
| (n+1) := (n+2) * (sequence_a n) / (n+1) + 2 * (n+1) + 2

def sequence_b (n : ℕ) : ℕ := sequence_a n / n

-- Statements to be proved
theorem bn_arithmetic_sequence : 
  ∀ n : ℕ, sequence_b (n + 1) - sequence_b n = 2 :=
sorry

theorem an_general_term :
  ∀ n : ℕ, sequence_a n = 2 * n^2 - n :=
sorry

theorem sn_sum_bn_sequence :
  ∀ n : ℕ, (finset.range n).sum (λ (k : ℕ), sequence_b (3 * k - 1)) = 3 * n^2 :=
sorry

end bn_arithmetic_sequence_an_general_term_sn_sum_bn_sequence_l562_562257


namespace find_boy_age_l562_562344

def boy_age (x : ℕ) : Prop :=
  let daughter_age := x / 5 in
  let wife_age := 5 * x in
  let countryman_age := 10 * x in
  x + daughter_age + wife_age + countryman_age = 81

theorem find_boy_age : ∃ (x : ℕ), boy_age x ∧ x = 5 :=
by
  use 5
  rw boy_age
  simp
  sorry

end find_boy_age_l562_562344


namespace area_regular_octagon_in_circle_l562_562286

theorem area_regular_octagon_in_circle :
  ∀ (r : ℝ), (π * r ^ 2 = 256 * π) → 
  (∃ A : ℝ, A = 512 * √2) :=
by
  -- assume radius from given circle area
  intro r
  assume h : π * r ^ 2 = 256 * π
  -- the goal is to prove the area of the octagon
  existsi (512 * √2)
  sorry

end area_regular_octagon_in_circle_l562_562286


namespace area_of_inscribed_regular_octagon_l562_562308

theorem area_of_inscribed_regular_octagon (r : ℝ) (h : r = 16) : 
  let A := 8 * (2 * r * sin (22.5 * π / 180))^2 * sqrt 2 / 4
  in A = 341.484 := 
by 
  -- Assume radius is given
  sorry

end area_of_inscribed_regular_octagon_l562_562308


namespace nth_smallest_perfect_square_l562_562201

theorem nth_smallest_perfect_square (n : ℕ) (h : ∃ k : ℕ, n = k^2) : 
  ∃ m : ℕ, m = (n + nat.sqrt n - 1)^2 := 
by 
  -- Proof goes here
  sorry

end nth_smallest_perfect_square_l562_562201


namespace unique_solution_x_l562_562424

theorem unique_solution_x (x : ℝ) :
  (∑ i in (finset.range 21), (x - (2020 - i)) / (i + 1)) = 
  (∑ i in (finset.range 21), (x - (i + 1)) / (2020 - i)) → 
  x = 2021 :=
by {
  sorry
}

end unique_solution_x_l562_562424
