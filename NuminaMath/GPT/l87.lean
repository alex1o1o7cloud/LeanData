import Mathlib

namespace smallest_integer_with_18_divisors_eq_240_l87_87590

theorem smallest_integer_with_18_divisors_eq_240 : 
  ∃ n : ℕ, (0 < n) ∧ (∃ (e : list ℕ), (∀ x ∈ e, x > 0) ∧ ((e.map(λ x => x + 1)).prod = 18) ∧ 
  n = (list.enum_from 2 e.length).zip_with_pow e (λ p e => p ^ e) ∧ 
  (∀ m : ℕ, (0 < m) ∧ (card_divisors m = 18) → n ≤ m)) :=
begin
  sorry
end

noncomputable def smallest_num_with_divisors_18 : ℕ :=
classical.some (smallest_integer_with_18_divisors_eq_240)

example : smallest_num_with_divisors_18 = 240 := 
begin
  sorry
end

end smallest_integer_with_18_divisors_eq_240_l87_87590


namespace zero_sum_of_squares_eq_zero_l87_87796

theorem zero_sum_of_squares_eq_zero {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_eq_zero_l87_87796


namespace tangent_line_y_intercept_l87_87637

noncomputable def y_intercept_of_tangent_line : ℝ :=
let 
  H := (3, 0) : ℝ × ℝ,
  J := (8, 0) : ℝ × ℝ,
  r₁ := 3,
  r₂ := 2 in
  -- The proof of correctness of this result is provided separately 
  sqrt 5

theorem tangent_line_y_intercept :
  let H := (3, 0) : ℝ × ℝ,
      J := (8, 0) : ℝ × ℝ,
      r₁ := 3,
      r₂ := 2 in
  y_intercept_of_tangent_line = sqrt 5 := 
sorry

end tangent_line_y_intercept_l87_87637


namespace min_expression_value_l87_87721

theorem min_expression_value (x : ℝ) : 
  (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1) >= 1/2 := 
sorry

end min_expression_value_l87_87721


namespace allocation_schemes_l87_87344

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l87_87344


namespace line_intersects_parabola_exactly_once_at_m_l87_87547

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l87_87547


namespace polynomial_expansion_l87_87705

variable (t : ℝ)

theorem polynomial_expansion :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-4 * t^3 + 3 * t - 5) = -12 * t^6 - 8 * t^5 + 25 * t^4 - 21 * t^3 - 22 * t^2 + 29 * t - 15 :=
by {
  sorry
}

end polynomial_expansion_l87_87705


namespace triangle_area_inequality_l87_87217

-- Define the triangles and their conditions
variables {ABC A1B1C1 : Type} 
variable [has_area ABC] 
variable [has_area A1B1C1]
variables {A B C A1 B1 C1 : Point}

-- Conditions: acute triangles
axiom acute_triangle (t : Triangle) : Prop
axiom acute_ABC : acute_triangle ABC
axiom acute_A1B1C1 : acute_triangle A1B1C1

-- B1 and C1 lie on BC, and A1 lies inside ABC
axiom B1_on_BC : lies_on B1 (segment B C)
axiom C1_on_BC : lies_on C1 (segment B C)
axiom A1_inside_ABC : lies_inside A1 ABC

-- Areas of triangles
noncomputable def area (t : Triangle) := sorry
noncomputable def S : real := area ABC
noncomputable def S1 : real := area A1B1C1

-- Side lengths
noncomputable def length (a b : Point) := sorry
noncomputable def AB := length A B
noncomputable def AC := length A C
noncomputable def A1B1 := length A1 B1
noncomputable def A1C1 := length A1 C1

-- Theorem to prove
theorem triangle_area_inequality (h1 : acute_triangle ABC) (h2 : lies_on B1 (segment B C)) 
  (h3 : lies_on C1 (segment B C)) (h4 : lies_inside A1 ABC) :
  \(\frac{S}{AB + AC}> \frac{S1}{A1B1 + A1C1}\) :=
  by sorry

end triangle_area_inequality_l87_87217


namespace point_A_in_second_quadrant_l87_87095

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l87_87095


namespace find_expression_intervals_of_monotonicity_max_min_values_l87_87415

theorem find_expression (a b : ℝ) (h1 : 3 * a + 2 * b = 0) (h2 : a + b = 3) :
  ∃ y : ℝ → ℝ, y = (-6:ℝ) * x^3 + (9:ℝ) * x^2 :=
by sorry

theorem intervals_of_monotonicity (y : ℝ → ℝ) (h : y = (-6:ℝ) * x^3 + (9:ℝ) * x^2) :
  (∀ x, (0 < x ∧ x < 1) → deriv y x > 0) ∧
  (∀ x, ((x < 0) ∨ (x > 1)) → deriv y x < 0) :=
by sorry

theorem max_min_values (y : ℝ → ℝ) (h : y = (-6:ℝ) * x^3 + (9:ℝ) * x^2) :
  (∀ x, -2 ≤ x ∧ x ≤ 2 → y(-2) = 84 ∧ y(2) = -12) :=
by sorry

end find_expression_intervals_of_monotonicity_max_min_values_l87_87415


namespace modulus_of_z_is_five_l87_87869

def z : Complex := 3 + 4 * Complex.I

theorem modulus_of_z_is_five : Complex.abs z = 5 := by
  sorry

end modulus_of_z_is_five_l87_87869


namespace zoo_problem_l87_87148

theorem zoo_problem (M B L : ℕ) (h1: 26 ≤ M + B + L) (h2: M + B + L ≤ 32) 
    (h3: M + L > B) (h4: B + L = 2 * M) (h5: M + B = 3 * L + 3) (h6: B = L / 2) : 
    B = 3 :=
by
  sorry

end zoo_problem_l87_87148


namespace point_A_in_second_quadrant_l87_87101

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l87_87101


namespace base3_to_base10_l87_87313

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l87_87313


namespace second_quadrant_necessary_not_sufficient_l87_87618

variable (α : ℝ) -- Assuming α is a real number for generality.

-- Define what it means for an angle to be in the second quadrant (90° < α < 180°).
def in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

-- Define what it means for an angle to be obtuse (90° < α ≤ 180°).
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α ≤ 180

-- State the theorem to prove: 
-- "The angle α is in the second quadrant" is a necessary but not sufficient condition for "α is an obtuse angle".
theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → in_second_quadrant α) ∧ 
  (∃ α, in_second_quadrant α ∧ ¬is_obtuse α) :=
sorry

end second_quadrant_necessary_not_sufficient_l87_87618


namespace plane_angles_comparison_l87_87427

noncomputable def planeAngleSum (P Q R : Point) (A : Point) : Real :=
  let α := angle P Q A
  let β := angle Q R A
  let γ := angle R P A
  α + β + γ

theorem plane_angles_comparison (A B C D A' : Point) (h1 : collinear B C D)
(h2 : inside_pyramid A' A B C D) :
planeAngleSum B C D A' > planeAngleSum B C D A :=
begin
  sorry
end

end plane_angles_comparison_l87_87427


namespace maximum_value_existence_l87_87489

open Real

theorem maximum_value_existence (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
    8 * a + 3 * b + 5 * c ≤ sqrt (373 / 36) := by
  sorry

end maximum_value_existence_l87_87489


namespace log_sqrt_defined_in_interval_l87_87362

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l87_87362


namespace village_population_l87_87266

theorem village_population (P : ℕ) (h : 80 * P = 32000 * 100) : P = 40000 :=
sorry

end village_population_l87_87266


namespace tile_floor_with_polygons_l87_87945

theorem tile_floor_with_polygons (x y z: ℕ) (h1: 3 ≤ x) (h2: 3 ≤ y) (h3: 3 ≤ z) 
  (h_seamless: ((1 - (2 / (x: ℝ))) * 180 + (1 - (2 / (y: ℝ))) * 180 + (1 - (2 / (z: ℝ))) * 180 = 360)) :
  (1 / (x: ℝ) + 1 / (y: ℝ) + 1 / (z: ℝ) = 1 / 2) :=
by
  sorry

end tile_floor_with_polygons_l87_87945


namespace minimum_sum_reciprocals_l87_87494

theorem minimum_sum_reciprocals {b : Fin 15 → ℝ} (h₀ : ∀ i, b i > 0) (h₁ : (Finset.univ.sum b) = 1) :
  (Finset.univ.sum (λ i, 1 / (b i))) ≥ 225 := 
sorry

end minimum_sum_reciprocals_l87_87494


namespace odd_function_solution_set_l87_87014

noncomputable def f (x : ℝ) : ℝ := sorry

theorem odd_function_solution_set :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, continuous_at f x) ∧
  (∀ x : ℝ, f (2 : ℝ) = 0) ∧
  (∀ x : ℝ, x > 1 → f' x < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → f' x > 0) →
  {x : ℝ | x * f x > 0} = (Ioc (-2 : ℝ) (-1 : ℝ)) ∪ (Ioc 0 2) :=
by
  intros h 
  sorry

end odd_function_solution_set_l87_87014


namespace range_of_a_l87_87403

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2) ^ x - 1 else real.sqrt x

theorem range_of_a (a : ℝ) : f a > 1 ↔ a ∈ (Set.Ioi 1 ∪ Set.Iio (-1)) := by
  sorry

end range_of_a_l87_87403


namespace compare_angles_in_isosceles_trapezoid_l87_87458

-- Definitions of the given problem
universe u
variable {Point : Type u}

structure Trapezoid :=
  (A B C D : Point)
  (is_isosceles : ∀P Q R S, P = A → Q = B → R = C → S = D → 
    (∃M N : Point, M ≠ N ∧ (M = A ∧ N = B) ∨ (M = D ∧ N = C)))
  (AD_parallel_BC : ∀P Q, P = A → Q = D → Parallel P Q B C)
  (AD : ℝ)
  (BC : ℝ)
  (height : ℝ)

-- Main proof problem
theorem compare_angles_in_isosceles_trapezoid
  (A B C D : Point)
  (isos_trap : Trapezoid)
  (h_is_isosceles : isos_trap.is_isosceles A B C D)
  (h_parallel : AD_parallel_BC A D)
  (h_AD_length : isos_trap.AD = 12)
  (h_BC_length : isos_trap.BC = 6)
  (h_height : isos_trap.height = 4) :
  ∠BAC > ∠CAD :=
sorry

end compare_angles_in_isosceles_trapezoid_l87_87458


namespace find_symmetric_point_l87_87086

-- Define the coordinates of the point P in the rectangular coordinate system.
def P : ℝ × ℝ × ℝ := (3, 4, 5)

-- Define the function that calculates the symmetric point about the yOz plane.
def symmetric_about_yOz_plane (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-P.1, P.2, P.3)

-- Define the resulting symmetric point of P(3, 4, 5) about the yOz plane.
def symmetric_point : ℝ × ℝ × ℝ := symmetric_about_yOz_plane P

-- State the theorem with required proof.
theorem find_symmetric_point : symmetric_point = (-3, 4, 5) :=
by
  -- Add the proof steps or simply replace this with sorry if proof steps are not provided
  sorry

end find_symmetric_point_l87_87086


namespace find_vector_p_l87_87785

noncomputable def vector_proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  let scale := dot_uv / dot_u
  (scale * u.1, scale * u.2)

theorem find_vector_p :
  ∃ p : ℝ × ℝ,
    vector_proj (5, -2) p = p ∧
    vector_proj (2, 6) p = p ∧
    p = (14 / 73, 214 / 73) :=
by
  sorry

end find_vector_p_l87_87785


namespace S8_is_80_l87_87747

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end S8_is_80_l87_87747


namespace stddev_of_symmetric_distribution_l87_87630

variable {α : Type*}

open_locale big_operators

def is_symmetric_about_mean (X : α → ℝ) (m : ℝ) :=
  ∀ x, X x = 2 * m - X x

def within_one_stddev (X : α → ℝ) (m σ : ℝ) :=
  ∃ p q, p = 0.34 ∧ q = 0.34 ∧
    (∀ x, x ∈ Ioc (m - σ) m → p * (card ({x ∈ univ | X x < m} : finset α)) = 0.34 * card univ) ∧
    (∀ x, x ∈ Icc m (m + σ) → q * (card ({x ∈ univ | X x < m + σ} : finset α)) = 0.34 * card univ)

axiom empirical_rule (X : α → ℝ) (m : ℝ) :
  ∀ σ, within_one_stddev X m σ

noncomputable def find_stddev (X : α → ℝ) (m : ℝ) (h₁ : is_symmetric_about_mean X m) (h₂ : ∃ r, (0.68 : ℝ) * card univ = card ({x ∈ univ | X x ∈ Icc (m - r * (0.68 : ℝ) / 2) (m + r * (0.68 : ℝ) / 2)} : finset α))
(h₃ : ∃ u, (0.84 : ℝ) * card univ = card ({x ∈ univ | X x < m + u} : finset α)) : ℝ :=
  let σ := ∃ r, (0.68 : ℝ) = 2 * r in
  classical.some σ

theorem stddev_of_symmetric_distribution (X : α → ℝ) (m : ℝ) :
  ∀ (h₁ : is_symmetric_about_mean X m) (h₂ : ∃ r, (0.68 : ℝ) * card univ = card ({x ∈ univ | X x ∈ Icc (m - r * (0.68 : ℝ) / 2) (m + r * (0.68 : ℝ) / 2)} : finset α)) (h₃ : ∃ u, (0.84 : ℝ) * card univ = card ({x ∈ univ | X x < m + u} : finset α)), let σ := find_stddev X m h₁ h₂ h₃ in σ = σ :=
by
  intros h₁ h₂ h₃
  have hσ : ∃ σ, (0.68 : ℝ) = 2 * (σ) := empirical_rule X m
  exact classical.some_spec hσ

end stddev_of_symmetric_distribution_l87_87630


namespace solution_set_a_neg5_solution_set_general_l87_87419

theorem solution_set_a_neg5 (x : ℝ) : (-5 * x^2 + 3 * x + 2 > 0) ↔ (-2/5 < x ∧ x < 1) := 
sorry

theorem solution_set_general (a x : ℝ) : 
  (ax^2 + (a + 3) * x + 3 > 0) ↔
  ((0 < a ∧ a < 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 3 ∧ x ≠ -1) ∨ 
   (a > 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 0 ∧ x > -1) ∨ 
   (a < 0 ∧ -1 < x ∧ x < -3/a)) := 
sorry

end solution_set_a_neg5_solution_set_general_l87_87419


namespace functions_with_inverses_l87_87429

-- Definitions for the conditions
def passes_Horizontal_Line_Test_A : Prop := false
def passes_Horizontal_Line_Test_B : Prop := true
def passes_Horizontal_Line_Test_C : Prop := true
def passes_Horizontal_Line_Test_D : Prop := false
def passes_Horizontal_Line_Test_E : Prop := false

-- Proof statement
theorem functions_with_inverses :
  (passes_Horizontal_Line_Test_A = false) ∧
  (passes_Horizontal_Line_Test_B = true) ∧
  (passes_Horizontal_Line_Test_C = true) ∧
  (passes_Horizontal_Line_Test_D = false) ∧
  (passes_Horizontal_Line_Test_E = false) →
  ([B, C] = which_functions_have_inverses) :=
sorry

end functions_with_inverses_l87_87429


namespace area_of_midpoint_quadrilateral_l87_87706

theorem area_of_midpoint_quadrilateral (a b : ℝ) :
  let AB := a; let BE := b;
  is_square ABCD AB ∧ is_square BEFG BE →
  area (midpoints_quadrilateral AB BE) = ( (a + b) / 2 )^2 :=
by sorry

end area_of_midpoint_quadrilateral_l87_87706


namespace find_special_integers_l87_87710

theorem find_special_integers 
  : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ (n = 66 ∨ n = 198 ∨ n = 398 ∨ n = 798) :=
by
  sorry

end find_special_integers_l87_87710


namespace solve_system_l87_87837

theorem solve_system :
  ∀ (x y m : ℝ), 
    (∀ P : ℝ × ℝ, P = (3,1) → 
      (P.2 = -P.1 + 4) ∧ (P.2 = 2 * P.1 + m)) → 
    x = 3 ∧ y = 1 ↔ (x + y - 4 = 0 ∧ 2*x - y + m = 0) :=
by
  intros x y m h
  split
  case mp =>
    intro hxy
    cases hxy
    use hxy_left, hxy_right
    have hP : (3,1) = (3 : ℝ, 1 : ℝ) := rfl
    specialize h (3,1) hP
    cases h with h1 h2
    rw [h1, h2],
    exact ⟨by simp, by linarith⟩
  case mpr =>
    intro hsys
    use 3, 1
    split
    case hp1 =>
      exact (by linarith : 3 + 1 - 4 = 0)
    case hp2 =>
      rw [← h.2 3 1 _ ⟨rfl, rfl⟩],
      simp,
    exact ⟨rfl, rfl⟩

end solve_system_l87_87837


namespace inequality_holds_for_any_xyz_l87_87885

theorem inequality_holds_for_any_xyz (x y z : ℝ) : 
  x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) := 
by 
  sorry

end inequality_holds_for_any_xyz_l87_87885


namespace point_on_graph_l87_87802

theorem point_on_graph {k : ℝ} (h₁ : (2 : ℝ), 4) ∈ (fun (x : ℝ) => k * x - 2) := 
  let k := (4 + 2) / 2 in
  ((1 : ℝ), 1) ∈ (fun (x : ℝ) => (3 : ℝ) * x - 2) :=
sorry

end point_on_graph_l87_87802


namespace sample_from_major_C_l87_87631

theorem sample_from_major_C 
    (students_A students_B students_C students_D : ℕ)
    (total_students_sampled : ℕ)
    (hA : students_A = 150)
    (hB : students_B = 150)
    (hC : students_C = 400)
    (hD : students_D = 300)
    (h_total_sampled : total_students_sampled = 40) :
    (students_C.to_rat / (students_A + students_B + students_C + students_D).to_rat) * total_students_sampled = 16 := 
by
  sorry

end sample_from_major_C_l87_87631


namespace allocation_schemes_l87_87347

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l87_87347


namespace averageSpeed_l87_87545

-- Define the total distance driven by Jane
def totalDistance : ℕ := 200

-- Define the total time duration from 6 a.m. to 11 a.m.
def totalTime : ℕ := 5

-- Theorem stating that the average speed is 40 miles per hour
theorem averageSpeed (h1 : totalDistance = 200) (h2 : totalTime = 5) : totalDistance / totalTime = 40 := 
by
  sorry

end averageSpeed_l87_87545


namespace minimum_time_to_serve_tea_equals_9_l87_87516

def boiling_water_time : Nat := 8
def washing_teapot_time : Nat := 1
def washing_teacups_time : Nat := 2
def fetching_tea_leaves_time : Nat := 2
def brewing_tea_time : Nat := 1

theorem minimum_time_to_serve_tea_equals_9 :
  boiling_water_time + brewing_tea_time = 9 := by
  sorry

end minimum_time_to_serve_tea_equals_9_l87_87516


namespace unique_intersection_l87_87549

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l87_87549


namespace cities_with_fewer_than_200000_residents_l87_87180

def percentage_of_cities_with_fewer_than_50000 : ℕ := 20
def percentage_of_cities_with_50000_to_199999 : ℕ := 65

theorem cities_with_fewer_than_200000_residents :
  percentage_of_cities_with_fewer_than_50000 + percentage_of_cities_with_50000_to_199999 = 85 :=
by
  sorry

end cities_with_fewer_than_200000_residents_l87_87180


namespace smallest_maximal_g_within_100_l87_87487

/-- Sum of divisors function, σ(n) -/
def sigma (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum id

/-- Function g(n) = σ(n) / n -/
def g (n : ℕ) : ℚ :=
  (sigma n) / n

/-- The smallest integer N such that g(N) > g(n) for all n ≠ N, 1 ≤ n ≤ 100 -/
theorem smallest_maximal_g_within_100 : ∃ N : ℕ, N = 6 ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 100 ∧ n ≠ N) → g N > g n := sorry

end smallest_maximal_g_within_100_l87_87487


namespace ellipse_equation_line_equation_both_tangent_lines_l87_87461

theorem ellipse_equation (a b c : ℝ) (h : a > 0) (h1 : b > 0) (h2 : a > b) (h3 : c = 1)
  (h4 : ∀ {x y : ℝ}, (x = 0) → (y = 1) → (x^2 / a^2 + y^2 / b^2 = 1)) : a^2 = 2 ∧ b = 1 :=
begin
  have h5: b^2 = 1, from sorry, -- Prove that b^2 = 1 using h4
  have h6: a^2 = 2, from sorry, -- Prove that a^2 = 2 using h5 and h3
  exact ⟨h6, sqrt_eq_iff_sqr_eq.mpr (or.inl h5)⟩
end

theorem line_equation (k m : ℝ)
  (hk : m = sqrt 2 / 2) (hm : k = sqrt 2) : y = k * x + m :=
begin
  sorry -- Prove that the line equation is as expected
end

theorem both_tangent_lines :
  (y = (sqrt 2 / 2) * x + sqrt 2) ∨ (y = -(sqrt 2 / 2) * x - sqrt 2) :=
begin
  sorry -- Prove that there are two solutions for line l being tangent to both ellips and parabola
end

end ellipse_equation_line_equation_both_tangent_lines_l87_87461


namespace find_angle_QPR_l87_87197

-- Definition of degrees as a type
def degrees := ℝ

-- Defining the given angles
def ∠PQR : degrees := 65
def ∠QRC : degrees := 30

-- The main theorem we aim to prove
theorem find_angle_QPR (P Q R C : Type) [triangle : Triangle P Q R] 
  (H1 : PQR.is_tangent_circle C) 
  (H2 : triangle.∠PQR = 65) 
  (H3 : triangle.∠QRC = 30) : 
  triangle.∠QPR = 55 := 
by 
  sorry

end find_angle_QPR_l87_87197


namespace correct_option_is_B_l87_87960

def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem correct_option_is_B :
  (∃ f : ℝ → ℝ, (f = λ x, real.log (x + 1)) ∧ is_increasing f {x : ℝ | -1 < x}) ∧ 
  ¬ (∃ f : ℝ → ℝ, (f = λ x, real.cos x) ∧ is_increasing f set.univ) ∧
  ¬ (∃ f : ℝ → ℝ, (f = λ x, real.exp (-x)) ∧ is_increasing f set.univ) ∧
  ¬ (∃ f : ℝ → ℝ, (f = λ x, abs (x + 1)) ∧ is_increasing f set.univ) :=
by
  sorry

end correct_option_is_B_l87_87960


namespace probability_sum_less_than_9_is_7_over_9_l87_87575

def dice_rolls : List (ℕ × ℕ) := 
  [ (i, j) | i ← [1, 2, 3, 4, 5, 6], j ← [1, 2, 3, 4, 5, 6] ]

def favorable_outcomes : List (ℕ × ℕ) :=
  dice_rolls.filter (λ p => p.1 + p.2 < 9)

def probability_sum_less_than_9 := 
  favorable_outcomes.length.toRat / dice_rolls.length.toRat

theorem probability_sum_less_than_9_is_7_over_9 : 
  probability_sum_less_than_9 = 7 / 9 :=
by
  sorry

end probability_sum_less_than_9_is_7_over_9_l87_87575


namespace simplify_expression_l87_87606

theorem simplify_expression (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ( ( ( ( 1 + a^(-1 / 2) )^(1 / 6) / ( a^(1 / 2) + 1 )^(-1 / 3) ) -
      ( ( a^(1 / 2) - 1 )^(1 / 3) / ( 1 - a^(-1 / 2) )^(-1 / 6) ) )^(-2) ) *
    (1 / 3 * a^(1 / 12) / ( sqrt(a) + sqrt(a - 1) )) = a ^ (1 / 4) / 6 := sorry

end simplify_expression_l87_87606


namespace average_value_of_set_T_is_55_l87_87533

theorem average_value_of_set_T_is_55
  {T : Finset ℕ}
  (h1 : ¬ T.is_empty)
  (max_removed_avg : (T \ {T.max'}) \ T \ {T.min'}).nonempty → (T.erase T.max').average = 50)
  (max_min_removed_avg : (T \ {T.max'}) \ (T \ {T.max'}).erase T.min').nonempty → (T \ {T.max'}).erase T.min').average = 55)
  (max_returned_avg : (T \ {T.min'}).nonempty → (insert (T.max') (T \ {T.min'})).average = 60)
  (max_min_difference : T.max' - T.min' = 80) :
  T.average = 55 :=
sorry

end average_value_of_set_T_is_55_l87_87533


namespace line_through_center_perpendicular_to_given_line_l87_87182

open Classical

theorem line_through_center_perpendicular_to_given_line :
  ∃ l, ∃ (λ : ℝ), ∀ (x y : ℝ), (x - l)^2 + (y + l)^2 = 2 → 2 * x + y = 0 → x - 2 * y + λ = 0 ∧ λ = -3 :=
by
  sorry

end line_through_center_perpendicular_to_given_line_l87_87182


namespace multiply_find_number_l87_87985

theorem multiply_find_number : ∃ x : ℕ, (72515 * x = 724787425) ∧ x = 10005 :=
by
  use 10005
  split
  . rfl
  . rfl

end multiply_find_number_l87_87985


namespace solve_for_log_div_three_l87_87404

def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * (x^(1/3 : ℝ)) + 4

theorem solve_for_log_div_three (a b : ℝ) (h : f a b (log 3) = 3) :
  f a b (log (1 / 3)) = 5 :=
sorry

end solve_for_log_div_three_l87_87404


namespace tabletop_qualification_l87_87628

theorem tabletop_qualification (length width diagonal : ℕ) :
  length = 60 → width = 32 → diagonal = 68 → (diagonal * diagonal = length * length + width * width) :=
by
  intros
  sorry

end tabletop_qualification_l87_87628


namespace negation_proof_l87_87551

theorem negation_proof :
  (¬ ∀ x : ℝ, x < 0 → 1 - x > Real.exp x) ↔ (∃ x_0 : ℝ, x_0 < 0 ∧ 1 - x_0 ≤ Real.exp x_0) :=
by
  sorry

end negation_proof_l87_87551


namespace xy_squared_in_parallelogram_l87_87460

theorem xy_squared_in_parallelogram (A B C D X Y : Type) [parallelogram A B C D]
  (h1 : side_length A B = 24) (h2 : side_length B C = 24)
  (h3 : side_length C D = 13) (h4 : side_length D A = 13)
  (h5 : angle D = 60) (mid_X : midpoint B C X) (mid_Y : midpoint D A Y) :
  length_squared X Y = 169 / 4 := 
by 
  -- Conditions imported from the problem statement
  sorry -- Proof not included

end xy_squared_in_parallelogram_l87_87460


namespace correct_operation_is_C_l87_87963

/--
Given the following statements:
1. \( a^3 \cdot a^2 = a^6 \)
2. \( (2a^3)^3 = 6a^9 \)
3. \( -6x^5 \div 2x^3 = -3x^2 \)
4. \( (-x-2)(x-2) = x^2 - 4 \)

Prove that the correct statement is \( -6x^5 \div 2x^3 = -3x^2 \) and the other statements are incorrect.
-/
theorem correct_operation_is_C (a x : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧
  ((2 * a^3)^3 ≠ 6 * a^9) ∧
  (-6 * x^5 / (2 * x^3) = -3 * x^2) ∧
  ((-x - 2) * (x - 2) ≠ x^2 - 4) := by
  sorry

end correct_operation_is_C_l87_87963


namespace complex_sequence_sum_bound_l87_87400

open Complex

theorem complex_sequence_sum_bound {z : ℕ → ℂ}
  (h1 : ∥z 1∥ = 1)
  (h2 : ∀ n : ℕ, 4 * (z (n + 1))^2 + 2 * (z n) * (z (n + 1)) + (z n)^2 = 0)
  (m : ℕ) :
  ∥∑ k in Finset.range m, z (k + 1)∥ < 2 * Real.sqrt 3 / 3 := 
sorry

end complex_sequence_sum_bound_l87_87400


namespace total_carriages_l87_87597

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l87_87597


namespace minimum_expense_is_5200_l87_87994

def family : Type := {mom : Type, dad : Type, child1 : Type, child2 : Type}

def duration := 5  -- 5 days
def trips_per_day := 10  -- 10 trips per day

def adult_ticket := 40  -- rubles for one trip
def child_ticket := 20  -- rubles for one trip
def day_pass_one := 350  -- rubles for unlimited day pass for one person
def day_pass_group := 1500  -- rubles for unlimited day pass for a group 
def three_day_pass_one := 900  -- rubles for unlimited three-day pass for one person
def three_day_pass_group := 3500  -- rubles for unlimited 3-day pass for a group

def minimum_family_expense : ℕ :=
  let cost_per_adult := (three_day_pass_one + 2 * day_pass_one) in
  let total_adult_cost := 2 * cost_per_adult in
  let cost_per_child := 5 * trips_per_day * child_ticket in
  let total_child_cost := 2 * cost_per_child in
  total_adult_cost + total_child_cost

theorem minimum_expense_is_5200 : minimum_family_expense = 5200 := by
  sorry


end minimum_expense_is_5200_l87_87994


namespace number_of_valid_pairs_l87_87648

theorem number_of_valid_pairs (a b : ℕ) (cond1 : b > a) (cond2 : ∃ a b, b > a ∧ (2*a*b = 3*(a-4)*(b-4))) : 
    ∃ (pairs : set (ℕ × ℕ)), pairs = {(13, 108), (14, 60), (15, 44)} ∧ pairs.size = 3 :=
begin
  sorry
end

end number_of_valid_pairs_l87_87648


namespace tangent_line_y_intercept_l87_87635

noncomputable def y_intercept_of_tangent_line : ℝ :=
let 
  H := (3, 0) : ℝ × ℝ,
  J := (8, 0) : ℝ × ℝ,
  r₁ := 3,
  r₂ := 2 in
  -- The proof of correctness of this result is provided separately 
  sqrt 5

theorem tangent_line_y_intercept :
  let H := (3, 0) : ℝ × ℝ,
      J := (8, 0) : ℝ × ℝ,
      r₁ := 3,
      r₂ := 2 in
  y_intercept_of_tangent_line = sqrt 5 := 
sorry

end tangent_line_y_intercept_l87_87635


namespace find_smallest_denominator_difference_l87_87861

theorem find_smallest_denominator_difference :
  ∃ (r s : ℕ), 
    r > 0 ∧ s > 0 ∧ 
    (5 : ℚ) / 11 < r / s ∧ r / s < (4 : ℚ) / 9 ∧ 
    ¬ ∃ t : ℕ, t < s ∧ (5 : ℚ) / 11 < r / t ∧ r / t < (4 : ℚ) / 9 ∧ 
    s - r = 11 := 
sorry

end find_smallest_denominator_difference_l87_87861


namespace sufficient_but_not_necessary_l87_87017

-- Definitions of conditions
def p (x : ℝ) : Prop := 1 / (x + 1) > 0
def q (x : ℝ) : Prop := (1/x > 0)

-- Main theorem statement
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
sorry

end sufficient_but_not_necessary_l87_87017


namespace find_value_of_a2010_b2010_l87_87656

theorem find_value_of_a2010_b2010 (a b : ℝ) (h1 : a ≠ 0)
  (h2 : b / a = 0)
  (h3 : {a^2, a, 0} = {a, 1, 0}) :
  a^2010 + b^2010 = 1 := 
sorry

end find_value_of_a2010_b2010_l87_87656


namespace units_digit_sum_l87_87955

theorem units_digit_sum (n1 n2 : ℕ) (h1 : n1 % 10 = 1) (h2 : n2 % 10 = 3) : ((n1^3 + n2^3) % 10) = 8 := 
by
  sorry

end units_digit_sum_l87_87955


namespace female_students_count_l87_87207

theorem female_students_count (m f : ℕ) (h1 : m + f = 8)
  (h2 : nat.choose m 2 * nat.choose f 1 = 30) : f = 2 ∨ f = 3 :=
by sorry

end female_students_count_l87_87207


namespace john_task_completion_time_l87_87109

/-- John can complete a task alone in 18 days given the conditions. -/
theorem john_task_completion_time :
  ∀ (John Jane taskDays : ℝ), 
    Jane = 12 → 
    taskDays = 10.8 → 
    (10.8 - 6) * (1 / 12) + 10.8 * (1 / John) = 1 → 
    John = 18 :=
by
  intros John Jane taskDays hJane hTaskDays hWorkDone
  sorry

end john_task_completion_time_l87_87109


namespace problem1_problem2_l87_87821

-- Definition for the conversion idea
def conversion_idea (n : ℕ) : Prop :=
  (1 : ℝ)/n - 1/(n + 1) = 1 / (n * (n + 1))

/-- Problem 1: Sum of fractions equals 1/3 given the conversion idea -/
theorem problem1 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6)) = (1 / 3) :=
by 
  have h := conversion_idea,
  sorry

/-- Problem 2: Impossible to pour out exactly 1L of water -/
theorem problem2 : 
  ∀ n : ℕ, (1 - 1 / (n + 1)) ≠ 1 :=
by 
  assume n,
  have h := conversion_idea,
  sorry

end problem1_problem2_l87_87821


namespace total_surface_area_l87_87900

-- Define the context and given conditions
variable (r : ℝ) (π : ℝ)
variable (base_area : ℝ) (curved_surface_area : ℝ)

-- Assume that the area of the base (circle) of the hemisphere is given as 225π
def base_of_hemisphere_area (r : ℝ) (π : ℝ) :=
  π * r^2 = 225 * π

-- Derive the radius from the base area
def radius_from_base_area (r : ℝ) :=
  r = Real.sqrt (225)

-- Define the curved surface area of the hemisphere
def curved_surface_area_hemisphere (r : ℝ) (π : ℝ) :=
  curved_surface_area = (1 / 2) * (4 * π * r^2)

-- Provide the final calculation for the total surface area
def total_surface_area_hemisphere (curved_surface_area : ℝ) (base_area : ℝ) :=
  curved_surface_area + base_area = 675 * π

-- Main theorem that combines everything and matches the problem statement
theorem total_surface_area (r : ℝ) (π : ℝ) (base_area : ℝ) (curved_surface_area : ℝ) :
  base_of_hemisphere_area r π →
  radius_from_base_area r →
  curved_surface_area_hemisphere r π →
  total_surface_area_hemisphere curved_surface_area base_area :=
by
  intros h_base_radius h_radius h_curved_area
  sorry

end total_surface_area_l87_87900


namespace exist_at_least_2020_optimal_partitions_l87_87742

namespace PartitionProblem

def optimalPartition (P : Set ℕ) (T U : Set ℕ) : Prop :=
  ∑ T = ∑ U ∧ T.union U = P ∧ T ∩ U = ∅

def atLeast2020OptimalPartitions (P : Set ℕ) : Prop :=
  ∃ partitions : Finset (Set ℕ × Set ℕ),
    partitions.card ≥ 2020 ∧ ∀ (T U) ∈ partitions, optimalPartition P T U

theorem exist_at_least_2020_optimal_partitions :
    ∃ P : Set ℕ, 
        (∀ x ∈ P, x > 0) ∧
        (∃ k : ℕ, ∑ P = 2 * k ∧ (∀ subsetP : Set ℕ, subsetP ⊆ P → ∑ subsetP ≠ k)) ∧ 
        atLeast2020OptimalPartitions P :=
  sorry

end PartitionProblem

end exist_at_least_2020_optimal_partitions_l87_87742


namespace pq_solution_l87_87183

theorem pq_solution :
  ∃ (p q : ℤ), (20 * x ^ 2 - 110 * x - 120 = (5 * x + p) * (4 * x + q))
    ∧ (5 * q + 4 * p = -110) ∧ (p * q = -120)
    ∧ (p + 2 * q = -8) :=
by
  sorry

end pq_solution_l87_87183


namespace hyperbola_asymptote_eccentricity_l87_87536

noncomputable def hyperbola_eccentricity (a b c : ℝ) : ℝ :=
  c / a

theorem hyperbola_asymptote_eccentricity (a b c : ℝ) (h₁ : c^2 = a^2 + b^2) 
    (h₂ : b/a = 1/2 ∨ a/b = 1/2) :
  hyperbola_eccentricity a b c = sqrt 5 ∨ hyperbola_eccentricity a b c = sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_eccentricity_l87_87536


namespace solve_system_of_equations_l87_87530

variable (a x y z : ℝ)

theorem solve_system_of_equations (h1 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
                                  (h2 : x + y + 2 * z = 4 * (a^2 + 1))
                                  (h3 : z^2 - x * y = a^2) :
                                  (x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨
                                  (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1) :=
by
  sorry

end solve_system_of_equations_l87_87530


namespace sum_of_arithmetic_sequence_l87_87591

noncomputable def arithmetic_sequence_sum : ℕ → ℕ → ℕ → ℕ
| a d n := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence :
  arithmetic_sequence_sum 5 7 6 = 156 := 
sorry

end sum_of_arithmetic_sequence_l87_87591


namespace min_expression_value_l87_87720

theorem min_expression_value (x : ℝ) : 
  (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1) >= 1/2 := 
sorry

end min_expression_value_l87_87720


namespace Z_in_third_quadrant_l87_87763

noncomputable def Z : ℂ := -2 * complex.I / (1 + 2 * complex.I)

def quadrant_iii (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem Z_in_third_quadrant : quadrant_iii Z :=
  sorry

end Z_in_third_quadrant_l87_87763


namespace probability_transform_in_S_l87_87652

noncomputable def S : set ℂ := {z : ℂ | let x := z.re, y := z.im in -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1}

noncomputable def T (z : ℂ) : ℂ := 
  let x := z.re in
  let y := z.im in
  (x - y) / 2 + (x + y) / 2 * ℂ.i

theorem probability_transform_in_S : ∀ (z : ℂ), z ∈ S → T(z) ∈ S := by
  sorry

end probability_transform_in_S_l87_87652


namespace projective_transformation_fixed_points_l87_87522

theorem projective_transformation_fixed_points (a b c d : ℝ) (h : a * d - b * c ≠ 0) :
  ∃ (n : ℕ), n ≤ 2 ∧ ∀ x : ℝ, (cx^2 + (d - a)x - b = 0 → (∃ i : Fin n, x = i)) := 
sorry

end projective_transformation_fixed_points_l87_87522


namespace color_naturals_l87_87611

/-- Theorem: It is possible to color the natural numbers using 2009 colors such that no triplet of 
numbers (x, y, z) with x * y = z are colored differently, and each color appears infinitely many times. -/
theorem color_naturals (colors : ℕ) (h_colors : colors = 2009) :
  ∃ (f : ℕ → ℕ), (∀ c, (∃∞ n, f n = c)) ∧
  (∀ x y z, x * y = z → f x = f y ∨ f y = f z ∨ f z = f x) :=
by
  sorry

end color_naturals_l87_87611


namespace exists_radius_rolling_wheel_l87_87612

theorem exists_radius_rolling_wheel (natural_points : Set ℝ) (wheel_marked : ℝ → ℝ → Prop)
  (infinite_nat_points : ∀ n : ℕ, (n : ℝ) ∈ natural_points)
  (mark_point : ∀ r t : ℝ, r ∈ natural_points → wheel_marked r t) :
  ∃ R : ℝ, ∀ angle : ℝ, 0 < angle ∧ angle < 360 →
    ∃ θ : ℝ, 0 ≤ θ ∧ θ < 1 → ∃ t : ℝ, wheel_marked t (θ * angle) :=
begin
  sorry
end

end exists_radius_rolling_wheel_l87_87612


namespace minimum_value_real_l87_87229

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l87_87229


namespace sum_of_reciprocals_of_square_roots_le_l87_87523

theorem sum_of_reciprocals_of_square_roots_le (n : ℕ) (hn : 0 < n) :
  (finset.range n).sum (λ i, 1 / real.sqrt (i + 1)) ≤ 2 * real.sqrt n - 1 :=
sorry

end sum_of_reciprocals_of_square_roots_le_l87_87523


namespace complex_number_in_third_quadrant_l87_87399

noncomputable def z : ℂ := sorry

theorem complex_number_in_third_quadrant :
  (1 + complex.I * real.sqrt 3) * z = 2 - complex.I * real.sqrt 3 →
  (z.re < 0 ∧ z.im < 0) :=
sorry

end complex_number_in_third_quadrant_l87_87399


namespace no_solution_l87_87315

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)))

theorem no_solution : problem_statement :=
by
  intro x
  have h₁ : ¬(85 + x = 3.5 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  have h₂ : ¬(55 + x = 2 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  exact sorry

end no_solution_l87_87315


namespace max_inequality_l87_87569

noncomputable def a : ℕ → ℝ
| 1 := 2
| (n+2) := b (n + 1) + 1 / c (n + 1)

noncomputable def b : ℕ → ℝ
| 1 := 4
| (n+2) := c (n + 1) + 1 / a (n + 1)

noncomputable def c : ℕ → ℝ
| 1 := 5
| (n+2) := a (n + 1) + 1 / b (n + 1)

theorem max_inequality (n : ℕ) (h : n > 0) : 
  max (a n) (max (b n) (c n)) > Real.sqrt (2 * n + 13) := 
sorry

end max_inequality_l87_87569


namespace ellipse_foci_coordinates_l87_87976

-- Define the parameters and conditions
def h := 2
def k := -3
def a := 5
def b := 4

-- Define the parametric equations
def x (θ : ℝ) := h + a * Real.cos θ
def y (θ : ℝ) := k + b * Real.sin θ

-- Define the distance to the foci
def c := Real.sqrt (a^2 - b^2)

theorem ellipse_foci_coordinates :
  (let f1 := (h - c, k),
       f2 := (h + c, k) in
    f1 = (-1, -3) ∧ f2 = (5, -3)) :=
by
  sorry

end ellipse_foci_coordinates_l87_87976


namespace find_X_l87_87622

theorem find_X : ∃ X : ℝ, 3889 + 12.952 - X = 3854.002 ∧ X = 47.95 := 
by
  use 47.95
  split
  · sorry -- This is where the proof would go to show 3889 + 12.952 - 47.95 = 3854.002
  · rfl

end find_X_l87_87622


namespace min_a_value_l87_87926

noncomputable def quad_min_a (a b c : ℚ) : ℚ :=
  if a > 0 ∧ a + b + c ∈ ℤ ∧ b = - (a / 2) ∧ c = (a / 16) - (9 / 8)
  then a else 0

theorem min_a_value : quad_min_a 2/9 (-1 / 9) (-53 / 72) = 2/9 :=
  by
    sorry

end min_a_value_l87_87926


namespace min_value_x_squared_plus_6x_l87_87226

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l87_87226


namespace probability_at_least_9_heads_in_12_flips_l87_87603

theorem probability_at_least_9_heads_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := favorable_outcomes / total_outcomes
  probability = 299 / 4096 := 
by
  sorry

end probability_at_least_9_heads_in_12_flips_l87_87603


namespace pascals_triangle_fifth_in_25_numbers_row_l87_87950

theorem pascals_triangle_fifth_in_25_numbers_row :
  ∀ (n k : ℕ), n = 24 → k = 4 → nat.choose n k = 12650 := 
by
  intros n k h₁ h₂
  rw [h₁, h₂]
  exact sorry

end pascals_triangle_fifth_in_25_numbers_row_l87_87950


namespace weight_difference_eq_1_or_19_l87_87459

variable (m1 m2 x1 x2 : ℝ)

-- Define the conditions that must hold
def condition1 : Prop := m1 = m2 + 1
def condition2 : Prop := (10 * (m2 + 1) - x1) / 9 = (10 * m2 - x2) / 9 + 1

-- The statement we are proving
theorem weight_difference_eq_1_or_19 :
  condition1 m1 m2 ∧ condition2 m1 m2 x1 x2 → (x1 - x2 = 1 ∨ x2 - x1 = 19) :=
by
  sorry

end weight_difference_eq_1_or_19_l87_87459


namespace equal_angles_in_triangle_l87_87840

theorem equal_angles_in_triangle (h_sum : 180 = ∑ i, [60, 60, 60].sum) (h_equal : (∀ i, i ∈ [60, 60, 60] → i = 60)) (x : ℝ) :
    x = 60 :=
by sorry

end equal_angles_in_triangle_l87_87840


namespace minimal_k_exists_l87_87482

open Polynomial

noncomputable def minimal_k (n : ℕ) : ℕ :=
  if h : 0 < n ∧ n % 2 = 0 then 
    let q := n.gcd (2^n)
      in 2^q
  else 
    0

theorem minimal_k_exists (n : ℕ) (h : 0 < n ∧ n % 2 = 0) :
  ∃ (k : ℕ), (∃ f g : Polynomial ℤ, ↑k = f * (X + 1)^n + g * (X^n + 1)) ∧ k = minimal_k n :=
by
  sorry

end minimal_k_exists_l87_87482


namespace local_minimum_interval_l87_87776

-- Definitions of the function and its derivative
def y (x a : ℝ) : ℝ := x^3 - 2 * a * x + a
def y_prime (x a : ℝ) : ℝ := 3 * x^2 - 2 * a

-- The proof problem statement
theorem local_minimum_interval (a : ℝ) : 
  (0 < a ∧ a < 3 / 2) ↔ ∃ (x : ℝ), (0 < x ∧ x < 1) ∧ y_prime x a = 0 :=
sorry

end local_minimum_interval_l87_87776


namespace evaluate_expression_l87_87704

theorem evaluate_expression : 8^3 + 4 * 8^2 + 6 * 8 + 3 = 1000 := by
  sorry

end evaluate_expression_l87_87704


namespace ratio_of_areas_l87_87082

theorem ratio_of_areas {A B C D E X : Point}
    (hABC: Triangle ABC)
    (right_ABC: ∠B = 90◦)
    (AB_eq: dist A B = 12)
    (BC_eq: dist B C = 16)
    (is_midpoint_D: midpoint D A B)
    (is_midpoint_E: midpoint E B C)
    (intersection_X: line AD ∩ line CE = {X}) :
  area(B D X E) / area(A X C) = 1 / 2 := by
  sorry

end ratio_of_areas_l87_87082


namespace circle_equation_tangent_to_line_l87_87909

def circle_center : (ℝ × ℝ) := (3, -1)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The equation of the circle with center at (3, -1) and tangent to the line 3x + 4y = 0 is (x - 3)^2 + (y + 1)^2 = 1 -/
theorem circle_equation_tangent_to_line : 
  ∃ r, ∀ x y: ℝ, ((x - 3)^2 + (y + 1)^2 = r^2) ∧ (∀ (cx cy: ℝ), cx = 3 → cy = -1 → (tangent_line cx cy → r = 1)) :=
by
  sorry

end circle_equation_tangent_to_line_l87_87909


namespace quadratic_solution_a_l87_87320

theorem quadratic_solution_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a + b ≠ 0) (h₃ : a + c ≠ 0) :
    ∃ a, (a = (-(b - c) + Real.sqrt (c^2 - 2*b*c - 3*b^2)) / 2 ∨ a = (-(b - c) - Real.sqrt (c^2 - 2*b*c - 3*b^2)) / 2) ∧
        (c^2 - 2*b*c - 3*b^2 ≥ 0 → a ∈ ℝ) ∧ (c^2 - 2*b*c - 3*b^2 < 0 → a ∈ Complex) :=
begin
  sorry
end

end quadratic_solution_a_l87_87320


namespace books_added_jerry_added_books_l87_87477

-- Define initial conditions
def initial_books : ℕ := 9
def total_books : ℕ := 19

-- Define the problem to prove the number of added books
theorem books_added (initial_books : ℕ) (total_books : ℕ) : (total_books - initial_books = 10) :=
begin
  -- mathematical proof to be added here
  sorry
end

-- Apply the initial conditions to assert the theorem
theorem jerry_added_books : books_added initial_books total_books := by
  apply books_added

end books_added_jerry_added_books_l87_87477


namespace constant_term_correct_l87_87463

noncomputable def constant_term_in_expansion : ℕ :=
  let a := 3 
  in 30

theorem constant_term_correct (a : ℕ) (h : (1 + a) = 4) : constant_term_in_expansion = 30 :=
by
  sorry

end constant_term_correct_l87_87463


namespace impossible_to_empty_pile_l87_87210

theorem impossible_to_empty_pile (a b c : ℕ) (h : a = 1993 ∧ b = 199 ∧ c = 19) : 
  ¬ (∃ x y z : ℕ, (x + y + z = 0) ∧ (x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧ z = a ∨ z = b ∨ z = c)) := 
sorry

end impossible_to_empty_pile_l87_87210


namespace box_capacity_l87_87269

theorem box_capacity (total_brownies full_boxes : ℕ) (h₁ : total_brownies = 349) (h₂ : full_boxes = 49) : total_brownies / full_boxes = 7 :=
by
  rw [h₁, h₂]
  norm_num

end box_capacity_l87_87269


namespace man_traveled_distance_l87_87583

-- Define the parameters and convert speeds from km/h to m/s
def trainA_length : ℝ := 300
def trainA_speed_kmh : ℝ := 76
def trainA_speed : ℝ := trainA_speed_kmh * 1000 / 3600

def trainB_length : ℝ := 200
def trainB_speed_kmh : ℝ := 68
def trainB_speed : ℝ := trainB_speed_kmh * 1000 / 3600

def initial_gap : ℝ := 100

def man_speed_kmh : ℝ := 8
def man_speed : ℝ := man_speed_kmh * 1000 / 3600

-- Define the relative speed and compute time taken for Train A to pass Train B
def relative_speed : ℝ := trainA_speed - trainB_speed
def total_distance_to_cover : ℝ := trainA_length + trainB_length + initial_gap
def time_to_pass : ℝ := total_distance_to_cover / relative_speed

-- Define the problem to prove
theorem man_traveled_distance : 
  let distance_travelled_by_man := man_speed * time_to_pass 
  in distance_travelled_by_man = 600 :=
by 
  sorry

end man_traveled_distance_l87_87583


namespace tedra_tomato_harvest_l87_87174

theorem tedra_tomato_harvest (W T F : ℝ) 
    (h1 : T = W / 2) 
    (h2 : W + T + F = 2000) 
    (h3 : F - 700 = 700) : 
    W = 400 := 
sorry

end tedra_tomato_harvest_l87_87174


namespace find_rabbits_l87_87812

theorem find_rabbits (heads rabbits chickens : ℕ) (h1 : rabbits + chickens = 40) (h2 : 4 * rabbits = 10 * 2 * chickens - 8) : rabbits = 33 :=
by
  -- We skip the proof here
  sorry

end find_rabbits_l87_87812


namespace general_formula_value_of_n_l87_87748

-- Definitions and conditions for part (I)
variable (a : ℕ → ℤ) -- defining a_n as a function from natural numbers to integers

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (a₁ d : ℤ), (a 1 = a₁) ∧ (∀ n, a (n + 1) = a n + d)

-- stating the conditions for part (I)
axiom a3 : a 3 = -1
axiom a6 : a 6 = -7

-- Definition for part (II)
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  match a 1 with
  | a1 := n * a1 + (n * (n - 1)) / 2 * (-2) -- using the derived d=-2
  end

-- stating the condition for part (II)
axiom Sn : sum_first_n_terms a 7 = -21

-- Proving the general formula for the sequence
theorem general_formula : is_arithmetic_sequence a ∧ a n = 5 - 2 * n :=
by
  sorry

-- Proving the value of n given the sum condition
theorem value_of_n : ∃ n, sum_first_n_terms a n = -21 ∧ n = 7 :=
by
  sorry

end general_formula_value_of_n_l87_87748


namespace water_depth_is_12_feet_l87_87290

variable (Ron_height Dean_height Water_depth : ℕ)

-- Given conditions
axiom H1 : Ron_height = 14
axiom H2 : Dean_height = Ron_height - 8
axiom H3 : Water_depth = 2 * Dean_height

-- Prove that the water depth is 12 feet
theorem water_depth_is_12_feet : Water_depth = 12 :=
by
  sorry

end water_depth_is_12_feet_l87_87290


namespace shaded_area_l87_87585

def is_shaded_shape (p : ℕ × ℕ) : Prop :=
  p = (3, 2) ∨ p = (4, 2) ∨ p = (2, 3) ∨ p = (5, 3) ∨ p = (2, 4) ∨ p = (5, 4) ∨ p = (3, 5) ∨ p = (4, 5)

theorem shaded_area : ∑ p in (finset.iic (6, 6)), if is_shaded_shape p then 1 else 0 = 3 := by
  sorry

end shaded_area_l87_87585


namespace max_value_of_ratio_l87_87072

noncomputable def triangle_area (a b c : ℝ) (α : ℝ) : ℝ := 
  0.5 * a * b * real.sin α

theorem max_value_of_ratio (a b c : ℝ) (h : 2 * c = a + real.sqrt 2 * b) :
  ∃ (α : ℝ), (real.angle α = 75 * real.pi / 180) ∧
  (triangle_area a b c α / (a^2 + b^2) = (3 - real.sqrt 3) / 20) :=
sorry

end max_value_of_ratio_l87_87072


namespace shaded_area_after_50_iterations_l87_87939

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

noncomputable def total_shaded_area (initial_area : ℝ) (iterations : ℕ) (ratio : ℝ) : ℝ :=
  let a := initial_area / 4
  (a * (1 - ratio^iterations)) / (1 - ratio)

theorem shaded_area_after_50_iterations :
  let s := 12
  let initial_area := equilateral_triangle_area s
  let ratio := 1 / 4
  total_shaded_area initial_area 50 ratio = 12 * sqrt 3 :=
by
  -- Sorry proof will be provided later.
  sorry

end shaded_area_after_50_iterations_l87_87939


namespace exam_failures_l87_87456

theorem exam_failures : ∀ (total_students : ℕ) (pass_percent : ℝ), total_students = 400 → pass_percent = 0.35 → 
  (total_students * (1 - pass_percent) = 260) :=
by
  intros total_students pass_percent h_total h_pass
  rw [h_total, h_pass]
  simp
  norm_num
  sorry

end exam_failures_l87_87456


namespace nature_reserve_birds_l87_87448

theorem nature_reserve_birds :
  ∀ N : ℕ, 
    (N > 0) →
    let hawks := 0.30 * N in
    let non_hawks := N - hawks in
    let paddyfield_warblers := 0.40 * non_hawks in
    let kingfishers := 0.25 * paddyfield_warblers in
    let non_hawks_paddyfield_warblers_kingfishers := hawks + paddyfield_warblers + kingfishers in
  N - non_hawks_paddyfield_warblers_kingfishers = 0.35 * N :=
by
  intros N hN hawks non_hawks paddyfield_warblers kingfishers non_hawks_paddyfield_warblers_kingfishers,
  sorry

end nature_reserve_birds_l87_87448


namespace unique_triangle_exists_l87_87966

theorem unique_triangle_exists : 
  (¬ (∀ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = 3 → a + b > c)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 30 → ∃ (C : ℝ), C > 0)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 100 → ∃ (C : ℝ), C > 0)) ∧
  (∀ (b c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45 → ∃! (a c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45) :=
by sorry

end unique_triangle_exists_l87_87966


namespace optimal_play_winner_l87_87653

theorem optimal_play_winner :
  ∀ (ribbon : ℕ → ℕ) (n : ℕ),
    (length ribbon = 2011) ∧
    (ribbon 0 = 1) ∧
    (ribbon 2010 = 2) ∧
    (∀ i : ℕ, 1 ≤ i ∧ i < 2010 → ribbon i = 1 ∨ ribbon i = 2) →
    ∃ (winner : String),
      winner = "Vasya" := by
  sorry

end optimal_play_winner_l87_87653


namespace question1_question2_l87_87782

def setA : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def setB (a : ℝ) : Set ℝ := {x : ℝ | abs (x - a) ≤ 1 }

def complementA : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 3}

theorem question1 : A = setA := sorry

theorem question2 (a : ℝ) : setB a ∩ complementA = setB a → a ∈ Set.union (Set.Iic (-2)) (Set.Ioi 4) := sorry

end question1_question2_l87_87782


namespace number_of_ordered_pairs_is_three_l87_87650

-- Define the problem parameters and conditions
variables (a b : ℕ)
variable (b_gt_a : b > a)

-- Define the equation for the areas based on the problem conditions
def area_equation : Prop :=
  a * b = 3 * (a - 4) * (b - 4)

-- Main theorem statement
theorem number_of_ordered_pairs_is_three (h₁ : a > 0) (h₂ : b > 0) (h3: b_gt_a) (h4: area_equation a b) :
  ∃! (n : ℕ), n = 3 :=
begin
  sorry  -- Proof is omitted
end

end number_of_ordered_pairs_is_three_l87_87650


namespace part1_part2_l87_87620

noncomputable def total_seating_arrangements : ℕ := 840
noncomputable def non_adjacent_4_people_arrangements : ℕ := 24
noncomputable def three_empty_adjacent_arrangements : ℕ := 120

theorem part1 : total_seating_arrangements - non_adjacent_4_people_arrangements = 816 := by
  sorry

theorem part2 : total_seating_arrangements - three_empty_adjacent_arrangements = 720 := by
  sorry

end part1_part2_l87_87620


namespace student_council_profit_l87_87930

def boxes : ℕ := 48
def erasers_per_box : ℕ := 24
def price_per_eraser : ℝ := 0.75

theorem student_council_profit :
  boxes * erasers_per_box * price_per_eraser = 864 := 
by
  sorry

end student_council_profit_l87_87930


namespace real_cube_inequality_l87_87120

theorem real_cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end real_cube_inequality_l87_87120


namespace cistern_capacity_l87_87251

def leak_rate (C : ℝ) : ℝ := C / 8
def inlet_rate : ℝ := 6 * 60
def net_emptying_rate (C : ℝ) : ℝ := C / 12

theorem cistern_capacity :
  ∀ (C : ℝ),
    inlet_rate - leak_rate C = net_emptying_rate C →
    C = 1728 :=
begin
  intros C h,
  -- the steps to prove the theorem would go here
  sorry,
end

end cistern_capacity_l87_87251


namespace probability_all_and_at_least_one_pass_l87_87043

-- Define conditions
def pA : ℝ := 0.8
def pB : ℝ := 0.6
def pC : ℝ := 0.5

-- Define the main theorem we aim to prove
theorem probability_all_and_at_least_one_pass :
  (pA * pB * pC = 0.24) ∧ ((1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.96) := by
  sorry

end probability_all_and_at_least_one_pass_l87_87043


namespace probability_sum_less_than_9_is_7_over_9_l87_87576

def dice_rolls : List (ℕ × ℕ) := 
  [ (i, j) | i ← [1, 2, 3, 4, 5, 6], j ← [1, 2, 3, 4, 5, 6] ]

def favorable_outcomes : List (ℕ × ℕ) :=
  dice_rolls.filter (λ p => p.1 + p.2 < 9)

def probability_sum_less_than_9 := 
  favorable_outcomes.length.toRat / dice_rolls.length.toRat

theorem probability_sum_less_than_9_is_7_over_9 : 
  probability_sum_less_than_9 = 7 / 9 :=
by
  sorry

end probability_sum_less_than_9_is_7_over_9_l87_87576


namespace distance_to_black_planet_l87_87935

namespace XiaoFeitian

-- Definitions of speeds and times

def speed_space_xiao := 100000 -- in km/s
def speed_light := 300000 -- in km/s
def time_light_reflect := 100 -- in s

-- Main theorem about the distance to the black planet
theorem distance_to_black_planet : 
  let D := 200000 in
  speed_light * time_light_reflect / 2 = speed_space_xiao * time_light_reflect + D :=
by
  sorry

end XiaoFeitian

end distance_to_black_planet_l87_87935


namespace min_value_of_a_l87_87412

theorem min_value_of_a 
  {f : ℕ → ℝ} 
  (h : ∀ x : ℕ, 0 < x → f x = (x^2 + a * x + 11) / (x + 1)) 
  (ineq : ∀ x : ℕ, 0 < x → f x ≥ 3) : a ≥ -8 / 3 :=
sorry

end min_value_of_a_l87_87412


namespace johns_final_push_time_l87_87846

-- Definitions based on given conditions
def john_initial_distance_behind : ℝ := 14
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.7
def john_final_distance_ahead : ℝ := 2

-- Main statement to prove
theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + john_initial_distance_behind + john_final_distance_ahead ∧ t = 32 :=
by
  -- Set up the equation based on the condition: John needs to cover 14 meters + 2 meters more than Steve's distance
  existsi (32 : ℝ)
  { split
    simp [john_speed, steve_speed, john_initial_distance_behind, john_final_distance_ahead]
    sorry
  }

end johns_final_push_time_l87_87846


namespace allocation_schemes_correct_l87_87341

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l87_87341


namespace initial_forks_l87_87567

variables (forks knives spoons teaspoons : ℕ)
variable (F : ℕ)

-- Conditions as given
def num_knives := F + 9
def num_spoons := 2 * (F + 9)
def num_teaspoons := F / 2
def total_cutlery := (F + 2) + (F + 11) + (2 * (F + 9) + 2) + (F / 2 + 2)

-- Problem statement to prove
theorem initial_forks :
  (total_cutlery = 62) ↔ (F = 6) :=
by {
  sorry
}

end initial_forks_l87_87567


namespace find_line_l87_87542

noncomputable def line_eq_through_point 
(l : ℝ → ℝ) (M : ℝ × ℝ)
(cond_ecc : eccentricity : ℝ)
(cond_focus_dist : ℝ)
(cond_normalized_vector : ℝ) : (ℝ → ℝ) 
| a > b > 0 := sorry

theorem find_line
  (a b x₁ y₁ x₂ y₂ x₀ x k c: ℝ)
  (h_eq_ellipse : a > b > 0) 
  (h_eccentricity : ((sqrt 3) / 2 = sqrt(c / a)))
  (h_distance : ((abs (c + sqrt 6))/(sqrt 2)) = 2 * sqrt 3)
  (h_line_through_M : ∀ x, (∀ y, y = k*x - 1))
  (h_intersection : ((x₁ - x₀, y₁) = -((7/5) * (x₂ - x₀, y₂))))
  (h_discriminant: ∀ y, 5*y^2 + 2*y - 7 > 0):
  y = (λ x, x - 1) ∨ y = (λ x, -x - 1) := sorry

end find_line_l87_87542


namespace base_seven_to_base_ten_l87_87948

theorem base_seven_to_base_ten (n : ℕ) (h : n = 54231) : 
  (1 * 7^0 + 3 * 7^1 + 2 * 7^2 + 4 * 7^3 + 5 * 7^4) = 13497 :=
by
  sorry

end base_seven_to_base_ten_l87_87948


namespace solution_set_for_f_geq_zero_l87_87544

theorem solution_set_for_f_geq_zero (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f3 : f 3 = 0) (h_cond : ∀ x : ℝ, x < 0 → x * (deriv f x) < f x) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -3 < x ∧ x < 0} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_for_f_geq_zero_l87_87544


namespace correct_statements_l87_87243

theorem correct_statements :
  (∀ (E : Event), probability E = 1) ∧
  (∀ (E1 E2 : Event), is_complementary E1 E2 → mutually_exclusive E1 E2) :=
begin
  sorry
end

end correct_statements_l87_87243


namespace range_of_a_l87_87069

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - a > 1 → 2x - 3 > a) → x > a + 1) →
  a ≥ 1 :=
by
  -- placeholder for actual proof
  sorry

end range_of_a_l87_87069


namespace roots_of_polynomial_l87_87326

theorem roots_of_polynomial :
  (∃ (r : List ℤ), r = [1, 3, 4] ∧ 
    (∀ x : ℤ, x ∈ r → x^3 - 8*x^2 + 19*x - 12 = 0)) ∧ 
  (∀ x, x^3 - 8*x^2 + 19*x - 12 = 0 → x ∈ [1, 3, 4]) := 
sorry

end roots_of_polynomial_l87_87326


namespace tangent_line_y_intercept_l87_87636

noncomputable def y_intercept_of_tangent_line : ℝ :=
let 
  H := (3, 0) : ℝ × ℝ,
  J := (8, 0) : ℝ × ℝ,
  r₁ := 3,
  r₂ := 2 in
  -- The proof of correctness of this result is provided separately 
  sqrt 5

theorem tangent_line_y_intercept :
  let H := (3, 0) : ℝ × ℝ,
      J := (8, 0) : ℝ × ℝ,
      r₁ := 3,
      r₂ := 2 in
  y_intercept_of_tangent_line = sqrt 5 := 
sorry

end tangent_line_y_intercept_l87_87636


namespace intersection_solution_l87_87838

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end intersection_solution_l87_87838


namespace find_counterfeit_coin_l87_87005

theorem find_counterfeit_coin (n : ℕ) :
  ∃ (k : ℕ), k ≤ ⌈log 3 n⌉₊ ∧ ∀ (coins : fin n → ℤ) (standard_coin : ℤ),
    (∃ (c : fin n), (∀ (i : fin n), i ≠ c → coins i = standard_coin) ∧ 
                    (coins c ≠ standard_coin)) →
    ∃ (weighings : ℕ), weighings = k ∧
      (∀ group1 group2 : list (fin n),
         weighter (group1 coins) (group2 coins) ∨ 
         weighter (group2 coins) (group1 coins)) :=
sorry

end find_counterfeit_coin_l87_87005


namespace smallest_base_l87_87589

theorem smallest_base (b : ℕ) (n : ℕ) : (n = 512) → (b^3 ≤ n ∧ n < b^4) → ((n / b^3) % b + 1) % 2 = 0 → b = 6 := sorry

end smallest_base_l87_87589


namespace fixed_points_circle_l87_87381

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 8 + (y^2) / 4 = 1

theorem fixed_points_circle (k : ℝ) (hk : k ≠ 0)
    (ellipse : ∀ (x y : ℝ), ellipse_equation x y → True)
    (point_on_ellipse : ellipse 2 (real.sqrt 2))
    (left_focus : (∀ (x : ℝ), x = -2))
    : ∃ P1 P2 : ℝ × ℝ, (P1 = (-2, 0)) ∧ (P2 = (2, 0)) :=
sorry

end fixed_points_circle_l87_87381


namespace linear_approximation_of_f_l87_87386

noncomputable def f (x : ℝ) : ℝ := x - 1/x

def is_k_order_linearly_approximated (f : ℝ → ℝ) (a b k : ℝ) : Prop :=
  ∀ (λ : ℝ), λ ∈ set.Icc 0 1 →
  let x := λ * a + (1 - λ) * b in
  let y_M := f x in
  let y_N := λ * f a + (1 - λ) * f b in
  abs (y_M - y_N) ≤ k

theorem linear_approximation_of_f : 
  is_k_order_linearly_approximated f 1 2 k → k ≥ (3/2 - sqrt 2) :=
sorry

end linear_approximation_of_f_l87_87386


namespace partition_exists_for_all_pos_k_l87_87122

open Finset

noncomputable def partition_sum_eq (k : ℕ) (hk : k > 0) : Prop :=
  ∃ (X Y : Finset ℕ), 
    (X ∪ Y = range (2^(k+1)) ∧ X ∩ Y = ∅) ∧ 
    ∀ (m : ℕ), m ∈ range(k+1) → ∑ x in X, x ^ m = ∑ y in Y, y ^ m

theorem partition_exists_for_all_pos_k : ∀ (k : ℕ), 0 < k → partition_sum_eq k := sorry

end partition_exists_for_all_pos_k_l87_87122


namespace two_digit_combinations_l87_87604

theorem two_digit_combinations (s : Finset ℕ) (h_s : s = {1, 2, 3, 4, 5, 6}) :
  (s.card = 6) -> 
  (∃ n, n = 6 * 5 ∧ n = 30) :=
by {
  intros _,
  use 30,
  split,
  {
    simp,
  },
  {
    refl,
  }
}

end two_digit_combinations_l87_87604


namespace minimum_ticket_cost_l87_87996

theorem minimum_ticket_cost :
  let adult_ticket := 40
  let child_ticket := 20
  let unlimited_day_pass_one := 350
  let unlimited_day_pass_group := 1500
  let unlimited_3day_pass_one := 900
  let unlimited_3day_pass_group := 3500
  (min_cost : ℕ) 
  in min_cost = 5200 :=
by
  let min_cost := 5200
  sorry

end minimum_ticket_cost_l87_87996


namespace line_of_intersection_canonical_form_l87_87615

def canonical_form_of_line (A B : ℝ) (x y z : ℝ) :=
  (x / A) = (y / B) ∧ (y / B) = (z)

theorem line_of_intersection_canonical_form :
  ∀ (x y z : ℝ),
  x + y - 2*z - 2 = 0 →
  x - y + z + 2 = 0 →
  canonical_form_of_line (-1) (-3) x (y - 2) (-2) :=
by
  intros x y z h_eq1 h_eq2
  sorry

end line_of_intersection_canonical_form_l87_87615


namespace neither_sufficient_nor_necessary_condition_l87_87751

theorem neither_sufficient_nor_necessary_condition (A B C : Set) (hA : A.nonempty) (hB : B.nonempty) (hC : C.nonempty) (h : A ∩ B = A ∩ C) : 
  ¬((B ⊆ C) ∧ (C ⊆ B)) :=
sorry

end neither_sufficient_nor_necessary_condition_l87_87751


namespace median_a_sq_correct_sum_of_medians_sq_l87_87971

noncomputable def median_a_sq (a b c : ℝ) := (2 * b^2 + 2 * c^2 - a^2) / 4
noncomputable def median_b_sq (a b c : ℝ) := (2 * a^2 + 2 * c^2 - b^2) / 4
noncomputable def median_c_sq (a b c : ℝ) := (2 * a^2 + 2 * b^2 - c^2) / 4

theorem median_a_sq_correct (a b c : ℝ) : 
  median_a_sq a b c = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

theorem sum_of_medians_sq (a b c : ℝ) :
  median_a_sq a b c + median_b_sq a b c + median_c_sq a b c = 
  3 * (a^2 + b^2 + c^2) / 4 :=
sorry

end median_a_sq_correct_sum_of_medians_sq_l87_87971


namespace limit_fractional_part_sqrt_sequence_l87_87042

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 6 ∧ a 2 = 20 ∧ ∀ n > 1, a n * (a n - 8) = a (n - 1) * a (n + 1) - 12

def fractional_part (x : ℝ) : ℝ := x - floor x

theorem limit_fractional_part_sqrt_sequence :
  (∃ a : ℕ → ℝ, sequence a) →
  lim (λ n, fractional_part (sqrt (a n))) = 1/2 :=
by
  sorry

end limit_fractional_part_sqrt_sequence_l87_87042


namespace parallel_lines_sufficient_but_not_necessary_l87_87500

theorem parallel_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 → (∀ x y : ℝ, a * x + 2 * y - 1 = 0 → ∃ x2 y2 : ℝ, x2 + (a + 1) * y2 + 4 = 0)) ∧
  (∃ x y : ℝ, x + 2 * y + 4 = 0 ∧ ¬a = 1) →
  a = 1 is_sufficient_but_not_necessary (λ x y : ℝ, a * x + 2 * y - 1 = 0)
    (λ x y : ℝ, x + (a + 1) * y + 4 = 0) :=
by
  sorry

end parallel_lines_sufficient_but_not_necessary_l87_87500


namespace range_f_cos_2alpha_minus_2beta_l87_87000

def sqrt3 : ℝ := Real.sqrt 3

def f (x : ℝ) : ℝ := sqrt3 * sin x * cos (x + π / 6) + cos x * sin (x + π / 3) + sqrt3 * cos x^2 - sqrt3 / 2

theorem range_f (x : ℝ) (hx : 0 < x ∧ x < π / 2) : 
  -sqrt3 < f(x) ∧ f(x) ≤ 2 :=
  sorry

variables (α β : ℝ)
  (hα : π / 12 < α ∧ α < π / 3)
  (hβ : -π / 6 < β ∧ β < π / 12)
  (hα_val : f α = 6 / 5)
  (hβ_val : f β = 10 / 13)

theorem cos_2alpha_minus_2beta : 
  cos (2 * α - 2 * β) = -33 / 65 :=
  sorry

end range_f_cos_2alpha_minus_2beta_l87_87000


namespace smallest_angle_of_pentagon_l87_87191

-- Defining the conditions:
def sum_of_pentagon_interior_angles : ℕ := 540

def smallest_angle (ratios : List ℕ) (k : ℝ) : ℝ := ratios.head * k

theorem smallest_angle_of_pentagon
  (ratios : List ℕ)
  (h_ratios_length : ratios.length = 5)
  (h_ratios : ratios = [3, 4, 5, 6, 7])
  (sum_angles_eq : ∑ x in (ratios.map (λ r, r * 21.6)), x = 540) : 
  smallest_angle ratios 21.6 = 64.8 :=
by
  -- We stipulate that we expect proof here
  sorry

end smallest_angle_of_pentagon_l87_87191


namespace cost_of_traveling_roads_is_2600_l87_87280

-- Define the lawn, roads, and the cost parameters
def width_lawn : ℝ := 80
def length_lawn : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 2

-- Area calculations
def area_road_1 : ℝ := road_width * length_lawn
def area_road_2 : ℝ := road_width * width_lawn
def area_intersection : ℝ := road_width * road_width

def total_area_roads : ℝ := area_road_1 + area_road_2 - area_intersection

def total_cost : ℝ := total_area_roads * cost_per_sq_meter

theorem cost_of_traveling_roads_is_2600 :
  total_cost = 2600 :=
by
  sorry

end cost_of_traveling_roads_is_2600_l87_87280


namespace max_donation_extra_pay_rate_l87_87512

noncomputable def maria_ivanovna :: MariaIvanovna := sorry

theorem max_donation (cost_per_hour: ℕ) 
(investment_income: ℕ) 
(working_days: ℕ)
(sleep_hours: ℕ)
(monthly_expenses: ℕ)
(total_daily_hours: ℕ)
(max_donation : (daily_hours_for_lessons * working_days)) : ℕ :=
  let daily_hours_for_lessons := sorry in
  let daily_hours_for_knitting := 2 * daily_hours_for_lessons in
  let daily_hours_for_rest := 15 - 3 * daily_hours_for_lessons in
  let max_donation := (5 - daily_hours_for_lessons) * working_days in
  max_donation

theorem extra_pay_rate (current_rate : ℕ) : ℕ := 
  let extra_hour_rate := 2 * current_rate in
  extra_hour_rate

end max_donation_extra_pay_rate_l87_87512


namespace triangle_min_area_l87_87981

theorem triangle_min_area :
  ∃ (p q : ℤ), (p, q).fst = 3 ∧ (p, q).snd = 3 ∧ 1/2 * |18 * p - 30 * q| = 3 := 
sorry

end triangle_min_area_l87_87981


namespace total_children_on_playground_l87_87617

theorem total_children_on_playground (girls boys : ℕ) (h_girls : girls = 28) (h_boys : boys = 35) : girls + boys = 63 := 
by 
  sorry

end total_children_on_playground_l87_87617


namespace no_distribution_satisfies_conditions_l87_87553

theorem no_distribution_satisfies_conditions :
  ¬ ∃ (bags : fin 11 → list ℕ),
    (∀ i, ∃ n ∈ bags i, n ∈ finset.range 1 51) ∧
    (∀ i, (∏ n in bags i, n) % 9 = 0) :=
sorry

end no_distribution_satisfies_conditions_l87_87553


namespace sequence_a_formula_proof_sum_b_formula_proof_lambda_range_proof_l87_87009

open BigOperators

noncomputable def sequence_a : ℕ → ℕ
| 0       := 3   -- though a_1 is given, the sequence naturally starts from a_0 = 1, hence inserting the base case.
| (n + 1) := 2 * (n + 1) + 1

def condition1 (a : ℕ → ℕ) : Prop :=
a 1 = 3 ∧ ∀ n, (a (n + 1) + a n) / (n + 1) = 8 / (a (n + 1) - a n)

-- For sequence b_n
def sequence_b (a : ℕ → ℕ) (n : ℕ) : ℝ :=
1 / (a n ^ 2 - 1)

-- For sequence c_n
def sequence_c (a : ℕ → ℕ) (n : ℕ) (λ : ℝ) : ℝ :=
3^((a n - 1) / 2) + (-1)^(n - 1) * λ * (a (2^(n-1)) - 1)

-- Proof statements
theorem sequence_a_formula_proof 
  (a : ℕ → ℕ) 
  (h : condition1(a)) : 
  ∀ n, a n = 2 * n + 1 := 
sorry

theorem sum_b_formula_proof 
  (a : ℕ → ℕ) 
  (b : ℕ → ℝ)
  (h_a : ∀ n, a n = 2 * n + 1) 
  (h_b : ∀ n, b n = 1 / (a n ^ 2 - 1)) :
  ∀ n, ∑ i in range (n + 1), b i = n / (4 * (n + 1)) :=
sorry

theorem lambda_range_proof 
  (a : ℕ → ℕ) 
  (c : ℕ → ℝ)
  (λ : ℝ)
  (h_a : ∀ n, a n = 2 * n + 1) 
  (h_c : ∀ n, c n = 3^((a n - 1) / 2) + (-1)^(n - 1) * λ * (a (2^(n-1)) - 1)) :
  (-3 / 2 < λ ∧ λ < 1 ∧ λ ≠ 0) ↔ ∀ n, c (n + 1) > c n :=
sorry

end sequence_a_formula_proof_sum_b_formula_proof_lambda_range_proof_l87_87009


namespace taxable_income_l87_87137

theorem taxable_income (tax_paid : ℚ) (state_tax_rate : ℚ) (months_resident : ℚ) (total_months : ℚ) (T : ℚ) :
  tax_paid = 1275 ∧ state_tax_rate = 0.04 ∧ months_resident = 9 ∧ total_months = 12 → 
  T = 42500 :=
by
  intros h
  sorry

end taxable_income_l87_87137


namespace det_D_eq_125_l87_87116

def D : Matrix (Fin 3) (Fin 3) ℝ := ![![5, 0, 0], ![0, 5, 0], ![0, 0, 5]]

theorem det_D_eq_125 : det D = 125 :=
by
  sorry

end det_D_eq_125_l87_87116


namespace original_puppies_l87_87889

-- Define the conditions
def friend_gave_puppies : ℕ := 4
def total_puppies : ℕ := 12

-- Prove the number of puppies Sandy's dog originally had
theorem original_puppies (friend_gave_puppies total_puppies : ℕ) : 
  (friend_gave_puppies = 4) → (total_puppies = 12) → (total_puppies - friend_gave_puppies = 8) :=
by 
  intros h1 h2
  rw [h1, h2]
  simp
  exact eq.refl 8

end original_puppies_l87_87889


namespace prob_factor_less_than_nine_l87_87222

theorem prob_factor_less_than_nine : 
  (∃ (n : ℕ), n = 72) ∧ (∃ (total_factors : ℕ), total_factors = 12) ∧ 
  (∃ (factors_lt_9 : ℕ), factors_lt_9 = 6) → 
  (6 / 12 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end prob_factor_less_than_nine_l87_87222


namespace select_three_defective_impossible_l87_87666

theorem select_three_defective_impossible :
  ∀ (products : List ℕ), products.length = 10 →
  (products.filter (λ x => x = 1)).length = 2 →
  (∃ subset : List ℕ, subset.length = 3 ∧ ∀ x ∈ subset, x = 1) → false :=
begin
  intros products len_products len_defective h,
  sorry
end

end select_three_defective_impossible_l87_87666


namespace percentage_increase_biographies_l87_87480

variable (B b n : ℝ)
variable (h1 : b = 0.20 * B)
variable (h2 : b + n = 0.32 * (B + n))

theorem percentage_increase_biographies (B b n : ℝ) (h1 : b = 0.20 * B) (h2 : b + n = 0.32 * (B + n)) :
  n / b * 100 = 88.24 := by
  sorry

end percentage_increase_biographies_l87_87480


namespace symmetric_points_sum_l87_87388

theorem symmetric_points_sum (a b : ℝ) (h₁ : M = (a, 3)) (h₂ : N = (4, b)) (h_sym : symmetry_about_x_axis M N) : a + b = 1 := 
by
  -- Definition of symmetry_about_x_axis
  -- Given two points M and N, they are symmetric about the x-axis if their y-coordinates are opposites and their x-coordinates are the same.
  def symmetry_about_x_axis (M N : ℝ × ℝ) : Prop :=
    (M.1 = N.1) ∧ (M.2 = -N.2)
  
  -- Translate the given problem conditions
  -- M(a, 3) implies M = (a, 3)
  -- N(4, b) implies N = (4, b)
  -- Symmetry condition implies (a = 4) ∧ (3 = -b)
  
  have h_x_coord : a = 4 := by
    rw [← h₁, ← h₂] at h_sym
    exact h_sym.1
  
  have h_y_coord : b = -3 := by
    rw [← h₁, ← h₂] at h_sym
    exact eq_neg_of_eq_neg (eq.symm h_sym.2)

  have h_sum : a + b = 4 + (-3) := by
    rw [h_x_coord, h_y_coord]

  exact h_sum
  
  -- sorry to skip the formal proof of each step if necessary
  sorry

end symmetric_points_sum_l87_87388


namespace isosceles_triangle_projections_equal_l87_87581

theorem isosceles_triangle_projections_equal
  (O : Point)
  (triangle1 triangle2 : Triangle)
  (h_iso1 : is_isosceles triangle1 O)
  (h_iso2 : is_isosceles triangle2 O)
  (h_sim : is_similar triangle1 triangle2)
  (M : Point)
  (N : Point)
  (hM : M = midpoint (base triangle1))
  (hN : N = midpoint (base triangle2))
  (k1 k2 : ℝ)
  (h_k1 : k1 = base_triangle1_length / height_triangle1_length)
  (h_k2 : k2 = base_triangle2_length / height_triangle2_length) :
  let proj1 := k1 * (dist O M) * sin (angle OMN)
      proj2 := k2 * (dist O N) * sin (angle ONM)
  in proj1 = proj2 :=
sorry

end isosceles_triangle_projections_equal_l87_87581


namespace parallel_lines_of_parallel_planes_and_intersections_l87_87063

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Conditions
axiom parallel_planes : α ∥ β
axiom intersection_α_γ : α ∩ γ = m
axiom intersection_β_γ : β ∩ γ = n

-- Statement
theorem parallel_lines_of_parallel_planes_and_intersections :
  (α ∥ β) → (α ∩ γ = m) → (β ∩ γ = n) → (m ∥ n) :=
by
  intro hαβ hαγ hβγ
  sorry

end parallel_lines_of_parallel_planes_and_intersections_l87_87063


namespace exists_radius_rolling_wheel_l87_87613

theorem exists_radius_rolling_wheel (natural_points : Set ℝ) (wheel_marked : ℝ → ℝ → Prop)
  (infinite_nat_points : ∀ n : ℕ, (n : ℝ) ∈ natural_points)
  (mark_point : ∀ r t : ℝ, r ∈ natural_points → wheel_marked r t) :
  ∃ R : ℝ, ∀ angle : ℝ, 0 < angle ∧ angle < 360 →
    ∃ θ : ℝ, 0 ≤ θ ∧ θ < 1 → ∃ t : ℝ, wheel_marked t (θ * angle) :=
begin
  sorry
end

end exists_radius_rolling_wheel_l87_87613


namespace total_candies_is_36_l87_87508

-- Defining the conditions
def candies_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" then 2 else 1

def total_candies_per_week : Nat :=
  (candies_per_day "Monday" + candies_per_day "Tuesday"
  + candies_per_day "Wednesday" + candies_per_day "Thursday"
  + candies_per_day "Friday" + candies_per_day "Saturday"
  + candies_per_day "Sunday")

def total_candies_in_weeks (weeks : Nat) : Nat :=
  weeks * total_candies_per_week

-- Stating the theorem
theorem total_candies_is_36 : total_candies_in_weeks 4 = 36 :=
  sorry

end total_candies_is_36_l87_87508


namespace max_value_of_expr_l87_87856

noncomputable def max_expr_value (a b c : ℝ) (h1 : a^3 + b^3 + c^3 = 1) : ℝ := 3 * a * b^2 * Real.sqrt 3 + 3 * a * c^2

theorem max_value_of_expr (a b c : ℝ) (h1 : a^3 + b^3 + c^3 = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) (h4 : c ≥ 0) :
  max_expr_value a b c h1 ≤ Real.cbrt (2 * Real.sqrt 3) + Real.cbrt 2 := sorry

end max_value_of_expr_l87_87856


namespace number_of_sampled_people_in_interval_l87_87659

theorem number_of_sampled_people_in_interval :
  ∃ n : ℕ, 
    let interval := 960 / 32 in
    let a_n : ℕ → ℕ := λ n, 30 * n - 21 in
    let numbers := finset.Icc 1 32 in 
    let valid_nums := numbers.filter (λ n, 1 ≤ a_n n ∧ a_n n ≤ 450) in
    valid_nums.card = 15 :=
begin
  sorry
end

end number_of_sampled_people_in_interval_l87_87659


namespace equation_no_solution_l87_87370

theorem equation_no_solution (k : ℝ) : 
  k = 7 → ¬∃ x : ℝ, x ≠ 4 ∧ x ≠ 8 ∧ (x - 3) / (x - 4) = (x - k) / (x - 8) :=
by
  intro hk
  rw hk
  intro h
  rcases h with ⟨x, h₁, h₂, h₃⟩
  simp at h₃
  sorry

end equation_no_solution_l87_87370


namespace monotoically_decreasing_log_base_half_l87_87772

theorem monotoically_decreasing_log_base_half :
  ∀ (a : ℝ), (∀ x ∈ set.Ici (2 : ℝ), (∀ y ∈ set.Ici (2 : ℝ), x < y → log (0.5) (x^2 - a*x + 3*a) ≥ log (0.5) (y^2 - a*y + 3*a))) ↔ a ∈ set.Icc (-4 : ℝ) (4 : ℝ) :=
sorry

end monotoically_decreasing_log_base_half_l87_87772


namespace min_value_of_y_l87_87414

def y (x : ℝ) : ℝ := 4^x - 6 * 2^x + 8

theorem min_value_of_y :
  ∃ x, y x = -1 ∧ x = Real.log 3 / Real.log 2 :=
by
  sorry

end min_value_of_y_l87_87414


namespace sin_beta_l87_87057

theorem sin_beta (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2)
                           (h₂ : 0 < β ∧ β < π / 2)
                           (h₃ : sin α = 4 / 5)
                           (h₄ : cos (α + β) = 5 / 13) : sin β = 16 / 65 :=
by
  sorry

end sin_beta_l87_87057


namespace log_sqrt_defined_l87_87365

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l87_87365


namespace perfect_square_mod_3_l87_87153

theorem perfect_square_mod_3 (k : ℤ) (hk : ∃ m : ℤ, k = m^2) : k % 3 = 0 ∨ k % 3 = 1 :=
by
  sorry

end perfect_square_mod_3_l87_87153


namespace perpendicular_through_AB_intersect_CD_l87_87420

noncomputable def sqrt3 : ℝ := real.sqrt 3

noncomputable def line_l (m : ℝ) : ℝ → ℝ → Prop := λ x y, m * x + y + 3 * m - sqrt3 = 0
noncomputable def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 12

theorem perpendicular_through_AB_intersect_CD (m : ℝ) (A B C D : ℝ × ℝ) :
  line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  circle A.1 A.2 ∧ circle B.1 B.2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * sqrt3)^2 →
  ∃ (CD_dist : ℝ), CD_dist = 4 :=
by
  sorry

end perpendicular_through_AB_intersect_CD_l87_87420


namespace S_15_eq_1695_l87_87426

open Nat

/-- Sum of the nth set described in the problem -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  (n * (first + last)) / 2

theorem S_15_eq_1695 : S 15 = 1695 :=
by
  sorry

end S_15_eq_1695_l87_87426


namespace find_x_plus_y_l87_87056

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 7 * y = 17) (h2 : 3 * x + 5 * y = 11) : x + y = 83 / 23 :=
sorry

end find_x_plus_y_l87_87056


namespace base5_to_base10_conversion_l87_87947

theorem base5_to_base10_conversion : 
  let n := 23412 in 
  let base := 5 in
  n.toNat base = 1732 := 
by 
  let n := 23412
  let base := 5
  sorry

end base5_to_base10_conversion_l87_87947


namespace count_sin_eq_neg_one_half_l87_87052

theorem count_sin_eq_neg_one_half :
  {x : ℝ | 0 ≤ x ∧ x < 360 ∧ Real.sin (x * π / 180) = -1/2}.finite.card = 2 :=
by
  sorry

end count_sin_eq_neg_one_half_l87_87052


namespace arithmetic_sequence_problem_l87_87818

open_locale big_operators

variables (a_1 a_2 a_3 a_4 a_5 a_9 d : ℝ)

theorem arithmetic_sequence_problem
  (h1 : a_1 + a_3 = 8)
  (h2 : a_4 ^ 2 = a_2 * a_9)
  (h3 : d ≠ 0)
  (h4: a_3 = a_1 + 2 * d)
  (h5: a_4 = a_1 + 3 * d)
  (h6: a_2 = a_1 + d)
  (h7: a_9 = a_1 + 8 * d)
  (h8: a_5 = a_1 + 4 * d) : 
  a_5 = 13 :=
sorry

end arithmetic_sequence_problem_l87_87818


namespace unique_zero_assignment_l87_87727

def vertices_sum_to_zero (n : ℕ) (h : n ≥ 2) (assign : Fin ((n+1)*(n+2)/2) → ℤ) : Prop :=
  ∀ (v1 v2 v3 : Fin ((n+1)*(n+2)/2)), 
    -- the condition that {v1, v2, v3} forms a triangle with edges parallel to the sides of the larger triangle
    -- would need to be specified explicitly here
    -- (assuming there would be a predicate to check if three vertices form such a triangle)
    triangle_form (v1, v2, v3) →
    assign v1 + assign v2 + assign v3 = 0

theorem unique_zero_assignment (n : ℕ) (h : n ≥ 2) (assign : Fin ((n+1)*(n+2)/2) → ℤ) :
  vertices_sum_to_zero n h assign →
  (∀ v : Fin ((n+1)*(n+2)/2), assign v = 0) :=
sorry

end unique_zero_assignment_l87_87727


namespace base7_subtraction_correct_l87_87302

-- Define a function converting base 7 number to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

-- Define the numbers in base 7
def a : Nat := 2456
def b : Nat := 1234

-- Define the expected result in base 7
def result_base7 : Nat := 1222

-- State the theorem: The difference of a and b in base 7 should equal result_base7
theorem base7_subtraction_correct :
  let diff_base10 := (base7_to_base10 a) - (base7_to_base10 b)
  let result_base10 := base7_to_base10 result_base7
  diff_base10 = result_base10 :=
by
  sorry

end base7_subtraction_correct_l87_87302


namespace minimum_strawberries_required_l87_87216

def Grid (n : ℕ) := { m : ℕ // m < n * n }

def has_strawberry (g : Grid 10 → Prop) := ∀ (x1 y1 x2 y2 : ℕ), 
  x1 < x2 → y1 < y2 → x2 < 10 → y2 < 10 →
  ∃ (i j : ℕ), x1 ≤ i ∧ i < x2 ∧ y1 ≤ j ∧ j < y2 ∧ g ⟨i * 10 + j, sorry⟩

theorem minimum_strawberries_required : ∃ g : Grid 10 → Prop, 
  (∀ (i : ℕ) (h : i < 100), g ⟨i, h⟩ → (i % 2 = 0)) ∧ 
  has_strawberry g ∧ 
  (∃ G' : Grid 10 → Prop, (∀ (i : ℕ) (h : i < 100), G' ⟨i, h⟩ → g ⟨i, h⟩) ∧ 
  has_strawberry G' ∧ (∑ i : ℕ in finset.range 100, if g ⟨i, sorry⟩ then 1 else 0 = 50)) := 
sorry

end minimum_strawberries_required_l87_87216


namespace intersection_M_N_l87_87425

def M : set ℝ := {x : ℝ | (x - 1)^2 < 4}
def N : set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l87_87425


namespace speed_of_second_train_l87_87219

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

end speed_of_second_train_l87_87219


namespace linda_age_l87_87538

theorem linda_age
  (j k l : ℕ)       -- Ages of Jane, Kevin, and Linda respectively
  (h1 : j + k + l = 36)    -- Condition 1: j + k + l = 36
  (h2 : l - 3 = j)         -- Condition 2: l - 3 = j
  (h3 : k + 4 = (1 / 2 : ℝ) * (l + 4))  -- Condition 3: k + 4 = 1/2 * (l + 4)
  : l = 16 := 
sorry

end linda_age_l87_87538


namespace find_foci_l87_87979

noncomputable def foci_of_ellipse : Prop := 
  let h := 2
  let k := -3
  let a := 5
  let b := 4
  let c := Real.sqrt (a^2 - b^2)
  (h - c = -1) ∧ (h + c = 5) ∧ (k = -3)

theorem find_foci : foci_of_ellipse := by
  let h := 2
  let k := -3
  let a := 5
  let b := 4
  let c := Real.sqrt (a^2 - b^2)
  have hc : c = 3 := by
    calc
      c = Real.sqrt (a^2 - b^2) : rfl
      ... = Real.sqrt (25 - 16) : by
        simp [a, b]
      ... = Real.sqrt 9 : by
        simp
      ... = 3 : Real.sqrt_eq_rfl _,
  simp [h, k, hc],
  exact And.intro (by calc
    h - c = 2 - 3 : by simp [h, hc]
         ... = -1 : by simp) (by
    calc
      h + c = 2 + 3 : by simp [h, hc]
            ... = 5 : by simp)

end find_foci_l87_87979


namespace count_4_edge_trips_l87_87914

noncomputable def number_of_4_edge_trips (X Y : Point) (prism : PentagonalPrism) : ℕ :=
  -- Here you would describe the problem setup in terms of the prism's structure.

theorem count_4_edge_trips (X Y : Point) (prism : PentagonalPrism)
  (shortest_trip_len : shortest_trip_length X Y prism = 4)
  (leads_closer : ∀ (p1 p2 : Point), is_adjacent_by_edge p1 p2 prism → is_closer_to p2 Y p1 prism) :
  number_of_4_edge_trips X Y prism = 54 := 
sorry

end count_4_edge_trips_l87_87914


namespace limit_seq_converges_to_one_l87_87677

noncomputable def limit_seq (n : ℕ) : ℝ := 
  (real.sqrt (n^4 + 2) + real.sqrt (n - 2)) / (real.sqrt (n^4 + 2) + real.sqrt (n - 2))

theorem limit_seq_converges_to_one : 
  tendsto (λ n : ℕ, limit_seq n) at_top (𝓝 (1 : ℝ)) :=
begin
  sorry
end

end limit_seq_converges_to_one_l87_87677


namespace inequality_solution_set_parameter_range_l87_87416

def f (m x : ℝ) := m * x^2 - m * x - 1

theorem inequality_solution_set (m x : ℝ) :
  (f m x > 1 - 2 * x) ↔
  (m = 0 → x > 1) ∧
  (m > 0 → x > 1 ∨ x < -2 / m) ∧
  (-2 < m ∧ m < 0 → 1 < x ∧ x < -2 / m) ∧
  (m < -2 → -2 / m < x ∧ x < 1) ∧
  (m = -2 → false) := sorry

theorem parameter_range (m x : ℝ) :
  (∀ x ∈ set.Icc 1 3, f m x < -m + 4) →
  m < 1 / 7 := sorry

end inequality_solution_set_parameter_range_l87_87416


namespace ring_toss_earning_per_day_l87_87195

theorem ring_toss_earning_per_day :
  ∀ (total_earnings days_per_week earnings_per_day : ℕ),
  total_earnings = 165 → days_per_week = 5 → earnings_per_day = total_earnings / days_per_week → earnings_per_day = 33 :=
by
  intros total_earnings days_per_week earnings_per_day total_eq earning_day_eq per_day_eq
  have eq : total_earnings / days_per_week = 33 := by 
    norm_num 
  rw[total_eq, earning_day_eq] at eq 
  assumption

end ring_toss_earning_per_day_l87_87195


namespace ratio_of_combined_ages_l87_87111

noncomputable def josh_age_at_marriage : ℕ := 22
noncomputable def anna_age_at_marriage : ℕ := 28
noncomputable def years_married : ℕ := 30

theorem ratio_of_combined_ages (josh_age_at_marriage == 22) 
  (anna_age_at_marriage == 28) 
  (years_married == 30)
  (combined_age : ℕ)
  (h_combined_age : combined_age = (josh_age_at_marriage + years_married) + (anna_age_at_marriage + years_married))
  (h_multiple : combined_age % josh_age_at_marriage = 0) :
  combined_age / josh_age_at_marriage = 5 :=
by
  -- sorry to skip the proof steps
  sorry

end ratio_of_combined_ages_l87_87111


namespace total_carriages_proof_l87_87598

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l87_87598


namespace product_is_correct_l87_87608

-- Define the numbers a and b
def a : ℕ := 72519
def b : ℕ := 9999

-- Theorem statement that proves the correctness of the product
theorem product_is_correct : a * b = 725117481 :=
by
  sorry

end product_is_correct_l87_87608


namespace quotient_is_36_l87_87076

-- Conditions
def divisor := 85
def remainder := 26
def dividend := 3086

-- The Question and Answer (proof required)
theorem quotient_is_36 (quotient : ℕ) (h : dividend = (divisor * quotient) + remainder) : quotient = 36 := by 
  sorry

end quotient_is_36_l87_87076


namespace common_rational_root_half_l87_87311

noncomputable def commonRationalRoot (a b c d e f g : ℚ) : ℚ := sorry

theorem common_rational_root_half (a b c d e f g : ℚ) :
  (120 * (1/2)^4 + a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + 18 = 0) ∧
  (18 * (1/2)^5 + d * (1/2)^4 + e * (1/2)^3 + f * (1/2)^2 + g * (1/2) + 120 = 0) :=
begin
  sorry
end

end common_rational_root_half_l87_87311


namespace find_hourly_wage_l87_87431

-- Definitions based on conditions
def hours_worked : ℕ := 8
def rides_given : ℕ := 3
def gas_gallons : ℕ := 17
def gas_price_per_gallon : ℕ := 3
def good_reviews : ℕ := 2
def total_owed : ℕ := 226
def reimbursement_rate_per_gallon : ℕ := gas_price_per_gallon -- $3 per gallon
def payment_per_ride : ℕ := 5
def bonus_per_review : ℕ := 20

-- The statement to prove
theorem find_hourly_wage : 
  let gas_reimbursement := gas_gallons * reimbursement_rate_per_gallon,
      ride_payment := rides_given * payment_per_ride,
      review_bonus := good_reviews * bonus_per_review
  in (total_owed - (gas_reimbursement + ride_payment + review_bonus)) / hours_worked = 15 := 
by
  -- Placeholder for the proof
  sorry

end find_hourly_wage_l87_87431


namespace intersection_complement_l87_87868

def A : set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : set ℝ := {x | x < 1}
def C_R_B : set ℝ := {x | x ≥ 1}
def Intersection : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  (A ∩ C_R_B) = Intersection :=
by
  sorry

end intersection_complement_l87_87868


namespace allocation_schemes_l87_87339

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l87_87339


namespace pyramid_relation_l87_87078

variables {Point : Type} [metric_space Point]
variables (S A B C D M N : Point)

-- Defining the squared distance alias
def dist_sq (p1 p2 : Point) : ℝ := dist p1 p2 ^ 2

-- Assuming we have the distance function and the orthogonality check
variables (SM SN SA : ℝ)

-- Condition: S, A, B, C, D form a regular quadrilateral pyramid
axiom regular_quadrilateral_pyramid : dist_sq S A = dist_sq A B

-- Condition: height \( SA \) and side length are equal
axiom height_side_equal : dist S A = dist A B

-- Condition: M and N are on edges SD and SB respectively such that AM ⊥ CN
axiom perpendicular_AM_CN : (vector_point_sub M A) ⬝ (vector_point_sub N C) = 0

-- The goal to prove
theorem pyramid_relation :
  2 * SA * (SM + SN) = SA ^ 2 + SM * SN := by
  sorry

end pyramid_relation_l87_87078


namespace area_of_shaded_sector_l87_87534

def side_length : ℝ := 6
def radius : ℝ := 6
def interior_angle : ℝ := 120
def sector_angle : ℝ := 120
def area_circle : ℝ := Float.pi * radius * radius
def area_sector : ℝ := (sector_angle / 360) * area_circle 

theorem area_of_shaded_sector : area_sector = 12 * Float.pi := by
  sorry

end area_of_shaded_sector_l87_87534


namespace problem1_part1_problem1_part2_problem2_l87_87173

-- Definitions
def quadratic (a b c x : ℝ) := a * x ^ 2 + b * x + c
def has_two_real_roots (a b c : ℝ) := b ^ 2 - 4 * a * c ≥ 0 
def neighboring_root_equation (a b c : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧ |x₁ - x₂| = 1

-- Proof problem 1: Prove whether x^2 + x - 6 = 0 is a neighboring root equation
theorem problem1_part1 : ¬ neighboring_root_equation 1 1 (-6) := 
sorry

-- Proof problem 2: Prove whether 2x^2 - 2√5x + 2 = 0 is a neighboring root equation
theorem problem1_part2 : neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 := 
sorry

-- Proof problem 3: Prove that m = -1 or m = -3 for x^2 - (m-2)x - 2m = 0 to be a neighboring root equation
theorem problem2 (m : ℝ) (h : neighboring_root_equation 1 (-(m-2)) (-2*m)) : 
  m = -1 ∨ m = -3 := 
sorry

end problem1_part1_problem1_part2_problem2_l87_87173


namespace operation_preserves_remainder_l87_87642

theorem operation_preserves_remainder (N : ℤ) (k : ℤ) (m : ℤ) 
(f : ℤ → ℤ) (hN : N = 6 * k + 3) (hf : f N = 6 * m + 3) : f N % 6 = 3 :=
by
  sorry

end operation_preserves_remainder_l87_87642


namespace problem1_problem2_problem3_l87_87826

-- Definition of "short distance"
def short_distance (x y : ℝ) : ℝ :=
  min (abs x) (abs y)

-- Equidistant points definition
def equidistant (P Q : ℝ × ℝ) : Prop :=
  short_distance P.1 P.2 = short_distance Q.1 Q.2

-- Problem 1: Prove that the "short distance" of point A(-5, -2) is 2.
theorem problem1 : short_distance (-5) (-2) = 2 :=
  sorry

-- Problem 2: Prove that if the "short distance" of point B(-2, -2m+1) is 1, then m is 1 or 0.
theorem problem2 (m : ℝ) (h : short_distance (-2) (-2 * m + 1) = 1) : m = 1 ∨ m = 0 :=
  sorry

-- Problem 3: Prove that if points C(-1, k+3) and D(4, 2k-3) are equidistant points, then k is 1 or 2.
theorem problem3 (k : ℝ) (h : equidistant (-1, k + 3) (4, 2 * k - 3)) : k = 1 ∨ k = 2 :=
  sorry

end problem1_problem2_problem3_l87_87826


namespace impossible_transformation_l87_87375

/-- Define the allowed transformation operation on two numbers. -/
def transform (a b : ℝ) : ℝ × ℝ :=
  ((a + b) / Real.sqrt 2, (a - b) / Real.sqrt 2)

/-- Define the function to calculate the sum of squares of a triplet. -/
def sum_of_squares (a b c : ℝ) : ℝ :=
  a^2 + b^2 + c^2

/-- The initial triplet. -/
def initial_triplet : ℝ × ℝ × ℝ :=
  (2, Real.sqrt 2, 1 / Real.sqrt 2)

/-- The target triplet. -/
def target_triplet : ℝ × ℝ × ℝ :=
  (1, Real.sqrt 2, 1 + Real.sqrt 2)

/-- Prove that it is impossible to transform the initial triplet to the target triplet using the allowed operations. -/
theorem impossible_transformation :
  ¬ ∃ (a b c : ℝ), 
    a = initial_triplet.1 ∧ 
    b = initial_triplet.2 ∧
    c = initial_triplet.3 ∧
    sum_of_squares a b c = sum_of_squares target_triplet.1 target_triplet.2 target_triplet.3 :=
by
  sorry

end impossible_transformation_l87_87375


namespace fertilizer_needed_l87_87166

def p_flats := 4
def p_per_flat := 8
def p_ounces := 8

def r_flats := 3
def r_per_flat := 6
def r_ounces := 3

def s_flats := 5
def s_per_flat := 10
def s_ounces := 6

def o_flats := 2
def o_per_flat := 4
def o_ounces := 4

def vf_quantity := 2
def vf_ounces := 2

def total_fertilizer : ℕ := 
  p_flats * p_per_flat * p_ounces +
  r_flats * r_per_flat * r_ounces +
  s_flats * s_per_flat * s_ounces +
  o_flats * o_per_flat * o_ounces +
  vf_quantity * vf_ounces

theorem fertilizer_needed : total_fertilizer = 646 := by
  -- proof goes here
  sorry

end fertilizer_needed_l87_87166


namespace integrating_factor_iff_condition_l87_87004

noncomputable def integrating_factor_condition
  (f g : ℝ × ℝ → ℝ)
  (h : ℝ → ℝ)
  (cont_partial_derivs : ∀ x y, Continuous (f (x, y)) ∧ Continuous (g (x, y)))
  (diff_eq : ∀ x y, 
    (λ xy, h(xy) * f(x, xy / x)) (x * y) * x = 
    (λ xy, h(xy) * g(x, xy / x)) (x * y) * y) : 
  Prop :=
  ∀ x y, 
    (g.partial_deriv 1 (x, y)) / (y * g(x, y) - x * f(x, y)) - 
    (f.partial_deriv 2 (x, y)) / (y * g(x, y) - x * f(x, y)) 
    = 
    (x * f(x, y) - y * g(x, y)) / h(x * y)

theorem integrating_factor_iff_condition
  (f g : ℝ × ℝ → ℝ)
  (cont_partial_derivs : ∀ x y, Continuous (f (x, y)) ∧ Continuous (g (x, y))) :
  (∃ h : ℝ → ℝ, ∀ x y, 
    (λ xy, h(xy) * f(x, xy / x)) (x * y) * x = 
    (λ xy, h(xy) * g(x, xy / x)) (x * y) * y) ↔ 
  (∃ k : ℝ → ℝ, ∀ x y,
    (g.partial_deriv 1 (x, y) - f.partial_deriv 2 (x, y)) / (y * g(x, y) - x * f(x, y)) 
    = k(x * y)) := 
sorry

end integrating_factor_iff_condition_l87_87004


namespace allocation_schemes_correct_l87_87340

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l87_87340


namespace katie_spending_l87_87259

theorem katie_spending :
  let price_per_flower : ℕ := 6
  let number_of_roses : ℕ := 5
  let number_of_daisies : ℕ := 5
  let total_number_of_flowers := number_of_roses + number_of_daisies
  let total_spending := total_number_of_flowers * price_per_flower
  total_spending = 60 :=
by
  sorry

end katie_spending_l87_87259


namespace tangent_line_at_0_range_of_f_on_interval_l87_87032

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_line_at_0 : 
  (∀ x y : ℝ, x - y + 1 = 0 ↔ y = x + 1) → 
  (∀ x : ℝ, (x = 0) → (f x = 1)) → 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by {
  intros H1 H2,
  have m := f' 0, -- where f' is the derivative of f,
  have tangent := (x - y + 1 = 0 ↔ y = x + 1), -- using the fact that y = mx + b where m = 1 and b = 1,
  sorry
}

theorem range_of_f_on_interval : 
  set.range (λ x, f x) (set.Icc 0 (Real.pi / 2)) = set.Icc 0 ((Real.exp (Real.pi / 4) * Real.sqrt 2) / 2) :=
by {
  sorry
}

end tangent_line_at_0_range_of_f_on_interval_l87_87032


namespace football_games_total_l87_87213

def total_football_games_per_season (games_per_month : ℝ) (num_months : ℝ) : ℝ :=
  games_per_month * num_months

theorem football_games_total (games_per_month : ℝ) (num_months : ℝ) (total_games : ℝ) :
  games_per_month = 323.0 ∧ num_months = 17.0 ∧ total_games = 5491.0 →
  total_football_games_per_season games_per_month num_months = total_games :=
by
  intros h
  have h1 : games_per_month = 323.0 := h.1
  have h2 : num_months = 17.0 := h.2.1
  have h3 : total_games = 5491.0 := h.2.2
  rw [h1, h2, h3]
  sorry

end football_games_total_l87_87213


namespace calc_expression1_calc_expression2_l87_87263

theorem calc_expression1 : (1 / 3)^0 + Real.sqrt 27 - abs (-3) + Real.tan (Real.pi / 4) = 1 + 3 * Real.sqrt 3 - 2 :=
by
  sorry

theorem calc_expression2 (x : ℝ) : (x + 2)^2 - 2 * (x - 1) = x^2 + 2 * x + 6 :=
by
  sorry

end calc_expression1_calc_expression2_l87_87263


namespace part1_short_distance_A_part2_short_distance_B_part3_equidistant_C_D_l87_87828

-- Define the "short distance" of a point
def short_distance (x y : ℝ) : ℝ :=
  min (|x|) (|y|)

-- Define equidistant points
def equidistant_points (P Q : ℝ × ℝ) : Prop :=
  short_distance (P.1) (P.2) = short_distance (Q.1) (Q.2)

-- Prove the parts of the problem
theorem part1_short_distance_A :
  short_distance (-5) (-2) = 2 := 
sorry

theorem part2_short_distance_B (m : ℝ) :
  short_distance (-2) (-2 * m + 1) = 1 ↔ m = 0 ∨ m = 1 :=
sorry

theorem part3_equidistant_C_D (k : ℝ) :
  equidistant_points (-1, k + 3) (4, 2 * k - 3) ↔ k = 1 ∨ k = 2 := 
sorry

end part1_short_distance_A_part2_short_distance_B_part3_equidistant_C_D_l87_87828


namespace a_n_general_term_sum_S_n_max_T_n_min_T_n_l87_87424

noncomputable def a₁ := 3 / 2

noncomputable def a (n : ℕ) (h : n ≥ 2) : ℝ :=
  a (n - 1) h + (9 / 2) * (-1 / 2) ^ (n - 1)

def S (n : ℕ) : ℝ :=
  1 - (-1 / 2) ^ n

def T (n : ℕ) : ℝ :=
  let Sn := S n in Sn - 1 / Sn

theorem a_n_general_term (n : ℕ) : ℝ :=
  a n = (3 / 2) - (3 / 2) * (-1 / 2) ^ (n - 1) := sorry

theorem sum_S_n (n : ℕ) : ℝ :=
  S n = 1 - (-1 / 2) ^ n := sorry

theorem max_T_n : T 1 = 5 / 6 := sorry

theorem min_T_n : T 2 = -7 / 12 := sorry

end a_n_general_term_sum_S_n_max_T_n_min_T_n_l87_87424


namespace proof_yield_and_conversion_l87_87605

def yield_of_ordinary_rice (x : ℝ) (y : ℝ) (a : ℝ) (b : ℝ) (diff : ℝ) (harvest_a : ℝ) (harvest_b : ℝ) : Prop := 
  y = 2 * x ∧ 
  harvest_a = a * y ∧ 
  harvest_b = b * x ∧ 
  b - a = diff

def conversion_of_acres (yield_goal : ℝ) (current_a : ℝ) (current_b : ℝ) (yield_per_acre_ordinary : ℝ) (yield_per_acre_hybrid : ℝ) : ℝ → Prop :=
  λ converted_acres : ℝ, 
  current_a * yield_per_acre_hybrid + (current_b - converted_acres) * yield_per_acre_ordinary + converted_acres * yield_per_acre_hybrid ≥ yield_goal

theorem proof_yield_and_conversion
  (x : ℝ) (y : ℝ) (a : ℝ) (b : ℝ) (diff : ℝ) (harvest_a : ℝ) (harvest_b : ℝ)
  (yield_goal : ℝ) (yield_per_acre_ordinary : ℝ) (yield_per_acre_hybrid : ℝ)
  (converted_acres : ℝ) :
  yield_of_ordinary_rice x y a b diff harvest_a harvest_b ∧ 
  conversion_of_acres yield_goal a b yield_per_acre_ordinary yield_per_acre_hybrid converted_acres → 
  (yield_per_acre_ordinary = 600 ∧ yield_per_acre_hybrid = 1200) ∧ converted_acres >= 1.5 :=
begin
  sorry
end

end proof_yield_and_conversion_l87_87605


namespace speed_of_current_l87_87275

theorem speed_of_current
  (boat_speed : ℕ := 16)
  (upstream_minutes : ℕ := 20)
  (downstream_minutes : ℕ := 15) :
  let c := (80 / 35 : ℚ) in
  c = 16 / 7 := 
by
  sorry

end speed_of_current_l87_87275


namespace calculate_v3_l87_87303

def f (x : ℤ) : ℤ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def v0 : ℤ := 2
def v1 (x : ℤ) : ℤ := v0 * x + 5
def v2 (x : ℤ) : ℤ := v1 x * x + 6
def v3 (x : ℤ) : ℤ := v2 x * x + 23

theorem calculate_v3 : v3 (-4) = -49 :=
by
sorry

end calculate_v3_l87_87303


namespace prove_trajectory_and_area_l87_87142

-- Define the circle E and point P on the coordinate plane
def circleE := {x y : ℝ // (x + 1)^2 + y^2 = 8}
def pointP := (1 : ℝ, 0 : ℝ)

-- Define the equation of trajectory C
def trajectoryC (M : ℝ × ℝ) : Prop :=
  M.fst^2 / 2 + M.snd^2 = 1

-- Define the line l and conditions for intersection and tangency
def line_tangent (O A B : ℝ × ℝ) (k m : ℝ) : Prop :=
  (A.snd = k * A.fst + m) ∧ (B.snd = k * B.fst + m) ∧
  ((sqrt (k^2 + 1)) * abs m = 1)

-- Condition for dot product and area of the triangle
def dot_product_condition (O A B : ℝ × ℝ) (k : ℝ) : Prop :=
  ((A.fst * B.fst + A.snd * B.snd) ∈ (2/3 : ℝ)..(3/4 : ℝ)) ∧ (1/2 ≤ k^2 ∧ k^2 ≤ 1)

def area_triangle (A B : ℝ × ℝ) : ℝ :=
  sqrt ((2 * ((k^2)^2 + k^2)) / (4 * ((k^2)^2 + k^2) + 1))

def area_range (S : ℝ) : Prop :=
  (sqrt 6 / 4 ≤ S ∧ S ≤ 2 / 3)

-- Main Lean 4 statement combining the conditions and results
theorem prove_trajectory_and_area : 
  (∀ M : ℝ × ℝ, trajectoryC M) ∧ 
  (∀ O A B : ℝ × ℝ, ∀ k m : ℝ, 
    line_tangent O A B k m → 
    dot_product_condition O A B k → 
    area_range (area_triangle A B)) :=
by {
  sorry,
  sorry
}

end prove_trajectory_and_area_l87_87142


namespace parabola_solution_l87_87040

noncomputable def parabola_focus (p : ℝ) := (0, p / 2)

def circle (x y : ℝ) := x^2 + (y + 4)^2 = 1

def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem parabola_solution (p : ℝ) (P : ℝ × ℝ) :
  (∀ (x y : ℝ), circle x y → 
   distance 0 (p / 2) x y - 1 = 4) → 
  p = 2 ∧ (∀ P : ℝ × ℝ, circle P.1 P.2 → 
    ∃ (A B : ℝ × ℝ), 
    max_area_triangle P A B = 20 * real.sqrt 5) :=
by
  have parabola := λ p, x^2 = 2 * p * y
  have F := parabola_focus p
  have M := circle
  have d := distance
  have H := 4
  sorry -- Complete the theorem proof here

end parabola_solution_l87_87040


namespace range_of_k_l87_87501

theorem range_of_k (x_1 x_2 : ℝ) (h1 : x_1 > 0) (h2 : x_2 > 0) :
  (∀ k > 0, (g x_1) / k ≤ (f x_2) / (k + 1) ↔ k ≥ 1) :=
by
  let f := λ x : ℝ, (exp 2 * x^2 + 1) / x
  let g := λ x : ℝ, (exp 2 * x) / (exp x)
  sorry

end range_of_k_l87_87501


namespace prove_x_eq_one_l87_87497

variables (x y : ℕ)

theorem prove_x_eq_one 
  (hx : x > 0) 
  (hy : y > 0) 
  (hdiv : ∀ n : ℕ, n > 0 → (2^n * y + 1) ∣ (x^2^n - 1)) : 
  x = 1 :=
sorry

end prove_x_eq_one_l87_87497


namespace num_memorable_numbers_l87_87688

-- Definitions for memorable telephone number criteria
def is_memorable (d : Fin 10 → Fin 10 → Fin 10 → Fin 10 → Fin 10 → Fin 10 → Fin 10 → Fin 10) : Prop :=
  (d 4 = d 0 ∧ d 5 = d 1 ∧ d 6 = d 2) ∨ (d 4 = d 1 ∧ d 5 = d 2 ∧ d 6 = d 3)

-- The main assertion
theorem num_memorable_numbers : 
  ∑ d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ in (Fin 10), if is_memorable (λ i, vector.nth ⟨[d₁, d₂, d₃, d₄, d₅, d₆, d₇, d₈], by simp⟩) then (1 : ℕ) else (0 : ℕ) = 199990 := 
sorry

end num_memorable_numbers_l87_87688


namespace decreasing_interval_quadratic_l87_87541

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 + x + 1

theorem decreasing_interval_quadratic :
  ∀ x ∈ set.Iic (-1/2 : ℝ), ∀ y ∈ set.Iic (-1/2 : ℝ), y < x → quadratic_function x ≤ quadratic_function y := by
  sorry

end decreasing_interval_quadratic_l87_87541


namespace rectangle_length_width_l87_87279

theorem rectangle_length_width 
  (x y : ℚ)
  (h1 : x - 5 = y + 2)
  (h2 : x * y = (x - 5) * (y + 2)) :
  x = 25 / 3 ∧ y = 4 / 3 :=
by
  sorry

end rectangle_length_width_l87_87279


namespace four_digit_perfect_square_l87_87970

theorem four_digit_perfect_square (N : ℕ) (a b : ℕ) :
  (1000 ≤ N ∧ N < 10000) ∧
  (∃ a b : ℕ, N = 1100 * a + 11 * b ∧ (a > 0 ∧ a < 10) ∧ (b ≥ 0 ∧ b < 10)) ∧
  (∃ k : ℕ, N = k * k) →
  N = 7744 :=
begin
  sorry
end

end four_digit_perfect_square_l87_87970


namespace max_annual_profit_at_x_9_l87_87272

noncomputable def annual_profit (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then
  8.1 * x - x^3 / 30 - 10
else
  98 - 1000 / (3 * x) - 2.7 * x

theorem max_annual_profit_at_x_9 (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 10) :
  annual_profit x ≤ annual_profit 9 :=
sorry

end max_annual_profit_at_x_9_l87_87272


namespace circle_no_intersect_axes_l87_87444

theorem circle_no_intersect_axes {k : ℝ} (h : k > 0) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * k * x + 2 * y + 2 ≠ 0 → 
    ¬(x = 0 ∨ y = 0)) ↔ 1 < k ∧ k < real.sqrt 2 :=
sorry

end circle_no_intersect_axes_l87_87444


namespace find_base_l87_87834

theorem find_base (r : ℕ) (h1 : 5 * r^2 + 3 * r + 4 + 3 * r^2 + 6 * r + 6 = r^3) : r = 10 :=
by
  sorry

end find_base_l87_87834


namespace find_PQ_l87_87824

theorem find_PQ (QR PR PQ : ℝ) (hQR : tan (arctan (3/5)) = (QR / PR)) (hPR : PR = 10) (hR : QR = 6) : PQ = 2 * real.sqrt 34 :=
by
  have h1 : QR = 3 / 5 * PR := by linarith
  have h2 : QR = 6 := by linarith
  have h3 : PR = 10 := by linarith
  have h4 : PQ = real.sqrt (QR^2 + PR^2) := by linarith
  sorry

end find_PQ_l87_87824


namespace total_area_is_82_l87_87833

/-- Definition of the lengths of each segment as conditions -/
def length1 : ℤ := 7
def length2 : ℤ := 4
def length3 : ℤ := 5
def length4 : ℤ := 3
def length5 : ℤ := 2
def length6 : ℤ := 1

/-- Rectangle areas based on the given lengths -/
def area_A : ℤ := length1 * length2 -- 7 * 4
def area_B : ℤ := length3 * length2 -- 5 * 4
def area_C : ℤ := length1 * length4 -- 7 * 3
def area_D : ℤ := length3 * length5 -- 5 * 2
def area_E : ℤ := length4 * length6 -- 3 * 1

/-- The total area is the sum of all rectangle areas -/
def total_area : ℤ := area_A + area_B + area_C + area_D + area_E

/-- Theorem: The total area is 82 square units -/
theorem total_area_is_82 : total_area = 82 :=
by
  -- Proof left as an exercise
  sorry

end total_area_is_82_l87_87833


namespace quadratic_term_coeff_and_constant_term_l87_87177

theorem quadratic_term_coeff_and_constant_term (x : ℝ) :
  let eqn := λ x, -3 * x^2 - 2 * x
  ∃ (a b c : ℝ), eqn x = a * x^2 + b * x + c ∧ a = -3 ∧ c = 0 := 
by
  sorry

end quadratic_term_coeff_and_constant_term_l87_87177


namespace truncated_cone_sphere_radius_l87_87663

noncomputable def radius_of_sphere {r1 r2 : ℝ} (r1_pos : 0 < r1) (r2_pos : 0 < r2)
  (r1_eq_20 : r1 = 20) (r2_eq_5 : r2 = 5) : ℝ :=
  let h := (15 * Real.sqrt 2 / 2) in h

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 : ℝ), (0 < r1) → (0 < r2) → (r1 = 20) → (r2 = 5) →
  radius_of_sphere r1_pos r2_pos r1_eq_20 r2_eq_5 = 15 * Real.sqrt 2 / 2 :=
  sorry

end truncated_cone_sphere_radius_l87_87663


namespace probability_winning_probability_not_winning_l87_87454

section Lottery

variable (p1 p2 p3 : ℝ)
variable (h1 : p1 = 0.1)
variable (h2 : p2 = 0.2)
variable (h3 : p3 = 0.4)

theorem probability_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  p1 + p2 + p3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

theorem probability_not_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  1 - (p1 + p2 + p3) = 0.3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end Lottery

end probability_winning_probability_not_winning_l87_87454


namespace rook_moves_on_chessboard_l87_87281

theorem rook_moves_on_chessboard :
  ∀ (moves : list (ℕ × ℕ)), 
  (∀ x ∈ moves, x ∈ [(0,1), (1,0), (0,-1), (-1,0)]) → -- Rook moves only vertically or horizontally
  ((• moved through) (moves) ∧ -- covered all squares
   head moves = last moves → -- returned to starting point
  (∃ m n : ℕ, m ≠ n ∧
    (count (0,1) moves = m ∧ count (0,-1) moves = m) ∧ -- Vertical moves
    (count (1,0) moves = n ∧ count (-1,0) moves = n))) -- Horizontal moves :=
sorry

end rook_moves_on_chessboard_l87_87281


namespace fractional_equation_m_value_l87_87803

theorem fractional_equation_m_value {x m : ℝ} (hx : 0 < x) (h : 3 / (x - 4) = 1 - (x + m) / (4 - x))
: m = -1 := sorry

end fractional_equation_m_value_l87_87803


namespace num_divisors_630_l87_87558

theorem num_divisors_630 : ∃ d : ℕ, (d = 24) ∧ ∀ n : ℕ, (∃ (a b c d : ℕ), (n = 2^a * 3^b * 5^c * 7^d) ∧ a ≤ 1 ∧ b ≤ 2 ∧ c ≤ 1 ∧ d ≤ 1) ↔ (n ∣ 630) := sorry

end num_divisors_630_l87_87558


namespace point_A_in_second_quadrant_l87_87094

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l87_87094


namespace dining_bill_sharing_l87_87201

theorem dining_bill_sharing : 
  ∃ n : ℤ, 211 + 0.15 * 211 = 40.44 * n ∧ n ≈ 6 := 
by
  have B := 211 : ℝ
  have tip_rate := 0.15
  have tip_amount := tip_rate * B
  have total_bill := B + tip_amount
  have final_share := 40.44
  have n := total_bill / final_share
  use Int.ofNat (Int.toNat (total_bill / final_share)) -- approximate n as nearest integer
  have h_n_approx : |n - 6| < 0.5 := sorry -- approximate n is 6
  exact ⟨Int.ofNat (Int.toNat (total_bill / final_share)), by
    have : n ≈ 6 ∨ n ≈ 5 := sorry  -- approximate with either 5 or 6 since Int approximation was used
    cases this with h0 h1
    { exact h0 }
    { exact h1 } ⟩

end dining_bill_sharing_l87_87201


namespace tiling_impossible_l87_87471

-- Definition of the equilateral triangle and its subdivision
def equilateral_triangle (n : ℕ) : Prop :=
  let total_triangles := n * (n + 1) / 2 in
  let smaller_triangles := total_triangles - (total_triangles / 2) in
  let pointing_up_triangles := (1 to n - 1).sum in
  let pointing_down_triangles := ((1 to n).sum) in
  total_triangles = 36 ∧ pointing_up_triangles = 15 ∧ pointing_down_triangles = 21

-- Definition of the sphinx shape
def sphinx_shape : Prop :=
  ∃ (up down : ℕ), (up, down) = (4, 2) ∨ (up, down) = (2, 4)

-- Prove tiling is impossible
theorem tiling_impossible : 
  ∀ n, equilateral_triangle n → sphinx_shape → 
  (∃ m : ℕ, m * (4 + 2) = 36 → 
  ∀ k : ℕ, k % 2 = 0 → 
  (36 - k * 4) % 2 = 0 ∧ (36 - k * 2) % 2 = 0) → 
  false
:= 
by
  sorry

end tiling_impossible_l87_87471


namespace even_product_permutation_l87_87858

theorem even_product_permutation (a : Fin 1995 → Fin 1995) (ha : Function.Bijective a) :
  Even (List.prod (List.map (λ i, (i.succ : ℕ) - (a i.succ : ℕ)) (List.finRange 1995))) :=
sorry

end even_product_permutation_l87_87858


namespace range_of_a_l87_87775

open Real

theorem range_of_a (a : ℝ) :
  (∀ x > 0, ae^x + x + x * log x ≥ x^2) → a ≥ 1 / exp 2 :=
sorry

end range_of_a_l87_87775


namespace christine_siri_total_money_l87_87681

-- Define the conditions
def christine_has_more_than_siri : ℝ := 20 -- Christine has 20 rs more than Siri
def christine_amount : ℝ := 20.5 -- Christine has 20.5 rs

-- Define the proof problem
theorem christine_siri_total_money :
  (∃ (siri_amount : ℝ), christine_amount = siri_amount + christine_has_more_than_siri) →
  ∃ total : ℝ, total = christine_amount + (christine_amount - christine_has_more_than_siri) ∧ total = 21 :=
by sorry

end christine_siri_total_money_l87_87681


namespace union_prob_inconsistency_l87_87759

noncomputable def p_a : ℚ := 2/15
noncomputable def p_b : ℚ := 4/15
noncomputable def p_b_given_a : ℚ := 3

theorem union_prob_inconsistency : p_a + p_b - p_b_given_a * p_a = 0 → false := by
  sorry

end union_prob_inconsistency_l87_87759


namespace machine_value_after_two_years_l87_87550

def machine_market_value (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  (0 : ℕ).subsequent years (λ v n, v * (1 - decrease_rate)) initial_value

theorem machine_value_after_two_years :
  machine_market_value 8000 0.10 2 = 6480 :=
by simp [machine_market_value] ; sorry

end machine_value_after_two_years_l87_87550


namespace number_starts_with_nines_l87_87154

-- Define the decimal number with 100 nines.
def nines100 : ℝ := 0.999... 999 -- placeholder for 100 nines

-- Lean does not handle repeating decimals well, so use the equivalent
def nines100_alt : ℝ := 1 - 10^(-100)

-- Main theorem statement
theorem number_starts_with_nines (a : ℝ) (h0 : 0 < a) (h1 : a < 1) (h2 : nines100_alt ≤ a^2 ∧ a^2 < 1) :
  nines100_alt ≤ a :=
sorry

end number_starts_with_nines_l87_87154


namespace train_lateness_l87_87286

variable (S T T' : ℝ)
variable (usual_time : T = 3)
variable (current_speed : S' = 6 / 7 * S)
variable (distance_unchanged : S * T = S' * T')

theorem train_lateness :
  T = 3 → S' = 6 / 7 * S → S * T = S' * T' → (T' = 3.5) :=
by
  intro hT hS' hDist
  rw [hT, hS']
  have h: S' = 6 / 7 * S := hS'
  rw h at hDist
  sorry

end train_lateness_l87_87286


namespace volume_ratio_theorem_l87_87973

noncomputable def volume_ratio (
  a b c : ℝ, 
  O A B C D A' B' C' D' : Type,
  h1 : OA' / OA = 1 / a,
  h2 : OB' / OB = 1 / b,
  h3 : OC' / OC = 1 / c,
  hParallelogram : Parallelogram ABCD
) : ℝ :=
  2 * a * b * c * (a - b + c) / (a + c)

theorem volume_ratio_theorem (
  a b c : ℝ, 
  O A B C D A' B' C' D' : Type,
  h1 : OA' / OA = 1 / a,
  h2 : OB' / OB = 1 / b,
  h3 : OC' / OC = 1 / c,
  hParallelogram : Parallelogram ABCD
) : 
  volume_ratio a b c O A B C D A' B' C' D' = 2 * a * b * c * (a - b + c) / (a + c) :=
begin
  sorry
end

end volume_ratio_theorem_l87_87973


namespace find_other_corners_l87_87143

def is_right_angle_coordinate_system (p1 p2 p3: (ℝ × ℝ)) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0

theorem find_other_corners 
  (c1 c2 : ℝ × ℝ) 
  (h1 : c1 = (1, 14)) 
  (h2 : c2 = (17, 2))
  (shorter_wall_half : (abs (sqrt ((c2.1 - c1.1) ^ 2 + (c2.2 - c1.2) ^ 2)) / 2))
  (c3 c4 : ℝ × ℝ) :
  is_right_angle_coordinate_system c1 c2 c3 → 
  is_right_angle_coordinate_system c1 c2 c4 → 
  c3 = (-5, 6) ∧ c4 = (11, -6) :=
begin
  sorry
end

end find_other_corners_l87_87143


namespace magnitude_sum_of_perpendicular_vectors_l87_87390

variable (a b : ℝ^3)
variables (h₁ : |a| = 1) (h₂ : |b| = √2) (h₃ : a ⬝ b = 0)

theorem magnitude_sum_of_perpendicular_vectors :
  |a + b| = √3 :=
  sorry

end magnitude_sum_of_perpendicular_vectors_l87_87390


namespace xiao_qiao_purchased_total_stamps_l87_87246

variable (number_4_yuan number_8_yuan : ℕ) -- Number of 4-yuan and 8-yuan stamps

/-- Given conditions -/
variable (total_spent : ℕ := 660)
variable (price_4_yuan price_8_yuan : ℕ := (4, 8))
variable (additional_8_yuan : ℕ := 30)

/-- Assuming the relationship between the number of 4-yuan and 8-yuan stamps -/
variable (relationship : number_8_yuan = number_4_yuan + additional_8_yuan)

theorem xiao_qiao_purchased_total_stamps :
  price_4_yuan * number_4_yuan + price_8_yuan * number_8_yuan = total_spent → 
  number_4_yuan + number_8_yuan = 100 := by
  intros h
  sorry

end xiao_qiao_purchased_total_stamps_l87_87246


namespace line_intersects_broken_line_l87_87077

theorem line_intersects_broken_line (n1 n2 m: ℕ) (polygonal_chain : Type*) (l l1 : Type*)
  (h1 : n1 + n2 + m = 1985)
  (h2 : m % 2 = 0)
  (h3 : n1 ≠ n2)
  (h4 : 2 * n1 + m > 1985) : 
  ∃ (l1 : Type*), l1 > 1985 :=
by sorry

end line_intersects_broken_line_l87_87077


namespace donation_sum_l87_87674

noncomputable def total_donation (x : ℝ) : ℝ :=
  let treetown := 570
  let forest_reserve := x + 140
  let total := treetown + forest_reserve + x + (3 * x / 2)
  total

theorem donation_sum :
  ∃ x : ℝ, 
  let treetown := 570
  let forest_reserve := x + 140
  let birds_sanctuary := 3 * x / 2
  (4 * x + 570 = (x + 140) * 5 / 4) ∧
  (x * 2 = x + 140 - x) ∧
  (570 + forest_reserve + x + birds_sanctuary = 1684) :=
by skip -- Proof goes here, using the specified conditions to show the donations total $1684.


end donation_sum_l87_87674


namespace sum_first_eight_terms_l87_87744

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (n : ℕ) (a_3 a_6 : ℝ)

-- Conditions
def arithmetic_sequence (a_3 a_6 : ℝ) : Prop := a_3 = 20 - a_6

def sum_terms (S : ℕ → ℝ) (a_3 a_6 : ℝ) : ℝ :=
  4 * (a_3 + a_6)

-- The proof goal
theorem sum_first_eight_terms (a_3 a_6 : ℝ) (h₁ : a_3 = 20 - a_6) : S 8 = 80 :=
by
  rw [arithmetic_sequence a_3 a_6] at h₁
  sorry

end sum_first_eight_terms_l87_87744


namespace find_initial_amount_l87_87629

-- In this statement, we define the conditions and the goal based on the problem formulated above.
theorem find_initial_amount
  (P R : ℝ) -- P: Initial principal amount, R: Rate of interest in percentage
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1061 = P * (1 + 3 * (R + 4) / 100)) :
  P = 875 :=
by sorry

end find_initial_amount_l87_87629


namespace parabola_circle_golden_ratio_l87_87441

-- Defining all necessary points and conditions in Lean 4
structure Circle (α : Type) [LinearOrder α] :=
(center : α × α)
(radius : α)

structure Parabola (α : Type) [LinearOrder α] :=
(vertex : α × α)
(focus : α × α)

noncomputable def GoldenRatio := (1 + Real.sqrt 5) / 2

-- Define the main theorem in Lean 4
theorem parabola_circle_golden_ratio {α : Type} [LinearOrder α] {c : Circle α} {p : Parabola α}
  (h_center : p.vertex = c.center)
  (h_focus : dist c.center p.focus = c.radius) :
  ∃ A B C : α × α, line_through A B ∧ intersect p c A ∧ intersect p c B ∧
  ( C = midpoint A B ∧ dist c.center C / dist C p.focus = GoldenRatio) :=
sorry

end parabola_circle_golden_ratio_l87_87441


namespace additional_rate_of_interest_l87_87531

variable (P A A' : ℝ) (T : ℕ) (R : ℝ)

-- Conditions
def principal_amount := (P = 8000)
def original_amount := (A = 9200)
def time_period := (T = 3)
def new_amount := (A' = 9440)

-- The Lean statement to prove the additional percentage of interest
theorem additional_rate_of_interest  (P A A' : ℝ) (T : ℕ) (R : ℝ)
    (h1 : principal_amount P)
    (h2 : original_amount A)
    (h3 : time_period T)
    (h4 : new_amount A') :
    (A' - P) / (P * T) * 100 - (A - P) / (P * T) * 100 = 1 :=
by
  sorry

end additional_rate_of_interest_l87_87531


namespace point_A_in_second_quadrant_l87_87102

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l87_87102


namespace log_sqrt_defined_l87_87369

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l87_87369


namespace roots_greater_than_3_l87_87767

theorem roots_greater_than_3 (m : ℝ) :
  (∀ x ∈ {x | x^2 - (3*m+2)*x + 2*(m+6) = 0}, x > 3) ↔ m ∈ (4/3 : ℚ) :< (15/7 : ℚ) :=
begin
  sorry
end

end roots_greater_than_3_l87_87767


namespace problem_l87_87015

theorem problem (x : ℝ) : (x^2 + 2 * x - 3 ≤ 0) → ¬(abs x > 3) :=
by sorry

end problem_l87_87015


namespace pasha_can_obtain_integer_root_polynomial_l87_87178

theorem pasha_can_obtain_integer_root_polynomial {a b c : ℕ} (h_sum : a + b + c = 2000) :
  ∃ a' b' c', (∃ n : ℕ, a' * n ^ 2 + b' * n + c' = 0) ∧ (abs ((a - a') + (b - b') + (c - c')) ≤ 1050) :=
  sorry

end pasha_can_obtain_integer_root_polynomial_l87_87178


namespace doubling_time_l87_87074

variables (b d i e : ℝ)
def natural_change : ℝ := b - d
def net_migration : ℝ := i - e
def total_annual_growth_rate : ℝ := natural_change b d + net_migration i e
def annual_growth_rate_percentage : ℝ := (total_annual_growth_rate b d i e / 1000) * 100

theorem doubling_time (h1 : b = 39.4) (h2 : d = 19.4) (h3 : i = 10.2) (h4 : e = 6.9) :
  ⌊70 / annual_growth_rate_percentage b d i e⌋ = 30 :=
by {
  sorry
}

end doubling_time_l87_87074


namespace initial_bananas_min_l87_87934

def first_monkey (a b c : ℕ) : ℕ := (2 * a) / 3 + (7 * b) / 30 + (2 * c) / 9
def second_monkey (a b c : ℕ) : ℕ := a / 3 + (b) / 5 + (2 * c) / 9
def third_monkey (a b c : ℕ) : ℕ := a / 3 + (4 * b) / 15 + (c) / 10

def ratio_valid (a b c : ℕ) : Prop :=
  ∃ x : ℕ, x = third_monkey a b c ∧
           2 * x = second_monkey a b c ∧
           4 * x = first_monkey a b c

theorem initial_bananas_min : ∃ n, n = 215 ∧
  ∃ a b c : ℕ,
    a + b + c = n ∧
    ratio_valid a b c ∧
    a = 30 ∧ b = 35 ∧ c = 150 :=
begin
  -- proof to be filled in
  sorry
end

end initial_bananas_min_l87_87934


namespace probability_of_sum_is_multiple_of_3_l87_87807

-- Define the set of the first ten prime numbers
def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function to check if the sum of two numbers is a multiple of 3
def sum_is_multiple_of_3 (a b : ℕ) : Prop := (a + b) % 3 = 0

-- Define the number of ways to choose two distinct primes from the ten primes
def number_of_ways_to_choose_two_primes : ℕ := (first_ten_primes.card.choose 2)

-- Define the set of pairs whose sum is a multiple of 3
def valid_pairs : Finset (ℕ × ℕ) :=
  (first_ten_primes.product first_ten_primes).filter (λ p, p.1 < p.2 ∧ sum_is_multiple_of_3 p.1 p.2)

-- The cardinality of the valid pairs
def number_of_valid_pairs : ℕ := valid_pairs.card

-- The probability that the sum is a multiple of 3
def probability : ℚ := (number_of_valid_pairs : ℚ) / (number_of_ways_to_choose_two_primes : ℚ)

-- The main proof statement
theorem probability_of_sum_is_multiple_of_3 :
  probability = 1/5 := by
  sorry

end probability_of_sum_is_multiple_of_3_l87_87807


namespace natalie_needs_10_bushes_l87_87702

-- Definitions based on the conditions
def bushes_to_containers (bushes : ℕ) := bushes * 10
def containers_to_zucchinis (containers : ℕ) := (containers * 3) / 4

-- The proof statement
theorem natalie_needs_10_bushes :
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) ≥ 72 ∧ bushes = 10 :=
sorry

end natalie_needs_10_bushes_l87_87702


namespace number_of_articles_l87_87446

-- Define the conditions
def gain := 1 / 9
def cp_one_article := 1  -- cost price of one article

-- Define the cost price for x articles
def cp (x : ℕ) := x * cp_one_article

-- Define the selling price for 45 articles
def sp (x : ℕ) := x / 45

-- Define the selling price equation considering gain
def sp_one_article := (cp_one_article * (1 + gain))

-- Main theorem to prove
theorem number_of_articles (x : ℕ) (h : sp x = sp_one_article) : x = 50 :=
by
  sorry

-- The theorem imports all necessary conditions and definitions and prepares the problem for proof.

end number_of_articles_l87_87446


namespace log_over_sqrt_defined_l87_87359

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l87_87359


namespace question_1_question_2_question_3_l87_87503

def Sn(n : ℕ) : Set ℕ := { i | 1 ≤ i ∧ i ≤ n }

def capacity(X : Set ℕ) : ℕ := X.sum

def is_odd_subset(X : Set ℕ) : Prop := X.sum % 2 = 1
def is_even_subset(X : Set ℕ) : Prop := X.sum % 2 = 0

def odd_subsets(n : ℕ) : Finset (Finset ℕ) := (Finset.powerset (Finset.range n)).filter (λ X, X.sum % 2 = 1)
def even_subsets(n : ℕ) : Finset (Finset ℕ) := (Finset.powerset (Finset.range n)).filter (λ X, X.sum % 2 = 0)

theorem question_1 : ∀ n, (odd_subsets n).card = (even_subsets n).card := by
  sorry

theorem question_2 : ∀ n, n ≥ 3 → (odd_subsets n).sum capacity = (even_subsets n).sum capacity := by
  sorry

theorem question_3 : ∀ n, n ≥ 3 → (odd_subsets n).sum capacity = 2^(n - 3) * n * (n + 1) := by
  sorry

end question_1_question_2_question_3_l87_87503


namespace savings_percentage_l87_87849

variables (S : ℝ) -- Salary last year

def last_year_savings := 0.06 * S
def this_year_salary := 1.10 * S
def this_year_savings := 0.08 * this_year_salary

theorem savings_percentage :
  (this_year_savings / last_year_savings) * 100 = 146.67 := 
sorry

end savings_percentage_l87_87849


namespace arithmetic_geometric_sequence_problem_l87_87760

variable {n : ℕ}

def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := n * (2 + (2 + (n - 1) * (3 - 1))) / 2 -- sum of an arithmetic sequence
def T (n : ℕ) : ℕ := (3 * n - 4) * 2 ^ (n + 1) + 8

theorem arithmetic_geometric_sequence_problem :
  (a 1 = 2) ∧ (b 1 = 2) ∧ (a 4 + b 4 = 27) ∧ (S 4 - b 4 = 10) →
  (∀ n, T n = (3 * n - 4) * 2 ^ (n + 1) + 8) := sorry

end arithmetic_geometric_sequence_problem_l87_87760


namespace david_marks_in_math_l87_87697

theorem david_marks_in_math
  (marks_in_english : ℝ)
  (marks_in_physics : ℝ)
  (marks_in_chemistry : ℝ)
  (marks_in_biology : ℝ)
  (average_marks : ℝ)
  (num_subjects : ℝ)
  (total_marks : ℝ)
  (marks_in_math : ℝ) :
  marks_in_english = 72 →
  marks_in_physics = 72 →
  marks_in_chemistry = 77 →
  marks_in_biology = 75 →
  average_marks = 68.2 →
  num_subjects = 5 →
  total_marks = average_marks * num_subjects →
  marks_in_math = total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) →
  marks_in_math = 45 :=
by
  intros h_me h_mp h_mc h_mb h_avgmarks h_numsubs h_totmarks h_mim
  rw [h_me, h_mp, h_mc, h_mb] at h_totmarks
  rw [h_avgmarks, h_numsubs] at h_totmarks
  have : total_marks = 341, from h_totmarks
  have : marks_in_math = 341 - (72 + 72 + 77 + 75), from h_mim
  rw this
  norm_num
  exact this

end david_marks_in_math_l87_87697


namespace solve_part1_solve_part2_l87_87875

def A (x : ℝ) : Prop := 2 < x ∧ x < 4

def B (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < 3 * a

theorem solve_part1 (a : ℝ) :
  (∀ x, A x ∨ B a x ↔ (2 < x ∧ x < 6)) → a = 2 :=
sorry

theorem solve_part2 (a : ℝ) :
  (∃ x, A x ∧ B a x) ↔ a ∈ set.Ioo (2/3 : ℝ) 4 :=
sorry

end solve_part1_solve_part2_l87_87875


namespace jason_money_left_l87_87476

-- Define the initial amount of money and the spent fractions.
def initialMoney : ℤ := 320
def spentOnBooks : ℤ := (1/4 : ℝ) * initialMoney + 10
def remainingAfterBooks : ℤ := initialMoney - spentOnBooks
def spentOnDVDs : ℤ := (2/5 : ℝ) * remainingAfterBooks + 8
def remainingAfterDVDs : ℤ := remainingAfterBooks - spentOnDVDs

-- State the problem
theorem jason_money_left (initialMoney = 320) (spentOnBooks = (1/4 : ℝ) * 320 + 10) 
    (remainingAfterBooks = 320 - spentOnBooks) 
    (spentOnDVDs = (2/5 : ℝ) * remainingAfterBooks + 8) 
    (remainingAfterDVDs = remainingAfterBooks - spentOnDVDs) :
    remainingAfterDVDs = 130 :=
    sorry

end jason_money_left_l87_87476


namespace no_points_C_exist_l87_87813

noncomputable def AC_len (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def BC_len (x y : ℝ) : ℝ :=
  Real.sqrt ((x - 8)^2 + y^2)

noncomputable def number_of_points_C : ℕ :=
  if (∀ x y, 8 + 2 * AC_len x y = 40 ∧ 4 * abs y = 80 ∧ AC_len x y = BC_len x y) 
  then 0 else sorry

theorem no_points_C_exist :
  number_of_points_C = 0 :=
sorry

end no_points_C_exist_l87_87813


namespace quadruple_solution_l87_87708

theorem quadruple_solution (a b p n : ℕ) (hp: Nat.Prime p) (hp_pos: p > 0) (ha_pos: a > 0) (hb_pos: b > 0) (hn_pos: n > 0) :
    a^3 + b^3 = p^n →
    (∃ k, k ≥ 1 ∧ (
        (a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k-2) ∨ 
        (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k-1) ∨ 
        (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k-1)
    )) := 
sorry

end quadruple_solution_l87_87708


namespace tank_fill_fraction_l87_87799

theorem tank_fill_fraction (a b c : ℝ) (h1 : a=9) (h2 : b=54) (h3 : c=3/4) : (c * b + a) / b = 23 / 25 := 
by 
  sorry

end tank_fill_fraction_l87_87799


namespace intersection_solution_l87_87839

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end intersection_solution_l87_87839


namespace david_telephone_numbers_l87_87291

theorem david_telephone_numbers : 
  let digits := {2, 3, 4, 5, 6, 7, 8, 9} in
  let telephone_numbers := { l : List ℕ // l.nodup ∧ l.sorted (· < ·) ∧ l.length = 8 ∧ ∀ x ∈ l, x ∈ digits } in
  telephone_numbers.card = 1 := 
by
  sorry

end david_telephone_numbers_l87_87291


namespace nancy_bought_43_bars_l87_87515

/-- 
Nancy bought 4 packs of soap from Brand A, each pack containing 3 bars.
Nancy bought 3 packs of soap from Brand B, each pack containing 5 bars.
Brand C has an offer: if you buy 2 packs with 6 bars each, you get an extra pack with 4 bars for free. Nancy utilized this offer.
Prove that Nancy bought a total of 43 bars.
-/
theorem nancy_bought_43_bars :
  let a_bars := 4 * 3,
      b_bars := 3 * 5,
      c_bars := 2 * 6 + 4 in
  a_bars + b_bars + c_bars = 43 :=
by
  let a_bars := (4 * 3)
  let b_bars := (3 * 5)
  let c_bars := (2 * 6 + 4)
  have total_bars := a_bars + b_bars + c_bars
  show total_bars = 43
  sorry

end nancy_bought_43_bars_l87_87515


namespace max_three_kopecks_l87_87936

def is_coin_placement_correct (n1 n2 n3 : ℕ) : Prop :=
  -- Conditions for the placement to be valid
  ∀ (i j : ℕ), i < j → 
  ((j - i > 1 → n1 = 0) ∧ (j - i > 2 → n2 = 0) ∧ (j - i > 3 → n3 = 0))

theorem max_three_kopecks (n1 n2 n3 : ℕ) (h : n1 + n2 + n3 = 101) (placement_correct : is_coin_placement_correct n1 n2 n3) :
  n3 = 25 ∨ n3 = 26 :=
sorry

end max_three_kopecks_l87_87936


namespace solution_set_of_inequality_l87_87377

noncomputable def f : ℝ → ℝ := sorry
axiom differentiable_f : Differentiable ℝ f
axiom f_at_1 : f 1 = 2
axiom f_condition : ∀ x : ℝ, f x + f'' x < 1

theorem solution_set_of_inequality :
  {x : ℝ | f x - 1 < Real.exp (1 - x)} = {x : ℝ | 1 < x} := sorry

end solution_set_of_inequality_l87_87377


namespace circle_graph_growth_pattern_l87_87665

theorem circle_graph_growth_pattern 
    (radii : List ℝ)
    (h_radii : radii = [1.5, 2.5, 3.5, 4.5, 5.5]) :
    ∃ (C A : ℝ → ℝ),
    (∀ r ∈ radii, C r = 2 * real.pi * r) ∧ 
    (∀ r ∈ radii, A r = real.pi * r^2) ∧ 
    (∃ f : ℝ → ℝ, 
    (∀ r ∈ radii, f (C r) = A r) ∧ (∃ a b : ℝ, ∀ x, f x = a * x^2 + b * x)) :=
by
  sorry

end circle_graph_growth_pattern_l87_87665


namespace triangle_ABC_right_angled_l87_87809

theorem triangle_ABC_right_angled 
  (A B C : ℝ) 
  (b c : ℝ) 
  (h1 : cos^2 (A / 2) = (b + c) / (2 * c)) :
  C = π / 2 :=
sorry

end triangle_ABC_right_angled_l87_87809


namespace minimum_value_is_14_div_27_l87_87714

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l87_87714


namespace dressing_p_percentage_l87_87609

-- Define the percentages of vinegar and oil in dressings p and q
def vinegar_in_p : ℝ := 0.30
def vinegar_in_q : ℝ := 0.10

-- Define the desired percentage of vinegar in the new dressing
def vinegar_in_new_dressing : ℝ := 0.12

-- Define the total mass of the new dressing
def total_mass_new_dressing : ℝ := 100.0

-- Define the mass of dressing p in the new dressing
def mass_of_p (x : ℝ) : ℝ := x

-- Define the mass of dressing q in the new dressing
def mass_of_q (x : ℝ) : ℝ := total_mass_new_dressing - x

-- Define the amount of vinegar contributed by dressings p and q
def vinegar_from_p (x : ℝ) : ℝ := vinegar_in_p * mass_of_p x
def vinegar_from_q (x : ℝ) : ℝ := vinegar_in_q * mass_of_q x

-- Define the total vinegar in the new dressing
def total_vinegar (x : ℝ) : ℝ := vinegar_from_p x + vinegar_from_q x

-- Problem statement: prove the percentage of dressing p in the new dressing
theorem dressing_p_percentage (x : ℝ) (hx : total_vinegar x = vinegar_in_new_dressing * total_mass_new_dressing) :
  (mass_of_p x / total_mass_new_dressing) * 100 = 10 :=
by
  sorry

end dressing_p_percentage_l87_87609


namespace find_g_l87_87797

variable (g : ℝ → ℝ)

-- conditions
axiom cond1 : ∀ x₁ x₂ : ℝ, g (x₁ + x₂) = g x₁ * g x₂
axiom cond2 : g 1 = 3
axiom cond3 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂

theorem find_g : g = λ x, 3 ^ x := by
  sorry

end find_g_l87_87797


namespace line_intersects_parabola_exactly_once_at_m_l87_87546

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l87_87546


namespace min_value_sin_cos_expression_l87_87719

theorem min_value_sin_cos_expression : ∀ x : ℝ, 
  ∃ y : ℝ, y = (9 / 10) ∧ (y = infi (fun x => (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1)) :=
begin
  sorry
end

end min_value_sin_cos_expression_l87_87719


namespace saucer_radius_correct_l87_87927

noncomputable def saucer_radius (A : ℝ) : ℝ :=
  real.sqrt (A / (real.pi))

theorem saucer_radius_correct : saucer_radius 28.26 = 3 := 
by
  sorry

end saucer_radius_correct_l87_87927


namespace sub_neg_eq_add_l87_87679

theorem sub_neg_eq_add {a : ℤ} : (-a) - (-a) = 0 := by
  sorry

end sub_neg_eq_add_l87_87679


namespace tomatoes_ruined_percentage_l87_87916

theorem tomatoes_ruined_percentage :
  ∀ (W : ℝ) (P : ℝ),
  (P > 0) ∧ (P < 1) ∧
  (W > 0) →
  (let cost_per_pound := 0.80 in
   let selling_price_per_pound := 0.968888888888889 in
   let target_profit_percentage := 0.09 in
   let total_cost := cost_per_pound * W in
   let target_profit := target_profit_percentage * total_cost in
   let remaining_weight := (1 - P) * W in
   let total_revenue := selling_price_per_pound * remaining_weight in
   total_revenue = total_cost + target_profit) →
   P = 0.1 :=
by
  intros W P h₁ h₂,
  sorry

end tomatoes_ruined_percentage_l87_87916


namespace min_value_x_squared_plus_6x_l87_87228

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l87_87228


namespace equal_households_at_time_specific_time_for_equal_households_l87_87289

noncomputable theory

def H1 (t : ℝ) : ℝ := 4.9 + 0.275 * t
def H2 (t : ℝ) : ℝ := 2.5 + 0.7 * t

-- Statement to prove
theorem equal_households_at_time :
  ∃ t : ℝ, H1 t = H2 t :=
by
  -- Skipping the proof
  sorry
  
-- To find the specific time when H1(t) = H2(t):
theorem specific_time_for_equal_households :
  H1 5.647 = H2 5.647 :=
by
  -- Skipping the proof
  sorry

end equal_households_at_time_specific_time_for_equal_households_l87_87289


namespace certain_event_is_eventC_l87_87959

-- Definitions for the conditions:
def eventA := "A vehicle randomly arriving at an intersection encountering a red light"
def eventB := "The sun rising from the west in the morning"
def eventC := "Two out of 400 people sharing the same birthday"
def eventD := "Tossing a fair coin with the head facing up"

-- The proof goal: proving that event C is the certain event.
theorem certain_event_is_eventC : eventC = "Two out of 400 people sharing the same birthday" :=
sorry

end certain_event_is_eventC_l87_87959


namespace brooke_science_problem_time_l87_87676

theorem brooke_science_problem_time (m s sc : ℕ) (t_math t_ss t_total t_sc n_sc : ℕ) 
  (h_m : m = 15)
  (h_s : s = 6)
  (h_sc : sc = 10)
  (h_t_math : t_math = 2 * m)
  (h_t_ss : t_ss = (1 / 2) * s)
  (h_t_total : t_total = 48) : 
  t_sc / sc = 1.5 :=
by
  sorry

end brooke_science_problem_time_l87_87676


namespace correct_calculation_l87_87241

theorem correct_calculation (a : ℝ) : (a^4 / a^3 = a) ∧ ¬ (a^4 + a^3 = a^7) ∧ ¬ ((-a^3)^2 = -a^6) ∧ ¬ (a^4 * a^3 = a^12) :=
by {
  split,
  {exact -- proof of a^4 / a^3 = a},
  split,
  {exact -- proof of not (a^4 + a^3 = a^7)},
  split,
  {exact -- proof of not ((-a^3)^2 = -a^6)},
  {exact -- proof of not (a^4 * a^3 = a^12)}
}

end correct_calculation_l87_87241


namespace min_value_of_x_squared_plus_6x_l87_87233

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l87_87233


namespace hyperbola_eccentricity_l87_87484

-- Definitions of the variables and conditions
variables {a b c x₀ y₀ e : ℝ}

-- Conditions for the hyperbola and the geometrical configuration
def hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (c = a * e) ∧ (e > 1)

def point_M : Prop := y₀ = sqrt 3 * b ∧ x₀ = c / 2

def area_OFMN : Prop := sqrt 3 * b * c = sqrt 3 * b * c

-- Main statement to be proven
theorem hyperbola_eccentricity
  (h_hyp : hyperbola)
  (h_point_m : point_M)
  (h_area : area_OFMN)
  (h_eq : c^2 / (4 * a^2) - 3 = 1) :
  e = 4 :=
sorry

end hyperbola_eccentricity_l87_87484


namespace parabola_directrix_eq_neg_2_l87_87329

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l87_87329


namespace find_square_root_l87_87724

theorem find_square_root : 
  let n := 25 * 26 * 27 * 28 + 1 in
  sqrt n = 701 :=
by
  sorry

end find_square_root_l87_87724


namespace unique_intersection_l87_87548

theorem unique_intersection {m : ℝ} :
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25 / 3 :=
by
  sorry

end unique_intersection_l87_87548


namespace exists_quadrilateral_area_l87_87483

noncomputable def ABCD_square : set (ℝ × ℝ) := {(x, y) | 0 <= x ∧ x <= 1 ∧ 0 <= y ∧ y <= 1}

-- Points definitions
def A := (0, 1)
def B := (1, 1)
def C := (1, 0)
def D := (0, 0)
def E_midpoint_AB := (0.5, 1)

-- Rotation: coordinates after rotation not needed directly, built into the formulation.
variable (α : ℝ)

-- Points after rotation
def C' := (Real.cos α, Real.sin α)

-- Condition for area calculation
def area_quadrilateral_DALC' (α : ℝ) : ℝ := 0.5 * (1 - (Real.sin α * Real.cos α))

-- Desired areas based on different conditions in the problem
axiom part_a : α = π / 4 → E_midpoint_AB = (0.5, 1) → area_quadrilateral_DALC' α = 1 / 2 * (1 - (Real.sin α * Real.cos α))
axiom part_b : (E = A ∨ E = B) → area_quadrilateral_DALC' α = 0.5
axiom part_c : (m : ℕ) → E ∈ AB → area_quadrilateral_DALC' α = 1 / m * (1 - (2 - 4 / m))

-- Compile check for the axioms, without proofs
theorem exists_quadrilateral_area :
  ∃ α : ℝ, area_quadrilateral_DALC' α = 0.5 ∧ 
  (∀ m : ℕ, m ≠ 0 → ∃ E ∈ AB, area_quadrilateral_DALC' α = 1 / m * (1 - (2 - 4 / m))) :=
sorry

end exists_quadrilateral_area_l87_87483


namespace coefficient_of_x2_in_binomial_expansion_l87_87395

theorem coefficient_of_x2_in_binomial_expansion 
  (n : ℕ) (h : (2 : ℕ)^n = 64) : 
  (binomial 6 4 * (2^2) * 81 = 4860) :=
by
  sorry

end coefficient_of_x2_in_binomial_expansion_l87_87395


namespace divide_gray_area_l87_87084

-- The conditions
variables {A_rectangle A_square : ℝ} (h : 0 ≤ A_square ∧ A_square ≤ A_rectangle)

-- The main statement
theorem divide_gray_area : ∃ l : ℝ → ℝ → Prop, (∀ (x : ℝ), l x (A_rectangle / 2)) ∧ (∀ (y : ℝ), l (A_square / 2) y) ∧ (A_rectangle - A_square) / 2 = (A_rectangle - A_square) / 2 := by sorry

end divide_gray_area_l87_87084


namespace prove_tan_sum_is_neg_sqrt3_l87_87733

open Real

-- Given conditions as definitions
def condition1 (α β : ℝ) : Prop := 0 < α ∧ α < π ∧ 0 < β ∧ β < π
def condition2 (α β : ℝ) : Prop := sin α + sin β = sqrt 3 * (cos α + cos β)

-- The statement of the proof
theorem prove_tan_sum_is_neg_sqrt3 (α β : ℝ) (h1 : condition1 α β) (h2 : condition2 α β) :
  tan (α + β) = -sqrt 3 :=
sorry

end prove_tan_sum_is_neg_sqrt3_l87_87733


namespace interval_of_defined_expression_l87_87350

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l87_87350


namespace Danielle_has_6_rooms_l87_87696

axiom Danielle_rooms : ℕ
axiom Heidi_rooms : ℕ
axiom Grant_rooms : ℕ

axiom Heidi_has_3_times_Danielle : Heidi_rooms = 3 * Danielle_rooms
axiom Grant_has_1_9_Heidi : Grant_rooms = Heidi_rooms / 9
axiom Grant_has_2_rooms : Grant_rooms = 2

theorem Danielle_has_6_rooms : Danielle_rooms = 6 :=
by {
  -- proof steps would go here
  sorry
}

end Danielle_has_6_rooms_l87_87696


namespace integral_inequality_l87_87790

noncomputable def concave_nonnegative (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) ≥ t * f x + (1 - t) * f y

theorem integral_inequality {f : ℝ → ℝ} 
  (h_nonneg : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x)
  (h_cont : continuous_on f (set.Icc 0 1))
  (h_concave : concave_nonnegative f)
  (h_boundary : f 0 = 1) :
  ∫ x in 0..1, x * f x ≤ (2 / 3) * (∫ x in 0..1, f x) ^ 2 :=
by
  sorry

end integral_inequality_l87_87790


namespace operation_none_of_these_l87_87528

theorem operation_none_of_these (op : ℕ → ℕ → ℚ) (h : op = (+) ∨ op = (-) ∨ op = (*) ∨ op = (/) ) :
  op 9 2 ≠ (19 : ℚ) / 3 :=
by
  sorry

end operation_none_of_these_l87_87528


namespace number_of_correct_propositions_is_zero_l87_87769

/--
A prism is defined to be a right prism if its lateral edges are perpendicular to the base and its base is a regular polygon.
A rectangular prism is a parallelepiped where all faces are rectangles.
A regular hexagonal pyramid is a pyramid whose base is a regular hexagon and all lateral edges are congruent.
A right square prism is a rectangular prism where all edges of the base are equal and the lateral edges are perpendicular to the base.

Define the four propositions:
1. A prism whose lateral faces are all congruent quadrilaterals must be a right prism.
2. A hexahedron whose opposite faces are congruent rectangles must be a rectangular prism.
3. If the lateral edges of a pyramid are equal in length to the sides of its base polygon, then the pyramid could be a regular hexagonal pyramid.
4. A rectangular prism must be a right square prism.

Prove that the number of correct propositions from the above four statements is 0.
-/
theorem number_of_correct_propositions_is_zero :
  (∀ (P1 P2 P3 P4 : Prop),
    (P1 ↔ (∀ (prism : Type) (base : prism) (lat_faces : list prism),
              (∀ lat_face ∈ lat_faces, congruent lat_face (lat_faces.head)) →
                is_right_prism prism → False)) ∧
    (P2 ↔ (∀ (hexahedron : Type) (opposite_faces : list hexahedron),
              (∀ opp_face ∈ opposite_faces, congruent opp_face (opposite_faces.head)) →
                is_rectangular_prism hexahedron → False)) ∧
    (P3 ↔ (∀ (pyramid : Type) (base_sides lat_edges : list pyramid),
              (∀ lat_edge ∈ lat_edges, congruent lat_edge (lat_edges.head)) ∧
                (∀ base_side ∈ base_sides, congruent base_side (base_sides.head)) →
                  is_regular_hexagonal_pyramid pyramid → False)) ∧
    (P4 ↔ (∀ (rect_prism : Type),
              is_rectangular_prism rect_prism ∧ is_right_square_prism rect_prism → False)) →
       (P1 = False ∧ P2 = False ∧ P3 = False ∧ P4 = False)) := by
    intros
    apply And.intro
    . -- Proof for P1
      intro P1_def
      specialize P1_def ℕ [1, 2, 3, 4] [1, 1, 1, 1]
      have : False := P1_def rfl
      exact this
    . apply And.intro
      . -- Proof for P2
        intro P2_def
        specialize P2_def ℕ [1, 1, 1, 1] [2, 2, 2, 2]
        have : False := P2_def rfl
        exact this
      . apply And.intro
        . -- Proof for P3
          intro P3_def
          specialize P3_def ℕ [1, 1] [2, 2]
          have : False := P3_def rfl
          exact this
        . -- Proof for P4
          intro P4_def
          specialize P4_def ℕ
          have : False := P4_def rfl
          exact this

end number_of_correct_propositions_is_zero_l87_87769


namespace eventually_increasing_s_l87_87349

-- Define s_n as the sum of digits of 2^n in decimal expansion
def sum_of_digits (n : ℕ) : ℕ :=
  let str := n.digits 10
  List.sum str

def s (n : ℕ) : ℕ := sum_of_digits (2^n)

theorem eventually_increasing_s : ∃ N : ℕ, ∀ n : ℕ, n > N → s(n+1) > s(n) := by
  sorry

end eventually_increasing_s_l87_87349


namespace tangent_line_minimum_value_l87_87036

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem tangent_line (a : ℝ) (h : a = -1) :
  let f := λ x, x + Real.log x in
  let f' := λ x, 1 + 1 / x in
  f 1 = 1 ∧ f' 1 = 2 ∧ (∀ x, f 1 + f' 1 * (x - 1) = 2 * x - 1) :=
by
  intros
  sorry

theorem minimum_value (a : ℝ) :
  (a ≤ 1 → ∀ x ∈ Set.Icc 1 Real.exp 1, f x a = 1) ∧
  (1 < a ∧ a < Real.exp 1 → ∀ x ∈ Set.Icc 1 Real.exp 1, f (a : ℝ) = a - a * Real.log a) ∧
  (a ≥ Real.exp 1 → ∀ x ∈ Set.Icc 1 Real.exp 1, f Real.exp 1 a = Real.exp 1 - a) :=
by
  intros
  sorry

end tangent_line_minimum_value_l87_87036


namespace coordinates_of_point_A_l87_87392

theorem coordinates_of_point_A (x y : ℝ) (hx1 : y > 0) (hx2 : x > 0) (h3 : y = 2) (h4 : x = 4) : (x, y) = (4, 2) :=
by
  rw [h3, h4]
  refl

end coordinates_of_point_A_l87_87392


namespace bucket_capacity_l87_87983

theorem bucket_capacity (x : ℝ) (h1 : 24 * x = 36 * 9) : x = 13.5 :=
by 
  sorry

end bucket_capacity_l87_87983


namespace minor_axis_length_of_ellipse_l87_87284

theorem minor_axis_length_of_ellipse (h : set.univ = {(-2, 1), (0,0), (0,3), (4,0), (4,3)}) :
  minor_axis_length_of_ellipse' (set.to_finset {p : ℝ × ℝ | p = (-2,1) ∨ p = (0,0) ∨ p = (0,3) ∨ p = (4,0) ∨ p = (4,3)}) = 2 * real.sqrt 3 :=
begin
  -- Only the statement is required
  sorry
end

end minor_axis_length_of_ellipse_l87_87284


namespace num_triangles_correct_num_lines_correct_l87_87140

-- Definition for the first proof problem: Number of triangles
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem num_triangles_correct :
  num_triangles 9 = 84 :=
by
  sorry

-- Definition for the second proof problem: Number of lines
def num_lines (n : ℕ) : ℕ := Nat.choose n 2

theorem num_lines_correct :
  num_lines 9 = 36 :=
by
  sorry

end num_triangles_correct_num_lines_correct_l87_87140


namespace probability_greater_than_4_l87_87256

theorem probability_greater_than_4 :
  let total_faces := 6
  let successful_faces := 2
  (successful_faces : ℚ) / total_faces = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l87_87256


namespace salaries_proof_l87_87690

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof_l87_87690


namespace part1_part2_part3_l87_87413

def f (x : ℝ) (m : ℝ) : ℝ := x^m - 2/x

theorem part1 : f 4 m = 7/2 → m = 1 := sorry

theorem part2 : ∀ x : ℝ, f (-x) 1 = - f x 1 := sorry

theorem part3 : ∀ x₁ x₂ : ℝ, 0 < x₂ ∧ x₂ < x₁ → f x₁ 1 > f x₂ 1 := sorry

end part1_part2_part3_l87_87413


namespace trajectory_of_P_constant_dot_product_l87_87757

variable {a : ℝ} (a_pos : a > 0)

-- Define the right focus of the ellipse
def F := (a, 0) : ℝ × ℝ

-- Define moving points M and N on the axes
variable {m n : ℝ}
def M := (m, 0) : ℝ × ℝ
def N := (0, n) : ℝ × ℝ

-- Dot product condition
def MN := (0 - m, n - 0) : ℝ × ℝ
def NF := (a - 0, 0 - n) : ℝ × ℝ
def dot_product_zero := (0 - m) * (a - 0) + (n - 0) * (0 - n) = 0

-- Point condition
def OM := (m, 0) : ℝ × ℝ
def ON := (0, n) : ℝ × ℝ
variable {P_x P_y : ℝ}
def P := (P_x, P_y) : ℝ × ℝ
def point_condition := (m, 0) = (2 * (0, n)) + (P_x, P_y)

-- First proof problem: Prove the trajectory of point P
theorem trajectory_of_P : P_y = -a * P_x :=
by sorry

-- Second proof problem: Prove the dot product constant value
def S := (-a, a^2) : ℝ × ℝ
def T := (-a, a^2) : ℝ × ℝ

def FS := (-a - a, a^2 - 0) : ℝ × ℝ
def FT := (-a - a, a^2 - 0) : ℝ × ℝ

theorem constant_dot_product :
  let FS_dot_FT := (FS.1 * FT.1) + (FS.2 * FT.2)
  in FS_dot_FT = 4 * a^2 + a^4 :=
by sorry

end trajectory_of_P_constant_dot_product_l87_87757


namespace maximum_area_triangle_l87_87842

theorem maximum_area_triangle 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : ∀ A B : ℝ, ∀ a b : ℝ, (\frac{cos A}{sin B} + \frac{cos B}{sin A}) = 2)
  (h₃ : a + b + c = 12)
  : let area := \frac{1}{2} * a * b
    ∃ max_area : ℝ, max_area = 36 * (3 - 2 * sqrt 2)
    := sorry

end maximum_area_triangle_l87_87842


namespace prob_gt_4_eq_prob_le_2_l87_87593

-- Definitions of the conditions
def outcomes : finset ℕ := {1, 2, 3, 4, 5, 6}
def outcomes_gt_4 : finset ℕ := {5, 6}
def outcomes_le_2 : finset ℕ := {1, 2}
def total_outcomes := (outcomes).card
def prob_gt_4 : ℚ := outcomes_gt_4.card / total_outcomes
def prob_le_2 : ℚ := outcomes_le_2.card / total_outcomes

-- The proof statement
theorem prob_gt_4_eq_prob_le_2 (h : total_outcomes = 6) : prob_gt_4 = prob_le_2 :=
by sorry

end prob_gt_4_eq_prob_le_2_l87_87593


namespace new_average_age_l87_87537

theorem new_average_age (avg_age : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_num_individuals : ℕ) (new_avg_age : ℕ) :
  avg_age = 15 ∧ num_students = 20 ∧ teacher_age = 36 ∧ new_num_individuals = 21 →
  new_avg_age = (num_students * avg_age + teacher_age) / new_num_individuals → new_avg_age = 16 :=
by
  intros
  sorry

end new_average_age_l87_87537


namespace relationship_abc_l87_87754

noncomputable def a : ℝ := 2^0.2
noncomputable def b : ℝ := 2^0.3
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem relationship_abc : c < a ∧ a < b :=
by
  sorry

end relationship_abc_l87_87754


namespace increasing_exponential_function_range_l87_87402

theorem increasing_exponential_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ (x : ℝ), f x = a ^ x) 
    (h2 : a > 0)
    (h3 : a ≠ 1)
    (h4 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) : a > 1 := 
sorry

end increasing_exponential_function_range_l87_87402


namespace gcd_of_459_and_357_l87_87982

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_of_459_and_357_l87_87982


namespace allocation_schemes_correct_l87_87343

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l87_87343


namespace problem1_problem2_problem3_l87_87825

-- Definition of "short distance"
def short_distance (x y : ℝ) : ℝ :=
  min (abs x) (abs y)

-- Equidistant points definition
def equidistant (P Q : ℝ × ℝ) : Prop :=
  short_distance P.1 P.2 = short_distance Q.1 Q.2

-- Problem 1: Prove that the "short distance" of point A(-5, -2) is 2.
theorem problem1 : short_distance (-5) (-2) = 2 :=
  sorry

-- Problem 2: Prove that if the "short distance" of point B(-2, -2m+1) is 1, then m is 1 or 0.
theorem problem2 (m : ℝ) (h : short_distance (-2) (-2 * m + 1) = 1) : m = 1 ∨ m = 0 :=
  sorry

-- Problem 3: Prove that if points C(-1, k+3) and D(4, 2k-3) are equidistant points, then k is 1 or 2.
theorem problem3 (k : ℝ) (h : equidistant (-1, k + 3) (4, 2 * k - 3)) : k = 1 ∨ k = 2 :=
  sorry

end problem1_problem2_problem3_l87_87825


namespace base_b_of_100_has_five_digits_l87_87262

theorem base_b_of_100_has_five_digits : 
  ∃ b : ℕ, (b^4 ≤ 100 ∧ 100 < b^5) ∧ b = 3 :=
by
  use 3
  split
  sorry

end base_b_of_100_has_five_digits_l87_87262


namespace angle_BSC_eq_angle_ASD_l87_87088

-- Definitions for the problem
variables {A B C D O S : Type} [Trapezoid ABCD] [Circumcircle OCD] 
variable (h_CD_perpendicular : ∀ {AB AD}, Perpendicular CD AB AD)
variable (h_O_intersection : O = (Intersection AC BD))
variable (h_S_diametrically_opposite : S = (DiametricallyOpposite O (Circumcircle OCD)))

-- The goal to prove
theorem angle_BSC_eq_angle_ASD 
  (h_CD_perpendicular : Perpendicular CD AB AD)
  (h_O_intersection : O = (Intersection AC BD))
  (h_S_diametrically_opposite : S = (DiametricallyOpposite O (Circumcircle OCD))) :
  ∠ BSC = ∠ ASD :=
by {
  sorry,
}

end angle_BSC_eq_angle_ASD_l87_87088


namespace sufficient_but_not_necessary_l87_87907

theorem sufficient_but_not_necessary (a : ℝ) : (a > 1) → (a > 0) ∧ ¬ (∀ a, a > 0 → a > 1) :=
by
  intro h1
  split
  { exact lt_of_lt_of_le h1 (le_of_lt h1) }
  { intro h2
    have : ¬ (∀ a, a > 0 → a > 1) := by
      use 0.5
      split
      { linarith }
      { linarith }
    exact this }

end sufficient_but_not_necessary_l87_87907


namespace gcd_8m_12n_l87_87734

noncomputable theory

theorem gcd_8m_12n (m n : ℕ) (h1 : Nat.gcd m n = 18) (hm : m > 0) (hn : n > 0) : Nat.gcd (8 * m) (12 * n) = 72 :=
sorry

end gcd_8m_12n_l87_87734


namespace sqrt_6_irrational_between_2_and_3_l87_87601

theorem sqrt_6_irrational_between_2_and_3:
  irrational (Real.sqrt 6) ∧ 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
  sorry

end sqrt_6_irrational_between_2_and_3_l87_87601


namespace g_of_neg_four_g_of_six_l87_87867

def g (x : ℝ) : ℝ :=
  if x < 0 then
    5 * x - 2
  else
    12 - 3 * x

theorem g_of_neg_four : g (-4) = -22 :=
by {
  have h1 : -4 < 0 := by linarith,
  simp [g, h1],
  norm_num,
  sorry
}

theorem g_of_six : g (6) = -6 :=
by {
  have h2 : 6 ≥ 0 := by linarith,
  simp [g, h2],
  norm_num,
  sorry
}

end g_of_neg_four_g_of_six_l87_87867


namespace max_area_of_triangle_ABC_l87_87844

open Classical

theorem max_area_of_triangle_ABC (A B C : ℝ) :
  ( ∀ (α β γ a b c : ℝ), 
      α + β = 90 ∧ a + b + c = 12 ∧ 
      (α + β + γ = 180) ∧ (c = sqrt (a ^ 2 + b ^ 2)) ∧ 
      γ = 90 ∧ 
      (cos(α) / sin(β) + cos(β) / sin(α)) = 2 
      → 
      ∃ area : ℝ, 0 ≤ area ∧ area = 36 * (3 - 2 * sqrt 2)
  )

end max_area_of_triangle_ABC_l87_87844


namespace problem_statement_l87_87493

theorem problem_statement (a b : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : ∀ n : ℕ, n ≥ 1 → 2^n * b + 1 ∣ a^(2^n) - 1) : a = 1 := by
  sorry

end problem_statement_l87_87493


namespace probability_of_drawing_2_ones_probability_distribution_of_X_expected_value_of_X_l87_87566

namespace BallDrawing

-- Define the problem conditions
def balls : Finset ℕ := {1, 1, 2, 3}
def drawTwo (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ t, t.card = 2)

-- Define the random variable X
def X (s : Finset ℕ) : ℕ := s.sum

-- Define the probability of drawing 2 balls labeled with number 1
theorem probability_of_drawing_2_ones : 
(∃ s ∈ drawTwo balls, s = {1, 1}) → (1 / (drawTwo balls).card) = 1 / 6 :=
by
  sorry

-- Define the probability distribution of X
theorem probability_distribution_of_X :
(∀ (x : ℕ), x ∈ {2, 3, 4, 5} → 
(∃ s ∈ drawTwo balls, X s = x) → 
(1 / (drawTwo balls).card)) :=
by
  sorry

-- Define the expected value of X
theorem expected_value_of_X : ∑ x in {2, 3, 4, 5}, x * (1 / (drawTwo balls).card) = 7 / 2 :=
by
  sorry

end BallDrawing

end probability_of_drawing_2_ones_probability_distribution_of_X_expected_value_of_X_l87_87566


namespace line_polar_to_cartesian_circle_polar_to_cartesian_chord_length_l87_87007

theorem line_polar_to_cartesian 
  (rho theta : ℝ) 
  (h_line : ∀ rho theta, rho * sin (theta - 2 * π / 3) = -√3) :
  ∃ x y : ℝ, sqrt 3 * x + y = 2 * sqrt 3 :=
sorry

theorem circle_polar_to_cartesian 
  (rho theta : ℝ) 
  (h_circle : ∀ rho theta, rho = 4 * cos theta + 2 * sin theta) :
  ∃ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y = 0 :=
sorry

theorem chord_length 
  (c : ℝ × ℝ := (2, 1)) (r d : ℝ := sqrt 5) 
  (h_line: sqrt 3 * c.1 + c.2 = 2 * sqrt 3) (h_distance: d = 1 / 2) :
  ∃ l : ℝ, l = sqrt 19 :=
sorry

end line_polar_to_cartesian_circle_polar_to_cartesian_chord_length_l87_87007


namespace minimum_value_real_l87_87230

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l87_87230


namespace coefficient_x2y2_expansion_l87_87906

theorem coefficient_x2y2_expansion (x y : ℝ) :
  (let t1 := (1 + x) ^ 7 in
   let t2 := (1 + y) ^ 4 in
   let expansion := t1 * t2 in
   coefficient_of_term expansion (x^2 * y^2)) = 126 :=
begin
  sorry
end

end coefficient_x2y2_expansion_l87_87906


namespace smallest_number_tens_place_digit_l87_87908

theorem smallest_number_tens_place_digit :
  ∃ (n : ℕ), (∃ (a b c d e : ℕ),
              n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
              n % 2 = 0 ∧
              (a + b + c + d + e) % 3 = 0 ∧
              List.perm [a, b, c, d, e] [1, 2, 3, 4, 9] ∧
              n = 1 * 10000 + 4 * 1000 + 9 * 100 + 3 * 10 + 2) ∨ 
              n = 1 * 10000 + 2 * 1000 + 9 * 100 + 4 * 10 + 3) ∨ 
              n = 1 * 10000 + 2 * 1000 + 4 * 100 + 9 * 10 + 3 ∧
              d ≠ 3,
  d = 3 := 
sorry

end smallest_number_tens_place_digit_l87_87908


namespace product_of_inradii_ratio_is_six_l87_87321

noncomputable def inscribedSphereRadiusCube (a : ℝ) : ℝ :=
  a / 2

noncomputable def inscribedSphereRadiusOctahedron (a : ℝ) : ℝ :=
  a / Real.sqrt 6

theorem product_of_inradii_ratio_is_six (a : ℝ) (h_a_pos : 0 < a) : ∃ m n : ℕ, ((Real.sqrt 6) / 2).isReducedFraction m n ∧ m * n = 6 :=
by
  let r1 := inscribedSphereRadiusCube a
  let r2 := inscribedSphereRadiusOctahedron a
  have h_rat : (r1 / r2) = (Real.sqrt 6) / 2 :=
    by
      rw [inscribedSphereRadiusCube, inscribedSphereRadiusOctahedron]
      field_simp [Real.sqrt_ne_zero_iff.mpr (by norm_num : (6:ℝ) ≠ 0)]
  obtain ⟨m, n, h_mn⟩ := Real.isReducedFraction_of_eq_theorem h_rat
  use [m, n]
  sorry

end product_of_inradii_ratio_is_six_l87_87321


namespace find_a_for_even_function_l87_87873

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, f(x) = x * (exp x + a * exp (-x)) ∧ f(x) = f(-x)) → a = -1 := by
  intro h
  sorry

end find_a_for_even_function_l87_87873


namespace floor_expression_equals_zero_l87_87685

theorem floor_expression_equals_zero
  (a b c : ℕ)
  (ha : a = 2010)
  (hb : b = 2007)
  (hc : c = 2008) :
  Int.floor ((a^3 : ℚ) / (b * c^2) - (c^3 : ℚ) / (b^2 * a)) = 0 := 
  sorry

end floor_expression_equals_zero_l87_87685


namespace Taso_riddles_correct_l87_87473

-- Definitions based on given conditions
def Josh_riddles : ℕ := 8
def Ivory_riddles : ℕ := Josh_riddles + 4
def Taso_riddles : ℕ := 2 * Ivory_riddles

-- The theorem to prove
theorem Taso_riddles_correct : Taso_riddles = 24 := by
  sorry

end Taso_riddles_correct_l87_87473


namespace speed_of_train_l87_87987

-- Define the given constants
def train_length : ℝ := 770.0
def crossing_time : ℝ := 62.994960403167745
def man_speed_km_hr : ℝ := 2.5

-- Convert the man's speed from km/hr to m/s
def man_speed_m_s := man_speed_km_hr * 1000 / 3600

-- Define the condition of the problem
def relative_speed := train_length / crossing_time

-- Define the speed of the train
def train_speed_m_s := relative_speed + man_speed_m_s

-- Convert the train's speed to km/hr
def train_speed_km_hr := train_speed_m_s * 3600 / 1000

-- The theorem we need to prove
theorem speed_of_train : train_speed_km_hr = 46.4992 := by
  sorry

end speed_of_train_l87_87987


namespace not_difference_of_squares_10_l87_87595

theorem not_difference_of_squares_10 (a b : ℤ) : a^2 - b^2 ≠ 10 :=
sorry

end not_difference_of_squares_10_l87_87595


namespace triangle_GP_length_l87_87089

noncomputable def length_GP (AB AC BC : ℝ) (AD BE CF : ℝ) (G P : ℝ) : ℝ :=
  -- Function definition
  sorry

theorem triangle_GP_length
  (AB AC BC : ℝ) (h1 : AB = 10) (h2 : AC = 14) (h3 : BC = 16)
  (AD BE CF : ℝ) (G P : ℝ)
  (h4 : true) -- Placeholder for medians intersecting at G
  (h5 : true) -- Placeholder for P being the foot of the altitude from G to BC
  :
  length_GP AB AC BC AD BE CF G P = 5 * real.sqrt 3 / 3 :=
sorry

end triangle_GP_length_l87_87089


namespace segment_parallel_divides_area_half_eq_l87_87196

theorem segment_parallel_divides_area_half_eq (ABCD : Type) 
  (M N : ABCD) (MN CD : ℝ) 
  (parallel : MN || CD) 
  (half_areas : 2 * (area ABCD = area (Region ABCD divided at MN)))
  (M_on_BC : M ∈ BC) 
  (N_on_AD : N ∈ AD) 
  (a b c : ℝ)
  (section_A_parallel_CD : from A parallel to CD at BC = a) 
  (section_B_parallel_CD : from B parallel to CD at AD = b) 
  (length_CD : CD = c) :
  MN^2 = (a * b + c^2) / 2 := 
sorry

end segment_parallel_divides_area_half_eq_l87_87196


namespace converse_and_inverse_l87_87695

-- Definitions
def is_circle (s : Type) : Prop := sorry
def has_no_corners (s : Type) : Prop := sorry

-- Converse Statement
def converse_false (s : Type) : Prop :=
  has_no_corners s → is_circle s → False

-- Inverse Statement
def inverse_true (s : Type) : Prop :=
  ¬ is_circle s → ¬ has_no_corners s

-- Main Proof Problem
theorem converse_and_inverse (s : Type) :
  (converse_false s) ∧ (inverse_true s) := sorry

end converse_and_inverse_l87_87695


namespace sum_of_geography_and_english_l87_87788

theorem sum_of_geography_and_english (G E : ℕ) : 
  let M := 70 in
  let H := (G + M + E) / 3 in
  let total_score := G + M + E + H in
  total_score = 248 → G + E = 116 :=
by
  intro h1,
  sorry

end sum_of_geography_and_english_l87_87788


namespace find_x_l87_87382

-- Define the angles AXB, CYX, and XYB as given in the problem.
def angle_AXB : ℝ := 150
def angle_CYX : ℝ := 130
def angle_XYB : ℝ := 55

-- Define a function that represents the sum of angles in a triangle.
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- Define the angles.
def angle_XYZ : ℝ := angle_AXB - angle_XYB
def angle_YXZ : ℝ := 180 - angle_CYX
def angle_YXZ_proof (x : ℝ) : Prop := sum_of_angles_in_triangle angle_XYZ angle_YXZ x

-- State the theorem to be proved.
theorem find_x : angle_YXZ_proof 35 :=
sorry

end find_x_l87_87382


namespace find_y_l87_87273

noncomputable def y_value : ℝ :=
  let x1 := 1
      y1 := 3
      x2 := -7
      dist := 12 in
  let eq := dist = Real.sqrt ((x2 - x1)^2 + (y_value - y1)^2) in
  let y := 3 + 4 * Real.sqrt 5 in
  if y > 0 then y else sorry

theorem find_y :
  ∃ y : ℝ, (dist: ℕ ) -> y > 0 ∧ 
  (Real.sqrt ((-7-1)^2 + (y - 3)^2) = 12) ∧
  y = 3 + 4 * Real.sqrt 5 :=
begin
  use 3 + 4 * Real.sqrt 5,
  intro dist,
  split,
  { linarith [Real.sqrt 5] },
  { sorry } -- Detailed proof to be provided later
end

end find_y_l87_87273


namespace balloon_distinct_arrangements_l87_87049

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_distinct_arrangements : 
  let total_letters := 7
  let freq_B := 1
  let freq_A := 1
  let freq_L := 2
  let freq_O := 2
  let freq_N := 1
  total_letters! / (freq_L! * freq_O!) = 1260 := by
    have total_letters := 7
    have freq_L := 2
    have freq_O := 2
    have h1 : total_letters! = 5040 := by sorry
    have h2 : freq_L! = 2 := by sorry
    have h3 : freq_O! = 2 := by sorry
    show total_letters! / (freq_L! * freq_O!) = 1260
    calc
      total_letters! / (freq_L! * freq_O!)
          = 5040 / (2 * 2) : by rw [h1, h2, h3]
      ... = 5040 / 4 : by rw mul_comm
      ... = 1260 : by norm_num

end balloon_distinct_arrangements_l87_87049


namespace log_sqrt_defined_in_interval_l87_87364

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l87_87364


namespace dice_sum_probability_l87_87574

theorem dice_sum_probability :
  let outcomes := 36 in
  let favorable := 26 in -- Total number of pairs where the sum is less than 9
  (favorable.toRat / outcomes.toRat) = (13 / 18) :=
by sorry

end dice_sum_probability_l87_87574


namespace max_torque_l87_87953

theorem max_torque
  (r : Real := 0.1) 
  (d : Real := 0.05) 
  (Q : Real := 10 ^ (-3)) 
  (k : Real := 8.99 * 10^9):
  ∃ (M_max : Real), M_max = 1.36 * 10^5 := by
  sorry

end max_torque_l87_87953


namespace area_of_triangle_ABC_l87_87469

noncomputable def area_triangle_ABC (AF BE : ℝ) (angle_FGB : ℝ) : ℝ :=
  let FG := AF / 3
  let BG := (2 / 3) * BE
  let area_FGB := (1 / 2) * FG * BG * Real.sin angle_FGB
  6 * area_FGB

theorem area_of_triangle_ABC
  (AF BE : ℕ) (hAF : AF = 10) (hBE : BE = 15)
  (angle_FGB : ℝ) (h_angle_FGB : angle_FGB = Real.pi / 3) :
  area_triangle_ABC AF BE angle_FGB = 50 * Real.sqrt 3 :=
by
  simp [area_triangle_ABC, hAF, hBE, h_angle_FGB]
  sorry

end area_of_triangle_ABC_l87_87469


namespace maximum_m_l87_87857

theorem maximum_m (a b c : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b + c = 10)
  (h₅ : a * b + b * c + c * a = 25) :
  ∃ m, (m = min (a * b) (min (b * c) (c * a)) ∧ m = 25 / 9) :=
sorry

end maximum_m_l87_87857


namespace henry_walk_duration_l87_87047

noncomputable def duration_of_walk (start end : Nat) : Nat :=
  let start_min := (8 * 60) + 44
  let end_min := (14 * 60) + 40
  (end_min - start_min) / 60

theorem henry_walk_duration : duration_of_walk ((8 * 60) + 44) ((14 * 60) + 40) = 6 := by
  sorry

end henry_walk_duration_l87_87047


namespace exists_a_decreasing_l87_87065

-- Define the function f(x) = ax^2 + 1/x
def f (a x : ℝ) : ℝ := a * x^2 + 1 / x

-- Statement of the problem: Prove that there exists a ∈ ℝ such that f(x) is decreasing on (0, +∞)
theorem exists_a_decreasing : ∃ (a : ℝ), ∀ (x : ℝ), x > 0 → (∀ (y : ℝ), 0 < y ∧ y < x → f a y > f a x) :=
sorry

end exists_a_decreasing_l87_87065


namespace max_area_of_triangle_ABC_l87_87843

open Classical

theorem max_area_of_triangle_ABC (A B C : ℝ) :
  ( ∀ (α β γ a b c : ℝ), 
      α + β = 90 ∧ a + b + c = 12 ∧ 
      (α + β + γ = 180) ∧ (c = sqrt (a ^ 2 + b ^ 2)) ∧ 
      γ = 90 ∧ 
      (cos(α) / sin(β) + cos(β) / sin(α)) = 2 
      → 
      ∃ area : ℝ, 0 ≤ area ∧ area = 36 * (3 - 2 * sqrt 2)
  )

end max_area_of_triangle_ABC_l87_87843


namespace find_range_of_a_l87_87021

noncomputable def A (a : ℝ) := { x : ℝ | 1 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = 5 * x - 6 ∧ 1 ≤ x ∧ x ≤ a }
noncomputable def C (a : ℝ) := { m : ℝ | ∃ x : ℝ, m = x^2 ∧ 1 ≤ x ∧ x ≤ a }

theorem find_range_of_a (a : ℝ) (h : B a ∩ C a = C a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end find_range_of_a_l87_87021


namespace specific_line_equation_l87_87190

theorem specific_line_equation :
  ∃ (m : ℝ) (b : ℝ), 
    (∀ (x y : ℝ), y = m * x + b ↔ x + y - 3 = 0) ∧ 
    (∀ (x y : ℝ), y = -1 * x + b ↔ x - y - 3 = 0) ∧ 
    (t : Type*) (hx : x = 2) (hy : y = -1), y = m * x + b :=
begin
  -- prove the slope m is -1 for the first line
  -- prove the slope of the perpendicular line to it is indeed 1
  -- use point-slope form to derive the equation and simplify
  sorry
end

end specific_line_equation_l87_87190


namespace new_boarders_count_l87_87560

def initial_boarders : ℕ := 330
def initial_ratio : ℚ := 5 / 12
def new_ratio : ℚ := 1 / 2
def initial_day_students (b : ℕ) (r : ℚ) : ℕ := b * (r.denom) / (r.num)

-- Given conditions
def day_students := initial_day_students initial_boarders initial_ratio

def final_boarders (initial : ℕ) (new : ℕ) : ℕ := initial + new

theorem new_boarders_count 
  (initial_b : ℕ := initial_boarders)
  (initial_r : ℚ := initial_ratio)
  (new_r : ℚ := new_ratio)
  (initial_days : ℕ := day_students)
  (final_days : ℕ := initial_days)
  (new_b : ℕ := 66) :
  final_boarders initial_b new_b * new_r.num = final_days * new_r.denom :=
by
  sorry

end new_boarders_count_l87_87560


namespace geometric_figures_prop_l87_87423

-- Definition of perpendicular and parallel relationships
variable {α : Type} [normed_add_comm_group α] [normed_space ℝ α]

def is_perpendicular (x y : α) : Prop := ∃ z, z ∈ orthogonal_projection 𝕜 := topological_space.closed

def is_parallel (x y : α) : Prop := ∃ k : ℝ, x = k • y

-- Given conditions and conclusion
def geometric_figures (x y z : α) (cond1 cond2 cond3 cond4 : Prop) : Prop :=
  (cond1 ∧ is_perpendicular x y ∧ is_parallel y z → is_perpendicular x z) ∧
  (cond2 ∧ is_perpendicular x y ∧ is_parallel y z → is_perpendicular x z) ∧
  (cond3 ∧ is_perpendicular x y ∧ is_parallel y z → is_perpendicular x z) ∧
  (cond4 ∧ is_perpendicular x y ∧ is_parallel y z → is_perpendicular x z)

-- Conditions for 4 scenarios
axiom cond1 : Prop -- All lines
axiom cond2 : Prop -- All planes
axiom cond3 : Prop -- x, y are lines; z is a plane
axiom cond4 : Prop -- x, z are planes; y is a line

-- Proving the correctness of the proposition for the given problem statement
theorem geometric_figures_prop (x y z : α) :
  geometric_figures x y z cond1 cond2 false cond4 := 
sorry

end geometric_figures_prop_l87_87423


namespace determinant_roots_l87_87864

theorem determinant_roots (s p q a b c : ℂ) 
  (h : ∀ x : ℂ, x^3 - s*x^2 + p*x + q = (x - a) * (x - b) * (x - c)) :
  (1 + a) * ((1 + b) * (1 + c) - 1) - ((1) * (1 + c) - 1) + ((1) - (1 + b)) = p + 3 * s :=
by {
  -- expanded determinant calculations
  sorry
}

end determinant_roots_l87_87864


namespace part1_short_distance_A_part2_short_distance_B_part3_equidistant_C_D_l87_87827

-- Define the "short distance" of a point
def short_distance (x y : ℝ) : ℝ :=
  min (|x|) (|y|)

-- Define equidistant points
def equidistant_points (P Q : ℝ × ℝ) : Prop :=
  short_distance (P.1) (P.2) = short_distance (Q.1) (Q.2)

-- Prove the parts of the problem
theorem part1_short_distance_A :
  short_distance (-5) (-2) = 2 := 
sorry

theorem part2_short_distance_B (m : ℝ) :
  short_distance (-2) (-2 * m + 1) = 1 ↔ m = 0 ∨ m = 1 :=
sorry

theorem part3_equidistant_C_D (k : ℝ) :
  equidistant_points (-1, k + 3) (4, 2 * k - 3) ↔ k = 1 ∨ k = 2 := 
sorry

end part1_short_distance_A_part2_short_distance_B_part3_equidistant_C_D_l87_87827


namespace calculate_P_AB_l87_87753

section Probability
-- Define the given probabilities
variables (P_B_given_A : ℚ) (P_A : ℚ)
-- Given conditions
def given_conditions := P_B_given_A = 3/10 ∧ P_A = 1/5

-- Prove that P(AB) = 3/50
theorem calculate_P_AB (h : given_conditions P_B_given_A P_A) : (P_A * P_B_given_A) = 3/50 :=
by
  rcases h with ⟨h1, h2⟩
  simp [h1, h2]
  -- Here we would include the steps leading to the conclusion; this part just states the theorem
  sorry

end Probability

end calculate_P_AB_l87_87753


namespace main_theorem_l87_87393

-- Definitions of the conditions from a)
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_x_axis (f : ℝ×ℝ) : Prop := f.snd = 0
def eccentricity (e : ℝ) : Prop := e = 1 / 2
def passes_through_point (x y : ℝ) : Prop := (x, y) = (2, 3)
def line_passing_through_focus (F : ℝ×ℝ) (f : ℝ) : Prop := F.fst = -f ∧ F.snd = 0

-- Statements to be proved
def equation_of_ellipse : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ,
    (passes_through_point x y ∧ eccentricity (1/2) ∧ center_at_origin 0 0) →
    (x^2 / a^2 + y^2 / b^2 = 1)

def range_of_area_S : Prop :=
  ∃ S : set ℝ, S = set.Ioo (9 / 4) 3 ∧ 
  (∀ P F1 G x y : ℝ × ℝ,
    eccentricity (1/2) ∧ passes_through_point P.1 P.2 ∧ center_at_origin 0 0 ∧ focus_on_x_axis F1 →
    line_passing_through_focus F1 x →
    G = intersection_xaxis_perpendicular_bisector P F1 (x,y) →
    S.contains (area_triangle P F1 G))

-- Main theorem to connect conditions with the outcomes
theorem main_theorem : equation_of_ellipse ∧ range_of_area_S :=
begin
  split,
  {   -- Prove the equation of the ellipse
      sorry },
  {   -- Prove the range of the area S of ΔPF₁G
      sorry }
end

end main_theorem_l87_87393


namespace trigonometric_simplification_l87_87892

theorem trigonometric_simplification (x : ℝ) (h : ∀ θ : ℝ, Real.cot θ - 2 * Real.cot (2 * θ) = Real.tan θ) : 
  Real.tan x + 2 * Real.tan (2 * x) + 4 * Real.tan (4 * x) + 8 * Real.tan (8 * x) + 16 * Real.cot (16 * x) = Real.cot x :=
by
  sorry

end trigonometric_simplification_l87_87892


namespace allocation_schemes_l87_87337

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l87_87337


namespace sum_of_possible_b_coefficient_l87_87136

theorem sum_of_possible_b_coefficient (b : ℤ) :
  (∃ r s : ℤ, r > 0 ∧ s > 0 ∧ r * s = 48 ∧ ∀ a b : ℤ, (x + a) * (x + b) = x^2 + b * x + 48 ∧ b = r + s) →
  ∑ (r, s : ℤ) in (46, 2), (48, 1), (24, 2), (16, 3), (12, 4), (8, 6), r + s = 124 :=
sorry

end sum_of_possible_b_coefficient_l87_87136


namespace negative_value_option_D_l87_87964

theorem negative_value_option_D :
  (-7) * (-6) > 0 ∧
  (-7) - (-15) > 0 ∧
  0 * (-2) * (-3) = 0 ∧
  (-6) + (-4) < 0 :=
by
  sorry

end negative_value_option_D_l87_87964


namespace neq_p_to_neq_q_sufficient_not_necessary_l87_87749

theorem neq_p_to_neq_q_sufficient_not_necessary (x : ℝ) :
  (|x + 1| > 2) → (5x - 6 > x^2) → 
  (∀ x, ¬ (|x + 1| > 2) → ¬ (5x - 6 > x^2)) :=
sorry


end neq_p_to_neq_q_sufficient_not_necessary_l87_87749


namespace digit_at_100_l87_87220

/-
  Define the sequence based on the given conditions.
  We use a list to represent the sequence.
-/

def sequence_digit (n : ℕ) : ℕ :=
  -- Define a helper function for the count of elements in each group
  let group_count (k : ℕ) := k
  -- Determine the group that contains the n-th element
  let rec find_group (i : ℕ) (sum : ℕ) : ℕ :=
    if sum + group_count i >= n then i else find_group (i + 1) (sum + group_count i)
  let group := find_group 1 0
  -- Determine the position within the identified group
  let pos_in_group := n - (group * (group - 1)) / 2
  if group % 5 == 1 then 1
  else if group % 5 == 2 then 2
  else if group % 5 == 3 then 3
  else if group % 5 == 4 then 4
  else 5

theorem digit_at_100 : sequence_digit 100 = 4 := by
  sorry

end digit_at_100_l87_87220


namespace simplify_polynomial_subtraction_l87_87168

/--
  Given the polynomials (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) and (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5),
  prove that their difference simplifies to x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3.
-/
theorem simplify_polynomial_subtraction  (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) = x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 :=
sorry

end simplify_polynomial_subtraction_l87_87168


namespace triangle_PQR_area_zero_l87_87682

-- Definitions for centers and radii of circles
def CircleCenterP := (ℝ × ℝ) 
def CircleCenterQ := (ℝ × ℝ) 
def CircleCenterR := (ℝ × ℝ) 

noncomputable def radiusP : ℝ := 2
noncomputable def radiusQ : ℝ := 3
noncomputable def radiusR : ℝ := 4

-- Collinearity of points due to their positioning on the x-axis
def PointsPQRCollinear (P Q R : CircleCenterP × CircleCenterQ × CircleCenterR) : Prop
  := (P.1.1 = -1) ∧ (P.2.1 = 0) ∧ (P.3.1 = 1) ∧ (P.2.2 = 0) 

-- We are to prove that the area of triangle PQR is 0
theorem triangle_PQR_area_zero : 
  ∀ (P Q R : CircleCenterP × CircleCenterQ × CircleCenterR), 
  (PointsPQRCollinear P Q R) → 
  (Geometry.Euclidean.Triangle.area ([(P.fst.fst, P.fst.snd), (Q.fst.fst, Q.fst.snd), (R.fst.fst, R.fst.snd)]) = 0) 
:= by
  sorry

end triangle_PQR_area_zero_l87_87682


namespace certain_number_is_48_l87_87621

theorem certain_number_is_48 (x : ℕ) (h : x = 4) : 36 + 3 * x = 48 := by
  sorry

end certain_number_is_48_l87_87621


namespace distance_between_particles_l87_87944

noncomputable def gravity := 9800 -- mm/s^2
noncomputable def height_cliff := 300 * 1000 -- mm
noncomputable def initial_distance_first_particle := 1 / 1000 -- mm

theorem distance_between_particles (g : ℝ) (h : ℝ) (s_1 : ℝ) : 
  (1 / 2) * g * (Real.sqrt((2 * h) / g) - Real.sqrt((2 * s_1) / g))^2 = 34.6 := 
by
  let height_cliff := h
  let initial_distance_first_particle := s_1
  let gravity := g
  sorry

#eval distance_between_particles gravity height_cliff initial_distance_first_particle

end distance_between_particles_l87_87944


namespace area_difference_l87_87991

noncomputable def area_circle (r : ℝ) : ℝ := π * r^2

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem area_difference (r : ℝ) (s : ℝ) :
  r = 5 → s = 10 →
  area_circle r - area_equilateral_triangle s = 25 * (π - sqrt 3) :=
by
  intros hr hs
  rw [hr, hs]
  have h1 : area_circle 5 = 25 * π := by
    simp [area_circle 5]
  have h2 : area_equilateral_triangle 10 = 25 * sqrt 3 := by
    simp [area_equilateral_triangle 10]
  rw [h1, h2]
  norm_num
  sorry

end area_difference_l87_87991


namespace locus_Y_arc_of_circle_l87_87602

-- Define triangle and point X on BC
variables {A B C X Y : Point}
variable (ABC : Triangle A B C)
variable (hX_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = segment_ratio BC t)

-- Define the common tangent and points
variable (tangent_ABX_ACX : CommonTangent (incircle ABC A B X) (incircle ABC A C X))
variable (hY_tangent_cross : tangent_ABX_ACX ∩ line_AX = Y)
variable (line_AX : Line A X)

theorem locus_Y_arc_of_circle :
  ∀ X, hX_on_BC X → (∃ O R, circle_arc O R A Y) :=
sorry

end locus_Y_arc_of_circle_l87_87602


namespace routes_on_3x3_grid_are_20_l87_87998

def number_of_routes_3x3 : ℕ := 
  Nat.choose 6 3

theorem routes_on_3x3_grid_are_20 : number_of_routes_3x3 = 20 :=
sorry

end routes_on_3x3_grid_are_20_l87_87998


namespace vector_dot_product_l87_87786

variables (a b : EuclideanSpace ℝ (Fin 3))
variable (theta : ℝ)

theorem vector_dot_product (h1 : (a + b) ⬝ (a - (2 : ℝ) • b) = -6)
    (h2 : ‖a‖ = 1) (h3 : ‖b‖ = 2) :
    (a ⬝ b = -1) ∧ (cos theta = -1/2 → theta = (2 * π) / 3) :=
by sorry

end vector_dot_product_l87_87786


namespace inequality_proof_l87_87372

open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_proof_l87_87372


namespace dandelion_ratio_l87_87675

theorem dandelion_ratio :
  ∃ (G : ℕ), 
  G > 0 ∧
  (∃ (avg_dandelions : ℕ), avg_dandelions = 34 ∧ 
    (36 + G + 10 + 10) = 2 * avg_dandelions) → 
  G = 12 → 
  (G / 36 : ℚ) = (1 / 3) :=
by 
  sorry

end dandelion_ratio_l87_87675


namespace mom_foster_dog_food_l87_87478

theorem mom_foster_dog_food
    (puppy_food_per_meal : ℚ := 1 / 2)
    (puppy_meals_per_day : ℕ := 2)
    (num_puppies : ℕ := 5)
    (total_food_needed : ℚ := 57)
    (days : ℕ := 6)
    (mom_meals_per_day : ℕ := 3) :
    (total_food_needed - (num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days)) / (↑days * ↑mom_meals_per_day) = 1.5 :=
by
  -- Definitions translation
  let puppy_total_food := num_puppies * puppy_food_per_meal * ↑puppy_meals_per_day * ↑days
  let mom_total_food := total_food_needed - puppy_total_food
  let mom_meals := ↑days * ↑mom_meals_per_day
  -- Proof starts with sorry to indicate that the proof part is not included
  sorry

end mom_foster_dog_food_l87_87478


namespace find_sum_first_9_terms_l87_87080

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Assume the sequence is arithmetic
axiom arithmetic_sequence (d : ℝ) (a₁ : ℝ) : ∀ n, a n = a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (n : ℕ) : ℝ := (n / 2) * (2 * a 1 + (n - 1) * d)

-- The given condition
axiom condition : a 2 + a 8 = 18 - a 5

-- The target theorem to prove
theorem find_sum_first_9_terms : sum_first_n_terms 9 = 54 := 
by sorry

end find_sum_first_9_terms_l87_87080


namespace distance_from_P_to_chord_l87_87018

-- Definition of the problem conditions
def point_P : ℝ × ℝ := (3, -4)
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Definition of the distance from a point to a line
def distance_to_line (A B C x y : ℝ) : ℝ :=
  (A * x + B * y + C).abs / (Real.sqrt (A^2 + B^2))

-- Setting up the main theorem with the given and to prove statement
theorem distance_from_P_to_chord :
  let A := 3;   -- Coefficient of x in the chord equation
  let B := -4;  -- Coefficient of y in the chord equation
  let C := -9;  -- Constant term in the chord equation
  let d := distance_to_line A B C (Prod.fst point_P) (Prod.snd point_P)
  d = 16 / 5 :=
by
  sorry   -- Proof is omitted

end distance_from_P_to_chord_l87_87018


namespace vikki_take_home_pay_correct_l87_87946

noncomputable def vikki_take_home_pay : ℕ := 5368 / 10

def total_hours := 50
def hours_job_a := 30
def rate_job_a := 12
def hours_job_b := 20
def rate_job_b := 15
def overtime_threshold := 40
def overtime_multiplier := 1.5
def federal_tax_first_threshold := 300
def federal_tax_first_rate := 0.15
def federal_tax_second_rate := 0.22
def state_tax_rate := 0.07
def retirement_contribution_rate := 0.06
def insurance_cover_rate := 0.03
def union_dues := 5

def vikki_gross_earnings (h_a h_b r_a r_b : ℕ) : ℕ :=
  let ft_h := min overtime_threshold h_a in
  let ft_e := min overtime_threshold (residual h_a overtime_threshold h_b) in
  gross_earnings (h_a, r_a) + gross_earnings (h_b, r_b) + overtime earnings (h_a, r_a, overtime_multiplier)

def deduction_total (e : ℕ) : ℕ := 
  deduction (e, federal_tax_first_threshold, federal_tax_first_rate, federal_tax_second_rate) +
  deduction (e, state_tax_rate) +
  deduction (e, retirement_contribution_rate) +
  deduction (e, insurance_cover_rate) +
  union_dues

theorem vikki_take_home_pay_correct :
  let earnings := vikki_gross_earnings tr_hours tj_hours tj_rate fj_rate in
  earnings - deduction_total earnings = vikki_take_home_pay := 
by sorry

end vikki_take_home_pay_correct_l87_87946


namespace y_intercept_l87_87203

theorem y_intercept (x y : ℝ) (h : y = -3 * x + 5) (hx : x = 0) : y = 5 :=
by
  rw [hx] at h
  exact h

end y_intercept_l87_87203


namespace correct_product_l87_87815

theorem correct_product (a b a' : ℕ) (h_reversal : a' = digit_reverse a) (h_product : a' * b = 396) : ab = 693 := sorry

end correct_product_l87_87815


namespace evaluate_expression_l87_87323

def from_base (n: ℕ) (b: ℕ) (digits: List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * b + d) 0

theorem evaluate_expression :
  let n1 := from_base 8 [2, 5, 4]
  let n2 := from_base 4 [1, 2]
  let n3 := from_base 5 [1, 3, 2]
  let n4 := from_base 3 [2, 2]
  (n1 / n2 : ℚ) + (n3 / n4) = 33.9167 := 
by
  sorry

end evaluate_expression_l87_87323


namespace coeff_x_in_expansion_l87_87832

theorem coeff_x_in_expansion : 
  (binomial.coeff 6 2 * (3 : ℝ) ^ (6 - 2) * (2 : ℝ) ^ 2) = 4860 := 
  by
  calc
    binomial.coeff 6 2 * (3 : ℝ) ^ 4 * (2 : ℝ) ^ 2 = (15 * 81 * 4 : ℝ) : by sorry
    ... = 4860 : by sorry

end coeff_x_in_expansion_l87_87832


namespace diagonal_difference_l87_87623

def original_matrix : Matrix (Fin 4) (Fin 4) ℕ :=
  ![[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]]

def transformed_matrix : Matrix (Fin 4) (Fin 4) ℕ :=
  ![[4, 3, 2, 1],
    [5, 6, 7, 8],
    [12, 11, 10, 9],
    [13, 14, 15, 16]]

theorem diagonal_difference :
  let main_diag_sum := transformed_matrix 0 0 + transformed_matrix 1 1 + transformed_matrix 2 2 + transformed_matrix 3 3,
      sec_diag_sum := transformed_matrix 0 3 + transformed_matrix 1 2 + transformed_matrix 2 1 + transformed_matrix 3 0
  in Int.natAbs (main_diag_sum - sec_diag_sum) = 4 := by
  sorry

end diagonal_difference_l87_87623


namespace PQ_over_EF_l87_87161

-- Definitions of points and segments based on problem conditions
def Rectangle (A B C D E F G : ℝ × ℝ) : Prop :=
  A = (0, 6) ∧ B = (8, 6) ∧ C = (8, 0) ∧ D = (0, 0) ∧
  E = (5, 6) ∧ F = (4, 0) ∧ G = (8, 4)

-- Definitions of lines' equations
def LineA (P Q : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), P (x, y) → y = -1/4 * x + 6 ∧ Q (x, y) → y = -3/4 * x + 6

def LineEF (P : ℝ × ℝ → Prop) : Prop := 
  ∀ (x y : ℝ), P (x, y) → y = 6 * x - 24

-- Intersection points using line equations
def IntersectionPoints (P Q : ℝ × ℝ) : Prop := 
  P = (120/27, 152/27) ∧ Q = (120/25, 196/25)

-- Proof goal: Ratio of intersections to length of segment EF
theorem PQ_over_EF {A B C D E F G P Q : ℝ × ℝ} :
  Rectangle A B C D E F G →
  LineA (λ(x y : ℝ), (x, y) = P) (λ(x y : ℝ), (x, y) = Q) →
  LineEF (λ(x y : ℝ), (x, y) = P) →
  LineEF (λ(x y : ℝ), (x, y) = Q) →
  IntersectionPoints P Q →
  (dist P Q) / (dist E F) = 4 / (45 * Real.sqrt 37) :=
by
  sorry

end PQ_over_EF_l87_87161


namespace arithmetic_sequence_s9_l87_87830

noncomputable def arithmetic_sum (a1 d n : ℝ) : ℝ :=
  n * (2*a1 + (n - 1)*d) / 2

noncomputable def general_term (a1 d n : ℝ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_s9 (a1 d : ℝ)
  (h1 : general_term a1 d 3 + general_term a1 d 4 + general_term a1 d 8 = 25) :
  arithmetic_sum a1 d 9 = 75 :=
by sorry

end arithmetic_sequence_s9_l87_87830


namespace find_a_of_parabola_l87_87186

theorem find_a_of_parabola (a b c : ℤ) (h_vertex : (2, 5) = (2, 5)) (h_point : 8 = a * (3 - 2) ^ 2 + 5) :
  a = 3 :=
sorry

end find_a_of_parabola_l87_87186


namespace at_least_two_equal_numbers_written_l87_87204

theorem at_least_two_equal_numbers_written (n : ℕ) (h_n : n > 3)
  (a : Fin n → ℕ)
  (h_distinct : Function.Injective a)
  (h_bound : ∀ i, a i < (n - 1)! + (n - 2)! ) :
  ∃ i j k l : Fin n, i ≠ j ∧ k ≠ l ∧ i ≠ l ∧ j ≠ k ∧ i ≠ k ∧ j ≠ l ∧
  ⌊a i / a j⌋ = ⌊a k / a l⌋ :=
by sorry

end at_least_two_equal_numbers_written_l87_87204


namespace perimeter_difference_is_8_l87_87986

-- Define the dimensions of the sheet and the number of pieces
def width : ℝ := 6
def length : ℝ := 10
def num_pieces : ℕ := 6

-- Define the possible cutting configurations and their resulting perimeters
def perimeter1 : ℝ := 2 * (1 + length)
def perimeter2 : ℝ := 2 * (width + length/num_pieces)
def perimeter3 : ℝ := 2 * (2 + 5)

-- Define the greatest and least perimeters
def greatest_perimeter : ℝ := max (max perimeter1 perimeter2) perimeter3
def least_perimeter : ℝ := min (min perimeter1 perimeter2) perimeter3

-- Define the difference in perimeters
def perimeter_difference : ℝ := greatest_perimeter - least_perimeter

-- The theorem to be proven
theorem perimeter_difference_is_8 : perimeter_difference = 8 :=
by sorry

end perimeter_difference_is_8_l87_87986


namespace biscuit_banana_ratio_l87_87172

variables {b x : ℝ}

-- Conditions from the problem
def susie_expense := 6 * b + 4 * x
def daisy_expense := 4 * b + 20 * x

-- Lean statement for the proof problem
theorem biscuit_banana_ratio (hb : susie_expense * 3 = daisy_expense) : b / x = 4 / 7 :=
by 
  sorry

end biscuit_banana_ratio_l87_87172


namespace length_AD_radius_circle_l87_87150

-- Definitions using the given conditions
variables (A B C D E F X Y : Type) [ch : Circle A] [ch : Circle B]
variables (CX XE XY BY YF : ℝ)
variables (hX : ∠ A D C = 90)
variables (h_intersect_1 : Intersect AD CE X)
variables (h_intersect_2 : Intersect AD BF Y)

-- Given values
variables (hCX : CX = 12)
variables (hXE : XE = 27)
variables (hXY : XY = 15)
variables (hBY : BY = 9)
variables (hYF : YF = 11)

-- Translate questions and answers into Lean theorem statements.

-- Part (a): Find the length of segment AD
theorem length_AD : length AD = 36 :=
sorry

-- Part (b): Find the radius of the circle
theorem radius_circle : radius A = 19.5 :=
sorry

end length_AD_radius_circle_l87_87150


namespace largest_of_four_numbers_l87_87045

theorem largest_of_four_numbers (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : -1 < b) (hb0 : b < 0) :
  ∀ x ∈ {a, a * b, a - b, a + b}, x ≤ a - b :=
begin
  intros x hx,
  finset.mem_insert.mp hx;
  sorry
end

end largest_of_four_numbers_l87_87045


namespace toms_investment_l87_87571

theorem toms_investment 
  (P : ℝ)
  (rA : ℝ := 0.06)
  (nA : ℝ := 1)
  (tA : ℕ := 4)
  (rB : ℝ := 0.08)
  (nB : ℕ := 2)
  (tB : ℕ := 4)
  (delta : ℝ := 100)
  (A_A := P * (1 + rA / nA) ^ (nA * tA))
  (A_B := P * (1 + rB / nB) ^ (nB * tB))
  (h : A_B - A_A = delta) : 
  P = 942.59 := by
sorry

end toms_investment_l87_87571


namespace sum_of_f_values_l87_87408

def f (x : ℝ) : ℝ := 2 * Real.sin ( (Real.pi / 2) * x + (Real.pi / 3) )

theorem sum_of_f_values : (Finset.range 2016).sum (λ i, f (i + 1)) = 0 :=
by sorry

end sum_of_f_values_l87_87408


namespace geometric_body_from_translating_trapezoid_is_prism_l87_87185

theorem geometric_body_from_translating_trapezoid_is_prism :
  ∀ (translate : ℝ → ℝ → ℝ) (trapezoid : Set ℝ), 
  (∃ (translated_body : Set ℝ), translated_body = {translate x | x ∈ trapezoid}) → 
  is_quadrangular_prism translated_body :=
sorry

end geometric_body_from_translating_trapezoid_is_prism_l87_87185


namespace four_digit_numbers_divisible_by_11_l87_87432

theorem four_digit_numbers_divisible_by_11 : 
  ∃ (count : ℕ), 
  count = (List.range' 1001 (9999 - 1001 + 1)).filter (λ n, n % 11 = 0).length 
    ∧ count = 819 := 
by 
  sorry

end four_digit_numbers_divisible_by_11_l87_87432


namespace incorrect_conclusions_l87_87316

open_locale classical

-- Definitions based on the conditions:
def equal_product_sequence (a : ℕ → ℝ) (Q : ℝ) : Prop :=
∀ n : ℕ, a n * a (n + 1) = Q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) (sum : ℝ) : Prop :=
(finset.range n).sum a = sum

-- Main proof goal stating the problem
theorem incorrect_conclusions
  (a : ℕ → ℝ)
  (Q : ℝ)
  (h_eq_prod_seq : equal_product_sequence a Q)
  (h_a1 : a 1 = 3)
  (h_sum : sum_first_n_terms a 7 14) :
  ¬ ((∀ n, a (n + 2) = a n) ∧ (a 2 = 2/3) ∧ (Q = 3) ∧ (∀ n, a n * a (n + 1) * a (n + 2) = 12)) :=
sorry

end incorrect_conclusions_l87_87316


namespace jacob_nickel_count_l87_87107

theorem jacob_nickel_count (n : ℕ) : 
  let total_cost := 42.50 * 1.08 in
  let jacob_money := 4 * 5 + 6 * 1 + 10 * 0.25 + n * 0.05 in
  total_cost ≤ jacob_money ↔ 348 ≤ n :=
by
  let total_cost := 42.50 * 1.08
  let jacob_money := 4 * 5 + 6 * 1 + 10 * 0.25 + n * 0.05
  have : total_cost = 45.90 := by norm_num
  have : jacob_money = 28.50 + n * 0.05 := by norm_num
  rw [this, this_1]
  sorry

end jacob_nickel_count_l87_87107


namespace find_number_l87_87984

theorem find_number (x : ℕ) (h : x / 4 + 3 = 5) : x = 8 :=
by sorry

end find_number_l87_87984


namespace range_of_a_l87_87067

noncomputable def curve_has_no_common_points_with_line (a : ℝ) (θ : ℝ) : Prop :=
  ∀ x y : ℝ, x + y = a → ¬ (x = f(θ) ∧ y = g(θ)) -- where f(θ) and g(θ) are parameter functions of θ

theorem range_of_a (a : ℝ): (∃ θ : ℝ, ¬curve_has_no_common_points_with_line a θ) →
  (a > 5 ∨ a < -5) :=
  sorry

end range_of_a_l87_87067


namespace total_difference_l87_87654

def period1_enrollments := [60, 30, 20, 5, 3, 2] : List ℕ
def period2_enrollments := [45, 35, 20, 10, 5, 5] : List ℕ

def t (n : ℕ) (lst : List ℕ) : ℕ := (List.sum lst) / n
def s (total : ℕ) (lst : List ℕ) : ℕ := List.sum (lst.map (fun x => x * x)) / total

def students := 120
def teachers := 6

theorem total_difference :
  let t1 := t teachers period1_enrollments
  let s1 := s students period1_enrollments
  let t2 := t teachers period2_enrollments
  let s2 := s students period2_enrollments
  (t1 + s1) - (t2 + s2) = 9.48 := by
  sorry

end total_difference_l87_87654


namespace helen_cookies_till_last_night_l87_87046

theorem helen_cookies_till_last_night 
  (cookies_yesterday : Nat := 31) 
  (cookies_day_before_yesterday : Nat := 419) : 
  cookies_yesterday + cookies_day_before_yesterday = 450 := 
by
  sorry

end helen_cookies_till_last_night_l87_87046


namespace part_a_part_b_l87_87305

noncomputable def area_of_figure : ℕ := 25
def side_length : ℕ := 5
def num_parts : ℕ := 3

-- This placeholder represents the figure "camel" with specific area and dimensions
constant camel_figure : Type

-- Define the division of the figure into parts along grid lines
def can_divide_along_grid_lines (figure : camel_figure) (parts : ℕ) : Prop := sorry

-- Define the division of the figure into parts not necessarily along grid lines
def can_divide_not_along_grid_lines (figure : camel_figure) (parts : ℕ) : Prop := sorry

-- State that the figure cannot be divided along the grid lines into parts that can be assembled into a 5x5 square
theorem part_a (figure : camel_figure) :
  (area_of_figure = side_length ^ 2) →
  ¬ can_divide_along_grid_lines figure num_parts :=
sorry

-- State that the figure can be divided not necessarily along grid lines into parts that can be assembled into a 5x5 square
theorem part_b (figure : camel_figure) :
  (area_of_figure = side_length ^ 2) →
  can_divide_not_along_grid_lines figure num_parts :=
sorry

end part_a_part_b_l87_87305


namespace find_n_l87_87592

theorem find_n (a b : ℝ) (h1 : 2 ≤ n) (h2 : a * b ≠ 0) (h3 : a = 3 * b) :
  ∃ n : ℕ, n = 11 ∧ 
  (binomial_theorem_term (2 * a + b) n 0 = - binomial_theorem_term (2 * a + b) n 2) :=
by {
  sorry
}

end find_n_l87_87592


namespace olivia_correct_answers_l87_87453

theorem olivia_correct_answers (c w : ℕ) 
  (h1 : c + w = 15) 
  (h2 : 6 * c - 3 * w = 45) : 
  c = 10 := 
  sorry

end olivia_correct_answers_l87_87453


namespace incorrect_proposition_C_l87_87853

-- Step A: Definitions for the vectors and properties
variables {a b : ℝ} -- Scalars to define conditions
variables (u v : ℝ) -- Scalars for defining general vectors

-- Define the non-zero vectors
variables (ab bc : ℝ)
variables (AB BC : Vector ℝ)

-- Step C: The final problem statement
theorem incorrect_proposition_C 
  (a b c : ℝ) 
  (ha_ne_0 : a ≠ 0) 
  (hb_ne_0 : b ≠ 0)
  (H1 : ∥a + b∥ = ∥a∥ - ∥b∥ → ∃ (λ : ℝ), a = λ * b)
  (H2 : ⟪a, b⟫ = 0 → ∥a + b∥ = ∥a - b∥)
  (H3 : AB = 1 → BC = 1 → ∥AB - BC∥ = √3 / 2)
  (H4 : (a ⋅ c = b ⋅ c) → (a = b → False)) :
  ∃ (incorrect : Prop), incorrect = true :=
by -- Proof steps are omitted here; the goal is to identify the incorrect proposition.
  sorry

end incorrect_proposition_C_l87_87853


namespace interval_of_defined_expression_l87_87352

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l87_87352


namespace domain_of_k_l87_87699

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 9)) + (1 / (x^2 + 9 * x + 20)) + (1 / (x^3 + 27))

theorem domain_of_k :
  ∀ x : ℝ, x ∉ {-9, -5, -4, -3} ↔ ∃ y : ℝ, k y
:= sorry

end domain_of_k_l87_87699


namespace chord_length_proof_l87_87780

-- Definition of polar curves
def polar_curve_1 (ρ θ : ℝ) : Prop := ρ = 2 * sin θ
def polar_curve_2 (ρ θ : ℝ) : Prop := θ = π / 3 ∧ ρ ∈ Set.univ

-- Definition of Cartesian curves derived from polar curves
def cartesian_curve_1 (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def cartesian_curve_2 (x y : ℝ) : Prop := y = sqrt 3 * x

-- The center of the circle (C1 is a circle with center (0, 1))
def center := (0, 1)

-- Distance from the center to the curve (line)
def distance_to_line (center : ℝ × ℝ) : ℝ := (1 / 2)

-- Radius of C1
def radius := 1

-- Length of chord MN
def chord_length := 2 * sqrt ((radius^2) - (distance_to_line center)^2)

-- The theorem statement
theorem chord_length_proof :
  ∀ {ρ θ : ℝ},
    (polar_curve_1 ρ θ ∧ polar_curve_2 ρ θ) →
    chord_length = sqrt 3 :=
by
  sorry

end chord_length_proof_l87_87780


namespace nancy_total_money_l87_87514

theorem nancy_total_money (n : ℕ) (d : ℕ) (h1 : n = 9) (h2 : d = 5) : n * d = 45 := 
by
  sorry

end nancy_total_money_l87_87514


namespace exponential_neg_sum_l87_87800

theorem exponential_neg_sum (γ δ : ℂ) (h : complex.exp (complex.I * γ) + complex.exp (complex.I * δ) = (2/5:ℂ) + (1/2:ℂ) * complex.I) : 
  complex.exp (-complex.I * γ) + complex.exp (-complex.I * δ) = (2/5:ℂ) - (1/2:ℂ) * complex.I :=
by
  sorry

end exponential_neg_sum_l87_87800


namespace correct_options_given_inequality_l87_87055

theorem correct_options_given_inequality (x y : ℝ) :
  3^(-x) - 3^(-y) < log 3 x - log 3 y →
  (exp (x - y) > 1 ∧ log (x - y + 1) > 0) :=
by
  sorry

end correct_options_given_inequality_l87_87055


namespace nina_total_sales_l87_87879

def price_necklace := 25.00
def price_bracelet := 15.00
def price_earrings := 10.00
def price_ensemble := 45.00

def sold_necklaces := 5
def sold_bracelets := 10
def sold_earrings := 20
def sold_ensembles := 2

def discount_necklace := 0.10
def discount_bracelet := 0.05
def discount_ensemble := 0.15

def customization_fee_necklace := 5.00
def customization_fee_bracelet := 3.00

def sales_tax := 0.08

theorem nina_total_sales : 
  let total := 
    ((price_necklace * sold_necklaces * (1 - discount_necklace) + customization_fee_necklace) +
     (price_bracelet * sold_bracelets * (1 - discount_bracelet) + customization_fee_bracelet * 2) + 
     (price_earrings * sold_earrings) + 
     (price_ensemble * sold_ensembles * (1 - discount_ensemble))) in
  let subtotal := total in
  let tax := subtotal * sales_tax in
  let total_amount := subtotal + tax in
  total_amount = 585.90 :=
  sorry

end nina_total_sales_l87_87879


namespace arithmetic_sequence_general_formula_l87_87068

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, 0 < n → (a n - 2 * a (n + 1) + a (n + 2) = 0)) : ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l87_87068


namespace dice_sum_probability_l87_87573

theorem dice_sum_probability :
  let outcomes := 36 in
  let favorable := 26 in -- Total number of pairs where the sum is less than 9
  (favorable.toRat / outcomes.toRat) = (13 / 18) :=
by sorry

end dice_sum_probability_l87_87573


namespace tan_theta_calc_l87_87445

theorem tan_theta_calc (θ : ℝ) (z : ℂ)
  (h : z = (sin θ - 3/5) + (cos θ - 4/5) * complex.I)
  (hz : ∃ b : ℝ, z = b * complex.I) :
  tan θ = -3/4 := 
by
  sorry

end tan_theta_calc_l87_87445


namespace triangles_similar_oppositely_oriented_l87_87614

variables (J₁ J₂ J₃ P₁ P₂ P₃ : Type) [has_coe ℝ J₁]
[has_coe ℝ J₂] [has_coe ℝ J₃] [has_coe ℝ P₁]
[has_coe ℝ P₂] [has_coe ℝ P₃]

-- Assume the existence of points or line segments 
-- J₁, J₂, J₃ and P₁, P₂, P₃.
-- The goal is to prove that triangles J₁J₂J₃ and P₁P₂P₃
-- are similar and oppositely oriented.
theorem triangles_similar_oppositely_oriented :
  ∃ (α₁ α₂ α₃ : ℝ), 
  (∠(J₁, J₂, J₃) = ∠(P₃, P₂, P₁)) ∧ 
  (∠(J₃, J₂, J₁) = ∠(P₁, P₂, P₃)) ∧
  (∠(J₃, J₁, J₂) = ∠(P₁, P₃, P₂)) := 
sorry

end triangles_similar_oppositely_oriented_l87_87614


namespace min_expression_value_l87_87722

theorem min_expression_value (x : ℝ) : 
  (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1) >= 1/2 := 
sorry

end min_expression_value_l87_87722


namespace proof_inequality_l87_87117

-- Declare the variables and geometric entities
variables {A B C P : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space P]
variables (a b c : ℝ) (PA PB PC : ℝ)

-- Define the sides and point measurements
def side_a (a b c : ℝ) := a
def side_b (a b c : ℝ) := b
def side_c (a b c : ℝ) := c

-- Define P B, P C, P A distances
def dist_PA (PA : ℝ) := PA
def dist_PB (PB : ℝ) := PB
def dist_PC (PC : ℝ) := PC

-- Theorem statement
theorem proof_inequality
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : PA ≥ 0) (h5 : PB ≥ 0) (h6 : PC ≥ 0) :
  (PB * PC) / (b * c) + (PC * PA) / (c * a) + (PA * PB) / (a * b) ≥ 1 :=
sorry

end proof_inequality_l87_87117


namespace find_radius_of_circle_l87_87421

def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / (sqrt (a ^ 2 + b ^ 2))

theorem find_radius_of_circle
  (r : ℝ)
  (h1 : r > 0)
  (h2 : ∀ x y : ℝ, (x + sqrt 3 * y - 2 = 0 ↔ x^2 + y^2 = r^2))
  (h3 : ∃ A B : ℝ × ℝ, A ≠ B ∧ (let ⟨xa, ya⟩ := A in (xa + sqrt 3 * ya - 2 = 0 ∧ xa^2 + ya^2 = r^2)) ∧ (let ⟨xb, yb⟩ := B in (xb + sqrt 3 * yb - 2 = 0 ∧ xb^2 + yb^2 = r^2)) ∧ real.angle (xa, ya) (0, 0) (xb, yb) = 2 * real.pi / 3) :
  r = 2 := 
by
  -- Proving the theorem manually below
  have d : ℝ := distance_point_to_line 0 0 1 (sqrt 3) (-2),
  have half_radius : d = r / 2,
  have calculated_d : d = |-2| / sqrt (1 + 3),
  
  sorry -- Completing proof would typically follow here.

end find_radius_of_circle_l87_87421


namespace allocation_schemes_l87_87346

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l87_87346


namespace area_trapezoid_BEHD_l87_87464

-- Define the geometric conditions
variables {A B C D E F G H : Type} -- Points in the plane

-- Definitions for the squares and equilateral triangles
def is_square (ABCD : quadrilateral) : Prop := -- assuming some definition for quadrilateral
  -- some formalization ensuring ABCD is a square
  sorry

def is_equilateral (triangle : triangle) : Prop := -- assuming some definition for triangle
  -- some formalization ensuring triangle is equilateral
  sorry

-- Given conditions
variables (ABCD EFGH : quadrilateral) (AEH BEF CFG DHG : triangle)

-- The condition areas and properties
axiom square_ABCD : is_square ABCD
axiom square_EFGH : is_square EFGH
axiom eq_triangle_AEH : is_equilateral AEH
axiom eq_triangle_BEF : is_equilateral BEF
axiom eq_triangle_CFG : is_equilateral CFG
axiom eq_triangle_DHG : is_equilateral DHG
axiom area_square_ABCD : (area ABCD) = 360

-- Theorem to prove that the area of trapezoid BEHD is 90
theorem area_trapezoid_BEHD : ∃ (BEHD : trapezoid), 
  (area BEHD) = 90 :=
by
  sorry

end area_trapezoid_BEHD_l87_87464


namespace find_functions_l87_87325

theorem find_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ^ 2) :=
by
  sorry

end find_functions_l87_87325


namespace tangent_line_y_intercept_l87_87632

open Real

theorem tangent_line_y_intercept
  (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 2)
  (h_c1 : c1 = (3, 0)) (h_c2 : c2 = (8, 0)) :
  ∃ l : ℝ, l = 13/4 ∧ is_tangent (circle r1 c1) (circle r2 c2) l ∧
  (∀ p : ℝ × ℝ, p ∈ first_quadrant → is_tangent_point p r1 c1 l ∧ is_tangent_point p r2 c2 l) := 
by
  sorry

-- Helper Definitions (assuming they are defined elsewhere in Mathlib or we can define them here if necessary)
noncomputable def circle (r : ℝ) (c : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p | dist p c = r }

def is_tangent (C1 C2 : set (ℝ × ℝ)) (l : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p ∈ C1 ∧ is_on_line l p) → (p ∈ C2 ∧ is_on_line l p)

def is_tangent_point (p : ℝ × ℝ) (r : ℝ) (c : ℝ × ℝ) (l : ℝ) : Prop :=
  dist p c = r ∧ is_on_line l p

def first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_on_line (l : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = l * p.1

-- Placeholder for required definitions not provided in Mathlib or requiring customization.
-- This can involve defining specific properties for tangency, distances, etc., that capture the geometry of the circles and lines.

end tangent_line_y_intercept_l87_87632


namespace number_of_routes_4x3_grid_l87_87309

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem number_of_routes_4x3_grid : binomial_coefficient 7 4 = 35 := by
  sorry

end number_of_routes_4x3_grid_l87_87309


namespace problem1_problem2_l87_87619

-- Problem 1: Calculate the expression
theorem problem1 : (-1 : ℤ)^2023 + (1 / 2 : ℚ) ^ -2 - (3 - real.pi) ^ 0 = 2 := by
  sorry

-- Problem 2: Simplify the expression using multiplication formula
theorem problem2 : (2024 * 2022 : ℤ) - 2023^2 = -1 := by
  sorry

end problem1_problem2_l87_87619


namespace quadratic_opens_downward_l87_87066

noncomputable def quadratic_function (a x : ℝ) : ℝ := (2 * a - 6) * x^2 + 4

theorem quadratic_opens_downward (a : ℝ) : (∃ (y : ℝ), y = quadratic_function a) → a < 3 :=
by
  intros h
  have h_coefficient : (2 * a - 6) < 0 := sorry
  linarith

end quadratic_opens_downward_l87_87066


namespace least_addition_for_palindrome_l87_87221

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

theorem least_addition_for_palindrome : ∃ (k : ℕ), k = 75 ∧ is_palindrome (52350 + k) := 
by
  use 75
  sorry

end least_addition_for_palindrome_l87_87221


namespace log_sqrt_defined_l87_87367

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l87_87367


namespace geometric_sequence_lambda_l87_87876

theorem geometric_sequence_lambda (n : ℕ) (S : ℕ → ℕ) (λ : ℤ) (S_def : ∀ n, S n = 2^(n+1) + λ) :
  λ = -2 :=
by
  sorry

end geometric_sequence_lambda_l87_87876


namespace problem_statement_l87_87910

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition : ∀ a b : ℝ, b^2 * f a = a^2 * f b
axiom f2_nonzero : f 2 ≠ 0

theorem problem_statement : (f 6 - f 3) / f 2 = 27 / 4 := 
by 
  sorry

end problem_statement_l87_87910


namespace close_one_station_without_loss_of_connectivity_l87_87820

-- Define the metro city N as a connected graph
variable {V : Type} -- Type of vertices (stations)
variable {G : SimpleGraph V} -- Simple graph representing the metro network

-- Condition: The graph G is connected
variable [h_connected : G.Connected]

-- Statement: Prove that one of the stations can be closed without disrupting the travel between any of the remaining stations.
theorem close_one_station_without_loss_of_connectivity : 
  ∃ (v : V), Connected (G.deleteVertex v) :=
sorry

end close_one_station_without_loss_of_connectivity_l87_87820


namespace common_divisors_count_9240_10010_l87_87434

def divisors (n : Nat) : Nat :=
(n.primeFactors.map (λ p => p.2 + 1)).foldl (· * ·) 1

theorem common_divisors_count_9240_10010 :
  let gcd_value := Nat.gcd 9240 10010;
  let num_common_divisors := divisors gcd_value;
  gcd_value = 210 ∧ num_common_divisors = 16 :=
by
  have : 9240 = 2^3 * 3^1 * 5^1 * 7^2 := by norm_num
  have : 10010 = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num
  have gcd_value_calc := Nat.gcd 9240 10010
  have : gcd_value_calc = 210 := by norm_num
  have num_common_divisors_calc := divisors gcd_value_calc
  have : num_common_divisors_calc = 16 := by norm_num
  exact ⟨this, by norm_num⟩

end common_divisors_count_9240_10010_l87_87434


namespace math_problem_l87_87689

theorem math_problem
  (x : ℤ)
  (h1 : 8 + x ≡ 27 [MOD 8])
  (h2 : 10 + x ≡ 16 [MOD 27])
  (h3 : 13 + x ≡ 36 [MOD 125]) :
  x ≡ 11 [MOD 120] :=
sorry

end math_problem_l87_87689


namespace telescope_visual_range_l87_87271

theorem telescope_visual_range (original_range : ℕ) (increase_percent : ℕ) :
  original_range = 50 ∧ increase_percent = 200 → 
  let increase := (increase_percent / 100) * original_range in
  original_range + increase = 150 := by
  intros
  -- Proof is omitted
  sorry

end telescope_visual_range_l87_87271


namespace number_of_tables_l87_87638

theorem number_of_tables (n_base7 : ℕ) (n : ℕ) (ppl_per_table : ℕ) (tables : ℕ) 
  (h1 : n_base7 = 315) 
  (h2 : ∀ k, 7^0 * (315 % 10) + 7^1 * ((315 / 10) % 10) + 7^2 * ((315 / 100) % 10) = n)
  (h3 : ppl_per_table = 3) 
  (h4 : tables = n / ppl_per_table) :
  tables = 53 := by 
  sorry

end number_of_tables_l87_87638


namespace problem_statement_l87_87033

theorem problem_statement 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 * sin x * cos x - sin x^2 + 1)
  (θ : ℝ)
  (hθ : f θ = let S := λ x, 2 * sin x * cos x - sin x^2 + 1 in 
              if ∃ z, ∀ y, S y ≥ S z then z else 0)
  (hmin : ∃ x, ∀ y, f x ≤ f y) :
  (sin (2 * θ) + cos (2 * θ)) / (sin (2 * θ) - cos (2 * θ)) = 3 :=
by
  sorry

end problem_statement_l87_87033


namespace average_speed_of_car_l87_87270

noncomputable def avgSpeed (Distance_uphill Speed_uphill Distance_downhill Speed_downhill : ℝ) : ℝ :=
  let Time_uphill := Distance_uphill / Speed_uphill
  let Time_downhill := Distance_downhill / Speed_downhill
  let Total_time := Time_uphill + Time_downhill
  let Total_distance := Distance_uphill + Distance_downhill
  Total_distance / Total_time

theorem average_speed_of_car:
  avgSpeed 100 30 50 60 = 36 := by
  sorry

end average_speed_of_car_l87_87270


namespace relatively_prime_subsequence_exists_l87_87156

theorem relatively_prime_subsequence_exists :
  ∃ (s : ℕ → ℕ), (∀ i j : ℕ, i ≠ j → Nat.gcd (2^(s i) - 3) (2^(s j) - 3) = 1) :=
by
  sorry

end relatively_prime_subsequence_exists_l87_87156


namespace eccentricities_ellipse_hyperbola_l87_87931

theorem eccentricities_ellipse_hyperbola :
  let a := 2
  let b := -5
  let c := 2
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  (x1 > 1) ∧ (0 < x2) ∧ (x2 < 1) :=
sorry

end eccentricities_ellipse_hyperbola_l87_87931


namespace common_divisors_count_9240_10010_l87_87435

def divisors (n : Nat) : Nat :=
(n.primeFactors.map (λ p => p.2 + 1)).foldl (· * ·) 1

theorem common_divisors_count_9240_10010 :
  let gcd_value := Nat.gcd 9240 10010;
  let num_common_divisors := divisors gcd_value;
  gcd_value = 210 ∧ num_common_divisors = 16 :=
by
  have : 9240 = 2^3 * 3^1 * 5^1 * 7^2 := by norm_num
  have : 10010 = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num
  have gcd_value_calc := Nat.gcd 9240 10010
  have : gcd_value_calc = 210 := by norm_num
  have num_common_divisors_calc := divisors gcd_value_calc
  have : num_common_divisors_calc = 16 := by norm_num
  exact ⟨this, by norm_num⟩

end common_divisors_count_9240_10010_l87_87435


namespace part_I_part_II_l87_87039

-- Part (I)
theorem part_I (a k : ℝ) (h : ∃ x y : ℝ, x^2 + 3 * y^2 = a^2 ∧ y = k * (x + 1)) : 
  a^2 > 3 * k^2 / (1 + 3 * k^2) := 
  sorry

-- Part (II)
theorem part_II (a k : ℝ) (h : ∀ x y : ℝ, x^2 + 3 * y^2 = a^2 ∧ y = k * (x + 1) ∧ 
    let ⟨x1, y1⟩ := some h in 
    let ⟨x2, y2⟩ := some (himplicity_of_h; ∀ A1 B1 : ℝ; ∥(-1 - A1, -B1) = 2 * ∥(B1 + 1, B2) := 
    x * y := 3 * abs k / (1 + 3 * k ^2; rfl :=
  x^2 + 3 * y^2 = 5 := 
  sorry

end part_I_part_II_l87_87039


namespace abs_eq_two_implies_l87_87798

theorem abs_eq_two_implies (x : ℝ) (h : |x - 3| = 2) : x = 5 ∨ x = 1 := 
sorry

end abs_eq_two_implies_l87_87798


namespace count_four_digit_even_numbers_l87_87607

-- Definitions for the conditions
def even_four_digit_numbers (digits : List ℕ) : ℕ :=
  (digits.filter (λ d => d % 2 == 0)).length

def valid_four_digit_number (n : ℕ) : Prop :=
  let n_digits := n.digits 10
  n_digits.length == 4 ∧ even_four_digit_numbers n_digits % 2 == 0 ∧ (n_digits.toFinset == {0, 1, 2, 3, 4, 5})

noncomputable def count_valid_numbers : ℕ :=
  (List.range 10000 10000).count (λ n => valid_four_digit_number n)

-- The proof statement
theorem count_four_digit_even_numbers : count_valid_numbers = 156 := 
by sorry

end count_four_digit_even_numbers_l87_87607


namespace log_sqrt_defined_in_interval_l87_87361

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l87_87361


namespace greatest_number_of_consecutive_even_integers_correct_number_of_consecutive_even_integers_l87_87951

theorem greatest_number_of_consecutive_even_integers (n : ℕ) (a : ℤ) :
  (∑ i in finset.range n, (2 * a + 2 * i)) = 180 →
  ∀ m : ℕ, (∑ i in finset.range m, (2 * a + 2 * i)) = 180 → m ≤ n :=
begin
  sorry
end

noncomputable def solution := 45

theorem correct_number_of_consecutive_even_integers (n : ℕ) (a : ℤ)
  (h : (∑ i in finset.range n, (2 * a + 2 * i)) = 180) : n = solution :=
begin
  sorry
end

end greatest_number_of_consecutive_even_integers_correct_number_of_consecutive_even_integers_l87_87951


namespace log_sqrt_defined_l87_87368

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l87_87368


namespace value_of_a_l87_87805

variable {a x : ℝ}

theorem value_of_a
  (h1 : ∀ x, (2 < x ∧ x < 4) ↔ (a - 1 < x ∧ x < a + 1)) :
  a = 3 :=
sorry

end value_of_a_l87_87805


namespace solve_for_z_l87_87128

theorem solve_for_z (z ω λ : ℂ) (h : |λ| ≠ 1) :
  (conj z - λ * z = ω) ↔ 
  (z = (conj λ * ω + conj ω) / (1 - |λ|^2)) :=
by
  sorry

end solve_for_z_l87_87128


namespace num_three_digit_integers_l87_87768

theorem num_three_digit_integers : 
  let digits := {3, 5, 7, 8, 9} in 
  ∀ (d1 d2 d3 : ℕ), 
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 →
  (d1 * 100 + d2 * 10 + d3).to_nat < 1000 →
  (finset.card (finset.filter (λ n, 100 ≤ n ∧ n < 1000) 
    (finset.image (λ (t : ℕ × ℕ × ℕ), t.1 * 100 + t.2 * 10 + t.3)
    ((finset.univ : finset ℕ).product (finset.univ : finset ℕ).product (finset.univ : finset ℕ)).filter 
    (λ t, t.1 ∈ digits ∧ t.2.1 ∈ digits ∧ t.2.2 ∈ digits ∧ t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2)))) = 60 :=
sorry

end num_three_digit_integers_l87_87768


namespace binomial_17_9_l87_87684

noncomputable def binom : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else (nat.choose n k)

theorem binomial_17_9 :
  (binom 15 6 = 5005) → (binom 15 8 = 6435) → binom 17 9 = 24310 :=
by
  intros h1 h2
  sorry

end binomial_17_9_l87_87684


namespace transform_123456_to_654321_l87_87641

-- Define the condition of what a valid move is
def valid_digit (n : Nat) : Prop := n ∈ List.range 10

-- Define the initial and target numbers
def initial_number : List Nat := [1, 2, 3, 4, 5, 6]
def target_number : List Nat := [6, 5, 4, 3, 2, 1]

-- Define the allowable moves
inductive Move
| swap (i j : Nat) : Move
| adjust (i : Nat) (delta : Int) : Move -- delta can be 3 or -3

-- Define the predicate for transformability in exactly 11 moves
def transformable_in_11_moves (start target : List Nat) : Prop :=
  ∃ moves : List Move, moves.length = 11 ∧
    (∀ m ∈ moves, match m with
     | Move.adjust i delta => (i < start.length ∧ (delta = 3 ∨ delta = -3) 
                               ∧ valid_digit (start.nth i).get_or_else 0 + delta) ∨ 
                             (i < start.length ∧ i + delta < start.length)
     | Move.swap i j => (i + 1 = j ∧ i < start.length ∧ j < start.length)
    ) ∧ 
    target = moves.foldl (λ x m =>
      match m with
        | Move.swap i j => match x.nth i, x.nth j with
                            | some xi, some xj => x.set i xj.set j xi
                            | _, _ => x
        | Move.adjust i delta => match x.nth i with
                                | some xi => let new_val :=xi + delta.toNat
                                             if valid_digit new_val then x.set i new_val else x
                                | _ => x
    ) start

theorem transform_123456_to_654321 : transformable_in_11_moves initial_number target_number :=
begin
  sorry
end

end transform_123456_to_654321_l87_87641


namespace sum_of_areas_constant_l87_87657

theorem sum_of_areas_constant (g : Type) (r : ℝ) (O P : g) (h1 h2 h3 : g) (OP : ℝ) (h1_perp_h2 : ∀ h1 h2, h1 ⊥ h2) (h1_thru_P : ∀ h1, P ∈ h1)
    (h2_thru_P : ∀ h2, P ∈ h2) (h3_thru_P : ∀ h3, P ∈ h3) :
    let circle_area (O : g) (r : ℝ) := π * r^2
    in (circle_area O ((r^2 - OP^2 + h1.straight_distinct_between(P) + h2.straight_distinct_between(P) + h3.straight_distinct_between(P)) / 2)) + 
       (circle_area O ((r^2 - OP^2 + h2.straight_distinct_between(P) + h1.straight_distinct_between(P) + h3.straight_distinct_between(P)) / 2)) +
       (circle_area O ((r^2 - OP^2 + h3.straight_distinct_between(P) + h1.straight_distinct_between(P) + h2.straight_distinct_between(P)) / 2)) =
    (3 * π * r^2 - π * OP^2) :=
sorry

end sum_of_areas_constant_l87_87657


namespace no_valid_star_placement_l87_87304

theorem no_valid_star_placement : 
  ¬ ∃ (stars : Fin 10 → Fin 10 → Prop),
    (∀ i j, stars i j → stars (i / 2) (j / 2) = 2) ∧
    (∀ i j, stars i j → stars (i / 3) (j / 1) = 1) ∧
    (∀ i j, ∀ k l, i ≠ k ∨ j ≠ l → stars i j ∧ stars k l → false) :=
sorry

end no_valid_star_placement_l87_87304


namespace irrational_inradius_l87_87519

theorem irrational_inradius 
  (b c : ℕ) 
  (h1 : 1 + b > c) 
  (h2 : 1 + c > b) 
  (h3 : b + c > 1) 
  (h4 : b = c) : 
  ¬ ∃ r : ℚ, r = (sqrt (b^2 - (1/4))) / (1 + 2*b) := 
sorry

end irrational_inradius_l87_87519


namespace exists_nat_b_digit_sum_fact_ge_10_pow_100_l87_87886

-- Sum of the digits function definition
def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

-- Main proof statement
theorem exists_nat_b_digit_sum_fact_ge_10_pow_100 : ∃ (b : ℕ), ∀ (n : ℕ), n > b → sum_of_digits (n !) ≥ 10 ^ 100 :=
  sorry

end exists_nat_b_digit_sum_fact_ge_10_pow_100_l87_87886


namespace exists_colored_subset_l87_87127

theorem exists_colored_subset (n : ℕ) (h_positive : n > 0) (colors : ℕ → ℕ) (h_colors : ∀ a b : ℕ, a < b → a + b ≤ n → 
  (colors a = colors b ∨ colors b = colors (a + b) ∨ colors a = colors (a + b))) :
  ∃ c, ∃ s : Finset ℕ, s.card ≥ (2 * n / 5) ∧ ∀ x ∈ s, colors x = c :=
sorry

end exists_colored_subset_l87_87127


namespace pointA_in_second_quadrant_l87_87097

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l87_87097


namespace savings_ratio_l87_87481

variable (S1 D1 : ℝ)
def S2 : ℝ := 1.1 * S1
def D2 : ℝ := 1.05 * D1
def X : ℝ := 0.099 * S1 + 0.126 * D1
def Y : ℝ := 0.06 * S1 + 0.08 * D1

theorem savings_ratio (h1 : S1 ≠ 0) (h2 : D1 ≠ 0) : X / Y = 1.6071 :=
by
  sorry

end savings_ratio_l87_87481


namespace sandy_molly_ratio_l87_87165

-- Define Sandy's age and the age difference
def sandy_age : ℕ := 63
def age_difference : ℕ := 18

-- Define Molly's age based on the given conditions
def molly_age : ℕ := sandy_age + age_difference

-- Compute the simplified ratio of Sandy's age to Molly's age
def gcd (a b : ℕ) : ℕ := 
  match b with
  | 0 => a
  | _ => gcd b (a % b)

def ratio (s m : ℕ) : ℕ × ℕ :=
  let g := gcd s m
  (s / g, m / g)

-- Prove that the ratio of Sandy's age to Molly's age is 7:9
theorem sandy_molly_ratio : ratio sandy_age molly_age = (7, 9) :=
by
  sorry

end sandy_molly_ratio_l87_87165


namespace unique_polynomial_with_three_integer_roots_l87_87687

theorem unique_polynomial_with_three_integer_roots
  (b : Fin 7 → ℕ)
  (h : ∀ i, b i = 0 ∨ b i = 1)
  (h_root_zero : ∃ x : ℕ, b x = 0) :
  set.count (set_of (p : polynomial ℤ) (polynomial.degree p = 7
    ∧ (∀ x, p.coeff x = b x) 
    ∧ (∀ r : ℤ, is_root p r → r = 0 ∨ r = 1 ∨ r = -1)
    ∧ (finite (root_set p ℤ))
    ∧ (set.count (root_set p ℤ) = 3)) 
  = 1 :=
sorry

end unique_polynomial_with_three_integer_roots_l87_87687


namespace complex_add_conjugate_magnitude_l87_87006

open Complex

theorem complex_add_conjugate_magnitude (z : ℂ) (hz : z = -1/2 - (sqrt 3 / 2) * I) :
    conj z + Complex.abs z = 1/2 + (sqrt 3 / 2) * I := by
  sorry

end complex_add_conjugate_magnitude_l87_87006


namespace incorrect_statement_l87_87862

/-- Given four distinct points in space, let us define four statements regarding the relationships between these points. -/
theorem incorrect_statement
  {A B C D : Type} [T : topological_space ℝ] [affine_space ℝ] 
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (SAB : line ℝ A B) (SCD : line ℝ C D)
  (SAC : line ℝ A C) (SBD : line ℝ B D)
  (SAD : line ℝ A D) (SBC : line ℝ B C) :
  ((¬ (A = B) ∧ ¬ (C = D)) ∧ (SAB ∩ SCD = ∅ → ¬(parallel SAB SCD)) ∧
  (coplanar SAC SBD → coplanar SAD SBC) ∧ (skew SAC SBD → skew SAD SBC) ∧ 
  ((A = C) ∧ (B = D) → perp SAD SBC)) :=
begin
  sorry
end

end incorrect_statement_l87_87862


namespace lending_interest_rate_l87_87643

theorem lending_interest_rate 
  (principal_borrowed : ℝ)
  (rate_borrowed : ℝ)
  (time_borrowed : ℝ)
  (gain_per_year : ℝ)
  (interest_paid : ℝ := (principal_borrowed * rate_borrowed * time_borrowed) / 100)
  (total_gain : ℝ := gain_per_year * time_borrowed)
  (interest_received : ℝ := total_gain + interest_paid) :
  let R := (interest_received * 100) / (principal_borrowed * time_borrowed)
  in R = 6 :=
sorry

end lending_interest_rate_l87_87643


namespace intersection_empty_when_a_is_minus_one_range_of_a_given_B_subset_complement_A_l87_87784

open Set

variable (U : Type) [LinearOrder U]

def A : Set U := { x | x < -1 }

def B (a : U) : Set U := { x | (2 : U) * a < x ∧ x < a + 3 }

theorem intersection_empty_when_a_is_minus_one : A (-1 : U) ∩ B U (-1 : U) = ∅ := sorry

theorem range_of_a_given_B_subset_complement_A : 
  (∀ a : U, (B U a) ⊆ (A U)ᶜ) → ∀ a, a ∈ Ici (-((1/2 : U))) := sorry

end intersection_empty_when_a_is_minus_one_range_of_a_given_B_subset_complement_A_l87_87784


namespace light_bulbs_in_kitchen_l87_87568

-- Definitions of conditions
def broken_ratio_kitchen : ℚ := 3 / 5
def broken_ratio_foyer : ℚ := 1 / 3
def broken_foyer : ℕ := 10
def not_broken_total : ℕ := 34

def total_foyer (b_f : ℕ) (r_f : ℚ) : ℕ := b_f / r_f
def not_broken_foyer (t_f : ℕ) (b_f : ℕ) : ℕ := t_f - b_f
def not_broken_kitchen (nb_t : ℕ) (nb_f : ℕ) : ℕ := nb_t - nb_f
def total_kitchen (nb_k : ℕ) (r_k : ℚ) : ℕ := nb_k / (1 - r_k)

-- Statement to prove
theorem light_bulbs_in_kitchen :
  total_kitchen (not_broken_kitchen not_broken_total (not_broken_foyer (total_foyer broken_foyer broken_ratio_foyer) broken_foyer)) (broken_ratio_kitchen) = 35 :=
sorry

end light_bulbs_in_kitchen_l87_87568


namespace systematic_sampling_sum_l87_87451

theorem systematic_sampling_sum (a b : ℕ) (h1 : a = 4 + 10) (h2 : b = 4 + 30) : a + b = 48 := 
by 
  rw [h1, h2]
  norm_num
  sorry

end systematic_sampling_sum_l87_87451


namespace better_regression_performance_l87_87835

-- Define the variables used in the problem
variables (total_sum_of_squares_of_deviations sum_of_squares_of_residuals sum_of_squares_due_to_regression : ℝ)

-- Define the condition given in the problem
def condition := sum_of_squares_due_to_regression = total_sum_of_squares_of_deviations - sum_of_squares_of_residuals

-- State the theorem to prove that a smaller total sum of squares of deviations indicates better regression performance
theorem better_regression_performance (h: condition total_sum_of_squares_of_deviations sum_of_squares_of_residuals sum_of_squares_due_to_regression):
  ∀ total1 total2, total1 > total2 → total_sum_of_squares_of_deviations = total1 → total_sum_of_squares_of_deviations = total2 → better_regression_performance :=
sorry

end better_regression_performance_l87_87835


namespace max_positive_integers_in_circle_l87_87933

theorem max_positive_integers_in_circle (a : ℕ → ℤ) (h_circle : ∀ i : ℕ, i < 100 → a i > a ((i + 1) % 100) + a ((i + 2) % 100)) : 
  ∃ S : finset (fin 100), S.card = 49 ∧ ∀ i ∈ S, 0 < a i :=
sorry

end max_positive_integers_in_circle_l87_87933


namespace fibonacci_lucas_polynomials_l87_87894

-- Definitions based on the problem conditions
def F (n : ℕ) (x : ℝ) : ℝ := sorry
def L (n : ℕ) (x : ℝ) : ℝ := sorry

-- Given recurrence relations
def F_recurrence (n : ℕ) (x : ℝ) : Prop := 
  F (n + 1) x = x * F n x + F (n - 1) x

def L_recurrence (n : ℕ) (x : ℝ) : Prop :=
  L (n + 1) x = 2 * F (n + 2) x - x * F (n + 1) x

-- Initial conditions
def F_initial_conditions : Prop :=
  F 0 x = 0 ∧ F 1 x = 1

-- Proof statement for the explicit forms using the binomial coefficients
theorem fibonacci_lucas_polynomials (n : ℕ) (x : ℝ) :
  F (n + 1) x = ∑ k in Finset.range (n / 2 + 1), Nat.choose (n - k) k * x^(n - 2 * k) ∧
  L n x = ∑ k in Finset.range (n / 2 + 1), (Nat.choose (n - k) k + Nat.choose (n - k - 1) (k - 1)) * x^(n - 2 * k)
:= sorry

end fibonacci_lucas_polynomials_l87_87894


namespace prob_between_300_and_425_prob_not_more_than_450_prob_more_than_300_l87_87932

noncomputable theory
open Real

def normal_dist (a σ : ℝ) (X : ℝ) :=
  (1 / (σ * sqrt (2 * π))) * exp (-(X - a) ^ 2 / (2 * σ ^ 2))

def Φ (z : ℝ) : ℝ :=
  (1 / 2) * (1 + erf (z / sqrt(2)))

theorem prob_between_300_and_425 :
  let a := 375
  let σ := 25
  P(300 < X ∧ X < 425) = Φ (2) - Φ (-3) := sorry

theorem prob_not_more_than_450 :
  let a := 375
  let σ := 25
  P(X ≤ 450) = Φ (3) := sorry

theorem prob_more_than_300 :
  let a := 375
  let σ := 25
  P(X > 300) = 1 - Φ (-3) := sorry

end prob_between_300_and_425_prob_not_more_than_450_prob_more_than_300_l87_87932


namespace interest_rate_per_annum_l87_87640

/--
Given:
- Simple Interest (SI) after 3 years is 3600
- Principal (P) is 10000
- Time (T) is 3 years

Prove:
- The interest rate (R) per annum is 0.12 or 12%
-/

theorem interest_rate_per_annum (SI P T : ℝ) (h1 : SI = 3600) (h2 : P = 10000) (h3 : T = 3) :
  let R := SI / (P * T) in R * 100 = 12 :=
by
  sorry

end interest_rate_per_annum_l87_87640


namespace profit_distribution_l87_87877

theorem profit_distribution (investment_LiWei investment_WangGang profit total_investment : ℝ)
  (h1 : investment_LiWei = 16000)
  (h2 : investment_WangGang = 12000)
  (h3 : profit = 14000)
  (h4 : total_investment = investment_LiWei + investment_WangGang) :
  (profit * (investment_LiWei / total_investment) = 8000) ∧ 
  (profit * (investment_WangGang / total_investment) = 6000) :=
by
  sorry

end profit_distribution_l87_87877


namespace distance_between_points_of_intersection_l87_87711

-- Define the equations as functions
def equation1 (x y : ℝ) : Prop := x^2 + y = 12
def equation2 (x y : ℝ) : Prop := x + y = 12

-- The main theorem statement
theorem distance_between_points_of_intersection :
  let P := (0, 12)
  let Q := (1, 11)
  dist P Q = real.sqrt 2 := 
by 
  -- Placeholder for the proof steps
  sorry

-- Define the distance function for points in 2D
def dist (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

end distance_between_points_of_intersection_l87_87711


namespace problem1_problem2_l87_87385

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 2*k - 24 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := 3 - k > 0 ∧ 3 + k < 0

-- Proof problem 1: If q is true, then k ∈ (-∞, -3)
theorem problem1 (k : ℝ) : q k → k ∈ set.Iio (-3) :=
  sorry

-- Proof problem 2: If p ∨ q is true and p ∧ q is false, then k ∈ (-∞, -4) ∪ [-3, 6]
theorem problem2 (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k ∈ set.Iio (-4) ∪ set.Icc (-3) 6 :=
  sorry

end problem1_problem2_l87_87385


namespace collinear_dot_probability_computation_l87_87467

def collinear_dot_probability : ℚ := 12 / Nat.choose 25 5

theorem collinear_dot_probability_computation :
  collinear_dot_probability = 12 / 53130 :=
by
  -- This is where the proof steps would be if provided.
  sorry

end collinear_dot_probability_computation_l87_87467


namespace AreaProportionality_l87_87738

variables {M A B C D A1 B1 C1 D1 : Point}
variables {S_BCD S_ABD S_ABC S_ACD : ℝ}
variables {MA MA1 MB MB1 MC MC1 MD MD1 : ℝ}

-- Assuming the geometry and intersection properties of points
def ConvexPyramid (M A B C D : Point) : Prop := sorry  -- Definition of convex pyramid

-- Assuming proportional intersection properties
def ProportionalIntersection (M A1 A MB1 B MC1 C MD1 D : Point) : Prop := sorry  -- Definition of proportional segments

theorem AreaProportionality 
  (h : ConvexPyramid M A B C D) 
  (hIntersect : ProportionalIntersection M A1 A MB1 B MC1 C MD1 D) :
  S_BCD * (MA / MA1) + S_ABD * (MC / MC1) = S_ABC * (MD / MD1) + S_ACD * (MB / MB1) := 
sorry

end AreaProportionality_l87_87738


namespace slope_at_1_monotonicity_range_of_a_l87_87771

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x + a / (x + 1)

theorem slope_at_1 (a : ℝ) (ha : a = 3) : 
    (deriv (λ x, f x a) 1 = 1 / 4) := sorry

theorem monotonicity (a : ℝ) (ha : 0 ≤ a) : 
    (∀ x, (0 < x ∧ x < (a - 2 + sqrt ((a - 2)^2 + 4)) / 2) → deriv (λ x, f x a) x < 0) ∧
    (∀ x, ((a - 2 + sqrt ((a - 2)^2 + 4)) / 2 < x) → deriv (λ x, f x a) x > 0) := sorry

theorem range_of_a (a : ℝ) (hf : ∀ x > 0, f x a ≤ (2016 - a) * x^3 + (x^2 + a - 1) / (x + 1)) :
    4 < a ∧ a ≤ 2016 := sorry

end slope_at_1_monotonicity_range_of_a_l87_87771


namespace min_value_of_x_squared_plus_6x_l87_87236

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l87_87236


namespace painting_time_equation_l87_87430

theorem painting_time_equation
  (Hannah_rate : ℝ)
  (Sarah_rate : ℝ)
  (combined_rate : ℝ)
  (temperature_factor : ℝ)
  (break_time : ℝ)
  (t : ℝ)
  (condition1 : Hannah_rate = 1 / 6)
  (condition2 : Sarah_rate = 1 / 8)
  (condition3 : combined_rate = (Hannah_rate + Sarah_rate) * temperature_factor)
  (condition4 : temperature_factor = 0.9)
  (condition5 : break_time = 1.5) :
  (combined_rate * (t - break_time) = 1) ↔ (t = 1 + break_time + 1 / combined_rate) :=
by
  sorry

end painting_time_equation_l87_87430


namespace bunchkin_total_distance_l87_87348

theorem bunchkin_total_distance
  (a b c d e : ℕ)
  (ha : a = 17)
  (hb : b = 43)
  (hc : c = 56)
  (hd : d = 66)
  (he : e = 76) :
  (a + b + c + d + e) / 2 = 129 :=
by
  sorry

end bunchkin_total_distance_l87_87348


namespace ratio_is_correct_l87_87804

-- Declare the variables needed for our problem
variables (r : ℝ) (h₀ : r ≠ 0)

-- Define the original circumference
def original_circumference (r : ℝ) : ℝ := 2 * real.pi * r

-- Define the new radius
def new_radius (r : ℝ) : ℝ := r + 2

-- Define the new circumference
def new_circumference (r : ℝ) : ℝ := 2 * real.pi * (new_radius r)

-- Define the ratio of the new circumference to the original circumference
def ratio (r : ℝ) : ℝ := new_circumference r / original_circumference r

-- Theorem to prove the ratio is equal to 1 + 2 / r
theorem ratio_is_correct : ratio r = 1 + 2 / r :=
by {
  sorry
}

end ratio_is_correct_l87_87804


namespace log_over_sqrt_defined_l87_87358

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l87_87358


namespace number_of_white_balls_possible_l87_87811

theorem number_of_white_balls_possible (t r w d : ℕ) (h_total : t = 10)
  (h_red : r = 7) (h_white : w = 3) (h_drawn : d = 3) :
  ∃ n, n ∈ {0, 1, 2, 3} ∧ n = d - r ∨ n = d - r + 1 ∨ n = d - r + 2 ∨ n = d - r + 3 := sorry

end number_of_white_balls_possible_l87_87811


namespace calculate_Y_payment_l87_87941

theorem calculate_Y_payment (X Y : ℝ) (h1 : X + Y = 600) (h2 : X = 1.2 * Y) : Y = 600 / 2.2 :=
by
  sorry

end calculate_Y_payment_l87_87941


namespace range_of_x_l87_87723

theorem range_of_x (x : ℝ) (h1 : 0 < x + 1) (h2 : 0 < 3 - x) (h3 : log (x + 1) < log (3 - x)) : -1 < x ∧ x < 1 := sorry

end range_of_x_l87_87723


namespace find_n_l87_87866

theorem find_n (x y : ℤ) (h1 : x = 3) (h2 : y = -3) : 
  let n := x - y ^ (x - 2 * y) in 
  n = 19686 := by
  sorry

end find_n_l87_87866


namespace ellipse_hyperbola_eccentricities_l87_87565

theorem ellipse_hyperbola_eccentricities :
  ∃ x y : ℝ, (2 * x^2 - 5 * x + 2 = 0) ∧ (2 * y^2 - 5 * y + 2 = 0) ∧ 
  ((2 > 1) ∧ (0 < (1/2) ∧ (1/2 < 1))) :=
by
  sorry

end ellipse_hyperbola_eccentricities_l87_87565


namespace Tyler_scissors_count_l87_87584

variable (S : ℕ)

def Tyler_initial_money : ℕ := 100
def cost_per_scissors : ℕ := 5
def number_of_erasers : ℕ := 10
def cost_per_eraser : ℕ := 4
def Tyler_remaining_money : ℕ := 20

theorem Tyler_scissors_count :
  Tyler_initial_money - (cost_per_scissors * S + number_of_erasers * cost_per_eraser) = Tyler_remaining_money →
  S = 8 :=
by
  sorry

end Tyler_scissors_count_l87_87584


namespace sum_of_z_values_l87_87491

def f (x : ℚ) : ℚ := x^2 + x + 1

theorem sum_of_z_values : ∃ z₁ z₂ : ℚ, f (4 * z₁) = 12 ∧ f (4 * z₂) = 12 ∧ (z₁ + z₂ = - 1 / 12) :=
by
  sorry

end sum_of_z_values_l87_87491


namespace bananas_to_grapes_l87_87296

theorem bananas_to_grapes (cost_per_4_bananas_cost_3_oranges : ℕ)
                           (cost_per_5_oranges_cost_8_grapes : ℕ) :
  let bananas := 16,
      oranges := (bananas * 3) / 4,
      grapes := (oranges * 8) / 5
  in grapes = 19 :=
by 
  let bananas := 16
  let oranges := (bananas * 3) / 4
  let grapes := (oranges * 8) / 5
  show grapes = 19
  sorry

end bananas_to_grapes_l87_87296


namespace pieces_left_l87_87616

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left_l87_87616


namespace isosceles_DXY_l87_87054

/-- I is the incenter of triangle ABC. 
    D is outside the triangle such that DA ∥ BC and DB = AC, but ABCD is not a parallelogram.
    The angle bisector of ∠BDC meets the line through I perpendicular to BC at X.
    The circumcircle of △CDX meets the line BC again at Y.
    Show that △DXY is isosceles --/

theorem isosceles_DXY
  (A B C D I X Y : Point)
  (I_incenter : incenter I A B C)
  (D_outside : ¬inside D (triangle A B C))
  (DA_parallel_BC : DA ∥ BC)
  (DB_eq_AC : DB = AC)
  (ABCD_not_parallelogram : ¬parallelogram A B C D)
  (angle_bisector_BDC : bisects_angle (BD_CD X) (BCD B C))
  (X_on_perpendicular : perpendicular (line_through I) (BC_center_line X))
  (Y_on_circumcircle : (circumcircle (triangle CDX)) Y)
  : isosceles_triangle D X Y :=
begin
  sorry
end

end isosceles_DXY_l87_87054


namespace mod_equiv_pow_diff_l87_87683

theorem mod_equiv_pow_diff (n : ℕ) : (47^1860 - 25^1860) % 6 = 0 := by
  -- Given conditions
  have h1 : 47 % 6 = 5 := by norm_num
  have h2 : 25 % 6 = 1 := by norm_num
  
  -- Simplification
  have h3 : (47^1860 - 25^1860) % 6 = (5^1860 - 1^1860) % 6 := by
    rw [← pow_mod, h1, ← pow_mod, h2]
  
  -- Calculation
  have h4 : 5^1860 % 6 = 1 := by sorry
  have h5 : 1^1860 % 6 = 1 := by norm_num
  
  -- Substitution and final result
  rw [h3, h4, h5]
  norm_num

end mod_equiv_pow_diff_l87_87683


namespace slope_of_line_l87_87929

-- Defining the parametric equations of the line
def parametric_x (t : ℝ) : ℝ := 3 + 4 * t
def parametric_y (t : ℝ) : ℝ := 4 - 5 * t

-- Stating the problem in Lean: asserting the slope of the line
theorem slope_of_line : 
  (∃ (m : ℝ), ∀ t : ℝ, parametric_y t = m * parametric_x t + (4 - 3 * m)) 
  → (∃ m : ℝ, m = -5 / 4) :=
  by sorry

end slope_of_line_l87_87929


namespace cost_of_red_socks_l87_87133

/-- Given the conditions
- num_red is the number of pairs of red socks Luis bought
- num_blue is the number of pairs of blue socks Luis bought
- cost_blue is the cost of each pair of blue socks
- total_cost is the total amount spent

Prove that the cost of each pair of red socks is $3. -/
theorem cost_of_red_socks 
  (num_red num_blue : ℕ) 
  (cost_blue total_cost : ℕ) 
  (h1 : num_red = 4) 
  (h2 : num_blue = 6) 
  (h3 : cost_blue = 5) 
  (h4 : total_cost = 42) :
  (R : ℕ) (h_red_cost : num_red * R + num_blue * cost_blue = total_cost) -> R = 3 :=
  sorry

end cost_of_red_socks_l87_87133


namespace min_value_x_squared_plus_6x_l87_87240

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l87_87240


namespace correct_option_is_C_l87_87244

theorem correct_option_is_C : 
  let A := "The square root of 36 is 6"
  let B := "Negative numbers do not have cube roots"
  let C := "The arithmetic square root that equals itself is 0 and 1"
  let D := "The square root of |81| is ±9"
  (A = "The square root of 36 is ±6" → false) ∧
  (B = "Negative numbers do have cube roots" → false) ∧
  (C = "The arithmetic square root that equals itself is 0 and 1" → true) ∧
  (D = "The square root of |81| is 9" → false) → 
  C :=
begin
  intros,
  sorry
end

end correct_option_is_C_l87_87244


namespace bus_travelled_distance_l87_87282

-- Define the conditions
def travel_time_minutes := 42
def travel_time_hours : ℝ := travel_time_minutes / 60.0
def speed_mph := 50
def distance := speed_mph * travel_time_hours

-- Prove the distance traveled is 35 miles
theorem bus_travelled_distance : distance = 35 := by
  sorry

end bus_travelled_distance_l87_87282


namespace find_foci_l87_87978

noncomputable def foci_of_ellipse : Prop := 
  let h := 2
  let k := -3
  let a := 5
  let b := 4
  let c := Real.sqrt (a^2 - b^2)
  (h - c = -1) ∧ (h + c = 5) ∧ (k = -3)

theorem find_foci : foci_of_ellipse := by
  let h := 2
  let k := -3
  let a := 5
  let b := 4
  let c := Real.sqrt (a^2 - b^2)
  have hc : c = 3 := by
    calc
      c = Real.sqrt (a^2 - b^2) : rfl
      ... = Real.sqrt (25 - 16) : by
        simp [a, b]
      ... = Real.sqrt 9 : by
        simp
      ... = 3 : Real.sqrt_eq_rfl _,
  simp [h, k, hc],
  exact And.intro (by calc
    h - c = 2 - 3 : by simp [h, hc]
         ... = -1 : by simp) (by
    calc
      h + c = 2 + 3 : by simp [h, hc]
            ... = 5 : by simp)

end find_foci_l87_87978


namespace largest_possible_integer_l87_87274

def list_properties (lst : List ℕ) : Prop :=
  lst.length = 5 ∧
  (1 ≤ lst.count 7) ∧ (lst.filter (λ x, x ≠ 7)).Nodup ∧ 
  lst.sorted.nth 2 = some 10 ∧
  (lst.sum / lst.length = 12)

theorem largest_possible_integer (lst : List ℕ) (a b d e : ℕ) :
  list_properties lst →
  lst.sorted = [a, b, 10, d, e] →
  a = 7 ∧ b = 7 →
  d > 10 →
  e = 36 - d →
  ∀ x ∈ lst, x ≤ 25 ∧ 25 ∈ lst :=
sorry

end largest_possible_integer_l87_87274


namespace pencil_eraser_cost_l87_87132

theorem pencil_eraser_cost (p e : ℕ) (hp : p > e) (he : e > 0)
  (h : 20 * p + 4 * e = 160) : p + e = 12 :=
sorry

end pencil_eraser_cost_l87_87132


namespace lines_parallel_if_perpendicular_to_same_plane_l87_87735

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables perpendicular : Line → Plane → Prop
variables parallel : Line → Line → Prop

-- Define that m and n are two different straight lines
axiom different_lines : m ≠ n

-- Define the conditions
axiom m_perpendicular_α : perpendicular m α
axiom n_perpendicular_α : perpendicular n α

-- Prove that m is parallel to n
theorem lines_parallel_if_perpendicular_to_same_plane :
  perpendicular m α →
  perpendicular n α →
  parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l87_87735


namespace problem_f_l87_87911

def f (x : ℝ) : ℝ := 2 * (Real.cos (x - Real.pi / 4)) ^ 2 - 1

theorem problem_f (x : ℝ) :
  f x = -(f (-x)) ∧ ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi := by
  sorry

end problem_f_l87_87911


namespace exists_n_divides_expression_l87_87125

theorem exists_n_divides_expression (k : ℕ) (hk1 : k ≥ 1) (hk2 : Nat.coprime k 6) :
  ∃ n : ℕ, k ∣ (2^n + 3^n + 6^n - 1) :=
sorry

end exists_n_divides_expression_l87_87125


namespace local_minima_count_l87_87318

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem local_minima_count : 
  ∃ n, n = 2016 ∧ (∀ x ∈ Set.Icc 0 (2015 * Real.pi), is_local_min f x → x ∈ (Set.Icc 0 (2015 * Real.pi))) :=
by
  sorry

end local_minima_count_l87_87318


namespace dance_pairs_exist_l87_87672

variable {Boy Girl : Type} 

-- Define danced_with relation
variable (danced_with : Boy → Girl → Prop)

-- Given conditions
variable (H1 : ∀ (b : Boy), ∃ (g : Girl), ¬ danced_with b g)
variable (H2 : ∀ (g : Girl), ∃ (b : Boy), danced_with b g)

-- Proof that desired pairs exist
theorem dance_pairs_exist :
  ∃ (M1 M2 : Boy) (D1 D2 : Girl),
    danced_with M1 D1 ∧
    danced_with M2 D2 ∧
    ¬ danced_with M1 D2 ∧
    ¬ danced_with M2 D1 :=
sorry

end dance_pairs_exist_l87_87672


namespace min_mag_of_z3_l87_87030

noncomputable def min_mag_z3 (z1 z2 z3 : ℂ) : ℝ :=
  if (|z1| = 1 ∧ |z2| = 1 ∧ |z1 + z2 + z3| = 1 ∧ (∀ t:ℂ, z1 / z2 = t ∧ t.im = t)) 
  then (Real.sqrt 2 - 1)
  else 0

theorem min_mag_of_z3 (z1 z2 z3 : ℂ) (hz1 : |z1| = 1) (hz2 : |z2| = 1)
  (hz_sum : |z1 + z2 + z3| = 1) (hz1hz2_imag : (z1 / z2).re = 0) :
  |z3| = Real.sqrt 2 - 1 :=
by
  sorry

end min_mag_of_z3_l87_87030


namespace salmon_weight_l87_87883

theorem salmon_weight :
  ∃ (wt_per_salmon : ℝ),
    let wt_trout := 8 in
    let wt_bass := 6 * 2 in
    let total_needed := 22 * 2 in
    let wt_non_salmon := wt_trout + wt_bass in
    let remaining_weight := total_needed - wt_non_salmon in
    remaining_weight / 2 = wt_per_salmon ∧ wt_per_salmon = 12 :=
begin
  sorry
end

end salmon_weight_l87_87883


namespace least_n_factorial_multiple_of_840_l87_87801

theorem least_n_factorial_multiple_of_840 :
  ∃ (n : ℕ), n ≥ 7 ∧ (∃ (k : ℕ), (n.factorial = 840 * k)) :=
sorry

end least_n_factorial_multiple_of_840_l87_87801


namespace lee_income_l87_87466

variable (q : ℝ) (X : ℝ) (S : ℝ)

-- Defining the conditions as given in the problem
def income_tax (X : ℝ) (q : ℝ) : ℝ :=
  if X <= 24000 then q * 0.01 * X
  else 24000 * q * 0.01 + (q + 3) * 0.01 * (X - 24000)

-- Given that income tax paid, S, amounts to (q + 0.45)% of the annual income X
def total_tax (S : ℝ) (X : ℝ) (q : ℝ) : Prop :=
  S = (q + 0.45) * 0.01 * X

-- The goal is to prove that given the conditions, Lee's annual income X is 28235
theorem lee_income (h1 : S = income_tax X q)
                   (h2 : total_tax S X q) :
  X = 28235 := 
sorry

end lee_income_l87_87466


namespace combinations_to_make_30_cents_l87_87048

theorem combinations_to_make_30_cents :
  ∃ (combinations : ℕ), combinations = 20 ∧ 
  (combinations = 
    count_combinations (1, 5, 10) 30 (tuple.zip_with (λ n dm, dm * n) ([1, 1, 1], [1, 5, 10])))
:= sorry

noncomputable def count_combinations (values : list ℕ) (target : ℕ) : ℕ :=
  (count_combinations_helper values target []).length

@[simp]
def count_combinations_helper : list ℕ → ℕ → list (list ℕ) → list (list ℕ)
| values, 0, acc := [acc]
| [], target, acc := []
| (v::vs), target, acc :=
  if target < 0 then []
  else
    count_combinations_helper (v::vs) (target - v) (v::acc) ++
    count_combinations_helper vs target acc

#eval count_combinations [1, 5, 10] 30  -- This should output 20

end combinations_to_make_30_cents_l87_87048


namespace total_local_percentage_approx_52_74_l87_87255

-- We provide the conditions as definitions
def total_arts_students : ℕ := 400
def local_arts_percentage : ℝ := 0.50
def total_science_students : ℕ := 100
def local_science_percentage : ℝ := 0.25
def total_commerce_students : ℕ := 120
def local_commerce_percentage : ℝ := 0.85

-- Calculate the expected total percentage of local students
noncomputable def calculated_total_local_percentage : ℝ :=
  let local_arts_students := local_arts_percentage * total_arts_students
  let local_science_students := local_science_percentage * total_science_students
  let local_commerce_students := local_commerce_percentage * total_commerce_students
  let total_local_students := local_arts_students + local_science_students + local_commerce_students
  let total_students := total_arts_students + total_science_students + total_commerce_students
  (total_local_students / total_students) * 100

-- State what we need to prove
theorem total_local_percentage_approx_52_74 :
  abs (calculated_total_local_percentage - 52.74) < 1 :=
sorry

end total_local_percentage_approx_52_74_l87_87255


namespace complete_square_l87_87958

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end complete_square_l87_87958


namespace house_number_proof_l87_87543

theorem house_number_proof
  (numbers : List ℕ)
  (VovaNums DimaNums : List ℕ)
  (x : ℕ)
  (h_numbers : numbers = [1, 3, 4, 6, 8, 9, 11, 12, 16])
  (h_all_sum : numbers.sum = 70)
  (h_Vova_sum : VovaNums.sum = 3 * DimaNums.sum)
  (h_partitions : VovaNums.length = 4 ∧ DimaNums.length = 4 ∧ 
                  (numbers.filter (λ n, n ∉ VovaNums ∧ n ∉ DimaNums)).length = 1)
  (h_partitions_sum : VovaNums.sum + DimaNums.sum + x = 70)
  (h_mod_cond : x % 4 = 2) :
  x = 6 := 
sorry

end house_number_proof_l87_87543


namespace arithmetic_sequence_k_value_l87_87563

theorem arithmetic_sequence_k_value (a : ℕ → ℤ) (S: ℕ → ℤ)
    (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
    (h2 : S 11 = S 4)
    (h3 : a 1 = 1)
    (h4 : ∃ k, a k + a 4 = 0) :
    ∃ k, k = 12 :=
by 
  sorry

end arithmetic_sequence_k_value_l87_87563


namespace markup_percentage_l87_87610

variable (W R : ℝ) -- W for Wholesale Cost, R for Retail Cost

-- Conditions:
-- 1. The sweater is sold at a 40% discount.
-- 2. When sold at a 40% discount, the merchant nets a 30% profit on the wholesale cost.
def discount_price (R : ℝ) : ℝ := 0.6 * R
def profit_price (W : ℝ) : ℝ := 1.3 * W

-- Hypotheses
axiom wholesale_cost_is_positive : W > 0
axiom discount_condition : discount_price R = profit_price W

-- Question: Prove that the percentage markup from wholesale to retail price is 116.67%.
theorem markup_percentage (W R : ℝ) 
  (wholesale_cost_is_positive : W > 0)
  (discount_condition : discount_price R = profit_price W) :
  ((R - W) / W * 100) = 116.67 := by
  sorry

end markup_percentage_l87_87610


namespace largest_circle_area_l87_87644

theorem largest_circle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l = 200) :
  let perimeter := 2 * (w + l)
  let r := perimeter / (2 * Real.pi)
  π * r^2 = 900 / π :=
by
  let perimeter := 2 * (w + l)
  let r := perimeter / (2 * Real.pi)
  have perimeter_calc : perimeter = 60 := by
    sorry
  have r_calc : r = 30 / Real.pi := by
    sorry
  show π * r^2 = 900 / π from
    by
      calc
        π * r^2 = π * (30 / π)^2 : by rw [r_calc]
        ... = π * (30^2 / π^2) : by sorry
        ... = π * (900 / π^2) : by sorry
        ... = 900 / π : by sorry

end largest_circle_area_l87_87644


namespace equation_verification_l87_87707

theorem equation_verification :
  (96 / 12 = 8) ∧ (45 - 37 = 8) := 
by
  -- We can add the necessary proofs later
  sorry

end equation_verification_l87_87707


namespace pointA_in_second_quadrant_l87_87096

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l87_87096


namespace negate_proposition_l87_87925

theorem negate_proposition (x : ℝ) :
  (¬(x > 1 → x^2 > 1)) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negate_proposition_l87_87925


namespace no_real_solution_l87_87327

noncomputable def f : ℝ → ℝ := λ x => 2^x + 3^x - 4^x

theorem no_real_solution : ∀ x : ℝ, f(x) ≠ 4 :=
begin
  intros x,
  sorry
end

end no_real_solution_l87_87327


namespace focus_chord_property_l87_87422

noncomputable theory

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the line equation intersecting the parabola
def line (x y k : ℝ) : Prop := y = k * (x - 2)

-- Define the formula for the reciprocals of the distances
def distance_reciprocal (P Q : ℝ × ℝ) : ℝ :=
  1 / (|P.1 + 2|) + 1 / (|Q.1 + 2|)

-- Main theorem to prove
theorem focus_chord_property (k : ℝ) (hk : k ≠ 0)
  (P Q : ℝ × ℝ) (hP : parabola P.1 P.2) (hQ : parabola Q.1 Q.2)
  (hP_line : line P.1 P.2 k) (hQ_line : line Q.1 Q.2 k) :
  distance_reciprocal P Q = 1 / 2 :=
sorry -- proof to be filled in

end focus_chord_property_l87_87422


namespace no_real_roots_of_ffx_eq_ninex_l87_87874

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ :=
  x^2 * Real.log (4*(a+1)/a) / Real.log 2 +
  2 * x * Real.log (2 * a / (a + 1)) / Real.log 2 +
  Real.log ((a + 1)^2 / (4 * a^2)) / Real.log 2

theorem no_real_roots_of_ffx_eq_ninex (a : ℝ) (h_pos : ∀ x, 1 ≤ x → f a x > 0) :
  ¬ ∃ x, 1 ≤ x ∧ f a (f a x) = 9 * x :=
  sorry

end no_real_roots_of_ffx_eq_ninex_l87_87874


namespace total_amount_shared_l87_87149

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared_l87_87149


namespace circumcenter_on_line_AB_l87_87011

variables {A B C H E F M : Type} [AffineSpace ℝ A] [Triangle ℝ A B C]
variables (hAcute : Triangle.acute A B C) (hABgtAC : length AB > length AC)
variables (H : Point) (hH : Orthocenter H A B C)
variables (E : Point) (hE : ReflectOverAltitude C H A E)
variables (F : Point) (hF : LineIntersect EH AC F)
variables (M : Point) (hM : Circumcenter A E F M)

theorem circumcenter_on_line_AB : LiesOn M (LineThrough A B) :=
sorry

end circumcenter_on_line_AB_l87_87011


namespace baseball_card_total_decrease_l87_87249

theorem baseball_card_total_decrease (v₀ : ℝ) :
  ((v₀ - (v₀ * 0.4)) - ((v₀ - (v₀ * 0.4)) * 0.1)) = v₀ * 0.54 → 
  let decrease_percent := (v₀ - ((v₀ - (v₀ * 0.4)) - ((v₀ - (v₀ * 0.4)) * 0.1))) / v₀ * 100 in
  decrease_percent = 46 :=
by
  sorry

end baseball_card_total_decrease_l87_87249


namespace find_f_neg_five_halves_l87_87035

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x ^ (1/2) else f (x + 2)

theorem find_f_neg_five_halves : f (-5 / 2) = (real.sqrt 6) / 2 :=
  sorry

end find_f_neg_five_halves_l87_87035


namespace sum_of_cubes_mod_n_l87_87865

noncomputable def m (n : ℕ) : ℕ :=
  ∑ k in Finset.range n.filter (λ k, Nat.gcd k n = 1), k^3

theorem sum_of_cubes_mod_n (n : ℕ) (h : 0 < n) :
  (m n) % n = 0 :=
sorry

end sum_of_cubes_mod_n_l87_87865


namespace hyperbola_eccentricity_l87_87417

theorem hyperbola_eccentricity {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (h1 : |(λ x, - b / a * x)| = b)
  (h2 : |λ x, a| = 2 * sqrt 2 * |λ x, a|) : 
  let c := sqrt (a^2 + b^2) in
  let e := c / a in 
  e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l87_87417


namespace trig_identity_l87_87731

variable (α : ℝ)
variable (h : Real.sin α = 3 / 5)

theorem trig_identity : Real.sin (Real.pi / 2 + 2 * α) = 7 / 25 :=
by
  sorry

end trig_identity_l87_87731


namespace allocation_schemes_l87_87336

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l87_87336


namespace sum_of_k_l87_87319

theorem sum_of_k (k: ℤ) : 
  (let p := λ x : ℤ, x^2 - 4 * x + 3 in
   let q := λ x : ℤ, x^2 - 6 * x + k in
   let roots_p := {1, 3} in
   let values_of_k := { (λ r : ℤ, -r * (r - 6) ): ℤ → ℤ | r ∈ roots_p } in
   (∑ x in values_of_k, x) = 14) := 
sorry

end sum_of_k_l87_87319


namespace simple_interest_sum_l87_87972

theorem simple_interest_sum :
  let P := 1750
  let CI := 4000 * ((1 + (10 / 100))^2) - 4000
  let SI := (1 / 2) * CI
  SI = (P * 8 * 3) / 100 
  :=
by
  -- Definitions
  let P := 1750
  let CI := 4000 * ((1 + 10 / 100)^2) - 4000
  let SI := (1 / 2) * CI
  
  -- Claim
  have : SI = (P * 8 * 3) / 100 := sorry

  exact this

end simple_interest_sum_l87_87972


namespace minimum_ticket_cost_l87_87997

theorem minimum_ticket_cost :
  let adult_ticket := 40
  let child_ticket := 20
  let unlimited_day_pass_one := 350
  let unlimited_day_pass_group := 1500
  let unlimited_3day_pass_one := 900
  let unlimited_3day_pass_group := 3500
  (min_cost : ℕ) 
  in min_cost = 5200 :=
by
  let min_cost := 5200
  sorry

end minimum_ticket_cost_l87_87997


namespace triangle_angle_area_l87_87810

theorem triangle_angle_area
  (A B C : ℝ) (a b c : ℝ)
  (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
  (h2 : C = Real.pi / 3)
  (h3 : c = 2)
  (h4 : a + b + c = 2 * Real.sqrt 3 + 2) :
  ∃ (area : ℝ), area = (2 * Real.sqrt 3) / 3 :=
by 
  -- Proof is omitted
  sorry

end triangle_angle_area_l87_87810


namespace value_of_a_if_perpendicular_l87_87806

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.sin x - 2 * Real.cos x

def point_of_tangency := (Real.pi / 2, 2)

def line (a : ℝ) (x y : ℝ) := x - a * y + 1 = 0

def derivative_at_point : ℝ := 2 * Real.cos (Real.pi / 2) + 2 * Real.sin (Real.pi / 2)

theorem value_of_a_if_perpendicular :
  ∀ (a : ℝ),
    derivative_at_point = -a → 
    a = -2 :=
by
  sorry

end value_of_a_if_perpendicular_l87_87806


namespace area_of_triangle_ABC_proof_l87_87850

noncomputable def area_of_triangle_ABC (I O: Type*)
  [MetricSpace I] [MetricSpace O] 
  (A B C : I) (AB AC : ℝ) (angle_AIO : ℝ) : ℝ :=
  let area := Real.sqrt (4.75 * (4.75 - 2) * (4.75 - 3) * (4.75 - 5 / 2))
  in area

theorem area_of_triangle_ABC_proof (I O: Type*)
  [MetricSpace I] [MetricSpace O] 
  (A B C : I) (AB AC : ℝ) (angle_AIO : ℝ) 
  (hAB : AB = 2) (hAC : AC = 3) (h_angle_AIO : angle_AIO = Real.pi / 2) :
  area_of_triangle_ABC I O A B C AB AC angle_AIO = 15 * Real.sqrt 7 / 16 :=
by
  sorry

end area_of_triangle_ABC_proof_l87_87850


namespace largest_fraction_l87_87961

theorem largest_fraction :
  (∀ (a b : ℚ), a = 2 / 5 → b = 1 / 3 → a < b) ∧  
  (∀ (a c : ℚ), a = 2 / 5 → c = 7 / 15 → a < c) ∧ 
  (∀ (a d : ℚ), a = 2 / 5 → d = 5 / 12 → a < d) ∧ 
  (∀ (a e : ℚ), a = 2 / 5 → e = 3 / 8 → a < e) ∧ 
  (∀ (b c : ℚ), b = 1 / 3 → c = 7 / 15 → b < c) ∧
  (∀ (b d : ℚ), b = 1 / 3 → d = 5 / 12 → b < d) ∧ 
  (∀ (b e : ℚ), b = 1 / 3 → e = 3 / 8 → b < e) ∧ 
  (∀ (c d : ℚ), c = 7 / 15 → d = 5 / 12 → c > d) ∧
  (∀ (c e : ℚ), c = 7 / 15 → e = 3 / 8 → c > e) ∧
  (∀ (d e : ℚ), d = 5 / 12 → e = 3 / 8 → d > e) :=
sorry

end largest_fraction_l87_87961


namespace calculate_value_l87_87443

-- Define the operations
def add (a b : ℕ) : ℕ := a + b
def sub (a b : ℕ) : ℕ := a - b
def mul (a b : ℕ) : ℕ := a * b

-- Define the sequence and order of operations
def expr1 := 5
def expr2 := 4
def expr3 := 6
def expr4 := 3

-- The proof statement
theorem calculate_value : 
  (∃ op1 op2 op3 : (ℕ → ℕ → ℕ), 
    {op1, op2, op3} = {add, sub, mul} ∧ 
    ((op1 (op2 (op3 expr1 expr2) expr3) expr4) = 19 
    ∨ (op1 (op2 expr1 (op3 expr2 expr3)) expr4) = 19 
    ∨ (op1 (op2 expr1 expr2) (op3 expr3 expr4)) = 19 
    ∨ (op1 (op2 expr1 expr2) expr3 op3 expr4) = 19)) := 
sorry

end calculate_value_l87_87443


namespace product_units_tens_not_divisible_by_8_l87_87518

theorem product_units_tens_not_divisible_by_8 :
  ¬ (1834 % 8 = 0) → (4 * 3 = 12) :=
by
  intro h
  exact (by norm_num : 4 * 3 = 12)

end product_units_tens_not_divisible_by_8_l87_87518


namespace crossing_time_is_24_seconds_l87_87582

-- Define the given constants
def speed_faster_train : ℝ := 210
def speed_slower_train : ℝ := 90
def length_faster_train : ℝ := 1.10
def length_slower_train : ℝ := 0.9

-- Define the total length and relative speed
def combined_length : ℝ := length_faster_train + length_slower_train
def relative_speed_kmph : ℝ := speed_faster_train + speed_slower_train
def relative_speed_kmpmin : ℝ := relative_speed_kmph / 60

-- Prove that the slower train takes 24 seconds to cross the faster train
theorem crossing_time_is_24_seconds :
  combined_length / relative_speed_kmpmin * 60 = 24 :=
by
  sorry

end crossing_time_is_24_seconds_l87_87582


namespace red_columns_in_k_rows_l87_87113

theorem red_columns_in_k_rows
  (n : ℕ) (k : ℕ)
  (h1 : n ≥ 2)
  (h2 : 1 < k)
  (h3 : k ≤ (n / 2) + 1)
  (h4 : ∃ (array : Array (Array Bool)), array.size = n ∧ (∀ arr, arr ∈ array → arr.size = 2 * n ∧ (array.foldl (λ acc row, acc + row.count id) 0 = n * n))) :
  ∃ (rows : List (Array Bool)),
    rows.length = k ∧
    (∃ (cols : Set ℕ), cols.size ≥ (nat.factorial k * (n - 2 * k + 2)) / ((List.range (n - k + 1)).product) ∧
    ∀ row in rows, ∀ col in cols, (row.get! col = true)) :=
by
  sorry

end red_columns_in_k_rows_l87_87113


namespace player_b_wins_l87_87577

theorem player_b_wins :
  let numbers := Finset.range 28 \ Finset.singleton 0, -- Sequence 1 to 27
      total_sum := Finset.sum numbers (λ x, x),
      remainder := total_sum % 5 in
  (∀ x ∈ numbers, ((total_sum - x) % 5 ≠ 0) → ∃ y ∈ numbers, y ≡ 3 [MOD 5]) ->
  ∃ strategy : ℕ → ℕ,
    (∀ n, n < 27 → strategy n ∈ numbers ∧ (numbers.erase (strategy n))).card = numbers.card - 1 → 
    ((∃ m : ℕ, m = total_sum % 5) → (m ≠ 3)) :=
begin
  sorry
end

end player_b_wins_l87_87577


namespace directrix_of_parabola_l87_87330

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l87_87330


namespace max_value_2ac_minus_abc_l87_87151

theorem max_value_2ac_minus_abc (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 7) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c <= 4) : 
  2 * a * c - a * b * c ≤ 28 :=
sorry

end max_value_2ac_minus_abc_l87_87151


namespace find_cost_price_l87_87254

theorem find_cost_price (selling_price profit_percent : ℝ) 
  (h1 : selling_price = 500)
  (h2 : profit_percent = 0.25) :
  ∃ (cost_price : ℝ), selling_price = cost_price * (1 + profit_percent) ∧ cost_price = 400 :=
by
  existsi 400
  split
  simp [h1, h2]
  sorry

end find_cost_price_l87_87254


namespace find_n_given_conditions_l87_87371

def modulo_60 := 60

def a : ℤ := 27
def b : ℤ := 94
def n : ℤ := 173

theorem find_n_given_conditions :
    (∃ n, 150 ≤ n ∧ n ≤ 211 ∧ a - b ≡ n [MOD modulo_60]) ↔ n = 173 := by
  sorry

end find_n_given_conditions_l87_87371


namespace binary_11111111_to_decimal_l87_87308

-- Define the binary to decimal conversion principle
def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.enum.sum (λ ⟨i, n⟩, n * (2 ^ i))

-- Binary number (11111111)_2 represented as a list of bits
def binary_11111111 : list ℕ := [1, 1, 1, 1, 1, 1, 1, 1]

-- Expected decimal value for the binary number (11111111)_2
def expected_decimal_value : ℕ := 2^8 - 1

-- Theorem to prove that converting the binary number (11111111)_2 results in 2^8 - 1
theorem binary_11111111_to_decimal :
  binary_to_decimal binary_11111111 = expected_decimal_value := by
  sorry

end binary_11111111_to_decimal_l87_87308


namespace num_factors_n_l87_87126

def n : ℕ := 2^3 * 5^6 * 8^9 * 10^10

theorem num_factors_n : (finset.range (40 + 1)).card * (finset.range (16 + 1)).card = 697 :=
sorry

end num_factors_n_l87_87126


namespace range_of_a_l87_87770

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (x < 1 → f a x = (1 - 2 * a) * x + 3 * a)
                ∧ (x ≥ 1 → f a x = Real.log x))
    →  -1 ≤ a ∧ a < 1/2 :=
by
  intros a h
  sorry

end range_of_a_l87_87770


namespace truncated_cone_sphere_radius_l87_87662

noncomputable def radius_of_sphere {r1 r2 : ℝ} (r1_pos : 0 < r1) (r2_pos : 0 < r2)
  (r1_eq_20 : r1 = 20) (r2_eq_5 : r2 = 5) : ℝ :=
  let h := (15 * Real.sqrt 2 / 2) in h

theorem truncated_cone_sphere_radius : 
  ∀ (r1 r2 : ℝ), (0 < r1) → (0 < r2) → (r1 = 20) → (r2 = 5) →
  radius_of_sphere r1_pos r2_pos r1_eq_20 r2_eq_5 = 15 * Real.sqrt 2 / 2 :=
  sorry

end truncated_cone_sphere_radius_l87_87662


namespace parallelogram_x_value_l87_87701

noncomputable def x_value (x : ℝ) : Prop :=
  let A := (0, 0)
  let B := (2, 4)
  let C := (x + 2, 4)
  let D := (x, 0)
  let base := (2 : ℝ)
  let height := (4 : ℝ)
  let area := base * height
  area * x = 36

theorem parallelogram_x_value : ∃ x : ℝ, x_value x ∧ x = 4.5 :=
by
  use 4.5
  unfold x_value
  simp
  exact rfl

end parallelogram_x_value_l87_87701


namespace find_point_coordinates_l87_87829

open Real

-- Define circles C1 and C2
def circle_C1 (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define mutually perpendicular lines passing through point P
def line_l1 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)
def line_l2 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = -1/k * (x - P.1)

-- Define the condition that chord lengths intercepted by lines on respective circles are equal
def equal_chord_lengths (P : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (-4 * k - 2 + P.2 - k * P.1) / sqrt ((k^2) + 1) = abs (5 + 6 * k - k * P.2 - P.1) / sqrt ((k^2) + 1)

-- Main statement to be proved
theorem find_point_coordinates :
  ∃ (P : ℝ × ℝ), 
  circle_C1 (P.1) (P.2) ∧
  circle_C2 (P.1) (P.2) ∧
  (∀ k : ℝ, k ≠ 0 → equal_chord_lengths P k) ∧
  (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) :=
sorry

end find_point_coordinates_l87_87829


namespace quotient_zero_l87_87181

theorem quotient_zero (D d R Q : ℕ) (hD : D = 12) (hd : d = 17) (hR : R = 8) (h : D = d * Q + R) : Q = 0 :=
by
  sorry

end quotient_zero_l87_87181


namespace minimum_area_of_quadrilateral_l87_87823

theorem minimum_area_of_quadrilateral
  (ABCD : Type)
  (O : Type)
  (S_ABO : ℝ)
  (S_CDO : ℝ)
  (BC : ℝ)
  (cos_angle_ADC : ℝ)
  (h1 : S_ABO = 3 / 2)
  (h2 : S_CDO = 3 / 2)
  (h3 : BC = 3 * Real.sqrt 2)
  (h4 : cos_angle_ADC = 3 / Real.sqrt 10) :
  ∃ S_ABCD : ℝ, S_ABCD = 6 :=
sorry

end minimum_area_of_quadrilateral_l87_87823


namespace limit_seq_converges_to_one_l87_87678

noncomputable def limit_seq (n : ℕ) : ℝ := 
  (real.sqrt (n^4 + 2) + real.sqrt (n - 2)) / (real.sqrt (n^4 + 2) + real.sqrt (n - 2))

theorem limit_seq_converges_to_one : 
  tendsto (λ n : ℕ, limit_seq n) at_top (𝓝 (1 : ℝ)) :=
begin
  sorry
end

end limit_seq_converges_to_one_l87_87678


namespace sufficient_cond_not_necessary_cond_l87_87980

variable (p q : Prop)

theorem sufficient_cond (h₁: p ∧ q): p ∨ q :=
by
  sorry

theorem not_necessary_cond (h₂: p ∨ q) (h₃: ¬(p ∧ q)): (p ∧ q, p ∨ q) :=
by
  sorry

end sufficient_cond_not_necessary_cond_l87_87980


namespace arithmetic_sequence_first_term_l87_87200

theorem arithmetic_sequence_first_term (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 500) 
  (h2 : 30 * (2 * a + 179 * d) = 2900) : 
  a = -34 / 3 := 
sorry

end arithmetic_sequence_first_term_l87_87200


namespace correctness_of_propositions_l87_87264

-- Define the problem conditions
def complex_number_quad (z: Complex) : Prop := (-1, 2) = (z.re, z.im) ∧ z.im > 0 ∧ z.re < 0

namespace ProofProblem

-- Proving the correct propositions are numbered 1 and 3
theorem correctness_of_propositions :
  (complex_number_quad ((2 + Complex.i) * Complex.i)) ∧
  (∀ P Q : Prop, (P ∧ ¬Q) ↔ (P -> Q) ∧ (¬P ∨ Q) ↔ (Q -> P)) :=
by { sorry }

end ProofProblem

end correctness_of_propositions_l87_87264


namespace chord_length_of_circle_l87_87737

noncomputable def length_of_chord (a : ℝ) : ℝ := 2

theorem chord_length_of_circle {a : ℝ} (h : circle.PQ (0, 1) (a, (1/2) * a^2) = 2) :
  length_of_chord a = 2 :=
begin
  sorry
end

end chord_length_of_circle_l87_87737


namespace count_three_digit_numbers_with_digit_sum_27_l87_87051

-- Defining the digits and their constraints
def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Define a three-digit number and the condition on the digits
def is_three_digit_number (a b c : ℕ) : Prop := (1 ≤ a ∧ a ≤ 9) ∧ is_valid_digit b ∧ is_valid_digit c

-- The main theorem: number of three-digit numbers with digit sum 27
theorem count_three_digit_numbers_with_digit_sum_27 :
  (finset.card {n : ℕ | ∃ a b c, n = 100 * a + 10 * b + c ∧ is_three_digit_number a b c ∧ a + b + c = 27}) = 1 :=
sorry

end count_three_digit_numbers_with_digit_sum_27_l87_87051


namespace num_common_divisors_of_9240_and_10010_l87_87436

def prime_factors_9240 := {2^3, 3, 5, 7, 11}
def prime_factors_10010 := {2, 3, 5, 7, 11, 13}

theorem num_common_divisors_of_9240_and_10010 : 
  let gcd_9240_10010 := 2310 in
  (∏ p in (finset.filter prime (finset.range 14)), nat.divisors p).card = 32 :=
by
  sorry

end num_common_divisors_of_9240_and_10010_l87_87436


namespace parabola_directrix_eq_neg_2_l87_87328

noncomputable def parabola_directrix (a b c : ℝ) : ℝ :=
  (b^2 - 4 * a * c) / (4 * a)

theorem parabola_directrix_eq_neg_2 (x : ℝ) :
  parabola_directrix 1 (-4) 4 = -2 :=
by
  -- proof steps go here
  sorry

end parabola_directrix_eq_neg_2_l87_87328


namespace part1_a1_a2_part2_relationship_part3_product_l87_87130

variable (a_n S_n : ℕ → ℚ)

axiom cond_Sn : ∀ n, S_n^2 - 2 * S_n - a_n n * S_n + 1 = 0

theorem part1_a1_a2 :
  a_n 1 = 1 / 2 ∧ a_n 2 = 1 / 6 :=
sorry

theorem part2_relationship :
  (∀ n ≥ 2, S_n n = 1 / (2 - S_n (n - 1))) ∧ 
  (∀ n ≥ 1, 1 / (S_n n - 1) = if n = 1 then -2 else (1 / (S_n (n - 1) - 1) - 1)) :=
sorry

theorem part3_product :
  ∏ k in Finset.range 2011, S_n (k + 1) = 1 / 2012 :=
sorry

end part1_a1_a2_part2_relationship_part3_product_l87_87130


namespace TK_min_max_l87_87882

-- Variables representing points in 3D space
variables {A A1 B C D B1 C1 D1 K T : Type*}
-- Definitions of the distances given
variables [metric_space A] [metric_space A1] [metric_space B] 
          [metric_space C] [metric_space T] [metric_space K]

-- Conditions given in the problem
def edge_length (x y : Type*) [metric_space x] [metric_space y] : ℝ := 3 * real.sqrt 2
def TB_distance (T B : Type*) [metric_space T] [metric_space B] : ℝ := 7
def TC_distance (T C : Type*) [metric_space T] [metric_space C] : ℝ := real.sqrt 67

-- The proof problem statement
theorem TK_min_max (T B C K : Type*) [metric_space T] [metric_space B] 
                      [metric_space C] [metric_space K] :
  TK_min_max_cond T B C K → 1 ≤ dist T K ∧ dist T K ≤ 13 :=
sorry

-- Specifying the conditions as a conjunction
def TK_min_max_cond (T B C K : Type*) [metric_space T] [metric_space B] 
                      [metric_space C] [metric_space K] : Prop :=
  edge_length A A1 = 3 * real.sqrt 2 ∧
  dist T B = 7 ∧
  dist T C = real.sqrt 67

end TK_min_max_l87_87882


namespace smallest_symmetric_set_l87_87655

noncomputable def is_symmetric_origin (T : set (ℝ × ℝ)) : Prop :=
∀ (a b : ℝ), (a, b) ∈ T → (-a, -b) ∈ T

noncomputable def is_symmetric_x_axis (T : set (ℝ × ℝ)) : Prop :=
∀ (a b : ℝ), (a, b) ∈ T → (a, -b) ∈ T

noncomputable def is_symmetric_y_axis (T : set (ℝ × ℝ)) : Prop :=
∀ (a b : ℝ), (a, b) ∈ T → (-a, b) ∈ T

noncomputable def is_symmetric_line_y_eq_x (T : set (ℝ × ℝ)) : Prop :=
∀ (a b : ℝ), (a, b) ∈ T → (b, a) ∈ T

noncomputable def is_symmetric_line_y_eq_neg_x (T : set (ℝ × ℝ)) : Prop :=
∀ (a b : ℝ), (a, b) ∈ T → (-b, -a) ∈ T

noncomputable def contains_point_3_4 (T : set (ℝ × ℝ)) : Prop :=
(3, 4) ∈ T

noncomputable def smallest_points_in_set : ℝ :=
8

theorem smallest_symmetric_set
  (T : set (ℝ × ℝ)) :
  is_symmetric_origin T ∧ 
  is_symmetric_x_axis T ∧ 
  is_symmetric_y_axis T ∧ 
  is_symmetric_line_y_eq_x T ∧ 
  is_symmetric_line_y_eq_neg_x T ∧
  contains_point_3_4 T →
  ∃ (n : ℕ), n = smallest_points_in_set ∧ ∀ (t : set (ℝ × ℝ)), (is_symmetric_origin t ∧ 
    is_symmetric_x_axis t ∧ 
    is_symmetric_y_axis t ∧ 
    is_symmetric_line_y_eq_x t ∧ 
    is_symmetric_line_y_eq_neg_x t ∧
    contains_point_3_4 t) → (fintype.card t ≤ n) :=
by sorry

end smallest_symmetric_set_l87_87655


namespace ellipse_and_tangents_l87_87012

theorem ellipse_and_tangents :
  (∃ a b c : ℝ, a > b ∧ b > 0 ∧ b = sqrt 3 ∧ c = sqrt 3 ∧ a ^ 2 = b ^ 2 + c ^ 2 ∧
    (∀ x y : ℝ, (x ^ 2 / 6) + (y ^ 2 / 3) = 1) ∧ 
    (∃ (x0 y0 : ℝ), (x0 ^ 2 / 6) + (y0 ^ 2 / 3) = 1 ∧ 
      ∀ k1 k2 : ℝ, (x0 ^ 2 = 2 * ((k1 * x0 - y0) / sqrt (k1 ^ 2 + 1) = sqrt 2)) ∧
        k1 * k2 = (3 * (1 - x0 ^ 2 / 6) - 2) / (x0 ^ 2 - 2) ∧ 
        k1 * k2 = - 1 / 2)) := sorry

end ellipse_and_tangents_l87_87012


namespace monotonic_increase_intervals_l87_87700

theorem monotonic_increase_intervals (k : ℤ) :
  let I := set.Icc (real.pi / 2 + 2 * k * real.pi) (3 * real.pi / 2 + 2 * k * real.pi) in
  ∀ {f : ℝ → ℝ}, (f = λ x, 3 - 2 * real.sin x) →
  ∀ (a b : ℝ), a ∈ I → b ∈ I → a ≤ b → f a ≤ f b :=
by
  sorry

end monotonic_increase_intervals_l87_87700


namespace eccentricity_of_hyperbola_l87_87387

noncomputable def hyperbolaEccentricity (a b : ℝ) (P : ℝ × ℝ) (c : ℝ) (F1 F2 : ℝ × ℝ) : ℝ :=
  let e := c / a
  -- define the semi-focal distance
  (e : ℝ)

theorem eccentricity_of_hyperbola (a b c : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∃ (x : ℝ), P = (x, 0) ∧ x = (5 / 4) * c)
  (h2 : |dist P F2| = |dist F1 F2| )
  (h3 : F1 = (-c, 0) ∧ F2 = (c, 0))
  (h4 : a > 0 ∧ b > 0 ∧ c = sqrt (a^2 + b^2) ): 
  hyperbolaEccentricity a b P c F1 F2 = 2 :=
by
  sorry

end eccentricity_of_hyperbola_l87_87387


namespace prime_triplet_unique_l87_87709

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def prime_triple := (ℕ × ℕ × ℕ)

theorem prime_triplet_unique : ∀ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  is_prime (abs (p - q)) ∧ is_prime (abs (q - r)) ∧ is_prime (abs (r - p)) ↔ (p, q, r) = (2, 5, 7) :=
by
  sorry

end prime_triplet_unique_l87_87709


namespace find_angle_x_l87_87989

-- Given Conditions
variables (S P T A : Type) 
variables (beam : S → P → T)
variables (reflects_off : beam S P T → Prop)
variables (perpendicular : T → P → S → Prop)
variables (SPA_angle sixty_four_twice_x : ℝ)
axiom SPA_perpendicular (PT_perpendicular_RS : perpendicular T P S) : SPA_angle = 64
axiom SPA_double_x (SPA_is_two_x: SPA_angle = 2 * sixty_four_twice_x) : sixty_four_twice_x = 32

-- Theorem to prove
theorem find_angle_x (PT_pep: perpendicular T P S) (SPA_sixty_four: SPA_angle = 64)
  (SPA_2x: SPA_angle = 2 * sixty_four_twice_x): sixty_four_twice_x = 32 :=
SPA_sixty_four.trans (eq.symm SPA_2x).symm.symm.symm.symm sorry

end find_angle_x_l87_87989


namespace square_value_l87_87062

theorem square_value {square : ℚ} (h : 8 / 12 = square / 3) : square = 2 :=
sorry

end square_value_l87_87062


namespace longest_side_similar_triangle_l87_87915

noncomputable def internal_angle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem longest_side_similar_triangle (a b c A : ℝ) (h₁ : a = 4) (h₂ : b = 6) (h₃ : c = 7) (h₄ : A = 132) :
  let k := Real.sqrt (132 / internal_angle 4 6 7)
  7 * k = 73.5 :=
by
  sorry

end longest_side_similar_triangle_l87_87915


namespace polynomial_even_iff_exists_Q_l87_87158

open Polynomial

noncomputable def exists_polynomial_Q (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P.eval z = (Q.eval z) * (Q.eval (-z))

theorem polynomial_even_iff_exists_Q (P : Polynomial ℂ) :
  (∀ z : ℂ, P.eval z = P.eval (-z)) ↔ exists_polynomial_Q P :=
by 
  sorry

end polynomial_even_iff_exists_Q_l87_87158


namespace min_polyline_distance_l87_87083

-- Define the polyline distance between two points P(x1, y1) and Q(x2, y2).
noncomputable def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Define the circle x^2 + y^2 = 1.
def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

-- Define the line 2x + y = 2√5.
def on_line (P : ℝ × ℝ) : Prop :=
  2 * P.1 + P.2 = 2 * Real.sqrt 5

-- Statement of the minimum distance problem.
theorem min_polyline_distance : 
  ∀ P Q : ℝ × ℝ, on_circle P → on_line Q → 
  polyline_distance P Q ≥ Real.sqrt 5 / 2 :=
sorry

end min_polyline_distance_l87_87083


namespace loaf_bread_cost_correct_l87_87248

-- Given conditions
def total : ℕ := 32
def candy_bar : ℕ := 2
def final_remaining : ℕ := 18

-- Intermediate calculations as definitions
def remaining_after_candy_bar : ℕ := total - candy_bar
def turkey_cost : ℕ := remaining_after_candy_bar / 3
def remaining_after_turkey : ℕ := remaining_after_candy_bar - turkey_cost
def loaf_bread_cost : ℕ := remaining_after_turkey - final_remaining

-- Theorem stating the problem question and expected answer
theorem loaf_bread_cost_correct : loaf_bread_cost = 2 :=
sorry

end loaf_bread_cost_correct_l87_87248


namespace crease_length_l87_87651

theorem crease_length (w θ : ℝ) (h40 : θ = 45) (hw : w = 8) :
  (let L := 4 * Real.sqrt 2 in L) = 4 * Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end crease_length_l87_87651


namespace grasshopper_position_after_100_jumps_l87_87295

theorem grasshopper_position_after_100_jumps :
  let start_pos := 1
  let jumps (n : ℕ) := n
  let total_positions := 6
  let total_distance := (100 * (100 + 1)) / 2
  (start_pos + (total_distance % total_positions)) % total_positions = 5 :=
by
  sorry

end grasshopper_position_after_100_jumps_l87_87295


namespace min_value_sin_cos_expression_l87_87718

theorem min_value_sin_cos_expression : ∀ x : ℝ, 
  ∃ y : ℝ, y = (9 / 10) ∧ (y = infi (fun x => (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1)) :=
begin
  sorry
end

end min_value_sin_cos_expression_l87_87718


namespace affix_inverse_l87_87486

-- Given a point M with affix z (where z is a non-zero complex number that is not real)
variables (z : ℂ) (hz_ne_zero : z ≠ 0) (hz_not_real : z.im ≠ 0)

-- Define the points M' with affix z' = 1/z
noncomputable def z' : ℂ := 1 / z

-- We need to prove the magnitude and argument of z' given the conditions
theorem affix_inverse (z : ℂ) (hz_ne_zero : z ≠ 0) (hz_not_real : z.im ≠ 0) :
  (complex.abs (1 / z) = 1 / complex.abs z) ∧ (complex.arg (1 / z) = -complex.arg z) :=
by
  sorry

end affix_inverse_l87_87486


namespace no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l87_87252

theorem no_solution_for_x_y_z (a : ℕ) : 
  ¬ ∃ (x y z : ℚ), x^2 + y^2 + z^2 = 8 * a + 7 :=
by
  sorry

theorem seven_n_plus_eight_is_perfect_square (n : ℕ) :
  ∃ x : ℕ, 7^n + 8 = x^2 ↔ n = 0 :=
by
  sorry

end no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l87_87252


namespace length_of_goods_train_is_correct_l87_87639

-- Define the conditions
def speed_man_train_kmh : ℝ := 50
def speed_goods_train_kmh : ℝ := 62
def time_to_pass_seconds : ℝ := 9

-- Conversion factor from km/h to m/s
def kmh_to_ms_factor : ℝ := 1000 / 3600

-- Define the relative speed in m/s
def relative_speed_ms : ℝ := (speed_man_train_kmh + speed_goods_train_kmh) * kmh_to_ms_factor

-- Define the expected length of the goods train
def expected_length_meters : ℝ := relative_speed_ms * time_to_pass_seconds

-- The main theorem statement
theorem length_of_goods_train_is_correct : expected_length_meters ≈ 280 := sorry

end length_of_goods_train_is_correct_l87_87639


namespace max_distance_correct_l87_87502

noncomputable def max_distance (z : ℂ) (hz : complex.norm z = 1) : ℝ :=
  let u := (3 * conj z + (1 + complex.I) * z : ℂ) in
  complex.abs u

theorem max_distance_correct (z : ℂ) (hz : complex.norm z = 1) : max_distance z hz = real.sqrt 17 :=
by
  sorry

end max_distance_correct_l87_87502


namespace number_of_ordered_pairs_is_three_l87_87649

-- Define the problem parameters and conditions
variables (a b : ℕ)
variable (b_gt_a : b > a)

-- Define the equation for the areas based on the problem conditions
def area_equation : Prop :=
  a * b = 3 * (a - 4) * (b - 4)

-- Main theorem statement
theorem number_of_ordered_pairs_is_three (h₁ : a > 0) (h₂ : b > 0) (h3: b_gt_a) (h4: area_equation a b) :
  ∃! (n : ℕ), n = 3 :=
begin
  sorry  -- Proof is omitted
end

end number_of_ordered_pairs_is_three_l87_87649


namespace smallest_positive_period_interval_of_monotonic_increase_l87_87410

noncomputable def smallest_period (f : ℝ → ℝ) : ℝ := sorry 
noncomputable def interval_increase (f : ℝ → ℝ) : Set (Set ℝ) := sorry

def f : ℝ → ℝ := λ x, 2 * Real.sin (π / 3 - x / 2)

theorem smallest_positive_period : smallest_period f = 4 * π := sorry

theorem interval_of_monotonic_increase : interval_increase f = {S | ∃ k : ℤ, S = Set.Icc (5 * π / 3 + 4 * π * k) (11 * π / 3 + 4 * π * k)} := sorry

end smallest_positive_period_interval_of_monotonic_increase_l87_87410


namespace geometric_ratio_l87_87669

-- Define the geometrical entities and conditions
variables (Γ : Type) [circle Γ]

variables (A B C P F E M N K D : Point Γ)

-- Condition: PA and PB are tangents to the circle Γ at A and B respectively.
variable [Tangent P A Γ]
variable [Tangent P B Γ]

-- Condition: C is a point on the minor arc AB
variable [OnMinorArc C A B Γ]

-- Condition: The tangent at C intersects PA and PB at F and E respectively
variable [IntersectsTangentAtC F E PA PB C Γ]

-- Condition: The circumcircle of triangle PFE intersects Γ at M and N.
variable [CircumcirclePFEIntersectsΓ M N P F E Γ Γ']

-- Condition: AC and MN intersect at K
variable [Intersect AC MN K]

-- Condition: The projection of B onto AC is D
variable [Projection D B AC]

-- Theorem to be proven
theorem geometric_ratio :
  ∀ (AC CK AD DK : Length) (AC_div_CK_eq_2AD_div_DK : AC / CK = 2 * AD / DK),
  AC_div_CK_eq_2AD_div_DK :=
begin
  sorry
end

end geometric_ratio_l87_87669


namespace samia_walk_distance_l87_87527

theorem samia_walk_distance :
  ∃ (x : ℝ), 
    (∀ (y : ℝ), y = 4 * x → 
      ((3 * x) / 15 + x / 4) = 56 / 60) ∧ 
    Real.floor (x * 10 + 0.5) / 10 = 2.1 :=
by 
  sorry

end samia_walk_distance_l87_87527


namespace isosceles_triangle_projections_equal_l87_87580

theorem isosceles_triangle_projections_equal
  (O : Point)
  (triangle1 triangle2 : Triangle)
  (h_iso1 : is_isosceles triangle1 O)
  (h_iso2 : is_isosceles triangle2 O)
  (h_sim : is_similar triangle1 triangle2)
  (M : Point)
  (N : Point)
  (hM : M = midpoint (base triangle1))
  (hN : N = midpoint (base triangle2))
  (k1 k2 : ℝ)
  (h_k1 : k1 = base_triangle1_length / height_triangle1_length)
  (h_k2 : k2 = base_triangle2_length / height_triangle2_length) :
  let proj1 := k1 * (dist O M) * sin (angle OMN)
      proj2 := k2 * (dist O N) * sin (angle ONM)
  in proj1 = proj2 :=
sorry

end isosceles_triangle_projections_equal_l87_87580


namespace monotonic_increasing_interval_l87_87922

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin x * cos x - 2 * cos x ^ 2 + 1

theorem monotonic_increasing_interval :
  ∀ k : ℤ, 
    (∀ x ∈ Ioo (k * π - π / 8) (k * π + 3 * π / 8), 
     f' x > 0) :=
by
  sorry

end monotonic_increasing_interval_l87_87922


namespace possible_values_of_n_l87_87975

theorem possible_values_of_n (S : Finset (Finset.Point ℝ ℝ)) (hS : 1 < S.card) :
  ∃ n, n = 3 ∨ n = 5 ∧ S.card = n ∧
    (∀ ⦃P Q : Finset.Point ℝ ℝ⦄, P ∈ S → Q ∈ S → ∃ R ∈ S, R ∈ (m P Q)) ∧
    (∀ ⦃P Q P' Q' P'' Q'' : Finset.Point ℝ ℝ⦄, P ≠ Q → (P', Q', P'', Q'' ∈ S) → 
      (∃ R ∈ S, R ∈ (m P Q) ∧ R ∈ (m P' Q') ∧ R ∈ (m P'' Q'')) → 
      false) :=
begin
  sorry -- Proof goes here
end

end possible_values_of_n_l87_87975


namespace maximum_area_triangle_l87_87841

theorem maximum_area_triangle 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : ∀ A B : ℝ, ∀ a b : ℝ, (\frac{cos A}{sin B} + \frac{cos B}{sin A}) = 2)
  (h₃ : a + b + c = 12)
  : let area := \frac{1}{2} * a * b
    ∃ max_area : ℝ, max_area = 36 * (3 - 2 * sqrt 2)
    := sorry

end maximum_area_triangle_l87_87841


namespace seq_a_seq_b_l87_87741

theorem seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (n + 1) = 3 * S n + 2) →
  (∀ n, a n = if n = 1 then 1 else 4 * 3 ^ (n - 2)) :=
by
  sorry

theorem seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (b n = 8 * n / (a (n + 1) - a n)) →
  (T n = 77 / 12 - (n / 2 + 3 / 4) * (1 / 3) ^ (n - 2)) :=
by
  sorry

end seq_a_seq_b_l87_87741


namespace perimeter_of_new_figure_l87_87170

theorem perimeter_of_new_figure (original_perimeter : ℕ) (add_tiles : ℕ) (new_perimeter : ℕ) :
  original_perimeter = 16 →
  add_tiles = 2 →
  new_perimeter = (5 + 4 + 5 + 4) →
  new_perimeter = 18 :=
by
  assume h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end perimeter_of_new_figure_l87_87170


namespace percentage_of_remaining_journeymen_l87_87671

def total_employees := 35105
def fraction_journeymen := 4 / 13
def fraction_apprentices := 7 / 23
def fraction_administrative_staff := 1 - (fraction_journeymen + fraction_apprentices)
def fraction_journeymen_laid_off := 5 / 11
def fraction_apprentices_laid_off := 7 / 15
def fraction_administrative_staff_laid_off := 3 / 7

theorem percentage_of_remaining_journeymen :
  let journeymen := fraction_journeymen * total_employees in
  let apprentices := fraction_apprentices * total_employees in
  let administrative_staff := total_employees - (journeymen + apprentices) in
  let laid_off_journeymen := fraction_journeymen_laid_off * journeymen in
  let laid_off_apprentices := fraction_apprentices_laid_off * apprentices in
  let laid_off_administrative_staff := fraction_administrative_staff_laid_off * administrative_staff in
  let remaining_journeymen := journeymen - laid_off_journeymen in
  let remaining_apprentices := apprentices - laid_off_apprentices in
  let remaining_administrative_staff := administrative_staff - laid_off_administrative_staff in
  let total_remaining_employees := remaining_journeymen + remaining_apprentices + remaining_administrative_staff in
  (remaining_journeymen / total_remaining_employees) * 100 = 30.42 :=
sorry

end percentage_of_remaining_journeymen_l87_87671


namespace parking_garage_stories_l87_87277

theorem parking_garage_stories :
  ∀ (spots_per_level open_first open_second_diff open_third_diff open_fourth full_spots : ℕ),
  spots_per_level = 100 →
  open_first = 58 →
  open_second_diff = 2 →
  open_third_diff = 5 →
  open_fourth = 31 →
  full_spots = 186 →
  let open_second := open_first + open_second_diff in
  let open_third := open_second + open_third_diff in
  let total_open := open_first + open_second + open_third + open_fourth in
  let total_slots := full_spots + total_open in
  total_slots / spots_per_level = 4 :=
by
  intros spots_per_level open_first open_second_diff open_third_diff open_fourth full_spots h1 h2 h3 h4 h5 h6,
  let open_second := open_first + open_second_diff in
  let open_third := open_second + open_third_diff in
  let total_open := open_first + open_second + open_third + open_fourth in
  let total_slots := full_spots + total_open in
  have h_total_slots : total_slots = 400, by
    rw [h1, h2, h3, h4, h5, h6],
    simp,
  exact nat.div_eq_of_eq_mul h_total_slots.symm rfl,
  sorry

end parking_garage_stories_l87_87277


namespace collinear_vectors_max_sin_value_l87_87044

noncomputable def vectorAB (θ : ℝ) : ℝ × ℝ := (1, 2 * Real.cos θ)
noncomputable def vectorBC (m : ℝ) : ℝ × ℝ := (m, -4)
noncomputable def vectorAC (θ : ℝ) (m : ℝ) : ℝ × ℝ := (1 + m, 2 * Real.cos θ - 4)

open Real

theorem collinear_vectors (θ : ℝ) (h1 : θ ∈ Ioo (-π / 2) (π / 2)) :
  ∀ m, (m = -4) →
  ∃ θ, θ = π / 3 ∨ θ = -π / 3 := by
  sorry

theorem max_sin_value (θ : ℝ) (h1 : θ ∈ Ioo (-π / 2) (π / 2)) :
  (∀ m ∈ Icc (-1) (0), (1 + m) * m + (2 * cos θ - 4) * (-4) ≤ 10) →
  ∃ θ, -sin (π / 2 - θ) = -3 / 4 := by
  sorry

end collinear_vectors_max_sin_value_l87_87044


namespace trajectory_companion_point_dot_product_range_area_triangle_l87_87401

-- Definitions from conditions
def ellipse (a b : ℝ) := { p : ℝ × ℝ // ∃ x y, p = (x, y) ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ a > 0 ∧ b > 0 }

-- Trajectory of companion point
theorem trajectory_companion_point (a b : ℝ) (h : ℝ > 0) (c : ellipse a b):
  let N (p : ℝ × ℝ) := (fst p / a, snd p / b) 
  in (∃ p, c p) → ∀ p, p ∈ c → (fst (N p))^2 + (snd (N p))^2 = 1 :=
by sorry

-- Range of dot product
theorem dot_product_range (a b : ℝ) (p : ℝ × ℝ) (h₀ : a = 2) (h₁ : b = sqrt 3) (h₂ : (p : ellipse a b)):
  let M := (1, 3/2)
  in p = (M) → 
  ∃ x₂ y₂, p = (x₂, y₂) ∧ (x₂^2 / a^2 + y₂^2 / b^2 = 1) →
  (sqrt 3 ≤ (fst M) * (fst (⟨fst M / a, snd M / b⟩)) + (snd M * (snd (⟨fst M / a, snd M / b⟩))) ∧
    (fst M) * (fst (⟨fst M / a, snd M / b⟩)) + (snd M * (snd (⟨fst M / a, snd M / b⟩))) ≤ 2) :=
by sorry

-- Area of Triangle OAB
theorem area_triangle (a b k m : ℝ) (h₀ : a = 2) (h₁ : b = sqrt 3) (h₂ : ∃ x y, (x, y) ∈ ellipse a b) (h₃ : (3 + 4 * k ^ 2 = 2 * m ^ 2)):
  ∃ (Δ: ℝ) (d : ℝ) AB, Δ = 48 * m ^ 2 ∧ d = ⟦m⟧ / sqrt (1 + k^2) 
  → |AB| = sqrt (1 + k^2) * (4 * sqrt 3 * |m|) / (3 + 4 * k ^ 2) 
  → area_ triangle = sqrt 3 :=
by sorry

end trajectory_companion_point_dot_product_range_area_triangle_l87_87401


namespace hyperbola_equation_l87_87418

theorem hyperbola_equation (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (e : ℝ) (h₂ : e = 5 / 4)
  (right_focus : ℝ × ℝ) (h₃ : right_focus = (5, 0))
  (h₄ : c = 5) (h₅ : a = 4) (h₆ : b = sqrt (c^2 - a^2)) :
  ∃ (eq : String), eq = "x^2/16 - y^2/9 = 1" :=
by
  sorry

end hyperbola_equation_l87_87418


namespace expected_value_coin_flip_l87_87293

-- Definitions based on conditions
def P_heads : ℚ := 2 / 3
def P_tails : ℚ := 1 / 3
def win_heads : ℚ := 4
def lose_tails : ℚ := -9

-- Expected value calculation
def expected_value : ℚ :=
  P_heads * win_heads + P_tails * lose_tails

-- Theorem statement to be proven
theorem expected_value_coin_flip : expected_value = -1 / 3 :=
by sorry

end expected_value_coin_flip_l87_87293


namespace surface_area_of_sphere_l87_87750

-- Define the points on the sphere using an abstract type Point
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Define three specific points based on the conditions
def P : Point := {x := 0, y := 0, z := 0}
def A : Point := {x := 1, y := 0, z := 0}
def B : Point := {x := 0, y := 1, z := 0}
def C : Point := {x := 0, y := 0, z := 1}

-- Define a function to calculate the distance between two points
def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Assert the distances based on the given conditions
example : distance P A = 1 := by
  sorry

example : distance P B = 1 := by
  sorry

example : distance P C = 1 := by
  sorry

-- State the theorem for surface area of the sphere given the conditions
theorem surface_area_of_sphere : 4 * Real.pi * ((Real.sqrt 3) / 2)^2 = 3 * Real.pi :=
by 
  sorry

end surface_area_of_sphere_l87_87750


namespace total_surface_area_l87_87901

-- Define the context and given conditions
variable (r : ℝ) (π : ℝ)
variable (base_area : ℝ) (curved_surface_area : ℝ)

-- Assume that the area of the base (circle) of the hemisphere is given as 225π
def base_of_hemisphere_area (r : ℝ) (π : ℝ) :=
  π * r^2 = 225 * π

-- Derive the radius from the base area
def radius_from_base_area (r : ℝ) :=
  r = Real.sqrt (225)

-- Define the curved surface area of the hemisphere
def curved_surface_area_hemisphere (r : ℝ) (π : ℝ) :=
  curved_surface_area = (1 / 2) * (4 * π * r^2)

-- Provide the final calculation for the total surface area
def total_surface_area_hemisphere (curved_surface_area : ℝ) (base_area : ℝ) :=
  curved_surface_area + base_area = 675 * π

-- Main theorem that combines everything and matches the problem statement
theorem total_surface_area (r : ℝ) (π : ℝ) (base_area : ℝ) (curved_surface_area : ℝ) :
  base_of_hemisphere_area r π →
  radius_from_base_area r →
  curved_surface_area_hemisphere r π →
  total_surface_area_hemisphere curved_surface_area base_area :=
by
  intros h_base_radius h_radius h_curved_area
  sorry

end total_surface_area_l87_87901


namespace circle_in_wide_polygon_l87_87992

theorem circle_in_wide_polygon (P : set (ℝ × ℝ)) (h_convex : convex P) (h_wide : ∀ l, ∃ a b, ∀ x ∈ P, (l (x.1, x.2)).fst ≥ a ∧ (l (x.1, x.2)).snd ≤ b ∧ b - a ≥ 1) :
  ∃ (c : ℝ × ℝ) (r : ℝ), r ≥ 1/3 ∧ ∀ x ∈ P, (x - c).2 < r := 
sorry

end circle_in_wide_polygon_l87_87992


namespace rectangle_bisectors_form_square_l87_87155

theorem rectangle_bisectors_form_square (a b : ℝ) :
  let A := (0 : ℝ, 0 : ℝ),
      B := (a, 0 : ℝ),
      C := (a, b),
      D := (0 : ℝ, b),
      bisector_A := λ (x : ℝ), x,
      bisector_B := λ (x : ℝ), -x + a,
      bisector_C := λ (x : ℝ), x + b - a,
      bisector_D := λ (x : ℝ), -x + b in
  -- Intersection points
  let I := (b / 2, b / 2),
      J := ((2*a - b) / 2, (2*a - b) / 2) in
  -- Verify that these points form a square
  I = (b / 2, b / 2) ∧ J = ((2*a - b) / 2, (2*a - b) / 2)
  -- and the distances exhibit properties of a square
  → sorry

end rectangle_bisectors_form_square_l87_87155


namespace isosceles_trapezoid_count_l87_87379

theorem isosceles_trapezoid_count (n : ℕ) : 
  let k_max := (n + 1) / 2 - 2 in
  ∑ i in finset.range (k_max + 1), n * i = n * ∑ i in finset.range (k_max + 1), i :=
by
  sorry

end isosceles_trapezoid_count_l87_87379


namespace proof_system_l87_87529

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  6 * x - 2 * y = 1 ∧ 2 * x + y = 2

-- Define the solution to the system of equations
def solution_equations (x y : ℝ) : Prop :=
  x = 0.5 ∧ y = 1

-- Define the system of inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1

-- Define the solution set for the system of inequalities
def solution_inequalities (x : ℝ) : Prop :=
  2 < x ∧ x < 5

-- The final theorem to be proved
theorem proof_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution_equations x y ∧ system_of_inequalities x ∧ solution_inequalities x :=
by
  sorry

end proof_system_l87_87529


namespace sum_first_eight_terms_l87_87745

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (n : ℕ) (a_3 a_6 : ℝ)

-- Conditions
def arithmetic_sequence (a_3 a_6 : ℝ) : Prop := a_3 = 20 - a_6

def sum_terms (S : ℕ → ℝ) (a_3 a_6 : ℝ) : ℝ :=
  4 * (a_3 + a_6)

-- The proof goal
theorem sum_first_eight_terms (a_3 a_6 : ℝ) (h₁ : a_3 = 20 - a_6) : S 8 = 80 :=
by
  rw [arithmetic_sequence a_3 a_6] at h₁
  sorry

end sum_first_eight_terms_l87_87745


namespace roots_have_different_signs_l87_87559

theorem roots_have_different_signs
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ∈ ℝ)
  (hc : c ∈ ℝ)
  (h_signs : (a * (1 / a : ℝ)^2 + b * (1 / a) + c) * (a * c^2 + b * c + c) < 0) :
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₁ * x₂ < 0) :=
sorry

end roots_have_different_signs_l87_87559


namespace monotonically_increasing_intervals_part_I_range_of_a_part_II_l87_87037

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a*x^2 + 10

-- Part (I)
theorem monotonically_increasing_intervals_part_I :
  (a = 1) → 
  (∀ x : ℝ, 
    (f' x 1 > 0) ↔ 
    (x < 0 ∨ x > (2/3))) :=
sorry

-- Part (II)
theorem range_of_a_part_II : 
  (∃ x ∈ (Set.Icc 1 2), f x a < 0) ↔ 
  a > 9/2 :=
sorry

end monotonically_increasing_intervals_part_I_range_of_a_part_II_l87_87037


namespace area_of_quadrilateral_ABB_l87_87779

theorem area_of_quadrilateral_ABB'A' (p θ : ℝ) (h : p > 0) (k : ℝ) (hk : k = Real.tan θ) :
  let F := (p / 2, p / 2),
      parabola := ∀ (x y : ℝ), y^2 = 2 * p * x,
      intersects : ∀ A B : ℝ × ℝ, parabola A.1 A.2 ∧ parabola B.1 B.2,
      O := (0, 0),
      directrix := ∀ x : ℝ, y = -p/2,
      A' := (O.1, -p^2 / A.2),
      B' := (O.1, -p^2 / B.2)
  in 
  S_AABB'A' = 2 * p ^ 2 * (1 + 1 / k ^ 2) ^ (3 / 2) :=
sorry

end area_of_quadrilateral_ABB_l87_87779


namespace number_is_4_l87_87267

theorem number_is_4 (x : ℕ) (h : x + 5 = 9) : x = 4 := 
by {
  sorry
}

end number_is_4_l87_87267


namespace books_difference_l87_87261

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h1 : bobby_books = 142) (h2 : kristi_books = 78) : bobby_books - kristi_books = 64 :=
by {
  -- Placeholder for the proof
  sorry
}

end books_difference_l87_87261


namespace centroid_of_equal_areas_l87_87105

theorem centroid_of_equal_areas (A B C O L M N : Point) (h_L : L ∈ segment A B) (h_M : M ∈ segment B C) (h_N : N ∈ segment C A)
  (h_OL_parallel : ∥ segment O L ∥ ∥ segment B C ∥)
  (h_OM_parallel : ∥ segment O M ∥ ∥ segment A C ∥)
  (h_ON_parallel : ∥ segment O N ∥ ∥ segment A B ∥)
  (h_area_eq : triangle_area B O L = triangle_area C O M ∧ triangle_area C O M = triangle_area A O N) :
  O = centroid A B C :=
sorry

end centroid_of_equal_areas_l87_87105


namespace similar_isosceles_triangle_projection_eq_l87_87579

variables {α : Type*} [linear_ordered_field α]

/-- Two similar isosceles triangles share a common vertex. Prove that the projections of their 
bases onto the line connecting the midpoints of the bases are equal. -/
theorem similar_isosceles_triangle_projection_eq
  {O M N : α}
  (a₁ a₂ h₁ h₂ : α)
  (k : α)
  (hb₁ : 0 < a₁)
  (hb₂ : 0 < a₂)
  (hh₁ : 0 < h₁)
  (hh₂ : 0 < h₂)
  (similar_triangles : a₁ / h₁ = a₂ / h₂)
  (k_value : k = a₁ / h₁) :
  let proj₁ := k * O * sin (M.angle_between N)
      proj₂ := k * O * sin (N.angle_between M) in
  proj₁ = proj₂ := 
by
  sorry

end similar_isosceles_triangle_projection_eq_l87_87579


namespace greatest_possible_integer_l87_87108

theorem greatest_possible_integer :
  ∃ (n : ℤ), n > 0 ∧ n < 100 ∧
             (∃ (k : ℤ), n = 9 * k - 2) ∧
             (∃ (l : ℤ), n = 6 * l - 4) ∧
             ∀ (m : ℤ), m > 0 ∧ m < 100 ∧
                         (∃ (k' : ℤ), m = 9 * k' - 2) ∧
                         (∃ (l' : ℤ), m = 6 * l' - 4) →
                         m ≤ n :=
begin
  use 86,
  split,
  { norm_num }, -- 86 > 0
  split,
  { norm_num }, -- 86 < 100
  split,
  { use 10, norm_num }, -- 86 = 9 * 10 - 2
  split,
  { use 15, norm_num }, -- 86 = 6 * 15 - 4
  intros m hm,
  rcases hm with ⟨hm1, hm2, ⟨k', hk'⟩, ⟨l', hl'⟩⟩,
  have h : 3 * k' - 2 * l' = -2,
  { 
    linarith [(hk' - hl').symm] 
  },
  -- Further elements should check if m gets above 86 assuming
  sorry
end

end greatest_possible_integer_l87_87108


namespace min_value_of_x_squared_plus_6x_l87_87235

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l87_87235


namespace parallelogram_sum_pa_l87_87276

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  2 * (distance p1 p2 + distance p2 p3)

def area (b h : ℝ) : ℝ := b * h

theorem parallelogram_sum_pa
  (A B C D : ℝ × ℝ)
  (h1 : A = (1, 2)) 
  (h2 : B = (5, 6)) 
  (h3 : C = (12, 6)) 
  (h4 : D = (8, 2))
  (base : ℝ)
  (height : ℝ)
  (h_base : base = distance A D)
  (h_height : height = (B.2 - A.2)) :
  (perimeter A B C D + area base height) = 42 + 8 * real.sqrt 2 := 
  sorry

end parallelogram_sum_pa_l87_87276


namespace time_to_pass_platform_l87_87968

-- Definitions based on the conditions
def train_length : ℕ := 1500 -- (meters)
def tree_crossing_time : ℕ := 120 -- (seconds)
def platform_length : ℕ := 500 -- (meters)

-- Define the train's speed
def train_speed := train_length / tree_crossing_time

-- Define the total distance the train needs to cover to pass the platform
def total_distance := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : 
  total_distance / train_speed = 160 :=
by sorry

end time_to_pass_platform_l87_87968


namespace sum_after_replacement_l87_87208

theorem sum_after_replacement (A B : ℕ) (h : A = 2 * B) :
  (A + B) % 3 = 0 ↔ (A + B) = (A / 2 + 2 * B) :=
by
  sorry

example : ¬ (2021 % 3 = 0) ∧ (2022 % 3 = 0) :=
by
  have h1 : 2021 % 3 = 2 := rfl
  have h2 : 2022 % 3 = 0 := rfl
  exact ⟨h1, h2⟩

end sum_after_replacement_l87_87208


namespace tangent_line_count_l87_87694

noncomputable def circles_tangent_lines (r1 r2 d : ℝ) : ℕ :=
if d = |r1 - r2| then 1 else 0 -- Define the function based on the problem statement

theorem tangent_line_count :
  circles_tangent_lines 4 5 3 = 1 := 
by
  -- Placeholder for the proof, which we are skipping as per instructions
  sorry

end tangent_line_count_l87_87694


namespace f_zero_f_positive_all_f_increasing_f_range_l87_87698

universe u

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_positive : ∀ x : ℝ, 0 < x → f x > 1
axiom f_add_prop : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(0) = 1
theorem f_zero : f 0 = 1 := sorry

-- Problem 2: Prove that for any x in ℝ, f(x) > 0
theorem f_positive_all (x : ℝ) : f x > 0 := sorry

-- Problem 3: Prove that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 4: Given f(x) * f(2x - x²) > 1, find the range of x
theorem f_range (x : ℝ) (h : f x * f (2*x - x^2) > 1) : 0 < x ∧ x < 3 := sorry

end f_zero_f_positive_all_f_increasing_f_range_l87_87698


namespace problem_l87_87389

noncomputable def polynomial : (x : ℤ) → ℤ :=
  λ x, (2 + x)^6 * (1 - 2*x)^5

theorem problem (a_i : Fin 12 → ℤ) :
  let P := polynomial;
  P = ∑ i in Finset.range 12, (a_i i) * (x ^ (i : ℕ)) →
  (a_i 0 + a_i 1 + a_i 2 + a_i 3 + a_i 4 + a_i 5 + a_i 6 + a_i 7 + a_i 8 + a_i 9 + a_i 10 + a_i 11 = -729) ∧
  (a_i 1 + a_i 3 + a_i 5 + a_i 7 + a_i 9 + a_i 11 = -486) ∧
  (∑ i in Finset.range 11, (a_i (i + 1)) / 2^(i + 1) = -64) :=
by
  sorry

end problem_l87_87389


namespace volume_between_concentric_spheres_l87_87211

theorem volume_between_concentric_spheres
  (r1 r2 : ℝ) (h_r1 : r1 = 5) (h_r2 : r2 = 10) :
  (4 / 3 * Real.pi * r2^3 - 4 / 3 * Real.pi * r1^3) = (3500 / 3) * Real.pi :=
by
  rw [h_r1, h_r2]
  sorry

end volume_between_concentric_spheres_l87_87211


namespace range_of_m_l87_87532

def P (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ^ 2 + m * x1 + 1 = 0) ∧ (x2 ^ 2 + m * x2 + 1 = 0) ∧ (x1 < 0) ∧ (x2 < 0)

def Q (m : ℝ) : Prop :=
  ∀ (x : ℝ), 4 * x ^ 2 + 4 * (m - 2) * x + 1 ≠ 0

def P_or_Q (m : ℝ) : Prop :=
  P m ∨ Q m

def P_and_Q (m : ℝ) : Prop :=
  P m ∧ Q m

theorem range_of_m (m : ℝ) : P_or_Q m ∧ ¬P_and_Q m ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 :=
by {
  sorry
}

end range_of_m_l87_87532


namespace simplify_expr_l87_87490

variable {a b c k : ℝ}

-- Conditions: a, b, c, k are nonzero real numbers.
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Definitions based on conditions
def x := b / c + c / b + k
def y := a / c + c / a
def z := a / b + b / a

theorem simplify_expr :
  x^2 + y^2 + z^2 - x * y * z =
    4 + k^2 + 2 * k * (b / c + c / b) - k * ((a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) + (a * b * c) / ((c^2 * b^2 * a^2))))
  :=
by
  sorry

end simplify_expr_l87_87490


namespace min_value_x_squared_plus_6x_l87_87238

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l87_87238


namespace find_values_of_a_l87_87783

def P : Set ℝ := { x | x^2 + x - 6 = 0 }
def S (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem find_values_of_a (a : ℝ) : (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) := by
  sorry

end find_values_of_a_l87_87783


namespace area_ratio_square_C_to_square_D_area_difference_square_C_to_square_D_l87_87562

theorem area_ratio_square_C_to_square_D (a_C a_D : ℕ) (hC : a_C = 45) (hD : a_D = 60) :
  ((a_C * a_C) : ℚ) / (a_D * a_D) = 9 / 16 :=
by
  -- provided side length conditions
  have hC : a_C = 45 := hC
  have hD : a_D = 60 := hD
  -- ensure correct typecasting for the division to succeed
  let area_C := (a_C : ℚ) * (a_C : ℚ)
  let area_D := (a_D : ℚ) * (a_D : ℚ)
  -- use ratio calc and validate the result
  show (area_C / area_D) = 9 / 16
  sorry

theorem area_difference_square_C_to_square_D (a_C a_D : ℕ) (hC : a_C = 45) (hD : a_D = 60) :
  (60 * 60) - (45 * 45) = 1575 :=
by
  -- provided side length conditions
  have hC : a_C = 45 := hC
  have hD : a_D = 60 := hD
  -- calculate the areas and validate the difference
  let area_C := a_C * a_C
  let area_D := a_D * a_D
  -- ensure the calculation confirms
  show (area_D - area_C) = 1575
  sorry

end area_ratio_square_C_to_square_D_area_difference_square_C_to_square_D_l87_87562


namespace martha_blue_butterflies_l87_87513
noncomputable theory

def total_butterflies : ℕ := 19
def blue_is_twice_yellow (B Y : ℕ) : Prop := B = 2 * Y
def black_butterflies : ℕ := 10

theorem martha_blue_butterflies (B Y : ℕ)
  (h_total : B + Y + black_butterflies = total_butterflies)
  (h_twice : blue_is_twice_yellow B Y) :
  B = 6 :=
sorry

end martha_blue_butterflies_l87_87513


namespace find_A_l87_87808

theorem find_A (A : ℤ) (h : A + 10 = 15) : A = 5 :=
sorry

end find_A_l87_87808


namespace comic_books_stack_l87_87139

theorem comic_books_stack (hulk_books ironman_books wolverine_books : Fin 8 × Fin 7 × Fin 6) : 
    (fact 8) * (fact 7) * (fact 6) * (fact 3) = 69657088000 :=
by
  sorry

end comic_books_stack_l87_87139


namespace f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l87_87016

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2^x + a

theorem f_monotone_on_0_to_2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2 :=
sorry

theorem find_range_a_part2 : (∀ x1 : ℝ, x1 ∈ (Set.Icc (1/2) 1) → 
  ∃ x2 : ℝ, x2 ∈ (Set.Icc 2 3) ∧ f x1 ≥ g x2 a) → a ≤ 1 :=
sorry

theorem find_range_a_part3 : (∃ x : ℝ, x ∈ (Set.Icc 0 2) ∧ f x ≤ g x a) → a ≥ 0 :=
sorry

end f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l87_87016


namespace regular_polygon_angle_ratio_5_3_l87_87317

theorem regular_polygon_angle_ratio_5_3 (a b : ℕ) (h1 : a ≥ 3) (h2 : b ≥ 3)
(h3 : (180 - 360 / a : ℝ) / (180 - 360 / b) = 5 / 3) : 
  (a, b) = (12, 4) → sorry

example : ∃! (a b : ℕ), a ≥ 3 ∧ b ≥ 3 ∧ (180 - 360 / a : ℝ) / (180 - 360 / b) = 5 / 3 :=
by {
  use [12, 4],
  split,
  { use 12, { split, use 12, 4; sorry }
}

end regular_polygon_angle_ratio_5_3_l87_87317


namespace minimum_expense_is_5200_l87_87995

def family : Type := {mom : Type, dad : Type, child1 : Type, child2 : Type}

def duration := 5  -- 5 days
def trips_per_day := 10  -- 10 trips per day

def adult_ticket := 40  -- rubles for one trip
def child_ticket := 20  -- rubles for one trip
def day_pass_one := 350  -- rubles for unlimited day pass for one person
def day_pass_group := 1500  -- rubles for unlimited day pass for a group 
def three_day_pass_one := 900  -- rubles for unlimited three-day pass for one person
def three_day_pass_group := 3500  -- rubles for unlimited 3-day pass for a group

def minimum_family_expense : ℕ :=
  let cost_per_adult := (three_day_pass_one + 2 * day_pass_one) in
  let total_adult_cost := 2 * cost_per_adult in
  let cost_per_child := 5 * trips_per_day * child_ticket in
  let total_child_cost := 2 * cost_per_child in
  total_adult_cost + total_child_cost

theorem minimum_expense_is_5200 : minimum_family_expense = 5200 := by
  sorry


end minimum_expense_is_5200_l87_87995


namespace distance_between_parallel_lines_l87_87764

theorem distance_between_parallel_lines :
  let A := 3
  let B := 4
  let C1 := -5
  let C2 := 5
  d = |C1 - C2| / Real.sqrt (A^2 + B^2) :=
  A = 3 ∧ B = 4 ∧ C1 = -5 ∧ C2 = 5 → d = 2 :=
begin
  sorry
end

end distance_between_parallel_lines_l87_87764


namespace points_M_B_C_N_concyclic_points_A_M_N_D_concyclic_and_lines_concurrent_l87_87124

variable (A B C D X Y Z P M N : Point)
variable [Collinear A B C D] [CirclesIntersect (Diameter A C) (Diameter B D) X Y]
variable [LineIntersects (Line X Y) (Segment B C) Z] [PointDistinct P Z]
variable [LineIntersectsCircle (Line C P) (Diameter A C) C M] [LineIntersectsCircle (Line B P) (Diameter B D) B N]

theorem points_M_B_C_N_concyclic :
  are_concyclic {M, B, C, N} := sorry

theorem points_A_M_N_D_concyclic_and_lines_concurrent :
  are_concyclic {A, M, N, D} ∧ lines_concurrent (Line A M) (Line D N) (Line X Y) := sorry

end points_M_B_C_N_concyclic_points_A_M_N_D_concyclic_and_lines_concurrent_l87_87124


namespace solve_system_l87_87836

theorem solve_system :
  ∀ (x y m : ℝ), 
    (∀ P : ℝ × ℝ, P = (3,1) → 
      (P.2 = -P.1 + 4) ∧ (P.2 = 2 * P.1 + m)) → 
    x = 3 ∧ y = 1 ↔ (x + y - 4 = 0 ∧ 2*x - y + m = 0) :=
by
  intros x y m h
  split
  case mp =>
    intro hxy
    cases hxy
    use hxy_left, hxy_right
    have hP : (3,1) = (3 : ℝ, 1 : ℝ) := rfl
    specialize h (3,1) hP
    cases h with h1 h2
    rw [h1, h2],
    exact ⟨by simp, by linarith⟩
  case mpr =>
    intro hsys
    use 3, 1
    split
    case hp1 =>
      exact (by linarith : 3 + 1 - 4 = 0)
    case hp2 =>
      rw [← h.2 3 1 _ ⟨rfl, rfl⟩],
      simp,
    exact ⟨rfl, rfl⟩

end solve_system_l87_87836


namespace transformation_of_f_l87_87895

-- Definitions and conditions
def g (x : ℝ) : ℝ := sin (2 * x)

-- The target expression for f(x)
def f (x : ℝ) : ℝ := sin (4 * x + π / 3)

-- The statement to be proved
theorem transformation_of_f :
  ∀ x, g (x - π / 6) / 2 = sin (4 * x + π / 3) :=
by
  sorry

end transformation_of_f_l87_87895


namespace problem_solution_l87_87488

noncomputable def aₙ (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 * 3^(n-1)

noncomputable def bₙ (n : ℕ) : ℝ :=
  n / (2 * 3^n)

noncomputable def Tₙ (n : ℕ) : ℝ :=
  (1 / 2) * (Finset.sum (Finset.range n) (λ i, (i + 1) / 3^(i + 1)))

theorem problem_solution :
  (∀ n > 0, aₙ n = 2 * 3^(n-1)) ∧
  (∀ n > 0, bₙ n = n / (2 * 3^n)) ∧
  (∀ n > 0, Tₙ n = (3 / 8) - ((2 * n + 3) / (8 * 3^n))) ∧
  (∀ n > 0, Tₙ n > real.log 2 / real.log m → (m ∈ (Set.Ioo 0 1) ∨ m ∈ (Set.Ioi 64)))
:=
by
  sorry

end problem_solution_l87_87488


namespace tan_theta_minus_pi_over4_l87_87058

theorem tan_theta_minus_pi_over4 (θ : Real) (h : Real.cos θ - 3 * Real.sin θ = 0) : 
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over4_l87_87058


namespace distance_between_points_l87_87949

-- Define the two points
def p1 : ℝ × ℝ := (-5, 3)
def p2 : ℝ × ℝ := (6, -9)

-- Define the distance formula
def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Statement to prove
theorem distance_between_points : distance p1 p2 = Real.sqrt 265 := by
  sorry

end distance_between_points_l87_87949


namespace pattern_proof_l87_87880

theorem pattern_proof (n : ℕ) (hn : n > 1) : 
  (sqrt (n + (n / (n^2 - 1))) = n * sqrt (n / (n^2 - 1))) :=
by
  sorry

end pattern_proof_l87_87880


namespace ball_bounce_height_l87_87625

theorem ball_bounce_height (a : ℝ) (r : ℝ) (threshold : ℝ) (k : ℕ) 
  (h_a : a = 20) (h_r : r = 1/2) (h_threshold : threshold = 0.5) :
  20 * (r^k) < threshold ↔ k = 5 :=
by sorry

end ball_bounce_height_l87_87625


namespace ellipse_foci_on_y_axis_l87_87766

theorem ellipse_foci_on_y_axis (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (m + 2)) - (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -3/2) := 
by
  sorry

end ellipse_foci_on_y_axis_l87_87766


namespace total_tomato_seeds_l87_87135

theorem total_tomato_seeds (morn_mike morn_morning ted_morning sarah_morning : ℕ)
    (aft_mike aft_ted aft_sarah : ℕ)
    (H1 : morn_mike = 50)
    (H2 : ted_morning = 2 * morn_mike)
    (H3 : sarah_morning = morn_mike + 30)
    (H4 : aft_mike = 60)
    (H5 : aft_ted = aft_mike - 20)
    (H6 : aft_sarah = sarah_morning + 20) :
    morn_mike + aft_mike + ted_morning + aft_ted + sarah_morning + aft_sarah = 430 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  sorry

end total_tomato_seeds_l87_87135


namespace ninggao_intercity_project_cost_in_scientific_notation_l87_87564

theorem ninggao_intercity_project_cost_in_scientific_notation :
  let length_kilometers := 55
  let cost_per_kilometer_million := 140
  let total_cost_million := length_kilometers * cost_per_kilometer_million
  let total_cost_scientific := 7.7 * 10^6
  total_cost_million = total_cost_scientific := 
  sorry

end ninggao_intercity_project_cost_in_scientific_notation_l87_87564


namespace coefficient_of_squared_term_l87_87645

theorem coefficient_of_squared_term (a b c : ℝ) (h_eq : 5 * a^2 + 14 * b + 5 = 0) :
  a = 5 :=
sorry

end coefficient_of_squared_term_l87_87645


namespace correct_indices_l87_87292

def proposition_1 := ∃ x : ℝ, x^2 + x - 1 < 0

def proposition_2 (p q : Prop) : Prop := (q → p) ∧ ¬(p → q) → (¬p → ¬q) ∧ ¬(¬q → ¬p)

def proposition_3 (x y : ℝ) : Prop := (sin x ≠ sin y) → (x ≠ y)

def proposition_4 (x y : ℝ) : Prop := (log x > log y) ↔ (x > y)

theorem correct_indices : [2, 3] = [2, 3] := by sorry

end correct_indices_l87_87292


namespace incorrect_statements_l87_87965

-- Defining the first condition
def condition1 : Prop :=
  let a_sq := 169
  let b_sq := 144
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  ¬((c_, 0) ∈ focal_points) ∧ ¬((-c_, 0) ∈ focal_points)

-- Defining the second condition
def condition2 : Prop :=
  let m := 1  -- Example choice since m is unspecified
  let a_sq := m^2 + 1
  let b_sq := m^2
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  (0, 1) ∈ focal_points ∧ (0, -1) ∈ focal_points

-- Defining the third condition
def condition3 : Prop :=
  let a1_sq := 16
  let b1_sq := 7
  let c1_sq := a1_sq - b1_sq
  let c1_ := Real.sqrt c1_sq
  let focal_points1 := [(c1_, 0), (-c1_, 0)]
  
  let m := 10  -- Example choice since m > 0 is unspecified
  let a2_sq := m - 5
  let b2_sq := m + 4
  let c2_sq := a2_sq - b2_sq
  let focal_points2 := [(0, Real.sqrt c2_sq), (0, -Real.sqrt c2_sq)]
  
  ¬ (focal_points1 = focal_points2)

-- Defining the fourth condition
def condition4 : Prop :=
  let B := (-3, 0)
  let C := (3, 0)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BC_dist := Real.sqrt (BC.1^2 + BC.2^2)
  let A_locus_eq := ∀ (x y : ℝ), x^2 / 36 + y^2 / 27 = 1
  2 * BC_dist = 12

-- Proof verification
theorem incorrect_statements : Prop :=
  condition1 ∧ condition3

end incorrect_statements_l87_87965


namespace radius_of_sphere_in_truncated_cone_l87_87661

theorem radius_of_sphere_in_truncated_cone :
  ∀ (r1 r2 : ℝ) (h : ℝ), 
  r1 = 20 → r2 = 5 →
  0 < h → -- Assuming positive height to ensure cone is well-defined
  ∃ (r_sphere : ℝ), 
  r_sphere = 10 :=
by
  assume r1 r2 h hr1 hr2 hh,
  use 10,
  sorry

end radius_of_sphere_in_truncated_cone_l87_87661


namespace KLM_area_is_fraction_of_ABC_l87_87521

-- Definitions of triangle and points
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A : Point) (B : Point) (C : Point)

noncomputable def area (T : Triangle) : ℝ :=
  0.5 * abs ((T.A.x * (T.B.y - T.C.y)) + (T.B.x * (T.C.y - T.A.y)) + (T.C.x * (T.A.y - T.B.y)))

-- Defining points K, L, M
noncomputable def KLM_area_fraction (T : Triangle) (K : Point) (L : Point) (M : Point) : ℝ :=
  let KLM : Triangle := ⟨K, L, M⟩
  area KLM / area T

-- Main theorem
theorem KLM_area_is_fraction_of_ABC (ABC : Triangle) (X : Point) (Y : Point) (Z : Point) (K : Point) (L : Point) (M : Point)
  (hX : ∃ k, X = Point.mk (k * ABC.B.x + (1 - k) * ABC.C.x) (k * ABC.B.y + (1 - k) * ABC.C.y))
  (hY : ∃ k, Y = Point.mk ((1 + k) * ABC.C.x + (1 - k) * ABC.A.x) / (2 + k) ((1 + k) * ABC.C.y + (1 - k) * ABC.A.y) / (2 + k))
  (hZ : ∃ k, Z = Point.mk ((1 + k) * ABC.B.x + (1 - k) * ABC.A.x) / (2 + k) ((1 + k) * ABC.B.y + (1 - k) * ABC.A.y) / (2 + k))
  (hK : K = Point.mk ((2 + k) * ABC.B.x + (1 + k)) / (4 + 2 * k) ((2 + k) * ABC.B.y + (1 + k)) / (4 + 2 * k))
  (hL : L = Point.mk ((1 + k) * ABC.C.x + (2 + k)) / (4 + 2 * k) ((1 + k) * ABC.C.y + (2 + k)) / (4 + 2 * k))
  (hM : M = Point.mk ((1 + k) * ABC.A.x + (2 + k) * ABC.B.x) / (4 + k) ((1 + k) * ABC.A.y + (2 + k) * ABC.B.y) / (4 + k)) :
  KLM_area_fraction ABC K L M = (7 - 3 * real.sqrt 5) / 4 :=
by sorry

end KLM_area_is_fraction_of_ABC_l87_87521


namespace find_measure_angle_AOD_l87_87075

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

end find_measure_angle_AOD_l87_87075


namespace catch_up_count_l87_87205

theorem catch_up_count (n m l : ℕ) (hn : n = 2015) (hm : m = 23) (hl : l = 13) :
  let total_flags := 2015 in
  let laps_A := 23 in
  let laps_B := 13 in
  n = total_flags →
  m = laps_A →
  l = laps_B →
  ∃ k, k = 5 ∧
        ∀ t, (laps_A - laps_B) * t / total_flags ∈ ℕ →
        (A catches B at (laps_A - laps_B)) - 1 = k :=
sorry

end catch_up_count_l87_87205


namespace number_of_valid_pairs_l87_87647

theorem number_of_valid_pairs (a b : ℕ) (cond1 : b > a) (cond2 : ∃ a b, b > a ∧ (2*a*b = 3*(a-4)*(b-4))) : 
    ∃ (pairs : set (ℕ × ℕ)), pairs = {(13, 108), (14, 60), (15, 44)} ∧ pairs.size = 3 :=
begin
  sorry
end

end number_of_valid_pairs_l87_87647


namespace inscribed_sphere_radius_of_tetrahedron_is_correct_l87_87010

noncomputable def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

noncomputable def unit_cube_midpoints : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → Prop :=
  λ A B A1 D1, 
    let L := midpoint A B in
    let M := midpoint A1 D1 in
    let N := midpoint A1 (1, 1, 1) in -- A1B1 assumed to end at (1, 1, 1) as B1 from problem statement
    let K := midpoint B (1, 0, 0) in -- BC assumed to end at (1, 0, 0) as C from problem statement
    true

theorem inscribed_sphere_radius_of_tetrahedron_is_correct 
  (A B A1 D1 : ℝ × ℝ × ℝ)
  (h : unit_cube_midpoints A B A1 D1) :
  ∃ r : ℝ, r = (sqrt 3 - sqrt 2) / 2 :=
begin
  sorry
end

end inscribed_sphere_radius_of_tetrahedron_is_correct_l87_87010


namespace ball_highest_point_at_l87_87184

noncomputable def h (a b t : ℝ) : ℝ := a * t^2 + b * t

theorem ball_highest_point_at (a b : ℝ) :
  (h a b 3 = h a b 7) →
  t = 4.9 :=
by
  sorry

end ball_highest_point_at_l87_87184


namespace maximize_T_n_l87_87854

noncomputable def q : ℝ := Real.sqrt 2

-- Definition of the geometric sequence
def a_n (a1 : ℝ) (n : ℕ) : ℝ := a1 * q^(n - 1)

-- Sum of the first n terms of the geometric sequence
def S_n (a1 : ℝ) (n : ℕ) : ℝ := a1 * (1 - q^n) / (1 - q)

-- Definition of T_n
def T_n (a1 : ℝ) (n : ℕ) (hn : n > 0) : ℝ := (17 * S_n a1 n - S_n a1 (2 * n)) / a_n a1 (n + 1)

-- Prove that T_m is maximized when m = 4
theorem maximize_T_n (a1 : ℝ) (m : ℕ) (hm : m > 0) : m = 4 :=
  sorry

end maximize_T_n_l87_87854


namespace clock_angle_7_15_l87_87586

theorem clock_angle_7_15 : 
  let degrees_per_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := degrees_per_circle / hours_on_clock
  let minutes_per_hour := 60
  let degrees_per_minute := degrees_per_circle / minutes_per_hour
  let hour := 7
  let minute := 15
  let hour_hand_position := hour * degrees_per_hour + (degrees_per_hour / minutes_per_hour) * minute
  let minute_hand_position := minute * degrees_per_minute
  let angle_between_hands := |hour_hand_position - minute_hand_position|
  let smaller_angle := min angle_between_hands (degrees_per_circle - angle_between_hands)
  in smaller_angle = 127.5 :=
by
  let degrees_per_circle := 360
  let hours_on_clock := 12
  let degrees_per_hour := degrees_per_circle / hours_on_clock
  let minutes_per_hour := 60
  let degrees_per_minute := degrees_per_circle / minutes_per_hour
  let hour := 7
  let minute := 15
  let hour_hand_position := hour * degrees_per_hour + (degrees_per_hour / minutes_per_hour) * minute
  let minute_hand_position := minute * degrees_per_minute
  let angle_between_hands := abs (hour_hand_position - minute_hand_position)
  let smaller_angle := min angle_between_hands (degrees_per_circle - angle_between_hands)
  show smaller_angle = 127.5, from sorry

end clock_angle_7_15_l87_87586


namespace simplest_square_root_l87_87242

noncomputable def is_simple (x : ℝ) : Prop :=
  (x = real.sqrt 0.5 → false) ∧
  (x = real.sqrt (2 / 5) → false) ∧
  (x = real.sqrt 8 → false)

theorem simplest_square_root :
  ∀ (x : ℝ), x ∈ {real.sqrt 3, real.sqrt 0.5, real.sqrt (2/5), real.sqrt 8} →
  is_simple x → x = real.sqrt 3 :=
by
  sorry

end simplest_square_root_l87_87242


namespace range_of_m_l87_87002

variable (m : ℝ)

def p : Prop :=
  ∀ x ∈ Icc (-1 : ℝ) 1, x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0

def q : Prop :=
  ∃ x ∈ Icc (1 : ℝ) 2, log (1/2) (x^2 - m*x + 1) < -1

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬ (p m ∧ q m)) : 
  m < (1 / 2) ∨ m = (3 / 2) := sorry

end range_of_m_l87_87002


namespace frog_jump_distance_l87_87912

noncomputable def grasshopper_jump : ℝ := 31
noncomputable def total_jump : ℝ := 66

theorem frog_jump_distance : ∃ frog_jump : ℝ, frog_jump = total_jump - grasshopper_jump ∧ frog_jump = 35 := 
by
  use total_jump - grasshopper_jump
  split
  { sorry }
  { sorry }

end frog_jump_distance_l87_87912


namespace complex_number_simplification_l87_87540

theorem complex_number_simplification : 
  (1 + 5 * complex.I) / (5 - complex.I) = complex.I := 
by sorry

end complex_number_simplification_l87_87540


namespace brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l87_87144

def march_13_2007_day_of_week : String := "Tuesday"

def days_until_brothers_birthday : Nat := 2000

def start_date := (2007, 3, 13)  -- (year, month, day)

def days_per_week := 7

def carlos_initial_age := 7

def day_of_week_after_n_days (start_day : String) (n : Nat) : String :=
  match n % 7 with
  | 0 => "Tuesday"
  | 1 => "Wednesday"
  | 2 => "Thursday"
  | 3 => "Friday"
  | 4 => "Saturday"
  | 5 => "Sunday"
  | 6 => "Monday"
  | _ => "Unknown" -- This case should never happen

def carlos_age_after_n_days (initial_age : Nat) (n : Nat) : Nat :=
  initial_age + n / 365

theorem brother_15th_birthday_day_of_week : 
  day_of_week_after_n_days march_13_2007_day_of_week days_until_brothers_birthday = "Sunday" := 
by sorry

theorem carlos_age_on_brothers_15th_birthday :
  carlos_age_after_n_days carlos_initial_age days_until_brothers_birthday = 12 :=
by sorry

end brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l87_87144


namespace spools_per_beret_l87_87845

theorem spools_per_beret (r b u : ℕ) (h_r : r = 12) (h_b : b = 15) (h_u : u = 6) (h_berets : r + b + u = 33) :
  (r + b + u) / 11 = 3 :=
by {
  simp [h_r, h_b, h_u, h_berets],
  sorry
}

end spools_per_beret_l87_87845


namespace usage_of_pencil_l87_87299

theorem usage_of_pencil (daily_puzzles : ℕ) (days_per_pencil : ℕ) (words_per_puzzle : ℕ) (days: ℕ) :
  days_per_pencil = 14 →
  words_per_puzzle = 75 →
  days = daily_puzzles →
  daily_puzzles = days_per_pencil →
  days * words_per_puzzle = 1050 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp only [Nat.mul_eq_mul_left_iff]
  exact Or.inl rfl

end usage_of_pencil_l87_87299


namespace allocation_schemes_l87_87345

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l87_87345


namespace john_rental_weeks_l87_87847

noncomputable def camera_value : ℝ := 5000
noncomputable def rental_fee_rate : ℝ := 0.10
noncomputable def friend_payment_rate : ℝ := 0.40
noncomputable def john_total_payment : ℝ := 1200

theorem john_rental_weeks :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let friend_payment := weekly_rental_fee * friend_payment_rate
  let john_weekly_payment := weekly_rental_fee - friend_payment
  let rental_weeks := john_total_payment / john_weekly_payment
  rental_weeks = 4 :=
by
  -- Place for proof steps
  sorry

end john_rental_weeks_l87_87847


namespace Jose_Jane_together_days_l87_87110

theorem Jose_Jane_together_days (J : ℝ) : 
  (1 / 15 + 1 / J)⁻¹ = 7.5 :=
by
  have H1 : Jose_rate = 1 / 15 := sorry
  have H2 : (1 / 15) * 7.5 = 1 / 2 := sorry
  have H3 : 7.5 * (1 / J) = 1 / 2 := sorry
  have H4 : J = 15 := sorry
  have H5 : (1 / 15 + 1 / 15)⁻¹ = 7.5 := sorry
  sorry

end Jose_Jane_together_days_l87_87110


namespace unit_prices_l87_87081

variables (x : ℝ) (unit_price_B unit_price_A : ℝ)

def unit_price_B : ℝ := 20
def unit_price_A : ℝ := 2.5 * unit_price_B

theorem unit_prices (h : 700 / x - 700 / (2.5 * x) = 21) :
  unit_price_B = 20 ∧ unit_price_A = 50 := by
  sorry

end unit_prices_l87_87081


namespace round_eight_sevenths_l87_87888

theorem round_eight_sevenths : Float.toThreeDecimalPlaces (8 / 7) = 1.143 := 
by
  sorry

end round_eight_sevenths_l87_87888


namespace series_sum_solution_l87_87855

noncomputable def series_sum (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) : ℝ :=
  ∑' n : ℕ, (1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b)))

theorem series_sum_solution (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) :
  series_sum a b c h₀ h₁ h₂ h₃ h₄ = 1 / ((c - b) * c) := 
  sorry

end series_sum_solution_l87_87855


namespace range_of_m_l87_87064

noncomputable def f (x m : ℝ) : ℝ := x / (1 + |x|) - m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = 0) ↔ m ∈ Ioo (-1 : ℝ) 1 :=
by
  sorry

end range_of_m_l87_87064


namespace simplify_expression_l87_87167

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (2 * x + 10) - (x + 3) * (3 * x - 2) = 3 * x^2 + 15 * x - 34 := 
by
  sorry

end simplify_expression_l87_87167


namespace gcd_18_30_l87_87333

theorem gcd_18_30 : Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l87_87333


namespace soviet_olympiad_1987_l87_87324

def is_coprime (a b : Nat) : Prop := Nat.gcd a b = 1

def is_composite (n : Nat) : Prop := ∃ p q : Nat, p > 1 ∧ q > 1 ∧ n = p * q

def number_set : Set Nat := {121, 241, 361, 481, 601}

theorem soviet_olympiad_1987: 
  (∀ a b ∈ number_set, a ≠ b → is_coprime a b) ∧
  (∀ s ⊆ number_set, 1 < s.card → is_composite (s.sum)) :=
sorry

end soviet_olympiad_1987_l87_87324


namespace smallest_value_W_n_l87_87728

noncomputable def W_n (n : ℕ) (x : ℝ) : ℝ := 
  (Finset.range n).sum (λ k, (k + 1) * x^(2 * n - k))

theorem smallest_value_W_n (n : ℕ) (h : 0 < n) : 
  ∃ x : ℝ, W_n n x = -n :=
by
  sorry

end smallest_value_W_n_l87_87728


namespace minimum_value_real_l87_87232

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l87_87232


namespace difference_rabbits_antelopes_l87_87848

variable (A R H W L : ℕ)
variable (x : ℕ)

def antelopes := 80
def rabbits := antelopes + x
def hyenas := (antelopes + rabbits) - 42
def wild_dogs := hyenas + 50
def leopards := rabbits / 2
def total_animals := 605

theorem difference_rabbits_antelopes
  (h1 : antelopes = 80)
  (h2 : rabbits = antelopes + x)
  (h3 : hyenas = (antelopes + rabbits) - 42)
  (h4 : wild_dogs = hyenas + 50)
  (h5 : leopards = rabbits / 2)
  (h6 : antelopes + rabbits + hyenas + wild_dogs + leopards = total_animals) : rabbits - antelopes = 70 := 
by
  -- Proof goes here
  sorry

end difference_rabbits_antelopes_l87_87848


namespace percentage_of_loss_is_25_l87_87999

def CP : ℝ := 1800
def SP : ℝ := 1350
def Loss : ℝ := CP - SP
def percentage_of_loss : ℝ := (Loss / CP) * 100

theorem percentage_of_loss_is_25 : percentage_of_loss = 25 :=
by
  -- Proof steps go here
  sorry

end percentage_of_loss_is_25_l87_87999


namespace axis_of_symmetry_l87_87034

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry :
  (|Real.pi / 3| < Real.pi / 2) →
  2 = 2 →
  f (-Real.pi / 6) = 0 →
  (∃ k : ℤ, axis_of_symmetry_formula (2 * k + 1) (-Real.pi / 12)) :=
by obviously_analyze

end axis_of_symmetry_l87_87034


namespace nina_money_l87_87257

variable (C : ℝ)

theorem nina_money (h1: 6 * C = 8 * (C - 1.15)) : 6 * C = 27.6 := by
  have h2: C = 4.6 := sorry
  rw [h2]
  norm_num
  done

end nina_money_l87_87257


namespace det_D_eq_125_l87_87115

def D : Matrix (Fin 3) (Fin 3) ℝ := ![![5, 0, 0], ![0, 5, 0], ![0, 0, 5]]

theorem det_D_eq_125 : det D = 125 :=
by
  sorry

end det_D_eq_125_l87_87115


namespace interval_of_defined_expression_l87_87351

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l87_87351


namespace seven_searchlights_shadow_length_l87_87891

noncomputable def searchlight_positioning (n : ℕ) (angle : ℝ) (shadow_length : ℝ) : Prop :=
  ∃ (positions : Fin n → ℝ × ℝ), ∀ i : Fin n, ∃ shadow : ℝ, shadow = shadow_length ∧
  (∀ j : Fin n, i ≠ j → ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  θ - angle / 2 < θ ∧ θ + angle / 2 > θ → shadow = shadow_length)

theorem seven_searchlights_shadow_length :
  searchlight_positioning 7 (Real.pi / 2) 7000 :=
sorry

end seven_searchlights_shadow_length_l87_87891


namespace expr_xn_add_inv_xn_l87_87732

theorem expr_xn_add_inv_xn {θ : ℝ} {x : ℂ} (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : x + x⁻¹ = 2 * (Real.cos θ)) (n : ℤ) : x^n + (x^n)⁻¹ = 2 * (Real.cos (n * θ)) :=
sorry

end expr_xn_add_inv_xn_l87_87732


namespace find_square_side_length_l87_87990

-- Define the necessary conditions and problem statement
def circle_tangent_side_length (s : ℝ) : Prop :=
  ∃ (r : ℝ) (OA : ℝ) (cos15 sin15 : ℝ),
  (r = 2) ∧
  (OA = 2 * Real.sqrt 2) ∧
  (cos15 = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (sin15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2)) ∧
  s = 2 * Real.sqrt 3

theorem find_square_side_length : 
  ∃ s : ℝ, circle_tangent_side_length s :=
by 
  use 2 * Real.sqrt 3
  unfold circle_tangent_side_length
  repeat {split}; 
  try {norm_num}; 
  try {apply Real.sqrt};
  try {ring};
  sorry

end find_square_side_length_l87_87990


namespace polynomial_base5_representation_l87_87600

-- Define the polynomials P and Q
def P(x : ℕ) : ℕ := 3 * 5^6 + 0 * 5^5 + 0 * 5^4 + 1 * 5^3 + 2 * 5^2 + 4 * 5 + 1
def Q(x : ℕ) : ℕ := 4 * 5^2 + 3 * 5 + 2

-- Define the representation of these polynomials in base-5
def base5_P : ℕ := 3001241
def base5_Q : ℕ := 432

-- Define the expected interpretation of the base-5 representation in decimal
def decimal_P : ℕ := P 0
def decimal_Q : ℕ := Q 0

-- The proof statement
theorem polynomial_base5_representation :
  decimal_P = base5_P ∧ decimal_Q = base5_Q :=
sorry

end polynomial_base5_representation_l87_87600


namespace polynomial_degree_and_terms_l87_87085

-- Condition: Define the polynomial 3ba^2 + ac + 1
def p (a b c : ℕ) : ℕ := 3 * b * a^2 + a * c + 1

-- The main theorem statement to prove:
theorem polynomial_degree_and_terms {a b c : ℕ} :
  (degree (p a b c) = 3 ∧ number_of_terms (p a b c) = 3) :=
by
  -- Proof omitted; use 'sorry' to skip
  sorry

end polynomial_degree_and_terms_l87_87085


namespace total_carriages_proof_l87_87599

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l87_87599


namespace nature_reserve_birds_l87_87449

theorem nature_reserve_birds :
  ∀ N : ℕ, 
    (N > 0) →
    let hawks := 0.30 * N in
    let non_hawks := N - hawks in
    let paddyfield_warblers := 0.40 * non_hawks in
    let kingfishers := 0.25 * paddyfield_warblers in
    let non_hawks_paddyfield_warblers_kingfishers := hawks + paddyfield_warblers + kingfishers in
  N - non_hawks_paddyfield_warblers_kingfishers = 0.35 * N :=
by
  intros N hN hawks non_hawks paddyfield_warblers kingfishers non_hawks_paddyfield_warblers_kingfishers,
  sorry

end nature_reserve_birds_l87_87449


namespace isosceles_triangle_area_l87_87819

theorem isosceles_triangle_area (PQ PR QR : ℝ) (PS : ℝ) (h1 : PQ = PR)
  (h2 : QR = 10) (h3 : PS^2 + (QR / 2)^2 = PQ^2) : 
  (1/2) * QR * PS = 60 :=
by
  sorry

end isosceles_triangle_area_l87_87819


namespace primes_for_integral_products_l87_87871

theorem primes_for_integral_products 
  (m : ℕ) (hm : 0 < m) (p : ℕ) (hp : Prime p) :
  (∀ (a : ℕ → ℕ) (h₁ : a 1 = 8 * p ^ m) (h₂ : ∀ n, 2 ≤ n → a n = (n + 1) ^ (a (n - 1) / n)),
  ∀ n ≥ 1, (a n * ∏ i in Finset.range n, (1 - 1 / a i.succ) : ℚ) ∈ ℤ) ↔ p ∈ {2, 5} :=
sorry

end primes_for_integral_products_l87_87871


namespace perpendicular_lines_b_value_l87_87187

theorem perpendicular_lines_b_value : (b : ℝ) (h1 : 5 * y + 2 * x - 7 = 0) (h2 : 4 * y + b * x - 8 = 0) 
  (h3 : y = - 2/5 * x + 7/5) (h4 : y = - b/4 * x + 2) : b = -10 :=
by sorry

end perpendicular_lines_b_value_l87_87187


namespace smallest_number_to_add_l87_87224

theorem smallest_number_to_add (a : ℕ) (b : ℕ) (h : a = 87908235) (k : b = 12587) : 
  let r := a % b in 
  if r = 0 then 0 else b - r = 0 := 
by 
  sorry

end smallest_number_to_add_l87_87224


namespace monotonicity_of_g_max_value_h_l87_87041

-- Problem 1: Monotonicity of g(x)
theorem monotonicity_of_g (a : ℝ) :
  (∀ x, -1 < x → 2 * (x + 1) ^ 2 + a ≥ 0) ∨
  (∀ x, -1 < x → 2 * (x + 1) ^ 2 + a ≤ 0 ∧ (exists l, x < l ∧ 2 * (l + 1) ^ 2 + a > 0)) → 
  (∀ x y, -1 < x ∧ -1 < y ∧ (x < y → g(x) ≤ g(y))) := sorry

-- Problem 2: Maximum value of h(x)
theorem max_value_h (x₀ : ℝ)
  (h : ℝ → ℝ)
  (h_def : ∀ x, h x = x^2 + 2*x - exp x)
  (x_max : (∀ x, h x ≤ h x₀))
  (window : 3 / 2 < x₀ ∧ x₀ < 2) :
  1 / 4 < h x₀ ∧ h x₀ < 2 := sorry

end monotonicity_of_g_max_value_h_l87_87041


namespace min_value_x_squared_plus_6x_l87_87237

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l87_87237


namespace shaded_area_10x12_floor_l87_87268

theorem shaded_area_10x12_floor :
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  total_shaded_area = 90 - 30 * π :=
by
  let tile_white_area := π + 1
  let tile_total_area := 4
  let tile_shaded_area := tile_total_area - tile_white_area
  let num_tiles := (10 / 2) * (12 / 2)
  let total_shaded_area := num_tiles * tile_shaded_area
  show total_shaded_area = 90 - 30 * π
  sorry

end shaded_area_10x12_floor_l87_87268


namespace proof_problem_l87_87787

variables (x y : ℝ)

-- Conditions
def condition1 : Prop := 5 = Real.sqrt (x - 5)
def condition2 : Prop := 3 = Real.cbrt (x + y)

-- Correct Answers
def answer1_x : Prop := x = 30
def answer1_y : Prop := y = -3
def answer2 : Prop := Real.sqrt (x + 2 * y) = 2 * Real.sqrt 6 ∨ Real.sqrt (x + 2 * y) = -2 * Real.sqrt 6

-- Theorem stating the proof problem
theorem proof_problem : condition1 ∧ condition2 → answer1_x ∧ answer1_y ∧ answer2 :=
by
  intro h,
  cases h with h1 h2,
  sorry

end proof_problem_l87_87787


namespace common_divisors_count_of_120_and_180_l87_87050

def factorization_120 := [2, 2, 2, 3, 5]
def factorization_180 := [2, 2, 3, 3, 5]

noncomputable def gcd_120_180 := Nat.gcd 120 180

def num_divisors (n : ℕ) : ℕ :=
  List.length (List.filter (λ d, n % d = 0) (List.range (n + 1)))

theorem common_divisors_count_of_120_and_180 :
  num_divisors gcd_120_180 = 24 :=
by
  sorry

end common_divisors_count_of_120_and_180_l87_87050


namespace spinner_prime_probability_l87_87587

def spinner_labels : List ℕ := [3, 6, 1, 4, 5, 2]

def total_outcomes : ℕ := spinner_labels.length

def is_prime (n : ℕ) : Bool := n = 2 ∨ n = 3 ∨ n = 5

def prime_count : ℕ := spinner_labels.countp is_prime

def probability_of_prime : ℚ := prime_count / total_outcomes

theorem spinner_prime_probability :
  probability_of_prime = 1 / 2 := by
  sorry

end spinner_prime_probability_l87_87587


namespace decrease_in_silver_coins_l87_87457

theorem decrease_in_silver_coins
  (a : ℕ) (h₁ : 2 * a = 3 * (50 - a))
  (h₂ : a + (50 - a) = 50) :
  (5 * (50 - a) - 3 * a = 10) :=
by
sorry

end decrease_in_silver_coins_l87_87457


namespace part_a_l87_87253

theorem part_a (a : ℤ) : (a^2 < 4) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := 
sorry

end part_a_l87_87253


namespace percentage_answered_second_question_l87_87061

theorem percentage_answered_second_question (q1 q2 both neith : ℝ) (h1 : q1 = 0.75) (h2 : both = 0.2) (h3 : neith = 0.2) :
  q2 = 0.25 :=
by
  have h_total : q1 + q2 - both = 1 - neith,
  { sorry },
  linarith

end percentage_answered_second_question_l87_87061


namespace telescoping_series_sum_l87_87462

theorem telescoping_series_sum (n : ℕ) (hn : n ≥ 2) :
  ∑ k in finset.range (n - 1), (n^2 / (k + 1) - n^2 / (k + 2)) = n * (n - 1) :=
by
  sorry

end telescoping_series_sum_l87_87462


namespace find_a_value_l87_87762

variable (α a : ℝ)

-- State the given conditions
def condition1 := ∃ (y : ℝ), (a, y) = (a, -2)
def condition2 := Real.tan (π + α) = 1 / 3

-- Define what we want to prove
theorem find_a_value (h1 : condition1 a) (h2 : condition2 α) : a = -6 := 
  sorry

end find_a_value_l87_87762


namespace cyrus_missed_shots_l87_87073

theorem cyrus_missed_shots (total_shots attempted_shots_percentage missed_shots: ℕ): 
  attempted_shots_percentage = 80 → total_shots = 20 → missed_shots = (total_shots - (attempted_shots_percentage * total_shots / 100)) → missed_shots = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2]
  simp
  sorry

end cyrus_missed_shots_l87_87073


namespace fill_tank_time_l87_87250

theorem fill_tank_time (time_A time_B : ℕ) (rate_A := 1 / (time_A : ℚ)) (rate_B := 1 / (time_B : ℚ)) :
  time_A = 60 → time_B = 40 →
  (∀ T : ℚ, (rate_B * (T / 2) + (rate_A + rate_B) * (T / 2) = 1) → T = 30) :=
by
  intros hA hB
  intro T
  rw [hA, hB] at *
  let rate_A' := 1 / 60
  let rate_B' := 1 / 40
  have h_rate_A : rate_A = rate_A', from rfl
  have h_rate_B : rate_B = rate_B', from rfl
  rw [h_rate_A, h_rate_B]
  sorry

end fill_tank_time_l87_87250


namespace log_sqrt_defined_l87_87366

theorem log_sqrt_defined (x : ℝ) : 2 < x ∧ x < 5 ↔ (∃ y z : ℝ, y = log (5 - x) ∧ z = sqrt (x - 2)) :=
by 
  sorry

end log_sqrt_defined_l87_87366


namespace third_circle_radius_l87_87940

-- Condition definitions
def radius1 : ℝ := 15
def radius2 : ℝ := 25
def area_between_circles : ℝ := π * radius2^2 - π * radius1^2

-- Question
theorem third_circle_radius :
  ∃ r : ℝ, π * r^2 = area_between_circles ∧ r = 20 :=
begin
  sorry
end

end third_circle_radius_l87_87940


namespace find_all_f_l87_87112

-- Definitions: Let's define the function f and express the conditions provided.

def R_poly := polynomial ℝ

def function_f (f : R_poly → R_poly) : Prop :=
  (f 0 = 0) ∧
  (∀ P : R_poly, P ≠ 0 → degree (f P) ≤ degree P + 1) ∧
  (∀ P Q : R_poly, (roots (P - f Q) = roots (Q - f P)))

-- Theorem statement: Prove that f(Q) = Q or f(Q) = -Q for all Q ∈ R[x]
theorem find_all_f (f : R_poly → R_poly) (hf : function_f f) :
  (∀ Q : R_poly, f Q = Q ∨ f Q = -Q) :=
sorry

end find_all_f_l87_87112


namespace nth_row_all_ones_l87_87887

/--
In the modified Pascal's triangle (where odd numbers are replaced with 1 and even numbers with 0),
prove that the nth row (counting all rows consisting entirely of 1s) is the 2^n - 1 row.
-/
theorem nth_row_all_ones (n : ℕ) : 
  nth_row_modified_pascals_triangle_all_ones n = 2^n - 1 :=
sorry

end nth_row_all_ones_l87_87887


namespace pyramid_base_edge_length_l87_87539

theorem pyramid_base_edge_length
  (hemisphere_radius : ℝ) (pyramid_height : ℝ) (slant_height : ℝ) (is_tangent: Prop) :
  hemisphere_radius = 3 ∧ pyramid_height = 8 ∧ slant_height = 10 ∧ is_tangent →
  ∃ (base_edge_length : ℝ), base_edge_length = 6 * Real.sqrt 2 :=
by
  sorry

end pyramid_base_edge_length_l87_87539


namespace prime_base_problem_l87_87131

theorem prime_base_problem :
  ∀ {p : ℕ}, p.prime → p ≠ 2 → p ≠ 3 → p ≠ 5 → p ≠ 7 →
  (1014*p^3 + 309*p^2 + 120*p + 7) = (153*p^4 + 276*p^3 + 371*p*^2) → 
  false := 
by
  intro p h_prime h2 h3 h5 h7 h_expansion_eq
  sorry

end prime_base_problem_l87_87131


namespace jancsi_pista_wrong_l87_87475

-- Condition: N is the concatenation of all integers from 1 to 1975
def N : ℕ := -- expression representing concatenated number
  sorry

-- Definitions for divisors count claims by Jancsi and Pista
def tau (n : ℕ) : ℕ := -- Number of divisors function
  sorry

-- Lean statement to prove both claims are incorrect
theorem jancsi_pista_wrong : tau(N) ≠ 25323 ∧ tau(N) ≠ 25322 :=
  sorry

end jancsi_pista_wrong_l87_87475


namespace ellipse_foci_coordinates_l87_87977

-- Define the parameters and conditions
def h := 2
def k := -3
def a := 5
def b := 4

-- Define the parametric equations
def x (θ : ℝ) := h + a * Real.cos θ
def y (θ : ℝ) := k + b * Real.sin θ

-- Define the distance to the foci
def c := Real.sqrt (a^2 - b^2)

theorem ellipse_foci_coordinates :
  (let f1 := (h - c, k),
       f2 := (h + c, k) in
    f1 = (-1, -3) ∧ f2 = (5, -3)) :=
by
  sorry

end ellipse_foci_coordinates_l87_87977


namespace f_increasing_on_positive_l87_87525

noncomputable def f (x : ℝ) : ℝ := - (1 / x) - 1

theorem f_increasing_on_positive (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f x1 > f x2 := by
  sorry

end f_increasing_on_positive_l87_87525


namespace BertrandOddConjectureCounterexample_l87_87300

theorem BertrandOddConjectureCounterexample :
  ∃ (n : ℤ), odd n ∧ n ≥ some_large_threshold ∧ ¬(∃ (p1 p2 p3 : ℤ), prime p1 ∧ prime p2 ∧ prime p3 ∧ odd p1 ∧ odd p2 ∧ odd p3 ∧ n = p1 + p2 + p3) :=
sorry

end BertrandOddConjectureCounterexample_l87_87300


namespace find_F3_l87_87758

variable (F1 F2 F3 : ℝ × ℝ)

def equilibrium_condition :=
  F1 + F2 + F3 = (0, 0)

-- Forces given in the problem
axiom F1_def : F1 = (1, 2)
axiom F2_def : F2 = (-1, -3)

-- Theorem to prove
theorem find_F3 (h : equilibrium_condition F1 F2 F3) : F3 = (0, 1) :=
by
  simp [equilibrium_condition, F1_def, F2_def] at h
  exact sorry

end find_F3_l87_87758


namespace problem_1_l87_87409

noncomputable def ω := 2
noncomputable def φ := - (Real.pi / 6)

def f (x : ℝ) := 2 * Real.sin (ω * x + φ)
def g (x : ℝ) := 2 * Real.sin (x / 2 - Real.pi / 3)

theorem problem_1 (hω : ω = 2) (hφ : φ = - Real.pi / 6):
  (∀ x : ℝ, 4 * (Int.cast x : ℝ) * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 4 * (Int.cast x : ℝ) * Real.pi + 5 * Real.pi / 3) ∧ 
  (∀ x : ℝ, 4 * (Int.cast x : ℝ) * Real.pi - Real.pi ≤ x ∧ x ≤ 4 * (Int.cast x : ℝ) * Real.pi + 7 * Real.pi / 3) :=
by
  sorry

end problem_1_l87_87409


namespace odd_numbers_divisible_by_5_with_form_abcd_l87_87507

theorem odd_numbers_divisible_by_5_with_form_abcd : 
  let form_condition (a b c d : ℕ) := (a ≠ d) ∧ (a ≠ 0) ∧ (b ≠ 5) 
    ∧ (c ≠ a) ∧ (c ≠ b) ∧ (c ≠ 5)
  in 
  let valid_digits (a b c d : ℕ) := 
    {d} = {5} ∧ (1 ≤ a ∧ a ≤ 9 ∧ a ≠ 5) 
    ∧ (0 ≤ b ∧ b ≤ 9 ∧ b ≠ a ∧ b ≠ 5)
    ∧ (0 ≤ c ∧ c ≤ 9 ∧ c ≠ a ∧ c ≠ b ∧ c ≠ 5)
  in 
  (finset.univ.filter (λ n: ℕ, ∃ a b c d: ℕ, 
    form_condition a b c d ∧ valid_digits a b c d ∧ a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + d = n)).card = 448 :=
by
  sorry

end odd_numbers_divisible_by_5_with_form_abcd_l87_87507


namespace probability_x_leq_one_l87_87890

theorem probability_x_leq_one (a b : ℝ) (h1 : a = -1) (h2 : b = 4) : 
  let interval_length := b - a,
      favorable_length := 1 - a,
      probability := favorable_length / interval_length in
  probability = 2/5 := 
by
  sorry

end probability_x_leq_one_l87_87890


namespace radians_to_degrees_conversion_l87_87312

theorem radians_to_degrees_conversion : 
  let radian_to_degree := (180 / Real.pi)
  in -((23 / 12) * Real.pi) * radian_to_degree = -345 := 
  sorry

end radians_to_degrees_conversion_l87_87312


namespace gcd_pow_minus_one_l87_87729

theorem gcd_pow_minus_one (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  Nat.gcd (2^m - 1) (2^n - 1) = 2^(Nat.gcd m n) - 1 :=
by
  sorry

end gcd_pow_minus_one_l87_87729


namespace symmetric_center_of_transformed_function_l87_87038

noncomputable def transformed_function := λ x : ℝ, -cos (2 * x)

theorem symmetric_center_of_transformed_function :
  ∃ k : ℤ, transformed_function (k * (π / 4)) = 0 :=
sorry

end symmetric_center_of_transformed_function_l87_87038


namespace sequence_sum_l87_87087

variable (P Q R S T U V : ℤ)
variable (hR : R = 7)
variable (h1 : P + Q + R = 36)
variable (h2 : Q + R + S = 36)
variable (h3 : R + S + T = 36)
variable (h4 : S + T + U = 36)
variable (h5 : T + U + V = 36)

theorem sequence_sum (P Q R S T U V : ℤ)
  (hR : R = 7)
  (h1 : P + Q + R = 36)
  (h2 : Q + R + S = 36)
  (h3 : R + S + T = 36)
  (h4 : S + T + U = 36)
  (h5 : T + U + V = 36) :
  P + V = 29 := 
sorry

end sequence_sum_l87_87087


namespace train_trip_length_l87_87287

theorem train_trip_length (v D : ℝ) :
  (3 + (3 * D - 6 * v) / (2 * v) = 4 + D / v) ∧ 
  (2.5 + 120 / v + (6 * D - 12 * v - 720) / (5 * v) = 3.5 + D / v) →
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by
  sorry

end train_trip_length_l87_87287


namespace baseball_opponents_score_l87_87988

theorem baseball_opponents_score :
  let team_scores := [2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16] in
  let loss_by_one_scores := [2, 6, 8, 10, 12, 16] in
  let twice_as_many_games := [3, 5, 7, 9, 13, 15] in
  let opponent_scores_by_loss := [3, 7, 9, 11, 13, 17] in
  let opponent_scores_by_twice := [1.5, 2.5, 3.5, 4.5, 6.5, 7.5] in
  (opponent_scores_by_loss.sum + opponent_scores_by_twice.sum) = 86 :=
by
  -- structure of the proof
  sorry

end baseball_opponents_score_l87_87988


namespace condition_c_defines_right_square_prism_l87_87179

-- Definitions based on the conditions from the problem
def is_square (base : Type) := ∃ a : ℝ, ∀ p q r s : base,  dist p q = a ∧ dist q r = a ∧ dist r s = a ∧ dist s p = a
def is_rhombus (base : Type) := ∃ a : ℝ, ∀ p q r s : base,  dist p q = a ∧ dist q r = a ∧ dist r s = a ∧ dist s p = a
def mutually_perpendicular_at_vertex (V : Type) (e1 e2 e3 : V) :=
  (inner e1 e2 = 0) ∧ (inner e1 e3 = 0) ∧ (inner e2 e3 = 0)

def right_square_prism (base : Type) (V : Type) (e1 e2 e3 : V) :=
  is_square base ∧ mutually_perpendicular_at_vertex V e1 e2 e3

-- Goal: The condition "the base is a rhombus, and at one vertex, three edges are mutually perpendicular" defines a right square prism.
theorem condition_c_defines_right_square_prism (base : Type) (V : Type) (e1 e2 e3 : V) (h_rhombus : is_rhombus base)
  (h_perpendicular : mutually_perpendicular_at_vertex V e1 e2 e3) : right_square_prism base V e1 e2 e3 :=
sorry

end condition_c_defines_right_square_prism_l87_87179


namespace angle_RS_eq_4x_l87_87943

-- Representing the points, angles, and geometric properties
variables {P Q R T O S : Type*}
variables [semiring S]
variables [is_semiring_hom P Q] [is_semiring_hom R T] [is_semiring_hom O S]

-- Define the geometric configuration and conditions
variables (pq_semicircle : semicircle O P Q)
variables (pr_qt_intersect_S : intersects_at P R Q T S)
variables (angle_TOP : ℕ → ℕ) (angle_ROQ: ℕ → ℕ)
variables (x : ℕ)

-- Given conditions
axiom pq_on_semicircle : ∀ p q r t, p ∈ pq_semicircle → q ∈ pq_semicircle → r ∈ pq_semicircle → t ∈ pq_semicircle
axiom qt_pr_intersection : pr_qt_intersect_S P R Q T = S
axiom angle_TOP_IS_CONSTANT : angle_TOP (3 * x)
axiom angle_ROQ_IS_CONSTANT : angle_ROQ (5 * x)

-- Statement to prove
theorem angle_RS_eq_4x : angle_RSQ = 4 * x :=
by sorry

end angle_RS_eq_4x_l87_87943


namespace opening_price_calculation_l87_87297

variable (Closing_Price : ℝ)
variable (Percent_Increase : ℝ)
variable (Opening_Price : ℝ)

theorem opening_price_calculation
    (H1 : Closing_Price = 28)
    (H2 : Percent_Increase = 0.1200000000000001) :
    Opening_Price = Closing_Price / (1 + Percent_Increase) := by
  sorry

end opening_price_calculation_l87_87297


namespace exists_set_B_l87_87743

noncomputable def T (A : Set ℝ) : ℝ :=
  finset.sum (finset.Icc 1 (A.to_finset.card - 1))
    (λ i, finset.sum (finset.Icc (i + 1) (A.to_finset.card))
      (λ j, abs (A.to_finset.nth j - A.to_finset.nth i)))

theorem exists_set_B (A : Set ℝ) (C : ℝ) :
  ∃ B : Set ℝ, T B = T A ∧ finset.sum (B.to_finset) id = C :=
by
  sorry

end exists_set_B_l87_87743


namespace allocation_schemes_correct_l87_87342

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end allocation_schemes_correct_l87_87342


namespace total_ages_l87_87967

theorem total_ages (Xavier Yasmin : ℕ) (h1 : Xavier = 2 * Yasmin) (h2 : Xavier + 6 = 30) : Xavier + Yasmin = 36 :=
by
  sorry

end total_ages_l87_87967


namespace cos_64x_cos_65x_rational_l87_87552

noncomputable def x : ℝ := sorry

def S : ℝ := sin (64 * x) + sin (65 * x)
def C : ℝ := cos (64 * x) + cos (65 * x)

axiom H1 : Rational S
axiom H2 : Rational C

theorem cos_64x_cos_65x_rational :
  Rational (cos (64 * x)) ∧ Rational (cos (65 * x)) :=
sorry

end cos_64x_cos_65x_rational_l87_87552


namespace sum_of_digits_smallest_N_l87_87298

theorem sum_of_digits_smallest_N :
  ∃ (N : ℕ), N ≤ 999 ∧ 72 * N < 1000 ∧ (N = 13) ∧ (1 + 3 = 4) := by
  sorry

end sum_of_digits_smallest_N_l87_87298


namespace isosceles_triangle_CBD_supplement_l87_87447

/-- Given an isosceles triangle ABC with AC = BC and angle C = 50 degrees,
    and point D such that angle CBD is supplementary to angle ABC,
    prove that angle CBD is 115 degrees. -/
theorem isosceles_triangle_CBD_supplement 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (angleBAC angleABC angleC angleCBD : ℝ)
  (isosceles : AC = BC)
  (angle_C_eq : angleC = 50)
  (supplement : angleCBD = 180 - angleABC) :
  angleCBD = 115 :=
sorry

end isosceles_triangle_CBD_supplement_l87_87447


namespace find_n_tangent_l87_87334

theorem find_n_tangent (n : ℤ) (h1 : -90 < n ∧ n < 90) : tan (n * (real.pi / 180)) = tan (225 * (real.pi / 180)) ↔ n = 45 :=
by
  sorry

end find_n_tangent_l87_87334


namespace triangle_incenter_properties_l87_87668

theorem triangle_incenter_properties 
  (A B C I D: Type)
  [Incenter I (triangle ABC)] 
  [D_intersection: Intersection D (Line AI) (Circumcircle (triangle ABC)) A] : 
  DB = DC ∧ DC = DI :=
by sorry

end triangle_incenter_properties_l87_87668


namespace part_i_increasing_part_i_decreasing_part_ii_max_l87_87691

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

theorem part_i_increasing : ∀ x, 0 < x ∧ x < 1 → deriv f x > 0 := by
  sorry

theorem part_i_decreasing : ∀ x, x > 1 → deriv f x < 0 := by
  sorry

theorem part_ii_max : is_max_on f { x : ℝ | 1 / 2 ≤ x ∧ x ≤ Real.exp 1 } 1 ∧ f 1 = 0 := by
  sorry

end part_i_increasing_part_i_decreasing_part_ii_max_l87_87691


namespace sum_modified_midpoint_coordinates_l87_87680

theorem sum_modified_midpoint_coordinates :
  let p1 : (ℝ × ℝ) := (10, 3)
  let p2 : (ℝ × ℝ) := (-4, 7)
  let midpoint : (ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let modified_x := 2 * midpoint.1 
  (modified_x + midpoint.2) = 11 := by
  sorry

end sum_modified_midpoint_coordinates_l87_87680


namespace arithmetic_sequence_problem_l87_87792

noncomputable def arithmetic_sequence_sum : ℕ → ℕ := sorry  -- Define S_n here

theorem arithmetic_sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 8 - S 3 = 10)
    (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) (h3 : a 6 = 2) : S 11 = 22 :=
  sorry

end arithmetic_sequence_problem_l87_87792


namespace quadrilateral_area_l87_87524

theorem quadrilateral_area (a b c d : ℝ) (φ : ℝ) (not_perpendicular : φ ≠ π / 2):
  (area : ℝ) = (Mathlib.tan φ * |a ^ 2 + c ^ 2 - b ^ 2 - d ^ 2|) / 4 := 
sorry

end quadrilateral_area_l87_87524


namespace circumcenter_concyclic_iff_l87_87376

variables {P A B C D E F G H O₁ O₂ O₃ O₄ : Type*}
variables [metric_space P] [metric_space A] [metric_space B] [metric_space C] [metric_space D]
          [metric_space E] [metric_space F] [metric_space G] [metric_space H]
          [metric_space O₁] [metric_space O₂] [metric_space O₃] [metric_space O₄]

-- Condition 1: Quadrilateral ABCD is convex
axiom convex_ABCD : convex (set.insert A (set.insert B (set.insert C (set.insert D ∅))))

-- Condition 2: Diagonals AC and BD intersect at point P
axiom intersection P : ∃ P, segment A C ∩ segment B D = {P}

-- Condition 3: E, F, G, H are midpoints of sides AB, BC, CD, DA respectively
axiom midpoint_E : ∀ M, midpoint M A B → E = M
axiom midpoint_F : ∀ M, midpoint M B C → F = M
axiom midpoint_G : ∀ M, midpoint M C D → G = M
axiom midpoint_H : ∀ M, midpoint M D A → H = M

-- Condition 4: O₁, O₂, O₃, O₄ are circumcenters of triangles PHE, PEF, PFG, PGH respectively
axiom circumcenter_O₁ : O₁ = circumcenter P H E
axiom circumcenter_O₂ : O₂ = circumcenter P E F
axiom circumcenter_O₃ : O₃ = circumcenter P F G
axiom circumcenter_O₄ : O₄ = circumcenter P G H

-- Prove that O₁, O₂, O₃, O₄ are concyclic if and only if A, B, C, D are concyclic
theorem circumcenter_concyclic_iff : cyclic (set.insert O₁ (set.insert O₂ (set.insert O₃ (set.insert O₄ ∅)))) ↔ 
                                     cyclic (set.insert A (set.insert B (set.insert C (set.insert D ∅)))) :=
begin
  sorry
end

end circumcenter_concyclic_iff_l87_87376


namespace alpha_cubed_plus_xalpha_plus_one_eq_zero_l87_87863

noncomputable def alpha (x : ℕ → ℤ) : FormalPowerSeries ℤ ℕ :=
  1 + ∑ n in (Filter (λ n, ∀ k ∈ (0..ℕ).finset, (nat.binary_digits n).length - (nat.binary_digits n).count(0)) (Finset.range n)), 
     x n ^ n
  
theorem alpha_cubed_plus_xalpha_plus_one_eq_zero (x : ℤ) :
  alpha^3 + x * alpha + 1 = 0 := 
by
  sorry

end alpha_cubed_plus_xalpha_plus_one_eq_zero_l87_87863


namespace part_one_part_two_l87_87394

noncomputable def geometric_sequence (a_n : ℕ → ℝ) :=
  ∃ (q : ℝ) (n : ℕ → ℝ), ∀ n > 0, a_n n - 1 = n * q

axiom a_sequence (a_n : ℕ → ℝ) : a_n 1 = 5
axiom a_sequence_sum : ∀ (a_n : ℕ → ℝ), a_n 1 + a_n 2 + a_n 3 = 87

noncomputable def sum_b_sequence (b_n : ℕ → ℝ) :=
  ∃ (S_n : ℕ → ℝ), ∀ n > 0, S_n n = (1 / 2) * n * b_n (n + 1)

axiom b_sequence (b_n : ℕ → ℝ) : b_n 1 = 1

theorem part_one (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) :
  geometric_sequence (λ n, a_n n - 1) →
  a_sequence a_n → a_sequence_sum a_n →
  sum_b_sequence b_n → b_sequence b_n →
  ∀ n > 0, a_n n = 4^n + 1 ∧ b_n n = n :=
sorry

theorem part_two (a_n b_n : ℕ → ℝ) :
  ∀ n > 0, (λ (c_n : ℕ → ℝ), c_1 + c_2 + ⋯ + c_n < 4/9) (λ n, b_n n / a_n n) :=
sorry

end part_one_part_two_l87_87394


namespace investment_duration_l87_87667

noncomputable def log (x : ℝ) := Real.log x

theorem investment_duration 
  (P A : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) 
  (hP : P = 3000) 
  (hA : A = 3630) 
  (hr : r = 0.10) 
  (hn : n = 1) 
  (ht : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 :=
by
  sorry

end investment_duration_l87_87667


namespace gridiron_football_club_members_count_l87_87878

theorem gridiron_football_club_members_count :
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  total_expenditure / total_cost_per_member = 104 :=
by
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  sorry

end gridiron_football_club_members_count_l87_87878


namespace radius_of_circle_centered_at_right_focus_and_tangent_to_asymptotes_l87_87739

theorem radius_of_circle_centered_at_right_focus_and_tangent_to_asymptotes
    (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2) in
  let right_focus := (c, 0 : ℝ) in
  ∀ (asymptote : ℝ → ℝ),
    (asymptote = λ x, (b / a) * x ∨ asymptote = λ x, -(b / a) * x) →
    let dist := fun point line => Real.abs (point.fst * line.snd - point.snd * line.fst) /
                                 Real.sqrt (line.snd ^ 2 + line.fst ^ 2) in
    dist right_focus (asymptote) = b :=
sorry

end radius_of_circle_centered_at_right_focus_and_tangent_to_asymptotes_l87_87739


namespace complex_pow_example_l87_87022

noncomputable def z : ℂ := (real.sqrt 3) / 2 + ((1 / 2) * complex.I)

theorem complex_pow_example (z := (real.sqrt 3) / 2 + ((1 / 2) * complex.I)) : z ^ 2016 = 1 :=
by
  -- Detailed proof steps elided, as they are not part of the exercise prompt
  sorry

end complex_pow_example_l87_87022


namespace complete_square_l87_87957

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end complete_square_l87_87957


namespace remainder_when_3_pow_2020_div_73_l87_87588

theorem remainder_when_3_pow_2020_div_73 :
  (3^2020 % 73) = 8 := 
sorry

end remainder_when_3_pow_2020_div_73_l87_87588


namespace probability_of_red_jelly_bean_l87_87624

-- Definitions based on conditions
def total_jelly_beans := 7 + 9 + 4 + 10
def red_jelly_beans := 7

-- Statement we want to prove
theorem probability_of_red_jelly_bean : (red_jelly_beans : ℚ) / total_jelly_beans = 7 / 30 :=
by
  -- Proof here
  sorry

end probability_of_red_jelly_bean_l87_87624


namespace range_of_f_l87_87001

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)*x + 1 else -(x - 1)^2

theorem range_of_f (x : ℝ) : f(x) ≥ -1 → x ∈ Set.Icc (-4 : ℝ) (2 : ℝ) := by
  sorry

end range_of_f_l87_87001


namespace tangent_line_y_intercept_l87_87633

open Real

theorem tangent_line_y_intercept
  (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 2)
  (h_c1 : c1 = (3, 0)) (h_c2 : c2 = (8, 0)) :
  ∃ l : ℝ, l = 13/4 ∧ is_tangent (circle r1 c1) (circle r2 c2) l ∧
  (∀ p : ℝ × ℝ, p ∈ first_quadrant → is_tangent_point p r1 c1 l ∧ is_tangent_point p r2 c2 l) := 
by
  sorry

-- Helper Definitions (assuming they are defined elsewhere in Mathlib or we can define them here if necessary)
noncomputable def circle (r : ℝ) (c : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p | dist p c = r }

def is_tangent (C1 C2 : set (ℝ × ℝ)) (l : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p ∈ C1 ∧ is_on_line l p) → (p ∈ C2 ∧ is_on_line l p)

def is_tangent_point (p : ℝ × ℝ) (r : ℝ) (c : ℝ × ℝ) (l : ℝ) : Prop :=
  dist p c = r ∧ is_on_line l p

def first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_on_line (l : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = l * p.1

-- Placeholder for required definitions not provided in Mathlib or requiring customization.
-- This can involve defining specific properties for tangency, distances, etc., that capture the geometry of the circles and lines.

end tangent_line_y_intercept_l87_87633


namespace cubic_root_equality_l87_87310

theorem cubic_root_equality (a b c : ℝ) (h1 : a + b + c = 12) (h2 : a * b + b * c + c * a = 14) (h3 : a * b * c = -3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 268 / 9 := 
by
  sorry

end cubic_root_equality_l87_87310


namespace exists_fewer_than_2_pow_m_b_integers_l87_87495

theorem exists_fewer_than_2_pow_m_b_integers
  (m : ℕ) (h_m_pos : 0 < m) 
  (a : fin m → ℕ) 
  (ha_pos : ∀ i, 0 < a i) : 
  ∃ (s : finset ℕ), s.card < 2^m ∧ ∀ i, a i ∈ s ∧ (∀ (t₁ t₂ : finset ℕ), t₁ ≠ t₂ → (∑ x in t₁, x) ≠ (∑ x in t₂, x)) :=
sorry

end exists_fewer_than_2_pow_m_b_integers_l87_87495


namespace point_A_in_second_quadrant_l87_87100

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l87_87100


namespace min_value_of_a_plus_b_l87_87793

theorem min_value_of_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a (3 * b) = -1) : a + b = (2 * Real.sqrt 3) / 3 :=
sorry

end min_value_of_a_plus_b_l87_87793


namespace find_y_l87_87752

def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

theorem find_y (y : ℝ) : slope (-3) 8 5 y = -5 / 4 → y = -2 := by
  intros h
  unfold slope at h
  sorry

end find_y_l87_87752


namespace toby_total_time_l87_87215

def speed_unloaded := 20 -- Speed of Toby pulling unloaded sled in mph
def speed_loaded := 10   -- Speed of Toby pulling loaded sled in mph

def distance_part1 := 180 -- Distance for the first part (loaded sled) in miles
def distance_part2 := 120 -- Distance for the second part (unloaded sled) in miles
def distance_part3 := 80  -- Distance for the third part (loaded sled) in miles
def distance_part4 := 140 -- Distance for the fourth part (unloaded sled) in miles

def time_part1 := distance_part1 / speed_loaded -- Time for the first part in hours
def time_part2 := distance_part2 / speed_unloaded -- Time for the second part in hours
def time_part3 := distance_part3 / speed_loaded -- Time for the third part in hours
def time_part4 := distance_part4 / speed_unloaded -- Time for the fourth part in hours

def total_time := time_part1 + time_part2 + time_part3 + time_part4 -- Total time in hours

theorem toby_total_time : total_time = 39 :=
by 
  sorry

end toby_total_time_l87_87215


namespace range_of_a_for_function_positivity_l87_87777

noncomputable def f (a x : ℝ) := log a (a * x^2 - x + 1/2)

theorem range_of_a_for_function_positivity (a : ℝ) :
  (∀ x ∈ set.Icc 1 2, f a x > 0) ↔ 
  (a ∈ set.Ioo (1/2) (5/8) ∪ set.Ioi (3/2)) :=
by
  sorry

end range_of_a_for_function_positivity_l87_87777


namespace number_of_odd_numbers_divisible_by_5_l87_87505

theorem number_of_odd_numbers_divisible_by_5 :
  let D := { d | d = 5 }
  let A := { a | a ∈ {1, 2, 3, 4, 6, 7, 8, 9} }
  let B := { b | b ∈ {0, 1, 2, 3, 4, 6, 7, 8, 9} }
  let C := { c | c ∈ {0, 1, 2, 3, 4, 6, 7, 8, 9} }
  ∀ a ∈ A, b ∈ B, c ∈ C, a ≠ b ∧ a ≠ c ∧ b ≠ c → 
  ∃ (count : ℕ), 
  count = 1 * 8 * 8 * 7 ∧ 
  count = 448 := 
by 
  sorry

end number_of_odd_numbers_divisible_by_5_l87_87505


namespace twelfth_sequence_value_l87_87141

theorem twelfth_sequence_value : 
  let sequence_value (n : ℕ) := (↑n / (n^2 + 1) : ℚ) * (-1 : ℚ)^(n + 1) 
  in sequence_value 12 = -12 / 145 := 
by
  sorry

end twelfth_sequence_value_l87_87141


namespace max_value_f_l87_87713

open Real

noncomputable def f (x : ℝ) := x * (1 - 2 * x)

theorem max_value_f : ∃ m : ℝ, (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f x ≤ m) ∧ (∃ x : ℝ, 0 < x ∧ x < (1 / 2) ∧ f x = m) :=
by
  unfold f
  -- Detailed proof with relevant approach goes here
  sorry

end max_value_f_l87_87713


namespace minimum_value_real_l87_87231

theorem minimum_value_real (x : ℝ) : ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  use -9,
  sorry

end minimum_value_real_l87_87231


namespace sum_of_smallest_alpha_l87_87692

def Q (x : ℂ) : ℂ := ( (x^22 - 1)/(x - 1) )^2 - x^21

-- Property of the complex zeros of the polynomial Q and their angles
theorem sum_of_smallest_alpha (z k r_k α_k : ℝ) (h₀ : 0 < α₁ ∧ ...(and so on) ... ∧ α₄₂ < 1) 
(h₁ : α₁ ≤ α₂ ∧ ...(and so on) ... ∧ α₄₁ ≤ α₄₂)
(h₂ : r_k > 0) 
(hz : k = 1 ∨ ... ∨ k = 42)
(halpha : z_k = r_k * (cos (2*π*α_k) + complex.I * sin (2*π*α_k)))
: α₁ + α₂ + α₃ + α₄ + α₅ = 191/483 :=
sorry

end sum_of_smallest_alpha_l87_87692


namespace range_of_expression_l87_87778

-- Define the function f
def f (x b c d : ℝ) := (1/3) * x^3 + (1/2) * b * x^2 + c * x + d

-- Assumptions for the interval and the conditions
variable (c b : ℝ)
axiom has_local_extrema : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ (x^2 + b * x + c) = 0

-- Main goal to be proved
theorem range_of_expression : 0 < c^2 + 2 * b * c + 4 * c ∧ c^2 + 2 * b * c + 4 * c < 1 :=
by 
  sorry

end range_of_expression_l87_87778


namespace num_common_divisors_of_9240_and_10010_l87_87437

def prime_factors_9240 := {2^3, 3, 5, 7, 11}
def prime_factors_10010 := {2, 3, 5, 7, 11, 13}

theorem num_common_divisors_of_9240_and_10010 : 
  let gcd_9240_10010 := 2310 in
  (∏ p in (finset.filter prime (finset.range 14)), nat.divisors p).card = 32 :=
by
  sorry

end num_common_divisors_of_9240_and_10010_l87_87437


namespace interval_of_defined_expression_l87_87354

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l87_87354


namespace point_O_is_circumcenter_l87_87520

theorem point_O_is_circumcenter (P A B C O : Type) [Point P] [Point A] [Point B] [Point C] [Point O]
  (plane_ABC : Plane) (P_outside_plane : P ∉ plane_ABC)
  (PO_perpendicular_to_plane : perpendicular (line P O) plane_ABC)
  (foot_of_perpendicular : foot_perpendicular P plane_ABC = O)
  (PA_eq_PB : distance P A = distance P B)
  (PB_eq_PC : distance P B = distance P C) :
  circumcenter O A B C := sorry

end point_O_is_circumcenter_l87_87520


namespace log_over_sqrt_defined_l87_87357

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l87_87357


namespace min_value_of_x_squared_plus_6x_l87_87234

theorem min_value_of_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y = x^2 + 6 * x ∧ ∀ z : ℝ, z = x^2 + 6 * x → y ≤ z :=
begin
  sorry
end

end min_value_of_x_squared_plus_6x_l87_87234


namespace average_t_value_is_15_l87_87027

noncomputable def average_of_distinct_t_values (t_vals : List ℤ) : ℤ :=
t_vals.sum / t_vals.length

theorem average_t_value_is_15 :
  average_of_distinct_t_values [8, 14, 18, 20] = 15 :=
by
  sorry

end average_t_value_is_15_l87_87027


namespace knight_diagonal_marking_l87_87145

noncomputable theory -- if necessary

def is_valid_knight_tour_checkerboard (n : ℕ) : Prop :=
  ∀ knight_tour : List (ℕ × ℕ), 
    knight_tour.head = (1, 1) ∧ knight_tour.last = (n, n) →
    (∃ (i : ℕ), i < knight_tour.length ∧ marked_cell knight_tour[i])

def has_marked_diagonals (n : ℕ) : Prop :=
  ∃ (i j : ℕ), 
    1 ≤ i ∧ i ≤ n ∧ 
    1 ≤ j ∧ j ≤ n ∧ 
    (i + 1, j + 1) ∈ marked ∧ 
    (i + 2, j + 2) ∈ marked ∧ 
    ((i + 1, j + 1) ∈ diagonal_cells ∨ (i + 2, j + 2) ∈ diagonal_cells)

def problem (n : ℕ) : Prop := 
  n > 3 ∧ is_valid_knight_tour_checkerboard n → has_marked_diagonals n

theorem knight_diagonal_marking (n : ℕ) : 
  n ∈ (3 * ℕ + 1) ↔ problem n :=
sorry

end knight_diagonal_marking_l87_87145


namespace modulus_z_l87_87896

noncomputable def complex_modulus (z : ℂ) : ℝ := complex.abs z

theorem modulus_z (w z : ℂ) (h1 : w * z = 15 - 20 * complex.I) (h2 : complex_modulus w = real.sqrt 13) :
  complex_modulus z = (25 * real.sqrt 13) / 13 := 
by 
  sorry

end modulus_z_l87_87896


namespace arrangement_total_l87_87169

-- Definition of people and positions
inductive Person
| A | B | C | D | E | F

open Person

def total_arrangements : ℕ :=
  let people := [A, B, C, D, E, F]
  let without_A := [B, C, D, E, F]
  -- Number of arrangements when the person at the far left is A
  let left_A := (5.factorial)
  -- Number of arrangements when the person at the far left is B
  let right_without_A := 4 * (4.factorial)
  -- Total arrangements
  left_A + right_without_A

theorem arrangement_total : total_arrangements = 216 := 
by
  sorry

end arrangement_total_l87_87169


namespace james_muffins_baked_l87_87260

-- Define the number of muffins Arthur baked
def muffinsArthur : ℕ := 115

-- Define the multiplication factor
def multiplicationFactor : ℕ := 12

-- Define the number of muffins James baked
def muffinsJames : ℕ := muffinsArthur * multiplicationFactor

-- The theorem that needs to be proved
theorem james_muffins_baked : muffinsJames = 1380 :=
by
  sorry

end james_muffins_baked_l87_87260


namespace range_of_abs_z_l87_87736

open Complex Real

noncomputable def z (t : ℝ) : ℂ :=
  (sin t / sqrt 2 + complex.I * cos t) / (sin t - complex.I * (cos t / sqrt 2))

theorem range_of_abs_z :
  ∀ t : ℝ, 1 / sqrt 2 ≤ abs (z t) ∧ abs (z t) ≤ sqrt 2 :=
by
  intro t
  sorry

end range_of_abs_z_l87_87736


namespace minimal_period_of_sum_l87_87152

theorem minimal_period_of_sum (a b : ℚ) (hA : ∃ p q : ℕ, q ≠ 0 ∧ a = p / (10^30 - 1) ∧ gcd p q = 1 ∧ period q = 30)
 (hB : ∃ r s : ℕ, s ≠ 0 ∧ b = r / (10^30 - 1) ∧ gcd r s = 1 ∧ period s = 30)
 (hAB : ∃ p q : ℕ, q ≠ 0 ∧ (a - b) = p / (q - 1) ∧ q = 10^15 ∧ period q = 15)
: ∃ n m : ℕ, m ≠ 0 ∧ (a + 6 * b) = n / (10^15 - 1) ∧ period (10^15 - 1) = 15 := 
by { sorry }

end minimal_period_of_sum_l87_87152


namespace triangle_angles_condition_l87_87450

theorem triangle_angles_condition (A B C : ℝ) (a b hc : ℝ) (h : sin A ^ 2 + sin B ^ 2 = 1 ∧ 1 / a ^ 2 + 1 / b ^ 2 = 1 / hc ^ 2) :
  C = Real.pi / 2 ∨ A - B = Real.pi / 2 ∨ B - A = Real.pi / 2 :=
by
  sorry

end triangle_angles_condition_l87_87450


namespace triple_composition_even_l87_87121

variable (g : ℝ → ℝ)

-- Definition: g is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem: g(g(g(x))) is even if g is even
theorem triple_composition_even (h : is_even g) : is_even (λ x, g (g (g x))) :=
sorry

end triple_composition_even_l87_87121


namespace bags_needed_l87_87897

-- Definitions for the condition
def total_sugar : ℝ := 35.5
def bag_capacity : ℝ := 0.5

-- Theorem statement to solve the problem
theorem bags_needed : total_sugar / bag_capacity = 71 := 
by 
  sorry

end bags_needed_l87_87897


namespace sum_of_angles_l87_87557

def P (x : ℝ) : ℝ := x^3 + real.sqrt 6 * x^2 - real.sqrt 2 * x - real.sqrt 3

theorem sum_of_angles (θs : List ℝ) 
  (hθ : ∀ θ ∈ θs, 0 ≤ θ ∧ θ < 360 ∧ P (real.tan (θ * real.pi / 180)) = 0) 
  (h_distinct : θs.nodup) : θs.sum = 660 :=
sorry

end sum_of_angles_l87_87557


namespace count_numbers_as_diff_two_primes_with_one_3_l87_87433

-- Define the given set based on the arithmetic progression
def given_set : Set ℕ := {n | ∃ k : ℕ, n = 5 + 12 * k ∧ n ≤ 197}

-- Define the property we are interested in: can be written as the difference of two prime numbers, where one is 3
def can_be_written_as_diff_of_prime_and_3 (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = p - 3

-- Main theorem:
theorem count_numbers_as_diff_two_primes_with_one_3 :
  (given_set.filter can_be_written_as_diff_of_prime_and_3).card = 0 :=
sorry

end count_numbers_as_diff_two_primes_with_one_3_l87_87433


namespace sum_of_perimeters_of_squares_l87_87198

theorem sum_of_perimeters_of_squares (x y : ℕ)
  (h1 : x^2 - y^2 = 19) : 4 * x + 4 * y = 76 := 
by
  sorry

end sum_of_perimeters_of_squares_l87_87198


namespace distinct_arrangements_l87_87789

-- Define the conditions: 7 books, 3 are identical
def total_books : ℕ := 7
def identical_books : ℕ := 3

-- Statement that the number of distinct arrangements is 840
theorem distinct_arrangements : (Nat.factorial total_books) / (Nat.factorial identical_books) = 840 := 
by
  sorry

end distinct_arrangements_l87_87789


namespace arithmetic_sequence_sum_l87_87380

variables {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_sum (S T R : α) (n : ℕ) (a : ℕ → α)
  (hS : S = ∑ i in range n, a i)
  (hT : T = ∑ i in range (2 * n), a i)
  (hR : R = ∑ i in range (3 * n), a i) :
  R = 3 * (T - S) :=
sorry

end arithmetic_sequence_sum_l87_87380


namespace tangent_slope_l87_87223

theorem tangent_slope
  (center : ℝ × ℝ)
  (point : ℝ × ℝ)
  (h_center : center = (1, -4))
  (h_point : point = (8, 3)) :
  let radius_slope := (point.2 - center.2) / (point.1 - center.1)
  (radius_slope = 1) →
  ∃ tangent_slope : ℝ, tangent_slope = -1 :=
by
  intros
  have radius_slope : ℝ := (8 - 1) / (8 - 1)
  have tangent_slope : ℝ := -1
  exact ⟨tangent_slope, rfl⟩

end tangent_slope_l87_87223


namespace smallest_positive_period_l87_87391

theorem smallest_positive_period (R : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = sqrt 3 * sin (π * x / R))
  (h_max_min : (∃ x_max y_max x_min y_min, 
    (y_max = sqrt 3 ∧ y_min = -sqrt 3) ∧ 
    ((x_max^2 + y_max^2 = R^2) ∧ (x_min^2 + y_min^2 = R^2)) ∧ 
    (exists diff, x_max = x_min + diff ∧ diff = R / 2))) :
  ∃ T, T = 4 := 
by
  sorry

end smallest_positive_period_l87_87391


namespace count_complex_numbers_l87_87059

theorem count_complex_numbers (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + b ≤ 5) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end count_complex_numbers_l87_87059


namespace number_of_participants_with_5_points_l87_87817

-- Definitions for conditions
def num_participants : ℕ := 254

def points_for_victory : ℕ := 1

def additional_point_condition (winner_points loser_points : ℕ) : ℕ :=
  if winner_points < loser_points then 1 else 0

def points_for_loss : ℕ := 0

-- Theorem statement
theorem number_of_participants_with_5_points :
  ∃ num_students_with_5_points : ℕ, num_students_with_5_points = 56 := 
sorry

end number_of_participants_with_5_points_l87_87817


namespace arithmetic_sequence_S_15_l87_87831

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ}

theorem arithmetic_sequence_S_15 :
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 1 + a 15 = 2 * a 8) →
  (a 4 + a 12 = 2 * a 8) →
  S 15 a = -30 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S_15_l87_87831


namespace interval_of_defined_expression_l87_87353

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l87_87353


namespace find_angle_C_l87_87468

noncomputable def angles (A B C : ℝ) : Prop := A + B + C = 180

theorem find_angle_C 
  (A B C : ℝ)
  (h_triangle : angles A B C)
  (h1 : 5 * sin A + 2 * cos B = 5)
  (h2 : 2 * sin B + 5 * cos A = 2) : 
  C = 180 := 
by
  sorry

end find_angle_C_l87_87468


namespace find_x_plus_y_l87_87428

-- Define the vectors and what it means for vectors to be parallel
def a : ℝ × ℝ × ℝ := (-1, 2, 1)
variables (x y : ℝ)
def b : ℝ × ℝ × ℝ := (3, x, y)

def are_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ w = (k * v.1, k * v.2, k * v.3)

-- Assume the condition that vectors a and b are parallel
theorem find_x_plus_y (h_parallel : are_parallel a (3, x, y)) : x + y = -9 :=
sorry

end find_x_plus_y_l87_87428


namespace piravena_trip_total_cost_l87_87884

-- Define the distances
def d_A_to_B : ℕ := 4000
def d_B_to_C : ℕ := 3000

-- Define the costs per kilometer
def bus_cost_per_km : ℝ := 0.15
def airplane_cost_per_km : ℝ := 0.12
def airplane_booking_fee : ℝ := 120

-- Define the individual costs and the total cost
def cost_A_to_B : ℝ := d_A_to_B * airplane_cost_per_km + airplane_booking_fee
def cost_B_to_C : ℝ := d_B_to_C * bus_cost_per_km
def total_cost : ℝ := cost_A_to_B + cost_B_to_C

-- Define the theorem we want to prove
theorem piravena_trip_total_cost :
  total_cost = 1050 := sorry

end piravena_trip_total_cost_l87_87884


namespace smallest_angle_of_quadrilateral_in_ratio_l87_87920

theorem smallest_angle_of_quadrilateral_in_ratio (k : ℚ) :
  let angles := [4 * k, 5 * k, 6 * k, 7 * k] in
  (angles.sum = 360) →
  4 * k = 720 / 11 :=
by
  intros
  sorry

end smallest_angle_of_quadrilateral_in_ratio_l87_87920


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_l87_87712

-- Problem 1
theorem problem1 (n : ℕ) (a : ℕ → ℝ) (h0 : a 1 = 2) (h : ∀ n, a (n + 1) = a n + real.log (1 + 1 / n)) : a n = 2 + real.log n :=
sorry

-- Problem 2
theorem problem2 (n : ℕ) (a : ℕ → ℤ) (h0 : a 1 = 5) (h : ∀ n, a (n + 1) = 2 * a n + 2 ^ (n + 1) - 1) : a n = (n + 1) * 2 ^ n + 1 :=
sorry

-- Problem 3
theorem problem3 (n : ℕ → ℤ) (a : ℤ) (h : a n = 2 * a n + 4 ^ n + 2) : a n = -4 ^ n - 2 :=
sorry

-- Problem 4
theorem problem4 (n : ℕ → ℝ) (a : nn) (h0 : a 1 = 1) (h : ∀ n, (n + 1) * a (n + 1) ^ 2 - n * a n ^ 2 + a (n + 1) * a n = 0) : a n = 1 / n :=
sorry

-- Problem 5
theorem problem5 (n : ℕ → ℝ) (a : ℝ) (h0 : a 1 = 1) (h : ∀ n, n * a n = ∑ i in finset.range (n - 1) + 1, i * a i - 1) : a n = 2 ^ (n - 1) / n :=
sorry

-- Problem 6
theorem problem6 (n : ℕ) (a : ℕ → ℚ) (h0 : a 1 = 1) (h : ∀ n, a (n + 1) = a n / (-7 * a n - 6)) : a n = 1 / (2 * (-6) ^ (n - 1) - 1) :=
sorry

-- Problem 7
theorem problem7 (n : ℕ) (a : ℕ → ℕ) (h0 : a 1 = 1) (h : ∀ n, a (n + 1) = (a n) ^ 2 + 2 * a n) : a n = 2 ^ (2 ^ (n - 1)) - 1 :=
sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_l87_87712


namespace vending_machine_failure_rate_correct_l87_87202

noncomputable def vending_machine_failure_rate 
  (total_attempts : ℕ) (extra_snack_rate : ℚ) (total_dropped_snacks : ℕ) 
  (expected_snacks_if_no_failure : ℕ) : ℚ :=
  let extra_snacks := total_attempts * extra_snack_rate in
  let expected_snacks := total_attempts + extra_snacks.to_nat in
  let failures := expected_snacks - total_dropped_snacks in
  failures / total_attempts

theorem vending_machine_failure_rate_correct :
  vending_machine_failure_rate 30 (1 / 10) 28 33 = 1 / 6 := by 
sorry

end vending_machine_failure_rate_correct_l87_87202


namespace radius_of_sphere_in_truncated_cone_l87_87660

theorem radius_of_sphere_in_truncated_cone :
  ∀ (r1 r2 : ℝ) (h : ℝ), 
  r1 = 20 → r2 = 5 →
  0 < h → -- Assuming positive height to ensure cone is well-defined
  ∃ (r_sphere : ℝ), 
  r_sphere = 10 :=
by
  assume r1 r2 h hr1 hr2 hh,
  use 10,
  sorry

end radius_of_sphere_in_truncated_cone_l87_87660


namespace compute_complex_expression_l87_87498

def A : ℂ := 3 + 2 * complex.I
def B : ℂ := -1 - 2 * complex.I
def C : ℂ := 5 * complex.I
def D : ℂ := 3 + complex.I

theorem compute_complex_expression : 2 * (A - B + C + D) = 8 + 20 * complex.I := by
  sorry

end compute_complex_expression_l87_87498


namespace proposition_1_proposition_2_proposition_3_l87_87773

def f (x : ℝ) := x^2 * Real.sin x

def sequence_condition (x : ℕ → ℝ) := ∀ i : ℕ, |x i| ≤ Real.pi / 2

def F (x : ℕ → ℝ) (n : ℕ) :=
  (Finset.range n).sum (λ i, x i) *
  (Finset.range n).sum (λ i, f (x i))

theorem proposition_1 (x : ℕ → ℝ) (h1 : sequence_condition x) (h2 : ∃ i, x i ≠ 0 ∧ ∀ j, j ≠ i → x j = 0):
  ∃ n, n ≥ 3 ∧ F x n = 0 :=
sorry

theorem proposition_2 : ∀ k : ℕ, k > 0 →
  let x := λ n, (-1/2)^n in F x (2*k) > 0 :=
sorry

theorem proposition_3 (x : ℕ → ℝ) (h1 : sequence_condition x) (h2 : ∀ n, x n = x 0 + n * (x 1 - x 0)) :
  ∀ n, n > 0 → F x n ≥ 0 :=
sorry

end proposition_1_proposition_2_proposition_3_l87_87773


namespace nth_number_in_set_s_l87_87872

-- Define the set of all elements that when divided by 8 have a remainder of 5
def s : Set ℤ := { x | ∃ k, x = 8 * k + 5 }

-- Define the proposition we want to prove
theorem nth_number_in_set_s (n : ℕ) (h : 557 ∈ s) : n = 70 :=
by {
  -- Define the specific member of the set s
  have k : ∃ k, 557 = 8 * k + 5 := h,
  -- Extract k
  rcases k with ⟨k, hk⟩,
  -- Show k = 69
  have k_val : k = 69 := by linarith,
  -- Since such numbers are indexed by k, the n-th element corresponding to 557 is k + 1
  rw k_val, 
  exact rfl }

end nth_number_in_set_s_l87_87872


namespace sum_infinite_geo_series_l87_87071

theorem sum_infinite_geo_series (x : ℝ) (h : (x * sqrt(x) - 1 / x) ^ 6 = 15 / 2) :
  tendsto (λ n, ∑ k in range n, x^(-k)) at_top (𝓝 1) :=
by
  sorry

end sum_infinite_geo_series_l87_87071


namespace circle_angle_l87_87554

theorem circle_angle {
  O A B C : Point  -- defining necessary points
  (hO_center : is_center O (circumscribed_circle (triangle A B C)))  -- O is the center of the circumscribed circle
  (h_angle_BOC : measure_angle B O C = 120)  -- ∠BOC = 120 degrees
  (h_angle_AOB : measure_angle A O B = 150) -- ∠AOB = 150 degrees
} :
  measure_angle A B C = 45 :=  -- Proving that ∠ABC = 45 degrees
sorry

end circle_angle_l87_87554


namespace hemisphere_surface_area_l87_87903

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l87_87903


namespace solve_equation_l87_87160

theorem solve_equation :
  ∃ x : ℝ, x = (Real.sqrt (x - 1/x)) + (Real.sqrt (1 - 1/x)) ∧ x = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end solve_equation_l87_87160


namespace mabel_transactions_l87_87881

/--
On Thursday, Mabel handled some transactions. 
Anthony handled 10% more transactions than Mabel,
Cal handled 2/3rds of the transactions that Anthony handled,
and Jade handled 17 more transactions than Cal. 
Jade handled 83 transactions.
How many transactions did Mabel handle?
-/
theorem mabel_transactions :
  ∃ (M : ℝ), 
    (∃ (A : ℝ), A = 1.10 * M) ∧
    (∃ (C : ℝ), C = (2 / 3) * A) ∧
    (∃ (J : ℝ), J = C + 17) ∧
    J = 83 ∧
    M = 90 :=
begin
  sorry
end

end mabel_transactions_l87_87881


namespace smallest_n_l87_87526

theorem smallest_n (n : ℕ) (h1 : n % 6 = 5) (h2 : n % 7 = 4) (h3 : n > 20) : n = 53 :=
sorry

end smallest_n_l87_87526


namespace parking_spaces_front_l87_87194

theorem parking_spaces_front (S B C A F : ℕ) (h1 : B = 38) 
(h2 : C = 39) (h3 : A = 32) 
(h4 : S = B / 2) (h5 : A = 32) : F = 33 :=
by 
  have h6 : S = 19 := by
    rw [h4, h1]
    norm_num
  
  have h7 : F = (C - S) + (A - S) := by
    rw [h3, h4, h6]
    norm_num
   
  sorry

end parking_spaces_front_l87_87194


namespace max_donation_extra_pay_rate_l87_87511

noncomputable def maria_ivanovna :: MariaIvanovna := sorry

theorem max_donation (cost_per_hour: ℕ) 
(investment_income: ℕ) 
(working_days: ℕ)
(sleep_hours: ℕ)
(monthly_expenses: ℕ)
(total_daily_hours: ℕ)
(max_donation : (daily_hours_for_lessons * working_days)) : ℕ :=
  let daily_hours_for_lessons := sorry in
  let daily_hours_for_knitting := 2 * daily_hours_for_lessons in
  let daily_hours_for_rest := 15 - 3 * daily_hours_for_lessons in
  let max_donation := (5 - daily_hours_for_lessons) * working_days in
  max_donation

theorem extra_pay_rate (current_rate : ℕ) : ℕ := 
  let extra_hour_rate := 2 * current_rate in
  extra_hour_rate

end max_donation_extra_pay_rate_l87_87511


namespace largest_integer_x_sq_div_50_l87_87218

theorem largest_integer_x_sq_div_50 (b : ℝ) (h : ℝ) (h1 : ℝ) (x : ℝ)
  (h_diff_bases : ∀ (b1 b2 : ℝ), b2 = b1 + 50)
  (h_area_ratio : 3 / 5 = (1 / 2 * (h / 2) * (b + (b + 25))) / (1 / 2 * (h / 2) * ((b + 25) + (b + 50)))) :
  (x ∈ {60, 15} → ⌊x^2 / 50⌋ = 72) :=
by
  sorry

end largest_integer_x_sq_div_50_l87_87218


namespace mean_of_two_numbers_l87_87904

theorem mean_of_two_numbers (a b : ℝ) (mean_twelve : ℝ) (mean_fourteen : ℝ) 
  (h1 : mean_twelve = 60) 
  (h2 : mean_fourteen = 75) 
  (sum_twelve : 12 * mean_twelve = 720) 
  (sum_fourteen : 14 * mean_fourteen = 1050) 
  : (a + b) / 2 = 165 :=
by
  sorry

end mean_of_two_numbers_l87_87904


namespace arithmetic_seq_equal_terms_l87_87496

variable {n : ℕ} (a b : Fin n → ℕ) 

theorem arithmetic_seq_equal_terms
  (h_n : n ≥ 2018)
  (h_distinct_a : Function.Injective a)
  (h_distinct_b : Function.Injective b)
  (h_bound_a : ∀ i, a i > 0 ∧ a i ≤ 5 * n)
  (h_bound_b : ∀ i, b i > 0 ∧ b i ≤ 5 * n)
  (h_arith_seq : ∃ d : ℚ, ∀ i : Fin (n-1), (a (i+1) : ℚ) / b (i+1) - (a i) / b i = d) :
  ∀ i j : Fin n, (a i : ℚ) / b i = (a j) / b j := 
by sorry

end arithmetic_seq_equal_terms_l87_87496


namespace problem1_problem2_l87_87373

-- Problem (1)
theorem problem1 (a : ℝ) (h : a = 1) (p q : ℝ → Prop) 
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0) 
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1) :
  (∀ x, (p x ∧ q x) ↔ (2 < x ∧ x < 3)) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℝ) (p q : ℝ → Prop)
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1)
  (hnpc : ∀ x, ¬p x → ¬q x) 
  (hnpc_not_necessary : ∃ x, ¬p x ∧ q x) :
  (4 / 3 ≤ a ∧ a ≤ 2) :=
by sorry

end problem1_problem2_l87_87373


namespace hyperbola_asymptotes_l87_87765

def asymptotes_of_hyperbola (a b : ℝ) (ha: a > 0) (hb: b > 0) (h: a > b) (eccentricity: a > b > 0 ∧ a^2 > b^2 ∧ (a^2 - b^2) / a^2 = 3/4)
: Prop := 
  ∀ x y : ℝ, (y = 2 * x ∨ y = -2 * x)

theorem hyperbola_asymptotes 
  (a b : ℝ) (ha: a > 0) (hb: b > 0) (h: a > b) 
  (eccentricity: (a > b ∧ a > 0 ∧ b > 0) ∧ ((a^2 - b^2)/a^2 = 3/4)) :
  asymptotes_of_hyperbola a b ha hb h eccentricity := 
begin
  sorry
end

end hyperbola_asymptotes_l87_87765


namespace total_weight_of_arrangement_l87_87285

def original_side_length : ℤ := 4
def original_weight : ℤ := 16
def larger_side_length : ℤ := 10

theorem total_weight_of_arrangement :
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  total_weight = 96 :=
by
  let original_area := original_side_length ^ 2
  let larger_area := larger_side_length ^ 2
  let number_of_pieces := (larger_area / original_area : ℤ)
  let total_weight := (number_of_pieces * original_weight)
  sorry

end total_weight_of_arrangement_l87_87285


namespace sin_alpha_terminal_side_l87_87028

theorem sin_alpha_terminal_side (α : ℝ) : (∃ (a : ℝ), (3 * a, a) ∈ {p : ℝ × ℝ | p.2 = (1 / 3) * p.1}) → (Real.sin α = ± (Real.sqrt 10 / 10)) :=
by
  intro h
  sorry

end sin_alpha_terminal_side_l87_87028


namespace rectangle_ABCD_area_l87_87162

def rectangle_area (x : ℕ) : ℕ :=
  let side_lengths := [x, x+1, x+2, x+3];
  let width := side_lengths.sum;
  let height := width - x;
  width * height

theorem rectangle_ABCD_area : rectangle_area 1 = 143 :=
by
  sorry

end rectangle_ABCD_area_l87_87162


namespace min_value_sin_cos_expression_l87_87717

theorem min_value_sin_cos_expression : ∀ x : ℝ, 
  ∃ y : ℝ, y = (9 / 10) ∧ (y = infi (fun x => (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1)) :=
begin
  sorry
end

end min_value_sin_cos_expression_l87_87717


namespace difference_in_areas_right_angle_difference_in_areas_general_l87_87517

variable {ABC : Type} [inst : Triangle ABC]
variable (A B C M N P Q K L : Point ABC)
variable (d : Real)

-- Conditions
axiom ABMN_sq : is_square (A, B, M, N)
axiom BCKL_sq : is_square (B, C, K, L)
axiom ACPQ_sq : is_square (A, C, P, Q)
axiom NQZT_sq : is_square (N, Q, Z, T)
axiom PKXY_sq : is_square (P, K, X, Y)
axiom diff_ABMN_BCKL : area (ABMN_sq) - area (BCKL_sq) = d

-- Case (a): Right-angle at B
theorem difference_in_areas_right_angle (h : ∠ABC = 90°) : 
    area(NQZT_sq) - area(PKXY_sq) = 3 * d := 
sorry

-- Case (b): General case
theorem difference_in_areas_general :
    area(NQZT_sq) - area(PKXY_sq) = 3 * d := 
sorry

end difference_in_areas_right_angle_difference_in_areas_general_l87_87517


namespace trig_identity_through_point_l87_87761

theorem trig_identity_through_point (x y r : ℝ) (h : x = -4) (k : y = 3) (r_pos : r = real.sqrt (x^2 + y^2)) :
  real.sin (real.atan2 y x) + real.cos (real.atan2 y x) = -1 / 5 :=
by
  have h₁ : x = -4 := h
  have h₂ : y = 3 := k
  have r : r = real.sqrt ((-4)^2 + 3^2) := r_pos
  have r_positive : r = 5 := by sorry
  have s := real.sin (real.atan2 y x)
  have c := real.cos (real.atan2 y x)
  have s_def : s = y / r := by sorry
  have c_def : c = x / r := by sorry
  show s + c = -1 / 5 := by sorry

end trig_identity_through_point_l87_87761


namespace find_yz_circle_radius_l87_87658

-- Define the center of the sphere derived from the problem conditions.
def sphere_center := (3 : ℝ, 5, -8)

-- Define the center of the circle on the xy-plane.
def xy_circle_center := (3 : ℝ, 5, 0)

-- Define the center of the circle on the yz-plane.
def yz_circle_center := (0 : ℝ, 5, -8)

-- Define the radius of the circle on the xy-plane.
def xy_circle_radius : ℝ := 3

-- Define the radius of the sphere based on the xy-plane circle.
def sphere_radius : ℝ := dist sphere_center xy_circle_center

-- Define the distance between the centers of the sphere and the yz-plane circle.
def distance_between_centers : ℝ := dist sphere_center yz_circle_center

-- State the radius of the circle on the yz-plane we need to prove.
def yz_circle_radius : ℝ := sqrt (sphere_radius ^ 2 - distance_between_centers ^ 2)

-- The theorem we need to prove.
theorem find_yz_circle_radius : yz_circle_radius = sqrt 55 :=
by
  -- We state that this is the answer based on given conditions and leave the proof to be filled in.
  sorry

end find_yz_circle_radius_l87_87658


namespace pairwise_relatively_prime_numbers_can_be_made_coprime_l87_87265

def gcd (a b : ℕ) : ℕ := nat.gcd a b

def all_pairwise_coprime (numbers : list ℕ) : Prop :=
  ∀ (i j : ℕ), i < numbers.length → j < numbers.length → i ≠ j → gcd (numbers.nth_le i sorry) (numbers.nth_le j sorry) = 1

def can_be_made_pairwise_coprime (numbers : list ℕ) : Prop :=
  ∀ k : ℕ, k ≤ numbers.length → ∃ numbers' : list ℕ, all_pairwise_coprime numbers' ∧ 
    ∀ i : ℕ, ∃ d : ℕ, 
    (i + 1 < numbers'.length → numbers'.nth_le i sorry + d = numbers'.nth_le (i + 1) sorry) ∧
    (i = numbers'.length - 1 → numbers'.nth_le i sorry + d = numbers'.nth_le 0 sorry)

theorem pairwise_relatively_prime_numbers_can_be_made_coprime (numbers : list ℕ) (h_len : numbers.length = 100)
  (h_rel_prime : all_pairwise_coprime numbers) :
  can_be_made_pairwise_coprime numbers :=
sorry

end pairwise_relatively_prime_numbers_can_be_made_coprime_l87_87265


namespace find_a_n_l87_87465

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 2 ∧ ∀ n, a (n + 1) = (n + 3) * a n

theorem find_a_n (a : ℕ → ℕ) (h : seq a) (n : ℕ) : a n = (n + 2)! :=
sorry

end find_a_n_l87_87465


namespace max_attendance_days_l87_87008

def Dan_not_available (d : String) : Prop :=
  d = "Mon" ∨ d = "Wed" ∨ d = "Fri"

def Eve_not_available (d : String) : Prop :=
  d = "Tue" ∨ d = "Thurs" ∨ d = "Fri"

def Frank_not_available (d : String) : Prop :=
  d = "Mon" ∨ d = "Tue" ∨ d = "Fri"

def Grace_not_available (d : String) : Prop :=
  d = "Wed" ∨ d = "Thurs"

def is_available (person_not_available : String → Prop) (d : String) : Prop :=
  ¬ person_not_available d

def num_people_available (d : String) : Nat :=
  [is_available Dan_not_available d,
   is_available Eve_not_available d,
   is_available Frank_not_available d,
   is_available Grace_not_available d].count (λ x => x)

theorem max_attendance_days :
  (num_people_available "Mon" = 2 ∧
   num_people_available "Tue" = 2 ∧
   num_people_available "Wed" = 2 ∧
   num_people_available "Thurs" = 2 ∧
   num_people_available "Fri" ≤ 1) →
  (num_people_available "Mon" = 2 ∨
   num_people_available "Tue" = 2 ∨
   num_people_available "Wed" = 2 ∨
   num_people_available "Thurs" = 2) :=
by
  intros h
  sorry

end max_attendance_days_l87_87008


namespace find_a_l87_87070

-- Definitions
def curve := λ x : ℝ, Real.log x + x^2 + 1
def tangent_slope := (deriv curve) 1
def line_slope (a : ℝ) : ℝ := (-1) / a

-- Theorem statement
theorem find_a (a : ℝ) : tangent_slope = 3 ∧ (tangent_slope * line_slope a = -1) → a = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end find_a_l87_87070


namespace odd_numbers_divisible_by_5_with_form_abcd_l87_87506

theorem odd_numbers_divisible_by_5_with_form_abcd : 
  let form_condition (a b c d : ℕ) := (a ≠ d) ∧ (a ≠ 0) ∧ (b ≠ 5) 
    ∧ (c ≠ a) ∧ (c ≠ b) ∧ (c ≠ 5)
  in 
  let valid_digits (a b c d : ℕ) := 
    {d} = {5} ∧ (1 ≤ a ∧ a ≤ 9 ∧ a ≠ 5) 
    ∧ (0 ≤ b ∧ b ≤ 9 ∧ b ≠ a ∧ b ≠ 5)
    ∧ (0 ≤ c ∧ c ≤ 9 ∧ c ≠ a ∧ c ≠ b ∧ c ≠ 5)
  in 
  (finset.univ.filter (λ n: ℕ, ∃ a b c d: ℕ, 
    form_condition a b c d ∧ valid_digits a b c d ∧ a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + d = n)).card = 448 :=
by
  sorry

end odd_numbers_divisible_by_5_with_form_abcd_l87_87506


namespace quadrilateral_smallest_angle_l87_87917

theorem quadrilateral_smallest_angle :
  ∃ (k : ℚ), (4 * k + 5 * k + 6 * k + 7 * k = 360) ∧ (4 * k = 720 / 11) :=
begin
  use 180 / 11,
  split,
  { -- Prove the sum condition
    rw [mul_comm, ←add_assoc, add_assoc (4 * 180 / 11)],
    ring,
  },
  { -- Prove the measure of the smallest angle
    ring,
  },
end

end quadrilateral_smallest_angle_l87_87917


namespace find_a_l87_87020

open Set

theorem find_a :
  ∀ (A B : Set ℕ) (a : ℕ),
    A = {1, 2, 3} →
    B = {2, a} →
    A ∪ B = {0, 1, 2, 3} →
    a = 0 :=
by
  intros A B a hA hB hUnion
  rw [hA, hB] at hUnion
  sorry

end find_a_l87_87020


namespace treasure_chest_l87_87288

theorem treasure_chest (n : ℕ) 
  (h1 : n % 8 = 2)
  (h2 : n % 7 = 6)
  (h3 : ∀ m : ℕ, (m % 8 = 2 → m % 7 = 6 → m ≥ n)) :
  n % 9 = 7 :=
sorry

end treasure_chest_l87_87288


namespace train_speed_correct_l87_87646

-- Define the length of the train and the time taken to pass the telegraph post
def length_of_train : ℝ := 90
def time_to_pass_post : ℝ := 7.363047319850775

-- Define conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- Define the expected speed in km/h
def expected_speed_kmph : ℝ := 43.9628

-- Define the speed in m/s
def speed_mps := length_of_train / time_to_pass_post

-- Define the speed in km/h
def speed_kmph := speed_mps * conversion_factor

-- The theorem stating that the computed speed is equal to the expected speed
theorem train_speed_correct : speed_kmph = expected_speed_kmph := by
  sorry

end train_speed_correct_l87_87646


namespace three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l87_87159

-- Problem (1)
theorem three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty : 3^40 > 4^30 ∧ 4^30 > 5^20 := 
by
  sorry

-- Problem (2)
theorem sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one : 16^31 > 8^41 ∧ 8^41 > 4^61 :=
by 
  sorry

-- Problem (3)
theorem a_lt_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a^5 = 2) (h4 : b^7 = 3) : a < b :=
by
  sorry

end three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l87_87159


namespace negation_of_exists_abs_lt_one_l87_87923

theorem negation_of_exists_abs_lt_one :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_of_exists_abs_lt_one_l87_87923


namespace work_fraction_left_l87_87626

theorem work_fraction_left (A_days B_days : ℕ) (work_days : ℕ)
  (hA : A_days = 15) (hB : B_days = 20) (h_work : work_days = 3) :
  1 - (work_days * ((1 / A_days) + (1 / B_days))) = 13 / 20 :=
by
  rw [hA, hB, h_work]
  simp
  sorry

end work_fraction_left_l87_87626


namespace max_donation_exists_accepted_extra_lesson_rate_l87_87510

-- Helper definitions for initial conditions:
def sleep_hours_per_day := 9
def work_days_per_month := 20
def hobby_ratio_per_work_hour := 2
def hourly_rate_rubles := 2000
def monthly_passive_income_rubles := 30000
def monthly_expenses_rubles := 60000 

-- Maximum daily hours equation and donations
def daily_hours (L k : ℝ) := (3 * L + k = 15)
def daily_donation (L : ℝ) := (5 - L)

-- Income and expenses equation
def monthly_balance (L k: ℝ) := 
    (monthly_passive_income_rubles / 1000 + work_days_per_month * 2 * L = 
     monthly_expenses_rubles / 1000 + work_days_per_month * daily_donation L)

-- Lean theorem for proof
theorem max_donation_exists :
  ∃ (L : ℝ), daily_hours L (15 - 3 * L) ∧ monthly_balance L (15 - 3 * L) ∧ (work_days_per_month * daily_donation L ≈ 56.67) :=
by
  sorry

theorem accepted_extra_lesson_rate (A: ℝ) : 
  A ≥ 4 :=
by 
  sorry

end max_donation_exists_accepted_extra_lesson_rate_l87_87510


namespace relationship_among_a_b_c_l87_87374

noncomputable def a : ℝ := 3 ^ 0.3
noncomputable def b : ℝ := Real.logBase π 3
noncomputable def c : ℝ := Real.logBase 0.3 2

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  sorry

end relationship_among_a_b_c_l87_87374


namespace circle_area_l87_87870

noncomputable def area_of_circle (d AD BC: ℝ) : ℝ :=
  let r := (AD * BC) / √(AD^2 + BC^2 / 4)
  π * r ^ 2

theorem circle_area (x : ℝ) 
  (diameter : ℝ) 
  (tangent_AD : ℝ := 3 * x)
  (tangent_BC : ℝ := 4 * x) 
  (intersect_p_on_circle : Prop) : 
  area_of_circle diameter tangent_AD tangent_BC = π * (20736 * x^2 / 625) := 
sorry

end circle_area_l87_87870


namespace modulus_of_complex_power_l87_87335

theorem modulus_of_complex_power (z : ℂ) (n : ℕ) (h : z = 2 + I) :
  |z^n| = (|z|^n) :=
begin
  sorry
end

example : |(2 + I)^4| = 25 :=
begin
  have h : 2 + I = (2 + I), from rfl,
  exact modulus_of_complex_power (2 + I) 4 h,
end

end modulus_of_complex_power_l87_87335


namespace log_sqrt_defined_in_interval_l87_87363

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l87_87363


namespace problem_1_problem_2_l87_87411

-- Definition of the function f
def f (x : Real) := 2 * sin(x + Real.pi / 3) * cos(x)

-- Problem 1: Range of f(x)
theorem problem_1 (x : Real) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 0 ≤ f(x) ∧ f(x) ≤ 1 + Real.sqrt 3 / 2 :=
sorry

-- Problem 2 data
structure Triangle :=
  (a b c : Real)
  (A : Real)
  (B : Real)
  (C : Real)
  (acute_A : A < Real.pi / 2)
  (b_eq : b = 2)
  (c_eq : c = 3)

theorem problem_2 (T : Triangle) (hfA : f(T.A) = Real.sqrt 3 / 2) : 
  cosine_law : T.cos(T.A - T.B) = 5 * Real.sqrt 7 / 14 :=
sorry

end problem_1_problem_2_l87_87411


namespace part_a_part_b_part_c_part_d_l87_87157

variable (r β γ α ra b p : ℝ)

-- Conditions
axiom cot_def : ∀ x, x ≠ 0 → Mathlib.sin x ≠ 0 → Mathlib.cot x = Mathlib.cos x / Mathlib.sin x
axiom tan_def : ∀ x, x ≠ 0 → Mathlib.cos x ≠ 0 → Mathlib.tan x = Mathlib.sin x / Mathlib.cos x
axiom sum_angle : α = Mathlib.pi - (β + γ)
axiom alpha_half_eq : Mathlib.cos (Mathlib.pi / 2 - (β + γ) / 2) = Mathlib.sin ((β + γ) / 2)
axiom cos_def : Mathlib.cos (Mathlib.pi / 2 - x) = Mathlib.sin (x)
axiom r_gt_zero : r > 0 
axiom ra_gt_zero : ra > 0 

-- Lean 4 statements
theorem part_a : a = r * (Mathlib.cos (α / 2) / (Mathlib.sin (β / 2) * Mathlib.sin (γ / 2))) :=
  sorry

theorem part_b : a = ra * (Mathlib.cos (α / 2) / (Mathlib.cos (β / 2) * Mathlib.cos (γ / 2))) :=
  sorry

theorem part_c : p - b = r * Mathlib.cot (β / 2) ∧ p - b = ra * Mathlib.tan (γ / 2) :=
  sorry

theorem part_d : p = ra * Mathlib.cot (α / 2) :=
  sorry

end part_a_part_b_part_c_part_d_l87_87157


namespace remainder_when_divided_by_x_minus_2_l87_87954

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

-- State the theorem to be proved with the given conditions
theorem remainder_when_divided_by_x_minus_2 :
  f 2 = -11 :=
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_x_minus_2_l87_87954


namespace cos_75_eq_l87_87307

theorem cos_75_eq : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_eq_l87_87307


namespace find_x_l87_87440

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_eq : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) :
  x = 7 :=
by
  sorry

end find_x_l87_87440


namespace min_value_x_squared_plus_6x_l87_87225

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l87_87225


namespace allocation_schemes_l87_87338

theorem allocation_schemes (volunteers events : ℕ) (h_vol : volunteers = 5) (h_events : events = 4) :
  (∃ allocation_scheme : ℕ, allocation_scheme = 10 * 24) :=
by
  use 240
  sorry

end allocation_schemes_l87_87338


namespace volume_of_region_is_62_5_l87_87726

noncomputable def regionVolume (x y z : ℝ) : Prop := 
(|x - y + z| + |x - y - z| ≤ 10) ∧ (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0)

theorem volume_of_region_is_62_5 :
  let volume := 62.5 in Σ [ x y z : ℝ ] , regionVolume x y z → volume = 62.5  :=
  sorry

end volume_of_region_is_62_5_l87_87726


namespace total_carriages_l87_87596

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l87_87596


namespace sum_perimeters_ratio_perimeters_l87_87118

-- Define the given conditions
variables (A B C : Point) (λ : ℝ) (hλ : λ > 0) -- λ is a positive real number
variables (A' B' C' : Point)
variable (H : Triangle A B C)
variable (H' : Triangle A' B' C')
variables (hA : lies_on_seg A' B C) (hB : lies_on_seg B' C A) (hC : lies_on_seg C' A B)
variables (cond : (dist A C' / dist C' B) = λ ∧ (dist B A' / dist A' C) = λ ∧ (dist C B' / dist B' A) = λ)

-- (a) Calculate the sum of the perimeters of the triangles H_n
theorem sum_perimeters (n : ℕ) :
  sum_perimeter_seq H λ = (λ + 1)^2 / (3 * λ) :=
sorry

-- (b) Prove that ratio of the sum of the perimeters of H_n and H_n' = ratio of the perimeters of H_0 and H_0'
theorem ratio_perimeters (H0 H0' : Triangle) (P_H0 : perimeter H0) (P_H0' : perimeter H0')
  (P_Hn : ∀ n, perimeter (triangle_seq H n)) (P_Hn' : ∀ n, perimeter (inscribed_triangle_seq H λ n)) :
  (sum_perimeter_seq H λ) / (sum_perimeter_seq H' (1/λ)) = P_H0 / P_H0' :=
sorry

end sum_perimeters_ratio_perimeters_l87_87118


namespace edward_total_money_l87_87322

-- define the amounts made and spent
def money_made_spring : ℕ := 2
def money_made_summer : ℕ := 27
def money_spent_supplies : ℕ := 5

-- total money left is calculated by adding what he made and subtracting the expenses
def total_money_end (m_spring m_summer m_supplies : ℕ) : ℕ :=
  m_spring + m_summer - m_supplies

-- the theorem to prove
theorem edward_total_money :
  total_money_end money_made_spring money_made_summer money_spent_supplies = 24 :=
by
  sorry

end edward_total_money_l87_87322


namespace odd_terms_in_expansion_l87_87439

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ n : ℕ, n = 4 ∧ count_odd_terms (expand_binomial p q 10) = n) :=
by sorry

end odd_terms_in_expansion_l87_87439


namespace find_slope_l87_87740

theorem find_slope 
  (k : ℝ)
  (y : ℝ -> ℝ)
  (P : ℝ × ℝ)
  (l : ℝ -> ℝ -> Prop)
  (A B F : ℝ × ℝ)
  (C : ℝ × ℝ -> Prop)
  (d : ℝ × ℝ -> ℝ × ℝ -> ℝ)
  (k_pos : P = (3, 0))
  (k_slope : ∀ x, y x = k * (x - 3))
  (k_int_hyperbola_A : C A)
  (k_int_hyperbola_B : C B)
  (k_focus : F = (2, 0))
  (k_sum_dist : d A F + d B F = 16) :
  k = 1 ∨ k = -1 :=
sorry

end find_slope_l87_87740


namespace find_f_6_minus_a_l87_87407

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^x - 2 else -real.logb 2 (x + 1)

-- Given conditions that f(a) = -3
variable (a : ℝ)
axiom hfa : f a = -3

-- Concluding the proof that f(6-a) = -3/2
theorem find_f_6_minus_a : f (6 - a) = -3 / 2 :=
by
  -- placeholder for proof
  sorry

end find_f_6_minus_a_l87_87407


namespace min_A_cardinality_l87_87383

theorem min_A_cardinality {m a b : ℕ} (H : Nat.gcd a b = 1) (A : Set ℕ) (non_empty : A ≠ ∅) 
  (Ha : ∀ n : ℕ, n > 0 → a * n ∈ A ∨ b * n ∈ A) :
  ∃ c, c = max a b ∧
  min_value_of (A ∩ {x | x ∈ Finset.range (m + 1)}).card =
    if a = 1 ∧ b = 1 then m
    else ∑ i in Finset.range (m + 1), (-1) ^ (i + 1) * ⌊m / c ^ i⌋ :=
sorry

end min_A_cardinality_l87_87383


namespace work_related_emails_count_l87_87294

-- Definitions based on the identified conditions and the question
def total_emails : ℕ := 1200
def spam_percentage : ℕ := 27
def promotional_percentage : ℕ := 18
def social_percentage : ℕ := 15

-- The statement to prove, indicated the goal
theorem work_related_emails_count :
  (total_emails * (100 - spam_percentage - promotional_percentage - social_percentage)) / 100 = 480 :=
by
  sorry

end work_related_emails_count_l87_87294


namespace two_dice_probability_even_l87_87956

theorem two_dice_probability_even :
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  let even_outcomes := {x ∈ outcomes | x % 2 = 0}
  let prob_even := (finset.card even_outcomes) / (finset.card outcomes)
  prob_even * prob_even = 1 / 4 :=
sorry

end two_dice_probability_even_l87_87956


namespace shaded_area_proof_l87_87670

-- Define the given conditions as hypotheses
variables {A B C D E : Type}
variables [fintype DCE → ℝ] [fintype ABC → ℝ]

-- Given conditions
hypothesis h1 : triangle_area ABC = 5
hypothesis h2 : AE = ED
hypothesis h3 : BD = 2 * DC

-- Definition of the total shaded area
def shaded_area : ℝ :=
  let triangle1 := triangle_area DCF in
  let triangle2 := 2 * triangle1 in
  triangle2

-- Target: Prove that the total shaded area is 2 square centimeters
theorem shaded_area_proof : shaded_area = 2 := by
  sorry

end shaded_area_proof_l87_87670


namespace min_value_x_squared_plus_6x_l87_87227

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l87_87227


namespace meet_first_time_at_starting_point_l87_87969

def speed_in_m_per_s (speed_in_kmph : ℕ) : ℕ := speed_in_kmph * 1000 / 3600

def time_to_complete_lap (track_length speed : ℕ) : ℕ := track_length / speed

def lcm (a b : ℕ) : ℕ :=
  (nat.gcd a b) * (a / (nat.gcd a b)) * (b / (nat.gcd b a))

theorem meet_first_time_at_starting_point
  (track_length : ℕ)
  (speed_a_kmph speed_b_kmph : ℕ) :
  let speed_a := speed_in_m_per_s speed_a_kmph,
      speed_b := speed_in_m_per_s speed_b_kmph,
      time_a := time_to_complete_lap track_length speed_a,
      time_b := time_to_complete_lap track_length speed_b
  in 
  lcm time_a time_b = 300 :=
by
  sorry

end meet_first_time_at_starting_point_l87_87969


namespace pointA_in_second_quadrant_l87_87099

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l87_87099


namespace sum_b_n_2015_l87_87129

noncomputable def a_n : ℕ → ℝ
| 1     := 3
| n + 1 := 3 ^ (n + 1)

noncomputable def b_n : ℕ → ℝ
| 1     := 9
| n + 1 := 2 * 3 ^ (n + 1)

theorem sum_b_n_2015 :
  (∑ i in range 2015, b_n (i + 1)) = 3^2016 :=
sorry

end sum_b_n_2015_l87_87129


namespace proof_problem_l87_87755

theorem proof_problem (x : ℝ) (a : ℝ) :
  (0 < x) → 
  (x + 1 / x ≥ 2) →
  (x + 4 / x^2 ≥ 3) →
  (x + 27 / x^3 ≥ 4) →
  a = 4^4 → 
  x + a / x^4 ≥ 5 :=
  sorry

end proof_problem_l87_87755


namespace jack_afternoon_emails_l87_87474

variable (morning_emails total_emails : ℕ)

theorem jack_afternoon_emails :
  morning_emails = 4 →
  total_emails = 5 →
  total_emails - morning_emails = 1 :=
by 
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end jack_afternoon_emails_l87_87474


namespace girls_count_l87_87816

variable (B G : ℕ)

theorem girls_count (h1: B = 387) (h2: G = (B + (54 * B) / 100)) : G = 596 := 
by 
  sorry

end girls_count_l87_87816


namespace find_t_l87_87555

variables {x y : Type} [linear_ordered_field x] [linear_ordered_field y]

-- Define the points given in the problem.
def point1 := (2 : x, 10 : y)
def point2 := (6 : x, 26 : y)
def point3 := (10 : x, 42 : y)

-- The problem is to find t when (45, t) lies on the same line
theorem find_t (t : y) :
  let line := λ x : x, (4 : y) * x + (2 : y) in
  ((45 : x), t) ∈ set_of (λ xy : x × y, xy.snd = line xy.fst) →
  t = 182 :=
by sorry

end find_t_l87_87555


namespace triangle_is_isosceles_l87_87090

variables (A B C : Point)
variables (A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4 : Point)
variables (Γ1 Γ2 : Circle)

-- Definitions based on given conditions
def circle_centered_at_A_with_radius_AB := Circle.mk A (dist A B)
def circle_centered_at_A_with_radius_AC := Circle.mk A (dist A C)

def circle_centered_at_B_with_radius_BA := Circle.mk B (dist B A)
def circle_centered_at_B_with_radius_BC := Circle.mk B (dist B C)

def circle_centered_at_C_with_radius_CA := Circle.mk C (dist C A)
def circle_centered_at_C_with_radius_CB := Circle.mk C (dist C B)

-- Intersection points based on given conditions
def intersection_points_A := { A1, A2, A3, A4 }
def intersection_points_B := { B1, B2, B3, B4 }
def intersection_points_C := { C1, C2, C3, C4 }

-- Two circles that contain the given 12 intersection points
def two_circles := ({A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4} ⊆ points_on_circle Γ1) ∧ 
                    ({A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4} ⊆ points_on_circle Γ2)

-- The theorem to be proved
theorem triangle_is_isosceles (h_cond: 
  two_circles ∧ 
  (∀ p ∈ intersection_points_A ∪ intersection_points_B ∪ intersection_points_C, p ∈ intersection points_on_circle Γ1 ∨ p ∈ points_on_circle Γ2)) : 
  is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l87_87090


namespace homogeneous_diff_eq_solution_l87_87470

open Real

theorem homogeneous_diff_eq_solution (C : ℝ) : 
  ∀ (x y : ℝ), (y^4 - 2 * x^3 * y) * (dx) + (x^4 - 2 * x * y^3) * (dy) = 0 ↔ x^3 + y^3 = C * x * y :=
by
  sorry

end homogeneous_diff_eq_solution_l87_87470


namespace value_of_x_l87_87397

variable (x : ℝ)

def data_values := [70, 110, x, 50, x, 210, 100, 85, 40]

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (λ a b => a < b)
  if sorted.length % 2 = 1 then
    sorted.get! (sorted.length / 2)
  else
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2

def mode (l : List ℝ) : ℝ :=
  let freq_map := l.foldl (λ acc x => acc.insert x (acc.find x |>.getD 0 + 1)) Std.HashMap.empty
  freq_map.toList.foldl (λ (max_val, max_count) (val, count) =>
    if count > max_count then (val, count) else (max_val, max_count)) (0, 0) |>.fst

theorem value_of_x :
  let values := data_values x
  mean values = x ∧ median values = x ∧ mode values = x → x = 95 :=
by
  intros values h
  have hmean := h.1
  have hmedian := h.2.1
  have hmode := h.2.2
  sorry  -- Proof of the theorem

end value_of_x_l87_87397


namespace f_equals_max_bound_l87_87472

noncomputable def f (a b : ℝ) : ℝ :=
| (| (b-a)/(a*b) | + (b+a)/(a*b) - 1) | + (| (b-a)/(a*b) |) + (b+a)/(a*b) + 1

theorem f_equals_max_bound (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f a b = 4 * max (1 / a) (max (1 / b) (1 / 2)) :=
sorry

end f_equals_max_bound_l87_87472


namespace cody_tickets_remaining_l87_87673

variable (initial_tickets : ℝ) (lost_tickets : ℝ) (spent_tickets : ℝ) (remaining_tickets : ℝ)

-- Define the conditions
def cody_tickets_conditions : Prop :=
  initial_tickets = 49.0 ∧
  lost_tickets = 6.0 ∧
  spent_tickets = 25.0

-- Define the goal proving Cody has 18 tickets left
theorem cody_tickets_remaining (h : cody_tickets_conditions) : remaining_tickets = 18.0 :=
  by
    -- Extract the conditions
    rcases h with ⟨h1, h2, h3⟩
    -- Compute the intermediate and final ticket counts
    let tickets_after_loss := initial_tickets - lost_tickets
    have h4 : tickets_after_loss = 43.0 := by linarith [h1, h2]
    let tickets_after_spent := tickets_after_loss - spent_tickets
    show remaining_tickets = 18.0, from by
      linarith [h4, h3]

end cody_tickets_remaining_l87_87673


namespace perimeter_shaded_region_l87_87814

-- Definitions based on conditions
def circle_radius : ℝ := 10
def central_angle : ℝ := 300

-- Statement: Perimeter of the shaded region
theorem perimeter_shaded_region 
  : (10 : ℝ) + (10 : ℝ) + ((5 / 6) * (2 * Real.pi * 10)) = (20 : ℝ) + (50 / 3) * Real.pi :=
by
  sorry

end perimeter_shaded_region_l87_87814


namespace point_A_in_second_quadrant_l87_87092

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l87_87092


namespace num_divisible_by_10_l87_87193

theorem num_divisible_by_10 (a b d : ℕ) (h1 : 100 ≤ a) (h2 : a ≤ 500) (h3 : 100 ≤ b) (h4 : b ≤ 500) (h5 : Nat.gcd d 10 = 10) :
  (b - a) / d + 1 = 41 := by
  sorry

end num_divisible_by_10_l87_87193


namespace card_selection_l87_87053

noncomputable def count_ways := 438400

theorem card_selection :
  let decks := 2
  let total_cards := 52 * decks
  let suits := 4
  let non_royal_count := 10 * decks
  let royal_count := 3 * decks
  let non_royal_options := non_royal_count * decks
  let royal_options := royal_count * decks
  1 * (non_royal_options)^4 + (suits.choose 1) * royal_options * (non_royal_options)^3 + (suits.choose 2) * (royal_options)^2 * (non_royal_options)^2 = count_ways :=
sorry

end card_selection_l87_87053


namespace find_equation_line_l87_87378

noncomputable def line_through_point_area (A : Real × Real) (S : Real) : Prop :=
  ∃ (k : Real), (k < 0) ∧ (2 * A.1 + A.2 - 4 = 0) ∧
    (1 / 2 * (2 - k) * (1 - 2 / k) = S)

theorem find_equation_line (A : ℝ × ℝ) (S : ℝ) (hA : A = (1, 2)) (hS : S = 4) :
  line_through_point_area A S →
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ 2 * x + y - 4 = 0 :=
by
  sorry

end find_equation_line_l87_87378


namespace translated_parabola_is_correct_l87_87572

def translate_function (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x, f (x - shift)

def original_parabola (x : ℝ) : ℝ := -x^2

theorem translated_parabola_is_correct :
  translate_function original_parabola 1 = (λ x, -(x - 1)^2) :=
by
  sorry

end translated_parabola_is_correct_l87_87572


namespace parallel_lines_l87_87119

open Real -- Open the real number namespace

/-- Definition of line l1 --/
def line_l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y - 1 = 0

/-- Definition of line l2 --/
def line_l2 (a : ℝ) (x y : ℝ) := x + (a + 1) * y + 4 = 0

/-- The proof statement --/
theorem parallel_lines (a : ℝ) : (a = 1) → (line_l1 a x y) → (line_l2 a x y) := 
sorry

end parallel_lines_l87_87119


namespace rectangle_area_l87_87278

def length : ℝ := 2
def width : ℝ := 4
def area := length * width

theorem rectangle_area : area = 8 := 
by
  -- Proof can be written here
  sorry

end rectangle_area_l87_87278


namespace total_pieces_of_gum_and_candy_l87_87164

theorem total_pieces_of_gum_and_candy 
  (packages_A : ℕ) (pieces_A : ℕ) (packages_B : ℕ) (pieces_B : ℕ) 
  (packages_C : ℕ) (pieces_C : ℕ) (packages_X : ℕ) (pieces_X : ℕ)
  (packages_Y : ℕ) (pieces_Y : ℕ) 
  (hA : packages_A = 10) (hA_pieces : pieces_A = 4)
  (hB : packages_B = 5) (hB_pieces : pieces_B = 8)
  (hC : packages_C = 13) (hC_pieces : pieces_C = 12)
  (hX : packages_X = 8) (hX_pieces : pieces_X = 6)
  (hY : packages_Y = 6) (hY_pieces : pieces_Y = 10) : 
  packages_A * pieces_A + packages_B * pieces_B + packages_C * pieces_C + 
  packages_X * pieces_X + packages_Y * pieces_Y = 344 := 
by
  sorry

end total_pieces_of_gum_and_candy_l87_87164


namespace units_digit_of_G_1000_l87_87693

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_of_G_1000 : (G 1000) % 10 = 2 := 
  sorry

end units_digit_of_G_1000_l87_87693


namespace big_bottle_volume_is_30_l87_87905

def big_bottle_cost := 2700
def small_bottle_volume := 6
def small_bottle_cost := 600
def savings := 300

theorem big_bottle_volume_is_30 : 
  ∃ x : ℕ, 
    let small_bottle_needed := x / small_bottle_volume in
    let cost_of_smalls := small_bottle_needed * small_bottle_cost in
    cost_of_smalls - big_bottle_cost = savings ∧ x = 30 := 
sorry

end big_bottle_volume_is_30_l87_87905


namespace people_got_on_at_second_stop_l87_87212

theorem people_got_on_at_second_stop : 
  ∀ (initial first_stopGotOff second_stopGotOff third_stopGotOff third_stopGotOn final : ℕ), 
  initial = 50 →
  first_stopGotOff = 15 →
  second_stopGotOff = 8 →
  third_stopGotOff = 4 →
  third_stopGotOn = 3 →
  final = 28 →
  let first_stopRemain := initial - first_stopGotOff in
  let second_stopRemain := first_stopRemain - second_stopGotOff in
  let peopleGotOnSecondStop := (final + third_stopGotOff - third_stopGotOn) - second_stopRemain in
  peopleGotOnSecondStop = 2 :=
by
  intros initial first_stopGotOff second_stopGotOff third_stopGotOff third_stopGotOn final h_init h_fs h_ss h_ts h_tson h_final
  let first_stopRemain := initial - first_stopGotOff
  let second_stopRemain := first_stopRemain - second_stopGotOff
  let peopleGotOnSecondStop := (final + third_stopGotOff - third_stopGotOn) - second_stopRemain
  have h1 : first_stopRemain = 35 := by
    rw [h_init, h_fs]
    norm_num
  have h2 : second_stopRemain = 27 := by
    rw [h1, h_ss]
    norm_num
  have h3 : peopleGotOnSecondStop = 2 := by
    rw [h_final, h_ts, h_tson, h2]
    norm_num
  exact h3

end people_got_on_at_second_stop_l87_87212


namespace point_A_in_second_quadrant_l87_87103

-- Define the conditions as functions
def x_coord : ℝ := -3
def y_coord : ℝ := 4

-- Define a set of quadrants for clarity
inductive Quadrant
| first
| second
| third
| fourth

-- Prove that if the point has specific coordinates, it lies in the specific quadrant
def point_in_quadrant (x: ℝ) (y: ℝ) : Quadrant :=
  if x < 0 ∧ y > 0 then Quadrant.second
  else if x > 0 ∧ y > 0 then Quadrant.first
  else if x < 0 ∧ y < 0 then Quadrant.third
  else Quadrant.fourth

-- The statement to prove:
theorem point_A_in_second_quadrant : point_in_quadrant x_coord y_coord = Quadrant.second :=
by 
  -- Proof would go here but is not required per instructions
  sorry

end point_A_in_second_quadrant_l87_87103


namespace most_reasonable_sampling_method_l87_87822

-- Definitions for the conditions
def significant_difference_by_stage : Prop := 
  -- There is a significant difference in vision condition at different educational stages
  sorry

def no_significant_difference_by_gender : Prop :=
  -- There is no significant difference in vision condition between male and female students
  sorry

-- Theorem statement
theorem most_reasonable_sampling_method 
  (h1 : significant_difference_by_stage) 
  (h2 : no_significant_difference_by_gender) : 
  -- The most reasonable sampling method is stratified sampling by educational stage
  sorry :=
by
  -- Proof skipped
  sorry

end most_reasonable_sampling_method_l87_87822


namespace arc_CND_measure_l87_87104

noncomputable def arc_measure (A B C E P F : Type) (arc_AMB arc_EPF : ℝ) : ℝ :=
  let angle_EQF := 180 - arc_AMB
  have h1: angle_EQF = 26, by sorry
  have h2: arc_EPF = 70, by sorry
  (2 * angle_EQF) - arc_EPF

theorem arc_CND_measure : arc_measure (Type : Type) (Type : Type) (Type : Type) (Type : Type) (Type : Type) (Type : Type) 154 70 = 18 := by
  sorry

end arc_CND_measure_l87_87104


namespace lines_parallel_to_same_line_are_parallel_l87_87942

-- Conditions
variables {L₁ L₂ L₃ : Type} [LinearOrder L₁] [LinearOrder L₂] [LinearOrder L₃]

def are_parallel (l m n : L₁) : Prop := ∃ p, l ∥ p ∧ m ∥ p ∧ n ∥ p

-- Problem Statement
theorem lines_parallel_to_same_line_are_parallel
  (L₁ L₂ L₃ : Type) [LinearOrder L₁] [LinearOrder L₂] [LinearOrder L₃] 
  (h₁ : are_parallel L₁ L₃) (h₂ : are_parallel L₂ L₃) : 
  are_parallel L₁ L₂ :=
sorry  -- Proof is omitted

end lines_parallel_to_same_line_are_parallel_l87_87942


namespace parallelogram_area_l87_87147

variables {A B C D E : Type} [euclidean_space A B C D E]
variables (AD AB DE : ℝ) (θ : ℝ)
variable (area : ℝ)

axiom angle_ADB_eq_150 : θ = 150
axiom side_AD_length_eq_10 : AD = 10
axiom side_AB_length_eq_20 : AB = 20
axiom height_DE_length_eq_5_sqrt_3 : DE = 5 * real.sqrt 3
axiom area_eq : area = AB * DE

theorem parallelogram_area :
  angle_ADB_eq_150 ∧ side_AD_length_eq_10 ∧ side_AB_length_eq_20 ∧ height_DE_length_eq_5_sqrt_3 → 
  area = 100 * real.sqrt 3 := 
by 
  intros
  rw [area_eq]
  cancel_denoms

end parallelogram_area_l87_87147


namespace cosine_angle_AE_SD_l87_87913

open Real

variables {S A B C D E : Type}
-- Setting up points and lengths
variables (edge_length: ℝ)
variables [RegularTetrahedron S A B C D edge_length]
variables [Midpoint E S B]

-- Defining the cosine of the angle
def angle_cosine (u v : ℝ) := u * v / (norm u * norm v)

theorem cosine_angle_AE_SD : angle_cosine AE SD = C := 
begin
  sorry
end

end cosine_angle_AE_SD_l87_87913


namespace valentine_giveaway_l87_87138

theorem valentine_giveaway (initial : ℕ) (left : ℕ) (given : ℕ) (h1 : initial = 30) (h2 : left = 22) : given = initial - left → given = 8 :=
by
  sorry

end valentine_giveaway_l87_87138


namespace hemisphere_surface_area_l87_87902

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l87_87902


namespace correct_propositions_l87_87928

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n + a (n+1) > 2 * a n

def prop1 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  a 2 > a 1 → ∀ n > 1, a n > a (n-1)

def prop4 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  ∃ d, ∀ n > 1, a n > a 1 + (n-1) * d

theorem correct_propositions {a : ℕ → ℝ}
  (h : sequence_condition a) :
  (prop1 a h) ∧ (prop4 a h) := 
sorry

end correct_propositions_l87_87928


namespace inverse_of_congruence_implies_equal_area_l87_87188

-- Definitions to capture conditions and relationships
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with congruency of two triangles
  sorry

def equal_areas (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with equal areas of two triangles
  sorry

-- Statement to prove the inverse proposition
theorem inverse_of_congruence_implies_equal_area :
  (∀ T1 T2 : Triangle, congruent_triangles T1 T2 → equal_areas T1 T2) →
  (∀ T1 T2 : Triangle, equal_areas T1 T2 → congruent_triangles T1 T2) :=
  sorry

end inverse_of_congruence_implies_equal_area_l87_87188


namespace minimum_value_is_14_div_27_l87_87716

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l87_87716


namespace correct_operation_l87_87962

theorem correct_operation : ∃ (a : ℝ), (3 + Real.sqrt 2 ≠ 3 * Real.sqrt 2) ∧ 
  ((a ^ 2) ^ 3 ≠ a ^ 5) ∧
  (Real.sqrt ((-7 : ℝ) ^ 2) ≠ -7) ∧
  (4 * a ^ 2 * a = 4 * a ^ 3) :=
by
  sorry

end correct_operation_l87_87962


namespace ratio_of_Jordyn_age_to_Zrinka_age_is_2_l87_87134

variable (Mehki_age : ℕ) (Jordyn_age : ℕ) (Zrinka_age : ℕ)

-- Conditions
def Mehki_is_10_years_older_than_Jordyn := Mehki_age = Jordyn_age + 10
def Zrinka_age_is_6 := Zrinka_age = 6
def Mehki_age_is_22 := Mehki_age = 22

-- Theorem statement: the ratio of Jordyn's age to Zrinka's age is 2.
theorem ratio_of_Jordyn_age_to_Zrinka_age_is_2
  (h1 : Mehki_is_10_years_older_than_Jordyn Mehki_age Jordyn_age)
  (h2 : Zrinka_age_is_6 Zrinka_age)
  (h3 : Mehki_age_is_22 Mehki_age) : Jordyn_age / Zrinka_age = 2 :=
by
  -- The proof would go here
  sorry

end ratio_of_Jordyn_age_to_Zrinka_age_is_2_l87_87134


namespace similar_isosceles_triangle_projection_eq_l87_87578

variables {α : Type*} [linear_ordered_field α]

/-- Two similar isosceles triangles share a common vertex. Prove that the projections of their 
bases onto the line connecting the midpoints of the bases are equal. -/
theorem similar_isosceles_triangle_projection_eq
  {O M N : α}
  (a₁ a₂ h₁ h₂ : α)
  (k : α)
  (hb₁ : 0 < a₁)
  (hb₂ : 0 < a₂)
  (hh₁ : 0 < h₁)
  (hh₂ : 0 < h₂)
  (similar_triangles : a₁ / h₁ = a₂ / h₂)
  (k_value : k = a₁ / h₁) :
  let proj₁ := k * O * sin (M.angle_between N)
      proj₂ := k * O * sin (N.angle_between M) in
  proj₁ = proj₂ := 
by
  sorry

end similar_isosceles_triangle_projection_eq_l87_87578


namespace min_value_of_f_in_interval_l87_87192

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos x ^ 2 + 1 / 2

theorem min_value_of_f_in_interval :
  ∃ x ∈ set.Icc 0 (π / 2), f x = -1 / 2 :=
sorry

end min_value_of_f_in_interval_l87_87192


namespace find_remainder_l87_87492

noncomputable theory

def q (x : ℂ) := ∑ i in range (1003 + 1), x ^ i

def s (x : ℂ) := q(x) % (x^3 + x^2 + 1)

theorem find_remainder :
  |s(1003)| % 1000 = 1 :=
by sorry

end find_remainder_l87_87492


namespace bettys_herb_garden_l87_87301

theorem bettys_herb_garden :
  ∀ (basil oregano thyme rosemary total : ℕ),
    oregano = 2 * basil + 2 →
    thyme = 3 * basil - 3 →
    rosemary = (basil + thyme) / 2 →
    basil = 5 →
    total = basil + oregano + thyme + rosemary →
    total ≤ 50 →
    total = 37 :=
by
  intros basil oregano thyme rosemary total h_oregano h_thyme h_rosemary h_basil h_total h_le_total
  sorry

end bettys_herb_garden_l87_87301


namespace intersection_subset_complement_l87_87485

open Set

variable (U A B : Set ℕ)

theorem intersection_subset_complement (U : Set ℕ) (A B : Set ℕ) 
  (hU: U = {1, 2, 3, 4, 5, 6}) 
  (hA: A = {1, 3, 5}) 
  (hB: B = {2, 4, 5}) : 
  A ∩ (U \ B) = {1, 3} := 
by
  sorry

end intersection_subset_complement_l87_87485


namespace gain_percentage_before_sale_l87_87283

namespace ProofProblem

-- Definitions of given condition parameters, using noncomputable definitions if necessary
noncomputable def marked_price (CP : ℝ) := (1.125 * CP) / 0.90
noncomputable def selling_price (MP : ℝ) := 0.90 * MP
noncomputable def gain_percentage (CP : ℝ) := 0.125 * CP + CP

-- Prove that the gain percentage before the clearance sale is 25%
theorem gain_percentage_before_sale (CP : ℝ) (SP : ℝ) (MP : ℝ)
  (hMP : MP = marked_price CP)
  (hSP1 : SP = selling_price MP)
  (hSP2 : SP = gain_percentage CP)
  (hSP_value : SP = 30) :
  let OSP := MP in
  (OSP - CP) / CP * 100 = 25 :=
by
  sorry

end ProofProblem

end gain_percentage_before_sale_l87_87283


namespace intersection_M_N_l87_87029

noncomputable def M := {x : ℝ | x > 1}
noncomputable def N := {x : ℝ | x < 2}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l87_87029


namespace general_formula_arithmetic_sequence_l87_87398

variable (a : ℕ → ℤ)

def isArithmeticSequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_formula_arithmetic_sequence :
  isArithmeticSequence a →
  a 5 = 9 →
  a 1 + a 7 = 14 →
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  intros h_seq h_a5 h_a17
  sorry

end general_formula_arithmetic_sequence_l87_87398


namespace proof_problem_l87_87384

variable (a : ℝ) (f : ℝ → ℝ)

def p : Prop :=
  ∀ (x : ℝ), 3 - a^(x + 1) = 3

def q : Prop :=
  ∀ (x : ℝ), f(x - 3) = f(x + 3) → f(x) = f(6 - x)

theorem proof_problem : p ∨ ¬ q :=
by
  sorry

end proof_problem_l87_87384


namespace percentage_of_games_won_l87_87561

theorem percentage_of_games_won (games_won games_lost : ℕ) (h : games_won = 13 * games_lost / 7) : 
  (games_won : ℚ) / (games_won + games_lost) * 100 = 65 :=
by
  sorry

end percentage_of_games_won_l87_87561


namespace exists_integer_diff_points_in_polygon_l87_87146

def polygon (P : Set (ℝ × ℝ)) : Prop :=
∃ (n : ℕ) (points : Fin n → ℝ × ℝ), ConvexHull (Finset.image points Finset.univ) = P

noncomputable def area (P : Set (ℝ × ℝ)) : ℝ :=
-- Appropriate definition of area needs to be filled in here
sorry

theorem exists_integer_diff_points_in_polygon (P : Set (ℝ × ℝ)) (h_area : area P > 1) :
  ∃ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ P ∧ (x2, y2) ∈ P ∧ (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (∃ (m n : ℤ), x1 - x2 = m ∧ y1 - y2 = n) :=
sorry

end exists_integer_diff_points_in_polygon_l87_87146


namespace question1_question2_l87_87730

def vector_m (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi / 3), 1)
def vector_n (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)

noncomputable def f (x : ℝ) := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2

theorem question1 {x : ℝ} (h : Real.sin (x - Real.pi / 3) = Real.cos x) : Real.tan x = 2 + Real.sqrt 3 :=
sorry

theorem question2 : ∀ x : ℝ, (0 ≤ x ∧ x ≤ Real.pi / 2) → f x ≤ (6 - Real.sqrt 3) / 4 :=
sorry

end question1_question2_l87_87730


namespace points_lie_on_sphere_l87_87899

-- Define the tetrahedron with vertices A, B, C, D and altitudes AA1, BB1, CC1, DD1 intersecting at H
structure Tetrahedron where
  A B C D : Point
  A1 B1 C1 D1 H : Point
  intersect_at_H : intersects_at A A1 B B1 C C1 D D1 H

-- Define points A2, B2, C2 such that they divide AA1, BB1, CC1 in ratio 2:1
def divides_ratio (A A1 A2 : Point) : Prop := dist A A2 = (2 / 3) * dist A A1

-- Tetrahedron with specific properties
structure TetrahedronSpecial extends Tetrahedron where
  A2 B2 C2 : Point
  divides_A2 : divides_ratio A A1 A2
  divides_B2 : divides_ratio B B1 B2
  divides_C2 : divides_ratio C C1 C2

-- Define the centroid M
def centroid (A B C : Point) : Point := (1 / 3 : ℝ) • (A + B + C)

-- The theorem
theorem points_lie_on_sphere (T : TetrahedronSpecial) :
  ∃ (sphere : Sphere), sphere.contains T.centroid ∧ sphere.contains T.A2 ∧ sphere.contains T.B2 ∧ sphere.contains T.C2 ∧ sphere.contains T.D1 ∧ sphere.contains T.H :=
by
  sorry

end points_lie_on_sphere_l87_87899


namespace base6_arithmetic_series_sum_l87_87725

theorem base6_arithmetic_series_sum :
  (finset.sum (finset.range 101) (λ n : ℕ, nat.base_repr 6 n) = nat.base_parse 6 "7222") :=
sorry

end base6_arithmetic_series_sum_l87_87725


namespace remaining_problems_l87_87664

theorem remaining_problems (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) :
  problems_per_worksheet = 4 → total_worksheets = 9 → graded_worksheets = 5 → 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end remaining_problems_l87_87664


namespace select_three_balls_distinct_numbers_no_consecutive_l87_87206

-- Definitions based on the conditions from part (a)
def colors := {red, blue, yellow, green}
def balls_per_color := 6
def ball_numbers := {1, 2, 3, 4, 5, 6}

-- Theorem statement based on the translation in part (c)
theorem select_three_balls_distinct_numbers_no_consecutive :
  (∃ (selections : finset (colors × ball_numbers)) (H : selections.card = 3),
    (∀ c : colors, (selections.filter (λ x, x.1 = c)).card ≤ 1) ∧
    (∀ {a b : ball_numbers}, a ∈ selections.val.map prod.snd → b ∈ selections.val.map prod.snd → a ≠ b → abs (a - b) ≠ 1)) →
  selections.card = 96 :=
sorry

end select_three_balls_distinct_numbers_no_consecutive_l87_87206


namespace negation_of_P_l87_87781

def P (x : ℝ) : Prop := x^2 + x - 1 < 0

theorem negation_of_P : (¬ ∀ x, P x) ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by
  sorry

end negation_of_P_l87_87781


namespace swap_values_l87_87003

theorem swap_values (A B : ℕ) (h₁ : A = 10) (h₂ : B = 20) : 
    let C := A 
    let A := B 
    let B := C
    A = 20 ∧ B = 10 := by
  let C := A
  let A := B
  let B := C
  have h₃ : C = 10 := h₁
  have h₄ : A = 20 := h₂
  have h₅ : B = 10 := h₃
  exact And.intro h₄ h₅

end swap_values_l87_87003


namespace proof_problem_l87_87438

variable {α : Type*} {f : α → ℝ} {x0 : α} [Topo α] [NormedField α] [OpensMeasurableSpace α]

-- Defining the condition
def condition : Prop :=
  has_deriv_at f (-3) x0  -- This means f'(x0) = -3

-- Defining the limit expression we want to prove
def limit_expression : Prop :=
  tendsto (λ h : ℝ, (f (x0 + h) - f (x0 - 3 * h)) / h) (nhds 0) (𝓝 (-12))

-- The theorem to be proven
theorem proof_problem (h : condition) : limit_expression := by
  sorry

end proof_problem_l87_87438


namespace log_sqrt_defined_in_interval_l87_87360

def defined_interval (x : ℝ) : Prop :=
  ∃ y, y = (5 - x) ∧ y > 0 ∧ (x - 2) ≥ 0

theorem log_sqrt_defined_in_interval {x : ℝ} :
  defined_interval x ↔ (2 < x ∧ x < 5) :=
sorry

end log_sqrt_defined_in_interval_l87_87360


namespace biographies_increase_percentage_l87_87479

def percentage_increase_biographies (B N: ℝ) (h1: 0.2 * B + N = 0.3 * (B + N)) : ℝ :=
  (N / (0.2 * B)) * 100

theorem biographies_increase_percentage (B N: ℝ) (h1: 0.2 * B + N = 0.3 * (B + N)) :
  percentage_increase_biographies B N h1 = 71.43 :=
begin
  -- use the condition to solve for N
  have h2 : N = 0.1 * B / 0.7,
  { -- proof steps omitted for this example
    sorry,
  },
  -- calculate the percentage increase
  have h3 : (0.1 * B / 0.7) / (0.2 * B) * 100 = 71.43,
  { -- proof steps omitted for this example
    sorry,
  },
  rw [percentage_increase_biographies, h2, h3],
end

end biographies_increase_percentage_l87_87479


namespace largest_multiple_of_13_below_neg_124_l87_87952

theorem largest_multiple_of_13_below_neg_124 : 
  ∃ k : ℤ, k < -124 ∧ (k % 13 = 0) ∧ (∀ m : ℤ, m < -124 ∧ (m % 13 = 0) → m ≤ k) ∧ k = -130 :=
begin
  sorry
end

end largest_multiple_of_13_below_neg_124_l87_87952


namespace find_coordinates_P_l87_87013

noncomputable def ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : ℝ × ℝ → Prop :=
  λ p : ℝ × ℝ, (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1

def foci (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((1, 0), (-1, 0))

def rightmost_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

def line_through_A_perpendicular_x (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = a

def max_angle_condition (y : ℝ) : Prop :=
  abs y = 1

theorem find_coordinates_P (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (P : ℝ × ℝ) (hEllipse : ellipse a b h1 h2 h3 (rightmost_vertex a))
  (hLine : line_through_A_perpendicular_x a P)
  (hAngle : max_angle_condition P.2) :
  P = (real.sqrt 2, 1) ∨ P = (real.sqrt 2, -1) :=
sorry

end find_coordinates_P_l87_87013


namespace jack_initial_flights_up_l87_87106

theorem jack_initial_flights_up
  (steps_per_flight : ℕ) (height_per_step : ℕ) (flights_down : ℕ) (final_position_down : ℕ) 
  (h_steps_per_flight : steps_per_flight = 12)
  (h_height_per_step : height_per_step = 8)
  (h_flights_down : flights_down = 6)
  (h_final_position_down : final_position_down = 24 * 12):
  let height_per_flight := steps_per_flight * height_per_step in
  let total_down_inches := flights_down * height_per_flight in
  let total_up_inches := total_down_inches + final_position_down in 
  total_up_inches / height_per_flight = 9 :=
by
  rw [h_steps_per_flight, h_height_per_step, h_flights_down, h_final_position_down]
  let height_per_flight := 12 * 8
  let total_down_inches := 6 * height_per_flight
  let total_up_inches := total_down_inches + (24 * 12)
  sorry

end jack_initial_flights_up_l87_87106


namespace transformed_sin_eq_l87_87938

def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := f (x + (Real.pi / 3))
def h (x : ℝ) : ℝ := g (x / 2)

theorem transformed_sin_eq :
  ∀ x : ℝ, h(x) = Real.sin (2 * x + (Real.pi / 3)) := by
  sorry

end transformed_sin_eq_l87_87938


namespace find_x_l87_87794

theorem find_x (x : ℝ) :
  log 5 (x^2 - 5*x + 14) = 2 ↔ x = (5 + Real.sqrt 69) / 2 ∨ x = (5 - Real.sqrt 69) / 2 :=
by
  sorry

end find_x_l87_87794


namespace circle_quad_perpendicular_l87_87556

open EuclideanGeometry

theorem circle_quad_perpendicular {A B C D O F : Point}
  (h_circle : OnCircle A O ∧ OnCircle B O ∧ OnCircle C O ∧ OnCircle D O)
  (h_perp : ∠ A C O = 90 ∧ ∠ B D O = 90)
  (h_F : IsPerpendicularFoot O A B F) :
  distance C D = 2 * distance O F := 
  sorry

end circle_quad_perpendicular_l87_87556


namespace travel_distance_156_l87_87247

-- Let x be the distance you have traveled
def x (total_distance : ℝ) (travel_distance : ℝ) : Prop := 
  total_distance = 234 ∧ travel_distance + travel_distance / 2 = total_distance

-- Define the proof problem
theorem travel_distance_156:
  ∀ total_distance travel_distance: ℝ, 
  x total_distance travel_distance → travel_distance = 156 :=
by
  intros total_distance travel_distance h
  cases h with h1 h2
  sorry

end travel_distance_156_l87_87247


namespace negation_of_universal_prop_l87_87924

theorem negation_of_universal_prop:
  (¬ (∀ x : ℝ, x ^ 3 - x ≥ 0)) ↔ (∃ x : ℝ, x ^ 3 - x < 0) := 
by 
sorry

end negation_of_universal_prop_l87_87924


namespace base3_to_base10_l87_87314

theorem base3_to_base10 (d0 d1 d2 d3 d4 : ℕ)
  (h0 : d4 = 2)
  (h1 : d3 = 1)
  (h2 : d2 = 0)
  (h3 : d1 = 2)
  (h4 : d0 = 1) :
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0 = 196 := by
  sorry

end base3_to_base10_l87_87314


namespace incorrect_locus_definitions_l87_87594

theorem incorrect_locus_definitions
(statement_A statement_B statement_C statement_D statement_E : Prop)
(hA : ∀ p, (p ∈ locus ↔ satisfies_conditions p) = true)
(hB : ∀ p, (p ∉ locus ↔ ¬satisfies_conditions p) = true ∧ ¬(p ∈ locus ↔ satisfies_conditions p) = true)
(hC : ∀ p, (p ∈ locus → satisfies_conditions p) = true ∧ (∃ p, satisfies_conditions p ∧ p ∉ locus))
(hD : ∀ p, (satisfies_conditions p ↔ p ∈ locus) = true)
(hE : ∀ p, (p ∈ locus ↔ satisfies_conditions p) ∧ (satisfies_conditions p ↔ p ∈ locus) = true) :
statement_B = false ∧ statement_C = false := 
by
  -- Here we would include the proof steps if required.
  sorry

end incorrect_locus_definitions_l87_87594


namespace minimum_value_is_14_div_27_l87_87715

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l87_87715


namespace difference_between_sum_and_difference_of_digits_l87_87258

-- Definitions of the digits and the number
variables (x y : ℕ)
-- Condition: ratio between the digits is 1:2, hence y = 2 * x
hypothesis h1 : y = 2 * x
-- Condition: difference between the two-digit number and the number obtained by interchanging the digits is 36
hypothesis h2 : (10 * x + y) - (10 * y + x) = 36

theorem difference_between_sum_and_difference_of_digits : 
  (x + y) - (y - x) = 8 :=
by 
  sorry

end difference_between_sum_and_difference_of_digits_l87_87258


namespace DEF_congruent_side_length_l87_87176

-- Problem and conditions
def DEF_isosceles_base : ℝ := 30
def DEF_area : ℝ := 90

-- Height calculated
def DEF_height : ℝ := (2 * DEF_area) / DEF_isosceles_base

-- Midpoint of base
def midpoint_DE : ℝ := DEF_isosceles_base / 2

-- Verification using Pythagorean Theorem
theorem DEF_congruent_side_length :
  ∃ (DF : ℝ), DF = sqrt (midpoint_DE^2 + DEF_height^2) ∧ DF = sqrt(261) :=
by
  sorry

end DEF_congruent_side_length_l87_87176


namespace parallel_lines_distance_l87_87026

-- Define the two lines as given in the conditions
def line1 (x y : ℝ) : Prop := x + 2 * y - 1 = 0
def line2 (x y m : ℝ) : Prop := 2 * x + (m + 1) * y + m - 2 = 0

-- Define the slopes of the lines and the condition that they are parallel
def slope1 : ℝ := -1 / 2
def slope2 (m : ℝ) : ℝ := -2 / (m + 1)

-- Conditions for parallelism
def lines_parallel (m : ℝ) : Prop :=
  slope1 = slope2 m

-- Distance formula for parallel lines Ax + By + C = 0 and given point (x1, y1)
def distance_between_lines (A B C x1 y1 : ℝ) : ℝ :=
  |A * x1 + B * y1 + C| / (Real.sqrt (A^2 + B^2))

-- Given conditions lead to defining the main problem to be proved
theorem parallel_lines_distance :
  (∀ m, lines_parallel m → m = 3) → 
  let A := 2
  let B := 4
  let C := 1
  let x1 := 0
  let y1 := 1 / 2
  distance_between_lines A B C x1 y1 = 3 * Real.sqrt 5 / 10 := 
by
  intros h_m_parallel
  have m_3 := h_m_parallel 3 sorry -- Proof of m = 3 given parallel condition
  sorry -- Proof of distance formula between the lines using m = 3 and given points

end parallel_lines_distance_l87_87026


namespace max_sum_of_squares_l87_87851

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 85) 
  (h3 : ad + bc = 196) 
  (h4 : cd = 120) : 
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 918 :=
by {
  sorry
}

end max_sum_of_squares_l87_87851


namespace number_of_subsets_l87_87852

def set_A : Set Int := {-1, 0, 1}
def set_B : Set Int := {0, 1, 2, 3}
def set_intersection : Set Int := set_A ∩ set_B
def set_union : Set Int := set_A ∪ set_B
def A_star_B : Set (Int × Int) := {p | p.1 ∈ set_intersection ∧ p.2 ∈ set_union}

theorem number_of_subsets (h_intersection : set_intersection = {0, 1})
                          (h_union : set_union = {-1, 0, 1, 2, 3})
                          (h_elements_A_star_B : (set_intersection.card * set_union.card = 10)) :
  (2 ^ (set_intersection.card * set_union.card)) = 2 ^ 10 :=
by
  sorry

end number_of_subsets_l87_87852


namespace quadrilateral_smallest_angle_l87_87918

theorem quadrilateral_smallest_angle :
  ∃ (k : ℚ), (4 * k + 5 * k + 6 * k + 7 * k = 360) ∧ (4 * k = 720 / 11) :=
begin
  use 180 / 11,
  split,
  { -- Prove the sum condition
    rw [mul_comm, ←add_assoc, add_assoc (4 * 180 / 11)],
    ring,
  },
  { -- Prove the measure of the smallest angle
    ring,
  },
end

end quadrilateral_smallest_angle_l87_87918


namespace average_age_l87_87175
open Nat

def age_to_months (years : ℕ) (months : ℕ) : ℕ := years * 12 + months

theorem average_age :
  let age1 := age_to_months 14 9
  let age2 := age_to_months 15 1
  let age3 := age_to_months 14 8
  let total_months := age1 + age2 + age3
  let avg_months := total_months / 3
  let avg_years := avg_months / 12
  let avg_remaining_months := avg_months % 12
  avg_years = 14 ∧ avg_remaining_months = 10 := by
  sorry

end average_age_l87_87175


namespace probability_real_roots_l87_87860

theorem probability_real_roots : 
  (1 / 5) * ∫ (p : ℝ) in 0..5, if p^2 - p - 2 ≥ 0 then 1 else 0 = 3 / 5 :=
by
  sorry

end probability_real_roots_l87_87860


namespace triangle_angle_division_l87_87123

theorem triangle_angle_division (α : ℝ) (A B C M H D : Point) 
  (h_triangle : triangle A B C) 
  (h_median : median A B C M)
  (h_altitude : altitude A B C H)
  (h_bisector : angle_bisector A B C D) 
  (h_angle_division : angle A B A B + angle A A C = 4 * α) : 
  α = 22.5 :=
by
  sorry

end triangle_angle_division_l87_87123


namespace tangent_line_y_intercept_l87_87634

open Real

theorem tangent_line_y_intercept
  (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 2)
  (h_c1 : c1 = (3, 0)) (h_c2 : c2 = (8, 0)) :
  ∃ l : ℝ, l = 13/4 ∧ is_tangent (circle r1 c1) (circle r2 c2) l ∧
  (∀ p : ℝ × ℝ, p ∈ first_quadrant → is_tangent_point p r1 c1 l ∧ is_tangent_point p r2 c2 l) := 
by
  sorry

-- Helper Definitions (assuming they are defined elsewhere in Mathlib or we can define them here if necessary)
noncomputable def circle (r : ℝ) (c : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p | dist p c = r }

def is_tangent (C1 C2 : set (ℝ × ℝ)) (l : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p ∈ C1 ∧ is_on_line l p) → (p ∈ C2 ∧ is_on_line l p)

def is_tangent_point (p : ℝ × ℝ) (r : ℝ) (c : ℝ × ℝ) (l : ℝ) : Prop :=
  dist p c = r ∧ is_on_line l p

def first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_on_line (l : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = l * p.1

-- Placeholder for required definitions not provided in Mathlib or requiring customization.
-- This can involve defining specific properties for tangency, distances, etc., that capture the geometry of the circles and lines.

end tangent_line_y_intercept_l87_87634


namespace smallest_angle_of_quadrilateral_in_ratio_l87_87919

theorem smallest_angle_of_quadrilateral_in_ratio (k : ℚ) :
  let angles := [4 * k, 5 * k, 6 * k, 7 * k] in
  (angles.sum = 360) →
  4 * k = 720 / 11 :=
by
  intros
  sorry

end smallest_angle_of_quadrilateral_in_ratio_l87_87919


namespace tom_sales_l87_87937

theorem tom_sales (old_salary new_salary sales_value commission_rate : ℕ) (salary_difference commission_per_sale : ℚ) (sales_needed : ℕ) :
  old_salary = 75000 →
  new_salary = 45000 →
  sales_value = 750 →
  commission_rate = 15 →
  salary_difference = old_salary - new_salary →
  commission_per_sale = (commission_rate / 100) * sales_value →
  sales_needed = nat_ceil (salary_difference / commission_per_sale.toNat) →
  sales_needed = 267 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end tom_sales_l87_87937


namespace probability_one_of_three_late_l87_87455

theorem probability_one_of_three_late (p_late : ℚ) (p_ontime : ℚ) :
  p_late = 1/40 ∧ p_ontime = 39/40 →
  let prob : ℚ := 3 * (p_late * p_ontime * p_ontime) in
  ((prob * 100).round = 7.1) :=
begin
  -- Definitions corresponding to conditions
  assume h,
  let p_late := 1 / 40,
  let p_ontime := 39 / 40,

  -- proving the result
  sorry
end

end probability_one_of_three_late_l87_87455


namespace find_geometric_arithmetic_progressions_l87_87214

theorem find_geometric_arithmetic_progressions
    (b1 b2 b3 : ℚ)
    (h1 : b2^2 = b1 * b3)
    (h2 : b2 + 2 = (b1 + b3) / 2)
    (h3 : (b2 + 2)^2 = b1 * (b3 + 16)) :
    (b1 = 1 ∧ b2 = 3 ∧ b3 = 9) ∨ (b1 = 1/9 ∧ b2 = -5/9 ∧ b3 = 25/9) :=
  sorry

end find_geometric_arithmetic_progressions_l87_87214


namespace tangent_line_through_point_l87_87332

theorem tangent_line_through_point (x y : ℝ) (tangent f : ℝ → ℝ) (M : ℝ × ℝ) :
  M = (1, 1) →
  f x = x^3 + 1 →
  tangent x = 3 * x^2 →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ ∀ x0 y0 : ℝ, (y0 = f x0) → (y - y0 = tangent x0 * (x - x0))) ∧
  (x, y) = M →
  (a = 0 ∧ b = 1 ∧ c = -1) ∨ (a = 27 ∧ b = -4 ∧ c = -23) :=
by
  sorry

end tangent_line_through_point_l87_87332


namespace B_and_C_work_time_l87_87627

open Real

noncomputable theory

def A_rate := 1 / 5
def B_rate := 1 / 30
def combined_A_C_rate := 1 / 2

theorem B_and_C_work_time:
  (∃ C_rate : ℝ, combined_A_C_rate = A_rate + C_rate ∧ 
  1 / (B_rate + C_rate) = 3) :=
begin
  let C_rate := combined_A_C_rate - A_rate,
  use C_rate,
  split,
  {
    rw C_rate,
    ring
  },
  {
    rw C_rate,
    field_simp,
    ring
  }
end

end B_and_C_work_time_l87_87627


namespace arithmetic_sequence_primes_leq_11_l87_87442

open Nat

theorem arithmetic_sequence_primes_leq_11
    (a d : Nat)
    (h_prime_seq : ∀ k < n, prime (a + k * d))
    (h_def : d < 2000):
    n ≤ 11 :=
by
  sorry

end arithmetic_sequence_primes_leq_11_l87_87442


namespace triangle_inequality_l87_87974

variable {a b c : ℝ}
variable {x y z : ℝ}

theorem triangle_inequality (ha : a ≥ b) (hb : b ≥ c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hx_yz_sum : x + y + z = π) :
  bc + ca - ab < bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ∧
  bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 := sorry

end triangle_inequality_l87_87974


namespace theta_in_Quadrant_IV_l87_87795

def theta_lies_in_Quadrant_IV (θ : ℝ) : Prop :=
  sin θ < 0 ∧ cos θ > 0

theorem theta_in_Quadrant_IV
  (θ : ℝ)
  (h1 : sin θ < 0)
  (h2 : cos θ > 0) :
  theta_lies_in_Quadrant_IV θ :=
by
  exact ⟨h1, h2⟩

end theta_in_Quadrant_IV_l87_87795


namespace smallest_gcd_value_l87_87060

open Int

theorem smallest_gcd_value (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : gcd a b = 18) : gcd (12 * a) (20 * b) = 72 := 
by 
  sorry

end smallest_gcd_value_l87_87060


namespace ellipse_eq_satisfied_slope_sum_constant_satisfied_l87_87031

-- Define the conditions of the ellipse and related lines and points
variables (a b : ℝ) (M N : ℝ × ℝ) 

-- Conditions
def ellipse_eq := ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1
def foci := (-sqrt 2, 0) ∧ (sqrt 2, 0)
def minor_axis_condition := M = (1, 0)
def line_through_point := ∀ (l : ℝ × ℝ → Prop), l (1, 0)

-- Verify that the ellipse equation is valid
def ellipse_valid (a b : ℝ) (condition : a > b ∧ b > 0 ∧ c = sqrt 2) :=
  b = 1 ∧ a = sqrt 3 → ∀ x y, (x^2)/(3) + y^2 = 1

-- Prove that the sum of slopes is constant
def slope_sum_constant (M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  ∀ l (A B : ℝ × ℝ), l M → (2 + 2k = 2)

theorem ellipse_eq_satisfied
  (a b : ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (condition1 : a > b ∧ b > 0 ∧ c = sqrt 2)
  (condition2 : ellipse_valid a b condition1)
  (condition3 : minor_axis_condition M)
: ellipse_eq a b :=
sorry

theorem slope_sum_constant_satisfied
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
: slope_sum_constant M N :=
sorry

end ellipse_eq_satisfied_slope_sum_constant_satisfied_l87_87031


namespace books_before_grant_correct_l87_87535

-- Definitions based on the given conditions
def books_purchased : ℕ := 2647
def total_books_now : ℕ := 8582

-- Definition and the proof statement
def books_before_grant : ℕ := 5935

-- Proof statement: The number of books before the grant plus the books purchased equals the total books now
theorem books_before_grant_correct :
  books_before_grant + books_purchased = total_books_now :=
by
  -- Predictably, no need to complete proof, 'sorry' is used.
  sorry

end books_before_grant_correct_l87_87535


namespace log_sum_inverse_l87_87019

open Real

theorem log_sum_inverse (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) 
  (h : 2 + log 2 a = 3 + log 3 b ∧ 3 + log 3 b = log 6 (a + b)) :
  1/a + 1/b = 108 := by
  sorry

end log_sum_inverse_l87_87019


namespace abs_complex_expression_l87_87859

noncomputable def imaginary_unit : ℂ := complex.I

theorem abs_complex_expression : 
  ∥(1 - imaginary_unit) - (2 / imaginary_unit)∥ = real.sqrt 2 :=
by
  sorry

end abs_complex_expression_l87_87859


namespace smallest_sum_of_18_consecutive_integers_is_perfect_square_l87_87199

theorem smallest_sum_of_18_consecutive_integers_is_perfect_square 
  (n : ℕ) 
  (S : ℕ) 
  (h1 : S = 9 * (2 * n + 17)) 
  (h2 : ∃ k : ℕ, 2 * n + 17 = k^2) 
  (h3 : ∀ m : ℕ, m < 5 → 2 * n + 17 ≠ m^2) : 
  S = 225 := 
by
  sorry

end smallest_sum_of_18_consecutive_integers_is_perfect_square_l87_87199


namespace determine_g_l87_87023

noncomputable def f : ℝ[X] := X^4 - 3 * X^2 + X - 1
noncomputable def h : ℝ[X] := 2 * X^2 + 3 * X - 5

theorem determine_g (g : ℝ[X]) (h_expr : f + g = h) : g = -X^4 + 5 * X^2 + 2 * X - 4 :=
by sorry

end determine_g_l87_87023


namespace directrix_of_parabola_l87_87331

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l87_87331


namespace sum_possible_seats_per_row_l87_87452

theorem sum_possible_seats_per_row:
  ∑ x in {x | ∃ y, x * y = 360 ∧ x ≥ 18 ∧ y ≥ 12}, x = 110 :=
by sorry

end sum_possible_seats_per_row_l87_87452


namespace pointA_in_second_quadrant_l87_87098

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l87_87098


namespace digits_subtraction_eq_zero_l87_87171

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero_l87_87171


namespace line_equation_intercept_l87_87189

noncomputable def point := (-1, 2)

def passes_through (x y : ℝ) (m : ℝ × ℝ) := x * fst m + y * snd m = 1

theorem line_equation_intercept:
  ∀ (a b : ℝ),
    ((a = b) ∧ passes_through (-1) 2 (a, b)) →
    ∀ (x y : ℝ), x + y - 1 = 0 :=
by
  intro a b h
  sorry

end line_equation_intercept_l87_87189


namespace tangent_line_at_point_tangent_through_origin_l87_87405

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_at_point (x : ℝ) (y : ℝ) :
  x = 2 → y = -6 → y = (3 * x^2 + 1) * x + f(x) - ((3 * 2^2 + 1) * 2 + f(2) - (3 * 2^2 + 1) * 2 - 32) :=
by sorry

theorem tangent_through_origin :
  ∃ (x : ℝ), (f(x) / x = (3 * x^2 + 1)) ∧ (x = -2) ∧ f(x) = -26 :=
by sorry

end tangent_line_at_point_tangent_through_origin_l87_87405


namespace point_A_in_second_quadrant_l87_87093

def A : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_A_in_second_quadrant : isSecondQuadrant A :=
by
  sorry

end point_A_in_second_quadrant_l87_87093


namespace find_A_find_y_range_l87_87091

noncomputable def sqrt3 := Real.sqrt 3 -- since we need to use sqrt(3)
noncomputable def pi := Real.pi -- since we need to use π

-- Definitions for the conditions
axiom b_squared_plus_c_squared_minus_a_squared_eq_bc {A B C : Real} (a b c : Real) :
  b^2 + c^2 - a^2 = b * c

axiom a_eq_sqrt3 : Real := sqrt3

-- Proof of part 1
theorem find_A (a b c : Real) (h1 : b_squared_plus_c_squared_minus_a_squared_eq_bc a b c) :
  ∃ A : Real, A = pi / 3 := 
sorry

-- Proof of part 2
theorem find_y_range (a b c A : Real)
  (h1 : b_squared_plus_c_squared_minus_a_squared_eq_bc a b c)
  (h2 : a = sqrt3)
  (h3 : A = pi / 3) :
  ∃ y : Set Real, y = {x : Real | 2 * sqrt3 < x ∧ x ≤ 3 * sqrt3} :=
sorry

end find_A_find_y_range_l87_87091


namespace initial_books_in_library_l87_87209

theorem initial_books_in_library 
  (initial_books : ℕ)
  (books_taken_out_Tuesday : ℕ := 120)
  (books_returned_Wednesday : ℕ := 35)
  (books_withdrawn_Thursday : ℕ := 15)
  (books_final_count : ℕ := 150)
  : initial_books - books_taken_out_Tuesday + books_returned_Wednesday - books_withdrawn_Thursday = books_final_count → initial_books = 250 :=
by
  intros h
  sorry

end initial_books_in_library_l87_87209


namespace tangent_line_at_2_range_of_a_l87_87406

section Part1

def f (x : ℝ) : ℝ := x^3 - x^2 + 10

theorem tangent_line_at_2 :
  let k := 3 * 2^2 - 2 * 2;
  let y2 := f 2;
  (k = 8) ∧ (y2 = 14) ∧ (8 * (2:ℝ) - y2 - 2 = 0) := 
by 
  -- Insert proof here
  sorry

end Part1

section Part2

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 10

lemma g_monotonic_decreasing (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) :
  let g := λ x, x + 10 / x^2;
  let g' := 1 - 20 / x^3;
  g x ≥ g 2 := 
by 
  -- Proof to show g is decreasing in [1, 2]
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (1:ℝ) 2, f x a < 0) → a > 9/2 :=
by 
  -- Use monotonicity of g to show a > 9/2
  sorry

end Part2

end tangent_line_at_2_range_of_a_l87_87406


namespace log_over_sqrt_defined_l87_87356

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l87_87356


namespace smallest_degree_of_f_l87_87114

open Real

noncomputable def smallest_degree_of_polynomial (n : ℕ) : ℕ := 2 * n 

theorem smallest_degree_of_f (n : ℕ) (f : (Fin 4 * n) → ℝ → ℝ) 
  (h₀ : n ≥ 2)
  (h₁ : ∀ (points : Fin 2 * n → ℝ × ℝ),
          f (λ i, points i).fst (λ i, points i).snd = 0 ↔ 
            ((∃ v : Fin 2 * n → ℂ, 
              (∀ i j, i ≠ j → v i ≠ v j) ∧ 
              (∃ θ : ℝ, ∀ i, v i = complex.polar 1 (θ + 2 * i * π / 2 * n))
            )
            ∨
            (∃ x y, ∀ i, points i = (x, y))
          )) : 
  (∀ k, (is_polynomial f k → k ≥ smallest_degree_of_polynomial n)) :=
sorry

end smallest_degree_of_f_l87_87114


namespace log_over_sqrt_defined_l87_87355

theorem log_over_sqrt_defined (x : ℝ) : (2 < x ∧ x < 5) ↔ ∃ f : ℝ, f = (log (5 - x) / sqrt (x - 2)) :=
by
  sorry

end log_over_sqrt_defined_l87_87355


namespace max_donation_exists_accepted_extra_lesson_rate_l87_87509

-- Helper definitions for initial conditions:
def sleep_hours_per_day := 9
def work_days_per_month := 20
def hobby_ratio_per_work_hour := 2
def hourly_rate_rubles := 2000
def monthly_passive_income_rubles := 30000
def monthly_expenses_rubles := 60000 

-- Maximum daily hours equation and donations
def daily_hours (L k : ℝ) := (3 * L + k = 15)
def daily_donation (L : ℝ) := (5 - L)

-- Income and expenses equation
def monthly_balance (L k: ℝ) := 
    (monthly_passive_income_rubles / 1000 + work_days_per_month * 2 * L = 
     monthly_expenses_rubles / 1000 + work_days_per_month * daily_donation L)

-- Lean theorem for proof
theorem max_donation_exists :
  ∃ (L : ℝ), daily_hours L (15 - 3 * L) ∧ monthly_balance L (15 - 3 * L) ∧ (work_days_per_month * daily_donation L ≈ 56.67) :=
by
  sorry

theorem accepted_extra_lesson_rate (A: ℝ) : 
  A ≥ 4 :=
by 
  sorry

end max_donation_exists_accepted_extra_lesson_rate_l87_87509


namespace ratio_Theresa_Timothy_2010_l87_87570

def Timothy_movies_2009 : Nat := 24
def Timothy_movies_2010 := Timothy_movies_2009 + 7
def Theresa_movies_2009 := Timothy_movies_2009 / 2
def total_movies := 129
def Timothy_total_movies := Timothy_movies_2009 + Timothy_movies_2010
def Theresa_total_movies := total_movies - Timothy_total_movies
def Theresa_movies_2010 := Theresa_total_movies - Theresa_movies_2009

theorem ratio_Theresa_Timothy_2010 :
  (Theresa_movies_2010 / Timothy_movies_2010) = 2 :=
by
  sorry

end ratio_Theresa_Timothy_2010_l87_87570


namespace computer_price_after_six_years_l87_87245

def price_decrease (p_0 : ℕ) (rate : ℚ) (t : ℕ) : ℚ :=
  p_0 * rate ^ (t / 2)

theorem computer_price_after_six_years :
  price_decrease 8100 (2 / 3) 6 = 2400 := by
  sorry

end computer_price_after_six_years_l87_87245


namespace parabola_vertex_locus_is_parabola_l87_87499

variables (a c : ℝ) (h_pos_a : a > 0) (h_pos_c : c > 0) 

theorem parabola_vertex_locus_is_parabola :
  (∀ t : ℝ, let xt := (-t / (2 * a)) in let yt := - (t^2 / (4 * a)) + c
            in yt = -a * xt^2 + c) := sorry

end parabola_vertex_locus_is_parabola_l87_87499


namespace cardinality_A17_union_A59_l87_87921

-- Define the arithmetic sequences A_17 and A_59 with the given conditions
def A (k : ℕ) : Set ℕ := { x | ∃ n : ℕ, x = 1 + n * k ∧ x ≤ 2007 }

-- Define the sets A_17 and A_59
def A_17 : Set ℕ := A 17
def A_59 : Set ℕ := A 59

-- Prove the cardinality of the union of A_17 and A_59 is 151
theorem cardinality_A17_union_A59 : (A_17 ∪ A_59).to_finset.card = 151 :=
by
  sorry

end cardinality_A17_union_A59_l87_87921


namespace arcsin_eq_solutions_l87_87893

theorem arcsin_eq_solutions (x : ℝ) (hx : arcsin x + arcsin (3 * x) = π / 4) : 
  x = real.sqrt ((39 + real.sqrt 77) / 722) ∨ 
  x = -real.sqrt ((39 + real.sqrt 77) / 722) ∨ 
  x = real.sqrt ((39 - real.sqrt 77) / 722) ∨ 
  x = -real.sqrt ((39 - real.sqrt 77) / 722) :=
by
  sorry

end arcsin_eq_solutions_l87_87893


namespace linear_regression_equation_l87_87396

theorem linear_regression_equation 
  (x y : ℝ) 
  (positive_correlation : x > 0 ∧ y > 0) 
  (sample_mean_x : ℝ := 2) 
  (sample_mean_y : ℝ := 3) 
  (equation : ℝ → ℝ := λ x, 2 * x - 1) :
  equation 2 = sample_mean_y :=
by
  -- proof
  sorry

end linear_regression_equation_l87_87396


namespace chris_pounds_of_nuts_l87_87306

theorem chris_pounds_of_nuts :
  ∀ (R : ℝ) (x : ℝ),
  (∃ (N : ℝ), N = 4 * R) →
  (∃ (total_mixture_cost : ℝ), total_mixture_cost = 3 * R + 4 * R * x) →
  (3 * R = 0.15789473684210525 * total_mixture_cost) →
  x = 4 :=
by
  intros R x hN htotal_mixture_cost hRA
  sorry

end chris_pounds_of_nuts_l87_87306


namespace complex_expression_eval_l87_87686

theorem complex_expression_eval :
  √3 * Real.tan (Real.pi / 6) + (1 / 2) ^ (-2 : ℤ) + Real.abs (√2 - 1) + Real.cbrt (-64) = √2 :=
by
  sorry

end complex_expression_eval_l87_87686


namespace min_value_x_squared_plus_6x_l87_87239

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l87_87239


namespace curve_intersections_l87_87024

theorem curve_intersections (m : ℝ) :
  (∃ x y : ℝ, ((x-1)^2 + y^2 = 1) ∧ (y = mx + m) ∧ (y ≠ 0) ∧ (y^2 = 0)) =
  ((m > -Real.sqrt 3 / 3) ∧ (m < 0)) ∨ ((m > 0) ∧ (m < Real.sqrt 3 / 3)) := 
sorry

end curve_intersections_l87_87024


namespace possible_values_count_eq_163_l87_87898

def initial_number := 147

def operation1 (n : ℕ) : ℕ := n / 2

def operation2 (n : ℕ) : ℕ := (n + 255) / 2

def operation3 (n : ℕ) : ℕ := n - 64

def is_valid_value (n : ℕ) : Prop :=
  ∃ (i : ℕ), 
    (initial_number = n ∨ 
    (even i → operation1 i = n) ∨ 
    (odd i → operation2 i = n) ∨ 
    (i ≥ 64 → operation3 i = n))

theorem possible_values_count_eq_163 : 
  (setOf is_valid_value).to_finset.card = 163 := sorry

end possible_values_count_eq_163_l87_87898


namespace number_of_odd_numbers_divisible_by_5_l87_87504

theorem number_of_odd_numbers_divisible_by_5 :
  let D := { d | d = 5 }
  let A := { a | a ∈ {1, 2, 3, 4, 6, 7, 8, 9} }
  let B := { b | b ∈ {0, 1, 2, 3, 4, 6, 7, 8, 9} }
  let C := { c | c ∈ {0, 1, 2, 3, 4, 6, 7, 8, 9} }
  ∀ a ∈ A, b ∈ B, c ∈ C, a ≠ b ∧ a ≠ c ∧ b ≠ c → 
  ∃ (count : ℕ), 
  count = 1 * 8 * 8 * 7 ∧ 
  count = 448 := 
by 
  sorry

end number_of_odd_numbers_divisible_by_5_l87_87504


namespace area_of_transformed_parallelogram_l87_87025

variables (a b : ℝ × ℝ × ℝ)
variable (h : ‖a.cross b‖ = 12)

theorem area_of_transformed_parallelogram : 
  ‖(3 • a + b).cross (2 • a - 4 • b)‖ = 168 := 
  sorry

end area_of_transformed_parallelogram_l87_87025


namespace evaluate_expression_l87_87703

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := 
by 
  -- skipping the actual proof
  sorry

end evaluate_expression_l87_87703


namespace spearman_significance_not_rejected_l87_87079

noncomputable def test_spearman_significance 
    (n : ℕ) 
    (rho_e : ℝ) 
    (alpha : ℝ) 
    (critical_value : ℝ) 
    (degrees_of_freedom : ℕ) 
    (t_crit : ℝ) : Prop :=
  let k := n - 2
  let T_kp := rho_e * real.sqrt ((k:ℝ) / (1 - rho_e^2))
  n = 10 ∧
  rho_e = 0.64 ∧
  alpha = 0.01 ∧
  degrees_of_freedom = 8 ∧
  t_crit = 3.36 ∧
  T_kp ≈ 2.35 ∧        -- Using ≈ for approximate value match in informal math language
  T_kp < t_crit

-- Here we state the theorem that encapsulates the conclusion from the solution steps
theorem spearman_significance_not_rejected : 
  test_spearman_significance 10 0.64 0.01 3.36 8 3.36 :=
by { sorry }

end spearman_significance_not_rejected_l87_87079


namespace discount_rate_l87_87993

theorem discount_rate (cost_shoes cost_socks cost_bag paid_price total_cost discount_amount amount_subject_to_discount discount_rate: ℝ)
  (h1 : cost_shoes = 74)
  (h2 : cost_socks = 2 * 2)
  (h3 : cost_bag = 42)
  (h4 : paid_price = 118)
  (h5 : total_cost = cost_shoes + cost_socks + cost_bag)
  (h6 : discount_amount = total_cost - paid_price)
  (h7 : amount_subject_to_discount = total_cost - 100)
  (h8 : discount_rate = (discount_amount / amount_subject_to_discount) * 100) :
  discount_rate = 10 := sorry

end discount_rate_l87_87993


namespace pieces_of_cake_maximized_l87_87163

noncomputable def unique_digits (l : List ℕ) : Prop := l.nodup

theorem pieces_of_cake_maximized :
  ∃ CAKE PIECE : ℕ,
    10000 ≤ CAKE ∧ CAKE ≤ 99999 ∧
    10000 ≤ PIECE ∧ PIECE ≤ 99999 ∧
    unique_digits ([CAKE / 10000 % 10, CAKE / 1000 % 10, CAKE / 100 % 10, CAKE / 10 % 10, CAKE % 10] ++
                   [PIECE / 10000 % 10, PIECE / 1000 % 10, PIECE / 100 % 10, PIECE / 10 % 10, PIECE % 10]) ∧
    (CAKE = 7 * PIECE) ∧
    (CAKE = 84357 ∨ CAKE = 86457 ∨ CAKE = 95207 ∨ CAKE = 98357) :=
begin
  sorry
end

end pieces_of_cake_maximized_l87_87163


namespace tangent_line_at_x_1_monotonic_intervals_a_range_l87_87774

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

-- Part I
theorem tangent_line_at_x_1 (a : ℝ) (h : a = 1) : 
  ∃ c : ℝ, ∃ m : ℝ, ∀ x, f x a = m * (x - 1) + c := 
sorry 

-- Part II
theorem monotonic_intervals (a : ℝ) : 
  (a ≤ 0 ∧ ∀ x > 0, deriv (λ x, f x a) x > 0 ∧ Ioi 0 = set.Ici 0) ∨ 
  (a > 0 ∧ (∀ x ∈ set.Ioo 0 (1/a), deriv (λ x, f x a) x > 0) ∧
  (∀ x ∈ set.Ioi (1/a), deriv (λ x, f x a) x < 0)) :=
sorry 

-- Part III
theorem a_range (h : ∀ x ∈ set.Icc 2 3, f x a ≥ 0) : 
  a ∈ set.Iic (Real.log 3 / 3) := 
sorry

end tangent_line_at_x_1_monotonic_intervals_a_range_l87_87774


namespace max_quotient_l87_87756

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient_l87_87756


namespace prob_ln_x0_lt_zero_l87_87791

theorem prob_ln_x0_lt_zero {a : ℝ} (h : ∃ a : ℝ, f 2 = 0) (f : ℝ → ℝ := λ x, x^3 - a * x) (h_a : a = 4) :
  ∀ x₀ ∈ set.Ioo 0 a, (x₀ ∈ set.Ioo 0 1) → (1 / 4 : ℝ) := by
sorry

end prob_ln_x0_lt_zero_l87_87791


namespace S8_is_80_l87_87746

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end S8_is_80_l87_87746
