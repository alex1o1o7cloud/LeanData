import Mathlib

namespace balls_to_boxes_l58_58171

theorem balls_to_boxes (balls boxes : ℕ) (h1 : balls = 5) (h2 : boxes = 3) :
  ∃ ways : ℕ, ways = 150 := by
  sorry

end balls_to_boxes_l58_58171


namespace triangle_perimeter_24_l58_58564

noncomputable def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

theorem triangle_perimeter_24 :
  ∃ (a b c : ℝ), (a^2 + b^2 = c^2) ∧ (c = 10) ∧ (∃ (x : ℝ), (x^2 + 6^2 = c^2)) ∧ (x = 8) ∧ 
  triangle_perimeter x 6 10 = 24 := 
begin
  -- Definitions based on the conditions
  let hrt := is_triangle_with_right_angle ABC _,
  have : triangle ABC,
  from sorry,

  have : AB = 10,
  from sorry,

  have : ∃ x, rectangle (CBWZ) with CB = BW = x and WZ = CZ = 6,
  from sorry,

  have : circle_contains_points [X, Y, Z, W],
  from sorry,

  use [10, 8, 6],  -- Use these values

  split,
  { exact sorry },   -- Proof that triangle ABC is a right triangle
  split,
  { exact sorry },   -- Proof that AB = 10
  split,
  { use 8, 
    split, exact sorry,   -- Proof of rectangle properties
    split, exact sorry }, -- Proof of x = 8
  { exact sorry },      -- Proof of perimeter calculation
end

end triangle_perimeter_24_l58_58564


namespace concert_attendance_l58_58992

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l58_58992


namespace angle_equality_l58_58335

-- Definition of the circles and their properties
variables {O1 O2 A B M1 M2 : Type}
variables (circle1 circle2 : O1 → A → Type) (circle2 : O2 → A → Type)
variables (intersect1 : circle1 O1 A) (intersect2 : circle2 O2 A)
variables (line : A → M1 → M2 → Type)

-- Conditions as hypotheses
hypothesis intersect_circles : ∃ A B, intersect1 ∧ intersect2
hypothesis line_intersects : ∀ A, ∃ M1 M2, line A M1 M2

-- Goal
theorem angle_equality (h1 : intersect_circles) (h2 : line_intersects A) :
  ∠ B O1 M1 = ∠ B O2 M2 := 
sorry

end angle_equality_l58_58335


namespace count_three_digit_perfect_squares_l58_58169

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_perfect_squares : 
  ∃ (count : ℕ), count = 22 ∧
  ∀ (n : ℕ), is_three_digit_number n → is_perfect_square n → true :=
sorry

end count_three_digit_perfect_squares_l58_58169


namespace complex_round_quadrant_l58_58929

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end complex_round_quadrant_l58_58929


namespace equilateral_triangle_perimeter_l58_58287

theorem equilateral_triangle_perimeter (s : ℝ) (h1 : s ≠ 0) (h2 : (s ^ 2 * real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * real.sqrt 3 := 
by
  sorry

end equilateral_triangle_perimeter_l58_58287


namespace division_of_couples_l58_58397

theorem division_of_couples (couples : Finset (Fin 10)) 
  (h_couples : ∀ c ∈ couples, ∃ a b : Fin 10, (a ≠ b ∧ (c = {a, b}) ∧ (∀ d1 d2 ∈ c, (d1 ≠ d2) → (d1, d2) ∉ couples))):
  ∃ group1 group2 : Finset (Fin 10), 
  (group1.card = 6 ∧ group2.card = 4) ∧ 
  (∃ (c1 c2 : Finset (Fin 10)), (c1.card = 2 ∧ c2.card = 2) ∧ c1 ⊆ group1 ∧ c2 ⊆ group1 ∧ 
  (group1 ∪ group2 = couples ∧ group1 ∩ group2 = ∅)) ∧ 
  Finset.card (Finset.powerset (Finset.univ.filter (λ x, x ∈ couples)).filter(λ g, (∃ (c1 c2 : Finset (Fin 10)), (c1.card = 2 ∧ c2.card = 2 ∧ c1 ⊆ g ∧ c2 ⊆ g)))) = 130 := 
by
  sorry

end division_of_couples_l58_58397


namespace empty_can_weight_l58_58766

theorem empty_can_weight (W w : ℝ) :
  (W + 2 * w = 0.6) →
  (W + 5 * w = 0.975) →
  W = 0.35 :=
by sorry

end empty_can_weight_l58_58766


namespace probability_A_inter_B_eq_zero_l58_58894

open Set

def f (x : ℕ) : ℤ := 6 * x - 4
def g (x : ℕ) : ℤ := 2 * x - 1

def A : Set ℤ := {y | ∃ x ∈ range (λ x, x ∈ (range (1:ℕ, 6) → (1 + list.range 6))), f x = y}
def B : Set ℤ := {y | ∃ x ∈ range (λ x, x ∈ (range (1:ℕ, 6) → (1 + list.range 6))), g x = y}

theorem probability_A_inter_B_eq_zero : 
  (∀ a ∈ A ∪ B, a ∈ A ∩ B → false) :=
by
  sorry

end probability_A_inter_B_eq_zero_l58_58894


namespace range_of_x_l58_58522

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else 1

theorem range_of_x (x : ℝ) : f (1 - x^2) > f (2x) ↔ x ∈ set.Ioo (-1 : ℝ) (real.sqrt 2 - 1) :=
begin
  sorry
end

end range_of_x_l58_58522


namespace derivative_e_neg_x_sin_2x_tangent_line_at_2_neg6_l58_58381

-- Problem (1)
theorem derivative_e_neg_x_sin_2x :
  deriv (λ x : ℝ, exp (-x) * sin (2 * x)) = λ x, exp (-x) * (2 * cos (2 * x) - sin (2 * x)) :=
sorry

-- Problem (2)
theorem tangent_line_at_2_neg6 :
  ∀ (x : ℝ), (x - 2) ≠ 0 → tangent_line (λ x : ℝ, x^3 - 3 * x) 2 (-6) x = 9 * x - 24 :=
sorry

end derivative_e_neg_x_sin_2x_tangent_line_at_2_neg6_l58_58381


namespace caleb_ice_cream_vs_frozen_yoghurt_l58_58442

theorem caleb_ice_cream_vs_frozen_yoghurt :
  let cost_chocolate_ice_cream := 6 * 5
  let discount_chocolate := 0.10 * cost_chocolate_ice_cream
  let total_chocolate_ice_cream := cost_chocolate_ice_cream - discount_chocolate

  let cost_vanilla_ice_cream := 4 * 4
  let discount_vanilla := 0.07 * cost_vanilla_ice_cream
  let total_vanilla_ice_cream := cost_vanilla_ice_cream - discount_vanilla

  let total_ice_cream := total_chocolate_ice_cream + total_vanilla_ice_cream

  let cost_strawberry_yoghurt := 3 * 3
  let tax_strawberry := 0.05 * cost_strawberry_yoghurt
  let total_strawberry_yoghurt := cost_strawberry_yoghurt + tax_strawberry

  let cost_mango_yoghurt := 2 * 2
  let tax_mango := 0.03 * cost_mango_yoghurt
  let total_mango_yoghurt := cost_mango_yoghurt + tax_mango

  let total_yoghurt := total_strawberry_yoghurt + total_mango_yoghurt

  (total_ice_cream - total_yoghurt = 28.31) := by
  sorry

end caleb_ice_cream_vs_frozen_yoghurt_l58_58442


namespace eq_triangle_perimeter_l58_58284

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l58_58284


namespace find_a_5_l58_58869

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem find_a_5 {a : ℕ → ℤ} {S : ℕ → ℤ}
  (h_seq : arithmetic_sequence a)
  (h_S6 : S 6 = 3)
  (h_a4 : a 4 = 2)
  (h_sum_first_n : sum_first_n a S) :
  a 5 = 5 := 
sorry

end find_a_5_l58_58869


namespace rationalize_denominator_cube_root_l58_58647

noncomputable def cubeRoot (x : ℝ) : ℝ := x ^ (1 / 3)

theorem rationalize_denominator_cube_root : 
  let a := cubeRoot 4 in
  let b := cubeRoot 8 in
  let numerator := 2 * (cubeRoot 16 + cubeRoot 32 + cubeRoot 64) in
  let denominator := -4 in
  (numerator / denominator / 2) = -1 ->
  (16 + 32 + 64 + 2 = 114) :=
by
  intros
  sorry

end rationalize_denominator_cube_root_l58_58647


namespace find_polynomial_l58_58835

noncomputable def p : ℝ[X] :=
  (5 / 4) * X^2 - 5 * X + (15 / 4)

theorem find_polynomial :
  (∀ x, (X^4 - 4 * X^3 + 4 * X^2 + 8 * X - 8) / p = Real_div (x - 1 = 0)
     has_vertical_asymptote x) : 
  ¬(x -> p = 0)) ∧ ¬(Real_function (0) = p = 10) →
  p = (5 / 4) * X^2 - 5 * X + (15 / 4) :=
begin
  unfold Real_div has_vertical_asymptote,
  intro h,
  simp at h,
  exact h,
end

end find_polynomial_l58_58835


namespace cube_volume_l58_58698

variables (x s : ℝ)
theorem cube_volume (h : 6 * s^2 = 6 * x^2) : s^3 = x^3 :=
by sorry

end cube_volume_l58_58698


namespace count_valid_functions_l58_58969

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Definition of the properties for functions
def valid_function (f : ℕ → ℕ) : Prop :=
  ∀ s ∈ S, f (f (f s)) = s ∧ (f s - s) % 3 ≠ 0

-- The theorem statement
theorem count_valid_functions :
  ∃ n, (n = 1728) ∧ (∃ f, valid_function f ∧ card {f | valid_function f} = n) :=
by
  sorry

end count_valid_functions_l58_58969


namespace sqrt_144000_simplified_l58_58268

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplified_l58_58268


namespace square_area_of_diagonal_10_l58_58373

open Real

theorem square_area_of_diagonal_10 :
  ∀ (d : ℝ), d = 10 → (d^2 / 2) = 50 :=
by
  intros d h
  rw h
  norm_num

end square_area_of_diagonal_10_l58_58373


namespace polynomial_solution_l58_58843

noncomputable def find_polynomial : Prop :=
  ∃ (P : ℂ[X]), (∀ X : ℂ, P.eval (2 * X) = P.eval (X - 1) * P.eval 1) →
  ∃ (α : ℂ), ∀ X : ℂ, P.eval X = α

-- Use sorry to mark the statement which would be proven
theorem polynomial_solution :
  find_polynomial :=
sorry

end polynomial_solution_l58_58843


namespace pencils_total_l58_58431

theorem pencils_total (p1 p2 : ℕ) (h1 : p1 = 3) (h2 : p2 = 7) : p1 + p2 = 10 := by
  sorry

end pencils_total_l58_58431


namespace optimal_fencing_cost_l58_58780

noncomputable def min_fencing_cost (area : ℕ) (uncovered_side : ℕ) (cost_A : ℕ) (max_span_A : ℕ) (cost_B : ℕ) : ℕ :=
  let width := area / uncovered_side in
  let length_A := 2 * width + uncovered_side in
  let cost_only_A := cost_A * length_A in
  let cost_comb_A_B := cost_B * uncovered_side + cost_A * 2 * width in
  if cost_only_A ≤ cost_comb_A_B then cost_only_A else cost_comb_A_B

theorem optimal_fencing_cost 
  (area : ℕ) 
  (uncovered_side : ℕ) 
  (cost_A : ℕ) 
  (max_span_A : ℕ) 
  (cost_B : ℕ) 
  (hlt : uncovered_side * (area / uncovered_side) = area)
  (hmA : length_A ≤ max_span_A) :
  min_fencing_cost area uncovered_side cost_A max_span_A cost_B = 784 := 
by
  sorry

end optimal_fencing_cost_l58_58780


namespace kanul_total_cash_l58_58742

theorem kanul_total_cash :
  ∃ T : ℝ, (3000 + 1000 + 0.30 * T = T) ∧ (T ≈ 5714.29) :=
by
  sorry

end kanul_total_cash_l58_58742


namespace cooperative_payment_divisibility_l58_58030

theorem cooperative_payment_divisibility (T_old : ℕ) (N : ℕ) 
  (hN : N = 99 * T_old / 100) : 99 ∣ N :=
by
  sorry

end cooperative_payment_divisibility_l58_58030


namespace distinct_configs_count_l58_58719

theorem distinct_configs_count :
  let grid_size := 3
  let num_stars := 5
  let num_circles := 4
  let total_positions := grid_size * grid_size
  (fact (total_positions) / (fact (num_stars) * fact (num_circles))) / 8 = 23 :=
by {
  -- Here you would write the proof steps, but we use sorry to indicate the proof is omitted.
  sorry
}

end distinct_configs_count_l58_58719


namespace range_of_a_l58_58235

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l58_58235


namespace mean_score_of_students_who_failed_l58_58307

noncomputable def mean_failed_score : ℝ := sorry

theorem mean_score_of_students_who_failed (t p proportion_passed proportion_failed : ℝ) (h1 : t = 6) (h2 : p = 8) (h3 : proportion_passed = 0.6) (h4 : proportion_failed = 0.4) : mean_failed_score = 3 :=
by
  sorry

end mean_score_of_students_who_failed_l58_58307


namespace medium_supermarkets_in_sample_l58_58194

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l58_58194


namespace turnip_pulled_by_mice_l58_58009

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l58_58009


namespace quadrilateral_isosceles_exists_l58_58113

noncomputable def exists_point_N (A B C D M : ℂ) (N : ℂ) : Prop :=
  let isosceles_120 (x y z : ℂ) := (abs (x - y) = abs (y - z)) ∧ (arg (y - x) - arg (z - y) = real.pi * 2 / 3)
  ∧ (arg (y - x) - arg (x - z) = real.pi * 2 / 3)
  in
  isosceles_120 A M B ∧ isosceles_120 C M D ∧
  isosceles_120 B N C ∧ isosceles_120 D N A

theorem quadrilateral_isosceles_exists (A B C D M : ℂ) :
  let isosceles_120 (x y z : ℂ) := (abs (x - y) = abs (y - z)) ∧ (arg (y - x) - arg (z - y) = real.pi * 2 / 3)
  ∧ (arg (y - x) - arg (x - z) = real.pi * 2 / 3)
  in
  isosceles_120 A M B ∧ isosceles_120 C M D → ∃ N : ℂ, exists_point_N A B C D M N :=
by sorry

end quadrilateral_isosceles_exists_l58_58113


namespace unique_triangle_constructions_l58_58147

structure Triangle :=
(a b c : ℝ) (A B C : ℝ)

-- Definitions for the conditions
def SSS (t : Triangle) : Prop := 
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def SAS (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180

def ASA (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.c > 0 ∧ t.A + t.B < 180

def SSA (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180 

-- The formally stated proof goal
theorem unique_triangle_constructions (t : Triangle) :
  (SSS t ∨ SAS t ∨ ASA t) ∧ ¬(SSA t) :=
by
  sorry

end unique_triangle_constructions_l58_58147


namespace _l58_58959

structure Triangle (α : Type*) :=
(A B C : α)

noncomputable def is_perp {α : Type*} [MetricSpace α] (D X Y : α) :=
∃ θ : ℝ, θ = π / 2 ∧ dist X D * dist Y D * cos θ = 0

/- Let ABC be a triangle and D a point such that AD is perpendicular to BC. -/
axiom perp_condition (α : Type*) [MetricSpace α] (T : Triangle α) (D : α) :
  is_perp D T.A T.B T.C

/- There exist points E, F, K, L on lines derived from these perpendicular conditions. -/
axiom points_definitions {α : Type*} [MetricSpace α] (T : Triangle α) (D : α)
  (E F K L: α) :
  is_perp D E K ∧ is_perp D F L ∧
  (∃ (X Y : α), is_circumcircle XY ⟨T.A, T.B, T.C⟩ E F ∧
  (∃ O1 O2 : α,
    is_circumcenter O1 ⟨D, X, Y⟩ ∧ is_circumcenter O2 ⟨D, K, L⟩ ∧ is_equal AO1 DO2))

noncomputable def main_theorem {α : Type*} [MetricSpace α] (T : Triangle α) (D E F K L X Y O1 O2: α) 
  (h1 : is_perp D T.A T.B T.C) (h2 : is_perp D E K) (h3 : is_perp D F L) 
  (h4 : is_circumcircle X Y ⟨T.A, T.B, T.C⟩ E F) 
  (h5 : is_circumcenter O1 ⟨D, X, Y⟩)
  (h6 : is_circumcenter O2 ⟨D, K, L⟩)
  : dist T.A O1 = dist D O2 :=
sorry

end _l58_58959


namespace arithmetic_progression_divisors_l58_58463

theorem arithmetic_progression_divisors (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n > 0) 
  (h2 : ∀ i, 1 ≤ i → i ≤ k → is_dvd (d i) n) 
  (h3 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d j - d i = (j - i) * (d 2 - d 1)) 
  (h4 : (finset.range k).sum d = n) 
  (h5 : k > 1) : 
  ∃ m, n = 6 * m :=
by
  sorry

end arithmetic_progression_divisors_l58_58463


namespace solve_for_x_l58_58007

def nabla (x y : ℝ) : ℝ := x^y + 4 * x

theorem solve_for_x (x : ℝ) (h1 : nabla x 2 = 12) : x = -6 ∨ x = 2 :=
by
  sorry

end solve_for_x_l58_58007


namespace triangle_formation_ways_l58_58338

-- Given conditions
def parallel_tracks : Prop := true -- The tracks are parallel, implicit condition not affecting calculation
def first_track_checkpoints := 6
def second_track_checkpoints := 10

-- The proof problem
theorem triangle_formation_ways : 
  (first_track_checkpoints * Nat.choose second_track_checkpoints 2) = 270 := by
  sorry

end triangle_formation_ways_l58_58338


namespace sum_reciprocal_roots_l58_58604

noncomputable def poly := (∑ i in Finset.range 2020, (λ i, (x : ℂ)^i)) - 1365

theorem sum_reciprocal_roots : 
  let a := (λ n, Root poly n) in
  ∑ n in Finset.range 2020, 1/(1 - a n) = 3101 :=
sorry

end sum_reciprocal_roots_l58_58604


namespace find_k_l58_58679

theorem find_k (k : ℝ) (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0)
  (h3 : r / s = 3) (h4 : r + s = 4) (h5 : r * s = k) : k = 3 :=
sorry

end find_k_l58_58679


namespace find_incorrect_statements_in_triangle_l58_58216

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Define that A, B, C are the angles and a, b, c are the sides of the triangle ABC
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧ ∀ {x y z}, x < y + z

-- State the conditions for each proposition
def condition_A (A B : ℝ) : Prop := A > B → sin A > sin B
def condition_B (A B : ℝ) : Prop := sin (2 * A) = sin (2 * B) → isosceles_triangle A B
def condition_C (a b c : ℝ) : Prop := a^2 + b^2 = c^2 → isosceles_triangle a b c
def condition_D (a b c : ℝ) : Prop := a^2 + b^2 > c^2 → is_obtuse_angle (max_angle A B C)

-- Define a function to check if a triangle is isosceles
def isosceles_triangle (A B : ℝ) : Prop := A = B

-- Define a function to check if the maximum angle is obtuse
def max_angle (A B C : ℝ) : ℝ := max A (max B C)
def is_obtuse_angle (θ : ℝ) : Prop := θ > π / 2

-- The main theorem proving the incorrect statements
theorem find_incorrect_statements_in_triangle (A B C : ℝ) (a b c : ℝ) :
  is_triangle A B C a b c →
  ¬ condition_B A B ∧ ¬ condition_C a b c ∧ ¬ condition_D a b c :=
sorry

end find_incorrect_statements_in_triangle_l58_58216


namespace semi_circle_perimeter_approx_l58_58686

variable (r : ℝ) (π : ℝ)

noncomputable def semi_circle_perimeter (r : ℝ) (π : ℝ) : ℝ :=
  π * r + 2 * r

theorem semi_circle_perimeter_approx (h₁ : r = 4.8) (h₂ : π = 3.14) :
  semi_circle_perimeter r π ≈ 24.672 := by
  sorry

end semi_circle_perimeter_approx_l58_58686


namespace distance_between_points_l58_58806

theorem distance_between_points : 
  ∀ (x1 y1 x2 y2 : ℝ), x1 = -5 → y1 = 2 → x2 = 7 → y2 = 7 → 
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 13 :=
begin
  intros,
  rw [h, h_1, h_2, h_3],
  calc
    sqrt ((7 - (-5))^2 + (7 - 2)^2)
      = sqrt ((7 + 5)^2 + 5^2)           : by congr; linarith
  ... = sqrt (12^2 + 5^2)                : by refl
  ... = sqrt (144 + 25)                  : by simp
  ... = sqrt 169                         : by refl
  ... = 13                               : by norm_num,
end

end distance_between_points_l58_58806


namespace problem1_problem2_problem3_l58_58515

variables (a b : ℝ × ℝ)  -- assuming vectors are in 2D for simplicity
variables (k : ℝ)

-- Condition definitions
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def angle_cos (v1 v2 : ℝ × ℝ) : ℝ := dot_product v1 v2 / (magnitude v1 * magnitude v2)

-- Given conditions
axiom a_magnitude : magnitude a = 3
axiom b_magnitude : magnitude b = 2
axiom angle_between_a_b : angle_cos a b = -1/2
axiom a_b_dot_product : dot_product a b = -3

-- Problem 1: Prove the dot product given the conditions
theorem problem1 : dot_product a b = -3 := 
by sorry

-- Problem 2: Prove the expanded dot product
theorem problem2 : dot_product (2 • a + b) (a - 2 • b) = 19 :=
by sorry

-- Problem 3: Prove the correct value of k for perpendicular vectors
theorem problem3 (h : dot_product (a + b) (a - k • b) = 0) : k = 6 :=
by sorry

end problem1_problem2_problem3_l58_58515


namespace angle_measure_l58_58348

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l58_58348


namespace inequality_proof_l58_58985

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

theorem inequality_proof : (a * b < a + b ∧ a + b < 0) :=
by
  sorry

end inequality_proof_l58_58985


namespace height_of_rectangular_boxes_l58_58426

theorem height_of_rectangular_boxes
  (Lw Bw Hw : ℝ) (N : ℕ) (Lr Wr : ℝ) (Hr : ℝ)
  (hLw : Lw = 8) (hBw : Bw = 7) (hHw : Hw = 6) (hN : N = 1000000) 
  (hLr : Lr = 8) (hWr : Wr = 7) :
  let Vw := (Lw * 100) * (Bw * 100) * (Hw * 100),
      Vr := Vw / N,
      area_base := Lr * Wr
  in Hr = Vr / area_base := by
  sorry

end height_of_rectangular_boxes_l58_58426


namespace napkins_given_by_Olivia_l58_58993

theorem napkins_given_by_Olivia 
  (O : ℕ)
  (h1 : ∀ (O : ℕ), Amelia_gave = 2 * Olivia_gave)
  (h2 : William_had_before = 15)
  (h3 : William_had_after = 45) :
  O = 10 :=
by
  have Amelia_gave := 2 * O
  have total_napkins_after := William_had_before + O + Amelia_gave
  have total_napkins_before := William_had_after - William_had_before
  have napkins_received := total_napkins_before
  linarith

end napkins_given_by_Olivia_l58_58993


namespace volume_box_values_l58_58364

theorem volume_box_values :
  let V := (x + 3) * (x - 3) * (x^2 - 10*x + 25)
  ∃ (x_values : Finset ℕ),
    ∀ x ∈ x_values, V < 1000 ∧ x > 0 ∧ x_values.card = 3 :=
by
  sorry

end volume_box_values_l58_58364


namespace B_can_do_job_alone_in_22point5_days_l58_58759

variable (A B : ℝ)

-- Define A's work rate in days
def A_can_do_job_in_18_days : Prop :=
  A = 1 / 18

-- Define the combined work rate of A and B in days
def together_can_do_job_in_10_days : Prop :=
  A + B = 1 / 10

-- Define B's work rate in days
def B_can_do_job_in_x_days (x : ℝ) : Prop :=
  B = 1 / x

-- Prove that B can do the job alone in 22.5 days
theorem B_can_do_job_alone_in_22point5_days :
  A_can_do_job_in_18_days A →
  together_can_do_job_in_10_days A B →
  B_can_do_job_in_x_days B 22.5 :=
begin
  intros hA hTogether,
  sorry
end

end B_can_do_job_alone_in_22point5_days_l58_58759


namespace original_price_before_discount_l58_58371

def discounted_price (P : ℝ) := 0.75 * P
def final_price (P : ℝ) := 0.75 * discounted_price P
theorem original_price_before_discount :
  (∃ P : ℝ, final_price P = 19) → ∃ P : ℝ, P = 33.78 :=
by {
  intro h,
  rcases h with ⟨P, hP⟩,
  have hP_def : final_price P = 19 := hP,
  rw [final_price, discounted_price] at hP_def,
  sorry
}

end original_price_before_discount_l58_58371


namespace angle_between_vectors_l58_58160

open Real

noncomputable def vector_a : ℝ × ℝ := (1, sqrt 3)
noncomputable def vector_b : ℝ × ℝ := (-1 / 2, sqrt 3 / 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (a : ℝ × ℝ) : ℝ :=
  sqrt (a.1 ^ 2 + a.2 ^ 2)

def cos_theta : ℝ :=
  (dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b)

theorem angle_between_vectors :
  arccos cos_theta = π / 3 :=
sorry

end angle_between_vectors_l58_58160


namespace circle_area_l58_58282

theorem circle_area (h : ∃ r : ℝ, 2 * real.pi * r = 24 * real.pi) : ∃ k : ℝ, π * (12^2 : ℝ) = k * π  :=
by
  obtain ⟨r, hr⟩ := h
  have hr := hr
  use 12 ^ 2
  simp
  sorry

end circle_area_l58_58282


namespace minute_hand_sweep_probability_l58_58997

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l58_58997


namespace length_OP_greater_than_3_l58_58883

-- Define a circle with a given radius
def circle (O : Point) (r : ℝ) : Set Point := { P | dist O P = r }

-- Define the center point O and point P
variable (O P : Point)

-- Condition: Radius of circle O is 3.
axiom h1 : ∃ r, r = 3

-- Condition: Point P is outside circle O.
axiom h2 : dist O P > 3

-- The proof statement to ensure the Lean code builds successfully
theorem length_OP_greater_than_3 (O P : Point) : dist O P > 3 :=
by {
  apply h2,  -- This uses the given condition that P is outside the circle O
  sorry
}

end length_OP_greater_than_3_l58_58883


namespace area_triangle_XYZ_l58_58224

variables (A B C D X Y Z : Type*) [euclidean_geometry A B C D] [point A B] [point A D] 
          [length A B = 8] [length A D = 11] [angle A B D = 60]
          [point X] [on_segment X C D] [ratio C X X D = 1/3]
          [point Y] [on_segment Y A D] [ratio A Y Y D = 1/2]
          [point Z] [on_segment Z A B] [concurrent A X B Y D Z]

theorem area_triangle_XYZ : area_triangle X Y Z = 181 / 11 :=
sorry

end area_triangle_XYZ_l58_58224


namespace kim_shirts_left_l58_58223

-- Conditions
def dozens_of_shirts : ℕ := 4
def shirts_per_dozen : ℕ := 12
def fraction_given : ℚ := 1 / 3

-- Definitions derived from conditions
def total_shirts : ℕ := dozens_of_shirts * shirts_per_dozen
def shirts_given : ℕ := (fraction_given * total_shirts).natAbs
def shirts_left : ℕ := total_shirts - shirts_given

-- Proof statement
theorem kim_shirts_left : shirts_left = 32 := by
  dsimp [shirts_left, shirts_given, total_shirts, dozens_of_shirts, shirts_per_dozen, fraction_given]
  sorry

end kim_shirts_left_l58_58223


namespace ms_hatcher_total_students_l58_58625

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l58_58625


namespace probability_of_snow_at_least_once_l58_58632

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l58_58632


namespace find_monthly_salary_l58_58402

variable (S : ℝ)

theorem find_monthly_salary
  (h1 : 0.20 * S - 0.20 * (0.20 * S) = 220) :
  S = 1375 :=
by
  -- Proof goes here
  sorry

end find_monthly_salary_l58_58402


namespace equation_of_parallel_line_passing_through_A_l58_58840

def point_A := (-2, 3 : ℝ)  -- Define the coordinates of point A
def parallel_line (x y : ℝ) := 4 * x - y - 7 = 0  -- Define the original line

-- Define the desired line passing through point A and being parallel to the original line
theorem equation_of_parallel_line_passing_through_A :
  ∃ t : ℝ, ∀ x y : ℝ, (4 * x - y + t = 0) → (x, y) = point_A → 4 * x - y - 11 = 0 :=
sorry

end equation_of_parallel_line_passing_through_A_l58_58840


namespace monthly_profit_10000_daily_profit_15000_maximize_profit_l58_58023

noncomputable def price_increase (c p: ℕ) (x: ℕ) : ℕ := c + x - p
noncomputable def sales_volume (s d: ℕ) (x: ℕ) : ℕ := s - d * x
noncomputable def monthly_profit (price cost volume: ℕ) : ℕ := (price - cost) * volume
noncomputable def monthly_profit_equation (x: ℕ) : ℕ := (40 + x - 30) * (600 - 10 * x)

theorem monthly_profit_10000 (x: ℕ) : monthly_profit_equation x = 10000 ↔ x = 10 ∨ x = 40 :=
by sorry

theorem daily_profit_15000 (x: ℕ) : ¬∃ x, monthly_profit_equation x = 15000 :=
by sorry

theorem maximize_profit (x p y: ℕ) : (∀ x, monthly_profit (40 + x) 30 (600 - 10 * x) ≤ y) ∧ y = 12250 ∧ x = 65 :=
by sorry

end monthly_profit_10000_daily_profit_15000_maximize_profit_l58_58023


namespace find_maximum_k_l58_58178

noncomputable def maximum_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k) : Prop :=
5 = k^2 * ((x^2) / (y^2) + (y^2) / (x^2)) + k * (x / y + y / x)

theorem find_maximum_k (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) :
  ∃ k : ℝ, 0 < k ∧ maximum_k x y k h1 h2 (5 = k^2 * ((x^2) / (y^2) + (y^2) / (x^2)) + k * (x / y + y / x)) ∧ 
  k = (-1 + real.sqrt 22) / 2 :=
sorry

end find_maximum_k_l58_58178


namespace min_sum_of_squares_l58_58249

open Real
open Finset

noncomputable def resistance_equiv : ℝ := (33/20: ℝ)

theorem min_sum_of_squares {V : Type} [fintype V] {E : finset (V × V)}
  (h_graph : V = 9) (h_edges : E.card = 11) (A I : V)
  (hA : 0 : ℝ) (hI : 1 : ℝ) :
  (∃ f : V → ℝ, f A = 0 ∧ f I = 1 
    ∧ (∑ e in E, (f e.fst - f e.snd)^2) = (20/33: ℝ)) :=
sorry

end min_sum_of_squares_l58_58249


namespace p_at_0_l58_58608

noncomputable def p : Polynomial ℚ := sorry

theorem p_at_0 :
  (∀ n : ℕ, n ≤ 6 → p.eval (2^n) = 1 / (2^n))
  ∧ p.degree = 6 → 
  p.eval 0 = 127 / 64 :=
sorry

end p_at_0_l58_58608


namespace nail_pierces_one_cardboard_l58_58267

structure CardboardPiece where
  shape : String -- This can be any shape identifier
  position : (Real × Real) -- Position of the center of the cardboard piece

structure Box where
  bottom_center : (Real × Real)

def placedOverlappingAndCovering (c1 c2 : CardboardPiece) (b : Box) : Prop := sorry
-- Assume or define condition that completely covered the bottom of the box

theorem nail_pierces_one_cardboard (c1 c2 : CardboardPiece) (b : Box)
  (h1 : placedOverlappingAndCovering c1 c2 b)
  (h2 : b.bottom_center = (0, 0)) :
  ∃ c : CardboardPiece, c ∈ {c1, c2} ∧
  ∀ c' ∈ {c1, c2}, c ≠ c' → (nail_through_center b c') = false :=
sorry
-- We'll define the necessary sorry parts later to ensure that the properties and statements about overlapping and being pierced by a nail are correctly modeled.

end nail_pierces_one_cardboard_l58_58267


namespace price_per_glass_on_first_day_eq_half_l58_58630

structure OrangeadeProblem where
  O : ℝ
  W : ℝ
  P1 : ℝ
  P2 : ℝ
  W_eq_O : W = O
  P2_value : P2 = 0.3333333333333333
  revenue_eq : 2 * O * P1 = 3 * O * P2

theorem price_per_glass_on_first_day_eq_half (prob : OrangeadeProblem) : prob.P1 = 0.50 := 
by
  sorry

end price_per_glass_on_first_day_eq_half_l58_58630


namespace sum_of_edges_of_square_l58_58161

theorem sum_of_edges_of_square (u v w x : ℕ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) 
(hsum : u * x + u * v + v * w + w * x = 15) : u + v + w + x = 8 :=
by
  sorry

end sum_of_edges_of_square_l58_58161


namespace buckets_required_l58_58005

theorem buckets_required (C : ℕ) (h : C > 0) : 
  let original_buckets := 25
  let reduced_capacity := 2 / 5
  let total_capacity := original_buckets * C
  let new_buckets := total_capacity / ((2 / 5) * C)
  new_buckets = 63 := 
by
  sorry

end buckets_required_l58_58005


namespace digits_to_replace_l58_58363

theorem digits_to_replace (a b c d e f : ℕ) :
  (a = 1) →
  (b < 5) →
  (c = 8) →
  (d = 1) →
  (e = 0) →
  (f = 4) →
  (100 * a + 10 * b + c)^2 = 10000 * d + 1000 * e + 100 * f + 10 * f + f :=
  by
    intros ha hb hc hd he hf 
    sorry

end digits_to_replace_l58_58363


namespace color_regions_l58_58721

theorem color_regions (n : ℕ) (circles : fin n → set point) :
  (∀ i j : fin n, i ≠ j → ¬ tangent (circles i) (circles j)) →
    ∃ (coloring : set point → Prop),
      (∀ p q : set point, adjacent p q →
        (coloring p ↔ ¬ coloring q)) :=
sorry

end color_regions_l58_58721


namespace range_of_y_l58_58921

theorem range_of_y (x y : ℝ) (h1 : |y - 2 * x| = x^2) (h2 : -1 < x) (h3 : x < 0) : -3 < y ∧ y < 0 :=
by
  sorry

end range_of_y_l58_58921


namespace wheel_distance_travelled_l58_58425

noncomputable def radius : ℝ := 3
noncomputable def num_revolutions : ℝ := 3
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def total_distance (r : ℝ) (n : ℝ) : ℝ := n * circumference r

theorem wheel_distance_travelled :
  total_distance radius num_revolutions = 18 * Real.pi :=
by 
  sorry

end wheel_distance_travelled_l58_58425


namespace num_pencils_in_grid_l58_58259

theorem num_pencils_in_grid :
  ∀ (n : ℕ), n = 10 → (inside_pencils : ℕ) → inside_pencils = (n - 2) * (n - 2) := by
  intros n hn inside_pencils
  rw hn
  simp
  exact eq.refl 64

-- sorry

end num_pencils_in_grid_l58_58259


namespace eq_triangle_perimeter_l58_58283

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l58_58283


namespace part1_part2_part3_l58_58124

noncomputable def z1 : ℂ := 2 - 3 * complex.I
noncomputable def z2 : ℂ := (15 - 5 * complex.I) / (2 + complex.I)^2

theorem part1 : z1 + conj(z2) = 3 := by
  sorry

theorem part2 : z1 * z2 = -7 - 9 * complex.I := by
  sorry

theorem part3 : z1 / z2 = (11 / 10) + (3 / 10) * complex.I := by
  sorry

end part1_part2_part3_l58_58124


namespace tino_jellybeans_l58_58706

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l58_58706


namespace evaluate_expression_at_2_l58_58830

-- Define the quadratic and linear components of the expression
def quadratic (x : ℝ) := 3 * x ^ 2 - 8 * x + 5
def linear (x : ℝ) := 4 * x - 7

-- State the proposition to evaluate the given expression at x = 2
theorem evaluate_expression_at_2 : quadratic 2 * linear 2 = 1 := by
  -- The proof is skipped by using sorry
  sorry

end evaluate_expression_at_2_l58_58830


namespace seating_arrangement_count_l58_58022

def seating_arrangements : ℕ :=
  (5.fact) * (Nat.choose 6 4) * (4.fact)

theorem seating_arrangement_count : seating_arrangements = 43200 := by
  sorry

end seating_arrangement_count_l58_58022


namespace price_increase_percentage_l58_58777

theorem price_increase_percentage (original_price : ℝ) (discount : ℝ) (reduced_price : ℝ) : 
  reduced_price = original_price * (1 - discount) →
  (original_price / reduced_price - 1) * 100 = 8.7 :=
by
  intros h
  sorry

end price_increase_percentage_l58_58777


namespace seating_arrangement_l58_58828

theorem seating_arrangement (n : ℕ) (h1 : n * 9 + (100 - n) * 10 = 100) : n = 10 :=
by sorry

end seating_arrangement_l58_58828


namespace arithmetic_sequence_properties_l58_58232

variable {a : ℕ → ℤ} (S : ℕ → ℤ) (d : ℤ)

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * d

axiom a1_val : a 1 = 30
axiom S12_eq_S19 : sum_of_first_n_terms a 12 = sum_of_first_n_terms a 19

-- Prove that d = -2 and S_n ≤ S_15 for any n
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d →
  (∀ n, S n = sum_of_first_n_terms a n) →
  d = -2 ∧ ∀ n, S n ≤ S 15 :=
by
  intros h_arith h_sum
  sorry

end arithmetic_sequence_properties_l58_58232


namespace parallel_line_distance_l58_58526

theorem parallel_line_distance (x : ℝ) (c : ℝ) :
  (c = 5 + 5 * real.sqrt 13 / 3 ∨ c = 5 - 5 * real.sqrt 13 / 3) ↔
  ∃ c2 : ℝ, y = (2/3 : ℝ) * x + c2 ∧
  (y = (2/3 : ℝ) * x + 5 + 5 * real.sqrt 13 / 3 ∨ 
   y = (2/3 : ℝ) * x + 5 - 5 * real.sqrt 13 / 3) ∧ 
  (real.abs (c2 - 5) = 5 * real.sqrt 13 / 3) :=
sorry

end parallel_line_distance_l58_58526


namespace grandchildren_ages_l58_58768

theorem grandchildren_ages (x : ℕ) (y : ℕ) :
  (x + y = 30) →
  (5 * (x * (x + 1) + (30 - x) * (31 - x)) = 2410) →
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) :=
by
  sorry

end grandchildren_ages_l58_58768


namespace gain_percent_l58_58002

variable (C S : ℝ)
variable (h : 65 * C = 50 * S)

theorem gain_percent (h : 65 * C = 50 * S) : (S - C) / C * 100 = 30 :=
by
  sorry

end gain_percent_l58_58002


namespace length_of_each_lateral_edge_l58_58183

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end length_of_each_lateral_edge_l58_58183


namespace girls_not_playing_soccer_l58_58208

theorem girls_not_playing_soccer 
  (total_students boys students_playing_soccer : ℕ) 
  (boys_playing_soccer_percentage : ℚ) 
  (h1 : total_students = 420) 
  (h2 : boys = 312) 
  (h3 : students_playing_soccer = 250) 
  (h4 : boys_playing_soccer_percentage = 0.78) :
  (total_students - boys) - (students_playing_soccer - (boys_playing_soccer_percentage * students_playing_soccer).to_nat) = 53 :=
by
  sorry

end girls_not_playing_soccer_l58_58208


namespace sum_of_digits_greatest_prime_divisor_6560_l58_58832

theorem sum_of_digits_greatest_prime_divisor_6560 :
  let p := 2 ^ 4 * 5 * 41 in
  let prime_factors := [2, 5, 41] in
  let greatest_prime := 41 in
  (4 + 1 = 5) :=
by
  have factored_6560 : ∃ p, p = 2 ^ 4 * 5 * 41 := ⟨6560, rfl⟩
  have greatest_prime := 41
  have sum_of_digits := 4 + 1
  show 4 + 1 = 5
  sorry

end sum_of_digits_greatest_prime_divisor_6560_l58_58832


namespace smallest_n_l58_58080

-- Definition of Feynman numbers
def Feynman (n : ℕ) := 2 ^ (2 ^ n) + 1

-- Definition of a_n
def a_n (n : ℕ) := log2 (Feynman n - 1)

-- Definition of S_n
def S_n (n : ℕ) := (Finset.range (n + 1)).sum (λ k, a_n k)

-- Inequality Proposition
theorem smallest_n (n : ℕ) :
  (Finset.range n).sum (λ k, 2 ^ (k + 1) / (S_n k * S_n (k + 1))) < 2 ^ n / 1200 :=
  n = 9 :=
begin
  sorry
end

end smallest_n_l58_58080


namespace selection_of_students_l58_58430

theorem selection_of_students (s : Finset ℕ) (A B : ℕ) (hAB : A ∈ s ∧ B ∈ s) (h_size : s.card = 10) : 
  ∃ t ⊆ s, t.card = 4 ∧ (A ∈ t ∨ B ∈ t) ∧ (s.card * (s.card - 1)) / 2 + (s.card * (s.card - 1) * (s.card - 2)) / 6 * 2 = 140 := sorry

end selection_of_students_l58_58430


namespace proof_problem_l58_58857

variable (a b : ℝ)
def a_def : a = 2 * Real.sin (1 / 2) := rfl
def b_def : b = Real.cos (1 / 2) := rfl

theorem proof_problem :
  (a <= 1 ∧ (2 * b^2 - 1 > 1 / 2) ∧ (a > b) ∧ (a + b < Real.sqrt 15 / 2)) :=
by
  sorry

end proof_problem_l58_58857


namespace line_intersects_circle_l58_58519

theorem line_intersects_circle (k : ℝ) :
  (∀ A B : ℝ × ℝ, (A ∈ metric.sphere (0, 2) 2) ∧ (B ∈ metric.sphere (0, 2) 2) ∧ (A.snd = k * A.fst) ∧ (B.snd = k * B.fst) ∧ (dist A B = 2 * real.sqrt 3)) →
  (k = real.sqrt 3 ∨ k = -real.sqrt 3) :=
begin
  sorry
end

end line_intersects_circle_l58_58519


namespace time_A_reaches_destination_l58_58951

theorem time_A_reaches_destination (x t : ℝ) (h_ratio : (4 * t) = 3 * (t + 0.5)) : (t + 0.5) = 2 :=
by {
  -- derived by algebraic manipulation
  sorry
}

end time_A_reaches_destination_l58_58951


namespace complex_quadrant_l58_58931

theorem complex_quadrant (z : ℂ) (h : z * complex.I = 2 - complex.I) : 
    ((z.re < 0) ∧ (z.im < 0)) :=
by
  sorry

end complex_quadrant_l58_58931


namespace sum_of_x_and_y_l58_58506

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l58_58506


namespace find_slope_of_line_l_l58_58577

-- Define the vectors OA and OB
def OA : ℝ × ℝ := (4, 1)
def OB : ℝ × ℝ := (2, -3)

-- The slope k is such that the lengths of projections of OA and OB on line l are equal
theorem find_slope_of_line_l (k : ℝ) :
  (|4 + k| = |2 - 3 * k|) → (k = 3 ∨ k = -1/2) :=
by
  -- Intentionally leave the proof out
  sorry

end find_slope_of_line_l_l58_58577


namespace root_in_interval_l58_58304

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2 ^ x)

theorem root_in_interval : ∃ x ∈ Ioo 1 (3 / 2), f x = 0 :=
by
  sorry

end root_in_interval_l58_58304


namespace factorial_divisibility_l58_58034

theorem factorial_divisibility 
  {n : ℕ} 
  (hn : bit0 (n.bits.count 1) == 1995) : 
  (2^(n-1995)) ∣ n! := 
sorry

end factorial_divisibility_l58_58034


namespace equilateral_triangle_perimeter_l58_58288

theorem equilateral_triangle_perimeter (s : ℝ) (h1 : s ≠ 0) (h2 : (s ^ 2 * real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * real.sqrt 3 := 
by
  sorry

end equilateral_triangle_perimeter_l58_58288


namespace min_even_sum_of_two_pairs_primes_l58_58813

theorem min_even_sum_of_two_pairs_primes (N : ℕ) (hN1 : 2 < N) (hN2 : Nat.even N) :
  (∃! (p1 p2 q1 q2 : ℕ), (Nat.prime p1 ∧ Nat.prime p2 ∧ Nat.prime q1 ∧ Nat.prime q2) ∧
   (N = p1 + p2 ∧ N = q1 + q2) ∧ ((p1, p2) ≠ (q1, q2) ∧ (p1, p2) ≠ (q2, q1)))) →
  N = 10 :=
by
  sorry

end min_even_sum_of_two_pairs_primes_l58_58813


namespace circumference_ratio_l58_58293

theorem circumference_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) : C / D = 3.14 :=
by {
  sorry
}

end circumference_ratio_l58_58293


namespace jenny_profit_l58_58593

def cost_per_pan : ℝ := 10.00
def number_of_pans : ℝ := 20
def price_per_pan : ℝ := 25.00

theorem jenny_profit :
  let total_cost := cost_per_pan * number_of_pans in
  let total_revenue := price_per_pan * number_of_pans in
  let profit := total_revenue - total_cost in
  profit = 300 :=
by
  sorry

end jenny_profit_l58_58593


namespace Tino_jellybeans_l58_58708

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l58_58708


namespace complex_round_quadrant_l58_58930

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end complex_round_quadrant_l58_58930


namespace trajectory_of_midpoint_of_chord_l58_58135

theorem trajectory_of_midpoint_of_chord (CD : ℝ² → ℝ²) (M : ℝ²) :
  (∀ t : ℝ, (CD t) ∈ {p : ℝ² | p.1^2 + p.2^2 = 25}) →
  (|CD 1 - CD 0| = 8) →
  (∃ t : ℝ, M = (CD t) / 2) →
  (M.1^2 + M.2^2 = 9) :=
begin
  intros h1 h2 h3,
  sorry,
end

end trajectory_of_midpoint_of_chord_l58_58135


namespace max_N_and_domain_point_l58_58239

noncomputable def g₁ (x : ℝ) : ℝ := real.sqrt (2 - x)

noncomputable def gn (n : ℕ) (x : ℝ) : ℝ :=
if n = 1 then 
  g₁ (x)
else 
  gn (n - 1) (real.sqrt (n ^ 3 - x))

theorem max_N_and_domain_point :
  ∃ N d, N = 3 ∧ d = -37 ∧ ∀ x, (x = d → g₁(g₁(real.sqrt (8 - x))) = g₁(real.sqrt (27 - x))) ∧ 
  (∀ x, x ∈ set.Icc (-37 : ℝ) (11 : ℝ) →
    ∀ n, (n > 3 → g₁ (real.sqrt (n ^ 3 - x)) = g₁(real.sqrt (64 - x))) ∧ 
  (¬(∃ x, -57 ≤ x ∧ x ≤ -1305))) :=
begin
  sorry
end

end max_N_and_domain_point_l58_58239


namespace length_OP_greater_than_3_l58_58884

-- Define a circle with a given radius
def circle (O : Point) (r : ℝ) : Set Point := { P | dist O P = r }

-- Define the center point O and point P
variable (O P : Point)

-- Condition: Radius of circle O is 3.
axiom h1 : ∃ r, r = 3

-- Condition: Point P is outside circle O.
axiom h2 : dist O P > 3

-- The proof statement to ensure the Lean code builds successfully
theorem length_OP_greater_than_3 (O P : Point) : dist O P > 3 :=
by {
  apply h2,  -- This uses the given condition that P is outside the circle O
  sorry
}

end length_OP_greater_than_3_l58_58884


namespace remainder_equal_to_zero_l58_58908

def A : ℕ := 270
def B : ℕ := 180
def M : ℕ := 25
def R_A : ℕ := A % M
def R_B : ℕ := B % M
def A_squared_B : ℕ := (A ^ 2 * B) % M
def R_A_R_B : ℕ := (R_A * R_B) % M

theorem remainder_equal_to_zero (h1 : A = 270) (h2 : B = 180) (h3 : M = 25) 
    (h4 : R_A = 20) (h5 : R_B = 5) : 
    A_squared_B = 0 ∧ R_A_R_B = 0 := 
by {
    sorry
}

end remainder_equal_to_zero_l58_58908


namespace olivia_used_pieces_l58_58627

-- Definition of initial pieces of paper and remaining pieces of paper
def initial_pieces : ℕ := 81
def remaining_pieces : ℕ := 25

-- Prove that Olivia used 56 pieces of paper
theorem olivia_used_pieces : (initial_pieces - remaining_pieces) = 56 :=
by
  -- Proof steps can be filled here
  sorry

end olivia_used_pieces_l58_58627


namespace sin_sum_eq_one_l58_58834

variable (a b : ℝ)

def sin_inv_45 : a = Real.arcsin (4 / 5) := sorry
def tan_inv_34 : b = Real.arctan (3 / 4) := sorry

def sin_a : Real.sin a = 4 / 5 := sorry
def tan_b_sin : Real.tan b = 3 / 4 := sorry

theorem sin_sum_eq_one : 
  Real.sin (a + b) = 1 := by 
  have h1 : a = Real.arcsin (4 / 5) := sin_inv_45 a b
  have h2 : b = Real.arctan (3 / 4) := tan_inv_34 a b
  have h3 : Real.sin a = 4 / 5 := sin_a a b
  have h4 : Real.tan b = 3 / 4 := tan_b_sin a b
  sorry

end sin_sum_eq_one_l58_58834


namespace probability_one_doctor_one_nurse_l58_58383

theorem probability_one_doctor_one_nurse 
    (doctors nurses : ℕ) 
    (total_selected : ℕ) 
    (h_doctors : doctors = 3)
    (h_nurses : nurses = 2)
    (h_total_selected : total_selected = 2) : 
    (Nat.choose doctors 1) * (Nat.choose nurses 1) / (Nat.choose (doctors + nurses) total_selected) = 0.6 :=
by
  have h1 : Nat.choose doctors 1 = 3, by sorry
  have h2 : Nat.choose nurses 1 = 2, by sorry
  have h3 : Nat.choose 5 2 = 10, by sorry
  rw [h1, h2, h3]
  norm_num
  sorry

end probability_one_doctor_one_nurse_l58_58383


namespace length_AH_l58_58600

def circle (P : Type*) := { center : P // ∃ radius : ℝ, radius > 0 }

variables {P : Type*} [metric_space P] [normed_group P] [normed_space ℝ P] 

variables (Ω Γ : circle P) (A B X Y T M F P H : P) 
variables (O1 O2 : circle P)

-- Given conditions
variables (h1 : ∀ (Q : P), Q ∈ (Γ : set P) → Q ∈ (Ω : set P)) -- Γ is contained in Ω
variables (h2 : ∃ P₀ : P, P₀ ∈ (Γ : set P) ∧ P₀ ∈ (Ω : set P)) -- Γ and Ω are tangent at P
variables (hAB : A ∈ (Ω : set P) ∧ B ∈ (Ω : set P)) -- A, B are points on Ω
variables (hPX : X ≠ P ∧ Y ≠ P ∧ (segment P A) ∩ (Γ : set P) = {X} ∧ (segment P B) ∩ (Γ : set P) = {Y} ) -- PX intersects Γ at X and PY intersects Γ at Y
variables (hF : is_orthogonal (line Y X) (line X P) F) -- F is the foot of the perpendicular from Y to XP
variables (hTM : tangent O1 O2 T M) -- TM is a common tangent to circles O1 and O2
variables (hH : orthocenter (A, B, P) H) -- H is the orthocenter of triangle ABP
variables (PF FX TM PB : ℝ) -- Given lengths
variables (hPX_len : PF = 12)
variables (hFX_len : FX = 15)
variables (hTM_len : TM = 18)
variables (hPB_len : PB = 50)

-- Proof goal
theorem length_AH : ∃ AH : ℝ, AH = 750 / real.sqrt 481 := sorry

end length_AH_l58_58600


namespace log_equation_solution_l58_58274

open Real

theorem log_equation_solution (y : ℝ)
  (h : log 3 ((2 * y + 6) / (4 * y - 1)) + log 3 ((4 * y - 1) / (y - 3)) = 3) :
  y = 87 / 25 := 
sorry

end log_equation_solution_l58_58274


namespace dennis_initial_money_l58_58453

theorem dennis_initial_money 
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (bills_paid : ℝ)
  (change_received : ℝ)
  (sale_price : ℝ)
  (initial_money : ℝ)
  (original_price = 125)
  (discount_percentage = 0.25)
  (bills_paid = 100 + 50 + 4 * 5)
  (change_received = 3 * 20 + 10 + 2 * 5 + 4)
  (sale_price = original_price * (1 - discount_percentage))
  (initial_money = sale_price + change_received) :
  initial_money = 177.75 :=
by
  sorry

end dennis_initial_money_l58_58453


namespace find_value_x_y_cube_l58_58107

variables (x y k c m : ℝ)

theorem find_value_x_y_cube
  (h1 : x^3 * y^3 = k)
  (h2 : 1 / x^3 + 1 / y^3 = c)
  (h3 : x + y = m) :
  (x + y)^3 = c * k + 3 * k^(1/3) * m :=
by
  sorry

end find_value_x_y_cube_l58_58107


namespace length_of_OP_l58_58888

theorem length_of_OP (O P : Point) (r : ℝ) (h_radius : r = 3) (h_outside : dist O P > r) : dist O P = 4 :=
by
  sorry

end length_of_OP_l58_58888


namespace polynomial_roots_AT_TB_l58_58399

theorem polynomial_roots_AT_TB (AT TB : ℝ) (h1 : AT = 2 * TB) (h2 : AT * TB = 16) :
    ∃ (p : polynomial ℝ), p.roots = {AT, TB} ∧ p = polynomial.X^2 - 6*sqrt(2)*polynomial.X + 16 :=
by
  sorry

end polynomial_roots_AT_TB_l58_58399


namespace selling_prices_correct_l58_58187

noncomputable def price_to_achieve_profit : Nat :=
  -- Let the initial conditions
  let purchase_price := 40
  let initial_selling_price := 50
  let initial_units_sold := 500
  let sales_decrease_per_unit_increase := 10
  let desired_profit := 8000

  -- Define the profit equation
  let profit (x : Nat) : Nat := (initial_selling_price + x - purchase_price) * (initial_units_sold - sales_decrease_per_unit_increase * x) 

  -- Calculate x values satisfying the profit equation
  let x1 := 10
  let x2 := 30

  -- Calculate the corresponding selling prices
  (initial_selling_price + x1, initial_selling_price + x2)

theorem selling_prices_correct :
  let (price1, price2) := price_to_achieve_profit
  price1 = 60 ∧ price2 = 80 :=
by
  sorry

end selling_prices_correct_l58_58187


namespace basic_astrophysics_deg_correct_l58_58763

/-- The given budget allocation percentages for different sectors:
  microphotonics: 13%
  home electronics: 24%
  food additives: 15%
  genetically modified microorganisms: 29%
  industrial lubricants: 8%
  basic astrophysics: 11%
-/
def microphotonics_perc := 13
def home_electronics_perc := 24
def food_additives_perc := 15
def gmo_perc := 29
def industrial_lubricants_perc := 8
def basic_astrophysics_perc := 100 - (microphotonics_perc + home_electronics_perc + food_additives_perc + gmo_perc + industrial_lubricants_perc)

/-- The degrees of the circle used to represent basic astrophysics research -/
def basic_astrophysics_degrees := (basic_astrophysics_perc / 100.0) * 360.0

/-- Proof that basic astrophysics research is represented by 39.6 degrees. -/
theorem basic_astrophysics_deg_correct : basic_astrophysics_degrees = 39.6 :=
by
  -- Proof goes here
  sorry

end basic_astrophysics_deg_correct_l58_58763


namespace cauchy_solution_l58_58460

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end cauchy_solution_l58_58460


namespace zero_in_interval_l58_58981

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem zero_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 :=
by
  have h_cont : Continuous f := by sorry
  have h_f1 : f 1 < 0 := by sorry
  have h_f2 : f 2 > 0 := by sorry
  exact IntermediateValueTheorem f h_cont 1 2 h_f1 h_f2

end zero_in_interval_l58_58981


namespace probability_snow_first_week_l58_58634

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l58_58634


namespace calculation_result_l58_58722

def initial_number : ℕ := 15
def subtracted_value : ℕ := 2
def added_value : ℕ := 4
def divisor : ℕ := 1
def second_divisor : ℕ := 2
def multiplier : ℕ := 8

theorem calculation_result : 
  (initial_number - subtracted_value + (added_value / divisor : ℕ)) / second_divisor * multiplier = 68 :=
by
  sorry

end calculation_result_l58_58722


namespace minute_hand_sweep_probability_l58_58995

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l58_58995


namespace ellipse_standard_eq_line_eq_l58_58502

-- Ellipse Constants
def b : ℝ := 2
def c : ℝ := 1
def a : ℝ := Real.sqrt 5
def e : ℝ := Real.sqrt 5 / 5

-- Problem 1: Standard equation of the ellipse
theorem ellipse_standard_eq (x y : ℝ) : 
    (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ 
    (x^2) / 5 + (y^2) / 4 = 1 := sorry

-- Problem 2: Equation of the line l
theorem line_eq (M N : ℝ × ℝ) (k : ℝ) :
    let x1 := M.1
    let x2 := N.1
    let y1 := M.2
    let y2 := N.2
    (x1 - (-Real.sqrt 5))^2 + y1^2 = 0 →
    (x2 - (-Real.sqrt 5))^2 + y2^2 = 0 →
    x2 - x1 ≠ 0 →
    y = k * (x + Real.sqrt 5) → 
    Real.sqrt ((x1 - x2)^2 * (1 + k^2)) = 16 / 9 * Real.sqrt 5 →
    (k = 1 ∨ k = -1) ↔ 
    (y = x + 1 ∨ y = -x - 1) := sorry

end ellipse_standard_eq_line_eq_l58_58502


namespace cone_sphere_surface_area_ratio_l58_58029

noncomputable def inscribed_cone_and_sphere_ratio (r : ℝ) : ℝ :=
  let base_radius := (r * real.sqrt 3) / 2
  let slant_height := r * real.sqrt 3
  let lateral_surface_area := real.pi * base_radius * slant_height
  let base_area := real.pi * base_radius ^ 2
  let cone_surface_area := lateral_surface_area + base_area
  let sphere_surface_area := 4 * real.pi * r ^ 2
  cone_surface_area / sphere_surface_area

theorem cone_sphere_surface_area_ratio (r : ℝ) (h_pos : 0 < r) :
  inscribed_cone_and_sphere_ratio r = 9 / 16 :=
by
  sorry

end cone_sphere_surface_area_ratio_l58_58029


namespace part_i1_part_i2_part_ii_l58_58617

def A : Set ℕ := { x | x > 0 ∧ x < 6 }
def B : Set ℕ := { x | (x-1) * (x-2) = 0 }
def C (m : ℝ) : Set ℝ := { x | (m-1) * x - 1 = 0 }

theorem part_i1 : A ∩ B = {1, 2} := 
by sorry

theorem part_i2 : A ∪ B = {1, 2, 3, 4, 5} := 
by sorry

theorem part_ii (m : ℝ) (hCsubsetB : ∀ x, x ∈ C m → x ∈ B) : m = 1 ∨ m = 2 ∨ m = 3/2 := 
by sorry

end part_i1_part_i2_part_ii_l58_58617


namespace correct_statement_l58_58240

variables (m n : Line) (α β : Plane) 

-- Given conditions
axiom lines_distinct : m ≠ n
axiom planes_distinct : α ≠ β

-- Given correct statement from options
axiom m_parallel_alpha : m ∥ α
axiom m_in_beta : m ⊆ β
axiom alpha_intersect_beta_at_n : α ∩ β = n

-- We need to prove that m is parallel to n.
theorem correct_statement : m ∥ n := 
  sorry

end correct_statement_l58_58240


namespace length_of_PQ_l58_58771

theorem length_of_PQ (p : ℝ) (h : p > 0) (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1x2 : x1 + x2 = 3 * p) (hy1 : y1^2 = 2 * p * x1) (hy2 : y2^2 = 2 * p * x2) 
  (focus : ¬ (y1 = 0)) : (abs (x1 - x2 + 2 * p) = 4 * p) := 
sorry

end length_of_PQ_l58_58771


namespace seq_positive_integers_seq_not_divisible_by_2109_l58_58689

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = (a (n + 1) ^ 2 + 9) / a n

theorem seq_positive_integers (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, 0 < a (n + 1) :=
sorry

theorem seq_not_divisible_by_2109 (a : ℕ → ℤ) (h : seq a) : ¬ ∃ m : ℕ, 2109 ∣ a (m + 1) :=
sorry

end seq_positive_integers_seq_not_divisible_by_2109_l58_58689


namespace find_y_l58_58544

noncomputable def F (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℝ) (h : F (3 : ℕ) (floor y) 8 14 = 900) : 
    y = Real.log 788 / Real.log 3 :=
sorry

end find_y_l58_58544


namespace length_of_OP_l58_58890

theorem length_of_OP (O P : Point) (r : ℝ) (h_radius : r = 3) (h_outside : dist O P > r) : dist O P = 4 :=
by
  sorry

end length_of_OP_l58_58890


namespace problem_geometric_sequence_l58_58131

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem problem_geometric_sequence (a₁ q : ℝ) (n : ℕ)
  (h1 : a₁ + a₁ * q^2 = 10)
  (h2 : a₁ * q^3 + a₁ * q^5 = 5/4)
  (h3 : a₁ = 8)
  (h4 : q = 1/2) :
  (geometric_sequence a₁ q 4 = 1) ∧
  (∑ i in finset.range n, geometric_sequence a₁ q (i + 1) = 16 * (1 - (1/2)^n)) :=
by
  { sorry }

end problem_geometric_sequence_l58_58131


namespace root_of_quadratic_is_3_l58_58478

theorem root_of_quadratic_is_3 (u : ℝ) :
  (∃ x : ℝ, 6 * x^2 + 19 * x + u = 0 ∧ x = (-19 + real.sqrt 289) / 12) → u = 3 :=
by
  sorry

end root_of_quadratic_is_3_l58_58478


namespace max_value_of_y_l58_58859

-- Define the function f(x) = log_3(x) + 2
def f (x : ℝ) : ℝ := log x / log 3 + 2

-- Define the function y(x) = [f(x)]^2 + f(x^2)
def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Prove that the maximum value of y(x) on the interval [1, 9] is 13
theorem max_value_of_y : x ∈ Icc 1 9 → (∀ x ∈ Icc 1 9, y x ≤ 13) ∧ (∃ x ∈ Icc 1 9, y x = 13) :=
by
  sorry

end max_value_of_y_l58_58859


namespace mean_properties_l58_58464

def prime_numbers (lst : List ℕ) : List ℕ := lst.filter Nat.Prime

def arithmetic_mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

noncomputable def geometric_mean (lst : List ℕ) : ℚ :=
  Real.to_rat ((lst.prod : ℚ) ^ (1 / lst.length : ℚ))

theorem mean_properties :
  let primes := prime_numbers [10, 13, 19, 28, 31] in
  primes = [13, 19, 31] ∧
  arithmetic_mean primes = 21 ∧
  geometric_mean primes = Real.to_rat (7657 ^ (1 / 3 : ℚ)) :=
by
  sorry

end mean_properties_l58_58464


namespace angle_measure_l58_58349

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l58_58349


namespace evaluate_expression_l58_58983

theorem evaluate_expression : 
  (let x := -2500 in abs (abs (abs x - x) - abs x) + x = 0) := 
by 
  let x := -2500
  sorry

end evaluate_expression_l58_58983


namespace apartments_in_each_complex_l58_58704

variable {A : ℕ}

theorem apartments_in_each_complex
    (h1 : ∀ (locks_per_apartment : ℕ), locks_per_apartment = 3)
    (h2 : ∀ (num_complexes : ℕ), num_complexes = 2)
    (h3 : 3 * 2 * A = 72) :
    A = 12 :=
by
  sorry

end apartments_in_each_complex_l58_58704


namespace value_of_f_5_l58_58018

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * Real.sin x - 2

theorem value_of_f_5 (a b : ℝ) (hf : f a b (-5) = 17) : f a b 5 = -21 := by
  sorry

end value_of_f_5_l58_58018


namespace binomial_sum_coeff_l58_58845

theorem binomial_sum_coeff (n : ℕ) :
  (∑ k in finset.range (n + 1), binomial n k) = 128 → n = 7 :=
begin
  sorry
end

end binomial_sum_coeff_l58_58845


namespace max_adjacent_differing_pairs_l58_58110

-- Definitions based on conditions
def grid : Type := ℕ × ℕ
def color := grid → bool  -- True represents black, False represents white

-- Conditions in the problem
def is_valid_grid (c : color) : Prop :=
  (∀ x, (Σ y, if c (x, y) then 1 else 0) = 50) ∧  -- Each column has the same number of black cells
  (∀ i j, i ≠ j → (Σ x, if c (x, i) then 1 else 0) ≠ (Σ x, if c (x, j) then 1 else 0))  -- Each row has a different number of black cells

-- Function to count adjacent differing pairs
def adjacent_differing_pairs (c : color) : ℕ :=
  Σ x y, if x < 99 ∧ c (x, y) ≠ c (x + 1, y) then 1 else 0 +  -- Horizontal differing pairs
         if y < 99 ∧ c (x, y) ≠ c (x, y + 1) then 1 else 0     -- Vertical differing pairs

-- Statement to prove the maximum number of adjacent differing pairs
theorem max_adjacent_differing_pairs :
  ∃ c : color, is_valid_grid c ∧ adjacent_differing_pairs c = 14751 :=
sorry

end max_adjacent_differing_pairs_l58_58110


namespace solution_part_one_solution_part_two_l58_58487

variable (x : ℝ)
variable (a b m : ℝ)
variable (h_ab_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_a : a = -2)
variable (h_b : b = 1)
variable (h_m : m = max 1 (max (abs a) (abs b)))

def f (x : ℝ) : ℝ := a / x + b / x^2

theorem solution_part_one : ∀ x : ℝ, |f x| < 1 ↔ x > 1 ∨ x < -2 := 
by { sorry }

theorem solution_part_two (h_x : |x| > m) : |f x| < 2 := 
by { sorry }

end solution_part_one_solution_part_two_l58_58487


namespace find_x0_l58_58106

def f (x : ℝ) : ℝ := x * (2016 + Real.log x)
def f' (x : ℝ) : ℝ := 2016 + Real.log x + 1

theorem find_x0 (x0 : ℝ) (h1 : f' x0 = 2017) : x0 = 1 :=
by
  sorry

end find_x0_l58_58106


namespace determine_a_l58_58984

variable (a : ℝ)
def f (x : ℝ) : ℝ := exp x - a * exp (-x)
def f'' (x : ℝ) : ℝ := (exp x + a * exp (-x))' -- This part ideally should be defined precisely but we'll refer it symbolically 

theorem determine_a (h : ∀ x : ℝ, f'' (-x) = -f'' x) : a = 1 := by
  sorry -- Proof omitted

end determine_a_l58_58984


namespace john_needs_to_study_4_hours_l58_58219

variable (hours_studied_1 hours_studied_2 score_1 score_2 : ℕ)

-- Conditions
def score_function (h : ℕ) : ℕ := 50 + 10 * h
def avg_score (s1 s2 : ℕ) : ℕ := (s1 + s2) / 2

-- Given data
def condition_1 : hours_studied_1 = 3 := rfl
def condition_2 : score_1 = 60 := rfl
def condition_3 : avg_goal : ℕ := 75
def condition_4 : score_1 = score_function hours_studied_1 := rfl

theorem john_needs_to_study_4_hours :
  avg_score score_1 score_2 = avg_goal →
  hours_studied_2 = 4 :=
begin
  intros h,
  sorry
end

end john_needs_to_study_4_hours_l58_58219


namespace count_integers_log_inequality_l58_58097

theorem count_integers_log_inequality :
  ∃ n : ℕ, n = 18 ∧ ∀ x : ℕ, 50 < x ∧ x < 70 → log 10 (x - 50) + log 10 (70 - x) < 1.5 :=
begin
  sorry
end

end count_integers_log_inequality_l58_58097


namespace length_OP_equals_4_l58_58886

-- Condition 1: The radius of circle O is 3.
def radius_circle (O : Type) [metric_space O] : ℝ := 3

-- Condition 2: Point P is outside circle O.
def point_outside_circle (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ) : Prop :=
  ∀ o ∈ O, ∀ p ∈ P, d o p > 3

-- Question: Prove the length OP is 4.
theorem length_OP_equals_4 
  (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ)
  (h_radius : radius_circle O = 3)
  (h_outside : point_outside_circle O P d) : 
  ∃ (p ∈ P) (o ∈ O), d o p = 4 :=
by
  sorry

end length_OP_equals_4_l58_58886


namespace decimal_to_base4_non_consecutive_digits_l58_58209

theorem decimal_to_base4_non_consecutive_digits : 
  ∀ (n : ℕ), n = 77 → 
  let b4_repr := nat_to_base4 n in
  digit_count_non_consecutive b4_repr = 3 :=
by
  intros,
  sorry

end decimal_to_base4_non_consecutive_digits_l58_58209


namespace conjugate_of_z_l58_58579

-- Given conditions
def z : ℂ := -1 + 2 * Complex.I

-- Proof statement
theorem conjugate_of_z : Complex.conj z = -1 - 2 * Complex.I :=
by
  sorry

end conjugate_of_z_l58_58579


namespace smallest_n_solution_l58_58711

def has_solution (n : ℕ) : Prop :=
  ∃ (x : Fin n → ℝ), 
    ( ∑ i in Finset.range n, Real.sin (x i) = 0 ) ∧
    ( ∑ i in Finset.range n, (i + 1) * Real.sin (x i) = 100 )

theorem smallest_n_solution : ∃ (n : ℕ), has_solution n ∧ (∀ m : ℕ, has_solution m → n ≤ m) :=
  sorry

end smallest_n_solution_l58_58711


namespace sequence_term_2012_l58_58126

theorem sequence_term_2012 :
  ∃ (a : ℕ → ℤ), a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2012 = 6 :=
sorry

end sequence_term_2012_l58_58126


namespace total_ingredients_l58_58328

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l58_58328


namespace math_problem_l58_58357

theorem math_problem :
  (1 / (1 / (1 / (1 / (3 + 2 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) = -13 / 9 :=
by
  -- proof goes here
  sorry

end math_problem_l58_58357


namespace least_milk_l58_58266

theorem least_milk (seokjin jungkook yoongi : ℚ) (h_seokjin : seokjin = 11 / 10)
  (h_jungkook : jungkook = 1.3) (h_yoongi : yoongi = 7 / 6) :
  seokjin < jungkook ∧ seokjin < yoongi :=
by
  sorry

end least_milk_l58_58266


namespace sum_of_x_and_y_l58_58504

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l58_58504


namespace find_line_equation_l58_58136

open Real

noncomputable def line_equation (x y : ℝ) (k : ℝ) : ℝ := k * x - y + 4 - 3 * k

noncomputable def distance_to_line (x1 y1 k : ℝ) : ℝ :=
  abs (k * x1 - y1 + 4 - 3 * k) / sqrt (k^2 + 1)

theorem find_line_equation :
  (∃ k : ℝ, (k = 2 ∨ k = -2 / 3) ∧
    (∀ x y, (x, y) = (3, 4) → (2 * x - y - 2 = 0 ∨ 2 * x + 3 * y - 18 = 0)))
    ∧ (line_equation (-2) 2 2 = line_equation 4 (-2) 2)
    ∧ (line_equation (-2) 2 (-2 / 3) = line_equation 4 (-2) (-2 / 3)) :=
sorry

end find_line_equation_l58_58136


namespace combine_ingredients_l58_58330

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l58_58330


namespace eraser_cost_l58_58787

noncomputable def price_of_erasers 
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  (bundle_count : ℝ) -- number of bundles sold
  (total_earned : ℝ) -- total amount earned
  (discount : ℝ) -- discount percentage for 20 bundles
  (bundle_contents : ℕ) -- 1 pencil and 2 erasers per bundle
  (price_ratio : ℝ) -- price ratio of eraser to pencil
  : Prop := 
  E = 0.5 * P ∧ -- The price of the erasers is 1/2 the price of the pencils.
  bundle_count = 20 ∧ -- The store sold a total of 20 bundles.
  total_earned = 80 ∧ -- The store earned $80.
  discount = 30 ∧ -- 30% discount for 20 bundles
  bundle_contents = 1 + 2 -- A bundle consists of 1 pencil and 2 erasers

theorem eraser_cost
  (P : ℝ) -- price of one pencil
  (E : ℝ) -- price of one eraser
  : price_of_erasers P E 20 80 30 (1 + 2) 0.5 → E = 1.43 :=
by
  intro h
  sorry

end eraser_cost_l58_58787


namespace trains_meet_distance_l58_58049

-- Definitions based on conditions
def firstTrainSpeed : ℕ := 30 -- speed of the first train in kmph
def secondTrainSpeed : ℕ := 40 -- speed of the second train in kmph
def firstTrainDepartureTime : ℕ := 9 -- departure time of the first train in hours
def secondTrainDepartureTime : ℕ := 14 -- departure time of the second train in hours

-- The main proof statement
theorem trains_meet_distance :
  let timeDifference := secondTrainDepartureTime - firstTrainDepartureTime,
      distanceCoveredByFirstTrainInThoseHours := firstTrainSpeed * timeDifference,
      relativeSpeed := secondTrainSpeed - firstTrainSpeed,
      timeToCatchUp := distanceCoveredByFirstTrainInThoseHours / relativeSpeed,
      meetDistance := secondTrainSpeed * timeToCatchUp
  in meetDistance = 600 :=
by
  let timeDifference := secondTrainDepartureTime - firstTrainDepartureTime
  let distanceCoveredByFirstTrainInThoseHours := firstTrainSpeed * timeDifference
  let relativeSpeed := secondTrainSpeed - firstTrainSpeed
  let timeToCatchUp := distanceCoveredByFirstTrainInThoseHours / relativeSpeed
  let meetDistance := secondTrainSpeed * timeToCatchUp
  sorry

end trains_meet_distance_l58_58049


namespace min_positive_S_n_is_19_l58_58500

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom arith_seq : ∀ n, a (n + 1) = a n + d
axiom rat_cond : a 11 / a 10 < -1
axiom sum_max_cond : ∃ n, S n = a 1 + ∑ i in (finset.range n), a (i + 1) ∧ (∀ m, S m ≤ S n)

-- To prove
theorem min_positive_S_n_is_19 (h : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_max : ∃ n, S n = a 1 + ∑ i in (finset.range n), a (i + 1) ∧ (∀ m, S m ≤ S n)) :
  ∃ n, n = 19 ∧ S n > 0 ∧ (∀ m, S m > 0 → S m ≥ S n) := 
sorry

end min_positive_S_n_is_19_l58_58500


namespace max_elements_l58_58248

def M : set ℕ := {x | 1 ≤ x ∧ x ≤ 2020}

def valid_subset (A : set ℕ) : Prop := ∀ x ∈ A, 4 * x ∉ A

theorem max_elements (A : set ℕ) (hA : A ⊆ M) (hValid : valid_subset A) : 
  ∃ B ⊆ M, valid_subset B ∧ B.card = 1616 :=
sorry

end max_elements_l58_58248


namespace vidya_mother_age_difference_l58_58340

theorem vidya_mother_age_difference :
  let vidya_age := 13
  let mother_age := 44
  mother_age - 3 * vidya_age = 5 :=
by
  let vidya_age := 13
  let mother_age := 44
  show mother_age - 3 * vidya_age = 5
  from sorry

end vidya_mother_age_difference_l58_58340


namespace eliot_account_balance_l58_58648

-- Definitions for the conditions
variables {A E : ℝ}

--- Conditions rephrased into Lean:
-- 1. Al has more money than Eliot.
def al_more_than_eliot (A E : ℝ) : Prop := A > E

-- 2. The difference between their two accounts is 1/12 of the sum of their two accounts.
def difference_condition (A E : ℝ) : Prop := A - E = (1 / 12) * (A + E)

-- 3. If Al's account were to increase by 10% and Eliot's account were to increase by 15%, 
--     then Al would have exactly $22 more than Eliot in his account.
def percentage_increase_condition (A E : ℝ) : Prop := 1.10 * A = 1.15 * E + 22

-- Prove the total statement
theorem eliot_account_balance : 
  ∀ (A E : ℝ), al_more_than_eliot A E → difference_condition A E → percentage_increase_condition A E → E = 146.67 :=
by
  intros A E h1 h2 h3
  sorry

end eliot_account_balance_l58_58648


namespace turnip_pulled_by_mice_l58_58008

theorem turnip_pulled_by_mice :
  ∀ (M B G D J C : ℕ),
    D = 2 * B →
    B = 3 * G →
    G = 4 * J →
    J = 5 * C →
    C = 6 * M →
    (D + B + G + J + C + M) ≥ (D + B + G + J + C) + M → 
    1237 * M ≤ (D + B + G + J + C + M) :=
by
  intros M B G D J C h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5]
  linarith

end turnip_pulled_by_mice_l58_58008


namespace malvina_card_value_sum_l58_58218

noncomputable def possible_values_sum: ℝ :=
  let value1 := 1
  let value2 := (-1 + Real.sqrt 5) / 2
  (value1 + value2) / 2

theorem malvina_card_value_sum
  (hx : ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ 
                 (x = Real.pi / 4 ∨ (Real.sin x = (-1 + Real.sqrt 5) / 2))):
  possible_values_sum = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end malvina_card_value_sum_l58_58218


namespace smallest_a₁_l58_58236

-- We define the sequence a_n and its recurrence relation
def a (n : ℕ) (a₁ : ℝ) : ℝ :=
  match n with
  | 0     => 0  -- this case is not used, but included for function completeness
  | 1     => a₁
  | (n+2) => 11 * a (n+1) a₁ - (n+2)

theorem smallest_a₁ : ∃ a₁ : ℝ, (a₁ = 21 / 100) ∧ ∀ n > 1, a n a₁ > 0 := 
  sorry

end smallest_a₁_l58_58236


namespace sum_of_coefficients_l58_58247

noncomputable def P : ℕ → Polynomial ℤ
| 0       := Polynomial.zero
| 1       := Polynomial.X ^ 2 - 1
| (n + 1) := Polynomial.comp (Polynomial.comp (P n) (Polynomial.X ^ 2 - 1) - (Polynomial.X ^ 2 - 1) ^ 2) (P (n - 1))

def coeff_sum_abs (p : Polynomial ℤ) : ℤ :=
(p.coeffs.map Int.natAbs).sum

open scoped Polynomial

theorem sum_of_coefficients (n : ℕ) :
  coeff_sum_abs (P n) = (Real.sqrt 2)⁻¹ * ((1 + Real.sqrt 2) ^ n - (1 - Real.sqrt 2) ^ n) :=
sorry

end sum_of_coefficients_l58_58247


namespace solve_equation_l58_58653

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2) ↔ (x = (Real.sqrt 6) / 3 ∨ x = -(Real.sqrt 6) / 3) :=
by sorry

end solve_equation_l58_58653


namespace math_problem_l58_58250

theorem math_problem (x y z: ℕ) 
  (h1: 0 < x) 
  (h2: 0 < y) 
  (h3: 0 < z) 
  (h4: 4 * Real.sqrt (Real.cbrt 7 - Real.cbrt 6) = Real.cbrt x + Real.cbrt y - Real.cbrt z) : 
  x + y + z = 75 := 
sorry

end math_problem_l58_58250


namespace parts_of_diagonal_in_rectangle_l58_58039

/-- Proving that a 24x60 rectangle divided by its diagonal results in 1512 parts --/

theorem parts_of_diagonal_in_rectangle :
  let m := 24
  let n := 60
  let gcd_mn := gcd m n
  let unit_squares := m * n
  let diagonal_intersections := m + n - gcd_mn
  unit_squares + diagonal_intersections = 1512 :=
by
  sorry

end parts_of_diagonal_in_rectangle_l58_58039


namespace area_of_circle_above_line_l58_58342

theorem area_of_circle_above_line (x y : ℝ) :
  (x^2 - 8 * x + y^2 - 10 * y + 25 = 0) ∧ (y = x + 3) →
  let r := 4 in
  let θ := real.pi in
  let area := real.pi * r^2 in
  (area / 2) = 8 * real.pi :=
by sorry

end area_of_circle_above_line_l58_58342


namespace solution_set_of_f_l58_58151

def f (x : ℝ) := x - (Real.exp 1 - 1) * Real.log x

theorem solution_set_of_f (x : ℝ) : (0 < x ∧ x < 1) ↔ (f (Real.exp x) < 1) :=
sorry

end solution_set_of_f_l58_58151


namespace unique_polynomial_representation_coefficients_calculation_l58_58368

noncomputable def binomial_polynomial (x : ℝ) (k : ℕ) : ℝ := 
  if k = 0 then 1 else x * binomial_polynomial (x - 1) (k - 1) / k

noncomputable def forward_difference (f : ℕ → ℝ) : ℕ → ℝ
| 0       := f 1 - f 0
| (n + 1) := forward_difference (λ i, forward_difference (λ j, f (j + i))) 0

theorem unique_polynomial_representation (f : ℝ → ℝ) (n : ℕ) (hf : ∀ x, polynomial.degree (polynomial.monomial n (f x)) = n)
  : ∃! d : (fin (n + 1) → ℝ), f(x) = ∑ k in finset.range (n + 1), (d k) * (binomial_polynomial x k) := sorry

theorem coefficients_calculation (f : ℝ → ℝ) (n : ℕ) (d : ℕ → ℝ) (hf : ∀ x, polynomial.degree (polynomial.monomial n (f x)) = n)
  (h_rep : ∀ x, f x = ∑ k in finset.range (n + 1), (d k) * (binomial_polynomial x k))
  : ∀ k ≤ n, d k = forward_difference (λ x, f x) k 0 := sorry

end unique_polynomial_representation_coefficients_calculation_l58_58368


namespace ratio_areas_of_octagons_l58_58202

theorem ratio_areas_of_octagons (side_len : ℝ) (h_side_len_pos : 0 < side_len)
  (AB PS RQ GH_parallel : Prop) (P Q R S : Point) 
  (h_P_on_BC : on_side P BC) (h_Q_on_DE : on_side Q DE)
  (h_R_on_FG : on_side R FG) (h_S_on_HA : on_side S HA)
  (equally_spaced : EquallySpaced AB PS RQ GH)
  (regular_octagon : RegularOctagon ABCDEFGH) :
  let area_original := area ABCDEFGH
  let area_smaller := area APBQCRDS
in area_smaller / area_original = 4 / 9 :=
by sorry

end ratio_areas_of_octagons_l58_58202


namespace smallest_possible_a_l58_58311

theorem smallest_possible_a
  (x₁ x₂ x₃ x₄ : ℕ)
  (h₁ : x^4 - (x₁ + x₂ + x₃ + x₄) * x^3 + _ = 0)
  (h₂ : x₁ * x₂ * x₃ * x₄ = 2520)
  (h₃ : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  : x₁ + x₂ + x₃ + x₄ = 29 :=
by
  sorry

end smallest_possible_a_l58_58311


namespace angle_measure_l58_58351

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l58_58351


namespace ratio_of_b_to_a_and_c_l58_58385

-- Define shares A, B, C
variables (A B C : ℕ)

-- Given conditions
def conditions : Prop :=
  A = 80 ∧ A + B + C = 200 ∧ A = (2 * (B + C)) / 3

-- The statement to be proved:
theorem ratio_of_b_to_a_and_c (h : conditions A B C) : B = (2 * (A + C)) / 3 :=
by
  -- Decompose the given conditions into individual parts for clarity
  cases h with hA hRest
  cases hRest with hTotal hRatio
  -- Use the sorry placeholder for the proof
  sorry

end ratio_of_b_to_a_and_c_l58_58385


namespace building_stories_l58_58251

-- Definitions based on the conditions in a)
def lola_time (n : ℕ) : ℕ := 10 * n
def tara_time (n : ℕ) : ℕ := 8 * n + 3 * (n - 1)

-- The theorem to prove
theorem building_stories : ∃ n : ℕ, tara_time n = 220 ∧ n = 20 :=
begin
  use 20,
  have h : tara_time 20 = 11 * 20 - 3 := by rfl,
  have h2 : 11 * 20 - 3 = 220 := by norm_num,
  simp [tara_time, h, h2],
  exact rfl,
end

end building_stories_l58_58251


namespace rational_coeff_count_l58_58091

theorem rational_coeff_count :
  let k : ℕ := 20 in
  let max_k : ℕ := 1250 in
  ∃ n : ℕ, (n - 1) * k = max_k - k → n = 63 :=
by
  sorry

end rational_coeff_count_l58_58091


namespace max_distance_from_center_is_area_of_coverage_ring_is_l58_58477

noncomputable def max_distance_from_center_equal_to (r : ℝ) (width : ℝ) : Prop :=
  r = 13 ∧ width = 10 → 
  real.angle.sin (real.pi / 5) ≠ 0 ∧
  ∃ max_distance : ℝ, max_distance = 12 / real.angle.sin (real.pi / 5)

noncomputable def area_of_coverage_ring (r : ℝ) (width : ℝ) : ℝ :=
  r = 13 ∧ width = 10 →
  real.angle.tan (real.pi / 5) ≠ 0 →
  π * (4 * (12 / real.angle.tan (real.pi / 5)) * 5)

theorem max_distance_from_center_is (r width : ℝ) :
  r = 13 ∧ width = 10 →
  max_distance_from_center_equal_to r width ∧
  max_distance r (width + r - r / 2) = 12 / (real.angle.sin (real.pi / 5)) :=
by
  sorry

theorem area_of_coverage_ring_is (r width max_distance : ℝ) :
  r = 13 ∧ width = 10 →
  area_of_coverage_ring r width = (240 * π) / (real.angle.tan (real.pi / 5)) :=
by
  sorry

end max_distance_from_center_is_area_of_coverage_ring_is_l58_58477


namespace ratio_flour_to_baking_soda_l58_58212

-- Define the given problem in Lean 4
theorem ratio_flour_to_baking_soda : 
  ∀ (sugar flour baking_soda : ℕ),
  let ratio_sf := 3,
      ratio_fs := 8,
      additional_baking_soda := 60,
      total_sugar := 900,
      total_flour := (total_sugar / ratio_sf) * ratio_fs,
      total_baking_soda := total_flour / (additional_baking_soda + (total_flour / 8)) - additional_baking_soda + 60
  in 
  total_flour / total_baking_soda = 10 :=
begin
  sorry
end

end ratio_flour_to_baking_soda_l58_58212


namespace quadratic_nonnegative_forall_x_l58_58838

theorem quadratic_nonnegative_forall_x (k : ℝ) :
  (∀ x : ℝ, x^2 - (k-2)*x - k + 8 ≥ 0) ↔ (-2*Real.sqrt 7 ≤ k ∧ k ≤ 2*Real.sqrt 7) :=
by
  sorry

end quadratic_nonnegative_forall_x_l58_58838


namespace number_of_valid_points_l58_58758

def A : ℝ × ℝ := (-4, 3)
def B : ℝ × ℝ := (4, -3)

def maxLength : ℝ := 18

def manhattan_distance (p q : ℝ × ℝ) : ℝ :=
  (abs (q.1 - p.1)) + (abs (q.2 - p.2))

def is_valid_point (xy : ℝ × ℝ) : Prop :=
  let x := abs xy.1
  let y := abs xy.2
  2 * x + 2 * y ≤ 10

def valid_integer_points : ℕ :=
  ((finset.range 6))
    .sum (λ x, 2 * (finset.card (finset.range ((6 - x.val)).succ)) - 1) - 1

theorem number_of_valid_points :
  valid_integer_points = 61 :=
sorry

end number_of_valid_points_l58_58758


namespace find_quotient_l58_58056

-- Definitions for the variables and conditions
variables (D d q r : ℕ)

-- Conditions
axiom eq1 : D = q * d + r
axiom eq2 : D + 65 = q * (d + 5) + r

-- Theorem statement
theorem find_quotient : q = 13 :=
by
  sorry

end find_quotient_l58_58056


namespace area_of_circular_segment_l58_58691

def a := 701
def b := 828.6
def c := 692

def s := (a + b + c) / 2
def K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
def R := (a * b * c) / (4 * K)
def phi := 2 * Real.arcsin (a / (2 * R))
def segment_area := ((phi * Real.pi / 360) - 1 / 2 * Real.sin phi) * R * R

theorem area_of_circular_segment :
  segment_area = 87646 := by
  sorry

end area_of_circular_segment_l58_58691


namespace trapezium_area_proof_l58_58086

-- Define the lengths of the parallel sides and the distance between them
def parallel_side1 : ℝ := 22
def parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15

-- Define the area of the trapezium
def trapezium_area (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- State the theorem to prove the given area
theorem trapezium_area_proof :
  trapezium_area parallel_side1 parallel_side2 distance_between_sides = 300 :=
by
  -- Skip the proof
  sorry

end trapezium_area_proof_l58_58086


namespace minute_hand_sweep_probability_l58_58998

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l58_58998


namespace percentage_of_4_or_more_years_is_37_5_l58_58028

def total_employees (y : ℕ) : ℕ :=
  3 * y + 4 * y + 7 * y + 6 * y + 3 * y + 3 * y + 1 * y + 2 * y + 1 * y + 1 * y + 1 * y

def employees_4_or_more_years (y : ℕ) : ℕ :=
  3 * y + 3 * y + 1 * y + 2 * y + 1 * y + 1 * y + 1 * y

def percentage_4_or_more_years (y : ℕ) : ℝ :=
  (employees_4_or_more_years y : ℝ) / (total_employees y : ℝ) * 100

theorem percentage_of_4_or_more_years_is_37_5 (y : ℕ) :
  percentage_4_or_more_years y = 37.5 := by
  sorry

end percentage_of_4_or_more_years_is_37_5_l58_58028


namespace PQRS_area_l58_58655

-- Define the conditions.
variables (LP NS x : ℝ)
variables (PQRS_is_square : ∃ (PQ QR RS SP : ℝ), PQ = x ∧ QR = x ∧ RS = x ∧ SP = x)
variables (similar_triangles : ∀ L M N P S R : ℝ, angle P R S = angle L M N ∧ angle R S P = angle M N L)

-- Define the lengths.
def LP_value := (LP = 32)
def NS_value := (NS = 64)

-- Theorem to prove the area of the square.
theorem PQRS_area (h₁ : LP_value) (h₂ : NS_value) (h₃ : PQRS_is_square) (h₄ : similar_triangles L M N P S R) :
  x^2 = 2048 := 
sorry

end PQRS_area_l58_58655


namespace quadratic_fixed_points_l58_58312

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
  a * x^2 + (3 * a - 1) * x - (10 * a + 3)

theorem quadratic_fixed_points (a : ℝ) (h : a ≠ 0) :
  quadratic_function a 2 = -5 ∧ quadratic_function a (-5) = 2 :=
by sorry

end quadratic_fixed_points_l58_58312


namespace leading_coefficient_l58_58090

-- Define the individual sub-polynomials
def poly1 := -5 * (λ x : ℝ, x^4 - 2*x^3 + 3*x)
def poly2 := 9 * (λ x : ℝ, x^4 - x + 4)
def poly3 := -6 * (λ x : ℝ, 3*x^4 - x^3 + 2)

-- Define the combined polynomial
def combined_poly := (λ x : ℝ, poly1 x + poly2 x + poly3 x)

-- Statement to prove
theorem leading_coefficient : 
  ∀ x : ℝ, 
  (combined_poly x).leading_coeff = -14 := 
by
  sorry

end leading_coefficient_l58_58090


namespace alternating_sum_of_coefficients_l58_58102

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x + 1)^5

theorem alternating_sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), polynomial_expansion x = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end alternating_sum_of_coefficients_l58_58102


namespace smallest_digit_d_l58_58092

theorem smallest_digit_d (d : ℕ) (hd : d < 10) :
  (∃ d, (20 - (8 + d)) % 11 = 0 ∧ d < 10) → d = 1 :=
by
  sorry

end smallest_digit_d_l58_58092


namespace triangle_area_l58_58214

theorem triangle_area (A B C : Type) [triangle A B C]
  (a b c : ℝ) (angle_A : ℝ) (h_angle_A : angle_A = 5 * Real.pi / 6)
  (h_b : b = 2) (h_c : c = 4)
  (S_ABC : ℝ) (h_area_def : S_ABC = 1/2 * b * c * Real.sin angle_A) :
  S_ABC = 2 :=
by
  rw [h_angle_A, h_b, h_c] at h_area_def
  sorry

end triangle_area_l58_58214


namespace part1_part2_l58_58576

noncomputable def Q (v : ℝ) : ℝ := 600 * v / (v^2 + 2 * v + 400)

theorem part1 (v : ℝ) (h1 : 0 < v ∧ v ≤ 25) :
    (Q v ≥ 10) → (8 ≤ v ∧ v ≤ 25) :=
sorry

theorem part2 (v : ℝ) (h1 : 0 < v ∧ v ≤ 25) :
    (∀ v', (0 < v' ∧ v' ≤ 25) → Q v' ≤ Q 20) ∧ (Q 20 ≈ 14.3) :=
sorry

end part1_part2_l58_58576


namespace bipartite_graph_all_vertices_even_degree_l58_58033

-- Define a bipartite graph
structure bipartite_graph (V : Type*) :=
(adj : V → V → Prop)
(color : V → Prop)
(coloring : ∀ v w : V, adj v w → color v ≠ color w)

-- Define the degree of a vertex
def degree {V : Type*} [fintype V] (G : bipartite_graph V) (v : V) : ℕ :=
fintype.card {w : V // G.adj v w}

-- The theorem statement: every vertex in a bipartite graph has an even degree
theorem bipartite_graph_all_vertices_even_degree {V : Type*} [fintype V] 
  (G : bipartite_graph V) :
  ∀ v : V, even (degree G v) :=
sorry

end bipartite_graph_all_vertices_even_degree_l58_58033


namespace factor_polynomial_l58_58841

def is_factor (p q : Polynomial ℤ) : Prop := ∃ r, q = p * r

theorem factor_polynomial (c d : ℤ) 
  (h_c : c = 1597) 
  (h_d : d = -2584) 
  (u v : ℤ) 
  (h_uv1 : u^2 - u - 1 = 0) 
  (h_uv2 : v^2 - v - 1 = 0) 
  (F : ℕ → ℤ) 
  (h_rec : ∀ n, F (n + 2) = F (n + 1) + F n)
  (F17 : F 17 = 1597)
  (F18 : F 18 = 2584)
  (F19 : F 19 = 4181) :
  is_factor (Polynomial.ofList [1, -1, -1]) (Polynomial.ofList [(c, 19), (d, 18), (1, 0)]) :=
  sorry

end factor_polynomial_l58_58841


namespace arithmetic_sequence_common_difference_l58_58230

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l58_58230


namespace length_OP_equals_4_l58_58885

-- Condition 1: The radius of circle O is 3.
def radius_circle (O : Type) [metric_space O] : ℝ := 3

-- Condition 2: Point P is outside circle O.
def point_outside_circle (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ) : Prop :=
  ∀ o ∈ O, ∀ p ∈ P, d o p > 3

-- Question: Prove the length OP is 4.
theorem length_OP_equals_4 
  (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ)
  (h_radius : radius_circle O = 3)
  (h_outside : point_outside_circle O P d) : 
  ∃ (p ∈ P) (o ∈ O), d o p = 4 :=
by
  sorry

end length_OP_equals_4_l58_58885


namespace solution_set_fxfplus5_neg_l58_58678

noncomputable def f : ℝ → ℝ := sorry

-- Definitions of the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_smooth : ∀ x, times_cont_diff_at ℝ ⊤ f x
axiom f_monotonic_increasing_neg : ∀ {a b : ℝ}, a ≤ b ∧ b ≤ -1 → f a ≤ f b
axiom f_monotonic_decreasing : ∀ {a b : ℝ}, a ≤ b ∧ -1 ≤ a ∧ b ≤ 1 → f a ≥ f b
axiom f_monotonic_increasing_pos : ∀ {a b : ℝ}, a ≤ b ∧ 1 ≤ a → f a ≤ f b
axiom f_3_zero : f 3 = 0

theorem solution_set_fxfplus5_neg : {x : ℝ | f x * f (x + 5) < 0} = {x : ℝ | (-8 < x ∧ x < -5) ∨ (-3 < x ∧ x < -2) ∨ (0 < x ∧ x < 3)} :=
sorry

end solution_set_fxfplus5_neg_l58_58678


namespace fourth_powers_count_l58_58917

theorem fourth_powers_count (n m : ℕ) (h₁ : n^4 ≥ 100) (h₂ : m^4 ≤ 10000) :
  ∃ k, k = m - n + 1 ∧ k = 7 :=
by
  sorry

end fourth_powers_count_l58_58917


namespace part1_l58_58868

-- Define the sequence of 0s and 1s
variables {n : ℕ} (x : fin n → ℕ)
variable (h : ∀ i, x i = 0 ∨ x i = 1)

/-- Define d_i according to the problem statement --/
def d_i (i : fin n) : ℕ := 
  (fin.sum ((fin.pre i) (λ j : fin i, if x j = x i then 1 else 0))) + 
  (fin.sum ((fin.suc i) (λ j : fin (n - i.succ), if x (i + j) ≠ x i then 1 else 0)))

/-- Define A according to the problem statement --/
def A (x : fin n → ℕ) : ℕ := 
  fin.sum ((fin.triplets_above n) (λ (i j k : fin n), if (x i = 0 ∧ x j = 1 ∧ x k = 0) ∨ (x i = 1 ∧ x j = 0 ∧ x k = 1) then 1 else 0))

/-- Statement of the theorem --/
theorem part1 : 
  A x = fin.choose 3 n - fin.sum ((fin.seq n) (λ i, fin.choose 2 (d_i i))) :=
sorry

end part1_l58_58868


namespace initial_capacity_l58_58757

theorem initial_capacity (x : ℝ) (h1 : 0.9 * x = 198) : x = 220 :=
by
  sorry

end initial_capacity_l58_58757


namespace option_C_correct_l58_58359

theorem option_C_correct {a : ℝ} : a^2 * a^3 = a^5 := by
  -- Proof to be filled
  sorry

end option_C_correct_l58_58359


namespace find_days_l58_58767

-- Definitions from conditions
variables (d s : ℕ)
constants (w p q : ℕ)

-- Total days
def total_days := w = 26

-- Pretzels per diligent day
def pretzels_diligent := p = 3

-- Pretzels per slacking day
def pretzels_slacking := q = 1

-- Final pretzels tally
def final_pretzels := 3 * d - s = 62

-- Calculate diligent and slacking days
theorem find_days (h1 : total_days) (h2 : pretzels_diligent) (h3 : pretzels_slacking) (h4 : final_pretzels) :
  d + s = 26 ∧ 3 * d - s = 62 := sorry

end find_days_l58_58767


namespace curve_C_equiv_ordinary_slope_line_l_BP_BQ_product_inequality_solution_set_range_a_l58_58427

-- Part A
def parametric_eqn_curve_C (θ : ℝ) := (x = 2 * Real.cos θ) ∧ (y = Real.sqrt 3 * Real.sin θ)

def ordinary_eqn_curve_C := ∀ (x y : ℝ), 
  (∃θ : ℝ, parametric_eqn_curve_C θ) ↔ (x^2 / 4 + y^2 / 3 = 1)

theorem curve_C_equiv_ordinary :
  ∀ x y, (∃ θ, parametric_eqn_curve_C θ) ↔ (x^2 / 4 + y^2 / 3 = 1) :=
sorry

def slope_line_l_polar (A B : ℝ × ℝ) : ℝ :=
  let (r1, θ1) := A; let (r2, θ2) := B;
  (-2 : ℝ)

theorem slope_line_l :
  slope_line_l_polar (Real.sqrt 2, Real.pi / 4) (3, Real.pi / 2) = -2 :=
sorry

def BP_BQ_product_line_l_intersect_C :=
  let t1 := -- root 1 of quadratic equation;
  let t2 := -- root 2 of quadratic equation;
  (Real.abs (t1 * t2))

theorem BP_BQ_product :
  BP_BQ_product_line_l_intersect_C = (120 / 19) :=
sorry

-- Part B
def f (x a : ℝ) := Real.abs (x - 1) + Real.abs (2 * x - a)

theorem inequality_solution_set (a : ℝ) :
  (∀ x, f x 1 ≥ 1) ↔ ((x < 1 / 2 ∧ x ≤ 1 / 3) ∨ (1 / 2 ≤ x ∧ x = 1) ∨ (x > 1 ∧ x ≥ 1 / 3)) :=
sorry

def range_a_for_f_ge_one :=
  ∀ x : ℝ, x ∈ [-1, 1] → f x a ≥ 1

theorem range_a :
  (range_a_for_f_ge_one 1) ↔ (a ≤ 0 ∨ a ≥ 3)
sorry

end curve_C_equiv_ordinary_slope_line_l_BP_BQ_product_inequality_solution_set_range_a_l58_58427


namespace volume_of_region_l58_58847

noncomputable def volume_region (x y z : ℝ) : ℝ :=
if |x + y + z| + |x - y + z| ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
then 62.5 else 0

theorem volume_of_region : 
  (∫ x in (0:ℝ)..5, ∫ y in (0:ℝ)..5, ∫ z in (0:ℝ)..(5 - x), 1) = 62.5 :=
begin
  sorry
end

end volume_of_region_l58_58847


namespace max_a_under_conditions_l58_58507

theorem max_a_under_conditions :
  ∀ (a b c d : ℝ), b + c + d = 3 - a ∧ 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2 → a ≤ 2 :=
by 
  intros a b c d h,
  sorry

example : ∃ (a : ℝ), ∀ (b c d : ℝ), b + c + d = 3 - a ∧ 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2 → a = 2 :=
by 
  use 2,
  intros b c d h,
  sorry

end max_a_under_conditions_l58_58507


namespace find_x_correct_l58_58225

noncomputable def find_x (n : ℕ) (a : Fin 2n → ℤ) : ℤ :=
  (Finset.univ.sum a) / (2 * n)

theorem find_x_correct (n : ℕ) (a : Fin 2n → ℤ) (h_distinct : Function.Injective a) :
  (∏ i in Finset.univ, (find_x n a - a i)) = (-1)^n * (n!)^2 := by
  sorry

end find_x_correct_l58_58225


namespace calculate_expression_l58_58062

theorem calculate_expression : 
  (let sqrt_12 := 2 * Real.sqrt 3
       tan_45 := 1
       sin_60 := Real.sqrt 3 / 2
       inv_2 := 2 in
   sqrt_12 + 2 * tan_45 - sin_60 - inv_2 = (3 * Real.sqrt 3) / 2) :=
by
  sorry

end calculate_expression_l58_58062


namespace part_I_geo_prog_part_I_general_formula_part_II_sum_sequence_l58_58493

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Condition: Given sequence a_n and S_n such that S_n = 2a_n - n
def satisfies_condition (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = 2 * a n - n

-- Theorem 1: The sequence {a_{n+1}} is a geometric progression.
theorem part_I_geo_prog (a : ℕ → ℤ) (S : ℕ → ℤ) (h : satisfies_condition a S) :
∀ n : ℕ, a (n + 1) = 2 * a n + 1 := sorry

-- Theorem 2: The general formula for the sequence {a_n} is a_n = 2^n - 1
theorem part_I_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h : satisfies_condition a S) :
∀ n : ℕ, a n = 2^(n : ℤ) - 1 := sorry

-- Definition of b_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
1 / a (n + 1) + 1 / (a n * a (n + 1))

-- Sum of the first n terms of the sequence {b_n}
def T (a : ℕ → ℤ) (n : ℕ) : ℤ := 
∑ i in finset.range n, b a i

-- Theorem 3: The sum of the first n terms T_n of the sequence {b_n} is T_n = 1 - 1 / (2^(n+1) - 1)
theorem part_II_sum_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : satisfies_condition a S) :
∀ n : ℕ, T a n = 1 - 1 / (2 ^ (n + 1 : ℤ) - 1) := sorry

end part_I_geo_prog_part_I_general_formula_part_II_sum_sequence_l58_58493


namespace machine_value_depletion_rate_l58_58400

theorem machine_value_depletion_rate :
  ∃ r : ℝ, 700 * (1 - r)^2 = 567 ∧ r = 0.1 := 
by
  sorry

end machine_value_depletion_rate_l58_58400


namespace coeff_x3_in_expansion_l58_58583

theorem coeff_x3_in_expansion :
  polynomial.coeff (expand_mul 5 (2 - X)) 3 = -40 :=
sorry

end coeff_x3_in_expansion_l58_58583


namespace train_speed_l58_58789

theorem train_speed
  (l_train : ℕ) (l_bridge : ℕ) (time : ℕ)
  (h_train : l_train = 100)
  (h_bridge : l_bridge = 300)
  (h_time : time = 30) :
  (l_train + l_bridge) / time = 13.33 :=
by
  sorry

end train_speed_l58_58789


namespace series_sum_eq_12_l58_58070

noncomputable def b : ℕ → ℝ
| 0     := 2
| 1     := 1
| (n+2) := (1/2) * b (n+1) + (1/3) * b n

theorem series_sum_eq_12 : (∑' n, b n) = 12 :=
sorry

end series_sum_eq_12_l58_58070


namespace total_area_of_squares_l58_58588

noncomputable theory

open_locale big_operators

-- Definitions based on the problem's conditions
def XY := 5
def XZ := 13
def angle_YXZ_is_right := true
def XY_square_area := XY^2
def YZ := Real.sqrt (XZ^2 - XY^2)
def YZ_square_area := YZ^2

-- The key theorem to prove
theorem total_area_of_squares :
  angle_YXZ_is_right → XY_square_area + YZ_square_area = 169 :=
begin
  intro h,
  have hXYAB : XY_square_area = 25, by norm_num,
  have hXZ_value : XZ = 13, by norm_num,
  have hYZ_value : YZ = 12, {
    simp [YZ, hXZ_value],
    norm_num,
  },
  have hYZ_square_area : YZ_square_area = YZ^2, by simp [YZ_square_area],
  rw hYZ_square_area,
  have hYZ_square_area_value : YZ_square_area = 144, by { norm_num, simp [hYZ_value], },
  calc
    XY_square_area + YZ_square_area = 25 + 144 : by { rw [hYZ_square_area_value, hXYAB], }
    ... = 169 : by norm_num,
end

end total_area_of_squares_l58_58588


namespace inequality_proof_l58_58970

open Real

theorem inequality_proof (n : ℕ) (a : ℕ → ℝ) (h : ∀ k < n, 0 < a k ∧ a (k + 1) - a k ≥ 1) :
  1 + (∑ k in finset.range (n+1), (1 / (a k - a 0))) ≤ ∏ k in finset.range (n+1), (1 + 1 / a k) :=
sorry

end inequality_proof_l58_58970


namespace Marc_average_speed_l58_58443

theorem Marc_average_speed
  (x : ℝ)
  (h_total_distance : 0 < x)
  (chantal_speed1 : ℝ := 3)
  (chantal_speed2 : ℝ := 1.5)
  (chantal_speed3 : ℝ := 2)
  (chantal_dist1 : ℝ := x)
  (chantal_dist2 : ℝ := 3 * x)
  (chantal_dist3 : ℝ := x)
  (marc_dist : ℝ := 3 * x) :
  let t1 := chantal_dist1 / chantal_speed1;
      t2 := chantal_dist2 / chantal_speed2;
      t3 := chantal_dist3 / chantal_speed3;
      T := t1 + t2 + t3
  in marc_dist / T = 18 / 17 := 
by {
  unfold chantal_dist1 chantal_speed1 chantal_speed2 chantal_speed3 chantal_dist2 chantal_dist3 marc_dist;
  simp only [div_eq_mul_inv];
  field_simp [chantal_speed1, chantal_speed2, chantal_speed3, t1, t2, t3];
  sorry
}

end Marc_average_speed_l58_58443


namespace problem1_problem2_l58_58441

-- Problem 1 Proof Statement
theorem problem1 : Real.sin (30 * Real.pi / 180) + abs (-1) - (Real.sqrt 3 - Real.pi) ^ 0 = 1 / 2 := 
  by sorry

-- Problem 2 Proof Statement
theorem problem2 (x: ℝ) (hx : x ≠ 2) : (2 * x - 3) / (x - 2) - (x - 1) / (x - 2) = 1 := 
  by sorry

end problem1_problem2_l58_58441


namespace late_fisherman_arrival_l58_58628

-- Definitions of conditions
variables (n d : ℕ) -- n is the number of fishermen on Monday, d is the number of days the late fisherman fished
variable (total_fish : ℕ := 370)
variable (fish_per_day_per_fisherman : ℕ := 10)
variable (days_fished : ℕ := 5) -- From Monday to Friday

-- Condition in Lean: total fish caught from Monday to Friday
def total_fish_caught (n d : ℕ) := 50 * n + 10 * d

theorem late_fisherman_arrival (n d : ℕ) (h : total_fish_caught n d = 370) : 
  d = 2 :=
by
  sorry

end late_fisherman_arrival_l58_58628


namespace problem_1_problem_2_l58_58470

-- For the first problem
theorem problem_1 :
  (2 + 1 / 4) ^ (1 / 2) - (-9.6) ^ 0 - (3 + 3 / 8) ^ (-2 / 3) + (1.5) ^ (-2) = 7 / 6 :=
by 
  sorry

-- For the second problem
theorem problem_2 :
  log 3 (real.sqrt 27 / 4) + log 10 25 + log 10 4 + log 2 25 * log 3 8 * log 5 9 = 55 / 4 :=
by 
  sorry

end problem_1_problem_2_l58_58470


namespace water_depth_l58_58428

theorem water_depth
  (ron_height : ℕ)
  (dean_taller_by : ℕ)
  (water_depth_factor : ℕ)
  (ron_height_value : ron_height = 13)
  (dean_taller_value : dean_taller_by = 4)
  (water_depth_factor_value : water_depth_factor = 15) :
  let dean_height := ron_height + dean_taller_by in
  let water_depth := water_depth_factor * dean_height in
  water_depth = 255 := by
  sorry

end water_depth_l58_58428


namespace steve_height_end_second_year_l58_58657

noncomputable def initial_height_ft : ℝ := 5
noncomputable def initial_height_inch : ℝ := 6
noncomputable def inch_to_cm : ℝ := 2.54

noncomputable def initial_height_cm : ℝ :=
  (initial_height_ft * 12 + initial_height_inch) * inch_to_cm

noncomputable def first_growth_spurt : ℝ := 0.15
noncomputable def second_growth_spurt : ℝ := 0.07
noncomputable def height_decrease : ℝ := 0.04

noncomputable def height_after_growths : ℝ :=
  let height_after_first_growth := initial_height_cm * (1 + first_growth_spurt)
  height_after_first_growth * (1 + second_growth_spurt)

noncomputable def final_height_cm : ℝ :=
  height_after_growths * (1 - height_decrease)

theorem steve_height_end_second_year : final_height_cm = 198.03 :=
  sorry

end steve_height_end_second_year_l58_58657


namespace distance_point_to_plane_correct_l58_58087

-- Definitions
variable (P : ℝ × ℝ × ℝ) (a b c d : ℝ)

-- Plane passes through the x-axis condition
def plane (x y z : ℝ) := a * x + b * y + c * z + d = 0

-- Third projection plane perpendicular to original plane condition
def third_projection_plane (x y z : ℝ) := -b * x + a * y = 0

-- Distance function to be proved
def distance_from_point_to_plane (P : ℝ × ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let z := P.3 in
  -- Calculations happen within this function
  sorry

-- Main theorem statement
theorem distance_point_to_plane_correct :
  ∃ d : ℝ, d = distance_from_point_to_plane P := sorry

end distance_point_to_plane_correct_l58_58087


namespace radius_tangent_to_three_semicircles_l58_58409

noncomputable def radius_tangent_circle (r1 r2 r3 : ℝ) : ℝ :=
  (6 / 7)

theorem radius_tangent_to_three_semicircles:
  ∀ (r1 r2 r3 : ℝ), r1 = 1 → r2 = 2 → r3 = 3 →
  radius_tangent_circle r1 r2 r3 = 6 / 7 :=
by
  intros r1 r2 r3 h1 h2 h3
  rw [h1, h2, h3]
  unfold radius_tangent_circle
  simp
  exact rfl

end radius_tangent_to_three_semicircles_l58_58409


namespace simplify_expression_l58_58468

theorem simplify_expression :
  (∃ (x : Real), x = 3 * (Real.sqrt 3 + Real.sqrt 7) / (4 * Real.sqrt (3 + Real.sqrt 5)) ∧ 
    x = Real.sqrt (224 - 22 * Real.sqrt 105) / 8) := sorry

end simplify_expression_l58_58468


namespace interval_of_increase_shift_sine_l58_58077

theorem interval_of_increase_shift_sine (k : ℤ) :
    let f := λ x : ℝ, Real.sin (2 * x)
    let g := λ x : ℝ, f (x - (Real.pi / 6))
    ∀ x : ℝ, k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 ↔ 
             g x = Real.sin (2 * x - Real.pi / 3) := 
by
  sorry

end interval_of_increase_shift_sine_l58_58077


namespace probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l58_58393

-- Definitions based on the conditions laid out in the problem
def fly_paths (n_right n_up : ℕ) : ℕ :=
  (Nat.factorial (n_right + n_up)) / ((Nat.factorial n_right) * (Nat.factorial n_up))

-- Probability for part a
theorem probability_at_8_10 : 
  (fly_paths 8 10) / (2 ^ 18) = (Nat.choose 18 8 : ℚ) / 2 ^ 18 := 
sorry

-- Probability for part b
theorem probability_at_8_10_through_5_6 :
  ((fly_paths 5 6) * (fly_paths 1 0) * (fly_paths 2 4)) / (2 ^ 18) = (6930 : ℚ) / 2 ^ 18 :=
sorry

-- Probability for part c
theorem probability_at_8_10_within_circle :
  (2 * fly_paths 2 7 * fly_paths 6 3 + 2 * fly_paths 3 6 * fly_paths 5 3 + (fly_paths 4 6) ^ 2) / (2 ^ 18) = 
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + (Nat.choose 9 4) ^ 2 : ℚ) / 2 ^ 18 :=
sorry

end probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l58_58393


namespace pyramid_coloring_count_l58_58809

-- The definitions directly follow from the conditions.
def coloring_vertices_pyramid (colors : Finset ℕ) (vertices : Finset (ℕ × ℕ)) : Prop :=
  ∀ (v₁ v₂ : ℕ), (v₁, v₂) ∈ vertices → (v₁ ≠ v₂ ∧ ∀ (c₁ c₂ : ℕ), c₁ ∈ colors ∧ c₂ ∈ colors → c₁ ≠ c₂)

noncomputable def count_colorings_pyramid : ℕ :=
  let colors := {1, 2, 3, 4, 5} in
  let vertices := { (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1) } in
  if h : coloring_vertices_pyramid colors vertices then
    150
  else
    0

theorem pyramid_coloring_count : count_colorings_pyramid = 150 :=
by
  sorry

end pyramid_coloring_count_l58_58809


namespace probability_two_asian_countries_probability_A1_not_B1_l58_58790

-- Scope: Definitions for the problem context
def countries : List String := ["A1", "A2", "A3", "B1", "B2", "B3"]

-- Probability of picking two Asian countries from a pool of six (three Asian, three European)
theorem probability_two_asian_countries : 
  (3 / 15) = (1 / 5) := by
  sorry

-- Probability of picking one country from the Asian group and 
-- one from the European group, including A1 but not B1
theorem probability_A1_not_B1 : 
  (2 / 9) = (2 / 9) := by
  sorry

end probability_two_asian_countries_probability_A1_not_B1_l58_58790


namespace a4_value_l58_58958

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 1 / (a (n-1)) + 1)

theorem a4_value : ∀ a : ℕ → ℚ,
  sequence a → a 4 = 5 / 3 :=
by
  intros a h
  sorry

end a4_value_l58_58958


namespace values_of_a_b_l58_58846

noncomputable def a : ℝ := 2 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 2 - 2 * Real.sqrt 5

theorem values_of_a_b (h1 : (a * a - 4 * a + 9 = 25) ∧ (b * b - 4 * b + 9 = 25)) (h2 : a ≥ b) : 
  3 * a + 2 * b = 10 + 2 * Real.sqrt 5 :=
begin
  sorry
end

end values_of_a_b_l58_58846


namespace product_units_digit_mod_10_l58_58354

theorem product_units_digit_mod_10 :
  let a := 8623
  let b := 2475
  let c := 56248
  let d := 1234
  (a * b * c * d) % 10 = 0 := by
  let ua := a % 10 -- units digit of a
  let ub := b % 10 -- units digit of b
  let uc := c % 10 -- units digit of c
  let ud := d % 10 -- units digit of d
  have : ua = 3 := by rfl
  have : ub = 5 := by rfl
  have : uc = 8 := by rfl
  have : ud = 4 := by rfl
  have units_product_mod_10 : (ua * ub * uc * ud) % 10 = 0 := by sorry
  show (a * b * c * d) % 10 = 0 from by sorry

end product_units_digit_mod_10_l58_58354


namespace two_dice_probability_sum_greater_than_eight_l58_58718

def probability_sum_greater_than_eight : ℚ := 5 / 18

theorem two_dice_probability_sum_greater_than_eight :
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let favorable := [(3,6), (4,5), (4,6), (5,4), (5,5), (5,6), (6,3), (6,4), (6,5), (6,6)] in
  outcomes.length = 36 ∧ favorable.length = 10 →
  favorable.length / outcomes.length = probability_sum_greater_than_eight := 
by
  intros
  sorry

end two_dice_probability_sum_greater_than_eight_l58_58718


namespace extreme_points_correct_l58_58490

variables (R : Type*) [OrderedSemiring R] [TopologicalSpace R] [has_sin R] [has_cos R] [has_tan R]

noncomputable def f (x λ : R) : R := x * sin x - λ * cos x

theorem extreme_points_correct (λ : R) (k : ℕ) (h : λ > -1) :
  (∃ n, list.of_fn (λ i, f' (x i)) = list.repeat 0 n).length = 2*(k+1) ∧ 
  (x k+1 = 0) ∧ 
  (k = 1 → ∃ seq : list R, arithmetic_seq seq) :=
sorry

end extreme_points_correct_l58_58490


namespace equilateral_triangle_perimeter_l58_58286

theorem equilateral_triangle_perimeter (s : ℝ) (h1 : s ≠ 0) (h2 : (s ^ 2 * real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * real.sqrt 3 := 
by
  sorry

end equilateral_triangle_perimeter_l58_58286


namespace palindrome_pair_count_is_36_l58_58403

-- Define what it means to be a four-digit palindrome
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (a < 10) ∧ (b < 10) ∧ n = 1000 * a + 100 * b + 10 * b + a

-- Define what it means to be a five-digit palindrome
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (e f : ℕ), (e < 10) ∧ (f < 10) ∧ n = 10000 * e + 1000 * f + 100 * e + 10 * f + e

-- Define the property of being a sum of two specific four-digit palindromes forming a five-digit palindrome
def valid_palindrome_pair_count : ℕ :=
  {n1 n2 : ℕ // is_four_digit_palindrome n1 ∧ is_four_digit_palindrome n2 ∧ 
   is_five_digit_palindrome (n1 + n2)}.to_finset.card

-- State that the number is 36
theorem palindrome_pair_count_is_36 : valid_palindrome_pair_count = 36 :=
by
  sorry

end palindrome_pair_count_is_36_l58_58403


namespace cone_volume_correct_l58_58440

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume_correct :
  cone_volume 6 5 = 60 * π :=
by
  sorry

end cone_volume_correct_l58_58440


namespace no_real_b_for_inequality_l58_58075

theorem no_real_b_for_inequality : ¬ ∃ b : ℝ, (∃ x : ℝ, |x^2 + 3 * b * x + 4 * b| = 5 ∧ ∀ y : ℝ, y ≠ x → |y^2 + 3 * b * y + 4 * b| > 5) := sorry

end no_real_b_for_inequality_l58_58075


namespace people_at_first_concert_l58_58990

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l58_58990


namespace sum_of_inverses_l58_58243

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2 + 1

theorem sum_of_inverses : 
  let f_inv_neg1 := if 2 < 1 + Real.sqrt 3 then 1 + Real.sqrt 3 else 0 in
  let f_inv_1 := if 2 = 2 then 2 else 0 in
  let f_inv_5 := if 2 < 1 + Real.sqrt 5 then 1 + Real.sqrt 5 else 0 in
  f_inv_neg1 + f_inv_1 + f_inv_5 = 4 + Real.sqrt 3 + Real.sqrt 5 := 
by 
  sorry

end sum_of_inverses_l58_58243


namespace min_triangle_area_is_three_fourths_l58_58693

-- Define the equation z - 4)^6 = 64
def is_solution (z : ℂ) := (z - 4)^6 = 64

-- Define the vertices D, E, F which are the solutions to the equation
noncomputable def vertices (k : ℕ) (h : k < 6) := 4 + 2 * complex.exp (2 * real.pi * complex.I * k / 6)

-- Define the minimum area of the triangle formed by three consecutive vertices
noncomputable def min_area_of_triangle_DEF : ℂ :=
  let D := vertices 0 (nat.zero_lt_succ 5)
  let E := vertices 1 (nat.succ_lt_succ nat.zero_lt_succ_succ)
  let F := vertices 2 (nat.succ_lt_succ (nat.succ_lt_succ (nat.zero_lt_succ_succ)))
  in complex.abs (D - E) * complex.abs (E - F) / 2

-- Theorem to be proved
theorem min_triangle_area_is_three_fourths : min_area_of_triangle_DEF = 3 / 4 := 
by
  -- Proof will be filled here
  sorry

end min_triangle_area_is_three_fourths_l58_58693


namespace smallest_positive_period_increasing_intervals_l58_58896

noncomputable theory

def f (x : ℝ) : ℝ := sin (x / 2) + sqrt 3 * cos (x / 2)

-- Proof Problem 1: Prove that the smallest positive period T of the function f(x) is 4π.
theorem smallest_positive_period :
  (∃ T > 0, ∀ (x : ℝ), f (x + T) = f x) ∧ T = 4 * real.pi :=
sorry

-- Proof Problem 2: Prove that the function f(x) is increasing on the interval [-5π/3, π/3] within [-2π, 2π].
theorem increasing_intervals :
  ∀ (x : ℝ), -2 * real.pi ≤ x ∧ x ≤ 2 * real.pi →
             -5 * real.pi / 3 ≤ x ∧ x ≤ real.pi / 3 →
             (∀ y z, x ≤ y ∧ y < z ∧ z ≤ 2 * real.pi → f y < f z) :=
sorry

end smallest_positive_period_increasing_intervals_l58_58896


namespace part_I_part_II_part_III_l58_58139

section
variable {f : ℝ → ℝ}
variable (hf : ∀ x > 0, HasDerivAt f (f' x) x)
variable (hf' : ∀ x > 0, f' x > f x / x)

-- Define F(x) = f(x) / x
def F (x : ℝ) : ℝ := f x / x

-- Part I: Show that F(x) is increasing on (0, +∞)
theorem part_I (x : ℝ) (hx : x > 0) : HasDerivAt F (f' x * x - f x) x →
  ∀ x > 0, (f x / x) < (f y / y) → x < y :=
sorry

-- Part II: Show that ∀ x1 x2 ∈ (0, +∞), f(x1) + f(x2) < f(x1 + x2)
theorem part_II (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) :
  f x1 + f x2 < f (x1 + x2) :=
sorry

-- Part III: Generalize the conclusion of (II) to any n ≥ 2
theorem part_III (n : ℕ) (h : n ≥ 2) (xs : Fin n → ℝ) (hxs : ∀ i, xs i > 0) :
  ∑ i, f (xs i) < f (∑ i, xs i) :=
sorry
end

end part_I_part_II_part_III_l58_58139


namespace Tino_jellybeans_l58_58710

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l58_58710


namespace hyperbola_eccentricity_l58_58688

variables {a b c m n : ℝ}
variables {O F B A : ℝ × ℝ}
variables (h_hyperbola : ∀ (x y : ℝ), (x / a) ^ 2 - (y / b) ^ 2 = 1)
variables (h_focus : F = (c, 0))
variables (h_point_B : B = (m, n))
variables (h_intersect_asymptote : ∃ (A : ℝ × ℝ), BF_intersects_asymptote_at_A A)
variables (h_dot_product : (O - B − F) • A = 0)
variables (h_line : 2 * A = B + F)

/-- The eccentricity of a hyperbola given the conditions -/
theorem hyperbola_eccentricity :
  c = sqrt 5 * a → ecc = sqrt 5 :=
sorry

end hyperbola_eccentricity_l58_58688


namespace simplify_sqrt_144000_l58_58271

theorem simplify_sqrt_144000 :
  (sqrt 144000 = 120 * sqrt 10) :=
by
  -- Assume given conditions
  have h1 : 144000 = 144 * 1000 := by
    calc 144000 = 144 * 1000 : by rfl

  have h2 : 144 = 12^2 := by rfl

  have h3 : sqrt (a * b) = sqrt a * sqrt b := by sorry

  have h4 : sqrt (10 ^ 3) = 10 * sqrt 10 := by sorry

  -- Prove the target
  calc
    sqrt 144000
    = sqrt (144 * 1000) : by rw [←h1]
    = sqrt (12^2 * 10^3) : by rw [h2, pow_succ]
    = sqrt (12^2) * sqrt (10^3) : by rw [h3]
    = 12 * sqrt (10^3) : by rw [sqrt_sq', h2, pow_two]
    = 12 * (10 * sqrt 10) : by rw [h4]
    = 12 * 10 * sqrt 10 : by rw [mul_assoc]
    = 120 * sqrt 10 : by sorry

-- sqrt_sq' and pow_two are used to simplify sqrt (12^2) == 12.

end simplify_sqrt_144000_l58_58271


namespace correct_number_of_inequalities_l58_58798

noncomputable def inequality1 : Prop :=
  ∫ x in 0..1, Real.sqrt x < ∫ x in 0..1, 3 * x

noncomputable def inequality2 : Prop :=
  ∫ x in 0..(Real.pi / 4), Real.sin x < ∫ x in 0..(Real.pi / 4), Real.cos x

noncomputable def inequality3 : Prop :=
  ∫ x in 0..1, Real.exp (-x) < ∫ x in 0..1, Real.exp (-(x^2))

noncomputable def inequality4 : Prop :=
  ∫ x in 0..2, Real.sin x < ∫ x in 0..2, x

theorem correct_number_of_inequalities : ([
  inequality1,
  inequality2,
  inequality3,
  inequality4
].count (λ P, P)) = 4 := by
  sorry

end correct_number_of_inequalities_l58_58798


namespace cos_double_angle_formula_l58_58860

theorem cos_double_angle_formula (α : ℝ) 
  (h : sin (π / 6 - α) - cos α = 1 / 3) : 
  cos (2 * α + π / 3) = 7 / 9 :=
by sorry

end cos_double_angle_formula_l58_58860


namespace generate_func_eq_l58_58375

noncomputable def generatingFunction (G : ℂ → ℂ) : Prop :=
  ∀ s : ℂ, abs s < 1 → 
    G s = 1 - exp (-∑' (n : ℕ) in {n | n ≥ 1}, (n⁻¹ : ℂ) * (𝒫 (λ (s_n : ℝ), s_n > 0) * s^n))

theorem generate_func_eq (X : ℕ → ℝ) (S : ℕ → ℝ) (τ : ℕ → ℝ) 
  (h_indep : ∀ i j, i ≠ j → independent (X i) (X j))
  (h_iid : ∀ i, X i = X 1)
  (h_S_def : ∀ n, S n = ∑ i in finset.range n, X i)
  (h_τ_zero : τ 0 = 0)
  (h_τ_recursive : ∀ k ≥ 1, τ k = inf {n | n > τ (k-1) ∧ S n > S (τ (k-1))})
  : generatingFunction (λ s : ℂ, ∑' (n : ℕ) in {n | n ≥ 1}, (𝒫 (λ (k : ℕ), (S 1 ≤ 0 ∧ S 2 ≤ 0 ... ∧ S (n-1) ≤ 0 ∧ S n) > 0) * s^n)) :=
begin
  sorry
end

end generate_func_eq_l58_58375


namespace count_dna_sequences_Rthea_l58_58265

-- Definition of bases
inductive Base | H | M | N | T

-- Function to check whether two bases can be adjacent on the same strand
def can_be_adjacent (x y : Base) : Prop :=
  match x, y with
  | Base.H, Base.M => False
  | Base.M, Base.H => False
  | Base.N, Base.T => False
  | Base.T, Base.N => False
  | _, _ => True

-- Function to count the number of valid sequences
noncomputable def count_valid_sequences : Nat := 12 * 7^4

-- Theorem stating the expected count of valid sequences
theorem count_dna_sequences_Rthea : count_valid_sequences = 28812 := by
  sorry

end count_dna_sequences_Rthea_l58_58265


namespace total_triangles_in_figure_l58_58919

-- Definitions directly from the conditions
def triangular_grid_rows := [1, 2, 3, 4]

-- Theorem statement to prove that the total number of triangles is 17
theorem total_triangles_in_figure : 
    (let small_triangles := list.sum triangular_grid_rows in
     let medium_triangles := 6 in
     let large_triangles := 1 in
     small_triangles + medium_triangles + large_triangles = 17) :=
by 
  let small_triangles := list.sum triangular_grid_rows
  let medium_triangles := 6
  let large_triangles := 1
  have h : small_triangles = 10 := sorry
  have total_triangles := small_triangles + medium_triangles + large_triangles
  show total_triangles = 17
  sorry

end total_triangles_in_figure_l58_58919


namespace concurrency_AD_l58_58386

open scoped Classical

variables {A B C D D' E E' F F' : Type} [EuclideanGeometry A] 

-- Points and concurrency condition
variables (triangleABC : Triangle A B C) 
          (circle : Circle) 
          (D D' : A) (E E' : A) (F F' : A)
          (hD : circle.Intersects_line (Line_Segment B C) D ∧ circle.Intersects_line (Line_Segment B C) D')
          (hE : circle.Intersects_line (Line_Segment C A) E ∧ circle.Intersects_line (Line_Segment C A) E')
          (hF : circle.Intersects_line (Line_Segment A B) F ∧ circle.Intersects_line (Line_Segment A B) F')
          (h_concurrent : Concurrent (Line_Through A D) (Line_Through B E) (Line_Through C F))

-- Statement to prove concurrency of AD', BE', and CF'
theorem concurrency_AD'_BE'_CF' : 
  Concurrent (Line_Through A D') (Line_Through B E') (Line_Through C F') :=
sorry

end concurrency_AD_l58_58386


namespace new_ticket_price_l58_58683

theorem new_ticket_price (P : ℕ) (num_spectators_increase rate_revenue_increase : ℚ)
(initial_amount_spectators initial_revenue new_revenue num_spectators : ℚ)
(hP : P = 400)
(hSpectatorsIncrease : num_spectators_increase = 0.25)
(hRevenueIncrease : rate_revenue_increase = 0.125)
(hInitialSpectators : initial_amount_spectators = 1)
(hInitialRevenue : initial_revenue = P * initial_amount_spectators)
(hNewSpectators : num_spectators = initial_amount_spectators * (1 + num_spectators_increase))
(hNewRevenue : new_revenue = initial_revenue * (1 + rate_revenue_increase))
(hNewRevenueValue : new_revenue = 450) : 
  (new_price : ℚ) -> (num_spectators * new_price = new_revenue) -> new_price = 360 :=
by 
  intros
  have h : new_price * 1.25 = 450 := by linarith [hNewSpectators, hNewRevenue]
  have h2 : new_price = (450 / 1.25) := by linarith
  /- we provide the expected value -/
  exact eq_of_mul_eq_mul_right (by norm_num) (h.trans h2.symm)

end new_ticket_price_l58_58683


namespace ratio_a_to_b_l58_58554

theorem ratio_a_to_b (a b : ℝ) (h : (a - 3 * b) / (2 * a - b) = 0.14285714285714285) : a / b = 4 :=
by 
  -- The proof goes here
  sorry

end ratio_a_to_b_l58_58554


namespace smallest_third_term_geometric_l58_58411

theorem smallest_third_term_geometric (d : ℝ) : 
  (∃ d, (7 + d) ^ 2 = 4 * (26 + 2 * d)) → ∃ g3, (g3 = 10 ∨ g3 = 36) ∧ g3 = min (10) (36) :=
by
  sorry

end smallest_third_term_geometric_l58_58411


namespace polynomial_remainder_l58_58455

def p : ℝ[X] := X^3 - 3*X + 1
def d : ℝ[X] := X^2 - X - 2
def r : ℝ[X] := -X^2 - X + 1

theorem polynomial_remainder :
  (p % d) = r := 
by sorry

end polynomial_remainder_l58_58455


namespace smallest_positive_integer_x_l58_58725

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end smallest_positive_integer_x_l58_58725


namespace triangle_DEF_area_l58_58026

noncomputable def point := ℝ × ℝ

structure Circle where
  center : point
  radius : ℝ

structure Triangle where
  vertex1 : point
  vertex2 : point
  vertex3 : point

def tangent (p : point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx) ^ 2 + (y - cy) ^ 2 = c.radius ^ 2

def congruent (a b : point) : Prop :=
  let (ax, ay) := a
  let (bx, by) := b
  (ax - bx) ^ 2 + (ay - by) ^ 2 = 0

def area (t : Triangle) : ℝ :=
  let (x1, y1) := t.vertex1
  let (x2, y2) := t.vertex2
  let (x3, y3) := t.vertex3
  1 / 2 * abs (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1)

theorem triangle_DEF_area :
  ∀ (c1 c2 : Circle) (t : Triangle),
    c1.radius = 2 →
    c2.radius = 3 →
    tangent t.vertex1 c1 →
    tangent t.vertex2 c1 →
    tangent t.vertex1 c2 →
    tangent t.vertex2 c2 →
    congruent t.vertex2 t.vertex3 →
    area t = 30 * real.sqrt 21 := 
by
  intros
  -- sorry provides a placeholder where the proof logic would be implemented.
  sorry

end triangle_DEF_area_l58_58026


namespace f_f_2_eq_neg_half_l58_58895

def f (x : ℝ) : ℝ :=
if x ≥ 0 then -x^2 - x - 2
else x / (x + 4) + Real.logb 4 (abs x)

theorem f_f_2_eq_neg_half : f (f 2) = -1/2 :=
by
  sorry

end f_f_2_eq_neg_half_l58_58895


namespace quadratic_root_zero_k_neg2_l58_58561

theorem quadratic_root_zero_k_neg2
  (k : ℝ)
  (f : ℝ → ℝ)
  (h : f = λ x, (k - 2) * x^2 + x + k^2 - 4)
  (hz : f 0 = 0)
  (hk : k - 2 ≠ 0) : k = -2 :=
by
  sorry

end quadratic_root_zero_k_neg2_l58_58561


namespace base7_to_base10_div_l58_58278

theorem base7_to_base10_div (x y : ℕ) (h : 546 = x * 10^2 + y * 10 + 9) : (x + y + 9) / 21 = 6 / 7 :=
by {
  sorry
}

end base7_to_base10_div_l58_58278


namespace time_sum_l58_58961

def addTime : ℕ → ℕ × ℕ → ℕ × ℕ × ℕ
| t, (h,m) => let h' := h + t % 12
              (h' % 12, m, (t / 12) * 60)

theorem time_sum (hours_add minutes_add seconds_add : ℕ) :
  let current_hour := 3
  let current_minute := 0
  let current_second := 0
  let new_hour := (current_hour + (hours_add % 12)) % 12
  let new_minute := (current_minute + minutes_add) % 60
  let new_second := (current_second + seconds_add) % 60
  new_hour + new_minute + new_second = 100 :=
by
  let hours_add := 315
  let minutes_add := 58
  let seconds_add := 36
  let current_hour := 3
  let current_minute := 0
  let current_second := 0
  let new_hour := (current_hour + (hours_add % 12)) % 12
  let new_minute := (current_minute + minutes_add) % 60
  let new_second := (current_second + seconds_add) % 60
  show (new_hour + new_minute + new_second = 100)
  -- Here we should put steps from the solution to prove the theorem.
  sorry

end time_sum_l58_58961


namespace rationalize_denominator_l58_58646

theorem rationalize_denominator :
  (∃ A B C D : ℤ, (D > 0) ∧ ∀ p : ℕ, prime p → ¬ (p^2 ∣ B) ∧ 
  ( (sqrt 50) / (sqrt 25 - sqrt 5) = (A * sqrt B + C) / D ) ∧ 
  (A + B + C + D = 12)) :=
sorry

end rationalize_denominator_l58_58646


namespace college_students_not_enrolled_l58_58551

variable (total_students : ℕ)
variable (bio_percentage : ℚ)
variable (chem_percentage : ℚ)
variable (both_percentage : ℚ)

def students_in_bio (total_students : ℕ) (bio_percentage : ℚ) : ℕ :=
  (total_students : ℚ) * bio_percentage

def students_in_chem (total_students : ℕ) (chem_percentage : ℚ) : ℕ :=
  (total_students : ℚ) * chem_percentage

def students_in_both (total_students : ℕ) (both_percentage : ℚ) : ℕ :=
  (total_students : ℚ) * both_percentage

def students_in_either (total_students : ℕ) (bio_percentage : ℚ) (chem_percentage : ℚ) (both_percentage : ℚ) : ℕ :=
  students_in_bio total_students bio_percentage + students_in_chem total_students chem_percentage - students_in_both total_students both_percentage

def students_not_enrolled (total_students : ℕ) (bio_percentage : ℚ) (chem_percentage : ℚ) (both_percentage : ℚ) : ℕ :=
  total_students - students_in_either total_students bio_percentage chem_percentage both_percentage

theorem college_students_not_enrolled : students_not_enrolled 880 0.40 0.30 0.10 = 352 :=
by
  sorry

end college_students_not_enrolled_l58_58551


namespace measure_angle_RPQ_l58_58956

-- Given conditions
variable (P Q R S : Type) (PQ PR: ℝ) (x : ℝ)
variable (angle_RSQ angle_RPQ angle_RQS : ℝ)
variable (is_bisector_QP_SQR : Bool)

-- Definitions and constraints based on conditions
def condition1 : Prop := PQ = PR
def condition2 : Prop := is_bisector_QP_SQR = true
def condition3 : Prop := angle_RSQ = x
def condition4 : Prop := angle_RPQ = 2 * x + 10
def condition5 : Prop := angle_RQS = 3 * x

-- Main statement: the proof goal
theorem measure_angle_RPQ : condition1 -> condition2 -> condition3 -> condition4 -> condition5 -> angle_RPQ = 66.67 :=
by
  intros
  sorry

end measure_angle_RPQ_l58_58956


namespace tino_jellybeans_l58_58705

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l58_58705


namespace total_spent_l58_58793

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l58_58793


namespace reaction_rate_unchanged_l58_58573

theorem reaction_rate_unchanged :
  ∀ (Fe₂O₃ H₂ N₂ : Type) (amount_Fe₂O₃ amount_H₂ internal_pressure : ℝ)
    (volume_constant catalyst : Prop),
  (amount_Fe₂O₃ ∈ Fe₂O₃ ∧ amount_H₂ ∈ H₂ ∧ 
   volume_constant ∧ catalyst ∧ 
   internal_pressure ∈ N₂) → 
  (change_rate (increase amount_Fe₂O₃) = no_change) :=
by sorry

end reaction_rate_unchanged_l58_58573


namespace find_x_plus_y_squared_l58_58181

variable (x y a b : ℝ)

def condition1 := x * y = b
def condition2 := (1 / (x ^ 2)) + (1 / (y ^ 2)) = a

theorem find_x_plus_y_squared (h1 : condition1 x y b) (h2 : condition2 x y a) : 
  (x + y) ^ 2 = a * b ^ 2 + 2 * b :=
by
  sorry

end find_x_plus_y_squared_l58_58181


namespace general_term_formaula_sum_of_seq_b_l58_58610

noncomputable def seq_a (n : ℕ) := 2 * n + 1

noncomputable def seq_b (n : ℕ) := 1 / ((seq_a n)^2 - 1)

noncomputable def sum_seq_a (n : ℕ) := (Finset.range n).sum seq_a

noncomputable def sum_seq_b (n : ℕ) := (Finset.range n).sum seq_b

theorem general_term_formaula (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  seq_a n = 2 * n + 1 :=
by
  intros
  sorry

theorem sum_of_seq_b (n : ℕ) (h : n > 0) :
  (∀ n, (seq_a n) > 0) ∧ (∀ n, (seq_a n)^2 + 2 * (seq_a n) = 4 * (sum_seq_a n) + 3) →
  sum_seq_b n = n / (4 * (n + 1)) :=
by
  intros
  sorry

end general_term_formaula_sum_of_seq_b_l58_58610


namespace find_k_l58_58924

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

theorem find_k (a b k : ℝ) (h1 : f a b k = 4) (h2 : f a b (f a b k) = 7) (h3 : f a b (f a b (f a b k)) = 19) :
  k = 13 / 4 := 
sorry

end find_k_l58_58924


namespace integer_radius_or_root_two_radius_l58_58635

theorem integer_radius_or_root_two_radius (R : ℝ) (hR : R > 0)
  (h1988 : ∃ p : ℤ × ℤ, p ≠ (0, 0) ∧ int.sqrt (↑(p.1) ^ 2 + ↑(p.2) ^ 2) = R ∧ 
  (finset.univ.filter (λ q : ℤ × ℤ, int.sqrt (↑(q.1) ^ 2 + ↑(q.2) ^ 2) = R)).card = 1988) :
  ∃ k : ℤ, R = k ∨ (ℂ.sqrt 2) * R = k :=
sorry

end integer_radius_or_root_two_radius_l58_58635


namespace greatest_possible_value_l58_58543

theorem greatest_possible_value (x : ℝ) (h : 15 = x^2 + 1 / x^2) : 
  x + 1 / x ≤ sqrt 17 :=
sorry

end greatest_possible_value_l58_58543


namespace matrix_product_correct_l58_58811

def A : Matrix (Fin 3) (Fin 3) Int := 
  ![![2, 0, -1], 
    ![1, 3, -2], 
    ![0, -1, 2]]

def B : Matrix (Fin 3) (Fin 3) Int := 
  ![![1, -1, 0], 
    ![0, 2, -3], 
    ![3, 1, 1]]

def C : Matrix (Fin 3) (Fin 3) Int := 
  ![[-1, -3, -1], 
    [-5, 3, -11], 
    [6, 0, 5]]

theorem matrix_product_correct : A ⬝ B = C := by
  sorry

end matrix_product_correct_l58_58811


namespace smallest_positive_integer_l58_58726

theorem smallest_positive_integer :
  ∃ x : ℤ, 0 < x ∧ (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 11 = 10) ∧ x = 384 :=
by
  sorry

end smallest_positive_integer_l58_58726


namespace eccentricity_of_ellipse_l58_58501

noncomputable def ellipse_eccentricity (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) := 
  let c := b in
  let e := c / (Real.sqrt (b^2 + c^2)) in
  e

theorem eccentricity_of_ellipse (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) (h₂ : b > 0) (h₃ : a = Real.sqrt (b^2 + b^2)) :
  ellipse_eccentricity a b h₀ h₁ = Real.sqrt 2 / 2 :=
by sorry

end eccentricity_of_ellipse_l58_58501


namespace apple_baskets_l58_58988

theorem apple_baskets (total_apples : ℕ) (apples_per_basket : ℕ) (total_apples_eq : total_apples = 495) (apples_per_basket_eq : apples_per_basket = 25) :
  total_apples / apples_per_basket = 19 :=
by
  sorry

end apple_baskets_l58_58988


namespace bird_mammal_difference_africa_asia_l58_58826

noncomputable def bird_families_to_africa := 42
noncomputable def bird_families_to_asia := 31
noncomputable def bird_families_to_south_america := 7

noncomputable def mammal_families_to_africa := 24
noncomputable def mammal_families_to_asia := 18
noncomputable def mammal_families_to_south_america := 15

noncomputable def reptile_families_to_africa := 15
noncomputable def reptile_families_to_asia := 9
noncomputable def reptile_families_to_south_america := 5

-- Calculate the total number of families migrating to Africa, Asia, and South America
noncomputable def total_families_to_africa := bird_families_to_africa + mammal_families_to_africa + reptile_families_to_africa
noncomputable def total_families_to_asia := bird_families_to_asia + mammal_families_to_asia + reptile_families_to_asia
noncomputable def total_families_to_south_america := bird_families_to_south_america + mammal_families_to_south_america + reptile_families_to_south_america

-- Calculate the combined total of bird and mammal families going to Africa
noncomputable def bird_and_mammal_families_to_africa := bird_families_to_africa + mammal_families_to_africa

-- Difference between bird and mammal families to Africa and total animal families to Asia
noncomputable def difference := bird_and_mammal_families_to_africa - total_families_to_asia

theorem bird_mammal_difference_africa_asia : difference = 8 := 
by
  sorry

end bird_mammal_difference_africa_asia_l58_58826


namespace count_positive_integers_eq_one_l58_58076

-- Define the polynomial function
def poly (n : ℕ) : ℤ :=
  n^3 - 9 * n^2 + 23 * n - 15

-- Define a predicate to check if a number is prime 
def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 → m < k → k % m ≠ 0

-- Define the main problem statement
theorem count_positive_integers_eq_one :
  (Finset.filter (λ n, is_prime (poly n)) (Finset.range 100)).card = 1 :=
by
  sorry

end count_positive_integers_eq_one_l58_58076


namespace johnny_bought_18_packs_l58_58220

-- Definitions for conditions
def total_red_pencils := 21
def extra_red_pencils_per_special_pack := 2
def num_special_packs := 3
def red_pencils_in_regular_pack := 1

-- Theorem statement
theorem johnny_bought_18_packs :
  let extra_red_pencils := num_special_packs * extra_red_pencils_per_special_pack in
  let total_regular_pencils := total_red_pencils - extra_red_pencils in
  let num_regular_packs := total_regular_pencils / red_pencils_in_regular_pack in
  num_regular_packs + num_special_packs = 18 :=
by
  -- This is where the proof would normally go
  sorry

end johnny_bought_18_packs_l58_58220


namespace stratified_sampling_medium_supermarkets_l58_58196

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l58_58196


namespace minute_hand_sweep_probability_l58_58999

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l58_58999


namespace count_ordered_triples_lcm_l58_58228

def lcm_of_pair (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_ordered_triples_lcm :
  (∃ (count : ℕ), count = 70 ∧
   ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) →
   lcm_of_pair a b = 1000 → lcm_of_pair b c = 2000 → lcm_of_pair c a = 2000 → count = 70) :=
sorry

end count_ordered_triples_lcm_l58_58228


namespace cubic_yard_to_cubic_meter_l58_58927

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter_l58_58927


namespace value_of_b1_l58_58745

noncomputable def a : ℕ → ℚ
| 0 := 0
| 1 := 1
| (n + 2) := a (n + 1) + a n

noncomputable def b (b1 : ℝ) : ℕ → ℝ
| 1 := b1
| (n + 1) := (a (n - 1) * b1 - a n) / (a (n + 1) - a n * b1)

theorem value_of_b1 (b1 : ℝ) :
  b (b1) 2009 = 1 / b1 + 1 → b1 = ((-1 + Real.sqrt 5) / 2) ∨ b1 = ((-1 - Real.sqrt 5) / 2) :=
sorry

end value_of_b1_l58_58745


namespace min_shift_phi_sin_to_cos_l58_58301

noncomputable def phi_min : ℝ := π / 4

theorem min_shift_phi_sin_to_cos : ∀ φ : ℝ, φ > 0 → (∀ x : ℝ, sin (2 * (x + φ)) = cos (2 * x)) → φ = phi_min :=
by
  assume φ hφ h
  sorry

end min_shift_phi_sin_to_cos_l58_58301


namespace find_n_such_that_sqrt_product_is_9_l58_58553

theorem find_n_such_that_sqrt_product_is_9 (n : ℕ) :
  (sqrt(∏ k in finset.range n, ((2 * k + 1) : ℚ) / ((2 * k - 1) : ℚ)) = 9) → n = 40 :=
by
  sorry

end find_n_such_that_sqrt_product_is_9_l58_58553


namespace aitana_jayda_total_spending_l58_58795

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l58_58795


namespace sum_of_nine_l58_58870

theorem sum_of_nine (S : ℕ → ℕ) (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), S n = n * (a 1 + a n) / 2)
(h₁ : S 3 = 30) (h₂ : S 6 = 100) : S 9 = 240 := 
sorry

end sum_of_nine_l58_58870


namespace probability_in_given_interval_l58_58035

noncomputable def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval (a b c d : ℝ) : ℝ :=
  (length_interval a b) / (length_interval c d)

theorem probability_in_given_interval : 
  probability_in_interval (-1) 1 (-2) 3 = 2 / 5 :=
by
  sorry

end probability_in_given_interval_l58_58035


namespace coeff_comparison_l58_58611

def a_k (k : ℕ) : ℕ := (2 ^ k) * Nat.choose 100 k

theorem coeff_comparison :
  (Finset.filter (fun r => a_k r < a_k (r + 1)) (Finset.range 100)).card = 67 :=
by
  sorry

end coeff_comparison_l58_58611


namespace candy_pack_cost_l58_58069

theorem candy_pack_cost (c : ℝ) (h1 : 20 + 78 = 98) (h2 : 2 * c = 98) : c = 49 :=
by {
  sorry
}

end candy_pack_cost_l58_58069


namespace stirling_number_recursion_l58_58108

open Finset

noncomputable def S : ℕ → ℕ → ℕ
| n, 0       := 0
| 0, k       := if k = 0 then 1 else 0
| (n+1), k+1 := (k+1) * S n (k+1) + S n k

theorem stirling_number_recursion (n k : ℕ) (hn : n ≥ 2) (hk : k ≥ 2) :
  S n k = k * S (n-1) k + S (n-1) (k-1) :=
sorry

end stirling_number_recursion_l58_58108


namespace length_of_lateral_edge_l58_58186

theorem length_of_lateral_edge (vertices : ℕ) (total_length : ℝ) (num_lateral_edges : ℕ) (length_each_edge : ℝ) : 
  vertices = 12 ∧ total_length = 30 ∧ num_lateral_edges = 6 → length_each_edge = 5 :=
by
  intros h
  cases h with h_vertices h_rest
  cases h_rest with h_length h_num_edges
  have h_calculation : length_each_edge = total_length / num_lateral_edges := sorry
  rw [h_vertices, h_length, h_num_edges] at h_calculation
  norm_num at h_calculation
  exact h_calculation
  sorry

end length_of_lateral_edge_l58_58186


namespace ellipse_equation_and_area_l58_58871

noncomputable def ellipse_params : Type := { 
  a b : ℝ // a > 0 ∧ b > 0 ∧ a > b }

theorem ellipse_equation_and_area 
  (a b : ℝ) 
  (h : ellipse_params) : 
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 1 ∧ y = (real.sqrt 2) / 2 ∧ 2 * a = 2 * real.sqrt 2 ∧ b^2 = 1) → 
  (∀ m : ℝ, ∃ S : ℝ, S ≤ real.sqrt 2 / 2) := sorry

end ellipse_equation_and_area_l58_58871


namespace find_a_l58_58905

-- Given function
def quadratic_func (a x : ℝ) := a * (x - 1)^2 - a

-- Conditions
def condition1 (a : ℝ) := a ≠ 0
def condition2 (x : ℝ) := -1 ≤ x ∧ x ≤ 4
def min_value (y : ℝ) := y = -4

theorem find_a (a : ℝ) (ha : condition1 a) :
  ∃ a, (∀ x, condition2 x → quadratic_func a x = -4) → (a = 4 ∨ a = -1 / 2) :=
sorry

end find_a_l58_58905


namespace average_of_last_three_l58_58665

theorem average_of_last_three (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : A + D = 11)
  (h3 : D = 4) : 
  (B + C + D) / 3 = 5 :=
by
  sorry

end average_of_last_three_l58_58665


namespace width_after_water_rises_l58_58730

noncomputable def parabolic_arch_bridge_width : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩,
  let a := -8 in
  (x^2 = a * y) ∧ (x = 4 ∧ y = -2)

theorem width_after_water_rises :
  ∀ (x y : ℝ), y = -2 + 1/2 →
  x^2 = -8 * y →
  2 * abs x = 4 * real.sqrt 3 :=
by
  intro x y h1 h2
  rw h1 at h2
  simp at h2
  rw abs at h2
  exact ⟨⟨_, _⟩⟩ -- sorry

end width_after_water_rises_l58_58730


namespace part_one_part_two_l58_58899

noncomputable theory

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2 * (a + 1) * Real.log x

theorem part_one (a : ℝ) (h1 : (∃x > 0, f.derivative x a = 0) ∧ ∃y > 0, f.derivative y a = 0 ∧ x ≠ y) :
  a > 2 + 2 * Real.sqrt 2 :=
sorry

theorem part_two (a : ℝ) (h1 : -1 < a ∧ a < 3) (x1 x2 : ℝ) (h2 : 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) : 
  (f x1 a - f x2 a) / (x1 - x2) > 2 :=
sorry

end part_one_part_two_l58_58899


namespace stratified_sampling_medium_supermarkets_l58_58195

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end stratified_sampling_medium_supermarkets_l58_58195


namespace ratio_areas_l58_58770

def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_point (k x : ℝ) : ℝ := k * (x - 2) - 1

def tangent_condition (k : ℝ) : Prop := 16 * k^2 - 4 * (8 * k + 4) = 0

def coordinates_of_tangents (k : ℝ) : 
  E = ⟨√2 + 1, 0⟩ ∧ F = ⟨1 - √2, 0⟩

def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area_triangle (base height : ℝ) : ℝ := 1 / 2 * base * height

theorem ratio_areas (P : ℝ × ℝ) (A B E F O : ℝ × ℝ) :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  O = ⟨0, 0⟩ ∧ P = ⟨2, -1⟩ ∧ 
  ∃ k, tangent_condition k ∧ 
  coordinates_of_tangents k E F ∧ 
  let SPEF := area_triangle (distance E.1 E.2 F.1 F.2) 1,
  let SOAB := area_triangle 8 (sqrt 2 / 2) in
  SPEF / SOAB = 1 / 2 :=
sorry

end ratio_areas_l58_58770


namespace binom_20_10_l58_58130

theorem binom_20_10 :
  (∑ j in finset.range (10 + 1), ∑ i in finset.range (20 - j + 1), 
    if i = 17 ∧ (j = 7 ∨ j = 8 ∨ j = 9) then 1 else 0 * 
    if j = 7 ∧ i = 17 then 19448 else 
    if j = 8 ∧ i = 17 then 24310 else 
    if j = 9 ∧ i = 17 then 24310 else 0) = 111826 := 
sorry

end binom_20_10_l58_58130


namespace question_1_question_2_question_3_l58_58362

def deck_size : Nat := 32

theorem question_1 :
  let hands_when_order_matters := deck_size * (deck_size - 1)
  hands_when_order_matters = 992 :=
by
  let hands_when_order_matters := deck_size * (deck_size - 1)
  sorry

theorem question_2 :
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  hands_when_order_does_not_matter = 496 :=
by
  let hands_when_order_does_not_matter := (deck_size * (deck_size - 1)) / 2
  sorry

theorem question_3 :
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  hands_3_cards_order_does_not_matter = 4960 :=
by
  let hands_3_cards_order_does_not_matter := (deck_size * (deck_size - 1) * (deck_size - 2)) / 6
  sorry

end question_1_question_2_question_3_l58_58362


namespace total_fencing_cost_l58_58839

noncomputable def pi : ℝ := Real.pi  -- To use the built-in constant for pi

def diameter : ℝ := 16
def rate_per_meter : ℝ := 3

theorem total_fencing_cost :
  let circumference := pi * diameter in
  let total_cost := circumference * rate_per_meter in
  total_cost ≈ 151 :=
by
  sorry

end total_fencing_cost_l58_58839


namespace average_age_combined_l58_58664

theorem average_age_combined (fifth_graders_count : ℕ) (fifth_graders_avg_age : ℚ)
                             (parents_count : ℕ) (parents_avg_age : ℚ)
                             (grandparents_count : ℕ) (grandparents_avg_age : ℚ) :
  fifth_graders_count = 40 →
  fifth_graders_avg_age = 10 →
  parents_count = 60 →
  parents_avg_age = 35 →
  grandparents_count = 20 →
  grandparents_avg_age = 65 →
  (fifth_graders_count * fifth_graders_avg_age + 
   parents_count * parents_avg_age + 
   grandparents_count * grandparents_avg_age) / 
  (fifth_graders_count + parents_count + grandparents_count) = 95 / 3 := sorry

end average_age_combined_l58_58664


namespace cobbler_mends_3_pairs_per_hour_l58_58390

def cobbler_hours_per_day_mon_thu := 8
def cobbler_hours_friday := 11 - 8
def cobbler_total_hours_week := 4 * cobbler_hours_per_day_mon_thu + cobbler_hours_friday
def cobbler_pairs_per_week := 105
def cobbler_pairs_per_hour := cobbler_pairs_per_week / cobbler_total_hours_week

theorem cobbler_mends_3_pairs_per_hour : cobbler_pairs_per_hour = 3 := 
by 
  -- Add the steps if necessary but in this scenario, we are skipping proof details
  sorry

end cobbler_mends_3_pairs_per_hour_l58_58390


namespace num_nat_factors_of_N_l58_58536

theorem num_nat_factors_of_N :
  let N := (2^4) * (3^3) * (5^2) * (7^1) in
  num_factors N = 120 :=
by
  sorry

end num_nat_factors_of_N_l58_58536


namespace solve_inequality_l58_58157

theorem solve_inequality (x : ℝ) : -4 * x - 8 > 0 → x < -2 := sorry

end solve_inequality_l58_58157


namespace domino_covering_impossible_odd_squares_l58_58773

theorem domino_covering_impossible_odd_squares
  (board1 : ℕ) -- 24 squares
  (board2 : ℕ) -- 21 squares
  (board3 : ℕ) -- 23 squares
  (board4 : ℕ) -- 35 squares
  (board5 : ℕ) -- 63 squares
  (h1 : board1 = 24)
  (h2 : board2 = 21)
  (h3 : board3 = 23)
  (h4 : board4 = 35)
  (h5 : board5 = 63) :
  (board2 % 2 = 1) ∧ (board3 % 2 = 1) ∧ (board4 % 2 = 1) ∧ (board5 % 2 = 1) :=
by {
  sorry
}

end domino_covering_impossible_odd_squares_l58_58773


namespace polynomial_at_3_l58_58241

theorem polynomial_at_3 (P : ℝ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 0 ≤ b i ∧ b i < 5)
  (h2 : P (Real.sqrt 5) = 40 + 30 * Real.sqrt 5)
  (h3 : P = λ x, ∑ i in Finset.range (n + 1), b i * x^i) :
  P 3 = 267 :=
sorry

end polynomial_at_3_l58_58241


namespace number_of_groups_of_three_marbles_l58_58332

-- Define the problem conditions 
def red_marble : ℕ := 1
def blue_marble : ℕ := 1
def black_marble : ℕ := 1
def white_marbles : ℕ := 4

-- The proof problem statement
theorem number_of_groups_of_three_marbles (red_marble blue_marble black_marble white_marbles : ℕ) :
  (white_marbles.choose 3) + (3.choose 2 * white_marbles.choose 1) = 16 :=
by
  sorry

end number_of_groups_of_three_marbles_l58_58332


namespace count_distinct_special_fraction_sums_l58_58063

def is_special_fraction (a b : ℕ) : Prop :=
  a + b = 18

def special_fractions_sum (a1 b1 a2 b2 : ℕ) (h1 : is_special_fraction a1 b1) (h2 : is_special_fraction a2 b2) : ℚ :=
  (a1 : ℚ) / (b1 : ℚ) + (a2 : ℚ) / (b2 : ℚ)
  
theorem count_distinct_special_fraction_sums : 
  ∃ n, n = 10 ∧ 
       ∀ a1 b1 a2 b2, 
         is_special_fraction a1 b1 → 
         is_special_fraction a2 b2 → 
         ∃ m : ℕ, 
           m ∈ { 
             nat_floor (special_fractions_sum a1 b1 a2 b2 _) :
             a1 b1 a2 b2 | 
             is_special_fraction a1 b1 ∧ 
             is_special_fraction a2 b2 
           } :=
sorry

end count_distinct_special_fraction_sums_l58_58063


namespace rounds_to_determine_l58_58797

theorem rounds_to_determine (n : ℕ) (x : Fin n → ℕ) (a b : Fin n → ℕ) (S₁ S₂ : ℕ) :
  S₁ = ∑ i, a i * x i →
  S₂ = ∑ i, b i * x i →
  (∀ y : Fin n → ℕ, (y ≠ x) → ∃ c d : Fin n → ℕ, (c ≠ a) ∧ (d ≠ b) ∧ ∑ i, c i * y i = S₁ ∧ ∑ i, d i * y i = S₂) →
  2 = 2 :=
by sorry

end rounds_to_determine_l58_58797


namespace spicy_hot_noodles_plates_l58_58025

theorem spicy_hot_noodles_plates (total_plates lobster_rolls seafood_noodles spicy_hot_noodles : ℕ) :
  total_plates = 55 →
  lobster_rolls = 25 →
  seafood_noodles = 16 →
  spicy_hot_noodles = total_plates - (lobster_rolls + seafood_noodles) →
  spicy_hot_noodles = 14 := by
  intros h_total h_lobster h_seafood h_eq
  rw [h_total, h_lobster, h_seafood] at h_eq
  exact h_eq

end spicy_hot_noodles_plates_l58_58025


namespace unique_triangle_AB_AC_angleB_l58_58215

theorem unique_triangle_AB_AC_angleB (AB : ℝ) (AC : ℝ) (BC : ℝ) (B : ℝ)
  (h1: AB = 2 * Real.sqrt 2)
  (h2: B = Real.pi / 4)
  (h3: AC = 3) :
  ∃! (triangle : Triangle ℝ), triangle.AB = AB ∧ triangle.AC = AC ∧ triangle.B = B := by
  sorry

end unique_triangle_AB_AC_angleB_l58_58215


namespace principal_amount_l58_58296

variable (P : ℝ)
variable (r : ℝ) (n : ℝ) (difference : ℝ)

def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  P * ((1 + r / 100) ^ n - 1)

def simpleInterest (P : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  P * r * n / 100

theorem principal_amount:
  ∀ (P r n difference : ℝ), 
  r = 20 → n = 2 → difference = 72 →
  (compoundInterest P r n - simpleInterest P r n = difference) →
  P = 1800 :=
by
  intros P r n difference hr hn hdiff hcondition
  sorry

end principal_amount_l58_58296


namespace magnitude_of_vector_difference_l58_58912

noncomputable theory

open Real

/-- The problem setup -/
def a := (1 : ℝ, 0 : ℝ)
def b (angle : ℝ) (mag : ℝ) : ℝ × ℝ := (mag * cos angle, mag * sin angle)

def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

/-- Given conditions -/
def angle := π / 6  -- 30 degrees in radians
def mag_b := sqrt 3
def vector_b := b angle mag_b

/-- The vector difference -/
def vector_diff := (a.1 - vector_b.1, a.2 - vector_b.2)

/-- Magnitude of the vector difference -/
def result := magnitude vector_diff

/-- Proof statement: given the conditions, the magnitude of vector difference is 1 -/
theorem magnitude_of_vector_difference : result = 1 :=
sorry

end magnitude_of_vector_difference_l58_58912


namespace point_on_line_l58_58310

theorem point_on_line :
  ∃ a b : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (x = 4 ∧ y = 5) ∨ (x = 8 ∧ y = 17) ∨ (x = 12 ∧ y = 29) → y = a * x + b) →
  (∃ t : ℝ, (15, t) ∈ {(x, y) | y = a * x + b} ∧ t = 38) :=
by
  sorry

end point_on_line_l58_58310


namespace radius_of_circle_from_spherical_coordinates_l58_58446

-- Defining spherical coordinates.
structure SphericalCoordinates where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Defining the specific instance given in the problem.
def specificPoint : SphericalCoordinates := { ρ := 2, θ := arbitrary _, φ := Real.pi / 4 }

-- Conversion from spherical to Cartesian coordinates.
def toCartesian (p : SphericalCoordinates) : EuclideanSpace ℝ 
  | (ρ, θ, φ) => {
    x := ρ * Real.sin φ * Real.cos θ,
    y := ρ * Real.sin φ * Real.sin θ,
    z := ρ * Real.cos φ
  }

-- Define the specific Cartesian coordinates given the specific spherical point
def specificPointCartesian : EuclideanSpace ℝ := toCartesian specificPoint

-- Proving the radius of the circle formed by the points.
theorem radius_of_circle_from_spherical_coordinates : 
  (sqrt ((toCartesian specificPoint).x^2 + (toCartesian specificPoint).y^2) = sqrt 2) := 
  sorry

end radius_of_circle_from_spherical_coordinates_l58_58446


namespace hiker_catches_up_after_fifteen_minutes_l58_58398

-- Define constants and parameters
def hiker_speed := 4 / 1 -- The hiker's speed in km/h
def cyclist_speed := 12 / 1 -- The cyclist's speed in km/h
def stop_time := 5 / 60 -- The time after which the cyclist stops in hours

-- Define the time the hiker gets in minutes
def time_needed_for_hiker_to_catch_up (hiker_speed cyclist_speed : ℝ) (stop_time : ℝ) : ℝ :=
  (cyclist_speed * stop_time / hiker_speed) * 60

theorem hiker_catches_up_after_fifteen_minutes :
  time_needed_for_hiker_to_catch_up hiker_speed cyclist_speed stop_time = 15 := by
  sorry

end hiker_catches_up_after_fifteen_minutes_l58_58398


namespace rearrangements_of_COMMITTEE_vowels_first_l58_58915

theorem rearrangements_of_COMMITTEE_vowels_first : 
  let vowels := ['O', 'I', 'E', 'E'],
      consonants := ['C', 'M', 'M', 'T', 'T'],
      total_vowel_arrangements := nat.factorial 4 / nat.factorial 2,
      total_consonant_arrangements := nat.factorial 5 / (nat.factorial 2 * nat.factorial 2),
      total_arrangements := total_vowel_arrangements * total_consonant_arrangements
  in 
    total_arrangements = 360 :=
by
  let vowels := ['O', 'I', 'E', 'E']
  let consonants := ['C', 'M', 'M', 'T', 'T']
  let total_vowel_arrangements := nat.factorial 4 / nat.factorial 2
  let total_consonant_arrangements := nat.factorial 5 / (nat.factorial 2 * nat.factorial 2)
  let total_arrangements := total_vowel_arrangements * total_consonant_arrangements
  exact eq.trans (by rfl) sorry

end rearrangements_of_COMMITTEE_vowels_first_l58_58915


namespace numbers_between_roots_excluding_divisibles_l58_58539

theorem numbers_between_roots_excluding_divisibles : 
  let sqrt_50 := Real.sqrt 50
  let sqrt_200 := Real.sqrt 200
  set numbers := {n : ℕ | sqrt_50 < n ∧ n < sqrt_200 ∧ ¬ (5 ∣ n)}
  numbers.card = 6 :=
by
  let sqrt_50 := Real.sqrt 50
  let sqrt_200 := Real.sqrt 200
  set numbers := {n : ℕ | sqrt_50 < n ∧ n < sqrt_200 ∧ ¬ (5 ∣ n)}
  have l : numbers = {8, 9, 11, 12, 13, 14} := sorry -- Placeholder for actual proof 
  have h : numbers.card = 6 := sorry -- Placeholder for actual proof
  exact h

end numbers_between_roots_excluding_divisibles_l58_58539


namespace functional_equation_solution_l58_58083

noncomputable def f (t : ℝ) (x : ℝ) := (t * (x - t)) / (t + 1)

noncomputable def g (t : ℝ) (x : ℝ) := t * (x - t)

theorem functional_equation_solution (t : ℝ) (ht : t ≠ -1) :
  ∀ x y : ℝ, f t (x + g t y) = x * f t y - y * f t x + g t x :=
by
  intros x y
  let fx := f t
  let gx := g t
  sorry

end functional_equation_solution_l58_58083


namespace scientific_notation_l58_58315

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l58_58315


namespace quadratic_form_sum_const_l58_58685

theorem quadratic_form_sum_const (a b c x : ℝ) (h : 4 * x^2 - 28 * x - 48 = a * (x + b)^2 + c) : 
  a + b + c = -96.5 :=
by
  sorry

end quadratic_form_sum_const_l58_58685


namespace area_of_triangle_DEF_l58_58037

theorem area_of_triangle_DEF
(point Q inside_triangle DEF :
  ∃ (Q : Point) (inside : inside_triangle Q DEF),
  ∃ (sub_triangle_areas : ℕ × ℕ × ℕ),
  sub_triangle_areas = (16, 25, 36)) :
  ∃ (area_DEF : ℕ),
  area_DEF = 225 := by
  sorry

end area_of_triangle_DEF_l58_58037


namespace simplify_trig_expression_l58_58652

theorem simplify_trig_expression :
  (sin (real.pi / 12) + sin (real.pi / 7.2) + sin (real.pi / 5.142857142857142856) +
  sin (real.pi / 4) + sin (11 * real.pi / 36) + sin (13 * real.pi / 36) +
  sin (17 * real.pi / 36) + sin (19 * real.pi / 36)) /
  (cos (real.pi / 18) * cos (real.pi / 12) * cos (real.pi / 7.2)) =
  4 * sin (5 * real.pi / 18) := 
sorry

end simplify_trig_expression_l58_58652


namespace arithmetic_sequence_properties_l58_58233

variable {a : ℕ → ℤ} (S : ℕ → ℤ) (d : ℤ)

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * d

axiom a1_val : a 1 = 30
axiom S12_eq_S19 : sum_of_first_n_terms a 12 = sum_of_first_n_terms a 19

-- Prove that d = -2 and S_n ≤ S_15 for any n
theorem arithmetic_sequence_properties :
  is_arithmetic_sequence a d →
  (∀ n, S n = sum_of_first_n_terms a n) →
  d = -2 ∧ ∀ n, S n ≤ S 15 :=
by
  intros h_arith h_sum
  sorry

end arithmetic_sequence_properties_l58_58233


namespace ant_impossibility_l58_58432

-- Define the vertices and edges of a cube
structure Cube :=
(vertices : Finset ℕ) -- Representing a finite set of vertices
(edges : Finset (ℕ × ℕ)) -- Representing a finite set of edges between vertices
(valid_edge : ∀ e ∈ edges, ∃ v1 v2, (v1, v2) = e ∨ (v2, v1) = e)
(starting_vertex : ℕ)

-- Ant behavior on the cube
structure AntOnCube (C : Cube) :=
(is_path_valid : List ℕ → Prop) -- A property that checks the path is valid

-- Problem conditions translated: 
-- No retracing and specific visit numbers
noncomputable def ant_problem (C : Cube) (A : AntOnCube C) : Prop :=
  ∀ (path : List ℕ), A.is_path_valid path → ¬ (
    (path.count C.starting_vertex = 25) ∧ 
    (∀ v ∈ C.vertices, v ≠ C.starting_vertex → path.count v = 20)
  )

-- The final theorem statement
theorem ant_impossibility (C : Cube) (A : AntOnCube C) : ant_problem C A :=
by
  -- providing the theorem framework; proof omitted with sorry
  sorry

end ant_impossibility_l58_58432


namespace perfect_squares_as_difference_l58_58454

theorem perfect_squares_as_difference (n : ℕ) :
  ∃ (m : ℕ), m < 12100 ∧ ∃ (k : ℕ), m = 2 * k + 1 ∧ ∀ j, j^2 = m → 55 = ∑ i : ℕ in finset.range 110, if odd (i^2) then 1 else 0 :=
sorry

end perfect_squares_as_difference_l58_58454


namespace simplify_sqrt_144000_l58_58270

theorem simplify_sqrt_144000 :
  (sqrt 144000 = 120 * sqrt 10) :=
by
  -- Assume given conditions
  have h1 : 144000 = 144 * 1000 := by
    calc 144000 = 144 * 1000 : by rfl

  have h2 : 144 = 12^2 := by rfl

  have h3 : sqrt (a * b) = sqrt a * sqrt b := by sorry

  have h4 : sqrt (10 ^ 3) = 10 * sqrt 10 := by sorry

  -- Prove the target
  calc
    sqrt 144000
    = sqrt (144 * 1000) : by rw [←h1]
    = sqrt (12^2 * 10^3) : by rw [h2, pow_succ]
    = sqrt (12^2) * sqrt (10^3) : by rw [h3]
    = 12 * sqrt (10^3) : by rw [sqrt_sq', h2, pow_two]
    = 12 * (10 * sqrt 10) : by rw [h4]
    = 12 * 10 * sqrt 10 : by rw [mul_assoc]
    = 120 * sqrt 10 : by sorry

-- sqrt_sq' and pow_two are used to simplify sqrt (12^2) == 12.

end simplify_sqrt_144000_l58_58270


namespace min_bottles_required_l58_58827

theorem min_bottles_required (bottle_ounces : ℕ) (total_ounces : ℕ) (h : bottle_ounces = 15) (ht : total_ounces = 150) :
  ∃ (n : ℕ), n * bottle_ounces >= total_ounces ∧ n = 10 :=
by
  sorry

end min_bottles_required_l58_58827


namespace max_regions_divided_by_lines_l58_58910

theorem max_regions_divided_by_lines (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) :
  ∃ r : ℕ, r = m * n + 2 * m + 2 * n - 1 :=
by
  sorry

end max_regions_divided_by_lines_l58_58910


namespace jenny_profit_l58_58595

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end jenny_profit_l58_58595


namespace finite_quadruples_n_factorial_l58_58645

theorem finite_quadruples_n_factorial (n a b c : ℕ) (h_pos : 0 < n) (h_cond : n! = a^(n-1) + b^(n-1) + c^(n-1)) : n ≤ 100 :=
by
  sorry

end finite_quadruples_n_factorial_l58_58645


namespace compare_abc_l58_58920

theorem compare_abc : 
  let a := (1/2)^3
  let b := log 3 2
  let c := sqrt 3 / 3
  a < c ∧ c < b :=
by
  let a := (1/2)^3
  let b := Real.log 2 / Real.log 3
  let c := Real.sqrt 3 / 3
  have ha : a = (1/2)^3 := rfl
  have hb : b = Real.log 2 / Real.log 3 := rfl
  have hc : c = Real.sqrt 3 / 3 := rfl
  have h1 : a < 1 / 2 := by linarith
  have h2 : 1 / 2 < b := sorry -- proof omitted for brevity
  have h3 : 1 / 2 < c := sorry -- proof omitted for brevity
  have h4 : b > c := sorry -- proof omitted for brevity
  exact ⟨by linarith, h4⟩

end compare_abc_l58_58920


namespace amount_diana_owes_l58_58078

-- Problem definitions
def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest := principal * rate * time
def total_owed := principal + interest

-- Theorem to prove that the total amount owed is $80.25
theorem amount_diana_owes : total_owed = 80.25 := by
  sorry

end amount_diana_owes_l58_58078


namespace area_of_quadrilateral_QTUS_l58_58380

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 3
def PR : ℝ := 3
def QR : ℝ := 2

-- Define the point S such that QR is extended by half its length
def RS : ℝ := QR / 2

-- Define the midpoint of PR
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def T : ℝ × ℝ := midpoint (⟨0, 0⟩) (⟨PR, 0⟩) -- Assuming a coordinate system

-- Define the intersection of TS with PQ and the properties used to find U
-- Placeholder, as intersection calculation requires more complex geometry
def U : ℝ × ℝ := sorry 

-- Calculate area of quadrilateral QTUS
noncomputable def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ := 
  let area_triangle := λ P Q R : ℝ × ℝ, ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2).abs
  in (area_triangle A B C + area_triangle A C D)

theorem area_of_quadrilateral_QTUS :
  let Q := ⟨QR, 0⟩
  let P := ⟨0, 0⟩
  let S := ⟨QR + RS, 0⟩
  area_of_quadrilateral Q T U S = /* expected area */ sorry := 
sorry

end area_of_quadrilateral_QTUS_l58_58380


namespace line_equations_through_point_with_intercepts_l58_58299

theorem line_equations_through_point_with_intercepts (x y : ℝ) :
  (x = -10 ∧ y = 10) ∧ (∃ a : ℝ, 4 * a = intercept_x ∧ a = intercept_y) →
  (x + y = 0 ∨ x + 4 * y - 30 = 0) :=
by
  sorry

end line_equations_through_point_with_intercepts_l58_58299


namespace triangle_area_proof_l58_58566

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (B : ℝ) (hB : B = 2 * Real.pi / 3) (hb : b = Real.sqrt 13) (h_sum : a + c = 4) :
  triangle_area a b c B = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_proof_l58_58566


namespace max_students_seating_l58_58801

theorem max_students_seating :
  ∀ (rows : ℕ) (initial_seats : ℕ),
  rows = 20 →
  initial_seats = 10 →
  (∀ i, 1 ≤ i ∧ i ≤ rows → 
       let n_i := initial_seats + (i - 1) in 
       let max_students_row := (n_i + 1) / 2 in True) →
  ∑ i in finset.range rows, (initial_seats + i + 1) / 2 = 200 :=
by
  intros rows initial_seats h_rows h_initial_seats h_max_students_row
  sorry

end max_students_seating_l58_58801


namespace angle_A_is_30_area_with_side_relation_area_with_angle_condition_l58_58590

open Real

-- Given vectors m and n, and their properties
def m : ℝ × ℝ := (-1, 1)
def n (B C : ℝ) : ℝ × ℝ := (cos B * cos C, sin B * sin C - sqrt 3 / 2)

-- Condition: m is perpendicular to n
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Given conditions
def condition_angle_A (B C : ℝ) : Prop := 
    perpendicular m (n B C) → 
    cos (B + C) = - sqrt 3 / 2

-- Given values
def a := 1
def B := π / 4  -- 45°
def A := π / 6  -- 30°

-- Triangle sides relations
def side_relation (b c : ℝ) : Prop :=
    2 * c - (sqrt 3 + 1) * b = 0

-- Calculate area
noncomputable def area_triangle (b c : ℝ) : ℝ :=
    (1 / 2) * b * c * (sin A)

-- Prove angle A is 30°
theorem angle_A_is_30 (B C : ℝ) (h : condition_angle_A B C) : 
    angle_A_is_30 : B + C = π - A :=
sorry

-- Prove area of triangle with condition ②
theorem area_with_side_relation (B C b c : ℝ) 
    (h : side_relation b c ∧ condition_angle_A B C ∧ A = π / 6) : 
    area_triangle b c = (sqrt 3 + 1) / 4 :=
sorry

-- Prove area of triangle with condition ③
theorem area_with_angle_condition (C b c : ℝ) 
    (h : condition_angle_A B C ∧ B = π / 4 ∧ A = π / 6) : 
    area_triangle b c = (sqrt 3 + 1) / 4 :=
sorry

end angle_A_is_30_area_with_side_relation_area_with_angle_condition_l58_58590


namespace exists_pair_in_six_cascades_exists_coloring_function_l58_58820

-- Define the notion of a cascade
def is_cascade (r : ℕ) (s : set ℕ) : Prop :=
  s = {n | ∃ k : ℕ, k ∈ set.Icc 1 12 ∧ n = k * r}

-- Part (a): Prove that there exist numbers a and b that belong to six different cascades.
theorem exists_pair_in_six_cascades :
  ∃ (a b : ℕ), a ≠ b ∧ ∃ (r1 r2 r3 r4 r5 r6 : ℕ),
  (is_cascade r1 {a, b} ∧ is_cascade r2 {a, b} ∧ is_cascade r3 {a, b} ∧
   is_cascade r4 {a, b} ∧ is_cascade r5 {a, b} ∧ is_cascade r6 {a, b}) := sorry

-- Part (b): Prove that there exists a coloring function such that all cascades have different colors.
theorem exists_coloring_function :
  ∃ (f : ℕ → ℕ), (∀ r : ℕ, set.pairwise (λ x y, f x ≠ f y) {n | ∃ k : ℕ, k ∈ set.Icc 1 12 ∧ n = k * r}) := sorry

end exists_pair_in_six_cascades_exists_coloring_function_l58_58820


namespace sum_third_largest_and_smallest_l58_58541

open List

def digits := [7, 6, 5, 8]

def allFourDigitNumbers (ds : List ℕ) : List ℕ :=
  ds.permutations.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)

def sortedNumbers := (allFourDigitNumbers digits).sort (· > ·)
def thirdLargest := sortedNumbers.nthLe 2 sorry
def thirdSmallest := sortedNumbers.nthLe (sortedNumbers.length - 3) sorry

theorem sum_third_largest_and_smallest :
  thirdLargest + thirdSmallest = 14443 :=
sorry

end sum_third_largest_and_smallest_l58_58541


namespace union_of_A_and_B_l58_58103

variables (A B : Set ℤ)
variable (a : ℤ)
theorem union_of_A_and_B : (A = {4, a^2}) → (B = {a-6, 1+a, 9}) → (A ∩ B = {9}) → (A ∪ B = {-9, -2, 4, 9}) :=
by
  intros hA hB hInt
  sorry

end union_of_A_and_B_l58_58103


namespace original_volume_of_ice_cube_l58_58054

theorem original_volume_of_ice_cube
  (V : ℝ)
  (h1 : V * (1/2) * (2/3) * (3/4) * (4/5) = 30)
  : V = 150 :=
sorry

end original_volume_of_ice_cube_l58_58054


namespace arithmetic_sequence_general_formula_l58_58934

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_general_formula :
  ∀ a : ℕ → ℝ, a 1 = -1 → a 2 = 1 → is_arithmetic_sequence a → ∀ n, a n = 2 * n - 3 :=
by
  intros a h1 h2 h_seq
  have d_eq : a 2 - a 1 = 2 := by linarith [h1, h2]
  specialize h_seq 1
  rw [←d_eq] at h_seq
  sorry

end arithmetic_sequence_general_formula_l58_58934


namespace AL_perpendicular_DK_l58_58968

variables {s : ℝ} {A B C D K L M : ℝ × ℝ}
variables (ABCD_square : is_square A B C D s)
variables (K_in_AB : on_segment A B K)
variables (L_in_BC : on_segment B C L)
variables (M_in_CD : on_segment C D M)
variables (KLM_right_isosceles : is_right_isosceles_triangle K L M)

theorem AL_perpendicular_DK : is_perpendicular (line_through A L) (line_through D K) :=
sorry

-- Helper Definitions (Assuming these helpers are defined somewhere in the library to make the code build)
def is_square (A B C D : ℝ × ℝ) (s : ℝ) : Prop :=
  sorry

def on_segment (P Q R : ℝ × ℝ) : Prop :=
  sorry

def is_right_isosceles_triangle (K L M : ℝ × ℝ) : Prop :=
  sorry

def is_perpendicular (l1 l2 : line) : Prop :=
  sorry

def line_through (P Q : ℝ × ℝ) : line :=
  sorry

end AL_perpendicular_DK_l58_58968


namespace train_length_proof_l58_58423

noncomputable def train_length :=
  let L : ℝ := some (λ L, ∃ v, v = L / 18 ∧ v = (L + 500) / 48) in
  L

theorem train_length_proof : ∃ (L : ℝ), 
  (∀ (v : ℝ), v = L / 18 ∧ v = (L + 500) / 48) →
  L = 300 :=
by
  let v := 300 / 18
  have eq1 : v = 300 / 18 := rfl
  simp at eq1
  have eq2 : v = (300 + 500) / 48 := by
    field_simp
    ring
    
  exact ⟨300, λ v h, by
    simp at h
    cases h with h1 h2
    rw [←h2, h1]
    norm_num⟩

#eval train_length -- Should evaluate to 300

end train_length_proof_l58_58423


namespace problem_statement_l58_58190

theorem problem_statement (p q : Prop) :
  ¬(p ∧ q) ∧ ¬¬p → ¬q := 
by 
  sorry

end problem_statement_l58_58190


namespace notebook_cost_l58_58735

theorem notebook_cost :
  ∃ N : ℕ, 
    let initial_money := 40 in
    let poster_cost := 2 * 5 in
    let bookmark_cost := 2 * 2 in
    let remaining_money := 14 in
    ∃ (N_cost : ℕ), 
        3 * N = N_cost ∧
        initial_money - (poster_cost + bookmark_cost + N_cost) = remaining_money ∧
        N = 4 := 
  sorry

end notebook_cost_l58_58735


namespace dora_knows_coin_position_l58_58914

-- Definitions
def R_is_dime_or_nickel (R : ℕ) (L : ℕ) : Prop := 
  (R = 10 ∧ L = 5) ∨ (R = 5 ∧ L = 10)

-- Theorem statement
theorem dora_knows_coin_position (R : ℕ) (L : ℕ) 
  (h : R_is_dime_or_nickel R L) :
  (3 * R + 2 * L) % 2 = 0 ↔ (R = 10 ∧ L = 5) :=
by
  sorry

end dora_knows_coin_position_l58_58914


namespace bus_routes_and_stops_l58_58670

-- Define the problem statement and conditions
variables (Route Stop : Type) [DecidableEq Route] [DecidableEq Stop]

-- Each route has at least three stops
variable (routes_have_at_least_three_stops : ∀ r : Route, ∃ s1 s2 s3 : Stop, s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3)

-- Travel from any stop to any other without transferring
variable (travel_without_transfer : ∀ s1 s2 : Stop, ∃ r : Route, ∀ r' : Route, (s1 ∈ stops_of_route r) → (s2 ∈ stops_of_route r') → (r = r' ∨ ∃ s : Stop, (s ∈ stops_of_route r) ∧ (s ∈ stops_of_route r')))

-- Exactly one stop is shared between any pair of routes
variable (exactly_one_shared_stop : ∀ r1 r2 : Route, r1 ≠ r2 → ∃! s : Stop, s ∈ stops_of_route r1 ∧ s ∈ stops_of_route r2 )

-- Declaring the result as a theorem to prove
theorem bus_routes_and_stops (total_routes : ℕ) (h : total_routes = 57) :
  ∃ n : ℕ, (∀ r : Route, ∃ stops : Finset Stop, (stops.card = n) ∧ (stops_of_route r = stops)) ∧
           (∀ s : Stop, ∃ routes : Finset Route, (routes.card = n) ∧ (s ∈ stops_of_routes routes)) ∧
           n = 8 :=
sorry

end bus_routes_and_stops_l58_58670


namespace five_digit_divisible_by_four_digit_l58_58612

theorem five_digit_divisible_by_four_digit (x y z u v : ℕ) (h1 : 1 ≤ x) (h2 : x < 10) (h3 : y < 10) (h4 : z < 10) (h5 : u < 10) (h6 : v < 10)
  (h7 : (x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v) % (x * 10^3 + y * 10^2 + u * 10 + v) = 0) : 
  ∃ N, 10 ≤ N ∧ N ≤ 99 ∧ 
  x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v = N * 10^3 ∧
  10 * (x * 10^3 + y * 10^2 + u * 10 + v) = x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v :=
sorry

end five_digit_divisible_by_four_digit_l58_58612


namespace equilateral_triangle_l58_58941

theorem equilateral_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) 
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) 
  (h4 : b = c) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end equilateral_triangle_l58_58941


namespace num_real_roots_f_eq_0_l58_58072

-- Define the odd function f
def f (x : ℝ) : ℝ := if x > 0 then 2010^x + Real.logb 2010 x else if x < 0 then - (2010^(-x) + Real.logb 2010 (-x)) else 0

-- State the problem
theorem num_real_roots_f_eq_0 : ∃ n : ℕ, (∀ x : ℝ, f x = 0 → x = 0) ∧ n = 3 := 
sorry

end num_real_roots_f_eq_0_l58_58072


namespace find_x8_l58_58550

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end find_x8_l58_58550


namespace value_cos_expr_correct_l58_58559

noncomputable def value_cos_expr (A B C : ℝ) [fact (A + B + C = π)] 
  (h_sinA_cosB_tanC : sin A = cos B ∧ cos B = tan C) : ℝ :=
cos A ^ 3 + cos A ^ 2 - cos A

theorem value_cos_expr_correct (A B C : ℝ) [fact (A + B + C = π)]
  (h_sinA_cosB_tanC : sin A = cos B ∧ cos B = tan C) :
  value_cos_expr A B C h_sinA_cosB_tanC = 1 / 2 :=
sorry

end value_cos_expr_correct_l58_58559


namespace AM_perp_DE_l58_58569

variables {A B C E P D R M : Type}
variables [metric_space A]

-- Assume A, B, C are points in a triangle, with B, A, E in a line (square constructed such that AE ⊥ BE)
-- and similarly for A, C, D (another square constructed such that AD ⊥ CD)
-- Assume M is the midpoint of the segment BC

-- Definitions of the main points involved
def is_square (ABEP A B E P : Type) := sorry
def is_square_ACRD (ACRD A C D R : Type) := sorry
def midpoint (M B C : Type) := sorry

-- Condition: Assume squares constructed on the sides
axiom ASquare : is_square ABEP A B E P
axiom CSquare : is_square_ACRD A C D R

-- Condition: Assume M is the midpoint of BC
axiom mid_point : midpoint M B C 

-- The proof problem
theorem AM_perp_DE (ABEP : Type) (ACRD : Type) (M : Type) (DE : Type) 
  [is_square ABEP A B E P] [is_square_ACRD A C D R] [midpoint M B C] : 
  AM ⊥ DE := 
sorry

end AM_perp_DE_l58_58569


namespace number_of_girls_at_Gills_school_l58_58480

-- Let g be the number of girls and b be the number of boys
variables (g b : ℕ)

-- Given conditions
def total_students := g + b = 600
def more_girls := g = b + 30

-- The proof statement
theorem number_of_girls_at_Gills_school (h1 : total_students) (h2 : more_girls) : g = 315 := 
by
  sorry

end number_of_girls_at_Gills_school_l58_58480


namespace range_of_y_l58_58179

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : Int.ceil y * Int.floor y = 72) : 
  -9 < y ∧ y < -8 :=
sorry

end range_of_y_l58_58179


namespace proof_problem_l58_58732

open Real

theorem proof_problem :
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 4) →
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 16/4) →
  (∀ x : ℤ, abs x = 4 → abs (-4) = 4) →
  (∀ x : ℤ, x^2 = 16 → (-4)^2 = 16) →
  (- sqrt 16 = -4) := 
by 
  simp
  sorry

end proof_problem_l58_58732


namespace min_value_expr_l58_58980

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 := 
sorry

end min_value_expr_l58_58980


namespace cube_coloring_schemes_l58_58064

theorem cube_coloring_schemes (colors : Finset ℕ) (h : colors.card = 6) :
  ∃ schemes : Nat, schemes = 230 :=
by
  sorry

end cube_coloring_schemes_l58_58064


namespace fraction_addition_l58_58308

def simplest_fraction (r : ℚ) : ℚ :=
let n := r.num.gcd(r.den)
in ⟨ r.num / n, r.den / n, by sorry ⟩

theorem fraction_addition (a b : ℕ) (ha : 519 = a) (hb : 1600 = b) :
  0.324375 = (519 : ℚ) / 1600 ∧ a + b = 2119 :=
by {
  have fraction_rep : 0.324375 = (519 : ℚ) / 1600 := by sorry,
  have sum_ab : a + b = 2119 := by {
    rw [ha, hb],
    exact rfl,
  },
  exact ⟨fraction_rep, sum_ab⟩,
}

end fraction_addition_l58_58308


namespace difference_between_two_smallest_integers_l58_58324

-- We will use the concepts of modular arithmetic and LCM
theorem difference_between_two_smallest_integers (k : ℕ) (hk : 2 ≤ k ∧ k ≤ 12) :
  let n1 := nat.lcm (list.range' 2 11) + 1,
      n2 := 2 * nat.lcm (list.range' 2 11) + 1
  in n2 - n1 = 4620 := 
by
  sorry

end difference_between_two_smallest_integers_l58_58324


namespace evaluate_ratio_is_negative_two_l58_58979

noncomputable def evaluate_ratio (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : ℂ :=
  (a^15 + b^15) / (a + b)^15

theorem evaluate_ratio_is_negative_two (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^4 + a^2 * b^2 + b^4 = 0) : 
  evaluate_ratio a b h = -2 := 
sorry

end evaluate_ratio_is_negative_two_l58_58979


namespace inequality_holds_l58_58844

noncomputable def valid_inequality (a : ℝ) : Prop :=
  ∀ θ : ℝ, (0 ≤ θ) → (θ ≤ real.pi / 2) →
    (sin (2*θ) - (2*real.sqrt 2 + real.sqrt 2 * a) * sin (θ + real.pi / 4) - 
    (2*real.sqrt 2) / cos (θ - real.pi / 4) > -3 - 2 * a)

theorem inequality_holds (a : ℝ) : valid_inequality a → a > 3 :=
by
  intros h
  sorry

end inequality_holds_l58_58844


namespace zero_point_in_interval_l58_58305

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2 - 3 / x - 1

theorem zero_point_in_interval : 
  ∃ c ∈ Ioo (3 : ℝ) (4 : ℝ), f c = 0 :=
by {
  have f_mono : ∀ x y, 0 < x → x < y → y < 5 → f x < f y, sorry,
  have f_3 : f 3 < 0, sorry,
  have f_4 : f 4 > 0, sorry,
  exact exists_intermediate_value f_mono f_3 f_4,
}

end zero_point_in_interval_l58_58305


namespace sum_of_squares_l58_58603

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares_l58_58603


namespace power_calculation_l58_58060

noncomputable def a : ℕ := 3 ^ 1006
noncomputable def b : ℕ := 7 ^ 1007
noncomputable def lhs : ℕ := (a + b)^2 - (a - b)^2
noncomputable def rhs : ℕ := 42 * (10 ^ 1007)

theorem power_calculation : lhs = rhs := by
  sorry

end power_calculation_l58_58060


namespace num_elements_divisible_by_7_l58_58982

def isInSet (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  a ∈ {1, 2, 3, 4, 5, 6, 7} ∧ b ∈ {1, 2, 3, 4, 5, 6, 7} ∧ c ∈ {1, 2, 3, 4, 5, 6, 7}

def isDivisibleBy7 (n : ℕ) : Prop :=
  n % 7 = 0

def cyclesDivisibleBy7 (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  isDivisibleBy7 (2 * a + 3 * b + c) ∨
  isDivisibleBy7 (2 * b + 3 * c + a) ∨
  isDivisibleBy7 (2 * c + 3 * a + b)

theorem num_elements_divisible_by_7 : 
  (finset.filter (λ n, isInSet n ∧ cyclesDivisibleBy7 n) 
  (finset.range 1000)).card = 127 := sorry

end num_elements_divisible_by_7_l58_58982


namespace vegetation_coverage_relationship_l58_58712

noncomputable def conditions :=
  let n := 20
  let sum_x := 60
  let sum_y := 1200
  let sum_xx := 80
  let sum_xy := 640
  (n, sum_x, sum_y, sum_xx, sum_xy)

theorem vegetation_coverage_relationship
  (n sum_x sum_y sum_xx sum_xy : ℕ)
  (h1 : n = 20)
  (h2 : sum_x = 60)
  (h3 : sum_y = 1200)
  (h4 : sum_xx = 80)
  (h5 : sum_xy = 640) :
  let b1 := sum_xy / sum_xx
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  (b1 = 8) ∧ (b1 * (sum_xx / sum_xy) ≤ 1) ∧ ((3, 60) = (mean_x, mean_y)) :=
by
  sorry

end vegetation_coverage_relationship_l58_58712


namespace range_of_a2_l58_58118

theorem range_of_a2 (a : ℕ → ℝ) (S : ℕ → ℝ) (a2 : ℝ) (a3 a6 : ℝ) (h1: 3 * a3 = a6 + 4) (h2 : S 5 < 10) :
  a2 < 2 := 
sorry

end range_of_a2_l58_58118


namespace series_divisibility_l58_58643

theorem series_divisibility (n : ℕ) (h : n > 1) : 
  let S := ∑ i in finset.range n, (2^i - 1)^(2^i - 1) in
  (2^n ∣ S) ∧ ¬(2^(n+1) ∣ S) :=
sorry

end series_divisibility_l58_58643


namespace quinary_to_septenary_conversion_l58_58451

def quinary_to_decimal (n : ℕ) : ℕ := 
  4 * 5^2 + 1 * 5^1 + 2 * 5^0

def decimal_to_septenary (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc: list ℕ) :=
    if n = 0 then acc else aux (n / 7) ((n % 7) :: acc)
  in aux n []

theorem quinary_to_septenary_conversion : 
  decimal_to_septenary (quinary_to_decimal 412) = [2, 1, 2] :=
by
  sorry

end quinary_to_septenary_conversion_l58_58451


namespace min_value_PA_PF_l58_58295

-- Definitions of the parabola and the points
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def A := (6, 3) : ℝ × ℝ
def F := (1, 0) : ℝ × ℝ

-- Definition of distance between two points
def dist (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- The main assertion of the minimum value of |PA| + |PF|
theorem min_value_PA_PF (P : ℝ × ℝ) (hP : parabola P) :
  dist P A + dist P F = 7 :=
sorry

end min_value_PA_PF_l58_58295


namespace rectangle_side_excess_percentage_l58_58201

theorem rectangle_side_excess_percentage (A B : ℝ) (x : ℝ) (h : A * (1 + x) * B * (1 - 0.04) = A * B * 1.008) : x = 0.05 :=
by
  sorry

end rectangle_side_excess_percentage_l58_58201


namespace price_of_each_sundae_l58_58384

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ := 125) 
  (num_sundaes : ℕ := 125) 
  (total_price : ℝ := 225)
  (price_per_ice_cream_bar : ℝ := 0.60) :
  ∃ (price_per_sundae : ℝ), price_per_sundae = 1.20 := 
by
  -- Variables for costs of ice-cream bars and sundaes' total cost
  let cost_ice_cream_bars := num_ice_cream_bars * price_per_ice_cream_bar
  let total_cost_sundaes := total_price - cost_ice_cream_bars
  let price_per_sundae := total_cost_sundaes / num_sundaes
  use price_per_sundae
  sorry

end price_of_each_sundae_l58_58384


namespace diameter_circumscribed_eq_20_sqrt_2_l58_58189

noncomputable def diameter_circumscribed {a : ℝ} {A : ℝ} (h1 : a = 20) (h2 : A = Real.pi / 4) : ℝ :=
  a / Real.sin A

theorem diameter_circumscribed_eq_20_sqrt_2 : diameter_circumscribed (by simp) (by simp) = 20 * Real.sqrt 2 :=
sorry

end diameter_circumscribed_eq_20_sqrt_2_l58_58189


namespace PipeB_times_faster_l58_58641

noncomputable def PipeRates (R_A : ℝ) (n : ℝ) (R_B : ℝ) (Combined_Rate : ℝ) : Prop :=
  R_A = 1/20 ∧ 
  R_B = n * R_A ∧ 
  Combined_Rate = 1/4 ∧ 
  R_A + R_B = Combined_Rate

theorem PipeB_times_faster (R_A R_B Combined_Rate : ℝ) (n : ℝ) :
  PipeRates R_A n R_B Combined_Rate → n = 4 :=
by
  intro h
  cases h with hRA hRest
  cases hRest with hRB hRest
  cases hRest with hCR hSUM
  -- Proof would go here but is skipped
  sorry

end PipeB_times_faster_l58_58641


namespace geom_seq_sum_five_l58_58143

theorem geom_seq_sum_five
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (a1 r : ℝ)
  (ha2a3 : a_n 2 * a_n 3 = 2 * a_n 1)
  (ha4_2a7 : (a_n 4 + 2 * a_n 7) / 2) :
  (S_n 5 = 31) :=
by
  -- Definitions
  let a_n (n : ℕ) := a1 * r^(n-1)
  let S_n (n : ℕ) := a1 * (1 - r^n) / (1 - r)
 
  -- Given conditions
  have ha2 : a_n 2 = a1 * r := by sorry
  have ha3 : a_n 3 = a1 * r^2 := by sorry
  have hcond : a1 * r * a1 * r^2 = 2 * a1 := by sorry
  have hsimp : (a1 * r^3 = 2) := by sorry
  
  -- Arithmetic mean condition (not fully utilized in this form)
  have hmean : ((a1 * r^3 + 2 * a1 * r^6) / 2) := by sorry

  -- Conclusion
  exact 31

end geom_seq_sum_five_l58_58143


namespace max_value_of_sum_of_reciprocals_l58_58058

-- Definitions for points and conditions
variable {A P B C : Type}
variable [MetricSpace P]

variable (angle_A : Angle A)
variable (P_in_angle : P ∈ interior angle_A)
variable (line_P : Line P)
variable (B_on_line_A : B ∈ line_P)
variable (C_on_line_A : C ∈ line_P)

-- The main statement to be proved
theorem max_value_of_sum_of_reciprocals (PB PC : ℝ) (h_P : ∀ B C, P ∈ interior ⟨A, B, C⟩) :
  max_value (1 / PB + 1 / PC) = ⟨AP ⊥ BC⟩ :=
sorry

end max_value_of_sum_of_reciprocals_l58_58058


namespace mike_earnings_l58_58377

theorem mike_earnings (total_games non_working_games price_per_game : ℕ) 
  (h1 : total_games = 15) (h2 : non_working_games = 9) (h3 : price_per_game = 5) : 
  total_games - non_working_games * price_per_game = 30 :=
by
  rw [h1, h2, h3]
  show 15 - 9 * 5 = 30
  sorry

end mike_earnings_l58_58377


namespace product_identity_l58_58066

theorem product_identity (x y : ℝ) : (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l58_58066


namespace range_of_a_l58_58234

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (x^2 + (a + 2) * x + 1) * ((3 - 2 * a) * x^2 + 5 * x + (3 - 2 * a)) ≥ 0) : a ∈ Set.Icc (-4 : ℝ) 0 := sorry

end range_of_a_l58_58234


namespace inscribed_right_triangle_in_circle_l58_58591

open real

-- Definition of a circle with center (o : ℝ × ℝ) and radius r : ℝ
structure circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Given a circle and two points inside the circle, prove the existence of points C and D to form right triangles.
theorem inscribed_right_triangle_in_circle (c : circle) (A B : ℝ × ℝ)
  (hA : dist c.center A < c.radius)
  (hB : dist c.center B < c.radius) :
  ∃ C D : ℝ × ℝ, 
    (C ≠ D) ∧
    (dist c.center C = c.radius) ∧
    (dist c.center D = c.radius) ∧
    let ⟨p1, p2⟩ := segment_eq_intersect_circle c A B hA hB in
    (p1 = C ∧ p2 = D ∧ 
      (angle_eq_two_right_triangles A B C ∧ angle_eq_two_right_triangles A B D)) :=
sorry

-- Hypothetical definitions used in the theorem statement for illustration purposes.
-- segment_eq_intersect_circle and angle_eq_two_right_triangles are hypothetical examples representing the needed geometric constructs and theorems.
-- You would define these constructs in actual usage according to your geometric library in Lean.

end inscribed_right_triangle_in_circle_l58_58591


namespace taxi_ride_cost_l58_58420

def base_fare := 2.00
def cost_per_mile := 0.30
def total_miles := 8
def total_cost := base_fare + cost_per_mile * total_miles

theorem taxi_ride_cost :
  total_cost = 4.40 := 
by
  sorry

end taxi_ride_cost_l58_58420


namespace range_of_k_l58_58940

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
by
  sorry

end range_of_k_l58_58940


namespace min_marked_points_l58_58572

theorem min_marked_points (n : ℕ) (h : n = 2018) (h1 : ∀ (p q : ℕ) (hp : p < n) (hq : q < n), p ≠ q → distance p q ≠ distance q p) :
  ∃ m, m = 404 ∧ (∀ p < n, ∃ q < n, q ≠ p ∧ closest_point p = q) :=
by {
  sorry
}

end min_marked_points_l58_58572


namespace final_price_percentage_l58_58412

theorem final_price_percentage (P : ℝ) (h₀ : P > 0) : 
  let sale_price := P * 0.8,
      final_price := sale_price * 0.8 
  in final_price = P * 0.64 :=
by
  sorry

end final_price_percentage_l58_58412


namespace simplify_radicals_l58_58272

theorem simplify_radicals :
  (Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by 
  sorry

end simplify_radicals_l58_58272


namespace find_f_2008_l58_58863

noncomputable def f : ℝ → ℝ := sorry

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x, f (a - x) = f (a + x)

theorem find_f_2008 (f : ℝ → ℝ) 
  (h1 : ∀ x, f(x + 16) = f(x) + f(8))
  (h2 : symmetric_about (λ x, f (x + 1)) (-1)) :
  f(2008) = 0 :=
sorry

end find_f_2008_l58_58863


namespace dad_strawberries_now_weight_l58_58252

-- Definitions based on the conditions given
def total_weight : ℕ := 36
def weight_lost_by_dad : ℕ := 8
def weight_of_marco_strawberries : ℕ := 12

-- Theorem to prove the question as an equality
theorem dad_strawberries_now_weight :
  total_weight - weight_lost_by_dad - weight_of_marco_strawberries = 16 := by
  sorry

end dad_strawberries_now_weight_l58_58252


namespace sum_of_possible_radii_l58_58764

theorem sum_of_possible_radii :
  let r := ∀ r : ℝ, (C : C = r ∧ C (r - 4)^2 + r^2 = (r + 1)^2) 
  r = 5 + sqrt 10 ∨ r = 5 - sqrt 10
  => (5 + sqrt) + (5 - sqrt) = (2 * 5) := sorry

end sum_of_possible_radii_l58_58764


namespace part_a_part_b_part_c_l58_58816

-- Part (a): Prove the existence of a circle tangent to two lines and passing through a given point.
theorem part_a (l1 l2 : Line) (A : Point) : ∃ S : Circle, tangent S l1 ∧ tangent S l2 ∧ passes_through S A :=
sorry

-- Part (b): Prove the existence of a circle passing through two points and tangent to a given line.
theorem part_b (A B : Point) (l : Line) : ∃ S : Circle, passes_through S A ∧ passes_through S B ∧ tangent S l :=
sorry

-- Part (c): Prove the existence of a circle tangent to two lines and a given circle.
theorem part_c (l1 l2 : Line) (\bar{S} : Circle) : ∃ S : Circle, tangent S l1 ∧ tangent S l2 ∧ tangent S \bar{S} :=
sorry

end part_a_part_b_part_c_l58_58816


namespace unique_six_digit_number_l58_58836

theorem unique_six_digit_number : ∃! (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = nat.reverse_digits n := by
  sorry

end unique_six_digit_number_l58_58836


namespace smallest_positive_period_l58_58824

def f (x : ℝ) : ℝ := sin (2 * x + (Real.pi / 4))

theorem smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f(x) = f(x + T) ∧ T = Real.pi := by
  sorry

end smallest_positive_period_l58_58824


namespace nancy_antacids_l58_58256

theorem nancy_antacids :
  ∀ (x : ℕ),
  (3 * 3 + x * 2 + 1 * 2) * 4 = 60 → x = 2 :=
by
  sorry

end nancy_antacids_l58_58256


namespace length_OP_l58_58856

-- Definitions and conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, 1)
def AP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2 - 2)
def AB : ℝ × ℝ := (3, -1)
axiom AP_relation (P : ℝ × ℝ) : 2 * AP P = AB

-- Proposition to prove
theorem length_OP (P : ℝ × ℝ) (h : AP_relation P) : |(P.1, P.2)| = (√10) / 2 :=
by
  sorry

end length_OP_l58_58856


namespace film_cost_eq_five_l58_58222

variable (F : ℕ)

theorem film_cost_eq_five (H1 : 9 * F + 4 * 4 + 6 * 3 = 79) : F = 5 :=
by
  -- This is a placeholder for your proof
  sorry

end film_cost_eq_five_l58_58222


namespace not_coplanar_vectors_l58_58172

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Defining the vectors and the statement
variables (a b c : V)

theorem not_coplanar_vectors 
  (h_basis : linear_independent ℝ ![a, b, c]) :
  ¬ coplanar ℝ ![a + b, a - b, c] :=
sorry

end not_coplanar_vectors_l58_58172


namespace consecutive_numbers_polynomial_l58_58116

theorem consecutive_numbers_polynomial :
  ∃ (a : ℕ) (P : ℕ → ℕ),
    (∀ n, P n = n + (n - a) * (n - (a + 6)) * 
                    (n - (a + 1)) * (n - (a + 2)) * (n - (a + 5))) ∧
    (∀ k ∈ {a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6}, 
       (P k = k ∨ P k = 0)) ∧
    {a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6} = {14, 15, 16, 17, 18, 19, 20} :=
begin
  sorry
end

end consecutive_numbers_polynomial_l58_58116


namespace jordan_walk_distance_l58_58221

theorem jordan_walk_distance
  (d t : ℝ)
  (flat_speed uphill_speed walk_speed : ℝ)
  (total_time : ℝ)
  (h1 : flat_speed = 18)
  (h2 : uphill_speed = 6)
  (h3 : walk_speed = 4)
  (h4 : total_time = 3)
  (h5 : d / (3 * 18) + d / (3 * 6) + d / (3 * 4) = total_time) :
  t = 6.6 :=
by
  -- Proof goes here
  sorry

end jordan_walk_distance_l58_58221


namespace function_satisfaction_l58_58051

theorem function_satisfaction 
(f : ℝ → ℝ) (hf : ∀ x : ℝ, f(x) + f(-x) = 0) (hdf : ∀ x : ℝ, deriv f x ≤ 0) : 
  f = λ x, -x * exp (abs x) := 
sorry

end function_satisfaction_l58_58051


namespace evaluate_f_of_f_l58_58486

def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 
  else -x

theorem evaluate_f_of_f : f (f (-2)) = 4 := by
  sorry

end evaluate_f_of_f_l58_58486


namespace parabola_focus_distance_correct_l58_58379

noncomputable theory

def parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ) (F : ℝ × ℝ) (Hp : p > 0) (Hparabola : M.2^2 = 2 * p * M.1) : ℝ :=
let dist := real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) in dist

theorem parabola_focus_distance_correct :
  parabola_focus_distance (2, 2) 1 (1/2, 0) (by norm_num) (by norm_num) = 5 / 2 :=
by {
  sorry
}

end parabola_focus_distance_correct_l58_58379


namespace exists_six_different_cascades_can_color_nat_12_colors_l58_58819

-- Definitions
def cascade (r : ℕ) : set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 12 ∧ n = k * r}

-- Part (a)
theorem exists_six_different_cascades (a b : ℕ) :
  ∃ rs : fin 6 → ℕ, ∀ i : fin 6, a ∈ cascade (rs i) ∧ b ∈ cascade (rs i) :=
sorry

-- Part (b)
theorem can_color_nat_12_colors :
  ∃ (coloring : ℕ → fin 12), ∀ r : ℕ, function.bijective (λ k : fin 12, coloring (k * r)) :=
sorry

end exists_six_different_cascades_can_color_nat_12_colors_l58_58819


namespace proof_CE_BF_l58_58855

variables {W A E F T Z : Point} {b c : ℝ}
variables (AC AB AE AF : Line) (WT WZ : PerpendicularLine) (AT AZ : Segment)

-- Conditions
def condition1 : Center(W) ∧ Radius(W, A) := sorry
def condition2 : Intersect(AC.extension, circle(W, WA)) = E ∧ Intersect(AB.extension, circle(W, WA)) = F := sorry
def condition3 : Perpendicular(W, T, AE) ∧ Perpendicular(W, Z, AF) := sorry
def condition4 : AT = AE / 2 := sorry
def condition5 : AZ = AF / 2 := sorry
def condition6 : AT = AZ := sorry
def condition7 : Length(AC) = b := sorry
def condition8 : Length(AB) = c := sorry

-- Proof goal
theorem proof_CE_BF : Length(CE) = c ∧ Length(BF) = b :=
by
  intro hc
  have h1 : condition1 := sorry
  have h2 : condition2 := sorry
  have h3 : condition3 := sorry
  have h4 : condition4 := sorry
  have h5 : condition5 := sorry
  have h6 : condition6 := sorry
  have h7 : condition7 := sorry
  have h8 : condition8 := sorry
  sorry

end proof_CE_BF_l58_58855


namespace sue_final_answer_l58_58421

-- Definitions based on conditions
def ben_initial_number : ℕ := 8
def ben_result (n : ℕ) : ℕ := 3 * (n + 3)
def sue_result (n : ℕ) : ℕ := 3 * (n - 3)

-- Lean 4 statement of the problem
theorem sue_final_answer : sue_result (ben_result ben_initial_number) = 90 :=
by {
  -- calculation steps for the proof problem, add 'sorry' to indicate it is for proof placeholder
  sorry,
}

end sue_final_answer_l58_58421


namespace initial_fraction_filled_time_to_complete_fill_l58_58765

structure CubicalTank where
  side_length : ℝ
  initial_volume : ℝ
  fill_rate : ℝ

variables (tank : CubicalTank)

-- Given conditions
def initial_water_level : ℝ := 1 -- initial water level is 1 foot
def initial_filled_fraction : ℝ := tank.initial_volume / (tank.side_length ^ 3)
def total_capacity : ℝ := tank.side_length ^ 3
def remaining_volume_to_fill : ℝ := total_capacity - tank.initial_volume
def time_to_fill_remaining : ℝ := remaining_volume_to_fill / tank.fill_rate

open Real

-- Main proof obligations
theorem initial_fraction_filled (h1 : tank.initial_volume = 16)
                                 (h2 : tank.side_length = sqrt 16)
                                 : initial_filled_fraction tank = 1 / 4 := by
  sorry

theorem time_to_complete_fill (h1 : tank.initial_volume = 16)
                              (h2 : tank.side_length = sqrt 16)
                              (h3 : tank.fill_rate = 2)
                              : time_to_fill_remaining tank = 24 := by
  sorry

end initial_fraction_filled_time_to_complete_fill_l58_58765


namespace four_digit_numbers_count_l58_58163

theorem four_digit_numbers_count :
  let primes := [2, 3, 5, 7]
  let perfect_squares := [0, 1, 4, 9]
  let count := primes.length * 9 * 8 * 1
  count = 288 :=
by
  let primes := [2, 3, 5, 7]
  let perfect_squares := [0, 1, 4, 9]
  let count := primes.length * 9 * 8 * 1
  show count = 288 from sorry

end four_digit_numbers_count_l58_58163


namespace line_intersects_all_convex_sets_l58_58471

open set

variables (C : ℕ → set (euclidean_space ℝ 3))

def three_intersect (C : ℕ → set (euclidean_space ℝ 3)) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → (C i ∩ C j ∩ C k).nonempty

theorem line_intersects_all_convex_sets (hC : ∀ n, is_convex ℝ (C n)) 
                                        (h_finite : ∃ n, ∀ m > n, C m = ∅) 
                                        (h_three_intersect : three_intersect C) :
  ∃ L : set (euclidean_space ℝ 3), is_line L ∧ ∀ n, (L ∩ C n).nonempty :=
sorry

end line_intersects_all_convex_sets_l58_58471


namespace certain_number_eq_neg17_l58_58727

theorem certain_number_eq_neg17 (x : Int) : 47 + x = 30 → x = -17 := by
  intro h
  have : x = 30 - 47 := by
    sorry  -- This is just to demonstrate the proof step. Actual manipulation should prove x = -17
  simp [this]

end certain_number_eq_neg17_l58_58727


namespace bricklayer_problem_l58_58713

theorem bricklayer_problem : ∀ (x : ℕ),
  (0 < x) →
  (let rate1 := x / 8 in
   let rate2 := x / 12 in
   let combined_rate := rate1 + rate2 - 15 in
   let total_time := 6 in
   total_time * combined_rate = x) →
  x = 360 :=
by
  intro x hx
  simp only []
  sorry

end bricklayer_problem_l58_58713


namespace tangent_product_constant_l58_58954

noncomputable def curve_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 5)^2 = 9

noncomputable def curve_C1 (x y : ℝ) : Prop :=
  y^2 = 20 * x

theorem tangent_product_constant (x y x₀ k₁ k₂ : ℝ) (h1 : curve_C2 x y) 
  (h2 : y = -4) (h3 : x₀ ≠ 3) (h4 : x₀ ≠ -3) 
  (h5 : k₁ + k₂ = - 18 * x₀ / (x₀ ^ 2 - 9))
  (h6 : k₁ * k₂ = - 72 / (x₀ ^ 2 - 9)) : 
  let x₁ := 20 * (k₁ * x₀ + 4),
      x₂ := 20 * (k₂ * x₀ + 4),
      x₃ := 20 * (k₁ * x₀ + 4),
      x₄ := 20 * (k₂ * x₀ + 4) in
  x₁ * x₂ * x₃ * x₄ = 6400 :=
by
  sorry

end tangent_product_constant_l58_58954


namespace geometric_sequence_general_term_sequence_sum_l58_58864

noncomputable def geometric_sequence (n : ℕ) : ℕ :=
  3^(n-2)

theorem geometric_sequence_general_term :
  (∀ (a_n : ℕ → ℝ) (a_1 a_2 a_3 a_4 a_5 : ℝ) (q : ℝ),
    a_1 * a_2 * a_3 * a_4 * a_5 = 243 ∧
    2 * a_3 = (3 * a_2 + a_4) / 2 ∧
    a_2 = a_3 / q ∧
    a_4 = a_3 * q ∧
    a_3^5 = 243 ∧
    a_3 = 3 ∧
    q = 3 → 
    ∀ (n : ℕ), a_n = geometric_sequence n) :=
sorry

theorem sequence_sum :
  (∀ (b : ℕ → ℝ) (b_1 : ℝ),
    b 1 = 1 ∧
    (∀ n : ℕ, 2 ≤ n → b n = b (n - 1) * log 3 (geometric_sequence (n + 2))) →
    ∀ n : ℕ,
    ∑ k in finset.range n, (n - 1)! / b (n + 1) = n / (n + 1)) :=
sorry

end geometric_sequence_general_term_sequence_sum_l58_58864


namespace safer_4_engines_l58_58435

variable (P : ℝ)
-- The conditions:
-- 1. Failure rate of each airplane engine during flight is \( 1 - P \).
def failure_rate (P : ℝ) : ℝ := 1 - P

-- 2. The events of each engine failing are independent.
-- This will be inherent in our probability calculations.

-- 3. Plane flies if at least 50% of engines are working normally:
-- Four-engine plane: at least 2 working.
-- Two-engine plane: at least 1 working.

-- Probability formula for binomial distributions:
def binomial (n k : ℕ) : ℝ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

def success_4_engines (P : ℝ) : ℝ :=
  let p := 1 - P in
  (binomial 4 4) * (p ^ 4) +
  (binomial 4 3) * (p ^ 3) * ((1 - p) ^ 1) +
  (binomial 4 2) * (p ^ 2) * ((1 - p) ^ 2)

def success_2_engines (P : ℝ) : ℝ :=
  let p := 1 - P in
  (binomial 2 2) * (p ^ 2) +
  (binomial 2 1) * (p ^ 1) * ((1 - p) ^ 1)

-- The main theorem to prove:
theorem safer_4_engines (P : ℝ) (h : P ∈ Ioo (2 / 3) 1) :
  (success_4_engines P) > (success_2_engines P) :=
sorry

end safer_4_engines_l58_58435


namespace figure_area_l58_58675

-- Define the set of points (x, y) satisfying the given conditions
def figure (x y : ℝ) : Prop :=
  ∀ t : ℝ, x^2 + y^2 < Real.pi^2 + t^2 ∧
           Real.cos y < 2 + Real.cos (2 * x) + Real.cos x * (4 * Real.sin t - 1) - Real.cos (2 * t)

noncomputable def area_of_figure : ℝ :=
  π^3 - 2 * π^2

theorem figure_area :
  ∃ A, ∀ x y, figure x y → A = π^3 - 2 * π^2 :=
sorry

end figure_area_l58_58675


namespace second_half_time_l58_58043

-- Define our conditions as constants
constant initial_speed : ℝ
constant distance : ℝ := 40
constant halfway_distance : ℝ := distance / 2
constant time_difference : ℝ := 4

-- Define times for the first and second halves of the run
def t1 : ℝ := halfway_distance / initial_speed
def t2 : ℝ := halfway_distance / (initial_speed / 2)

-- Define our main theorem to state the desired outcome
theorem second_half_time : t2 = t1 + 4 → t2 = 8 :=
by
  -- This is where we would provide our proof
  sorry

end second_half_time_l58_58043


namespace telephone_number_problem_l58_58788

theorem telephone_number_problem
    (A B C D E F G H I J : ℕ)
    (h_distinct : ∀ x y : ℕ, ({A, B, C, D, E, F, G, H, I, J} : set ℕ).pairwise (≠))
    (h_ABC : A > B ∧ B > C)
    (h_DEF : D > E ∧ E > F)
    (h_GHIJ : G > H ∧ H > I ∧ I > J)
    (h_consecutive_odd_DEF : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
    (h_consecutive_even_GHIJ : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
    (h_sum_ABC : A + B + C = 11) :
    A = 7 :=
sorry

end telephone_number_problem_l58_58788


namespace coin_flip_tails_probability_l58_58015

open Nat

noncomputable def binom (n k : ℕ) : ℕ := choose n k

theorem coin_flip_tails_probability :
  ∃ n : ℕ, (n(n-1)) / 2^(n+1) = 1 / 32 ∧ (∃ k : ℕ, k = binom n 2)

end coin_flip_tails_probability_l58_58015


namespace total_votes_cast_l58_58365

theorem total_votes_cast (V : ℕ) (h1 : V > 0) (h2 : ∃ c r : ℕ, c = 40 * V / 100 ∧ r = 40 * V / 100 + 5000 ∧ c + r = V):
  V = 25000 :=
by
  sorry

end total_votes_cast_l58_58365


namespace length_of_rhombus_side_l58_58141

theorem length_of_rhombus_side :
  (∃ (d₁ d₂ : ℝ), (d₁ ≠ d₂ ∧ (d₁^2 - 3*d₁ + 2 = 0) ∧ (d₂^2 - 3*d₂ + 2 = 0))
    → ∃ s : ℝ, s = sqrt ((1/2) * (d₁/2)^2 + (1/2) * (d₂/2)^2) ∧ s = sqrt(5) / 2 ) :=
sorry

end length_of_rhombus_side_l58_58141


namespace cos_minus_sin_l58_58510

theorem cos_minus_sin (α : ℝ) (h1 : Real.sin (2 * α) = 1 / 4) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.cos α - Real.sin α = - (Real.sqrt 3) / 2 :=
sorry

end cos_minus_sin_l58_58510


namespace smallest_positive_integer_divisible_conditions_l58_58355

theorem smallest_positive_integer_divisible_conditions :
  ∃ (M : ℕ), M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M = 419 :=
sorry

end smallest_positive_integer_divisible_conditions_l58_58355


namespace find_matrix_N_l58_58842

def N : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![ -1, 3, 5], 
    ![ 4, 0, -3],
    ![ 9, 6, 2]]

def i : Vector ℤ (Fin 3) := ![1, 0, 0]
def j : Vector ℤ (Fin 3) := ![0, 1, 0]
def k : Vector ℤ (Fin 3) := ![0, 0, 1]

theorem find_matrix_N : 
  (N.mulVec i = ![-1, 4, 9]) ∧ 
  (N.mulVec j = ![3, 0, 6]) ∧ 
  (N.mulVec k = ![5, -3, 2]) := by
  sorry

end find_matrix_N_l58_58842


namespace dot_product_of_vectors_l58_58191

theorem dot_product_of_vectors
  (AB BC CA : ℝ)
  (h1 : AB = 7)
  (h2 : BC = 5)
  (h3 : CA = 6)
  : (let cosB := (AB^2 + BC^2 - CA^2) / (2 * AB * BC) in
     -AB * BC * cosB = -19) :=
by
  let cosB := (AB^2 + BC^2 - CA^2) / (2 * AB * BC)
  rw [h1, h2, h3]
  sorry

end dot_product_of_vectors_l58_58191


namespace hexadecagon_diagonals_l58_58162

/-- The number of diagonals in a convex n-sided polygon is given by the formula
  n * (n - 3) / 2. We need to prove that for n = 16, the number of diagonals is 104. -/
theorem hexadecagon_diagonals : 
  let n := 16 in (n * (n - 3)) / 2 = 104 := 
by
  let n := 16
  have h1 : n * (n - 3) = 16 * 13 := rfl
  have h2 : (16 * 13) / 2 = 104 := rfl
  exact h2

end hexadecagon_diagonals_l58_58162


namespace intersection_A_complement_B_l58_58483

def A := {y : ℕ | ∃ x : ℤ, y = -x^2 + 4 * x}
def B := {x : ℝ | Real.log x > 1}
def complement_B := {x : ℝ | x ≤ Real.exp 1}

theorem intersection_A_complement_B :
  A ∩ (complement_B : Set ℕ) = {0} :=
by
  sorry

end intersection_A_complement_B_l58_58483


namespace chessboard_rice_difference_l58_58775

theorem chessboard_rice_difference :
  let grains (k : ℕ) := 2^k in
  let square_12 := grains 12 in
  let sum_first_10 := (Finset.range 10).sum (λ k => grains (k+1)) in
  square_12 - sum_first_10 = 2050 :=
by
  let grains (k : ℕ) := 2^k
  let square_12 := grains 12
  let sum_first_10 := (Finset.range 10).sum (λ k => grains (k+1))
  have h1 : square_12 = 4096 := by sorry -- Proof skipped
  have h2 : sum_first_10 = 2046 := by sorry -- Proof skipped
  rw [h1, h2]
  norm_num

end chessboard_rice_difference_l58_58775


namespace ms_hatcher_students_l58_58624

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l58_58624


namespace abs_neg_one_tenth_l58_58281

theorem abs_neg_one_tenth : |(-1 : ℚ) / 10| = 1 / 10 :=
by
  sorry

end abs_neg_one_tenth_l58_58281


namespace smallest_n_for_swiss_cross_l58_58754

-- Define a Swiss cross
structure SwissCross where
  -- Swiss cross consists of five unit squares, one in the center and four on the sides.
  unit_squares : Fin 5 → set (ℝ × ℝ)

-- Define the condition for the points
def points_in_swiss_cross (sc : SwissCross) (points : Fin n → (ℝ × ℝ)) : Prop :=
  ∀ (i : Fin n), ∃ (j : Fin 5), points i ∈ sc.unit_squares j

-- Define the distance condition
def pairwise_distance_less_than_one (points : Fin n → (ℝ × ℝ)) : Prop :=
  ∃ (i j : Fin n), i ≠ j ∧ dist (points i) (points j) < 1

-- Prove the smallest n for which the condition holds for any placement of points in the Swiss cross
theorem smallest_n_for_swiss_cross (sc : SwissCross) :
  ∃ (n : ℕ), (∀ (points : Fin n → (ℝ × ℝ)), points_in_swiss_cross sc points → pairwise_distance_less_than_one points) ∧ 
  ∀ (m : ℕ), (m < n → ∃ (points : Fin m → (ℝ×ℝ)), points_in_swiss_cross sc points ∧ ¬pairwise_distance_less_than_one points) :=
sorry

end smallest_n_for_swiss_cross_l58_58754


namespace integer_solutions_of_system_l58_58275

theorem integer_solutions_of_system :
  { (x : ℤ) × (y : ℤ) × (z : ℤ) | x^2 - y^2 = z ∧ 3 * x * y + (x - y) * z = z^2 } =
  { (2, 1, 3), (1, 2, -3), (1, 0, 1), (0, 1, -1), (0, 0, 0) } :=
by
  sorry

end integer_solutions_of_system_l58_58275


namespace decision_has_two_exit_paths_l58_58731

-- Define types representing different flowchart symbols
inductive FlowchartSymbol
| Terminal
| InputOutput
| Process
| Decision

-- Define a function that states the number of exit paths given a flowchart symbol
def exit_paths (s : FlowchartSymbol) : Nat :=
  match s with
  | FlowchartSymbol.Terminal   => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process    => 1
  | FlowchartSymbol.Decision   => 2

-- State the theorem that Decision has two exit paths
theorem decision_has_two_exit_paths : exit_paths FlowchartSymbol.Decision = 2 := by
  sorry

end decision_has_two_exit_paths_l58_58731


namespace jenny_profit_l58_58596

-- Define the constants given in the problem
def cost_per_pan : ℝ := 10.00
def price_per_pan : ℝ := 25.00
def num_pans : ℝ := 20.0

-- Define the total revenue function
def total_revenue (num_pans : ℝ) (price_per_pan : ℝ) : ℝ := num_pans * price_per_pan

-- Define the total cost function
def total_cost (num_pans : ℝ) (cost_per_pan : ℝ) : ℝ := num_pans * cost_per_pan

-- Define the profit function as the total revenue minus the total cost
def total_profit (num_pans : ℝ) (price_per_pan : ℝ) (cost_per_pan : ℝ) : ℝ := 
  total_revenue num_pans price_per_pan - total_cost num_pans cost_per_pan

-- The statement to prove in Lean
theorem jenny_profit : total_profit num_pans price_per_pan cost_per_pan = 300.00 := 
by 
  sorry

end jenny_profit_l58_58596


namespace perpendicular_bisector_parallel_line_through_intersection_l58_58378

section perpendicular_bisector

variables (A B : ℝ × ℝ) (a b c : ℝ)
variables (x y : ℝ)

-- Conditions for Part 1
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Perpendicular bisector equation of segment AB
theorem perpendicular_bisector (A B : ℝ × ℝ) :
  A = (7, -4) ∧ B = (-5, 6) →
  (∀ x y : ℝ, 6 * x - 5 * y - 1 = 0) := sorry

end perpendicular_bisector

section parallel_line

-- Conditions for Part 2
def intersection (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  let (x, y) := (3, 2) in (x, y)  -- This is where we solve the system

-- Parallel line passing through the intersection
theorem parallel_line_through_intersection :
  (∀ x y : ℝ, 2 * x + y - 8 = 0) ∧ (∀ x y : ℝ, x - 2 * y + 1 = 0) ∧
  (∀ x y : ℝ, 4 * x - 3 * y - 7 = 0) →
  (∀ x y : ℝ, 4 * x - 3 * y - 6 = 0) := sorry

end parallel_line

end perpendicular_bisector_parallel_line_through_intersection_l58_58378


namespace new_area_is_726_l58_58038

variable (l w : ℝ)
variable (h_area : l * w = 576)
variable (l' : ℝ := 1.20 * l)
variable (w' : ℝ := 1.05 * w)

theorem new_area_is_726 : l' * w' = 726 := by
  sorry

end new_area_is_726_l58_58038


namespace triangle_similarity_length_l58_58333

theorem triangle_similarity_length
  (M N P X Y Z : Point)
  (h_similar : Similar (△ M N P) (△ X Y Z))
  (h_MN : dist M N = 8)
  (h_NP : dist N P = 20)
  (h_YZ : dist Y Z = 30) :
  dist X Y = 12 :=
sorry

end triangle_similarity_length_l58_58333


namespace sufficient_condition_even_function_l58_58153

noncomputable def f (x : ℝ) (α : ℝ) : ℝ :=
if x ≤ 0 then sin (x + α) else cos (x + α)

theorem sufficient_condition_even_function (α : ℝ) :
(α = π / 4) → (∀ x : ℝ, f x α = f (-x) α) :=
by
  sorry

end sufficient_condition_even_function_l58_58153


namespace value_range_of_log_function_l58_58320

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 2*x + 4

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  Real.log x / Real.log 3

theorem value_range_of_log_function :
  ∀ x : ℝ, log_base_3 (quadratic_function x) ≥ 1 := by
  sorry

end value_range_of_log_function_l58_58320


namespace delta_x_not_zero_l58_58581

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (delta_x : ℝ) : ℝ :=
  (f (x + delta_x) - f x) / delta_x

theorem delta_x_not_zero (f : ℝ → ℝ) (x delta_x : ℝ) (h_neq : delta_x ≠ 0):
  average_rate_of_change f x delta_x ≠ 0 := 
by
  sorry

end delta_x_not_zero_l58_58581


namespace max_f_at_1_f_inequality_l58_58149

-- Define the function f(x)
def f (a b x : ℝ) := a * log x + 0.5 * b * x^2 - (b + a) * x

-- Problem (I): Maximum value of f(x) when a = 1 and b = 0
theorem max_f_at_1 (x : ℝ) (h_pos : 0 < x) : f 1 0 x ≤ f 1 0 1 :=
by sorry -- Prove that f(x) reaches its maximum value at x = 1

-- Problem (II): Prove the inequality for f(x) when b = 1
theorem f_inequality 
  (a : ℝ) (e : ℝ) (h_e : real.exp 1 = e)  
  (h1 : 1 < a) (h2 : a ≤ e) (x1 x2 : ℝ) (h3 : 1 ≤ x1) (h4 : x1 ≤ a) (h5 : 1 ≤ x2) (h6 : x2 ≤ a) :
  |f a 1 x1 - f a 1 x2| < 1 :=
by sorry -- Prove the required inequality

end max_f_at_1_f_inequality_l58_58149


namespace find_four_digit_numbers_l58_58085

theorem find_four_digit_numbers :
  ∃ N : ℤ, 
    (1000 ≤ N ∧ N ≤ 9999) ∧ 
    (N % 2 = 0) ∧ 
    (N % 3 = 1) ∧ 
    (N % 5 = 3) ∧ 
    (N % 7 = 5) ∧ 
    (N % 11 = 9) :=
by
  use [2308, 4618, 6928, 9238]
  sorry

end find_four_digit_numbers_l58_58085


namespace friends_contribution_l58_58198

theorem friends_contribution (x : ℝ) 
  (h1 : 4 * (x - 5) = 0.75 * 4 * x) : 
  0.75 * 4 * x = 60 :=
by 
  sorry

end friends_contribution_l58_58198


namespace sum_of_real_solutions_l58_58093

theorem sum_of_real_solutions :
  (∑ x in {x : ℝ | sqrt x + sqrt (4 / x) + sqrt (x + 4 / x) = 6}.to_finset, x) = 64 / 9 :=
sorry

end sum_of_real_solutions_l58_58093


namespace arithmetic_sequence_sum_l58_58498

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) (S : ℕ → ℕ) (d : ℕ) : 
  (∀ n, a n = a 1 + (n - 1) * d) → 
  S 13 = 13 * a 1 + (13 * 12) / 2 * d → 
  S 13 = 39 → 
  a 6 + a 7 + a 8 = 9 :=
by
  intros a_def S_def S_val,
  have h1 : a 6 = a 1 + 5 * d := sorry,
  have h2 : a 7 = a 1 + 6 * d := sorry,
  have h3 : a 8 = a 1 + 7 * d := sorry,
  have h4 : S 13 = 39 := S_val,
  have h5 : 13 * a 1 + 78 * d = 39 := S_def,
  have h6 : a_1 + 6 * d = 3 := sorry,
  calc
    a 6 + a 7 + a 8 = (a 1 + 5 * d) + (a 1 + 6 * d) + (a 1 + 7 * d) : by rw [h1, h2, h3]
    ... = 3 * a 1 + 18 * d : by ring
    ... = 3 * (a 1 + 6 * d) : by ring
    ... = 3 * 3 : by rw h6
    ... = 9 : by norm_num

end arithmetic_sequence_sum_l58_58498


namespace num_positive_integers_satisfying_condition_l58_58165

theorem num_positive_integers_satisfying_condition :
  {n : ℕ // (150 * n) ^ 40 > n ^ 80 ∧ n ^ 80 > 3 ^ 240}.card = 122 :=
sorry

end num_positive_integers_satisfying_condition_l58_58165


namespace find_sin_2a_l58_58875

noncomputable def problem_statement (a : ℝ) : Prop :=
a ∈ Set.Ioo (Real.pi / 2) Real.pi ∧
3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin ((Real.pi / 4) - a)

theorem find_sin_2a (a : ℝ) (h : problem_statement a) : Real.sin (2 * a) = -8 / 9 :=
sorry

end find_sin_2a_l58_58875


namespace distance_sum_l58_58123

-- Definitions for points and centroids
variable {Point : Type}
variable [MetricSpace Point]

-- Given triangle vertices A, B, C and a point X outside the triangle, and S is the centroid
variable (A B C X S : Point)

-- S is the centroid of triangle ABC
variable (hS : S = centroid [A, B, C])

-- The goal is to prove the given equation
theorem distance_sum (hX : X ∉ triangle [A, B, C]) :
  dist A X ^ 2 + dist B X ^ 2 + dist C X ^ 2 =
  dist S A ^ 2 + dist S B ^ 2 + dist S C ^ 2 + 3 * (dist S X ^ 2) :=
by sorry

end distance_sum_l58_58123


namespace sector_area_72_20_eq_80pi_l58_58292

open Real

def sectorArea (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * π * r^2

theorem sector_area_72_20_eq_80pi :
  sectorArea 72 20 = 80 * π := by
  sorry

end sector_area_72_20_eq_80pi_l58_58292


namespace infinite_sum_problem_l58_58457

theorem infinite_sum_problem :
  (∑' n : ℕ, if n = 0 then 0 else (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = (1 / 4) := 
by
  sorry

end infinite_sum_problem_l58_58457


namespace a2_value_general_term_a_sum_T_l58_58872

section sequence_problem

variable {n : ℕ}
def a : ℕ → ℝ
def S : ℕ → ℝ
def b : ℕ → ℝ
def T : ℕ → ℝ

-- Given conditions
axiom increasing_a : ∀ n m, n < m → a n < a m
axiom a1 : a 1 = 2
axiom Sn : ∀ n ≥ 2, a n ^ 2 + 2 = 3 * (S n + S (n - 1))
axiom log_b : ∀ n, log 2 (b n / a n) = n

-- Problem statements
theorem a2_value : a 2 = 5 := sorry

theorem general_term_a : ∀ n, a n = 3 * n - 1 := sorry

theorem sum_T : ∀ n, T n = 8 + (3 * n - 4) * 2^(n+1) := sorry

end sequence_problem

end a2_value_general_term_a_sum_T_l58_58872


namespace width_decrease_percentage_l58_58779

noncomputable def original_length : ℝ := 140
noncomputable def original_width : ℝ := 40
noncomputable def original_area : ℝ := original_length * original_width
noncomputable def new_length : ℝ := original_length * 1.3
noncomputable def new_width : ℝ := original_area / new_length
noncomputable def percentage_decrease : ℝ := ((original_width - new_width) / original_width) * 100

theorem width_decrease_percentage : percentage_decrease ≈ 23.075 := 
by
  sorry

end width_decrease_percentage_l58_58779


namespace telephone_pole_height_l58_58047

noncomputable def height_of_pole
  (AC AD DE : ℝ) (h_AC : AC = 5) (h_AD : AD = 3) (h_DE : DE = 1.6)
  (AB : ℝ) (h_triangle_sim : (AB / AC) = (DE / (AC - AD))) : Prop :=
  AB = 4

theorem telephone_pole_height :
  ∃ (AB AC AD DE : ℝ),
    AC = 5 ∧
    AD = 3 ∧
    DE = 1.6 ∧
    (AB / AC) = (DE / (AC - AD)) ∧
    AB = 4 :=
by
  use 4, 5, 3, 1.6
  split; [refl, split; [refl, split; [refl, split; [refl, refl]]]]
  sorry

end telephone_pole_height_l58_58047


namespace problem_1_problem_2_l58_58382

variables (e₁ e₂ : Vector) (A B C D : Point)
  (h_nonzero_e1 : e₁ ≠ 0)
  (h_nonzero_e2 : e₂ ≠ 0)
  (h_not_collinear : ¬Collinear e₁ e₂)
  (AB BC CD : Vector)

def condition_1 := AB = 2 • e₁ + 3 • e₂
def condition_2 := BC = 6 • e₁ + 23 • e₂
def condition_3 := CD = 4 • e₁ - 8 • e₂

theorem problem_1 :
  condition_1 e₁ e₂ AB →
  condition_2 e₁ e₂ BC →
  condition_3 e₁ e₂ CD →
  Collinear (A,B,D) := 
by intros; sorry

variables (k : ℝ) (λ : ℝ)
def condition_4 := AB = 2 • e₁ + k • e₂
def condition_5 := CB = e₁ + 3 • e₂
def condition_6 := CD = 2 • e₁ - e₂
def condition_7 := Collinear (A, B, D)

theorem problem_2 : 
  condition_4 e₁ e₂ AB k →
  condition_5 e₁ e₂ CB →
  condition_6 e₁ e₂ CD →
  condition_7 (A, B, D) →
  k = -8 :=
by intros; sorry

end problem_1_problem_2_l58_58382


namespace f_2_pow_2011_l58_58676

-- Define the function
def f (x : ℝ) : ℝ := sorry -- Placeholder for the actual function definition

-- The conditions
axiom fx_positive (x : ℝ) (hx : 0 < x) : 0 < f(x)
axiom f1_f2_sum : f(1) + f(2) = 10
axiom f_add (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : f(a + b) = f(a) + f(b) + 2 * real.sqrt (f(a) * f(b))

-- The goal
theorem f_2_pow_2011 : f(2 ^ 2011) = 2 ^ 4023 :=
sorry

end f_2_pow_2011_l58_58676


namespace two_dice_probability_sum_greater_than_eight_l58_58717

def probability_sum_greater_than_eight : ℚ := 5 / 18

theorem two_dice_probability_sum_greater_than_eight :
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let favorable := [(3,6), (4,5), (4,6), (5,4), (5,5), (5,6), (6,3), (6,4), (6,5), (6,6)] in
  outcomes.length = 36 ∧ favorable.length = 10 →
  favorable.length / outcomes.length = probability_sum_greater_than_eight := 
by
  intros
  sorry

end two_dice_probability_sum_greater_than_eight_l58_58717


namespace no_injective_function_satisfying_conditions_l58_58592

open Real

theorem no_injective_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)
  ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x : ℝ, f (x ^ 2) - (f (a * x + b)) ^ 2 ≥ 1 / 4) :=
by
  sorry

end no_injective_function_satisfying_conditions_l58_58592


namespace general_term_formula_sum_b_n_up_to_10_l58_58119

open BigOperators

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 2

-- Define the sequence {b_n} based on {a_n}
def b_n (n : ℕ) : ℕ := 2 ^ (a_n n - 2) + n

-- Problem statements
theorem general_term_formula (n : ℕ) : a_n n = n + 2 := by
  sorry

theorem sum_b_n_up_to_10 : (∑ i in finset.range 10, b_n (i + 1)) = 2101 := by
  sorry

end general_term_formula_sum_b_n_up_to_10_l58_58119


namespace angle_between_EF_and_AB_is_90_degrees_l58_58206

-- Define the geometric setup
variables (S A B C E F : Point)
variables (α β : ℝ) (ABC : Triangle)

-- Conditions based on the problem
def is_equilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

def is_regular_tetrahedron (S A B C : Point) : Prop := 
  dist S A = dist S B ∧ dist S B = dist S C ∧ dist S C = dist A B

def midpoint (P Q : Point) (M : Point) : Prop := 
  dist P M = dist M Q ∧ P ≠ Q

-- Define variables for the points E and F
def E_midpoint_SC := midpoint S C E
def F_midpoint_AB := midpoint A B F

-- Proposition: angle between EF and AB is 90 degrees
noncomputable def angle_EF_AB := sorry

-- The main theorem we aim to prove
theorem angle_between_EF_and_AB_is_90_degrees :
  is_equilateral ABC ∧ is_regular_tetrahedron S A B C ∧ E_midpoint_SC ∧ F_midpoint_AB → angle_EF_AB = 90 := 
sorry

end angle_between_EF_and_AB_is_90_degrees_l58_58206


namespace sqrt_144000_simplified_l58_58269

theorem sqrt_144000_simplified : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end sqrt_144000_simplified_l58_58269


namespace range_of_x_pow_k_l58_58134

theorem range_of_x_pow_k (k : ℝ) (h : k > 0) :
  set.range (λ x: ℝ, x^k) = set.Ioi (0 : ℝ) :=
sorry

end range_of_x_pow_k_l58_58134


namespace steel_parts_count_l58_58456

-- Definitions for conditions
variables (a b : ℕ)

-- Conditions provided in the problem
axiom machines_count : a + b = 21
axiom chrome_parts : 2 * a + 4 * b = 66

-- Statement to prove
theorem steel_parts_count : 3 * a + 2 * b = 51 :=
by
  sorry

end steel_parts_count_l58_58456


namespace min_value_in_interval_l58_58476

def f (a x : ℝ) : ℝ := Real.exp (a * x) - Real.log x

theorem min_value_in_interval :
  ∀ x : ℝ, f (1/2) x = Real.exp ((1/2) * x) - Real.log x
  ∧ (∀ x : ℝ, x ∈ Ioo 1 2 → has_minimum_value (f (1/2)) x)
  ∧ ((∃ x₀ ∈ Ioo 1 2, is_min_value (f (1/2)) x₀))
:= sorry

end min_value_in_interval_l58_58476


namespace shaded_area_floor_l58_58752

-- Define the dimensions of the floor
def floor_length : ℝ := 12
def floor_width : ℝ := 15

-- Define the dimensions of each tile
def tile_length : ℝ := 1
def tile_width : ℝ := 1

-- Define the radius of the quarter circle in each tile
def radius : ℝ := 0.5

-- Compute the area of one tile excluding the white quarter circles
def shaded_area_per_tile : ℝ := tile_length * tile_width - (real.pi * radius^2 / 4)

-- Compute the total number of tiles
def total_tiles : ℝ := (floor_length / tile_length) * (floor_width / tile_width)

-- Compute the total shaded area
def total_shaded_area : ℝ := total_tiles * shaded_area_per_tile

-- The theorem to prove
theorem shaded_area_floor : total_shaded_area = 180 - 45 * real.pi := by
  sorry

end shaded_area_floor_l58_58752


namespace range_of_m_l58_58911

open Real

def p (m : ℝ) := ∃ x : ℝ, (cos (2 * x) - sin x + 2) ≤ m

def q (m : ℝ) := ∀ x : ℝ, 2 ≤ x → (2 * x^2 - m * x + 2) ≤ (2 * (x + 1)^2 - m * (x + 1) + 2)

noncomputable def valid_range (m : ℝ) : Prop :=
  (m > 8 ∨ m < 0) ∧ (sorry : p m → m ≥ 0) ∧ (sorry : q m → m ≤ 8)

theorem range_of_m (m : ℝ) : valid_range m :=
sorry

end range_of_m_l58_58911


namespace length_of_each_lateral_edge_l58_58184

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end length_of_each_lateral_edge_l58_58184


namespace range_of_m_iff_necessary_condition_l58_58482

def proposition_p (x m : ℝ) : Prop := (x - m)^2 > 3 * (x - m)
def proposition_q (x : ℝ) : Prop := x^2 + 3 * x - 4 < 0
def necessary_not_sufficient (p q : Prop) : Prop := ∀ (x : ℝ), q x → p x

theorem range_of_m_iff_necessary_condition (m : ℝ) :
  necessary_not_sufficient (proposition_p x m) proposition_q ↔ (m ≤ -7) ∨ (m ≥ 1) :=
by
  sorry

end range_of_m_iff_necessary_condition_l58_58482


namespace round_pi_minus_one_round_sqrt_fifteen_round_x_minus_one_three_integer_solutions_conditions_round_x_eq_fractional_l58_58473

-- Define the rounding function
def round (x : ℝ) : ℕ :=
  if ∃ n : ℕ, n - 0.5 ≤ x ∧ x < n + 0.5 then nat.floor x else 0

-- Part (1a)
theorem round_pi_minus_one : round (Real.pi - 1) = 2 :=
  sorry

-- Part (1b)
theorem round_sqrt_fifteen : round (Real.sqrt 15) = 4 :=
  sorry

-- Part (2)
theorem round_x_minus_one (x : ℝ) (h : round (x - 1) = 4) : 4.5 ≤ x ∧ x < 5.5 :=
  sorry

-- Part (3)
theorem three_integer_solutions_conditions (a : ℝ)
  (h : ∀ (x : ℝ), (2 * x - 7) / 3 ≤ x - 2 ∧ round a - x > 0) : 
  1.5 ≤ a ∧ a < 2.5 :=
  sorry

-- Part (4)
theorem round_x_eq_fractional (x : ℝ) (h : round x = (3 / 5) * x + 1) :
  x = 5 / 3 ∨ x = 10 / 3 :=
  sorry

end round_pi_minus_one_round_sqrt_fifteen_round_x_minus_one_three_integer_solutions_conditions_round_x_eq_fractional_l58_58473


namespace vector_magnitude_difference_l58_58861
-- Lean 4 code statement

variables {𝕜 : Type*} [IsROrC 𝕜]
variables {E : Type*} [InnerProductSpace 𝕜 E] [NormedSpace ℝ E]
open IsROrC

-- Define the given conditions for the vectors a and b
variables (a b : E)
variables (ha : ∥a∥ = 1)
variables (hb : ∥b∥ = 1)
variables (hab : ∥a + b∥ = 1)

-- The goal is to show that ∥a - b∥ = sqrt 3
theorem vector_magnitude_difference :
  ∥a - b∥ = sqrt 3 :=
by sorry

end vector_magnitude_difference_l58_58861


namespace parabola_chords_reciprocal_sum_l58_58903

theorem parabola_chords_reciprocal_sum (x y : ℝ) (AB CD : ℝ) (p : ℝ) :
  (y = (4 : ℝ) * x) ∧ (AB ≠ 0) ∧ (CD ≠ 0) ∧
  (p = (2 : ℝ)) ∧
  (|AB| = (2 * p / (Real.sin (Real.pi / 4))^2)) ∧ 
  (|CD| = (2 * p / (Real.cos (Real.pi / 4))^2)) →
  (1 / |AB| + 1 / |CD| = 1 / 4) :=
by
  sorry

end parabola_chords_reciprocal_sum_l58_58903


namespace max_participants_winning_two_matches_l58_58019

theorem max_participants_winning_two_matches (n : Nat) (h : n = 100) : ∃ k, k = 49 ∧ ∀ m, m ≤ k → participants_won_exactly_two_matches n m :=
by
  sorry

end max_participants_winning_two_matches_l58_58019


namespace quadratic_function_expression_range_of_a_for_positive_maximum_value_l58_58112

variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (h₁ : (1,3) = {x | f(x) > -2 * x})
variable (h₂ : f x + 6 * a = 0 ∧ has_double_root f)

theorem quadratic_function_expression (h₁ : (1,3) = {x | f(x) > -2 * x}) (h₂ : f x + 6 * a = 0 ∧ has_double_root f) :
  f = λ x, - (1 / 5 : ℝ) * x^2 - (6 / 5 : ℝ) * x - (3 / 5 : ℝ) := by
  sorry

theorem range_of_a_for_positive_maximum_value (h₁: f = λ x, a * x^2 - 2 * (1 + 2 * a) * x + 3 * a) (h₃ : ∃ b, f b > 0) :
  a ∈ ((-∞ : ℝ), -2 - sqrt 3) ∪ (-2 + sqrt 3, 0) := by
  sorry

end quadratic_function_expression_range_of_a_for_positive_maximum_value_l58_58112


namespace ribbons_given_away_in_morning_l58_58254

theorem ribbons_given_away_in_morning (M : ℕ) :
  ∃ M, 
    let total_ribbons := 38 in
    let remaining_ribbons := 8 in
    let ribbons_given_in_afternoon := 16 in
    let total_given_away := total_ribbons - remaining_ribbons in
    M = total_given_away - ribbons_given_in_afternoon :=
begin
  use 14,
  sorry
end

end ribbons_given_away_in_morning_l58_58254


namespace sum_of_tan_roots_l58_58094

noncomputable def r1 : ℝ := (7 + Real.sqrt 41) / 2
noncomputable def r2 : ℝ := (7 - Real.sqrt 41) / 2

theorem sum_of_tan_roots :
  ∑ x in ({Real.arctan r1, Real.arctan r1 + Real.pi, Real.arctan r2, Real.arctan r2 + Real.pi} : finset ℝ), x = 4 * Real.pi :=
sorry

end sum_of_tan_roots_l58_58094


namespace valid_word_combinations_l58_58217

-- Definition of valid_combination based on given conditions
def valid_combination : ℕ :=
  26 * 5 * 26

-- Statement to prove the number of valid four-letter combinations is 3380
theorem valid_word_combinations : valid_combination = 3380 := by
  sorry

end valid_word_combinations_l58_58217


namespace shortest_path_on_cube_surface_l58_58055

theorem shortest_path_on_cube_surface (a : ℝ) (ha : a = 2) :
  let s := a / 2 in
  let square_diagonal := s * Real.sqrt 2 in
  square_diagonal = Real.sqrt 2 :=
by
  -- Unfolding the cube and considering only the surface
  -- Starting and ending at midpoints of edges 
  sorry

end shortest_path_on_cube_surface_l58_58055


namespace factorial_square_ge_power_l58_58925

theorem factorial_square_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := 
by sorry

end factorial_square_ge_power_l58_58925


namespace tic_tac_toe_ways_l58_58943

def isWinningRow (board : List (List Char)) (symbol : Char) : Bool :=
  board.any (fun row => row.count (fun c => c = symbol) = 3)

def isWinningColumn (board : List (List Char)) (symbol : Char) : Bool :=
  List.transpose board |>
  List.any (fun column => column.count (fun c => c = symbol) = 3)

def isWinningDiagonal (board : List (List Char)) (symbol : Char) : Bool :=
  let diagonal1 := List.range 4 |>.map (fun i => board.get! i |>.get! i)
  let diagonal2 := List.range 4 |>.map (fun i => board.get! i |>.get! (3 - i))
  diagonal1.count (fun c => c = symbol) = 3 || diagonal2.count (fun c => c = symbol) = 3

def isWinningBoard (board : List (List Char)) (symbol : Char) : Bool :=
  isWinningRow board symbol || isWinningColumn board symbol || isWinningDiagonal board symbol

theorem tic_tac_toe_ways (board : List (List Char)) :
  ∃ board : List (List Char), isWinningBoard board 'O' ∧ 
  (∃ other_board : List (List Char), 
    board.countp (fun c => c = 'X') = 3 ∧ 
    board.countp (fun c => c = 'O') = 3) ∧
  (find_all_ways_to_win board 'O' = 114400) :=
sorry

end tic_tac_toe_ways_l58_58943


namespace sugar_needed_for_partial_recipe_l58_58756

theorem sugar_needed_for_partial_recipe :
  let initial_sugar := 5 + 3/4
  let part := 3/4
  let needed_sugar := 4 + 5/16
  initial_sugar * part = needed_sugar := 
by 
  sorry

end sugar_needed_for_partial_recipe_l58_58756


namespace mice_needed_l58_58011

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l58_58011


namespace arithmetic_sequence_common_difference_l58_58231

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l58_58231


namespace speed_of_water_current_l58_58046

theorem speed_of_water_current (v : ℝ) 
  (swimmer_speed_still_water : ℝ := 4) 
  (distance : ℝ := 3) 
  (time : ℝ := 1.5)
  (effective_speed_against_current : ℝ := swimmer_speed_still_water - v) :
  effective_speed_against_current = distance / time → v = 2 := 
by
  -- Proof
  sorry

end speed_of_water_current_l58_58046


namespace ferris_break_length_l58_58370

-- Definitions of the given conditions
def audrey_work_rate := 1 / 4  -- Audrey completes 1/4 of the job per hour
def ferris_work_rate := 1 / 3  -- Ferris completes 1/3 of the job per hour
def total_work_time := 2       -- They worked together for 2 hours
def num_breaks := 6            -- Ferris took 6 breaks during the work period

-- The theorem to prove the length of each break Ferris took
theorem ferris_break_length (break_length : ℝ) :
  (audrey_work_rate * total_work_time) + 
  (ferris_work_rate * (total_work_time - (break_length / 60) * num_breaks)) = 1 →
  break_length = 2.5 :=
by
  sorry

end ferris_break_length_l58_58370


namespace triangle_third_side_range_l58_58881

-- Definitions of side lengths based on given conditions
def a : ℕ := 4
def b : ℕ := 7

-- The theorem that needs to be proven
theorem triangle_third_side_range (c : ℕ) (a b : ℕ) (h₁ : a = 4) (h₂ : b = 7) :
  (3 < c) ∧ (c < 11) :=
by
  have h₃ : a + b > c := by rw [h₁, h₂]; exact 11 > c
  have h₄ : b - a < c := by rw [h₁, h₂]; exact 3 < c
  sorry

end triangle_third_side_range_l58_58881


namespace scientific_notation_of_100000000_l58_58313

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l58_58313


namespace a_2015_eq_2_l58_58528

theorem a_2015_eq_2 :
  (∃ a : ℕ → ℚ, a 1 = 1 / 2 ∧ (∀ n : ℕ, a (n + 1) = 1 / (1 - a n)) ∧ a 2015 = 2) :=
by
  let a : ℕ → ℚ := λ n, (if n % 3 = 1 then 1 / 2 else if n % 3 = 2 then 2 else -1)
  existsi a
  simp
  split
  -- condition for a 1
  { refl }
  split
  -- condition for recurrence relation
  { intros n
    cases nat.mod_eq_zero_or_pos n 3 with h h
    { rw [h, nat.succ, nat.succ_eq_add_one, nat.mod_self],
      exact show a (n + 1) = -1 from rfl }
    { cases n % 3 with
        h₀ h₀; cases h₀
      { rw [nat.mod_eq_of_lt h₀, nat.succ, (nat.add_mod _ 3)],
        change nat.succ (nat.mod n 3) = 1,
        rw [nat.succ_eq_add_one, add_comm],
        rw_mod_eq }
      { rw [nat.mod_eq_of_lt h₀, nat.succ, (nat.add_mod _ 3)],
        change nat.succ (nat.mod n 3) = 2,
        rw [nat.succ_eq_add_one, add_comm (1 : ℕ) 1, one_add_one_eq_two],
        rw_mod_eq }
      { rw [nat.succ_eq_add_one, add_comm],
        rw_mod_eq } } }
  -- condition for a 2015
  { calc a 2015 = a (3 * 671 + 2) : by rw [nat.div_add_mod _ 3]
                ... = 2            : by refl }

end a_2015_eq_2_l58_58528


namespace min_abs_difference_on_hyperbola_l58_58552

theorem min_abs_difference_on_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 8 - y^2 / 4 = 1) → abs (x - y) ≥ 2 := 
by
  intros x y hxy
  sorry

end min_abs_difference_on_hyperbola_l58_58552


namespace extremum_and_equal_values_l58_58605

theorem extremum_and_equal_values {f : ℝ → ℝ} {a b x_0 x_1 : ℝ} 
    (hf : ∀ x, f x = (x - 1)^3 - a * x + b)
    (h'x0 : deriv f x_0 = 0)
    (hfx1_eq_fx0 : f x_1 = f x_0)
    (hx1_ne_x0 : x_1 ≠ x_0) :
  x_1 + 2 * x_0 = 3 := sorry

end extremum_and_equal_values_l58_58605


namespace g_of_3_l58_58557

variable (g : ℝ → ℝ)
variable (h_cond : ∀ x : ℝ, g(x^2 + 2) = 2*x^2 + 3)

theorem g_of_3 : g 3 = 5 :=
by
  sorry

end g_of_3_l58_58557


namespace area_enclosed_by_inscribed_angle_l58_58854

theorem area_enclosed_by_inscribed_angle (R α : ℝ) : 
  ∃ (S : ℝ), S = R^2 * (α + Real.sin α) :=
by
  use R^2 * (α + Real.sin α)
  sorry

end area_enclosed_by_inscribed_angle_l58_58854


namespace find_projection_matrix_elements_l58_58902

noncomputable def projection_matrix (b d : ℚ) : matrix (fin 2) (fin 2) ℚ := 
  ![![b, 12/25], 
    ![d, 13/25]]

theorem find_projection_matrix_elements :
  ∃ (b d : ℚ), projection_matrix b d * projection_matrix b d = projection_matrix b d ∧
               (b = 37/50 ∧ d = 19/50) := by
  sorry

end find_projection_matrix_elements_l58_58902


namespace scientific_notation_of_100000000_l58_58314

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end scientific_notation_of_100000000_l58_58314


namespace popular_best_friend_popular_popular_not_best_friend_inf_friends_l58_58322

-- Definitions for the social network Mugbook
structure Person := (id : ℕ)
structure Friendship := (A B : Person)
def friends (A B : Person) (R : list Friendship) : Prop := 
  (A ≠ B) ∧ (Friendship.mk A B ∈ R ∨ Friendship.mk B A ∈ R)
def best_friend (A B : Person) (B_list : list (Person × Person)) : Prop := 
  (A, B) ∈ B_list
def n_best_friend (n : ℕ) (P : Person) (B_list : list (Person × Person)) : Prop :=
  ∃ k, k = n ∧ (∀ i (H : i ≤ k), best_friend P (B_list.nth_le i (Nat.lt_of_succ_lt_succ H)).snd B_list)
def popular (P : Person) (B_list : list (Person × Person)) : Prop :=
  ∀ k, n_best_friend k P B_list

-- Problem (a): Prove that every popular person is the best friend of a popular person.
theorem popular_best_friend_popular (P : Person) (B_list : list (Person × Person)) (R : list Friendship) :
  (∀ Q, friends P Q R → best_friend Q P B_list) → popular P B_list → 
  ∃ Q, friends P Q R ∧ popular Q B_list := 
sorry

-- Problem (b): If people can have infinitely many friends, it is possible that a popular person is not the best friend of a popular person.
theorem popular_not_best_friend_inf_friends (P : Person) (B_list : list (Person × Person)) :
  (∃ R, ∀ x, ∃ y, friends x y R) → popular P B_list →
  ∃ P', friends P P' (R : list Friendship) ∧ ¬ popular P' B_list :=
sorry

end popular_best_friend_popular_popular_not_best_friend_inf_friends_l58_58322


namespace decreasing_function_l58_58900

theorem decreasing_function (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (m + 3) * x1 - 2 > (m + 3) * x2 - 2) ↔ m < -3 :=
by
  sorry

end decreasing_function_l58_58900


namespace total_amount_received_l58_58737
noncomputable section

variables (B : ℕ) (H1 : (1 / 3 : ℝ) * B = 50)
theorem total_amount_received (H2 : (2 / 3 : ℝ) * B = 100) (H3 : ∀ (x : ℕ), x = 5): 
  100 * 5 = 500 := 
by
  sorry

end total_amount_received_l58_58737


namespace sum_d_1_to_729_l58_58475

def d (n : ℕ) : ℕ :=
  let j := nat.floor (real.cbrt (n : ℝ))
  in if abs (j - real.cbrt (n : ℝ)) < 1 / 2 then j else j + 1

theorem sum_d_1_to_729 : (∑ n in finset.range 730, d n) = 4293 := 
by
  sorry

end sum_d_1_to_729_l58_58475


namespace wheel_revolutions_l58_58680

theorem wheel_revolutions (d : ℝ) (mile_in_feet : ℝ) (r : ℝ) (circumference : ℝ) (distance : ℝ) :
  d = 4 ∧ mile_in_feet = 5280 ∧ r = 2 ∧ circumference = 4 * Real.pi ∧ distance = 5280 →
  distance / circumference = 1320 / Real.pi :=
by
  intros h
  cases h
  norm_num at *
  rw [h_left, h_right_left, div_eq_mul_inv, mul_comm]
  exact eq_iff_4



end wheel_revolutions_l58_58680


namespace max_value_of_y_l58_58601

noncomputable def max_y_of_circle : ℝ 
:= 20 + Real.sqrt 481

theorem max_value_of_y (x y : ℝ) (h : x^2 + y^2 = 18 * x + 40 * y) :
  y ≤ max_y_of_circle :=
begin
  sorry
end

end max_value_of_y_l58_58601


namespace hexagon_reflect_area_l58_58408

theorem hexagon_reflect_area (P : Point) (hex : RegularHexagon 1) : 
  let P1 := reflect_point_over_midpoint P hex 1,
      P2 := reflect_point_over_midpoint P hex 2,
      P3 := reflect_point_over_midpoint P hex 3,
      P4 := reflect_point_over_midpoint P hex 4,
      P5 := reflect_point_over_midpoint P hex 5,
      P6 := reflect_point_over_midpoint P hex 6
  in area_of_hexagon P1 P2 P3 P4 P5 P6 = (9 * real.sqrt 3) / 2 := sorry

end hexagon_reflect_area_l58_58408


namespace dealer_profit_percentage_l58_58392

theorem dealer_profit_percentage :
  (let CP_per_article := 25 / 15;
       SP_per_article := 33 / 12;
       Profit_per_article := SP_per_article - CP_per_article;
       Profit_Percentage := (Profit_per_article / CP_per_article) * 100 in
    Profit_Percentage = 65) :=
by
  let CP_per_article := 25 / 15
  let SP_per_article := 33 / 12
  let Profit_per_article := SP_per_article - CP_per_article
  let Profit_Percentage := (Profit_per_article / CP_per_article) * 100
  exact sorry

end dealer_profit_percentage_l58_58392


namespace fencing_cost_correct_l58_58323

noncomputable def total_cost_fencing
  (area1 area2 area3 : ℝ) 
  (cost1 cost2 cost3 : ℝ) : ℝ :=
let r1 := real.sqrt (area1 * 10000 / real.pi),
    r2 := real.sqrt (area2 * 10000 / real.pi),
    r3 := real.sqrt (area3 * 10000 / real.pi),
    c1 := 2 * real.pi * r1,
    c2 := 2 * real.pi * r2,
    c3 := 2 * real.pi * r3 in
c1 * cost1 + c2 * cost2 + c3 * cost3

theorem fencing_cost_correct :
  total_cost_fencing 13.86 21.54 9.42 4.70 5.50 6.30 ≈ 22099.90 :=
begin
  -- proof omitted
  sorry
end

end fencing_cost_correct_l58_58323


namespace find_C_l58_58050

variables (A B C : ℝ)

-- Conditions
def cond1 := A + B + C = 450
def cond2 := A + C = 200
def cond3 := B + C = 350

-- The goal to prove
theorem find_C (h1 : cond1) (h2 : cond2) (h3 : cond3) : C = 100 :=
sorry

end find_C_l58_58050


namespace unique_solution_system_eqns_l58_58276

theorem unique_solution_system_eqns (a b c : ℕ) :
  (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (b + c)) ↔ (a = 2 ∧ b = 1 ∧ c = 1) := by 
  sorry

end unique_solution_system_eqns_l58_58276


namespace maximum_ab_l58_58525

-- Define the conditions
def line (a b : ℝ) (P : ℝ × ℝ) : Prop := a * P.1 + b * P.2 = 8
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 1) ^ 2 + (Q.2 - 2) ^ 2 = 5
def chord_length (L : ℝ) : Prop := L = 2 * real.sqrt 5

-- Main statement to prove
theorem maximum_ab (a b : ℝ) (L : ℝ) (H1: a > 0) (H2: b > 0) 
  (H3: (∃ (P : ℝ × ℝ), line a b P ∧ circle P)) 
  (H4: chord_length L) : 
  ab ≤ 8 := sorry

end maximum_ab_l58_58525


namespace modulus_of_purely_imaginary_l58_58137

theorem modulus_of_purely_imaginary (a : ℝ) (h : (a + 6) = 0) : 
  let z := (a + 3 * complex.I) / (1 + 2 * complex.I)
  in complex.abs z = 3 :=
by 
  sorry

end modulus_of_purely_imaginary_l58_58137


namespace find_number_l58_58720

theorem find_number (x : ℝ) :
  9 * (((x + 1.4) / 3) - 0.7) = 5.4 ↔ x = 2.5 :=
by sorry

end find_number_l58_58720


namespace probability_snow_first_week_l58_58633

theorem probability_snow_first_week :
  let p1 := 1/4
  let p2 := 1/3
  let no_snow := (3/4)^4 * (2/3)^3
  let snows_at_least_once := 1 - no_snow
  snows_at_least_once = 29 / 32 := by
  sorry

end probability_snow_first_week_l58_58633


namespace find_theta_l58_58977

theorem find_theta (P : ℂ) (r θ : ℝ) (hP : P = r * (complex.cos θ + complex.sin θ * complex.I))
  (hr : 0 < r) (hθ : 0 ≤ θ ∧ θ < 360) :
  (∃ z : ℂ, (z^6 + z^5 + z^4 + z^3 + z^2 + z + 1 = 0) → (im z > 0)) → θ = 308.57 :=
sorry

end find_theta_l58_58977


namespace radius_of_circumscribed_circle_l58_58027

open Real

/-- Given an isosceles triangle ABC with AB = BC, a circle passing through A and C 
intersects sides AB and BC at points M and N respectively. Chord MK of this circle 
has length 2√5 and contains point H on AC, where H is the foot of the altitude from 
B to AC. A line through C and perpendicular to BC intersects line MN at L. 
Given that cos ∠ABK = 2/3, prove that the radius of the circumscribed circle around 
triangle MKL is 3. -/
theorem radius_of_circumscribed_circle (A B C M N K L H : Point)
  (h_iso : AB = BC) 
  (h_circle : ∃ (circle : Circle), circle.contains A ∧ circle.contains C ∧ circle.contains M ∧ circle.contains N)
  (h_MK_length : dist M K = 2 * √5)
  (h_H_on_AC : is_foot H B A C)
  (h_CL_perpendicular : is_perpendicular (line_through C L) (line_through B C))
  (h_cos_ABK : cos (angle A B K) = 2 / 3) :
  ∃ (circle_circumscribed : Circle), 
    circle_circumscribed.contains M ∧ 
    circle_circumscribed.contains K ∧ 
    circle_circumscribed.contains L ∧ 
    circle_circumscribed.radius = 3 :=
sorry

end radius_of_circumscribed_circle_l58_58027


namespace triangle_solution_l58_58496

theorem triangle_solution (a b c : ℝ) (A B C : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_A : A = 45) (h_B : B = 30) :
    b = 2 ∧ c = Real.sqrt 6 + Real.sqrt 2 :=
by
  let h_C : C = 180 - A - B
  let h_sinC : Real.sin C = Real.sin (A + B)
  let b_def : b = a * Real.sin B / Real.sin A
  let c_def : c = a * Real.sin C / Real.sin A
  have h_b : b = 2
  have h_c : c = Real.sqrt 6 + Real.sqrt 2
  sorry

end triangle_solution_l58_58496


namespace triangle_area_sum_property_l58_58213

theorem triangle_area_sum_property {A B C D E : Type*} {BC : ℝ} {BAC ABC ACB : ℝ} 
  (hBC : BC = 1)
  (hBAC : BAC = 40)
  (hABC : ABC = 90)
  (hACB : ACB = 50)
  (h_mid_pt_D : D = midpoint A B)
  (h_mid_pt_E : E = midpoint A C)
  (hCDE : ∠CDE = 50) :
  area_of_triangle A B C + 2 * area_of_triangle C D E = 5 / 16 :=
sorry

end triangle_area_sum_property_l58_58213


namespace scale_balance_shift_l58_58637

-- Define the main problem statement
theorem scale_balance_shift (n : ℕ) (w : finset (finset ℕ)) (weights : finset ℕ → ℤ)
  (condition1 : ∀ s ∈ w, ∀ i ∈ s, i < n)
  (condition2 : ∀ s ∈ w, weights s ≠ 0) :
  ∃ (x : fin n → ℤ), ∑ s in w, (weights s * ∏ i in s, x i) < 0 :=
begin
  -- proof would go here
  sorry
end

end scale_balance_shift_l58_58637


namespace slant_asymptote_sum_coefficients_l58_58450

noncomputable def rational_function (x : ℝ) : ℝ :=
  (3 * x^3 - x^2 + 4 * x - 8) / (x - 2)

theorem slant_asymptote_sum_coefficients :
  let asymptote := 3 * x + 5 in  -- Linear slant asymptote from the problem
  3 + 5 = 8 :=                   -- Sum of coefficients
sorry

end slant_asymptote_sum_coefficients_l58_58450


namespace comic_books_ratio_l58_58804

variable (S : ℕ)

def initial_comics := 22
def remaining_comics := 17
def comics_bought := 6

theorem comic_books_ratio (h1 : initial_comics - S + comics_bought = remaining_comics) :
  (S : ℚ) / initial_comics = 1 / 2 := by
  sorry

end comic_books_ratio_l58_58804


namespace polygons_divided_by_diagonals_of_convex_ngon_polygons_divided_by_diagonals_of_even_ngon_l58_58014

theorem polygons_divided_by_diagonals_of_convex_ngon {n : ℕ} (h : 3 ≤ n) :
  (∀ (poly : Set (Set ℝ)), polygon_is_subdivision_of_convex_ngon poly n → (|poly| ≤ n)) := 
sorry

theorem polygons_divided_by_diagonals_of_even_ngon {n : ℕ} (h : 3 ≤ n) (h_even : even n) :
  (∀ (poly : Set (Set ℝ)), polygon_is_subdivision_of_convex_ngon poly n → (|poly| ≤ n - 1)) := 
sorry

end polygons_divided_by_diagonals_of_convex_ngon_polygons_divided_by_diagonals_of_even_ngon_l58_58014


namespace probability_three_greens_correct_l58_58639

noncomputable def probability_three_greens : ℝ :=
  let p_green := 7 / 10
  let p_purple := 3 / 10
  let specific_order_probability := (p_green^3) * (p_purple^3)
  let combinations := 20
  (specific_order_probability * combinations).round 3

theorem probability_three_greens_correct :
  probability_three_greens = 0.185 :=
by sorry

end probability_three_greens_correct_l58_58639


namespace sequence_formula_l58_58853

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (diff : ∀ n, a (n + 1) - a n = 3^n) :
  ∀ n, a n = (3^n - 1) / 2 :=
by
  sorry

end sequence_formula_l58_58853


namespace determine_b_l58_58542

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

theorem determine_b (i : imaginary_unit) (b : ℝ) : 
  (2 - i.val) * 4 * i.val = 4 - b * i.val → b = -8 :=
by
  sorry

end determine_b_l58_58542


namespace reward_model_meets_requirements_l58_58391

noncomputable def reward_function (x : ℝ) : ℝ :=
  Real.log x / Real.log 10 + 1

theorem reward_model_meets_requirements :
  ∀ x ∈ Icc (10 : ℝ) (1000 : ℝ), 
    (Function.monotoneOn reward_function (Icc 10 1000)) ∧
    (∀ x, x ∈ Icc 10 1000 → reward_function x ≤ 5) ∧
    (∀ x, x ∈ Icc 10 1000 → reward_function x ≤ 0.25 * x) := 
by
  sorry

end reward_model_meets_requirements_l58_58391


namespace Sn_value_l58_58144

noncomputable def Sn (n : ℕ) : ℕ := sorry -- Placeholder for the function giving the sum of the first n terms

def arithmetic_seq_sum (S3 S6 S9 : ℕ) : Prop :=
  S3 = 3 ∧ S6 = 15 ∧ 2 * (S6 - S3) = S3 + S9 - S6

theorem Sn_value (S3 S6 S9 : ℕ) (h : arithmetic_seq_sum S3 S6 S9) : S9 = 36 := by
  cases h with h1 h2 h3
  sorry

end Sn_value_l58_58144


namespace double_sum_evaluation_l58_58079

theorem double_sum_evaluation :
  (∑ m in (Finset.range (1000)), ∑ n in (Finset.range (1000)).filter (λ k, k ≥ m), 1 / (m^2 * n * (n + m + 2))) = 1 := 
begin
  sorry
end

end double_sum_evaluation_l58_58079


namespace volume_Q3_l58_58115

theorem volume_Q3 : 
  let Q0 := 4
  let Q1 := Q0 + 4 * (1/3) * (1/4) * (1/2) * Q0
  let next_volume := (v : ℕ → ℚ) (i : ℕ) => v i + 6^i * 4 * ((1/2)^(3*i)) * (1/6)
  let volumes := Nat.iterate next_volume Q1
  vol.Q3 := volumes 3 in
  vol.Q3 = 39 / 8 :=
by
  sorry

end volume_Q3_l58_58115


namespace range_of_a_l58_58128

open Set

variable (A B : Set ℝ)
variable (a : ℝ)

def A_def : Set ℝ := { x | x ≥ 0 }
def B_def : Set ℝ := { x | x ≤ a }

theorem range_of_a
  (hA : A = A_def)
  (hB : B = B_def)
  (hUnion : A ∪ B = Set.univ) :
  a ≥ 0 :=
sorry

end range_of_a_l58_58128


namespace water_for_bathing_per_horse_per_day_l58_58962

-- Definitions of the given conditions
def initial_horses : ℕ := 3
def additional_horses : ℕ := 5
def total_horses : ℕ := initial_horses + additional_horses
def drink_water_per_horse_per_day : ℕ := 5
def total_days : ℕ := 28
def total_water_needed : ℕ := 1568

-- The proven statement
theorem water_for_bathing_per_horse_per_day :
  ((total_water_needed - (total_horses * drink_water_per_horse_per_day * total_days)) / (total_horses * total_days)) = 2 :=
by
  sorry

end water_for_bathing_per_horse_per_day_l58_58962


namespace P_Q_homothetic_l58_58681

-- Definitions based on the given conditions
variables {M : Type*} [bicentric M] -- M is a bicentric polygon

-- Define the polygons P and Q
def P (M : Type*) [bicentric M] : Type* :=
  {p : M // point_of_contact_with_incircle p}

def Q (M : Type*) [bicentric M] : Type* :=
  {q : M // external_angle_bisector q}

-- The main theorem to be proven
theorem P_Q_homothetic (M : Type*) [bicentric M] :
  P M ≃ₕ Q M :=
sorry

end P_Q_homothetic_l58_58681


namespace line_intersects_x_axis_at_point_l58_58437

-- Lean 4 statement
theorem line_intersects_x_axis_at_point :
  ∃ x, ∃ y, (x, y) = (-1, 0) ∧ 
            (∃ m b : ℝ, m = (10 - 6) / (4 - 2) ∧
                        6 = m * 2 + b ∧
                        y = m * x + b ∧
                        y = 0 ∧
                        (x, y) ∈ (λ p : ℝ × ℝ, ∃ a b : ℝ, b = 1 ∧ a ≠ b))
:= by
  sorry

end line_intersects_x_axis_at_point_l58_58437


namespace smallest_divisor_sum_of_squares_of_1_to_7_l58_58802

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_divisor_sum_of_squares_of_1_to_7 (S : ℕ) (h : S = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) :
  ∃ m, is_divisor m S ∧ (∀ d, is_divisor d S → 2 ≤ d) :=
sorry

end smallest_divisor_sum_of_squares_of_1_to_7_l58_58802


namespace shortest_distance_is_one_l58_58723

-- Define the problem conditions
def circle_eq (x y : ℝ) := x^2 - 6*x + y^2 - 8*y + 9 = 0

-- Define the function to calculate distance from the origin
def distance (x y : ℝ) := Real.sqrt (x^2 + y^2)

noncomputable def shortest_distance_to_circle : ℝ :=
  let center_x := 3
  let center_y := 4
  let radius := 4
  let origin_to_center := distance center_x center_y
  origin_to_center - radius

-- The statement of the proof
theorem shortest_distance_is_one : shortest_distance_to_circle = 1 :=
by 
  sorry

end shortest_distance_is_one_l58_58723


namespace positive_difference_l58_58353

theorem positive_difference 
  (a b : ℝ)
  (h1 : a = (8^2 + 8^2) / 8)
  (h2 : b = (8^2 - 8^2) / 8) :
  abs(a - b) = 16 :=
by {
  sorry
}

end positive_difference_l58_58353


namespace tangent_line_slope_l58_58318

def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

def point : ℝ × ℝ := (1, -(5/3))

theorem tangent_line_slope  : deriv f point.fst = 1 := by
  sorry

end tangent_line_slope_l58_58318


namespace unique_sequence_length_l58_58447

theorem unique_sequence_length :
  ∃ k : ℕ, (∃ (b : fin k → ℕ), strict_mono b ∧ (∑ i in finrange k, 2^b i) = (2^217 + 1) / (2^13 + 1)) ∧ k = 209 :=
sorry

end unique_sequence_length_l58_58447


namespace arithmetic_array_center_value_l58_58800

theorem arithmetic_array_center_value :
  ∃ X : ℝ,
    ∀ (a₁ a₁₄ a₄ a₄₄ : ℝ)
      (d₁ d₄ : ℝ)
      (ar seq₁ seq₄ : ℕ → ℝ),
      a₁ = 4 →
      a₁₄ = 16 →
      a₄₀ = 10 →
      a₄₄ = 28 →
      d₁ = (a₁₄ - a₁) / 3 →
      d₄ = (a₄₄ - a₄) / 3 →
      seq₁ 0 = a₁ →
      seq₄ 0 = a₄ →
      (∀ n : ℕ, seq₁ (n+1) = seq₁ n + d₁) →
      (∀ n : ℕ, seq₄ (n+1) = seq₄ n + d₄) →
      X = ((seq₄ 1 - seq₁ 1) / 3 + seq₁ 1) :=
begin
  -- Sorry to skip the proof
  sorry,
end

end arithmetic_array_center_value_l58_58800


namespace binomial_sum_mod_p_squared_l58_58372

open Nat

theorem binomial_sum_mod_p_squared (p : ℕ) (hp_prime: Prime p) (hp_gt_two: 2 < p):
  ∑ n in Finset.range (p + 1), (binomial p n) * (binomial (p + n) p) ≡ 2 * p + 1 [MOD p^2] :=
by 
  sorry

end binomial_sum_mod_p_squared_l58_58372


namespace AB_l58_58497

variables {A B C B' C' O O' B'' C'': Type}
variables [EuclideanGeometry A B C B' C' O O' B'' C'']

-- Given a triangle ABC
axiom triangle_ABC : Triangle A B C
-- Points B' and C' on sides AB and AC such that BB' = CC'
axiom B'_C'_properties : ∃ (B' C': Point) (h1: B' ∈ Segment A B) (h2: C' ∈ Segment A C), (Distance (Segment B B') (Segment C C')) 
-- O and O' are the circumcenters of triangles ABC and AB'C'
axiom circumcenters_O_O' : Circumcenter O (Triangle A B C) ∧ Circumcenter O' (Triangle A B' C')
-- OO' intersects lines AB' and AC' at B'' and C'' respectively
axiom intersection_OO' : ∃ (B'' C'': Point), (Line O O').Intersection (Line A B') = B'' ∧ (Line O O').Intersection (Line A C') = C''
-- Given AB = 1/2 AC
axiom given_AB_half_AC : Distance (Segment A B) = 0.5 * (Distance (Segment A C))

-- The proof statement that AB'' = AC''
theorem AB''_equals_AC'' :
  Distance (Segment A B'') = Distance (Segment A C'') :=
sorry

end AB_l58_58497


namespace find_utilities_second_l58_58401

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

end find_utilities_second_l58_58401


namespace greatest_possible_perimeter_l58_58574

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), ( 5 * x > 12 ∧ x < 4 ∧ x > 0 ) ∧ (4 * x + x + 12 = 27) := by
sory

end greatest_possible_perimeter_l58_58574


namespace area_ratio_of_R_to_ABC_l58_58336

variable (A B C: Point) (R: Region)

-- Establish the conditions based on the problem statement
def equilateral_triangle (A B C: Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def particle_at_time (A B C: Point) (t: ℝ) : Point :=
  if t ≤ 1 then (t, 0)
  else if t ≤ 2 then (1, t - 1)
  else (2 - t, sqrt 3 * (1 - t + 2))

def midpoint_path (A B C: Point) (t: ℝ) : Point :=
  let p1 := particle_at_time A B C (t / 2)
  let p2 := particle_at_time A B C t
  ( (p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
  
theorem area_ratio_of_R_to_ABC
  (hABC: equilateral_triangle A B C)
  (h_initial_A : particle_at_time A B C 0 = A)
  (h_initial_B : particle_at_time A B C 1 = B)
  (R: Region)
  : area R / area (triangle A B C) = 9 / 64
  := sorry

end area_ratio_of_R_to_ABC_l58_58336


namespace inradius_of_quadrilateral_l58_58812

noncomputable def inradius_max (AB BC CD DA : ℕ) (h_AB: AB = 10) (h_BC: BC = 12) (h_CD: CD = 8) (h_DA: DA = 14) : ℝ :=
  4 * Real.sqrt 3

theorem inradius_of_quadrilateral :
  ∀ (AB BC CD DA : ℕ),
    AB = 10 ∧ BC = 12 ∧ CD = 8 ∧ DA = 14 →
    ∃ r : ℝ, r = inradius_max AB BC CD DA ∧ r = 4 * Real.sqrt 3 :=
by intros AB BC CD DA h
   use inradius_max AB BC CD DA
   sorry

end inradius_of_quadrilateral_l58_58812


namespace prob_C_gets_position_is_correct_prob_B_or_E_gets_position_is_correct_l58_58949

-- Definitions based on the conditions
def job_seekers : List String := ["A", "B", "C", "D", "E"]
def positions_available : ℕ := 2
def total_applicants : ℕ := job_seekers.length
def equal_chance_of_hiring := ∀ s ∈ job_seekers, (1/total_applicants : ℚ)

-- Combinatorial calculations
noncomputable def num_combinations : ℕ := Nat.choose total_applicants positions_available
noncomputable def event_C : ℕ := (List.filter (λ (x : List String), "C" ∈ x) (List.powersetLen positions_available job_seekers)).length
noncomputable def event_B_or_E : ℕ := (List.filter (λ (x : List String), "B" ∈ x ∨ "E" ∈ x) (List.powersetLen positions_available job_seekers)).length

-- Probabilities
noncomputable def prob_C_gets_position : ℚ := event_C / num_combinations
noncomputable def prob_B_or_E_gets_position : ℚ := event_B_or_E / num_combinations

-- Theorem statements
theorem prob_C_gets_position_is_correct : prob_C_gets_position = 2/5 := by
  sorry

theorem prob_B_or_E_gets_position_is_correct : prob_B_or_E_gets_position = 7/10 := by
  sorry

end prob_C_gets_position_is_correct_prob_B_or_E_gets_position_is_correct_l58_58949


namespace hypotenuse_of_isosceles_right_triangle_l58_58199

theorem hypotenuse_of_isosceles_right_triangle (BC : ℝ) (hBC : BC = 10)
  (angleBAC : Real.Angle) (angleBCA : Real.Angle) (angleABC : Real.Angle)
  (triangle_isosceles : angleBAC = Real.arctan 1 ∧ angleBCA = Real.arctan 1 ∧ angleABC = Real.pi / 2) :
  ∃ AB : ℝ, AB = 10 * Real.sqrt 2 :=
by
  use 10 * Real.sqrt 2
  sorry

end hypotenuse_of_isosceles_right_triangle_l58_58199


namespace finite_operations_and_average_L_l58_58294

-- Define the problem setup
def Coin := Bool -- A coin can be in state "H" (true) or "T" (false)
def State := List Coin -- A state is a list of coins

/-- 
 (1) Prove that for any initial state, Harry always stops after a finite number of operations. 
 (2) Find the average value of L(C) when C takes all 2^n possible initial states.
-/
theorem finite_operations_and_average_L (n : ℕ) (initial_state : State) :
  ∃ (finite_steps : ℕ), (finite_steps < 2^n) ∧ 
  ((finset.univ : Finset (List Bool)).sum (λ C, L C) / (2^n) = average_L C) := by
  sorry

end finite_operations_and_average_L_l58_58294


namespace son_completion_time_l58_58032

theorem son_completion_time (M S F : ℝ) 
  (h1 : M = 1 / 10) 
  (h2 : M + S = 1 / 5) 
  (h3 : S + F = 1 / 4) : 
  1 / S = 10 := 
  sorry

end son_completion_time_l58_58032


namespace arithmetic_and_geometric_sequence_l58_58562

-- Definitions based on given conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Main statement to prove
theorem arithmetic_and_geometric_sequence :
  ∀ (x y a b c : ℝ), 
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1 / 4 :=
by
  sorry

end arithmetic_and_geometric_sequence_l58_58562


namespace functional_eq_is_linear_l58_58462

theorem functional_eq_is_linear (f : ℚ → ℚ)
  (h : ∀ x y : ℝ, f ((x + y) / 2) = (f x / 2) + (f y / 2)) : ∃ k : ℚ, ∀ x : ℚ, f x = k * x :=
by
  sorry

end functional_eq_is_linear_l58_58462


namespace quadratic_points_order_l58_58939

theorem quadratic_points_order (c y1 y2 : ℝ) 
  (hA : y1 = 0^2 - 6 * 0 + c)
  (hB : y2 = 4^2 - 6 * 4 + c) : 
  y1 > y2 := 
by 
  sorry

end quadratic_points_order_l58_58939


namespace find_a_l58_58619

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a (a : ℝ) (h : ∃ x, f x a = 3) : a = 1 ∨ a = 7 := 
sorry

end find_a_l58_58619


namespace time_for_x_to_complete_work_l58_58374

-- Definitions from the conditions:
def work_done_by_y_per_day : ℝ := 1 / 30
def work_done_by_x_in_8_days (W_x : ℝ) := 8 * W_x
def work_done_by_y_in_24_days : ℝ := 24 * work_done_by_y_per_day

-- The total work W:
def total_work : ℝ := 30 * work_done_by_y_per_day

-- Time taken by y to finish the work. 
def remaining_work_provided_by_y : ℝ :=
  work_done_by_x_in_8_days W_x + work_done_by_y_in_24_days

-- The goal is to prove that x can complete the work in 40 days.
theorem time_for_x_to_complete_work (W_x W_y: ℝ):
  work_done_by_y_per_day = W_y → 8 * W_x = 6 * W_y → W_x / W_y = 3 / 4 → 30 / 3 * 4 = 40 := 
by
  intros, 
  sorry

end time_for_x_to_complete_work_l58_58374


namespace part1_geometric_and_formula_l58_58866

variable {S : ℕ → ℝ}
variable {a : ℕ → ℝ}

axiom (h : ∀ n : ℕ, S n + n = 2 * a n)

theorem part1_geometric_and_formula :
  (∀ n : ℕ, 1 < n → a n + 1 = 2 * (a (n - 1) + 1)) ∧ (a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a n = 2^n - 1) :=
by 
  sorry

end part1_geometric_and_formula_l58_58866


namespace alloy_b_amount_l58_58020

theorem alloy_b_amount
    (a_weight : ℝ)
    (b_weight : ℝ)
    (lead_tin_ratio_a : ℝ)
    (tin_copper_ratio_b : ℝ)
    (tin_total_new_alloy : ℝ) :
    a_weight = 170 →
    lead_tin_ratio_a = 1 / 4 →
    tin_copper_ratio_b = 3 / 8 →
    tin_total_new_alloy = 221.25 →
    let tin_in_a := lead_tin_ratio_a * a_weight
    let tin_in_b := tin_copper_ratio_b * b_weight in
    tin_in_a + tin_in_b = tin_total_new_alloy →
    b_weight = 250 :=
begin
  intros h_a_weight h_lead_tin_ratio_a h_tin_copper_ratio_b h_tin_total_new_alloy,
  sorry
end

end alloy_b_amount_l58_58020


namespace cos_alpha_l58_58104

-- Given condition
axiom sin_eq_one_fifth : ∀ (α : ℝ), sin (5 * π / 2 + α) = 1 / 5

-- Theorem statement
theorem cos_alpha : ∀ (α : ℝ), sin (5 * π / 2 + α) = 1 / 5 → cos α = 1 / 5 :=
by
  intro α h
  rw [sin_eq_one_fifth α] at h
  sorry

end cos_alpha_l58_58104


namespace no_possible_values_of_k_l58_58805

theorem no_possible_values_of_k :
  (∃ (p q : ℕ), p.prime ∧ q.prime ∧ p + q = 22 ∧ p * q = k) → False :=
  by {
    sorry
  }

end no_possible_values_of_k_l58_58805


namespace g_is_odd_l58_58937

variables (f : ℝ → ℝ) (h k : ℝ)

def symmetric_about_point (f : ℝ → ℝ) (h k : ℝ) : Prop :=
∀ x, f(h - x + h) = 2*k - f(h + x)

-- Given that f is symmetric about the point (h, k),
-- we need to show that g(x) = f(x + h) - k is an odd function
theorem g_is_odd (hf : symmetric_about_point f h k) : 
  odd (λ x, f(x + h) - k) :=
sorry

end g_is_odd_l58_58937


namespace count_surjective_mappings_l58_58129
open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℤ := {-1, -2}

theorem count_surjective_mappings (A B : Set α) 
  (f : ℤ → ℕ)
  (hA : A = {1, 2, 3, 4}) 
  (hB : B = {-1, -2}) 
  (h : ∀ b ∈ B, ∃ a ∈ A, f a = b) :
  ∃ n : ℕ, n = 14 :=
by
  sorry

end count_surjective_mappings_l58_58129


namespace find_range_of_a_l58_58606

noncomputable def f (x : ℝ) : ℝ :=
if (-2 ≤ x ∧ x ≤ 0) then (1/2)^x - 1
else if (0 ≤ x ∧ x ≤ 2) then 2^x - 1
else if (2 ≤ x ∧ x ≤ 6) then 2^(x-4) - 1
else 0 -- We can assume it's zero outside of the specified intervals for this proof

lemma f_even (x : ℝ) : f x = f (-x) :=
begin
  sorry,
end

lemma f_periodic (x : ℝ) : f x = f (x + 4) :=
begin
  sorry,
end

theorem find_range_of_a (a : ℝ) (h : 1 < a) :
  (∃! x : ℝ, x ∈ Icc (-2) 6 ∧ f x = real.log (x+2) / real.log a) ∧ -- Exactly three distinct real roots
  2^(2/3) < a ∧ a < 2 :=
begin
  sorry,
end

end find_range_of_a_l58_58606


namespace find_other_solution_l58_58877

theorem find_other_solution (x : ℚ) (hx : 45 * (2 / 5 : ℚ)^2 + 22 = 56 * (2 / 5 : ℚ) - 9) : x = 7 / 9 :=
by 
  sorry

end find_other_solution_l58_58877


namespace greatest_value_of_squares_l58_58971

-- Given conditions
variables (a b c d : ℝ)
variables (h1 : a + b = 20)
variables (h2 : ab + c + d = 105)
variables (h3 : ad + bc = 225)
variables (h4 : cd = 144)

theorem greatest_value_of_squares : a^2 + b^2 + c^2 + d^2 ≤ 150 := by
  sorry

end greatest_value_of_squares_l58_58971


namespace investment_principal_approx_equal_60000_l58_58761

theorem investment_principal_approx_equal_60000 (P : ℝ) 
    (h1 : 0.02 = 0.02) 
    (h2 : 20 = 20) 
    (h3 : 2 = 2) 
    (h4 : 1446 = 1446) :
    P * (1 + 0.1)^4 - P * (1 + 0.2)^2 = 1446 → 
    P ≈ 60000 :=
begin
  sorry
end

end investment_principal_approx_equal_60000_l58_58761


namespace expected_team_a_score_l58_58661

open ProbabilityTheory

noncomputable def team_a_win_prob (match : ℕ) : ℝ :=
match match with
| 0 => 2/3  -- A1 vs B1
| 1 => 2/5  -- A2 vs B2
| _ => 2/5  -- A3 vs B3
end

noncomputable def team_a_expected_score : ℝ :=
  let p_win_0 := 2 / 3
  let p_win_1 := 2 / 5
  let p_win_2 := 2 / 5
  3 * p_win_0 * p_win_1 * p_win_2 +
  2 * (p_win_0 * p_win_1 * (3 / 5) + (1 / 3) * p_win_1 * p_win_2 + p_win_0 * (3 / 5) * p_win_2) +
  1 * (p_win_0 * (3 / 5) * (3 / 5) + (1 / 3) * p_win_1 * (3 / 5) + (1 / 3) * (3 / 5) * p_win_2) +
  0 * ((1 / 3) * (3 / 5) * (3 / 5))

theorem expected_team_a_score : team_a_expected_score = 22 / 15 :=
by
  sorry

end expected_team_a_score_l58_58661


namespace sum_of_ab_l58_58246

variable {ℝ : Type}

def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 5
  else if x < 0 then b * x - 3
  else a + b

theorem sum_of_ab (a b : ℝ) (h1 : f a b 1 = 7) (h2 : f a b 0 = 8) (h3 : f a b (-2) = -11) :
  a + b = 8 :=
by
  sorry

end sum_of_ab_l58_58246


namespace min_combined_horses_ponies_l58_58778

theorem min_combined_horses_ponies (P H : ℕ) (h1 : P % 6 = 0) (h2 : P % 9 = 0) (h3 : H = P + 4) 
    (h4 : ∃ p_sh : ℕ, p_sh = (5 / 6 : ℚ) * P) (h5 : ∃ ic_p_sh : ℕ, ic_p_sh = (2 / 3 : ℚ) * (5 / 6 : ℚ) * P) :
  P + H = 40 :=
begin
  sorry
end

end min_combined_horses_ponies_l58_58778


namespace no_such_polynomial_exists_l58_58823

noncomputable def P : Polynomial ℝ := sorry
def Q : Polynomial ℝ := sorry

theorem no_such_polynomial_exists :
  (∃ Q : Polynomial ℝ, P = (X^3 - C 2) * Q + Q^4 ∧ P.eval (-2) + P.eval 2 = -34) → False := sorry

end no_such_polynomial_exists_l58_58823


namespace min_sum_b_l58_58499

-- Definitions and conditions as per the problem statement
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a n = 4 * n - 2

def S (a : ℕ → ℕ) (n : ℕ) : ℚ :=
(1 / 8 : ℚ) * ((a n + 2) ^ 2)

def b (a : ℕ → ℕ) (n : ℕ) : ℤ :=
(1 / 2 : ℚ) * a n - 30

-- Proving that the minimum sum of first terms of sequence b occurs at term 15
theorem min_sum_b {a : ℕ → ℕ} (h_arith : arithmetic_seq a) :
  ∃ n : ℕ, n = 15 ∧ ∀ m : ℕ, m → n → S a m = S a n :=
sorry

end min_sum_b_l58_58499


namespace length_of_lateral_edge_l58_58185

theorem length_of_lateral_edge (vertices : ℕ) (total_length : ℝ) (num_lateral_edges : ℕ) (length_each_edge : ℝ) : 
  vertices = 12 ∧ total_length = 30 ∧ num_lateral_edges = 6 → length_each_edge = 5 :=
by
  intros h
  cases h with h_vertices h_rest
  cases h_rest with h_length h_num_edges
  have h_calculation : length_each_edge = total_length / num_lateral_edges := sorry
  rw [h_vertices, h_length, h_num_edges] at h_calculation
  norm_num at h_calculation
  exact h_calculation
  sorry

end length_of_lateral_edge_l58_58185


namespace sum_of_possible_values_of_d_l58_58068

theorem sum_of_possible_values_of_d : 
  ∀ b c d e f g h : ℕ,
  (72 * b * c = d * e * f) ∧
  (d * e * f = g * h * 4) ∧
  (72 * e * 4 = 288 * e = 4 * g * h = P) ∧
  (288 = 4 * g * h) ∧
  e > 0 ∧ b > 0 ∧ c > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ 
  e ∈ [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] ∧
  4 divides (g * h) ∧ g ≥ h
  → sum (filter (λ (d : ℕ), 288 % d = 0) (range (288 + 1))) = 45 :=
sorry

end sum_of_possible_values_of_d_l58_58068


namespace area_triangle_half_hexagon_l58_58580

-- Define the area function
noncomputable def area (s : Set Point) : ℝ := sorry

structure ConvexHexagon :=
(A B C A1 B1 C1 : Point)
(convex : Convex {p : Point | p ∈ {A, B, C, A1, B1, C1}})
(eq1 : dist A B1 = dist A C1)
(eq2 : dist B C1 = dist B A1)
(eq3 : dist C A1 = dist C B1)
(angles : angle A + angle B + angle C = angle A1 + angle B1 + angle C1)

theorem area_triangle_half_hexagon (hex : ConvexHexagon) :
  area {hex.A, hex.B, hex.C} = (area {hex.A, hex.C1, hex.B, hex.A1, hex.C, hex.B1} / 2) :=
sorry

end area_triangle_half_hexagon_l58_58580


namespace horner_method_correct_l58_58527

noncomputable def horner_method_value (x : ℕ) (coeffs : List ℕ) (values : List ℕ) : ℕ :=
  coeffs.reverse.foldl (λ acc c, acc * x + c) 0

theorem horner_method_correct :
  let f := λ x : ℕ, 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7
  v₀ := 2
  v₁ := 2 * 5 - 5
  v₂ := 5 * 5 - 4
  v₃ := 21 * 5 + 3
  in v₃ = 108 := sorry

end horner_method_correct_l58_58527


namespace equal_segments_l58_58609

open_locale classical
noncomputable theory

-- Definitions and conditions
def regular_pentagon (ABCDE : set (affine ℝℝ)) : Prop :=
  let vertices := [(A : Point), B, C, D, E] in
  ∀ i j, i ≠ j → ∃ r, ∀ {k} (hk : k ∈ vertices), dist (vertices.nth_le i hk) (vertices.nth_le j hk) = r
  
variables {A B C D E P Q R M : Point}

-- Regular pentagon ABCDE
axiom pentagon_ABCDE : regular_pentagon {A, B, C, D, E}

-- Key points location
axiom point_between : P ≠ M ∧ collinear {M, D, P}

-- Intersection conditions
axiom circumcircle_intersection : ∃ Q, Q ∈ circle_through {A, B, P} ∧ Q ∈ line_through A E ∧ Q ≠ A
axiom perpendicular_line : ∃ R, collinear {P, R} ∧ perpendicular (line_through P R) (line_through C D)

-- Required to prove
theorem equal_segments : dist A R = dist Q R :=
sorry

end equal_segments_l58_58609


namespace bikers_meet_again_in_36_minutes_l58_58334

theorem bikers_meet_again_in_36_minutes :
    Nat.lcm 12 18 = 36 :=
sorry

end bikers_meet_again_in_36_minutes_l58_58334


namespace tangent_intersect_x_axis_l58_58387

theorem tangent_intersect_x_axis :
  ∀ (x : ℝ), 
    (∃ (C₁ C₂ : ℝ × ℝ), 
      C₁ = (0, 0) ∧ C₂ = (15, 0) ∧
      (∃ (r₁ r₂ : ℝ), 
        r₁ = 2 ∧ r₂ = 7 ∧ 
        (∃ (tangent_line : ℝ → ℝ), 
          (∀ (y : ℝ), tangentLine y = 0 * y + x) ∧ 
          (∃ (pt₁ pt₂ : ℝ × ℝ), 
            tangentLine pt₁.snd = pt₁.fst ∧ 
            tangentLine pt₂.snd = pt₂.fst ∧ 
            ∀ (r : ℝ), r > 0 → 
              (f pt₁ - (0, 0)).norm = r₁ ∧ 
              (f pt₂ - (15, 0)).norm = r₂ ∧ 
              ((2 / x) = (7 / (15 - x)) → x = 10 / 3))))) :=
begin 
  sorry 
end

end tangent_intersect_x_axis_l58_58387


namespace scientific_notation_l58_58316

theorem scientific_notation (a n : ℝ) (h1 : 100000000 = a * 10^n) (h2 : 1 ≤ a) (h3 : a < 10) : 
  a = 1 ∧ n = 8 :=
by
  sorry

end scientific_notation_l58_58316


namespace simplify_f_increasing_intervals_l58_58893

noncomputable def f (x : ℝ) : ℝ :=
  2 * (cos x) ^ 2 + 2 * sqrt 3 * sin x * cos x - 1

theorem simplify_f (x : ℝ) :
  f x = 2 * sin (2 * x + π / 6) := by
  sorry

theorem increasing_intervals :
  ∀ k : ℤ, (k : ℝ) * π - π / 3 ≤ x ∧ x ≤ (k : ℝ) * π + π / 6 → (∀ x, differentiable ℝ f x ∧ 0 < deriv (λ x, f x) x) := by
  sorry

end simplify_f_increasing_intervals_l58_58893


namespace rank_from_right_l58_58419

theorem rank_from_right (total_students rank_from_left : ℕ) (h1 : total_students = 31) (h2 : rank_from_left = 11) : 
  (total_students - rank_from_left + 1 = 21) :=
by
  rw [h1, h2]
  simp
  sorry

end rank_from_right_l58_58419


namespace max_salary_single_player_l58_58410

theorem max_salary_single_player 
  (min_salary : ℕ)
  (team_salary_cap : ℕ)
  (num_players : ℕ) 
  (min_salaries_total : ℕ) 
  (total_salary_eq : 11 * min_salary + min_salaries_total = team_salary_cap) :
  ∃ max_salary : ℕ, max_salary = team_salary_cap - min_salaries_total := 
  by 
    let max_salary := team_salary_cap - min_salaries_total
    use max_salary
    exact sorry

-- Now we provide the specific values from the problem
def basketball_problem : max_salary_single_player 20000 500000 12 220000 :=
by 
  apply max_salary_single_player
  exact rfl

end max_salary_single_player_l58_58410


namespace max_area_ratio_l58_58142

noncomputable theory

variables (A B C I P : ℝ)
variables (r : ℝ := 2) (PI : ℝ := 1) (S_APB S_APC : ℝ)

-- Define the conditions given in the problem
def conditions := 
  (inradius : r = 2) ∧   -- Radius of incircle
  (center : I = 2*r) ∧   -- Center I derived from equilateral triangle property
  (pointP : (P - I) = 1) -- PI = 1

-- Define the areas of the triangles
def area_ratio := S_APB / S_APC

-- The main theorem statement
theorem max_area_ratio (h : conditions) : 
  ∃ S_APB S_APC, area_ratio = (3 + Real.sqrt 5) / 2 :=
begin
  -- We would provide proof here, but since it is requested to skip the proof...
  sorry,
end

end max_area_ratio_l58_58142


namespace option_D_l58_58512

-- Definitions for perpendicular and parallel relationships
variables {Line Plane : Type}

-- Conditions stated in natural language elaborated in variables and assumptions
variables (m n : Line) (α β : Plane)

-- Assume conditions given in the problem
variables (h1 : m ⟂ α) (h2 : m ⟂ β) (h3 : n ⟂ α)

-- Proof goal based on the correct answer
theorem option_D : n ⟂ β :=
by
  sorry

end option_D_l58_58512


namespace defective_units_l58_58417

theorem defective_units (D : ℕ) : 
  (20 - D = 3 + 5 + 7) → D = 5 :=
by
  intro h1
  have h2 : 3 + 5 + 7 = 15 := rfl
  rw h2 at h1
  linarith

end defective_units_l58_58417


namespace find_y_coordinate_of_Q_l58_58582

-- Given points and their coordinates
def P : ℝ × ℝ := (4, 2)
def R : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (2, 10)

-- Function to calculate the slope between two points
def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

-- Definition of perpendicular slopes
def perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Proof statement
theorem find_y_coordinate_of_Q:
  ∀ (P R Q : ℝ × ℝ),
    slope P R = 1 / 4 →
    perpendicular_slopes (slope P R) (slope P Q) →
    P = (4, 2) →
    R = (0, 1) →
    Q.1 = 2 →
    Q.2 = 10 := 
by
  intros P R Q h_slope_PR h_perpendicular_slope hP hR hQx
  sorry

end find_y_coordinate_of_Q_l58_58582


namespace incenter_on_segment_dividing_triangle_l58_58784

theorem incenter_on_segment_dividing_triangle 
  {A B C E F : Point} (a b c: ℝ)
  (k l : ℝ)
  (hE : E ∈ line_segment B C)
  (hF : F ∈ line_segment A C)
  (h_area : 2 * k * l = 1)
  (h_perimeter : c = (2 * k - 1) * a + (2 * l - 1) * b)
  (incenter : Point) :
  incenter = incenter_triangle A B C →
  incenter ∈ line_segment E F :=
by
  sorry

end incenter_on_segment_dividing_triangle_l58_58784


namespace find_roots_of_op_eq_zero_l58_58852

def op (a b : ℝ) : ℝ := a^2 - 5 * a + 2 * b

theorem find_roots_of_op_eq_zero :
  (∀ x : ℝ, op x 3 = 0 ↔ x = 2 ∨ x = 3) :=
by
  intro x
  unfold op
  simp only [eq_self_iff_true, or_true, sub_self, zero_add]
  sorry

end find_roots_of_op_eq_zero_l58_58852


namespace solution_to_equation_l58_58071

def star (a b : ℝ) : ℝ := a^2 + b^2
def _star_ (a b : ℝ) : ℝ := (a * b) / 2

theorem solution_to_equation : ∀ (x : ℝ), star 3 x = _star_ x 12 → x = 3 :=
by
  intros x h
  sorry

end solution_to_equation_l58_58071


namespace fraction_of_grid_covered_by_triangle_l58_58117

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (6, 2)
def C : (ℝ × ℝ) := (4, 5)

-- Define the dimensions of the grid
def grid_width : ℝ := 7
def grid_height : ℝ := 6

-- The problem statement
theorem fraction_of_grid_covered_by_triangle :
  let area_triangle := (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  let area_grid := grid_width * grid_height in
  (area_triangle / area_grid) = 1 / 7 :=
by
  -- Proof goes here
  sorry

end fraction_of_grid_covered_by_triangle_l58_58117


namespace exists_six_different_cascades_can_color_nat_12_colors_l58_58818

-- Definitions
def cascade (r : ℕ) : set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 12 ∧ n = k * r}

-- Part (a)
theorem exists_six_different_cascades (a b : ℕ) :
  ∃ rs : fin 6 → ℕ, ∀ i : fin 6, a ∈ cascade (rs i) ∧ b ∈ cascade (rs i) :=
sorry

-- Part (b)
theorem can_color_nat_12_colors :
  ∃ (coloring : ℕ → fin 12), ∀ r : ℕ, function.bijective (λ k : fin 12, coloring (k * r)) :=
sorry

end exists_six_different_cascades_can_color_nat_12_colors_l58_58818


namespace part1_part2_l58_58138

noncomputable theory

-- Define the given conditions
def z (m : ℝ) : ℂ := ⟨m, 2⟩

-- First part: Prove the magnitude of z 
theorem part1 (m : ℝ) (h : m^2 + 6 * m + 13 = 0) : abs (z m) = real.sqrt 13 :=
sorry

-- Second part: Define z_1 and prove the range of a
def z1 (a : ℝ) (z : ℂ) : ℂ := (a + (complex.I ^ 2023)) / (complex.conj z)

theorem part2 (a : ℝ) (m : ℝ) (h : m = -3)
(h_range : (z1 a (z m)).re < 0 ∧ (z1 a (z m)).im < 0) : (-2 / 3) < a ∧ a < (3 / 2) :=
sorry

end part1_part2_l58_58138


namespace no_single_two_three_digit_solution_l58_58744

theorem no_single_two_three_digit_solution :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x ∧ x ≤ 9) ∧
    (10 ≤ y ∧ y ≤ 99) ∧
    (100 ≤ z ∧ z ≤ 999) ∧
    (1/x : ℝ) = 1/y + 1/z :=
by
  sorry

end no_single_two_three_digit_solution_l58_58744


namespace solve_for_p_l58_58598

-- Define the complex number structure if necessary (standard in Lean's Mathlib)
def Complex : Type := ℂ

-- State the given conditions as hypotheses
variables (q p v : Complex)
variables (h1 : q = 3)
variables (h2 : v = 3 + (75 : ℂ) * Complex.I)

-- State the primary equation as a hypothesis
variables (h3 : 3 * p - v = 5000)

-- State the theorem to prove
theorem solve_for_p : p = 1667 + (25 : ℂ) * Complex.I :=
by
  -- Placeholder for proof steps
  sorry

end solve_for_p_l58_58598


namespace probability_A_speaks_truth_l58_58413

variable (P : α → ℝ)
variable A B : α

-- Conditions
def P_B : Prop := P B = 0.60
def P_A_and_B : Prop := P (A ∧ B) = 0.51

-- Theorem statement: The percentage of the time A speaks the truth is 85%
theorem probability_A_speaks_truth
  (h₁ : P_B)
  (h₂ : P_A_and_B) :
  P A = 0.85 :=
sorry

end probability_A_speaks_truth_l58_58413


namespace assignment_plans_count_l58_58192

theorem assignment_plans_count :
  ∃ (V A : ℕ), (V = 5) ∧ (A = 3) ∧ (∀ t, t = (3 ^ V - 3 * 2 ^ V + 3 * 1 ^ V) → t = 150) :=
begin
  use [5, 3],
  split,
  { refl },
  split,
  { refl },
  intro t,
  intro h_eq,
  simp at h_eq,
  exact h_eq,
end

end assignment_plans_count_l58_58192


namespace domain_of_function1_l58_58673

def domain_eq_set (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  ∀ x, (x ∈ S ↔ x ∈ domain f)

def function1 (x : ℝ) : ℝ :=
  real.sqrt (1 - 2 * real.cos x) + real.log (2 * real.sin x - real.sqrt 2)

noncomputable def domain_function1 : set ℝ :=
  {x | ∃ k : ℤ, x ∈ set.Icc (2 * k * real.pi + real.pi / 3) (2 * k * real.pi + 3 * real.pi / 4)}

theorem domain_of_function1 :
  domain_eq_set function1 domain_function1 :=
sorry

end domain_of_function1_l58_58673


namespace gcd_lcm_divisible_l58_58694

theorem gcd_lcm_divisible (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b + Nat.lcm a b = a + b) : a % b = 0 ∨ b % a = 0 := 
sorry

end gcd_lcm_divisible_l58_58694


namespace incenter_of_triangle_intersects_l58_58302

theorem incenter_of_triangle_intersects (
  ABC : Triangle,
  A1 B1 C1 A2 B2 C2 : Point,
  h₁ : incircle ABC touches_side BC at A1,
  h₂ : incircle ABC touches_side AC at B1,
  h₃ : incircle ABC touches_side AB at C1,
  h₄ : A2 = incenter (Triangle.mk A B1 C1),
  h₅ : B2 = incenter (Triangle.mk B A1 C1),
  h₆ : C2 = incenter (Triangle.mk C A1 B1)
) : 
  Concurrent (Line.mk A1 A2) (Line.mk B1 B2) (Line.mk C1 C2) :=
sorry

end incenter_of_triangle_intersects_l58_58302


namespace primes_eq_2_3_7_l58_58074

theorem primes_eq_2_3_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 :=
by
  sorry

end primes_eq_2_3_7_l58_58074


namespace plane_through_points_l58_58088

def point := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℤ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points : 
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd (Int.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1) ∧
  plane_equation A B C D 2 (-3) 5 ∧
  plane_equation A B C D (-1) (-3) 7 ∧
  plane_equation A B C D (-4) (-5) 6 ∧
  (A = 2) ∧ (B = -9) ∧ (C = 3) ∧ (D = -46) :=
sorry

end plane_through_points_l58_58088


namespace candle_height_ratio_l58_58714

noncomputable def burning_rate_first (initial_height : ℝ) : ℝ := initial_height / 5
noncomputable def burning_rate_second (initial_height : ℝ) : ℝ := initial_height / 8

theorem candle_height_ratio (t : ℝ) (initial_height : ℝ) 
  (h1 : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 → (initial_height - burning_rate_first initial_height * t)) 
  (h2 : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 8 → (initial_height - burning_rate_second initial_height * t)) :
  (initial_height - burning_rate_first initial_height * (80 / 7) = 
  3 * (initial_height - burning_rate_second initial_height * (80 / 7))) :=
by
  sorry

end candle_height_ratio_l58_58714


namespace determine_expr_l58_58851

noncomputable def expr (a b c d : ℝ) : ℝ :=
  (1 + a + a * b) / (1 + a + a * b + a * b * c) +
  (1 + b + b * c) / (1 + b + b * c + b * c * d) +
  (1 + c + c * d) / (1 + c + c * d + c * d * a) +
  (1 + d + d * a) / (1 + d + d * a + d * a * b)

theorem determine_expr (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  expr a b c d = 2 :=
sorry

end determine_expr_l58_58851


namespace remaining_family_member_age_l58_58695

variable (total_age father_age sister_age : ℕ) (remaining_member_age : ℕ)

def mother_age := father_age - 2
def brother_age := father_age / 2
def known_total_age := father_age + mother_age + brother_age + sister_age

theorem remaining_family_member_age : 
  total_age = 200 ∧ 
  father_age = 60 ∧ 
  sister_age = 40 ∧ 
  known_total_age = total_age - remaining_member_age → 
  remaining_member_age = 12 := by
  sorry

end remaining_family_member_age_l58_58695


namespace num_roots_of_unity_satisfy_cubic_l58_58042

def root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def cubic_eqn_root (z : ℂ) (a b c : ℤ) : Prop :=
  z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) = 0

theorem num_roots_of_unity_satisfy_cubic (a b c : ℤ) (n : ℕ) 
    (h_n : n ≥ 1) : ∃! z : ℂ, root_of_unity z n ∧ cubic_eqn_root z a b c := sorry

end num_roots_of_unity_satisfy_cubic_l58_58042


namespace perpendicular_vectors_l58_58913

-- Define the vectors a and b
def vector_a : ℝ × ℝ × ℝ := (2, -1, 3)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the theorem statement
theorem perpendicular_vectors (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = 10 / 3 :=
by
  sorry

end perpendicular_vectors_l58_58913


namespace pentagon_area_correct_l58_58615

noncomputable def pentagon_area : ℝ :=
  let FG := 6 in
  let GH := 7 in
  let HI := 7 in
  let IJ := 7 in
  let JF := 8 in
  let r := 1.25 in -- inscribed circle radius (to reflect symmetry/condition indirectly given by inscribed circle)
  2 * 6.125 -- Direct computation based on the given configuration and symmetry.

theorem pentagon_area_correct :
  pentagon_area = 12.25 :=
by
  sorry

end pentagon_area_correct_l58_58615


namespace probability_of_snow_at_least_once_l58_58631

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end probability_of_snow_at_least_once_l58_58631


namespace geometric_arithmetic_sequence_ratio_l58_58472

-- Given a positive geometric sequence {a_n} with a_3, a_5, a_6 forming an arithmetic sequence,
-- we need to prove that (a_3 + a_5) / (a_4 + a_6) is among specific values {1, (sqrt 5 - 1) / 2}

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos: ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 :=
by
  -- The proof is omitted
  sorry

end geometric_arithmetic_sequence_ratio_l58_58472


namespace total_population_l58_58571

theorem total_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) : b + g + t = 13 * t :=
by
  sorry

end total_population_l58_58571


namespace smallest_degree_of_poly_l58_58280

theorem smallest_degree_of_poly :
  ∃ (P : Polynomial ℚ), P ≠ 0 ∧
    (Polynomial.root P (3 - Real.sqrt 8)) ∧
    (Polynomial.root P (5 + Real.sqrt 12)) ∧
    (Polynomial.root P (16 - 3 * Real.sqrt 7)) ∧
    (Polynomial.root P (- Real.sqrt 3)) ∧
    (Polynomial.degree P = 8) :=
sorry

end smallest_degree_of_poly_l58_58280


namespace angle_measure_supplement_complement_l58_58344

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l58_58344


namespace probability_between_R_and_S_l58_58642

open Set

variables {P Q R S : ℝ} -- Assume the points are represented by real numbers
variables (PQ PR QR : ℝ)
-- The conditions of the problem
def points_on_line_segment : Prop := (PQ = 4 * PR) ∧ (PQ = 8 * QR)

-- The question translated into Lean statement
theorem probability_between_R_and_S (h : points_on_line_segment PQ PR QR) : 
  let RS := PQ - PR - QR in
  (RS / PQ) = 5 / 8 :=
by
  rcases h with ⟨h1, h2⟩
  sorry -- proof will be filled here

end probability_between_R_and_S_l58_58642


namespace number_of_same_tens_units_digit_from_1985_to_4891_l58_58100

open Int

def has_same_tens_units_digit (n : ℤ) : Prop :=
  (n / 10) % 10 = n % 10

def count_same_tens_units_digit_in_range (start end : ℤ) : ℤ :=
  finset.card (finset.filter has_same_tens_units_digit (finset.Icc start end))

theorem number_of_same_tens_units_digit_from_1985_to_4891 :
  count_same_tens_units_digit_in_range 1985 4891 = 291 :=
by
  sorry

end number_of_same_tens_units_digit_from_1985_to_4891_l58_58100


namespace count_arithmetic_sequence_digits_l58_58168

def valid_tuple (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  2 * b = a + c

theorem count_arithmetic_sequence_digits :
  (finset.univ.filter (λ (t : ℕ × ℕ × ℕ), valid_tuple t.1 t.2.1 t.2.2)).card = 45 :=
by {
  sorry
}

end count_arithmetic_sequence_digits_l58_58168


namespace count_perfect_squares_ones_digit_5_6_7_lt_100_l58_58538

theorem count_perfect_squares_ones_digit_5_6_7_lt_100 : 
  (finset.card (finset.filter (λ n : ℤ, (n % 10 = 5 ∨ n % 10 = 6 ∨ n % 10 = 7)
                                  ∧ (n : ℤ)^2 < 100) 
                                  (finset.range 10))) = 1 :=
begin
  sorry
end

end count_perfect_squares_ones_digit_5_6_7_lt_100_l58_58538


namespace hyperbola_asymptotes_l58_58901

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                              (h3 : 2 * a = 4) (h4 : 2 * b = 6) : 
                              ∀ x y : ℝ, (y = (3 / 2) * x) ∨ (y = - (3 / 2) * x) := by
  sorry

end hyperbola_asymptotes_l58_58901


namespace sum_of_angles_of_roots_l58_58837

noncomputable def theta_sum (z : ℂ) : ℝ :=
if ∃ θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ z = 2 * complex.ofReal (cos θ) + 2 * complex.I * complex.ofReal (sin θ)
then
  θ
else
  0 -- This case should ideally never happen, but we include it for completeness.

theorem sum_of_angles_of_roots :
  let solutions := [2 * complex.ofReal (cos 36) + 2 * complex.I * complex.ofReal (sin 36),
                    2 * complex.ofReal (cos 108) + 2 * complex.I * complex.ofReal (sin 108),
                    2 * complex.ofReal (cos 180) + 2 * complex.I * complex.ofReal (sin 180),
                    2 * complex.ofReal (cos 252) + 2 * complex.I * complex.ofReal (sin 252),
                    2 * complex.ofReal (cos 324) + 2 * complex.I * complex.ofReal (sin 324)] in
  ∑ z in solutions, theta_sum z = 900 := sorry

end sum_of_angles_of_roots_l58_58837


namespace unit_vector_is_orthogonal_and_unit_l58_58458

def vec1 : ℝ × ℝ × ℝ := (2, 3, 1)
def vec2 : ℝ × ℝ × ℝ := (-1, 2, 4)

def unit_vector : ℝ × ℝ × ℝ := (10 / Real.sqrt 230, -9 / Real.sqrt 230, 7 / Real.sqrt 230)

theorem unit_vector_is_orthogonal_and_unit:
  (vec1.1 * unit_vector.1 + vec1.2 * unit_vector.2 + vec1.3 * unit_vector.3 = 0) ∧
  (vec2.1 * unit_vector.1 + vec2.2 * unit_vector.2 + vec2.3 * unit_vector.3 = 0) ∧
  (unit_vector.1^2 + unit_vector.2^2 + unit_vector.3^2 = 1) :=
by
  sorry

end unit_vector_is_orthogonal_and_unit_l58_58458


namespace minute_hand_sweep_probability_l58_58996

theorem minute_hand_sweep_probability :
  ∀ t : ℕ, ∃ p : ℚ, p = 1 / 3 →
  (t % 60 = 0 ∨ t % 60 = 5 ∨ t % 60 = 10 ∨ t % 60 = 15 ∨
   t % 60 = 20 ∨ t % 60 = 25 ∨ t % 60 = 30 ∨ t % 60 = 35 ∨
   t % 60 = 40 ∨ t % 60 = 45 ∨ t % 60 = 50 ∨ t % 60 = 55) →
  (∃ m : ℕ, m = (t + 20) % 60 ∧
   (m % 60 = 0 ∨ m % 60 = 3 ∨ m % 60 = 6 ∨ m % 60 = 9) → 
   (m - t) % 60 ∈ ({20} : set ℕ) → 
   probability_sweep (flies := {12, 3, 6, 9})
     (minute_hand := (λ t, t % 60)) 
     (swept_flies := 2) (t := t) = p) :=
sorry

end minute_hand_sweep_probability_l58_58996


namespace right_triangle_leg_product_eq_half_hypotenuse_squared_l58_58650

theorem right_triangle_leg_product_eq_half_hypotenuse_squared 
  (ABC : Triangle)
  (hABC_rt : ABC.is_right_at B)
  (hCBA_acute : ABC.∠CBA = 15) :
  let AB := ABC.side_length₁_AB in
  let BC := ABC.side_length₂_BC in
  let CA := ABC.side_length₃_CA in
  (BC * CA = (1 / 2) * AB ^ 2) :=
sorry

end right_triangle_leg_product_eq_half_hypotenuse_squared_l58_58650


namespace total_spent_l58_58794

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l58_58794


namespace puppies_per_cage_l58_58404

-- Conditions
variables (total_puppies sold_puppies cages initial_puppies per_cage : ℕ)
variables (h_total : total_puppies = 13)
variables (h_sold : sold_puppies = 7)
variables (h_cages : cages = 3)
variables (h_equal_cages : total_puppies - sold_puppies = cages * per_cage)

-- Question
theorem puppies_per_cage :
  per_cage = 2 :=
by {
  sorry
}

end puppies_per_cage_l58_58404


namespace union_complement_eq_l58_58987

open Finset

variable (U P Q : Finset ℕ) (U_def : U = {1, 2, 3, 4}) (P_def : P = {1, 2}) (Q_def : Q = {1, 3})

theorem union_complement_eq : P ∪ (U \ Q) = {1, 2, 4} :=
by
  sorry

end union_complement_eq_l58_58987


namespace A_is_three_times_faster_than_B_l58_58031

-- Define the rates of A and B
def rate_A := (1 : ℝ) / 24
def rate_B (C : ℝ) := rate_A / C

-- Combined rate of A and B
def combined_rate (C : ℝ) := rate_A + rate_B C

-- Define the main theorem statement
theorem A_is_three_times_faster_than_B : ∃ C : ℝ, combined_rate C = 1 / 18 ∧ C = 3 :=
by
  sorry

end A_is_three_times_faster_than_B_l58_58031


namespace suitable_comprehensive_survey_l58_58052

def investigate_service_life_of_lamps : Prop := 
  -- This would typically involve checking a subset rather than every lamp
  sorry

def investigate_water_quality : Prop := 
  -- This would typically involve sampling rather than checking every point
  sorry

def investigate_sports_activities : Prop := 
  -- This would typically involve sampling rather than collecting data on every student
  sorry

def test_components_of_rocket : Prop := 
  -- Given the critical importance and manageable number of components, this requires comprehensive examination
  sorry

def most_suitable_for_comprehensive_survey : Prop :=
  test_components_of_rocket ∧ ¬investigate_service_life_of_lamps ∧ 
  ¬investigate_water_quality ∧ ¬investigate_sports_activities

theorem suitable_comprehensive_survey : most_suitable_for_comprehensive_survey :=
  sorry

end suitable_comprehensive_survey_l58_58052


namespace visible_factor_numbers_count_l58_58036

def visible_factor_number (n : ℕ) : Prop :=
  let digits := Int.toDigits 10 n
  let non_zero_digits := digits.filter (· ≠ 0)
  let sum_digits := digits.sum
  non_zero_digits.all (λ d => n % d = 0) ∧ n % sum_digits = 0

theorem visible_factor_numbers_count : 
  ∃! n, n = 8 ∧ ∀ m, 200 ≤ m ∧ m ≤ 250 → visible_factor_number m := 
by sorry

end visible_factor_numbers_count_l58_58036


namespace number_of_irrational_in_my_set_l58_58587

-- Definitions based on the conditions in a)
def my_set : set ℝ := {0, real.pi, 1/3, real.sqrt 2, -3}

-- Define irrational numbers using the given set
def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Main theorem: proving the number of irrational numbers in the set is 2
theorem number_of_irrational_in_my_set : 
  (∃ (irrationals : finset ℝ), ∀ x ∈ irrationals, x ∈ my_set ∧ is_irrational x) ∧ 
  (∀ irrationals, (∀ x ∈ irrationals, x ∈ my_set ∧ is_irrational x) → irrationals.card = 2) :=
by
  sorry

end number_of_irrational_in_my_set_l58_58587


namespace polynomial_min_value_l58_58776

-- Define the variables and conditions
variables {P : Polynomial ℤ} {n : ℤ} (hP : ∀ i : ℤ, i ∈ {r | P.eval r = 0} → i ≠ n) 

-- Define the theorem
theorem polynomial_min_value (hP : ∃ (r : Finset ℤ), r.card ≥ 13 ∧ ∀ i ∈ r, P.eval i = 0) : 
  ∃ n : ℤ, P.eval n ≠ 0 → |P.eval n| ≥ 7 * (nat.factorial 6)^2 := 
by {
  sorry
}

end polynomial_min_value_l58_58776


namespace correct_statement_D_l58_58361

theorem correct_statement_D (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
sorry

end correct_statement_D_l58_58361


namespace chess_tournament_no_immediate_loss_l58_58944

theorem chess_tournament_no_immediate_loss (n : ℕ) (lost : Fin n → Fin n → Prop) 
    (symm_lost : ∀ i j, lost i j → ¬ lost j i) : 
    ∃ (p : List (Fin n)), p.Nodup ∧ (∀ i j, p.Nth i = some j → p.Nth (i + 1) = some k → ¬ lost j k) := 
sorry

end chess_tournament_no_immediate_loss_l58_58944


namespace simplify_expression_eq_sqrt3_l58_58273

theorem simplify_expression_eq_sqrt3
  (a : ℝ)
  (h : a = Real.sqrt 3 + 1) :
  ( (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) ) = Real.sqrt 3 := sorry

end simplify_expression_eq_sqrt3_l58_58273


namespace volume_sphere_identification_l58_58321

def is_variable (x : Type) : Prop := sorry
def is_constant (x : Type) : Prop := sorry

theorem volume_sphere_identification (V R : ℝ) (pi : ℝ) :
  (is_variable V ∧ is_variable R) ∧
  (is_constant (4 / 3 : ℝ) ∧ is_constant pi) :=
sorry

end volume_sphere_identification_l58_58321


namespace S2017_eq_l58_58317

noncomputable def a : ℕ → ℤ := sorry

axiom a_seq (n : ℕ) : (a(n + 1) - 1) * (1 - a(n)) = a(n)
axiom a_initial : a(8) = 2

theorem S2017_eq : (∑ i in range 2017, a(i)) = 1008 :=
by
  sorry

end S2017_eq_l58_58317


namespace grid_to_L_shape_l58_58849

-- Given a positive integer n, we need to prove that the remaining part of a 2^n x 2^n piece of paper
-- can be entirely cut along the grid lines into L-shaped pieces after cutting out one small square.
theorem grid_to_L_shape (n : ℕ) (h : 0 < n) : 
  ∃ l_pieces : list (set (ℕ × ℕ)), 
    (∀ l ∈ l_pieces, ∃ a b c : ℕ × ℕ, 
      l = {a, b, c} ∧ ((a.1 = b.1 ∧ b.2 = c.2) ∨ (a.2 = b.2 ∧ b.1 = c.1))
    ) ∧
    ∀ x ∈ { p : ℕ × ℕ | p.1 < 2^n ∧ p.2 < 2^n } \ {one_square}, 
      ∃ l ∈ l_pieces, x ∈ l :=
sorry

end grid_to_L_shape_l58_58849


namespace arithmetic_geometric_inequality_l58_58873

open Real

noncomputable def arithmetic_mean (x : List ℝ) : ℝ := (x.sum) / (x.length)

noncomputable def geometric_mean (x : List ℝ) : ℝ := (x.prod) ^ (1 / (x.length))

theorem arithmetic_geometric_inequality (n : ℕ) (x : List ℝ) 
  (h1 : n ≥ 2) 
  (h2 : x.length = n)
  (h3 : ∀ i j : ℕ, i ≤ j → i < x.length → j < x.length → x.get i ≤ x.get j)
  (h4 : ∀ i : ℕ, i < x.length → x.get i ≥ x.get (i + 1) / (i + 2)) :
  arithmetic_mean x / geometric_mean x ≤ (n + 1) / (2 * (n.factorial) ^ (1 / n)) :=
by
  sorry

end arithmetic_geometric_inequality_l58_58873


namespace determine_a_l58_58125

noncomputable def f : ℝ → ℝ := λ x, 2 ^ x
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, log a x

theorem determine_a :
  (∀ P ∈ (λ P : ℝ × ℝ, P.2 = f P.1), ∃ Q ∈ (λ Q : ℝ × ℝ, Q.2 = g a Q.1),
  (P.1 * Q.1 + P.2 * Q.2 = 0) ∧ (P.1^2 + P.2^2 = Q.1^2 + Q.2^2)) →
  a = (1 / 2) :=
sorry

end determine_a_l58_58125


namespace inequality_max_lambda_value_l58_58614

section Proof1

variable {α β γ : ℂ} (λ : ℝ)

axiom abs_not_all_le_1 (h : ¬(abs α ≤ 1 ∧ abs β ≤ 1 ∧ abs γ ≤ 1)) : Prop

theorem inequality (h : abs_not_all_le_1 (¬(abs α ≤ 1 ∧ abs β ≤ 1 ∧ abs γ ≤ 1))) (hλ : λ ≤ 2/3) :
  1 + abs (α + β + γ) + abs (α * β + β * γ + γ * α) + abs (α * β * γ) ≥ λ * (abs α + abs β + abs γ) := sorry

end Proof1


section Proof2

variable {α β γ : ℂ}

def max_lambda_ineq := (sqrt 2)^(1/3) / 2

theorem max_lambda_value :
  ∀ (α β γ : ℂ), 1 + abs (α + β + γ) + abs (α * β + β * γ + γ * α) + abs (α * β * γ) >= max_lambda_ineq * (abs α + abs β + abs γ) :=
sorry

end Proof2

end inequality_max_lambda_value_l58_58614


namespace agnes_weekly_hours_l58_58622

-- Given conditions
def mila_hourly_rate : ℝ := 10
def agnes_hourly_rate : ℝ := 15
def mila_hours_per_month : ℝ := 48

-- Derived condition that Mila's earnings in a month equal Agnes's in a month
def mila_monthly_earnings : ℝ := mila_hourly_rate * mila_hours_per_month

-- Prove that Agnes must work 8 hours each week to match Mila's monthly earnings
theorem agnes_weekly_hours (A : ℝ) : 
  agnes_hourly_rate * 4 * A = mila_monthly_earnings → A = 8 := 
by
  intro h
  -- sorry here is a placeholder for the proof
  sorry

end agnes_weekly_hours_l58_58622


namespace abs_f_0_eq_34_div_3_l58_58237

-- Define the assumptions
variables {R : Type*} [LinearOrderedField R] -- Ensure we are working within a real-number context
variable (f : R → R) -- Assume f is a function from reals to reals

-- Conditions: f is a third-degree polynomial with real coefficients
axiom third_degree_poly (f : R → R) : ∃ (a b c d : R), ∀ x, f(x) = a*x^3 + b*x^2 + c*x + d

-- Given conditions
axiom abs_f_1_eq_10 : |f 1| = 10
axiom abs_f_2_eq_10 : |f 2| = 10
axiom abs_f_4_eq_10 : |f 4| = 10

-- The goal to prove
theorem abs_f_0_eq_34_div_3 : |f 0| = 34 / 3 := sorry

end abs_f_0_eq_34_div_3_l58_58237


namespace angle_measure_l58_58352

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l58_58352


namespace can_form_triangle_cannot_form_triangle_triangle_formation_cases_l58_58053

theorem can_form_triangle {a b c : ℕ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : True :=
  trivial

theorem cannot_form_triangle {a b c : ℕ} (h: ¬(a + b > c ∧ b + c > a ∧ c + a > b)) : True :=
  trivial

theorem triangle_formation_cases :
  (¬(1 + 2 > 3 ∧ 2 + 3 > 1 ∧ 3 + 1 > 2)) ∧
  (¬(2 + 3 > 5 ∧ 3 + 5 > 2 ∧ 5 + 2 > 3)) ∧
  (¬(3 + 4 > 8 ∧ 4 + 8 > 3 ∧ 8 + 3 > 4)) ∧
  (3 + 4 > 5 ∧ 4 + 5 > 3 ∧ 5 + 3 > 4) :=
  by {
    apply And.intro,
    { apply cannot_form_triangle,
      simp,
    },
    apply And.intro,
    { apply cannot_form_triangle,
      simp,
    },
    apply And.intro,
    { apply cannot_form_triangle,
      simp,
    },
    { apply can_form_triangle,
      repeat {linarith},
    },
  }

end can_form_triangle_cannot_form_triangle_triangle_formation_cases_l58_58053


namespace percentage_decrease_in_area_l58_58048

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 0.70 * L
def new_breadth (B : ℝ) : ℝ := 0.85 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem percentage_decrease_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  ((original_area L B - new_area L B) / original_area L B) * 100 = 40.5 :=
by
  sorry

end percentage_decrease_in_area_l58_58048


namespace mean_variance_shift_l58_58474

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m) ^ 2)).sum / data.length

theorem mean_variance_shift (x : List ℝ) (m : ℝ) (h : m ≠ 0) :
  mean (x.map (λ xi => xi - m)) = mean x - m ∧ variance (x.map (λ xi => xi - m)) = variance x :=
by
  sorry

end mean_variance_shift_l58_58474


namespace sum_of_ages_l58_58253

variables (M A : ℕ)

def Maria_age_relation : Prop :=
  M = A + 8

def future_age_relation : Prop :=
  M + 10 = 3 * (A - 6)

theorem sum_of_ages (h₁ : Maria_age_relation M A) (h₂ : future_age_relation M A) : M + A = 44 :=
by
  sorry

end sum_of_ages_l58_58253


namespace solve_linear_equation_l58_58546

theorem solve_linear_equation (a b x : ℝ) (h : a - b = 0) (ha : a ≠ 0) : ax + b = 0 ↔ x = -1 :=
by sorry

end solve_linear_equation_l58_58546


namespace spadesuit_calculation_l58_58098

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 5 (spadesuit 3 2) = 0 :=
by
  sorry

end spadesuit_calculation_l58_58098


namespace pascal_even_sum_prop_l58_58607

def pascal_even_sum (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ k, k % 2 = 0).sum (λ k, nat.choose n k)

noncomputable def g (n : ℕ) : ℝ :=
  real.logb 2 (pascal_even_sum n)

theorem pascal_even_sum_prop (n : ℕ) (h : 0 < n) (H1 : pascal_even_sum n = 2^(n-1)) :
  g n / real.logb 2 3 = (n - 1) / real.logb 2 3 :=
by 
  have H2 : g n = n - 1, from sorry,
  rw H2,
  simp,
  sorry

end pascal_even_sum_prop_l58_58607


namespace min_value_f_l58_58858

noncomputable def f (x : ℝ) : ℝ := (real.exp x - 1)^2 + (real.exp 1 - x - 1)^2

theorem min_value_f : ∃ x : ℝ, f x = -2 :=
sorry

end min_value_f_l58_58858


namespace curves_intersect_l58_58203

-- Declare points and curve definitions
def C1 (t : ℝ) : ℝ × ℝ := (4 + t, 5 + 2 * t)
def C2 (α : ℝ) : ℝ × ℝ := (3 + 5 * Real.cos α, 5 + 5 * Real.sin α)

-- Equation forms of the curves
def C1_cartesian (x y : ℝ) : Prop := y = 2 * x - 3
def C2_cartesian (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 25

theorem curves_intersect : ∃ x y, C1_cartesian x y ∧ C2_cartesian x y :=
begin
  sorry
end

end curves_intersect_l58_58203


namespace angle_between_faces_l58_58291

noncomputable def volume : ℝ := 75
noncomputable def side_length : ℝ := 10
noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def point_T (M N : ℝ) : ℝ := midpoint M N
noncomputable def inscribed_radius_equality : Prop := true -- treat the equality as a trivially true condition

theorem angle_between_faces (V S r1 r2 : ℝ) (h1 : V = 75) (h2 : S = (10^2 * real.sqrt 3) / 4) (h3 : r1 = r2) 
: ∃ θ, θ = real.arccos (±4/5) :=
by
  -- The proof goes here
  sorry

end angle_between_faces_l58_58291


namespace abs_inequality_range_l58_58175

theorem abs_inequality_range (x : ℝ) (b : ℝ) (h : 0 < b) : (b > 2) ↔ ∃ x : ℝ, |x - 5| + |x - 7| < b :=
sorry

end abs_inequality_range_l58_58175


namespace polynomial_value_at_minus_1_l58_58728

-- Definitions for the problem conditions
def polynomial_1 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x + 1
def polynomial_2 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x - 2

theorem polynomial_value_at_minus_1 :
  ∀ (a b : ℤ), (a + b = 2022) → polynomial_2 a b (-1) = -2024 :=
by
  intro a b h
  sorry

end polynomial_value_at_minus_1_l58_58728


namespace triangle_AB1M_sim_BA1M_l58_58367

variables {A B C A1 B1 M : Type*} [triangle ABC]
variables (ω : circle A1 B1 C)
variables (Ω : circumcircle ABC)
variables (M : point)

-- Assume geometric relations and other necessary properties here
-- Example: The circle ω intersects AC and BC at A1 and B1
-- Example: The circle Ω passes through A, B, C and M

theorem triangle_AB1M_sim_BA1M 
  (hA1: ω.intersect(AC) = A1)
  (hB1: ω.intersect(BC) = B1)
  (hM: Ω.intersect(ω) = M)
  : similar_triangles AB_1M BA_1M :=
begin
  sorry
end

end triangle_AB1M_sim_BA1M_l58_58367


namespace percentage_same_grade_l58_58389

def total_students : ℕ := 30
def students_grade_A : ℕ := 3
def students_grade_B : ℕ := 6
def students_grade_C : ℕ := 2
def students_grade_D : ℕ := 2
def same_grade_students : ℕ := students_grade_A + students_grade_B + students_grade_C + students_grade_D

theorem percentage_same_grade :
  (same_grade_students.to_real / total_students.to_real) * 100 = 43.33 := 
sorry

end percentage_same_grade_l58_58389


namespace number_of_integers_between_sqrt5_and_sqrt50_l58_58534

theorem number_of_integers_between_sqrt5_and_sqrt50 :
  let lower_bound := 3
  let upper_bound := 7
  card (set_of (λ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ int x = x)) = 5 :=
by
  sorry

end number_of_integers_between_sqrt5_and_sqrt50_l58_58534


namespace arc_length_of_YZ_semicircle_l58_58589

variables (XY XZ YZ : ℝ)

-- Conditions extracted from part (a)
def is_right_triangle (X Y Z : Type) (triangle : triangle X Y Z) : Prop :=
  ∃ (rXY : ℝ) (rXZ : ℝ), 
    (π * rXY^2 / 2 = 12.5 * π) ∧ 
    (π * rXZ^2 / 2 = 25 * π) ∧ 
    (XY = 2 * rXY) ∧ 
    (XZ = 2 * rXZ) ∧ 
    (XY^2 + XZ^2 = YZ^2)

-- Theorem to prove
theorem arc_length_of_YZ_semicircle (X Y Z : Type) 
  (triangle : triangle X Y Z)
  (h : is_right_triangle XY XZ YZ) :
  arc_length YZ = 5 * sqrt 3 * π :=
begin
  sorry -- proof to be filled in
end

end arc_length_of_YZ_semicircle_l58_58589


namespace isosceles_triangle_perimeter_l58_58200

theorem isosceles_triangle_perimeter 
  (a b c : ℝ)  (h_iso : a = b ∨ b = c ∨ c = a)
  (h_len1 : a = 4 ∨ b = 4 ∨ c = 4)
  (h_len2 : a = 9 ∨ b = 9 ∨ c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 22 :=
sorry

end isosceles_triangle_perimeter_l58_58200


namespace total_points_of_players_l58_58197

variables (Samanta Mark Eric Daisy Jake : ℕ)
variables (h1 : Samanta = Mark + 8)
variables (h2 : Mark = 3 / 2 * Eric)
variables (h3 : Eric = 6)
variables (h4 : Daisy = 3 / 4 * (Samanta + Mark + Eric))
variables (h5 : Jake = Samanta - Eric)
 
theorem total_points_of_players :
  Samanta + Mark + Eric + Daisy + Jake = 67 :=
sorry

end total_points_of_players_l58_58197


namespace ZBLMS_position_l58_58815

theorem ZBLMS_position : 
  let letters := ['B', 'L', 'M', 'S', 'Z'];
  let permutations := (letters.permutations (list.length letters));
  let sorted_permutations := list.sort permutations 
  in 
  nth sorted_permutations 103 = ['Z','B','L','M','S'] :=
begin
  sorry,
end

end ZBLMS_position_l58_58815


namespace parabola_is_8x_l58_58406

noncomputable def parabola_equation (p : ℝ) : Prop :=
  ∃ (y : ℝ), y^2 = 2 * p * 3 ∧ sqrt ((3 - p / 2)^2 + y^2) = 5

theorem parabola_is_8x : 
  (∃ (p : ℝ), p > 0 ∧ parabola_equation p) →
  (∃ (y : ℝ), y^2 = 8 * 3) :=
begin
  intro h,
  rcases h with ⟨p, hp, heq⟩,
  use sqrt (8 * 3),
  have hp' : p = 4, from sorry,  -- Proof steps would go here
  rw [parabola_equation, hp'],
  sorry  -- Detailed steps skipped
end

end parabola_is_8x_l58_58406


namespace trapezoid_perimeter_l58_58495

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def perimeter (J K L M : Point) : ℝ :=
  distance J K + distance K L + distance L M + distance M J

theorem trapezoid_perimeter 
  (J K L : Point)
  (hJ : J = ⟨-2, -4⟩)
  (hK : K = ⟨-2, 2⟩)
  (hL : L = ⟨6, 8⟩) 
  (h_trapezoid : true) -- This can be refined if we formalize the definition of trapezoid
  : ∃ M : Point, M = ⟨6, -10⟩ ∧ perimeter J K L M = 44 :=
by
  use ⟨6, -10⟩
  sorry

end trapezoid_perimeter_l58_58495


namespace pyramid_apex_angles_l58_58667

theorem pyramid_apex_angles :
  ∀ (S A B C : Point) 
    (h_base : EquilateralTriangle A B C)
    (h_perp_sba : Perpendicular (Plane S A B) (Plane A B C))
    (h_perp_sbc : Perpendicular (Plane S B C) (Plane A B C))
    (h_sum : ∠ASB + ∠CSB = π / 2),
  ∃ α β : ℝ, α = Real.arccos (Real.sqrt 3 - 1) ∧ β = π / 2 - Real.arccos (Real.sqrt 3 - 1) :=
begin
  sorry
end

end pyramid_apex_angles_l58_58667


namespace jenny_house_value_l58_58965

/-- Jenny's property tax rate is 2% -/
def property_tax_rate : ℝ := 0.02

/-- Her house's value increases by 25% due to the new high-speed rail project -/
noncomputable def house_value_increase_rate : ℝ := 0.25

/-- Jenny can afford to spend $15,000/year on property tax -/
def max_affordable_tax : ℝ := 15000

/-- Jenny can make improvements worth $250,000 to her house -/
def improvement_value : ℝ := 250000

/-- Current worth of Jenny's house -/
noncomputable def current_house_worth : ℝ := 500000

theorem jenny_house_value :
  property_tax_rate * (current_house_worth + improvement_value) = max_affordable_tax :=
by
  sorry

end jenny_house_value_l58_58965


namespace minimum_value_m_l58_58867

noncomputable def a_n (n : ℕ) : ℕ :=
  6 * (1 / 3)^(n-1)

noncomputable def S_n (n : ℕ) : ℕ :=
  a_n n

def b : ℕ → ℕ
| 1 := 2
| (n + 1) := b n + 2 * (Nat.log (a_n n / 18) / Nat.log (1 / 3)).natAbs

noncomputable def T (n : ℕ) : ℚ :=
  (Finset.range (n+1)).sum (λ i, 1 / b (i + 1))

theorem minimum_value_m (n : ℕ) : T n < 1 :=
sorry

end minimum_value_m_l58_58867


namespace find_x_pow_8_l58_58547

theorem find_x_pow_8 (x : ℂ) (h : x + x⁻¹ = real.sqrt 2) : x^8 = 1 := 
sorry

end find_x_pow_8_l58_58547


namespace evaluate_product_l58_58831

theorem evaluate_product (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5 * n^3 + 4 * n^2 + 4 * n := 
by
  -- Omitted proof steps
  sorry

end evaluate_product_l58_58831


namespace complex_exp_addition_l58_58644

theorem complex_exp_addition (z w : ℂ) : exp(z) * exp(w) = exp(z + w) := 
by 
sory

end complex_exp_addition_l58_58644


namespace false_statement_about_f_and_g_l58_58523

noncomputable def f : ℝ → ℝ := λ x, Real.exp x
noncomputable def g : ℝ → ℝ := λ x, x + 1

theorem false_statement_about_f_and_g :
  ¬ (∀ x : ℝ, f x > g x) :=
by 
  sorry

end false_statement_about_f_and_g_l58_58523


namespace find_angle_B_find_range_b_l58_58567

-- Define a noncomputable section since we are working with trigonometric functions.
noncomputable section

-- Define the main constants and variables used in the problems.
variables {A B C a b c : ℝ}

-- Triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively.
def triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0

-- Given condition: cos C + (cos A - sqrt(3) * sin A) * cos B = 0
def condition1 (A B C : ℝ) : Prop :=
  cos C + (cos A - (real.sqrt 3) * sin A) * cos B = 0

-- Given condition: a + c = 1
def condition2 (a c : ℝ) : Prop :=
  a + c = 1

-- Theorem 1: Prove that B = π / 3
theorem find_angle_B (A B C a b c : ℝ) (h1: triangle A B C a b c) (h2: condition1 A B C) : 
  B = π / 3 :=
sorry

-- Theorem 2: Prove that 1/2 ≤ b < 1 given B = π / 3
theorem find_range_b (A B C a b c : ℝ) (h1: triangle A B C a b c) (h2: condition1 A B C) (h3: condition2 a c) : 
  1 / 2 ≤ b ∧ b < 1 :=
sorry

end find_angle_B_find_range_b_l58_58567


namespace BD_length_l58_58942

theorem BD_length {A B C D : Point} (h1 : dist A C = 10) (h2 : dist B C = 10) (h3 : dist A B = 4) 
    (h4 : between B A D) (h5 : dist C D = 11) : dist B D = 3 := by sorry

end BD_length_l58_58942


namespace border_area_correct_l58_58781

theorem border_area_correct : 
  ∀ (h w b : ℕ), h = 12 → w = 15 → b = 3 →
  let photo_area := h * w,
      framed_height := h + 2 * b,
      framed_width := w + 2 * b,
      framed_area := framed_height * framed_width,
      border_area := framed_area - photo_area
  in border_area = 198 :=
by
  intros h w b h_eq w_eq b_eq
  let photo_area := h * w
  let framed_height := h + 2 * b
  let framed_width := w + 2 * b
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  sorry

end border_area_correct_l58_58781


namespace sum_of_solutions_of_quadratic_l58_58469

theorem sum_of_solutions_of_quadratic :
  ∑ x in ({x | -24 * x^2 + 72 * x - 120 = 0}.to_finset), x = 3 :=
sorry

end sum_of_solutions_of_quadratic_l58_58469


namespace range_of_f_l58_58394

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 else if x < 0 then -2 else 0

theorem range_of_f :
  is_odd_function f → 
  (∀ x : ℝ, (x > 0 → f x = 2) → (x < 0 → f x = -2) → (x = 0 → f x = 0)) →
  set.range f = { -2, 0, 2 } :=
by
  intro h1 h2
  sorry

end range_of_f_l58_58394


namespace number_of_subsets_of_A_l58_58167

-- Define the set S which contains the elements 1, 2, 3, 4, 5, 6, 7
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the subset of S that contains elements which are either odd or multiples of 3
def A : Finset ℕ := S.filter (λ x => x % 2 = 1 ∨ x % 3 = 0)

-- Prove that the number of subsets of A is 31
theorem number_of_subsets_of_A : (2 ^ A.card) = 32 - 1 := by
  have hA_card : A.card = 5 := by
  sorry
  calc
    2 ^ A.card - 1 = 2 ^ 5 - 1 : by rw [hA_card]
                ... = 31       : by norm_num

end number_of_subsets_of_A_l58_58167


namespace angle_measure_supplement_complement_l58_58345

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l58_58345


namespace maximum_ratio_l58_58244

-- Define the conditions
def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def mean_is_45 (x y : ℕ) : Prop :=
  (x + y) / 2 = 45

-- State the theorem
theorem maximum_ratio (x y : ℕ) (hx : is_two_digit_positive_integer x) (hy : is_two_digit_positive_integer y) (h_mean : mean_is_45 x y) : 
  ∃ (k: ℕ), (x / y = k) ∧ k = 8 :=
sorry

end maximum_ratio_l58_58244


namespace polar_eq_of_circle_length_segment_PQ_l58_58204

-- Define parametric equations of circle C
def circle_parametric (φ : ℝ) : ℝ × ℝ :=
  (1 + cos φ, sin φ)

-- Convert parametric to polar equation
theorem polar_eq_of_circle : 
  (∀ φ : ℝ, ∃ θ : ℝ, circle_parametric φ = (⟨2*cos θ, θ⟩)) := sorry

-- Define polar equation of line l and ray OM
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * sin (θ + π / 3) = 3 * sqrt 3

def ray_OM (θ : ℝ) : Prop :=
  θ = π / 3

-- Polar coordinates of points P and Q
def point_P : ℝ × ℝ :=
  (1, π / 3)

def point_Q : ℝ × ℝ :=
  (3, π / 3)

-- Length of segment PQ
theorem length_segment_PQ :
  abs (point_P.1 - point_Q.1) = 2 := sorry

end polar_eq_of_circle_length_segment_PQ_l58_58204


namespace find_DF_l58_58485

-- Conditions
variables {A B C D E F : Type}
variables {BC EF AC DF : ℝ}
variable (h_similar : similar_triangles A B C D E F)
variable (h_BC : BC = 6)
variable (h_EF : EF = 4)
variable (h_AC : AC = 9)

-- Question: Prove DF = 6 given the above conditions
theorem find_DF : DF = 6 :=
by
  sorry

end find_DF_l58_58485


namespace hike_ratio_l58_58810

variables (C A S : ℕ)

-- Definitions
def camila_hikes : ℕ := 7
def hikes_per_week : ℕ := 4
def weeks : ℕ := 16
def steven_difference : ℕ := 15

-- Conditions
-- Camila plans to have hiked as many times as Steven
def total_camila_hikes : ℕ := camila_hikes + hikes_per_week * weeks
def amanda_hikes : ℕ := S - steven_difference
def ratio : ℕ := amanda_hikes / camila_hikes

-- The proof statement
theorem hike_ratio
  (H1 : C = camila_hikes)
  (H2 : total_camila_hikes = C + hikes_per_week * weeks)
  (H3 : S = total_camila_hikes)
  (H4 : A = S - steven_difference)
  (H5 : ratio = A / C)
  : ratio = 8 := 
sorry

end hike_ratio_l58_58810


namespace sum_digits_500_l58_58449

noncomputable def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_500 (k : ℕ) (h : k = 55) :
  sum_digits (63 * 10^k - 64) = 500 :=
by
  sorry

end sum_digits_500_l58_58449


namespace cubic_identity_l58_58109

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 11) (h3 : abc = -6) : a^3 + b^3 + c^3 = 94 :=
by
  sorry

end cubic_identity_l58_58109


namespace sandwich_count_l58_58059

noncomputable def sandwich_combinations (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

theorem sandwich_count (breads meats cheeses : ℕ) (H_breads : breads = 5) 
                       (H_meats : meats = 7) (H_cheeses : cheeses = 6) 
                       (H_no_turkey_swiss : ℕ) (H_no_rye_salami : ℕ) 
                       (H_no_turkey_swiss = 5) (H_no_rye_salami = 6) :
  sandwich_combinations breads meats cheeses - H_no_turkey_swiss - H_no_rye_salami = 199 :=
by
  sorry

end sandwich_count_l58_58059


namespace initial_number_of_orchids_l58_58703

theorem initial_number_of_orchids 
  (initial_orchids : ℕ)
  (cut_orchids : ℕ)
  (final_orchids : ℕ)
  (h_cut : cut_orchids = 19)
  (h_final : final_orchids = 21) :
  initial_orchids + cut_orchids = final_orchids → initial_orchids = 2 :=
by
  sorry

end initial_number_of_orchids_l58_58703


namespace sequence_formula_Sn_formula_l58_58586

noncomputable def sequence (n: ℕ) : ℤ := -2 * n + 10

def Sn (n: ℕ) : ℤ :=
  if n ≤ 5 then -n^2 + 9 * n
  else n^2 - 9 * n + 40

theorem sequence_formula (n : ℕ) : 
  (∀ n: ℕ, n > 0 → sequence n = -2 * n + 10)
  ∧ (sequence 1 = 8)
  ∧ (sequence 4 = 2)
  ∧ (∀ n: ℕ, sequence (n + 2) - 2 * sequence (n + 1) + sequence n = 0) :=
by {
  sorry
}

theorem Sn_formula (n : ℕ) : 
  ∀ n : ℕ, Sn n = 
  if n ≤ 5 then -n^2 + 9 * n
  else n^2 - 9 * n + 40 :=
by {
  sorry
}

end sequence_formula_Sn_formula_l58_58586


namespace people_pay_taxes_per_week_l58_58749

theorem people_pay_taxes_per_week
  (pct_no_tax : Real) (daily_shoppers : Nat) (days_per_week : Nat) :
  pct_no_tax = 0.06 → daily_shoppers = 1000 → days_per_week = 7 →
  let pct_pay_tax := 1 - pct_no_tax
  let daily_tax_payers := pct_pay_tax * daily_shoppers
  daily_tax_payers * days_per_week = 6580 :=
by
  intros h1 h2 h3
  subst h1 
  subst h2 
  subst h3 
  let pct_pay_tax := 1 - 0.06
  let daily_tax_payers := pct_pay_tax * 1000
  have h4 : daily_tax_payers * 7 = 6580 := sorry
  exact h4.sorry

end people_pay_taxes_per_week_l58_58749


namespace fractional_part_shaded_l58_58416

theorem fractional_part_shaded (a : ℝ) (r : ℝ) (sum : ℝ) 
    (h0 : a = 1 / 4) 
    (h1 : r = 1 / 16) 
    (h2 : sum = a / (1 - r)) : 
    sum = 4 / 15 :=
by
  rw [h0, h1] at h2
  exact h2

end fractional_part_shaded_l58_58416


namespace angle_measure_supplement_complement_l58_58346

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end angle_measure_supplement_complement_l58_58346


namespace hexagon_can_be_divided_into_congruent_triangles_l58_58452

section hexagon_division

-- Definitions
variables {H : Type} -- H represents the type for hexagon

-- Conditions
variables (is_hexagon : H → Prop) -- A predicate stating that a shape is a hexagon
variables (lies_on_grid : H → Prop) -- A predicate stating that the hexagon lies on the grid
variables (can_cut_along_grid_lines : H → Prop) -- A predicate stating that cuts can only be made along the grid lines
variables (identical_figures : Type u → Prop) -- A predicate stating that the obtained figures must be identical
variables (congruent_triangles : Type u → Prop) -- A predicate stating that the obtained figures are congruent triangles
variables (area_division : H → Prop) -- A predicate stating that the area of the hexagon is divided equally

-- Theorem statement
theorem hexagon_can_be_divided_into_congruent_triangles (h : H)
  (H_is_hexagon : is_hexagon h)
  (H_on_grid : lies_on_grid h)
  (H_cut : can_cut_along_grid_lines h) :
  ∃ (F : Type u), identical_figures F ∧ congruent_triangles F ∧ area_division h :=
sorry

end hexagon_division

end hexagon_can_be_divided_into_congruent_triangles_l58_58452


namespace dice_probability_greater_than_eight_l58_58716

open Finset

theorem dice_probability_greater_than_eight :
  (card {pair : ℕ × ℕ | (1 ≤ pair.1 ∧ pair.1 ≤ 6) ∧ (1 ≤ pair.2 ∧ pair.2 ≤ 6) ∧ (pair.1 + pair.2 > 8)} : ℕ) / 
  (card {pair : ℕ × ℕ | (1 ≤ pair.1 ∧ pair.1 ≤ 6) ∧ (1 ≤ pair.2 ∧ pair.2 ≤ 6)} : ℕ) = 5 / 18 :=
by
  sorry

end dice_probability_greater_than_eight_l58_58716


namespace fencing_required_l58_58434

theorem fencing_required :
  ∃ (L W : ℝ), (L * W = 1200) ∧ (W = 15) ∧ (L = 80) ∧ ((2 * L + 2 * W) - (10 + 15) = 165) :=
by
  use [80, 15]
  sorry

end fencing_required_l58_58434


namespace triangle_properties_l58_58568

variable {a b c A B C : ℝ}

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively, and the conditions:
1. √3 (a - c cos B) = b sin C
2. The area of the triangle is √3 / 3
3. a + b = 4
Prove:
1. C = 60 degrees
2. sin A sin B = 1 / 12
3. cos A cos B = 5 / 12
-/
theorem triangle_properties 
  (h1 : √3 * (a - c * Real.cos B) = b * Real.sin C)
  (h2 : 1 / 2 * a * b * Real.sin C = √3 / 3)
  (h3 : a + b = 4) :
  C = 60 ∧
  Real.sin A * Real.sin B = 1 / 12 ∧
  Real.cos A * Real.cos B = 5 / 12 :=
sorry

end triangle_properties_l58_58568


namespace gas_price_this_year_l58_58656

-- Define the given conditions
variable (x : ℝ)
variable (price_increase_rate : ℝ := 1.25)
variable (december_bill : ℝ := 96)
variable (may_bill : ℝ := 90)
variable (consumption_difference : ℝ := 10)

-- Gas consumption in December
def december_consumption : ℝ := december_bill / x

-- Gas consumption in May
def may_consumption : ℝ := december_consumption x - consumption_difference

-- Equation relating consumption and bills
def may_bill_equation (price_new : ℝ) : Prop := 
  (may_consumption x) * price_new = may_bill

-- Define the new price this year
def price_new : ℝ := price_increase_rate * x

-- The theorem to prove
theorem gas_price_this_year : price_new x = 3 := by
  sorry

end gas_price_this_year_l58_58656


namespace triangle_inequality_sin_l58_58791

theorem triangle_inequality_sin (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  2 * (sin A / A) + 2 * (sin B / B) + 2 * (sin C / C) ≤ (1 / B + 1 / C) * sin A + (1 / C + 1 / A) * sin B + (1 / A + 1 / B) * sin C :=
by
  sorry

end triangle_inequality_sin_l58_58791


namespace math_equivalence_proof_l58_58395

-- Define the function f : ℝ → ℝ that is valid on (0, +∞)
variable (f : ℝ → ℝ)

-- Assume the derivative condition
variable (h : ∀ x > 0, (√x) * (deriv f x) < (1/2))

-- The theorem we want to prove
theorem math_equivalence_proof (hf : ∀ x > 0, (√x) * (deriv f x) < (1/2)) :
  (f 9 - 1) < (f 4) ∧ (f 4) < (f 1 + 1) := 
sorry

end math_equivalence_proof_l58_58395


namespace concert_attendance_l58_58991

/-
Mrs. Hilt went to a concert. A total of some people attended the concert. 
The next week, she went to a second concert, which had 119 more people in attendance. 
There were 66018 people at the second concert. 
How many people attended the first concert?
-/

variable (first_concert second_concert : ℕ)

theorem concert_attendance (h1 : second_concert = first_concert + 119)
    (h2 : second_concert = 66018) : first_concert = 65899 := 
by
  sorry

end concert_attendance_l58_58991


namespace avg_difference_even_avg_difference_odd_l58_58004

noncomputable def avg (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

def even_ints_20_to_60 := [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
def even_ints_10_to_140 := [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140]

def odd_ints_21_to_59 := [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
def odd_ints_11_to_139 := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139]

theorem avg_difference_even :
  avg even_ints_20_to_60 - avg even_ints_10_to_140 = -35 := sorry

theorem avg_difference_odd :
  avg odd_ints_21_to_59 - avg odd_ints_11_to_139 = -35 := sorry

end avg_difference_even_avg_difference_odd_l58_58004


namespace range_of_m_l58_58904

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m > 0) → m > 1 :=
by
  -- Proof goes here
  sorry

end range_of_m_l58_58904


namespace perpendicular_bisector_eq_min_y_intercept_of_line_AB_l58_58602

noncomputable def point := (ℝ, ℝ) -- define a point as a tuple of real numbers

def on_ellipse (p : point) : Prop := (p.1^2 / 9) + p.2^2 = 1 -- point p is on the ellipse

def above_x_axis (p : point) : Prop := p.2 > 0 -- point p is above the x-axis

theorem perpendicular_bisector_eq {A B : point} (hA : on_ellipse A) (hB : on_ellipse B)
  (hA_above : above_x_axis A) (hB_above : above_x_axis B)
  (hx₁x₂ : A.1 + B.1 = 2) (hy₁y₂ : A.2 + B.2 = 1) :
  ∃ (M : point), M.1 = 1 ∧ M.2 = 1/2 ∧ 9 * M.1 - 2 * M.2 - 8 = 0 := 
sorry -- perpendicular bisector equation is as stated

theorem min_y_intercept_of_line_AB {A B : point} (hA : on_ellipse A) (hB : on_ellipse B)
  (hA_above : above_x_axis A) (hB_above : above_x_axis B)
  (hx₁x₂ : A.1 + B.1 = 2) (hy₁y₂ : A.2 + B.2 = 1) :
  ∃ (k m : ℝ), k < 0 ∧ m > 0 ∧ m = 2/3 ∧ ∀ y_intercept, y_intercept >= m :=
sorry -- minimum y-intercept of line AB is 2/3

end perpendicular_bisector_eq_min_y_intercept_of_line_AB_l58_58602


namespace min_value_x_y_z_l58_58245

theorem min_value_x_y_z (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 + x + 2 * y + 3 * z = 13 / 4) :
  x + y + z = (-3 + real.sqrt 22) / 2 :=
sorry

end min_value_x_y_z_l58_58245


namespace jane_can_buy_9_tickets_l58_58964

-- Definitions
def ticket_price : ℕ := 15
def jane_amount_initial : ℕ := 160
def scarf_cost : ℕ := 25
def jane_amount_after_scarf : ℕ := jane_amount_initial - scarf_cost
def max_tickets (amount : ℕ) (price : ℕ) := amount / price

-- The main statement
theorem jane_can_buy_9_tickets :
  max_tickets jane_amount_after_scarf ticket_price = 9 :=
by
  -- Proof goes here (proof steps would be outlined)
  sorry

end jane_can_buy_9_tickets_l58_58964


namespace lights_on_now_l58_58769

def initially_all_off : Prop := true -- since this is an initial fact we assume true
def total_lights : ℕ := 100

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0
def is_multiple_of_six (n : ℕ) : Prop := n % 6 = 0

def xiao_ming_pulled (n : ℕ) : Prop := is_even n
def xiao_cong_pulled (n : ℕ) : Prop := is_multiple_of_three n

theorem lights_on_now : initially_all_off → total_lights = 100 →
  (∀ n, 1 ≤ n ∧ n ≤ total_lights → (xiao_ming_pulled n ∨ xiao_cong_pulled n) →
  ((is_even n ∧ ¬is_multiple_of_six n) ∨ (is_multiple_of_three n ∧ ¬is_multiple_of_six n) ∨ (is_multiple_of_six n)) ∧
   (51 = (finset.sum (finset.range total_lights) 
            (λ n, if (xiao_ming_pulled n ∧ ¬is_multiple_of_six n) ∨ 
                    (xiao_cong_pulled n ∧ ¬is_multiple_of_six n) ∨ 
                    (is_multiple_of_six n) then 1 else 0)))) sorry

end lights_on_now_l58_58769


namespace candy_store_profit_l58_58760

def price_of_fudge_per_pound := 2.50
def pounds_of_fudge_sold := 37
def price_of_truffle := 1.50
def number_of_truffles_sold := 82
def price_of_pretzel := 2.00
def number_of_pretzels_sold := 48
def discount_on_fudge := 0.10
def sales_tax := 0.05
def truffles_per_set := 4
def truffles_paid_per_set := 3

-- Define helper functions to compute sales and discounted prices.
def total_sales_fudge : Float := pounds_of_fudge_sold * price_of_fudge_per_pound
def total_discounted_fudge : Float := total_sales_fudge * (1 - discount_on_fudge)

def number_of_truffles_paid := 
  (number_of_truffles_sold / truffles_per_set).floor * truffles_paid_per_set + (number_of_truffles_sold % truffles_per_set) 
def total_sales_truffles : Float := number_of_truffles_paid * price_of_truffle

def total_sales_pretzels : Float := number_of_pretzels_sold * price_of_pretzel

-- Combine all sales and apply tax.
def total_sales_before_tax : Float := total_discounted_fudge + total_sales_truffles + total_sales_pretzels
def total_sales_tax : Float := total_sales_before_tax * sales_tax
def total_money_made : Float := total_sales_before_tax + total_sales_tax

-- The main theorem to prove the total money made.
theorem candy_store_profit :
  total_money_made = 285.86 := sorry

end candy_store_profit_l58_58760


namespace fraction_before_addition_is_correct_l58_58922

-- Define the problem context and the constants for the problem
def capacity : ℝ := 54
def added_gasoline : ℝ := 9
def filled_fraction_after_add : ℝ := 9 / 10

-- Define the fraction x that we want to prove
def initial_fraction : ℝ := 7 / 10

theorem fraction_before_addition_is_correct : 
  (initial_fraction * capacity + added_gasoline = filled_fraction_after_add * capacity) :=
by
  sorry

end fraction_before_addition_is_correct_l58_58922


namespace largest_divisor_of_n_l58_58188

open Nat

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 850 ∣ n^4) (h_primes : ∀ p : ℕ, prime p → 20 < p → ¬ p ∣ n) : 
  ∃ d, d = 10 ∧ d ∣ n :=
by
  sorry

end largest_divisor_of_n_l58_58188


namespace area_ratio_l58_58874

open Real

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 2⟩
def C : Point := ⟨3, 2⟩
def D : Point := ⟨2, 0⟩

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def E : Point := midpoint B D

def F : Point :=
  let DA_length := (D.x - A.x) -- which is 2
  ⟨D.x - 0.5, 0⟩ -- since DF = 0.5 = 1/4 * DA_length which is 0.5

def G : Point := midpoint B C

def area_triangle (P Q R : Point) : ℝ :=
  abs (P.x * Q.y + Q.x * R.y + R.x * P.y 
      - P.y * Q.x - Q.y * R.x - R.y * P.x) / 2

def area_ABG := area_triangle A B G
def area_AFG := area_triangle A F G
def area_quadrilateral_ABFG := area_ABG + area_AFG
def area_DFE := area_triangle D F E

theorem area_ratio : area_DFE / area_quadrilateral_ABFG = 1 / (4 * sqrt 5 + 6) := by
  sorry

end area_ratio_l58_58874


namespace pipe_B_fill_time_l58_58337

def pipe_fill_rate_A : ℝ := 1 / 45
def pipe_empty_rate : ℝ := 1 / 72
def net_fill_rate : ℝ := 1 / 40

def pipe_fill_rate_B (t : ℝ) : ℝ := 1 / t

theorem pipe_B_fill_time (t : ℝ) : pipe_fill_rate_A + pipe_fill_rate_B t - pipe_empty_rate = net_fill_rate → t = 60 := 
by
  intros h
  sorry

end pipe_B_fill_time_l58_58337


namespace find_root_and_m_l58_58878

theorem find_root_and_m (m x₂ : ℝ) (h₁ : (1 : ℝ) * x₂ = 3) (h₂ : (1 : ℝ) + x₂ = -m) : 
  x₂ = 3 ∧ m = -4 :=
by
  sorry

end find_root_and_m_l58_58878


namespace inverse_function_log_l58_58935

noncomputable def f (a x : ℝ) : ℝ := Real.log a x

theorem inverse_function_log : 
  ∃ f : ℝ → ℝ, ∀ (x : ℝ), (0 < x) ∧ (a > 0) ∧ (a ≠ 1) ∧ (f 4 = -2) → 
  f x = Real.log (1 / 2) x := 
begin
  sorry
end

end inverse_function_log_l58_58935


namespace clock_angle_l58_58388

theorem clock_angle : 
  ∀ (n : ℕ), (n = 12) → 
  let central_angle := 360 / n in 
  let angle_3 := 3 * central_angle in 
  let angle_7 := 7 * central_angle in 
  let difference := angle_7 - angle_3 in 
  min difference (360 - difference) = 120 :=
by
  intro n h_n
  let central_angle := 360 / n
  let angle_3 := 3 * central_angle
  let angle_7 := 7 * central_angle
  let difference := angle_7 - angle_3
  have h : min difference (360 - difference) = 120 := sorry
  exact h

end clock_angle_l58_58388


namespace prove_question_eq_answer_l58_58986

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x < 1 then sqrt x else 2 * (x - 1)

theorem prove_question_eq_answer (a : ℝ) (ha : 0 < a) (ha1 : a < 1)
    (h_eq : f a = f (a + 1)) : f (1 / a) = 6 :=
by
  sorry

end prove_question_eq_answer_l58_58986


namespace RobertoOutfitCount_l58_58264

theorem RobertoOutfitCount (T S J H : ℕ) (hT : T = 4) (hS : S = 7) (hJ : J = 5) (hH : H = 2) :
  T * S * J * H = 280 :=
by
  rw [hT, hS, hJ, hH]
  norm_num
  exact eq.refl 280

end RobertoOutfitCount_l58_58264


namespace length_of_OP_l58_58889

theorem length_of_OP (O P : Point) (r : ℝ) (h_radius : r = 3) (h_outside : dist O P > r) : dist O P = 4 :=
by
  sorry

end length_of_OP_l58_58889


namespace ArithmeticSequenceSum_l58_58145

theorem ArithmeticSequenceSum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 + a 2 = 10) 
  (h2 : a 4 = a 3 + 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 3 + a 4 = 18 :=
by
  sorry

end ArithmeticSequenceSum_l58_58145


namespace sum_of_odd_numbers_equiv_l58_58260

theorem sum_of_odd_numbers_equiv (n k : ℕ) (hn : 1 < n) (hk : 1 < k) :
  ∃ a : ℕ, n^k = ∑ i in finset.range(n), (a + 2 * i) ∧ a = (n^(k-1) - n + 1) :=
by sorry

end sum_of_odd_numbers_equiv_l58_58260


namespace max_area_basketball_court_l58_58325

theorem max_area_basketball_court : 
  ∃ l w : ℝ, 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l * w = 10000 :=
by {
  -- We are skipping the proof for now
  sorry
}

end max_area_basketball_court_l58_58325


namespace integer_pair_unique_solution_l58_58916

theorem integer_pair_unique_solution :
  ∃! (m n : ℤ), (m + 2 * n = m * n + 2) := 
begin
  use (0 : ℤ),
  use (0 : ℤ),
  split,
  { -- show that (m, n) = (0, 0) is indeed a solution
    simp,
  },
  { -- show that it's the only solution
    intros x y h,
    have h_eq := h,
    calc
      x + 2 * y = x * y + 2 : by exact h_eq
      ... = x * y + 2 : by sorry,
  },
end

end integer_pair_unique_solution_l58_58916


namespace no_participant_loses_to_next_l58_58947

-- Define participants and the game relation
def Participant := ℕ  -- We use natural numbers to represent participants

-- Games relation (i.e., who lost to whom)
def lost_to (p1 p2 : Participant) : Prop := sorry

-- The main theorem: existence of a desired numbering
theorem no_participant_loses_to_next (n : ℕ) (h : n ≥ 2) : 
  ∃ (f : Fin n → Participant), ∀ i : Fin (n - 1), ¬ lost_to (f i) (f (Fin.succ i)) := sorry

end no_participant_loses_to_next_l58_58947


namespace fixed_point_l58_58466

variable (a : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

def f (x : ℝ) : ℝ := 2 * a^(x + 1) - 3

theorem fixed_point :
  f a (-1) = -1 :=
by
  -- Proof goes here
  sorry

end fixed_point_l58_58466


namespace min_value_of_x2_plus_y2_l58_58508

open Real

theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 - 4 * x + 1 = 0) :
  x^2 + y^2 ≥ 7 - 4 * sqrt 3 := sorry

end min_value_of_x2_plus_y2_l58_58508


namespace painting_houses_l58_58629

theorem painting_houses : 
  let house := Fin 5
  let colors := house -> Bool -- True for red, False for green
  let valid (p : colors) : Prop := ∀ i : house, p i -> ( ∀ j : Fin 5, j = i-1 ∨ j = i+1 -> ¬(p j) )
  (∃ (paintings : Finset colors), paintings.card = 13 ∧ ∀ p ∈ paintings, valid p)

end painting_houses_l58_58629


namespace average_output_l58_58739

theorem average_output (time1 time2 rate1 rate2 cogs1 cogs2 total_cogs total_time: ℝ) :
  rate1 = 20 → cogs1 = 60 → time1 = cogs1 / rate1 →
  rate2 = 60 → cogs2 = 60 → time2 = cogs2 / rate2 →
  total_cogs = cogs1 + cogs2 → total_time = time1 + time2 →
  (total_cogs / total_time = 30) :=
by
  intros hrate1 hcogs1 htime1 hrate2 hcogs2 htime2 htotalcogs htotaltime
  sorry

end average_output_l58_58739


namespace unique_solution_of_quadratic_l58_58277

theorem unique_solution_of_quadratic (b c x : ℝ) (h_eqn : 9 * x^2 + b * x + c = 0) (h_one_solution : ∀ y: ℝ, 9 * y^2 + b * y + c = 0 → y = x) (h_b2_4c : b^2 = 4 * c) : 
  x = -b / 18 := 
by 
  sorry

end unique_solution_of_quadratic_l58_58277


namespace find_lambda_l58_58620

-- Variables and definitions for vectors and real numbers.
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) (λ : ℝ)

-- Given conditions
def cond_1 := (inner_product a b = ∥a∥ * ∥b∥ * real.cos (π / 3))
def cond_2 := (∥a∥ = 2 * ∥b∥)
def cond_3 := (c = (inner_product b a / ∥a∥ ^ 2) • a)

-- Theorem statement to prove λ = 1/4
theorem find_lambda (h1 : cond_1 a b) (h2 : cond_2 a b) (h3 : cond_3 a b c) :
  λ = 1 / 4 :=
sorry

end find_lambda_l58_58620


namespace packed_lunch_needs_l58_58422

-- Definitions based on conditions
def students_A : ℕ := 10
def students_B : ℕ := 15
def students_C : ℕ := 20

def total_students : ℕ := students_A + students_B + students_C

def slices_per_sandwich : ℕ := 4
def sandwiches_per_student : ℕ := 2
def bread_slices_per_student : ℕ := sandwiches_per_student * slices_per_sandwich
def total_bread_slices : ℕ := total_students * bread_slices_per_student

def bags_of_chips_per_student : ℕ := 1
def total_bags_of_chips : ℕ := total_students * bags_of_chips_per_student

def apples_per_student : ℕ := 3
def total_apples : ℕ := total_students * apples_per_student

def granola_bars_per_student : ℕ := 1
def total_granola_bars : ℕ := total_students * granola_bars_per_student

-- Proof goals
theorem packed_lunch_needs :
  total_bread_slices = 360 ∧
  total_bags_of_chips = 45 ∧
  total_apples = 135 ∧
  total_granola_bars = 45 :=
by
  sorry

end packed_lunch_needs_l58_58422


namespace find_line_equation_l58_58111

variable (x y k : ℝ)
variable (P A B : ℝ × ℝ)

-- Defining the fixed points
def A : ℝ × ℝ := (-Real.sqrt 2, 0)
def B : ℝ × ℝ := (Real.sqrt 2, 0)

-- Condition on the product of slopes
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := (Prod.snd P) / (Prod.fst P + Real.sqrt 2)
  let k_PB := (Prod.snd P) / (Prod.fst P - Real.sqrt 2)
  k_PA * k_PB = -1 / 2

-- Equation of trajectory
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Definition of line l and its intersection property
def line (k : ℝ) : (ℝ × ℝ) -> Prop :=
  fun P => Prod.snd P = k * (Prod.fst P) + 1

-- Given condition |MN|
axiom MN_length (k : ℝ) : ℝ
axiom MN_length_condition : (k ≠ 0) -> MN_length k = 4 * Real.sqrt 2 / 3

-- Final theorem statement
theorem find_line_equation (k : ℝ) :
  let l := line k in
  (∀ P, slope_condition P -> trajectory (Prod.fst P) (Prod.snd P)) ->
  MN_length_condition k ->
  l = λ P, (Prod.fst P - Prod.snd P = 1) ∨ (Prod.fst P + Prod.snd P = 1) :=
sorry

end find_line_equation_l58_58111


namespace angle_A_of_triangle_area_and_dot_product_l58_58516

theorem angle_A_of_triangle_area_and_dot_product 
  (A B C : ℝ)
  (h_area : (√3) / 2 = abs (1/2 * B * C * real.sin A))
  (h_dot : -3 = B * C * real.cos A)
  (h_interval : 0 < A ∧ A < real.pi):
  A = 5 * real.pi / 6 :=
by
  sorry

end angle_A_of_triangle_area_and_dot_product_l58_58516


namespace collatz_sum_path_length_10_l58_58492

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def path_length (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1 + path_length (collatz n)

theorem collatz_sum_path_length_10 (S : ℕ) :
  (S = ∑ n in Finset.filter (λ n => path_length n = 10) (Finset.range 10000), id n) → S = 1604 :=
sorry

end collatz_sum_path_length_10_l58_58492


namespace sum_of_x_and_y_l58_58505

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l58_58505


namespace identical_functions_pair_4_l58_58814

def f4 (x : ℝ) : ℝ := sqrt ((x^2 - 1) / (x + 2))
def g4 (x : ℝ) : ℝ := sqrt (x^2 - 1) / sqrt (x + 2)

theorem identical_functions_pair_4 :
  ∀ x : ℝ, x ≠ -2 → (f4 x = g4 x) :=
by sorry

end identical_functions_pair_4_l58_58814


namespace mice_needed_l58_58010

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l58_58010


namespace parametric_area_l58_58438

noncomputable def x (t : ℝ) : ℝ := 3 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 3 * (1 - Real.cos t)

theorem parametric_area : 
  ∫ t in (π/2)..(3*π/2), y t * (deriv x t) = 9 * π + 18 :=
by
  -- Proof steps
  sorry

end parametric_area_l58_58438


namespace cost_per_slice_in_cents_l58_58326

def loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def total_payment : ℕ := 2 * 20
def change : ℕ := 16
def total_cost : ℕ := total_payment - change
def total_slices : ℕ := loaves * slices_per_loaf

theorem cost_per_slice_in_cents :
  (total_cost : ℕ) * 100 / total_slices = 40 :=
by
  sorry

end cost_per_slice_in_cents_l58_58326


namespace prime_divides_sum_implies_congruent_l58_58226

theorem prime_divides_sum_implies_congruent n (p : ℕ) (hp : p > 2) (hdiv : p ∣ int ((2 - real.sqrt 3) ^ n + (2 + real.sqrt 3) ^ n)) : p ≡ 1 [MOD 3] :=
by
  sorry

end prime_divides_sum_implies_congruent_l58_58226


namespace a_value_monotonicity_f_l58_58518

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - (a + 1) / x + a

-- Define function is odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), f (-x) = -f (x)

-- Function f is odd
axiom odd_f : is_odd (λ x, f x a)

-- Solution part (1): The value of a
theorem a_value : a = 0 :=
by
  -- Omitted proof
  sorry

-- Define monotonicity of a function
def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Solution part (2): Monotonicity on (0, ∞)
theorem monotonicity_f (a : ℝ) (I : set ℝ) (hI : I = {x : ℝ | x > 0}) :
  (a = 0) → is_increasing (λ x, f x a) I :=
by
  assume ha : a = 0
  -- Omitted proof
  sorry

end a_value_monotonicity_f_l58_58518


namespace percentage_of_50_of_125_l58_58748

theorem percentage_of_50_of_125 : (50 / 125) * 100 = 40 :=
by
  sorry

end percentage_of_50_of_125_l58_58748


namespace lambda_mu_eq_one_l58_58532

variables (λ μ : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def lambda_a_plus_b : ℝ × ℝ := (λ + 1, λ - 1)
noncomputable def mu_a_minus_b : ℝ × ℝ := (μ - 1, μ + 1)

theorem lambda_mu_eq_one (h : is_perpendicular (lambda_a_plus_b λ) (mu_a_minus_b μ)) : λ * μ = 1 :=
by
    sorry

end lambda_mu_eq_one_l58_58532


namespace probability_of_selecting_rational_l58_58479

theorem probability_of_selecting_rational :
  let numbers := {0, (4 / 3), real.sqrt 2, (-7 : ℚ), ((real.sqrt 5 - 1) / 2)}
  rational_set_card := set.count (λ x, x ∈ numbers ∧ is_rational x) >= 3
  total_numbers := set.count (λ x, x ∈ numbers)
  rational_set_card = 3
  total_numbers = 5
in rational_set_card / total_numbers = (3 / 5) := by
  sorry

end probability_of_selecting_rational_l58_58479


namespace AdamSimon_distance_apart_l58_58792

noncomputable def AdamSimon_time (dist : ℝ) (v_Adam v_Simon : ℝ) :=
  dist / (Real.sqrt (v_Adam ^ 2 + v_Simon ^ 2))

theorem AdamSimon_distance_apart
  (dist : ℝ) (v_Adam v_Simon : ℝ) (target_dist : ℝ)
  (h_Adam : v_Adam = 10) (h_Simon : v_Simon = 15) (h_target : target_dist = 150) :
  AdamSimon_time target_dist v_Adam v_Simon = 150 / Real.sqrt (10 ^ 2 + 15 ^ 2) :=
by
  rw [h_Adam, h_Simon, h_target]
  dsimp [AdamSimon_time]
  sorry

end AdamSimon_distance_apart_l58_58792


namespace parametric_to_cartesian_plane_l58_58405

-- Definition of the parametric plane
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 1 + s, 4 - s + t)

-- Definition of the equation form for the plane
def plane_equation (x y z A B C D : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

-- Theorem to be proved
theorem parametric_to_cartesian_plane :
  ∃ A B C D : ℤ, A > 0 ∧
  (∀ s t : ℝ, 
    let ⟨x, y, z⟩ := plane_parametric s t in
    plane_equation x y z A B C D) ∧ 
  int.gcd A.natAbs (int.gcd B.natAbs (int.gcd C.natAbs D.natAbs)) = 1 ∧
  (A = 1 ∧ B = 4 ∧ C = 3 ∧ D = -18) :=
begin
  sorry
end

end parametric_to_cartesian_plane_l58_58405


namespace tangents_intersect_at_common_point_l58_58521

noncomputable def tangent_point_intersection (p q : ℝ → ℝ) (x : ℝ) : ℝ × ℝ :=
  (x + 1 / p x, q x / p x)

theorem tangents_intersect_at_common_point (p q : ℝ → ℝ) :
  ∀ Cα : ℝ → ℝ,
  ∀ (x0 : ℝ) (y0 : ℝ),
    (Cα x0 = y0) →
    (∃! S : ℝ × ℝ, 
      ∀ x (y : ℝ), 
        (Cα x = y) → 
        y' + p x * y = q x → 
        let T := λ ξ η, η = y + (q x - p x * y) * (ξ - x)
        in T S.1 S.2) :=
begin
  sorry
end

end tangents_intersect_at_common_point_l58_58521


namespace value_of_M_plus_N_l58_58073

def my_oplus (a b : ℚ) : ℚ := a^(1/2) + b^(1/3)
def my_otimes (a b : ℚ) : ℚ := log (a^2 : ℝ) - log (b^(1/2) : ℝ)

noncomputable def M : ℚ := my_oplus (9/4) (8/125)
noncomputable def N : ℚ := my_otimes (real.sqrt 2) (1/25)

theorem value_of_M_plus_N : M + N = 5 := by
  sorry

end value_of_M_plus_N_l58_58073


namespace aluminium_copper_adjacent_l58_58662
-- Import the necessary library for the problem

-- Define the conditions mentioned in the problem
def condition (n k : ℕ) : Prop :=
  1 ≤ k ∧ k ≤ 2*n ∧ n ≤ k ∧ k ≤ (3*n+1)/2

-- Statement of the problem
theorem aluminium_copper_adjacent (n k : ℕ) (h : condition n k) :
  ∀ initial_configuration : list (fin 2), 
  ∃ m, ∃ time_steps : ℕ, 
  ∀ t ≥ time_steps,
    ∀i, |initial_configuration.nth (m + t).mod length initial_configuration| = 1 ∧ (i > 0 ∧ i < length initial_configuration - 1 ∧ initial_configuration.nth i ≠ initial_configuration.nth (i + 1)) = false :=
sorry

end aluminium_copper_adjacent_l58_58662


namespace monotonic_intervals_a_eq_2_minimum_integer_a_for_positivity_l58_58448

noncomputable def f (a : ℝ) (x : ℝ) := -x^2 + a*x + 2*(x^2 - x)*(Real.log x)

-- Prove (I)
theorem monotonic_intervals_a_eq_2 :
  let f := f 2 in
  (∀ x ∈ set.Icc (0 : ℝ) (1 / 2), deriv f x > 0) ∧
  (∀ x ∈ set.Icc (1 / 2) 1, deriv f x < 0) ∧
  (∀ x ∈ set.Ioo 1 (Real.to_nnreal inf), deriv f x > 0) :=
sorry

-- Prove (II)
theorem minimum_integer_a_for_positivity :
  (∀ x ∈ set.Ioi (0 : ℝ), f 1 x + x^2 > 0) :=
sorry

end monotonic_intervals_a_eq_2_minimum_integer_a_for_positivity_l58_58448


namespace geometric_series_sum_l58_58808

theorem geometric_series_sum (a r : ℕ) (n : ℕ) (h_a : a = 3) (h_r : r = 2) (h_n : n = 6) :
  a * (r ^ n - 1) / (r - 1) = 189 :=
by
  have calc1 : r ^ n = 64 := sorry
  have calc2 : a * (64 - 1) = 189 := sorry
  exact calc2

end geometric_series_sum_l58_58808


namespace A_inter_B_l58_58616

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.exp x}

theorem A_inter_B : A ∩ B = {1} :=
by
  sorry

end A_inter_B_l58_58616


namespace external_tangent_length_l58_58669

theorem external_tangent_length
  (A B : ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 10) 
  (h2 : r2 = 8) 
  (h3 : A = 0) 
  (h4 : B = 50) :
  length_external_tangent r1 r2 A B = sqrt 2496 :=
by
  sorry

end external_tangent_length_l58_58669


namespace merchant_gross_profit_l58_58366

variable (S : ℝ) (purchase_price : ℝ := 48) (markup_rate : ℝ := 0.40) (discount_rate : ℝ := 0.20)

def selling_price := purchase_price / (1 - markup_rate)

def discounted_selling_price := selling_price - (discount_rate * selling_price)

def gross_profit := discounted_selling_price - purchase_price

theorem merchant_gross_profit : 
  gross_profit = 16 := by
  sorry

end merchant_gross_profit_l58_58366


namespace moscow_inequality_l58_58013

theorem moscow_inequality 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_pos : ∀ i, 1 ≤ i ≤ 1987 → 0 < b i) : 
  (∑ i in finset.range 1987, a i) ^ 2 / (∑ i in finset.range 1987, b i) 
  ≤ ∑ i in finset.range 1987, (a i) ^ 2 / (b i) :=
sorry

end moscow_inequality_l58_58013


namespace max_possible_value_of_S_l58_58122

variable (n : ℕ) (a b : ℕ → ℝ)

noncomputable def S (n : ℕ) (a b : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range n, Real.sqrt (a i) + 
  Real.sqrt (b 0 + ∑ i in Finset.range n, (a i - a ((i + 1) % n))^2 / (2 * n))

theorem max_possible_value_of_S (n : ℕ) (a b : ℕ → ℝ) (h_n : n ≥ 2) (h_b_nonneg : 0 ≤ b 0)
  (h_a_nonneg : ∀ i, 0 ≤ a i) (h_sum_eq : b 0 + ∑ i in Finset.range n, a i = 1) :
  S n a b ≤ if n = 2 then 3 * Real.sqrt 6 / 4 else Real.sqrt (n + 1) := by
  sorry

end max_possible_value_of_S_l58_58122


namespace harriet_trip_time_to_B_l58_58743

variables (D : ℝ) (t1 t2 : ℝ)

-- Definitions based on the given problem
def speed_to_b_town := 100
def speed_to_a_ville := 150
def total_time := 5

-- The condition for the total time for the trip
def total_trip_time_eq := t1 / speed_to_b_town + t2 / speed_to_a_ville = total_time

-- Prove that the time Harriet took to drive from A-ville to B-town is 3 hours.
theorem harriet_trip_time_to_B (h : total_trip_time_eq D D) : t1 = 3 :=
sorry

end harriet_trip_time_to_B_l58_58743


namespace optionC_has_no_real_roots_l58_58734

-- Definitions for the quadratic equations
def quadraticA (x : ℝ) := x^2 - 2
def quadraticB (x : ℝ) := x^2 - 2x
def quadraticC (x : ℝ) := x^2 + x + 1
def quadraticD (x : ℝ) := (x - 1) * (x - 3)

-- Discriminant of a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Conditions for the discriminants of each quadratic
def discA := discriminant 1 0 (-2)
def discB := discriminant 1 (-2) 0
def discC := discriminant 1 1 1
def discD := -- Calculated directly as solving (x-1)(x-3), so no need for discriminant here

-- The proof statement that needs to be proved
theorem optionC_has_no_real_roots : discC < 0 := 
by
  -- Proof here
  sorry

end optionC_has_no_real_roots_l58_58734


namespace max_min_PA_l58_58279

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 9 = 1

-- Define the parametric form of curve C
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 3 * Real.sin θ)

-- Define the line l in parametric form
def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + t, 2 - 2 * t)

-- General equation of the line l
def general_line_l (x y : ℝ) : Prop :=
  2 * x + y = 6

-- Distance between point P and line l
def distance_P_to_l (θ: ℝ) : ℝ :=
  (Real.sqrt 5 / 5) * abs (4 * Real.cos θ + 3 * Real.sin θ - 6)

-- Length PA considering angle between PA and l
def PA_length (θ : ℝ) : ℝ :=
  (2 * Real.sqrt 5 / 5) * abs (5 * Real.sin (θ + (π / 6)) - 6)

theorem max_min_PA :
  ∃ θ α : ℝ, 
  PA_length θ = 2 * Real.sqrt 5 / 5 :=
by sorry

end max_min_PA_l58_58279


namespace breadth_of_rectangular_plot_l58_58003

variable (A b l : ℝ)

theorem breadth_of_rectangular_plot :
  (A = 15 * b) ∧ (l = b + 10) ∧ (A = l * b) → b = 5 :=
by
  intro h
  sorry

end breadth_of_rectangular_plot_l58_58003


namespace number_of_diff_prime_numbers_in_S_up_to_1000_l58_58537

-- Definition for checking prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Definition for the set S
def S (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k + 7

-- Definition for the condition that a number is the difference of two primes
def diff_of_two_primes (n : ℕ) : Prop := ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ x - y = n

-- Goal: There are 70 numbers in the set S up to 1000 that can be expressed as the difference of two primes
theorem number_of_diff_prime_numbers_in_S_up_to_1000 : 
  let count := (finset.filter (λ n, diff_of_two_primes n) 
                (finset.filter (λ n, S n) 
                (finset.range 1001))).to_list.length in
  count = 70 :=
by sorry

end number_of_diff_prime_numbers_in_S_up_to_1000_l58_58537


namespace melted_ice_cream_depth_l58_58414

theorem melted_ice_cream_depth
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ) (V_cylinder : ℝ)
  (h : ℝ)
  (hr_sphere : r_sphere = 3)
  (hr_cylinder : r_cylinder = 10)
  (hV_sphere : V_sphere = 4 / 3 * Real.pi * r_sphere^3)
  (hV_cylinder : V_cylinder = Real.pi * r_cylinder^2 * h)
  (volume_conservation : V_sphere = V_cylinder) :
  h = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l58_58414


namespace target_runs_l58_58957

def run_rate_first_10_overs := 2.1
def overs_first := 10
def runs_first_10_overs : ℝ := overs_first * run_rate_first_10_overs

def run_rate_remaining_30_overs := 8.7
def overs_remaining := 30
def runs_remaining_30_overs : ℝ := overs_remaining * run_rate_remaining_30_overs

def total_runs := runs_first_10_overs + runs_remaining_30_overs

theorem target_runs (run_rate_first_10_overs : ℝ) (overs_first : ℝ) (run_rate_remaining_30_overs : ℝ) (overs_remaining : ℝ)
    (runs_first_10_overs : ℝ := overs_first * run_rate_first_10_overs)
    (runs_remaining_30_overs : ℝ := overs_remaining * run_rate_remaining_30_overs)
    (total_runs : ℝ := runs_first_10_overs + runs_remaining_30_overs) :
    total_runs = 282 :=
by
    sorry  -- proof to be filled in

end target_runs_l58_58957


namespace exists_pair_in_six_cascades_exists_coloring_function_l58_58821

-- Define the notion of a cascade
def is_cascade (r : ℕ) (s : set ℕ) : Prop :=
  s = {n | ∃ k : ℕ, k ∈ set.Icc 1 12 ∧ n = k * r}

-- Part (a): Prove that there exist numbers a and b that belong to six different cascades.
theorem exists_pair_in_six_cascades :
  ∃ (a b : ℕ), a ≠ b ∧ ∃ (r1 r2 r3 r4 r5 r6 : ℕ),
  (is_cascade r1 {a, b} ∧ is_cascade r2 {a, b} ∧ is_cascade r3 {a, b} ∧
   is_cascade r4 {a, b} ∧ is_cascade r5 {a, b} ∧ is_cascade r6 {a, b}) := sorry

-- Part (b): Prove that there exists a coloring function such that all cascades have different colors.
theorem exists_coloring_function :
  ∃ (f : ℕ → ℕ), (∀ r : ℕ, set.pairwise (λ x y, f x ≠ f y) {n | ∃ k : ℕ, k ∈ set.Icc 1 12 ∧ n = k * r}) := sorry

end exists_pair_in_six_cascades_exists_coloring_function_l58_58821


namespace percentage_increase_l58_58177

variable (S : ℝ) (P : ℝ)
variable (h1 : S + 0.10 * S = 330)
variable (h2 : S + P * S = 324)

theorem percentage_increase : P = 0.08 := sorry

end percentage_increase_l58_58177


namespace tan_45_plus_alpha_l58_58511

noncomputable def angle_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, α = (2 * k + 1) * π + β ∧ 0 < β ∧ β < π

theorem tan_45_plus_alpha 
  (α : Real)
  (h1 : sin α = -3 / 5)
  (h2 : angle_in_third_quadrant α) :
  tan (π / 4 + α) = 7 := 
sorry

end tan_45_plus_alpha_l58_58511


namespace cross_section_area_example_l58_58690

-- Define the regular quadrilateral pyramid
structure Pyramid :=
  (base_side : ℝ)
  (lateral_angle : ℝ)

-- Example pyramid instance according to the given problem
def ABCDP := Pyramid.mk (4 * Real.sqrt 2) (120 * Real.pi / 180)

-- Define a condition where a plane passes through diagonal BD and parallel to CP
structure CrossSection :=
  (pyramid : Pyramid)
  (diagonal_BD : ℝ)
  (parallel_plane_length : ℝ)

def cross_section_example : CrossSection :=
  CrossSection.mk ABCDP 8 (2 * Real.sqrt 6)

-- Define the target theorem as verifying the area of the cross-section
theorem cross_section_area_example (cs : CrossSection) : 
  cs.parallel_plane_length = 2 * Real.sqrt 6 → 
  cs.diagonal_BD = 8 → 
  let area := 1 / 2 * cs.diagonal_BD * (cs.parallel_plane_length / 2) in
  (area = 4 * Real.sqrt 6) :=
by
  intros h1 h2
  have h_area_calculation : (1 / 2 * cs.diagonal_BD * (cs.parallel_plane_length / 2)) = 4 * Real.sqrt 6 :=
    by sorry
  exact h_area_calculation

end cross_section_area_example_l58_58690


namespace problem1_problem2_l58_58017
open Nat

theorem problem1 (n : ℕ) (a : Fin n → ℤ) 
  (h_prod : (∏ i, a i) = n) 
  (h_sum : (∑ i, a i) = 0) : 4 ∣ n :=
sorry

theorem problem2 (n : ℕ) 
  (h_div4 : 4 ∣ n) : 
  ∃ (a : Fin n → ℤ), (∏ i, a i) = n ∧ (∑ i, a i) = 0 :=
sorry

end problem1_problem2_l58_58017


namespace find_f_log_2_9_l58_58148

noncomputable def f : ℝ → ℝ 
| x := if x < 1 then 2^x else f (x-1)

theorem find_f_log_2_9 : f (Real.log 9 / Real.log 2) = 9 / 8 := by
  sorry

end find_f_log_2_9_l58_58148


namespace tan_addition_sin_cos_expression_l58_58484

noncomputable def alpha : ℝ := sorry -- this is where alpha would be defined

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem tan_addition (alpha : ℝ) (h : Real.tan alpha = 2) : (Real.tan (alpha + Real.pi / 4) = -3) :=
by sorry

theorem sin_cos_expression (alpha : ℝ) (h : Real.tan alpha = 2) : 
  (Real.sin (2 * alpha) / (Real.sin (alpha) ^ 2 - Real.cos (2 * alpha) + 1) = 1 / 3) :=
by sorry

end tan_addition_sin_cos_expression_l58_58484


namespace cauchy_solution_l58_58459

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end cauchy_solution_l58_58459


namespace people_at_first_concert_l58_58989

def number_of_people_second_concert : ℕ := 66018
def additional_people_second_concert : ℕ := 119

theorem people_at_first_concert :
  number_of_people_second_concert - additional_people_second_concert = 65899 := by
  sorry

end people_at_first_concert_l58_58989


namespace hundredth_digit_of_fraction_l58_58341

theorem hundredth_digit_of_fraction :
  ∃ n : ℕ, 
  n = 1944 ∧ 
  (∀ k : ℕ, k ≥ 100 → (decimal_periodic_function (864 / n) k) = 4) := 
begin
  sorry
end

noncomputable def decimal_periodic_function (x : ℝ) (k : ℕ) : ℕ :=
  sorry

end hundredth_digit_of_fraction_l58_58341


namespace sequence_sum_impossible_l58_58785

theorem sequence_sum_impossible (a : ℕ → ℕ) :
  (∀ i < 199, a (i+1) = 9 * a i ∨ a (i+1) = a i / 2) →
  (a (199+1) = a 200) →
  ¬(finset.univ.sum (λ i, a i) = 24^2022) :=
by 
  intros h1 h2
  sorry

end sequence_sum_impossible_l58_58785


namespace max_n_value_l58_58089

def max_n := 505

theorem max_n_value : ∃ (A B : ℤ), A * B = 72 ∧ 7 * B + A = max_n :=
by
  use 1
  use 72
  simp
  split
  exact rfl
  exact rfl

end max_n_value_l58_58089


namespace convex_hull_perimeter_decrease_outer_polygon_perimeter_ge_inner_l58_58369

-- Definitions for part (a)
structure Polygon := 
  (vertices : List Point)
  (non_convex : Bool) -- true if polygon is non-convex

def convex_hull (p : Polygon) : Polygon := sorry

def perimeter (p : Polygon) : ℝ := sorry

theorem convex_hull_perimeter_decrease (p : Polygon) (h : p.non_convex) : 
  perimeter (convex_hull p) ≤ perimeter p := 
sorry

-- Definitions for part (b)
structure ConvexPolygon extends Polygon

theorem outer_polygon_perimeter_ge_inner (p_in p_out : ConvexPolygon) (h : ∀ v, v ∈ p_in.vertices → v ∈ p_out.vertices) : 
  perimeter p_out ≥ perimeter p_in := 
sorry

end convex_hull_perimeter_decrease_outer_polygon_perimeter_ge_inner_l58_58369


namespace problem_1_l58_58746

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |2 * x + 1| + |x - 2| ≥ a ^ 2 - a + (1 / 2)) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_l58_58746


namespace axis_of_symmetry_shifted_sine_l58_58936

theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f := λ x : ℝ, 2 * sin (2 * x)
  let g := λ x : ℝ, 2 * sin (2 * (x + π / 12))
  ∃ x : ℝ, x = (k * π) / 2 + π / 6 :=
sorry

end axis_of_symmetry_shifted_sine_l58_58936


namespace number_of_bricks_needed_l58_58533

-- Define the dimensions of the brick and the wall
def brick_length : ℝ := 25
def brick_width : ℝ := 11
def brick_height : ℝ := 6

def wall_length : ℝ := 200
def wall_height : ℝ := 300
def wall_thickness : ℝ := 2

-- Calculate the volume of the brick and the wall
def volume_of_brick : ℝ := brick_length * brick_width * brick_height
def volume_of_wall : ℝ := wall_length * wall_height * wall_thickness

-- Calculate the number of bricks needed
def number_of_bricks : ℝ := ⌈volume_of_wall / volume_of_brick⌉

-- State the theorem
theorem number_of_bricks_needed : number_of_bricks = 73 :=
by
  -- Proof goes here
  sorry

end number_of_bricks_needed_l58_58533


namespace find_x8_l58_58549

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end find_x8_l58_58549


namespace ellipse_equation_AN_BM_constant_l58_58146

-- Define the conditions given for the first problem (eccentricity and area of the triangle)
variables {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b)
variables {e : ℝ} (he : e = sqrt 3 / 2)
variables {S : ℝ} (hS : 1 / 2 * a * b = 1)
noncomputable def c := sqrt (a^2 - b^2)
-- Define the centroid condition based on the solution
variables (h_ellipse : (a^2 - b^2) = (sqrt (3))^2)
-- Define the resulting ellipse equation
theorem ellipse_equation : (∃ a b : ℝ, a > b > 0 ∧ 1 / 2 * a * b = 1 ∧ sqrt(a^2 - b^2) / a = sqrt 3 / 2) 
  → ellipse_equation : (dfrac{x^2}{4} + y^2 = 1)

variables {x_0 y_0 : ℝ}
variables {P : ℝ × ℝ} (hP : dfrac{x_0^2}{4} + y_0^2 = 1)
noncomputable def ellipse_condition := (x_0, y_0)

theorem AN_BM_constant :
  (P ∈ ellipse_condition) →
  (|2 + x_0 / (y_0 - 1)| * |1 + 2 * y_0 / (x_0 - 2)| = 4) :=
sorry

end ellipse_equation_AN_BM_constant_l58_58146


namespace degree_of_p_is_unbounded_l58_58096

theorem degree_of_p_is_unbounded (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval (x^2 - 1) = (p.eval x) * (p.eval (-x))) : False :=
sorry

end degree_of_p_is_unbounded_l58_58096


namespace program_run_time_l58_58672

theorem program_run_time
  (os_overhead : ℝ := 1.07)
  (cost_per_millisecond : ℝ := 0.023)
  (tape_mount_cost : ℝ := 5.35)
  (total_cost : ℝ := 40.92) :
  let t := (total_cost - os_overhead - tape_mount_cost) / cost_per_millisecond / 1000
  in t = 1.5 := 
by
  sorry

end program_run_time_l58_58672


namespace ratio_SI_CI_l58_58692

-- Defining parameters and conditions.
def P_SI : ℝ := 1750  -- Principal for Simple Interest
def R_SI : ℝ := 8    -- Rate for Simple Interest
def T_SI : ℝ := 3    -- Time for Simple Interest

def P_CI : ℝ := 4000  -- Principal for Compound Interest
def R_CI : ℝ := 10    -- Rate for Compound Interest
def T_CI : ℝ := 2     -- Time for Compound Interest

-- Defining Simple Interest formula calculation.
def SI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

-- Defining Compound Interest formula calculation.
def CI (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

-- Using the given problem definitions in Lean functions.
theorem ratio_SI_CI :
  let simple_interest := SI P_SI R_SI T_SI,
      compound_interest := CI P_CI R_CI T_CI
  in simple_interest / compound_interest = 1 / 2 := sorry

end ratio_SI_CI_l58_58692


namespace parallelepiped_volume_l58_58891

variables {u v w : ℝ^3}
variables (volUVW : abs (u.dot (v.cross w)) = 6)

theorem parallelepiped_volume :
  abs ((2 • u - v).dot ((v + 4 • w).cross (w + 5 • u))) = 12 := by
sorry

end parallelepiped_volume_l58_58891


namespace pirate_uses_five_chests_l58_58057

def pirate_treasure : Prop :=
  ∀ (gold silver total_coins_per_chest : ℕ),
  gold = 3500 →
  silver = 500 →
  total_coins_per_chest = 1000 →
  let bronze := 2 * silver in
  let total_coins := gold + silver + bronze in
  total_coins / total_coins_per_chest = 5

theorem pirate_uses_five_chests : pirate_treasure :=
by
  intros gold silver total_coins_per_chest
  intros gold_eq silver_eq total_coins_per_chest_eq
  simp only
  sorry

end pirate_uses_five_chests_l58_58057


namespace square_area_is_nine_l58_58994

open Real

-- Define the points in Cartesian coordinate plane
noncomputable def point1 := (1 : ℝ, 2 : ℝ)
noncomputable def point2 := (-2 : ℝ, 2 : ℝ)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the side length as the distance between two points
noncomputable def side_length : ℝ := distance point1 point2

-- Define the area of the square
noncomputable def area_square (side : ℝ) : ℝ := side ^ 2

-- Prove that the area of the square
theorem square_area_is_nine : area_square side_length = 9 := by
  sorry

end square_area_is_nine_l58_58994


namespace domain_g_l58_58879

def domain_f (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 2
def g (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ((1 < x) ∧ (x ≤ Real.sqrt 3)) ∧ domain_f (x^2 - 1) ∧ (0 < x - 1 ∧ x - 1 < 1)

theorem domain_g (x : ℝ) (f : ℝ → ℝ) (hf : ∀ a, domain_f a → True) : 
  g x f ↔ 1 < x ∧ x ≤ Real.sqrt 3 :=
by 
  sorry

end domain_g_l58_58879


namespace probability_four_green_marbles_exactly_four_times_l58_58640

theorem probability_four_green_marbles_exactly_four_times :
    let total_marbles := 12,
        green_marbles := 8,
        purple_marbles := 4,
        num_draws := 8,
        successful_draws := 4,
        P := (binomial num_draws successful_draws * (( green_marbles / total_marbles ) ^ successful_draws * ( purple_marbles / total_marbles ) ^ (num_draws - successful_draws))) in
        P ≈ 0.171 :=
by sorry

end probability_four_green_marbles_exactly_four_times_l58_58640


namespace problem_statement_l58_58439

-- Definitions corresponding to the given condition
noncomputable def sum_to_n (n : ℕ) : ℤ := (n * (n + 1)) / 2
noncomputable def alternating_sum_to_n (n : ℕ) : ℤ := if n % 2 = 0 then -(n / 2) else (n / 2 + 1)

-- Lean statement for the problem
theorem problem_statement :
  (alternating_sum_to_n 2022) * (sum_to_n 2023 - 1) - (alternating_sum_to_n 2023) * (sum_to_n 2022 - 1) = 2023 :=
sorry

end problem_statement_l58_58439


namespace smallest_positive_integer_x_l58_58724

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end smallest_positive_integer_x_l58_58724


namespace term_2003_l58_58262

def is_perfect_square (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

def sequence_without_squares : ℕ → ℕ
| 0       := 0  -- The function is zero-based, so the 0th term is zero (arbitrary initial value)
| (n + 1) := 
  let previous_seq := sequence_without_squares n + 1 in
  if is_perfect_square previous_seq then 
    sequence_without_squares n + 1
  else 
    previous_seq

theorem term_2003 : sequence_without_squares 2003 = 2048 :=
sorry

end term_2003_l58_58262


namespace eccentricity_of_geometric_conic_section_l58_58907

theorem eccentricity_of_geometric_conic_section 
    (m : ℝ) (h : ∃ a b c : ℝ, a = 2 ∧ b = m ∧ c = 8 ∧ b^2 = a * c) :
    (∃ e : ℝ, e = (√2)/2 ∨ e = √3) :=
by
  -- Given that 2, m, 8 form a geometric sequence, we have m^2 = 2 * 8
  -- When m = 4, the equation represents an ellipse with eccentricity (√2)/2
  -- When m = -4, the equation represents a hyperbola with eccentricity √3
  sorry

end eccentricity_of_geometric_conic_section_l58_58907


namespace number_of_green_balls_l58_58755

variable (b g : Nat) (P_b : Rat)

-- Given conditions
def num_blue_balls : Nat := 10
def prob_blue : Rat := 1 / 5

-- Define the total number of balls available and the probability equation
def total_balls : Nat := num_blue_balls + g
def probability_equation : Prop := (num_blue_balls : Rat) / (total_balls : Rat) = prob_blue

-- The main statement to be proven
theorem number_of_green_balls :
  probability_equation → g = 40 := 
sorry

end number_of_green_balls_l58_58755


namespace Tino_jellybeans_l58_58709

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l58_58709


namespace find_loss_percentage_l58_58424

theorem find_loss_percentage (CP SP_new : ℝ) (h1 : CP = 875) (h2 : SP_new = CP * 1.04) (h3 : SP_new = SP + 140) : 
  ∃ L : ℝ, SP = CP - (L / 100 * CP) → L = 12 := 
by 
  sorry

end find_loss_percentage_l58_58424


namespace kolya_always_wins_l58_58012

theorem kolya_always_wins (a b : ℤ) (petya_move : ℤ → ℤ) (kolya_move : ℤ → ℤ) :
  (∀ a₁ b₁, petya_move a₁ = a₁ + 1 ∨ b₁ = b₁ + 1) →
  (∀ a₂ b₂ k, kolya_move a₂ = a₂ + k ∨ b₂ = b₂ + k ∧ (k = 1 ∨ k = 3)) →
  ∃ a₃ b₃, x^2 + a₃ * x + b₃ = (x - r) * (x - s) ∧ r s : ℤ :=
sorry

end kolya_always_wins_l58_58012


namespace m_range_l58_58865

/-- Given a point (x, y) on the circle x^2 + (y - 1)^2 = 2, show that the real number m,
such that x + y + m ≥ 0, must satisfy m ≥ 1. -/
theorem m_range (x y m : ℝ) (h₁ : x^2 + (y - 1)^2 = 2) (h₂ : x + y + m ≥ 0) : m ≥ 1 :=
sorry

end m_range_l58_58865


namespace eq_has_two_solutions_l58_58166

theorem eq_has_two_solutions : 
  {x : ℝ | |x-2| = |x-4| + |x-6| + 1}.finite.to_finset.card = 2 :=
sorry

end eq_has_two_solutions_l58_58166


namespace sqrt_rational_if_sum_rational_l58_58376

theorem sqrt_rational_if_sum_rational (a b : ℚ) (h : (Real.sqrt a + Real.sqrt b : ℝ) ∈ ℚ) :
  (Real.sqrt a : ℝ) ∈ ℚ ∧ (Real.sqrt b : ℝ) ∈ ℚ :=
sorry

end sqrt_rational_if_sum_rational_l58_58376


namespace ricciana_jump_distance_l58_58263

theorem ricciana_jump_distance (R : ℕ) :
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1
  Total_distance_Margarita = Total_distance_Ricciana → R = 22 :=
by
  -- Definitions
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1

  -- Given condition
  intro h
  sorry

end ricciana_jump_distance_l58_58263


namespace copper_percentage_correct_l58_58772

-- Define the constants provided in the problem
def weight_lead : ℝ := 5
def weight_copper : ℝ := 12
def total_weight : ℝ := weight_lead + weight_copper

-- Assertion about the percentage of copper in the mixture
def percentage_copper : ℝ := (weight_copper / total_weight) * 100

-- Theorem stating the problem's solution
theorem copper_percentage_correct : percentage_copper = 70.59 := by
  -- This part of the code would include the mathematical proof steps
  sorry

end copper_percentage_correct_l58_58772


namespace number_of_paths_in_triangle_l58_58659

theorem number_of_paths_in_triangle : 
  let paths : ℕ := 22 in 
  ∃ (triangle : Type) (A B : triangle) 
      (destinations : triangle → list triangle)
      (count_paths : (triangle → triangle → ℕ) → triangle → ℕ)
      (path_count : triangle → triangle → ℕ),
  A = destinations.head ∧ 
  B = destinations.last ∧ 
  (∀ t ∈ destinations, ∃ u ∈ destinations, ∃ v ∈ destinations, ∃ w ∈ destinations, 
    t = u ∨ t = v ∨ t = w) ∧ 
  count_paths path_count A = paths := 
sorry

end number_of_paths_in_triangle_l58_58659


namespace paths_to_spell_AMC8_l58_58955

structure Point :=
(x : Int)
(y : Int)

def adjacent (p1 p2 : Point) : Bool :=
  (abs (p1.x - p2.x) = 1 ∧ p1.y = p2.y) ∨
  (abs (p1.y - p2.y) = 1 ∧ p1.x = p2.x)

def A := Point.mk 0 0
def M := [Point.mk -1 0, Point.mk 1 0, Point.mk 0 -1, Point.mk 0 1]
def C (m : Point) : List Point :=
  List.filter (adjacent m)
  [Point.mk (m.x - 1) m.y, Point.mk (m.x + 1) m.y, Point.mk m.x (m.y - 1), Point.mk m.x (m.y + 1)]
def Eight (c : Point) : List Point :=
  List.filter (adjacent c)
  [Point.mk (c.x - 1) c.y, Point.mk (c.x + 1) c.y, Point.mk c.x (c.y - 1), Point.mk c.x (c.y + 1)]

theorem paths_to_spell_AMC8 : List.foldr Nat.mul 1
  (List.map (λ m => (List.length (C m)) * (List.foldr (λ c acc => acc + (List.length (Eight c))) 0 (C m)))
  M) = 24 := 
by 
  sorry

end paths_to_spell_AMC8_l58_58955


namespace length_OP_greater_than_3_l58_58882

-- Define a circle with a given radius
def circle (O : Point) (r : ℝ) : Set Point := { P | dist O P = r }

-- Define the center point O and point P
variable (O P : Point)

-- Condition: Radius of circle O is 3.
axiom h1 : ∃ r, r = 3

-- Condition: Point P is outside circle O.
axiom h2 : dist O P > 3

-- The proof statement to ensure the Lean code builds successfully
theorem length_OP_greater_than_3 (O P : Point) : dist O P > 3 :=
by {
  apply h2,  -- This uses the given condition that P is outside the circle O
  sorry
}

end length_OP_greater_than_3_l58_58882


namespace min_value_mn_squared_l58_58876

theorem min_value_mn_squared (a b c m n : ℝ) 
  (h_triangle: a^2 + b^2 = c^2)
  (h_line: a * m + b * n + 2 * c = 0):
  m^2 + n^2 = 4 :=
by
  sorry

end min_value_mn_squared_l58_58876


namespace fewest_posts_required_l58_58040

def dimensions_garden : ℕ × ℕ := (32, 72)
def post_spacing : ℕ := 8

theorem fewest_posts_required
  (d : ℕ × ℕ := dimensions_garden)
  (s : ℕ := post_spacing) :
  d = (32, 72) ∧ s = 8 → 
  ∃ N, N = 26 := 
by 
  sorry

end fewest_posts_required_l58_58040


namespace combine_ingredients_l58_58329

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l58_58329


namespace equivalent_base_10_of_45321_7_l58_58343

-- Define the base 7 number as a list of digits
def digits_45321 := [4, 5, 3, 2, 1]

-- Base 7 representation of the number
def base_7_rep (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ), acc * 7 + digit) 0

-- The theorem to prove
theorem equivalent_base_10_of_45321_7 : base_7_rep digits_45321 = 11481 :=
by
  -- This is where the proof would go
  sorry

end equivalent_base_10_of_45321_7_l58_58343


namespace freds_total_marbles_l58_58099

theorem freds_total_marbles :
  let red := 38
  let green := red / 2
  let dark_blue := 6
  red + green + dark_blue = 63 := by
  sorry

end freds_total_marbles_l58_58099


namespace john_sells_n_bags_l58_58597

theorem john_sells_n_bags (cost_price selling_price total_profit : ℕ) (h_cost_price : cost_price = 4) (h_selling_price : selling_price = 8) (h_total_profit : total_profit = 120) : 
  ∃ n : ℕ, n = total_profit / (selling_price - cost_price) ∧ n = 30 :=
by {
  -- Let profit per bag
  let profit_per_bag := selling_price - cost_price,
  -- Let number of bags sold
  let bags_sold := total_profit / profit_per_bag,
  -- We need to show that bags_sold equals 30
  use bags_sold,
  -- We need to prove the number of bags sold is 30
  simp [h_cost_price, h_selling_price, h_total_profit],
  split,
  -- number of bags sold calculated equals 30
  { refl },
  -- number of bags sold is indeed 30
  { norm_num }
}

end john_sells_n_bags_l58_58597


namespace vernal_equinox_shadow_length_l58_58205

-- Lean 4 statement
theorem vernal_equinox_shadow_length :
  ∀ (a : ℕ → ℝ), (a 4 = 10.5) → (a 10 = 4.5) → 
  (∀ (n m : ℕ), a (n + 1) = a n + (a 2 - a 1)) → 
  a 7 = 7.5 :=
by
  intros a h_4 h_10 h_progression
  sorry

end vernal_equinox_shadow_length_l58_58205


namespace angle_A_calculation_l58_58563

noncomputable def angle_A_in_triangle (a b : ℝ) (B A : ℝ) : Prop :=
  ∃ A : ℝ, (0 < A ∧ A < 180) ∧ (a / real.sin A = b / real.sin B)

theorem angle_A_calculation (B A : ℝ) (a b : ℝ) (h1 : a = real.sqrt 3) (h2 : b = 1) (h3 : B = 30): 
  ∃ A : ℝ, A = 60 ∨ A = 120 :=
  by
    sorry

end angle_A_calculation_l58_58563


namespace jenny_profit_l58_58594

def cost_per_pan : ℝ := 10.00
def number_of_pans : ℝ := 20
def price_per_pan : ℝ := 25.00

theorem jenny_profit :
  let total_cost := cost_per_pan * number_of_pans in
  let total_revenue := price_per_pan * number_of_pans in
  let profit := total_revenue - total_cost in
  profit = 300 :=
by
  sorry

end jenny_profit_l58_58594


namespace geometric_sequence_sum_l58_58227

variable (a₁ q : ℝ) -- First term and common ratio

-- Definitions for the terms in the geometric sequence
def a (n : ℕ) : ℝ := a₁ * q^(n-1)

-- S_n is the sum of the first n terms of the sequence
def S (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (a₁ q : ℝ) (h : 8 * a 2 - a 5 = 0) : (S 4 / S 2) = 5 :=
by
  sorry

end geometric_sequence_sum_l58_58227


namespace find_k_l58_58001

theorem find_k (k t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : t = 75) : k = 167 := 
by 
  sorry

end find_k_l58_58001


namespace integral_of_product_eq_l58_58006

theorem integral_of_product_eq :
  ∫ (7 * x - 10) * sin (4 * x) dx =
  - (1 / 4) * (7 * x - 10) * cos (4 * x) - (7 / 16) * sin (4 * x) + C :=
sorry

end integral_of_product_eq_l58_58006


namespace measure_angle_T_l58_58953

theorem measure_angle_T (P Q R S T : ℝ) (h₀ : P = R) (h₁ : R = T) (h₂ : Q + S = 180)
  (h_sum : P + Q + R + T + S = 540) : T = 120 :=
by
  sorry

end measure_angle_T_l58_58953


namespace angle_measure_l58_58350

theorem angle_measure (x : ℝ) : 
  (180 - x = 7 * (90 - x)) → x = 75 :=
by
  sorry

end angle_measure_l58_58350


namespace max_min_difference_f_l58_58556

noncomputable def f (x : ℝ) : ℝ := if x ≥ 1 / 2 then Real.log2 (3 * x - 1) else f (1 - x)

theorem max_min_difference_f :
  (∃ (a b : ℝ), a ∈ Icc (-2 : ℝ) (0 : ℝ) ∧ b ∈ Icc (-2 : ℝ) (0 : ℝ) ∧ 
  ∀ x ∈ Icc (-2 : ℝ) (0 : ℝ), f x ≤ f b ∧ f a ≤ f x) →
  ∃ (max_val min_val : ℝ), 
  max_val = 3 ∧ min_val = 1 ∧ max_val - min_val = 2 :=
by sorry

end max_min_difference_f_l58_58556


namespace fencing_required_for_field_l58_58000

noncomputable def fence_length (L W : ℕ) : ℕ := 2 * W + L

theorem fencing_required_for_field :
  ∀ (L W : ℕ), (L = 20) → (440 = L * W) → fence_length L W = 64 :=
by
  intros L W hL hA
  sorry

end fencing_required_for_field_l58_58000


namespace graph_of_equation_is_two_lines_l58_58261

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (x * y - 2 * x + 3 * y - 6 = 0) ↔ ((x + 3 = 0) ∨ (y - 2 = 0)) := 
by
  intro x y
  sorry

end graph_of_equation_is_two_lines_l58_58261


namespace no_participant_loses_to_next_l58_58946

-- Define participants and the game relation
def Participant := ℕ  -- We use natural numbers to represent participants

-- Games relation (i.e., who lost to whom)
def lost_to (p1 p2 : Participant) : Prop := sorry

-- The main theorem: existence of a desired numbering
theorem no_participant_loses_to_next (n : ℕ) (h : n ≥ 2) : 
  ∃ (f : Fin n → Participant), ∀ i : Fin (n - 1), ¬ lost_to (f i) (f (Fin.succ i)) := sorry

end no_participant_loses_to_next_l58_58946


namespace vector_b_coordinates_l58_58906

noncomputable def sqrt (x : ℝ) := Real.sqrt x

def vector_a : ℝ × ℝ := (sqrt 3, 1)

def vector_b := (-2 * sqrt 3, -2)

def rotate (v : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  let x' := v.1 * Real.cos angle - v.2 * Real.sin angle
  let y' := v.1 * Real.sin angle + v.2 * Real.cos angle
  (x', y')

theorem vector_b_coordinates :
  rotate vector_b (2 * Real.pi / 3) = (2 * sqrt 3, -2) :=
by
  sorry

end vector_b_coordinates_l58_58906


namespace number_of_correct_propositions_is_1_l58_58701

-- Definitions corresponding to each condition
def quad_diagonals_bisect_each_other (Q : Type) [Quadrilateral Q] : Prop :=
  bisect (diagonals Q)

def quad_diagonals_equal (Q : Type) [Quadrilateral Q] : Prop :=
  equal (length (diagonal1 Q)) (length (diagonal2 Q))

def quad_diagonals_perpendicular (Q : Type) [Quadrilateral Q] : Prop :=
  perpendicular (diagonal1 Q) (diagonal2 Q)

def quad_diagonals_equal_and_perpendicular (Q : Type) [Quadrilateral Q] : Prop :=
  equal (length (diagonal1 Q)) (length (diagonal2 Q)) ∧ perpendicular (diagonal1 Q) (diagonal2 Q)

-- Question: Proving the number of correct propositions
theorem number_of_correct_propositions_is_1 : 
  ∃ quadrilateral (Q : Type) [Quadrilateral Q], 
  quad_diagonals_bisect_each_other Q ∧ 
  ¬ quad_diagonals_equal Q ∧ 
  ¬ quad_diagonals_perpendicular Q ∧ 
  ¬ quad_diagonals_equal_and_perpendicular Q :=
begin
  sorry -- Proof to be provided
end

end number_of_correct_propositions_is_1_l58_58701


namespace equality_proof_l58_58513

variable {a b c : ℝ}

theorem equality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (1 / 2) * (a + b + c) :=
by
  sorry

end equality_proof_l58_58513


namespace spirit_concentration_l58_58671

theorem spirit_concentration (vol_a vol_b vol_c : ℕ) (conc_a conc_b conc_c : ℝ)
(h_a : conc_a = 0.45) (h_b : conc_b = 0.30) (h_c : conc_c = 0.10)
(h_vola : vol_a = 4) (h_volb : vol_b = 5) (h_volc : vol_c = 6) : 
  (conc_a * vol_a + conc_b * vol_b + conc_c * vol_c) / (vol_a + vol_b + vol_c) * 100 = 26 := 
by
  sorry

end spirit_concentration_l58_58671


namespace incorrect_statement_l58_58530

noncomputable def geometric_statements (α β Plane m n Line) :=
  (m ⊥ α ∧ m ⊥ β → α ∥ β) ∧
  (m ∥ n ∧ m ⊥ α → n ⊥ α) ∧
  (¬(m ∥ α ∧ α ∩ β = n → m ∥ n)) ∧
  (m ⊥ α ∧ m ∈ β → α ⊥ β)

theorem incorrect_statement (α β Plane m n Line) :
  ¬((∀ α β m n : _, geometric_statements α β m n) ∧
    (m ∥ α ∧ α ∩ β = n → m ∥ n)) :=
sorry

end incorrect_statement_l58_58530


namespace general_term_formula_find_n_l58_58121

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2 ^ n
noncomputable def b_n (n : ℕ) : ℕ := geometric_sequence n * (Real.log2_div 2 (geometric_sequence n))

theorem general_term_formula (n : ℕ) :
  geometric_sequence n = 2 ^ n := sorry

theorem find_n (n : ℕ) :
  let T_n := ∑ i in Finset.range n, b_n (i + 1) in
  T_n + n * 2^(n + 1) = 30 → n = 4 := sorry

end general_term_formula_find_n_l58_58121


namespace cos_two_alpha_sin_beta_l58_58509

variable (α β : ℝ)

axiom alpha_acute : 0 < α ∧ α < π / 2
axiom beta_acute : 0 < β ∧ β < π / 2
axiom sin_alpha_eq : 3 * Real.sin α = 4 * Real.cos α
axiom cos_alpha_beta_eq : Real.cos (α + β) = -2 * Real.sqrt 5 / 5

theorem cos_two_alpha : Real.cos (2 * α) = -7 / 25 :=
by
  sorry

theorem sin_beta : Real.sin β = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_two_alpha_sin_beta_l58_58509


namespace count_pos_integers_divisible_by_lcm_l58_58918

theorem count_pos_integers_divisible_by_lcm (L : ℕ) (hL : L = Nat.lcm 4 (Nat.lcm 6 9)) (m : ℕ) (hm : m = 300) :
  (Finset.card (Finset.filter (λ x => x % L = 0) (Finset.range m))) = 8 :=
by
  have : L = 36 := by 
    rw [hL, Nat.lcm_assoc, Nat.lcm_comm 6 9, Nat.lcm 6 9, Nat.lcm_comm 4 36, Nat.lcm_self, Nat.lcm_comm 4 36]
    norm_num
  rw this
  sorry

end count_pos_integers_divisible_by_lcm_l58_58918


namespace true_discount_correct_l58_58696

noncomputable def annual_interest_rate (TD FV : ℝ) (T : ℝ) : ℝ :=
(TD * 100) / (FV * T)

theorem true_discount_correct (TD FV : ℝ) (T : ℝ) :
  TD = 270 ∧ FV = 2520 ∧ T = 0.75 → annual_interest_rate TD FV T ≈ 16 :=
by
  intros h
  cases h with hTd hFV
  cases hFV with hFV hT
  -- sorry proof goes here
  sorry

end true_discount_correct_l58_58696


namespace length_of_one_side_of_square_l58_58540

variable (total_ribbon_length : ℕ) (triangle_perimeter : ℕ)

theorem length_of_one_side_of_square (h1 : total_ribbon_length = 78)
                                    (h2 : triangle_perimeter = 46) :
  (total_ribbon_length - triangle_perimeter) / 4 = 8 :=
by
  sorry

end length_of_one_side_of_square_l58_58540


namespace find_D_l58_58923

theorem find_D (P Q : ℕ) (h_pos : 0 < P ∧ 0 < Q) (h_eq : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end find_D_l58_58923


namespace rectangle_area_l58_58407

variables (a b c d : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem rectangle_area :
  (a + b + 5) * (c + d + 5) = ac + ad + bc + bd + 5a + 5b + 5c + 5d + 25 :=
sorry

end rectangle_area_l58_58407


namespace triangle_properties_l58_58697

-- Defining the vertices
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 5)
def c : ℝ × ℝ := (3, 4)

-- Calculate vector differences
def AB : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
def AC : ℝ × ℝ := (c.1 - a.1, c.2 - a.2)
def BC : ℝ × ℝ := (c.1 - b.1, c.2 - b.2)

-- Calculate magnitudes of vectors
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt(v.1 * v.1 + v.2 * v.2)

def AB_length : ℝ := magnitude AB
def AC_length : ℝ := magnitude AC
def BC_length : ℝ := magnitude BC

-- Calculate dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Calculate angles
def cos_angle_A : ℝ := dot_product AB AC / (AB_length * AC_length)

def angle_A : ℝ := real.acos cos_angle_A
def angle_C : ℝ := real.pi / 2
def angle_B : ℝ := real.pi / 2 - angle_A

-- The proof statement
theorem triangle_properties :
  (AB_length = real.sqrt 10) ∧ (AC_length = 2 * real.sqrt 2) ∧ (BC_length = real.sqrt 2) ∧ 
  (angle_C = real.pi / 2) ∧ (angle_B = real.acos (dot_product AB AC / (AB_length * AC_length))) :=
sorry

end triangle_properties_l58_58697


namespace chess_tournament_no_immediate_loss_l58_58945

theorem chess_tournament_no_immediate_loss (n : ℕ) (lost : Fin n → Fin n → Prop) 
    (symm_lost : ∀ i j, lost i j → ¬ lost j i) : 
    ∃ (p : List (Fin n)), p.Nodup ∧ (∀ i j, p.Nth i = some j → p.Nth (i + 1) = some k → ¬ lost j k) := 
sorry

end chess_tournament_no_immediate_loss_l58_58945


namespace farthest_point_from_origin_l58_58360

def points : List (ℝ × ℝ) := [(1, 5), (2, -3), (4, -1), (3, 3), (-2.5, 2)]

def distance (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

theorem farthest_point_from_origin :
    ∀ p ∈ points, distance (1, 5) ≥ distance p := 
by
  intro p hp
  sorry

end farthest_point_from_origin_l58_58360


namespace find_f_2_pow_2011_l58_58300

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_positive (x : ℝ) : x > 0 → f x > 0

axiom f_initial_condition : f 1 + f 2 = 10

axiom f_functional_equation (a b : ℝ) : f a + f b = f (a+b) - 2 * Real.sqrt (f a * f b)

theorem find_f_2_pow_2011 : f (2^2011) = 2^4023 := 
by 
  sorry

end find_f_2_pow_2011_l58_58300


namespace birth_date_16_Jan_1993_l58_58638

noncomputable def year_of_birth (current_date : Nat) (age_years : Nat) :=
  current_date - age_years * 365

noncomputable def month_of_birth (current_date : Nat) (age_years : Nat) (age_months : Nat) :=
  current_date - (age_years * 12 + age_months) * 30

theorem birth_date_16_Jan_1993 :
  let boy_age_years := 10
  let boy_age_months := 1
  let current_date := 16 + 31 * 12 * 2003 -- 16th February 2003 represented in days
  let full_months_lived := boy_age_years * 12 + boy_age_months
  full_months_lived - boy_age_years = 111 → 
  year_of_birth current_date boy_age_years = 1993 ∧ month_of_birth current_date boy_age_years boy_age_months = 31 * 1 * 1993 := 
sorry

end birth_date_16_Jan_1993_l58_58638


namespace range_of_a_l58_58127

variables (a : ℝ) (p q: Prop)

def prop_p := ∀ x ∈ set.Icc (1 : ℝ) 2, x^2 - a ≥ 0
def prop_q := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

theorem range_of_a (h1 : prop_p a ∨ prop_q a) (h2 : ¬ (prop_p a ∧ prop_q a)) :
  a ∈ set.Icc (-1 : ℝ) 1 ∨ 3 < a :=
sorry

end range_of_a_l58_58127


namespace first_person_days_l58_58654

theorem first_person_days (x : ℝ) (hp : 30 ≥ 0) (ht : 10 ≥ 0) (h_work : 1/x + 1/30 = 1/10) : x = 15 :=
by
  -- Begin by acknowledging the assumptions: hp, ht, and h_work
  sorry

end first_person_days_l58_58654


namespace eq_triangle_perimeter_l58_58285

theorem eq_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end eq_triangle_perimeter_l58_58285


namespace inverse_of_exponential_l58_58558

theorem inverse_of_exponential (a : ℝ) (f : ℝ → ℝ)
  (ha1 : a > 0) (ha2 : a ≠ 1)
  (h_inv : ∀ x, f (a ^ x) = x ∧ a ^ (f x) = x)
  (h_point : f (sqrt a) = a) :
  f x = log a⁻¹ x :=
sorry

end inverse_of_exponential_l58_58558


namespace sum_of_coefficients_l58_58825

-- Given polynomial definition
def P (x : ℝ) : ℝ := (1 + x - 3 * x^2) ^ 1965

-- Lean 4 statement for the proof problem
theorem sum_of_coefficients :
  P 1 = -1 :=
by
  -- Proof placeholder
  sorry

end sum_of_coefficients_l58_58825


namespace sale_price_is_approx_5996_l58_58751

-- Define the original price
def original_price : ℝ := 74.95

-- Define the discount percentage (as a decimal)
def discount_percentage : ℝ := 20.0133422281521 / 100

-- Calculate the discount amount
def discount_amount : ℝ := original_price * discount_percentage

-- Calculate the sale price
def sale_price : ℝ := original_price - discount_amount

-- Proof statement that sale price is approximately 59.96
theorem sale_price_is_approx_5996 : abs (sale_price - 59.96) < 0.01 :=
by 
  sorry  -- proof omitted

end sale_price_is_approx_5996_l58_58751


namespace part1_part2_l58_58152

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part1 (x : ℝ) : f x < 4 ↔ -3 / 2 < x ∧ x < 5 / 2 := 
by
  sorry

noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem part2 {m n : ℝ} (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2) :
  ∀ x, g x ≥ 2 →
  (∃ x, g x = 2) →
  let a := 2 in
  ∃ min_val, min_val = (2 + 2 * Real.sqrt 2) / 2 ∧ 
  ∀ val, val = (m^2 + 2) / m + (n^2 + 1) / n → 
  (min_val ≤ val) :=
by
  sorry

end part1_part2_l58_58152


namespace problem_l58_58972

namespace MathProof

-- Definitions of A, B, and conditions
def A (x : ℤ) : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

-- Prove x = ± 1 when A ⊆ B, 
-- A ∪ B = { -1, 0, 1 }, 
-- and complement of A in B is { -1 }
theorem problem (x : ℤ) (hx : A x ⊆ B) : 
  (x = 1 ∨ x = -1) ∧ 
  (A x ∪ B = {-1, 0, 1}) ∧ 
  (B \ (A x) = {-1}) := 
sorry 

end MathProof

end problem_l58_58972


namespace intersection_M_N_l58_58529

open Set

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l58_58529


namespace total_hexagons_calculation_l58_58584

-- Define the conditions
-- Regular hexagon side length
def hexagon_side_length : ℕ := 3

-- Number of smaller triangles
def small_triangle_count : ℕ := 54

-- Small triangle side length
def small_triangle_side_length : ℕ := 1

-- Define the total number of hexagons calculated
def total_hexagons : ℕ := 36

-- Theorem stating that given the conditions, the total number of hexagons is 36
theorem total_hexagons_calculation :
    (hexagon_side_length = 3) →
    (small_triangle_count = 54) →
    (small_triangle_side_length = 1) →
    total_hexagons = 36 :=
    by
    intros
    sorry

end total_hexagons_calculation_l58_58584


namespace remainder_of_N_mod_45_l58_58974

def concatenated_num_from_1_to_52 : ℕ := 
  -- This represents the concatenated number from 1 to 52.
  -- We define here in Lean as a placeholder 
  -- since Lean cannot concatenate numbers directly.
  12345678910111213141516171819202122232425262728293031323334353637383940414243444546474849505152

theorem remainder_of_N_mod_45 : 
  concatenated_num_from_1_to_52 % 45 = 37 := 
sorry

end remainder_of_N_mod_45_l58_58974


namespace erase_one_to_make_sum_even_l58_58699

theorem erase_one_to_make_sum_even (a b c d e f g : ℕ) :
  ∃ x ∈ {a, b, c, d, e, f, g}, ∑ s in ({a, b, c, d, e, f, g} \ {x}), id s % 2 = 0 :=
by sorry

end erase_one_to_make_sum_even_l58_58699


namespace sets_equal_l58_58159

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }
def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem sets_equal : E = F :=
  sorry

end sets_equal_l58_58159


namespace midpoint_on_circle_l58_58960

variable {A B C : Point}
variable {B1 B2 C1 C2 : Point}
variable {m : Point}

-- Given triangle ABC with squares ACC1C2B and ABB1B2C drawn outwardly on sides AC and AB
def triangle (A B C : Point) : Prop := sorry
def square (A B1 B2 C : Point) : Prop := sorry

-- Define the squares on sides AB and AC
def square_on_AC (A C1 C2 B : Point) : Prop := sorry
def square_on_AB (A B1 B2 C : Point) : Prop := sorry

-- Assume that B2C2 is the segment between vertices B2 and C2
def segment (P1 P2 : Point) : LineSegment := sorry

-- Define the midpoint of segment B2C2
def midpoint (P1 P2 : Point) : Point := sorry

-- Define the circle with BC as diameter
def circle_diameter (B C : Point) : Circle := sorry

-- Statement of the problem to prove: midpoint m of segment B2C2 lies on the circle with diameter BC.
theorem midpoint_on_circle
  (h_triangle : triangle A B C)
  (h_square_on_AC : square_on_AC A C1 C2 B)
  (h_square_on_AB : square_on_AB A B1 B2 C)
  (h_midpoint : m = midpoint B2 C2)
  (h_diameter : circle_diameter B C)
  : on_circle m (circle_diameter B C) := 
sorry

end midpoint_on_circle_l58_58960


namespace susan_trips_l58_58660

-- Definitions of the problem's conditions
def radius_tank := 8 -- inches
def height_tank := 20 -- inches
def radius_bucket := 5 -- inches

-- Calculation of volumes
def volume_tank := Real.pi * radius_tank^2 * height_tank
def volume_bucket := (1 / 2) * (4 / 3) * Real.pi * radius_bucket^3

-- Calculation of the number of trips and rounding up
def trips := Nat.ceil (volume_tank / volume_bucket)

-- Target proof statement
theorem susan_trips : trips = 16 :=
  sorry

end susan_trips_l58_58660


namespace square_area_from_diagonal_l58_58045

theorem square_area_from_diagonal :
  ∀ (d : ℝ), d = 10 * Real.sqrt 2 → (d / Real.sqrt 2) ^ 2 = 100 :=
by
  intros d hd
  sorry -- Skipping the proof

end square_area_from_diagonal_l58_58045


namespace problem_proof_l58_58575

noncomputable def acute_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 
  0 < A ∧ A < π / 2 ∧ 
  0 < B ∧ B < π / 2 ∧ 
  0 < C ∧ C < π / 2

noncomputable def sin_2A (A : ℝ) : ℝ := 2 * Real.sin A * Real.cos A

noncomputable def cos_B_plus_C_by_2 (B C : ℝ) : ℝ := Real.cos ((B + C) / 2)

theorem problem_proof (A B C a b c : ℝ) 
  (h1 : acute_triangle A B C)
  (h2 : 2 * (cos_B_plus_C_by_2 B C) ^ 2 + sin_2A A = 1)
  (h3 : a = 2 * Real.sqrt 3 - 2)
  (h4 : 1 / 2 * b * c * Real.sin A = 2) :
  A = π / 6 ∧ b + c = 4 * Real.sqrt 2 :=
by
  sorry

end problem_proof_l58_58575


namespace total_goals_other_members_l58_58210

theorem total_goals_other_members (x y : ℕ) (h1 : y = (7 * x) / 15 - 18)
  (h2 : 1 / 3 * x + 1 / 5 * x + 18 + y = x)
  (h3 : ∀ n, 0 ≤ n ∧ n ≤ 3 → ¬(n * 8 > y))
  : y = 24 :=
by
  sorry

end total_goals_other_members_l58_58210


namespace savings_correct_l58_58303

def income := 20000
def ratio_income := 5
def ratio_expenditure := 4

def x := income / ratio_income
def expenditure := ratio_expenditure * x
def savings := income - expenditure

theorem savings_correct : savings = 4000 := by
  have hx : x = 4000 := by
    sorry
  have hexpenditure : expenditure = 16000 := by
    sorry
  show savings = 4000 from by
    sorry

end savings_correct_l58_58303


namespace same_color_probability_l58_58702

noncomputable def bag1 : List ℕ := [2, 2, 1] -- 2 red balls (2), 1 white ball (1)
noncomputable def bag2 : List ℕ := [2, 2, 0] -- 2 red balls (2), 1 yellow ball (0)

theorem same_color_probability :
  probability_of_same_color bag1 bag2 = 4 / 9 := 
sorry

def probability_of_same_color (bag1 bag2 : List ℕ) : ℚ := 
  let total_outcomes := bag1.length * bag2.length
  let favorable_outcomes := 
    (bag1.filter (λ c1, c1 = 2)).length * (bag2.filter (λ c2, c2 = 2)).length
  favorable_outcomes / total_outcomes

end same_color_probability_l58_58702


namespace investment_principal_l58_58433

-- Definitions provided by the conditions
def monthly_interest_payment : ℝ := 225
def annual_interest_rate : ℝ := 0.09 -- 9% in decimal form
def monthly_interest_rate : ℝ := annual_interest_rate / 12

-- Theorem to prove the principal amount
theorem investment_principal (P : ℝ) :
  monthly_interest_payment = P * monthly_interest_rate * 1 ↔ P = 30000 :=
by
  sorry

end investment_principal_l58_58433


namespace ducks_percentage_l58_58257

theorem ducks_percentage (total_birds geese_percentage swans_percentage herons_percentage : ℕ)
  (h_geese : geese_percentage = 25)
  (h_swans : swans_percentage = 30)
  (h_herons : herons_percentage = 15)
  (total_birds = geese_percentage + swans_percentage + herons_percentage + ducks_percentage)
  (remaining_percentage := 100 - geese_percentage - swans_percentage) :
  (ducks_percentage : ℚ) / remaining_percentage * 100 = 66.67 :=
by
  sorry

end ducks_percentage_l58_58257


namespace determine_m_l58_58682

-- Conditions
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2 * m + 1) * x^(2 * m - 1)

-- Lean 4 statement to prove
theorem determine_m (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (f m x) = ((m^2 - 2 * m + 1) * x^(2 * m - 1)) ∧ (0 < (m^2 - 2 * m + 1) * x^(2 * m - 2))) → m = 2 :=
sorry

end determine_m_l58_58682


namespace problem_solution_l58_58084

noncomputable def solve_equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x + 36 / (x - 4) = -9)

theorem problem_solution : {x : ℝ | solve_equation x} = {0, -5} :=
by
  sorry

end problem_solution_l58_58084


namespace find_value_l58_58774

theorem find_value (number : ℕ) (h : number / 5 + 16 = 58) : number / 15 + 74 = 88 :=
sorry

end find_value_l58_58774


namespace value_of_f_at_4_l58_58938

noncomputable def f (α : ℝ) (x : ℝ) := x^α

theorem value_of_f_at_4 : 
  (∃ α : ℝ, f α 2 = (Real.sqrt 2) / 2) → f (-1 / 2) 4 = 1 / 2 :=
by
  intros h
  sorry

end value_of_f_at_4_l58_58938


namespace part_i_part_ii_l58_58494

section

-- Definitions
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}
variable {n : ℕ}

-- Conditions
axiom Sn_eq_2an_minus_nq_minus_1 : ∀ n, S n = 2 * a n - (n - 1) * q - 1

-- Proof of problem (Ⅰ)
theorem part_i (n : ℕ) (hn : 1 ≤ n) (hq : q = 0) : a n = 2 ^ (n - 1) :=
sorry

-- Proof of problem (Ⅱ)
theorem part_ii (n : ℕ) (hn : 2 ≤ n) (hq : q > 1) :
  (∑ i in finset.range n, 1 / (1 + a (i + 1))) < 1 :=
sorry

end

end part_i_part_ii_l58_58494


namespace pizza_slices_l58_58753

theorem pizza_slices (T C O : ℕ) (hT : T = 24) (hC : C = 16) (hO : O = 18) : ∃ B : ℕ, C + O - B = T ∧ B = 10 :=
by {
  use 10,
  split,
  { rw [hT, hC, hO],
    norm_num, },
  { refl, },
}

end pizza_slices_l58_58753


namespace initial_books_in_library_l58_58700

theorem initial_books_in_library
  (books_out_tuesday : ℕ)
  (books_in_thursday : ℕ)
  (books_out_friday : ℕ)
  (final_books : ℕ)
  (h1 : books_out_tuesday = 227)
  (h2 : books_in_thursday = 56)
  (h3 : books_out_friday = 35)
  (h4 : final_books = 29) : 
  initial_books = 235 :=
by
  sorry

end initial_books_in_library_l58_58700


namespace medium_supermarkets_in_sample_l58_58193

-- Define the conditions
def large_supermarkets : ℕ := 200
def medium_supermarkets : ℕ := 400
def small_supermarkets : ℕ := 1400
def total_supermarkets : ℕ := large_supermarkets + medium_supermarkets + small_supermarkets
def sample_size : ℕ := 100
def proportion_medium := (medium_supermarkets : ℚ) / (total_supermarkets : ℚ)

-- The main theorem to prove
theorem medium_supermarkets_in_sample : sample_size * proportion_medium = 20 := by
  sorry

end medium_supermarkets_in_sample_l58_58193


namespace find_a_l58_58677

theorem find_a 
  (a b c : ℤ) 
  (h_vertex : ∀ x, (a * (x - 2)^2 + 5 = a * x^2 + b * x + c))
  (h_point : ∀ y, y = a * (1 - 2)^2 + 5)
  : a = -1 := by
  sorry

end find_a_l58_58677


namespace regular_octagon_diagonal_length_l58_58783

noncomputable def length_of_diagonal_AD (s : ℝ) : ℝ :=
  s * Real.sqrt(1 + Real.sqrt(2) / 2) / Real.sqrt(2 - Real.sqrt(2))

theorem regular_octagon_diagonal_length (s : ℝ) :
  ∃ (AD : ℝ), AD = length_of_diagonal_AD s :=
by
  use length_of_diagonal_AD s
  sorry

end regular_octagon_diagonal_length_l58_58783


namespace tiling_possible_l58_58736

theorem tiling_possible (n x : ℕ) (hx : 7 * x = n^2) : ∃ k : ℕ, n = 7 * k :=
by sorry

end tiling_possible_l58_58736


namespace cannot_erase_l58_58636

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := x^2
noncomputable def f3 (x : ℝ) := (x - 1)^2

theorem cannot_erase (x : ℝ) : 
  (∃ g : ℝ → ℝ, g ∈ {f1, f2, f3} ∧ g x = 1 / x) ∧ 
  (∀ f : ℝ → ℝ, f ∉ {f1, f2, f3} → f ≠ 1 / x) := 
sorry

end cannot_erase_l58_58636


namespace nat_numbers_count_l58_58535

theorem nat_numbers_count : 
  ∃ n_count : ℕ, n_count = 5229 ∧ 
  (∀ N : ℕ, N > 700 → 
  (
    ((
      ((1000 <= 3 * N) ∧ (3 * N < 10000)) ∨ ((1000 <= N - 700) ∧ (N - 700 < 10000)) ∨ 
      ((1000 <= N + 35) ∧ (N + 35 < 10000)) ∨ ((1000 <= 2 * N) ∧ (2 * N < 10000))
    ) = 2) →
    n_count = n_count + 1
  ) ) :=
sorry

end nat_numbers_count_l58_58535


namespace log_expression_a_squared_expression_a_half_expression_l58_58016

open Real

-- Question 1: Prove that the given logarithmic expression equals to the simplified form
theorem log_expression :
  (2 / 3 * log 8 + (log 5)^2 + log 2 * log 50 + log 25) = 
  (2 * (log 2 + log 5) + log 5 * (log 5 + log 2) + log 2) := by
  sorry

-- Question 2
-- Given a + a^-1 = 5, prove a^2 + a^-2 = 23 and a^(1/2) + a^(-1/2) = sqrt(7)

variable (a : ℝ)

-- Condition: a + a^-1 = 5
def condition : Prop := a + a⁻¹ = 5

theorem a_squared_expression (h : condition a) : a^2 + a^(-2) = 23 := by
  sorry

theorem a_half_expression (h : condition a) : a^(1/2) + a^(-1/2) = Real.sqrt 7 := by
  sorry

end log_expression_a_squared_expression_a_half_expression_l58_58016


namespace proposition_A_proposition_B_proposition_C_proposition_D_l58_58733

-- Proposition A: For all x ∈ R, x² - x ≥ x - 1
theorem proposition_A (x : ℝ) : x^2 - x ≥ x - 1 :=
by sorry

-- Proposition B: There exists x ∈ (1, +∞) such that x + 4/(x-1) = 6
theorem proposition_B : ∃ x : ℝ, 1 < x ∧ x + 4/(x-1) = 6 :=
by sorry

-- Proposition C: For any non-zero real numbers a and b, b/a + a/b ≥ 2 (Disproof)
theorem proposition_C (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ¬ (b/a + a/b ≥ 2) :=
by sorry

-- Proposition D: The minimum value of the function y = (x² + 10) / √(x² + 9) is 2 (Disproof)
theorem proposition_D : ¬ (∃ x : ℝ, (x^2 + 10) / real.sqrt (x^2 + 9) = 2) :=
by sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l58_58733


namespace votes_candidate_X_l58_58950

theorem votes_candidate_X (X Y Z : ℕ) (h1 : X = (3 / 2 : ℚ) * Y) (h2 : Y = (3 / 5 : ℚ) * Z) (h3 : Z = 25000) : X = 22500 :=
by
  sorry

end votes_candidate_X_l58_58950


namespace functional_eq_is_linear_l58_58461

theorem functional_eq_is_linear (f : ℚ → ℚ)
  (h : ∀ x y : ℝ, f ((x + y) / 2) = (f x / 2) + (f y / 2)) : ∃ k : ℚ, ∀ x : ℚ, f x = k * x :=
by
  sorry

end functional_eq_is_linear_l58_58461


namespace two_digit_perfect_cubes_divisible_by_4_count_l58_58170

theorem two_digit_perfect_cubes_divisible_by_4_count : 
  ∃ (count : ℕ), count = 2 ∧ 
  (∀ n, n ∈ {64, 512} → (n ≥ 10 ∧ n ≤ 99) ∧ (∃ k, n = k^3) ∧ n % 4 = 0) :=
by
sorry

end two_digit_perfect_cubes_divisible_by_4_count_l58_58170


namespace dice_probability_greater_than_eight_l58_58715

open Finset

theorem dice_probability_greater_than_eight :
  (card {pair : ℕ × ℕ | (1 ≤ pair.1 ∧ pair.1 ≤ 6) ∧ (1 ≤ pair.2 ∧ pair.2 ≤ 6) ∧ (pair.1 + pair.2 > 8)} : ℕ) / 
  (card {pair : ℕ × ℕ | (1 ≤ pair.1 ∧ pair.1 ≤ 6) ∧ (1 ≤ pair.2 ∧ pair.2 ≤ 6)} : ℕ) = 5 / 18 :=
by
  sorry

end dice_probability_greater_than_eight_l58_58715


namespace toms_speed_l58_58331

variables {d : ℝ} -- distance from B to C
variables {v : ℝ} -- Tom's speed from B to C
variables (t1 : ℝ) -- time from Q to B
variables (t2 : ℝ) -- time from B to C
variables (total_time : ℝ) -- total time for journey
variables (total_distance : ℝ) -- total distance for journey

-- Conditions
def condition1 := ∀ d : ℝ, t1 = 2 * d / 60
def condition2 := total_distance = 3 * d
def condition3 := total_time = t1 + d / v
def condition4 := total_distance / total_time = 36

-- Theorem to prove
theorem toms_speed (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : v = 20 :=
by sorry

end toms_speed_l58_58331


namespace ellipse_standard_equation_line_slope_range_l58_58120

-- Define the ellipse and the eccentricity
def is_ellipse (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1

def is_eccentricity (e : ℝ) : Prop :=
  e = (Real.sqrt 3) / 2

-- Define conditions for points M and N on the ellipse, the midpoint P, and point Q
def line_through (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, y2 = k * x + y1

def dot_product_positive (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 > 0

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Definitions for the problems concerning the range of slopes
def slope_range (k' : ℝ) : Prop :=
  (-Real.sqrt 3 / 6 < k' ∧ k' < -1 / 8) ∨ (1 / 8 < k' ∧ k' < Real.sqrt 3 / 6)

-- Statements to be proved
theorem ellipse_standard_equation (a : ℝ) (h1 : a = 2) (e : ℝ) (h2 : is_eccentricity e) :
  is_ellipse 2 x y :=
sorry

theorem line_slope_range (x1 y1 x2 y2 : ℝ) (h1 : line_through x1 y1 x2 y2 (0, 2))
  (h2 : dot_product_positive x1 y1 x2 y2) :
  slope_range k' :=
sorry

end ellipse_standard_equation_line_slope_range_l58_58120


namespace trig_identity_t_half_l58_58095

theorem trig_identity_t_half (a t : ℝ) (ht : t = Real.tan (a / 2)) :
  Real.sin a = (2 * t) / (1 + t^2) ∧
  Real.cos a = (1 - t^2) / (1 + t^2) ∧
  Real.tan a = (2 * t) / (1 - t^2) := 
sorry

end trig_identity_t_half_l58_58095


namespace painting_ways_3x3_grid_l58_58750

theorem painting_ways_3x3_grid : 
  let grid := fin 3 × fin 3,
  let color := {0, 1}, -- 0 represents red, 1 represents green
  ∃ (choices : grid → color), 
    (∀ (i j : fin 3), choices (i, j) = 1 → (i = 0 ∨ choices (i - 1, j) ≠ 0) ∧ (j = 0 ∨ choices (i, j - 1) ≠ 0)) ∧ 
    (finset.card (set_of choices) = 9) :=
sorry

end painting_ways_3x3_grid_l58_58750


namespace distance_between_points_l58_58465

-- Define the distance function for 3-dimensional space
def distance_3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

-- The given points
def point1 : ℝ × ℝ × ℝ := (2, 5, 1)
def point2 : ℝ × ℝ × ℝ := (0, 0, 4)

theorem distance_between_points :
  distance_3d point1 point2 = real.sqrt 38 :=
by sorry

end distance_between_points_l58_58465


namespace distinct_values_f_of_nonneg_x_l58_58238

def f (x : ℝ) : ℝ := ∑ k in (Finset.range 9).image (λ n, n + 2), (⌊k * x⌋ - k * ⌊x⌋)

theorem distinct_values_f_of_nonneg_x : ∃ (S : Finset ℝ), S.card = 32 ∧ ∀ x : ℝ, 0 ≤ x → f(x) ∈ S :=
sorry

end distinct_values_f_of_nonneg_x_l58_58238


namespace fraction_product_equals_l58_58807

theorem fraction_product_equals :
  (7 / 4) * (14 / 49) * (10 / 15) * (12 / 36) * (21 / 14) * (40 / 80) * (33 / 22) * (16 / 64) = 1 / 12 := 
  sorry

end fraction_product_equals_l58_58807


namespace focus_of_parabola_l58_58155

theorem focus_of_parabola (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x, y = 3 * x → 3 / a = 3) :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 8) :=
by
  -- The proof goes here
  sorry

end focus_of_parabola_l58_58155


namespace total_balloons_l58_58429

-- Define the number of balloons Alyssa, Sandy, and Sally have.
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Theorem stating that the total number of balloons is 104.
theorem total_balloons : alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  -- Proof is omitted for the purpose of this task.
  sorry

end total_balloons_l58_58429


namespace cubic_identity_l58_58180

theorem cubic_identity (a b c : ℝ) 
  (h1 : a + b + c = 12)
  (h2 : ab + ac + bc = 30)
  : a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end cubic_identity_l58_58180


namespace problem_l58_58933

theorem problem (triangle square : ℕ) (h1 : triangle + 5 ≡ 1 [MOD 7]) (h2 : 2 + square ≡ 3 [MOD 7]) :
  triangle = 3 ∧ square = 1 := by
  sorry

end problem_l58_58933


namespace geometric_sequence_trig_l58_58545

theorem geometric_sequence_trig (x : ℝ) (k : ℤ) 
  (h₁ : cos x ≠ 0 ∧ sin x = tan x * cos x):
  (cot x := cos x / sin x), cot x^4 + cot x^2 = undefined :=
sorry

end geometric_sequence_trig_l58_58545


namespace domain_of_f_l58_58298

def f (x : ℝ) : ℝ := real.sqrt (2^x - 4)

theorem domain_of_f :
  {x : ℝ | 2^x - 4 ≥ 0} = {x : ℝ | x ≥ 2} :=
by {
  ext x,
  simp,
  exact ⟨λ h, real.log (h + 4) / real.log 2, sorry⟩
}

end domain_of_f_l58_58298


namespace impossible_sequence_l58_58668

def letters_order : List ℕ := [1, 2, 3, 4, 5]

def is_typing_sequence (order : List ℕ) (seq : List ℕ) : Prop :=
  sorry -- This function will evaluate if a sequence is possible given the order

theorem impossible_sequence : ¬ is_typing_sequence letters_order [4, 5, 2, 3, 1] :=
  sorry

end impossible_sequence_l58_58668


namespace equation_of_hyperbola_value_of_m_l58_58524

-- Part 1: Equation of the Hyperbola
theorem equation_of_hyperbola (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c / a = sqrt 3) (h4 : c^2 + b^2 = 5) (h5 : c^2 = a^2 + b^2) : 
  (C : set (ℝ × ℝ)) = { p : ℝ × ℝ | p.1^2 - p.2^2 / 2 = 1 } :=
sorry

-- Part 2: Finding the value of m
theorem value_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2 - y^2 / 2 = 1) → (x - y + m = 0)) → 
  ((x + y + m = 0) → ((∀ a b : ℝ, (a + b)^2 + (x + y + m)^2 = 5)) → 
   m = 1 ∨ m = -1) :=
sorry

end equation_of_hyperbola_value_of_m_l58_58524


namespace compute_g4_cubed_l58_58658

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom condition1 (x : ℝ) (hx : x ≥ 1) : f (g x) = x^2
axiom condition2 (x : ℝ) (hx : x ≥ 1) : g (f x) = x^3
axiom condition3 : g 16 = 16

theorem compute_g4_cubed : [g 4]^3 = 16 := by
  sorry

end compute_g4_cubed_l58_58658


namespace coord_of_point_M_in_third_quadrant_l58_58514

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0 ∧ abs y = 1 ∧ abs x = 2

theorem coord_of_point_M_in_third_quadrant : 
  ∃ (x y : ℝ), point_coordinates x y ∧ (x, y) = (-2, -1) := 
by {
  sorry
}

end coord_of_point_M_in_third_quadrant_l58_58514


namespace geometric_sequence_sum_b_n_l58_58211

noncomputable def sequence_term (a_n : ℕ → ℝ) (n : ℕ) : ℝ := - (2/3)^(n - 1)

theorem geometric_sequence (a : ℕ → ℝ) (h_ratio : ∀ n : ℕ, a n = - (2 / 3 : ℝ)^(n)) (h : a 1 * (a 1 * (2/3)^4) = 8/27) :
  ∀ n : ℕ, a (n + 1) = (2 / 3) * a n :=
sorry

noncomputable def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := a n + n

theorem sum_b_n (a : ℕ → ℝ) (h : ∀ n, a n = - (2 / 3)^(n - 1)) :
  ∀ n, ∑ i in finset.range n, b_n a i  = (n^2 + n + 6) / 2 - 3 * (2/3)^n :=
sorry

end geometric_sequence_sum_b_n_l58_58211


namespace lateral_surface_area_of_pyramid_l58_58666

theorem lateral_surface_area_of_pyramid (a α β : ℝ) :
  let lateral_surface_area := 2 * a^2 * sin β * (cos ((π/4) - (α/2)))^2 / (cos α) in
  lateral_surface_area = (2 * a^2 * sin β * (cos ((π/4) - (α/2)))^2 / (cos α)) :=
by
  sorry

end lateral_surface_area_of_pyramid_l58_58666


namespace physics_class_size_l58_58436

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 53)
  (h2 : both = 7)
  (h3 : physics_only = 2 * (math_only + both))
  (h4 : total_students = physics_only + math_only + both) :
  physics_only + both = 40 :=
by
  sorry

end physics_class_size_l58_58436


namespace number_of_correct_conclusions_l58_58229

-- Define the conditions as hypotheses
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {a_1 : ℝ}
variable {n : ℕ}

-- Arithmetic sequence definition for a_n
def arithmetic_sequence (a_n : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Problem statement
theorem number_of_correct_conclusions 
  (h_seq : arithmetic_sequence a_n a_1 d)
  (h_sum : sum_arithmetic_sequence S a_1 d)
  (h1 : S 5 < S 6)
  (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  ∃ n, n = 3 ∧ 
       (d < 0) ∧ 
       (a_n 7 = 0) ∧ 
       ¬(S 9 = S 5) ∧ 
       (S 6 = S 7 ∧ ∀ m, m > 7 → S m < S 6) := 
sorry

end number_of_correct_conclusions_l58_58229


namespace translation_parallel_vectors_l58_58140

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, -3)

theorem translation_parallel_vectors : 
  (∃ v : ℝ × ℝ, v = (2,3) ∧ ∀ u : ℝ × ℝ, u = (2, 3) ∨ u = (-2, -3)) → 
  vec_a ∥ vec_b := 
sorry

end translation_parallel_vectors_l58_58140


namespace B_speaks_truth_l58_58044

variable (pA pAB pB : ℝ)

-- Given conditions
def pA_condition : Prop := pA = 0.55
def pAB_condition : Prop := pAB = 0.33

-- Proven Statement
theorem B_speaks_truth :
  pA = 0.55 →
  pAB = 0.33 →
  pB = 0.6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end B_speaks_truth_l58_58044


namespace ellipse_parabola_problem_l58_58880

theorem ellipse_parabola_problem :
  (let F1 := (-1, 0)
   let F2 := (1, 0)
   let P := (-1, (Real.sqrt 2) / 2)
   ∃ a b : ℝ, a = Real.sqrt 2 ∧ b = 1 ∧
     (∀ x y : ℝ, (x^2 / 2) + y^2 = 1 ↔ (x, y) ∈ set_of (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1))) ∧
  (∃ (p : ℝ), p = 1 / 4 ∧ ∀ (x₀ y₀ > 0 : ℝ),
    (∃ M N : (ℝ × ℝ), M = (x₀, y₀) ∧ N = (x₀, -y₀) ∧
    (y₀^2 = 2 * p * x₀ ∧ (x₀^2 / 2) + y₀^2 = 1)
    → (x₀ * y₀) = (Real.sqrt 2) / 2 * 1))) :=
by
  sorry

end ellipse_parabola_problem_l58_58880


namespace popsicle_sticks_left_correct_l58_58481

noncomputable def popsicle_sticks_left (initial : ℝ) (given : ℝ) : ℝ :=
  initial - given

theorem popsicle_sticks_left_correct :
  popsicle_sticks_left 63 50 = 13 :=
by
  sorry

end popsicle_sticks_left_correct_l58_58481


namespace projection_on_a_l58_58158

variables (m : ℝ)

def vector_a : ℝ × ℝ := (2 * m - 1, 2)
def vector_b : ℝ × ℝ := (-2, 3 * m - 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def scaled_vector (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

def vector_projection (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / magnitude w

theorem projection_on_a (h : magnitude (vector_add vector_a vector_b) = magnitude (vector_sub vector_a vector_b)) :
  vector_projection (scaled_vector 5 vector_a - scaled_vector 3 vector_b) vector_a = 5 * Real.sqrt 5 :=
sorry

end projection_on_a_l58_58158


namespace differences_repeated_at_least_four_l58_58488

theorem differences_repeated_at_least_four (a : Fin 20 → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : ∀ i, a i ≤ 70) :
  ∃ d, Finset.card (Finset.filter (λ x, x = d) { d | ∃ i j, i < j ∧ d = a j - a i }) ≥ 4 :=
by
  sorry

end differences_repeated_at_least_four_l58_58488


namespace matrix_commutation_l58_58976

def A : matrix (fin 2) (fin 2) ℤ := ![![1, 3], ![0, 2]]
def B (x y z w : ℤ) : matrix (fin 2) (fin 2) ℤ := ![![x, y], ![z, w]]

theorem matrix_commutation (x y z w : ℤ) (h : A * B x y z w = B x y z w * A) (hz : z ≠ 3 * y) :
  (x - w) / (z - 3 * y) = -1 / 9 :=
by
  sorry

end matrix_commutation_l58_58976


namespace equilateral_triangle_area_l58_58309

open Real

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) :
  let s := p in
  let area := sqrt 3 / 4 * s^2 in
  s = p →
  area = sqrt 3 / 4 * p^2 :=
by
  intros s area h_s
  -- sorry to indicate we are skipping the proof steps
  sorry

end equilateral_triangle_area_l58_58309


namespace range_of_x_l58_58133

noncomputable def f : ℝ → ℝ := sorry

-- Defining the statement with given conditions and result
theorem range_of_x
  (even_f : ∀ x, f x = f (-x))
  (decreasing_f : ∀ {a b : ℝ},  0 ≤ a → a ≤ b → f b ≤ f a)
  (h : ∀ x, f (log x) > f 1) :
  ∀ x, (f (log x) > f 1) ↔ (1/10 < x ∧ x < 10) :=
sorry

end range_of_x_l58_58133


namespace part1_part2_l58_58150
noncomputable theory

def f (x : ℝ) (m : ℝ) := 2 * sqrt 3 * cos x ^ 2 + 2 * sin x * cos x - m

theorem part1 (hx : ∃ x, f x m = 2) : m = sqrt 3 :=
sorry

theorem part2 (A B C a b c : ℝ) (h1 : A < π / 2) (h2 : f A 0 = 0)
  (h3 : sin B = 3 * sin C) (h4 : 1 / 2 * b * c * sin A = 3 * sqrt 3 / 4) : a = sqrt 7 :=
begin
  sorry
end

end part1_part2_l58_58150


namespace sum_of_all_possible_4_digit_numbers_l58_58356

theorem sum_of_all_possible_4_digit_numbers : 
  let digits := [2, 4, 5, 6] in
  ∑ n in {n | n.digits 10 = digits.permutations ∧ (1000 ≤ n ∧ n < 10000)}, n = 113322 :=
by 
  let digits := [2, 4, 5, 6]
  have : digits.permutations.toFinset.card = 24, from by sorry
  sorry

end sum_of_all_possible_4_digit_numbers_l58_58356


namespace storm_damage_in_usd_l58_58418

theorem storm_damage_in_usd 
  (yen_damage : ℕ) 
  (yen_to_euro : ℕ) 
  (euro_to_usd : ℚ) 
  (h_yen_damage : yen_damage = 40000000000) 
  (h_yen_to_euro : yen_to_euro = 120) 
  (h_euro_to_usd : euro_to_usd = 5 / 4) :
  (yen_damage / yen_to_euro : ℚ) * euro_to_usd = 416666667 := 
begin
  sorry
end

end storm_damage_in_usd_l58_58418


namespace ratio_of_mixture_l58_58621

theorem ratio_of_mixture (x y : ℚ)
  (h1 : 0.6 = (4 * x + 7 * y) / (9 * x + 9 * y))
  (h2 : 50 = 9 * x + 9 * y) : x / y = 8 / 7 := 
sorry

end ratio_of_mixture_l58_58621


namespace total_ingredients_l58_58327

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end total_ingredients_l58_58327


namespace find_XY_l58_58082

-- Defining the points X, Y, Z
variables {X Y Z : Point}

-- Defining the triangle XYZ
def triangle (X Y Z : Point) : Prop := true

-- Defining the 30-60-90 triangle condition
def is_30_60_90_triangle (X Y Z : Point) : Prop :=
  triangle X Y Z ∧ ∠XYZ = 60 ∧ hypotenuse_length (X Y Z) = 12

-- Defining the function to calculate hypotenuse length
def hypotenuse_length (X Y Z : Point) : ℝ := sorry

-- The main theorem to prove
theorem find_XY (X Y Z : Point) (h : is_30_60_90_triangle X Y Z) : XY = 6 :=
by
  sorry

end find_XY_l58_58082


namespace kite_area_l58_58415

noncomputable def area_of_kite (s : ℝ) (d : ℝ) : ℝ :=
  let h := (√3 / 2) * s in
  let new_base := s - d in
  new_base * h

theorem kite_area : area_of_kite 4 1 = 6 * √3 :=
by
  sorry

end kite_area_l58_58415


namespace part1_part2_l58_58154

-- Given definitions for f(x) and g(x)
def f (k : ℝ) (x : ℝ) := k * x^2 + (2 * k - 1) * x + k
def g (k : ℝ) (x : ℝ) := log (x + k) / log 2

theorem part1 (k : ℝ) (h : f k 0 = 7) : 
  ∃ m, m = g k 9 ∧ m = 4 :=
by sorry

theorem part2 (k : ℝ) (h1 : 0 < g k 1 ∧ g k 1 ≤ 5) (h2 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → f k x ≥ 4) : 
  1 ≤ k ∧ k ≤ 31 :=
by sorry

end part1_part2_l58_58154


namespace no_point_on_line_y_eq_2x_l58_58258

theorem no_point_on_line_y_eq_2x
  (marked : Set (ℕ × ℕ))
  (initial_points : { p // p ∈ [(1, 1), (2, 3), (4, 5), (999, 111)] })
  (rule1 : ∀ a b, (a, b) ∈ marked → (b, a) ∈ marked ∧ (a - b, a + b) ∈ marked)
  (rule2 : ∀ a b c d, (a, b) ∈ marked ∧ (c, d) ∈ marked → (a * d + b * c, 4 * a * c - 4 * b * d) ∈ marked) :
  ∃ x, (x, 2 * x) ∈ marked → False := sorry

end no_point_on_line_y_eq_2x_l58_58258


namespace max_n_value_l58_58898

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem max_n_value (m : ℝ) (x_i : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, i < n → f (x_i i) / (x_i i) = m)
  (h2 : ∀ i, i < n → -2 * Real.pi ≤ x_i i ∧ x_i i ≤ 2 * Real.pi) :
  n ≤ 12 :=
sorry

end max_n_value_l58_58898


namespace coefficient_of_x4_in_expansion_l58_58207

-- Define the problem and necessary constants
noncomputable def expr := (x + (1/(2 * x^(1/3))))^8

-- Define the goal to prove
theorem coefficient_of_x4_in_expansion :
  (coeff (expand expr) x^4) = 7 :=
sorry

end coefficient_of_x4_in_expansion_l58_58207


namespace cos_of_6_arccos_one_fourth_l58_58061

theorem cos_of_6_arccos_one_fourth : 
  cos (6 * arccos (1/4)) = -7/128 := 
by 
  sorry

end cos_of_6_arccos_one_fourth_l58_58061


namespace sum_of_integers_l58_58684

theorem sum_of_integers (n : ℤ) (h : n * (n + 2) = 20400) : n + (n + 2) = 286 ∨ n + (n + 2) = -286 :=
by
  sorry

end sum_of_integers_l58_58684


namespace relationship_among_abc_l58_58105

noncomputable def a : ℝ := 2 ^ 12
noncomputable def b : ℝ := (1 / 2) ^ (-0.8)
noncomputable def c : ℝ := 3 ^ (-0.8)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  -- Definitions
  let a := 2 ^ 12,
  let b := (1 / 2) ^ (-0.8),
  let c := 3 ^ (-0.8),

  -- Proof of the statements would go here
  sorry
}

end relationship_among_abc_l58_58105


namespace tino_jellybeans_l58_58707

variable (Tino Lee Arnold : ℕ)

-- Conditions
variable (h1 : Tino = Lee + 24)
variable (h2 : Arnold = Lee / 2)
variable (h3 : Arnold = 5)

-- Prove Tino has 34 jellybeans
theorem tino_jellybeans : Tino = 34 :=
by
  sorry

end tino_jellybeans_l58_58707


namespace ratio_length_breadth_landscape_l58_58306

theorem ratio_length_breadth_landscape
  (L : ℝ) (A_playground : ℝ) (fraction_playground : ℝ) (A_landscape : ℝ)
  (b : ℝ) :
  L = 240 ∧ A_playground = 1200 ∧ fraction_playground = 1/6 ∧ A_landscape = 7200 ∧ 240 * b = A_landscape →
  L / b = 8 :=
by
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h7 h8,
  sorry

end ratio_length_breadth_landscape_l58_58306


namespace range_f1_l58_58747
open Function

theorem range_f1 (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Ici (-1) → y ∈ Set.Ici (-1) → x ≤ y → (x^2 + 2*a*x + 3) ≤ (y^2 + 2*a*y + 3)) →
  6 ≤ (1^2 + 2*a*1 + 3) :=
by
  intro h
  sorry

end range_f1_l58_58747


namespace geom_seq_limit_l58_58491

theorem geom_seq_limit (a : ℕ → ℝ) (h1 : a 2 = 2) (h2 : a 3 = 1) :
  (∃ q a_1, q = a 3 / a 2 ∧ a_1 = a 2 / q ∧ 
  (∀ n, a 1 * a 2 + a 2 * a 3 + ... + a n * a (n + 1) = 
  8 * (1 - (q^2)^n) / (1 - q^2)) ∧ 
  (lim (λ n, a 1 * a 2 + a 2 * a 3 + ... + a n * a (n + 1)) = 32 / 3)) :=
sorry

end geom_seq_limit_l58_58491


namespace product_of_divisors_of_45_l58_58467

theorem product_of_divisors_of_45 : ∏ d in (finset.filter (λ (d : ℕ), 45 % d = 0) (finset.range 46)), d = 91125 := 
sorry

end product_of_divisors_of_45_l58_58467


namespace common_elements_count_l58_58531

def sequence1 (n : ℕ) : ℕ := 2 * n - 1
def sequence2 (n : ℕ) : ℕ := 5 * n - 4

theorem common_elements_count : 
  (λ m, ∃ n, sequence1 n = m) ∩ (λ m, ∃ n, sequence2 n = m) {x | x ≤ 1991} 
  = 200 := sorry

end common_elements_count_l58_58531


namespace regular_hexadecagon_area_l58_58041

noncomputable def area_of_hexadecagon (r : ℝ) : ℝ :=
  4 * r^2 * (Real.sqrt 2 - Real.sqrt 2.sqrt)

theorem regular_hexadecagon_area (r : ℝ) (h : 0 < r):
  area_of_hexadecagon r = 4 * r^2 * (Real.sqrt 2 - Real.sqrt 2.sqrt) :=
by
  -- Proof to be provided
  sorry

end regular_hexadecagon_area_l58_58041


namespace length_OP_equals_4_l58_58887

-- Condition 1: The radius of circle O is 3.
def radius_circle (O : Type) [metric_space O] : ℝ := 3

-- Condition 2: Point P is outside circle O.
def point_outside_circle (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ) : Prop :=
  ∀ o ∈ O, ∀ p ∈ P, d o p > 3

-- Question: Prove the length OP is 4.
theorem length_OP_equals_4 
  (O P : Type) [metric_space O] [metric_space P] (d : O → P → ℝ)
  (h_radius : radius_circle O = 3)
  (h_outside : point_outside_circle O P d) : 
  ∃ (p ∈ P) (o ∈ O), d o p = 4 :=
by
  sorry

end length_OP_equals_4_l58_58887


namespace general_formula_sum_T_l58_58978

variable {a : ℕ → ℕ} 
variable {S : ℕ → ℕ}
variable {b : ℕ → ℕ}
variable {T : ℕ → ℝ}

-- Define the conditions given in the problem
axiom a_pos (n : ℕ) (h : n > 0) : a n > 0
axiom eq_sum (n : ℕ) : a n ^ 2 + 2 * a n = 4 * S n - 1
axiom sum_fn (S : ℕ → ℕ) : S 0 = 0 ∧ (∀ n, S (n + 1) = S n + a (n + 1))
axiom b_seq (n : ℕ) : b n = 3^(n - 1)

-- Prove the general formula for the sequence {a_n}
theorem general_formula (n : ℕ) : a n = 2 * n - 1 :=
  sorry

-- Prove the sum of the first n terms of the sequence {a_n / b_n} denoted as T_n
theorem sum_T (n : ℕ) : 
  T n = (∑ k in Finset.range n, (a (k + 1) : ℝ) / (b (k + 1) : ℝ)) → 
  T n = 3 - (n + 1) / (3^(n - 1) : ℝ) :=
  sorry

end general_formula_sum_T_l58_58978


namespace axis_of_symmetry_l58_58674

noncomputable def f (x : ℝ) := x^2 - 2 * x + Real.cos (x - 1)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (1 + x) = f (1 - x) :=
by 
  sorry

end axis_of_symmetry_l58_58674


namespace hyperbola_asymptotes_correct_l58_58156

-- Declare the hyperbola with given conditions
variables (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (eccentricity : ℝ)
def hyperbola_eq : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Define the correct answer for asymptotes
def asymptotes_eq : Prop := ∀ x : ℝ, y = x * (√2) ∨ y = - x * (√2)

-- Eccentricity condition given in the problem
def eccentricity_hyperbola : Prop := eccentricity = √3

theorem hyperbola_asymptotes_correct :
  (hyperbola_eq a b) → (eccentricity_hyperbola a b eccentricity) → (asymptotes_eq a b) :=
by sorry

end hyperbola_asymptotes_correct_l58_58156


namespace problem_1_problem_2_l58_58565

-- Definitions required for the proof
variables {A B C : ℝ} (a b c : ℝ)
variable (cos_A cos_B cos_C : ℝ)
variables (sin_A sin_C : ℝ)

-- Given conditions
axiom given_condition : (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b
axiom cos_B_eq : cos_B = 1 / 4
axiom b_eq : b = 2

-- First problem: Proving the value of sin_C / sin_A
theorem problem_1 :
  (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b → (sin_C / sin_A) = 2 :=
by
  intro h
  sorry

-- Second problem: Proving the area of triangle ABC
theorem problem_2 :
  (cos_B = 1 / 4) → (b = 2) → ((cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b) → (1 / 2 * a * c * sin_A) = (Real.sqrt 15) / 4 :=
by
  intros h1 h2 h3
  sorry

end problem_1_problem_2_l58_58565


namespace probability_first_green_probability_both_green_conditional_probability_second_green_given_first_green_l58_58948

-- Define the conditions
def total_eggs : ℕ := 5
def green_eggs : ℕ := 3
def white_eggs : ℕ := 2

-- Calculation of probabilities
theorem probability_first_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  ((green_eggs.to_real / total_eggs.to_real) = (3 / 5)) :=
by
  intros h1 h2 h3
  sorry

theorem probability_both_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  (((green_eggs.to_real / total_eggs.to_real) *
   ((green_eggs - 1).to_real / (total_eggs - 1).to_real)) = (3 / 10)) :=
by
  intros h1 h2 h3
  sorry

theorem conditional_probability_second_green_given_first_green :
  total_eggs = 5 →
  green_eggs = 3 →
  white_eggs = 2 →
  ((((green_eggs - 1).to_real / (total_eggs - 1).to_real)) = (1 / 2)) :=
by
  intros h1 h2 h3
  sorry

end probability_first_green_probability_both_green_conditional_probability_second_green_given_first_green_l58_58948


namespace probability_of_odd_even_l58_58358

def set_of_numbers : set ℕ := {1, 2, 3, 4}

def draw_two_numbers (s : set ℕ) := {x ∈ s | ∃ y ∈ s, x ≠ y ∧ (x, y)}

def is_odd (x : ℕ) : Prop := x % 2 = 1
def is_even (x : ℕ) : Prop := x % 2 = 0

def odd_even_pairs (s : set ℕ) : set (ℕ × ℕ) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 ≠ p.2 ∧ ((is_odd p.1 ∧ is_even p.2) ∨ (is_even p.1 ∧ is_odd p.2))}

theorem probability_of_odd_even :
  let total_events := C (set_of_numbers.card) 2 in
  let favorable_events := (odd_even_pairs set_of_numbers).card in
  (favorable_events : ℚ) / total_events = 2 / 3 :=
by
  sorry

end probability_of_odd_even_l58_58358


namespace order_of_operations_calc_example_l58_58578

theorem order_of_operations (a b c d : ℕ) (mul : b * c = d) : a - d = 5 :=
by sorry

-- Specific case for the given problem
theorem calc_example :
  order_of_operations 35 5 6 30 (by norm_num) = 5 :=
by norm_num

end order_of_operations_calc_example_l58_58578


namespace max_unique_sums_is_nine_l58_58396

theorem max_unique_sums_is_nine (coins : Finset ℕ) 
  (h : coins = {1, 1, 5, 5, 25, 50, 50}) : 
  ∃ s : Finset ℕ, s.card = 9 ∧ ∀ a b ∈ coins, a ≠ b → a + b ∈ s := 
sorry

end max_unique_sums_is_nine_l58_58396


namespace max_area_overlapped_region_l58_58585

noncomputable def right_triangle_example : Type :=
  {ABC : Type}
  [right_triangle ABC {A B C : ABC}] -- Assume right triangle structure
  
noncomputable def line_perpendicular (l : Type) (ABC : Type) :=
  ∃ M N : ABC, perpendicular l (AC M) ∧ perpendicular l (AB N)

theorem max_area_overlapped_region :
  ∀ (ABC : right_triangle_example)
    (A B C : ABC)
    (hC : angle C = 90)
    (hAC : side_length AC = 4)
    (hBC : side_length BC = 3)
    (l : line_perpendicular ABC),
  max_area_folded ABC l = 2.63 :=
by
  sorry

end max_area_overlapped_region_l58_58585


namespace correct_answer_statement_C_l58_58799

-- Define the conditions as per the problem statement.
def QuadrilateralWithPerpendicularEqualDiagonals (Q : Type) :=
  ∃ a b c d : Q, -- need the actual definition of quadrilateral and how it might be represented

def QuadrilateralWithOnePairSidesParallelOtherPairEqual (Q : Type) :=
  ∃ a b c d : Q, -- need the actual definition of quadrilateral and how it might be represented

def ParallelogramWithEqualDiagonals (P : Type) [Parallelogram P] :=
  ∃ a b : P, -- assume existence of points a and b such that diagonals are equal

def ParallelogramWithEqualSides (P : Type) [Parallelogram P] :=
  ∃ a b : P, -- assume existence of points a and b such that sides are equal

-- The goal is to prove that a parallelogram with equal diagonals is a rectangle.
theorem correct_answer_statement_C
  (P : Type) [Parallelogram P]
  (h_eq_diagonals : ParallelogramWithEqualDiagonals P) :
  Rectangle P :=
sorry

end correct_answer_statement_C_l58_58799


namespace charge_for_cat_l58_58255

theorem charge_for_cat (D N_D N_C T C : ℝ) 
  (h1 : D = 60) (h2 : N_D = 20) (h3 : N_C = 60) (h4 : T = 3600)
  (h5 : 20 * D + 60 * C = T) :
  C = 40 := by
  sorry

end charge_for_cat_l58_58255


namespace sum_binomial_coefficients_l58_58319

theorem sum_binomial_coefficients :
  let a := 1
  let b := 1
  let binomial := (2 * a + 2 * b)
  (binomial)^7 = 16384 := by
  -- Proof omitted
  sorry

end sum_binomial_coefficients_l58_58319


namespace terminate_decimal_l58_58833

theorem terminate_decimal :
  ∃ x : ℝ, x = 0.45625 ∧ (∃ (a b : ℤ) (h : b ≠ 0), (a / b : ℝ) = x ∧ b.factorization.keys ⊆ {2, 5}) :=
by
  use 0.45625
  split
  use [(73 : ℤ), (160 : ℤ), by norm_num]
  split
  norm_num
  sorry -- placeholder for detailed factorization and arithmetic steps

end terminate_decimal_l58_58833


namespace fred_weekend_earnings_l58_58599

noncomputable def fred_initial_dollars : ℕ := 19
noncomputable def fred_final_dollars : ℕ := 40

theorem fred_weekend_earnings :
  fred_final_dollars - fred_initial_dollars = 21 :=
by
  sorry

end fred_weekend_earnings_l58_58599


namespace ms_hatcher_students_l58_58623

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l58_58623


namespace find_m_and_star_l58_58850

-- Definitions from conditions
def star (x y m : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

-- Given conditions
def given_star (x y : ℚ) (m : ℚ) : Prop := star x y m = 2 / 5

-- Target: Proving m = 1 and 2 * 6 = 6 / 7 given the conditions
theorem find_m_and_star :
  ∀ m : ℚ, 
  (given_star 1 2 m) → 
  (m = 1 ∧ star 2 6 m = 6 / 7) := 
sorry

end find_m_and_star_l58_58850


namespace total_rent_of_pasture_l58_58738

theorem total_rent_of_pasture (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (b_months : ℕ) (c_oxen : ℕ) (c_months : ℕ) (c_payment : ℕ) 
  (h1 : a_oxen = 10) (h2 : a_months = 7) (h3 : b_oxen = 12) (h4 : b_months = 5) (h5 : c_oxen = 15) (h6 : c_months = 3) (h7 : c_payment = 36) :
  let total_oxen_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months,
      cost_per_oxen_month := c_payment / (c_oxen * c_months)
  in total_oxen_months * cost_per_oxen_month = 140 :=
by
sory

end total_rent_of_pasture_l58_58738


namespace z_minus_conj_eq_l58_58862

-- Define the complex number z
def z : ℂ := (2 - Complex.i) / (2 + Complex.i)

-- Define the conjugate of z
def z_conj : ℂ := Complex.conj z

-- State the theorem
theorem z_minus_conj_eq : z - z_conj = -8 / 5 * Complex.i := by
  sorry

end z_minus_conj_eq_l58_58862


namespace equation_of_circle_find_k_intersection_l58_58517

-- Definition for Problem (I):
theorem equation_of_circle (a : ℝ) (r : ℝ) (x y : ℝ) :
  (x - a)^2 + (y - 4 * a)^2 = r^2 ∧
  (∀ a, (4 * a - 2) / (sqrt (2)) = r) ∧ 
  (1 + 4 * 0 - 2 = 0) ->
  x^2 + y^2 = 2 := 
  by sorry

-- Definition for Problem (II):
theorem find_k_intersection (k : ℝ) (x y : ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  ((kx - y + 3 = 0) ∧ ((fst A - fst B)^2 + (snd A + snd B)^2 = 7)) ∧
  (M = A + B) ∧ ((fst M)^2 + (snd M)^2 = 2) ->
  (k = sqrt 17) ∨ (k = - sqrt 17) := 
  by sorry

end equation_of_circle_find_k_intersection_l58_58517


namespace aitana_jayda_total_spending_l58_58796

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l58_58796


namespace problem_conditions_l58_58114

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := 2 * a n * (Real.log (a n) + 1) + 1

theorem problem_conditions (n : ℕ) (h_ge_2 : n ≥ 2) :
  (∀ n, (a (n+1) ≤ 2 * (a n)^2 + 1)) ∧
  (3 / 4 ≤ ∑ i in Finset.range n.succ, 1 / (a i + 1) ∧ ∑ i in Finset.range n.succ, 1 / (a i + 1) < 1) ∧
  (∑ i in Finset.range n, Real.log (a i + 1) ≤ (2^n - 1) * Real.log 2) :=
by
  sorry

end problem_conditions_l58_58114


namespace infinite_perpendiculars_from_point_on_line_unique_perpendicular_from_point_outside_line_unique_parallel_from_point_outside_line_l58_58803

-- Problem 1: Infinitely many perpendiculars from a point on a line
theorem infinite_perpendiculars_from_point_on_line (A : Point) (ℓ : Line) (hA : A ∈ ℓ) :
  ∃ (α : Type) (l : α → Line), ∀ (x y : α), x ≠ y → l x ⊥ ℓ ∧ l y ⊥ ℓ ∧ l x ≠ l y := 
sorry

-- Problem 2: Only one perpendicular from a point outside the line
theorem unique_perpendicular_from_point_outside_line (P : Point) (ℓ : Line) (hP : P ∉ ℓ) :
  ∃! Q : Line, Q ⊥ ℓ ∧ P ∈ Q := 
sorry

-- Problem 3: Only one parallel from a point outside the line
theorem unique_parallel_from_point_outside_line (P : Point) (ℓ : Line) (hP : P ∉ ℓ) :
  ∃! Q : Line, Q ∥ ℓ ∧ P ∈ Q := 
sorry

end infinite_perpendiculars_from_point_on_line_unique_perpendicular_from_point_outside_line_unique_parallel_from_point_outside_line_l58_58803


namespace value_of_polynomial_at_3_l58_58339

theorem value_of_polynomial_at_3 :
  let f := λ x : ℝ, -6 * x^4 + 5 * x^3 + 2 * x + 6 in
  f 3 = -115 :=
by
  let f := λ x : ℝ, -6 * x^4 + 5 * x^3 + 2 * x + 6
  have h : f 3 = -115 := sorry
  exact h

end value_of_polynomial_at_3_l58_58339


namespace num_ways_equals_fib_l58_58952

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
    fibonacci (n - 1) + fibonacci (n - 2)

def number_of_ways (n : ℕ) : ℕ :=
  if n = 1 then 0 else
  if n = 2 then 1 else
  if n = 3 then 1 else
    number_of_ways (n - 1) + number_of_ways (n - 2)

theorem num_ways_equals_fib (n : ℕ) (h : n ≥ 2) :
  number_of_ways n = fibonacci (n - 1) :=
sorry

end num_ways_equals_fib_l58_58952


namespace small_circle_radius_l58_58021

def design_width := 2 -- Total width of the design in cm
def R := Real.sqrt 2 -- Radius of the large arcs
def r := 2 - Real.sqrt 2 -- Radius of the small circle

theorem small_circle_radius :
  (r = 2 - Real.sqrt 2) :=
sorry

end small_circle_radius_l58_58021


namespace fill_in_operations_l58_58081

noncomputable def exists_operations : Prop :=
  ∃ (op1 op2 op3 : ℝ → ℝ → ℝ), 
    (op1 = (+) ∨ op1 = (-) ∨ op1 = (•) ∨ op1 = (/)) ∧
    (op2 = (+) ∨ op2 = (-) ∨ op2 = (•) ∨ op2 = (/)) ∧
    (op3 = (+) ∨ op3 = (-) ∨ op3 = (•) ∨ op3 = (/)) ∧
    op1 6 (op2 3 (op3 2 12)) = 24

theorem fill_in_operations : exists_operations :=
sorry

end fill_in_operations_l58_58081


namespace number_of_pentominoes_with_two_lines_of_symmetry_l58_58164

-- Define the set of fifteen pentominoes
def pentomino : Type := sorry  -- Replace with an appropriate definition if necessary

-- Define the property of having exactly two lines of reflectional symmetry
def exactly_two_lines_of_symmetry (p : pentomino) : Prop := sorry  -- Replace with an appropriate condition

-- Given set of fifteen pentominoes
def pentominoes : list pentomino := sorry  -- Replace with the actual list of 15 pentominoes

-- Main theorem
theorem number_of_pentominoes_with_two_lines_of_symmetry :
  (list.filter exactly_two_lines_of_symmetry pentominoes).length = 3 :=
begin
  sorry
end

end number_of_pentominoes_with_two_lines_of_symmetry_l58_58164


namespace ms_hatcher_total_students_l58_58626

theorem ms_hatcher_total_students :
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  third_graders + fourth_graders + fifth_graders = 70 :=
by 
  let third_graders := 20
  let fourth_graders := 2 * third_graders
  let fifth_graders := third_graders / 2
  show third_graders + fourth_graders + fifth_graders = 70
  sorry

end ms_hatcher_total_students_l58_58626


namespace min_distance_PQ_l58_58973

def AC : ℝ := 16
def BD : ℝ := 30

noncomputable def PQ_min (N P Q : ℝ) (hN_on_AB : N ∈ (set.interval 0 30)) 
                        (hP_perp_AC : N = P) (hQ_perp_BD : N = Q) : ℝ :=
  let AE := AC / 2
  let BE := BD / 2
  let min_PQ := 7/-
  7

theorem min_distance_PQ :
  ∃ (ABCD : Type) [hs : nonempty ABCD] (h : AC = 16 ∧ BD = 30)
  (N P Q : ABCD) (hN_on_AB : N ∈ (set.interval 0 30))
  (hP_perp_AC : P ≠ N) (hQ_perp_BD : Q ≠ N),
  PQ_min N P Q hN_on_AB hP_perp_AC hQ_perp_BD = 7
:=
begin
  sorry
end

end min_distance_PQ_l58_58973


namespace triangle_area_from_prism_l58_58782

theorem triangle_area_from_prism (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 30)
  (h3 : c * a = 32) :
  let d1 := Real.sqrt (a^2 + b^2)
      d2 := Real.sqrt (b^2 + c^2)
      d3 := Real.sqrt (c^2 + a^2)
      s := (d1 + d2 + d3) / 2 in
  (Real.sqrt (s * (s - d1) * (s - d2) * (s - d3))) = 25 := sorry

end triangle_area_from_prism_l58_58782


namespace simplify_polynomial_l58_58651

open Polynomial -- Open the polynomial namespace

def p1 : Polynomial ℤ := 2 * X ^ 6 + X ^ 5 + 3 * X ^ 4 + X ^ 2 + 15
def p2 : Polynomial ℤ := X ^ 6 + 2 * X ^ 5 - X ^ 4 + X ^ 3 + 17

theorem simplify_polynomial : p1 - p2 = X ^ 6 - X ^ 5 + 4 * X ^ 4 - X ^ 3 + X ^ 2 - 2 :=
by
  -- Solution is omitted; we only need the statement
  sorry

end simplify_polynomial_l58_58651


namespace prism_slicing_surface_area_l58_58786

noncomputable def surface_area_solid_BGPR : ℝ := 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2

theorem prism_slicing_surface_area (h_prism : ∀ (ABCDGH : Type) (P Q R : Point), 
  height ABCDGH = 20 ∧ 
  ∀ T : EquilateralTriangle, side_length T = 10 ∧ 
  P = midpoint (AB : edge ABCDGH), 
  Q = midpoint (BG : edge ABCDGH), 
  R = midpoint (DG : edge ABCDGH) → 
  surface_area_solid_BGPR = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2) :
    surface_area_solid_BGPR = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
sorry

end prism_slicing_surface_area_l58_58786


namespace isosceles_triangle_CDF_l58_58967

theorem isosceles_triangle_CDF
  (A B C D E F : Point)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : MeetAt CD BA E)
  (h3 : TangentLineThrough D (Circle A D E) MeetsLine CB F) :
  IsIsoscelesTriangle C D F :=
sorry

end isosceles_triangle_CDF_l58_58967


namespace equilateral_triangle_center_l58_58966

variables {A B C D E F P : Type}

-- Let ABC be a triangle with concurrent cevians AD, BE, and CF at point P.
axiom triangle_ABC : Triangle A B C
axiom concurrent_cevians : Concurrent (Cevian triangle_ABC D) (Cevian triangle_ABC E) (Cevian triangle_ABC F) at P

-- Assume PDCE, PEAF, and PFBD each have a circumcircle and an incircle.
axiom PDCE_circumcircle : Cyclic Quadrilateral P D C E
axiom PDCE_incircle : Incircle Quadrilateral P D C E

axiom PEAF_circumcircle : Cyclic Quadrilateral P E A F
axiom PEAF_incircle : Incircle Quadrilateral P E A F

axiom PFBD_circumcircle : Cyclic Quadrilateral P F B D
axiom PFBD_incircle : Incircle Quadrilateral P F B D

-- Prove that triangle ABC is equilateral and P coincides with the center of the triangle.
theorem equilateral_triangle_center (triangle_ABC : Triangle A B C)
  (concurrent_cevians : Concurrent (Cevian triangle_ABC D) (Cevian triangle_ABC E) (Cevian triangle_ABC F) at P)
  (PDCE_circumcircle : Cyclic Quadrilateral P D C E) (PDCE_incircle : Incircle Quadrilateral P D C E)
  (PEAF_circumcircle : Cyclic Quadrilateral P E A F) (PEAF_incircle : Incircle Quadrilateral P E A F)
  (PFBD_circumcircle : Cyclic Quadrilateral P F B D) (PFBD_incircle : Incircle Quadrilateral P F B D) :
  (Equilateral (Triangle A B C) ∧ Center (Triangle A B C) = P) := sorry

end equilateral_triangle_center_l58_58966


namespace new_ratio_books_to_clothes_l58_58687

-- Given initial conditions
def initial_ratio := (7, 4, 3)
def electronics_weight : ℕ := 12
def clothes_removed : ℕ := 8

-- Definitions based on the problem
def part_weight : ℕ := electronics_weight / initial_ratio.2.2
def initial_books_weight : ℕ := initial_ratio.1 * part_weight
def initial_clothes_weight : ℕ := initial_ratio.2.1 * part_weight
def new_clothes_weight : ℕ := initial_clothes_weight - clothes_removed

-- Proof of the new ratio
theorem new_ratio_books_to_clothes : (initial_books_weight, new_clothes_weight) = (7 * part_weight, 2 * part_weight) :=
sorry

end new_ratio_books_to_clothes_l58_58687


namespace blue_tshirts_per_pack_is_9_l58_58817

-- Definitions (conditions)
def total_white_tshirts(packs_white: Nat, tshirts_per_pack_white: Nat) : Nat :=
  packs_white * tshirts_per_pack_white

def total_blue_tshirts(total_tshirts: Nat, total_white_tshirts: Nat) : Nat :=
  total_tshirts - total_white_tshirts

def blue_tshirts_per_pack(total_blue_tshirts: Nat, packs_blue: Nat) : Nat :=
  total_blue_tshirts / packs_blue

-- Given values
def packs_white := 5
def packs_blue := 3
def tshirts_per_pack_white := 6
def total_tshirts := 57

-- Prove the number of blue t-shirts per pack is 9
theorem blue_tshirts_per_pack_is_9 : blue_tshirts_per_pack (total_blue_tshirts total_tshirts (total_white_tshirts packs_white tshirts_per_pack_white)) packs_blue = 9 :=
by
  sorry

end blue_tshirts_per_pack_is_9_l58_58817


namespace chemical_reaction_proof_l58_58822

theorem chemical_reaction_proof :
  ∀ (KOH NH4I : ℕ),
  (balanced_eq : "KOH + NH4I → KI + NH3 + H2O") →
  (init_KOH : KOH = 3) →
  (init_NH4I : NH4I = 2) →
  (limiting_reactant : NH4I_is_limiting) →
  (moles_of_H2O : H2O = 2) →
  (excess_KOH : remaining_KOH = 1) :=
by
  -- Definitions
  assume KOH NH4I : ℕ
  have balanced_eq : "KOH + NH4I → KI + NH3 + H2O" := sorry
  have init_KOH : KOH = 3 := sorry
  have init_NH4I : NH4I = 2 := sorry

  -- Theorem statements
  have limiting_reactant : NH4I_is_limiting := sorry
  have moles_of_H2O : H2O = 2 := sorry
  have excess_KOH : remaining_KOH = 1 := sorry

  -- Proof placeholder (to be replaced by actual proof steps)
  sorry

end chemical_reaction_proof_l58_58822


namespace suitcase_weight_l58_58963

def perfume_weight_ounces := 5 * 1.2 -- 5 bottles each 1.2 ounces
def soap_weight_ounces := 2 * 5     -- 2 bars each 5 ounces
def jam_weight_ounces := 2 * 8      -- 2 jars each 8 ounces
def total_ounces := perfume_weight_ounces + soap_weight_ounces + jam_weight_ounces
def ounces_to_pounds := total_ounces / 16
def items_weight_pounds := ounces_to_pounds + 4 -- add chocolate divided by total ounces
def initial_suitcase_weight := 5
def total_suitcase_weight := initial_suitcase_weight + items_weight_pounds

theorem suitcase_weight (initial_suitcase_weight = 5)
                       (perfume_weight_ounces = 5 * 1.2)
                       (soap_weight_ounces = 2 * 5)
                       (jam_weight_ounces = 2 * 8)
                       (total_ounces = perfume_weight_ounces + soap_weight_ounces + jam_weight_ounces)
                       (ounces_to_pounds = total_ounces / 16)
                       (items_weight_pounds = ounces_to_pounds + 4):
                       total_suitcase_weight = 11 := 
sorry

end suitcase_weight_l58_58963


namespace rectangle_length_l58_58555

noncomputable theory

def given_area : ℝ := 36.48
def given_width : ℝ := 6.08

theorem rectangle_length (A : ℝ) (W : ℝ) (L : ℝ) (hA : A = given_area) (hW : W = given_width) : L = A / W := by
  sorry

end rectangle_length_l58_58555


namespace maximize_profit_price_l58_58762

-- Definitions from the conditions
def initial_price : ℝ := 80
def initial_sales : ℝ := 200
def price_reduction_per_unit : ℝ := 1
def sales_increase_per_unit : ℝ := 20
def cost_price_per_helmet : ℝ := 50

-- Profit function
def profit (x : ℝ) : ℝ :=
  (x - cost_price_per_helmet) * (initial_sales + (initial_price - x) * sales_increase_per_unit)

-- The theorem statement
theorem maximize_profit_price : 
  ∃ x, (x = 70) ∧ (∀ y, profit y ≤ profit x) :=
sorry

end maximize_profit_price_l58_58762


namespace sum_of_first_28_natural_numbers_l58_58740

theorem sum_of_first_28_natural_numbers : 
  (28 * (28 + 1)) / 2 = 406 := 
by
  exact (28 * 29) / 2 = 406

end sum_of_first_28_natural_numbers_l58_58740


namespace quadratic_iff_alternating_sum_zero_l58_58445

-- Define the points on the graph of a nonlinear function f
variable {n : ℕ}
variable {f : ℝ → ℝ}
variable {P : Fin (n+1) → ℝ × ℝ}

-- Assume points are given on the graph of f
axiom P_on_graph (i : Fin (n+1)) : P i = (P i).1, f ((P i).1)

-- Define the slope between points Pi and Pi+1.
noncomputable def slope (i : Fin (n+1)) : ℝ :=
  if h : i.1 < n then ((f ((P ⟨i.1 + 1, Nat.lt_succ_of_lt i.is_lt⟩).1) - f ((P i).1)) / (((P ⟨i.1 + 1, Nat.lt_succ_of_lt i.is_lt⟩).1) - (P i).1))
  else ((f ((P ⟨0, Nat.zero_lt_succ n⟩).1) - f ((P i).1)) / (((P ⟨0, Nat.zero_lt_succ n⟩).1) - (P i).1))

-- Main theorem statement
theorem quadratic_iff_alternating_sum_zero :
  (∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c) ↔ (∃ P0 : ℝ × ℝ, (∀ i : Fin (n+1), P 0 ≠ P i) ∧ (∑ i in (Fin.range (n+1)), (-1) ^ (i : ℕ) * slope i = 0)) := sorry

end quadratic_iff_alternating_sum_zero_l58_58445


namespace mutually_exclusive_complementary_l58_58101

def set := {1, 2, 3, 4, 5}

def eventA (x : ℕ) : Prop := x ∈ {2, 4}
def eventB (x : ℕ) : Prop := x ∈ {1, 3, 5}

theorem mutually_exclusive_complementary :
  (∀ x, ¬ (eventA x ∧ eventB x)) ∧ (∀ x, eventA x ∨ eventB x) :=
by {
  sorry
}

end mutually_exclusive_complementary_l58_58101


namespace tea_leaves_costs_l58_58024

theorem tea_leaves_costs (a_1 b_1 a_2 b_2 : ℕ) (c_A c_B : ℝ) :
  a_1 * c_A = 4000 ∧ 
  b_1 * c_B = 8400 ∧ 
  b_1 = a_1 + 10 ∧ 
  c_B = 1.4 * c_A ∧ 
  a_2 + b_2 = 100 ∧ 
  (300 - c_A) * (a_2 / 2) + (300 * 0.7 - c_A) * (a_2 / 2) + 
  (400 - c_B) * (b_2 / 2) + (400 * 0.7 - c_B) * (b_2 / 2) = 5800 
  → c_A = 200 ∧ c_B = 280 ∧ a_2 = 40 ∧ b_2 = 60 := 
sorry

end tea_leaves_costs_l58_58024


namespace point_on_line_segment_l58_58975

-- Definition of the variables and conditions
variables (A B P : Point) (ratio AP PB : ℝ)
axiom h : AP / PB = 3 / 2

-- Required constants t and u
def t : ℝ := 2 / 5
def u : ℝ := 3 / 5

-- The goal to be proven
theorem point_on_line_segment (h : AP / PB = 3 / 2) :
  P = t * A + u * B :=
sorry

end point_on_line_segment_l58_58975


namespace fiona_pairs_l58_58848

theorem fiona_pairs : Nat.choose 12 2 = 66 := by
  sorry

end fiona_pairs_l58_58848


namespace remainder_correct_l58_58729

def x : ℝ := 96.12 * 24.99999999999905
def y : ℝ := 24.99999999999905
def quotient : ℝ := 96
def decimal_part : ℝ := 0.12

theorem remainder_correct : (x / y - quotient) * y = 3 := by
  -- Definitions per problem conditions
  let div_result := 96.12
  let y := 24.99999999999905
  let r := 0.12 * y
  -- Calculation according to the solution
  have h1 : x = y * (div_result - decimal_part) := sorry
  have h2 : (x / y - quotient) * y = r := sorry
  
  -- Final result
  show (x / y - quotient) * y = 3 from sorry

end remainder_correct_l58_58729


namespace B_gain_correct_l58_58182

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_of_B : ℝ :=
  let principal : ℝ := 3150
  let interest_rate_A_to_B : ℝ := 0.08
  let annual_compound : ℕ := 1
  let time_A_to_B : ℝ := 3

  let interest_rate_B_to_C : ℝ := 0.125
  let semiannual_compound : ℕ := 2
  let time_B_to_C : ℝ := 2.5

  let amount_A_to_B := compound_interest principal interest_rate_A_to_B annual_compound time_A_to_B
  let amount_B_to_C := compound_interest principal interest_rate_B_to_C semiannual_compound time_B_to_C

  amount_B_to_C - amount_A_to_B

theorem B_gain_correct : gain_of_B = 282.32 :=
  sorry

end B_gain_correct_l58_58182


namespace pq_parallel_ab_l58_58489

theorem pq_parallel_ab
  {O1 O2 O : Type} 
  (tangent_to_each_other : extern_tangent O1 O2) 
  (externally_tangent_to_O : extern_tangent O1 O ∧ extern_tangent O2 O) 
  (AB_common_tangent_and_chord : common_tangent_and_chord O1 O2 O AB) 
  (CD_EF_internal_tangents : internal_tangents CD EF O1 O2 AB intersects C E) 
  (D_F_in_O_side_of_AB : DF_in_O_side_of_AB D F O O1 O2 AB) 
  (EF_AD_intersects_P : intersects EF AD P) 
  (CD_BF_intersects_Q : intersects CD BF Q) :
  parallel PQ AB :=
by { sorry }

end pq_parallel_ab_l58_58489


namespace max_min_values_find_cos_2x0_l58_58897

-- Define the function f(x)
def f (x : ℝ) := 2 * cos x * (sqrt 3 * sin x + cos x) - 1

-- Define the intervals
def interval1 : set ℝ := set.Icc 0 (π / 2)
def interval2 : set ℝ := set.Icc (π / 4) (π / 2)

theorem max_min_values :
  (∀ x ∈ interval1, f x ≤ 2 ∧ f x ≥ -1)
  ∧ (∃ x_max ∈ interval1, f x_max = 2)
  ∧ (∃ x_min ∈ interval1, f x_min = -1) :=
by
  sorry

theorem find_cos_2x0 (x0 : ℝ) (hx0 : x0 ∈ interval2) (hfx0 : f x0 = 6 / 5) :
  cos (2 * x0) = (3 - 4 * sqrt 3) / 10 :=
by
  sorry

end max_min_values_find_cos_2x0_l58_58897


namespace probability_of_both_defective_given_one_defective_l58_58892

-- Definitions based on conditions in part a)
def totalProducts : ℕ := 6
def defectiveProducts : ℕ := 2
def selectedProducts : ℕ := 2

theorem probability_of_both_defective_given_one_defective (h : selectedProducts = 2) 
(h1 : 1 ∈ ({i : ℕ | i ≤ totalProducts}.filter (λ x, x <= defectiveProducts))) : 
  (∃ (n : ℚ), n = 1/15) :=
by
  -- Sorry is used to skip the proof step
  sorry

end probability_of_both_defective_given_one_defective_l58_58892


namespace find_x_pow_8_l58_58548

theorem find_x_pow_8 (x : ℂ) (h : x + x⁻¹ = real.sqrt 2) : x^8 = 1 := 
sorry

end find_x_pow_8_l58_58548


namespace major_premise_incorrect_l58_58173

theorem major_premise_incorrect (a b : ℝ) (h : a > b) : ¬ (a^2 > b^2) :=
by {
  sorry
}

end major_premise_incorrect_l58_58173


namespace radius_pyramid_l58_58290

-- Define the given conditions as constants
constant TH : ℝ := 4
constant ctg_HCP : ℝ := sqrt 2
constant TK_CK : ℝ := 4
constant V_pyramid : ℝ := 16 / 3

-- Define the base of the pyramid as consisting of equivalent area triangles
constant base_area : ℝ
constant height : ℝ

-- Given volume relation
axiom volume_relation : 1 / 3 * base_area * height = V_pyramid

-- Radius of the largest sphere that can fit inside the pyramid
def radius_of_largest_inscribed_sphere (A h : ℝ) : ℝ :=
  16 / A

-- The target theorem to prove
theorem radius_pyramid :
    let h := (16:ℝ) / base_area in
    radius_of_largest_inscribed_sphere base_area h = 2 - sqrt 2 := by
  sorry

end radius_pyramid_l58_58290


namespace min_divisors_f_l58_58242

def f : ℕ → ℕ → ℕ
noncomputable def L : ℕ := Nat.lcm (List.range 22).tail (1) 22

open Nat

theorem min_divisors_f {f : ℕ} (h₁ : ∀ m n: ℕ, 1 ≤ m → m ≤ 22 → 1 ≤ n → n ≤ 22 → m * n ∣ f m + f n) : 
  ∃ (d : ℕ), d = 2016 := sorry

end min_divisors_f_l58_58242


namespace sum_of_x_and_y_l58_58503

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l58_58503


namespace round_down_l58_58649

theorem round_down (x : ℝ) (hx : x = 7293847.2635142) : Int.round x = 7293847 := by
  rw [hx]
  norm_num
  sorry

end round_down_l58_58649


namespace initial_students_19_l58_58289

theorem initial_students_19 (n W : ℕ) 
  (h1 : W = n * 15) 
  (h2 : W + 7 = (n + 1) * 14.6) : 
  n = 19 := 
by
  sorry

end initial_students_19_l58_58289


namespace find_m_given_solution_l58_58926

theorem find_m_given_solution (m x y : ℚ) (h₁ : x = 4) (h₂ : y = 3) (h₃ : m * x - y = 4) : m = 7 / 4 :=
by
  sorry

end find_m_given_solution_l58_58926


namespace sin_value_l58_58174

-- Define the conditions
def a_in_range (a : ℝ) : Prop := 0 < a ∧ a < π / 2
def cos_condition (a : ℝ) : Prop := cos (a + π / 6) = 4 / 5

-- Define the proof problem
theorem sin_value (a : ℝ) (h1 : a_in_range a) (h2 : cos_condition a) :
  sin (2 * a + π / 3) = 24 / 25 :=
sorry

end sin_value_l58_58174


namespace area_enclosed_eq_30_l58_58663

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem area_enclosed_eq_30 {c : ℝ} (h : ∫ x in -c/2..c/2, f x - c = 30) : c = 8 :=
sorry

end area_enclosed_eq_30_l58_58663


namespace eccentricity_of_ellipse_l58_58520

variables {a b c : ℝ}
variable (x : ℝ)
variable (y : ℝ)
variable (e : ℝ)

def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def geometric_sequence (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (2 * b, a, c).isGeomSeq

def eccentricity (c a e : ℝ) : Prop :=
  e = c / a

theorem eccentricity_of_ellipse (a b c e : ℝ)
  (h_ellipse : ellipse a b x y)
  (h_geo : geometric_sequence a b c)
  (h_eccentricity : eccentricity c a e) :
  e = Real.sqrt (2) / 2 :=
sorry

end eccentricity_of_ellipse_l58_58520


namespace sum_intersection_points_l58_58067

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y + 1)^2 - 3

theorem sum_intersection_points :
  ∑ (x, y) in { (x, y) | parabola1 x = y ∧ parabola2 y = x }, (x + y) = 4 :=
by sorry

end sum_intersection_points_l58_58067


namespace union_of_M_and_N_l58_58618

def M : Set ℝ := {x | x^2 - 6 * x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5 * x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by
  sorry

end union_of_M_and_N_l58_58618


namespace angle_measure_l58_58347

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l58_58347


namespace domain_of_f_l58_58297

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ Real.log (x + 1) ≠ 0 ∧ 4 - x^2 ≥ 0} =
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l58_58297


namespace lines_perpendicular_m_eq_half_l58_58560

theorem lines_perpendicular_m_eq_half (m : ℝ) :
    (∀ x y, mx + y - 1 = 0) ∧ (∀ x y, x + (m - 1)y + 2 = 0) ∧ 
    (∀ x y, (-m) * (-(1 / (m - 1))) = -1) → 
    m = 1 / 2 :=
sorry

end lines_perpendicular_m_eq_half_l58_58560


namespace polynomial_odd_coefficients_count_l58_58613

open Polynomial

-- Define the function for odd coefficients count.
def oddCoefficientsCount (n : ℕ) : ℕ :=
  let binaryRep := n.binaryDigits
  let segments := binaryRep.segmentLengths
  segments.foldr (fun ai acc => acc * (2^(ai + 2) - (-1)^ai) / 3) 1

-- Main theorem statement.
theorem polynomial_odd_coefficients_count (n : ℕ) (hn : n > 0) :
  oddCoefficientsCount n = 
  let binaryRep := n.binaryDigits
  let segments := binaryRep.segmentLengths
  segments.foldr (fun ai acc => acc * (2^(ai + 2) - (-1)^ai) / 3) 1 := 
  sorry

end polynomial_odd_coefficients_count_l58_58613


namespace min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l58_58132

section ProofProblem

theorem min_value_a_cube_plus_b_cube {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

theorem no_exist_2a_plus_3b_eq_6 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hfra : 1/a + 1/b = Real.sqrt (a * b)) :
  ¬ (2 * a + 3 * b = 6) :=
sorry

end ProofProblem

end min_value_a_cube_plus_b_cube_no_exist_2a_plus_3b_eq_6_l58_58132


namespace proof_problem_l58_58909

variables (α β : Plane) (m n : Line)

theorem proof_problem (h1 : Parallel α β) (h2 : Perpendicular n α) (h3 : Perpendicular m β) :
  Parallel m n :=
begin
  sorry
end

end proof_problem_l58_58909


namespace max_revenue_l58_58570

-- Define the conditions and the function relationship
def revenue (x : ℝ) : ℝ := (1 + 50 * (x - 0.8)^2) * (x - 0.5)

-- Assume the price that maximizes revenue
def price := 0.64

-- Theorem statement
theorem max_revenue : revenue price = 0.3192 := by
  sorry

end max_revenue_l58_58570


namespace Dave_tiles_210_square_feet_l58_58444

theorem Dave_tiles_210_square_feet
  (ratio_charlie_dave : ℕ := 5 / 7)
  (total_area : ℕ := 360)
  : ∀ (work_done_by_dave : ℕ), work_done_by_dave = 210 :=
by
  sorry

end Dave_tiles_210_square_feet_l58_58444


namespace sum_telescope_series_l58_58065

theorem sum_telescope_series :
  (∑ n in Finset.range 1023 \ Finset.singleton 0, (1 : ℝ) / (n + 1)^2 * (sqrt n) + (n^2) * (sqrt (n + 1))) = 31 / 32 :=
by
  have h :
    ∀ n ∈ Finset.range 1023 \ Finset.singleton 0,
      (1 : ℝ) / ((n + 1) ^ 2 * sqrt n + (n ^ 2) * sqrt (n + 1)) = (1 / sqrt n - 1 / sqrt (n + 1)) :=
    sorry
  calc
    (∑ n in Finset.range 1023 \ Finset.singleton 0, 1 / (n + 1)^2 * sqrt n + (n^2) * sqrt (n + 1))
        = ∑ n in Finset.range 1023 \ Finset.singleton 0, (1 / sqrt n - 1 / sqrt (n + 1)) : by rw h
    ... = 1 - 1 / sqrt 1024 : by sorry
    ... = 1 - 1 / 32 : by norm_num
    ... = 31 / 32 : by norm_num

end sum_telescope_series_l58_58065


namespace cos_of_angle_at_point_P_l58_58928

def P : (Int × Int) := (-3, 4)

def distance (x y : ℝ) : ℝ := Real.sqrt (x ^ 2 + y ^ 2)

def cos_alpha (x y : ℝ) : ℝ := x / distance x y

theorem cos_of_angle_at_point_P :
  cos_alpha (-3) 4 = -3 / 5 :=
by
  rw [cos_alpha, distance]
  sorry

end cos_of_angle_at_point_P_l58_58928


namespace complex_quadrant_l58_58932

theorem complex_quadrant (z : ℂ) (h : z * complex.I = 2 - complex.I) : 
    ((z.re < 0) ∧ (z.im < 0)) :=
by
  sorry

end complex_quadrant_l58_58932


namespace average_chemistry_mathematics_l58_58741

-- Define the conditions 
variable {P C M : ℝ} -- Marks in Physics, Chemistry, and Mathematics

-- The given condition in the problem
theorem average_chemistry_mathematics (h : P + C + M = P + 130) : (C + M) / 2 = 65 := 
by
  -- This will be the main proof block (we use 'sorry' to omit the actual proof)
  sorry

end average_chemistry_mathematics_l58_58741


namespace emma_wraps_in_6_hours_l58_58829

noncomputable def wrapping_time : ℝ :=
  let E := 6 in
  let T := 8 in
  let together_hours := 2 in
  let emma_alone_hours := 2.5 in
  
  -- Define the rates of work per hour
  let emma_rate := 1 / E in
  let troy_rate := 1 / T in

  -- Total work done by Emma and Troy together, and Emma alone
  let total_together_work := together_hours * (emma_rate + troy_rate) in
  let total_emma_work := emma_alone_hours * emma_rate in

  -- Combined work should sum to 1 (complete task)
  total_together_work + total_emma_work

theorem emma_wraps_in_6_hours :
  wrapping_time = 1 := by
  sorry

end emma_wraps_in_6_hours_l58_58829


namespace proportionality_cube_and_fourth_root_l58_58176

variables (x y z : ℝ) (k j m n : ℝ)

theorem proportionality_cube_and_fourth_root (h1 : x = k * y^3) (h2 : y = j * z^(1/4)) : 
  ∃ m : ℝ, ∃ n : ℝ, x = m * z^n ∧ n = 3/4 :=
by
  sorry

end proportionality_cube_and_fourth_root_l58_58176
