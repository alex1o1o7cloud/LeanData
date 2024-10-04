import Mathlib

namespace B_complete_work_in_6_days_l524_524126

variable {A_days : ℕ} (A_days_eq : A_days = 9)
variable {B_eff : ℚ} (B_eff_eq : B_eff = 1.5)
variable {B_days : ℕ} (B_days_eq : B_days = 6)

theorem B_complete_work_in_6_days (hA : A_days_eq) (hB_eff : B_eff_eq) :
  B_days = 6 :=
sorry

end B_complete_work_in_6_days_l524_524126


namespace ducks_in_the_marsh_l524_524547

-- Define the conditions
def number_of_geese : ℕ := 58
def total_number_of_birds : ℕ := 95
def number_of_ducks : ℕ := total_number_of_birds - number_of_geese

-- Prove the conclusion
theorem ducks_in_the_marsh : number_of_ducks = 37 := by
  -- subtraction to find number_of_ducks
  sorry

end ducks_in_the_marsh_l524_524547


namespace range_a_range_x_min_value_ab_l524_524276

noncomputable def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem range_a (a : ℝ) :
  (3 - 2 * real.sqrt 2 < a) ↔ (∀ x : ℝ, 2 < x ∧ x < 5 → quadratic a (-a - 1) x > 0) :=
sorry

theorem range_x (x : ℝ) :
  ((1 - real.sqrt 17) / 4 < x ∧ x < (1 + real.sqrt 17) / 4) ↔ (∀ a : ℝ, -2 ≤ a ∧ a ≤ -1 → quadratic a (-a - 1) x > 0) :=
sorry

theorem min_value_ab (b : ℝ) (hb : b > 0) :
  ((a + 2) / b ≥ 1) ↔ (∀ x : ℝ, quadratic a b x ≥ 0) :=
sorry

end range_a_range_x_min_value_ab_l524_524276


namespace find_a_l524_524897

theorem find_a (b a : ℤ) (h1 : ∀ x, f (f_inv x) = x) (h2 : ∀ y, f_inv (f y) = y)
  (h3 : f (-4) = a) (h4 : f_inv (-4) = a)
  (f := fun x : ℤ => 2 * x + b)
  (f_inv := fun x : ℤ => (x - b) / 2) :
  a = -4 := by
  -- proof goes here
  sorry

end find_a_l524_524897


namespace find_alpha_polar_eq_l524_524716

noncomputable def point := {x : ℝ, y : ℝ}

def line_parametric (α : ℝ) (t : ℝ) : point :=
  {x := 2 + t * Real.cos α, y := 1 + t * Real.sin α}

def condition_PA_PB (α : ℝ) (t1 t2 : ℝ) :=
  let A := line_parametric α t1
  let B := line_parametric α t2
  let PA := Real.sqrt ((A.x - 2)^2 + (A.y - 1)^2)
  let PB := Real.sqrt ((B.x - 2)^2 + (B.y - 1)^2)
  PA * PB = 4

theorem find_alpha (α : ℝ) (t1 t2 : ℝ) (H : condition_PA_PB α t1 t2) :
  α = 3 * Real.pi / 4 :=
sorry

def polar_equation (α : ℝ) (ρ θ : ℝ) :=
  ρ * (Real.cos θ + Real.sin θ) = 3

theorem polar_eq (θ : ℝ) :
  polar_equation (3 * Real.pi / 4) (3 / (Real.cos θ + Real.sin θ)) θ :=
sorry

end find_alpha_polar_eq_l524_524716


namespace hyperbola_eccentricity_l524_524524

theorem hyperbola_eccentricity
  (a b m : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (PA_perpendicular_to_l2 : (b/a * m) / (m + a) * (-b/a) = -1)
  (PB_parallel_to_l2 : (b/a * m) / (m - a) = -b/a) :
  (∃ e, e = 2) :=
by sorry

end hyperbola_eccentricity_l524_524524


namespace quad_with_axis_of_symmetry_is_special_l524_524626

-- Assuming required definitions for quadrilateral, axis of symmetry, isosceles trapezoid, and rectangle
variables {P : Type*} [EuclideanGeometry P]

-- Definition of an axis of symmetry
def axis_of_symmetry (q : quadrilateral P) (l : line P) : Prop :=
  ∃ (A B C D : P), q = ⟨A, B, C, D⟩ ∧ symmetric_about_line (A, B, C, D) l

-- Definition of isosceles trapezoid
def isosceles_trapezoid (q : quadrilateral P) : Prop :=
  ∃ (A B C D : P), q = ⟨A, B, C, D⟩ ∧ trapezoid ⟨A, B, C, D⟩ ∧ (|AB| = |CD|)

-- Definition of rectangle
def rectangle (q : quadrilateral P) : Prop :=
  ∃ (A B C D : P), q = ⟨A, B, C, D⟩ ∧ parallelogram ⟨A, B, C, D⟩ ∧ right_angles ⟨A, B, C, D⟩

-- The main theorem statement
theorem quad_with_axis_of_symmetry_is_special
  {q : quadrilateral P} {l : line P}
  (h1 : axis_of_symmetry q l)
  (h2 : ¬ passes_through_diagonals q l) :
  isosceles_trapezoid q ∨ rectangle q ∨ symmetric_with_diagonal q :=
sorry  -- Proof placeholder

end quad_with_axis_of_symmetry_is_special_l524_524626


namespace jellybean_count_l524_524214

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l524_524214


namespace smallest_positive_angle_l524_524908

theorem smallest_positive_angle (theta : ℝ) (h_theta : theta = -2002) :
  ∃ α : ℝ, 0 < α ∧ α < 360 ∧ ∃ k : ℤ, theta = α + k * 360 ∧ α = 158 := 
by
  sorry

end smallest_positive_angle_l524_524908


namespace product_of_third_sides_l524_524869

-- Definition of the problem
variables (T1 T2 : Type) (AB BC : ℝ) (DE DF : ℝ) (AC : ℝ) (z1 z2 : ℝ)

-- Given conditions
def triangle_area (a b : ℝ) : ℝ := (a * b) / 2

-- Conditions for T1 and T2
def area_T1 : triangle_area (2 * AB) (2 * BC) = 2 := by sorry
def area_T2 : triangle_area DE DF = 3 := by sorry

-- Sides proportions between T1 and T2
def side_condition1 : 2 * AB = DE := by sorry
def side_condition2 : 2 * BC = DF := by sorry

-- Pythagorean theorem for right triangles
def pythagorean_T1 : (2 * AB)^2 + (2 * BC)^2 = z1^2 := by sorry
def pythagorean_T2 : (DE)^2 + (DF)^2 = z2^2 := by sorry

-- Proof statement
theorem product_of_third_sides : (z1 * z2)^2 = 676 / 9 := by sorry

end product_of_third_sides_l524_524869


namespace sum_y_intercepts_l524_524553

theorem sum_y_intercepts : 
  let points1 := (finset.range 2000).map (λ k, (-1, k + 1)) in 
  let points2 := (finset.range 2000).map (λ k, (1, k + 1)) in
  let intercept_sum := (finset.range 2000).sum (λ k, (k + 1) + (some_fn k) / 2) in
  intercept_sum = 2001000 :=
sorry

end sum_y_intercepts_l524_524553


namespace percent_prime_divisible_by_5_l524_524580

def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]
def primes_divisible_by_5 := [p in primes_less_than_20 | p % 5 = 0]

theorem percent_prime_divisible_by_5 :
  (primes_divisible_by_5.length / primes_less_than_20.length * 100 = 12.5) :=
by
  sorry

end percent_prime_divisible_by_5_l524_524580


namespace find_angle_opposite_side_c_l524_524791

open Real

theorem find_angle_opposite_side_c
  (a b c : ℝ)
  (h : (a + 2 * b + c) * (a + b - c - 2) = 4 * a * b) :
  ∠C = 60 :=
begin
  sorry
end

end find_angle_opposite_side_c_l524_524791


namespace line_circle_no_intersection_l524_524343

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524343


namespace solve_for_m_l524_524766

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, real.sqrt 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)

-- Define the dot product for two-dimensional vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the length (norm) of a two-dimensional vector
def norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- The given condition on the projection
def projection_condition (m : ℝ) : Prop :=
  dot_product vec_a (vec_b m) / norm vec_a = 3

-- The theorem to prove
theorem solve_for_m : ∃ m : ℝ, projection_condition m ∧ m = real.sqrt 3 :=
sorry

end solve_for_m_l524_524766


namespace donkey_always_eats_hay_l524_524520

variable (EatsOats EatsHay : Prop)

-- Conditions
variable (d_eats_oats h_eats_oats c_eats_hay : Prop)
variable (h_eats_same_as_c : Prop)
variable (d_eats_not_same_as_c_when_h_eats_oats : Prop)
variable (d_eats_same_as_h_when_c_eats_hay : Prop)

-- If the donkey eats oats, then the horse eats the same as the cow
axiom cond1 : d_eats_oats → h_eats_same_as_c

-- If the horse eats oats, then the donkey eats what the cow does not eat
axiom cond2 : h_eats_oats → d_eats_not_same_as_c_when_h_eats_oats

-- If the cow eats hay, then the donkey eats the same as the horse
axiom cond3 : c_eats_hay → d_eats_same_as_h_when_c_eats_hay

-- Conclusion: The donkey always eats from the feeder with hay
theorem donkey_always_eats_hay : (EatsOats = d_eats_oats ∧ EatsHay = c_eats_hay ∧ cond1 ∧ cond2 ∧ cond3) → EatsHay := 
by sorry

end donkey_always_eats_hay_l524_524520


namespace scalar_product_of_trisection_points_l524_524271

theorem scalar_product_of_trisection_points (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] (R : ℝ) 
  (h₀ : dist O A = R) (h₁ : dist O B = R) (h₂ : dist A B = sqrt (3) * R) : 
  let OA := sorry in
  let AB := sorry in
  OA • AB = - (3 / 2) * R ^ 2 := sorry

end scalar_product_of_trisection_points_l524_524271


namespace prop1_prop2_prop3_prop4_l524_524622

-- Definitions of vectors and operations
structure Vec2 :=
  (m : ℝ)
  (n : ℝ)

def vec_star (a b : Vec2) : ℝ :=
  a.m * b.n - a.n * b.m

def vec_dot (a b : Vec2) : ℝ :=
  a.m * b.m + a.n * b.n

def vec_norm_sq (a : Vec2) : ℝ :=
  a.m^2 + a.n^2

-- Hypotheses and Propositions
variables (a b : Vec2) (λ : ℝ)

-- Propositions to be proven
theorem prop1 (H : a.m * b.n = a.n * b.m) : vec_star a b = 0 :=
sorry

theorem prop2 : vec_star a b ≠ vec_star b a :=
sorry

theorem prop3 : vec_star (Vec2.mk (λ * a.m) (λ * a.n)) b = λ * vec_star a b :=
sorry

theorem prop4 : vec_star a b ^ 2 + vec_dot a b ^ 2 = vec_norm_sq a * vec_norm_sq b :=
sorry

end prop1_prop2_prop3_prop4_l524_524622


namespace arithmetic_sequence_and_sum_correct_geometric_sequence_correct_main_proof_l524_524720

def a_is_arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d)

def b_is_geometric_sequence (b : ℕ → ℝ) : Prop := 
  ∃ q : ℝ, (∀ n : ℕ, b (n + 1) = b n * q)

theorem arithmetic_sequence_and_sum_correct :
  (∀ n, a n = 8 - 6 * n) → (∀ n, S n = -3 * n^2 + 5 * n) :=
sorry

theorem geometric_sequence_correct :
  (∀ n, b n = -2^n) :=
sorry

theorem main_proof :
  ∃ (a b : ℕ → ℝ) (S : ℕ → ℝ),
    a_is_arithmetic_sequence a ∧
    b_is_geometric_sequence b ∧
    a 1 = 2 ∧
    a 1 * a 2 = 40 ∧
    b 2 = a 2 ∧
    b 4 = a 4 ∧
    (∀ n, a n = 8 - 6 * n) ∧
    (∀ n, S n = -3 * n^2 + 5 * n) ∧
    (∀ n, b n = -2^n) :=
by {
  use λ n, 8 - 6 * n,
  use λ n, -2^n,
  use λ n, -3 * n^2 + 5 * n,
  split,
  { use -6, intros n, sorry },
  split,
  { use 2, intros n, sorry },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { intros n, norm_num },
  split,
  { intros n, norm_num },
  { intros n, norm_num }
}

end arithmetic_sequence_and_sum_correct_geometric_sequence_correct_main_proof_l524_524720


namespace problem_solution_l524_524776

theorem problem_solution (y : ℝ) (h : 8^y - 8^(y - 1) = 112) :
  (3 * y)^y = (7^(1 / 3))^7 :=
sorry

end problem_solution_l524_524776


namespace percentage_of_fruits_in_good_condition_l524_524157

theorem percentage_of_fruits_in_good_condition :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges := (15 / 100.0) * total_oranges
  let rotten_bananas := (8 / 100.0) * total_bananas
  let good_condition_oranges := total_oranges - rotten_oranges
  let good_condition_bananas := total_bananas - rotten_bananas
  let total_fruits := total_oranges + total_bananas
  let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
  let percentage_fruits_in_good_condition := (total_fruits_in_good_condition / total_fruits) * 100
  percentage_fruits_in_good_condition = 87.8 := sorry

end percentage_of_fruits_in_good_condition_l524_524157


namespace total_baseball_cards_l524_524009
-- Import the broad Mathlib library

-- The conditions stating the number of cards each person has
def melanie_cards : ℕ := 3
def benny_cards : ℕ := 3
def sally_cards : ℕ := 3
def jessica_cards : ℕ := 3

-- The theorem to prove the total number of cards they have is 12
theorem total_baseball_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by
  sorry

end total_baseball_cards_l524_524009


namespace find_natural_number_n_l524_524226

theorem find_natural_number_n (n : ℕ) : 
  (∃ n : ℕ, n = 256 ∧ (n⁷ / 8 = 2) ∧ n < 2217) :=
begin
  sorry
end

end find_natural_number_n_l524_524226


namespace Moe_has_least_l524_524655

variable (Money : Type)
variables (Bo Coe Flo Jo Moe Zoe : Money)
variable [LinearOrder Money]       -- All amounts of money are linearly ordered

-- Conditions as definitions in Lean 4
def Condition1 : Prop := Flo > Jo ∧ Flo > Bo
def Condition2 : Prop := Bo > Moe ∧ Coe > Moe
def Condition3 : Prop := Jo > Moe ∧ Jo < Bo
def Condition4 : Prop := Zoe > Coe ∧ Zoe < Flo
def Condition5 : Prop := Bo ≠ Coe ∧ Bo ≠ Flo ∧ Bo ≠ Jo ∧ Bo ≠ Moe ∧ Bo ≠ Zoe ∧ Coe ≠ Flo ∧ Coe ≠ Jo ∧ Coe ≠ Moe ∧ Coe ≠ Zoe ∧ Flo ≠ Jo ∧ Flo ≠ Moe ∧ Flo ≠ Zoe ∧ Jo ≠ Moe ∧ Jo ≠ Zoe ∧ Moe ≠ Zoe

-- The theorem stating Moe has the least amount of money given the above conditions
theorem Moe_has_least (hc1 : Condition1) (hc2 : Condition2) (hc3 : Condition3) (hc4 : Condition4) (hc5 : Condition5) : 
  ∀ x ∈ {Bo, Coe, Flo, Jo, Moe, Zoe}, Moe ≤ x :=
by
  sorry

end Moe_has_least_l524_524655


namespace imag_conjugate_of_z_l524_524272

-- Define the complex number
def z : ℂ := (ⅈ / (1 + ⅈ))

-- Define the conjugate of z
def conj_z : ℂ := conj z

-- Define the imaginary part of the conjugate
def imag_conj_z : ℝ := conj_z.im

-- The goal is to prove that the imaginary part of the conjugate of z is -1/2
theorem imag_conjugate_of_z : imag_conj_z = -1 / 2 :=
by
  -- skipping the proof for now
  sorry

end imag_conjugate_of_z_l524_524272


namespace isosceles_triangle_projection_angle_equality_l524_524444

theorem isosceles_triangle_projection_angle_equality
  (A B C M F H : Type)
  [IsoscelesTriangle A B C (λ AC BC, AC = BC)] -- Isosceles condition
  [OnSegment M A B (λ AM MB, AM = 2 * MB)] -- M divides AB in the ratio 2:1
  [Midpoint F B C] -- F is the midpoint of BC
  [OrthogonalProjection H M A F] -- H is the orthogonal projection of M onto AF
  : Angle B H F = Angle A B C := sorry

end isosceles_triangle_projection_angle_equality_l524_524444


namespace interest_earned_after_4_years_l524_524166

noncomputable def calculate_total_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let A := P * (1 + r) ^ t
  A - P

theorem interest_earned_after_4_years :
  calculate_total_interest 2000 0.12 4 = 1147.04 :=
by
  sorry

end interest_earned_after_4_years_l524_524166


namespace sum_of_divisors_24_l524_524957

theorem sum_of_divisors_24 : (∑ d in (finset.filter (λ n, 24 % n = 0) (finset.range 25)), d) = 60 :=
by
  sorry

end sum_of_divisors_24_l524_524957


namespace n_squared_plus_n_divisible_by_2_l524_524874

theorem n_squared_plus_n_divisible_by_2 (n : ℤ) : 2 ∣ (n^2 + n) :=
sorry

end n_squared_plus_n_divisible_by_2_l524_524874


namespace triangle_ratio_proof_l524_524807

/-
We have triangle ABC, where D is the midpoint of BC,
E divides AC such that AE:EC = 2:3, and F is on AD such that AF:FD = 3:2.
Our goal is to prove: (\frac{EF}{FB} + \frac{BF}{FD}) = \frac{1129}{345}
-/
theorem triangle_ratio_proof (
  (A B C D F E : Type*)
  [is_point A] [is_point B] [is_point C] [is_point D] [is_point F] 
  [is_point E]
  (AE_EC : ratio A E C 2 3) 
  (AF_FD : ratio A F D 3 2)
  (midpoint_D : midpoint D B C)
) : (ratio_measure E F B / ratio_measure B F D + ratio_measure E F B / ratio_measure F D B = 1129 / 345) := 
sorry

end triangle_ratio_proof_l524_524807


namespace geometric_sequence_sixth_term_l524_524893

/-- 
The statement: 
The first term of a geometric sequence is 1000, and the 8th term is 125. Prove that the positive,
real value for the 6th term is 31.25.
-/
theorem geometric_sequence_sixth_term :
  ∀ (a1 a8 a6 : ℝ) (r : ℝ),
    a1 = 1000 →
    a8 = 125 →
    a8 = a1 * r^7 →
    a6 = a1 * r^5 →
    a6 = 31.25 :=
by
  intros a1 a8 a6 r h1 h2 h3 h4
  sorry

end geometric_sequence_sixth_term_l524_524893


namespace problem_1_problem_2_find_vertex_l524_524306

def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 3 = 1

def vertex := (2 : ℝ, 0 : ℝ)

def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 12 / 7

def line_eq (x y : ℝ) (m : ℝ) : Prop :=
  y = x + m

def line_normal_vector := (1 : ℝ, -1 : ℝ)

theorem problem_1 (x y : ℝ) : circle_eq x y :=
sorry

theorem problem_2 (x y m : ℝ) (hl : line_eq x y m) : 
  ∃ (d : ℝ), (d = √2/2 ∨ d = 3*√2/2) ∧
  (∃ P1 P2 P3 : ℝ × ℝ, hyperbola P1.1 P1.2 ∧ hyperbola P2.1 P2.2 ∧ hyperbola P3.1 P3.2 ∧ 
    dist_to_line l P1.1 P1.2 = d ∧
    dist_to_line l P2.1 P2.2 = d ∧
    dist_to_line l P3.1 P3.2 = d) :=
sorry

noncomputable def dist_to_line (l : ℝ × ℝ) (x y : ℝ) : ℝ :=
  abs (l.1 * x + l.2 * y)

theorem find_vertex : vertex = (2, 0) :=
sorry

end problem_1_problem_2_find_vertex_l524_524306


namespace playground_area_is_correct_l524_524898

-- Definitions and conditions
def landscape_breadth (B : ℝ) := true
def landscape_length (L : ℝ) := L = 4 * landscape_breadth B ∧ L = 120
def total_landscape_area (A : ℝ) := A = landscape_length L * landscape_breadth B
def playground_area (A_playground : ℝ) := A_playground = (1 / 3) * total_landscape_area A

-- The theorem to prove the playground area
theorem playground_area_is_correct (B L A A_playground : ℝ) 
  (h1 : L = 4 * B)
  (h2 : L = 120)
  (h3 : A = L * B)
  (h4 : A_playground = (1 / 3) * A) :
  A_playground = 1200 :=
by
  -- This is where the proof would go,
  -- it should use the conditions to prove the theorem.
  sorry

end playground_area_is_correct_l524_524898


namespace kenneth_past_finish_line_l524_524177

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end kenneth_past_finish_line_l524_524177


namespace jim_percentage_profit_l524_524813

variable (selling_price cost_price : ℝ)

def profit (selling_price cost_price : ℝ) : ℝ :=
  selling_price - cost_price

def percentage_profit (selling_price cost_price : ℝ) : ℝ :=
  (profit selling_price cost_price / cost_price) * 100

theorem jim_percentage_profit :
  selling_price = 660 →
  cost_price = 550 →
  percentage_profit selling_price cost_price = 20 :=
by
  intros h1 h2
  rw [h1, h2]
  dsimp [profit, percentage_profit]
  norm_num
  sorry

end jim_percentage_profit_l524_524813


namespace line_circle_no_intersection_l524_524359

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524359


namespace line_circle_no_intersection_l524_524375

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524375


namespace jellybeans_original_count_l524_524219

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l524_524219


namespace problem_statement_l524_524965

theorem problem_statement (n : ℕ) (h : n ≥ 2) :
  ∃ p : ℕ, 0 + 2 + 2^2 + ⋯ + 2^(4 * n - 1) = 5 * p :=
by
  sorry

end problem_statement_l524_524965


namespace triangles_congruent_l524_524127

variables (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Type) [euclidean_geometry A B C A₁ B₁ C₁ A₂ B₂ C₂]

-- Definitions of Perpendicular
def perp1 : Prop := euclidean_geometry.perp_segment A₁ C₁ B
def perp2 : Prop := euclidean_geometry.perp_segment A₁ B₁ C
def perp3 : Prop := euclidean_geometry.perp_segment B₁ C₁ A
def perp4 : Prop := euclidean_geometry.perp_segment B₂ A₂ B
def perp5 : Prop := euclidean_geometry.perp_segment C₂ B₂ C
def perp6 : Prop := euclidean_geometry.perp_segment A₂ C₂ A

theorem triangles_congruent (h1 : perp1)
                           (h2 : perp2)
                           (h3 : perp3)
                           (h4 : perp4)
                           (h5 : perp5)
                           (h6 : perp6) :
                           euclidean_geometry.triangles_congruent A₁ B₁ C₁ A₂ B₂ C₂ :=
sorry

end triangles_congruent_l524_524127


namespace value_of_expression_l524_524422

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by
  sorry

end value_of_expression_l524_524422


namespace sphere_volume_after_increase_l524_524544

noncomputable def new_volume_of_sphere (surface_area : ℝ) (radius_increase : ℝ) : ℝ :=
  let r := Real.sqrt (surface_area / (4 * Real.pi))
  let new_radius := r + radius_increase
  (4/3) * Real.pi * new_radius^3

theorem sphere_volume_after_increase (surface_area : ℝ) (radius_increase : ℝ) :
  surface_area = 256 * Real.pi → radius_increase = 2 →
  new_volume_of_sphere surface_area radius_increase = (4000/3) * Real.pi :=
by
  intros hs hr
  rw [hs, hr]
  have hr_sqrt : Real.sqrt (256 * Real.pi / (4 * Real.pi)) = 8 := by sorry
  have new_r : 8 + 2 = 10 := by sorry
  have new_volume : (4/3) * Real.pi * 10^3 = (4000/3) * Real.pi := by sorry
  exact new_volume

end sphere_volume_after_increase_l524_524544


namespace triangle_inequality_l524_524294

variable (R r e f : ℝ)

theorem triangle_inequality (h1 : ∃ (A B C : ℝ × ℝ), true)
                            (h2 : true) :
  R^2 - e^2 ≥ 4 * (r^2 - f^2) :=
by sorry

end triangle_inequality_l524_524294


namespace line_circle_no_intersection_l524_524387

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524387


namespace domain_of_k_l524_524200

noncomputable def k (x : ℝ) := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^5 + 9)) + (1 / (x - 9))

theorem domain_of_k :
  ∀ x : ℝ, x ≠ -9 ∧ x ≠ -1.551 ∧ x ≠ 9 → ∃ y, y = k x := 
by
  sorry

end domain_of_k_l524_524200


namespace largest_number_systematic_sampling_l524_524150

theorem largest_number_systematic_sampling 
  (n_students : ℕ)
  (students : Fin n_students)
  (sampled_students : Finset ℕ)
  (sorted_sampled_students : List ℕ)
  (h_n_students : n_students = 60)
  (h_students_num : ∀ x ∈ sampled_students, x ≤ 60)
  (h_student_03 : 3 ∈ sampled_students)
  (h_student_09 : 9 ∈ sampled_students)
  (h_sorted : sorted_sampled_students = [3, 9] ++ (list.range 60).filter (λ x, (x % 6 = 3 ∧ x ≥ 15)))
  : 57 ∈ sampled_students :=
begin
  sorry
end

end largest_number_systematic_sampling_l524_524150


namespace root_in_interval_l524_524198

def f (x : ℝ) : ℝ := log x / log 2 + 2 * x - 4

theorem root_in_interval : ∃ c ∈ set.Ioo (1 : ℝ) 2, f c = 0 :=
by
  sorry

end root_in_interval_l524_524198


namespace num_ways_always_lead_l524_524857

def factorial (n : ℕ) : ℕ :=
  if h : n > 0 then List.prod (List.range' 1 n.succ) else 1

notation n "!" => factorial n

theorem num_ways_always_lead (a b : ℕ) (h : a > b) : ℕ :=
  (a - 1 + b)! / ((a - 1)! * b!) - (a - 1 + b)! / (a! * (b - 1)!)

#eval num_ways_always_lead 4 2 sorry

end num_ways_always_lead_l524_524857


namespace determine_nice_or_naughty_l524_524688

-- Oracle's function to sum nice divisors of a number
def f (u : ℕ) : ℕ := sorry  -- The actual function is not given, so we leave it as a placeholder.

-- Define the main theorem stating the problem's solution
theorem determine_nice_or_naughty (n : ℕ) (h1 : n > 0) (h2 : n < 1000000) :
  (∃ (s1 s2 s3 s4 : ℕ), 
     let s1 := f(n),
         s2 := f(2*n),
         s3 := f(3*n),
         s4 := f(n+1)
     in (      
       -- Additional conditions or constraints related to these answers
       -- are part of the problem and solution but will require precise formal definitions 
       sorry)) :=
sorry

-- Placeholder statements to ensure the theorem compiles without errors
definition n := 1
example : 0 < n := by sorry
example : n < 1000000 := by sorry

end determine_nice_or_naughty_l524_524688


namespace sum_seq_eq_l524_524909

-- defining the sequence and the range of k
def seq (k : ℕ) : ℕ := 2 * k + 1

-- stating the theorem to sum the sequence from k = 2 to n+1
theorem sum_seq_eq (n : ℕ) : ((finset.range (n + 2)).filter (λ k, 2 ≤ k)).sum seq = n * (n + 4) := by
  sorry

end sum_seq_eq_l524_524909


namespace average_of_numbers_not_1380_l524_524942

def numbers : List ℤ := [1200, 1300, 1400, 1520, 1530, 1200]

theorem average_of_numbers_not_1380 :
  let s := numbers.sum
  let n := numbers.length
  n > 0 → (s / n : ℚ) ≠ 1380 := by
  sorry

end average_of_numbers_not_1380_l524_524942


namespace squares_on_grid_l524_524174

-- Defining the problem conditions
def grid_size : ℕ := 5
def total_points : ℕ := grid_size * grid_size
def used_points : ℕ := 20

-- Stating the theorem to prove the total number of squares formed
theorem squares_on_grid : 
  (total_points = 25) ∧ (used_points = 20) →
  (∃ all_squares : ℕ, all_squares = 21) :=
by
  intros
  sorry

end squares_on_grid_l524_524174


namespace cell_phone_plan_cost_equality_l524_524970

theorem cell_phone_plan_cost_equality (x : ℝ) :
  let cost_A := 0.25 * x + 9
  let cost_B := 0.40 * x
  cost_A = cost_B → x = 60 := 
by
  intros cost_A_eq_cost_B
  have : 0.25 * x + 9 = 0.40 * x := cost_A_eq_cost_B
  sorry

end cell_phone_plan_cost_equality_l524_524970


namespace compare_y1_y2_l524_524503

theorem compare_y1_y2 : 
  (∃ y1 y2 : ℝ, (y1 = 8 * 3 - 1) ∧ (y2 = 8 * 4 - 1)) → y1 < y2 :=
by
  sorry

end compare_y1_y2_l524_524503


namespace hyperbola_equation_l524_524307

noncomputable def hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def parabola (x y : ℝ) : Prop :=
  y^2 = 24 * x

noncomputable def triangle_area (P F1 F2 : ℝ × ℝ) : ℝ :=
  1 / 2 * (F2.1 - F1.1) * (P.2 - F1.2).abs

theorem hyperbola_equation :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (λ x y, hyperbola x y a b) = (λ x y, x^2 / 9 - y^2 / 27 = 1) →
  let F1 := (-6, 0)
  let F2 := (6, 0)
  let P := (9, 6 * real.sqrt 6)
  triangle_area P F1 F2 = 36 * real.sqrt 6 :=
sorry

end hyperbola_equation_l524_524307


namespace ratio_last_day_to_first_6_days_l524_524930

def daily_visitors := 100
def first_6_days := 6
def earnings_per_visit := 0.01
def total_earnings := 18

def total_visitors_first_6_days := daily_visitors * first_6_days
def total_visitors_week := total_earnings / earnings_per_visit
def visitors_last_day := total_visitors_week - total_visitors_first_6_days

theorem ratio_last_day_to_first_6_days :
  visitors_last_day / total_visitors_first_6_days = 2 / 1 :=
by
  have h1 : total_visitors_first_6_days = 600 := 
    by simp [total_visitors_first_6_days, daily_visitors, first_6_days]
  have h2 : total_visitors_week = 1800 := 
    by norm_num [total_visitors_week, total_earnings, earnings_per_visit]
  have h3 : visitors_last_day = 1200 := 
    by norm_num [visitors_last_day, h1, h2]
  simp [h3, h1]
  norm_num


end ratio_last_day_to_first_6_days_l524_524930


namespace complex_real_implies_a_eq_2_l524_524266

noncomputable def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_real_implies_a_eq_2 (a : ℝ) (i : ℂ) (hi : i = complex.I) 
  (h : is_real ((a + 2 * i) / (1 + i))) : a = 2 :=
by sorry

end complex_real_implies_a_eq_2_l524_524266


namespace dot_product_range_l524_524760

theorem dot_product_range {A B F : ℝ × ℝ} 
  (h1 : ∃ x y, 2 * y ^ 2 - 2 * x ^ 2 = 1) 
  (h2 : F = (0, 1)) 
  (IntersectA : ∃ x y, A = (x, y) ∧ 2 * y ^ 2 - 2 * x ^ 2 = 1)
  (IntersectB : ∃ x y, B = (x, y) ∧ 2 * y ^ 2 - 2 * x ^ 2 = 1)
  (LineThroughF : (A.1 = 0 ∧ B.1 = 0) ∨ ∃ k, A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1):
  let FA := (A.1 - F.1, A.2 - F.2)
  let FB := (B.1 - F.1, B.2 - F.2)
  in (FA.1 * FB.1 + FA.2 * FB.2) ∈ set.Iio (-1 / 2) ∪ set.Ici (1 / 2) :=
sorry

end dot_product_range_l524_524760


namespace part_one_part_two_l524_524316

section math_problems

open Set Real

variables {a x: ℝ}

-- Definition of the sets A and B
def A (a : ℝ) : Set ℝ := { x | (x - 2) / (x - (3 * a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - (a^2 + 2)) / (x - a) < 0 }

-- (1) Given a = 1/2, find (C_U B) ∩ A
theorem part_one :
  let a : ℝ := 1/2 in
  (compl (B a)) ∩ (A a) = { x | (9 / 4) ≤ x ∧ x < 5 / 2 } :=
  sorry

-- (2) If q is a necessary condition for p, find the range of a
theorem part_two :
  (∀ x, x ∈ A a → x ∈ B a) ↔ 
    a ∈ {a | (-1 / 2 ≤ a ∧ a < 1 / 3) ∨ (1 / 3 < a ∧ a ≤ (3 - sqrt 5) / 2)} :=
  sorry

end math_problems

end part_one_part_two_l524_524316


namespace find_second_expression_l524_524047

theorem find_second_expression (a : ℕ) (x : ℕ) (h1 : (2 * a + 16 + x) / 2 = 69) (h2 : a = 26) : x = 70 := 
by
  sorry

end find_second_expression_l524_524047


namespace inscribed_circle_radius_l524_524793

theorem inscribed_circle_radius (h_base : 30 = 30) (h_lateral : 39 = 39) : 
  let base := 30
  let lateral := 39
  let height := sqrt (lateral^2 - (base / 2)^2)
  let S := 1 / 2 * base * height
  let p := (base + 2 * lateral) / 2
  r = S / p :=
  r = 10 :=
sorry

end inscribed_circle_radius_l524_524793


namespace students_between_l524_524540

theorem students_between (E J : ℕ) (hE : E = 20) (hJ : J = 14) : E - J - 1 = 5 :=
by
  rw [hE, hJ]
  exact rfl

end students_between_l524_524540


namespace ratio_length_to_width_l524_524900

theorem ratio_length_to_width
  (w l : ℕ)
  (pond_length : ℕ)
  (field_length : ℕ)
  (pond_area : ℕ)
  (field_area : ℕ)
  (pond_to_field_area_ratio : ℚ)
  (field_length_given : field_length = 28)
  (pond_length_given : pond_length = 7)
  (pond_area_def : pond_area = pond_length * pond_length)
  (pond_to_field_area_ratio_def : pond_to_field_area_ratio = 1 / 8)
  (field_area_def : field_area = pond_area * 8)
  (field_area_calc : field_area = field_length * w) :
  (field_length / w) = 2 :=
by
  sorry

end ratio_length_to_width_l524_524900


namespace line_circle_no_intersection_l524_524355

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524355


namespace mike_amy_remaining_after_expenses_l524_524533

def profit_ratios := (2.5, 5.2, 3.8, 4.5)
def johnson_share := 3120
def mike_expense := 200
def amy_expense := 150

theorem mike_amy_remaining_after_expenses : 
  let ratio_sum := profit_ratios.1 + profit_ratios.2 + profit_ratios.3 + profit_ratios.4
      value_per_part := johnson_share / profit_ratios.2
      mike_share := profit_ratios.1 * value_per_part
      amy_share := profit_ratios.3 * value_per_part
      mike_remaining := mike_share - mike_expense 
      amy_remaining := amy_share - amy_expense 
  in mike_remaining + amy_remaining = 3430 :=
by calc
  sorry

end mike_amy_remaining_after_expenses_l524_524533


namespace hilton_initial_marbles_l524_524323

variables (M : ℕ)

def initial_marbles_hilton (find_loss : ℕ) (end_marbles : ℕ) (lost_factor : ℕ) : Prop :=
  let found := 6 in
  let lost := 10 in
  let given := 2 * lost in
  let end := found - lost + given + M in
  end = end_marbles

theorem hilton_initial_marbles : initial_marbles_hilton M 6 10 2 42 :=
  sorry -- Proof would go here, omitted as per the instructions

end hilton_initial_marbles_l524_524323


namespace inequality_part1_l524_524268

theorem inequality_part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hne : a ≠ b) :
  sqrt (a * b) < (a - b) / (log a - log b) ∧ 
  (a - b) / (log a - log b) < (a + b) / 2 := 
sorry

end inequality_part1_l524_524268


namespace volume_prism_eq_9sqrt2_l524_524631

noncomputable def volume_triangular_prism (CK DK : ℝ) : ℝ :=
  let r := Real.sqrt 2 in
  let h := 2 * Real.sqrt 6 in
  let a := Real.sqrt 6 in
  let base_area := (a^2 * Real.sqrt 3) / 4 in
  base_area * h
  
theorem volume_prism_eq_9sqrt2 : 
  ∀ (CK DK : ℝ), CK = 2 * Real.sqrt 3 → DK = 2 * Real.sqrt 2 → 
  volume_triangular_prism CK DK = 9 * Real.sqrt 2 :=
by 
  intros CK DK hCK hDK
  rw [volume_triangular_prism, hCK, hDK]
  simp
  sorry

end volume_prism_eq_9sqrt2_l524_524631


namespace find_ABC_l524_524511

variables (A B C D : ℕ)

-- Conditions
def non_zero_distinct_digits_less_than_7 : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def ab_c_seven : Prop := 
  (A * 7 + B) + C = C * 7

def ab_ba_dc_seven : Prop :=
  (A * 7 + B) + (B * 7 + A) = D * 7 + C

-- Theorem to prove
theorem find_ABC 
  (h1 : non_zero_distinct_digits_less_than_7 A B C) 
  (h2 : ab_c_seven A B C) 
  (h3 : ab_ba_dc_seven A B C D) : 
  A * 100 + B * 10 + C = 516 :=
sorry

end find_ABC_l524_524511


namespace sweet_cookie_count_l524_524015

def initial_sweet_cookies (S : ℕ) : Prop :=
  ∃ (R : ℕ), S = R + 10 ∧ R = 9 ∧ 11 - 5 = 6

theorem sweet_cookie_count : initial_sweet_cookies 19 :=
by {
  use 9,
  split; {sorry},
}

end sweet_cookie_count_l524_524015


namespace sum_factors_24_l524_524961

theorem sum_factors_24 : (∑ d in (finset.filter (λ d, 24 % d = 0) (finset.range (25))), d) = 60 :=
by
  sorry

end sum_factors_24_l524_524961


namespace fourth_square_area_l524_524552

theorem fourth_square_area (AB BC CD AD AC : ℝ) (h1 : AB^2 = 25) (h2 : BC^2 = 49) (h3 : CD^2 = 64) (h4 : AC^2 = AB^2 + BC^2)
  (h5 : AD^2 = AC^2 + CD^2) : AD^2 = 138 :=
by
  sorry

end fourth_square_area_l524_524552


namespace sum_factors_24_l524_524962

theorem sum_factors_24 : (∑ d in (finset.filter (λ d, 24 % d = 0) (finset.range (25))), d) = 60 :=
by
  sorry

end sum_factors_24_l524_524962


namespace range_of_3x_plus_2y_is_correct_l524_524261

def range_of_3x_plus_2y := {z : ℝ | ∃ x y : ℝ, 1 ≤ x + y ∧ x + y ≤ 3 ∧ -1 ≤ x - y ∧ x - y ≤ 4 ∧ z = 3 * x + 2 * y}

theorem range_of_3x_plus_2y_is_correct : range_of_3x_plus_2y = set.Icc 2 9.5 :=
by
  sorry

end range_of_3x_plus_2y_is_correct_l524_524261


namespace final_limes_count_l524_524674

def limes_initial : ℕ := 9
def limes_by_Sara : ℕ := 4
def limes_used_for_juice : ℕ := 5
def limes_given_to_neighbor : ℕ := 3

theorem final_limes_count :
  limes_initial + limes_by_Sara - limes_used_for_juice - limes_given_to_neighbor = 5 :=
by
  sorry

end final_limes_count_l524_524674


namespace minimized_side_c_l524_524748

theorem minimized_side_c (t : ℝ) (C : ℝ) (hC : 0 < C ∧ C < π) :
  ∃ a b c : ℝ, a = b ∧ c = 2 * √(t * tan (C / 2)) ∧ 
  triangle_area_from_sides a b C = t ∧ 
  minimized_side_c_condition a b c C t :=
sorry

end minimized_side_c_l524_524748


namespace exists_neg_monomial_l524_524595

theorem exists_neg_monomial (a : ℤ) (x y : ℤ) (m n : ℕ) (hq : a < 0) (hd : m + n = 5) :
  ∃ a m n, a < 0 ∧ m + n = 5 ∧ a * x^m * y^n = -x^2 * y^3 :=
by
  sorry

end exists_neg_monomial_l524_524595


namespace distinct_remainders_mod_p_l524_524477
-- Import all necessary modules

-- Problem conditions
variable {p : ℕ} [hp_prime : Fact (Nat.Prime p)]
variable (k : ℕ) (h_k_range : 1 ≤ k ∧ k ≤ p)
variable {a : Fin k → ℕ} (h_a_not_div_p : ∀ i, ¬ p ∣ a i)

-- The main theorem
theorem distinct_remainders_mod_p : 
  ∃ S : Finset ℕ, S.card ≥ k ∧ 
  ∀ s ∈ S, ∃ e : Fin k → ℕ, (∀ i, e i = 0 ∨ e i = 1) ∧ s = ∑ i, e i * a i % p :=
sorry

end distinct_remainders_mod_p_l524_524477


namespace largest_integer_le_1_l524_524943

theorem largest_integer_le_1 (x : ℤ) (h : (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7) : x ≤ 1 :=
sorry

end largest_integer_le_1_l524_524943


namespace bela_always_wins_l524_524653

theorem bela_always_wins (n : ℤ) (h : n > 10) :
  (∀ x : ℝ, x ∈ set.Icc 0 n → 
    (∃ y : ℝ, y ∈ set.Icc 0 n ∧ ∀ z : ℝ, z ∈ set.Icc 0 n → |x - z| > 2 → ∃ w : ℝ, w ∈ set.Icc 0 n ∧ |w - y| > 2)) :=
sorry

end bela_always_wins_l524_524653


namespace part1_part2_l524_524475

-- Define the weak arithmetic progression for real numbers
def weak_arith_prog_3 (a1 a2 a3 : ℝ) : Prop :=
  ∃ (x0 x1 x2 x3 d : ℝ), 
  x0 ≤ a1 ∧ a1 < x1 ∧ x1 ≤ a2 ∧ a2 < x2 ∧ x2 ≤ a3 ∧ a3 < x3 ∧
  x1 - x0 = d ∧ x2 - x1 = d ∧ x3 - x2 = d

-- Part 1: Prove that a1 < a2 < a3 implies a1, a2, a3 form a weak arithmetic progression
theorem part1 (a1 a2 a3 : ℝ) (h : a1 < a2 ∧ a2 < a3) : weak_arith_prog_3 a1 a2 a3 := 
  sorry

-- Define the weak arithmetic progression of length 10 for natural numbers
def weak_arith_prog_10 (A : set ℕ) : Prop :=
  ∃ (x : ℕ → ℕ) (d : ℕ) (i : fin 11),
  (∀ i ≤ 9, x i ≤ 999) ∧ 
  ∀ (i : fin 10), x i ∈ A ∧ x (i + 1) - x i = d

-- Part 2: Prove that if |A| ≥ 730, then A contains a weak arithmetic progression of length 10
theorem part2 (A : set ℕ) (h : A ⊆ {n | n ≤ 999} ∧ A.card ≥ 730) : weak_arith_prog_10 A :=
  sorry

end part1_part2_l524_524475


namespace exponent_property_l524_524666

theorem exponent_property (a b : ℕ) : (a * b^2)^3 = a^3 * b^6 :=
by sorry

end exponent_property_l524_524666


namespace max_sum_in_grid_l524_524792

theorem max_sum_in_grid : 
  ∀ (grid : (Fin 8) → (Fin 8) → Bool),
  let sum := (fun (i j : Fin 8) => if grid i j then 0 else 
  (∑ k, if grid i k then 1 else 0) + (∑ k, if grid k j then 1 else 0)) in 
  ∑ i j, sum i j ≤ 256 := 
sorry

end max_sum_in_grid_l524_524792


namespace max_students_received_less_than_given_l524_524649

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l524_524649


namespace geometric_sequence_and_sum_l524_524314

theorem geometric_sequence_and_sum (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_a1 : a 1 = 3/2)
  (h_a_recur : ∀ n : ℕ, a (n + 1) = 3 * a n - 1)
  (h_b_def : ∀ n : ℕ, b n = a n - 1/2) :
  (∀ n : ℕ, b (n + 1) = 3 * b n ∧ b 1 = 1) ∧ 
  (∀ n : ℕ, S n = (3^n + n - 1) / 2) :=
sorry

end geometric_sequence_and_sum_l524_524314


namespace min_questions_to_determine_number_l524_524593

theorem min_questions_to_determine_number : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 50) → 
  ∃ (q : ℕ), q = 15 ∧ 
  ∀ (primes : ℕ → Prop), 
  (∀ p, primes p → Nat.Prime p ∧ p ≤ 50) → 
  (∀ p, primes p → (n % p = 0 ↔ p ∣ n)) → 
  (∃ m, (∀ k, k < m → primes k → k ∣ n)) :=
sorry

end min_questions_to_determine_number_l524_524593


namespace percentage_of_primes_less_than_20_divisible_by_5_l524_524569

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 := {n : ℕ | n < 20 ∧ is_prime n}

def primes_less_than_20_divisible_by_5 := {n ∈ primes_less_than_20 | 5 ∣ n}

theorem percentage_of_primes_less_than_20_divisible_by_5 : 
  (primes_less_than_20_divisible_by_5.to_finset.card : ℝ) / (primes_less_than_20.to_finset.card : ℝ) * 100 = 12.5 :=
begin
  -- Proving this statement directly would involve showing the calculations explicitly.
  -- However, we just set up the framework here.
  sorry
end

end percentage_of_primes_less_than_20_divisible_by_5_l524_524569


namespace dukes_wish_fulfilled_l524_524927

theorem dukes_wish_fulfilled
    (k : ℕ)
    (h1 : 1 < k)
    (h2 : k < 13)
    (warriors : ℕ)
    (h3 : warriors = 13)
    (has_gold_goblet : fin warriors → Prop)
    (castle : fin warriors → fin k)
    (gold_goblets : fin warriors → Prop)
    (h4 : (finset.univ.filter gold_goblets).card = k) :
    ∃ (i j : fin warriors), i ≠ j ∧ castle i = castle j ∧ gold_goblets i ∧ gold_goblets j := 
begin
  -- Proof is omitted as it is not required.
  sorry
end

end dukes_wish_fulfilled_l524_524927


namespace fuel_tank_capacity_l524_524171

theorem fuel_tank_capacity (C : ℝ) (h1 : 0.12 * 106 + 0.16 * (C - 106) = 30) : C = 214 :=
by
  sorry

end fuel_tank_capacity_l524_524171


namespace factor_x4_plus_64_l524_524892

theorem factor_x4_plus_64 (x : ℝ) : 
  (x^4 + 64) = (x^2 - 4 * x + 8) * (x^2 + 4 * x + 8) :=
sorry

end factor_x4_plus_64_l524_524892


namespace grasshoppers_no_overlap_l524_524806

-- Define the initial positions of the grasshoppers
def initial_positions (n : ℕ) : list (ℕ × ℕ) :=
  [(0, 0), (0, 3^n), (3^n, 3^n), (3^n, 0)]

-- Function to calculate the centroid of three points
def centroid (p1 p2 p3 : (ℕ × ℕ)) : ℝ × ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let (x3, y3) := p3 in
  ( (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3 )

-- Function to model the jump of a grasshopper
def jump (p : ℕ × ℕ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in
  let (cx, cy) := c in
  (2 * cx - x, 2 * cy - y)

-- Prove that at no point can a grasshopper land on another
theorem grasshoppers_no_overlap (n : ℕ) :
  ∀ (t : ℕ), 
    let positions := initial_positions n in
    ∀ (ps : list (ℕ × ℕ)), ps ≠ [] -> length ps = 4 ->
    (∀ (p ∈ ps), ∃ (c : ℝ × ℝ), jump p c ∉ ps) :=
by
  sorry

end grasshoppers_no_overlap_l524_524806


namespace num_distinct_increasing_digits_in_range_l524_524774

theorem num_distinct_increasing_digits_in_range :
  let S := {n : ℕ | 1950 ≤ n ∧ n < 2100 ∧ (∀ (i j : ℕ) (hi : i < 4) (hj : j < 4), 
                                            n.digits i ≠ n.digits j → n.digits i < n.digits j)} in
  set.card S = 21 :=
by
  sorry

end num_distinct_increasing_digits_in_range_l524_524774


namespace trapezoid_diagonal_comparison_l524_524022

variable {A B C D: Type}
variable (α β : Real) -- Representing angles
variable (AB CD BD AC : Real) -- Representing lengths of sides and diagonals
variable (h : Real) -- Height
variable (A' B' : Real) -- Projections

noncomputable def trapezoid (AB CD: Real) := True -- Trapezoid definition placeholder
noncomputable def angle_relation (α β : Real) := α < β -- Angle relationship

theorem trapezoid_diagonal_comparison
  (trapezoid_ABCD: trapezoid AB CD)
  (angle_relation_ABC_DCB : angle_relation α β)
  : BD > AC :=
sorry

end trapezoid_diagonal_comparison_l524_524022


namespace number_of_bad_arrangements_l524_524531

def is_bad_arrangement (l : List ℕ) : Prop :=
  ∃ (n : ℕ), (n ≥ 1) ∧ (n ≤ 16) ∧
  ¬ ∃ (k : ℕ) (s : List ℕ), s = l.rotate k.take 4 ∧ s.sum = n

def count_bad_arrangements : Nat :=
  List.filter is_bad_arrangement (List.permutations [1, 2, 3, 4, 6]).length

theorem number_of_bad_arrangements : count_bad_arrangements = 1 := by
  sorry

end number_of_bad_arrangements_l524_524531


namespace Nigel_initial_amount_l524_524849

theorem Nigel_initial_amount :
  ∃ W : ℕ, W + 55 = 2 * W + 10 ∧ W = 45 :=
by
  use 45
  split
  · simp
  · rfl
  sorry

end Nigel_initial_amount_l524_524849


namespace cos_alpha_plus_beta_l524_524604

variable {α β : ℝ}
variable (sin_alpha : Real.sin α = 3/5)
variable (cos_beta : Real.cos β = 4/5)
variable (α_interval : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (β_interval : β ∈ Set.Ioo 0 (Real.pi / 2))

theorem cos_alpha_plus_beta: Real.cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_l524_524604


namespace max_elevation_l524_524623

def elevation (t : ℝ) : ℝ := 144 * t - 18 * t^2

theorem max_elevation : ∃ t : ℝ, elevation t = 288 :=
by
  use 4
  sorry

end max_elevation_l524_524623


namespace vector_combination_correct_l524_524765

def vector_a := ⟨-3, 5, 2⟩
def vector_b := ⟨6, -1, -3⟩
def vector_c := ⟨1, 2, 3⟩

def vector_combination (a b c : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (ax, ay, az) := a
  let (bx, by, bz) := b
  let (cx, cy, cz) := c
  in (ax - 4 * bx + 2 * cx, ay - 4 * by + 2 * cy, az - 4 * bz + 2 * cz)

theorem vector_combination_correct :
  vector_combination vector_a vector_b vector_c = ⟨-25, 13, 20⟩ :=
by
  sorry

end vector_combination_correct_l524_524765


namespace rectangle_rhombus_diagonal_l524_524014

theorem rectangle_rhombus_diagonal (A B C D E F : ℝ) 
    (hAB: A = 16) (hBC: B = 12) 
    (rhombus_AEFC: AF = AE ∧ EF = FC ∧ AB = BC)
    (rhombus_properties: ∀ (AC EF : ℝ), 
        sqrt(AB ^ 2 + BC ^ 2) = 20 ∧ EF * EF = (AC / 2) * (AC / 2) + (AC / 2) * (AC / 2) ∧ EFC_angle_between = 90 ∧ bisect := true) :
  
  EF = 15 :=
by {
  sorry
}

end rectangle_rhombus_diagonal_l524_524014


namespace length_of_AB_l524_524859

variable {A B C D E F G : Type}
variable [metric_space A]
variables [metric_space B]
variables [has_distance C] [has_distance D] [has_distance E]
variables [has_distance F] [has_distance G]

def is_midpoint (X Y Z : point) : Prop := dist X Y = dist X Z ∧ 2 * dist X Y = dist Y Z

theorem length_of_AB 
  (C_midpoint_AB : is_midpoint A B C)
  (D_midpoint_AC : is_midpoint A C D)
  (E_midpoint_AD : is_midpoint A D E)
  (F_midpoint_AE : is_midpoint A E F)
  (G_midpoint_AF : is_midpoint A F G)
  (AG_eq_4 : dist A G = 4) : 
  dist A B = 128 := 
sorry

end length_of_AB_l524_524859


namespace longest_path_within_rect_5x8_l524_524011

-- Definitions arising from the conditions
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

def valid_path (path : list (ℕ × ℕ)) (rect : Rectangle) : Prop :=
  -- The path should be closed, meaning it starts and ends at the same vertex
  (path.head = path.last) ∧
  -- The path should follow the diagonals of 1 x 2 rectangles
  (∀ i, i < path.length - 1 →
    let (x1, y1) := path.nth i
        (x2, y2) := path.nth (i + 1)
    in abs (x2 - x1) = 1 ∧ abs (y2 - y1) = 2 ∨
       abs (x2 - x1) = 2 ∧ abs (y2 - y1) = 1) ∧
  -- No diagonal is retraced
  (∀ (p1 p2 : ℕ × ℕ), (p1, p2) ∈ path.pairs → (p1 ≠ p2))

-- The rectangle in question
def rect_5x8 : Rectangle := { width := 5, height := 8 }

-- The main theorem
theorem longest_path_within_rect_5x8 :
  ∃ path : list (ℕ × ℕ), valid_path path rect_5x8 ∧ path.length = 25 :=
begin
  sorry
end

end longest_path_within_rect_5x8_l524_524011


namespace correct_choice_C_l524_524800

def geometric_sequence (n : ℕ) : ℕ := 
  2^(n - 1)

def sum_geometric_sequence (n : ℕ) : ℕ := 
  2^n - 1

theorem correct_choice_C (n : ℕ) (h : 0 < n) : sum_geometric_sequence n < geometric_sequence (n + 1) := by
  sorry

end correct_choice_C_l524_524800


namespace determine_nice_or_naughty_l524_524687

-- Definitions
def nice (n : ℕ) : Prop := sorry  -- Assume we have a predicate to determine if n is nice or not
def sum_nice_divisors (n : ℕ) : ℕ := sorry -- Assume a function that gives the sum of all nice divisors of n

-- Main theorem to prove
theorem determine_nice_or_naughty (n : ℕ) (h : n < 1000000) : 
  ∃ oracle_calls : fin 4 → ℕ, 
    ∀ k : fin 4, oracle_calls k = sum_nice_divisors ?m_1 ∧ (nice n ∨ ¬ nice n) :=
begin
  -- Proof steps would go here
  sorry
end

end determine_nice_or_naughty_l524_524687


namespace coordinates_of_point_l524_524447

theorem coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, 3)) : (x, y) = (-2, 3) :=
by
  exact h

end coordinates_of_point_l524_524447


namespace mutually_exclusive_A_C_l524_524711

-- Definitions based on the given conditions
def all_not_defective (A : Prop) : Prop := A
def all_defective (B : Prop) : Prop := B
def at_least_one_defective (C : Prop) : Prop := C

-- Theorem to prove A and C are mutually exclusive
theorem mutually_exclusive_A_C (A B C : Prop) 
  (H1 : all_not_defective A) 
  (H2 : all_defective B) 
  (H3 : at_least_one_defective C) : 
  (A ∧ C) → False :=
sorry

end mutually_exclusive_A_C_l524_524711


namespace line_circle_no_intersection_l524_524389

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524389


namespace incorrect_statement_maximum_value_l524_524108

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_statement_maximum_value :
  ∃ (a b c : ℝ), 
    (quadratic_function a b c 1 = -40) ∧
    (quadratic_function a b c (-1) = -8) ∧
    (quadratic_function a b c (-3) = 8) ∧
    (∀ (x_max : ℝ), (x_max = -b / (2 * a)) →
      (quadratic_function a b c x_max = 10) ∧
      (quadratic_function a b c x_max ≠ 8)) :=
by
  sorry

end incorrect_statement_maximum_value_l524_524108


namespace distance_between_vertices_hyperbola_l524_524240

-- Definitions as per conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Statement of the problem in Lean
theorem distance_between_vertices_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → ∃ d : ℝ, d = 8 :=
by
  intros x y h
  use 8
  sorry

end distance_between_vertices_hyperbola_l524_524240


namespace citizen_income_l524_524114

theorem citizen_income (I : ℝ) 
  (h1 : I > 0)
  (h2 : 0.12 * 40000 + 0.20 * (I - 40000) = 8000) : 
  I = 56000 := 
sorry

end citizen_income_l524_524114


namespace sin_graph_intersection_l524_524284

theorem sin_graph_intersection (a b : ℕ) (h : a ≠ b) : 
  ∃ c : ℕ, c ≠ a ∧ c ≠ b ∧ (∀ x₀, sin (a * x₀) = sin (b * x₀) → sin (c * x₀) = sin (a * x₀))
  ∧ c = 2 * (a^2 - b^2) + a := 
by {
  sorry
}

end sin_graph_intersection_l524_524284


namespace Omega2_tangent_to_CD_l524_524936

open Classical

variables {Ω Ω₁ Ω₂ : Type} [circle Ω] [circle Ω₁] [circle Ω₂]
variables {M N A B C D : Point}
variables {center_of_Ω : Ω → Point}
variables {center_of_Ω₁ : Ω₁ → Point}
variables {center_of_Ω₂ : Ω₂ → Point}
variables {common_chord : Ω₁ → Ω₂ → Line}
variables {tangency_point : circle → circle → Point}

-- Assume the necessary conditions as typeclass instances
instance : touches_internally Ω Ω₁ (tangency_point Ω Ω₁) := by sorry
instance : touches_internally Ω Ω₂ (tangency_point Ω Ω₂) := by sorry
instance : center_of_Ω₂_on_Ω₁ : (center_of_Ω₁ Ω₁ = center_of_Ω₂ Ω₂) := by sorry
instance : intersects_Ω_at_points (common_chord Ω₁ Ω₂) Ω A B := by sorry
instance : intersects_Ω₁_at_points (Line_through M A) Ω₁ C := by sorry
instance : intersects_Ω₁_at_points (Line_through M B) Ω₁ D := by sorry

-- Define the required theorem to prove the tangency
theorem Omega2_tangent_to_CD :
  tangent Ω₂ (Line_through C D) := sorry

end Omega2_tangent_to_CD_l524_524936


namespace incorrect_proposition_b_l524_524297

axiom plane (α β : Type) : Prop
axiom line (m n : Type) : Prop
axiom parallel (a b : Type) : Prop
axiom perpendicular (a b : Type) : Prop
axiom intersection (α β : Type) (n : Type) : Prop
axiom contained (a b : Type) : Prop

theorem incorrect_proposition_b (α β m n : Type)
  (hαβ_plane : plane α β)
  (hmn_line : line m n)
  (h_parallel_m_α : parallel m α)
  (h_intersection : intersection α β n) :
  ¬ parallel m n :=
sorry

end incorrect_proposition_b_l524_524297


namespace inequality_system_range_k_l524_524308

theorem inequality_system_range_k :
  (∃ x : ℤ, (x^2 - 2 * x - 8 > 0) ∧ (2 * x^2 + (2 * k + 7) * x + 7 * k < 0)) ↔
    (k ∈ (-5 : ℝ) ∪ Ico -5 3 ∪ Icc 4 5) :=
by
  sorry

end inequality_system_range_k_l524_524308


namespace anita_smallest_number_of_candies_l524_524645

theorem anita_smallest_number_of_candies :
  ∃ x : ℕ, x ≡ 5 [MOD 6] ∧ x ≡ 3 [MOD 8] ∧ x ≡ 7 [MOD 9] ∧ ∀ y : ℕ,
  (y ≡ 5 [MOD 6] ∧ y ≡ 3 [MOD 8] ∧ y ≡ 7 [MOD 9]) → x ≤ y :=
  ⟨203, by sorry⟩

end anita_smallest_number_of_candies_l524_524645


namespace compound_interest_semiannual_l524_524983

theorem compound_interest_semiannual
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (initial_amount : P = 900)
  (interest_rate : r = 0.10)
  (compounding_periods : n = 2)
  (time_period : t = 1) :
  P * (1 + r / n) ^ (n * t) = 992.25 :=
by
  sorry

end compound_interest_semiannual_l524_524983


namespace percentage_not_caught_l524_524786

theorem percentage_not_caught (x : ℝ) (h1 : 22 + x = 25.88235294117647) : x = 3.88235294117647 :=
sorry

end percentage_not_caught_l524_524786


namespace percentage_primes_divisible_by_five_l524_524577

def primes := [2, 3, 5, 7, 11, 13, 17, 19]
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem percentage_primes_divisible_by_five : 
  (∃ count, count = (list.filter divisible_by_five primes).length) 
  →  (list.length primes) = 8
  →  count = 1
  → (count : ℝ) / 8 * 100 = 12.5 :=
by
  sorry

end percentage_primes_divisible_by_five_l524_524577


namespace minimize_side_c_l524_524745

variables (t : ℝ) (C : ℝ)

theorem minimize_side_c (a b c : ℝ) (h_area : (1/2) * a * b * sin C = t) (h_eq : a = b) :
  c = 2 * sqrt (t * tan (C / 2)) := 
sorry

end minimize_side_c_l524_524745


namespace surface_integral_ellipsoid_l524_524658

variable (a b c H : ℝ)

def ellipsoid_eq (x y z : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1

def surface_integral (S : Set (ℝ × ℝ × ℝ)) : ℝ :=
  ∫∫ S (λ (x y z : ℝ), x * dy * dz + y * dz * dx + z * dx * dy)

theorem surface_integral_ellipsoid :
  ∀ (S : Set (ℝ × ℝ × ℝ)), (∀ (x y z : ℝ), (x, y, z) ∈ S → ellipsoid_eq a b c x y z) →
  ∀ (z : ℝ), z ∈ [-H, H] →
  surface_integral a b c S = -4 * Real.pi * a * b * c :=
sorry

end surface_integral_ellipsoid_l524_524658


namespace max_students_gave_away_balls_more_l524_524651

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l524_524651


namespace probability_3x_gt_4y_l524_524497

theorem probability_3x_gt_4y:
  let area_triangle := (1 / 2) * 2021 * 1515.75
  let area_rectangle := 2021 * 2022
  let probability := area_triangle / area_rectangle
  probability = 1515750 / 4044 :=
begin
  -- Definitions based on the conditions
  let vertices := [(0, 0), (2021, 0), (2021, 2022), (0, 2022)],
  let inequality := 3 * x > 4 * y,
  let line := λ x, 3 / 4 * x,
  let intersection := 2021 * 1515.75 / 2,
  -- The statement which does not assume the solution steps
  sorry
end

end probability_3x_gt_4y_l524_524497


namespace derivative_y_l524_524974

variable (x : ℝ)

def y : ℝ := (Real.sin (2 * x - 1))^2

theorem derivative_y : (deriv y x) = 2 * Real.sin(2 * (2 * x - 1)) :=
by 
  sorry

end derivative_y_l524_524974


namespace mass_equality_l524_524915

variable m_circ : ℝ
variable m_tria : ℝ
variable m_sq : ℝ

-- Conditions
axiom h1 : 3 * m_circ = 2 * m_tria
axiom h2 : m_sq + m_circ + m_tria = 2 * m_sq

-- The theorem (problem we need to prove)
theorem mass_equality : m_circ + 3 * m_tria = 3 * m_sq :=
by
  sorry

end mass_equality_l524_524915


namespace angle_distance_relation_l524_524825

variables {S1 S2 : Type*} [plane S1] [plane S2]
variables {m : line S1 S2} {e : line} 
variables {D1 D2 : point} (D1_ne_D2 : D1 ≠ D2)
variables {α1 α2 : ℝ} 
variables (angle_e_S1 : e.angle_with_plane S1 = α1) (angle_e_S2 : e.angle_with_plane S2 = α2)
variables (intersection_e_S1 : e.intersects_with_plane_at S1 D1)
variables (intersection_e_S2 : e.intersects_with_plane_at S2 D2)
variables (distance_D1_m : ℝ) (distance_D2_m : ℝ)

theorem angle_distance_relation : 
  (α1 > α2) ↔ (distance_D1_m < distance_D2_m) := 
sorry

end angle_distance_relation_l524_524825


namespace sandcastle_container_volume_l524_524223

noncomputable def volume_cylinder (r h : ℝ) : ℝ := 
  π * r^2 * h

theorem sandcastle_container_volume : volume_cylinder 4 15 = 240 * π := 
by 
  sorry

end sandcastle_container_volume_l524_524223


namespace sqrt_domain_l524_524782

theorem sqrt_domain (x : ℝ) : x - 8 ≥ 0 ↔ x ≥ 8 :=
begin
  sorry
end

end sqrt_domain_l524_524782


namespace fill_tank_with_XZ_l524_524928

theorem fill_tank_with_XZ
  (x y z : ℝ)
  (h1 : x + y + z = 1 / 1.3)
  (h2 : x + y = 1 / 2)
  (h3 : y + z = 1 / 3) :
  (1 / (x + z) ≈ 1.42) :=
by
  sorry

end fill_tank_with_XZ_l524_524928


namespace binomial_sum_trigonometric_identity_l524_524024

theorem binomial_sum_trigonometric_identity (n : ℕ) (m : ℕ) (hn : 4 * m - 3 ≤ n) :
  (finset.range (n + 1)).sum (λ k, if k % 4 = 1 then nat.choose n k else 0) =
    1 / 2 * (2 ^ (n - 1) + 2 ^ (n / 2) * real.sin (n * real.pi / 4)) :=
by sorry

end binomial_sum_trigonometric_identity_l524_524024


namespace line_circle_no_intersection_l524_524372

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524372


namespace probability_of_all_female_l524_524494

noncomputable def probability_all_females_final (females males total chosen : ℕ) : ℚ :=
  (females.choose chosen) / (total.choose chosen)

theorem probability_of_all_female:
  probability_all_females_final 5 3 8 3 = 5 / 28 :=
by
  sorry

end probability_of_all_female_l524_524494


namespace line_circle_no_intersection_l524_524327

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524327


namespace line_circle_no_intersection_l524_524401

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524401


namespace geom_series_example_l524_524668

noncomputable def geom_series_sum (a r n : ℤ) : ℤ :=
  (a * (r ^ n - 1)) / (r - 1)

theorem geom_series_example :
  let a := 1
  let r := -3
  let n := 7
  let S := 547
  geom_series_sum a r n = S :=
by
  trivial

end geom_series_example_l524_524668


namespace student_weight_l524_524982

theorem student_weight (S W : ℕ) (h1 : S - 5 = 2 * W) (h2 : S + W = 104) : S = 71 :=
by {
  sorry
}

end student_weight_l524_524982


namespace max_x_for_lcm_120_l524_524067

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem max_x_for_lcm_120 (x : ℕ) (h : lcm (lcm x 8) 12 = 120) : x ≤ 120 :=
by
-- sorry proof steps not required
sorry

end max_x_for_lcm_120_l524_524067


namespace smallest_positive_integer_n_l524_524700

def rotation_matrix := ![![ (1 : ℝ) / 2, (Real.sqrt 3) / 2],![ - (Real.sqrt 3) / 2, (1 : ℝ) / 2]]
def identity_matrix := ![![1, 0],![0, 1]]

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), 0 < n ∧ (matrix.pow rotation_matrix n) = identity_matrix ∧
  ∀ m : ℕ, 0 < m ∧ (matrix.pow rotation_matrix m) = identity_matrix → n ≤ m :=
sorry

end smallest_positive_integer_n_l524_524700


namespace standard_equation_of_hyperbola_l524_524293

-- Define the conditions
def c : ℝ := Real.sqrt 6
def P : ℝ × ℝ := (-5, 2)

-- Assumption that the hyperbola has its foci on the x-axis.
def hyperbola_standard_eq (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (c = Real.sqrt 6) ∧ (a^2 + b^2 = 6) ∧ ((P.1^2) / a^2 - (P.2^2) / b^2 = 1)

-- Prove that the standard equation has the given form
theorem standard_equation_of_hyperbola :
  ∃ (a b : ℝ), hyperbola_standard_eq a b ∧
  (a^2 = 5) ∧ (b^2 = 1) ∧ (∀ x y, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 5 - y^2 / 1 = 1)) :=
by
  sorry

end standard_equation_of_hyperbola_l524_524293


namespace int_combination_factorial_l524_524678

theorem int_combination_factorial {n m : ℤ} (hn : n ≥ 0) (hm : m ≥ 0) :
  (n * n.factorial + m * m.factorial = 4032) → 
  (n = 3) ∧ (m = 6) :=
by
  sorry

end int_combination_factorial_l524_524678


namespace room_length_l524_524899

theorem room_length (length width rate cost : ℝ)
    (h_width : width = 3.75)
    (h_rate : rate = 1000)
    (h_cost : cost = 20625)
    (h_eq : cost = length * width * rate) :
    length = 5.5 :=
by
  -- the proof will go here
  sorry

end room_length_l524_524899


namespace sequence_b_100_eq_10002_l524_524196

def sequence (b : ℕ → ℕ) := 
b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 2 * n + 1

theorem sequence_b_100_eq_10002 (b : ℕ → ℕ)
  (h : sequence b) : b 100 = 10002 :=
sorry

end sequence_b_100_eq_10002_l524_524196


namespace sum_of_first_11_terms_l524_524448

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h : ∀ x : ℝ, x^2 - 2 * x - 6 = 0 → (x = a 5 ∨ x = a 7)) : 
  (finset.range 11).sum a = 11 :=
by sorry

end sum_of_first_11_terms_l524_524448


namespace solve_trig_system_l524_524035

theorem solve_trig_system :
  (∃ x y : ℝ, ∃ k l : ℤ, cos x = 2 * (cos y) ^ 3 ∧ sin x = 2 * (sin y) ^ 3 ∧
    x = 2 * l * π + (k * π) / 2 + π / 4 ∧ y = (k * π) / 2 + π / 4) :=
by
  sorry

end solve_trig_system_l524_524035


namespace parabola_focus_l524_524427

noncomputable def coordinates_of_focus {a : ℝ} (x y : ℝ) : Prop :=
  let focus_x := (a / 4)
  let focus_y := 0
  x = focus_x ∧ y = focus_y

theorem parabola_focus 
  (a : ℝ)
  (h_eqn : ∀ x y : ℝ, y ^ 2 = a * x)
  (h_directrix : ∀ x : ℝ, x = -1 → ∀ y : ℝ, True) :
  coordinates_of_focus 1 0 :=
by
  sorry

end parabola_focus_l524_524427


namespace line_circle_no_intersection_l524_524399

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524399


namespace simplify_trig_expression_l524_524878

theorem simplify_trig_expression (α : ℝ) :
  sin (α - 4 * π) * sin (π - α) - 2 * cos^2 (3 * π / 2 + α) - sin (α + π) * cos (π / 2 + α) 
  = -2 * sin^2 α :=
sorry

end simplify_trig_expression_l524_524878


namespace line_circle_no_intersection_l524_524380

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524380


namespace coefficient_x2_term_l524_524695

theorem coefficient_x2_term :
  let f := (λ (k : ℕ), (1 + x)^k)
  in (let c := sum (λ k, if k >= 2 then binom k 2 else 0) (finset.range 10) in c = 120) := 
by
  -- We let 'f' denote the term (1+x)^k.
  -- The sum is from k = 2 to k = 9 of binomial coefficients binom(k, 2).
  -- Using the Hockey-Stick Identity, this sum can be simplified.
  -- The resulting sum can be calculated to be binom(10, 3) = 120.
  sorry

end coefficient_x2_term_l524_524695


namespace number_of_cars_l524_524881

theorem number_of_cars (people_per_car : ℝ) (total_people : ℝ) (h1 : people_per_car = 63.0) (h2 : total_people = 189) : total_people / people_per_car = 3 := by
  sorry

end number_of_cars_l524_524881


namespace sum_of_positive_integer_factors_of_24_l524_524952

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l524_524952


namespace ineq_pos_xy_l524_524866

theorem ineq_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x := 
sorry

end ineq_pos_xy_l524_524866


namespace minimal_k_for_integer_b_l524_524081

-- Definitions and conditions from the problem
def b_seq : ℕ → ℝ
| 1 := 1
| (n + 2) := b_seq (n + 1) + (Real.log ((4 * (n + 1) + 5) / (4 * (n + 1) + 1)) / Real.log 3)

-- The Lean theorem stating that the least integer k > 1 such that b_k is an integer is 20
theorem minimal_k_for_integer_b :
  ∃ k : ℕ, k > 1 ∧ (b_seq k).denom = 1 ∧ k = 20 :=
by
  use 20
  split
  · exact nat.succ_lt_succ (nat.one_lt_bit0 nat.zero_lt_one)
  · sorry

end minimal_k_for_integer_b_l524_524081


namespace roses_problem_l524_524090

-- Defining the conditions 
variables (R_initial R_thrown R_given R_final R_cut : ℕ)
variables (H_initial : R_initial = 25)
variables (H_thrown : R_thrown = 40)
variables (H_given : R_given = 10)
variables (H_final : R_final = 45)

-- The statement to be proven
theorem roses_problem :
  R_cut = R_final - R_initial →
  ∣R_cut - (R_thrown + R_given)∣ = 30 :=
by
  intro h_cut
  sorry

end roses_problem_l524_524090


namespace remainder_of_num_five_element_subsets_with_two_consecutive_l524_524470

-- Define the set and the problem
noncomputable def num_five_element_subsets_with_two_consecutive (n : ℕ) : ℕ := 
  Nat.choose 14 5 - Nat.choose 10 5

-- Main Lean statement: prove the final condition
theorem remainder_of_num_five_element_subsets_with_two_consecutive :
  (num_five_element_subsets_with_two_consecutive 14) % 1000 = 750 :=
by
  -- Proof goes here
  sorry

end remainder_of_num_five_element_subsets_with_two_consecutive_l524_524470


namespace line_circle_no_intersect_l524_524392

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524392


namespace smallest_d_l524_524828

def g (x : ℝ) : ℝ := (x - 3)^2 - 4

def is_one_to_one (f : ℝ → ℝ) (dom : set ℝ) : Prop := 
  ∀ x1 x2 ∈ dom, f x1 = f x2 → x1 = x2

theorem smallest_d (d : ℝ) (hd : d = 3) : 
  is_one_to_one g { x : ℝ | x ≥ d } :=
by
  sorry

end smallest_d_l524_524828


namespace triangle_area_l524_524539

theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) (h4 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 270 :=
by
  sorry

end triangle_area_l524_524539


namespace line_circle_no_intersect_l524_524397

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524397


namespace sum_of_digits_of_n_l524_524089

theorem sum_of_digits_of_n (n : ℕ) (h : (n + 1)! + (n + 3)! = 440 * n!) : 
  (→ (1 + 8 = 9)) := 
sorry

end sum_of_digits_of_n_l524_524089


namespace distance_between_vertices_of_hyperbola_l524_524243

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) → 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l524_524243


namespace length_of_QR_l524_524882

theorem length_of_QR
  (Q : ℝ) (QP QR : ℝ)
  (cos_Q_eq_half : cos Q = 0.5)
  (QP_eq_10 : QP = 10) :
  QR = 20 :=
by
  sorry

end length_of_QR_l524_524882


namespace range_of_a_l524_524763

open Set

variable {α : Type*} [LinearOrder α]

def A (x a : α) : Set α := {x | (x - a) * (x - 1) ≤ 0}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : A a ∪ B = B ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end range_of_a_l524_524763


namespace rectangle_square_overlap_l524_524506

theorem rectangle_square_overlap (ABCD EFGH : Type) (s x y : ℝ)
  (h1 : 0.3 * s^2 = 0.6 * x * y)
  (h2 : AB = 2 * s)
  (h3 : AD = y)
  (h4 : x * y = 0.5 * s^2) :
  x / y = 8 :=
sorry

end rectangle_square_overlap_l524_524506


namespace find_g_six_l524_524522

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_five : g 5 = 6

theorem find_g_six : g 6 = 36/5 := 
by 
  -- proof to be filled in
  sorry

end find_g_six_l524_524522


namespace average_of_roots_l524_524199

theorem average_of_roots (c : ℝ) (hc1 : c ≠ 0) 
(hc2 : ∃ x1 x2 : ℝ, (3 * x1^2 - 6 * x1 + c = 0) ∧ (3 * x2^2 - 6 * x2 + c = 0) ∧ (x1 ≠ x2)) :
  let x1 := (-6 + real.sqrt(36 - 12 * c)) / 6
  let x2 := (-6 - real.sqrt(36 - 12 * c)) / 6
  (x1 + x2) / 2 = 1 := 
by
  sorry

end average_of_roots_l524_524199


namespace line_circle_no_intersection_l524_524328

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524328


namespace right_angled_triangle_not_axisymmetric_l524_524639

-- Define a type for geometric figures
inductive Figure
| Angle : Figure
| EquilateralTriangle : Figure
| LineSegment : Figure
| RightAngledTriangle : Figure

open Figure

-- Define a function to determine if a figure is axisymmetric
def is_axisymmetric: Figure -> Prop
| Angle => true
| EquilateralTriangle => true
| LineSegment => true
| RightAngledTriangle => false

-- Statement of the problem
theorem right_angled_triangle_not_axisymmetric : 
  is_axisymmetric RightAngledTriangle = false :=
by
  sorry

end right_angled_triangle_not_axisymmetric_l524_524639


namespace cost_of_article_l524_524642

variable (C : ℝ)

theorem cost_of_article (h : (350 - C) = (340 - C) + 0.05 * (340 - C)) : C = 140 := by
  have eq1 : 350 - C = 340 - C + 0.05 * (340 - C) := h
  calc
    350 - C = 340 - C + 0.05 * (340 - C) : eq1
          ... = 340 - C + 0.05 * (340 - C)
          ... = 340 - C + 0.05 * (340 - C)
          ... ≠ 350 - C -- This part uses the contradiction to derive the final result
  sorry

end cost_of_article_l524_524642


namespace max_students_received_less_than_given_l524_524650

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l524_524650


namespace probability_of_odd_distinct_digits_l524_524172

def is_between_1000_and_9999 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def digits_are_distinct (n : ℕ) : Prop :=
  let ds := (n.digits 10);
  ds.nodup

noncomputable def favorable_outcomes : ℕ :=
  -- Bypass actual calculation, assuming we obtain the value from the solution
  2240

noncomputable def total_outcomes : ℕ :=
  9000

def probability_correctness : Prop :=
  (favorable_outcomes.toRat / total_outcomes) = (56 / 225)

theorem probability_of_odd_distinct_digits :
  probability_correctness :=
sorry

end probability_of_odd_distinct_digits_l524_524172


namespace line_circle_no_intersection_l524_524382

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524382


namespace solve_for_p_l524_524076

variable (p q : ℝ)
noncomputable def binomial_third_term : ℝ := 55 * p^9 * q^2
noncomputable def binomial_fourth_term : ℝ := 165 * p^8 * q^3

theorem solve_for_p (h1 : p + q = 1) (h2 : binomial_third_term p q = binomial_fourth_term p q) : p = 3 / 4 :=
by sorry

end solve_for_p_l524_524076


namespace common_ratio_geometric_sequence_l524_524056

theorem common_ratio_geometric_sequence (a : ℝ) :
  let log2_3 := Real.log 3 / Real.log 2 in
  let log4_3 := Real.log 3 / (2 * Real.log 2) in
  let log8_3 := Real.log 3 / (3 * Real.log 2) in
  (q = (a + log4_3) / (a + log2_3)) ∧ (q = (a + log8_3) / (a + log4_3)) →
  q = 1 / 3 :=
by
  intros
  sorry

end common_ratio_geometric_sequence_l524_524056


namespace hyperbola_vertex_distance_l524_524235

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 16 - y^2 / 9 = 1) → (vertex_distance : ℝ := 8) := sorry

end hyperbola_vertex_distance_l524_524235


namespace geometric_quadratic_root_l524_524535

theorem geometric_quadratic_root (a b c : ℝ) (h1 : a > 0) (h2 : b = a * (1 / 4)) (h3 : c = a * (1 / 16)) (h4 : a * a * (1 / 4)^2 = 4 * a * a * (1 / 16)) : 
    -b / (2 * a) = -1 / 8 :=
by 
    sorry

end geometric_quadratic_root_l524_524535


namespace volume_of_tetrahedron_OABC_l524_524934

-- Definitions of side lengths and their squared values
def side_length_A_B := 7
def side_length_B_C := 8
def side_length_C_A := 9

-- Squared values of coordinates
def a_sq := 33
def b_sq := 16
def c_sq := 48

-- Main statement to prove the volume
theorem volume_of_tetrahedron_OABC :
  (1/6) * (Real.sqrt a_sq) * (Real.sqrt b_sq) * (Real.sqrt c_sq) = 2 * Real.sqrt 176 :=
by
  -- Proof steps would go here
  sorry

end volume_of_tetrahedron_OABC_l524_524934


namespace function_b_increasing_l524_524592

theorem function_b_increasing : ∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (x2 < 4) → (2*x1^2 + 3 < 2*x2^2 + 3) :=
by
  intros x1 x2 hx1 h12 hx2
  calc
    2 * x1^2 + 3 < 2 * x2^2 + 3 : sorry

end function_b_increasing_l524_524592


namespace no_intersection_l524_524368

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524368


namespace line_circle_no_intersection_l524_524333

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524333


namespace jeffs_mean_l524_524460

-- Define Jeff's scores as a list or array
def jeffsScores : List ℚ := [86, 94, 87, 96, 92, 89]

-- Prove that the arithmetic mean of Jeff's scores is 544 / 6
theorem jeffs_mean : (jeffsScores.sum / jeffsScores.length) = (544 / 6) := by
  sorry

end jeffs_mean_l524_524460


namespace diagonal_length_possibilities_l524_524709

theorem diagonal_length_possibilities :
  ∃ n : ℕ, n = 15 ∧ (∀ x : ℕ, 6 ≤ x ∧ x ≤ 20 → true) :=
by {
  -- Define the lengths of the segments according to the problem statement
  let AB : ℕ := 9,
  let BC : ℕ := 12,
  let CD : ℕ := 20,
  let DA : ℕ := 15,
  -- Define the length of diagonal AC satisfies the triangle inequalities
  let AC_upper_triangle : ∀ x : ℤ, 5 < x ∧ x < 21 → true := λ x h, h,
  -- All possible whole number values for AC and the correct count
  have range_AC_possible : 6 <= 6 ∧ 6 <= 7 → true := by sorry,
  have count_AC_possible : ((20 - 6 + 1): ℕ = 15) := by norm_num,
  exact ⟨15, count_AC_possible, range_AC_possible⟩
}

end diagonal_length_possibilities_l524_524709


namespace new_shoes_cost_greater_by_21_74_percent_l524_524998

def repair_cost : ℝ := 11.50
def repair_lifespan : ℝ := 1
def new_cost : ℝ := 28.00
def new_lifespan : ℝ := 2

def avg_cost_repaired : ℝ := repair_cost / repair_lifespan
def avg_cost_new : ℝ := new_cost / new_lifespan

def percentage_increase : ℝ := ((avg_cost_new - avg_cost_repaired) / avg_cost_repaired) * 100

theorem new_shoes_cost_greater_by_21_74_percent :
  percentage_increase = 21.74 :=
sorry

end new_shoes_cost_greater_by_21_74_percent_l524_524998


namespace distance_P_1_2_to_line_2x_minus_1_coordinates_of_M_on_line_x_plus_2_value_of_k_for_min_distance_l524_524739

-- 1. Prove distance from P(1,2) to the line y=2x-1 is sqrt(5)/5
theorem distance_P_1_2_to_line_2x_minus_1 : 
  ∃ d : ℚ, d = (Real.sqrt 5) / 5 ∧ 
  (let k := 2 in let b := -1 in 
  let x0 := 1 in let y0 := 2 in
  d = (Real.abs (k * x0 - y0 + b)) / (Real.sqrt (1 + k^2))) := 
sorry

-- 2. Given point M on y=x+2, prove distances and coordinates of M are (13,15) or (-7,-5)
theorem coordinates_of_M_on_line_x_plus_2 : 
  ∃ M : ℚ × ℚ, 
  let xm := M.1 in
  M.2 = xm + 2 ∧ 
  let d := 2 * Real.sqrt 5 in
  let k := 2 in let b := -1 in
  d = (Real.abs (k * xm - (xm + 2) + b)) / (Real.sqrt (1 + k^2)) ∧
  (M = (13, 15) ∨ M = (-7, -5)) := 
sorry

-- 3. Prove value of k for minimum distance sqrt(2) from line segment y=kx+4 to line y=x+2 is k=1
theorem value_of_k_for_min_distance : 
  ∃ k : ℚ, k = 1 ∧ 
  ∀ x : ℚ, -1 ≤ x ∧ x ≤ 2 → 
  let b := 4 in 
  let d := Real.sqrt 2 in
  let y := k * x + b in 
  let k' := 1 in let b' := 2 in
  d = (Real.abs (k' * x - y + b')) / (Real.sqrt (1 + k'^2)) := 
sorry

end distance_P_1_2_to_line_2x_minus_1_coordinates_of_M_on_line_x_plus_2_value_of_k_for_min_distance_l524_524739


namespace sandy_age_l524_524116

theorem sandy_age (S M : ℕ) (h1 : M = S + 18) (h2 : S * 9 = M * 7) : S = 63 := by
  sorry

end sandy_age_l524_524116


namespace general_formula_l524_524723

noncomputable def S : ℕ → ℕ
| 0       := 0
| (n + 1) := a (n + 1) + S n

noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := 2 * S n

theorem general_formula (n : ℕ) :
  a n = if n = 1 then 1 else 2 * 3 ^ (n - 2) := sorry

end general_formula_l524_524723


namespace count_lucky_numbers_eq_126_l524_524834

-- Define the sequence and lucky condition
def nextTerm (a : ℕ) : ℕ := a / 3

def isLucky (n : ℕ) : Prop :=
  let a := @Nat.iterate ℕ nextTerm n
  ∀ i, a i ≠ 0 → a i % 3 ≠ 0

-- Define the task to find the count of lucky numbers less than or equal to 1000
def countLuckyNumbers : ℕ :=
  Nat.card (SetOf (λ n : ℕ, n ≤ 1000 ∧ isLucky n))

-- The statement we need to prove
theorem count_lucky_numbers_eq_126 : countLuckyNumbers = 126 := by
  sorry

end count_lucky_numbers_eq_126_l524_524834


namespace least_three_digit_with_factors_l524_524948

-- Define the three prime factors
def a := 3
def b := 5
def c := 7

-- Define the candidate for the least three-digit positive integer
def candidate := 105

-- Define the properties of the candidate
def is_least_three_digit_with_factors (n : ℕ) : Prop :=
  (a ∣ n) ∧ (b ∣ n) ∧ (c ∣ n) ∧ (100 ≤ n) ∧ (n < 1000)

-- Prove that 105 is the least three-digit positive integer satisfying these properties
theorem least_three_digit_with_factors : is_least_three_digit_with_factors candidate := 
by {
  -- Proof of this theorem is out of scope, hence replaced with sorry
  sorry,
}

end least_three_digit_with_factors_l524_524948


namespace cube_diagonal_in_sphere_l524_524059

theorem cube_diagonal_in_sphere (R : ℝ) (hR : R > 0) : 
  let d := 2 * R in
  d = 2 * R :=
by
  sorry

end cube_diagonal_in_sphere_l524_524059


namespace solve_for_a_l524_524413

theorem solve_for_a (x : ℤ) (a : ℤ) (h : 3 * x + 2 * a + 1 = 2) (hx : x = -1) : a = 2 :=
by
  sorry

end solve_for_a_l524_524413


namespace sequence_correct_l524_524453

-- Define the sequence a_n
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n > 1, a (n + 1) - a n = (a 1) * ((⅓)^(n - 1))

noncomputable def a_n (n : ℕ) : ℝ :=
  (3/2) * (1 - (1/(3^n)))

-- Prove the sequence satisfies the given geometric properties
theorem sequence_correct (a : ℕ → ℝ) (n : ℕ) : sequence a → a n = a_n n := 
by
  intro h1,
  sorry

end sequence_correct_l524_524453


namespace number_of_points_P_l524_524280

theorem number_of_points_P (ABC : Triangle) (P : Point):
  scalene ABC →
  (A₁ B₁ C₁ : Points) →
  (A₂ B₂ C₂ : Points) →
  is_foot_of_perpendicular P ABC A₁ B₁ C₁ →
  is_intersection_with_circumcircle P ABC A₂ B₂ C₂ → 
  are_similar A₁ B₁ C₁ A₂ B₂ C₂ →
  are_similar A₁ B₁ C₁ ABC → 
  are_similar A₂ B₂ C₂ ABC → 
  ∃ (P_set : Set Point), 
  (P ∈ P_set → size_of P_set = 8) := sorry

end number_of_points_P_l524_524280


namespace find_missing_number_l524_524685

theorem find_missing_number {n : ℕ} (hn : n > 1) :
  ∏ k in {k | k = 11 ∨ k = 12 ∨ k = 13 ∨ ... ∨ k = 99 ∨ k = 100 ∨ k = n}.toFinset, (1 - (1 / k : ℝ)) = 0.09 → 
  n = 10 :=
sorry

end find_missing_number_l524_524685


namespace inequality_system_range_k_l524_524309

theorem inequality_system_range_k :
  (∃ x : ℤ, (x^2 - 2 * x - 8 > 0) ∧ (2 * x^2 + (2 * k + 7) * x + 7 * k < 0)) ↔
    (k ∈ (-5 : ℝ) ∪ Ico -5 3 ∪ Icc 4 5) :=
by
  sorry

end inequality_system_range_k_l524_524309


namespace time_second_pipe_filling_l524_524097

-- Definitions based on problem conditions:
def rate_first_pipe : ℝ := 1 / 10
def rate_third_pipe : ℝ := -1 / 50
def combined_fill_time : ℝ := 6.1224489795918355
def combined_rate : ℝ := 1 / combined_fill_time

-- Question translated to a proof problem:
theorem time_second_pipe_filling :
  ∃ T : ℝ, (rate_first_pipe + (1 / T) + rate_third_pipe = combined_rate) ∧ T = 191.58163265306122 :=
sorry

end time_second_pipe_filling_l524_524097


namespace original_number_l524_524976

theorem original_number (x : ℝ) (h : x * 1.20 = 1080) : x = 900 :=
sorry

end original_number_l524_524976


namespace sequence_periodicity_l524_524538

noncomputable def sequence (m : ℝ) : ℕ → ℝ
| 0       := m
| (n + 1) := -1 / (sequence n + 1)

theorem sequence_periodicity (m : ℝ) (hm : 0 < m) : sequence m 16 = m := 
sorry

end sequence_periodicity_l524_524538


namespace part_i_part_ii_l524_524302

def f (x : ℝ) : ℝ := real.log x - (x - 1)

theorem part_i (x : ℝ) (hx : x > 0) : f x ≤ 0 :=
sorry

def g (t : ℝ) : ℝ := t * real.log t / (t - 1)

theorem part_ii (a : ℝ) (ha : -e ^ (1 / (e - 1)) < a) : ∀ t ≥ e, g t > real.log (-a) + 1 :=
sorry

end part_i_part_ii_l524_524302


namespace valid_multiplications_display_26_l524_524886

open Nat

-- Definitions
def digit_four_button_broken (n : ℕ) : Prop :=
  ¬ (n.toString.contains '4')

def result_displayed (actual : ℕ) (display : ℕ) : Prop :=
  (display.toString = actual.toString.filter (λ c => c ≠ '4'))

-- Main theorem statement
theorem valid_multiplications_display_26 :
  {p : ℕ × ℕ // p.1 ∈ range 1 10 ∧ p.2 ∈ range 10 100 ∧ digit_four_button_broken p.1 ∧ digit_four_button_broken p.2 ∧ result_displayed (p.1 * p.2) 26}.size = 6 :=
sorry

end valid_multiplications_display_26_l524_524886


namespace radius_of_larger_circle_l524_524441

theorem radius_of_larger_circle (A O B : Point) (r R : ℝ) (K L : Point)
  (h_angle : ∠ A O B = 60) (h_r : r = 1) (h_tangent : tangent K L) :
  R = 3 :=
sorry

end radius_of_larger_circle_l524_524441


namespace quadratic_rewrite_l524_524486

theorem quadratic_rewrite :
  ∃ d e f : ℤ, (4 * (x : ℝ)^2 - 24 * x + 35 = (d * x + e)^2 + f) ∧ (d * e = -12) :=
by
  sorry

end quadratic_rewrite_l524_524486


namespace scientific_notation_of_12_06_million_l524_524853

theorem scientific_notation_of_12_06_million :
  12.06 * 10^6 = 1.206 * 10^7 :=
sorry

end scientific_notation_of_12_06_million_l524_524853


namespace largest_possible_difference_l524_524661

-- Define the given conditions
def cassandra_estimate_chicago : ℕ := 40000
def dave_estimate_denver : ℕ := 70000

def within_percentage (x : ℕ) (percent : ℕ) : set ℕ := 
  { n : ℕ | x * (100 - percent) / 100 ≤ n ∧ n ≤ x * (100 + percent) / 100 }

-- Define conditions
def chicago_attendance : set ℕ := within_percentage cassandra_estimate_chicago 5
def denver_attendance : set ℕ := within_percentage dave_estimate_denver 5

-- Define the proof problem
theorem largest_possible_difference :
  let max_denver := 73684  -- max possible D
  let min_chicago := 38000 -- min possible C 
  in abs (max_denver - min_chicago) = 36000 :=
by
  sorry

end largest_possible_difference_l524_524661


namespace maximum_value_l524_524603

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

theorem maximum_value (x1 x2 t : ℝ) (h1 : 0 < t) (h2 : f x1 = t) (h3 : g x2 = t) : 
  ∃ x1 x2, (t > 0) ∧ (f x1 = t) ∧ (g x2 = t) ∧ ((x1 / (x2 * Real.exp t)) = 1 / Real.exp 1) := 
sorry

end maximum_value_l524_524603


namespace prob_class1_two_mcq_from_A_expected_value_best_of_five_l524_524616

-- Part 1
theorem prob_class1_two_mcq_from_A :
  let P_B1 := (5.choose 2) / (8.choose 2)
  let P_B2 := (5.choose 1 * 3.choose 1) / (8.choose 2)
  let P_B3 := (3.choose 2) / (8.choose 2)
  let P_A_given_B1 := 6 / 9
  let P_A_given_B2 := 5 / 9
  let P_A_given_B3 := 4 / 9
  let P_A := P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2 + P_B3 * P_A_given_B3
  let P_B1_given_A := (P_B1 * P_A_given_B1) / P_A
  P_B1_given_A = 20 / 49 :=
by
  sorry

-- Part 2
theorem expected_value_best_of_five :
  let P_X3 := (3/5 * 2/5 * 2/5) + (2/5 * 2/5 * 2/5)
  let P_X4 := 3/5 * 3/5 * 3/5 * 2/5 + 2/5 * 3/5 * 3/5 * 2/5 + 2/5 * 2/5 * 3/5 * 3/5 + 2/5 * 3/5 * 2/5 * 2/5 + 3/5 * 3/5 * 3/5 * 3/5 + 3/5 * 2/5 * 3/5 * 3/5
  let P_X5 := 1 - P_X3 - P_X4
  let E_X := 3 * P_X3 + 4 * P_X4 + 5 * P_X5
  E_X = 537 / 125 :=
by
  sorry

end prob_class1_two_mcq_from_A_expected_value_best_of_five_l524_524616


namespace parabola_vertex_eq_l524_524203

theorem parabola_vertex_eq : 
  ∃ (x y : ℝ), y = -3 * x^2 + 6 * x + 1 ∧ (x = 1) ∧ (y = 4) := 
by
  sorry

end parabola_vertex_eq_l524_524203


namespace determine_m_l524_524429

theorem determine_m (m : ℤ)
  (h : ∃ k : ℤ, k = (√(2024 - 2023 * m) : ℚ) + (√(2023 - 2024 * m) : ℚ)) :
  m = -1 :=
by {
  sorry
}

end determine_m_l524_524429


namespace dice_sum_probability_l524_524107

theorem dice_sum_probability (a : ℕ) (h₀ : a ≠ 10) :
  (∃ dice_faces : Fin 7 → Fin 7, ∑ i, dice_faces i = 10) ↔ 
  (∃ dice_faces' : Fin 7 → Fin 7, ∑ i, dice_faces' i = a) :=
 ∀ dice_faces : Fin 7 → Fin 7, (∑ i, dice_faces i = 10) ↔ (∑ i, (7 - dice_faces i) = a) → a = 39 :=
by
  sorry

end dice_sum_probability_l524_524107


namespace nth_equation_l524_524012

theorem nth_equation (n : ℕ) : 
  (finset.range (2 * n)).sum (λ k, if even k then -1 / (k + 1) else 1 / (k + 1)) = 
  (finset.range (n)).sum (λ k, 1 / (n + 1 + k)) := 
sorry

end nth_equation_l524_524012


namespace min_value_quadratic_l524_524562

theorem min_value_quadratic :
  ∀ x : ℝ, let y := x^2 + 16 * x + 20 in y ≥ -44 :=
by
  sorry

end min_value_quadratic_l524_524562


namespace total_rent_of_pasture_l524_524636

theorem total_rent_of_pasture 
  (oxen_A : ℕ) (months_A : ℕ) (oxen_B : ℕ) (months_B : ℕ)
  (oxen_C : ℕ) (months_C : ℕ) (share_C : ℕ) (total_rent : ℕ) :
  oxen_A = 10 →
  months_A = 7 →
  oxen_B = 12 →
  months_B = 5 →
  oxen_C = 15 →
  months_C = 3 →
  share_C = 72 →
  total_rent = 280 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2 hC3
  sorry

end total_rent_of_pasture_l524_524636


namespace general_formula_of_a_sum_of_b_l524_524281

variables (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (n : ℕ)

noncomputable def aₙ := 2 * n

def a_conditions : Prop :=
  (a 1 + a 3 = 8) ∧ (a 6 + a 12 = 36)

def a_general_formula : Prop :=
  ∀ n : ℕ, a n = 2 * n

def b_conditions : Prop :=
  b 1 = 2 ∧ ∀ n : ℕ, b (n + 1) = a (n + 1) - 2 * a n

def b_sum : ℕ → ℤ
| 0       := 0
| (n + 1) := b (n + 1) + b_sum n

theorem general_formula_of_a (h : a_conditions a) : a_general_formula a :=
sorry

theorem sum_of_b (h1 : a_general_formula a) (h2 : b_conditions a b) :
  ∀ n : ℕ, b_sum b n = -(n ^ 2 : ℤ) + 3 * n :=
sorry

end general_formula_of_a_sum_of_b_l524_524281


namespace calculate_K_ion_l524_524182

-- Definition of conditions
def degree_ionization (C_ion C_total : ℝ) : ℝ := C_ion / C_total
def C_ion_value : ℝ := 1.33 * 10 ^ (-3)
def ionization_constant (NH4_plus OH_minus NH4OH : ℝ) : ℝ := (NH4_plus * OH_minus) / NH4OH

-- Main statement
theorem calculate_K_ion :
  ∀ (C_total : ℝ) (NH4OH : ℝ),
  degree_ionization C_ion_value C_total = 1.33 * 10 ^ (-2) →
  NH4OH = 0.1 →
  ionization_constant C_ion_value C_ion_value NH4OH = 1.76 * 10 ^ (-5) :=
by
  intros C_total NH4OH h_degree h_NH4OH
  sorry

end calculate_K_ion_l524_524182


namespace percent_prime_divisible_by_5_l524_524583

def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]
def primes_divisible_by_5 := [p in primes_less_than_20 | p % 5 = 0]

theorem percent_prime_divisible_by_5 :
  (primes_divisible_by_5.length / primes_less_than_20.length * 100 = 12.5) :=
by
  sorry

end percent_prime_divisible_by_5_l524_524583


namespace line_circle_no_intersection_l524_524357

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524357


namespace tiling_10000x10000_board_l524_524033

theorem tiling_10000x10000_board (board : matrix (fin 10000) (fin 10000) ℕ) (t_i t_j : ℕ) :
  (t_i = 4999 ∧ t_j = 4999) →
  ∃ central_tiling : bool, central_tiling ∧ (∀ (x : ℕ) (y : ℕ), (x < 5000 ∧ x ≥ 4997) → 
  (y < 5000 ∧ y ≥ 4997) → board (fin.mk x sorry) (fin.mk y sorry)  ≠ 0) 
  ∧ 
  (¬ central_tiling ∧ (∀ (a b : ℕ), (a = 0 ∨ a = 1) ∧ (b = 0 ∨ b = 1) → 
  ∃ (x y : fin 10000), (x.val - a + y.val - b) % 3 = 0 → 
  board x y = 0)) → 
  ∃ tiling_possible : bool, tiling_possible := 
by 
  sorry

end tiling_10000x10000_board_l524_524033


namespace percentage_of_primes_divisible_by_5_l524_524568

def primes_less_than_twenty : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def percentage (num total : ℕ) : ℚ := (num.to_rat / total.to_rat) * 100

theorem percentage_of_primes_divisible_by_5 :
  percentage (primes_less_than_twenty.count divisible_by_five) (primes_less_than_twenty.size) = 12.5 :=
by
  sorry

end percentage_of_primes_divisible_by_5_l524_524568


namespace largest_inner_sphere_radius_l524_524682

theorem largest_inner_sphere_radius :
  ∀ (spheres : Fin 8 → ℝ) (cube_side_length : ℝ) (sphere_diameter : ℝ),
    (∀ i, spheres i = sphere_diameter / 2) →
    cube_side_length = 40 →
    sphere_diameter = 20 →
    let r_inner := (Real.sqrt 300) - (sphere_diameter / 2)
    in r_inner ≈ 7.3 :=
by
  intro spheres cube_side_length sphere_diameter hs hc hd
  let r_inner := (Real.sqrt 300) - (sphere_diameter / 2)
  sorry

end largest_inner_sphere_radius_l524_524682


namespace digits_in_equation_l524_524690

theorem digits_in_equation :
  ∃ (a b c d e f g h i : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i ∧
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (a : ℚ) / b = (c : ℚ) / d ∧
    (a : ℚ) / b = (((e * 100) + (f * 10) + g) : ℚ) / (79 : ℚ) :=
begin
  -- given by the reference solution
  use 4, 2, 6, 3, 1, 5, 8, 7, 9,
  -- distinct digits from 1 to 9
  repeat { split },
  -- all elements are distinct
  all_goals { norm_num },
  -- set of elements is {1, 2, 3, 4, 5, 6, 7, 8, 9}
  refl,
  -- verify relationships
  all_goals { norm_num }
end


end digits_in_equation_l524_524690


namespace weighted_average_correct_l524_524840

-- Let x, y, and z be the average scores for three separate groups of students
variables (x y z : ℝ)

-- Define the number of students in each group
def students_group_a := 15
def students_group_b := 10
def students_group_c := 15
def total_students := students_group_a + students_group_b + students_group_c

-- Define the weighted average calculation
def weighted_average (x y z : ℝ) : ℝ :=
  (students_group_a * x + students_group_b * y + students_group_c * z) / total_students

-- The theorem we aim to prove
theorem weighted_average_correct :
  weighted_average x y z = (15 * x + 10 * y + 15 * z) / 40 :=
by
  unfold weighted_average
  unfold students_group_a students_group_b students_group_c total_students
  sorry

end weighted_average_correct_l524_524840


namespace ninth_grade_class_notification_l524_524072

theorem ninth_grade_class_notification (n : ℕ) (h1 : 1 + n + n * n = 43) : n = 6 :=
by
  sorry

end ninth_grade_class_notification_l524_524072


namespace pow_two_sub_one_not_square_l524_524417

theorem pow_two_sub_one_not_square (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end pow_two_sub_one_not_square_l524_524417


namespace hyperbola_vertex_distance_l524_524237

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 16 - y^2 / 9 = 1) → (vertex_distance : ℝ := 8) := sorry

end hyperbola_vertex_distance_l524_524237


namespace max_visible_sum_is_101_l524_524548

-- Define the given set of numbers on each cube's faces
def faces : finset ℕ := {1, 3, 5, 7, 9, 11}

-- The sums that appear when arranged to maximize visibility
def bottom_cube_sum_visible := 3 + 5 + 7 + 9 + 11
def middle_cube_sum_visible := 1 + 5 + 7 + 9 + 11
def top_cube_sum_visible := 1 + 5 + 7 + 9 + 11

-- The maximum visible sum of the cubes per the problem statement
def total_visible_sum := bottom_cube_sum_visible + middle_cube_sum_visible + top_cube_sum_visible

-- Assert that this sum is the maximum we will get (101)
theorem max_visible_sum_is_101 : total_visible_sum = 101 := by
  unfold total_visible_sum
  unfold bottom_cube_sum_visible
  unfold middle_cube_sum_visible
  unfold top_cube_sum_visible
  norm_num -- normalizing the numeric expressions
  done


end max_visible_sum_is_101_l524_524548


namespace y1_lt_y2_l524_524501

-- Define the linear function
def linear_function (x : ℝ) : ℝ := 8 * x - 1

-- Define the points P1 and P2
def P1 : ℝ × ℝ := (3, linear_function 3)
def P2 : ℝ × ℝ := (4, linear_function 4)

-- The proof statement that y1 is less than y2
theorem y1_lt_y2 : P1.2 < P2.2 :=
by
  unfold P1 P2 linear_function
  simp
  -- y1 is 23 and y2 is 31, so 23 < 31
  exact Nat.lt.base 22 -- or any method to show 23 < 31

end y1_lt_y2_l524_524501


namespace circle_tangent_to_line_l524_524737

theorem circle_tangent_to_line 
  (C_center : ℝ × ℝ)
  (h_center_parabola : C_center.2 ^ 2 = 4 * C_center.1)
  (C_passes_through_fixed_point : (C_center.1 - 1) ^ 2 + C_center.2 ^ 2 = 1)
  (line : ℝ × ℝ → Bool := λ p, p.1 + 1 = 0) :
  ∃ (r : ℝ), ∀ (p : ℝ × ℝ), ((p.1 - C_center.1)^2 + (p.2 - C_center.2)^2 = r^2) ∧ line p → False :=
begin
  sorry
end

end circle_tangent_to_line_l524_524737


namespace percentage_of_primes_divisible_by_5_l524_524564

def primes_less_than_twenty : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def percentage (num total : ℕ) : ℚ := (num.to_rat / total.to_rat) * 100

theorem percentage_of_primes_divisible_by_5 :
  percentage (primes_less_than_twenty.count divisible_by_five) (primes_less_than_twenty.size) = 12.5 :=
by
  sorry

end percentage_of_primes_divisible_by_5_l524_524564


namespace part_1_solution_set_part_2_range_of_a_l524_524683

def f (x : ℝ) : ℝ :=
if x < -1 / 2 then -x - 2
else if x <= 1 then 3 * x
else x + 2

theorem part_1_solution_set :
  { x : ℝ | |2 * x + 1| - |x - 1| ≤ log 2 4 } =
  { x : ℝ | -4 ≤ x ∧ x ≤ 2 / 3 } :=
sorry

theorem part_2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, |2 * x + 1| - |x - 1| ≤ log 2 a) ↔ a ≥ (real.sqrt 2) / 4 :=
sorry

end part_1_solution_set_part_2_range_of_a_l524_524683


namespace tangent_line_monotonicity_g_slope_k_l524_524301

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) := f x + a*x^2 - (2*a + 1)*x

theorem tangent_line (x : ℝ) (a : ℝ) : 
  a = 1 → x = 1 → (g x a = -2) := sorry

theorem monotonicity_g (a : ℝ) :
  (0 < a ∧ a < 1/2 → 
     ∀ x, (x > 0 ∧ x < 1 ∨ x > 1/(2*a) ∧ x > 0) ∧ 
          (x > 1 ∧ x < 1/(2*a) → g' x a < 0) ∧
          (x > 0 ∧ x < 1 ∨ (1/2*a < x)) ∧ (x > 1 ∧ x < 1/(2*a) → g' x a < 0)) ∧
  (a = 1/2 → ∀ x, (x > 0 → g' x a > 0)) ∧
  (a > 1/2 → 
     ∀ x, (x > 0 ∧ x < 1/(2*a) ∨ x > 1 ∧ x > 0) ∧ 
          (x > 1/(2*a) ∧ x < 1 → g' x a < 0)) := sorry

theorem slope_k (x1 x2 : ℝ) (a : ℝ) : 
  x1 < x2 → k = (f x2 - f x1) / (x2 - x1) → 1/x2 < k ∧ k < 1/x1 := sorry

end tangent_line_monotonicity_g_slope_k_l524_524301


namespace jellybean_count_l524_524212

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l524_524212


namespace largest_two_digit_number_l524_524945

theorem largest_two_digit_number :
  ∃ (A B C D : ℕ), {A, B, C, D} = {3, 9, 5, 8} ∧ A ≠ B ∧ C ≠ D ∧ 
  A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ 
  (98 = max (A * 10 + B) (B * 10 + A)) ∧ (C + D = 8) := 
  sorry

end largest_two_digit_number_l524_524945


namespace min_distance_tangent_exp_curve_l524_524753

noncomputable def f (x : ℝ) : ℝ := -1 * (exp x) + 2 * x

def tangent_line_at_origin (x : ℝ) : ℝ := x - 1

def exp_curve (x : ℝ) : ℝ := exp x

theorem min_distance_tangent_exp_curve :
  let P := (0 : ℝ, -1)
  let Q := (0 : ℝ, 1)
  dist P Q = sqrt 2 :=
by
  sorry

end min_distance_tangent_exp_curve_l524_524753


namespace length_of_platform_is_300_meters_l524_524161

-- Definitions used in the proof
def kmph_to_mps (v: ℕ) : ℕ := (v * 1000) / 3600

def speed := kmph_to_mps 72

def time_cross_man := 15

def length_train := speed * time_cross_man

def time_cross_platform := 30

def total_distance_cross_platform := speed * time_cross_platform

def length_platform := total_distance_cross_platform - length_train

theorem length_of_platform_is_300_meters :
  length_platform = 300 :=
by
  sorry

end length_of_platform_is_300_meters_l524_524161


namespace distribution_of_K_l524_524992

theorem distribution_of_K (x y z : ℕ) 
  (h_total : x + y + z = 370)
  (h_diff : y + z - x = 50)
  (h_prop : x * z = y^2) :
  x = 160 ∧ y = 120 ∧ z = 90 := by
  sorry

end distribution_of_K_l524_524992


namespace diophantine_solution_exists_l524_524605

theorem diophantine_solution_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x : Fin n → ℕ), (∑ i : Fin n, (1 : ℝ) / x i) + (1 / ∏ i : Fin n, x i) = 1 :=
by
  sorry

end diophantine_solution_exists_l524_524605


namespace sequence_perfect_square_property_l524_524482

-- Define the sequence a_n as specified
def a (n : ℕ) : ℚ :=
  1 / Real.sqrt 5 * ((1 + Real.sqrt 5) / 2) ^ n - 1 / Real.sqrt 5 * ((1 - Real.sqrt 5) / 2) ^ n

-- Prove that for infinitely many positive integers m, a_{m+4} * a_m - 1 is a perfect square
theorem sequence_perfect_square_property :
  ∃∞ m : ℕ, ∃ k : ℕ, a (m + 4) * a m - 1 = k ^ 2 :=
sorry

end sequence_perfect_square_property_l524_524482


namespace eccentricity_of_ellipse_l524_524298

variable (a b : ℝ) -- Declare the variables a and b as real numbers
variable (h : a > b ∧ b > 0) -- Declare the conditions a > b > 0

-- Define the eccentricity of an ellipse
def eccentricity (a b : ℝ) : ℝ := sqrt (1 - (b^2 / a^2))

-- Statement of the proof problem
theorem eccentricity_of_ellipse : 
  ∀ (a b : ℝ), a > b ∧ b > 0 → eccentricity a b = sqrt (1 - (b^2 / a^2)) := 
by
  intros a b h
  sorry

end eccentricity_of_ellipse_l524_524298


namespace projection_matrix_correct_l524_524823

def projection_matrix (v : ℝ^3) : ℝ^3 :=
  let n := ![2, -1, 2]  -- normal vector
  let c := (2 * v 0 + -1 * v 1 + 2 * v 2) / (2^2 + (-1)^2 + 2^2)  -- coefficient for projection
  v - c • n

def Q : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![5/9, 2/9, -4/9], 
    ![2/9, 10/9, -2/9], 
    ![-4/9, 2/9, 5/9]]

theorem projection_matrix_correct (v : ℝ^3) : 
  projection_matrix v = Q.mulVec v := 
  by sorry

end projection_matrix_correct_l524_524823


namespace solution_3across_is_295_l524_524681

noncomputable def digit_not_zero (d : ℕ) := d > 0 ∧ d < 10

def composite_factor_of_1001 (n : ℕ) := n ∣ 1001 ∧ n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

def is_prime (n : ℕ) := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def not_palindrome (n : ℕ) : Prop :=
  let digits := to_digits n in
  digits ≠ list.reverse digits

def p_q_prime_condition (p q : ℕ) := is_prime p ∧ is_prime q ∧ p ≠ q

def valid_5across (n : ℕ) (p q : ℕ) := p_q_prime_condition p q ∧ n = p * q^3

def valid_4down (n : ℕ) (p q : ℕ) := p_q_prime_condition p q ∧ n = p^3 * q

def valid_1down (n : ℕ) := ∃ (p : ℕ), is_prime p ∧ n = p + 1 ∧ ∀ m, is_prime m → n = m - 1

def valid_2down (n : ℕ) := n % 9 = 0

def valid_3across (n : ℕ) (d1 d2 d3 : ℕ) := digit_not_zero d1 ∧ digit_not_zero d2 ∧ digit_not_zero d3 ∧ 
  ¬palindrome n ∧ n = d1*100 + d2*10 + d3

theorem solution_3across_is_295 
  (d1 d2 d3 : ℕ) 
  (h1 : d1 = 2) 
  (h2 : d2 = 9) 
  (h3 : d3 = 5)
  (h1_not_zero : digit_not_zero d1)
  (h2_not_zero : digit_not_zero d2)
  (h3_not_zero : digit_not_zero d3)
  (valid_1_across : Valid_1Across 77) 
  (valid_5_across : valid_5across 24 3 2) 
  (valid_1_down : valid_1down 72) 
  (valid_2_down : valid_2down 792) 
  (valid_4_down : valid_4down 54 3 2) : 
  valid_3across 295 d1 d2 d3 :=
by
  sorry

end solution_3across_is_295_l524_524681


namespace third_circle_radius_l524_524550

theorem third_circle_radius :
  let r1 := 13
  let r2 := 23
  let A_shaded := π * r2 ^ 2 - π * r1 ^ 2
  ∃ r : ℝ, π * r ^ 2 = A_shaded ∧ r = 6 * Real.sqrt 10 :=
begin
  -- Definitions
  let r1 := 13
  let r2 := 23
  let A_shaded := π * r2 ^ 2 - π * r1 ^ 2,

  -- Proof
  use 6 * Real.sqrt 10,
  split,
  {
    -- Simplifying the area equation 
    sorry,
  },
  {
    -- Showing that r = 6 * sqrt(10)
    sorry,
  }
end

end third_circle_radius_l524_524550


namespace midpoint_segment_length_l524_524862

theorem midpoint_segment_length (A B C D E F G : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G]
  [midpoint AB C] [midpoint AC D] [midpoint AD E] [midpoint AE F] [midpoint AF G]
  (hG : dist A G = 4) : dist A B = 128 := by
  sorry

end midpoint_segment_length_l524_524862


namespace trapezoid_midline_length_l524_524053

theorem trapezoid_midline_length
  (OC OD : ℝ)
  (hOC : OC = 4)
  (hOD : OD = 8)
  (hTriangle : ∀ (O C D : Type) [MetricSpace O] [MetricSpace C] [MetricSpace D], right_triangle O C D):
  (midline_length O C D = (18 * real.sqrt 5) / 5) := by
  sorry

end trapezoid_midline_length_l524_524053


namespace installation_time_l524_524602

-- Definitions (based on conditions)
def total_windows := 14
def installed_windows := 8
def hours_per_window := 8

-- Define what we need to prove
def remaining_windows := total_windows - installed_windows
def total_install_hours := remaining_windows * hours_per_window

theorem installation_time : total_install_hours = 48 := by
  sorry

end installation_time_l524_524602


namespace hyperbola_vertex_distance_l524_524231

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l524_524231


namespace distance_AB_l524_524295

-- Definition of points A and B
def A : ℝ × ℝ × ℝ := (1, 1, 0)
def B : ℝ × ℝ × ℝ := (0, 1, 2)

-- The theorem stating the distance between points A and B is sqrt(5)
theorem distance_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2) = Real.sqrt 5 := by
  sorry

end distance_AB_l524_524295


namespace distribution_count_l524_524154

-- Let "distributions" be the set of all possible valid distributions
def valid_distribution (n : ℕ) (classroom : ℕ) (students : ℕ) : Prop :=
  n = classroom + students ∧ 2 ≤ classroom ∧ 3 ≤ students

def number_of_valid_distributions : ℕ :=
  (finset.range (8 + 1)).filter (λ classroom, valid_distribution 8 classroom (8 - classroom)).card

theorem distribution_count : number_of_valid_distributions = 4 := 
by {
  -- The set range here indicates the number of textbooks in the classroom
  simp [number_of_valid_distributions, valid_distribution, finset.filter, finset.card],
  -- We exclude invalid cases and count the valid ones manually
  sorry
}

end distribution_count_l524_524154


namespace sum_of_integral_c_l524_524703

theorem sum_of_integral_c (c : ℤ) : 
  (c ≤ 30) ∧ (∃ (k : ℤ), k^2 = 64 + 4 * c) → (∑ (i : ℤ) in {-16, -15, -12, -7, 0, 9, 20}.to_finset, i) = -11 :=
by 
  sorry

end sum_of_integral_c_l524_524703


namespace sum_factors_24_l524_524959

theorem sum_factors_24 : (∑ d in (finset.filter (λ d, 24 % d = 0) (finset.range (25))), d) = 60 :=
by
  sorry

end sum_factors_24_l524_524959


namespace geom_series_example_l524_524667

noncomputable def geom_series_sum (a r n : ℤ) : ℤ :=
  (a * (r ^ n - 1)) / (r - 1)

theorem geom_series_example :
  let a := 1
  let r := -3
  let n := 7
  let S := 547
  geom_series_sum a r n = S :=
by
  trivial

end geom_series_example_l524_524667


namespace cone_height_l524_524430

theorem cone_height (l : ℝ) (A : ℝ) (h : ℝ)
  (hl : l = 2 * real.sqrt 2) 
  (hA : A = 4) 
  (hslant : h = 2) 
  (area_eq : A = 0.5 * 2 * real.sqrt (l^2 - h^2) * h) :
  h = 2 :=
by 
  subst hslant
  linarith

end cone_height_l524_524430


namespace ratio_of_radii_l524_524799

-- Define the parameters and conditions
def blue_area (R r : ℝ) : ℝ := π * R ^ 2 - π * r ^ 2
def white_area (r : ℝ) : ℝ := π * r ^ 2

-- The main statement
theorem ratio_of_radii (R r : ℝ) (h : blue_area R r = 4 * white_area r) : r / R = 1 / Real.sqrt 5 :=
  sorry

end ratio_of_radii_l524_524799


namespace no_intersection_l524_524366

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524366


namespace mul_198_202_l524_524664

theorem mul_198_202 : 198 * 202 = 39996 := by
  calc
    198 * 202 = (200 - 2) * (200 + 2) : by rw [sub_add_eq_add_sub]
          ... = 200^2 - 2^2         : by rw [mul_sub, sub_mul, sub_sub, sub_self, zero_sub, sub_neg_eq_add, add_sub, sub_self, zero_add, mul_add, mul_add]
          ... = 40000 - 4           : by norm_num
          ... = 39996               : by norm_num

end mul_198_202_l524_524664


namespace sum_and_m_l524_524278

-- Define the sequence as described in the problem.
def sequence (a : ℕ → ℤ) (d : ℤ) (r : ℤ) (m : ℕ) : Prop :=
  (∀ n, 1 ≤ n ∧ n < m → a (n + 1) = a n + d) ∧ 
  (∀ n, n ≥ m - 1 → a (n + 1) = a n * r)

-- Define the first term and the required conditions
def conditions (a : ℕ → ℤ) (m : ℕ) : Prop :=
  a 1 = -2 ∧ m ≥ 4 ∧ 
  sequence a 2 2 m

-- Prove statement m = 4 and 
-- the sum of the first 6 terms S6 = 28.
theorem sum_and_m (a : ℕ → ℤ) (m S6 : ℕ) (h : conditions a m) : 
  m = 4 ∧ S6 = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) := 
sorry

end sum_and_m_l524_524278


namespace volume_of_circumscribed_sphere_l524_524521

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the function to calculate the space diagonal of a cube given its edge length
def space_diagonal (a : ℝ) : ℝ := a * Real.sqrt 3

-- Calculate the radius of the circumscribed sphere
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := (space_diagonal a) / 2

-- Define the volume of the sphere given its radius
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- The theorem to prove
theorem volume_of_circumscribed_sphere : volume_of_sphere (radius_of_circumscribed_sphere edge_length) = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end volume_of_circumscribed_sphere_l524_524521


namespace cuboid_length_l524_524228

theorem cuboid_length (b h : ℝ) (A : ℝ) (l : ℝ) : b = 6 → h = 5 → A = 120 → 2 * (l * b + b * h + h * l) = A → l = 30 / 11 :=
by
  intros hb hh hA hSurfaceArea
  rw [hb, hh] at hSurfaceArea
  sorry

end cuboid_length_l524_524228


namespace exists_irrational_sum_and_product_l524_524646

noncomputable def α : Real := Real.cbrt 2
noncomputable def β : Real := Real.cbrt 2
noncomputable def γ : Real := -2 * Real.cbrt 2

theorem exists_irrational_sum_and_product : 
  (∀ x : Real, ¬ Rational x → ∀ y : Real, ¬ Rational y → ∀ z : Real, ¬ Rational z → (x + y + z = 0) → (x * y * z = -4) → (x = α ∧ y = β ∧ z = γ)) :=
by
  intro x hx y hy z hz hsum hprod
  have h1 : x = α := sorry
  have h2 : y = β := sorry
  have h3 : z = γ := sorry
  exact ⟨h1, h2, h3⟩

end exists_irrational_sum_and_product_l524_524646


namespace limit_example_l524_524863

theorem limit_example (ε : ℝ) (hε : 0 < ε) :
  ∃ δ : ℝ, 0 < δ ∧ 
  (∀ x : ℝ, 0 < |x - 1/2| ∧ |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε) :=
sorry -- The proof is not provided

end limit_example_l524_524863


namespace volume_of_earth_dug_out_l524_524133

-- Definition of the volume of a cylindrical well
def volume_of_cylinder (r : ℝ) (h : ℝ) : ℝ :=
  real.pi * r^2 * h

-- Given conditions
def diameter := 2
def depth := 10

-- Derived condition from given conditions
def radius := diameter / 2

-- Main statement to be proved
theorem volume_of_earth_dug_out : volume_of_cylinder radius depth = 10 * real.pi := 
by
  sorry

end volume_of_earth_dug_out_l524_524133


namespace speed_with_stream_l524_524146

variable (V_m V_s : ℝ)

def against_speed : Prop := V_m - V_s = 13
def still_water_rate : Prop := V_m = 6

theorem speed_with_stream (h1 : against_speed V_m V_s) (h2 : still_water_rate V_m) : V_m + V_s = 13 := 
sorry

end speed_with_stream_l524_524146


namespace cost_of_burger_l524_524168

theorem cost_of_burger :
  ∃ (b s f : ℕ), 
    4 * b + 3 * s + f = 540 ∧
    3 * b + 2 * s + 2 * f = 580 ∧
    b = 100 :=
by {
  sorry
}

end cost_of_burger_l524_524168


namespace value_of_y_at_x8_l524_524419

theorem value_of_y_at_x8 (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = k * x^(1 / 3)) (h2 : f 64 = 4) : f 8 = 2 :=
sorry

end value_of_y_at_x8_l524_524419


namespace sum_of_positive_integer_factors_of_24_l524_524953

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l524_524953


namespace ellipse_standard_form_proof_l524_524282

open Classical
noncomputable section

def EllipseStandardForm : Prop := 
  ∀ (x y a b : ℝ), 
    a > 0 → b > 0 → a > b → e = (Real.sqrt (a^2 - b^2)) / a → 
    (e = Real.sqrt 2 / 2) →
    (x = Real.sqrt 6 / 2) → (y = 1 / 2) →
    ((x^2 / a^2) + (y^2 / b^2) = 1) →
    a^2 = 2 * b^2

theorem ellipse_standard_form_proof : EllipseStandardForm := 
begin
  -- Proof will go here
  sorry
end

end ellipse_standard_form_proof_l524_524282


namespace class1_draws_two_multiple_choice_questions_expected_value_of_X_l524_524613

-- Problem 1 in Lean 4 statement
theorem class1_draws_two_multiple_choice_questions (pB1 pB2 pB3 : ℚ)
    (pA_given_B1 pA_given_B2 pA_given_B3 pA : ℚ)
    (h1 : pB1 = 5 / 14) (h2 : pB2 = 15 / 28) (h3 : pB3 = 3 / 28)
    (h4 : pA_given_B1 = 6 / 9) (h5 : pA_given_B2 = 5 / 9) (h6 : pB3 = 4 / 9)
    (h7 : pA = (pB1 * 6 / 9) + (pB2 * 5 / 9) + (pB3 * 4 / 9)) :
  (pB1 * pA_given_B1 / pA) = 20 / 49 := by
  sorry

-- Problem 2 in Lean 4 statement
theorem expected_value_of_X (p3 p4 p5 eX : ℚ)
    (h1 : p3 = 4 / 25) (h2 : p4 = 48 / 125) (h3 : p5 = 57 / 125)
    (h4 : eX = 3 * p3 + 4 * p4 + 5 * p5) :
  eX = 537 / 125 := by
  sorry

end class1_draws_two_multiple_choice_questions_expected_value_of_X_l524_524613


namespace zero_of_func_in_interval_l524_524066

noncomputable def func (x : ℝ) : ℝ := (2 / x) - Real.log x

theorem zero_of_func_in_interval :
  ∃ x ∈ Ioo 2 3, func x = 0 := by
  sorry

end zero_of_func_in_interval_l524_524066


namespace midpoint_segment_length_l524_524861

theorem midpoint_segment_length (A B C D E F G : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G]
  [midpoint AB C] [midpoint AC D] [midpoint AD E] [midpoint AE F] [midpoint AF G]
  (hG : dist A G = 4) : dist A B = 128 := by
  sorry

end midpoint_segment_length_l524_524861


namespace range_of_k_one_integer_solution_l524_524311

theorem range_of_k_one_integer_solution :
  { k : ℝ // (∀ x : ℝ, x^2 - 2*x - 8 > 0 ↔ x < -2 ∨ x > 4) ∧ 
              (∀ x : ℝ, 2*x^2 + (2*k + 7)*x + 7*k < 0) } = 
  [| k in ((Ioo (-5 : ℝ) (3 : ℝ)) ∪ (Ioc (4 : ℝ) (5 : ℝ))) |] :=
by sorry

end range_of_k_one_integer_solution_l524_524311


namespace cubic_meter_to_cubic_feet_l524_524771

noncomputable def meter_to_feet : ℝ := 3.28084

theorem cubic_meter_to_cubic_feet :
  (meter_to_feet ^ 3) ≈ 35.3147 :=
by sorry

end cubic_meter_to_cubic_feet_l524_524771


namespace exists_set_A_with_property_l524_524507

open Nat
open Set

theorem exists_set_A_with_property :
  ∃ A : Set ℕ, (∀ S : Set ℕ, (Infinite S → (∀ p ∈ S, Prime p) → 
    ∃ (m ∈ A) (n ∉ A), ∃ k ≥ 2, ∃ T ⊆ S, (T.card = k ∧ (∏ i in T, i = m ∨ ∏ i in T, i = n)))) :=
sorry

end exists_set_A_with_property_l524_524507


namespace percent_primes_less_than_20_divisible_by_5_l524_524584

theorem percent_primes_less_than_20_divisible_by_5 :
  let primes := [ 2, 3, 5, 7, 11, 13, 17, 19 ]
  let divisible_by_5 := 1
  let total_primes := 8
  (divisible_by_5:ℚ ÷ total_primes * 100) = 12.5 := by 
  sorry

end percent_primes_less_than_20_divisible_by_5_l524_524584


namespace reduced_price_l524_524152

variable (P : ℝ)  -- the original price per kg
variable (reduction_factor : ℝ := 0.5)  -- 50% reduction
variable (extra_kgs : ℝ := 5)  -- 5 kgs more
variable (total_cost : ℝ := 800)  -- Rs. 800

theorem reduced_price :
  total_cost / (P * (1 - reduction_factor)) = total_cost / P + extra_kgs → 
  P / 2 = 80 :=
by
  sorry

end reduced_price_l524_524152


namespace find_k_of_parallelepiped_volume_l524_524918

theorem find_k_of_parallelepiped_volume 
  (k : ℝ) 
  (h_pos : k > 0)
  (h_volume : Abs (3 * k^2 - 11 * k + 6) = 20) : 
  k = 14 / 3 := 
sorry

end find_k_of_parallelepiped_volume_l524_524918


namespace forty_second_rising_number_does_not_contain_five_l524_524438

def is_rising (l : List ℕ) : Prop :=
  l = (l.nodup.erase{l} ∧ l.length = 5 ∧ ∀ i, i < l.length.pred → l.nth i < l.nth (i + 1))

def five_digit_rising_numbers : List (List ℕ) :=
  (Finset.range 9).powerset.filter (λ s, s.card = 5).toList.map Finset.toList

def nth_rising_number (n : ℕ) : option (List ℕ) :=
  let sorted := five_digit_rising_numbers.sort (λ a b, a.lex (<) b)
  sorted.nth n

def does_not_contain_digit (l : List ℕ) (d : ℕ) : Prop :=
  d ∉ l

theorem forty_second_rising_number_does_not_contain_five :
  does_not_contain_digit (nth_rising_number 41).get_or_else [] 5 :=
by sorry

end forty_second_rising_number_does_not_contain_five_l524_524438


namespace units_digit_x_pow_75_plus_6_eq_9_l524_524563

theorem units_digit_x_pow_75_plus_6_eq_9 (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9)
  (h3 : (x ^ 75 + 6) % 10 = 9) : x = 3 :=
sorry

end units_digit_x_pow_75_plus_6_eq_9_l524_524563


namespace line_circle_no_intersection_l524_524377

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524377


namespace largest_integer_x_l524_524679

theorem largest_integer_x (x : ℤ) : (x / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) → x ≤ 7 ∧ (7 / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) :=
by
  sorry

end largest_integer_x_l524_524679


namespace malia_first_bush_berries_l524_524485

theorem malia_first_bush_berries :
  ∃ n : ℕ, 
    ( ∀ (k : ℕ), k ≥ 2 → berry_count (k) = berry_count (k-1) + (2*k - 3)) ∧
    berry_count 2 = 4 ∧
    berry_count 3 = 7 ∧
    berry_count 4 = 12 ∧
    berry_count 5 = 19 ∧
    berry_count 1 = n :=
  ∃ n, 3 = n := by      
sorry

end malia_first_bush_berries_l524_524485


namespace line_circle_no_intersection_l524_524330

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524330


namespace problem_statement_l524_524109

theorem problem_statement : 2017 - (1 / 2017) = (2018 * 2016) / 2017 :=
by
  sorry

end problem_statement_l524_524109


namespace cos_in_third_quadrant_l524_524421

theorem cos_in_third_quadrant (B : Real) (h1 : (π < B) ∧ (B < 3 * π / 2)) (h2 : Real.sin B = -5/13) :
  Real.cos B = -12 / 13 :=
begin
  sorry
end

end cos_in_third_quadrant_l524_524421


namespace negation_of_union_l524_524312

theorem negation_of_union (x : α) (A B : set α) (p : x ∈ A ∪ B) : ¬(x ∈ A ∪ B) = (x ∉ A ∧ x ∉ B) := 
by 
  sorry

end negation_of_union_l524_524312


namespace percentage_of_primes_less_than_20_divisible_by_5_l524_524573

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 := {n : ℕ | n < 20 ∧ is_prime n}

def primes_less_than_20_divisible_by_5 := {n ∈ primes_less_than_20 | 5 ∣ n}

theorem percentage_of_primes_less_than_20_divisible_by_5 : 
  (primes_less_than_20_divisible_by_5.to_finset.card : ℝ) / (primes_less_than_20.to_finset.card : ℝ) * 100 = 12.5 :=
begin
  -- Proving this statement directly would involve showing the calculations explicitly.
  -- However, we just set up the framework here.
  sorry
end

end percentage_of_primes_less_than_20_divisible_by_5_l524_524573


namespace jellybeans_original_count_l524_524217

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l524_524217


namespace number_of_planters_l524_524873

variable (a b : ℕ)

-- Conditions
def tree_planting_condition_1 : Prop := a * b = 2013
def tree_planting_condition_2 : Prop := (a - 5) * (b + 2) < 2013
def tree_planting_condition_3 : Prop := (a - 5) * (b + 3) > 2013

-- Theorem stating the number of people who participated in the planting is 61
theorem number_of_planters (h1 : tree_planting_condition_1 a b) 
                           (h2 : tree_planting_condition_2 a b) 
                           (h3 : tree_planting_condition_3 a b) : 
                           a = 61 := 
sorry

end number_of_planters_l524_524873


namespace correct_statements_count_l524_524751

-- Definitions based on given conditions
def isMidpoint (A B M : Point) : Prop :=
  2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y

def isOnEllipse (P : Point) : Prop :=
  (P.x ^ 2) / 2 + (P.y ^ 2) / 4 = 1

def lineEquation (k : ℝ) (b : ℝ) (P : Point) : Prop :=
  P.y = k * P.x + b

def lineIntersection (k : ℝ) (b : ℝ) (P1 P2 : Point) : Prop :=
  isOnEllipse P1 ∧ isOnEllipse P2 ∧
  lineEquation k b P1 ∧ lineEquation k b P2

-- Prove the number of correct statements
theorem correct_statements_count (ellipse_eq : ∀ P, isOnEllipse P) :
  let P1 := Point.mk (x1) (y1);
      P2 := Point.mk (x2) (y2);
      M := Point.mk 1 1 in
  (¬(lineIntersection k_nonzero b ≠ (lineEquation (-1 / k_nonzero) b M))) ∧
  (isMidpoint P1 P2 M) ∧
  (lineEquation 1 1 P1 ∧ lineEquation 1 1 P2 ∧ M = Point.mk (1 / 3) (4 / 3)) ∧
  (lineEquation 1 2 P1 ∧ lineEquation 1 2 P2 ∧ dist P1 P2 = 4 * sqrt 2 / 3) →
  ∃ k b, count (statement_correct ellipse_eq k b) = 2 := sorry

end correct_statements_count_l524_524751


namespace foldable_box_configurations_l524_524193

theorem foldable_box_configurations : 
  ∀ (L_shape : fin 6 → ℕ × ℕ) (additional_square : ℕ × ℕ),
  (is_L_shape L_shape) →
  (is_adjacent L_shape additional_square) →
  (num_foldable_configurations L_shape additional_square = 4) :=
sorry

noncomputable def is_L_shape (L_shape : fin 6 → ℕ × ℕ) : Prop :=
  -- Definition that checks whether the given squares form an L-shape

noncomputable def is_adjacent (L_shape : fin 6 → ℕ × ℕ) (square : ℕ × ℕ) : Prop :=
  -- Definition that checks whether the given square is adjacent to the L-shape

noncomputable def num_foldable_configurations (L_shape : fin 6 → ℕ × ℕ) (square : ℕ × ℕ) : ℕ :=
  -- Definition that computes the number of foldable configurations

end foldable_box_configurations_l524_524193


namespace work_completion_time_equal_l524_524978

/-- Define the individual work rates of a, b, c, and d --/
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 6
def work_rate_c : ℚ := 1 / 12
def work_rate_d : ℚ := 1 / 10

/-- Define the combined work rate when they work together --/
def combined_work_rate : ℚ := work_rate_a + work_rate_b + work_rate_c + work_rate_d

/-- Define total work as one unit divided by the combined work rate --/
def total_days_to_complete : ℚ := 1 / combined_work_rate

/-- Main theorem to prove: When a, b, c, and d work together, they complete the work in 120/47 days --/
theorem work_completion_time_equal : total_days_to_complete = 120 / 47 :=
by
  sorry

end work_completion_time_equal_l524_524978


namespace min_value_max_value_l524_524028

theorem min_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 0) := sorry

theorem max_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 44) := sorry

end min_value_max_value_l524_524028


namespace exam_correct_answers_count_l524_524115

theorem exam_correct_answers_count (x y : ℕ) (h1 : x + y = 80) (h2 : 4 * x - y = 130) : x = 42 :=
by {
  -- (proof to be completed later)
  sorry
}

end exam_correct_answers_count_l524_524115


namespace distance_between_vertices_of_hyperbola_l524_524245

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) → 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l524_524245


namespace find_k_of_parallelepiped_volume_l524_524919

theorem find_k_of_parallelepiped_volume 
  (k : ℝ) 
  (h_pos : k > 0)
  (h_volume : Abs (3 * k^2 - 11 * k + 6) = 20) : 
  k = 14 / 3 := 
sorry

end find_k_of_parallelepiped_volume_l524_524919


namespace circumscribed_circle_around_hexagon_l524_524801

theorem circumscribed_circle_around_hexagon (A B C D E F : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F] 
  (h1 : parallel AB DE) (h2 : parallel BC EF) 
  (h3 : parallel CD FA) (h4 : AD = BE) (h5 : BE = CF) :
  ∃ (O : Type), exists_circle (O A B C D E F) :=
sorry

end circumscribed_circle_around_hexagon_l524_524801


namespace minimized_side_c_l524_524747

theorem minimized_side_c (t : ℝ) (C : ℝ) (hC : 0 < C ∧ C < π) :
  ∃ a b c : ℝ, a = b ∧ c = 2 * √(t * tan (C / 2)) ∧ 
  triangle_area_from_sides a b C = t ∧ 
  minimized_side_c_condition a b c C t :=
sorry

end minimized_side_c_l524_524747


namespace lines_and_planes_parallel_l524_524831

-- Assume m and n are different lines
-- Assume alpha and beta are different planes
-- Assume m parallel to n and m parallel to alpha

variables (m n : Line) (alpha beta : Plane)

axiom different_lines : m ≠ n
axiom different_planes : alpha ≠ beta
axiom parallel_m_n : m ∥ n
axiom parallel_m_alpha : m ∥ α

theorem lines_and_planes_parallel (m n : Line) (alpha beta : Plane) 
  (h1 : m ≠ n) (h2 : alpha ≠ beta) (h3: m ∥ n) (h4 : m ∥ alpha) : n ∥ α := 
by
  sorry

end lines_and_planes_parallel_l524_524831


namespace three_times_volume_l524_524134

-- Defining the volume of a cylinder
def volume_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

-- Given original cylinder dimensions 
def r1 : ℝ := 8
def h1 : ℝ := 6

-- Given the new cylinder dimensions
def r2 : ℝ := 12
def h2 : ℝ := 8

-- Proof that the volume of the new cylinder is three times the volume of the original cylinder
theorem three_times_volume : volume_cylinder r2 h2 = 3 * volume_cylinder r1 h1 := by
  sorry

end three_times_volume_l524_524134


namespace andrew_ant_probability_l524_524643

theorem andrew_ant_probability :
  let p_clockwise := (2/3 : ℚ)
  let p_counter_clockwise := (1/3 : ℚ)
  in (binomial 6 3) * (p_clockwise^3) * (p_counter_clockwise^3) = (160/729 : ℚ) := 
  sorry

end andrew_ant_probability_l524_524643


namespace value_of_expression_l524_524105

theorem value_of_expression (x : ℝ) (h : x = 3) : x^5 - 10 * x = 213 :=
by {
  rw [h],
  norm_num,
}

end value_of_expression_l524_524105


namespace range_of_a_f_function_properties_l524_524754

-- Defining the function f(x)
def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Domain of f(x) being (-1, 1)
def domain_f : Set ℝ := Set.Ioo (-1) 1

-- B must be a subset of A
def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

-- Lean statement for the first part of the problem
theorem range_of_a (a : ℝ) : (B a ⊆ domain_f) ↔ (a ∈ Set.Icc (-1) 0) :=
by {
    sorry
}

-- Lean statement for the second part of the problem
theorem f_function_properties : (∀ x : ℝ, x ∈ domain_f → f (-x) = -f x) ∧ 
                                (∃ x : ℝ, x ∈ domain_f ∧ f x ≠ f (-x)) :=
by {
    sorry
}

end range_of_a_f_function_properties_l524_524754


namespace full_time_worked_year_l524_524601

-- Define the conditions as constants
def total_employees : ℕ := 130
def full_time : ℕ := 80
def worked_year : ℕ := 100
def neither : ℕ := 20

-- Define the question as a theorem stating the correct answer
theorem full_time_worked_year : full_time + worked_year - total_employees + neither = 70 :=
by
  sorry

end full_time_worked_year_l524_524601


namespace ratio_of_dinner_to_lunch_is_two_l524_524816

noncomputable def breakfast := 500
noncomputable def lunch := breakfast + 0.25 * breakfast
noncomputable def shake := 300
noncomputable def total_daily_calories := 3275
noncomputable def total_calories_from_shakes := 3 * shake
noncomputable def total_calories_except_dinner := breakfast + lunch + total_calories_from_shakes
noncomputable def dinner := total_daily_calories - total_calories_except_dinner

theorem ratio_of_dinner_to_lunch_is_two : (dinner / lunch) = 2 :=
by
  sorry

end ratio_of_dinner_to_lunch_is_two_l524_524816


namespace compute_expression_l524_524663

theorem compute_expression :
  (5 + 7)^2 + 5^2 + 7^2 = 218 :=
by
  sorry

end compute_expression_l524_524663


namespace maximize_probability_when_n_is_12_l524_524091

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def valid_triplets (S : Finset ℕ) : Finset (Finset ℕ) :=
  { t | ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b + c = 20 }

def probability_with_n_removed (n : ℕ) : ℚ :=
  let S' := S.erase n in
  (valid_triplets S').card / (nat.choose S'.card 3)

theorem maximize_probability_when_n_is_12 :
  ∀ n ∈ S, probability_with_n_removed 12 ≥ probability_with_n_removed n :=
by
  sorry

end maximize_probability_when_n_is_12_l524_524091


namespace rectangle_side_length_l524_524673

theorem rectangle_side_length (b d : ℝ) (h : d > 0) (h' : d > b) : 
  ∃ a : ℝ, (d - a) = d - (d - sqrt (d^2 - b^2)) ∧ (a = d - sqrt (d^2 - b^2)) :=
by {
  sorry
}

end rectangle_side_length_l524_524673


namespace proof_problem_l524_524304

-- Definitions corresponding to the conditions
def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

-- The main theorem to be proved
theorem proof_problem 
  (φ : ℝ) (a b c A B : ℝ) 
  (hφ_range : 0 < φ ∧ φ < π)
  (h_f_passes : f (π / 12) φ = 1)
  (h_triangle_cond : a ^ 2 + b ^ 2 - c ^ 2 = a * b)
  (h_f_A2 : f (A / 2 + π / 12) φ = sqrt 2 / 2) : 
  φ = π / 3 ∧ sin B = (sqrt 2 + sqrt 6) / 4 := 
sorry

end proof_problem_l524_524304


namespace smallest_three_digit_number_satisfying_conditions_l524_524169

theorem smallest_three_digit_number_satisfying_conditions :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n + 6) % 9 = 0 ∧ (n - 4) % 6 = 0 ∧ n = 112 :=
by
  -- Proof goes here
  sorry

end smallest_three_digit_number_satisfying_conditions_l524_524169


namespace inequality_problem_l524_524606

-- Define the conditions for a, b, c.
variables a b c : ℝ
variables (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
variable (h_sum : a + 2 * b + 3 * c = 9)

-- State the theorem
theorem inequality_problem :
  1 / a + 1 / b + 1 / c ≥ 9 / 2 :=
sorry -- Proof to be implemented

end inequality_problem_l524_524606


namespace angle_of_inclination_l524_524694

theorem angle_of_inclination :
  ∀ (t : ℝ), (∃ (θ : ℝ), (θ = 230) ∧ (let x := 3 + t * Real.cos θ, y := -1 + t * Real.sin θ in
  ∀ θ', θ' = 50)) :=
by
suffices h : ∀ (θ : ℝ), θ = 230 → 50 = θ + 180 - 230 
exact h
sorry

end angle_of_inclination_l524_524694


namespace minimum_number_of_students_l524_524541

theorem minimum_number_of_students (b g : ℕ) (hb : 2 * b = 3 * g) : 17 :=
  sorry

end minimum_number_of_students_l524_524541


namespace percent_primes_less_than_20_divisible_by_5_l524_524587

theorem percent_primes_less_than_20_divisible_by_5 :
  let primes := [ 2, 3, 5, 7, 11, 13, 17, 19 ]
  let divisible_by_5 := 1
  let total_primes := 8
  (divisible_by_5:ℚ ÷ total_primes * 100) = 12.5 := by 
  sorry

end percent_primes_less_than_20_divisible_by_5_l524_524587


namespace line_circle_no_intersection_l524_524336

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524336


namespace trey_more_than_tim_l524_524932

theorem trey_more_than_tim :
  let Kristen := 18
  let Kris := Kristen * (1/3)
  let Trey := 9 * Kris
  let Tim := Kristen * (1/2)
  in Trey - Tim = 45 := by
sorry

end trey_more_than_tim_l524_524932


namespace problem_lean_l524_524096

noncomputable def intersection_area (XY YE FX EX YF : ℝ) (E F : ℝ) : ℝ :=
  let w := 18 / 5 in
  1/2 * XY * w

theorem problem_lean : 
  ∀ (XY YE FX EX YF : ℕ),
    XY = 12 → YE = 13 → FX = 13 → EX = 20 → YF = 20 →
    intersection_area XY YE FX EX YF 108 5 = 108 / 5 → 
    (108 + 5 = 113) := 
by 
  intros XY YE FX EX YF hXY hYE hFX hEX hYF hArea
  simp [hArea]
  sorry

end problem_lean_l524_524096


namespace line_circle_no_intersection_l524_524386

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524386


namespace min_value_of_reciprocal_sum_l524_524269

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 2) : 
  ∃ (z : ℝ), z = (1 / x + 1 / y) ∧ z = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_of_reciprocal_sum_l524_524269


namespace line_circle_no_intersection_l524_524400

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524400


namespace part_i_part_ii_l524_524164

-- Define the operations for the weird calculator.
def Dsharp (n : ℕ) : ℕ := 2 * n + 1
def Dflat (n : ℕ) : ℕ := 2 * n - 1

-- Define the initial starting point.
def initial_display : ℕ := 1

-- Define a function to execute a sequence of button presses.
def execute_sequence (seq : List (ℕ → ℕ)) (initial : ℕ) : ℕ :=
  seq.foldl (fun x f => f x) initial

-- Problem (i): Prove there is a sequence that results in 313 starting from 1 after eight presses.
theorem part_i : ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = 313 :=
by sorry

-- Problem (ii): Describe all numbers that can be achieved from exactly eight button presses starting from 1.
theorem part_ii : 
  ∀ n : ℕ, n % 2 = 1 ∧ n < 2^9 →
  ∃ seq : List (ℕ → ℕ), seq.length = 8 ∧ execute_sequence seq 1 = n :=
by sorry

end part_i_part_ii_l524_524164


namespace men_per_table_l524_524163

theorem men_per_table (total_customers total_tables women_per_table : ℕ) 
  (h1 : total_tables = 7) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 63) 
  : (total_customers - total_tables * women_per_table) / total_tables = 2 :=
by
  have total_women := total_tables * women_per_table,
  have total_men := total_customers - total_women,
  have men_per_table := total_men / total_tables,
  rw [h1, h2, h3],
  norm_num,
  exact men_per_table,
  sorry  -- proof is omitted

end men_per_table_l524_524163


namespace total_number_of_fish_l524_524870

-- Define the number of each type of fish
def goldfish : ℕ := 23
def blue_fish : ℕ := 15
def angelfish : ℕ := 8
def neon_tetra : ℕ := 12

-- Theorem stating the total number of fish
theorem total_number_of_fish : goldfish + blue_fish + angelfish + neon_tetra = 58 := by
  sorry

end total_number_of_fish_l524_524870


namespace common_region_area_l524_524555

-- Definitions for the problem
def square_side_length : ℝ := 3
def circle_radius : ℝ := 3

-- The theorem statement
theorem common_region_area :
  ∃ (s : ℝ) (r : ℝ), 
    s = 3 ∧ r = 3 ∧
    ∀ (A B C D : ℝ), 
      A = square_side_length ∧
      B = square_side_length ∧
      C = square_side_length ∧
      D = square_side_length →
      let common_area := 3 * π + 9 * (1 - real.sqrt 3) in
      common_area = 3 * π + 9 * (1 - real.sqrt 3) := sorry

end common_region_area_l524_524555


namespace malcolm_route_fraction_l524_524484

def time_spent_on_last_stage_fraction : ℕ :=
let time_uphill := 6 in
let time_path := 2 * time_uphill in
let fraction_time := 1 / 3 in
fraction_time

theorem malcolm_route_fraction:
  let time_uphill := 6 in
  let time_path := 2 * time_uphill in
  let total_first_route := time_uphill + time_path + x * (time_uphill + time_path) in
  let time_flat := 14 in
  let total_second_route := time_flat + 2 * time_flat in
  total_second_route = total_first_route + 18 →
  fraction_time = 1 / 3 :=
by
  let x := (1 : ℚ) / 3
  sorry

end malcolm_route_fraction_l524_524484


namespace convergent_inequalities_l524_524867

theorem convergent_inequalities (α : ℝ) (P Q : ℕ → ℤ) (h_convergent : ∀ n ≥ 1, abs (α - P n / Q n) < 1 / (2 * (Q n) ^ 2) ∨ abs (α - P (n - 1) / Q (n - 1)) < 1 / (2 * (Q (n - 1))^2))
  (h_continued_fraction : ∀ n ≥ 1, P (n-1) * Q n - P n * Q (n-1) = (-1)^(n-1)) :
  ∃ p q : ℕ, 0 < q ∧ abs (α - p / q) < 1 / (2 * q^2) :=
sorry

end convergent_inequalities_l524_524867


namespace tangent_sum_eq_l524_524819

-- Define the equilateral triangle ABC and its circumcircle k
variables (ABC : Type) [triangle_eq ABC] (k : circle ABC)

-- Define the external circle q that touches k at point D
variables (q : circle) (D : point)

-- Points D and C lie on different sides of the line AB
variables (A B C : point) (h₁ : different_sides D C (line A B))

-- Tangents from A, B, and C to circle q
variables (a b c : ℝ) (tangent_length_a : tangent_length A q = a)
  (tangent_length_b : tangent_length B q = b)
  (tangent_length_c : tangent_length C q = c)

-- The proof we need to show: a + b = c
theorem tangent_sum_eq (ABC : Type) [triangle_eq ABC] (k : circle ABC) (q : circle) (D : point)
                       (A B C : point) (h₁ : different_sides D C (line A B))
                       (a b c : ℝ)
                       (tangent_length_a : tangent_length A q = a)
                       (tangent_length_b : tangent_length B q = b)
                       (tangent_length_c : tangent_length C q = c) :
  a + b = c := 
sorry

end tangent_sum_eq_l524_524819


namespace number_of_shirts_l524_524789

-- Definitions based on the conditions
def total_price_shirts : ℝ := 400
def number_of_sweaters : ℝ := 75
def total_price_sweaters : ℝ := 1500
def price_difference : ℝ := 4

-- Given conditions
def avg_price_sweater : ℝ := total_price_sweaters / number_of_sweaters
def avg_price_shirt : ℝ := avg_price_sweater - price_difference 

-- Statement to be proven
theorem number_of_shirts : ∃ S : ℝ, S * avg_price_shirt = total_price_shirts ∧ S = 25 := 
by {
  -- Proof will go here
  sorry
}

end number_of_shirts_l524_524789


namespace value_of_k_l524_524707

theorem value_of_k : ∃ k : ℕ, 64 / k = 4 ∧ k = 16 := by
  use 16
  split
  · norm_num
  · norm_num

end value_of_k_l524_524707


namespace multiplication_factor_l524_524049

theorem multiplication_factor
  (n : ℕ) (avg_orig avg_new : ℝ) (F : ℝ)
  (H1 : n = 7)
  (H2 : avg_orig = 24)
  (H3 : avg_new = 120)
  (H4 : (n * avg_new) = F * (n * avg_orig)) :
  F = 5 :=
by {
  sorry
}

end multiplication_factor_l524_524049


namespace line_circle_no_intersect_l524_524391

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524391


namespace find_magnitude_of_z_l524_524719

noncomputable def magnitude_of_z (z : ℂ) (h : i * z = 1 + real.sqrt 3 * i) : ℝ :=
  ∥z∥

theorem find_magnitude_of_z (z : ℂ) (h : i * z = 1 + real.sqrt 3 * i) : magnitude_of_z z h = 2 := 
  by
    sorry

end find_magnitude_of_z_l524_524719


namespace sequence_term_exists_l524_524315

noncomputable def a_n (n : ℕ) (hn : n > 0) : ℚ :=
  1 / (n * (n + 2))

theorem sequence_term_exists :
  ∃ (n : ℕ) (hn : n > 0), a_n n hn = 1 / 120 :=
by
  use 10
  split
  · exact nat.succ_pos'
  · exact eq.refl _
  sorry

end sequence_term_exists_l524_524315


namespace binom_19_10_proof_l524_524189

theorem binom_19_10_proof (b_17_7 : Nat) (b_17_9 : Nat) (h1 : b_17_7 = 19448) (h2 : b_17_9 = 24310) :
  binomial 19 10 = 87516 := by
  sorry

end binom_19_10_proof_l524_524189


namespace modulus_z_l524_524619

noncomputable def z : ℂ := 4 / (complex.I - 1)

theorem modulus_z : complex.abs z = 2 * real.sqrt 2 :=
by
  sorry

end modulus_z_l524_524619


namespace line_circle_no_intersection_l524_524379

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524379


namespace product_PA_PB_is_correct_l524_524450

noncomputable def PA_PB_product : ℝ := 
  let P := (1, 0)
  let line := λ θ : ℝ, (ρ * cos θ - ρ * sin θ - 1 = 0)
  let ellipse := λ θ : ℝ, (x = 2 * cos θ ∧ y = sin θ)
  let standard_ellipse := λ x y : ℝ, (x ^ 2 + 4 * y ^ 2 = 4)
  let A := ∃ θ : ℝ, (x = 1 + (sqrt 2 / 2) * θ ∧ y = (sqrt 2 / 2) * θ)
  let B := ∃ θ : ℝ, (x = 1 + (sqrt 2 / 2) * θ ∧ y = (sqrt 2 / 2) * θ)
  let quadratic_solution := λ t : ℝ, (5 * t ^ 2 + 2 * sqrt 2 * t - 6 = 0)
  let product_of_roots := λ t1 t2 : ℝ, t1 * t2 = (6 / 5)
  let PA_PB := |product_of_roots t1 t2|
  PA_PB

theorem product_PA_PB_is_correct :
  PA_PB_product = (6 / 5) :=
by 
  sorry

end product_PA_PB_is_correct_l524_524450


namespace expected_value_of_white_balls_l524_524624

-- Definitions for problem conditions
def totalBalls : ℕ := 6
def whiteBalls : ℕ := 2
def redBalls : ℕ := 4
def ballsDrawn : ℕ := 2

-- Probability calculations
def P_X_0 : ℚ := (Nat.choose 4 2) / (Nat.choose totalBalls ballsDrawn)
def P_X_1 : ℚ := ((Nat.choose whiteBalls 1) * (Nat.choose redBalls 1)) / (Nat.choose totalBalls ballsDrawn)
def P_X_2 : ℚ := (Nat.choose whiteBalls 2) / (Nat.choose totalBalls ballsDrawn)

-- Expected value calculation
def expectedValue : ℚ := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2)

theorem expected_value_of_white_balls :
  expectedValue = 2 / 3 :=
by
  sorry

end expected_value_of_white_balls_l524_524624


namespace length_of_room_l524_524070

theorem length_of_room {L : ℝ} (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 4)
  (h2 : cost_per_sqm = 750)
  (h3 : total_cost = 16500) :
  L = 5.5 ↔ (L * width) * cost_per_sqm = total_cost := 
by
  sorry

end length_of_room_l524_524070


namespace permutations_remainder_l524_524822

theorem permutations_remainder (s : String) : 
  (let M := number_of_valid_permutations s ("AAAABBBBBCCCCDDD".toList) in
  M % 1000 = 75) :=
by
  let number_of_valid_permutations : String → List Char → ℕ := sorry
  sorry

end permutations_remainder_l524_524822


namespace coordinates_after_5_seconds_l524_524499

-- Define the initial coordinates of point P
def initial_coordinates : ℚ × ℚ := (-10, 10)

-- Define the velocity vector of point P
def velocity_vector : ℚ × ℚ := (4, -3)

-- Asserting the coordinates of point P after 5 seconds
theorem coordinates_after_5_seconds : 
   initial_coordinates + 5 • velocity_vector = (10, -5) :=
by 
  sorry

end coordinates_after_5_seconds_l524_524499


namespace fraction_one_bedroom_apartments_l524_524433

theorem fraction_one_bedroom_apartments :
  ∃ x : ℝ, (x + 0.33 = 0.5) ∧ x = 0.17 :=
by
  sorry

end fraction_one_bedroom_apartments_l524_524433


namespace largest_prime_factor_sum_of_divisors_l524_524472

/-- Let M be the sum of the divisors of 180. Prove that the largest prime factor of M is 13. -/
theorem largest_prime_factor_sum_of_divisors :
  let n := 180
  let s := Finset.sum (Finset.divisors n)
  Nat.prime_factorization.max' (Nat.divisors (Nat.sum_divisors_prime n)) = 13 := by
  sorry

end largest_prime_factor_sum_of_divisors_l524_524472


namespace line_circle_no_intersection_l524_524332

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524332


namespace centers_of_squares_on_parallelogram_form_square_l524_524040

theorem centers_of_squares_on_parallelogram_form_square
  (A B C D P Q R S : Type)
  [AddCommGroup P] [AddCommGroup Q] [AddCommGroup R] [AddCommGroup S]
  [VectorSpace ℝ P] [VectorSpace ℝ Q] [VectorSpace ℝ R] [VectorSpace ℝ S]
  (parallelogram : Parallelogram ABCD)
  (square1 : Square P A B)
  (square2 : Square Q B C)
  (square3 : Square R C D)
  (square4 : Square S D A)
  (h1 : Center P P A B)
  (h2 : Center Q Q B C)
  (h3 : Center R R C D)
  (h4 : Center S S D A) :
  IsSquare P Q R S :=
sorry

end centers_of_squares_on_parallelogram_form_square_l524_524040


namespace monotonic_increasing_iff_inequality_midpoint_l524_524303

-- Define the function f(x)
def f (x a : ℝ) := log x - a * (x - 1) / (x + 1)

-- Define the derivative of the function f(x)
def f' (x a : ℝ) := 1 / x - a * ((x + 1) - (x - 1)) / (x + 1) ^ 2

-- Monotonicity statement: \( f(x) \) is monotonically increasing on \( (0, +\infty) \) if \( a \leq 2 \)
theorem monotonic_increasing_iff (a : ℝ) : (∀ x > 0, f' x a ≥ 0) ↔ a ≤ 2 :=
sorry

-- Prove inequality involving midpoint M and slope k
theorem inequality_midpoint (k m n : ℝ) (h_mn : m > n ∧ n > 0) : 
  let x0 := (m + n) / 2 in
  k * x0 > 1 :=
sorry

end monotonic_increasing_iff_inequality_midpoint_l524_524303


namespace yuna_initial_pieces_l524_524847

variable (Y : ℕ)

theorem yuna_initial_pieces
  (namjoon_initial : ℕ := 250)
  (given_pieces : ℕ := 60)
  (namjoon_after : namjoon_initial - given_pieces = Y + given_pieces - 20) :
  Y = 150 :=
by
  sorry

end yuna_initial_pieces_l524_524847


namespace find_correct_value_l524_524597

theorem find_correct_value (incorrect_value : ℝ) (subtracted_value : ℝ) (added_value : ℝ) (h_sub : subtracted_value = -added_value)
(h_incorrect : incorrect_value = 8.8) (h_subtracted : subtracted_value = -4.3) (h_added : added_value = 4.3) : incorrect_value + added_value + added_value = 17.4 :=
by
  sorry

end find_correct_value_l524_524597


namespace solve_inequality_l524_524034

theorem solve_inequality (x : ℝ) : (x - 2) / (x + 5) ≥ 0 ↔ x ∈ set.Iio (-5) ∪ set.Ici 2 :=
by {
  sorry
}

end solve_inequality_l524_524034


namespace area_ratio_ABCP_EFGHPQ_l524_524445

theorem area_ratio_ABCP_EFGHPQ (A B C D E F G H P Q : Point)
  (h_octagon : RegularOctagon A B C D E F G H)
  (h_midpoint_P : P = midpoint C D)
  (h_midpoint_Q : Q = midpoint G H) :
  area (quadrilateral A B C P) / area (hexagon E F G H P Q) = 1 / 2 := 
sorry

end area_ratio_ABCP_EFGHPQ_l524_524445


namespace find_a8_l524_524063

noncomputable def a (n : ℕ) : ℤ := sorry

noncomputable def b (n : ℕ) : ℤ := a (n + 1) - a n

theorem find_a8 :
  (a 1 = 3) ∧
  (∀ n : ℕ, b n = b 1 + n * 2) ∧
  (b 3 = -2) ∧
  (b 10 = 12) →
  a 8 = 3 :=
by sorry

end find_a8_l524_524063


namespace smallest_N_is_1005_l524_524471

noncomputable def smallest_N (p q r s t : ℕ) (h_positive : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0) 
    (h_sum : p + q + r + s + t = 2015) : ℕ := 
  max (max (p + q) (q + r)) (max (r + s) (s + t))

theorem smallest_N_is_1005 : ∀ (p q r s t : ℕ),
  p > 0 → q > 0 → r > 0 → s > 0 → t > 0 → p + q + r + s + t = 2015 → smallest_N p q r s t = 1005 :=
by
  intros p q r s t hp hq hr hs ht hsum
  sorry

end smallest_N_is_1005_l524_524471


namespace max_black_horses_theorem_l524_524220

def chessboard : Type := fin 8 × fin 8

def is_knight_move (p1 p2 : chessboard) : Prop :=
  (abs (p1.1 - p2.1) = 2 ∧ abs (p1.2 - p2.2) = 1) ∨ 
  (abs (p1.1 - p2.1) = 1 ∧ abs (p1.2 - p2.2) = 2)
  
def max_black_horses : ℕ := 16

theorem max_black_horses_theorem :
  ∀ (placement : chessboard → bool), 
    (∀ (p1 p2 : chessboard), 
      placement p1 = tt → placement p2 = tt → is_knight_move p1 p2 → ¬ (∃ p3 : chessboard, p3 ≠ p1 ∧ p3 ≠ p2 ∧ placement p3 = tt ∧ is_knight_move p2 p3)) →
    (∑ p, if placement p then 1 else 0) ≤ max_black_horses := 
sorry

end max_black_horses_theorem_l524_524220


namespace analogical_reasoning_proof_l524_524078

-- Definitions based on conditions
def plane_geometry_fact : Prop :=
  ∀ (triangle : Type) (center : triangle → ℝ) (side1 side2 side3: triangle → ℝ), 
    (center triangle = side1 triangle) ∧ (center triangle = side2 triangle) ∧ (center triangle = side3 triangle)

def space_geometry_fact : Prop :=
  ∀ (tetrahedron : Type) (center : tetrahedron → ℝ) (face1 face2 face3 face4 : tetrahedron → ℝ), 
    (center tetrahedron = face1 tetrahedron) ∧ (center tetrahedron = face2 tetrahedron) ∧ (center tetrahedron = face3 tetrahedron) ∧ (center tetrahedron = face4 tetrahedron)

-- The proof problem to show the reasoning type
theorem analogical_reasoning_proof (h1: plane_geometry_fact) (h2: space_geometry_fact) : 
  ¬(A) ∧ B ∧ ¬(C) ∧ ¬(D) := 
by 
  sorry -- Proof will be done separately

end analogical_reasoning_proof_l524_524078


namespace line_circle_no_intersect_l524_524396

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524396


namespace percentage_difference_l524_524980

theorem percentage_difference:
  let x1 := 0.4 * 60
  let x2 := 0.8 * 25
  x1 - x2 = 4 :=
by
  sorry

end percentage_difference_l524_524980


namespace complex_number_is_real_l524_524426

theorem complex_number_is_real (x : ℝ) (z : ℂ) : z = complex.log (x^2 - 3*x - 2) / complex.log 2 + complex.I * (complex.log (x - 3) / complex.log 2) → (z.im = 0) → x = 4 :=
by
  intros h1 h2
  have hx: x - 3 = (2:ℝ),
  {
    sorry, -- Detailed proof step checking that log_2(x - 3) = 0 implies x - 3 = 1.
  }
  exact hx.symm


end complex_number_is_real_l524_524426


namespace sum_of_three_distinct_integers_product_625_l524_524906

theorem sum_of_three_distinct_integers_product_625 :
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 5^4 ∧ a + b + c = 131 :=
by
  sorry

end sum_of_three_distinct_integers_product_625_l524_524906


namespace primitive_triples_l524_524692

theorem primitive_triples :
  { (a, b, c) : ℕ × ℕ × ℕ // a ≤ b ∧ b ≤ c ∧ ∃ k, k ∈ {a, b, c} \{ 1 } 
                         ∧ a ∣ (b + c) 
                         ∧ b ∣ (a + c) 
                         ∧ c ∣ (a + b)
                         ∧ Nat.gcd a b = 1 
                         ∧ Nat.gcd b c = 1 
                         ∧ Nat.gcd a c = 1 } =
  { (1, 1, 1), (1, 1, 2), (1, 2, 3) } :=
by
  sorry

end primitive_triples_l524_524692


namespace sin_minus_cos_eq_one_l524_524227

theorem sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) : x = Real.pi / 2 :=
by sorry

end sin_minus_cos_eq_one_l524_524227


namespace triangular_25_l524_524101

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l524_524101


namespace complement_U_P_l524_524743

def U (y : ℝ) : Prop := y > 0
def P (y : ℝ) : Prop := 0 < y ∧ y < 1/3

theorem complement_U_P :
  {y : ℝ | U y} \ {y : ℝ | P y} = {y : ℝ | y ≥ 1/3} :=
by
  sorry

end complement_U_P_l524_524743


namespace dealer_overall_gain_is_9_89_l524_524137

-- Definitions for the problem conditions
def actual_weight (claimed_weight actual_per_claimed : ℝ) : ℝ := 
  claimed_weight * (actual_per_claimed / 1000)

def gain_weight (claimed_weight actual_weight : ℝ) : ℝ := 
  claimed_weight - actual_weight

def gain_percentage (gain actual_weight : ℝ) : ℝ := 
  (gain / actual_weight) * 100

def total_gain_and_actual_weight (items : List (ℝ × ℝ)) : (ℝ × ℝ) :=
  items.foldl (λ (acc : ℝ × ℝ) (item : ℝ × ℝ) => 
    (acc.1 + item.1, acc.2 + item.2)) (0, 0)

def overall_gain_percentage (total_gain total_actual_weight : ℝ) : ℝ := 
  (total_gain / total_actual_weight) * 100

-- Specific conditions
def potatoes_claimed_weight := 10
def potatoes_actual_per_claimed := 900

def onions_claimed_weight := 15
def onions_actual_per_claimed := 850

def carrots_claimed_weight := 25
def carrots_actual_per_claimed := 950

-- Quantities calculation
def potatoes_actual_weight := actual_weight potatoes_claimed_weight potatoes_actual_per_claimed
def onions_actual_weight := actual_weight onions_claimed_weight onions_actual_per_claimed
def carrots_actual_weight := actual_weight carrots_claimed_weight carrots_actual_per_claimed

def potatoes_gain := gain_weight potatoes_claimed_weight potatoes_actual_weight
def onions_gain := gain_weight onions_claimed_weight onions_actual_weight
def carrots_gain := gain_weight carrots_claimed_weight carrots_actual_weight

-- Total values
def total_gain_and_weight := total_gain_and_actual_weight [
  (potatoes_gain, potatoes_actual_weight),
  (onions_gain, onions_actual_weight),
  (carrots_gain, carrots_actual_weight)
]

def total_gain := total_gain_and_weight.1
def total_actual_weight := total_gain_and_weight.2

def dealer_overall_gain_percentage := overall_gain_percentage total_gain total_actual_weight

-- The proof statement
theorem dealer_overall_gain_is_9_89 : dealer_overall_gain_percentage ≈ 9.89 :=
by
  sorry

end dealer_overall_gain_is_9_89_l524_524137


namespace trig_identity_l524_524732

theorem trig_identity (α : ℝ) (h0 : Real.tan α = Real.sqrt 3) (h1 : π < α) (h2 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_identity_l524_524732


namespace distance_between_vertices_of_hyperbola_l524_524246

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) → 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l524_524246


namespace line_circle_no_intersection_l524_524385

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524385


namespace average_salary_all_workers_l524_524515

/-- The total number of workers in the workshop is 15 -/
def total_number_of_workers : ℕ := 15

/-- The number of technicians is 5 -/
def number_of_technicians : ℕ := 5

/-- The number of other workers is given by the total number minus technicians -/
def number_of_other_workers : ℕ := total_number_of_workers - number_of_technicians

/-- The average salary per head of the technicians is Rs. 800 -/
def average_salary_per_technician : ℕ := 800

/-- The average salary per head of the other workers is Rs. 650 -/
def average_salary_per_other_worker : ℕ := 650

/-- The total salary for all the workers -/
def total_salary : ℕ := (number_of_technicians * average_salary_per_technician) + (number_of_other_workers * average_salary_per_other_worker)

/-- The average salary per head of all the workers in the workshop is Rs. 700 -/
theorem average_salary_all_workers :
  total_salary / total_number_of_workers = 700 := by
  sorry

end average_salary_all_workers_l524_524515


namespace no_intersection_l524_524371

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524371


namespace cyclic_quadrilateral_EFYZ_l524_524558

theorem cyclic_quadrilateral_EFYZ 
  (A B C I D E F Y Z : Type) 
  [Geometry A B C I D E F Y Z] 
  (h1 : incenter I A B C) 
  (h2 : intersection D A I B C) 
  (h3 : intersection E B I C A) 
  (h4 : intersection F C I A B) 
  (h5 : intersection Y C I D E) 
  (h6 : intersection Z B I D F) 
  (h7 : ∠A B C = 120) : cyclic_quad E F Y Z :=
sorry

end cyclic_quadrilateral_EFYZ_l524_524558


namespace sum_of_valid_c_values_l524_524701

theorem sum_of_valid_c_values : 
  (∑ c in {c : ℤ | c ≤ 30 ∧ ∃ k : ℤ, k^2 = 64 + 4*c}, c) = 29 :=
by {
  sorry 
}

end sum_of_valid_c_values_l524_524701


namespace assign_numbers_friends_l524_524123

open GraphTheory

-- Define a graph with vertices and edges categorized as red (friends) or blue (not friends)
variable {α : Type*} [Inhabited α]

structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (edge_color : V → V → Prop) -- true for red edges (friendship), false for blue edges

noncomputable def is_friend_divisibility
  (G : Graph α)
  (N : ℕ)
  (assign_number : α → ℕ) : Prop :=
∀ (u v : α), (G.edge_color u v = true ↔ ∃ k1 k2 : ℕ, (assign_number u) * (assign_number v) = k1 * N ∧ (assign_number u) * (assign_number v) = (assign_number v) * k2)

-- Now define the main theorem statement
theorem assign_numbers_friends
  (G : Graph α)
  : ∃ (N : ℕ) (assign_number : α → ℕ), is_friend_divisibility G N assign_number :=
sorry

end assign_numbers_friends_l524_524123


namespace total_pencils_l524_524599

theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) (total : ℕ) 
  (h_pencils_per_child : pencils_per_child = 4)
  (h_num_children : num_children = 8)
  (h_total : total = pencils_per_child * num_children) :
    total = 32 :=
by
  rw [h_pencils_per_child, h_num_children] at h_total
  rw h_total
  norm_num
  sorry

end total_pencils_l524_524599


namespace sum_S_l524_524007

noncomputable def S (k : ℕ) : ℚ := 
  if k > 0 then 1/2 * (abs ((1:ℚ)/k - (1:ℚ)/(k + 1)))
  else 0

theorem sum_S : (∑ k in Finset.range 2012, S k) = 2011 / 4024 := by
  sorry

end sum_S_l524_524007


namespace prism_vertex_difference_l524_524514

theorem prism_vertex_difference 
  (V : Finset ℕ) -- Vertices set containing {1, 2, ..., 100}
  (E : Finset (ℕ × ℕ)) -- Edges set for the prism structure
  (hV_card : V.card = 100) -- |V| = 100
  (hE_card : E.card = 150) -- |E| for 50-gon prism where base and lateral surfaces count
  (h_labels : ∀ v ∈ V, 1 ≤ v ∧ v ≤ 100) -- Vertices are labeled 1 to 100
  (h_prism : ∀ {u v}, (u, v) ∈ E → (u = v ± 1 ∨ u = v ± 50 ∨ u = v ± 49)) -- Prism constraints
: ∃ u v, (u, v) ∈ E 
         ∧ abs (u - v) ≤ 48 := 
sorry

end prism_vertex_difference_l524_524514


namespace matchstick_triangle_sides_l524_524938

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end matchstick_triangle_sides_l524_524938


namespace rectangle_area_l524_524525

variable (w l : ℕ)
variable (A : ℕ)
variable (H1 : l = 5 * w)
variable (H2 : 2 * l + 2 * w = 180)

theorem rectangle_area : A = 1125 :=
by
  sorry

end rectangle_area_l524_524525


namespace jellybeans_initial_amount_l524_524211

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l524_524211


namespace line_circle_no_intersection_l524_524362

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524362


namespace probability_even_sum_rows_columns_l524_524074

open Probability

def has_even_sum_rows_and_columns (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i, (∑ j, grid i j) % 2 = 0 ∧ (∑ j, grid j i) % 2 = 0

def all_1_to_9 : Fin 3 → Fin 3 → Finset ℕ :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem probability_even_sum_rows_columns :
  let grids := { grid : Matrix (Fin 3) (Fin 3) ℕ // ∀ i j, grid i j ∈ all_1_to_9 ∧ ∀ n ∈ all_1_to_9, ∃ (i j : Fin 3), grid i j = n }
  ∃! p : ℚ, (∀ g ∈ grids, has_even_sum_rows_and_columns g → Probs g = p) ∧ p = 3 / 32 :=
sorry

end probability_even_sum_rows_columns_l524_524074


namespace proof_of_P_value_l524_524839

theorem proof_of_P_value (x y : ℝ) (h1 : x^2 + y^2 - x * y = 2) (h2 : x^4 + y^4 + (x * y)^2 = 8) : 
  x^8 + y^8 + x^2014 * y^2014 = 48 := 
begin
  sorry
end

end proof_of_P_value_l524_524839


namespace text_message_cost_eq_l524_524973

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end text_message_cost_eq_l524_524973


namespace expected_socks_until_pair_l524_524986

open ProbabilityTheory -- Open relevant modules

-- Define the random variable and conditions given
def n_pairs_of_socks (n : ℕ) : Prop :=
    True  -- Abstracting the condition that n pairs of socks hanging in random order

def no_identical_pairs : Prop :=
    True  -- Abstracting the condition that there are no identical pairs

def scientist_takes_out_socks (n : ℕ) : Prop :=
    n_pairs_of_socks n ∧ no_identical_pairs  -- Combining conditions

-- Expected number of socks until a pair is found 
noncomputable def E_xi (n : ℕ) : ℝ :=
    sorry -- Placeholder definition for the expected value

-- The main theorem: Expected number of socks taken until a pair is found
theorem expected_socks_until_pair (n : ℕ) (hn : n > 0) :
    scientist_takes_out_socks n → E_xi n ≈ Real.sqrt (π * n) :=
by
  intros _ 
  sorry

end expected_socks_until_pair_l524_524986


namespace exists_sequence_to_one_friend_l524_524085

structure Graph (V : Type) :=
  (E : V → V → Prop)
  (symmetric : ∀ {x y : V}, E x y → E y x)

variable {V : Type} [Fintype V]

def initial_graph (G : Graph V) : V → ℕ
| v => if v ∈ finset.range 1010 then 1009 else 1010

def operation_possible (G : Graph V) (A B C : V) : Prop :=
  G.E A B ∧ G.E A C ∧ ¬ G.E B C

def perform_operation (G : Graph V) (A B C : V) : Graph V :=
{ E := λ x y, if x = B ∧ y = C ∨ x = C ∧ y = B then true
              else if (x = A ∧ y = B ∨ x = B ∧ y = A) ∨ (x = A ∧ y = C ∨ x = C ∧ y = A)
              then false
              else G.E x y,
  symmetric := sorry }

theorem exists_sequence_to_one_friend :
  ∃ (ops : list (Σ A B C : V, unit)), -- list of operations on triples of vertices
    ∀ v : V, ((ops.foldl (λ G op, perform_operation G op.1 op.2 op.3) (initial_graph G)) v ≤ 1) :=
sorry

end exists_sequence_to_one_friend_l524_524085


namespace line_circle_no_intersection_l524_524402

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524402


namespace johnny_fishes_l524_524036

theorem johnny_fishes
  (total_fishes : ℕ)
  (sony_ratio : ℕ)
  (total_is_40 : total_fishes = 40)
  (sony_is_4x_johnny : sony_ratio = 4)
  : ∃ (johnny_fishes : ℕ), johnny_fishes + sony_ratio * johnny_fishes = total_fishes ∧ johnny_fishes = 8 :=
by
  sorry

end johnny_fishes_l524_524036


namespace C_over_D_equals_17_l524_524675

noncomputable def seriesC : ℝ :=
  ∑' n in (finset.range 100).filter (λ n, n % 4 ≠ 0 ∧ n % 2 = 0), 1 / (n^2)

noncomputable def seriesD : ℝ :=
  ∑' n in (finset.range 100).filter (λ n, n % 4 = 0 ∧ n ≠ 0), 1 / (n^2)

theorem C_over_D_equals_17 : seriesC / seriesD = 17 := 
by sorry

end C_over_D_equals_17_l524_524675


namespace inequality_on_circle_l524_524858

theorem inequality_on_circle (x y c : ℝ) (h : x^2 + (y - 1)^2 = 1) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 → x + y + c ≥ 0) ↔ c ≥ real.sqrt 2 - 1 :=
by sorry

end inequality_on_circle_l524_524858


namespace line_circle_no_intersection_l524_524340

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524340


namespace no_intersection_l524_524365

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524365


namespace impossible_planting_trees_l524_524922

theorem impossible_planting_trees : 
  ∀ (trees : Fin 50 → Fin 25), 
  (∀ (i j : Fin 50), (trees i = trees j) → i ≠ j → (i + 2 = j + 1 ∨ i + 4 = j + 3) → False :=
begin
  sorry
end

end impossible_planting_trees_l524_524922


namespace smallest_of_five_consecutive_even_sum_500_l524_524885

theorem smallest_of_five_consecutive_even_sum_500 : 
  ∃ (n : Int), (n - 4, n - 2, n, n + 2, n + 4).1 = 96 ∧ 
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4) = 500) :=
by
  sorry

end smallest_of_five_consecutive_even_sum_500_l524_524885


namespace shaded_fraction_of_octagon_area_l524_524498

def regular_octagon (O : Point) (A B C D E F G H : Point) : Prop := 
  -- Defining a regular octagon centered at O with points A, B, C, D, E, F, G, H
  sorry

def midpoint (X D C : Point) : Prop := 
  -- Defining X as the midpoint of segment CD
  sorry

def isosceles_triangle (O D C : Point) : Prop := 
  -- Defining isosceles triangles formed by the center O and vertices D, C
  sorry

theorem shaded_fraction_of_octagon_area
  (O A B C D E F G H X : Point)
  (h1 : regular_octagon O A B C D E F G H)
  (h2 : midpoint X D C)
  (h3 : isosceles_triangle O D C)
  : (area (Triangle O D E) + area (Triangle O E F) + area (Triangle O F G) + area (Triangle O D C) / 2) / area (Octagon O A B C D E F G H) = 7 / 16 :=
by
  sorry

end shaded_fraction_of_octagon_area_l524_524498


namespace tangent_line_intercept_l524_524662

theorem tangent_line_intercept :
  ∃ (m : ℚ) (b : ℚ), m > 0 ∧ b = 740 / 171 ∧
    ∀ (x y : ℚ), ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x - 15)^2 + (y - 8)^2 = 100) →
                 (y = m * x + b) ↔ False := 
sorry

end tangent_line_intercept_l524_524662


namespace common_ratio_geometric_sequence_l524_524057

theorem common_ratio_geometric_sequence (a : ℝ) :
  let log2_3 := Real.log 3 / Real.log 2 in
  let log4_3 := Real.log 3 / (2 * Real.log 2) in
  let log8_3 := Real.log 3 / (3 * Real.log 2) in
  (q = (a + log4_3) / (a + log2_3)) ∧ (q = (a + log8_3) / (a + log4_3)) →
  q = 1 / 3 :=
by
  intros
  sorry

end common_ratio_geometric_sequence_l524_524057


namespace line_bc_equation_l524_524738

noncomputable def find_line_bc (a b c : ℝ) (A : ℝ × ℝ) (h1 : a + b = 0) (h2 : 2 * a - 3 * b + c = 0) (h3 : A = (1, 2)) : Prop :=
  let a := 1
  let b := 2
  ∃ k l m : ℝ, k * x + l * y + m = 0 ∧ k = 2 ∧ l = 3 ∧ m = 7

theorem line_bc_equation : find_line_bc x y z (1, 2) :=
begin
  sorry
end

end line_bc_equation_l524_524738


namespace annual_simple_interest_rate_is_8_percent_l524_524128

-- Define the problem conditions
def principal : ℝ := 150
def total_amount_paid : ℝ := 162

-- Define the simple interest rate calculation
def interest (P A : ℝ) : ℝ := A - P
def interest_rate (P I : ℝ) : ℝ := (I / P) * 100

-- Define the theorem that needs to be proved
theorem annual_simple_interest_rate_is_8_percent : 
  interest_rate principal (interest principal total_amount_paid) = 8 :=
by
  sorry

end annual_simple_interest_rate_is_8_percent_l524_524128


namespace calculate_taxi_fare_l524_524814

def peak_hour := true
def initial_fee := if peak_hour then 3.50 else 2.25
def rate_0_2 := 0.15 * 5 * 2
def rate_2_5 := 0.20 * 3 * 1.6
def wait_time := 8
def wait_charge := 0.10 * wait_time
def base_distance_cost := rate_0_2 + rate_2_5
def peak_surcharge := 0.10 * base_distance_cost
def total_distance_cost := base_distance_cost + peak_surcharge
def total_charge := initial_fee + total_distance_cost + wait_charge

theorem calculate_taxi_fare :
  (total_charge).round(2) = 7.01 :=
by
  -- This is where the proof would go, but it's skipped with "sorry".
  sorry

end calculate_taxi_fare_l524_524814


namespace leibo_orange_price_l524_524483

variable (x y m : ℝ)

theorem leibo_orange_price :
  (3 * x + 2 * y = 78) ∧ (2 * x + 3 * y = 72) ∧ (18 * m + 12 * (100 - m) ≤ 1440) → (x = 18) ∧ (y = 12) ∧ (m ≤ 40) :=
by
  intros h
  sorry

end leibo_orange_price_l524_524483


namespace PQRS_product_l524_524712

theorem PQRS_product :
  let P := (Real.sqrt 2012 + Real.sqrt 2013)
  let Q := (-Real.sqrt 2012 - Real.sqrt 2013)
  let R := (Real.sqrt 2012 - Real.sqrt 2013)
  let S := (Real.sqrt 2013 - Real.sqrt 2012)
  P * Q * R * S = 1 :=
by
  sorry

end PQRS_product_l524_524712


namespace geometric_sequence_sum_example_l524_524008

theorem geometric_sequence_sum_example (a : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : q > 1) 
  (h3 : a 2 * a 6 = 64) 
  (h4 : a 3 + a 5 = 20) 
  : 
  let S (n : ℕ) := a 0 * (1 - q^(n+1)) / (1 - q) in
  S 5 = 63 := 
by 
  -- the proof goes here
  sorry

end geometric_sequence_sum_example_l524_524008


namespace line_circle_no_intersection_l524_524381

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524381


namespace average_of_all_results_equal_l524_524778

variables {x y : ℝ} (S1 S2 : ℝ)

-- Definitions based on conditions
def avg_first_x_results (x y : ℝ) (S1 : ℝ) := S1 / x = y
def avg_other_y_results (x y : ℝ) (S2 : ℝ) := S2 / y = x

theorem average_of_all_results_equal (hx : avg_first_x_results x y S1)
    (hy : avg_other_y_results x y S2) :
    (S1 + S2) / (x + y) = (2 * x * y) / (x + y) :=
begin
    sorry
end

end average_of_all_results_equal_l524_524778


namespace weight_of_each_piece_l524_524016

theorem weight_of_each_piece 
  (x : ℝ)
  (h : 2 * x + 0.08 = 0.75) : 
  x = 0.335 :=
by
  sorry

end weight_of_each_piece_l524_524016


namespace log_increasing_l524_524267

theorem log_increasing (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → log a (2 - x₁) < log a (2 - x₂)) ↔ (0 < a ∧ a < 1) := by
  sorry

end log_increasing_l524_524267


namespace Andy_earnings_l524_524644

/-- Andy's total earnings during an 8-hour shift. --/
theorem Andy_earnings (hours : ℕ) (hourly_wage : ℕ) (num_racquets : ℕ) (pay_per_racquet : ℕ)
  (num_grommets : ℕ) (pay_per_grommet : ℕ) (num_stencils : ℕ) (pay_per_stencil : ℕ)
  (h_shift : hours = 8) (h_hourly : hourly_wage = 9) (h_racquets : num_racquets = 7)
  (h_pay_racquets : pay_per_racquet = 15) (h_grommets : num_grommets = 2)
  (h_pay_grommets : pay_per_grommet = 10) (h_stencils : num_stencils = 5)
  (h_pay_stencils : pay_per_stencil = 1) :
  (hours * hourly_wage + num_racquets * pay_per_racquet + num_grommets * pay_per_grommet +
  num_stencils * pay_per_stencil) = 202 :=
by
  sorry

end Andy_earnings_l524_524644


namespace term_with_largest_coefficient_in_binomial_expansion_l524_524229

theorem term_with_largest_coefficient_in_binomial_expansion (x y : ℝ) :
  ∃ k, ∀ m : ℕ, (binomial 9 k > binomial 9 m → k = 4 ∨ k = 5) ∧ binomial 9 4 = 126 :=
by
  sorry

end term_with_largest_coefficient_in_binomial_expansion_l524_524229


namespace max_f_value_l524_524071

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log10 x + 9 / Real.log10 x

theorem max_f_value : ∀ x, 0 < x ∧ x < 1 → f x ≤ -5 := 
λ x h, sorry

end max_f_value_l524_524071


namespace line_through_C_circumcircle_OBC_l524_524794

/-- Problem setup: definition of points O, B, and C --/
def pointO : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 2)
def pointC : ℝ × ℝ := (4, 0)

/-- Statement for the line equation problem --/
theorem line_through_C (l : ℝ → ℝ) :
  (∀ x, (l x = x - 4 ∨ l x = - (1 / 3) * x + (4 / 3))) ↔
  (∃ k, (l x = k * (x - 4)) ∧ 
    ((|k * 0 - 4 * k| / real.sqrt (k ^ 2 + 1)) = 
    (|(2 * k - 2 - 4 * k)| / real.sqrt (k ^ 2 + 1)))) :=
sorry

/-- Statement for the circumcircle equation problem --/
theorem circumcircle_OBC :
  ∃ D E F, (D = -4 ∧ E = 0 ∧ F = 0) ∧ 
  (∀ x y, 
    (x^2 + y^2 + D*x + E*y + F = 0) ↔ 
    ((x = 0 ∧ y = 0) ∨ 
     (x = 2 ∧ y = 2) ∨ 
     (x = 4 ∧ y = 0))) :=
sorry

end line_through_C_circumcircle_OBC_l524_524794


namespace three_digit_integers_count_l524_524772

def digits := {1, 3, 4, 4, 7, 7, 7}

def count_three_digit_integers (s : set ℕ) : ℕ :=
  let d := s.to_finset.val in
  let freq := d.map (λ x, (x, d.count x)) in
  let possible_triples := {x | x ∈ s ∧ 1 ≤ x ∧ x ≤ 9} in
  let valid_combinations := possible_triples.filter (λ x, ∀ (y : ℕ), y ∈ x → d.count y ≤ s.to_mul (x.count y)) in
  valid_combinations.card

theorem three_digit_integers_count : count_three_digit_integers digits = 43 :=
  by sorry

end three_digit_integers_count_l524_524772


namespace find_AB_length_l524_524988

structure Triangle (α : Type) where
  A B C : α

structure Point (α : Type) where
  x y : α

noncomputable def length {α : Type} [InnerProductSpace ℝ α] (p1 p2 : α) : ℝ :=
  dist p1 p2

variables {α : Type} [EuclideanSpace ℝ α]

-- Given conditions
variables 
  (A B C D M : Point α)
  (hAM_angle_bisector : AngleBisector (Triangle.mk A B C) A M)
  (hBM : length B M = 2)
  (hCM : length C M = 3)
  (hD_on_circumcircle : OnCircumcircle (Triangle.mk A B C) D)
  (hMD : length M D = 2)

-- Prove length AB = sqrt(10)
theorem find_AB_length : length A B = Real.sqrt 10 :=
sorry

end find_AB_length_l524_524988


namespace line_circle_no_intersect_l524_524395

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524395


namespace min_sum_sequence_n_l524_524805

theorem min_sum_sequence_n (S : ℕ → ℤ) (h : ∀ n, S n = n * n - 48 * n) : 
  ∃ n, n = 24 ∧ ∀ m, S n ≤ S m :=
by
  sorry

end min_sum_sequence_n_l524_524805


namespace hyperbola_eccentricity_tangent_to_circle_l524_524283

theorem hyperbola_eccentricity_tangent_to_circle :
  ∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b),
  let C := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 4} in
  let E := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} in
  (∀ x y : ℝ, (x^2 + (y - 4)^2 = 4) →
  (y = (b / a) * x ∨ y = -(b / a) * x) →
    (∃ k : ℝ, k = 2) ∧ sqrt (1 + (b^2 / a^2)) = 2 := sorry

end hyperbola_eccentricity_tangent_to_circle_l524_524283


namespace sum_even_coeffs_eq_two_pow_n_l524_524465

theorem sum_even_coeffs_eq_two_pow_n
  (n : ℕ)
  (b : ℕ → ℤ)
  (h : ∀ x : ℂ, (1 - x + x^2)^n = ∑ i in Finset.range (2*n + 1), (b i) * (x^i)) :
  ∑ k in Finset.range (n + 1), b (2 * k) = 2^n := 
sorry

end sum_even_coeffs_eq_two_pow_n_l524_524465


namespace rational_root_even_coeff_l524_524019

open Int

theorem rational_root_even_coeff (a b c : ℤ) (p q : ℤ) (h_ne_zero : q ≠ 0) (coprime_pq : gcd p q = 1)
    (h_root : a * p^2 + b * p * q + c * q^2 = 0)
    (h_odd_a : ¬ even a) (h_odd_b : ¬ even b) (h_odd_c : ¬ even c) : false :=
sorry

end rational_root_even_coeff_l524_524019


namespace find_a_intervals_of_monotonicity_l524_524755

noncomputable def f (x a b : ℝ) : ℝ :=
  (1 / 3) * x^3 - a * x^2 + b

theorem find_a (b : ℝ) :
  (∃ (f' : ℝ → ℝ), f' = λ x => x^2 - 2 * a * x) →
  (∀ f'(2) = 0) →
  a = 1 :=
by
  sorry

theorem intervals_of_monotonicity (b : ℝ) :
  let a := 1
  f (x : ℝ) a b = (1 / 3) * x^3 - x^2 + b →
  (∀ x, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x, x < 0 ∨ x > 2 → f' x > 0) :=
by
  sorry

end find_a_intervals_of_monotonicity_l524_524755


namespace runs_scored_today_is_69_l524_524505

-- Definitions for conditions
def current_batting_average : ℕ := 51
def matches_played_before : ℕ := 5
def new_batting_average : ℕ := 54
def total_matches_after : ℕ := 6

-- The total number of runs before today's match
def total_runs_before : ℕ := current_batting_average * matches_played_before

-- The total number of runs needed after today's match
def total_runs_needed : ℕ := new_batting_average * total_matches_after

-- The number of runs Rahul scored in today's match
def runs_scored_today : ℕ := total_runs_needed - total_runs_before

-- Proof statement
theorem runs_scored_today_is_69 : runs_scored_today = 69 := by
  unfold runs_scored_today total_runs_needed total_runs_before
  rw [mul_comm 51 5, mul_comm 54 6]
  calc
    54 * 6 - 51 * 5 = 324 - 255 := by rfl
    ...                = 69 := by rfl

end runs_scored_today_is_69_l524_524505


namespace line_circle_no_intersection_l524_524334

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524334


namespace find_alpha_and_polar_equation_l524_524714

def point (α β : ℝ) := (α, β)

def line (α : ℝ) (t : ℝ) := (2 + t * Real.cos α, 1 + t * Real.sin α)

def distance (P A : ℝ × ℝ) := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

theorem find_alpha_and_polar_equation :
  ∃ (α : ℝ),
  (
    let P := point 2 1 in
    let l := line α in
    let A := line α (-(1 / Real.sin α)) in
    let B := line α (-(2 / Real.cos α)) in
    distance P A * distance P B = 4 ∧ α = 3 * Real.pi / 4
  )
∧
  (
    let α := 3 * Real.pi / 4 in
    ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 3
  ) := 
begin
  sorry
end

end find_alpha_and_polar_equation_l524_524714


namespace line_circle_no_intersect_l524_524394

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524394


namespace gcd_35_x_eq_7_in_range_80_90_l524_524523

theorem gcd_35_x_eq_7_in_range_80_90 {n : ℕ} (h₁ : Nat.gcd 35 n = 7) (h₂ : 80 < n) (h₃ : n < 90) : n = 84 :=
by
  sorry

end gcd_35_x_eq_7_in_range_80_90_l524_524523


namespace last_two_nonzero_digits_70_factorial_l524_524528

theorem last_two_nonzero_digits_70_factorial :
  let N := Nat.factorial 70 / 10^16 in 
  N % 100 = 44 :=
by
  -- We skip the detailed proof steps with sorry
  sorry

end last_two_nonzero_digits_70_factorial_l524_524528


namespace correct_sum_of_satisfying_values_l524_524285

def g (x : Nat) : Nat :=
  match x with
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def f (x : Nat) : Nat :=
  match x with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def satisfies_condition (x : Nat) : Bool :=
  f (g x) > g (f x)

def sum_of_satisfying_values : Nat :=
  List.sum (List.filter satisfies_condition [0, 1, 2])

theorem correct_sum_of_satisfying_values : sum_of_satisfying_values = 2 :=
  sorry

end correct_sum_of_satisfying_values_l524_524285


namespace cost_of_items_l524_524010

theorem cost_of_items (x : ℝ) (cost_caramel_apple cost_ice_cream_cone : ℝ) :
  3 * cost_caramel_apple + 4 * cost_ice_cream_cone = 2 ∧
  cost_caramel_apple = cost_ice_cream_cone + 0.25 →
  cost_ice_cream_cone = 0.17857 ∧ cost_caramel_apple = 0.42857 :=
sorry

end cost_of_items_l524_524010


namespace equivalent_mean_l524_524598

def calculate_mean (lst : List ℕ) : ℕ :=
(lst.sum) / (lst.length)

theorem equivalent_mean :
  ∃ x : ℤ, calculate_mean ([3, 117, 915, 138, 1917, 2114] ++ [x]) = 12 → x = -5120 :=
by
  sorry

end equivalent_mean_l524_524598


namespace bacteria_fill_sixteenth_of_dish_in_26_days_l524_524600

theorem bacteria_fill_sixteenth_of_dish_in_26_days
  (days_to_fill_dish : ℕ)
  (doubling_rate : ℕ → ℕ)
  (H1 : days_to_fill_dish = 30)
  (H2 : ∀ n, doubling_rate (n + 1) = 2 * doubling_rate n) :
  doubling_rate 26 = doubling_rate 30 / 2^4 :=
sorry

end bacteria_fill_sixteenth_of_dish_in_26_days_l524_524600


namespace heather_walk_distance_l524_524322

theorem heather_walk_distance :
  ∀ (d1 d2 d_total d_back : ℝ),
    d1 = 0.3333333333333333 →
    d2 = 0.3333333333333333 →
    d_total = 0.75 →
    d_back = d_total - (d1 + d2) →
    d_back = 0.08333333333333337 :=
by 
  intros d1 d2 d_total d_back h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end heather_walk_distance_l524_524322


namespace mean_of_all_students_l524_524846

-- Definitions based on the given conditions
def mean_first_section : Real := 92
def mean_second_section : Real := 78
def ratio_students : Real := 2 / 3

-- Calculated Answer
theorem mean_of_all_students (f s : Real) (hf : mean_first_section * f = 184 / 3 * s)
                                      (hs : mean_second_section * s = 78 * s)
                                      (hr : f / s = ratio_students) :
  (mean_first_section * f + mean_second_section * s) / (f + s) = 83.6 := 
by
  -- Begin proof here
  -- Placeholder for the actual proof steps
  sorry

end mean_of_all_students_l524_524846


namespace sum_of_divisors_24_l524_524955

theorem sum_of_divisors_24 : (∑ d in (finset.filter (λ n, 24 % n = 0) (finset.range 25)), d) = 60 :=
by
  sorry

end sum_of_divisors_24_l524_524955


namespace sum_of_possible_values_of_s_r_l524_524836

noncomputable def r : ℤ → ℤ
| -2 := -1
| -1 := 0
| 0 := 3
| 1 := 5
| _ := 0

noncomputable def s (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem sum_of_possible_values_of_s_r :
  {r (-2), r (-1), r 0, r 1} ∩ {0, 1, 2, 3} = {0, 1, 3} →
  s 0 + s 1 + s 3 = 21 :=
by
  intro h
  rw [s, s, s]
  sorry

end sum_of_possible_values_of_s_r_l524_524836


namespace problem_g1_eq_l524_524469

theorem problem_g1_eq (f g : ℝ → ℝ)
  (a b c d : ℝ) 
  (ha : 1 < a) (hb : a < b) (hc : b < c) (hd : c < d)
  (hf : f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (hg : g x = x * (x - 1/p) * (x - 1/q) * (x - 1/r))
  (hroots : f (1) = 1 + a + b + c + d) :
  g(1) = (1 + a + b + c + d) / d :=
sorry

end problem_g1_eq_l524_524469


namespace tangent_intersects_x_axis_l524_524084

theorem tangent_intersects_x_axis (x0 x1 : ℝ) (hx : ∀ x : ℝ, x1 = x0 - 1) :
  x1 - x0 = -1 :=
by
  sorry

end tangent_intersects_x_axis_l524_524084


namespace avg_salary_increases_by_150_l524_524887

def avg_salary_increase
  (emp_avg_salary : ℕ) (num_employees : ℕ) (mgr_salary : ℕ) : ℕ :=
  let total_salary_employees := emp_avg_salary * num_employees
  let total_salary_with_mgr := total_salary_employees + mgr_salary
  let new_avg_salary := total_salary_with_mgr / (num_employees + 1)
  new_avg_salary - emp_avg_salary

theorem avg_salary_increases_by_150 :
  avg_salary_increase 1800 15 4200 = 150 :=
by
  sorry

end avg_salary_increases_by_150_l524_524887


namespace statement_bug_travel_direction_l524_524013

/-
  Theorem statement: On a plane with a grid formed by regular hexagons of side length 1,
  if a bug traveled from node A to node B along the shortest path of 100 units,
  then the bug traveled exactly 50 units in one direction.
-/
theorem bug_travel_direction (side_length : ℝ) (total_distance : ℝ) 
  (hexagonal_grid : Π (x y : ℝ), Prop) (A B : ℝ × ℝ) 
  (shortest_path : ℝ) :
  side_length = 1 ∧ shortest_path = 100 →
  ∃ (directional_travel : ℝ), directional_travel = 50 :=
by
  sorry

end statement_bug_travel_direction_l524_524013


namespace sum_F_coordinates_is_56_over_11_l524_524798

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def one_third_point (B C : ℝ × ℝ) : ℝ × ℝ :=
  ((2 * B.1 + C.1) / 3, (2 * B.2 + C.2) / 3)

def line_eq (P Q : ℝ × ℝ) : ℝ → ℝ :=
  let slope := (P.2 - Q.2) / (P.1 - Q.1)
  in λ x, P.2 + slope * (x - P.1)

def intersection (f g : ℝ → ℝ) : ℝ × ℝ :=
  let x := (f 0 - g 0) / ((g 1 - g 0) - (f 1 - f 0))
  (x, f x)

def F_coordinates : ℝ × ℝ :=
  intersection (line_eq (0,6) (one_third_point (0,0) (8,0))) 
               (line_eq (midpoint (0,6) (0,0)) (8,0))

def sum_of_coordinates (P : ℝ × ℝ) : ℝ :=
  P.1 + P.2

theorem sum_F_coordinates_is_56_over_11 : 
  sum_of_coordinates F_coordinates = 56 / 11 :=
sorry

end sum_F_coordinates_is_56_over_11_l524_524798


namespace percentage_of_primes_less_than_20_divisible_by_5_l524_524572

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 := {n : ℕ | n < 20 ∧ is_prime n}

def primes_less_than_20_divisible_by_5 := {n ∈ primes_less_than_20 | 5 ∣ n}

theorem percentage_of_primes_less_than_20_divisible_by_5 : 
  (primes_less_than_20_divisible_by_5.to_finset.card : ℝ) / (primes_less_than_20.to_finset.card : ℝ) * 100 = 12.5 :=
begin
  -- Proving this statement directly would involve showing the calculations explicitly.
  -- However, we just set up the framework here.
  sorry
end

end percentage_of_primes_less_than_20_divisible_by_5_l524_524572


namespace king_inevitably_checked_l524_524854

/--
Given a 20x20 chessboard, 10 rooks, and one king:

- The king starts at (1, 1) and moves to (20, 20) along the diagonal.
- Moves are made alternately: first, the king moves, then one of the rooks can move.
- The king is not in check at the start.

Prove that the king will inevitably come under check.
-/
theorem king_inevitably_checked :
  ∀ (king_moves rooks_moves : ℕ),
  (king_moves = 19 ∧ rooks_moves ≥ 20) → 
  ∃ m, m ∈ List.range king_moves → ∃ r, r ∈ List.range rooks_moves → 
  is_check (move_king m) (move_rook r) :=
by
  sorry

end king_inevitably_checked_l524_524854


namespace An_Bn_contains_one_element_sequence_b_n_geometric_l524_524192

open Finset

variables {n : ℕ} (n_ge_2 : n ≥ 2)

def Sn := univ.permutations

def An (n : ℕ) : Finset (List (Fin n)) :=
univ.filter (fun (a : List (Fin n)) => ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ n → (a.nth i).get_or_else 0 - i ≤ (a.nth j).get_or_else 0 - j)

def Bn (n : ℕ) : Finset (List (Fin n)) :=
univ.filter (fun (a : List (Fin n)) => ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ n → (a.nth i).get_or_else 0 + i ≤ (a.nth j).get_or_else 0 + j)

def count_An_cap_Bn (n : ℕ) : ℕ :=
((An n) ∩ (Bn n)).card

def geometric_b_n (n : ℕ) : Prop :=
∀ (m : ℕ), m ≥ 3 → 2 * (count_An_cap_Bn (m - 1)) = count_An_cap_Bn m

theorem An_Bn_contains_one_element (n : ℕ) (hn : n ≥ 2) : count_An_cap_Bn n = 1 := sorry

theorem sequence_b_n_geometric (n : ℕ) (hn : n ≥ 3) : geometric_b_n n := sorry

end An_Bn_contains_one_element_sequence_b_n_geometric_l524_524192


namespace final_result_of_fractional_subtraction_l524_524510

theorem final_result_of_fractional_subtraction : 
  let n := 2015 in
  let subseq := List.range n in
  subseq.foldl (λ acc k, acc * (1 - 1 / (k + 2))) n = 1 :=
by 
  let n := 2015;
  let subseq := List.range n;
  show subseq.foldl (λ acc k, acc * (1 - 1 / (k + 2))) n = 1;
  sorry

end final_result_of_fractional_subtraction_l524_524510


namespace sum_of_radii_ge_inscribed_radius_l524_524287

variable {ABC : Type} [triangle ABC]

-- Define the radius of the inscribed circle of the triangle ABC
variables (r : ℝ) (r_inscribed : ∀ (ABC : Type), radius_of_inscribed_circle ABC = r)

-- Define the radii of the circles touching the pairs of sides and the inscribed circle
variables (r_a r_b r_c : ℝ)
variables (radius_ra : ∀ (ABC : Type), radius_of_circle_touching_sides ABC inscribed_radius = r_a)
variables (radius_rb : ∀ (ABC : Type), radius_of_circle_touching_sides ABC inscribed_radius = r_b)
variables (radius_rc : ∀ (ABC : Type), radius_of_circle_touching_sides ABC inscribed_radius = r_c)

theorem sum_of_radii_ge_inscribed_radius 
  (r_inscribed : radius_of_inscribed_circle ABC = r)
  (radius_ra : radius_of_circle_touching_sides ABC inscribed_radius = r_a)
  (radius_rb : radius_of_circle_touching_sides ABC inscribed_radius = r_b)
  (radius_rc : radius_of_circle_touching_sides ABC inscribed_radius = r_c)
: 
r_a + r_b + r_c ≥ r ∧ ∀ (ABC : triangle), is_equilateral ABC →
  (is_equilateral ABC → r_a + r_b + r_c = r) := by sorry

end sum_of_radii_ge_inscribed_radius_l524_524287


namespace divide_equilateral_triangle_into_25_parts_l524_524493

theorem divide_equilateral_triangle_into_25_parts :
  ∀ (A B C : Point), equilateral_triangle A B C →
  ∃ S : Set Point, (∃ T : Set (Set Point), (∀ s ∈ T, ∃ a b c : Point, triangle a b c ∧ equilateral a b c) ∧ card(T) = 25 ∧ ∀ t ∈ T, t ⊆ S)
  →
  ∃ F : Fin 5 → Set (Set Point), (∀ i : Fin 5, card (F i) = 5) := sorry

end divide_equilateral_triangle_into_25_parts_l524_524493


namespace sum_real_roots_eq_minus3_l524_524003

def myCeiling (x : ℝ) : ℤ := ⌈x⌉

theorem sum_real_roots_eq_minus3 :
  (∑ x in {x : ℝ | myCeiling (3 * x + 1) = 2 * x - 1 / 2}, x) = -3 :=
by
  sorry

end sum_real_roots_eq_minus3_l524_524003


namespace fuel_spending_reduction_l524_524222

-- Define the variables and the conditions
variable (x c : ℝ) -- x for efficiency and c for cost
variable (newEfficiency oldEfficiency newCost oldCost : ℝ)

-- Define the conditions
def conditions := (oldEfficiency = x) ∧ (newEfficiency = 1.75 * oldEfficiency)
                 ∧ (oldCost = c) ∧ (newCost = 1.3 * oldCost)

-- Define the expected reduction in cost
def expectedReduction : ℝ := 25.7142857142857 -- approximately 25 5/7 %

-- Define the assertion that Elmer will reduce his fuel spending by the expected reduction percentage
theorem fuel_spending_reduction : conditions x c oldEfficiency newEfficiency oldCost newCost →
  ((oldCost - (newCost / newEfficiency) * oldEfficiency) / oldCost) * 100 = expectedReduction :=
by
  sorry

end fuel_spending_reduction_l524_524222


namespace interest_rate_l524_524964

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r/100)^t - P

theorem interest_rate (P t : ℝ) (diff : ℝ) (r : ℝ) (h : P = 1000) (t_eq : t = 4) 
  (diff_eq : diff = 64.10) : 
  compound_interest P r t - simple_interest P r t = diff → r = 10 :=
by
  sorry

end interest_rate_l524_524964


namespace line_circle_no_intersection_l524_524361

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524361


namespace triangular_square_l524_524817

def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_square (m n : ℕ) (h1 : 1 ≤ m) (h2 : 1 ≤ n) (h3 : 2 * triangular m = triangular n) :
  ∃ k : ℕ, triangular (2 * m - n) = k * k :=
by
  sorry

end triangular_square_l524_524817


namespace no_valid_permutation_0_to_9_l524_524990

open List

def is_adjacent (x y : ℕ) : Prop :=
  abs (x - y) = 3 ∨ abs (x - y) = 4 ∨ abs (x - y) = 5

theorem no_valid_permutation_0_to_9 :
  ¬ ∃ (l : List ℕ), l ~ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
  ∧ ∀ (i : ℕ), i < l.length → is_adjacent (l.get i) (l.get ((i + 1) % l.length)) :=
sorry

end no_valid_permutation_0_to_9_l524_524990


namespace joann_lollipops_third_day_l524_524462

noncomputable def joann_lollipops (a : ℕ) (H : 5 * a + 80 = 150) : ℕ := a + 16

theorem joann_lollipops_third_day : ∃ a : ℕ, 5 * a + 80 = 150 ∧ joann_lollipops a (by sorry) = 30 :=
begin
  use 14,
  split,
  { norm_num }, -- This proves 5 * 14 + 80 = 150
  { unfold joann_lollipops,
    norm_num } -- This proves (joann_lollipops 14 (by sorry)) = 30
end

end joann_lollipops_third_day_l524_524462


namespace angle_ACB_is_60_degrees_l524_524940

/-- Given a triangle ABC and points D and E on the line AB such that AD = AC and BE = BC, 
    with the arrangement of points D - A - B - E,
    Let the circumscribed circles of triangles DBC and EAC meet again at the point X ≠ C,
    and the circumscribed circles of triangles DEC and ABC meet again at the point Y ≠ C.
    Given that DY + EY = 2XY, we want to prove that ∠ACB = 60° -/
theorem angle_ACB_is_60_degrees (A B C D E X Y : Point) 
  (h1 : Triangle ABC)
  (h2 : D ∈ Line AB)
  (h3 : E ∈ Line AB)
  (h4 : AD = AC)
  (h5 : BE = BC)
  (h6 : Between D A B)
  (h7 : Between A B E)
  (h8 : X ∈ Circle (CircumscribedCircle B D C))
  (h9 : X ∈ Circle (CircumscribedCircle E A C))
  (h10 : X ≠ C)
  (h11 : Y ∈ Circle (CircumscribedCircle D E C))
  (h12 : Y ∈ Circle (CircumscribedCircle A B C))
  (h13 : Y ≠ C)
  (h14 : DY + EY = 2 * XY) : angle A C B = 60 :=
sorry

end angle_ACB_is_60_degrees_l524_524940


namespace pollywogs_disappear_in_44_days_l524_524221

theorem pollywogs_disappear_in_44_days :
  ∀ (initial_count rate_mature rate_caught first_period_days : ℕ),
  initial_count = 2400 →
  rate_mature = 50 →
  rate_caught = 10 →
  first_period_days = 20 →
  (initial_count - first_period_days * (rate_mature + rate_caught)) / rate_mature + first_period_days = 44 := 
by
  intros initial_count rate_mature rate_caught first_period_days h1 h2 h3 h4
  sorry

end pollywogs_disappear_in_44_days_l524_524221


namespace range_of_f_one_l524_524756

theorem range_of_f_one {m : ℝ} (h : ∀ x y : ℝ, -2 ≤ x ∧ x ≤ y → f 4 x m + 5 ≤ f 4 y m + 5) :
  Set.range (λ x : ℝ, 4 * 1 ^ 2 - m * 1 + 5) = Set.Ici 25 := by
  sorry

def f (a : ℝ) (x : ℝ) (b : ℝ) : ℝ :=
  a * x ^ 2 - b * x + 5

end range_of_f_one_l524_524756


namespace pow_1986_mod_7_l524_524950

theorem pow_1986_mod_7 : (5 ^ 1986) % 7 = 1 := by
  sorry

end pow_1986_mod_7_l524_524950


namespace building_max_floors_l524_524129

noncomputable def max_floors (num_elevators floors_per_elevator : ℕ) (connected_pairs : ℕ → Prop) : ℕ :=
  let max_pairs_per_elevator := floors_per_elevator * (floors_per_elevator - 1) / 2
  let max_pairs := num_elevators * max_pairs_per_elevator
  if ∃ n : ℕ, connected_pairs n ∧ (n * (n - 1) / 2 ≤ max_pairs) then 14 else 0

theorem building_max_floors :
  max_floors 7 6 (λ n, ∀ (i j : ℕ), i ≠ j → ∃ e : ℕ, e < 7 ∧ i < 6 ∧ j < 6) ≤ 14 :=
sorry

end building_max_floors_l524_524129


namespace V_shape_max_area_l524_524277

noncomputable def max_area (k : ℝ) : ℝ := 
  k^2 / (8 * Real.sqrt 2 + 2 * Real.pi - 4)

theorem V_shape_max_area (x y: ℝ) (k : ℝ) (h_pos : 0 < x) (h_perimeter : 2 * y + (Real.sqrt 2 + (Real.pi / 2)) * x = k) 
  : ∃ (max_area : ℝ) (y_0 x_0 : ℝ), max_area = k^2 / (8 * Real.sqrt 2 + 2 * Real.pi - 4) ∧ 
    y_0 / x_0 = (Real.sqrt 2 - 1) / 2 :=
begin
  sorry
end

end V_shape_max_area_l524_524277


namespace geometric_ratio_l524_524912

theorem geometric_ratio (a₁ q : ℝ) (h₀ : a₁ ≠ 0) (h₁ : a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) : q = -2 ∨ q = 1 :=
by
  sorry

end geometric_ratio_l524_524912


namespace volume_of_union_of_regular_tetrahedrons_l524_524098

-- Definitions for the conditions set up
def unit_cube_vertices : Set (ℝ × ℝ × ℝ) := {
  (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
}

-- Given two specific regular tetrahedrons A and B given by vertices of the cube
def tet_A : Set (ℝ × ℝ × ℝ) := { (0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0) }
def tet_B : Set (ℝ × ℝ × ℝ) := { (0, 0, 1), (1, 1, 1), (0, 1, 0), (1, 0, 0) }

-- Statement to prove the volume of the union of these tetrahedrons
theorem volume_of_union_of_regular_tetrahedrons (A B : Set (ℝ × ℝ × ℝ)) (hA : A = tet_A) (hB : B = tet_B) :
  volume (A ∪ B) = 1 / 2 :=
  sorry

end volume_of_union_of_regular_tetrahedrons_l524_524098


namespace even_numbers_relatively_prime_to_18_l524_524324

def even_nat (n : ℕ) : Prop := n % 2 = 0
def greater_than_10 (n : ℕ) : Prop := n > 10
def less_than_100 (n : ℕ) : Prop := n < 100
def relatively_prime_to_18 (n : ℕ) : Prop := GCD n 18 = 1

theorem even_numbers_relatively_prime_to_18 : 
  {n : ℕ | even_nat n ∧ greater_than_10 n ∧ less_than_100 n ∧ relatively_prime_to_18 n}.toFinset.card = 29 :=
by
  sorry

end even_numbers_relatively_prime_to_18_l524_524324


namespace find_coordinates_l524_524262

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def satisfiesCondition (A B P : Point) : Prop :=
  2 * (P.x - A.x) = (B.x - P.x) ∧ 2 * (P.y - A.y) = (B.y - P.y)

theorem find_coordinates (P : Point) (h : satisfiesCondition A B P) : 
  P = ⟨6, -9⟩ :=
  sorry

end find_coordinates_l524_524262


namespace OB_leq_quarter_OA1_l524_524728

variables {ℝ : Type} [LinearOrderedField ℝ]

-- Define the four lines and their intersections
variables (m1 m2 m3 m4 : ℝ) -- need to be vectors in 2D space, here simplified as real numbers representing directions
variables (O A1 A2 A3 A4 B : ℝ)

-- Assume the conditions described in the problem statement
-- 1. Four lines intersect at O
axiom lines_intersect_at_O : 
  (A1 = m1 * O ∨ A1 = m2 * O) ∧ 
  (A2 = m2 * O ∨ A2 = m3 * O) ∧ 
  (A3 = m3 * O ∨ A3 = m4 * O) ∧ 
  (A4 = m4 * O ∨ A4 = m1 * O)

-- 2. Given point A1 on line m1
axiom A1_on_m1 : 
  A1 = m1 * O

-- 3. A2 is on m2 and parallel to m4 from A1
axiom A2_on_m2_parallel_to_m4_from_A1 : 
  ∃ (p : ℝ), A2 = m2 * p

-- 4. A3 is on m3 and parallel to m1 from A2
axiom A3_on_m3_parallel_to_m1_from_A2 : 
  ∃ (p : ℝ), A3 = m3 * p

-- 5. A4 is on m4 and parallel to m2 from A3
axiom A4_on_m4_parallel_to_m2_from_A3 : 
  ∃ (p : ℝ), A4 = m4 * p

-- 6. B is on m1 and parallel to m3 from A4
axiom B_on_m1_parallel_to_m3_from_A4 : 
  ∃ (p : ℝ), B = m1 * p

-- To prove the inequality
theorem OB_leq_quarter_OA1 : 
  OB ≤ OA1 / 4 := sorry

end OB_leq_quarter_OA1_l524_524728


namespace Jims_apples_fits_into_average_l524_524415

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l524_524415


namespace y1_lt_y2_l524_524500

-- Define the linear function
def linear_function (x : ℝ) : ℝ := 8 * x - 1

-- Define the points P1 and P2
def P1 : ℝ × ℝ := (3, linear_function 3)
def P2 : ℝ × ℝ := (4, linear_function 4)

-- The proof statement that y1 is less than y2
theorem y1_lt_y2 : P1.2 < P2.2 :=
by
  unfold P1 P2 linear_function
  simp
  -- y1 is 23 and y2 is 31, so 23 < 31
  exact Nat.lt.base 22 -- or any method to show 23 < 31

end y1_lt_y2_l524_524500


namespace centers_of_squares_on_parallelogram_form_square_l524_524039

theorem centers_of_squares_on_parallelogram_form_square
  (A B C D P Q R S : Type)
  [AddCommGroup P] [AddCommGroup Q] [AddCommGroup R] [AddCommGroup S]
  [VectorSpace ℝ P] [VectorSpace ℝ Q] [VectorSpace ℝ R] [VectorSpace ℝ S]
  (parallelogram : Parallelogram ABCD)
  (square1 : Square P A B)
  (square2 : Square Q B C)
  (square3 : Square R C D)
  (square4 : Square S D A)
  (h1 : Center P P A B)
  (h2 : Center Q Q B C)
  (h3 : Center R R C D)
  (h4 : Center S S D A) :
  IsSquare P Q R S :=
sorry

end centers_of_squares_on_parallelogram_form_square_l524_524039


namespace prob_class1_two_mcq_from_A_expected_value_best_of_five_l524_524615

-- Part 1
theorem prob_class1_two_mcq_from_A :
  let P_B1 := (5.choose 2) / (8.choose 2)
  let P_B2 := (5.choose 1 * 3.choose 1) / (8.choose 2)
  let P_B3 := (3.choose 2) / (8.choose 2)
  let P_A_given_B1 := 6 / 9
  let P_A_given_B2 := 5 / 9
  let P_A_given_B3 := 4 / 9
  let P_A := P_B1 * P_A_given_B1 + P_B2 * P_A_given_B2 + P_B3 * P_A_given_B3
  let P_B1_given_A := (P_B1 * P_A_given_B1) / P_A
  P_B1_given_A = 20 / 49 :=
by
  sorry

-- Part 2
theorem expected_value_best_of_five :
  let P_X3 := (3/5 * 2/5 * 2/5) + (2/5 * 2/5 * 2/5)
  let P_X4 := 3/5 * 3/5 * 3/5 * 2/5 + 2/5 * 3/5 * 3/5 * 2/5 + 2/5 * 2/5 * 3/5 * 3/5 + 2/5 * 3/5 * 2/5 * 2/5 + 3/5 * 3/5 * 3/5 * 3/5 + 3/5 * 2/5 * 3/5 * 3/5
  let P_X5 := 1 - P_X3 - P_X4
  let E_X := 3 * P_X3 + 4 * P_X4 + 5 * P_X5
  E_X = 537 / 125 :=
by
  sorry

end prob_class1_two_mcq_from_A_expected_value_best_of_five_l524_524615


namespace sages_can_guarantee_more_than_500_correct_l524_524258

noncomputable def canGuaranteeMoreThan500CorrectGuesses 
  (hats : Finset ℕ) (hidden_hat : ℕ) (sages : ℕ → Finset ℕ) : Prop :=
  ∀ (hidden_hat ≤ 1001) (∀ hat ∈ hats, hat ≠ hidden_hat ∧ 1 ≤ hat ∧ hat ≤ 1001), 
  ∃ strategy : (ℕ → ℕ) → list ℕ, strategy hidden_hat = list.range 1000 

theorem sages_can_guarantee_more_than_500_correct : 
  ∀ (hats : Finset ℕ) (hidden_hat : ℕ) (sages : ℕ → Finset ℕ), 
  canGuaranteeMoreThan500CorrectGuesses hats hidden_hat sages :=
sorry

end sages_can_guarantee_more_than_500_correct_l524_524258


namespace sum_of_possible_values_of_z_l524_524821

theorem sum_of_possible_values_of_z (x y z : ℂ) 
  (h₁ : z^2 + 5 * x = 10 * z)
  (h₂ : y^2 + 5 * z = 10 * y)
  (h₃ : x^2 + 5 * y = 10 * x) :
  z = 0 ∨ z = 9 / 5 := by
  sorry

end sum_of_possible_values_of_z_l524_524821


namespace combinations_to_50_cents_l524_524770

def coin_combinations (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes + 25 * quarters

theorem combinations_to_50_cents : 
  {n : ℕ // ∃ (pennies nickels dimes quarters : ℕ),
    coin_combinations pennies nickels dimes quarters = 50 ∧
    n = 47 } :=
begin
  use 47,
  sorry 
end

end combinations_to_50_cents_l524_524770


namespace root_product_expression_value_l524_524474

theorem root_product_expression_value :
  ∀ (a b c d : ℝ), 
    (a + b = -2000 ∧ ab = 1) ∧ (c + d = 2008 ∧ cd = 1) →
    (a + c) * (b + c) * (a - d) * (b - d) = 32064 :=
by
  intros a b c d
  assume h
  sorry

end root_product_expression_value_l524_524474


namespace farmer_field_arrangement_l524_524138

/-- 
There are four types of crops: Corn (C), Wheat (W), Soybeans (S), and Potatoes (P).
Given the constraints:
1. Corn (C) cannot be next to Wheat (W) or Soybeans (S).
2. Potatoes (P) cannot be next to Soybeans (S).
We need to prove that there are 28 different ways to arrange these crops in three consecutive sections.
-/
def crop := ℕ -- There are 4 types of crops we'll encode as 0, 1, 2, and 3 for C, W, S, P

def is_valid_arrangement (a b : crop) : Prop :=
  match (a, b) with
  | (0, 1) => false -- C next to W
  | (0, 2) => false -- C next to S
  | (3, 2) => false -- P next to S
  | (1, 0) => false -- W next to C (since it's symmetrical with C next to W)
  | (2, 0) => false -- S next to C (since it's symmetrical with C next to S)
  | (2, 3) => false -- S next to P (since it's symmetrical with P next to S)
  | _     => true -- all other combinations are valid

theorem farmer_field_arrangement : ∃ (arr : list crop), 
   list.length arr = 3 ∧
   ∑ (a b : crop) (H : a ≠ b ∧ is_valid_arrangement a b), 1 = 28 :=
sorry

end farmer_field_arrangement_l524_524138


namespace triangle_ADE_to_BCED_area_ratio_l524_524455

theorem triangle_ADE_to_BCED_area_ratio (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
    (AB AC BC: ℝ)
    (AD AE: ℝ)
    (hAB : AB = 30)
    (hAC : AC = 48)
    (hBC : BC = 45)
    (hAD : AD = 18)
    (hAE : AE = 36) 
    (hDAB : D ∈ line_segment A B)
    (hEAC : E ∈ line_segment A C)
    (hAD_D : dist A D = 18)
    (hAE_E : dist A E = 36) : 
    (area (triangle A D E)) / area (quadrilateral B C E D) = 9 / 16 :=
by
  sorry

end triangle_ADE_to_BCED_area_ratio_l524_524455


namespace sqrt_fraction_3_l524_524250

theorem sqrt_fraction_3 (x : ℚ) (hx : x ≥ 0 ∧ x + 2 ≠ 0) : 
  (sqrt (7 * x) / sqrt (4 * (x + 2)) = 3) → x = (-72) / 29 := by
sorry

end sqrt_fraction_3_l524_524250


namespace incorrect_calculation_C_l524_524591

theorem incorrect_calculation_C :
  (∀ (a b : ℝ), real.sqrt a * real.sqrt b = real.sqrt (a * b)) ∧
  (∀ (a b : ℝ), real.sqrt a / real.sqrt b = real.sqrt (a / b)) ∧
  (∀ (a b : ℝ), (-real.sqrt a)^2 = a) →
  ¬ (real.sqrt 2 + real.sqrt 3 = real.sqrt 5) :=
by
  intro h,
  cases h with h1 h23,
  cases h23 with h2 h3,
  sorry

end incorrect_calculation_C_l524_524591


namespace shortest_side_of_similar_triangle_proof_l524_524153

noncomputable def shortest_side_of_similar_triangle (a c c₂ : ℕ) (h₁ : a = 15) (h₂ : c = 39) (h₃ : c₂ = 117) : ℕ :=
  let sc : ℕ := c₂ / c in
  let a₂ : ℕ := sc * a in
  a₂

theorem shortest_side_of_similar_triangle_proof :
  shortest_side_of_similar_triangle 15 39 117 15 39 117 = 45 :=
by
  sorry

end shortest_side_of_similar_triangle_proof_l524_524153


namespace solve_for_b_l524_524896

theorem solve_for_b (b : ℝ) : 
  let slope1 := -(3 / 4 : ℝ)
  let slope2 := -(b / 6 : ℝ)
  slope1 * slope2 = -1 → b = -8 :=
by
  intro h
  sorry

end solve_for_b_l524_524896


namespace degree_even_l524_524473

noncomputable def P (X : ℝ) : ℝ[X] := 
  sorry -- To be replaced by the actual non-zero polynomial

theorem degree_even (P : ℝ[X]) (h1 : P ≠ 0) (h2 : P ∣ P.comp (λ X, X^2 + X + 1)) : even (degree P) := 
by
  sorry

end degree_even_l524_524473


namespace line_circle_no_intersection_l524_524403

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524403


namespace coefficient_x4_in_expansion_sum_l524_524889

theorem coefficient_x4_in_expansion_sum :
  (Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4 = 55) :=
by
  sorry

end coefficient_x4_in_expansion_sum_l524_524889


namespace robot_min_steps_l524_524833

theorem robot_min_steps {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : ∃ n, n = a + b - Nat.gcd a b :=
by
  sorry

end robot_min_steps_l524_524833


namespace bacteria_growth_time_l524_524888

theorem bacteria_growth_time (initial_population : ℕ) (target_population : ℕ) (growth_factor : ℕ) (time_per_tripling : ℕ) :
  initial_population = 300 → target_population = 72900 → growth_factor = 3 → time_per_tripling = 3 → 
  ∃ t : ℕ, t * time_per_tripling = 15 ∧ initial_population * growth_factor ^ (t / 3) = target_population :=
by
  intros h_init h_target h_factor h_time
  have ht : t = 15, sorry
  existsi t
  split
  . exact ht
  . have h_exp : 15 / 3 = 5, sorry
    rw [ht, h_exp]
    calc
      initial_population * growth_factor ^ 5
      = 300 * 243 : by rw [h_init, h_factor]
      ... = 72900 : by rw h_target

end bacteria_growth_time_l524_524888


namespace irrationals_among_examples_l524_524968

theorem irrationals_among_examples :
  ¬ ∃ (r : ℚ), r = π ∧
  (∃ (a b : ℚ), a * a = 4) ∧
  (∃ (r : ℚ), r = 0) ∧
  (∃ (r : ℚ), r = -22 / 7) := 
sorry

end irrationals_among_examples_l524_524968


namespace conjugate_in_first_quadrant_l524_524733

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := (2 + 4 * i) / (1 + i)^2
noncomputable def z_conjugate : ℂ := complex.conj z

theorem conjugate_in_first_quadrant {i z z_conjugate : ℂ} (h1 : i = complex.I) (h2 : z = (2 + 4 * i) / (1 + i)^2) 
    (h3 : z_conjugate = complex.conj z) :
    (0 < z_conjugate.re) ∧ (0 < z_conjugate.im) :=
by
  sorry

end conjugate_in_first_quadrant_l524_524733


namespace number_of_subsets_with_desired_mean_l524_524326

theorem number_of_subsets_with_desired_mean :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let desired_mean := 7
  let removed_elements_sum := 8
  let valid_pairs := [(1, 7), (2, 6), (3, 5)]
  S.card = 12 → 
  valid_pairs.length = 3 → 
  S.sum = 78 →  
  ( ∀ x ∈ S, x > 0 ) →
  valid_pairs = [(1, 7), (2, 6), (3, 5)] →
  valid_pairs.length = 3 := 
by 
  sorry

end number_of_subsets_with_desired_mean_l524_524326


namespace max_value_of_M_l524_524251

def J (k : ℕ) := 10^(k + 3) + 256

def M (k : ℕ) := Nat.factors (J k) |>.count 2

theorem max_value_of_M (k : ℕ) (hk : k > 0) :
  M k = 8 := by
  sorry

end max_value_of_M_l524_524251


namespace largest_constant_inequality_l524_524247

theorem largest_constant_inequality (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) + 
  sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e)) + 
  sqrt (e / (a + b + c + d)) > 2 := 
sory

end largest_constant_inequality_l524_524247


namespace remaining_area_correct_l524_524143

-- Define the side lengths of the large rectangle
def large_rectangle_length1 (x : ℝ) := 2 * x + 5
def large_rectangle_length2 (x : ℝ) := x + 8

-- Define the side lengths of the rectangular hole
def hole_length1 (x : ℝ) := 3 * x - 2
def hole_length2 (x : ℝ) := x + 1

-- Define the area of the large rectangle
def large_rectangle_area (x : ℝ) := (large_rectangle_length1 x) * (large_rectangle_length2 x)

-- Define the area of the hole
def hole_area (x : ℝ) := (hole_length1 x) * (hole_length2 x)

-- Prove the remaining area after accounting for the hole
theorem remaining_area_correct (x : ℝ) : 
  large_rectangle_area x - hole_area x = -x^2 + 20 * x + 42 := 
  by 
    sorry

end remaining_area_correct_l524_524143


namespace problem_proof_l524_524749

noncomputable section

def ellipse_standard_eq (a b : ℝ) (h₁ : a > b > 0) (ecc : a / b = (√2) / 2) (d : (b^2 + (a/2)^2)^0.5 = √2) :
  Prop :=
  (a = √2 ∧ b = 1) ∧ (∀ x y : ℝ, (x^2 / 2 + y^2 = 1 ↔ (x / √2)^2 + y^2 = 1))

noncomputable def exists_line_m (a b : ℝ) (h₁ : a > b > 0) (ecc : a / b = (√2) / 2) (d : (b^2 + (a/2)^2)^0.5 = √2) :
  Prop :=
  (∃ k : ℝ, k ≠ 0 ∧ (k = 42 ∨ k = -42) ∧ (∀ x y : ℝ, (y = k * x + 1/k ∨ y = -k * x - 1/k)))

theorem problem_proof :
  ∃ (a b : ℝ), 
    a > b > 0 ∧
    a / b = (√2 / 2) ∧
    (b^2 + (a / 2)^2)^ 0.5 = √2 ∧
    ellipse_standard_eq a b ∧ exists_line_m a b :=
sorry

end problem_proof_l524_524749


namespace cats_awake_l524_524546

theorem cats_awake (total_cats asleep_cats cats_awake : ℕ) (h1 : total_cats = 98) (h2 : asleep_cats = 92) (h3 : cats_awake = total_cats - asleep_cats) : cats_awake = 6 :=
by
  -- Definitions and conditions
  subst h1
  subst h2
  subst h3
  -- The statement we need to prove
  sorry

end cats_awake_l524_524546


namespace directed_distance_props_l524_524060

theorem directed_distance_props (A B C x1 y1 x2 y2 : ℝ) (h : A^2 + B^2 ≠ 0) :
  let d1 := (A * x1 + B * y1 + C) / real.sqrt (A^2 + B^2)
  let d2 := (A * x2 + B * y2 + C) / real.sqrt (A^2 + B^2)
  (d1 * d2 > 0 → ( ∃ a b : ℝ, A * b = -B * a ∧
           ((y2 - y1) * a = (x2 - x1) * b ∨
           (A * (y2 - y1) = B * (x2 - x1)))) ∧
  (d1 * d2 < 0 → ∃ a b : ℝ, A * b ≠ -B * a) :=
by
  sorry

end directed_distance_props_l524_524060


namespace sin_x_eq_x_has_unique_root_in_interval_l524_524529

theorem sin_x_eq_x_has_unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ x = Real.sin x :=
sorry

end sin_x_eq_x_has_unique_root_in_interval_l524_524529


namespace num_digits_ending_in_same_digit_l524_524442

theorem num_digits_ending_in_same_digit (a : ℕ) (k : ℕ) (p : ℕ -> ℕ) (h1 : a = ∏ i in range k, p i) :
  ∃ b : ℕ, b = 2^k ∧ (∀ n : ℕ, n < a → (b^2 ≡ b [MOD a] ∧ b*(b-1) ≡ 0 [MOD a])) := 
sorry

end num_digits_ending_in_same_digit_l524_524442


namespace correct_propositions_l524_524744

variables {α β : Type} [Plane α] [Plane β]
          {m n : Type} [Line m] [Line n]

/-- Proposition 1: If m ⊥ α, α ⊥ β, and m || n, then n || β. -/
def prop1 (h1 : m ⊥ α) (h2 : α ⊥ β) (h3 : m ∥ n) : n ∥ β :=
sorry

/-- Proposition 2: If m ⊥ α, m || n, α || β, then n ⊥ β. -/
def prop2 (h1 : m ⊥ α) (h2 : m ∥ n) (h3 : α ∥ β) : n ⊥ β :=
sorry

/-- Proposition 3: If α ⊥ β, α ∩ β = m, and n ⊊ β, n ⊥ m, then n ⊥ α. -/
def prop3 (h1 : α ⊥ β) (h2 : α ∩ β = m) (h3 : n ⊊ β) (h4 : n ⊥ m) : n ⊥ α :=
sorry

/-- Proposition 4: If α ∩ β = m, n || m, and n is not on α or β, 
    then n || α and n || β. -/
def prop4 (h1 : α ∩ β = m) (h2 : n ∥ m) (h3 : ¬(n ⊂ α ∨ n ⊂ β)) : n ∥ α ∧ n ∥ β :=
sorry

/-- Theorem: There are exactly 3 correct propositions given the above conditions. -/
theorem correct_propositions : ∀ (h1 : m ⊥ α) (h2 : m ∥ n) 
  (h3 : α ⊥ β) (h4 : α ∥ β) 
  (h5 : α ∩ β = m) (h6 : ¬(n ⊂ α ∨ n ⊂ β)) 
  (h7 : n ⊊ β) (h8 : n ⊥ m),
  (¬prop1 h1 h3 h2) ∧ prop2 h1 h2 h4 ∧ prop3 h3 h5 h7 h8 ∧ prop4 h5 h2 h6 :=
sorry

end correct_propositions_l524_524744


namespace no_real_intersections_l524_524345

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524345


namespace relationship_among_abc_l524_524291

-- Define the variables as given in the problem
def a := Real.sqrt 2
def b := Real.log 3 / Real.log Real.pi
def c := Real.log (1 / Real.exp 1) / Real.log 2

-- State the theorem to be proven
theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- Proof steps omitted with "sorry", focus on statement and definitions
  sorry

end relationship_among_abc_l524_524291


namespace find_f_of_2011_l524_524265

-- Define the function f
def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 7

-- The main statement we need to prove
theorem find_f_of_2011 (a b c : ℝ) (h : f (-2011) a b c = -17) : f 2011 a b c = 31 :=
by
  sorry

end find_f_of_2011_l524_524265


namespace option_B_correct_option_C_correct_option_D_correct_l524_524802

variables {a b : ℝ} (h : 0 < a ∧ a < b)

def v1 : ℝ := (a + b) / 2
def v2 : ℝ := (2 * a * b) / (a + b)

theorem option_B_correct : a < v2 h ∧ v2 h < Real.sqrt (a * b) :=
sorry

theorem option_C_correct : Real.sqrt (a * b) < v1 h ∧ v1 h < Real.sqrt ((a^2 + b^2) / 2) :=
sorry

theorem option_D_correct : v1 h > v2 h :=
sorry

end option_B_correct_option_C_correct_option_D_correct_l524_524802


namespace length_of_train_is_correct_l524_524977

-- Conditions
def speed_kmph : ℝ := 50
def time_sec : ℝ := 9

-- Conversion from km/hr to m/s
def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Question: the length of the train
def length_of_train : ℝ := speed_mps * time_sec

-- Proof problem statement
theorem length_of_train_is_correct :
  length_of_train = 125 :=
by
  -- Here you can assume the necessary conversion and calculations have been done
  -- This is just to represent the statement in Lean
  sorry

end length_of_train_is_correct_l524_524977


namespace range_of_a_l524_524305

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem range_of_a :
  (∃ (a : ℝ), (a ≤ -2 ∨ a ≥ 0) ∧ (∃ (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4), f x ≤ a^2 + 2 * a)) :=
by sorry

end range_of_a_l524_524305


namespace sandra_socks_l524_524032

variables (x y z : ℕ)

theorem sandra_socks :
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≤ 6 →
  y ≤ 6 →
  z ≤ 6 →
  x = 11 :=
by
  sorry

end sandra_socks_l524_524032


namespace correct_props_l524_524757

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + (1/2:ℝ) * x^2
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1 + x
noncomputable def x0 : ℝ := sorry   -- x0 is an extremum of f, computed but not explicitly defined here

theorem correct_props (hx0 : x0 > 0) 
  (h_extremum : f' x0 = 0) : 
  (0 < x0 ∧ x0 < 1 / Real.exp 1) ∧ 
  x0 ≤ 1 / Real.exp 1 ∧ 
  f(x0) + x0 < 0 :=
by
  sorry

end correct_props_l524_524757


namespace count_valid_four_digit_numbers_l524_524325

def four_digit_numbers_count : Nat :=
  let valid_a := [3, 4]
  let valid_d := [0, 5]
  let valid_pairs_bc := [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
  valid_a.length * valid_d.length * valid_pairs_bc.length

theorem count_valid_four_digit_numbers :
  four_digit_numbers_count = 24 :=
by
  -- To be proved
  sorry

end count_valid_four_digit_numbers_l524_524325


namespace sufficient_but_not_necessary_for_or_not_necessary_for_and_l524_524987

-- Define propositions p and q
variable (p q : Prop)

-- State the theorem
theorem sufficient_but_not_necessary_for_or (hpq : p ∧ q) : p ∨ q :=
by {
  -- From the condition "if p and q are true", we need to show "p or q is true"
  exact hpq.left,
  
  -- (We could use hpq.right as well, but exact hpq.left suffices for this statement.)
}

theorem not_necessary_for_and (hp : p ∨ q) : ¬ (p ∧ q) :=
by {
  -- From the condition "if p or q is true", we need to show "not (p and q is true)"
  intro h,
  exact false.of_not_not_and hp h
}

end sufficient_but_not_necessary_for_or_not_necessary_for_and_l524_524987


namespace interest_rate_eq_5p685_l524_524607

theorem interest_rate_eq_5p685 
  (P : ℝ) (A : ℝ) (x : ℝ) (h1 : P = 10000) (h2 : A = 17800)
  (h3 : ∀ t : ℝ, 0 ≤ t ≤ 5 → compound_interest P (x + 0.5) t = P * (1 + (x + 0.5) / 100) ^ t)
  (h4 : ∀ t : ℝ, 5 < t ≤ 10 → compound_interest P x t = P * (1 + x / 100) ^ (t - 5) * (1 + (x + 0.5) / 100) ^ 5) :
  x = 5.685 := 
sorry

end interest_rate_eq_5p685_l524_524607


namespace permutations_satisfying_conditions_l524_524939

open Function

def count_valid_permutations : ℕ := 180

theorem permutations_satisfying_conditions :
  ∃ l : List (List ℕ), (l.length = 6!) ∧
    (∀ p : List ℕ, p ∈ l → nodup p ∧ (1 < 2) ∧ (3 < 4)) ∧
      l.filter (λ p, p.indexOf 1 < p.indexOf 2 ∧ p.indexOf 3 < p.indexOf 4).length = count_valid_permutations :=
sorry

end permutations_satisfying_conditions_l524_524939


namespace roots_diff_eq_4_l524_524478

theorem roots_diff_eq_4 {r s : ℝ} (h₁ : r ≠ s) (h₂ : r > s) (h₃ : r^2 - 10 * r + 21 = 0) (h₄ : s^2 - 10 * s + 21 = 0) : r - s = 4 := 
by
  sorry

end roots_diff_eq_4_l524_524478


namespace miles_mike_ride_l524_524488

theorem miles_mike_ride
  (cost_per_mile : ℝ) (start_fee : ℝ) (bridge_toll : ℝ)
  (annie_miles : ℝ) (annie_total_cost : ℝ)
  (mike_total_cost : ℝ) (M : ℝ)
  (h1 : cost_per_mile = 0.25)
  (h2 : start_fee = 2.50)
  (h3 : bridge_toll = 5.00)
  (h4 : annie_miles = 26)
  (h5 : annie_total_cost = start_fee + bridge_toll + cost_per_mile * annie_miles)
  (h6 : mike_total_cost = start_fee + cost_per_mile * M)
  (h7 : mike_total_cost = annie_total_cost) :
  M = 36 := 
sorry

end miles_mike_ride_l524_524488


namespace line_circle_no_intersection_l524_524358

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524358


namespace scout_hourly_base_pay_l524_524871

theorem scout_hourly_base_pay :
  ∃ x: ℝ, (4 * x + 5 * 5 + 5 * x + 8 * 5 = 155) ∧ x = 10 :=
by
  use 10
  split
  {
    calc
      4 * 10 + 5 * 5 + 5 * 10 + 8 * 5
      = 40 + 25 + 50 + 40 : by norm_num
      ... = 155 : by norm_num,
  }
  {
    rfl,
  }

end scout_hourly_base_pay_l524_524871


namespace average_price_per_bottle_is_correct_l524_524984

-- Conditions
def large_bottles : ℕ := 1375
def price_per_large_bottle : ℝ := 1.75
def small_bottles : ℕ := 690
def price_per_small_bottle : ℝ := 1.35

-- Calculation of total cost for large bottles
def total_cost_large_bottles : ℝ := large_bottles * price_per_large_bottle

-- Calculation of total cost for small bottles
def total_cost_small_bottles : ℝ := small_bottles * price_per_small_bottle

-- Combine total cost for all bottles
def combined_total_cost : ℝ := total_cost_large_bottles + total_cost_small_bottles

-- Calculation of total number of bottles
def total_bottles : ℕ := large_bottles + small_bottles

-- Calculation of average price paid per bottle
def approximate_average_price_per_bottle : ℝ := combined_total_cost / total_bottles

-- Given above that approximate average price per bottle should match the calculated value.
theorem average_price_per_bottle_is_correct :
  approximate_average_price_per_bottle ≈ 1.62 :=
by sorry

end average_price_per_bottle_is_correct_l524_524984


namespace universal_proposition_example_l524_524110

theorem universal_proposition_example :
  (∀ n : ℕ, n % 2 = 0 → ∃ k : ℕ, n = 2 * k) :=
sorry

end universal_proposition_example_l524_524110


namespace candy_distribution_count_l524_524491

theorem candy_distribution_count :
  (∑ r in finset.range (8), ∑ b in finset.range (8 - r),
  nat.choose 9 (r + 1) * nat.choose (9 - (r + 1)) (b + 1)) = 504 := 
by {
  sorry
}

end candy_distribution_count_l524_524491


namespace arithmetic_sequence_sum_l524_524797

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def S9_is_90 (S : ℕ → ℝ) := S 9 = 90

-- The proof goal
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : S9_is_90 S) :
  a 3 + a 5 + a 7 = 30 :=
by
  sorry

end arithmetic_sequence_sum_l524_524797


namespace solve_for_multiplier_l524_524112

namespace SashaSoup
  
-- Variables representing the amounts of salt
variables (x y : ℝ)

-- Condition provided: amount of salt added today
def initial_salt := 2 * x
def additional_salt_today := 0.5 * y

-- Given relationship
axiom salt_relationship : x = 0.5 * y

-- The multiplier k to achieve the required amount of salt
def required_multiplier : ℝ := 1.5

-- Lean theorem statement
theorem solve_for_multiplier :
  (2 * x) * required_multiplier = x + y :=
by
  -- Mathematical proof goes here but since asked to skip proof we use sorry
  sorry

end SashaSoup

end solve_for_multiplier_l524_524112


namespace inverse_proportional_properties_l524_524764

-- Conditions
def inverse_proportional_function (k : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ a : ℝ, P = (a, k / a)

def another_inverse_proportional_function (P : ℝ × ℝ) : Prop :=
  ∃ a : ℝ, P = (a, 1 / a)

-- Correct Answer
theorem inverse_proportional_properties (k : ℝ)
  (P : ℝ × ℝ) (hP : inverse_proportional_function k P) :
  let C : ℝ × ℝ := (P.1, 0)
  let A : ℝ × ℝ := (P.1, 1 / P.1)
  let B : ℝ × ℝ := (P.1 / k, P.2)
  let D : ℝ × ℝ := (0, P.2)
in
  (let S_OCA := 1 / 2 * P.1 * (1 / P.1)
   let OBD := 1 / 2 * (P.1 / k) * (k / P.1)
   in S_OCA = OBD) ∧
  (let S_OAPB := k - ((1 / 2 * P.1 * (1 / P.1)) + (1 / 2 * (P.1 / k) * (k / P.1)))
   in S_OAPB = k - 1) ∧
  (k = 2 → inverse_proportional_function 2 B) :=
by
  sorry

end inverse_proportional_properties_l524_524764


namespace symmetric_graph_min_value_l524_524428

theorem symmetric_graph_min_value (a b : ℝ)
  (h_symm : ∀ x, (x + 1)^2 * (x + 1)^2 = (x^2 - 4) * (x^2 + a * x + b)) :
  a = 4 ∧ b = 0 ∧ ∃ x, f(x) = -16 :=
by
  have h0 : f(0) = f(-2) := h_symm 0
  have h1 : f(-4) = f(2) := h_symm (-4)
  sorry

end symmetric_graph_min_value_l524_524428


namespace negation_of_universal_prop_l524_524073

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 - 5 * x + 3 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 5 * x + 3 > 0) :=
by sorry

end negation_of_universal_prop_l524_524073


namespace line_circle_no_intersection_l524_524384

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524384


namespace bananas_to_oranges_l524_524042

theorem bananas_to_oranges :
  (3 / 4 : ℝ) * 16 = 12 →
  (2 / 3 : ℝ) * 9 = 6 :=
by
  intro h
  sorry

end bananas_to_oranges_l524_524042


namespace car_y_average_speed_l524_524660

theorem car_y_average_speed
  (car_x_speed : ℕ := 35)
  (car_y_travel_start : ℕ := 72)
  (car_x_additional_travel : ℕ := 105)
  (car_y_speed : ℕ) :
  let car_x_speed_in_mph := car_x_speed,
      car_y_speed_in_mph := car_y_speed,
      car_y_travel_start_hours := (car_y_travel_start / 60 : ℝ),
      car_x_distance_before_y := car_x_speed_in_mph * car_y_travel_start_hours,
      car_x_total_distance := car_x_distance_before_y + car_x_additional_travel,
      car_y_travel_time_hours := car_x_additional_travel / car_x_speed_in_mph,
      car_y_speed_computed := car_x_additional_travel / car_y_travel_time_hours
  in car_y_speed_computed = 35 :=
begin
  sorry
end

end car_y_average_speed_l524_524660


namespace time_for_second_half_l524_524113

-- Definitions based on the conditions provided
def initial_speed (v : ℝ) := v > 0
def total_distance := 40
def half_distance := total_distance / 2
def first_half_time (v : ℝ) := half_distance / v
def second_half_time (v : ℝ) := half_distance / (v / 2)

-- Given the second half takes 11 hours longer than the first half.
axiom second_half_larger_by_eleven (v : ℝ) : second_half_time v = first_half_time v + 11

-- We need to prove that the time to run the second half is 22 hours
theorem time_for_second_half : ∀ (v : ℝ), initial_speed v → (second_half_time v = 22) :=
by
  -- This is the place where the actual proof will reside
  sorry

end time_for_second_half_l524_524113


namespace base_of_512_is_7_l524_524710

theorem base_of_512_is_7 :
  ∃ b : ℕ, (b^3 ≤ 512 ∧ 512 < b^4 ∧ (512 % b = 1)) :=
by
  existsi (7 : ℕ)
  split
  sorry
  split
  sorry
  sorry

end base_of_512_is_7_l524_524710


namespace find_angle_C_l524_524454

open Real -- Opening Real to directly use real number functions and constants

noncomputable def triangle_angles_condition (A B C: ℝ) : Prop :=
  2 * sin A + 5 * cos B = 5 ∧ 5 * sin B + 2 * cos A = 2

-- Theorem statement
theorem find_angle_C (A B C: ℝ) (h: triangle_angles_condition A B C):
  C = arcsin (1 / 5) ∨ C = 180 - arcsin (1 / 5) :=
sorry

end find_angle_C_l524_524454


namespace original_price_petrol_in_euros_l524_524629

theorem original_price_petrol_in_euros
  (P : ℝ) -- The original price of petrol in USD per gallon
  (h1 : 0.865 * P * 7.25 + 0.135 * 325 = 325) -- Condition derived from price reduction and additional gallons
  (h2 : P > 0) -- Ensure original price is positive
  (exchange_rate : ℝ) (h3 : exchange_rate = 1.15) : 
  P / exchange_rate = 38.98 :=
by 
  let price_in_euros := P / exchange_rate 
  have h4 : price_in_euros = 38.98 := sorry
  exact h4

end original_price_petrol_in_euros_l524_524629


namespace percentage_primes_divisible_by_five_l524_524578

def primes := [2, 3, 5, 7, 11, 13, 17, 19]
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem percentage_primes_divisible_by_five : 
  (∃ count, count = (list.filter divisible_by_five primes).length) 
  →  (list.length primes) = 8
  →  count = 1
  → (count : ℝ) / 8 * 100 = 12.5 :=
by
  sorry

end percentage_primes_divisible_by_five_l524_524578


namespace line_circle_no_intersection_l524_524378

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524378


namespace distance_between_A_and_B_l524_524654

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end distance_between_A_and_B_l524_524654


namespace opposite_of_2023_l524_524532

def opposite (n x : ℤ) := n + x = 0 

theorem opposite_of_2023 : ∃ x : ℤ, opposite 2023 x ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l524_524532


namespace determine_nice_or_naughty_l524_524689

-- Oracle's function to sum nice divisors of a number
def f (u : ℕ) : ℕ := sorry  -- The actual function is not given, so we leave it as a placeholder.

-- Define the main theorem stating the problem's solution
theorem determine_nice_or_naughty (n : ℕ) (h1 : n > 0) (h2 : n < 1000000) :
  (∃ (s1 s2 s3 s4 : ℕ), 
     let s1 := f(n),
         s2 := f(2*n),
         s3 := f(3*n),
         s4 := f(n+1)
     in (      
       -- Additional conditions or constraints related to these answers
       -- are part of the problem and solution but will require precise formal definitions 
       sorry)) :=
sorry

-- Placeholder statements to ensure the theorem compiles without errors
definition n := 1
example : 0 < n := by sorry
example : n < 1000000 := by sorry

end determine_nice_or_naughty_l524_524689


namespace hyperbola_vertex_distance_l524_524236

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 16 - y^2 / 9 = 1) → (vertex_distance : ℝ := 8) := sorry

end hyperbola_vertex_distance_l524_524236


namespace minimum_value_of_d_l524_524018

def min_d (a b d : ℕ) : Prop :=
  a < b ∧ b < d + 1 ∧
  (∃ x y : ℤ, 2 * x + y = 2023 ∧ y = |x - a| + |x - b| + |x - (d + 1|)) ∧
  (∀ x' y' : ℤ, (2 * x' + y' = 2023 ∧ y' = |x' - a| + |x' - b| + |x' - (d + 1|)) → (x = x' ∧ y = y')) 

theorem minimum_value_of_d (x y a b d : ℕ) (h : min_d a b d) : d = 2020 :=
by sorry

end minimum_value_of_d_l524_524018


namespace geometric_series_sum_l524_524669

theorem geometric_series_sum :
  let a : ℤ := 1
  let r : ℤ := -3
  let n : ℕ := 8
  let last_term : ℤ := -2187
  last_term = a * (r ^ (n - 1)) →
  ∑ i in finset.range n, a * (r ^ i) = -1640 := 
by
  intros a r n last_term h_last_term
  simp only [n, a, r] at h_last_term
  sorry

end geometric_series_sum_l524_524669


namespace mode_median_of_data_set_l524_524156

theorem mode_median_of_data_set :
  let data := [13, 15, 18, 16, 21, 13, 13, 11, 10] in
  (mode data = 13) ∧ (median data = 13) :=
by
  let data := [13, 15, 18, 16, 21, 13, 13, 11, 10]
  unfold mode median
  sorry

end mode_median_of_data_set_l524_524156


namespace centers_of_constructed_squares_form_square_l524_524037

open EuclideanGeometry

structure Parallelogram (A B C D : Point) :=
  (ab // is_parallel A B B C)
  (bc // is_parallel B C C D)
  (cd // is_parallel C D D A)
  (da // is_parallel D A A B)

structure Square (A B C D : Point) :=
  (side_length : ℝ)
  (angle_abc : angle A B C = 90)
  (angle_bcd : angle B C D = 90)
  (angle_cda : angle C D A = 90)
  (angle_dab : angle D A B = 90)

def centers_form_square (par : Parallelogram A B C D) (squares : List (Square Point)) : Prop :=
  ∃ P Q R S : Point,
    center_of_square P ∧ center_of_square Q ∧
    center_of_square R ∧ center_of_square S ∧
    is_square P Q R S

theorem centers_of_constructed_squares_form_square
  (A B C D : Point)
  (par : Parallelogram A B C D)
  (sq_ab : Square A B X Y)
  (sq_bc : Square B C Z W)
  (sq_cd : Square C D U V)
  (sq_da : Square D A T S) :
  centers_form_square par [sq_ab, sq_bc, sq_cd, sq_da] := sorry

end centers_of_constructed_squares_form_square_l524_524037


namespace apples_harvested_l524_524557

theorem apples_harvested (weight_juice weight_restaurant weight_per_bag sales_price total_sales : ℤ) 
  (h1 : weight_juice = 90) 
  (h2 : weight_restaurant = 60) 
  (h3 : weight_per_bag = 5) 
  (h4 : sales_price = 8) 
  (h5 : total_sales = 408) : 
  (weight_juice + weight_restaurant + (total_sales / sales_price) * weight_per_bag = 405) :=
by
  sorry

end apples_harvested_l524_524557


namespace quadratic_roots_interval_l524_524693

theorem quadratic_roots_interval (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 < a ∧ a < x2 ∧ 3 * x1^2 - 4 * (3 * a - 2) * x1 + a^2 + 2 * a = 0 ∧
  3 * x2^2 - 4 * (3 * a - 2) * x2 + a^2 + 2 * a = 0) ↔ a ∈ set.Iio 0 ∪ set.Ioi (5 / 4) :=
by
  sorry

end quadratic_roots_interval_l524_524693


namespace no_intersection_l524_524367

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524367


namespace line_circle_no_intersection_l524_524342

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524342


namespace solve_problem1_l524_524742

noncomputable def problem1 (a : ℝ) (ha : a < 0) : Prop :=
  let α : ℝ := real.atan2 (3*a) (-4*a)
  2 * real.sin α + real.cos α = -2/5

theorem solve_problem1 (a : ℝ) (ha : a < 0) : problem1 a ha :=
by
  sorry

end solve_problem1_l524_524742


namespace max_radius_of_circle_l524_524425

theorem max_radius_of_circle (r : ℝ) (π_pos : 0 < π) : (π * r^2 < 140 * π) → r ≤ 11 := 
by
  intro h_area
  have h1 : r^2 < 140 :=
    by linarith [h_area]
  have h2 : r < real.sqrt 140 :=
    by exact real.sqrt_lt.mp h1
  linarith [real.sqrt_lt.mp h1]

end max_radius_of_circle_l524_524425


namespace percentage_primes_divisible_by_five_l524_524574

def primes := [2, 3, 5, 7, 11, 13, 17, 19]
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem percentage_primes_divisible_by_five : 
  (∃ count, count = (list.filter divisible_by_five primes).length) 
  →  (list.length primes) = 8
  →  count = 1
  → (count : ℝ) / 8 * 100 = 12.5 :=
by
  sorry

end percentage_primes_divisible_by_five_l524_524574


namespace jellybeans_initial_amount_l524_524210

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l524_524210


namespace consecutive_ints_contains_abundant_l524_524021

theorem consecutive_ints_contains_abundant (n : ℕ) :
  ∃ k ∈ finset.range 12, ∃ d : ℕ → finset ℕ, 
  (∀ m ≤ n + k, m ∣ (n + k) → m ≠ 1 ∧ m ≠ (n + k) → m ∈ d (n + k)) ∧
  (n + k < (d (n + k)).sum id) :=
begin
  sorry
end

end consecutive_ints_contains_abundant_l524_524021


namespace loan_difference_eq_1896_l524_524031

/-- 
  Samantha borrows $12,000 with two repayment schemes:
  1. A twelve-year loan with an annual interest rate of 8% compounded semi-annually. 
     At the end of 6 years, she must make a payment equal to half of what she owes, 
     and the remaining balance accrues interest until the end of 12 years.
  2. A twelve-year loan with a simple annual interest rate of 10%, paid as a lump-sum at the end.

  Prove that the positive difference between the total amounts to be paid back 
  under the two schemes is $1,896, rounded to the nearest dollar.
-/
theorem loan_difference_eq_1896 :
  let P := 12000
  let r1 := 0.08
  let r2 := 0.10
  let n := 2
  let t := 12
  let t1 := 6
  let A1 := P * (1 + r1 / n) ^ (n * t1)
  let payment_after_6_years := A1 / 2
  let remaining_balance := A1 / 2
  let compounded_remaining := remaining_balance * (1 + r1 / n) ^ (n * t1)
  let total_compound := payment_after_6_years + compounded_remaining
  let total_simple := P * (1 + r2 * t)
  (total_simple - total_compound).round = 1896 := 
by
  sorry

end loan_difference_eq_1896_l524_524031


namespace simplify_expression_equals_five_l524_524876

noncomputable def simplify_complex_expression : ℂ := 2 * (3 - complex.i) + complex.i * (2 + complex.i)

theorem simplify_expression_equals_five : simplify_complex_expression = 5 := by
  sorry

end simplify_expression_equals_five_l524_524876


namespace length_of_train_correct_l524_524160

-- Define constants for given conditions
def time_to_cross_tree : ℝ := 80
def time_to_cross_platform : ℝ := 146.67
def platform_length : ℝ := 1000

-- Define the length of the train
def train_length : ℝ := 1200

-- Define the speed of the train passing the tree
def speed_passing_tree (L : ℝ) : ℝ := L / time_to_cross_tree

-- Define the speed of the train passing the platform
def speed_passing_platform (L : ℝ) : ℝ := (L + platform_length) / time_to_cross_platform

-- The theorem to prove
theorem length_of_train_correct :
  ∀ (L : ℝ), speed_passing_tree L = speed_passing_platform L → L = train_length :=
by 
sory

end length_of_train_correct_l524_524160


namespace number_of_ways_to_shuffle_32_cards_is_32_factorial_l524_524408

theorem number_of_ways_to_shuffle_32_cards_is_32_factorial : 
  ∃ n : ℕ, n = 32 ∧ (nat.factorial n = nat.factorial 32) := by
  use 32
  split
  · rfl
  · sorry

end number_of_ways_to_shuffle_32_cards_is_32_factorial_l524_524408


namespace sum_of_valid_c_values_l524_524702

theorem sum_of_valid_c_values : 
  (∑ c in {c : ℤ | c ≤ 30 ∧ ∃ k : ℤ, k^2 = 64 + 4*c}, c) = 29 :=
by {
  sorry 
}

end sum_of_valid_c_values_l524_524702


namespace expected_value_of_boy_girl_pairs_l524_524512

noncomputable def expected_value_of_T (boys girls : ℕ) : ℚ :=
  24 * ((boys / 24) * (girls / 23) + (girls / 24) * (boys / 23))

theorem expected_value_of_boy_girl_pairs (boys girls : ℕ) (h_boys : boys = 10) (h_girls : girls = 14) :
  expected_value_of_T boys girls = 12 :=
by
  rw [h_boys, h_girls]
  norm_num
  sorry

end expected_value_of_boy_girl_pairs_l524_524512


namespace sin_double_angle_l524_524431

-- Lean code to define the conditions and represent the problem
variable (α : ℝ)
variable (x y : ℝ) 
variable (r : ℝ := Real.sqrt (x^2 + y^2))

-- Given conditions
def point_on_terminal_side (x y : ℝ) (h : x = 1 ∧ y = -2) : Prop :=
  ∃ α, (⟨1, -2⟩ : ℝ × ℝ) = ⟨Real.cos α * (Real.sqrt (1^2 + (-2)^2)), Real.sin α * (Real.sqrt (1^2 + (-2)^2))⟩

-- The theorem to prove
theorem sin_double_angle (h : point_on_terminal_side 1 (-2) ⟨rfl, rfl⟩) : 
  Real.sin (2 * α) = -4 / 5 := 
sorry

end sin_double_angle_l524_524431


namespace general_eqn_of_line_and_curve_l524_524446

theorem general_eqn_of_line_and_curve :
  (∀ α t, α ≠ π / 2 → ∃ (x y : ℝ), x = 1 + t * cos α ∧ y = t * sin α → y = tan α * (x - 1)) ∧
  (∀ ρ θ, ρ * cos θ * cos θ - 4 * sin θ = 0 → ∃ (x y : ℝ), x^2 = 4 * y) ∧
  let P := (1, 0), M := (0, 1), Q := midpoint A B,
  | Q - P | = 3 * sqrt 2 :=
begin
  sorry
end

end general_eqn_of_line_and_curve_l524_524446


namespace cost_of_450_candies_l524_524996

theorem cost_of_450_candies :
  let cost_per_box := 8
  let candies_per_box := 30
  let num_candies := 450
  cost_per_box * (num_candies / candies_per_box) = 120 := 
by 
  sorry

end cost_of_450_candies_l524_524996


namespace line_circle_no_intersection_l524_524331

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524331


namespace discount_rate_pony_l524_524979

-- Definitions of the conditions
def fox_price := 15
def pony_price := 18
def total_savings := 8.91
def sum_of_discounts := 22

-- Define the expressions for savings
def savings_fox (F : ℝ) := 3 * fox_price * (F / 100)
def savings_pony (P : ℝ) := 2 * pony_price * (P / 100)

-- Define total savings equation
def total_savings_eq (F P : ℝ) := savings_fox F + savings_pony P = total_savings

-- Define the sum of discounts equation
def sum_of_discounts_eq (F P : ℝ) := F + P = sum_of_discounts

-- Declare the theorem to prove the discount rate of Pony jeans
theorem discount_rate_pony : ∃ (P : ℝ), (
  total_savings_eq (sum_of_discounts - P) P ∧
  sum_of_discounts_eq (sum_of_discounts - P) P ∧
  P = 11
) :=
sorry

end discount_rate_pony_l524_524979


namespace range_of_f_l524_524409

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin x + cos x

theorem range_of_f :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ π/2 → - (sqrt 3) ≤ f x ∧ f x ≤ 2 := by
  sorry

end range_of_f_l524_524409


namespace line_circle_no_intersection_l524_524406

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524406


namespace remainder_cd_mod_40_l524_524842

theorem remainder_cd_mod_40 (c d : ℤ) (hc : c % 80 = 75) (hd : d % 120 = 117) : (c + d) % 40 = 32 :=
by
  sorry

end remainder_cd_mod_40_l524_524842


namespace number_of_students_second_center_l524_524256

noncomputable def number_of_students := 500
noncomputable def sample_size := 50
noncomputable def first_selected := 3
noncomputable def interval := number_of_students / sample_size
noncomputable def range_second_center := {i // 201 ≤ i ∧ i ≤ 355}

theorem number_of_students_second_center : 
  (set_of (λ i : ℕ, i % interval = first_selected % interval)).filter (λ i, i ∈ range_second_center).card = 16 := 
  sorry

end number_of_students_second_center_l524_524256


namespace geometric_ratio_of_T_l524_524824

-- Definitions used in Lean 4 statement
def T_n (n : ℕ) : ℕ := (2^(n * (n - 1) / 2)) * (n : ℕ) -- This is a placeholder. The product of the first n terms of a geometric sequence with ratio 2 needs to be defined accurately.

-- Statement equivalent to the proof problem
theorem geometric_ratio_of_T :
  let r := 2 in
  let T (n : ℕ) := T_n n in
  geometric_sequence (λ n, T (3 * (n + 1)) / T (3 * n)) ∧ 
  common_ratio (λ n, T (3 * (n + 1)) / T (3 * n)) = 512 :=
begin
  sorry
end

end geometric_ratio_of_T_l524_524824


namespace no_intersection_l524_524363

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524363


namespace exists_additive_func_l524_524286

theorem exists_additive_func (f : ℝ → ℝ) (h : ∀ x y : ℝ, |f(x + y) - f(x) - f(y)| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x y : ℝ, g (x + y) = g x + g y) ∧ (∀ x : ℝ, |f x - g x| ≤ 1) :=
sorry

end exists_additive_func_l524_524286


namespace solve_coffee_problem_l524_524140

variables (initial_stock new_purchase : ℕ)
           (initial_decaf_percentage new_decaf_percentage : ℚ)
           (total_stock total_decaf weight_percentage_decaf : ℚ)

def coffee_problem :=
  initial_stock = 400 ∧
  initial_decaf_percentage = 0.20 ∧
  new_purchase = 100 ∧
  new_decaf_percentage = 0.50 ∧
  total_stock = initial_stock + new_purchase ∧
  total_decaf = initial_stock * initial_decaf_percentage + new_purchase * new_decaf_percentage ∧
  weight_percentage_decaf = (total_decaf / total_stock) * 100

theorem solve_coffee_problem : coffee_problem 400 100 0.20 0.50 500 130 26 :=
by {
  sorry
}

end solve_coffee_problem_l524_524140


namespace sum_factors_24_l524_524960

theorem sum_factors_24 : (∑ d in (finset.filter (λ d, 24 % d = 0) (finset.range (25))), d) = 60 :=
by
  sorry

end sum_factors_24_l524_524960


namespace no_real_intersections_l524_524347

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524347


namespace triangular_25_eq_325_l524_524099

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l524_524099


namespace Durakavalyanie_last_lesson_class_1C_l524_524985

theorem Durakavalyanie_last_lesson_class_1C :
  ∃ (class_lesson : String × Nat → String), 
  class_lesson ("1B", 1) = "Kurashenie" ∧
  (∃ (k m n : Nat), class_lesson ("1A", k) = "Durakavalyanie" ∧ class_lesson ("1B", m) = "Durakavalyanie" ∧ m > k) ∧
  class_lesson ("1A", 2) ≠ "Nizvedenie" ∧
  class_lesson ("1C", 3) = "Durakavalyanie" :=
sorry

end Durakavalyanie_last_lesson_class_1C_l524_524985


namespace probability_at_tree_correct_expected_distance_correct_l524_524492

-- Define the initial conditions
def initial_tree (n : ℕ) : ℕ := n + 1
def total_trees (n : ℕ) : ℕ := 2 * n + 1

-- Define the probability that the drunkard is at each tree T_i (1 <= i <= 2n+1) at the end of the nth minute
def probability_at_tree (n i : ℕ) : ℚ :=
  if 1 ≤ i ∧ i ≤ total_trees n then
    (Nat.choose (2*n) (i-1)) / (2^(2*n))
  else
    0

-- Define the expected distance between the final position and the initial tree T_{n+1}
def expected_distance (n : ℕ) : ℚ :=
  n * (Nat.choose (2*n) n) / (2^(2*n))

-- Statements to prove
theorem probability_at_tree_correct (n i : ℕ) (hi : 1 ≤ i ∧ i ≤ total_trees n)  :
  probability_at_tree n i = (Nat.choose (2*n) (i-1)) / (2^(2*n)) :=
by
  sorry

theorem expected_distance_correct (n : ℕ) :
  expected_distance n = n * (Nat.choose (2*n) n) / (2^(2*n)) :=
by
  sorry

end probability_at_tree_correct_expected_distance_correct_l524_524492


namespace fraction_to_decimal_l524_524194

theorem fraction_to_decimal : (22 / 8 : ℝ) = 2.75 := 
sorry

end fraction_to_decimal_l524_524194


namespace flour_needed_per_pizza_l524_524461

-- Definitions based on the problem
def total_hours : ℕ := 7
def flour_kg : ℝ := 22
def time_per_pizza_min : ℕ := 10
def remaining_pizzas : ℕ := 2

-- Total minutes available
def total_minutes := total_hours * 60

-- Number of pizzas made in the available time
def pizzas_made := total_minutes / time_per_pizza_min

-- Total number of pizzas including those made with remaining flour
def total_pizzas := pizzas_made + remaining_pizzas

-- Amount of flour per pizza
def flour_per_pizza : ℝ := flour_kg / total_pizzas

-- We now state the theorem for the proof problem
theorem flour_needed_per_pizza : flour_per_pizza = 0.5 := sorry

end flour_needed_per_pizza_l524_524461


namespace age_difference_l524_524781

theorem age_difference (x : ℕ) (older_age younger_age : ℕ) 
  (h1 : 3 * x = older_age)
  (h2 : 2 * x = younger_age)
  (h3 : older_age + younger_age = 60) : 
  older_age - younger_age = 12 := 
by
  sorry

end age_difference_l524_524781


namespace inverse_proportion_relationship_l524_524740

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem inverse_proportion_relationship (k x1 x2 y1 y2 : ℝ)
    (hk : k < 0)
    (hx1 : x1 < x2)
    (hx2 : x2 < 0)
    (hy1 : y1 = inverse_proportion_function k x1)
    (hy2 : y2 = inverse_proportion_function k x2) :
    y2 > y1 ∧ y1 > 0 :=
begin
  sorry
end

end inverse_proportion_relationship_l524_524740


namespace apples_in_box_l524_524516

-- Define the initial conditions
def oranges : ℕ := 12
def removed_oranges : ℕ := 6
def target_percentage : ℚ := 0.70

-- Define the function that models the problem
def fruit_after_removal (apples : ℕ) : ℕ := apples + (oranges - removed_oranges)
def apples_percentage (apples : ℕ) : ℚ := (apples : ℚ) / (fruit_after_removal apples : ℚ)

-- The theorem states the question and expected answer
theorem apples_in_box : ∃ (apples : ℕ), apples_percentage apples = target_percentage ∧ apples = 14 :=
by
  sorry

end apples_in_box_l524_524516


namespace m_condition_sufficient_not_necessary_l524_524423

theorem m_condition_sufficient_not_necessary (m : ℤ) : 
  let A := {-1, m^2}
  let B := {2, 9}
  (m = 3 → A ∩ B = {9}) ∧ (A ∩ B = {9} → m = 3 ∨ m = -3) :=
by
  let A := {-1, m^2}
  let B := {2, 9}
  sorry

end m_condition_sufficient_not_necessary_l524_524423


namespace no_six_digit_number_meets_criteria_l524_524769

def valid_digit (n : ℕ) := 2 ≤ n ∧ n ≤ 8

theorem no_six_digit_number_meets_criteria :
  ¬ ∃ (digits : Finset ℕ), digits.card = 6 ∧ (∀ x ∈ digits, valid_digit x) ∧ (digits.sum id = 42) :=
by {
  sorry
}

end no_six_digit_number_meets_criteria_l524_524769


namespace domain_of_f_l524_524561

def f (x : ℝ) : ℝ := 1 / (Real.log x)

theorem domain_of_f :
  ∀ x : ℝ, (0 < x ∧ x ≠ 1) ↔ x ∈ set.Ioo 0 1 ∪ set.Ioi 1 :=
by
  sorry

end domain_of_f_l524_524561


namespace simplify_expression_l524_524508

theorem simplify_expression (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 :=
sorry

end simplify_expression_l524_524508


namespace number_of_cars_repaired_l524_524186

theorem number_of_cars_repaired
  (oil_change_cost repair_cost car_wash_cost : ℕ)
  (oil_changes repairs car_washes total_earnings : ℕ)
  (h₁ : oil_change_cost = 20)
  (h₂ : repair_cost = 30)
  (h₃ : car_wash_cost = 5)
  (h₄ : oil_changes = 5)
  (h₅ : car_washes = 15)
  (h₆ : total_earnings = 475)
  (h₇ : 5 * oil_change_cost + 15 * car_wash_cost + repairs * repair_cost = total_earnings) :
  repairs = 10 :=
by sorry

end number_of_cars_repaired_l524_524186


namespace percentage_of_primes_divisible_by_5_l524_524567

def primes_less_than_twenty : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def percentage (num total : ℕ) : ℚ := (num.to_rat / total.to_rat) * 100

theorem percentage_of_primes_divisible_by_5 :
  percentage (primes_less_than_twenty.count divisible_by_five) (primes_less_than_twenty.size) = 12.5 :=
by
  sorry

end percentage_of_primes_divisible_by_5_l524_524567


namespace find_length_of_AB_l524_524736

noncomputable def area_ΔABC : ℝ := Real.sqrt 3
def BC_len : ℝ := 2
def angle_C : ℝ := Real.pi / 3 -- 60 degrees in radians

theorem find_length_of_AB :
  ∀ (AC AB : ℝ), 
    (1 / 2 * AC * BC_len * Real.sin angle_C = area_ΔABC) → 
    (AB^2 = AC^2 + BC_len^2 - 2 * AC * BC_len * Real.cos angle_C) → 
    AB = 2 := 
by
  intros AC AB h_area h_cos
  sorry

end find_length_of_AB_l524_524736


namespace kenneth_past_finish_line_when_biff_finishes_l524_524179

-- Given conditions:
def race_distance : ℕ := 500
def biff_speed : ℕ := 50
def kenneth_speed : ℕ := 51

-- The statement to prove:
theorem kenneth_past_finish_line_when_biff_finishes :
  let time_biff_to_finish := race_distance / biff_speed in
  let distance_kenneth_in_that_time := kenneth_speed * time_biff_to_finish in
  distance_kenneth_in_that_time - race_distance = 10 :=
by
  sorry

end kenneth_past_finish_line_when_biff_finishes_l524_524179


namespace equation_of_tangent_line_l524_524750

theorem equation_of_tangent_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + a * y - 17 = 0) →
   (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 4 * x - 3 * y + 11 = 0) :=
sorry

end equation_of_tangent_line_l524_524750


namespace area_sum_of_triangles_l524_524432

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  (s * (s - a) * (s - b) * (s - c))^.sqrt

theorem area_sum_of_triangles (A B C D E : EuclideanSpace ℝ (Fin 2))
  (hE : E = (B + C) / 2) (hD : ∃ t, D = t • A + (1 - t) • C)
  (h_AC_length : dist A C = 2)
  (h_ABC_right : ∠ A B C = 45)
  (h_BAC_right : ∠ B A C = 90)
  (h_ACB_right : ∠ A C B = 45)
  (h_DEC_right : ∠ D E C = 45) :
  let ΔABC_area := area_of_triangle (dist A B) (dist B C) (dist C A)
  let ΔCDE_area := area_of_triangle (dist C D) (dist D E) (dist E C)
  in ΔABC_area + 2 * ΔCDE_area = 3 :=
by
  sorry

end area_sum_of_triangles_l524_524432


namespace tetrahedron_properties_l524_524122

def A1 : ℝ × ℝ × ℝ := (-1, 2, -3)
def A2 : ℝ × ℝ × ℝ := (4, -1, 0)
def A3 : ℝ × ℝ × ℝ := (2, 1, -2)
def A4 : ℝ × ℝ × ℝ := (3, 4, 5)

noncomputable def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

noncomputable def scalar_triple_product (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) - 
  v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) + 
  v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)

noncomputable def tetrahedron_volume (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * abs (scalar_triple_product (vector_sub A1 A2) (vector_sub A1 A3) (vector_sub A1 A4))

noncomputable def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def triangle_area (A1 A2 A3 : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 2) * vector_magnitude (cross_product (vector_sub A1 A2) (vector_sub A1 A3))

noncomputable def height_from_vertex (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ :=
  3 * tetrahedron_volume A1 A2 A3 A4 / triangle_area A1 A2 A3

theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 20 / 3 ∧
  height_from_vertex A1 A2 A3 A4 = 5 * real.sqrt 2 :=
by sorry

end tetrahedron_properties_l524_524122


namespace cognitive_movement_fits_l524_524640

-- Definitions for the phrases as conditions
def generalizing_indiscriminately : Prop := "Generalizing indiscriminately"
def rumors_make_man_out_of_tiger : Prop := "Rumors can make a man out of a tiger"
def flood_of_emotions : Prop := "A flood of emotions"
def thousand_worries_one_insight : Prop := "A thousand worries yield one insight"

-- Definition of repetitiveness and infiniteness of cognitive movement
def cognitive_repetitiveness (p : Prop) : Prop :=
p = "Correct cognition requires multiple cycles of moving from practice to cognition and then back from cognition to practice"

def cognitive_infiniteness (p : Prop) : Prop :=
p = "The pursuit of truth is an endless process"

-- The actual proof problem
theorem cognitive_movement_fits :
  thousand_worries_one_insight ∧ 
  cognitive_repetitiveness "A thousand worries yield one insight" ∧
  cognitive_infiniteness "A thousand worries yield one insight" := 
sorry

end cognitive_movement_fits_l524_524640


namespace arithmetic_series_sum_l524_524657

theorem arithmetic_series_sum : 
  let a := -48 in
  let d := 3 in
  let n := (0 - a) / d + 1 in
  let total_sum := (n * (a + 0)) / 2 in
  n = 17 ∧ total_sum = -408 :=
by
  let a := -48
  let d := 3
  let n := (0 - a) / d + 1
  have h1 : n = 17 := sorry
  let total_sum := (n * (a + 0)) / 2
  have h2 : total_sum = -408 := sorry
  exact ⟨h1, h2⟩

end arithmetic_series_sum_l524_524657


namespace sufficient_but_not_necessary_condition_l524_524410

theorem sufficient_but_not_necessary_condition (a b : ℝ) (hb : b < -1) : |a| + |b| > 1 := 
by
  sorry

end sufficient_but_not_necessary_condition_l524_524410


namespace Berry_read_pages_thursday_l524_524175

theorem Berry_read_pages_thursday :
  ∀ (pages_per_day : ℕ) (pages_sunday : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) 
    (pages_wednesday : ℕ) (pages_friday : ℕ) (pages_saturday : ℕ),
    (pages_per_day = 50) →
    (pages_sunday = 43) →
    (pages_monday = 65) →
    (pages_tuesday = 28) →
    (pages_wednesday = 0) →
    (pages_friday = 56) →
    (pages_saturday = 88) →
    pages_sunday + pages_monday + pages_tuesday +
    pages_wednesday + pages_friday + pages_saturday + x = 350 →
    x = 70 := by
  sorry

end Berry_read_pages_thursday_l524_524175


namespace triangle_area_bounds_l524_524162

theorem triangle_area_bounds (s : ℝ) 
  (vertex : (0, -2))
  (intersections : { P : ℝ × ℝ | P = ((- (s + 2)^(1/3)), s) ∨ P = ((s + 2)^(1/3), s)})
  (area_eq : (s + 2)^(4/3))
  (area_bounds : 12 ≤ (s + 2)^(4/3) ∧ (s + 2)^(4/3) ≤ 72) :
  (3:ℝ).sqrt ≤ s + 2 ∧ s + 2 ≤ (72:ℝ)^(3/4) :=
by
  sorry

end triangle_area_bounds_l524_524162


namespace g_five_eq_one_l524_524065

noncomputable def g : ℝ → ℝ := sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_nonzero : ∀ x : ℝ, g x ≠ 0

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l524_524065


namespace percentage_primes_divisible_by_five_l524_524575

def primes := [2, 3, 5, 7, 11, 13, 17, 19]
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem percentage_primes_divisible_by_five : 
  (∃ count, count = (list.filter divisible_by_five primes).length) 
  →  (list.length primes) = 8
  →  count = 1
  → (count : ℝ) / 8 * 100 = 12.5 :=
by
  sorry

end percentage_primes_divisible_by_five_l524_524575


namespace unique_pairs_count_l524_524872

theorem unique_pairs_count : ∃ f m : ℕ, 
  (f, m) ∈ {(0, 7), (2, 7), (4, 7), (6, 7), (7, 7), (7, 0), (7, 2), (7, 4), (7, 6)} ∧ 
  (f, m) = (f', m') ∧ 
  (f' , m' ) ∈ {(0, 7), (2, 7), (4, 7), (6, 7), (7, 7), (7, 0), (7, 2), (7, 4), (7, 6)} ∧
  (7: ℕ) = 9 :=
begin
  sorry -- Proof not required
end

end unique_pairs_count_l524_524872


namespace avg_adjacent_boy_girl_pairs_l524_524884

theorem avg_adjacent_boy_girl_pairs :
  let boys := 6
  let girls := 14
  let total_people := boys + girls
  let total_pairs := total_people - 1
  let prob_boy_girl := (boys / total_people) * (girls / (total_people - 1))
  let total_prob := 2 * prob_boy_girl
  let expected_pairs := total_pairs * total_prob
  ⌊expected_pairs⌉ = 8
  := by
  let boys := 6
  let girls := 14
  let total_people := boys + girls
  let total_pairs := total_people - 1
  let prob_boy_girl := (boys / total_people.toReal) * (girls / (total_people - 1).toReal)
  let total_prob := 2 * prob_boy_girl
  let expected_pairs := total_pairs.toReal * total_prob
  have h := Real.floor_eq 8 expected_pairs
  sorry

end avg_adjacent_boy_girl_pairs_l524_524884


namespace remainder_2012_digit_number_l524_524526

/-- A sequence of natural numbers 1, 2, 3, ..., 9, repeated to form a 2012-digit number. -/
noncomputable def repeated_sequence_digits := λ n : ℕ, if n % 9 = 8 then 9 else (n % 9) + 1

/-- The sum of a list of natural numbers. -/
def sum_list (l : list ℕ) : ℕ := l.sum

/-- The remainder when the 2012-digit number formed by repeating the digits 1, 2, ..., 9 is divided by 9 is 6. -/
theorem remainder_2012_digit_number : 
  sum_list (list.map repeated_sequence_digits (list.range 2012)) % 9 = 6 :=
sorry

end remainder_2012_digit_number_l524_524526


namespace percentage_decrease_is_approx_l524_524517

def transaction_processing_fees_last_year : ℝ := 40.0
def transaction_processing_fees_this_year : ℝ := 28.8
def data_processing_fees_last_year : ℝ := 25.0
def data_processing_fees_this_year : ℝ := 20.0
def cross_border_transactions_last_year : ℝ := 20.0
def cross_border_transactions_this_year : ℝ := 17.6

noncomputable def total_revenue_last_year :=
  transaction_processing_fees_last_year + data_processing_fees_last_year + cross_border_transactions_last_year

noncomputable def total_revenue_this_year :=
  transaction_processing_fees_this_year + data_processing_fees_this_year + cross_border_transactions_this_year

noncomputable def decrease_in_revenue := total_revenue_last_year - total_revenue_this_year

noncomputable def percentage_decrease := (decrease_in_revenue / total_revenue_last_year) * 100

theorem percentage_decrease_is_approx :
  percentage_decrease ≈ 21.88 :=
sorry

end percentage_decrease_is_approx_l524_524517


namespace a7b7_eq_twenty_nine_l524_524850

noncomputable theory

def a : ℝ := sorry
def b : ℝ := sorry

axiom ab_eq_one : a + b = 1
axiom a2b2_eq_three : a^2 + b^2 = 3
axiom a3b3_eq_four : a^3 + b^3 = 4
axiom a4b4_eq_seven : a^4 + b^4 = 7
axiom a5b5_eq_eleven : a^5 + b^5 = 11

theorem a7b7_eq_twenty_nine : a^7 + b^7 = 29 := by
  have h1 : a^6 + b^6 = 18 := by sorry
  have h2 : a^7 + b^7 = 29 := by sorry
  exact h2

end a7b7_eq_twenty_nine_l524_524850


namespace sum_max_min_of_f_is_16_l524_524913

def f (x : ℝ) : ℝ := 7 - 4 * sin x * cos x + 4 * (cos x)^2 - 4 * (cos x)^4

theorem sum_max_min_of_f_is_16 : (max (f x ∣ x ∈ ℝ)) + (min (f x ∣ x ∈ ℝ)) = 16 :=
by sorry

end sum_max_min_of_f_is_16_l524_524913


namespace unique_gcd_triplet_l524_524017

lemma gcd_lemma {a b c m : ℕ} 
  (h1 : m ∣ (gcd a b)) 
  (h2 : m ∣ (gcd b c)) 
  : m ∣ (gcd c a) := by sorry

theorem unique_gcd_triplet 
  (a b c : ℕ) 
  (x y z : ℕ)
  (hx : x = gcd a b)
  (hy : y = gcd b c)
  (hz : z = gcd c a)
  (hx_set : x ∈ {6, 8, 12, 18, 24})
  (hy_set : y ∈ {14, 20, 28, 44, 56})
  (hz_set : z ∈ {5, 15, 18, 27, 42}) 
  : x = 8 ∧ y = 14 ∧ z = 18 := by
  sorry

end unique_gcd_triplet_l524_524017


namespace line_circle_no_intersection_l524_524335

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524335


namespace nickys_pace_l524_524848

-- Define the conditions using variables and constants in Lean
def cristina_pace : ℕ := 5
def time_run : ℕ := 15
def head_start : ℕ := 30
def cristina_distance := cristina_pace * time_run

-- The statement we want to prove: Nicky's pace is 3 meters per second.
theorem nickys_pace (v : ℕ) (nicky_distance : ℕ) (condition : nicky_distance = v * time_run + head_start) : v = 3 :=
by
  have h : cristina_distance = 75 := rfl
  have h1 : cristina_distance = nicky_distance := by rw condition
  have h2 : 75 = v * time_run + head_start := by rwa [h, h1]
  have h3 : v * 15 + 30 = 75 := by rw h2
  have h4 : v * 15 = 45 := by linarith 
  have h5 : v = 3 := by norm_num
  exact h5

end nickys_pace_l524_524848


namespace distance_between_vertices_of_hyperbola_l524_524244

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) → 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l524_524244


namespace green_jelly_bean_probability_l524_524610

def total_jelly_beans (red green yellow blue black : ℕ) : ℕ :=
  red + green + yellow + blue + black

def probability_green (green total : ℕ) : ℚ :=
  green / total

theorem green_jelly_bean_probability :
  let red := 8 in
  let green := 10 in
  let yellow := 9 in
  let blue := 12 in
  let black := 5 in
  let total := total_jelly_beans red green yellow blue black in
  probability_green green total = 5 / 22 :=
by
  sorry

end green_jelly_bean_probability_l524_524610


namespace basketball_team_total_wins_l524_524611

theorem basketball_team_total_wins : 
  let w1 := 40 in 
  let w2 := (5 / 8) * w1 in 
  let w3 := w1 + w2 in 
  let w4 := (3 / 5) * (w1 + w2 + w3) in 
  let total_wins := w1 + w2 + w3 + w4 in 
  total_wins = 208 :=
by
  sorry

end basketball_team_total_wins_l524_524611


namespace sin_double_angle_l524_524260

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = -3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l524_524260


namespace projection_of_a_on_b_l524_524318

open Real

-- Define the vectors a and b
def a : EuclideanSpace ℝ _ := (2 : ℝ, 1 : ℝ) : EuclideanSpace ℝ (Fin 2)
def b : EuclideanSpace ℝ _ := (-1 : ℝ, 1 : ℝ) : EuclideanSpace ℝ (Fin 2)

-- Dot product definition
def dotProduct (x y : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (Finset.univ.sum (λ i, x i * y i))

-- Magnitude of vector b
def bNorm : ℝ := √(dotProduct b b)

-- Projection of a onto b
noncomputable def projection (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (dotProduct a b) / (bNorm)

-- Theorem stating the required proof
theorem projection_of_a_on_b : projection a b = -√2 / 2 :=
  sorry

end projection_of_a_on_b_l524_524318


namespace rational_segment_count_l524_524093

noncomputable def number_of_rational_segments (AB AC BC : ℝ) (n : ℕ) : ℕ :=
  let number_of_regions := n + 1
  let lengths := λ k, (Real.sqrt (3 * k) / (7 * Real.sqrt 2)).denom = 1
  let valid_lengths := Finset.filter lengths (Finset.range (number_of_regions + 1))
  valid_lengths.card

theorem rational_segment_count :
  number_of_rational_segments 9 12 (5 * Real.sqrt 3) 2449 = 28 :=
sorry

end rational_segment_count_l524_524093


namespace total_trash_pieces_l524_524092

theorem total_trash_pieces (classroom_trash : ℕ) (outside_trash : ℕ)
  (h1 : classroom_trash = 344) (h2 : outside_trash = 1232) : 
  classroom_trash + outside_trash = 1576 :=
by
  sorry

end total_trash_pieces_l524_524092


namespace five_distinct_lines_sum_l524_524253

noncomputable def sum_all_possible_N : ℕ := 45

theorem five_distinct_lines_sum (lines : set (set (ℝ × ℝ))) (h : lines.card = 5) :
  let N := {p : ℝ × ℝ | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ p ∈ l1 ∧ p ∈ l2} in
  ∑ n in (N).to_finset, 1 = sum_all_possible_N :=
sorry

end five_distinct_lines_sum_l524_524253


namespace square_length_n_graph_l524_524883

def p (x : ℝ) : ℝ := sorry
def q (x : ℝ) : ℝ := sorry
def r (x : ℝ) : ℝ := sorry

def m (x : ℝ) : ℝ := max (max (p x) (q x)) (r x)
def n (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

def l : ℝ := (
  let d1 := real.sqrt (((-1 : ℝ) - (-4))^2 + (6 - 1)^2) in
  let d2 := real.sqrt ((4 - (-1 : ℝ))^2 + (1 - 6)^2) in
  d1 + d2
)

theorem square_length_n_graph : 
  (∀ x, -4 ≤ x ∧ x ≤ 4 → (m x = if x = -4 then 6 else if x = -1 then 1 else if x = 4 then 6 else sorry)) →
  l^2 = 166.46 :=
by
  intros,
  sorry

end square_length_n_graph_l524_524883


namespace helen_baked_chocolate_chip_cookies_l524_524768

theorem helen_baked_chocolate_chip_cookies (chocolate_chip_yesterday raisin_yesterday raisin_today c_y: ℕ) 
  (h1: chocolate_chip_yesterday = 519) 
  (h2: raisin_yesterday = 300) 
  (h3: raisin_today = 280) 
  (h4: raisin_yesterday = raisin_today + 20):
  c_y = 539 := 
by
  have total_yesterday := 519 + 300
  have total_today := total_yesterday
  have chocolate_chip_today := total_today - 280
  exact chocolate_chip_today
  
/-
Conditions:
chocolate_chip_yesterday = 519
raisin_yesterday = 300
raisin_today = 280
raisin_yesterday = raisin_today + 20
Proof that:
Helen baked c_y = 539 chocolate chip cookies today.
-/

end helen_baked_chocolate_chip_cookies_l524_524768


namespace angle_is_2pi_over_3_l524_524319

-- Definitions as per the problem conditions
variables {a b : ℝ^3}

axiom mag_a : ∥a∥ = 2
axiom mag_b : ∥b∥ = 1
axiom perp_condition : (a + b) ⬝ b = 0

-- Proposition to prove the angle θ between vectors a and b
noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ :=
real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem angle_is_2pi_over_3 (a b : ℝ^3) [mag_a : ∥a∥ = 2] [mag_b : ∥b∥ = 1] [perp_condition : (a + b) ⬝ b = 0] :
  angle_between_vectors a b = 2 * real.pi / 3 :=
sorry

end angle_is_2pi_over_3_l524_524319


namespace centers_of_constructed_squares_form_square_l524_524038

open EuclideanGeometry

structure Parallelogram (A B C D : Point) :=
  (ab // is_parallel A B B C)
  (bc // is_parallel B C C D)
  (cd // is_parallel C D D A)
  (da // is_parallel D A A B)

structure Square (A B C D : Point) :=
  (side_length : ℝ)
  (angle_abc : angle A B C = 90)
  (angle_bcd : angle B C D = 90)
  (angle_cda : angle C D A = 90)
  (angle_dab : angle D A B = 90)

def centers_form_square (par : Parallelogram A B C D) (squares : List (Square Point)) : Prop :=
  ∃ P Q R S : Point,
    center_of_square P ∧ center_of_square Q ∧
    center_of_square R ∧ center_of_square S ∧
    is_square P Q R S

theorem centers_of_constructed_squares_form_square
  (A B C D : Point)
  (par : Parallelogram A B C D)
  (sq_ab : Square A B X Y)
  (sq_bc : Square B C Z W)
  (sq_cd : Square C D U V)
  (sq_da : Square D A T S) :
  centers_form_square par [sq_ab, sq_bc, sq_cd, sq_da] := sorry

end centers_of_constructed_squares_form_square_l524_524038


namespace algebraic_expression_comparison_l524_524590

theorem algebraic_expression_comparison 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_neq : a ≠ b) :
  (let expr1 := (a + 1 / a) * (b + 1 / b),
       expr2 := (√(a * b) + 1 / √(a * b)) ^ 2,
       expr3 := ((a + b) / 2 + 2 / (a + b)) ^ 2 in
    ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a ≠ b) ∧ (expr1 > expr3 ∨ expr3 > expr1)) :=
by
  sorry

end algebraic_expression_comparison_l524_524590


namespace find_y_l524_524043

open Real

structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

def parallel (v₁ v₂ : Vec3) : Prop := ∃ s : ℝ, v₁ = ⟨s * v₂.x, s * v₂.y, s * v₂.z⟩

def orthogonal (v₁ v₂ : Vec3) : Prop := (v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z) = 0

noncomputable def correct_y (x y : Vec3) : Vec3 :=
  ⟨(8 : ℝ) - 2 * (2 : ℝ), (-4 : ℝ) - 2 * (2 : ℝ), (2 : ℝ) - 2 * (2 : ℝ)⟩

theorem find_y :
  ∀ (x y : Vec3),
    (x.x + y.x = 8) ∧ (x.y + y.y = -4) ∧ (x.z + y.z = 2) →
    (parallel x ⟨2, 2, 2⟩) →
    (orthogonal y ⟨1, -1, 0⟩) →
    y = ⟨4, -8, -2⟩ :=
by
  intros x y Hxy Hparallel Horthogonal
  sorry

end find_y_l524_524043


namespace determine_nice_or_naughty_l524_524686

-- Definitions
def nice (n : ℕ) : Prop := sorry  -- Assume we have a predicate to determine if n is nice or not
def sum_nice_divisors (n : ℕ) : ℕ := sorry -- Assume a function that gives the sum of all nice divisors of n

-- Main theorem to prove
theorem determine_nice_or_naughty (n : ℕ) (h : n < 1000000) : 
  ∃ oracle_calls : fin 4 → ℕ, 
    ∀ k : fin 4, oracle_calls k = sum_nice_divisors ?m_1 ∧ (nice n ∨ ¬ nice n) :=
begin
  -- Proof steps would go here
  sorry
end

end determine_nice_or_naughty_l524_524686


namespace students_taking_neither_l524_524851

-- Definitions based on conditions
def total_students : ℕ := 60
def students_CS : ℕ := 40
def students_Elec : ℕ := 35
def students_both_CS_and_Elec : ℕ := 25

-- Lean statement to prove the number of students taking neither computer science nor electronics
theorem students_taking_neither : total_students - (students_CS + students_Elec - students_both_CS_and_Elec) = 10 :=
by
  sorry

end students_taking_neither_l524_524851


namespace percentage_of_primes_divisible_by_5_l524_524566

def primes_less_than_twenty : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def percentage (num total : ℕ) : ℚ := (num.to_rat / total.to_rat) * 100

theorem percentage_of_primes_divisible_by_5 :
  percentage (primes_less_than_twenty.count divisible_by_five) (primes_less_than_twenty.size) = 12.5 :=
by
  sorry

end percentage_of_primes_divisible_by_5_l524_524566


namespace find_line_eq_l524_524288

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

variables {a b : ℝ}
def P := (a, b)
def Q := (3 - b, 3 - a)
def midP_Q := midpoint P Q

lemma symmetric_point_eq : x + y = 3 := sorry

theorem find_line_eq {x y : ℝ} (h1 : y = (3 - a + b) / 2) (h2 : x = (3 - b + a) / 2) :
  x + y - 3 = 0 :=
by {
  rw [h1, h2],
  sorry
}

end find_line_eq_l524_524288


namespace range_of_a_l524_524699

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end range_of_a_l524_524699


namespace total_distance_traveled_is_63_l524_524993

-- defining the initial conditions
def initial_height : ℝ := 20
def bounce_ratio : ℝ := 2/3
def wind_adjustment : ℝ := 0.9

-- Auxiliary definitions for the computed distances
def first_ascent := initial_height * bounce_ratio * wind_adjustment
def second_descent := first_ascent
def second_ascent := second_descent * bounce_ratio * wind_adjustment
def third_descent := second_ascent
def third_ascent := third_descent * bounce_ratio * wind_adjustment

-- Sum of all distances traveled
def total_distance : ℝ :=
  initial_height +
  second_descent +
  first_ascent +
  third_descent +
  second_ascent +
  third_ascent

-- Proof statement
theorem total_distance_traveled_is_63 : round total_distance = 63 :=
by
  exact sorry

end total_distance_traveled_is_63_l524_524993


namespace cos_in_third_quadrant_l524_524420

theorem cos_in_third_quadrant (B : ℝ) (h_sin_B : Real.sin B = -5/13) (h_quadrant : π < B ∧ B < 3 * π / 2) : Real.cos B = -12/13 :=
by
  sorry

end cos_in_third_quadrant_l524_524420


namespace no_intersection_l524_524364

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524364


namespace track_length_proof_l524_524180

noncomputable def track_length : ℝ :=
  let x := 541.67
  x

theorem track_length_proof
  (p : ℝ)
  (q : ℝ)
  (h1 : p = 1 / 4)
  (h2 : q = 120)
  (h3 : ¬(p = q))
  (h4 : ∃ r : ℝ, r = 180)
  (speed_constant : ∃ b_speed, ∃ s_speed, b_speed * t = q ∧ s_speed * t = r) :
  track_length = 541.67 :=
sorry

end track_length_proof_l524_524180


namespace socks_count_l524_524459

theorem socks_count :
  ∃ (x y z : ℕ), x + y + z = 12 ∧ x + 3 * y + 4 * z = 24 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 <= z ∧ x = 7 :=
by
  sorry

end socks_count_l524_524459


namespace tan_A_eq_2_compute_c_l524_524279

-- Definitions for given conditions
variables {A B C : ℝ} -- Internal angles in the triangle ABC
variables {a b c : ℝ} -- Opposite sides to the angles A, B, C respectively

-- The equation given in the problem
axiom given_equation : (a + b) * (sin A - sin B) = (c - b * sin A) * sin C

-- Part 1: Proving that tan A = 2
theorem tan_A_eq_2 (h1 : (a + b) * (sin A - sin B) = (c - b * sin A) * sin C) : tan A = 2 :=
sorry

-- Part 2: Given a = 2 and C = π/3, proving that c = sqrt(15)/2
theorem compute_c (h1 : (a + b) * (sin A - sin B) = (c - b * sin A) * sin C)
    (h2 : a = 2) (h3 : C = π / 3) : c = sqrt(15) / 2 :=
sorry

end tan_A_eq_2_compute_c_l524_524279


namespace correct_average_is_26_l524_524048

noncomputable def initial_average : ℕ := 20
noncomputable def number_of_numbers : ℕ := 10
noncomputable def incorrect_number : ℕ := 26
noncomputable def correct_number : ℕ := 86
noncomputable def incorrect_total_sum : ℕ := initial_average * number_of_numbers
noncomputable def correct_total_sum : ℕ := incorrect_total_sum + (correct_number - incorrect_number)
noncomputable def correct_average : ℕ := correct_total_sum / number_of_numbers

theorem correct_average_is_26 :
  correct_average = 26 := by
  sorry

end correct_average_is_26_l524_524048


namespace sum_of_coefficients_l524_524775

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ) (h₁ : (1 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 := by
  sorry

end sum_of_coefficients_l524_524775


namespace least_common_multiple_problem_l524_524467

theorem least_common_multiple_problem :
  let M := Nat.lcm 20 (Nat.lcm 21 (Nat.lcm 22 (Nat.lcm 23 (Nat.lcm 24 (Nat.lcm 25 (Nat.lcm 26 (Nat.lcm 27 (Nat.lcm 28 (Nat.lcm 29 (Nat.lcm 30 (Nat.lcm 31 (Nat.lcm 32 (Nat.lcm 33 (Nat.lcm 34 (Nat.lcm 35 (Nat.lcm 36 (Nat.lcm 37 (Nat.lcm 38 (Nat.lcm 39 40)))))))))))))))))))))
  let N := Nat.lcm M (Nat.lcm 41 (Nat.lcm 42 (Nat.lcm 43 (Nat.lcm 44 (Nat.lcm 45 (Nat.lcm 46 (Nat.lcm 47 (Nat.lcm 48 49)))))))))
  N / M = 57649 := 
by {
  sorry
}

end least_common_multiple_problem_l524_524467


namespace regular_tetrahedron_height_l524_524630

/- Define the problem constants -/
def side_length : ℝ := 15
def height_B : ℝ := 15
def height_C : ℝ := 17
def height_D : ℝ := 20

/- Given that the distance from the vertex A to the plane is represented as (r - sqrt s) / t,
where r, s, and t are positive integers. Prove that r + s + t = 930. -/
theorem regular_tetrahedron_height :
  ∃ (r s t : ℕ), r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧
  ((distance_to_plane (r, s, t, side_length, height_B, height_C, height_D)) = (r - real.sqrt s) / t) ∧
  (r + s + t) = 930 := 
sorry

/- Dummy function to represent the distance formula and relationship between r, s, t, and the tetrahedron properties -/
noncomputable def distance_to_plane 
  (params: ℕ × ℕ × ℕ × ℝ × ℝ × ℝ × ℝ) : ℝ :=
  0 -- (implementing the exact calculation isn't necessary for the problem)

#check regular_tetrahedron_height

end regular_tetrahedron_height_l524_524630


namespace no_real_intersections_l524_524351

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524351


namespace line_circle_no_intersection_l524_524374

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524374


namespace cos_expression_identity_l524_524062

noncomputable def trig_sum_to_product := 
  ∃ (a b c d : ℕ), a = 4 ∧ b = 1 ∧ c = 4 ∧ d = 7 ∧
    (∀ x : ℝ, cos x + cos (5 * x) + cos (9 * x) + cos (13 * x) = a * cos (b * x) * cos (c * x) * cos (d * x))

theorem cos_expression_identity : trig_sum_to_product :=
by
  use [4, 1, 4, 7]
  split
  repeat {split}
  all_goals { sorry }

end cos_expression_identity_l524_524062


namespace translated_B_is_B_l524_524795

def point : Type := ℤ × ℤ

def A : point := (-4, -1)
def A' : point := (-2, 2)
def B : point := (1, 1)
def B' : point := (3, 4)

def translation_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2)

def translate_point (p : point) (v : point) : point :=
  (p.1 + v.1, p.2 + v.2)

theorem translated_B_is_B' : translate_point B (translation_vector A A') = B' :=
by
  sorry

end translated_B_is_B_l524_524795


namespace sum_first_150_div_6250_remainder_l524_524104

theorem sum_first_150_div_6250_remainder : 
  let S := (150 * 151) / 2 
  in S % 6250 = 5075 := 
by
  let S := (150 * 151) / 2
  have h_cond : S = 11325 := by calc S = (150 * 151) / 2 : by sorry
  show S % 6250 = 5075 from calc S % 6250 = 11325 % 6250 : by rw h_cond
                                                 ... = 5075 : by sorry
  sorry

end sum_first_150_div_6250_remainder_l524_524104


namespace angle_CHX_l524_524638

-- Define the given conditions
def triangle_acute (A B C : Type) [Triangle A B C] :=
  ∀ x, angle A B C < 90 ∧ angle B C A < 90 ∧ angle C A B < 90

def altitudes_intersect_at_orthocenter (A B C X Y H : Type) [Triangle A B C] :=
  is_altitude A X ∧ is_altitude B Y ∧ intersects_at_orthocenter A X B Y H

def angle_BAC (A B C : Type) [Triangle A B C] :=
  angle A B C = 55

def angle_ABC (A B C : Type) [Triangle A B C] :=
  angle B C A = 80

-- state the problem of finding angle CHX
theorem angle_CHX (A B C X Y H : Type) [Triangle A B C] :
  triangle_acute A B C →
  altitudes_intersect_at_orthocenter A B C X Y H →
  angle_BAC A B C →
  angle_ABC A B C →
  angle C H X = 45 :=
by
  sorry

end angle_CHX_l524_524638


namespace B_is_guilty_l524_524124

/-
  Three defendants: A, B, and C each accuse one of the other two.
  A tells the truth, it is unknown if B tells the truth, and it is also unknown if C tells the truth.
  Who is guilty?
-/

def Defendant : Type := ℕ  -- Let's denote defendants with natural numbers for simplicity.

-- Definitions of the three defendants
def A : Defendant := 1
def B : Defendant := 2
def C : Defendant := 3

-- Axioms
axiom A_tells_truth : (A = A) → True
axiom B_uncertain : (B = B) → (True ∨ False)
axiom C_uncertain : (C = C) → (True ∨ False)

theorem B_is_guilty : ∃ (guilty : Defendant), guilty = B :=
by
  -- placeholder for the proof
  let A_accuses := B -- A truthfully accuses B
  let C_accuses := A -- Assume some accusation dynamics
  have : A_tells_truth (A = A) := by trivial
  exists B
  sorry

end B_is_guilty_l524_524124


namespace chi_squared_degrees_of_freedom_for_normal_distribution_l524_524111

theorem chi_squared_degrees_of_freedom_for_normal_distribution (s : ℕ) :
  k = s - 3 :=
by
  let r := 2
  have h : k = s - 1 - r := sorry
  rw [nat.sub_add_comm (by linarith) h]
  sorry

end chi_squared_degrees_of_freedom_for_normal_distribution_l524_524111


namespace problem_f_min_value_and_x_l524_524481

noncomputable def f (x : ℝ) : ℝ := (9 / (8 * Real.cos (2 * x) + 16)) - (Real.sin x)^2

theorem problem_f_min_value_and_x (m n : ℝ) (h1 : ∀ x, f x ≥ m) (h2 : f n = m) (h3 : 0 < n) (h4 : ∀ x, cos (2 * x) = -1 / 2 → x = n) :
  m + n = π / 3 :=
sorry

end problem_f_min_value_and_x_l524_524481


namespace algebra_expression_value_l524_524264

theorem algebra_expression_value (a b : ℝ) 
  (h₁ : a - b = 5) 
  (h₂ : a * b = -1) : 
  (2 * a + 3 * b - 2 * a * b) 
  - (a + 4 * b + a * b) 
  - (3 * a * b + 2 * b - 2 * a) = 21 := 
by
  sorry

end algebra_expression_value_l524_524264


namespace binomial_sum_trigonometric_identity_l524_524023

theorem binomial_sum_trigonometric_identity (n : ℕ) (m : ℕ) (hn : 4 * m - 3 ≤ n) :
  (finset.range (n + 1)).sum (λ k, if k % 4 = 1 then nat.choose n k else 0) =
    1 / 2 * (2 ^ (n - 1) + 2 ^ (n / 2) * real.sin (n * real.pi / 4)) :=
by sorry

end binomial_sum_trigonometric_identity_l524_524023


namespace no_intersection_l524_524369

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524369


namespace ratio_of_wire_lengths_l524_524656

theorem ratio_of_wire_lengths :
  (Bonnie_wire_pieces : ℕ) (Bonnie_wire_length_per_piece : ℕ) (Carter_unit_cube_side_length : ℕ) (Carter_wire_per_cube : ℕ)
  (Bonnie_total_wire : ℕ) (Bonnie_cube_volume : ℕ) (Carter_number_of_cubes : ℕ) (Carter_total_wire : ℕ) :
  Bonnie_wire_pieces = 12 →
  Bonnie_wire_length_per_piece = 8 →
  Carter_unit_cube_side_length = 1 →
  Carter_wire_per_cube = 24 →
  Bonnie_total_wire = Bonnie_wire_pieces * Bonnie_wire_length_per_piece →
  Bonnie_cube_volume = Bonnie_wire_length_per_piece ^ 3 →
  Carter_number_of_cubes = Bonnie_cube_volume / Carter_unit_cube_side_length ^ 3 →
  Carter_total_wire = Carter_number_of_cubes * Carter_wire_per_cube →
  Bonnie_total_wire / Carter_total_wire = 1 / 128 :=
by
  sorry

end ratio_of_wire_lengths_l524_524656


namespace smallest_d_value_l524_524625

theorem smallest_d_value :
  ∃ d : ℝ, (0 < d ∧ 
  (dist (4 * real.sqrt 5, d + 4) (0, 0) = 4 * d)) 
  ∧ d ≈ 2.81 := 
begin
  sorry
end

end smallest_d_value_l524_524625


namespace ratio_of_frames_sold_is_two_to_one_l524_524206

-- Given conditions
variable (jemma_price_per_frame : ℕ) (jemma_frames_sold : ℕ) (total_income : ℕ)
variable (dorothy_price_per_frame : ℕ)
-- Jemma sells the glass frames at 5 dollars each
axiom h1 : jemma_price_per_frame = 5
-- Jemma sold 400 frames
axiom h2 : jemma_frames_sold = 400
-- They made 2500 dollars together in total from the sale of the glass frames
axiom h3 : total_income = 2500
-- Dorothy sells her frames at half the price of Jemma's
axiom h4 : dorothy_price_per_frame = jemma_price_per_frame / 2

-- Prove the ratio of the number of frames Jemma sold to the number of frames Dorothy sold is 2:1
theorem ratio_of_frames_sold_is_two_to_one :
  let dorothy_income := total_income - (jemma_price_per_frame * jemma_frames_sold),
      dorothy_frames_sold := dorothy_income / dorothy_price_per_frame in
  jemma_frames_sold = 2 * dorothy_frames_sold :=
by 
  sorry

end ratio_of_frames_sold_is_two_to_one_l524_524206


namespace triangle_area_side_length_l524_524784

-- Definitions of the problem 
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: In triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
-- Condition: cos A = 3 / 5
def cos_A := 3 / 5

-- Condition: bc = 5
def bc := b * c = 5

-- First proof: the area of triangle ABC is 2
theorem triangle_area (hcos : cos_A = 3 / 5) (hbc : bc = 5) :
  (0.5 * b * c * (Real.sqrt (1 - cos_A^2))) = 2 :=
sorry

-- Second proof: if b + c = 6, the value of a is 2 * sqrt 5
theorem side_length (hcos : cos_A = 3 / 5) (hbc : bc = 5) (hsum : b + c = 6) :
  a = 2 * Real.sqrt 5 :=
sorry

end triangle_area_side_length_l524_524784


namespace sum_of_real_solutions_l524_524249

noncomputable def problem_statement : Prop :=
  let eq := (λ x : ℝ, sqrt x + sqrt (16 / x) + sqrt (x + 16 / x) = 8)
  in ∑ x in { x : ℝ | eq x}, x = 9

theorem sum_of_real_solutions : problem_statement :=
by { sorry }

end sum_of_real_solutions_l524_524249


namespace neil_baked_cookies_l524_524490

theorem neil_baked_cookies (total_cookies : ℕ) (given_to_friend : ℕ) (cookies_left : ℕ)
    (h1 : given_to_friend = (2 / 5) * total_cookies)
    (h2 : cookies_left = (3 / 5) * total_cookies)
    (h3 : cookies_left = 12) : total_cookies = 20 :=
by
  sorry

end neil_baked_cookies_l524_524490


namespace books_per_shelf_l524_524921

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_books : total_books = 14240) (h_shelves : total_shelves = 1780) :
    total_books / total_shelves = 8 :=
by
  rw [h_books, h_shelves]
  norm_num

end books_per_shelf_l524_524921


namespace stone_travel_distance_l524_524518

/-- Define the radii --/
def radius_fountain := 15
def radius_stone := 3

/-- Prove the distance the stone needs to travel along the fountain's edge --/
theorem stone_travel_distance :
  let circumference_fountain := 2 * Real.pi * ↑radius_fountain
  let circumference_stone := 2 * Real.pi * ↑radius_stone
  let distance_traveled := circumference_stone
  distance_traveled = 6 * Real.pi := by
  -- Placeholder for proof, based on conditions given
  sorry

end stone_travel_distance_l524_524518


namespace find_larger_box_ounces_l524_524852

-- Define the conditions
def ounces_smaller_box : ℕ := 20
def cost_smaller_box : ℝ := 3.40
def cost_larger_box : ℝ := 4.80
def best_value_price_per_ounce : ℝ := 0.16

-- Define the question and its expected answer
def expected_ounces_larger_box : ℕ := 30

-- Proof statement
theorem find_larger_box_ounces :
  (cost_larger_box / best_value_price_per_ounce = expected_ounces_larger_box) :=
by
  sorry

end find_larger_box_ounces_l524_524852


namespace competition_result_l524_524929

variables (x1 x2 x3 : ℝ)
axioms (h1 : x1 + x3 = 2 * x2) (h2 : x2 + x3 = 3 * x1)

-- Theorem statement to verify the ratio of production volumes
theorem competition_result : 
  (x1 : x2 : x3) = (3 : 4 : 5) :=
sorry

end competition_result_l524_524929


namespace negation_example_l524_524527

theorem negation_example :
  ¬(∀ x : ℝ, exp x > x^2) ↔ ∃ x : ℝ, exp x ≤ x^2 :=
by sorry

end negation_example_l524_524527


namespace equation1_solution_equation2_solution_equation3_solution_l524_524880

theorem equation1_solution :
  ∀ x : ℝ, x^2 + 4 * x = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, 2 * (x - 1) + x * (x - 1) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 4 = 0 ↔ x = (1 + Real.sqrt 13) / 3 ∨ x = (1 - Real.sqrt 13) / 3 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l524_524880


namespace population_main_task_l524_524187

theorem population_main_task (prop_0_14_decrease : Prop)
                             (prop_65_increase : Prop)
                             (total_pop_increase : Prop)
                             : (prop_0_14_decrease ∧ prop_65_increase ∧ total_pop_increase) → 
                               "The main task of population work in the new century is to continue stabilizing the low fertility level." := 
by
  intro h
  sorry

end population_main_task_l524_524187


namespace limit_problem_l524_524191

open Real
open Topology

theorem limit_problem :
  tendsto (λ (n : ℕ), ((4 * n^2 + 4 * n - 1) / (4 * n^2 + 2 * n + 3))^(1 - 2 * n)) at_top (𝓝 (exp (-1))) :=
sorry

end limit_problem_l524_524191


namespace differentiation_step_irrelevant_l524_524204

theorem differentiation_step_irrelevant (f : ℝ → ℝ) (x_0 : ℝ) :
  ¬ (f'(x_0) = (λ x, deriv f x) x_0) → (f(x_0) = f x_0) :=
sorry

end differentiation_step_irrelevant_l524_524204


namespace intersection_range_l524_524296

noncomputable def function1 (x : ℝ) : ℝ := abs (x^2 - 1) / (x - 1)
noncomputable def function2 (k x : ℝ) : ℝ := k * x - 2

theorem intersection_range (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ function1 x₁ = function2 k x₁ ∧ function1 x₂ = function2 k x₂) ↔ 
  (0 < k ∧ k < 1) ∨ (1 < k ∧ k < 4) := 
sorry

end intersection_range_l524_524296


namespace largest_spherical_scoop_radius_l524_524845

def cone_height : ℝ := 1
def cone_base_radius : ℝ := 1

theorem largest_spherical_scoop_radius : 
  ∃ r : ℝ, (r = (sqrt 5 - 1) / 2) ∧ 
           ∀ (r' : ℝ), r' < (sqrt 5 - 1) / 2 → 
                       (volume_of_scoop_inside_cone r' < volume_of_scoop_inside_cone r) :=
sorry

end largest_spherical_scoop_radius_l524_524845


namespace f_f_five_eq_five_l524_524064

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom h1 : ∀ x : ℝ, f (x + 2) = -f x
axiom h2 : f 1 = -5

-- Theorem to prove
theorem f_f_five_eq_five : f (f 5) = 5 :=
sorry

end f_f_five_eq_five_l524_524064


namespace pentagon_perimeter_l524_524103

-- Define the side lengths
def FG : ℝ := 2
def GH : ℝ := 2
def HI : ℝ := Real.sqrt 5
def IJ : ℝ := Real.sqrt 5
def JF : ℝ := 3

-- Define the perimeter
def P : ℝ := FG + GH + HI + IJ + JF

-- Define the theorem to prove
theorem pentagon_perimeter : P = 7 + 2 * Real.sqrt 5 :=
by
  -- Condition we need to prove
  have FG_def : FG = 2 := rfl
  have GH_def : GH = 2 := rfl
  have HI_def : HI = Real.sqrt 5 := rfl
  have IJ_def : IJ = Real.sqrt 5 := rfl
  have JF_def : JF = 3 := rfl
  sorry -- proof steps go here

end pentagon_perimeter_l524_524103


namespace binomial_sum_equality_l524_524026

noncomputable def binomial_sum (n m : ℕ) : ℂ :=
  ∑ k in Finset.range m, Complex.binomial n (4 * k + 1)

theorem binomial_sum_equality (n m t : ℕ) (h: 4*t - 3 ≤ n) :
  binomial_sum n (m+1) = 
      (1 / 2 : ℂ) * 
          ((2 : ℂ)^(n-1) + 
          (2 : ℂ)^(n / 2) * Complex.sin(Real.pi * n / 4)) :=
sorry

end binomial_sum_equality_l524_524026


namespace pythagorean_numbers_b_l524_524504

-- Define Pythagorean numbers and conditions
variable (a b c m : ℕ)
variable (h1 : a = 1/2 * m^2 - 1/2)
variable (h2 : c = 1/2 * m^2 + 1/2)
variable (h3 : m > 1 ∧ ¬ even m)

theorem pythagorean_numbers_b (h4 : c^2 = a^2 + b^2) : b = m :=
sorry

end pythagorean_numbers_b_l524_524504


namespace missing_coins_l524_524184

-- Definition representing the total number of coins Charlie received
variable (y : ℚ)

-- Conditions
def initial_lost_coins (y : ℚ) := (1 / 3) * y
def recovered_coins (y : ℚ) := (2 / 9) * y

-- Main Theorem
theorem missing_coins (y : ℚ) :
  y - (y * (8 / 9)) = y * (1 / 9) :=
by
  sorry

end missing_coins_l524_524184


namespace find_a_first_part_find_a_second_part_l524_524718

-- Define the circle equation.
def circle (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - 2)^2 = 4

-- Define the line equation.
def line (x y : ℝ) : Prop :=
  x - y + 3 = 0

-- Definition of the length of the chord AB.
def chord_length := 2 * Real.sqrt 2

-- Conditions for the first part of the problem.
def condition1 (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle a x y ∧ line x y

-- Prove the first part of the problem.
theorem find_a_first_part : condition1 1 → 1 = 1 :=
by
  sorry

-- Conditions for the second part of the problem.
def vector_condition (c : ℝ) : Prop :=
  ∃ x_A y_A x_B y_B x_P y_P : ℝ,
    circle c x_A y_A ∧ circle c x_B y_B ∧
    line x_A y_A ∧ line x_B y_B ∧
    circle c x_P y_P ∧
    (x_A - c) + (x_B - c) = (x_P - c) ∧
    (y_A - 2) + (y_B - 2) = (y_P - 2)

theorem find_a_second_part : vector_condition (2 * Real.sqrt 2 - 1) → 2 * Real.sqrt 2 - 1 = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end find_a_first_part_find_a_second_part_l524_524718


namespace cone_cannot_have_rectangular_cross_section_l524_524810

noncomputable def solid := Type

def is_cylinder (s : solid) : Prop := sorry
def is_cone (s : solid) : Prop := sorry
def is_rectangular_prism (s : solid) : Prop := sorry
def is_cube (s : solid) : Prop := sorry

def has_rectangular_cross_section (s : solid) : Prop := sorry

axiom cylinder_has_rectangular_cross_section (s : solid) : is_cylinder s → has_rectangular_cross_section s
axiom rectangular_prism_has_rectangular_cross_section (s : solid) : is_rectangular_prism s → has_rectangular_cross_section s
axiom cube_has_rectangular_cross_section (s : solid) : is_cube s → has_rectangular_cross_section s

theorem cone_cannot_have_rectangular_cross_section (s : solid) : is_cone s → ¬has_rectangular_cross_section s := 
sorry

end cone_cannot_have_rectangular_cross_section_l524_524810


namespace xu_jun_age_l524_524634

variable (x y : ℕ)

def condition1 : Prop := y - 2 = 3 * (x - 2)
def condition2 : Prop := y + 8 = 2 * (x + 8)

theorem xu_jun_age (h1 : condition1 x y) (h2 : condition2 x y) : x = 12 :=
by 
sorry

end xu_jun_age_l524_524634


namespace proper_fraction_reciprocal_sum_l524_524864

theorem proper_fraction_reciprocal_sum 
  (m n : ℕ) (h : m < n) :
  ∃ (l : List ℕ), (∀ x ∈ l, x > 0) ∧ (l.Nodup) ∧ 
  ((∑ k in l, (1 / (k : ℚ))) = (m / n : ℚ)) :=
sorry

end proper_fraction_reciprocal_sum_l524_524864


namespace max_distance_theorem_l524_524937

-- Definition of the condition: two unit squares with parallel sides
structure UnitSquare where
  center : ℝ × ℝ
  side_length : ℝ := 1

-- Condition: overlap area
def overlap_area (s1 s2 : UnitSquare) : ℝ := 1 / 8

-- Definition of the maximum distance between centers of two unit squares overlapping by a rectangle of area 1/8
noncomputable def max_distance_between_centers (s1 s2 : UnitSquare) (h : overlap_area s1 s2 = 1 / 8) : ℝ :=
  Real.sqrt 2 - 1 / 2

-- The theorem stating the maximum distance
theorem max_distance_theorem (s1 s2 : UnitSquare) (h : overlap_area s1 s2 = 1 / 8) :
  max_distance_between_centers s1 s2 h = Real.sqrt 2 - 1 / 2 :=
sorry

end max_distance_theorem_l524_524937


namespace line_circle_no_intersection_l524_524373

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524373


namespace line_circle_no_intersect_l524_524398

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524398


namespace find_x_of_equation_l524_524963

theorem find_x_of_equation :
  ∃ x : ℤ, 3 * x - 6 = | -20 + 5 | := 
begin
  use 7,
  sorry
end

end find_x_of_equation_l524_524963


namespace geometric_sequence_b_value_l524_524077

theorem geometric_sequence_b_value
  (b : ℝ)
  (hb_pos : b > 0)
  (hgeom : ∃ r : ℝ, 30 * r = b ∧ b * r = 3 / 8) :
  b = 7.5 := by
  sorry

end geometric_sequence_b_value_l524_524077


namespace find_P_l524_524050

variable (a b c d P : ℝ)

theorem find_P 
  (h1 : (a + b + c + d) / 4 = 8) 
  (h2 : (a + b + c + d + P) / 5 = P) : 
  P = 8 := 
by 
  sorry

end find_P_l524_524050


namespace limit_sin_over_x_l524_524020

theorem limit_sin_over_x (α : ℝ) :
  (∀ α, (0 < α ∧ α < π/2) → (cos α < sin α / α ∧ sin α / α < 1)) ∧
  (∀ α, (-π/2 < α ∧ α < 0) → (cos (-α) = cos α ∧ sin (-α) / (-α) = sin α / α)) →
  (∀ l, l = (1 : ℝ) → (∀ ε > 0, ∃ δ > 0, ∀ (α : ℝ), abs α < δ → abs  (sin α / α - l) < ε)) :=
by
  sorry

end limit_sin_over_x_l524_524020


namespace emilia_blueberries_l524_524684

def cartons_needed : Nat := 42
def cartons_strawberries : Nat := 2
def cartons_bought : Nat := 33

def cartons_blueberries (needed : Nat) (strawberries : Nat) (bought : Nat) : Nat :=
  needed - (strawberries + bought)

theorem emilia_blueberries : cartons_blueberries cartons_needed cartons_strawberries cartons_bought = 7 :=
by
  sorry

end emilia_blueberries_l524_524684


namespace empty_seats_arrangement_l524_524923

theorem empty_seats_arrangement (chairs : ℕ) (students : ℕ) : 
  chairs = 8 ∧ students = 4 →
  (∃ ways_adjacent : ℕ, ways_adjacent = 120 ∧ 
   ∃ ways_not_adjacent : ℕ, ways_not_adjacent = 120) :=
begin
  intros h,
  have h1 : ways_adjacent = 120, sorry,
  have h2 : ways_not_adjacent = 120, sorry,
  exact ⟨ways_adjacent, h1, ways_not_adjacent, h2⟩
end

end empty_seats_arrangement_l524_524923


namespace percent_prime_divisible_by_5_l524_524581

def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]
def primes_divisible_by_5 := [p in primes_less_than_20 | p % 5 = 0]

theorem percent_prime_divisible_by_5 :
  (primes_divisible_by_5.length / primes_less_than_20.length * 100 = 12.5) :=
by
  sorry

end percent_prime_divisible_by_5_l524_524581


namespace line_circle_no_intersection_l524_524338

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524338


namespace percentage_volume_taken_by_cubes_l524_524627

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

noncomputable def total_cubes_fit (l w h side : ℝ) : ℝ := 
  (l / side) * (w / side) * (h / side)

theorem percentage_volume_taken_by_cubes (l w h side : ℝ) (hl : l = 12) (hw : w = 6) (hh : h = 9) (hside : side = 3) :
  volume_of_box l w h ≠ 0 → 
  (total_cubes_fit l w h side * volume_of_cube side / volume_of_box l w h) * 100 = 100 :=
by
  intros
  rw [hl, hw, hh, hside]
  simp only [volume_of_box, volume_of_cube, total_cubes_fit]
  sorry

end percentage_volume_taken_by_cubes_l524_524627


namespace kenneth_past_finish_line_l524_524176

theorem kenneth_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (time_biff : ℕ) (distance_kenneth : ℕ) :
  race_distance = 500 → biff_speed = 50 → kenneth_speed = 51 → time_biff = race_distance / biff_speed → distance_kenneth = kenneth_speed * time_biff → 
  distance_kenneth - race_distance = 10 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end kenneth_past_finish_line_l524_524176


namespace percent_primes_less_than_20_divisible_by_5_l524_524586

theorem percent_primes_less_than_20_divisible_by_5 :
  let primes := [ 2, 3, 5, 7, 11, 13, 17, 19 ]
  let divisible_by_5 := 1
  let total_primes := 8
  (divisible_by_5:ℚ ÷ total_primes * 100) = 12.5 := by 
  sorry

end percent_primes_less_than_20_divisible_by_5_l524_524586


namespace miles_for_fare_l524_524914

-- Define the conditions
def initial_fare : ℝ := 3.0
def additional_rate_per_01mile : ℝ := 0.25
def total_spending : ℝ := 15.0
def tip : ℝ := 3.0

-- Available fare excluding the tip
def available_fare := total_spending - tip

-- Prove the total number of miles
theorem miles_for_fare : 
  ∀ (y : ℝ), initial_fare + additional_rate_per_01mile * ((y - 1) / 0.1) = available_fare → y = 4.6 :=
begin
  intros y h,
  sorry
end

end miles_for_fare_l524_524914


namespace jellybeans_initial_amount_l524_524208

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l524_524208


namespace Visited_Norway_l524_524437

-- Definitions based on conditions
def Total_People : ℕ := 50
def Visited_Iceland : ℕ := 25
def Visited_Both : ℕ := 21
def Visited_Neither : ℕ := 23

-- Proof statement for the number of people who have visited Norway
theorem Visited_Norway : ℕ :=
  let I := Visited_Iceland
  let B := Visited_Both
  let N := Visited_Neither
  let Total := Total_People
  -- Using principle of inclusion-exclusion
  N := Total - ((I + N - B) + N) in
  N sorry

end Visited_Norway_l524_524437


namespace arithmetic_sequence_max_sum_l524_524796

noncomputable def max_S_n (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum :
  ∃ d, ∃ a : ℕ → ℝ, 
  (a 1 = 1) ∧ (3 * (a 1 + 7 * d) = 5 * (a 1 + 12 * d)) ∧ 
  (∀ n, max_S_n n a d ≤ max_S_n 20 a d) := 
sorry

end arithmetic_sequence_max_sum_l524_524796


namespace minimum_tests_needed_l524_524559

def battery := Type

def is_good : battery → Prop := sorry

def num_good (bats : list battery) : ℕ := 
  list.countp is_good bats

-- A sublist of batteries where we only care in case two batteries are good
def can_operate_radio (bats : list battery) : Prop := num_good bats = 2

-- Main theorem: Minimum tests required to guarantee the radio operating with two good batteries
theorem minimum_tests_needed (bats : list battery) (h : bats.length = 8) (k : num_good bats = 4) :
  ∃ t : ℕ, t = 7 ∧ (∀ l, list.length l = t → 
    ∃ t_pairs, (∀ pair ∈ t_pairs, list.sublist _ pair ∧ can_operate_radio pair)) :=
sorry

end minimum_tests_needed_l524_524559


namespace unique_fraction_difference_l524_524254

theorem unique_fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  (1 / x) - (1 / y) = (y - x) / (x * y) :=
by sorry

end unique_fraction_difference_l524_524254


namespace no_intersection_l524_524370

-- Definitions
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem
theorem no_intersection (x y : ℝ) :
  ¬ (line x y ∧ circle x y) :=
begin
  sorry
end

end no_intersection_l524_524370


namespace one_million_mm_in_km_l524_524969

theorem one_million_mm_in_km : 
  (1 : ℕ) * 1000000 = (1 : ℕ) * 1000 * 1000 :=
by
  -- Giving given information
  have meter_to_mm : (1 : ℕ) * 1000 = 1000 := by simp
  have km_to_meter : (1 : ℕ) * 1000 = 1000 := by simp

  conv
  begin
    -- Convert 1 million millimeters to kilometers using given conversions
    to_rhs,
    congr,
    rw ←km_to_meter,  
    rw ←meter_to_mm,
  end 
  simp
  sorry

end one_million_mm_in_km_l524_524969


namespace rectangle_ratio_l524_524440

theorem rectangle_ratio (a b : ℝ) (side : ℝ) (M N : ℝ → ℝ) (P Q : ℝ → ℝ)
  (h_side : side = 4)
  (h_M : M 0 = 4 / 3 ∧ M 4 = 8 / 3)
  (h_N : N 0 = 4 / 3 ∧ N 4 = 8 / 3)
  (h_perpendicular : P 0 = Q 0 ∧ P 4 = Q 4)
  (h_area : side * side = 16) :
  let UV := 6 / 5
  let VW := 40 / 3
  UV / VW = 9 / 100 :=
sorry

end rectangle_ratio_l524_524440


namespace square_side_length_l524_524029

theorem square_side_length 
  (A B C D E : Type) 
  (AB AC hypotenuse square_side_length : ℝ) 
  (h1: AB = 9) 
  (h2: AC = 12) 
  (h3: hypotenuse = Real.sqrt (9^2 + 12^2)) 
  (h4: square_side_length = 300 / 41) 
  : square_side_length = 300 / 41 := 
by 
  sorry

end square_side_length_l524_524029


namespace Q_50_l524_524041

noncomputable def S (n : ℕ) : ℕ :=
  2 * (1 + 2 + 3 + ... + n) -- S_n

def Q (n : ℕ) : ℕ :=
  (∏ k in finset.range n, S k / (S k - 2))

theorem Q_50 : Q 50 = 51 / 2 :=
sorry

end Q_50_l524_524041


namespace collinear_points_l524_524671

open Point

noncomputable def is_collinear (A B C : Point) : Prop := ∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

noncomputable def Menelaos_theorem (ABC : Triangle) (l : Line) : Prop :=
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C in
  ∃ (X Y Z : Point), X ∈ l ∧ Y ∈ l ∧ Z ∈ l ∧
  ∃ (k1 k2 k3 : ℝ),
     k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
     dist A X = k1 * dist X B ∧
     dist B Y = k2 * dist Y C ∧
     dist C Z = k3 * dist Z A ∧
     (k1 * k2 * k3) = 1

noncomputable def Ceva_theorem (ABC : Triangle) (E : Point) : Prop :=
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C in
  ∃ (A' B' C' : Point),
    A' ∈ Line_through B C ∧
    B' ∈ Line_through A C ∧
    C' ∈ Line_through A B ∧
    dist C A' * dist A B' * dist B C' = dist A A' * dist B B' * dist C C'

theorem collinear_points (ABC : Triangle) (E : Point) :
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C in
  ∃ (A' B' C' : Point),
    A' ∈ Line_through B C ∧
    B' ∈ Line_through A C ∧
    C' ∈ Line_through A B ∧
    let A'' := Line_intersect (Line_through A B) (Line_through A' B') in
    let B'' := Line_intersect (Line_through B C) (Line_through B' C') in
    let C'' := Line_intersect (Line_through C A) (Line_through C' A') in
    Menelaos_theorem ABC (Line_through A B A'' B'') ∧
    Ceva_theorem ABC E →
    is_collinear A'' B'' C'' :=
sorry

end collinear_points_l524_524671


namespace distance_between_vertices_hyperbola_l524_524239

-- Definitions as per conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Statement of the problem in Lean
theorem distance_between_vertices_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → ∃ d : ℝ, d = 8 :=
by
  intros x y h
  use 8
  sorry

end distance_between_vertices_hyperbola_l524_524239


namespace sufficient_condition_l524_524197

theorem sufficient_condition (a : ℝ) (h : a ≥ 5) : ∀ x ∈ Icc 1 2, x^2 - a ≤ 0 :=
by
  intro x hx
  have h1 : x^2 ≤ 4 := by
    interval_cases hx with _ _
    { norm_num }
    { nlinarith }
  linarith 

#eval sufficient_condition

end sufficient_condition_l524_524197


namespace prime_solution_l524_524411

theorem prime_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 :=
by
  sorry

end prime_solution_l524_524411


namespace boys_meet_once_l524_524094

theorem boys_meet_once
  (length1 length2 : ℕ)
  (speed1 speed2 : ℕ)
  (length1_eq : length1 = 40)
  (length2_eq : length2 = 30)
  (speed1_eq : speed1 = 6)
  (speed2_eq : speed2 = 10)
  (direction : speed1 + speed2 > 0) :
  ∃ n : ℕ, n = 1 :=
by {
  rw [length1_eq, length2_eq, speed1_eq, speed2_eq] at *,
  let perimeter := 2 * (40 + 30),
  let relative_speed := 16,
  let time_to_meet_again := perimeter / relative_speed,
  let laps1_per_sec := speed1 / perimeter,
  let laps2_per_sec := speed2 / perimeter,
  let lcm_laps_per_sec := perimeter / gcd speed1 speed2,
  let meetings_per_lap := lcm_laps_per_sec / perimeter * time_to_meet_again,
  let total_meetings := meetings_per_lap,
  have meet_exactly_once := total_meetings = 1,
  exact ⟨1, meet_exactly_once⟩
}
sorry

end boys_meet_once_l524_524094


namespace osaka_university_exam_1996_l524_524252

noncomputable def integral_function (m : ℕ) (x : ℝ) : ℝ :=
  x + 1 - Real.sqrt (x^2 + 2 * x * Real.cos (2 * Real.pi / (2 * m + 1)) + 1)

theorem osaka_university_exam_1996 (m : ℕ) (hm : 0 < m) :
    0 ≤ ∫ x in 0..1, integral_function m x ∧
    (∫ x in 0..1, integral_function m x) ≤ 1 :=
  sorry

end osaka_university_exam_1996_l524_524252


namespace initial_stops_eq_l524_524135

-- Define the total number of stops S
def total_stops : ℕ := 7

-- Define the number of stops made after the initial deliveries
def additional_stops : ℕ := 4

-- Define the number of initial stops as a proof problem
theorem initial_stops_eq : total_stops - additional_stops = 3 :=
by
sorry

end initial_stops_eq_l524_524135


namespace min_value_f_l524_524248

noncomputable def f (x : ℝ) : ℝ :=
  7 * (Real.sin x)^2 + 5 * (Real.cos x)^2 + 2 * Real.sin x

theorem min_value_f : ∃ x : ℝ, f x = 4.5 :=
  sorry

end min_value_f_l524_524248


namespace text_message_cost_eq_l524_524972

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end text_message_cost_eq_l524_524972


namespace least_three_digit_with_3_5_7_l524_524947

def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def least_three_digit_number_with_factors (a b c : ℕ) : ℕ :=
  let lcm := (Nat.lcm (Nat.lcm a b) c)
  ∃ n, is_three_digit_integer n ∧ is_divisible_by n lcm ∧ (∀ m, is_three_digit_integer m ∧ is_divisible_by m lcm → n ≤ m)

theorem least_three_digit_with_3_5_7 : least_three_digit_number_with_factors 3 5 7 = 105 :=
by
  sorry

end least_three_digit_with_3_5_7_l524_524947


namespace percentage_of_primes_less_than_20_divisible_by_5_l524_524571

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 := {n : ℕ | n < 20 ∧ is_prime n}

def primes_less_than_20_divisible_by_5 := {n ∈ primes_less_than_20 | 5 ∣ n}

theorem percentage_of_primes_less_than_20_divisible_by_5 : 
  (primes_less_than_20_divisible_by_5.to_finset.card : ℝ) / (primes_less_than_20.to_finset.card : ℝ) * 100 = 12.5 :=
begin
  -- Proving this statement directly would involve showing the calculations explicitly.
  -- However, we just set up the framework here.
  sorry
end

end percentage_of_primes_less_than_20_divisible_by_5_l524_524571


namespace sequences_are_correct_and_k_range_l524_524911

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := a.range(n+1).sum

noncomputable def a_n (n : ℕ) : ℝ := 3^(n-1)

def b_n (n : ℕ) : ℝ := 3*n - 6

theorem sequences_are_correct_and_k_range (k : ℝ) :
  (∀ n : ℕ, (n > 0) → (Sn a_n n + 1/2) * k ≥ b_n n) → k ≥ 2/9 :=
sorry

end sequences_are_correct_and_k_range_l524_524911


namespace exist_common_point_other_than_A_l524_524463

theorem exist_common_point_other_than_A 
  (A B C D E : Point)
  (O₁ O₂ : Point)
  (hABC_acute : acute_triangle A B C)
  (hscalene : ¬(scalene_triangle A B C))
  (hD : D ∈ line_segment A B)
  (hE : E ∈ line_segment A C)
  (hBD_CE : dist B D = dist C E)
  (hO₁_circumcenter : is_circumcenter_of O₁ (triangle A B E))
  (hO₂_circumcenter : is_circumcenter_of O₂ (triangle A C D)) :
  ∃ P ≠ A, is_point_on_circumcircle P (triangle A B C) ∧ is_point_on_circumcircle P (triangle A D E) ∧ is_point_on_circumcircle P (triangle A O₁ O₂) :=
sorry

end exist_common_point_other_than_A_l524_524463


namespace midpoints_collinear_l524_524837

variables {E : Ellipse}
variables {r1 r2 r3 : Line}

-- Assume r1, r2, r3 are parallel
axiom parallel_lines : Parallel r1 r2 ∧ Parallel r2 r3

-- Assume r1, r2, r3 intersect ellipse E at points A1, B1, A2, B2, A3, B3 respectively
variables (A1 B1 A2 B2 A3 B3 : Point)

axiom intersections : 
  (A1, B1) ∈ (E ∩ r1) ∧ 
  (A2, B2) ∈ (E ∩ r2) ∧ 
  (A3, B3) ∈ (E ∩ r3)

-- Define midpoints of the segments (Ai, Bi)
def midpoint (P Q : Point) : Point := 
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

variable m1 := midpoint A1 B1
variable m2 := midpoint A2 B2
variable m3 := midpoint A3 B3

-- Statement to prove: the midpoints m1, m2, m3 are collinear
theorem midpoints_collinear : collinear {m1, m2, m3} :=
sorry

end midpoints_collinear_l524_524837


namespace min_value_x_plus_y_l524_524717

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 2) : x + y = 8 :=
sorry

end min_value_x_plus_y_l524_524717


namespace angles_around_point_sum_l524_524452

theorem angles_around_point_sum 
  (x y : ℝ)
  (h1 : 130 + x + y = 360)
  (h2 : y = x + 30) :
  x = 100 ∧ y = 130 :=
by
  sorry

end angles_around_point_sum_l524_524452


namespace line_circle_no_intersect_l524_524390

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524390


namespace part_I_part_II_l524_524299

section

-- Define the function f
def f (x a : ℝ) : ℝ := (2 * x - 4) * Real.exp x + a * (x + 2) ^ 2

-- Define the derivative f'
def f_prime (x a : ℝ) : ℝ := (2 * x - 2) * Real.exp x + 2 * a * (x + 2)

-- Condition 1: Prove that if f is monotonically increasing, then a >= 1/2
theorem part_I (a : ℝ) (x : ℝ) (h : x > 0) (h_monotonic : ∀ x > 0, f_prime x a ≥ 0) : a ≥ 1 / 2 := by
  sorry

-- Condition 2: Prove that if a is in the interval (0, 1/2), f has a minimum value in the range (-2e, -2)
theorem part_II (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1 / 2) : ∃ t > 0, f_prime t a = 0 ∧
  (∀ x ∈ (0, t), f_prime x a < 0) ∧ 
  (∀ x ∈ (t, +∞), f_prime x a > 0) ∧
  -2 * Real.exp 1 < f t a ∧ f t a < -2 := by
  sorry

end

end part_I_part_II_l524_524299


namespace apple_crisps_calculation_l524_524030

theorem apple_crisps_calculation (apples crisps : ℕ) (h : crisps = 3 ∧ apples = 12) : 
  (36 / apples) * crisps = 9 := by
  sorry

end apple_crisps_calculation_l524_524030


namespace find_k_l524_524916

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end find_k_l524_524916


namespace product_fraction_simplification_l524_524659

theorem product_fraction_simplification : 
  2 * ∏ (i : ℕ) in (Finset.range 99).map (Finset.add 2), (1 - (1 : ℚ)/i) = 1/50 := 
sorry

end product_fraction_simplification_l524_524659


namespace hypotenuse_length_l524_524904

-- Definitions derived from conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Proposed theorem
theorem hypotenuse_length (a c : ℝ) 
  (h1 : is_isosceles_right_triangle a a c) 
  (h2 : perimeter a a c = 8 + 8 * Real.sqrt 2) :
  c = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l524_524904


namespace radius_of_smaller_circle_l524_524608

/-
We are given:
1. A $30^\circ$-$60^\circ$-$90^\circ$ triangle $PQR$ with hypotenuse $PQ = 2$.
2. A circle with center $S$ is tangent to hypoetenuse $PQ$, the x-axis, and y-axis.
3. The center $S$ lies in the first quadrant.
4. A larger circle is tangent to the y-axis and shares the tangent point on $PQ$ with the smaller circle.

We need to prove:
The radius of the smaller circle is approximately 0.366.
-/

theorem radius_of_smaller_circle
  (P Q R : Point)
  (S : Point)
  (circle1 circle2 : Circle)
  (hypotenuse_PQ : length P Q = 2)
  (angle_PQR : Angle P Q R = 60)
  (tangent_S_PQ : isTangent S.circle PQ)
  (tangent_S_xaxis : isTangent S.circle xAxis)
  (tangent_S_yaxis : isTangent S.circle yAxis)
  (center_S_first_quadrant : S.x > 0 ∧ S.y > 0)
  (large_circle_tangent_yaxis : isTangent circle2 yAxis)
  (shared_tangent : sharedTangent circle1 circle2 PQ)
  : radius circle1 ≈ 0.366 := 
sorry

end radius_of_smaller_circle_l524_524608


namespace quadratic_monotonic_decreasing_l524_524895

theorem quadratic_monotonic_decreasing (a : ℝ) :
  (∀ x, x ≤ 4 → (deriv (λ x : ℝ, x^2 + 2 * (a - 1) * x + 2) x) ≤ 0) → a ≤ -3 :=
by
  -- proof goes here
  sorry

end quadratic_monotonic_decreasing_l524_524895


namespace cell_phone_plan_cost_equality_l524_524971

theorem cell_phone_plan_cost_equality (x : ℝ) :
  let cost_A := 0.25 * x + 9
  let cost_B := 0.40 * x
  cost_A = cost_B → x = 60 := 
by
  intros cost_A_eq_cost_B
  have : 0.25 * x + 9 = 0.40 * x := cost_A_eq_cost_B
  sorry

end cell_phone_plan_cost_equality_l524_524971


namespace math_problem_l524_524752

theorem math_problem (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.injective a) :
  (∑ i, (a i)^7) + (∑ i, (a i)^5) ≥ 2 * (∑ i, (a i)^3)^2 := by
  sorry

end math_problem_l524_524752


namespace initial_number_is_11_l524_524697

theorem initial_number_is_11 :
  ∃ (N : ℤ), ∃ (k : ℤ), N - 11 = 17 * k ∧ N = 11 :=
by
  sorry

end initial_number_is_11_l524_524697


namespace red_pencils_in_box_l524_524087

theorem red_pencils_in_box (B R G : ℕ) 
  (h1 : B + R + G = 20)
  (h2 : B = 6 * G)
  (h3 : R < B) : R = 6 := by
  sorry

end red_pencils_in_box_l524_524087


namespace fractional_equation_solution_l524_524259

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (3 / (x + 1) = 2 / (x - 1)) → (x = 5) :=
sorry

end fractional_equation_solution_l524_524259


namespace hyperbola_vertex_distance_l524_524232

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l524_524232


namespace seats_in_fifteenth_row_l524_524785

open Nat

theorem seats_in_fifteenth_row : 
  ∀ (a_1 d : ℕ), 
  (a_1 = 5) → 
  (d = 2) → 
  (a_1 + (14 * d) = 33) := 
  by
    intros a_1 d h_a1 h_d
    rw [h_a1, h_d]
    simp
    sorry

end seats_in_fifteenth_row_l524_524785


namespace necessary_condition_for_x_gt_5_l524_524148

theorem necessary_condition_for_x_gt_5 (x : ℝ) : x > 5 → x > 3 :=
by
  intros h
  exact lt_trans (show 3 < 5 from by linarith) h

end necessary_condition_for_x_gt_5_l524_524148


namespace infinite_solutions_implies_a_eq_2_l524_524589

theorem infinite_solutions_implies_a_eq_2 (a b : ℝ) (h : b = 1) :
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → a = 2 :=
by
  intro H
  sorry

end infinite_solutions_implies_a_eq_2_l524_524589


namespace tan_beta_value_l524_524289

theorem tan_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 4 / 3)
  (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2 / 11 := 
sorry

end tan_beta_value_l524_524289


namespace line_circle_no_intersect_l524_524393

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l524_524393


namespace roses_carnations_price_comparison_l524_524458

variables (x y : ℝ)

theorem roses_carnations_price_comparison
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y :=
sorry

end roses_carnations_price_comparison_l524_524458


namespace geometric_seq_ratio_l524_524054

theorem geometric_seq_ratio (a : ℝ) : 
  let r₂ := (a + real.log 3) / log 2,
      r₄ := (a + real.log 3) / (2 * log 2),
      r₈ := (a + real.log 3) / (3 * log 2)
  in (a + r₂) / (a + r₄) = (a + r₄) / (a + r₈) → (a + r₄) / (a + r₂) = 1 / 3 :=
begin
  sorry
end

end geometric_seq_ratio_l524_524054


namespace distance_from_origin_to_line_l524_524519

theorem distance_from_origin_to_line :
  let line := (4 : ℝ) * x + (3 : ℝ) * y - 15 = 0,
  let origin := (0 : ℝ, 0 : ℝ),
  ∀ x y : ℝ,
  (line ⟹ (sqrt ((4 * 0 + 3 * 0 - 15)^2) / sqrt (4^2 + 3^2)) = 3) :=
by
  sorry

end distance_from_origin_to_line_l524_524519


namespace no_real_intersections_l524_524346

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524346


namespace part2_l524_524725

noncomputable def arithmetic_seq_sum (a1 d : ℤ) (n : ℤ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

def b (n : ℕ) : ℕ := 2 * n - 1

def c (n : ℕ) : ℚ :=
  (-1) ^ n * (2 * n + 1) / (((2 * (n + 1) - 1) + 1) * ((2 * n - 1) + 1))

def T (n : ℕ) : ℚ :=
  ∑ k in Finset.range (2 * n + 1), c k

theorem part2 (t : ℕ) (h₁ : 0 < t) (h₂ : ∀ (m1 m2 : ℕ), m1 < m2 ∧ arithmetic_seq_sum a1 d m1 = arithmetic_seq_sum a1 d m2 → m2 - m1 = b n) :
  T t < -1 / 6 :=
sorry

end part2_l524_524725


namespace find_angle_A_l524_524933

theorem find_angle_A (A B C D : Type*) [EuclideanGeometry A B C D] 
  (h1 : ¬∃ (angle : ℝ), angle ∈ TriangleAngles A B C ∧ angle > 90)
  (h2 : OnLine D A C ∧ distance A D = (3 / 4) * distance A C)
  (h3 : SimilarTriangle ΔABC ΔABD ∧ SimilarTriangle ΔABC ΔCBD) : 
  angle A = 30 :=
by
  sorry

end find_angle_A_l524_524933


namespace sum_sequence_l524_524721

def f (x : ℝ) (m a: ℝ) : ℝ := x^m + a * x

theorem sum_sequence (m a : ℝ) (h_deriv : ∀ x : ℝ, deriv (f x m a) = 2 * x + 1) (n : ℕ) (h_n_pos : 0 < n) : 
  ∑ i in Finset.range n, (f i m a / (i * 2^i)) = 3 - (n + 3) / 2^n := 
sorry

end sum_sequence_l524_524721


namespace line_circle_no_intersection_l524_524354

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524354


namespace geometric_seq_ratio_l524_524055

theorem geometric_seq_ratio (a : ℝ) : 
  let r₂ := (a + real.log 3) / log 2,
      r₄ := (a + real.log 3) / (2 * log 2),
      r₈ := (a + real.log 3) / (3 * log 2)
  in (a + r₂) / (a + r₄) = (a + r₄) / (a + r₈) → (a + r₄) / (a + r₂) = 1 / 3 :=
begin
  sorry
end

end geometric_seq_ratio_l524_524055


namespace min_value_is_9_over_2_l524_524734

noncomputable def min_value_fracs (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1/3)^(x-2) = 3^y) : ℝ :=
  4/x + 1/y

theorem min_value_is_9_over_2 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (1/3)^(x-2) = 3^y) :
  min_value_fracs x y h1 h2 h3 = 9/2 :=
sorry

end min_value_is_9_over_2_l524_524734


namespace vectors_are_collinear_l524_524468

noncomputable def vectors_collinear 
  (a b c : V) [AddCommMonoid V] [Module ℝ V] [MetricSpace V] [NormedAddCommGroup V]
  [NormedSpace ℝ V] 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0) 
  (h_c_nonzero : c ≠ 0)
  (h_a_parallel_b : ∃ k : ℝ, a = k • b)
  (h_a_parallel_c : ∃ k : ℝ, a = k • c) : Prop := 
  ∃ k : ℝ, b = k • c

theorem vectors_are_collinear 
  (a b c : V) [AddCommMonoid V] [Module ℝ V] [MetricSpace V] [NormedAddCommGroup V]
  [NormedSpace ℝ V] 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0) 
  (h_c_nonzero : c ≠ 0)
  (h_a_parallel_b : ∃ k : ℝ, a = k • b)
  (h_a_parallel_c : ∃ k : ℝ, a = k • c) : 
  vectors_collinear a b c h_a_nonzero h_b_nonzero h_c_nonzero h_a_parallel_b h_a_parallel_c :=
sorry

end vectors_are_collinear_l524_524468


namespace find_alpha_polar_eq_l524_524715

noncomputable def point := {x : ℝ, y : ℝ}

def line_parametric (α : ℝ) (t : ℝ) : point :=
  {x := 2 + t * Real.cos α, y := 1 + t * Real.sin α}

def condition_PA_PB (α : ℝ) (t1 t2 : ℝ) :=
  let A := line_parametric α t1
  let B := line_parametric α t2
  let PA := Real.sqrt ((A.x - 2)^2 + (A.y - 1)^2)
  let PB := Real.sqrt ((B.x - 2)^2 + (B.y - 1)^2)
  PA * PB = 4

theorem find_alpha (α : ℝ) (t1 t2 : ℝ) (H : condition_PA_PB α t1 t2) :
  α = 3 * Real.pi / 4 :=
sorry

def polar_equation (α : ℝ) (ρ θ : ℝ) :=
  ρ * (Real.cos θ + Real.sin θ) = 3

theorem polar_eq (θ : ℝ) :
  polar_equation (3 * Real.pi / 4) (3 / (Real.cos θ + Real.sin θ)) θ :=
sorry

end find_alpha_polar_eq_l524_524715


namespace circle_area_below_axis_left_of_line_l524_524560

variable (x y : ℝ)

def circle_equation := (x - 7) ^ 2 + y ^ 2 = 114
def line_equation := y = x - 4

theorem circle_area_below_axis_left_of_line : 
  ∃ A : ℝ, A = 14.25 * real.pi ∧ 
  ∀ (x y : ℝ), circle_equation x y ∧ y < 0 ∧ y < x - 4 → 
  A = 14.25 * real.pi := 
sorry

end circle_area_below_axis_left_of_line_l524_524560


namespace minimize_side_c_l524_524746

variables (t : ℝ) (C : ℝ)

theorem minimize_side_c (a b c : ℝ) (h_area : (1/2) * a * b * sin C = t) (h_eq : a = b) :
  c = 2 * sqrt (t * tan (C / 2)) := 
sorry

end minimize_side_c_l524_524746


namespace find_b_when_a_is_negative12_l524_524902

theorem find_b_when_a_is_negative12 (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∃ k, a * b = k) : b = -56.25 :=
sorry

end find_b_when_a_is_negative12_l524_524902


namespace train_speed_is_60_kmph_l524_524158

noncomputable def length_of_train : ℝ := 110 -- meters
noncomputable def passing_time : ℝ := 6 -- seconds
noncomputable def speed_of_man_kmph : ℝ := 6 -- kmph

noncomputable def speed_of_man_mps : ℝ := (speed_of_man_kmph * 1000) / 3600 -- convert kmph to mps

theorem train_speed_is_60_kmph :
  let relative_speed := length_of_train / passing_time in
  let speed_of_train_mps := relative_speed + speed_of_man_mps in
  let speed_of_train_kmph := (speed_of_train_mps * 3600) / 1000 in
  speed_of_train_kmph = 60 :=
by
  sorry

end train_speed_is_60_kmph_l524_524158


namespace distinct_license_plates_count_l524_524144

noncomputable def num_distinct_license_plates : ℕ :=
  let digits := 10 ^ 5
  let letters := 26 ^ 3
  let positions := 6
  in positions * digits * letters

theorem distinct_license_plates_count :
  num_distinct_license_plates = 10_584_576_000 :=
by 
  sorry

end distinct_license_plates_count_l524_524144


namespace ferry_boat_problem_l524_524620

theorem ferry_boat_problem : 
  ∃ (n a d S : ℕ), 
  n = 15 ∧ 
  a = 120 ∧ 
  d = -2 ∧ 
  S = 1590 ∧ 
  S = (n * (2 * a + (n - 1) * d)) / 2 :=
begin
  sorry
end

end ferry_boat_problem_l524_524620


namespace line_circle_no_intersection_l524_524383

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524383


namespace sum_of_digits_f_3_to_300_l524_524830

noncomputable def g_k (k x : ℕ) : ℕ := (x^3 / 10^(2 * k)).floor

def x_n (k : ℕ) : ℕ := 
  let n := Nat.find (λ a, g_k k a - g_k k (a-1) ≥ 5) sorry
  n

def f (k : ℕ) : ℕ := g_k k (x_n k - 1) + 1

theorem sum_of_digits_f_3_to_300 : digit_sum (∑ i in (Finset.range 100).filter (λ n, (n+1) % 3 = 0), f ((n+1) * 3)) = S :=
sorry

end sum_of_digits_f_3_to_300_l524_524830


namespace integral_is_two_thirds_l524_524665

noncomputable def integral_proof : Prop :=
  ∫ x in -1..1, (x^2 + sin x) = 2 / 3

theorem integral_is_two_thirds : integral_proof := 
by
  sorry

end integral_is_two_thirds_l524_524665


namespace angle_between_bisectors_is_90_degrees_l524_524317

theorem angle_between_bisectors_is_90_degrees
  (α β : ℝ)
  (A B C : Type)
  [linear_ordered_field ℝ]
  [metric_space ℝ]
  [normed_group A]
  [normed_group B]
  [normed_group C]
  [normed_space ℝ A]
  [normed_space ℝ B]
  [normed_space ℝ C]
  (triangle_ABC : triangle ℝ A B C)
  (O : incenter triangle_ABC)
  (O1 O2 O3 : ℝ)
  (O1O2_bisector O2O3_bisector O1O3_bisector: is_external_bisector triangle_ABC O1 O2 O3)
  : angle O1 O2 O O3 = 90 := 
begin
  sorry
end

end angle_between_bisectors_is_90_degrees_l524_524317


namespace class1_draws_two_multiple_choice_questions_expected_value_of_X_l524_524614

-- Problem 1 in Lean 4 statement
theorem class1_draws_two_multiple_choice_questions (pB1 pB2 pB3 : ℚ)
    (pA_given_B1 pA_given_B2 pA_given_B3 pA : ℚ)
    (h1 : pB1 = 5 / 14) (h2 : pB2 = 15 / 28) (h3 : pB3 = 3 / 28)
    (h4 : pA_given_B1 = 6 / 9) (h5 : pA_given_B2 = 5 / 9) (h6 : pB3 = 4 / 9)
    (h7 : pA = (pB1 * 6 / 9) + (pB2 * 5 / 9) + (pB3 * 4 / 9)) :
  (pB1 * pA_given_B1 / pA) = 20 / 49 := by
  sorry

-- Problem 2 in Lean 4 statement
theorem expected_value_of_X (p3 p4 p5 eX : ℚ)
    (h1 : p3 = 4 / 25) (h2 : p4 = 48 / 125) (h3 : p5 = 57 / 125)
    (h4 : eX = 3 * p3 + 4 * p4 + 5 * p5) :
  eX = 537 / 125 := by
  sorry

end class1_draws_two_multiple_choice_questions_expected_value_of_X_l524_524614


namespace range_of_a_for_monotonic_increasing_f_l524_524275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 2 * Real.log x

theorem range_of_a_for_monotonic_increasing_f (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → (x - a - 2 / x) ≥ 0) : a ≤ -1 :=
by {
  -- Placeholder for the detailed proof steps
  sorry
}

end range_of_a_for_monotonic_increasing_f_l524_524275


namespace fourth_guard_run_distance_l524_524080

-- Define the rectangle's dimensions
def length : ℝ := 300
def width : ℝ := 200

-- Define the perimeter of the rectangle
def perimeter : ℝ := 2 * (length + width)

-- Given the sum of the distances run by three guards
def sum_of_three_guards : ℝ := 850

-- The fourth guard's distance is what we need to prove
def fourth_guard_distance := perimeter - sum_of_three_guards

-- The proof goal: we need to show that the fourth guard's distance is 150 meters
theorem fourth_guard_run_distance : fourth_guard_distance = 150 := by
  sorry  -- This placeholder means that the proof is omitted

end fourth_guard_run_distance_l524_524080


namespace intersection_M_N_l524_524424

-- Define the sets M and N according to the given conditions
def M := {x : ℝ | -3 < x ∧ x < 1}
def N := {x : ℤ | -1 ≤ x ∧ x ≤ 2}

-- Prove the intersection of M and N is {-1, 0}
theorem intersection_M_N : {x : ℝ | x ∈ M ∧ x ∈ N} = ({-1, 0} : Set ℝ) :=
by
  sorry

end intersection_M_N_l524_524424


namespace value_of_n_div_p_l524_524534

noncomputable def quad_roots_related (k n p : ℝ) (k_ne_zero : k ≠ 0) (n_ne_zero : n ≠ 0) (p_ne_zero : p ≠ 0) : Prop :=
  ∃ s₁ s₂ : ℝ, s₁ + s₂ = -p ∧ s₁ * s₂ = k ∧ 3 * (s₁ + s₂) = -k ∧ 9 * s₁ * s₂ = n

theorem value_of_n_div_p (k n p : ℝ) (h : quad_roots_related k n p (by linarith) (by linarith) (by linarith)) : n / p = 27 :=
by sorry

end value_of_n_div_p_l524_524534


namespace count_distinct_products_l524_524002

/-- Let p1, p2, ..., pk be distinct prime numbers. 
    The number of distinct products p1^α1 * p2^α2 * ... * pk^αk satisfying αi ∈ ℕ and 
    p1 * p2 * ... * pk = α1 * α2 * ... * αk is k^k. -/
theorem count_distinct_products (p : ℕ -> ℕ) (α : ℕ -> ℕ) (k : ℕ) 
    (h_k_gt_0 : k > 0)
    (h_distinct_primes : ∀ i j, i ≠ j → p i ≠ p j)
    (h_prime : ∀ i, nat.prime (p i)) 
    (h_product_eq : ∏ i in finset.range k, p i = ∏ i in finset.range k, α i) :
    (finset.card {x : ℕ // ∃ (α : ℕ -> ℕ), (∀ i, α i ∈ ℕ) ∧ (∏ i in finset.range k, p i = ∏ i in finset.range k, α i) ∧ (x = ∏ i in finset.range k, (p i) ^ (α i))} = k^k) := 
sorry

end count_distinct_products_l524_524002


namespace angle_of_sum_cis_sequence_l524_524181

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (seq : List ℂ) (start : ℂ) (difference : ℂ) (n : ℕ) : Prop :=
  seq = List.range n |>.map (λ i, start * Complex.exp (⟨0, (i * difference).toReal⟩ * Complex.I))

-- Define the sum of cis from 60° to 160° with 10° difference
noncomputable def sum_cis_sequence : ℂ :=
  (List.range 11 |>.map (λ i, Complex.exp (⟨0, (60 + i * 10).toReal⟩ * Complex.I))).sum

-- State the problem: prove that sum_cis_sequence can be written in form r * cis 110°
theorem angle_of_sum_cis_sequence : ∃ r : ℂ, sum_cis_sequence = r * Complex.exp (⟨0, 110.toReal⟩ * Complex.I) := by
  sorry

end angle_of_sum_cis_sequence_l524_524181


namespace line_circle_no_intersection_l524_524404

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524404


namespace books_per_shelf_l524_524920

theorem books_per_shelf (total_books : ℕ) (total_shelves : ℕ) (h_books : total_books = 14240) (h_shelves : total_shelves = 1780) :
    total_books / total_shelves = 8 :=
by
  rw [h_books, h_shelves]
  norm_num

end books_per_shelf_l524_524920


namespace clock_hands_30_degrees_l524_524787

theorem clock_hands_30_degrees (single_day : Prop) 
                               (daylight_saving : Prop) 
                               (leap_year : Prop) : 
  ∃ times : ℕ, times = 48 :=
begin
  use 48,
  sorry
end

end clock_hands_30_degrees_l524_524787


namespace roots_sum_of_squares_l524_524190

noncomputable def cubic_eq_roots (p q r : ℝ) : Prop :=
  polynomial.root_of (3*x^3 + 2*x^2 - 3*x - 8) p ∧
  polynomial.root_of (3*x^3 + 2*x^2 - 3*x - 8) q ∧
  polynomial.root_of (3*x^3 + 2*x^2 - 3*x - 8) r

theorem roots_sum_of_squares (p q r : ℝ) (h : cubic_eq_roots p q r) :
  p^2 + q^2 + r^2 = -14/9 :=
sorry

end roots_sum_of_squares_l524_524190


namespace find_other_number_l524_524997

theorem find_other_number
  (a b : ℕ)  -- Define the numbers as natural numbers
  (h1 : a = 300)             -- Condition stating the certain number is 300
  (h2 : a = 150 * b)         -- Condition stating the ratio is 150:1
  : b = 2 :=                 -- Goal stating the other number should be 2
  by
    sorry                    -- Placeholder for the proof steps

end find_other_number_l524_524997


namespace K6_edge_coloring_exists_l524_524088

open SimpleGraph

theorem K6_edge_coloring_exists :
  ∃ (f : sym2 (Fin 6) → Fin 5), (∀ (v : Fin 6), (∃ (edges : Finset (sym2 (Fin 6))), edges.card = 5 ∧ ∀ e1 e2 ∈ edges, e1 ≠ e2 ∧ (∃ c1 c2, f e1 = c1 ∧ f e2 = c2 ∧ c1 ≠ c2))) :=
sorry

end K6_edge_coloring_exists_l524_524088


namespace continuity_at_1_jump_discontinuity_at_neg1_l524_524456

def f (x : ℝ) : ℝ :=
if |x| ≤ 1 then x^3 else 1

theorem continuity_at_1 :
  continuous_at f 1 :=
sorry

theorem jump_discontinuity_at_neg1 :
  ¬ (continuous_at f (-1)) ∧
  (∃ L₁ L₂ : ℝ, (L₁ ≠ L₂) ∧ 
    (filter.tendsto f (nhds_within (-1) (set.Iio (-1))) (nhds L₁)) ∧ 
    (filter.tendsto f (nhds_within (-1) (set.Ici (-1))) (nhds L₂))) :=
sorry

end continuity_at_1_jump_discontinuity_at_neg1_l524_524456


namespace probability_of_B_not_losing_is_70_l524_524994

-- Define the probabilities as given in the conditions
def prob_A_winning : ℝ := 0.30
def prob_draw : ℝ := 0.50

-- Define the probability of B not losing
def prob_B_not_losing : ℝ := 0.50 + (1 - prob_A_winning - prob_draw)

-- State the theorem
theorem probability_of_B_not_losing_is_70 :
  prob_B_not_losing = 0.70 := by
  sorry -- Proof to be filled in

end probability_of_B_not_losing_is_70_l524_524994


namespace remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l524_524496

-- Definition of initial conditions
def initial_sweet_cookies := 34
def initial_salty_cookies := 97
def initial_chocolate_cookies := 45

def sweet_cookies_eaten := 15
def salty_cookies_eaten := 56
def chocolate_cookies_given_away := 22
def chocolate_cookies_given_back := 7

-- Calculate remaining cookies
def remaining_sweet_cookies : Nat := initial_sweet_cookies - sweet_cookies_eaten
def remaining_salty_cookies : Nat := initial_salty_cookies - salty_cookies_eaten
def remaining_chocolate_cookies : Nat := (initial_chocolate_cookies - chocolate_cookies_given_away) + chocolate_cookies_given_back

-- Theorem statements
theorem remaining_sweet_cookies_correct : remaining_sweet_cookies = 19 := 
by sorry

theorem remaining_salty_cookies_correct : remaining_salty_cookies = 41 := 
by sorry

theorem remaining_chocolate_cookies_correct : remaining_chocolate_cookies = 30 := 
by sorry

end remaining_sweet_cookies_correct_remaining_salty_cookies_correct_remaining_chocolate_cookies_correct_l524_524496


namespace line_circle_no_intersection_l524_524341

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524341


namespace problem_statement_l524_524549

structure Triangle (α : Type _) :=
(P Q R : α)
(PQ QR RP : ℝ)
(PQ_pos : 0 < PQ)
(QR_pos : 0 < QR)
(RP_pos : 0 < RP)

noncomputable def circumradius {α : Type _} [EuclideanGeometry α] (T : Triangle α) : ℝ :=
-- definition omitted for brevity
sorry

noncomputable def length_PS {α : Type _} [EuclideanGeometry α] (T : Triangle α) : ℝ :=
-- definition omitted for brevity
sorry

theorem problem_statement : ∃ (a b : ℕ), 0 < b ∧ (¬ ∃ p, prime p ∧ p^2 ∣ b) ∧ (length_PS ⟨(0 : ℝ), 0, 0, 40, 9, 41, by linarith, by linarith, by linarith⟩ = a * real.sqrt b) ∧ (⌊a + real.sqrt b⌋ = 37) :=
begin
  sorry
end

end problem_statement_l524_524549


namespace largest_invertible_interval_l524_524875

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 4

def domain_includes_neg1 (d : Set ℝ) : Prop :=
  -1 ∈ d

def g_invertible_on (d : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ d → y ∈ d → g x = g y → x = y

theorem largest_invertible_interval :
  ∃ (d : Set ℝ), domain_includes_neg1 d ∧ g_invertible_on d ∧ 
  ∀ (d' : Set ℝ), domain_includes_neg1 d' → g_invertible_on d' → d' ⊆ d :=
begin
  use {x : ℝ | x ≤ -1},
  split,
  { exact set.mem_set_of_eq.2 (le_refl (-1)) },
  split,
  { assume x y hx hy hxy,
    have := congr_arg (λ x, x + 7) hxy,
    simp [g, add_assoc, add_comm, add_left_comm] at this,
    rw [←sub_eq_zero, ←sub_sub, ←sqrt_eq_zero, ←mul_eq_zero] at this,
    rcases this,
    iterate 2 {simp [*] },
    exact this},
  {  -- Prove that this is the largest interval that includes -1 where g(x) is invertible
    -- Unfortunately this statement needs the complete formal proof involving properties of parabola and complete the square
    sorry } -- skip complete proof steps
end

end largest_invertible_interval_l524_524875


namespace line_circle_no_intersection_l524_524344

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524344


namespace probability_of_drawing_one_white_two_red_l524_524545

-- Definitions based on the conditions
def num_white_balls := 4
def num_red_balls := 5
def total_balls := num_white_balls + num_red_balls
def balls_drawn := 3

-- Definitions based on combinatorial calculations
def comb (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

def total_ways_to_draw := comb total_balls balls_drawn
def ways_to_draw_one_white := comb num_white_balls 1
def ways_to_draw_two_red := comb num_red_balls 2

def ways_to_draw_desired := ways_to_draw_one_white * ways_to_draw_two_red

-- Correct answer based on calculated probability
def desired_probability :=
  (ways_to_draw_desired : ℚ) / total_ways_to_draw

theorem probability_of_drawing_one_white_two_red : desired_probability = 10 / 21 :=
by sorry

end probability_of_drawing_one_white_two_red_l524_524545


namespace imaginary_unit_pure_imaginary_l524_524829

theorem imaginary_unit_pure_imaginary (a : ℝ) (h : ∃ y : ℝ, 
  (1 - a * Complex.i) / (1 + Complex.i) = Complex.i * y) : 
  a = 1 :=
by
  sorry

end imaginary_unit_pure_imaginary_l524_524829


namespace general_term_formula_l524_524722

def sequence_sums (n : ℕ) : ℕ := 2 * n^2 + n

theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : S = sequence_sums) :
  (∀ n, a n = S n - S (n-1)) → ∀ n, a n = 4 * n - 1 :=
by
  sorry

end general_term_formula_l524_524722


namespace no_real_intersections_l524_524350

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524350


namespace recruits_count_l524_524536

def x := 50
def y := 100
def z := 170

theorem recruits_count :
  ∃ n : ℕ, n = 211 ∧ (∀ a b c : ℕ, (b = 4 * a ∨ a = 4 * c ∨ c = 4 * b) → (b + 100 = a + 150) ∨ (a + 50 = c + 150) ∨ (c + 170 = b + 100)) :=
sorry

end recruits_count_l524_524536


namespace problem_equivalence_l524_524741

-- Condition Definitions
def fixed_point : ℝ × ℝ := (-Real.sqrt 3, 0)
def line_m (x : ℝ) : Prop := x = -(4 * Real.sqrt 3) / 3
def dist_ratio : ℝ := Real.sqrt 3 / 2

-- Equation of curve C
def curve_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Hypotheses about points A, B, D, E, and lines l1, l2
def line_l (k t : ℝ) (x y : ℝ) : Prop := y = k * x + t

def quadrilateral_vertices (k t1 t2 : ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :=
  line_l k t1 x1 y1 ∧ curve_C x1 y1 ∧ line_l k t1 x2 y2 ∧ curve_C x2 y2 ∧
  line_l k t2 x3 y3 ∧ curve_C x3 y3 ∧ line_l k t2 x4 y4 ∧ curve_C x4 y4 ∧
  x1 ≠ x2 ∧ x3 ≠ x4 ∧ t1 ≠ t2 ∧ (Real.abs (x1 - x2) = Real.abs (x3 - x4))

-- Maximum area of convex quadrilateral
def max_area (S : ℝ) : Prop := S = 4

-- Theorem Statement
theorem problem_equivalence (k t1 t2 x1 x2 y1 y2 x3 x4 y3 y4 S : ℝ) :
  curve_C x y →
  (∀ x y, line_l k t1 x y → curve_C x y → ¬ line_l k t2 x y) →
  quadrilateral_vertices k t1 t2 x1 y1 x2 y2 x3 y3 x4 y4 →
  max_area S :=
sorry

end problem_equivalence_l524_524741


namespace ratio_eq_one_l524_524981

variable {a b : ℝ}

theorem ratio_eq_one (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) : (a / 8) / (b / 7) = 1 := 
by
  sorry

end ratio_eq_one_l524_524981


namespace line_circle_no_intersection_l524_524376

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l524_524376


namespace triangle_least_perimeter_l524_524901

theorem triangle_least_perimeter (x : ℤ) (h1 : x + 27 > 34) (h2 : 34 + 27 > x) (h3 : x + 34 > 27) : 27 + 34 + x ≥ 69 :=
by
  have h1' : x > 7 := by linarith
  sorry

end triangle_least_perimeter_l524_524901


namespace quadratic_inequality_solution_l524_524202

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 + k * x - (3 / 4) < 0) ↔ -3 < k ∧ k ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l524_524202


namespace smallest_positive_period_fx_range_fx_on_interval_l524_524300

noncomputable def f (x : ℝ) := 2 * sin x ^ 2 + 2 * sqrt 3 * sin x * sin (x + π / 2)

theorem smallest_positive_period_fx : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := 
sorry

theorem range_fx_on_interval : ∀ x ∈ Icc (0 : ℝ) ((2 * π) / 3),
  0 ≤ f x ∧ f x ≤ 3 := 
sorry

end smallest_positive_period_fx_range_fx_on_interval_l524_524300


namespace boxes_needed_l524_524841

-- Let's define the conditions
def total_paper_clips : ℕ := 81
def paper_clips_per_box : ℕ := 9

-- Define the target of our proof, which is that the number of boxes needed is 9
theorem boxes_needed : total_paper_clips / paper_clips_per_box = 9 := by
  sorry

end boxes_needed_l524_524841


namespace solitaire_game_one_piece_l524_524855

theorem solitaire_game_one_piece (n : ℕ) (h : n > 0) :
  (∃ seq : List ℕ, valid_seq seq n) ↔ ¬ (∃ k : ℕ, n = 3 * k) :=
sorry

-- Definitions necessary for the theorem

-- valid_seq is a placeholder for the definition that checks if a given sequence of moves is valid and ends with one piece
def valid_seq (seq : List ℕ) (n : ℕ) : Prop :=
  sorry

end solitaire_game_one_piece_l524_524855


namespace unique_circle_arrangement_l524_524903

theorem unique_circle_arrangement (n : ℕ) (h : n ≥ 1) :
  ∃! (f : {i // 1 ≤ i ∧ i ≤ n} → {j // 1 ≤ j ∧ j ≤ n}), 
    (∀ (i : {i // 1 ≤ i ∧ i ≤ n}), abs ((f ⟨i.1%n.succ, sorry⟩).val - f i).val ≤ 2) := sorry

end unique_circle_arrangement_l524_524903


namespace no_real_intersections_l524_524348

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524348


namespace necklaces_currently_on_stand_l524_524142

-- Definition of the number of necklaces, rings, and bracelets needed to fill the displays
def necklaces_needed (max_necklaces current_necklaces : ℕ) : ℕ := max_necklaces - current_necklaces
def rings_needed (max_rings current_rings : ℕ) : ℕ := max_rings - current_rings
def bracelets_needed (max_bracelets current_bracelets : ℕ) : ℕ := max_bracelets - current_bracelets

-- Definition of the total cost to fill the displays
def total_cost (cost_necklace cost_ring cost_bracelet necklaces rings bracelets : ℕ) : ℕ := 
  cost_necklace * necklaces + cost_ring * rings + cost_bracelet * bracelets

-- The proof problem statement
theorem necklaces_currently_on_stand : 
  necklaces_needed 12 n + 7 = 5 :=
begin
  let N := necklaces_needed 12 12,
  have h : total_cost 4 10 5 N 12 7 = 183,
  { sorry },
  sorry,
end

end necklaces_currently_on_stand_l524_524142


namespace number_greater_than_neg_two_by_one_is_neg_one_l524_524966

theorem number_greater_than_neg_two_by_one_is_neg_one (
  A : ℤ := -3,
  B : ℤ := -1,
  C : ℤ := 3,
  D : ℤ := 1
) : (-2 + 1 = -1) ∧ (B = -1) :=
by
  sorry

end number_greater_than_neg_two_by_one_is_neg_one_l524_524966


namespace quadratic_roster_method_l524_524224

theorem quadratic_roster_method :
  {x : ℝ | x^2 - 3 * x + 2 = 0} = {1, 2} :=
by
  sorry

end quadratic_roster_method_l524_524224


namespace range_of_m_l524_524274

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing_neg : ∀ a b : ℝ, a < 0 ∧ b < 0 ∧ a < b → (f a < f b))
  (h_2m_plus_1_gt_2m : ∀ m, f (2 * m + 1) > f (2 * m)) :
  ∀ m, m < -1 / 4 :=
begin
  sorry
end

end range_of_m_l524_524274


namespace rabbit_count_l524_524926

-- Define the conditions
def original_rabbits : ℕ := 8
def new_rabbits_born : ℕ := 5

-- Define the total rabbits based on the conditions
def total_rabbits : ℕ := original_rabbits + new_rabbits_born

-- The statement to prove that the total number of rabbits is 13
theorem rabbit_count : total_rabbits = 13 :=
by
  -- Proof not needed, hence using sorry
  sorry

end rabbit_count_l524_524926


namespace charlie_book_pages_l524_524185

theorem charlie_book_pages :
  (2 * 40) + (4 * 45) + 20 = 280 :=
by 
  sorry

end charlie_book_pages_l524_524185


namespace six_digit_even_numbers_count_l524_524530

theorem six_digit_even_numbers_count :
  let digits : Finset ℕ := {1, 2, 3, 4, 5, 6},
      even_digits := {2, 4, 6},
      positions := Finset.range 6 in
  ∃ count : ℕ,
  count = 360 ∧
  ∀ (perm : List ℕ), perm ∈ List.permutations digits.to_list →
                       perm.length = 6 →
                       (perm.last ∈ even_digits) →
                       (1 ∈ perm ∧ 3 ∈ perm) →
                       ¬ (List.indexOf 1 perm + 1 = List.indexOf 3 perm ∨
                          List.indexOf 3 perm + 1 = List.indexOf 1 perm) →
  List.countp (λ p : List ℕ, true) [perm] = count := by
  sorry

end six_digit_even_numbers_count_l524_524530


namespace min_area_ellipse_l524_524139

theorem min_area_ellipse (a b : ℕ) (h : (π * (3 * (a + b) - real.sqrt ((3 * a + b) * (a + 3 * b)))) = 200) :
  π * a * b ≥ 30 * π := 
sorry

end min_area_ellipse_l524_524139


namespace hyperbola_vertex_distance_l524_524234

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l524_524234


namespace hyperbola_vertex_distance_l524_524238

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), (x^2 / 16 - y^2 / 9 = 1) → (vertex_distance : ℝ := 8) := sorry

end hyperbola_vertex_distance_l524_524238


namespace value_of_leftover_coins_is_1_70_l524_524632

def num_quarters_toledo := 95
def num_dimes_toledo := 172
def num_quarters_brian := 137
def num_dimes_brian := 290

def roll_quarters := 30
def roll_dimes := 50

def combined_quarters := num_quarters_toledo + num_quarters_brian
def combined_dimes := num_dimes_toledo + num_dimes_brian

def leftover_quarters := combined_quarters % roll_quarters
def leftover_dimes := combined_dimes % roll_dimes

def value_leftover_quarters := leftover_quarters * 0.25
def value_leftover_dimes := leftover_dimes * 0.10

def value_leftover_coins := value_leftover_quarters + value_leftover_dimes

theorem value_of_leftover_coins_is_1_70 : value_leftover_coins = 1.70 := by
  sorry

end value_of_leftover_coins_is_1_70_l524_524632


namespace a2017_value_l524_524761

def seq (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) = a n / (a n + 1)

theorem a2017_value :
  ∃ (a : ℕ → ℝ),
  seq a ∧ a 1 = 1 / 2 ∧ a 2017 = 1 / 2018 :=
by
  sorry

end a2017_value_l524_524761


namespace operation_parity_l524_524464

-- Assumptions based on the problem conditions
variables {a : ℕ → ℕ} 
variable {S : Fin 2007 → ℕ}
variable {seq : sequence (Fin 2007)}

-- Conditions given in the problem
hypothesis (a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
hypothesis (a_pos : ∀ i, a i > 0)
hypothesis (a_sum : (Finset.univ.sum (λ i => a i)) < 2007)

noncomputable def operation (seq : Fin 2007 → ℕ) (i : ℕ) : Fin 2007 → ℕ :=
if h : 1 ≤ i ∧ i ≤ 11 then
  seq.update i (seq i + a i)
else if h : 12 ≤ i ∧ i ≤ 22 then
  seq.update (i - 11) (seq (i - 11) - a (i - 11))
else
  seq

-- Theorem stating the conclusion
theorem operation_parity : 
    let odd_count := 0 -- placeholder for actual counting of odd operations
    let even_count := 0 -- placeholder for actual counting of even operations
    even_count - odd_count = 0 :=
sorry

end operation_parity_l524_524464


namespace probability_AM_less_than_AC_l524_524783

-- Definitions based on the conditions
variables {A B C M : Type}
variable (triangle : triangle A B C)
variable (h1 : AC = BC)
variable (h2 : angle C = 90)
variable (M : point_on_segment AB)

-- Definition of the probability function and statement of the theorem
noncomputable def geometric_probability (p : ℝ) : ℝ := p

theorem probability_AM_less_than_AC :
  geometric_probability (AM < AC) = (real.sqrt 2) / 2 :=
sorry

end probability_AM_less_than_AC_l524_524783


namespace sequence_square_perfect_square_l524_524676

def sequence (n : ℕ) : ℕ 
| 0       := 1
| 1       := 1
| (n + 1) := 2 * sequence n + sequence (n - 1)

theorem sequence_square_perfect_square (n : ℕ) : ∃ m : ℕ, 2 * (sequence (2 * n) ^ 2 - 1) = m ^ 2 := sorry

end sequence_square_perfect_square_l524_524676


namespace compare_y1_y2_l524_524502

theorem compare_y1_y2 : 
  (∃ y1 y2 : ℝ, (y1 = 8 * 3 - 1) ∧ (y2 = 8 * 4 - 1)) → y1 < y2 :=
by
  sorry

end compare_y1_y2_l524_524502


namespace ways_to_sum_420_as_consecutive_integers_l524_524443

def sum_of_consecutive_integers (k n : ℕ) : ℕ :=
k * n + (k * (k - 1)) / 2

theorem ways_to_sum_420_as_consecutive_integers :
  (finset.filter (λ k : ℕ, k ≥ 2 ∧ (420 % k = 0) ∧ 
  ∃ n : ℕ, n = 420 / k - (k - 1) / 2 ∧ n > 0)
  (finset.range 421)).card = 3 := by
  sorry

end ways_to_sum_420_as_consecutive_integers_l524_524443


namespace infinite_solutions_l524_524868

theorem infinite_solutions (D : ℤ) (hD : D ≠ 0) : ∃ᶠ (x y z : ℕ) in at_top, 
  (x^2 - D * y^2 = z^2) ∧ (nat.gcd x y = 1) :=
sorry

end infinite_solutions_l524_524868


namespace shortest_distance_F1_max_distance_F2_to_line_max_value_PF2_plus_QF2_range_inner_product_F1P_F1Q_l524_524726

-- Ellipse conditions and foci definitions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 5) = 1
def F1 := (-2, 0 : ℝ)
def F2 := (2, 0 : ℝ)

-- Line passing through F1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 2)

-- Mathematical propositions
theorem shortest_distance_F1 :
  (∃ P : ℝ × ℝ, ellipse P.1 P.2 ∧ dist P F1 = 1) := sorry
  
theorem max_distance_F2_to_line (k : ℝ) :
  (∃ l : ℝ → ℝ, (l = line_l k) ∧
   (∀ x : ℝ, l x = F1.2 → (dist F2 (x, l x)) = 4)) := sorry

theorem max_value_PF2_plus_QF2 :
  (∃ P Q : ℝ × ℝ, ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧
   (|dist P F2 + dist Q F2| = 26/3)) := sorry

theorem range_inner_product_F1P_F1Q :
  (∃ P Q : ℝ × ℝ, ellipse P.1 P.2 ∧ ellipse Q.1 Q.2 ∧
   (∀ k : ℝ, line_l k P.1 = P.2 ∧ line_l k Q.1 = Q.2 →
   (inner (F1.1 - P.1, F1.2 - P.2) (F1.1 - Q.1, F1.2 - Q.2)) ∈ set.Icc (-5 : ℝ) (-25 / 9))) := sorry

end shortest_distance_F1_max_distance_F2_to_line_max_value_PF2_plus_QF2_range_inner_product_F1P_F1Q_l524_524726


namespace no_real_intersections_l524_524352

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524352


namespace jellybeans_original_count_l524_524216

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l524_524216


namespace sequence_general_formula_l524_524270

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (rec : ∀ n : ℕ, n > 0 → a n = n * (a (n + 1) - a n)) : 
  ∀ n, a n = n := 
by 
  sorry

end sequence_general_formula_l524_524270


namespace second_rice_price_correct_l524_524808

-- Non-computable definition for the cost of the second type of rice
noncomputable def second_rice_price (c1 c_m ratio : ℝ) : ℝ :=
  let x := (ratio * (c_m - c1)) + c_m in x

-- Theorem statement that asserts the proof of the calculated price of the second type of rice
theorem second_rice_price_correct :
    (second_rice_price 16 18 3) = 24 :=
by
  -- Use the definition and simplify directly
  sorry

end second_rice_price_correct_l524_524808


namespace average_speed_of_trip_l524_524131

theorem average_speed_of_trip :
  let distance1 := 40
  let speed1 := 20
  let distance2 := 10
  let speed2 := 15
  let distance3 := 5
  let speed3 := 10
  let distance4 := 180
  let speed4 := 60
  let distance5 := 25
  let speed5 := 30
  let distance6 := 20
  let speed6 := 45
  let total_distance := distance1 + distance2 + distance3 + distance4 + distance5 + distance6
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let time5 := distance5 / speed5
  let time6 := distance6 / speed6
  let total_time := time1 + time2 + time3 + time4 + time5 + time6
  let average_speed := total_distance / total_time
  average_speed ≈ 37.62 :=
by
  sorry

end average_speed_of_trip_l524_524131


namespace reflection_of_P_over_line_e_l524_524537

-- Definitions of points A, B, C and their reflections over line e
variables {A B C A1 B1 C1 A2 B2 C2 P : Type}
          [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A1]
          [AddGroup B1] [AddGroup C1] [AddGroup A2] [AddGroup B2]
          [AddGroup C2] [AddGroup P]

-- Definition of vector sum at the midpoint
hypothesis (h : A1 + A2 + B1 + B2 + C1 + C2 = (0 : A))

-- Main theorem statement
theorem reflection_of_P_over_line_e :
  P = P :=
sorry

end reflection_of_P_over_line_e_l524_524537


namespace faye_age_l524_524205
open Nat

theorem faye_age :
  ∃ (C D E F : ℕ), 
    (D = E - 3) ∧ 
    (E = C + 4) ∧ 
    (F = C + 3) ∧ 
    (D = 14) ∧ 
    (F = 16) :=
by
  sorry

end faye_age_l524_524205


namespace vertical_asymptote_at_x3_l524_524418

noncomputable def function_y (x : ℝ) : ℝ := (x^3 + x^2 + 1) / (x - 3)

theorem vertical_asymptote_at_x3 (x : ℝ) : (x = 3) → ¬ (x^3 + x^2 + 1 = 0) ∧ x - 3 = 0 :=
begin
  intro h,
  split,
  { rw h, norm_num },  -- Show that the numerator is not zero at x = 3
  { rw h, norm_num }   -- Show that the denominator is zero at x = 3
end

end vertical_asymptote_at_x3_l524_524418


namespace team_division_is_possible_l524_524434

variable {V : Type} [Fintype V]
variable (students : V)
variable (dislikes : V → Finset V)

def prop_dislikes : Prop :=
  ∀ v, dislikes v ⊆ univ ∧ dislikes v.card ≤ 3 ∧ (∀ v₁ v₂ ∈ dislikes v, v₁ ≠ v₂)

def bipartite_graph (G : SimpleGraph V) : Prop := 
  ∃ (A B : Finset V), (V = A ∪ B ∧ A ∩ B = ∅) ∧ 
  ∀ v ∈ A, ∀ u ∈ B, (G.adj v u) ∧ (∀ v₁ v₂ ∈ A, ¬ G.adj v₁ v₂) ∧ (∀ u₁ u₂ ∈ B, ¬ G.adj u₁ u₂) 

theorem team_division_is_possible (G : SimpleGraph V) (h : ∀ v : V, (G.degree v) ≤ 3) : 
  bipartite_graph G :=
by
  sorry

end team_division_is_possible_l524_524434


namespace MattRate_l524_524844

variable (M : ℝ) (t : ℝ)

def MattRateCondition : Prop := M * t = 220
def TomRateCondition : Prop := (M + 5) * t = 275

theorem MattRate (h1 : MattRateCondition M t) (h2 : TomRateCondition M t) : M = 20 := by
  sorry

end MattRate_l524_524844


namespace martin_diff_l524_524487

noncomputable theory

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def semi_annual_compound (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  compound_interest P r 2 t

def annual_compound (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  compound_interest P r 1 t

theorem martin_diff (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 8000) (hr : r = 0.10) (ht : t = 5) :
  semi_annual_compound P r t - annual_compound P r t = 147.04 := 
by
  rw [hP, hr, ht]
  sorry

end martin_diff_l524_524487


namespace find_x_of_means_l524_524119

noncomputable def arithmetic_mean (a : ℕ) (b : ℕ) (c : ℕ) : ℕ :=
(a + b + c) / 3

theorem find_x_of_means :
  ∃ x : ℕ, arithmetic_mean 20 40 60 = 5 + arithmetic_mean 10 60 x :=
begin
  use 35,
  sorry -- The proof is not required as per the instructions.
end

end find_x_of_means_l524_524119


namespace slope_sum_is_284_l524_524058

theorem slope_sum_is_284 :
  ∃ (m n : ℕ), let A := (30, 200), D := (31, 217) in
    -- A and D defining the conditions of the vertices
    -- Integer vertices for isosceles trapezoid with these specific properties
       (∀ B C : ℤ × ℤ, (B.1 ≠ C.1 ∧
                       B.2 ≠ C.2 ∧
                       (C.1 - B.1 ≠ 0) ∧
                       (B.2 - A.2 ≠ 0) ∧
                       (D.2 - C.2 ≠ 0)) → True) →
       -- Parallel sides
       let slope1 := (B.2 - A.2) / (B.1 - A.1) in
       let slope2 := (D.2 - C.2) / (D.1 - C.1) in
       slope1 = slope2  →
       -- Slopes summing to 271 / 13 translated to m + n = 284
       (∀ (slope1 slope2 : ℚ), (slope1 = 1 ∨ slope1 = 3 / 4 ∨ slope1 = -1  ∨ slope1 = -(14 / 13) ∨ slope1 = 2 ∨ slope1 = 1 / 3 ∨ slope1 = -3 ∨ slope1 = -(1 / 2))
       ∧ ∃ s1 s2, abs s1 + abs s2 = 271 / 13 → 
       m + n = 284 :=
        sorry

end slope_sum_is_284_l524_524058


namespace common_ratio_geometric_sequence_natural_numbers_l524_524637

theorem common_ratio_geometric_sequence_natural_numbers (b1 q : ℕ) (hq : q ∈ {1, 2, 3, 4})
  (h_sum : b1 * q^2 * (1 + q^2 + q^4) = 819 * 6^2016) : q ∈ {1, 2, 3, 4} :=
sorry

end common_ratio_geometric_sequence_natural_numbers_l524_524637


namespace hyperbola_eccentricity_l524_524759

theorem hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (intersect_cond : ∀ e : ℝ, bx a b x ∧ ay a b y → (x - 2)^2 + y^2 = 2) : 
  1 < e a b ∧ e a b < Math.sqrt 2 := 
sorry

def e (a b : ℝ) : ℝ := Math.sqrt (1 + b^2 / a^2)
def bx (a b : ℝ) (x : ℝ) : Prop := b * x = a
def ay (a b : ℝ) (y : ℝ) : Prop := b * y = -a

end hyperbola_eccentricity_l524_524759


namespace second_player_cannot_lose_l524_524551

-- A grid is represented as a set of cells
structure Grid :=
  (cells : set (ℕ × ℕ))

-- A game state is represented by the set of occupied edges
structure GameState :=
  (occupied_edges : set (ℕ × ℕ × ℕ × ℕ))
  (broken_line_end : ℕ × ℕ)

-- Initial game state configuration
def initial_state : GameState :=
  { occupied_edges := ∅, broken_line_end := (0, 0) }

-- Function that describes a move along the grid, resulting in a new game state
def move (s : GameState) (new_end : ℕ × ℕ) : GameState :=
  { occupied_edges := s.occupied_edges ∪ { (s.broken_line_end, new_end) },
    broken_line_end := new_end }

-- Predicate that checks if a move is valid (aligned with sides of cells, not revisiting edges)
def valid_move (s : GameState) (new_end : ℕ × ℕ) : Prop :=
  (new_end.1 = s.broken_line_end.1 ∧ (new_end.2 = s.broken_line_end.2 + 1 ∨ new_end.2 = s.broken_line_end.2 - 1)) ∨
  (new_end.2 = s.broken_line_end.2 ∧ (new_end.1 = s.broken_line_end.1 + 1 ∨ new_end.1 = s.broken_line_end.1 - 1)) ∧
  ((s.broken_line_end, new_end) ∉ s.occupied_edges)

-- Condition to check if a player loses
def player_loses (s : GameState) : Prop :=
  ∀ (new_end : ℕ × ℕ), ¬ valid_move s new_end

theorem second_player_cannot_lose (s : GameState) :
  ∀ (n : ℕ), ¬ player_loses s :=
by sorry

end second_player_cannot_lose_l524_524551


namespace match_probability_l524_524995

/-- There are four different types of rare plants and four corresponding seed pictures,
with a random guess determining the matching. The probability that all types of plants
are matched correctly to their corresponding seeds is 1/24. -/
theorem match_probability (types_seeds : Finset (Fin 4 × Fin 4)) (h1 : types_seeds.card = 24)
  : Prob_correct := 1 / 24 :=
begin
  sorry
end

end match_probability_l524_524995


namespace sum_of_positive_integer_factors_of_24_l524_524951

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l524_524951


namespace geometric_series_sum_l524_524670

theorem geometric_series_sum :
  let a : ℤ := 1
  let r : ℤ := -3
  let n : ℕ := 8
  let last_term : ℤ := -2187
  last_term = a * (r ^ (n - 1)) →
  ∑ i in finset.range n, a * (r ^ i) = -1640 := 
by
  intros a r n last_term h_last_term
  simp only [n, a, r] at h_last_term
  sorry

end geometric_series_sum_l524_524670


namespace average_marks_combined_l524_524120

theorem average_marks_combined (avg1 : ℝ) (students1 : ℕ) (avg2 : ℝ) (students2 : ℕ) :
  avg1 = 30 → students1 = 30 → avg2 = 60 → students2 = 50 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2) = 48.75 := 
by
  intros h_avg1 h_students1 h_avg2 h_students2
  sorry

end average_marks_combined_l524_524120


namespace line_with_equal_intercepts_l524_524890

theorem line_with_equal_intercepts (P : ℝ × ℝ) (hP : P = (2, 3))
    (hequal_intercepts : ∃ a b : ℝ, a = b):
    (∃ m : ℝ, (line_eq m 3 2) = true) ∨
    (∃ p : ℝ, (line_eq p 5 0) = true) := 
sorry

def line_eq (m b : ℝ) (x y: ℝ) := y = m * x + b

end line_with_equal_intercepts_l524_524890


namespace min_value_frac_sum_l524_524290

open Real

theorem min_value_frac_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  1 / a + 2 / b = 8 :=
sorry

end min_value_frac_sum_l524_524290


namespace no_real_intersections_l524_524353

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524353


namespace derivative_cos3_l524_524230

theorem derivative_cos3 : ∀ (x : ℝ), 
  deriv (λ x, (cos (2*x + 3))^3) x = -6 * (cos (2*x + 3))^2 * sin (2*x + 3)  := 
by 
  -- The proof is not required
  sorry

end derivative_cos3_l524_524230


namespace course_selection_schemes_l524_524155

theorem course_selection_schemes (h1 : 7 > 4) (h2 : 2> 1) (h3 : ∀ (A B : Type) (s : Finset A), card s = 7) :
  (choose 7 4) - ((choose 2 2) * (choose 5 2)) = 25 :=
by sorry

end course_selection_schemes_l524_524155


namespace first_place_points_l524_524790

-- Definitions for the conditions
def num_teams : Nat := 4
def points_win : Nat := 2
def points_draw : Nat := 1
def points_loss : Nat := 0

def games_played (n : Nat) : Nat :=
  let pairs := n * (n - 1) / 2  -- Binomial coefficient C(n, 2)
  2 * pairs  -- Each pair plays twice

def total_points_distributed (n : Nat) (points_per_game : Nat) : Nat :=
  (games_played n) * points_per_game

def last_place_points : Nat := 5

-- The theorem to prove
theorem first_place_points : ∃ a b c : Nat, a + b + c = total_points_distributed num_teams points_win - last_place_points ∧ (a = 7 ∨ b = 7 ∨ c = 7) :=
by
  sorry

end first_place_points_l524_524790


namespace total_arrangements_equal_to_288_l524_524991

-- Let's define the given conditions
def students := ℕ -- We represent number of students as natural numbers.
def boys := 3
def girls := 3
def boy_A := 1
def total_students := boys + girls  -- This equals 6 due to condition 1

-- Conditions
def boy_A_not_at_ends (arrangement : list ℕ) : Prop :=
  arrangement.nth 0 ≠ boy_A ∧ arrangement.nth (arrangement.length - 1) ≠ boy_A

def exactly_two_girls_adjacent (arrangement : list ℕ) : Prop :=
  count_adjacent_girls arrangement = 1

def count_adjacent_girls (arrangement : list ℕ) : ℕ := 
  arrangement.foldl (λ acc x, if acc > 0 && x > 3 then acc - 1 else acc) 0
-- This is a placeholder for checking adjacent girls condition

-- The theorem to prove
theorem total_arrangements_equal_to_288 :
  ∀ (arrangement : list ℕ), 
  (arrangement.length = total_students) → 
  boy_A_not_at_ends arrangement →
  exactly_two_girls_adjacent arrangement →
  count_arrangements_valid arrangement = 288 :=
begin
  sorry
end

end total_arrangements_equal_to_288_l524_524991


namespace carla_count_total_l524_524207

theorem carla_count_total :
  let monday_counts := 1 + 1 in
  let tuesday_counts := 2 + 3 in
  let wednesday_counts := 4 in
  let thursday_counts := 3 + 2 in
  let friday_counts := 2 + 3 + 1 in
  monday_counts + tuesday_counts + wednesday_counts + thursday_counts + friday_counts = 22 := 
by
  sorry

end carla_count_total_l524_524207


namespace sum_of_divisors_24_l524_524956

theorem sum_of_divisors_24 : (∑ d in (finset.filter (λ n, 24 % n = 0) (finset.range 25)), d) = 60 :=
by
  sorry

end sum_of_divisors_24_l524_524956


namespace sum_of_positive_integer_factors_of_24_l524_524954

-- Define the number 24
def n : ℕ := 24

-- Define the list of positive factors of 24
def pos_factors_of_24 : List ℕ := [1, 2, 4, 8, 3, 6, 12, 24]

-- Define the sum of the factors
def sum_of_factors : ℕ := pos_factors_of_24.sum

-- The theorem statement
theorem sum_of_positive_integer_factors_of_24 : sum_of_factors = 60 := by
  sorry

end sum_of_positive_integer_factors_of_24_l524_524954


namespace root_interval_sum_l524_524779

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) (h1 : b - a = 1) (h2 : ∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) : a + b = -3 :=
by
  sorry

end root_interval_sum_l524_524779


namespace ratio_a7_b7_l524_524257

variable (a b : ℕ → ℕ) -- Define sequences a and b
variable (S T : ℕ → ℕ) -- Define sums S and T

-- Define conditions: arithmetic sequences and given ratio
variable (h_arith_a : ∀ n, a (n + 1) - a n = a 1)
variable (h_arith_b : ∀ n, b (n + 1) - b n = b 1)
variable (h_sum_a : ∀ n, S n = (n + 1) * a 1 + n * a n)
variable (h_sum_b : ∀ n, T n = (n + 1) * b 1 + n * b n)
variable (h_ratio : ∀ n, (S n) / (T n) = (3 * n + 2) / (2 * n))

-- Define the problem statement using the given conditions
theorem ratio_a7_b7 : (a 7) / (b 7) = 41 / 26 :=
by
  sorry

end ratio_a7_b7_l524_524257


namespace rice_mixture_ratio_l524_524635

-- Definitions for the given conditions
def cost_per_kg_rice1 : ℝ := 5
def cost_per_kg_rice2 : ℝ := 8.75
def cost_per_kg_mixture : ℝ := 7.50

-- The problem: ratio of two quantities
theorem rice_mixture_ratio (x y : ℝ) (h : cost_per_kg_rice1 * x + cost_per_kg_rice2 * y = 
                                     cost_per_kg_mixture * (x + y)) :
  y / x = 2 := 
sorry

end rice_mixture_ratio_l524_524635


namespace range_of_m_l524_524730

variable (m t : ℝ)

namespace proof_problem

def proposition_p : Prop :=
  ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1) → (t + 2) * (t - 10) < 0

def proposition_q (m : ℝ) : Prop :=
  -m < t ∧ t < m + 1 ∧ m > 0

theorem range_of_m :
  (∃ t, proposition_q m t) → proposition_p t → 0 < m ∧ m ≤ 2 := by
  sorry

end proof_problem

end range_of_m_l524_524730


namespace min_value_f_on_interval_l524_524758

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

theorem min_value_f_on_interval :
  ∀ x ∈ set.Icc 1 2, f x ≥ f 1 :=
by
  sorry

end min_value_f_on_interval_l524_524758


namespace det_of_skew_symmetric_matrix_nonneg_l524_524141

def is_skew_symmetric {n : Type} [fintype n] [decidable_eq n] (A : matrix n n ℝ) : Prop :=
  Aᵀ = -A

theorem det_of_skew_symmetric_matrix_nonneg {A : matrix (fin 4) (fin 4) ℝ} 
  (h : is_skew_symmetric A) : det A ≥ 0 :=
sorry

end det_of_skew_symmetric_matrix_nonneg_l524_524141


namespace size_ratio_l524_524183

variable {U : ℝ} (h1 : C = 1.5 * U) (h2 : R = 4 / 3 * C)

theorem size_ratio : R = 8 / 3 * U :=
by
  sorry

end size_ratio_l524_524183


namespace infinite_solutions_l524_524000

theorem infinite_solutions (a : ℤ) (h_a : a > 1) 
  (h_sol : ∃ x y : ℤ, x^2 - a * y^2 = -1) : 
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ, (f n).fst^2 - a * (f n).snd^2 = -1 :=
sorry

end infinite_solutions_l524_524000


namespace xn_yn_tan_eq_l524_524820

noncomputable def x_n (n : ℕ) (t : ℝ) : ℝ := 
  ∑ k in Finset.range n, (k+1) * (n - k - 1) * Real.cos (t * (k + 1))

noncomputable def y_n (n : ℕ) (t : ℝ) : ℝ := 
  ∑ k in Finset.range n, (k+1) * (n - k - 1) * Real.sin (t * (k + 1))

theorem xn_yn_tan_eq (n : ℕ) (t : ℝ) (h1 : n ≥ 2) (h2 : ∀ k : ℤ, t ≠ k * Real.pi) :
  (x_n n t = 0 ∧ y_n n t = 0) ↔ Real.tan (n * t / 2) = n * Real.tan (t / 2) :=
sorry

end xn_yn_tan_eq_l524_524820


namespace simplify_and_rationalize_l524_524877

noncomputable def expression : ℚ := (real.sqrt 2 / real.sqrt 3) * (real.sqrt 4 / real.sqrt 5) * (real.sqrt 6 / real.sqrt 7)
noncomputable def answer : ℚ := 4 * real.sqrt 35 / 35

theorem simplify_and_rationalize : expression = answer :=
by
  sorry

end simplify_and_rationalize_l524_524877


namespace book_selection_l524_524086

theorem book_selection :
  let chinese_books := 8
  let math_books := 6
  let english_books := 5
  let diff_subject_books := chinese_books * math_books + chinese_books * english_books + math_books * english_books
  diff_subject_books = 118 :=
by
  let chinese_books := 8
  let math_books := 6
  let english_books := 5
  let diff_subject_books := chinese_books * math_books + chinese_books * english_books + math_books * english_books
  show diff_subject_books = 118, from sorry

end book_selection_l524_524086


namespace painting_wall_problem_l524_524414

theorem painting_wall_problem (h_time : ℕ) (t_time : ℕ) (combined_time : ℕ)
  (heidi_rate : ℚ := 1 / h_time)
  (tim_rate : ℚ := 1 / t_time)
  (combined_rate : ℚ := heidi_rate + tim_rate)
  (painted_fraction : ℚ := combined_time * combined_rate) :
  (h_time = 45) → (t_time = 30) → (combined_time = 9) →
  painted_fraction = 1 / 2 :=
by
  intros h_time_eq t_time_eq combined_time_eq
  simp [heidi_rate, tim_rate, combined_rate, painted_fraction, h_time_eq, t_time_eq, combined_time_eq]
  sorry

end painting_wall_problem_l524_524414


namespace sum_of_angles_around_common_point_l524_524439

theorem sum_of_angles_around_common_point (n : ℕ) (h1 : n > 0) (h2 : ∀ i j : ℕ, i ≠ j → (i % n) ≠ (j % n)) : 
  (finset.range n).sum (λ i, 360 / n) = 360 :=
by
  sorry

end sum_of_angles_around_common_point_l524_524439


namespace solve_main_theorem_l524_524195

def g (x : ℝ) : ℝ :=
if x < 10 then 3 * x + 6 else 5 * x - 5

-- Define the conditions for g⁻¹(18)
def inv_g_18 (y : ℝ) : Prop :=
g y = 18

-- Define the conditions for g⁻¹(55)
def inv_g_55 (y : ℝ) : Prop :=
g y = 55

def main_theorem : Prop :=
∃ (x₁ x₂ : ℝ), inv_g_18 x₁ ∧ inv_g_55 x₂ ∧ x₁ + x₂ = 16

theorem solve_main_theorem : main_theorem := by
  sorry

end solve_main_theorem_l524_524195


namespace new_average_weight_is_correct_l524_524051

variable (avg_weight_19 : ℝ) (num_students_19 : ℕ) (weight_new_student : ℝ) (num_students_new : ℕ) (new_avg_weight : ℝ)

-- Defining the given conditions
def condition1 := avg_weight_19 = 15
def condition2 := num_students_19 = 19
def condition3 := weight_new_student = 3 
def condition4 := num_students_new = 20

-- The theorem to be proven
theorem new_average_weight_is_correct :
  condition1 → condition2 → condition3 → condition4 → new_avg_weight = (285 + weight_new_student) / num_students_new := 
by
  sorry

end new_average_weight_is_correct_l524_524051


namespace aragorn_wins_game_l524_524173

-- Define the initial state and game conditions
structure GameState where
  arrows : Fin 2019 → Bool -- True means pointing north, False means pointing south

def initial_state : GameState :=
  { arrows := fun _ => true }

def flip_arrow (s : GameState) (n : Fin 2019) : GameState :=
  { s with arrows := s.arrows.update n (!s.arrows n) }

def valid_move (seen_states : List GameState) (new_state : GameState) : Bool :=
  not (new_state ∈ seen_states)

-- Define the main theorem stating that Aragorn can always win
theorem aragorn_wins_game : ∀ (s : GameState), (∀ seen_states : List GameState, true) → (∃ (winning_strategy : List (Fin 2019)), true) :=
by
  sorry

end aragorn_wins_game_l524_524173


namespace shaded_area_l524_524648

def diameter (BC : ℝ) : Prop := BC = 8
def is_isosceles (AB AC : ℝ) : Prop := AB = AC
def midpoint (D AC : ℝ) : Prop := D = AC / 2
noncomputable def pi : ℝ := 3.14

theorem shaded_area (BC AB AC D : ℝ) (h₁ : diameter BC) (h₂ : is_isosceles AB AC) (h₃ : midpoint D AC) : 
shaded_area BC AB AC D = 9.12 :=
sorry

end shaded_area_l524_524648


namespace triangular_25_eq_325_l524_524100

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l524_524100


namespace parallel_tangent_line_l524_524061

theorem parallel_tangent_line (b : ℝ) :
  (∃ b : ℝ, (∀ x y : ℝ, x + 2 * y + b = 0 → (x^2 + y^2 = 5))) →
  (b = 5 ∨ b = -5) :=
by
  sorry

end parallel_tangent_line_l524_524061


namespace overall_class_average_proof_l524_524777

noncomputable def group_1_weighted_average := (0.40 * 80) + (0.60 * 80)
noncomputable def group_2_weighted_average := (0.30 * 60) + (0.70 * 60)
noncomputable def group_3_weighted_average := (0.50 * 40) + (0.50 * 40)
noncomputable def group_4_weighted_average := (0.20 * 50) + (0.80 * 50)

noncomputable def overall_class_average := (0.20 * group_1_weighted_average) + 
                                           (0.50 * group_2_weighted_average) + 
                                           (0.25 * group_3_weighted_average) + 
                                           (0.05 * group_4_weighted_average)

theorem overall_class_average_proof : overall_class_average = 58.5 :=
by 
  unfold overall_class_average
  unfold group_1_weighted_average
  unfold group_2_weighted_average
  unfold group_3_weighted_average
  unfold group_4_weighted_average
  -- now perform the arithmetic calculations
  sorry

end overall_class_average_proof_l524_524777


namespace line_circle_no_intersection_l524_524337

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524337


namespace time_to_meet_l524_524935

-- Define the conditions
def car1_speed : ℝ := 100 -- Car 1 speed is 100 km/h
def car2_speed := car1_speed / 1.25 -- Car 2 speed is 25% slower than Car 1
def distance_between_cars : ℝ := 720 -- Distance between the cars is 720 km
def combined_speed : ℝ := car1_speed + car2_speed

-- Define the main theorem
theorem time_to_meet : distance_between_cars / combined_speed = 4 := by
  -- Use sorry to skip the proof
  sorry

end time_to_meet_l524_524935


namespace total_trip_hours_l524_524612

-- Define the given conditions
def speed1 := 50 -- Speed in mph for the first 4 hours
def time1 := 4 -- First 4 hours
def distance1 := speed1 * time1 -- Distance covered in the first 4 hours

def speed2 := 80 -- Speed in mph for additional hours
def average_speed := 65 -- Average speed for the entire trip

-- Define the proof problem
theorem total_trip_hours (T : ℕ) (A : ℕ) :
  distance1 + (speed2 * A) = average_speed * T ∧ T = time1 + A → T = 8 :=
by
  sorry

end total_trip_hours_l524_524612


namespace geometric_progression_common_ratio_l524_524436

theorem geometric_progression_common_ratio (r : ℝ) :
  (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) ↔
  r = ( -1 + ((19 + 3 * Real.sqrt 33)^(1/3)) + ((19 - 3 * Real.sqrt 33)^(1/3)) ) / 3 :=
by
  sorry

end geometric_progression_common_ratio_l524_524436


namespace part_a_part_b_l524_524480

noncomputable def periodic_function_condition (f : ℝ → ℝ) (hf_periodic : ∀ x : ℝ, f(x + 1) = f(x)) : Prop :=
  ∀ (a : ℝ), ∫ x in a..(a + 1), f x = ∫ x in 0..1, f x

theorem part_a (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_periodic : ∀ x : ℝ, f(x + 1) = f(x)) 
  (hf_nonneg : ∀ x, 0 ≤ f x) :
  periodic_function_condition f hf_periodic :=
begin
  sorry
end

theorem part_b (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_periodic : ∀ x : ℝ, f(x + 1) = f(x)) 
  (hf_nonneg : ∀ x, 0 ≤ f x) :
  (lim (n → ∞) ∫ x in 0..1, f x * f (n * x)) = (∫ x in 0..1, f x) ^ 2 :=
begin
  sorry
end

end part_a_part_b_l524_524480


namespace number_of_spectators_l524_524435

theorem number_of_spectators (total_wristbands : ℕ) (wristbands_per_person : ℕ) (h_total_wristbands : total_wristbands = 290) (h_wristbands_per_person : wristbands_per_person = 2) : 
  total_wristbands / wristbands_per_person = 145 :=
by
  rw [h_total_wristbands, h_wristbands_per_person]
  norm_num
  -- sorry to skip actual proof step following.
  sorry

end number_of_spectators_l524_524435


namespace max_ordered_pairs_l524_524479

theorem max_ordered_pairs (π : Equiv.Perm (Fin 2015)) :
  let pairs := {(i, j) | i < j ∧ π i * π j > i * j}.toFinset.card,
  pairs = (2014 * 1007) :=
sorry

end max_ordered_pairs_l524_524479


namespace find_t_l524_524729

noncomputable def is_angle_between_lines_correct (a b : ℝ) (t : ℝ) : Prop :=
  let slope_l1 := a
  let slope_l2 := -1 / t
  let angle_rad := Real.pi / 3 -- 60 degrees in radians
  Real.tan angle_rad = abs ((slope_l1 - slope_l2) / (1 + slope_l1 * slope_l2))

theorem find_t (t : ℝ) : 
  is_angle_between_lines_correct (√3 / 3) (x : ℝ) t → (t = 0 ∨ t = √3) :=
sorry

end find_t_l524_524729


namespace problem_1_problem_2_l524_524989

def f (x : ℝ) (k : ℝ) : ℝ := x^2 - x + k

theorem problem_1 (a k : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
    (h3 : log 2 (f a k) = 2) (h4 : f (log 2 a) k = k) : a = 2 ∧ k = 2 :=
sorry

theorem problem_2 (f : ℝ → ℝ → ℝ) (a : ℝ) (k : ℝ)
    (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log 2 (f a k) = 2) (h4 : f (log 2 a) k = k)
    (h6 : a = 2) (h7 : k = 2) : ∃ x, f (log 2 x) k = 2 :=
sorry

end problem_1_problem_2_l524_524989


namespace profit_percentage_calculation_l524_524079

noncomputable def sale_price_with_tax : ℝ := 616
noncomputable def tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 526.50

-- Lean 4 statement to prove the profit percentage
theorem profit_percentage_calculation :
  let actual_sale_price := sale_price_with_tax / (1 + tax_rate)
  let profit := actual_sale_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage ≈ 6.36 :=
by
  sorry

end profit_percentage_calculation_l524_524079


namespace line_circle_no_intersection_l524_524388

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end line_circle_no_intersection_l524_524388


namespace win_sector_area_l524_524618

namespace MathProof

-- Define the radius of the circular spinner
def radius : ℝ := 15

-- Define the probability of winning
def winning_probability : ℝ := 1 / 3

-- Define the total area of the circle
def total_area : ℝ := Real.pi * radius^2

-- Define the area of the WIN sector
def win_area : ℝ := winning_probability * total_area

-- Theorem stating that the area of the WIN sector is 75π square centimeters
theorem win_sector_area : win_area = 75 * Real.pi := by
  unfold win_area
  unfold winning_probability
  unfold total_area
  -- Proof omitted
  sorry

end MathProof

end win_sector_area_l524_524618


namespace concurrency_of_lines_l524_524826

theorem concurrency_of_lines 
  (A B C D E F : Point) 
  (Γ : Circle) 
  (hΓ : Γ = circumcircle A B C)
  (hD_on_BC : D ∈ line_segment B C)
  (tangent_A : line_through A tangent_to Γ)
  (line_ED_parallel_to_BA : line_parallel E D line BA)
  (hCE_meets_Γ_at_F_ag : meets_circle_again CE F Γ)
  (hBDEF_concyclic : concyclic B D E F) 
  : concurrent (line_through A C) (line_through B F) (line_through D E) :=
sorry

end concurrency_of_lines_l524_524826


namespace solve_for_x_l524_524691

theorem solve_for_x :
  { x : Real | ⌊ 2 * x * ⌊ x ⌋ ⌋ = 58 } = {x : Real | 5.8 ≤ x ∧ x < 5.9} :=
sorry

end solve_for_x_l524_524691


namespace basketball_minutes_played_l524_524594

-- Definitions of the conditions in Lean
def football_minutes : ℕ := 60
def total_hours : ℕ := 2
def total_minutes : ℕ := total_hours * 60

-- The statement we need to prove (that basketball_minutes = 60)
theorem basketball_minutes_played : 
  (120 - football_minutes = 60) := by
  sorry

end basketball_minutes_played_l524_524594


namespace simplify_expression_l524_524509

variable (x : ℝ)

theorem simplify_expression :
  2 * x * (4 * x^2 - 3 * x + 1) - 4 * (2 * x^2 - 3 * x + 5) =
  8 * x^3 - 14 * x^2 + 14 * x - 20 := 
  sorry

end simplify_expression_l524_524509


namespace line_circle_no_intersection_l524_524405

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524405


namespace percent_primes_less_than_20_divisible_by_5_l524_524588

theorem percent_primes_less_than_20_divisible_by_5 :
  let primes := [ 2, 3, 5, 7, 11, 13, 17, 19 ]
  let divisible_by_5 := 1
  let total_primes := 8
  (divisible_by_5:ℚ ÷ total_primes * 100) = 12.5 := by 
  sorry

end percent_primes_less_than_20_divisible_by_5_l524_524588


namespace distance_between_vertices_hyperbola_l524_524242

-- Definitions as per conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Statement of the problem in Lean
theorem distance_between_vertices_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → ∃ d : ℝ, d = 8 :=
by
  intros x y h
  use 8
  sorry

end distance_between_vertices_hyperbola_l524_524242


namespace mrs_hilt_water_fountain_trips_l524_524489

theorem mrs_hilt_water_fountain_trips (d : ℕ) (t : ℕ) (n : ℕ) 
  (h1 : d = 30) 
  (h2 : t = 120) 
  (h3 : 2 * d * n = t) : 
  n = 2 :=
by
  -- Proof omitted
  sorry

end mrs_hilt_water_fountain_trips_l524_524489


namespace smallest_even_number_l524_524542

theorem smallest_even_number (n1 n2 n3 n4 n5 n6 n7 : ℤ) 
  (h_sum_seven : n1 + n2 + n3 + n4 + n5 + n6 + n7 = 700)
  (h_sum_first_three : n1 + n2 + n3 > 200)
  (h_consecutive : n2 = n1 + 2 ∧ n3 = n2 + 2 ∧ n4 = n3 + 2 ∧ n5 = n4 + 2 ∧ n6 = n5 + 2 ∧ n7 = n6 + 2) :
  n1 = 94 := 
sorry

end smallest_even_number_l524_524542


namespace max_students_gave_away_balls_more_l524_524652

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l524_524652


namespace angles_of_polyhedra_equal_l524_524809

theorem angles_of_polyhedra_equal (n : ℕ) (A B : ℝ^3) 
  (A_i B_i : fin n → ℝ^3) 
  (hA_in_convex : ∀ (x : ℝ^3), x ∈ convex_hull (set.range A_i) → (∃ a, x = A + a))
  (hB_in_convex : ∀ (x : ℝ^3), x ∈ convex_hull (set.range B_i) → (∃ b, x = B + b))
  (h_angles_le : ∀ i j : fin n, ∠ (A_i i) A (A_i j) ≤ ∠ (B_i i) B (B_i j) ) :
  ∀ i j : fin n, ∠ (A_i i) A (A_i j) = ∠ (B_i i) B (B_i j) := 
by sorry

end angles_of_polyhedra_equal_l524_524809


namespace line_circle_no_intersection_l524_524329

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  -- Sorry to skip the actual proof
  sorry

end line_circle_no_intersection_l524_524329


namespace eldorado_license_plates_count_l524_524804

def is_vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def valid_license_plates_count : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits

theorem eldorado_license_plates_count : valid_license_plates_count = 338000 := by
  sorry

end eldorado_license_plates_count_l524_524804


namespace segment_half_difference_of_bases_l524_524910

-- Defining the trapezoid and its properties
variables {A B C D K : Type*} [EuclideanGeometry : convex ℝ] 
(trapezoid : EuclideanSpace ℝ)
(baseAD baseBC : Segment)
(h_sum_angles : ∀ {α β : Real}, α + β = 90)

-- Hypothesize that the given trapezoid has bases AD and BC and the sum of the base angles is 90 degrees.
def is_trapezoid (trapezoid : EuclideanSpace ℝ) : Prop :=
∃ (A B C D : EuclideanSpace ℝ),
  Segment A B ∥ Segment C D ∧ Segment B C - Segment A D

-- Define the function calculating the segment connecting the midpoints of the bases.
noncomputable def segment_connecting_midpoints (baseAD baseBC : Segment) : EuclideanSpace ℝ :=
(CK / 2 * (AD - BC))

-- The theorem to be proved
theorem segment_half_difference_of_bases
  (h_trapezoid : is_trapezoid trapezoid)
  (h_angles : h_sum_angles = 90) :
  segment_connecting_midpoints baseAD baseBC = (1 / 2) * (AD - BC) :=
sorry

end segment_half_difference_of_bases_l524_524910


namespace average_speed_of_train_l524_524159

theorem average_speed_of_train (x : ℝ) (h₀ : x > 0) :
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  avg_speed = 48 := by
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  sorry

end average_speed_of_train_l524_524159


namespace least_three_digit_with_factors_l524_524949

-- Define the three prime factors
def a := 3
def b := 5
def c := 7

-- Define the candidate for the least three-digit positive integer
def candidate := 105

-- Define the properties of the candidate
def is_least_three_digit_with_factors (n : ℕ) : Prop :=
  (a ∣ n) ∧ (b ∣ n) ∧ (c ∣ n) ∧ (100 ≤ n) ∧ (n < 1000)

-- Prove that 105 is the least three-digit positive integer satisfying these properties
theorem least_three_digit_with_factors : is_least_three_digit_with_factors candidate := 
by {
  -- Proof of this theorem is out of scope, hence replaced with sorry
  sorry,
}

end least_three_digit_with_factors_l524_524949


namespace line_circle_no_intersection_l524_524407

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 → false :=
by
  intro x y h
  obtain ⟨hx, hy⟩ := h
  have : y = 3 - (3 / 4) * x := by linarith
  rw [this] at hy
  have : x^2 + ((3 - (3 / 4) * x)^2) = 4 := hy
  simp at this
  sorry

end line_circle_no_intersection_l524_524407


namespace percent_prime_divisible_by_5_l524_524582

def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]
def primes_divisible_by_5 := [p in primes_less_than_20 | p % 5 = 0]

theorem percent_prime_divisible_by_5 :
  (primes_divisible_by_5.length / primes_less_than_20.length * 100 = 12.5) :=
by
  sorry

end percent_prime_divisible_by_5_l524_524582


namespace find_suspects_l524_524803

variable (A B C D : Prop)

-- Definitions for the conditions
def involved (x y : Prop) : Prop := x ∧ y

def condition1 : Prop := ∃ x y, involved x y

def condition2 : Prop := A → ¬C

def condition3 : Prop := B → D

def condition4 : Prop := ¬C → ¬D

-- Statement to be proven
theorem find_suspects : 
  involved C D ∧ 
  condition1 ∧ 
  condition2 ∧ 
  condition3 ∧ 
  condition4 → involved C D := 
by
  sorry

end find_suspects_l524_524803


namespace number_of_sets_even_n_l524_524838

theorem number_of_sets_even_n (n : ℕ) (h_even : n % 2 = 0) (h_n_ge_2 : 2 ≤ n) :
  ∃ (M : set ℕ), (M ⊆ {x : ℕ | x < n}) ∧ 
    (∀ a ∈ M, (n - a) ∈ M) ∧ 
    (|M| = 2^(n/2) - 1) :=
sorry

end number_of_sets_even_n_l524_524838


namespace wall_length_is_800_l524_524132

def brick_volume : ℝ := 50 * 11.25 * 6
def total_brick_volume : ℝ := 3200 * brick_volume
def wall_volume (x : ℝ) : ℝ := x * 600 * 22.5

theorem wall_length_is_800 :
  ∀ (x : ℝ), total_brick_volume = wall_volume x → x = 800 :=
by
  intros x h
  sorry

end wall_length_is_800_l524_524132


namespace polyhedron_with_12_edges_l524_524170

def prism_edges (n : Nat) : Nat :=
  3 * n

def pyramid_edges (n : Nat) : Nat :=
  2 * n

def Quadrangular_prism : Nat := prism_edges 4
def Quadrangular_pyramid : Nat := pyramid_edges 4
def Pentagonal_pyramid : Nat := pyramid_edges 5
def Pentagonal_prism : Nat := prism_edges 5

theorem polyhedron_with_12_edges :
  (Quadrangular_prism = 12) ∧
  (Quadrangular_pyramid ≠ 12) ∧
  (Pentagonal_pyramid ≠ 12) ∧
  (Pentagonal_prism ≠ 12) := by
  sorry

end polyhedron_with_12_edges_l524_524170


namespace geometric_properties_l524_524273

-- Given structures to represent points, lines, and properties
section GeometricProof

variables (A B C D E F R P M N : Point)
variable (circle_inscribed : ∃ circle, QuadrilateralInscribedInCircle A B C D circle)
variable (extension_AD_BC_intersect_E : LineIntersection (LineThrough A D) (LineThrough B C) E)
variable (extension_AB_DC_intersect_F : LineIntersection (LineThrough A B) (LineThrough D C) F)
variable (internal_bisectors_intersect_R : InternalAngleBisectorsIntersect E F R)
variable (external_bisectors_intersect_P : ExternalAngleBisectorsIntersect E F P)
variable (midpoints_M_N : MidpointsOfDiagonals M N A C B D)

-- Theorem Statement
theorem geometric_properties :
  (Perpendicular (LineThrough R E) (LineThrough R F)) ∧
  (Perpendicular (LineThrough P E) (LineThrough P F)) ∧
  (HarmonicSet M N R P) :=
sorry

end GeometricProof

end geometric_properties_l524_524273


namespace index_cards_students_l524_524815

theorem index_cards_students
  (packs_per_student : ℕ)
  (class_count : ℕ)
  (total_packs : ℕ)
  (packs_per_student_eq : packs_per_student = 2)
  (class_count_eq : class_count = 6)
  (total_packs_eq : total_packs = 360) :
  (total_packs / class_count) / packs_per_student = 30 :=
by
  rw [class_count_eq, packs_per_student_eq, total_packs_eq]
  norm_num
  sorry

end index_cards_students_l524_524815


namespace S5_eq_0_l524_524083

-- Define the sum of an arithmetic sequence
def S (n : ℕ) : ℕ

-- Given conditions
axiom S2_eq_3 : S 2 = 3
axiom S3_eq_3 : S 3 = 3

-- Proving the desired result
theorem S5_eq_0 : S 5 = 0 :=
by {
  sorry
}

end S5_eq_0_l524_524083


namespace distinct_roots_magnitude_d_l524_524672

theorem distinct_roots_magnitude_d (d : ℂ) :
  (∀ x : ℂ, (x^2 - 3 * x + 3) * (x^2 - d * x + 5) * (x^2 - 5 * x + 10) = 0 ↔ 
    x ∈ {x | (x^2 - 3 * x + 3) = 0} ∪ {x | (x^2 - d * x + 5) = 0} ∪ {x | (x^2 - 5 * x + 10) = 0}) →
  (∃ s : Finset ℂ, s.card = 5 ∧ ∀ x : ℂ, (x^2 - 3 * x + 3) * (x^2 - d * x + 5) * (x^2 - 5 * x + 10) = 0 → x ∈ s) →
  |d| = Real.sqrt 22 :=
by
  sorry

end distinct_roots_magnitude_d_l524_524672


namespace pints_in_5_liters_l524_524735

-- Define the condition based on the given conversion factor from liters to pints
def conversion_factor : ℝ := 2.1

-- The statement we need to prove
theorem pints_in_5_liters : 5 * conversion_factor = 10.5 :=
by sorry

end pints_in_5_liters_l524_524735


namespace intersection_complement_l524_524006

open Set

variable (U : Type) [Nonempty U] [Encodable U]
variable (P Q : Set ℝ)

noncomputable def complement (U : Set ℝ) (s : Set ℝ) := U \ s

theorem intersection_complement (P Q : Set ℝ) (U : Set ℝ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5}) (hU : U = univ) : P ∩ (complement U Q) = {1, 2} :=
by
  sorry

end intersection_complement_l524_524006


namespace relationship_among_abc_l524_524292

noncomputable def a : ℝ := (1 / 2) ^ (1 / 2)
noncomputable def b : ℝ := (1 / 3) ^ (-2)
noncomputable def c : ℝ := Real.logBase (1 / 2) 2

theorem relationship_among_abc : b > a ∧ a > c := by
  sorry

end relationship_among_abc_l524_524292


namespace range_of_k_one_integer_solution_l524_524310

theorem range_of_k_one_integer_solution :
  { k : ℝ // (∀ x : ℝ, x^2 - 2*x - 8 > 0 ↔ x < -2 ∨ x > 4) ∧ 
              (∀ x : ℝ, 2*x^2 + (2*k + 7)*x + 7*k < 0) } = 
  [| k in ((Ioo (-5 : ℝ) (3 : ℝ)) ∪ (Ioc (4 : ℝ) (5 : ℝ))) |] :=
by sorry

end range_of_k_one_integer_solution_l524_524310


namespace approx_positive_numbers_l524_524835

-- Define the given sequence with parameter p
def sequence (p : ℝ) : List ℝ := List.range (100 : Nat) >>= λ n, [p ^ n, p ^ (-n : ℤ)]

-- Define the core approximation problem
theorem approx_positive_numbers (p : ℝ) (hp : p > 0) : 
  (∀ ε > 0, ∀ x > 0, ∃ s : Finset ℝ, (s.sum id ∈ sequence p) ∧ |s.sum id - x| < ε) ↔ 
    (1 / 2 ≤ p ∧ p < 1 ∨ 1 < p ∧ p ≤ 2) := 
by
sorry  -- Proof omitted

end approx_positive_numbers_l524_524835


namespace percent_primes_less_than_20_divisible_by_5_l524_524585

theorem percent_primes_less_than_20_divisible_by_5 :
  let primes := [ 2, 3, 5, 7, 11, 13, 17, 19 ]
  let divisible_by_5 := 1
  let total_primes := 8
  (divisible_by_5:ℚ ÷ total_primes * 100) = 12.5 := by 
  sorry

end percent_primes_less_than_20_divisible_by_5_l524_524585


namespace problem_n6_l524_524125

def is_fibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, n = fibonacci k

theorem problem_n6 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  is_fibonacci ((a + b) / gcd a b ^ 2) :=
by
  sorry

end problem_n6_l524_524125


namespace solve_for_I_l524_524449

def V : ℂ := 2 + 3 * complex.I
def Z : ℂ := 2 - complex.I
noncomputable def I : ℂ := V / Z

theorem solve_for_I : I = (1 / 5) + (8 / 5) * complex.I :=
by
  sorry

end solve_for_I_l524_524449


namespace log_10_of_2_bounds_l524_524554

theorem log_10_of_2_bounds (a b : ℝ)
  (h1 : 10 ^ 2 = 100)
  (h2 : 10 ^ 3 = 1000)
  (h3 : 2 ^ 7 = 128)
  (h4 : 2 ^ 10 = 1024)
  (ha : a = 2 / 7)
  (hb : b = 3 / 10) :
  a < log 10 2 ∧ log 10 2 < b :=
sorry

end log_10_of_2_bounds_l524_524554


namespace triangular_25_l524_524102

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l524_524102


namespace parabola_equation_l524_524145

theorem parabola_equation {p x1 x2 y1 y2 : ℝ} (h1 : p > 0) (h2 : x1 + x2 = 2) (h3 : |(P - Q)| = 4) :
    y^2 = 4x :=
by
  sorry

end parabola_equation_l524_524145


namespace kenneth_past_finish_line_when_biff_finishes_l524_524178

-- Given conditions:
def race_distance : ℕ := 500
def biff_speed : ℕ := 50
def kenneth_speed : ℕ := 51

-- The statement to prove:
theorem kenneth_past_finish_line_when_biff_finishes :
  let time_biff_to_finish := race_distance / biff_speed in
  let distance_kenneth_in_that_time := kenneth_speed * time_biff_to_finish in
  distance_kenneth_in_that_time - race_distance = 10 :=
by
  sorry

end kenneth_past_finish_line_when_biff_finishes_l524_524178


namespace mixture_contains_67_5_percent_water_volume_of_mixture_is_50_percent_P_l524_524117

variable (P Q : Type) 
variable (volumeP volumeQ : ℝ)
variable (percentageLemonadeP percentageWaterP percentageLemonadeQ percentageWaterQ : ℝ)
noncomputable def mixtureVolume (x y : ℝ) : ℝ := x + y

noncomputable def carbonatedWaterMix (x y volumeP volumeQ : ℝ) : ℝ := (0.80 * volumeP * x + 0.55 * volumeQ * y) / (volumeP * x + volumeQ * y)

theorem mixture_contains_67_5_percent_water :
  percentageLemonadeP = 20 / 100 → percentageWaterP = 80 / 100 → 
  percentageLemonadeQ = 45 / 100 → percentageWaterQ = 55 / 100 → 
  carbonatedWaterMix P Q volumeP volumeQ = 67.5 / 100 → 
  volumeP = volumeQ := 
  by {
    intros h1 h2 h3 h4 h5,
    sorry
  }

theorem volume_of_mixture_is_50_percent_P : 
  volumeP = volumeQ → 
  (volumeP / (mixtureVolume volumeP volumeQ)) * 100 = 50 := 
  by {
    intros h,
    sorry
  }

end mixture_contains_67_5_percent_water_volume_of_mixture_is_50_percent_P_l524_524117


namespace part1_part2_l524_524001

-- Define the function f_n
def f (n : ℕ) (x : ℝ) : ℝ :=
  (x ^ (n + 1) - x^(- (n + 1))) / (x - x^(-1))

-- Define y
def y (x : ℝ) : ℝ :=
  x + 1 / x

-- Prove the first part of the problem
theorem part1 (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) (h4 : n > 1) :
  f (n + 1) x = y x * f n x - f (n - 1) x :=
sorry

-- Prove the second part of the problem using mathematical induction
theorem part2 (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  f n x = (if n % 2 = 0 then
          ∑ i in finset.range (n / 2 + 1), (-1)^i * (nat.choose (n-i) i) * (y x)^(n - 2 * i)
        else
          ∑ i in finset.range ((n + 1) / 2), (-1)^i * (nat.choose (n-i) i) * (y x)^(n - 2 * i)) :=
sorry

end part1_part2_l524_524001


namespace max_of_x_l524_524832

theorem max_of_x (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 10) : x ≤ 3 := by
  sorry

end max_of_x_l524_524832


namespace range_of_a_l524_524762

def A (a : ℝ) := ({-1, 0, a} : Set ℝ)
def B := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B ≠ ∅) : 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l524_524762


namespace outfits_count_l524_524596

theorem outfits_count (RedShirts GreenShirts BlueShirts Pants GreenHats RedHats BlueHats : ℕ) 
  (h1 : RedShirts = 4) (h2 : GreenShirts = 4) (h3 : BlueShirts = 4)
  (h4 : Pants = 6) (h5 : GreenHats = 8) (h6 : RedHats = 8) (h7 : BlueHats = 8) :
  (RedShirts * (GreenHats + BlueHats) + GreenShirts * (RedHats + BlueHats) + BlueShirts * (RedHats + GreenHats)) * Pants = 1152 := by
  simp [h1, h2, h3, h4, h5, h6, h7]
  sorry

end outfits_count_l524_524596


namespace jelly_bean_matching_probability_l524_524165

theorem jelly_bean_matching_probability :
  let Abe_jelly_beans := [2, 3] : List ℕ, -- 2 green, 3 red
      Bob_jelly_beans := [2, 2, 3] : List ℕ, -- 2 green, 2 yellow, 3 red
      total_beans := List.sum Abe_jelly_beans, -- 5 for Abe
      total_beans_bob := List.sum Bob_jelly_beans, -- 7 for Bob
      p_abe_green := (Abe_jelly_beans.head!) / total_beans,
      p_bob_green := (Bob_jelly_beans.head!) / total_beans_bob,
      p_abe_red := (Abe_jelly_beans.tail!.head!) / total_beans,
      p_bob_red := (Bob_jelly_beans.tail!.tail!.head!) / total_beans_bob
  in (p_abe_green * p_bob_green + p_abe_red * p_bob_red) = (13 / 35) := by
  sorry

end jelly_bean_matching_probability_l524_524165


namespace max_x_for_lcm_120_l524_524068

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem max_x_for_lcm_120 (x : ℕ) (h : lcm (lcm x 8) 12 = 120) : x ≤ 120 :=
by
-- sorry proof steps not required
sorry

end max_x_for_lcm_120_l524_524068


namespace trapezoid_problem_l524_524931

noncomputable def p : ℕ := 40
noncomputable def q : ℕ := 1

theorem trapezoid_problem (EF FG GH HE : ℕ) (hEF : EF = 120) (hFG : FG = 50) (hGH : GH = 25) (hHE : HE = 70)
  (hParallel : EF ∥ GH) (Q : ℝ) (hQ_on_EF : Q ∈ set.Icc 0 120)
  (circle_center_Q : ∃ (circle : { center : ℝ // center ∈ set.Icc 0 120 }), circle.center = Q) 
  (hTangent_to_FG : ∃ (circle : { center : ℝ // center ∈ set.Icc 0 120 }), ∃ (x : ℝ), circle.center = Q ∧ x^2 - 120*x + 8400 = 0) :
  EQ = 40 ∧ p + q = 41 :=
begin
  sorry
end

end trapezoid_problem_l524_524931


namespace initial_number_of_girls_l524_524130

-- Define variables and conditions
variables {q : ℚ} (initial_girls : ℚ) (final_girls : ℚ) (initial_people : ℚ) (final_people : ℚ)

-- Initial number of people in the group
def initial_people := 0.5 * q 

-- Initial number of girls
def initial_girls := 0.5 * q 

-- Number of people after 3 boys leave and 1 girl joins
def final_people := q - 2

-- Number of girls after 3 boys leave and 1 girl joins
def final_girls := 0.5 * q + 1

-- Given condition: Girls are now 60% of the group
theorem initial_number_of_girls (h1 : initial_girls = 0.5 * q) 
  (h2 : final_people = q - 2) 
  (h3 : final_girls = 0.5 * q + 1) 
  (h4 : 0.6 * final_people = final_girls) :
  initial_girls = 11 := 
by
  sorry

end initial_number_of_girls_l524_524130


namespace tangent_line_equation_l524_524891

theorem tangent_line_equation (x y : ℝ) : 
  let curve := λ x, x^3 in
  let derivative := λ x, 3*x^2 in
  let point := (2, 8) in
  let slope := derivative 2 in
  y - point.2 = slope * (x - point.1) → 
  y = 12*x - 16 :=
sorry

end tangent_line_equation_l524_524891


namespace sum_first_10_terms_arithmetic_sequence_l524_524313

def a : ℕ → ℕ
| 0     := 3
| (n+1) := a n + 2

def S (n : ℕ) := (n + 1) * (a 0 + a n) / 2

theorem sum_first_10_terms_arithmetic_sequence : S 9 = 120 := 
by {
  -- This is where the proof would go
  sorry
}

end sum_first_10_terms_arithmetic_sequence_l524_524313


namespace problem_statement_l524_524780

-- Defining the propositions p and q as Boolean variables
variables (p q : Prop)

-- Assume the given conditions
theorem problem_statement (hnp : ¬¬p) (hnpq : ¬(p ∧ q)) : p ∧ ¬q :=
by {
  -- Derived steps to satisfy the conditions are implicit within this scope
  sorry
}

end problem_statement_l524_524780


namespace sheepdog_catches_sheep_in_73_seconds_l524_524843

noncomputable def sheep_speed : ℝ := 18
noncomputable def sheepdog_speed : ℝ := 30
noncomputable def rest_time_per_100_feet : ℝ := 3
noncomputable def distance_per_rest : ℝ := 100
noncomputable def initial_distance_between_them : ℝ := 480

theorem sheepdog_catches_sheep_in_73_seconds :
  ∃ t : ℝ, t = 73 ∧ (
    ∀ d1 v1 t1 d2 v2 t2 d,
      d1 = initial_distance_between_them →
      v1 = sheep_speed →
      v2 = sheepdog_speed →
      t1 = rest_time_per_100_feet →
      d2 = distance_per_rest →
      d = (t - (t / ((d / d2) * (t1 + (d2 / v2)))) * t1) * v2 ∧
      d1 + d1 ≤ d
) := sorry

end sheepdog_catches_sheep_in_73_seconds_l524_524843


namespace difference_between_numbers_l524_524543

-- Given definitions based on conditions
def sum_of_two_numbers (x y : ℝ) : Prop := x + y = 15
def difference_of_two_numbers (x y : ℝ) : Prop := x - y = 10
def difference_of_squares (x y : ℝ) : Prop := x^2 - y^2 = 150

theorem difference_between_numbers (x y : ℝ) 
  (h1 : sum_of_two_numbers x y) 
  (h2 : difference_of_two_numbers x y) 
  (h3 : difference_of_squares x y) :
  x - y = 10 :=
by
  sorry

end difference_between_numbers_l524_524543


namespace solve_floor_equation_l524_524879

-- Define positive integers
variable (m n : ℕ)
variable pos_m : 0 < m
variable pos_n : 0 < n

-- Define the main theorem
theorem solve_floor_equation : 
  ∃ m n : ℕ, 0 < m ∧ 0 < n ∧ 
  (⟨m, pos_m⟩^2 / n.floor + ⟨n, pos_n⟩^2 / m.floor = 
  (⟨m, pos_m⟩ / n + ⟨n, pos_n⟩ / m).floor + m * n) ∧ 
  m = 2 ∧ n = 1 :=
begin
  sorry
end

end solve_floor_equation_l524_524879


namespace cot_45_eq_1_l524_524225

def cot (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem cot_45_eq_1 : cot (Real.pi / 4) = 1 :=
by
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [cot, h1]
  norm_num
  sorry

end cot_45_eq_1_l524_524225


namespace sum_of_odd_numbers_from_1_to_200_l524_524705

theorem sum_of_odd_numbers_from_1_to_200 : ∑ i in Finset.filter (λ x => x % 2 = 1) (Finset.range 201) = 10000 := by
  sorry

end sum_of_odd_numbers_from_1_to_200_l524_524705


namespace percentage_primes_divisible_by_five_l524_524576

def primes := [2, 3, 5, 7, 11, 13, 17, 19]
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0

theorem percentage_primes_divisible_by_five : 
  (∃ count, count = (list.filter divisible_by_five primes).length) 
  →  (list.length primes) = 8
  →  count = 1
  → (count : ℝ) / 8 * 100 = 12.5 :=
by
  sorry

end percentage_primes_divisible_by_five_l524_524576


namespace brad_more_pages_than_greg_l524_524767

def greg_pages_first_week : ℕ := 7 * 18
def greg_pages_next_two_weeks : ℕ := 14 * 22
def greg_total_pages : ℕ := greg_pages_first_week + greg_pages_next_two_weeks

def brad_pages_first_5_days : ℕ := 5 * 26
def brad_pages_remaining_12_days : ℕ := 12 * 20
def brad_total_pages : ℕ := brad_pages_first_5_days + brad_pages_remaining_12_days

def total_required_pages : ℕ := 800

theorem brad_more_pages_than_greg : brad_total_pages - greg_total_pages = 64 :=
by
  sorry

end brad_more_pages_than_greg_l524_524767


namespace sum_of_integral_c_l524_524704

theorem sum_of_integral_c (c : ℤ) : 
  (c ≤ 30) ∧ (∃ (k : ℤ), k^2 = 64 + 4 * c) → (∑ (i : ℤ) in {-16, -15, -12, -7, 0, 9, 20}.to_finset, i) = -11 :=
by 
  sorry

end sum_of_integral_c_l524_524704


namespace tickets_sold_l524_524147

def movie_theater_capacity : ℕ := 50
def ticket_price : ℕ := 8
def loss : ℕ := 208

theorem tickets_sold : ∀ (capacity price loss revenue tickets_sold), 
  capacity = 50 →
  price = 8 →
  loss = 208 →
  revenue = capacity * price - loss →
  tickets_sold = revenue / price →
  tickets_sold = 24 :=
by
  intros capacity price loss revenue tickets_sold h_capacity h_price h_loss h_revenue h_calc
  rw [h_capacity, h_price, h_loss] at h_revenue
  simp at h_revenue
  rw h_revenue at h_calc
  simp [h_calc]
  sorry

end tickets_sold_l524_524147


namespace shortest_distance_correct_l524_524082

noncomputable def shortest_distance : ℝ :=
  let C := curve := λ x, real.log (2 * x - 1),
      L := λ x, 2 * x - 8,
      tangent_slope := λ x, 2 / (2 * x - 1),
      m := 1,
      n := real.log 2,
      point := (m, 0)
  in real.abs (2 * 1 + 8) / real.sqrt (2^2 + (-1)^2)

theorem shortest_distance_correct :
  shortest_distance = 2 * real.sqrt 5 :=
by
  sorry

end shortest_distance_correct_l524_524082


namespace exists_integer_coefficient_polynomial_l524_524865

theorem exists_integer_coefficient_polynomial (n : ℕ) (hn : 0 < n) :
  ∃ (p : Polynomial ℤ),
  ∃ (k : Fin n → ℕ),
    (∀ i : Fin n, p.eval (i + 1) = 2 ^ k i) ∧
    (∀ i j : Fin n, i ≠ j → 2 ^ k i ≠ 2 ^ k j) :=
sorry

end exists_integer_coefficient_polynomial_l524_524865


namespace sufficient_condition_for_parallel_l524_524827

variables {Line : Type} {Plane : Type} [nonempty Line] [nonempty Plane]

variables (a b c : Line) (α β : Plane)

def non_overlapping_lines (l1 l2 l3 : Line) : Prop := 
  ¬ (∃ p : Point, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3)

def non_overlapping_planes (p1 p2 : Plane) : Prop := 
  ¬ (∃ l : Line, l ⊆ p1 ∧ l ⊆ p2)

def parallel_to (l : Line) (p : Plane) : Prop := 
  ∀ (x y : Point), x ∈ l → y ∈ l → ∀ z : Point, z ∈ p → 
    (vector.from_points x y) ∥ (vector.from_points x z)

def subset_of (l : Line) (p : Plane) : Prop := 
  ∀ (x : Point), x ∈ l → x ∈ p

def intersection (p1 p2 : Plane) : Line := 
  sorry -- Assuming this is a function that returns the line of intersection 

axiom a_parallel_to_alpha : parallel_to a α
axiom alpha_intersect_beta_eq_b : intersection α β = b
axiom a_subset_of_beta : subset_of a β

theorem sufficient_condition_for_parallel : a_parallel_to_alpha →
  alpha_intersect_beta_eq_b →
  a_subset_of_beta →
  (parallels a b) := 
sorry

end sufficient_condition_for_parallel_l524_524827


namespace cube_points_l524_524647

theorem cube_points (A B C D E F : ℕ) 
  (h1 : A + B = 13)
  (h2 : C + D = 13)
  (h3 : E + F = 13)
  (h4 : A + C + E = 16)
  (h5 : B + D + E = 24) :
  F = 6 :=
by
  sorry  -- Proof to be filled in by the user

end cube_points_l524_524647


namespace maximize_profit_l524_524621

-- Definitions
def x (m k : ℝ) : ℝ := 3 - k / (m + 1)
def profit (m : ℝ) : ℝ := 29 - 16 / (m + 1) - m

-- Given conditions in the problem statement
def k_value : ℝ := 2

-- Theorem statement
theorem maximize_profit :
  (∀ m : ℝ, 0 ≤ m → profit m ≤ 21) ∧ profit 3 = 21 :=
by
  sorry

end maximize_profit_l524_524621


namespace jellybeans_initial_amount_l524_524209

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_initial_amount_l524_524209


namespace tangent_slope_at_zero_l524_524907

def f (x : ℝ) : ℝ := Real.sin x + 2 * x

theorem tangent_slope_at_zero : deriv f 0 = 3 := 
by 
  sorry

end tangent_slope_at_zero_l524_524907


namespace product_of_real_parts_complex_solutions_l524_524201

noncomputable def complex_eq (x : ℂ) : Prop := x^2 + 2 * x + complex.I = 0

noncomputable def real_part_product : ℂ := (1 - real.sqrt 2) / 2

theorem product_of_real_parts_complex_solutions :
  ∀ x1 x2 : ℂ, complex_eq x1 → complex_eq x2 → ( x1.re * x2.re = real_part_product ) :=
by sorry

end product_of_real_parts_complex_solutions_l524_524201


namespace jellybeans_original_count_l524_524218

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l524_524218


namespace Ingrid_cookie_percentage_l524_524457

theorem Ingrid_cookie_percentage : 
  let irin_ratio := 9.18
  let ingrid_ratio := 5.17
  let nell_ratio := 2.05
  let kim_ratio := 3.45
  let linda_ratio := 4.56
  let total_cookies := 800
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio + kim_ratio + linda_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let ingrid_cookies := ingrid_share * total_cookies
  let ingrid_percentage := (ingrid_cookies / total_cookies) * 100
  ingrid_percentage = 21.25 :=
by
  sorry

end Ingrid_cookie_percentage_l524_524457


namespace Jeanette_juggles_21_objects_by_end_of_5th_week_l524_524812

def objects_juggled_at_end_of_week (start_objects : ℕ) (weeks : ℕ) : ℕ :=
  let one_session_more_objects := start_objects + 1
  let two_session_more_objects := one_session_more_objects + 1
  let end_of_week := two_session_more_objects
  if weeks = 1 then end_of_week else
    objects_juggled_at_end_of_week (end_of_week + 2) (weeks - 1)

theorem Jeanette_juggles_21_objects_by_end_of_5th_week :
  objects_juggled_at_end_of_week 3 5 = 21 :=
by
  sorry

end Jeanette_juggles_21_objects_by_end_of_5th_week_l524_524812


namespace find_k_l524_524917

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : |3 * (k^2 - 9) - 2 * (4 * k - 15) + 2 * (12 - 5 * k)| = 20) : k = 4 := by
  sorry

end find_k_l524_524917


namespace max_students_unfamiliar_l524_524999

theorem max_students_unfamiliar (n : ℕ)
  (h1 : 100 ∃ subjects)
  (h2 : ∀ A_i A_j, (A_i ≠ A_j) → (count (λ k, A_i k ≠ A_j k) (range 100) ≥ 51)) :
  n ≤ 34 :=
  sorry

end max_students_unfamiliar_l524_524999


namespace compute_expression_l524_524188

theorem compute_expression : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end compute_expression_l524_524188


namespace intersection_points_hyperbola_l524_524255

theorem intersection_points_hyperbola (t : ℝ) :
  let x := (3 * t^2 + 3) / (t^2 - 1),
      y := (3 * t) / (t^2 - 1)
  in (x^2 / 9) - (y^2 / (9 / 4)) = 1 :=
by
  sorry

end intersection_points_hyperbola_l524_524255


namespace coin_flip_probability_l524_524556

theorem coin_flip_probability :
  let total_flips := 12
  let first_flip_tail : Bool := true
  let remaining_flips := total_flips - 1
  let success_events := 9
  first_flip_tail = true →
  (nat.binomial remaining_flips success_events) / (2 ^ remaining_flips : ℝ) = 55 / 2048 :=
by {
  sorry
}

end coin_flip_probability_l524_524556


namespace variance_of_red_ball_draws_l524_524513

noncomputable def variance_red_ball_draws : ℚ :=
let n := 3
let p := (2 : ℚ) / 3
in n * p * (1 - p)

theorem variance_of_red_ball_draws :
  variance_red_ball_draws = (2 : ℚ) / 3 :=
by
  -- We assume the conditions in the problem
  let n := 3
  let p := (2 : ℚ) / 3
  have h1 : variance_red_ball_draws = n * p * (1 - p) := rfl
  rw [h1]
  sorry

end variance_of_red_ball_draws_l524_524513


namespace percent_prime_divisible_by_5_l524_524579

def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]
def primes_divisible_by_5 := [p in primes_less_than_20 | p % 5 = 0]

theorem percent_prime_divisible_by_5 :
  (primes_divisible_by_5.length / primes_less_than_20.length * 100 = 12.5) :=
by
  sorry

end percent_prime_divisible_by_5_l524_524579


namespace max_value_of_k_l524_524818

-- Define the set A with n elements and the subsets
variables {α : Type*} (A : finset α) (n : ℕ)
variable (hA : A.card = n)

-- Define the condition for the subsets
variable (A_subs : finset (finset α))
variable (h_subsets : ∀ {B1 B2 : finset α}, B1 ∈ A_subs → B2 ∈ A_subs → B1 ≠ B2 → B1 ⊆ B2 ∨ B2 ⊆ B1 ∨ disjoint B1 B2)

-- The theorem statement to be proved
theorem max_value_of_k (A n : ℕ) (hA : A.card = n) (A_subs : finset (finset α)) (h_subsets : ∀ {B1 B2 : finset α}, B1 ∈ A_subs → B2 ∈ A_subs → B1 ≠ B2 → B1 ⊆ B2 ∨ B2 ⊆ B1 ∨ disjoint B1 B2) :
  A_subs.card ≤ 2 * n - 1 :=
sorry -- Proof to be provided

end max_value_of_k_l524_524818


namespace distance_between_vertices_hyperbola_l524_524241

-- Definitions as per conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Statement of the problem in Lean
theorem distance_between_vertices_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → ∃ d : ℝ, d = 8 :=
by
  intros x y h
  use 8
  sorry

end distance_between_vertices_hyperbola_l524_524241


namespace union_of_sets_l524_524466

open Set

variable (a b : ℕ)

noncomputable def M : Set ℕ := {3, 2 * a}
noncomputable def N : Set ℕ := {a, b}

theorem union_of_sets (h : M a ∩ N a b = {2}) : M a ∪ N a b = {1, 2, 3} :=
by
  -- skipped proof
  sorry

end union_of_sets_l524_524466


namespace polygon_centroid_disproof_l524_524052

axiom exists_pentagon_counterexample :
∃ (P : Polygon) (vertices_centroid plate_centroid : Point),
  (P.shape = "pentagon") ∧
  (vertices_centroid ≠ plate_centroid)

theorem polygon_centroid_disproof :
  ¬ ∀ (P : Polygon), P.centroid = P.vertices_centroid :=
begin
  obtain ⟨P, vertices_centroid, plate_centroid, P_shape, centroid_diff⟩ := exists_pentagon_counterexample,
  exact ⟨P, vertices_centroid, plate_centroid, P_shape, centroid_diff⟩
end

end polygon_centroid_disproof_l524_524052


namespace quadrilateral_is_parallelogram_l524_524027

theorem quadrilateral_is_parallelogram 
  (A B C D K L M N K₁ L₁ M₁ N₁ : Point) 
  (circle_inscribed_in_Quad : IsCircleInscribedInQuadrilateral A B C D)
  (ext_angle_bisectors_A_B : ExternalAngleBisectorsIntersect A B K)
  (ext_angle_bisectors_B_C : ExternalAngleBisectorsIntersect B C L)
  (ext_angle_bisectors_C_D : ExternalAngleBisectorsIntersect C D M)
  (ext_angle_bisectors_D_A : ExternalAngleBisectorsIntersect D A N)
  (orthocenter_ΔABK : OrthocenterTriangle A B K K₁)
  (orthocenter_ΔBCL : OrthocenterTriangle B C L L₁)
  (orthocenter_ΔCDM : OrthocenterTriangle C D M M₁)
  (orthocenter_ΔDAN : OrthocenterTriangle D A N N₁) 
  : IsParallelogram K₁ L₁ M₁ N₁ :=
sorry

end quadrilateral_is_parallelogram_l524_524027


namespace line_circle_no_intersection_l524_524339

/-- The equation of the line is given by 3x + 4y = 12 -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The equation of the circle is given by x^2 + y^2 = 4 -/
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The proof we need to show is that there are no real points (x, y) that satisfy both the line and the circle equations -/
theorem line_circle_no_intersection : ¬ ∃ (x y : ℝ), line x y ∧ circle x y :=
by {
  sorry
}

end line_circle_no_intersection_l524_524339


namespace largest_three_digit_sum_fifteen_l524_524944

theorem largest_three_digit_sum_fifteen : ∃ (a b c : ℕ), (a = 9 ∧ b = 6 ∧ c = 0 ∧ 100 * a + 10 * b + c = 960 ∧ a + b + c = 15 ∧ a < 10 ∧ b < 10 ∧ c < 10) := by
  sorry

end largest_three_digit_sum_fifteen_l524_524944


namespace rate_per_sq_meter_l524_524069

def length : ℝ := 5.5
def width : ℝ := 3.75
def totalCost : ℝ := 14437.5

theorem rate_per_sq_meter : (totalCost / (length * width)) = 700 := 
by sorry

end rate_per_sq_meter_l524_524069


namespace mass_percentage_of_c_in_co_l524_524698

noncomputable def atomic_mass_c : ℝ := 12.01
noncomputable def atomic_mass_o : ℝ := 16.00
noncomputable def molecular_mass_co : ℝ := atomic_mass_c + atomic_mass_o

theorem mass_percentage_of_c_in_co : (atomic_mass_c / molecular_mass_co) * 100 = 42.88 :=
by 
  have h_molecular_mass_co : molecular_mass_co = 28.01 := by sorry
  have h_calc : (atomic_mass_c / molecular_mass_co) * 100 = (12.01 / 28.01) * 100 := by sorry
  have h_res : (12.01 / 28.01) * 100 = 42.88 := by sorry
  exact h_res

end mass_percentage_of_c_in_co_l524_524698


namespace two_pi_irrational_l524_524967

-- Assuming \(\pi\) is irrational as is commonly accepted
def irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

theorem two_pi_irrational : irrational (2 * Real.pi) := 
by 
  sorry

end two_pi_irrational_l524_524967


namespace total_ants_approx_30_million_l524_524151

theorem total_ants_approx_30_million :
  let width_ft := 350
  let length_ft := 300
  let ants_per_sq_in := 2
  let inches_per_foot := 12
  let width_in := width_ft * inches_per_foot
  let length_in := length_ft * inches_per_foot
  let area_sq_in := width_in * length_in
  let total_ants := ants_per_sq_in * area_sq_in
  total_ants ≈ 30_000_000 := by sorry

end total_ants_approx_30_million_l524_524151


namespace not_all_two_digit_numbers_formed_from_valid_digits_are_prime_l524_524095

open Nat

-- Defining the digits that can be used
def valid_digits := [1, 3, 7, 9]

-- Function to check if a number formed is prime
def is_prime_conditional (n : Nat) :=
  List.all valid_digits (λ d1 => List.all valid_digits (λ d2 => ¬ Prime (10 * d1 + d2)))

/-- It is not possible for all two-digit numbers formed by the digits 1, 3, 7, and 9
    written on two cards to be prime. -/
theorem not_all_two_digit_numbers_formed_from_valid_digits_are_prime :
  ¬ is_prime_conditional 1 :=
by sorry

end not_all_two_digit_numbers_formed_from_valid_digits_are_prime_l524_524095


namespace number_of_cubes_is_15_l524_524628

-- Define the conditions of the problem.
def totalSurfaceAreaCuboid : ℝ := 2448 -- Total surface area of the cuboid in cm^2
def surfaceAreaCube : ℝ := 216         -- Surface area of one cube in cm^2
def wastePerCarving : ℝ := 0.2         -- Waste per carving in cm

-- Define the side length of a face of the cube.
def sideLengthOfCube : ℝ := real.sqrt (surfaceAreaCube / 6)

-- Define the height of the cuboid based on its surface area.
def heightOfCuboid :=
  let a := sideLengthOfCube in
  (totalSurfaceAreaCuboid - 2 * a^2) / (4 * a)

-- Define the effective height per cube including waste.
def effectiveHeightPerCube := sideLengthOfCube + wastePerCarving

-- Calculate the maximum number of cubes that can be obtained.
def maxNumberOfCubes : ℝ :=
  let h := heightOfCuboid in
  real.floor (h / effectiveHeightPerCube)

-- Prove that the maximum number of cubes is 15.
theorem number_of_cubes_is_15 : maxNumberOfCubes = 15 :=
  by
    -- The detailed proof would go here.
    sorry

end number_of_cubes_is_15_l524_524628


namespace greatest_integer_less_than_M_over_100_l524_524731

theorem greatest_integer_less_than_M_over_100 :
  (1 / (Nat.factorial 3 * Nat.factorial 16) +
   1 / (Nat.factorial 4 * Nat.factorial 15) +
   1 / (Nat.factorial 5 * Nat.factorial 14) +
   1 / (Nat.factorial 6 * Nat.factorial 13) +
   1 / (Nat.factorial 7 * Nat.factorial 12) +
   1 / (Nat.factorial 8 * Nat.factorial 11) +
   1 / (Nat.factorial 9 * Nat.factorial 10) = M / (Nat.factorial 2 * Nat.factorial 17)) →
  (⌊(M : ℚ) / 100⌋ = 27) := 
sorry

end greatest_integer_less_than_M_over_100_l524_524731


namespace triangle_area_l524_524941

theorem triangle_area (y1 y2 y3 : ℝ → ℝ) (h1 : ∀ x, y1 x = 3) (h2 : ∀ x, y2 x = x) (h3 : ∀ x, y3 x = -x) :
  let p1 := (3, y1 3)
      p2 := (-3, y1 (-3))
      p3 := (0, y2 0)
      area := 0.5 * ((3 * y1 3) + (-3 * y1 (-3)) + (0 * y2 0) 
                     - (3 * (-3) + 3 * 0 + 0 * 3)) in
  area = 9 := 
by
  sorry

end triangle_area_l524_524941


namespace total_cards_in_stack_l524_524706

theorem total_cards_in_stack (n : ℕ) (H1: 252 ≤ 2 * n) (H2 : (2 * n) % 2 = 0)
                             (H3 : ∀ k : ℕ, k ≤ 2 * n → (if k % 2 = 0 then k / 2 else (k + 1) / 2) * 2 = k) :
  2 * n = 504 := sorry

end total_cards_in_stack_l524_524706


namespace smallest_positive_sum_l524_524680

theorem smallest_positive_sum : ∃ (b : Fin 97 → ℤ), (∀ i, b i = 1 ∨ b i = -1) ∧ (∑ i in Fin.range 97, ∑ j in Fin.range 97, if i < j then b i * b j else 0) = 12 :=
sorry

end smallest_positive_sum_l524_524680


namespace binomial_sum_equality_l524_524025

noncomputable def binomial_sum (n m : ℕ) : ℂ :=
  ∑ k in Finset.range m, Complex.binomial n (4 * k + 1)

theorem binomial_sum_equality (n m t : ℕ) (h: 4*t - 3 ≤ n) :
  binomial_sum n (m+1) = 
      (1 / 2 : ℂ) * 
          ((2 : ℂ)^(n-1) + 
          (2 : ℂ)^(n / 2) * Complex.sin(Real.pi * n / 4)) :=
sorry

end binomial_sum_equality_l524_524025


namespace line_circle_no_intersection_l524_524356

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524356


namespace grayson_test_questions_l524_524321

theorem grayson_test_questions (h1 : ∀ q, ((2 * 60 / 2) = q)) (h2 : ∀ u, (u = 40)) : 
   (h1 + h2 = 100) :=
by 
  sorry

end grayson_test_questions_l524_524321


namespace least_three_digit_with_3_5_7_l524_524946

def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def least_three_digit_number_with_factors (a b c : ℕ) : ℕ :=
  let lcm := (Nat.lcm (Nat.lcm a b) c)
  ∃ n, is_three_digit_integer n ∧ is_divisible_by n lcm ∧ (∀ m, is_three_digit_integer m ∧ is_divisible_by m lcm → n ≤ m)

theorem least_three_digit_with_3_5_7 : least_three_digit_number_with_factors 3 5 7 = 105 :=
by
  sorry

end least_three_digit_with_3_5_7_l524_524946


namespace op_comm_op_not_assoc_l524_524004

variables {x y z : ℝ}
variable (h_pos : ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0)

def op (x y : ℝ) : ℝ := (2 * x * y) / (x + y)

theorem op_comm : ∀ x y, op x y = op y x :=
by
  intros x y
  unfold op
  rw [mul_comm x y, add_comm x y]

theorem op_not_assoc : ∃ x y z, op (op x y) z ≠ op x (op y z) :=
by
  use 1, 2, 3 -- Example values to demonstrate non-associativity
  unfold op
  dsimp
  -- Proof of non-associativity skipped
  sorry

end op_comm_op_not_assoc_l524_524004


namespace turn_off_all_lamps_l524_524925

theorem turn_off_all_lamps (n : ℕ) (buttons : Finset (Finset ℕ))
  (lamps : Finset ℕ) (h : ∀ s ⊆ lamps, ∃ b ∈ buttons, (b ∩ s).card % 2 = 1) :
  ∃ S ⊆ buttons, (lamps.fold (⊕) false (λ l, ∃ s ∈ S, l ∈ s)) = false := 
sorry

end turn_off_all_lamps_l524_524925


namespace find_y_l524_524412

-- Definitions from conditions
def x := Real.pi / 6  -- 30 degrees in radians
def lhs (y : ℝ) := Real.sin x * Real.cos x * y - 2 * (Real.sin x)^2 * y + Real.cos x * y

-- Statement of the theorem
theorem find_y : ∃ y : ℝ, lhs y = 0.5 ∧ y = (6 * Real.sqrt 3 + 4) / 23 :=
by 
  sorry

end find_y_l524_524412


namespace number_of_minibuses_l524_524044

theorem number_of_minibuses (total_students : ℕ) (capacity : ℕ) (h : total_students = 48) (h_capacity : capacity = 8) : 
  ∃ minibuses, minibuses = (total_students + capacity - 1) / capacity ∧ minibuses = 7 :=
by
  have h1 : (48 + 8 - 1) = 55 := by simp [h, h_capacity]
  have h2 : 55 / 8 = 6 := by simp [h, h_capacity]
  use 7
  sorry

end number_of_minibuses_l524_524044


namespace find_alpha_and_polar_equation_l524_524713

def point (α β : ℝ) := (α, β)

def line (α : ℝ) (t : ℝ) := (2 + t * Real.cos α, 1 + t * Real.sin α)

def distance (P A : ℝ × ℝ) := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

theorem find_alpha_and_polar_equation :
  ∃ (α : ℝ),
  (
    let P := point 2 1 in
    let l := line α in
    let A := line α (-(1 / Real.sin α)) in
    let B := line α (-(2 / Real.cos α)) in
    distance P A * distance P B = 4 ∧ α = 3 * Real.pi / 4
  )
∧
  (
    let α := 3 * Real.pi / 4 in
    ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 3
  ) := 
begin
  sorry
end

end find_alpha_and_polar_equation_l524_524713


namespace find_locus_of_M_l524_524724

variables {P : Type*} [MetricSpace P] 
variables (A B C M : P)

def on_perpendicular_bisector (A B M : P) : Prop := 
  dist A M = dist B M

def on_circle (center : P) (radius : ℝ) (M : P) : Prop := 
  dist center M = radius

def M_AB (A B M : P) : Prop :=
  (on_perpendicular_bisector A B M) ∨ (on_circle A (dist A B) M) ∨ (on_circle B (dist A B) M)

def M_BC (B C M : P) : Prop :=
  (on_perpendicular_bisector B C M) ∨ (on_circle B (dist B C) M) ∨ (on_circle C (dist B C) M)

theorem find_locus_of_M :
  {M : P | M_AB A B M} ∩ {M : P | M_BC B C M} = {M : P | M_AB A B M ∧ M_BC B C M} :=
by sorry

end find_locus_of_M_l524_524724


namespace no_real_intersections_l524_524349

theorem no_real_intersections (x y : ℝ) (h1 : 3 * x + 4 * y = 12) (h2 : x^2 + y^2 = 4) : false :=
by
  sorry

end no_real_intersections_l524_524349


namespace function_decreasing_implies_a_range_l524_524894

open Real

-- Given conditions and the question to proof
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → ((log 1/2 a) ^ x > (log 1/2 a) ^ y)) →
  (1/2 < a ∧ a < 1) :=
begin
  sorry
end

end function_decreasing_implies_a_range_l524_524894


namespace digit_864_div_5_appending_zero_possibilities_l524_524136

theorem digit_864_div_5_appending_zero_possibilities :
  ∀ X : ℕ, (X * 1000 + 864) % 5 ≠ 0 :=
by sorry

end digit_864_div_5_appending_zero_possibilities_l524_524136


namespace paco_ate_more_salty_than_sweet_l524_524495

-- Define the initial conditions
def sweet_start := 8
def salty_start := 6
def sweet_ate := 20
def salty_ate := 34

-- Define the statement to prove
theorem paco_ate_more_salty_than_sweet : (salty_ate - sweet_ate) = 14 := by
    sorry

end paco_ate_more_salty_than_sweet_l524_524495


namespace _l524_524451

noncomputable def area_of_triangle_PEF
  (radius_M : ℝ) (length_EF : ℝ) (parallel_EF_PQ : Prop)
  (length_PR : ℝ) (collinear_P_R_M_Q : Prop) : ℝ :=
  if h : (radius_M = 10) ∧ (length_EF = 12) ∧ (length_PR = 20) ∧ collinear_P_R_M_Q ∧ parallel_EF_PQ
  then 48
  else 0

noncomputable theorem area_of_triangle_PEF_proof
  (h : (10 : ℝ) = 10 ∧ (12 : ℝ) = 12 ∧ (20 : ℝ) = 20 ∧ True ∧ True) :
  area_of_triangle_PEF 10 12 True 20 True = 48 :=
  by sorry

end _l524_524451


namespace jellybean_count_l524_524215

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l524_524215


namespace sqrt_xyz_sum_l524_524005

variables {R : Type*} [linear_ordered_field R]

-- Define the conditions
def y_plus_z (x y z : R) : Prop := y + z = 15
def z_plus_x (x y z : R) : Prop := z + x = 17
def x_plus_y (x y z : R) : Prop := x + y = 16

-- Main theorem statement in Lean 4
theorem sqrt_xyz_sum (x y z : R) (h1 : y_plus_z x y z) (h2 : z_plus_x x y z) (h3 : x_plus_y x y z) :
  Real.sqrt (x * y * z * (x + y + z)) = 72 * Real.sqrt 7 :=
sorry

end sqrt_xyz_sum_l524_524005


namespace tangent_line_at_point_l524_524696

open Real

/-- Find the equation of the tangent line to the curve y = 2x - ln x at the point (1, 2) -/
theorem tangent_line_at_point :
  let f : ℝ → ℝ := λ x, 2 * x - log x
  let df : ℝ → ℝ := λ x, 2 - 1 / x
  f 1 = 2 → (∀ x, df x = deriv f x) → (∀ x, f x = 2 * x - log x) → 
  tangent_line (f) 1 = λ x, x + 1 :=
by
  sorry

end tangent_line_at_point_l524_524696


namespace find_k_perpendicular_l524_524320

def vector2D := (ℝ × ℝ)

def a : vector2D := (1, 1)
def b : vector2D := (2, -3)

def dot_product (u v : vector2D) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k_perpendicular (k : ℝ) (h : dot_product (k • a - 2 • b) a = 0) : k = -1 :=
sorry

end find_k_perpendicular_l524_524320


namespace line_circle_no_intersection_l524_524360

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
by
  intro x y
  intro h
  cases h with h1 h2
  let y_val := (12 - 3 * x) / 4
  have h_subst : (x^2 + y_val^2 = 4) := by
    rw [←h2, h1, ←y_val]
    sorry
  have quad_eqn : (25 * x^2 - 72 * x + 80 = 0) := by
    sorry
  have discrim : (−72)^2 - 4 * 25 * 80 < 0 := by
    sorry
  exact discrim false

end line_circle_no_intersection_l524_524360


namespace go_board_configurations_l524_524633

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem go_board_configurations : 
  let numerator := 3^361
  let denominator := 10000^52
  let result := numerator / denominator
  let log_approximations := lg 3 ≈ 0.477 ∧ lg 10000 = 4
  let log_result := lg result
  let closest_value := 10^(-36)
  log_approximations →
  |log_result + 36| < |log_result + 35| →
  |log_result + 36| < |log_result + 34| →
  |log_result + 36| < |log_result + 37| →
  result ≈ closest_value :=
by
  sorry

end go_board_configurations_l524_524633


namespace period_of_cos2x_minus_sin2x_l524_524905

noncomputable def period (f : Real → Real) : Real := sorry

theorem period_of_cos2x_minus_sin2x :
  period (λ x => cos x ^ 2 - sin x ^ 2) = π :=
sorry

end period_of_cos2x_minus_sin2x_l524_524905


namespace ball_rebound_distance_l524_524856

-- Definitions of initial conditions
def initial_height := 104
def rebound_ratio := 0.5
def total_distance := 260

-- Problem statement in Lean 4
theorem ball_rebound_distance :
  ∃ (n : ℕ), (n + 1 = 3) ∧ (let travel (n : ℕ) := initial_height + initial_height * (1 - rebound_ratio^n) / (1 - rebound_ratio)
                               in travel n = total_distance) :=
by
  sorry

end ball_rebound_distance_l524_524856


namespace exists_n_lt_p_minus_1_not_div_p2_l524_524476

theorem exists_n_lt_p_minus_1_not_div_p2 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ (n : ℕ), n < p - 1 ∧ ¬(p^2 ∣ (n^((p - 1)) - 1)) ∧ ¬(p^2 ∣ ((n + 1)^((p - 1)) - 1)) := 
sorry

end exists_n_lt_p_minus_1_not_div_p2_l524_524476


namespace hyperbola_vertex_distance_l524_524233

theorem hyperbola_vertex_distance (a b : ℝ) (h_eq : a^2 = 16) (hyperbola_eq : ∀ x y : ℝ, 
  (x^2 / 16) - (y^2 / 9) = 1) : 
  (2 * a) = 8 :=
by
  have h_a : a = 4 := by sorry
  rw [h_a]
  norm_num

end hyperbola_vertex_distance_l524_524233


namespace alia_markers_count_l524_524167

theorem alia_markers_count :
  ∀ (Alia Austin Steve Bella : ℕ),
  (Alia = 2 * Austin) →
  (Austin = (1 / 3) * Steve) →
  (Steve = 60) →
  (Bella = (3 / 2) * Alia) →
  Alia = 40 :=
by
  intros Alia Austin Steve Bella H1 H2 H3 H4
  sorry

end alia_markers_count_l524_524167


namespace jellybean_count_l524_524213

theorem jellybean_count (x : ℝ) (h : (0.75^3) * x = 27) : x = 64 :=
sorry

end jellybean_count_l524_524213


namespace number_of_drawings_on_first_page_l524_524811

-- Let D be the number of drawings on the first page.
variable (D : ℕ)

-- Conditions:
-- 1. D is the number of drawings on the first page.
-- 2. The number of drawings increases by 5 after every page.
-- 3. The total number of drawings in the first five pages is 75.

theorem number_of_drawings_on_first_page (h : D + (D + 5) + (D + 10) + (D + 15) + (D + 20) = 75) :
    D = 5 :=
by
  sorry

end number_of_drawings_on_first_page_l524_524811


namespace percentage_of_primes_divisible_by_5_l524_524565

def primes_less_than_twenty : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}
def divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def percentage (num total : ℕ) : ℚ := (num.to_rat / total.to_rat) * 100

theorem percentage_of_primes_divisible_by_5 :
  percentage (primes_less_than_twenty.count divisible_by_five) (primes_less_than_twenty.size) = 12.5 :=
by
  sorry

end percentage_of_primes_divisible_by_5_l524_524565


namespace sum_of_divisors_24_l524_524958

theorem sum_of_divisors_24 : (∑ d in (finset.filter (λ n, 24 % n = 0) (finset.range 25)), d) = 60 :=
by
  sorry

end sum_of_divisors_24_l524_524958


namespace length_of_AB_l524_524860

variable {A B C D E F G : Type}
variable [metric_space A]
variables [metric_space B]
variables [has_distance C] [has_distance D] [has_distance E]
variables [has_distance F] [has_distance G]

def is_midpoint (X Y Z : point) : Prop := dist X Y = dist X Z ∧ 2 * dist X Y = dist Y Z

theorem length_of_AB 
  (C_midpoint_AB : is_midpoint A B C)
  (D_midpoint_AC : is_midpoint A C D)
  (E_midpoint_AD : is_midpoint A D E)
  (F_midpoint_AE : is_midpoint A E F)
  (G_midpoint_AF : is_midpoint A F G)
  (AG_eq_4 : dist A G = 4) : 
  dist A B = 128 := 
sorry

end length_of_AB_l524_524860


namespace triangle_inequality_integer_length_l524_524773

theorem triangle_inequality_integer_length :
  let a := 9
      b := 4
  in 
  let num_possible_c := (finset.Ico 6 13).card  -- 6 to 12 inclusive
  in num_possible_c = 7 := 
by 
  sorry

end triangle_inequality_integer_length_l524_524773


namespace cos_x_minus_y_l524_524263

noncomputable theory

variables {x y : ℝ}

theorem cos_x_minus_y :
  (cos x + cos y = 1 / 2) →
  (sin x + sin y = 1 / 3) →
  cos (x - y) = -59 / 72 :=
by
  {intros h1 h2,
   -- Use the provided identity here, calculations omitted
   sorry}

end cos_x_minus_y_l524_524263


namespace square_problem_solution_l524_524118

theorem square_problem_solution
  (x : ℝ)
  (h1 : ∃ s1 : ℝ, s1^2 = x^2 + 12*x + 36)
  (h2 : ∃ s2 : ℝ, s2^2 = 4*x^2 - 12*x + 9)
  (h3 : 4 * (s1 + s2) = 64) :
  x = 13 / 3 :=
by
  sorry

end square_problem_solution_l524_524118


namespace feta_price_calculation_l524_524045

noncomputable def feta_price_per_pound (sandwiches_price : ℝ) (sandwiches_count : ℕ) 
  (salami_price : ℝ) (brie_factor : ℝ) (olive_price_per_pound : ℝ) 
  (olive_weight : ℝ) (bread_price : ℝ) (total_spent : ℝ)
  (feta_weight : ℝ) :=
  (total_spent - (sandwiches_count * sandwiches_price + salami_price + brie_factor * salami_price + olive_price_per_pound * olive_weight + bread_price)) / feta_weight

theorem feta_price_calculation : 
  feta_price_per_pound 7.75 2 4.00 3 10.00 0.25 2.00 40.00 0.5 = 8.00 := 
by
  sorry

end feta_price_calculation_l524_524045


namespace value_of_expression_eq_34_l524_524106

theorem value_of_expression_eq_34 : (2 - 6 + 10 - 14 + 18 - 22 + 26 - 30 + 34 - 38 + 42 - 46 + 50 - 54 + 58 - 62 + 66 - 70 + 70) = 34 :=
by
  sorry

end value_of_expression_eq_34_l524_524106


namespace number_of_x_for_P_eq_zero_l524_524708

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x) - Complex.exp (4 * Complex.I * x)

theorem number_of_x_for_P_eq_zero : 
  ∃ (n : ℕ), n = 4 ∧ ∃ (xs : Fin n → ℝ), (∀ i, 0 ≤ xs i ∧ xs i < 2 * Real.pi ∧ P (xs i) = 0) ∧ Function.Injective xs := 
sorry

end number_of_x_for_P_eq_zero_l524_524708


namespace parallel_planes_perpendicular_same_line_l524_524641

theorem parallel_planes_perpendicular_same_line : 
  (∀ (P₁ P₂ : Plane) (l : Line), (Plane.parallel_to_line P₁ l ∧ Plane.parallel_to_line P₂ l → ¬Plane.parallel P₁ P₂)) ∧
  (∀ (L₁ L₂ : Line) (P : Plane), (Line.parallel_to_plane L₁ P ∧ Line.parallel_to_plane L₂ P → ¬Line.parallel L₁ L₂)) ∧
  (∀ (P₁ P₂ : Plane) (l : Line), Plane.perpendicular_to_line P₁ l ∧ Plane.perpendicular_to_line P₂ l → Plane.parallel P₁ P₂) ∧
  (∀ (P₁ P₂ : Plane) (P : Plane), (Plane.perpendicular_to_plane P₁ P ∧ Plane.perpendicular_to_plane P₂ P → ¬Plane.parallel P₁ P₂)) → 
  (CorrectProposition = ③) :=
by
  sorry

end parallel_planes_perpendicular_same_line_l524_524641


namespace max_expression_value_l524_524075

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -7.5 ≤ a ∧ a ≤ 7.5)
  (h2 : -7.5 ≤ b ∧ b ≤ 7.5)
  (h3 : -7.5 ≤ c ∧ c ≤ 7.5)
  (h4 : -7.5 ≤ d ∧ d ≤ 7.5) :
  ∃ x, x = a + 2*b + c + 2*d - (a*b + b*c + c*d + d*a) ∧ x ≤ 240 :=
begin
  sorry
end

end max_expression_value_l524_524075


namespace Jims_apples_fits_into_average_l524_524416

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end Jims_apples_fits_into_average_l524_524416


namespace speed_against_current_is_10_l524_524975

-- Define variables for the man's speed with the current and the speed of the current
variables (speed_with_current speed_of_current man's_speed_against_current : ℝ)

-- Given conditions
def condition1 : Prop := speed_with_current = 15
def condition2 : Prop := speed_of_current = 2.5

-- Definition for the man's speed against the current
def calculate_speed_against_current (v_with_current v_current : ℝ) :=
  (v_with_current - v_current)

-- The Lean 4 statement
theorem speed_against_current_is_10 :
  condition1 → condition2 → man's_speed_against_current = 10 :=
by
  intros hc1 hc2
  have h1 : speed_with_current = 15 := hc1
  have h2 : speed_of_current = 2.5 := hc2
  let v := (speed_with_current - speed_of_current)
  have hv : v = 12.5 := by 
    rw [h1, h2]
    norm_num
  have hs : calculate_speed_against_current v speed_of_current = 10 := by
    rw [hv, h2]
    norm_num
  exact hs

end speed_against_current_is_10_l524_524975


namespace cannot_determine_x_l524_524149

theorem cannot_determine_x
  (n m : ℝ) (x : ℝ)
  (h1 : n + m = 8) 
  (h2 : n * x + m * (1/5) = 1) : true :=
by {
  sorry
}

end cannot_determine_x_l524_524149


namespace find_multiplier_l524_524727

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_multiplier : ∃ (m : ℝ), m * f 3 - 10 = f 1 ∧ m = 2 :=
by {
  use 2,
  split,
  {
    -- We need to show m * f 3 - 10 = f 1
    have h1 : f 3 = 4, by { sorry },  -- Proof that f 3 = 4
    have h2 : f 1 = -2, by { sorry }, -- Proof that f 1 = -2
    calc
      2 * f 3 - 10 = 2 * 4 - 10 : by rw h1
      ...             = 8 - 10   : by ring
      ...             = -2       : by ring
      ...             = f 1      : by rw h2,
  },
  {
    -- We need to show m = 2
    refl,
  }
}

end find_multiplier_l524_524727


namespace valid_positive_integer_pairs_l524_524677

theorem valid_positive_integer_pairs (a b : ℕ) (h : ab^2 + b + 7 ∣ a^2 b + a + b) :
    (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ (k : ℕ), k > 0 ∧ b = 7 * k ∧ a = 7 * k^2) :=
by
  -- Proof steps would go here
  sorry

end valid_positive_integer_pairs_l524_524677


namespace percentage_of_primes_less_than_20_divisible_by_5_l524_524570

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_less_than_20 := {n : ℕ | n < 20 ∧ is_prime n}

def primes_less_than_20_divisible_by_5 := {n ∈ primes_less_than_20 | 5 ∣ n}

theorem percentage_of_primes_less_than_20_divisible_by_5 : 
  (primes_less_than_20_divisible_by_5.to_finset.card : ℝ) / (primes_less_than_20.to_finset.card : ℝ) * 100 = 12.5 :=
begin
  -- Proving this statement directly would involve showing the calculations explicitly.
  -- However, we just set up the framework here.
  sorry
end

end percentage_of_primes_less_than_20_divisible_by_5_l524_524570


namespace evaluate_polynomial_at_6_l524_524046

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

theorem evaluate_polynomial_at_6 : polynomial 6 = 3658 :=
by 
  sorry

end evaluate_polynomial_at_6_l524_524046


namespace tangent_line_circumscribed_circle_l524_524788

-- Definitions for the problem setup
variables {R : Type*} [ordered_ring R]

-- Square vertices
structure Square (R : Type*) := 
(A B C D : R × R)

-- IsMidpoint predicate
def IsMidpoint (p1 p2 mid : R × R) : Prop :=
  mid.1 = (p1.1 + p2.1) / 2 ∧ mid.2 = (p1.2 + p2.2) / 2

-- Conditions 
variables (A B C D F E H : R × R)

def square_ABCD (s : Square R) : Prop :=
  s.A = A ∧ s.B = B ∧ s.C = C ∧ s.D = D ∧ (
  -- Verify square properties (right angles and equal sides)
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  (A.1 = 0) ∧ (A.2 = 0) ∧ (B.2 = 0) ∧ (A.1 < B.1) ∧ (D.1 = 0) ∧ (A.2 < D.2))

def midpoint_CD_F (mid : R × R) : Prop :=
  IsMidpoint C D mid

def point_on_AB_E (pt : R × R) (t : R) : Prop :=
  t > 0 ∧ t < (B.1 - A.1) ∧ pt.1 = t ∧ pt.2 = 0

def AE_greater_EB (A E B : R × R) : Prop :=
  (A.1 < E.1) ∧ (E.1 < B.1) 
  -- no need to include (A.1 = 0) and (B.1 > 0) as these are derived from square properties.

def line_parallel_DE_meets_BC_H (E DE_line H : R × R) (BC_line : R) : Prop :=
  ∃ slope : R, ∃ c : R,
    -- line DE
    (slope = (D.2 - E.2) / (D.1 - E.1) ∧ c = F.2 - slope * F.1 ∧
    -- line parallel to DE at F
    H.1 = B.1 ∧ 
    H.2 = slope * H.1 + c)

-- Circumcircle definition
def is_circumscribed (circle_center : R×R) (circle_radius : R) (A B C D : R × R) : Prop :=
  dist circle_center A = circle_radius ∧
  dist circle_center B = circle_radius ∧
  dist circle_center C = circle_radius ∧
  dist circle_center D = circle_radius

-- Tangency property
def is_tangent (line : R → R) (center : R × R) (radius : R) : Prop :=
  let d := (line center.1 - center.2)/ sqrt (1 + (line center.1)^2) in
  d = radius

-- Problem statement: Prove that the line (EH) is tangent to the circle circumscribed with ABCD
theorem tangent_line_circumscribed_circle 
  (s : Square R)
  (hF_mid : midpoint_CD_F F)
  (hE_on_AB : ∃ t, point_on_AB_E E t)
  (hAE_gEB : AE_greater_EB A E B)
  (hpar_DE_H : ∃ line, line_parallel_DE_meets_BC_H E line H)
  (circle_center : R × R) (circle_radius : R)
  (hcircle : is_circumscribed circle_center circle_radius A B C D) :
  ∃ line EH_line, is_tangent EH_line circle_center circle_radius :=
sorry

end tangent_line_circumscribed_circle_l524_524788


namespace bus_passengers_at_last_stop_l524_524924

theorem bus_passengers_at_last_stop :
  let capacity := 150 in
  let initial_passengers := 120 in
  let stop1_on := 26 in
  let stop2_off := 15 in let stop2_on := 25 in
  let stop3_off := 30 in let stop3_on := 10 in
  let stop4_off := 45 in let stop4_on := 37 in
  let stop5_off := 22 in let stop5_on := 16 in
  let stop6_off := 40 in let stop6_on := 20 in
  let stop7_off := 12 in let stop7_on := 32 in
  let stop8_off := 34 in let stop8_on := 0 in

  let after_stop1 := initial_passengers + stop1_on in
  let after_stop2 := min capacity (after_stop1 - stop2_off + stop2_on) in
  let after_stop3 := after_stop2 - stop3_off + stop3_on in
  let after_stop4 := min capacity (after_stop3 - stop4_off + stop4_on) in
  let after_stop5 := after_stop4 - stop5_off + stop5_on in
  let after_stop6 := after_stop5 - stop6_off + stop6_on in
  let after_stop7 := min capacity (after_stop6 - stop7_off + stop7_on) in
  let after_stop8 := after_stop7 - stop8_off + stop8_on in
  after_stop8 = 82 :=
by
  sorry

end bus_passengers_at_last_stop_l524_524924


namespace smaller_angle_of_parallelogram_l524_524121

theorem smaller_angle_of_parallelogram : 
  ∀ (small large : ℝ), large = small + 90 ∧ small + large = 180 → small = 45 :=
by
  intros small large h
  cases h with h1 h2
  sorry -- This is where the proof would go

end smaller_angle_of_parallelogram_l524_524121


namespace budget_for_supplies_l524_524617

-- Conditions as definitions
def percentage_transportation := 20
def percentage_research_development := 9
def percentage_utilities := 5
def percentage_equipment := 4
def degrees_salaries := 216
def total_degrees := 360
def total_percentage := 100

-- Mathematical problem: Prove the percentage spent on supplies
theorem budget_for_supplies :
  (total_percentage - (percentage_transportation +
                       percentage_research_development +
                       percentage_utilities +
                       percentage_equipment) - 
   ((degrees_salaries * total_percentage) / total_degrees)) = 2 := by
  sorry

end budget_for_supplies_l524_524617


namespace good_trapezoid_angles_equal_l524_524609

theorem good_trapezoid_angles_equal (ABCD_good : Trapezoid)
  (h_circumcircle : Circumcircle ABCD_good)
  (AB_parallel_CD : Parallel ABCD_good.AB ABCD_good.CD)
  (CD_shorter_AB : Length ABCD_good.CD < Length ABCD_good.AB)
  (BSE_FSC_equal : ∠(B, S, E) = ∠(F, S, C))
  (E_tangent_S : Tangent (Circumcircle ABCD_good) E S)
  (F_tangent_S : Tangent (Circumcircle ABCD_good) F S)
  (E_side_CD : E_sameside_CD : OnSameSide E A CD) : 
  (\angle ABCD_good.ABC = 60° ∨ Length ABCD_good.AB = Length ABCD_good.AD → 
  \angle BSE = \angle FSC) :=
sorry

end good_trapezoid_angles_equal_l524_524609
