import Mathlib

namespace problem_l714_714652

variable {ℝ : Type} [LinearOrder ℝ]

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem problem (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : increasing_on f {x | x ≤ -1}) :
  f 2 < f (-1.5) ∧ f (-1.5) < f -1 :=
by
  -- proof goes here
  sorry

end problem_l714_714652


namespace intersection_of_sets_l714_714113

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {-1, 0, 1}

theorem intersection_of_sets :
  A ∩ B = {0, 1} :=
sorry

end intersection_of_sets_l714_714113


namespace f_2012_eq_zero_symmetric_about_2_compare_f_values_l714_714105

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f(-x) = -f(x)
axiom f_periodic (x : ℝ) : f(x-4) = -f(x)
axiom f_increasing (a b : ℝ) (ha : 0 ≤ a) (hb : b ≤ 2) (hab : a ≤ b) : f(a) ≤ f(b)

-- Prove that f(2012) = 0
theorem f_2012_eq_zero : f(2012) = 0 := sorry

-- Prove that the graph of f(x) is symmetric about the line x = 2
theorem symmetric_about_2 : ∀ x : ℝ, f(x + 2) = f(2 - x) := sorry

-- Prove that f(-25) < f(80) < f(11)
theorem compare_f_values : f(-25) < f(80) ∧ f(80) < f(11) := sorry

end f_2012_eq_zero_symmetric_about_2_compare_f_values_l714_714105


namespace nat_min_a_plus_b_l714_714935

theorem nat_min_a_plus_b (a b : ℕ) (h1 : b ∣ a^a) (h2 : ¬ b ∣ a) (h3 : Nat.coprime b 210) : a + b = 374 :=
sorry

end nat_min_a_plus_b_l714_714935


namespace function_formula_l714_714725

noncomputable def f : ℝ → ℝ :=
  fun x => x^2 - x + 1

theorem function_formula (f_def : ∀ x : ℝ, f x = x^2 - x + 1) :
  (f 0 = 1) ∧ (∀ a b : ℝ, f (a - b) = f a - b * (2 * a - b + 1)) :=
by
  split
  · rw [f_def 0]
    norm_num

  · intros a b
    rw [f_def (a - b), f_def a]
    ring

end function_formula_l714_714725


namespace leila_yards_l714_714665

variable (mile_yards : ℕ := 1760)
variable (marathon_miles : ℕ := 28)
variable (marathon_yards : ℕ := 1500)
variable (marathons_ran : ℕ := 15)

theorem leila_yards (m y : ℕ) (h1 : marathon_miles = 28) (h2 : marathon_yards = 1500) (h3 : mile_yards = 1760) (h4 : marathons_ran = 15) (hy : 0 ≤ y ∧ y < mile_yards) :
  y = 1200 :=
sorry

end leila_yards_l714_714665


namespace jake_depth_l714_714235

-- Defining Sara's birdhouse dimensions in inches
def sara_width : ℝ := 12
def sara_height : ℝ := 24
def sara_depth : ℝ := 24

-- Defining Jake's birdhouse known dimensions in inches
def jake_width : ℝ := 16
def jake_height : ℝ := 20
def volume_difference : ℝ := 1152

-- Given the following conditions
-- Calculate the depth of Jake's birdhouse
theorem jake_depth :
  let sara_volume := sara_width * sara_height * sara_depth in
  ∃ D : ℝ, jake_width * jake_height * D - sara_volume = volume_difference ∧ D = 25.2 :=
by {
  sorry,
}

end jake_depth_l714_714235


namespace count_points_for_half_area_l714_714175

noncomputable def triangle_area := 50
noncomputable def sub_triangle_area := 25
noncomputable def total_positions := 41

theorem count_points_for_half_area (P : Point) (F G H : Point) 
  (inside_triangle : Point -> Triangle -> Prop) (FGH : Triangle)
  (area_triangle : Triangle -> ℝ) (area_condition : ℝ -> ℝ -> ℝ -> Prop) :
  (area_triangle FGH = triangle_area) ∧ 
  (∀ P, inside_triangle P FGH) ∧ 
  (∀ (FPG GPH HPF : Triangle), 
    area_condition (area_triangle FPG) (area_triangle GPH) (area_triangle HPF)) →
  ∃ count = 9, ∀ (num_positions : ℕ), num_positions = 9 :=
sorry

end count_points_for_half_area_l714_714175


namespace find_100k_l714_714988

-- Define the square and properties
structure Square (A B C D : Point) :=
(side_length : ℝ)
(vertex_positions : A = (0,0) ∧ B = (2,0) ∧ C = (2,2) ∧ D = (0,2))

-- Define the property of line segments
def line_segment (P Q : Point) :=
(dist P Q = 2)

-- Define set S as the set of all line segments with the given property
def S (A B C D : Point) : set (Point × Point) :=
{ PQ : Point × Point | line_segment PQ.1 PQ.2 ∧
(PQ.1 ∈ {A, B, C, D} ∨ PQ.2 ∈ {A, B, C, D}) }

-- Midpoint definition for a line segment
def midpoint (P Q : Point) : Point :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Function to check the enclosed area formed by the midpoints
def enclosed_area (s : set (Point × Point)) : ℝ :=
4 - Real.pi

-- Define k as the enclosed area
def k_calculated (_ : Square (0, 0) (2, 0) (2, 2) (0, 2)) :=
enclosed_area (S (0, 0) (2, 0) (2, 2) (0, 2))

-- Main theorem to prove 100k = 86
theorem find_100k (sq : Square (0, 0) (2, 0) (2, 2) (0, 2)) : 
  100 * k_calculated sq = 86 := 
begin
  -- proof steps will be filled here
  sorry
end

end find_100k_l714_714988


namespace multiple_root_primes_pq_l714_714044

theorem multiple_root_primes_pq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ a : ℝ, (a ≠ 0) ∧ ((Polynomial.C (0 : ℝ) + Polynomial.X) ^ 2 * (Polynomial.C (3 * a^2) + 
  Polynomial.X ^ 2 + Polynomial.C (-2 * a))) = Polynomial.C q + Polynomial.X * Polynomial.C (p^2) + Polynomial.X^4) → p = 2 ∧ q = 3 :=
by
  sorry

end multiple_root_primes_pq_l714_714044


namespace system_solutions_system_solutions_alternative_interval_l714_714646

variables (x a : ℝ)

def linear_eq (x a : ℝ) : Prop := 5 * (x + 4) - 7 * (a + 2) = 0
def poly_eq (x a : ℝ) : Prop := (x + 1) ^ 2 + ((abs (x + 1) / (x + 1)) + (abs (x - 3) / (x - 3)) + a) ^ 2 = 25
def domain_validity (x a : ℝ) : Prop := (x ≠ -1) ∧ (x ≠ 3) ∧ (5 * x - 7 * a + 6 ≥ 0)

theorem system_solutions (a : ℝ) :
  a ∈ (-5:ℝ, -3) → ∃ x, linear_eq x a ∧ poly_eq x a ∧ domain_validity x a ∧ 
  (x = (7/5) * a - 6/5 ∨ x = -1 + sqrt (25 - a^2) ∨ x = -1 + sqrt (25 - (a + 2)^2)) :=
sorry

theorem system_solutions_alternative_interval (a : ℝ):
  a ∈ (-3:ℝ, -2) → ∃ x, linear_eq x a ∧ poly_eq x a ∧ domain_validity x a ∧ 
  (x = (7/5) * a - 6/5 ∨ x = -1 - sqrt (25 - (a - 2)^2) ∨ x = -1 + sqrt (25 - (a + 2)^2)) :=
sorry

end system_solutions_system_solutions_alternative_interval_l714_714646


namespace unicycles_count_l714_714607

-- Definitions of the number of bicycles, tricycles, and total wheels
def num_bicycles : ℕ := 3
def num_tricycles : ℕ := 4
def total_wheels : ℕ := 25
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3
def wheels_per_unicycle : ℕ := 1

-- Theorem stating the number of unicycles
theorem unicycles_count : 
  num_unicycles = (num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle)
   sorry

end unicycles_count_l714_714607


namespace prove_P_plus_V_eq_zero_l714_714170

variable (P Q R S T U V : ℤ)

-- Conditions in Lean
def sequence_conditions (P Q R S T U V : ℤ) :=
  S = 7 ∧
  P + Q + R = 27 ∧
  Q + R + S = 27 ∧
  R + S + T = 27 ∧
  S + T + U = 27 ∧
  T + U + V = 27 ∧
  U + V + P = 27

-- Assertion that needs to be proved
theorem prove_P_plus_V_eq_zero (P Q R S T U V : ℤ) (h : sequence_conditions P Q R S T U V) : 
  P + V = 0 := by
  sorry

end prove_P_plus_V_eq_zero_l714_714170


namespace quadratic_roots_sin_cos_l714_714427

theorem quadratic_roots_sin_cos (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (m : ℝ) :
  (∀ x, 2 * x^2 - (real.sqrt 3 + 1) * x + m = 0 → (x = real.sin θ ∨ x = real.cos θ)) →
  (m = real.sqrt 3 / 2) ∧
  (real.tan θ * real.sin θ / (real.tan θ - 1) + real.cos θ / (1 - real.tan θ) = (real.sqrt 3 + 1) / 2) ∧
  ((real.sin θ = real.sqrt 3 / 2 ∧ real.cos θ = 1 / 2) ∨ (real.sin θ = 1 / 2 ∧ real.cos θ = real.sqrt 3 / 2)) :=
by
  sorry

end quadratic_roots_sin_cos_l714_714427


namespace areaEnclosedByMidpoints_l714_714986

namespace MathProof

noncomputable def lineSegmentLengthEqTwo (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem areaEnclosedByMidpoints : 
  (let k := 4 - π in 100 * k = 86) :=
by 
  let k := 4 - π
  have h1 : 100 * k = 100 * (4 - π) := rfl
  have h2 : 100 * (4 - π) = 86 := sorry
  exact h2

end MathProof

end areaEnclosedByMidpoints_l714_714986


namespace remainder_div_x_plus_1_l714_714291

noncomputable def polynomial1 : Polynomial ℝ := Polynomial.X ^ 11 - 1

theorem remainder_div_x_plus_1 :
  Polynomial.eval (-1) polynomial1 = -2 := by
  sorry

end remainder_div_x_plus_1_l714_714291


namespace ceil_neg_sqrt_64_div_9_l714_714736

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l714_714736


namespace arithmetic_progression_geometric_progression_l714_714798

-- Arithmetic Progression
theorem arithmetic_progression (a : ℕ → ℤ) 
    (h_arith_1 : a 4 + a 7 = 2) 
    (h_arith_2 : a 5 * a 6 = -8) 
    (A : ∀ n m : ℕ, (a n - a m) = (n - m) * (a 2 - a 1)) : 
    a 1 * a 10 = -728 := 
begin 
    sorry 
end

-- Geometric Progression
theorem geometric_progression (a : ℕ → ℤ) 
    (h_geom_1 : a 4 + a 7 = 2) 
    (h_geom_2 : a 5 * a 6 = -8) 
    (G : ∀ n m : ℕ, (a n * a m) = (a 1 * (a 2 ^ (n-1))) * (a 1 * (a 2 ^ (m-1)))) : 
    a 1 + a 10 = -7 := 
begin 
    sorry 
end

end arithmetic_progression_geometric_progression_l714_714798


namespace solution_set_inequality_l714_714452

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - 1 / a) > 0) ↔ (a < x ∧ x < 1 / a) :=
by
  sorry

end solution_set_inequality_l714_714452


namespace solve_for_s_l714_714239

-- Definition of the condition
def condition (s : ℝ) : Prop := (s - 60) / 3 = (6 - 3 * s) / 4

-- Theorem stating that if the condition holds, then s = 19.85
theorem solve_for_s (s : ℝ) : condition s → s = 19.85 := 
by {
  sorry -- Proof is skipped as per requirements
}

end solve_for_s_l714_714239


namespace find_five_digit_number_l714_714061

open Matrix

def grid := Matrix (Fin 6) (Fin 6) ℕ

def valid_grid (g : grid) : Prop :=
  (∀ i : Fin 6, Finset.univ.card (Finset.image (λ j, g i j) Finset.univ) = 6 ∧
                Finset.univ.card (Finset.image (λ i, g i j) Finset.univ) = 6) ∧
  g ⟨1, _⟩ ⟨1, _⟩ = 5 ∧ g ⟨2, _⟩ ⟨3, _⟩ = 6

noncomputable def last_row_first_five_digits (g : grid) : ℕ :=
  g ⟨5, _⟩ ⟨0, _⟩ * 10^4 + g ⟨5, _⟩ ⟨1, _⟩ * 10^3 + g ⟨5, _⟩ ⟨2, _⟩ * 10^2 +
  g ⟨5, _⟩ ⟨3, _⟩ * 10^1 + g ⟨5, _⟩ ⟨4, _⟩

theorem find_five_digit_number : ∃ g : grid, valid_grid g ∧ last_row_first_five_digits g = 46123 :=
by
  sorry

end find_five_digit_number_l714_714061


namespace probability_ann_ben_in_photo_l714_714352

-- Defines the problem setting and parameters.
def ann_lap_time : ℕ := 75
def ben_lap_time : ℕ := 60
def min_time : ℕ := 12 * 60
def max_time : ℕ := 15 * 60
def one_sixth_track : ℚ := 1 / 6

-- Define the main theorem with the final correct probability as the solution
theorem probability_ann_ben_in_photo :
  ∀ t : ℕ, min_time ≤ t ∧ t ≤ max_time → 
  (t % ben_lap_time ≤ ben_lap_time * one_sixth_track) ∧ 
  (t % ann_lap_time ≤ ann_lap_time * one_sixth_track) →
  (1 / 6 : ℚ) := 
sorry

end probability_ann_ben_in_photo_l714_714352


namespace ratio_PQT_PTR_ratio_PQT_PQR_l714_714856

open Real

-- Define the problem conditions
variables {P Q R T : Type} [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited T]

-- Assume the lengths of QT, TR, and QR
variables (QT TR QR : ℝ)
variables (h : ℝ) -- the common height

-- Additional required constraints for the problem
axiom lengths : QT = 9 ∧ TR = 12 ∧ QR = QT + TR

-- Definition of areas using height
noncomputable def area_PQT : ℝ := 0.5 * QT * h
noncomputable def area_PTR : ℝ := 0.5 * TR * h
noncomputable def area_PQR : ℝ := 0.5 * QR * h

-- Proof statements
theorem ratio_PQT_PTR : area_PQT QT h / area_PTR TR h = 3 / 4 :=
by {
  rw [area_PQT, area_PTR], -- use the area definitions
  sorry
}

theorem ratio_PQT_PQR : area_PQT QT h / area_PQR QR h = 3 / 7 :=
by {
  rw [area_PQT, area_PQR], -- use the area definitions
  sorry
}

end ratio_PQT_PTR_ratio_PQT_PQR_l714_714856


namespace pascal_triangle_coprime_power_prime_l714_714754

theorem pascal_triangle_coprime_power_prime (n : ℕ) : 
  (∀ᶠ k in at_top, ∀ i, 0 ≤ i ∧ i ≤ k → Nat.gcd (Nat.choose k i) n = 1) → 
  ∃ p : ℕ, p.prime ∧ (∃ e : ℕ, n = p ^ e) :=
by
  sorry

end pascal_triangle_coprime_power_prime_l714_714754


namespace largest_C_l714_714042

def sum_div (x : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range n, 1 / (x k)

def sum_x3_plus_2x (x : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range n, (x k)^3 + 2 * (x k)

def sum_sqrt (x : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in finset.range n, nat.sqrt ((x k)^2 + 1)

theorem largest_C (x : fin 100 → ℕ) : 
  (∀ i j: fin 100, i ≠ j → x i ≠ x j) →
  (∑ k in finset.range 100, (1 / (x k)) * (sum_x3_plus_2x x 100) - (sum_sqrt x 100)^2) ≥ 33340000 := 
  by 
  sorry

end largest_C_l714_714042


namespace simplify_trig_expression_l714_714961

theorem simplify_trig_expression (x : ℝ) (h₁ : sin x ≠ 0) (h₂ : cos x ≠ -1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) := 
sorry

end simplify_trig_expression_l714_714961


namespace ceil_neg_sqrt_frac_l714_714741

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l714_714741


namespace g_24_divisible_by_4_pow_8_l714_714301

-- Let's define the function g(x) as described in the conditions
def g (x : Nat) : Nat :=
  List.prod (List.filter (λ k, k % 2 = 0) (List.range' 1 x))

-- The main statement we want to prove
theorem g_24_divisible_by_4_pow_8 : ∃ n : Nat, n = 8 ∧ 4^n ∣ g 24 :=
begin
  use 8,
  split,
  { refl },
  { sorry }
end

end g_24_divisible_by_4_pow_8_l714_714301


namespace solution_set_inequality_l714_714120

theorem solution_set_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x, f x + (deriv^[2] f) x < 1) (h_f0 : f 0 = 2018) :
  ∀ x, x > 0 → f x < 2017 * Real.exp (-x) + 1 :=
by
  sorry

end solution_set_inequality_l714_714120


namespace positive_solution_range_l714_714853

theorem positive_solution_range (a : ℝ) (h : a > 0) (x : ℝ) : (∃ x, (a / (x + 3) = 1 / 2) ∧ x > 0) ↔ a > 3 / 2 := by
  sorry

end positive_solution_range_l714_714853


namespace sum_of_squares_of_solutions_l714_714578

def is_solution (x y : ℝ) : Prop := 
  9 * y^2 - 4 * x^2 = 144 - 48 * x ∧ 
  9 * y^2 + 4 * x^2 = 144 + 18 * x * y

theorem sum_of_squares_of_solutions : 
  (∀ (x y : ℝ), is_solution x y → x^2 + y^2 ∈ {0, 16, 36}) → 
  ∑ (x, y) in {(0, 4), (0, -4), (6, 0)}, x^2 + y^2 = 68 :=
by 
  intros h
  -- the steps can be filled later
  sorry

end sum_of_squares_of_solutions_l714_714578


namespace factorial_sum_equals_arithmetic_sum_l714_714723

theorem factorial_sum_equals_arithmetic_sum (k n : ℕ) (h_pos_k : 0 < k) (h_pos_n : 0 < n) :
  (∑ i in Finset.range (k + 1), Nat.factorial i) = (∑ i in Finset.range (n + 1), i) ↔ 
  (k = 1 ∧ n = 1) ∨ (k = 2 ∧ n = 2) ∨ (k = 5 ∧ n = 17) :=
sorry

end factorial_sum_equals_arithmetic_sum_l714_714723


namespace james_total_cost_l714_714330

def subscription_cost (base_cost : ℕ) (free_hours : ℕ) (extra_hour_cost : ℕ) (movie_rental_cost : ℝ) (streamed_hours : ℕ) (rented_movies : ℕ) : ℝ :=
  let extra_hours := max (streamed_hours - free_hours) 0
  base_cost + extra_hours * extra_hour_cost + rented_movies * movie_rental_cost

theorem james_total_cost 
  (base_cost : ℕ)
  (free_hours : ℕ)
  (extra_hour_cost : ℕ)
  (movie_rental_cost : ℝ)
  (streamed_hours : ℕ)
  (rented_movies : ℕ)
  (h_base_cost : base_cost = 15)
  (h_free_hours : free_hours = 50)
  (h_extra_hour_cost : extra_hour_cost = 2)
  (h_movie_rental_cost : movie_rental_cost = 0.10)
  (h_streamed_hours : streamed_hours = 53)
  (h_rented_movies : rented_movies = 30) :
  subscription_cost base_cost free_hours extra_hour_cost movie_rental_cost streamed_hours rented_movies = 24 := 
by {
  sorry
}

end james_total_cost_l714_714330


namespace circles_separate_l714_714833

def circles_position (R1 R2 d : ℝ) : Prop :=
  R1 ≠ R2 ∧ R1 + R2 = d → "Separate"

theorem circles_separate
  (R1 R2 d : ℝ)
  (h1 : R1 ≠ R2)
  (h2 : R1 + R2 = d)
  (h3 : ∀ x : ℝ, (x^2 - 2*R1*x + R2^2 - d*(R2 - R1) = 0) → x ∈ ℝ) :
  circles_position R1 R2 d :=
by
  sorry

end circles_separate_l714_714833


namespace determine_b_for_constant_remainder_l714_714048

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end determine_b_for_constant_remainder_l714_714048


namespace correct_analogical_reasoning_l714_714293

-- Definitions of the statements in the problem
def statement_A : Prop := ∀ (a b : ℝ), a * 3 = b * 3 → a = b → a * 0 = b * 0 → a = b
def statement_B : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → (a * b) * c = a * c * b * c
def statement_C : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → c ≠ 0 → (a + b) / c = a / c + b / c
def statement_D : Prop := ∀ (a b : ℝ) (n : ℕ), (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n

-- The theorem stating that option C is the only correct analogical reasoning
theorem correct_analogical_reasoning : statement_C ∧ ¬statement_A ∧ ¬statement_B ∧ ¬statement_D := by
  sorry

end correct_analogical_reasoning_l714_714293


namespace probability_of_at_least_one_pen_l714_714305

/-- Definitions of probabilities of possessing a ballpoint and ink pen,
and assuming the independence of the events --/
namespace Probability
noncomputable def P_A : ℚ := 3 / 5
noncomputable def P_B : ℚ := 2 / 3
noncomputable def P_A_and_B : ℚ := P_A * P_B
noncomputable def P_A_or_B : ℚ := P_A + P_B - P_A_and_B

/-- Theorem: Prove the probability of possessing at least one type of pen --/
theorem probability_of_at_least_one_pen : P_A_or_B = 13 / 15 :=
by
  sorry

end Probability

end probability_of_at_least_one_pen_l714_714305


namespace common_ratio_of_geometric_sequence_l714_714415

variable (a : ℕ → ℝ) (d : ℝ)
variable (a1 : ℝ) (h_d : d ≠ 0)
variable (h_arith : ∀ n, a (n + 1) = a n + d)

theorem common_ratio_of_geometric_sequence :
  (a 0 = a1) →
  (a 4 = a1 + 4 * d) →
  (a 16 = a1 + 16 * d) →
  (a1 + 4 * d) / a1 = (a1 + 16 * d) / (a1 + 4 * d) →
  (a1 + 16 * d) / (a1 + 4 * d) = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l714_714415


namespace hyperbola_standard_equation_range_of_k_l714_714809

theorem hyperbola_standard_equation :
  ∃ a b : ℝ, a = sqrt 3 ∧ b = 1 ∧ hyperbola_of_equation (x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

theorem range_of_k (k : ℝ) :
  (line_equation y = k * x + sqrt 2) →
  (hyperbola_line_intersect_at_two_distinct_points (x^2 / 3 - y^2 = 1) y = k * x + sqrt 2) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (coordinates_of_points x₁ y₁ x₂ y₂) → 
  dot_product (⟨x₁, y₁⟩) (⟨x₂, y₂⟩) > 2) →
  (1 / 3 < k^2 ∧ k^2 < 1) :=
sorry

end hyperbola_standard_equation_range_of_k_l714_714809


namespace reciprocal_of_negative_one_sixth_l714_714311

theorem reciprocal_of_negative_one_sixth : ∃ x : ℚ, - (1/6) * x = 1 ∧ x = -6 :=
by
  use -6
  constructor
  . sorry -- Need to prove - (1 / 6) * (-6) = 1
  . sorry -- Need to verify x = -6

end reciprocal_of_negative_one_sixth_l714_714311


namespace percent_increase_first_quarter_l714_714704

theorem percent_increase_first_quarter (P : ℝ) (X : ℝ) (h1 : P > 0) 
  (end_of_second_quarter : P * 1.8 = P*(1 + X / 100) * 1.44) : 
  X = 25 :=
by
  sorry

end percent_increase_first_quarter_l714_714704


namespace parabola_intersects_x_axis_find_m_value_l714_714411

-- Define the quadratic function and conditions
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2
def m_condition (m : ℝ) : Prop := m ≠ 0 ∧ m ≠ 2

-- Prove that the discriminant is positive given the conditions
theorem parabola_intersects_x_axis (m : ℝ) 
  (h : m_condition m) : 
  let Δ := (m + 2)^2 - 4 * m * 2 in Δ > 0 := 
by
  have hnz : m ≠ 0 := h.left
  have hneq2 : m ≠ 2 := h.right
  let Δ := (m + 2)^2 - 4 * m * 2
  have hΔ : Δ = (m - 2)^2 := by
    calc
      (m + 2)^2 - 4 * m * 2
          = m^2 + 4m + 4 - 8m    : by ring
      ... = m^2 - 4m + 4         : by norm_num
      ... = (m - 2)^2            : by ring
  have hΔ_pos : (m - 2)^2 > 0 := by
    have h_ne_zero : m - 2 ≠ 0 := by intro h_eq; exact hneq2 h_eq.symm
    exact pow_pos (sub_ne_zero_of_ne hneq2) 2
  exact hΔ ▸ hΔ_pos

-- Prove that m = 1 when the roots are integers
theorem find_m_value (m : ℝ) 
  (h : m_condition m) 
  (h_int : ∃ x1 x2 : ℤ, (quadratic_function m x1 = 0) ∧ (quadratic_function m x2 = 0)) : 
  m = 1 :=
by
  have hnz : m ≠ 0 := h.left
  have hneq2 : m ≠ 2 := h.right
  cases h_int with x1 hx1
  cases hx1 with x2 hx2
  have h_factor : quadratic_function m x1 * quadratic_function m x2 = 0 := by
    apply and.intro (hx2.left) (hx2.right)
  -- Since x1 and x2 are integers and roots of the polynomial, m must be a positive integer divisor of 2.
  sorry

end parabola_intersects_x_axis_find_m_value_l714_714411


namespace area_triangle_BCD_length_CD_l714_714106

-- Definition of triangle and conditions
def triangle (A B C : Point) : Prop := True

variables (A B C D : Point)
variables (BC : ℝ) (angle_DBC : ℝ) (CD₁ CD₂ : ℝ)
variables (AB : ℝ) (angle_C : Prop) (sin_A : ℝ)

-- Conditions
def condition₁ : Prop := D ∈ segment AC  -- D is a point on AC
def condition₂ : Prop := BC = 2 * Real.sqrt 2
def condition₃ : Prop := angle_DBC = 45
def condition₄ : Prop := CD₁ = 2 * Real.sqrt 5
def condition₅ : Prop := AB = 6 * Real.sqrt 2
def condition₆ : Prop := sin_A = Real.sqrt 10 / 10
def condition₇ : Prop := acute_angle C -- angle C is acute

-- Problem (I)
theorem area_triangle_BCD :
  (triangle A B C) → condition₁ → condition₂ → condition₃ → condition₄ →
  calc_area B C D = 6 := by sorry

-- Problem (II)
theorem length_CD :
  (triangle A B C) → condition₁ → condition₂ → condition₃ → condition₅ → 
  condition₆ → condition₇ →
  CD₂ = Real.sqrt 5 := by sorry

end area_triangle_BCD_length_CD_l714_714106


namespace millionaire_allocation_l714_714611

def numWaysToAllocateMillionaires (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem millionaire_allocation :
  numWaysToAllocateMillionaires 13 3 = 36 :=
by
  sorry

end millionaire_allocation_l714_714611


namespace range_of_a_l714_714785

def f (x : ℝ) : ℝ :=
  |Real.logBase 3 x|

theorem range_of_a (a : ℝ) : (f a > f 2) ↔ (0 < a ∧ a < 1/2) ∨ (2 < a) :=
by
  sorry

end range_of_a_l714_714785


namespace cos_120_eq_neg_half_l714_714017

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714017


namespace N_is_necessary_but_not_sufficient_l714_714441

-- Define sets M and N
def M := { x : ℝ | 0 < x ∧ x < 1 }
def N := { x : ℝ | -2 < x ∧ x < 1 }

-- State the theorem to prove that "a belongs to N" is necessary but not sufficient for "a belongs to M"
theorem N_is_necessary_but_not_sufficient (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M → False) :=
by sorry

end N_is_necessary_but_not_sufficient_l714_714441


namespace find_50th_integer_l714_714592

theorem find_50th_integer :
  let digits := [1, 2, 3, 4, 5],
      permutations := multiset.permutations digits,
      sorted_permutations := multiset.sort int.lt permutations
  in sorted_permutations.nth 49 = some [3, 1, 2, 5, 4] :=
sorry

end find_50th_integer_l714_714592


namespace is_periodic_l714_714250

noncomputable def f : ℝ → ℝ := sorry

axiom domain (x : ℝ) : true
axiom not_eq_neg1_and_not_eq_0 (x : ℝ) : f x ≠ -1 ∧ f x ≠ 0
axiom functional_eq (x y : ℝ) : f (x - y) = - (f x / (1 + f y))

theorem is_periodic : ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end is_periodic_l714_714250


namespace winning_strategy_l714_714215

noncomputable def winning_player (n : ℕ) (h : n ≥ 2) : String :=
if n = 2 ∨ n = 4 ∨ n = 8 then "Ariane" else "Bérénice"

theorem winning_strategy (n : ℕ) (h : n ≥ 2) :
  (winning_player n h = "Ariane" ↔ (n = 2 ∨ n = 4 ∨ n = 8)) ∧
  (winning_player n h = "Bérénice" ↔ ¬ (n = 2 ∨ n = 4 ∨ n = 8)) :=
sorry

end winning_strategy_l714_714215


namespace smallest_n_l714_714405

theorem smallest_n (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
    (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n)
    (h3 : ∀ n : ℕ, S n = ∑ k in finset.range n, a (k + 1)) :
  ∃ n : ℕ, n ≥ 11 ∧ S n > 1025 :=
by
  sorry

end smallest_n_l714_714405


namespace solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l714_714832

section B_zero

variables {x y z b : ℝ}

-- Given conditions for the first system when b = 0
variables (hb_zero : b = 0)
variables (h1 : x + y + z = 0)
variables (h2 : x^2 + y^2 - z^2 = 0)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_zero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_zero

section B_nonzero

variables {x y z b : ℝ}

-- Given conditions for the first system when b ≠ 0
variables (hb_nonzero : b ≠ 0)
variables (h1 : x + y + z = 2 * b)
variables (h2 : x^2 + y^2 - z^2 = b^2)
variables (h3 : 3 * x * y * z - x^3 - y^3 - z^3 = b^3)

theorem solve_system_b_nonzero :
  ∃ x y z, 3 * x * y * z - x^3 - y^3 - z^3 = b^3 :=
by { sorry }

end B_nonzero

section Second_System

variables {x y z a : ℝ}

-- Given conditions for the second system
variables (h4 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
variables (h5 : x + y + 2 * z = 4 * (a^2 + 1))
variables (h6 : z^2 - x * y = a^2)

theorem solve_second_system :
  ∃ x y z, z^2 - x * y = a^2 :=
by { sorry }

end Second_System

end solve_system_b_zero_solve_system_b_nonzero_solve_second_system_l714_714832


namespace milk_savings_l714_714470

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end milk_savings_l714_714470


namespace focal_length_of_ellipse_l714_714997

theorem focal_length_of_ellipse :
  let a := sqrt 5
  let b := sqrt 3
  let c := sqrt (a^2 - b^2)
  2 * c = 2 * sqrt 2 :=
by
  let a := sqrt 5
  let b := sqrt 3
  let c := sqrt (a^2 - b^2)
  show 2 * c = 2 * sqrt 2
  sorry

end focal_length_of_ellipse_l714_714997


namespace ducks_in_smaller_pond_l714_714481

structure PondDucks (smaller larger : ℕ) :=
  (green_smaller : ℕ := nat.ltd [smaller].toNat.toNat)
  (green_larger : ℕ := nat.ltd [0.20 * smaller].toNat.toNat)
  (green_total : ℕ := 0.20 * smaller + 12)
  (total : ℕ := smaller + 80)

variables (smaller larger : ℕ)
def green_small := 0.20 * (smaller : ℝ)
def green_large := 0.15 * 80
def total_green := green_small smaller + green_large

def expected_green_ducks := 0.16 * (smaller + 80 : ℝ)

theorem ducks_in_smaller_pond (hyp : total_green smaller = expected_green_ducks smaller)
  : smaller = 20 :=
by
  sorry

# In this case, total_green smaller corresponds to the total number of green ducks,
# and expected_green_ducks smaller corresponds to the expected 16% of the total ducks.
# We want to prove that given the conditions, the smaller pond must have 20 ducks.

end ducks_in_smaller_pond_l714_714481


namespace lambda_in_regular_hexagon_l714_714537

theorem lambda_in_regular_hexagon (A₁ A₂ A₃ A₄ A₅ A₆ B₁ B₂ B₃ B₄ B₅ B₆ : Type) 
  (is_regular_hexagon : regular_hexagon A₁ A₂ A₃ A₄ A₅ A₆) 
  (divides_ratio : ∀ i, divides_ratio A_{i-1} B_i A_{i+1} (λ/(1-λ)))
  (collinear_condition : ∀ i, collinear {A_i, B_i, B_{i+2}}) :
  λ = 1 / real.sqrt 3 := 
sorry

end lambda_in_regular_hexagon_l714_714537


namespace no_regular_n_gon_lattice_vertices_l714_714799

theorem no_regular_n_gon_lattice_vertices (n : ℕ) (h : n ≠ 4) : ¬(∃ (P : Fin n → ℤ × ℤ), ∀ i j, i ≠ j → ∥P i - P j∥ = ∥P 0 - P 1∥) := sorry

end no_regular_n_gon_lattice_vertices_l714_714799


namespace manuscript_acceptance_prob_manuscript_acceptance_distribution_l714_714329

-- Define the probabilities given in the problem
def initial_review_pass_prob := 0.5
def third_review_pass_prob := 0.3

-- Define the events for passing both initial reviews and one initial review and the third review
variable (passes_both_initial : Event)
variable (passes_one_initial_one_third : Event)

-- Define the acceptance event
def acceptance : Event :=
  passes_both_initial ∨ passes_one_initial_one_third

theorem manuscript_acceptance_prob :
  P(acceptance) = 0.4 :=
  sorry

-- Define X as the number of accepted manuscripts out of 4, following binomial distribution
def X : ℕ → ℝ := binomial 4 0.4

theorem manuscript_acceptance_distribution :
  ∀ k, (k ∈ {0, 1, 2, 3, 4}) → (P(X = k) = (nat.choose 4 k) * (0.4)^k * (0.6)^(4 - k)) :=
  sorry

end manuscript_acceptance_prob_manuscript_acceptance_distribution_l714_714329


namespace isosceles_triangle_angle_between_vectors_l714_714172

theorem isosceles_triangle_angle_between_vectors 
  (α β γ : ℝ) 
  (h1: α + β + γ = 180)
  (h2: α = 120) 
  (h3: β = γ):
  180 - β = 150 :=
sorry

end isosceles_triangle_angle_between_vectors_l714_714172


namespace tree_inequality_l714_714679

noncomputable def sum_of_edge_products (edges : List (ℕ × ℕ)) (x : ℕ → ℝ) : ℝ :=
  edges.sum (λ e => x e.fst * x e.snd)

theorem tree_inequality 
  (n : ℕ) (x : Fin n → ℝ) (edges : List (ℕ × ℕ)) (h_tree : ∀i, i < n) (hn : n ≥ 2)
  (S : ℝ := sum_of_edge_products edges x) :
  Real.sqrt (n - 1) * ((Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2)) ≥ 2 * S := 
by
  sorry

end tree_inequality_l714_714679


namespace average_speed_l714_714678

   theorem average_speed (x : ℝ) : 
     let s1 := 40
     let s2 := 20
     let d1 := x
     let d2 := 2 * x
     let total_distance := d1 + d2
     let time1 := d1 / s1
     let time2 := d2 / s2
     let total_time := time1 + time2
     total_distance / total_time = 24 :=
   by
     sorry
   
end average_speed_l714_714678


namespace circle_equation_l714_714251

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * y = 0

theorem circle_equation
  (x y : ℝ)
  (center_on_y_axis : ∃ r : ℝ, r > 0 ∧ x^2 + (y - r)^2 = r^2)
  (tangent_to_x_axis : ∃ r : ℝ, r > 0 ∧ y = r)
  (passes_through_point : x = 3 ∧ y = 1) :
  equation_of_circle x y :=
by
  sorry

end circle_equation_l714_714251


namespace probability_not_siblings_l714_714484

noncomputable def num_individuals : ℕ := 6
noncomputable def num_pairs : ℕ := num_individuals / 2
noncomputable def total_pairs : ℕ := num_individuals * (num_individuals - 1) / 2
noncomputable def sibling_pairs : ℕ := num_pairs
noncomputable def non_sibling_pairs : ℕ := total_pairs - sibling_pairs

theorem probability_not_siblings :
  (non_sibling_pairs : ℚ) / total_pairs = 4 / 5 := 
by sorry

end probability_not_siblings_l714_714484


namespace grocer_pounds_purchased_l714_714640

-- Definitions of given conditions
def cost_price_per_pound := (0.50 : ℝ) / 3
def selling_price_per_pound := (1.00 : ℝ) / 4
def total_profit := 7.00

-- Theorem to prove the number of pounds purchased by the grocer
theorem grocer_pounds_purchased (x : ℝ) :
  total_profit = (selling_price_per_pound - cost_price_per_pound) * x → 
  x ≈ 84 :=
by
  sorry

end grocer_pounds_purchased_l714_714640


namespace simplify_fraction_l714_714572

theorem simplify_fraction (a b : ℕ) (h₁ : a = 84) (h₂ : b = 144) :
  a / gcd a b = 7 ∧ b / gcd a b = 12 := 
by
  sorry

end simplify_fraction_l714_714572


namespace max_value_inequality_l714_714776

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 4 :=
sorry

end max_value_inequality_l714_714776


namespace f_increasing_f_odd_zero_l714_714909

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Prove that f(x) is always an increasing function for any real a.
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- 2. Determine the value of a such that f(-x) + f(x) = 0 always holds.
theorem f_odd_zero (a : ℝ) : (∀ x : ℝ, f a (-x) + f a x = 0) → a = 1 :=
by
  sorry

end f_increasing_f_odd_zero_l714_714909


namespace circle_intervals_inequality_l714_714275

variable (n : ℕ) (m a : ℕ)
variable (F : set (set ℕ))
variable (maximal_intervals non_maximal_intervals : set (set ℕ))

-- Define the conditions
def is_interval (A : set ℕ) : Prop := -- condition that A is an interval
  sorry

def is_subinterval (A B : set ℕ) : Prop := -- condition that A ⊆ B
  A ⊆ B

def maximal (A : set ℕ) : Prop :=
  is_interval A ∧ ∀ B ∈ F, is_subinterval A B → A = B

def non_maximal (A : set ℕ) : Prop :=
  is_interval A ∧ ∃ B ∈ F, is_subinterval A B ∧ A ≠ B

-- Define the set of maximal and non-maximal intervals
def maximal_intervals := { A ∈ F | maximal A }
def non_maximal_intervals := { A ∈ F | non_maximal A }

-- Prove the main statement
theorem circle_intervals_inequality (h1 : n > 1)
    (h2 : ∀ A ∈ F, ∃ B ∈ F, B ≠ A ∧ is_subinterval A B ∨ is_subinterval B A)
    (h3 : m = set.cardinal maximal_intervals)
    (h4 : a = set.cardinal non_maximal_intervals) :
    n ≥ m + a / 2 :=
begin
  sorry
end

end circle_intervals_inequality_l714_714275


namespace prove_inequality_l714_714691

-- Define the sequence {b_n}
noncomputable def b_n (α : ℕ → ℕ) : ℕ → ℚ
| 1 := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b_n α n)

-- Example α values for simplification like α_k = 1
def example_α (k : ℕ) : ℕ := 1

-- The statement to be proved
theorem prove_inequality (α : ℕ → ℕ) (h : ∀ k, 0 < α k) : (b_n α 4 < b_n α 7) :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end prove_inequality_l714_714691


namespace cos_120_degrees_l714_714001

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l714_714001


namespace coefficient_x6_in_expansion_l714_714287

theorem coefficient_x6_in_expansion (f : ℕ → ℕ → ℕ) (a : ℤ) (b : ℤ) (n : ℕ):
  (f 6 2) * (a^4) * ((b*x^3)^2) = 135 * x^6 :=
by
  let a := 1
  let b := -3 * x^3
  let n := 6
  have binomial_theorem := λ (a : ℤ) (b : ℤ) (n : ℕ), 
    ∑ k in finset.range n, 
      (nat.choose n k) * a^(n - k) * b^k
  sorry

end coefficient_x6_in_expansion_l714_714287


namespace combined_age_l714_714234

-- Define the conditions given in the problem
def Hezekiah_age : Nat := 4
def Ryanne_age := Hezekiah_age + 7

-- The statement to prove
theorem combined_age : Ryanne_age + Hezekiah_age = 15 :=
by
  -- we would provide the proof here, but for now we'll skip it with 'sorry'
  sorry

end combined_age_l714_714234


namespace height_decrease_percentage_l714_714851

theorem height_decrease_percentage (b h : ℝ) (x : ℝ) (hb : b > 0) (hh : h > 0)
  (h_area : b * h = 1.1 * b * h * (1 - x / 100)) : x = 100 / 11 := 
begin
  sorry
end

end height_decrease_percentage_l714_714851


namespace Aiden_sleep_fraction_l714_714699

theorem Aiden_sleep_fraction (minutes_slept : ℕ) (hour_minutes : ℕ) (h : minutes_slept = 15) (k : hour_minutes = 60) :
  (minutes_slept : ℚ) / hour_minutes = 1/4 :=
by
  sorry

end Aiden_sleep_fraction_l714_714699


namespace polynomial_example_l714_714951

noncomputable def exists_poly : Prop :=
  ∃ P : polynomial ℝ, 
    (P.degree > 1) ∧ 
    (∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 → P.eval a + P.eval b + P.eval c ≥ 2021) ∧ 
    (∃∞ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ P.eval a + P.eval b + P.eval c = 2021)

theorem polynomial_example : exists_poly :=
  sorry

end polynomial_example_l714_714951


namespace problem1_l714_714711

theorem problem1 : 2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15 / 2 := 
by
  sorry

end problem1_l714_714711


namespace midpoint_perpendicular_to_line_l714_714187

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := sorry
variable (A B C D E : Point)

def is_parallelogram (A B C D : Point) : Prop := 
Collinear A B C D ∧ Collinear B C D A

def is_isosceles (A B C : Point) : Prop := dist A B = dist A C

def is_perpendicular (A B C D : Point) : Prop := 
Angle A B C = 90 ∧ Angle B C D = 90

theorem midpoint_perpendicular_to_line (A B C D E : Point)
  (h1 : is_parallelogram A B C D)
  (h2 : dist C D = dist C E) :
  is_perpendicular (midpoint A E) (midpoint B C) D E := sorry

end midpoint_perpendicular_to_line_l714_714187


namespace total_days_2005_to_2010_l714_714449

theorem total_days_2005_to_2010 :
  let leap_year_days := 366
  let regular_year_days := 365
  let years := [2005, 2006, 2007, 2008, 2009, 2010]
  let leap_years := [2008]
  let non_leap_years := [2005, 2006, 2007, 2009, 2010]
  (leap_years.length * leap_year_days + non_leap_years.length * regular_year_days) = 2191 :=
by
  let leap_year_days := 366
  let regular_year_days := 365
  let years := [2005, 2006, 2007, 2008, 2009, 2010]
  let leap_years := [2008]
  let non_leap_years := [2005, 2006, 2007, 2009, 2010]
  have h1 : leap_years.length = 1 := rfl
  have h2 : non_leap_years.length = 5 := rfl
  have total_leap_days : leap_years.length * leap_year_days = 366 := by rw [h1]; exact rfl
  have total_non_leap_days : non_leap_years.length * regular_year_days = 1825 := by rw [h2]; exact rfl
  have total_days : (leap_years.length * leap_year_days + non_leap_years.length * regular_year_days) = 2191 := 
    by rw [total_leap_days, total_non_leap_days]; exact rfl
  exact total_days

end total_days_2005_to_2010_l714_714449


namespace pipe_flow_rate_is_correct_l714_714318

-- Definitions for the conditions
def tank_capacity : ℕ := 10000
def initial_water : ℕ := tank_capacity / 2
def fill_time : ℕ := 60
def drain1_rate : ℕ := 1000
def drain1_interval : ℕ := 4
def drain2_rate : ℕ := 1000
def drain2_interval : ℕ := 6

-- Calculation based on conditions
def total_water_needed : ℕ := tank_capacity - initial_water
def drain1_loss (time : ℕ) : ℕ := (time / drain1_interval) * drain1_rate
def drain2_loss (time : ℕ) : ℕ := (time / drain2_interval) * drain2_rate
def total_drain_loss (time : ℕ) : ℕ := drain1_loss time + drain2_loss time

-- Target flow rate for the proof
def total_fill (time : ℕ) : ℕ := total_water_needed + total_drain_loss time
def pipe_flow_rate : ℕ := total_fill fill_time / fill_time

-- Statement to prove
theorem pipe_flow_rate_is_correct : pipe_flow_rate = 500 := by  
  sorry

end pipe_flow_rate_is_correct_l714_714318


namespace number_of_ordered_pairs_l714_714209

theorem number_of_ordered_pairs (n : ℕ) (k : ℕ) (A B : Finset ℕ) (a : ℕ) (b : ℕ) 
  (h_nonemptyA : A.nonempty) (h_nonemptyB : B.nonempty) 
  (h_disjoint : A ∩ B = ∅) (h_union : A ∪ B = Finset.range (n + 1)) 
  (h_an_in_B : a ∈ B) (h_bn_in_A : b ∈ A) :
  (n = 2 * k + 1 → (∃ a_n, a_n = 2 ^ (2 * k - 1))) ∧
  (n = 2 * k + 2 → (∃ a_n, a_n = 2 ^ (2 * k) - nat.choose (2 * k) k)) :=
by 
  sorry

end number_of_ordered_pairs_l714_714209


namespace starting_number_of_SetB_l714_714236

-- Define Set A as the set of integers from 4 to 15
def SetA : Set ℤ := { n : ℤ | 4 ≤ n ∧ n ≤ 15 }

-- Define condition for Set B: integers from some number startB to 20
def SetB (startB : ℤ) : Set ℤ := { n : ℤ | startB ≤ n ∧ n ≤ 20 }

-- Define the proof statement that the starting number of Set B is 6
theorem starting_number_of_SetB (startB : ℤ) (h_intersect: (SetA ∩ SetB startB).card = 10) : startB = 6 :=
by
  sorry

end starting_number_of_SetB_l714_714236


namespace expression_meaningful_l714_714280

theorem expression_meaningful (x : ℝ) : 
  (x - 1 ≠ 0 ∧ true) ↔ x ≠ 1 := 
sorry

end expression_meaningful_l714_714280


namespace find_50th_integer_l714_714594

theorem find_50th_integer :
  let digits := [1, 2, 3, 4, 5],
      permutations := multiset.permutations digits,
      sorted_permutations := multiset.sort int.lt permutations
  in sorted_permutations.nth 49 = some [3, 1, 2, 5, 4] :=
sorry

end find_50th_integer_l714_714594


namespace parabola_tangent_to_line_condition_l714_714465

theorem parabola_tangent_to_line_condition (a k : ℝ) (hk : k ≠ 6) :
  (∀ x : ℝ, ax^2 + 6 = 2x + k → (4 * a * k - 24 * a + 4 = 0)) → a = -1 / (k - 6) := sorry

end parabola_tangent_to_line_condition_l714_714465


namespace female_wins_probability_l714_714497

theorem female_wins_probability :
  let p_alexandr := 3 * p_alexandra,
      p_evgeniev := (1 / 3) * p_evgenii,
      p_valentinov := (3 / 2) * p_valentin,
      p_vasilev := 49 * p_vasilisa,
      p_alexandra := 1 / 4,
      p_alexandr := 3 / 4,
      p_evgeniev := 1 / 12,
      p_evgenii := 11 / 12,
      p_valentinov := 3 / 5,
      p_valentin := 2 / 5,
      p_vasilev := 49 / 50,
      p_vasilisa := 1 / 50,
      p_female := 
        (1 / 4) * p_alexandra + 
        (1 / 4) * p_evgeniev + 
        (1 / 4) * p_valentinov + 
        (1 / 4) * p_vasilisa 
  in p_female ≈ 0.355 := 
sorry

end female_wins_probability_l714_714497


namespace simplify_trig_expression_l714_714982

theorem simplify_trig_expression (x : ℝ) (hx : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l714_714982


namespace correct_statements_l714_714129

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x, f (1 - x) + f (1 + x) = 0

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

theorem correct_statements (f : ℝ → ℝ) :
  is_even f →
  is_monotonically_increasing f (-1) 0 →
  satisfies_condition f →
  (f (-3) = 0 ∧
   is_monotonically_increasing f 1 2 ∧
   is_symmetric_about_line f 1) :=
by
  intros h_even h_mono h_cond
  sorry

end correct_statements_l714_714129


namespace calculate_f_50_l714_714585

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_50 (f : ℝ → ℝ) (h_fun : ∀ x y : ℝ, f (x * y) = y * f x) (h_f2 : f 2 = 10) :
  f 50 = 250 :=
by
  sorry

end calculate_f_50_l714_714585


namespace solve_for_x_l714_714290

noncomputable def is_satisfied (x : ℝ) : Prop :=
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2

theorem solve_for_x :
  ∀ x : ℝ, 0 < x → x ≠ 1 ↔ is_satisfied x := by
  sorry

end solve_for_x_l714_714290


namespace g_five_four_eq_twentyfive_l714_714992

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom fg_equation : ∀ x : ℝ, x ≥ 1 → f(g(x)) = x^2
axiom gf_equation : ∀ x : ℝ, x ≥ 1 → g(f(x)) = x^4
axiom g_25 : g(25) = 25

theorem g_five_four_eq_twentyfive : [g(5)]^4 = 25 := 
by 
    sorry

end g_five_four_eq_twentyfive_l714_714992


namespace ceiling_of_expression_l714_714056

theorem ceiling_of_expression : 
  (Real.ceil (4 * (8 - (3 / 4)))) = 29 := 
by {
  sorry
}

end ceiling_of_expression_l714_714056


namespace area_of_triangle_l714_714907

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l714_714907


namespace recipe_book_cost_l714_714553

theorem recipe_book_cost (R : ℝ) 
    (h_baking_dish : 2 * R) 
    (h_ingredients : 5 * 3 = 15) 
    (h_apron : R + 1) 
    (h_total_cost : R + 2 * R + 15 + (R + 1) = 40) 
    : R = 6 := 
sorry

end recipe_book_cost_l714_714553


namespace b_4_lt_b_7_l714_714687

def α : ℕ → ℕ := λ k, k

def b : ℕ → ℚ
| 1      := 1 + (1 / (α 1))
| n + 1  := 1 + (1 / (α 1 + b_aux n))

noncomputable def b_aux : ℕ → ℚ
| 1      := (1 / (α 1))
| n + 1  := (1 / (α 1 + b_aux n))

theorem b_4_lt_b_7 : b 4 < b 7 := by
  sorry

end b_4_lt_b_7_l714_714687


namespace tank_capacity_l714_714677

-- Conditions
def half_full_water (V : ℝ) : Prop := V / 2
def fill_rate : ℝ := 1 / 2 -- kiloliters per minute
def drain_rate1 : ℝ := 1 / 4 -- kiloliters per minute
def drain_rate2 : ℝ := 1 / 6 -- kiloliters per minute
def net_fill_rate : ℝ := fill_rate - (drain_rate1 + drain_rate2) -- Net fill rate in kiloliters per minute
def time_to_fill : ℝ := 6 -- minutes
def added_water : ℝ := net_fill_rate * time_to_fill -- amount of water added in kiloliters

-- Problem: Prove the capacity of the tank in liters
theorem tank_capacity : (added_water * 2 * 1000) = 1000 := sorry

end tank_capacity_l714_714677


namespace ceil_neg_sqrt_l714_714749

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l714_714749


namespace graph_shifted_upwards_by_three_is_E_l714_714718

def f : ℝ → ℝ := λ x, if x ≥ -3 ∧ x <= 0 then -2 - x
                    else if x > 0 ∧ x <= 2 then real.sqrt (4 - (x - 2)^2) - 2
                    else if x > 2 ∧ x <= 3 then 2 * (x - 2)
                    else 0

def h : ℝ → ℝ := λ x, f x + 3

theorem graph_shifted_upwards_by_three_is_E :
    (∀ x : ℝ, h x = f x + 3) :=
sorry

end graph_shifted_upwards_by_three_is_E_l714_714718


namespace ratio_sum_ineq_l714_714418

theorem ratio_sum_ineq 
  (a b α β : ℝ) 
  (hαβ : 0 < α ∧ 0 < β) 
  (h_range : α ≤ a ∧ a ≤ β ∧ α ≤ b ∧ b ≤ β) : 
  (b / a + a / b ≤ β / α + α / β) ∧ 
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β ∨ a = β ∧ b = α)) :=
by
  sorry

end ratio_sum_ineq_l714_714418


namespace trigonometric_sum_identity_l714_714563

theorem trigonometric_sum_identity :
  (∑ k in Finset.range 89, 1 / (Real.cos (k * Real.pi / 180) * Real.cos ((k+1) * Real.pi / 180))) = Real.cos (Real.pi / 180) / (Real.sin (Real.pi / 180) ^ 2) :=
by
  sorry

end trigonometric_sum_identity_l714_714563


namespace find_number_l714_714620

theorem find_number (x : ℤ) (h : 27 + 2 * x = 39) : x = 6 :=
sorry

end find_number_l714_714620


namespace shop_width_l714_714256

theorem shop_width 
  (monthly_rent : ℝ) 
  (shop_length : ℝ) 
  (annual_rent_per_sqft : ℝ) 
  (width : ℝ) 
  (monthly_rent_eq : monthly_rent = 2244) 
  (shop_length_eq : shop_length = 22) 
  (annual_rent_per_sqft_eq : annual_rent_per_sqft = 68) 
  (width_eq : width = 18) : 
  (12 * monthly_rent) / annual_rent_per_sqft / shop_length = width := 
by 
  sorry

end shop_width_l714_714256


namespace ceil_neg_sqrt_l714_714750

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l714_714750


namespace total_cost_of_pencils_and_erasers_l714_714462

theorem total_cost_of_pencils_and_erasers 
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (pencils_bought : ℕ)
  (erasers_bought : ℕ)
  (total_cost_dollars : ℝ)
  (cents_to_dollars : ℝ)
  (hc : pencil_cost = 2)
  (he : eraser_cost = 5)
  (hp : pencils_bought = 500)
  (he2 : erasers_bought = 250)
  (cents_to_dollars_def : cents_to_dollars = 100)
  (total_cost_calc : total_cost_dollars = 
    ((pencils_bought * pencil_cost + erasers_bought * eraser_cost : ℕ) : ℝ) / cents_to_dollars) 
  : total_cost_dollars = 22.50 :=
sorry

end total_cost_of_pencils_and_erasers_l714_714462


namespace cube_side_length_l714_714274

theorem cube_side_length (V : ℝ) (hV : V = 729) : ∃ (a : ℝ), a ^ 3 = V ∧ a = 9 :=
by
exists 9
split
sorry

end cube_side_length_l714_714274


namespace ceil_neg_sqrt_frac_l714_714743

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l714_714743


namespace top_left_corner_value_l714_714857

theorem top_left_corner_value :
  (∃ t : ℕ, 
    (∀ i j : ℕ, 
      (i = 0 ∨ j = 0 → matrix (Fin 6) (Fin 6) ℕ i j = t) ∧
      (i ≠ 0 ∧ j ≠ 0 → matrix (Fin 6) (Fin 6) ℕ i j = matrix (Fin 6) (Fin 6) ℕ (i - 1) j + matrix (Fin 6) (Fin 6) ℕ i (j - 1))
    ) ∧ 
    matrix (Fin 6) (Fin 6) ℕ 5 5 = 2016
  ) → t = 8 :=
by
  sorry

end top_left_corner_value_l714_714857


namespace real_roots_iff_le_one_l714_714161

theorem real_roots_iff_le_one (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 :=
by
  sorry

end real_roots_iff_le_one_l714_714161


namespace explosion_point_trajectory_l714_714444

noncomputable def explosion_trajectory (A B : ℝ × ℝ) (v_sound : ℝ) (time_lag : ℝ) (M : ℝ × ℝ) : Prop :=
  let MA := real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) in
  let MB := real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) in
  abs (MA - MB) = v_sound * time_lag

theorem explosion_point_trajectory :
  ∃ (A B : ℝ × ℝ) (v_sound : ℝ) (time_lag : ℝ) (M : ℝ × ℝ),
    A = (-400, 0) ∧ B = (400, 0) ∧
    v_sound = 340 ∧ time_lag = 2 ∧
    explosion_trajectory A B v_sound time_lag M ∧ 
    (branch_of_hyperbola_closer_to_B A B M) :=
begin
  -- Proof is not required, hence we use sorry to complete the statement.
  sorry
end

-- Assume branch_of_hyperbola_closer_to_B is a definition that needs to be defined elsewhere.

end explosion_point_trajectory_l714_714444


namespace proof_problem_l714_714136

theorem proof_problem
  (a b c d : ℝ)
  (f g : ℝ → ℝ)
  (P : ℝ × ℝ)
  (hf : f = λ x, x^2 + a * x + b)
  (hg : g = λ x, Real.exp x * (c * x + d))
  (hP : P = (0, 2))
  (htangent : ∀ y : ℝ → ℝ, y 0 = 2 ∧ y' 0 = 4 → (y = f ∨ y = g)) :
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ≥ -2 → f x ≤ k * g x) ↔ (1 ≤ k ∧ k ≤ Real.exp 2)) := by
sory

end proof_problem_l714_714136


namespace inequality_proof_l714_714564

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) :=
by
  sorry

end inequality_proof_l714_714564


namespace max_a3_a18_in_arithmetic_seq_l714_714422

variable {A : Type} [LinearOrder A] [Field A] [CharZero A]

-- Definitions of conditions: 
def isArithmeticSequence (a : ℕ → A) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sumFirstN (a : ℕ → A) (N : ℕ) :=
  ∑ k in finset.range N, a k

-- The problem statement: 
theorem max_a3_a18_in_arithmetic_seq 
  (a : ℕ → A)
  (h_seq : isArithmeticSequence a)
  (pos_terms : ∀ n, 0 < a n) 
  (sum_20 : sumFirstN a 20 = 100) :
  a 3 * a 18 ≤ 25 := sorry

end max_a3_a18_in_arithmetic_seq_l714_714422


namespace zero_ending_of_A_l714_714272

theorem zero_ending_of_A (A : ℕ) (h : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c ∣ A ∧ a + b + c = 8 → a * b * c = 10) : 
  (10 ∣ A) ∧ ¬(100 ∣ A) :=
by
  sorry

end zero_ending_of_A_l714_714272


namespace charles_initial_bananas_l714_714296

theorem charles_initial_bananas (W C : ℕ) (h1 : W = 48) (h2 : C = C - 35 + W - 13) : C = 35 := by
  -- W = 48
  -- Charles loses 35 bananas
  -- Willie will have 13 bananas
  sorry

end charles_initial_bananas_l714_714296


namespace percentage_saved_is_11_percent_l714_714674

-- Define the initial conditions
def amount_saved : ℝ := 5.50
def amount_spent : ℝ := 44.00
def original_price : ℝ := amount_spent + amount_saved

-- Calculate the percentage saved
def percentage_saved : ℝ := (amount_saved / original_price) * 100

-- The theorem we want to prove
theorem percentage_saved_is_11_percent : percentage_saved ≈ 11 :=
by
  sorry -- Proof goes here

end percentage_saved_is_11_percent_l714_714674


namespace total_black_dots_l714_714276

-- Define the types and their properties
structure ButterflyType where
  blackDots : ℕ
  count : ℕ

def TypeA : ButterflyType := { blackDots := 12, count := 145 }
def TypeB : ButterflyType := { blackDots := 8.5.toNat, count := 112 }
def TypeC : ButterflyType := { blackDots := 19, count := 140 }

-- Prove the total number of black dots
theorem total_black_dots : (TypeA.blackDots * TypeA.count + TypeB.blackDots * TypeB.count + TypeC.blackDots * TypeC.count) = 5352 := by
  sorry

end total_black_dots_l714_714276


namespace find_50th_integer_l714_714593

theorem find_50th_integer :
  let digits := [1, 2, 3, 4, 5],
      permutations := multiset.permutations digits,
      sorted_permutations := multiset.sort int.lt permutations
  in sorted_permutations.nth 49 = some [3, 1, 2, 5, 4] :=
sorry

end find_50th_integer_l714_714593


namespace problem_opposites_l714_714295

variable (a b : ℝ)

-- Define what it means for two expressions to be opposites
def are_opposites (x y : ℝ) : Prop := x = -y

-- State the conditions of the problem as variables
def expr1 := a - b
def expr2 := -a - b
def expr3 := a + b
def expr4 := 1 - a
def expr5 := a + 1
def expr6 := -a + b

-- State the proof problem in Lean 4 terms
theorem problem_opposites :
  are_opposites expr3 expr2 ∧ are_opposites expr6 expr1 := sorry

end problem_opposites_l714_714295


namespace monotonicity_of_f_range_of_m_l714_714823

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a / 2) * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) (m : ℝ) : ℝ := f x a + (m / 2) * x

theorem monotonicity_of_f
  (a : ℝ)
  (h : ∀ x : ℝ, 0 < x → x^2 - (a / 2) * Real.log x)
  (h_deriv : ∀ x : ℝ, 0 < x → deriv (fun x => x^2 - (a / 2) * Real.log x) x = 1 - a)
  (a_eq_1 : a = 1) :
  (∀ x : ℝ, (1 / 2) < x → (deriv (f x a)) > 0) ∧ (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → (deriv (f x a)) < 0) := by
  sorry

theorem range_of_m
  (m : ℝ)
  (h : ∀ x : ℝ, 1 < x → g x 1 m > 0) :
  m ∈ Set.Ici (-2) := by
  sorry

end monotonicity_of_f_range_of_m_l714_714823


namespace tangent_line_length_l714_714127

noncomputable def circle_tangent_length : ℝ -> ℝ -> ℝ -> ℝ -> ℝ -> ℝ -> ℝ :=
λ x y a b c d, 
  let center_x := (2*a)/2 in
  let center_y := (2*c)/2 in
  let radius := real.sqrt((a^2 + c^2 - b)/2) in
  let distance_PC := real.sqrt((center_x - x)^2 + (center_y - y)^2) in
  real.sqrt(distance_PC ^ 2 - radius ^ 2)

theorem tangent_line_length : 
  circle_tangent_length 1 0 1 2 3 9 = 2 * real.sqrt 2 :=
by
  sorry

end tangent_line_length_l714_714127


namespace happy_pair_205800_35k_count_l714_714284

-- Defining the gcd and checks
def is_happy_pair (m n : ℕ) : Prop :=
  let d := Nat.gcd m n
  ∃ (k : ℕ), d = k * k

-- Defining the factorization and gcd criteria
noncomputable def possible_k (k : ℕ) : Prop :=
  is_happy_pair 205800 (35 * k) ∧ k ≤ 2940

-- Prove that the number of possible values of k is 30
theorem happy_pair_205800_35k_count :
  (Finset.card (Finset.filter possible_k (Finset.range 2941))) = 30 := sorry

end happy_pair_205800_35k_count_l714_714284


namespace complete_square_l714_714524

theorem complete_square {a b c : ℤ} (h1 : 100*x^2 + 60*x - 49 = 0) (h2 : 0 < a)
  (h3 : 100*x^2 + 60*x + 9 = c + 49) 
  (h4 : (a*x + b)^2 = c) 
  (h5 : √100 = a) 
  (h6 : 2*a*b = 60) : 
  a + b + c = 71 :=
sorry

end complete_square_l714_714524


namespace triangle_area_l714_714901

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area (a b : ℝ × ℝ) : 
  let area_parallelogram := (a.1 * b.2 - a.2 * b.1).abs in
  (1 / 2) * area_parallelogram = 4.5 :=
by
  sorry

end triangle_area_l714_714901


namespace sum_of_digits_least_n_l714_714206

theorem sum_of_digits_least_n :
  ∃ n : ℕ, (n > 500 ∧ gcd 42 (n + 80) = 14 ∧ gcd (n + 42) 80 = 40) ∧ (nat.digits 10 n).sum = 17 :=
sorry

end sum_of_digits_least_n_l714_714206


namespace length_of_AE_l714_714544

theorem length_of_AE 
  (A B C D E : Type*)
  [Triangle A B C]
  (AB BC CA AD : ℝ)
  (AB_eq : AB = 13)
  (BC_eq : BC = 14)
  (CA_eq : CA = 15)
  (AD_altitude_from_A : AD)
  (ω1 ω2 : Circle)
  [incircle_ABC_D : ω1 ∈ incircle_ABD A B D]
  [incircle_ABC_F : ω2 ∈ incircle_ACD A C D]
  (common_tangent_intersects_AD_at_E : (∀ (adc: Real), tangent adc ω1 ω2) ) :
  AE = 7 := 
sorry

end length_of_AE_l714_714544


namespace plane_intersects_dodecahedron_to_form_hexagon_l714_714174

-- Define a regular dodecahedron in space
structure Dodecahedron (P : Type) [EuclideanSpace P] := 
(symmetric_edges : ∀ (v1 v2 : P), Prop)
(symmetric_faces : ∀ (face1 face2 : set P), Prop)
(centre : P)

noncomputable def regular_dodecahedron : Dodecahedron P := {
  -- Regular dodecahedron's properties and symmetries
  symmetric_edges := sorry,
  symmetric_faces := sorry,
  centre := sorry
}

-- Prove there are 30 distinct ways for a plane to intersect the dodecahedron forming a regular hexagon.
theorem plane_intersects_dodecahedron_to_form_hexagon (D : Dodecahedron P) : 
  ∃! n : ℕ, n = 30 := 
sorry

end plane_intersects_dodecahedron_to_form_hexagon_l714_714174


namespace inheritance_amount_l714_714894

-- Define necessary variables and constants
variable (x : ℝ) -- x is Ms. Sarah Conner's inheritance
variable (federal_tax_rate : ℝ := 0.25)
variable (state_tax_rate : ℝ := 0.15)
variable (luxury_tax_rate : ℝ := 0.05)
variable (total_tax_paid : ℝ := 20000)

-- Conditions stated as Lean definitions
def federal_tax (x : ℝ) : ℝ := federal_tax_rate * x
def remaining_after_federal (x : ℝ) : ℝ := x - federal_tax x

def state_tax (x : ℝ) : ℝ := state_tax_rate * remaining_after_federal x
def remaining_after_state (x : ℝ) : ℝ := remaining_after_federal x - state_tax x

def luxury_tax (x : ℝ) : ℝ := luxury_tax_rate * remaining_after_state x

def total_tax (x : ℝ) : ℝ := federal_tax x + state_tax x + luxury_tax x

-- Theorem to prove
theorem inheritance_amount : total_tax x = total_tax_paid → x = 50700 := by sorry

end inheritance_amount_l714_714894


namespace tensor_computation_l714_714848

-- Define the tensor operation
def tensor (a b : ℝ) : ℝ := (a + b) / (a - b)

-- State the proof goal
theorem tensor_computation : tensor (tensor 8 6) 2 = 9 / 5 :=
by
  -- The proof goes here
  sorry

end tensor_computation_l714_714848


namespace basketball_team_win_rate_l714_714321

theorem basketball_team_win_rate (won_first : ℕ) (total : ℕ) (remaining : ℕ)
    (desired_rate : ℚ) (x : ℕ) (H_won : won_first = 30) (H_total : total = 100)
    (H_remaining : remaining = 55) (H_desired : desired_rate = 13/20) :
    (30 + x) / 100 = 13 / 20 ↔ x = 35 := by
    sorry

end basketball_team_win_rate_l714_714321


namespace pieces_left_after_third_day_l714_714556

theorem pieces_left_after_third_day : 
  let initial_pieces := 1000
  let first_day_pieces := initial_pieces * 0.10
  let remaining_first_day := initial_pieces - first_day_pieces
  let second_day_pieces := remaining_first_day * 0.20
  let remaining_second_day := remaining_first_day - second_day_pieces
  let third_day_pieces := remaining_second_day * 0.30
  let pieces_left := remaining_second_day - third_day_pieces 
  in pieces_left = 504 :=
by
  sorry

end pieces_left_after_third_day_l714_714556


namespace problem_integer_part_of_S_l714_714541

noncomputable def closest_integer (n : ℕ) : ℕ :=
if h : ∃ k : ℕ, (k - 1 / 2 : ℝ) < real.sqrt n ∧ real.sqrt n < (k + 1 / 2 : ℝ)
then classical.some h else 0

instance : DecidableEq ℝ := classical.decEq

noncomputable def S : ℝ :=
(1 / closest_integer 1) + (1 / closest_integer 2) + ... + (1 / closest_integer 2000)

theorem problem_integer_part_of_S : ⌊S⌋ = 88 := 
sorry

end problem_integer_part_of_S_l714_714541


namespace composite_prop_true_l714_714800

def p : Prop := ∀ (x : ℝ), x > 0 → x + (1/(2*x)) ≥ 1

def q : Prop := ∀ (x : ℝ), x > 1 → (x^2 + 2*x - 3 > 0)

theorem composite_prop_true : p ∨ q :=
by
  sorry

end composite_prop_true_l714_714800


namespace analytical_expression_and_period_monotonic_intervals_l714_714086

noncomputable def m : ℝ → ℝ × ℝ := λ x, (√3 * Real.sin x, 2)
noncomputable def n : ℝ → ℝ × ℝ := λ x, (2 * Real.cos x, Real.cos x ^ 2)
noncomputable def f : ℝ → ℝ := λ x, let dot_product := (m x).fst * (n x).fst + (m x).snd * (n x).snd in dot_product

theorem analytical_expression_and_period :
  (∀ x, f(x) = 2 * Real.sin(2 * x + Real.pi / 6) + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi) :=
begin
  sorry
end

theorem monotonic_intervals :
  ∀ k : ℤ, ∀ x, (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) ↔
  (∀ x1 x2, (k * Real.pi - Real.pi / 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ k * Real.pi + Real.pi / 6) → f(x1) < f(x2)) :=
begin
  sorry
end

end analytical_expression_and_period_monotonic_intervals_l714_714086


namespace not_a_cube_l714_714201

theorem not_a_cube (a b : ℤ) : ¬ ∃ c : ℤ, a^3 + b^3 + 4 = c^3 := 
sorry

end not_a_cube_l714_714201


namespace smallest_possible_value_l714_714938

def smallest_ab (a b : ℕ) : Prop :=
  a^a % b^b =  0 ∧ a % b ≠ 0 ∧ Nat.gcd b 210 = 1

theorem smallest_possible_value : ∃ (a b : ℕ), smallest_ab a b ∧ a + b = 374 :=
by {
  existsi 253,
  existsi 121,
  unfold smallest_ab,
  simp,
  split,
  { sorry }, -- Proof that 253^253 % 121^121 = 0
  split,
  { exact dec_trivial }, -- Proof that 253 % 121 ≠ 0
  { exact dec_trivial }, -- Proof that Nat.gcd 121 210 = 1
  { refl },
}

end smallest_possible_value_l714_714938


namespace compute_expression_l714_714715

theorem compute_expression : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3 ^ 2 = 28 := by
  sorry

end compute_expression_l714_714715


namespace total_savings_l714_714471

-- Define the conditions
def cost_of_two_packs : ℝ := 2.50
def cost_of_single_pack : ℝ := 1.30

-- Define the problem statement
theorem total_savings :
  let price_per_pack_when_in_set := cost_of_two_packs / 2,
      savings_per_pack := cost_of_single_pack - price_per_pack_when_in_set,
      total_packs := 10 * 2,
      total_savings := savings_per_pack * total_packs in
  total_savings = 1 :=
by
  sorry

end total_savings_l714_714471


namespace find_expression_value_l714_714788

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l714_714788


namespace collinear_points_distance_k_l714_714394

/-- 
Given five collinear points where the distances between every pair of points 
are sorted as follows: 2, 4, 5, 7, 8, k, 13, 15, 17, 19, 
prove that k equals 12.
-/
theorem collinear_points_distance_k (A B C D E : ℝ) 
  (h : A < B < C < D < E)
  (distances : list ℝ := [dist A B, dist A C, dist A D, dist A E, dist B C, dist B D, dist B E, dist C D, dist C E, dist D E].sort) :
  distances = [2, 4, 5, 7, 8, k, 13, 15, 17, 19] → k = 12 :=
by sorry

-- Auxiliary definition for distance between two points on a line
def dist (x y : ℝ) : ℝ := abs (x - y)

end collinear_points_distance_k_l714_714394


namespace problem1_problem2_l714_714825

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 4|

-- Problem 1: Prove that the solution set for f(x) > 2 is (5/3, 3)
theorem problem1 : {x : ℝ | f x > 2} = set.Ioo (5 / 3) 3 := sorry

-- Problem 2: Prove that if the solution set for f(x) > t^2 + 2t is non-empty, then t ∈ (-3, 1)
theorem problem2 (t : ℝ) (h : ∃ x : ℝ, f x > t ^ 2 + 2 * t) : t ∈ set.Ioo (-3 : ℝ) 1 := sorry

end problem1_problem2_l714_714825


namespace ceil_eval_l714_714059

-- Define the ceiling function and the arithmetic operations involved
example : Real := let inside := (8 - (3 / 4)) in 
                  let multiplied := 4 * inside in 
                  ⌈multiplied⌉
                  
theorem ceil_eval :  ⌈4 * (8 - (3 / 4))⌉ = 29 := 
by
-- We'll skip the proof part using sorry
sorry

end ceil_eval_l714_714059


namespace b_4_lt_b_7_l714_714684

def α : ℕ → ℕ := λ k, k

def b : ℕ → ℚ
| 1      := 1 + (1 / (α 1))
| n + 1  := 1 + (1 / (α 1 + b_aux n))

noncomputable def b_aux : ℕ → ℚ
| 1      := (1 / (α 1))
| n + 1  := (1 / (α 1 + b_aux n))

theorem b_4_lt_b_7 : b 4 < b 7 := by
  sorry

end b_4_lt_b_7_l714_714684


namespace intersection_count_l714_714728

-- Declare the main theorem
theorem intersection_count (m n : ℕ) :
  (∃ m n : ℕ, (∀ a b : ℕ, (703 * a ≥ 299 * b ∧ 703 * (a + 1) ≤ 299 * (b + 1) → 
  (m = m + 1 ) ∧ (dist (a, b) line eqn (703*x, 299*x) < 1/5 → n = n + 1 ))) ∧ 
  m + n = 2109 :=
  sorry

end intersection_count_l714_714728


namespace smallest_m_divisible_by_15_l714_714914

-- Define conditions
def is_largest_prime_2011_digit (q : ℕ) : Prop :=
  prime q ∧ (∃ p : ℕ, prime p ∧ (number_of_digits p = 2011 ∧ p > q))

def number_of_digits (n : ℕ) : ℕ :=
  nat.log10 n + 1

-- Main theorem statement
theorem smallest_m_divisible_by_15 (q : ℕ) (h : is_largest_prime_2011_digit q) :
  ∃ m : ℕ, (q^2 - m) % 15 = 0 ∧ m = 1 :=
sorry

end smallest_m_divisible_by_15_l714_714914


namespace jackson_running_miles_end_program_l714_714888

theorem jackson_running_miles_end_program (initial_miles: ℕ) (weeks: ℕ) (additional_miles_per_week: ℕ): 
  initial_miles = 3 → weeks = 4 → additional_miles_per_week = 1 → 
  (initial_miles + weeks * additional_miles_per_week) = 7 :=
by {
  intro h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end jackson_running_miles_end_program_l714_714888


namespace arithmetic_progression_geometric_progression_l714_714797

-- Arithmetic Progression
theorem arithmetic_progression (a : ℕ → ℤ) 
    (h_arith_1 : a 4 + a 7 = 2) 
    (h_arith_2 : a 5 * a 6 = -8) 
    (A : ∀ n m : ℕ, (a n - a m) = (n - m) * (a 2 - a 1)) : 
    a 1 * a 10 = -728 := 
begin 
    sorry 
end

-- Geometric Progression
theorem geometric_progression (a : ℕ → ℤ) 
    (h_geom_1 : a 4 + a 7 = 2) 
    (h_geom_2 : a 5 * a 6 = -8) 
    (G : ∀ n m : ℕ, (a n * a m) = (a 1 * (a 2 ^ (n-1))) * (a 1 * (a 2 ^ (m-1)))) : 
    a 1 + a 10 = -7 := 
begin 
    sorry 
end

end arithmetic_progression_geometric_progression_l714_714797


namespace b_investment_correct_l714_714642

-- Constants for shares and investments
def a_investment : ℕ := 11000
def a_share : ℕ := 2431
def b_share : ℕ := 3315
def c_investment : ℕ := 23000

-- Goal: Prove b's investment given the conditions
theorem b_investment_correct (b_investment : ℕ) (h : 2431 * b_investment = 11000 * 3315) :
  b_investment = 15000 := by
  sorry

end b_investment_correct_l714_714642


namespace geometric_sequence_common_ratio_positive_l714_714813

theorem geometric_sequence_common_ratio_positive (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) (h_prod : (∏ i in finset.range 10, a i) = 32) :
  q > 0 :=
sorry

end geometric_sequence_common_ratio_positive_l714_714813


namespace emma_correct_percentage_l714_714166

theorem emma_correct_percentage (t : ℕ) (lt : t > 0)
  (liam_correct_alone : ℝ := 0.70)
  (liam_overall_correct : ℝ := 0.82)
  (emma_correct_alone : ℝ := 0.85)
  (joint_error_rate : ℝ := 0.05)
  (liam_solved_together_correct : ℝ := liam_overall_correct * t - liam_correct_alone * (t / 2)) :
  (emma_correct_alone * (t / 2) + (1 - joint_error_rate) * liam_solved_together_correct) / t * 100 = 87.15 :=
by
  sorry

end emma_correct_percentage_l714_714166


namespace find_radius_of_wheel_l714_714264

noncomputable def radius_of_wheel 
  (speed_kmh : ℝ) 
  (revolutions_per_minute : ℝ) 
  (π_approx : ℝ) : ℝ :=
  let speed_cm_min := speed_kmh * 100000 / 60
  in speed_cm_min / (revolutions_per_minute * 2 * π_approx)

theorem find_radius_of_wheel :
  radius_of_wheel 66 250.22747952684256 3.141592653589793 ≈ 70.007 := 
sorry

end find_radius_of_wheel_l714_714264


namespace ratio_shortest_to_middle_tree_l714_714609

theorem ratio_shortest_to_middle_tree (height_tallest : ℕ) 
  (height_middle : ℕ) (height_shortest : ℕ)
  (h1 : height_tallest = 150) 
  (h2 : height_middle = (2 * height_tallest) / 3) 
  (h3 : height_shortest = 50) : 
  height_shortest / height_middle = 1 / 2 := by sorry

end ratio_shortest_to_middle_tree_l714_714609


namespace fiftieth_number_is_31254_l714_714595

open List

-- Define the list of digits
def digits := [1, 2, 3, 4, 5]

-- All permutations of digits
def permutations := List.perms digits

-- Function to convert a list of digits to a number
def listToNumber (l : List Nat) : Nat :=
  l.foldl (λ acc d => 10 * acc + d) 0

-- Define the ordered list of numbers
def orderedNumbers := permutations.map listToNumber |>.sort (≤)

-- The 50th integer in the list
noncomputable def fiftiethNumber := orderedNumbers.get (50 - 1) (by sorry)

-- Prove that the 50th integer is 31254
theorem fiftieth_number_is_31254 : fiftiethNumber = 31254 :=
  by
  sorry

end fiftieth_number_is_31254_l714_714595


namespace annual_income_is_1500_l714_714300

noncomputable def investment_amount : ℝ := 6800
noncomputable def dividend_rate : ℝ := 0.30
noncomputable def price_per_stock : ℝ := 136
noncomputable def face_value_stock : ℝ := 100

theorem annual_income_is_1500 :
  let number_of_stocks := investment_amount / price_per_stock in
  let dividend_per_stock := dividend_rate * face_value_stock in
  let annual_income := number_of_stocks * dividend_per_stock in
  annual_income = 1500 :=
by 
  sorry

end annual_income_is_1500_l714_714300


namespace area_of_circle_above_line_l714_714285

theorem area_of_circle_above_line :
  let circle_eq := (x y : ℝ) → x^2 - 12 * x + y^2 - 18 * y + 89 = 0
  let line_y := 4
  let area := 28 * Real.pi
  ∀ x y : ℝ, circle_eq x y → y > line_y → area = 28 * Real.pi := by
    sorry

end area_of_circle_above_line_l714_714285


namespace find_T2023_l714_714199

noncomputable def a_seq (n : ℕ) : ℕ := sorry  -- Define the sequence a_n (to be filled)

def S_sum (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a_seq (k + 1))

noncomputable def b_seq (n : ℕ) : ℚ :=
  let a_n := a_seq n in
  (-1 : ℚ)^n * 4 * a_n / (4 * (a_n^2) - 1)

def T_sum (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b_seq (k + 1))

theorem find_T2023 : T_sum 2023 = - 4048 / 4047 := sorry

end find_T2023_l714_714199


namespace simplify_trig_expression_l714_714978

theorem simplify_trig_expression (x : ℝ) (hx : x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := 
by 
  sorry

end simplify_trig_expression_l714_714978


namespace simplify_trig_l714_714974

theorem simplify_trig (x : ℝ) (h_cos_sin : cos x ≠ -1) (h_sin_ne_zero : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
    sorry

end simplify_trig_l714_714974


namespace range_for_which_f_ge_neg1_l714_714406

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 1/2 * x + 1 else -(x - 1)^2

theorem range_for_which_f_ge_neg1 : 
  (setOf (λ x, f x ≥ -1)) = set.Icc (-4 : ℝ) 2 :=
by
  sorry

end range_for_which_f_ge_neg1_l714_714406


namespace student_A_claps_4_times_l714_714990

/-- Define the counting sequence of students A, B, C, and D -/
def student_sequence (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1 + 4 * (n / 4)
  | 1 => 2 + 4 * (n / 4)
  | 2 => 3 + 4 * (n / 4)
  | _ => 4 + 4 * (n / 4)
  end

/-- Define if a number is a multiple of 3 -/
def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

/-- Define if a number is called by student A -/
def is_called_by_A (n : ℕ) : Prop :=
  n % 4 = 0

/-- The main theorem stating the number of times student A claps -/
theorem student_A_claps_4_times : 
  (finset.range 50).filter (λ n, is_multiple_of_3 (student_sequence (n + 1)) ∧ is_called_by_A (n + 1)).card = 4 :=
by
  sorry

end student_A_claps_4_times_l714_714990


namespace beautiful_permutation_n6_exists_beautiful_permutation_l714_714397

-- Definition of a beautiful permutation
def beautiful_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i < n, a i + a (n + i) = 2 * n + 1) ∧
  (∀ i < 2 * n - 1, ∀ j < 2 * n - 1, i < j → (a i - a (i + 1)) % (2 * n + 1) ≠ (a j - a (j + 1)) % (2 * n + 1))

-- Specific beautiful permutation for n = 6
noncomputable def n6_permutation : ℕ → ℕ
| 0 := 1
| 1 := 3
| 2 := 5
| 3 := 7
| 4 := 9
| 5 := 11
| 6 := 12
| 7 := 10
| 8 := 8
| 9 := 6
| 10 := 4
| 11 := 2
| _ := 0

-- Prove that n6_permutation is a beautiful permutation for n = 6
theorem beautiful_permutation_n6 : beautiful_permutation 6 n6_permutation :=
  sorry

-- Prove that there exists a beautiful permutation for every positive integer n
theorem exists_beautiful_permutation : ∀ n > 0, ∃ a : ℕ → ℕ, beautiful_permutation n a :=
  sorry

end beautiful_permutation_n6_exists_beautiful_permutation_l714_714397


namespace find_principal_sum_l714_714298

theorem find_principal_sum (SI R T : ℝ) (hSI : SI = 4034.25) (hR : R = 9) (hT : T = 5) :
  let P := SI / (R * T / 100) in P = 8965 := 
by 
  sorry 

end find_principal_sum_l714_714298


namespace prove_inequality_l714_714690

-- Define the sequence {b_n}
noncomputable def b_n (α : ℕ → ℕ) : ℕ → ℚ
| 1 := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b_n α n)

-- Example α values for simplification like α_k = 1
def example_α (k : ℕ) : ℕ := 1

-- The statement to be proved
theorem prove_inequality (α : ℕ → ℕ) (h : ∀ k, 0 < α k) : (b_n α 4 < b_n α 7) :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end prove_inequality_l714_714690


namespace cos_120_eq_neg_half_l714_714032

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714032


namespace number_of_small_pizzas_ordered_l714_714707

-- Define the problem conditions
def benBrothers : Nat := 2
def slicesPerPerson : Nat := 12
def largePizzaSlices : Nat := 14
def smallPizzaSlices : Nat := 8
def numLargePizzas : Nat := 2

-- Define the statement to prove
theorem number_of_small_pizzas_ordered : 
  ∃ (s : Nat), (benBrothers + 1) * slicesPerPerson - numLargePizzas * largePizzaSlices = s * smallPizzaSlices ∧ s = 1 :=
by
  sorry

end number_of_small_pizzas_ordered_l714_714707


namespace find_s_l714_714758

theorem find_s (s : ℝ) :
  (3 * (x:ℝ)^2 - 4 * x + 8) * (5 * x^2 + s * x + 15) = 15 * x^4 - 29 * x^3 + 87 * x^2 - 60 * x + 120 → s = -3 := by
sorrry

end find_s_l714_714758


namespace smallest_mu_real_number_l714_714371

theorem smallest_mu_real_number (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) :
  a^2 + b^2 + c^2 + d^2 ≤ ab + (3/2) * bc + cd :=
sorry

end smallest_mu_real_number_l714_714371


namespace gcd_204_85_l714_714999

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  have h1 : 204 = 2 * 85 + 34 := by rfl
  have h2 : 85 = 2 * 34 + 17 := by rfl
  have h3 : 34 = 2 * 17 := by rfl
  sorry

end gcd_204_85_l714_714999


namespace pyramid_lateral_faces_l714_714669

theorem pyramid_lateral_faces {T : Type} [Topology T] (b : T → Prop) (is_right_angle : ∀ t ∈ b, ∃ a b c : ℝ, a * a + b * b = c * c) : 
  ∃ l1 l2 l3 : T, (l1 ∈ b ∧ l2 ∈ b ∧ l3 ∈ b) → (∃ r1 r2 : T, (r1 ∈ {l1, l2, l3} ∧ r2 ∈ {l1, l2, l3}) ∧ 
  ∀ r ∈ {l1, l2, l3}, ¬ ∃ a b c : ℝ, a * a + b * b = c * c) :=
sorry

end pyramid_lateral_faces_l714_714669


namespace find_y_l714_714292

theorem find_y (y : ℕ) (h : (2 * y) / 5 = 10) : y = 25 :=
sorry

end find_y_l714_714292


namespace total_minutes_exercised_l714_714520

-- Defining the conditions
def Javier_minutes_per_day : Nat := 50
def Javier_days : Nat := 10

def Sanda_minutes_day_90 : Nat := 90
def Sanda_days_90 : Nat := 3

def Sanda_minutes_day_75 : Nat := 75
def Sanda_days_75 : Nat := 2

def Sanda_minutes_day_45 : Nat := 45
def Sanda_days_45 : Nat := 4

-- Main statement to prove
theorem total_minutes_exercised : 
  (Javier_minutes_per_day * Javier_days) + 
  (Sanda_minutes_day_90 * Sanda_days_90) +
  (Sanda_minutes_day_75 * Sanda_days_75) +
  (Sanda_minutes_day_45 * Sanda_days_45) = 1100 := by
  sorry

end total_minutes_exercised_l714_714520


namespace work_completion_days_l714_714306

theorem work_completion_days (d : ℝ) : (1 / 15 + 1 / d = 1 / 11.25) → d = 45 := sorry

end work_completion_days_l714_714306


namespace kamal_marks_english_l714_714892

theorem kamal_marks_english (M : ℕ) (P : ℕ) (C : ℕ) (B : ℕ) (avg_marks : ℕ) (E : ℕ) 
  (hM : M = 60) 
  (hP : P = 82) 
  (hC : C = 67) 
  (hB : B = 85) 
  (h_avg : avg_marks = 74) 
  (h_avg_calc : E + M + P + C + B = avg_marks * 5) : 
  E = 76 := 
begin 
  rw [hM, hP, hC, hB, h_avg] at h_avg_calc,
  linarith,
end

end kamal_marks_english_l714_714892


namespace cos_120_eq_neg_half_l714_714018

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714018


namespace number_of_combinations_l714_714579

-- Define the set of letters in "IMOHKPRELIM" and their respective frequencies
def letters : List Char := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L', 'I', 'M']

-- Count the number of distinct letters and calculate the total combinations
theorem number_of_combinations : 
  let distinct_letters := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L']
  let count_distinct := nat.combination 9 3
  let count_repeats := 2 * 8
  count_distinct + count_repeats = 100 :=
by
  let distinct_letters := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L']
  let count_distinct := nat.combination 9 3
  let count_repeats := 2 * 8
  show count_distinct + count_repeats = 100
  · sorry

end number_of_combinations_l714_714579


namespace area_of_triangle_COB_l714_714946

theorem area_of_triangle_COB (p : ℝ) (hC_bounds : 0 < p ∧ p < 15) :
  let C := (0 : ℝ, p),
      O := (0 : ℝ, 0),
      B := (15 : ℝ, 0) in
  let base := (B.1 - O.1) in
  let height := (C.2 - O.2) in
  (1 / 2) * base * height = (15 * p) / 2 := 
by
  simp only [C, O, B, base, height]
  rw [sub_self, sub_zero, mul_comm, mul_div_assoc, div_self]
  sorry

end area_of_triangle_COB_l714_714946


namespace initial_average_daily_production_l714_714777

variable (A : ℝ) -- Initial average daily production
variable (n : ℕ) -- Number of days

theorem initial_average_daily_production (n_eq_5 : n = 5) (new_production_eq_90 : 90 = 90) 
  (new_average_eq_65 : (5 * A + 90) / 6 = 65) : A = 60 :=
by
  sorry

end initial_average_daily_production_l714_714777


namespace greatest_k_divides_n_l714_714334

theorem greatest_k_divides_n (n : ℕ) (h₁ : number_of_divisors n = 48) (h₂ : number_of_divisors (5 * n) = 72) :
  ∃ k : ℕ, (5^k ∣ n) ∧ (∀ k' : ℕ, (5^(k + 1) ∣ n) → k' ≤ k) ∧ k = 1 := sorry

end greatest_k_divides_n_l714_714334


namespace curve_C2_eqn_l714_714420

theorem curve_C2_eqn (p : ℝ) (x y : ℝ) :
  (∃ x y, (x^2 - y^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (2 * p = 3/4)) →
  (y^2 = (3/2) * x) :=
by
  sorry

end curve_C2_eqn_l714_714420


namespace female_wins_probability_l714_714494

theorem female_wins_probability :
  let p_alexandr := 3 * p_alexandra,
      p_evgeniev := (1 / 3) * p_evgenii,
      p_valentinov := (3 / 2) * p_valentin,
      p_vasilev := 49 * p_vasilisa,
      p_alexandra := 1 / 4,
      p_alexandr := 3 / 4,
      p_evgeniev := 1 / 12,
      p_evgenii := 11 / 12,
      p_valentinov := 3 / 5,
      p_valentin := 2 / 5,
      p_vasilev := 49 / 50,
      p_vasilisa := 1 / 50,
      p_female := 
        (1 / 4) * p_alexandra + 
        (1 / 4) * p_evgeniev + 
        (1 / 4) * p_valentinov + 
        (1 / 4) * p_vasilisa 
  in p_female ≈ 0.355 := 
sorry

end female_wins_probability_l714_714494


namespace find_Ricky_sales_l714_714523

-- Definitions of the given conditions
def Katya_sales : ℕ := 8
def Ricky_sales : ℕ
def Tina_sales : ℕ := 2 * (Katya_sales + Ricky_sales)

-- The proof problem statement
theorem find_Ricky_sales : 
  (∃ R : ℕ, 
    Tina_sales = 2 * (Katya_sales + R) ∧ 
    Tina_sales = Katya_sales + 26 ∧ 
    R = 9) :=
sorry

end find_Ricky_sales_l714_714523


namespace find_min_value_p_s_l714_714197

variables {α β : ℝ}
variable  {S : set ℝ}
variable  (f : ℝ → ℝ)
variables (a b c : ℝ)

/-- conditions: α < 0 < β, S is the set such that polynomial f(x) - s has three different real roots -/
def condition₁ := α < 0
def condition₂ := 0 < β
def set_S := {s : ℝ | ∃ (p q r : ℝ), p < q ∧ q < r ∧ f (p) - s  = 0 ∧ f (q) - s = 0 ∧ f (r) - s = 0}
def polynomial := f = λ x, x * (x - α) * (x - β)

/-- goal: -/
theorem find_min_value_p_s (H1 : condition₁) (H2 : condition₂) (H3 : polynomial) (H4 : S = set_S) :
∃ s ∈ S, ∀ t ∈ S, p s ≤ p t ∧ p s = - (1 / 4) * (β - α)^2 := sorry

end find_min_value_p_s_l714_714197


namespace audrey_older_than_heracles_l714_714357

variable (A H : ℕ)
variable (hH : H = 10)
variable (hFutureAge : A + 3 = 2 * H)

theorem audrey_older_than_heracles : A - H = 7 :=
by
  have h1 : H = 10 := by assumption
  have h2 : A + 3 = 2 * H := by assumption
  -- Proof is omitted
  sorry

end audrey_older_than_heracles_l714_714357


namespace cos_120_eq_neg_half_l714_714027

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714027


namespace triangle_area_l714_714903

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l714_714903


namespace find_min_a_plus_b_l714_714931

open Nat

noncomputable def smallest_a_plus_b : ℕ :=
  let candidates := (1, 1).succ_powerset (1000, 1000).succ_powerset -- to explore a range up to 1000; this range should be set reasonably high
  candidates.filter (λ (a, b), (a^a % b^b = 0) ∧ (a % b ≠ 0) ∧ (gcd b 210 = 1)) |>.map (λ (a, b), a + b) |> min

theorem find_min_a_plus_b : smallest_a_plus_b = 374 := by
  sorry

end find_min_a_plus_b_l714_714931


namespace total_time_to_4864_and_back_l714_714304

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance : ℝ)
variable (Sboat : speed_boat = 14) (Sstream : speed_stream = 1.2) (Dist : distance = 4864)

theorem total_time_to_4864_and_back :
  let speed_downstream := speed_boat + speed_stream
  let speed_upstream := speed_boat - speed_stream
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_time := time_downstream + time_upstream
  total_time = 700 :=
by
  sorry

end total_time_to_4864_and_back_l714_714304


namespace best_play_maximum_product_l714_714561

-- Definition of the given problem conditions
def non_negative_eleven_sums_to_one (x : Fin 11 → ℝ) : Prop :=
  (∀ i, 0 ≤ x i) ∧ (∑ i, x i) = 1

-- Main theorem statement
theorem best_play_maximum_product {x : Fin 11 → ℝ} 
  (h : non_negative_eleven_sums_to_one x) :
  ∃ y : Fin 10 → ℝ, 
    (∀ i, y i = (x i) * (x (i + 1))) ∧ (∀ i, y i ≤ (1 / 40)) :=
sorry

end best_play_maximum_product_l714_714561


namespace Mabel_transactions_l714_714942

theorem Mabel_transactions (M : ℝ) (A : ℝ) (C : ℝ) (J : ℝ) (hA : A = 1.10 * M) 
  (hC : C = (2 / 3) * A) (hJ : J = C + 15) (hJ_val : J = 81) : M = 90 :=
by
  -- setting up the given equations
  have hC_eq : C = 66 := by linarith,
  have hM_eq : M = 66 / 0.7333 := by linarith,
  linarith -- M equals approximately 90

end Mabel_transactions_l714_714942


namespace fourth_degree_polynomial_representation_l714_714949

theorem fourth_degree_polynomial_representation (k4 k3 k2 k1 k0 : ℝ) (h : k4 ≠ 0) :
  ∃ (P Q R S : ℝ[X]), degree P = 2 ∧ degree Q = 2 ∧ degree R = 2 ∧ degree S = 2 ∧ 
  (∀ (x : ℝ), k4 * x^4 + k3 * x^3 + k2 * x^2 + k1 * x + k0 = P.eval (Q.eval x) + R.eval (S.eval x)) :=
sorry

end fourth_degree_polynomial_representation_l714_714949


namespace triangle_problem_l714_714184

noncomputable def angle_B (a b c : ℝ) (h : a^2 + c^2 = b^2 + real.sqrt 2 * a * c) : Prop :=
  arccos ((a^2 + c^2 - b^2) / (2 * a * c)) = π / 4

noncomputable def max_value (A C : ℝ) (hAC : C = 3 * π / 4 - A) : Prop :=
  ∀ A, A ∈ set.Ioo 0 (3 * π / 4) → real.sqrt 2 * real.cos A + real.cos C ≤ 1

theorem triangle_problem (a b c : ℝ) (h : a^2 + c^2 = b^2 + real.sqrt 2 * a * c) :
  angle_B a b c h ∧ max_value A C hAC := by 
  sorry

end triangle_problem_l714_714184


namespace determine_b_for_constant_remainder_l714_714047

theorem determine_b_for_constant_remainder (b : ℚ) :
  ∃ r : ℚ, ∀ x : ℚ,  (12 * x^3 - 9 * x^2 + b * x + 8) / (3 * x^2 - 4 * x + 2) = r ↔ b = -4 / 3 :=
by sorry

end determine_b_for_constant_remainder_l714_714047


namespace convex_power_function_l714_714764

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end convex_power_function_l714_714764


namespace difference_between_even_and_odd_sums_1500_l714_714288

def sum_first_n_even_numbers (n : ℕ) : ℕ := (n * (n - 1))
def sum_first_n_odd_numbers (n : ℕ) : ℕ := n^2

theorem difference_between_even_and_odd_sums_1500 :
  abs (sum_first_n_even_numbers 1500 - sum_first_n_odd_numbers 1500) = 1500 :=
by
  sorry

end difference_between_even_and_odd_sums_1500_l714_714288


namespace vector_dot_product_AP_PB_PC_l714_714855

variables {A B C M P : Type} -- Define the types of points
variables [inner_product_space ℝ (euclidean_space ℝ (fin 2))] -- Use real Euclidean space

-- Definitions of points and vectors 
variable (vec_AP vec_PM vec_PB vec_PC : euclidean_space ℝ (fin 2))
variables (hM_midpoint : (2 : ℝ) • M = B + C) -- M is the midpoint of BC
variables (hAM_eq_one : (norm (M - A) = 1)) -- AM = 1
variables (hP_on_AM : vec_AP = 2 • vec_PM) -- AP = 2 * PM

-- Proof statement
theorem vector_dot_product_AP_PB_PC (hAP_2_PM : vec_AP = 2 • vec_PM) : 
  inner_product_space.real_inner (euclidean_space ℝ (fin 2)) vec_AP (vec_PB + vec_PC) = 4 / 9 := 
by {
    sorry
}

end vector_dot_product_AP_PB_PC_l714_714855


namespace cos_120_eq_neg_half_l714_714025

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714025


namespace ceil_neg_sqrt_64_div_9_l714_714738

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l714_714738


namespace ellipse_equation_hyperbola_equation_l714_714314

-- Problem (1)
theorem ellipse_equation {a : ℝ} (F A B P : ℝ × ℝ) (h1 : A = (a, 0)) (h2 : B = (0, 1)) (h3 : P ∈ {p : ℝ × ℝ | p.1^2 + (p.2^2 / a^2) = 1}) 
  (h4 : P.2 = F.2) (h5 : F.1 = 0) (h6 : F.2 < 0) (h7 : ∃ (c : ℝ), c = sqrt (a^2 - 1)) (h8 : a^2 = 2) : 
  ∀ x y, (x, y) ∈ { p : ℝ × ℝ | p.1^2 + (p.2^2 / 2) = 1 } :=
by 
  sorry

-- Problem (2)
theorem hyperbola_equation (k : ℝ) (t : ℝ) (h1 : ∀ x, (0.5 * x) = abs (k * x))
  (h2 : |4| = |2 * sqrt t|) 
  (ht_pos : t > 0 ∨ t < 0) : 
  if ht_pos then
    ∀ x y, (x, y) ∈ { p : ℝ × ℝ | x^2 / 4 - y^2 = 1 }
  else
    ∀ x y, (x, y) ∈ { p : ℝ × ℝ | y^2 / 4 - x^2 / 16 = 1 } :=
by 
  sorry

end ellipse_equation_hyperbola_equation_l714_714314


namespace right_triangle_bisector_intersection_l714_714483

open EuclideanGeometry

noncomputable def midpoint (A B : Point) : Point := line_segment A B.midpoint

theorem right_triangle_bisector_intersection
  (A B C M D E : Point) 
  (h_triangle : right_triangle A B C)
  (h_angle_BAC : ∠BAC = 90)
  (h_angle_ABC : ∠ABC = 54)
  (h_midpoint : M = midpoint B C)
  (h_angle_bisector : is_angle_bisector C A D)
  (h_intersection : line_AM ∩ line_CD = {E}) :
  distance A B = distance C E := sorry

end right_triangle_bisector_intersection_l714_714483


namespace turn_on_camera_in_four_attempts_l714_714277

theorem turn_on_camera_in_four_attempts (B : ℕ → Prop) (charged : ℕ → Prop)
  (h1 : ∃ B1 B2 B3 B4 B5, B1 ≠ B2 ∧ B1 ≠ B3 ∧ B1 ≠ B4 ∧ B1 ≠ B5 ∧ B2 ≠ B3 ∧ B2 ≠ B4 ∧ B2 ≠ B5 ∧ B3 ≠ B4 ∧ B3 ≠ B5 ∧ B4 ≠ B5)
  (h2 : ∃ B1 B2 B3, charged B1 ∧ charged B2 ∧ charged B3)
  (camera_operates : ∀ B1 B2, charged B1 ∧ charged B2 → B 1 ∧ B 2) :
  ∃ (attempts : list (ℕ × ℕ)), attempts.length ≤ 4 ∧ ∃ B1 B2, B B1 ∧ B B2 :=
by
  sorry

end turn_on_camera_in_four_attempts_l714_714277


namespace smallest_m_divisible_15_l714_714912

noncomputable def largest_prime_2011_digits : ℕ := sorry -- placeholder for the largest prime with 2011 digits

theorem smallest_m_divisible_15 :
  let q := largest_prime_2011_digits
  in ∃ m : ℕ, (q^2 - m) % 15 = 0 ∧ m = 1 :=
begin
  let q := largest_prime_2011_digits,
  use 1,
  sorry
end

end smallest_m_divisible_15_l714_714912


namespace werewolf_cannot_reach_goal_l714_714342

def is_black (x y : ℤ) : Prop :=
  (x + y) % 2 = 0

def is_reachable (start_x start_y : ℤ) (goal_x goal_y : ℤ) (moves : list (ℤ × ℤ)) : Prop :=
  ∃ (steps : list (ℤ × ℤ)), 
  list.head steps = (start_x, start_y) ∧ 
  list.last steps = some (goal_x, goal_y) ∧ 
  ∀ move ∈ steps, move ∈ moves

theorem werewolf_cannot_reach_goal :
  ¬ is_reachable 26 10 42 2017 [ (3, 2), (2, 3), (-2, 3), (-3, 2), (-3, -2), (-2, -3), (2, -3), (3, -2) ] :=
by
  sorry

end werewolf_cannot_reach_goal_l714_714342


namespace a_dot_d_l714_714200

variables {a b c d : ℝ^3} -- Four distinct vectors in ℝ^3

-- Conditions
axiom a_unit : ∥a∥ = 1
axiom b_unit : ∥b∥ = 1
axiom c_unit : ∥c∥ = 1
axiom d_unit : ∥d∥ = 1
axiom a_neq_b : a ≠ b
axiom a_neq_c : a ≠ c
axiom a_neq_d : a ≠ d
axiom b_neq_c : b ≠ c
axiom b_neq_d : b ≠ d
axiom c_neq_d : c ≠ d
axiom a_dot_b : a ⬝ b = -1 / 7
axiom a_dot_c : a ⬝ c = -1 / 7
axiom b_dot_c : b ⬝ c = -1 / 7
axiom b_dot_d : b ⬝ d = -1 / 7
axiom c_dot_d : c ⬝ d = -1 / 7

-- Theorem statement
theorem a_dot_d : a ⬝ d = -19 / 21 := 
sorry

end a_dot_d_l714_714200


namespace rain_on_tuesday_l714_714865

/-- Let \( R_M \) be the event that a county received rain on Monday. -/
def RM : Prop := sorry

/-- Let \( R_T \) be the event that a county received rain on Tuesday. -/
def RT : Prop := sorry

/-- Let \( R_{MT} \) be the event that a county received rain on both Monday and Tuesday. -/
def RMT : Prop := RM ∧ RT

/-- The probability that a county received rain on Monday is 0.62. -/
def prob_RM : ℝ := 0.62

/-- The probability that a county received rain on both Monday and Tuesday is 0.44. -/
def prob_RMT : ℝ := 0.44

/-- The probability that no rain fell on either day is 0.28. -/
def prob_no_rain : ℝ := 0.28

/-- The probability that a county received rain on at least one of the days is 0.72. -/
def prob_at_least_one_day : ℝ := 1 - prob_no_rain

/-- The probability that a county received rain on Tuesday is 0.54. -/
theorem rain_on_tuesday : (prob_at_least_one_day = prob_RM + x - prob_RMT) → (x = 0.54) :=
by
  intros h
  sorry

end rain_on_tuesday_l714_714865


namespace breadthOfRectangularPart_l714_714338

variable (b l : ℝ)

def rectangularAreaProblem : Prop :=
  (l * b + (1 / 12) * b * l = 24 * b) ∧ (l - b = 10)

theorem breadthOfRectangularPart :
  rectangularAreaProblem b l → b = 12.15 :=
by
  intros
  sorry

end breadthOfRectangularPart_l714_714338


namespace no_such_function_exists_l714_714753

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x y : ℝ, f(x * y) = f(x) * f(y) + 2 * x * y :=
by
  sorry -- Proof is intentionally omitted

end no_such_function_exists_l714_714753


namespace interest_rate_proof_l714_714164

theorem interest_rate_proof : 
  ∃ (R : ℝ), (∀ (P1 P2 : ℝ) (r2 t1 t2 : ℝ), 
  P1 = 100 → P2 = 200 → r2 = 0.10 → t1 = 8 → t2 = 2 → 
  (P1 * R * t1 = P2 * r2 * t2) → 
  R = 0.05) :=
begin
  use 0.05,
  intros P1 P2 r2 t1 t2 hP1 hP2 hr2 ht1 ht2 h,
  rw [hP1, hP2, hr2, ht1, ht2] at h,
  linarith,
end

end interest_rate_proof_l714_714164


namespace cos_120_eq_neg_half_l714_714029

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714029


namespace chord_length_when_m_is_1_line_equation_for_shortest_chord_range_of_x_coordinate_of_point_P_l714_714503

-- Definition of the circle C and line l
def circle (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 11 = 0
def line (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that the chord length for m = 1 is 6√13 / 13
theorem chord_length_when_m_is_1 : 
  ∀ x y : ℝ, 
  circle x y → line x y 1 → 
  (∃ d : ℝ, d = 6 * real.sqrt 13 / 13) :=
sorry

-- Prove the equation of line l when the chord length is the shortest is x - y - 2 = 0
theorem line_equation_for_shortest_chord : 
  ∀ x y : ℝ, 
  ∃ m d : ℝ, 
  (circle x y → line x y m) → 
  (∃ l : ℝ, l = 2 * real.sqrt (5 - (x-4)^2 - (y-5)^2) → 
   (line x y (-2/3)) :=
sorry

-- Prove the range of the x-coordinate of point P is (0, 6)
theorem range_of_x_coordinate_of_point_P :
  ∀ x y : ℝ, 
  P x → line x (x - 2) (-2/3) → 
  (∃ x0 : ℝ, (x0 = 0 ∨ x0 = 6)  ∧ (0 < x0 → x0 < 6)) :=
sorry

end chord_length_when_m_is_1_line_equation_for_shortest_chord_range_of_x_coordinate_of_point_P_l714_714503


namespace count_isosceles_triangles_l714_714513

/-
We define the required geometric setup and translate the conditions into definitions 
and hypothesis, followed by proving the expected result.
-/

noncomputable def angle_XYZ : ℝ := 60
noncomputable def congruent {α : Type*} [has_dist α] (a b : α) : Prop := dist a b = 0

structure Triangle := (X Y Z : ℝ×ℝ)

structure GeomSetup := 
  (XYZ : Triangle)
  (M : ℝ×ℝ)
  (N : ℝ×ℝ)
  (P : ℝ×ℝ)
  (XMY_bisect : true) -- helper flag indicating XM bisects angle XYZ, detailed setup abstracted
  (MN_parallel_XY : true) -- helper flag indicating MN parallel to XY, detailed setup abstracted
  (NP_parallel_XM : true) -- helper flag indicating NP parallel to XM, detailed setup abstracted

axiom main_geom_setup : GeomSetup

noncomputable def num_isosceles_triangles (g : GeomSetup) : ℕ := 
  6 -- Directly inputting the correct answer based on the problem

theorem count_isosceles_triangles : num_isosceles_triangles main_geom_setup = 6 :=
by sorry

end count_isosceles_triangles_l714_714513


namespace special_fractions_distinct_integers_count_l714_714038

def is_special_fraction (a b : ℕ) : Prop := a + b = 20 ∧ a > 0 ∧ b > 0

def special_fractions : List ℚ :=
  List.filterMap (λ (p : ℕ × ℕ), if is_special_fraction p.1 p.2 then some (p.1 / (p.2 : ℚ)) else none)
    (List.product (List.range 20) (List.range 20))

def sum_of_three_special_fractions : List ℚ :=
  List.bind special_fractions (λ x, List.bind special_fractions (λ y, List.map (λ z, x + y + z) special_fractions))

def distinct_integers_from_special_fractions : Finset ℤ :=
  (List.filterMap (λ q, if q.den = 1 then some q.num else none) sum_of_three_special_fractions).toFinset

theorem special_fractions_distinct_integers_count : distinct_integers_from_special_fractions.card = 2 := 
  by
  sorry

end special_fractions_distinct_integers_count_l714_714038


namespace responses_needed_750_l714_714850

section Responses
  variable (q_min : ℕ) (response_rate : ℝ)

  def responses_needed : ℝ := response_rate * q_min

  theorem responses_needed_750 (h1 : q_min = 1250) (h2 : response_rate = 0.60) : responses_needed q_min response_rate = 750 :=
  by
    simp [responses_needed, h1, h2]
    sorry
end Responses

end responses_needed_750_l714_714850


namespace function_increasing_iff_l714_714821

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x

theorem function_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 - a) ↔ a ≤ 0 :=
by
  sorry

end function_increasing_iff_l714_714821


namespace cos_120_eq_neg_one_half_l714_714006

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714006


namespace cos_120_eq_neg_one_half_l714_714003

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714003


namespace geometry_problem_l714_714176

theorem geometry_problem
  (AB : ℝ)
  (angle_ADB : ∠)
  (sin_A : ℝ)
  (sin_C : ℝ)
  (BD : ℝ := (sin_A * AB))
  (BC : ℝ := BD / sin_C)
  (CD : ℝ := real.sqrt (BC^2 - BD^2)) :
  AB = 30 →
  angle_ADB = 90 →
  sin_A = 4 / 5 →
  sin_C = 1 / 4 →
  CD = 24 * real.sqrt 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  let BD := 24
  let BC := 96
  have : sin_A = BD / AB := by sorry  -- Placeholder for detailed sine logic, derived earlier
  have : sin_C = BD / BC := by sorry  -- Placeholder for sine logic 
  have : CD = real.sqrt (BC^2 - BD^2) := by sorry -- Placeholder for Pythagorean logic
  show CD = 24 * real.sqrt 15 := sorry -- Comprehensive final result

end geometry_problem_l714_714176


namespace cos_120_eq_neg_one_half_l714_714008

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714008


namespace domain_of_composed_function_l714_714852

theorem domain_of_composed_function :
  ∀ {f : ℝ → ℝ}, (∀ x, 1 ≤ x → x ≤ 2 → f x = f x) → (∀ x, 1 ≤ x → x ≤ 4 → f (Real.sqrt x) = f (Real.sqrt x)) :=
by
  intros f h1
  intros x hx1 hx2
  have h3 : 1 ≤ Real.sqrt x ∧ Real.sqrt x ≤ 2 :=
    by
      split
      { apply Real.le_sqrt hx1 }
      { apply Real.sqrt_le hx2 }
  exact h1 (Real.sqrt x) h3.1 h3.2
  sorry

end domain_of_composed_function_l714_714852


namespace probability_real_roots_l714_714078

open Finset big_operators

theorem probability_real_roots (S : Finset ℝ) (hS : S = {1, 2, 3, 4}) :
  (∃ (a b c ∈ S), b^2 - 4 * a * c ≥ 0) →
  let choices := S.powerset.filter (λ t, t.card = 3) in
  let feasible := choices.filter (λ t, match t.to_list with
                                          | [a, b, c] := b^2 - 4 * a * c ≥ 0
                                          | _ := false
                                        end) in
  (feasible.card : ℝ) / choices.card = 0.25 :=
by
  sorry

end probability_real_roots_l714_714078


namespace binomial_expansion_sum_abs_coeffs_eq_64_l714_714802

theorem binomial_expansion_sum_abs_coeffs_eq_64 (a : Fin 7 → ℝ) :
  (∀ x : ℝ, (1 - x)^6 = ∑ i in Finset.range 7, a i * x^i) →
  (Finset.range 7).sum (λ i, |a i|) = 64 :=
by sorry

end binomial_expansion_sum_abs_coeffs_eq_64_l714_714802


namespace no_solution_for_eq_l714_714985

theorem no_solution_for_eq (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) → False :=
sorry

end no_solution_for_eq_l714_714985


namespace value_on_interval_one_two_l714_714252

def f (x : ℝ) : ℝ :=
if h : x ∈ (set.Ioo 0 1) then 
  x + 1 
else 
  if h : x ∈ (set.Ioo 1 2) then 
    3 - x
  else 
    0 -- Placeholder for other cases, not needed for this problem

lemma even_function (x : ℝ) : f(-x) = f(x) :=
sorry

lemma periodic_function (x : ℝ) : f(x + 2) = f(x) :=
sorry

theorem value_on_interval_one_two (x : ℝ) (h : x ∈ (set.Ioo 1 2)) : f(x) = 3 - x :=
by 
  -- Prove using the given conditions
  sorry

end value_on_interval_one_two_l714_714252


namespace n_cubed_plus_5_div_by_6_l714_714959

theorem n_cubed_plus_5_div_by_6  (n : ℤ) : 6 ∣ n * (n^2 + 5) :=
sorry

end n_cubed_plus_5_div_by_6_l714_714959


namespace rahul_min_moves_guarantee_l714_714233

theorem rahul_min_moves_guarantee :
  ∃ m, m ≤ 4 ∧ ∀ (cards : list (fin 10)), (∀ i ∈ (finset.univ : finset (fin 10)), card_matching_strategy card_pair card_face_up m) → game_ends_before_or_at (cards: set (fin 10)) m := 
sorry

end rahul_min_moves_guarantee_l714_714233


namespace problem_1_problem_2_l714_714132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - a + 1) * x^(a + 1)
noncomputable def g (x : ℝ) : ℝ := x + real.sqrt (1 - 2 * x)

theorem problem_1 (h : ∀ x : ℝ, f 0 x = x) : 
  ∃ a : ℝ, (f a x = (a^2 - a + 1) * x^(a + 1)) ∧ 
           (∀ x : ℝ, f a (-x) = -f a x) → 
           a = 0 := 
sorry

theorem problem_2 : 
  ∃(range_of_g : set ℝ), 
    (∀ x : ℝ, x ∈ [0, (1 : ℝ) / 2] → g x ∈ range_of_g) ∧
    (range_of_g = set.Icc (1 / 2) 1) := 
sorry

end problem_1_problem_2_l714_714132


namespace count_balanced_numbers_l714_714332

-- Definition of a balanced number
def is_balanced (n : ℕ) : Prop :=
  let d1 := (n / 1000) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 = d3 + d4

-- Define the range of numbers to check
def range_0000_9999 : List ℕ := List.range' 0 10000

-- Statement of the problem
theorem count_balanced_numbers : (range_0000_9999.filter is_balanced).length = 670 := by
  sorry

end count_balanced_numbers_l714_714332


namespace cos_120_eq_neg_one_half_l714_714004

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714004


namespace christine_commission_rate_l714_714361

theorem christine_commission_rate (C : ℝ) (H1 : 24000 ≠ 0) (H2 : 0.4 * (C / 100 * 24000) = 1152) :
  C = 12 :=
by
  sorry

end christine_commission_rate_l714_714361


namespace geometric_progression_roots_l714_714517

theorem geometric_progression_roots :
  ∃ p : ℤ, (p = -8) ∧ ∃ x k : ℂ, x ≠ 0 ∧ k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1 ∧
  let r1 := x,
      r2 := k * x,
      r3 := k^2 * x in
    r1 + r2 + r3 = -7 ∧ 
    r1 * r2 + r2 * r3 + r3 * r1 = 14 ∧ 
    r1 * r2 * r3 = p :=
by
  sorry

end geometric_progression_roots_l714_714517


namespace line_solutions_l714_714792

-- Definition for points
def point := ℝ × ℝ

-- Conditions for lines and points
def line1 (p : point) : Prop := 3 * p.1 + 4 * p.2 = 2
def line2 (p : point) : Prop := 2 * p.1 + p.2 = -2
def line3 : Prop := ∃ p : point, line1 p ∧ line2 p

def lineL (p : point) : Prop := 2 * p.1 + p.2 = -2 -- Line l we need to prove
def perp_lineL : Prop := ∃ p : point, lineL p ∧ p.1 - 2 * p.2 = 1

-- Symmetry condition for the line
def symmetric_line (p : point) : Prop := 2 * p.1 + p.2 = 2 -- Symmetric line we need to prove

-- Main theorem to prove
theorem line_solutions :
  line3 →
  perp_lineL →
  (∀ p, lineL p ↔ 2 * p.1 + p.2 = -2) ∧
  (∀ p, symmetric_line p ↔ 2 * p.1 + p.2 = 2) :=
sorry

end line_solutions_l714_714792


namespace triangle_is_right_angled_l714_714268

variable {m : ℝ} (h : m > 0)

def a := m^2 - 1
def b := m^2 + 1
def c := 2 * m

theorem triangle_is_right_angled (h : m > 0) : (a m)^2 + (c m)^2 = (b m)^2 :=
by
  sorry

end triangle_is_right_angled_l714_714268


namespace geometric_location_eq_perpendicular_bisector_l714_714653

theorem geometric_location_eq_perpendicular_bisector (A B P : Point) :
  (dist P A = dist P B) ↔ (P ∈ perpendicular_bisector A B) :=
sorry

end geometric_location_eq_perpendicular_bisector_l714_714653


namespace part1_part2_l714_714416

variable (n : ℕ) (t : ℝ)

noncomputable def fibonacci (m : ℕ) : ℕ :=
  if m = 0 then 0
  else if m = 1 then 1
  else fibonacci (m - 1) + fibonacci (m - 2)

theorem part1 (a b : ℤ) (n_pos : n > 1) (t_ge_one : t ≥ 1) : 
  ∃! (j k : ℤ), j * (fibonacci (n+1), fibonacci n) + k * (fibonacci n, fibonacci (n-1)) = (a, b) := 
sorry

theorem part2 (L : ℤ) (M : ℝ) (n_pos : n > 1) (t_ge_one : t ≥ 1) (P_area : M = t^2 * fibonacci (2 * n + 1)) : 
  | real.sqrt (L : ℝ) - real.sqrt M | ≤ real.sqrt 2 := 
sorry

end part1_part2_l714_714416


namespace product_of_cubes_correctness_l714_714362

theorem product_of_cubes_correctness :
  (∏ n in (finset.range 6).map (λ k, k + 2), ((n^3 - 1) / (n^3 + 1))) = 19 / 56 :=
by
  sorry

end product_of_cubes_correctness_l714_714362


namespace prime_rect_no_valid_pairs_l714_714670

theorem prime_rect_no_valid_pairs (a b : ℕ) (h_prime_a : Nat.Prime a) (h_prime_b : Nat.Prime b) (h_b_gt_a : b > a) (h_unpainted_area : ab / 2 = ab - (a - 4) * (b - 4)) : 
  0 = 0 :=
begin
  sorry
end

end prime_rect_no_valid_pairs_l714_714670


namespace water_overflow_volume_is_zero_l714_714659

noncomputable def container_depth : ℝ := 30
noncomputable def container_outer_diameter : ℝ := 22
noncomputable def container_wall_thickness : ℝ := 1
noncomputable def water_height : ℝ := 27.5

noncomputable def iron_block_base_diameter : ℝ := 10
noncomputable def iron_block_height : ℝ := 30

theorem water_overflow_volume_is_zero :
  let inner_radius := (container_outer_diameter - 2 * container_wall_thickness) / 2,
      initial_water_volume := Real.pi * inner_radius^2 * water_height,
      max_container_volume := Real.pi * inner_radius^2 * container_depth,
      iron_block_radius := iron_block_base_diameter / 2,
      iron_block_volume := Real.pi * iron_block_radius^2 * iron_block_height,
      new_total_volume := max_container_volume - iron_block_volume in
  initial_water_volume = new_total_volume → 0 = 0 :=
by
  sorry

end water_overflow_volume_is_zero_l714_714659


namespace circle_area_l714_714286

theorem circle_area :
  let circle := {p : ℝ × ℝ | (p.fst - 8) ^ 2 + p.snd ^ 2 = 64}
  let line := {p : ℝ × ℝ | p.snd = 10 - p.fst}
  ∃ area : ℝ, 
    (area = 8 * Real.pi) ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle → p.snd ≥ 0 → p ∈ line → p.snd ≥ 10 - p.fst →
  sorry := sorry

end circle_area_l714_714286


namespace avg_X_Y_Z_l714_714650

variable {α : Type}

-- Define the sets X, Y, and Z as sets of people
variables (X Y Z : Set α)

-- Define conditions
variables (h1 : Disjoint X Y)
variables (h2 : Disjoint X Z)
variables (h3 : Disjoint Y Z)
variables (avg_X : ∀ x ∈ X, age x / card X = 37)
variables (avg_Y : ∀ y ∈ Y, age y / card Y = 23)
variables (avg_Z : ∀ z ∈ Z, age z / card Z = 41)
variables (avg_XY : ∀ x ∪ y ∈ X ∪ Y, age (x ∪ y) / card (X ∪ Y) = 29)
variables (avg_XZ : ∀ x ∪ z ∈ X ∪ Z, age (x ∪ z) / card (X ∪ Z) = 39.5)
variables (avg_YZ : ∀ y ∪ z ∈ Y ∪ Z, age (y ∪ z) / card (Y ∪ Z) = 33)

theorem avg_X_Y_Z : (age (X ∪ Y ∪ Z) / card (X ∪ Y ∪ Z)) = 34 :=
sorry

end avg_X_Y_Z_l714_714650


namespace selection_methods_count_l714_714780

theorem selection_methods_count :
  let students := {A, B, C, D, E}
  let languages := {Russian, Arabic, Hebrew}
  let students_unwilling := {A, B}
  let remaining_students := students \ students_unwilling
  ∃ M : students → languages, 
    M A ≠ Hebrew ∧ M B ≠ Hebrew ∧ M C ≠ Hebrew → card (students \ students_unwilling) = 3 ∧ 
    card (students) = 5 ∧ card (languages) = 3 ∧
    card (remaining_students) * (factorial (card students - 1) / factorial (card students - 3)) = 36 :=
by
  sorry

end selection_methods_count_l714_714780


namespace b_4_lt_b_7_l714_714686

def α : ℕ → ℕ := λ k, k

def b : ℕ → ℚ
| 1      := 1 + (1 / (α 1))
| n + 1  := 1 + (1 / (α 1 + b_aux n))

noncomputable def b_aux : ℕ → ℚ
| 1      := (1 / (α 1))
| n + 1  := (1 / (α 1 + b_aux n))

theorem b_4_lt_b_7 : b 4 < b 7 := by
  sorry

end b_4_lt_b_7_l714_714686


namespace solve_for_P5_l714_714639

noncomputable theory

variables (w x y z : ℝ)

def condition1 : Prop := w + x + y + z = 5
def condition2 : Prop := 2*w + 4*x + 8*y + 16*z = 7
def condition3 : Prop := 3*w + 9*x + 27*y + 81*z = 11
def condition4 : Prop := 4*w + 16*x + 64*y + 256*z = 1

theorem solve_for_P5 (h1 : condition1 w x y z) (h2 : condition2 w x y z) (h3 : condition3 w x y z) (h4 : condition4 w x y z) : 
  5*w + 25*x + 125*y + 625*z = -60 :=
  sorry

end solve_for_P5_l714_714639


namespace skew_lines_distance_AC_BD_l714_714872

theorem skew_lines_distance_AC_BD (A B C D : Type) [MetricSpace A]
  (AB BC CD DA AC BD : ℝ)
  (h_AB : AB = 1) (h_BC : BC = 5) (h_CD : CD = 7)
  (h_DA : DA = 5) (h_AC : AC = 5) (h_BD : BD = 2 * Real.sqrt 6) :
  distance (line_through_points A C) (line_through_points B D) = 3 * Real.sqrt 11 / 10 :=
  sorry

end skew_lines_distance_AC_BD_l714_714872


namespace solution_interval_l714_714806

open Real

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := log x / log 2 - 1 / (x * log 2)

theorem solution_interval (f_is_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
  (f_composition : ∀ x : ℝ, 0 < x → f (f x - log x / log 2) = 3) :
  ∃ a b : ℝ, 1 < a ∧ a < b ∧ b < 2 ∧ (∀ x : ℝ, 1 < x ∧ x < 2 → f x - f' x = 2) :=
sorry

end solution_interval_l714_714806


namespace integral_x_squared_l714_714376

theorem integral_x_squared : ∫ x in -1..1, x^2 = 2 / 3 :=
by
  sorry

end integral_x_squared_l714_714376


namespace cos_120_eq_neg_half_l714_714013

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714013


namespace ratio_G_F_ge_13_l714_714884

-- We first define the areas of the triangle ABC and the extended areas forming the hexagon.
variables (a b c : ℝ) -- Side lengths of triangle ABC

-- Define F as the area of triangle ABC
def area_triangle_ABC : ℝ := 
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define G as the area of the hexagon formed by extending sides
def area_hexagon (a b c : ℝ) : ℝ := 
  4 * area_triangle_ABC a b c + 
  (a * (a + b) * (a + c)) / (4 * Real.sqrt((a + b + c) * ((a + b + c) - a) * ((a + b + c) - b) * ((a + b + c) - c))) +
  (b * (a + b) * (b + c)) / (4 * Real.sqrt((a + b + c) * ((a + b + c) - a) * ((a + b + c) - b) * ((a + b + c) - c))) +
  (c * (b + c) * (c + a)) / (4 * Real.sqrt((a + b + c) * ((a + b + c) - a) * ((a + b + c) - b) * ((a + b + c) - c)))

-- The theorem we want to prove
theorem ratio_G_F_ge_13 (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  let F := area_triangle_ABC a b c in
  let G := area_hexagon a b c in
  G / F ≥ 13 :=
by
  sorry

end ratio_G_F_ge_13_l714_714884


namespace pre_image_of_f_l714_714812

theorem pre_image_of_f (x y : ℝ) (f : ℝ × ℝ → ℝ × ℝ) 
  (h : f = λ p => (2 * p.1 + p.2, p.1 - 2 * p.2)) :
  f (1, 0) = (2, 1) := by
  sorry

end pre_image_of_f_l714_714812


namespace integral_problem_solution_l714_714033

noncomputable def integral_problem_statement : ℝ :=
  ∫ x in -2..2, (sqrt (4 - x^2) + x^2)

theorem integral_problem_solution : integral_problem_statement = 2 * Real.pi + 16 / 3 := by
  sorry

end integral_problem_solution_l714_714033


namespace perfect_square_subsequence_l714_714282

theorem perfect_square_subsequence (n N : ℕ) (a : Fin n → ℕ) (b : Fin N → ℕ) (hN : N ≥ 2^n) :
  ∃ (i j : ℕ), i < j ∧ j ≤ N ∧ ∃ (p : ℕ), p * p = ∏ k in Finset.range (j - i), b (i + k) :=
sorry

end perfect_square_subsequence_l714_714282


namespace limit_of_T_l714_714439

-- Define the sequence a_n
def a (n : Nat) : ℝ := 1 - (2 ^ (2 ^ n)) / (2 ^ (2 ^ (n + 1)) + 1)

-- Define the product sequence T_n
def T (n : Nat) : ℝ := ∏ i in Finset.range (n + 1), a i

-- Theorem statement
theorem limit_of_T : limit at_top (fun n => T n) = 3 / 7 :=
by
  sorry

end limit_of_T_l714_714439


namespace equation_condition1_equation_condition2_equation_condition3_l714_714762

-- Definitions for conditions
def condition1 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ ∀ (x1 x2 : ℝ), (l x2 - l x1) / (x2 - x1) = -1/2)
def condition2 : Prop := ∃(l : ℝ → ℝ), (l 1 = 4 ∧ l 2 = 3)
def condition3 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ (∃ a : ℝ, l 0 = a ∧ l a = 0) ∨ (∃ a : ℝ, a > 0 ∧ l 0 = a = l a))

-- Proving equations given conditions
theorem equation_condition1 : condition1 → ∀ (x y : ℝ), x + 2 * y - 4 = 0 
:= sorry

theorem equation_condition2 : condition2 → ∀ (x y : ℝ), x + y - 5 = 0 
:= sorry

theorem equation_condition3 :
    condition3 → (∀ (x y : ℝ), (x - 2 * y = 0) ∨ (x + y - 3 = 0)) 
:= sorry

end equation_condition1_equation_condition2_equation_condition3_l714_714762


namespace alpha_eq_beta_not_sufficient_nor_necessary_l714_714835

theorem alpha_eq_beta_not_sufficient_nor_necessary (α β : ℝ) :
  (α = β → tan α = tan β) ∧ (tan α = tan β → α = β) ↔ false :=
sorry

end alpha_eq_beta_not_sufficient_nor_necessary_l714_714835


namespace smallest_x_for_multiple_l714_714633

theorem smallest_x_for_multiple (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 640 = 2^7 * 5^1) :
  (450 * x) % 640 = 0 ↔ x = 64 :=
sorry

end smallest_x_for_multiple_l714_714633


namespace perfect_square_a_i_l714_714214

theorem perfect_square_a_i (a : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n + 2) = 18 * a (n + 1) - a n) :
  ∀ i, ∃ k, 5 * (a i) ^ 2 - 1 = k ^ 2 :=
by
  -- The proof is missing the skipped definitions from the problem and solution context
  sorry

end perfect_square_a_i_l714_714214


namespace lisa_savings_l714_714220

-- Define the conditions
def originalPricePerNotebook : ℝ := 3
def numberOfNotebooks : ℕ := 8
def discountRate : ℝ := 0.30
def additionalDiscount : ℝ := 5

-- Define the total savings calculation
def calculateSavings (originalPricePerNotebook : ℝ) (numberOfNotebooks : ℕ) (discountRate : ℝ) (additionalDiscount : ℝ) : ℝ := 
  let totalPriceWithoutDiscount := originalPricePerNotebook * numberOfNotebooks
  let discountedPricePerNotebook := originalPricePerNotebook * (1 - discountRate)
  let totalPriceWith30PercentDiscount := discountedPricePerNotebook * numberOfNotebooks
  let totalPriceWithAllDiscounts := totalPriceWith30PercentDiscount - additionalDiscount
  totalPriceWithoutDiscount - totalPriceWithAllDiscounts

-- Theorem for the proof problem
theorem lisa_savings :
  calculateSavings originalPricePerNotebook numberOfNotebooks discountRate additionalDiscount = 12.20 :=
by
  -- Inserting the proof as sorry
  sorry

end lisa_savings_l714_714220


namespace coefficient_is_20_l714_714119

noncomputable def findCoefficient (a : ℝ) : ℕ :=
  if 2 * (∫ (x : ℝ) in 0..1, (√(a * x))) = 4 / 3 then
    let k := 19 in
    nat.choose 20 k
  else
    0

theorem coefficient_is_20 : a > 0 → (∫ (x : ℝ) in 0..1, (√(a * x))) = 2 / 3 → findCoefficient a = 20 :=
  by
    intros h_a h_area
    sorry

end coefficient_is_20_l714_714119


namespace point_M_projective_transformation_l714_714445

theorem point_M'_independence_of_line
  (a b : Line)
  (O M : Point)
  (h_parallel : a ∥ b)
  (h_not_on_O : ∀ l : Line, M ∉ l ∨ O ∉ l)
  (intersect_a : ∀ l : Line, M ∈ l → ∃ A : Point, A ∈ l ∧ A ∈ a)
  (intersect_b : ∀ l : Line, M ∈ l → ∃ B : Point, B ∈ l ∧ B ∈ b)
  (parallel_through_A : ∀ A B : Point, B ∈ b → A ∈ a → line_through A ∥ OB)
  (M_line : ∃ c : Line, c = line_through A ∧ c ∥ OB) :
  (∀ l₁ l₂ : Line, M ∈ l₁ ∧ M ∈ l₂ → point_of_intersection_of_lines OM (parallel_through_A A B) = point_of_intersection_of_lines OM (parallel_through_A A B)) := sorry

theorem projective_transformation
  (a b : Line)
  (O M M' : Point)
  (h_parallel : a ∥ b)
  (h_not_on_O : ∀ l : Line, M ∉ l ∨ O ∉ l)
  (intersect_a : ∀ l : Line, M ∈ l → ∃ A : Point, A ∈ l ∧ A ∈ a)
  (intersect_b : ∀ l : Line, M ∈ l → ∃ B : Point, B ∈ l ∧ B ∈ b)
  (parallel_through_A : ∀ A B : Point, B ∈ b → A ∈ a → line_through A ∥ OB)
  (M_line : ∃ c : Line, c = line_through A ∧ c ∥ OB)
  (P : Point → Point)
  (h_projective : ∀ M : Point, P(M) = M') :
  is_projective_transformation P := sorry

end point_M_projective_transformation_l714_714445


namespace prove_inequality_l714_714693

-- Define the sequence {b_n}
noncomputable def b_n (α : ℕ → ℕ) : ℕ → ℚ
| 1 := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b_n α n)

-- Example α values for simplification like α_k = 1
def example_α (k : ℕ) : ℕ := 1

-- The statement to be proved
theorem prove_inequality (α : ℕ → ℕ) (h : ∀ k, 0 < α k) : (b_n α 4 < b_n α 7) :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end prove_inequality_l714_714693


namespace intersection_M_N_l714_714141

noncomputable def M := {x : ℤ | x < 3}
noncomputable def N := {x : ℝ | 1 ≤ Real.exp x ∧ Real.exp x ≤ Real.exp 1}

theorem intersection_M_N : (M ∩ N) = {0, 1} :=
by
  sorry

end intersection_M_N_l714_714141


namespace ellipse_equation_max_area_triangle_ABC_l714_714104

open Real

-- Conditions
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (c a: ℝ) := c / a = (sqrt 2) / 2
def minor_axis_length (b : ℝ) := 2 * b = 2

-- Targets
theorem ellipse_equation (a b : ℝ) (x y : ℝ) :
  a > 0 ∧ b > 0 ∧ minor_axis_length b ∧ eccentricity (sqrt (a^2 - b^2)) a → 
  ellipse (a b) :=
sorry

theorem max_area_triangle_ABC (x y : ℝ) :
  a > 0 ∧ b > 0 ∧ minor_axis_length b ∧ eccentricity (sqrt (a^2 - b^2)) a → 
  ∃ A B C : ℝ × ℝ, is_on_ellipse A (1 / sqrt 2) ∧ area_ABC A B C = sqrt 2 :=
sorry

end ellipse_equation_max_area_triangle_ABC_l714_714104


namespace ratio_of_areas_l714_714663

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s^2

theorem ratio_of_areas :
  let large_side := 18
  let small_side := 6
  let large_area := area_equilateral_triangle large_side
  let small_area := area_equilateral_triangle small_side
  let pentagonal_area := large_area - small_area
  (small_area / pentagonal_area) = (1 / 8) :=
by
  sorry

end ratio_of_areas_l714_714663


namespace quadratic_inequality_solution_l714_714382

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l714_714382


namespace range_f_l714_714516

noncomputable def f (x : ℝ) : ℝ := x^3 + (1 / x^3)

theorem range_f (x : ℝ) (h1 : x > 0) (h2 : x + (1 / x) ≤ 4) : 
  set.range (λ x, f x) = set.Icc 2 52 ∪ set.Iic (-2) := 
sorry

end range_f_l714_714516


namespace cos_120_eq_neg_half_l714_714019

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714019


namespace probability_of_picking_letter_in_mathematics_l714_714158

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_picking_letter_in_mathematics :
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l714_714158


namespace proof_problem_l714_714429

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

axiom A1 (ω : ℝ) (hΩ : ω ≥ 0) : ∀ x : ℝ, f ω (x - Real.pi / 4) = -f ω (x + Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 6)

theorem proof_problem : 
  (∀ x, g (Real.pi / 6 - x) + g (x + Real.pi / 6) = 0) ∧
  (∃ T > 0, ∀ x, g (x + T) = g x ∧ T = Real.pi) ∧
  (¬ ∀ x, g (x + 2 * Real.pi / 3) = -g (x + 2 * Real.pi / 3)) ∧
  (∀ x ∈ Icc (Real.pi / 6) (Real.pi / 3), g x < g (x + Real.pi / 3)) :=
sorry

end proof_problem_l714_714429


namespace rubble_initial_money_l714_714569

def initial_money (cost_notebook cost_pen : ℝ) (num_notebooks num_pens : ℕ) (money_left : ℝ) : ℝ :=
  (num_notebooks * cost_notebook + num_pens * cost_pen) + money_left

theorem rubble_initial_money :
  initial_money 4 1.5 2 2 4 = 15 :=
by
  sorry

end rubble_initial_money_l714_714569


namespace gaokao_probability_l714_714507

open Finset

noncomputable def choose_2_out_of_5 (s : Finset ℕ) : Finset (Sym2 ℕ) := 
  s.powerset.filter (λ x, x.card = 2).map (λ x, ⟨x.1, x.2⟩)

theorem gaokao_probability :
  let subjects : Finset ℕ := {1, 2, 3, 4, 5} -- where 1, 2, 3, 4, 5 represent Chemistry, Biology, IPE, History, Geography respectively
  let total_outcomes := 10  -- since choosing 2 out of 5 elements
  let history_geography := {4, 5}
  let favorable_outcomes := (choose_2_out_of_5 subjects).filter (λ x, x.1 ∈ history_geography ∨ x.2 ∈ history_geography)
  in
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 10 := sorry

end gaokao_probability_l714_714507


namespace area_of_medians_l714_714877

-- Define the vertices and medians of the triangle
variables (A B C P D E F : Type) [Point A] [Point B] [Point C] [Point P] [Point D] [Point E] [Point F]
variables (triangle_ABC : Triangle A B C)
variables (median_AD : Median A D triangle_ABC)
variables (median_BE : Median B E triangle_ABC)
variables (median_CF : Median C F triangle_ABC)

-- Define the concept of area of triangles
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the triangles PAD, PBE, PCF
def triangle_PAD : Triangle P A D := sorry
def triangle_PBE : Triangle P B E := sorry
def triangle_PCF : Triangle P C F := sorry

-- State the theorem
theorem area_of_medians (h1 : Median A D triangle_ABC) (h2 : Median B E triangle_ABC) (h3 : Median C F triangle_ABC) :
  (area (triangle_PAD P A D)) = (area (triangle_PBE P B E)) + (area (triangle_PAF P A F)) := sorry

end area_of_medians_l714_714877


namespace frankie_pets_total_l714_714778

noncomputable def total_pets (c : ℕ) : ℕ :=
  let dogs := 2
  let cats := c
  let snakes := c + 5
  let parrots := c - 1
  dogs + cats + snakes + parrots

theorem frankie_pets_total (c : ℕ) (hc : 2 + 4 + (c + 1) + (c - 1) = 19) : total_pets c = 19 := by
  sorry

end frankie_pets_total_l714_714778


namespace smallest_prime_and_largest_three_divisors_sum_l714_714281

theorem smallest_prime_and_largest_three_divisors_sum :
  let m := (2 : ℕ)
      n := (7 ^ 2 : ℕ)
  in m + n = 51 :=
by
  let m := 2
  let n := 49
  show m + n = 51
  sorry

end smallest_prime_and_largest_three_divisors_sum_l714_714281


namespace angle_between_vectors_is_90_degrees_l714_714145

open Real

variables (a b : ℝ)

def vector_a : ℝ × ℝ := (sin (15 * pi / 180), cos (15 * pi / 180))
def vector_b : ℝ × ℝ := (cos (15 * pi / 180), sin (15 * pi / 180))

def vector_add : ℝ × ℝ := (sin (15 * pi / 180) + cos (15 * pi / 180), sin (15 * pi / 180) + cos (15 * pi / 180))
def vector_sub : ℝ × ℝ := (sin (15 * pi / 180) - cos (15 * pi / 180), cos (15 * pi / 180) - sin (15 * pi / 180))

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors_is_90_degrees :
  dot_product vector_add vector_sub = 0 :=
sorry

end angle_between_vectors_is_90_degrees_l714_714145


namespace total_earnings_dave_l714_714721

theorem total_earnings_dave : 
  let wMon := 6
      hMon := 6
      wTue := 7
      hTue := 2
      wWed := 9
      hWed := 3
      wThu := 8
      hThu := 5
  in wMon * hMon + wTue * hTue + wWed * hWed + wThu * hThu = 117 := by
  sorry

end total_earnings_dave_l714_714721


namespace julia_total_kids_l714_714193

def kidsMonday : ℕ := 7
def kidsTuesday : ℕ := 13
def kidsThursday : ℕ := 18
def kidsWednesdayCards : ℕ := 20
def kidsWednesdayHideAndSeek : ℕ := 11
def kidsWednesdayPuzzle : ℕ := 9
def kidsFridayBoardGame : ℕ := 15
def kidsFridayDrawingCompetition : ℕ := 12

theorem julia_total_kids : 
  kidsMonday + kidsTuesday + kidsThursday + kidsWednesdayCards + kidsWednesdayHideAndSeek + kidsWednesdayPuzzle + kidsFridayBoardGame + kidsFridayDrawingCompetition = 105 :=
by
  sorry

end julia_total_kids_l714_714193


namespace fraction_equivalence_l714_714644

theorem fraction_equivalence (a b c : ℝ) (h : (c - a) / (c - b) = 1) : 
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by
  sorry

end fraction_equivalence_l714_714644


namespace max_total_weight_l714_714245

-- Definitions
def A_max_weight := 5
def E_max_weight := 2 * A_max_weight
def total_swallows := 90
def A_to_E_ratio := 2

-- Main theorem statement
theorem max_total_weight :
  ∃ A E, (A = A_to_E_ratio * E) ∧ (A + E = total_swallows) ∧ ((A * A_max_weight + E * E_max_weight) = 600) :=
  sorry

end max_total_weight_l714_714245


namespace decahedral_dice_sum_17_probability_l714_714242

noncomputable def probability_of_sum_17_with_decahedral_dice : ℚ :=
let total_outcomes := 10 * 10 in
let successful_outcomes := 4 in
successful_outcomes / total_outcomes

theorem decahedral_dice_sum_17_probability :
  probability_of_sum_17_with_decahedral_dice = 1 / 25 :=
by sorry

end decahedral_dice_sum_17_probability_l714_714242


namespace measure_of_angle_A_l714_714098

noncomputable def measureAngleA (A B C O O₁ : Point) (excircle : Circle O) (circumcircle : Circle ABC) : Prop :=
  ∃ (reflection : ∀ {O O₁ BC : Point}, reflection_of O O₁ BC)
    (tangent : ∀ {A B C O : Point}, ExcircleTangent O A B C),
    O₁.liesOn (circumcircle) ∧ 
    measure_angle A = 60

-- Main theorem to be proved
theorem measure_of_angle_A (A B C O O₁ : Point)
    (h_excircle : ∃ (excircle : Circle O), excircle.tangent_to BC ∧ excircle.tangent_at_extension AB ∧ excircle.tangent_at_extension AC)
    (h_reflection : ∀ {O O₁ BC : Point}, reflection_of O O₁ BC)
    (h_circumcircle : ∃ (circumcircle : Circle ABC), O₁.liesOn circumcircle) :
  measure_angle A = 60 :=
by sorry

end measure_of_angle_A_l714_714098


namespace solve_for_x_l714_714574

theorem solve_for_x :
  ∀ x : ℚ, 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) → x = 22 / 5 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l714_714574


namespace speed_of_train_A_l714_714626

noncomputable def train_speed_A (V_B : ℝ) (T_A T_B : ℝ) : ℝ :=
  (T_B / T_A) * V_B

theorem speed_of_train_A : train_speed_A 165 9 4 = 73.33 :=
by
  sorry

end speed_of_train_A_l714_714626


namespace find_f2_plus_g2_l714_714118

variable (f g : ℝ → ℝ)

def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem find_f2_plus_g2 (hf : even_function f) (hg : odd_function g) (h : ∀ x, f x - g x = x^3 - 2 * x^2) :
  f 2 + g 2 = -16 :=
sorry

end find_f2_plus_g2_l714_714118


namespace sales_volume_relation_maximize_profit_l714_714552

-- Define the conditions as given in the problem
def cost_price : ℝ := 6
def sales_data : List (ℝ × ℝ) := [(10, 4000), (11, 3900), (12, 3800)]
def price_range (x : ℝ) : Prop := 6 ≤ x ∧ x ≤ 32

-- Define the functional relationship y in terms of x
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

-- Define the profit function w in terms of x
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - cost_price)

-- Prove that the functional relationship holds within the price range
theorem sales_volume_relation (x : ℝ) (h : price_range x) :
  ∀ (y : ℝ), (x, y) ∈ sales_data → y = sales_volume x := by
  sorry

-- Prove that the profit is maximized when x = 28 and the profit is 48400 yuan
theorem maximize_profit :
  ∃ x, price_range x ∧ x = 28 ∧ profit x = 48400 := by
  sorry

end sales_volume_relation_maximize_profit_l714_714552


namespace count_valid_six_digit_numbers_l714_714369

open Finset

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits: Finset ℕ := {1, 3, 5}
def even_digits: Finset ℕ := {0, 2, 4}

def permute_even_odd (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 0
  | _ => 6 * 6 + 2 * 6 * 2 -- This handles the cases of alternating digits

theorem count_valid_six_digit_numbers : permute_even_odd 6 = 60 := by
  sorry

end count_valid_six_digit_numbers_l714_714369


namespace smallest_possible_value_l714_714937

def smallest_ab (a b : ℕ) : Prop :=
  a^a % b^b =  0 ∧ a % b ≠ 0 ∧ Nat.gcd b 210 = 1

theorem smallest_possible_value : ∃ (a b : ℕ), smallest_ab a b ∧ a + b = 374 :=
by {
  existsi 253,
  existsi 121,
  unfold smallest_ab,
  simp,
  split,
  { sorry }, -- Proof that 253^253 % 121^121 = 0
  split,
  { exact dec_trivial }, -- Proof that 253 % 121 ≠ 0
  { exact dec_trivial }, -- Proof that Nat.gcd 121 210 = 1
  { refl },
}

end smallest_possible_value_l714_714937


namespace right_triangle_bc_is_3_l714_714871

-- Define the setup: a right triangle with given side lengths
structure RightTriangle :=
  (AB AC BC : ℝ)
  (right_angle : AB^2 = AC^2 + BC^2)
  (AB_val : AB = 5)
  (AC_val : AC = 4)

-- The goal is to prove that BC = 3 given the conditions
theorem right_triangle_bc_is_3 (T : RightTriangle) : T.BC = 3 :=
  sorry

end right_triangle_bc_is_3_l714_714871


namespace greatest_mean_weight_combined_EF_is_65_l714_714649

noncomputable def mean_weight_combined_EF (Dn En : ℕ) (m n : ℕ) : ℚ :=
  let total_weight_D := 30 * Dn
  let total_weight_E := 60 * En
  let total_weight_F := m
  (total_weight_E + total_weight_F).toRat / (En + n)

theorem greatest_mean_weight_combined_EF_is_65 (Dn En n : ℕ) (m : ℕ) : 
  (Dn = En) → 
  (30 * Dn + 60 * En = 45 * (Dn + En)) →
  (30 * Dn + m = 50 * (Dn + n)) →
  (∃ k, mean_weight_combined_EF Dn En m n = 50 + 15) :=
by
  unfold mean_weight_combined_EF
  intros hDnEn hDE hDF
  use 65
  sorry -- The proof is omitted.

end greatest_mean_weight_combined_EF_is_65_l714_714649


namespace tangent_difference_l714_714123

theorem tangent_difference (x y θ : ℝ) (h₁ : x = 1) (h₂ : y = -2) (h₃ : θ = real.arctan(y/x)) :
  real.tan (π/4 - θ) = -3 := 
sorry

end tangent_difference_l714_714123


namespace general_formula_a_sum_first_n_terms_b_l714_714139

-- Definition and conditions for sequence a_n
def seq_a (n : ℕ) : ℕ := n * 2^(n + 1)
def sum_seq_a (n : ℕ) : ℕ := ∑ k in finset.range(n + 1), (seq_a k / 2^(k + 1))

-- Prove the general formula for a_n
theorem general_formula_a {n : ℕ} (h : sum_seq_a n = n^2 + n) :
  seq_a n = n * 2^(n + 1) :=
sorry

-- Definitions for sequence b_n
def seq_b (n : ℕ) : ℤ := int.log 2 (seq_a n)

-- Sum of first n terms of sequence b_n
def sum_seq_b (n : ℕ) : ℤ := ∑ k in finset.range(n + 1), seq_b k

-- Prove the sum of the first n terms of sequence b_n
theorem sum_first_n_terms_b {n : ℕ} (h : ∀ k, seq_b k = (k + 1 : ℕ) + int.log 2 k) :
  sum_seq_b n = (n * (n + 3) / 2 + int.log 2 (fact n) : ℤ) :=
sorry

end general_formula_a_sum_first_n_terms_b_l714_714139


namespace eval_M_N_l714_714528

noncomputable def M_N_evaluation : Nat :=
  let perm := [(1 : Nat), 2, 3, 4, 6]
  let cyclic_sum (l : List Nat) : Nat :=
    match l with
    | [x1, x2, x3, x4, x5] => x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1
    | _                    => 0
  let perms := (List.permutations perm).filter (λ l, l.length = 5)
  let sums := perms.map cyclic_sum
  let M := Nat.maximum sums.getD 0
  let N := perms.count (λ l, cyclic_sum l = M)
  M + N

theorem eval_M_N : M_N_evaluation = 67 := by
  sorry

end eval_M_N_l714_714528


namespace sequence_formula_range_of_a_l714_714091

/-- The sequence {a_n} defined by a₁³ + a₂³ + ... + aₙ³ = (a₁ + a₂ + ... + aₙ)² 
    and with aₙ > 0 for all n ∈ ℕ*, has the general formula aₙ = n. -/
theorem sequence_formula (a : ℕ → ℕ) (hinit : ∀ n ∈ ℕ*, (list.sum (list.map (λ i, a i ^ 3) (list.range n))) = (list.sum (list.map a (list.range n))) ^ 2) (hpos : ∀ n ∈ ℕ*, 0 < a n) :
  ∀ n : ℕ, a n = n :=
  sorry

/-- Given the sequence {aₙ} where aₙ = n, the sum of the first n terms of 
    {1 / (aₙ * a_{ₙ+2}} is greater than 1/3 logₐ(1-ₐ) for all positive n if and only if 0 < a < 1/2. -/
theorem range_of_a (a : ℝ) (S : ℕ → ℝ) (hn_pos : ∀ n : ℕ, 0 < n → S n > 1 / 3 * real.log (1 - a)) :
  0 < a ∧ a < 1 / 2 :=
  sorry

end sequence_formula_range_of_a_l714_714091


namespace mario_time_on_moving_sidewalk_l714_714331

theorem mario_time_on_moving_sidewalk (d w v : ℝ) (h_walk : d = 90 * w) (h_sidewalk : d = 45 * v) : 
  d / (w + v) = 30 :=
by
  sorry

end mario_time_on_moving_sidewalk_l714_714331


namespace find_BC_l714_714881

theorem find_BC (A B C D E : Type)
  [geometry.Line A D] [geometry.RightAngle ∠A] [geometry.RightAngle ∠D]
  [geometry.Perpendicular DE AC E]
  (h1 : ∠ACD = 30) (h2 : ∠EBC = 30) (h3 : AD = sqrt 3) :
  BC = 3 :=
sorry

end find_BC_l714_714881


namespace largest_five_digit_palindromic_number_l714_714089

theorem largest_five_digit_palindromic_number (a b c d e : ℕ)
  (h1 : ∃ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
                 ∃ d e, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
                 (10001 * a + 1010 * b + 100 * c = 45 * (1001 * d + 110 * e))) :
  10001 * 5 + 1010 * 9 + 100 * 8 = 59895 :=
by
  sorry

end largest_five_digit_palindromic_number_l714_714089


namespace johns_original_salary_l714_714521

theorem johns_original_salary (new_salary : ℝ) (percentage_increase : ℝ) (original_salary : ℝ) 
    (h1 : new_salary = 100) (h2 : percentage_increase = 66.67 / 100) 
    (h3 : new_salary = original_salary * (1 + percentage_increase)) : 
    original_salary ≈ 60 := 
by 
  sorry

end johns_original_salary_l714_714521


namespace hexagon_chord_proof_l714_714662

noncomputable def hexagon_chord_length : ℝ :=
  let side_length1 : ℝ := 4
  let side_length2 : ℝ := 6
  let angle_inclination : ℝ := 80
  4 * real.sqrt 3

/- 
  Given a hexagon inscribed in a circle with three consecutive sides of length 4 
  and three consecutive sides of length 6, and a chord of the circle dividing the 
  hexagon into two polygons (one with sides of lengths 4, 4, 4, and the other 
  with sides of lengths 6, 6, 6) inclined at an angle of 80 degrees to one side 
  of the hexagon, prove the length of the chord is 4√3.
-/
theorem hexagon_chord_proof :
  ∃ (hexagon_chord : ℝ), 
    (hexagon_chord = 4 * real.sqrt 3) :=
by
  use hexagon_chord_length
  sorry

end hexagon_chord_proof_l714_714662


namespace prob_female_l714_714489

/-- Define basic probabilities for names and their gender associations -/
variables (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions for the probabilities
axiom h1 : P_Alexander = 3 * P_Alexandra
axiom h2 : P_Yevgeny = 3 * P_Evgenia
axiom h3 : P_Valentin = 1.5 * P_Valentina
axiom h4 : P_Vasily = 49 * P_Vasilisa

/-- The problem we need to prove: the probability that the lot was won by a female student is approximately 0.355 -/
theorem prob_female : 
  let P_female := (P_Alexandra * 1 / 4) + (P_Evgenia * 1 / 4) + (P_Valentina * 1 / 4) + (P_Vasilisa * 1 / 4) in
  abs (P_female - 0.355) < 0.001 :=
sorry

end prob_female_l714_714489


namespace convert_to_polar_l714_714720

open Real

theorem convert_to_polar (x y : ℝ) (r θ : ℝ) (h : r > 0) (hθ: 0 ≤ θ ∧ θ < 2 * π) :
  (x, y) = (3, -3) →
  r = Real.sqrt (x^2 + y^2) →
  θ = if y < 0 then 2 * π - Real.arctan (x / -y) else Real.arctan (x / y) →
  (r, θ) = (3 * sqrt 2, 7 * π / 4) :=
by
  intros hxy hr hθ_calculated
  rw [hxy] at hr hθ_calculated
  rw [Real.sqrt_eq_rpow, Real.add_rpow, Real.rpow_nat_cast, Real.rpow_nat_cast] at hr
  rw [Real.sqrt_eq_rpow, Real.add_rpow, Real.rpow_nat_cast, Real.rpow_nat_cast] at hθ_calculated
  
  -- Since we are asked only for the theorem statement, we now skip the actual proof
  sorry

end convert_to_polar_l714_714720


namespace unique_quadratic_polynomial_with_conditions_l714_714842

-- Define the conditions
def quadratic_polynomial (a b c : ℝ) : ℝ → ℝ := λ x => a * x ^ 2 + b * x + c
def is_root (a b c r : ℝ) := quadratic_polynomial a b c r = 0

-- State the main theorem to be proved
theorem unique_quadratic_polynomial_with_conditions :
  ∃! (a b c : ℝ) (r s: ℝ), 
    a ≠ 0 ∧ 
    {a, b, c} = {r, s} ∧ 
    a + b + c = 1 ∧ 
    is_root a b c r ∧ 
    is_root a b c s ∧ 
    quadratic_polynomial a b c 1 = 1 := 
sorry

end unique_quadratic_polynomial_with_conditions_l714_714842


namespace Chandler_sold_rolls_to_grandmother_l714_714774

theorem Chandler_sold_rolls_to_grandmother :
  ∀ (total_needed sold_to_uncle sold_to_neighbor needed_more : ℕ),
    total_needed = 12 →
    sold_to_uncle = 4 →
    sold_to_neighbor = 3 →
    needed_more = 2 →
    (total_needed - needed_more) - (sold_to_uncle + sold_to_neighbor) = 3 :=
by
  intros total_needed sold_to_uncle sold_to_neighbor needed_more ht hu hn hm
  rw [ht, hu, hn, hm]
  norm_num
  sorry

end Chandler_sold_rolls_to_grandmother_l714_714774


namespace fiftieth_number_is_31254_l714_714596

open List

-- Define the list of digits
def digits := [1, 2, 3, 4, 5]

-- All permutations of digits
def permutations := List.perms digits

-- Function to convert a list of digits to a number
def listToNumber (l : List Nat) : Nat :=
  l.foldl (λ acc d => 10 * acc + d) 0

-- Define the ordered list of numbers
def orderedNumbers := permutations.map listToNumber |>.sort (≤)

-- The 50th integer in the list
noncomputable def fiftiethNumber := orderedNumbers.get (50 - 1) (by sorry)

-- Prove that the 50th integer is 31254
theorem fiftieth_number_is_31254 : fiftiethNumber = 31254 :=
  by
  sorry

end fiftieth_number_is_31254_l714_714596


namespace solution_set_of_inequality_l714_714121

variable {R : Type*} [LinearOrder R] [OrderedAddCommGroup R]

def odd_function (f : R → R) := ∀ x, f (-x) = -f x

def monotonic_increasing_on (f : R → R) (s : Set R) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : odd_function f)
  (h_mono_inc : monotonic_increasing_on f (Set.Ioi 0))
  (h_f_neg1 : f (-1) = 2) : 
  {x : ℝ | 0 < x ∧ f (x-1) + 2 ≤ 0 } = Set.Ioc 1 2 :=
by
  sorry

end solution_set_of_inequality_l714_714121


namespace positional_relationship_l714_714107

universe u

variables {α : Type u} [ordered_ring α] 

-- Defining lines and planes
variables Line Plane : Type u
variable (a b : Line)
variable (α : Plane)

-- Relations between lines and planes
variable parallel (l : Line) (p : Plane) : Prop
variable subset (l : Line) (p : Plane) : Prop

-- Positional relationships between lines
inductive PosRel
| Parallel
| Skew

-- Conditions
variables (h₁ : parallel a α) (h₂ : subset b α)

-- Problem Statement
theorem positional_relationship :
  PosRel a b = PosRel.Parallel ∨ PosRel a b = PosRel.Skew :=
sorry

end positional_relationship_l714_714107


namespace comparison_l714_714696

def sequence (α : ℕ → ℕ) : ℕ → ℚ
| 1     := 1 + 1 / (α 1)
| (n+1) := 1 + 1 / (α 1 + sequence (λ k, α (k+1) n))

theorem comparison (α : ℕ → ℕ) (h : ∀ k, 1 ≤ α k) :
  sequence α 4 < sequence α 7 := 
sorry

end comparison_l714_714696


namespace triangle_eval_l714_714117

def triangle (a b : ℝ) : ℝ := (a * b) / (-6)

theorem triangle_eval : triangle 4 (triangle 3 2) = 2 / 3 := by
  sorry

end triangle_eval_l714_714117


namespace toast_three_slices_in_3_minutes_l714_714923

theorem toast_three_slices_in_3_minutes :
  ∀ (A B C : Type) (toaster : Type) (toasts : A → B → C)
  (toast_time : ℕ → ℕ → Prop),
  (∀ a b : A, toast_time 2 1 → toast_time 2 2 → toast_time 1 1 → toast_time 1 2) → 
  (toaster.slot = 2) → (time_per_side = 1) → ∃ (complete_time : ℕ), complete_time = 3 :=
begin
  sorry
end

end toast_three_slices_in_3_minutes_l714_714923


namespace number_of_bases_final_digit_2_l714_714398

theorem number_of_bases_final_digit_2 : 
  (finset.filter (λ b : ℕ, b ≥ 2 ∧ b ≤ 20 ∧ 625 % b = 2) (finset.range 21)).card = 1 :=
by
  sorry

end number_of_bases_final_digit_2_l714_714398


namespace turtles_remaining_l714_714703

/-- 
In one nest, there are x baby sea turtles, while in the other nest, there are 2x baby sea turtles.
One-fourth of the turtles in the first nest and three-sevenths of the turtles in the second nest
got swept to the sea. Prove the total number of turtles still on the sand is (53/28)x.
-/
theorem turtles_remaining (x : ℕ) (h1 : ℕ := x) (h2 : ℕ := 2 * x) : ((3/4) * x + (8/7) * (2 * x)) = (53/28) * x :=
by
  sorry

end turtles_remaining_l714_714703


namespace range_of_lambda_l714_714816

open Complex

theorem range_of_lambda (m θ λ : ℝ) (h1 : Complex.mk m (4 - m^2) = Complex.mk (2 * cos θ) (λ + 3 * sin θ)) :
  -9 / 16 ≤ λ ∧ λ ≤ 7 :=
sorry

end range_of_lambda_l714_714816


namespace compute_65_sq_minus_55_sq_l714_714716

theorem compute_65_sq_minus_55_sq : 65^2 - 55^2 = 1200 :=
by
  -- We'll skip the proof here for simplicity
  sorry

end compute_65_sq_minus_55_sq_l714_714716


namespace bows_problem_l714_714479

theorem bows_problem 
    (total_bows : ℕ)
    (red_fraction blue_fraction green_fraction : ℚ)
    (white_bows : ℕ)
    (one_fifth : red_fraction = 1/5)
    (one_half : blue_fraction = 1/2)
    (one_tenth : green_fraction = 1/10)
    (white_bows_eq : white_bows = 30)
    (total_bows_eq : total_bows = white_bows / (1 - (red_fraction + blue_fraction + green_fraction))) :
    (green_bows : ℕ) (green_bows = total_bows * green_fraction) → green_bows = 15 := 
by 
  sorry

end bows_problem_l714_714479


namespace descent_distance_correct_l714_714326

-- Define the conditions for ascent
def ascent_forest_distance := 8
def ascent_rocky_distance := 5
def ascent_snowy_distance := 3 / 2

-- Define the rates for the descent
def descent_grasslands_rate := 1.5 * ascent_forest_distance
def descent_sandy_rate := 1.5 * ascent_rocky_distance

-- Define the distances covered on descent days
def descent_grasslands_distance := descent_grasslands_rate
def descent_sandy_distance := descent_sandy_rate

-- Calculate the total distance of the route down the mountain
def total_descent_distance := descent_grasslands_distance + descent_sandy_distance

theorem descent_distance_correct : total_descent_distance = 19.5 := by
  sorry

end descent_distance_correct_l714_714326


namespace equalize_sugar_impossible_l714_714608

structure Jars :=
  (jar1 jar2 jar3 : ℕ) -- volumes of jars in ml
  (sugar2 sugar3 : ℕ) -- grams of sugar in jar2 and jar3

def initial_jars : Jars :=
  { jar1 := 0, jar2 := 700, jar3 := 800, sugar2 := 50, sugar3 := 60 }

def transfer (j : Jars) (source target : ℕ) : Jars :=
  if source = 1 then
    if j.jar1 < 100 then j else
    { jar1 := j.jar1 - 100, jar2 := if target = 2 then j.jar2 + 100 else j.jar2, jar3 := if target = 3 then j.jar3 + 100 else j.jar3, ..j }
  else if source = 2 then
    if j.jar2 < 100 then j else
    { jar1 := if target = 1 then j.jar1 + 100 else j.jar1, jar2 := j.jar2 - 100, jar3 := if target = 3 then j.jar3 + 100 else j.jar3, ..j }
  else if source = 3 then
    if j.jar3 < 100 then j else
    { jar1 := if target = 1 then j.jar1 + 100 else j.jar1, jar2 := if target = 2 then j.jar2 + 100 else j.jar2, jar3 := j.jar3 - 100, ..j }
  else
    j

theorem equalize_sugar_impossible : (∀ (j : Jars), ∃ n : ℕ, n ≥ 0 ∧ 
    ((iterate n (transfer j 2 1)).jar1 = 0 ∧ 
    (iterate n (transfer j 3 1)).jar1 = 0 ∧ 
    (iterate n (transfer j 2 3)).sugar2 = (iterate n (transfer j 3 2)).sugar3)) → False :=
begin
  sorry
end

end equalize_sugar_impossible_l714_714608


namespace sum_first_9_terms_l714_714092

-- Let's define the sequence and the conditions
variable (a : ℕ → ℝ)
variable (l : ℝ → ℝ)
variable (h_line : ∀ n : ℕ, n > 0 → n ∈ {n | l n = a n})
variable (h_passes : l 5 = 3)

-- Define the sum of the first n terms of the sequence
noncomputable def sum_n (n : ℕ) : ℝ := ∑ i in Finset.range n.succ, a i

-- The main theorem statement
theorem sum_first_9_terms : sum_n a 8 = 27 :=
by
  sorry

end sum_first_9_terms_l714_714092


namespace count_base8_including_5_or_6_l714_714841

/- 
  Prove that the number of first 512 positive integers written in base 8 which include the digit 5 or 6 is 387.
-/
theorem count_base8_including_5_or_6 : 
  ∃ n : ℕ, n = 512 → 
  let range := list.range n in 
  let base8_digits := ['0', '1', '2', '3', '4', '5', '6', '7'] in
  let valid_digits := ['5', '6'] in
  let count_including_5_or_6 := range.filter (λ x, 
    (x.to_string.to_list.any (λ d, d ∈ valid_digits))
  ).length in
  count_including_5_or_6 = 387 :=
begin
  sorry
end

end count_base8_including_5_or_6_l714_714841


namespace confidence_of_related_events_l714_714344

-- Define the relationship between K^2 and the event confidence
def related_events {A B : Prop} (K_squared : ℝ) (p_value : ℝ) [fact (0 < p_value ∧ p_value < 1)] : Prop :=
  K_squared > 3.841 → p_value = 0.05 → (1 - p_value) = 0.95

-- Assume that K^2 > 3.841
axiom K_squared_gt_3_841 {A B : Prop} (K_squared : ℝ) : fact (K_squared > 3.841)

-- Theorem statement
theorem confidence_of_related_events {A B : Prop} (K_squared : ℝ) (p_value : ℝ) [fact (K_squared > 3.841)] [fact (0 < p_value ∧ p_value < 1)] :
  K_squared > 3.841 → p_value = 0.05 → (1 - p_value) = 0.95 :=
by
  -- The proof is omitted
  sorry

end confidence_of_related_events_l714_714344


namespace BK_length_l714_714254

-- Definitions for the problem setup
variables (A B C C₁ A₁ B₁ K : Point)
variables (AB BC AC : ℝ)
variables [triangle : is_triangle A B C]

-- Conditions given in the problem
def AB_eq_BC_eq_5 : AB = 5 ∧ BC = 5 := by sorry
def AC_eq_6 : AC = 6 := by sorry
def incircle_touches : touches_incircle A B C A₁ B₁ C₁ := by sorry
def B1_tangent_point : tangent_point A C B₁ := by sorry
def B_intersects_incircle : intersects_incircle B B₁ K := by sorry

-- The goal statement to prove
theorem BK_length : BK = 1 := by
  have s := (AB + BC + AC) / 2
  have s_eq_8 : s = 8 := by sorry
  have AB1_AC1_eq_3 : AB1 = AC1 = s - AB := by sorry
  have BC1_BA1_eq_3 : BC1 = BA1 = s - BC := by sorry
  have CA1_CB1_eq_2 : CA1 = CB1 = s - AC := by sorry
  show BK = 1, from sorry

end BK_length_l714_714254


namespace dirocks_rectangular_fence_count_l714_714726

/-- Dirock's backyard problem -/
def grid_side : ℕ := 32

def rock_placement (i j : ℕ) : Prop := (i % 3 = 0) ∧ (j % 3 = 0)

noncomputable def dirocks_rectangular_fence_ways : ℕ :=
  sorry

theorem dirocks_rectangular_fence_count : dirocks_rectangular_fence_ways = 1920 :=
sorry

end dirocks_rectangular_fence_count_l714_714726


namespace range_of_a_l714_714801

theorem range_of_a {a : ℝ} :
  (∃ (A : Set ℝ), (∀ x, x ∈ A ↔ log a (a * x - 1) > 1) ∧ 2 ∈ A) →
  (1 / 2 < a ∧ a < 1) ∨ (1 < a) :=
by
  intro h
  obtain ⟨A, hA, h2⟩ := h
  have := hA 2
  sorry

end range_of_a_l714_714801


namespace otimes_calculation_l714_714364

def otimes (x y : ℝ) : ℝ := x^2 + y^2

theorem otimes_calculation (x : ℝ) : otimes x (otimes x x) = x^2 + 4 * x^4 :=
by
  sorry

end otimes_calculation_l714_714364


namespace water_left_in_bucket_l714_714359

theorem water_left_in_bucket :
  let initial_water := 3 / 4
  let poured_out := 1 / 3
  initial_water - poured_out = 5 / 12 :=
by
  sorry

end water_left_in_bucket_l714_714359


namespace compute_expression_l714_714035

theorem compute_expression : 7^2 - 2 * 6 + (3^2 - 1) = 45 :=
by
  sorry

end compute_expression_l714_714035


namespace smallest_number_among_greaters_or_equals_is_1_2_l714_714348

noncomputable def answer : ℚ := 
  let numbers := [1.4, 9/10, 1.2, 0.5, 13/10] in
  let decimals := numbers.map (λ x, (x : ℚ)) in
  let filtered := decimals.filter (λ x, x ≥ 1.1) in
  list.minimum filtered

theorem smallest_number_among_greaters_or_equals_is_1_2 :
  answer = 1.2 :=
by
  sorry

end smallest_number_among_greaters_or_equals_is_1_2_l714_714348


namespace midpoints_form_parallelogram_l714_714916

-- Define the type for points in a 3D space
structure Point3 := (x y z : ℝ)

-- Define vector operations on points
def midpoint (p1 p2 : Point3) : Point3 :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

def parallel (v1 v2 : Point3) : Prop :=
  ∃ k : ℝ, v1.x = k * v2.x ∧ v1.y = k * v2.y ∧ v1.z = k * v2.z

def is_parallelogram (A' B' C' D' : Point3) : Prop :=
  let A'B' := Point3.mk (B'.x - A'.x) (B'.y - A'.y) (B'.z - A'.z);
      C'D' := Point3.mk (D'.x - C'.x) (D'.y - C'.y) (D'.z - C'.z);
      A'D' := Point3.mk (D'.x - A'.x) (D'.y - A'.y) (D'.z - A'.z);
      B'C' := Point3.mk (C'.x - B'.x) (C'.y - B'.y) (C'.z - B'.z)
  in (parallel A'B' C'D' ∧ parallel A'D' B'C')

theorem midpoints_form_parallelogram (A B C D : Point3) (h: ∀ a b c d : Point3, ¬collinear a b c ∨ ¬collinear a b d ∨ ¬collinear a c d ∨ ¬collinear b c d):
  let A' := midpoint A B;
      B' := midpoint B C;
      C' := midpoint C D;
      D' := midpoint D A
  in is_parallelogram A' B' C' D' :=
sorry

end midpoints_form_parallelogram_l714_714916


namespace password_combinations_l714_714890

theorem password_combinations :
  let digits := [1, 2, 3, 4, 5, 6]
  let odd_digits := [1, 3, 5]
  let even_digits := [2, 4, 6]
  let password_length := 6
  ∑ n in finset.range password_length, if n % 2 = 0 then even_digits.length else odd_digits.length = 1458 :=
by
  sorry

end password_combinations_l714_714890


namespace correct_propositions_l714_714566

variable (f : ℝ → ℝ) (a b : ℝ)
variable [DifferentiableOn ℝ f (Set.Ioo a b)]

theorem correct_propositions :
  (∀ x, UniqueExtremum f x -> MaxOrMin f x) ∧
  (∀ x, IsExtremum f x ↔ (f' (x - ε) * f' (x + ε) < 0)) :=
sorry

end correct_propositions_l714_714566


namespace b_4_lt_b_7_l714_714688

def α : ℕ → ℕ := λ k, k

def b : ℕ → ℚ
| 1      := 1 + (1 / (α 1))
| n + 1  := 1 + (1 / (α 1 + b_aux n))

noncomputable def b_aux : ℕ → ℚ
| 1      := (1 / (α 1))
| n + 1  := (1 / (α 1 + b_aux n))

theorem b_4_lt_b_7 : b 4 < b 7 := by
  sorry

end b_4_lt_b_7_l714_714688


namespace trigonometric_identity_l714_714845

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h₁ : tan α + 1 / tan α = 10 / 3)
  (h₂ : π / 4 < α ∧ α < π / 2) :
  sin (2 * α + π / 4) + 2 * cos (π / 4) * sin α ^ 2 = 4 * sqrt 2 / 5 :=
by
  sorry

end trigonometric_identity_l714_714845


namespace pos_int_solns_to_eq_l714_714755

open Int

theorem pos_int_solns_to_eq (x y z : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x^2 + y^2 - z^2 = 9 - 2 * x * y ↔ 
    (x, y, z) = (5, 0, 4) ∨ (x, y, z) = (4, 1, 4) ∨ (x, y, z) = (3, 2, 4) ∨ 
    (x, y, z) = (2, 3, 4) ∨ (x, y, z) = (1, 4, 4) ∨ (x, y, z) = (0, 5, 4) ∨ 
    (x, y, z) = (3, 0, 0) ∨ (x, y, z) = (2, 1, 0) ∨ (x, y, z) = (1, 2, 0) ∨ 
    (x, y, z) = (0, 3, 0) :=
by sorry

end pos_int_solns_to_eq_l714_714755


namespace z_in_fourth_quadrant_l714_714157

noncomputable def z : ℂ := (1 - complex.I * real.sqrt 3) / (2 * complex.I)

theorem z_in_fourth_quadrant :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ z = complex.mk (↑x) (↑y) :=
sorry

end z_in_fourth_quadrant_l714_714157


namespace incorrect_statement_d_l714_714953

noncomputable def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem incorrect_statement_d (n : ℤ) :
  (n < cbrt 9 ∧ cbrt 9 < n+1) → n ≠ 3 :=
by
  intro h
  have h2 : (2 : ℤ) < cbrt 9 := sorry
  have h3 : cbrt 9 < (3 : ℤ) := sorry
  exact sorry

end incorrect_statement_d_l714_714953


namespace correct_system_of_equations_l714_714350

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : x - y = 5) (h2 : y - (1/2) * x = 5) : 
  (x - y = 5) ∧ (y - (1/2) * x = 5) :=
by { sorry }

end correct_system_of_equations_l714_714350


namespace express_one_million_with_nines_l714_714378

noncomputable def expr1 := 999999 + 9 / 9
noncomputable def expr2 := (999999 * 9 + 9) / 9
noncomputable def expr3 := (9 + 9 / 9) ^ (9 - Real.sqrt 9)

theorem express_one_million_with_nines :
    expr1 = 1000000 ∧
    expr2 = 1000000 ∧
    expr3 = 1000000 :=
by {
    sorry
}

end express_one_million_with_nines_l714_714378


namespace cyclic_quadrilateral_ABCD_l714_714947

-- Definitions
def is_square (A B C D : Point) : Prop := -- (Definition of square)
def midpoint (M A B : Point) : Prop := -- (M is midpoint of A and B)
def center (P : Point) (polygon : List Point) : Prop := -- (P is the center of given polygon)

-- Problem statement
theorem cyclic_quadrilateral_ABCD (A B C D M N P Q R S: Point)
  (h_sq : is_square A B C D)
  (h_M : midpoint M A B)
  (h_N : midpoint N B C)
  (h_P_center_hexagon : center P [A B C D M N]) -- Regular hexagon vertices
  (h_Q_center_square : center Q [A B C D])
  (h_R_center_dodecagon : center R [A B C D M N]) -- Regular 12-gon vertices
  (h_cyclic : is_cyclic_quadrilateral P Q R S) : Prop :=
  sorry

end cyclic_quadrilateral_ABCD_l714_714947


namespace volume_of_regular_triangular_pyramid_l714_714771

theorem volume_of_regular_triangular_pyramid
  (R γ : ℝ)
  (h₀ : R > 0)
  (h₁ : 0 < γ ∧ γ < π) :
  ∃ V : ℝ, V = \frac{2R^3 \sqrt{3} \cos^2 (\gamma / 2) (1 - 2 \cos γ)}{27 \sin^6 (\gamma / 2)} :=
sorry

end volume_of_regular_triangular_pyramid_l714_714771


namespace sequence_integer_count_l714_714267

theorem sequence_integer_count :
  let seq : ℕ → ℕ := λ n => 16200 / 5^n in
  let is_integer (n : ℕ) : Prop := ∃ k : ℕ, 16200 / 5^n = k in
  (∀ n : ℕ, 5^n ∣ 16200 → is_integer n) →
  16200 / 5^3 ∉ ℕ →
  {n : ℕ | is_integer n}.finite ∧ {n : ℕ | is_integer n}.to_finset.card = 3
:= by
  sorry

end sequence_integer_count_l714_714267


namespace ceil_neg_sqrt_frac_l714_714742

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l714_714742


namespace simplify_trig_l714_714971

theorem simplify_trig (x : ℝ) (h_cos_sin : cos x ≠ -1) (h_sin_ne_zero : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
    sorry

end simplify_trig_l714_714971


namespace chris_should_split_l714_714682

noncomputable section

-- Define the total number of swimming sessions
def total_sessions : Nat := 15

-- Define the number of times Adam, Bill, and Chris paid
def adam_paid : Nat := 8
def bill_paid : Nat := 7
def total_cost_for_chris : Int := 30

-- The theorem that needs to be proven
theorem chris_should_split 
  (total_sessions = 15) 
  (adam_paid = 8)
  (bill_paid = 7)
  (total_cost_for_chris = 30) : 
  (chris_should_pay_adam := 18) ∧ (chris_should_pay_bill := 12) := 
sorry

end chris_should_split_l714_714682


namespace problem_statement_l714_714634

theorem problem_statement : (3^1 - 2 + 6^2 - 1) ^ (-2) * 6 = 1 / 216 :=
by
  -- First, we define the conditions directly from the problem statement
  have h1 : 3 ^ 1 = 3 := by norm_num
  have h2 : 6 ^ 2 = 36 := by norm_num

  -- Now we use these conditions to simplify and prove the statement
  calc
    ((3 ^ 1 - 2 + 6 ^ 2 - 1) ^ (-2)) * 6
        = ((3 - 2 + 36 - 1) ^ (-2)) * 6 : by rw [h1, h2]
    ... = (36 ^ (-2)) * 6 : by norm_num
    ... = (1 / 36 ^ 2) * 6 : by rw [zpow_neg, one_div]
    ... = (1 / 1296) * 6 : by norm_num
    ... = 1 / 216 : by field_simp [mul_comm]

-- The proof body contains steps but ensures it concludes without 'sorry'
sorry -- This is to indicate the proof structure but avoid built-in proof check

end problem_statement_l714_714634


namespace area_of_right_triangle_l714_714618

variables {A B C D : Type} [metric_space B] [metric_space C] 

-- Conditions
variables (triangle_ABC : Triangle A B C) (is_right_angle : triangle_ABC.angle B = 90)
                  (D : Point) (D_on_AC : D ∈ triangle_ABC.AC) 
                  (AD : Real) (DC : Real)
                  (AD_eq : AD = 5) (DC_eq : DC = 6)

-- Statement of the problem
theorem area_of_right_triangle (h₁ : RightTriangle triangle_ABC B)  -- ABC is a right triangle with a right angle at B.
                               (h₂ : FootOfAltitude B triangle_ABC.AC D)  -- D is the foot of the altitude from B to AC.
                               (h₃ : SegmentLength triangle_ABC.A D = 5)  -- AD = 5
                               (h₄ : SegmentLength D C = 6) :  -- DC = 6
  Area triangle_ABC = (11 * Real.sqrt 30) / 2 := 
by
sorry

end area_of_right_triangle_l714_714618


namespace smallest_m_divisible_15_l714_714911

noncomputable def largest_prime_2011_digits : ℕ := sorry -- placeholder for the largest prime with 2011 digits

theorem smallest_m_divisible_15 :
  let q := largest_prime_2011_digits
  in ∃ m : ℕ, (q^2 - m) % 15 = 0 ∧ m = 1 :=
begin
  let q := largest_prime_2011_digits,
  use 1,
  sorry
end

end smallest_m_divisible_15_l714_714911


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l714_714733

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l714_714733


namespace two_pow_ge_n_cubed_l714_714077

theorem two_pow_ge_n_cubed (n : ℕ) : 2^n ≥ n^3 ↔ n ≥ 10 := 
by sorry

end two_pow_ge_n_cubed_l714_714077


namespace isosceles_trapezoid_height_l714_714066

theorem isosceles_trapezoid_height (S h : ℝ) (h_nonneg : 0 ≤ h) 
  (diag_perpendicular : S = (1 / 2) * h^2) : h = Real.sqrt S :=
by
  sorry

end isosceles_trapezoid_height_l714_714066


namespace prove_angle_PIQ_is_right_angle_l714_714580

noncomputable def problem_statement
  (A B C P Q I : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space Q] [metric_space I]
  (angle : triangle → ℝ)
  (angle_bisectors : ∀ (T : triangle), point)
  (on_extension : ∀ (T : triangle), point → point → Prop)
  (equal_length : ∀ (a b : point), ℝ → Prop) :
  Prop :=
  let TABC := triangle A B C in
  let I := angle_bisectors TABC in
  let P := on_extension TABC A B in
  let Q := on_extension TABC C B in
  (equal_length AP AC) ∧ (equal_length CQ AC) →
  (angle TABC = 120) →
  let angle_PIQ := angle_is 90 in
  angle_PIQ TABC P I Q

def triangle (A B C : Type) := Σ (A B C : Type), A × B × C

def point (P : Type) := Σ (P : Type), P

def metric_space (X : Type) := 
  ∀ a b : X, 𝓓 X -- distance between points a and b

def angle_is (θ : ℝ) := 
  λ (T : triangle) (P Q R : point T), angle T P Q R = θ

theorem prove_angle_PIQ_is_right_angle
  (A B C P Q I : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space Q] [metric_space I]
  (angle : triangle → ℝ)
  (angle_bisectors : ∀ (T : triangle), point)
  (on_extension : ∀ (T : triangle), point → point → Prop)
  (equal_length : ∀ (a b : point), ℝ → Prop) :
  problem_statement A B C P Q I angle angle_bisectors on_extension equal_length :=
by
  sorry

end prove_angle_PIQ_is_right_angle_l714_714580


namespace range_of_independent_variable_l714_714599

theorem range_of_independent_variable (x : ℝ) (h : ∃ y, y = 2 / (Real.sqrt (x - 3))) : x > 3 :=
sorry

end range_of_independent_variable_l714_714599


namespace find_number_l714_714459

theorem find_number (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := 
sorry

end find_number_l714_714459


namespace cos_120_eq_neg_half_l714_714015

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714015


namespace ceil_eval_l714_714057

-- Define the ceiling function and the arithmetic operations involved
example : Real := let inside := (8 - (3 / 4)) in 
                  let multiplied := 4 * inside in 
                  ⌈multiplied⌉
                  
theorem ceil_eval :  ⌈4 * (8 - (3 / 4))⌉ = 29 := 
by
-- We'll skip the proof part using sorry
sorry

end ceil_eval_l714_714057


namespace value_of_f_neg1_plus_f_4_l714_714784

def f (x : ℝ) : ℝ :=
  if x < 2 then 
    -x^2 + 3 * x 
  else 
    2 * x - 1

theorem value_of_f_neg1_plus_f_4 : f (-1) + f 4 = 3 := by
  sorry

end value_of_f_neg1_plus_f_4_l714_714784


namespace lines_positional_relationship_l714_714468

noncomputable def linesParallelToPlane (l1 l2 : Line) (P : Plane) : Prop :=
  Parallel l1 P ∧ Parallel l2 P

theorem lines_positional_relationship (l1 l2 : Line) (P : Plane)
  (h : linesParallelToPlane l1 l2 P) :
  (Parallel l1 l2 ∨ Intersect l1 l2 ∨ Skew l1 l2) :=
sorry

end lines_positional_relationship_l714_714468


namespace triangle_area_l714_714900

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area (a b : ℝ × ℝ) : 
  let area_parallelogram := (a.1 * b.2 - a.2 * b.1).abs in
  (1 / 2) * area_parallelogram = 4.5 :=
by
  sorry

end triangle_area_l714_714900


namespace dictionary_prices_unique_l714_714308

variables (x y m: ℕ)

theorem dictionary_prices_unique (
    h1: x + 2 * y = 170,
    h2: 2 * x + 3 * y = 290,
    h3: 70 * m + 50 * (300 - m) ≤ 16000
): x = 70 ∧ y = 50 ∧ m ≤ 50 :=
begin
  sorry
end

end dictionary_prices_unique_l714_714308


namespace find_k_l714_714322

theorem find_k
  (k : ℕ) (hk : 0 < k)
  (ratio_boys_girls : 5 / 8 = 5 : 8)
  (total_students : 80 * k)
  (prop_boys_like_soccer : 3 / 5)
  (prop_girls_like_soccer : 1 / 3)
  (chi_squared_formula : (80 * k * (30 * k * 20 * k - 20 * k * 10 * k) ^ 2) / (40 * k * 40 * k * 50 * k * 30 * k))
  (chi_squared_threshold : 3.841 ≤ chi_squared_formula / (16 / 3) ∧ chi_squared_formula / (16 / 3) < 6.635) :
  k = 1 :=
begin
  sorry
end

end find_k_l714_714322


namespace problem_statement_l714_714407

theorem problem_statement (m n : ℝ) (h : m + n = 1 / 2 * m * n) : (m - 2) * (n - 2) = 4 :=
by sorry

end problem_statement_l714_714407


namespace luke_jigsaw_puzzle_l714_714554

noncomputable def remaining_pieces_after_third_day (initial_pieces: ℕ) (day1_percent: ℝ) (day2_percent: ℝ) (day3_percent: ℝ) : ℕ :=
  let day1_done := initial_pieces * day1_percent in
  let remaining_after_day1 := initial_pieces - day1_done in
  let day2_done := remaining_after_day1 * day2_percent in
  let remaining_after_day2 := remaining_after_day1 - day2_done in
  let day3_done := remaining_after_day2 * day3_percent in
  let remaining_after_day3 := remaining_after_day2 - day3_done in
  remaining_after_day3.to_nat

theorem luke_jigsaw_puzzle : remaining_pieces_after_third_day 1000 0.1 0.2 0.3 = 504 :=
by 
  let initial_pieces := 1000
  let day1_percent := 0.1
  let day2_percent := 0.2
  let day3_percent := 0.3
  let day1_done := initial_pieces * day1_percent
  let remaining_after_day1 := initial_pieces - day1_done
  let day2_done := remaining_after_day1 * day2_percent
  let remaining_after_day2 := remaining_after_day1 - day2_done
  let day3_done := remaining_after_day2 * day3_percent
  let remaining_after_day3 := remaining_after_day2 - day3_done
  have h : remaining_after_day3 = 504 := by sorry
  exact h

end luke_jigsaw_puzzle_l714_714554


namespace cos_120_eq_neg_one_half_l714_714010

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714010


namespace store_purchases_part1_store_purchases_part2_l714_714373

theorem store_purchases_part1 (total_funds : ℕ) (tv_cost : ℕ) (wm_cost : ℕ) (total_items : ℕ) : 
  total_funds = 160000 → tv_cost = 2000 → wm_cost = 1000 → total_items = 100 →
  ∃ (tv_count wm_count : ℕ), tv_count = 60 ∧ wm_count = 40 :=
by
  intros h_funds h_tv_cost h_wm_cost h_total_items
  use 60, 40
  split
  sorry

theorem store_purchases_part2 (total_funds : ℕ) (tv_cost : ℕ) (rf_cost : ℕ) (wm_cost : ℕ) (total_items : ℕ) :
  total_funds = 160000 → tv_cost = 2000 → rf_cost = 1600 → wm_cost = 1000 → total_items = 100 →
  ∃ (a : ℕ), 100 / 3 ≤ a ∧ a ≤ 37 ∧ 4_possible_plans a ∧ max_profit a 17400 :=
by
  intros h_funds h_tv_cost h_rf_cost h_wm_cost h_total_items
  use 37
  split; sorry

end store_purchases_part1_store_purchases_part2_l714_714373


namespace duration_of_first_part_hour_l714_714515

noncomputable def duration_first_part (total_time : ℝ) (total_cost : ℝ) (partial_hour_cost : ℝ) (hourly_cost : ℝ) : ℝ :=
(let x := 4.4167 - (total_cost - partial_hour_cost) / hourly_cost in
  x * 60)

theorem duration_of_first_part_hour :
  duration_first_part 4.4167 39.33 6 8 = 15 :=
by
  sorry

end duration_of_first_part_hour_l714_714515


namespace T_2018_l714_714794

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, a (n+2) = 2 * a (n+1) - a (n)

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range (n), a (i + 1)

def Tn (a : ℕ → ℕ) (n : ℕ) : ℝ :=
  ∑ j in range (n), 1 / (Sn a (j+1))

theorem T_2018 : ∃ a : ℕ → ℕ, sequence a ∧ Tn a 2018 = 4036 / 2019 :=
by
  sorry

end T_2018_l714_714794


namespace arith_prog_a1_a10_geom_prog_a1_a10_l714_714795

-- First we define our sequence and conditions for the arithmetic progression case
def is_arith_prog (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + d * (n - 1)

-- Arithmetic progression case
theorem arith_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_ap : is_arith_prog a) :
  a 1 * a 10 = -728 := 
  sorry

-- Then we define our sequence and conditions for the geometric progression case
def is_geom_prog (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Geometric progression case
theorem geom_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_gp : is_geom_prog a) :
  a 1 + a 10 = -7 := 
  sorry

end arith_prog_a1_a10_geom_prog_a1_a10_l714_714795


namespace truncated_pyramid_lateral_surface_area_l714_714680

noncomputable def lateral_surface_area_of_truncated_pyramid (lower_base upper_base height : ℝ) : ℝ :=
  let base1 := lower_base
  let base2 := upper_base
  let h := height
  let slope_height := real.sqrt (h^2 + ((base1 - base2) / 2)^2)
  let trapezoid_area := (base1 + base2) / 2 * slope_height
  4 * trapezoid_area

theorem truncated_pyramid_lateral_surface_area :
  lateral_surface_area_of_truncated_pyramid 10 5 6 = 195 :=
by
  let base1 := 10
  let base2 := 5
  let h := 6
  let slope_height := real.sqrt (h^2 + ((base1 - base2) / 2)^2)
  let trapezoid_area := (base1 + base2) / 2 * slope_height
  have eqn : 4 * trapezoid_area = 195 := by sorry
  exact eqn

end truncated_pyramid_lateral_surface_area_l714_714680


namespace effective_quote_of_stock_in_A_l714_714319

-- Define the conditions
def yield_percent : ℝ := 0.08
def face_value_in_B : ℝ := 100
def exchange_rate : ℝ := 1.5
def tax_percent : ℝ := 0.15
def stock_percent : ℝ := 0.20

-- Translate the given conditions and the result into Lean statement
theorem effective_quote_of_stock_in_A :
  ∀ (yield_percent face_value_in_B exchange_rate tax_percent stock_percent : ℝ),
  yield_percent = 0.08 →
  face_value_in_B = 100 →
  exchange_rate = 1.5 →
  tax_percent = 0.15 →
  stock_percent = 0.20 →
  let yield_in_B_before_tax  := yield_percent * face_value_in_B,
      yield_in_A_before_tax  := yield_in_B_before_tax / exchange_rate,
      tax_on_yield_in_A      := tax_percent * yield_in_A_before_tax,
      yield_in_A_after_tax   := yield_in_A_before_tax - tax_on_yield_in_A,
      effective_quote_in_A   := yield_in_A_after_tax / stock_percent in
  abs (effective_quote_in_A - 22.67) < 0.01 := 
by intros; sorry

end effective_quote_of_stock_in_A_l714_714319


namespace reciprocal_lcm_24_195_l714_714247

theorem reciprocal_lcm_24_195 :
  let a := 24
  let b := 195
  let lcm_ab := Nat.lcm a b
  lcm_ab = 1560 → 1 / lcm_ab = 1 / 1560 :=
by
  intros a b lcm_ab h
  rw [h]
  rfl

end reciprocal_lcm_24_195_l714_714247


namespace largest_k_divides_all_S_l714_714196

def largest_three_divisors_excluding_self (n : ℕ) : list ℕ :=
  (list.filter (λ d, d < n) (list.range (n+1))).reverse.take 3

def S : set ℕ := {n | (largest_three_divisors_excluding_self n).sum > n}

theorem largest_k_divides_all_S : ∃ k : ℕ, (∀ n ∈ S, k ∣ n) ∧ k = 6 :=
by
  sorry

end largest_k_divides_all_S_l714_714196


namespace quadratic_inequality_solution_l714_714381

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end quadratic_inequality_solution_l714_714381


namespace simplify_trig_l714_714973

theorem simplify_trig (x : ℝ) (h_cos_sin : cos x ≠ -1) (h_sin_ne_zero : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
    sorry

end simplify_trig_l714_714973


namespace total_teachers_students_l714_714601

noncomputable def buses_trip :=
  let seats_per_bus := 39
  let seats_per_public_bus := 27
  let buses_diff := 2
  let extra_seats := 3
  let extra_teachers := 3
  exists (x y : ℕ), 
    x = 9 ∧ 
    y = 2 * x + 330 ∧ 
    (buses_diff = (x + 2) - x) ∧ 
    (seats_per_bus * x - 2 * x - (37 * x - 3) = extra_seats) ∧ 
    (seats_per_public_bus * (x + 2) - (x - extra_teachers) = 27 * (x + 2)) ∧ 
    (2 * x = 18) ∧ 
    (37 * x - 3 = 330)

theorem total_teachers_students : ∃ (teachers students : ℕ), teachers = 18 ∧ students = 330 :=
by
  rw buses_trip
  sorry

end total_teachers_students_l714_714601


namespace exists_quadrilateral_with_obtuse_diagonals_l714_714886

-- Defining the notion of a quadrilateral
structure Quadrilateral (A B C D : Type*) :=
    (segments : (A × B) × (B × C) × (C × D) × (D × A))

-- Defining a predicate for obtuse triangles
def is_obtuse (α β γ : ℝ) : Prop :=
    (α > π/2 ∧ β < π) ∨ (β > π/2 ∧ β < π) ∨ (γ > π/2 ∧ γ < π)

-- Defining a non-convex quadrilateral
def non_convex_quadrilateral (A B C D : Type*) : Prop :=
    ∃ q : Quadrilateral A B C D, true  -- Placeholder; refining this to specify non-convex may require more detail

-- Defining the property that each diagonal splits the quadrilateral into two obtuse triangles
def diagonal_splits_into_obtuse_triangles (A B C D : Type*) (q : Quadrilateral A B C D) : Prop :=
    ∀ (diag : (A × C) ∨ (B × D)), 
        match diag with
        | Sum.inl _ => 
          -- Validate obtuse for triangles ABC and ACD
          ∃ α β γ δ ε : ℝ, is_obtuse α β γ ∧ is_obtuse δ ε α
        | Sum.inr _ => 
          -- Validate obtuse for triangles ABD and BCD
          ∃ α β γ δ ε : ℝ, is_obtuse α β γ ∧ is_obtuse δ ε α
        end

-- Formal statement
theorem exists_quadrilateral_with_obtuse_diagonals : 
    ∃ (A B C D : Type*) (q : Quadrilateral A B C D), non_convex_quadrilateral A B C D ∧ diagonal_splits_into_obtuse_triangles A B C D q :=
sorry

end exists_quadrilateral_with_obtuse_diagonals_l714_714886


namespace four_digit_numbers_ordered_digits_count_l714_714151

theorem four_digit_numbers_ordered_digits_count :
  let n := (9.choose 4)
  2 * n = 252 :=
by
  sorry

end four_digit_numbers_ordered_digits_count_l714_714151


namespace find_ages_l714_714328

def ages (S D M : ℕ) : Prop :=
  M = S + 20 ∧
  M = D + 15 ∧
  M + 2 = 2 * (S + 2) ∧
  M + 2 = 3 * (D + 2)

theorem find_ages : ∃ S D M : ℕ, ages S D M ∧ S = 18 ∧ D = 23 ∧ M = 38 := 
by {
  use [18, 23, 38],
  unfold ages,
  repeat { split },
  { refl }, -- 38 = 18 + 20
  { refl }, -- 38 = 23 + 15
  { linarith }, -- 40 = 2 * 20
  { linarith }, -- 40 = 3 * 10
}

end find_ages_l714_714328


namespace geometric_sequence_value_of_b_l714_714605

theorem geometric_sequence_value_of_b : 
  ∃ b : ℝ, 180 * (b / 180) = b ∧ (b / 180) * b = 64 / 25 ∧ b > 0 ∧ b = 21.6 :=
by sorry

end geometric_sequence_value_of_b_l714_714605


namespace fib_eq_solution_l714_714126

def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_eq_solution (x : ℝ) :
  x^2024 = fib 2023 * x + fib 2022 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
sorry

end fib_eq_solution_l714_714126


namespace product_of_integers_whose_cubes_sum_to_35_l714_714258

theorem product_of_integers_whose_cubes_sum_to_35 : 
  ∃ (a b : ℤ), a^3 + b^3 = 35 ∧ a * b = 6 :=
by
  use 3
  use 2
  split
  · norm_num
  · norm_num

end product_of_integers_whose_cubes_sum_to_35_l714_714258


namespace find_x_l714_714203

def star (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b)) / (Real.sqrt (a - b))

theorem find_x (x : ℝ) (h : star x 30 = 5) : x ≈ 37.475 :=
by
  have h1 : 0 < x - 30 := sorry  -- Because x must be greater than 30 based on the problem description
  have h2 : (Real.sqrt (x^2 + 30)) / (Real.sqrt (x - 30)) = 5 := h
  have h3 : (x^2 + 30) / (x - 30) = 25 := sorry  -- Based on squaring the previous equation
  have h4 : x^2 + 30 = 25 * (x - 30) := sorry  -- Cross-multiplying
  have h5 : x^2 - 25 * x + 780 = 0 := sorry  -- Rearranging like terms
  have roots : x = (25 ± Real.sqrt (25^2 - 4 * 780)) / 2 := sorry  -- Solving quadratic equation
  have root_positive : x = (25 + 49.95) / 2 := sorry  -- Selecting positive root
  have approx : (25 + 49.95) / 2 ≈ 37.475 := sorry
  exact approx

end find_x_l714_714203


namespace sum_of_roots_l714_714261

noncomputable def Q := sorry -- Define Q(x) as a monic quartic polynomial

-- Given conditions
def phi : ℝ := sorry -- Specify phi in the given range (π/4 < φ < π/2)
def Q_monic : ∀ x, lead_coeff (Q x) = 1 := sorry -- Q(x) is a monic polynomial
def roots_condition : ∀ x, (root Q x ↔ x = cos phi + I * sin phi ∨ x = cos phi - I * sin phi ∨ sorry) := sorry -- Roots' conditions forming a square
def area_condition : 2 * Q 0 = ((sin phi - sin (sorry))^2) := sorry -- Area condition

-- Statement to prove the sum of roots
theorem sum_of_roots : 
  (sum_of_roots Q) = 2 * cos phi + 2 * cos (sorry) := 
sorry

end sum_of_roots_l714_714261


namespace triangle_incircle_semicircles_radius_inequality_l714_714443

theorem triangle_incircle_semicircles_radius_inequality
  (triangle : Type)
  [is_triangle triangle]
  (semiperimeter : ℝ)
  (incircle_radius : ℝ)
  (semicircle_BC : semicircle)
  (semicircle_CA : semicircle)
  (semicircle_AB : semicircle)
  (tau : circle)
  (tau_radius : ℝ)
  (h1 : is_semiperimeter triangle semiperimeter)
  (h2 : has_incircle_with_radius triangle incircle_radius)
  (h3 : externally_constructed_semicircles triangle semicircle_BC semicircle_CA semicircle_AB)
  (h4 : tangent_to_all_semicircles tau semicircle_BC semicircle_CA semicircle_AB tau_radius semiperimeter)
  : (semiperimeter / 2) < tau_radius ∧ tau_radius ≤ (semiperimeter / 2) + ((1 - (Real.sqrt 3 / 2)) * incircle_radius) :=
by sorry

end triangle_incircle_semicircles_radius_inequality_l714_714443


namespace proof_ratio_problem_l714_714600

noncomputable def ratio_problem (a b c d : ℚ) : Prop :=
  (a / b = 5 / 4) ∧
  (c / d = 4 / 3) ∧
  (d / b = 1 / 5) →
  (a / c = 75 / 16)

theorem proof_ratio_problem (a b c d : ℚ) :
  ratio_problem a b c d :=
begin
  sorry
end

end proof_ratio_problem_l714_714600


namespace probability_A_and_B_in_same_group_l714_714837

def players : List Char := ['A', 'B', 'C', 'D', 'E']

def groups (l : List Char) : List (List Char × List Char) :=
  l.combinations 3 |>.map (λ g => (g, l.diff g))

def is_valid_grouping (g1 g2 : List Char) : Prop :=
  g1.length = 3 ∧ g2.length = 2

def fav_groupings : List (List Char × List Char) :=
  groups players |>.filter (λ ⟨g1, g2⟩ => 'A' ∈ g1 ∧ 'B' ∈ g1)

def total_groupings : List (List Char × List Char) :=
  groups players |>.filter (λ ⟨g1, g2⟩ => is_valid_grouping g1 g2)

theorem probability_A_and_B_in_same_group :
  fav_groupings.length / total_groupings.length = 2 / 5 := sorry

end probability_A_and_B_in_same_group_l714_714837


namespace midpoint_of_segment_l714_714631

theorem midpoint_of_segment : 
  let p1 := (12 : ℤ, -8 : ℤ)
  let p2 := (-6 : ℤ, 16 : ℤ)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  in midpoint = (3, 4) :=
by
  sorry

end midpoint_of_segment_l714_714631


namespace angle_between_vectors_with_offset_l714_714385

noncomputable def vector_angle_with_offset : ℝ :=
  let v1 := (4, -1)
  let v2 := (6, 8)
  let dot_product := 4 * 6 + (-1) * 8
  let magnitude_v1 := Real.sqrt (4 ^ 2 + (-1) ^ 2)
  let magnitude_v2 := Real.sqrt (6 ^ 2 + 8 ^ 2)
  let cos_theta := dot_product / (magnitude_v1 * magnitude_v2)
  Real.arccos cos_theta + 30

theorem angle_between_vectors_with_offset :
  vector_angle_with_offset = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 := 
sorry

end angle_between_vectors_with_offset_l714_714385


namespace probability_alternating_colors_l714_714657

/--
A box contains 6 white balls and 6 black balls.
Balls are drawn one at a time.
What is the probability that all of my draws alternate colors?
-/
theorem probability_alternating_colors :
  let total_arrangements := Nat.factorial 12 / (Nat.factorial 6 * Nat.factorial 6)
  let successful_arrangements := 2
  successful_arrangements / total_arrangements = (1 : ℚ) / 462 := 
by
  sorry

end probability_alternating_colors_l714_714657


namespace smallest_non_prime_sums_consecutive_integers_l714_714519

theorem smallest_non_prime_sums_consecutive_integers :
  ∃ (n : ℕ), (∀ m, m ∈ {3 * n + 6, 3 * n + 5, 3 * n + 4, 3 * n + 3} → ¬Nat.Prime m) ∧ n = 7 :=
begin
  sorry
end

end smallest_non_prime_sums_consecutive_integers_l714_714519


namespace larinjaitis_age_l714_714893

theorem larinjaitis_age : 
  ∀ (birth_year : ℤ) (death_year : ℤ), birth_year = -30 → death_year = 30 → (death_year - birth_year + 1) = 1 :=
by
  intros birth_year death_year h_birth h_death
  sorry

end larinjaitis_age_l714_714893


namespace jackson_running_miles_end_program_l714_714889

theorem jackson_running_miles_end_program (initial_miles: ℕ) (weeks: ℕ) (additional_miles_per_week: ℕ): 
  initial_miles = 3 → weeks = 4 → additional_miles_per_week = 1 → 
  (initial_miles + weeks * additional_miles_per_week) = 7 :=
by {
  intro h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end jackson_running_miles_end_program_l714_714889


namespace exists_common_ratio_of_geometric_progression_l714_714476

theorem exists_common_ratio_of_geometric_progression (a r : ℝ) (h_pos : 0 < r) 
(h_eq: a = a * r + a * r^2 + a * r^3) : ∃ r : ℝ, r^3 + r^2 + r - 1 = 0 :=
by sorry

end exists_common_ratio_of_geometric_progression_l714_714476


namespace root_in_interval_l714_714586

theorem root_in_interval : ∃ c ∈ Set.Icc 1 2, (λ x : ℝ, x^5 + x - 3) c = 0 :=
by
  -- f(x) = x^5 + x - 3
  let f : ℝ → ℝ := λ x, x^5 + x - 3
  have f_continuous : Continuous f := sorry
  have f_at_1 : f 1 < 0 := by norm_num
  have f_at_2 : f 2 > 0 := by norm_num
  exact ⟨IntermediateValueTheorem f f_continuous 1 2 f_at_1 f_at_2, sorry⟩

end root_in_interval_l714_714586


namespace polyhedron_volume_l714_714878

def right_triangle (a b : ℝ) := ∃ c, a^2 + b^2 = c^2
def rectangle (l w : ℝ) := l > 0 ∧ w > 0
def regular_pentagon (s : ℝ) := ∃ a, 5 * s^2 / (4 * tan (54 * π / 180)) = a

-- Define the polyhedron construction condition
def polyhedron_condition (polygons : Type) :=
  -- Polygons A, E, F are right triangles with legs 2 and 1
  (∀ (A E F : polygons), right_triangle 2 1) ∧
  -- Polygons B, C, D are rectangles with length 2 and width 1
  (∀ (B C D : polygons), rectangle 2 1) ∧
  -- Polygon G is a regular pentagon with side length 2
  (∀ G : polygons, regular_pentagon 2)

-- Define the theorem to be proved
theorem polyhedron_volume : polyhedron_condition polyhedron → (volume_of polyhedron = 9.62) :=
sorry

end polyhedron_volume_l714_714878


namespace quadratic_inequality_solution_set_max_value_of_a_mul_b_l714_714814

open Classical

variable {m n a b : ℝ}
variable (x : ℝ) [h1 : x ∈ Set.Ioo 1 n]

theorem quadratic_inequality_solution_set :
  (x^2 - 3*x + m < 0 ↔ 1 < x ∧ x < n) ↔ (m = 2) ∧ (n = 2) := sorry

theorem max_value_of_a_mul_b (a b : ℝ) (h2 : 2*a + 4*b = 3) : 
  a * b ≤ 9 / 32 := sorry

end quadratic_inequality_solution_set_max_value_of_a_mul_b_l714_714814


namespace maria_savings_after_purchase_l714_714924

theorem maria_savings_after_purchase
  (cost_sweater : ℕ)
  (cost_scarf : ℕ)
  (cost_mittens : ℕ)
  (num_family_members : ℕ)
  (savings : ℕ)
  (total_cost_one_set : ℕ)
  (total_cost_all_sets : ℕ)
  (amount_left : ℕ)
  (h1 : cost_sweater = 35)
  (h2 : cost_scarf = 25)
  (h3 : cost_mittens = 15)
  (h4 : num_family_members = 10)
  (h5 : savings = 800)
  (h6 : total_cost_one_set = cost_sweater + cost_scarf + cost_mittens)
  (h7 : total_cost_all_sets = total_cost_one_set * num_family_members)
  (h8 : amount_left = savings - total_cost_all_sets)
  : amount_left = 50 :=
sorry

end maria_savings_after_purchase_l714_714924


namespace a_formula_b_formula_t_range_l714_714093

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * (n + 1)

-- Define the sequence b_n
def b (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n + 1) \ Finset.range (n + 1), 1 / a (i + 1).to_real

-- State the theorem about a_n
theorem a_formula (n : ℕ) (hn : n ≥ 2) : a n = n * (n + 1) := sorry

-- State the theorem about b_n
theorem b_formula (n : ℕ) : b n = 1 / (n + 1) - 1 / (2 * n + 1) := sorry

-- State the theorem about the range of t
theorem t_range (t : ℝ) (n : ℕ) (hn : n ≥ 1) : 
  t^2 - 2 * t + 1 / 6 > b n → t ∈ (-∞ : ℝ, 0) ∪ (2, +∞) := sorry

end a_formula_b_formula_t_range_l714_714093


namespace quadratic_roots_r12_s12_l714_714546

theorem quadratic_roots_r12_s12 (r s : ℝ) (h1 : r + s = 2 * Real.sqrt 3) (h2 : r * s = 1) :
  r^12 + s^12 = 940802 :=
sorry

end quadratic_roots_r12_s12_l714_714546


namespace length_of_common_chord_AB_l714_714143

-- Define the first circle equation
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0

-- Define the second circle equation
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the equation for the common chord
def common_chord (x y : ℝ) : Prop := 4*x + 3*y - 10 = 0

-- Define the center of the first circle
def center_C1 : ℝ × ℝ := (5, 5)

-- Define the radius of the first circle
def radius_C1 : ℝ := 5 * Real.sqrt 2

-- Define the distance from the center of the first circle to the common chord
def distance_to_chord : ℝ := abs (20 + 15 - 10) / 5

-- State the theorem about the length of the common chord AB
theorem length_of_common_chord_AB : 
  2 * Real.sqrt ((radius_C1) ^ 2 - (distance_to_chord) ^ 2) = 10 :=
by
  sorry

end length_of_common_chord_AB_l714_714143


namespace students_with_dog_and_cat_only_l714_714181

theorem students_with_dog_and_cat_only
  (U : Finset (ℕ)) -- Universe of students
  (D C B : Finset (ℕ)) -- Sets of students with dogs, cats, and birds respectively
  (hU : U.card = 50)
  (hD : D.card = 30)
  (hC : C.card = 35)
  (hB : B.card = 10)
  (hIntersection : (D ∩ C ∩ B).card = 5) :
  ((D ∩ C) \ B).card = 25 := 
sorry

end students_with_dog_and_cat_only_l714_714181


namespace eighth_grade_probability_female_win_l714_714490

theorem eighth_grade_probability_female_win:
  let P_Alexandra : ℝ := 1 / 4,
      P_Alexander : ℝ := 3 / 4,
      P_Evgenia : ℝ := 1 / 4,
      P_Yevgeny : ℝ := 3 / 4,
      P_Valentina : ℝ := 2 / 5,
      P_Valentin : ℝ := 3 / 5,
      P_Vasilisa : ℝ := 1 / 50,
      P_Vasily : ℝ := 49 / 50 in
  let P_female : ℝ :=
    1 / 4 * (P_Alexandra + 
             P_Evgenia + 
             P_Valentina + 
             P_Vasilisa) in
  P_female = 1 / 16 + 1 / 48 + 3 / 20 + 1 / 200 :=
sorry

end eighth_grade_probability_female_win_l714_714490


namespace simplify_trig_expression_l714_714977

theorem simplify_trig_expression (x : ℝ) (hx : x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := 
by 
  sorry

end simplify_trig_expression_l714_714977


namespace vases_needed_l714_714340

theorem vases_needed
  (vase_capacity : ℕ) (num_carnations : ℕ) (num_roses : ℕ)
  (total_flowers : ℕ) (vases : ℕ) :
  vase_capacity = 9 →
  num_carnations = 4 →
  num_roses = 23 →
  total_flowers = num_carnations + num_roses →
  vases = total_flowers / vase_capacity →
  vases = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h2, h3] at h4
  rw h1 at h5
  rw [h4, nat.add_comm 4 23] at h5
  norm_num at h5
  exact h5

end vases_needed_l714_714340


namespace problem_integer_part_of_S_l714_714542

noncomputable def closest_integer (n : ℕ) : ℕ :=
if h : ∃ k : ℕ, (k - 1 / 2 : ℝ) < real.sqrt n ∧ real.sqrt n < (k + 1 / 2 : ℝ)
then classical.some h else 0

instance : DecidableEq ℝ := classical.decEq

noncomputable def S : ℝ :=
(1 / closest_integer 1) + (1 / closest_integer 2) + ... + (1 / closest_integer 2000)

theorem problem_integer_part_of_S : ⌊S⌋ = 88 := 
sorry

end problem_integer_part_of_S_l714_714542


namespace sum_of_max_min_of_f_l714_714549

noncomputable def f (x : ℝ) : ℝ :=
  (x + 1) ^ 2 + real.log (real.sqrt (x ^ 2 + 1) + x) / (x ^ 2 + 1)

theorem sum_of_max_min_of_f : 
  let M := real.to_nnreal ∘ max (f 0), N := real.to_nnreal ∘ min (f 0)
  in M + N = 2 := sorry

end sum_of_max_min_of_f_l714_714549


namespace bisect_segment_EF_of_AP_l714_714874

section Geometry

open Classical
noncomputable theory

variables {A B C O E F P: Point}
variables (circumcircle : Circle)
variables (segmentE : Segment O A B E)
variables (segmentF : Segment O A C F)
variables (tangentB : Tangent B)
variables (tangentC : Tangent C)
variables (O_center_A : IsCenter O)
variables (perp_OA : Perpendicular (Radius O A) (Line O E F))

theorem bisect_segment_EF_of_AP 
  (circumcircle_A : OnCircle A circumcircle)
  (circumcircle_B : OnCircle B circumcircle)
  (circumcircle_C : OnCircle C circumcircle)
  (perp_E : Perpendicular (Line O A) (Line A E))
  (perp_F : Perpendicular (Line O A) (Line A F))
  (tangent_point_P : IntersectTangents B C P)
  : Bisects (Line A P) (Segment E F) :=
by
  sorry

end Geometry

end bisect_segment_EF_of_AP_l714_714874


namespace sum_of_squares_2222_l714_714063

theorem sum_of_squares_2222 :
  ∀ (N : ℕ), (∃ (k : ℕ), N = 2 * 10^k - 1) → (∀ (a b : ℤ), N = a^2 + b^2 ↔ N = 2) :=
by sorry

end sum_of_squares_2222_l714_714063


namespace number_of_roses_now_l714_714279

-- Given Conditions
def initial_roses : Nat := 7
def initial_orchids : Nat := 12
def current_orchids : Nat := 20
def orchids_more_than_roses : Nat := 9

-- Question to Prove: 
theorem number_of_roses_now :
  ∃ (R : Nat), (current_orchids = R + orchids_more_than_roses) ∧ (R = 11) :=
by {
  sorry
}

end number_of_roses_now_l714_714279


namespace maximal_area_sum_l714_714920

variable (A B C D R P P' Q Q' : Type) 
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited R] [Inhabited P] [Inhabited P'] [Inhabited Q] [Inhabited Q']

-- Variables representing the lengths of the sides of triangle ABC
variable (AB BC CA : ℝ)
variable [Nonempty ℝ] -- Ensure non-empty reals for defining lengths

-- Conditions from the problem
hypothesis h1: AB = 7
hypothesis h2: BC = 9
hypothesis h3: CA = 4
hypothesis h4: AB ∥ CD
hypothesis h5: CA ∥ BD
hypothesis h6: (line_through R ∥ CA) ∧ (line_through R ∥ AB)

-- Sum of the areas of triangles BPP', RP'Q', and CQQ'
noncomputable def S : ℝ := area (triangle B P P') + area (triangle R P' Q') + area (triangle C Q Q')

-- The theorem to be proven
theorem maximal_area_sum : (S ^ 2) = 180 := 
by 
  -- The detailed construction and calculation are skipped, relying on given implicit hypothesis
  sorry

end maximal_area_sum_l714_714920


namespace mary_characters_initials_l714_714925

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l714_714925


namespace additional_terms_left_side_l714_714948

theorem additional_terms_left_side (k : ℕ) (h : 1 + ∑ i in (finset.range (2^k)), (1 : ℚ)/ (i+1) > k / 2) :
  (2^(k+1) - 2^k) = 2^k :=
by
  sorry

end additional_terms_left_side_l714_714948


namespace solve_for_r_l714_714238

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end solve_for_r_l714_714238


namespace ceil_neg_sqrt_64_div_9_l714_714740

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l714_714740


namespace two_digit_numbers_divide_all_their_relatives_l714_714908

def is_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_relative (n ab : ℕ) : Prop :=
  let a := ab / 10
  let b := ab % 10
  is_digit a ∧ is_digit b ∧ -- condition to ensure only two-digit numbers
  (n % 10 = b) ∧
  (let d_sum := sum_of_digits (n / 10)
   in d_sum ≠ 0 ∧ d_sum = a)

def divides_all_relatives (ab : ℕ) : Prop :=
  ∀ (n : ℕ), is_relative n ab → ab ∣ n

theorem two_digit_numbers_divide_all_their_relatives :
  {ab : ℕ | is_digit (ab / 10) ∧ is_digit (ab % 10) ∧ divides_all_relatives ab } =
  {ab : ℕ | ab ∈ {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 45, 90}} :=
by sorry

end two_digit_numbers_divide_all_their_relatives_l714_714908


namespace measure_angle_ACB_l714_714875

theorem measure_angle_ACB (angle_ABD angle_BAC angle_ABD_plus_ABC_sum : ℝ) (h1 : angle_ABD = 135) (h2 : angle_BAC = 108) (h3 : angle_ABD + angle_BAC + angle_ACB = 180) : angle_ACB = 27 :=
by
  have angle_ABC := 180 - angle_ABD
  have angle_sum := angle_ABC + angle_BAC
  have angle_ACB := 180 - angle_sum
  have h4 : angle_ABC = 45 := by simp [h1]
  have h5 : angle_sum = 153 := by simp [h2, h4]
  have h6 : angle_ACB = 27 := by simp [h3, h5]
  sorry

end measure_angle_ACB_l714_714875


namespace maximum_dot_product_l714_714783

noncomputable def max_value_OP_OC (OA OB OC OP : ℝ^3) : ℝ :=
  sqrt 3 / 7

/-- Given OA, OB, OC are three unit vectors in space, OA orthogonal to OB, OA orthogonal to
OC with an angle of 60 degrees between OB and OC. And point P is an arbitrary point in space such
that |OP| = 1, with |dot(OP, OC)| ≤ |dot(OP, OB)| ≤ |dot(OP, OA)|. Then the maximum value of |dot(OP, OC)|
is √(21)/7 -/
theorem maximum_dot_product {OA OB OC OP : ℝ^3}
  (h1: ∥OA∥ = 1) (h2: ∥OB∥ = 1) (h3: ∥OC∥ = 1)
  (h4: dot OA OB = 0) (h5: dot OA OC = 0)
  (h6: cos 60 = dot OB OC)
  (h7: ∥OP∥ = 1)
  (h8: abs (dot OP OC) ≤ abs (dot OP OB))
  (h9: abs (dot OP OB) ≤ abs (dot OP OA)) :
  abs (dot OP OC) ≤ sqrt (21) / 7 :=
sorry

end maximum_dot_product_l714_714783


namespace range_of_omega_l714_714786

theorem range_of_omega (ω : ℝ) (hω : ω > 2/3) :
  (∀ x : ℝ, x = (k : ℤ) * π / ω + 3 * π / (4 * ω) → (x ≤ π ∨ x ≥ 2 * π) ) →
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) :=
by
  sorry

end range_of_omega_l714_714786


namespace coefficients_polynomial_l714_714408

theorem coefficients_polynomial (a : Fin 11 → ℤ) :
  (∀ (x : ℝ), (x + 1) ^ 10 = ∑ i in Finset.range 11, a i * (1 - x) ^ i) →
  a 1 = -5120 :=
by
  sorry

end coefficients_polynomial_l714_714408


namespace cos_120_eq_neg_one_half_l714_714011

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714011


namespace minimum_sum_of_areas_l714_714804

theorem minimum_sum_of_areas (y₁ y₂ : ℝ) (h₁ : y₁ * y₂ = -16) (h₂ : y₁ ≠ y₂) (h₃ : y₁ ≠ 0) (h₄ : y₂ ≠ 0) :
  let x₁ := (y₁^2) / 4
  let x₂ := (y₂^2) / 4
  let area_AOB := (1/2) * real.abs (x₁ * y₂ - x₂ * y₁)
  let area_AOF := (1/2) * real.abs (y₁ - 2 * y₂)
  (area_AOB + area_AOF) ≥ 8 * real.sqrt 5 :=
by
  let x₁ := (y₁^2) / 4
  let x₂ := (y₂^2) / 4
  let area_AOB := (1/2) * real.abs (x₁ * y₂ - x₂ * y₁)
  let area_AOF := (1/2) * real.abs (y₁ - 2 * y₂)
  sorry

end minimum_sum_of_areas_l714_714804


namespace drive_back_distance_l714_714060

variables (x D : ℝ)

-- On Sunday, Daniel drove back from work at a constant speed of x miles per hour
def T_sunday := D / x

-- On Monday, he drove the first 32 miles at 2x miles per hour
def T_monday_first_32 := 32 / (2 * x)

-- and the rest at x/2 miles per hour
def T_monday_rest := (D - 32) / (x / 2)

-- Total time on Monday
def T_monday := T_monday_first_32 + T_monday_rest

-- The given condition that time on Monday is 52% longer than on Sunday
def time_condition := T_monday = 1.52 * T_sunday

theorem drive_back_distance : D = 100 :=
by
  assume x > 0,
  assume D > 0,
  have T_sunday_def : T_sunday = D / x := by rfl,
  have T_monday_first_32_def : T_monday_first_32 = 32 / (2 * x) := by rfl,
  have T_monday_rest_def : T_monday_rest = (D - 32) / (x / 2) := by rfl,
  have T_monday_def : T_monday = T_monday_first_32 + T_monday_rest := by rfl,
  have time_condition_def : time_condition := T_monday_def ▸ T_sunday_def ▸ sorry,
  exact sorry

end drive_back_distance_l714_714060


namespace find_length_DF_l714_714870

-- Definitions based on the conditions
variables {A B C D E F : Type} -- points
variables [linear_ordered_field ℝ]
variables (left_diagonal right_diagonal length_EB : ℝ)
variables (length_DE length_DF correct_val_AB correct_val_BC : ℝ)

-- We are given these conditions
def condition_1 := correct_val_BC = 15
def condition_2 := length_EB = 5
def condition_3 := length_DE = 6
def condition_4 := correct_val_AB = 15

-- The goal to prove
theorem find_length_DF (h1 : correct_val_BC = 15) (h2 : length_EB = 5) (h3 : length_DE = 6) (h4 : correct_val_AB = 15) : length_DF = 6 :=
by {
  sorry, -- this is where the proof would go
}

end find_length_DF_l714_714870


namespace number_of_three_digit_numbers_divisible_by_17_l714_714152

theorem number_of_three_digit_numbers_divisible_by_17 : 
  let k_min := Nat.ceil (100 / 17)
  let k_max := Nat.floor (999 / 17)
  ∃ n, 
    (n = k_max - k_min + 1) ∧ 
    (n = 53) := 
by
    sorry

end number_of_three_digit_numbers_divisible_by_17_l714_714152


namespace relationship_between_profit_and_tomato_area_number_of_planting_schemes_l714_714341

def total_area := 100
def profit_per_hectare_tomato := 0.01
def profit_per_hectare_green_pepper := 0.015
def profit_per_hectare_potato := 0.02

def planting_areas_satisfy (x : ℕ) (gp_area : ℕ) (p_area : ℕ) : Prop :=
  total_area = x + gp_area + p_area

def profit (x gp_area p_area : ℕ) : ℝ :=
  profit_per_hectare_tomato * x + profit_per_hectare_green_pepper * gp_area + profit_per_hectare_potato * p_area

theorem relationship_between_profit_and_tomato_area (x gp_area p_area y : ℕ) 
  (h1 : planting_areas_satisfy x gp_area p_area)
  (h2 : gp_area = 2 * x)
  (h3 : p_area = total_area - x - gp_area)
  (h4 : y = profit x gp_area p_area) :
  y = -2 * x + 200 :=
  sorry

theorem number_of_planting_schemes
  (x gp_area p_area y : ℕ) 
  (h1 : planting_areas_satisfy x gp_area p_area)
  (h2 : gp_area = 2 * x)
  (h3 : p_area = total_area - x - gp_area)
  (h4 : y = profit x gp_area p_area) 
  (h5 : y ≥ 180)
  (h6 : x ≥ 8) :
  ∃ (n_schemes : ℕ), n_schemes = 3 :=
  sorry

end relationship_between_profit_and_tomato_area_number_of_planting_schemes_l714_714341


namespace min_value_theorem_l714_714403

noncomputable def min_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_theorem (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  min_value a b h₀ h₁ h₂ ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_theorem_l714_714403


namespace find_side_b_l714_714854

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

def angle_A := degrees_to_radians 75
def angle_C := degrees_to_radians 60
def side_c := 1

theorem find_side_b (b : ℝ) :
  let B := Real.pi - angle_A - angle_C
  ∃ b, b = (side_c * Real.sin B) / (Real.sin angle_C) ∧ b = (Real.sqrt 6) / 3 :=
begin
  -- proof goes here
  sorry
end

end find_side_b_l714_714854


namespace angle_between_side_and_diagonal_of_triangular_prism_correct_l714_714327

noncomputable def angle_between_side_and_diagonal_of_triangular_prism (a : ℝ) : ℝ :=
  real.arccos (real.sqrt 2 / 4)

theorem angle_between_side_and_diagonal_of_triangular_prism_correct (a : ℝ) :
  let α := angle_between_side_and_diagonal_of_triangular_prism a in
  α = real.arccos (real.sqrt 2 / 4) :=
by
  sorry

end angle_between_side_and_diagonal_of_triangular_prism_correct_l714_714327


namespace system_sum_of_squares_l714_714575

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end system_sum_of_squares_l714_714575


namespace intersecting_circles_single_point_thm_l714_714169

/-- Representation of the problem involving intersecting circles in a segment -/
theorem intersecting_circles_single_point_thm
    (A B P : ℝ)
    (h_eqd : dist A P = dist B P)
    (inscribed_circles : ∀ S1 S2 : Circle, inscribed S1 A B ∧ inscribed S2 A B → intersects S1 S2)
    (line_intersection : ∀ (S1 S2 : Circle) (M N : ℝ), intersects_at S1 S2 M N → Line M N) :
  ∀ (S1 S2 : Circle) (M N : ℝ), 
    intersects_at S1 S2 M N → Line_through M N P :=
sorry

end intersecting_circles_single_point_thm_l714_714169


namespace ratio_S15_S5_l714_714122

variable {α : Type*} [LinearOrderedField α]

namespace ArithmeticSequence

def sum_of_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem ratio_S15_S5
  {a d : α}
  {S5 S10 S15 : α}
  (h1 : S5 = sum_of_first_n_terms a d 5)
  (h2 : S10 = sum_of_first_n_terms a d 10)
  (h3 : S15 = sum_of_first_n_terms a d 15)
  (h_ratio : S5 / S10 = 2 / 3) :
  S15 / S5 = 3 / 2 := 
sorry

end ArithmeticSequence

end ratio_S15_S5_l714_714122


namespace odd_numbers_divisibility_l714_714232

theorem odd_numbers_divisibility (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  (ab_minus_one_is_divisible_by_4 : ((a * b - 1) % 4 = 0) ∨
   (bc_minus_one_is_divisible_by_4 : (b * c - 1) % 4 = 0) ∨
   (ca_minus_one_is_divisible_by_4 : (c * a - 1) % 4 = 0)) :=
  sorry

end odd_numbers_divisibility_l714_714232


namespace system_sum_of_squares_l714_714576

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end system_sum_of_squares_l714_714576


namespace cos_120_eq_neg_half_l714_714014

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714014


namespace find_x_l714_714316

theorem find_x (x : ℝ) (h : 121 * x^4 = 75625) : x = 5 :=
sorry

end find_x_l714_714316


namespace find_ab_unique_l714_714447

theorem find_ab_unique (a b : ℕ) (h1 : a > 1) (h2 : b > a) (h3 : a ≤ 20) (h4 : b ≤ 20) (h5 : a * b = 52) (h6 : a + b = 17) : a = 4 ∧ b = 13 :=
by {
  -- Proof goes here
  sorry
}

end find_ab_unique_l714_714447


namespace sum_of_three_consecutive_numbers_l714_714603

theorem sum_of_three_consecutive_numbers (smallest : ℕ) (h : smallest = 29) :
  (smallest + (smallest + 1) + (smallest + 2)) = 90 :=
by
  sorry

end sum_of_three_consecutive_numbers_l714_714603


namespace simplify_trig_expression_l714_714983

theorem simplify_trig_expression (x : ℝ) (hx : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l714_714983


namespace necessary_but_not_sufficient_cos_α_l714_714125

open Real

theorem necessary_but_not_sufficient_cos_α (α : ℝ) : 
  (cos α = -√3 / 2) ↔ (∃ k : ℤ, α = 2 * k * π + 5 * π / 6 ∨ α = 2 * k * π - 5 * π / 6) :=
sorry

end necessary_but_not_sufficient_cos_α_l714_714125


namespace exists_X_Y_l714_714464

theorem exists_X_Y {A n : ℤ} (h_coprime : Int.gcd A n = 1) :
  ∃ X Y : ℤ, |X| < Int.sqrt n ∧ |Y| < Int.sqrt n ∧ n ∣ (A * X - Y) :=
sorry

end exists_X_Y_l714_714464


namespace value_of_s_l714_714919

theorem value_of_s (n : ℕ) : 
  let a := (1 - x + x^2)^n = a_0 + a_1*x + a_2*x^2 + ⋯ + a_{2n}*x^{2n} in
  let s := a_0 + a_2 + a_4 + ⋯ + a_{2n} in
  s = (1 + 3^n) / 2 :=
by
  sorry

end value_of_s_l714_714919


namespace sum_of_squares_of_solutions_l714_714577

def is_solution (x y : ℝ) : Prop := 
  9 * y^2 - 4 * x^2 = 144 - 48 * x ∧ 
  9 * y^2 + 4 * x^2 = 144 + 18 * x * y

theorem sum_of_squares_of_solutions : 
  (∀ (x y : ℝ), is_solution x y → x^2 + y^2 ∈ {0, 16, 36}) → 
  ∑ (x, y) in {(0, 4), (0, -4), (6, 0)}, x^2 + y^2 = 68 :=
by 
  intros h
  -- the steps can be filled later
  sorry

end sum_of_squares_of_solutions_l714_714577


namespace equation_of_circle_area_of_triangle_l714_714808

-- Problem 1: Prove the equation of the circle M
theorem equation_of_circle
  (M : ℝ × ℝ)
  (hx : M.snd = 0)
  (r : ℝ)
  (hr : r = 1)
  (chord_len : ℝ)
  (hchord : chord_len = sqrt 3)
  (hc : M.snd < l) :
  (x y : ℝ) (intersects_chord_on_l : (x - 1)^2 + y^2 = 1) : 
  ((x - 1)^2 + y^2 = 1) :=
sorry

-- Problem 2: Prove the maximum and minimum areas of triangle ABC
theorem area_of_triangle 
  (A B : ℝ × ℝ)
  (hA : A = (0, t))
  (hB : B = (0, t + 6))
  (h_t_range : -5 ≤ t ∧ t ≤ -2)
  (M : ℝ × ℝ)
  (hx : M.snd = 0)
  (r : ℝ)
  (hr : r = 1)
  (inscribed_circle : circle M r) :
  max_min_area_of_triangle (ABC : triangle (A B C)) 
  (max_area : ℝ) (min_area : ℝ)
  (hmax : max_area = 15 / 2)
  (hmin : min_area = 27 / 4) :=
sorry

end equation_of_circle_area_of_triangle_l714_714808


namespace part_I_part_II_l714_714921

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 2 }
def B (a : ℝ) : Set ℝ := { x | 2 * a - 1 < x ∧ x < 2 * a + 1 }

-- Part (Ⅰ): Given A ⊆ B, prove that 1/2 ≤ a ≤ 1
theorem part_I (a : ℝ) : A ⊆ B a → (1 / 2 ≤ a ∧ a ≤ 1) :=
by sorry

-- Part (Ⅱ): Given A ∩ B = ∅, prove that a ≥ 3/2 or a ≤ 0
theorem part_II (a : ℝ) : A ∩ B a = ∅ → (a ≥ 3 / 2 ∨ a ≤ 0) :=
by sorry

end part_I_part_II_l714_714921


namespace rows_eq_cols_with_stars_l714_714482

variable {Row Col : Type}
variable (stars : Row → Col → Prop)

def num_rows_with_stars : ℕ := (Set.toFinset {r : Row | ∃ c, stars r c}).card
def num_cols_with_stars : ℕ := (Set.toFinset {c : Col | ∃ r, stars r c}).card

theorem rows_eq_cols_with_stars (h : ∀ r c, stars r c → 
  (Set.toFinset {c' : Col | stars r c'}).card = (Set.toFinset {r' : Row | stars r' c}).card) :
  num_rows_with_stars stars = num_cols_with_stars stars := sorry

end rows_eq_cols_with_stars_l714_714482


namespace sqrt_expression_l714_714456

theorem sqrt_expression (h : n < m ∧ m < 0) : 
  (Real.sqrt (m^2 + 2 * m * n + n^2) - Real.sqrt (m^2 - 2 * m * n + n^2)) = -2 * m := 
by {
  sorry
}

end sqrt_expression_l714_714456


namespace combined_average_age_l714_714248

theorem combined_average_age 
    (avgA : ℕ → ℕ → ℕ) -- defines the average function
    (avgA_cond : avgA 6 240 = 40) 
    (avgB : ℕ → ℕ → ℕ)
    (avgB_cond : avgB 4 100 = 25) 
    (combined_total_age : ℕ := 340) 
    (total_people : ℕ := 10) : avgA (total_people) (combined_total_age) = 34 := 
by
  sorry

end combined_average_age_l714_714248


namespace socks_arrangement_count_l714_714450

def number_of_socks_arrangements (n : ℕ) : ℕ := (2 * n)!

theorem socks_arrangement_count (n : ℕ) : 
  (number_of_socks_arrangements n) / (2 ^ n) = ((2 * n)! / 2^n) := 
sorry

end socks_arrangement_count_l714_714450


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l714_714732

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l714_714732


namespace fibonacci_inequality_l714_714529

def fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (hn : 0 < n) : 
  real.root (n : ℝ) (fibonacci (n + 2) : ℝ) ≥ 1 + 1 / real.root (n : ℝ) (fibonacci (n + 1) : ℝ) :=
by 
  sorry

end fibonacci_inequality_l714_714529


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l714_714114

theorem leftmost_three_nonzero_digits_of_ring_arrangements :
  let n := (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3) in
  n.leftmost_three_nonzero_digits = 126 :=
by
  sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l714_714114


namespace gribblean_words_count_l714_714505

universe u

-- Define the Gribblean alphabet size
def alphabet_size : Nat := 3

-- Words of length 1 to 4
def words_of_length (n : Nat) : Nat :=
  alphabet_size ^ n

-- All possible words count
def total_words : Nat :=
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3) + (words_of_length 4)

-- Theorem statement
theorem gribblean_words_count : total_words = 120 :=
by
  sorry

end gribblean_words_count_l714_714505


namespace addition_and_rounding_l714_714683

def round_to_nearest_ten_thousandth (x : ℝ) : ℝ :=
  (Real.round (x * 10000)) / 10000

theorem addition_and_rounding :
  round_to_nearest_ten_thousandth (174.39875 + 28.06754) = 202.4663 :=
by
  sorry

end addition_and_rounding_l714_714683


namespace slices_left_large_slices_left_medium_slices_left_small_l714_714240

open Real

def pizza_slices := ℕ

variable (L : pizza_slices) (M : pizza_slices) (S : pizza_slices)
variable (slices_large : L = 12)
variable (slices_medium : M = 10)
variable (slices_small : S = 8)
variable (stephen_ate_large : L = 12 → L - (L * 25 / 100) = 9)
variable (stephen_ate_medium : M = 10 → M - (M * 15 / 100) = 9)
variable (pete_ate_large : L - (L * 25 / 100) = 9 → 9 - (9 * 50 / 100) = 5)
variable (pete_ate_medium : M - (M * 15 / 100) = 9 → 9 - (9 * 20 / 100) = 8)
variable (laura_ate_small : S = 8 → 8 - (8 * 30 / 100) = 6)

theorem slices_left_large (L : pizza_slices) (slices_large : L = 12)
    (stephen_ate_large : L = 12 → L - (L * 25 / 100) = 9)
    (pete_ate_large : L - (L * 25 / 100) = 9 → 9 - (9 * 50 / 100) = 5) : 5 := 
    have h₁ : L - (L * 25 / 100) = 9, by exact stephen_ate_large L
    have h₂ : 9 - (9 * 50 / 100) = 5, by exact pete_ate_large L
    5

theorem slices_left_medium (M : pizza_slices) (slices_medium : M = 10)
    (stephen_ate_medium : M = 10 → M - (M * 15 / 100) = 9)
    (pete_ate_medium : M - (M * 15 / 100) = 9 → 9 - (9 * 20 / 100) = 8) : 8 := 
    have h₁ : M - (M * 15 / 100) = 9, by exact stephen_ate_medium M
    have h₂ : 9 - (9 * 20 / 100) = 8, by exact pete_ate_medium M
    8

theorem slices_left_small (S : pizza_slices) (slices_small : S = 8)
    (laura_ate_small : S = 8 → 8 - (8 * 30 / 100) = 6) : 6 := 
    have h₁ : 8 - (8 * 30 / 100) = 6, by exact laura_ate_small S
    6

#check slices_left_large
#check slices_left_medium
#check slices_left_small

end slices_left_large_slices_left_medium_slices_left_small_l714_714240


namespace max_value_mn_l714_714828

theorem max_value_mn (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h : 2 * m + n = 1) : mn ≤ 1 / 8 :=
begin
  sorry  -- Skip the proof
end

end max_value_mn_l714_714828


namespace determine_counterfeit_coin_l714_714709

theorem determine_counterfeit_coin (wt_1 wt_2 wt_3 wt_5 : ℕ) (coin : ℕ) :
  (wt_1 = 1) ∧ (wt_2 = 2) ∧ (wt_3 = 3) ∧ (wt_5 = 5) ∧
  (coin = wt_1 ∨ coin = wt_2 ∨ coin = wt_3 ∨ coin = wt_5) ∧
  (coin ≠ 1 ∨ coin ≠ 2 ∨ coin ≠ 3 ∨ coin ≠ 5) → 
  ∃ (counterfeit : ℕ), (counterfeit = 1 ∨ counterfeit = 2 ∨ counterfeit = 3 ∨ counterfeit = 5) ∧ 
  (counterfeit ≠ 1 ∧ counterfeit ≠ 2 ∧ counterfeit ≠ 3 ∧ counterfeit ≠ 5) :=
by
  sorry

end determine_counterfeit_coin_l714_714709


namespace five_segments_acute_angle_l714_714409

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_obtuse (a b c : ℝ) : Prop :=
  c^2 > a^2 + b^2

def is_acute (a b c : ℝ) : Prop :=
  c^2 < a^2 + b^2

theorem five_segments_acute_angle (a b c d e : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (T1 : is_triangle a b c) (T2 : is_triangle a b d) (T3 : is_triangle a b e)
  (T4 : is_triangle a c d) (T5 : is_triangle a c e) (T6 : is_triangle a d e)
  (T7 : is_triangle b c d) (T8 : is_triangle b c e) (T9 : is_triangle b d e)
  (T10 : is_triangle c d e) : 
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
           is_triangle x y z ∧ is_acute x y z :=
by
  sorry

end five_segments_acute_angle_l714_714409


namespace probability_female_win_l714_714501

variable (P_Alexandr P_Alexandra P_Evgeniev P_Evgenii P_Valentinov P_Valentin P_Vasilev P_Vasilisa : ℝ)

-- Conditions
axiom h1 : P_Alexandr = 3 * P_Alexandra
axiom h2 : P_Evgeniev = (1 / 3) * P_Evgenii
axiom h3 : P_Valentinov = (1.5) * P_Valentin
axiom h4 : P_Vasilev = 49 * P_Vasilisa
axiom h5 : P_Alexandr + P_Alexandra = 1
axiom h6 : P_Evgeniev + P_Evgenii = 1
axiom h7 : P_Valentinov + P_Valentin = 1
axiom h8 : P_Vasilev + P_Vasilisa = 1

-- Statement to prove
theorem probability_female_win : 
  let P_female := (1/4) * P_Alexandra + (1/4) * P_Evgeniev + (1/4) * P_Valentinov + (1/4) * P_Vasilisa in
  P_female = 0.355 :=
by
  sorry

end probability_female_win_l714_714501


namespace angle_of_inclination_eq_90_l714_714109

theorem angle_of_inclination_eq_90 {a : ℝ} (A B : ℝ × ℝ) (hA : A = (1 + a, 2 * a)) (hB : B = (1 - a, 3)) :
  (∃ θ : ℝ, θ = 90 ∧ (A.2 - B.2) / (A.1 - B.1) = real.tan (θ * real.pi / 180)) → a = 0 :=
by
  sorry

end angle_of_inclination_eq_90_l714_714109


namespace sum_of_corners_l714_714717

theorem sum_of_corners : 
  let board := List.range' 1 73,   -- Generate the list [1,2,...,72]
      rows := board.chunk 12       -- Chunk the list into rows of 12 each
  in rows.head.head + rows.head[11]! + rows[5]!.head + rows[5]![11]! = 146 :=
by 
  let board := List.range' 1 73
  let rows := board.chunk 12
  have h1 : rows.head.head = 1 := rfl
  have h2 : rows.head[11]! = 12 := rfl
  have h3 : rows[5]!.head = 61 := rfl
  have h4 : rows[5]![11]! = 72 := rfl
  calc
  1 + 12 + 61 + 72 = 13 + 61 + 72 : by rw [add_assoc]
           ... = 74 + 72            : by rw [add_assoc]
           ... = 146                : rfl

end sum_of_corners_l714_714717


namespace boat_distance_along_stream_one_hour_l714_714869

-- Definitions of the given conditions
def speed_in_still_water : ℝ := 10
def distance_against_stream_in_one_hour : ℝ := 5

-- Definition of the unknown speed of the stream
def stream_speed : ℝ := sorry

-- Proof goal: The distance travelled along the stream in one hour
theorem boat_distance_along_stream_one_hour :
  let v_b := speed_in_still_water,
      d_u := distance_against_stream_in_one_hour,
      v_s := stream_speed in
  v_s = v_b - d_u →
  (v_b + v_s) = 15 :=
by sorry

end boat_distance_along_stream_one_hour_l714_714869


namespace last_two_nonzero_digits_of_70_fact_l714_714368

/-- The number obtained from the last two nonzero digits of 70! is 48. -/
theorem last_two_nonzero_digits_of_70_fact : (∃ (x : ℕ), x = (70.factorial / 10^16) % 100) → (x % 100) = 48 := by
  sorry

end last_two_nonzero_digits_of_70_fact_l714_714368


namespace interval_increasing_l714_714075

open Real

noncomputable def interval_monotonic_increasing 
  (f : ℝ → ℝ)
  (a b : ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

noncomputable def f (x : ℝ) := 2 * sin (π / 6 - 2 * x)

theorem interval_increasing : interval_monotonic_increasing f (π / 3) (5 * π / 6) :=
sorry

end interval_increasing_l714_714075


namespace cyclic_quad_circumcenter_orthocenter_l714_714791

-- Definitions based on the conditions
variables (A B C D P E M O O_star : Point)
variables (h1 : CyclicQuadrilateral A B C D)
variables (h2 : ¬(Parallel AB CD))
variables (h3 : IntersectsAt AC BD P)
variables (h4 : IntersectsAt AD BC E)
variables (h5 : Midpoint M O P)

-- Main theorem to prove
theorem cyclic_quad_circumcenter_orthocenter :
  Orthocenter (triangle O M E) O_star := sorry

end cyclic_quad_circumcenter_orthocenter_l714_714791


namespace john_protest_days_l714_714190

theorem john_protest_days (days1: ℕ) (days2: ℕ) (days3: ℕ): 
  days1 = 4 → 
  days2 = (days1 + (days1 / 4)) → 
  days3 = (days2 + (days2 / 2)) → 
  (days1 + days2 + days3) = 17 :=
by
  intros h1 h2 h3
  sorry

end john_protest_days_l714_714190


namespace equation_condition1_equation_condition2_equation_condition3_l714_714761

-- Definitions for conditions
def condition1 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ ∀ (x1 x2 : ℝ), (l x2 - l x1) / (x2 - x1) = -1/2)
def condition2 : Prop := ∃(l : ℝ → ℝ), (l 1 = 4 ∧ l 2 = 3)
def condition3 : Prop := ∃(l : ℝ → ℝ), (l 2 = 1 ∧ (∃ a : ℝ, l 0 = a ∧ l a = 0) ∨ (∃ a : ℝ, a > 0 ∧ l 0 = a = l a))

-- Proving equations given conditions
theorem equation_condition1 : condition1 → ∀ (x y : ℝ), x + 2 * y - 4 = 0 
:= sorry

theorem equation_condition2 : condition2 → ∀ (x y : ℝ), x + y - 5 = 0 
:= sorry

theorem equation_condition3 :
    condition3 → (∀ (x y : ℝ), (x - 2 * y = 0) ∨ (x + y - 3 = 0)) 
:= sorry

end equation_condition1_equation_condition2_equation_condition3_l714_714761


namespace max_value_of_8q_minus_9p_is_zero_l714_714485

theorem max_value_of_8q_minus_9p_is_zero (p : ℝ) (q : ℝ) (h1 : 0 < p) (h2 : p < 1) (hq : q = 3 * p ^ 2 - 2 * p ^ 3) : 
  8 * q - 9 * p ≤ 0 :=
by
  sorry

end max_value_of_8q_minus_9p_is_zero_l714_714485


namespace simplify_trig_expression_l714_714964

theorem simplify_trig_expression (x : ℝ) (h₁ : sin x ≠ 0) (h₂ : cos x ≠ -1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) := 
sorry

end simplify_trig_expression_l714_714964


namespace cos_120_eq_neg_half_l714_714024

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714024


namespace closest_point_on_plane_exists_l714_714767

def point_on_plane : Type := {P : ℝ × ℝ × ℝ // ∃ (x y z : ℝ), P = (x, y, z) ∧ 2 * x - 3 * y + 4 * z = 20}

def point_A : ℝ × ℝ × ℝ := (0, 1, -1)

theorem closest_point_on_plane_exists (P : point_on_plane) :
  ∃ (x y z : ℝ), (x, y, z) = (54 / 29, -80 / 29, 83 / 29) := sorry

end closest_point_on_plane_exists_l714_714767


namespace distinct_integers_in_form_l714_714918

theorem distinct_integers_in_form (x : ℝ) :
  ∃ n : ℕ, n = 600 ∧ ∀ i : ℕ, 
  1 ≤ i ∧ i ≤ 1000 →
  ∃ x : ℝ, f x = ⌊2*x⌋ + ⌊4*x⌋ + ⌊6*x⌋ + ⌊8*x⌋ ∧ f x = i
sorry

end distinct_integers_in_form_l714_714918


namespace range_of_function_l714_714153

theorem range_of_function :
  ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) →
  let f := λ x, (1/4)^x - 3 * (1/2)^x + 2
  in (∃ y, y = f x ∧ -1/4 ≤ y ∧ y ≤ 6) :=
by
  sorry

end range_of_function_l714_714153


namespace eighth_grade_probability_female_win_l714_714491

theorem eighth_grade_probability_female_win:
  let P_Alexandra : ℝ := 1 / 4,
      P_Alexander : ℝ := 3 / 4,
      P_Evgenia : ℝ := 1 / 4,
      P_Yevgeny : ℝ := 3 / 4,
      P_Valentina : ℝ := 2 / 5,
      P_Valentin : ℝ := 3 / 5,
      P_Vasilisa : ℝ := 1 / 50,
      P_Vasily : ℝ := 49 / 50 in
  let P_female : ℝ :=
    1 / 4 * (P_Alexandra + 
             P_Evgenia + 
             P_Valentina + 
             P_Vasilisa) in
  P_female = 1 / 16 + 1 / 48 + 3 / 20 + 1 / 200 :=
sorry

end eighth_grade_probability_female_win_l714_714491


namespace sum_of_digits_B_l714_714210

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

def A : ℕ := sum_of_digits (4444 ^ 4444)
def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : sum_of_digits B = 7 := 
  sorry

end sum_of_digits_B_l714_714210


namespace count_valid_three_digit_numbers_divisible_by_5_l714_714400

open Nat

def valid_three_digit_numbers_divisible_by_5 : Set (Fin 1000) :=
  { n | let d2 := n / 100, d1 := (n / 10) % 10, d0 := n % 10 in
        n >= 100 ∧ d0 = 5 ∧ d2 ≠ d1 ∧ d2 ≠ d0 ∧ d1 ≠ d0 ∧
        d2 ∈ ({0, 1, 2, 3, 4, 5} : Set Nat) ∧
        d1 ∈ ({0, 1, 2, 3, 4, 5} : Set Nat) ∧
        d0 ∈ ({0, 1, 2, 3, 4, 5} : Set Nat)
  }

theorem count_valid_three_digit_numbers_divisible_by_5 :
  (valid_three_digit_numbers_divisible_by_5.toFinset.card = 36) :=
by
  sorry

end count_valid_three_digit_numbers_divisible_by_5_l714_714400


namespace find_abc_value_l714_714404

noncomputable def abc_value_condition (a b c : ℝ) : Prop := 
  a + b + c = 4 ∧
  b * c + c * a + a * b = 5 ∧
  a^3 + b^3 + c^3 = 10

theorem find_abc_value (a b c : ℝ) (h : abc_value_condition a b c) : a * b * c = 2 := 
sorry

end find_abc_value_l714_714404


namespace find_roots_l714_714392

variable {n : ℕ} (a_2 a_3 ... a_n : ℝ)

def polynomial (x : ℝ) : ℝ :=
  x^n + n * x^(n-1) + a_2 * x^(n-2) + ... + a_n

theorem find_roots (r : ℕ → ℝ) (h : ∑ k in finset.range 16, r k ^ 16 = n) :
  (∀ i, r i = -1) :=
begin
  sorry,
end

end find_roots_l714_714392


namespace prove_inequality_l714_714689

-- Define the sequence {b_n}
noncomputable def b_n (α : ℕ → ℕ) : ℕ → ℚ
| 1 := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b_n α n)

-- Example α values for simplification like α_k = 1
def example_α (k : ℕ) : ℕ := 1

-- The statement to be proved
theorem prove_inequality (α : ℕ → ℕ) (h : ∀ k, 0 < α k) : (b_n α 4 < b_n α 7) :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end prove_inequality_l714_714689


namespace comparison_l714_714697

def sequence (α : ℕ → ℕ) : ℕ → ℚ
| 1     := 1 + 1 / (α 1)
| (n+1) := 1 + 1 / (α 1 + sequence (λ k, α (k+1) n))

theorem comparison (α : ℕ → ℕ) (h : ∀ k, 1 ≤ α k) :
  sequence α 4 < sequence α 7 := 
sorry

end comparison_l714_714697


namespace daily_soda_consumption_l714_714037

theorem daily_soda_consumption : 
  ∀ (total_soda_liters : ℝ) (days : ℝ) (ml_per_liter : ℝ), 
    total_soda_liters = 2 → 
    days = 4 → 
    ml_per_liter = 1000 → 
    (total_soda_liters * ml_per_liter) / days = 500 :=
by
  intros total_soda_liters days ml_per_liter h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_soda_consumption_l714_714037


namespace employee_Y_payment_l714_714612

theorem employee_Y_payment :
  ∀ (X Y Z : ℝ),
  (X + Y + Z = 900) →
  (X = 1.2 * Y) →
  (Z = (X + Y) / 2) →
  (Y = 272.73) :=
begin
  intros X Y Z h1 h2 h3,
  have h4 : 3.3 * Y = 900,
  { rw [←h2, ←h3] at h1,
    linarith,},
  exact (div_eq_iff (by norm_num : 3.3 ≠ 0)).mp h4,
end

end employee_Y_payment_l714_714612


namespace ceil_neg_sqrt_l714_714748

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l714_714748


namespace sin_theta_plus_2cos_theta_l714_714460

theorem sin_theta_plus_2cos_theta (a θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : sin (2 * θ) = a) :
  sin θ + 2 * cos θ = sqrt (5 - a) :=
sorry

end sin_theta_plus_2cos_theta_l714_714460


namespace x_value_is_22_l714_714413

-- Define the given conditions
variable (x : ℕ) -- x is a natural number
variable (data : List ℕ) -- data is a list of natural numbers
variable (median_value : ℕ) -- median_value is a natural number

-- Set the specific values for the problem
def data := [14, 19, x, 23, 27]
def median_value := 22

-- Define a function to find the median of a list
noncomputable def median (l : List ℕ) : ℕ :=
  if h : l.length % 2 = 1 then
    l.nth_le (l.length / 2) (by rw List.length_div_two_pos l h)
  else
    (l.nth_le (l.length / 2) (by rw List.length_div_two_pos l (ne_of_lt (by apply l.length_div_two_pos))).val +
     l.nth_le (l.length / 2 - 1) (by rw List.length_div_two_pos l (ne_of_lt (by apply l.length_div_two_pos))).val) / 2

-- The theorem to be proven
theorem x_value_is_22 : median data = median_value → x = 22 :=
by
  intro h_median
  sorry

end x_value_is_22_l714_714413


namespace problem_1_problem_2_problem_3_l714_714807

section

variables {x x1 x2 a : ℝ}
noncomputable def f (x : ℝ) := real.log x
noncomputable def g (x : ℝ) : ℝ := f (x + 1) - a * x
noncomputable def x0 (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def kAB (x1 x2 : ℝ) : ℝ := (f x2 - f x1) / (x2 - x1)
noncomputable def f_prime (x : ℝ) := (1 / x)

-- Problem 1: Monotonic intervals of g(x)
theorem problem_1 (h1 : a > 0) (h2 : a <= 0) (h3 : x > -1) :
  (forall x, a > 0 -> (g'(x) > 0 <-> x ∈ Ioc (-1, 1/a - 1)) /\ (g'(x) < 0 <-> x > 1/a - 1)) /\
  (forall x, a <= 0 -> g'(x) >= 0) := sorry

-- Problem 2: Compare slope kAB with f′(x0)
theorem problem_2 (h : 0 < x1) (h' : x1 < x2) :
  kAB x1 x2 > f_prime (x0 x1 x2) := sorry

-- Problem 3: Prove the inequality for x > 1
theorem problem_3 (h : x > 1) :
  (real.exp x / (x + 1) > (x - 1) / real.log x) := sorry

end

end problem_1_problem_2_problem_3_l714_714807


namespace area_of_triangle_l714_714906

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l714_714906


namespace oblique_asymptote_eq_l714_714629

-- Define the function
noncomputable def rational_function (x : ℝ) := (3 * x^3 + 4 * x^2 + 9 * x + 7) / (3 * x + 1)

-- Define the oblique asymptote
noncomputable def oblique_asymptote (x : ℝ) := x^2 + (1 / 3) * x + 3

-- Theorem stating the oblique asymptote of the given rational function
theorem oblique_asymptote_eq : ∀ x : ℝ, ∃ L : ℝ, ∀ ε > 0, ∃ N : ℝ, x > N → | (rational_function x) - (oblique_asymptote x) | < ε :=
sorry

end oblique_asymptote_eq_l714_714629


namespace tissue_pallets_ratio_l714_714339

-- Define the total number of pallets received
def total_pallets : ℕ := 20

-- Define the number of pallets of each type
def paper_towels_pallets : ℕ := total_pallets / 2
def paper_plates_pallets : ℕ := total_pallets / 5
def paper_cups_pallets : ℕ := 1

-- Calculate the number of pallets of tissues
def tissues_pallets : ℕ := total_pallets - (paper_towels_pallets + paper_plates_pallets + paper_cups_pallets)

-- Prove the ratio of pallets of tissues to total pallets is 1/4
theorem tissue_pallets_ratio : (tissues_pallets : ℚ) / total_pallets = 1 / 4 :=
by
  -- Proof goes here
  sorry

end tissue_pallets_ratio_l714_714339


namespace bridge_height_at_distance_l714_714666

theorem bridge_height_at_distance :
  (∃ (a : ℝ), ∀ (x : ℝ), (x = 25) → (a * x^2 + 25 = 0)) →
  (∀ (x : ℝ), (x = 10) → (-1/25 * x^2 + 25 = 21)) :=
by
  intro h1
  intro x h2
  have h : 625 * (-1 / 25) * (-1 / 25) = -25 := sorry
  sorry

end bridge_height_at_distance_l714_714666


namespace minimum_value_of_fraction_l714_714156

theorem minimum_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (4 / a + 9 / b) ≥ 25 :=
by
  sorry

end minimum_value_of_fraction_l714_714156


namespace suff_but_not_nec_condition_l714_714294

noncomputable def f (x : ℝ) (p : ℝ) : ℝ := x + p / x

theorem suff_but_not_nec_condition (p : ℝ) :
  (0 ≤ p ∧ p ≤ 4) → ∀ x > 2, (f x p)' > 0 :=
sorry

end suff_but_not_nec_condition_l714_714294


namespace simplify_trigonometric_expression_l714_714966

theorem simplify_trigonometric_expression (x : ℝ) (hx : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 / sin x := 
by sorry

end simplify_trigonometric_expression_l714_714966


namespace percent_democrats_voting_for_A_l714_714860

theorem percent_democrats_voting_for_A :
  ∀ (V : ℝ), (60 / 100) * V * D + (20 / 100) * (40 / 100) * V = (50 / 100) * V → 
  D = 7 / 10 :=
by
  intros V cond
  have h1 : (20 / 100) * (40 / 100) * V = 8 / 100 * V := by
    ring
  have h2 : (60 / 100) * V * D + 8 / 100 * V = (50 / 100) * V := by
    rw h1
    exact cond
  have h3 : (60 / 100) * V * D = (50 / 100) * V - 8 / 100 * V := by
    linarith
  have h4 : (60 / 100) * V * D = 42 / 100 * V := by
    linarith
  have h5 : (60 / 100) * D = 42 / 100 := by
    fact_nomega
  have h6 : D = 42 / 100 / (60 / 100) := by
    ring_nf
  have h7 : D = 7 / 10 := by
    syntax



end percent_democrats_voting_for_A_l714_714860


namespace arithmetic_sequence_general_formula_sequence_sum_l714_714101

theorem arithmetic_sequence_general_formula (d : ℤ) (a : ℕ → ℤ) (h₀ : d = 1) 
    (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
    (h₂ : a 3^2 = a 1 * a 4) : 
    ∀ n, a n = n - 5 :=
begin
  sorry
end

theorem sequence_sum (a : ℕ → ℤ) (b : ℕ → ℤ) (h₀ : ∀ n, a n = n - 5) 
    (h₁ : ∀ n, b n = 2^(a n + 5) + n) :
    ∀ n, (finset.range n).sum b = 2^(n + 1) - 2 + n * (n + 1) / 2 :=
begin
  sorry
end

end arithmetic_sequence_general_formula_sequence_sum_l714_714101


namespace cos_120_eq_neg_half_l714_714026

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714026


namespace concur_PS_QR_CI_l714_714212

variables {α : Type*} [euclidean_space α]

open euclidean_space

-- Variables representing the points and elements in the problem
variables (A B C P Q R S I : point α)
variable (k : circle α)

-- Hypothesis and conditions
hypothesis hI : is_incenter I (triangle.mk A B C)
hypothesis hk : k ∈ circle_passing_through A ∧ k ∈ circle_passing_through B
hypothesis hAP : ∃ AP, is_line_through AP A ∧ is_line_through AP I ∧ k.intersects AP (some_point_in k A I)
hypothesis hBQ : ∃ BQ, is_line_through BQ B ∧ is_line_through BQ I ∧ k.intersects BQ (some_point_in k B I)
hypothesis hAR : ∃ AR, is_line_through AR A ∧ is_line_through AR C ∧ k.intersects AR (some_point_in k A C)
hypothesis hBS : ∃ BS, is_line_through BS B ∧ is_line_through BS C ∧ k.intersects BS (some_point_in k B C)
hypothesis hDistinct : distinct_points [A, B, P, Q, R, S]

-- Additional geometric constraints
hypothesis hR : lies_on_segment R A C
hypothesis hS : lies_on_segment S B C

-- Theorem to be proved
theorem concur_PS_QR_CI :
  ∃ X, concurrent (line_through P S) (line_through Q R) (line_through C I) :=
begin
  sorry
end

end concur_PS_QR_CI_l714_714212


namespace num_Q_polynomials_l714_714897

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5)

#check Exists

theorem num_Q_polynomials :
  ∃ (Q : Polynomial ℝ), 
  (∃ (R : Polynomial ℝ), R.degree = 3 ∧ P (Q.eval x) = P x * R.eval x) ∧
  Q.degree = 2 ∧ (Q.coeff 1 = 6) ∧ (∃ (n : ℕ), n = 22) :=
sorry

end num_Q_polynomials_l714_714897


namespace sum_of_areas_of_squares_l714_714610

theorem sum_of_areas_of_squares (a b x : ℕ) 
  (h_overlapping_min : 9 ≤ (min a b) ^ 2)
  (h_overlapping_max : (min a b) ^ 2 ≤ 25)
  (h_sum_of_sides : a + b + x = 23) :
  a^2 + b^2 + x^2 = 189 := 
sorry

end sum_of_areas_of_squares_l714_714610


namespace jackson_eats_pizza_fraction_l714_714518

theorem jackson_eats_pizza_fraction :
  let calories_lettuce := 50
  let calories_carrots := 2 * calories_lettuce
  let calories_dressing := 210
  let calories_salad := calories_lettuce + calories_carrots + calories_dressing
  let calories_crust := 600
  let calories_pepperoni := (1 / 3) * calories_crust
  let calories_cheese := 400
  let calories_pizza := calories_crust + calories_pepperoni + calories_cheese
  let jackson_calories := 1 / 4 * calories_salad + (some_fraction * calories_pizza)
  total_calories := 330
  in jackson_calories = total_calories → some_fraction = 1 / 5 :=
by
  sorry

end jackson_eats_pizza_fraction_l714_714518


namespace quadratic_interval_solution_l714_714072

open Set

def quadratic_function (x : ℝ) : ℝ := x^2 + 5 * x + 6

theorem quadratic_interval_solution :
  {x : ℝ | 6 ≤ quadratic_function x ∧ quadratic_function x ≤ 12} = {x | -6 ≤ x ∧ x ≤ -5} ∪ {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_interval_solution_l714_714072


namespace complex_number_equality_l714_714402

noncomputable def imaginaryUnit := complex.I

theorem complex_number_equality (a b : ℝ) (h : (a + 2 * imaginaryUnit) / imaginaryUnit = b + imaginaryUnit) : a + b = 1 := by
  sorry

end complex_number_equality_l714_714402


namespace frog_distribution_l714_714623

theorem frog_distribution (n : ℕ) (n_pos : n > 0) :
  ∃ t : ℕ, ∀ part ∈ parts, 
    (frogs_in_part part t > 0) ∨ (∀ neighbor ∈ neighbors part, frogs_in_part neighbor t > 0) :=
sorry

end frog_distribution_l714_714623


namespace comparison_l714_714694

def sequence (α : ℕ → ℕ) : ℕ → ℚ
| 1     := 1 + 1 / (α 1)
| (n+1) := 1 + 1 / (α 1 + sequence (λ k, α (k+1) n))

theorem comparison (α : ℕ → ℕ) (h : ∀ k, 1 ≤ α k) :
  sequence α 4 < sequence α 7 := 
sorry

end comparison_l714_714694


namespace binomial_parameters_correct_l714_714412

open Probability

variables {n : ℕ} {p : ℝ} (ξ : binomial_distribution n p)
noncomputable def binomial_params :=
  E ξ = 2.4 ∧ variance ξ = 1.44 → (n = 6 ∧ p = 0.4)

theorem binomial_parameters_correct (ξ : binomial_distribution n p) :
  binomial_params ξ :=
by
  sorry

end binomial_parameters_correct_l714_714412


namespace remainder_4351_div_101_l714_714388

-- Given conditions
def G : ℕ := 101
def n1 : ℕ := 4351
def n2 : ℕ := 5161
def R2 : ℕ := 10

-- Correct answer to be proved
theorem remainder_4351_div_101 : n1 % G = 8 :=
by
  have R1 : ℕ := n1 % G
  have H1 : n2 % G = R2 := by sorry
  have H2 : G = 101 := by rfl
  rw H2 at H1
  simp at H1
  exact H1

end remainder_4351_div_101_l714_714388


namespace find_a_l714_714130

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 2^x + 1 else Real.log2 x + a

theorem find_a (a : ℝ) (h : f (f 0 a) a = 3 * a) : a = 1 / 2 := by
  sorry

end find_a_l714_714130


namespace equilateral_triangle_of_dot_products_equal_l714_714185

variables (a b c : ℝ³)

def vectors_sum_zero (a b c : ℝ³) : Prop := a + b + c = 0

def dot_product_equal (a b c : ℝ³) : Prop := a • b = b • c ∧ b • c = c • a

theorem equilateral_triangle_of_dot_products_equal
  (h1 : vectors_sum_zero a b c)
  (h2 : dot_product_equal a b c) :
  (|a| = |b| ∧ |b| = |c|) :=
by
  sorry

end equilateral_triangle_of_dot_products_equal_l714_714185


namespace div_by_30_l714_714049

theorem div_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end div_by_30_l714_714049


namespace polynomial_identity_holds_l714_714043

theorem polynomial_identity_holds :
  ∀ x : ℝ, ∃ (a b c : ℝ), x^3 - a * x^2 + b * x - c = (x - a) * (x - b) * (x - c) ↔ (a = -1 ∧ b = -1 ∧ c = 1) :=
by
  intro x
  use [-1, -1, 1]
  apply Iff.intro
  .intro hypothesis sorry -- Proof that identity holds
  .intro hypothesis sorry -- Proof that if identity holds, then a = -1, b = -1, c = 1

end polynomial_identity_holds_l714_714043


namespace simplify_trigonometric_expression_l714_714965

theorem simplify_trigonometric_expression (x : ℝ) (hx : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 / sin x := 
by sorry

end simplify_trigonometric_expression_l714_714965


namespace AI_eq_IQ_l714_714545

noncomputable def point := sorry

variables (A B C I D P Q : point) (circumcircle : set point)
variables [triangleABC : triangle A B C]
variable [incircleI : incenter I]
variable [midpointD : midpoint_arc D (C B)]
variable [reflectP : reflection P I (side BC)]
variable [meetQ : line_meeting DP circumcircle Q]
variable [arcAB : minor_arc_arc_AB Q]

theorem AI_eq_IQ : dist A I = dist I Q :=
by sorry

end AI_eq_IQ_l714_714545


namespace mary_characters_initial_D_l714_714928

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l714_714928


namespace area_of_ADE_l714_714512

-- Define the initial conditions
def AB : ℝ := 7
def BC : ℝ := 8
def AC : ℝ := 9
def AD : ℝ := 3
def AE : ℝ := 6

-- The theorem to prove
theorem area_of_ADE : 
  let s := (AB + BC + AC) / 2 in
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC)) in
  let sinA := (area_ABC * 2) / (AB * AC) in
  let area_ADE := (1 / 2) * AD * AE * sinA in
  area_ADE = (24 * Real.sqrt 5) / 7 :=
by
  sorry

end area_of_ADE_l714_714512


namespace logarithmic_ratio_l714_714154

theorem logarithmic_ratio (m n : ℝ) (h1 : Real.log 2 = m) (h2 : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2 * m + n) / (1 - m + n) := 
sorry

end logarithmic_ratio_l714_714154


namespace x_intercept_of_line_l714_714756

theorem x_intercept_of_line :
  (∃ x : ℝ, 5 * x - 7 * 0 = 35 ∧ (x, 0) = (7, 0)) :=
by
  use 7
  simp
  sorry

end x_intercept_of_line_l714_714756


namespace three_tenths_of_number_l714_714303

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 15) : (3/10) * x = 54 :=
by
  sorry

end three_tenths_of_number_l714_714303


namespace cauliflower_sales_l714_714929

noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := 16 + (1 / 2 * carrot_sales)
noncomputable def total_sales : ℝ := 380
noncomputable def other_sales : ℝ := broccoli_sales + carrot_sales + spinach_sales

theorem cauliflower_sales :
  total_sales - other_sales = 136 :=
by
  -- proof skipped
  sorry

end cauliflower_sales_l714_714929


namespace exists_separating_polynomial_l714_714793

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (color : Bool) -- True represents red, False represents blue

def separates (P : ℝ → ℝ) (points : List Point) : Prop :=
  (∀ p ∈ points, p.color → p.y ≤ P p.x) ∨
  (∀ p ∈ points, ¬p.color → p.y ≥ P p.x)

theorem exists_separating_polynomial (N : ℕ) (hN : N ≥ 3) (points : List Point) (hDistinct : points.map (·.x).nodup) :
  ∃ P : ℝ → ℝ, ∃ k ≤ N - 2, (∃ a : ℝ → ℝ, polynomial.degree a ≤ k) ∧
  separates P points :=
begin
  sorry
end

end exists_separating_polynomial_l714_714793


namespace maximize_profit_l714_714660

noncomputable def annual_profit : ℝ → ℝ
| x => if x < 80 then - (1/3) * x^2 + 40 * x - 250 
       else 1200 - (x + 10000 / x)

theorem maximize_profit : ∃ x : ℝ, x = 100 ∧ annual_profit x = 1000 :=
by
  sorry

end maximize_profit_l714_714660


namespace switches_in_A_position_l714_714167

def switch_label (x y z : Nat) : Nat := 2^x * 3^y * 7^z

def toggle_count (x y z : Nat) : Nat := (9 - x) * (9 - y) * (9 - z)

def is_position_A (x y z : Nat) : Bool := (toggle_count x y z) % 4 = 0

def total_switches_A : Nat := 
  let all_cases : List (Nat × Nat × Nat) := 
    List.product (List.product (List.range 9) (List.range 9)) (List.range 9)
  all_cases.count (λ ⟨⟨x, y⟩, z⟩ => is_position_A x y z)

theorem switches_in_A_position : total_switches_A = 409 := by
  sorry

end switches_in_A_position_l714_714167


namespace flour_needed_l714_714335

theorem flour_needed (flour_per_24_cookies : ℝ) (cookies_per_recipe : ℕ) (desired_cookies : ℕ) 
  (h : flour_per_24_cookies = 1.5) (h1 : cookies_per_recipe = 24) (h2 : desired_cookies = 72) : 
  flour_per_24_cookies / cookies_per_recipe * desired_cookies = 4.5 := 
  by {
    -- The proof is omitted
    sorry
  }

end flour_needed_l714_714335


namespace sum_of_90_degree_angles_l714_714358

def number_of_90_degree_angles_rectangular_park : ℕ := 4
def number_of_90_degree_angles_square_field : ℕ := 4

theorem sum_of_90_degree_angles : 
  number_of_90_degree_angles_rectangular_park + number_of_90_degree_angles_square_field = 8 := 
by canonically sorry

end sum_of_90_degree_angles_l714_714358


namespace volume_ratio_l714_714195

open Real

noncomputable def Vx (a b : ℝ) : ℝ :=
  π * ∫ x in 0..b, ((-a * x^2 + 2 * a * b * x)^2 - (a * x^2)^2)

noncomputable def Vy (a b : ℝ) : ℝ :=
  2 * π * ∫ x in 0..b, x * (-2 * a * x^3 + 2 * a * b * x)

theorem volume_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Vx a b) / (Vy a b) = 14 / 5 :=
by
  sorry

end volume_ratio_l714_714195


namespace color_copies_comparison_l714_714775

theorem color_copies_comparison :
  ∃ n : ℤ, (1.25 * n) + 60 = (2.75 * n) ∧ n = 40 :=
by
  use 40
  refine ⟨_, rfl⟩
  sorry

end color_copies_comparison_l714_714775


namespace no_solution_for_k_eq_9_l714_714399

theorem no_solution_for_k_eq_9 (k : ℝ) (x : ℝ) : k = 9 → (x ≠ 1 ∧ x ≠ 7) → ¬(x - 3) / (x - 1) = (x - k) / (x - 7) :=
by
  intros k_eq_9 x_domain
  rw k_eq_9
  sorry

end no_solution_for_k_eq_9_l714_714399


namespace fifth_inequality_l714_714941

theorem fifth_inequality :
  (1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2))
  < (11 / 6) :=
begin
  let ineq1 := 1 + (1 / 2^2) < (3 / 2),
  let ineq2 := 1 + (1 / 2^2) + (1 / 3^2) < (5 / 3),
  let ineq3 := 1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) < (7 / 4),
  sorry
end

end fifth_inequality_l714_714941


namespace find_p_from_parabola_and_distance_l714_714829

theorem find_p_from_parabola_and_distance 
  (p : ℝ) (hp : p > 0) 
  (M : ℝ × ℝ) (hM : M = (8 / p, 4))
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hMF : dist M F = 4) : 
  p = 4 :=
sorry

end find_p_from_parabola_and_distance_l714_714829


namespace simplify_trigonometric_expression_l714_714967

theorem simplify_trigonometric_expression (x : ℝ) (hx : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 / sin x := 
by sorry

end simplify_trigonometric_expression_l714_714967


namespace alpha_value_l714_714434

theorem alpha_value (α : ℝ) (h : (α * (α - 1) * (-1 : ℝ)^(α - 2)) = 4) : α = -4 :=
by
  sorry

end alpha_value_l714_714434


namespace renata_final_money_is_correct_l714_714226

def exchange_rate_pound_to_dollar : ℝ := 1.35
def exchange_rate_euro_to_dollar : ℝ := 1.10

def initial_money : ℝ := 50
def donation : ℝ := 10
def prize_pounds : ℝ := 50
def loss_euro_first_slot : ℝ := 30
def loss_pounds_second_slot : ℝ := 20
def loss_dollars_last_slot : ℝ := 15
def sunglasses_original_price_euro : ℝ := 15
def sunglasses_discount_rate : ℝ := 0.20
def water_price_pounds : ℝ := 1
def lottery_ticket_price : ℝ := 1
def lottery_win : ℝ := 30

def meal_original_price_euro : ℝ := 10
def meal_discount_rate : ℝ := 0.30
def coffee_price_euro : ℝ := 3
def total_meals : ℕ := 2

def final_money (initial_money donation prize_pounds loss_euro_first_slot loss_pounds_second_slot loss_dollars_last_slot sunglasses_original_price_euro sunglasses_discount_rate water_price_pounds lottery_ticket_price lottery_win meal_original_price_euro meal_discount_rate coffee_price_euro total_meals : ℝ) : ℝ :=
(let money_after_donation := initial_money - donation in
 let prize_dollars := prize_pounds * exchange_rate_pound_to_dollar in
 let money_after_prize := money_after_donation + prize_dollars in
 let loss_first_slot_dollars := loss_euro_first_slot * exchange_rate_euro_to_dollar in
 let money_after_first_slot := money_after_prize - loss_first_slot_dollars in
 let loss_second_slot_dollars := loss_pounds_second_slot * exchange_rate_pound_to_dollar in
 let money_after_second_slot := money_after_first_slot - loss_second_slot_dollars in
 let money_after_last_slot := money_after_second_slot - loss_dollars_last_slot in
 let sunglasses_discount := sunglasses_original_price_euro * sunglasses_discount_rate in
 let sunglasses_discounted_price_euro := sunglasses_original_price_euro - sunglasses_discount in
 let sunglasses_price_dollars := sunglasses_discounted_price_euro * exchange_rate_euro_to_dollar in
 let money_after_sunglasses := money_after_last_slot - sunglasses_price_dollars in
 let water_price_dollars := water_price_pounds * exchange_rate_pound_to_dollar in
 let money_after_water := money_after_sunglasses - water_price_dollars in
 let money_after_lottery_ticket := money_after_water - lottery_ticket_price in
 let money_after_lottery_win := money_after_lottery_ticket + lottery_win in
 let meal_discount := meal_original_price_euro * meal_discount_rate in
 let meal_discounted_price_euro := meal_original_price_euro - meal_discount in
 let total_meal_cost_euro := meal_discounted_price_euro * total_meals in
 let total_coffee_cost_euro := coffee_price_euro * total_meals in
 let total_lunch_cost_euro := total_meal_cost_euro + total_coffee_cost_euro in
 let total_lunch_cost_dollars := total_lunch_cost_euro * exchange_rate_euro_to_dollar in
 let renata_share := total_lunch_cost_dollars / 2 in
 let remaining_money := money_after_lottery_win - renata_share in
 remaining_money)

theorem renata_final_money_is_correct : final_money initial_money donation prize_pounds loss_euro_first_slot loss_pounds_second_slot loss_dollars_last_slot sunglasses_original_price_euro sunglasses_discount_rate water_price_pounds lottery_ticket_price lottery_win meal_original_price_euro meal_discount_rate coffee_price_euro total_meals = 35.95 :=
by
  -- calculation steps
  sorry

end renata_final_money_is_correct_l714_714226


namespace find_fx_when_x_gt_0_l714_714790

variable (f : ℝ → ℝ)

-- Condition: f is odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition: when x < 0, f(x) = 1 - e^(-x + 1)
def condition_neg_x (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = 1 - exp (-x + 1)

-- Theorem statement: when x > 0, f(x) = e^(x + 1) - 1
theorem find_fx_when_x_gt_0 (h_odd : odd_function f) (h_cond_neg_x : condition_neg_x f) :
  ∀ x, x > 0 → f x = exp (x + 1) - 1 :=
sorry

end find_fx_when_x_gt_0_l714_714790


namespace conjugate_of_z_l714_714915

variable (i : ℂ)
local notation "z" := (2 + i) / (1 + i ^ 2 + i ^ 5)

theorem conjugate_of_z (hi2: i^2 = -1) (hi5: i^5 = i) : conj z = 1 + 2 * i := 
by
  sorry

end conjugate_of_z_l714_714915


namespace algebra_sum_l714_714587

def letter_value : ℕ → ℤ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 1
| 5 := 0
| 6 := -1
| 7 := -2
| 8 := -3
| 9 := -1
| 10 := 0
| 11 := 1
| 12 := 2
| (n + 13) := letter_value (n % 12 + 1)

def letter_position (c : Char) : ℕ :=
if h : (c = 'a') then 1
else if h : (c = 'b') then 2
else if h : (c = 'c') then 3
else if h : (c = 'd') then 4
else if h : (c = 'e') then 5
else if h : (c = 'f') then 6
else if h : (c = 'g') then 7
else if h : (c = 'h') then 8
else if h : (c = 'i') then 9
else if h : (c = 'j') then 10
else if h : (c = 'k') then 11
else if h : (c = 'l') then 12
else if h : (c = 'm') then 13
else if h : (c = 'n') then 14
else if h : (c = 'o') then 15
else if h : (c = 'p') then 16
else if h : (c = 'q') then 17
else if h : (c = 'r') then 18
else if h : (c = 's') then 19
else if h : (c = 't') then 20
else if h : (c = 'u') then 21
else if h : (c = 'v') then 22
else if h : (c = 'w') then 23
else if h : (c = 'x') then 24
else if h : (c = 'y') then 25
else 26

def sum_of_values (s : String) : ℤ :=
s.data.foldl (λ acc c, acc + letter_value (letter_position c)) 0

theorem algebra_sum : sum_of_values "algebra" = 1 := 
by 
  sorry -- Proof omitted

end algebra_sum_l714_714587


namespace find_m_range_l714_714824

theorem find_m_range (m : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Icc 0 2, f x = 0 → True) :
  (∀ x, f x = x^2 + (m-1)*x + 1) → ∃ x, x ∈ Icc (0 : ℝ) (2 : ℝ) ∧ f x = 0 → m ≤ -1 :=
by
  intros h1 h2
  sorry

end find_m_range_l714_714824


namespace runner_a_33_seconds_l714_714858

noncomputable def runner_a_time (distance : ℝ) (incline : ℝ) (hurdles : ℕ) (distance_diff : ℝ) (time_diff : ℝ) : ℝ :=
  let speed_diff := distance_diff / time_diff in
  let t_b := distance / speed_diff in
  t_b - time_diff

theorem runner_a_33_seconds 
  (distance : ℝ) (incline : ℝ) (hurdles : ℕ) (distance_diff : ℝ) (time_diff : ℝ)
  (h1 : distance = 200) (h2 : incline = 5 / 100) (h3 : hurdles = 3) (h4 : distance_diff = 35) (h5 : time_diff = 7) :
  runner_a_time distance incline hurdles distance_diff time_diff = 33 :=
by
  sorry

end runner_a_33_seconds_l714_714858


namespace function_characterization_l714_714387

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, f (⟨floor x⟩ * y) = f x * ⟨floor (f y)⟩) →
  (∀ x : ℝ, f x = 0) ∨ (∃ v : ℝ, v ∈ Ico 1 2 ∧ ∀ x : ℝ, f x = v) :=
begin
  sorry
end

end function_characterization_l714_714387


namespace percentage_increase_is_fifty_l714_714315

-- Define the initial and final numbers as constants.
def initial : ℕ := 100
def final : ℕ := 150

-- Define the function to calculate percentage increase.
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) / initial.toRat) * 100

-- The statement we want to prove.
theorem percentage_increase_is_fifty :
  percentage_increase initial final = 50 := 
by
  sorry

end percentage_increase_is_fifty_l714_714315


namespace length_of_each_movie_l714_714838

-- Defining the amount of time Grandpa Lou watched movies on Tuesday in minutes
def time_tuesday : ℕ := 4 * 60 + 30   -- 4 hours and 30 minutes

-- Defining the number of movies watched on Tuesday
def movies_tuesday (x : ℕ) : Prop := time_tuesday / x = 90

-- Defining the total number of movies watched in both days
def total_movies_two_days (x : ℕ) : Prop := x + 2 * x = 9

theorem length_of_each_movie (x : ℕ) (h₁ : total_movies_two_days x) (h₂ : movies_tuesday x) : time_tuesday / x = 90 :=
by
  -- Given the conditions, we can prove the statement:
  sorry

end length_of_each_movie_l714_714838


namespace purple_candy_minimum_cost_l714_714730

theorem purple_candy_minimum_cost (r g b n : ℕ) (h : 10 * r = 15 * g) (h1 : 15 * g = 18 * b) (h2 : 18 * b = 24 * n) : 
  ∃ k, k = n ∧ k ≥ 1 ∧ ∀ m, (24 * m = 360) → (m ≥ k) :=
by
  sorry

end purple_candy_minimum_cost_l714_714730


namespace percent_decrease_in_area_l714_714477

/-- Given a circle with an area of 25π square inches and a diameter that is decreased by 10%, show that the percent decrease in the area of the circle is 19% -/
theorem percent_decrease_in_area
  (initial_area_circle : ℝ) (initial_diameter_circle : ℝ) (decrease_percent : ℝ) :
  initial_area_circle = 25 * Real.pi →
  initial_diameter_circle = 2 * Real.sqrt (25 * Real.pi / Real.pi) →
  decrease_percent = 0.10 →
  let new_diameter_circle := initial_diameter_circle * (1 - decrease_percent) in
  let new_radius_circle := new_diameter_circle / 2 in
  let new_area_circle := Real.pi * (new_radius_circle ^ 2) in
  let percent_decrease := (initial_area_circle - new_area_circle) / initial_area_circle * 100 in
  percent_decrease = 19 := 
by
  sorry

end percent_decrease_in_area_l714_714477


namespace gunny_bag_capacity_l714_714228

def pounds_per_ton : ℝ := 2500
def ounces_per_pound : ℝ := 16
def packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

theorem gunny_bag_capacity :
  (packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound) / pounds_per_ton) = 13 := 
by
  sorry

end gunny_bag_capacity_l714_714228


namespace sum_of_perimeters_l714_714271

theorem sum_of_perimeters (x y : ℝ) (h₁ : x^2 + y^2 = 125) (h₂ : x^2 - y^2 = 65) : 4 * x + 4 * y = 60 := 
by
  sorry

end sum_of_perimeters_l714_714271


namespace degree_of_f_plus_c_g_is_3_l714_714719

-- Define the two polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 2 - 15 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℝ) : ℝ := 4 - 3 * x - 8 * x^3 + 12 * x^4

-- Define the statement to be proven
theorem degree_of_f_plus_c_g_is_3 (c : ℝ) (h : f x + c * g x is of degree 3) : c = -7 / 12 :=
by
  -- Lean proof to be filled here
  sorry

end degree_of_f_plus_c_g_is_3_l714_714719


namespace maximal_x2009_l714_714299

theorem maximal_x2009 (x : ℕ → ℝ) 
    (h_seq : ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0)
    (h_x0 : x 0 = 1)
    (h_x20 : x 20 = 9)
    (h_x200 : x 200 = 6) :
    x 2009 ≤ 6 :=
sorry

end maximal_x2009_l714_714299


namespace mary_overtime_percentage_increase_l714_714221

theorem mary_overtime_percentage_increase
  (max_hours : ℕ)
  (regular_hours : ℕ)
  (regular_rate : ℕ)
  (max_earnings : ℕ)
  (overtime_rate : ℕ)
  (percentage_increase : ℕ) :
  max_hours = 80 →
  regular_hours = 20 →
  regular_rate = 8 →
  max_earnings = 760 →
  (let regular_earnings := regular_hours * regular_rate in
  let overtime_earnings := max_earnings - regular_earnings in
  let overtime_hours := max_hours - regular_hours in
  let calculated_overtime_rate := overtime_earnings / overtime_hours in
  let calculated_percentage_increase := ((calculated_overtime_rate - regular_rate) * 100) / regular_rate in
  calculated_percentage_increase = 25)
  :=
sorry

end mary_overtime_percentage_increase_l714_714221


namespace net_gain_A_correct_l714_714224

-- Define initial values and transactions
def initial_cash_A : ℕ := 20000
def house_value : ℕ := 20000
def car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000
def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500
def house_repurchase_price : ℕ := 19000
def car_depreciation : ℕ := 10
def car_repurchase_price : ℕ := 4050

-- Define the final cash calculations
def final_cash_A := initial_cash_A + house_sale_price + car_sale_price - house_repurchase_price - car_repurchase_price
def final_cash_B := initial_cash_B - house_sale_price - car_sale_price + house_repurchase_price + car_repurchase_price

-- Define the net gain calculations
def net_gain_A := final_cash_A - initial_cash_A
def net_gain_B := final_cash_B - initial_cash_B

-- Theorem to prove
theorem net_gain_A_correct : net_gain_A = 2000 :=
by 
  -- Definitions and calculations would go here
  sorry

end net_gain_A_correct_l714_714224


namespace quadratic_range_of_a_l714_714598

theorem quadratic_range_of_a {f : ℝ → ℝ} (h1 : ∀ x, f(x) = f(4 - x)) (h2 : ∃ a b c, f(x) = a*x^2 + b*x + c ∧ 0 < a)
    (a : ℝ) : f(2 - a^2) < f(1 + a - a^2) ↔ a < 1 := 
by
  sorry

end quadratic_range_of_a_l714_714598


namespace sum_of_squares_fw_bw_l714_714230

theorem sum_of_squares_fw_bw (n : ℕ) : 
  (∑ i in range (n+1), i^2) + (∑ i in range n, i^2) = (1/3 : ℚ) * n * (2 * n^2 + 1) :=
sorry

end sum_of_squares_fw_bw_l714_714230


namespace areaEnclosedByMidpoints_l714_714987

namespace MathProof

noncomputable def lineSegmentLengthEqTwo (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem areaEnclosedByMidpoints : 
  (let k := 4 - π in 100 * k = 86) :=
by 
  let k := 4 - π
  have h1 : 100 * k = 100 * (4 - π) := rfl
  have h2 : 100 * (4 - π) = 86 := sorry
  exact h2

end MathProof

end areaEnclosedByMidpoints_l714_714987


namespace constant_term_expansion_l714_714041

def f (x : ℝ) : ℝ := x^3 + x^2 + 3
def g (x : ℝ) : ℝ := 2x^4 + x^2 + 7

theorem constant_term_expansion : f 0 * g 0 = 21 :=
by
  simp [f, g]
  norm_num
  sorry

end constant_term_expansion_l714_714041


namespace tan_sum_to_expression_l714_714085

theorem tan_sum_to_expression (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 :=
by 
  sorry

end tan_sum_to_expression_l714_714085


namespace min_max_distance_sums_l714_714880

noncomputable def coords_A : ℝ × ℝ := (2 * Real.sqrt 3, 0)
noncomputable def coords_B : ℝ × ℝ := (0, 2)
noncomputable def center_C : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def radius_C : ℝ := 2
noncomputable def parametric_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem min_max_distance_sums (θ : ℝ) :
  let P := parametric_C θ in
  let PO := (P.1 - 0) ^ 2 + (P.2 - 0) ^ 2 in
  let PA := (P.1 - (2 * Real.sqrt 3)) ^ 2 + (P.2 - 0) ^ 2 in
  let PB := (P.1 - 0) ^ 2 + (P.2 - 2) ^ 2 in
  16 ≤ PO + PA + PB ∧ PO + PA + PB ≤ 32 :=
by
  sorry

end min_max_distance_sums_l714_714880


namespace point_lies_on_line_l714_714162

-- Define the linear function y = 2x - b
def linearFunction (x : ℝ) (b : ℝ) : ℝ := 2 * x - b

-- Define the point through which the line passes
def point_on_line : (ℝ × ℝ) := (0, -3)

-- The correct answer to be proven
def correct_point : (ℝ × ℝ) := (2, 1)

-- Define the property that a point lies on the line
def lies_on_line (p : ℝ × ℝ) (b : ℝ) : Prop :=
  let (x, y) := p in y = linearFunction x b

-- The main theorem to prove
theorem point_lies_on_line :
  (∃ b : ℝ, point_on_line.2 = linearFunction point_on_line.1 b) →
  lies_on_line correct_point 3 :=
by
  sorry

end point_lies_on_line_l714_714162


namespace proof_50th_permutation_l714_714590

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def nth_permutation (l : List ℕ) (n : ℕ) : List ℕ :=
  let f := factorial (l.length - 1)
  if l = [] then [] else
  let (q, r) := n.div_mod f
  let x := l.nth_le q ((Nat.lt_of_sub_lt_sub (Nat.zero_lt_succ _)).1 (Nat.pos_iff_ne_zero.2 (factorial_ne_zero (l.length - 1))))
  x::nth_permutation (List.erase l x) r

theorem proof_50th_permutation : (nth_permutation [1, 2, 3, 4, 5] 49) = [3, 1, 2, 5, 4] :=
  by sorry

end proof_50th_permutation_l714_714590


namespace area_of_closed_figure_l714_714757

-- Entity definitions
def parabola_eq (x : ℝ) (y : ℝ) : Prop := (x^2 = 4 * y)
def line_eq (x : ℝ) (y : ℝ) : Prop := (y = 2 * x)

-- The integral function
def area_between_parabola_and_line (a b : ℝ) : ℝ :=
  ∫ x ∂[interval_integral 0 8], (2 * x - (x^2 / 4))

-- Theorem stating the required proof problem
theorem area_of_closed_figure : 
  area_between_parabola_and_line 0 8 = (64 / 3) := 
by
  sorry

end area_of_closed_figure_l714_714757


namespace conference_room_seating_l714_714423

theorem conference_room_seating :
  let seats := 8
  let people := 3
  let empty_seats := seats - people
  the number_of_arrangements = (number_of_arrangements_distribute_empty_seats * number_of_arrangements_position_people) ∧
  number_of_arrangements_distribute_empty_seats = 4 ∧
  number_of_arrangements_position_people = 6
  → number_of_arrangements = 24 := 
sorry

end conference_room_seating_l714_714423


namespace ceil_neg_sqrt_l714_714747

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l714_714747


namespace q_range_l714_714534

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n = 1 then 1
  else (Nat.find_greatest (λ p => is_prime p ∧ p ∣ n) (λ m => is_prime m ∧ m ≤ n))

def q (x : ℝ) : ℝ :=
  if is_prime (Int.floor x) ∧ Int.floor x % 2 = 1 then x + 2
  else q (greatest_prime_factor (Int.floor x)) + (x + 2 - Int.floor x)

theorem q_range : set.Icc 3 15 ⊆ set.range q ∧ set.range q = set.Ico 5 10 ∪ set.Ico 13 14 :=
sorry

end q_range_l714_714534


namespace cubic_root_59319_digits_cubic_root_59319_unit_cubic_root_59319_tens_digit_cube_root_6859_cube_root_19683_cube_root_110592_l714_714188

noncomputable def cube_root_digits (n : ℕ) : ℕ :=
  if n < 1000000 then 
    if n < 1000 then 1
    else 2
  else
    3

theorem cubic_root_59319_digits : cube_root_digits 59319 = 2 :=
by
  sorry

theorem cubic_root_59319_unit : Nat.digits 10 59319 |> List.head = some 9 :=
by
  sorry

theorem cubic_root_59319_tens_digit : 
  let n := 59319 / 1000 in 
  Nat.digits 10 n |> List.head = some 3 :=
by
  sorry

theorem cube_root_6859 : ∀ n : ℕ, n ^ 3 = 6859 → n = 19 :=
by
  intros n hn
  sorry

theorem cube_root_19683 : ∀ n : ℕ, n ^ 3 = 19683 → n = 27 :=
by
  intros n hn
  sorry

theorem cube_root_110592 : ∀ n : ℕ, n ^ 3 = 110592 → n = 48 :=
by
  intros n hn
  sorry

end cubic_root_59319_digits_cubic_root_59319_unit_cubic_root_59319_tens_digit_cube_root_6859_cube_root_19683_cube_root_110592_l714_714188


namespace integral_f_x_l714_714846

theorem integral_f_x (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ t in (0 : ℝ)..1, f t) : 
  ∫ t in (0 : ℝ)..1, f t = -1 / 3 := by
  sorry

end integral_f_x_l714_714846


namespace divisibility_by_six_l714_714076

theorem divisibility_by_six (a x: ℤ) : ∃ t: ℤ, x = 3 * t ∨ x = 3 * t - a^2 → 6 ∣ a * (x^3 + a^2 * x^2 + a^2 - 1) :=
by
  sorry

end divisibility_by_six_l714_714076


namespace evaluate_expression_l714_714377

theorem evaluate_expression :
  3000 * (3000 ^ 1500 + 3000 ^ 1500) = 2 * 3000 ^ 1501 :=
by sorry

end evaluate_expression_l714_714377


namespace monotonicity_and_extreme_values_range_of_m_l714_714826

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem monotonicity_and_extreme_values (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧ 
  (a > 0 → ∃ c : ℝ, c = Real.ln (Real.sqrt a) ∧ (∀ x : ℝ, x ≠ c → f c a ≤ f x a)) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ≥ 0, f x (-1) ≥ m * x) ↔ m ∈ Set.Iic 2 :=
sorry

end monotonicity_and_extreme_values_range_of_m_l714_714826


namespace power_rounding_l714_714645

-- Define the mathematical expression and the rounding operation
def approx_value : ℝ := (1.003 : ℝ) ^ 4

-- The property to be proven
theorem power_rounding : Real.round_to_3_decimals approx_value = 1.012 := by
  sorry

end power_rounding_l714_714645


namespace tractor_brigades_l714_714229
noncomputable def brigade_plowing : Prop :=
∃ x y : ℝ,
  x * y = 240 ∧
  (x + 3) * (y + 2) = 324 ∧
  x > 20 ∧
  (x + 3) > 20 ∧
  x = 24 ∧
  (x + 3) = 27

theorem tractor_brigades:
  brigade_plowing :=
sorry

end tractor_brigades_l714_714229


namespace range_of_m_l714_714087

theorem range_of_m {x m : ℝ} 
  (α : 2 / (x + 1) > 1) 
  (β : m ≤ x ∧ x ≤ 2) 
  (suff_condition : ∀ x, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) :
  m ≤ -1 :=
sorry

end range_of_m_l714_714087


namespace sum_of_y_values_l714_714205

def g (x : ℚ) : ℚ := 2 * x^2 - x + 3

theorem sum_of_y_values (y1 y2 : ℚ) (hy : g (4 * y1) = 10 ∧ g (4 * y2) = 10) :
  y1 + y2 = 1 / 16 :=
sorry

end sum_of_y_values_l714_714205


namespace lines_non_intersecting_l714_714945

-- Definitions
variables {α β : Type} [plane α] [plane β]
variables (A B C D : point)
variables (AB : line) (CD : line)

-- Conditions
def plane_parallel (α β : Type) [plane α] [plane β] : Prop := α ∥ β
def line_in_plane (AB : line) (α : Type) [plane α] : Prop := AB ⊆ α
def line_in_plane' (CD : line) (β : Type) [plane β] : Prop := CD ⊆ β

-- Problem statement
theorem lines_non_intersecting 
  (h1 : plane_parallel α β)
  (h2 : line_in_plane AB α)
  (h3 : line_in_plane' CD β) :
  non_intersecting_lines AB CD :=
sorry

end lines_non_intersecting_l714_714945


namespace probability_at_least_two_consecutive_heads_l714_714323

theorem probability_at_least_two_consecutive_heads :
  let Ω := (Fin 2 → Fin 2 → ℝ) in
  let P : MeasureTheory.Measure Ω := MeasureTheory.ProbabilityMeasure.uniform (MeasureTheory.fintypeFiniteOf Ω) in
  let event := {ω : Ω | ∃ i : Fin 4, ω i = 1 ∧ ω (i+1) = 1} in
  P event = 9/16 :=
begin
  sorry
end

end probability_at_least_two_consecutive_heads_l714_714323


namespace eccentricity_of_conic_section_l714_714219

variables {C : Type} {F1 F2 P : C}
variables (dPF1 dF1F2 dPF2 : ℝ)
variables (m : ℝ) (h : dPF1 = 4 * m ∧ dF1F2 = 3 *m ∧ dPF2 = 2 * m)

theorem eccentricity_of_conic_section (h1: dPF1 : dF1F2 : dPF2 = 4 : 3 : 2) :
  (∃ e : ℝ, e = 1/2 ∨ e = 3/2) :=
by {
  sorry
}

end eccentricity_of_conic_section_l714_714219


namespace problem_solution_l714_714112

open Real

def p : Prop := ∀ x : ℕ+, (1 / 2)^x ≥ (1 / 3)^x

def q : Prop := ∃ x : ℕ+, 2^x + 2^(1 - x) = 2 * sqrt 2

theorem problem_solution : p ∧ ¬q :=
by
  sorry

end problem_solution_l714_714112


namespace conjugate_of_z_is_neg_i_div_five_l714_714128

-- Define the complex number condition
def complex_condition (z : ℂ) : Prop :=
  1 / z = (-2 + complex.i) * (2 * complex.i - 1)

-- Prove that given the condition, the conjugate of z is -i/5
theorem conjugate_of_z_is_neg_i_div_five (z : ℂ) (h : complex_condition z) :
  complex.conj z = - complex.i / 5 :=
begin
  sorry
end

end conjugate_of_z_is_neg_i_div_five_l714_714128


namespace eighth_grade_probability_female_win_l714_714492

theorem eighth_grade_probability_female_win:
  let P_Alexandra : ℝ := 1 / 4,
      P_Alexander : ℝ := 3 / 4,
      P_Evgenia : ℝ := 1 / 4,
      P_Yevgeny : ℝ := 3 / 4,
      P_Valentina : ℝ := 2 / 5,
      P_Valentin : ℝ := 3 / 5,
      P_Vasilisa : ℝ := 1 / 50,
      P_Vasily : ℝ := 49 / 50 in
  let P_female : ℝ :=
    1 / 4 * (P_Alexandra + 
             P_Evgenia + 
             P_Valentina + 
             P_Vasilisa) in
  P_female = 1 / 16 + 1 / 48 + 3 / 20 + 1 / 200 :=
sorry

end eighth_grade_probability_female_win_l714_714492


namespace esther_commute_distance_l714_714050

theorem esther_commute_distance (D : ℝ) (h_morning_speed : D / 45) (h_evening_speed : D / 30)
    (h_total_time : (D / 45) + (D / 30) = 1) : D = 18 :=
by
  sorry

end esther_commute_distance_l714_714050


namespace percentage_gain_on_powerlifting_total_l714_714706

def initialTotal : ℝ := 2200
def initialWeight : ℝ := 245
def weightIncrease : ℝ := 8
def finalWeight : ℝ := initialWeight + weightIncrease
def liftingRatio : ℝ := 10
def finalTotal : ℝ := finalWeight * liftingRatio

theorem percentage_gain_on_powerlifting_total :
  ∃ (P : ℝ), initialTotal * (1 + P / 100) = finalTotal :=
by
  sorry

end percentage_gain_on_powerlifting_total_l714_714706


namespace total_checked_papers_is_correct_l714_714615

-- Definitions based on conditions
def papers_in_school_A : ℕ := 1260
def papers_in_school_B : ℕ := 720
def papers_in_school_C : ℕ := 900
def drawn_papers_school_C : ℕ := 45

-- Calculate proportion based on School C
def proportion_school_C : ℚ := drawn_papers_school_C / papers_in_school_C

-- Calculate checked papers in School A and B using the proportion
def drawn_papers_school_A : ℕ := (papers_in_school_A * proportion_school_C).natAbs
def drawn_papers_school_B : ℕ := (papers_in_school_B * proportion_school_C).natAbs

-- Total checked papers in the survey
def total_checked_papers : ℕ := drawn_papers_school_A + drawn_papers_school_B + drawn_papers_school_C

-- Theorem statement to prove the total checked papers is 144
theorem total_checked_papers_is_correct : total_checked_papers = 144 :=
by
  sorry

end total_checked_papers_is_correct_l714_714615


namespace exists_integer_div_15_sqrt_range_l714_714752

theorem exists_integer_div_15_sqrt_range :
  ∃ n : ℕ, (25^2 ≤ n ∧ n ≤ 26^2) ∧ (n % 15 = 0) :=
by
  sorry

end exists_integer_div_15_sqrt_range_l714_714752


namespace problem_statement_l714_714673

def num_sequences (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | _ => num_sequences (n - 1) + num_sequences (n - 2) + num_sequences (n - 3)

def p_q_sum : ℕ :=
  let p := num_sequences 12
  let q := 2 ^ 12
  p + q

theorem problem_statement : p_q_sum = 5023 :=
  by
    unfold p_q_sum num_sequences
    -- Calculation steps can be inserted here, but we use sorry to skip proof
    sorry

end problem_statement_l714_714673


namespace problem_1a_problem_1b_l714_714133

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x - x

theorem problem_1a (a : ℝ) : 
  a = 1 → (0 < x ∧ x < 1 → HasDerivAt (f a) (1 / x - 1) x ∧ (f a)' x > 0) ∧ 
          (x > 1 → HasDerivAt (f a) (1 / x - 1) x ∧ (f a)' x < 0) := 
by
  sorry

theorem problem_1b (a : ℝ) (x : ℝ) (h : 0 < x ∧ x ≤ 2) : 
  a > 0 → (
    (a < 2 → ∃ c ∈ (0, -- sorry
end 끝
 1), HasDerivAt (f a) (1 / x - 1) c ∧ (f a)' c > 0 ∧ ∀ b ∈ (c, 2], HasDerivAt (f a) (1 / x - 1) b ∧ (f a)' b < 0 ∧ f a x ≤ f a a) ∨
    (a ≥ 2 → ∀ d ∈ (0, 2], HasDerivAt (f a) (1 / x - 1) d ∧ (f a)' d > 0 ∧ f a x ≤ f a 2) 
  ∧ (
    (a < 2 → f a (a) = a * Real.log a - a) ∨ 
    (a ≥ 2 → f a (2) = 2 * Real.log 2 - 2)
) := 
by
  sorry

end problem_1a_problem_1b_l714_714133


namespace cos_120_eq_neg_half_l714_714023

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714023


namespace coefficient_x8_in_expansion_l714_714876

theorem coefficient_x8_in_expansion :
  coefficient (x^8) ((1 - x)^2 * (2 - x)^8) = 145 :=
by
  sorry

end coefficient_x8_in_expansion_l714_714876


namespace triangle_area_l714_714904

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l714_714904


namespace cos_120_degrees_l714_714000

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l714_714000


namespace least_integer_remainder_condition_l714_714289

def is_least_integer_with_remainder_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k ∈ [3, 4, 5, 6, 7, 10, 11], n % k = 1)

theorem least_integer_remainder_condition : ∃ (n : ℕ), is_least_integer_with_remainder_condition n ∧ n = 4621 :=
by
  -- The proof will go here.
  sorry

end least_integer_remainder_condition_l714_714289


namespace ratio_sub_div_eq_l714_714081

theorem ratio_sub_div_eq 
  (a b : ℚ) 
  (h : a / b = 5 / 2) : 
  (a - b) / a = 3 / 5 := 
sorry

end ratio_sub_div_eq_l714_714081


namespace find_m_l714_714849

noncomputable def circle_c_eq (m : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ,
    (A.1 = 0 ∧ B.1 = 0) ∧
    (let (x, y) := A in (x^2 + y^2 - 4*x + 2*y + m = 0)) ∧
    (let (x, y) := B in (x^2 + y^2 - 4*x + 2*y + m = 0)) ∧
    ∀ C : ℝ × ℝ, C = (2, -1) → ∃ D : ℝ × ℝ,
      D = (0, -1) → 
      let (a, b) := A in
      let (c, d) := B in
      ((a - 2) ^ 2 + (b + 1) ^ 2) = ((c - 2）^ 2 + (d + 1) ^ 2) /
      (∠ (vector.fromCartesian (2, -1)) (vector.fromCartesian (A.1, A.2)) (vector.fromCartesian (B.1, B.2))) = 90

theorem find_m {m : ℝ} : (circle_c_eq m) → m = -3 :=
begin
  sorry,
end

end find_m_l714_714849


namespace arith_prog_a1_a10_geom_prog_a1_a10_l714_714796

-- First we define our sequence and conditions for the arithmetic progression case
def is_arith_prog (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + d * (n - 1)

-- Arithmetic progression case
theorem arith_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_ap : is_arith_prog a) :
  a 1 * a 10 = -728 := 
  sorry

-- Then we define our sequence and conditions for the geometric progression case
def is_geom_prog (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Geometric progression case
theorem geom_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_gp : is_geom_prog a) :
  a 1 + a 10 = -7 := 
  sorry

end arith_prog_a1_a10_geom_prog_a1_a10_l714_714796


namespace total_graph_length_2012_l714_714917

def f (x : ℝ) : ℝ :=
  if x ≤ 1 / 2 then 2 * x else 2 - 2 * x

theorem total_graph_length_2012 :
  let g := (function.iterate f 2012)
  \exists len : ℝ, 
  (∀ x ∈ Icc 0 1, ∥g x - f x∥ = 1 / (2 : ℝ) ^ 2012) * (2^2012) = sqrt (4^2012 + 1) :=
sorry

end total_graph_length_2012_l714_714917


namespace max_value_of_k_l714_714896

theorem max_value_of_k (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m)) ≥ k) ↔ k ≤ 8 := 
sorry

end max_value_of_k_l714_714896


namespace find_length_of_BC_l714_714183

-- Define the geometrical objects and lengths
variable {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable (AB AC AM BC : ℝ)
variable (is_midpoint : Midpoint M B C)
variable (known_AB : AB = 7)
variable (known_AC : AC = 6)
variable (known_AM : AM = 4)

theorem find_length_of_BC : BC = Real.sqrt 106 := by
  sorry

end find_length_of_BC_l714_714183


namespace no_infinite_repeats_l714_714773

-- Conditions
def n : ℕ := sorry  -- positive integer n, the number of vertices
def lbl : ℕ := 10000  -- initial label for each vertex

-- Procedure P
def P (x y z : ℕ) : (ℕ × ℕ × ℕ) :=
  (x - 1, y + 2023, z - 1)

-- Main theorem stating that procedure P cannot be repeated infinitely
theorem no_infinite_repeats (n : ℕ) (h : n > 0) :
  ∀ (vertices : list ℕ), (∀ v ∈ vertices, v = lbl) → ¬ ∀ P, true :=
sorry

end no_infinite_repeats_l714_714773


namespace passing_coach_count_l714_714621

theorem passing_coach_count
  (d : ℝ) (ht : 0 < d) -- ensures positive circumference
  (tb: ℝ) (htb1 : tb = d / 16) -- defines meeting time
  (a1 : ℝ) (a1_speed: a1 = 6) -- speed of the first boy
  (a2 : ℝ) (a2_speed: a2 = 10) -- speed of the second boy
  (half_d : tb / 2 = d / 2) -- point B is halfway
  : (⌊(a1 * tb / d) * 2⌋ = 0) ∧ (⌊(a2 * tb / d) * 2⌋ = 1) := 
begin
  sorry
end

end passing_coach_count_l714_714621


namespace tom_pie_portion_l714_714356

theorem tom_pie_portion :
  let pie_left := 5 / 8
  let friends := 4
  let portion_per_person := pie_left / friends
  portion_per_person = 5 / 32 := by
  sorry

end tom_pie_portion_l714_714356


namespace find_100k_l714_714989

-- Define the square and properties
structure Square (A B C D : Point) :=
(side_length : ℝ)
(vertex_positions : A = (0,0) ∧ B = (2,0) ∧ C = (2,2) ∧ D = (0,2))

-- Define the property of line segments
def line_segment (P Q : Point) :=
(dist P Q = 2)

-- Define set S as the set of all line segments with the given property
def S (A B C D : Point) : set (Point × Point) :=
{ PQ : Point × Point | line_segment PQ.1 PQ.2 ∧
(PQ.1 ∈ {A, B, C, D} ∨ PQ.2 ∈ {A, B, C, D}) }

-- Midpoint definition for a line segment
def midpoint (P Q : Point) : Point :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Function to check the enclosed area formed by the midpoints
def enclosed_area (s : set (Point × Point)) : ℝ :=
4 - Real.pi

-- Define k as the enclosed area
def k_calculated (_ : Square (0, 0) (2, 0) (2, 2) (0, 2)) :=
enclosed_area (S (0, 0) (2, 0) (2, 2) (0, 2))

-- Main theorem to prove 100k = 86
theorem find_100k (sq : Square (0, 0) (2, 0) (2, 2) (0, 2)) : 
  100 * k_calculated sq = 86 := 
begin
  -- proof steps will be filled here
  sorry
end

end find_100k_l714_714989


namespace area_of_triangle_abc_l714_714619

noncomputable def inradius_isosceles_right_triangle (a : ℝ) : ℝ :=
  a * (1 - real.sqrt 2) / 2

noncomputable def area_isosceles_right_triangle (a : ℝ) : ℝ :=
  1 / 2 * a^2

theorem area_of_triangle_abc :
  let r := 3 in
  let a := 6 * (1 + real.sqrt 2) in
  inradius_isosceles_right_triangle a = r →
  area_isosceles_right_triangle a = 54 + 36 * real.sqrt 2 :=
by
  intros
  sorry

end area_of_triangle_abc_l714_714619


namespace two_digit_numbers_count_l714_714395

theorem two_digit_numbers_count : 
  ∃ (count : ℕ), (
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b = 2 * a → 
      (10 * b + a = 7 / 4 * (10 * a + b))) 
      ∧ count = 4
  ) :=
sorry

end two_digit_numbers_count_l714_714395


namespace find_n_l714_714071

theorem find_n :
  (∑ k in Finset.range (n+1), (1 : ℝ) / (Real.sqrt k + Real.sqrt (k + 1)) = 2016) ↔
  n = 4068288 := by
  sorry

end find_n_l714_714071


namespace competition_unique_correct_subsets_l714_714861

/-- Let S be a set of 557 students, and P a set of 30 problems. Estimate the number of subsets R ⊆ P
where some student answered only the questions in R correctly and no others. The estimated number A
is approximately 450. -/
theorem competition_unique_correct_subsets :
  let S := 557
  let P := 30
  let N := 2 ^ P
  let students_answer (s : Fin S) : Finset (Fin P) := sorry -- Assuming a function that maps each student to a subset of problems they answered correctly
  let A := (∑ (_ : Fin S), 1) / N
  A ≈ 450 :=
begin
  sorry
end

end competition_unique_correct_subsets_l714_714861


namespace probability_red_nonjoker_then_black_or_joker_l714_714864

theorem probability_red_nonjoker_then_black_or_joker :
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  probability = 5 / 17 :=
by
  -- Definitions for the conditions
  let total_cards := 60
  let red_non_joker := 26
  let black_or_joker := 40
  let total_ways_to_draw_two_cards := total_cards * (total_cards - 1)
  let probability := (red_non_joker * black_or_joker : ℚ) / total_ways_to_draw_two_cards
  -- Add sorry placeholder for proof
  sorry

end probability_red_nonjoker_then_black_or_joker_l714_714864


namespace interest_rate_equivalence_l714_714269

noncomputable def principal (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  SI * 100 / (R * T)

noncomputable def new_rate (SI : ℝ) (P : ℝ) (T' : ℝ) : ℝ :=
  SI * 100 / (P * T')

theorem interest_rate_equivalence :
  ∃ R' : ℝ,
    let SI := 840 in
    let R := 5 in
    let T := 8 in
    let P := principal SI R T in
    let T' := 5 in
    new_rate SI P T' = 8 :=
begin
  sorry
end

end interest_rate_equivalence_l714_714269


namespace pq1_eq_pq2_l714_714354

-- Define the problem
theorem pq1_eq_pq2
  (O₁ O₂ M N A B P Q₁ Q₂ : Point)
  (c₁ : Circle)
  (c₂ : Circle)
  (h₁ : A ∈ c₁ ∧ A ∈ c₂ ∧ B ∈ c₁ ∧ B ∈ c₂)
  (h₂ : Line_perp_to_AB : Line)
  (h₃ : MN : Line)
  (h₄ : Midpoint P M N)
  (h₅ : Q₁ ∈ c₁)
  (h₆ : Q₂ ∈ c₂)
  (h₇ : angle_eq (angle A O₁ Q₁) (angle A O₂ Q₂))
  (h₈ : P ∈ MN ∧ line_perpendicular MN AB) :
  dist P Q₁ = dist P Q₂ :=
by
  sorry

end pq1_eq_pq2_l714_714354


namespace complement_union_correct_l714_714218

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {2, 3, 4})
variable (hB : B = {1, 4})

theorem complement_union_correct :
  (compl A ∪ B) = {1, 4, 5} :=
by
  sorry

end complement_union_correct_l714_714218


namespace prob_female_l714_714488

/-- Define basic probabilities for names and their gender associations -/
variables (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions for the probabilities
axiom h1 : P_Alexander = 3 * P_Alexandra
axiom h2 : P_Yevgeny = 3 * P_Evgenia
axiom h3 : P_Valentin = 1.5 * P_Valentina
axiom h4 : P_Vasily = 49 * P_Vasilisa

/-- The problem we need to prove: the probability that the lot was won by a female student is approximately 0.355 -/
theorem prob_female : 
  let P_female := (P_Alexandra * 1 / 4) + (P_Evgenia * 1 / 4) + (P_Valentina * 1 / 4) + (P_Vasilisa * 1 / 4) in
  abs (P_female - 0.355) < 0.001 :=
sorry

end prob_female_l714_714488


namespace cos_120_eq_neg_one_half_l714_714012

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714012


namespace constant_remainder_l714_714046

def polynomial := (12 : ℚ) * (x^3) - (9 : ℚ) * (x^2) + b * x + (8 : ℚ)
def divisor_polynomial := (3 : ℚ) * (x^2) - (4 : ℚ) * x + (2 : ℚ)

theorem constant_remainder (b : ℚ) :
  (∃ r : ℚ, ∀ x : ℚ, (12 * (x^3) - 9 * (x^2) + b * x + 8) % (3 * (x^2) - 4 * x + 2) = r) ↔ b = -4 / 3 :=
by
  sorry

end constant_remainder_l714_714046


namespace correct_function_l714_714347

def f1 (x : ℝ) : ℝ := Real.sin (Real.abs x)
def f2 (x : ℝ) : ℝ := Real.cos (Real.abs x)
def f3 (x : ℝ) : ℝ := Real.abs (Real.cot x)
def f4 (x : ℝ) : ℝ := Real.log (Real.abs (Real.sin x))

theorem correct_function :
    ∃ f : ℝ → ℝ,
    (f = f4) ∧
    (∀ x, f x = f (x + π)) ∧
    (∀ x, f x = f (-x)) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) :=
by {
    sorry
}

end correct_function_l714_714347


namespace fixed_point_of_logarithmic_function_l714_714253

theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, x = -2 ∧ y = -2 ∧ y = log a (x + 3) - 2 := 
sorry

end fixed_point_of_logarithmic_function_l714_714253


namespace tablecloth_length_l714_714955

def napkin_width : ℤ := 6
def napkin_length : ℤ := 7
def num_napkins : ℤ := 8
def tablecloth_width : ℤ := 54
def total_material : ℤ := 5844

theorem tablecloth_length :
  let napkin_area := napkin_width * napkin_length
  let total_napkin_area := num_napkins * napkin_area
  let tablecloth_area := total_material - total_napkin_area
  tablecloth_area / tablecloth_width = 102 := 
by 
  let napkin_area := napkin_width * napkin_length
  let total_napkin_area := num_napkins * napkin_area
  let tablecloth_area := total_material - total_napkin_area
  show tablecloth_area / tablecloth_width = 102 from sorry

end tablecloth_length_l714_714955


namespace range_of_a_l714_714131

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x^2 + a * x else a * x - 4

theorem range_of_a :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ a < 4 :=
sorry

end range_of_a_l714_714131


namespace find_min_a_plus_b_l714_714932

open Nat

noncomputable def smallest_a_plus_b : ℕ :=
  let candidates := (1, 1).succ_powerset (1000, 1000).succ_powerset -- to explore a range up to 1000; this range should be set reasonably high
  candidates.filter (λ (a, b), (a^a % b^b = 0) ∧ (a % b ≠ 0) ∧ (gcd b 210 = 1)) |>.map (λ (a, b), a + b) |> min

theorem find_min_a_plus_b : smallest_a_plus_b = 374 := by
  sorry

end find_min_a_plus_b_l714_714932


namespace inequality_proof_l714_714543

noncomputable def positive_real_inequality (n : ℕ) (x : ℕ → ℝ) 
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → x i > 0) (h_cycle: x (n + 1) = x 1) : 
  Prop := 
  (∑ i in Finset.range n, (x i * x (i + 1) / (x i + x (i + 1)))) ≤
  (1/2) * (∑ i in Finset.range n, x i)

theorem inequality_proof {n : ℕ} {x : ℕ → ℝ}
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → x i > 0) (h_cycle: x (n + 1) = x 1) : 
  positive_real_inequality n x h_pos h_cycle :=
sorry

end inequality_proof_l714_714543


namespace decreasing_function_inequality_l714_714922

variable {α : Type*} [LinearOrder α] {f : α → ℝ}

def is_decreasing (f : α → ℝ) := ∀ {x y : α}, x < y → f x > f y

theorem decreasing_function_inequality (f_dec : is_decreasing f) (a : ℝ) : f (a^2 + 1) < f a :=
  by 
  sorry

end decreasing_function_inequality_l714_714922


namespace angle_between_given_vectors_l714_714124

noncomputable def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ := arccos ((a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / (real.sqrt (a.1^2 + a.2^2 + a.3^2) * real.sqrt (b.1^2 + b.2^2 + b.3^2)))

theorem angle_between_given_vectors (a b : ℝ × ℝ × ℝ)
  (ha : a.1^2 + a.2^2 + a.3^2 = 1)
  (hb : b.1^2 + b.2^2 + b.3^2 = 1)
  (h : real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2) = real.sqrt 3 * real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2 + (a.3 + b.3)^2)) 
  : angle_between_vectors a b = 2 * real.pi / 3 :=
sorry

end angle_between_given_vectors_l714_714124


namespace general_term_sequence_l714_714509

open Nat

noncomputable def sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 2

theorem general_term_sequence (a : ℕ → ℕ) (h : sequence a) : ∀ n, a n = 2 * n - 1 := 
by
  sorry

end general_term_sequence_l714_714509


namespace minimum_possible_value_l714_714533

theorem minimum_possible_value :
  ∀ (p q r s t u v w : ℤ),
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
    r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
    s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
    t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
    u ≠ v ∧ u ≠ w ∧
    v ≠ w ∧
    p ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    q ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    r ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    s ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    t ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    u ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    v ∈ {-8, -6, -4, -1, 3, 5, 7, 14} ∧ 
    w ∈ {-8, -6, -4, -1, 3, 5, 7, 14} →
    (p + q + r + s = 7 ∧ t + u + v + w = 7) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 98 :=
by
  intros
  sorry

end minimum_possible_value_l714_714533


namespace pieces_left_after_third_day_l714_714557

theorem pieces_left_after_third_day : 
  let initial_pieces := 1000
  let first_day_pieces := initial_pieces * 0.10
  let remaining_first_day := initial_pieces - first_day_pieces
  let second_day_pieces := remaining_first_day * 0.20
  let remaining_second_day := remaining_first_day - second_day_pieces
  let third_day_pieces := remaining_second_day * 0.30
  let pieces_left := remaining_second_day - third_day_pieces 
  in pieces_left = 504 :=
by
  sorry

end pieces_left_after_third_day_l714_714557


namespace min_x_plus_9y_l714_714110

variable {x y : ℝ}

theorem min_x_plus_9y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / y = 1) : x + 9 * y ≥ 16 :=
  sorry

end min_x_plus_9y_l714_714110


namespace arithmetic_mean_first_n_positive_even_integers_l714_714628

theorem arithmetic_mean_first_n_positive_even_integers (n : ℕ) : 
  let S_n := n * (n + 1) in
  (S_n / n = n + 1) :=
by
  sorry

end arithmetic_mean_first_n_positive_even_integers_l714_714628


namespace round_18_4831_to_nearest_hundredth_l714_714567

def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  let factor := 100
  let scaled_x := x * factor
  let rounded_scaled_x := if scaled_x.floor % 10 < 5 then scaled_x.floor else (scaled_x.floor + 1)
  (rounded_scaled_x : ℝ) / factor

theorem round_18_4831_to_nearest_hundredth :
  round_to_nearest_hundredth 18.4831 = 18.48 :=
by
  sorry

end round_18_4831_to_nearest_hundredth_l714_714567


namespace tetrahedron_ratio_sum_eq_one_l714_714401

variables {A B C D O E F G H : Type} [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] 
[EuclideanSpace ℝ C] [EuclideanSpace ℝ D] [EuclideanSpace ℝ O]
[EuclideanSpace ℝ E] [EuclideanSpace ℝ F] [EuclideanSpace ℝ G] [EuclideanSpace ℝ H]
(V : EuclideanSpace ℝ → EuclideanSpace ℝ → EuclideanSpace ℝ → EuclideanSpace ℝ → ℝ)

theorem tetrahedron_ratio_sum_eq_one
  (h1 : E ∈ affine_span ℝ {A, B, C, D}) 
  (h2 : F ∈ affine_span ℝ {A, B, C, D}) 
  (h3 : G ∈ affine_span ℝ {A, B, C, D}) 
  (h4 : H ∈ affine_span ℝ {A, B, C, D}) 
  (h5 : affine_independent ℝ ![A, B, C, D]) :
  (V O B C D / V A B C D) + (V O A C D / V D A B C) + 
  (V O A B D / V B A C D) + (V O A B C / V C A B D) = 1 :=
sorry

end tetrahedron_ratio_sum_eq_one_l714_714401


namespace height_of_building_l714_714324

def flagpole_height : ℝ := 18
def flagpole_shadow_length : ℝ := 45

def building_shadow_length : ℝ := 65
def building_height : ℝ := 26

theorem height_of_building
  (hflagpole : flagpole_height / flagpole_shadow_length = building_height / building_shadow_length) :
  building_height = 26 :=
sorry

end height_of_building_l714_714324


namespace cone_has_infinite_generatrices_l714_714840

noncomputable def cone_generatrix_infinite : Prop :=
  ∃ base: set ℝ², (∀ p ∈ base, (∃ l: ℝ² → ℝ², true)) ∧ (∀ q₁ q₂ : ℝ², q₁ ≠ q₂ → q₁ ∈ base ∧ q₂ ∈ base) → false

theorem cone_has_infinite_generatrices : cone_generatrix_infinite :=
by {
  sorry
}

end cone_has_infinite_generatrices_l714_714840


namespace daily_earnings_from_oil_refining_l714_714223

-- Definitions based on conditions
def daily_earnings_from_mining : ℝ := 3000000
def monthly_expenses : ℝ := 30000000
def fine : ℝ := 25600000
def profit_percentage : ℝ := 0.01
def months_in_year : ℝ := 12
def days_in_month : ℝ := 30

-- The question translated as a Lean theorem statement
theorem daily_earnings_from_oil_refining : ∃ O : ℝ, O = 5111111.11 ∧ 
  fine = profit_percentage * months_in_year * 
    (days_in_month * (daily_earnings_from_mining + O) - monthly_expenses) :=
sorry

end daily_earnings_from_oil_refining_l714_714223


namespace perimeter_ABCD_l714_714177

theorem perimeter_ABCD (A B C D E : Point)
  (h1 : RightAngle A E B)
  (h2 : RightAngle B E C)
  (h3 : RightAngle C E D)
  (h4 : ∠ A E B = 60)
  (h5 : ∠ B E C = 60)
  (h6 : ∠ C E D = 60)
  (hAE : dist A E = 30) :
  dist A B + dist B C + dist C D + dist D A = 33.75 + 26.25 * Real.sqrt 3 := 
sorry

end perimeter_ABCD_l714_714177


namespace probability_female_win_l714_714500

variable (P_Alexandr P_Alexandra P_Evgeniev P_Evgenii P_Valentinov P_Valentin P_Vasilev P_Vasilisa : ℝ)

-- Conditions
axiom h1 : P_Alexandr = 3 * P_Alexandra
axiom h2 : P_Evgeniev = (1 / 3) * P_Evgenii
axiom h3 : P_Valentinov = (1.5) * P_Valentin
axiom h4 : P_Vasilev = 49 * P_Vasilisa
axiom h5 : P_Alexandr + P_Alexandra = 1
axiom h6 : P_Evgeniev + P_Evgenii = 1
axiom h7 : P_Valentinov + P_Valentin = 1
axiom h8 : P_Vasilev + P_Vasilisa = 1

-- Statement to prove
theorem probability_female_win : 
  let P_female := (1/4) * P_Alexandra + (1/4) * P_Evgeniev + (1/4) * P_Valentinov + (1/4) * P_Vasilisa in
  P_female = 0.355 :=
by
  sorry

end probability_female_win_l714_714500


namespace rational_squared_is_rational_l714_714637

theorem rational_squared_is_rational (x : ℚ) : x^2 ∈ ℚ :=
by sorry

end rational_squared_is_rational_l714_714637


namespace inequality_abc_l714_714540

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c)) :=
by
  sorry

end inequality_abc_l714_714540


namespace part1_part2_l714_714102

variables {a_n b_n : ℕ → ℤ} {k m : ℕ}

-- Part 1: Arithmetic Sequence
axiom a2_eq_3 : a_n 2 = 3
axiom S5_eq_25 : (5 * (2 * (a_n 1 + 2 * (a_n 1 + 1)) / 2)) = 25

-- Part 2: Geometric Sequence
axiom b1_eq_1 : b_n 1 = 1
axiom q_eq_3 : ∀ n, b_n n = 3^(n-1)

noncomputable def arithmetic_seq (n : ℕ) : ℤ :=
  2 * n - 1

theorem part1 : (a_n 2 + a_n 4) / 2 = 5 :=
  sorry

theorem part2 (k : ℕ) (hk : 0 < k) : ∃ m, b_n k = arithmetic_seq m ∧ m = (3^(k-1) + 1) / 2 :=
  sorry

end part1_part2_l714_714102


namespace collinear_O_P_Q_sum_of_slopes_constant_l714_714702

open Real

namespace GeometricProblem

variables {a b λ : ℝ}
variable (A B P Q O : ℝ × ℝ)

-- Conditions
def ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1^2 / (a^2) + P.2^2 / (b^2) = 1

def hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1^2 / (a^2) - P.2^2 / (b^2) = 1

def condition (A B P Q : ℝ × ℝ) (λ : ℝ) : Prop :=
  vector.of_tuple (P - A) + vector.of_tuple (P - B) = λ • (vector.of_tuple (Q - A) + vector.of_tuple (Q - B))

variables (h_ellipse_A : ellipse a b A)
          (h_hyperbola_A : hyperbola a b A)
          (h_ellipse_B : ellipse a b B)
          (h_hyperbola_B : hyperbola a b B)
          (h_hyperbola_P : hyperbola a b P)
          (h_ellipse_Q : ellipse a b Q)
          (h_condition : condition A B P Q λ)
          (h_λ_gt1 : |λ| > 1)

-- Prove that points O, P, and Q are collinear
theorem collinear_O_P_Q :
  collinear O P Q := sorry

-- Prove that sum of slopes is constant
theorem sum_of_slopes_constant (k1 k2 k3 k4 : ℝ) :
  slopes A P B Q k1 k2 k3 k4 →
  k1 + k2 + k3 + k4 = constant := sorry

end GeometricProblem

end collinear_O_P_Q_sum_of_slopes_constant_l714_714702


namespace range_of_a_l714_714455

theorem range_of_a (a : ℝ) (h1 : a > -1) (h2 : a ≠ 0)
    (h3 : ∀ x ∈ Icc (1:ℝ) 2, deriv (λ x, -x^2 + 2*a*x) x < 0)
    (h4 : ∀ x ∈ Icc (1:ℝ) 2, deriv (λ x, (a+1)^(1-x)) x < 0) :
    0 < a ∧ a ≤ 1 :=
begin
  sorry
end

end range_of_a_l714_714455


namespace wire_length_is_sqrt_337_l714_714583

noncomputable def wire_length_between_poles : ℝ :=
  let horizontal_distance : ℝ := 16
  let height_difference : ℝ := 15 - 6
  real.sqrt (horizontal_distance^2 + height_difference^2)

theorem wire_length_is_sqrt_337 : wire_length_between_poles = real.sqrt 337 := by
  -- Proof steps are not required
  sorry

end wire_length_is_sqrt_337_l714_714583


namespace second_car_distance_l714_714622

variables 
  (distance_apart : ℕ := 105)
  (d1 d2 d3 : ℕ := 25) -- distances 25 km, 15 km, 25 km respectively
  (d_road_back : ℕ := 15)
  (final_distance : ℕ := 20)

theorem second_car_distance 
  (car1_total_distance := d1 + d2 + d3 + d_road_back)
  (car2_distance : ℕ) :
  distance_apart - (car1_total_distance + car2_distance) = final_distance →
  car2_distance = 5 :=
sorry

end second_car_distance_l714_714622


namespace smallest_N_l714_714722

def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ (n' : ℕ) (a_n : ℕ), a_n + Nat.floor (Real.sqrt a_n))

theorem smallest_N (N : ℕ) (hN : seq N > 2017) (hN_minus : ∀ m < N, seq m ≤ 2017) : N = 45 := 
sorry

end smallest_N_l714_714722


namespace sum_of_integral_c_l714_714372

theorem sum_of_integral_c (c : ℤ) (h1 : c ≤ 30) (h2 : c ≥ -20) :
  ((x^2 - 9 * x - c = 0) -> ∃ (k : ℤ), 81 + 4 * c = k^2) -> 
  (\sum c, (c ≤ 30 ∧ c ≥ -20 ∧ ∃ (k : ℤ), 81 + 4 * c = k^2)) = -28 := 
sorry

end sum_of_integral_c_l714_714372


namespace function_decreasing_on_interval_l714_714571

open Real

def declining_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → deriv f x < 0

theorem function_decreasing_on_interval :
  declining_interval (λ x : ℝ, 2 * x ^ 3 + 3 * x ^ 2 - 12 * x + 1) (-2) 1 :=
sorry

end function_decreasing_on_interval_l714_714571


namespace original_price_hat_l714_714283

theorem original_price_hat 
  (x : ℝ)
  (discounted_price := x / 5)
  (final_price := discounted_price * 1.2)
  (h : final_price = 8) :
  x = 100 / 3 :=
by
  sorry

end original_price_hat_l714_714283


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l714_714734

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l714_714734


namespace trajectory_of_intersection_l714_714417

-- Define the conditions and question in Lean
structure Point where
  x : ℝ
  y : ℝ

def on_circle (C : Point) : Prop :=
  C.x^2 + C.y^2 = 1

def perp_to_x_axis (C D : Point) : Prop :=
  C.x = D.x ∧ C.y = -D.y

theorem trajectory_of_intersection (A B C D M : Point)
  (hA : A = {x := -1, y := 0})
  (hB : B = {x := 1, y := 0})
  (hC : on_circle C)
  (hD : on_circle D)
  (hCD : perp_to_x_axis C D)
  (hM : ∃ m n : ℝ, C = {x := m, y := n} ∧ M = {x := 1 / m, y := n / m}) :
  M.x^2 - M.y^2 = 1 ∧ M.y ≠ 0 :=
by
  sorry

end trajectory_of_intersection_l714_714417


namespace find_x_that_satisfies_log_l714_714383

theorem find_x_that_satisfies_log (x : ℝ) : log 8 (x + 8) = 3 → x = 504 := by
  sorry

end find_x_that_satisfies_log_l714_714383


namespace debbys_sister_candy_l714_714772

-- Defining the conditions
def debby_candy : ℕ := 32
def eaten_candy : ℕ := 35
def remaining_candy : ℕ := 39

-- The proof problem
theorem debbys_sister_candy : ∃ S : ℕ, debby_candy + S - eaten_candy = remaining_candy → S = 42 :=
by
  sorry  -- The proof goes here

end debbys_sister_candy_l714_714772


namespace arithmetic_mean_after_removal_l714_714582

theorem arithmetic_mean_after_removal (S : Finset ℝ) (h₁ : S.card = 60) (h₂ : (S.sum / 60 : ℝ) = 42) 
  (h₃ : 50 ∈ S) (h₄ : 60 ∈ S) : 
  ((S.erase 50).erase 60).sum / 58 = 41.5 :=
by 
  sorry

end arithmetic_mean_after_removal_l714_714582


namespace ball_drop_height_l714_714700

theorem ball_drop_height
  (h : ℕ)
  (half_rise_each_bounce : ∀ (n : ℕ), n > 0 → h bounces respecting the height rule)
  (total_distance_travelled : ∑ k in (range 5), h * (2 ^ (-k)) = 45)
  (bounces : ℕ) (bounces = 4) :
  h = 16 :=
by
  sorry

end ball_drop_height_l714_714700


namespace angle_a_b_l714_714083

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b : ℝ × ℝ := (3, 0)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between_vectors (v w : ℝ × ℝ) : ℝ :=
  acos (dot_product v w / (norm v * norm w))

theorem angle_a_b :
  angle_between_vectors a b = π / 3 :=
sorry

end angle_a_b_l714_714083


namespace walking_distance_l714_714648

-- Define the pace in miles per hour.
def pace : ℝ := 2

-- Define the duration in hours.
def duration : ℝ := 8

-- Define the total distance walked.
def total_distance (pace : ℝ) (duration : ℝ) : ℝ := pace * duration

-- Define the theorem we need to prove.
theorem walking_distance :
  total_distance pace duration = 16 := by
  sorry

end walking_distance_l714_714648


namespace draw_odds_l714_714165

theorem draw_odds (x : ℝ) (bet_Zubilo bet_Shaiba bet_Draw payout : ℝ) (h1 : bet_Zubilo = 3 * x) (h2 : bet_Shaiba = 2 * x) (h3 : payout = 6 * x) : 
  bet_Draw * 6 = payout :=
by
  sorry

end draw_odds_l714_714165


namespace determine_k_l714_714769

-- Define the function f(x) = log x + x - 2
def f (x : ℝ) : ℝ := log x + x - 2

-- Define the main theorem
theorem determine_k (k : ℤ) (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_range : x₀ ∈ set.Ioo (k : ℝ) (k + 1)) : k = 1 :=
  sorry

end determine_k_l714_714769


namespace cos_sum_angle_l714_714453

theorem cos_sum_angle (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
begin
  sorry,
end

end cos_sum_angle_l714_714453


namespace ceil_neg_sqrt_frac_l714_714745

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l714_714745


namespace mary_characters_initials_l714_714926

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l714_714926


namespace cosine_value_l714_714146

def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
def n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)

theorem cosine_value (x : ℝ) (h : (m x).1 * (n x).1 + (m x).2 * (n x).2 = 0) :
  cos (x + π / 3) = 1 / 2 :=
by 
  sorry

end cosine_value_l714_714146


namespace closest_integer_to_35_div_4_l714_714636

theorem closest_integer_to_35_div_4 : closest_integer (35 / 4) = 9 :=
by
  sorry

end closest_integer_to_35_div_4_l714_714636


namespace proof_50th_permutation_l714_714591

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def nth_permutation (l : List ℕ) (n : ℕ) : List ℕ :=
  let f := factorial (l.length - 1)
  if l = [] then [] else
  let (q, r) := n.div_mod f
  let x := l.nth_le q ((Nat.lt_of_sub_lt_sub (Nat.zero_lt_succ _)).1 (Nat.pos_iff_ne_zero.2 (factorial_ne_zero (l.length - 1))))
  x::nth_permutation (List.erase l x) r

theorem proof_50th_permutation : (nth_permutation [1, 2, 3, 4, 5] 49) = [3, 1, 2, 5, 4] :=
  by sorry

end proof_50th_permutation_l714_714591


namespace cos_120_eq_neg_one_half_l714_714005

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714005


namespace num_triples_l714_714766

/-- Theorem statement:
There are exactly 2 triples of positive integers (a, b, c) satisfying the conditions:
1. ab + ac = 60
2. bc + ac = 36
3. ab + bc = 48
--/
theorem num_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (ab + ac = 60) → (bc + ac = 36) → (ab + bc = 48) → 
  (a, b, c) ∈ [(1, 4, 8), (1, 12, 3)] →
  ∃! (a b c : ℕ), (ab + ac = 60) ∧ (bc + ac = 36) ∧ (ab + bc = 48) :=
sorry

end num_triples_l714_714766


namespace nat_min_a_plus_b_l714_714936

theorem nat_min_a_plus_b (a b : ℕ) (h1 : b ∣ a^a) (h2 : ¬ b ∣ a) (h3 : Nat.coprime b 210) : a + b = 374 :=
sorry

end nat_min_a_plus_b_l714_714936


namespace neg_ex_proposition_l714_714547

open Classical

theorem neg_ex_proposition :
  ¬ (∃ n : ℕ, n^2 > 2^n) ↔ ∀ n : ℕ, n^2 ≤ 2^n :=
by sorry

end neg_ex_proposition_l714_714547


namespace odometer_sevens_l714_714307

theorem odometer_sevens (N : ℕ) (h1 : N < 1000000) (h2 : N.digits.count 7 = 4) : 
  (N + 900).digits.count 7 ≠ 1 :=
sorry

end odometer_sevens_l714_714307


namespace problem1_problem2_problem3_problem4_l714_714712

-- Problem (1): Prove that sqrt(8) - sqrt(1/2) = 3*sqrt(2)/2
theorem problem1 : sqrt 8 - sqrt (1 / 2) = (3 * sqrt 2) / 2 := sorry

-- Problem (2): Prove that (sqrt(75) - sqrt(3))/sqrt(3) - sqrt(1/5)*sqrt(20) = 2
theorem problem2 : (sqrt 75 - sqrt 3) / sqrt 3 - sqrt (1 / 5) * sqrt 20 = 2 := sorry

-- Problem (3): Prove that (3 + 2*sqrt(2)) * (2*sqrt(2) - 3) = -1
theorem problem3 : (3 + 2 * sqrt 2) * (2 * sqrt 2 - 3) = -1 := sorry

-- Problem (4): Prove that (-sqrt(2)) * sqrt(6) + |sqrt(3) - 2| - (1/2)^{-1} = -3*sqrt(3)
theorem problem4 : (- sqrt 2) * sqrt 6 + abs (sqrt 3 - 2) - (1/2)⁻¹ = -3 * sqrt 3 := sorry

end problem1_problem2_problem3_problem4_l714_714712


namespace circumcircles_intersect_on_AB_l714_714873

theorem circumcircles_intersect_on_AB 
  (A B C E F S T : Type) 
  [Subsingleton A] [Subsingleton B] [Subsingleton C] [Subsingleton E] [Subsingleton F] [Subsingleton S] [Subsingleton T]
  (CA CB : ℝ) (h1 : CA > CB)
  (h2 : is_on_semicircle E A B)
  (h3 : is_on_semicircle F A B)
  (h4 : is_angle_bisector (E, O, F))
  (h5 : is_angle_bisector (A, C, B))
  (h6 : intersection (angle_bisector (E, O, F)) (angle_bisector (A, C, B)) = S) :
  is_on_line T A B :=
by
  sorry

end circumcircles_intersect_on_AB_l714_714873


namespace find_c_value_l714_714386

noncomputable def remainder_c (x : ℝ) (c : ℝ): ℝ :=
  let poly := 3 * x^3 + c * x^2 - 17 * x + 53
  let divisor := 3 * x + 5
  let rem := poly % divisor
  rem

theorem find_c_value:
  ∃ c : ℝ, (remainder_c x c = 6) ∧ c = -42.36 :=
begin
  sorry
end

end find_c_value_l714_714386


namespace max_matches_l714_714317

theorem max_matches (x y z m : ℕ) (h1 : x + y + z = 19) (h2 : x * y + y * z + x * z = m) : m ≤ 120 :=
sorry

end max_matches_l714_714317


namespace problem_I_problem_II_l714_714135

-- Problem (I)
theorem problem_I (f : ℝ → ℝ) (x : ℝ) (h : f x = |x - 1|) :
  ∀ x : ℝ, |f x - 3| ≤ 4 ↔ -6 ≤ x ∧ x ≤ 8 :=
sorry

-- Problem (II)
theorem problem_II (f : ℝ → ℝ) (x : ℝ) (h : f x = |x - 1|) :
  ∀ m : ℝ, (∀ x : ℝ, f x + f (x + 3) ≥ m^2 - 2m) ↔ (-1 ≤ m ∧ m ≤ 3) :=
sorry

end problem_I_problem_II_l714_714135


namespace expression_value_l714_714565

theorem expression_value (a b c : ℝ) (h₁ : 2^a = 5) (h₂ : 2^b = 10) (h₃ : 2^c = 80) :
    2006 * a - 3344 * b + 1338 * c = 2008 := 
by
  sorry

end expression_value_l714_714565


namespace simplify_trig_expression_l714_714960

theorem simplify_trig_expression (x : ℝ) (h₁ : sin x ≠ 0) (h₂ : cos x ≠ -1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) := 
sorry

end simplify_trig_expression_l714_714960


namespace not_factorial_tail_numbers_lt_1992_l714_714467

noncomputable def factorial_tail_number_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + factorial_tail_number_count (n / 5)

theorem not_factorial_tail_numbers_lt_1992 :
  ∃ n, n < 1992 ∧ n = 1992 - (1992 / 5 + (1992 / 25 + (1992 / 125 + (1992 / 625 + 0)))) :=
sorry

end not_factorial_tail_numbers_lt_1992_l714_714467


namespace probability_odd_product_l714_714436

-- Given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the proof problem
theorem probability_odd_product (h: choices = 15 ∧ odd_choices = 3) :
  (odd_choices : ℚ) / choices = 1 / 5 :=
by sorry

end probability_odd_product_l714_714436


namespace quadratic_inequality_solution_l714_714380

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l714_714380


namespace john_total_weight_moved_l714_714191

-- Definitions
def initial_back_squat := 200 -- kg
def back_squat_increase := 50 -- kg
def back_squat (initial_back_squat back_squat_increase : ℝ) : ℝ :=
  initial_back_squat + back_squat_increase

def front_squat_percent := 0.8
def front_squat (bs : ℝ) (fsp : ℝ) : ℝ :=
  fsp * bs

def triple_percent := 0.9
def triple_weight (fs : ℝ) (tp : ℝ) : ℝ :=
  tp * fs

def total_triple_reps := 3
def total_weight_moved (tw : ℝ) (reps : ℝ) : ℝ :=
  tw * reps

-- Theorem
theorem john_total_weight_moved :
  total_weight_moved (triple_weight (front_squat (back_squat initial_back_squat back_squat_increase) front_squat_percent) triple_percent) total_triple_reps = 540 := by
  sorry

end john_total_weight_moved_l714_714191


namespace smallest_positive_period_intervals_of_monotonicity_max_min_values_l714_714430

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Prove the smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x := sorry

-- Prove the intervals of monotonicity
theorem intervals_of_monotonicity (k : ℤ) : 
  ∀ x y, (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
         (k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ k * Real.pi + Real.pi / 6) → 
         (x < y → f x < f y) ∨ (y < x → f y < f x) := sorry

-- Prove the maximum and minimum values on [0, π/2]
theorem max_min_values : ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧ 
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val ∧ f x ≥ min_val := sorry

end smallest_positive_period_intervals_of_monotonicity_max_min_values_l714_714430


namespace haley_trees_initially_grew_l714_714839

-- Given conditions
def num_trees_died : ℕ := 2
def num_trees_survived : ℕ := num_trees_died + 7

-- Prove the total number of trees initially grown
theorem haley_trees_initially_grew : num_trees_died + num_trees_survived = 11 :=
by
  -- here we would provide the proof eventually
  sorry

end haley_trees_initially_grew_l714_714839


namespace estimate_rabbit_population_l714_714502

theorem estimate_rabbit_population (marked_released marked_recaptured total_captured : ℕ) 
  (h1 : marked_released = 100)
  (h2 : marked_recaptured = 5)
  (h3 : total_captured = 40) : 
  let estimate := (total_captured * marked_released) / marked_recaptured
  in estimate = 800 :=
by
  sorry

end estimate_rabbit_population_l714_714502


namespace savings_using_raspberries_instead_of_blueberries_l714_714708

-- Definitions
def blueberry_price_per_carton := 5.0 -- dollars per 6 ounces
def raspberry_price_per_carton := 3.0 -- dollars per 8 ounces
def ounces_per_blueberry_carton := 6 -- ounces
def ounces_per_raspberry_carton := 8 -- ounces
def batches := 4
def ounces_per_batch := 12

-- Theorem statement
theorem savings_using_raspberries_instead_of_blueberries :
  (batches * ounces_per_batch / ounces_per_blueberry_carton * blueberry_price_per_carton) - 
  (batches * ounces_per_batch / ounces_per_raspberry_carton * raspberry_price_per_carton) = 22 := 
by
  sorry

end savings_using_raspberries_instead_of_blueberries_l714_714708


namespace discount_percentage_clearance_sale_l714_714957

noncomputable def float_eq (a b : ℚ) (ε : ℚ) : Prop :=
  abs (a - b) < ε

theorem discount_percentage_clearance_sale :
  ∃ (D : ℚ), float_eq D 9.99 0.01 :=
begin
  let SP := 30 : ℚ,
  let gain_percent := 0.30 : ℚ,
  let sale_gain_percent := 0.17 : ℚ,
  let CP := SP / (1 + gain_percent),
  let SP_sale := CP * (1 + sale_gain_percent),
  let MP := 30 : ℚ,
  let discount := MP - SP_sale,
  let D_percent := (discount / MP) * 100,
  use D_percent,
  sorry
end

end discount_percentage_clearance_sale_l714_714957


namespace area_of_triangle_ABC_l714_714885

/-- In triangle ABC, point K is taken on side AB such that AK: BK = 1: 2, and point L is taken on side BC
such that CL: BL = 2: 1. Let Q be the point of intersection of lines AL and CK. Prove that the area of 
triangle ABC is 7 / 4 given that the area of triangle BQC is 1. -/
theorem area_of_triangle_ABC
  (A B C K L Q : Type*)
  (hAKBK : AK : BK = 1 : 2)
  (hCLBL : CL : BL = 2 : 1)
  (hQ_inter : Q = intersection (line_through A L) (line_through C K))
  (h_area_BQC : area (triangle B Q C) = 1) :
  area (triangle A B C) = 7 / 4 :=
sorry

end area_of_triangle_ABC_l714_714885


namespace closed_polygon_can_be_enclosed_in_circle_l714_714231

theorem closed_polygon_can_be_enclosed_in_circle {P : ℝ} (hP : P > 0) :
  ∃ (R : ℝ), R = P / 4 ∧ ∀ (polygon : list (ℝ × ℝ)), (is_closed_polygon polygon) ∧ (perimeter polygon = P) → (encloses_in_circle polygon R) := 
sorry

end closed_polygon_can_be_enclosed_in_circle_l714_714231


namespace system_of_equations_solution_l714_714602

theorem system_of_equations_solution :
  ∀ x y k, (x + y = 3) ∧ (2 * x + y = k) ∧ (x = 2) → (k = 5) ∧ (y = 1) :=
by
  intros x y k h
  cases h with h1 h2
  cases h2 with h2 h3
  have y_val : y = 1 := by
    rw [h3] at h1
    exact nat.sub_left_inj 3 2 h1
  rw [y_val] at h2
  have k_val : k = 5 := by
    simp at h2
    exact h2
  exact ⟨k_val, y_val⟩

end system_of_equations_solution_l714_714602


namespace tangency_count_l714_714090

/-- Problem statement from the conditions -/
def circle_tangency_problem : Prop :=
  ∀ (r R : ℝ) (O : ℝ × ℝ), ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 8

theorem tangency_count (r R : ℝ) (O : ℝ × ℝ) :
  circle_tangency_problem :=
begin
  use 0,
  split,
  { linarith, },
  { linarith, },
end

end tangency_count_l714_714090


namespace log_base_28_of_5_eq_l714_714805

variable (a b : ℝ)

-- Given conditions:
axiom lg_2_eq_a : real.log 2 = a
axiom lg_7_eq_b : real.log 7 = b

-- Question to prove:
theorem log_base_28_of_5_eq (a b : ℝ) (h1 : real.log 2 = a) (h2 : real.log 7 = b) : 
  real.logb 28 5 = (1 - a) / (2 * a + b) := 
by
  sorry

end log_base_28_of_5_eq_l714_714805


namespace decreasing_function_range_l714_714424

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) (h_decreasing : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 → -1 < x2 ∧ x2 < 1 ∧ x1 > x2 → f x1 < f x2)
  (h_ineq: f (1 - a) < f (3 * a - 1)) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l714_714424


namespace train_number_of_cars_l714_714194

theorem train_number_of_cars (lena_cars : ℕ) (time_counted : ℕ) (total_time : ℕ) 
  (cars_in_train : ℕ)
  (h1 : lena_cars = 8) 
  (h2 : time_counted = 15)
  (h3 : total_time = 210)
  (h4 : (8 / 15 : ℚ) * 210 = 112)
  : cars_in_train = 112 :=
sorry

end train_number_of_cars_l714_714194


namespace angle_equality_l714_714198

variable (C1 C2 : Circle)
variable (O1 O2 A P1 P2 Q1 Q2 M1 M2 : Point)

-- Conditions
axiom circles_coplanar_and_non_identical : C1 ≠ C2 ∧ coplanar C1 C2
axiom intersection_points_distinct : ∃ (A B : Point), A ≠ B ∧ A ∈ (C1 ∩ C2) ∧ B ∈ (C1 ∩ C2)
axiom common_external_tangent_touch_points_1 : tangent_line C1 P1 ∧ tangent_line C2 P2
axiom common_external_tangent_touch_points_2 : tangent_line C1 Q1 ∧ tangent_line C2 Q2
axiom midpoint_def_1 : midpoint P1 Q1 M1
axiom midpoint_def_2 : midpoint P2 Q2 M2

-- Proof statement
theorem angle_equality : ∠ O1 A O2 = ∠ M1 A M2 :=
sorry

end angle_equality_l714_714198


namespace trapezoid_CD_length_l714_714617

theorem trapezoid_CD_length
  (A B C D : Type)
  (AD_parallel_BC : ∃ (AD BC : set (ℝ × ℝ)), is_parallel AD BC)
  (BD_length : BD = 2)
  (angle_DBA : angle DBA = 30)
  (angle_BDC : angle BDC = 60)
  (BC_to_AD_ratio : ratio BC AD = 7/4)
  : CD = 3/2 :=
sorry

end trapezoid_CD_length_l714_714617


namespace projection_eq_l714_714263

-- Define the given conditions
def w := (1 : ℝ, 5 : ℝ)
def proj_w := (-2 : ℝ, 1 : ℝ)

-- Define the vector to project
def v := (3 : ℝ, 2 : ℝ)

def inner_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def scalar_mult (r : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (r * a.1, r * a.2)

-- The projection formula
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let denom := inner_product w w
  scalar_mult (inner_product v w / denom) w

-- Given condition
def condition : Prop :=
  projection (1, 5) (-2, 1) = (-2, 1)

-- The statement to prove
theorem projection_eq : condition →
  projection v (-2, 1) = (8/5, -4/5) :=
by
  intros h
  sorry

end projection_eq_l714_714263


namespace cos_triple_identity_l714_714952

theorem cos_triple_identity (α : ℝ) :
  (cos α) ^ 3 + (cos (α + 2 * π / 3)) ^ 3 + (cos (α - 2 * π / 3)) ^ 3 = (3 / 4) * cos (3 * α) :=
sorry

end cos_triple_identity_l714_714952


namespace ceil_neg_sqrt_64_div_9_l714_714737

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l714_714737


namespace amount_giving_l714_714393

noncomputable def total_earnings := 18 + 24 + 30 + 36 + 45
noncomputable def share_each := total_earnings / 5
noncomputable def amount_to_give := 45 - share_each

theorem amount_giving :
  amount_to_give = 14.4 :=
by
  sorry

end amount_giving_l714_714393


namespace banknotes_combination_l714_714273

theorem banknotes_combination (a b c d : ℕ) (h : a + b + c + d = 10) (h_val : 2000 * a + 1000 * b + 500 * c + 200 * d = 5000) :
  (a = 0 ∧ b = 0 ∧ c = 10 ∧ d = 0) ∨ 
  (a = 1 ∧ b = 0 ∧ c = 4 ∧ d = 5) ∨ 
  (a = 0 ∧ b = 3 ∧ c = 2 ∧ d = 5) :=
by
  sorry

end banknotes_combination_l714_714273


namespace find_sum_of_tangent_points_l714_714363

noncomputable def f (x : ℝ) : ℝ :=
max (-7 * x - 10) (max (2 * x - 3) (5 * x + 4))

def tangent_points : ℝ × ℝ × ℝ := sorry

theorem find_sum_of_tangent_points :
  let x1 := tangent_points.1,
      x2 := tangent_points.2,
      x3 := tangent_points.3 in
  x1 + x2 + x3 = -221 / 63 :=
sorry

end find_sum_of_tangent_points_l714_714363


namespace simplify_trig_expression_l714_714984

theorem simplify_trig_expression (x : ℝ) (hx : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l714_714984


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l714_714735

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l714_714735


namespace no_perfect_square_in_range_l714_714887

theorem no_perfect_square_in_range :
  ¬∃ (x : ℕ), 99990000 ≤ x ∧ x ≤ 99999999 ∧ ∃ (n : ℕ), x = n * n :=
by
  sorry

end no_perfect_square_in_range_l714_714887


namespace probability_AC_adjacent_l714_714705

noncomputable def probability_AC_adjacent_given_AB_adjacent : ℚ :=
  let total_permutations_with_AB_adjacent := 48
  let permutations_with_ABC_adjacent := 12
  permutations_with_ABC_adjacent / total_permutations_with_AB_adjacent

theorem probability_AC_adjacent :  
  probability_AC_adjacent_given_AB_adjacent = 1 / 4 :=
by
  sorry

end probability_AC_adjacent_l714_714705


namespace limit_expression_l714_714463

variable (f : ℝ → ℝ) (x₀ a : ℝ)

-- Provides the hypothesis that the derivative of f at x₀ is equal to a
axiom derivative_f_at_x0 : deriv f x₀ = a

theorem limit_expression : 
  lim (λ Δx, (f (x₀ + Δx) - f (x₀ - Δx)) / Δx) (𝓝 0) = 2 * a :=
by
  sorry

end limit_expression_l714_714463


namespace compute_a_l714_714115

theorem compute_a (a b : ℚ) :
  (-2 - 5 * real.sqrt 3) is_root (λ x, x^3 + a * x^2 + b * x + (45 : ℚ)) →
  a = 239 / 71 :=
sorry

end compute_a_l714_714115


namespace f3_plus_f_neg3_l714_714433

def f (x : ℝ) : ℝ := log (sqrt (1 + 4 * x ^ 2) - 2 * x) + 1

theorem f3_plus_f_neg3 : f 3 + f (-3) = 2 := by
  sorry

end f3_plus_f_neg3_l714_714433


namespace determine_a_l714_714040

theorem determine_a:
  ∀ (a : ℝ),
  (∃ (x1 x2 x3 x4 x5 : ℝ),
    (x1 > 0) ∧ (x2 > 0) ∧ (x3 > 0) ∧ (x4 > 0) ∧ (x5 > 0) ∧
    ((1 * x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 = a) ∧
     (1^3 * x1 + 2^3 * x2 + 3^3 * x3 + 4^3 * x4 + 5^3 * x5 = a^2) ∧
     (1^5 * x1 + 2^5 * x2 + 3^5 * x3 + 4^5 * x4 + 5^5 * x5 = a^3)))
  → (a = 1 ∨ a = 4 ∨ a = 9 ∨ a = 16 ∨ a = 25) :=
begin
  sorry
end

end determine_a_l714_714040


namespace student_failed_by_l714_714675

-- Conditions
def total_marks : ℕ := 440
def passing_percentage : ℝ := 0.50
def marks_obtained : ℕ := 200

-- Calculate passing marks
noncomputable def passing_marks : ℝ := passing_percentage * total_marks

-- Definition of the problem to be proved
theorem student_failed_by : passing_marks - marks_obtained = 20 := 
by
  sorry

end student_failed_by_l714_714675


namespace min_operations_to_equal_elements_l714_714944

open Locale Classical

theorem min_operations_to_equal_elements:
  ∀ (a : Fin 2009 → ℕ),
    (∀ i, a i ≤ 100) →
    ∃ (k : ℕ),
      (∀ m : ℕ, m < 2009 → a 0 + m * 2 ≤ a m) →
      ∀ x : Fin 2009 → ℕ,
        (∀ i : Fin 2009, x i ≥ 0) ∧
        (∀ i : Nat, a i + x i = a ((i + 1) % 2009) + x ((i + 1) % 2009)) →
        (∑ i, x i) ≤ k :=
  λ a h₁ =>
    ⟨100 * 1004 * 1005, λ m h₂ h₃ =>
      have h₄ : a 0 + m * 2 ≤ a m := h₂ m (by exact h₂)
      ⟨λ x hx, by 
        admit -- Skipping the detailed proof
        -- Proving that the sum ∑ i has an upper bound 100 * 1004 * 1005
⟩⟩

end min_operations_to_equal_elements_l714_714944


namespace speed_ratio_l714_714641

variables (H D : ℝ)
variables (duck_leaps hen_leaps : ℕ)
-- hen_leaps and duck_leaps denote the leaps taken by hen and duck respectively

-- conditions given
axiom cond1 : hen_leaps = 6 ∧ duck_leaps = 8
axiom cond2 : 4 * D = 3 * H

-- goal to prove
theorem speed_ratio (H D : ℝ) (hen_leaps duck_leaps : ℕ) (cond1 : hen_leaps = 6 ∧ duck_leaps = 8) (cond2 : 4 * D = 3 * H) : 
  (6 * H) = (8 * D) :=
by
  intros
  sorry

end speed_ratio_l714_714641


namespace roy_total_pens_l714_714568

theorem roy_total_pens :
  let blue_pens := 2 in
  let B := 2 * blue_pens in
  let R := 2 * B - 2 in
  blue_pens + B + R = 12 := by
  sorry

end roy_total_pens_l714_714568


namespace imaginary_part_of_one_plus_i_pow_5_l714_714531

noncomputable def imaginary_part_of_complex_power (z : ℂ) (n : ℕ) : ℝ :=
  (z^n).im

theorem imaginary_part_of_one_plus_i_pow_5 : imaginary_part_of_complex_power (1 + complex.I) 5 = -4 := by
  sorry

end imaginary_part_of_one_plus_i_pow_5_l714_714531


namespace monochromatic_triangle_exists_l714_714729

theorem monochromatic_triangle_exists : 
  ∀ (G : simple_graph (fin 17)) (c : G.edge_set → fin 3),
    (∀ (x y : fin 17), x ≠ y → (∃ t : fin 3, c ⟦(x, y)⟧ = t)) →
    ∃ (a b c : fin 17), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                        ∃ t : fin 3, 
                        c ⟦(a, b)⟧ = t ∧ c ⟦(b, c)⟧ = t ∧ c ⟦(c, a)⟧ = t :=
begin
  sorry
end

end monochromatic_triangle_exists_l714_714729


namespace light_year_scientific_notation_l714_714664

def sci_not_eq : Prop := 
  let x := 9500000000000
  let y := 9.5 * 10^12
  x = y

theorem light_year_scientific_notation : sci_not_eq :=
  by sorry

end light_year_scientific_notation_l714_714664


namespace ratio_sub_div_eq_l714_714080

theorem ratio_sub_div_eq 
  (a b : ℚ) 
  (h : a / b = 5 / 2) : 
  (a - b) / a = 3 / 5 := 
sorry

end ratio_sub_div_eq_l714_714080


namespace inhabitants_reach_ball_on_time_l714_714473

theorem inhabitants_reach_ball_on_time
  (kingdom_side_length : ℝ)
  (messenger_sent_at : ℕ)
  (ball_begins_at : ℕ)
  (inhabitant_speed : ℝ)
  (time_available : ℝ)
  (max_distance_within_square : ℝ)
  (H_side_length : kingdom_side_length = 2)
  (H_messenger_time : messenger_sent_at = 12)
  (H_ball_time : ball_begins_at = 19)
  (H_speed : inhabitant_speed = 3)
  (H_time_avail : time_available = 7)
  (H_max_distance : max_distance_within_square = 2 * Real.sqrt 2) :
  ∃ t : ℝ, t ≤ time_available ∧ max_distance_within_square / inhabitant_speed ≤ t :=
by
  -- You would write the proof here.
  sorry

end inhabitants_reach_ball_on_time_l714_714473


namespace prob_female_l714_714487

/-- Define basic probabilities for names and their gender associations -/
variables (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions for the probabilities
axiom h1 : P_Alexander = 3 * P_Alexandra
axiom h2 : P_Yevgeny = 3 * P_Evgenia
axiom h3 : P_Valentin = 1.5 * P_Valentina
axiom h4 : P_Vasily = 49 * P_Vasilisa

/-- The problem we need to prove: the probability that the lot was won by a female student is approximately 0.355 -/
theorem prob_female : 
  let P_female := (P_Alexandra * 1 / 4) + (P_Evgenia * 1 / 4) + (P_Valentina * 1 / 4) + (P_Vasilisa * 1 / 4) in
  abs (P_female - 0.355) < 0.001 :=
sorry

end prob_female_l714_714487


namespace people_eating_both_l714_714863

theorem people_eating_both {a b c : ℕ} (h1 : a = 20) (h2 : b = 11) : c = a - b → c = 9 :=
by {
  intro h3,
  rw [h1, h2] at h3,
  exact h3
}

end people_eating_both_l714_714863


namespace law_of_sines_l714_714570

theorem law_of_sines (a b c : ℝ) (α β γ : ℝ) (S R : ℝ)
  (ha : a = 2 * S / (b * sin γ))
  (hb : b = 2 * S / (a * sin α))
  (hc : c = 2 * S / (a * sin γ))
  (hS : S = 1/2 * a * c * sin γ)
  (hR : R = a / (2 * sin α)) :
  ∃ k : ℝ, k = 2 * R ∧
  a / (sin α) = k ∧ 
  b / (sin β) = k ∧ 
  c / (sin γ) = k ∧
  k = abc / (2 * S) :=
by {
  sorry
}

end law_of_sines_l714_714570


namespace trapezoid_perimeter_230_l714_714142

theorem trapezoid_perimeter_230 (A B C D P Q : Point)
  (h_trapezoid : Trapezoid A B C D)
  (h_base_AD_BC : BaseUnits AD BC)
  (h_AD_ge_BC : Length AD > Length BC)
  (h_BC_len : Length BC = 60)
  (h_altitudes : AltitudesIntersect AD P Q)
  (h_AP : Length AP = 24)
  (h_DQ : Length DQ = 11)
  (h_AB : Length AB = 40)
  (h_CD : Length CD = 35)
  : Perimeter A B C D = 230 := by
  sorry

end trapezoid_perimeter_230_l714_714142


namespace area_of_triangle_l714_714905

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l714_714905


namespace find_a_l714_714457

-- Conditions: x = 5 is a solution to the equation 2x - a = -5
-- We need to prove that a = 15 under these conditions

theorem find_a (x a : ℤ) (h1 : x = 5) (h2 : 2 * x - a = -5) : a = 15 :=
by
  -- We are required to prove the statement, so we skip the proof part here
  sorry

end find_a_l714_714457


namespace least_possible_value_l714_714879

theorem least_possible_value : 
  ∃ (p q : ℚ), 
  (p < y ∧ y < q) →
  (0 < y) →
  (y + 5 + 4y > y + 10) ∧ 
  (y + 5 + y + 10 > 4y) ∧ 
  (4y + y + 10 > y + 5) ∧
  (y + 10 > y + 5) ∧ 
  (y + 10 > 4y) → 
  (q - p = 25/12) :=
by
  sorry

end least_possible_value_l714_714879


namespace eighth_grade_probability_female_win_l714_714493

theorem eighth_grade_probability_female_win:
  let P_Alexandra : ℝ := 1 / 4,
      P_Alexander : ℝ := 3 / 4,
      P_Evgenia : ℝ := 1 / 4,
      P_Yevgeny : ℝ := 3 / 4,
      P_Valentina : ℝ := 2 / 5,
      P_Valentin : ℝ := 3 / 5,
      P_Vasilisa : ℝ := 1 / 50,
      P_Vasily : ℝ := 49 / 50 in
  let P_female : ℝ :=
    1 / 4 * (P_Alexandra + 
             P_Evgenia + 
             P_Valentina + 
             P_Vasilisa) in
  P_female = 1 / 16 + 1 / 48 + 3 / 20 + 1 / 200 :=
sorry

end eighth_grade_probability_female_win_l714_714493


namespace ceiling_of_expression_l714_714055

theorem ceiling_of_expression : 
  (Real.ceil (4 * (8 - (3 / 4)))) = 29 := 
by {
  sorry
}

end ceiling_of_expression_l714_714055


namespace part1_part2_l714_714094

def sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
match n with
| 0     => a 0
| (n+1) => if a.nat_mod 2 = 1 then 3 * (a n) + 5 else let k := (nat.find (λ k, (a n) / 2^k % 2 = 1)) in (a n) / 2^k

def a_n : ℕ → ℕ
| 0     => 11
| (n+1) => sequence a_n n

theorem part1 : a_n 2012 = 152 :=
sorry

noncomputable def find_p : ℕ :=
if (∃ m : ℕ, ∀ n : ℕ, n > m → a_n n = a_n m)
then (a_n (nat.find (λ m, ∀ n : ℕ, n > m → a_n n = a_n m))) else 0

theorem part2 : find_p = 5 ∨ find_p = 1 :=
sorry

end part1_part2_l714_714094


namespace survival_rate_is_100_percent_l714_714638

-- Definitions of conditions
def planted_trees : ℕ := 99
def survived_trees : ℕ := 99

-- Definition of survival rate
def survival_rate : ℕ := (survived_trees * 100) / planted_trees

-- Proof statement
theorem survival_rate_is_100_percent : survival_rate = 100 := by
  sorry

end survival_rate_is_100_percent_l714_714638


namespace unique_natural_number_l714_714062

def product_of_digits (n : ℕ) : ℕ :=
  n.digits 10.prod

theorem unique_natural_number (n : ℕ) :
  product_of_digits n = n^2 - 10 * n - 22 ↔ n = 12 :=
by
   sorry

end unique_natural_number_l714_714062


namespace radius_of_tangent_circle_l714_714658

theorem radius_of_tangent_circle :
  ∀ (r : ℝ), (∃ (R : ℝ), R = 1 + Real.sqrt 2 ∧ r ≠ 0 ∧ (∃ (triangle : Triangle), (IsoscelesRightTriangle triangle) ∧
  ∃ (C : Circle), Circumscribed circle triangle C ∧ Circle.radius C = R ∧
  ∃ (S : Circle), TangentToLegsOfTriangle S triangle ∧ TangentInternally C S ∧ Circle.radius S = r))
  → r = 2 :=
sorry

end radius_of_tangent_circle_l714_714658


namespace integral_evaluation_l714_714710

noncomputable def definite_integral : ℝ :=
  ∫ x in Real.pi / 4..Real.arctan 3, (4 * Real.tan x - 5) / (1 - Real.sin (2 * x) + 4 * Real.cos x^2)

theorem integral_evaluation :
  ∫ x in Real.pi / 4..Real.arctan 3, (4 * Real.tan x - 5) / (1 - Real.sin (2 * x) + 4 * Real.cos x^2) = 2 * Real.log 2 - Real.pi / 8 :=
  by sorry

end integral_evaluation_l714_714710


namespace g_8_value_l714_714039

def g (x : ℝ) : ℝ := -3/2 * x^2 + 2

theorem g_8_value : g 8 = -94 := by
  sorry

end g_8_value_l714_714039


namespace value_of_sums_l714_714419

-- Define the even function property
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = f(x)

-- Define the odd function property for shifted function
def is_shifted_function_odd (f : ℝ → ℝ) (shift : ℝ) : Prop := ∀ x : ℝ, f(-(x + shift)) = -f(x + shift)

-- The main theorem to be proved
theorem value_of_sums (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_shifted_odd : is_shifted_function_odd f 1)
  (h_f_2 : f 2 = -1) :
  (∑ i in Finset.range 2013, f i) = 1 :=
by
  sorry

end value_of_sums_l714_714419


namespace inequality_one_inequality_two_l714_714345

-- First Inequality Problem
theorem inequality_one (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) + (1 / d^2) ≤ (1 / (a^2 * b^2 * c^2 * d^2)) :=
sorry

-- Second Inequality Problem
theorem inequality_two (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + (1 / d^3) ≤ (1 / (a^3 * b^3 * c^3 * d^3)) :=
sorry

end inequality_one_inequality_two_l714_714345


namespace min_set_size_B_l714_714647

-- Definition of real number between 0 and 1, and its decimal expansion subsequence set B
def is_between_zero_and_one (x : ℝ) : Prop := 0 < x ∧ x < 1
def B (x : ℝ) : set (set (fin 6 → ℕ)) := 
  { s | ∃ c : ℕ → ℕ, (∀ n, c n ∈ finset.range 10) ∧ 
                      (x = ∑ n, c n * (10 : ℝ) ^(-n - 1)) ∧
                      s ⊆ set.range (fin 6 → ℕ) ∧ 
                      (∀ f : fin 6 → ℕ, f ∈ s ↔ ∃ k, ∀ i j, i < 6 → j < 6 → c (k + i) = f i) }

-- Proposition formalizing the problem statement
theorem min_set_size_B (x : ℝ) (hx_irrational : irrational x) (hx_interval : is_between_zero_and_one x) : 
  ∃ (n : ℕ), n = 7 ∧ ∀ s ∈ B(x), finset.size s ≥ n :=
sorry

end min_set_size_B_l714_714647


namespace trailing_zeroes_60_plus_120_l714_714843

theorem trailing_zeroes_60_plus_120 :
  let n := (nat.factorial 60) + (nat.factorial 120) in
  ∃ k : ℕ, (nat.digits 10 n).length - (nat.prev 10 n).length = 14 :=
by sorry

end trailing_zeroes_60_plus_120_l714_714843


namespace simplify_trig_expression_l714_714976

theorem simplify_trig_expression (x : ℝ) (hx : x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := 
by 
  sorry

end simplify_trig_expression_l714_714976


namespace journey_time_equality_l714_714522

variables {v : ℝ} (h : v > 0)

theorem journey_time_equality (v : ℝ) (hv : v > 0) :
  let t1 := 80 / v
  let t2 := 160 / (2 * v)
  t1 = t2 :=
by
  sorry

end journey_time_equality_l714_714522


namespace infinite_t_no_zero_same_digit_sum_l714_714950

open Nat

-- Definition: digit_sum of a natural number
def digit_sum (n : ℕ) : ℕ := 
  (n.digits 10).sum

-- Predicate: t has no zeros in its decimal representation
def no_zeros (t : ℕ) : Prop := 
  ¬(t.digits 10).any (λ d, d = 0)

-- Lean 4 statement
theorem infinite_t_no_zero_same_digit_sum (k : ℕ) : 
  ∃ᶠ t in atTop, no_zeros t ∧ digit_sum t = digit_sum (k * t) := 
sorry

end infinite_t_no_zero_same_digit_sum_l714_714950


namespace simplify_trigonometric_expression_l714_714969

theorem simplify_trigonometric_expression (x : ℝ) (hx : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 / sin x := 
by sorry

end simplify_trigonometric_expression_l714_714969


namespace final_state_of_marbles_after_operations_l714_714186

theorem final_state_of_marbles_after_operations :
  ∃ (b w : ℕ), b + w = 2 ∧ w = 2 ∧ (∀ n : ℕ, n % 2 = 0 → n = 100 - k * 2) :=
sorry

end final_state_of_marbles_after_operations_l714_714186


namespace find_n_l714_714260

-- Definitions of the log terms and arithmetic sequence conditions
variable {a b : ℝ}

def term1 := log (a^2 * b^4)
def term2 := log (a^6 * b^9)
def term3 := log (a^10 * b^14)

-- Using A = log(a) and B = log(b) for simplicity in arithmetic operations
def A := log a
def B := log b

-- Condition that confirms the terms form an arithmetic sequence
def is_arithmetic_sequence :=
  (term2 - term1 = 4 * A + 5 * B) ∧ (term3 - term2 = 4 * A + 5 * B)

-- General term of the arithmetic sequence
def general_term (k : ℕ) := (2 * A + 4 * B) + (k - 1) * (4 * A + 5 * B)

-- The 10th term of the sequence
def T10 := general_term 10

-- Prove that the 10th term is log(a^n) for some n and find that n
theorem find_n (n : ℕ) (h : is_arithmetic_sequence)
  (h10 : T10 = log (a^n)) : n = 38 :=
  sorry

end find_n_l714_714260


namespace simplify_trig_l714_714972

theorem simplify_trig (x : ℝ) (h_cos_sin : cos x ≠ -1) (h_sin_ne_zero : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
    sorry

end simplify_trig_l714_714972


namespace units_digit_S_2016_l714_714217

def sequence (n : ℕ) : ℕ :=
match n with
| 0       => 6
| (n + 1) => ⌊(5/4) * sequence n + (3/4) * (sequence n)^2.sqrt⌋

def sum_of_sequence (m : ℕ) : ℕ :=
(m + 1).sum (fun k => sequence k)

theorem units_digit_S_2016 : 
  (sum_of_sequence 2015 % 10) = 1 :=
sorry

end units_digit_S_2016_l714_714217


namespace simplify_trig_expression_l714_714981

theorem simplify_trig_expression (x : ℝ) (hx : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l714_714981


namespace ceiling_of_expression_l714_714054

theorem ceiling_of_expression : 
  (Real.ceil (4 * (8 - (3 / 4)))) = 29 := 
by {
  sorry
}

end ceiling_of_expression_l714_714054


namespace freds_bank_passwords_l714_714349

theorem freds_bank_passwords : 
  let total_passwords := 10 ^ 5 in
  let restricted_78 := 10 ^ 3 in
  let restricted_90 := 10 ^ 3 in
  let total_restricted := restricted_78 + restricted_90 in
  let valid_passwords := total_passwords - total_restricted in
  valid_passwords = 98000 :=
by 
  let total_passwords := 10 ^ 5;
  let restricted_78 := 10 ^ 3;
  let restricted_90 := 10 ^ 3;
  let total_restricted := restricted_78 + restricted_90;
  let valid_passwords := total_passwords - total_restricted;
  exact sorry

end freds_bank_passwords_l714_714349


namespace problem1_problem2_l714_714451

def op_add (x y : ℝ) : ℝ := (1 / (x - y)) + y

theorem problem1 : op_add 2 (-3) = -14 / 5 :=
  sorry

theorem problem2 : op_add (op_add (-4) (-1)) (-5) = -52 / 11 :=
  sorry

end problem1_problem2_l714_714451


namespace minimum_value_proof_l714_714525

noncomputable def minimum_integral_value (a b : ℝ) : ℝ :=
  ∫ x in (0 : ℝ) .. 1, abs((x - a) * (x - b))

theorem minimum_value_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ 1) : 
  minimum_integral_value a b = 1 / 12 :=
by
  sorry

end minimum_value_proof_l714_714525


namespace eval_piecewise_function_l714_714428

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then log (x) / log (1 / 2)
  else 1 - (2 ^ x)

theorem eval_piecewise_function :
  f (f 2) = 1 / 2 :=
by
  sorry

end eval_piecewise_function_l714_714428


namespace cos_120_eq_neg_half_l714_714016

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714016


namespace roger_gave_candies_l714_714954

theorem roger_gave_candies :
  ∀ (original_candies : ℕ) (remaining_candies : ℕ) (given_candies : ℕ),
  original_candies = 95 → remaining_candies = 92 → given_candies = original_candies - remaining_candies → given_candies = 3 :=
by
  intros
  sorry

end roger_gave_candies_l714_714954


namespace problem1_problem2_l714_714310

theorem problem1 :
  (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (60 * Real.pi / 180) - abs (1 - Real.sqrt 3) = 3 :=
by 
  sorry

theorem problem2 (x : ℝ) :
  (2 / (x + 1) + 1 = x / (x - 1)) → x = 3 :=
by 
  sorry

end problem1_problem2_l714_714310


namespace probability_female_win_l714_714499

variable (P_Alexandr P_Alexandra P_Evgeniev P_Evgenii P_Valentinov P_Valentin P_Vasilev P_Vasilisa : ℝ)

-- Conditions
axiom h1 : P_Alexandr = 3 * P_Alexandra
axiom h2 : P_Evgeniev = (1 / 3) * P_Evgenii
axiom h3 : P_Valentinov = (1.5) * P_Valentin
axiom h4 : P_Vasilev = 49 * P_Vasilisa
axiom h5 : P_Alexandr + P_Alexandra = 1
axiom h6 : P_Evgeniev + P_Evgenii = 1
axiom h7 : P_Valentinov + P_Valentin = 1
axiom h8 : P_Vasilev + P_Vasilisa = 1

-- Statement to prove
theorem probability_female_win : 
  let P_female := (1/4) * P_Alexandra + (1/4) * P_Evgeniev + (1/4) * P_Valentinov + (1/4) * P_Vasilisa in
  P_female = 0.355 :=
by
  sorry

end probability_female_win_l714_714499


namespace female_wins_probability_l714_714496

theorem female_wins_probability :
  let p_alexandr := 3 * p_alexandra,
      p_evgeniev := (1 / 3) * p_evgenii,
      p_valentinov := (3 / 2) * p_valentin,
      p_vasilev := 49 * p_vasilisa,
      p_alexandra := 1 / 4,
      p_alexandr := 3 / 4,
      p_evgeniev := 1 / 12,
      p_evgenii := 11 / 12,
      p_valentinov := 3 / 5,
      p_valentin := 2 / 5,
      p_vasilev := 49 / 50,
      p_vasilisa := 1 / 50,
      p_female := 
        (1 / 4) * p_alexandra + 
        (1 / 4) * p_evgeniev + 
        (1 / 4) * p_valentinov + 
        (1 / 4) * p_vasilisa 
  in p_female ≈ 0.355 := 
sorry

end female_wins_probability_l714_714496


namespace weight_of_new_person_weight_difference_l714_714478

variable {W : ℝ} 
variable {w_new : ℝ}
variable {group_weight_before : ℝ := W + 65}
variable {group_weight_after : ℝ := W + 65 - 65 + 128}

theorem weight_of_new_person (avg_increase : ℝ) (fixed_weight : ℝ) (group_size : ℕ) :
  avg_increase = 6.3 → fixed_weight = 65 → group_size = 10 → 
  let weight_of_old_person := fixed_weight in
  let total_weight_before := W + weight_of_old_person in
  let total_weight_after := total_weight_before + avg_increase * group_size in 
  let new_person_weight := total_weight_after - (total_weight_before - weight_of_old_person) in
  new_person_weight = 128 :=
begin
  intros,
  let weight_of_old_person := fixed_weight,
  let total_weight_before := W + weight_of_old_person,
  let total_weight_after := total_weight_before + avg_increase * group_size,
  let new_person_weight := total_weight_after - (total_weight_before - weight_of_old_person),
  rw [avg_increase, fixed_weight, group_size],
  calc new_person_weight = total_weight_after - (total_weight_before - weight_of_old_person) : by sorry
                     ... = W + 65 + 63 - 65 : by sorry
                     ... = 128 : by sorry
end

theorem weight_difference (avg_increase : ℝ) (fixed_weight : ℝ) (group_size : ℕ) :
  avg_increase = 6.3 → fixed_weight = 65 → group_size = 10 → 
  let weight_of_old_person := fixed_weight in
  let total_weight_before := W + weight_of_old_person in
  let total_weight_after := total_weight_before + avg_increase * group_size in 
  let weight_difference := total_weight_after - total_weight_before in
  weight_difference = 63 :=
begin
  intros,
  let weight_of_old_person := fixed_weight,
  let total_weight_before := W + weight_of_old_person,
  let total_weight_after := total_weight_before + avg_increase * group_size,
  let weight_difference := total_weight_after - total_weight_before,
  rw [avg_increase, fixed_weight, group_size],
  calc weight_difference = total_weight_after - total_weight_before : by sorry
                     ... = (W + 65 + 63) - (W + 65) : by sorry
                     ... = 63 : by sorry
end

end weight_of_new_person_weight_difference_l714_714478


namespace intersection_point_distance_to_center_l714_714474

noncomputable def circle_radius : ℝ := 5
noncomputable def chord_AB_len : ℝ := 9
noncomputable def chord_CD_len : ℝ := 8
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem intersection_point_distance_to_center :
  let SQ := sqrt (circle_radius^2 - (chord_AB_len / 2)^2),
      SR := sqrt (circle_radius^2 - (chord_CD_len / 2)^2)
  in distance SQ SR = sqrt 55 / 2 :=
sorry

end intersection_point_distance_to_center_l714_714474


namespace perpendicular_condition_l714_714313

theorem perpendicular_condition (a : ℝ) :
  (a = 1) ↔ (∀ x : ℝ, (a*x + 1 - ((a - 2)*x - 1)) * ((a * x + 1 - (a * x + 1))) = 0) :=
by
  sorry

end perpendicular_condition_l714_714313


namespace compare_Bn_Cn_l714_714100

variable {a : ℝ} (h_a : 0 < a)

def arithmetic_sequence (n : ℕ) : ℝ := a * n
def sum_first_n_terms (n : ℕ) : ℝ := a * n * (n + 1) / 2

def b_n (n : ℕ) : ℝ := 1 / sum_first_n_terms n
def c_n (n : ℕ) : ℝ := 1 / (a * 2 ^ (n - 1))
def B_n (n : ℕ) : ℝ := (Finset.range n).sum (b_n)
def C_n (n : ℕ) : ℝ := (Finset.range n).sum (c_n)

theorem compare_Bn_Cn (n : ℕ) (h_n : 2 ≤ n) : B_n n < C_n n :=
by
  sorry

end compare_Bn_Cn_l714_714100


namespace rate_second_year_l714_714384

/-- Define the principal amount at the start. -/
def P : ℝ := 4000

/-- Define the rate of interest for the first year. -/
def rate_first_year : ℝ := 0.04

/-- Define the final amount after 2 years. -/
def A : ℝ := 4368

/-- Define the amount after the first year. -/
def P1 : ℝ := P + P * rate_first_year

/-- Define the interest for the second year. -/
def Interest2 : ℝ := A - P1

/-- Define the principal amount for the second year, which is the amount after the first year. -/
def P2 : ℝ := P1

/-- Prove that the rate of interest for the second year is 5%. -/
theorem rate_second_year : (Interest2 / P2) * 100 = 5 :=
by
  sorry

end rate_second_year_l714_714384


namespace add_terms_increasing_n_l714_714179

theorem add_terms_increasing_n (n : ℕ) (h : 0 < n) :
  ((1 - (1/2) + (1/3) - (1/4) + ... + (1/(2*n-1)) - (1/(2*n)))
  + ((1/(2*n+1)) - (1/(2*n+2)))) =
  (1 - (1/2) + (1/3) - (1/4) + ... + (1/(2*(n+1)-1)) - (1/(2*(n+1)))) := sorry

end add_terms_increasing_n_l714_714179


namespace probability_positive_factor_less_than_8_l714_714632

theorem probability_positive_factor_less_than_8 (n : ℕ) (h : n = 72) : 
  let factors := [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72]
  let count_less_than_8 := 5
  let total_factors := factors.length
  let probability := (count_less_than_8 : ℚ) / total_factors
  probability = 5 / 12 :=
begin
  sorry
end

end probability_positive_factor_less_than_8_l714_714632


namespace count_five_letter_words_l714_714149

theorem count_five_letter_words : (∃ (A1 A5 : string), A1 = "A" ∧ A5 = "A" ∧
                                    ∃ (M2 M3 M4 : string), 
                                      (M2.length = 1 ∧ M3.length = 1 ∧ M4.length = 1 ∧
                                       ∀ ch, ch ∈ M2 ∧ ch ∈ M3 ∧ ch ∈ M4 → ch.is_alpha ∧ ch.is_upper)) → 
                                  26^3 = 17576 :=
by sorry

end count_five_letter_words_l714_714149


namespace horse_cow_difference_l714_714943

def initial_conditions (h c : ℕ) : Prop :=
  4 * c = h

def transaction (h c : ℕ) : Prop :=
  (h - 15) * 7 = (c + 15) * 13

def final_difference (h c : ℕ) : Prop := 
  h - 15 - (c + 15) = 30

theorem horse_cow_difference (h c : ℕ) (hc : initial_conditions h c) (ht : transaction h c) : final_difference h c :=
    by
      sorry

end horse_cow_difference_l714_714943


namespace vacation_days_l714_714994

theorem vacation_days (total_miles miles_per_day : ℕ) 
  (h1 : total_miles = 1250) (h2 : miles_per_day = 250) :
  total_miles / miles_per_day = 5 := by
  sorry

end vacation_days_l714_714994


namespace subset_existence_l714_714895

theorem subset_existence (n : ℕ) (h : n ≥ 2) : 
  ∃ (S : ℕ → set ℕ), 
    (∀ i j, i ≠ j → disjoint (S i) (S j)) ∧
    (∀ i, S i ≠ ∅) ∧
    (∀ m, ∃! A : finset ℕ, (∀ k ∈ A, ∃ i, k ∈ S i) ∧ m = A.sum id ∧ A.card ≤ n) :=
sorry

end subset_existence_l714_714895


namespace comparison_l714_714695

def sequence (α : ℕ → ℕ) : ℕ → ℚ
| 1     := 1 + 1 / (α 1)
| (n+1) := 1 + 1 / (α 1 + sequence (λ k, α (k+1) n))

theorem comparison (α : ℕ → ℕ) (h : ∀ k, 1 ≤ α k) :
  sequence α 4 < sequence α 7 := 
sorry

end comparison_l714_714695


namespace part_a_not_unique_minimum_part_b_compare_c_m_l714_714097

variable (n : ℕ)
variable (x : fin n → ℝ)

noncomputable def d (t : ℝ) : ℝ :=
  (min (Finset.image (λ i, |x i - t|) Finset.univ) + max (Finset.image (λ i, |x i - t|) Finset.univ)) / 2

theorem part_a_not_unique_minimum :
  ¬ ∀ s : fin n → ℝ, ∃! t : ℝ, is_min {t | ∃ t', d x t' = d x t} t :=
sorry

def c := (Finset.min' (Finset.image (λ i, x i) Finset.univ) ⟪0⟫ + Finset.max' (Finset.image (λ i, x i) Finset.univ) ⟪0⟫) / 2

def m := Median.univ (Finset.image (λ i, x i) Finset.univ)

theorem part_b_compare_c_m :
  d x c ≤ d x m :=
sorry

end part_a_not_unique_minimum_part_b_compare_c_m_l714_714097


namespace deriv_exponential_power_fn_l714_714759

variables {x : ℝ} (f φ : ℝ → ℝ)

noncomputable def u := f x
noncomputable def v := φ x
noncomputable def y := (u f φ) ^ (v f φ)
noncomputable def u' := deriv f x
noncomputable def v' := deriv φ x

theorem deriv_exponential_power_fn (h : f x > 0) :
  deriv y x = (u f φ) ^ (v f φ) * (v' f φ * log (u f φ) + (v f φ) * (u' f φ) / (u f φ)) :=
by
  sorry

end deriv_exponential_power_fn_l714_714759


namespace fiftieth_number_is_31254_l714_714597

open List

-- Define the list of digits
def digits := [1, 2, 3, 4, 5]

-- All permutations of digits
def permutations := List.perms digits

-- Function to convert a list of digits to a number
def listToNumber (l : List Nat) : Nat :=
  l.foldl (λ acc d => 10 * acc + d) 0

-- Define the ordered list of numbers
def orderedNumbers := permutations.map listToNumber |>.sort (≤)

-- The 50th integer in the list
noncomputable def fiftiethNumber := orderedNumbers.get (50 - 1) (by sorry)

-- Prove that the 50th integer is 31254
theorem fiftieth_number_is_31254 : fiftiethNumber = 31254 :=
  by
  sorry

end fiftieth_number_is_31254_l714_714597


namespace eval_inverse_sum_l714_714530

variable (g : ℕ → ℕ)
variable [invertible g]

def g_condition_1 : Prop := g 4 = 7
def g_condition_2 : Prop := g 6 = 2
def g_condition_3 : Prop := g 3 = 6

theorem eval_inverse_sum :
  g_condition_1 g →
  g_condition_2 g →
  g_condition_3 g →
  g⁻¹ (g⁻¹ 6 + g⁻¹ 7 - 1) = 3 :=
by
  sorry

end eval_inverse_sum_l714_714530


namespace percentage_caught_customers_l714_714859

noncomputable def total_sampling_percentage : ℝ := 0.25
noncomputable def caught_percentage : ℝ := 0.88

theorem percentage_caught_customers :
  total_sampling_percentage * caught_percentage = 0.22 :=
by
  sorry

end percentage_caught_customers_l714_714859


namespace smallest_m_divisible_by_15_l714_714913

-- Define conditions
def is_largest_prime_2011_digit (q : ℕ) : Prop :=
  prime q ∧ (∃ p : ℕ, prime p ∧ (number_of_digits p = 2011 ∧ p > q))

def number_of_digits (n : ℕ) : ℕ :=
  nat.log10 n + 1

-- Main theorem statement
theorem smallest_m_divisible_by_15 (q : ℕ) (h : is_largest_prime_2011_digit q) :
  ∃ m : ℕ, (q^2 - m) % 15 = 0 ∧ m = 1 :=
sorry

end smallest_m_divisible_by_15_l714_714913


namespace max_value_g_eq_3_in_interval_l714_714367

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_g_eq_3_in_interval : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3) ∧ (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3) :=
by
  sorry

end max_value_g_eq_3_in_interval_l714_714367


namespace second_row_same_as_first_l714_714312

variable {α : Type} [linear_order α] [decidable_eq α]

theorem second_row_same_as_first {n : ℕ} (a : fin n → α) 
  (h_increasing : ∀ i j : fin n, i < j → a i < a j ) 
  (b : fin n → α) 
  (h_permutation : ∀ i : fin n, ∃ j : fin n, b i = a j)
  (h_third_row_increasing : ∀ i j : fin n, i < j → a i + b i < a j + b j) 
: ∀ i : fin n, b i = a i := 
sorry

end second_row_same_as_first_l714_714312


namespace triangle_shape_l714_714883

-- Defining the conditions:
variables (A B C a b c : ℝ)
variable (h1 : c - a * Real.cos B = (2 * a - b) * Real.cos A)

-- Defining the property to prove:
theorem triangle_shape : 
  (A = Real.pi / 2 ∨ A = B ∨ B = C ∨ C = A + B) :=
sorry

end triangle_shape_l714_714883


namespace never_2003_pieces_l714_714297

theorem never_2003_pieces :
  ¬∃ n : ℕ, (n = 5 + 4 * k) ∧ (n = 2003) :=
by
  sorry

end never_2003_pieces_l714_714297


namespace min_sum_of_distances_l714_714137

noncomputable def distance_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / sqrt (a^2 + b^2)

noncomputable def sum_of_distances (a : ℝ) : ℝ :=
  let P := (a^2, 2 * a)
  distance_point_to_line P 4 (-3) 6 + abs (P.1 + 1)

theorem min_sum_of_distances : ∃ a: ℝ, sum_of_distances a = 2 :=
by
  sorry

end min_sum_of_distances_l714_714137


namespace probability_rain_all_three_days_l714_714262

-- Define the probabilities as constant values
def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.3
def prob_rain_sunday_given_fri_sat : ℝ := 0.6

-- Define the probability of raining all three days considering the conditional probabilities
def prob_rain_all_three_days : ℝ :=
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday_given_fri_sat

-- Prove that the probability of rain on all three days is 12%
theorem probability_rain_all_three_days : prob_rain_all_three_days = 0.12 :=
by
  sorry

end probability_rain_all_three_days_l714_714262


namespace cos_120_degrees_l714_714002

theorem cos_120_degrees : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cos_120_degrees_l714_714002


namespace range_of_a_l714_714820

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ 2 * a - (1 / 2) * a^2) ↔ 0 ≤ a :=
by
  sorry

end range_of_a_l714_714820


namespace probability_female_win_l714_714498

variable (P_Alexandr P_Alexandra P_Evgeniev P_Evgenii P_Valentinov P_Valentin P_Vasilev P_Vasilisa : ℝ)

-- Conditions
axiom h1 : P_Alexandr = 3 * P_Alexandra
axiom h2 : P_Evgeniev = (1 / 3) * P_Evgenii
axiom h3 : P_Valentinov = (1.5) * P_Valentin
axiom h4 : P_Vasilev = 49 * P_Vasilisa
axiom h5 : P_Alexandr + P_Alexandra = 1
axiom h6 : P_Evgeniev + P_Evgenii = 1
axiom h7 : P_Valentinov + P_Valentin = 1
axiom h8 : P_Vasilev + P_Vasilisa = 1

-- Statement to prove
theorem probability_female_win : 
  let P_female := (1/4) * P_Alexandra + (1/4) * P_Evgeniev + (1/4) * P_Valentinov + (1/4) * P_Vasilisa in
  P_female = 0.355 :=
by
  sorry

end probability_female_win_l714_714498


namespace sum_of_differences_is_l714_714108

-- Definitions
def a (n : ℕ) : ℚ :=
  match n with
  | 1   => 1 - 1/3
  | 2   => 1/2 - 1/4
  | _  => 1/n - 1/(n + 2)

-- Theorem statement
theorem sum_of_differences_is (sum : ℚ) :
  (sum = (Finset.sum (Finset.range 98) (λ i, a (i + 1)))) → sum = 9701 / 9900 :=
by
  -- Proof obviously goes here, but we skip it according to instructions.
  intros
  sorry

end sum_of_differences_is_l714_714108


namespace f_monotonic_f_odd_find_a_k_range_l714_714432
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a

-- (1) Prove the monotonicity of the function f
theorem f_monotonic (a : ℝ) : ∀ {x y : ℝ}, x < y → f a x < f a y := sorry

-- (2) If f is an odd function, find the value of the real number a
theorem f_odd_find_a : ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -f a x) → a = -1/2 := sorry

-- (3) Under the condition in (2), if the inequality holds for all x ∈ ℝ, find the range of values for k
theorem k_range (k : ℝ) :
  (∀ x : ℝ, f (-1/2) (x^2 - 2*x) + f (-1/2) (2*x^2 - k) > 0) → k < -1/3 := sorry

end f_monotonic_f_odd_find_a_k_range_l714_714432


namespace probability_number_l714_714259

-- Let P be the number representing the likelihood of a random event occurring.
def number_represents_likelihood (P : ℝ) : Prop :=
  0 < P ∧ P < 1

-- If a random event is very likely to occur, then its probability value, P, is close to but not equal to 1.
def event_very_likely (P : ℝ) : Prop :=
  0 < P ∧ P < 1 ∧ P ≈ 1

-- Prove that P is the probability given the conditions.
theorem probability_number (P : ℝ) (h : number_represents_likelihood P ∧ event_very_likely P) : 
  ∃ (p : ℝ), p = P :=
sorry

end probability_number_l714_714259


namespace cos_120_eq_neg_half_l714_714020

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714020


namespace relationship_between_a_b_c_l714_714454

noncomputable def a : ℝ := ∫ x in 0..2, x^2
noncomputable def b : ℝ := ∫ x in 0..2, real.exp x
noncomputable def c : ℝ := ∫ x in 0..2, real.sin x

theorem relationship_between_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_between_a_b_c_l714_714454


namespace problem_1_problem_2_problem_3_l714_714654

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) : (∀ x : ℝ, f (x + 1) = x^2 + 4*x + 1) → (∀ x : ℝ, f x = x^2 + 2*x - 2) :=
by
  intro h
  sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) : (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) → (∀ x : ℝ, f x = x + 3) :=
by
  intros h1 h2
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) : (∀ x : ℝ, 2 * f x + f (1 / x) = 3 * x) → (∀ x : ℝ, f x = 2 * x - 1 / x) :=
by
  intro h
  sorry

end problem_1_problem_2_problem_3_l714_714654


namespace range_of_k_l714_714466

theorem range_of_k (k : ℝ) (H : ∀ x : ℤ, |(x : ℝ) - 1| < k * x ↔ x ∈ ({1, 2, 3} : Set ℤ)) : 
  (2 / 3 : ℝ) < k ∧ k ≤ (3 / 4 : ℝ) :=
by
  sorry

end range_of_k_l714_714466


namespace juanita_spends_more_than_grant_l714_714148

noncomputable def grant_annual_cost_before_discounts : ℝ := 200.0
noncomputable def grant_loyalty_discount : ℝ := 0.10
noncomputable def grant_additional_discount_months : ℕ := 2
noncomputable def grant_additional_discount : ℝ := 0.05

noncomputable def juanita_weekdays_cost : ℕ → ℝ
| 1, 2, 3 => 0.50
| 4, 5 => 0.60
| 6 => 0.80
| 7 => 3.00
| _ => 0

noncomputable def julanita_coupon_savings_per_month : ℝ := 0.25
noncomputable def juanita_holiday_extra_cost_per_month : ℝ := 0.50

-- Grant's total cost after discounts
def grant_annual_cost : ℝ :=
  let after_loyalty_discount := grant_annual_cost_before_discounts * (1 - grant_loyalty_discount)
  let after_additional_discount := after_loyalty_discount - (after_loyalty_discount / 12) * grant_additional_discount_months * grant_additional_discount
  after_additional_discount

-- Juanita's total weekly cost
def juanita_weekly_cost : ℝ :=
  (Juanita_weekdays_cost 1 + Juanita_weekdays_cost 2 + Juanita_weekdays_cost 3 +
  Juanita_weekdays_cost 4 + Juanita_weekdays_cost 5 +
  Juanita_weekdays_cost 6 + Juanita_weekdays_cost 7)

-- Juanita's total annual cost
def juanita_annual_cost : ℝ :=
  (juanita_weekly_cost * 52) - (juanita_coupon_savings_per_month * 12) + (juanita_holiday_extra_cost_per_month * 12)

-- Prove that Juanita spends $162.50 more than Grant
theorem juanita_spends_more_than_grant : juanita_annual_cost - grant_annual_cost = 162.50 :=
by
  sorry

end juanita_spends_more_than_grant_l714_714148


namespace photo_size_l714_714333

def drive_capacity (P : ℕ) : ℕ := 2000 * P
def space_taken_by_400_photos (P : ℕ) : ℕ := 400 * P
def space_required_for_videos : ℕ := 12 * 200

theorem photo_size (P : ℚ) (h₁ : drive_capacity P = 2000 * P) 
  (h₂ : 2000 * P - 400 * P  = 2400) : 
  P = 1.5 := by
  sorry

end photo_size_l714_714333


namespace increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l714_714134

noncomputable def f (x : ℝ) := x ^ 2 * Real.exp x - Real.log x

theorem increasing_f_for_x_ge_1 : ∀ (x : ℝ), x ≥ 1 → ∀ y > x, f y > f x :=
by
  sorry

theorem f_gt_1_for_x_gt_0 : ∀ (x : ℝ), x > 0 → f x > 1 :=
by
  sorry

end increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l714_714134


namespace distance_inequality_equality_condition_l714_714998

variables (A B C D : ℝ^3)

theorem distance_inequality (h : (A - C).norm_sq + (B - D).norm_sq + (A - D).norm_sq + (B - C).norm_sq ≥ (A - B).norm_sq + (C - D).norm_sq) : 
  (A - C).dist^2 + (B - D).dist^2 + (A - D).dist^2 + (B - C).dist^2 ≥ (A - B).dist^2 + (C - D).dist^2 :=
by 
  sorry

theorem equality_condition (h : (A - C) = (D - B)) : 
  (A - C).dist^2 + (B - D).dist^2 + (A - D).dist^2 + (B - C).dist^2 = (A - B).dist^2 + (C - D).dist^2 :=
by 
  sorry

end distance_inequality_equality_condition_l714_714998


namespace math_problem_l714_714036

noncomputable def a : ℝ := 3.67
noncomputable def b : ℝ := 4.83
noncomputable def c : ℝ := 2.57
noncomputable def d : ℝ := -0.12
noncomputable def x : ℝ := 7.25
noncomputable def y : ℝ := -0.55

theorem math_problem :
  (3 * a * (4 * b - 2 * y)^2) / (5 * c * d^3 * 0.5 * x) - (2 * x * y^3) / (a * b^2 * c) = -57.179729 := 
sorry

end math_problem_l714_714036


namespace find_T_2017_l714_714095

-- Define the sequence a_n and sum S_n
def seq_a_n (n : ℕ) : ℝ := sorry
def sum_S_n (n : ℕ) : ℝ := (-1 : ℝ) ^ n * seq_a_n n + 1 / 2 ^ n

-- Define T_n as the sum of the first n terms of S_n
def sum_T_n (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), sum_S_n i

-- The main theorem we need to prove
theorem find_T_2017 : sum_T_n 2017 = (1 / 3) * (1 - (1 / 2) ^ 2016) :=
sorry

end find_T_2017_l714_714095


namespace prob_all_females_l714_714781

open Finset

variable (students : Finset ℕ) (males : Finset ℕ) (females : Finset ℕ)

def num_males : ℕ := 5
def num_females : ℕ := 4

-- Defining the total students as a finite set of 9 distinct elements
def total_students : Finset ℕ := (range (num_males + num_females)).erase 0

-- Defining the males as a finite set of the first 5 distinct elements
def male_students : Finset ℕ := (range num_males).erase 0

-- Defining the females as a finite set of the next 4 distinct elements
def female_students : Finset ℕ := ((range (num_males + num_females)).filter (λ x, x ≥ num_males))

-- Defining combinations
def choose (n k : ℕ) : ℕ := (range n).powerset.filter (λ s, s.card = k).card

theorem prob_all_females :
  (choose (num_males + num_females) 3) ≠ 0 → 
  (choose num_females 3) / (choose (num_males + num_females) 3) = 1 / 21 := 
by 
  sorry

end prob_all_females_l714_714781


namespace fraction_power_of_6_l714_714751

theorem fraction_power_of_6 : 
  ∃ (s : ℕ → ℚ), (∀ n, s n = 2 / (6 ^ (n + 1)) ∧ s 0 = 3 / 6 + 2 / 6) ∧ has_sum s (9 / 10) :=
by
  sorry

end fraction_power_of_6_l714_714751


namespace simplify_trig_expression_l714_714975

theorem simplify_trig_expression (x : ℝ) (hx : x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := 
by 
  sorry

end simplify_trig_expression_l714_714975


namespace proof_equiv_AF_eq_CD_l714_714836

-- Definitions and assumptions based on the given problem
variables {A B C D F E : Point}
variable [semicircle : is_semicircle A B C]
variable (hAC_lt_BC : AC < BC)
variable (D_on_BC : lies_on D (line_segment B C))
variable (BD_eq_AC : distance B D = distance A C)
variable (F_on_AC : lies_on F (line_segment A C))
variable (intersect_BF_AD_at_E : intersection (line B F) (line A D) = E)
variable (angle_BED_45 : angle B E D = 45)

-- Goal: Prove that AF = CD
theorem proof_equiv_AF_eq_CD :
  distance A F = distance C D :=
sorry

end proof_equiv_AF_eq_CD_l714_714836


namespace area_of_transformed_triangle_l714_714993

noncomputable def area_of_triangle := 50

theorem area_of_transformed_triangle (a b c : ℝ) (g : ℝ → ℝ) 
  (h_domain : ∀ x, x ∈ {a, b, c} → ∃ y, y = g x) 
  (h_area : ∃ (p1 p2 p3 : ℝ × ℝ), p1 = (a, g a) ∧ p2 = (b, g b) ∧ p3 = (c, g c) ∧ 
      (1 / 2) * abs ((b - a) * (g c - g a) - (c - a) * (g b - g a)) = area_of_triangle) :
  ∃ (q1 q2 q3 : ℝ × ℝ), q1 = (a / 3, 3 * g a) ∧ q2 = (b / 3, 3 * g b) ∧ q3 = (c / 3, 3 * g c) ∧ 
      (1 / 2) * abs ((b / 3 - a / 3) * (3 * g c - 3 * g a) - (c / 3 - a / 3) * (3 * g b - 3 * g a)) = area_of_triangle :=
by
  sorry

end area_of_transformed_triangle_l714_714993


namespace no_such_natural_numbers_exist_l714_714727

theorem no_such_natural_numbers_exist :
  ¬ ∃ (x y : ℕ), ∃ (k m : ℕ), x^2 + x + 1 = y^k ∧ y^2 + y + 1 = x^m := 
by sorry

end no_such_natural_numbers_exist_l714_714727


namespace red_peaches_per_basket_l714_714374

theorem red_peaches_per_basket (R : ℕ) (green_peaches_per_basket : ℕ) (number_of_baskets : ℕ) (total_peaches : ℕ) (h1 : green_peaches_per_basket = 4) (h2 : number_of_baskets = 15) (h3 : total_peaches = 345) : R = 19 :=
by
  sorry

end red_peaches_per_basket_l714_714374


namespace simplify_trig_expression_l714_714980

theorem simplify_trig_expression (x : ℝ) (hx : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l714_714980


namespace value_of_t_l714_714410

noncomputable def f (x t k : ℝ) : ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem value_of_t (a b t k : ℝ) (h1 : 0 < t) (h2 : 0 < k) 
  (h3 : a + b = t) (h4 : a * b = k) (h5 : 2 * a = b - 2) (h6 : (-2)^2 = a * b) : 
  t = 5 := 
  sorry

end value_of_t_l714_714410


namespace measure_angle_B_triangle_area_correct_l714_714414

noncomputable def triangle_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) → B = Real.pi / 3

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area1 := (3 + Real.sqrt 3)
  let area2 := Real.sqrt 3
  let m : ℝ × ℝ := (Real.sin C - Real.sin A, Real.sin C - Real.sin B)
  let n : ℝ × ℝ := (b + c, a)
  a * (Real.sin C - Real.sin A) = (b + c) * (Real.sin C - Real.sin B) →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  let sinA1 := (Real.sqrt 2 / 2)
  let sinA2 := (Real.sqrt 6 - Real.sqrt 2) / 4
  let S1 := (1 / 2) * b * c * sinA1
  let S2 := (1 / 2) * b * c * sinA2
  S1 = area1 ∨ S2 = area2

theorem measure_angle_B :
  ∀ (a b c A B C : ℝ),
    triangle_angle_B a b c A B C := sorry

theorem triangle_area_correct :
  ∀ (a b c A B C : ℝ),
    triangle_area a b c A B C := sorry

end measure_angle_B_triangle_area_correct_l714_714414


namespace series_value_l714_714202

noncomputable def sum_series (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) : ℝ :=
∑' n : ℕ, (if h : n > 0 then
             1 / (((n - 1) * c - (n - 2) * b) * (n * c - (n - 1) * a))
           else 
             0)

theorem series_value (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) :
  sum_series a b c h_positivity h_order = 1 / ((c - a) * b) :=
by
  sorry

end series_value_l714_714202


namespace magnitude_of_z_is_sqrt_10_l714_714426

-- Define the complex number z
def z : ℂ := (1 + Complex.i) * (2 - Complex.i)

-- Define the expected magnitude |z|
def expected_magnitude : ℝ := Real.sqrt 10

-- Prove that the magnitude of z is equal to the expected magnitude
theorem magnitude_of_z_is_sqrt_10 : Complex.abs z = expected_magnitude := by
  sorry

end magnitude_of_z_is_sqrt_10_l714_714426


namespace relay_team_order_l714_714192

theorem relay_team_order (team : Fin 5 → String) (jordan : Fin 5) (remaining : Fin 4 → String) :
    jordan = 4 →
    team jordan = "Jordan" →
    (∀ i, i < 4 → team i ∈ remaining) →
    (∀ i j, i < j ∧ j < 4 → team i ≠ team j) →
    (Π i : Fin 4, team i = remaining i) →
    fintype.card (Fin 4) * fintype.card (Fin 3) * fintype.card (Fin 2) * fintype.card (Fin 1) = 24 := by
  sorry

end relay_team_order_l714_714192


namespace brother_money_left_l714_714930

theorem brother_money_left (michael_money : ℕ) (brother_initial_money : ℕ) (candy_cost : ℕ) :
  michael_money = 42 → brother_initial_money = 17 → candy_cost = 3 →
  brother_initial_money + michael_money / 2 - candy_cost = 35 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end brother_money_left_l714_714930


namespace max_sqrt_expr_l714_714067

theorem max_sqrt_expr : ∃ x ∈ Icc (0 : ℝ) 17, 
  sqrt (x + 34) + sqrt (17 - x) + sqrt (2 * x) = 12.97 :=
by
  let f := λ (x : ℝ), sqrt (x + 34) + sqrt (17 - x) + sqrt (2 * x)
  have : ∀ x ∈ Icc (0 : ℝ) 17, f x ≤ 12.97,
    -- detailed steps to show f(x) has an upper bound 12.97
    sorry
  have h0 : f 0 = sqrt (34) + sqrt (17) := by
    simp [f, sqrt_nonneg]
  have h17 : f 17 = sqrt (51) + sqrt (34) := by
    simp [f, sqrt_nonneg]
  have h_bounds : sqrt (34) + sqrt (17) ≤ 12.97 ∧ sqrt (51) + sqrt (34) ≤ 12.97 := by
    -- detailed evaluations of sqrt(34) + sqrt(17) and sqrt(51) + sqrt(34)
    sorry
  existsi 17
  split
  · exact right_mem_Icc.mpr (le_refl 17)
  · rw h17
    exact h_bounds.right

end max_sqrt_expr_l714_714067


namespace discount_percentage_l714_714475

theorem discount_percentage (p : ℝ) : 
  (1 + 0.25) * p * (1 - 0.20) = p :=
by
  sorry

end discount_percentage_l714_714475


namespace cos_120_eq_neg_half_l714_714030

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714030


namespace point_on_imaginary_axis_l714_714082

noncomputable def imaginary_axis (z : ℂ) : Prop :=
z.re = 0

theorem point_on_imaginary_axis (z : ℂ) (h : z.conj = complex.I) : imaginary_axis z :=
by
  sorry

end point_on_imaginary_axis_l714_714082


namespace S_128_eq_12_l714_714701

def S (n : ℕ) : ℚ :=
  let factors := (list.filter (λ (p : ℕ × ℕ), p.1 * p.2 = n ∧ p.1 ≤ p.2) 
                    ((list.range (n + 1)).product (list.range (n + 1)))) 
  let best_dec := (list.argmin (λ (p : ℕ × ℕ), abs (p.1 - p.2)) factors).get
  ((best_dec.1 : ℚ) / (best_dec.2 : ℚ))

theorem S_128_eq_12 : S 128 = 1 / 2 := 
by
  sorry

end S_128_eq_12_l714_714701


namespace angle_C_value_value_of_c_l714_714868

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (S : ℝ)

-- Given conditions
axiom triangle_ABC_is_acute : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ 0 < γ ∧ γ < π / 2
axiom sides_opposite_angles : a = 2 ∧ S = (3 * sqrt 3) / 2 ∧ α = π / 3
axiom sqrt_3a_eq_2c_sinA : sqrt 3 * a = 2 * c * sin α

-- Question 1: Find the measure of angle C
theorem angle_C_value : γ = π / 3 :=
by sorry

-- Question 2: Find the value of c given a = 2 and the area of triangle ABC
theorem value_of_c : c = sqrt 7 :=
by sorry

end angle_C_value_value_of_c_l714_714868


namespace sin_cos_sum_complex_expression_value_l714_714803

variable (α : ℝ)

axiom alpha_bound : 0 < α ∧ α < π / 2
axiom trig_identity : cos (2 * π - α) - sin (π - α) = - (sqrt 5) / 5

theorem sin_cos_sum :
  cos α + sin α = (3 * sqrt 5) / 5 :=
by
  -- Problem statement without proof
  sorry

theorem complex_expression_value :
  (cos^2 (3 * π / 2 + α) + 2 * cos α * cos (π / 2 - α)) / (1 + sin^2 (π / 2 - α)) = 5 :=
by
  -- Problem statement without proof
  sorry

end sin_cos_sum_complex_expression_value_l714_714803


namespace farm_horses_cows_ratio_l714_714227

variable (x y : ℕ)  -- x is the base variable related to the initial counts, y is the number of horses sold (and cows bought)

theorem farm_horses_cows_ratio (h1 : 4 * x / x = 4)
    (h2 : 13 * (x + y) = 7 * (4 * x - y))
    (h3 : 4 * x - y = (x + y) + 30) :
    y = 15 := sorry

end farm_horses_cows_ratio_l714_714227


namespace cos_120_eq_neg_one_half_l714_714009

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714009


namespace student_allowance_l714_714302

theorem student_allowance (A : ℝ) (h1 : A * (2/5) = A - (A * (3/5)))
  (h2 : (A - (A * (2/5))) * (1/3) = ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) * (1/3))
  (h3 : ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) = 1.20) :
  A = 3.00 :=
by
  sorry

end student_allowance_l714_714302


namespace michael_and_truck_meet_l714_714558

/--
Assume:
1. Michael walks at 6 feet per second.
2. Trash pails are every 240 feet.
3. A truck travels at 10 feet per second and stops for 36 seconds at each pail.
4. Initially, when Michael passes a pail, the truck is 240 feet ahead.

Prove:
Michael and the truck meet every 120 seconds starting from 120 seconds.
-/
theorem michael_and_truck_meet (t : ℕ) : t ≥ 120 → (t - 120) % 120 = 0 :=
sorry

end michael_and_truck_meet_l714_714558


namespace complex_in_third_quadrant_l714_714815

theorem complex_in_third_quadrant (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 < 0) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complex_in_third_quadrant_l714_714815


namespace simplify_trig_expression_l714_714979

theorem simplify_trig_expression (x : ℝ) (hx : x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := 
by 
  sorry

end simplify_trig_expression_l714_714979


namespace largest_value_among_A_to_E_is_B_l714_714366

def val_A := Real.sqrt (Real.cbrt 56)
def val_B := Real.sqrt (Real.cbrt 3584)
def val_C := Real.sqrt (Real.cbrt 2744)
def val_D := Real.sqrt (Real.cbrt 392)
def val_E := Real.sqrt (Real.cbrt 448)

theorem largest_value_among_A_to_E_is_B : 
  val_B > val_A ∧ val_B > val_C ∧ val_B > val_D ∧ val_B > val_E := by
  sorry

end largest_value_among_A_to_E_is_B_l714_714366


namespace find_60th_number_l714_714353

def is_natural_number (n : ℕ) : Prop := 0 ≤ n

def set_definition (S : Set ℕ) : Prop :=
  ∀ x y : ℕ, is_natural_number x → is_natural_number y → x < y → (2^x + 2^y ∈ S)

theorem find_60th_number (S : Set ℕ)
  (H_def : set_definition S) : ∃ n : ℕ, ∃ k : ℕ, k = 60 ∧ (∀ m : ℕ, m < k → S.median_of_sorted n (n + 1)) := sorry

end find_60th_number_l714_714353


namespace cos_angle_l714_714147

-- Given vectors u and v and their magnitudes
variables (u v : ℝ^3) -- Assuming ℝ^3 for simplicity
variable (hu : ‖u‖ = 5)
variable (hv : ‖v‖ = 7)
variable (huv : ‖u + v‖ = 10)

-- Goal: Prove that the cosine of the angle φ between u and v is 13/35.
theorem cos_angle (u v : ℝ^3) (hu : ‖u‖ = 5) (hv : ‖v‖ = 7) (huv : ‖u + v‖ = 10) :
  real.cos (real.inner_product_space.angle u v) = 13 / 35 :=
by
  sorry

end cos_angle_l714_714147


namespace ceil_neg_sqrt_l714_714746

variable (x : ℚ) (h1 : x = -real.sqrt (64 / 9))

theorem ceil_neg_sqrt : ⌈x⌉ = -2 :=
by
  have h2 : x = - (8 / 3) := by rw [h1, real.sqrt_div, real.sqrt_eq_rpow, real.sqrt_eq_rpow, pow_succ, fpow_succ frac.one_ne_zero, pow_half, real.sqrt_eq_rpow, pow_succ, pow_two]
  rw h2
  have h3 : ⌈- (8 / 3)⌉ = -2 := by linarith
  exact h3

end ceil_neg_sqrt_l714_714746


namespace longest_side_range_l714_714672

-- Definitions and conditions
def is_triangle (x y z : ℝ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

-- Problem statement
theorem longest_side_range (l x y z : ℝ) 
  (h_triangle: is_triangle x y z) 
  (h_perimeter: x + y + z = l / 2) 
  (h_longest: x ≥ y ∧ x ≥ z) : 
  l / 6 ≤ x ∧ x < l / 4 :=
by
  sorry

end longest_side_range_l714_714672


namespace total_savings_l714_714472

-- Define the conditions
def cost_of_two_packs : ℝ := 2.50
def cost_of_single_pack : ℝ := 1.30

-- Define the problem statement
theorem total_savings :
  let price_per_pack_when_in_set := cost_of_two_packs / 2,
      savings_per_pack := cost_of_single_pack - price_per_pack_when_in_set,
      total_packs := 10 * 2,
      total_savings := savings_per_pack * total_packs in
  total_savings = 1 :=
by
  sorry

end total_savings_l714_714472


namespace luke_jigsaw_puzzle_l714_714555

noncomputable def remaining_pieces_after_third_day (initial_pieces: ℕ) (day1_percent: ℝ) (day2_percent: ℝ) (day3_percent: ℝ) : ℕ :=
  let day1_done := initial_pieces * day1_percent in
  let remaining_after_day1 := initial_pieces - day1_done in
  let day2_done := remaining_after_day1 * day2_percent in
  let remaining_after_day2 := remaining_after_day1 - day2_done in
  let day3_done := remaining_after_day2 * day3_percent in
  let remaining_after_day3 := remaining_after_day2 - day3_done in
  remaining_after_day3.to_nat

theorem luke_jigsaw_puzzle : remaining_pieces_after_third_day 1000 0.1 0.2 0.3 = 504 :=
by 
  let initial_pieces := 1000
  let day1_percent := 0.1
  let day2_percent := 0.2
  let day3_percent := 0.3
  let day1_done := initial_pieces * day1_percent
  let remaining_after_day1 := initial_pieces - day1_done
  let day2_done := remaining_after_day1 * day2_percent
  let remaining_after_day2 := remaining_after_day1 - day2_done
  let day3_done := remaining_after_day2 * day3_percent
  let remaining_after_day3 := remaining_after_day2 - day3_done
  have h : remaining_after_day3 = 504 := by sorry
  exact h

end luke_jigsaw_puzzle_l714_714555


namespace negation_of_universal_statement_l714_714257

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔ ∃ a : ℝ, ∀ x : ℝ, ¬(x > 0 ∧ a * x^2 - 3 * x - a = 0) :=
by sorry

end negation_of_universal_statement_l714_714257


namespace hall_volume_l714_714661

theorem hall_volume (length breadth : ℝ) (height : ℝ := 20 / 3)
  (h1 : length = 15)
  (h2 : breadth = 12)
  (h3 : 2 * (length * breadth) = 54 * height) :
  length * breadth * height = 8004 :=
by
  sorry

end hall_volume_l714_714661


namespace valves_fill_pool_l714_714635

theorem valves_fill_pool
  (a b c d : ℝ)
  (h1 : 1 / a + 1 / b + 1 / c = 1 / 12)
  (h2 : 1 / b + 1 / c + 1 / d = 1 / 15)
  (h3 : 1 / a + 1 / d = 1 / 20) :
  1 / a + 1 / b + 1 / c + 1 / d = 1 / 10 := 
sorry

end valves_fill_pool_l714_714635


namespace coefficient_m5n7_in_m_plus_n_12_l714_714724

theorem coefficient_m5n7_in_m_plus_n_12 :
  (12.choose 5) = 792 := by
  sorry

end coefficient_m5n7_in_m_plus_n_12_l714_714724


namespace linear_function_integral_l714_714651

-- Definition of linear function f(x)
def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x + b

-- Definition of the integral conditions
def integral_condition_1 (f : ℝ → ℝ) : Prop :=
  ∫ x in 0..1, f(x) = 5

def integral_condition_2 (f : ℝ → ℝ) : Prop :=
  ∫ x in 0..1, x * f(x) = 17 / 6

-- The main theorem to prove
theorem linear_function_integral :
  ∀ f : ℝ → ℝ,
  linear_function f →
  integral_condition_1 f →
  integral_condition_2 f →
  ∀ x, f(x) = 4 * x + 3 :=
by
  sorry

end linear_function_integral_l714_714651


namespace angle_measure_l714_714065

theorem angle_measure (x : ℝ) 
  (h1 : 90 - x = (2 / 5) * (180 - x)) :
  x = 30 :=
by
  sorry

end angle_measure_l714_714065


namespace area_of_triangle_ABC_l714_714506

/-- Given:
    * The area of square WXYZ is 49 cm^2.
    * The four smaller squares inside it have sides 2 cm long.
    * In ΔABC, AB = AC.
    * When ΔABC is folded over side BC, point A coincides with O, the center of square WXYZ.
    
    Prove:
    The area of ΔABC is 3/4 cm^2. -/
theorem area_of_triangle_ABC :
  (∃ (WXYZ : Type) (side_WXYZ : ℝ) (side_small : ℝ) (O M A B C : WXYZ),
    (side_WXYZ * side_WXYZ = 49) ∧ (side_small = 2) ∧ 
    (∃ (dist_O_to_edge : ℝ), dist_O_to_edge = side_WXYZ / 2 - 2) ∧ 
    let BC := side_WXYZ - 2 * side_small 
    let AM := dist_O_to_edge / 2 in
    -- The area condition
      (∃ (area : ℝ), area = (1 / 2) * BC * AM) ∧ 
      AM = dist_O_to_edge / 2
  ) → 
  (3 / 4) = (1 / 2) * 3 * (1 / 2) := 
sorry -- proof goes here

end area_of_triangle_ABC_l714_714506


namespace symmetric_about_line_5pi12_l714_714822

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem symmetric_about_line_5pi12 :
  ∀ x : ℝ, f (5 * Real.pi / 12 - x) = f (5 * Real.pi / 12 + x) :=
by
  intros x
  sorry

end symmetric_about_line_5pi12_l714_714822


namespace smallest_positive_period_and_monotonic_intervals_range_of_m_l714_714548

def f (x : ℝ) : ℝ :=
  -1 / 2 * sin (2 * x + π / 6)

theorem smallest_positive_period_and_monotonic_intervals :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → 0 < derive f x) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-π / 6) (π / 4) → abs (f x - m) ≤ 2) →
  -7 / 4 ≤ m ∧ m ≤ 3 / 2 :=
sorry

end smallest_positive_period_and_monotonic_intervals_range_of_m_l714_714548


namespace cos_120_eq_neg_half_l714_714021

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714021


namespace measure_angle_XYZ_l714_714671

-- Define the conditions
variables (X Y Z : Type) [IsRegularOctagon X] [IsRegularSquare Y] [IsRegularOctagon Z]
variables (O : Circle) (hO : Inscribed O X) (hO' : Inscribed O Y)
variables (shared_vertex : Vertex X) (common_vertex_of_square : Vertex Y) (common_vertex_of_octagon : Vertex Z)
variables (next_vertex_square : Vertex Y) (next_vertex_octagon : Vertex Z)
variables (interior_angle_octagon : ℝ := 135) (interior_angle_square : ℝ := 90)

-- Define the target
theorem measure_angle_XYZ : ∠XYZ = 45 := by
  sorry

end measure_angle_XYZ_l714_714671


namespace smallest_degree_for_horizontal_asymptote_l714_714370

-- Definitions based on conditions
def numerator : Polynomial ℤ := -Polynomial.X^7 + 4 * Polynomial.X^4 + 2 * Polynomial.X^3 - 5

-- Statement of the proof problem
theorem smallest_degree_for_horizontal_asymptote (p : Polynomial ℤ) :
  Polynomial.degree p ≥ 7 → (∃ p, Polynomial.degree p = 7 ∧
  ∀ p, (Polynomial.degree p >= 7) → Function.has_horizontal_asymptote (fun x => (numerator.eval x) / (p.eval x))) :=
sorry

end smallest_degree_for_horizontal_asymptote_l714_714370


namespace maximum_value_S_l714_714389

def max_area_without_overlap (T : set (ℝ × ℝ)) (side_length_T : ∀ (x y ∈ T), abs (x - y) ≤ 1)
  (S : ℝ) (squares : list (set (ℝ × ℝ))) (area_squares : ∑ square in squares, (set.volume square) = S) : Prop :=
  ∀ square1 square2 ∈ squares, square1 ≠ square2 → square1 ∩ square2 = ∅

theorem maximum_value_S {T : set (ℝ × ℝ)} (hT : ∀ (x y ∈ T), abs (x - y) ≤ 1) :
  ∃ (S : ℝ), max_area_without_overlap T hT S squares → S ≤ 0.5 :=
sorry

end maximum_value_S_l714_714389


namespace milk_savings_l714_714469

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end milk_savings_l714_714469


namespace similar_terms_solution_l714_714159

theorem similar_terms_solution
  (a b : ℝ)
  (m n x y : ℤ)
  (h1 : m - 1 = n - 2 * m)
  (h2 : m + n = 3 * m + n - 4)
  (h3 : m * x + (n - 2) * y = 24)
  (h4 : 2 * m * x + n * y = 46) :
  x = 9 ∧ y = 2 := by
  sorry

end similar_terms_solution_l714_714159


namespace max_chord_length_line_l714_714255

def circle (x y : ℝ) := x^2 + y^2 + 4 * x + 3 = 0

def line (x y : ℝ) := 3 * x - 4 * y + 6 = 0

theorem max_chord_length_line : 
  (∃ x y : ℝ, line x y ∧ circle x y ∧ (x = 2 ∧ y = 3)) →
  (∀ x y : ℝ, circle x y → (x, y) = (-2, 0)) →
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = 0) → line x y) :=
by
  intros h line_intersects_circle center_coordinates
  sorry

end max_chord_length_line_l714_714255


namespace range_m_of_nonmonotonic_on_interval_l714_714827

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1) + x^2 - m * x
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m / (x + 1) + 2 * x - m

theorem range_m_of_nonmonotonic_on_interval :
  ∀ (m : ℝ), ¬ MonotoneOn (f m) (Set.Ioi 1) → m ∈ Set.Ioi 4 :=
by
  sorry

end range_m_of_nonmonotonic_on_interval_l714_714827


namespace b_4_lt_b_7_l714_714685

def α : ℕ → ℕ := λ k, k

def b : ℕ → ℚ
| 1      := 1 + (1 / (α 1))
| n + 1  := 1 + (1 / (α 1 + b_aux n))

noncomputable def b_aux : ℕ → ℚ
| 1      := (1 / (α 1))
| n + 1  := (1 / (α 1 + b_aux n))

theorem b_4_lt_b_7 : b 4 < b 7 := by
  sorry

end b_4_lt_b_7_l714_714685


namespace each_person_pays_23_point_79_l714_714237

/-- Meal costs of each friend before tips --/
def meal_costs : List ℝ := [30, 20, 25, 25, 15, 15, 15]

/-- Tip percentages for each meal --/
def tip_percentages : List ℝ := [0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.2]

/-- Calculate the tip amount for each meal --/
def tip_amounts := (List.zipWith (λ cost percent => cost * percent) meal_costs tip_percentages)

/-- Total cost including tips --/
def total_cost := (List.zipWith (λ cost tip => cost + tip) meal_costs tip_amounts).sum

/-- Number of friends --/
def number_of_friends := 7

/-- Each person's share of the total bill --/
def each_persons_share := total_cost / number_of_friends

/-- Proof that each person's share is $23.79 --/
theorem each_person_pays_23_point_79 : each_persons_share = 23.79 :=
by
  sorry

end each_person_pays_23_point_79_l714_714237


namespace hyperbola_asymptotes_l714_714811

theorem hyperbola_asymptotes (a b : ℝ) (h : ∀ c : ℝ, 2 * c = 2 * 2 * a → c ^ 2 = a ^ 2 + b ^ 2) 
  (hyperbola_eq : ∀ x y : ℝ, (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1) :
  ∀ x : ℝ, ((y = sqrt 3 * x) ∨ (y = -sqrt 3 * x)) :=
begin
  sorry
end

end hyperbola_asymptotes_l714_714811


namespace simplify_trig_expression_l714_714963

theorem simplify_trig_expression (x : ℝ) (h₁ : sin x ≠ 0) (h₂ : cos x ≠ -1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) := 
sorry

end simplify_trig_expression_l714_714963


namespace probability_meets_l714_714351

-- Define the conditions
def time_range := set.Icc 0 50  -- Interval [0, 50]

def meets_condition (x y : ℝ) : Prop :=
  abs (x - y) ≤ 10

def total_area := 2500
def favorable_area := 900

-- Lean statement: Prove question == answer given conditions
theorem probability_meets :
  ∃ (p : ℝ),
    (0 ≤ p) ∧ (p ≤ 1) ∧
    (∀ x y ∈ time_range, meets_condition x y → p = favorable_area / total_area) ∧
    p = 0.36 :=
begin
  sorry
end

end probability_meets_l714_714351


namespace original_cost_111_l714_714343

theorem original_cost_111 (P : ℝ) (h1 : 0.76 * P * 0.90 = 760) : P = 111 :=
by sorry

end original_cost_111_l714_714343


namespace constant_remainder_l714_714045

def polynomial := (12 : ℚ) * (x^3) - (9 : ℚ) * (x^2) + b * x + (8 : ℚ)
def divisor_polynomial := (3 : ℚ) * (x^2) - (4 : ℚ) * x + (2 : ℚ)

theorem constant_remainder (b : ℚ) :
  (∃ r : ℚ, ∀ x : ℚ, (12 * (x^3) - 9 * (x^2) + b * x + 8) % (3 * (x^2) - 4 * x + 2) = r) ↔ b = -4 / 3 :=
by
  sorry

end constant_remainder_l714_714045


namespace simplify_trig_expression_l714_714962

theorem simplify_trig_expression (x : ℝ) (h₁ : sin x ≠ 0) (h₂ : cos x ≠ -1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) := 
sorry

end simplify_trig_expression_l714_714962


namespace ceil_neg_sqrt_frac_l714_714744

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := by
  sorry

end ceil_neg_sqrt_frac_l714_714744


namespace ceil_neg_sqrt_64_div_9_eq_neg2_l714_714731

def sqrt_64_div_9 : ℚ := real.sqrt (64 / 9)
def neg_sqrt_64_div_9 : ℚ := -sqrt_64_div_9
def ceil_neg_sqrt_64_div_9 : ℤ := real.ceil neg_sqrt_64_div_9

theorem ceil_neg_sqrt_64_div_9_eq_neg2 : ceil_neg_sqrt_64_div_9 = -2 := 
by sorry

end ceil_neg_sqrt_64_div_9_eq_neg2_l714_714731


namespace lawn_width_l714_714337

theorem lawn_width (W : ℝ) : 
  (let length_lawn := 80 in 
   let width_road := 10 in 
   let cost_traveling := 3900 in 
   let cost_per_m2 := 3 in 
   let area_roads_total := (width_road * W) + (width_road * length_lawn) - (width_road * width_road) in 
   let area_by_cost := cost_traveling / cost_per_m2 in 
   area_roads_total = area_by_cost) → 
   W = 60 :=
by
  intro h
  sorry

end lawn_width_l714_714337


namespace probability_of_x_gt_3y_from_rectangle_l714_714562

def point (x y : ℝ) := (x, y)

def rectangle_region := { p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 3000) ∧ (0 ≤ p.2 ∧ p.2 ≤ 4000) }

def probability_x_gt_3y := 
  let triangle_region := { p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 3000) ∧ (0 ≤ p.2 ∧ p.2 ≤ p.1 / 3) } in
  let area_triangle := 1 / 2 * 3000 * 1000 in 
  let area_rectangle := 3000 * 4000 in
  area_triangle / area_rectangle

theorem probability_of_x_gt_3y_from_rectangle : probability_x_gt_3y = 1 / 8 := by
  sorry

end probability_of_x_gt_3y_from_rectangle_l714_714562


namespace coat_cost_l714_714714

theorem coat_cost (saved_per_week : ℕ) (weeks_saved : ℕ) (fraction_used_week7 : ℚ) (gift_week8 : ℕ) 
  (total_saved_until_week7 : ℕ) (used_week7 : ℚ) (remaining_after_week7 : ℚ) (final_savings : ℚ)
  (calculated_until_week7 : total_saved_until_week7 = saved_per_week * weeks_saved)
  (calculated_used_week7 : used_week7 = total_saved_until_week7 / fraction_used_week7)
  (calculated_remaining : remaining_after_week7 = total_saved_until_week7 - used_week7)
  (calculated_final_savings : final_savings = remaining_after_week7 + gift_week8) :
  saved_per_week = 25 → weeks_saved = 6 → fraction_used_week7 = 3 → gift_week8 = 70 → final_savings = 170 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4] at calculated_until_week7 calculated_used_week7 calculated_remaining calculated_final_savings
  simp [calculated_until_week7, calculated_used_week7, calculated_remaining, calculated_final_savings]
  sorry

end coat_cost_l714_714714


namespace find_n_l714_714510

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       := 2
| (n + 1) := 2 * a n

-- Define the sum S_n of the first n terms of the sequence {a_n}
def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a n

-- The problem statement:
theorem find_n (n : ℕ) (h : S n = 126) : n = 6 :=
by
  sorry

end find_n_l714_714510


namespace map_length_25_cm_represents_125_km_l714_714560

-- Define the conditions
def map_scale (cm: ℝ) : ℝ := 5 * cm

-- Define the main statement to be proved
theorem map_length_25_cm_represents_125_km : map_scale 25 = 125 := by
  sorry

end map_length_25_cm_represents_125_km_l714_714560


namespace magnitude_b_is_l714_714834

noncomputable def vector_a : ℝ × ℝ := (-2, -1)

axiom dot_product_a_b : ℝ × ℝ → ℝ := λ b, (-2 * b.1) + (-1 * b.2)
axiom dot_product_result : dot_product_a_b = 10

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1) * (v.1) + (v.2) * (v.2))

axiom magnitude_a_b_diff : ∀ b, magnitude (vector_a.1 - b.1, vector_a.2 - b.2) = Real.sqrt 5

theorem magnitude_b_is : ∀ b : ℝ × ℝ, ∃ b : ℝ × ℝ, magnitude b = 2 * Real.sqrt 5 :=
sorry

end magnitude_b_is_l714_714834


namespace simplify_trig_l714_714970

theorem simplify_trig (x : ℝ) (h_cos_sin : cos x ≠ -1) (h_sin_ne_zero : sin x ≠ 0) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
    sorry

end simplify_trig_l714_714970


namespace distinct_z_values_count_l714_714991

def digit_range (a b c d : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9

def z_value (a b c d : ℕ) : ℕ := 
  2 * (999 * (a - d)).natAbs + 90 * (b - c).natAbs

theorem distinct_z_values_count : 
  (∀ (a b c d : ℕ), digit_range a b c d → True) → 
  (|{z | ∃ (a b c d : ℕ), digit_range a b c d ∧ z = z_value a b c d}| = 512) :=
by
  sorry

end distinct_z_values_count_l714_714991


namespace find_angle_A_l714_714099

theorem find_angle_A 
  (A B C : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (triangle_ABC : IsTriangle A B C)
  (acute_triangle : AcuteTriangle A B C)
  (altitude_BM : Perpendicular BM A)
  (altitude_CN : Perpendicular CN A)
  (perpendicular_ML : Perpendicular ML NC)
  (perpendicular_NK : Perpendicular NK BM)
  (ratio_KL_BC : 3 / 4 = KL / (BC))
  : angle_at_vertex_A = 30 :=
sorry

end find_angle_A_l714_714099


namespace log_base_eight_l714_714844

theorem log_base_eight (y : ℕ) (h : log 8 y = 3) : y = 512 :=
  sorry

end log_base_eight_l714_714844


namespace identical_cubes_probability_l714_714624

/-- Statement of the problem -/
theorem identical_cubes_probability :
  let total_ways := 3^8 * 3^8  -- Total ways to paint two cubes
  let identical_ways := 3 + 72 + 252 + 504  -- Ways for identical appearance after rotation
  (identical_ways : ℝ) / total_ways = 1 / 51814 :=
by
  sorry

end identical_cubes_probability_l714_714624


namespace find_n_constant_term_in_expansion_max_binom_coeff_term_l714_714425

-- Given conditions
variables (n : ℕ)

-- Define the given condition that the sum of the binomial coefficients of 
-- the first three terms of the expansion of (2x + 1/sqrt(x))^n is 22.
axiom sum_of_binomials_eq_22 : 1 + n + (n * (n-1)) / 2 = 22

theorem find_n : n = 6 :=
by
  sorry

theorem constant_term_in_expansion
  (h_n : n = 6) : ∀ {x : ℝ}, (∃ c : ℝ, c = algebra.geom_series (2*x) (1/(sqrt(x))) 6 (∅))[c] => c = 60 :=
by
  sorry

theorem max_binom_coeff_term
  (h_n : n = 6) : ∀ {x : ℝ}, (∃ t : (ℝ)^[k=3], C(6,k)(2x)^(6-k)(1/sqrt(x))^k <- (t) => t = 160 * x^(3/2)) :=
by
  sorry

end find_n_constant_term_in_expansion_max_binom_coeff_term_l714_714425


namespace sequence_sum_2023_l714_714831

theorem sequence_sum_2023 :
  (let a : ℕ → ℚ :=
          λ n, nat.rec_on n (1 / 2)
                    (λ n An, An / (2 * (n + 1) * An + 1))
   in ∑ k in finset.range 2023 \ finset.range 1, a k.succ) = 2023 / 2024 := 
sorry

end sequence_sum_2023_l714_714831


namespace cherry_tomatoes_left_l714_714606

theorem cherry_tomatoes_left (initial_tomatoes : ℕ) (fraction_eaten : ℚ) 
  (birds : ℕ) (one_third : fraction_eaten = 1/3) (initial_eq : initial_tomatoes = 21) : 
  initial_tomatoes - (fraction_eaten * initial_tomatoes).natAbs = 14 := 
by 
  sorry

end cherry_tomatoes_left_l714_714606


namespace general_term_sum_of_terms_l714_714438

-- Given conditions
def a (n : ℕ) : ℝ := 3^n
def b (n : ℕ) : ℝ := 2 * (Real.log 3^(n : ℝ) / Real.log 3) + 1
def T (n : ℕ) : ℝ := n^2 + 2 * n

-- The general term formula of the sequence {a_n}
theorem general_term (n : ℕ) : a n = 3^n := by
  sorry

-- The sum of the first n terms of the sequence {b_n}
theorem sum_of_terms (n : ℕ) : (finset.range n).sum b = T n := by
  sorry

end general_term_sum_of_terms_l714_714438


namespace transformed_line_l714_714138

-- Define the matrix transformation
def M : Matrix (Fin 2) (Fin 2) ℝ := ![[1, 1], [0, 1]]

-- Define a point on the original line
def point_on_line (x₀ y₀ : ℝ) : Prop :=
  x₀ + y₀ + 2 = 0

-- Define the transformation of the point
def transformed_point (x y x₀ y₀ : ℝ) : Prop :=
  M.mul_vec ![x, y] = ![x₀, y₀]

-- State the line equation obtained by the transformation
theorem transformed_line (x y x₀ y₀ : ℝ) (h₀ : point_on_line x₀ y₀) (h₁ : transformed_point x y x₀ y₀) : x + 2 * y + 2 = 0 :=
  sorry

end transformed_line_l714_714138


namespace weighted_arithmetic_geometric_mean_l714_714613
-- Importing required library

-- Definitions of the problem variables and conditions
variables (a b c : ℝ)

-- Non-negative constraints on the lengths of the line segments
variables (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

-- Problem statement, we need to prove
theorem weighted_arithmetic_geometric_mean :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c)^(1/3) :=
sorry

end weighted_arithmetic_geometric_mean_l714_714613


namespace monkeys_and_apples_l714_714461

theorem monkeys_and_apples
  {x a : ℕ}
  (h1 : a = 3 * x + 6)
  (h2 : 0 < a - 4 * (x - 1) ∧ a - 4 * (x - 1) < 4)
  : (x = 7 ∧ a = 27) ∨ (x = 8 ∧ a = 30) ∨ (x = 9 ∧ a = 33) :=
sorry

end monkeys_and_apples_l714_714461


namespace xy_maximum_value_l714_714208

theorem xy_maximum_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2 * y) : x - 2 * y ≤ 2 / 3 :=
sorry

end xy_maximum_value_l714_714208


namespace camille_total_birds_count_l714_714713

theorem camille_total_birds_count :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry

end camille_total_birds_count_l714_714713


namespace solve_inequality_l714_714431

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + x else -x^2 - x

theorem solve_inequality (x : ℝ) : f(x) + 2 > 0 ↔ x > -2 := by
  sorry

end solve_inequality_l714_714431


namespace vector_dot_product_l714_714155

variable {V : Type*} [InnerProductSpace ℝ V]

theorem vector_dot_product (x y z : V) (h1 : ⟪x, y⟫ = 5) (h2 : ⟪x, z⟫ = -2) (h3 : ⟪y, z⟫ = 3) :
  ⟪y, 4 • z - 3 • x⟫ = -3 :=
by
  sorry

end vector_dot_product_l714_714155


namespace average_score_of_seniors_l714_714355

theorem average_score_of_seniors
    (total_students : ℕ)
    (average_score_all : ℚ)
    (num_seniors num_non_seniors : ℕ)
    (mean_score_senior mean_score_non_senior : ℚ)
    (h1 : total_students = 120)
    (h2 : average_score_all = 84)
    (h3 : num_non_seniors = 2 * num_seniors)
    (h4 : mean_score_senior = 2 * mean_score_non_senior)
    (h5 : num_seniors + num_non_seniors = total_students)
    (h6 : num_seniors * mean_score_senior + num_non_seniors * mean_score_non_senior = total_students * average_score_all) :
  mean_score_senior = 126 :=
by
  sorry

end average_score_of_seniors_l714_714355


namespace product_of_set_is_positive_l714_714096

open Set

theorem product_of_set_is_positive (S : Set ℝ) 
  (hS : S.card = 100)
  (h_distinct : ∀a ∈ S, ∀b ∈ S, a ≠ b → a + b ≠ 0)
  (h_property : ∀a ∈ S, (∑ x in S, x) - a ∈ S) :
  0 < ∏ x in S, x :=
sorry

end product_of_set_is_positive_l714_714096


namespace minimize_y_at_4_l714_714079

noncomputable def a : ℝ := Real.log 120 / Real.log Real.pi

def y (x : ℕ) : ℝ := Real.abs (x - a)

theorem minimize_y_at_4 : (∀ x : ℕ, y x ≥ y 4) := sorry

end minimize_y_at_4_l714_714079


namespace incorrect_statement_l714_714144

variable α β : ℝ

def a : ℝ × ℝ := (Real.cos α, Real.sin α)
def b : ℝ × ℝ := (Real.cos β, Real.sin β)

theorem incorrect_statement :
  let θ := Real.arccos ((Real.cos α * Real.cos β) + (Real.sin α * Real.sin β))
  θ ≠ α - β := sorry

end incorrect_statement_l714_714144


namespace max_angle_SB_SAM_l714_714116

/-- Assuming the geometric problem conditions -/
variables (A B C M S O : Point)
variables (SA : Segment S A) (SO : Segment S O) (SC : Segment S C) (OM : Segment O M)
variables (circleO : Circle O (Radius.calculate 4 (2 * sqrt 3)))
variables (isDiameter : Diameter AC circleO)
variables (isOnCircleO : OnCircle B circleO)
variables (onSegment : OnSegment M SC)
variables (M_not_endpoints : M ≠ S ∧ M ≠ C)

/-- Proof statement -/
theorem max_angle_SB_SAM : 
  Angle (LineThroughSegment S B) (PlaneThroughPoints S A M) ≤ π / 6 :=
sorry

end max_angle_SB_SAM_l714_714116


namespace trapezoid_diag_length_l714_714182

theorem trapezoid_diag_length (AD BC AC BD : ℝ)
    (h_AD : AD = 20)
    (h_BC : BC = 10)
    (h_AC : AC = 18) :
    ∃ O : Point, circles_intersect (circle (AB.sideDiameter) O) (circle (BC.sideDiameter) O) ∧
                 circles_intersect (circle (BC.sideDiameter) O) (circle (CD.sideDiameter) O) ∧
                 circles_intersect (circle (AB.sideDiameter) O) (circle (CD.sideDiameter) O) →
    BD = 24 :=
by 
  sorry

end trapezoid_diag_length_l714_714182


namespace custom_op_repeated_2012_times_l714_714676

def custom_op (a b : ℕ) : ℕ := 
  if a = 3 ∧ b = 4 then 2
  else if a = 1 ∧ b = 2 then 2
  else if a = 2 ∧ b = 2 then 4
  else if a = 4 ∧ b = 2 then 3
  else if a = 3 ∧ b = 2 then 1
  else if a = 1 ∧ b = 2 then 2
  else 0 -- default case for undefined operations

theorem custom_op_repeated_2012_times : 
  (nat.iterate (λ n, custom_op n 2) 2012 2) = 1 :=
sorry

end custom_op_repeated_2012_times_l714_714676


namespace prob_one_defective_without_replacement_prob_one_defective_with_replacement_l714_714782

noncomputable theory

-- Definition of the sets and conditions
def items := ["a", "b", "c"]
def is_defective (x : String) : Bool := x = "c"
def exactly_one_defective (pair : (String × String)) : Bool :=
  (is_defective pair.1 && ¬is_defective pair.2) ||
  (¬is_defective pair.1 && is_defective pair.2)

-- Problem statement for without replacement
theorem prob_one_defective_without_replacement : 
  ((1/3 : ℝ) * (1/2 : ℝ) + (1/2 : ℝ) * (1/3 : ℝ)) = 2/3 :=
by sorry

-- Problem statement for with replacement
theorem prob_one_defective_with_replacement : 
  (2 * (1/3 : ℝ) * (2/3 : ℝ)) = 4/9 :=
by sorry

end prob_one_defective_without_replacement_prob_one_defective_with_replacement_l714_714782


namespace distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l714_714325

-- We will assume the depth of the well as a constant
def well_depth : ℝ := 4.0

-- Climb and slide distances as per each climb
def first_climb : ℝ := 1.2
def first_slide : ℝ := 0.4
def second_climb : ℝ := 1.4
def second_slide : ℝ := 0.5
def third_climb : ℝ := 1.1
def third_slide : ℝ := 0.3
def fourth_climb : ℝ := 1.2
def fourth_slide : ℝ := 0.2

noncomputable def net_gain_four_climbs : ℝ :=
  (first_climb - first_slide) + (second_climb - second_slide) +
  (third_climb - third_slide) + (fourth_climb - fourth_slide)

noncomputable def distance_from_top_after_four : ℝ := 
  well_depth - net_gain_four_climbs

noncomputable def total_distance_covered_four_climbs : ℝ :=
  first_climb + first_slide + second_climb + second_slide +
  third_climb + third_slide + fourth_climb + fourth_slide

noncomputable def can_climb_out_fifth_climb : Bool :=
  well_depth < (net_gain_four_climbs + first_climb)

-- Now we state the theorems we need to prove

theorem distance_from_top_correct :
  distance_from_top_after_four = 0.5 := by
  sorry

theorem total_distance_covered_correct :
  total_distance_covered_four_climbs = 6.3 := by
  sorry

theorem fifth_climb_success :
  can_climb_out_fifth_climb = true := by
  sorry

end distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l714_714325


namespace roots_difference_l714_714760

theorem roots_difference (p : ℝ) :
  let f := (fun x => x^2 - 4*p*x + (4*p^2 - 9)) in
  let r1 := (4*p + real.sqrt (4*p^2 - 4*(4*p^2 - 9)))/2 in
  let r2 := (4*p - real.sqrt (4*p^2 - 4*(4*p^2 - 9)))/2 in
  r1 - r2 = 6 :=
by
  sorry

end roots_difference_l714_714760


namespace problem_proof_l714_714204

theorem problem_proof (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x^2 + A)^2 + B) - (B * (A * x^2 + B)^2 + A) = A^2 - B^2) :
  A^2 + B^2 = - (A * B) := 
sorry

end problem_proof_l714_714204


namespace sum_of_first_50_terms_of_progressions_is_5205_l714_714910

noncomputable def seq_a (d_a : ℝ) (n : ℕ) : ℝ := 
  10 + (n - 1) * d_a

noncomputable def seq_b (d_b : ℝ) (n : ℕ) : ℝ := 
  50 + (n - 1) * d_b

noncomputable def seq_sum (d_a d_b : ℝ) (n : ℕ) : ℝ := 
  seq_a d_a n + seq_b d_b n

theorem sum_of_first_50_terms_of_progressions_is_5205
  (d_a d_b : ℝ)
  (h : seq_a d_a 50 + seq_b d_b 50 = 150) :
  Finset.sum (Finset.range 50) (λ n, seq_sum d_a d_b (n+1)) = 5205 :=
by 
  sorry

end sum_of_first_50_terms_of_progressions_is_5205_l714_714910


namespace player_b_has_optimal_strategy_l714_714625

-- Define the setup for the game with the specified number of faces and sum of numbers
def num_faces := 26^(5^2019)
def sum_faces := 27^(5^2019)

-- Define the condition for the sum of the faces
noncomputable def valid_die (die : list ℕ) : Prop :=
  die.length = num_faces ∧ die.sum = sum_faces

-- State the main theorem: Player B can always create a die that is more optimal than Player A's die
theorem player_b_has_optimal_strategy :
  ∀ die_A : list ℕ, (valid_die die_A) →
  ∃ die_B : list ℕ, (valid_die die_B) ∧
                 (∃ strategy : (ℕ → ℕ → Prop), ∀ x y, strategy x y → 
                 (list.count die_B x : ℤ) * (list.count die_A y : ℤ) 
                 > (list.count die_A x : ℤ) * (list.count die_B y : ℤ)) :=
by sorry

end player_b_has_optimal_strategy_l714_714625


namespace find_QR_l714_714241

variables (Q P R : Type) [MetricSpace Q] [MetricSpace P] [MetricSpace R]

def right_triangle (P Q R : Type) := 
  sorry -- Define what it means to be a right triangle.

def cos (Q : ℝ) : ℝ := 
  sorry -- Define cosine.

theorem find_QR (cosQ : ℝ) (QP : ℝ) (QR : ℝ) (h1 : cosQ = 0.6) (h2 : QP = 15) 
  (h3 : QR = QP / cosQ) : QR = 25 := 
by 
  sorry 

#check find_QR 

end find_QR_l714_714241


namespace sequence_values_sequence_general_formula_l714_714437

noncomputable def sequence : ℕ → ℚ
| 0       := 3
| (n + 1) := (3 * sequence n - 4) / (sequence n - 1)

theorem sequence_values :
  sequence 1 = 5 / 2 ∧ sequence 2 = 7 / 3 ∧ sequence 3 = 9 / 4 :=
by {
  split,
  { show sequence 1 = 5 / 2, sorry },
  split,
  { show sequence 2 = 7 / 3, sorry },
  { show sequence 3 = 9 / 4, sorry }
}

theorem sequence_general_formula (n : ℕ) :
  sequence n = (2 * ↑n + 1) / ↑n :=
by {
  induction n with k hk,
  { show sequence 0 = 3, sorry },
  { show sequence (k + 1) = (2 * ↑(k + 1) + 1) / ↑(k + 1), sorry }
}

end sequence_values_sequence_general_formula_l714_714437


namespace minimum_subsidy_minimum_avg_cost_at_40_l714_714173

def processing_cost (x : ℝ) : ℝ :=
  if x ∈ Icc 10 29 then (1/25) * x^3 + 640
  else if x ∈ Icc 30 50 then x^2 - 40 * x + 1600
  else 0

def chemical_product_value (x : ℝ) : ℝ :=
  20 * x

def profit (x : ℝ) : ℝ :=
  chemical_product_value x - processing_cost x

def average_cost (x : ℝ) : ℝ :=
  (processing_cost x) / x

theorem minimum_subsidy (x : ℝ) (hx : x ∈ Icc 30 50) : profit x < 0 → ∃ s ≥ 700, profit x + s = 0 :=
by sorry

theorem minimum_avg_cost_at_40 (x : ℝ) (hx : 10 ≤ x ∧ x < 50) : ∀ y ∈ Icc 10 50, average_cost y ≥ average_cost 40 :=
by sorry

end minimum_subsidy_minimum_avg_cost_at_40_l714_714173


namespace right_triangle_acute_angles_l714_714168

theorem right_triangle_acute_angles (A B C P O : Type) 
  [right_triangle : is_right_triangle A B C]
  [angle_bisector : angle_bisector A P B C]
  [incenter : is_incenter O A B C]
  (h_ratio : ratio AO OP = (Real.sqrt 3 + 1) / (Real.sqrt 3 - 1)) :
  ∃ α β : ℝ, α = 30 ∧ β = 60 ∧ acute_angle A B C α ∧ acute_angle A B C β :=
by
  sorry

end right_triangle_acute_angles_l714_714168


namespace ceiling_example_l714_714052

theorem ceiling_example : ⌈4 * (8 - 3 / 4)⌉ = 29 := 
by
  sorry

end ceiling_example_l714_714052


namespace find_expression_value_l714_714787

theorem find_expression_value (x : ℝ) (h : x^2 - 3 * x - 1 = 0) : -3 * x^2 + 9 * x + 4 = 1 :=
by sorry

end find_expression_value_l714_714787


namespace trace_bag_weight_is_two_l714_714616

-- Given the conditions in the problem
def weight_gordon_bag₁ : ℕ := 3
def weight_gordon_bag₂ : ℕ := 7
def num_traces_bag : ℕ := 5

-- Total weight of Gordon's bags is 10
def total_weight_gordon := weight_gordon_bag₁ + weight_gordon_bag₂

-- Trace's bags weight
def total_weight_trace := total_weight_gordon

-- All conditions must imply this equation is true
theorem trace_bag_weight_is_two :
  (num_traces_bag * 2 = total_weight_trace) → (2 = 2) :=
  by
    sorry

end trace_bag_weight_is_two_l714_714616


namespace cos_120_eq_neg_half_l714_714028

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714028


namespace measure_angle_ACB_l714_714504

-- Define angle measures as degrees
def angle_BAC := 105
def angle_ABD := 140

-- Define the supplementary relationship
def angle_ABC := 180 - angle_ABD

-- Prove that the measure of angle ACB is 35 degrees
theorem measure_angle_ACB : angle_ABC + angle_BAC < 180 → ∃ angle_ACB, angle_ACB = 180 - (angle_BAC + angle_ABC) ∧ angle_ACB = 35 :=
by
  intros h
  use (180 - (angle_BAC + angle_ABC))
  split
  {
    sorry
  }
  {
    sorry
  }

end measure_angle_ACB_l714_714504


namespace sqrt_inequality_l714_714527

theorem sqrt_inequality
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 3) :
  2 * real.sqrt (x + real.sqrt y) + 2 * real.sqrt (y + real.sqrt z) + 2 * real.sqrt (z + real.sqrt x) ≤ 
  real.sqrt (8 + x - y) + real.sqrt (8 + y - z) + real.sqrt (8 + z - x) := 
by
  sorry

end sqrt_inequality_l714_714527


namespace locus_of_P_l714_714614

-- Definitions based on conditions
def F : ℝ × ℝ := (2, 0)
def Q (k : ℝ) : ℝ × ℝ := (0, -2 * k)
def T (k : ℝ) : ℝ × ℝ := (-2 * k^2, 0)
def P (k : ℝ) : ℝ × ℝ := (2 * k^2, -4 * k)

-- Theorem statement based on the proof problem
theorem locus_of_P (x y : ℝ) (k : ℝ) (hf : F = (2, 0)) (hq : Q k = (0, -2 * k))
  (ht : T k = (-2 * k^2, 0)) (hp : P k = (2 * k^2, -4 * k)) :
  y^2 = 8 * x :=
sorry

end locus_of_P_l714_714614


namespace num_adults_attended_l714_714588

-- Definitions for the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_children : ℕ := 28
def total_revenue : ℕ := 5122

-- The goal is to prove the number of adults who attended the show
theorem num_adults_attended :
  ∃ (A : ℕ), A * ticket_price_adult + num_children * ticket_price_child = total_revenue ∧ A = 183 :=
by
  sorry

end num_adults_attended_l714_714588


namespace collinear_DEF_l714_714538

-- Define the points and the conditions of the problem
variables (O A B C P D E F : Type) [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited D] [Inhabited E] [Inhabited F]

-- Assume that PA, PB, and PC are chords of the circle centered at O
axiom chord_PA (PA : Set P) : Set O
axiom chord_PB (PB : Set P) : Set O
axiom chord_PC (PC : Set P) : Set O

-- Assume circles are drawn with these chords as diameters and intersect at points D, E, F
axiom circle_PF_diameter (PF : Set P) : Set O
axiom circle_PB_diameter (PB : Set P) : Set O
axiom circle_PA_diameter (PA : Set P) : Set O
axiom intersection_at_D : D ∈ circle_PB_diameter PB ∧ D ∈ circle_PA_diameter PA
axiom intersection_at_E : E ∈ circle_PC_diameter PC ∧ E ∈ circle_PA_diameter PA
axiom intersection_at_F : F ∈ circle_PB_diameter PB ∧ F ∈ circle_PC_diameter PC

-- Required to prove D, E, F are collinear
theorem collinear_DEF : ∀ (PA PB PC : Set P) (D E F : P), 
    (∃ (circle_PB PB : Set O) (circle_PA PA : Set O) (circle_PC PC : Set O) (intersection_at_D : D ∈ circle_PB PB ∧ D ∈ circle_PA PA) 
    (intersection_at_F : F ∈ circle_PB PB ∧ F ∈ circle_PC PC) 
    (intersection_at_E : E ∈ circle_PC PC ∧ E ∈ circle_PA PA),
    D ∈ line_through E F) :=
sorry

end collinear_DEF_l714_714538


namespace remainder_division_l714_714391

-- Define p(x) and d(x)
def p (x : ℝ) := x^6 - x^5 - x^4 + x^3 + x^2
def d (x : ℝ) := (x^2 - 4) * (x + 1)

-- Prove the remainder is as specified
theorem remainder_division :
  let r (x : ℝ) := 15 * x^2 - 12 * x - 24
  in ∃ q : ℝ → ℝ, p(x) = d(x) * q(x) + r(x) :=
by
  sorry

end remainder_division_l714_714391


namespace smallest_second_sum_l714_714270

theorem smallest_second_sum (n : ℕ) (S : ℝ) (numbers : Fin n → ℝ)
  (h1 : 2 < n)
  (h2 : ∀ i, numbers i ≠ 0)
  (h3 : ∑ i, numbers i = 0)
  (h4 : bags.nonempty (subsets numbers))
  (h5 : ∃ (subsums : Multiset ℝ), (∀ x ∈ subsums, x ∈ (subsets numbers).map (λ s, ∑ j in s, numbers j)) ∧ multiset.sort (≥) subsums (cons S (some (multiset.drop 1 subsums))))
  : ∃ T : ℝ, T = S - S / (↑(n + 1) / 2).ceil :=
sorry

end smallest_second_sum_l714_714270


namespace area_under_transformed_graph_l714_714266

noncomputable def g (x : ℝ) : ℝ := sorry

theorem area_under_transformed_graph :
  (∫ x in set.Icc a b, g x) = 15 → (∫ x in set.Icc ((z - 1) / 3) (w - 1) / 3, 2 * g (3 * x + 1)) = 30 := by
  sorry

end area_under_transformed_graph_l714_714266


namespace quadratic_polynomial_value_at_zero_eq_zero_l714_714768

theorem quadratic_polynomial_value_at_zero_eq_zero
  (p q : ℝ)
  (polynomial_form : ∀ x, p x = x^2 - (p+q) * x + pq)
  (distinct_roots_condition : (p(p(x)) = (p(x))^2 - (p+q)(p(x)) + pq) → (∀ x, four_distinct_real_roots p(p(x)))):
  p(0) = 0 := sorry

end quadratic_polynomial_value_at_zero_eq_zero_l714_714768


namespace mary_characters_initial_D_l714_714927

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l714_714927


namespace sum_of_faces_of_rectangular_prism_l714_714573

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism_l714_714573


namespace eccentricity_ellipse_example_l714_714819

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (F1 O P Q : ℝ × ℝ)
  (h3 : (P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1))
  (h4 : (Q.1 = 2 * F1.1))
  (h5 : (Q.1 ^ 2 + Q.2 ^ 2 = λ * ((F1.1 - P.1) / (F1.1 ^ 2 + F1.2 ^ 2).sqrt
    + (F1.1 - O.1) / (F1.1 ^ 2 + F1.2 ^ 2).sqrt))) 
  (h6 : 0 < λ) : ℝ :=
  (sqrt 5 - 1) / 2

theorem eccentricity_ellipse_example (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (F1 O P Q : ℝ × ℝ)
  (h3 : (P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1))
  (h4 : (Q.1 = 2 * F1.1))
  (h5 : (Q.1 ^ 2 + Q.2 ^ 2 = λ * ((F1.1 - P.1) / (F1.1 ^ 2 + F1.2 ^ 2).sqrt
    + (F1.1 - O.1) / (F1.1 ^ 2 + F1.2 ^ 2).sqrt))) 
  (h6 : 0 < λ) :
  eccentricity_of_ellipse a b h1 h2 F1 O P Q h3 h4 h5 h6 = (sqrt 5 - 1) / 2 :=
begin
  sorry
end

end eccentricity_ellipse_example_l714_714819


namespace ceil_neg_sqrt_64_div_9_l714_714739

theorem ceil_neg_sqrt_64_div_9 : ⌈-real.sqrt (64 / 9)⌉ = -2 := 
by
  sorry

end ceil_neg_sqrt_64_div_9_l714_714739


namespace prove_congruences_l714_714940

def pairwise_coprime (ms : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < ms.length → j < ms.length → i ≠ j → Nat.coprime (ms[i]) (ms[j])

noncomputable def euler_totient (n : ℕ) := Nat.totient n

theorem prove_congruences
  (n : ℕ) (ms : List ℕ) (h_len : ms.length = n) (h_coprime : pairwise_coprime ms) :
  ∀ (x : ℕ), x = (List.prod (ms.drop 1)) ^ (euler_totient (ms[0])) →
  (x ≡ 1 [MOD ms[0]]) ∧ ∀ i, 1 ≤ i → i < n → (x ≡ 0 [MOD ms[i]]) := 
by
  sorry

end prove_congruences_l714_714940


namespace remainder_polynomial_l714_714069

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (forall x : ℝ, polynomial.eval x p =  p.eval 1 = 4 ∧ p.eval 3 = 6 ∧ p.eval 5 = 8 ) 
  -> (r = λ x, x + 3) 
  -> (forall x : ℝ, p.eval (x-1)(x-3)(x-5) = r x) :=
sorry
  -- Proof omitted intentionally

end remainder_polynomial_l714_714069


namespace range_of_a_l714_714244

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ set.Icc (1 : ℝ) (3 : ℝ), abs (x_0^2 - a * x_0 + 4) ≤ 3 * x_0) ↔ (1 ≤ a ∧ a ≤ 8) := 
sorry

end range_of_a_l714_714244


namespace sandy_spent_home_currency_l714_714956

variable (A B C D : ℝ)

def total_spent_home_currency (A B C D : ℝ) : ℝ :=
  let total_foreign := A + B + C
  total_foreign * D

theorem sandy_spent_home_currency (D : ℝ) : 
  total_spent_home_currency 13.99 12.14 7.43 D = 33.56 * D := 
by
  sorry

end sandy_spent_home_currency_l714_714956


namespace intersection_M_N_l714_714550

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | abs (x - 1) ≤ 1}

def N : Set ℝ := {x : ℝ | log x > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l714_714550


namespace comparison_l714_714698

def sequence (α : ℕ → ℕ) : ℕ → ℚ
| 1     := 1 + 1 / (α 1)
| (n+1) := 1 + 1 / (α 1 + sequence (λ k, α (k+1) n))

theorem comparison (α : ℕ → ℕ) (h : ∀ k, 1 ≤ α k) :
  sequence α 4 < sequence α 7 := 
sorry

end comparison_l714_714698


namespace max_value_xy_xz_yz_l714_714536

theorem max_value_xy_xz_yz (x y z : ℝ) (h : x + 2 * y + z = 6) :
  xy + xz + yz ≤ 6 :=
sorry

end max_value_xy_xz_yz_l714_714536


namespace describe_S_is_two_rays_l714_714539

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ common : ℝ, 
     (common = 5 ∧ (p.1 + 3 = common ∧ p.2 - 2 ≥ common ∨ p.1 + 3 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.1 + 3 ∧ (5 = common ∧ p.2 - 2 ≥ common ∨ 5 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.2 - 2 ∧ (5 = common ∧ p.1 + 3 ≥ common ∨ 5 ≥ common ∧ p.1 + 3 = common))}

theorem describe_S_is_two_rays :
  S = {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 ≥ 7) ∨ (p.2 = 7 ∧ p.1 ≥ 2)} :=
  by
    sorry

end describe_S_is_two_rays_l714_714539


namespace chessboard_no_single_black_square_l714_714088

theorem chessboard_no_single_black_square :
  (∀ (repaint : (Fin 8) × Bool → (Fin 8) × Bool), False) :=
by 
  sorry

end chessboard_no_single_black_square_l714_714088


namespace triangle_area_l714_714902

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- Statement of the theorem
theorem triangle_area : (1 / 2) * |(a.1 * b.2 - a.2 * b.1)| = 4.5 := by
  sorry

end triangle_area_l714_714902


namespace train_speed_in_kmh_l714_714656

noncomputable def train_length : ℝ := 480
noncomputable def platform_length : ℝ := 620
noncomputable def time_to_cross : ℝ := 71.99424046076314

theorem train_speed_in_kmh :
  let total_distance := train_length + platform_length in
  let speed_m_per_s := total_distance / time_to_cross in
  let speed_kmh := speed_m_per_s * 3.6 in
  speed_kmh ≈ 54.964 :=
by
  sorry

end train_speed_in_kmh_l714_714656


namespace quadratic_inequality_solution_l714_714379

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 + 7 * x + 6 < 0) ↔ (-6 < x ∧ x < -1) :=
sorry

end quadratic_inequality_solution_l714_714379


namespace greater_number_l714_714604

theorem greater_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : x = 35 := 
by sorry

end greater_number_l714_714604


namespace convex_power_function_l714_714763

theorem convex_power_function (n : ℕ) (h : 0 < n) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (↑n * (↑n - 1) * x ^ (↑n - 2))) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end convex_power_function_l714_714763


namespace limit_series_ratio_l714_714034

noncomputable def S1 (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (1 / (3 ^ k))
noncomputable def S2 (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (1 / (2 ^ k))

theorem limit_series_ratio : 
  tendsto (λ n, S1 n / S2 n) atTop (𝓝 (3 / 4)) :=
  sorry

end limit_series_ratio_l714_714034


namespace female_wins_probability_l714_714495

theorem female_wins_probability :
  let p_alexandr := 3 * p_alexandra,
      p_evgeniev := (1 / 3) * p_evgenii,
      p_valentinov := (3 / 2) * p_valentin,
      p_vasilev := 49 * p_vasilisa,
      p_alexandra := 1 / 4,
      p_alexandr := 3 / 4,
      p_evgeniev := 1 / 12,
      p_evgenii := 11 / 12,
      p_valentinov := 3 / 5,
      p_valentin := 2 / 5,
      p_vasilev := 49 / 50,
      p_vasilisa := 1 / 50,
      p_female := 
        (1 / 4) * p_alexandra + 
        (1 / 4) * p_evgeniev + 
        (1 / 4) * p_valentinov + 
        (1 / 4) * p_vasilisa 
  in p_female ≈ 0.355 := 
sorry

end female_wins_probability_l714_714495


namespace axis_of_symmetry_l714_714435

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem axis_of_symmetry (φ : ℝ) (hφ : 0 ≤ φ ∧ φ ≤ 2 * Real.pi) :
  (Real.sin (2 * x + φ)) = Real.cos (2 * (x - Real.pi / 3)) →
  ∃ k : ℤ, 2 * x + φ = Real.pi / 2 + k * Real.pi → x = Real.pi / 6 :=
begin
  sorry
end

end axis_of_symmetry_l714_714435


namespace compute_sum_of_digits_t_l714_714532

def trailing_zeros (n : ℕ) : ℕ :=
  nat.floor (n / 5) + nat.floor (n / 25) + nat.floor (n / 125) + nat.floor (n / 625) + nat.floor (n / 3125)

theorem compute_sum_of_digits_t :
  let m_values := {m : ℕ | m > 5 ∧ 
                   let p := trailing_zeros m in
                   trailing_zeros (3 * m) = 4 * p} in
  let t := 10 + 11 + 16 + 17 in
  let sum_of_digits := (to_digits 10 t).sum in
  sum_of_digits = 9 :=
by
  sorry

end compute_sum_of_digits_t_l714_714532


namespace quartic_polynomial_sum_l714_714770

noncomputable def quartic_sum (q : ℕ → ℕ) : ℕ :=
  (∑ x in Finset.range 22, q x)

theorem quartic_polynomial_sum :
  ∀ (q : ℕ → ℕ),
    q 2 = 20 →
    q 8 = 30 →
    q 14 = 22 →
    q 20 = 40 →
    quartic_sum q = 630 :=
by
  intros q h2 h8 h14 h20
  sorry

end quartic_polynomial_sum_l714_714770


namespace particles_tend_to_unit_circle_as_time_goes_to_infinity_l714_714073

variables {x y t : ℝ}

-- Condition: Velocity field as a function
def velocity_field (x y : ℝ) : ℝ × ℝ :=
  (y + 2*x - 2*x^3 - 2*x*y^2, -x)

-- Formal statement to prove: Particles tend towards the unit circle as t -> ∞.
theorem particles_tend_to_unit_circle_as_time_goes_to_infinity
  (f : ℝ × ℝ → ℝ × ℝ)
  (hf : ∀ (x y : ℝ), f (x, y) = velocity_field x y) :
  (∃ (R : ℝ), ∀ (x y t : ℝ), (t > 0) → ‖(x, y) - (0, 0)‖ < R → r = 1) :=
sorry

end particles_tend_to_unit_circle_as_time_goes_to_infinity_l714_714073


namespace find_m_l714_714140

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 3, 2*m - 1}) (hB: B = {3, m^2}) (h_subset: B ⊆ A) : m = 1 :=
by
  sorry

end find_m_l714_714140


namespace combined_weight_l714_714681

variable (a b c d : ℕ)

theorem combined_weight :
  a + b = 260 →
  b + c = 245 →
  c + d = 270 →
  a + d = 285 :=
by
  intros hab hbc hcd
  sorry

end combined_weight_l714_714681


namespace count_powers_of_2_not_4_under_2000000_l714_714448

theorem count_powers_of_2_not_4_under_2000000 :
  ∃ n, ∀ x, x < 2000000 → (∃ k, x = 2 ^ k ∧ (∀ m, x ≠ 4 ^ m)) ↔ x > 0 ∧ x < 2 ^ (n + 1) := by
  sorry

end count_powers_of_2_not_4_under_2000000_l714_714448


namespace prob_female_l714_714486

/-- Define basic probabilities for names and their gender associations -/
variables (P_Alexander P_Alexandra P_Yevgeny P_Evgenia P_Valentin P_Valentina P_Vasily P_Vasilisa : ℝ)

-- Define the conditions for the probabilities
axiom h1 : P_Alexander = 3 * P_Alexandra
axiom h2 : P_Yevgeny = 3 * P_Evgenia
axiom h3 : P_Valentin = 1.5 * P_Valentina
axiom h4 : P_Vasily = 49 * P_Vasilisa

/-- The problem we need to prove: the probability that the lot was won by a female student is approximately 0.355 -/
theorem prob_female : 
  let P_female := (P_Alexandra * 1 / 4) + (P_Evgenia * 1 / 4) + (P_Valentina * 1 / 4) + (P_Vasilisa * 1 / 4) in
  abs (P_female - 0.355) < 0.001 :=
sorry

end prob_female_l714_714486


namespace sequence_general_formula_l714_714830

variable (a : ℝ)

def sequence (n : ℕ) : ℝ :=
  Nat.recOn n a (λ n an, 1 / (2 - an))

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) :
  sequence a n = ((n-1) - (n-2)*a) / (n - (n-1)*a) := by
  sorry

end sequence_general_formula_l714_714830


namespace smallest_possible_value_l714_714939

def smallest_ab (a b : ℕ) : Prop :=
  a^a % b^b =  0 ∧ a % b ≠ 0 ∧ Nat.gcd b 210 = 1

theorem smallest_possible_value : ∃ (a b : ℕ), smallest_ab a b ∧ a + b = 374 :=
by {
  existsi 253,
  existsi 121,
  unfold smallest_ab,
  simp,
  split,
  { sorry }, -- Proof that 253^253 % 121^121 = 0
  split,
  { exact dec_trivial }, -- Proof that 253 % 121 ≠ 0
  { exact dec_trivial }, -- Proof that Nat.gcd 121 210 = 1
  { refl },
}

end smallest_possible_value_l714_714939


namespace units_digit_S_5432_l714_714213

def x : ℝ := 4 + Real.sqrt 15
def y : ℝ := 4 - Real.sqrt 15

def S (n : ℕ) : ℝ := 0.5 * (x^n + y^n)

def units_digit (z : ℝ) : ℕ := 
  (Int.natAbs (Real.toInt z)) % 10

theorem units_digit_S_5432 : units_digit (S 5432) = 1 :=
sorry

end units_digit_S_5432_l714_714213


namespace cos_120_eq_neg_half_l714_714022

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l714_714022


namespace ceiling_example_l714_714051

theorem ceiling_example : ⌈4 * (8 - 3 / 4)⌉ = 29 := 
by
  sorry

end ceiling_example_l714_714051


namespace inverse_variation_solution_l714_714243

theorem inverse_variation_solution :
  ∀ (x y k : ℝ),
    (x * y^3 = k) →
    (∃ k, x = 8 ∧ y = 1 ∧ k = 8) →
    (y = 2 → x = 1) :=
by
  intros x y k h1 h2 hy2
  sorry

end inverse_variation_solution_l714_714243


namespace nat_min_a_plus_b_l714_714934

theorem nat_min_a_plus_b (a b : ℕ) (h1 : b ∣ a^a) (h2 : ¬ b ∣ a) (h3 : Nat.coprime b 210) : a + b = 374 :=
sorry

end nat_min_a_plus_b_l714_714934


namespace giyoon_above_average_subjects_l714_714446

def points_korean : ℕ := 80
def points_mathematics : ℕ := 94
def points_social_studies : ℕ := 82
def points_english : ℕ := 76
def points_science : ℕ := 100
def number_of_subjects : ℕ := 5

def total_points : ℕ := points_korean + points_mathematics + points_social_studies + points_english + points_science
def average_points : ℚ := total_points / number_of_subjects

def count_above_average_points : ℕ := 
  (if points_korean > average_points then 1 else 0) + 
  (if points_mathematics > average_points then 1 else 0) +
  (if points_social_studies > average_points then 1 else 0) +
  (if points_english > average_points then 1 else 0) +
  (if points_science > average_points then 1 else 0)

theorem giyoon_above_average_subjects : count_above_average_points = 2 := by
  sorry

end giyoon_above_average_subjects_l714_714446


namespace number_of_elements_indeterminate_l714_714278

theorem number_of_elements_indeterminate 
  (S : Set ℝ) (n : ℕ) (avg_old : ℝ) (avg_new : ℝ)
  (h1 : avg_old = 20)
  (h2 : avg_new = 100)
  (h3 : ∀ x ∈ S, x * 5 ∈ (λ y, y / 5 '' S)) :
  ¬∃ n, true :=
by
  sorry

end number_of_elements_indeterminate_l714_714278


namespace cos_120_eq_neg_one_half_l714_714007

theorem cos_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_120_eq_neg_one_half_l714_714007


namespace units_digit_37_power_l714_714070

theorem units_digit_37_power (k n : ℕ) :
  let cycle := [7, 9, 3, 1],
      effective_exponent := (5 * (14^14)) % 4 in
  effective_exponent = 0 →
  cycle.nth effective_exponent = 1 :=
by
  let cycle := [7, 9, 3, 1]
  let effective_exponent := (5 * (14^14)) % 4
  have h1 : effective_exponent = 0 := sorry
  show cycle.nth effective_exponent = 1, from sorry

end units_digit_37_power_l714_714070


namespace range_of_a_l714_714440

theorem range_of_a (a : ℝ) (h : ¬ (1^2 - 2*1 + a > 0)) : 1 ≤ a := sorry

end range_of_a_l714_714440


namespace order_stats_markov_two_values_order_stats_not_markov_three_values_l714_714898

open ProbTheory

noncomputable section

variable {Ω : Type*} (X : ℕ → Ω → ℝ)

def are_iid (X : ℕ → Ω → ℝ) [fintype Ω] [measure_space Ω] : Prop :=
∀ i j, independent (X i) (X j) ∧ identically_distributed (X i) (X j)

def order_statistics (X : ℕ → Ω → ℝ) (n : ℕ) : ℕ → Ω → ℝ :=
λ k ω, ... -- define the order statistic function (e.g., using a sorting mechanism)

def is_Markov_chain {α : Type*} (X : ℕ → α) : Prop :=
∀ n a b, Pr[X (n + 1) = b | X n = a, X (n - 1) = a', ... X 0 = a_0]
  = Pr[X (n + 1) = b | X n = a]

theorem order_stats_markov_two_values (X : ℕ → Ω → ℝ)
  [fintype Ω] [measure_space Ω] :
  are_iid X ∧ (∀ ω, X 0 ω ∈ {0, 1}) →
  is_Markov_chain (order_statistics X n) :=
sorry

theorem order_stats_not_markov_three_values (X : ℕ → Ω → ℝ)
  [fintype Ω] [measure_space Ω] :
  are_iid X ∧ (∀ ω, X 0 ω ∈ {1, 2, 3}) →
  ¬ is_Markov_chain (order_statistics X n) :=
sorry

end order_stats_markov_two_values_order_stats_not_markov_three_values_l714_714898


namespace exists_small_rectangle_all_distances_odd_or_even_l714_714336

theorem exists_small_rectangle_all_distances_odd_or_even
  (m n : ℕ) (hm : Odd m) (hn : Odd n)
  (rectangles : list (ℕ × ℕ × ℕ × ℕ)) :
  ∃ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ rectangles ∧
    ((Even x1 ∧ Even (m - x2) ∧ Even y1 ∧ Even (n - y2)) ∨ 
     (Odd x1 ∧ Odd (m - x2) ∧ Odd y1 ∧ Odd (n - y2))) :=
sorry

end exists_small_rectangle_all_distances_odd_or_even_l714_714336


namespace trajectory_of_midpoint_l714_714421

noncomputable def midpoint_trajectory_equation (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16)

theorem trajectory_of_midpoint
  (C D : ℝ × ℝ)
  (h_CD_moving_chord : ∃ t : ℝ, C = (5 * t, 5 * real.sqrt (1 - t^2)) ∧ D = (5 * t, -5 * real.sqrt (1 - t^2)))
  (h_cd_length : real.dist C D = 6) :
  ∃ M : ℝ × ℝ, M = (((C.fst + D.fst) / 2), ((C.snd + D.snd) / 2)) ∧ midpoint_trajectory_equation M.fst M.snd :=
by
  sorry

end trajectory_of_midpoint_l714_714421


namespace transformed_sine_function_l714_714958

-- Lean 4 statement to verify the resulting function after transformations
theorem transformed_sine_function :
  ∀ x : ℝ, (sin (2 * (x - π / 4)) + 1) = 2 * sin^2 x :=
by
  sorry

end transformed_sine_function_l714_714958


namespace rate_per_kg_mangoes_l714_714360

-- Defining the given conditions
def quantity_grapes : ℕ := 7
def rate_per_kg_grapes : ℕ := 70
def total_amount_paid : ℕ := 985
def quantity_mangoes : ℕ := 9

-- Theorem statement using these definitions
theorem rate_per_kg_mangoes :
  let cost_grapes := quantity_grapes * rate_per_kg_grapes in
  let cost_mangoes := total_amount_paid - cost_grapes in
  cost_mangoes / quantity_mangoes = 55 :=
by
  -- Lean proof will go here; skipping for now as per instructions
  sorry

end rate_per_kg_mangoes_l714_714360


namespace math_proof_problem_l714_714584

def valid_numbers (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def ABC (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
noncomputable def DE (d e : ℕ) : ℕ := 10 * d + e

theorem math_proof_problem :
  ∃ (a b c d e f g h : ℕ),
    valid_numbers a ∧ valid_numbers b ∧ valid_numbers c ∧ valid_numbers d ∧ valid_numbers e ∧ valid_numbers f ∧ valid_numbers g ∧ valid_numbers h ∧
    list.nodup [a, b, c, d, e, g, h] ∧
    ABC a b c = 182 ∧
    DE d e = 91 ∧
    f = 2 ∧
    f = h - 7 ∧
    A / E = f :=
by
  sorry

end math_proof_problem_l714_714584


namespace required_lemons_for_20_gallons_l714_714847

-- Conditions
def lemons_for_50_gallons : ℕ := 40
def gallons_for_lemons : ℕ := 50
def additional_lemons_per_10_gallons : ℕ := 1
def number_of_gallons : ℕ := 20
def base_lemons (g: ℕ) : ℕ := (lemons_for_50_gallons * g) / gallons_for_lemons
def additional_lemons (g: ℕ) : ℕ := (g / 10) * additional_lemons_per_10_gallons
def total_lemons (g: ℕ) : ℕ := base_lemons g + additional_lemons g

-- Proof statement
theorem required_lemons_for_20_gallons : total_lemons number_of_gallons = 18 :=
by
  sorry

end required_lemons_for_20_gallons_l714_714847


namespace length_FL_l714_714180

variable {K L M G F : Point}
variable {FL KG LG : ℝ}

-- Assume KLM is a right triangle at L, G is on KL such that KG = 5 and LG = 4
-- A circle is constructed with KM as diameter intersecting KL at G
-- Tangent to the circle at G intersects ML at F

axiom right_triangle_KLM : right_triangle K L M
axiom circle_KM_diameter : ∃ (circle : Circle), circle.diameter = KM ∧ circle.contains G
axiom tangent_at_G : tangent_circle_at G circle_KM_diameter ∧ intersects F ML
axiom KG_eq_5 : dist K G = 5
axiom LG_eq_4 : dist L G = 4

theorem length_FL : dist F L = 3 :=
sorry

end length_FL_l714_714180


namespace number_of_permutations_l714_714390

-- Define the conditions
def perm_condition (p : Fin 5 → Fin 5) : Prop :=
  ¬ (p 0 = 0 ∨ p 0 = 1) ∧
  ¬ (p 1 = 1 ∨ p 1 = 2) ∧
  ¬ (p 2 = 4) ∧
  ¬ (p 3 = 3 ∨ p 3 = 4) ∧
  ¬ (p 4 = 2 ∨ p 4 = 3)

-- Define the main theorem
theorem number_of_permutations : Finset.filter perm_condition (Finset.univ : Finset (Fin 5 → Fin 5)).card = 16 :=
by sorry

end number_of_permutations_l714_714390


namespace min_value_eq_l714_714160

open Real

noncomputable def min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 4 * a + 3 * b - 1 = 0) : ℝ :=
  1 / (2 * a + b) + 1 / (a + b)

theorem min_value_eq : ∀ (a b : ℝ), 0 < a → 0 < b → 4 * a + 3 * b - 1 = 0 →
  (min_value a b (by assumption) (by assumption) (by assumption) = 3 + 2 * sqrt 2) :=
by
  intros
  sorry

end min_value_eq_l714_714160


namespace probability_four_green_marbles_l714_714891

open_locale big_operators

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def probability_green : ℚ :=
  8 / 15

noncomputable def probability_purple : ℚ :=
  7 / 15

theorem probability_four_green_marbles :
  (binomial 7 4) * (probability_green ^ 4) * (probability_purple ^ 3) = 49172480 / 170859375 :=
by
  sorry

end probability_four_green_marbles_l714_714891


namespace simplify_trigonometric_expression_l714_714968

theorem simplify_trigonometric_expression (x : ℝ) (hx : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 / sin x := 
by sorry

end simplify_trigonometric_expression_l714_714968


namespace total_games_scheduled_l714_714246

noncomputable def numTeamsPerDivision := 5
noncomputable def numDivisions := 3
noncomputable def numInterDivisionGamesPerTeam := 10
noncomputable def numNonConferenceGamesPerTeam := 2

theorem total_games_scheduled : 
  let intra_division_games := (numTeamsPerDivision * (numTeamsPerDivision - 1) / 2) * 3 * numDivisions
  let inter_division_games := numInterDivisionGamesPerTeam * numTeamsPerDivision * numDivisions
  let non_conference_games := numNonConferenceGamesPerTeam * (numTeamsPerDivision * numDivisions)
  intra_division_games + inter_division_games + non_conference_games = 270 :=
by
  let intra_division_games := (numTeamsPerDivision * (numTeamsPerDivision - 1) / 2) * 3 * numDivisions
  let inter_division_games := numInterDivisionGamesPerTeam * numTeamsPerDivision * numDivisions
  let non_conference_games := numNonConferenceGamesPerTeam * (numTeamsPerDivision * numDivisions)
  have h1 : intra_division_games = 90 := by
    unfold Coe.coe
    simp
    sorry
  have h2 : inter_division_games = 150 := by
    unfold Coe.coe
    simp
    sorry
  have h3 : non_conference_games = 30 := by
    unfold Coe.coe
    simp
    sorry
  show intra_division_games + inter_division_games + non_conference_games = 270 from
    by
      rw [h1, h2, h3]
      rfl

end total_games_scheduled_l714_714246


namespace delta_max_success_ratio_l714_714867

theorem delta_max_success_ratio (x y z w : ℕ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_z_pos : 0 < z) (h_w_pos : 0 < w) 
  (h_x_ratio : (x:ℚ) / y < 9/14) (h_z_ratio : (z:ℚ) / w < 6/11) 
  (h_y_constraint : y = 500 - w) 
  (h_x_y_ineq : (14:ℚ) / 9 * x < y) (h_z_w_ineq : (11:ℚ) / 6 * z < w) :
  (x + z : ℚ) / 500 ≤ 409 / 500 := 
begin
  sorry
end

end delta_max_success_ratio_l714_714867


namespace tan_alpha_plus_pi_over_4_rational_expression_of_trig_l714_714084

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + Real.pi / 4) = -1 / 7 := 
by 
  sorry

theorem rational_expression_of_trig (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_rational_expression_of_trig_l714_714084


namespace sector_area_l714_714810

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) :
  (1 / 2) * α * r ^ 2 = Real.pi := by
  sorry

end sector_area_l714_714810


namespace ellipse_equation_proof_fixed_point_proof_l714_714818

variables (a b : ℝ) (e : ℝ)
variables (k : ℝ)
variables (x₁ x₂ y₁ y₂ : ℝ)

-- Conditions
def is_ellipse := (a > b) ∧ (b > 0)
def eccentricity := e = (real.sqrt 6) / 3 
def major_axis_length := 2 * a = 2 * real.sqrt 3
def ellipse_equation := ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1

def coordinates_intersect (A B : ℝ × ℝ) :=
  A.2 = k * A.1 - (1 / 2) ∧
  B.2 = k * B.1 - (1 / 2) ∧
  (A.1 ^ 2 / 3) + (A.2 ^ 2) = 1 ∧
  (B.1 ^ 2 / 3) + (B.2 ^ 2) = 1

def vector_perpendicular (A B M : ℝ × ℝ) :=
  let PA := (A.1 - M.1, A.2 - M.2)
  let PB := (B.1 - M.1, B.2 - M.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0

-- Proof problem statements
theorem ellipse_equation_proof (h1 : is_ellipse a b) (h2 : eccentricity e) (h3 : major_axis_length a):
  ellipse_equation a b :=
sorry

theorem fixed_point_proof (h1 : is_ellipse a b)
                         (h2 : eccentricity e) 
                         (h3 : major_axis_length a)
                         (h4 : ellipse_equation a b)
                         (h5 : coordinates_intersect (x₁, y₁) (x₂, y₂))
                         (k : ℝ) :
  ∃ M, M = (0, 1) ∧ vector_perpendicular (x₁, y₁) (x₂, y₂) M :=
sorry

end ellipse_equation_proof_fixed_point_proof_l714_714818


namespace median_of_data_median_mean_of_sorted_data_l714_714511

def data : List ℕ := [190, 197, 184, 188, 191, 187]

def sorted_data : List ℕ := data.qsort (· < ·)

theorem median_of_data : (sorted_data.nth 2).get_or_else 0 + (sorted_data.nth 3).get_or_else 0 = 378 :=
by
  sorry

theorem median_mean_of_sorted_data : (sorted_data.nth 2).get_or_else 0 + (sorted_data.nth 3).get_or_else 0 = 378 → (378 / 2 = 189) :=
by
  sorry

end median_of_data_median_mean_of_sorted_data_l714_714511


namespace MN_squared_l714_714171

theorem MN_squared (PQ QR RS SP : ℝ) (h1 : PQ = 15) (h2 : QR = 15) (h3 : RS = 20) (h4 : SP = 20) (angle_S : ℝ) (h5 : angle_S = 90)
(M N: ℝ) (Midpoint_M : M = (QR / 2)) (Midpoint_N : N = (SP / 2)) : 
MN^2 = 100 := by
  sorry

end MN_squared_l714_714171


namespace rhombus_diagonal_l714_714249

theorem rhombus_diagonal (d1 d2 Area : ℝ) (h1 : Area = 90) (h2 : d1 = 15) : d2 = 12 :=
by
  -- Area formula for a rhombus: Area = (d1 * d2) / 2
  have h : 90 = (15 * d2) / 2 := by
    rw [h1, h2]
    sorry
  -- Solve for d2
  sorry

end rhombus_diagonal_l714_714249


namespace no_solution_for_triples_l714_714068

theorem no_solution_for_triples :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 66) ∧ (a * c + b * c = 35) :=
by {
  sorry
}

end no_solution_for_triples_l714_714068


namespace equivalent_statements_l714_714309

section 
    variables (P Q : Prop)

    theorem equivalent_statements (P Q : Prop) :
        (P → Q) ↔ (P → Q) ∧ (¬ Q → ¬ P) ∧ (¬ P ∨ Q) :=
    by
        split
        sorry
end

end equivalent_statements_l714_714309


namespace largest_six_digit_integer_with_product_40320_l714_714630

/-- 
Prove that the largest six-digit integer such that the product
of its digits equals to 8*7*6*5*4*3*2*1, is 987744 
-/
theorem largest_six_digit_integer_with_product_40320 : 
  ∃ (n : ℤ), (n > 99999) ∧ (n < 1000000) ∧ (∏ (d : ℕ) in (n.digits 10), d = 40320) ∧ (∀ m, (m > 99999) ∧ (m < 1000000) ∧ (∏ (d : ℕ) in (m.digits 10), d = 40320) → m ≤ n) :=
sorry

end largest_six_digit_integer_with_product_40320_l714_714630


namespace problem_l714_714996

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem problem (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∀ a b : ℝ, (0 ≤ a ∧ a ≤ 2 * π) → (0 ≤ b ∧ b ≤ 2 * π) → a ≤ b → f a ≤ f b := 
sorry

end problem_l714_714996


namespace remove_n_rows_n_columns_no_pieces_l714_714627

theorem remove_n_rows_n_columns_no_pieces (n : ℕ) (pieces : ℕ) (board : fin (2 * n) → fin (2 * n) → bool) :
  (pieces = 3 * n) →
  (∀ i j, board i j = true → pieces = pieces - 1) →
  ∃ rows cols : fin (2 * n) → bool, 
    (fin.sum rows = n) ∧ (fin.sum cols = n) ∧
    ∀ i j, rows i = true → cols j = true → board i j = false :=
by
  sorry

end remove_n_rows_n_columns_no_pieces_l714_714627


namespace probability_systematic_sampling_l714_714779

theorem probability_systematic_sampling :
  ∀ (students : ℕ) (eliminated : ℕ) (selected : ℕ),
  students = 2004 → eliminated = 4 → selected = 50 →
  (selected / (students - eliminated)) = 25 / 1002 :=
by
  intros students eliminated selected h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact div_eq_of_eq_mul_left (by norm_num) (by norm_num) sorry

end probability_systematic_sampling_l714_714779


namespace max_min_x2_min_xy_plus_y2_l714_714535

theorem max_min_x2_min_xy_plus_y2 (x y : ℝ) (h : x^2 + x * y + y^2 = 3) :
  1 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 9 :=
by sorry

end max_min_x2_min_xy_plus_y2_l714_714535


namespace coefficient_of_x4_in_expansion_l714_714064

theorem coefficient_of_x4_in_expansion : 
  let x := sorry in
  let expr := (x^2 + (2 / x))^5 in
  let general_term (r : ℕ) := (Nat.choose 5 r) * 2^r * x^(10 - 3 * r) in
  (∃ r : ℕ, 10 - 3 * r = 4 ∧ general_term 2 = 40) ∧ 
  coefficient (expr) 4 = 40 := 
by 
  sorry

end coefficient_of_x4_in_expansion_l714_714064


namespace cos_120_eq_neg_half_l714_714031

-- Definitions based on the given conditions
def Q := (120 : ℝ)
def reference_angle := (60 : ℝ)

-- Given angle relations
def angle_in_second_quadrant := (Q = 180 - reference_angle)
def cosine_reference_angle := real.cos reference_angle = 1/2

-- Conditions connected to the specific quadrant properties
def cosine_negative_in_second_quadrant := (Q > 90 ∧ Q < 180)

-- Prove that cosine of 120 degrees is -1/2
theorem cos_120_eq_neg_half : real.cos (Q * real.pi / 180) = -1/2 :=
by
  -- Include the formal proof here
  sorry

end cos_120_eq_neg_half_l714_714031


namespace lambda_range_l714_714163

theorem lambda_range (λ : ℝ) (h : 0 < λ) :
  (∀ x : ℝ, 0 < x → e^(λ * x) - log x / λ ≥ 0) → λ ≥ 1 / Real.exp 1 :=
sorry

end lambda_range_l714_714163


namespace multiples_of_5_or_7_but_not_35_l714_714150

theorem multiples_of_5_or_7_but_not_35 (n : ℕ) : n = 943 :=
  let multiples_of (k m : ℕ) := k / m
  let five := multiples_of 3000 5
  let seven := multiples_of 3000 7
  let thirtyfive := multiples_of 3000 35
  by calc
    n = five + seven - thirtyfive : sorry
      ... = 943 : sorry

end multiples_of_5_or_7_but_not_35_l714_714150


namespace pond_75_percent_free_on_day_18_l714_714668

-- Definitions based on conditions
noncomputable def algae_coverage (n : ℕ) : ℝ := (1 / 3^n)

-- Main theorem statement
theorem pond_75_percent_free_on_day_18 :
  (algae_coverage 18 : ℝ) ≤ 1/4 :=
begin
  have coverage_18 : algae_coverage 18 = 1 / 3^18 := rfl,
  have sub_fraction : (1 / 3^18 : ℝ) ≤ 1/4,
  { sorry },
  exact sub_fraction,
end

end pond_75_percent_free_on_day_18_l714_714668


namespace triangle_diameter_l714_714346

theorem triangle_diameter (n : ℕ) (a : ℝ) (h1 : n = 3) (h2 : a = 1) : diameter n a = 1 := 
sorry

end triangle_diameter_l714_714346


namespace sqrt_t6_plus_t4_l714_714396

open Real

theorem sqrt_t6_plus_t4 (t : ℝ) : sqrt (t^6 + t^4) = t^2 * sqrt (t^2 + 1) :=
by sorry

end sqrt_t6_plus_t4_l714_714396


namespace probability_of_odd_score_l714_714866

axiom inner_ring : Finset ℕ := {1, 2}
axiom middle_ring : Finset ℕ := {1, 2, 3}
axiom outer_ring : Finset ℕ := {1, 2, 3, 4}

def score (a b c : ℕ) : ℕ := a + b + c

-- Define the event that a score is odd
def score_is_odd (a b c : ℕ) : Prop := (score a b c) % 2 = 1

-- Define the total number of possible outcomes
def total_outcomes : ℕ := inner_ring.card * middle_ring.card * outer_ring.card

-- Count the number of outcomes where the score is odd
noncomputable def count_odd_scores : ℕ :=
  (inner_ring.product (middle_ring.product outer_ring)).count (λ ⟨a, ⟨b, c⟩⟩, score_is_odd a b c)

-- Define the probability as a ratio of favorable outcomes to total outcomes
noncomputable def probability_odd_score : ℚ :=
  (count_odd_scores : ℚ) / (total_outcomes : ℚ)

theorem probability_of_odd_score : probability_odd_score = 5 / 6 :=
by sorry

end probability_of_odd_score_l714_714866


namespace total_marbles_l714_714320

namespace MarbleBag

def numBlue : ℕ := 5
def numRed : ℕ := 9
def probRedOrWhite : ℚ := 5 / 6

theorem total_marbles (total_mar : ℕ) (numWhite : ℕ) (h1 : probRedOrWhite = (numRed + numWhite) / total_mar)
                      (h2 : total_mar = numBlue + numRed + numWhite) :
  total_mar = 30 :=
by
  sorry

end MarbleBag

end total_marbles_l714_714320


namespace price_of_fruit_l714_714995

theorem price_of_fruit
  (price_milk_per_liter : ℝ)
  (milk_per_batch : ℝ)
  (fruit_per_batch : ℝ)
  (cost_for_three_batches : ℝ)
  (F : ℝ)
  (h1 : price_milk_per_liter = 1.5)
  (h2 : milk_per_batch = 10)
  (h3 : fruit_per_batch = 3)
  (h4 : cost_for_three_batches = 63)
  (h5 : 3 * (milk_per_batch * price_milk_per_liter + fruit_per_batch * F) = cost_for_three_batches) :
  F = 2 :=
by sorry

end price_of_fruit_l714_714995


namespace ceil_eval_l714_714058

-- Define the ceiling function and the arithmetic operations involved
example : Real := let inside := (8 - (3 / 4)) in 
                  let multiplied := 4 * inside in 
                  ⌈multiplied⌉
                  
theorem ceil_eval :  ⌈4 * (8 - (3 / 4))⌉ = 29 := 
by
-- We'll skip the proof part using sorry
sorry

end ceil_eval_l714_714058


namespace cubic_polynomial_range_l714_714265

-- Define the conditions and the goal in Lean
theorem cubic_polynomial_range :
  ∀ x : ℝ, (x^2 - 5 * x + 6 < 0) → (41 < x^3 + 5 * x^2 + 6 * x + 1) ∧ (x^3 + 5 * x^2 + 6 * x + 1 < 91) :=
by
  intros x hx
  have h1 : 2 < x := sorry
  have h2 : x < 3 := sorry
  have h3 : (x^3 + 5 * x^2 + 6 * x + 1) > 41 := sorry
  have h4 : (x^3 + 5 * x^2 + 6 * x + 1) < 91 := sorry
  exact ⟨h3, h4⟩ 

end cubic_polynomial_range_l714_714265


namespace rotated_line_intercept_x_l714_714551

open Real

theorem rotated_line_intercept_x (m : ℝ) (n : ℝ) (p q : ℝ) (rotation_angle : ℝ) :
  (∀ x y, 2 * x + 3 * y - 6 = 0 → y = - (2 / 3) * x + 2) ∧
  (rotation_angle = π / 6) ∧ 
  (p, q) = (3, -2) →
  (∃ x_intercept : ℝ, x_intercept = ((-2 * sqrt 3 + 6) * (3 * sqrt 3 - 2)) / (2 * sqrt 3 + 3)) :=
begin
  sorry
end

end rotated_line_intercept_x_l714_714551


namespace ellipse_equation_slope_constant_max_triangle_area_l714_714103

noncomputable def ellipse : Type := ℝ × ℝ → Prop

noncomputable def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (P : ℝ × ℝ) : Prop :=
  P.1^2 / a^2 + P.2^2 / b^2 = 1

def point_F : ℝ × ℝ := (2, 0)
def point_A : ℝ × ℝ := (2, real.sqrt 2)

def focuses_at_F (a : ℝ) : Prop := a = 2

def equation_of_ellipse (P : ℝ × ℝ) : Prop := P.1^2 / 8 + P.2^2 / 4 = 1

theorem ellipse_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ is_ellipse a b (and.intro sorry sorry) point_A ∧ focuses_at_F a ∧ ∀ P : ℝ × ℝ, P ∈ ellipse → equation_of_ellipse P :=
sorry

theorem slope_constant :
  ∀ M N : ℝ × ℝ, (M.1^2 / 8 + M.2^2 / 4 = 1) ∧ (N.1^2 / 8 + N.2^2 / 4 = 1) ∧ ((N.2 - M.2) / (N.1 - M.1) = real.sqrt 2 / 2) :=
sorry

theorem max_triangle_area :
  ∃ max_area : ℝ, (∃ M N : ℝ × ℝ, (M.1^2 / 8 + M.2^2 / 4 = 1) ∧ (N.1^2 / 8 + N.2^2 / 4 = 1) ∧ max_area = 2 * real.sqrt 2) :=
sorry

end ellipse_equation_slope_constant_max_triangle_area_l714_714103


namespace price_bound_l714_714074

namespace SequencePrice

def price (s : List ℝ) : ℝ :=
  s.scanl (+) 0 |>.tail |>.map abs |>.maximum' sorry

def minimumPrice (s : List ℝ) : ℝ :=
  sorry -- Definition of the minimum price D by Dave

def greedyPrice (s : List ℝ) : ℝ :=
  sorry -- Definition of the greedy price G by George

theorem price_bound (s : List ℝ) :
  ∃ c : ℝ, (∀ s, greedyPrice s ≤ c * minimumPrice s) ∧ c = 2 :=
sorry

end SequencePrice

end price_bound_l714_714074


namespace votes_ratio_l714_714882

theorem votes_ratio (E S R : ℕ) (h1 : E = 2 * S) (h2 : E = 160) (h3 : R = 16) : S / R = 5 := by
  -- Definitions based on the conditions
  have h4 : S = 160 / 2, from (eq_div_of_mul_eq (by norm_num) h2).symm
  have h5 : S = 80, from (by norm_num : 160 / 2 = 80).trans h4
  -- Simplification and the final ratio
  sorry

end votes_ratio_l714_714882


namespace find_m_l714_714178

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {m : ℕ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def q_condition (q : ℝ) : Prop :=
  abs q ≠ 1

def a_m_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem to prove
theorem find_m (h1 : geometric_sequence a q) (h2 : initial_condition a) (h3 : q_condition q) (h4 : a_m_condition a m) : m = 11 :=
  sorry

end find_m_l714_714178


namespace proof_50th_permutation_l714_714589

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def nth_permutation (l : List ℕ) (n : ℕ) : List ℕ :=
  let f := factorial (l.length - 1)
  if l = [] then [] else
  let (q, r) := n.div_mod f
  let x := l.nth_le q ((Nat.lt_of_sub_lt_sub (Nat.zero_lt_succ _)).1 (Nat.pos_iff_ne_zero.2 (factorial_ne_zero (l.length - 1))))
  x::nth_permutation (List.erase l x) r

theorem proof_50th_permutation : (nth_permutation [1, 2, 3, 4, 5] 49) = [3, 1, 2, 5, 4] :=
  by sorry

end proof_50th_permutation_l714_714589


namespace common_chords_equal_l714_714480

theorem common_chords_equal
  (A B C H A1 B1 C1 : Point)
  (h_a : Line)
  (circle_on_A1H circle_on_B1H circle_on_C1H : Circle)
  (not_right_triangle : ¬(triangle_is_right A B C))
  (orthocenter_H : orthocenter A B C H)
  (feet_of_altitudes : feet_altitudes A B C H A1 B1 C1)
  (circle_constr_A1H : circle_on_segment circle_on_A1H A1 H)
  (circle_constr_B1H : circle_on_segment circle_on_B1H B1 H)
  (circle_constr_C1H : circle_on_segment circle_on_C1H C1 H)
  : common_chords_equal circle_on_A1H circle_on_B1H circle_on_C1H := sorry

end common_chords_equal_l714_714480


namespace percentage_problem_l714_714667

theorem percentage_problem (P : ℕ) (n : ℕ) (h_n : n = 16)
  (h_condition : (40: ℚ) = 0.25 * n + 2) : P = 250 :=
by
  sorry

end percentage_problem_l714_714667


namespace dice_probability_sum_6_l714_714655

-- Define the die sides
def sides := {1, 2, 3, 4}

-- Define the number of dice and desired sum
def num_dice := 3
def desired_sum := 6

-- Define the probability function
noncomputable def probability_of_sum (s : Finset (Finset ℕ)) (sum : ℕ) : ℚ :=
  (s.filter (λ xs => xs.sum = sum)).card.toRat / s.card.toRat

-- Calculate the probability that the sum of the dice is 6
theorem dice_probability_sum_6 :
  let outcomes := {xs : Finset ℕ | xs.card = num_dice ∧ xs ⊆ sides}
  ∃ p q : ℕ, probability_of_sum outcomes desired_sum = p / q ∧ Nat.gcd p q = 1 ∧ (p + q = 37) := 
  by
  sorry

end dice_probability_sum_6_l714_714655


namespace triangle_area_l714_714899

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area (a b : ℝ × ℝ) : 
  let area_parallelogram := (a.1 * b.2 - a.2 * b.1).abs in
  (1 / 2) * area_parallelogram = 4.5 :=
by
  sorry

end triangle_area_l714_714899


namespace min_circle_area_min_circle_area_is_pi_l714_714365

noncomputable def f (x : ℝ) : ℝ :=
  1 + x - (x^2) / 2 + (x^3) / 3 - (x^4) / 4 + ... - (x^2016) / 2016

noncomputable def F (x : ℝ) : ℝ := f (x + 4)

theorem min_circle_area (a b : ℤ) (h : a < b) 
  (h1 : ∀ x, F x = 0 → a < x ∧ x < b) :
  b - a = 1 :=
sorry

theorem min_circle_area_is_pi (a b : ℤ) (h : a < b) 
  (h1 : ∀ x, F x = 0 → a < x ∧ x < b) :
  let r := real.sqrt (b - a) in
  real.pi * r^2 = real.pi :=
sorry

end min_circle_area_min_circle_area_is_pi_l714_714365


namespace prove_inequality_l714_714692

-- Define the sequence {b_n}
noncomputable def b_n (α : ℕ → ℕ) : ℕ → ℚ
| 1 := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b_n α n)

-- Example α values for simplification like α_k = 1
def example_α (k : ℕ) : ℕ := 1

-- The statement to be proved
theorem prove_inequality (α : ℕ → ℕ) (h : ∀ k, 0 < α k) : (b_n α 4 < b_n α 7) :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end prove_inequality_l714_714692


namespace find_a_range_l714_714111

open Set

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

theorem find_a_range :
  (p a ∧ ¬ q a ∨ ¬ p a ∧ q a) ↔ (-1 < a ∧ a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l714_714111


namespace tangent_circumcircle_l714_714211

noncomputable theory
open_locale classical

universe u
variables {α : Type u} [decidable_eq α]

structure Triangle (α : Type u) :=
(A B C : α)

structure Circle (α : Type u) :=
(center : α)
(radius : ℝ)

structure Line (α : Type u) :=
(point1 point2 : α)

def reflect_in_line {α : Type u} (l : Line α) (p : α) : α := sorry

def circumcircle (t : Triangle α) : Circle α := sorry

def tangent (c : Circle α) (l : Line α) : Prop := sorry

def is_tangent (c1 c2 : Circle α) : Prop := sorry

variables {A B C T : α} (ω : Circle α) (t t_a t_b t_c : Line α)

theorem tangent_circumcircle
  (hABC : acute_triangle A B C)
  (hω : circumcircle (Triangle.mk A B C) = ω)
  (ht : tangent ω t)
  (hta : t_a = reflect_in_line (Line.mk B C) t)
  (htb : t_b = reflect_in_line (Line.mk C A) t)
  (htc : t_c = reflect_in_line (Line.mk A B) t) :
  is_tangent (circumcircle (Triangle.mk t_a.point1 t_b.point1 t_c.point1)) ω :=
sorry

end tangent_circumcircle_l714_714211


namespace triangle_side_length_l714_714514

theorem triangle_side_length (A B C : ℝ) (h1 : AC = Real.sqrt 2) (h2: AB = 2)
  (h3 : (Real.sqrt 3 * Real.sin A + Real.cos A) / (Real.sqrt 3 * Real.cos A - Real.sin A) = Real.tan (5 * Real.pi / 12)) :
  BC = Real.sqrt 2 := 
sorry

end triangle_side_length_l714_714514


namespace xy_nonzero_implies_iff_l714_714458

variable {x y : ℝ}

theorem xy_nonzero_implies_iff (h : x * y ≠ 0) : (x + y = 0) ↔ (x / y + y / x = -2) :=
sorry

end xy_nonzero_implies_iff_l714_714458


namespace enclosed_area_eq_8_over_3_l714_714581

noncomputable def f : ℝ → ℝ := λ x, -2 * x^2 + 7 * x - 6
noncomputable def g : ℝ → ℝ := λ x, - x

theorem enclosed_area_eq_8_over_3 : 
  ∫ x in 1..3, (f x - g x) = 8 / 3 :=
by sorry

end enclosed_area_eq_8_over_3_l714_714581


namespace find_cost_price_l714_714643

def selling_price : ℝ := 150
def profit_percentage : ℝ := 25

theorem find_cost_price (cost_price : ℝ) (h : profit_percentage = ((selling_price - cost_price) / cost_price) * 100) : 
  cost_price = 120 := 
sorry

end find_cost_price_l714_714643


namespace find_min_a_plus_b_l714_714933

open Nat

noncomputable def smallest_a_plus_b : ℕ :=
  let candidates := (1, 1).succ_powerset (1000, 1000).succ_powerset -- to explore a range up to 1000; this range should be set reasonably high
  candidates.filter (λ (a, b), (a^a % b^b = 0) ∧ (a % b ≠ 0) ∧ (gcd b 210 = 1)) |>.map (λ (a, b), a + b) |> min

theorem find_min_a_plus_b : smallest_a_plus_b = 374 := by
  sorry

end find_min_a_plus_b_l714_714933


namespace find_a_plus_b_l714_714789

-- Definitions of the conditions
def equation (x a b : ℝ) : ℝ := a + b / (x + 1)

theorem find_a_plus_b (a b : ℝ)
  (h1 : equation (-2) a b = 2)
  (h2 : equation (-6) a b = 6) :
  a + b = 12 :=
sorry

end find_a_plus_b_l714_714789


namespace matthew_ate_8_l714_714222

variable (M P A K : ℕ)

def kimberly_ate_5 : Prop := K = 5
def alvin_eggs : Prop := A = 2 * K - 1
def patrick_eggs : Prop := P = A / 2
def matthew_eggs : Prop := M = 2 * P

theorem matthew_ate_8 (M P A K : ℕ) (h1 : kimberly_ate_5 K) (h2 : alvin_eggs A K) (h3 : patrick_eggs P A) (h4 : matthew_eggs M P) : M = 8 := by
  sorry

end matthew_ate_8_l714_714222


namespace diplomats_attended_conference_l714_714559

variable (D : ℕ) -- declare D

-- declare the conditions
def French_diplomats := 20
def not_Hindi := 32
def neither_FnorH := 0.20 * D
def both_F_and_H := 0.10 * D
def either_F_or_H_or_both := D - neither_FnorH

-- the key sets and their cardinalities
def F := French_diplomats
def H := D - not_Hindi
def F_inter_H := both_F_and_H
def F_union_H := either_F_or_H_or_both

theorem diplomats_attended_conference 
  (h_union: F_union_H = F + H - F_inter_H) : 
  D = 120 := by
  sorry

end diplomats_attended_conference_l714_714559


namespace ceiling_example_l714_714053

theorem ceiling_example : ⌈4 * (8 - 3 / 4)⌉ = 29 := 
by
  sorry

end ceiling_example_l714_714053


namespace last_four_digits_5_pow_2011_l714_714225

theorem last_four_digits_5_pow_2011 : 
  (5^2011 % 10000) = 8125 :=
by
  -- Definitions based on conditions in the problem
  have h5 : 5^5 % 10000 = 3125 := sorry
  have h6 : 5^6 % 10000 = 5625 := sorry
  have h7 : 5^7 % 10000 = 8125 := sorry
  
  -- Prove using periodicity and modular arithmetic
  sorry

end last_four_digits_5_pow_2011_l714_714225


namespace curve_C1_equation_curve_C2_equation_min_distance_midpoint_C3_min_max_distance_Q_C3_range_2x_plus_y_range_a_l714_714817

noncomputable theory

-- Definitions of the curves
def C1 (t : ℝ) : ℝ × ℝ := (-4 + cos t, 3 + sin t)
def C2 (θ : ℝ) : ℝ × ℝ := (6 * cos θ, 2 * sin θ)

-- Definition of the line C3 in parametric form
def C3 (t : ℝ) : ℝ × ℝ := (-3 * sqrt 3 + sqrt 3 * t, -3 - t)

-- 1. Prove standard form equations of C1 and C2
theorem curve_C1_equation : ∀ (x y : ℝ), (∃ t : ℝ, (x, y) = C1 t) ↔ (x + 4)^2 + (y - 3)^2 = 1 := sorry
theorem curve_C2_equation : ∀ (x y : ℝ), (∃ θ : ℝ, (x, y) = C2 θ) ↔ (x^2 / 36) + (y^2 / 4) = 1 := sorry

-- 2. Prove the minimum distance between M and the line C3
-- where P corresponds to t = π/2 and Q is a point on C2
theorem min_distance_midpoint_C3 : 
  let P := C1 (π / 2) 
  → ∀ (θ : ℝ), 
  let Q := C2 θ 
  → let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) 
  → ∀ t : ℝ, (M = C3 t) 
  → distance M C3 = |sqrt 3 * sin(θ + π / 3) + 4 * sqrt 3 - 1|
  → distance M C3 ≥ 3 * sqrt 3 - 1 := sorry

-- 3. Prove min and max distances between Q on C2 and the line C3
theorem min_max_distance_Q_C3 : 
  ∀ (θ : ℝ), 
  let Q := C2 θ 
  → ∀ t : ℝ, (Q = C3 t) 
  → distance Q C3 = |sqrt 3 * sin(θ + π / 3) + 4 * sqrt 3|
  → distance Q C3 ≥ 3 * sqrt 3 - 1 
  → distance Q C3 ≤ 5 * sqrt 3 - 1 := sorry

-- 4. Prove range of 2x + y for a moving point P on C1
theorem range_2x_plus_y : 
  ∀ (t : ℝ), 
  let P := C1 t 
  → ∀ x y : ℝ, (P = (x, y)) 
  → -5 - sqrt 5 ≤ 2 * x + y 
  ∧ 2 * x + y ≤ -5 + sqrt 5 := sorry

-- 5. Prove range of a given x + y + a ≥ 0 for (x, y) on C1
theorem range_a : 
  ∀ (t : ℝ), 
  let P := C1 t 
  → ∀ a x y : ℝ, (P = (x, y)) 
  → x + y + a ≥ 0 
  → a ≥ 1 + sqrt 2 := sorry

end curve_C1_equation_curve_C2_equation_min_distance_midpoint_C3_min_max_distance_Q_C3_range_2x_plus_y_range_a_l714_714817


namespace max_good_numberings_l714_714508

noncomputable def goodNumberings : finset (fin (8 × 8)) := sorry

theorem max_good_numberings (points : finset (ℝ × ℝ)) (h_distinct : points.card = 8) :
  goodNumberings.card = 56 :=
sorry

end max_good_numberings_l714_714508


namespace newtons_theorem_l714_714862

theorem newtons_theorem
    {A B C D E F L M N : Type}
    (hABCDEF : is_complete_quadrilateral A B C D E F)
    (hL : midpoint L A C)
    (hM : midpoint M B D)
    (hN : midpoint N E F) : collinear [L, M, N] :=
sorry

end newtons_theorem_l714_714862


namespace find_m_l714_714442

-- Define the sets A and B
def setA := {x : ℝ | -4 < x ∧ x < 2}
def setB (m : ℝ) := {x : ℝ | m-1 < x ∧ x < m+1}

-- Define the conditions A ∩ B = B and A ∩ B ≠ ∅
theorem find_m (m : ℝ) :
  (∀ x, x ∈ setB m → x ∈ setA) ∧ (∃ x, x ∈ setA ∧ x ∈ setB m) ↔ m ∈ Icc (-3 : ℝ) (1 : ℝ) := by
  sorry

end find_m_l714_714442


namespace series_solution_l714_714207

theorem series_solution (r : ℝ) (h : (r^3 - r^2 + (1 / 4) * r - 1 = 0) ∧ r > 0) :
  (∑' (n : ℕ), (n + 1) * r^(3 * (n + 1))) = 16 * r :=
by
  sorry

end series_solution_l714_714207


namespace textbook_weight_l714_714189

theorem textbook_weight
  (w : ℝ)
  (bookcase_limit : ℝ := 80)
  (hardcover_books : ℕ := 70)
  (hardcover_weight_per_book : ℝ := 0.5)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (knick_knack_weight : ℝ := 6)
  (over_limit : ℝ := 33)
  (total_items_weight : ℝ := bookcase_limit + over_limit)
  (hardcover_total_weight : ℝ := hardcover_books * hardcover_weight_per_book)
  (knick_knack_total_weight : ℝ := knick_knacks * knick_knack_weight)
  (remaining_weight : ℝ := total_items_weight - (hardcover_total_weight + knick_knack_total_weight)) :
  remaining_weight = textbooks * 2 :=
by
  sorry

end textbook_weight_l714_714189


namespace concyclic_P_a_P_b_P_c_P_d_l714_714526

open EuclideanGeometry

variable (A B C D M_ac H_d H_b P_d P_b P_a P_c : Point)
variable (cyclic : CyclicQuadrilateral A B C D)
variable (midpoint_M_ac : Midpoint M_ac A C)
variable (orthocenter_H_d : Orthocenter H_d (Triangle.mk A B C))
variable (orthocenter_H_b : Orthocenter H_b (Triangle.mk A D C))
variable (projection_P_d : Projection P_d H_d (Line.mk B M_ac))
variable (projection_P_b : Projection P_b H_b (Line.mk D M_ac))
variable (projection_P_a : Projection P_a H_d (Line.mk D M_ac))
variable (projection_P_c : Projection P_c H_b (Line.mk B M_ac))

theorem concyclic_P_a_P_b_P_c_P_d :
  ConcyclicPoints [P_a, P_b, P_c, P_d] :=
sorry

end concyclic_P_a_P_b_P_c_P_d_l714_714526


namespace modular_inverse_31_mod_37_l714_714765

theorem modular_inverse_31_mod_37 : ∃ a : ℤ, 0 ≤ a ∧ a < 37 ∧ (31 * a) % 37 = 1 :=
begin
  use 36,
  norm_num,
  exact dec_trivial,
end

end modular_inverse_31_mod_37_l714_714765


namespace lambda_value_l714_714216

-- Define the function f
def f (x a : ℝ) : ℝ := (x^2 - a) * exp (1 - x)

-- Define the derivative of f
def f' (x a : ℝ) : ℝ := (-x^2 + 2*x + a) * exp (1 - x)

-- Main theorem statement
theorem lambda_value (a λ : ℝ) (h_deriv : f' (λ - 1) a = (λ^2 - 2*λ + a) * exp (λ - 1)) 
  (h_ext : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (x₂ * f x₁ a ≤ λ * (f' x₁ a - a * (exp (1 - x₁) + 1)))) :
  λ = 2*exp(1)/(exp(1) + 1) := sorry

end lambda_value_l714_714216


namespace Eva_is_16_l714_714375

def Clara_age : ℕ := 12
def Nora_age : ℕ := Clara_age + 3
def Liam_age : ℕ := Nora_age - 4
def Eva_age : ℕ := Liam_age + 5

theorem Eva_is_16 : Eva_age = 16 := by
  sorry

end Eva_is_16_l714_714375
