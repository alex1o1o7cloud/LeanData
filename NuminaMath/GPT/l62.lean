import Mathlib

namespace imaginary_part_of_conjugate_l62_62348

variables (a b : ℝ) (z : ℂ)
#check Complex.conj

theorem imaginary_part_of_conjugate:
  (a - 2 * Complex.I = (b - Complex.I) * Complex.I) →
  (z = a + b * Complex.I) →
  Im (Complex.conj z) = 2 :=
begin
  intros h₁ h₂,
  sorry
end

end imaginary_part_of_conjugate_l62_62348


namespace sum_of_coefficients_l62_62567

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), (512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 60) :=
by
  sorry

end sum_of_coefficients_l62_62567


namespace ratio_perimeter_to_inscribed_ratio_perimeter_to_circumscribed_l62_62740

-- Given: A trapezoid ABCD, where diagonal BD forms a 30° angle with the base.
structure Trapezoid :=
  (A B C D : Point)
  (diagonalAngle : ∠BD ∥ base = 30)
  (inscribable : Bool)
  (circumscribable : Bool)

-- Define the perimeter of the trapezoid
def Trapezoid.perimeter (T : Trapezoid) : ℝ := sorry -- Placeholder, to be defined

-- Define the circumference of the inscribed circle
def inscribedCircumference (T : Trapezoid) : ℝ := sorry -- Placeholder, to be defined

-- Define the circumference of the circumscribed circle
def circumscribedCircumference (T : Trapezoid) : ℝ := sorry -- Placeholder, to be defined

-- Proof goal 1: ratio of the perimeter to the inscribed circle's circumference
theorem ratio_perimeter_to_inscribed (T : Trapezoid) (h : T.inscribable) :
  (T.perimeter / inscribedCircumference T) = (4 * Real.sqrt 3 / Real.pi) :=
sorry

-- Proof goal 2: ratio of the perimeter to the circumscribed circle's circumference
theorem ratio_perimeter_to_circumscribed (T : Trapezoid) (h : T.circumscribable) :
  (T.perimeter / circumscribedCircumference T) = (2 / Real.pi) :=
sorry

end ratio_perimeter_to_inscribed_ratio_perimeter_to_circumscribed_l62_62740


namespace ranking_of_girls_l62_62686

variables (score : String → ℕ)
variables (Daisy Eloise Fiona Gabrielle : String)

-- Conditions
axiom gabrielle_condition : score Gabrielle > score Daisy
axiom fiona_condition : score Fiona > score Eloise

-- Theorem to prove
theorem ranking_of_girls :
  ∀ (Daisy Eloise Fiona Gabrielle : String), 
    score Gabrielle > score Daisy ∧ 
    score Fiona > score Eloise → 
      [Gabrielle, Fiona, Daisy, Eloise] = 
        List.reverse (List.sort (λ a b => score a > score b) [Daisy, Eloise, Fiona, Gabrielle]) :=
by 
  intros Daisy Eloise Fiona Gabrielle
  intros h
  sorry

end ranking_of_girls_l62_62686


namespace correct_option_is_C_l62_62126

-- Define the conditions and the statement
def probability := "The probability of rain tomorrow in this area is 80%".
def option_A := "About 80% of the time tomorrow in this area will have rain, and 20% of the time it will not rain."
def option_B := "About 80% of the places in this area will have rain tomorrow, and 20% of the places will not."
def option_C := "The possibility of rain tomorrow in this area is 80%."
def option_D := "About 80% of the people in this area think it will rain tomorrow, and 20% of the people think it will not rain."

-- The theorem to prove
theorem correct_option_is_C : 
  probability = option_C :=
sorry

end correct_option_is_C_l62_62126


namespace compare_neg_fractions_l62_62285

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l62_62285


namespace find_angle_C_find_a_plus_b_l62_62491

variables (A B C a b c : ℝ)
variables (area : ℝ)
variables (h : 2 * cos C * (a * cos B + b * cos A) = c)
variables (h1 : c = sqrt 7)
variables (h2 : area = 3 * sqrt 3 / 2)

-- Prove: Given \(2 \cos C \cdot (a \cos B + b \cos A) = c\), show that \(C = \frac{\pi}{3}\)
theorem find_angle_C (h : 2 * cos C * (a * cos B + b * cos A) = c) : C = π / 3 :=
  sorry

-- Prove: Given c = sqrt 7 and the area of triangle ABC is 3 sqrt 3 / 2, show that a + b = 5
theorem find_a_plus_b (h1 : c = sqrt 7) (h2 : area = 3 * sqrt 3 / 2) 
  (hC : C = π / 3) : a + b = 5 :=
  sorry

end find_angle_C_find_a_plus_b_l62_62491


namespace number_of_antlers_l62_62923

-- Variables and Conditions
variable (total_deer : ℕ) (percentage_with_antlers : ℝ) (percentage_albino : ℝ) (albino_deer_with_antlers : ℕ)

-- Definitions and given values
def deer_with_antlers := total_deer * percentage_with_antlers
def albino_deer := deer_with_antlers * percentage_albino

-- Theorem to prove
theorem number_of_antlers (h1 : total_deer = 920) 
                          (h2 : percentage_with_antlers = 0.10) 
                          (h3 : percentage_albino = 1/4) 
                          (h4 : albino_deer_with_antlers = 23) :
                          deer_with_antlers / 4 = 23 := 
by 
  sorry

end number_of_antlers_l62_62923


namespace find_polynomial_P_l62_62835

theorem find_polynomial_P
  (a b c P : ℝ → ℝ)
  (h_roots : ∀ x, x^3 + 5 * x^2 + 8 * x + 13 = 0 → x = a ∨ x = b ∨ x = c)
  (h1 : P a = b + c + 2)
  (h2 : P b = a + c + 2)
  (h3 : P c = a + b + 2)
  (h4 : P (a + b + c) = -22) :
  P = λ x, (19 * x^3 + 95 * x^2 + 152 * x + 247) / 52 - x - 3 :=
sorry

end find_polynomial_P_l62_62835


namespace regression_equation_correct_prediction_correct_for_14_l62_62227

section regression_proof

def x_i := [4, 5, 6, 7, 8] : List ℝ
def y_i := [2, 2.1, 2.5, 2.9, 3.2] : List ℝ

def sum_xi_squared : ℝ := 190
def sum_xi_yi : ℝ := 79.4

def x_bar : ℝ := (List.sum x_i) / (List.length x_i)
def y_bar : ℝ := (List.sum y_i) / (List.length y_i)

def b_hat : ℝ := (sum_xi_yi - (List.length x_i) * x_bar * y_bar) / (sum_xi_squared - (List.length x_i) * x_bar^2)
def a_hat : ℝ := y_bar - b_hat * x_bar
def regression_eq (x : ℝ) : ℝ := b_hat * x + a_hat

def y_pred (x : ℝ) : ℝ := regression_eq x

theorem regression_equation_correct :
  b_hat = 0.32 ∧ a_hat = 0.62 :=
  by
    -- derivations in solution are recreated to be proven here
    sorry -- steps omitted

theorem prediction_correct_for_14 :
  y_pred 14 = 5.1 :=
  by
    -- solution uses prediction based on regression_eq function
    sorry -- derivations omitted

end regression_proof

end regression_equation_correct_prediction_correct_for_14_l62_62227


namespace rhombus_area_l62_62890

/-- Define the polynomial we'll work with. -/
def quartic_polynomial (z : ℂ) : ℂ := 2 * z^4 + 8 * complex.I * z^3 + (-9 + 9 * complex.I) * z^2 + (-18 - 2 * complex.I) * z + (3 - 12 * complex.I)

/-- Define the roots of the polynomial. -/
noncomputable def roots := {z : ℂ | quartic_polynomial z = 0}

/-- Define the average of the roots, which should be the center of the rhombus. -/
def average_root : ℂ := -complex.I

/-- A lemma stating that the average of the roots is -i, using Vieta's formula. -/
noncomputable example : (∑ x in roots, x) / 4 = -complex.I :=
by sorry

/-- A lemma stating that the absolute value product condition using roots and polynomial evaluation. -/
noncomputable example : 4 * ∥a + complex.I∥^2 * ∥b + complex.I∥^2 = 10 :=
by sorry

/-- Finally, state the main goal as a theorem in Lean 4 -/
theorem rhombus_area : 
  let p := ∥a + complex.I∥ in
  let q := ∥b + complex.I∥ in
  2 * p * q = real.sqrt 10 :=
by sorry

end rhombus_area_l62_62890


namespace problem_l62_62061

theorem problem (p q r : ℂ)
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2)
  (h3 : p * q * r = 2)
  (hp : p ^ 3 = 2 * p + 2)
  (hq : q ^ 3 = 2 * q + 2)
  (hr : r ^ 3 = 2 * r + 2) :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = -18 := by
  sorry

end problem_l62_62061


namespace transformed_curve_l62_62305

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]

theorem transformed_curve :
  (∀ x y : ℝ, 5 * x ^ 2 + 8 * x * y + 4 * y ^ 2 = 1 →
   let (x' : ℝ) := x + 2 * y in
   let (y' : ℝ) := 3 * x + 2 * y in
   x' ^ 2 + y' ^ 2 = 2) :=
by
  intros x y h
  sorry

end transformed_curve_l62_62305


namespace number_of_outfits_l62_62098

theorem number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) (formal_shoes cas_shoes : ℕ) :
  shirts = 7 → ties = 5 → pants = 4 → formal_shoes = 1 → cas_shoes = 1 →
  (shirts * pants * ties * formal_shoes + shirts * pants * (ties - 5 + 1) * cas_shoes) = 168 :=
by
  intros h_shirts h_ties h_pants h_formal_shoes h_cas_shoes
  have h1 := congr_arg (λ x, x * (4:ℕ) * (5:ℕ) * 1) h_shirts
  have h2 := congr_arg (λ x, x * (4:ℕ) * 1 * 1) h_shirts
  simp_all [Nat.sub_self]
  rw [Nat.one_mul] at h2
  simp [h1, h2, h_ties, h_pants, h_formal_shoes, h_cas_shoes]
  norm_num
  sorry

end number_of_outfits_l62_62098


namespace caravan_humps_l62_62971

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62971


namespace vector_addition_l62_62680

def v1 : ℝ × ℝ := (3, -6)
def v2 : ℝ × ℝ := (2, -9)
def v3 : ℝ × ℝ := (-1, 3)
def c1 : ℝ := 4
def c2 : ℝ := 5
def result : ℝ × ℝ := (23, -72)

theorem vector_addition :
  c1 • v1 + c2 • v2 - v3 = result :=
by
  sorry

end vector_addition_l62_62680


namespace angle_equality_l62_62563

-- Define trapezoid ABCD with respective bases AD and BC.
-- Define the points O, B', and C'.
variable {A B C D O B' C' : Type}

-- Assume A, B, C, D, and O are points in a trapezoid with AD and BC as bases.
axiom trapezoid : A ≠ B → B ≠ C → C ≠ D → D ≠ A ∧ collinear A D ∧ collinear B C

-- Assume the diagonals AC and BD intersect at point O.
axiom intersect_at_O : collinear A C → collinear B D → ∃ O, line_segment A C = line_segment B D

-- Define symmetry of points B' and C' relative to angle bisector of ∠BOC.
axiom symmetric_points : symmetric_with_respect_to (angle_bisector (angle_ B O C)) B B' → symmetric_with_respect_to (angle_bisector (angle_ B O C)) C C'

-- Define the relationship that needs to be proved
theorem angle_equality (h1 : trapezoid A B C D)
  (h2 : intersect_at_O A C B D)
  (h3 : symmetric_points B B' C C') :
  angle_ C' A C = angle_ B' D B := sorry

end angle_equality_l62_62563


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62164

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62164


namespace caravan_humps_l62_62968

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62968


namespace medal_awarding_ways_l62_62588

-- Defining the conditions of the problem
def sprinters : ℕ := 10
def americans : ℕ := 4
def canadians : ℕ := 3
def medals : ℕ := 3
def max_americans_with_medals : ℕ := 2

-- Define the problem statement in Lean
theorem medal_awarding_ways :
  ∃ (ways : ℕ), ways = 552 ∧
  (∀ (s : ℕ) (a : ℕ) (c : ℕ) (m : ℕ),
    s = sprinters → a = americans → c = canadians → m = medals →
    (s = 10 → a = 4 → c = 3 → m = 3 →
      ways = (6 * 5 * 4) + (4 * 3 * 6 * 5) + (nat.choose 4 2 * 2 * 6))) :=
by
sorry

end medal_awarding_ways_l62_62588


namespace solve_logarithmic_equation_l62_62092

theorem solve_logarithmic_equation (x : ℝ) (log : ℝ → ℝ → ℝ) 
    (h : log 2 x * log 2 (2 * x) = log 2 (4 * x)) : 
    x = 2 ^ Real.sqrt 2 ∨ x = 2 ^ -Real.sqrt 2 := 
by
  sorry

end solve_logarithmic_equation_l62_62092


namespace monthly_interest_payment_is_correct_l62_62669

-- Define the conditions
def principal : ℝ := 32000
def annual_rate : ℝ := 0.09

-- Define the annual interest calculation
def annual_interest (P : ℝ) (R : ℝ) : ℝ := P * R

-- Define the monthly interest calculation
def monthly_interest_payment (annual_interest : ℝ) : ℝ := annual_interest / 12

-- Prove that the monthly interest payment is $240
theorem monthly_interest_payment_is_correct : 
  monthly_interest_payment (annual_interest principal annual_rate) = 240 :=
by sorry

end monthly_interest_payment_is_correct_l62_62669


namespace board_stabilization_l62_62531

-- Definition of the problem
def transformation_invariant (n : ℕ) (a : Fin n → ℕ) : Prop :=
  (∀ i : Fin n, a i > 1) ∧
  (∀ a b : ℕ, a ≠ b →
    ∃ q : ℕ, ∃ (a' b' : Fin n → ℕ),
    (∃ (q_p_factors : ℕ), q_p_factors = (∏ p in (Nat.primeFactors (a * b)), p)) ∧
    a' = a.update (λ i, if i = a then q else a i) ∧
    b' = b.update (λ i, if i = b then q_squared else b i) ∧
    q_squared = q^2 ∧
    (end_condition : ∀ c d : Fin n → ℕ, (transformation_invariant c) → c = d))

-- Statement of the theorem for stabilization
theorem board_stabilization (n : ℕ) (a : Fin n → ℕ) :
  transformation_invariant n a → ∃ (s : Fin n → ℕ), ∀ i : Fin n, ∀ (a' : Fin n → ℕ), transformation_invariant n a' → a' = s :=
begin
  sorry
end

end board_stabilization_l62_62531


namespace find_theta_l62_62453

def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry
noncomputable def i : ℂ := complex.I
noncomputable def z (θ : ℝ) : ℂ := cos θ + i * sin θ

theorem find_theta (θ : ℝ) (k : ℤ) : z θ ^ 2 = -1 ↔ θ = k * real.pi + real.pi / 2 := by
  sorry

end find_theta_l62_62453


namespace even_function_a_zero_l62_62433

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62433


namespace binomial_expansion_a0_a1_a3_a5_l62_62412

theorem binomial_expansion_a0_a1_a3_a5 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h : (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_0 + a_1 + a_3 + a_5 = 123 :=
sorry

end binomial_expansion_a0_a1_a3_a5_l62_62412


namespace motorcycle_wheels_l62_62783

/--
In a parking lot, there are cars and motorcycles. Each car has 5 wheels (including one spare) 
and each motorcycle has a certain number of wheels. There are 19 cars in the parking lot.
Altogether all vehicles have 117 wheels. There are 11 motorcycles at the parking lot.
--/
theorem motorcycle_wheels (num_cars num_motorcycles total_wheels wheels_per_car wheels_per_motorcycle : ℕ)
  (h1 : wheels_per_car = 5) 
  (h2 : num_cars = 19) 
  (h3 : total_wheels = 117) 
  (h4 : num_motorcycles = 11) 
  : wheels_per_motorcycle = 2 :=
by
  sorry

end motorcycle_wheels_l62_62783


namespace find_f_x_l62_62749

-- Given conditions and required structure.
variable {F : ℝ → ℝ}

-- Goal: f(x) = x^2 - 1 given that f(x-1) = x^2 - 2x.
theorem find_f_x (h : ∀ x : ℝ, F (x - 1) = x^2 - 2x) : ∀ x : ℝ, F x = x^2 - 1 := by
  sorry

end find_f_x_l62_62749


namespace prove_a_zero_l62_62421

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62421


namespace solve_rational_r_l62_62090

theorem solve_rational_r :
  (let r := (√(8^2 + 15^2)) / (√(25 + 16))
  in r = 17 / 5) :=
by
  let r := (sqrt (8^2 + 15^2)) / (sqrt (25 + 16))
  have h_num : sqrt (8^2 + 15^2) = 17 := by sorry
  have h_den : sqrt (25 + 16) = sqrt 41 := by sorry
  have h_r : r = 17 / sqrt 41 := by sorry
  have r_simp_fl : 17 / sqrt 41 = 17 / 5 := by sorry
  exact h_r.trans r_simp_fl

end solve_rational_r_l62_62090


namespace prime_factor_of_sum_of_four_consecutive_integers_l62_62172

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l62_62172


namespace num_small_triangles_l62_62530

-- Define the lengths of the legs of the large and small triangles
variables (a h b k : ℕ)

-- Define the areas of the large and small triangles
def area_large_triangle (a h : ℕ) : ℕ := (a * h) / 2
def area_small_triangle (b k : ℕ) : ℕ := (b * k) / 2

-- Define the main theorem
theorem num_small_triangles (ha : a = 6) (hh : h = 4) (hb : b = 2) (hk : k = 1) :
  (area_large_triangle a h) / (area_small_triangle b k) = 12 :=
by
  sorry

end num_small_triangles_l62_62530


namespace even_function_a_zero_l62_62429

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62429


namespace bf_eq_cg_l62_62825

theorem bf_eq_cg (A B C D E P G F : Point) (h_triangle : Triangle A B C)
  (h_angle_bisector : AngleBisector A D B C)
  (h_D_on_BC : OnSegment D B C)
  (h_E_on_BC : OnSegment E B C)
  (h_BD_eq_EC : BD = EC)
  (h_line_through_E_parallel_AD : Parallel (LineThrough E P) (LineThrough A D))
  (h_P_inside_triangle : InsideTriangle P A B C)
  (h_G_intersection_BP_AC : IntersectionPoint (LineThrough B P) (LineThrough A C) G)
  (h_F_intersection_CP_AB : IntersectionPoint (LineThrough C P) (LineThrough A B) F) :
  Distance (LineThrough B F) = Distance (LineThrough C G) := 
sorry

end bf_eq_cg_l62_62825


namespace C_integer_l62_62839

noncomputable def one_k (k : ℕ) : ℕ := (∑ i in finset.range k, 10 ^ i)

def factorial_prod : ℕ → ℕ
| 0     := 1
| (m+1) := one_k (m+1) * factorial_prod m

def C (m n : ℕ) : ℕ := factorial_prod (m + n) / (factorial_prod m * factorial_prod n)

theorem C_integer (m n : ℕ) : m ≥ 0 → n ≥ 0 → (C m n : ℤ) = C m n :=
by sorry

end C_integer_l62_62839


namespace find_polynomial_Q_l62_62841

noncomputable def Q (x : ℝ) : ℝ := Q 0 + Q 1 * x + Q 2 * x^2

theorem find_polynomial_Q :
  (∀ Q : ℝ → ℝ, (Q = λ x, Q 0 + Q 1 * x + Q 2 * x^2) → Q (-1) = 3 → Q = λ x, 3 * (1 + x + x^2)) :=
begin
  sorry,
end

end find_polynomial_Q_l62_62841


namespace calculate_expression_correct_l62_62274

noncomputable def calculate_expression : ℝ :=
  let sin30 := Real.sin (Real.pi / 6)
  let pow_zero := (3.14 - Real.pi) ^ 0
  let neg_half_inv_sq := (- (1 / 2)) ^ (-2)
  sin30 - pow_zero + neg_half_inv_sq

theorem calculate_expression_correct :
  calculate_expression = 7 / 2 :=
by
  sorry

end calculate_expression_correct_l62_62274


namespace critical_points_range_f_inequality_l62_62392

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

def f' (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f' a x + 1 / (x + 1)

def g' (a : ℝ) (x : ℝ) : ℝ := 
  (a * x^2 + (2 * a - 1) * x + a) / (x * (x + 1)^2)

theorem critical_points_range (a : ℝ) (h : a ≠ 0) (h1 : ∃ x : ℝ, g' a x = 0) :
  0 < a ∧ a < 1 / 4 := 
sorry

theorem f_inequality (x : ℝ) (h : 0 < x) : 
  f 1 x < Real.exp x + Real.sin x - 1 := 
sorry

end critical_points_range_f_inequality_l62_62392


namespace retailer_received_extra_boxes_l62_62529
-- Necessary import for mathematical proofs

-- Define the conditions
def dozen_boxes := 12
def dozens_ordered := 3
def discount_percent := 25

-- Calculate the total boxes ordered and the discount factor
def total_boxes := dozen_boxes * dozens_ordered
def discount_factor := (100 - discount_percent) / 100

-- Define the number of boxes paid for and the extra boxes received
def paid_boxes := total_boxes * discount_factor
def extra_boxes := total_boxes - paid_boxes

-- Statement of the proof problem
theorem retailer_received_extra_boxes : extra_boxes = 9 :=
by
    -- This is the place where the proof would be written
    sorry

end retailer_received_extra_boxes_l62_62529


namespace Douglas_vote_percentage_in_Y_l62_62024

noncomputable def votes_in_Y_percentage (V : ℝ) (dx votes_perc_Y : ℝ) : Prop :=
  dx = 0.72 * 2 * V ∧
  votes_perc_Y = 36 * V / 100

theorem Douglas_vote_percentage_in_Y (V : ℝ) (total_perc_XY dx votes_total votes_perc_Y : ℝ) 
    (h1 : total_perc_XY = 0.60 * 3 * V)
    (h2 : dx = 0.72 * 2 * V) 
    (h3 : votes_total = total_perc_XY - dx) :
  votes_perc_Y = 36 :=
by 
  have h_votes_Y : votes_total / V * 100 = 36 := sorry
  exact h_votes_Y

end Douglas_vote_percentage_in_Y_l62_62024


namespace solve_linear_system_l62_62548

theorem solve_linear_system:
  ∃ (x y z: ℝ), 
    x^2 - 22*y - 69*z + 703 = 0 ∧
    y^2 + 23*x + 23*z - 1473 = 0 ∧
    z^2 - 63*x + 66*y + 2183 = 0 ∧
    x = 20 ∧ y = -22 ∧ z = 23 :=
by
  use 20, -22, 23
  split
  {
    calc (20: ℝ)^2 - 22 * (-22) - 69 * 23 + 703 = 
      400 + 484 - 1587 + 703 : by norm_num
    ... = 0 : by norm_num
  }
  split
  {
    calc (-22: ℝ)^2 + 23 * 20 + 23 * 23 - 1473 =
      484 + 460 + 529 - 1473 : by norm_num
    ... = 0 : by norm_num
  }
  {
    calc (23: ℝ)^2 - 63 * 20 + 66 * (-22) + 2183 =
      529 - 1260 - 1452 + 2183 : by norm_num
    ... = 0 : by norm_num
  }
  sorry

end solve_linear_system_l62_62548


namespace total_loss_incurred_l62_62259

variable (P : ℝ) -- Capital of Pyarelal
variable (A : ℝ := 1 / 9 * P) -- Capital of Ashok
variable (pyarelal_loss : ℝ := 810) -- Loss of Pyarelal
variable (total_loss : ℝ)

theorem total_loss_incurred : total_loss = 900 :=
by
  let total_capital := A + P
  let ashok_loss := (A / total_capital) * total_loss
  let pyarelal_loss := (P / total_capital) * total_loss
  have h1 : A = 1 / 9 * P := rfl
  have h2 : pyarelal_loss = 810 := rfl
  have h3 : pyarelal_loss = (P / total_capital) * total_loss := 
    by sorry
  have h4 : 810 = (P / (A + P)) * total_loss := 
    by sorry
  have h5 : 810 = (P / ((1 / 9) * P + P)) * total_loss := 
    by sorry
  have h6 : 810 = (P / (10 / 9 * P)) * total_loss := 
    by sorry
  have h7 : 810 = (9 / 10) * total_loss :=
    by sorry
  have h8 : total_loss = 900 :=
    by sorry
  exact h8

end total_loss_incurred_l62_62259


namespace bisect_incenter_segment_l62_62077

def point : Type := ℝ × ℝ

namespace Geometry

structure Circle :=
  (center : point)
  (radius : ℝ)

structure Triangle :=
  (A B C : point)

def incenter (T : Triangle) : point := sorry

def is_midpoint (p m q : point) :=
  p.1 + q.1 = 2 * m.1 ∧ p.2 + q.2 = 2 * m.2

structure Quadrilateral :=
  (A B C D : point)
  (circumscribed : Circle)

noncomputable def midpoint_of_arc (A B : point) (C : Circle) : point := sorry

theorem bisect_incenter_segment 
  {A B C D : point}
  (Q : Quadrilateral)
  (C_centered : Q.circumscribed.center.
    in Circle)
  (M : point) (N : point)
  (H_M : M = midpoint_of_arc Q.A Q.B Q.circumscribed)
  (H_N : N = midpoint_of_arc Q.C Q.D Q.circumscribed)
  (I_A : point) (H_I_A : I_A = incenter ⟨Q.A, Q.B, Q.D⟩)
  (I_B : point) (H_I_B : I_B = incenter ⟨Q.B, Q.C, Q.D⟩) :
  is_midpoint I_A ((M.1 + N.1) / 2, (M.2 + N.2) / 2) I_B :=
sorry

end Geometry

end bisect_incenter_segment_l62_62077


namespace determine_c_square_of_binomial_l62_62300

theorem determine_c_square_of_binomial (c : ℝ) : (∀ x : ℝ, 16 * x^2 + 40 * x + c = (4 * x + 5)^2) → c = 25 :=
by
  intro h
  have key := h 0
  -- By substitution, we skip the expansion steps and immediately conclude the value of c
  sorry

end determine_c_square_of_binomial_l62_62300


namespace pages_per_sheet_is_one_l62_62822

-- Definition of conditions
def stories_per_week : Nat := 3
def pages_per_story : Nat := 50
def num_weeks : Nat := 12
def reams_bought : Nat := 3
def sheets_per_ream : Nat := 500

-- Calculate total pages written over num_weeks (short stories only)
def total_pages : Nat := stories_per_week * pages_per_story * num_weeks

-- Calculate total sheets available
def total_sheets : Nat := reams_bought * sheets_per_ream

-- Calculate pages per sheet, rounding to nearest whole number
def pages_per_sheet : Nat := (total_pages / total_sheets)

-- The main statement to prove
theorem pages_per_sheet_is_one : pages_per_sheet = 1 :=
by
  sorry

end pages_per_sheet_is_one_l62_62822


namespace AM_eq_AN_l62_62062

noncomputable def circle : Type := sorry -- placeholder for the type of circle

variables {A B C E F M N : circle} (Γ : circle)
variables (is_on_circle_A : A ∈ Γ) (is_on_circle_B : B ∈ Γ) (is_on_circle_C : C ∈ Γ)
variables (E_is_midpoint_arc_AB : ∃ arc_AB, arc_AB = circle_arc Γ A B ∧ E = midpoint arc_AB)
variables (F_is_midpoint_arc_AC : ∃ arc_AC, arc_AC = circle_arc Γ A C ∧ F = midpoint arc_AC)
variables (M_is_intersection_EF_AB : ∃ EF AB, EF = line_segment E F ∧ AB = line_segment A B ∧ M = intersection EF AB)
variables (N_is_intersection_EF_AC : ∃ EF AC, EF = line_segment E F ∧ AC = line_segment A C ∧ N = intersection EF AC)

theorem AM_eq_AN : distance A M = distance A N :=
sorry

end AM_eq_AN_l62_62062


namespace possible_values_of_N_count_l62_62947

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62947


namespace ordered_triple_eq_l62_62683

noncomputable def f (x : ℝ) := x^3 + 2*x^2 + 3*x + 4

noncomputable def g (x : ℝ) := x^3 - 2*x^2 + x - 12

example (r : ℝ) (hr : f r = 0) : g (r^2) = 0 :=
by
  sorry

theorem ordered_triple_eq : (b, c, d) = (-2, 1, -12) :=
by
  sorry

end ordered_triple_eq_l62_62683


namespace sum_of_consecutive_odds_l62_62698

theorem sum_of_consecutive_odds (a : ℤ) : ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x + 2 = y ∧ 4 * a = x + y :=
by
  existsi (2 * a - 1)
  existsi (2 * a + 1)
  split
  {
    exact Mod_def
  }
  split
  {
    exact Mod_def
  }
  split
  {
    exact rfl
  }
  sorry

end sum_of_consecutive_odds_l62_62698


namespace num_factors_M_l62_62506

def M := 57^5 + 5 * 57^4 + 10 * 57^3 + 10 * 57^2 + 5 * 57 + 1

theorem num_factors_M : Nat.num_divisors M = 36 :=
by
  sorry

end num_factors_M_l62_62506


namespace circumscribed_sphere_surface_area_l62_62773

theorem circumscribed_sphere_surface_area
  (a b c : ℝ)
  (ha : a = 1)
  (hb : b = sqrt 2)
  (hc : c = sqrt 3)
  (h_perpendicular : true)  -- This condition implies the mutual perpendicularity
  : 4 * Real.pi * (sqrt (a ^ 2 + b ^ 2 + c ^ 2) / 2) ^ 2 = 6 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l62_62773


namespace possible_values_of_N_count_l62_62944

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62944


namespace find_X_l62_62480

-- Definition of the conditions
def top_side_lengths : List ℕ := [2, 3, 4]
def bottom_side_lengths : List ℕ := [1, 2, 4, 6]
-- X is the unknown length we aim to find

theorem find_X (X : ℕ) :
  (2 + 3 + 4 + X = 1 + 2 + 4 + 6) ↔ (X = 4) :=
by
  -- stating the conditions
  have h1 : (2 + 3 + 4 = 9) := by norm_num
  have h2 : (1 + 2 + 4 + 6 = 13) := by norm_num
  split
  -- proving the forward direction
  case mp =>
    intro h
    rw [h1, h2] at h
    linarith
  -- proving the backward direction
  case mpr =>
    intro h
    rw [h1, h2]
    rw [h]
    linarith

-- conclusion
#check find_X

end find_X_l62_62480


namespace solve_math_problem_l62_62618

-- Definitions for the problem statements
def converse_proof : Prop :=
  ∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0)

def sufficient_proof : Prop :=
  ∀ x : ℝ, (x > 2 → x^2 - 3 * x + 2 > 0)

def negation_proof : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) = ∀ x : ℝ, x^2 + x + 1 ≥ 0

def conjunction_proof {p q : Prop} : Prop :=
  ¬ (p ∧ q) → ¬ p ∨ ¬ q

-- Main equivalent proof problem:
def math_problem : Prop :=
  (converse_proof ∧ sufficient_proof ∧ negation_proof ∧ conjunction_proof) → 3 = 3

theorem solve_math_problem : math_problem :=
by sorry

end solve_math_problem_l62_62618


namespace functions_symmetric_wrt_line_l62_62891

theorem functions_symmetric_wrt_line (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x) (h₂ : ∀ x, g x = log 2 x) : 
  ∀ x, g (f x) = x :=
by
  intro x
  sorry

end functions_symmetric_wrt_line_l62_62891


namespace remainder_5_pow_207_mod_7_l62_62611

theorem remainder_5_pow_207_mod_7 :
  (∃ (n : ℕ), n < 7 ∧ 5^207 % 7 = n) :=
begin
  use 6,
  have h0 : 5^1 % 7 = 5, by norm_num,
  have h1 : 5^2 % 7 = 4, by norm_num,
  have h2 : 5^3 % 7 = 6, by norm_num,
  have h3 : 5^4 % 7 = 2, by norm_num,
  have h4 : 5^5 % 7 = 3, by norm_num,
  have h5 : 5^6 % 7 = 1, by norm_num,
  have h6 : 5^(6 * 34 + 3) % 7 = (5^6)^34 * 5^3 % 7, 
  { rw [← pow_add, Nat.add_comm], },
  rw [h5, one_pow, one_mul] at h6,
  exact h2,
end

end remainder_5_pow_207_mod_7_l62_62611


namespace tan_alpha_eq_one_l62_62374

open Real

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h_cos_sin_eq : cos (α + β) = sin (α - β)) : tan α = 1 :=
by
  sorry

end tan_alpha_eq_one_l62_62374


namespace domain_of_g_l62_62383

noncomputable def f : ℝ → ℝ := sorry

def is_in_domain_f (x : ℝ) : Prop := -7 ≤ x ∧ x ≤ 1

def g (x : ℝ) : ℝ := (f (2 * x + 1)) / (x + 2)

def is_in_domain_g (x : ℝ) : Prop :=
  (-4 ≤ x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 0)

theorem domain_of_g :
  (∀ x, is_in_domain_g x ↔ (is_in_domain_f (2 * x + 1) ∧ (x ≠ -2))) :=
begin
  sorry
end

end domain_of_g_l62_62383


namespace smallest_c_value_l62_62897

theorem smallest_c_value :
  ∃ a b c : ℕ, a * b * c = 3990 ∧ a + b + c = 56 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
by {
  -- Skipping proof as instructed
  sorry
}

end smallest_c_value_l62_62897


namespace probability_closer_to_6_than_0_l62_62245

theorem probability_closer_to_6_than_0:
  ∃ p : ℝ, p = 0.6 ∧ (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 7 → ((x > 3) → (x is closer to 6 than 0))) := 
sorry

end probability_closer_to_6_than_0_l62_62245


namespace num_distinct_terms_expansion_a_b_c_10_l62_62488

-- Define the expansion of (a+b+c)^10
def num_distinct_terms_expansion (n : ℕ) : ℕ :=
  Nat.choose (n + 3 - 1) (3 - 1)

-- Theorem statement
theorem num_distinct_terms_expansion_a_b_c_10 : num_distinct_terms_expansion 10 = 66 :=
by
  sorry

end num_distinct_terms_expansion_a_b_c_10_l62_62488


namespace magnitude_z1_condition_z2_range_condition_l62_62360

-- Define and set up the conditions and problem statements
open Complex

def complex_number_condition (z₁ : ℂ) (m : ℝ) : Prop :=
  z₁ = 1 + m * I ∧ ((z₁ * (1 - I)).re = 0)

def z₂_condition (z₂ z₁ : ℂ) (n : ℝ) : Prop :=
  z₂ = z₁ * (n - I) ∧ z₂.re < 0 ∧ z₂.im < 0

-- Prove that if z₁ = 1 + m * I and z₁ * (1 - I) is pure imaginary, then |z₁| = sqrt 2
theorem magnitude_z1_condition (m : ℝ) (z₁ : ℂ) 
  (h₁ : complex_number_condition z₁ m) : abs z₁ = Real.sqrt 2 :=
by sorry

-- Prove that if z₂ = z₁ * (n + i^3) is in the third quadrant, then n is in the range (-1, 1)
theorem z2_range_condition (n : ℝ) (m : ℝ) (z₁ z₂ : ℂ)
  (h₁ : complex_number_condition z₁ m)
  (h₂ : z₂_condition z₂ z₁ n) : -1 < n ∧ n < 1 :=
by sorry

end magnitude_z1_condition_z2_range_condition_l62_62360


namespace smallest_b_value_l62_62058

variable {a b c d : ℝ}

-- Definitions based on conditions
def is_arithmetic_series (a b c : ℝ) (d : ℝ) : Prop :=
  a = b - d ∧ c = b + d

def abc_product (a b c : ℝ) : Prop :=
  a * b * c = 216

theorem smallest_b_value (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (arith_series : is_arithmetic_series a b c d)
  (abc_216 : abc_product a b c) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_value_l62_62058


namespace possible_N_values_l62_62952

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62952


namespace largest_multiple_of_7_less_than_100_l62_62604

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l62_62604


namespace surface_area_circumscribed_sphere_prism_l62_62258

noncomputable def centroid_bc_d (A B C D G : Type) [affine_space A] [vector_space B] [B →ₗ A]
  (BD BC CD : B) : Prop := 
  G = (BD + BC + CD) / 3

noncomputable def midpoint_a_g (AG AM : Type) [affine_space AG] [vector_space AM] [AM →ₗ AG]
  (A G M : AM) : Prop := 
  M = (A + G) / 2

theorem surface_area_circumscribed_sphere_prism (A B C D G M : Type)
  [affine_space A] [vector_space B] [vector_space G] [vector_space M]
  [B →ₗ A] [G →ₗ A] [M →ₗ A]
  (tetrahedron_regular : ∀ (X Y : A), (X ≠ Y) → (A.dist X Y = 1))
  (centroid_condition : centroid_bc_d A B C D)
  (midpoint_condition : midpoint_a_g A G M) :
  let R := (sqrt 6) / 4 in
  4 * π * R^2 = 3/2 * π :=
sorry

end surface_area_circumscribed_sphere_prism_l62_62258


namespace median_number_of_moons_l62_62996

theorem median_number_of_moons :
  let moons := [0, 0, 1, 2, 3, 3, 10, 16, 17, 22]
  List.median moons = 3 := by
  sorry

end median_number_of_moons_l62_62996


namespace sum_of_four_consecutive_integers_divisible_by_two_l62_62155

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l62_62155


namespace gcd_72_108_l62_62189

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l62_62189


namespace find_three_digit_numbers_l62_62315

def is_three_digit_number (A B C : ℕ) : Prop :=
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧ (0 ≤ C) ∧ (C ≤ 9) ∧ (100 * A + 10 * B + C ≥ 100)

def satisfies_condition (A B C : ℕ) : Prop :=
  100 * A + 10 * B + C = 2 * (10 * A + B + 10 * B + C + 10 * A + C)

theorem find_three_digit_numbers :
  ∀ A B C : ℕ, is_three_digit_number A B C ∧ satisfies_condition A B C → 
  (100 * A + 10 * B + C = 134 ∨
   100 * A + 10 * B + C = 144 ∨
   100 * A + 10 * B + C = 150 ∨
   100 * A + 10 * B + C = 288 ∨
   100 * A + 10 * B + C = 294) := 
begin
  assume A B C,
  sorry
end

end find_three_digit_numbers_l62_62315


namespace range_of_a_l62_62129

-- Defining the core problem conditions in Lean
def prop_p (a : ℝ) : Prop := ∃ x₀ : ℝ, a * x₀^2 + 2 * a * x₀ + 1 < 0

-- The original proposition p is false, thus we need to show the range of a is 0 ≤ a ≤ 1
theorem range_of_a (a : ℝ) : ¬ prop_p a → 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l62_62129


namespace quadratic_inequality_solution_l62_62744

theorem quadratic_inequality_solution (a b c : ℝ) (h₁ : a < 0) (h₂ : (set_of (λ x : ℝ, ax^2 + bx + c < 0)) = set.union (set.Ioo (-∞) (-1)) (set.Ioo (1/2) +∞)) :
  set_of (λ x : ℝ, cx^2 - bx + a < 0) = set.Ioo (-2) 1 :=
by {
  sorry
}

end quadratic_inequality_solution_l62_62744


namespace fish_stock_l62_62097

theorem fish_stock {
  initial_stock fish_sold new_stock : ℕ,
  spoil_fraction : ℚ,
  initial_stock = 200 → fish_sold = 50 → new_stock = 200 →
  spoil_fraction = 1 / 3 →
  ∃ (final_stock : ℕ), final_stock = initial_stock - fish_sold - ⌊(initial_stock - fish_sold) * spoil_fraction⌋ + new_stock ∧ final_stock = 300 :=
begin
  intros h_initial h_sold h_new h_spoil,
  use initial_stock - fish_sold - (⌊(initial_stock - fish_sold) * spoil_fraction⌋ : ℕ) + new_stock,
  split,
  {
    rw [h_initial, h_sold, h_new, h_spoil],
    norm_num,
  },
  exact ⟨300⟩,
end

end fish_stock_l62_62097


namespace tan_add_pi_over_4_l62_62347

theorem tan_add_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_add_pi_over_4_l62_62347


namespace minimum_value_of_tan_sum_l62_62786

open Real

theorem minimum_value_of_tan_sum :
  ∀ {A B C : ℝ}, 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ 
  2 * sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 ->
  ( ∃ t : ℝ, ( t = 1 / tan A + 1 / tan B + 1 / tan C ) ∧ t = sqrt 13 / 2 ) := 
sorry

end minimum_value_of_tan_sum_l62_62786


namespace charlie_paints_140_square_feet_l62_62255

-- Define the conditions
def total_area : ℕ := 320
def ratio_allen : ℕ := 4
def ratio_ben : ℕ := 5
def ratio_charlie : ℕ := 7
def total_parts : ℕ := ratio_allen + ratio_ben + ratio_charlie
def area_per_part := total_area / total_parts
def charlie_parts := 7

-- Prove the main statement
theorem charlie_paints_140_square_feet : charlie_parts * area_per_part = 140 := by
  sorry

end charlie_paints_140_square_feet_l62_62255


namespace remainder_eq_52_l62_62710

noncomputable def polynomial : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 7

theorem remainder_eq_52 : Polynomial.eval (-3) polynomial = 52 :=
by
    sorry

end remainder_eq_52_l62_62710


namespace max_min_expression_l62_62324

def sqrt (x : ℝ) : ℝ := real.sqrt x

theorem max_min_expression :
  (∃ x y : ℝ, sqrt (x - 3) + sqrt (y - 4) = 4 ∧ 2 * x + 3 * y = 66) ∧
  (∃ x y : ℝ, sqrt (x - 3) + sqrt (y - 4) = 4 ∧ 2 * x + 3 * y = 37.2) :=
by
  sorry

end max_min_expression_l62_62324


namespace number_of_correct_statements_l62_62590

theorem number_of_correct_statements : 
    (∀ (residual_plot : Prop), (residual_points_evenly_distributed_within_horizontal_band residual_plot → chosen_model_appropriate residual_plot)) ∧
    (∀ (regression_model : Prop), (coefficient_of_determination R2 regression_model → better_model_fits regression_model)) ∧
    (∀ (model1 model2 : Prop), (sum_of_squared_residuals model1 < sum_of_squared_residuals model2 → better_fitting_effect model1 model2)) →
    number_of_correct_statements = 3 := 
by
  sorry

end number_of_correct_statements_l62_62590


namespace tetrahedron_circumscribed_sphere_l62_62670

-- Define the properties of a tetrahedron
structure Tetrahedron (V : Type) :=
(vertices : (fin 4) → V)

-- The orthocentric system condition
def orthocentric {V : Type} [inner_product_space ℝ V] (T : Tetrahedron V) : Prop :=
∃ O : V, ∀ (i : fin 4), ∃ height : V, (∀ j ≠ i, inner_product height (T.vertices j - O) = 0)

-- Prove that for any tetrahedron, there exists a circumscribed sphere
theorem tetrahedron_circumscribed_sphere {V : Type} [inner_product_space ℝ V] (T : Tetrahedron V) :
  orthocentric T ↔ (∃ O : V, ∀ v ∈ finset.univ.image T.vertices, dist O v = dist O (T.vertices 0)) :=
sorry

end tetrahedron_circumscribed_sphere_l62_62670


namespace circle_tangent_to_hyperbola_asymptotes_radius_l62_62369

theorem circle_tangent_to_hyperbola_asymptotes_radius :
  ∀ (r : ℝ), (r > 0) →
  (∀ (x y : ℝ), ((x - real.sqrt 2)^2 + y^2 = r^2) →
  (x^2 - y^2 = 1) →
  (∀ y : ℝ, y = x ∨ y = -x) →
  (x - real.sqrt 2)^2 + y^2 = r^2) →
  r = 1 :=
by
  intros r h1 h2
  sorry

end circle_tangent_to_hyperbola_asymptotes_radius_l62_62369


namespace graph_passes_through_point_l62_62003

theorem graph_passes_through_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 0 ∧ y = a^(x - 2) - 1 :=
by
  use 2, 0
  split
  { rfl }
  split
  { rfl }
  { sorry }

end graph_passes_through_point_l62_62003


namespace regular_decagon_inscribed_m_value_l62_62539

noncomputable def m_value : ℝ := 90 * real.sqrt 2 / 7

theorem regular_decagon_inscribed_m_value
  (B : fin 10 → ℝ × ℝ)
  (Q : ℝ × ℝ)
  (circle_area : ℝ)
  (H1 : circle_area = 4)
  (arc_area_B1B2 : ℝ)
  (arc_area_B4B5 : ℝ)
  (arc_area_B1B2_cond : arc_area_B1B2 = 1 / 10)
  (arc_area_B4B5_cond : arc_area_B4B5 = 1 / 12) :
  (∃ m : ℝ, m = m_value ∧
    let arc_area_B7B8 := 1 / 9 - real.sqrt 2 / m in
    arc_area_B7B8 = arc_area_B1B2 - (arc_area_B1B2 - arc_area_B4B5)) :=
begin
  sorry
end

end regular_decagon_inscribed_m_value_l62_62539


namespace find_shorter_piece_length_l62_62640

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x = 8

theorem find_shorter_piece_length : ∃ x : ℕ, (20 - x) > 0 ∧ 2 * x = (20 - x) + 4 ∧ shorter_piece_length x :=
by
  -- There exists an x that satisfies the conditions
  use 8
  -- Prove the conditions are met
  sorry

end find_shorter_piece_length_l62_62640


namespace even_function_a_zero_l62_62432

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62432


namespace cube_root_of_27_l62_62112

theorem cube_root_of_27 : 
  ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
begin
  sorry
end

end cube_root_of_27_l62_62112


namespace four_painters_workdays_l62_62805

theorem four_painters_workdays :
  (∃ (c : ℝ), ∀ (n : ℝ) (d : ℝ), n * d = c) →
  (p5 : ℝ) (d5 : ℝ) (p5 * d5 = 7.5) →
  ∀ D : ℝ, 4 * D = 7.5 →
  D = (1 + 7/8) := 
by {
  sorry
}

end four_painters_workdays_l62_62805


namespace compare_negative_fractions_l62_62282

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l62_62282


namespace intersection_distance_l62_62701

noncomputable def distance_between_intersections : ℝ :=
  let f₁ := λ (x y : ℝ), x^2 + y^2 = 25
  let f₂ := λ (x y : ℝ), x^2 + y = 17
  if f₁ (sqrt 22) (-5) ∧ f₂ (sqrt 22) (-5) then
    2 * sqrt 22
  else 0 -- In a full proof, we would handle all cases correctly

theorem intersection_distance :
  let f₁ := λ (x y : ℝ), x^2 + y^2 = 25 in
  let f₂ := λ (x y : ℝ), x^2 + y = 17 in
  ∀ x y : ℝ, (f₁ x y ∧ f₂ x y) → distance_between_intersections = 2 * sqrt 22 :=
sorry

end intersection_distance_l62_62701


namespace possible_N_values_l62_62975

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62975


namespace loop_execution_count_correct_l62_62134

def loop_exec_count (start : ℕ) (end : ℕ) : ℕ :=
  end - start + 1

theorem loop_execution_count_correct :
  loop_exec_count 2 20 = 19 :=
by
  -- proof goes here
  sorry

end loop_execution_count_correct_l62_62134


namespace letian_estimate_l62_62593

variables (x y z w : ℝ)

theorem letian_estimate :
  x > y → y > 0 → z > 0 → w > 0 → z > w → (x + z) - (y - w) > x - y :=
by
  intros hxy hy hz hw hzw
  calc (x + z) - (y - w) = x - y + (z + w) : by linarith
              ... > x - y : by linarith

end letian_estimate_l62_62593


namespace cookie_cost_l62_62679

theorem cookie_cost 
    (initial_amount : ℝ := 100)
    (latte_cost : ℝ := 3.75)
    (croissant_cost : ℝ := 3.50)
    (days : ℕ := 7)
    (num_cookies : ℕ := 5)
    (remaining_amount : ℝ := 43) :
    (initial_amount - remaining_amount - (days * (latte_cost + croissant_cost))) / num_cookies = 1.25 := 
by
  sorry

end cookie_cost_l62_62679


namespace num_possible_values_l62_62933

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62933


namespace monotone_decreasing_interval_l62_62747

def f (x : ℝ) : ℝ := cos (2 * x - π / 6) + sin (2 * x)

theorem monotone_decreasing_interval :
  ∃ (a b : ℝ), a = π / 6 ∧ b = 2 * π / 3 ∧ 
               (∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x) := by
  sorry

end monotone_decreasing_interval_l62_62747


namespace average_income_independent_of_distribution_l62_62467

namespace AverageIncomeProof

variables (A E : ℝ) (total_employees : ℕ) (h : total_employees = 10)

/-- The average income in December does not depend on the distribution method -/
theorem average_income_independent_of_distribution : 
  (∃ (income : ℝ), income = (A + E) / total_employees) :=
by
  have h1 : total_employees = 10 := h
  exists (A + E) / total_employees
  sorry

end AverageIncomeProof

end average_income_independent_of_distribution_l62_62467


namespace sum_of_four_consecutive_integers_prime_factor_l62_62160

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l62_62160


namespace even_function_a_eq_zero_l62_62438

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62438


namespace relationship_abc_l62_62384

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_abc (h_symm : ∀ x, f x = f (-x))
                         (h_ineq : ∀ x, x < 0 → f x + x * (deriv f x) < 0) :
  (log π 3) * (f (log π 3)) > 20.2 * (f 20.2) ∧
  20.2 * (f 20.2) > (log 3 9) * (f (log 3 9)) :=
by
  sorry

end relationship_abc_l62_62384


namespace scarlet_savings_l62_62542

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l62_62542


namespace values_of_b_for_real_root_l62_62700

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^5 + b * x^4 - x^3 + b * x^2 - x + b = 0

theorem values_of_b_for_real_root :
  {b : ℝ | polynomial_has_real_root b} = {b : ℝ | b ≤ -1 ∨ b ≥ 1} :=
sorry

end values_of_b_for_real_root_l62_62700


namespace quad_perimeter_l62_62325

open Real

variables (α a x y : ℝ)
variables (AB AD BC CD : ℝ)
variables (P : ℝ)

noncomputable def perimeter_of_quadrilateral (α : ℝ) (a : ℝ) (x y : ℝ) : ℝ :=
  2 * a * (1 + cos α)

theorem quad_perimeter (α : ℝ) (a x y : ℝ) 
  (h1 : AB = a)
  (h2 : CD = a)
  (h3 : ∠BAD = α)
  (h4 : ∠BCD = α)
  (h5 : α < π / 2)
  (h6 : BC ≠ AD)
  (h7 : x + y = 2 * a * cos α) : 
  P = 2 * a * (1 + cos α) :=
by
  -- Preliminary proof steps showing the calculation
  sorry

end quad_perimeter_l62_62325


namespace days_in_month_l62_62803

theorem days_in_month (days_took : ℕ) (days_forgot : ℕ) (h_took : days_took = 29) (h_forgot : days_forgot = 2) :
  days_took + days_forgot = 31 :=
by
  rw [h_took, h_forgot]
  sorry

end days_in_month_l62_62803


namespace real_part_sqrt_sum_le_sum_abs_l62_62068

variable {n : ℕ}
variable {x y : Fin n → ℝ}

theorem real_part_sqrt_sum_le_sum_abs (z : ℂ) (hz : z = ∑ k, (x k + I * y k) ^ 2) (p q : ℝ)
  (h_sqrt_z : complex.sqrt z = p + q * I) :
  |p| ≤ ∑ k, |x k| := by
  sorry

end real_part_sqrt_sum_le_sum_abs_l62_62068


namespace number_of_ways_to_sum_consecutive_integers_equals_16_l62_62895

def S (k n : ℕ) : ℕ := k * (2 * n + k - 1) / 2

theorem number_of_ways_to_sum_consecutive_integers_equals_16 :
  (∑ k (n : ℕ), if S k n = 2015 then 1 else 0) = 16 :=
sorry

end number_of_ways_to_sum_consecutive_integers_equals_16_l62_62895


namespace cube_root_of_27_eq_3_l62_62113

theorem cube_root_of_27_eq_3 : real.cbrt 27 = 3 :=
by {
  have h : 27 = 3 ^ 3 := by norm_num,
  rw real.cbrt_eq_iff_pow_eq (by norm_num : 0 ≤ 27) h,
  norm_num,
  sorry
}

end cube_root_of_27_eq_3_l62_62113


namespace cost_of_one_round_l62_62011

-- Define the conditions
def total_cost : ℝ := 400
def rounds : ℕ := 5
def cost_per_round : ℝ := total_cost / rounds

-- State the theorem to be proven
theorem cost_of_one_round : cost_per_round = 80 := 
by
  sorry

end cost_of_one_round_l62_62011


namespace arithmetic_sequence_general_term_l62_62057

theorem arithmetic_sequence_general_term :
  ∃ d : ℤ, d = 6 ∧ (∀ n : ℕ, a_n = 6 * n - 3)
:= by
  -- Conditions
  let a_1 := 3
  let d := 6
  have h1 : a_2 + a_5 = 36 := by
    let a_2 := a_1 + d
    let a_5 := a_1 + 4 * d
    change (a_1 + d) + (a_1 + 4 * d) = 36
    ring
    exact a_2 + a_5 -- continues proof needed
  have h2 : 6 + 5 * d = 36 := by
    -- Proof steps here as needed
    -- Should be expanded to complete the proof
    sorry

  -- Assuming the result already
  use d
  split
  exact h2
  intro n
  exact 6 * n - 3 -- continue the proof 
  sorry

end arithmetic_sequence_general_term_l62_62057


namespace distance_to_x_axis_P_l62_62027

-- The coordinates of point P
def P : ℝ × ℝ := (3, -2)

-- The distance from point P to the x-axis
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs (point.snd)

theorem distance_to_x_axis_P : distance_to_x_axis P = 2 :=
by
  -- Use the provided point P and calculate the distance
  sorry

end distance_to_x_axis_P_l62_62027


namespace problem_l62_62518

def f (x : ℝ) : ℝ :=
  if x ≥ 6 then x^2 - 4 * x + 4
  else if x > 0 then -x^2 + 3 * x + 4
  else 4 * x + 8

theorem problem : f (-5) + f 3 + f 8 = 28 := by
  sorry

end problem_l62_62518


namespace asymptotes_of_hyperbola_l62_62124

-- Definition of hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- The main theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l62_62124


namespace solve_eqn_solution_l62_62547

noncomputable def solve_equation (z : ℂ) : Prop :=
  (4 * (sin z)^4 / (1 + cos (2 * z))^2 - 2 / cos z^2 - 1 = 0)

theorem solve_eqn_solution (z : ℂ) (k : ℤ) 
  (h1 : cos z ≠ 0) 
  (h2 : cos (2 * z) ≠ -1) : 
  solve_equation z ↔ 
  ∃ k : ℤ, z = (π / 3) * (3 * k + 1) ∨ z = (π / 3) * (3 * k - 1) := 
sorry

end solve_eqn_solution_l62_62547


namespace trig_problem_1_trig_problem_2_l62_62273

noncomputable def trig_expr_1 : ℝ :=
  Real.cos (-11 * Real.pi / 6) + Real.sin (12 * Real.pi / 5) * Real.tan (6 * Real.pi)

noncomputable def trig_expr_2 : ℝ :=
  Real.sin (420 * Real.pi / 180) * Real.cos (750 * Real.pi / 180) +
  Real.sin (-330 * Real.pi / 180) * Real.cos (-660 * Real.pi / 180)

theorem trig_problem_1 : trig_expr_1 = Real.sqrt 3 / 2 :=
by
  sorry

theorem trig_problem_2 : trig_expr_2 = 1 :=
by
  sorry

end trig_problem_1_trig_problem_2_l62_62273


namespace expression_positive_intervals_l62_62690

theorem expression_positive_intervals (x : ℝ) : 
  (x + 1) * (x - 1) * (x - 2) > 0 ↔ (x ∈ set.Ioo (-1) 1 ∨ x ∈ set.Ioi 2) :=
sorry

end expression_positive_intervals_l62_62690


namespace triangle_ABC_AC_l62_62776

-- Define the given conditions
def AB : ℝ := 3
def BC : ℝ := 2
def angle_B : ℝ := real.pi / 3  -- 60 degrees in radians

-- The question to prove:
theorem triangle_ABC_AC : AC = real.sqrt 7 :=
  by
    sorry  -- The proof is omitted as per the instruction.

end triangle_ABC_AC_l62_62776


namespace runway_show_duration_l62_62654

theorem runway_show_duration
  (evening_wear_time : ℝ) (bathing_suits_time : ℝ) (formal_wear_time : ℝ) (casual_wear_time : ℝ)
  (evening_wear_sets : ℕ) (bathing_suits_sets : ℕ) (formal_wear_sets : ℕ) (casual_wear_sets : ℕ)
  (num_models : ℕ) :
  evening_wear_time = 4 → bathing_suits_time = 2 → formal_wear_time = 3 → casual_wear_time = 2.5 →
  evening_wear_sets = 4 → bathing_suits_sets = 2 → formal_wear_sets = 3 → casual_wear_sets = 5 →
  num_models = 10 →
  (evening_wear_time * evening_wear_sets + bathing_suits_time * bathing_suits_sets
   + formal_wear_time * formal_wear_sets + casual_wear_time * casual_wear_sets) * num_models = 415 :=
by
  intros
  sorry

end runway_show_duration_l62_62654


namespace side_salad_cost_l62_62076

theorem side_salad_cost (T S : ℝ)
  (h1 : T + S + 4 + 2 = 2 * T) 
  (h2 : (T + S + 4 + 2) + T = 24) : S = 2 :=
by
  sorry

end side_salad_cost_l62_62076


namespace solution_set_l62_62327

-- Let π be the mathematical constant pi
-- Definition of the inequality given in the problem
def inequality (x : ℝ) : Prop :=
  (π / 2) ^ (x - 1) ^ 2 ≤ (2 / π) ^ (x ^ 2 - 5 * x - 5)

-- The proof statement that checks the solution set of the inequality
theorem solution_set : Set.Icc (-1/2 : ℝ) (4 : ℝ) = {x : ℝ | inequality x} :=
sorry

end solution_set_l62_62327


namespace resulting_number_not_power_of_2_l62_62304

/-- Each card has a five-digit number written on it from 11111 to 99999, 
and these cards are arranged in any order to form one long number with 444,445 digits. 
Prove that this resulting number cannot be a power of 2. -/
theorem resulting_number_not_power_of_2 
  (cards : List ℕ)
  (h1 : ∀ n ∈ cards, 11111 ≤ n ∧ n ≤ 99999)
  (h2 : cards.length = 88889)
  (h3 : ∃ A, nat.digits 10 A = cards.flat_map (λ n, nat.digits 10 n))
  (h4 : (nat.digits 10 A).length = 444445) :
  ¬ ∃ k, A = 2^k :=
by
  sorry

end resulting_number_not_power_of_2_l62_62304


namespace sum_le_101_l62_62693

-- Define the grid size and properties
def grid_size : ℕ := 101

noncomputable def grid := fin grid_size → fin grid_size → ℝ

def abs_bound (a : grid) : Prop :=
  ∀ i j, |a i j| ≤ 1

def subgrid_sum_zero (a : grid) : Prop :=
  ∀ i j, i < grid_size - 1 → j < grid_size - 1 →
    a i j + a i (j + 1) + a (i + 1) j + a (i + 1) (j + 1) = 0

def total_sum (a : grid) : ℝ :=
  ∑ i, ∑ j, a i j

theorem sum_le_101 (a : grid) :
  abs_bound a →
  subgrid_sum_zero a →
  total_sum a ≤ 101 :=
sorry

end sum_le_101_l62_62693


namespace camel_humps_l62_62941

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62941


namespace min_width_of_garden_l62_62500

theorem min_width_of_garden (w : ℝ) (h : 0 < w) (h1 : w * (w + 20) ≥ 120) : w ≥ 4 :=
sorry

end min_width_of_garden_l62_62500


namespace even_function_a_zero_l62_62424

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62424


namespace bug_probability_l62_62642

def tetrahedron_vertices := {a, b, c, d}

variable start_vertex : tetrahedron_vertices
variable probability_of_visiting_all_vertices : ℕ → ℕ → ℚ

theorem bug_probability :
  (probability_of_visiting_all_vertices 3 4) = 1 / 4 :=
by
  -- Start at one vertex of a tetrahedron
  -- Each move along edges has equal probability
  -- After three moves:
  -- Check if every vertex has been visited exactly once
  sorry

end bug_probability_l62_62642


namespace power_function_value_l62_62015

theorem power_function_value {a : ℝ} (f : ℝ → ℝ) (h₁ : ∀ x, f(x) = x^a) (h₂ : f(4) = 2) : f(9) = 3 :=
sorry

end power_function_value_l62_62015


namespace perpendicular_vectors_l62_62408

def vec_a : ℝ × ℝ × ℝ := (2, 1, -3)
def vec_b (λ : ℝ) : ℝ × ℝ × ℝ := (4, 2, λ)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem perpendicular_vectors (λ : ℝ) (h : dot_product vec_a (vec_b λ) = 0) : λ = 10 / 3 :=
by
  sorry

end perpendicular_vectors_l62_62408


namespace remainder_of_2_pow_23_mod_5_l62_62999

theorem remainder_of_2_pow_23_mod_5 
    (h1 : (2^2) % 5 = 4)
    (h2 : (2^3) % 5 = 3)
    (h3 : (2^4) % 5 = 1) :
    (2^23) % 5 = 3 :=
by
  sorry

end remainder_of_2_pow_23_mod_5_l62_62999


namespace correct_propositions_count_l62_62688

-- Define the double factorial for any positive integer n
def double_factorial : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := if n % 2 = 0 then n * double_factorial (n - 2) else n * double_factorial (n - 2)

-- Define the four propositions
def prop1 : Prop := (double_factorial 2009) * (double_factorial 2008) = (Nat.factorial 2009)
def prop2 : Prop := double_factorial 2008 = 2 * Nat.factorial 1004
def prop3 : Prop := (double_factorial 2008 % 10) = 0
def prop4 : Prop := (double_factorial 2009 % 10) = 5

-- Main theorem to confirm how many of the propositions are correct
theorem correct_propositions_count : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 :=
by
  sorry

end correct_propositions_count_l62_62688


namespace min_n_conditions_l62_62067

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem min_n_conditions :
  (∀ i, 0 ≤ x i) →
  (∑ i, x i = 1) →
  (∑ i, (x i)^2 ≤ 1 / 50) →
  n ≥ 50 :=
by
  intro h1 h2 h3
  sorry

end min_n_conditions_l62_62067


namespace avg_comparisons_sequential_search_l62_62123

theorem avg_comparisons_sequential_search 
  (n : ℕ) 
  (h : n = 100) 
  (unordered : true) 
  (not_present : true) : 
  avg_comparisons_needed n = 100 := 
by 
  -- We add "sorry" to skip the proof
  sorry

end avg_comparisons_sequential_search_l62_62123


namespace original_rent_l62_62624

theorem original_rent {avg_rent_before avg_rent_after : ℝ} (total_before total_after increase_percentage diff_increase : ℝ) :
  avg_rent_before = 800 → 
  avg_rent_after = 880 → 
  total_before = 4 * avg_rent_before → 
  total_after = 4 * avg_rent_after → 
  diff_increase = total_after - total_before → 
  increase_percentage = 0.20 → 
  diff_increase = increase_percentage * R → 
  R = 1600 :=
by sorry

end original_rent_l62_62624


namespace final_remaining_coffee_l62_62857

-- Definitions based on conditions
def initial_coffee_amount : ℝ := 12
def consumed_on_way_to_work : ℝ := initial_coffee_amount / 4
def consumed_at_office : ℝ := initial_coffee_amount / 2
def consumed_when_remembered : ℝ := 1

-- Total consumed
def total_consumed : ℝ :=
  consumed_on_way_to_work + consumed_at_office + consumed_when_remembered

-- Remaining coffee
def remaining_coffee : ℝ :=
  initial_coffee_amount - total_consumed

-- Proof statement
theorem final_remaining_coffee : remaining_coffee = 2 := by
  sorry

end final_remaining_coffee_l62_62857


namespace number_eight_is_desired_number_l62_62617

def array : list (list ℕ) :=
  [ [ 12, 7, 9, 5, 6 ],
    [ 14, 8, 16, 14, 10 ],
    [ 10, 4, 9, 7, 11 ],
    [ 15, 5, 18, 13, 3 ],
    [ 9, 3, 6, 11, 4 ] ]

def largest_in_column (col : ℕ) (n : ℕ) : Prop :=
  ∀ row, list.nth (list.nth array row) col ≤ some n

def smallest_in_row (row : ℕ) (n : ℕ) : Prop :=
  ∀ col, list.nth (list.nth array row) col ≥ some n

theorem number_eight_is_desired_number : 
  ∃ row col, 
  list.nth (list.nth array row) col = some 8 ∧
  largest_in_column col 8 ∧
  smallest_in_row row 8 :=
by
  sorry

end number_eight_is_desired_number_l62_62617


namespace problem_l62_62772

theorem problem (n : ℝ) (h : (n - 2009)^2 + (2008 - n)^2 = 1) : (n - 2009) * (2008 - n) = 0 := 
by
  sorry

end problem_l62_62772


namespace find_constants_range_of_f_const_l62_62355

-- Define the function and its derivative
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2
def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

-- State the conditions
theorem find_constants (a b : ℝ) (h₀ : a > 1) (h₁ : f' (-1) a b = 0) (h₂ : f (-1) a b = 0) :
  a = 2 ∧ b = 9 := sorry

-- State the function definition again with found constants
def f_const (x : ℝ) : ℝ := f x 2 9

-- State interval endpoint values
def interval_endpoints (x : ℝ) : Prop := x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0

-- Determine the range of f_const on [-4, 0]
theorem range_of_f_const :
  ∀ x : ℝ, (interval_endpoints x) → f_const x ∈ set.Icc 0 4 := sorry

end find_constants_range_of_f_const_l62_62355


namespace ratio_of_areas_l62_62800

noncomputable def triangle_PQR := {P : Type} [inhabited P] {Q R S : P}
variables {PQ PR QR : ℝ}
variables (PQS_area PRS_area : ℝ)

def angle_bisector (PQ PR QR : ℝ) : Prop :=
  PQ = 18 ∧ PR = 27 ∧ QR = 22

theorem ratio_of_areas (PQ PR QR PQS_area PRS_area : ℝ)
  (h : angle_bisector PQ PR QR) (hPS : PQS_area / PRS_area = PQ / PR):
  PQS_area / PRS_area = 2 / 3 := 
by {
  sorry
}

end ratio_of_areas_l62_62800


namespace average_income_independence_l62_62474

theorem average_income_independence (A E : ℝ) (n : ℕ) (h : n = 10) :
  let avg_income := (A + E) / (n : ℝ)
  in avg_income = (A + E) / 10 :=
by
  intros
  have h1 : (n : ℝ) = 10 := by simp [h]
  rw h1
  sorry

end average_income_independence_l62_62474


namespace combined_prism_volume_is_66_l62_62253

noncomputable def volume_of_combined_prisms
  (length_rect : ℝ) (width_rect : ℝ) (height_rect : ℝ)
  (base_tri : ℝ) (height_tri : ℝ) (length_tri : ℝ) : ℝ :=
  let volume_rect := length_rect * width_rect * height_rect
  let area_tri := (1 / 2) * base_tri * height_tri
  let volume_tri := area_tri * length_tri
  volume_rect + volume_tri

theorem combined_prism_volume_is_66 :
  volume_of_combined_prisms 6 4 2 3 3 4 = 66 := by
  sorry

end combined_prism_volume_is_66_l62_62253


namespace seats_between_T17_and_T39_l62_62484

theorem seats_between_T17_and_T39 :
  ∀ (n1 n2 : ℕ), n1 = 17 → n2 = 39 → (n2 - n1 - 1 = 21) :=
by
  intros n1 n2 h1 h2
  rw [h1, h2]
  sorry

end seats_between_T17_and_T39_l62_62484


namespace choose_with_at_least_one_girl_l62_62342

theorem choose_with_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_students := boys + girls
  let ways_choose_4 := Nat.choose total_students 4
  let ways_all_boys := Nat.choose boys 4
  ways_choose_4 - ways_all_boys = 14 := by
  sorry

end choose_with_at_least_one_girl_l62_62342


namespace compare_negative_fractions_l62_62280

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l62_62280


namespace hyperbola_equation_l62_62322

theorem hyperbola_equation {x y : ℝ} (h1 : x ^ 2 / 2 - y ^ 2 = 1) 
  (h2 : x = -2) (h3 : y = 2) : y ^ 2 / 2 - x ^ 2 / 4 = 1 :=
by sorry

end hyperbola_equation_l62_62322


namespace total_blue_balloons_l62_62498

theorem total_blue_balloons (Joan_balloons : ℕ) (Melanie_balloons : ℕ) (Alex_balloons : ℕ) 
  (hJoan : Joan_balloons = 60) (hMelanie : Melanie_balloons = 85) (hAlex : Alex_balloons = 37) :
  Joan_balloons + Melanie_balloons + Alex_balloons = 182 :=
by
  sorry

end total_blue_balloons_l62_62498


namespace cory_fruit_eating_orders_l62_62291

theorem cory_fruit_eating_orders :
  let total_fruits := 9
  let apples := 4
  let oranges := 2
  let bananas := 2
  let pear := 1
  factorial total_fruits / (factorial apples * factorial oranges * factorial bananas * factorial pear) = 3780 :=
by
  sorry

end cory_fruit_eating_orders_l62_62291


namespace average_income_proof_l62_62470

def average_income_independent_of_bonus_distribution (A E : ℝ) : Prop :=
  ∀ (distribution_method : (fin 10 → ℝ) → Prop), 
    (∑ i, (distribution_method (λ _, (A + E) / 10)) = A + E) →
    (∀ i, distribution_method (λ _, (A + E) / 10) i = (A + E) / 10)

theorem average_income_proof (A E : ℝ) :
  average_income_independent_of_bonus_distribution A E := by
  sorry

end average_income_proof_l62_62470


namespace average_age_increase_l62_62886

theorem average_age_increase (a t : ℕ) (n : ℕ) (h₁ : n = 22) (h₂ : a = 21) (h₃ : t = 44) :
  let total_age_students := n * a,
      total_age_with_teacher := total_age_students + t,
      new_average_with_teacher := total_age_with_teacher / (n + 1) in
  new_average_with_teacher - a = 1 :=
by
  sorry

end average_age_increase_l62_62886


namespace water_spill_l62_62589

theorem water_spill (num_players : ℕ) (initial_water_liters : ℝ) (water_per_player_ml : ℝ) 
    (remaining_water_ml : ℝ) (water_spilled_ml : ℝ) :
  num_players = 30 →
  initial_water_liters = 8 →
  water_per_player_ml = 200 →
  remaining_water_ml = 1750 →
  water_spilled_ml = (initial_water_liters * 1000) - (num_players * water_per_player_ml) - remaining_water_ml →
  water_spilled_ml = 250 := by
  intros
  simp
  sorry

end water_spill_l62_62589


namespace complex_number_identity_l62_62560

theorem complex_number_identity (i : ℂ) (hi : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_number_identity_l62_62560


namespace problem_statement_l62_62055

noncomputable theory

variables (a b c x y z : ℝ)

def equations (a b c x y z : ℝ) :=
17 * x + b * y + c * z = 0 ∧
a * x + 29 * y + c * z = 0 ∧
a * x + b * y + 37 * z = 0

theorem problem_statement (h_eqns : equations a b c x y z) (h_a : a ≠ 17) (h_x : x ≠ 0) :
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) = 1 :=
sorry

end problem_statement_l62_62055


namespace time_A_to_complete_race_l62_62781

noncomputable def km_race_time (V_B : ℕ) : ℚ :=
  940 / V_B

theorem time_A_to_complete_race : km_race_time 6 = 156.67 := by
  sorry

end time_A_to_complete_race_l62_62781


namespace unique_solution_exists_q_l62_62314

theorem unique_solution_exists_q :
  (∃ q : ℝ, q ≠ 0 ∧ (∀ x y : ℝ, (2 * q * x^2 - 20 * x + 5 = 0) ∧ (2 * q * y^2 - 20 * y + 5 = 0) → x = y)) ↔ q = 10 := 
sorry

end unique_solution_exists_q_l62_62314


namespace even_function_a_zero_l62_62428

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62428


namespace retail_price_machine_l62_62249

theorem retail_price_machine (P : ℝ) :
  let wholesale_price := 99
  let discount_rate := 0.10
  let profit_rate := 0.20
  let selling_price := wholesale_price + (profit_rate * wholesale_price)
  0.90 * P = selling_price → P = 132 :=

by
  intro wholesale_price discount_rate profit_rate selling_price h
  sorry -- Proof will be handled here

end retail_price_machine_l62_62249


namespace true_equation_l62_62721

variable (a b c d : ℕ)

noncomputable def eqRat (x y : ℕ) := 
  (a * y) = (b * x)

theorem true_equation (h : eqRat a d ∧ eqRat c b) : 
  eqRat (a + c) (b + d) :=
by
  sorry

end true_equation_l62_62721


namespace lisa_eggs_l62_62847

theorem lisa_eggs :
  ∃ x : ℕ, (5 * 52) * (4 * x + 3 + 2) = 3380 ∧ x = 2 :=
by
  sorry

end lisa_eggs_l62_62847


namespace even_function_a_eq_zero_l62_62442

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62442


namespace find_largest_number_l62_62199

-- Define the given conditions
variables (d b c a : ℕ)
hypothesis h1 : 0 ≤ b ∧ b < 13
hypothesis h2 : 0 ≤ c ∧ c < 15
hypothesis h3 : b - c = 12
hypothesis h4 : a = 13 * d + b
hypothesis h5 : a = 15 * d + c

-- The statement that we need to prove
theorem find_largest_number : a = 90 :=
by
  sorry  -- proof goes here

end find_largest_number_l62_62199


namespace possible_N_values_l62_62951

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62951


namespace point_closer_to_F_prob_l62_62799

open Triangle

noncomputable def triangle_DEF := {DE : 7, EF : 6, FD : 5}

/-- The probability that a point selected randomly inside triangle DEF is closer to F than to either D or E is 1/4. -/
theorem point_closer_to_F_prob :
  ∃ (Q : Point), (is_in_triangle Q triangle_DEF) → (probability_closer_to_F_than_D_or_E Q = 1 / 4) :=
sorry

end point_closer_to_F_prob_l62_62799


namespace billie_bakes_3_pies_per_day_l62_62309

theorem billie_bakes_3_pies_per_day :
  ∀ (P : ℕ), (11 * P - 4) * 2 = 58 → P = 3 :=
by
  intro P,
  intro h,
  sorry

end billie_bakes_3_pies_per_day_l62_62309


namespace product_square_neq_sum_of_cubes_l62_62632

theorem product_square_neq_sum_of_cubes (x : ℕ → ℕ) (n : ℕ) 
  (h1 : ∀ i, i < n → x i > 1) (h2 : n ≥ 3) : 
  ((∏ i in finset.range n, (x i)) ^ 2) ≠ (∑ i in finset.range n, (x i) ^ 3) :=
sorry

end product_square_neq_sum_of_cubes_l62_62632


namespace triangle_height_l62_62884

theorem triangle_height (base area height : ℝ)
    (h_base : base = 4)
    (h_area : area = 16)
    (h_area_formula : area = (base * height) / 2) :
    height = 8 :=
by
  sorry

end triangle_height_l62_62884


namespace rectangle_length_difference_l62_62732

variable (s l w : ℝ)

-- Conditions
def condition1 : Prop := 2 * (l + w) = 4 * s + 4
def condition2 : Prop := w = s - 2

-- Theorem to prove
theorem rectangle_length_difference
  (s l w : ℝ)
  (h1 : condition1 s l w)
  (h2 : condition2 s w) : l = s + 4 :=
by
sorry

end rectangle_length_difference_l62_62732


namespace josh_anna_marriage_years_l62_62050

theorem josh_anna_marriage_years :
  ∃ x : ℕ, x + 25 = 55 :=
begin
  sorry
end

end josh_anna_marriage_years_l62_62050


namespace find_a_if_f_even_l62_62443

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62443


namespace metal_waste_calculation_l62_62651

variables (l w : ℝ)
hypothesis (h_w_l : l > w)

theorem metal_waste_calculation : l * w - (w ^ 2 / 2) = lw - w ^ 2 / 2 :=
sorry

end metal_waste_calculation_l62_62651


namespace sum_f_2011_l62_62356

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, sin x + cos x
| (n + 1) := λ x, (deriv (f n)) x

theorem sum_f_2011 (x : ℝ) : (∑ i in Finset.range 2011, (f i x)) = -sin x + cos x :=
by
  sorry

end sum_f_2011_l62_62356


namespace closest_point_on_plane_l62_62707

def vec3 := ℝ × ℝ × ℝ -- A definition for 3-dimensional vectors

-- Definitions for the plane equation and the point A
def plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 2 * z = 40
def point_A : vec3 := (3, 1, 4)

-- The statement that the point P is on the plane and is the closest to point A
theorem closest_point_on_plane : 
  ∃ P : vec3, plane P.1 P.2 P.3 ∧ P = (92 / 19, -2 / 19, 90 / 19) :=
sorry

end closest_point_on_plane_l62_62707


namespace painters_work_days_l62_62809

/-
It takes five painters working at the same rate 1.5 work-days to finish a job.
If only four painters are available, prove how many work-days will it take them to finish the job, working at the same rate.
-/

theorem painters_work_days (days5 : ℚ) (h : days5 = 3 / 2) :
  ∃ days4 : ℚ, 5 * days5 = 4 * days4 ∧ days4 = 15 / 8 :=
  by
    use 15 / 8
    split
    · calc
        5 * days5 = 5 * (3 / 2) : by rw h
        ... = 15 / 2 : by norm_num
        ... = 4 * (15 / 8) : by norm_num
    · refl

end painters_work_days_l62_62809


namespace solution_l62_62516

noncomputable def is_in_circle (x y : ℤ) : Prop :=
  (x - 2) ^ 2 + (y - 5) ^ 2 < 9

noncomputable def gaussian_integers_in_circle : set (ℤ × ℤ) :=
  { z | let x := z.fst, y := z.snd in -1 < x ∧ x < 5 ∧ 2 < y ∧ y < 8 ∧ is_in_circle x y }

theorem solution :
  gaussian_integers_in_circle =
    { (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
      (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
      (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
      (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
      (4, 3), (4, 4), (4, 5), (4, 6), (4, 7) } :=
sorry

end solution_l62_62516


namespace csc_150_degrees_l62_62699

theorem csc_150_degrees : csc (ofReal 150) = 2 :=
  sorry

end csc_150_degrees_l62_62699


namespace max_triangles_in_quadrilateral_l62_62078

-- Define the setup of the quadrilateral and points with the non-collinear condition
def is_quadrilateral_with_14_points (points : Finset Point) : Prop :=
  points.card = 14 ∧
  ∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → ¬ Collinear (a :: b :: c :: [])

-- Define the main theorem stating the problem
theorem max_triangles_in_quadrilateral {points : Finset Point} : 
  is_quadrilateral_with_14_points points →
  (number_of_distinct_nonoverlapping_triangles points = 22) :=
begin
  sorry
end

end max_triangles_in_quadrilateral_l62_62078


namespace stratified_sampling_category_A_l62_62778

def total_students_A : ℕ := 2000
def total_students_B : ℕ := 3000
def total_students_C : ℕ := 4000
def total_students : ℕ := total_students_A + total_students_B + total_students_C
def total_selected : ℕ := 900

theorem stratified_sampling_category_A :
  (total_students_A * total_selected) / total_students = 200 :=
by
  sorry

end stratified_sampling_category_A_l62_62778


namespace proof_problem_l62_62306

-- Define the parametric equations of line L
def parametric_line (α t : ℝ) : ℝ × ℝ := (2 + t * cos α, t * sin α)

-- Define the polar equation of curve C
def polar_curve (p θ : ℝ) : Prop := p^2 * cos θ ^ 2 + 2 * p^2 * sin θ ^ 2 = 12

-- Convert polar to Cartesian and define curve C in Cartesian coordinates
def cartesian_curve (x y : ℝ) : Prop := x^2 + 2 * y^2 = 12

-- Fixed point A through which line L always passes
def fixed_point_A : (ℝ × ℝ) := (2, 0)

-- Defining the general equation of line L under the given condition
def line_conditions (α : ℝ) (l : ℝ × ℝ → Prop) :=
(∃ x y, l (x, y) ∧ (y = (sqrt 2 / 2) * (x - 2) ∨ y = -(sqrt 2 / 2) * (x - 2)))

-- Lean theorem statement
theorem proof_problem 
  (α : ℝ)
  (C : ℝ × ℝ → Prop := λ (p_θ : ℝ × ℝ), polar_curve p_θ.1 p_θ.2)
  (L : ℝ × ℝ → Prop := λ (xy : ℝ × ℝ), ∃ t, xy = parametric_line α t)
  (P Q : ℝ × ℝ)
  (A : ℝ × ℝ := fixed_point_A)
  (h_curve_cartesian : ∀ (x y : ℝ), C (sqrt (x^2 + y^2), atan2 y x) → cartesian_curve x y)
  (h_intersect : ∃ t1 t2, L (parametric_line α t1) ∧ L (parametric_line α t2) ∧ 
                           P = parametric_line α t1 ∧ Q = parametric_line α t2)
  (h_distance : abs (sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) * 
                abs (sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2)) = 6)
: ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), L (x, y) → (y = f x) ∧ line_conditions α L ∧
  (y = (sqrt 2 / 2) * (x - 2) ∨ y = -(sqrt 2 / 2) * (x - 2)).
sorry

end proof_problem_l62_62306


namespace problem1_solution_problem2_solution_l62_62982

-- Define the conditions for Problem 1
variables {m n : ℚ}
axiom problem1_condition : (m - 3) * (√6) + (n - 3) = 0

-- Proof of the Problem 1 statement
theorem problem1_solution (h : problem1_condition) : real.sqrt (m * n) = 3 ∨ real.sqrt (m * n) = -3 :=
sorry

-- Define the conditions for Problem 2
variables {m n x : ℚ}
axiom problem2_condition1 : (2 + (√3)) * m - (1 - (√3)) * n = 5
axiom problem2_condition2 : m * m = x
axiom problem2_condition3 : n * n = x

-- Proof of the Problem 2 statement
theorem problem2_solution (h1 : problem2_condition1) (h2 : problem2_condition2) (h3 : problem2_condition3) : x = 25 / 9 :=
sorry

end problem1_solution_problem2_solution_l62_62982


namespace solve_a_pow4_plus_a_inv4_l62_62554

variable (a : ℝ) [invertible a]

theorem solve_a_pow4_plus_a_inv4
    (h : 5 = a + ⅟a) : a^4 + ⅟(a^4) = 527 := 
sorry

end solve_a_pow4_plus_a_inv4_l62_62554


namespace probability_of_both_making_basket_l62_62989

noncomputable def P : Set ℕ → ℚ :=
  sorry

def A : Set ℕ := sorry
def B : Set ℕ := sorry

axiom prob_A : P A = 2 / 5
axiom prob_B : P B = 1 / 2
axiom independent : P (A ∩ B) = P A * P B

theorem probability_of_both_making_basket :
  P (A ∩ B) = 1 / 5 :=
by
  rw [independent, prob_A, prob_B]
  norm_num

end probability_of_both_making_basket_l62_62989


namespace range_of_positive_integers_l62_62524

theorem range_of_positive_integers :
  let D := (list.range' (-4) 12)
  let positive_integers := D.filter (λ x, x > 0)
  (positive_integers.last sorry - positive_integers.head sorry) = 6 := 
by
  let D := (list.range' (-4) 12)
  let positive_integers := D.filter (λ x, x > 0)
  sorry

end range_of_positive_integers_l62_62524


namespace solve_for_b_l62_62319

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end solve_for_b_l62_62319


namespace food_consumption_reduction_l62_62780

noncomputable def reduction_factor (n p : ℝ) : ℝ :=
  (n * p) / ((n - 0.05 * n) * (p + 0.2 * p))

theorem food_consumption_reduction (n p : ℝ) (h : n > 0 ∧ p > 0) :
  (1 - reduction_factor n p) * 100 = 12.28 := by
  sorry

end food_consumption_reduction_l62_62780


namespace even_odd_array_equality_l62_62339

theorem even_odd_array_equality (n : ℕ) (a : List ℕ) (h₁ : a.length = n) (h₂ : ∑ (i : ℕ) in (Finset.range n), (i + 1) * a.nth_le i sorry = 1979) :
    (∃ m, 2 * m = n) → (∃ m, 2 * m + 1 = n) → (∀ n1 n2, (n1 % 2 = 0 ∧ n2 % 2 = 1) → a.length = n1 ↔ a.length = n2) :=
sorry

end even_odd_array_equality_l62_62339


namespace possible_N_values_l62_62978

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62978


namespace sum_of_four_consecutive_integers_divisible_by_two_l62_62156

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l62_62156


namespace range_r_l62_62299

noncomputable def r (x : ℝ) : ℝ := x^6 + x^4 + 4 * x^2 + 4

theorem range_r : ∀ y ∈ Set.range (λ x : ℝ, r x), y ≥ 16 ∧ ∃ x ≥ 0, r x = y := 
by
  sorry

end range_r_l62_62299


namespace caravan_humps_l62_62972

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62972


namespace prime_factors_of_four_consecutive_integers_sum_l62_62150

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l62_62150


namespace even_function_a_zero_l62_62423

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62423


namespace decreasing_function_inequality_l62_62363

def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then (a - 3) * x + 5 else 3 * a / x

theorem decreasing_function_inequality (a : ℝ) :
  (0 < a ∧ a ≤ 1) ↔
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → (x1 - x2) * (f a x1 - f a x2) < 0 :=
by {
  sorry
}

end decreasing_function_inequality_l62_62363


namespace xiaohua_distance_rounds_l62_62656

def length := 5
def width := 3
def perimeter (a b : ℕ) := (a + b) * 2
def total_distance (perimeter : ℕ) (laps : ℕ) := perimeter * laps

theorem xiaohua_distance_rounds :
  total_distance (perimeter length width) 3 = 30 :=
by sorry

end xiaohua_distance_rounds_l62_62656


namespace simplify_expression_l62_62877

theorem simplify_expression (x : ℝ) :
  4*x^3 + 5*x + 6*x^2 + 10 - (3 - 6*x^2 - 4*x^3 + 2*x) = 8*x^3 + 12*x^2 + 3*x + 7 :=
by
  sorry

end simplify_expression_l62_62877


namespace caravan_humps_l62_62967

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62967


namespace find_common_ratio_l62_62489

noncomputable def geometric_sequence := seq ℕ ℝ

variables {a : ℕ → ℝ} (S : ℕ → ℝ) (q : ℝ)

axiom a3_cond : a 3 = 2 * S 2 + 1
axiom a4_cond : a 4 = 2 * S 3 + 1
axiom S3_def : S 3 = a 1 + a 2 + a 3
axiom S2_def : S 2 = a 1 + a 2

theorem find_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (a 3 = 2 * S 2 + 1) → (a 4 = 2 * S 3 + 1) → 
  (S 3 = a 1 + a 2 + a 3) → (S 2 = a 1 + a 2) → 
  q = 3 :=
by
  intros a3_cond a4_cond S3_def S2_def
  sorry

end find_common_ratio_l62_62489


namespace average_income_independent_of_distribution_l62_62466

namespace AverageIncomeProof

variables (A E : ℝ) (total_employees : ℕ) (h : total_employees = 10)

/-- The average income in December does not depend on the distribution method -/
theorem average_income_independent_of_distribution : 
  (∃ (income : ℝ), income = (A + E) / total_employees) :=
by
  have h1 : total_employees = 10 := h
  exists (A + E) / total_employees
  sorry

end AverageIncomeProof

end average_income_independent_of_distribution_l62_62466


namespace generalized_ptolemy_theorem_l62_62517

variable (A B C D : Point)
variable (α β γ δ : Circle)
variable (ABCD_cirumscribed : isCyclicQuadrilateral A B C D)
variable (α_tangent_to_ABCD_at_A : TangentToCircle α ABCD_cirumscribed A)
variable (β_tangent_to_ABCD_at_B : TangentToCircle β ABCD_cirumscribed B)
variable (γ_tangent_to_ABCD_at_C : TangentToCircle γ ABCD_cirumscribed C)
variable (δ_tangent_to_ABCD_at_D : TangentToCircle δ ABCD_cirumscribed D)
variable (tαβ tγδ tβγ tδα tαγ tβδ : ℝ)
variable (tαβ_def : isCommonTangentSegment tαβ α β)
variable (tγδ_def : isCommonTangentSegment tγδ γ δ)
variable (tβγ_def : isCommonTangentSegment tβγ β γ)
variable (tδα_def : isCommonTangentSegment tδα δ α)
variable (tαγ_def : isCommonTangentSegment tαγ α γ)
variable (tβδ_def : isCommonTangentSegment tβδ β δ)

theorem generalized_ptolemy_theorem :
  tαβ * tγδ + tβγ * tδα = tαγ * tβδ :=
sorry

end generalized_ptolemy_theorem_l62_62517


namespace statement_I_true_statement_II_false_statement_III_false_correct_statement_l62_62715

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem statement_I_true (x : ℝ) : floor (x + 1) = floor x + 1 :=
by sorry

theorem statement_II_false (x y : ℝ) : floor (x + y) ≠ floor x + floor y :=
by sorry

theorem statement_III_false (x y : ℝ) : floor (x * y) ≠ floor x * floor y :=
by sorry

theorem correct_statement : 
  (∀ x, floor (x + 1) = floor x + 1) ∧ 
  (∀ x y, floor (x + y) ≠ floor x + floor y) ∧ 
  (∀ x y, floor (x * y) ≠ floor x * floor y) :=
by 
  exact ⟨statement_I_true, statement_II_false, statement_III_false⟩

end statement_I_true_statement_II_false_statement_III_false_correct_statement_l62_62715


namespace eccentricity_of_hyperbola_equation_of_hyperbola_l62_62729

noncomputable def hyperbola_eccentricity (a b : ℝ) (theta : ℝ) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  in c / a

theorem eccentricity_of_hyperbola (a b : ℝ) (theta : ℝ) (ha : a = 2 / real.sqrt 3)
  (hb : b = 2) (h : theta = real.pi / 3) :
  hyperbola_eccentricity a b theta = 2 :=
by
  sorry

noncomputable def hyperbola_equation_lhs (x b a : ℝ) : ℝ :=
  (x^2) / (a^2 / (4 / 3)) - (x^2) / b^2

theorem equation_of_hyperbola (a b : ℝ) (ha : a = 2 / real.sqrt 3)
  (hb : b = 2) :
  (λ x y : ℝ, hyperbola_equation_lhs x b a - y^2 = 1) :=
by
  sorry

end eccentricity_of_hyperbola_equation_of_hyperbola_l62_62729


namespace half_sum_same_parity_same_color_same_coloring_possible_iff_N_odd_l62_62211

variables (N : ℕ) (color : ℕ → ℕ)
           (infinite_colors : ∀ k : ℕ, ∃ x : ℕ, color x = k)
           (parity_color_condition : ∀ {a b : ℕ}, color a = color b → (a + b) % 2 = 0 → color ((a + b) / 2) = color a)

-- Part (a) Statement
theorem half_sum_same_parity_same_color_same (a b : ℕ) (h_color : color a = color b) (h_parity : (a + b) % 2 = 0) : 
  color ((a + b) / 2) = color a :=
parity_color_condition h_color h_parity

-- Part (b) Statement
theorem coloring_possible_iff_N_odd :
  (∀ color : ℕ → ℕ, infinite_colors → ∀ {a b : ℕ}, color a = color b → (a + b) % 2 = 0 → color ((a + b) / 2) = color a) ↔ (N % 2 = 1) :=
sorry

end half_sum_same_parity_same_color_same_coloring_possible_iff_N_odd_l62_62211


namespace holders_inequality_bad_students_inequality_l62_62212

-- Define Hölder's inequality in Lean 4
theorem holders_inequality 
  (n : ℕ) (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (hpq : 1/p + 1/q = 1)
  (a b : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i) :
  (∑ i, a i ^ p) ^ (1 / p) * (∑ i, b i ^ q) ^ (1 / q) ≥ ∑ i, a i * b i :=
sorry

-- Define the "bad students' inequality" in Lean 4
theorem bad_students_inequality 
  (n : ℕ) (r : ℝ) (hr : 0 < r)
  (x y : Fin n → ℝ) (hx : ∀ i, 0 < x i) (hy : ∀ i, 0 < y i) :
  ∑ i, x i ^ (r+1) / y i ^ r ≥ (∑ i, x i) ^ (r+1) / (∑ i, y i) ^ r :=
sorry

end holders_inequality_bad_students_inequality_l62_62212


namespace sin_double_angle_l62_62381

theorem sin_double_angle (θ : ℝ) (h1 : θ ∈ Icc (3 * Real.pi / 2) (2 * Real.pi)) (h2 : Real.cos θ = 4 / 5) : 
  Real.sin (2 * θ) = -24 / 25 := 
sorry

end sin_double_angle_l62_62381


namespace prism_volume_l62_62559

theorem prism_volume 
  (a b c : ℝ)
  (h1 : a * b = 30)
  (h2 : a * c = 50)
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 :=
sorry

end prism_volume_l62_62559


namespace joseph_savings_ratio_l62_62176

theorem joseph_savings_ratio
    (thomas_monthly_savings : ℕ)
    (thomas_years_saving : ℕ)
    (total_savings : ℕ)
    (joseph_total_savings_is_total_minus_thomas : total_savings = thomas_monthly_savings * 12 * thomas_years_saving + (total_savings - thomas_monthly_savings * 12 * thomas_years_saving))
    (thomas_saves_each_month : thomas_monthly_savings = 40)
    (years_saving : thomas_years_saving = 6)
    (total_amount : total_savings = 4608) :
    (total_savings - thomas_monthly_savings * 12 * thomas_years_saving) / (12 * thomas_years_saving) / thomas_monthly_savings = 3 / 5 :=
by
  sorry

end joseph_savings_ratio_l62_62176


namespace volume_formula_l62_62684

noncomputable def volume_set_of_points (length width height : ℕ) (unit : ℝ) : ℝ := 
  let original_volume := length * width * height
  let extended_volume :=  
    2 * (unit * length * width) + 
    2 * (unit * length * height) +  
    2 * (unit * width * height)
  let cylinder_volume := 
    4 * (unit * π * length / 4) + 
    4 * (unit * π * width / 4) + 
    4 * (unit * π * height / 4)
  let octant_volume := 
    8 * (1 / 8 * 4 / 3 * π * unit^3)
  original_volume + extended_volume + cylinder_volume + octant_volume

theorem volume_formula (m n p : ℕ) (unit : ℝ) : 
  m = 424 → n = 58 → p = 3 →
  volume_set_of_points 5 6 7 unit = (424 + 58 * π) / 3 ∧ Nat.coprime n p → 
  m + n + p = 485 :=
by
  assume h1 : m = 424
  assume h2 : n = 58
  assume h3 : p = 3
  assume h_vol : volume_set_of_points 5 6 7 1 = (424 + 58 * π) / 3 ∧ Nat.coprime n p
  sorry

end volume_formula_l62_62684


namespace correct_propositions_l62_62519

variable (A : Set ℝ)
variable (oplus : ℝ → ℝ → ℝ)

def condition_a1 : Prop := ∀ a b : ℝ, a ∈ A → b ∈ A → (oplus a b) ∈ A
def condition_a2 : Prop := ∀ a : ℝ, a ∈ A → (oplus a a) = 0
def condition_a3 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus (oplus a b) c) = (oplus a c) + (oplus b c) + c

def proposition_1 : Prop := 0 ∈ A
def proposition_2 : Prop := (1 ∈ A) → (oplus (oplus 1 1) 1) = 0
def proposition_3 : Prop := ∀ a : ℝ, a ∈ A → (oplus a 0) = a → a = 0
def proposition_4 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus a 0) = a → (oplus a b) = (oplus c b) → a = c

theorem correct_propositions 
  (h1 : condition_a1 A oplus) 
  (h2 : condition_a2 A oplus)
  (h3 : condition_a3 A oplus) : 
  (proposition_1 A) ∧ (¬proposition_2 A oplus) ∧ (proposition_3 A oplus) ∧ (proposition_4 A oplus) := by
  sorry

end correct_propositions_l62_62519


namespace number_of_members_l62_62230

variables
  (n : ℕ) -- number of members in the team
  (T : ℕ) -- total age of the team
  (average_age : ℝ) (wicket_keeper_age : ℝ) (excluded_member_age : ℝ) 

-- Given: The average age of the cricket team is 25 years.
axiom average_age_condition : average_age = 25

-- Given: The wicket keeper is 3 years older than the average age of the team.
axiom wicket_keeper_condition : wicket_keeper_age = average_age + 3

-- Given: When the ages of the wicket keeper and one other member are excluded,
-- the average age of the remaining players is 1 year less than the average age of the whole team.
axiom remaining_players_condition : 
  ((n - 2) * (average_age - 1)) = (T - wicket_keeper_age - excluded_member_age)

-- Given: The other excluded member is exactly average_age.
axiom excluded_member_condition : excluded_member_age = average_age

-- Total age of team with known average age relationship
axiom total_age_condition : T = n * average_age

-- Prove the number of members in the team equals 5.
theorem number_of_members (h1 : total_age_condition) (h2 : wicket_keeper_condition)
    (h3 : excluded_member_condition) (h4 : remaining_players_condition) : 
  n = 5 :=
by
  -- skip the proof for this theorem
  sorry

end number_of_members_l62_62230


namespace sum_of_x_satisfies_equation_l62_62060

theorem sum_of_x_satisfies_equation :
  let g (x : ℝ) := 3 * x - 2
  let g_inv (x : ℝ) := (x + 2) / 3
  (solver : ∀ (x : ℝ), g_inv x = g (1 / x) → x = -9 ∨ x = 1)
  ∑ x in {x : ℝ | g_inv x = g (1 / x)}, x = -8 :=
by
  let g := λ (x : ℝ), 3 * x - 2
  let g_inv := λ (x : ℝ), (x + 2) / 3
  have solver : ∀ (x : ℝ), g_inv x = g (1 / x) → x = -9 ∨ x = 1 := sorry
  have : ∑ x in {x | g_inv x = g (1 / x)}, x = -8 := sorry
  exact this

end sum_of_x_satisfies_equation_l62_62060


namespace gcf_72_108_l62_62194

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62194


namespace negation_of_proposition_l62_62900

theorem negation_of_proposition (x y : ℝ): (x + y > 0 → x > 0 ∧ y > 0) ↔ ¬ ((x + y ≤ 0) → (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end negation_of_proposition_l62_62900


namespace circle_standard_eq_l62_62727

theorem circle_standard_eq :
  ∃ (x₀ y₀ r : ℝ), (l : ℝ) (h₀ : x - 2 * y - 1 = 0) (h₁ : (x₀ - 2 * y₀ - 1 = 0)) (h₂ : (x₀ = 7/4) ∧ (y₀ = 3/8) ∧ (r^2 = 205/64)),
  (x - x₀)^2 + (y - y₀)^2 = r^2 :=
by sorry

end circle_standard_eq_l62_62727


namespace find_other_vertices_l62_62561

theorem find_other_vertices
  (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (S : ℝ × ℝ) (M : ℝ × ℝ)
  (hA : A = (7, 3))
  (hS : S = (5, -5 / 3))
  (hM : M = (3, -1))
  (h_centroid : S = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) 
  (h_orthocenter : ∀ u v : ℝ × ℝ, u ≠ v → u - v = (4, 4) → (u - v) • (C - B) = 0) :
  B = (1, -1) ∧ C = (7, -7) :=
sorry

end find_other_vertices_l62_62561


namespace part_a_part_b_l62_62532

variables {ABC : Type} [triangle ABC]
variables {A1 B1 C1 : ABC → ABC → ABC} -- Points on the sides BC, CA, and AB respectively
variables {area : ABC → ℝ} -- Function that returns the area of a triangle

theorem part_a :
  ∃ (AB1C1 A1BC1 A1B1C : ABC),
  area AB1C1 ≤ (area ABC) / 4 ∨ area A1BC1 ≤ (area ABC) / 4 ∨ area A1B1C ≤ (area ABC) / 4 :=
sorry

theorem part_b :
  ∃ (AB1C1 A1BC1 A1B1C : ABC),
  area AB1C1 ≤ area A1B1C1 ∨ area A1BC1 ≤ area A1B1C1 ∨ area A1B1C ≤ area A1B1C1 :=
sorry

end part_a_part_b_l62_62532


namespace painters_workdays_l62_62810

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l62_62810


namespace caravan_humps_l62_62966

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62966


namespace gcd_72_108_l62_62188

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l62_62188


namespace john_arcade_fraction_l62_62821

-- Define John's weekly allowance
def weekly_allowance : ℝ := 3.30

-- Define the remaining allowance after spending at the arcade
def remaining_after_arcade (A : ℝ) : ℝ := weekly_allowance - A

-- Define the amount spent at the toy store
def spent_at_toy_store (A : ℝ) : ℝ := (1/3) * remaining_after_arcade A

-- Define the remaining allowance after spending at the toy store
def remaining_after_toy_store (A : ℝ) : ℝ := remaining_after_arcade A - spent_at_toy_store A

-- Define the remaining allowance after spending at the candy store
def remaining_after_candy_store (A : ℝ) : ℝ := remaining_after_toy_store A - 0.88

-- Condition: After spending at the candy store, he had $0 left.
def final_condition (A : ℝ) : Prop := remaining_after_candy_store A = 0

-- Theorem: Fraction of allowance spent at the arcade
theorem john_arcade_fraction (A : ℝ) (h : final_condition A) : (A / weekly_allowance) = 3 / 5 :=
sorry

end john_arcade_fraction_l62_62821


namespace midpoint_sum_l62_62830

-- Define the points
def P : ℝ × ℝ := (2, 7)
def D : ℝ × ℝ := (4, 3)
noncomputable def Q : ℝ × ℝ := (6, -1)

-- Prove that x + y = 5 given the conditions
theorem midpoint_sum (x y : ℝ) (h1 : (2 + x) / 2 = 4) (h2 : (7 + y) / 2 = 3) : x + y = 5 :=
  by
    have hx : x = 6 := by
      linarith
    have hy : y = -1 := by
      linarith
    rw [hx, hy]
    linarith

end midpoint_sum_l62_62830


namespace hallTheorem_l62_62139

variable {V : Type*} [Fintype V]   -- Type for vertices, assuming finite graph
variable (G : Graph V)             -- Graph G with vertices of type V
variable (A B : Finset V)          -- The partite sets A and B
variable [DecidablePred (∈ A)]     -- Making membership decidable
variable [DecidablePred (∈ B)]

-- Condition: G is bipartite with partite sets A and B
axiom bipartite : G.isBipartite A B

-- Condition: For all subsets S of A, |N(S)| ≥ |S|
axiom hallCondition : ∀ S ⊆ A, (G.neighbors S).card ≥ S.card

-- Theorem: There exists a matching in G that saturates A if and only if Hall's condition holds
theorem hallTheorem : (∃ M : Finset (Sym2 V), G.isMatching M ∧ ∀ a ∈ A, ∃ b ∈ B, ⟦(a, b)⟧ ∈ M) ↔ 
                       (∀ S ⊆ A, (G.neighbors S).card ≥ S.card) :=
begin
  sorry
end

end hallTheorem_l62_62139


namespace fourth_term_correct_l62_62297

noncomputable def fourth_term_of_expansion (a x : ℝ) : ℝ :=
  let term := (a / real.sqrt x - real.sqrt x / a^2)^8
  in -56 / (a * x)

theorem fourth_term_correct (a x : ℝ) : fourth_term_of_expansion a x = -56 / (a * x) :=
  sorry

end fourth_term_correct_l62_62297


namespace simplest_radical_form_l62_62664

def is_simplest_radical_form (x : ℝ) : Prop :=
  (∀ d : ℝ, (∃ n : ℕ, d * d = x) → (x = d ^ 2 ∨ ¬ (d ≠ 0 ∧ x = d ^ 2 * n ^ 2))) ∧
  (x ≠ 0 → ∀ d : ℕ, ¬ (d^2 * n ^ 2 = x))

theorem simplest_radical_form (x y z w : ℝ) (hx : x = 7) (hy : y = 9) (hz : z = 20) (hw : w = 1/3) :
  is_simplest_radical_form (sqrt x) ∧ ¬ is_simplest_radical_form (sqrt y) ∧
  ¬ is_simplest_radical_form (sqrt z) ∧ ¬ is_simplest_radical_form (sqrt w) :=
by
  sorry

end simplest_radical_form_l62_62664


namespace locusOfPointInIsoscelesTriangle_l62_62365

variable (A B C P : Point)
variable {AB AC BC : Length}
variable (d_BC d_AB d_AC : P → Length)
variable (geo_mean : Length → Length → Length)

noncomputable def isoscelesTriangle (A B C : Point) : Prop :=
  AB = AC

noncomputable def locusCondition (P : Point) (d_BC d_AB d_AC : P → Length) :=
  d_BC P = geo_mean (d_AB P) (d_AC P)

theorem locusOfPointInIsoscelesTriangle :
  (isoscelesTriangle A B C) →
  (locusCondition P d_BC d_AB d_AC) →
  (∃ (α : Circle), tangent α A B ∧ tangent α A C ∧ P ∈ α ∧ P ∉ {B, C}) :=
sorry

end locusOfPointInIsoscelesTriangle_l62_62365


namespace circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l62_62610

-- Definitions encapsulated in theorems with conditions and desired results
theorem circle_touch_externally {d R r : ℝ} (h1 : d = 10) (h2 : R = 8) (h3 : r = 2) : 
  d = R + r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_one_inside_other_without_touching {d R r : ℝ} (h1 : d = 4) (h2 : R = 17) (h3 : r = 11) : 
  d < R - r :=
by 
  rw [h1, h2, h3]
  sorry

theorem circle_completely_outside {d R r : ℝ} (h1 : d = 12) (h2 : R = 5) (h3 : r = 3) : 
  d > R + r :=
by 
  rw [h1, h2, h3]
  sorry

end circle_touch_externally_circle_one_inside_other_without_touching_circle_completely_outside_l62_62610


namespace equilateral_triangle_angle_right_l62_62178

theorem equilateral_triangle_angle_right 
  (A B C E F K P: Type*)
  [equilateral_triangle A B C]
  (h1 : E ∈ segment A B)
  (h2 : F ∈ segment A C)
  (h3 : P ∈ segment E F)
  (h4 : midpoint P E F)
  (h5 : AE = CF)
  (h6 : AE = BK)
  (h7 : K ∈ extension A B) 
  : angle K P C = 90 := by
  sorry

end equilateral_triangle_angle_right_l62_62178


namespace g_range_l62_62523

variable {R : Type*} [LinearOrderedRing R]

-- Let y = f(x) be a function defined on R with a period of 1
def periodic (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f x

-- If g(x) = f(x) + 2x
def g (f : R → R) (x : R) : R := f x + 2 * x

-- If the range of g(x) on the interval [1,2] is [-1,5]
def rangeCondition (f : R → R) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → -1 ≤ g f x ∧ g f x ≤ 5

-- Then the range of the function g(x) on the interval [-2020,2020] is [-4043,4041]
theorem g_range (f : R → R) 
  (hf_periodic : periodic f) 
  (hf_range : rangeCondition f) : 
  ∀ x, -2020 ≤ x ∧ x ≤ 2020 → -4043 ≤ g f x ∧ g f x ≤ 4041 :=
sorry

end g_range_l62_62523


namespace max_points_right_triangle_l62_62195

theorem max_points_right_triangle (n : ℕ) :
  (∀ (pts : Fin n → ℝ × ℝ), ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    let p1 := pts i
    let p2 := pts j
    let p3 := pts k
    let a := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
    let b := (p3.1 - p2.1)^2 + (p3.2 - p2.2)^2
    let c := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
    a + b = c ∨ b + c = a ∨ c + a = b) →
  n ≤ 4 :=
sorry

end max_points_right_triangle_l62_62195


namespace largest_number_is_34_l62_62137

-- Definitions based on the conditions
variables {a b c : ℕ}

-- Conditions in the problem
def conditions : Prop :=
  a < b ∧ b < c ∧ a + b + c = 80 ∧ c = b + 9 ∧ b = a + 4 ∧ a * b = 525

-- The statement we need to prove
theorem largest_number_is_34 (h : conditions) : c = 34 :=
sorry

end largest_number_is_34_l62_62137


namespace smallest_positive_multiple_l62_62711

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l62_62711


namespace octal_to_decimal_equiv_l62_62658

-- Definitions for the octal number 724
def d0 := 4
def d1 := 2
def d2 := 7

-- Definition for the base
def base := 8

-- Calculation of the decimal equivalent
def calc_decimal : ℕ :=
  d0 * base^0 + d1 * base^1 + d2 * base^2

-- The proof statement
theorem octal_to_decimal_equiv : calc_decimal = 468 := by
  sorry

end octal_to_decimal_equiv_l62_62658


namespace valid_number_of_two_humped_camels_within_range_l62_62959

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62959


namespace odd_vertices_contradiction_l62_62689

theorem odd_vertices_contradiction (B P : ℕ) (h1 : ∀ v, ∃ (e₁ e₂ e₃ : ℕ), true) (h2 : B % 2 = 1)
: false := by
sorry

end odd_vertices_contradiction_l62_62689


namespace possible_N_values_l62_62954

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62954


namespace triple_composition_even_l62_62059

-- Definition of an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g(x) = g(-x)

-- Theorem: If g is an even function, then g(g(g(x))) is also an even function
theorem triple_composition_even {g : ℝ → ℝ} (h_even : even_function g) : 
  even_function (λ x, g(g(g(x)))) :=
by 
  sorry

end triple_composition_even_l62_62059


namespace sqrt_x_minus_3_defined_l62_62414

theorem sqrt_x_minus_3_defined (x : ℝ) (h : ∃ y : ℝ, ∀ x, y = real.sqrt (x - 3)) : x ≥ 3 :=
sorry

end sqrt_x_minus_3_defined_l62_62414


namespace find_set_T_l62_62876

namespace MathProof 

theorem find_set_T (S : Finset ℕ) (hS : ∀ x ∈ S, x > 0) :
  ∃ T : Finset ℕ, S ⊆ T ∧ ∀ x ∈ T, x ∣ (T.sum id) :=
by
  sorry

end MathProof 

end find_set_T_l62_62876


namespace arithmetic_sequence_product_l62_62836

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_product (a_1 a_6 a_7 a_4 a_9 : ℝ) (d : ℝ) :
  a_1 = 2 →
  a_6 = a_1 + 5 * d →
  a_7 = a_1 + 6 * d →
  a_6 * a_7 = 15 →
  a_4 = a_1 + 3 * d →
  a_9 = a_1 + 8 * d →
  a_4 * a_9 = 234 / 25 :=
sorry

end arithmetic_sequence_product_l62_62836


namespace speed_in_still_water_l62_62237

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 24 := by
  sorry

end speed_in_still_water_l62_62237


namespace books_remaining_in_special_collection_l62_62243

theorem books_remaining_in_special_collection
  (initial_books : ℕ)
  (loaned_books : ℕ)
  (returned_percentage : ℕ)
  (initial_books_eq : initial_books = 75)
  (loaned_books_eq : loaned_books = 45)
  (returned_percentage_eq : returned_percentage = 80) :
  ∃ final_books : ℕ, final_books = initial_books - (loaned_books - (loaned_books * returned_percentage / 100)) ∧ final_books = 66 :=
by
  sorry

end books_remaining_in_special_collection_l62_62243


namespace sum_of_reciprocals_B_l62_62505

def B : Set ℕ := {n | ∀ p, prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 7}

theorem sum_of_reciprocals_B : ∑' (n : ℕ) in B, (1 : ℚ) / n = 7 / 2 :=
sorry

end sum_of_reciprocals_B_l62_62505


namespace fish_stock_l62_62096

theorem fish_stock {
  initial_stock fish_sold new_stock : ℕ,
  spoil_fraction : ℚ,
  initial_stock = 200 → fish_sold = 50 → new_stock = 200 →
  spoil_fraction = 1 / 3 →
  ∃ (final_stock : ℕ), final_stock = initial_stock - fish_sold - ⌊(initial_stock - fish_sold) * spoil_fraction⌋ + new_stock ∧ final_stock = 300 :=
begin
  intros h_initial h_sold h_new h_spoil,
  use initial_stock - fish_sold - (⌊(initial_stock - fish_sold) * spoil_fraction⌋ : ℕ) + new_stock,
  split,
  {
    rw [h_initial, h_sold, h_new, h_spoil],
    norm_num,
  },
  exact ⟨300⟩,
end

end fish_stock_l62_62096


namespace proposition_1_proposition_4_l62_62371

variables (m l : Line)
variables (α β : Plane)

-- Conditions
axiom m_perp_α : Perpendicular m α
axiom l_in_β : l ∈ β

-- Propositions
theorem proposition_1 (h_parallel : α ∥ β) : Perpendicular m l := sorry
theorem proposition_4 (h_parallel : Parallel m l) : Perpendicular α β := sorry

end proposition_1_proposition_4_l62_62371


namespace tan_a_tan_b_calc_l62_62787

noncomputable def triangle_isosceles (A B C : Type) [triangle A B C] : Prop :=
AB = AC

noncomputable def altitude_hits_point (D : Type) (C : Type) (line_segment : Type) : Prop :=
altitude_from C AB = line_segment.from_point D 

noncomputable def orthocenter_divides_altitude (H D : Type) (line_segment : Type) (length_HD : ℕ) (length_HC : ℕ) : Prop :=
orthocenter H = true ∧ line_segment.from_point D = HD ∧ length_HD = 7 ∧ length_HC = 9 ∧ line_segment.from_point H = length_HD + length_HC

theorem tan_a_tan_b_calc
  (ABC : Type) [triangle_isosceles ABC]
  (D : Type) [altitude_hits_point D ABC]
  (H : Type) [orthocenter_divides_altitude H D ABC 7 9] : 
  tan A * tan B = 225 / 49 :=
by
  sorry

end tan_a_tan_b_calc_l62_62787


namespace length_of_grass_field_l62_62247

theorem length_of_grass_field (L : ℝ) (L + 2 * 3.5 = total_length) (55 + 2 * 3.5 = total_width) 
  (1918 : ℝ) : 1918 = (total_length * total_width) - (L * 55) := 
begin
  -- Given:
  -- Width of the grass field: 55 meters
  -- Width of the path: 3.5 meters
  -- Area of the path: 1918 square meters

  -- Let L be the length of the grass field
  -- The area of the entire field (including the path) is:
  -- (L + 7) * (55 + 7) - L * 55 = 1918
  -- We need to prove that the length L is equal to 212 meters.

  have h1 : total_length = L + 7, from sorry,
  have h2 : total_width = 62, from sorry,
  have h3 : 1918 = total_length * total_width - L * 55, from sorry,

  -- Consequence
  have L = 212 from sorry,
  -- Hence proved
end

end length_of_grass_field_l62_62247


namespace min_value_of_F_l62_62520

def f (x : ℝ) : ℝ := Real.log2 (x^2 + 1)
def g (x : ℝ) : ℝ := Real.log2 (|x| + 7)

def F (x : ℝ) : ℝ :=
  if f x ≥ g x then f x else g x

theorem min_value_of_F :
  ∃ y : ℝ, (∀ x : ℝ, F x ≥ y) ∧ y = Real.log2 7 :=
sorry

end min_value_of_F_l62_62520


namespace auston_height_in_meters_l62_62263

-- Define given conditions
def height_in_inches : ℝ := 72
def inches_to_cm : ℝ := 2.54
def cm_to_meter : ℝ := 1 / 100

-- The target height in meters
def height_in_meters : ℝ := 1.83

-- Main statement to prove
theorem auston_height_in_meters :
  (height_in_inches * inches_to_cm * cm_to_meter) = height_in_meters :=
by
  -- proof is omitted
  sorry

end auston_height_in_meters_l62_62263


namespace ratio_of_B_to_C_l62_62879

-- Definitions based on conditions
def A := 40
def C := A + 20
def total := 220
def B := total - A - C

-- Theorem statement
theorem ratio_of_B_to_C : B / C = 2 :=
by
  -- Placeholder for proof
  sorry

end ratio_of_B_to_C_l62_62879


namespace domain_of_function_l62_62464

-- Define the function y = 1 / (x - 1)
def f (x : ℝ) : ℝ := 1 / (x - 1)

-- State the given condition: The range of the function is (-∞, -1) ∪ (1, ∞)
def range_condition (y : ℝ) : Prop := y ∈ set.Iio (-1) ∪ set.Ioi 1

-- State the mathematically equivalent proof problem
theorem domain_of_function :
  (∀ y : ℝ, range_condition y ↔ ∃ x : ℝ, f x = y) →
  ∀ x, (0 < x ∧ x < 1 ∨ 1 < x ∧ x < 2) ↔ ∃ y : ℝ, f x = y :=
sorry

end domain_of_function_l62_62464


namespace will_remaining_balance_l62_62621

theorem will_remaining_balance :
  ∀ (initial_money conversion_fee : ℝ) 
    (exchange_rate : ℝ)
    (sweater_cost tshirt_cost shoes_cost hat_cost socks_cost : ℝ)
    (shoes_refund_percentage : ℝ)
    (discount_percentage sales_tax_percentage : ℝ),
  initial_money = 74 →
  conversion_fee = 2 →
  exchange_rate = 1.5 →
  sweater_cost = 13.5 →
  tshirt_cost = 16.5 →
  shoes_cost = 45 →
  hat_cost = 7.5 →
  socks_cost = 6 →
  shoes_refund_percentage = 0.85 →
  discount_percentage = 0.10 →
  sales_tax_percentage = 0.05 →
  (initial_money - conversion_fee) * exchange_rate -
  ((sweater_cost + tshirt_cost + shoes_cost + hat_cost + socks_cost - shoes_cost * shoes_refund_percentage) *
   (1 - discount_percentage) * (1 + sales_tax_percentage)) /
  exchange_rate = 39.87 :=
by
  intros initial_money conversion_fee exchange_rate
        sweater_cost tshirt_cost shoes_cost hat_cost socks_cost
        shoes_refund_percentage discount_percentage sales_tax_percentage
        h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end will_remaining_balance_l62_62621


namespace trapezoid_perimeter_l62_62796

open Real EuclideanGeometry

/-- In the trapezoid ABCD, the bases AD and BC are 8 and 18 respectively. It is known that
the circumcircle of triangle ABD is tangent to the lines BC and CD. Prove that the perimeter
of the trapezoid is 56. -/
theorem trapezoid_perimeter (A B C D : Point)
  (h1 : dist A D = 8) (h2 : dist B C = 18)
  (h_circum : ∃ O r, Circle O r ∧ Tangent O r A B D B C ∧ Tangent O r A B D C D) :
  dist A B + dist A D + dist B C + dist C D = 56 := 
sorry

end trapezoid_perimeter_l62_62796


namespace yolara_distance_one_fourth_l62_62128

/-- Yolara follows an elliptical path with the star situated at one focus. -/
def yolara_ellipse : Type := sorry

/-- At perigee, Yolara is 3 AU away from the star. -/
def perigee_distance (Y : yolara_ellipse) : ℝ := 3

/-- At apogee, Yolara is 8 AU away from the star. -/
def apogee_distance (Y : yolara_ellipse) : ℝ := 8

/-- Yolara's distance from its star when it is one-fourth the way from perigee to apogee. -/
theorem yolara_distance_one_fourth (Y : yolara_ellipse) :
  let M_distance_from_A := (perigee_distance Y + apogee_distance Y) / 4 in
  yolara_distance_one_fourth Y = 10.75 :=
by {
  sorry
}

end yolara_distance_one_fourth_l62_62128


namespace even_function_a_eq_zero_l62_62439

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62439


namespace possible_values_of_N_count_l62_62945

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62945


namespace paint_one_third_of_square_l62_62132

theorem paint_one_third_of_square (n k : ℕ) (h1 : n = 18) (h2 : k = 6) :
    Nat.choose n k = 18564 :=
by
  rw [h1, h2]
  sorry

end paint_one_third_of_square_l62_62132


namespace ordered_pairs_count_l62_62336

theorem ordered_pairs_count : 
  (∃ bs cs : List ℕ, 
      (List.all bs (λ b => 1 ≤ b ∧ b ≤ 6) ∧ 
       List.all cs (λ c => 1 ≤ c ∧ c ≤ 6) ∧ 
       List.length (List.filter (λ (bc : ℕ × ℕ), 
         (bc.1 ^ 2 - 4 * bc.2 = 0 ∨ bc.2 ^ 2 - 4 * bc.1 = 0) ∧ 
          1 ≤ bc.1 ∧ bc.1 ≤ 6 ∧ 1 ≤ bc.2 ∧ bc.2 ≤ 6) 
         (List.product bs cs))) = 3) :=
sorry

end ordered_pairs_count_l62_62336


namespace albert_earnings_l62_62456

theorem albert_earnings (E E_final : ℝ) : 
  (0.90 * (E * 1.14) = 678) → 
  (E_final = 0.90 * (E * 1.15 * 1.20)) → 
  E_final = 819.72 :=
by
  sorry

end albert_earnings_l62_62456


namespace minimum_numbers_l62_62848

theorem minimum_numbers (N : ℕ) (S : finset ℕ) (H : S.card = N) 
  (C1 : ∀ x ∈ S, x > 0) 
  (C2 : ∀ n ∈ (finset.range 1000).filter (λ k, k > 0), ∃ x y ∈ S, x > y ∧ x^2 - y^2 = n): 
  N ≥ 252 := 
sorry

end minimum_numbers_l62_62848


namespace locus_of_P_l62_62368

theorem locus_of_P (A B C P L M N : Point) (h_triangle : isosceles_triangle A B C) 
  (h_P_in_triangle : inside_triangle P A B C) 
  (h_dist_eq_geom_mean : dist P (line BC) = real.sqrt (dist P (line AB) * dist P (line AC))) : 
  ∃ (Γ : Circle), is_arc_of_circle P B C Γ :=
sorry

end locus_of_P_l62_62368


namespace cube_root_of_27_l62_62111

theorem cube_root_of_27 : 
  ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
begin
  sorry
end

end cube_root_of_27_l62_62111


namespace bottom_pipe_drain_rate_l62_62534

-- Definitions for the conditions
def capacity : ℕ := 850
def rateA : ℕ := 40
def rateB : ℕ := 30
def cycles : ℕ := 51 / 3
def cycle_time : ℕ := 3 minutes

-- Proof statement
theorem bottom_pipe_drain_rate : 
  ∃ x : ℕ, (17 * (rateA + rateB - x) = capacity) → x = 20 :=
by
  -- Skipping the proof
  sorry

end bottom_pipe_drain_rate_l62_62534


namespace seq_bound_l62_62905

def sequence (x : ℕ → ℝ) : Prop :=
  x 1 = 0.001 ∧ ∀ n : ℕ, n ≥ 1 → x (n + 1) = x n - x n ^ 2

theorem seq_bound (x : ℕ → ℝ) (h : sequence x) : x 1001 < 0.0005 :=
by
  sorry

end seq_bound_l62_62905


namespace lcm_150_294_l62_62994

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end lcm_150_294_l62_62994


namespace triangle_leg_length_l62_62792

theorem triangle_leg_length (perimeter_square : ℝ)
                            (base_triangle : ℝ)
                            (area_equality : ∃ (side_square : ℝ) (height_triangle : ℝ),
                                4 * side_square = perimeter_square ∧
                                side_square * side_square = (1/2) * base_triangle * height_triangle)
                            : ∃ (y : ℝ), y = 22.5 :=
by
  -- Placeholder proof
  sorry

end triangle_leg_length_l62_62792


namespace grasshopper_flea_adjacency_l62_62533

-- We assume that grid cells are indexed by pairs of integers (i.e., positions in ℤ × ℤ)
-- Red cells and white cells are represented as sets of these positions
variable (red_cells : Set (ℤ × ℤ))
variable (white_cells : Set (ℤ × ℤ))

-- We define that the grasshopper can only jump between red cells
def grasshopper_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ red_cells ∧ new_pos ∈ red_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- We define that the flea can only jump between white cells
def flea_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ white_cells ∧ new_pos ∈ white_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- Main theorem to be proved
theorem grasshopper_flea_adjacency (g_start : ℤ × ℤ) (f_start : ℤ × ℤ) :
    g_start ∈ red_cells → f_start ∈ white_cells →
    ∃ g1 g2 g3 f1 f2 f3 : ℤ × ℤ,
    (
      grasshopper_jump red_cells g_start g1 ∧
      grasshopper_jump red_cells g1 g2 ∧
      grasshopper_jump red_cells g2 g3
    ) ∧ (
      flea_jump white_cells f_start f1 ∧
      flea_jump white_cells f1 f2 ∧
      flea_jump white_cells f2 f3
    ) ∧
    (abs (g3.1 - f3.1) + abs (g3.2 - f3.2) = 1) :=
  sorry

end grasshopper_flea_adjacency_l62_62533


namespace desired_average_sale_l62_62646

theorem desired_average_sale
  (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7991) :
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 7000 :=
by
  sorry

end desired_average_sale_l62_62646


namespace ratio_of_male_democrats_to_total_males_l62_62144

noncomputable def F : ℕ := 135 * 2
noncomputable def M : ℕ := 810 - F
noncomputable def total_democrats : ℕ := 810 / 3
noncomputable def male_democrats : ℕ := total_democrats - 135

theorem ratio_of_male_democrats_to_total_males : male_democrats.to_rat / M.to_rat = (1 : ℚ) / 4 :=
by sorry

end ratio_of_male_democrats_to_total_males_l62_62144


namespace even_function_a_zero_l62_62425

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62425


namespace polynomial_value_at_2_l62_62181

def f (x : ℤ) : ℤ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_value_at_2:
  f 2 = 1538 := by
  sorry

end polynomial_value_at_2_l62_62181


namespace inverse_of_g_l62_62573

variable (X Y : Type) [Group X] [Group Y]
variables (s t u v : X → X) (s_inv t_inv u_inv v_inv : X → X)

-- Assume the functions s, t, u, v are invertible
hypothesis hs : Function.LeftInverse s_inv s ∧ Function.RightInverse s_inv s
hypothesis ht : Function.LeftInverse t_inv t ∧ Function.RightInverse t_inv t
hypothesis hu : Function.LeftInverse u_inv u ∧ Function.RightInverse u_inv u
hypothesis hv : Function.LeftInverse v_inv v ∧ Function.RightInverse v_inv v

-- Define the function g as the composition of t, u, s, and v
def g := t ∘ u ∘ s ∘ v

-- Define the intended inverse function of g
def g_inv := v_inv ∘ s_inv ∘ u_inv ∘ t_inv

-- The theorem to prove
theorem inverse_of_g : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g := by 
  -- proof goes here
  sorry

end inverse_of_g_l62_62573


namespace alice_min_spent_to_win_l62_62254

-- Definition of the problem conditions
def alice_wins (beans : ℕ) : Prop :=
  beans > 2008 ∧ beans % 100 = 42

-- The minimum number of cents spent by Alice to win the game
def min_spent_cents_to_win : ℕ :=
  36

-- Main theorem statement
theorem alice_min_spent_to_win : ∃ cents : ℕ, 
  (∀ beans : ℕ, 
    ( ∀ operations : list (ℕ → ℕ), 
      (∀ f ∈ operations, f beans = 5 * beans ∨ f beans = beans + 1) 
      → alice_wins (operations.foldl (λ acc op, op acc) 0) → cents = min_spent_cents_to_win ) ) :=
sorry

end alice_min_spent_to_win_l62_62254


namespace incenter_circumcenter_common_point_collinear_l62_62983

open EuclideanGeometry

noncomputable def incenter (A B C : Point) : Point :=
  sorry

noncomputable def circumcenter (A B C : Point) : Point :=
  sorry

noncomputable def identical_circles_common_point (A B C K : Point) : Prop :=
  exists (r : ℝ) (O1 O2 O3 : Point),
    (circle O1 r ∧ circle O2 r ∧ circle O3 r) ∧
    (K = O1 ∧ K = O2 ∧ K = O3) ∧
    (is_tangent_to (circle O1 r) (line_through A B) ∧ 
     is_tangent_to (circle O1 r) (line_through A C)) ∧
    (is_tangent_to (circle O2 r) (line_through B A) ∧ 
     is_tangent_to (circle O2 r) (line_through B C)) ∧
    (is_tangent_to (circle O3 r) (line_through C A) ∧ 
     is_tangent_to (circle O3 r) (line_through C B)) 

theorem incenter_circumcenter_common_point_collinear
  (A B C K : Point) :
  identical_circles_common_point A B C K →
  collinear {incenter A B C, circumcenter A B C, K} :=
begin
  sorry
end

end incenter_circumcenter_common_point_collinear_l62_62983


namespace percentage_decrease_l62_62029

theorem percentage_decrease (x y : ℝ) :
  let x' := 0.8 * x
  let y' := 0.7 * y
  let original_expr := x^2 * y^3
  let new_expr := (x')^2 * (y')^3
  let perc_decrease := (original_expr - new_expr) / original_expr * 100
  perc_decrease = 78.048 := by
  sorry

end percentage_decrease_l62_62029


namespace prime_factors_of_four_consecutive_integers_sum_l62_62153

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l62_62153


namespace gcf_72_108_l62_62184

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62184


namespace bug_total_distance_l62_62222

def total_distance (start middle end : ℤ) : ℤ :=
  (middle - start).natAbs + (end - middle).natAbs

theorem bug_total_distance : total_distance 3 (-4) 8 = 19 := by
  sorry

end bug_total_distance_l62_62222


namespace lizard_crossing_probability_l62_62650

theorem lizard_crossing_probability : Q(1) = 70 / 171 :=
begin
  let Q : ℕ → ℝ,
  assume Q,
  -- Initial conditions
  Q(0) = 0,
  Q(11) = 1,
  -- Recursive relation for Q(N) where 0 < N < 11
  assume h : 0 < N ∧ N < 11,
  Q(N) = (N / 11) * Q(N - 1) + (1 - N / 11) * Q(N + 1),
  sorry
end

end lizard_crossing_probability_l62_62650


namespace reciprocal_self_eq_one_or_neg_one_l62_62009

theorem reciprocal_self_eq_one_or_neg_one (x : ℝ) (h : x = 1 / x) : x = 1 ∨ x = -1 := sorry

end reciprocal_self_eq_one_or_neg_one_l62_62009


namespace optimal_heaviest_backpack_weight_l62_62145

-- Given weights of the rock specimens
def weights : List ℝ := [8.5, 6.0, 4.0, 4.0, 3.0, 2.0]

-- Number of backpacks
def num_backpacks : ℕ := 3

-- Define the main theorem to be proved
theorem optimal_heaviest_backpack_weight :
  ∃ (distribution : List (List ℝ)), 
    (∀ rocks, rocks ∈ distribution → rocks ≠ []) ∧ 
    distribution.length = num_backpacks ∧
    (∀ rocks, sum rocks ≤ 10.5) ∧
    max (distribution.map sum) = 10 := sorry

end optimal_heaviest_backpack_weight_l62_62145


namespace groups_of_medal_winners_l62_62627

theorem groups_of_medal_winners : ∀ (n k : ℕ), n = 6 → k = 3 → nat.choose n k = 20 :=
by sorry

end groups_of_medal_winners_l62_62627


namespace december_revenue_times_average_l62_62007

variable (D : ℝ) -- December's revenue
variable (N : ℝ) -- November's revenue
variable (J : ℝ) -- January's revenue

-- Conditions
def revenue_in_november : N = (2/5) * D := by sorry
def revenue_in_january : J = (1/2) * N := by sorry

-- Statement to be proved
theorem december_revenue_times_average :
  D = (10/3) * ((N + J) / 2) :=
by sorry

end december_revenue_times_average_l62_62007


namespace condition_iff_odd_function_l62_62370

theorem condition_iff_odd_function (f : ℝ → ℝ) :
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
by
  sorry

end condition_iff_odd_function_l62_62370


namespace final_fish_stock_l62_62094

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end final_fish_stock_l62_62094


namespace sum_of_four_consecutive_integers_divisible_by_two_l62_62154

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l62_62154


namespace intersecting_triangles_three_points_l62_62859

theorem intersecting_triangles_three_points (S : Set (Set (ℝ × ℝ))) 
  (h1 : ∀ T ∈ S, ∃ a b c : ℝ × ℝ, equilateral T a b c)
  (h2 : ∀ T₁ T₂ ∈ S, ∃ d : ℝ × ℝ, d ∈ T₁ ∧ d ∈ T₂) :
  ∃ X Y Z : ℝ × ℝ, ∀ T ∈ S, X ∈ T ∨ Y ∈ T ∨ Z ∈ T :=
by
  sorry

def equilateral (T : Set (ℝ × ℝ)) (a b c : ℝ × ℝ) : Prop :=
  T = {a, b, c} ∧ dist a b = dist b c ∧ dist b c = dist c a

end intersecting_triangles_three_points_l62_62859


namespace license_plates_count_is_2400_l62_62881

def is_vowel (c : Char) : Prop :=
  ∃ (v : Char), v = 'A' ∨ v = 'E' ∨ v = 'I' ∨ v = 'O' ∨ v = 'U' ∧ c = v

def is_consonant (c : Char) : Prop :=
  ∃ (v : Char), v ≠ 'A' ∧ v ≠ 'E' ∧ v ≠ 'I' ∧ v ≠ 'O' ∧ v ≠ 'U' ∧ v ≠ 'G' ∧ c = v

noncomputable def num_license_plates : Nat :=
  (fin.enum 5).choose_M_1_1_by (is_vowel.start ∧ is_consonant.middle_3 ∧ is_vowel.end) 

theorem license_plates_count_is_2400 :
  num_license_plates = 2400 :=
sorry

end license_plates_count_is_2400_l62_62881


namespace least_x_value_l62_62380

variable (a b : ℕ)
variable (positive_int_a : 0 < a)
variable (positive_int_b : 0 < b)
variable (h : 2 * a^5 = 3 * b^2)

theorem least_x_value (h : 2 * a^5 = 3 * b^2) (positive_int_a : 0 < a) (positive_int_b : 0 < b) : ∃ x, x = 15552 ∧ x = 2 * a^5 ∧ x = 3 * b^2 :=
sorry

end least_x_value_l62_62380


namespace point_not_on_graph_l62_62619

theorem point_not_on_graph : ∀ (x y : ℝ), (x, y) = (-1, 1) → ¬ (∃ z : ℝ, z ≠ -1 ∧ y = z / (z + 1)) :=
by {
  sorry
}

end point_not_on_graph_l62_62619


namespace average_income_proof_l62_62469

def average_income_independent_of_bonus_distribution (A E : ℝ) : Prop :=
  ∀ (distribution_method : (fin 10 → ℝ) → Prop), 
    (∑ i, (distribution_method (λ _, (A + E) / 10)) = A + E) →
    (∀ i, distribution_method (λ _, (A + E) / 10) i = (A + E) / 10)

theorem average_income_proof (A E : ℝ) :
  average_income_independent_of_bonus_distribution A E := by
  sorry

end average_income_proof_l62_62469


namespace find_a_if_f_even_l62_62444

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62444


namespace imaginary_part_of_conjugate_l62_62351

noncomputable def a (b : ℝ) : ℂ := 1

noncomputable def z (b : ℝ) : ℂ := a b + b * complex.I

theorem imaginary_part_of_conjugate (b : ℝ) (h : b = -2) : complex.im (complex.conj (z b)) = 2 :=
by
  sorry

end imaginary_part_of_conjugate_l62_62351


namespace num_possible_values_l62_62928

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62928


namespace apples_in_bag_l62_62924

theorem apples_in_bag :
  ∃ N : ℤ,
    (|20 - N| ∈ {1, 3, 6} ∧
     |22 - N| ∈ {1, 3, 6} ∧
     |25 - N| ∈ {1, 3, 6} ∧
     |20 - N| ≠ |22 - N| ∧
     |20 - N| ≠ |25 - N| ∧
     |22 - N| ≠ |25 - N| ) ∧
    N = 19 :=
begin
  sorry
end

end apples_in_bag_l62_62924


namespace collinear_B_H_Q_l62_62514

theorem collinear_B_H_Q
  (A B C H I P K Q : Type)
  [triangle ABC]
  (H_orthocenter : is_orthocenter ABC H)
  (I_incenter : is_incenter ABC I)
  (circumcircle_BCI : circumscribes BCI (circumcircle BCI))
  (P_on_AB : P ≠ B ∧ lies_on P (segment A B) ∧ lies_on P (circumcircle BCI))
  (projection_HA_to_AI : projection H (line A I) = K)
  (Q_reflection_P_K : reflection P K = Q)
  : collinear {B, H, Q} :=
sorry

end collinear_B_H_Q_l62_62514


namespace find_angle_A_area_bound_given_a_l62_62724

-- (1) Given the condition, prove that \(A = \frac{\pi}{3}\).
theorem find_angle_A
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C)) :
  A = Real.pi / 3 :=
sorry

-- (2) Given a = 4, prove the area S satisfies \(S \leq 4\sqrt{3}\).
theorem area_bound_given_a
  {A B C : ℝ} {a b c S : ℝ}
  (ha : a = 4)
  (hA : A = Real.pi / 3)
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C))
  (hS : S = 1 / 2 * b * c * Real.sin A) :
  S ≤ 4 * Real.sqrt 3 :=
sorry

end find_angle_A_area_bound_given_a_l62_62724


namespace ptolemy_theorem_l62_62022

-- Let Z1, Z2, Z3, Z4 be complex numbers representing the vertices of a cyclic quadrilateral.
variables (Z1 Z2 Z3 Z4 : ℂ)

-- State that Z1, Z2, Z3, Z4 form a cyclic quadrilateral
def cyclic_quadrilateral (Z1 Z2 Z3 Z4 : ℂ) := sorry

-- Assertion to be proven using the given conditions
theorem ptolemy_theorem (h : cyclic_quadrilateral Z1 Z2 Z3 Z4) :
  |Z1 - Z3| * |Z2 - Z4| = |Z1 - Z2| * |Z3 - Z4| + |Z1 - Z4| * |Z2 - Z3| :=
sorry

end ptolemy_theorem_l62_62022


namespace m_value_proof_l62_62301

noncomputable def quadratic_function_vertex_m (m : ℝ) : Prop :=
    let y := λ x : ℝ, 3 * x^2 + 2 * (m-1) * x + (0 : ℝ) in
    ∃ n : ℝ, y = 3 * x^2 + 2 * (m-1) * x + n ∧
    (∀ x : ℝ, x < 1 → y.decreasing_on (set.Iic x)) ∧
    (∀ x : ℝ, x ≥ 1 → y.increasing_on (set.Ici x))

theorem m_value_proof : quadratic_function_vertex_m (-2) :=
by {
    sorry
}

end m_value_proof_l62_62301


namespace not_arithmetic_progression_l62_62510

-- Define the sequence as a function: ℕ → ℕ
def a : ℕ → ℕ
| 0     := 2
| 1     := 3
| (n+2) := a n + a (n + 1)

-- Define a property to check if a sequence is an arithmetic progression
def is_arithmetic_progression (a : ℕ → ℕ) : Prop :=
∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

-- The theorem stating that the sequence is not an arithmetic progression
theorem not_arithmetic_progression (a : ℕ → ℕ)
  (h₀ : a 0 = 2)
  (h₁ : a 1 = 3)
  (h_rec : ∀ n : ℕ, a (n + 2) = a n + a (n + 1)) :
  ¬ is_arithmetic_progression a :=
by {
  -- This is the place for the proof, but we leave it as sorry
  sorry
}

end not_arithmetic_progression_l62_62510


namespace karlson_max_candies_l62_62919

noncomputable def max_candies_29_minutes : ℕ :=
  406

theorem karlson_max_candies (n : ℕ) (h : n = 29) : 
  ∑ (k : ℕ) in finset.range (n - 1), (k * (n - k)) = max_candies_29_minutes :=
by
  sorry

end karlson_max_candies_l62_62919


namespace muffins_for_sale_l62_62455

theorem muffins_for_sale :
  let boys_1 := 3 * 12,
      boys_2 := 2 * 18,
      girls_1 := 2 * 20,
      girls_2 := 15,
      total_made := boys_1 + boys_2 + girls_1 + girls_2,
      not_for_sale := (15 * total_made) / 100,
      muffins_for_sale := total_made - Int.floor not_for_sale
  in muffins_for_sale = 108 :=
by
  let boys_1 := 3 * 12
  let boys_2 := 2 * 18
  let girls_1 := 2 * 20
  let girls_2 := 15
  let total_made := boys_1 + boys_2 + girls_1 + girls_2
  let not_for_sale := (15 * total_made) / 100
  let muffins_for_sale := total_made - Int.floor not_for_sale
  have : muffins_for_sale = 108 := by sorry
  exact this

end muffins_for_sale_l62_62455


namespace painters_work_days_l62_62808

/-
It takes five painters working at the same rate 1.5 work-days to finish a job.
If only four painters are available, prove how many work-days will it take them to finish the job, working at the same rate.
-/

theorem painters_work_days (days5 : ℚ) (h : days5 = 3 / 2) :
  ∃ days4 : ℚ, 5 * days5 = 4 * days4 ∧ days4 = 15 / 8 :=
  by
    use 15 / 8
    split
    · calc
        5 * days5 = 5 * (3 / 2) : by rw h
        ... = 15 / 2 : by norm_num
        ... = 4 * (15 / 8) : by norm_num
    · refl

end painters_work_days_l62_62808


namespace parallelogram_sides_are_parallel_l62_62655

theorem parallelogram_sides_are_parallel 
  {a b c : ℤ} (h_area : c * (a^2 + b^2) = 2011 * b) : 
  (∃ k : ℤ, a = 2011 * k ∧ (b = 2011 ∨ b = -2011)) :=
by
  sorry

end parallelogram_sides_are_parallel_l62_62655


namespace cute_5_digit_integers_unique_l62_62409

theorem cute_5_digit_integers_unique :
  ∃! (n : ℕ), (∃ (d1 d2 d3 d4 d5 : ℕ),
    (d1 ∈ {1, 2, 3, 4, 5}) ∧ (d2 ∈ {1, 2, 3, 4, 5}) ∧
    (d3 ∈ {1, 2, 3, 4, 5}) ∧ (d4 ∈ {1, 2, 3, 4, 5}) ∧
    (d5 ∈ {1, 2, 3, 4, 5}) ∧
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧
    (d3 ≠ d4) ∧ (d3 ≠ d5) ∧
    (d4 ≠ d5) ∧
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    d1 % 1 = 0 ∧
    (d1 * 10 + d2) % 2 = 0 ∧
    (d1 * 100 + d2 * 10 + d3) % 3 = 0 ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4) % 4 = 0 ∧
    (d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) % 5 = 0) :=
by sorry

end cute_5_digit_integers_unique_l62_62409


namespace cube_root_of_27_eq_3_l62_62116

theorem cube_root_of_27_eq_3 : real.cbrt 27 = 3 :=
by {
  have h : 27 = 3 ^ 3 := by norm_num,
  rw real.cbrt_eq_iff_pow_eq (by norm_num : 0 ≤ 27) h,
  norm_num,
  sorry
}

end cube_root_of_27_eq_3_l62_62116


namespace neon_signs_blink_together_l62_62341

theorem neon_signs_blink_together :
  Nat.lcm (Nat.lcm (Nat.lcm 7 11) 13) 17 = 17017 :=
by
  sorry

end neon_signs_blink_together_l62_62341


namespace area_of_triangle_AMC_l62_62987

/-- Triangle AMC is isosceles with AM = AC,
and medians MV and CU are perpendicular to each other,
and MV = CU = 12.
We want to prove that the area of triangle AMC is 288. -/
theorem area_of_triangle_AMC :
  ∀ (A M C U V : EuclideanGeometry.Point)
  (h_isosceles : EuclideanGeometry.distance A M = EuclideanGeometry.distance A C)
  (h_perpendicular : EuclideanGeometry.perpendicular (EuclideanGeometry.lineSegment M V) (EuclideanGeometry.lineSegment C U))
  (h_equal_medians_MV : EuclideanGeometry.distance M V = 12)
  (h_equal_medians_CU : EuclideanGeometry.distance C U = 12),
  EuclideanGeometry.area (EuclideanGeometry.triangle A M C) = 288 :=
begin
  sorry
end

end area_of_triangle_AMC_l62_62987


namespace additional_savings_together_l62_62251

def usual_price_per_window : ℕ := 120

def free_windows_per_five_purchased : ℕ := 2

def dave_windows_needed : ℕ := 9

def doug_windows_needed : ℕ := 10

theorem additional_savings_together
  (usual_price : ℕ)
  (free_per_five : ℕ)
  (dave_needed : ℕ)
  (doug_needed : ℕ) :
  ∀ usual_price_per_window == usual_price,
  ∀ free_windows_per_five_purchased == free_per_five,
  ∀ dave_windows_needed == dave_needed,
  ∀ doug_windows_needed == doug_needed,
  (additional_savings_together usual_price free_per_five dave_needed doug_needed) = 0 :=
by
  sorry

end additional_savings_together_l62_62251


namespace maria_score_l62_62525

theorem maria_score (m j : ℕ) (h1 : m = j + 50) (h2 : (m + j) / 2 = 112) : m = 137 :=
by
  sorry

end maria_score_l62_62525


namespace domain_of_logarithm_l62_62565

theorem domain_of_logarithm (x : ℝ) : x^2 - 5 * x + 4 > 0 → x ∈ (-∞, 1) ∪ (4, +∞) :=
by
  sorry

end domain_of_logarithm_l62_62565


namespace find_constants_range_of_f_const_l62_62354

-- Define the function and its derivative
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2
def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 6 * a * x + b

-- State the conditions
theorem find_constants (a b : ℝ) (h₀ : a > 1) (h₁ : f' (-1) a b = 0) (h₂ : f (-1) a b = 0) :
  a = 2 ∧ b = 9 := sorry

-- State the function definition again with found constants
def f_const (x : ℝ) : ℝ := f x 2 9

-- State interval endpoint values
def interval_endpoints (x : ℝ) : Prop := x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0

-- Determine the range of f_const on [-4, 0]
theorem range_of_f_const :
  ∀ x : ℝ, (interval_endpoints x) → f_const x ∈ set.Icc 0 4 := sorry

end find_constants_range_of_f_const_l62_62354


namespace zoe_remaining_pictures_l62_62213

-- Definitions based on the conditions
def total_pictures : Nat := 88
def colored_pictures : Nat := 20

-- Proof statement
theorem zoe_remaining_pictures : total_pictures - colored_pictures = 68 := by
  sorry

end zoe_remaining_pictures_l62_62213


namespace find_a_if_f_even_l62_62449

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62449


namespace player_one_win_l62_62914

theorem player_one_win (total_coins : ℕ) (coins_taken_by_player1 : ℕ) 
  (player1_moves : ∀ coins_remaining : ℕ, (1 ≤ coins_remaining ∧ coins_remaining ≤ 99) → coins_remaining % 2 = 1)
  (player2_moves : ∀ coins_remaining : ℕ, (2 ≤ coins_remaining ∧ coins_remaining ≤ 100) → coins_remaining % 2 = 0) :
  total_coins = 2015 → coins_taken_by_player1 = 95 → 
  ∃ strategy, ∀ turn : ℕ, (turn % 2 = 1 → strategy turn coins_taken_by_player1 = true) ∧ (turn % 2 = 0 → strategy turn coins_taken_by_player1 = false) :=
begin
  sorry
end

end player_one_win_l62_62914


namespace pompeiu_theorem_l62_62364

theorem pompeiu_theorem
  (X : Point)
  (A B C : Point)
  (h1 : is_equilateral_triangle A B C) :
  (∃ T : Triangle, T = triangle_mk X A B ∧ T.degenerate ∧ 
                   T = triangle_mk B X C ∧ T.degenerate ∧
                   T = triangle_mk C X A ∧ T.degenerate) ↔
  (lies_on_circumcircle X A B C) :=
sorry

end pompeiu_theorem_l62_62364


namespace birdseed_squirrel_ratio_l62_62495

-- Define the conditions
def each_cup_feeds_birds : ℕ := 14
def total_cups_in_feeder : ℕ := 2
def birds_fed_weekly : ℕ := 21

-- Define the question in Lean statement form
theorem birdseed_squirrel_ratio :
  let birdseed_needed := each_cup_feeds_birds * total_cups_in_feeder,
      birdseed_stolen := birdseed_needed - birds_fed_weekly,
      cups_stolen := birdseed_stolen / each_cup_feeds_birds,
      ratio := cups_stolen / total_cups_in_feeder
  in ratio = 1 / 4 :=
by {
  sorry
}

end birdseed_squirrel_ratio_l62_62495


namespace first_four_eq_last_four_l62_62260

-- Let S be a sequence of length n consisting of 0s and 1s.
def sequence (n : ℕ) : Type := fin n → bool

-- Define the condition that every two sections of successive 5 terms in the sequence S are different
def different_sections {n : ℕ} (S : sequence n) :=
  ∀ i j : fin n, 1 ≤ i.val ∧ i.val < j.val ∧ j.val ≤ (n - 4) →
    (S i, S ⟨i.val+1, _⟩, S ⟨i.val+2, _⟩, S ⟨i.val+3, _⟩, S ⟨i.val+4, _⟩) ≠
    (S j, S ⟨j.val+1, _⟩, S ⟨j.val+2, _⟩, S ⟨j.val+3, _⟩, S ⟨j.val+4, _⟩)

-- State the theorem
theorem first_four_eq_last_four {n : ℕ} {S : sequence n} (h : different_sections S) :
  (S 0, S ⟨1, nat.lt_succ_self 1⟩, S ⟨2, nat.succ_lt_succ (nat.lt_succ_self 1)⟩, S ⟨3, (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 1)))⟩) =
  (S ⟨n-4, nat.sub_lt (show 4 > 0 by norm_num) (show n ≠ 0 by nat.ne_zero_of_pos (show n > 0 by sorry))⟩, S ⟨n-3, _⟩, S ⟨n-2, _⟩, S ⟨n-1, nat.pred_lt (nat.ne_zero_of_pos (show n > 0 by sorry))⟩) :=
sorry

end first_four_eq_last_four_l62_62260


namespace lesser_fraction_l62_62138

theorem lesser_fraction 
  (x y : ℚ)
  (h_sum : x + y = 13 / 14)
  (h_prod : x * y = 1 / 5) :
  min x y = 87 / 700 := sorry

end lesser_fraction_l62_62138


namespace sum_of_four_consecutive_integers_prime_factor_l62_62162

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l62_62162


namespace max_candies_eaten_in_29_minutes_l62_62921

theorem max_candies_eaten_in_29_minutes :
  let n := 29,
      sum_edges_complete_graph := (n * (n - 1)) / 2
  in sum_edges_complete_graph = 406 :=
by
  let n := 29
  let sum_edges_complete_graph := (n * (n - 1)) / 2
  have h : sum_edges_complete_graph = 406,
  { sorry },
  exact h

end max_candies_eaten_in_29_minutes_l62_62921


namespace ruslan_kolya_coins_l62_62874

/-- 
Ruslan and Kolya need to pay 2006 rubles each using the same number of coins, 
but they have no coins of the same denomination. 
-/
theorem ruslan_kolya_coins (k : ℕ) 
  (coins_r : (ℕ → ℕ) → Prop)
  (coins_k : (ℕ → ℕ) → Prop)
  (hr : ∀ n, coins_r n → (n 1 = 0 ∨ n 2 = 0 ∨ n 5 = 0))
  (hk : ∀ n, coins_k n → (n 1 = 0 ∨ n 2 = 0 ∨ n 5 = 0))
  (hrk : ∀ n, coins_r n → coins_k n → (n 1 = 0 ↔ n 1 ≠ 0) ∧ (n 2 = 0 ↔ n 2 ≠ 0) ∧ (n 5 = 0 ↔ n 5 ≠ 0))
  (hr₁ : coins_r (λ n, if n = 1 then k else 0))
  (hk₁ : coins_k (λ n, if n = 2 then k else 0))
  (h_total_r : (λ n, if n = 1 then k else 0) 1 * 1 + (λ n, if n = 2 then k else 0) 2 * 2 + (λ n, if n = 5 then k else 0) 5 * 5 = 2006)
  (h_total_k : (λ n, if n = 2 then k else 0) 1 * 1 + (λ n, if n = 2 then k else 0) 2 * 2 + (λ n, if n = 5 then k else 0) 5 * 5 = 2006) 
  : false :=
begin
  sorry
end

end ruslan_kolya_coins_l62_62874


namespace probability_no_adjacent_same_roll_l62_62332

theorem probability_no_adjacent_same_roll :
  let A := 1 -- rolls a six-sided die
  let B := 2 -- rolls a six-sided die
  let C := 3 -- rolls a six-sided die
  let D := 4 -- rolls a six-sided die
  let E := 5 -- rolls a six-sided die
  let people := [A, B, C, D, E]
  -- A and C are required to roll different numbers
  let prob_A_C_diff := 5 / 6
  -- B must roll different from A and C
  let prob_B_diff := 4 / 6
  -- D must roll different from C and A
  let prob_D_diff := 4 / 6
  -- E must roll different from D and A
  let prob_E_diff := 3 / 6
  (prob_A_C_diff * prob_B_diff * prob_D_diff * prob_E_diff) = 10 / 27 :=
by
  sorry

end probability_no_adjacent_same_roll_l62_62332


namespace color_points_l62_62600

def point := ℝ × ℝ

variable {n : ℕ}
variable (X : fin n → point)

theorem color_points (X : fin n → point) :
  ∃ (color : fin n → Prop), ∀ d : ℝ, d ∈ X.horizontalLines ∨ d ∈ X.verticalLines →
    |(# (i : fin n) | color i ∧ X i.1 = d) - (# (i : fin n) | ¬color i ∧ X i.1 = d)| ≤ 1 :=
sorry

end color_points_l62_62600


namespace BC_vector_MN_vector_l62_62038

variables (A B C D M N : Type) [AddGroup A] [VectorSpace ℝ A]
variables (a b : A)

-- Given conditions
variables (AB_parallel_CD : ∀ (X Y Z W : A), X - Y = 2 * (Z - W))
           (midpoint_M : M = (D + C) / 2)
           (midpoint_N : N = (A + B) / 2)
           (AB : A) [is_ab : AB = α]
           (AD : A) [is_ad : AD = β]

-- Theorems to prove
theorem BC_vector (h₁ : BC = b - 0.5 * a) : True :=
by sorry

theorem MN_vector (h₂ : MN = 0.25 * a - b) : True :=
by sorry


end BC_vector_MN_vector_l62_62038


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62168

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62168


namespace speed_of_car_first_hour_98_l62_62131

def car_speed_in_first_hour_is_98 (x : ℕ) : Prop :=
  (70 + x) / 2 = 84 → x = 98

theorem speed_of_car_first_hour_98 (x : ℕ) (h : car_speed_in_first_hour_is_98 x) : x = 98 :=
  by
  sorry

end speed_of_car_first_hour_98_l62_62131


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62167

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62167


namespace bat_wings_area_calculation_l62_62086

noncomputable def calculate_bat_wings_area : ℝ :=
  let E := (0 : ℝ, 0 : ℝ)
  let F := (4 : ℝ, 0 : ℝ)
  let A := (4 : ℝ, 5 : ℝ)
  let D := (0 : ℝ, 5 : ℝ)
  let C := (2 : ℝ, 5 : ℝ)
  let B := (4 : ℝ, 3 : ℝ)
  let Z_x := 60 / 19
  let Z_y := 45 / 19
  let triangle_area := (a b c : ℝ × ℝ) → (a.1 * b.2 + b.1 * c.2 + c.1 * a.2 - (a.2 * b.1 + b.2 * c.1 + c.2 * a.1)) / 2
  let area_ECZ := triangle_area E C (Z_x, Z_y)
  let area_FZB := triangle_area F (Z_x, Z_y) B
  let quarter_circle_area := (π * 2 * 2) / 4
  (area_ECZ + area_FZB - quarter_circle_area).abs

theorem bat_wings_area_calculation : calculate_bat_wings_area = 4 :=
by
  sorry

end bat_wings_area_calculation_l62_62086


namespace power_mod_l62_62709

theorem power_mod (n : ℕ) : (3 ^ 2017) % 17 = 3 := 
by
  sorry

end power_mod_l62_62709


namespace cube_root_of_27_eq_3_l62_62115

theorem cube_root_of_27_eq_3 : real.cbrt 27 = 3 :=
by {
  have h : 27 = 3 ^ 3 := by norm_num,
  rw real.cbrt_eq_iff_pow_eq (by norm_num : 0 ≤ 27) h,
  norm_num,
  sorry
}

end cube_root_of_27_eq_3_l62_62115


namespace complex_quadrant_l62_62791

theorem complex_quadrant :
  let z := complex.I * (1 + 2 * complex.I) in
  let x := z.re in
  let y := z.im in
  x < 0 ∧ y > 0 :=
by
  let z := complex.I * (1 + 2 * complex.I)
  let x := z.re
  let y := z.im
  sorry

end complex_quadrant_l62_62791


namespace collinear_A_P_Q_l62_62685

open EuclideanGeometry

variables {A B C D E P F G Q : Point}
variables {s t u v w x y : Line}

-- Basic assumptions and conditions
axiom DE_parallel_BC : parallel (line_through D E) (line_through B C)
axiom point_D_on_AB : lies_on D (segment A B)
axiom point_E_on_AC : lies_on E (segment A C)
axiom point_F_on_DE_BP : lies_on F (intersection (line_through D E) (line_through B P))
axiom point_G_on_DE_CP : lies_on G (intersection (line_through D E) (line_through C P))
axiom circumcircle_PD_G : circle (circumcircle_through P D G)  -- circumcircle through points P, D, G
axiom circumcircle_PF_E : circle (circumcircle_through P F E)  -- circumcircle through points P, F, E
axiom point_Q_second_intersection : lies_on Q (second_intersection 
  (circumcircle_through P D G) (circumcircle_through P F E))

-- The statement to be proven
theorem collinear_A_P_Q : collinear A P Q :=
sorry

end collinear_A_P_Q_l62_62685


namespace will_total_heroes_l62_62620

theorem will_total_heroes (heroes_front : ℕ) (heroes_back : ℕ) (h1 : heroes_front = 2) (h2 : heroes_back = 7) : heroes_front + heroes_back = 9 :=
by
  rw [h1, h2]
  sorry

end will_total_heroes_l62_62620


namespace possible_N_values_l62_62950

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62950


namespace proof_problem_l62_62577

def proposition1 (p q : Prop) : Prop := (p ∧ ¬q) → (p ∧ q)
def proposition2 (α : ℝ) : Prop := (sin α = 1/2) ↔ (α = real.pi / 6)
def proposition3 : Prop := (¬∀ x : ℝ, 2^x > 0) ↔ ∃ x : ℝ, 2^x ≤ 0

def problem : Prop :=
  (∃ α : ℝ, (sin α = 1/2) ∧ (α ≠ real.pi / 6)) ∧
  ((¬ ∀ α : ℝ, (sin α = 1/2) → α = real.pi / 6) ∨
   ((¬∀ x : ℝ, 2^x > 0) ∧ (∃ x : ℝ, 2^x ≤ 0))) ∧
  (¬ (∃ p q : Prop, (p ∧ ¬q) → (p ∧ q))) ∧
  2 = 2

theorem proof_problem : problem := by
  sorry

end proof_problem_l62_62577


namespace period_tan_transformed_l62_62998

theorem period_tan_transformed (x : ℝ) : 
  (∃ p : ℝ, ∀ x : ℝ, tan (3 * x / 5) = tan ((3 * (x + p)) / 5)) → p = 5 * π / 3 :=
by
  sorry

end period_tan_transformed_l62_62998


namespace unique_a_exists_l62_62316

theorem unique_a_exists (n : ℕ) (h : 0 < n) : 
  (∃! a : ℕ, 0 ≤ a ∧ a < n! ∧ n! ∣ (a^n + 1)) ↔ (n = 2 ∨ (Nat.prime n)) :=
by
  sorry

end unique_a_exists_l62_62316


namespace gcd_72_108_l62_62187

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l62_62187


namespace annual_sales_volume_last_year_l62_62597

variable (x : ℝ) -- annual sales volume of last year in ten thousand kilograms
variable sales_increase : (∀ this_year last_year: ℝ, this_year = 4 * last_year)
variable price_increase : (∀ curr_price last_price: ℝ, curr_price = last_price + 20)
variable revenue_this_year : (∀ sales_curr_price revenue_now: ℝ, revenue_now = sales_curr_price * 4 * x)
variable revenue_last_year : (∀ last_sales_price revenue_prev: ℝ, revenue_prev = last_sales_price * x)

theorem annual_sales_volume_last_year :
  (∃ (prev_price curr_price prev_sales this_year_sales : ℝ),
    prev_sales = x * 10^4 ∧
    this_year_sales = 4 * prev_sales ∧
    revenue_last_year is_correct ∧
    revenue_this_year is_correct ∧
    price_increase is_correct ∧
    sales_increase is_correct)
→ x = 3.75 :=
by
  sorry

end annual_sales_volume_last_year_l62_62597


namespace sum_difference_sets_l62_62875

theorem sum_difference_sets :
  let SetA := {x : ℕ | 8 ≤ x ∧ x ≤ 56},
      SetB := {x : ℕ | 104 ≤ x ∧ x ≤ 156},
      sumA := (∑ x in SetA, x),
      sumB := (∑ x in SetB, x)
  in
  sumB - sumA = 5322 :=
by
  sorry

end sum_difference_sets_l62_62875


namespace correct_representation_l62_62289

-- Definitions based on conditions
variable (x : ℝ) (hx : x > 0)

def red_segment : ℝ := 3 * x
def blue_segment : ℝ := x
def green_segment : ℝ := 0.5 * x
def total_proportion : ℝ := x + 3 * x + 0.5 * x

-- Bar graph representation (Answer C)
def red_bar_height : ℝ := red_segment x / total_proportion x hx
def blue_bar_height : ℝ := blue_segment x / total_proportion x hx
def green_bar_height : ℝ := green_segment x / total_proportion x hx

-- Theorem to prove the correctness of the bar graph representation
theorem correct_representation :
  red_bar_height x hx = 2 / 3 ∧ 
  blue_bar_height x hx = 1 / 4.5 ∧ 
  green_bar_height x hx = 1 / 9 :=
by
  sorry

end correct_representation_l62_62289


namespace prob_all_black_is_6_over_25_prob_one_white_is_12_over_25_l62_62382

-- Define the initial conditions
def bagA_white : ℕ := 2
def bagA_black : ℕ := 4
def bagB_white : ℕ := 1
def bagB_black : ℕ := 4

-- Define the total number of balls in each bag
def total_balls_A : ℕ := bagA_white + bagA_black
def total_balls_B : ℕ := bagB_white + bagB_black

-- Define the draw events for bag A and bag B
def event_all_black_A : ℕ := (bagA_black * (bagA_black - 1)) / (total_balls_A * (total_balls_A - 1))
def event_all_black_B : ℕ := (bagB_black * (bagB_black - 1)) / (total_balls_B * (total_balls_B - 1))

-- Define the probability for both events happening
def prob_all_black : ℚ := (event_all_black_A * event_all_black_B) / ((total_balls_A - 1) * (total_balls_B - 1))

-- Define the events for exactly one white ball drawn
def event_one_white_C : ℝ := (bagA_black * (bagA_black - 1)) / (total_balls_A * (total_balls_A - 1)) *
                             (bagB_white * bagB_black) / ((total_balls_B - 1) * total_balls_B)
def event_one_white_D : ℝ := (bagA_white * bagA_black) / (total_balls_A * (total_balls_A - 1)) *
                             (bagB_black * (bagB_black - 1)) / ((total_balls_B - 1) * total_balls_B)

-- Define the probability for exactly one white event
def prob_one_white : ℚ := event_one_white_C + event_one_white_D

-- Theorem statements
theorem prob_all_black_is_6_over_25 :
  prob_all_black = 6 / 25 := sorry

theorem prob_one_white_is_12_over_25 :
  prob_one_white = 12 / 25 := sorry

end prob_all_black_is_6_over_25_prob_one_white_is_12_over_25_l62_62382


namespace totalPeaches_l62_62638

-- Define the number of red, yellow, and green peaches
def redPeaches := 7
def yellowPeaches := 15
def greenPeaches := 8

-- Define the total number of peaches and the proof statement
theorem totalPeaches : redPeaches + yellowPeaches + greenPeaches = 30 := by
  sorry

end totalPeaches_l62_62638


namespace min_value_of_a2_b2_l62_62768

theorem min_value_of_a2_b2 (a b : ℝ) : 
  (∀ x : ℝ, (∃ r : ℕ, r = 3 ∧ binomial 6 r * a^(6-r) * b^r * x^(12 - 3*r) = 20 * x^3)) → a * b = 1 → a^2 + b^2 ≥ 2 := 
by
  sorry

end min_value_of_a2_b2_l62_62768


namespace students_range_score_100_140_l62_62229

noncomputable def normal_distribution_condition (X : Type) [norm : NormedAddCommGroup X] [measure_space X]
  (μ : ℝ) (σ : ℝ) (t : ℝ) :=
  ∀ (x : X), (measure_prob (measure_normal μ σ) {y | y > t} = 0.2)

theorem students_range_score_100_140 :
  ∀ (X : Type) [norm : NormedAddCommGroup X] [measure_space X]
    (μ σ : ℝ),
    let students := 50 in
    let P := measure_normal μ σ in
    normal_distribution_condition X μ σ 140 →
    (X = ℝ ∧ μ = 120 ∧ P X = μ ∧ (students * (measure_prob P {y | y > 140} = 0.2)) →
    ∑ in {y | 100 ≤ y ∧ y ≤ 140} = 30) :=
by
  sorry

end students_range_score_100_140_l62_62229


namespace marble_selection_probability_l62_62820

noncomputable def probability_of_exactly_three_green_marbles : ℚ :=
  (choose 7 3) * (8 / 15)^3 * (7 / 15)^4

theorem marble_selection_probability : 
  probability_of_exactly_three_green_marbles = 860818 / 3421867 :=
by sorry

end marble_selection_probability_l62_62820


namespace union_A_B_l62_62345

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℝ := { x | x^3 = x }

theorem union_A_B : A ∪ B = { -1, 0, 1, 2 } := by
  sorry

end union_A_B_l62_62345


namespace unique_point_exists_l62_62755

variables (A B : Point) (e : Line) -- Points A and B, and line e
variables (P Q : Point) -- Points P and Q
variables (H1 : P ∈ e) (H2 : Q ∈ e) -- P and Q are on line e
variables (H3 : ∠P A Q = 90) -- Angle PAQ is 90 degrees

theorem unique_point_exists (A B : Point) (e : Line) (P Q : Point)
  (H1 : P ∈ e) (H2 : Q ∈ e) (H3 : ∠P A Q = 90) : 
  ∃ X : Point, X ≠ B ∧ ∀ (P Q : Point), (circle_through_points B P Q).passes_through X := 
sorry

end unique_point_exists_l62_62755


namespace membership_condition_l62_62070

def M : set ℝ := { y | ∃ x > 0, y = Real.log x }
def N : set ℝ := { x | x > 0 }

theorem membership_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M) :=
sorry

end membership_condition_l62_62070


namespace valid_number_of_two_humped_camels_within_range_l62_62962

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62962


namespace sum_of_four_consecutive_integers_divisible_by_two_l62_62158

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l62_62158


namespace even_function_a_eq_zero_l62_62441

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62441


namespace find_a_if_f_even_l62_62445

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62445


namespace problem1_log_expression_problem2_f_x_plus_1_l62_62217

-- Define Problem 1 in Lean 4
theorem problem1_log_expression :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - real.exp (log 5 (3 * log 5 1)) = -1 :=
sorry

-- Define Problem 2 in Lean 4
noncomputable def f (x : ℝ) : ℝ :=
  (x + 1 / x) ^ 2

theorem problem2_f_x_plus_1 (x : ℝ) :
  f (x + 1) = (x + 1) ^ 2 + 2 * (x + 1) + 2 / (x + 1) + (1 / (x + 1)^2) :=
sorry

end problem1_log_expression_problem2_f_x_plus_1_l62_62217


namespace parallel_condition_l62_62739

theorem parallel_condition (a : ℝ) : (a = -1) ↔ (¬ (a = -1 ∧ a ≠ 1)) ∧ (¬ (a ≠ -1 ∧ a = 1)) :=
by
  sorry

end parallel_condition_l62_62739


namespace overlap_exists_l62_62784

-- Define the conditions of the problem
variable (R: Type) [Nonempty R]

-- Given a rectangle of area 5 square units
constant area : R → ℝ
axiom big_rectangle : R
axiom big_rectangle_area : area big_rectangle = 5

-- Given 9 smaller rectangles each with area 1 square unit
axiom small_rectangle_set : Set R
axiom small_rectangle_count : small_rectangle_set.card = 9
axiom each_small_rectangle_area : ∀ r ∈ small_rectangle_set, area r = 1

-- Define the overlap condition
constant overlap : R → R → ℝ

-- Prove that there's at least one overlap of area >= 1/9
theorem overlap_exists : 
  ∃ (r1 r2 ∈ small_rectangle_set), r1 ≠ r2 ∧ overlap r1 r2 ≥ (1 / 9) := sorry

end overlap_exists_l62_62784


namespace production_average_l62_62340

-- Define the conditions and question
theorem production_average (n : ℕ) (P : ℕ) (P_new : ℕ) (h1 : P = n * 70) (h2 : P_new = P + 90) (h3 : P_new = (n + 1) * 75) : n = 3 := 
by sorry

end production_average_l62_62340


namespace month_length_l62_62046

def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def total_cost : ℝ := 6

theorem month_length : (total_cost / cost_per_treat) / treats_per_day = 30 := by
  sorry

end month_length_l62_62046


namespace problem_1_problem_2_problem_3_l62_62513

open Complex

def z (n : ℕ) : ℂ :=
  if n = 1 then 3 + 4 * Complex.I else (1 + Complex.I) * z (n - 1)

theorem problem_1 :
  z 2 = -1 + 7 * Complex.I ∧
  z 3 = -8 + 6 * Complex.I ∧
  z 4 = -14 - 2 * Complex.I :=
sorry

theorem problem_2 (k : ℕ) (h : k > 0) :
  ∀ (n : ℕ), n = 4 * k + 1 → ∃ λ : ℝ, λ ≠ 0 ∧ z n = λ * z 1 :=
sorry

theorem problem_3 :
  ∑ n in Finset.range 100, (z (n+1)).re * (z (n+1)).im = 1 - 2^100 :=
sorry

end problem_1_problem_2_problem_3_l62_62513


namespace angle_bisector_slope_l62_62883

theorem angle_bisector_slope (k : ℚ) : 
  (∀ x : ℚ, (y = 2 * x ∧ y = 4 * x) → (y = k * x)) → k = -12 / 7 :=
sorry

end angle_bisector_slope_l62_62883


namespace cube_root_of_27_l62_62106

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { rfl }

end cube_root_of_27_l62_62106


namespace grain_cracker_price_l62_62135

theorem grain_cracker_price 
  (P : ℝ)
  (H1 : (3 * P + 4 * 1.50 + 4 * 1.00) / 6 = 2.79) : 
  P = 2.25 :=
by
  have H2 : 3 * P + 4 * 1.5 + 4 * 1 = 3 * P + 6 + 4 := by sorry
  have H3 : 3 * P + 10 = 3 * P + 6 + 4 := by rw [H2]
  have H4 : (3 * P + 10) / 6 = 2.79 := by rw [H3, H1]
  have H5 : 3 * P + 10 = 16.74 := by linarith
  have H6 : 3 * P = 6.74 := by linarith
  have H7 : P = 2.25 := by linarith
  show P = 2.25 from H7

end grain_cracker_price_l62_62135


namespace sum_of_integers_l62_62592

theorem sum_of_integers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
  (h4 : a * b * c = 19683) (h5 : Nat.coprime a b) (h6 : Nat.coprime b c) (h7 : Nat.coprime a c) :
  a + b + c = 741 :=
sorry

end sum_of_integers_l62_62592


namespace pies_difference_l62_62551

def total_apple_pies : ℕ :=
  16 + 16 + 20 + 10 + 6

def total_cherry_pies : ℕ :=
  14 + 18 + 8 + 12

theorem pies_difference :
  total_apple_pies - total_cherry_pies = 16 :=
by
  have h_apple : total_apple_pies = 68 := by rfl
  have h_cherry : total_cherry_pies = 52 := by rfl
  rw [h_apple, h_cherry]
  norm_num
  sorry

end pies_difference_l62_62551


namespace rhombus_area_and_side_length_l62_62562

theorem rhombus_area_and_side_length (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 8) :
  let area := d1 * d2 / 2 in
  let side := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
  (area = 120 ∧ side = Real.sqrt 241) :=
by
  let area := d1 * d2 / 2;
  let side := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2);
  have h_area : area = 120 := sorry;
  have h_side : side = Real.sqrt 241 := sorry;
  exact ⟨h_area, h_side⟩

end rhombus_area_and_side_length_l62_62562


namespace min_distance_PS_l62_62535

theorem min_distance_PS (P Q R S : Type) [metric_space P]
  (dPQ : dist P Q = 12)
  (dQR : dist Q R = 7)
  (dRS : dist R S = 2) :
  ∃ (PS : ℝ), PS = 3 ∧ ∀ q r s, dist P q = dPQ → dist q r = dQR → dist r s = dRS → dist P s = PS :=
begin
  sorry
end

end min_distance_PS_l62_62535


namespace even_function_a_zero_l62_62435

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62435


namespace statement_B_statement_D_l62_62746

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem statement_B (x₁ x₂ : ℝ) (h1 : -π / 12 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 5 * π / 12) :
  f x₁ < f x₂ := sorry

theorem statement_D (x₁ x₂ x₃ : ℝ) (h1 : π / 3 ≤ x₁) (h2 : x₁ ≤ π / 2) (h3 : π / 3 ≤ x₂) (h4 : x₂ ≤ π / 2) (h5 : π / 3 ≤ x₃) (h6 : x₃ ≤ π / 2) :
  f x₁ + f x₂ - f x₃ > 2 := sorry

end statement_B_statement_D_l62_62746


namespace simplify_harmonic_quadratic_radical_l62_62538

noncomputable def harmonic_quadratic_radical := real.sqrt(11 + 2 * real.sqrt(28))

theorem simplify_harmonic_quadratic_radical :
  harmonic_quadratic_radical = 2 + real.sqrt(7) :=
by
  sorry

end simplify_harmonic_quadratic_radical_l62_62538


namespace cos_tan_values_l62_62346

theorem cos_tan_values (α : ℝ) (h : Real.sin α = -1 / 2) :
  (∃ (quadrant : ℕ), 
    (quadrant = 3 ∧ Real.cos α = -Real.sqrt 3 / 2 ∧ Real.tan α = Real.sqrt 3 / 3) ∨ 
    (quadrant = 4 ∧ Real.cos α = Real.sqrt 3 / 2 ∧ Real.tan α = -Real.sqrt 3 / 3)) :=
sorry

end cos_tan_values_l62_62346


namespace rectangle_square_ratio_l62_62085

theorem rectangle_square_ratio (l w s : ℝ) (h1 : 0.4 * l * w = 0.25 * s * s) : l / w = 15.625 :=
by
  sorry

end rectangle_square_ratio_l62_62085


namespace axial_symmetry_maps_line_to_line_and_plane_to_plane_lines_perpendicular_to_s_perpendicular_after_sym_planes_perpendicular_to_s_map_to_themselves_l62_62871

-- Define geometry setup
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

def line (p1 p2 : Point) : Set Point := 
  { p : Point | ∃ t : ℝ, p = ⟨p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y), p1.z + t * (p2.z - p1.z)⟩ }

noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

constant s : line k1 k2

-- Define symmetry transformation

def sym_transform (s : line k1 k2) (A : Point) : Point :=
if A ∈ s then A else 
let M := orthogonal_projection s A in -- Assuming we have an orthogonal projection function
⟨2 * M.x - A.x, 2 * M.y - A.y, 2 * M.z - A.z⟩

-- Theorems
theorem axial_symmetry_maps_line_to_line_and_plane_to_plane (A B : Point) (hA : A ∉ s) (hB : B ∉ s) :
  let A' := sym_transform s A
  let B' := sym_transform s B in
  line A B = line A' B' ∧ plane_of_two_lines (line A B) (line C D) = plane_of_two_lines (line A' B') (line C' D') :=
sorry

theorem lines_perpendicular_to_s_perpendicular_after_sym (l : line P Q) (h : is_perpendicular_to s l) :
  let l' := sym_transform_line s l in is_perpendicular_to s l' ∧ (l ∩ s ≠ ∅ → l = l') :=
sorry

theorem planes_perpendicular_to_s_map_to_themselves (plane : Plane) (h : is_perpendicular_to s plane) :
  let plane' := sym_transform_plane s plane in plane = plane' :=
sorry

end axial_symmetry_maps_line_to_line_and_plane_to_plane_lines_perpendicular_to_s_perpendicular_after_sym_planes_perpendicular_to_s_map_to_themselves_l62_62871


namespace correct_sum_of_integers_l62_62177

theorem correct_sum_of_integers (x y : ℕ) (h1 : x - y = 4) (h2 : x * y = 192) : x + y = 28 := by
  sorry

end correct_sum_of_integers_l62_62177


namespace proof_problem_l62_62828

open Real

-- Define the parabola points
def parabola (x : ℝ) := 4 * x

-- Define the condition and questions as a combined problem statement
theorem proof_problem
  (A B : ℝ × ℝ)
  (hA : A.1 = parabola (A.2))
  (hB : B.1 = parabola (B.2))
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (product_slopes : (A.2 / (A.1^2)) * (B.2 / (B.1^2)) = -4) : Prop :=
  (∥A - B∥ ≥ 4) ∧ 
  (∃ x : ℝ, x = parabola A.2 ∧ x = parabola B.2) ∧ 
  (let d := abs (-1 / sqrt (1 + (A.2 / B.2)^2)) in
  ∥A - B∥ * d / 2 ≥ 2)

-- Providing a placeholder proof
example : proof_problem :=
by
  sorry

end proof_problem_l62_62828


namespace even_function_a_eq_zero_l62_62440

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62440


namespace sequence_sum_correct_l62_62406

/-- Problem Statement:
Let {a_n} be a sequence such that a_n = 2^{b_n} 
and {b_n} is an arithmetic sequence. 
If a_9 * a_2009 = 4, what is the sum b_1 + b_2 + ... + b_2017?
-/

noncomputable def sequence_sum : ℕ := 
  let a : ℕ → ℝ := λ n, 2^(b n) in
  let b : ℕ → ℝ := λ n, sorry in  -- Assume a definition of b as an arithmetic sequence
  if h1 : a 9 * a 2009 = 4 
  then (b 1 + b 2017) / 2 * 2017  -- Calculate the sum
  else 0  -- Fallback value

theorem sequence_sum_correct (b : ℕ → ℝ) (d : ℝ) (h_arith: ∀ n, b (n + 1) = b n + d)
  (h_sum : (b 1 + b 2017) = 2) : 
  (b 1 + b 2 + b 3 + ... + b 2017) = 2017 := 
by
  sorry

end sequence_sum_correct_l62_62406


namespace camel_humps_l62_62934

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62934


namespace trapezoid_perimeter_l62_62794

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end trapezoid_perimeter_l62_62794


namespace number_of_fish_caught_second_time_l62_62477

-- Conditions
variables (N x : ℕ)
variables (tagged_initial caught_tagged : ℕ)
variable (approx_number_of_fish : ℕ)

-- Hypotheses
axiom tagged_initial_hyp : tagged_initial = 50
axiom caught_tagged_hyp : caught_tagged = 2
axiom approx_number_of_fish_hyp : approx_number_of_fish = 1250
axiom proportion_hyp : 2 * approx_number_of_fish = 50 * x

-- Theorem: How many fish were caught the second time?
theorem number_of_fish_caught_second_time : x = 50 :=
by
  -- Facts from the conditions
  have h1 : approx_number_of_fish = 1250 := rfl
  have h2 : 2 * approx_number_of_fish = 50 * x := proportion_hyp

  -- Substitute the known value
  rw [h1] at h2

  -- Simplify to find x
  have h3 : 2 * 1250 = 50 * x := h2
  have h4 : 2500 = 50 * x := h3
  have h5 : 50 * x = 2500 := by rwa Nat.mul_comm at h4 -- rearrange for canonical form
  have h6 : x = 50 := by rw [Nat.div_eq_self (Nat.eq_of_mul_eq_mul_left (Nat.zero_lt_succ 49) h5)]

  -- Conclude the theorem
  exact h6

end number_of_fish_caught_second_time_l62_62477


namespace graph_passes_through_point_l62_62893

theorem graph_passes_through_point {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) : 
  (2, 2) ∈ (λ (x : ℝ), (x, a^(x-2) + 1)) :=
sorry

end graph_passes_through_point_l62_62893


namespace line_x_intercept_l62_62648

theorem line_x_intercept (P Q : ℝ × ℝ) (hP : P = (2, 3)) (hQ : Q = (6, 7)) :
  ∃ x, (x, 0) = (-1, 0) ∧ ∃ (m : ℝ), m = (Q.2 - P.2) / (Q.1 - P.1) ∧ ∀ (x y : ℝ), y = m * (x - P.1) + P.2 := 
  sorry

end line_x_intercept_l62_62648


namespace total_amount_spent_l62_62218

theorem total_amount_spent
  (num_pens : ℕ) (num_pencils : ℕ)
  (avg_price_pen : ℕ) (avg_price_pencil : ℕ)
  (h1 : num_pens = 30) (h2 : num_pencils = 75)
  (h3 : avg_price_pen = 16) (h4 : avg_price_pencil = 2) 
  :
  num_pens * avg_price_pen + num_pencils * avg_price_pencil = 630 :=
by
  rw [h1, h2, h3, h4]
  exact calc
    30 * 16 + 75 * 2 = 480 + 150 : by norm_num
                ... = 630       : by norm_num

end total_amount_spent_l62_62218


namespace min_value_of_a2_b2_l62_62767

theorem min_value_of_a2_b2 (a b : ℝ) : 
  (∀ x : ℝ, (∃ r : ℕ, r = 3 ∧ binomial 6 r * a^(6-r) * b^r * x^(12 - 3*r) = 20 * x^3)) → a * b = 1 → a^2 + b^2 ≥ 2 := 
by
  sorry

end min_value_of_a2_b2_l62_62767


namespace domain_sqrt_log_l62_62703

noncomputable def f (x : ℝ) := real.sqrt (1 - 2 * real.cos x) + real.log (real.sin x - real.sqrt 2 / 2)

theorem domain_sqrt_log :
  { x : ℝ | 1 - 2 * real.cos x ≥ 0 ∧ real.sin x - real.sqrt 2 / 2 > 0 } =
  { x : ℝ | ∃ k : ℤ, (π / 3) + 2 * k * π ≤ x ∧ x < (3 * π / 4) + 2 * k * π } :=
by
  sorry

end domain_sqrt_log_l62_62703


namespace Jack_and_Jill_same_speed_l62_62496

-- Define Jack's speed
def Jack_speed (x : ℝ) := x^2 - 9*x - 18

-- Define Jill's speed as the simplified version of the given expression
def Jill_speed (x : ℝ) := (x^2 - 5*x - 66) / (x + 6)

-- Define the proof statement
theorem Jack_and_Jill_same_speed (x : ℝ) (h : Jill_speed x = Jack_speed x) : x = 7 → Jack_speed x = -4 :=
by
  intro hx
  rw [<-h, hx]
  sorry -- Proof omitted

end Jack_and_Jill_same_speed_l62_62496


namespace value_of_v3_using_horner_method_l62_62598

def f (x : ℝ) : ℝ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem value_of_v3_using_horner_method :
  let v0 := 2
  let v1 := v0 * 3 - 1
  let v2 := v1 * 3 + 3
  let v3 := v2 * 3
  v3 = 54 :=
by
  let v0 := 2
  let v1 := v0 * 3 - 1
  let v2 := v1 * 3 + 3
  let v3 := v2 * 3
  show v3 = 54

end value_of_v3_using_horner_method_l62_62598


namespace solve_a_pow4_plus_a_inv4_l62_62555

variable (a : ℝ) [invertible a]

theorem solve_a_pow4_plus_a_inv4
    (h : 5 = a + ⅟a) : a^4 + ⅟(a^4) = 527 := 
sorry

end solve_a_pow4_plus_a_inv4_l62_62555


namespace player_one_win_l62_62915

theorem player_one_win (total_coins : ℕ) (coins_taken_by_player1 : ℕ) 
  (player1_moves : ∀ coins_remaining : ℕ, (1 ≤ coins_remaining ∧ coins_remaining ≤ 99) → coins_remaining % 2 = 1)
  (player2_moves : ∀ coins_remaining : ℕ, (2 ≤ coins_remaining ∧ coins_remaining ≤ 100) → coins_remaining % 2 = 0) :
  total_coins = 2015 → coins_taken_by_player1 = 95 → 
  ∃ strategy, ∀ turn : ℕ, (turn % 2 = 1 → strategy turn coins_taken_by_player1 = true) ∧ (turn % 2 = 0 → strategy turn coins_taken_by_player1 = false) :=
begin
  sorry
end

end player_one_win_l62_62915


namespace r_plus_s_54_l62_62649

theorem r_plus_s_54 (p q r s t u : ℕ) 
  (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) (h5 : t < u) 
  (h6 : List.pairwise (λ x y, x < y) [p, q, r, s, t, u]) 
  (h7 : Multiset.map (λ (x : ℕ × ℕ), x.1 + x.2) 
        (Multiset.filter (λ x : ℕ × ℕ, x.1 < x.2) 
        (Multiset.pairwise_prod [p, q, r, s, t, u])) 
        = [25, 30, 38, 41, 49, 52, 54, 63, 68, 76, 79, 90, 95, 103, 117]) :
  r + s = 54 := 
sorry

end r_plus_s_54_l62_62649


namespace polar_eq_area_l62_62579

theorem polar_eq_area {θ : ℝ} (h : ∀ θ, ρ = 2 * sqrt 2 * cos (π / 4 - θ)) : 
  let ρ := 2 * sqrt 2 * cos (π / 4 - θ) in 
  ∃ r : ℝ, r = 2 ∧ ρ^2 = r^2 :=
sorry

end polar_eq_area_l62_62579


namespace find_length_of_cd_l62_62901

-- Definitions and conditions related to the problem
def cylinder_volume (r h : ℝ) := π * r^2 * h
def hemisphere_volume (r : ℝ) := (2/3) * π * r^3
def total_volume (r h : ℝ) := cylinder_volume r h + 2 * hemisphere_volume r

-- Given volume V = 1024/3 * π, and radius r = 4, find the length h such that the total volume matches V
theorem find_length_of_cd (L : ℝ) (h : ℝ) : total_volume 4 L = (1024 / 3) * π → L = 16 :=
by
  intros h_eq;
  sorry

end find_length_of_cd_l62_62901


namespace range_of_s_l62_62716

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ n = p * q

def s (n : ℕ) : ℕ :=
  if h : n = 1 then 0 -- Since 1 is not composite, we define s(1) to be 0
  else let primes := n.factorization in
    2 * (finset.sum primes.support (λ p, (primes p) * 2 * p))

theorem range_of_s : 
  ∀ n, is_composite n → ∃ m : ℕ, m ≥ 8 ∧ s(n) = m :=
by 
  -- Proof goes here
  sorry

end range_of_s_l62_62716


namespace cube_root_of_27_l62_62109

theorem cube_root_of_27 : 
  ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
begin
  sorry
end

end cube_root_of_27_l62_62109


namespace total_winning_team_points_l62_62262

/-!
# Lean 4 Math Proof Problem

Prove that the total points scored by the winning team at the end of the game is 50 points given the conditions provided.
-/

-- Definitions
def losing_team_points_first_quarter : ℕ := 10
def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter
def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + 10
def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

-- Theorem statement
theorem total_winning_team_points : winning_team_points_third_quarter = 50 :=
by
  sorry

end total_winning_team_points_l62_62262


namespace part1_part2_l62_62395

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + a * x^2
noncomputable def m (x : ℝ) (a : ℝ) := (f x a).deriv
noncomputable def m' (x : ℝ) (a : ℝ) := (m x a).deriv
noncomputable def g (x : ℝ) (a : ℝ) := f x a - a * x^2 + a * x
noncomputable def g' (x : ℝ) (a : ℝ) := (g x a).deriv

-- Part 1
theorem part1 (m1_eq_3 : m' 1 a = 3) : a = 2 :=
by
  sorry

-- Part 2
theorem part2 (mono_incr : ∀ x, 0 < x → g' x a ≥ 0) : a ≥ 0 :=
by
  sorry

end part1_part2_l62_62395


namespace line_through_midpoint_l62_62388

theorem line_through_midpoint (x y : ℝ)
  (ellipse : x^2 / 25 + y^2 / 16 = 1)
  (midpoint : P = (2, 1)) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (x = 32*y - 25*x - 89) :=
sorry

end line_through_midpoint_l62_62388


namespace num_possible_values_l62_62929

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62929


namespace sum_sequence_geq_n_l62_62553

theorem sum_sequence_geq_n (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ k, a (k+1) ≥ (k * a k) / (a k ^ 2 + (k - 1))) :
  ∀ n, n ≥ 2 → (∑ i in Finset.range n, a (i + 1)) ≥ n :=
by
  sorry

end sum_sequence_geq_n_l62_62553


namespace bushels_given_away_l62_62673

-- Definitions from the problem conditions
def initial_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

-- Theorem to prove the number of bushels given away
theorem bushels_given_away : 
  initial_bushels * ears_per_bushel - remaining_ears = 24 * ears_per_bushel :=
by
  sorry

end bushels_given_away_l62_62673


namespace factorization_of_m_squared_minus_4_l62_62311

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l62_62311


namespace cubic_three_positive_integer_roots_l62_62317

theorem cubic_three_positive_integer_roots (p : ℝ) :
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
   a ∈ ℤ ∧ b ∈ ℤ ∧ c ∈ ℤ ∧
   5 * a^3 - 5 * (p + 1) * a^2 + (71 * p - 1) * a + 1 = 66 * p ∧
   5 * b^3 - 5 * (p + 1) * b^2 + (71 * p - 1) * b + 1 = 66 * p ∧
   5 * c^3 - 5 * (p + 1) * c^2 + (71 * p - 1) * c + 1 = 66 * p)
    ↔ p = 76 :=
sorry

end cubic_three_positive_integer_roots_l62_62317


namespace question_l62_62616

-- Define the conditions of the problem
variables {A B C D E : Type} [EuclideanGeometry.Triangle A B C D E]
variables (T t : ℝ)
variables (CB CD AC AB AD AE : ℝ)

-- Given conditions
axiom cond1 : ∠ACD = ∠DBC
axiom cond2 : T = EuclideanGeometry.area_triangle A B C
axiom cond3 : t = EuclideanGeometry.area_triangle A C D
axiom cond4 : EuclideanGeometry.similar_ABC_ACD T t CB CD
axiom cond5 : EuclideanGeometry.common_height_A_C_B_D T t AB AD
axiom cond6 : CB^2 = AC^2 + AB^2 - 2 * AB * AE
axiom cond7 : CD^2 = AC^2 + AD^2 - 2 * AD * AE

-- The statement we need to prove
theorem question : AC^2 = AB * AD :=
sorry

end question_l62_62616


namespace compare_negative_fractions_l62_62281

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end compare_negative_fractions_l62_62281


namespace abs_diff_gk_le_factorial_l62_62373

noncomputable def g (n : ℕ) : ℝ :=
  1 / (2 + (1 / (3 + 1 / (4 + … / (n - 1)))))

noncomputable def k (n : ℕ) : ℝ :=
  1 / (2 + (1 / (3 + 1 / (4 + … / (n - 1 + 1 / n)))))

theorem abs_diff_gk_le_factorial (n : ℕ) : 
  abs (g n - k n) ≤ 1 / ((n - 1)! * n!) :=
sorry

end abs_diff_gk_le_factorial_l62_62373


namespace euler_totient_formula_l62_62066

-- Define the Euler's totient function φ(n)
def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (Nat.coprime n).length

-- Define the formula to be proven
theorem euler_totient_formula (n : ℕ) (p : ℕ → ℕ) (a : ℕ → ℕ) (k : ℕ) 
  (h1 : n = (List.prod (List.concatMap (λ i, List.repeat (p i) (a i)) (List.range k))))
  (h2 : ∀ i < k, Nat.Prime (p i)) :
  euler_totient n = n * List.prod (List.map (λ i, 1 - 1 / (p i)) (List.range k)) := by
  sorry

end euler_totient_formula_l62_62066


namespace condition_P_condition_q_false_correct_choice_C_l62_62720

theorem condition_P : ∀ x : ℝ, x^2 - 4 * x + 5 > 0 := by
  intros x
  calc
    x^2 - 4 * x + 5 = (x - 2)^2 + 1 := by ring
    _ > 0 := by linarith

theorem condition_q_false : ¬ (∃ x : ℝ, 0 < x ∧ cos x > 1) := by
  intro h
  cases h with x hx
  cases hx with h1 h2
  have cos_le_one : cos x ≤ 1 := by apply real.cos_le
  linarith

theorem correct_choice_C : (condition_P ∨ ¬condition_q_false) := by
  left
  exact condition_P

end condition_P_condition_q_false_correct_choice_C_l62_62720


namespace find_a_if_f_even_l62_62447

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62447


namespace possible_values_of_N_count_l62_62943

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62943


namespace sequence_an_correct_l62_62034

noncomputable def seq_an (n : ℕ) : ℚ :=
if h : n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3))

theorem sequence_an_correct (n : ℕ) (S : ℕ → ℚ)
  (h1 : S 1 = 1)
  (h2 : ∀ n ≥ 2, S n ^ 2 = seq_an n * (S n - 0.5)) :
  seq_an n = if n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3)) :=
sorry

end sequence_an_correct_l62_62034


namespace price_on_hot_day_l62_62088

noncomputable def regular_price_P (P : ℝ) : Prop :=
  7 * 32 * (P - 0.75) + 3 * 32 * (1.25 * P - 0.75) = 450

theorem price_on_hot_day (P : ℝ) (h : regular_price_P P) : 1.25 * P = 2.50 :=
by sorry

end price_on_hot_day_l62_62088


namespace problem_statement_l62_62023

noncomputable def number_of_points_C (A B : ℝ × ℝ) (AB_distance perimeter area : ℝ) : ℕ :=
  if H1 : AB_distance = dist A B
  then 
    let points_C := {C : ℝ × ℝ | dist A B + dist B C + dist C A = perimeter ∧ 1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = area} in
    if finite points_C
    then cardinal_to_nat (card points_C)
    else 0
  else 0

theorem problem_statement : number_of_points_C (0, 0) (12, 0) 12 60 72 = 4 := sorry

end problem_statement_l62_62023


namespace segment_bisection_l62_62504

variables (A B C D E : Type) 
variables [EuclideanGeometry A B C D E]
variables (AB BC : Prop)
variable (Γ : Circle ABC)
variables [IsoscelesTriangle B AB BC]
variables (TangentsIntersect : TangentsIntersection A B Γ D)
variables (SecondIntersection : SecondIntersection D C Γ E)

theorem segment_bisection 
  (hIso : IsoscelesTriangle B AB BC)
  (hTangents : TangentsIntersection A B Γ D)
  (hSecond : SecondIntersection D C Γ E) :
  bisects AE DB :=
sorry

end segment_bisection_l62_62504


namespace point_on_transformed_plane_l62_62629

-- Point A with coordinates (4, 3, 1)
def pointA : ℝ × ℝ × ℝ := (4, 3, 1)

-- Plane equation: 3x - 4y + 5z - 6 = 0
def planeEq (x y z : ℝ) : Prop := 3 * x - 4 * y + 5 * z - 6 = 0

-- Similarity transformation coefficient
def k : ℝ := 5 / 6

-- Transformed plane equation: 3x - 4y + 5z - k * 6 = 0 -> 3x - 4y + 5z - 5 = 0
def transformedPlaneEq (x y z : ℝ) : Prop := 3 * x - 4 * y + 5 * z - 5 = 0

-- The proof problem statement
theorem point_on_transformed_plane : 
  let (x, y, z) := pointA in transformedPlaneEq x y z :=
by 
  -- substitute the coordinates and simplify
  sorry

end point_on_transformed_plane_l62_62629


namespace average_income_independence_l62_62472

theorem average_income_independence (A E : ℝ) (n : ℕ) (h : n = 10) :
  let avg_income := (A + E) / (n : ℝ)
  in avg_income = (A + E) / 10 :=
by
  intros
  have h1 : (n : ℝ) = 10 := by simp [h]
  rw h1
  sorry

end average_income_independence_l62_62472


namespace gcf_72_108_l62_62186

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62186


namespace parallel_vectors_tan_l62_62407

/-- Given vector a and vector b, and given the condition that a is parallel to b,
prove that the value of tan α is 1/4. -/
theorem parallel_vectors_tan (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.sin α, Real.cos α - 2 * Real.sin α))
  (hb : b = (1, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) : 
  Real.tan α = 1 / 4 := 
by 
  sorry

end parallel_vectors_tan_l62_62407


namespace tiling_problem_solution_l62_62851

-- Definition of the problem scenario
def room_length := 30
def room_width := 20
def short_edge_border_length := 20
def one_foot_tile_count := 2 * short_edge_border_length
def pillar_side := 2
def central_area_length := room_length
def central_area_width := room_width - (2 * (pillar_side / pillar_side))
def central_area_excluding_pillar := (central_area_length * central_area_width) - (pillar_side * pillar_side)
def three_foot_tile_area := 3 * 3
def three_foot_tile_count := (central_area_excluding_pillar / three_foot_tile_area).ceil
def total_tile_count := one_foot_tile_count + three_foot_tile_count

-- The statement to prove
theorem tiling_problem_solution : total_tile_count = 100 := by
  -- Placeholder for the proof
  sorry

end tiling_problem_solution_l62_62851


namespace factorization_of_m_squared_minus_4_l62_62310

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l62_62310


namespace bread_baked_on_monday_l62_62882

def loaves_wednesday : ℕ := 5
def loaves_thursday : ℕ := 7
def loaves_friday : ℕ := 10
def loaves_saturday : ℕ := 14
def loaves_sunday : ℕ := 19

def increment (n m : ℕ) : ℕ := m - n

theorem bread_baked_on_monday : 
  increment loaves_wednesday loaves_thursday = 2 →
  increment loaves_thursday loaves_friday = 3 →
  increment loaves_friday loaves_saturday = 4 →
  increment loaves_saturday loaves_sunday = 5 →
  loaves_sunday + 6 = 25 :=
by 
  sorry

end bread_baked_on_monday_l62_62882


namespace elena_alex_total_dollars_l62_62695

theorem elena_alex_total_dollars :
  (5 / 6 : ℚ) + (7 / 15 : ℚ) = (13 / 10 : ℚ) :=
by
    sorry

end elena_alex_total_dollars_l62_62695


namespace net_gain_for_Mr_C_l62_62852

theorem net_gain_for_Mr_C :
  let initial_worth := 15000
  let selling_price_to_D := 1.20 * initial_worth
  let buying_price_from_D := 0.85 * selling_price_to_D
  let transaction_fee := 300
  let total_cost := buying_price_from_D + transaction_fee
  let net_gain := selling_price_to_D - total_cost
  net_gain = 2400 :=
by
  let initial_worth := 15000
  let selling_price_to_D := 1.20 * initial_worth
  let buying_price_from_D := 0.85 * selling_price_to_D
  let transaction_fee := 300
  let total_cost := buying_price_from_D + transaction_fee
  let net_gain := selling_price_to_D - total_cost
  have h : net_gain = 2400 := by sorry
  exact h

end net_gain_for_Mr_C_l62_62852


namespace jerry_total_hours_at_field_l62_62816
-- Import the entire necessary library

-- Lean statement of the problem
theorem jerry_total_hours_at_field 
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (daughters : ℕ)
  (h1: games_per_daughter = 8)
  (h2: practice_hours_per_game = 4)
  (h3: game_duration = 2)
  (h4: daughters = 2)
 : (game_duration * games_per_daughter * daughters + practice_hours_per_game * games_per_daughter * daughters) = 96 :=
by
  -- Proof not required, so we skip it with sorry
  sorry

end jerry_total_hours_at_field_l62_62816


namespace find_y_l62_62198

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 12) (h2 : x = 6) : y = 3 :=
by
  sorry

end find_y_l62_62198


namespace combinatorial_group_l62_62083

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem combinatorial_group (n : ℕ) (h : choose (2 * n) n > n^2 + 1) :
  ∃ (g : Finset ℕ), g.card = n + 1 ∧ (∀ x ∈ g, ∀ y ∈ g, x ≠ y → (knows x y ∨ ¬knows x y))
 := sorry

end combinatorial_group_l62_62083


namespace min_E_of_tan_xy_l62_62501

noncomputable def E (x y : ℝ) : ℝ :=
  Real.cos x + Real.cos y

theorem min_E_of_tan_xy {x y m : ℝ} (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2) (hm : 2 < m) (hxy : Real.tan x * Real.tan y = m) :
  ∃ c, E x y = c ∧ ∀ z, z = E x y → z ≥ 2 :=
begin
  sorry
end

end min_E_of_tan_xy_l62_62501


namespace valid_number_of_two_humped_camels_within_range_l62_62964

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62964


namespace rhombus_diagonals_length_l62_62993

variable (rhombus : Type) [MetricSpace rhombus]
variable (side_length : ℝ)
variable (angle : ℝ)
variable (d1 d2 : ℝ)

def is_rhombus (side_length : ℝ) (angle : ℝ) : Prop :=
  ∃ (r : rhombus), ∀ (s : rhombus), (Metric.dist r s = side_length) ∧ (angle = 120)

theorem rhombus_diagonals_length (r : rhombus) (h_rhombus : is_rhombus r 1 120) :
  ∃ (d1 d2 : ℝ), d1 = 1 ∧ d2 = Real.sqrt 3 :=
sorry

end rhombus_diagonals_length_l62_62993


namespace find_a_l62_62359

theorem find_a (a : ℝ) : (∃ z : ℂ, z = (a + complex.i) / (2 - complex.i) ∧ z.im = 0) ↔ a = -2 := by
  sorry

end find_a_l62_62359


namespace gcd_72_108_l62_62190

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l62_62190


namespace length_of_second_train_l62_62252

theorem length_of_second_train
  (L1 : ℝ) (V1 : ℝ) (V2 : ℝ) (D : ℝ) (T : ℝ)
  (hL1 : L1 = 100)
  (hV1 : V1 = 10)
  (hV2 : V2 = 15)
  (hD : D = 50)
  (hT : T = 60) :
  ∃ L2 : ℝ, L2 = 150 :=
by
  use 150
  sorry

end length_of_second_train_l62_62252


namespace max_acute_angles_convex_polygon_l62_62995

theorem max_acute_angles_convex_polygon (n : ℕ) (h₁ : 3 ≤ n) :
  ∀ (θ : Fin n → ℝ), 
    (∀ i, 0 < θ i ∧ θ i < 180) ∧ 
    (Finset.univ.sum θ = (n - 2) * 180) → 
  (Finset.card {i | θ i < 90} ≤ 3) := 
by 
  intro θ h
  sorry

end max_acute_angles_convex_polygon_l62_62995


namespace consistency_condition_l62_62865

theorem consistency_condition (x y z a b c d : ℝ)
  (h1 : y + z = a)
  (h2 : x + y = b)
  (h3 : x + z = c)
  (h4 : x + y + z = d) : a + b + c = 2 * d :=
by sorry

end consistency_condition_l62_62865


namespace cinnamon_swirl_eaters_l62_62714

theorem cinnamon_swirl_eaters (total_pieces : ℝ) (jane_pieces : ℝ) (equal_pieces : total_pieces / jane_pieces = 3 ) : 
  (total_pieces = 12) ∧ (jane_pieces = 4) → total_pieces / jane_pieces = 3 := 
by 
  sorry

end cinnamon_swirl_eaters_l62_62714


namespace possible_values_of_N_count_l62_62949

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62949


namespace number_of_tiles_needed_l62_62759

def tile_size : ℕ := 15 * 15
def wall_length_cm : ℕ := 3 * 100 + 6 * 10
def wall_width_cm : ℕ := 27 * 10
def wall_area_cm2 : ℕ := wall_length_cm * wall_width_cm

theorem number_of_tiles_needed :
  wall_area_cm2 / tile_size = 432 := by
  unfold wall_length_cm wall_width_cm wall_area_cm2 tile_size
  -- The necessary conversions and calculations would follow here
  sorry

end number_of_tiles_needed_l62_62759


namespace stock_and_bond_value_relation_l62_62850

-- Definitions for conditions
def more_valuable_shares : ℕ := 14
def less_valuable_shares : ℕ := 26
def face_value_bond : ℝ := 1000
def coupon_rate_bond : ℝ := 0.06
def discount_rate_bond : ℝ := 0.03
def total_assets_value : ℝ := 2106

-- Lean statement for the proof problem
theorem stock_and_bond_value_relation (x y : ℝ) 
    (h1 : face_value_bond * (1 - discount_rate_bond) = 970)
    (h2 : 27 * x + y = total_assets_value) :
    y = 2106 - 27 * x :=
by
  sorry

end stock_and_bond_value_relation_l62_62850


namespace possible_N_values_l62_62980

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62980


namespace sugar_in_first_combination_l62_62104

def cost_per_pound : ℝ := 0.45
def cost_combination_1 (S : ℝ) : ℝ := cost_per_pound * S + cost_per_pound * 16
def cost_combination_2 : ℝ := cost_per_pound * 30 + cost_per_pound * 25
def total_weight_combination_2 : ℕ := 30 + 25
def total_weight_combination_1 (S : ℕ) : ℕ := S + 16

theorem sugar_in_first_combination :
  ∀ (S : ℕ), cost_combination_1 S = 26 ∧ cost_combination_2 = 26 → total_weight_combination_1 S = total_weight_combination_2 → S = 39 :=
by sorry

end sugar_in_first_combination_l62_62104


namespace camel_humps_l62_62936

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62936


namespace rustling_leaves_sound_level_l62_62887

theorem rustling_leaves_sound_level :
  ∀ (I I0 : ℝ), I = 10^(-12) ∧ I0 = 10^(-12) →
  ∃ β : ℝ, β = 10 * log (I / I0) ∧ β = 0 :=
by
  intros I I0 h
  obtain ⟨hI, hI0⟩ := h
  use 10 * log (I / I0)
  split
  { sorry }
  { sorry }

end rustling_leaves_sound_level_l62_62887


namespace caravan_humps_l62_62973

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62973


namespace sector_area_l62_62013

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = π / 3) (hr : r = 4) : 
  (1/2) * (r * θ) * r = 8 * π / 3 :=
by
  -- Implicitly use the given values of θ and r by substituting them in the expression.
  sorry

end sector_area_l62_62013


namespace perpendicular_lines_exist_l62_62010

-- Definition of line "a" not being perpendicular to plane "\alpha"
def not_perpendicular (a : Line) (α : Plane) : Prop :=
  ¬ ∀ (l : LineInPlane α), a.is_perpendicular l

-- Definition of "countless" in this context (let's define it as infinite)
def countless_perpendicular_lines (a : Line) (α : Plane) : Prop :=
  ∃ (S : Set (LineInPlane α)), S.countable ∧ ∀ l ∈ S, a.is_perpendicular l

-- Theorem: If line "a" is not perpendicular to plane "\alpha", then there are countless lines in plane "\alpha" which are perpendicular to "a".
theorem perpendicular_lines_exist (a : Line) (α : Plane)
  (h : not_perpendicular a α) : countless_perpendicular_lines a α :=
by
  sorry

end perpendicular_lines_exist_l62_62010


namespace tangent_line_parallel_coordinates_l62_62586

theorem tangent_line_parallel_coordinates :
  ∃ (x y : ℝ), y = x^3 + x - 2 ∧ (3 * x^2 + 1 = 4) ∧ (x, y) = (-1, -4) :=
by
  sorry

end tangent_line_parallel_coordinates_l62_62586


namespace part_a_part_b_l62_62623

-- Define the setup for the problems
variable (R : Type) [LinearOrderedField R]
variable (r : R) -- the radius of the circle

-- The regular dodecagon area problem
def regular_dodecagon_area_inscribed (r : R) : R := 3 * r * r / 2 * (Real.sqrt 3)

theorem part_a (H : ∀ (square_side : R),
    ∀ (dodecagon_area : R), 
    dodecagon_area > 0 →
    (4 * (square_side ^ 2) = dodecagon_area) →
    (4 * (square_side ^ 2)) / 12 = dodecagon_area / 12
  ) : Prop := sorry

theorem part_b : regular_dodecagon_area_inscribed 1 = 3 := sorry

end part_a_part_b_l62_62623


namespace largest_multiple_of_7_less_than_100_l62_62603

theorem largest_multiple_of_7_less_than_100 : ∃ (n : ℕ), n * 7 < 100 ∧ ∀ (m : ℕ), m * 7 < 100 → m * 7 ≤ n * 7 :=
  by
  sorry

end largest_multiple_of_7_less_than_100_l62_62603


namespace ratio_of_interests_l62_62908

noncomputable def simple_interest (P R T : ℝ) := (P * R * T) / 100

noncomputable def compound_interest (P R T : ℝ) := P * ((1 + R / 100) ^ T - 1)

theorem ratio_of_interests :
  let SI := simple_interest 1750 8 3 in 
  let CI := compound_interest 4000 10 2 in 
  SI / CI = 1 / 2 := 
by 
  sorry

end ratio_of_interests_l62_62908


namespace num_positive_integers_l62_62717

-- Definitions
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Problem statement
theorem num_positive_integers (n : ℕ) (h : n = 2310) :
  (∃ count, count = 3 ∧ (∀ m : ℕ, m > 0 → is_divisor (m^2 - 2) n → count = 3)) := by
  sorry

end num_positive_integers_l62_62717


namespace correct_average_marks_l62_62628

theorem correct_average_marks :
  ∀ (n : ℕ) (incorrect_avg correct_mark wrong_mark : ℝ),
  n = 25 →
  incorrect_avg = 100 →
  correct_mark = 10 →
  wrong_mark = 60 →
  let incorrect_total := incorrect_avg * n in
  let difference := wrong_mark - correct_mark in
  let correct_total := incorrect_total - difference in
  let correct_avg := correct_total / n in
  correct_avg = 98 :=
begin
  intros n incorrect_avg correct_mark wrong_mark hn h_avg h_corr h_wrong,
  dsimp only,
  rw [hn, h_avg, h_corr, h_wrong],
  dsimp only [incorrect_total, difference, correct_total, correct_avg],
  norm_num,
end

end correct_average_marks_l62_62628


namespace sum_of_four_consecutive_integers_prime_factor_l62_62159

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l62_62159


namespace arithmetic_square_root_of_16_l62_62101

theorem arithmetic_square_root_of_16 : ∃! (x : ℝ), x^2 = 16 ∧ x ≥ 0 :=
by
  sorry

end arithmetic_square_root_of_16_l62_62101


namespace camel_humps_l62_62940

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62940


namespace work_completed_in_days_l62_62206

theorem work_completed_in_days (time_a time_b time_c : ℕ)
  (ha : time_a = 15) (hb : time_b = 20) (hc : time_c = 45) :
  let combined_work_rate := (1 / rat.of_int time_a) + 
                            (1 / rat.of_int time_b) + 
                            (1 / rat.of_int time_c) in
  let days_to_complete_work := 1 / combined_work_rate in
  days_to_complete_work = 7.2 := 
by
  sorry

end work_completed_in_days_l62_62206


namespace solution_set_of_inequality_l62_62584

theorem solution_set_of_inequality :
  { x : ℝ | x ^ 2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
by 
  sorry

end solution_set_of_inequality_l62_62584


namespace possible_N_values_l62_62976

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62976


namespace solve_eq1_solve_eq2_l62_62093

theorem solve_eq1 (x : ℝ) : (x^2 - 2 * x - 8 = 0) ↔ (x = 4 ∨ x = -2) :=
sorry

theorem solve_eq2 (x : ℝ) : (2 * x^2 - 4 * x + 1 = 0) ↔ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
sorry

end solve_eq1_solve_eq2_l62_62093


namespace Kaleb_second_half_points_l62_62030

theorem Kaleb_second_half_points (first_half_points total_points : ℕ) (h1 : first_half_points = 43) (h2 : total_points = 66) : total_points - first_half_points = 23 := by
  sorry

end Kaleb_second_half_points_l62_62030


namespace even_function_a_zero_l62_62422

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62422


namespace valid_number_of_two_humped_camels_within_range_l62_62965

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62965


namespace trajectory_of_Q_l62_62743

variables {P Q M : ℝ × ℝ}

-- Define the conditions as Lean predicates
def is_midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (0, 4) ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 2 = 0

-- Define the theorem that needs to be proven
theorem trajectory_of_Q :
  (∃ P Q M : ℝ × ℝ, is_midpoint M P Q ∧ point_on_line P) →
  ∃ Q : ℝ × ℝ, (∀ P : ℝ × ℝ, point_on_line P → is_midpoint (0,4) P Q → Q.1 + Q.2 - 6 = 0) :=
by sorry

end trajectory_of_Q_l62_62743


namespace event_C_is_certain_l62_62777

-- Definitions
def black_balls : ℕ := 2
def white_balls : ℕ := 1
def total_balls : ℕ := black_balls + white_balls

-- Event C: Drawing at least one black ball
def certain_event_C : Prop := 
  ∀ (drawn_balls : Finset (Fin total_balls)), 
    drawn_balls.card = 2 → 
    ∃ (b ∈ drawn_balls), b < black_balls

theorem event_C_is_certain : certain_event_C :=
sorry

end event_C_is_certain_l62_62777


namespace possible_N_values_l62_62979

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62979


namespace vacuum_upstairs_more_than_twice_downstairs_l62_62494

theorem vacuum_upstairs_more_than_twice_downstairs 
  (x y : ℕ) 
  (h1 : 27 = 2 * x + y) 
  (h2 : x + 27 = 38) : 
  y = 5 :=
by 
  sorry

end vacuum_upstairs_more_than_twice_downstairs_l62_62494


namespace possible_shapes_after_reflection_l62_62614

def Triangle (A B C : Type) := inhabited (A) ∧ inhabited (B) ∧ inhabited (C)

def reflect_triangle (T : Triangle ℝ ℝ ℝ) (side : ℝ × ℝ) : Type :=
  match T with
  | ⟨inhabited.mk A, inhabited.mk B, inhabited.mk C⟩ =>
    if side = (B, C) then -- reflecting across BC
      inhabited (A') -- A' is some transformation of A due to reflection

-- Proving possible shapes
theorem possible_shapes_after_reflection (T : Triangle ℝ ℝ ℝ) (side : ℝ × ℝ) :
  reflect_triangle T side ∈ {deltoid, concave_deltoid, isosceles_triangle, rhombus, square} :=
sorry

end possible_shapes_after_reflection_l62_62614


namespace sequence_exceeds_10000_l62_62889

theorem sequence_exceeds_10000 :
  ∃ n, 1 ≤ n ∧ (∀ k, 1 ≤ k ∧ k < n → (let a : ℕ → ℕ := λ n, if n = 1 then 1 else (∑ i in range (n-1 + 1), a i) + 1 in a k ≤ 10000)) ∧ 
    (let a : ℕ → ℕ := λ n, if n = 1 then 1 else (∑ i in range (n-1 + 1), a i) + 1 in a n > 10000) ∧
    (let a : ℕ → ℕ := λ n, if n = 1 then 1 else (∑ i in range (n-1 + 1), a i) + 1 in a n = 16384) :=
sorry

end sequence_exceeds_10000_l62_62889


namespace radius_perpendicular_to_tangent_l62_62631

theorem radius_perpendicular_to_tangent
  (O M : Point) -- Center of the circle and point of tangency
  (circle : Circle O)
  (line : Line)
  (H1 : is_tangent line circle M)
  : is_perpendicular (Line.mk O M) line := 
sorry

end radius_perpendicular_to_tangent_l62_62631


namespace john_total_spent_l62_62045

def silver_ounces : ℝ := 2.5
def silver_price_per_ounce : ℝ := 25
def gold_ounces : ℝ := 3.5
def gold_price_multiplier : ℝ := 60
def platinum_ounces : ℝ := 4.5
def platinum_price_per_ounce_gbp : ℝ := 80
def palladium_ounces : ℝ := 5.5
def palladium_price_per_ounce_eur : ℝ := 100

def usd_per_gbp_monday : ℝ := 1.3
def usd_per_gbp_friday : ℝ := 1.4
def usd_per_eur_wednesday : ℝ := 1.15
def usd_per_eur_saturday : ℝ := 1.2

def discount_rate : ℝ := 0.05
def tax_rate : ℝ := 0.08

def total_amount_john_spends_usd : ℝ := 
  (silver_ounces * silver_price_per_ounce * (1 - discount_rate)) + 
  (gold_ounces * (gold_price_multiplier * silver_price_per_ounce) * (1 - discount_rate)) + 
  (((platinum_ounces * platinum_price_per_ounce_gbp) * (1 + tax_rate)) * usd_per_gbp_monday) + 
  ((palladium_ounces * palladium_price_per_ounce_eur) * usd_per_eur_wednesday)

theorem john_total_spent : total_amount_john_spends_usd = 6184.815 := by
  sorry

end john_total_spent_l62_62045


namespace prime_factor_of_sum_of_four_consecutive_integers_l62_62171

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l62_62171


namespace geometric_series_sum_example_l62_62676

-- Define the finite geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- State the theorem
theorem geometric_series_sum_example :
  geometric_series_sum (1/2) (1/2) 8 = 255 / 256 :=
by
  sorry

end geometric_series_sum_example_l62_62676


namespace solve_linear_system_l62_62549

theorem solve_linear_system:
  ∃ (x y z: ℝ), 
    x^2 - 22*y - 69*z + 703 = 0 ∧
    y^2 + 23*x + 23*z - 1473 = 0 ∧
    z^2 - 63*x + 66*y + 2183 = 0 ∧
    x = 20 ∧ y = -22 ∧ z = 23 :=
by
  use 20, -22, 23
  split
  {
    calc (20: ℝ)^2 - 22 * (-22) - 69 * 23 + 703 = 
      400 + 484 - 1587 + 703 : by norm_num
    ... = 0 : by norm_num
  }
  split
  {
    calc (-22: ℝ)^2 + 23 * 20 + 23 * 23 - 1473 =
      484 + 460 + 529 - 1473 : by norm_num
    ... = 0 : by norm_num
  }
  {
    calc (23: ℝ)^2 - 63 * 20 + 66 * (-22) + 2183 =
      529 - 1260 - 1452 + 2183 : by norm_num
    ... = 0 : by norm_num
  }
  sorry

end solve_linear_system_l62_62549


namespace circle_radius_zero_l62_62708

theorem circle_radius_zero : ∀ (x y : ℝ), x^2 + 10 * x + y^2 - 4 * y + 29 = 0 → 0 = 0 :=
by intro x y h
   sorry

end circle_radius_zero_l62_62708


namespace magnitude_of_p_plus_qi_l62_62741

theorem magnitude_of_p_plus_qi (p q : ℝ) (hp : Polynomial.root (Polynomial.X^2 + Polynomial.C p * Polynomial.X + Polynomial.C q) (Complex.ofReal 1 + Complex.I)) :
    |Complex.mk p q| = 2 * Real.sqrt 2 :=
by
  -- The theorem's proof would go here
  sorry

end magnitude_of_p_plus_qi_l62_62741


namespace necessary_and_sufficient_condition_l62_62633

noncomputable def line_perpendicular_condition (l : Type) (a : Type) : Prop := 
  ∀ (countless_lines : set Type), (∀ line ∈ countless_lines, l ⊥ line) → l ⊥ a

theorem necessary_and_sufficient_condition 
    (l : Type) (a : Type) (countless_lines : set Type) :
    (∀ line ∈ countless_lines, l ⊥ line) ↔ l ⊥ a :=
by sorry

end necessary_and_sufficient_condition_l62_62633


namespace max_elements_X_l62_62052

structure GameState where
  fire : Nat
  stone : Nat
  metal : Nat

def canCreateX (state : GameState) (x : Nat) : Bool :=
  state.metal >= x ∧ state.fire >= 2 * x ∧ state.stone >= 3 * x

def maxCreateX (state : GameState) : Nat :=
  if h : canCreateX state 14 then 14 else 0 -- we would need to show how to actually maximizing the value

theorem max_elements_X : maxCreateX ⟨50, 50, 0⟩ = 14 := 
by 
  -- Proof would go here, showing via the conditions given above
  -- We would need to show no more than 14 can be created given the initial resources
  sorry

end max_elements_X_l62_62052


namespace sum_maximized_at_5_or_6_l62_62790

-- Defining the arithmetic sequence and the properties
variables {n : ℕ} {a d : ℤ}
def a_seq (n: ℕ) := a + n * d
def sum_first_n_terms (n: ℕ) := ∑ i in range(n), a_seq i

theorem sum_maximized_at_5_or_6 {a : ℤ} {d : ℤ} (h1 : |a + 2 * d| = |a + 8 * d|) (h2: d < 0) :
  ∃ n, n = 5 ∨ n = 6 ∧ ∀ m, m = 5 ∨ m = 6 → sum_first_n_terms m ≤ sum_first_n_terms n :=
sorry

end sum_maximized_at_5_or_6_l62_62790


namespace percentage_neither_l62_62647

theorem percentage_neither (total_teachers high_blood_pressure heart_trouble both_conditions : ℕ)
  (h1 : total_teachers = 150)
  (h2 : high_blood_pressure = 90)
  (h3 : heart_trouble = 60)
  (h4 : both_conditions = 30) :
  100 * (total_teachers - (high_blood_pressure + heart_trouble - both_conditions)) / total_teachers = 20 :=
by
  sorry

end percentage_neither_l62_62647


namespace tangent_line_at_point_tangent_lines_with_slope_4_l62_62398

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line_at_point :
  ∀ (x y : ℝ), x = 2 → y = f 2 →
  (13 * x - y - 32 = 0) :=
by sorry

theorem tangent_lines_with_slope_4 :
  ∀ (x₀ y₀ : ℝ), 
  (f' : ℝ → ℝ) := (λ x, 3 * x^2 + 1) →
  (f' x₀ = 4) →
  (x₀ = 1 → y₀ = f 1 → 4 * x₀ - y₀ - 18 = 0) ∧
  (x₀ = -1 → y₀ = f -1 → 4 * x₀ - y₀ - 14 = 0) :=
by sorry

end tangent_line_at_point_tangent_lines_with_slope_4_l62_62398


namespace animals_in_reptile_house_l62_62880

theorem animals_in_reptile_house (rain_forest_animals : ℕ) (h1 : rain_forest_animals = 7) : 
  let reptile_house_animals := 3 * rain_forest_animals - 5 in
  reptile_house_animals = 16 :=
by
  sorry

end animals_in_reptile_house_l62_62880


namespace prove_a_zero_l62_62417

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62417


namespace a_power_l62_62557

theorem a_power (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end a_power_l62_62557


namespace num_possible_values_l62_62930

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62930


namespace frustum_proof_l62_62361

noncomputable def frustum_solutions (r1 r2 : ℝ) (S_lat : ℝ) : ℝ × ℝ :=
let S_top := Real.pi * r1^2,
    S_bottom := Real.pi * r2^2,
    S_bases := S_top + S_bottom,
    l := S_lat / (Real.pi * (r1 + r2)),
    h := Real.sqrt (l^2 - (r2 - r1)^2),
    V := (1 / 3) * Real.pi * h * (S_top + S_bottom + Real.sqrt (S_top * S_bottom))
in (l, V)

theorem frustum_proof :
  let r_top := 2,
      r_bottom := 5,
      S_lat := 29 * Real.pi in
  frustum_solutions r_top r_bottom S_lat =
  (29 / 7, 260 * Real.pi / 7) :=
by
  sorry

end frustum_proof_l62_62361


namespace mark_owes_linda_l62_62073

-- Define the payment per room and the number of rooms painted
def payment_per_room := (13 : ℚ) / 3
def rooms_painted := (8 : ℚ) / 5

-- State the theorem and the proof
theorem mark_owes_linda : (payment_per_room * rooms_painted) = (104 : ℚ) / 15 := by
  sorry

end mark_owes_linda_l62_62073


namespace cube_root_of_27_l62_62120

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l62_62120


namespace fraction_sum_reciprocal_ge_two_l62_62867

theorem fraction_sum_reciprocal_ge_two (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a / b) + (b / a) ≥ 2 :=
sorry

end fraction_sum_reciprocal_ge_two_l62_62867


namespace find_equation_l62_62665

theorem find_equation (x : ℝ) : 
  (3 + x < 1 → false) ∧
  ((x - 67 + 63 = x - 4) → false) ∧
  ((4.8 + x = x + 4.8) → false) ∧
  (x + 0.7 = 12 → true) := 
sorry

end find_equation_l62_62665


namespace scarlet_savings_l62_62541

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l62_62541


namespace inequality_proof_l62_62826

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1 / a - 1 / b + 1 / c) ≥ 1 :=
by
  sorry

end inequality_proof_l62_62826


namespace compute_F_2_f_3_l62_62837

def f (a : ℝ) : ℝ := a^2 - 3 * a + 2
def F (a b : ℝ) : ℝ := b + a^3

theorem compute_F_2_f_3 : F 2 (f 3) = 10 :=
by
  sorry

end compute_F_2_f_3_l62_62837


namespace sqrt3_612_expression_log_expression_l62_62675

-- Define the first statement using properties of exponents
theorem sqrt3_612_expression : (sqrt 3) * 612 * (3 + 1/2: ℝ) = 3 := sorry

-- Define the second statement using properties of logarithms
theorem log_expression : (log 5)^2 - (log 2)^2 + log 4 = 1 := sorry

end sqrt3_612_expression_log_expression_l62_62675


namespace points_among_transformations_within_square_l62_62907

def projection_side1 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, 2 - A.2)
def projection_side2 (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)
def projection_side3 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)
def projection_side4 (A : ℝ × ℝ) : ℝ × ℝ := (2 - A.1, A.2)

def within_square (A : ℝ × ℝ) : Prop := 
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

theorem points_among_transformations_within_square (A : ℝ × ℝ)
  (H1 : within_square A)
  (H2 : within_square (projection_side1 A))
  (H3 : within_square (projection_side2 (projection_side1 A)))
  (H4 : within_square (projection_side3 (projection_side2 (projection_side1 A))))
  (H5 : within_square (projection_side4 (projection_side3 (projection_side2 (projection_side1 A))))) :
  A = (1 / 3, 1 / 3) := sorry

end points_among_transformations_within_square_l62_62907


namespace tree_height_l62_62239

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l62_62239


namespace field_trip_buses_needed_l62_62581

theorem field_trip_buses_needed (classrooms : ℕ) (students_per_classroom : ℕ) (seats_per_bus : ℕ) :
  classrooms = 67 → students_per_classroom = 66 → seats_per_bus = 6 →
  let total_students := classrooms * students_per_classroom,
      buses_needed := (total_students + seats_per_bus - 1) / seats_per_bus
  in buses_needed = 738 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  let total_students := 67 * 66
  have h_total_students : total_students = 4422 := by norm_num
  rw h_total_students
  let buses_needed := (4422 + 6 - 1) / 6
  have h_buses_needed : buses_needed = 738 := by norm_num
  exact h_buses_needed

end field_trip_buses_needed_l62_62581


namespace min_sin4_cos4_l62_62706

theorem min_sin4_cos4 {α : ℝ} (hα : 0 ≤ α ∧ α ≤ π / 2) : 
  ∃ (c : ℝ), c = sin α ^ 4 + cos α ^ 4 ∧ c = 1 / 2 :=
by
  sorry

end min_sin4_cos4_l62_62706


namespace minimum_value_of_x_l62_62450

noncomputable def math_problem (x : ℝ) :=
  x > 0 ∧ log x ≥ 2 * log 3 - (1 / 3) * log x

theorem minimum_value_of_x : ∀ (x : ℝ), math_problem x → x ≥ 27 :=
by
  intro x
  assume hx : math_problem x
  sorry

end minimum_value_of_x_l62_62450


namespace even_function_a_zero_l62_62430

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62430


namespace probability_bottom_red_given_top_red_l62_62148

noncomputable def card1 : Prop := true
noncomputable def card2 : Prop := true
noncomputable def both_sides_red (c : Prop) : Bool :=
  if c = card1 then true else false
noncomputable def one_side_red_one_side_blue (c : Prop) : Bool :=
  if c = card2 then true else false
noncomputable def same_probability : Bool := true

theorem probability_bottom_red_given_top_red : 
  (both_sides_red card1 = true ∧ both_sides_red card2 = false ∧ same_probability) →
  (one_side_red_one_side_blue card1 = false ∧ one_side_red_one_side_blue card2 = true) →
  (probability_top_red : Float) →   
  (probability_bottom_red : Float) :=
begin
  sorry
end

end probability_bottom_red_given_top_red_l62_62148


namespace barbecue_chicken_orders_l62_62599

-- Defining the constants and variables
def pieces_per_chicken_pasta := 2
def pieces_per_barbecue_chicken := 3
def pieces_per_fried_chicken_dinner := 8

def chicken_pasta_orders := 6
def fried_chicken_dinner_orders := 2
def total_chicken_pieces := 37

-- The statement to be proved
theorem barbecue_chicken_orders :
  ∃ (barbecue_chicken_orders : ℕ), 
    fried_chicken_dinner_orders * pieces_per_fried_chicken_dinner 
    + chicken_pasta_orders * pieces_per_chicken_pasta 
    + barbecue_chicken_orders * pieces_per_barbecue_chicken = total_chicken_pieces :=
by 
  -- Equation calculation based on the problem conditions
  have calc_eq : 
    fried_chicken_dinner_orders * pieces_per_fried_chicken_dinner 
    + chicken_pasta_orders * pieces_per_chicken_pasta 
    = 16 + 12 := by sorry,
  have total_calc : 
    ∃ (barbecue_chicken_orders : ℕ),
    16 + 12 + barbecue_chicken_orders * pieces_per_barbecue_chicken = 37 := by sorry,
  exact ⟨3, by sorry⟩

end barbecue_chicken_orders_l62_62599


namespace count_integers_l62_62335

theorem count_integers (f : ℤ → ℤ) : 
  (f = λ x, x^4 - 63 * x^2 + 126) → (card {x : ℤ | f x < 0} = 12) :=
begin
  intro h,
  sorry
end

end count_integers_l62_62335


namespace angles_of_triangle_BDC_l62_62490

noncomputable def Triangle_ABC : Type := sorry
noncomputable def Point_D : Type := sorry

open Real

-- Assuming Triangle_ABC and Point_D respects the given conditions, we formulate the theorem 
theorem angles_of_triangle_BDC :
  (∀ (ABC : Triangle_ABC) (D : Point_D),
    side_length ABC AB = 2 ∧
    angle ABC A = 60 ∧
    angle ABC B = 70 ∧
    (∃ (C : Point), on_segment D A C ∧ length_segment D A C = 1)  →
    (∃ (angleBDC angleBCD angleCBD : ℝ), angleBDC = 90 ∧ angleBCD = 40 ∧ angleCBD = 50)) :=
sorry

end angles_of_triangle_BDC_l62_62490


namespace kaleb_savings_l62_62051

theorem kaleb_savings (x : ℕ) (h : x + 25 = 8 * 8) : x = 39 := 
by
  sorry

end kaleb_savings_l62_62051


namespace negation_of_existential_statement_l62_62127

theorem negation_of_existential_statement {f : ℝ → ℝ} :
  (¬ ∃ x₀ : ℝ, f x₀ < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by
  sorry

end negation_of_existential_statement_l62_62127


namespace largest_multiple_of_7_less_than_100_l62_62606

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l62_62606


namespace probability_at_most_one_incorrect_l62_62460

variable (p : ℝ)

theorem probability_at_most_one_incorrect (h : 0 ≤ p ∧ p ≤ 1) :
  p^9 * (10 - 9*p) = p^10 + 10 * (1 - p) * p^9 := by
  sorry

end probability_at_most_one_incorrect_l62_62460


namespace unfolded_tetrahedron_not_right_triangle_l62_62228

theorem unfolded_tetrahedron_not_right_triangle 
  (D A B C : Point)
  (T : Tetrahedron D A B C)
  (unfolded_triangle : Triangle (midpoint D A) (midpoint D B) (midpoint D C)) :
  ¬ is_right_triangle unfolded_triangle := 
sorry

end unfolded_tetrahedron_not_right_triangle_l62_62228


namespace max_rooks_on_8x8_l62_62196

-- Definition of the chessboard and rook attack constraints
structure Chessboard (n : Nat) :=
  board : Fin n → Fin n → Bool

def attacks (C : Chessboard 8) (x y : Fin 8) : Prop :=
  ∃i : Fin 8, C.board x i || C.board i y

-- Maximum number of rooks function definition and theorem statement
def MaxRooks (C : Chessboard 8) : Nat :=
  Fin 64 |>.count (λ x => ∃i : Fin 8, attacks C i x)

theorem max_rooks_on_8x8 : ∀ (C : Chessboard 8), MaxRooks C ≤ 10 :=
by
  sorry

end max_rooks_on_8x8_l62_62196


namespace verify_digits_representation_l62_62037

theorem verify_digits_representation :
  ∃ (a b c d e f g h i j : ℕ),
    a = 1 ∧
    b = 9 ∧
    c = 8 ∧
    d = 5 ∧
    e = 4 ∧
    f = 0 ∧
    g = 6 ∧
    h = 7 ∧
    i = 2 ∧
    j = 3 ∧
    a * 100 + b * 10 + c - (d * 10 + c) = a * 100 + e * 10 + f ∧
    g * h = e * 10 + i ∧
    j * 10 + j + g * 10 + d = b * 10 + c :=
  begin
    use [1, 9, 8, 5, 4, 0, 6, 7, 2, 3],
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    
    -- proving the calculations
    repeat {linarith},
  end

end verify_digits_representation_l62_62037


namespace calculate_g_val_l62_62521

noncomputable def f (ω ϕ : ℝ) (x : ℝ) := 4 * real.cos (ω * x + ϕ)
noncomputable def g (ω ϕ : ℝ) (x : ℝ) := real.sin (ω * x + ϕ) - 2

theorem calculate_g_val (ω ϕ : ℝ) (h : ∀ x : ℝ, f ω ϕ (-x) = f ω ϕ (x + (π / 3))) : 
  g ω ϕ (π / 6) = -2 :=
by
  sorry

end calculate_g_val_l62_62521


namespace average_income_independence_l62_62473

theorem average_income_independence (A E : ℝ) (n : ℕ) (h : n = 10) :
  let avg_income := (A + E) / (n : ℝ)
  in avg_income = (A + E) / 10 :=
by
  intros
  have h1 : (n : ℝ) = 10 := by simp [h]
  rw h1
  sorry

end average_income_independence_l62_62473


namespace deformable_to_triangle_l62_62244

-- We define a planar polygon with n rods connected by hinges
structure PlanarPolygon (n : ℕ) :=
  (rods : Fin n → ℝ)
  (connections : Fin n → Fin n → Prop)

-- Define the conditions for the rods being rigid and connections (hinges)
def rigid_rod (n : ℕ) : PlanarPolygon n → Prop := λ poly => 
  ∀ i j, poly.connections i j → poly.rods i = poly.rods j

-- Defining the theorem for deformation into a triangle
theorem deformable_to_triangle (n : ℕ) (p : PlanarPolygon n) : 
  (n > 4) ↔ ∃ q : PlanarPolygon 3, true :=
by
  sorry

end deformable_to_triangle_l62_62244


namespace total_boxes_correct_l62_62234

def boxes_chocolate : ℕ := 2
def boxes_sugar : ℕ := 5
def boxes_gum : ℕ := 2
def total_boxes : ℕ := boxes_chocolate + boxes_sugar + boxes_gum

theorem total_boxes_correct : total_boxes = 9 := by
  sorry

end total_boxes_correct_l62_62234


namespace player_one_wins_l62_62916

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end player_one_wins_l62_62916


namespace total_bouncy_balls_l62_62074

-- Definitions based on the conditions of the problem
def packs_of_red := 4
def packs_of_yellow := 8
def packs_of_green := 4
def balls_per_pack := 10

-- Theorem stating the conclusion to be proven
theorem total_bouncy_balls :
  (packs_of_red + packs_of_yellow + packs_of_green) * balls_per_pack = 160 := 
by
  sorry

end total_bouncy_balls_l62_62074


namespace B_completes_work_in_15_days_l62_62220

theorem B_completes_work_in_15_days :
  (∃ (B : ℝ),
    let A_share := 1860 / 3100 in
    let A_work_rate := 1 / 10 in
    let B_work_rate := 1 / B in
    let combined_work_rate := A_work_rate + B_work_rate in
    A_work_rate / combined_work_rate = A_share) →
  B = 15 :=
by
  let A_share := 1860 / 3100
  let A_work_rate := 1 / 10
  let combined_work_rate := (A_work_rate + 1 / 15)
  have h1: combined_work_rate = A_work_rate + 1 / 15, by sorry
  have h2: A_work_rate / combined_work_rate = A_share, by sorry
  exact 15

end B_completes_work_in_15_days_l62_62220


namespace gcf_72_108_l62_62193

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62193


namespace coin_difference_l62_62042

noncomputable def max_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d
noncomputable def min_value (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem coin_difference (p n d : ℕ) (h₁ : p + n + d = 3030) (h₂ : 10 ≤ p) (h₃ : 10 ≤ n) (h₄ : 10 ≤ d) :
  max_value 10 10 3010 - min_value 3010 10 10 = 27000 := by
  sorry

end coin_difference_l62_62042


namespace prob_answer_4_questions_correct_l62_62020

noncomputable def prob_correct : ℝ := 0.8
noncomputable def prob_incorrect : ℝ := 1 - prob_correct
noncomputable def prob_ans_4_questions : ℝ :=
  prob_correct * prob_correct * prob_incorrect * prob_correct * prob_correct

theorem prob_answer_4_questions_correct :
  prob_ans_4_questions = 0.128 :=
by
  definition sorry

end prob_answer_4_questions_correct_l62_62020


namespace gas_volume_at_20_l62_62334

variable (V : ℝ) (T : ℝ) (k : ℝ)

-- Given conditions
def volume_at_32 := (T = 32) → (V = 24)
def expansion_per_3 := (∀ k, V = 24 + 4 * k) ∧ (∀ k, T = 32 + 3 * k)

-- Prove the volume at the temperature of 20 degrees Celsius
theorem gas_volume_at_20:
  ∀ V T, volume_at_32 T V →
  expansion_per_3 k →
  T = 20 →
  V = 8 :=
sorry

end gas_volume_at_20_l62_62334


namespace inradius_of_regular_tetrahedron_l62_62788

theorem inradius_of_regular_tetrahedron (h r : ℝ) (S : ℝ) 
  (h_eq: 4 * (1/3) * S * r = (1/3) * S * h) : r = (1/4) * h :=
sorry

end inradius_of_regular_tetrahedron_l62_62788


namespace largest_c_gap_l62_62180

def adjacent (a b : ℕ) : Prop :=
  ∃ (i j k l : ℕ), a = 8 * i + j + 1 ∧ b = 8 * k + l + 1 ∧
  (((i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ (j = l ∧ (i = k + 1 ∨ i + 1 = k))) ∨
  ((i = k + 1 ∨ i + 1 = k) ∧ (j = l + 1 ∨ j + 1 = l)))

def numbering (squares : Fin 64 → ℕ) : Prop :=
  ∀ i j, i ≠ j → squares i ≠ squares j

-- Mathematically equivalent proof problem in Lean 4
theorem largest_c_gap :
  ∀ (squares : Fin 64 → ℕ), numbering squares →
  ∃ (a b : Fin 64), adjacent (squares a) (squares b) ∧ abs (squares a - squares b) ≥ 9 :=
by
  sorry

end largest_c_gap_l62_62180


namespace closest_multiple_of_17_to_3513_is_3519_l62_62613

theorem closest_multiple_of_17_to_3513_is_3519 :
  ∃ n, (17 * n = 3519) ∧ (∀ k, abs (3513 - (17 * k)) ≥ abs (3513 - 3519)) :=
sorry

end closest_multiple_of_17_to_3513_is_3519_l62_62613


namespace gcd_m_n_is_one_l62_62602

def m : ℕ := 122^2 + 234^2 + 344^2

def n : ℕ := 123^2 + 235^2 + 343^2

theorem gcd_m_n_is_one : Nat.gcd m n = 1 :=
by
  sorry

end gcd_m_n_is_one_l62_62602


namespace even_function_a_zero_l62_62431

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62431


namespace domino_arrangements_l62_62288

theorem domino_arrangements :
  let m := 6 in let n := 5 in
  let domino_size := 1 in
  let num_right_moves := 5 in let num_down_moves := 4 in
  let total_moves := num_right_moves + num_down_moves in
  (total_moves.factorial / (num_right_moves.factorial * num_down_moves.factorial) = 126) :=
by
  -- Definitions for factorial calculations and combinatorial logic would plug in here.
  sorry

end domino_arrangements_l62_62288


namespace solve_equation_l62_62091

noncomputable def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * x ^ (2021 / 2022) - 1

theorem solve_equation : ∀ x : ℝ, equation x ↔ x = 1 :=
by
  intro x
  sorry

end solve_equation_l62_62091


namespace gcf_72_108_l62_62183

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62183


namespace chord_slope_of_ellipse_bisected_by_point_A_l62_62457

theorem chord_slope_of_ellipse_bisected_by_point_A :
  ∀ (P Q : ℝ × ℝ),
  (P.1^2 / 36 + P.2^2 / 9 = 1) ∧ (Q.1^2 / 36 + Q.2^2 / 9 = 1) ∧ 
  ((P.1 + Q.1) / 2 = 1) ∧ ((P.2 + Q.2) / 2 = 1) →
  (Q.2 - P.2) / (Q.1 - P.1) = -1 / 4 :=
by
  intros
  sorry

end chord_slope_of_ellipse_bisected_by_point_A_l62_62457


namespace EllenBreadMakingTime_l62_62696

-- Definitions based on the given problem
def RisingTimeTypeA : ℕ → ℝ := λ n => n * 4
def BakingTimeTypeA : ℕ → ℝ := λ n => n * 2.5
def RisingTimeTypeB : ℕ → ℝ := λ n => n * 3.5
def BakingTimeTypeB : ℕ → ℝ := λ n => n * 3

def TotalTime (nA nB : ℕ) : ℝ :=
  (RisingTimeTypeA nA + BakingTimeTypeA nA) +
  (RisingTimeTypeB nB + BakingTimeTypeB nB)

theorem EllenBreadMakingTime :
  TotalTime 3 2 = 32.5 := by
  sorry

end EllenBreadMakingTime_l62_62696


namespace eval_f_at_1_l62_62396

noncomputable def f (x : ℝ) : ℝ := log (3 + x) / log 2 + log (3 - x) / log 2

theorem eval_f_at_1 : f 1 = 3 := 
  by
  sorry

end eval_f_at_1_l62_62396


namespace perpendiculars_foot_circle_l62_62984

-- Define a structure to host the setup of the problem:
structure AngleConfig :=
(O C A B : Point)  -- Points O, C, A, and B
(alpha : Angle)    -- Given angle alpha
(vertex : C = vertex(O, A, B))  -- C is the vertex formed by angle point

-- Definition of the point M where perpendicular from O intersects AB:
def perpendicular_foot (cfg : AngleConfig) : Set Point :=
  { M : Point | exists K L : Point, (is_perpendicular (cfg.O, M) (M, cfg.A)) ∧ 
                                      (is_perpendicular (cfg.O, M) (M, cfg.B)) ∧
                                      cyclic_quadrilateral cfg.O cfg.K cfg.A cfg.L }

-- Main theorem that states the set of the feet of the perpendiculars forms a circle:
theorem perpendiculars_foot_circle (cfg : AngleConfig) :
  ∃ circle : Set Point, 
  (∀ M : Point, M ∈ perpendicular_foot cfg ↔ M ∈ circle) :=
sorry

end perpendiculars_foot_circle_l62_62984


namespace num_possible_values_l62_62931

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62931


namespace tan_phi_triangle_bisectors_l62_62661

theorem tan_phi_triangle_bisectors :
  ∀ {a b c : ℝ} (H : a = 5 ∧ b = 12 ∧ c = 13) (perimeter : ℝ) (area : ℝ) (φ : ℝ),
  perimeter = a + b + c ∧ 
  area = real.sqrt ((perimeter / 2) * ((perimeter / 2) - a) * ((perimeter / 2) - b) * ((perimeter / 2) - c)) ∧ 
  tan φ = 17 / 7 -> 
  tan φ = 17 / 7 :=
by
  intros a b c H perimeter area φ H_cond
  -- Given the conditions define the proof steps here.
  sorry

end tan_phi_triangle_bisectors_l62_62661


namespace Wendy_total_sales_is_correct_l62_62182

noncomputable def morning_sales_apple := 40 * 1.50
noncomputable def morning_sales_orange := 30 * 1
noncomputable def total_morning_sales := morning_sales_apple + morning_sales_orange

noncomputable def afternoon_sales_apple := 50 * 1.50
noncomputable def afternoon_sales_orange := 40 * 1
noncomputable def total_afternoon_sales := afternoon_sales_apple + afternoon_sales_orange

noncomputable def total_sales := total_morning_sales + total_afternoon_sales

theorem Wendy_total_sales_is_correct : total_sales = 205 := by
  sorry

end Wendy_total_sales_is_correct_l62_62182


namespace farmer_land_l62_62645

theorem farmer_land (A : ℝ) (A_nonneg : A ≥ 0) (cleared_land : ℝ) 
  (soybeans wheat potatoes vegetables corn : ℝ) 
  (h_cleared : cleared_land = 0.95 * A) 
  (h_soybeans : soybeans = 0.35 * cleared_land) 
  (h_wheat : wheat = 0.40 * cleared_land) 
  (h_potatoes : potatoes = 0.15 * cleared_land) 
  (h_vegetables : vegetables = 0.08 * cleared_land) 
  (h_corn : corn = 630) 
  (cleared_sum : soybeans + wheat + potatoes + vegetables + corn = cleared_land) :
  A = 33158 := 
by 
  sorry

end farmer_land_l62_62645


namespace positive_number_square_root_l62_62018

theorem positive_number_square_root (a n : ℝ) (h1 : sqrt n = a + 3) (h2 : sqrt n = 2 * a - 15) : n = 49 := 
sorry

end positive_number_square_root_l62_62018


namespace car_speed_kmph_l62_62223

noncomputable def speed_of_car (d : ℝ) (t : ℝ) : ℝ :=
  (d / t) * 3.6

theorem car_speed_kmph : speed_of_car 10 0.9999200063994881 = 36000.29 := by
  sorry

end car_speed_kmph_l62_62223


namespace possible_N_values_l62_62957

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62957


namespace student_selection_l62_62142

theorem student_selection (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 4) : a + b + c = 12 :=
by {
  sorry
}

end student_selection_l62_62142


namespace part_a_part_b_l62_62053

theorem part_a (ABC : Triangle) (O : Point) (Γ1 : Circle) (A B C D E F G : Point) :
  A ∈ Γ1 ∧ B ∈ Γ1 ∧ C ∈ Γ1 ∧
  B ≠ A ∧ A ≠ C ∧ B ≠ C ∧
  AC < BC ∧ AB < AC ∧ is_acute_triangle ABC ∧
  ∃ Γ2 : Circle, center Γ2 = A ∧ radius Γ2 = AC ∧
  D ∈ (BC ∩ Γ2) ∧ E ∈ (AΓ2 ∩ Γ1) ∧
  F ∈ (AD ∩ Γ1) ∧ G ∈ (Γ3 ∩ BC) ∧
  circumscribed_circle DEF = Γ3 ∧ center Γ3 = B :=
sorry

theorem part_b (ABC : Triangle) (O : Point) (Γ1 : Circle) (A B C D E F G : Point) :
  A ∈ Γ1 ∧ B ∈ Γ1 ∧ C ∈ Γ1 ∧
  B ≠ A ∧ A ≠ C ∧ B ≠ C ∧
  AC < BC ∧ AB < AC ∧ is_acute_triangle ABC ∧
  ∃ Γ2 : Circle, center Γ2 = A ∧ radius Γ2 = AC ∧
  D ∈ (BC ∩ Γ2) ∧ E ∈ (AΓ2 ∩ Γ1) ∧
  F ∈ (AD ∩ Γ1) ∧ G ∈ (Γ3 ∩ BC) ∧
  circumscribed_circle DEF = Γ3 ∧
  circumscribed_circle CEG ⊥ AC :=
sorry

end part_a_part_b_l62_62053


namespace right_triangle_unique_value_l62_62216

theorem right_triangle_unique_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
(h1 : a + b + c = (1/2) * a * b) (h2 : c^2 = a^2 + b^2) : a + b - c = 4 :=
by
  sorry

end right_triangle_unique_value_l62_62216


namespace silvia_trip_shorter_than_jerry_l62_62043

theorem silvia_trip_shorter_than_jerry
  (length width : ℕ) (H_length : length = 3) (H_width : width = 4) :
  (let j := length + width,
       s := Real.sqrt (length^2 + width^2),
       difference := j - s,
       percentage_reduction := (difference / j) * 100 in percentage_reduction ≈ 30) :=
by
  sorry

end silvia_trip_shorter_than_jerry_l62_62043


namespace john_marbles_problem_l62_62819

open Nat

-- Define the conditions provided in the problem
def total_marbles := 15
def red_marbles := 2
def green_marbles := 2
def blue_marbles := 2
def other_marbles := total_marbles - red_marbles - green_marbles - blue_marbles

-- Define the function for binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)))
  else 0

-- The main problem statement
theorem john_marbles_problem : 
  let ways_red := binom red_marbles 1,
      ways_green := binom green_marbles 1,
      ways_remaining := binom (total_marbles - 2) 3
  in ways_red * ways_green * ways_remaining = 660 :=
by
  sorry

end john_marbles_problem_l62_62819


namespace tangent_with_min_slope_has_given_equation_l62_62667

-- Define the given function f(x)
def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the coordinates of the tangent point
def tangent_point : ℝ × ℝ := (-1, f (-1))

-- Define the equation of the tangent line at the point with the minimum slope
def tangent_line_equation (x y : ℝ) : Prop := 3 * x - y - 11 = 0

-- Main theorem statement that needs to be proved
theorem tangent_with_min_slope_has_given_equation :
  tangent_line_equation (-1) (f (-1)) :=
sorry

end tangent_with_min_slope_has_given_equation_l62_62667


namespace compare_neg_rationals_l62_62279

-- Definition and conditions
def abs_neg_one_third : ℚ := |(-1 / 3 : ℚ)|
def abs_neg_one_fourth : ℚ := |(-1 / 4 : ℚ)|

-- Problem statement
theorem compare_neg_rationals : (-1 : ℚ) / 3 < -1 / 4 :=
by
  -- Including the conditions here, even though they are straightforward implications in Lean
  have h1 : abs_neg_one_third = 1 / 3 := abs_neg_one_third
  have h2 : abs_neg_one_fourth = 1 / 4 := abs_neg_one_fourth
  -- We would include steps to show that -1 / 3 < -1 / 4 using the above facts
  sorry

end compare_neg_rationals_l62_62279


namespace inequality_proof_l62_62537

-- Define the main theorem to be proven.
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end inequality_proof_l62_62537


namespace ratio_of_areas_of_two_concentric_circles_l62_62596

theorem ratio_of_areas_of_two_concentric_circles
  (C₁ C₂ : ℝ)
  (h1 : ∀ θ₁ θ₂, θ₁ = 30 ∧ θ₂ = 24 →
      (θ₁ / 360) * C₁ = (θ₂ / 360) * C₂):
  (C₁ / C₂) ^ 2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_two_concentric_circles_l62_62596


namespace line_through_circumcenter_l62_62121

open EuclideanGeometry

variables {M M1 A B C : Point}
variables {dM_A dM_B dM_C dM1_A dM1_B dM1_C : ℝ}

def dist (P Q : Point) : ℝ := sorry -- Assuming Euclidean distance for the sake of the placeholder

def circumcenter (A B C : Point) : Point := sorry -- Assuming circumcenter computation for the sake of the placeholder

theorem line_through_circumcenter
  (hM_A : dist M A = 1)
  (hM_B : dist M B = 2)
  (hM_C : dist M C = 3)
  (hM1_A : dist M1 A = 3)
  (hM1_B : dist M1 B = sqrt 15)
  (hM1_C : dist M1 C = 5) :
  ∃ O : Point, O = circumcenter A B C ∧ collinear O M M1 :=
sorry

end line_through_circumcenter_l62_62121


namespace max_value_of_k_l62_62376

noncomputable def set_of_b_vectors (a1 a2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  { b | (dist a1 b = 1 ∨ dist a1 b = 2 ∨ dist a1 b = 3) ∧
         (dist a2 b = 1 ∨ dist a2 b = 2 ∨ dist a2 b = 3) }

theorem max_value_of_k (a1 a2 : ℝ × ℝ) (h : dist a1 a2 = 1) :
  ∃ (k : ℕ), k = 10 ∧ ∀ b ∈ set_of_b_vectors a1 a2, 
  (∀ i j, i ≠ j → b ≠ i ∧ b ≠ j ∧ b ≠ 0) := sorry

end max_value_of_k_l62_62376


namespace thousandth_digit_is_three_l62_62512

def get_nth_digit (n : ℕ) : ℕ :=
  let seq := String.join (List.map toString [1:500])
  seq.nth (n - 1).getD '0'.toNat - '0'.toNat

theorem thousandth_digit_is_three :
  get_nth_digit 1000 = 3 :=
by rfl -- the actual proof is omitted

end thousandth_digit_is_three_l62_62512


namespace caravan_humps_l62_62970

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62970


namespace compute_a_plus_b_l62_62337

theorem compute_a_plus_b (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (∏ i in finset.range (b-a), log ((a + i) : ℝ) / log ((a + i - 1) : ℝ)) = 2)
  (h4 : b - a = 580) :
  a + b = 870 :=
sorry

end compute_a_plus_b_l62_62337


namespace min_max_sets_share_common_elem_l62_62141

open Set Nat

/-- Given 11 sets each containing 5 elements, with the condition that every pair of sets has a non-empty intersection, 
prove that the minimum possible value of the maximum number of sets that share a common element is 4. --/
theorem min_max_sets_share_common_elem : 
  ∀ (M : Fin 11 → Finset (Fin 55)), (∀ i, (M i).card = 5) ∧ (∀ i j, i ≠ j → (M i ∩ M j).nonempty) →
  ∃ x, 4 = max {n | ∃ j, M j x} := 
by 
  sorry

end min_max_sets_share_common_elem_l62_62141


namespace pythagorean_triple_probability_l62_62465

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def nCr (n r : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem pythagorean_triple_probability : 
  let nums := {1, 2, 3, 4, 5}
  let triples := finset.filter (λ (t : ℕ × ℕ × ℕ), is_pythagorean_triple t.1 t.2 t.3)
                             ((finset.univ : finset (ℕ × ℕ × ℕ)).filter (λ (t : ℕ × ℕ × ℕ), 
                              t.1 ≠ t.2 ∧ t.1 ≠ t.3 ∧ t.2 ≠ t.3 ∧ 
                              t.1 ∈ nums ∧ t.2 ∈ nums ∧ t.3 ∈ nums))
in triples.card = 1 / nCr 5 3 :=
sorry

end pythagorean_triple_probability_l62_62465


namespace length_of_segment_AB_l62_62033

-- Definitions of given parametric equations of the line l.
def line_l_parametric (t : ℝ) : ℝ × ℝ :=
(1 + t, 2 - 2 * t)

-- Polar equation of circle C in rectangular coordinates
def circle_C_rect (x y : ℝ) : Prop :=
x^2 + y^2 = 2 * x 

-- Length of the segment AB
def segment_AB_length : ℝ :=
2 * (Real.sqrt (1 - (2 / (Real.sqrt (5)))^2))

-- Main theorem statement
theorem length_of_segment_AB: 
  segment_AB_length = 2 * (Real.sqrt (1 - (4 / 5))) := 
  by
    sorry

end length_of_segment_AB_l62_62033


namespace sin_cos_sum_correct_l62_62587

noncomputable def sin_cos_sum (x y : ℤ) (r : ℝ) : ℝ := 
  let sin_alpha := (y : ℝ) / r
  let cos_alpha := (x : ℝ) / r
  sin_alpha + 2 * cos_alpha

theorem sin_cos_sum_correct (x y : ℤ) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  sin_cos_sum x y r = 2 / 5 := by
  -- The point P(3, -4)
  let x := 3
  let y := -4
  -- The radius r
  have r_eq : r = 5 := by
    rw [Real.sqrt_eq_rpow, Real.rpow_nat_cast, Int.cast_bit0, Int.cast_bit1]
    norm_num
    
  -- Applying the values to sin_cos_sum definition
  rw [sin_cos_sum]
  have sin_alpha := (y : ℝ) / r
  have cos_alpha := (x : ℝ) / r
  have result := sin_alpha + 2 * cos_alpha
  
  -- Asserting the expected value for sin_alpha and cos_alpha
  rw [Int.cast_of_nat x, Int.cast_neg y, Int.cast_of_nat (3 : ℤ), Int.cast_neg (4 : ℤ)]
  norm_num
  
  sorry

end sin_cos_sum_correct_l62_62587


namespace place_chess_pieces_l62_62482

-- Defining the chessboard size and the condition for marked squares
def chessboard := fin 8 × fin 8
def marked_squares (s : set chessboard) : Prop := 
  (∀ r : fin 8, ∃ c1 c2 : fin 8, (r, c1) ∈ s ∧ (r, c2) ∈ s ∧ c1 ≠ c2) ∧
  (∀ c : fin 8, ∃ r1 r2 : fin 8, (r1, c) ∈ s ∧ (r2, c) ∈ s ∧ r1 ≠ r2)

-- The main problem statement: proving the placement of pieces
theorem place_chess_pieces (s : set chessboard) (hs : marked_squares s) :
  ∃ b w : set chessboard, 
    disjoint b w ∧
    s = b ∪ w ∧
    (∀ r : fin 8, ∃! c : fin 8, (r, c) ∈ b) ∧
    (∀ r : fin 8, ∃! c : fin 8, (r, c) ∈ w) ∧
    (∀ c : fin 8, ∃! r : fin 8, (r, c) ∈ b) ∧
    (∀ c : fin 8, ∃! r : fin 8, (r, c) ∈ w) :=
by sorry

end place_chess_pieces_l62_62482


namespace painters_work_days_l62_62807

/-
It takes five painters working at the same rate 1.5 work-days to finish a job.
If only four painters are available, prove how many work-days will it take them to finish the job, working at the same rate.
-/

theorem painters_work_days (days5 : ℚ) (h : days5 = 3 / 2) :
  ∃ days4 : ℚ, 5 * days5 = 4 * days4 ∧ days4 = 15 / 8 :=
  by
    use 15 / 8
    split
    · calc
        5 * days5 = 5 * (3 / 2) : by rw h
        ... = 15 / 2 : by norm_num
        ... = 4 * (15 / 8) : by norm_num
    · refl

end painters_work_days_l62_62807


namespace probability_of_inequality_l62_62733

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def monotonically_decreasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

noncomputable def satisfies_inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
f (x - 1) ≥ f 1

theorem probability_of_inequality (f : ℝ → ℝ) :
  even_function f →
  monotonically_decreasing_on_nonnegative f →
  (∫ x in -4..4, if satisfies_inequality f x then 1 else 0) / 8 = 1 / 4 :=
by
  sorry

end probability_of_inequality_l62_62733


namespace functions_identified_l62_62726

variable (n : ℕ) (hn : n > 1)
variable {f : ℕ → ℝ → ℝ}

-- Define the conditions f1, f2, ..., fn
axiom cond_1 (x y : ℝ) : f 1 x + f 1 y = f 2 x * f 2 y
axiom cond_2 (x y : ℝ) : f 2 (x^2) + f 2 (y^2) = f 3 x * f 3 y
axiom cond_3 (x y : ℝ) : f 3 (x^3) + f 3 (y^3) = f 4 x * f 4 y
-- ... Similarly define conditions up to cond_n
axiom cond_n (x y : ℝ) : f n (x^n) + f n (y^n) = f 1 x * f 1 y

theorem functions_identified (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ n) (x : ℝ) :
  f i x = 0 ∨ f i x = 2 :=
sorry

end functions_identified_l62_62726


namespace prime_factors_of_four_consecutive_integers_sum_l62_62152

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l62_62152


namespace problem_statement_l62_62751

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 
  real.sin (2 * x + φ)

theorem problem_statement (φ : ℝ) (x1 x2 : ℝ)
  (h1 : |φ| < π / 2)
  (h2 : 2 * (π / 3) + φ = int.cast ((1 : ℤ) * π))
  (h3 : x1 ∈ Ioo (π / 12) (7 * π / 12))
  (h4 : x2 ∈ Ioo (π / 12) (7 * π / 12))
  (h5 : x1 ≠ x2)
  (h6 : f x1 φ + f x2 φ = 0)
  : f (x1 + x2) φ = - (real.sqrt 3) / 2 := by
  sorry

end problem_statement_l62_62751


namespace middle_number_is_10_l62_62922

theorem middle_number_is_10 (A B C : ℝ) (h1 : B - C = A - B) (h2 : A * B = 85) (h3 : B * C = 115) : B = 10 :=
by
  sorry

end middle_number_is_10_l62_62922


namespace gcf_72_108_l62_62192

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62192


namespace intersection_A_B_union_A_B_intersection_complements_C_A_C_B_l62_62403

-- Defining the universal set as real numbers
def U := set ℝ

-- Defining set A
def A := {x : ℝ | 1 ≤ x ∧ x < 5}

-- Defining set B
def B := {x : ℝ | 2 < x ∧ x < 8}

-- The first proof problem: proving the intersection A ∩ B
theorem intersection_A_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 5} :=
by sorry

-- The second proof problem: proving the union A ∪ B
theorem union_A_B : (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 8} :=
by sorry

-- Defining the complement of A and B relative to U (ℝ in this case)
def complement (s : set ℝ) := {x : ℝ | x ∉ s}
def C_A := complement A
def C_B := complement B

-- The third proof problem: proving the intersection of complements
theorem intersection_complements_C_A_C_B : (C_A ∩ C_B) = {x : ℝ | x < 1 ∨ x ≥ 8} :=
by sorry

end intersection_A_B_union_A_B_intersection_complements_C_A_C_B_l62_62403


namespace line_passes_through_fixed_point_l62_62008

theorem line_passes_through_fixed_point
  {a b c : ℝ} (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_intercepts_constant : (1 / a) + (1 / b) = 1 / c) :
  (∃ (x y : ℝ), x = c ∧ y = c ∧ (x / a) + (y / b) = 1) :=
by
  use [c, c]
  split
  . refl
  split
  . refl
  calc
    (c / a) + (c / b)
      = (c / a) + (c / b) : rfl
    ... = c * ((1 / a) + (1 / b)) : by rw [←mul_add, mul_one_div]
    ... = c * (1 / c) : by rw h_intercepts_constant
    ... = 1 : by rw [mul_inv_cancel (ne.symm h3)]

end line_passes_through_fixed_point_l62_62008


namespace white_area_correct_l62_62912

def total_sign_area : ℕ := 8 * 20
def black_area_C : ℕ := 8 * 1 + 2 * (1 * 3)
def black_area_A : ℕ := 2 * (8 * 1) + 2 * (1 * 2)
def black_area_F : ℕ := 8 * 1 + 2 * (1 * 4)
def black_area_E : ℕ := 3 * (1 * 4)

def total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
def white_area : ℕ := total_sign_area - total_black_area

theorem white_area_correct : white_area = 98 :=
  by 
    sorry -- State the theorem without providing the proof.

end white_area_correct_l62_62912


namespace exists_polynomial_satisfying_condition_l62_62302

noncomputable def quadratic_polynomial : (ℕ → ℕ) :=
  λ n => 90 * n^2 + 20 * n + 1

theorem exists_polynomial_satisfying_condition :
  ∃ P : ℕ → ℕ, (∀ n, (∃ k : ℕ, n = (10^k - 1) / 9) → (P n = (10^(2 * k + 1) - 1) / 9)) :=
begin
  use quadratic_polynomial,
  intros n hn,
  rcases hn with ⟨k, rfl⟩,
  sorry
end

end exists_polynomial_satisfying_condition_l62_62302


namespace distance_between_points_l62_62296

def point := (ℝ, ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_points : 
  dist (2, 3) (-1, 10) = real.sqrt 58 :=
by
  sorry

end distance_between_points_l62_62296


namespace firm_partners_initial_count_l62_62231

theorem firm_partners_initial_count
  (x : ℕ)
  (h1 : 2*x/(63*x + 35) = 1/34)
  (h2 : 2*x/(20*x + 10) = 1/15) :
  2*x = 14 :=
by
  sorry

end firm_partners_initial_count_l62_62231


namespace main_theorem_l62_62752

variable (a : ℝ)

def M : Set ℝ := {x | x > 1 / 2 ∧ x < 1} ∪ {x | x > 1}
def N : Set ℝ := {x | x > 0 ∧ x ≤ 1 / 2}

theorem main_theorem : M ∩ N = ∅ :=
by
  sorry

end main_theorem_l62_62752


namespace g_2023_l62_62570

noncomputable def g : ℕ → ℕ :=
  by sorry

axiom g_property_1 : ∀ (n : ℕ), 0 < n → g(g(n)) = 3 * n

axiom g_property_2 : ∀ (n : ℕ), 0 < n → g(5 * n + 2) = 5 * n + 4

theorem g_2023 :
  g(2023) = 2034 :=
by
  sorry

end g_2023_l62_62570


namespace cube_root_of_27_l62_62119

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l62_62119


namespace first_player_cannot_prevent_second_player_l62_62904

theorem first_player_cannot_prevent_second_player (f : ℕ → ℕ) (cube : Fin 24 → ℕ) :
  (∀ i, 1 ≤ cube i ∧ cube i ≤ 24) →
  (∀ ring : Fin 3 → Fin 8, (∑ i, cube (ring i)) = 100) →
  ∃ g : Fin 24 → ℕ, (∀ i, 1 ≤ g i ∧ g i ≤ 24) ∧ (∀ ring : Fin 3 → Fin 8, (∑ i, g (ring i)) = 100)
:=
sorry

end first_player_cannot_prevent_second_player_l62_62904


namespace bird_stork_difference_l62_62637

theorem bird_stork_difference :
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  total_birds - initial_storks = 1 := 
by
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  show total_birds - initial_storks = 1
  sorry

end bird_stork_difference_l62_62637


namespace painters_workdays_l62_62811

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l62_62811


namespace angles_sum_of_roots_l62_62902

theorem angles_sum_of_roots :
  let z_roots : list ℝ := 
    [(135 / 5), (135 / 5 + 360 / 5), (135 / 5 + 2 * 360 / 5), 
     (135 / 5 + 3 * 360 / 5), (135 / 5 + 4 * 360 / 5)]
  in z_roots.sum = 1125 :=
by
  let z_roots : list ℝ := 
    [(135 / 5), (135 / 5 + 360 / 5), (135 / 5 + 2 * 360 / 5), 
     (135 / 5 + 3 * 360 / 5), (135 / 5 + 4 * 360 / 5)]
  exact sorry

end angles_sum_of_roots_l62_62902


namespace arithmetic_progression_K_l62_62866

theorem arithmetic_progression_K (K : ℕ) : 
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
sorry

end arithmetic_progression_K_l62_62866


namespace nougat_caramel_ratio_l62_62276

variable (N C T P : ℕ)

noncomputable def chocolates_conditions : Prop :=
  C = 3 ∧
  T = C + 6 ∧
  P = 0.64 * 50 ∧
  C + N + T + P = 50

theorem nougat_caramel_ratio (h : chocolates_conditions N C T P) :
  (N / C = 2) :=
sorry

end nougat_caramel_ratio_l62_62276


namespace least_possible_sum_of_two_factors_of_10_factorial_l62_62607

theorem least_possible_sum_of_two_factors_of_10_factorial : 
  ∃ (a b : ℕ), (a > 0) ∧ (b > 0) ∧ (a * b = 10!) ∧ (∀ (c d : ℕ), (c > 0) → (d > 0) → (c * d = 10!) → (a + b ≤ c + d)) ∧ (a + b = 3960) :=
by
  sorry

end least_possible_sum_of_two_factors_of_10_factorial_l62_62607


namespace falsity_of_proposition_implies_a_range_l62_62462

theorem falsity_of_proposition_implies_a_range (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, a * Real.sin x₀ + Real.cos x₀ ≥ 2) →
  a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
by 
  sorry

end falsity_of_proposition_implies_a_range_l62_62462


namespace factorization_of_difference_of_squares_l62_62313

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l62_62313


namespace wind_velocity_l62_62899

variables (k : ℝ) (V : ℝ) (P : ℝ) (A : ℝ)

-- Definitions from conditions in part a):
def proportionality_relation := P = k * A * V^2
def initial_condition : Prop := (1 = k * 1 * 16^2)
def new_conditions : Prop := (k = 1/256) ∧ (A = 9) ∧ (P = 36)

-- Lean 4 statement to prove the question
theorem wind_velocity (h₀ : proportionality_relation) 
                      (h₁ : initial_condition) 
                      (h₂ : new_conditions) :
  V = 32 :=
by {
  sorry
}

end wind_velocity_l62_62899


namespace transform_expression_l62_62764

theorem transform_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) : 
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := 
by 
  sorry

end transform_expression_l62_62764


namespace cooking_time_waffles_is_10_l62_62275

-- Let W be the time it takes to cook a batch of waffles
def cooking_time_waffles := ℕ

-- Given conditions:
-- 1. Cooking one chicken-fried steak takes 6 minutes.
def cooking_time_steak : ℕ := 6

-- 2. It takes 28 minutes to cook 3 chicken-fried steaks and a batch of waffles.
def total_time (W : cooking_time_waffles) : Prop := 3 * cooking_time_steak + W = 28

-- Prove that it takes 10 minutes to cook a batch of waffles
theorem cooking_time_waffles_is_10 : total_time 10 :=
sorry

end cooking_time_waffles_is_10_l62_62275


namespace triangle_side_AC_l62_62041

noncomputable section

variables {A B C D K E M N : Type*} [Point E] [Triangle A B C] [IsMedian B K A C] [IsAngleBisector B E] [IsAltitude A D] [Divides AD B K 3] [Divides AD B E 3] (h₁ : Length B A = 4)

theorem triangle_side_AC :
  length A C = √13 :=
sorry

end triangle_side_AC_l62_62041


namespace tiffany_lives_after_bonus_stage_l62_62985

theorem tiffany_lives_after_bonus_stage :
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  final_lives = 768 :=
by
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  exact sorry

end tiffany_lives_after_bonus_stage_l62_62985


namespace possible_N_values_l62_62956

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62956


namespace sum_of_four_consecutive_integers_prime_factor_l62_62161

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l62_62161


namespace find_n_l62_62824

theorem find_n (n : ℕ) (h₀ : (∑ i in range (n + 1), (i.digits.count 0)) = 5)
                  (h₉ : (∑ i in range (n + 1), (i.digits.count 9)) = 6) : n = 59 :=
  sorry

end find_n_l62_62824


namespace tree_height_by_time_boy_is_36_inches_l62_62241

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l62_62241


namespace largest_proper_n_exists_l62_62691

theorem largest_proper_n_exists : ∃ (n : ℕ), 
  (n = 7) ∧ 
  ∀ (grid : matrix (fin n) (fin n) bool) (selected : fin n → fin n), 
  (∀ (r : fin n), ∃ (c : fin n), selected r = c) ∧ 
  (∀ (rect : set (fin n × fin n)), rect.card ≥ n → ∃ (r c : fin n), (r, c) ∈ rect ∧ ∃ (i : fin n), r = selected i ∨ c = selected i) :=
by {
  existsi 7,
  split,
  { refl },
  { sorry }
}

end largest_proper_n_exists_l62_62691


namespace alternating_sum_formula_l62_62855

def alternating_sum (n : ℕ) : ℤ :=
  ∑ k in range n, (-1)^(k + 1) * (k + 1)^2

theorem alternating_sum_formula (n : ℕ) (hn : n > 0) :
  alternating_sum n = (-1)^(n + 1) :=
by
  sorry
  
end alternating_sum_formula_l62_62855


namespace bisector_P_on_perpendicular_bisector_CD_l62_62829

open EuclideanGeometry -- Assuming this contains all necessary definitions for Euclidean geometry.

-- Definitions of points and line segments.
variable (O A B C D E P : Point)
variable (circle : Circle O)
variable (AB : LineSegment A B)
variable (CD : LineSegment C D)
variable (E : Midpoint AB)
variable (CD_perpendicular_AB : Perpendicular CD AB E)
variable (O_in_circle : O ∈ circle)
variable (C_in_circle : C ∈ circle)
variable (C_on_arc_AB : OnArc C A B)
variable (P_bisector_OCD : AngleBisector P O C D)
variable (P_on_circle : P ∈ circle)

theorem bisector_P_on_perpendicular_bisector_CD :
  ∀ (C : Point) (CD : LineSegment C D), 
    C ∈ circle → 
    C_on_arc_AB C → 
    Perpendicular CD AB E → 
    P_bisector_OCD P O C D → 
    P ∈ perpendicular_bisector CD :=
by
  -- placeholder for the proof
  sorry

end bisector_P_on_perpendicular_bisector_CD_l62_62829


namespace min_value_frac_l62_62754

theorem min_value_frac (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) (h3 : a * c = 4) : 
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, y = (1 / c + 9 / a) → y ≥ x :=
by sorry

end min_value_frac_l62_62754


namespace right_isosceles_triangle_non_right_angle_l62_62785

theorem right_isosceles_triangle_non_right_angle 
  (h a : ℝ) (h_isosceles : ∀ (x : ℝ), h = x * sqrt 2 → a = x / sqrt 2) 
  (hyp_product : h * a^2 = 90) :
  ∃ θ : ℝ, θ = 45 :=
by
  -- Proof goes here, omitted as per instructions
  sorry

end right_isosceles_triangle_non_right_angle_l62_62785


namespace valid_number_of_two_humped_camels_within_range_l62_62963

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62963


namespace cube_root_of_27_l62_62110

theorem cube_root_of_27 : 
  ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
begin
  sorry
end

end cube_root_of_27_l62_62110


namespace part1_part1_eq_part1_gt_part2_part3_l62_62745

noncomputable def f (a x : ℝ) := x^2 - (a + 2) * x + 4
noncomputable def g (m x : ℝ) := m * x + 5 - 2 * m

theorem part1 (a x : ℝ) (h : a < 2) : (a ≤ x ∧ x ≤ 2) → f a x ≤ 4 - 2 * a := sorry

theorem part1_eq (x : ℝ) : f 2 x = 4 - 2 * 2 ↔ x = 2 := sorry

theorem part1_gt (a x : ℝ) (h : 2 < a) : (2 ≤ x ∧ x ≤ a) → f a x ≤ 4 - 2 * a := sorry

theorem part2 (a : ℝ) : (∀ x ∈ set.Icc 1 4, f a x + a + 1 ≥ 0) → a ≤ 4 := sorry

theorem part3 (m : ℝ) :
  (∀ x1 ∈ set.Icc 1 4, ∃ x2 ∈ set.Icc 1 4, f 2 x1 = g m x2) → (m ∈ set.Iic (-5/2) ∨ m ∈ set.Ici 5) := sorry

end part1_part1_eq_part1_gt_part2_part3_l62_62745


namespace math_problem_l62_62736

def p : Prop := 1 ∈ {x | (x + 2) * (x - 3) < 0}
def q : Prop := ∅ = {0}

theorem math_problem : p ∨ q := 
  sorry

end math_problem_l62_62736


namespace tristan_stops_after_finite_steps_l62_62179

theorem tristan_stops_after_finite_steps (n : ℕ) (a : Fin n → ℕ) :
  (∀ j k : Fin n, j < k → ¬ (a j ∣ a k) → (∃ m < n, a m = gcd (a j) (a k) ∧ a (m+1) = lcm (a j) (a k))) →
  ∃ m : ℕ, m < n ∧ (∀ i < n, a i = a (i - m)) :=
sorry

end tristan_stops_after_finite_steps_l62_62179


namespace negation_of_existence_l62_62069

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by
  sorry

end negation_of_existence_l62_62069


namespace paper_area_difference_l62_62762

def sheet1_length : ℕ := 14
def sheet1_width : ℕ := 12
def sheet2_length : ℕ := 9
def sheet2_width : ℕ := 14

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def combined_area (length : ℕ) (width : ℕ) : ℕ := 2 * area length width

theorem paper_area_difference :
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 84 := 
by 
  sorry

end paper_area_difference_l62_62762


namespace graph_edges_upper_bound_l62_62056

theorem graph_edges_upper_bound {G : Type*} [graph : SimpleGraph G]
  (V : Finset G) (E : Finset (Set G))
  (h1 : V.card = 2 * n)
  (h2 : E.card = m)
  (h3 : ∀ (e₁ e₂ : Set G), (e₁ ∈ E) → (e₂ ∈ E) → (∃ v w, e₁ = {v, w} ∧ e₂ = {v, w} → False))
  : m ≤ n^2 + 1 :=
begin
  sorry,
end

end graph_edges_upper_bound_l62_62056


namespace range_of_eccentricity_l62_62522

variable (a b c : ℝ) (P Q B : ℝ × ℝ)

def hyperbola (a b : ℝ) := ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1

def right_vertex (a : ℝ) := (a, 0)

def right_focus (c : ℝ) := (c, 0)

def chord_PQ (c b a : ℝ) := (c, b^2 / a) ∧ (c, -b^2 / a)

def point_B (x : ℝ) := (x, 0)

def perpendicular_condition (b a c : ℝ) (x : ℝ) := 
  (- (b^2 / a) / (c - x) * (b^2 / a) / (c - a)) = -1

def distance_condition (b a c : ℝ) (x : ℝ) := 
  |-(b^4 / (a^2 * (a - c)))| > 2 * (a + c)

def eccentricity (a b : ℝ) := ∃ e, e = sqrt (1 + (b / a)^2)

theorem range_of_eccentricity (a b c : ℝ) (h : a > b ∧ b > 0) 
    (B : ∃ x, point_B x) :
    hyperbola a b ∧ right_vertex a ∧ right_focus c ∧ chord_PQ c b a ∧ 
    ∀ x, perpendicular_condition b a c x ∧ distance_condition b a c x → 
    eccentricity a b → ∃ e, e > sqrt 3 := 
sorry

end range_of_eccentricity_l62_62522


namespace biscuits_more_than_cookies_l62_62203

theorem biscuits_more_than_cookies :
  let morning_butter_cookies := 20
  let morning_biscuits := 40
  let afternoon_butter_cookies := 10
  let afternoon_biscuits := 20
  let total_butter_cookies := morning_butter_cookies + afternoon_butter_cookies
  let total_biscuits := morning_biscuits + afternoon_biscuits
  total_biscuits - total_butter_cookies = 30 :=
by
  sorry

end biscuits_more_than_cookies_l62_62203


namespace ratio_of_areas_l62_62550

variables (s : ℝ) -- side length of the squares

-- conditions
def square_area (s : ℝ) := s * s
def midpoint (x y : ℝ) := (x + y) / 2

-- assumption of equal area squares and specific points as midpoints
variables (ABCD_area EFGH_area GHIJ_area : ℝ)
variables (A B C D E F G H I J : ℝ×ℝ)
variables (C_is_midpoint : C = midpoint IH GHIJ_area)
variables (D_is_midpoint : D = midpoint HE EFGH_area)

-- now we state the ratio to prove
theorem ratio_of_areas : 
  ( (polygon_area A J I C B) / (square_area s + square_area s + square_area s) ) = 1 / 3 :=
sorry

end ratio_of_areas_l62_62550


namespace origin_outside_circle_l62_62389

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle_eq := (x : ℝ) (y : ℝ) => x^2 + y^2 + 2 * a * x + 2 * y + (a - 1)^2 
  (circle_eq 0 0) > 0 := by {
  sorry
}

end origin_outside_circle_l62_62389


namespace books_sold_thu_l62_62818

-- Define the given conditions
def initial_stock : ℕ := 700
def books_sold_mon : ℕ := 50
def books_sold_tue : ℕ := 82
def books_sold_wed : ℕ := 60
def books_sold_fri : ℕ := 40
def unsold_percentage : ℝ := 0.60

-- Define the problem to be proved
theorem books_sold_thu : 
  let total_unsold_books := (unsold_percentage * initial_stock).to_nat in
  let total_books_sold := initial_stock - total_unsold_books in
  let total_books_sold_mon_to_fri := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_fri in
  let books_sold_thu := total_books_sold - total_books_sold_mon_to_fri in
  books_sold_thu = 48 :=
by {
  -- Proof would be here
  sorry
}

end books_sold_thu_l62_62818


namespace no_two_consecutive_periods_l62_62411

noncomputable def ways_to_schedule_subjects : ℕ :=
  120

theorem no_two_consecutive_periods :
  ∃ (subjects : Finset ℕ), 
  subjects.card = 4 ∧
  (∀ p ∈ subjects, p ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)) ∧
  ∀ (p1 p2 : ℕ), p1 ∈ subjects → p2 ∈ subjects → p1 ≠ p2 → |p1 - p2| ≠ 1 ∧ (ways_to_schedule_subjects = 120) := 
sorry

end no_two_consecutive_periods_l62_62411


namespace average_of_fractions_l62_62885

theorem average_of_fractions : 
  (let avg := (1/5 + 1/10) / 2 in 1/x = avg) → x = (20 : ℚ)/3 :=
by
  intros h
  rw ← h
  rw ← one_div
  sorry

end average_of_fractions_l62_62885


namespace hexagon_colorings_correct_l62_62894

noncomputable def hexagon_colorings : Nat :=
  let colors := ["blue", "orange", "purple"]
  2 -- As determined by the solution.

theorem hexagon_colorings_correct :
  hexagon_colorings = 2 :=
by
  sorry

end hexagon_colorings_correct_l62_62894


namespace possible_values_of_N_count_l62_62946

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62946


namespace no_real_roots_iff_range_m_l62_62463

open Real

theorem no_real_roots_iff_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + (m + 3) ≠ 0) ↔ (-2 < m ∧ m < 6) :=
by
  sorry

end no_real_roots_iff_range_m_l62_62463


namespace polar_rectangular_equivalence_l62_62081

-- Definitions for point A.
def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

-- Definitions for point B.
def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let rho := Real.sqrt (x^2 + y^2)
  let theta := Real.atan2 y x
  (rho, theta)

-- Proof of the problem stated above.
theorem polar_rectangular_equivalence :
  (polar_to_rectangular 2 (7 * Real.pi / 6) = (-Real.sqrt 3, -1)) ∧
  (rectangular_to_polar (-1) (Real.sqrt 3) = (2, 4 * Real.pi / 3)) :=
by
  split
  . sorry -- Proof that (2, 7π/6) -> (-√3, -1)
  . sorry -- Proof that (-1, √3) -> (2, 4π/3)

end polar_rectangular_equivalence_l62_62081


namespace birthday_check_value_l62_62204

theorem birthday_check_value : 
  ∃ C : ℝ, (150 + C) / 4 = C ↔ C = 50 :=
by
  sorry

end birthday_check_value_l62_62204


namespace asthma_expectation_l62_62860

theorem asthma_expectation (suffer_fraction : ℚ) (sample_size : ℚ) (expected_number : ℚ) : suffer_fraction = 1 / 8 → sample_size = 320 → expected_number = 40 → (suffer_fraction * sample_size = expected_number) :=
by
  intros h1 h2 h3
  rw [h1, h2]
  norm_num
  exact h3

end asthma_expectation_l62_62860


namespace probability_multiple_of_3_l62_62497

theorem probability_multiple_of_3 :
  let spins := ["LL", "LR", "LS", "RL", "RR", "RS", "SL", "SR", "SS"]
  let outcomes := [3, 6, 9]
  let n := 10
  let outcomes_prob := ∑ p in outcomes, 
                        if p == "SS" then 1/16 + 1/16 else 0
                      + if p == "RS" then 1/16 else 0
                      + if p == "SR" then 1/16 else 0
                      + if p == "LS" then 1/16 else 0
                      + if p == "SL" then 1/16 else 0
                      +(3/10 * 1/16) + 
                      +(4/10 * 1/16 * (1/16))
                      +(3/10 * 1/16)
                      + (3/10) in 
 (∑ (q : ∀ s in outcomes, 11/32 sorry))

end probability_multiple_of_3_l62_62497


namespace points_on_diagonals_l62_62844

def A := (0, 0)
def B := (1, 0)
def C := (1, 1)
def D := (0, 1)

def is_diagonal (P : ℝ × ℝ) : Prop :=
  (P.1 = P.2) ∨ (P.1 + P.2 = 1)

theorem points_on_diagonals (P : ℝ × ℝ) (h : (real.sqrt (P.1 ^ 2 + P.2 ^ 2) * real.sqrt ((1 - P.1) ^ 2 + (1 - P.2) ^ 2) +
    real.sqrt ((1 - P.1) ^ 2 + P.2 ^ 2) * real.sqrt (P.1 ^ 2 + (1 - P.2) ^ 2) = 1)) : 
    is_diagonal P :=
sorry

end points_on_diagonals_l62_62844


namespace gcd_840_1764_gcd_153_119_l62_62634

open Int

-- Definition of GCD using Euclidean algorithm
def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd_euclidean b (a % b)

-- Definition of GCD using subtraction method
def gcd_subtraction (a b : ℕ) : ℕ :=
  if a = b then a
  else if a > b then gcd_subtraction (a - b) b
  else gcd_subtraction a (b - a)

-- Lean statements for the proof problems
theorem gcd_840_1764 : gcd_euclidean 840 1764 = 84 := by
  sorry

theorem gcd_153_119 : gcd_subtraction 153 119 = 17 := by
  sorry

end gcd_840_1764_gcd_153_119_l62_62634


namespace distance_between_points_l62_62702

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem distance_between_points : distance (1, 4, -3) (-2, 1, 2) = sqrt 43 := by
  sorry

end distance_between_points_l62_62702


namespace part1_part2_l62_62372

theorem part1 (x m : ℝ) (C_n : ℕ → ℝ) (n : ℕ) :
  m = 3 ∧ 
  (x^2 - 2 * x - 8 ≤ ∑ i in finset.range (n + 1), (-1 : ℝ)^i * C_n i) ∧ 
  (|x - 2| ≤ m) →
  (-1 ≤ x ∧ x ≤ 4) :=
by sorry

theorem part2 (m : ℝ) (C_n : ℕ → ℝ) (n : ℕ) :
  (∀ x, (x^2 - 2 * x - 8 > ∑ i in finset.range (n + 1), (-1 : ℝ)^i * C_n i) → 
    (|x - 2| > m)) ∧ 
  (∃ x, |x - 2| ≤ m ∧ x^2 - 2 * x - 8 > ∑ i in finset.range (n + 1), (-1 : ℝ)^i * C_n i) →
  4 ≤ m :=
by sorry

end part1_part2_l62_62372


namespace hexagon_diagonals_intersect_l62_62728

def convex_hexagon (A B C D E F : Type) := sorry 

def divides_area (A B C D E F : Type) (diagonal1 diagonal2 : Type) := sorry 

def diagonals_intersect_at_single_point 
  (A B C D E F X Y Z : Type) 
  (h1 : convex_hexagon A B C D E F) 
  (h2 : divides_area A D) 
  (h3 : divides_area B E) 
  (h4 : divides_area C F) : 
  Prop := sorry

theorem hexagon_diagonals_intersect
  (A B C D E F : Type)
  (h1 : convex_hexagon A B C D E F)
  (h2 : divides_area A B C D E F A D)
  (h3 : divides_area A B C D E F B E)
  (h4 : divides_area A B C D E F C F) :
  diagonals_intersect_at_single_point A B C D E F sorry sorry sorry :=
sorry

end hexagon_diagonals_intersect_l62_62728


namespace camel_humps_l62_62935

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62935


namespace compare_neg_rationals_l62_62277

-- Definition and conditions
def abs_neg_one_third : ℚ := |(-1 / 3 : ℚ)|
def abs_neg_one_fourth : ℚ := |(-1 / 4 : ℚ)|

-- Problem statement
theorem compare_neg_rationals : (-1 : ℚ) / 3 < -1 / 4 :=
by
  -- Including the conditions here, even though they are straightforward implications in Lean
  have h1 : abs_neg_one_third = 1 / 3 := abs_neg_one_third
  have h2 : abs_neg_one_fourth = 1 / 4 := abs_neg_one_fourth
  -- We would include steps to show that -1 / 3 < -1 / 4 using the above facts
  sorry

end compare_neg_rationals_l62_62277


namespace max_smaller_boxes_fit_l62_62766

theorem max_smaller_boxes_fit (length_large width_large height_large : ℝ)
  (length_small width_small height_small : ℝ)
  (h1 : length_large = 6)
  (h2 : width_large = 5)
  (h3 : height_large = 4)
  (hs1 : length_small = 0.60)
  (hs2 : width_small = 0.50)
  (hs3 : height_small = 0.40) :
  length_large * width_large * height_large / (length_small * width_small * height_small) = 1000 := 
  by
  sorry

end max_smaller_boxes_fit_l62_62766


namespace fraction_exponentiation_multiplication_l62_62287

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l62_62287


namespace final_fish_stock_l62_62095

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end final_fish_stock_l62_62095


namespace solution_set_l62_62377

-- Definitions reflecting the given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Conditions of the problem
variables (f : ℝ → ℝ)
variable h_odd : odd_function f
variable h_mono_inc : monotonically_increasing f
variable h_f1_zero : f 1 = 0

-- The statement we aim to prove
theorem solution_set (x : ℝ) : f (x + 1) < 0 ↔ (-1 < x ∧ x < 0) ∨ (x < -2) :=
by
  sorry

end solution_set_l62_62377


namespace binom_floor_divisible_l62_62843

theorem binom_floor_divisible {p n : ℕ}
  (hp : Prime p) :
  (Nat.choose n p - n / p) % p = 0 := 
by
  sorry

end binom_floor_divisible_l62_62843


namespace valid_number_of_two_humped_camels_within_range_l62_62961

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62961


namespace total_purchase_rounded_l62_62048

theorem total_purchase_rounded 
  (p1 p2 p3 : ℝ)
  (h1 : p1 = 2.45)
  (h2 : p2 = 3.58)
  (h3 : p3 = 7.96) : 
  Int.round (p1 + p2 + p3) = 14 := 
by 
  sorry

end total_purchase_rounded_l62_62048


namespace sum_of_x_intercepts_l62_62536

theorem sum_of_x_intercepts (a b : ℕ) (h : 0 < a ∧ 0 < b) (h1 : ∃ x, ax+3 = 0 ∧ 4x+b = 0) :
  (∑ x in {x | x = -3 / a ∧ ab = 12}, x) = -6.5 :=
sorry

end sum_of_x_intercepts_l62_62536


namespace gcf_72_108_l62_62185

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62185


namespace BA2_plus_BH2_eq_CA2_plus_CH2_l62_62576

noncomputable def equal_distance
  (S A B C O : Point)
  (dist_eq : ∀ P : Point, P ∈ {S, A, B, C} → dist O P = R) : Prop :=
  dist O S = dist O A ∧ dist O A = dist O B ∧ dist O B = dist O C

noncomputable def midpoint (S A O : Point) : Prop :=
  dist S O = dist A O

noncomputable def height_of_pyramid (S H A B C : Point) : Prop :=
  ∃ (n : Vector), ∀ P : Point, P ∈ {A, B, C} → ⟪n, P⟫ = 0 ∧ ⟪n, S⟫ ≠ 0 ∧ ⟪n, H⟫ = 0

theorem BA2_plus_BH2_eq_CA2_plus_CH2
  {S A B C O H : Point}
  (midpoint_condition : midpoint S A O)
  (equal_distance_condition : equal_distance S A B C O)
  (height_condition : height_of_pyramid S H A B C) :
  dist B A ^ 2 + dist B H ^ 2 = dist C A ^ 2 + dist C H ^ 2 := sorry

end BA2_plus_BH2_eq_CA2_plus_CH2_l62_62576


namespace kelly_games_left_l62_62823

-- Definitions based on conditions
def original_games := 80
def additional_games := 31
def games_to_give_away := 105

-- Total games after finding more games
def total_games := original_games + additional_games

-- Number of games left after giving away
def games_left := total_games - games_to_give_away

-- Theorem statement
theorem kelly_games_left : games_left = 6 :=
by
  -- The proof will be here
  sorry

end kelly_games_left_l62_62823


namespace common_card_cost_l62_62986

def totalDeckCost (rareCost uncommonCost commonCost numRares numUncommons numCommons : ℝ) : ℝ :=
  (numRares * rareCost) + (numUncommons * uncommonCost) + (numCommons * commonCost)

theorem common_card_cost (numRares numUncommons numCommons : ℝ) (rareCost uncommonCost totalCost : ℝ) : 
  numRares = 19 → numUncommons = 11 → numCommons = 30 → 
  rareCost = 1 → uncommonCost = 0.5 → totalCost = 32 → 
  commonCost = 0.25 :=
by 
  intros 
  sorry

end common_card_cost_l62_62986


namespace hexagon_diagonals_l62_62719

-- Define a hexagon as having 6 vertices
def hexagon_vertices : ℕ := 6

-- From one vertex of a hexagon, there are (6 - 1) vertices it can potentially connect to
def potential_connections (vertices : ℕ) : ℕ := vertices - 1

-- Remove the two adjacent vertices to count diagonals
def diagonals_from_vertex (connections : ℕ) : ℕ := connections - 2

theorem hexagon_diagonals : diagonals_from_vertex (potential_connections hexagon_vertices) = 3 := by
  -- The proof is intentionally left as a sorry placeholder.
  sorry

end hexagon_diagonals_l62_62719


namespace peanuts_added_l62_62140

theorem peanuts_added (initial_peanuts final_peanuts added_peanuts : ℕ) 
(h1 : initial_peanuts = 10) 
(h2 : final_peanuts = 18) 
(h3 : final_peanuts = initial_peanuts + added_peanuts) : 
added_peanuts = 8 := 
by {
  sorry
}

end peanuts_added_l62_62140


namespace sufficient_condition_perp_l62_62834

-- Definitions of perpendicular relationship between planes and lines
variables {Plane Line : Type}
variable perp : Plane → Plane → Prop
variable perp_line_plane : Line → Plane → Prop

-- Specific types of planes and lines 
variables (α β γ : Plane) (m n l : Line)

-- Conditions from option B
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- Proof goal: under these conditions, show that m ⊥ β
theorem sufficient_condition_perp (n_perp_α : perp_line_plane n α)
                                   (n_perp_β : perp_line_plane n β)
                                   (m_perp_α : perp_line_plane m α) :
  perp_line_plane m β := 
begin
  sorry
end

end sufficient_condition_perp_l62_62834


namespace concurrent_incircle_tangent_lines_l62_62481

variables {A B C D E F K_a K_b K_c A_1 B_1 C_1 : Type}
variables [incircle A B C] [bc_midpoint A_1] [ca_midpoint B_1] [ab_midpoint C_1]
variables [touches_incircle D K_a] [touches_incircle E K_b] [touches_incircle F K_c]
variables [angle_bisector_AD D] [angle_bisector_BE E] [angle_bisector_CF F]

theorem concurrent_incircle_tangent_lines
  (h1 : touches_incircle D K_a) 
  (h2 : touches_incircle E K_b)
  (h3 : touches_incircle F K_c)
  (h4 : bc_midpoint A_1)
  (h5 : ca_midpoint B_1)
  (h6 : ab_midpoint C_1)
  : concurrent_lines_at_incircle A_1 K_a B_1 K_b C_1 K_c :=
sorry

end concurrent_incircle_tangent_lines_l62_62481


namespace box_surface_area_l62_62585

theorem box_surface_area (a b c : ℝ) (h1 : a = 10) (h2 : 4 * a + 4 * b + 4 * c = 180) (h3 : real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + c * a) = 1400 :=
by {
  sorry
}

end box_surface_area_l62_62585


namespace angles_cos_double_iff_increasing_l62_62493

theorem angles_cos_double_iff_increasing (A B C : ℝ) (hA : 0 < A) (hB : A < B) (hC : B < C) (hABC : C < π) : 
  (cos (2 * A) > cos (2 * B) ∧ cos (2 * B) > cos (2 * C)) ↔ (A < B ∧ B < C) :=
sorry

end angles_cos_double_iff_increasing_l62_62493


namespace total_students_l62_62146

theorem total_students (S : ℕ) (h1 : 0.50 * 0.25 * S = 125) : S = 1000 :=
by
  have h2 : 0.125 * S = 125 := by rwa [mul_assoc] at h1
  have h3 : S = 125 / 0.125 := by rwa [eq_div_of_mul_eq, div_self] at h2
  rw [div_self] at h3
  rw [eq_comm] at h3
  sorry

end total_students_l62_62146


namespace jerry_total_games_l62_62044

-- Conditions
def initial_games : ℕ := 7
def birthday_games : ℕ := 2

-- Statement
theorem jerry_total_games : initial_games + birthday_games = 9 := by sorry

end jerry_total_games_l62_62044


namespace binomial_8_choose_4_l62_62681

theorem binomial_8_choose_4 : nat.choose 8 4 = 70 :=
by sorry

end binomial_8_choose_4_l62_62681


namespace min_operations_needed_to_reduce_to_less_than_ten_l62_62205

-- Define a function that counts the number of perfect squares up to a given limit
def count_perfect_squares (n : ℕ) : ℕ :=
  (Nat.sqrt n).succ

-- Define a function that removes perfect squares and then odd numbers from a set of tiles
def reduce_tiles (n : ℕ) : ℕ :=
  let without_squares := n - count_perfect_squares n
  let without_odds := without_squares / 2
  without_odds

-- Define a function to compute the number of iterations needed to reduce to less than 10 tiles
def min_operations (initial_tiles : ℕ) : ℕ :=
  let rec count_ops (count : ℕ) (remaining : ℕ) :=
    if remaining < 10 then
      count
    else
      count_ops (count + 1) (reduce_tiles remaining)
  count_ops 0 initial_tiles

-- Define the problem statement
theorem min_operations_needed_to_reduce_to_less_than_ten :
  min_operations 150 = 4 :=
by simp only [min_operations, reduce_tiles, count_perfect_squares]; sorry

end min_operations_needed_to_reduce_to_less_than_ten_l62_62205


namespace impossible_seed_germinate_without_water_l62_62202

-- Definitions for the conditions
def heats_up_when_conducting (conducts : Bool) : Prop := conducts
def determines_plane (non_collinear : Bool) : Prop := non_collinear
def germinates_without_water (germinates : Bool) : Prop := germinates
def wins_lottery_consecutively (wins_twice : Bool) : Prop := wins_twice

-- The fact that a seed germinates without water is impossible
theorem impossible_seed_germinate_without_water 
  (conducts : Bool) 
  (non_collinear : Bool) 
  (germinates : Bool) 
  (wins_twice : Bool) 
  (h1 : heats_up_when_conducting conducts) 
  (h2 : determines_plane non_collinear) 
  (h3 : ¬germinates_without_water germinates) 
  (h4 : wins_lottery_consecutively wins_twice) :
  ¬germinates_without_water true :=
sorry

end impossible_seed_germinate_without_water_l62_62202


namespace sequence_behavior_l62_62840

theorem sequence_behavior (a : ℝ) (h : 0 < a ∧ a < 1) :
  (∀ n : ℕ, if odd n then x n < x (n+2) else x n > x (n+2))
where
  x : ℕ → ℝ
  | 0 => a
  | (n+1) => a ^ (x n) :=
by sorry

end sequence_behavior_l62_62840


namespace p_values_for_expression_l62_62295

def positive_integer (n : ℤ) : Prop := n > 0

theorem p_values_for_expression (p : ℕ) : 
  positive_integer (p) → positive_integer (4 * p + 34) / (3 * p - 7) →
  p = 3 ∨ p = 23 :=
sorry

end p_values_for_expression_l62_62295


namespace distance_between_intersections_l62_62032

-- Define the parametric equations of C1
def parametric_C1_x (t : ℝ) : ℝ := 6 + (Real.sqrt 3 / 2) * t
def parametric_C1_y (t : ℝ) : ℝ := (1 / 2) * t

-- Define the polar equation C2 transformed to Cartesian coordinates
def polar_C2_eq (x y : ℝ) : Prop := x^2 + y^2 = 10 * x

-- Define the points of intersection A and B and the parametric parameter values t1 and t2 corresponding to these points
def intersection_points (t1 t2 : ℝ) : Prop :=
  polar_C2_eq (parametric_C1_x t1) (parametric_C1_y t1) ∧
  polar_C2_eq (parametric_C1_x t2) (parametric_C1_y t2)

-- Define the distance |AB| between the two intersection points A and B
def distance_AB (t1 t2 : ℝ) : ℝ :=
  Real.sqrt ((t2 - t1)^2)

-- The main theorem
theorem distance_between_intersections (t1 t2 : ℝ)
  (h_intersect : intersection_points t1 t2) :
  distance_AB t1 t2 = 3 * Real.sqrt 11 :=
sorry

end distance_between_intersections_l62_62032


namespace find_n_value_l62_62378

theorem find_n_value (n : ℕ) (h : ∃ k : ℤ, n^2 + 5 * n + 13 = k^2) : n = 4 :=
by
  sorry

end find_n_value_l62_62378


namespace num_possible_values_l62_62927

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62927


namespace second_car_left_later_l62_62224

noncomputable def time_difference (speed1 speed2 time1 time2: ℝ) : ℝ := 
  (time1 * speed1) / speed2

theorem second_car_left_later {time1 time2 departure_total: ℝ} 
  (first_car_speed: ℝ) (first_car_travel_time: ℝ) (second_car_speed: ℝ) (meeting_time: ℝ) : 
  second_car_left_later departure_total = (meeting_time - time_difference first_car_speed second_car_speed first_car_travel_time time2) * 60 :=
  by
  -- Given conditions
  let departure_total := 90 -- in minutes
  let first_car_speed := 30 -- in mph
  let first_car_travel_time := 1.5 -- in hours
  let second_car_speed := 60 -- in mph
  let meeting_time := 1.5 -- in hours
  let time2 := 0.75 -- in hours (time for the second car to travel 45 miles)
  -- The second car left 45 minutes (0.75 hours * 60 minutes/hour) after the first car
  sorry

end second_car_left_later_l62_62224


namespace g_constant_term_l62_62511

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

-- Conditions from the problem
def f_has_constant_term_5 : f.coeff 0 = 5 := sorry
def h_has_constant_term_neg_10 : h.coeff 0 = -10 := sorry
def g_is_quadratic : g.degree ≤ 2 := sorry

-- Statement of the problem
theorem g_constant_term : g.coeff 0 = -2 :=
by
  have h_eq_fg : h = f * g := rfl
  have f_const := f_has_constant_term_5
  have h_const := h_has_constant_term_neg_10
  have g_quad := g_is_quadratic
  sorry

end g_constant_term_l62_62511


namespace even_function_a_eq_zero_l62_62436

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62436


namespace num_divisors_less_than_n_but_not_divide_n_l62_62065

noncomputable def n : ℕ := 2^29 * 5^17

theorem num_divisors_less_than_n_but_not_divide_n : 
  let n_squared := n^2 in
  let total_divisors_n_squared := (58 + 1) * (34 + 1) in
  let total_divisors_n := (29 + 1) * (17 + 1) in
  let divisors_less_than_n := (total_divisors_n_squared - 1) / 2 in
  let result := divisors_less_than_n - total_divisors_n in
  result = 492 :=
by
  after n def
  sorry

end num_divisors_less_than_n_but_not_divide_n_l62_62065


namespace four_painters_workdays_l62_62806

theorem four_painters_workdays :
  (∃ (c : ℝ), ∀ (n : ℝ) (d : ℝ), n * d = c) →
  (p5 : ℝ) (d5 : ℝ) (p5 * d5 = 7.5) →
  ∀ D : ℝ, 4 * D = 7.5 →
  D = (1 + 7/8) := 
by {
  sorry
}

end four_painters_workdays_l62_62806


namespace order_of_x_given_conditions_l62_62630

variables (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)

def system_equations :=
  x₁ + x₂ + x₃ = a₁ ∧
  x₂ + x₃ + x₄ = a₂ ∧
  x₃ + x₄ + x₅ = a₃ ∧
  x₄ + x₅ + x₁ = a₄ ∧
  x₅ + x₁ + x₂ = a₅

def a_descending_order :=
  a₁ > a₂ ∧
  a₂ > a₃ ∧
  a₃ > a₄ ∧
  a₄ > a₅

theorem order_of_x_given_conditions (h₁ : system_equations x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅) :
  a_descending_order a₁ a₂ a₃ a₄ a₅ →
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := sorry

end order_of_x_given_conditions_l62_62630


namespace solution1_solution2_l62_62476

open Real

noncomputable def problem1 (a b : ℝ) : Prop :=
a = 2 ∧ b = 2

noncomputable def problem2 (b : ℝ) : Prop :=
b = (2 * (sqrt 3 + sqrt 2)) / 3

theorem solution1 (a b : ℝ) (c : ℝ) (C : ℝ) (area : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : area = sqrt 3)
  (h4 : (1 / 2) * a * b * sin C = area) :
  problem1 a b :=
by sorry

theorem solution2 (a b : ℝ) (c : ℝ) (C : ℝ) (cosA : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : cosA = sqrt 3 / 3)
  (h4 : sin (arccos (sqrt 3 / 3)) = sqrt 6 / 3)
  (h5 : (a / (sqrt 6 / 3)) = (2 / (sqrt 3 / 2)))
  (h6 : ((b / ((3 + sqrt 6) / 6)) = (2 / (sqrt 3 / 2)))) :
  problem2 b :=
by sorry

end solution1_solution2_l62_62476


namespace evaluate_f_half_l62_62394

noncomputable def f (x : ℝ) : ℝ := 4^x + Real.logBase 2 x

theorem evaluate_f_half : f (1 / 2) = 1 :=
by
  sorry

end evaluate_f_half_l62_62394


namespace cube_root_of_27_l62_62108

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { rfl }

end cube_root_of_27_l62_62108


namespace probability_of_point_above_curve_l62_62552

-- Definitions of conditions
def is_valid_point (a b : ℤ) : Prop :=
  a ∈ ({1, 2, 3, 4} : Set ℤ) ∧ b ∈ ({1, 2, 3, 4} : Set ℤ) ∧ b < (8 * a^3) / 3

def count_valid_pairs : ℕ := Finset.card $ Finset.filter (λ p : ℤ × ℤ, is_valid_point p.1 p.2) $ Finset.product (Finset.range 5) (Finset.range 5)

def total_pairs : ℕ := 4 * 4

-- Main theorem
theorem probability_of_point_above_curve :
  count_valid_pairs = 14 ∧ total_pairs = 16 →
  (count_valid_pairs / total_pairs : ℚ) = 7 / 8 :=
sorry

end probability_of_point_above_curve_l62_62552


namespace solution_l62_62012

noncomputable def problem_statement (x y : ℝ) : Prop :=
  sqrt (x - 2) + (x - y - 12)^2 = 0

theorem solution (x y : ℝ) (h : problem_statement x y) : real.cbrt (x + y) = -2 :=
sorry

end solution_l62_62012


namespace find_fraction_l62_62988

noncomputable def condition_eq : ℝ := 5
noncomputable def condition_gq : ℝ := 7

theorem find_fraction {FQ HQ : ℝ} (h : condition_eq * FQ = condition_gq * HQ) :
  FQ / HQ = 7 / 5 :=
by
  have eq_mul : condition_eq = 5 := by rfl
  have gq_mul : condition_gq = 7 := by rfl
  rw [eq_mul, gq_mul] at h
  have h': 5 * FQ = 7 * HQ := h
  field_simp [←h']
  sorry

end find_fraction_l62_62988


namespace product_of_three_numbers_l62_62911

theorem product_of_three_numbers:
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ 
    a = 2 * (b + c) ∧ 
    b = 5 * c ∧ 
    a * b * c = 2500 / 9 :=
by {
  sorry
}

end product_of_three_numbers_l62_62911


namespace compare_neg_rationals_l62_62278

-- Definition and conditions
def abs_neg_one_third : ℚ := |(-1 / 3 : ℚ)|
def abs_neg_one_fourth : ℚ := |(-1 / 4 : ℚ)|

-- Problem statement
theorem compare_neg_rationals : (-1 : ℚ) / 3 < -1 / 4 :=
by
  -- Including the conditions here, even though they are straightforward implications in Lean
  have h1 : abs_neg_one_third = 1 / 3 := abs_neg_one_third
  have h2 : abs_neg_one_fourth = 1 / 4 := abs_neg_one_fourth
  -- We would include steps to show that -1 / 3 < -1 / 4 using the above facts
  sorry

end compare_neg_rationals_l62_62278


namespace possible_N_values_l62_62974

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62974


namespace calorie_difference_l62_62047

theorem calorie_difference (calories_burrito : ℕ) (price_burrito : ℕ) (burritos : ℕ)
                           (calories_burger : ℕ) (price_burger : ℕ) (burgers : ℕ) :
    let total_calories_burrito := burritos * calories_burrito
    let total_calories_burger := burgers * calories_burger
    let calories_per_dollar_burrito := total_calories_burrito / price_burrito
    let calories_per_dollar_burger := total_calories_burger / price_burger
    calories_burrito = 120 ∧ price_burrito = 6 ∧ burritos = 10 ∧ 
    calories_burger = 400 ∧ price_burger = 8 ∧ burgers = 5 →
    (calories_per_dollar_burger - calories_per_dollar_burrito) = 50 :=
by
  intros calories_burrito price_burrito burritos calories_burger price_burger burgers
         total_calories_burrito total_calories_burger calories_per_dollar_burrito calories_per_dollar_burger h
  sorry

end calorie_difference_l62_62047


namespace shift_down_two_units_l62_62571

def original_function (x : ℝ) : ℝ := 2 * x + 1

def shifted_function (x : ℝ) : ℝ := original_function x - 2

theorem shift_down_two_units :
  ∀ x : ℝ, shifted_function x = 2 * x - 1 :=
by 
  intros x
  simp [shifted_function, original_function]
  sorry

end shift_down_two_units_l62_62571


namespace increasing_on_neg_infty_l62_62201

def f (x : ℝ) : ℝ := x^2 - 4 * x
def g (x : ℝ) : ℝ := 3 * x + 1
def h (x : ℝ) : ℝ := 3^(-x)
def t (x : ℝ) : ℝ := Real.tan x

theorem increasing_on_neg_infty (x : ℝ) : 
  (∀ x y, x < y → g x < g y) ∧ 
  ¬ (∀ x y, x < y → f x < f y) ∧ 
  ¬ (∀ x y, x < y → h x < h y) ∧ 
  ¬ (∀ x y, x < y → t x < t y) := by
    sorry

end increasing_on_neg_infty_l62_62201


namespace internal_diagonal_passes_through_cubes_l62_62639

def dimensions := (160, 330, 380)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem internal_diagonal_passes_through_cubes :
  let d := dimensions in
  let gcd_xy := gcd d.1 d.2 in
  let gcd_yz := gcd d.2 d.3 in
  let gcd_zx := gcd d.3 d.1 in
  let gcd_xyz := gcd (gcd d.1 d.2) d.3 in
  d.1 + d.2 + d.3 - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 810 :=
by
  let d := dimensions
  have gcd_xy := gcd d.1 d.2
  have gcd_yz := gcd d.2 d.3
  have gcd_zx := gcd d.3 d.1
  have gcd_xyz := gcd (gcd d.1 d.2) d.3
  show d.1 + d.2 + d.3 - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 810
  sorry

end internal_diagonal_passes_through_cubes_l62_62639


namespace prove_a_zero_l62_62418

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62418


namespace num_possible_values_l62_62926

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62926


namespace chemists_reagents_l62_62694

theorem chemists_reagents (n : ℕ) (chemists : Fin n → Set (Fin n)) 
  (send_package : ∀ (i j : Fin n), i ≠ j → Prop) :

  (∀ i, chemists i = ⋃ j, send_package j i → chemists j) →
  (∀ i j, send_package i j → chemists j = chemists j ∪ chemists i) →
  -- The main condition ensuring Every chemist received each of the reagents
  (∀ i, chemists i = Finset.univ.toSet) →
  -- No chemist received any other chemist's reagent more than once
  (∀ i j, ∀ k, send_package i j → k ∈ chemists i → k ∈ chemists j) →
  -- Some chemists received packages that included their own reagent
  (∃ i j, send_package i j ∧ i ∈ chemists j) →
  -- Prove that at least n-1 chemists received their own reagent in a package.
  (Finset.card { i : Fin n | ∃ j, send_package j i ∧ i ∈ chemists i} ≥ n - 1) :=
begin
  sorry
end

end chemists_reagents_l62_62694


namespace least_integer_x_l62_62323

theorem least_integer_x (x : ℤ) : (2 * |x| + 7 < 17) → x = -4 := by
  sorry

end least_integer_x_l62_62323


namespace hungarian_teams_probability_l62_62175

theorem hungarian_teams_probability :
  ∀ (total_teams hungarian_teams pairing_choices favorable_choices : ℕ)
  (all_pairings equally_likely : Prop),
  total_teams = 8 →
  hungarian_teams = 3 →
  pairing_choices = total_teams - 1 →
  favorable_choices = 5 * 4 * 3 →
  all_pairings = 7 * 5 * 3 ∧ equally_likely →
  (favorable_choices : ℚ) / (all_pairings : ℚ) = 4 / 7 := by
  intros total_teams hungarian_teams pairing_choices favorable_choices all_pairings equally_likely
  intros h_total h_hungarian h_pairing_choices h_favorable_choices h_all_eq
  rw [h_total, h_hungarian, h_pairing_choices, h_favorable_choices, h_all_eq.left]
  norm_num
  sorry

end hungarian_teams_probability_l62_62175


namespace all_rhombuses_inscribed_circumscribed_same_circle_l62_62082

noncomputable def rhombus_inscribed_in_ellipse (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) 
  (r : ℝ) (h₃ : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∃ (k : ℝ), (r * r = 1 / 
  ((1 / (x^2 + k * k * y^2)) + (1 / (y^2 + (1 / k)^2 * x^2))))): Prop := 
  1 / r^2 = 1 / a^2 + 1 / b^2

theorem all_rhombuses_inscribed_circumscribed_same_circle (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
  (r : ℝ) (h₃ : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∃ (k : ℝ), (r * r = 1 / 
  ((1 / (x^2 + k * k * y^2)) + (1 / (y^2 + (1 / k)^2 * x^2))))): 
  rhombus_inscribed_in_ellipse a b h₁ h₂ r h₃ :=
begin
  -- Proof omitted
  sorry
end

end all_rhombuses_inscribed_circumscribed_same_circle_l62_62082


namespace find_f_2_l62_62753

variables {a b c : ℝ}

def f(x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x + 1

theorem find_f_2 (h : f (-2) = -1) : f 2 = 3 :=
by
  sorry

end find_f_2_l62_62753


namespace time_2880717_minutes_ago_l62_62292

theorem time_2880717_minutes_ago (current_hours : ℕ) (current_minutes : ℕ) (subtract_minutes : ℕ) : 
  current_hours = 18 → current_minutes = 27 → subtract_minutes = 2880717 → 
  let minutes_in_hour := 60,
      hours_in_day := 24,
      total_current_minutes := current_hours * minutes_in_hour + current_minutes,
      total_earlier_minutes := total_current_minutes - subtract_minutes,
      earlier_days := total_earlier_minutes / (minutes_in_hour * hours_in_day),
      earlier_minutes := total_earlier_minutes % (minutes_in_hour * hours_in_day),
      final_hours := earlier_minutes / minutes_in_hour,
      final_minutes := earlier_minutes % minutes_in_hour
  in final_hours = 6 ∧ final_minutes = 30 := 
by intros; sorry

end time_2880717_minutes_ago_l62_62292


namespace digitized_number_exists_l62_62687

def is_digitized_number (n : ℕ) : Prop :=
  let digits := (List.ofDigits 10 n).reverse
  n >= 10^9 ∧ n < 10^10 ∧
  (List.range 10).all (λ k, (List.count (Eq k) digits) = digits.nthLe k (by auto))

theorem digitized_number_exists :
  is_digitized_number 6210001000 :=
by {
  have h : 6210001000 >= 10^9 ∧ 6210001000 < 10^10, by norm_num,
  split,
  { exact h.left },
  split,
  { exact h.right },
  {
    change (List.range 10).all _,
    simp,
    split,
    { refl },
    -- verify that each digit count matches position
    repeat { split; try { refl } }, 
    -- the remaining verify steps are similar, ensuring all digits match positions,
    -- listing all cases explicitly.
    sorry
  }
}

end digitized_number_exists_l62_62687


namespace mary_total_cards_l62_62849

def mary_initial_cards := 33
def torn_cards := 6
def cards_given_by_sam := 23

theorem mary_total_cards : mary_initial_cards - torn_cards + cards_given_by_sam = 50 :=
  by
    sorry

end mary_total_cards_l62_62849


namespace gardener_hourly_wage_l62_62269

-- Conditions
def rose_bushes_count : Nat := 20
def cost_per_rose_bush : Nat := 150
def hours_per_day : Nat := 5
def days_worked : Nat := 4
def soil_volume : Nat := 100
def cost_per_cubic_foot_soil : Nat := 5
def total_cost : Nat := 4100

-- Theorem statement
theorem gardener_hourly_wage :
  let cost_of_rose_bushes := rose_bushes_count * cost_per_rose_bush
  let cost_of_soil := soil_volume * cost_per_cubic_foot_soil
  let total_material_cost := cost_of_rose_bushes + cost_of_soil
  let labor_cost := total_cost - total_material_cost
  let total_hours_worked := hours_per_day * days_worked
  (labor_cost / total_hours_worked) = 30 := 
by {
  -- Proof placeholder
  sorry
}

end gardener_hourly_wage_l62_62269


namespace find_number_l62_62021

theorem find_number (N : ℕ) (h1 : N / 3 = 8) (h2 : N / 8 = 3) : N = 24 :=
by
  sorry

end find_number_l62_62021


namespace possible_N_values_l62_62981

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62981


namespace smallest_c_plus_d_l62_62842

theorem smallest_c_plus_d (c d : ℝ) (h1 : 0 < c) (h2 : 0 < d)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) : c + d ≥ (16 / 3) * real.sqrt 3 :=
by
  sorry

end smallest_c_plus_d_l62_62842


namespace hexagon_area_correct_l62_62087

noncomputable def point := (Real × Real)

def A : point := (0, 0)
def C : point := (5, 3)

def distance (p1 p2 : point) : Real :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def side_length (p1 p2 : point) : Real :=
  distance p1 p2 / 2

def hexagon_area (s : Real) : Real :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem hexagon_area_correct :
  let s := side_length A C in
  hexagon_area s = 51 * Real.sqrt 3 / 4 :=
by
  sorry

end hexagon_area_correct_l62_62087


namespace white_spotted_mushrooms_total_l62_62267

variable (red_mushrooms_bill : ℕ)
variable (brown_mushrooms_bill : ℕ)
variable (green_mushrooms_ted : ℕ)
variable (blue_mushrooms_ted : ℕ)
variable (white_spots_blue : ℚ)
variable (white_spots_red : ℚ)
variable (white_spots_brown : ℚ)

-- Define the conditions given in the problem
def conditions : Prop :=
  red_mushrooms_bill = 12 ∧
  brown_mushrooms_bill = 6 ∧
  green_mushrooms_ted = 14 ∧
  blue_mushrooms_ted = 6 ∧
  white_spots_blue = 0.5 ∧  -- Half of the blue mushrooms have white spots
  white_spots_red = 2 / 3 ∧  -- Two-thirds of the red mushrooms have white spots
  white_spots_brown = 1  -- All of the brown mushrooms have white spots

-- Define the theorem to be proven
theorem white_spotted_mushrooms_total (hb : conditions) : 
  (red_mushrooms_bill * (white_spots_red) + 
   brown_mushrooms_bill * white_spots_brown + 
   blue_mushrooms_ted * white_spots_blue).toNat = 17 := 
by 
  sorry

end white_spotted_mushrooms_total_l62_62267


namespace sum_exponential_p_k_l62_62545

variable (x : Real) (n : ℕ)
variable (p : ℕ → Real)
variable (log_x_k : ℕ → Real)

-- Conditions
def p_k (k : ℕ) := (log_x_k k) ** k / log (x ** k)

-- Goal to prove
theorem sum_exponential_p_k :
  (∑ k in Finset.range n, x ^ p_k k) = (n / 2) * x ^ p_k (n + 1) := 
sorry

end sum_exponential_p_k_l62_62545


namespace Malou_third_quiz_score_l62_62075

theorem Malou_third_quiz_score (q1 q2 q3 : ℕ) (avg_score : ℕ) (total_quizzes : ℕ) : 
  q1 = 91 ∧ q2 = 90 ∧ avg_score = 91 ∧ total_quizzes = 3 → q3 = 92 :=
by
  intro h
  sorry

end Malou_third_quiz_score_l62_62075


namespace area_of_rectangle_l62_62214

-- Declare the points and distances involved
variables {A B C D G H A' : Point}
variable [PrettyPrint Point]

-- Declare the given conditions as hypotheses
axiom h1 : IsRectangle ABCD
axiom h2 : Distance B G = 12
axiom h3 : Distance A G = 5
axiom h4 : Distance D H = 7

-- Prove the area of the rectangle
theorem area_of_rectangle : area ABCD = 455 := by
  sorry

end area_of_rectangle_l62_62214


namespace common_roots_product_l62_62125

theorem common_roots_product
  (p q r s : ℝ)
  (hpqrs1 : p + q + r = 0)
  (hpqrs2 : pqr = -20)
  (hpqrs3 : p + q + s = -4)
  (hpqrs4 : pqs = -80)
  : p * q = 20 :=
sorry

end common_roots_product_l62_62125


namespace num_of_irrational_numbers_l62_62666

-- Defining each number
def num1 : ℝ := 25 / 7
def num2 : ℝ := Real.pi / 3
def num3 : ℝ := 3.14159
def num4 : ℝ := -Real.sqrt 9
noncomputable def num5 : ℝ := 0.3030030003 -- this notation represents the non-terminating, non-repeating decimal

-- Define which numbers are irrational
def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Proving the number of irrational numbers in the list
theorem num_of_irrational_numbers : 
  (if is_irrational num1 then 1 else 0) + 
  (if is_irrational num2 then 1 else 0) + 
  (if is_irrational num3 then 1 else 0) + 
  (if is_irrational num4 then 1 else 0) +
  (if is_irrational num5 then 1 else 0) = 2 := 
by sorry

end num_of_irrational_numbers_l62_62666


namespace find_BD_l62_62509

noncomputable def right_triangle_area (a b c : ℝ) (right_angle : a^2 + b^2 = c^2) : ℝ := 1/2 * a * b

theorem find_BD
  (A B C D : Point)
  (hB : ∠B = 90)
  (hArea : right_triangle_area 30 BD 225)
  (hCircle : circle_intersects_diameter BC D A C B)
  : BD = 15 := by
  sorry

end find_BD_l62_62509


namespace find_g_function_l62_62892

theorem find_g_function (g : ℝ → ℝ) 
  (h₁ : g 1 = 1)
  (h₂ : ∀ x y : ℝ, g (x + y) = 2^y * g x + 1^x * g y) :
  ∀ x : ℝ, g x = 2^x - 1 :=
by
  sorry

end find_g_function_l62_62892


namespace scarlet_savings_l62_62543

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l62_62543


namespace parametric_eq_of_line_l62_62896

theorem parametric_eq_of_line (x y t : ℝ) 
(h_eq : y = 2 * x + 1) 
(hx : x = t - 1) 
(hy : y = 2 * t - 1) : 
y = 2 * (t - 1) + 1 :=
by 
  -- We need to show that the given parametric equations satisfy the original line equation
  have h1 : y = 2 * (t - 1) + 1, by
    sorry
  exact h1

end parametric_eq_of_line_l62_62896


namespace valid_number_of_two_humped_camels_within_range_l62_62960

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62960


namespace point_Q_coordinates_l62_62862

theorem point_Q_coordinates :
    ∀ (P : ℝ × ℝ), P = (1, 0) →
    ∀ (θ : ℝ), θ = (2 * Real.pi)/3 →
    ∃ Q : ℝ × ℝ,
    Q = ((Real.cos θ, -Real.sin θ), (Real.sin θ, Real.cos θ)) • (1, 0) →
    Q = (-1/2 : ℝ, Real.sqrt 3 / 2 : ℝ) :=
by
    intros
    use (-1 / 2, Real.sqrt 3 / 2)
    split
    show θ = (2 * Real.pi) / 3, from by assumption
    show Q = _,
    from by sorry -- just to indicate incomplete proof
    sorry

end point_Q_coordinates_l62_62862


namespace smallest_positive_multiple_l62_62712

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l62_62712


namespace prime_factor_of_sum_of_four_consecutive_integers_l62_62173

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l62_62173


namespace prove_a_zero_l62_62419

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62419


namespace calculate_taxes_l62_62774

def gross_pay : ℝ := 4500
def tax_rate_1 : ℝ := 0.10
def tax_rate_2 : ℝ := 0.15
def tax_rate_3 : ℝ := 0.20
def income_bracket_1 : ℝ := 1500
def income_bracket_2 : ℝ := 2000
def income_bracket_remaining : ℝ := gross_pay - income_bracket_1 - income_bracket_2
def standard_deduction : ℝ := 100

theorem calculate_taxes :
  let tax_1 := tax_rate_1 * income_bracket_1
  let tax_2 := tax_rate_2 * income_bracket_2
  let tax_3 := tax_rate_3 * income_bracket_remaining
  let total_tax := tax_1 + tax_2 + tax_3
  let tax_after_deduction := total_tax - standard_deduction
  tax_after_deduction = 550 :=
by
  sorry

end calculate_taxes_l62_62774


namespace last_triangle_perimeter_l62_62833

def triangle_sequence (a b c : ℝ) (n : ℕ) : ℝ × ℝ × ℝ :=
  let AD := (b + c - a) / 2 ^ (n - 1)
  let BE := (a + c - b) / 2 ^ (n - 1)
  let CF := (a + b - c) / 2 ^ (n - 1)
  (AD, BE, CF)

noncomputable def perimeter (a b c : ℝ) (n : ℕ) : ℝ :=
  2 * ((a + b + c) / 2 ^ (n - 1))

theorem last_triangle_perimeter (a b c : ℝ) :
  a = 101 → b = 102 → c = 100 →
  perimeter a b c 2 = 151.5 :=
by intros; 
   sorry

end last_triangle_perimeter_l62_62833


namespace blocks_used_l62_62870

theorem blocks_used (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 78) (h_left : initial_blocks - used_blocks = 59) : used_blocks = 19 := by
  sorry

end blocks_used_l62_62870


namespace possible_values_of_N_count_l62_62942

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62942


namespace quadratic_roots_relationship_l62_62330

-- Define the problem conditions
def quadratic_equation (a b c x : ℝ) := a * x^2 + b * x + c = 0

def roots_reciprocal (r : ℝ) := r * (1 / r) = 1

def sum_four_times_product (r : ℝ) := r + (1 / r) = 4

-- Prove the conditions imply the relationships
theorem quadratic_roots_relationship (a b c : ℝ) (h : ∀ x, quadratic_equation a b c x) 
  (r : ℝ) (h_reciprocal : roots_reciprocal r) (h_sum_product : sum_four_times_product r) :
  a = c ∧ b = -4 * a :=
by 
  -- Normalization and conditions derived from the problem
  have h_normalized : quadratic_equation 1 (b / a) (c / a) r := by sorry
  
  -- Applying Vieta's formulas
  have sum_roots : r + 1 / r = -b / a := by sorry
  have product_roots : r * (1 / r) = c / a := by sorry

  -- Given conditions
  have sum_condition : r + 1 / r = 4 := h_sum_product
  have product_condition : r * (1 / r) = 1 := h_reciprocal

  -- Derive the relationships
  have b_val : b = -4 * a := by sorry
  have c_val : c = a := by sorry

  -- Combine the results
  exact ⟨c_val, b_val⟩

end quadratic_roots_relationship_l62_62330


namespace sum_xyz_l62_62006

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_xyz :
  (∀ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 →
  x + y + z = 932) := 
by
  sorry

end sum_xyz_l62_62006


namespace max_numbers_within_1983_l62_62608

theorem max_numbers_within_1983 (n : ℕ) (h_n : n = 1983) :
    ∃ (S : set ℕ), (∀ a b ∈ S, a * b ∉ S) ∧ S.card = 1939 :=
sorry

end max_numbers_within_1983_l62_62608


namespace regression_line_equation_chi_square_relation_l62_62662

-- Assuming the necessary dataset and functions are defined

-- Data1 for regression line calculation
def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [120, 100, 90, 75, 65]
def sum_xy : ℝ := 1215
def n : ℝ := 5
def mean_x : ℝ := (x.sum) / n
def mean_y : ℝ := (y.sum) / n

-- Data2 for chi-square test
def a : ℝ := 15
def b : ℝ := 10
def c : ℝ := 25
def d : ℝ := 50
def total_accidents : ℝ := 100
def chi_square_critical_value_95 : ℝ := 3.841

-- Part 1: Statement for Regression Line Equation
theorem regression_line_equation :
  let β := (sum_xy - n * mean_x * mean_y) / (x.foldr (λ xi acc, xi^2 + acc) 0 - n * mean_x^2),
      α := mean_y - β * mean_x in
  β = -13.5 ∧ α = 130.5 :=
by  sorry

-- Part 2: Statement for Chi-Square Test
theorem chi_square_relation :
  let chi_square := (total_accidents * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  chi_square > chi_square_critical_value_95 :=
by sorry

end regression_line_equation_chi_square_relation_l62_62662


namespace description_of_S_l62_62831

def S (x y : ℝ) : Prop := 
  (3 = x + 2 ∧ y - 4 ≤ 3) ∨ (3 = y - 4 ∧ x + 2 ≤ 3) ∨ (x + 2 = y - 4 ∧ 3 ≤ x + 2)

theorem description_of_S : 
  ∀ (x y : ℝ), S x y → S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 7) ∨ (p.1 ≤ 1 ∧ p.2 = 7) ∨ (p.2 = p.1 + 6 ∧ p.1 ≥ 1 ∧ p.2 ≥ 7)} :=
by
  sorry

end description_of_S_l62_62831


namespace sum_of_roots_is_30_l62_62572

-- let g be a function ℝ → ℝ, satisfying the symmetry condition
variable (g : ℝ → ℝ)
variable (h_symm : ∀ x : ℝ, g (5 + x) = g (5 - x))

-- let g(x) = 0 have exactly six distinct real roots
variable (roots : Fin 6 → ℝ)
variable (h_roots : ∀ i, g (roots i) = 0)
variable (h_distinct : Function.Injective roots)

-- the goal is to prove that the sum of these roots is 30
theorem sum_of_roots_is_30 : (∑ i, roots i) = 30 := 
  sorry

end sum_of_roots_is_30_l62_62572


namespace find_angle_B_find_triangle_area_l62_62475

open Real

theorem find_angle_B (B : ℝ) (h : sqrt 3 * sin (2 * B) = 1 - cos (2 * B)) : B = π / 3 :=
sorry

theorem find_triangle_area (BC A B : ℝ) (hBC : BC = 2) (hA : A = π / 4) (hB : B = π / 3) :
  let AC := BC * (sin B / sin A)
  let C := π - A - B
  let area := (1 / 2) * AC * BC * sin C
  area = (3 + sqrt 3) / 2 :=
sorry


end find_angle_B_find_triangle_area_l62_62475


namespace camel_humps_l62_62938

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62938


namespace graph_symmetric_about_x_eq_pi_div_8_l62_62390

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem graph_symmetric_about_x_eq_pi_div_8 :
  ∀ x, f (π / 8 - x) = f (π / 8 + x) :=
sorry

end graph_symmetric_about_x_eq_pi_div_8_l62_62390


namespace dj_eq_dl_l62_62063

noncomputable def point := ℝ × ℝ
noncomputable def line := point × point
noncomputable def hexagon := point × point × point × point × point × point
noncomputable def circle := point × ℝ

variables (A B C D E F O J K L : point)
variables (ω : circle)
variables (EO : line)

-- Definitions needed based on the conditions:
-- 1. ABCDEF is a convex hexagon tangent to circle ω
def is_convex_hexagon_tangent_to (h : hexagon) (c : circle) : Prop := sorry

-- 2. Circumcircle of triangle ACE is concentric with ω
def are_concentric (c1 c2 : circle) : Prop := sorry

-- 3. J is the foot of the perpendicular from B to CD
def is_foot_of_perpendicular (p point) (l : line) : Prop := sorry

-- 4. The perpendicular from B to DF intersects EO at K
def intersection_of_perpendicular (p : point) (l1 l2 : line) : point := sorry

-- 5. L is the foot of the perpendicular from K to DE
-- (assuming definition from earlier point)
-- Combination of conditions to reach the proof statement
theorem dj_eq_dl (h: hexagon) (c: circle) (p1 p2 : point)
  (hc1: is_convex_hexagon_tangent_to h c)
  (hc2: are_concentric (circumcircle_of_triangle (A, C, E)) c)
  (hc3: is_foot_of_perpendicular p1 (C, D))
  (hc4: p2 = intersection_of_perpendicular B (DF, EO))
  (hc5: is_foot_of_perpendicular p2 (D, E)) :
  dist D J = dist D L :=
sorry

end dj_eq_dl_l62_62063


namespace final_remaining_coffee_l62_62858

-- Definitions based on conditions
def initial_coffee_amount : ℝ := 12
def consumed_on_way_to_work : ℝ := initial_coffee_amount / 4
def consumed_at_office : ℝ := initial_coffee_amount / 2
def consumed_when_remembered : ℝ := 1

-- Total consumed
def total_consumed : ℝ :=
  consumed_on_way_to_work + consumed_at_office + consumed_when_remembered

-- Remaining coffee
def remaining_coffee : ℝ :=
  initial_coffee_amount - total_consumed

-- Proof statement
theorem final_remaining_coffee : remaining_coffee = 2 := by
  sorry

end final_remaining_coffee_l62_62858


namespace sum_f_2023_l62_62122

noncomputable def f : ℤ → ℤ := λ x, sorry

theorem sum_f_2023 :
  (∀ x : ℤ, f(x + 2) = -f(x + 1) - f(x)) →
  (∀ x : ℤ, f(x) = f(2 - x)) →
  f(365) = -1 →
  ∑ k in finset.range 2023, f(k + 1) = 2 :=
by
  intros h1 h2 h3
  sorry

end sum_f_2023_l62_62122


namespace card_probability_l62_62005

theorem card_probability :
  let deck := (range 52).toFinset,
      suits := {0, 1, 2, 3},
      hearts := {i | i % 4 = 0} in
  ∃ (picked_cards : List ℕ), picked_cards.length = 5 ∧
  (∀ card ∈ picked_cards, card ∈ deck) ∧
  picked_cards (1) % 4 = 0 ∧
  (picked_cards.drop 4).forall (λ card, card % 4 ≠ 0) ∧
  sorted picked_cards ∧
  let prob := (1 : ℚ) * (13/17) * (13/25) * (13/49) * (1/4) in
  prob = 2197 / 83300 :=
sorry

end card_probability_l62_62005


namespace even_function_a_zero_l62_62427

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62427


namespace sum_of_product_digits_eq_66_l62_62798

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

def p : Nat := 33331111
def q : Nat := 77772222

theorem sum_of_product_digits_eq_66 :
  sum_of_digits (p * q) = 66 :=
by
  sorry

end sum_of_product_digits_eq_66_l62_62798


namespace baseball_card_ratio_l62_62265

theorem baseball_card_ratio 
  (newCards : ℕ) 
  (remainingCards : ℕ)
  (h_newCards : newCards = 4)
  (h_remainingCards : remainingCards = 34) : 
  (let totalCards := remainingCards + newCards in
   let eatenCards := totalCards - remainingCards in
   ∃ gcd : ℕ, gcd = Nat.gcd eatenCards totalCards ∧ (eatenCards / gcd = 2) ∧ (totalCards / gcd = 19)) :=
by 
  let totalCards := remainingCards + newCards
  let eatenCards := totalCards - remainingCards
  use Nat.gcd eatenCards totalCards
  split 
  . rfl
  . split
    sorry
    sorry

end baseball_card_ratio_l62_62265


namespace find_fraction_l62_62705

theorem find_fraction (x y : ℕ) (h₀ : 0.46666666666666673 ≈ 7 / 15)
                      (h₁ : (2 / 5) / (3 / 7) = (7 / 15) / (x / y)) :
                      x = 1 → y = 2 → (x / y) = 1 / 2 :=
by sorry

end find_fraction_l62_62705


namespace ebay_ordered_cards_correct_l62_62814

noncomputable def initial_cards := 4
noncomputable def father_cards := 13
noncomputable def cards_given_to_dexter := 29
noncomputable def cards_kept := 20
noncomputable def bad_cards := 4

theorem ebay_ordered_cards_correct :
  let total_before_ebay := initial_cards + father_cards
  let total_after_giving_and_keeping := cards_given_to_dexter + cards_kept
  let ordered_before_bad := total_after_giving_and_keeping - total_before_ebay
  let ebay_ordered_cards := ordered_before_bad + bad_cards
  ebay_ordered_cards = 36 :=
by
  sorry

end ebay_ordered_cards_correct_l62_62814


namespace initial_amount_l62_62991

-- Define the conditions as Lean definitions
def total_money_left (x : ℝ) : Prop :=
  0.70 * x = 840

-- Prove the total amount of money "x" is $1200.
theorem initial_amount : ∃ x : ℝ, total_money_left x ∧ x = 1200 :=
by
  use 1200
  unfold total_money_left
  split
  · norm_num
  · rfl

end initial_amount_l62_62991


namespace area_of_triangle_ABC_l62_62040

theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 2) (h2 : c = 3) (h3 : C = 2 * B): 
  ∃ S : ℝ, S = 1/2 * b * c * (Real.sin A) ∧ S = 15 * (Real.sqrt 7) / 16 :=
by
  sorry

end area_of_triangle_ABC_l62_62040


namespace right_triangle_lengths_l62_62303

theorem right_triangle_lengths (a b c : ℝ) (h1 : c + b = 2 * a) (h2 : c^2 = a^2 + b^2) : 
  b = 3 / 4 * a ∧ c = 5 / 4 * a := 
by
  sorry

end right_triangle_lengths_l62_62303


namespace average_income_proof_l62_62471

def average_income_independent_of_bonus_distribution (A E : ℝ) : Prop :=
  ∀ (distribution_method : (fin 10 → ℝ) → Prop), 
    (∑ i, (distribution_method (λ _, (A + E) / 10)) = A + E) →
    (∀ i, distribution_method (λ _, (A + E) / 10) i = (A + E) / 10)

theorem average_income_proof (A E : ℝ) :
  average_income_independent_of_bonus_distribution A E := by
  sorry

end average_income_proof_l62_62471


namespace abs_x_plus_abs_y_eq_one_area_l62_62100

theorem abs_x_plus_abs_y_eq_one_area : 
  (∃ (A : ℝ), ∀ (x y : ℝ), |x| + |y| = 1 → A = 2) :=
sorry

end abs_x_plus_abs_y_eq_one_area_l62_62100


namespace number_of_valid_64_digit_numbers_div_by_101_not_even_l62_62802

-- Define what it means for a number to be a 64-digit number not containing zeros.
def is_valid_64_digit_number (N : ℕ) : Prop :=
  (10^63 ≤ N) ∧ (N < 10^64) ∧ (∀ i, i ∈ N.digits 10 → i ≠ 0)

-- Define the condition for a number to be divisible by 101.
def divisible_by_101 (N : ℕ) : Prop := N % 101 = 0

-- The theorem stating the problem and the conclusion.
theorem number_of_valid_64_digit_numbers_div_by_101_not_even :
  ¬ even (set_of (λ N : ℕ, is_valid_64_digit_number N ∧ divisible_by_101 N)).card :=
by
  sorry

end number_of_valid_64_digit_numbers_div_by_101_not_even_l62_62802


namespace exist_two_quadrilaterals_l62_62692

-- Define the structure of a quadrilateral with four sides and two diagonals
structure Quadrilateral :=
  (s1 : ℝ) -- side 1
  (s2 : ℝ) -- side 2
  (s3 : ℝ) -- side 3
  (s4 : ℝ) -- side 4
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2

-- The theorem stating the existence of two quadrilaterals satisfying the given conditions
theorem exist_two_quadrilaterals :
  ∃ (quad1 quad2 : Quadrilateral),
  quad1.s1 < quad2.s1 ∧ quad1.s2 < quad2.s2 ∧ quad1.s3 < quad2.s3 ∧ quad1.s4 < quad2.s4 ∧
  quad1.d1 > quad2.d1 ∧ quad1.d2 > quad2.d2 :=
by
  sorry

end exist_two_quadrilaterals_l62_62692


namespace lily_patch_cover_entire_lake_l62_62782

noncomputable def days_to_cover_half (initial_days : ℕ) := 33

theorem lily_patch_cover_entire_lake (initial_days : ℕ) (h : days_to_cover_half initial_days = 33) :
  initial_days + 1 = 34 :=
by
  sorry

end lily_patch_cover_entire_lake_l62_62782


namespace sibling_age_difference_l62_62771

theorem sibling_age_difference 
  (x : ℕ) 
  (h : 3 * x + 2 * x + 1 * x = 90) : 
  3 * x - x = 30 := 
by 
  sorry

end sibling_age_difference_l62_62771


namespace six_box_four_div_three_eight_box_two_div_four_l62_62084

def fills_middle_zero (d : Nat) : Prop :=
  d < 3

def fills_last_zero (d : Nat) : Prop :=
  (80 + d) % 4 = 0

theorem six_box_four_div_three {d : Nat} : fills_middle_zero d → ((600 + d * 10 + 4) / 3) % 100 / 10 = 0 :=
  sorry

theorem eight_box_two_div_four {d : Nat} : fills_last_zero d → ((800 + d * 10 + 2) / 4) % 10 = 0 :=
  sorry

end six_box_four_div_three_eight_box_two_div_four_l62_62084


namespace inequality_holds_for_a_in_1_5_l62_62515

theorem inequality_holds_for_a_in_1_5 (a x : ℝ) (hx : x ∈ Iio 1 ∨ x ∈ Ioi 5) (ha : 1 < a ∧ a ≤ 5) :
  x^2 - 2*(a-2)*x + a > 0 :=
by
  sorry 

end inequality_holds_for_a_in_1_5_l62_62515


namespace evaluate_product_l62_62697

open BigOperators

noncomputable def product_term (n : ℕ) : ℝ := 1 - 2 / n

theorem evaluate_product :
  ∏ (n : ℕ) in Finset.range 98 \ {0, 1, 2}, product_term (n + 3) = 49 / 150 :=
by
  rw [Finset.range_eq_Ico, Finset.Ico_filter_lt_eq_Ico]; simp
  sorry

end evaluate_product_l62_62697


namespace volunteer_arrangements_l62_62143

theorem volunteer_arrangements : 
  let volunteers := 7
  let participants := 6
  let per_day := 3
  (volunteers = 7) → 
  (participants = 6) → 
  (per_day = 3) → 
  ∃ assembles, ((choose 7 3) * (choose 4 3) = assembles) ∧ assembles = 140 :=
by
  intros h1 h2 h3
  sorry

end volunteer_arrangements_l62_62143


namespace find_circle_equation_l62_62358

theorem find_circle_equation (a b r : ℝ) :
  let C := {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2} in
  (a, b) ∈ {c : ℝ × ℝ | c.2 = -4 * c.1} →
  (3, -2) ∈ C ∧ dist (a, b) (3, -2) = r →
  (p ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 4)^2 = 8})

end find_circle_equation_l62_62358


namespace vector_ratio_l62_62722

variable {R : Type*} [LinearOrderedField R]
variable (a1 a2 a3 : R) (b1 b2 b3 : R)
variable (a b : Fin 3 → R)

noncomputable def len (v : Fin 3 → R) : R := Real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)
noncomputable def dot (x y : Fin 3 → R) : R := (x 0 * y 0 + x 1 * y 1 + x 2 * y 2)

theorem vector_ratio
  (ha : a = ![a1, a2, a3])
  (hb : b = ![b1, b2, b3])
  (norm_a : len a = 3)
  (norm_b : len b = 4)
  (dot_ab : dot a b = 12) :
  (a1 + a2 + a3) / (b1 + b2 + b3) = 3 / 4 :=
by
  sorry

end vector_ratio_l62_62722


namespace max_value_divisible_by_13_l62_62653

-- Define the digits and the number constraints.
def digits := {n : ℕ // n < 10}
def distinct (A B D : digits) : Prop := A ≠ B ∧ B ≠ D ∧ A ≠ D

-- Define the number in the specific format.
def number (A B D : digits) : ℕ :=
  10000 * A + 1000 * B + 100 * D + 10 * B + A

-- Define the divisibility condition by 13.
def divisible_by (n d : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Main theorem statement.
theorem max_value_divisible_by_13
  (A B D : digits)
  (h_distinct : distinct A B D)
  (h_divisibility : divisible_by (number A B D) 13) :
  number A B D ≤ 96769 := sorry

end max_value_divisible_by_13_l62_62653


namespace compare_neg_fractions_l62_62283

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l62_62283


namespace find_nk_find_d_and_constant_l62_62582

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d n : ℕ) := a + d * (n - 1)

-- Definition of the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a d n : ℕ) := n * a + (n * (n - 1) * d) / 2

-- Problem 1: Finding n_k
theorem find_nk
  (a1 a3 nk : ℕ)
  (d ≠ 0)
  (a1_eq : a1 = 2)
  (a3_eq : a3 = 6)
  (geo_seq: ∀ (k : ℕ), 2 * (3 ^ (k + 1))):
  nk = 3^(k + 1) :=
sorry

-- Problem 2: Common difference for specific ratio of sums
theorem find_d_and_constant
  (a1 a3 n : ℕ)
  (d ≠ 0)
  (a1_eq : a1 = 2)
  (a3_eq : a3 = 6)
  (lambda : ℕ)
  (sum3n : λ = 5)
  (sum_n : arithmetic_sum 2 4 n) :
d = 4 ∧ lambda = 5 :=
sorry

end find_nk_find_d_and_constant_l62_62582


namespace incorrect_sym_center_max_value_f_pi_six_f_monotonically_increasing_graph_transformation_l62_62393

noncomputable def f (x : Real) : Real := (Real.sqrt 3 * Real.sin x + Real.cos x) * Real.cos x

theorem incorrect_sym_center : ¬ (f (5 * Real.pi / 12) = 0 ∧ Real.sym_center (f) = (5 * Real.pi / 12, 0)) :=
  sorry

theorem max_value_f_pi_six : f (Real.pi / 6) = 3 / 2 :=
  sorry

theorem f_monotonically_increasing : ∀ x y : Real, -Real.pi / 3 ≤ x → x ≤ y → y ≤ Real.pi / 6 → f x ≤ f y :=
  sorry

theorem graph_transformation : ∀ x : Real, f x = Real.sin (2 * x + Real.pi / 6) + 1 / 2 :=
  sorry

end incorrect_sym_center_max_value_f_pi_six_f_monotonically_increasing_graph_transformation_l62_62393


namespace total_number_of_animals_l62_62925

theorem total_number_of_animals 
  (rabbits ducks chickens : ℕ)
  (h1 : chickens = 5 * ducks)
  (h2 : ducks = rabbits + 12)
  (h3 : rabbits = 4) : 
  chickens + ducks + rabbits = 100 :=
by
  sorry

end total_number_of_animals_l62_62925


namespace smallest_possible_z_l62_62487

theorem smallest_possible_z (w x y z : ℕ) (k : ℕ) (h1 : w = x - 1) (h2 : y = x + 1) (h3 : z = x + 2)
  (h4 : w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z) (h5 : k = 2) (h6 : w^3 + x^3 + y^3 = k * z^3) : z = 6 :=
by
  sorry

end smallest_possible_z_l62_62487


namespace games_left_is_correct_l62_62758

-- Define the initial number of DS games
def initial_games : ℕ := 98

-- Define the number of games given away
def games_given_away : ℕ := 7

-- Define the number of games left
def games_left : ℕ := initial_games - games_given_away

-- Theorem statement to prove that the number of games left is 91
theorem games_left_is_correct : games_left = 91 :=
by
  -- Currently, we use sorry to skip the actual proof part.
  sorry

end games_left_is_correct_l62_62758


namespace probability_A_shot_l62_62641

/-- Given that A and B take turns randomly spinning the cylinder and shooting, with A starting. 
    They are using a six-shot revolver with one bullet. 
    Each attempt has a probability of 1/6 of firing a shot and 5/6 of not firing a shot. --/
theorem probability_A_shot : 
  let p_shot := (1 : ℚ) / 6 
  let p_not_shot := (5 : ℚ) / 6 
  p_NOT_SHOT =
  -- The total probability that the shot happens when player A has the revolver is 6/11
  (1/6) / (1 - (25/36)) = 6/11 :=
begin
  sorry
end

end probability_A_shot_l62_62641


namespace func_has_one_zero_in_interval_l62_62461

theorem func_has_one_zero_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0) ∧
  (∀ y₁ y₂ : ℝ, 0 < y₁ ∧ y₁ < 1 → 0 < y₂ ∧ y₂ < 1 → 2 * a * y₁^2 - y₁ - 1 = 0 → 
   2 * a * y₂^2 - y₂ - 1 = 0 → y₁ = y₂) ↔ a ∈ set.Ioi (1 : ℝ) :=
by sorry

end func_has_one_zero_in_interval_l62_62461


namespace scarlet_savings_l62_62540

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l62_62540


namespace problem_statement_l62_62071

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable [Differentiable ℝ f]
variable [Differentiable ℝ g]
variable [IsDerivative f f']
variable [IsDerivative g g']

def condition_1 : Prop := ∀ x, f(x+3) = g(-x) + 2
def condition_2 : Prop := ∀ x, f'(x-1) = g'(x)
def condition_3 : Prop := ∀ x, g(-x+1) = -g(x+1)

theorem problem_statement (h1 : condition_1 f g) (h2 : condition_2 f' g') (h3 : condition_3 g) :
  (g(1) = 0) ∧
  (∀ x, g'(x+1) = -g'(3-x)) ∧
  (∀ x, g(x+1) = g(3-x)) ∧
  (∀ x, g(x+4) = g(x)) :=
by sorry

end problem_statement_l62_62071


namespace percent_of_x_eq_to_y_l62_62625

variable {x y : ℝ}

theorem percent_of_x_eq_to_y (h: 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x :=
by
  sorry

end percent_of_x_eq_to_y_l62_62625


namespace largest_multiple_of_7_less_than_100_l62_62605

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l62_62605


namespace number_of_avocados_l62_62232

-- Constants for the given problem
def banana_cost : ℕ := 1
def apple_cost : ℕ := 2
def strawberry_cost_per_12 : ℕ := 4
def avocado_cost : ℕ := 3
def grape_cost_half_bunch : ℕ := 2
def total_cost : ℤ := 28

-- Quantities of the given fruits
def banana_qty : ℕ := 4
def apple_qty : ℕ := 3
def strawberry_qty : ℕ := 24
def grape_qty_full_bunch_cost : ℕ := 4 -- since half bunch cost $2, full bunch cost $4

-- Definition to calculate the cost of the known fruits
def known_fruit_cost : ℤ :=
  (banana_qty * banana_cost) +
  (apple_qty * apple_cost) +
  (strawberry_qty / 12 * strawberry_cost_per_12) +
  grape_qty_full_bunch_cost

-- The cost of avocados needed to fill the total cost
def avocado_cost_needed : ℤ := total_cost - known_fruit_cost

-- Finally, we need to prove that the number of avocados is 2
theorem number_of_avocados (n : ℕ) : n * avocado_cost = avocado_cost_needed → n = 2 :=
by
  -- Problem data
  have h_banana : ℕ := banana_qty * banana_cost
  have h_apple : ℕ := apple_qty * apple_cost
  have h_strawberry : ℕ := (strawberry_qty / 12) * strawberry_cost_per_12
  have h_grape : ℕ := grape_qty_full_bunch_cost
  have h_known : ℕ := h_banana + h_apple + h_strawberry + h_grape
  
  -- Calculation for number of avocados
  have h_avocado : ℤ := total_cost - h_known
  
  -- Proving number of avocados
  sorry

end number_of_avocados_l62_62232


namespace roots_of_polynomial_inequality_l62_62838

theorem roots_of_polynomial_inequality :
  (∃ (p q r s : ℂ), (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧
  (p * q * r * s = 3) ∧ (p*q + p*r + p*s + q*r + q*s + r*s = 11)) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3) :=
by
  sorry

end roots_of_polynomial_inequality_l62_62838


namespace angle_in_second_quadrant_l62_62413

theorem angle_in_second_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : π - α ∈ Ioo (π / 2) π :=
sorry

end angle_in_second_quadrant_l62_62413


namespace regular_2n_gon_can_be_inscribed_l62_62248

noncomputable def regular_polygon_with_rotation (n : ℕ) (A : fin n → ℝ × ℝ) 
  (θ : ℝ) := 
  ∃ B : fin n → ℝ × ℝ, 
    ∀ i : fin n, 
      dist (A i) (A ((i + 1) % n)) = dist (B i) (B ((i + 1) % n)) 
      ∧ dist (A i) (B i) = dist (A ((i + 1) % n)) (B ((i + 1) % n)) 
      ∧ dist (A i) (A ((i + 1) % n)) = dist (A i) (B i) 
      ∧ ∀ j : fin n, j ≠ i → dist (A j) (B j) = dist (A ((j + 1) % n)) (B ((j + 1) % n))

theorem regular_2n_gon_can_be_inscribed (n : ℕ) (A : fin n → ℝ × ℝ) (θ : ℝ) 
  (h : θ = 360 / n ∧ θ < 120) : 
  regular_polygon_with_rotation n A θ :=
by
  sorry

end regular_2n_gon_can_be_inscribed_l62_62248


namespace max_area_of_rectangle_l62_62578

-- Question: Prove the largest possible area of a rectangle given the conditions
theorem max_area_of_rectangle :
  ∀ (x : ℝ), (2 * x + 2 * (x + 5) = 60) → x * (x + 5) ≤ 218.75 :=
by
  sorry

end max_area_of_rectangle_l62_62578


namespace integer_solutions_count_l62_62133

theorem integer_solutions_count : 
  (∃ (s : Finset ℤ), (∀ x ∈ s, (5:ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 7) ∧ s.card = 8) :=
by
  sorry

end integer_solutions_count_l62_62133


namespace painters_workdays_l62_62812

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l62_62812


namespace num_common_tangents_eq_two_l62_62595

noncomputable def circle1 : set (ℝ × ℝ) :=
  {p | (p.1 - 3) ^ 2 + (p.2 + 8) ^ 2 = 121}

noncomputable def circle2 : set (ℝ × ℝ) :=
  {p | (p.1 + 2) ^ 2 + (p.2 - 4) ^ 2 = 64}

theorem num_common_tangents_eq_two :
  let C1 := (3, -8)
  let C2 := (-2, 4)
  let r1 := 11
  let r2 := 8
  let d := Real.sqrt ((5:ℝ) ^ 2 + (-12) ^ 2)
  3 < d ∧ d < 19 → 2 := 
by
  sorry

end num_common_tangents_eq_two_l62_62595


namespace total_profit_correct_total_boxes_sold_correct_most_profitable_day_correct_l62_62528

variables (M : ℕ)

def boxes_sold_mon := M
def boxes_sold_tue := M + 10
def boxes_sold_wed := M + 20
def boxes_sold_thu := M + 30
def boxes_sold_fri := 30
def boxes_sold_sat := 60
def boxes_sold_sun := 45

def total_boxes_sold := boxes_sold_mon M + boxes_sold_tue M + boxes_sold_wed M + boxes_sold_thu M + boxes_sold_fri + boxes_sold_sat + boxes_sold_sun

def regular_boxes_sold := total_boxes_sold M / 2
def wholegrain_boxes_sold := total_boxes_sold M / 2

def profit_regular := regular_boxes_sold M * 3
def profit_wholegrain := wholegrain_boxes_sold M * 6

def total_profit := profit_regular M + profit_wholegrain M

def most_profitable_day := max (max (max (max (max (boxes_sold_mon M, boxes_sold_tue M), boxes_sold_wed M), boxes_sold_thu M), boxes_sold_fri), max (boxes_sold_sat, boxes_sold_sun))

theorem total_profit_correct : total_profit M = (4 * M + 205) * 4.5 := by
  sorry

theorem total_boxes_sold_correct : total_boxes_sold M = 4 * M + 205 := by
  sorry

theorem most_profitable_day_correct : most_profitable_day M = boxes_sold_sat := by
  sorry

end total_profit_correct_total_boxes_sold_correct_most_profitable_day_correct_l62_62528


namespace distinct_patterns_count_l62_62760

def grid_size : nat := 4

def num_shaded : nat := 3

def patterns_equiv_by_symmetry (pattern1 pattern2 : list (nat × nat)) : Prop :=
-- Definition that determines if two patterns are equivalent under rotations/flips

theorem distinct_patterns_count :
  ∃ (patterns : finset (list (nat × nat))), patterns.card = 12 ∧
  ∀ pattern ∈ patterns, pattern.length = num_shaded ∧
  ∀ alt_pattern, patterns_equiv_by_symmetry pattern alt_pattern → pattern = alt_pattern :=
sorry

end distinct_patterns_count_l62_62760


namespace valid_number_of_two_humped_camels_within_range_l62_62958

variable (N : ℕ)

def is_valid_number_of_two_humped_camels (N : ℕ) : Prop :=
  ∀ (S : ℕ) (hS : S = 62), 
    let total_humps := 100 + N in 
    S * 1 + (S - (S * 1)) * 2 ≥ total_humps / 2

theorem valid_number_of_two_humped_camels_within_range :
  ∃ (count : ℕ), count = 72 ∧ 
    ∀ (N : ℕ), (1 ≤ N ∧ N ≤ 99) → 
      is_valid_number_of_two_humped_camels N ↔ 
        (1 ≤ N ∧ N ≤ 24) ∨ (52 ≤ N ∧ N ≤ 99) :=
by
  sorry

end valid_number_of_two_humped_camels_within_range_l62_62958


namespace max_students_seated_l62_62668

-- Define the number of seats in the i-th row
def seats_in_row (i : ℕ) : ℕ := 10 + 2 * i

-- Define the maximum number of students that can be seated in the i-th row
def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

-- Sum the maximum number of students for all 25 rows
def total_max_students : ℕ := (Finset.range 25).sum max_students_in_row

-- The theorem statement
theorem max_students_seated : total_max_students = 450 := by
  sorry

end max_students_seated_l62_62668


namespace equilateral_triangle_side_length_l62_62763

theorem equilateral_triangle_side_length (side_length_of_square : ℕ) (h : side_length_of_square = 21) :
    let total_length_of_string := 4 * side_length_of_square
    let side_length_of_triangle := total_length_of_string / 3
    side_length_of_triangle = 28 :=
by
  sorry

end equilateral_triangle_side_length_l62_62763


namespace mass_of_added_water_with_temp_conditions_l62_62219

theorem mass_of_added_water_with_temp_conditions
  (m_l : ℝ) (t_pi t_B t : ℝ) (c_B c_l lambda : ℝ) :
  m_l = 0.05 →
  t_pi = -10 →
  t_B = 10 →
  t = 0 →
  c_B = 4200 →
  c_l = 2100 →
  lambda = 3.3 * 10^5 →
  (0.0028 ≤ (2.1 * m_l * 10 + lambda * m_l) / (42 * 10) 
  ∧ (2.1 * m_l * 10) / (42 * 10) ≤ 0.418) :=
by
  sorry

end mass_of_added_water_with_temp_conditions_l62_62219


namespace tangent_line_at_P_two_extreme_points_gt_e_square_l62_62748

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x - x - 1/2 * a * x^2

theorem tangent_line_at_P (x : ℝ) (a : ℝ) (hx : a = -2) : 
  let f := fun x => x * Real.log x - x + x^2 in
  let P := (1 : ℝ, 0 : ℝ) in
  let tangent_line := fun x => 2 * (x - 1) in 
  ∃ y : ℝ, tangent_line x + y = 0 :=
sorry

theorem two_extreme_points_gt_e_square (a : ℝ)
  (hx1x2 : ∃ x1 x2 : ℝ, x1 < x2 ∧ ∂ (f x₁ a) / ∂x = 0 ∧ ∂ (f x₂ a) / ∂x = 0) : 
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1 * x2 > Real.exp 2) :=
sorry

end tangent_line_at_P_two_extreme_points_gt_e_square_l62_62748


namespace evaluate_expression_l62_62308

variable (b : ℝ) -- assuming b is a real number, (if b should be of different type, modify accordingly)

theorem evaluate_expression (y : ℝ) (h : y = b + 9) : y - b + 5 = 14 :=
by
  sorry

end evaluate_expression_l62_62308


namespace find_a_if_f_even_l62_62446

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62446


namespace even_function_a_zero_l62_62426

noncomputable def f (x a : ℝ) : ℝ := (x + a) * real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) →
  (2 * x - 1) / (2 * x + 1) > 0 → 
  x > 1 / 2 ∨ x < -1 / 2 →
  a = 0 :=
by {
  sorry
}

end even_function_a_zero_l62_62426


namespace simplify_and_evaluate_l62_62089

-- Define the expression
def expression (x : ℝ) := -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2)

-- State the theorem
theorem simplify_and_evaluate : expression (-2) = -10 :=
by
  -- The proof goes here
  sorry

end simplify_and_evaluate_l62_62089


namespace a_power_l62_62556

theorem a_power (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end a_power_l62_62556


namespace ramu_profit_percent_l62_62869

def ramu_bought_car : ℝ := 48000
def ramu_repair_cost : ℝ := 14000
def ramu_selling_price : ℝ := 72900

theorem ramu_profit_percent :
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 17.58 := 
by
  -- Definitions and setting up the proof environment
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  sorry

end ramu_profit_percent_l62_62869


namespace sum_of_squares_invariant_l62_62405

-- Definitions and conditions
variables {O : Type} [metric_space O]
variables (R1 R2 : ℝ) (A B C : O)

-- Definitions of the circles being concentric with center O
-- Defining points A and B as endpoints of a diameter of a circle of radius R1
-- and C as a point on a circle of radius R2.

def on_circle (center : O) (radius : ℝ) (P : O) : Prop := dist center P = radius

-- Statement
theorem sum_of_squares_invariant (hA : on_circle O R1 A) (hB : on_circle O R1 B)
    (hAB_diameter : dist A B = 2 * R1) (hC : on_circle O R2 C) :
    dist A C ^ 2 + dist B C ^ 2 = 2 * R1 ^ 2 + 2 * R2 ^ 2 :=
by
  sorry

end sum_of_squares_invariant_l62_62405


namespace problem_range_of_a_l62_62333

theorem problem_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |3 + x| ≥ a^2 - 4 * a) ↔ -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end problem_range_of_a_l62_62333


namespace point_C_moves_constant_speed_l62_62025

-- Define structures for the problem
structure Circle (O : Type) :=
(center : O)
(radius : ℝ)

structure EquilateralTriangle (P : Type) :=
(A B C : P)
(equilateral : dist A B = dist B C ∧ dist B C = dist A C ∧ ∀ θ : ℝ, ∠ A B C = θ → θ = 60)

-- Define the conditions accurately
variables {P : Type} [MetricSpace P]
variable (O1 O2 O3 : P)
variable (C : Circle P)
variable (T : EquilateralTriangle P)

-- Conditions
variables
  (A B A' B' : P)
  (HA : A ∈ C.center)
  (HB : B ∈ C.center)
  (angular_velocity_A : ℝ)
  (angular_velocity_B : ℝ)
  (same_angular_velocity : angular_velocity_A = angular_velocity_B)

-- The proof statement
theorem point_C_moves_constant_speed (hT : T.A = A ∧ T.B = B ∧ T.C = C) :
  ∃ (C_circle : Circle P), (C ∈ C_circle.center) ∧
    (∀ t : ℝ, dist C (rotate C_circle t) = radius C_circle) := sorry

end point_C_moves_constant_speed_l62_62025


namespace equal_angles_of_tangents_l62_62558

open EuclideanGeometry

theorem equal_angles_of_tangents
    {A B C O M : Point}
    [Circle O]
    (hAB : tangent A B O)
    (hAC : tangent A C O)
    (hAMO : ∠A O M = 90°) :
    ∠O B M = ∠O C M :=
sorry

end equal_angles_of_tangents_l62_62558


namespace day_of_week_after_45_days_l62_62678

theorem day_of_week_after_45_days (day_of_week : ℕ → String) (birthday_is_tuesday : day_of_week 0 = "Tuesday") : day_of_week 45 = "Friday" :=
by
  sorry

end day_of_week_after_45_days_l62_62678


namespace angles_on_y_axis_l62_62906

theorem angles_on_y_axis :
  {θ : ℝ | ∃ k : ℤ, (θ = 2 * k * Real.pi + Real.pi / 2) ∨ (θ = 2 * k * Real.pi + 3 * Real.pi / 2)} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by 
  sorry

end angles_on_y_axis_l62_62906


namespace find_constants_and_extreme_values_l62_62352

def f (x a b : ℝ) := x^3 + 3 * a * x^2 + b * x + a^2

theorem find_constants_and_extreme_values :
  let a := 2
  let b := 9 in
  (∀ (x : ℝ), a > 1 → (3 * x^2 + 6 * a * x + b = 0 → x = -1 → f x a b = 0)) ∧ 
  ((∃ x ∈ set.Icc (-4 : ℝ) (0 : ℝ), f x a b = 0) ∧ (∃ x ∈ set.Icc (-4 : ℝ) (0 : ℝ), f x a b = 4)) :=
by
  sorry

end find_constants_and_extreme_values_l62_62352


namespace area_of_region_eq_8_pi_l62_62270

theorem area_of_region_eq_8_pi :
  let eq := λ x y : ℝ, x^2 + y^2 + 8*x - 6*y + 17 = 0 in
  (∃ c : ℝ, ∀ x y : ℝ, (eq x y) ↔ (x + 4)^2 + (y - 3)^2 = c) ∧
  (∀ x y : ℝ, (x + 4)^2 + (y - 3)^2 = 8) →
  c = 8 →
  π * 8 = 8 * π :=
by
  sorry

end area_of_region_eq_8_pi_l62_62270


namespace possible_N_values_l62_62977

noncomputable def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (subset : Finset ℕ), subset.card = 62 → 
  ∑ x in subset, if x < N then 1 else 2 ≥ (100 + N) / 2)

theorem possible_N_values : Finset.card ((Finset.range 100).filter is_valid_N) = 72 := 
by 
  sorry

end possible_N_values_l62_62977


namespace tree_height_by_time_boy_is_36_inches_l62_62240

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l62_62240


namespace relationship_between_a_and_b_l62_62379

open Real

theorem relationship_between_a_and_b
  (a b x : ℝ)
  (h1 : a ≠ 1)
  (h2 : b ≠ 1)
  (h3 : 4 * (log x / log a)^3 + 5 * (log x / log b)^3 = 7 * (log x)^3) :
  b = a ^ (3 / 5)^(1 / 3) := 
sorry

end relationship_between_a_and_b_l62_62379


namespace arrangement_splits_subsets_l62_62502

noncomputable def splits (arr : List ℕ) (S : Set ℕ) : Prop :=
  ∃ (i j : ℕ) (x : ℕ), i < j ∧ i < arr.length ∧ j < arr.length ∧
  arr.nth i = some x ∧ arr.nth j = some x ∧ x ∉ S ∧
  ∃ y z, y ∈ S ∧ z ∈ S ∧ y ≠ z ∧
  List.indexOf arr y < List.indexOf arr x ∧ List.indexOf arr x < List.indexOf arr z

theorem arrangement_splits_subsets (n : ℕ) (hn : n ≥ 3) (subsets : Fin n.succ → Set ℕ)
  (h : ∀ i, (2 ≤ (subsets i).card ∧ (subsets i).card ≤ n - 1)) :
  ∃ (arr : List ℕ), ∀ i, splits arr (subsets i) :=
sorry

end arrangement_splits_subsets_l62_62502


namespace sqrt_inequality_l62_62357

theorem sqrt_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (habc : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := 
sorry

end sqrt_inequality_l62_62357


namespace possible_N_values_l62_62953

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62953


namespace fruit_baskets_l62_62079

theorem fruit_baskets {a b c : ℕ} 
  (h1 : a = 18) 
  (h2 : b = 27) 
  (h3 : c = 12) : gcd (gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  apply gcd_assoc
  compute_gcd
  apply gcd_comm
  compute_gcd
  apply gcd_rec
  compute_gcd
  exact 3
  sorry

end fruit_baskets_l62_62079


namespace deepak_present_age_l62_62210

theorem deepak_present_age 
  (x : ℕ) 
  (rahul_age_now : ℕ := 5 * x)
  (deepak_age_now : ℕ := 2 * x)
  (rahul_age_future : ℕ := rahul_age_now + 6) 
  (future_condition : rahul_age_future = 26) 
  : deepak_age_now = 8 :=
by
  sorry

end deepak_present_age_l62_62210


namespace average_income_independent_of_distribution_l62_62468

namespace AverageIncomeProof

variables (A E : ℝ) (total_employees : ℕ) (h : total_employees = 10)

/-- The average income in December does not depend on the distribution method -/
theorem average_income_independent_of_distribution : 
  (∃ (income : ℝ), income = (A + E) / total_employees) :=
by
  have h1 : total_employees = 10 := h
  exists (A + E) / total_employees
  sorry

end AverageIncomeProof

end average_income_independent_of_distribution_l62_62468


namespace domain_of_f_l62_62601

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ (5 + Real.sqrt 13) / 2 ∧ x ≠ (5 - Real.sqrt 13) / 2 → ∃ y : ℝ, y = f x :=
by
  sorry

end domain_of_f_l62_62601


namespace compare_neg_fractions_l62_62284

theorem compare_neg_fractions : - (1 : ℝ) / 3 < - (1 : ℝ) / 4 :=
  sorry

end compare_neg_fractions_l62_62284


namespace locus_of_P_l62_62367

theorem locus_of_P (A B C P L M N : Point) (h_triangle : isosceles_triangle A B C) 
  (h_P_in_triangle : inside_triangle P A B C) 
  (h_dist_eq_geom_mean : dist P (line BC) = real.sqrt (dist P (line AB) * dist P (line AC))) : 
  ∃ (Γ : Circle), is_arc_of_circle P B C Γ :=
sorry

end locus_of_P_l62_62367


namespace angle_DAB_l62_62775

noncomputable def triangle_data := Type

variables (triangle_AB := equilateral (60 : ℝ)) (triangle_BC := equilateral (60 : ℝ)) (x : ℝ)

def θ : ℝ := 180 - 60 - 60

theorem angle_DAB (ABC_equilateral : triangle_data) (BCD_equilateral : triangle_data) : θ = 60 :=
by 
  unfold θ
  ring
  exact eq.refl θ

end angle_DAB_l62_62775


namespace min_abs_sum_of_x1_x2_l62_62452

open Real

theorem min_abs_sum_of_x1_x2 (x1 x2 : ℝ) (h : 1 / ((2 + sin x1) * (2 + sin (2 * x2))) = 1) : 
  abs (x1 + x2) = π / 4 :=
sorry

end min_abs_sum_of_x1_x2_l62_62452


namespace number_of_readers_who_read_both_l62_62479

theorem number_of_readers_who_read_both (S L B total : ℕ) (hS : S = 250) (hL : L = 550) (htotal : total = 650) (h : S + L - B = total) : B = 150 :=
by {
  /-
  Given:
  S = 250 (number of readers who read science fiction)
  L = 550 (number of readers who read literary works)
  total = 650 (total number of readers)
  h : S + L - B = total (relationship between sets)
  We need to prove: B = 150
  -/
  sorry
}

end number_of_readers_who_read_both_l62_62479


namespace time_between_shark_sightings_l62_62861

def earnings_per_photo : ℕ := 15
def fuel_cost_per_hour : ℕ := 50
def hunting_hours : ℕ := 5
def expected_profit : ℕ := 200

theorem time_between_shark_sightings :
  (hunting_hours * 60) / ((expected_profit + (fuel_cost_per_hour * hunting_hours)) / earnings_per_photo) = 10 :=
by 
  sorry

end time_between_shark_sightings_l62_62861


namespace find_a_if_f_even_l62_62448

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l62_62448


namespace four_painters_workdays_l62_62804

theorem four_painters_workdays :
  (∃ (c : ℝ), ∀ (n : ℝ) (d : ℝ), n * d = c) →
  (p5 : ℝ) (d5 : ℝ) (p5 * d5 = 7.5) →
  ∀ D : ℝ, 4 * D = 7.5 →
  D = (1 + 7/8) := 
by {
  sorry
}

end four_painters_workdays_l62_62804


namespace total_miles_ran_l62_62853

theorem total_miles_ran (miles_monday miles_wednesday miles_friday : ℕ)
  (h1 : miles_monday = 3)
  (h2 : miles_wednesday = 2)
  (h3 : miles_friday = 7) :
  miles_monday + miles_wednesday + miles_friday = 12 := 
by
  sorry

end total_miles_ran_l62_62853


namespace paperclips_in_64_volume_box_l62_62250

def volume_16 : ℝ := 16
def volume_32 : ℝ := 32
def volume_64 : ℝ := 64
def paperclips_50 : ℝ := 50
def paperclips_100 : ℝ := 100

theorem paperclips_in_64_volume_box :
  ∃ (k p : ℝ), 
  (paperclips_50 = k * volume_16^p) ∧ 
  (paperclips_100 = k * volume_32^p) ∧ 
  (200 = k * volume_64^p) :=
by
  sorry

end paperclips_in_64_volume_box_l62_62250


namespace max_candies_eaten_in_29_minutes_l62_62920

theorem max_candies_eaten_in_29_minutes :
  let n := 29,
      sum_edges_complete_graph := (n * (n - 1)) / 2
  in sum_edges_complete_graph = 406 :=
by
  let n := 29
  let sum_edges_complete_graph := (n * (n - 1)) / 2
  have h : sum_edges_complete_graph = 406,
  { sorry },
  exact h

end max_candies_eaten_in_29_minutes_l62_62920


namespace factorization_of_difference_of_squares_l62_62312

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l62_62312


namespace remainder_of_q2_l62_62765

noncomputable def q1 (x : ℝ) := Polynomial.divByMonomial (Polynomial.X ^ 9) (Polynomial.C (1 / 3) * Polynomial.X - Polynomial.C 1)
noncomputable def r1 (x : ℝ) := Polynomial.eval (1 / 3) (Polynomial.C (1 / 3) * Polynomial.X - Polynomial.C 1)

noncomputable def q2 (x : ℝ) := Polynomial.divByMonomial (q1 x) (Polynomial.C (1 / 3) * Polynomial.X - Polynomial.C 1)
noncomputable def r2 (x : ℝ) := Polynomial.eval (1 / 3) (q1 x)

theorem remainder_of_q2 :
  r2 (1 / 3) = 1 / 6561 := by
  sorry

end remainder_of_q2_l62_62765


namespace ratio_of_profits_is_2_to_3_l62_62863

-- Conditions
def Praveen_initial_investment := 3220
def Praveen_investment_duration := 12
def Hari_initial_investment := 8280
def Hari_investment_duration := 7

-- Effective capital contributions
def Praveen_effective_capital : ℕ := Praveen_initial_investment * Praveen_investment_duration
def Hari_effective_capital : ℕ := Hari_initial_investment * Hari_investment_duration

-- Theorem statement to be proven
theorem ratio_of_profits_is_2_to_3 : (Praveen_effective_capital : ℚ) / Hari_effective_capital = 2 / 3 :=
by sorry

end ratio_of_profits_is_2_to_3_l62_62863


namespace intersecting_lines_and_angle_condition_l62_62704

theorem intersecting_lines_and_angle_condition 
  (l1 : ∀ x y : ℝ, x + y + 1 = 0) 
  (l2 : ∀ x y : ℝ, 5x - y - 1 = 0) 
  (angle_condition : ∀ x y : ℝ, 3x + 2y + 1 = 0) :
  ∃ line : ℝ → ℝ → Prop, 
    (line = (λ x y, x + 5y + 5 = 0) ∨ line = (λ x y, 5x - y - 1 = 0)) ∧
    (∀ x y, l1 x y ∧ l2 x y → line x y) ∧
    (∃ θ : ℝ, θ = π / 4 ∧ θ = atan ((5 / 1) - (3 / 2)) / (1 + (5 / 1) * (3 / 2))) :=
sorry

end intersecting_lines_and_angle_condition_l62_62704


namespace range_of_m_l62_62750

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log x + m / x

theorem range_of_m (m : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (f m b - f m a) / (b - a) < 1) →
  m ≥ 1 / 4 :=
by
  sorry

end range_of_m_l62_62750


namespace find_a_values_l62_62738

noncomputable def f (x a : ℝ) : ℝ := x / (sqrt (a - x ^ 2) - sqrt (1 - x ^ 2))

theorem find_a_values (a : ℝ) (h : ∃ x, f x a = -2/3) : a = 4 ∨ a = 1/4 := 
sorry

end find_a_values_l62_62738


namespace cube_root_of_27_l62_62105

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { rfl }

end cube_root_of_27_l62_62105


namespace spent_on_veggies_l62_62594

noncomputable def total_amount : ℕ := 167
noncomputable def spent_on_meat : ℕ := 17
noncomputable def spent_on_chicken : ℕ := 22
noncomputable def spent_on_eggs : ℕ := 5
noncomputable def spent_on_dog_food : ℕ := 45
noncomputable def amount_left : ℕ := 35

theorem spent_on_veggies : 
  total_amount - (spent_on_meat + spent_on_chicken + spent_on_eggs + spent_on_dog_food + amount_left) = 43 := 
by 
  sorry

end spent_on_veggies_l62_62594


namespace ellipse_equation_given_conditions_OM_ON_constant_l62_62026

theorem ellipse_equation_given_conditions
  (b : ℝ) (hb : 0 < b ∧ b < 2)
  (t1 : ℝ) (ht1 : t1 = 0)
  (t2 : ℝ) (ht2 : t2 = 4 * real.sqrt 2 / 3)
  (PQ AP : ℝ)
  (hPQ : 2 * AP = PQ)
  (A P Q : ℝ × ℝ)
  (hA : A = (2, 0))
  (hC : ∀ (x y : ℝ), x^2 / 4 + y^2 / b^2 = 1 → (x, y) = P ∨ (x, y) = Q)
  (hP : P = (1, 1)) :
  (∀ (x y : ℝ), x^2 / 4 + y^2 / (4 / 3) = 1) :=
sorry

theorem OM_ON_constant
  (b : ℝ) (hb : 0 < b ∧ b < 2)
  (A P Q M N : ℝ × ℝ)
  (hA : A = (2, 0))
  (hP : P = (1, 1))
  (hQ : Q = (-1, -1))
  (hM : M = (0, -2 / (1 - 2)))
  (hN : N = (0, -2 / (1 + 2)))
  (hEllipse : ∀ (x y : ℝ), x^2 / 4 + y^2 / (4 / 3) = 1) :
  (|M.1 - 0| * |N.1 - 0| = b^2) :=
sorry

end ellipse_equation_given_conditions_OM_ON_constant_l62_62026


namespace monkey_tree_height_l62_62652

theorem monkey_tree_height (hours: ℕ) (hop ft_per_hour : ℕ) (slip ft_per_hour : ℕ) (net_progress : ℕ) (final_hour : ℕ) (total_height : ℕ) :
  (hours = 18) ∧
  (hop = 3) ∧
  (slip = 2) ∧
  (net_progress = hop - slip) ∧
  (net_progress = 1) ∧
  (final_hour = 1) ∧
  (total_height = (hours - 1) * net_progress + hop) ∧
  (total_height = 20) :=
by
  sorry

end monkey_tree_height_l62_62652


namespace simplify_expression_l62_62546

variable (a : ℚ)
def expression := ((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4 * a + 4) / (a^2 - a))

theorem simplify_expression (h : a = 3) : expression a = 3 / 5 :=
by
  rw [h]
  -- additional simplifications would typically go here if the steps were spelled out
  sorry

end simplify_expression_l62_62546


namespace cat_finishes_food_on_next_wednesday_l62_62873

def cat_food_consumption_per_day : ℚ :=
  (1 / 4) + (1 / 6)

def total_food_on_day (n : ℕ) : ℚ :=
  n * cat_food_consumption_per_day

def total_cans : ℚ := 8

theorem cat_finishes_food_on_next_wednesday :
  total_food_on_day 10 = total_cans := sorry

end cat_finishes_food_on_next_wednesday_l62_62873


namespace sum_of_squares_mod_13_l62_62789

theorem sum_of_squares_mod_13 (n : ℕ) (h : n = 1000) :
  (∑ x in Finset.range n.succ, ∑ y in Finset.range n.succ, ∑ z in Finset.range n.succ, (x^2 + y^2 + z^2)) % 13 = 0 :=
by {
  -- This is the statement only, skipping the actual proof
  sorry
}

end sum_of_squares_mod_13_l62_62789


namespace hyperbola_sum_l62_62290

-- Define the center, vertex, and focus points
def center := (3, -2)
def vertex := (3, 0)
def focus := (3, 5)

-- Define the distances a and c
def a := abs (-2 - vertex.snd)
def c := abs (-2 - focus.snd)

-- Given the relationship c^2 = a^2 + b^2, we find b
noncomputable def b := real.sqrt (c^2 - a^2)

-- Define coordinates of the center
def h := center.fst
def k := center.snd

-- Define the value to be proven
def sum := h + k + a + b

-- The theorem to prove
theorem hyperbola_sum : sum = 3 + 3 * real.sqrt 5 := by
  sorry

end hyperbola_sum_l62_62290


namespace minimum_sum_S2019_abs_sum_seq_l62_62659

def is_absolute_sum_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, |a(n+1)| + |a(n)| = d

theorem minimum_sum_S2019_abs_sum_seq :
  ∃ a : ℕ → ℤ, is_absolute_sum_sequence a 3 ∧ a 1 = 2 ∧ (∑ i in finset.range 2019, a i) = -3025 :=
by
  sorry

end minimum_sum_S2019_abs_sum_seq_l62_62659


namespace exists_a_for_system_solution_l62_62320

theorem exists_a_for_system_solution (b : ℝ) :
  (∃ a x y : ℝ, y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*sqrt 2 + 1/4 :=
begin
  sorry
end

end exists_a_for_system_solution_l62_62320


namespace sum_of_positive_integer_values_of_b_with_rational_roots_l62_62328

theorem sum_of_positive_integer_values_of_b_with_rational_roots : 
  (∑ b in {b : ℕ | ∃ k : ℕ, (36 - k^2) / 12 = b ∧ 36 - k^2 ≥ 0 ∧ (36 - k^2) % 12 = 0 ∧ b > 0}, b) = 3 := 
sorry

end sum_of_positive_integer_values_of_b_with_rational_roots_l62_62328


namespace john_total_skateboarded_miles_l62_62499

-- Definitions
def distance_skateboard_to_park := 16
def distance_walk := 8
def distance_bike := 6
def distance_skateboard_home := distance_skateboard_to_park

-- Statement to prove
theorem john_total_skateboarded_miles : 
  distance_skateboard_to_park + distance_skateboard_home = 32 := 
by
  sorry

end john_total_skateboarded_miles_l62_62499


namespace cloth_woven_equal_l62_62643

theorem cloth_woven_equal :
  let rate := 0.129
  let time := 116.27906976744185
  15 ≈ rate * time :=
by
  sorry

end cloth_woven_equal_l62_62643


namespace average_sacks_per_day_l62_62568

theorem average_sacks_per_day {
  (total_oranges : ℕ)
  (total_apples : ℕ)
  (total_days : ℕ)
  (harvest_oranges_every : ℕ)
  (harvest_apples_every : ℕ)
  (LCM: ℕ)
  (harvest_days: ℕ)
} :
  total_oranges = 56 →
  total_apples = 35 →
  total_days = 20 →
  harvest_oranges_every = 2 →
  harvest_apples_every = 3 →
  LCM = Nat.lcm harvest_oranges_every harvest_apples_every →
  harvest_days = total_days / LCM →
  (total_oranges / (total_days / harvest_oranges_every).toFloat + total_apples / (total_days / harvest_apples_every).toFloat) = 11.4333 :=
by
  sorry

end average_sacks_per_day_l62_62568


namespace candidate_X_win_percentage_l62_62209

theorem candidate_X_win_percentage
    (R : ℝ) 
    (h1 : ∀r, r ∈ {3R, 2R})
    (h2 : 0.75 * (3 * R) + 0.15 * (2 * R) = 2.55 * R)
    (h3 : (1 - 0.75) * (3 * R) + (1 - 0.15) * (2 * R) = 2.45 * R)
    (h4 : 2.55 * R - 2.45 * R = 0.1 * R)
    (h5 : 2.55 * R + 2.45 * R = 5 * R) :
    (0.1 * R / (5 * R)) * 100 = 2 :=
by sorry

end candidate_X_win_percentage_l62_62209


namespace cosine_of_supplementary_angle_l62_62028

theorem cosine_of_supplementary_angle (angle_TPQ angle_TPS : ℝ) 
  (h1 : angle_TPQ + angle_TPS = 180)
  (h2 : real.cos angle_TPQ = 4/5) :
  real.cos angle_TPS = - (4/5) :=
by
  sorry

end cosine_of_supplementary_angle_l62_62028


namespace min_deg_g_proof_l62_62992

noncomputable def min_deg_g (f g h : Polynomial ℝ) : ℕ :=
  if 5 * f + 2 * g = h ∧ f.degree = 10 ∧ h.degree = 12 ∧ h.coeff (nat_degree h) = 2 then 
    12
  else 
    sorry

theorem min_deg_g_proof (f g h : Polynomial ℝ) 
  (H : 5 * f + 2 * g = h)
  (Hf : f.degree = 10)
  (Hh : h.degree = 12)
  (Hh_coef : h.coeff (nat_degree h) = 2) :
  min_deg_g f g h = 12 :=
by sorry

end min_deg_g_proof_l62_62992


namespace max_papers_l62_62268

theorem max_papers (p c r : ℕ) (h1 : p ≥ 2) (h2 : c ≥ 1) (h3 : 3 * p + 5 * c + 9 * r = 72) : r ≤ 6 :=
sorry

end max_papers_l62_62268


namespace tire_circumference_is_correct_l62_62626

-- Definitions of the given conditions
def car_speed_kmh : ℝ := 72
def tire_rotations_per_minute : ℝ := 400

-- Circumference of the tire to be proved
def circumference_of_tire : ℝ := 3

theorem tire_circumference_is_correct :
  let car_speed_mpm := car_speed_kmh * 1000 / 60 in
  let distance_covered_by_tire_per_minute := tire_rotations_per_minute * circumference_of_tire in
  car_speed_mpm = distance_covered_by_tire_per_minute → circumference_of_tire = 3 := 
by 
  intros car_speed_mpm_eq_tire_distance
  unfold car_speed_mpm at car_speed_mpm_eq_tire_distance
  unfold distance_covered_by_tire_per_minute at car_speed_mpm_eq_tire_distance
  exact car_speed_mpm_eq_tire_distance
  sorry

end tire_circumference_is_correct_l62_62626


namespace trapezoid_perimeter_l62_62797

open Real EuclideanGeometry

/-- In the trapezoid ABCD, the bases AD and BC are 8 and 18 respectively. It is known that
the circumcircle of triangle ABD is tangent to the lines BC and CD. Prove that the perimeter
of the trapezoid is 56. -/
theorem trapezoid_perimeter (A B C D : Point)
  (h1 : dist A D = 8) (h2 : dist B C = 18)
  (h_circum : ∃ O r, Circle O r ∧ Tangent O r A B D B C ∧ Tangent O r A B D C D) :
  dist A B + dist A D + dist B C + dist C D = 56 := 
sorry

end trapezoid_perimeter_l62_62797


namespace circles_externally_tangent_l62_62898

-- Definitions of the circles
noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 + 16 = 0 }

-- Centers and radii extracted
def center_circle1 := (0 : ℝ, 0 : ℝ)
def radius_circle1 := 2

def center_circle2 := (5 : ℝ, 0 : ℝ)
def radius_circle2 := 3

-- Distance between centers
def distance_centers := real.sqrt ((center_circle2.1 - center_circle1.1)^2 + (center_circle2.2 - center_circle1.2)^2)

-- Main theorem statement
theorem circles_externally_tangent :
  distance_centers = radius_circle1 + radius_circle2 := by
  sorry

end circles_externally_tangent_l62_62898


namespace sum_of_double_factorials_l62_62293

noncomputable def double_factorial (n : ℕ) : ℕ :=
if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

theorem sum_of_double_factorials :
  let S := ∑ i in (Finset.range 1005).map (λ i, i + 1), (double_factorial (2 * i - 1) : ℚ) / double_factorial (2 * i)
  let p := ∑ i in (Finset.range (Nat.floor (Math.sqrt 1005 : ℝ) : ℕ)), (1005 / 2^i) ∧
  let a := 2 * 1005 - (p - 4) ∧
  let b := 1 ∧
  ab_div_10 := (a * b) / 10
  in S = ab_div_10 :=
by sorry

end sum_of_double_factorials_l62_62293


namespace neg_univ_prop_l62_62400

theorem neg_univ_prop (p : Prop) : 
  (∀ x : ℝ, x^2 ≥ 0) → ¬ (∀ x : ℝ, x^2 ≥ 0) = ∃ x : ℝ, x^2 < 0 :=
by
  intro h
  funext
  apply propext
  constructor;
  intro h1;
  cases h1;
  sorry

end neg_univ_prop_l62_62400


namespace min_value_of_squares_l62_62769

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end min_value_of_squares_l62_62769


namespace diff_operator_additivity_l62_62845

variable {α : Type} [Add α]

-- Define the sequence terms as functions from natural numbers to type α
def w (n : ℕ) : α := -- particular function representing w sequence
def u (n : ℕ) : α := -- particular function representing u sequence
def v (n : ℕ) : α := -- particular function representing v sequence

-- Define the difference operator
def Δ (f : ℕ → α) (n : ℕ) : α := f (n + 1) - f n

-- State the problem conditions and the theorem
theorem diff_operator_additivity (h : ∀ n, w n = u n + v n) : ∀ n, Δ w n = Δ u n + Δ v n := by
  sorry

end diff_operator_additivity_l62_62845


namespace circle_area_and_circumference_changes_l62_62271

noncomputable section

structure Circle :=
  (r : ℝ)

def area (c : Circle) : ℝ := Real.pi * c.r^2

def circumference (c : Circle) : ℝ := 2 * Real.pi * c.r

def percentage_change (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem circle_area_and_circumference_changes
  (r1 r2 : ℝ) (c1 : Circle := {r := r1}) (c2 : Circle := {r := r2})
  (h1 : r1 = 5) (h2 : r2 = 4) :
  let original_area := area c1
  let new_area := area c2
  let original_circumference := circumference c1
  let new_circumference := circumference c2
  percentage_change original_area new_area = 36 ∧
  new_circumference = 8 * Real.pi ∧
  percentage_change original_circumference new_circumference = 20 :=
by
  sorry

end circle_area_and_circumference_changes_l62_62271


namespace kids_in_group_l62_62671

theorem kids_in_group :
  ∃ (K : ℕ), (∃ (A : ℕ), A + K = 9 ∧ 2 * A = 14) ∧ K = 2 :=
by
  sorry

end kids_in_group_l62_62671


namespace total_people_l62_62544

theorem total_people (P G NG : ℕ) (hP : P = 1) (hG : G = 6) (hNG : NG = 7) : P + G + NG = 14 := 
by {
  rw [hP, hG, hNG],
  norm_num,
}

end total_people_l62_62544


namespace total_apples_after_transactions_l62_62903

def initial_apples : ℕ := 65
def percentage_used : ℕ := 20
def apples_bought : ℕ := 15

theorem total_apples_after_transactions :
  (initial_apples * (1 - percentage_used / 100)) + apples_bought = 67 := 
by
  sorry

end total_apples_after_transactions_l62_62903


namespace division_remainder_example_l62_62242

theorem division_remainder_example :
  ∃ n, n = 20 * 10 + 10 ∧ n = 210 :=
by
  sorry

end division_remainder_example_l62_62242


namespace sum_base_6_l62_62272

-- Define base 6 numbers
def n1 : ℕ := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
def n2 : ℕ := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0

-- Define the expected result in base 6
def expected_sum : ℕ := 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

-- The theorem to prove
theorem sum_base_6 : n1 + n2 = expected_sum := by
    sorry

end sum_base_6_l62_62272


namespace expression_right_side_l62_62338

variables {a b : ℝ} (x : ℝ)
noncomputable def equation1 := (a * b) ^ x - 2
noncomputable def equation2 := (b * a) ^ x - 7

theorem expression_right_side :
  equation1 a b 4.5 = equation2 a b 4.5 :=
sorry

end expression_right_side_l62_62338


namespace GQ_length_l62_62492

/-- Geometry problem setup:
   In triangle XYZ with XY = 13, XZ = 15, YZ = 24, we need to prove that the distance
   from the centroid G to the foot Q of the altitude from G to YZ is √286/36.
-/
noncomputable def X : ℝ := 13
noncomputable def Y : ℝ := 15
noncomputable def Z : ℝ := 24

noncomputable def triangle_XYZ := (X, Y, Z)

-- The centroid G properties and the altitude to YZ.
theorem GQ_length :
  let s := (X + Y + Z) / 2,
      area := Real.sqrt (s * (s - X) * (s - Y) * (s - Z)),
      RQ := (2 * area) / Z in
  let GQ := RQ / 3 in
  GQ = Real.sqrt 286 / 36 :=
begin
  sorry
end

end GQ_length_l62_62492


namespace molecular_weight_correct_l62_62997

-- Define the atomic weights
def atomic_weights : ℕ → ℝ
| 1 := 1.008 -- Hydrogen
| 6 := 12.01 -- Carbon
| 8 := 16.00 -- Oxygen
| _ := 0

-- Define the molecular weight calculation
noncomputable def molecular_weight (n_C n_H n_O : ℕ) : ℝ :=
  n_C * atomic_weights 6 + n_H * atomic_weights 1 + n_O * atomic_weights 8

-- Given conditions
def n_C := 2
def n_O := 2
def total_molecular_weight := 60.0

-- Proven requirement
def n_H := 4

theorem molecular_weight_correct : molecular_weight n_C n_H n_O = total_molecular_weight :=
  by calculate from given the conditions and correct answer ensure it is proven

end molecular_weight_correct_l62_62997


namespace tree_height_l62_62238

theorem tree_height (boy_initial_height tree_initial_height boy_final_height boy_growth_rate tree_growth_rate : ℝ) 
  (h1 : boy_initial_height = 24) 
  (h2 : tree_initial_height = 16) 
  (h3 : boy_final_height = 36) 
  (h4 : boy_growth_rate = boy_final_height - boy_initial_height) 
  (h5 : tree_growth_rate = 2 * boy_growth_rate) 
  : tree_initial_height + tree_growth_rate = 40 := 
by
  subst h1 h2 h3 h4 h5;
  sorry

end tree_height_l62_62238


namespace theseus_path_proof_l62_62174

-- Definitions for the moves in the plane
structure Point where
  x : Int
  y : Int

def step_north (p : Point) : Point := { x := p.x, y := p.y + 1 }
def step_west (p : Point) : Point := { x := p.x - 1, y := p.y }
def step_south (p : Point) : Point := { x := p.x, y := p.y - 1 }
def step_east (p : Point) : Point := { x := p.x + 1, y := p.y }

-- Main proof statement
theorem theseus_path_proof (moves : List (Point → Point)) :
  (moves.head (Point.mk 0 0)).y = -1 -> -- First move south
  (List.last (moves ++ [id]) { x := 0, y := 0 }) = { x := 0, y := 0 } -> -- Ends at (0,0)
  ∀ (p : Point), (p ∈ moves.map id) → List.count p (moves.map id) = 1 -> -- Visits no point more than once
  ∀ (X Y : Nat), -- Setup for X and Y counts
  X = List.count (λ (a : Point → Point), a = step_north ∘ step_west) moves ->
  Y = List.count (λ (a : Point → Point), a = step_west ∘ step_north) moves ->
  abs (X - Y) = 1 := sorry

end theseus_path_proof_l62_62174


namespace mr_brown_at_least_five_sons_l62_62526

open ProbabilityTheory

noncomputable def at_least_five_sons_probability (n : ℕ) (p_son : ℝ) (p_daughter : ℝ) :=
  ∑ k in finset.range 4, (nat.choose 8 (5 + k)) * (p_son ^ (5 + k)) * (p_daughter ^ (8 - (5 + k)))

theorem mr_brown_at_least_five_sons : 
  at_least_five_sons_probability 8 0.6 0.4 = 0.594 :=
by
  sorry

end mr_brown_at_least_five_sons_l62_62526


namespace player_one_wins_l62_62917

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end player_one_wins_l62_62917


namespace f_decreasing_on_negative_interval_and_min_value_l62_62014

noncomputable def f : ℝ → ℝ := sorry

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def minimum_value (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m ∧ ∃ x0, f x0 = m

-- Given the conditions
variables (condition1 : even_function f)
          (condition2 : increasing_on_interval f 3 7)
          (condition3 : minimum_value f 2)

-- Prove that f is decreasing on [-7,-3] and minimum value is 2
theorem f_decreasing_on_negative_interval_and_min_value :
  ∀ x y, -7 ≤ x → x ≤ y → y ≤ -3 → f y ≤ f x ∧ minimum_value f 2 :=
sorry

end f_decreasing_on_negative_interval_and_min_value_l62_62014


namespace pascal_triangle_ratio_l62_62019

theorem pascal_triangle_ratio (n r : ℕ) (hn1 : 5 * r = 2 * n - 3) (hn2 : 7 * r = 3 * n - 11) : n = 34 :=
by
  -- The proof steps will fill here eventually
  sorry

end pascal_triangle_ratio_l62_62019


namespace same_number_of_heads_probability_l62_62813

theorem same_number_of_heads_probability :
  let p := (1 / 2 * (2 / 5 + 3 / 5)) in
  ((1 / 5) ^ 2 + (1 / 2) ^ 2 + (3 / 10) ^ 2) = (19 / 50)
 : m + n = 69 :=
by
  -- Placeholder for the proof
  sorry

end same_number_of_heads_probability_l62_62813


namespace lines_concur_l62_62036

-- Define the given conditions.
variables {A B C D E F G J K M N : Type}
variable [InnerProductSpace ℝ A]
variables (ABC : Triangle A B C) (D : Point A B C) 
variables (E : Incenter A B D) (F : Incenter A C D)
variables (circleE : Circle E (distance E D)) (circleF : Circle F (distance F D))
variables (G : Point (circleE ∩ circleF))
variables (J K : Point (circleE ∩ (segment A B)) (circleE ∩ (segment B C)))
variables (M N : Point (circleF ∩ (segment A C)) (circleF ∩ (segment B C)))

-- The statement we wish to prove.
theorem lines_concur (ABC_triangle : Triangle ABC)
  (point_D_on_BC : Segment B C contains D)
  (incenter_E_ABD : E is incenter of triangle ABD)
  (incenter_F_ACD : F is incenter of triangle ACD)
  (circle_E_with_center_E_and_radius_ED : circleE center E radius distance E D)
  (circle_F_with_center_F_and_radius_FD : circleF center F radius distance F D)
  (intersection_G : G ∈ (circleE ∩ circleF))
  (intersection_JK : J ∈ (circleE ∩ segment A B) ∧ K ∈ (circleE ∩ segment B C))
  (intersection_MN : M ∈ (circleF ∩ segment A C) ∧ N ∈ (circleF ∩ segment B C)) :
  Lines (line_through J K) (line_through M N) (line_through G D) are concurrent :=
sorry

end lines_concur_l62_62036


namespace study_tour_arrangement_l62_62657

def number_of_arrangements (classes routes : ℕ) (max_selected_route : ℕ) : ℕ :=
  if classes = 4 ∧ routes = 4 ∧ max_selected_route = 2 then 240 else 0

theorem study_tour_arrangement :
  number_of_arrangements 4 4 2 = 240 :=
by sorry

end study_tour_arrangement_l62_62657


namespace number_divisible_by_3_or_5_or_both_l62_62761

theorem number_divisible_by_3_or_5_or_both : 
  let num_div_by_3 := (60 / 3).toNat
  let num_div_by_5 := (60 / 5).toNat
  let num_div_by_15 := (60 / 15).toNat
  num_div_by_3 + num_div_by_5 - num_div_by_15 = 28 :=
by {
  let num_div_by_3 := (60 / 3).toNat
  let num_div_by_5 := (60 / 5).toNat
  let num_div_by_15 := (60 / 15).toNat
  show num_div_by_3 + num_div_by_5 - num_div_by_15 = 28
}

end number_divisible_by_3_or_5_or_both_l62_62761


namespace marble_price_proof_l62_62256

noncomputable def price_per_colored_marble (total_marbles white_percentage black_percentage white_price black_price total_earnings : ℕ) : ℕ :=
  let white_marbles := total_marbles * white_percentage / 100
  let black_marbles := total_marbles * black_percentage / 100
  let colored_marbles := total_marbles - (white_marbles + black_marbles)
  let earnings_from_white := white_marbles * white_price
  let earnings_from_black := black_marbles * black_price
  let earnings_from_colored := total_earnings - (earnings_from_white + earnings_from_black)
  earnings_from_colored / colored_marbles

theorem marble_price_proof : price_per_colored_marble 100 20 30 5 10 1400 = 20 := 
sorry

end marble_price_proof_l62_62256


namespace cube_root_of_27_l62_62118

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l62_62118


namespace find_point_M_l62_62035

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem find_point_M : 
  ∃ y : ℝ, (dist (0, y, 0) (1, 0, 2) = dist (0, y, 0) (1, -3, 1)) → (0, y, 0) = (0, -1, 0) :=
by
  sorry

end find_point_M_l62_62035


namespace gcf_72_108_l62_62191

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l62_62191


namespace even_function_a_eq_zero_l62_62437

theorem even_function_a_eq_zero :
  ∀ a, (∀ x, (x + a) * log ((2 * x - 1) / (2 * x + 1)) = (a - x) * log ((1 - 2 * x) / (2 * x + 1)) → a = 0) :=
by
  sorry

end even_function_a_eq_zero_l62_62437


namespace house_value_l62_62331

open Nat

-- Define the conditions
variables (V x : ℕ)
variables (split_amount money_paid : ℕ)
variables (houses_brothers youngest_received : ℕ)
variables (y1 y2 : ℕ)

-- Hypotheses from the conditions
def conditions (V x split_amount money_paid houses_brothers youngest_received y1 y2 : ℕ) :=
  (split_amount = V / 5) ∧
  (houses_brothers = 3) ∧
  (money_paid = 2000) ∧
  (youngest_received = 3000) ∧
  (3 * houses_brothers * money_paid = 6000) ∧
  (y1 = youngest_received) ∧
  (y2 = youngest_received) ∧
  (3 * x + 6000 = V)

-- Main theorem stating the value of one house
theorem house_value (V x : ℕ) (split_amount money_paid houses_brothers youngest_received y1 y2: ℕ) :
  conditions V x split_amount money_paid houses_brothers youngest_received y1 y2 →
  x = 3000 :=
by
  intros
  simp [conditions] at *
  sorry

end house_value_l62_62331


namespace max_m_condition_l62_62682

theorem max_m_condition 
  (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (T_n : ℕ → ℚ)
  (c : ℤ) 
  (h_Sn : ∀ n : ℕ, S_n n = (1 / 2) * n * a_n n + a_n n - c)
  (h_a2 : a_n 2 = 6)
  (b_n : ℕ → ℚ := λ n, (a_n n - 2) / 2 ^ (n+1))
  (h_Tn : ∀ n : ℕ, T_n n = (∑ i in finset.range n, b_n i)) :
  2 * (T_n 1) > max_m - 2 → max_m = 2 := 
by
  sorry

end max_m_condition_l62_62682


namespace custom_operation_example_l62_62294

def custom_operation (a b : ℝ) : ℝ :=
  a - a / b

theorem custom_operation_example :
  custom_operation 8 4 = 6 :=
by
  -- proof steps go here
  sorry

end custom_operation_example_l62_62294


namespace area_of_square_l62_62102

theorem area_of_square : ∀ (x y : ℝ), (x^2 - 2*x + y^2 - 4*y = 12) → (∀ a b : ℝ, (a - 1)^2 + (b - 2)^2 ≤ 17 → (x - 1)^2 + (y - 2)^2 = 17) → (x^2 - 2*x + y^2 - 4*y = 12) → 
  ∃ (s : ℝ), (s = 2* √ 17) → (s^2 = 68) :=
begin
   intros x y h1 h2 h3,
   use 2 * Real.sqrt 17,
   split,
   {
     sorry,
   },
   {
     sorry,
   }
end

end area_of_square_l62_62102


namespace circle_radius_l62_62099

theorem circle_radius (r : ℝ) (A : ℝ) (hA : A = 36 * real.pi) : (real.pi * r^2 = A) → r = 6 :=
  by
  sorry

end circle_radius_l62_62099


namespace caravan_humps_l62_62969

theorem caravan_humps (N : ℕ) (h1 : 1 ≤ N) (h2 : N ≤ 99) 
  (h3 : ∀ (S : set ℕ), S.card = 62 → (∑ x in S, (if x ≤ N then 2 else 1)) ≥ (100 + N) / 2) :
  (∃ A : set ℕ, A.card = 72 ∧ ∀ n ∈ A, 1 ≤ n ∧ n ≤ N) :=
sorry

end caravan_humps_l62_62969


namespace valid_digit_removal_l62_62663

def sum_digits (a b c d e : ℕ) : ℕ := a + b + c + d + e

def alt_sum (a b c d e : ℕ) : ℤ := a - b + c - d + e

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0
def is_divisible_by_11 (n : ℤ) : Prop := n % 11 = 0
def is_divisible_by_99 (n : ℕ) : Prop := is_divisible_by_9 n ∧ is_divisible_by_11 n

theorem valid_digit_removal 
  (a b c d e : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b)
  (h_orig_99 : is_divisible_by_99 (10000 * a + 1000 * b + 100 * c + 10 * d + e)) :
  ∃ r, r ≠ a ∧ r ≠ b ∧ 
  (is_divisible_by_99 (sum_digits a b c d e - r)) :=
begin
  -- This is where the proof would go.
  sorry
end

end valid_digit_removal_l62_62663


namespace intersect_C_UM_N_l62_62846

open Set

variable {U : Type} [DecidableEq U] (M : Set ℤ) (N : Set ℤ) (C : Set ℤ) (U_set : Set ℤ)

def M := {x : ℤ | x < 3}
def N := {x : ℤ | x < 4}
def U_set := set.univ
def C_U_M := {x : ℤ | x ≥ 3}

theorem intersect_C_UM_N : (C_U_M ∩ N) = {3} := by
  sorry

end intersect_C_UM_N_l62_62846


namespace min_value_sin_shifted_l62_62574

theorem min_value_sin_shifted (φ : ℝ) (hφ : |φ| < π / 2) :
    (f(x) = sin (2 * x - π / 3)) ∈ set.Icc 0 (π / 2) → ∃ x ∈ set.Icc 0 (π / 2), f(x) = -√3 / 2 := 
by
  sorry

end min_value_sin_shifted_l62_62574


namespace fraction_exponentiation_multiplication_l62_62286

theorem fraction_exponentiation_multiplication :
  (1 / 3) ^ 4 * (1 / 8) = 1 / 648 :=
by
  sorry

end fraction_exponentiation_multiplication_l62_62286


namespace time_to_fill_tank_with_leak_l62_62257

-- Definitions as per the given problem conditions
def pump_rate : ℝ := 1 / 10
def leak_rate : ℝ := 1 / 20
def effective_filling_rate : ℝ := pump_rate - leak_rate

-- Theorem statement proving the time taken to fill the tank with the leak
theorem time_to_fill_tank_with_leak : effective_filling_rate = 1 / 20 → 
  (1 / effective_filling_rate) = 20 :=
by 
  intros h_effective_rate
  rw h_effective_rate
  norm_num
  sorry

end time_to_fill_tank_with_leak_l62_62257


namespace greatest_int_sqrt10_minus_5_l62_62635

theorem greatest_int_sqrt10_minus_5 :
  (let x := Real.sqrt 10 in ⌊x - 5⌋ = -2) :=
by
  sorry

end greatest_int_sqrt10_minus_5_l62_62635


namespace possible_N_values_l62_62955

theorem possible_N_values : 
  let total_camels := 100 in
  let humps n := total_camels + n in
  let one_humped_camels n := total_camels - n in
  let condition1 (n : ℕ) := (62 ≥ (humps n) / 2)
  let condition2 (n : ℕ) := ∀ y : ℕ, 1 ≤ y → 62 + y ≥ (humps n) / 2 → n ≥ 52 in
  ∃ N, 1 ≤ N ∧ N ≤ 24 ∨ 52 ≤ N ∧ N ≤ 99 → N = 72 :=
by 
  -- Placeholder proof
  sorry

end possible_N_values_l62_62955


namespace nina_basketball_cards_l62_62854

theorem nina_basketball_cards (cost_toy cost_shirt cost_card total_spent : ℕ) (n_toys n_shirts n_cards n_packs_result : ℕ)
  (h1 : cost_toy = 10)
  (h2 : cost_shirt = 6)
  (h3 : cost_card = 5)
  (h4 : n_toys = 3)
  (h5 : n_shirts = 5)
  (h6 : total_spent = 70)
  (h7 : n_packs_result =  2)
  : (3 * cost_toy + 5 * cost_shirt + n_cards * cost_card = total_spent) → n_cards = n_packs_result :=
by
  sorry

end nina_basketball_cards_l62_62854


namespace possible_values_of_N_count_l62_62948

def total_camels : ℕ := 100

def total_humps (N : ℕ) : ℕ := total_camels + N

def subset_condition (N : ℕ) (subset_size : ℕ) : Prop :=
  ∀ (s : finset ℕ), s.card = subset_size → ∑ x in s, if x < N then 2 else 1 ≥ (total_humps N) / 2

theorem possible_values_of_N_count : 
  ∃ N_set : finset ℕ, N_set = (finset.range 100).filter (λ N, 1 ≤ N ∧ N ≤ 99 ∧ subset_condition N 62) ∧ 
  N_set.card = 72 :=
sorry

end possible_values_of_N_count_l62_62948


namespace circle_equation_l62_62742

theorem circle_equation (x y : ℝ) :
  let A := (2 : ℝ, 0 : ℝ)
  let B := (2 : ℝ, (-2) : ℝ)
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let radius := real.dist A B / 2
  center = (2, -1) ∧ radius = 1 →
  (x - 2)^2 + (y + 1)^2 = 1 :=
begin
  sorry
end

end circle_equation_l62_62742


namespace prime_factor_of_sum_of_four_consecutive_integers_l62_62170

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l62_62170


namespace smallest_square_condition_l62_62326

-- Definition of the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_last_digit_not_zero (n : ℕ) : Prop := n % 10 ≠ 0

def remove_last_two_digits (n : ℕ) : ℕ :=
  n / 100

-- The statement of the theorem we need to prove
theorem smallest_square_condition : 
  ∃ n : ℕ, is_square n ∧ has_last_digit_not_zero n ∧ is_square (remove_last_two_digits n) ∧ 121 ≤ n :=
sorry

end smallest_square_condition_l62_62326


namespace determine_winner_l62_62990

-- Mathematical conditions
variable (a : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a ≠ 1)

-- Theorem statement
theorem determine_winner (h₀ : a ≠ 0) (h₁ : a ≠ 1) : 
  (a > 1 ∨ (0 < a ∧ a < 1)) → SashoWins a ∧ a < 0 → DeniWins a := sorry

end determine_winner_l62_62990


namespace sum_of_squares_of_roots_l62_62329

theorem sum_of_squares_of_roots :
  let s1 := ((10 + real.sqrt (10^2 - 4*9))/2 : ℝ)
  let s2 := ((10 - real.sqrt (10^2 - 4*9))/2 : ℝ)
  s1^2 + s2^2 = 82 :=
by
  let s1 := ((10 + real.sqrt (10^2 - 4*9))/2 : ℝ)
  let s2 := ((10 - real.sqrt (10^2 - 4*9))/2 : ℝ)
  have h1 : s1 + s2 = 10 := by
    sorry
  have h2 : s1 * s2 = 9 := by
    sorry
  calc
  s1^2 + s2^2 = (s1 + s2)^2 - 2 * (s1 * s2) : by sorry
  ... = 82 : by
    rw [h1, h2]
    norm_num
end

end sum_of_squares_of_roots_l62_62329


namespace root_interval_sum_l62_62391

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 8

def has_root_in_interval (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  a < b ∧ b - a = 1 ∧ f a < 0 ∧ f b > 0

theorem root_interval_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : has_root_in_interval a b h1 h2) : 
  a + b = 5 :=
sorry

end root_interval_sum_l62_62391


namespace ellipse_equation_hyperbola_equation_l62_62215

-- Ellipse problem statement
theorem ellipse_equation (c : ℝ := 2) (x1 y1 : ℝ := (-2, -real.sqrt 2)) (a b : ℝ)
  (h_foci : c^2 = a^2 - b^2) (h_passes_point : (x1 / a)^2 + (y1 / b)^2 = 1) :
  (∃ a b : ℝ, (x1 / a)^2 + (y1 / b)^2 = 1 ∧ c^2 = a^2 - b^2) → 
  (x / a)^2 + (y / b)^2 = 1 :=
sorry

-- Hyperbola problem statement
theorem hyperbola_equation (c : ℝ := 5) (m : ℝ := 4 / 3) (a b : ℝ)
  (h_common_asymptote : m = a / b) (h_foci : c^2 = a^2 + b^2) :
  (∃ a b : ℝ, c^2 = a^2 + b^2 ∧ m = a / b) → 
  (y / a)^2 - (x / b)^2 = 1 :=
sorry

end ellipse_equation_hyperbola_equation_l62_62215


namespace cube_root_of_27_eq_3_l62_62114

theorem cube_root_of_27_eq_3 : real.cbrt 27 = 3 :=
by {
  have h : 27 = 3 ^ 3 := by norm_num,
  rw real.cbrt_eq_iff_pow_eq (by norm_num : 0 ≤ 27) h,
  norm_num,
  sorry
}

end cube_root_of_27_eq_3_l62_62114


namespace prove_a_zero_l62_62415

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62415


namespace frustum_views_l62_62569

theorem frustum_views : 
  (∀{F S : view}, congruent_isosceles_trapezoids F S) ∧ 
  (∀{T : view}, two_concentric_circles T) :=
sorry

end frustum_views_l62_62569


namespace three_person_subcommittees_from_eight_l62_62410

theorem three_person_subcommittees_from_eight : 
  (combinatorics.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l62_62410


namespace boy_returns_home_eventually_l62_62221

universe u

variable {V : Type u}

structure Graph (V : Type u) :=
  (adj : V → V → Prop)
  (regular_3 : ∀ v, ∃! w₁ w₂ w₃, adj v w₁ ∧ adj v w₂ ∧ adj v w₃ ∧ w₁ ≠ w₂ ∧ w₂ ≠ w₃ ∧ w₃ ≠ w₁)

variable (G : Graph V) (home : V)

inductive Direction
| Left
| Right

def walk_pattern : ℕ → Direction
| 0     => Direction.Left
| (n+1) => if n % 2 = 0 then Direction.Right else Direction.Left

def next_vertex (v : V) (d : Direction) : V := sorry -- Placeholder for moving along the next vertex depending on direction.

noncomputable def boy_returns_home_after_steps (n : ℕ) : Prop :=
  ∃ i j, i < j ∧ ∀ k, i ≤ k ∧ k ≤ j → next_vertex home (walk_pattern k) = home

theorem boy_returns_home_eventually :
  ∃ n, boy_returns_home_after_steps G home n :=
sorry

end boy_returns_home_eventually_l62_62221


namespace cos_log_equivalent_l62_62002

theorem cos_log_equivalent (a : ℝ) (h : log a (a + 6) = 2) :
  (cos (-22 / 3 * π)) ^ a = -1 / 8 :=
by 
  sorry

end cos_log_equivalent_l62_62002


namespace Jessie_current_weight_l62_62817

-- Define Jessie's initial weight and the weight she lost
def initialWeight : ℕ := 74
def weightLost : ℕ := 7

-- Define the current weight based on the initial weight and the weight lost
def currentWeight : ℕ := initialWeight - weightLost

-- Prove that the current weight is 67 kilograms
theorem Jessie_current_weight : currentWeight = 67 := by
  unfold currentWeight
  unfold initialWeight
  unfold weightLost
  simp
  sorry

end Jessie_current_weight_l62_62817


namespace T_10_value_l62_62385

noncomputable theory

open_locale big_operators

-- Definitions that help setting up the conditions and notation
def a_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d
def S_n (a₁ d : ℕ) (n : ℕ) : ℕ := n * a₁ + (n * (n - 1) / 2) * d
def T_n (a_n : ℕ → ℕ) (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (a_n i * a_n (i + 1))

-- Stating the problem conditions and question as a theorem
theorem T_10_value :
  ∃ a₁ d, (a_seq a₁ d 3 + a_seq a₁ d 4 = 7) ∧ (S_n a₁ d 5 = 15) ∧ (T_n (a_seq a₁ d) 10 = 10 / 11) :=
begin
  sorry
end

end T_10_value_l62_62385


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62166

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62166


namespace semicircle_perimeter_approx_l62_62207

def radius : ℝ := 12
def pi_approx : ℝ := 3.14159
def half_circumference (r : ℝ) (π : ℝ) : ℝ := π * r
def diameter (r : ℝ) : ℝ := 2 * r
def perimeter (r : ℝ) (π : ℝ) : ℝ := half_circumference r π + diameter r

theorem semicircle_perimeter_approx (r : ℝ) (π : ℝ) (h_r : r = 12) (h_π : π = 3.14159) : perimeter r π ≈ 61.7 :=
by 
  rw [h_r, h_π]
  -- replace the function applications with their values to make the arithmetic clearer
  have half_c : half_circumference 12 3.14159 = 37.69908 := by sorry
  have diam : diameter 12 = 24 := by sorry
  rw [half_c, diam]
  -- add the components of the perimeter
  have perim : 37.69908 + 24 ≈ 61.7 := by sorry
  exact perim

end semicircle_perimeter_approx_l62_62207


namespace concurrency_of_lines_l62_62064

noncomputable def is_incenter (P A B : Point) : Prop :=
  ∀ Q, Q ≠ P → (distance Q A) + (distance Q B) > (distance P A) + (distance P B)

noncomputable def are_concurrent (A B C D E : Point) : Prop :=
  ∃ P, collinear P A D ∧ collinear P B E

theorem concurrency_of_lines
  (A B C P D E : Point)
  (h1 : ∡ A P B - ∡ A C B = ∡ A P C - ∡ A B C)
  (h2 : is_incenter D A P B)
  (h3 : is_incenter E A P C) :
  are_concurrent A B C D E :=
sorry

end concurrency_of_lines_l62_62064


namespace tangent_line_at_point_M_l62_62566

-- Define the conditions: the circle and the point M.
def circle (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_M : ℝ × ℝ := (1, 2)

-- Define the statement: the equation of the tangent line at point M on the circle.
theorem tangent_line_at_point_M : 
  circle 1 2 → ∃ (a b c : ℝ), a * 1 + b * 2 + c = 0 ∧ (a = 1 ∧ b = 2 ∧ c = -5) := 
by
  sorry

end tangent_line_at_point_M_l62_62566


namespace evaluate_expression_l62_62307

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end evaluate_expression_l62_62307


namespace fewer_shots_third_period_l62_62049

noncomputable def shotsInFirstPeriod : ℕ := 4

noncomputable def shotsInSecondPeriod : ℕ := 2 * shotsInFirstPeriod

variable (shotsInThirdPeriod : ℕ) (shotsInFourthPeriod : ℕ)

axiom shotsTotal : shotsInFirstPeriod + shotsInSecondPeriod + shotsInThirdPeriod + shotsInFourthPeriod = 21

theorem fewer_shots_third_period :
  |shotsInSecondPeriod - shotsInThirdPeriod| = 1 :=
by
  have shotsInFirst := shotsInFirstPeriod
  have shotsInSecond := shotsInSecondPeriod
  have shotsSum := shotsInFirst + shotsInSecond
  have shotsInAllPeriods := 21
  have shotsRemained := shotsInAllPeriods - shotsSum
  have h : shotsInThirdPeriod = shotsRemained
  rw [h]
  sorry

end fewer_shots_third_period_l62_62049


namespace probability_angle_CAM_less_45_l62_62731

theorem probability_angle_CAM_less_45 {ABC : Triangle}
  (h_right : ABC.is_right ∠C)
  (h_A : ∠A = 60)
  (ray_AM : Ray AM ∈ interior_angle ∠CAB) :
  P(∠CAM < 45) = ¾ :=
sorry

end probability_angle_CAM_less_45_l62_62731


namespace square_of_QP_l62_62779

-- Define the necessary entities and conditions
variables (r1 r2 d x : ℝ)
variables (P : Type) -- Type of points in the circles

-- Assume the radii and distance between the centers
axiom radii_def : r1 = 10 ∧ r2 = 7
axiom distance_def : d = 15

-- Assume the length of the chords are equal
axiom chords_equal : ∃ (Q R : P), QP = x ∧ PR = x

-- State and prove the main theorem that given these conditions, the square of x equals 309
theorem square_of_QP : x^2 = 309 :=
by 
sorry

end square_of_QP_l62_62779


namespace solve_system_of_equations_l62_62878

theorem solve_system_of_equations (n : ℕ) (x : Fin n → ℝ) :
  (∀ i : Fin n, 1 - x i * x ((i + 1) % n) = 0) → 
  (if n % 2 = 1 
   then ∃ k : ℝ, k ≠ 0 ∧ (∀ i : Fin n, x i = k)
   else ∃ a : ℝ, a ≠ 0 ∧ (∀ i : Fin n, (if i % 2 = 0 then x i = a else x i = 1 / a))) :=
by
  sorry

end solve_system_of_equations_l62_62878


namespace prove_a_zero_l62_62420

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62420


namespace total_weekly_airflow_l62_62147

-- Definitions from conditions
def fanA_airflow : ℝ := 10  -- liters per second
def fanA_time_per_day : ℝ := 10 * 60  -- converted to seconds (10 minutes * 60 seconds/minute)

def fanB_airflow : ℝ := 15  -- liters per second
def fanB_time_per_day : ℝ := 20 * 60  -- converted to seconds (20 minutes * 60 seconds/minute)

def fanC_airflow : ℝ := 25  -- liters per second
def fanC_time_per_day : ℝ := 30 * 60  -- converted to seconds (30 minutes * 60 seconds/minute)

def days_in_week : ℝ := 7

-- Theorem statement to be proven
theorem total_weekly_airflow : fanA_airflow * fanA_time_per_day * days_in_week +
                               fanB_airflow * fanB_time_per_day * days_in_week +
                               fanC_airflow * fanC_time_per_day * days_in_week = 483000 := 
by
  -- skip the proof
  sorry

end total_weekly_airflow_l62_62147


namespace find_y_given_z_25_l62_62004

theorem find_y_given_z_25 (k m x y z : ℝ) 
  (hk : y = k * x) 
  (hm : z = m * x)
  (hy5 : y = 10) 
  (hx5z15 : z = 15) 
  (hz25 : z = 25) : 
  y = 50 / 3 := 
  by sorry

end find_y_given_z_25_l62_62004


namespace minimum_perimeter_l62_62399

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) := x^2 / 4 - y^2 / 2 = 1

-- Define the right focus F
def F := (sqrt 6, 0)

-- Define point A
def A := (0, sqrt 2)

-- Define the distance function
def dist (P Q : ℝ × ℝ) : ℝ := sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the left focus M
def M := (-sqrt 6, 0)

-- Define AP + PM collinearity condition
def collinear (AP PM AM : ℝ) := AP + PM = AM

-- Define the minimum perimeter problem
theorem minimum_perimeter : 
  (∃ (P : ℝ × ℝ), hyperbola_eq P.1 P.2 ∧ P.1 < 0) → 
  dist A F + 4 + ∃ (P : ℝ × ℝ), hyperbola_eq P.1 P.2 ∧ P.1 < 0) + dist M P.2 = 4 * (1 + sqrt 2) := 
sorry

end minimum_perimeter_l62_62399


namespace projection_equal_p_l62_62200

open Real EuclideanSpace

noncomputable def vector1 : ℝ × ℝ := (-3, 4)
noncomputable def vector2 : ℝ × ℝ := (1, 6)
noncomputable def v : ℝ × ℝ := (4, 2)
noncomputable def p : ℝ × ℝ := (-2.2, 4.4)

theorem projection_equal_p (p_ortho : (p.1 * v.1 + p.2 * v.2) = 0) : p = (4 * (1 / 5) - 3, 2 * (1 / 5) + 4) :=
by
  sorry

end projection_equal_p_l62_62200


namespace imaginary_part_of_conjugate_l62_62350

noncomputable def a (b : ℝ) : ℂ := 1

noncomputable def z (b : ℝ) : ℂ := a b + b * complex.I

theorem imaginary_part_of_conjugate (b : ℝ) (h : b = -2) : complex.im (complex.conj (z b)) = 2 :=
by
  sorry

end imaginary_part_of_conjugate_l62_62350


namespace min_value_of_squares_l62_62770

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end min_value_of_squares_l62_62770


namespace curve_equation_areas_ratio_proof_l62_62404

noncomputable def curve_satisfies_condition : Prop := 
  ∀ (x y : ℝ), 
  let M := (x, y) in 
  let A := (-1, 1) in 
  let B := (2, 1) in 
  ∥((-2 - x), (1 - y)) + ((2 - x), (1 - y))∥ = 
  (x * 0 + y * 2 + 2) :=
  {x : ℝ // x^2 = 4 * y}

theorem curve_equation : curve_satisfies_condition :=
sorry

noncomputable def ratio_of_areas : Prop := 
  ∀ (x0 y0 : ℝ), 
  -2 < x0 ∧ x0 < 2 ∧ y0 = x0^2 / 4 →
  ∃ D E : (ℝ × ℝ), 
  let Q := (x0, y0) in 
  let P := (0, -1) in 
  let A := (-1, 1) in 
  let B := (2, 1) in
  let PA := {x : ℝ × ℝ | ∃ (t : ℝ), x = (-t, -1 + t)} in
  let PB := {x : ℝ × ℝ | ∃ (t : ℝ), x = (t, -1 + t)} in
  let l := {y : ℝ | y = x0 / 2 * x - x0^2 / 4} in
  let triangle_area (P Q R : (ℝ × ℝ)) := 
    0.5 * |P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)| in
  let ratio := 
    triangle_area Q A B / triangle_area P D E in 
  ratio = 2

theorem areas_ratio_proof : ratio_of_areas :=
sorry

end curve_equation_areas_ratio_proof_l62_62404


namespace probability_of_positive_l62_62483

-- Definitions based on the conditions
def balls : List ℚ := [-2, 0, 1/4, 3]
def total_balls : ℕ := 4
def positive_filter (x : ℚ) : Bool := x > 0
def positive_balls : List ℚ := balls.filter positive_filter
def positive_count : ℕ := positive_balls.length
def probability : ℚ := positive_count / total_balls

-- Statement to prove
theorem probability_of_positive : probability = 1 / 2 := by
  sorry

end probability_of_positive_l62_62483


namespace angle_BPC_proof_l62_62793

-- Define the square ABCD with side length 6
def side_length : ℝ := 6

-- Define the lengths in the isosceles triangle ABE
def AB : ℝ := side_length
def AE : ℝ := side_length
def BE : ℝ := 3 * Real.sqrt 5

-- Define the intersection and perpendicular conditions
def P_intersection (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ (A B E : ℝ × ℝ), P = (AC_intersect_BE A B E)

def Q_on_BC (Q : ℝ × ℝ) : Prop :=
  ∃ (B C : ℝ × ℝ), Q ∈ BC B C ∧ PQ_perpendicular_BC Q

-- Define the known angles in the square
def angle_ABC : ℝ := 90
def angle_BCA : ℝ := 45

-- Statement of the problem in Lean 4
theorem angle_BPC_proof :
  ∀ {A B C D E P Q : ℝ × ℝ},
  P_intersection P →
  Q_on_BC Q →
  angle_BPC = 45 + Real.arccos (3 / 8) :=
sorry -- Proof omitted

end angle_BPC_proof_l62_62793


namespace sqrt_nine_l62_62677

theorem sqrt_nine : sqrt 9 = 3 :=
by sorry

end sqrt_nine_l62_62677


namespace sum_series_eq_l62_62713

/--
Given the series \(T_n = \sum_{k=1}^{n} k \cdot 2^{k-1}\),
prove that \(T_n = 1 + (n-1) \cdot 2^n\).
-/
theorem sum_series_eq (n : ℕ) : 
  (∑ k in Finset.range n, (k + 1) * 2^k) = 1 + (n - 1) * 2^n := 
sorry

end sum_series_eq_l62_62713


namespace problem_proof_l62_62298

open Matrix

noncomputable def proof_example : Prop :=
  ∃ c d : ℚ, (matrix.mul (λ i j, if i = 0 then if j = 0 then 2 else -2 else if j = 0 then c else d) 
                       (λ i j, if i = 0 then if j = 0 then 2 else -2 else if j = 0 then c else d) 
          = (1 : ℚ) • (1 : matrix (fin 2) (fin 2) ℚ)) ∧ c = 3/2 ∧ d = -2

theorem problem_proof : proof_example := by
  sorry

end problem_proof_l62_62298


namespace locusOfPointInIsoscelesTriangle_l62_62366

variable (A B C P : Point)
variable {AB AC BC : Length}
variable (d_BC d_AB d_AC : P → Length)
variable (geo_mean : Length → Length → Length)

noncomputable def isoscelesTriangle (A B C : Point) : Prop :=
  AB = AC

noncomputable def locusCondition (P : Point) (d_BC d_AB d_AC : P → Length) :=
  d_BC P = geo_mean (d_AB P) (d_AC P)

theorem locusOfPointInIsoscelesTriangle :
  (isoscelesTriangle A B C) →
  (locusCondition P d_BC d_AB d_AC) →
  (∃ (α : Circle), tangent α A B ∧ tangent α A C ∧ P ∈ α ∧ P ∉ {B, C}) :=
sorry

end locusOfPointInIsoscelesTriangle_l62_62366


namespace sum_of_cubes_l62_62856

theorem sum_of_cubes (n : ℕ) : 1^3 + 2^3 + 3^3 + ... + n^3 = (n * (n + 1) / 2)^2 :=
sorry

end sum_of_cubes_l62_62856


namespace fraction_of_new_releases_l62_62261

theorem fraction_of_new_releases (total_books : ℕ) (historical_fiction_percent : ℝ) (historical_new_releases_percent : ℝ) (other_new_releases_percent : ℝ)
  (h1 : total_books = 100)
  (h2 : historical_fiction_percent = 0.4)
  (h3 : historical_new_releases_percent = 0.4)
  (h4 : other_new_releases_percent = 0.2) :
  (historical_fiction_percent * historical_new_releases_percent * total_books) / 
  ((historical_fiction_percent * historical_new_releases_percent * total_books) + ((1 - historical_fiction_percent) * other_new_releases_percent * total_books)) = 4 / 7 :=
by
  have h_books : total_books = 100 := h1
  have h_fiction : historical_fiction_percent = 0.4 := h2
  have h_new_releases : historical_new_releases_percent = 0.4 := h3
  have h_other_new_releases : other_new_releases_percent = 0.2 := h4
  sorry

end fraction_of_new_releases_l62_62261


namespace tangent_identity_l62_62868

theorem tangent_identity 
  (α β γ : ℝ) 
  (hα : tan α = sin α / cos α)
  (hβ : tan β = sin β / cos β)
  (hγ : tan γ = sin γ / cos γ) 
  :
  tan α + tan β + tan γ - sin (α + β + γ) / (cos α * cos β * cos γ) = tan α * tan β * tan γ := 
sorry

end tangent_identity_l62_62868


namespace molecular_weight_of_compound_l62_62197

theorem molecular_weight_of_compound :
  let Cu_atoms := 2
  let C_atoms := 3
  let O_atoms := 5
  let N_atoms := 1
  let atomic_weight_Cu := 63.546
  let atomic_weight_C := 12.011
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  Cu_atoms * atomic_weight_Cu +
  C_atoms * atomic_weight_C +
  O_atoms * atomic_weight_O +
  N_atoms * atomic_weight_N = 257.127 :=
by
  sorry

end molecular_weight_of_compound_l62_62197


namespace area_of_convex_polygon_projections_l62_62580

open Real

def convex_polygon_projection_OX := 4
def convex_polygon_projection_bisector_1_3 := 3 * sqrt 2
def convex_polygon_projection_OY := 5
def convex_polygon_projection_bisector_2_4 := 4 * sqrt 2

theorem area_of_convex_polygon_projections (S : ℝ) 
  (hOX : convex_polygon_projection_OX = 4)
  (hBisector13 : convex_polygon_projection_bisector_1_3 = 3 * sqrt 2)
  (hOY : convex_polygon_projection_OY = 5)
  (hBisector24 : convex_polygon_projection_bisector_2_4 = 4 * sqrt 2) 
  : S ≥ 10 := 
sorry

end area_of_convex_polygon_projections_l62_62580


namespace sanity_indeterminable_transylvanian_is_upyr_l62_62226

noncomputable def transylvanianClaim := "I have lost my mind."

/-- Proving whether the sanity of the Transylvanian can be determined from the statement -/
theorem sanity_indeterminable (claim : String) : 
  claim = transylvanianClaim → 
  ¬ (∀ (sane : Prop), sane ∨ ¬ sane) := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

/-- Proving the nature of whether the Transylvanian is an upyr or human from the statement -/
theorem transylvanian_is_upyr (claim : String) : 
  claim = transylvanianClaim → 
  ∀ (human upyr : Prop), ¬ human ∧ upyr := 
by 
  intro h
  rw [transylvanianClaim] at h
  sorry

end sanity_indeterminable_transylvanian_is_upyr_l62_62226


namespace polar_eqn_of_curve_length_of_segment_ab_l62_62387

noncomputable def parameter_eqn := (x y θ : ℝ) → 
  (x = 1 + sqrt 3 * cos θ) ∧ (y = sqrt 3 * sin θ)

noncomputable def polar_eqn_of_line := (ρ θ : ℝ) → 
  ρ * cos (θ - π / 6) = 3 * sqrt 3

noncomputable def ray_ot := (θ : ℝ) → 
  θ = π / 3

theorem polar_eqn_of_curve (x y θ ρ : ℝ)
    (h1 : parameter_eqn x y θ)
    (h2 : ρ^2 - 2 * ρ * cos θ - 2 = 0) : 
  (x - 1)^2 + y^2 = 3 := by
  sorry -- proof skipped

theorem length_of_segment_ab (ρ : ℝ) 
    (h1 : ρ^2 - 2 * ρ * cos (π / 3) - 2 = 0)
    (h2 : polar_eqn_of_line ρ (π / 3)) : 
  6 - 2 = 4 := by
  sorry -- proof skipped

end polar_eqn_of_curve_length_of_segment_ab_l62_62387


namespace sum_of_coeffs_l62_62344

theorem sum_of_coeffs (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^8 = a + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + 
  a_4 * (x - 1)^4 + a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + 
  a_8 * (x - 1)^8 + a_9 * (x - 1)^9 + a_{10} * (x - 1)^{10)) →
  (2 = a) →
  (0 = a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10}) →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} = -2 :=
by
  sorry

end sum_of_coeffs_l62_62344


namespace average_of_four_numbers_l62_62734

theorem average_of_four_numbers (a b c d : ℝ) 
  (h1 : b + c + d = 24) (h2 : a + c + d = 36)
  (h3 : a + b + d = 28) (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := 
sorry

end average_of_four_numbers_l62_62734


namespace sin_log_zeros_infinitely_many_l62_62000

theorem sin_log_zeros_infinitely_many : 
  ∃ (f : ℝ → ℝ), (∀ x, f x = Real.sin (Real.log x)) ∧ (0 < x ∧ x < 1) → ∃ (n : ℕ), ∞ ∈ set_of (λ x, f x = 0) :=
by 
  sorry

end sin_log_zeros_infinitely_many_l62_62000


namespace camel_humps_l62_62939

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62939


namespace quadrant_of_half_angle_l62_62737

theorem quadrant_of_half_angle (α : ℝ) (k : ℤ) (h : 2 * k * real.pi + real.pi / 2 < α ∧ α < 2 * k * real.pi + real.pi) : 
  (∃ n : ℤ, (2 * n * real.pi + real.pi / 4 < α / 2 ∧ α / 2 < 2 * n * real.pi + real.pi / 2) ∨ ((2 * n + 1) * real.pi + real.pi / 4 < α / 2 ∧ α / 2 < (2 * n + 1) * real.pi + real.pi / 2)) :=
begin
  sorry
end

end quadrant_of_half_angle_l62_62737


namespace total_fencing_cost_is_5300_l62_62575

-- Define the conditions
def length_more_than_breadth_condition (l b : ℕ) := l = b + 40
def fencing_cost_per_meter : ℝ := 26.50
def given_length : ℕ := 70

-- Define the perimeter calculation
def perimeter (l b : ℕ) := 2 * l + 2 * b

-- Define the total cost calculation
def total_cost (P : ℕ) (cost_per_meter : ℝ) := P * cost_per_meter

-- State the theorem to be proven
theorem total_fencing_cost_is_5300 (b : ℕ) (l := given_length) :
  length_more_than_breadth_condition l b →
  total_cost (perimeter l b) fencing_cost_per_meter = 5300 :=
by
  sorry

end total_fencing_cost_is_5300_l62_62575


namespace prove_a_zero_l62_62416

noncomputable def f (x a : ℝ) := (x + a) * log ((2 * x - 1) / (2 * x + 1))

theorem prove_a_zero (a : ℝ) : 
  (∀ x, f (-x a) = f (x a)) → a = 0 :=
by 
  sorry

end prove_a_zero_l62_62416


namespace b_and_c_together_take_days_l62_62622

def workRateTogether (a b : ℝ) : ℝ := 1 / 16
def workRateAlone (a : ℝ) : ℝ := 1 / 20
def workRateC (c : ℝ) : ℝ := 1 / 25

def workRateB (a b : ℝ) : ℝ := workRateTogether a b - workRateAlone a
def combinedWorkRate (b c : ℝ) : ℝ := workRateB b c + workRateC c

theorem b_and_c_together_take_days (a b c : ℝ) (W : ℝ) 
  (h₁ : workRateTogether a b * 16 = W)
  (h₂ : workRateAlone a * 20 = W)
  (h₃ : workRateC c * 25 = W) :
  1 / combinedWorkRate b c = 200 / 33 :=
sorry

end b_and_c_together_take_days_l62_62622


namespace min_total_cost_minimize_cost_l62_62235

theorem min_total_cost (x : ℝ) (h₀ : x > 0) :
  (900 / x * 3 + 3 * x) ≥ 180 :=
by sorry

theorem minimize_cost (x : ℝ) (h₀ : x > 0) :
  x = 30 ↔ (900 / x * 3 + 3 * x) = 180 :=
by sorry

end min_total_cost_minimize_cost_l62_62235


namespace smallest_n_satisfying_inequality_l62_62401
-- Import the Mathlib library

-- Define the sequence \( \{a_n\} \) satisfying the given recurrence relation and initial condition
def sequence (n : ℕ) : ℕ → ℝ
| 0       := 9
| (n + 1) := if n = 0 then 9 else (4 - sequence n) / 3

-- Define the sum of the first \( n \) terms \( S_n \)
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, sequence i

-- State the main theorem we want to prove
theorem smallest_n_satisfying_inequality : ∃ (n : ℕ), 1 ≤ n ∧ abs (S n - n - 6) < 1/125 ∧ ∀ (m : ℕ), 1 ≤ m ∧ m < n → ¬ (abs (S m - m - 6) < 1/125) := by
  sorry

end smallest_n_satisfying_inequality_l62_62401


namespace min_integral_value_l62_62054

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 + x else 1 + 3 * x

theorem min_integral_value : 
  (∀ a b : ℝ, ∫ x in -1..1, (f x - (a * |x| + b))^2 ≥ 8/3) :=
sorry

end min_integral_value_l62_62054


namespace stable_performance_l62_62672

theorem stable_performance (s2_A s2_B : ℝ) (hA : s2_A = 0.1) (hB : s2_B = 0.02) :
  s2_B < s2_A :=
by
  rw [hA, hB]
  exact lt_of_le_of_ne (by norm_num) (by norm_num) -- this would go towards completing the proof but we place;
  sorry -- placeholder for the actual proof

-- This theorem states that given the variances s2_A = 0.1 and s2_B = 0.02, we prove that s2_B < s2_A.

end stable_performance_l62_62672


namespace probability_red_blue_green_l62_62609

def total_marbles : ℕ := 5 + 4 + 3 + 6
def favorable_marbles : ℕ := 5 + 4 + 3

theorem probability_red_blue_green : 
  (favorable_marbles : ℚ) / total_marbles = 2 / 3 := 
by 
  sorry

end probability_red_blue_green_l62_62609


namespace berengere_contribution_l62_62266

theorem berengere_contribution :
  let dessert_cost := 8
  let tom_dollars := 10
  let exchange_rate := 1 / 1.10
  let tom_euros := tom_dollars * exchange_rate
  in tom_euros >= dessert_cost -> 0 = 0 := by
sorry

end berengere_contribution_l62_62266


namespace ratio_melina_alma_age_l62_62910

theorem ratio_melina_alma_age
  (A M : ℕ)
  (alma_score : ℕ)
  (h1 : M = 60)
  (h2 : alma_score = 40)
  (h3 : A + M = 2 * alma_score)
  : M / A = 3 :=
by
  sorry

end ratio_melina_alma_age_l62_62910


namespace greatest_integer_multiple_of_9_unique_digits_div_100_is_81_l62_62507

theorem greatest_integer_multiple_of_9_unique_digits_div_100_is_81 :
  let M := greatest_integer_multiple_of_9_with_unique_digits in 
  M % 100 = 81 :=
sorry

end greatest_integer_multiple_of_9_unique_digits_div_100_is_81_l62_62507


namespace sufficient_condition_of_necessary_condition_l62_62001

-- Define the necessary condition
def necessary_condition (A B : Prop) : Prop := A → B

-- The proof problem statement
theorem sufficient_condition_of_necessary_condition
  {A B : Prop} (h : necessary_condition A B) : necessary_condition A B :=
by
  exact h

end sufficient_condition_of_necessary_condition_l62_62001


namespace sum_of_angles_l62_62130

theorem sum_of_angles (x y : ℝ) :
  (4 * (180 / (4 + 11)) + 7 * (360 / (5 + 6 + 7 + 12))) = 132 :=
by
  have h1 : 4 * (180 / (4 + 11)) = 48 :=
    by
      -- Calculation of the smaller angle of the parallelogram
      sorry
  have h2 : 7 * (360 / (5 + 6 + 7 + 12)) = 84 :=
    by
      -- Calculation of the second largest angle of the quadrilateral
      sorry
  rw [h1, h2]
  norm_num
  sorry

end sum_of_angles_l62_62130


namespace expected_value_10_expected_value_final_answer_l62_62674

noncomputable def E : ℕ → ℚ
| n => if n % 2 = 1 then 1 else (∑ i in finset.range n, if i > 0 ∧ i < n ∧ i % 2 = 1 ∧ (n - i) % 2 = 1 then E i + E (n - i) else 0) / (finset.range n).card

theorem expected_value_10 : E 10 = 9 / 2 :=
by
  sorry

theorem expected_value_final_answer : 100 * 9 + 2 = 902 :=
by
  exact rfl

end expected_value_10_expected_value_final_answer_l62_62674


namespace power_function_increasing_l62_62016

   theorem power_function_increasing (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → x^a < y^a) : 0 < a :=
   by
   sorry
   
end power_function_increasing_l62_62016


namespace eggs_broken_l62_62072

theorem eggs_broken (b w_o t w_l : ℕ) (h1 : b = 5) (h2 : w_o = 3 * b) (h3 : t = 12) (h4 : w_l = t - b) :
  w_o - w_l = 8 :=
by
  rw [h1, h2, h3, h4]
  rw [Nat.sub_sub]
  norm_num
  sorry

end eggs_broken_l62_62072


namespace prime_factors_of_four_consecutive_integers_sum_l62_62149

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l62_62149


namespace necessary_but_not_sufficient_condition_l62_62508

variable {M N P : Set α}

theorem necessary_but_not_sufficient_condition (h : M ∩ P = N ∩ P) : 
  (M = N) → (M ∩ P = N ∩ P) :=
sorry

end necessary_but_not_sufficient_condition_l62_62508


namespace cloth_selling_problem_l62_62660

theorem cloth_selling_problem (C S : ℝ) (H1 : S = 1.5 * C) (H2 : 10 * S = (30:S) * S - 30 * C) : true :=
by
  sorry

end cloth_selling_problem_l62_62660


namespace tent_ratio_l62_62527

-- Define variables for tents in different parts of the camp
variables (N E C S T : ℕ)

-- Given conditions
def northernmost (N : ℕ) := N = 100
def center (C N : ℕ) := C = 4 * N
def southern (S : ℕ) := S = 200
def total (T N C E S : ℕ) := T = N + C + E + S

-- Main theorem statement for the proof
theorem tent_ratio (N E C S T : ℕ) 
  (hn : northernmost N)
  (hc : center C N) 
  (hs : southern S)
  (ht : total T N C E S) :
  E / N = 2 :=
by sorry

end tent_ratio_l62_62527


namespace original_chairs_count_l62_62236

theorem original_chairs_count (n : ℕ) (m : ℕ) :
  (∀ k : ℕ, (k % 4 = 0 → k * (2 * n / 4) = k * (3 * n / 4) ) ∧ 
  (m = (4 / 2) * 15) ∧ (n = (4 * m / (2 * m)) - ((2 * m) / m)) ∧ 
  n + (n + 9) = 72) → n = 63 :=
by
  sorry

end original_chairs_count_l62_62236


namespace camel_humps_l62_62937

theorem camel_humps (N : ℕ) (h₁ : 1 ≤ N) (h₂ : N ≤ 99)
  (h₃ : ∀ S : Finset ℕ, S.card = 62 → 
                         (62 + S.count (λ n, n < 62 + N)) * 2 ≥ 100 + N) :
  (∃ n : ℕ, n = 72) :=
by
  sorry

end camel_humps_l62_62937


namespace solve_x_l62_62756

/-- Define the vectors a and b -/
def a : ℝ × ℝ := (3, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

/-- Define the dot product of two vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- The condition that (a - b) is orthogonal to a -/
def orthogonal_condition (x : ℝ) : Prop :=
  dot_product ((a.1 - b(x).1), (a.2 - b(x).2)) a = 0

/-- The main theorem to prove -/
theorem solve_x : ∃ x : ℝ, orthogonal_condition x ∧ x = 7 := 
by {
  -- Proof to be filled in
  sorry
}

end solve_x_l62_62756


namespace exponent_simplification_l62_62725

open Real

variable (a : ℝ) (h : 0 < a)

theorem exponent_simplification : (a ^ (1 / 2)) * (a ^ (2 / 3)) / (a ^ (1 / 6)) = a :=
by
  calc
  (a ^ (1 / 2)) * (a ^ (2 / 3)) / (a ^ (1 / 6))
      = (a ^ (1 / 2 + 2 / 3)) / (a ^ (1 / 6)) : by sorry
  ... = a ^ (7 / 6) / a ^ (1 / 6): by sorry
  ... = a ^ ((7 / 6) - (1 / 6)): by sorry
  ... = a ^ 1: by sorry
  ... = a: by sorry

end exponent_simplification_l62_62725


namespace sum_first_13_terms_l62_62486

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, m = n + k → a m = a n + k * (a 2 - a 1)

theorem sum_first_13_terms
  (a : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 4 + a 10 = 4) :
  (∑ i in Finset.range 13, a (i + 1)) = 26 :=
sorry

end sum_first_13_terms_l62_62486


namespace encryption_game_team_composition_l62_62636

theorem encryption_game_team_composition :
  ∃ (two_member_teams three_member_teams four_member_teams five_member_teams : ℕ),
    two_member_teams = 7 ∧
    three_member_teams = 20 ∧
    four_member_teams = 21 ∧
    five_member_teams = 2 ∧
    (two_member_teams + three_member_teams + four_member_teams + five_member_teams = 50) ∧
    (2 * two_member_teams + 3 * three_member_teams + 4 * four_member_teams + 5 * five_member_teams = 168) ∧
    three_member_teams = 20 ∧
    four_member_teams ≥ 21 ∧
    five_member_teams ≥ 1 :=
begin
  sorry,
end

end encryption_game_team_composition_l62_62636


namespace sum_of_four_consecutive_integers_divisible_by_two_l62_62157

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l62_62157


namespace sin_product_identity_l62_62864

theorem sin_product_identity : 
  (sin 70 * sin 70) * (sin 50 * sin 50) * (sin 10 * sin 10) = 1 / 64 :=
by
  sorry

end sin_product_identity_l62_62864


namespace cuboid_edge_length_l62_62888

theorem cuboid_edge_length (x : ℝ) (h1 : 5 * 6 * x = 120) : x = 4 :=
by
  sorry

end cuboid_edge_length_l62_62888


namespace number_equals_14_l62_62454

theorem number_equals_14 (n : ℕ) (h1 : 2^n - 2^(n-2) = 3 * 2^12) (h2 : n = 14) : n = 14 := 
by 
  sorry

end number_equals_14_l62_62454


namespace max_not_in_T_min_in_T_l62_62832

noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3) - (x + 1) / (x + 2)

def set_T : set ℝ := {y | ∃ x, x ≥ 0 ∧ y = g x}

def M : ℝ := 2
def m : ℝ := 1 / 3

theorem max_not_in_T : M ∉ set_T :=
by
  sorry

theorem min_in_T : m ∈ set_T :=
by
  sorry

end max_not_in_T_min_in_T_l62_62832


namespace triangle_DEF_EF_length_l62_62039

noncomputable def sin_degree : ℝ → ℝ := sorry
noncomputable def cos_degree : ℝ → ℝ := sorry

theorem triangle_DEF_EF_length (
  D E : ℝ)
  (h1 : cos_degree (3 * D - E) + sin_degree (D + E) = 2)
  (h2 : DE = 6)
  (h3 : ∠DEF = E) 
  (h4 : ∠EDF = D) :
  EF = 3 * sqrt (2 - sqrt 2) :=
sorry

end triangle_DEF_EF_length_l62_62039


namespace trapezoid_perimeter_l62_62795

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end trapezoid_perimeter_l62_62795


namespace karlson_max_candies_l62_62918

noncomputable def max_candies_29_minutes : ℕ :=
  406

theorem karlson_max_candies (n : ℕ) (h : n = 29) : 
  ∑ (k : ℕ) in finset.range (n - 1), (k * (n - k)) = max_candies_29_minutes :=
by
  sorry

end karlson_max_candies_l62_62918


namespace general_term_of_sequence_l62_62402

theorem general_term_of_sequence (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) :
  ∀ n, a n = (n^2 + n + 2) / 2 :=
by 
  sorry

end general_term_of_sequence_l62_62402


namespace fraction_is_one_fourth_l62_62564

-- Defining the numbers
def num1 : ℕ := 16
def num2 : ℕ := 8

-- Conditions
def difference_correct : Prop := num1 - num2 = 8
def sum_of_numbers : ℕ := num1 + num2
def fraction_of_sum (f : ℚ) : Prop := f * sum_of_numbers = 6

-- Theorem stating the fraction
theorem fraction_is_one_fourth (f : ℚ) (h1 : difference_correct) (h2 : fraction_of_sum f) : f = 1 / 4 :=
by {
  -- This will use the conditions and show that f = 1/4
  sorry
}

end fraction_is_one_fourth_l62_62564


namespace total_cost_ice_cream_l62_62080

noncomputable def price_Chocolate : ℝ := 2.50
noncomputable def price_Vanilla : ℝ := 2.00
noncomputable def price_Strawberry : ℝ := 2.25
noncomputable def price_Mint : ℝ := 2.20
noncomputable def price_WaffleCone : ℝ := 1.50
noncomputable def price_ChocolateChips : ℝ := 1.00
noncomputable def price_Fudge : ℝ := 1.25
noncomputable def price_WhippedCream : ℝ := 0.75

def scoops_Pierre : ℕ := 3  -- 2 scoops Chocolate + 1 scoop Mint
def scoops_Mother : ℕ := 4  -- 2 scoops Vanilla + 1 scoop Strawberry + 1 scoop Mint

noncomputable def price_Pierre_BeforeOffer : ℝ :=
  2 * price_Chocolate + price_Mint + price_WaffleCone + price_ChocolateChips

noncomputable def free_Pierre : ℝ := price_Mint -- Mint is the cheapest among Pierre's choices

noncomputable def price_Pierre_AfterOffer : ℝ := price_Pierre_BeforeOffer - free_Pierre

noncomputable def price_Mother_BeforeOffer : ℝ :=
  2 * price_Vanilla + price_Strawberry + price_Mint + price_WaffleCone + price_Fudge + price_WhippedCream

noncomputable def free_Mother : ℝ := price_Vanilla -- Vanilla is the cheapest among Mother's choices

noncomputable def price_Mother_AfterOffer : ℝ := price_Mother_BeforeOffer - free_Mother

noncomputable def total_BeforeDiscount : ℝ := price_Pierre_AfterOffer + price_Mother_AfterOffer

noncomputable def discount_Amount : ℝ := total_BeforeDiscount * 0.15

noncomputable def total_AfterDiscount : ℝ := total_BeforeDiscount - discount_Amount

theorem total_cost_ice_cream : total_AfterDiscount = 14.83 := by
  sorry


end total_cost_ice_cream_l62_62080


namespace alpha_condition_for_V_l62_62503

variable {f : ℝ → ℝ}

def continuous_on_Icc (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ Icc 0 1, continuous_on f (Icc (min x y) (max x y))

def differentiable_on_Ioo (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ Ioo 0 1, differentiable_on ℝ f (Ioo (min x y) (max x y))

def V := {f : ℝ → ℝ | continuous_on_Icc f ∧ differentiable_on_Ioo f ∧ f 0 = 0 ∧ f 1 = 1 }

noncomputable def alpha := 1 / (Real.exp 1 - 1)

theorem alpha_condition_for_V :
  ∀ f ∈ V, ∃ ξ ∈ Ioo 0 1, f ξ + alpha = deriv f ξ :=
sorry

end alpha_condition_for_V_l62_62503


namespace section_area_correct_l62_62644

-- Definitions of conditions based on the problem
def radius : ℝ := 5
def height : ℝ := 10
def arc_degree : ℝ := 150
def expected_area : ℝ := 48.295

-- Definition to be used in the proof
def cylinder_section_area (r h θ : ℝ) : ℝ :=
  2 * (1/2 * (2 * r * Real.sin (θ / 2)) * r * Real.sin θ)

-- Theorem statement
theorem section_area_correct : 
  (cylinder_section_area radius height arc_degree) = expected_area :=
by
  sorry

end section_area_correct_l62_62644


namespace cube_root_of_27_l62_62117

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end cube_root_of_27_l62_62117


namespace prime_factor_of_sum_of_four_consecutive_integers_l62_62169

-- Define four consecutive integers and their sum
def sum_four_consecutive_integers (n : ℤ) : ℤ := (n - 1) + n + (n + 1) + (n + 2)

-- The theorem states that 2 is a divisor of the sum of any four consecutive integers
theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p ∣ sum_four_consecutive_integers n :=
begin
  use 2,
  split,
  {
    apply Prime_two,
  },
  {
    unfold sum_four_consecutive_integers,
    norm_num,
    exact dvd.intro (2 * n + 1) rfl,
  },
end

end prime_factor_of_sum_of_four_consecutive_integers_l62_62169


namespace sio2_bond_is_polar_covalent_l62_62583

-- Definitions of electronegativity values as per the conditions
def chi_Si : Float := 1.90
def chi_O : Float := 3.44

-- Definition of a polar covalent bond based on electronegativity difference
def isPolarCovalentBond (chi_A chi_B : Float) : Prop :=
  let delta_chi := abs (chi_A - chi_B)
  0.5 < delta_chi ∧ delta_chi < 1.7

-- The theorem to prove that the Si-O bond in SiO2 is polar covalent.
theorem sio2_bond_is_polar_covalent :
  isPolarCovalentBond chi_Si chi_O :=
by
  -- Our goal is to establish that the bond is polar covalent.
  sorry

end sio2_bond_is_polar_covalent_l62_62583


namespace constant_term_in_expansion_l62_62103

/-!
# Math Proof Problem

Prove that the constant term in the expansion of $\left( x^{2}- \frac{1}{2x} \right)^{6}$ is $\frac{15}{16}$.

## Conditions:
- Binomial theorem applies to the expansion.
- Only the exponent of $x$ needs to be set to 0 to find the constant term.
- The binomial expansion involves binomial coefficients.
- Calculation of the coefficient involves combinatorial binomial coefficients.

## Goal:
- Prove that the constant term is equal to $\frac{15}{16}$.
-/

theorem constant_term_in_expansion :
  let x : ℝ := x in
  ∃ (c : ℚ), c = 15 / 16 ∧
  (∀ (x : ℝ), (\sum r in range 7, (binom 6 r) * (x^2)^(6-r) * ((-1/(2*x))^r) * (if 12-3*r = 0 then 1 else 0)) = c) :=
by { sorry }

end constant_term_in_expansion_l62_62103


namespace total_players_ground_l62_62478

-- Define the number of players for each type of sport
def c : ℕ := 10
def h : ℕ := 12
def f : ℕ := 16
def s : ℕ := 13

-- Statement of the problem to prove that the total number of players is 51
theorem total_players_ground : c + h + f + s = 51 :=
by
  -- proof will be added later
  sorry

end total_players_ground_l62_62478


namespace weight_of_dried_grapes_l62_62718

def fresh_grapes_initial_weight : ℝ := 25
def fresh_grapes_water_percentage : ℝ := 0.90
def dried_grapes_water_percentage : ℝ := 0.20

theorem weight_of_dried_grapes :
  (fresh_grapes_initial_weight * (1 - fresh_grapes_water_percentage)) /
  (1 - dried_grapes_water_percentage) = 3.125 := by
  -- Proof omitted
  sorry

end weight_of_dried_grapes_l62_62718


namespace compare_x1_x2_l62_62397

def f (x : ℝ) : ℝ := x * (Real.exp x - Real.exp (-x))

theorem compare_x1_x2 (x1 x2 : ℝ) (h : f x1 < f x2) : x1^2 < x2^2 :=
by
  -- Proof will be added here
  sorry

end compare_x1_x2_l62_62397


namespace solution_set_f_l62_62362

noncomputable def f : ℝ → ℝ :=
sorry

def is_solution_set (s : set ℝ) : Prop :=
∀ x, x ∈ s ↔ f (2^x - 2) < 2^x

theorem solution_set_f :
  (∀ x, x ∈ (0, +∞) → f x = 4 → (f' x > 1)) → 
  is_solution_set {x | 1 < x ∧ x < 2} :=
sorry

end solution_set_f_l62_62362


namespace find_constants_and_extreme_values_l62_62353

def f (x a b : ℝ) := x^3 + 3 * a * x^2 + b * x + a^2

theorem find_constants_and_extreme_values :
  let a := 2
  let b := 9 in
  (∀ (x : ℝ), a > 1 → (3 * x^2 + 6 * a * x + b = 0 → x = -1 → f x a b = 0)) ∧ 
  ((∃ x ∈ set.Icc (-4 : ℝ) (0 : ℝ), f x a b = 0) ∧ (∃ x ∈ set.Icc (-4 : ℝ) (0 : ℝ), f x a b = 4)) :=
by
  sorry

end find_constants_and_extreme_values_l62_62353


namespace sum_of_four_consecutive_integers_prime_factor_l62_62163

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l62_62163


namespace solution_set_inequality_l62_62017

theorem solution_set_inequality 
  (a b : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + 3 > 0 ↔ -1 < x ∧ x < 1/2) :
  ((-1:ℝ) < x ∧ x < 2) ↔ 3 * x^2 + b * x + a < 0 :=
by 
  -- Write the proof here
  sorry

end solution_set_inequality_l62_62017


namespace find_DF_l62_62485

def DEF_right_triangle (D E F : Type) := sorry

theorem find_DF (D E F : Type) 
  (h_triangle : DEF_right_triangle D E F)
  (h_sinE : sin E = (8 * sqrt 145) / 145)
  (h_DE : DE = sqrt 145) :
  DF = 8 := 
by 
  -- sorry to skip proof
  sorry

end find_DF_l62_62485


namespace max_k_value_condition_l62_62459

theorem max_k_value_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ k, k = 100 ∧ (∀ k < 100, ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c), 
   (k * a * b * c / (a + b + c) <= (a + b)^2 + (a + b + 4 * c)^2)) :=
sorry

end max_k_value_condition_l62_62459


namespace problem1_problem2_l62_62730

variable (c : ℝ) (p : ℝ) (n : ℕ) (a : ℕ → ℝ)

-- Assume the conditions
axiom c_gt_zero : c > 0
axiom p_gt_one : p > 1
axiom x_gt_neg_one : ∀ (x : ℝ), x > -1 → x ≠ 0 → (1 + x)^p > 1 + p * x

-- Define the sequence
axiom a_one_gt_c_root : a 0 > c ^ (1 / p)
axiom a_sequence : ∀ n, a (n + 1) = ((p - 1) / p) * a n + (c / p) * (a n)^(1 - p)

-- Problem 1: (1+x)^p > 1+px when x > -1 and x ≠ 0
theorem problem1 (x : ℝ) (hx₁ : x > -1) (hx₂ : x ≠ 0) : (1 + x)^p > 1 + p * x := 
  x_gt_neg_one x hx₁ hx₂

-- Problem 2: For the sequence, show that a_n > a_(n+1) > c^(1/p)
theorem problem2 : ∀ n, a n > a (n + 1) ∧ a (n + 1) > c^(1 / p) :=
sorry

end problem1_problem2_l62_62730


namespace prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62165

theorem prime_factor_of_sum_of_four_consecutive_integers_is_2 (n : ℤ) : 
  ∃ p : ℕ, prime p ∧ ∀ k : ℤ, (k-1) + k + (k+1) + (k+2) ∣ p :=
by
  -- Proof goes here.
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_is_2_l62_62165


namespace cube_root_of_27_l62_62107

theorem cube_root_of_27 : ∃ x : ℝ, x^3 = 27 ∧ x = 3 :=
by
  use 3
  split
  { norm_num }
  { rfl }

end cube_root_of_27_l62_62107


namespace count_odd_three_digit_numbers_l62_62343

open Finset

theorem count_odd_three_digit_numbers : 
  (card {n | ∃ (a b c : ℕ), a ∈ {1, 3, 5} ∧ b ∈ {0, 1, 2, 3, 4, 5} ∧ c ∈ {0, 1, 2, 3, 4, 5} ∧ 
       a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ n = 100 * b + 10 * c + a}) = 48 :=
begin
  -- Conditions have been set for the selection of digits to form a number
  sorry
end

end count_odd_three_digit_numbers_l62_62343


namespace interval_monotonically_decreasing_l62_62031

theorem interval_monotonically_decreasing :
  (∀ x : ℝ, (-2 < x ∧ x < 0) → (f'(x) < 0))
  → (f = λ x, x^2 * exp x)
  → (f'(x) = x * (x + 2) * exp x)
  → ∀ x : ℝ, (x ∈ set.Ioo (-1 : ℝ) (0 : ℝ)) ↔ (f'(x) < 0) :=
by
  sorry

end interval_monotonically_decreasing_l62_62031


namespace triangle_area_is_96_l62_62458

/-- Given a square with side length 8 and an overlapping area that is both three-quarters
    of the area of the square and half of the area of a triangle, prove the triangle's area is 96. -/
theorem triangle_area_is_96 (a : ℕ) (area_of_square : ℕ) (overlapping_area : ℕ) (area_of_triangle : ℕ) 
  (h1 : a = 8) 
  (h2 : area_of_square = a * a) 
  (h3 : overlapping_area = (3 * area_of_square) / 4) 
  (h4 : overlapping_area = area_of_triangle / 2) : 
  area_of_triangle = 96 := 
by 
  sorry

end triangle_area_is_96_l62_62458


namespace maximum_area_of_rectangular_playground_l62_62757

theorem maximum_area_of_rectangular_playground (P : ℕ) (A : ℕ) (h : P = 150) :
  ∃ (x y : ℕ), x + y = 75 ∧ A ≤ x * y ∧ A = 1406 :=
sorry

end maximum_area_of_rectangular_playground_l62_62757


namespace F_of_2_eq_5p5_l62_62264

def F (x : ℝ) : ℝ :=
  sqrt (abs (x + 2)) + (10 / Real.pi) * Real.arctan (sqrt (abs (x - 1))) + 1

theorem F_of_2_eq_5p5 : F 2 = 5.5 :=
by
  sorry

end F_of_2_eq_5p5_l62_62264


namespace poland_problem_l62_62827

-- Given definitions
variables {n : ℕ} (F : set (set ℕ))

-- Conditions in Lean
-- n must be greater than or equal to 6
-- F is a system of 3-element subsets of the set {1, 2, ..., n}
def valid_system (n : ℕ) (F : set (set ℕ)) : Prop :=
  n ≥ 6 ∧ (∀ (A ∈ F), A.card = 3 ∧ A.subset (finset.range (n+1))) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → (∃ B ⊆ F, {i, j}.subset B ∧ B.card ≥ ⌊(n / 3 : ℝ)⌋.to_nat - 1))

-- Main theorem statement
theorem poland_problem (n : ℕ) (F : set (set ℕ)) (h : valid_system n F) :
  ∃ (m : ℕ) (A : finset (set ℕ)), m ≥ 1 ∧ A.card = m ∧ (∀ a b ∈ A, a ≠ b → a ∩ b = ∅) ∧
  (A.bUnion id).card ≥ n - 5 :=
sorry

end poland_problem_l62_62827


namespace fraction_of_coins_1780_to_1799_l62_62872

theorem fraction_of_coins_1780_to_1799 :
  let num_states_1780_to_1789 := 12,
      num_states_1790_to_1799 := 5,
      total_states := 30 in
  (num_states_1780_to_1789 + num_states_1790_to_1799) / total_states = 17 / 30 :=
by
  sorry

end fraction_of_coins_1780_to_1799_l62_62872


namespace geometric_sequence_solution_l62_62386

theorem geometric_sequence_solution (x : ℝ) (h : ∃ r : ℝ, 12 * r = x ∧ x * r = 3) : x = 6 ∨ x = -6 := 
by
  sorry

end geometric_sequence_solution_l62_62386


namespace number_of_tangent_circles_l62_62225

theorem number_of_tangent_circles (R r : ℝ) (hR : R = 5) (hr : r = 2) :
  ∃ (n : ℕ), n = 12 ∧ (∀ (i : ℕ) (hi : i < n), tangent_circle (center (circle R)) (center (circle r i)) (circle r i)) ∧
     (∀ (i j : ℕ) (hi : i < n) (hj : j < n) (hij : i ≠ j), tangent_circle (circle r i) (circle r j)) :=
by
  sorry

end number_of_tangent_circles_l62_62225


namespace solution_set_of_inequality_l62_62909

theorem solution_set_of_inequality : {x : ℝ | x * (x - 2) ≤ 0} = set.Icc 0 2 := by
  sorry

end solution_set_of_inequality_l62_62909


namespace one_possible_value_m_l62_62735

open Real

def circle_eqn (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x = 0

def is_perpendicular (p a b : ℝ × ℝ) : Prop :=
  (fst a - fst p) * (fst b - fst p) + (snd a - snd p) * (snd b - snd p) = 0

theorem one_possible_value_m:
  ∀ (m : ℝ),
  let A := (1, m) in
  let B := (1, 2 * sqrt 5 - m) in
  (∃ P : ℝ × ℝ, circle_eqn (P.1) (P.2) ∧ is_perpendicular P A B) →
  m = sqrt 5 + 2 :=
by
  sorry

end one_possible_value_m_l62_62735


namespace imaginary_part_of_conjugate_l62_62349

variables (a b : ℝ) (z : ℂ)
#check Complex.conj

theorem imaginary_part_of_conjugate:
  (a - 2 * Complex.I = (b - Complex.I) * Complex.I) →
  (z = a + b * Complex.I) →
  Im (Complex.conj z) = 2 :=
begin
  intros h₁ h₂,
  sorry
end

end imaginary_part_of_conjugate_l62_62349


namespace exists_a_for_system_solution_l62_62321

theorem exists_a_for_system_solution (b : ℝ) :
  (∃ a x y : ℝ, y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ b ≤ 2*sqrt 2 + 1/4 :=
begin
  sorry
end

end exists_a_for_system_solution_l62_62321


namespace num_possible_values_l62_62932

variable (N : ℕ)

def is_valid_N (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 99 ∧
  (∀ (num_camels selected_camels : ℕ) (humps : ℕ),
    num_camels = 100 → 
    selected_camels = 62 →
    humps = 100 + N →
    selected_camels ≤ num_camels →
    selected_camels + min (selected_camels - 1) (N - (selected_camels - 1)) ≥ humps / 2)

theorem num_possible_values :
  (finset.Icc 1 24 ∪ finset.Icc 52 99).card = 72 :=
by sorry

end num_possible_values_l62_62932


namespace sufficient_condition_l62_62375

variables (a x : ℝ)

def f (x : ℝ) : ℝ := a^(x^2 + 2*x)

theorem sufficient_condition (h : a > 1) (hx : -1 < x ∧ x < 0) : f a x < 1 :=
by
  sorry

end sufficient_condition_l62_62375


namespace max_value_x_plus_one_over_x_l62_62913

theorem max_value_x_plus_one_over_x (n : ℕ) (sum_x sum_reciprocal : ℝ) (xs : Fin n → ℝ) (hpos : ∀ i, 0 < xs i)
  (hsum : ∑ i, xs i = sum_x) (hreciprocal : ∑ i, 1 / xs i = sum_reciprocal) :
  ∃ x, x ∈ (set.range xs) ∧ x + 1 / x = 2010 ∧
    2009 = n ∧ sum_x = 2010 ∧ sum_reciprocal = 2010 :=
sorry

end max_value_x_plus_one_over_x_l62_62913


namespace cosine_of_five_pi_over_three_l62_62612

theorem cosine_of_five_pi_over_three :
  Real.cos (5 * Real.pi / 3) = 1 / 2 :=
sorry

end cosine_of_five_pi_over_three_l62_62612


namespace prime_factors_of_four_consecutive_integers_sum_l62_62151

theorem prime_factors_of_four_consecutive_integers_sum : 
  ∃ p, Prime p ∧ ∀ n : ℤ, p ∣ ((n-2) + (n-1) + n + (n+1)) :=
by {
  use 2,
  split,
  { norm_num },
  { intro n,
    simp,
    exact dvd.intro (2 * n - 1) rfl }
}

end prime_factors_of_four_consecutive_integers_sum_l62_62151


namespace second_odd_integer_l62_62136

theorem second_odd_integer (n : ℤ) (h : (n - 2) + (n + 2) = 128) : n = 64 :=
by
  sorry

end second_odd_integer_l62_62136


namespace perimeter_ratio_l62_62246

-- Definition and conditions in a)
def original_rectangle_length : ℝ := 6 
def original_rectangle_width : ℝ := 8
def folded_length : ℝ := original_rectangle_length / 2
def folded_width : ℝ := original_rectangle_width
def small_rectangle_length : ℝ := folded_length / 2
def small_rectangle_width : ℝ := folded_width / 2

-- Calculation of perimeters for proof
def small_rectangle_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
def original_rectangle_perimeter : ℝ := 2 * (original_rectangle_length + original_rectangle_width)

-- Statement of the proof
theorem perimeter_ratio : (small_rectangle_perimeter / original_rectangle_perimeter) = (1 / 2) := by
  sorry

end perimeter_ratio_l62_62246


namespace total_cost_decrease_l62_62208

/-- Theorem: Given the conditions on the cost decreases and the initial relationship between paint
and canvas costs, the total cost for paint and canvas decreased by 56%. -/
theorem total_cost_decrease :
  ∀ (C P : ℝ), 
    P = 4 * C →
    let new_cost_canvas := 0.60 * C in
    let new_cost_paint := 0.40 * P in
    let total_original_cost := C + P in
    let total_new_cost := new_cost_canvas + new_cost_paint in
    100 * (total_original_cost - total_new_cost) / total_original_cost = 56 :=
by
  sorry

end total_cost_decrease_l62_62208


namespace calculate_total_usd_l62_62615

theorem calculate_total_usd : 
  let quarters := 23 * 0.25
  let dimes := 15 * 0.10
  let nickels := 17 * 0.05
  let pennies := 29 * 0.01
  let half_dollars := 6 * 0.50
  let dollar_coins := 10 * 1.00
  let total_usd := quarters + dimes + nickels + pennies + half_dollars + dollar_coins
  total_usd = 21.39 :=
by {
  let quarters := 23 * 0.25
  let dimes := 15 * 0.10
  let nickels := 17 * 0.05
  let pennies := 29 * 0.01
  let half_dollars := 6 * 0.50
  let dollar_coins := 10 * 1.00
  let total_usd := quarters + dimes + nickels + pennies + half_dollars + dollar_coins
  show total_usd = 21.39, from sorry
}

end calculate_total_usd_l62_62615


namespace assignments_in_batch_l62_62591

theorem assignments_in_batch :
  let x := 14 in
  6 * x = 84 ∧ 
  (2 * 6) + (8 * (x - 5)) = 6 * x := 
by
  sorry

end assignments_in_batch_l62_62591


namespace negative_x_y_l62_62451

theorem negative_x_y (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 :=
by
  sorry

end negative_x_y_l62_62451


namespace inequality1_inequality2_l62_62233

variable (f : ℝ → ℝ)
variable (t : ℝ)

-- Assume f is twice differentiable
axiom twice_differentiable (h : ∀ x, Differentiable ℝ f) :
  ∀ x, Differentiable ℝ (derivative f)

-- Assume f''(x) < 0
axiom condition1 : ∀ x, (derivative^[2]) f x < 0

-- Assume t ≥ 0
axiom condition2 : t ≥ 0

-- First inequality
theorem inequality1 :
  f 0 + (derivative f t) * t ≤ f t ∧ f t ≤ f 0 + (derivative f 0) * t := sorry

-- Second inequality
theorem inequality2 :
  (f 0 * t + f t * t) / 2 ≤ ∫ u in 0..t, f u ∧ ∫ u in 0..t, f u ≤ f 0 * t + (derivative f 0) * t^2 / 2 := sorry

end inequality1_inequality2_l62_62233


namespace six_points_four_segments_possible_l62_62801

theorem six_points_four_segments_possible :
  ∃ (G : SimpleGraph (Fin 6)), (G.degree = λ_, 4) ∧ G.isPlanar :=
sorry

end six_points_four_segments_possible_l62_62801


namespace initial_juggling_objects_l62_62815

theorem initial_juggling_objects (x : ℕ) : (∀ i : ℕ, i = 5 → x + 2*i = 13) → x = 3 :=
by 
  intro h
  sorry

end initial_juggling_objects_l62_62815


namespace solve_for_b_l62_62318

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end solve_for_b_l62_62318


namespace even_function_a_zero_l62_62434

section

variable (a : ℝ)

def f (x : ℝ) := (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

theorem even_function_a_zero : ∀ x : ℝ, f a x = f a (-x) → a = 0 := by
  sorry

end

end even_function_a_zero_l62_62434


namespace exists_integer_between_sqrt2_and_sqrt11_l62_62723

theorem exists_integer_between_sqrt2_and_sqrt11 : ∃ (m : ℤ), (sqrt 2: ℝ) < (m: ℝ) ∧ (m: ℝ) < (sqrt 11: ℝ) :=
by
  sorry

end exists_integer_between_sqrt2_and_sqrt11_l62_62723
