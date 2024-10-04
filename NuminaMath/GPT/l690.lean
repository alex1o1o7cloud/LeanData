import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Calculus.Maximal
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Triangle
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Points
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace two_digit_primes_with_digit_sum_ten_l690_690796

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690796


namespace num_two_digit_primes_with_digit_sum_10_l690_690824

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690824


namespace airplane_seats_business_class_l690_690010

theorem airplane_seats_business_class :
  ∀ (E F B occupiedE occupiedF : ℕ),
  E = 50 ∧ F = 10 ∧ occupiedE = E / 2 ∧ (occupiedF = 3) ∧ (occupiedB = (occupiedE - occupiedF)) ∧ (unoccupiedB = 8) ∧ 
  occupiedB + unoccupiedB = B → B = 30 :=
by
  intros E F B occupiedE occupiedF
  assume h
  cases h with h_E h_rest
  cases h_rest with h_F h_rest
  cases h_rest with h_occupiedE h_rest
  cases h_rest with h_occupiedF h_rest
  cases h_rest with h_occupiedB h_rest
  cases h_rest with h_unoccupiedB h_total
  sorry

end airplane_seats_business_class_l690_690010


namespace cheese_initial_weight_l690_690293

theorem cheese_initial_weight (initial_weight : ℕ) 
  (take_bites : ℕ → ℕ × ℕ) 
  (final_state : ℕ × ℕ) 
  (final_weight : ℕ) :
  let final_state := (20, 20)
  ∃ initial_weight, take_bites 3 (initial_weight, 0) = final_state ∧ 
  initial_weight = 680 := 
sorry

end cheese_initial_weight_l690_690293


namespace smallest_c_ineq_l690_690561

noncomputable def smallest_c {d : ℕ → ℕ} (h_d : ∀ n > 0, d n ≤ d n + 1) := Real.sqrt 3

theorem smallest_c_ineq (d : ℕ → ℕ) (h_d : ∀ n > 0, (d n) ≤ d n + 1) :
  ∀ n : ℕ, n > 0 → d n ≤ smallest_c h_d * (Real.sqrt n) :=
sorry

end smallest_c_ineq_l690_690561


namespace sums_equal_l690_690233

variable {Person : Type} [DecidableEq Person]
variable (row : List Person)
variable (isGirl : Person → Bool)
variable (isBoy : Person → Bool)

def G : ℕ := (row.enum.filter (λ (p : ℕ × Person), isGirl p.snd)).sum (λ (p : ℕ × Person), row.take p.fst |>.countp isBoy)
def B : ℕ := (row.enum.filter (λ (p : ℕ × Person), isBoy p.snd)).sum (λ (p : ℕ × Person), row.drop (p.fst + 1) |>.countp isGirl)

theorem sums_equal : G row isGirl isBoy = B row isGirl isBoy := 
sorry

end sums_equal_l690_690233


namespace ellipse_eccentricity_bound_l690_690076

theorem ellipse_eccentricity_bound (a b : ℝ) (hx : a > b) (hy : b > 0)
    (h : ∃ (P : ℝ × ℝ), P ∈ (λ x y: ℝ, x^2 / a^2 + y^2 / b^2 = 1) ∧ 
                       (let ε := sqrt (a^2 - b^2) / a, 
                        ∀ A : ℝ × ℝ, 
                        A = (a / ε, 0) → 
                        (∃ F : ℝ × ℝ, F = (a * ε, 0) ∧ 
                        F = ((P.1 + A.1) / 2, P.2 / 2))))
    : 1 / 2 = sqrt (a^2 - b^2) / a ∧ 0 < sqrt (a^2 - b^2) / a < 1 := 
sorry

end ellipse_eccentricity_bound_l690_690076


namespace solve_arctan_eq_l690_690232

theorem solve_arctan_eq (x : ℝ) (h : arctan (1/x) + arctan (1/(x^2)) + arctan (1/(x^3)) = π / 4) : x = 1 :=
sorry

end solve_arctan_eq_l690_690232


namespace count_two_digit_primes_with_digit_sum_10_l690_690885

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690885


namespace rectangle_perimeter_l690_690350

variable (a b : ℕ)

theorem rectangle_perimeter (h1 : a ≠ b) (h2 : ab = 8 * (a + b)) : 
  2 * (a + b) = 66 := 
sorry

end rectangle_perimeter_l690_690350


namespace second_number_in_sequence_is_nine_l690_690599

theorem second_number_in_sequence_is_nine :
  (8 + (∃ n, (8 + n = 16)) ↔ 9) :=
by
  -- Given conditions
  let (six_plus_seven_eq_twelve : 6 + 7 = 12) := sorry
  let (five_plus_six_eq_ten : 5 + 6 = 10) := sorry
  let (seven_plus_eight_eq_fourteen : 7 + 8 = 14) := sorry
  let (three_plus_three_eq_five : 3 + 3 = 5) := sorry
  -- Prove the main statement
  have num := 9
  use num
  have calculation := 8 + num
  have expected_sum := 8 + 9
  have given_sum := 16
  show (8 + num = given_sum ↔ num = 9), by {
    -- This is the main goal we need to prove
    sorry
  }

end second_number_in_sequence_is_nine_l690_690599


namespace determine_roots_related_to_k_l690_690032

noncomputable def roots_of_quadratic := 
  { a' b' c' d' : ℂ // 
    ∃ k' : ℂ, k' ≠ 0 ∧ (a' * k'^2 + b' * k' + c' = 0) ∧ 
                          (b' * k'^2 + c' * k' + d' = 0) }

theorem determine_roots_related_to_k
  {a' b' c' d' : ℂ} (h : roots_of_quadratic ⟨a', b', c', d'⟩) : 
  ∃ k' : ℂ, k' = 1 ∨ k' = (-1 + complex.I * real.sqrt 3) / 2 ∨ 
                k' = (-1 - complex.I * real.sqrt 3) / 2 :=
begin
  sorry
end

end determine_roots_related_to_k_l690_690032


namespace range_q_l690_690613

def q (x : ℝ ) : ℝ := x^4 + 4 * x^2 + 4

theorem range_q :
  (∀ y, ∃ x, 0 ≤ x ∧ q x = y ↔ y ∈ Set.Ici 4) :=
sorry

end range_q_l690_690613


namespace solve_for_c_solve_for_b_l690_690110

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := -x^3 - b * x^2 - 5 * c * x

theorem solve_for_c (b c : ℝ) (h_monotone_dec : ∀ x : ℝ, x ≤ 0 → f'(x) ≤ 0)
  (h_monotone_inc : ∀ x : ℝ, 0 ≤ x → x ≤ 6 → f'(x) ≥ 0) : c = 0 := 
by
  sorry

theorem solve_for_b (b c : ℝ) (h_c_zero : c = 0) 
  (h_monotone_inc : ∀ x : ℝ, 0 ≤ x → x ≤ 6 → f'(x) ≥ 0) : b > 0 :=
by
  sorry

end solve_for_c_solve_for_b_l690_690110


namespace two_digit_prime_sum_digits_10_count_l690_690937

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690937


namespace num_two_digit_primes_with_digit_sum_10_l690_690832

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690832


namespace part1_solution_set_part2_m_range_l690_690472

section Part1

def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1_solution_set (x : ℝ) : f(x) ≥ 1 ↔ x ≥ 1 :=
sorry

end Part1

section Part2

theorem part2_m_range (m : ℝ) (x : ℝ) :
  abs (m - 2) ≥ abs (abs (x + 1) - abs (x - 2)) ↔ m ≥ 5 ∨ m ≤ -1 :=
sorry

end Part2

end part1_solution_set_part2_m_range_l690_690472


namespace number_of_arrangements_l690_690072

-- Definitions for clarity
inductive Student
| A | B | C | D | E

-- Condition 1: Student A cannot be at the far left
def A_not_far_left (arrangement : List Student) : Prop :=
  arrangement.head? ≠ some Student.A

-- Condition 2: Student B must be to the left of Student C
def B_left_of_C (arrangement : List Student) : Prop :=
  ∃ i j, arrangement.get? i = some Student.B ∧ arrangement.get? j = some Student.C ∧ i < j

-- Theorem: There are 48 valid arrangements of the students
theorem number_of_arrangements : 
  ∃ arrangement : List Student, A_not_far_left arrangement ∧ B_left_of_C arrangement ∧ arrangement.perm [Student.A, Student.B, Student.C, Student.D, Student.E] := 
  sorry

end number_of_arrangements_l690_690072


namespace omega_range_l690_690086

theorem omega_range (ω : ℝ) (h_pos : ω > 0)
  (h_mono : ∀ x y : ℝ, π / 2 < x ∧ x < y ∧ y < π → 
                      sin (ω * y + π / 4) < sin (ω * x + π / 4)) :
  ω ∈ set.Icc (1 / 2 : ℝ) (5 / 4 : ℝ) := 
sorry

end omega_range_l690_690086


namespace volume_KPQR_l690_690525

-- Defining the geometric setup
def tetrahedron_side : ℝ := Real.sqrt 2
def vector_AP (A B : ℝ^3) : ℝ^3 := (1/2) • (B - A)
def vector_AQ (A C : ℝ^3) : ℝ^3 := (1/3) • (C - A)
def vector_AR (A D : ℝ^3) : ℝ^3 := (1/4) • (D - A)
def centroid_K (B C D : ℝ^3) : ℝ^3 := (1/3) • (B + C + D)

-- Defining the volume calculation function
noncomputable def volume_tetrahedron (a b c d : ℝ^3) : ℝ :=
  (1/6) * Real.abs ((b - a).cross (c - a) ⋅ (d - a))

-- Hypothetical points A, B, C, D
variables (A B C D P Q R K : ℝ^3)

-- The main theorem statement
theorem volume_KPQR : 
  K = centroid_K B C D →
  vector_AP A B = P - A →
  vector_AQ A C = Q - A →
  vector_AR A D = R - A →
  volume_tetrahedron K P Q R = 1/36 :=
by
  -- Definitions of points and volume calculations will go here in proof.
  sorry

end volume_KPQR_l690_690525


namespace two_FA_inequality_l690_690723

variable {A B C D F : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space F]
variables [add_lt_add : ∀ {a b c d : real}, a < b → c < d → a + c < b + d]

-- Definitions for quadrilateral ABCD with angle A obtuse and F midpoint of BC
def quadrilateral (A B C D : Type) : Prop := sorry
def obtuse_angle (A : Type) : Prop := sorry
def midpoint (F : Type) (B C: Type) : Prop := sorry

-- The condition: quadrilateral ABCD with obtuse angle A and midpoint F of BC
variables (h1 : quadrilateral A B C D)
          (h2 : obtuse_angle A)
          (h3 : midpoint F B C)

-- Theorem to prove: 2FA <= BD + CD
theorem two_FA_inequality (h1 : quadrilateral A B C D)
                          (h2 : obtuse_angle A)
                          (h3 : midpoint F B C) :
  2 * (dist F A) ≤ (dist B D) + (dist C D) :=
sorry

end two_FA_inequality_l690_690723


namespace tan_of_sum_l690_690451

variable (α : Real)
hypothesis (h : Real.tan α = 1 / 2)

theorem tan_of_sum : Real.tan (α + 3 * Real.pi / 4) = -1 / 3 :=
by 
  sorry

end tan_of_sum_l690_690451


namespace problem_f_neg2_f_1_l690_690466

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2^(-x) else Real.sqrt x

theorem problem_f_neg2_f_1 :
  f (-2) + f (1) = 5 :=
by
  sorry

end problem_f_neg2_f_1_l690_690466


namespace minimum_value_l690_690063

noncomputable def y (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem minimum_value :
  ∃ (c : ℝ), c ∈ Set.Icc 0 Real.pi ∧ ∀ x ∈ Set.Icc 0 Real.pi, y(x) ≥ y(c) ∧ y(c) = (5 * Real.pi / 6) - Real.sqrt 3 := sorry

end minimum_value_l690_690063


namespace find_a_from_area_l690_690512

noncomputable def tangent_triangle_area := 
  ∀ (a : ℝ), 
  let y := λ x : ℝ, x^(-1/2) in
  let y' := λ x : ℝ, -1/2 * x^(-3/2) in
  let tangent := λ x : ℝ, y a + y' a * (x - a) in
  let x_intercept := 3 * a in
  let y_intercept := 3/2 * a^(-1/2) in
  let area := (1/2) * x_intercept * y_intercept in
  area = 18 → a = 64

theorem find_a_from_area (h : tangent_triangle_area) : True := sorry

end find_a_from_area_l690_690512


namespace A_beats_C_by_28_meters_l690_690522

variable (T T_B : ℝ)
variables (V_A V_B V_C : ℝ)

noncomputable def speed_A : ℝ := 100 / T
noncomputable def speed_B_race1 : ℝ := 75 / T
noncomputable def speed_B_race2 : ℝ := 100 / T_B
noncomputable def speed_C : ℝ := 96 / T_B
noncomputable def T_B_value := (100 / 75) * T
noncomputable def distance_C_in_T : ℝ := (speed_C * T)

theorem A_beats_C_by_28_meters (h1 : V_A = speed_A) (h2 : V_B = speed_B_race1) (h3 : V_B = speed_B_race2) (h4 : T_B = T_B_value) (h5 : V_C = speed_C) : 100 - distance_C_in_T = 28 :=
by 
  unfold speed_A speed_B_race1 speed_B_race2 speed_C T_B_value distance_C_in_T
  sorry

end A_beats_C_by_28_meters_l690_690522


namespace ratio_EG_FH_l690_690220

theorem ratio_EG_FH (EF FG EH : ℝ) (hEF : EF = 3) (hFG : FG = 7) (hEH : EH = 20) :
  (EF + FG) / (EH - EF) = 10 / 17 :=
by
  sorry

end ratio_EG_FH_l690_690220


namespace expansion_coeff_x_cubed_l690_690516

-- Define the primary condition
def sum_coeffs_condition (n : ℕ) := (3 : ℕ) ^ n = 32

-- Define the main expression
def binom_expansion_term (n r : ℕ) : ℤ := (-1) ^ r * (3 ^ (n - r)) * (Nat.binomial n r) * (x : ℕ ^ (n - 2 * r))

-- Define the target term we are interested in
def target_term_coefficient (n r : ℕ) (target_power : ℕ) :=
  n = 5 ∧ r = 1 ∧ target_power = 3

theorem expansion_coeff_x_cubed :
  Π (n r : ℕ), sum_coeffs_condition n → target_term_coefficient n r (5 - 2 * r) →
  binom_expansion_term n r = -405 :=
by
  intros n r h_n h_term
  -- Proof steps will go here.
  -- Adding sorry to skip the proof as instructed.
  sorry

end expansion_coeff_x_cubed_l690_690516


namespace painting_price_percentage_l690_690675

theorem painting_price_percentage (P: ℝ) (P1: ℝ) (P2: ℝ) :
  (P1 = P * 1.25) ∧ (P2 = P1 * 0.85) → P2 = P * 1.0625 :=
by
  intros h,
  have h1 : P1 = P * 1.25, from h.left,
  have h2 : P2 = P1 * 0.85, from h.right,
  rw [h1, mul_assoc] at h2,
  rw [mul_comm 0.85, ←mul_assoc] at h2,
  norm_num at h2,
  exact h2,
  sorry

end painting_price_percentage_l690_690675


namespace min_chord_length_l690_690347

-- Definitions of the problem conditions
def circle_center : ℝ × ℝ := (2, 3)
def circle_radius : ℝ := 3
def point_P : ℝ × ℝ := (1, 1)

-- The mathematical statement to prove
theorem min_chord_length : 
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ ((B.1 - 2)^2 + (B.2 - 3)^2 = 9) ∧ 
  ((A.1 - 1) / (B.1 - 1) = (A.2 - 1) / (B.2 - 1)) → 
  dist A B ≥ 4 := 
sorry

end min_chord_length_l690_690347


namespace new_paint_intensity_l690_690600

theorem new_paint_intensity (V : ℝ) (h1 : V > 0) :
    let initial_intensity := 0.5
    let replaced_fraction := 0.4
    let replaced_intensity := 0.25
    let new_intensity := (0.3 + 0.1 * replaced_fraction)  -- derived from (0.6 * 0.5 + 0.4 * 0.25)
    new_intensity = 0.4 :=
by
    sorry

end new_paint_intensity_l690_690600


namespace count_two_digit_primes_with_digit_sum_10_l690_690989

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690989


namespace max_strong_boys_l690_690630

section StrongBoys

-- Define the conditions for heights and weights
variables (Height Weight : Fin 10 → ℕ)
variables (h := ∀ (i j : Fin 10), i < j ↔ Height i < Height j)
variables (w := ∀ (i j : Fin 10), i < j ↔ Weight i > Weight j)

-- Definition of "not inferior" relation
def not_inferior_to (i j : Fin 10) : Prop :=
  (Weight i > Weight j) ∨ (Height i > Height j)

-- Definition of a strong boy
def is_strong_boy (i : Fin 10) : Prop :=
  ∀ (j : Fin 10), i ≠ j → not_inferior_to i j

-- Prove the number of strong boys is 10
theorem max_strong_boys : ∃ (count : ℕ), count = 10 ∧ (∀ (i : Fin 10), is_strong_boy Height Weight i) :=
  sorry

end StrongBoys

end max_strong_boys_l690_690630


namespace equation_line_AB_length_median_AM_equation_altitude_A_l690_690112

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def line (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := slope A B
  (k, -1, k * (A.1 - B.1))

noncomputable def altitude (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := slope B C
  let perp_k := -1 / k
  (perp_k, -1, A.2 - perp_k * A.1)

/- Question 1 -/
theorem equation_line_AB :
  line (-1, 5) (-2, -1) = (6, -1, 11) := 
sorry

/- Question 2 -/
theorem length_median_AM :
  length (-1, 5) (midpoint (-2, -1) (4, 3)) = 2 * real.sqrt 5 := 
sorry

/- Question 3 -/
theorem equation_altitude_A :
  altitude (-1, 5) (-2, -1) (4, 3) = (1, 6, -22) := 
sorry

end equation_line_AB_length_median_AM_equation_altitude_A_l690_690112


namespace set_inter_complement_l690_690208

def U := {-2, -1, 0, 1, 2}
def A := {-2, 0, 1}
def B := {-1, 0, 2}

theorem set_inter_complement :
  A ∩ (U \ B) = {-2, 1} := 
by sorry

end set_inter_complement_l690_690208


namespace value_of_N_l690_690272

theorem value_of_N (a b c N : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = N)
  (h3 : 8 * b = N)
  (h4 : c / 8 = N) :
  N = 960 / 73 :=
by
  sorry

end value_of_N_l690_690272


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690973

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690973


namespace integral_quarter_circle_l690_690046

open Real
open IntervalIntegral

theorem integral_quarter_circle :
  (∫ x in 0..1, sqrt (1 - x^2)) = π / 4 :=
by
  sorry

end integral_quarter_circle_l690_690046


namespace largest_sum_of_digits_l690_690506

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) 
  (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10)
  (h₃ : 0.abc = (1 : ℚ) / y)
  (h₄ : y > 0 ∧ y ≤ 16) : a + b + c ≤ 8 :=
by sorry

end largest_sum_of_digits_l690_690506


namespace sum_of_digits_of_N_eq_14_l690_690365

theorem sum_of_digits_of_N_eq_14 (N : ℕ) (h1 : ∑ i in Finset.range (N + 1), i = 3003) : (digits 10 N).sum = 14 :=
sorry

end sum_of_digits_of_N_eq_14_l690_690365


namespace triangle_pieces_count_l690_690304

theorem triangle_pieces_count :
  (∃ (a b c : ℕ), (a + b + c = 22) ∧ (a ≤ b) ∧ (b ≤ c) ∧ (a + b > c) ∧ 
  (finset.card { (x, y, z) | 
    x + y + z = 22 ∧ x ≤ y ∧ y ≤ z ∧ x + y > z } = 10)) :=
begin
  sorry
end

end triangle_pieces_count_l690_690304


namespace find_C_when_F_10_l690_690143

theorem find_C_when_F_10 : (∃ C : ℚ, ∀ F : ℚ, F = 10 → F = (9 / 5 : ℚ) * C + 32 → C = -110 / 9) :=
by
  sorry

end find_C_when_F_10_l690_690143


namespace number_of_trees_planted_l690_690279

def current_trees : ℕ := 34
def final_trees : ℕ := 83
def planted_trees : ℕ := final_trees - current_trees

theorem number_of_trees_planted : planted_trees = 49 :=
by
  -- proof goes here, but it is skipped for now
  sorry

end number_of_trees_planted_l690_690279


namespace cheese_original_weight_l690_690299

theorem cheese_original_weight (final_weight : ℕ)
    (after_bites : ∀(n: ℕ) (w: ℕ), n ≤ 3 → (n = 0 → w = final_weight) ∧
                  (n = 1 → w = 40) ∧
                  (n = 2 → w = 100 ∧ 60) ∧
                  (n = 3 → w = 260 ∧ 160) ∧
                  (n = 4 → w = 420 ∧ 260)) :
    final_weight = 680 := by
    sorry

end cheese_original_weight_l690_690299


namespace count_two_digit_primes_with_digit_sum_10_l690_690988

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690988


namespace fox_alice_two_digit_number_l690_690081

theorem fox_alice_two_digit_number : 
    {n : ℕ | 10 ≤ n ∧ n < 100 ∧ 
                  ( (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ (n % 5 ≠ 0 ∨ n % 6 ≠ 0)) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 6 ≠ 0) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0 ∧ n % 6 ≠ 0) ∨
                    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0)
                 )}.card = 8 := 
by 
  sorry

end fox_alice_two_digit_number_l690_690081


namespace local_minimum_interval_l690_690156

theorem local_minimum_interval 
  (b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : f = λ x, x^3 - 3 * b * x + 3 * b) 
  (has_local_minimum : ∃ x ∈ Ioo 0 1, ∀ y ∈ Ioo 0 1, f x ≤ f y) 
  : 0 < b ∧ b < 1 :=
sorry

end local_minimum_interval_l690_690156


namespace smallest_n_satisfying_conditions_l690_690708

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n > 0 ∧ ¬ (n % 2 = 0) ∧ ¬ (n % 3 = 0) ∧
  (∀ a b : ℕ, |2^a - 3^b| ≠ n) ∧ n = 35 := sorry

end smallest_n_satisfying_conditions_l690_690708


namespace count_two_digit_primes_with_digit_sum_10_l690_690993

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690993


namespace earbuds_cost_before_tax_l690_690549

-- Define the conditions
variable (C : ℝ) -- The cost before tax
variable (taxRate : ℝ := 0.15)
variable (totalPaid : ℝ := 230)

-- Define the main question in Lean
theorem earbuds_cost_before_tax : C + taxRate * C = totalPaid → C = 200 :=
by
  sorry

end earbuds_cost_before_tax_l690_690549


namespace artifacts_per_wing_l690_690349

theorem artifacts_per_wing
  (total_wings : ℕ)
  (num_paintings : ℕ)
  (num_artifacts : ℕ)
  (painting_wings : ℕ)
  (large_paintings_wings : ℕ)
  (small_paintings_wings : ℕ)
  (small_paintings_per_wing : ℕ)
  (artifact_wings : ℕ)
  (wings_division : total_wings = painting_wings + artifact_wings)
  (paintings_division : painting_wings = large_paintings_wings + small_paintings_wings)
  (num_large_paintings : large_paintings_wings = 2)
  (num_small_paintings : small_paintings_wings * small_paintings_per_wing = num_paintings - large_paintings_wings)
  (num_artifact_calc : num_artifacts = 8 * num_paintings)
  (artifact_wings_div : artifact_wings = total_wings - painting_wings)
  (artifact_calc : num_artifacts / artifact_wings = 66) :
  num_artifacts / artifact_wings = 66 := 
by
  sorry

end artifacts_per_wing_l690_690349


namespace num_two_digit_primes_with_digit_sum_10_l690_690778

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690778


namespace functional_equation_solution_l690_690743

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y, f (f (f x)) + f (f y) = f y + x) → (∀ x, f x = x) :=
by
  intros f h x
  -- Proof goes here
  sorry

end functional_equation_solution_l690_690743


namespace probability_divisible_by_seven_l690_690353

open BigOperators

noncomputable def nums : Finset ℕ := (Finset.range 900).map ⟨(λ n => n + 100), Nat.add_left_injective 100⟩

def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

def count_divisibles : ℕ :=
  (nums.filter is_divisible_by_seven).card

def probability : ℚ := count_divisibles / nums.card

theorem probability_divisible_by_seven :
  probability = 32 / 225 :=
by
  -- Proof steps will go here
  sorry

end probability_divisible_by_seven_l690_690353


namespace count_two_digit_prime_with_digit_sum_10_l690_690872

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690872


namespace fries_to_popcorn_ratio_l690_690551

-- Define the time spent on the first movie, second movie, and making popcorn
def first_movie_time : ℕ := 90
def second_movie_time : ℕ := 120
def popcorn_time : ℕ := 10
def total_time : ℕ := 240

-- Calculate the time spent watching movies
def movies_time : ℕ := first_movie_time + second_movie_time

-- Calculate the time spent making fries
def fries_time : ℕ := total_time - movies_time - popcorn_time

-- Express the given conditions and required ratio in Lean
theorem fries_to_popcorn_ratio : fries_time / popcorn_time = 2 :=
by
  have h_fries_time : fries_time = 20 := by
    unfold fries_time movies_time total_time first_movie_time second_movie_time popcorn_time
    norm_num,
  have h_popcorn_time : popcorn_time = 10 := by
    unfold popcorn_time
    norm_num,
  rw [h_fries_time, h_popcorn_time],
  norm_num,
  sorry

end fries_to_popcorn_ratio_l690_690551


namespace two_digit_primes_with_digit_sum_10_count_l690_690905

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690905


namespace find_other_number_l690_690262

def a : ℝ := 0.5
def d : ℝ := 0.16666666666666669
def b : ℝ := 0.3333333333333333

theorem find_other_number : a - d = b := by
  sorry

end find_other_number_l690_690262


namespace f_3_equals_1000_l690_690104

-- Define the function property f(lg x) = x
axiom f : ℝ → ℝ
axiom lg : ℝ → ℝ -- log function
axiom f_property : ∀ x : ℝ, f (lg x) = x

-- Prove that f(3) = 10^3
theorem f_3_equals_1000 : f 3 = 10^3 :=
by 
  -- Sorry to skip the proof
  sorry

end f_3_equals_1000_l690_690104


namespace sum_xyz_l690_690636

theorem sum_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + 2 * (y - 1) * (z - 1) = 85)
  (h2 : y^2 + 2 * (z - 1) * (x - 1) = 84)
  (h3 : z^2 + 2 * (x - 1) * (y - 1) = 89) :
  x + y + z = 18 := 
by
  sorry

end sum_xyz_l690_690636


namespace harry_travel_time_l690_690487

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l690_690487


namespace smallest_x_no_triangle_l690_690363

def triangle_inequality_violated (a b c : ℝ) : Prop :=
a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_x_no_triangle (x : ℕ) (h : ∀ x, triangle_inequality_violated (7 - x : ℝ) (24 - x : ℝ) (26 - x : ℝ)) : x = 5 :=
sorry

end smallest_x_no_triangle_l690_690363


namespace sqrt_xyz_sum_l690_690199

noncomputable theory

variables {x y z : ℝ}

theorem sqrt_xyz_sum {x y z : ℝ} (h1 : y + z = 13) (h2 : z + x = 14) (h3 : x + y = 15) :
  Real.sqrt (x * y * z * (x + y + z)) = 84 :=
sorry

end sqrt_xyz_sum_l690_690199


namespace num_two_digit_primes_with_digit_sum_10_l690_690782

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690782


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690923

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690923


namespace simplify_fraction1_simplify_fraction2_simplify_fraction3_l690_690597

-- Statement 1: \frac{a^4 - b^4}{a^2 - b^2} = a^2 + b^2
theorem simplify_fraction1 (a b : ℝ) : 
  (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := 
  sorry

-- Statement 2: \frac{a^2 + a - 90}{a^2 + 2a - 99} = \frac{a + 10}{a + 11}
theorem simplify_fraction2 (a : ℝ) : 
  (a^2 + a - 90) / (a^2 + 2a - 99) = (a + 10) / (a + 11) := 
  sorry

-- Statement 3: \frac{2a^3 - 7a^2 + 2a + 3}{2a^3 - 9a^2 + 70a - 3} = \frac{2a + 1}{2a - 1}
theorem simplify_fraction3 (a : ℝ) : 
  (2a^3 - 7a^2 + 2a + 3) / (2a^3 - 9a^2 + 70a - 3) = (2a + 1) / (2a - 1) := 
  sorry

end simplify_fraction1_simplify_fraction2_simplify_fraction3_l690_690597


namespace count_two_digit_primes_with_digit_sum_10_l690_690854

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690854


namespace cost_of_dog_l690_690577

-- Given conditions
def dollars_misha_has : ℕ := 34
def dollars_misha_needs_earn : ℕ := 13

-- Formal statement of the mathematic proof
theorem cost_of_dog : dollars_misha_has + dollars_misha_needs_earn = 47 := by
  sorry

end cost_of_dog_l690_690577


namespace two_digit_prime_sum_to_ten_count_l690_690847

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690847


namespace volume_hemisphere_correct_l690_690015

def total_volume : ℝ := 10976
def num_hemispheres : ℝ := 2744
def volume_hemisphere : ℝ := total_volume / num_hemispheres

theorem volume_hemisphere_correct : volume_hemisphere = 4 := by
  sorry

end volume_hemisphere_correct_l690_690015


namespace count_two_digit_prime_with_digit_sum_10_l690_690879

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690879


namespace sum_of_digits_of_n_l690_690634

theorem sum_of_digits_of_n 
  (n : ℕ) 
  (h1 : (n + 1)! + (n + 3)! = n! * 840) : 
  (Nat.digits 10 n).sum = 7 := 
by
  sorry

end sum_of_digits_of_n_l690_690634


namespace exists_divisible_by_2011_l690_690595

def a (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc + 10 ^ i) 0

theorem exists_divisible_by_2011 : ∃ n, 1 ≤ n ∧ n ≤ 2011 ∧ 2011 ∣ a n := by
  sorry

end exists_divisible_by_2011_l690_690595


namespace factor_t_squared_minus_144_l690_690411

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l690_690411


namespace two_digit_prime_sum_digits_10_count_l690_690928

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690928


namespace exists_three_irrationals_l690_690669

theorem exists_three_irrationals
    (x1 x2 x3 : ℝ)
    (h1 : ¬ ∃ q : ℚ, x1 = q)
    (h2 : ¬ ∃ q : ℚ, x2 = q)
    (h3 : ¬ ∃ q : ℚ, x3 = q)
    (sum_integer : ∃ n : ℤ, x1 + x2 + x3 = n)
    (sum_reciprocals_integer : ∃ m : ℤ, (1/x1) + (1/x2) + (1/x3) = m) :
  true :=
sorry

end exists_three_irrationals_l690_690669


namespace first_train_speed_l690_690301

noncomputable def train_speed_proof 
  (train_length_m : ℝ)
  (second_train_speed_kmh : ℝ)
  (passing_time_seconds : ℝ)
  (first_train_speed_kmh : ℝ) 
  : Prop :=
  let total_length_km := (2 * train_length_m) / 1000
  let passing_time_hours := passing_time_seconds / 3600
  let relative_speed_kmh := total_length_km / passing_time_hours in 
  (relative_speed_kmh = first_train_speed_kmh + second_train_speed_kmh)

theorem first_train_speed 
  (train_length_m := 170)
  (second_train_speed_kmh := 50)
  (passing_time_seconds := 11.657142857142858)
  : ∃ first_train_speed_kmh, train_speed_proof train_length_m second_train_speed_kmh passing_time_seconds first_train_speed_kmh ∧ first_train_speed_kmh = 55 :=
begin
  use 55,
  unfold train_speed_proof,
  simp,
  apply eq_true_intro,
  sorry
end

end first_train_speed_l690_690301


namespace polynomial_P_floor_l690_690003

theorem polynomial_P_floor (P : ℕ → ℝ) (h_deg : nat_degree P = 2015)
  (h_values : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2016 → P n = 1 / (n^2)) : 
  (⌊2017 * (P 2017)⌋ = -9) :=
sorry

end polynomial_P_floor_l690_690003


namespace determine_m_l690_690109

-- Define the points and the slope condition
variable (m : ℝ)

def point_P := (3, m)
def point_M := (0, 1)
def slope := -1

-- The line passing through (0,1) with slope -1
def on_line (P M : ℕ × ℝ) (s : ℝ) : Prop :=
  (P.snd - M.snd) / (P.fst - M.fst) = s

theorem determine_m : on_line (point_P m) point_M slope → m = -2 :=
by
  intro h
  sorry

end determine_m_l690_690109


namespace determine_edges_and_volume_l690_690602

/-- Given edge lengths 12, 18, 20, 24, 30, 33, and 70 cm for a rectangular prism, 
Determine that 33, 70, and 24 cm form a valid group based on their prime factors. -/
theorem determine_edges_and_volume : 
  let available_edges := [12, 18, 20, 24, 30, 33, 70] in
  (33 in available_edges ∧ 70 in available_edges ∧ 24 in available_edges) ∧
  (33 * 70 * 24 = 88 * 3^3 * 5)
:=
by
  sorry

end determine_edges_and_volume_l690_690602


namespace num_two_digit_primes_with_digit_sum_10_l690_690833

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690833


namespace harry_travel_time_l690_690494

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l690_690494


namespace XA_perpendicular_BC_l690_690201

open EuclideanGeometry

variables {A B C L M N U V X : Point}
variables (triangle_ABC : Triangle A B C)
variables (acute_triangle : triangle_ABC.is_acute)

variables (L_mid : L.is_midpoint B C)
variables (M_mid : M.is_midpoint A B)
variables (N_mid : N.is_midpoint A C)

variables (circle_Γ1 : Circle (segment A B).midpoint (A - B).r / 2)
variables (circle_Γ2 : Circle (segment A C).midpoint (A - C).r / 2)

variables (U_on_Γ1 : U ∈ circle_Γ1 ∧ U ∈ segment L M)
variables (V_on_Γ2 : V ∈ circle_Γ2 ∧ V ∈ segment L N)

variables (tangent_U : is_tangent_at circle_Γ1 U X)
variables (tangent_V : is_tangent_at circle_Γ2 V X)

theorem XA_perpendicular_BC 
  (h1 : acute_triangle)
  (h2 : L_mid)
  (h3 : M_mid)
  (h4 : N_mid)
  (h5 : U_on_Γ1)
  (h6 : V_on_Γ2)
  (h7 : tangent_U)
  (h8 : tangent_V) :
  is_perpendicular XA BC :=
by { sorry }

end XA_perpendicular_BC_l690_690201


namespace probability_calculation_l690_690263

-- Define the set of numbers from 1 to 200
def numbers := set.range (200 + 1)

-- Define the set of perfect squares under 200
def perfect_squares := {n | ∃ (k : ℕ), k^2 = n ∧ n ∈ numbers}

-- Define the set of perfect cubes under 200
def perfect_cubes := {n | ∃ (k : ℕ), k^3 = n ∧ n ∈ numbers}

-- Define the set of products of two distinct primes under 200
def products_distinct_primes := {n | ∃ (p q : ℕ), nat.prime p ∧ nat.prime q ∧ p ≠ q ∧ p * q = n ∧ n ∈ numbers}

-- Define the set of numbers that are either perfect squares, perfect cubes, or products of two distinct primes
def special_numbers := perfect_squares ∪ perfect_cubes ∪ products_distinct_primes

-- Define the probability that a randomly selected number from 1 to 200 is neither a perfect square, a perfect cube, nor a product of any two distinct primes
noncomputable def probability_neither_special (p : ℕ) : ℚ :=
  (200 - (special_numbers.to_finset.card)) / 200

theorem probability_calculation (p : ℕ) : 
  probability_neither_special p = (183 - p : ℚ) / 200 := 
by
  sorry

end probability_calculation_l690_690263


namespace gym_towel_usage_l690_690727

/-- Proof problem ends as follows: Given the conditions of guest arrivals and towel usage at the 
gym, the number of towels the gym needs to wash in the evening is 807 --/
theorem gym_towel_usage
  (guests_hour1 : ℕ := 40)
  (increase_percent_hour2 : ℝ := 0.2)
  (increase_percent_hour3 : ℝ := 0.25)
  (increase_fraction_hour4 : ℚ := 1 / 3)
  (decrease_percent_hour5 : ℝ := 0.15)
  (decrease_percent_hour7 : ℝ := 0.3)
  (decrease_percent_hour8 : ℝ := 0.5)
  (staff_towels : ℕ := 20)
  (percent3 : ℝ := 0.10)
  (percent2 : ℝ := 0.60)
  (percent1 : ℝ := 0.30) :
  let hour2 := guests_hour1 + Nat.floor (guests_hour1 * increase_percent_hour2)
  let hour3 := hour2 + Nat.floor (hour2 * increase_percent_hour3)
  let hour4 := hour3 + Nat.floor (hour3 * increase_fraction_hour4.to_real)
  let hour5 := hour4 - Nat.floor (hour4 * decrease_percent_hour5)
  let hour6 := hour5
  let hour7 := hour6 - Nat.floor (hour6 * decrease_percent_hour7)
  let hour8 := hour7 - Nat.floor (hour7 * decrease_percent_hour8)
  let total_guests := guests_hour1 + hour2 + hour3 + hour4 + hour5 + hour6 + hour7 + hour8
  let towels_guests := 44 * 3 + 262 * 2 + 131 * 1
  let total_towels := towels_guests + staff_towels
  total_towels = 807 :=
begin
  sorry
end

end gym_towel_usage_l690_690727


namespace count_two_digit_primes_with_digit_sum_10_l690_690855

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690855


namespace smallest_positive_integer_n_l690_690425

noncomputable def matrix_330 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (330 * Real.pi / 180), -Real.sin (330 * Real.pi / 180)],
    ![Real.sin (330 * Real.pi / 180), Real.cos (330 * Real.pi / 180)]
  ]

theorem smallest_positive_integer_n (n : ℕ) (h : matrix_330 ^ n = 1) : n = 12 := sorry

end smallest_positive_integer_n_l690_690425


namespace last_number_remains_l690_690188

theorem last_number_remains 
    (numbers : List ℕ)
    (start_skip : ℕ)
    (total_numbers : ℕ)
    (initial_step : ℕ)
    (step : ℕ)
    (target : ℕ) :
  (numbers = List.range (total_numbers + 1).tail) →
  (start_skip = 2) →
  (initial_step = 1) →
  (step = 3) →
  (total_numbers = 150) →
  (target = 147) →
  (remaining_numbers : List ℕ := numbers) →
  -- Function to mark out every nth number starting from a given index:
  let mark_out_list : (List ℕ) → ℕ → ℕ → ℕ → List ℕ :=
    λ l start step total, (λ l index step, l.filter (λ x, (l.indexOf x + 1 + start) % step ≠ 1)) l 0 step in

  -- Iterate the marking out process until only one number remains:
  let rec process (l : List ℕ) : ℕ :=
    if l.length = 1 then l.head!
    else process (mark_out_list l start_skip step total_numbers) in

  process numbers = target :=
  sorry

end last_number_remains_l690_690188


namespace bob_investment_correct_l690_690730

noncomputable def initial_investment_fundA : ℝ := 2000
noncomputable def interest_rate_fundA : ℝ := 0.12
noncomputable def initial_investment_fundB : ℝ := 1000
noncomputable def interest_rate_fundB : ℝ := 0.30
noncomputable def fundA_after_two_years := initial_investment_fundA * (1 + interest_rate_fundA)
noncomputable def fundB_after_two_years (B : ℝ) := B * (1 + interest_rate_fundB)^2
noncomputable def extra_value : ℝ := 549.9999999999998

theorem bob_investment_correct :
  fundA_after_two_years = fundB_after_two_years initial_investment_fundB + extra_value :=
by
  sorry

end bob_investment_correct_l690_690730


namespace count_two_digit_primes_with_digit_sum_10_l690_690857

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690857


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690920

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690920


namespace constant_term_expansion_l690_690542

theorem constant_term_expansion (x : ℂ) : 
  let term (k : ℕ) := (Nat.choose 6 k) * x ^ (6 - 2 * k) in
  term 3 = 20 :=
by
  sorry

end constant_term_expansion_l690_690542


namespace sum_f_l690_690467

def f (x : ℕ) : ℝ := sin (real.pi / 6 * x) * cos (real.pi / 6 * x)

theorem sum_f (n : ℕ) (n_eq_2018 : n = 2018) : 
  (∑ i in finset.range (n + 1), f i) = 0 :=
sorry

end sum_f_l690_690467


namespace two_digit_prime_sum_to_ten_count_l690_690845

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690845


namespace round_to_nearest_tenth_l690_690223

def orig_number : ℝ := 52.63487
def rounded_number : ℝ := 52.6

theorem round_to_nearest_tenth : (Real.round_to_nearest_tenth orig_number) = rounded_number := 
by 
  sorry

end round_to_nearest_tenth_l690_690223


namespace find_y_coordinate_C_l690_690585

noncomputable def y_coordinate_C (A B C D E : Point) (h : ℝ) : Prop :=
  A = (0, 0) ∧ B = (0, 5) ∧ D = (5, 5) ∧ E = (5, 0) ∧
  let AE := 5 in
  let BD := 5 in
  let area_pentagon := 50 in
  let area_ABDE := AE * BD in
  let area_BCD := area_pentagon - area_ABDE in
  (1 / 2) * AE * (h - BD) = area_BCD ∧ C = (2.5, h)

theorem find_y_coordinate_C : ∃ h, y_coordinate_C (0, 0) (0, 5) (2.5, h) (5, 5) (5, 0) (15) :=
by {
  sorry
}

end find_y_coordinate_C_l690_690585


namespace smallest_n_l690_690567

theorem smallest_n (n : ℕ) (x : Fin n → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum : (Finset.univ.sum x) = 1)
  (h_sqrsum : (Finset.univ.sum (λ i, (x i)^2)) ≤ 1 / 100) :
  n ≥ 100 :=
sorry

end smallest_n_l690_690567


namespace proof_OI_perp_DE_and_OI_eq_DE_l690_690159

noncomputable def angle (A B C : Point) : ℝ := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def distance (P Q : Point) : ℝ := sorry

variables {A B C O I D E : Point}

-- Conditions
axiom cond1 : angle A B C = 30
axiom cond2 : O = circumcenter A B C
axiom cond3 : I = incenter A B C
axiom cond4a : distance A D = distance A B
axiom cond4b : distance B E = distance A B 
axiom cond4c : distance A B = distance D E

theorem proof_OI_perp_DE_and_OI_eq_DE
  (cond1 : angle A B C = 30)
  (cond2 : O = circumcenter A B C)
  (cond3 : I = incenter A B C)
  (cond4a : distance A D = distance A B)
  (cond4b : distance B E = distance A B)
  (cond4c : distance A B = distance D E) :
  ∃ (OI_perp_DE : Prop) (OI_eq_DE : Prop), OI_perp_DE ∧ OI_eq_DE :=
begin
  sorry
end

end proof_OI_perp_DE_and_OI_eq_DE_l690_690159


namespace inequality_sqrt_sum_l690_690440

theorem inequality_sqrt_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  sqrt (x / (y + z)) + sqrt (y / (z + x)) + sqrt (z / (x + y)) ≥ 2 :=
by sorry

end inequality_sqrt_sum_l690_690440


namespace find_m_from_preservation_interval_l690_690460

/-- If the function g(x) = x + m - ln(x) has a preservation interval [2, +∞), 
then the value of m is equal to ln(2). -/
theorem find_m_from_preservation_interval (g : ℝ → ℝ) (m : ℝ) (A : set ℝ)
  (hA : A = set.Ici 2)  (h_g_def : ∀ x, g x = x + m - real.log x)
  (h_g_preserve : ∀ x ∈ A, g x ∈ A) : 
  m = real.log 2 := 
sorry

end find_m_from_preservation_interval_l690_690460


namespace savings_calculation_l690_690703

noncomputable def calculate_savings (spent_price : ℝ) (saving_pct : ℝ) : ℝ :=
  let original_price := spent_price / (1 - (saving_pct / 100))
  original_price - spent_price

-- Define the spent price and saving percentage
def spent_price : ℝ := 20
def saving_pct : ℝ := 12.087912087912088

-- Statement to be proved
theorem savings_calculation : calculate_savings spent_price saving_pct = 2.75 :=
  sorry

end savings_calculation_l690_690703


namespace count_two_digit_primes_with_digit_sum_10_l690_690985

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690985


namespace polygon_division_l690_690373

open Real

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def O : Point2D := {x := 0, y := 0}
def A : Point2D := {x := 0, y := 6}
def B : Point2D := {x := 4, y := 6}
def C : Point2D := {x := 4, y := 4}
def D : Point2D := {x := 6, y := 4}
def E : Point2D := {x := 6, y := 0}
def M : Point2D := {x := 2, y := 3}

noncomputable def line_eq (l : ℝ → ℝ) : Prop :=
  ∀ x, l x = - (1 / 3) * x + 11 / 3

theorem polygon_division :
  (∃ l : ℝ → ℝ, l M.x = M.y ∧ line_eq l ∧ (polygon O A B C D E).area / 2 = (area_of_subpolygon l) / 2 )  :=
sorry

end polygon_division_l690_690373


namespace find_remainder_l690_690197

theorem find_remainder (x : ℤ) (h : 0 < x ∧ 7 * x % 26 = 1) : (13 + 3 * x) % 26 = 6 :=
sorry

end find_remainder_l690_690197


namespace original_wage_l690_690367

theorem original_wage (W : ℝ) (h : 1.5 * W = 42) : W = 28 :=
by
  sorry

end original_wage_l690_690367


namespace x_intercept_of_line_through_AB_l690_690616

-- Given points A and B
def A : ℝ × ℝ := (3, 9)
def B : ℝ × ℝ := (-1, 1)

-- Definition of a line through two given points A and B
def line_through (A B : ℝ × ℝ) : ℝ → ℝ := fun x => 
  let (x1, y1) := A
  let (x2, y2) := B
  (y2 - y1) / (x2 - x1) * (x - x1) + y1

-- Goal
theorem x_intercept_of_line_through_AB : 
  let ℓ := line_through A B in
  ℓ (-3/2) = 0 :=
by
  sorry

end x_intercept_of_line_through_AB_l690_690616


namespace integer_solutions_count_2009_l690_690139

theorem integer_solutions_count_2009 :
  ∃ s : Finset (ℤ × ℤ × ℤ), (∀ (x y z : ℤ), (x, y, z) ∈ s ↔ x * y * z = 2009) ∧ s.card = 72 :=
  sorry

end integer_solutions_count_2009_l690_690139


namespace max_ab_if_circles_tangent_l690_690481

theorem max_ab_if_circles_tangent (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : ∀ x y : ℝ, (x + a)^2 + (y - 2)^2 = 1 → (x - b)^2 + (y - 2)^2 = 4)
  (h_tangent : (a + b) = 3):
  ab ≤ 9 / 4 := by
  sorry

end max_ab_if_circles_tangent_l690_690481


namespace two_digit_prime_sum_to_ten_count_l690_690839

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690839


namespace ratio_josh_to_selena_l690_690225

def total_distance : ℕ := 36
def selena_distance : ℕ := 24

def josh_distance (td sd : ℕ) : ℕ := td - sd

theorem ratio_josh_to_selena : (josh_distance total_distance selena_distance) / selena_distance = 1 / 2 :=
by
  sorry

end ratio_josh_to_selena_l690_690225


namespace delta_minus_star_l690_690172

variable (Δ * upr : ℕ) (R1 R2 C1 C2 : ℕ)

-- Conditions (from sums of rows and columns)
def row1_sum : Prop := Δ + * = R1
def row2_sum : Prop := upr + * = R2
def col1_sum : Prop := Δ + upr = C1
def col2_sum : Prop := * + upr = C2
def total_sum : Prop := Δ + upr + * = 105

-- Given the sums of rows and columns, prove the required difference
theorem delta_minus_star : 
    row1_sum Δ * R1 → row2_sum upr * R2 → col1_sum Δ upr C1 → col2_sum * upr C2 → total_sum Δ upr * → 
    Δ - * = 23 :=
by 
  intros 
  sorry

end delta_minus_star_l690_690172


namespace fox_alice_possible_numbers_l690_690079

-- Definition of the problem conditions.
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by (n d : ℕ) : Prop :=
  n % d = 0

-- Main theorem statement.
theorem fox_alice_possible_numbers :
  {n : ℕ | is_two_digit n ∧
   (divisible_by n 2 ∨ divisible_by n 3 ∨ divisible_by n 4 ∨ divisible_by n 5 ∨ divisible_by n 6) ∧
   ((divisible_by n 2 ∧ ¬divisible_by n 3 ∧ divisible_by n 4 ∧ ¬divisible_by n 5 ∧ divisible_by n 6) ∨
    (divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 4 ∧ ¬divisible_by n 5 ∧ ¬divisible_by n 6) ∨
    (¬divisible_by n 2 ∧ divisible_by n 3 ∧  divisible_by n 4 ∧ divisible_by n 5 ∧ ¬divisible_by n 6))}.card = 8 :=
by
  -- SORRY is used here to indicate the theorem proof is omitted.
  sorry

end fox_alice_possible_numbers_l690_690079


namespace total_people_l690_690695

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l690_690695


namespace new_student_bmi_l690_690674

theorem new_student_bmi 
(average_weight_29 : ℚ)
(average_height_29 : ℚ)
(average_weight_30 : ℚ)
(average_height_30 : ℚ)
(new_student_height : ℚ)
(bmi : ℚ)
(h1 : average_weight_29 = 28)
(h2 : average_height_29 = 1.5)
(h3 : average_weight_30 = 27.5)
(h4 : average_height_30 = 1.5)
(h5 : new_student_height = 1.4)
: bmi = 6.63 := 
sorry

end new_student_bmi_l690_690674


namespace two_digit_primes_with_digit_sum_10_count_l690_690900

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690900


namespace problem_conversion_l690_690665

noncomputable def similar_radical_expressions (a : ℝ) : Prop :=
  let expr1 := real.cbrt (3 * a^3)
  let expr2 := 3 in
  expr1 = a * real.cbrt(3 * a)

theorem problem_conversion (a : ℝ) : similar_radical_expressions a := by
  sorry

end problem_conversion_l690_690665


namespace call_cost_5_7_minutes_l690_690247

def cost_of_call (m : ℝ) : ℝ :=
  1.06 * (0.5 * (⌊m⌋) + 1)

theorem call_cost_5_7_minutes : cost_of_call 5.7 = 3.71 := 
by 
  sorry

end call_cost_5_7_minutes_l690_690247


namespace correct_transformation_l690_690668

-- Conditions given in the problem
def cond_A (a : ℤ) : Prop := a + 3 = 9 → a = 3 + 9
def cond_B (x : ℤ) : Prop := 4 * x = 7 * x - 2 → 4 * x - 7 * x = 2
def cond_C (a : ℤ) : Prop := 2 * a - 2 = -6 → 2 * a = 6 + 2
def cond_D (x : ℤ) : Prop := 2 * x - 5 = 3 * x + 3 → 2 * x - 3 * x = 3 + 5

-- Prove that the transformation in condition D is correct
theorem correct_transformation : (∀ a : ℤ, ¬cond_A a) ∧ (∀ x : ℤ, ¬cond_B x) ∧ (∀ a : ℤ, ¬cond_C a) ∧ (∀ x : ℤ, cond_D x) :=
by {
  -- Proof is provided in the solution and skipped here
  sorry
}

end correct_transformation_l690_690668


namespace boy_run_time_l690_690133

section
variables {d1 d2 d3 d4 : ℝ} -- distances
variables {v1 v2 v3 v4 : ℝ} -- velocities
variables {t : ℝ} -- time

-- Define conditions
def distances_and_velocities (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) :=
  d1 = 25 ∧ d2 = 30 ∧ d3 = 40 ∧ d4 = 35 ∧
  v1 = 3.33 ∧ v2 = 3.33 ∧ v3 = 2.78 ∧ v4 = 2.22

-- Problem statement
theorem boy_run_time
  (h : distances_and_velocities d1 d2 d3 d4 v1 v2 v3 v4) :
  t = (d1 / v1) + (d2 / v2) + (d3 / v3) + (d4 / v4) := 
sorry
end

end boy_run_time_l690_690133


namespace areas_of_triangles_equal_sum_quadrilateral_area_l690_690568

-- Definitions of various points and their positions
variables {A B C D E F G H : Type} 
[ConvexQuadrilateral ABCD]
[midpoint E A B]
[midpoint F C D]
[intersection G AF DE]
[intersection H BF CE]

-- Area calculation function
variable (area : ∀ {α β γ : Type}, α ≠ β → β ≠ γ → γ ≠ α → α × β × γ → ℝ)

-- Theorem: statement to prove
theorem areas_of_triangles_equal_sum_quadrilateral_area
  (h1: E = midpoint A B)
  (h2: F = midpoint C D)
  (h3: G = intersection AF DE)
  (h4: H = intersection BF CE) :
  area A G D + area B H C = area E H F G :=
by sorry

end areas_of_triangles_equal_sum_quadrilateral_area_l690_690568


namespace monotonic_decreasing_0_1_function_definition_on_interval_range_of_a_l690_690742

open Set

def f : ℝ → ℝ := λ x, 
  if -1 < x ∧ x < 0 then - (2^x / (4^x + 1))
  else if 0 < x ∧ x < 1 then 2^x / (4^x + 1)
  else 0

theorem monotonic_decreasing_0_1 : ∀ x y : ℝ, (0 < x ∧ x < 1) → (0 < y ∧ y < 1) → (x < y) → (f y < f x) :=
by
  sorry

theorem function_definition_on_interval : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = 
  if -1 < x ∧ x < 0 then - (2^x / (4^x + 1)) 
  else if 0 < x ∧ x < 1 then 2^x / (4^x + 1) 
  else 0 :=
by
  sorry

theorem range_of_a : ∀ a : ℝ, (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = a) ↔ a ∈ ((-1/2 : ℝ), -2/5) ∪ {0} ∪ ((2/5) : ℝ, 1/2) :=
by
  sorry

end monotonic_decreasing_0_1_function_definition_on_interval_range_of_a_l690_690742


namespace problem_equivalent_l690_690966

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690966


namespace tan_x_cot_sum_sec2_x_csc2_sum_l690_690202

-- Definitions of angles A, B, C of a triangle
variables {A B C x : ℝ}

-- The condition to be verified
def angle_condition (A B C : ℝ) : Prop :=
  A + B + C = π

-- The given equation which x satisfies
def given_equation (x A B C : ℝ) : Prop :=
  cos (x + A) * cos (x + B) * cos (x + C) + cos x ^ 3 = 0

-- The theorem to prove tan x = cot A + cot B + cot C given the conditions
theorem tan_x_cot_sum (h_angle : angle_condition A B C) (h_eq : given_equation x A B C) :
    tan x = cot A + cot B + cot C := sorry

-- The theorem to prove sec^2 x = csc^2 A + csc^2 B + csc^2 C given the conditions
theorem sec2_x_csc2_sum (h_angle : angle_condition A B C) (h_eq : given_equation x A B C) :
    sec x ^ 2 = csc A ^ 2 + csc B ^ 2 + csc C ^ 2 := sorry

end tan_x_cot_sum_sec2_x_csc2_sum_l690_690202


namespace inequality_valid_for_positive_x_l690_690588

theorem inequality_valid_for_positive_x (x : ℝ) (hx : 0 < x) : 
  2 ^ (x ^ (1/12)) + 2 ^ (x ^ (1/4)) ≥ 2 ^ (1 + x ^ (1/6)) :=
by
  sorry

end inequality_valid_for_positive_x_l690_690588


namespace compute_factorial_expression_l690_690735

theorem compute_factorial_expression (N : ℕ) (h : N ≥ 1) : 
  ( ((N - 1)!) * N ) / ( (N + 1)! ) = 1 / (N + 1) := 
by 
  sorry

end compute_factorial_expression_l690_690735


namespace teal_total_sales_l690_690531

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l690_690531


namespace presidency_meeting_count_l690_690339

noncomputable def presidency_meeting_ways
  (num_schools : ℕ)
  (members_per_school : ℕ)
  (host_representatives : ℕ)
  (non_host_representatives : ℕ)
  (host_schools : fin num_schools)
  (senior_per_school : ℕ) :=
  let host_choices := fintype.card {s : fin num_schools // s ≠ host_schools} in
  let combinations := nat.choose members_per_school host_representatives in
  let senior_choices := nat.choose senior_per_school 1 in
  let non_senior_choices := nat.choose (members_per_school - senior_per_school) 1 in
  let non_host_representatives_ways := senior_choices * non_senior_choices in
  let total_non_host_ways := non_host_representatives_ways ^ (num_schools - 1) in
  combinations * total_non_host_ways * host_choices

theorem presidency_meeting_count:
  presidency_meeting_ways 4 6 3 2 1 3 = 58320 :=
by
  sorry

end presidency_meeting_count_l690_690339


namespace main_theorem_l690_690537

open Classical

noncomputable def CartesianEquationOfL : Prop := 
  ∀ t : ℝ, 2 * (1 - t) + (-1 + 2 * t) - 1 = 0

noncomputable def CartesianEquationOfC : Prop :=
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 3) = 1

noncomputable def ReciprocalSumPAPlusPB : Prop :=
  let A B : ℝ × ℝ := (1 - t, -1 + 2 * t) in -- points of intersection
  let PA PB : ℝ := dist A (1, -1) ∧ dist B (1, -1) in
  (1 / PA) + (1 / PB) = 12 * Real.sqrt 30 / 25 

theorem main_theorem :
  CartesianEquationOfC ∧ CartesianEquationOfL ∧ 
  ReciprocalSumPAPlusPB :=
by
  constructor
  case left => 
    -- proof of Cartesian equation of C
    sorry
  case right =>
    constructor
    case left =>
      -- proof of Cartesian equation of L
      sorry
    case right =>
      -- proof of Reciprocal Sum PA + PB
      sorry

end main_theorem_l690_690537


namespace sqrt_sum_inequality_l690_690438

theorem sqrt_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a * b / (c + a * b)) + sqrt (b * c / (a + b * c)) + sqrt (c * a / (b + c * a)) ≤ 3 / 2 :=
by
  sorry

end sqrt_sum_inequality_l690_690438


namespace two_digit_primes_with_digit_sum_10_count_l690_690909

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690909


namespace digit_before_y_l690_690308

theorem digit_before_y (x y : ℕ) (h1 : List.Sum [x, 8, 0, 3] - List.Sum [y, 6, 8] = 0) (h2 : y = 3) : x = 6 :=
sorry

end digit_before_y_l690_690308


namespace notebook_words_per_page_l690_690351

-- Given conditions as assumptions
def words_per_page (p : ℕ) : Prop := 
  180 * p % 221 = 246 ∧ p ≤ 150

-- Problem statement
theorem notebook_words_per_page : ∃ p : ℕ, words_per_page p ∧ p = 24 := 
by
  use 24
  simp [words_per_page]
  split
  {
    -- This proves (180 * 24 ≡ 246 % 221) part
    norm_num1,
  }
  {
    -- This proves (24 ≤ 150) part
    norm_num,
  }

end notebook_words_per_page_l690_690351


namespace find_a_dot_d_l690_690194

-- Given conditions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)

-- a, b, c, d are unit vectors
axiom unit_a : ∥a∥ = 1
axiom unit_b : ∥b∥ = 1
axiom unit_c : ∥c∥ = 1
axiom unit_d : ∥d∥ = 1

-- dot product conditions
axiom a_dot_b : ⟪a, b⟫ = -1 / 7
axiom b_dot_c : ⟪b, c⟫ = -1 / 7
axiom c_dot_d : ⟪c, d⟫ = -1 / 7
axiom b_dot_d : ⟪b, d⟫ = 1 / 3

-- Proof statement to find a dot d
theorem find_a_dot_d : ⟪a, d⟫ = -1 / 7 := sorry

end find_a_dot_d_l690_690194


namespace two_digit_primes_with_digit_sum_ten_l690_690793

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690793


namespace min_real_roots_l690_690564

open Polynomial

-- Define the polynomial conditions
variables (g : Polynomial ℝ)
hyp : g.natDegree = 2010
roots : Multiset ℕ := g.roots.map abs_val.to_nat

-- Translate problem to Lean 4 statement
theorem min_real_roots (g : Polynomial ℝ) (h_coeff : g.natDegree = 2010)
  (dist_vals : Multiset.card (roots g) = 1008) : 
  ∃ (n : ℕ), (roots g).count(ℕ) ≥ 6 :=
sorry

end min_real_roots_l690_690564


namespace probability_at_least_one_second_class_product_l690_690523

theorem probability_at_least_one_second_class_product :
  let totalProducts := 100
  let firstClassProducts := 90
  let secondClassProducts := 10
  let numSelections := 4
  let p_first_draw := (firstClassProducts : ℚ) / totalProducts
  let p_second_draw := (firstClassProducts - 1 : ℚ) / (totalProducts - 1)
  let p_third_draw := (firstClassProducts - 2 : ℚ) / (totalProducts - 2)
  let p_fourth_draw := (firstClassProducts - 3 : ℚ) / (totalProducts - 3)
  let p_all_first_class := p_first_draw * p_second_draw * p_third_draw * p_fourth_draw
  in (1 - p_all_first_class).round(2) = 0.35 :=
begin
  -- The proof would go here, but is omitted
  sorry
end

end probability_at_least_one_second_class_product_l690_690523


namespace find_a_find_b_l690_690094

def sequence_a (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧
  a 3 + a 5 = 8 ∧
  ∀ n > 1, 2 * a n = a (n - 1) + a (n + 1)

def sequence_b (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  b 1 = a 1 ∧
  ∀ n, b (n + 1) - b n = (1 : ℕ) / ((𝓝.sqrt (a (n + 1))) + (𝓝.sqrt (a n)))

theorem find_a (a : ℕ → ℕ) (h : sequence_a a) : ∀ n > 1, a n = n := sorry

theorem find_b (b a : ℕ → ℕ) (ha : sequence_a a) (hb : sequence_b b a) : ∀ n > 1, b n = 𝓝.sqrt n := sorry

end find_a_find_b_l690_690094


namespace cube_root_neg_64_l690_690248

theorem cube_root_neg_64 : ∃ b : ℤ, b^3 = -64 ∧ b = -4 :=
by {
  use (-4),
  split,
  -- proof of b^3 = -64
  sorry,
  -- proof of b = -4
  sorry
}

end cube_root_neg_64_l690_690248


namespace func_relationship_l690_690122

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + Real.log2 (x + Real.sqrt (x^2 + 1))

theorem func_relationship {a b : ℝ} (h : a + b ≥ 0) : f a + f b ≥ 0 :=
sorry

end func_relationship_l690_690122


namespace chord_not_intersect_inner_circle_probability_l690_690641

noncomputable def radius_inner := 2
noncomputable def radius_outer := 3

def probability_chord_not_intersect_inner_circle : ℝ :=
  286.26 / 360.0

theorem chord_not_intersect_inner_circle_probability :
  ∀ (r1 r2 : ℝ), r1 = radius_inner → r2 = radius_outer →
  (r1 < r2) →
  (∀ (p1 p2 : ℝ), p1 ∈ set.Icc (0:ℝ) (2 * Real.pi) ∧
   p2 ∈ set.Icc (0:ℝ) (2 * Real.pi) →
    -- Here would follow the mathematical proof conditions specific to points p1, p2 generating chords
    ((probability_chord_not_intersect_inner_circle = 286.26 / 360.0))
  ) :=
by sorry

end chord_not_intersect_inner_circle_probability_l690_690641


namespace coins_division_arithmetic_sequence_l690_690236

theorem coins_division_arithmetic_sequence :
  ∃ (a d : ℚ), (5 * a = 5) ∧ ((a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d))) 
  ∧ ((a - 2 * d) = (4 / 3)) :=
begin
  sorry
end

end coins_division_arithmetic_sequence_l690_690236


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690982

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690982


namespace trains_cross_time_l690_690332

noncomputable def timeToCross (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (1000 / 3600)
  let speed2_mps := speed2_kmph * (1000 / 3600)
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_time:
  ∀ (length1 length2 speed1_kmph speed2_kmph : ℝ),
    length1 = 270 →
    speed1_kmph = 120 →
    length2 = 230 →
    speed2_kmph = 80 →
    timeToCross length1 length2 speed1_kmph speed2_kmph ≈ 9 :=
by
  intros
  sorry

end trains_cross_time_l690_690332


namespace group_population_l690_690697

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l690_690697


namespace prime_remainder_theorem_l690_690501

open Nat

-- Definitions of the problem's conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def prime_remainder_condition (p : ℕ) : Prop := is_prime p ∧ (50 < p) ∧ (p < 100) ∧ is_prime (p % 10)

-- Statement of the problem
theorem prime_remainder_theorem : 
  (Finset.univ.filter prime_remainder_condition).card = 5 := 
  sorry

end prime_remainder_theorem_l690_690501


namespace unique_polynomial_l690_690423

theorem unique_polynomial (p : ℝ → ℝ)
  (h : ∀ x : ℝ, p(x^3) - p(x^3 - 2) = p(x)^2 + 18) :
  p = fun x => 6 * x ^ 3 + real.sqrt 30 ∨ p = fun x => 6 * x ^ 3 - real.sqrt 30 :=
sorry

end unique_polynomial_l690_690423


namespace region_area_l690_690060

def r1 (θ : ℝ) : ℝ := 2 / Real.cos θ
def r2 (θ : ℝ) : ℝ := 2 / Real.sin θ

theorem region_area :
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2} in
  ∃ area : ℝ, area = 4 := 
begin
  sorry
end

end region_area_l690_690060


namespace divide_segment_l690_690645

-- Definitions based on conditions 
variable {A B X X1 X2 X3 C D : Type}

-- Given points A, B
structure Segment (A B : Type) := 
(startPoint : A)
(endPoint : B)

variable (AB : Segment A B)

-- Auxiliary ray starting from point A
structure Ray (startPoint : A) :=
  (direction : Type)

variable (AX : Ray A)

-- Mark segments AX1, X1X2, X2X3 on the ray AX as equal
structure EqualSegments :=
  (AX1 X1X2 X2X3 : Segment A A)
  (equalLength : AX1.endPoint = X1X2.startPoint ∧
                X1X2.endPoint = X2X3.startPoint ∧
                true)

variable (equalSegments : EqualSegments)

-- Connect X3 to B
structure Connect (X A B : Type) :=
  (line : Segment X B)

variable (X3B : Connect X3 A B)

-- Draw parallel lines
structure ParallelLines (X A C D : Type) :=
  (parallel1 : Segment X A)
  (parallel2 : Segment A C)
  (parallel3 : Segment A D)

variable (parallelLines : ParallelLines X3 X1 X2 B)

-- Intersection points C, D on AB
structure IntersectionPoints (AB AX : Segment A A) :=
  (C D : Type)
  (intersect : AB.startPoint = AX.endPoint ∧
              AX1.endPoint = C ∧
              C = D ∧
              D = AB.endPoint)

variable (intersections : IntersectionPoints)

-- Thales's Theorem Application for equality
theorem divide_segment (thales : Segment A B) (intersections : IntersectionPoints) :
  thales.startPoint = thales.endPoint / 3 :=
sorry -- Proof omitted

end divide_segment_l690_690645


namespace problem1_problem2_problem3_l690_690546

section SequenceProofs

-- Definitions of sequences a_n and b_n
def a : ℕ → ℝ
| 0 := 0 
| 1 := 1
| (n + 1) := (1 + 1 / n) * a n + (n + 1) / (2^n)

def b (n : ℕ) : ℝ := if n = 0 then 0 else (a n) / n

-- Problem 1: Prove b_{n-1} - b_n = 1 / 2^n
theorem problem1 (n : ℕ) : n > 0 → b (n - 1) - b n = 1 / (2^n) :=
by sorry

-- Problem 2: Prove the general formula b_n = 2 - 1 / 2^(n-1)
theorem problem2 (n : ℕ) : n > 0 → b n = 2 - 1 / (2^(n-1)) :=
by sorry

-- Problem 3: Prove the sum of first n terms S_n = n(n+1) + (n+2)/(2^(n-1)) - 4
def S : ℕ → ℝ := λ n, ∑ k in range (n + 1), a k

theorem problem3 (n : ℕ) : S n = n * (n + 1) + (n + 2) / (2^(n-1)) - 4 :=
by sorry

end SequenceProofs

end problem1_problem2_problem3_l690_690546


namespace milk_for_6_cookies_l690_690637

/-- Given conditions for baking cookies -/
def quarts_to_pints : ℕ := 2 -- 2 pints in a quart
def milk_for_24_cookies : ℕ := 5 -- 5 quarts of milk for 24 cookies

/-- Theorem to prove the number of pints needed to bake 6 cookies -/
theorem milk_for_6_cookies : 
  (milk_for_24_cookies * quarts_to_pints * 6 / 24 : ℝ) = 2.5 := 
by 
  sorry -- Proof is omitted

end milk_for_6_cookies_l690_690637


namespace bus_stoppage_time_per_hour_l690_690756

/-- Given the average speeds of a bus with and without stoppages,
    this statement asserts the bus stops for 30 minutes per hour. -/
theorem bus_stoppage_time_per_hour : 
  ∀ (v₁ v₂ : ℕ), v₁ = 60 ∧ v₂ = 30 → (60 * ((1 - v₂ / v₁) * 60)= 30) :=
by
  intros v₁ v₂ h
  cases h with hv₁ hv₂
  rw [hv₁, hv₂]
  sorry

end bus_stoppage_time_per_hour_l690_690756


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l690_690576

-- Problem 1: 1 / 0.25 = 4
theorem problem1 : 1 / 0.25 = 4 :=
by sorry

-- Problem 2: 0.25 / 0.1 = 2.5
theorem problem2 : 0.25 / 0.1 = 2.5 :=
by sorry

-- Problem 3: 1.2 / 1.2 = 1
theorem problem3 : 1.2 / 1.2 = 1 :=
by sorry

-- Problem 4: 4.01 * 1 = 4.01
theorem problem4 : 4.01 * 1 = 4.01 :=
by sorry

-- Problem 5: 0.25 * 2 = 0.5
theorem problem5 : 0.25 * 2 = 0.5 :=
by sorry

-- Problem 6: 0 / 2.76 = 0
theorem problem6 : 0 / 2.76 = 0 :=
by sorry

-- Problem 7: 0.8 / 1.25 = 0.64
theorem problem7 : 0.8 / 1.25 = 0.64 :=
by sorry

-- Problem 8: 3.5 * 2.7 = 9.45
theorem problem8 : 3.5 * 2.7 = 9.45 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l690_690576


namespace ram_account_balance_increase_l690_690644

theorem ram_account_balance_increase 
  (initial_deposit : ℕ := 500)
  (first_year_balance : ℕ := 600)
  (second_year_percentage_increase : ℕ := 32)
  (second_year_balance : ℕ := initial_deposit + initial_deposit * second_year_percentage_increase / 100) 
  (second_year_increase : ℕ := second_year_balance - first_year_balance) 
  : (second_year_increase * 100 / first_year_balance) = 10 := 
sorry

end ram_account_balance_increase_l690_690644


namespace count_two_digit_primes_with_digit_sum_10_l690_690882

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690882


namespace trigonometric_product_identity_l690_690025

theorem trigonometric_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) = 2.25 :=
by
  -- Let x = Real.pi / 12 and y = 5 * Real.pi / 12
  have h1 : Real.sin (11 * Real.pi / 12) = Real.sin (Real.pi - (Real.pi / 12)) := by sorry,
  have h2 : Real.sin (7 * Real.pi / 12) = Real.sin (Real.pi - (5 * Real.pi / 12)) := by sorry,
  have h3 : Real.sin (5 * Real.pi / 12) = Real.cos (Real.pi / 12) := by sorry,
  have h4 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry,
  have h5 : Real.sin (2 * Real.pi / 12) = 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) := by sorry,
  sorry

end trigonometric_product_identity_l690_690025


namespace count_two_digit_primes_with_digit_sum_10_l690_690997

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690997


namespace power_function_decreasing_m_eq_2_l690_690621

theorem power_function_decreasing_m_eq_2 
    (m : ℝ) 
    (h1 : m^2 - m - 1 = 1) 
    (h2 : m^2 - 2m - 3 < 0) : 
    m = 2 := 
by
  sorry

end power_function_decreasing_m_eq_2_l690_690621


namespace selection_methods_count_l690_690281

theorem selection_methods_count (students lectures : ℕ) (h_students : students = 4) (h_lectures : lectures = 3) : 
  (lectures ^ students) = 81 :=
by
  rw [h_students, h_lectures]
  exact (3 ^ 4)
  sorry

end selection_methods_count_l690_690281


namespace part_a_part_b_l690_690221

variables {V : Type*} [normed_group V] [normed_space ℝ V]

-- This assumes that the wedge product and related operations are defined appropriately.
-- If these operations are not defined in the imported libraries, they would need to be defined.
def wedge_product (a b : V) : V := sorry  -- Placeholder for the actual wedge product definition
noncomputable def scalar_mul (λ : ℝ) (a : V) : V := λ • a
noncomputable def vector_add (a b : V) : V := a + b

theorem part_a (λ : ℝ) (a b : V) : wedge_product (scalar_mul λ a) b = scalar_mul λ (wedge_product a b) := sorry

theorem part_b (a b c : V) : wedge_product a (vector_add b c) = vector_add (wedge_product a b) (wedge_product a c) := sorry

end part_a_part_b_l690_690221


namespace harry_total_travel_time_l690_690490

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l690_690490


namespace mass_percentage_h_in_water_l690_690399

def abundance_protium : ℝ := 0.9998                         
def abundance_deuterium : ℝ := 0.0002                       
def mass_protium : ℝ := 1                                  
def mass_deuterium : ℝ := 2                                
def mass_oxygen : ℝ := 16                                  
def avg_mass_hydrogen : ℝ := (abundance_protium * mass_protium) + (abundance_deuterium * mass_deuterium)

theorem mass_percentage_h_in_water : (Mass Percentage of H in H₂O):
  let molar_mass_h2o := (2 * avg_mass_hydrogen) + mass_oxygen in
  let mass_percentage_h := ((2 * avg_mass_hydrogen) / molar_mass_h2o) * 100 in
  abs (mass_percentage_h - 11.113) < 0.001 :=
by
  sorry

end mass_percentage_h_in_water_l690_690399


namespace problem1_problem2_l690_690679

theorem problem1 : sqrt 18 - abs (2 - 3) + sqrt (1 / 2) = (7 * sqrt 2 / 2) - 1 := by
  sorry

theorem problem2 : real.cbrt 27 + sqrt 3 * (sqrt 3 + 2) - sqrt 16 - abs (3 - sqrt 12) = 5 := by
  sorry

end problem1_problem2_l690_690679


namespace harry_travel_time_l690_690495

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l690_690495


namespace solve_for_y_l690_690051

theorem solve_for_y (y : ℝ) (h : log 16 (3 * y - 4) = 2) : y = 260 / 3 :=
sorry

end solve_for_y_l690_690051


namespace count_two_digit_prime_with_digit_sum_10_l690_690866

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690866


namespace find_angle_KDA_l690_690593

-- Definitions for the problem
variables (A B C D M K : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq M] [DecidableEq K]
variables (rectangle : A → B → C → D → Prop)
variables (midpoint : M → A → D → Prop)
variables (is_inside : K → A → B → C → D → Prop)
variables (angle : K → M → A → Real)
variables (angle_bisector : K → D → M → C → Prop)

-- Conditions
axiom h1 : rectangle A B C D
axiom h2 : midpoint M A D
axiom h3 : is_inside K A B C D
axiom h4 : angle K M A = 80
axiom h5 : angle_bisector K D M C

-- Proof statement
theorem find_angle_KDA : ∃ KDA, KDA = 35 := by
  -- The proof would go here
  sorry

end find_angle_KDA_l690_690593


namespace problem1_problem2_l690_690323

-- Problem1
theorem problem1 (a : ℤ) (h : a = -2) :
    ( (a^2 + a) / (a^2 - 3 * a) / (a^2 - 1) / (a - 3) - 1 / (a + 1) = 2 / 3) :=
by 
  sorry

-- Problem2
theorem problem2 (x : ℤ) :
    ( (x^2 - 1) / (x - 4) / (x + 1) / (4 - x) = 1 - x) :=
by 
  sorry

end problem1_problem2_l690_690323


namespace range_of_m_in_third_quadrant_values_of_m_on_line_l690_690454

open Real

-- Define z based on given conditions
def z (m : ℝ) : ℂ := log 2 (1 + m : ℂ) + log (1/2 : ℂ) (3 - m : ℂ)

-- Question 1: Range of m when z lies in the third quadrant
theorem range_of_m_in_third_quadrant (m : ℝ) (hz : Complex.arg (z m) > π ∧ Complex.arg (z m) < 3 * π / 2) : -1 < m ∧ m < 0 :=
sorry

-- Question 2: Values of m when z lies on the line x - y - 1 = 0
theorem values_of_m_on_line (m : ℝ) (hline : Complex.re (z m) - Complex.im (z m) - 1 = 0) : m = 1 - sqrt 2 ∨ m = 1 + sqrt 2 :=
sorry

end range_of_m_in_third_quadrant_values_of_m_on_line_l690_690454


namespace distinct_prime_factors_exists_l690_690587

theorem distinct_prime_factors_exists (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ (nat.factorization (2^k - m)).support.card ≥ n :=
by
  sorry

end distinct_prime_factors_exists_l690_690587


namespace geometric_sequence_term_l690_690177

theorem geometric_sequence_term 
  (a_n : ℕ → ℚ)
  (S3 S6 : ℚ)
  (hS3 : S3 = 13 / 9)
  (hS6 : S6 = 364 / 9)
  (hS3_sum : S3 = a_n 0 + a_n 1 + a_n 2)
  (hS6_sum : S6 = a_n 0 + a_n 1 + a_n 2 + a_n 3 + a_n 4 + a_n 5) : 
  a_n = λ n, (1 / 6) * 3 ^ n := 
sorry

end geometric_sequence_term_l690_690177


namespace problem_equivalent_l690_690955

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690955


namespace dog_food_per_meal_l690_690280

theorem dog_food_per_meal :
  ∃ (D : ℕ), (∃ (num_dogs num_puppies : ℕ), num_dogs = 3 ∧ num_puppies = 4 ∧
  (∀ (dog_meals : ℕ) (puppy_meals : ℕ) (dog_food : ℕ),
    dog_meals = 3 ∧ puppy_meals = 9 ∧ dog_food = D ∧
    let puppy_food := dog_food / 2 in
    let total_food := num_dogs * dog_meals * dog_food + num_puppies * puppy_meals * puppy_food in
    total_food = 108)) →
  D = 4 :=
by
  sorry

end dog_food_per_meal_l690_690280


namespace find_length_CD_l690_690169

-- Definitions based on the conditions:
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AB AD BD BC CD : ℝ) -- Lengths associated with the quadrilateral
variables (α β γ : ℝ) -- Angles between the sides

-- Given conditions
def quadrilateral_ABCD : Prop :=
  (α = β) ∧ (γ = β) ∧ (AB = 9) ∧ (BD = 12) ∧ (BC = 7)

-- Theorem stating the required proof that CD = 12
theorem find_length_CD (h: quadrilateral_ABCD A B C D E AB AD BD BC CD α β γ) :  
  CD = 12 :=
sorry

end find_length_CD_l690_690169


namespace work_completion_l690_690276

theorem work_completion (W : ℕ) (n : ℕ) (h1 : 0 < n) (H1 : 0 < W) :
  (∀ w : ℕ, w ≤ W / n) → 
  (∀ k : ℕ, k = (7 * n) / 10 → k * (3 * W) / (10 * n) ≥ W / 3) → 
  (∀ m : ℕ, m = (3 * n) / 10 → m * (7 * W) / (10 * n) ≥ W / 3) → 
  ∃ g1 g2 g3 : ℕ, g1 + g2 + g3 < W / 3 :=
by
  sorry

end work_completion_l690_690276


namespace range_of_f_l690_690116

noncomputable def f (x : ℕ) : ℤ := x^2 - 3 * x

def domain : Finset ℕ := {1, 2, 3}

def range : Finset ℤ := {-2, 0}

theorem range_of_f :
  Finset.image f domain = range :=
by
  sorry

end range_of_f_l690_690116


namespace distinct_lines_in_4_by_4_grid_l690_690736

theorem distinct_lines_in_4_by_4_grid : 
  ∃ n : ℕ, n = 72 ∧ 
  ∀ (grid : set (ℕ × ℕ)), 
    (∀ x y, (x, y) ∈ grid → x < 4 ∧ y < 4) → 
    (∃ lines : set (ℕ × ℕ) → Prop,
      (∀ {l}, lines l → ∃ p1 p2, p1 ∈ grid ∧ p2 ∈ grid ∧ p1 ≠ p2 ∧
      (∃ a b c : ℕ, (∀ {p}, p ∈ l ↔ (∃ (x y : ℕ), p = (x, y) ∧ a * x + b * y = c)))) ∧
      ∀ q, lines q → (∃! r, r = q)) :=
begin
  -- proof omitted
  sorry
end

end distinct_lines_in_4_by_4_grid_l690_690736


namespace polygon_properties_l690_690707

theorem polygon_properties :
  ∀ (x : ℝ), (∀ internal external : ℝ, internal = x ∧ external = (2/3 : ℝ) * x)
  ∧ x + (2/3 : ℝ) * x = 180 → 
  (5, 540) :=
by
  intro x conditions
  obtain ⟨internal, external, h1, h2⟩ := conditions
  have h3 : internal + external = 180 := h2
  have h4 : external = (2 / 3 : ℝ) * internal := h1
  have h5 : x = 108 := sorry 
  have num_sides : ℕ := 5
  have sum_internal_angles : ℝ := (num_sides - 2) * 180
  exact (num_sides, sum_internal_angles)
  sorry

end polygon_properties_l690_690707


namespace problem_l690_690342

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l690_690342


namespace total_people_l690_690693

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l690_690693


namespace product_xyz_is_minus_one_l690_690147

-- Definitions of the variables and equations
variables (x y z : ℝ)

-- Assumptions based on the given conditions
def condition1 : Prop := x + (1 / y) = 2
def condition2 : Prop := y + (1 / z) = 2
def condition3 : Prop := z + (1 / x) = 2

-- The theorem stating the conclusion to be proved
theorem product_xyz_is_minus_one (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) : x * y * z = -1 :=
by sorry

end product_xyz_is_minus_one_l690_690147


namespace hyperbola_C_eq_l690_690421

-- Defining the given curve equation
def given_curve (x y : ℝ) : Prop :=
  real.sqrt ((x - 1)^2 + y^2) = (real.sqrt 2 / 2) * (2 - x)

-- Defining the form of the hyperbola equation
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / (1 - a^2)) = 1

-- Point (3, -2 * sqrt 39 / 3) lies on the hyperbola
def point_on_hyperbola (x y : ℝ) : Prop :=
  x = 3 ∧ y = - (2 * real.sqrt 39 / 3)

noncomputable def hyperbola_C : ℝ → Prop :=
  λ a, ∀ x y : ℝ, given_curve 1 0 ∧ point_on_hyperbola 3 (- (2 * real.sqrt 39 / 3)) → 
    3 * x^2 - (3 / 2) * y^2 = 1

-- The theorem stating the equation of the hyperbola C
theorem hyperbola_C_eq (a : ℝ) (x y : ℝ) : Prop :=
  ∃ a, hyperbola_C a

end hyperbola_C_eq_l690_690421


namespace R2_increase_SSR_decrease_l690_690170

section RegressionAnalysis

variable (R2 : ℝ) (SSR : ℝ)

axiom R2_def : R2 ≥ 0 ∧ R2 ≤ 1
axiom SSR_def : SSR ≥ 0

theorem R2_increase_SSR_decrease (h : R2_1 < R2_2) (h1 : SSR_1 > SSR_2) : SSR_1 > SSR_2 :=
sorry

end RegressionAnalysis

end R2_increase_SSR_decrease_l690_690170


namespace max_view_angle_exists_l690_690216

-- Definitions based on problem conditions
variables (O : Point) (A B : Point)
variable (α : Angle) -- representing the given acute angle with vertex O
variable (OC : Length) -- representing the distance from O to C

-- The final theorem statement
theorem max_view_angle_exists :
  ∃ C : Point, (C ≠ A ∧ C ≠ B ∧ ∠ACB = max_angle AB) ∧ (C lies_on_other_side O α) :=
sorry

end max_view_angle_exists_l690_690216


namespace part_I_part_II_l690_690465

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∀ x : ℝ, g x 3 > -1 ↔ x = -3) :=
by
  sorry

theorem part_II (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x a ≥ g x m) ↔ (a < 4) :=
by
  sorry

end part_I_part_II_l690_690465


namespace checkerboard_traversal_possible_l690_690244

def is_adjacent (p q : ℤ × ℤ) : Prop := 
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 = q.2 - 1)) ∨ 
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 = q.1 - 1))

def is_checkerboard_black (n : ℤ) (x y : ℤ) : Prop :=
  (x + y) % 2 = 0

def at_least_once (visited : set (ℤ × ℤ)) (n : ℤ) : Prop :=
  visited = { (i, j) | 1 <= i ∧ i <= n ∧ 1 <= j ∧ j <= n }

noncomputable def traverse_board (n : ℤ) (A B : ℤ × ℤ) : Prop :=
  ∃ (path : List (ℤ × ℤ)),
    path.head = A ∧
    path.last = B ∧
    ∀ (i : ℕ), i < path.length - 1 → is_adjacent (path.nth i).get (path.nth (i + 1)).get ∧ 
    at_least_once (path.to_finset) n

theorem checkerboard_traversal_possible : 
  ∀ (n : ℤ) (A B : ℤ × ℤ), 
     (n = 2015 ∧ is_checkerboard_black n A.1 A.2 ∧ is_checkerboard_black n B.1 B.2) →
     traverse_board n A B :=
by sorry

end checkerboard_traversal_possible_l690_690244


namespace product_even_l690_690456

theorem product_even {n : ℕ} (h : n = 2011) (a : Fin n → ℕ) (ha : Function.Injective a) (b : Fin n → ℕ) (hb : ∀ i, ∃ j, b i = a j) :
  ∃ k, k * 2 = (Finset.univ.prod (λ i, a i - b i)) :=
by
  sorry

end product_even_l690_690456


namespace exists_consecutive_equal_digits_of_prime_power_l690_690229

theorem exists_consecutive_equal_digits_of_prime_power (p : ℕ) (hp : p.prime) (n : ℕ) (hn : 2 ≤ n) :
  ∃ k : ℕ, ∃ d : ℕ, d ≠ 0 ∧ d ≤ 9 ∧ (p^k).to_digits.count d ≥ n :=
sorry

end exists_consecutive_equal_digits_of_prime_power_l690_690229


namespace problem_equivalent_l690_690964

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690964


namespace count_two_digit_primes_with_digit_sum_10_l690_690850

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690850


namespace prescribedDosageLessThanTypical_l690_690341

noncomputable def prescribedDosage : ℝ := 12
noncomputable def bodyWeight : ℝ := 120
noncomputable def typicalDosagePer15Pounds : ℝ := 2
noncomputable def typicalDosage : ℝ := (bodyWeight / 15) * typicalDosagePer15Pounds
noncomputable def percentageDecrease : ℝ := ((typicalDosage - prescribedDosage) / typicalDosage) * 100

theorem prescribedDosageLessThanTypical :
  percentageDecrease = 25 :=
by
  sorry

end prescribedDosageLessThanTypical_l690_690341


namespace volume_of_tetrahedron_l690_690603

def volume_tetrahedron (PQ PR PS QR QS RS : ℚ) : ℚ :=
  (1 / 6) * (Real.sqrt (det (λ i j, 
    match (i, j) with
    | (0, 1) | (1, 0) | (2, 0) | (3, 0) | (4, 0) => 1
    | (1, 1) => 0
    | (1, 2) => PQ ^ 2
    | (1, 3) => PR ^ 2
    | (1, 4) => PS ^ 2
    | (2, 1) | (3, 2) | (4, 3) => PR ^ 2
    | (2, 3) => QR ^ 2
    | (2, 4) => QS ^ 2
    | (3, 1) | (4, 2) => PS ^ 2
    | (3, 3) => 0
    | (3, 4) => RS ^ 2
    | (4, 3) => QR ^ 2
    | (4, 4) => 0
    | _      => 0 end)))

theorem volume_of_tetrahedron : volume_tetrahedron 6 5 4 7 5 (15/4 * Real.sqrt 2) = 10 / 3 := sorry

end volume_of_tetrahedron_l690_690603


namespace calc_two_pow_a_mul_two_pow_b_l690_690510

theorem calc_two_pow_a_mul_two_pow_b {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : (2^a)^b = 2^2) :
  2^a * 2^b = 8 :=
sorry

end calc_two_pow_a_mul_two_pow_b_l690_690510


namespace find_n_l690_690471

noncomputable def f (x : ℝ) := x * Real.log x + (1/2) * x^2 - 3 * x
noncomputable def f' (x : ℝ) := Real.log x + x - 2

theorem find_n : ∀ n : ℤ, ∃ x : ℝ, (x ∈ Ioo (n - 1/2 : ℝ) n) ∧ f' x = 0 ↔ n = 2 := by
  sorry

end find_n_l690_690471


namespace two_digit_prime_sum_to_ten_count_l690_690843

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690843


namespace count_two_digit_primes_with_digit_sum_10_l690_690990

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690990


namespace valid_digit_for_multiple_of_5_l690_690430

theorem valid_digit_for_multiple_of_5 (d : ℕ) (h : d < 10) : (45670 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 :=
by
  sorry

end valid_digit_for_multiple_of_5_l690_690430


namespace find_whole_number_l690_690759

theorem find_whole_number (M : ℤ) : (9.5 < M / 5 ∧ M / 5 < 10.5) → M = 49 ∨ M = 50 ∨ M = 51 := by
  sorry

end find_whole_number_l690_690759


namespace carpet_exchange_impossible_l690_690335

noncomputable def transform1 (a b : ℝ) : ℝ × ℝ :=
(1/a, 1/b)

noncomputable def transform2 (a b c : ℝ) : ℝ × ℝ :=
(c * b, (a / c) * b)

noncomputable def transform3 (a b c : ℝ) : ℝ × ℝ :=
(c * a, (b / c) * a)

theorem carpet_exchange_impossible (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) :
  ∀ (k : ℕ) (dimensions : ℕ → ℝ × ℝ),
  (∀ i, 0 ≤ i → i < k → ((fst (dimensions i) > 1 ∧ snd (dimensions i) > 1) ∨ (fst (dimensions i) < 1 ∧ snd (dimensions i) < 1)))
  →
  ∀ (transformations : list (ℕ × (ℝ → ℝ → Option (ℝ × ℝ)))) (initial : ℕ),
  (fst (dimensions initial) = a ∧ snd (dimensions initial) = b) →
  (∀ t in transformations, ∃ i j, j < k ∧ dimensions (t.fst) = dimensions j ∨ ∃ c : ℝ, transform1 (fst (dimensions i)) (snd (dimensions i)) = dimensions j ∨ transform2 (fst (dimensions i)) (snd (dimensions i)) c = dimensions j ∨ transform3 (fst (dimensions i)) (snd (dimensions i)) c = dimensions j)
  → False :=
by
  sorry

end carpet_exchange_impossible_l690_690335


namespace modulus_correct_l690_690311

def complex_number := (1 : ℂ) - (2 : ℂ) * I

theorem modulus_correct : abs complex_number = Real.sqrt 5 :=
by
  let z := complex_number
  have h1 : complex.re z = 1 := rfl
  have h2 : complex.im z = -2 := rfl
  calc
    abs z = Real.sqrt ((complex.re z)^2 + (complex.im z)^2) : by simp [Complex.abs]
    ...   = Real.sqrt (1^2 + (-2)^2)                        : by rw [h1, h2]
    ...   = Real.sqrt (1 + 4)                               : by norm_num
    ...   = Real.sqrt 5                                     : by norm_num

end modulus_correct_l690_690311


namespace dice_labeling_possible_l690_690633

theorem dice_labeling_possible :
  ∃ (die1 die2 : fin 6 → nat), 
  (∀ i j, 1 ≤ die1 i + die2 j ∧ die1 i + die2 j ≤ 36) ∧ 
  (set.range (λ (x : fin 6 × fin 6), die1 x.1 + die2 x.2) = finset.range 36.succ) :=
by {
  sorry
}

end dice_labeling_possible_l690_690633


namespace expected_product_two_uniform_numbers_l690_690639

open ProbabilityTheory

noncomputable def expected_value_uniform (n : ℕ) : ℝ :=
  (Real.ofNat (n * (n + 1)) / 2) / n

theorem expected_product_two_uniform_numbers :
  let X Y : ℕ := 10,
  let E := expected_value_uniform 10
  in E * E = 30.25 :=
by
  sorry

end expected_product_two_uniform_numbers_l690_690639


namespace count_two_digit_primes_with_digit_sum_10_l690_690858

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690858


namespace minValue_at_least_9_minValue_is_9_l690_690203

noncomputable def minValue (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) : ℝ :=
  1 / a + 4 / b + 9 / c

theorem minValue_at_least_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) :
  minValue a b c h_pos h_sum ≥ 9 :=
by
  sorry

theorem minValue_is_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4)
  (h_abc : a = 2/3 ∧ b = 4/3 ∧ c = 2) : minValue a b c h_pos h_sum = 9 :=
by
  sorry

end minValue_at_least_9_minValue_is_9_l690_690203


namespace smallest_x_l690_690068

theorem smallest_x (x : Real) (hx1: (x ≥ 0 → |x| = x)) (hx2: (x < 0 → |x| = -x)) :
  (∃ x : Real, x|x| = 3x + 2 ∧ (∀ y : Real, y|y| = 3y + 2 → y ≥ x)) :=
by
  sorry

end smallest_x_l690_690068


namespace draw_perpendicular_l690_690441

-- Define the necessary geometric constructs
structure Point :=
(x : ℝ)
(y : ℝ)

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

structure Diameter (C : Circle) :=
(A : Point)
(B : Point)
(property : dist C.center A = dist C.center B = C.radius 
  ∧ ∃ l : Line, l.p1 = A ∧ l.p2 = B ∧ dist A B = 2 * C.radius)


-- Define the main theorem
theorem draw_perpendicular (C : Circle) (D : Diameter C) (M : Point) :
  ∃ l : Line, ∀ d : Line, (d.p1 = D.A ∧ d.p2 = D.B) → Line.perp l d :=
sorry

end draw_perpendicular_l690_690441


namespace conic_section_of_eq_l690_690452

open Real

theorem conic_section_of_eq (θ : ℝ) (x y : ℝ) (h1 : 0 < θ ∧ θ < π)
  (h2 : sin θ + cos θ = 3 / 4) :
  (∃ a b : ℝ, x^2 * sin θ - y^2 * cos θ = 1 ∧ a < b) → (x, y).1 axis_major_axis :=
sorry

end conic_section_of_eq_l690_690452


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690980

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690980


namespace number_of_positive_numbers_is_two_l690_690254

def positive_numbers : List ℝ := [-2, 3.5, 0, -2/3, -0.7, 11]

def count_positives (l : List ℝ) : Nat :=
  l.countp (λ x => 0 < x)

theorem number_of_positive_numbers_is_two :
  count_positives positive_numbers = 2 :=
by
  sorry

end number_of_positive_numbers_is_two_l690_690254


namespace smoothie_one_serving_ingredients_in_cups_containers_needed_l690_690408

theorem smoothie_one_serving_ingredients_in_cups :
  (0.2 + 0.1 + 0.2 + 1 * 0.125 + 2 * 0.0625 + 0.5).round = 1.25.round := sorry

theorem containers_needed :
  (5 * 1.25 / 1.5).ceil = 5 := sorry

end smoothie_one_serving_ingredients_in_cups_containers_needed_l690_690408


namespace perimeter_of_fence_l690_690635

-- Definitions based on the conditions
def num_posts : ℕ := 36
def post_width : ℝ := 4 / 12 -- in feet
def gap_length : ℝ := 3.5
def rectangular_ratio := 2 -- Length is twice the width

-- Main theorem to prove
theorem perimeter_of_fence : 
  let w := (num_posts - 4) / (2 * rectangular_ratio + 2), 
      l := rectangular_ratio * w,
      num_gaps := num_posts - 4,
      perimeter := (l + w) * 2 + num_gaps * gap_length in
  perimeter = 238 :=
sorry

end perimeter_of_fence_l690_690635


namespace two_digit_primes_with_digit_sum_10_count_l690_690902

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690902


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690913

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690913


namespace train_length_l690_690716

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end train_length_l690_690716


namespace total_cost_of_square_park_l690_690706

-- Define the cost per side and number of sides
def cost_per_side : ℕ := 56
def sides_of_square : ℕ := 4

-- The total cost of fencing the park
def total_cost_of_fencing (cost_per_side : ℕ) (sides_of_square : ℕ) : ℕ := cost_per_side * sides_of_square

-- The statement we need to prove
theorem total_cost_of_square_park : total_cost_of_fencing cost_per_side sides_of_square = 224 :=
by sorry

end total_cost_of_square_park_l690_690706


namespace find_p_l690_690005

-- Define the conditions for the problem.
-- Random variable \xi follows binomial distribution B(n, p).
axiom binomial_distribution (n : ℕ) (p : ℝ) : Type
variables (ξ : binomial_distribution n p)

-- Given conditions: Eξ = 300 and Dξ = 200.
axiom Eξ (ξ : binomial_distribution n p) : ℝ
axiom Dξ (ξ : binomial_distribution n p) : ℝ

-- Given realizations of expectations and variance.
axiom h1 : Eξ ξ = 300
axiom h2 : Dξ ξ = 200

-- Prove that p = 1/3
theorem find_p (n : ℕ) (p : ℝ) (ξ : binomial_distribution n p)
  (h1 : Eξ ξ = 300) (h2 : Dξ ξ = 200) : p = 1 / 3 :=
sorry

end find_p_l690_690005


namespace find_function_l690_690416

theorem find_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + c * x) :=
by
  -- The proof details will be filled here.
  sorry

end find_function_l690_690416


namespace two_digit_prime_sum_digits_10_count_l690_690938

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690938


namespace equation_solutions_equivalence_l690_690190

theorem equation_solutions_equivalence {n k : ℕ} (hn : 1 < n) (hk : 1 < k) (hnk : n > k) :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^(n - k)) :=
by
  sorry

end equation_solutions_equivalence_l690_690190


namespace count_prime_two_digit_sum_ten_is_three_l690_690810

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690810


namespace largest_sum_of_digits_l690_690508

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) 
  (h_digits : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_fractions : 0.abc = 1 / y)
  (h_y_bounds : 0 < y ∧ y ≤ 16) : (a + b + c) ≤ 8 :=
sorry

end largest_sum_of_digits_l690_690508


namespace min_possible_b1_b2_sum_l690_690273

theorem min_possible_b1_b2_sum :
  ∃ (b_1 b_2 : ℕ), (∀ n : ℕ, n ≥ 1 → 
    b_(n + 2) = (b_n + 2021) / (1 + b_(n + 1))) ∧
    (∀ i : ℕ, b_i > 0) ∧
    b_1 + b_2 = 90 := sorry

end min_possible_b1_b2_sum_l690_690273


namespace g_2009_l690_690107

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f(x) + 1 - x

axiom f_defined : ∀ x : ℝ, ∃ y : ℝ, f x = y
axiom f_at_1 : f 1 = 1
axiom f_increase_by_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_increase_by_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

theorem g_2009 : g 2009 = 1 :=
by
  sorry

end g_2009_l690_690107


namespace quadratic_has_single_solution_l690_690052

theorem quadratic_has_single_solution (q : ℚ) (h : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 16 * x + 9 = 0 → q = 64 / 9) := by
  sorry

end quadratic_has_single_solution_l690_690052


namespace total_area_needed_l690_690354

-- Definitions based on conditions
def oak_trees_first_half := 100
def pine_trees_first_half := 100
def oak_trees_second_half := 150
def pine_trees_second_half := 150
def oak_tree_planting_ratio := 4
def pine_tree_planting_ratio := 2
def oak_tree_space := 4
def pine_tree_space := 2

-- Total area needed for tree planting during the entire year
theorem total_area_needed : (oak_trees_first_half * oak_tree_planting_ratio * oak_tree_space) + ((pine_trees_first_half + pine_trees_second_half) * pine_tree_planting_ratio * pine_tree_space) = 2600 :=
by
  sorry

end total_area_needed_l690_690354


namespace students_left_correct_l690_690530

-- Define the initial number of students
def initial_students : ℕ := 8

-- Define the number of new students
def new_students : ℕ := 8

-- Define the final number of students
def final_students : ℕ := 11

-- Define the number of students who left during the year
def students_who_left : ℕ :=
  (initial_students + new_students) - final_students

theorem students_left_correct : students_who_left = 5 :=
by
  -- Instantiating the definitions
  let initial := initial_students
  let new := new_students
  let final := final_students

  -- Calculation of students who left
  let L := (initial + new) - final

  -- Asserting the result
  show L = 5
  sorry

end students_left_correct_l690_690530


namespace range_independent_variable_l690_690176

theorem range_independent_variable (x : ℝ) (h : x + 1 > 0) : x > -1 :=
sorry

end range_independent_variable_l690_690176


namespace constant_term_of_A_plus_B_l690_690435

noncomputable def A (x m : ℝ) := 3 * x^3 + 2 * x^2 - 5 * x + 7 * m + 2
noncomputable def B (x m : ℝ) := 2 * x^2 + m * x - 3

theorem constant_term_of_A_plus_B (x : ℝ) (m : ℝ) (h : (A x m + B x m).coeff 1 = 0) : (A x 5 + B x 5).coeff 0 = 34 :=
by
  sorry

end constant_term_of_A_plus_B_l690_690435


namespace G_odd_inequality_solution_l690_690204

noncomputable def F (x : ℝ) : ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := sorry

variables {F_cond : (∀ a b : ℝ, F(a * b) = F(a) * F(b))} 
variables {F_2 : F(2) = 8} 
variables {f_odd : ∀ x : ℝ, f(-x) = -f(x)} 
variables {f_even_shifted : ∀ x : ℝ, f(-x + 2) = f(x + 2)} 
variables {f_F_restricted : ∀ x ∈ [-2, 2], f(x) = F(x)}

def G (x : ℝ) := f(x + 4)

theorem G_odd (x : ℝ) : G(-x) = -G(x) := sorry

theorem inequality_solution (x k : ℤ) (h : -5 + 8 * k ≤ x ∧ x ≤ 1 + 8 * k) : f(x) ≤ 1 := sorry

end G_odd_inequality_solution_l690_690204


namespace amount_exceeds_l690_690511

theorem amount_exceeds (N : ℕ) (A : ℕ) (h1 : N = 1925) (h2 : N / 7 - N / 11 = A) :
  A = 100 :=
sorry

end amount_exceeds_l690_690511


namespace perpendicular_bisector_AB_l690_690062

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 2⟩
def B : Point := ⟨4, 0⟩

noncomputable def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

noncomputable def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

noncomputable def perpendicular_bisector (P Q : Point) : (ℝ × ℝ × ℝ) :=
  let m := midpoint P Q
  let k := -1 / slope P Q
  (k, -1, k * m.x - m.y)

theorem perpendicular_bisector_AB :
  perpendicular_bisector A B = (2, -1, -3) :=
    sorry

end perpendicular_bisector_AB_l690_690062


namespace percentage_of_students_with_life_vests_l690_690369

theorem percentage_of_students_with_life_vests (total_students : ℕ) (total_instructors : ℕ) (on_hand_life_vests : ℕ) (needed_life_vests : ℕ) : 
  total_students = 40 → 
  total_instructors = 10 → 
  on_hand_life_vests = 20 →
  needed_life_vests = 22 →
  (∀ life_vests_brought : ℕ, 50 - (on_hand_life_vests + needed_life_vests) = life_vests_brought) →
  (life_vests_brought / total_students : ℚ) * 100 = 20 := 
by
  intros h1 h2 h3 h4 h5
  have h6 : total_students + total_instructors = 50, sorry
  have h7 : on_hand_life_vests + needed_life_vests = 42, sorry
  have h8 : 50 - 42 = 8, sorry
  have h9 : 8 / 40 = 0.2, sorry
  have h10 : 0.2 * 100 = 20, sorry
  exact sorry

end percentage_of_students_with_life_vests_l690_690369


namespace two_digit_primes_with_digit_sum_ten_l690_690804

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690804


namespace count_two_digit_primes_with_digit_sum_10_l690_690998

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690998


namespace height_of_carton_l690_690752

def widgets_per_carton := 3
def carton_width := 4
def carton_length := 4
def shipping_box_width := 20
def shipping_box_length := 20
def shipping_box_height := 20
def widgets_per_shipping_box := 300

theorem height_of_carton :
∀ (widgets_per_carton : ℕ) (carton_width carton_length : ℕ) (shipping_box_width shipping_box_length shipping_box_height : ℕ) (widgets_per_shipping_box : ℕ),
widgets_per_carton = 3 →
carton_width = 4 →
carton_length = 4 →
shipping_box_width = 20 →
shipping_box_length = 20 →
shipping_box_height = 20 →
widgets_per_shipping_box = 300 →
∃ (carton_height : ℕ), carton_height = 5 :=
by
  intros widgets_per_carton carton_width carton_length shipping_box_width shipping_box_length shipping_box_height widgets_per_shipping_box
  intros h1 h2 h3 h4 h5 h6 h7
  use 5
  sorry

end height_of_carton_l690_690752


namespace part1_part2_part3_l690_690439

variables {a b c : ℝ}
noncomputable def f (x : ℝ) := a * x^2 + b * x + c
noncomputable def g (x : ℝ) := a * x + b

theorem part1 (h : ∀ (x : ℝ), |x| ≤ 1 → |f x| ≤ 1) : |c| ≤ 1 :=
sorry

theorem part2 (h : ∀ (x : ℝ), |x| ≤ 1 → |f x| ≤ 1) : ∀ (x : ℝ), |x| ≤ 1 → |g x| ≤ 2 :=
sorry

theorem part3 (h : ∀ (x : ℝ), |x| ≤ 1 → |f x| ≤ 1) (h_a_pos : a > 0) (h_max_g : ∀ (x : ℝ), |x| ≤ 1 → g x ≤ 2) : f = λ x, 2 * x^2 - 1 :=
sorry

end part1_part2_part3_l690_690439


namespace sum_of_even_proper_divisors_1728_l690_690684

theorem sum_of_even_proper_divisors_1728 (n : ℕ) (h1 : n = 1728) : 
  ∑ d in (finset.filter (λ x, even x ∧ x < n) (finset.divisors n)), d = 126 :=
by
  sorry

end sum_of_even_proper_divisors_1728_l690_690684


namespace base_of_triangle_eq_l690_690245

-- Define the radii of the circles
variables (R r : ℝ)

-- Assume the given conditions
axiom circles_touch_externally : true
axiom isosceles_triangle_common_tangents : true
axiom base_touches_larger_circle : true

-- State the theorem we want to prove
theorem base_of_triangle_eq : 
  circles_touch_externally ∧
  isosceles_triangle_common_tangents ∧
  base_touches_larger_circle → 
  (∃ b : ℝ, b = (2 * R ^ 2) / (Real.sqrt (R * r))) :=
sorry

end base_of_triangle_eq_l690_690245


namespace largest_sum_of_digits_l690_690507

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) 
  (h_digits : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9)
  (h_fractions : 0.abc = 1 / y)
  (h_y_bounds : 0 < y ∧ y ≤ 16) : (a + b + c) ≤ 8 :=
sorry

end largest_sum_of_digits_l690_690507


namespace two_digit_primes_with_digit_sum_10_count_l690_690895

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690895


namespace count_two_digit_primes_with_digit_sum_10_l690_690888

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690888


namespace finalAnswer_l690_690667

open Real

section propositions

-- Proposition A: The function f(x) = ln(x) - 2/x has a root in the interval (2, 3)
def PropA : Prop :=
  ∃ x ∈ Ioo 2 3, ln x - 2 / x = 0

-- Proposition B: ∀x ∈ ℝ, x^(4/3) ≥ 0
def PropB : Prop :=
  ∀ x : ℝ, x ^ (4 / 3) ≥ 0

-- Proposition C: Given a > 0 and b > 0, and a + b = 1, then sqrt(a) + sqrt(b) ≤ sqrt(2)
def PropC (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : Prop :=
  sqrt a + sqrt b ≤ sqrt 2

-- Proposition D: If the chord length of a central angle of 2 radians is 4, then the arc length is 2 / sin(1)
def PropD (R : ℝ) (hR : 2 * R * sin 1 = 4) : Prop :=
  2 * R = 2 / sin 1

end propositions

-- Combine the propositions to summarize the final answer
theorem finalAnswer : PropA ∧ PropB ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → PropC a b) ∧ ¬ PropD :=
by sorry

end finalAnswer_l690_690667


namespace tasty_permutations_eq_partitions_l690_690589

def is_tasty {α : Type*} (perm : list α) : Prop :=
  ∀ (i j : ℕ), i < j ∧ j < perm.length - 1 → ¬ (perm.nth i < perm.nth (j + 1) ∧ perm.nth (j + 1) < perm.nth j)

theorem tasty_permutations_eq_partitions (n : ℕ) :
  (finset.univ.filter (λ (perm : list (fin n)), is_tasty perm)).card = 
  (finset.univ.filter multiset.is_partition).card :=
sorry

end tasty_permutations_eq_partitions_l690_690589


namespace range_of_a_for_two_critical_points_l690_690469

theorem range_of_a_for_two_critical_points (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧
   (λ x => x * Real.exp x - a * Real.exp (2 * x))'.critical_points x1 x2) ↔
  0 < a ∧ a < 1/2 :=
by
  sorry

end range_of_a_for_two_critical_points_l690_690469


namespace train_length_proof_l690_690717

-- Define speeds and time taken
def speed_train_kmph : ℝ := 63
def speed_man_kmph : ℝ := 3
def time_crossing_seconds : ℝ := 41.9966402687785

-- Speed conversion factor
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train_kmph - speed_man_kmph)

-- Length of the train
def length_of_train : ℝ := relative_speed_mps * time_crossing_seconds

-- Proof stating the length of the train is approximately 699.94400447975 meters
theorem train_length_proof : abs (length_of_train - 699.94400447975) < 1e-6 := by
  sorry

end train_length_proof_l690_690717


namespace determine_d_div_b_l690_690121

-- Given function definition
def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

-- The function f(x) is even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- The solution for f(x) < 0 is (-2, 2)
def solution_set (f : ℝ → ℝ) : set ℝ := {x | f x < 0}
def solution_interval (f : ℝ → ℝ) : Prop :=
  solution_set f = set.Ioo (-2 : ℝ) 2

theorem determine_d_div_b (a b c d : ℝ) (h : is_even_function (λ x, f x))
  (h_sol : solution_interval (λ x, f x)) : d / b = -4 :=
sorry

end determine_d_div_b_l690_690121


namespace natural_numbers_divisor_square_l690_690417

-- Definition: number of divisors of a natural number
def num_divisors (n : ℕ) : ℕ :=
  if h : n > 0 then (finset.range (n+1)).filter (λ d, n % d = 0).card else 0

-- Main statement: n = (num_divisors(n))^2 implies n = 1 or n = 9
theorem natural_numbers_divisor_square :
  ∀ n : ℕ, n = (num_divisors n) ^ 2 ↔ n = 1 ∨ n = 9 :=
by
  intros n
  sorry

end natural_numbers_divisor_square_l690_690417


namespace two_digit_prime_sum_to_ten_count_l690_690841

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690841


namespace num_two_digit_primes_with_digit_sum_10_l690_690823

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690823


namespace coffee_mix_total_eq_100_l690_690548

noncomputable def total_coffee_mix (B : ℝ) : ℝ :=
  let columbian_coffee := 52
  let price_columbian := 8.75
  let price_brazilian := 3.75
  let price_mix := 6.35
  let total_weight := columbian_coffee + B
  let total_cost := columbian_coffee * price_columbian + B * price_brazilian
  let target_cost := total_weight * price_mix
  if total_cost = target_cost then total_weight else 0

theorem coffee_mix_total_eq_100 : total_coffee_mix 48 = 100 :=
by
  unfold total_coffee_mix
  have h1 : (52 : ℝ) * 8.75 = 455, by norm_num
  have h2 : (48 : ℝ) * 3.75 = 180, by norm_num
  have h3 : (52 : ℝ) + 48 = 100, by norm_num
  have h4 : 100 * 6.35 = 635, by norm_num
  have h5 : 455 + 180 = 635, by norm_num
  simp [h1, h2, h3, h4, h5]
  exact rfl

end coffee_mix_total_eq_100_l690_690548


namespace average_multiples_4_2_5_between_15_55_l690_690021

theorem average_multiples_4_2_5_between_15_55 : 
  (∃ (l : List ℕ), (∀ x ∈ l, 15 < x ∧ x < 55 ∧ x % 4 = 0 ∧ ∃ p q, ∃ (prime_p : Nat.Prime p) (prime_q : Nat.Prime q), p * q = 10 ∧ Nat.gcd x 20 = 20) ∧ (l.sum / l.length = 30)) := 
begin
  sorry
end

end average_multiples_4_2_5_between_15_55_l690_690021


namespace total_pieces_is_100_l690_690277

-- Definitions based on conditions
def total_pieces_of_bread (B : ℕ) : Prop :=
  let duck1 := B / 2 in      -- The first duck eats half of all the pieces of bread
  let duck2 := 13 in         -- The second duck eats 13 pieces
  let duck3 := 7 in          -- The third duck eats 7 pieces
  let left_in_water := 30 in -- There are 30 pieces left in the water
  duck1 + duck2 + duck3 + left_in_water = B

-- The statement to be proved
theorem total_pieces_is_100 (B : ℕ) : total_pieces_of_bread B → B = 100 :=
by
  -- Proof would be provided here
  sorry

end total_pieces_is_100_l690_690277


namespace distinct_terms_in_sequence_l690_690033

/-- The function f is defined as the floor of k squared divided by 1998. -/
def f (k : ℕ) := (k^2 / 1998 : ℕ)

theorem distinct_terms_in_sequence :
  (set.range (λ k, f k)).to_finset.card = 1498 :=
sorry

end distinct_terms_in_sequence_l690_690033


namespace Vasya_chips_placement_l690_690219

theorem Vasya_chips_placement :
  ∀ (board : Fin 50 → Fin 50 → Prop), 
  (∀ i j, board i j → i < 50 ∧ j < 50) → 
  ∃ (placement : Fin 50 → Fin 50 → Prop), 
  (∀ i j, (board i j ∨ placement i j) → (i < 50 ∧ j < 50)) ∧
  (∑ i, ∑ j, if board i j ∨ placement i j then 1 else 0 ≤ 99) ∧
  (∀ i, ∑ j, if board i j ∨ placement i j then 1 else 0 % 2 = 0) ∧
  (∀ j, ∑ i, if board i j ∨ placement i j then 1 else 0 % 2 = 0) :=
sorry

end Vasya_chips_placement_l690_690219


namespace min_dot_product_l690_690108

-- Define the parameters of the ellipse
def a : ℝ := 2
def b : ℝ := 1
def c : ℝ := sqrt (a^2 - b^2)

-- Define the coordinates of the foci F1 and F2
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)

-- Define the coordinates of the vertices A and B
def A : ℝ × ℝ := (-a, 0)
def B : ℝ × ℝ := (0, b)

-- Define an arbitrary point P on the line segment AB
noncomputable def P (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) : ℝ × ℝ := 
  (-a * t, b * t)

-- Define the vectors PF1 and PF2
noncomputable def PF1 (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) : ℝ × ℝ :=
  let P_t := P t ht in
  ((fst P_t) - (fst F1), (snd P_t) - (snd F1))

noncomputable def PF2 (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) : ℝ × ℝ :=
  let P_t := P t ht in
  ((fst P_t) - (fst F2), (snd P_t) - (snd F2))

-- Define the dot product of PF1 and PF2
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (fst v1) * (fst v2) + (snd v1) * (snd v2)

-- Define the expression for the dot product of PF1 and PF2
noncomputable def dot_product_PF1_PF2 (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1) : ℝ :=
  dot_product (PF1 t ht) (PF2 t ht)

-- Lean statement to prove that the minimum value of the dot product is -11/5
theorem min_dot_product : ∀ (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1), dot_product_PF1_PF2 t ht = -11 / 5 :=
  sorry

end min_dot_product_l690_690108


namespace two_digit_primes_with_digit_sum_10_count_l690_690906

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690906


namespace problem_equivalent_l690_690968

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690968


namespace num_two_digit_primes_with_digit_sum_10_l690_690787

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690787


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690948

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690948


namespace carB_travel_time_l690_690640

noncomputable def travelTimeCarB : ℕ :=
  let t1 := 3 * 60 in     -- Time for the first part (3 hours)
  let t2 := 3.5 * 60 in   -- Time for the second part (from meeting point to point D)
  let t3 := 0.7 * 60 in   -- Time for the final part to cover AD
  t1 + t2 + t3            -- Total time in minutes

theorem carB_travel_time :
  travelTimeCarB = 432 := by
  sorry

end carB_travel_time_l690_690640


namespace find_num_terms_in_AP_l690_690527

-- Define the necessary conditions and prove the final result
theorem find_num_terms_in_AP
  (a d : ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h_last_term_difference : (n - 1 : ℝ) * d = 7.5)
  (h_sum_odd_terms : n * (a + (n - 2 : ℝ) / 2 * d) = 60)
  (h_sum_even_terms : n * (a + ((n - 1 : ℝ) / 2) * d + d) = 90) :
  n = 12 := 
sorry

end find_num_terms_in_AP_l690_690527


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690912

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690912


namespace count_triangles_in_3x3_grid_l690_690142

/--
In a 3x3 grid of dots, the number of triangles formed by connecting the dots is 20.
-/
def triangles_in_3x3_grid : Prop :=
  let num_rows := 3
  let num_cols := 3
  let total_triangles := 20
  ∃ (n : ℕ), n = total_triangles ∧ n = 20

theorem count_triangles_in_3x3_grid : triangles_in_3x3_grid :=
by {
  -- Insert the proof here
  sorry
}

end count_triangles_in_3x3_grid_l690_690142


namespace two_digit_primes_with_digit_sum_10_count_l690_690901

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690901


namespace problem_equivalent_l690_690962

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690962


namespace count_two_digit_primes_with_digit_sum_10_l690_690894

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690894


namespace min_max_f_l690_690029

-- Defining the set A
def A (a : ℤ) : set (ℤ × ℤ) :=
  {p : ℤ × ℤ | 
    let m := p.1, n := p.2 in
    m < 2 * a ∧
    2 * n ∣ (2 * a * m - m^2 + n^2) ∧
    n^2 - m^2 + 2 * m * n ≤ 2 * a * (n - m)}

-- The function f
def f (a : ℤ) (m n : ℤ) : ℤ :=
  (2 * a * m - m^2 - m * n) / n

-- The main theorem
theorem min_max_f (a : ℤ) (h : a = 2001) :
  ∃ (min_f max_f : ℤ),
    (∀ (p : ℤ × ℤ), p ∈ A a → f a p.1 p.2 ≥ min_f) ∧
    (∀ (p : ℤ × ℤ), p ∈ A a → f a p.1 p.2 ≤ max_f) ∧
    min_f = 2 ∧
    max_f = 3750 :=
sorry

end min_max_f_l690_690029


namespace radius_of_inscribed_semicircle_l690_690359

theorem radius_of_inscribed_semicircle (R : ℝ) (hR : R = 2) : ∃ r : ℝ, r = Real.sqrt 2 ∧ 
  ∀ r' : ℝ, (R - r')^2 + r'^2 = R^2 → r' = Real.sqrt 2 :=
by
  intros
  use Real.sqrt 2
  split
  {
    refl
  }
  {
    intros r' hr'
    have hr : R = 2 := by assumption
    rw [hr] at hr'
    simp at hr'
    sorry -- Here, you would complete the proof by solving the equation.
  }

end radius_of_inscribed_semicircle_l690_690359


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690915

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690915


namespace find_ab_l690_690152

theorem find_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end find_ab_l690_690152


namespace angle_between_vectors_l690_690105

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2)
variables (hb : ‖b‖ = 1)
variables (hab : ‖a - 3 • b‖ = Real.sqrt 7)

theorem angle_between_vectors : Real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖)) = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l690_690105


namespace probability_cond_satisfied_l690_690565

-- Define the floor and log conditions
def cond1 (x : ℝ) : Prop := ⌊Real.log x / Real.log 2 + 1⌋ = ⌊Real.log x / Real.log 2⌋
def cond2 (x : ℝ) : Prop := ⌊Real.log (2 * x) / Real.log 2 + 1⌋ = ⌊Real.log (2 * x) / Real.log 2⌋
def valid_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Main theorem stating the proof problem
theorem probability_cond_satisfied : 
  (∀ (x : ℝ), valid_interval x → cond1 x → cond2 x → x ∈ Set.Icc (0.25:ℝ) 0.5) → 
  (0.5 - 0.25) / 1 = 1 / 4 := 
by
  -- Proof omitted
  sorry

end probability_cond_satisfied_l690_690565


namespace calculation_l690_690379

theorem calculation : 120 / 5 / 3 * 2 = 16 := by
  sorry

end calculation_l690_690379


namespace count_two_digit_primes_with_digit_sum_10_l690_690996

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690996


namespace geometric_sequence_first_term_l690_690274

-- Define factorial values for convenience
def fact (n : ℕ) : ℕ := Nat.factorial n
#eval fact 6 -- This should give us 720
#eval fact 7 -- This should give us 5040

-- State the hypotheses and the goal
theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^5 = 5040) :
  a = 720 / (7^(2/3 : ℝ)) :=
by
  sorry

end geometric_sequence_first_term_l690_690274


namespace length_of_PT_l690_690689

/--
Let circles O and P have radii 10 units and 3 units respectively, and they are externally tangent to each other at point Q. Segment TS is the common external tangent at points T and S, respectively. Prove that the length of PT is √69.
-/
theorem length_of_PT :
  ∀ (O P T S Q : Type) (rO rP : ℝ),  rO = 10 ∧ rP = 3 ∧
  ∃ (centerO centerP : O), ∃ (pointQ : Q), 
  (dist centerO centerP = 13) ∧ (dist pointQ centerP = 3) ∧  
  (dist centerO pointQ = 10) ∧
  -- TS is a common external tangent
  ∃ (pointT pointS: T), dist pointT pointT = 0 ∧ dist pointS pointS = 0 →
  dist pointT pointS = sqrt 69 := 
sorry

end length_of_PT_l690_690689


namespace sqrt_53_between_and_product_l690_690629

theorem sqrt_53_between_and_product : 
  let n := 7 in let m := 8 in
  (n * n < 53) ∧ (53 < m * m) ∧ (n + 1 = m) →
  (∃ p q, (p * q = 56) ∧ (p = 7) ∧ (q = 8)) :=
  by
  intros h,
  use 7,
  use 8,
  sorry

end sqrt_53_between_and_product_l690_690629


namespace base8_to_base10_conversion_l690_690386

theorem base8_to_base10_conversion : 
  let n := 4523 in (4 * 8^3 + 5 * 8^2 + 2 * 8^1 + 3 * 8^0) = 2387 :=
by
  rfl

end base8_to_base10_conversion_l690_690386


namespace surface_area_of_circumscribed_sphere_l690_690073

theorem surface_area_of_circumscribed_sphere :
  let a := 2
  let AD := Real.sqrt (a^2 - (a/2)^2)
  let r := Real.sqrt (1 + 1 + AD^2) / 2
  4 * Real.pi * r^2 = 5 * Real.pi := by
  sorry

end surface_area_of_circumscribed_sphere_l690_690073


namespace count_two_digit_primes_with_digit_sum_10_l690_690992

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690992


namespace sum_of_k_in_base_neg4_i_l690_690445

theorem sum_of_k_in_base_neg4_i {n : ℕ} (h : n = 4) :
  ∑ k in {k : ℕ | ∃ (a_2 a_1 a_0 : ℕ), a_2 ≠ 0 ∧ 
                         a_2 ∈ (finset.range 17) ∧
                         a_1 = 4 * a_2 ∧
                         a_1 ∈ (finset.range 17) ∧
                         a_0 ∈ (finset.range 17) ∧
                         k = a_2 * ((-4+i)^2).re + a_1 * (-4+i).re + a_0}
  , k = 544 :=
begin
  sorry,
end

end sum_of_k_in_base_neg4_i_l690_690445


namespace angle_ABC_measure_l690_690173

-- Define the basic parameters of the problem
variable (A B C : Type) [angle : A → A → A → ℝ]

-- Given conditions:
-- Triangle is isosceles with AB = BC
variable (AB_eq_BC : (AB = BC))

-- Angle at BAC is t degrees
variable (t : ℝ)
variable (angle_BAC : angle A B C = t)

-- Define the theorem to prove
theorem angle_ABC_measure (ABC_isosceles : isosceles A B C) : angle A B C = 180 - 2 * t :=
by sorry

end angle_ABC_measure_l690_690173


namespace coefficient_x3y_in_expansion_of_2x_minus_y_power_4_l690_690175

theorem coefficient_x3y_in_expansion_of_2x_minus_y_power_4 :
  ∃ (r : ℕ), r = 1 ∧ 
    ∃ (C : ℤ), 
    (4.choose r) * (2 ^ (4 - r)) * (-1)^r = C ∧ 
    C = -32 :=
by
  sorry

end coefficient_x3y_in_expansion_of_2x_minus_y_power_4_l690_690175


namespace petya_cannot_prevent_vasya_from_winning_l690_690336

/-- 
  Proposition: Petya cannot prevent Vasya from winning.
  Conditions:
  - A circle is divided into 99 equal arcs by 99 points.
  - Petya and Vasya take turns coloring uncolored points adjacent to already colored points.
  - Petya colors first.
  - Vasya wins if an equilateral triangle with uniformly colored vertices is formed after coloring all points.
-/
theorem petya_cannot_prevent_vasya_from_winning :
  ∀ (circle : Type) (points : Fin 99 → circle) (color : circle → Prop),
  (∀ p : circle, color p → p ∈ points) →
  (∃ p₁ p₂ : circle, color p₁ ∧ color p₂ ∧ p₁ ≠ p₂ ∧ eq_triangle p₁ p₂ color) :=
begin
  sorry
end

end petya_cannot_prevent_vasya_from_winning_l690_690336


namespace two_digit_primes_with_digit_sum_ten_l690_690800

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690800


namespace sin_sum_values_l690_690088

-- Definitions of the given conditions
def COS_ALPHA_PLUS_COS_BETA: ℝ := 1/4 * Real.sqrt 2
def TAN_ALPHA_PLUS_BETA: ℝ := 4 / 3

-- The proof task
theorem sin_sum_values (α β : ℝ) (h1 : Real.cos α + Real.cos β = COS_ALPHA_PLUS_COS_BETA)
  (h2 : Real.tan (α + β) = TAN_ALPHA_PLUS_BETA) :
  Real.sin α + Real.sin β = -Real.sqrt 2 / 2 ∨ Real.sin α + Real.sin β = Real.sqrt 2 / 8 :=
sorry

end sin_sum_values_l690_690088


namespace stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l690_690646

theorem stratified_sampling_number_of_boys (total_students : Nat) (num_girls : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : num_girls = 50) (h3 : selected_students = 25) :
  (total_students - num_girls) * selected_students / total_students = 15 :=
  sorry

theorem stratified_sampling_probability_of_boy (total_students : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : selected_students = 25) :
  selected_students / total_students = 1 / 5 :=
  sorry

end stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l690_690646


namespace find_a_of_power_function_l690_690126

noncomputable def power_function (a : ℝ) := λ x : ℝ, x ^ a

theorem find_a_of_power_function :
  ∃ a : ℝ, power_function a (1/2) = (Real.sqrt 2) / 2 ∧ a = 1/2 :=
by
  use 1/2
  simp [power_function]
  sorry

end find_a_of_power_function_l690_690126


namespace calc_15_op_and_op2_l690_690397

def op1 (x : ℤ) : ℤ := 10 - x
def op2 (x : ℤ) : ℤ := x - 10

theorem calc_15_op_and_op2 :
  op2 (op1 15) = -15 :=
by
  sorry

end calc_15_op_and_op2_l690_690397


namespace find_a_l690_690625

-- Define the coordinates of the points
def M : ℝ × ℝ := (-2, a)
def N : ℝ × ℝ := (a, 4)

-- Define the condition that the slope is 1
def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
def condition (a : ℝ) : Prop := slope M N = 1

-- Prove that a = 1 given the condition
theorem find_a (a : ℝ) (h : condition a) : a = 1 :=
sorry

end find_a_l690_690625


namespace teal_sales_l690_690533

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l690_690533


namespace two_digit_prime_sum_to_ten_count_l690_690849

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690849


namespace count_two_digit_primes_with_digit_sum_10_l690_690853

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690853


namespace vector_magnitude_proof_l690_690098

variable (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom cond1 : ‖a‖ = Real.sqrt 10
axiom cond2 : ‖b‖ = 2
axiom cond3 : (2 • a + b) ⬝ (a - b) = 14

-- Proof statement
theorem vector_magnitude_proof : ‖a + b‖ = 3 * Real.sqrt 2 :=
by {
  -- Use the given conditions here
  sorry
}

end vector_magnitude_proof_l690_690098


namespace paper_clips_in_two_cases_l690_690688

theorem paper_clips_in_two_cases (c b : ℕ) : 
  2 * c * b * 200 = 2 * (c * b * 200) :=
by
  sorry

end paper_clips_in_two_cases_l690_690688


namespace count_prime_two_digit_sum_ten_is_three_l690_690808

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690808


namespace two_digit_primes_with_digit_sum_ten_l690_690801

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690801


namespace smallest_integer_greater_than_20_with_remainder_2_l690_690306

theorem smallest_integer_greater_than_20_with_remainder_2 :
  ∃ m : ℕ, 20 < m ∧ 
           (m % 3 = 2) ∧ 
           (m % 4 = 2) ∧ 
           (m % 5 = 2) ∧ 
           (m % 6 = 2) ∧ 
           ∀ n : ℕ, 20 < n ∧ 
                    (n % 3 = 2) ∧ 
                    (n % 4 = 2) ∧ 
                    (n % 5 = 2) ∧ 
                    (n % 6 = 2) →
                    m ≤ n :=
begin
  existsi 62,
  split,
  { exact by norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  intros n h1 h2 h3 h4 h5,
  cases h1,
  cases h2,
  cases h3,
  cases h4,
  cases h5,
  sorry
end

end smallest_integer_greater_than_20_with_remainder_2_l690_690306


namespace triangle_sides_l690_690179

theorem triangle_sides
    (A B C P E D : Type*) -- Points
    [linear_ordered_field ℝ] -- Reals
    (dist : A → A → ℝ) [metric_space A dist] -- Distance function
    (BE : B → E) -- Angle bisector from B to E
    (AD : A → D) -- Median from A to D
    (h_perpendicular : ∀ (x y : A), dist x y = 60 → BE x = AD y) -- Perpendicular and equal length condition 
    (ha : dist A B = 15 * real.sqrt 13)
    (hb : dist B C = 30 * real.sqrt 13)
    (hc : dist A C = 45 * real.sqrt 5) :
    ∃ (a b c : ℝ), a = dist A B ∧ b = dist B C ∧ c = dist A C ∧ a = 15 * real.sqrt 13 ∧ b = 30 * real.sqrt 13 ∧ c = 45 * real.sqrt 5 := 
sorry

end triangle_sides_l690_690179


namespace log_eq_exp_solution_l690_690049

theorem log_eq_exp_solution (y : ℝ) : log 16 (3 * y - 4) = 2 → y = 260 / 3 :=
by
  intro h
  sorry

end log_eq_exp_solution_l690_690049


namespace two_digit_prime_sum_digits_10_count_l690_690927

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690927


namespace sum_of_decimals_l690_690027

theorem sum_of_decimals : (5.76 + 4.29 = 10.05) :=
by
  sorry

end sum_of_decimals_l690_690027


namespace nonnegative_solution_count_l690_690141

theorem nonnegative_solution_count: 
  (set.univ.filter (λ x: ℝ, x^2 = -6*x ∧ 0 ≤ x)).card = 1 :=
sorry

end nonnegative_solution_count_l690_690141


namespace ratio_AB_AC_l690_690035

-- Define the given conditions in Lean
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

variables (angle : A → B → ℝ)
variables (intersect_at_D : A → C → D)
variables (intersect_at_E : C → A → B → E)
variables (ratio : E → D → ℝ)

-- Given conditions converted to definitions
def angle_A_eq_2_angle_B (A B : A) : Prop :=
  angle A = 2 * angle B

def angle_bisectors_intersect_at_D (A C D : A) : Prop := 
  intersect_at_D A C = D

def angle_bisector_C_intersects_AB_at_E (C A B E : A) : Prop := 
  intersect_at_E C A B = E

def DE_eq_1_3_DC (DE DC : E) : Prop :=
  ratio DE DC = 1 / 3

-- Statement of the proof problem in Lean
theorem ratio_AB_AC {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  (angle : A → B → ℝ) (intersect_at_D : A → C → D) (intersect_at_E : C → A → B → E) (ratio : E → D → ℝ)
  (h1 : angle_A_eq_2_angle_B A B)
  (h2 : angle_bisectors_intersect_at_D A C D)
  (h3 : angle_bisector_C_intersects_AB_at_E C A B E)
  (h4 : DE_eq_1_3_DC DE DC) :
  AB / AC = 7 / 9 :=
by
  sorry

end ratio_AB_AC_l690_690035


namespace PQRS_is_bicentric_l690_690611

variables {A B C D F P Q R S : Type} 

-- Define cyclic quadrilateral
def cyclic_quadrilateral (A B C D : Type) : Prop :=
  ∃ (circle : Type) (O : circle) (r : real), -- There exists a circle with center O and radius r
  (A ∈ circle) ∧ (B ∈ circle) ∧ (C ∈ circle) ∧ (D ∈ circle)

-- Define perpendicular diagonals at point F
def perpendicular_diagonals (A B C D F : Type) : Prop :=
  ∃ (AC BD : Type), (AC = A ∧ C = BD) ∧ (BD intersects AC at F) ∧ (∠ACF = 90°)

-- Define projections of F onto the sides
def projections (F A B C D P Q R S : Type) : Prop :=
  (P is the projection of F on AB) ∧
  (Q is the projection of F on BC) ∧
  (R is the projection of F on CD) ∧
  (S is the projection of F on DA)

-- Define bicentric quadrilateral
def bicentric_quadrilateral (P Q R S : Type) : Prop :=
  is_cyclic_quadrilateral P Q R S ∧ is_tangential_quadrilateral P Q R S

-- Main theorem statement
theorem PQRS_is_bicentric (h_cyclic: cyclic_quadrilateral A B C D)
                          (h_perpendicular: perpendicular_diagonals A B C D F)
                          (h_projections: projections F A B C D P Q R S) :
  bicentric_quadrilateral P Q R S :=
sorry

end PQRS_is_bicentric_l690_690611


namespace total_people_in_group_l690_690701

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l690_690701


namespace perpendicular_condition_l690_690664

-- Condition definition
def is_perpendicular (a : ℝ) : Prop :=
  let line1_slope := -1
  let line2_slope := - (a / 2)
  (line1_slope * line2_slope = -1)

-- Statement of the theorem
theorem perpendicular_condition (a : ℝ) :
  is_perpendicular a ↔ a = -2 :=
sorry

end perpendicular_condition_l690_690664


namespace divisor_is_10_l690_690663

variable (x y d : ℕ)

-- Condition 1: x = dy + 3
def cond1 := x = d * y + 3

-- Condition 2: 2x = 21y + 1
def cond2 := 2 * x = 21 * y + 1

-- Condition 3: 11y - x = 2
def cond3 := 11 * y - x = 2

-- Theorem statement: prove d = 10 given the conditions
theorem divisor_is_10 (h1 : cond1 x y d) (h2 : cond2 x y d) (h3 : cond3 x y d) : d = 10 :=
by {
  sorry
}

end divisor_is_10_l690_690663


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690916

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690916


namespace number_of_dress_designs_l690_690346

theorem number_of_dress_designs :
  let colors := 5
  let patterns := 4
  let sleeve_designs := 3
  colors * patterns * sleeve_designs = 60 := by
  sorry

end number_of_dress_designs_l690_690346


namespace smallest_real_constant_C_exists_l690_690067

theorem smallest_real_constant_C_exists :
  (∃ C, ∀ (a1 a2 a3 a4 a5 : ℝ), 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5 →
    ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
    | (a1 / a2) - (a3 / a4) | ≤ C) ∧
  (∀ C > 0, (∃ a1 a2 a3 a4 a5 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5 ∧
      ∀ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
      | (a1 / a2) - (a3 / a4) | > C → C ≥ 1 / 2)) :=
begin
  sorry
end

end smallest_real_constant_C_exists_l690_690067


namespace find_cost_price_l690_690000

variable (SP : ℕ) (rate : ℕ)

def cost_price (SP : ℕ) (rate : ℕ) : ℕ :=
  SP / (1 + rate / 100)

theorem find_cost_price (h1 : SP = 100) (h2 : rate = 100) : cost_price SP rate = 50 :=
by
  dsimp [cost_price]
  rw [h1, h2]
  norm_num
  sorry

end find_cost_price_l690_690000


namespace price_per_kg_l690_690322

-- Definitions based on conditions
variables (x y m : ℕ)
variables (k₁ k₂ : ℕ)
axiom h₁ : 9 * x + 10 * y = 163
axiom h₂ : 12 * x + 8 * y = 164
axiom profit_A : ℕ → ℕ := λ A, 4 * A
axiom profit_B : ℕ → ℕ := λ B, 6 * B
axiom max_B_qty : m ≤ 13
axiom case_two : (m : ℕ) → ℕ := λ m, 2 * m + 4
axiom profit_condition : λ m, profit_A (case_two m) + profit_B m ≥ 160

-- Theorem statement
theorem price_per_kg : x = 7 ∧ y = 10 ∧ 
( (m = 11 ∧ case_two 11 = 26)
  ∨ (m = 12 ∧ case_two 12 = 28)
  ∨ (m = 13 ∧ case_two 13 = 30) ) :=
by {
  sorry
}

end price_per_kg_l690_690322


namespace log_comparison_l690_690103

theorem log_comparison :
  let a := log 3 π
  let b := log 7 6
  let c := log 2 0.8
  a > b ∧ b > c :=
by
  sorry

end log_comparison_l690_690103


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690976

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690976


namespace find_relationship_l690_690457

theorem find_relationship (n m : ℕ) (a : ℚ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (n > m ↔ (1 / n < a)) → m = ⌊1 / a⌋ :=
sorry

end find_relationship_l690_690457


namespace train_speed_l690_690643

theorem train_speed (V_B T_A T_B : ℝ) (h1 : V_B = 105) (h2 : T_A = 9) (h3 : T_B = 4) :
  let V_A := (T_B / T_A) * V_B in V_A = 46.67 :=
by
  sorry

end train_speed_l690_690643


namespace perpendicular_bk_cm_l690_690093

variables {A B C K H M : Type*}
variables [EuclideanGeometry A B C K H M]

noncomputable def right_triangle (ABC : triangle) : Prop := 
  ∃ (A B C : point), right_triangle ABC ∧ hypotenuse ABC = AB

noncomputable def isosceles_right_triangle (ABC : triangle) : Prop :=
  ∃ (A B C : point), isosceles_right_triangle ABC ∧ hypotenuse ABC = AB

noncomputable def symmetric_point (B H M : point) : Prop :=
  let M := symmetric_point B H in M = symmetric_point B H

theorem perpendicular_bk_cm (ABC : triangle) (h : right_triangle ABC) 
  {AC K CH : Type*} (H : altitude ABC = CH)
  (h_sym : symmetric_point B H = M) 
  (h_ak_eq_bk : parallel (line_segment A K) (line_segment B K)) : 
  perpendicular (line_segment BK) (line_segment CM) := 
sorry

end perpendicular_bk_cm_l690_690093


namespace count_two_digit_prime_with_digit_sum_10_l690_690876

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690876


namespace angle_at_2_15_is_none_of_these_l690_690239

theorem angle_at_2_15_is_none_of_these :
    let hour_hand_angle := 60 + 7.5
    let minute_hand_angle := 15 * 6
    abs (minute_hand_angle - hour_hand_angle) = 22.5 → none_of_these :=
by
    sorry

end angle_at_2_15_is_none_of_these_l690_690239


namespace count_two_digit_prime_with_digit_sum_10_l690_690875

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690875


namespace dugu_team_prob_l690_690524

def game_prob (prob_win_first : ℝ) (prob_increase : ℝ) (prob_decrease : ℝ) : ℝ :=
  let p1 := prob_win_first
  let p2 := prob_win_first + prob_increase
  let p3 := prob_win_first + 2 * prob_increase
  let p4 := prob_win_first + 3 * prob_increase
  let p5 := prob_win_first + 4 * prob_increase
  let win_in_3 := p1 * p2 * p3
  let lose_first := (1 - prob_win_first)
  let win_then := prob_win_first
  let win_in_4a := lose_first * (prob_win_first - prob_decrease) * 
    prob_win_first * p2 * p3
  let win_in_4b := win_then * (1 - (prob_win_first + prob_increase)) *
    p2 * p3
  let win_in_4c := win_then * p2 * (1 - prob_win_first + prob_increase - 
    prob_decrease) * p4

  win_in_3 + win_in_4a + win_in_4b + win_in_4c

theorem dugu_team_prob : 
  game_prob 0.4 0.1 0.1 = 0.236 :=
by
  sorry

end dugu_team_prob_l690_690524


namespace polyline_intersection_ratio_l690_690442

-- Define points in space as elements of ℝ³
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Basic operations for segments lengths and ratios, assumed to be pre-defined in the background of Mathlib.

-- Definitions for polyline and intersection points
variables {n : ℕ} (A : fin n → Point3D) (B : fin n → Point3D)

-- The theorem statement
theorem polyline_intersection_ratio (h_closed : A n = A 0) :
    (∏ i in finset.fin_range n, (dist (A i) (B i)) / (dist (B i) (A (i + 1) % n))) = 1 :=
  sorry

end polyline_intersection_ratio_l690_690442


namespace sum_of_digits_of_triangular_number_3003_l690_690719

theorem sum_of_digits_of_triangular_number_3003 :
  ∃ N : ℕ, (N * (N + 1) / 2 = 3003) ∧ (nat.digits 10 N).sum = 14 :=
by
  sorry

end sum_of_digits_of_triangular_number_3003_l690_690719


namespace exists_quadratic_open_upwards_passing_through_origin_l690_690316

-- Define the general form of a quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Constants for the conditions
constants (a b c : ℝ)

-- Conditions
axiom a_pos : a > 0
axiom passes_through_origin : quadratic_function a b c 0 = 1

-- Goal: Prove that under the given conditions, the quadratic function exists (and hence provide an example of such function).
theorem exists_quadratic_open_upwards_passing_through_origin : 
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 :=
by
  use 1, 0, 1
  split
  { exact zero_lt_one }
  { simp [quadratic_function] }

end exists_quadratic_open_upwards_passing_through_origin_l690_690316


namespace prime_remainder_theorem_l690_690502

open Nat

-- Definitions of the problem's conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def prime_remainder_condition (p : ℕ) : Prop := is_prime p ∧ (50 < p) ∧ (p < 100) ∧ is_prime (p % 10)

-- Statement of the problem
theorem prime_remainder_theorem : 
  (Finset.univ.filter prime_remainder_condition).card = 5 := 
  sorry

end prime_remainder_theorem_l690_690502


namespace train_speed_correct_l690_690014

theorem train_speed_correct :
  ∀ (L : ℝ) (V_man : ℝ) (T : ℝ) (V_train : ℝ),
    L = 220 ∧ V_man = 6 * (1000 / 3600) ∧ T = 11.999040076793857 ∧ 
    L / T - V_man = V_train ↔ V_train * 3.6 = 60 :=
by
  intros L V_man T V_train
  sorry

end train_speed_correct_l690_690014


namespace max_gt_min_exp_l690_690078

theorem max_gt_min_exp (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x_max x_min : ℝ, 1 ≤ x_min ∧ x_min ≤ 2 ∧ 1 ≤ x_max ∧ x_max ≤ 2 ∧
  (∀ x ∈ set.Icc 1 2, a ^ x ≤ a ^ x_max) ∧ (∀ x ∈ set.Icc 1 2, a ^ x ≥ a ^ x_min) ∧
  a ^ x_max > a ^ x_min := 
by
  sorry

end max_gt_min_exp_l690_690078


namespace count_prime_two_digit_sum_ten_is_three_l690_690807

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690807


namespace count_prime_two_digit_sum_ten_is_three_l690_690813

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690813


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l690_690344

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l690_690344


namespace jigi_scored_55_percent_l690_690528

noncomputable def jigi_percentage (max_score : ℕ) (avg_score : ℕ) (gibi_pct mike_pct lizzy_pct : ℕ) : ℕ := sorry

theorem jigi_scored_55_percent :
  jigi_percentage 700 490 59 99 67 = 55 :=
sorry

end jigi_scored_55_percent_l690_690528


namespace teal_sales_l690_690534

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l690_690534


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690949

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690949


namespace rationalize_sqrt_denominator_l690_690592

theorem rationalize_sqrt_denominator :
  (sqrt 81)^(4:ℤ) = 3 → 
    1 / (sqrt 3 + 3)^(1 / 4:ℤ) = 1 / 27 := 
by
  intro h
  sorry

end rationalize_sqrt_denominator_l690_690592


namespace triangle_MAN_similar_triangle_ABC_l690_690218

variables {A B C D M N : Type*}
variables [EuclideanGeometry α] [DecidableEq α]
variables (A B C D M N : α)

-- Conditions
variable (parallelogram_ABCD : is_parallelogram A B C D)
variable (AM_perpendiculiar_BC : ⟂ AM BC)
variable (AN_perpendiculiar_CD : ⟂ AN CD)

-- Goal
theorem triangle_MAN_similar_triangle_ABC :
  ∆ MAN ~ ∆ ABC :=
begin
  sorry
end

end triangle_MAN_similar_triangle_ABC_l690_690218


namespace defective_percentage_is_0_05_l690_690722

-- Define the problem conditions as Lean definitions
def total_meters : ℕ := 4000
def defective_meters : ℕ := 2

-- Define the percentage calculation function
def percentage_defective (defective total : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * 100

-- Rewrite the proof statement using these definitions
theorem defective_percentage_is_0_05 :
  percentage_defective defective_meters total_meters = 0.05 :=
by
  sorry

end defective_percentage_is_0_05_l690_690722


namespace hexagon_area_problem_l690_690222

theorem hexagon_area_problem :
  ∃ (a b : ℕ), (nat.gcd a b = 1) ∧ (area_of_bounded_hexagon (construct_hexagon_of_side_length 2) = real.sqrt (a / b)) ∧ (100 * a + b = 1603) := sorry

def construct_hexagon_of_side_length (s : ℝ) : regular_hexagon := sorry

def regular_hexagon := {P : ℝ × ℝ // is_regular_hexagon P}

def is_regular_hexagon (P : ℝ × ℝ) : Prop := sorry

def area_of_bounded_hexagon (H : regular_hexagon) : ℝ := sorry

#print axioms hexagon_area_problem

end hexagon_area_problem_l690_690222


namespace count_two_digit_primes_with_digit_sum_10_l690_690893

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690893


namespace min_value_of_f_l690_690422

def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 10 

theorem min_value_of_f : (x y : ℝ) -> f(x, y) ≥ 14/5 := sorry

end min_value_of_f_l690_690422


namespace remainder_x6_p_l690_690749

-- Define the polynomials x^2 - x + 1 and x^6 - 1
noncomputable def p := (λ x : ℤ, x^2 - x + 1)
noncomputable def q := (λ x : ℤ, x^6 - 1)

-- Statement: remainder of q divided by p is -2
theorem remainder_x6_p :
  ∀ (x : ℤ), (q x) % (p x) = -2 :=
begin
  intro x,
  sorry
end

end remainder_x6_p_l690_690749


namespace num_two_digit_primes_with_digit_sum_10_l690_690826

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690826


namespace find_a_l690_690470

noncomputable def f (x a : ℝ) : ℝ := (x / 4) + (a / 4) - Real.log x - (3 / 2)

theorem find_a (a : ℝ) : (∃ a : ℝ, ∀ x : ℝ, x ≠ 0 → deriv (λ x, f x a) 1 = -2) → a = 5 / 4 :=
by
  sorry

end find_a_l690_690470


namespace problem_lean_l690_690106

theorem problem_lean (x y : ℝ) (h₁ : (|x + 2| ≥ 0) ∧ (|y - 4| ≥ 0)) : 
  (|x + 2| = 0 ∧ |y - 4| = 0) → x + y - 3 = -1 :=
by sorry

end problem_lean_l690_690106


namespace unique_students_amc8_l690_690725

theorem unique_students_amc8 (germain: ℕ) (newton: ℕ) (young: ℕ) (overlap: ℕ) :
  (germain = 12) → (newton = 10) → (young = 11) → (overlap = 5) → 
  (germain + newton + young - overlap = 28) :=
by
  intros hG hN hY hO
  rw [hG, hN, hY, hO]
  calc
    12 + 10 + 11 - 5 = 33 - 5 := by norm_num
    ...            = 28     := by norm_num

end unique_students_amc8_l690_690725


namespace two_digit_primes_with_digit_sum_10_count_l690_690897

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690897


namespace find_i_when_x_is_0_point3_l690_690509

noncomputable def find_i (x : ℝ) (i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

theorem find_i_when_x_is_0_point3 : find_i 0.3 2.9993 :=
by
  sorry

end find_i_when_x_is_0_point3_l690_690509


namespace primes_2_3_unique_l690_690056

theorem primes_2_3_unique (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) (hr_pos : 0 < r) 
  (h : 1 / (p + 1 : ℝ) + 1 / (q + 1 : ℝ) - 1 / ((((p + 1) * (q + 1) : ℕ) : ℝ)) = 1 / (r : ℝ)) : 
  {p, q} = {2, 3} ∨ {p, q} = {3, 2} := 
  sorry

end primes_2_3_unique_l690_690056


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690943

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690943


namespace overall_avg_speed_is_correct_l690_690691

def segment1 := (distance := 9, speed := 12)
def segment2 := (distance := 12, speed := 9)
def segment3 := (distance := 15, speed := 18)
def segment4 := (distance := 20, speed := 10)
def segment5 := (distance := 10, speed := 15)

noncomputable def total_distance :=
  segment1.distance + segment2.distance + segment3.distance + segment4.distance + segment5.distance

noncomputable def total_time :=
  segment1.distance / segment1.speed + 
  segment2.distance / segment2.speed + 
  segment3.distance / segment3.speed + 
  segment4.distance / segment4.speed + 
  segment5.distance / segment5.speed

noncomputable def overall_avg_speed :=
  total_distance / total_time

theorem overall_avg_speed_is_correct : overall_avg_speed = 11.83 := by
  sorry

end overall_avg_speed_is_correct_l690_690691


namespace largest_pos_int_divisible_l690_690653

theorem largest_pos_int_divisible (n : ℕ) (h1 : n > 0) (h2 : n + 11 ∣ n^3 + 101) : n = 1098 :=
sorry

end largest_pos_int_divisible_l690_690653


namespace count_two_digit_primes_with_digit_sum_10_l690_690863

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690863


namespace count_prime_two_digit_sum_ten_is_three_l690_690812

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690812


namespace dodecagon_diagonals_l690_690135

theorem dodecagon_diagonals : 
  let n := 12 in 
  let d := n * (n - 3) / 2 in 
  d = 54 :=
by
  let n := 12
  let d := n * (n - 3) / 2
  show d = 54
  sorry

end dodecagon_diagonals_l690_690135


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690945

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690945


namespace num_two_digit_primes_with_digit_sum_10_l690_690781

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690781


namespace trigonometric_product_identity_l690_690026

theorem trigonometric_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) = 2.25 :=
by
  -- Let x = Real.pi / 12 and y = 5 * Real.pi / 12
  have h1 : Real.sin (11 * Real.pi / 12) = Real.sin (Real.pi - (Real.pi / 12)) := by sorry,
  have h2 : Real.sin (7 * Real.pi / 12) = Real.sin (Real.pi - (5 * Real.pi / 12)) := by sorry,
  have h3 : Real.sin (5 * Real.pi / 12) = Real.cos (Real.pi / 12) := by sorry,
  have h4 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry,
  have h5 : Real.sin (2 * Real.pi / 12) = 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) := by sorry,
  sorry

end trigonometric_product_identity_l690_690026


namespace two_digit_primes_with_digit_sum_ten_l690_690802

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690802


namespace cory_fruit_eating_orders_l690_690393

open Nat

theorem cory_fruit_eating_orders : 
    let apples := 4
    let oranges := 3
    let bananas := 2
    let grape := 1
    let total_fruits := apples + oranges + bananas + grape
    apples + oranges + bananas + grape = 10 →
    total_fruits = 10 →
    apples ≥ 1 →
    factorial 9 / (factorial 3 * factorial 3 * factorial 2 * factorial 1) = 5040 :=
by
  intros apples oranges bananas grape total_fruits h_total h_sum h_apples
  sorry

end cory_fruit_eating_orders_l690_690393


namespace problem_equivalent_l690_690963

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690963


namespace last_page_to_be_read_l690_690687

theorem last_page_to_be_read : 
  (∀ (P : ℕ), 2 ≤ P ∧ P ≤ 400 → 
    (∃ (H : P = 397), 
      ∀ (Q : ℕ), 2 ≤ Q ∧ Q ≤ 400 ∧ Q ≠ 397 → 
        (∃ last_unread : ℕ, last_unread = 400 ∨ 
          (∀ n ≤ last_unread, ¬ (∃ p, p.prime ∧ p ∣ last_unread ∧ p ∣ n))) ∧ ¬ (∃ q : ℕ, q.prime ∧ q > 19 ∧ ¬ (q ∣ last_unread) ∧ ¬ (∃ p : ℕ, p.prime ∧ p ∣ q)))) :=
sorry

end last_page_to_be_read_l690_690687


namespace unknown_number_eq_0_5_l690_690660

theorem unknown_number_eq_0_5 : 
  ∃ x : ℝ, x + ((2 / 3) * (3 / 8) + 4) - (8 / 16) = 4.25 ∧ x = 0.5 :=
by
  use 0.5
  sorry

end unknown_number_eq_0_5_l690_690660


namespace count_two_digit_primes_with_digit_sum_10_l690_690881

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690881


namespace probability_of_same_color_l690_690160

theorem probability_of_same_color (W B : ℕ) (hW : W = 3) (hB : B = 2) : 
  let total_pairs := W * (W - 1) / 2 + B * (B - 1) / 2 + W * B,
  let same_color_pairs := W * (W - 1) / 2 + B * (B - 1) / 2,
  same_color_pairs / total_pairs = 2 / 5 :=
by
  sorry

end probability_of_same_color_l690_690160


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690983

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690983


namespace rearrange_balls_l690_690405

-- Defining the problem setup
def Girl : Type := fin 100
def Ball : Type := fin 10000
def Color : Type := fin 100

/-- Each girl has initially 100 balls -/
def owns_initially (g : Girl) : finset Ball := sorry

/-- Each ball belongs to a color -/
def ball_color (b : Ball) : Color := sorry

-- The goal of the problem
theorem rearrange_balls :
  (∃ (exchange: finset (Ball × Ball)), 
  (∀ (b : Ball), (∃ (b' : Ball), (b, b') ∈ exchange ∨ (b', b) ∈ exchange) → b ≠ b') ∧ -- No ball takes part in more than one trade
  (∀ (g : Girl), (∀ c : Color, ∃ b : Ball, b ∈ owns_initially g ∨ ((∃ (b' : Ball), ((b, b') ∈ exchange ∨ (b', b) ∈ exchange) ∧ (owns_initially g).contains b'))) ∧ ball_color b = c)) -- Each girl ends up with 100 different colors
: true :=
sorry

end rearrange_balls_l690_690405


namespace length_of_AB_l690_690519

theorem length_of_AB (BC AC : ℝ) (area : ℝ) (hBC : BC = 2 * real.sqrt 5) (hAC : AC = 2) (hArea : area = 4) :
  ∃ (AB : ℝ), AB = 4 ∨ AB = 4 * real.sqrt 2 :=
by
  use 4
  use 4 * real.sqrt 2
  sorry

end length_of_AB_l690_690519


namespace min_positive_value_7_l690_690041

def sum_conditions (a : Fin 50 → ℝ) (i j : ℕ) : Bool :=
  (i < j) && (i < 50) && (j < 50)

def termwise_product (a : Fin 50 → ℝ) : ℝ :=
  ∑ i in Fin.range 50, ∑ j in Fin.range 50, if sum_conditions a i j then a i * a j else 0

theorem min_positive_value_7 (a : Fin 50 → ℝ) (ha : ∀ i, a i = 1 ∨ a i = -1) :
  (termwise_product a) > 0 ∧ (termwise_product a) = 7 :=
by
  sorry

end min_positive_value_7_l690_690041


namespace equilibrium_temperature_l690_690348

-- Initial conditions for heat capacities and masses
variables (c_B c_W m_B m_W : ℝ) (h : c_W * m_W = 3 * c_B * m_B)

-- Initial temperatures
def T_W_initial := 100
def T_B_initial := 20
def T_f_initial := 80

-- Final equilibrium temperature after second block is added
def final_temp := 68

theorem equilibrium_temperature (t : ℝ)
  (h_first_eq : c_W * m_W * (T_W_initial - T_f_initial) = c_B * m_B * (T_f_initial - T_B_initial))
  (h_second_eq : c_W * m_W * (T_f_initial - t) + c_B * m_B * (T_f_initial - t) = c_B * m_B * (t - T_B_initial)) :
  t = final_temp :=
by 
  sorry

end equilibrium_temperature_l690_690348


namespace valid_votes_for_other_candidate_l690_690164

def total_votes : ℕ := 5500
def percentage_invalid : ℕ := 20
def percentage_candidate1 : ℕ := 55

theorem valid_votes_for_other_candidate :
  let valid_votes := total_votes * (100 - percentage_invalid) / 100 in
  let valid_votes_candidate1 := valid_votes * percentage_candidate1 / 100 in
  let valid_votes_other_candidate := valid_votes - valid_votes_candidate1 in
  valid_votes_other_candidate = 1980 :=
by 
  sorry

end valid_votes_for_other_candidate_l690_690164


namespace tile_coverage_fraction_l690_690044

structure Room where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  
structure Tiles where
  square_tiles : ℕ
  triangular_tiles : ℕ
  triangle_base : ℝ
  triangle_height : ℝ
  tile_area : ℝ
  triangular_tile_area : ℝ
  
noncomputable def fractionalTileCoverage (room : Room) (tiles : Tiles) : ℝ :=
  let rect_area := room.rect_length * room.rect_width
  let tri_area := (room.tri_base * room.tri_height) / 2
  let total_room_area := rect_area + tri_area
  let total_tile_area := (tiles.square_tiles * tiles.tile_area) + (tiles.triangular_tiles * tiles.triangular_tile_area)
  total_tile_area / total_room_area

theorem tile_coverage_fraction
  (room : Room) (tiles : Tiles)
  (h1 : room.rect_length = 12)
  (h2 : room.rect_width = 20)
  (h3 : room.tri_base = 10)
  (h4 : room.tri_height = 8)
  (h5 : tiles.square_tiles = 40)
  (h6 : tiles.triangular_tiles = 4)
  (h7 : tiles.tile_area = 1)
  (h8 : tiles.triangular_tile_area = (1 * 1) / 2) :
  fractionalTileCoverage room tiles = 3 / 20 :=
by 
  sorry

end tile_coverage_fraction_l690_690044


namespace count_two_digit_prime_with_digit_sum_10_l690_690867

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690867


namespace original_team_size_l690_690283

theorem original_team_size (n : ℕ) (W : ℕ) :
  (W = n * 94) →
  ((W + 110 + 60) / (n + 2) = 92) →
  n = 7 :=
by
  intro hW_avg hnew_avg
  -- The proof steps would go here
  sorry

end original_team_size_l690_690283


namespace Tom_builds_wall_in_2_hours_l690_690181

noncomputable def time_taken_by_Tom_to_build_wall : ℝ :=
  let T := 2.0000000000000005
  let Avery_work_rate := 1 / 3
  let Tom_work_rate := 1 / T
  let total_time_Tom_works_after_Avery_leaves := 20.000000000000007 / 60
  assert_work_done := (Avery_work_rate + Tom_work_rate) * 1 + Tom_work_rate * total_time_Tom_works_after_Avery_leaves = 1
  T

theorem Tom_builds_wall_in_2_hours :
  time_taken_by_Tom_to_build_wall = 2 := by
  sorry

end Tom_builds_wall_in_2_hours_l690_690181


namespace probability_interval_l690_690623

/-- The probability that event A occurs --/
def P_A : ℚ := 5 / 6

/-- The probability that event B occurs --/
def P_B : ℚ := 3 / 4

/-- Events A and B are independent --/
axiom independent : ∀ (A B : Prop), (P(A and B) = P(A) * P(B))

/-- Let p be the probability that both A and B occur --/
def p : ℚ := P_A * P_B

theorem probability_interval :
  p = 5 / 8 :=
by
  sorry

end probability_interval_l690_690623


namespace common_elements_214_l690_690192

def multiples (n k : ℕ) : set ℕ := {x | ∃ (m : ℕ), m ≤ k ∧ x = n * m}

def common_multiples_count (n₁ n₂ k₁ k₂ : ℕ) : ℕ :=
  let lcm_n₁_n₂ := Nat.lcm n₁ n₂
  (k₁ * n₁) / lcm_n₁_n₂

theorem common_elements_214 :
  let S := multiples 5 1500
  let T := multiples 7 1500
  common_multiples_count 5 7 1500 1500 = 214 :=
by
  sorry

end common_elements_214_l690_690192


namespace bounding_consecutive_integers_l690_690762

noncomputable def sequence (n : ℕ) : ℝ :=
  Nat.recOn n (1/3 : ℝ) (λ n xn, xn^2 + xn)

noncomputable def sum_seq (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, 1 / (sequence k + 1))

theorem bounding_consecutive_integers :
  2 < sum_seq 2002 ∧ sum_seq 2002 < 3 :=
sorry

end bounding_consecutive_integers_l690_690762


namespace quarters_for_chips_l690_690579

def total_quarters : ℕ := 16
def quarters_for_soda : ℕ := 12

theorem quarters_for_chips : (total_quarters - quarters_for_soda) = 4 :=
  by 
    sorry

end quarters_for_chips_l690_690579


namespace age_of_eldest_child_l690_690626

theorem age_of_eldest_child (age_sum : ∀ (x : ℕ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 40) :
  ∃ x, x + 8 = 12 :=
by {
  sorry
}

end age_of_eldest_child_l690_690626


namespace count_two_digit_primes_with_digit_sum_10_l690_690994

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690994


namespace count_two_digit_prime_with_digit_sum_10_l690_690865

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690865


namespace min_value_2k_l690_690566

noncomputable def minimum_value (k : ℝ) (x y z : ℝ) : ℝ :=
  (k / ((2 - x) * (2 - y) * (2 - z))) + (k / ((2 + x) * (2 + y) * (2 + z)))

theorem min_value_2k (k : ℝ) (h_k : k > 0) :
  ∃ x y z : ℝ, -2 < x ∧ x < 2 ∧ -2 < y ∧ y < 2 ∧ -2 < z ∧ z < 2 → 
  minimum_value k x y z = 2 * k :=
begin
  sorry
end

end min_value_2k_l690_690566


namespace relative_complement_correct_l690_690111

noncomputable def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}
def complement_M_N : Set ℤ := {x ∈ M | x ∉ N}

theorem relative_complement_correct : complement_M_N = {-1, 0, 3} := 
by
  sorry

end relative_complement_correct_l690_690111


namespace count_two_digit_primes_with_digit_sum_10_l690_690892

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690892


namespace num_two_digit_primes_with_digit_sum_10_l690_690834

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690834


namespace max_projection_value_l690_690448

variable (O A : ℝ×ℝ) (P : ℝ → ℝ×ℝ)
variable (x : ℝ)
def pointA := (1,0 : ℝ×ℝ)
def curve := λ x : ℝ, (x, x^2 + 1 : ℝ×ℝ)

theorem max_projection_value : 
  ∃ P : ℝ → ℝ×ℝ, 
  (P = curve) → 
  ∀ (O A : ℝ×ℝ), 
  O = (0, 0 : ℝ×ℝ) → 
  A = pointA → 
  ∃ x : ℝ,
  (∃ max_value : ℝ, max_value = (|((1:ℝ),0) ⬝ (x, x^2 + 1)| 
  / √((x:ℝ)^2 + ((x:ℝ)^2 + (1:ℝ))^2)) ∧ max_value = (√5 / 5)) :=
begin
  sorry
end

end max_projection_value_l690_690448


namespace certain_amount_l690_690053

theorem certain_amount (n : ℤ) (x : ℤ) : n = 5 ∧ 7 * n - 15 = 2 * n + x → x = 10 :=
by
  sorry

end certain_amount_l690_690053


namespace reciprocal_of_repeating_decimal_l690_690658

theorem reciprocal_of_repeating_decimal : (∀ x : ℚ, x = 36 / 99 → (1 / (4 / 11) = 11 / 4)) :=
begin
  assume x,
  assume h : x = 36 / 99,
  sorry
end

end reciprocal_of_repeating_decimal_l690_690658


namespace parallelogram_with_adjacent_equal_sides_and_right_angle_is_square_l690_690310

variable {Q : Type} [quadrilateral Q]

-- The conditions
def is_parallelogram (q : Q) : Prop := 
  ∃ (a b c d : point), parallelogram a b c d

def adjacent_sides_equal (q : Q) : Prop :=
  ∃ (a b : point), adjacent a b ∧ length a b = length b c

def has_right_angle (q : Q) : Prop := 
  ∃ (a : point), right_angle a

-- The goal
theorem parallelogram_with_adjacent_equal_sides_and_right_angle_is_square 
  (q : Q) (hp : is_parallelogram q) (ha : adjacent_sides_equal q) (hr : has_right_angle q) : is_square q := 
sorry

end parallelogram_with_adjacent_equal_sides_and_right_angle_is_square_l690_690310


namespace calculate_abc_over_def_l690_690673

theorem calculate_abc_over_def
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  (a * b * c) / (d * e * f) = 1 / 2 :=
by
  sorry

end calculate_abc_over_def_l690_690673


namespace directrix_of_parabola_l690_690251

-- Define the given condition: the equation of the parabola
def given_parabola (x : ℝ) : ℝ := 4 * x ^ 2

-- State the theorem to be proven
theorem directrix_of_parabola : 
  (∀ x : ℝ, given_parabola x = 4 * x ^ 2) → 
  (y = -1 / 16) :=
sorry

end directrix_of_parabola_l690_690251


namespace rational_sequence_l690_690360

noncomputable def sequence (a : ℕ → ℝ) := a 1 = 2 ∧ a 2 = 11 ∧ ∀ (n : ℕ), n ≥ 1 → 2 * a (n + 2) = 3 * a n + Real.sqrt (5 * ((a n) ^ 2 + (a (n + 1)) ^ 2))

theorem rational_sequence (a : ℕ → ℝ) (h : sequence a) :
  ∀ (n : ℕ), n ≥ 1 → ∃ (q : ℚ), a n = q :=
sorry

end rational_sequence_l690_690360


namespace cubic_feet_total_l690_690496

def cubic_conversion (yards : ℕ) : ℕ := 
  yards * 27

theorem cubic_feet_total 
  (yards_to_feet : ℕ) 
  (cubic_yard_to_feet : ℕ) 
  (five_cubic_yards : ℕ)
  (additional_feet : ℕ) : cubic_conversion 5 + 4 = 139 :=
by
  have h1 : yards_to_feet = 3 := rfl
  have h2 : cubic_yard_to_feet = 27 := by rw [h1, h1, h1]
  have h3 : five_cubic_yards = cubic_conversion 5 := rfl
  have h4 : 5 * 27 = 135 := by norm_num
  have h5 : cubic_conversion 5 + 4 = 139 := by rw [h4] 
  exact h5

end cubic_feet_total_l690_690496


namespace perfect_cubes_in_sequence_l690_690178

theorem perfect_cubes_in_sequence : 
  let seq := (List.range' 100 (876 - 100 + 1)).map (λ n => n^n)
  let is_perfect_cube (m : ℕ) : Prop := ∃ k : ℕ, k^3 = m
  let count_perfect_cubes := (seq.filter is_perfect_cube).length
  count_perfect_cubes = 262 := 
sorry

end perfect_cubes_in_sequence_l690_690178


namespace clever_point_classification_l690_690090

-- Definition of a clever point
def is_clever_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = (deriv f) x

-- Functions to be evaluated
def f1 (x : ℝ) : ℝ := x^2
def f2 (x : ℝ) : ℝ := Real.exp (-x)
def f3 (x : ℝ) : ℝ := Real.log x
def f4 (x : ℝ) : ℝ := Real.tan x

-- Proof goal statement for clever points
theorem clever_point_classification :
  (∃ x : ℝ, is_clever_point f1 x) ∧
  ¬ (∃ x : ℝ, is_clever_point f2 x) ∧
  (∃ x : ℝ, is_clever_point f3 x) ∧
  ¬ (∃ x : ℝ, is_clever_point f4 x) := 
sorry

end clever_point_classification_l690_690090


namespace two_digit_prime_sum_digits_10_count_l690_690931

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690931


namespace yellow_white_flowers_count_l690_690370

theorem yellow_white_flowers_count
    (RY RW : Nat)
    (hRY : RY = 17)
    (hRW : RW = 14)
    (hRedMoreThanWhite : (RY + RW) - (RW + YW) = 4) :
    ∃ YW, YW = 13 := 
by
  sorry

end yellow_white_flowers_count_l690_690370


namespace length_PQ_45_value_of_k_l690_690100

-- Definitions for the problem conditions

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def left_focus (f : ℝ × ℝ) : Prop := f = (-1, 0)
def line_eq_45 (x y : ℝ) : Prop := y = x + 1
def line_eq_k (x y k : ℝ) : Prop := y = k * (x + 1)
def slope_PP'QQ' (x1 y1 x2 y2 k : ℝ) : ℝ := abs ((y1 - y2) / (x1 + x2))

-- The statements to be proven

-- Part I: Length of segment PQ when slope angle is 45 degrees
theorem length_PQ_45 :
  ∀ x1 y1 x2 y2 : ℝ,
    left_focus (-1, 0) →
    (ellipse_eq x1 y1 ∧ ellipse_eq x2 y2) →
    line_eq_45 x1 y1 →
    line_eq_45 x2 y2 →
    dist (x1, y1) (x2, y2) = 24 / 7 :=
sorry

-- Part II: Value of k given |k'| = 2
theorem value_of_k :
  ∀ x1 y1 x2 y2 k k' : ℝ,
    left_focus (-1, 0) →
    (ellipse_eq x1 y1 ∧ ellipse_eq x2 y2) →
    line_eq_k x1 y1 k →
    line_eq_k x2 y2 k →
    k' = slope_PP'QQ' (-x1) (-y1) x2 (-y2) k →
    abs k' = 2 →
    abs k = 3 / 7 * sqrt 7 :=
sorry

end length_PQ_45_value_of_k_l690_690100


namespace average_age_of_girls_l690_690163

theorem average_age_of_girls (total_students : ℕ) (avg_age_boys : ℕ) (num_girls : ℕ) (avg_age_school : ℚ) 
  (h1 : total_students = 604) 
  (h2 : avg_age_boys = 12) 
  (h3 : num_girls = 151) 
  (h4 : avg_age_school = 11.75) : 
  (total_age_of_girls / num_girls) = 11 :=
by
  -- Definitions
  let num_boys := total_students - num_girls
  let total_age := avg_age_school * total_students
  let total_age_boys := avg_age_boys * num_boys
  let total_age_girls := total_age - total_age_boys
  -- Proof goal
  have : total_age_of_girls = total_age_girls := sorry
  have : total_age_of_girls / num_girls = 11 := sorry
  sorry

end average_age_of_girls_l690_690163


namespace algebraic_expression_value_l690_690102

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : -2 * a^2 + 8 * a - 5 = 1 := 
by 
  sorry 

end algebraic_expression_value_l690_690102


namespace light_glows_l690_690284

noncomputable def seconds_between (start_hour start_minute start_second end_hour end_minute end_second : ℕ) : ℕ :=
  let start := start_hour * 3600 + start_minute * 60 + start_second
  let end := end_hour * 3600 + end_minute * 60 + end_second
  end - start

noncomputable def glows_count (total_seconds interval : ℕ) : ℕ :=
  total_seconds / interval

def LightGloDurations : Prop :=
  let total_seconds := seconds_between 1 57 58 3 20 47
  let glows_A := glows_count total_seconds 16
  let glows_B := glows_count total_seconds 35
  let glows_C := glows_count total_seconds 42
  total_seconds = 4969 ∧
  glows_A = 310 ∧
  glows_B = 141 ∧
  glows_C = 118 ∧
  glows_A + glows_B + glows_C = 569

theorem light_glows : LightGloDurations :=
by
  sorry

end light_glows_l690_690284


namespace two_digit_prime_sum_to_ten_count_l690_690836

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690836


namespace rope_rounds_second_cylinder_l690_690317

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem rope_rounds_second_cylinder :
  let length_of_rope := 70 * circumference 14 in
  length_of_rope / (circumference 20) = 49 :=
by
  sorry

end rope_rounds_second_cylinder_l690_690317


namespace ice_cream_flavors_l690_690504

theorem ice_cream_flavors (chocolate vanilla strawberry : ℕ) (scoops : ℕ) :
  chocolate = 1 → vanilla = 1 → strawberry = 1 → scoops = 4 → 
  ∃ n : ℕ, n = Nat.choose (scoops + 2) 2 ∧ n = 15 :=
by
  intros h1 h2 h3 h4
  use Nat.choose (scoops + 2) 2
  split
  case left => rfl
  case right => 
    simp only [Nat.choose, *] at *
    sorry

end ice_cream_flavors_l690_690504


namespace inequality_a_pow_n_l690_690429

open Real

theorem inequality_a_pow_n (a : ℝ) (n : ℕ) (hn : 1 ≤ n) : a ≠ 0 →
  a^n + (a ^ (-n)) - 2 ≥ n^2 * (a + (a ^ (-1)) - 2) :=
sorry

end inequality_a_pow_n_l690_690429


namespace sum_f_expected_value_l690_690087

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_1 : f 1 = 1
axiom f_equation_for_neg_x : ∀ x : ℝ, x < 0 → f (1 / x) = x * f (1 / (1 - x))

theorem sum_f_expected_value :
  (∑ k in finset.range (1009 + 1), f (1 / k) * f (1 / (2019 - k))) = (2 ^ 2016) / 2017.factorial :=
sorry

end sum_f_expected_value_l690_690087


namespace resulting_polygon_has_25_sides_l690_690036

def quadrilateral := 4 -- sides of the quadrilateral
def rectangle := 4 -- sides of the rectangle
def pentagon := 5 -- sides of the pentagon
def hexagon := 6 -- sides of the hexagon
def heptagon := 7 -- sides of the heptagon
def nonagon := 9 -- sides of the nonagon

def shared_sides {n : Nat} {m : Nat} (n_adj : Nat) (m_adj : Nat) : Nat :=
  n + m - n_adj - m_adj -- function to calculate shared sides

theorem resulting_polygon_has_25_sides : 
  let sides_quad := quadrilateral in
  let sides_non := nonagon in
  let sides_others := rectangle + pentagon + hexagon + heptagon in
  let shared_quad_non := shared_sides sides_quad sides_non 1 1 in 
  let shared_others := shared_sides sides_others (4 * 2) in -- considering all other shapes share 2 sides each
  shared_quad_non + shared_others = 25 := by
  sorry

end resulting_polygon_has_25_sides_l690_690036


namespace convert_cylindrical_to_rectangular_l690_690392

-- Definitions of the conversion from cylindrical to rectangular coordinates
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- The given cylindrical coordinates point
def point_cylindrical : ℝ × ℝ × ℝ := (5, Real.pi / 3, 2)

-- The expected rectangular coordinates result
def expected_point_rectangular : ℝ × ℝ × ℝ := (2.5, 5 * Real.sqrt 3 / 2, 2)

-- The theorem statement to prove
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 5 (Real.pi / 3) 2 = expected_point_rectangular :=
by
  sorry

end convert_cylindrical_to_rectangular_l690_690392


namespace count_two_digit_primes_with_digit_sum_10_l690_690859

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690859


namespace max_product_price_l690_690382

/-- Conditions: 
1. Company C sells 50 products.
2. The average retail price of the products is $2,500.
3. No product sells for less than $800.
4. Exactly 20 products sell for less than $2,000.
Goal:
Prove that the greatest possible selling price of the most expensive product is $51,000.
-/
theorem max_product_price (n : ℕ) (avg_price : ℝ) (min_price : ℝ) (threshold_price : ℝ) (num_below_threshold : ℕ) :
  n = 50 → 
  avg_price = 2500 → 
  min_price = 800 → 
  threshold_price = 2000 → 
  num_below_threshold = 20 → 
  ∃ max_price : ℝ, max_price = 51000 :=
by 
  sorry

end max_product_price_l690_690382


namespace magnitude_of_pure_imaginary_number_l690_690513

noncomputable def z (m : ℝ) : ℂ := (m^2 - 9 : ℝ) + (m^2 + 2 * m - 3 : ℝ) * Complex.I

theorem magnitude_of_pure_imaginary_number (m : ℝ) (h : z m = (0 : ℂ) + (m^2 + 2 * m - 3 : ℝ) * Complex.I) :
  ∥z m∥ = 12 :=
by {
  sorry
}

end magnitude_of_pure_imaginary_number_l690_690513


namespace num_two_digit_primes_with_digit_sum_10_l690_690820

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690820


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690947

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690947


namespace find_sin_2_gamma_l690_690586

variable (M N O P Q : Point)
variable (MN NO OP : ℝ)
variable (α β γ : ℝ)

-- Conditions: points M, N, O, P are equally spaced, and given cosine values for angles MQO and NQP
axiom eq_spacing : MN = NO ∧ NO = OP
axiom cos_alpha : Real.cos α = 3 / 5
axiom cos_beta : Real.cos β = 1 / 2

-- To prove that sin(2 * γ) = 6 * sqrt(3) / 10
theorem find_sin_2_gamma (hα : α = angle M Q O) (hβ : β = angle N Q P) (hγ : γ = angle N Q O) :
  Real.sin (2 * γ) = 6 * Real.sqrt 3 / 10 := by
  sorry

end find_sin_2_gamma_l690_690586


namespace impossible_to_turn_all_off_l690_690166

-- Define the initial conditions and necessary types
def grid_size : ℕ := 6
def initial_bug_count : fin grid_size.succ × fin grid_size.succ → bool :=
  λ p, if p = (0, 0) then true else false

-- Definition of a single move: toggling a vertical or horizontal triplet
def toggle_triplet (g : fin grid_size.succ × fin grid_size.succ → bool) 
  (p1 p2 p3 : fin grid_size.succ × fin grid_size.succ) : (fin grid_size.succ × fin grid_size.succ → bool) :=
  λ p, if p = p1 ∨ p = p2 ∨ p = p3 then bnot (g p) else g p

-- Since the problem is to show impossibility, we state it as follows:
theorem impossible_to_turn_all_off :
  ∀ (moves : list (fin grid_size.succ × fin grid_size.succ) × (fin grid_size.succ × fin grid_size.succ) × (fin grid_size.succ × fin grid_size.succ)),
  let final_state := (moves.foldl (λ g m, toggle_triplet g m.fst m.snd.fst m.snd.snd) initial_bug_count) in 
  (∀ p, final_state p = false) → false := by sorry

end impossible_to_turn_all_off_l690_690166


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690977

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690977


namespace find_sin_beta_l690_690148

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2) -- α is acute
variable (hβ : 0 < β ∧ β < π/2) -- β is acute

variable (hcosα : Real.cos α = 4/5)
variable (hcosαβ : Real.cos (α + β) = 5/13)

theorem find_sin_beta (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
    (hcosα : Real.cos α = 4/5) (hcosαβ : Real.cos (α + β) = 5/13) : 
    Real.sin β = 33/65 := 
sorry

end find_sin_beta_l690_690148


namespace y_squared_minus_2x_correct_l690_690037

def decimal_to_binary (n : ℕ) : list ℕ := 
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) : list ℕ :=
      if n = 0 then [] else aux (n / 2) ++ [n % 2]
    aux n

def count_zeros_ones (l : list ℕ) : ℕ × ℕ :=
  l.foldl (λ (acc : ℕ × ℕ) (x : ℕ) => 
    if x = 0 then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1)) (0, 0)

theorem y_squared_minus_2x_correct :
  let n := 235
  let binary_n := decimal_to_binary n
  let (x, y) := count_zeros_ones binary_n
  y^2 - 2*x = 32 :=
by {
  sorry
}

end y_squared_minus_2x_correct_l690_690037


namespace count_two_digit_prime_with_digit_sum_10_l690_690873

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690873


namespace count_two_digit_primes_with_digit_sum_10_l690_690995

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690995


namespace log_eq_exp_solution_l690_690048

theorem log_eq_exp_solution (y : ℝ) : log 16 (3 * y - 4) = 2 → y = 260 / 3 :=
by
  intro h
  sorry

end log_eq_exp_solution_l690_690048


namespace not_coplanar_probability_of_tetrahedron_points_l690_690709

/-- Given a regular tetrahedron with 10 points (including vertices and midpoints of edges),
if 4 points are selected from these 10 points, prove that the probability of these four points 
being not coplanar is 47/70. --/
theorem not_coplanar_probability_of_tetrahedron_points :
  let points := (finset.range 10).choose 4
  (∃ tetrahedron : finset (finset ℝ ^ 3) (hedron_vertices : finset (ℝ ^ 3)) (hedron_midpoints : finset (ℝ ^ 3)),
   is_regular_tetrahedron tetrahedron  ∧
   (hedron_vertices ∪ hedron_midpoints).card = 10 ∧
   is_coplanar := sorry ∧
   (∃ subset : finset (ℝ ^ 3), subset.card = 4 ∧ (is_coplanar subset) → ¬is_coplanar subset ∨
   subset.card_choose_4.totally_not_coplanar 
  (209 / 300)) / 
  cardinality points 
  = 47 / 70 := sorry.

end not_coplanar_probability_of_tetrahedron_points_l690_690709


namespace find_C_distance_A_to_BC_l690_690097

-- Define point and line structures for better readability
structure Point where
  x : ℚ
  y : ℚ

-- Definitions for points A, B and the line containing the angle bisector of ∠C
def A : Point := ⟨2, 1⟩
def B : Point := ⟨-1, -1⟩
def line_angle_bisector (p : Point) := p.x + 2 * p.y - 1 = 0

-- Coordinates of the point C as solved in the problem
def C : Point := ⟨31 / 5, -13 / 5⟩

-- Distance formula from a point to a line Ax + By + C = 0
def distance_from_point_to_line (P : Point) (A B C : ℚ) : ℚ :=
  abs (A * P.x + B * P.y + C) / real.sqrt (A^2 + B^2)

-- Statement for the problem
theorem find_C_distance_A_to_BC :
  (∃ C : Point, C = ⟨31 / 5, -13 / 5⟩) ∧
  distance_from_point_to_line A 2 9 11 = 24 / real.sqrt 85 := by
  sorry

end find_C_distance_A_to_BC_l690_690097


namespace ways_to_divide_five_people_into_two_rooms_l690_690773

theorem ways_to_divide_five_people_into_two_rooms : 
  (finset.card (finset.powerset len_univ).filter(λ s, s.card = 3)) = 10 :=
by
  sorry

end ways_to_divide_five_people_into_two_rooms_l690_690773


namespace sum_of_angles_l690_690402

theorem sum_of_angles :
  (∑ x in {x ∈ finset.range 361 | sin x ^ 6 - cos x ^ 6 = (1 / cos x ^ 2) - (1 / sin x ^ 2)}, x) = 720 :=
  sorry

end sum_of_angles_l690_690402


namespace num_two_digit_primes_with_digit_sum_10_l690_690783

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690783


namespace two_digit_prime_sum_digits_10_count_l690_690929

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690929


namespace two_digit_primes_with_digit_sum_ten_l690_690791

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690791


namespace water_left_l690_690187

theorem water_left (initial_water: ℚ) (science_experiment_use: ℚ) (plant_watering_use: ℚ)
  (h1: initial_water = 3)
  (h2: science_experiment_use = 5 / 4)
  (h3: plant_watering_use = 1 / 2) :
  (initial_water - science_experiment_use - plant_watering_use = 5 / 4) :=
by
  rw [h1, h2, h3]
  norm_num

end water_left_l690_690187


namespace first_player_guarantees_win_l690_690642

theorem first_player_guarantees_win :
  ∃ strategy, ∀ moves, win_by_strategy moves strategy ∧ ∀ other_moves, ¬ win_by_other_strategy other_moves :=
by
  -- No proof is provided, only statement is described as per problem analysis.
  sorry

end first_player_guarantees_win_l690_690642


namespace max_true_statements_l690_690198

theorem max_true_statements :
  ∃ x : ℝ, 
  (0 < x ∧ x < 1) ∧ -- Statement 4
  (0 < x^3 ∧ x^3 < 1) ∧ -- Statement 1
  (0 < x - x^3 ∧ x - x^3 < 1) ∧ -- Statement 5
  ¬(x^3 > 1) ∧ -- Not Statement 2
  ¬(-1 < x ∧ x < 0) := -- Not Statement 3
sorry

end max_true_statements_l690_690198


namespace not_always_greater_quotient_l690_690352

theorem not_always_greater_quotient (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : 0 < b) : ¬ (∀ b < 1, a / b > a) ∧ ¬ (∀ b > 1, a / b > a) :=
by sorry

end not_always_greater_quotient_l690_690352


namespace behavior_of_r_as_m_approaches_zero_l690_690738

open Real

def L (m : ℝ) : ℝ := -1 - sqrt (m + 7)
def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem behavior_of_r_as_m_approaches_zero :
  tendsto (λ m, r m) (nhds 0) (nhds (1 / sqrt 7)) :=
sorry

end behavior_of_r_as_m_approaches_zero_l690_690738


namespace g_at_8_l690_690614

def g (x : ℝ) : ℝ := sorry

axiom g_property : ∀ x y : ℝ, x * g y = y * g x

axiom g_at_24 : g 24 = 12

theorem g_at_8 : g 8 = 4 := by
  sorry

end g_at_8_l690_690614


namespace select_arcs_cover_circle_l690_690337

-- Define a concept of an arc on the circle
structure Arc :=
  (measure : ℝ)
  (covers : ℝ → Prop)   -- Predicate describing whether the arc covers a point on the circle

-- Define a concept of a circle covered by several arcs
def covers_circle (arcs : set Arc) : Prop :=
  ∀ θ : ℝ, ∃ a ∈ arcs, a.covers θ

-- Define the property of arcs not covering the entire circle alone
def not_covering_alone (arcs : set Arc) : Prop :=
  ∀ a ∈ arcs, ∃ θ : ℝ, ¬ a.covers θ 

-- Main theorem statement
theorem select_arcs_cover_circle (arcs : set Arc) : covers_circle arcs ∧ not_covering_alone arcs → 
  ∃ subset_arcs ⊆ arcs, covers_circle subset_arcs ∧ ∑ (a ∈ subset_arcs), a.measure ≤ 720 :=
  by
    sorry

end select_arcs_cover_circle_l690_690337


namespace problem_l690_690118

def f (x : ℝ) : ℝ :=
if x < 1 then 1 + Real.log10 (2 - x)
else 10^(x - 1)

theorem problem (x : ℝ) (x_log : x = Real.log10 30) : f (-98) + f x = 6 := by
  sorry

end problem_l690_690118


namespace general_solution_l690_690768

-- Definitions and conditions
def equation (y : ℝ → ℝ) := deriv (deriv y) + 4 * y = 12 * Real.cos (2 * x)

-- Statement to prove
theorem general_solution (C₁ C₂ : ℝ) : 
  ∃ y : ℝ → ℝ, y = λ x, C₁ * Real.cos (2 * x) + C₂ * Real.sin (2 * x) + 3 * x * Real.sin (2 * x) 
  ∧ equation y := 
sorry

end general_solution_l690_690768


namespace find_initial_number_l690_690066

theorem find_initial_number (x : ℕ) (h : ∃ y : ℕ, x * y = 4 ∧ y = 2) : x = 2 :=
by
  sorry

end find_initial_number_l690_690066


namespace necessary_but_not_sufficient_condition_l690_690680

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l690_690680


namespace marco_might_need_at_least_n_tables_n_tables_are_sufficient_l690_690040
open Function

variables (n : ℕ) (friends_sticker_sets : Fin n → Finset (Fin n))

-- Each friend is missing exactly one unique sticker
def each_friend_missing_one_unique_sticker :=
  ∀ i : Fin n, ∃ j : Fin n, friends_sticker_sets i = (Finset.univ \ {j})

-- A pair of friends is wholesome if their combined collection has all stickers
def is_wholesome_pair (i j : Fin n) :=
  ∀ s : Fin n, s ∈ friends_sticker_sets i ∨ s ∈ friends_sticker_sets j

-- Main problem statements
-- Problem 1: Marco might need to reserve at least n different tables
theorem marco_might_need_at_least_n_tables 
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) : 
  ∃ i j : Fin n, i ≠ j ∧ is_wholesome_pair n friends_sticker_sets i j :=
sorry

-- Problem 2: n tables will always be enough for Marco to achieve his goal
theorem n_tables_are_sufficient
  (h : each_friend_missing_one_unique_sticker n friends_sticker_sets) :
  ∃ arrangement : Fin n → Fin n, ∀ i j, i ≠ j → arrangement i ≠ arrangement j :=
sorry

end marco_might_need_at_least_n_tables_n_tables_are_sufficient_l690_690040


namespace count_two_digit_primes_with_digit_sum_10_l690_690883

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690883


namespace two_digit_prime_sum_to_ten_count_l690_690837

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690837


namespace factor_difference_of_squares_l690_690413

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l690_690413


namespace two_digit_prime_sum_digits_10_count_l690_690926

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690926


namespace two_digit_prime_sum_digits_10_count_l690_690934

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690934


namespace blind_box_problem_l690_690375

theorem blind_box_problem (x y : ℕ) :
  x + y = 135 ∧ 2 * x = 3 * y :=
sorry

end blind_box_problem_l690_690375


namespace max_real_numbers_on_board_l690_690018

theorem max_real_numbers_on_board (S : set ℝ)
  (h : ∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → a + b ∈ S ∨ b + c ∈ S ∨ a + c ∈ S) :
  card S ≤ 7 :=
sorry

end max_real_numbers_on_board_l690_690018


namespace probability_product_is_six_l690_690433

-- Define the set of numbers
def S : Set ℕ := {1, 2, 3, 6}

-- Define the event of drawing two numbers
def draw_two_numbers (s : Set ℕ) : Set (ℕ × ℕ) := 
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x < y }

-- Define the event where the product of two numbers is 6
def product_is_six (e : Set (ℕ × ℕ)) : Set (ℕ × ℕ) :=
  { (x, y) | (x, y) ∈ e ∧ x * y = 6 }

-- Calculate the probability
def probability (e f : Set (ℕ × ℕ)) : ℚ :=
  (f.card : ℚ) / (e.card : ℚ)

theorem probability_product_is_six : 
  probability (draw_two_numbers S) (product_is_six (draw_two_numbers S)) = 1 / 3 :=
sorry

end probability_product_is_six_l690_690433


namespace num_two_digit_primes_with_digit_sum_10_l690_690780

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690780


namespace number_of_positive_numbers_is_two_l690_690253

def positive_numbers : List ℝ := [-2, 3.5, 0, -2/3, -0.7, 11]

def count_positives (l : List ℝ) : Nat :=
  l.countp (λ x => 0 < x)

theorem number_of_positive_numbers_is_two :
  count_positives positive_numbers = 2 :=
by
  sorry

end number_of_positive_numbers_is_two_l690_690253


namespace fibonacci_factorial_series_last_two_digits_sum_l690_690028

-- Define the list of Fibonacci terms as stated in the problem.
def fibonacci_terms : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34]

-- A function to compute the last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Sum of the last two digits of the factorials of the given Fibonacci terms
def sum_last_two_digits_of_factorials : ℕ :=
  fibonacci_terms.map (λ n => last_two_digits (Nat.factorial n)).sum

-- The theorem that states the problem
theorem fibonacci_factorial_series_last_two_digits_sum :
  sum_last_two_digits_of_factorials = 50 :=
by
  sorry

end fibonacci_factorial_series_last_two_digits_sum_l690_690028


namespace cheese_original_weight_l690_690298

theorem cheese_original_weight (final_weight : ℕ)
    (after_bites : ∀(n: ℕ) (w: ℕ), n ≤ 3 → (n = 0 → w = final_weight) ∧
                  (n = 1 → w = 40) ∧
                  (n = 2 → w = 100 ∧ 60) ∧
                  (n = 3 → w = 260 ∧ 160) ∧
                  (n = 4 → w = 420 ∧ 260)) :
    final_weight = 680 := by
    sorry

end cheese_original_weight_l690_690298


namespace primes_between_50_and_100_with_prime_remainder_is_7_l690_690499
-- Importing the Mathlib library

-- The proof problem in Lean 4 statement
theorem primes_between_50_and_100_with_prime_remainder_is_7 :
  let primes_range := [p | p ∈ (list.range (100 - 50 + 1)).map (fun i => i + 50), Nat.prime p]
      primes_remainders := [r for r in primes_range, r % 10 == 1 ∨ r % 10 == 3 ∨ r % 10 == 7]
  in primes_remainders.length = 7 := 
by
  sorry

end primes_between_50_and_100_with_prime_remainder_is_7_l690_690499


namespace f_sum_eq_nine_l690_690563

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.logb 2 (2 - x)
  else 2^(x - 1)

theorem f_sum_eq_nine : f (-2) + f (Real.logb 2 12) = 9 := by
  sorry

end f_sum_eq_nine_l690_690563


namespace part_I_part_II_cosB_range_part_III_f_max_l690_690327

-- Definitions
def f (x : ℝ) := 2 * real.cos x * real.sin (x + real.pi / 4) - real.sqrt 2 / 2

-- Statements
theorem part_I : real.T = real.pi := sorry

theorem part_II_cosB_range (a b c : ℝ) (B : ℝ) (h1 : b^2 = a * c) (h2 : b < a) (h3 : b < c) :
  1/2 ≤ real.cos B ∧ real.cos B < 1 :=
sorry

theorem part_III_f_max (a b c B : ℝ) (h1 : b^2 = a * c) (h2 : b < a) (h3 : b < c)
  (h4 : 0 < B) (h5 : B ≤ real.pi / 2) :
  ∃ B_max, f B_max = -real.sqrt 2 / 2 :=
sorry

end part_I_part_II_cosB_range_part_III_f_max_l690_690327


namespace minimize_squared_distances_l690_690324

variable {P : ℝ}

/-- Points A, B, C, D, E are collinear with distances AB = 3, BC = 3, CD = 5, and DE = 7 -/
def collinear_points : Prop :=
  ∀ (A B C D E : ℝ), B = A + 3 ∧ C = B + 3 ∧ D = C + 5 ∧ E = D + 7

/-- Define the squared distance function -/
def squared_distances (P A B C D E : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Statement of the proof problem -/
theorem minimize_squared_distances :
  collinear_points →
  ∀ (A B C D E P : ℝ), 
    squared_distances P A B C D E ≥ 181.2 :=
by
  sorry

end minimize_squared_distances_l690_690324


namespace radioactive_substance_probability_no_emission_l690_690539

noncomputable def λ : ℝ := 3.87 / 7.5
noncomputable def probability_of_no_emission : ℝ := real.exp (-λ)

theorem radioactive_substance_probability_no_emission (h : λ = 3.87 / 7.5) : 
    abs (probability_of_no_emission - 0.596) < 0.001 := by 
  sorry

end radioactive_substance_probability_no_emission_l690_690539


namespace failed_in_both_percent_l690_690165

-- Definitions
def total_candidates : ℕ := 3000
def failed_english_percent : ℝ := 0.49
def failed_hindi_percent : ℝ := 0.36
def passed_english_alone : ℝ := 630 / total_candidates

-- Theorem statement
theorem failed_in_both_percent
    (total_candidates = 3000)
    (failed_english_percent = 0.49)
    (failed_hindi_percent = 0.36)
    (passed_english_alone = 630 / 3000) : 
    ∃ x : ℝ, x = 0.15 := sorry

end failed_in_both_percent_l690_690165


namespace value_at_2_100_l690_690572

def sequence (b : ℕ → ℕ) : Prop :=
  (b 1 = 2) ∧ (∀ n : ℕ, n > 0 → b (2 * n) = n * b n) ∧ (∀ n : ℕ, n > 0 → b (2 * n + 1) = b (2 * n) + 1)

theorem value_at_2_100 (b : ℕ → ℕ) (h : sequence b) : b (2 ^ 100) = 2 ^ 101 :=
by
  sorry

end value_at_2_100_l690_690572


namespace const_term_binom_expansion_l690_690750

theorem const_term_binom_expansion : 
  let expr := (x - 2/x) ^ 6 in 
  ∀ (C : ℕ -> ℕ -> ℕ), (C 6 3) * (-2) ^ 3 = -160 := by sorry

end const_term_binom_expansion_l690_690750


namespace jennifer_interviews_both_clubs_l690_690182

theorem jennifer_interviews_both_clubs :
  let total_students := 30
  let chess_club := 22
  let drama_club := 20
  let both_clubs := chess_club + drama_club - total_students
  let only_chess := chess_club - both_clubs
  let only_drama := drama_club - both_clubs
  let total_ways := Nat.choose total_students 2
  let chess_ways := Nat.choose only_chess 2
  let drama_ways := Nat.choose only_drama 2
  let neither_ways := chess_ways + drama_ways
  1 - (neither_ways / total_ways) = 362 / 435 :=
by
  sorry

end jennifer_interviews_both_clubs_l690_690182


namespace convert_cylindrical_to_rectangular_l690_690388

noncomputable theory

open Real

-- Define the cylindrical coordinates
def r : ℝ := 5
def theta : ℝ := π / 3
def z_cylindrical : ℝ := 2

-- Define the expected rectangular coordinates
def x_rect : ℝ := 2.5
def y_rect : ℝ := 5 * sqrt(3) / 2
def z_rect : ℝ := 2

-- Lean 4 statement to verify conversion
theorem convert_cylindrical_to_rectangular
  (r θ z_cylindrical x_rect y_rect z_rect : ℝ)
  (hr : r = 5) (htheta : θ = π / 3) (hz : z_cylindrical = 2)
  (hx : x_rect = 5 * cos (π / 3)) (hy : y_rect = 5 * sin (π / 3)) (hz_rect : z_rect = z_cylindrical) :
  (x_rect, y_rect, z_rect) = (2.5, 5 * sqrt(3) / 2, 2) :=
by { 
  rw [hr, htheta, hz] at *,
  rw cos_pi_div_three at hx, 
  rw sin_pi_div_three at hy,
  exact ⟨hx, hy, hz_rect⟩,
  sorry
}

end convert_cylindrical_to_rectangular_l690_690388


namespace count_two_digit_prime_with_digit_sum_10_l690_690868

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690868


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690919

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690919


namespace find_x_value_l690_690151

noncomputable def check_x (x : ℝ) : Prop :=
  (0 < x) ∧ (Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10)

theorem find_x_value (x : ℝ) (h : check_x x) : x = 1 / 6 :=
by 
  sorry

end find_x_value_l690_690151


namespace smallest_x_l690_690307

theorem smallest_x 
  (x : ℝ)
  (h : ( ( (5 * x - 20) / (4 * x - 5) ) ^ 2 + ( (5 * x - 20) / (4 * x - 5) ) ) = 6 ) :
  x = -10 / 3 := sorry

end smallest_x_l690_690307


namespace num_two_digit_primes_with_digit_sum_10_l690_690825

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690825


namespace fox_alice_two_digit_number_l690_690082

theorem fox_alice_two_digit_number : 
    {n : ℕ | 10 ≤ n ∧ n < 100 ∧ 
                  ( (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ (n % 5 ≠ 0 ∨ n % 6 ≠ 0)) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 6 ≠ 0) ∨
                    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0 ∧ n % 6 ≠ 0) ∨
                    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0)
                 )}.card = 8 := 
by 
  sorry

end fox_alice_two_digit_number_l690_690082


namespace total_people_l690_690694

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l690_690694


namespace exists_simple_closed_ngon_l690_690570

theorem exists_simple_closed_ngon (n : ℕ) (hn : 3 ≤ n)
  (points : Fin n → ℝ × ℝ) (hcollinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → collinear (points i) (points j) (points k) → false) :
  ∃ (polygon : List (ℝ × ℝ)), simple_closed_ngon polygon ∧ (∀ i, points i ∈ polygon) :=
  sorry

end exists_simple_closed_ngon_l690_690570


namespace roll_contains_25_coins_l690_690754

variable (coins_per_roll : ℕ)

def rolls_per_teller := 10
def number_of_tellers := 4
def total_coins := 1000

theorem roll_contains_25_coins : 
  (number_of_tellers * rolls_per_teller * coins_per_roll = total_coins) → 
  (coins_per_roll = 25) :=
by
  sorry

end roll_contains_25_coins_l690_690754


namespace perimeter_non_shaded_region_l690_690240

theorem perimeter_non_shaded_region
  (total_area : ℝ)
  (shaded_area : ℝ)
  (length_large_rect : ℝ)
  (width_large_rect : ℝ)
  (length_small_rect : ℝ)
  (width_small_rect : ℝ)
  (non_shaded_perimeter : ℝ) :
  total_area = (length_large_rect * width_large_rect) 
    + (length_small_rect * width_small_rect) →
  shaded_area = 104 →
  total_area - shaded_area = (length_large_rect - 2 * length_small_rect) * width_non_shaded →
  width_non_shaded = (total_area - shaded_area) / (length_large_rect - 2 * length_small_rect) →
  non_shaded_perimeter = 2 * (length_large_rect - 2 * length_small_rect + width_non_shaded) →
  non_shaded_perimeter = 21.34 :=
begin
  sorry
end

end perimeter_non_shaded_region_l690_690240


namespace five_letter_words_with_two_vowels_l690_690134

theorem five_letter_words_with_two_vowels (S : set char) (hS : S = {'A', 'B', 'C', 'D', 'E', 'F'}) : 
    ∃ n : ℕ, n = 672 ∧ ∀ w : string, (w.length = 5 ∧ ∀ c : char, c ∈ S → count c w ≤ 1 ∧ count (char) w 'A' + count (char) w 'E' ≥ 2) ↔ n = 672 :=
by
  sorry

end five_letter_words_with_two_vowels_l690_690134


namespace count_two_digit_prime_with_digit_sum_10_l690_690878

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690878


namespace max_value_of_function_y_l690_690769

noncomputable def function_y (x : ℝ) : ℝ :=
  tan (x + 3 * Real.pi / 4) - tan (x + Real.pi / 3) + cos (x + Real.pi / 3) + sin (x + Real.pi / 4)

theorem max_value_of_function_y :
  ∃ x : ℝ, (-5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 4) ∧
           function_y x = (3 / 2 : ℝ) - Real.sqrt 3 - Real.sqrt 2 / 2 :=
sorry

end max_value_of_function_y_l690_690769


namespace division_largest_smallest_l690_690285

theorem division_largest_smallest :
  let nums := [10, 11, 12]
  let largest := list.maximum nums
  let smallest := list.minimum nums
  largest / smallest = 1.2 :=
by {
  have h_max : largest = 12, from sorry,
  have h_min : smallest = 10, from sorry,
  rw [h_max, h_min],
  norm_num,
}

end division_largest_smallest_l690_690285


namespace characterization_of_points_l690_690385

def satisfies_eq (x : ℝ) (y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem characterization_of_points :
  ∀ x y : ℝ,
  satisfies_eq x y ↔
  ((x < 0 ∨ x > 1) ∧ (y < 0 ∨ y > 1) ∧ y ≤ 0 ∧ y = 1 - x^2) ∨
  ((x < 0 ∨ x > 1) ∧ (0 < y ∧ y < 1) ∧ x^2 + y^2 = 1 ∧ x ≤ -1 ∧ x > 0) ∨
  ((0 < x ∧ x < 1) ∧ (y < 0 ∨ y > 1) ∧ false) ∨
  ((0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y^2 = 1 - x) :=
sorry

end characterization_of_points_l690_690385


namespace range_of_a_l690_690515

noncomputable def f (x a : ℝ) : ℝ := (sin x + a)/(cos x)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Ioo 0 (π/2), monotone_on (λ x, f x a) (Ioo 0 (π/2))) → (a ≥ -1) :=
begin
  sorry
end

end range_of_a_l690_690515


namespace arithmetic_sequence_ninth_term_l690_690517

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem arithmetic_sequence_ninth_term
  (a: ℕ → ℕ)
  (h_arith: is_arithmetic_sequence a)
  (h_sum_5: sum_of_first_n_terms a 5 = 75)
  (h_a4: a 4 = 2 * a 2) :
  a 9 = 45 :=
sorry

end arithmetic_sequence_ninth_term_l690_690517


namespace horse_food_needed_l690_690726

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed_l690_690726


namespace count_two_digit_primes_with_digit_sum_10_l690_690986

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690986


namespace burgers_per_day_l690_690075

theorem burgers_per_day (cost_per_burger : ℕ) (total_spent : ℕ) (days_in_june : ℕ) (h_cost : cost_per_burger = 12) (h_spent : total_spent = 720) (h_days : days_in_june = 30) : (total_spent / cost_per_burger) / days_in_june = 2 :=
by
  rw [h_cost, h_spent, h_days]
  norm_num
  sorry

end burgers_per_day_l690_690075


namespace count_two_digit_primes_with_digit_sum_10_l690_690890

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690890


namespace george_total_blocks_l690_690434

-- Definitions (conditions).
def large_boxes : ℕ := 5
def small_boxes_per_large_box : ℕ := 8
def blocks_per_small_box : ℕ := 9
def individual_blocks : ℕ := 6

-- Mathematical proof problem statement.
theorem george_total_blocks :
  (large_boxes * small_boxes_per_large_box * blocks_per_small_box + individual_blocks) = 366 :=
by
  -- Placeholder for proof.
  sorry

end george_total_blocks_l690_690434


namespace shortest_distance_log_exp_l690_690271

noncomputable def curve (x : ℝ) : ℝ := Real.log x
noncomputable def line (x : ℝ) : ℝ := Real.exp x - 2

theorem shortest_distance_log_exp (x : ℝ) (h : x > 0) :
  ∃ (P : ℝ × ℝ), P = (x, curve x) ∧ line x = curve x → 0 :=
by sorry

end shortest_distance_log_exp_l690_690271


namespace inscribed_triangle_l690_690326

theorem inscribed_triangle (XYZ : Triangle) (M N P : Point)
  (on_YZ : PointOnLineSegment M YZ) (on_XZ : PointOnLineSegment N XZ) (on_XY : PointOnLineSegment P XY)
  (XY_len : dist X Y = 21) (YZ_len : dist Y Z = 27) (XZ_len : dist X Z = 26)
  (arc_NP_eq_arc_MZ : measure Arc NP = measure Arc MZ)
  (arc_MP_eq_arc_NZ : measure Arc MP = measure Arc NZ)
  (arc_MN_eq_arc_PZ : measure Arc MN = measure Arc PZ) :
  let PZ := 14 / 1 in ∀ m n : ℕ, m + n = 15 := by
  sorry

end inscribed_triangle_l690_690326


namespace perp_2n_sub_m_projection_n_dir_n_add_m_l690_690483

variables {R : Type*} [inner_product_space ℝ R] {m n : R}
variable (n m)

-- a) Define unit vectors
def is_unit_vector (v : R) : Prop := ∥v∥ = 1

-- a) Define angle condition
def angle_sixty_degrees (u v : R) : Prop := real_inner u v = ∥u∥ * ∥v∥ * 0.5

-- b) Statement for part 1: proving perpendicularity
theorem perp_2n_sub_m (h1 : is_unit_vector m) (h2 : is_unit_vector n) (h3 : angle_sixty_degrees m n) :
  real_inner (2 • n - m) m = 0 := sorry

-- b) Statement for part 2: projection calculation
theorem projection_n_dir_n_add_m (h1 : is_unit_vector m) (h2 : is_unit_vector n) (h3 : angle_sixty_degrees m n) :
  ∥(real_inner n (n + m) / ∥n + m∥ * (n + m))∥ = real.sqrt(3) / 2 := sorry

end perp_2n_sub_m_projection_n_dir_n_add_m_l690_690483


namespace salad_cost_is_correct_l690_690426

-- Definitions of costs according to the given conditions
def muffin_cost : ℝ := 2
def coffee_cost : ℝ := 4
def soup_cost : ℝ := 3
def lemonade_cost : ℝ := 0.75

def breakfast_cost : ℝ := muffin_cost + coffee_cost
def lunch_cost : ℝ := breakfast_cost + 3

def salad_cost : ℝ := lunch_cost - (soup_cost + lemonade_cost)

-- Statement to prove
theorem salad_cost_is_correct : salad_cost = 5.25 :=
by
  sorry

end salad_cost_is_correct_l690_690426


namespace perpendicular_tangents_l690_690275

noncomputable def intersection_point (a : ℝ) : ℝ :=
Real.cbrt a

noncomputable def slope_curve_1 (a x : ℝ) : ℝ :=
- a / (x^2)

noncomputable def slope_curve_2 (x : ℝ) : ℝ :=
2 * x

theorem perpendicular_tangents (a : ℝ) (h : slope_curve_1 a (intersection_point a) * slope_curve_2 (intersection_point a) = -1) :
  a = sqrt 2 / 4 ∨ a = - sqrt 2 / 4 :=
by
  sorry

end perpendicular_tangents_l690_690275


namespace total_people_in_group_l690_690700

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l690_690700


namespace empty_holes_count_l690_690320

theorem empty_holes_count (w h : ℕ) (interval : ℕ) (total_diagonals : ℕ) (overlap : ℕ) : 
  w = 10 → h = 10 → interval = 1 → total_diagonals = 2 → overlap = 1 →
  let total_holes := ((w + 1) * (h + 1)) in
  let diagonal_holes := (w + 1) * total_diagonals - overlap in
  let empty_holes := total_holes - diagonal_holes in
  empty_holes = 100 :=
by
  intros w_eq h_eq interval_eq total_diagonals_eq overlap_eq
  have total_holes : (w + 1) * (h + 1) = 121 := by
    simp [w_eq, h_eq, interval_eq]
  have diagonal_holes : (w + 1) * total_diagonals - overlap = 21 := by
    simp [w_eq, total_diagonals_eq, overlap_eq]
  have empty_holes : total_holes - diagonal_holes = 100 := by
    simp [total_holes, diagonal_holes]
  sorry

end empty_holes_count_l690_690320


namespace trig_expression_l690_690144

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := 
  sorry

end trig_expression_l690_690144


namespace count_two_digit_primes_with_digit_sum_10_l690_690889

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690889


namespace problem_equivalent_l690_690960

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690960


namespace proof_problem_l690_690154

noncomputable def question_conditions : Prop :=
  ∀ z : ℂ, (complex.I * z = -(1 / 2 : ℂ) * (1 + complex.I)) → 
  complex.imag (complex.conj z) = -1 / 2

theorem proof_problem : question_conditions := 
by 
  sorry

end proof_problem_l690_690154


namespace monotonic_intervals_and_min_distance_l690_690205

noncomputable def f (x : ℝ) := 1/x
noncomputable def g (x : ℝ) := Real.log2 (2 - |x + 1|)

theorem monotonic_intervals_and_min_distance :
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1), ∀ y ∈ Set.Ioo (-3 : ℝ) (-1), x < y → g x < g y) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) (1), ∀ y ∈ Set.Ioo (-1 : ℝ) (1), x < y → g x > g y) ∧
  (let M : ℝ × ℝ := (-1, 1)
   ∃ x0 > 0, (x0 - 1/x0 + 1)^2 + 3 = 3 ∧
              (|M.fst - x0|^2 + |M.snd - f x0|^2)^2 = 3) := sorry

end monotonic_intervals_and_min_distance_l690_690205


namespace find_polynomial_satisfies_conditions_l690_690055

noncomputable def polynomial_satisfies_conditions : Prop :=
  ∀ P : Polynomial ℝ, 
    (∀ a b c : ℝ, ab + bc + ca = 0 →
      P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) ↔ 
        ∃ a b : ℝ, P = (Polynomial.C a * X^4 + Polynomial.C b * X^2)

theorem find_polynomial_satisfies_conditions : polynomial_satisfies_conditions :=
  sorry

end find_polynomial_satisfies_conditions_l690_690055


namespace count_two_digit_primes_with_digit_sum_10_l690_690887

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690887


namespace green_hat_cost_l690_690647

theorem green_hat_cost (G : ℝ) (total_hats : ℕ) (blue_hats : ℕ) (green_hats : ℕ) (blue_cost : ℝ) (total_cost : ℝ) 
    (h₁ : blue_hats = 85) (h₂ : blue_cost = 6) (h₃ : green_hats = 90) (h₄ : total_cost = 600) 
    (h₅ : total_hats = blue_hats + green_hats) 
    (h₆ : total_cost = blue_hats * blue_cost + green_hats * G) : 
    G = 1 := by
  sorry

end green_hat_cost_l690_690647


namespace arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l690_690328

open Nat

axiom students : Fin 7 → Type -- Define students indexed by their position in the line.

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem arrangements_A_and_B_together :
  (2 * fact 6) = 1440 := 
by 
  sorry

theorem arrangements_A_not_head_B_not_tail :
  (fact 7 - 2 * fact 6 + fact 5) = 3720 := 
by 
  sorry

theorem arrangements_A_and_B_not_next :
  (3600) = 3600 := 
by 
  sorry

theorem arrangements_one_person_between_A_and_B :
  (fact 5 * 2) = 1200 := 
by 
  sorry

end arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l690_690328


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690972

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690972


namespace segment_length_equality_l690_690682

-- Definitions of points and structures
variables {A B C L M N : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space L] [metric_space M] [metric_space N]

-- Define the triangle ABC and incircle condition
def incircle_touches_L (t : triangle A B C) (i : incircle t) (L : Point L) : Prop :=
  i.touches t.BC L

-- Define the diameter and intersections
def diameter_LM (t : triangle A B C) (i : incircle t) (L M : Point) : Prop :=
  i.diameter L M

def ray_AM_meets_BC_at_N (A M : Point) (BC : line B C) (N : Point) : Prop :=
  ray_intersects_line A M BC N

-- The final goal statement: showing |NL| = |AB - AC|
theorem segment_length_equality (A B C L M N : Point) (t : triangle A B C) (i : incircle t) :
  incircle_touches_L t i L →
  diameter_LM t i L M →
  ray_AM_meets_BC_at_N A M t.BC N →
  |NL| = |AB - AC| := 
sorry

end segment_length_equality_l690_690682


namespace two_digit_prime_sum_digits_10_count_l690_690933

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690933


namespace count_numbers_le_0_l690_690064

theorem count_numbers_le_0.4 : 
  ({0.8, 1/2, 0.3, 1/3} : Set ℝ).count (λ x => x ≤ 0.4) = 2 := 
by sorry

end count_numbers_le_0_l690_690064


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690984

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690984


namespace inheritance_problem_l690_690721

variables (x1 x2 x3 x4 : ℕ)

theorem inheritance_problem
  (h1 : x1 + x2 + x3 + x4 = 1320)
  (h2 : x1 + x4 = x2 + x3)
  (h3 : x2 + x4 = 2 * (x1 + x3))
  (h4 : x3 + x4 = 3 * (x1 + x2)) :
  x1 = 55 ∧ x2 = 275 ∧ x3 = 385 ∧ x4 = 605 :=
by sorry

end inheritance_problem_l690_690721


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690954

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690954


namespace darnel_jogging_l690_690395

variable (j s : ℝ)

theorem darnel_jogging :
  s = 0.875 ∧ s = j + 0.125 → j = 0.750 :=
by
  intros h
  have h1 : s = 0.875 := h.1
  have h2 : s = j + 0.125 := h.2
  sorry

end darnel_jogging_l690_690395


namespace find_a16_l690_690252

def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 4
  | 2 => 6
  | 3 => 9
  | 4 => 10
  | 5 => 14
  | 6 => 15
  | 7 => 21
  | 8 => 22
  | 9 => 25
  | 10 => 26
  | 11 => 33
  | 12 => 34
  | 13 => 35
  | 14 => 38
  | 15 => 39 -- The next term we need to derive correctly which is 46
  | _ => 0  -- Undefined for other cases in this context

noncomputable def isProductOfTwoPrimes (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ n = p * q

noncomputable def a₁₆ : ℕ :=
  Nat.find (λ n, isProductOfTwoPrimes n ∧ n > 38)

theorem find_a16 : a₁₆ = 46 :=
  by
    -- The proof is omitted.
    sorry

end find_a16_l690_690252


namespace f_g_3_is_290_l690_690479

def f(x : ℝ) : ℝ := x^2 + 1
def g(x : ℝ) : ℝ := 2 * x^2 - 1

theorem f_g_3_is_290 : f(g(3)) = 290 := 
by
  sorry

end f_g_3_is_290_l690_690479


namespace avg_consecutive_integers_l690_690737

theorem avg_consecutive_integers (a : ℝ) (b : ℝ) 
  (h₁ : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5)) / 6) :
  (a + 5) = (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5)) / 6 :=
by sorry

end avg_consecutive_integers_l690_690737


namespace closed_form_formula_l690_690058

-- Formalizing the problem
theorem closed_form_formula (n : ℕ) : 
  (∑ k in Finset.range (n+1), if k ≠ 0 then (k * (-1)^k * Nat.choose n k) / (n+k+1) else 0) = 
  -((Nat.factorial n * Nat.factorial (n+1)) / Nat.factorial (2*n+1)) :=
by
  sorry

end closed_form_formula_l690_690058


namespace group_population_l690_690698

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l690_690698


namespace conic_section_eq_of_same_foci_and_point_l690_690765

theorem conic_section_eq_of_same_foci_and_point
  (x y : ℝ)
  (hx : (x = 1))
  (hy : (y = 2))
  (hx_focus : (0, sqrt 6))
  : (y ^ 2 / 8 + x ^ 2 / 2 = 1) ∨ (y ^ 2 / 3 - x ^ 2 / 3 = 1) :=
by
  sorry

end conic_section_eq_of_same_foci_and_point_l690_690765


namespace distance_focus_to_asymptote_l690_690476

-- Define the hyperbola, its asymptotes, and focus
def hyperbola_eq := (x : ℝ) (y : ℝ) := (x^2 / 9) - (y^2 / 5) = 1
def asymptote_eq := (x : ℝ) := (y : ℝ) := y = (sqrt 5 / 3) * x
def focus_1 : ℝ × ℝ := (sqrt 14, 0)

-- Distance formula definition from point to line (asymptote)
noncomputable def distance_point_to_line (focus : ℝ × ℝ) (asymptote : ℝ → ℝ) : ℝ := 
  abs ((focus.1 * (sqrt 5 / 3))) / sqrt (1 + (sqrt 5 / 3)^2)

-- The proof statement
theorem distance_focus_to_asymptote : 
  distance_point_to_line focus_1 asymptote_eq = sqrt 5 := 
sorry

end distance_focus_to_asymptote_l690_690476


namespace integer_solutions_count_2009_l690_690138

theorem integer_solutions_count_2009 :
  ∃ s : Finset (ℤ × ℤ × ℤ), (∀ (x y z : ℤ), (x, y, z) ∈ s ↔ x * y * z = 2009) ∧ s.card = 72 :=
  sorry

end integer_solutions_count_2009_l690_690138


namespace two_digit_prime_sum_to_ten_count_l690_690835

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690835


namespace zero_interval_l690_690114

noncomputable def f (x : ℝ) : ℝ := 2^x + (1/4) * x - 5

theorem zero_interval : ∃ x ∈ set.Ioo (2:ℝ) 3, f x = 0 :=
by
  -- We'd proceed to give a rigorous proof here, but for now we insert a placeholder.
  sorry

end zero_interval_l690_690114


namespace interval_length_max_min_l690_690038

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

theorem interval_length_max_min 
  (a b : ℝ)
  (h1 : ∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ 3^|x| ∧ 3^|x| ≤ 9)
  (h2 : ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ x = 0)
  : (max (interval_length a b) (interval_length b a) = 4) ∧ 
    (min (interval_length a b) (interval_length b a) = 2) :=
sorry

end interval_length_max_min_l690_690038


namespace range_of_a_div_b_l690_690099

theorem range_of_a_div_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : 2 < b ∧ b < 8) : 
  1 / 8 < a / b ∧ a / b < 2 :=
sorry

end range_of_a_div_b_l690_690099


namespace probability_top_card_queen_of_hearts_l690_690713

constant Rank : Type
constant Suit : Type
constant Card : Type

constant Ace : Rank
constant Two : Rank
constant Three : Rank
constant Four : Rank
constant Five : Rank
constant Six : Rank
constant Seven : Rank
constant Eight : Rank
constant Nine : Rank
constant Ten : Rank
constant Jack : Rank
constant Queen : Rank
constant King : Rank

constant Spade : Suit
constant Heart : Suit
constant Diamond : Suit
constant Club : Suit

constant Deck : List Card
constant QueenOfHearts : Card

axiom rank_suit_unique :
  (∀ (r1 r2 : Rank) (s1 s2 : Suit), (r1 ≠ r2 ∨ s1 ≠ s2) → (∃! (c : Card), c ∈ Deck ∧ (c.rank = r1 ∧ c.suit = s1) ∧ (c.rank ≠ r2 ∨ c.suit ≠ s2)))

axiom deck_size : Deck.length = 52

axiom deck_shuffle : ∀ c : Card, c ∈ Deck → (∀ c' : Card, c' ∈ Deck → (c ≠ c' → c.rank ≠ c'.rank ∧ c.suit ≠ c'.suit))

axiom queen_of_hearts_unique : 
  ∃! (c : Card), c ∈ Deck ∧ c.rank = Queen ∧ c.suit = Heart

theorem probability_top_card_queen_of_hearts
  (h : (∀ c : Card, c ∈ Deck → (∀ c' : Card, c' ∈ Deck → (c ≠ c' → c.rank ≠ c'.rank ∧ c.suit ≠ c'.suit)))
  (d_size : Deck.length = 52)
  (q_unique : ∃! (c : Card), c ∈ Deck ∧ c.rank = Queen ∧ c.suit = Heart) :
  (1 / 52 : ℚ) = 1 / 52 :=
by
  sorry

end probability_top_card_queen_of_hearts_l690_690713


namespace find_face_value_l690_690243

theorem find_face_value (BD TD : ℕ) (h1 : BD = 144) (h2 : TD = 120) : ∃ FV : ℕ, FV = 840 :=
by
  use 840
  sorry

end find_face_value_l690_690243


namespace minimum_protein_content_is_at_least_1_8_l690_690334

-- Define the net weight of the can and the minimum protein percentage
def netWeight : ℝ := 300
def minProteinPercentage : ℝ := 0.006

-- Prove that the minimum protein content is at least 1.8 grams
theorem minimum_protein_content_is_at_least_1_8 :
  netWeight * minProteinPercentage ≥ 1.8 := 
by
  sorry

end minimum_protein_content_is_at_least_1_8_l690_690334


namespace no_token_in_initial_square_at_a_given_moment_l690_690012
-- Lean 4 statement: Prove there exists a moment when no token is in its initial square 

theorem no_token_in_initial_square_at_a_given_moment
  (n : ℕ) (h_n_pos : n > 0)
  (tokens : fin n → fin n → bool) -- tokens in certain squares, represented by a boolean matrix
  (initial_state : fin n → fin n → bool) -- initial state of the tokens, similar representation
  (moves : ℕ → (fin n → fin n → bool)) -- function describing the board configuration after each second
  (final_condition : ∀ t, ∃ s : fin n → fin n → bool, s = moves t ∧
    (∀ i j : fin n, (tokens i j = tt → s i j = initial_state i j) ∧ 
    (tokens i j = ff → s i j ≠ initial_state i j))) -- The condition ensuring each token returns to its initial square after visiting all others exactly once 
  : ∃ t, ∀ i j : fin n, moves t i j ≠ initial_state i j := sorry

end no_token_in_initial_square_at_a_given_moment_l690_690012


namespace count_prime_two_digit_sum_ten_is_three_l690_690806

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690806


namespace total_time_is_correct_l690_690632

def initial_distance : ℝ := 300
def run_speed : ℝ := 10
def bike_speed : ℝ := 50

-- Function to calculate the total time
def total_time : ℝ :=
  let first_trip := initial_distance / bike_speed
  let round_trip_for_second := 2 * first_trip
  let second_person_distance_covered := round_trip_for_second * run_speed / bike_speed
  let second_person_trip := (initial_distance - second_person_distance_covered) / bike_speed
  let third_round_trip := 2 * (initial_distance - second_person_distance_covered) / bike_speed
  let third_person_distance_covered := third_round_trip * run_speed / bike_speed
  let third_person_trip := (initial_distance - third_person_distance_covered) / bike_speed
  first_trip + round_trip_for_second + second_person_trip + third_round_trip + third_person_trip

theorem total_time_is_correct : total_time = 35.28 :=
by 
  sorry

end total_time_is_correct_l690_690632


namespace find_roots_of_equation_l690_690761

theorem find_roots_of_equation
  (a b c d x : ℝ)
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c)
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) :
  x = 1007.5 :=
by
  sorry

end find_roots_of_equation_l690_690761


namespace num_two_digit_primes_with_digit_sum_10_l690_690776

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690776


namespace count_two_digit_primes_with_digit_sum_10_l690_690891

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690891


namespace arc_length_DE_inscribed_circle_l690_690338

theorem arc_length_DE_inscribed_circle (r : ℝ) (A B C D E : ℝ) (angle_BAC : ℝ) :
  r = 2 ∧ angle_BAC = 45 ∧ A = C ∧ B = E ∧ D = E → 
  let circumference := 2 * Real.pi * r in
  let arc_length_DE := (angle_BAC * 2 / 360) * circumference in
  arc_length_DE = Real.pi := 
by 
  sorry

end arc_length_DE_inscribed_circle_l690_690338


namespace red_squares_in_block_l690_690149

theorem red_squares_in_block {board : Fin 9 × Fin 9 → Prop} (h : (∑ i, (∑ j, if board (i, j) then 1 else 0)) = 46) :
  ∃ (i j : Fin 8), 
    (if board (i, j) then 1 else 0) + 
    (if board (i, ⟨j.val + 1, by linarith[j.is_lt]⟩) then 1 else 0) + 
    (if board (⟨i.val + 1, by linarith[i.is_lt]⟩, j) then 1 else 0) + 
    (if board (⟨i.val + 1, by linarith[i.is_lt]⟩, ⟨j.val + 1, by linarith[j.is_lt]⟩) then (1 : ℕ) else 0) ≥ 3 := 
by
  sorry

end red_squares_in_block_l690_690149


namespace speed_of_man_l690_690331

theorem speed_of_man (L : ℝ) (T t : ℝ) (Vt_km_h : ℝ) (Vm_km_h : ℝ) 
  (h1 : L = 220) 
  (h2 : t = 10.999120070394369) 
  (h3 : Vt_km_h = 80) 
  (h4 : Vt_km_h * 1000 / 3600 = T)
  (h5 : (T - (Vm_km_h * 1000 / 3600)) * t = L) 
  : Vm_km_h ≈ 8 := 
sorry

end speed_of_man_l690_690331


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690981

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690981


namespace eight_people_lineup_l690_690618

theorem eight_people_lineup : (Nat.fact 8) = 40320 := by
  sorry

end eight_people_lineup_l690_690618


namespace max_min_sum_of_perpendicular_chords_l690_690224

-- Definitions taken directly from the conditions
variables {C : Type*} [metric_space C] (circle : set C) (d : ℝ)
variables (X : C) (AC BD : set C) [is_metric_circle circle d]
variables [metric_chord_perpendicular_through circle X AC BD]

-- The main statement of the theorem
theorem max_min_sum_of_perpendicular_chords (X : C) (hX : X ∈ circle ∨ X ∈ interior_path circle) :
  d ≤ (AC.length + BD.length) ∧ (AC.length + BD.length) ≤ 2 * d := sorry

end max_min_sum_of_perpendicular_chords_l690_690224


namespace angle_BAP_18_degrees_l690_690480

theorem angle_BAP_18_degrees (A B C P : Point)
  (h1 : ∠ P B C = 24)
  (h2 : ∠ P C B = 24)
  (h3 : ∠ A B P = 30)
  (h4 : ∠ A C P = 54) :
  ∠ B A P = 18 :=
sorry

end angle_BAP_18_degrees_l690_690480


namespace infinitely_many_x_for_sin3x_eq_sin2x_plus_sinx_l690_690077

theorem infinitely_many_x_for_sin3x_eq_sin2x_plus_sinx :
  ∃ᶠ x in (Filter.atTop : Filter ℝ), sin (3 * x) = sin (2 * x) + sin x :=
by
  sorry

end infinitely_many_x_for_sin3x_eq_sin2x_plus_sinx_l690_690077


namespace num_two_digit_primes_with_digit_sum_10_l690_690784

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690784


namespace two_digit_primes_with_digit_sum_ten_l690_690790

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690790


namespace binomial_coeff_x_squared_l690_690610

theorem binomial_coeff_x_squared :
  let x := 1 in
  let coeff := (8.choose 3) * (-2)^3 in
  coeff = -448 :=
by 
  let x := 1
  let coeff := (8.choose 3) * (-2)^3
  show coeff = -448
sorry

end binomial_coeff_x_squared_l690_690610


namespace pairs_sold_l690_690318

theorem pairs_sold (total_sales : ℝ) (avg_price_per_pair : ℝ) (h1 : total_sales = 490) (h2 : avg_price_per_pair = 9.8) :
  total_sales / avg_price_per_pair = 50 :=
by
  rw [h1, h2]
  norm_num

end pairs_sold_l690_690318


namespace jimmy_needs_4_packs_of_bread_l690_690185

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l690_690185


namespace num_two_digit_primes_with_digit_sum_10_l690_690831

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690831


namespace determine_pet_crocodile_l690_690582

-- Definitions based on the conditions provided
inductive Resident
| knight    -- A knight who always tells the truth
| liar      -- A liar who always lies

-- Predicate reflecting whether a resident has a pet crocodile
def has_pet_crocodile : Resident → Prop

-- The key question to ask a resident
def key_question (r : Resident) : String :=
  "What would you have answered if I asked you one minute ago whether you have a pet crocodile at home?"

-- The theorem we need to prove
theorem determine_pet_crocodile (r : Resident) :
  ¬ (has_pet_crocodile r = tt ↔ key_question r ="No") := 
sorry

end determine_pet_crocodile_l690_690582


namespace slower_truck_passing_time_l690_690676

noncomputable def speed_kmh_to_ms (v : ℕ) : ℝ :=
  (v * 1000) / (60 * 60 : ℕ)

noncomputable def faster_truck_speed : ℝ := speed_kmh_to_ms 30
noncomputable def slower_truck_speed : ℝ := speed_kmh_to_ms 20

noncomputable def relative_speed : ℝ :=
  faster_truck_speed + slower_truck_speed

noncomputable def truck_length : ℕ := 250

noncomputable def passing_time : ℝ :=
  truck_length / relative_speed

theorem slower_truck_passing_time : 
  passing_time ≈ 18 :=
by
  sorry

end slower_truck_passing_time_l690_690676


namespace count_two_digit_primes_with_digit_sum_10_l690_690852

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690852


namespace largest_k_l690_690628

/-- Define the recursive tower function T -/
def T : ℕ → ℕ
| 1     := 3
| (n+1) := 3 ^ T n

/-- Define A as (T 6)^2 -/
def A : ℕ := (T 6) ^ 2

/-- Define B as T 6 to the power of A -/
def B : ℕ := (T 6) ^ A

/-- Prove that the largest integer k for which log_3^k B is defined is 6 -/
theorem largest_k (k : ℕ) : k = 6 ↔ (∃ B : ℕ, B = (T 6) ^ ((T 6) ^ 2)) :=
by {
  sorry
}

end largest_k_l690_690628


namespace num_two_digit_primes_with_digit_sum_10_l690_690785

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690785


namespace inverse_of_log10_l690_690574

noncomputable def f (x : ℝ) : ℝ := log10 x

noncomputable def f_inv (x : ℝ) : ℝ := 10^x

theorem inverse_of_log10 : ∀ x : ℝ, f (f_inv x) = x :=
by
  sorry

end inverse_of_log10_l690_690574


namespace ratio_of_fraction_part_l690_690584

theorem ratio_of_fraction_part (N : ℕ) (h1 : (1 / 3) * (2 / 5) * N = 20) (h2 : 0.4 * N = 240) :
  20 / ((1 / 3) * (2 / 5) * N) = 2 / 15 :=
by
  sorry

end ratio_of_fraction_part_l690_690584


namespace count_prime_two_digit_sum_ten_is_three_l690_690809

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690809


namespace num_two_digit_primes_with_digit_sum_10_l690_690821

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690821


namespace diff_of_two_numbers_l690_690612

theorem diff_of_two_numbers :
  ∃ D S : ℕ, (1650 = 5 * S + 5) ∧ (D = 1650 - S) ∧ (D = 1321) :=
sorry

end diff_of_two_numbers_l690_690612


namespace triangle_probability_calculation_l690_690031

-- Definitions
noncomputable def num_vertices : ℕ := 15
noncomputable def total_segments : ℕ := (num_vertices * (num_vertices - 1)) / 2
noncomputable def possible_segments : list ℝ := list.map (λ k, 2 * real.sin (k * real.pi / num_vertices)) (list.range' 1 (num_vertices / 2 + 1))

-- Proposition: The probability that three randomly selected segments form a triangle with positive area
noncomputable def valid_triangle_count : ℕ := begin
  let possible_triplets := list.triplets possible_segments,
  let valid_triplets := possible_triplets.filter (λ ⟨a, b, c⟩, (a + b > c) ∧ (a + c > b) ∧ (b + c > a)),
  exact valid_triplets.length,
end

noncomputable def total_triplets : ℕ := nat.choose total_segments 3

noncomputable def probability_triangle : ℚ := valid_triangle_count / total_triplets

theorem triangle_probability_calculation :
  probability_triangle ∈ {
    341 // 455,
    323 // 455,
    309 // 455,
    297 // 455,
    280 // 455
  } :=
sorry

end triangle_probability_calculation_l690_690031


namespace count_two_digit_primes_with_digit_sum_10_l690_690856

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690856


namespace largest_sum_of_digits_l690_690505

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) 
  (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10)
  (h₃ : 0.abc = (1 : ℚ) / y)
  (h₄ : y > 0 ∧ y ≤ 16) : a + b + c ≤ 8 :=
by sorry

end largest_sum_of_digits_l690_690505


namespace count_two_digit_prime_with_digit_sum_10_l690_690877

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690877


namespace minimum_distance_parabola_line_l690_690558

def parabola (x : ℝ) : ℝ := x^2 - 4*x + 7
def line (x : ℝ) : ℝ := 2*x - 5

def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem minimum_distance_parabola_line :
  ∃ p q : ℝ, parabola p = p^2 - 4*p + 7 ∧ line q = 2*q - 5 ∧ 
    distance (p, parabola p) (q, line q) = 3 * real.sqrt 5 / 5 :=
sorry

end minimum_distance_parabola_line_l690_690558


namespace total_teachers_correct_l690_690226

-- Define the number of departments and the total number of teachers
def num_departments : ℕ := 7
def total_teachers : ℕ := 140

-- Proving that the total number of teachers is 140
theorem total_teachers_correct : total_teachers = 140 := 
by
  sorry

end total_teachers_correct_l690_690226


namespace chord_length_of_circle_l690_690607

theorem chord_length_of_circle (O A B : Point) (R : ℝ) 
  (h1: circle_with_radius_OA O A B R)
  (h2: angle_OA_OB_is_60_deg O A B) :
  chord_length_AB_is_R O A B R := 
sorry

end chord_length_of_circle_l690_690607


namespace correct_average_is_20_87_l690_690161

-- Defining the problem parameters
def n : ℕ := 15
def incorrect_avg : ℝ := 20
def incorrect_readings : list ℝ := [35, 60, 25]
def correct_readings : list ℝ := [45, 58, 30]

-- Incorrect sum calculation
def incorrect_sum : ℝ := incorrect_avg * n

-- Sum of incorrect and correct values
def sum_incorrect_values : ℝ := incorrect_readings.sum
def sum_correct_values : ℝ := correct_readings.sum

-- Difference to adjust the sum
def difference : ℝ := sum_correct_values - sum_incorrect_values

-- Correct total sum
def correct_sum : ℝ := incorrect_sum + difference

-- Correct average calculation
def correct_avg : ℝ := correct_sum / n

-- The required proof statement
theorem correct_average_is_20_87 : correct_avg = 20.87 := by
  sorry

end correct_average_is_20_87_l690_690161


namespace rows_seating_l690_690043

theorem rows_seating (x y : ℕ) (h : 7 * x + 6 * y = 52) : x = 4 :=
by
  sorry

end rows_seating_l690_690043


namespace two_digit_primes_with_digit_sum_10_count_l690_690908

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690908


namespace factorize_1729_min_diff_l690_690619

theorem factorize_1729_min_diff :
  ∃ (a1 a2 b1 b2 : ℕ), 
    1729 = (nat.factorial a1) * (nat.factorial a2) / ((nat.factorial b1) * (nat.factorial b2)) ∧ 
    a1 ≥ a2 ∧
    b1 ≥ b2 ∧ 
    (a1 + b1 = 36) ∧  -- since 19+17=36 is the minimized sum that makes up the prime factors.
    (|a1 - b1| = 2) :=
sorry

end factorize_1729_min_diff_l690_690619


namespace arithmetic_sequence_sum_l690_690325

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ), (∃ (A B C O : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≠ O ∧ B ≠ O ∧ C ≠ O ∧
    (∃ a₁ a₁₀₀ : ℝ,
      (\lambda n, a n) 1 = a₁ ∧ (\lambda n, a n) 100 = a₁₀₀ ∧
      a₁ + a₁₀₀ = 1 ∧
      let S := (100 / 2) * (a₁ + a₁₀₀) in
      S = 50)) :=
sorry

end arithmetic_sequence_sum_l690_690325


namespace integral_equals_pi_l690_690378

noncomputable def integral_of_quarter_circle_area : ℝ :=
  ∫ x in (0 : ℝ)..2, real.sqrt (4 - x^2)

theorem integral_equals_pi :
  integral_of_quarter_circle_area = real.pi :=
sorry

end integral_equals_pi_l690_690378


namespace problem_equivalent_l690_690961

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690961


namespace problem_equivalent_l690_690958

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690958


namespace geometric_sequence_problem_l690_690443

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, a n = a 0 * (1 / 2) ^ n

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = (a 0 * (1 - (1 / 2 : ℝ) ^ n)) / (1 - (1 / 2)))
  (h3 : a 0 + a 2 = 5 / 2)
  (h4 : a 1 + a 3 = 5 / 4) :
  ∀ n, S n / a n = 2 ^ n - 1 :=
by
  sorry

end geometric_sequence_problem_l690_690443


namespace two_digit_prime_sum_digits_10_count_l690_690939

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690939


namespace complex_transformation_l690_690302

theorem complex_transformation :
  let z := (-3 : ℂ) - 8 * complex.I
  let rotation := complex.I
  let dilation := 2
  (z * (rotation * dilation)) = 16 - 6 * complex.I :=
by
  simp
  sorry

end complex_transformation_l690_690302


namespace two_digit_primes_with_digit_sum_ten_l690_690798

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690798


namespace logs_in_stack_l690_690362

theorem logs_in_stack : 
  let a₁ := 12 in 
  let d := -1 in 
  let aₙ := 3 in 
  let n := (a₁ - aₙ) / d + 1 in 
  n = 10 ∧
  (n * (a₁ + aₙ)) / 2 = 75 :=
by 
  let a₁ := 12
  let d := -1
  let aₙ := 3 
  let n := (a₁ - aₙ) / d + 1
  have h1 : n = 10 := sorry
  have h2 : (n * (a₁ + aₙ)) / 2 = 75 := sorry
  exact ⟨h1, h2⟩

end logs_in_stack_l690_690362


namespace trainers_hours_split_equally_l690_690617

noncomputable def dolphins := 12
noncomputable def hours_per_dolphin := 5
noncomputable def trainers := 4

theorem trainers_hours_split_equally :
  (dolphins * hours_per_dolphin) / trainers = 15 :=
by
  sorry

end trainers_hours_split_equally_l690_690617


namespace cross_product_correct_l690_690061

def vector1 : ℝ × ℝ × ℝ := (3, 4, -7)
def vector2 : ℝ × ℝ × ℝ := (2, -1, 4)

theorem cross_product_correct : 
  cross_product vector1 vector2 = (9, -26, -11) :=
sorry

end cross_product_correct_l690_690061


namespace count_3_primable_lt_1000_l690_690356

def is_prime_digit (d : ℕ) : Prop := d ∈ [2, 3, 5, 7]

def is_3_primable (n : ℕ) : Prop :=
  n % 3 = 0 ∧ (∀ d in n.digits 10, is_prime_digit d)

theorem count_3_primable_lt_1000 : 
  (Finset.filter is_3_primable (Finset.range 1000)).card = 28 :=
by
  sorry

end count_3_primable_lt_1000_l690_690356


namespace count_prime_two_digit_sum_ten_is_three_l690_690811

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690811


namespace two_digit_primes_with_digit_sum_10_count_l690_690903

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690903


namespace lambda_range_l690_690085

open Real

def vector_dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def angle_between_obtuse (a b : ℝ × ℝ) : Prop := vector_dot a b < 0

theorem lambda_range (λ : ℝ) (h : angle_between_obtuse (-2, -1) (λ, 1)) : λ > -1 / 2 ∧ λ ≠ 2 :=
by
  unfold angle_between_obtuse at h
  simp [vector_dot] at h
  sorry

end lambda_range_l690_690085


namespace geometric_sequence_trig_angles_l690_690746

theorem geometric_sequence_trig_angles :
  ∃! count : ℕ, count = 4 ∧ 
  ∀ θ : ℝ, (0 ≤ θ) ∧ (θ < 2*Real.pi) ∧ (∃ k : ℤ, θ ≠ k * Real.pi / 2) → 
  (∃! order : List ℝ, (order ~ [Real.sin θ, Real.cos θ, Real.tan θ]) ∧ 
  ((order.nthLe 0 _ * order.nthLe 1 _ = order.nthLe 2 _ ^ 2) ∨ 
  (order.nthLe 0 _ * order.nthLe 2 _ = order.nthLe 1 _ ^ 2) ∨ 
  (order.nthLe 1 _ * order.nthLe 2 _ = order.nthLe 0 _ ^ 2))) :=
sorry

end geometric_sequence_trig_angles_l690_690746


namespace points_collinear_find_lambda_l690_690130

-- Definition of planar vectors e1 and e2 with conditions
variables (e1 e2 : EuclideanSpace ℝ (Fin 2))
variables (h1 : ∥e1∥ = 1) (h2 : ∥e2∥ = 1)
variables (angle_e1_e2 : real.angle.e1 e2 = real.angle.pi / 3)

-- Problem 1: Proving points A, B, and D are collinear
theorem points_collinear
  (AB : EuclideanSpace ℝ (Fin 2) := e1 - e2)
  (BC : EuclideanSpace ℝ (Fin 2) := 2 • e1 - 6 • e2)
  (CD : EuclideanSpace ℝ (Fin 2) := 3 • e1 + e2) :
  are_collinear [AB, BC, CD] :=
sorry

-- Problem 2: Finding the value of λ
theorem find_lambda (a : EuclideanSpace ℝ (Fin 2) := e1 + 2 * λ • e2)
  (b : EuclideanSpace ℝ (Fin 2) := λ • e1 - e2)
  (perp_cond : a ⬝ b = 0) :
  λ = (1 + sqrt 2) / 2 ∨ λ = (1 - sqrt 2) / 2 :=
sorry

end points_collinear_find_lambda_l690_690130


namespace chess_tournament_rankings_l690_690583

theorem chess_tournament_rankings :
  let playerA := 0
  let playerB := 1
  let playerC := 2
  let playerD := 3
  let matches := [(playerA, playerB), (playerC, playerD)]
  let winners (match : (Nat × Nat)) := [match.1, match.2]
  let losers (match : (Nat × Nat)) := [match.1, match.2] 
  let finals (w : Nat × Nat) := [w.1, w.2]
  let finals_losers (l : Nat × Nat) := [l.1, l.2]
  (∃ results : List (Nat × Nat),
    results.length = 3 ∧
    (∀ i < 3, results[i] ∈ matches) ∧
    (∀ (w l : Nat) (i < 3), w ∈ winners results[i] ∧ l ∈ losers results[i])) →
  ∃ n : Nat, n = 8 :=
by
  let playerA := 0
  let playerB := 1
  let playerC := 2
  let playerD := 3
  let matches := [(playerA, playerB), (playerC, playerD)]
  let winners (match : (Nat × Nat)) := [match.1, match.2]
  let losers (match : (Nat × Nat)) := [match.1, match.2]
  let finals (w : (Nat × Nat)) := [w.1, w.2]
  let finals_losers (l : (Nat × Nat)) := [l.1, l.2]
  exists 8
  sorry

end chess_tournament_rankings_l690_690583


namespace correct_graph_l690_690381

-- Defining speeds and distances
def speed_P := v
def speed_Q := 2 * v
def speed_R := 3 * v

def distance (speed : ℝ) (time : ℝ) := speed * time

-- Constants:
constant v : ℝ
constant t : ℝ
axiom dist_eq : distance speed_P t = distance speed_Q (t / 2)
axiom dist_eq2 : distance speed_P t = distance speed_R (t / 3)

-- Prove that the graph describing the travel times and speeds correctly is the option B
theorem correct_graph : 
  (distance speed_Q (t / 2) = distance speed_P t) ∧
  (distance speed_R (t / 3) = distance speed_P t) :=
by 
  sorry

end correct_graph_l690_690381


namespace remainder_calculation_l690_690659

theorem remainder_calculation : 
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 :=
by
  intros dividend divisor quotient remainder hdividend hdivisor hquotient heq
  sorry

end remainder_calculation_l690_690659


namespace cab_ride_total_cost_l690_690376

theorem cab_ride_total_cost :
  let cost_per_mile := 2.5 in
  let distance_one_way := 200 in
  let days := 7 in
  let total_cost := 2 * distance_one_way * cost_per_mile * days in
  total_cost = 7000 :=
by
  sorry

end cab_ride_total_cost_l690_690376


namespace oil_purchase_l690_690753

theorem oil_purchase (reduced_price percent_reduction total_money : ℝ)
  (h1 : reduced_price = 55)
  (h2 : percent_reduction = 0.25)
  (h3 : total_money = 1100) :
  let P := reduced_price / (1 - percent_reduction) in
  let amount1 := total_money / P in
  let amount2 := total_money / reduced_price in
  let more_oil := amount2 - amount1 in
  more_oil = 5 := 
by
  sorry

end oil_purchase_l690_690753


namespace total_pieces_is_100_l690_690278

-- Definitions based on conditions
def total_pieces_of_bread (B : ℕ) : Prop :=
  let duck1 := B / 2 in      -- The first duck eats half of all the pieces of bread
  let duck2 := 13 in         -- The second duck eats 13 pieces
  let duck3 := 7 in          -- The third duck eats 7 pieces
  let left_in_water := 30 in -- There are 30 pieces left in the water
  duck1 + duck2 + duck3 + left_in_water = B

-- The statement to be proved
theorem total_pieces_is_100 (B : ℕ) : total_pieces_of_bread B → B = 100 :=
by
  -- Proof would be provided here
  sorry

end total_pieces_is_100_l690_690278


namespace inequality_P_ge_binom_squared_equality_condition_P_eq_binom_squared_l690_690428

noncomputable def P (k : ℕ) : ℕ :=
  number of all positive 4k-digit numbers composed of the digits 2 and 0 that are divisible by 2020 and do not start with 0

theorem inequality_P_ge_binom_squared (k : ℕ) (hk : 0 < k) :
  P k ≥ Nat.choose (2*k - 1) k ^ 2 := sorry

theorem equality_condition_P_eq_binom_squared (k : ℕ) (hk : 0 < k) :
  P k = Nat.choose (2*k - 1) k ^ 2 ↔  k ≤ 9 := sorry

end inequality_P_ge_binom_squared_equality_condition_P_eq_binom_squared_l690_690428


namespace correct_average_weight_is_58_6_l690_690608

noncomputable def initial_avg_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def incorrect_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 60
noncomputable def correct_avg_weight := (initial_avg_weight * num_boys + (correct_weight - incorrect_weight)) / num_boys

theorem correct_average_weight_is_58_6 :
  correct_avg_weight = 58.6 :=
sorry

end correct_average_weight_is_58_6_l690_690608


namespace problem_equivalent_l690_690965

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690965


namespace num_two_digit_primes_with_digit_sum_10_l690_690788

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690788


namespace two_digit_prime_sum_to_ten_count_l690_690840

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690840


namespace rhombus_area_is_correct_l690_690763

def calculate_rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  calculate_rhombus_area (3 * 6) (3 * 4) = 108 := by
  sorry

end rhombus_area_is_correct_l690_690763


namespace count_two_digit_prime_with_digit_sum_10_l690_690874

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690874


namespace exists_x0_l690_690474

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3 * x))^2 - 2 * a * (x + 3 * Real.log (3 * x)) + 10 * a^2

theorem exists_x0 (a : ℝ) (h : a = 1 / 30) : ∃ x0 : ℝ, f x0 a ≤ 1 / 10 := 
by
  sorry

end exists_x0_l690_690474


namespace dot_product_property_l690_690449

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

variables (x_P y_P : ℝ) (F1 F2 : ℝ × ℝ)

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0) ∨ F = (-1, 0)

def radius_of_inscribed_circle (r : ℝ) : Prop :=
  r = 1 / 2

theorem dot_product_property (h1 : point_on_ellipse x_P y_P)
  (h2 : is_focus F1) (h3 : is_focus F2) (h4: radius_of_inscribed_circle (1/2)):
  (x_P^2 - 1 + y_P^2) = 9 / 4 :=
sorry

end dot_product_property_l690_690449


namespace petya_recover_x_y_l690_690303

theorem petya_recover_x_y (x y a b c d : ℝ)
    (hx_pos : x > 0) (hy_pos : y > 0)
    (ha : a = x + y) (hb : b = x - y) (hc : c = x / y) (hd : d = x * y) :
    ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ a = x' + y' ∧ b = x' - y' ∧ c = x' / y' ∧ d = x' * y' :=
sorry

end petya_recover_x_y_l690_690303


namespace two_digit_prime_sum_digits_10_count_l690_690930

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690930


namespace distance_between_vertices_of_hyperbola_l690_690420

theorem distance_between_vertices_of_hyperbola :
  (let a := Real.sqrt 16 in 2 * a) = 8 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l690_690420


namespace find_alpha_l690_690145

-- Define the necessary trigonometric values
def cos_10 := Real.cos (10 * Real.pi / 180)
def sin_10 := Real.sin (10 * Real.pi / 180)
def sqrt3_tan_10 := Real.sqrt 3 * (sin_10 / cos_10)

-- State that given condition
theorem find_alpha (α : ℝ) : 
  Real.cos α * (1 + sqrt3_tan_10) = 1 → (α = 40 * Real.pi / 180 ∨ α = 320 * Real.pi / 180) :=
  by
  sorry

end find_alpha_l690_690145


namespace find_k_l690_690054

open Nat Int

-- Given conditions
axiom k_exists (k : ℤ) (f : ℕ → ℤ) : 
  (f 1995 = 1996) ∧ 
  (∀ x y : ℕ, f (x * y) = f x + f y + k * f (gcd x y)) 

theorem find_k : 
  ∀ k : ℤ, ∃ f : ℕ → ℤ, (f 1995 = 1996) ∧ 
           (∀ x y : ℕ, f (x * y) = f x + f y + k * f (gcd x y)) → 
           (k = -1 ∨ k = 0) :=
by
  assume k,
  assume h : ∃ f : ℕ → ℤ, (f 1995 = 1996) ∧ 
                           (∀ x y : ℕ, f (x * y) = f x + f y + k * f (gcd x y)),
  sorry

end find_k_l690_690054


namespace smurfs_cannot_return_home_l690_690605

theorem smurfs_cannot_return_home:
  ∀ (n : ℕ) (plane : Type)
  (travel : plane → plane → Prop),
  (∀ x y, plane travel x y → travel y.x y.y → (y.x = north)) →
  (∀ path : list plane, path.head = home → 
    (∀ step in path, change_direction step ∧ step ≠ head ∧ visited_once (step)) →
    (∀ segment in path, segment ≠ repeat visited positions) → 
    ∑ angles ∈ path, angle = (path.length / 2)  * 360° → 
    False)

end smurfs_cannot_return_home_l690_690605


namespace perpendicular_medians_of_triangle_l690_690321

theorem perpendicular_medians_of_triangle (A B C : ℝ × ℝ) (O : ℝ × ℝ)
    (h_med_A : O = (A + (B+C)/2)/3)  -- O is centroid, median from A
    (h_med_B : O = (B + (A+C)/2)/3)  -- O is centroid, median from B
    (h_med_perpendicular : ∀ D E : ℝ × ℝ, D ≠ E →
        (O = (A+D)/2 ∧ O = (B+E)/2 → (O = (0, 0) ∧ (D - O = O - B) ∨ (E - O = O - A))) -- two medians are perpendicular.
    :
    A = (0, 6) ∧ B = (6, 0) ∧ C = (6, 6) :=
sorry

end perpendicular_medians_of_triangle_l690_690321


namespace direction_vector_correct_l690_690704

structure Vector := (x : Int) (y : Int)

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/5, 4/5],![4/5, -3/5]]

def vec := Vector.mk 2 1

theorem direction_vector_correct :
  (A.mulVec ![vec.x, vec.y] = ![vec.x, vec.y]) ∧
  (vec.x > 0) ∧
  (Int.gcd (Int.natAbs vec.x) (Int.natAbs vec.y) = 1) :=
by
  sorry

end direction_vector_correct_l690_690704


namespace probability_at_least_two_red_balls_l690_690333

theorem probability_at_least_two_red_balls:
  let total_balls := 18 in
  let red_balls := 6 in
  let blue_balls := 5 in
  let green_balls := 4 in
  let yellow_balls := 3 in
  let total_ways := Nat.choose total_balls 4 in
  let ways_2_red := Nat.choose red_balls 2 * Nat.choose (total_balls - red_balls) 2 in
  let ways_3_red := Nat.choose red_balls 3 * Nat.choose (total_balls - red_balls) 1 in
  let ways_4_red := Nat.choose red_balls 4 in
  let probability_at_least_2_red := (ways_2_red + ways_3_red + ways_4_red) * 1 / total_ways in
  probability_at_least_2_red = 83 / 204 := 
  by sorry

end probability_at_least_two_red_balls_l690_690333


namespace committee_count_is_972_l690_690374

-- Definitions of the problem conditions
def departments := ["Data Science", "Software Engineering", "Cybersecurity"]
def professors_per_department := 6
def males_per_department := 3
def females_per_department := 3
def total_committee_members := 9
def male_committee_members := 4
def female_committee_members := 5
def members_per_department := 3

-- Statement of the theorem
theorem committee_count_is_972
  (h1 : ∀ d ∈ departments, males_per_department = 3)
  (h2 : ∀ d ∈ departments, females_per_department = 3)
  (h3 : ∃ committee, set.size committee = total_committee_members 
                     ∧ ∀ prof ∈ committee, prof.department ∈ departments 
                     ∧ ∑ (prof ∈ committee), prof.gender = "male" = male_committee_members
                     ∧ ∑ (prof ∈ committee), prof.gender = "female" = female_committee_members
                     ∧ ∀ d ∈ departments, set.size {prof ∈ committee | prof.department = d} = members_per_department) :
  number_of_possible_committees = 972 :=
sorry

end committee_count_is_972_l690_690374


namespace find_number_l690_690319

theorem find_number (n : ℤ) (h : 7 * n - 15 = 2 * n + 10) : n = 5 :=
sorry

end find_number_l690_690319


namespace maximum_value_n_l690_690569

def maximum_subset_size (S : Finset (Fin 50)) : Prop :=
  ∀ (x y : Fin 50), x ∈ S → y ∈ S → x ≠ y → (x + y) % 7 ≠ 0

theorem maximum_value_n : ∃ (S : Finset (Fin 50)), maximum_subset_size S ∧ S.card = 23 := 
begin
  sorry
end

end maximum_value_n_l690_690569


namespace polar_bear_ratio_l690_690016

theorem polar_bear_ratio (penguins : ℕ) (total_animals : ℕ) (h_penguins : penguins = 21) (h_total : total_animals = 63) :
  let polar_bears := total_animals - penguins in polar_bears / penguins = 2 :=
by
  sorry

end polar_bear_ratio_l690_690016


namespace csc_squared_sum_geq_12_l690_690329

theorem csc_squared_sum_geq_12 (α β γ : ℝ) (h : α + β + γ = π) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) :
  Real.csc (α / 2) ^ 2 + Real.csc (β / 2) ^ 2 + Real.csc (γ / 2) ^ 2 ≥ 12 :=
sorry

end csc_squared_sum_geq_12_l690_690329


namespace upper_limit_is_58_l690_690552

def KarthikWeightProblem (weight : ℝ) : Prop :=
  (50 < weight ∧ weight < upper_limit) ∧ 
  (55 < weight ∧ weight < 62) ∧ 
  (weight ≤ 58) ∧ 
  (average_weight = 56.5)

theorem upper_limit_is_58 (upper_limit : ℝ) (average_weight : ℝ) (weight : ℝ) 
  (h1 : 50 < weight) 
  (h2 : weight < upper_limit)
  (h3 : 55 < weight) 
  (h4 : weight < 62) 
  (h5 : weight ≤ 58) 
  (h6 : average_weight = (55 + upper_limit) / 2)
  (h7 : average_weight = 56.5) : 
  upper_limit = 58 :=
sorry

end upper_limit_is_58_l690_690552


namespace dr_jones_remaining_salary_l690_690404

theorem dr_jones_remaining_salary:
  let salary := 6000
  let house_rental := 640
  let food_expense := 380
  let electric_water_bill := (1/4) * salary
  let insurances := (1/5) * salary
  let taxes := (10/100) * salary
  let transportation := (3/100) * salary
  let emergency_costs := (2/100) * salary
  let total_expenses := house_rental + food_expense + electric_water_bill + insurances + taxes + transportation + emergency_costs
  let remaining_salary := salary - total_expenses
  remaining_salary = 1380 :=
by
  sorry

end dr_jones_remaining_salary_l690_690404


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690944

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690944


namespace solve_for_y_l690_690050

theorem solve_for_y (y : ℝ) (h : log 16 (3 * y - 4) = 2) : y = 260 / 3 :=
sorry

end solve_for_y_l690_690050


namespace value_of_number_l690_690153

theorem value_of_number (number y : ℝ) 
  (h1 : (number + 5) * (y - 5) = 0) 
  (h2 : ∀ n m : ℝ, (n + 5) * (m - 5) = 0 → n^2 + m^2 ≥ 25) 
  (h3 : number^2 + y^2 = 25) : number = -5 :=
sorry

end value_of_number_l690_690153


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690918

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690918


namespace sum_of_digits_of_N_l690_690364

theorem sum_of_digits_of_N :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ N.digits 10 = [7, 7] :=
by
  sorry

end sum_of_digits_of_N_l690_690364


namespace eval_expression_l690_690047

theorem eval_expression :
  let x := 2
  let y := -3
  let z := 1
  x^2 + y^2 - z^2 + 2 * x * y + 3 * z = 0 := by
sorry

end eval_expression_l690_690047


namespace coefficient_of_x_4_in_expansion_l690_690543

theorem coefficient_of_x_4_in_expansion :
  let expr := (x - (2 / x)) ^ 6
  have h1 : expr = (x - 2 / x) ^ 6,
  -- expansion of (x - (2 / x))^6
  -- finding the coefficient of x^4
  (-2) * 6 = -12 :=
sorry

end coefficient_of_x_4_in_expansion_l690_690543


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690946

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690946


namespace eval_expression_correct_l690_690661

theorem eval_expression_correct : ((3^1 - 2 + 7^2 - 3 : ℝ) ^ -2 * 2) = (2 / 2209) :=
by
  -- Proof will go here
  sorry

end eval_expression_correct_l690_690661


namespace function_with_period_pi_l690_690019

-- Definitions of the given functions
def f1 (x : ℝ) : ℝ := sin (x / 2)
def f2 (x : ℝ) : ℝ := tan (2 * x)
def f3 (x : ℝ) : ℝ := abs (sin (2 * x))
def f4 (x : ℝ) : ℝ := 5 + cos (2 * x)

-- The problem statement in Lean
theorem function_with_period_pi : ∃ f : ℝ → ℝ, (∀ x, f x = 5 + cos (2 * x)) ∧ (∀ x, f (x + π) = f x) := by
  use f4
  split
  { intros x,
    refl, }
  { sorry }

end function_with_period_pi_l690_690019


namespace problem1_l690_690403

noncomputable def a (i : ℕ) (x : ℝ) : ℝ := ∑ n, x^(3 * n + i) / (nat.factorial (3 * n + i))

theorem problem1 (x : ℝ) :
  (a 0 x)^3 + (a 1 x)^3 + (a 2 x)^3 - 3 * (a 0 x) * (a 1 x) * (a 2 x) = 1 :=
sorry

end problem1_l690_690403


namespace system_of_equations_l690_690129

theorem system_of_equations (x y k : ℝ) 
  (h1 : x + 2 * y = k + 2) 
  (h2 : 2 * x - 3 * y = 3 * k - 1) : 
  x + 9 * y = 7 :=
  sorry

end system_of_equations_l690_690129


namespace cat_weight_problem_l690_690372

variable (female_cat_weight male_cat_weight : ℕ)

theorem cat_weight_problem
  (h1 : male_cat_weight = 2 * female_cat_weight)
  (h2 : female_cat_weight + male_cat_weight = 6) :
  female_cat_weight = 2 :=
by
  sorry

end cat_weight_problem_l690_690372


namespace max_tan_angle_BAC_unbounded_l690_690291

theorem max_tan_angle_BAC_unbounded (A B C D : Type) 
  [T : triangle A B C] 
  (h_angle_C : ∠ACB = 45) 
  (h_BC : BC = 6) 
  (h_D_mid : midpoint D B C) : 
  ∀ x : ℝ, ∃ A', ∃ B', ∃ C', ∠BAC → A' B' C' ∙ tan (A' B' C') > x :=
by sorry

end max_tan_angle_BAC_unbounded_l690_690291


namespace harry_travel_time_l690_690493

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l690_690493


namespace find_polynomial_with_root_l690_690415

noncomputable def is_root (p : Polynomial ℚ) (α : ℝ) : Prop := Polynomial.aeval α p = 0 

/-- A monic polynomial of degree 4 with rational coefficients such that √5 + √7 is a root. -/
theorem find_polynomial_with_root :
  Polynomial.monic (Polynomial.X^4 - 24 * Polynomial.X^2 + 4) ∧
  Polynomial.degree (Polynomial.X^4 - 24 * Polynomial.X^2 + 4) = 4 ∧
  ∀ r : ℝ, is_root (Polynomial.X^4 - 24 * Polynomial.X^2 + 4) r → r = √5 + √7 ∨ r = √5 - √7 :=
begin
  sorry,
end

end find_polynomial_with_root_l690_690415


namespace hash_hash_hash_72_eq_12_5_l690_690741

def hash (N : ℝ) : ℝ := 0.5 * N + 2

theorem hash_hash_hash_72_eq_12_5 : hash (hash (hash 72)) = 12.5 := 
by
  sorry

end hash_hash_hash_72_eq_12_5_l690_690741


namespace correct_age_equation_l690_690288

variable (x : ℕ)

def age_older_brother (x : ℕ) : ℕ := 2 * x
def age_younger_brother_six_years_ago (x : ℕ) : ℕ := x - 6
def age_older_brother_six_years_ago (x : ℕ) : ℕ := 2 * x - 6

theorem correct_age_equation (h1 : age_younger_brother_six_years_ago x + age_older_brother_six_years_ago x = 15) :
  (x - 6) + (2 * x - 6) = 15 :=
by
  sorry

end correct_age_equation_l690_690288


namespace projection_result_l690_690357

-- Define the projection function
def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2 in
  let scalar_proj := dot_product u v / dot_product v v in
  (scalar_proj * v.1, scalar_proj * v.2)

-- Proof statement that the projection of vector (3, -1) onto the vector (1, -1) equals to (2, -2)
theorem projection_result : projection (3, -1) (1, -1) = (2, -2) :=
by sorry

end projection_result_l690_690357


namespace two_digit_prime_sum_digits_10_count_l690_690935

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690935


namespace count_prime_two_digit_sum_ten_is_three_l690_690819

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690819


namespace car_distance_on_dirt_road_l690_690728

theorem car_distance_on_dirt_road
  (initial_distance: ℝ)
  (speed_highway: ℝ)
  (speed_dirt: ℝ)
  (speed_muddy: ℝ) :

  (initial_distance = 400) →
  (speed_highway = 160) →
  (speed_dirt = 60) →
  (speed_muddy = 20) →
  let time_to_dirt_boundary := 400 / (160 * 1000) in -- 0.0025 hours
  let distance_red_car_on_dirt := time_to_dirt_boundary * (60 * 1000) in -- 0.15 km => 150 meters
  let distance_between_cars_on_dirt := distance_red_car_on_dirt in
  let distance_between_cars_on_muddy := (distance_red_car_on_dirt / (60 * 1000)) * (20 * 1000) in -- 50 meters

  distance_between_cars_on_dirt = 150 ∧ 
  distance_between_cars_on_muddy = 50 := by
  sorry

end car_distance_on_dirt_road_l690_690728


namespace print_shop_x_charge_l690_690427

theorem print_shop_x_charge :
  ∃ (x : ℝ), 60 * x + 90 = 60 * 2.75 ∧ x = 1.25 :=
by
  sorry

end print_shop_x_charge_l690_690427


namespace exists_large_subset_l690_690556

variables (I : Finset ℕ) (n : ℕ)
variables (A : Finset (Finset ℕ)) (hA : ∀ a b ∈ A, a ≠ b → (a ∩ b).card ≤ 1)

theorem exists_large_subset (hI : I = (Finset.range (n + 1)).filter (λ x, x > 0)) :
  ∃ (X : Finset ℕ), (∀ a ∈ A, ¬ a ⊆ X) ∧ X.card ≥ nat.floor (real.sqrt (2 * n)) :=
by
  sorry

end exists_large_subset_l690_690556


namespace greatest_rational_root_99x2_100x_1_l690_690650

theorem greatest_rational_root_99x2_100x_1 :
  ∀ a b c : ℕ, a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧ (a = 99 ∧ b = 100 ∧ c = 1) →
  ∃ x : ℚ, is_root (⇑(polynomial.C a) * polynomial.X^2 + ⇑(polynomial.C b) * polynomial.X + ⇑(polynomial.C c)) x ∧ x = -1 / 99 :=
by
  sorry

end greatest_rational_root_99x2_100x_1_l690_690650


namespace yz_sub_zx_sub_xy_l690_690436

theorem yz_sub_zx_sub_xy (x y z : ℝ) (h1 : x - y - z = 19) (h2 : x^2 + y^2 + z^2 ≠ 19) :
  yz - zx - xy = 171 := by
  sorry

end yz_sub_zx_sub_xy_l690_690436


namespace sum_of_costs_of_antipalindromic_sequences_of_length_2020_l690_690007

def is_antipalindromic (s : List Char) : Prop :=
  s.reverse.map (λ c, if c = 'A' then 'B' else 'A') = s

noncomputable def cost_of_sequence (s : List Char) : ℕ :=
  (s.enum.filter (λ (i, c), c = 'A')).map (λ (i, _), i + 1).prod

noncomputable def sum_of_costs_of_antipalindromic_sequences_of_length (n : ℕ) : ℕ :=
  list_sum $ (List.replicate n ['A', 'B']).zip .map (λ s, if is_antipalindromic s then cost_of_sequence s else 0)

theorem sum_of_costs_of_antipalindromic_sequences_of_length_2020 :
  sum_of_costs_of_antipalindromic_sequences_of_length 2020 = 2021^1010 :=
by sorry

end sum_of_costs_of_antipalindromic_sequences_of_length_2020_l690_690007


namespace max_largest_element_l690_690008

theorem max_largest_element (a b c d e f g : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) (h7 : 0 < g)
  (h_median : (list.sort (<=) [a, b, c, d, e, f, g]).nth 3 = some 5)
  (h_mean : (a + b + c + d + e + f + g) = 98) :
  max a (max b (max c (max d (max e (max f g))))) = 80 :=
sorry

end max_largest_element_l690_690008


namespace frequented_by_beggars_due_to_railway_mother_treated_beggars_equally_mother_provided_newspapers_for_work_mother_l690_690211

-- Define conditions
def cond_1 := "The writer's house was within a mile of the railroad tracks."
def cond_2 := "Beggars frequently knocked at neighborhood doors asking for food, and were often turned away or ignored."
def cond_3 := "The writer's mother always welcomed beggars, invited them to eat, and treated them respectfully."
def cond_4 := "She would set a nice table for the beggars, serve them food and dessert, and often send them off with a packed lunch."
def cond_5 := "She sometimes provided newspapers to those who asked about work."
def cond_6 := "She never asked beggars about their personal situations."
def cond_7 := "Her actions were influenced by the thoughtfulness of people who fed her father during the depression."

-- Proof Problem 1
theorem frequented_by_beggars_due_to_railway :
  cond_1 ∧ cond_2 -> "Beggars frequently visited the writer's house because there was a railway near their house." :=
sorry

-- Proof Problem 2
theorem mother_treated_beggars_equally :
  cond_1 ∧ cond_3 ∧ cond_4 ∧ cond_5 -> "The writer’s mother treated beggars with equal respect." :=
sorry

-- Proof Problem 3
theorem mother_provided_newspapers_for_work :
  cond_5 -> "The writer’s mother provided newspapers to help beggars who asked about work." :=
sorry

-- Proof Problem 4
theorem mother's_attitude_explained_by_fathers_experience :
  cond_7 -> "The mother’s attitude towards beggars can be explained by her father being helped by others." :=
sorry

end frequented_by_beggars_due_to_railway_mother_treated_beggars_equally_mother_provided_newspapers_for_work_mother_l690_690211


namespace count_two_digit_primes_with_digit_sum_10_l690_690862

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690862


namespace pyramid_angles_sum_pi_over_four_l690_690162

theorem pyramid_angles_sum_pi_over_four :
  ∃ (α β : ℝ), 
    α + β = Real.pi / 4 ∧ 
    α = Real.arctan ((Real.sqrt 17 - 3) / 4) ∧ 
    β = Real.pi / 4 - Real.arctan ((Real.sqrt 17 - 3) / 4) :=
by
  sorry

end pyramid_angles_sum_pi_over_four_l690_690162


namespace prize_calculations_l690_690260

-- Definitions for the conditions
def total_prizes := 50
def first_prize_unit_price := 20
def second_prize_unit_price := 14
def third_prize_unit_price := 8
def num_second_prize (x : ℕ) := 3 * x - 2
def num_third_prize (x : ℕ) := total_prizes - x - num_second_prize x
def total_cost (x : ℕ) := first_prize_unit_price * x + second_prize_unit_price * num_second_prize x + third_prize_unit_price * num_third_prize x

-- Proof problem statement
theorem prize_calculations (x : ℕ) (h : num_second_prize x = 22) : 
  num_second_prize x = 3 * x - 2 ∧ 
  num_third_prize x = 52 - 4 * x ∧ 
  total_cost x = 30 * x + 388 ∧ 
  total_cost 8 = 628 :=
by
  sorry

end prize_calculations_l690_690260


namespace raj_earns_more_l690_690591

theorem raj_earns_more :
  let cost_per_sqft := 2
  let raj_length := 30
  let raj_width := 50
  let lena_length := 40
  let lena_width := 35
  let raj_area := raj_length * raj_width
  let lena_area := lena_length * lena_width
  let raj_earnings := raj_area * cost_per_sqft
  let lena_earnings := lena_area * cost_per_sqft
  raj_earnings - lena_earnings = 200 :=
by
  sorry

end raj_earns_more_l690_690591


namespace negation_of_p_l690_690745

-- Definitions for the conditions and propositions
variables {a b c : ℝ}
def p : Prop := (ac > bc) → (a > b)

-- Main theorem stating the equivalence
theorem negation_of_p : ¬p ↔ ((ac ≤ bc) → (a ≤ b)) :=
by
sory

end negation_of_p_l690_690745


namespace count_prime_two_digit_sum_ten_is_three_l690_690818

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690818


namespace business_card_exchanges_l690_690235

-- Definition for individuals and ages
def individuals : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Condition: Each participant only exchanges business cards with people older than themselves
def exchanges (n : ℕ) : ℕ :=
  individuals.filter (λ x => x > n).length

-- Summing the exchanges for all participants
def total_exchanges : ℕ :=
  individuals.foldr (λ x acc => exchanges x + acc) 0

-- Theorem that proves the number of exchanges is 45
theorem business_card_exchanges : total_exchanges = 45 :=
by
  -- Excluding the implementation steps, we immediately provide the correct answer with sorry.
  sorry

end business_card_exchanges_l690_690235


namespace volleyball_team_lineup_count_l690_690366

theorem volleyball_team_lineup_count :
  let num_members := 18 in
  let num_lineup := 8 in
  let num_setter := 1 in
  let num_captain := 1 in
  let num_other := num_lineup - num_setter - num_captain in
  let choose_setter := num_members in
  let choose_captain := num_members - num_setter in
  let choose_others := Nat.choose (num_members - num_setter - num_captain) num_other in
  choose_setter * choose_captain * choose_others = 2448272 :=
by
  sorry

end volleyball_team_lineup_count_l690_690366


namespace arithmetic_sequence_y_solution_l690_690034

theorem arithmetic_sequence_y_solution : 
  ∃ y : ℚ, (y + 2 - - (1 / 3)) = (4 * y - (y + 2)) ∧ y = 13 / 6 :=
by
  sorry

end arithmetic_sequence_y_solution_l690_690034


namespace friday_lending_tuesday_vs_thursday_total_lending_l690_690268

def standard_lending_rate : ℕ := 50
def monday_excess : ℤ := 0
def tuesday_excess : ℤ := 8
def wednesday_excess : ℤ := 6
def thursday_shortfall : ℤ := -3
def friday_shortfall : ℤ := -7

theorem friday_lending : (standard_lending_rate + friday_shortfall) = 43 := by
  sorry

theorem tuesday_vs_thursday : (tuesday_excess - thursday_shortfall) = 11 := by
  sorry

theorem total_lending : 
  (5 * standard_lending_rate + (monday_excess + tuesday_excess + wednesday_excess + thursday_shortfall + friday_shortfall)) = 254 := by
  sorry

end friday_lending_tuesday_vs_thursday_total_lending_l690_690268


namespace fraction_in_between_l690_690193

variable {r u s v : ℤ}

/-- Assumes r, u, s, v be positive integers such that su - rv = 1 --/
theorem fraction_in_between (h1 : r > 0) (h2 : u > 0) (h3 : s > 0) (h4 : v > 0) (h5 : s * u - r * v = 1) :
  ∀ ⦃x num denom : ℤ⦄, r * denom = num * u → s * denom = (num + 1) * v → r * v ≤ num * denom - 1 / u * v * denom
   ∧ num * denom - 1 / u * v * denom ≤ s * v :=
sorry

end fraction_in_between_l690_690193


namespace two_digit_primes_with_digit_sum_10_count_l690_690904

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690904


namespace marbles_in_larger_bottle_l690_690011

theorem marbles_in_larger_bottle 
  (small_bottle_volume : ℕ := 20)
  (small_bottle_marbles : ℕ := 40)
  (larger_bottle_volume : ℕ := 60) :
  (small_bottle_marbles / small_bottle_volume) * larger_bottle_volume = 120 := 
by
  sorry

end marbles_in_larger_bottle_l690_690011


namespace two_digit_primes_with_digit_sum_10_count_l690_690907

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690907


namespace two_digit_primes_with_digit_sum_ten_l690_690799

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690799


namespace group_population_l690_690696

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l690_690696


namespace roses_picked_later_l690_690692

/-- Represents the initial number of roses the florist had. -/
def initial_roses : ℕ := 37

/-- Represents the number of roses the florist sold. -/
def sold_roses : ℕ := 16

/-- Represents the final number of roses the florist ended up with. -/
def final_roses : ℕ := 40

/-- Theorem which states the number of roses picked later is 19 given the conditions. -/
theorem roses_picked_later : (final_roses - (initial_roses - sold_roses)) = 19 :=
by
  -- proof steps are omitted, sorry as a placeholder
  sorry

end roses_picked_later_l690_690692


namespace greatest_n_det_A_nonzero_l690_690200

-- Definition of the matrix A
def matrix_A (n : ℕ) : matrix (fin n) (fin n) ℤ :=
  λ i j, ((i + 1) ^ (j + 1) + (j + 1) ^ (i + 1)) % 3

-- Lean statement of the proof problem
theorem greatest_n_det_A_nonzero :
  ∃ (n : ℕ), (∀ k > n, det (matrix_A k) = 0) ∧ det (matrix_A 5) ≠ 0 :=
by sorry

end greatest_n_det_A_nonzero_l690_690200


namespace count_valid_4_digit_numbers_l690_690497

open Finset

def digits : Finset ℕ := {0, 1, 2, 3}

def no_repeated_digits (n : ℕ) : Prop :=
  (multiset.card (multiset.dedup n.digits) = 4)

def no_adjacent_0_2 (n : ℕ) : Prop :=
  let ds := n.digits in
  ¬ (list.pairwise (λ x y, ¬ ((x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0))) ds)

def is_valid (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧
  (∀ d ∈ n.digits, d ∈ digits) ∧
  no_repeated_digits n ∧
  no_adjacent_0_2 n

theorem count_valid_4_digit_numbers : {n : ℕ | is_valid n}.card = 8 :=
by
  sorry

end count_valid_4_digit_numbers_l690_690497


namespace jimmy_bread_packs_needed_l690_690183

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l690_690183


namespace remainder_r_15_minus_1_l690_690424

theorem remainder_r_15_minus_1 (r : ℝ) : (eval (-1) (X^15 - 1) = -2) :=
by {
  -- proof steps
  sorry
}

end remainder_r_15_minus_1_l690_690424


namespace count_prime_two_digit_sum_ten_is_three_l690_690817

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690817


namespace range_of_a_l690_690478

open Set

theorem range_of_a (a : ℝ) :
  (M : Set ℝ) = { x | -1 ≤ x ∧ x ≤ 2 } →
  (N : Set ℝ) = { x | 1 - 3 * a < x ∧ x ≤ 2 * a } →
  M ∩ N = M →
  1 ≤ a :=
by
  intro hM hN h_inter
  sorry

end range_of_a_l690_690478


namespace two_digit_primes_with_digit_sum_10_count_l690_690899

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690899


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690971

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690971


namespace inequality_implies_strict_inequality_l690_690590

theorem inequality_implies_strict_inequality (x y z : ℝ) (h : x^2 + x * y + x * z < 0) : y^2 > 4 * x * z :=
sorry

end inequality_implies_strict_inequality_l690_690590


namespace correct_propositions_true_l690_690282

-- Definitions for the propositions
def prop1 :=
  ∀ x, y = cos (x - π / 4) * cos (x + π / 4) → dist_adj_symm_centers y x = π

def prop2 :=
  ¬(∀ a b, a = 0 → ab = 0) ↔ ∀ a b, a = 0 → ab ≠ 0

def prop3 :=
  ∀ a b, (a ≠ 5 ∧ b ≠ -5) → a + b ≠ 0

def prop4 :=
  (∀ x ∈ ℝ, sin x ≤ 1) → (∃ x₀ ∈ ℝ, sin x₀ > 1)

def prop5 :=
  ∀ a, (0 < a ∧ a < 1) → log a (a + 1) > log a (1 + 1 / a)

def prop6 :=
  ∀ (a b : ℝ), ‖a - b‖ ≤ ‖a + b‖

def prop7 :=
  ∀ (a b : ℝ), (a ⬝ b = 0) → a ⊥ b

-- Main theorem
theorem correct_propositions_true :
  prop3 ∧ prop4 ∧ prop5 ∧ prop7 :=
by { 
  split, -- for prop3
  sorry, -- complete proof for prop3

  split, -- for prop4
  sorry, -- complete proof for prop4
  
  split, -- for prop5
  sorry, -- complete proof for prop5

  sorry  -- complete proof for prop7
}

end correct_propositions_true_l690_690282


namespace num_two_digit_primes_with_digit_sum_10_l690_690830

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690830


namespace convex_heptagon_four_equal_angles_l690_690690

theorem convex_heptagon_four_equal_angles
  (heptagon : Type)
  (angles : heptagon → ℝ)
  (h_convex : (∀ θ ∈ set.range angles, 0 < θ ∧ θ < π))
  (h_sum_independent : ∀ (sines cosines : finset heptagon), 
                       sines.card = 4 → cosines.card = 3 → 
                       sines.disjoint cosines → sines ∪ cosines = set.univ.to_finset → 
                       set.sum (fun θ => sin (angles θ)) sines + set.sum (fun θ => cos (angles θ)) cosines =
                       set.sum (fun θ => cos (angles θ)) sines + set.sum (fun θ => sin (angles θ)) cosines) :
  ∃ θ, ∃ θ', θ ≠ θ' ∧ θ ≠ θ'' ∧ θ'' ≠ θ' ∧ angles θ = angles θ'' :=
begin
  sorry
end

end convex_heptagon_four_equal_angles_l690_690690


namespace trig_product_identity_l690_690023

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end trig_product_identity_l690_690023


namespace combined_area_l690_690712

noncomputable def diagonal : ℝ := 12 * Real.sqrt 2

noncomputable def side_of_square (d : ℝ) : ℝ := d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

theorem combined_area (d : ℝ) (h : d = diagonal) :
  let s := side_of_square d
  let area_sq := area_of_square s
  let r := radius_of_circle d
  let area_circ := area_of_circle r
  area_sq + area_circ = 144 + 72 * Real.pi :=
by
  sorry

end combined_area_l690_690712


namespace num_two_digit_primes_with_digit_sum_10_l690_690822

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690822


namespace prism_volume_proof_l690_690710

structure Prism :=
  (A B C A₁ B₁ C₁ : Point)
  (regular : is_regular_prism A B C A₁ B₁ C₁)
  (inscribed_in_sphere : inscribed_in_sphere (Prism A B C A₁ B₁ C₁) 6)
  (D C : Point)
  (CD_diameter : is_diameter C D 6)
  (AD_distance: dist A D = 4 * sqrt 6)

noncomputable def volume_of_prism (p : Prism) : ℝ :=
  prism_volume p.A p.B p.C p.A₁ p.B₁ p.C₁

theorem prism_volume_proof (p : Prism) (h1 : is_regular_prism p.A p.B p.C p.A₁ p.B₁ p.C₁)
                          (h2 : inscribed_in_sphere p 6)
                          (h3 : is_diameter p.C p.D 6)
                          (h4 : dist p.A p.D = 4 * sqrt 6) : 
  volume_of_prism p = 48 * sqrt 15 :=
sorry

end prism_volume_proof_l690_690710


namespace num_two_digit_primes_with_digit_sum_10_l690_690786

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690786


namespace GreatPyramidProperties_l690_690238

noncomputable def GreatPyramidBaseLength (h : ℝ) (θ : ℝ) : ℝ :=
  2 * (h / Real.tan θ)

noncomputable def GreatPyramidSum (h : ℝ) (θ : ℝ) : ℝ :=
  h + GreatPyramidBaseLength h θ

theorem GreatPyramidProperties :
  let θ := Real.pi * 52 / 180 in
  let h := 520 in
  GreatPyramidBaseLength h θ ≈ 812.56 ∧
  GreatPyramidSum h θ ≈ 1332.56 :=
by
  sorry

end GreatPyramidProperties_l690_690238


namespace count_integer_solutions_xyz_eq_2009_l690_690137

theorem count_integer_solutions_xyz_eq_2009 :
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  ∃ n : ℕ, n = 72 ∧
  ∀ (x y z : ℤ), (x, y, z) ∈ equations2009 ∨ (x, z, y) ∈ equations2009 ∨ (y, x, z) ∈ equations2009 ∨
    (y, z, x) ∈ equations2009 ∨ (z, x, y) ∈ equations2009 ∨ (z, y, x) ∈ equations2009 ∨ 
    ((-x), (-y), z) ∈ equations2009 ∨ ((-x), y, (-z)) ∈ equations2009 ∨ (x, (-y), (-z)) ∈ equations2009 ∨ 
    ((-y), (-z), x) ∈ equations2009 ∨ ((-y), z, (-x)) ∈ equations2009 ∨ (y, (-z), (-x)) ∈ equations2009 ∨ 
    ((-z), (-x), y) ∈ equations2009 ∨ ((-z), x, (-y)) ∈ equations2009 ∨ (z, (-x), (-y)) ∈ equations2009 → 
    x * y * z = 2009 := 
by
  intros
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  existsi (72 : ℕ)
  split
  · refl
  intro x y z H
  cases H;
  { repeat { cases H; 
      { repeat {
          cases H with (eq : xyz = _),
          { 
            rw eq,
            simp,
            sorry
          } }
        } } }

end count_integer_solutions_xyz_eq_2009_l690_690137


namespace range_of_a_l690_690115

open Real

noncomputable def is_increasing (f: ℝ → ℝ) (I: set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f x ≤ f y

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log 2 (x^2 - a * x + 3 * a)

theorem range_of_a (a : ℝ) : is_increasing (f a) (set.Ici 2) → a ∈ set.Ioc (-4 : ℝ) 4 :=
sorry

end range_of_a_l690_690115


namespace cosine_of_eight_times_alpha_l690_690358

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end cosine_of_eight_times_alpha_l690_690358


namespace least_value_of_k_l690_690361

def sequence (n : ℕ) : Type := vector (fin 3) n

def operation (n : ℕ) (a b : sequence n) : sequence n :=
⟨λ i, fin.of_nat (⌊ (a.1 i).val + (b.1 i).val + 1 / 2 ⌋) % 3, sorry⟩  -- Use lambda and floor operation

def initial_set (n : ℕ) : set (sequence n) := 
  { seq | 
    seq = ⟨vector.repeat 0 n, sorry⟩ ∨ 
    (∃ i, i < n ∧ seq = ⟨vector.of_fn (λ j, if j = i then 2 else 0), sorry⟩) 
  }

theorem least_value_of_k (n : ℕ) : ∃ (k : ℕ) (S : set (sequence n)),
  k = n + 1 ∧
  (initial_set n ⊆ S) ∧
  ((∀ a b ∈ S, operation n a b ∈ S) ∧
  (∀ seq : sequence n, seq ∈ S)) :=
sorry

end least_value_of_k_l690_690361


namespace smallest_b_not_divisible_by_5_l690_690065

theorem smallest_b_not_divisible_by_5 :
  ∃ b : ℕ, b > 2 ∧ ¬ (5 ∣ (2 * b^3 - 1)) ∧ ∀ b' > 2, ¬ (5 ∣ (2 * (b'^3) - 1)) → b = 6 :=
by
  sorry

end smallest_b_not_divisible_by_5_l690_690065


namespace min_distance_point_l690_690683

def distance (P Q : ℝ × ℝ) : ℝ := (real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))

def sum_of_distances (M : ℝ × ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (λ acc P => acc + distance M P) 0

theorem min_distance_point :
  let A := (1, 2) in
  let B := (1, 5) in
  let C := (3, 6) in
  let D := (7, -1) in
  (2, 4) = argmin (sum_of_distances • [A, B, C, D]) sorry :=
sorry

end min_distance_point_l690_690683


namespace proof_problem_l690_690396

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 7
  else let x := sequence (n - 1)
       in (x^2 + 7*x + 12) / (x + 8)

def question : ℕ → Prop :=
  λ m, sequence m ≤ 6 + 1 / 2^18

theorem proof_problem :
  ∃ m : ℕ, m ≥ 243 ∧ m ≤ 728 ∧ question m :=
sorry

end proof_problem_l690_690396


namespace locus_of_centers_l690_690609

theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (9 - r)^2) →
  12 * a^2 + 169 * b^2 - 36 * a - 1584 = 0 :=
by
  sorry

end locus_of_centers_l690_690609


namespace sum_rational_indices_l690_690383

def sequence (a : ℕ → ℚ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = (a n * real.sqrt (n^2 + n)) / (real.sqrt (n^2 + n + 2 * (a n)^2))

def is_rational (a : ℕ → ℚ) (n: ℕ) : Prop :=
∃ k: ℚ, a(n) = k 

theorem sum_rational_indices :
  ∃ a : ℕ → ℚ, sequence a ∧
  (∑ n in finset.filter (λ n, is_rational (λ n, real.sqrt (n / (3 * n - 2))) n) (finset.range 1000), n) = 131 :=
sorry

end sum_rational_indices_l690_690383


namespace total_people_in_group_l690_690699

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l690_690699


namespace sum_XA_XB_XC_eq_21_l690_690638

-- Define the conditions mentioned in part a
structure Triangle :=
  (A B C : Point)
  (AB AC BC : ℝ)
  (hAB : AB = 12)
  (hAC : AC = 15)
  (hBC : BC = 13)

structure Point := 
  (x y : ℝ)

def D (A B : Point) : Point :=
  { x := (A.x + 2 * B.x) / 3, y := (A.y + 2 * B.y) / 3 }

def E (B C : Point) : Point :=
  { x := (2 * B.x + C.x) / 3, y := (2 * B.y + C.y) / 3 }

def F (A C : Point) : Point :=
  { x := (A.x + C.x) / 2, y := (A.y + C.y) / 2 }

structure TriangleCircumcircle (A B C : Point) :=
  (circumcircle : Circle)
  (has_points : circumcircle.contains A ∧ circumcircle.contains B ∧ circumcircle.contains C)

structure Circle := 
  (center : Point)
  (radius : ℝ)

axiom intersect_circles (circ1 circ2 : Circle) : Point

noncomputable def X (circ1 circ2 : Circle) (E : Point) : Point :=
  if hE : (intersect_circles circ1 circ2) = E then sorry else (intersect_circles circ1 circ2)

def XA (A X : Point) : ℝ :=
  real.sqrt ((X.x - A.x)^2 + (X.y - A.y)^2)

def XB (B X : Point) : ℝ :=
  real.sqrt ((X.x - B.x)^2 + (X.y - B.y)^2)

def XC (C X : Point) : ℝ :=
  real.sqrt ((X.x - C.x)^2 + (X.y - C.y)^2)

-- Definition of the proof problem
theorem sum_XA_XB_XC_eq_21 (A B C D E F X : Point) (triangle : Triangle):
  XA A X + XB B X + XC C X = 21 := sorry

end sum_XA_XB_XC_eq_21_l690_690638


namespace contrapositive_of_neg_and_inverse_l690_690157

theorem contrapositive_of_neg_and_inverse (p r s : Prop) (h1 : r = ¬p) (h2 : s = ¬r) : s = (¬p → false) :=
by
  -- We have that r = ¬p
  have hr : r = ¬p := h1
  -- And we have that s = ¬r
  have hs : s = ¬r := h2
  -- Now we need to show that s is the contrapositive of p, which is ¬p → false
  sorry

end contrapositive_of_neg_and_inverse_l690_690157


namespace somu_age_relation_l690_690606

-- Somu’s present age (S) is 20 years
def somu_present_age : ℕ := 20

-- Somu’s age is one-third of his father’s age (F)
def father_present_age : ℕ := 3 * somu_present_age

-- Proof statement: Y years ago, Somu's age was one-fifth of his father's age
theorem somu_age_relation : ∃ (Y : ℕ), somu_present_age - Y = (1 : ℕ) / 5 * (father_present_age - Y) ∧ Y = 10 :=
by
  have h := "" -- Placeholder for the proof steps
  sorry

end somu_age_relation_l690_690606


namespace math_problem_l690_690432

noncomputable def A (k : ℝ) : ℝ := k - 5
noncomputable def B (k : ℝ) : ℝ := k + 2
noncomputable def C (k : ℝ) : ℝ := k / 2
noncomputable def D (k : ℝ) : ℝ := 2 * k

theorem math_problem (k : ℝ) (h : A k + B k + C k + D k = 100) : 
  (A k) * (B k) * (C k) * (D k) =  (161 * 224 * 103 * 412) / 6561 :=
by
  sorry

end math_problem_l690_690432


namespace two_digit_prime_sum_digits_10_count_l690_690936

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690936


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690942

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690942


namespace evaluated_product_floor_ceil_l690_690045

theorem evaluated_product_floor_ceil:
  ( ∏ (n : ℕ) in (Finset.range 5).map (λ n, (- (n + 1)^2 : ℤ)) ) = -14400 := 
by
  sorry

end evaluated_product_floor_ceil_l690_690045


namespace probability_units_digit_1_l690_690705

noncomputable def set_m : Set ℕ := {11, 12, 13, 16, 17}
noncomputable def set_n : Set ℕ := {i | 2010 ≤ i ∧ i ≤ 2020}

def has_units_digit_1 (m n : ℕ) : Prop := (m ^ n) % 10 = 1

theorem probability_units_digit_1 :
  let total_cases := (set_m.card * set_n.card : ℝ)
  let favorable_cases := (set_m.filter (λ m, (set_n.filter (λ n, has_units_digit_1 m n)).card ≠ 0)).card * 3
  (favorable_cases / total_cases) = (4 / 11) :=
by
  sorry

end probability_units_digit_1_l690_690705


namespace probability_ab_adjacent_l690_690289

theorem probability_ab_adjacent (A B C : Type) [decidable_eq A] 
  (arrangements : finset (finset A)) 
  (total_permutations : arrangements.card = 6) 
  (adjacent_permutations : (arrangements.filter (λ s, s.contains A ∧ s.contains B)).card = 2) :
  (adjacent_permutations.to_real / total_permutations.to_real) = 1 / 3 :=
by
  sorry

end probability_ab_adjacent_l690_690289


namespace f_value_at_neg_11_pi_over_12_g_max_min_value_l690_690119

def f (x : ℝ) : ℝ := ((1 + cos (2 * x))^2 - 2 * cos (2 * x) - 1) / (sin (π / 4 + x) * sin (π / 4 - x))
def g (x : ℝ) : ℝ := (1 / 2) * f(x) + sin (2 * x)

theorem f_value_at_neg_11_pi_over_12 : 
  f (-11 * π / 12) = √3 := 
sorry

theorem g_max_min_value : 
  ∃ x_min x_max, (0 ≤ x_min ∧ x_min < π / 4 ∧ g x_min = 1) ∧ (0 ≤ x_max ∧ x_max < π / 4 ∧ g x_max = √2) := 
sorry

end f_value_at_neg_11_pi_over_12_g_max_min_value_l690_690119


namespace Eddy_travel_time_l690_690406

theorem Eddy_travel_time :
  ∀ (T_F D_F D_E : ℕ) (S_ratio : ℝ),
    T_F = 4 →
    D_F = 360 →
    D_E = 600 →
    S_ratio = 2.2222222222222223 →
    ((D_F / T_F : ℝ) * S_ratio ≠ 0) →
    D_E / ((D_F / T_F) * S_ratio) = 3 :=
by
  intros T_F D_F D_E S_ratio ht hf hd hs hratio
  sorry  -- Proof to be provided

end Eddy_travel_time_l690_690406


namespace count_integer_solutions_xyz_eq_2009_l690_690136

theorem count_integer_solutions_xyz_eq_2009 :
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  ∃ n : ℕ, n = 72 ∧
  ∀ (x y z : ℤ), (x, y, z) ∈ equations2009 ∨ (x, z, y) ∈ equations2009 ∨ (y, x, z) ∈ equations2009 ∨
    (y, z, x) ∈ equations2009 ∨ (z, x, y) ∈ equations2009 ∨ (z, y, x) ∈ equations2009 ∨ 
    ((-x), (-y), z) ∈ equations2009 ∨ ((-x), y, (-z)) ∈ equations2009 ∨ (x, (-y), (-z)) ∈ equations2009 ∨ 
    ((-y), (-z), x) ∈ equations2009 ∨ ((-y), z, (-x)) ∈ equations2009 ∨ (y, (-z), (-x)) ∈ equations2009 ∨ 
    ((-z), (-x), y) ∈ equations2009 ∨ ((-z), x, (-y)) ∈ equations2009 ∨ (z, (-x), (-y)) ∈ equations2009 → 
    x * y * z = 2009 := 
by
  intros
  let equations2009 := [
    (1, 1, 2009),
    (1, 7, 287),
    (7, 7, 41),
    (1, 49, 41)
  ]
  existsi (72 : ℕ)
  split
  · refl
  intro x y z H
  cases H;
  { repeat { cases H; 
      { repeat {
          cases H with (eq : xyz = _),
          { 
            rw eq,
            simp,
            sorry
          } }
        } } }

end count_integer_solutions_xyz_eq_2009_l690_690136


namespace discount_percentage_l690_690009

noncomputable def cost_price := 100
noncomputable def labeled_price := 150
noncomputable def actual_profit := 135

theorem discount_percentage (cp lp sp : ℝ) (h1 : lp = 1.5 * cp) (h2 : sp = 1.35 * cp) :
  ((lp - sp) / lp * 100) = 10 :=
by
  simp [cp, lp, sp, h1, h2]
  sorry

end discount_percentage_l690_690009


namespace reciprocal_of_repeating_decimal_l690_690657

theorem reciprocal_of_repeating_decimal : (∀ x : ℚ, x = 36 / 99 → (1 / (4 / 11) = 11 / 4)) :=
begin
  assume x,
  assume h : x = 36 / 99,
  sorry
end

end reciprocal_of_repeating_decimal_l690_690657


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690951

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690951


namespace find_y_coordinate_l690_690030

noncomputable def y_coordinate (x : ℝ) : ℝ :=
  2 * x^2

def parabola_focus : ℝ × ℝ :=
  (0, 1 / 8)

def distance_to_focus (M : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - 0)^2 + (M.2 - 1 / 8)^2)

noncomputable def point_on_parabola (M : ℝ × ℝ) : Prop :=
  y_coordinate M.1 = M.2

noncomputable def problem_statement : ℝ :=
  { y : ℝ // ∃ (x : ℝ), point_on_parabola (x, y) ∧ distance_to_focus (x, y) = 1 }

theorem find_y_coordinate (M : problem_statement) :
  M.1 = 7 / 8 :=
begin
  sorry
end

end find_y_coordinate_l690_690030


namespace largest_possible_beta_l690_690545

theorem largest_possible_beta 
  (O : point) 
  (P : ℕ → point) 
  (r : ℕ → ℝ) 
  (α : ℝ) 
  (h0 : 0 < α) 
  (h1 : α < 1) 
  (h2 : ∀ n m : ℕ, n ≠ m → dist (P n) (P m) ≥ (r n) ^ α) 
  (h3 : ∀ i j : ℕ, i < j → r i ≤ r j)
  : ∃ (C : ℝ), ∀ n : ℕ, r n ≥ C * n ^ (1 / (2 * (1 - α))) :=
by
  sorry

end largest_possible_beta_l690_690545


namespace find_angle_C_find_a_plus_b_l690_690520

variable (a b c A B C : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

-- Definitions and conditions
def condition_1 := (m = (Real.cos (C / 3), Real.sin (C / 2)))
def condition_2 := (n = (Real.cos (C / 2), -Real.sin (C / 2)))
def condition_3 := (angle_between := π / 3)
def condition_4 := (c = 7 / 2)
def condition_5 := (S = (3 / 2) * Real.sqrt 3)
def condition_6 := (triangle_area := (1 / 2) * a * b * Real.sin C)

-- Theorems to prove
theorem find_angle_C (h1 : condition_1) (h2 : condition_2) (h3 : angle_between = π / 3) :
  C = π / 3 := by sorry

theorem find_a_plus_b (h1 : condition_4) (h2 : condition_5) (h3 : S = (1 / 2) * a * b * Real.sin C)
  (h4 : C = π / 3) :
  a + b = 11 / 2 := by sorry

end find_angle_C_find_a_plus_b_l690_690520


namespace ellipse_focal_distance_l690_690377

theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 9) → (∃ c : ℝ, c = 2 * Real.sqrt 3) :=
by
  sorry

end ellipse_focal_distance_l690_690377


namespace fraction_spent_at_arcade_l690_690550

theorem fraction_spent_at_arcade :
  ∃ f : ℝ, 
    (2.25 - (2.25 * f) - ((2.25 - (2.25 * f)) / 3) = 0.60) → 
    f = 3 / 5 :=
by
  sorry

end fraction_spent_at_arcade_l690_690550


namespace percentage_increase_consumption_l690_690627

theorem percentage_increase_consumption
  (T C : ℝ) 
  (h_tax : ∀ t, t = 0.60 * T)
  (h_revenue : ∀ r, r = 0.75 * T * C) :
  1.25 * C = (0.75 * T * C) / (0.60 * T) := by
sorry

end percentage_increase_consumption_l690_690627


namespace bill_profit_difference_l690_690729

noncomputable def original_purchase_price (SP : ℝ) := SP / 1.10
noncomputable def new_purchase_price (P : ℝ) := 0.90 * P
noncomputable def new_selling_price (NP : ℝ) := 1.30 * NP

theorem bill_profit_difference :
  let SP := 1099.999999999999 in
  let P := original_purchase_price SP in
  let NP := new_purchase_price P in
  let NSP := new_selling_price NP in
  NSP - SP = 70 :=
by sorry

end bill_profit_difference_l690_690729


namespace circumcircle_radius_l690_690573

/-- Let the common foci of the ellipse x^2/10 + y^2 = 1 and the hyperbola x^2/8 - y^2 = 1 be F_1 and F_2, respectively.
If P is a point of intersection of these two curves, then the radius of the circumcircle of triangle PF_1F_2 is 3.
-/
theorem circumcircle_radius (F₁ F₂ P : ℝ × ℝ)
  (hF₁ : F₁ = (sqrt 10, 0) ∨ F₁ = (-sqrt 10, 0))
  (hF₂ : F₂ = (sqrt 8, 0) ∨ F₂ = (-sqrt 8, 0))
  (hP : (P.1^2 / 10 + P.2^2 = 1) ∧ (P.1^2 / 8 - P.2^2 = 1)) :
  ∃ R, R = 3 :=
by
  sorry

end circumcircle_radius_l690_690573


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690953

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690953


namespace right_triangle_side_length_l690_690536

theorem right_triangle_side_length (A B C : Type) [metric_space A B C]
  (AB BC AC : ℝ) (h_angle : 90 = 90) (h_AB : AB = 5) (h_BC : BC = 13) :
  AC = 12 := 
by
  sorry

end right_triangle_side_length_l690_690536


namespace jimmy_needs_4_packs_of_bread_l690_690186

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end jimmy_needs_4_packs_of_bread_l690_690186


namespace no_strictly_greater_polynomials_l690_690553

noncomputable def transformation (P : Polynomial ℝ) (k : ℕ) (a : ℝ) : Polynomial ℝ := 
  P + Polynomial.monomial k (2 * a) - Polynomial.monomial (k + 1) a

theorem no_strictly_greater_polynomials (P Q : Polynomial ℝ) 
  (H1 : ∃ (n : ℕ) (a : ℝ), Q = transformation P n a)
  (H2 : ∃ (n : ℕ) (a : ℝ), P = transformation Q n a) : 
  ∃ x : ℝ, P.eval x = Q.eval x :=
sorry

end no_strictly_greater_polynomials_l690_690553


namespace two_digit_primes_with_digit_sum_ten_l690_690803

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690803


namespace matrices_are_inverses_l690_690113

-- Define variables
variables {a b c d s : ℚ}

-- Define matrices
def matrix1 : Matrix (Fin 2) (Fin 2) ℚ := ![![a, -1], ![3, b]]
def matrix2 : Matrix (Fin 2) (Fin 2) ℚ := ![![4, c], ![d, -2]]

-- Define the identity matrix
def identity_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

-- Prove that matrix1 and matrix2 are inverses
theorem matrices_are_inverses (h1 : matrix1 * matrix2 = identity_matrix) :
  a + b + c + d = s :=
by
  sorry

end matrices_are_inverses_l690_690113


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690941

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690941


namespace internal_diagonal_cubes_l690_690330

theorem internal_diagonal_cubes (a b c : ℕ) (h1 : a = 240) (h2 : b = 360) (h3 : c = 400) :
  let g := Nat.gcd (Nat.gcd a b) c in
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + g = 800 :=
by
  sorry

end internal_diagonal_cubes_l690_690330


namespace num_two_digit_primes_with_digit_sum_10_l690_690828

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690828


namespace problem_l690_690155

open Real

-- The given problem involves mathematical definitions related to binomial expansions.
theorem problem (x : ℝ) (n A : ℕ) :
  let exp := (2 / (x^(1/3)) + x^(1/2))^n in
  -- Question 1: Given that the 5th term is a constant, n = 10
  (∀ m, (4 : ℕ) = m → m = 2 + (4 - n) / 3) → n = 10 ∧
  -- Question 2: Sum of all coefficients raised to the 10th root is 3.
  (x = 1 → A = 3^10) ∧  (sqrt 10 (A) = 3) ∧
  -- Question 3: Term with the largest coefficient
  (∀ r T, r = 4 → 
    T = 2^(10 - r) * binom(n, r) * x^((5 * r - 10) / 6) →
    T = 2^4 * binom(10, 4) * x^(5/3)) := 
begin
  sorry,
end

end problem_l690_690155


namespace proposition_A_necessary_for_B_l690_690444

def is_hyperbola_trajectory (M F1 F2 : ℝ → ℝ) : Prop := 
  ∃ c : ℝ, ∀ t : ℝ, ||M t - F1 t| - |M t - F2 t|| = c

def condition_proposition_A (M F1 F2 : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ t : ℝ, ||M t - F1 t| - |M t - F2 t|| = c

theorem proposition_A_necessary_for_B {M F1 F2 : ℝ → ℝ} :
  (is_hyperbola_trajectory M F1 F2 → condition_proposition_A M F1 F2) ∧ 
  ¬(condition_proposition_A M F1 F2 → is_hyperbola_trajectory M F1 F2) :=
by
  sorry

end proposition_A_necessary_for_B_l690_690444


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690910

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690910


namespace find_k_l690_690132

noncomputable def k_value (k : ℝ) : Prop :=
  let a := (2, 1) : ℝ × ℝ
  let b := (-1, k)
  a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0

theorem find_k : k_value 12 :=
by 
  -- Applying the given conditions and properties,
  -- it should be possible to deduce that k = 12.
  sorry

end find_k_l690_690132


namespace original_weight_of_cheese_l690_690296

theorem original_weight_of_cheese (initial_pieces final_piece_weight : ℕ) 
  (h1 : initial_pieces = (420, 260))
  (h2 : final_piece_weight = 20)
  (h3 : ∀ n ≥ 0, n ≤ 3 → n * final_piece_weight = initial_pieces.2 ∨ n * final_piece_weight = initial_pieces.1):
  initial_pieces.1 + initial_pieces.2 = 680 :=
by
  apply eq.refl 680

end original_weight_of_cheese_l690_690296


namespace parallel_vectors_determine_m_l690_690486

theorem parallel_vectors_determine_m {m : ℝ} (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (1, m)) (h2 : b = (-1, 2m+1)) (h3 : a.1 * b.2 = a.2 * b.1) : 
  m = -1/3 :=
by { sorry }

end parallel_vectors_determine_m_l690_690486


namespace cone_volume_from_half_sector_l690_690702

theorem cone_volume_from_half_sector (r l : ℝ) (h : ℝ) 
    (h_r : r = 3) (h_l : l = 6) (h_h : h = 3 * Real.sqrt 3) : 
    (1 / 3) * Real.pi * r^2 * h = 9 * Real.pi * Real.sqrt 3 := 
by
  -- Sorry to skip the proof
  sorry

end cone_volume_from_half_sector_l690_690702


namespace minimize_sin_cos_half_l690_690070

noncomputable def find_angle (A : ℝ) : ℝ :=
  sin (A / 2) - cos (A / 2)

theorem minimize_sin_cos_half :
  ∃ A : ℝ, ((A = 150 * (Float.pi / 180)) ∧ (∀ B ∈ {-90 * (Float.pi / 180), 150 * (Float.pi / 180), -150 * (Float.pi / 180), 90 * (Float.pi / 180)}, find_angle A ≤ find_angle B)) :=
begin
  use 150 * (Float.pi / 180),
  split,
  { refl, },
  { intros B hB,
    fin_cases hB; sorry, }
end

end minimize_sin_cos_half_l690_690070


namespace two_digit_prime_sum_digits_10_count_l690_690925

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690925


namespace num_two_digit_primes_with_digit_sum_10_l690_690777

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690777


namespace particle_hits_origin_l690_690001

def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| x+1, y+1 => 0.25 * P x (y+1) + 0.25 * P (x+1) y + 0.5 * P x y

theorem particle_hits_origin :
    ∃ m n : ℕ, m ≠ 0 ∧ m % 4 ≠ 0 ∧ P 5 5 = m / 4^n :=
sorry

end particle_hits_origin_l690_690001


namespace concurrency_of_ad_l690_690733

-- Variables for triangle
variables {A B C D E F D' E' F' O : Point}

-- Assuming the following conditions:
-- O is the incircle of triangle ABC
-- D, E, F are the points of tangency on BC, CA, and AB respectively
-- DD', EE', FF' are diameters of circle O
axiom incircle (O : Circle) (A B C : Point) : Incircle O A B C
axiom tangency_points (O : Circle) (A B C : Point) (D E F : Point) :
  Tangent (O, D) B C ∧ Tangent (O, E) A C ∧ Tangent (O, F) A B
axiom diameters (O : Circle) (D E F D' E' F' : Point) :
  Diameter (O, D, D') ∧ Diameter (O, E, E') ∧ Diameter (O, F, F')

-- The statement to be proven:
theorem concurrency_of_ad'_be'_cf' (O : Circle) (A B C D E F D' E' F' : Point)
  (hincircle : Incircle O A B C)
  (htangency : Tangent (O, D) B C ∧ Tangent (O, E) A C ∧ Tangent (O, F) A B)
  (hdiameters : Diameter (O, D, D') ∧ Diameter (O, E, E') ∧ Diameter (O, F, F')) :
  Concurrent (Line.through A D') (Line.through B E') (Line.through C F') :=
sorry

end concurrency_of_ad_l690_690733


namespace artist_pictures_ratio_l690_690074

theorem artist_pictures_ratio :
  let total_pictures : ℝ := 180.5
  let pictures_sold : ℝ := 76.3
  let pictures_still_with_artist := total_pictures - pictures_sold
  let ratio := pictures_still_with_artist / pictures_sold
  ratio = 1042 / 763 :=
by 
  let total_pictures : ℝ := 180.5
  let pictures_sold : ℝ := 76.3
  let pictures_still_with_artist := total_pictures - pictures_sold
  let ratio := pictures_still_with_artist / pictures_sold
  have h : 104.2 = 1042 / 10 := by norm_num
  have h1 : 76.3 = 763 / 10 := by norm_num
  have h2 : pictures_still_with_artist = 104.2 := by norm_num
  have h3 : ratio = 104.2 / 76.3 := by norm_num
  rw [h, h1] at h3
  calc
    pictures_still_with_artist / pictures_sold = 1042 / 10 / (763 / 10) : by rw [h2, h1]
    ... = 1042 / 763 : by field_simp

end artist_pictures_ratio_l690_690074


namespace two_digit_primes_with_digit_sum_10_count_l690_690898

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690898


namespace trig_product_identity_l690_690024

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end trig_product_identity_l690_690024


namespace eliza_irons_dress_in_20_minutes_l690_690755

def eliza_iron_time : Prop :=
∃ d : ℕ, 
  (d ≠ 0 ∧  -- To avoid division by zero
  8 + 180 / d = 17 ∧
  d = 20)

theorem eliza_irons_dress_in_20_minutes : eliza_iron_time :=
sorry

end eliza_irons_dress_in_20_minutes_l690_690755


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690978

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690978


namespace count_two_digit_primes_with_digit_sum_10_l690_690864

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690864


namespace solution_exists_l690_690760

noncomputable def f (n : ℕ) : ℤ

axiom functional_eq (m n : ℕ) (hm : m > 0) (hn : n > 0) : ⌊(f (m * n) : ℚ) / n⌋ = f m

theorem solution_exists (α : ℝ) :
  (∀ n : ℕ, f n = ⌊α * n⌋) ∨ (∀ n : ℕ, f n = ⌈α * n⌉ - 1) := by
  sorry

end solution_exists_l690_690760


namespace initial_men_in_camp_l690_690255

theorem initial_men_in_camp (M F : ℕ) 
  (h1 : F = M * 50)
  (h2 : F = (M + 10) * 25) : 
  M = 10 :=
by
  sorry

end initial_men_in_camp_l690_690255


namespace square_fence_perimeter_l690_690287

-- Definition of the conditions
def num_posts : ℕ := 36
def post_width_in_inches : ℕ := 6
def space_between_posts_in_feet : ℕ := 4

-- Conversion of the post width to feet
def post_width_in_feet : ℝ := post_width_in_inches / 12.0

-- The main statement to prove
theorem square_fence_perimeter : 
  (4 * ((num_posts / 4) * post_width_in_feet + 
       (num_posts / 4 - 1) * space_between_posts_in_feet)) = 130 := 
by 
  sorry

end square_fence_perimeter_l690_690287


namespace line_through_origin_and_intersection_eqn_l690_690615

theorem line_through_origin_and_intersection_eqn :
  (∃ l : ℝ → ℝ → Prop, (∀ x y : ℝ, l x y ↔ 2 * x - y = 0) ∧ 
  (∃ x y : ℝ, 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0 ∧ l x y)) :=
by
  use (λ x y, 2 * x - y = 0)
  split
  { intros x y
    refl }
  { use (-1)
    use (-2)
    split <;> norm_num }
  sorry

end line_through_origin_and_intersection_eqn_l690_690615


namespace max_distance_point_on_circle_l690_690234

theorem max_distance_point_on_circle :
  (∃ (P : ℝ × ℝ), (P.1 + 4)^2 + (P.2 - 2)^2 = 5 ∧
    ∀ (O : ℝ × ℝ), O = (0, 0) → (|P.1^2 + P.2^2| ≤ (3 * sqrt 5) * (3 * sqrt 5))) :=
by
  sorry

end max_distance_point_on_circle_l690_690234


namespace f_diff_l690_690146

-- Given conditions
def f (x : ℝ) : ℝ := 3^x

-- Statement to be proved
theorem f_diff (x : ℝ) : f (x + 2) - f x = 8 * f x :=
by
  sorry

end f_diff_l690_690146


namespace peaches_total_l690_690368

theorem peaches_total (n P : ℕ) (h1 : P - 6 * n = 57) (h2 : P = 9 * (n - 6) + 3) : P = 273 :=
by
  sorry

end peaches_total_l690_690368


namespace triangle_area_y_intercepts_l690_690237

theorem triangle_area_y_intercepts :
    ∀ (c₁ c₂ : ℝ) (h₁ : c₁ + c₂ = 4) 
    (B : ℝ × ℝ) (hB : B = (3, 4)),
    let R : ℝ × ℝ := (0, c₁),
        S : ℝ × ℝ := (0, c₂),
        BRS_area : ℝ := 0.5 * (Real.sqrt ((3 - 0)^2 + (4 - c₁)^2)) * |c₁ - c₂| in
    BRS_area = 2 * Real.sqrt 13 := 
    by sorry

end triangle_area_y_intercepts_l690_690237


namespace determine_ABC_l690_690544

-- Define values in the new base system
def base_representation (A B C : ℕ) : ℕ :=
  A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + C * (A+1)^4 + B * (A+1)^3 + B * (A+1)^2 + B * (A+1) + C

-- The conditions given by the problem
def condition (A B C : ℕ) : Prop :=
  ((A+1)^8 - 2*(A+1)^4 + 1) = base_representation A B C

-- The theorem to be proved
theorem determine_ABC : ∃ (A B C : ℕ), A = 2 ∧ B = 0 ∧ C = 1 ∧ condition A B C :=
by
  existsi 2
  existsi 0
  existsi 1
  unfold condition base_representation
  sorry

end determine_ABC_l690_690544


namespace two_digit_primes_with_digit_sum_10_count_l690_690896

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l690_690896


namespace cos_identity_of_angle_l690_690458

open Real

theorem cos_identity_of_angle (α : ℝ) :
  sin (π / 6 + α) = sqrt 3 / 3 → cos (π / 3 - α) = sqrt 3 / 3 :=
by
  intro h
  sorry

end cos_identity_of_angle_l690_690458


namespace Megan_popsicles_l690_690581

def minutes_in_hour : ℕ := 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * minutes_in_hour + minutes

def popsicle_time : ℕ := 18

def popsicles_consumed (total_minutes : ℕ) (popsicle_time : ℕ) : ℕ :=
  total_minutes / popsicle_time

theorem Megan_popsicles (hours : ℕ) (minutes : ℕ) (popsicle_time : ℕ)
  (total_minutes : ℕ) (h_hours : hours = 5) (h_minutes : minutes = 36) (h_popsicle_time : popsicle_time = 18)
  (h_total_minutes : total_minutes = (5 * 60 + 36)) :
  popsicles_consumed 336 popsicle_time = 18 :=
by 
  sorry

end Megan_popsicles_l690_690581


namespace problem_not_equivalent_l690_690313

theorem problem_not_equivalent :
  (0.0000396 ≠ 3.9 * 10^(-5)) ∧ 
  (0.0000396 = 3.96 * 10^(-5)) ∧ 
  (0.0000396 = 396 * 10^(-7)) ∧ 
  (0.0000396 = (793 / 20000) * 10^(-5)) ∧ 
  (0.0000396 = 198 / 5000000) :=
by
  sorry

end problem_not_equivalent_l690_690313


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690924

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690924


namespace minimum_magnitude_vector_diff_l690_690484

namespace VectorMinimum

def vector_a (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 - t, 2 * t - 1, 0)

def vector_b (t : ℝ) : ℝ × ℝ × ℝ :=
  (2, t, t)

def vector_diff (t : ℝ) : ℝ × ℝ × ℝ :=
  let a := vector_a t
  let b := vector_b t
  (b.1 - a.1, b.2 - a.2, b.3 - a.3)

def magnitude_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2 + v.3 * v.3

theorem minimum_magnitude_vector_diff : 
  (∀ t : ℝ, magnitude_squared (vector_diff t) ≥ 2) ∧ 
  (∃ t : ℝ, magnitude_squared (vector_diff t) = 2) := 
by
  sorry

end VectorMinimum

end minimum_magnitude_vector_diff_l690_690484


namespace mr_curtis_farm_chickens_l690_690578

open Nat

theorem mr_curtis_farm_chickens (roosters hens_non_egg_laying hens_egg_laying : ℕ) 
  (h_roosters : roosters = 28)
  (h_hens_non_egg_laying : hens_non_egg_laying = 20)
  (h_hens_egg_laying : hens_egg_laying = 277) :
  let total_hens := hens_non_egg_laying + hens_egg_laying in
  let total_chickens := total_hens + roosters in
  total_chickens = 325 :=
by
  sorry

end mr_curtis_farm_chickens_l690_690578


namespace vector_expression_l690_690485

open Real -- Open the real number namespace

theorem vector_expression (m n : ℝ) 
    (a b c : ℝ × ℝ) 
    (ha : a = (1, 2)) 
    (hb : b = (-2, 3)) 
    (hc : c = (4, 1))
    (h : c = m • a + n • b) :
  m = 2 ∧ n = -1 :=
by sorry

end vector_expression_l690_690485


namespace probability_x_lt_2y_in_rect_is_1_over_6_l690_690355

open set real

-- Define the vertices of the rectangle
def rect : set (ℝ × ℝ) := {p | (0 ≤ p.1 ∧ p.1 ≤ 4) ∧ (0 ≤ p.2 ∧ p.2 ≤ 3)}

-- Define the subset where x < 2y
def region_of_interest : set (ℝ × ℝ) := {p | p.1 < 2 * p.2}

-- Calculate the measure of the region of interest within the rectangle
noncomputable def region_measure : ennreal := volume ((rect ∩ region_of_interest).to_set)

-- Calculate the measure of the rectangle
noncomputable def rect_measure : ennreal := volume (rect.to_set)

-- Define the probability as the ratio of the region's measure to the rectangle's measure
noncomputable def probability : ennreal := region_measure / rect_measure

-- Theorem statement
theorem probability_x_lt_2y_in_rect_is_1_over_6 : probability = 1 / 6 := sorry

end probability_x_lt_2y_in_rect_is_1_over_6_l690_690355


namespace problem1_problem2_l690_690475

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- Problem 1: Find x such that f(x) < -2 when a = 1
theorem problem1 : 
  {x : ℝ | f x 1 < -2} = {x | x > 3 / 2} :=
sorry

-- Problem 2: Find the range of values for 'a' when -2 + f(y) ≤ f(x) ≤ 2 + f(y) for all x, y ∈ ℝ
theorem problem2 : 
  (∀ x y : ℝ, -2 + f y a ≤ f x a ∧ f x a ≤ 2 + f y a) ↔ (-3 ≤ a ∧ a ≤ -1) :=
sorry

end problem1_problem2_l690_690475


namespace largest_of_roots_l690_690418

theorem largest_of_roots {p q r s t : ℝ} 
  (h1 : p + q + r + s + t = 3)
  (h2 : p * q + p * r + p * s + p * t + q * r + q * s + q * t + r * s + r * t + s * t = -8)
  (h3 : p * q * r * s * t = -21) :
  max p (max q (max r (max s t))) = sqrt 7 := 
sorry

end largest_of_roots_l690_690418


namespace count_two_digit_primes_with_digit_sum_10_l690_690860

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690860


namespace two_digit_primes_with_digit_sum_ten_l690_690794

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690794


namespace harry_total_travel_time_l690_690492

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l690_690492


namespace Shekar_marks_in_Science_l690_690227

theorem Shekar_marks_in_Science (S : ℕ) (h : (76 + S + 82 + 67 + 85) / 5 = 75) : S = 65 :=
sorry

end Shekar_marks_in_Science_l690_690227


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690917

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690917


namespace two_digit_primes_with_digit_sum_ten_l690_690797

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690797


namespace find_natural_solution_l690_690059

theorem find_natural_solution : ∃ (a b c : ℕ), a^3 + b^4 = c^5 :=
by
  let a := 4
  let b := 16
  let c := 18
  use a
  use b
  use c
  have h : a^3 + b^4 = c^5,
  {
    -- Proof omitted for brevity
    sorry
  }
  exact h

end find_natural_solution_l690_690059


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690940

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690940


namespace force_for_18_inch_wrench_l690_690256

theorem force_for_18_inch_wrench (F : ℕ → ℕ → ℕ) : 
  (∀ L : ℕ, ∃ k : ℕ, F 300 12 = F (F L k) L) → 
  ((F 12 300) = 3600) → 
  (∀ k : ℕ, F (F 6 k) 6 = 3600) → 
  (∀ k : ℕ, F (F 18 k) 18 = 3600) → 
  (F 18 200 = 3600) :=
by
  sorry

end force_for_18_inch_wrench_l690_690256


namespace triangle_angle_proof_l690_690290

theorem triangle_angle_proof
  (ABC : Type) [triangle ABC]
  (AB AC BC : ℝ)
  (D E F : point)
  (BAE ACD BFE : angle)
  (CFE : triangle) 
  (isosceles_BFE : is_isosceles ∠ BFE)
  (equilateral_CFE : is_equilateral ∠ CFE)
  (AB_eq_3AC : AB = 3 * AC) 
  (angle_eq : ∠ BAE = ∠ ACD) 
  (intersection : point ∈ (line AE) ∩ (line CD))
  : ∠ ACB = 60 :=
sorry

end triangle_angle_proof_l690_690290


namespace general_term_sum_first_n_terms_b_l690_690462

-- Define the sequence S_n
def S (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a i

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n

-- Define the given condition 2 S_n = a_n^2 + a_n
axiom seq_a_property : ∀ n : ℕ, 2 * S n = a n ^ 2 + a n

-- Proving the general term formula a_n = n
theorem general_term (n : ℕ) : a n = n := by
  sorry  -- Proof

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 3^(-a n) + (1 / (a n * a (n + 1))

-- Proving the sum of the first n terms of sequence b_n
theorem sum_first_n_terms_b (n : ℕ) : ∑ i in Finset.range (n + 1), b i = (3 / 2) - (1 / (2 * 3^n)) - (1 / (n + 1)) := by
  sorry  -- Proof

end general_term_sum_first_n_terms_b_l690_690462


namespace min_value_of_f_in_interval_l690_690128

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem min_value_of_f_in_interval (k : ℝ) :
  (f 1 k = -k ∧ k ≤ 2) ∨ 
  (∃ k', k' = 2 ∧ f (k'/2) k = - (k'^2) / 4 - 1 ∧ 2 < k ∧ k < 8) ∨ 
  (f 4 k = 15 - 4 * k ∧ k ≥ 8) :=
by sorry

end min_value_of_f_in_interval_l690_690128


namespace arithmetic_sequence_sum_first_10_terms_l690_690540

variable (a : ℕ → ℝ) (a5 a6 : ℝ)
variable [Pow ℝ ℕ]

def S10 (a : ℕ → ℝ) : ℝ := (a 1 + a 10) / 2 * 10

theorem arithmetic_sequence_sum_first_10_terms (h : a 5 + a 6 = 10) : S10 a = 50 :=
  by sorry

end arithmetic_sequence_sum_first_10_terms_l690_690540


namespace train_length_l690_690715

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end train_length_l690_690715


namespace min_area_triangle_ABC_l690_690091

noncomputable def y (x : ℝ) : ℝ := sqrt (2 * x)

def on_parabola (A : ℝ × ℝ) : Prop := A.2 ^ 2 = 2 * A.1

def on_circle (B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  (B.1 - 1) ^ 2 + B.2 ^ 2 = 1 ∧ (C.1 - 1) ^ 2 + C.2 ^ 2 = 1

def tangent_condition (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let l := (A.2 - B.2) / (A.1 - B.1)
  l = -(A.1 / A.2)

def triangle_area (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (B.2 - C.2)) * A.1

theorem min_area_triangle_ABC 
  (A B C : ℝ × ℝ) 
  (hA : on_parabola A) 
  (hA_x : A.1 > 2) 
  (hB : B.1 = 0) 
  (hC : C.1 = 0) 
  (hb : on_circle B C) 
  (ht : tangent_condition A B C) 
  : triangle_area A B C = 8 :=
sorry

end min_area_triangle_ABC_l690_690091


namespace part_I_part_II_part_III_l690_690095

-- Definitions based on given problem
def is_perfect_set (P : Set ℕ) [fintype P] : Prop :=
  ∃ (A B C : Set ℕ), 
  A ∪ B ∪ C = P ∧ 
  A ∩ B = ∅ ∧ 
  A ∩ C = ∅ ∧ 
  B ∩ C = ∅ ∧ 
  same_cardinality A B C ∧ 
  sorted C ∧ 
  equal_elements_sum A B C 

-- Prove that {1, 2, 3} is a perfect set and {1, 2, 3, 4, 5, 6} is not
theorem part_I : is_perfect_set {1, 2, 3} ∧ ¬ is_perfect_set {1, 2, 3, 4, 5, 6} := 
  sorry

-- Prove possible values of x in {1, x, 3, 4, 5, 6} to make it a perfect set are 7, 9, or 11
theorem part_II (x : ℕ) (hx : {1, x, 3, 4, 5, 6}.card = 6):
  is_perfect_set {1, x, 3, 4, 5, 6} ↔ (x = 7 ∨ x = 9 ∨ x = 11) := 
  sorry

-- Prove necessary condition for {x ∣ 1 ≤ x ≤ 3n, n ∈ ℕ*} to be a perfect set is n = 4k or n = 4k + 1
theorem part_III (n : ℕ) (hn : 0 < n):
  is_perfect_set { x | 1 ≤ x ∧ x ≤ 3 * n } → (∃ k, n = 4 * k ∨ n = 4 * k + 1) := 
  sorry

end part_I_part_II_part_III_l690_690095


namespace count_two_digit_prime_with_digit_sum_10_l690_690869

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690869


namespace mathequivalent_problem_l690_690557

noncomputable def probability_slope_ge_one : ℚ := 1 / 8

theorem mathequivalent_problem 
  (P : ℝ × ℝ) 
  (hP : 0 < P.1 ∧ P.1 < 1 ∧ 0 < P.2 ∧ P.2 < 1) :
  let Q := (3/4, 1/4)
  in (P.2 - Q.2) / (P.1 - Q.1) ≥ 1 → probability_slope_ge_one = 1/8 := 
sorry

end mathequivalent_problem_l690_690557


namespace max_set_T_l690_690559

def max_elements_T : ℕ :=
  let S : set ℕ := { n | n > 0 ∧ n ≤ 75 }
  in {T : set ℕ | T ⊆ S ∧ (∀ ⦃a b : ℕ⦄, a ∈ T → b ∈ T → a ≠ b → ¬(a + b) % 5 = 0)}.Sup (set.size)

/-- Theorem: The maximum number of elements in the set T satisfying the given conditions is 30. -/
theorem max_set_T (T : set ℕ) (h_T : T ⊆ S)
  (no_divisible_pairs : ∀ ⦃a b : ℕ⦄, a ∈ T → b ∈ T → a ≠ b → ¬(a + b) % 5 = 0) : 
  set.size T ≤ 30 := by
  sorry

end max_set_T_l690_690559


namespace articles_gain_percent_l690_690246

theorem articles_gain_percent (C S : ℝ) (h : 20 * C = 10 * S) : (S = 2 * C) → (S - C) / C * 100 = 100 := by
  intro hS
  rw hS
  have gain : S - C = C := by linarith
  rw gain
  have percent : (C / C) * 100 = 100 := by
    rw div_self
    norm_num
    exact (ne_of_gt (by linarith))
  exact percent

end articles_gain_percent_l690_690246


namespace unique_solution_l690_690678

-- Define the conditions on the coefficients
variables {a11 a22 a33 : ℝ}
variables {a12 a13 a21 a23 a31 a32 : ℝ}
variables (x1 x2 x3 : ℝ)

-- Hypotheses as conditions
def conditions := 
  (0 < a11) ∧ (0 < a22) ∧ (0 < a33) ∧
  (a12 < 0) ∧ (a13 < 0) ∧ (a21 < 0) ∧ (a23 < 0) ∧ (a31 < 0) ∧ (a32 < 0) ∧
  ((a11 + a12 + a13 > 0) ∧ (a21 + a22 + a23 > 0) ∧ (a31 + a32 + a33 > 0))

-- The system of equations
def system_of_equations := 
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧ 
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧ 
  a31 * x1 + a32 * x2 + a33 * x3 = 0

-- The theorem stating the unique solution
theorem unique_solution (x1 x2 x3 : ℝ) (h1 : conditions) (h2 : system_of_equations) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 :=
sorry

end unique_solution_l690_690678


namespace convert_degrees_to_dms_l690_690410

def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

theorem convert_degrees_to_dms {deg min sec : ℝ} :
  (deg = 21) → (min = 0.24) →
  (sec = degrees_to_minutes min) →
  (minutes_to_seconds (sec - sec.to_int) = 24) →
  deg + sec.to_int / 60 + minutes_to_seconds (sec - sec.to_int) / 3600 = 21 + 14 / 60 + 24 / 3600 :=
by
  intros h_deg h_min h_sec h_sec_correct
  rw [h_deg, h_min, h_sec]
  sorry

end convert_degrees_to_dms_l690_690410


namespace lice_checks_time_in_hours_l690_690269

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l690_690269


namespace sum_of_absolute_values_l690_690463

theorem sum_of_absolute_values (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n + 2) →
  a 1 = -1 →
  (∀ n, 1 < n → a n = 2 * n - 5) →
  ((abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) +
    abs (a 6) + abs (a 7) + abs (a 8) + abs (a 9) + abs (a 10)) = 66) :=
by
  intros hS a1_eq ha_eq
  sorry

end sum_of_absolute_values_l690_690463


namespace original_weight_of_cheese_l690_690295

theorem original_weight_of_cheese (initial_pieces final_piece_weight : ℕ) 
  (h1 : initial_pieces = (420, 260))
  (h2 : final_piece_weight = 20)
  (h3 : ∀ n ≥ 0, n ≤ 3 → n * final_piece_weight = initial_pieces.2 ∨ n * final_piece_weight = initial_pieces.1):
  initial_pieces.1 + initial_pieces.2 = 680 :=
by
  apply eq.refl 680

end original_weight_of_cheese_l690_690295


namespace planes_parallel_or_overlap_l690_690461

def vector (α : Type) := α × α × α

def is_parallel_or_same_plane (u v : vector ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v = (k * u.1, k * u.2, k * u.3)

theorem planes_parallel_or_overlap
  (u : vector ℝ)
  (v : vector ℝ)
  (h1 : u = (1, 2, -2))
  (h2 : v = (-3, -6, 6)) :
  is_parallel_or_same_plane u v :=
by {
  sorry
}

end planes_parallel_or_overlap_l690_690461


namespace convert_cylindrical_to_rectangular_l690_690387

noncomputable theory

open Real

-- Define the cylindrical coordinates
def r : ℝ := 5
def theta : ℝ := π / 3
def z_cylindrical : ℝ := 2

-- Define the expected rectangular coordinates
def x_rect : ℝ := 2.5
def y_rect : ℝ := 5 * sqrt(3) / 2
def z_rect : ℝ := 2

-- Lean 4 statement to verify conversion
theorem convert_cylindrical_to_rectangular
  (r θ z_cylindrical x_rect y_rect z_rect : ℝ)
  (hr : r = 5) (htheta : θ = π / 3) (hz : z_cylindrical = 2)
  (hx : x_rect = 5 * cos (π / 3)) (hy : y_rect = 5 * sin (π / 3)) (hz_rect : z_rect = z_cylindrical) :
  (x_rect, y_rect, z_rect) = (2.5, 5 * sqrt(3) / 2, 2) :=
by { 
  rw [hr, htheta, hz] at *,
  rw cos_pi_div_three at hx, 
  rw sin_pi_div_three at hy,
  exact ⟨hx, hy, hz_rect⟩,
  sorry
}

end convert_cylindrical_to_rectangular_l690_690387


namespace keys_arrangement_l690_690167
-- Import the entire Mathlib library for broad functionality.

-- Expressing the problem in Lean 4 as a non-computable definition.
noncomputable def distinct_keychain_arrangements : ℕ := 
  let block_permutations := 2
  let keychain_positions := (4 - 1)! / 2
  block_permutations * keychain_positions

-- Defining the main theorem to prove the number of arrangements is 6.
theorem keys_arrangement : distinct_keychain_arrangements = 6 := by
  sorry

end keys_arrangement_l690_690167


namespace count_prime_two_digit_sum_ten_is_three_l690_690816

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690816


namespace factor_difference_of_squares_l690_690414

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l690_690414


namespace min_value_l690_690770

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 10 * x - 6 * y + 3

theorem min_value : ∃ x y : ℝ, (x + y = 2) ∧ (f x y = -(1/7)) :=
by
  sorry

end min_value_l690_690770


namespace draw_line_ways_l690_690503

theorem draw_line_ways : 
  (∀ (x y : ℝ), x * sin (sqrt (16 - x^2 - y^2)) = 0 → 
    (x = 0 ∨ ∃ k : ℤ, sqrt (16 - x^2 - y^2) = k * π) → 
    ∃ n : ℕ, n = 72) :=
by
  sorry

end draw_line_ways_l690_690503


namespace minimum_value_of_expression_l690_690150

variables {x y z : ℝ}

theorem minimum_value_of_expression (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x^2 + y^2 + z^2 = 1) :
  (minimizing (λ x y z, (yz/x + xz/y + xy/z)) with the condition x^2 + y^2 + z^2 = 1) = sqrt 3 :=
sorry

end minimum_value_of_expression_l690_690150


namespace camilla_blueberry_jelly_beans_l690_690380

theorem camilla_blueberry_jelly_beans (b c : ℕ) 
  (h1 : b = 3 * c)
  (h2 : b - 20 = 2 * (c - 5)) : 
  b = 30 := 
sorry

end camilla_blueberry_jelly_beans_l690_690380


namespace count_prime_two_digit_sum_ten_is_three_l690_690815

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690815


namespace largest_divisor_proof_l690_690652

noncomputable def largest_divisor (P : Nat -> Nat) := ∀ n : Nat, (n > 0) ∧ (n % 2 = 0) → P n

theorem largest_divisor_proof :
  largest_divisor (λ n => (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11)) = 105 :=
by
  sorry

end largest_divisor_proof_l690_690652


namespace numOxygenAtoms_l690_690340

-- Define the conditions as hypothesis
def numCarbonAtoms : ℕ := 4
def numHydrogenAtoms : ℕ := 8
def molecularWeight : ℕ := 88
def atomicWeightCarbon : ℕ := 12
def atomicWeightHydrogen : ℕ := 1
def atomicWeightOxygen : ℕ := 16

-- The statement to be proved
theorem numOxygenAtoms :
  let totalWeightC := numCarbonAtoms * atomicWeightCarbon
  let totalWeightH := numHydrogenAtoms * atomicWeightHydrogen
  let totalWeightCH := totalWeightC + totalWeightH
  let weightOxygenAtoms := molecularWeight - totalWeightCH
  let numOxygenAtoms := weightOxygenAtoms / atomicWeightOxygen
  numOxygenAtoms = 2 :=
by {
  sorry
}

end numOxygenAtoms_l690_690340


namespace quadrilateral_area_l690_690744

def point := (ℝ × ℝ)

def vertices : List point := [(2,1), (1,6), (4,5), (9,9)]

def shoelace_formula (vertices : List point) : ℝ :=
  let xs := vertices.map Prod.fst
  let ys := vertices.map Prod.snd
  0.5 * abs (
    xs.zipWith (*) (ys.tail.concat (ys.head!)).sum -
    ys.zipWith (*) (xs.tail.concat (xs.head!)).sum
  )

theorem quadrilateral_area :
  shoelace_formula vertices = 13 := by
  sorry

end quadrilateral_area_l690_690744


namespace count_n_values_not_divisible_by_3_l690_690774

theorem count_n_values_not_divisible_by_3 : 
  (∃ n : ℕ, n ≤ 1500 ∧
   (∃ c : ℕ, 
    \sum(i : [1498, 1499, 1500], ⌊i/n⌋) % 3 ≠ 0) 
      ↥(0 < n) ↥(n ≤ 1500) ↥(n ≠ 0 ∧ ∃ m : ℤ, m ≠ 0 )) = 2 := 
sorry

end count_n_values_not_divisible_by_3_l690_690774


namespace count_two_digit_primes_with_digit_sum_10_l690_690880

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690880


namespace convert_cylindrical_to_rectangular_l690_690391

-- Definitions of the conversion from cylindrical to rectangular coordinates
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- The given cylindrical coordinates point
def point_cylindrical : ℝ × ℝ × ℝ := (5, Real.pi / 3, 2)

-- The expected rectangular coordinates result
def expected_point_rectangular : ℝ × ℝ × ℝ := (2.5, 5 * Real.sqrt 3 / 2, 2)

-- The theorem statement to prove
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 5 (Real.pi / 3) 2 = expected_point_rectangular :=
by
  sorry

end convert_cylindrical_to_rectangular_l690_690391


namespace circle_condition_l690_690464

noncomputable def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, circle_eq x y m) → m < 1 / 4 :=
by
  sorry

end circle_condition_l690_690464


namespace game_result_l690_690266

theorem game_result (a : ℤ) : ((2 * a + 6) / 2 - a = 3) :=
by
  sorry

end game_result_l690_690266


namespace green_to_blue_ratio_l690_690740

-- Definition of the problem conditions
variable (G B R : ℕ)
variable (H1 : 2 * G = R)
variable (H2 : B = 80)
variable (H3 : R = 1280)

-- Theorem statement: the ratio of the green car's speed to the blue car's speed is 8:1
theorem green_to_blue_ratio (G B R : ℕ) (H1 : 2 * G = R) (H2 : B = 80) (H3 : R = 1280) :
  G / B = 8 :=
by
  sorry

end green_to_blue_ratio_l690_690740


namespace reciprocal_of_repeating_decimal_l690_690656

theorem reciprocal_of_repeating_decimal : ∃ x : ℚ, (x = (36 / 99 : ℚ)) ∧ (x⁻¹ = (11 / 4 : ℚ)) :=
by
  use 36 / 99
  split
  { sorry }
  { sorry }

end reciprocal_of_repeating_decimal_l690_690656


namespace sarah_minimum_correct_answers_l690_690604

noncomputable def min_correct_answers (total_problems : ℕ) (points_per_correct : ℕ) 
  (points_per_unanswered : ℚ) (unanswered : ℕ) (minimum_points : ℚ) : ℕ :=
  let unanswered_points := unanswered * points_per_unanswered
  let needed_points := minimum_points - unanswered_points
  let correct_answers := needed_points / points_per_correct
  ⌈correct_answers⌉

theorem sarah_minimum_correct_answers :
  min_correct_answers 22 6 (3 / 2) 3 100 = 16 :=
by
  simp [min_correct_answers]
  sorry

end sarah_minimum_correct_answers_l690_690604


namespace primes_between_50_and_100_with_prime_remainder_is_7_l690_690500
-- Importing the Mathlib library

-- The proof problem in Lean 4 statement
theorem primes_between_50_and_100_with_prime_remainder_is_7 :
  let primes_range := [p | p ∈ (list.range (100 - 50 + 1)).map (fun i => i + 50), Nat.prime p]
      primes_remainders := [r for r in primes_range, r % 10 == 1 ∨ r % 10 == 3 ∨ r % 10 == 7]
  in primes_remainders.length = 7 := 
by
  sorry

end primes_between_50_and_100_with_prime_remainder_is_7_l690_690500


namespace problem_l690_690343

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l690_690343


namespace number_of_games_each_team_plays_l690_690631

theorem number_of_games_each_team_plays
  (teams : ℕ)
  (games_played : ℕ)
  (teams_eq : teams = 10)
  (games_played_eq : games_played = 45) :
  ∃ x : ℕ, (teams * (teams - 1) / 2) * x = games_played := by
  have h : teams * (teams - 1) / 2 = 45 from
    calc
      teams * (teams - 1) / 2
        = 10 * 9 / 2 : by rw [teams_eq]
        ... = 90 / 2 : by norm_num
        ... = 45 : by norm_num
  use 1
  rw [h, games_played_eq]
  have : 45 * 1 = 45 := by norm_num
  exact this

end number_of_games_each_team_plays_l690_690631


namespace coefficient_x2_l690_690649

open Polynomials

noncomputable def polynomial1 : Polynomial ℝ := 3 * X ^ 3 + 4 * X ^ 2 + 2 * X - 3
noncomputable def polynomial2 : Polynomial ℝ := 2 * X ^ 2 + 7 * X - 4

theorem coefficient_x2 (p1 p2 : Polynomial ℝ) (h1 : p1 = 3 * X ^ 3 + 4 * X ^ 2 + 2 * X - 3) (h2 : p2 = 2 * X ^ 2 + 7 * X - 4) :
  coeff (p1 * p2) 2 = -8 :=
by
  rw [h1, h2]
  rw Polynomial.mul_def
  -- The detailed proof would continue here, but we'll skip it
  -- So assigning sorry to make sure the code can build
  sorry

end coefficient_x2_l690_690649


namespace two_digit_prime_sum_digits_10_count_l690_690932

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l690_690932


namespace count_two_digit_primes_with_digit_sum_10_l690_690987

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690987


namespace sum_over_divisors_of_60_l690_690069

-- Definitions of functions d, sigma, and phi
def d (n : ℕ) := (finset.divisors n).card
def sigma (n : ℕ) := (finset.divisors n).sum
def phi (n : ℕ) := (finset.range (n + 1)).filter (λ m, n.gcd m = 1).card

-- Define the specific problem
def sum_divisors_expr (n : ℕ) : ℚ :=
  (finset.divisors n).sum (λ i, (sigma i * d i : ℚ) / (phi i))

theorem sum_over_divisors_of_60 :
  sum_divisors_expr 60 = 350 :=
by {
  sorry
}

end sum_over_divisors_of_60_l690_690069


namespace count_more_4s_than_8s_l690_690042

-- Let us define the necessary information:
def book_pages : List Nat := List.range' 1 589

-- Define a function to count the occurrences of a digit in the list of page numbers
def count_digit (d : Nat) (pages : List Nat) : Nat :=
  pages.foldl (λ acc n => acc + (n.digits 10).count (== d)) 0

theorem count_more_4s_than_8s :
  count_digit 4 book_pages - count_digit 8 book_pages = 100 :=
by
  sorry

end count_more_4s_than_8s_l690_690042


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690921

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690921


namespace largest_perfect_square_factor_of_2520_l690_690305

noncomputable def largest_perfect_square_factor_2520 : ℕ :=
  36

theorem largest_perfect_square_factor_of_2520 :
  (∀ n : ℕ, n * n ∣ 2520 → n * n ≤ 36 ∨ n * n = 36) :=
begin
  sorry
end

end largest_perfect_square_factor_of_2520_l690_690305


namespace smallest_int_rel_prime_150_l690_690772

theorem smallest_int_rel_prime_150 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 150 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 150 = 1) → x ≤ y :=
by
  sorry

end smallest_int_rel_prime_150_l690_690772


namespace parabola_focus_l690_690767

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1) = (0, 1) :=
by 
  -- key steps would go here
  sorry

end parabola_focus_l690_690767


namespace two_digit_prime_sum_to_ten_count_l690_690838

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690838


namespace house_height_proof_l690_690651

noncomputable def height_of_house (house_shadow tree_height tree_shadow : ℕ) : ℕ :=
  house_shadow * tree_height / tree_shadow

theorem house_height_proof
  (house_shadow_length : ℕ)
  (tree_height : ℕ)
  (tree_shadow_length : ℕ)
  (expected_house_height : ℕ)
  (Hhouse_shadow_length : house_shadow_length = 56)
  (Htree_height : tree_height = 21)
  (Htree_shadow_length : tree_shadow_length = 24)
  (Hexpected_house_height : expected_house_height = 49) :
  height_of_house house_shadow_length tree_height tree_shadow_length = expected_house_height :=
by
  rw [Hhouse_shadow_length, Htree_height, Htree_shadow_length, Hexpected_house_height]
  -- Here we should compute the value and show it is equal to 49
  sorry

end house_height_proof_l690_690651


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690952

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690952


namespace train_length_proof_l690_690718

-- Define speeds and time taken
def speed_train_kmph : ℝ := 63
def speed_man_kmph : ℝ := 3
def time_crossing_seconds : ℝ := 41.9966402687785

-- Speed conversion factor
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train_kmph - speed_man_kmph)

-- Length of the train
def length_of_train : ℝ := relative_speed_mps * time_crossing_seconds

-- Proof stating the length of the train is approximately 699.94400447975 meters
theorem train_length_proof : abs (length_of_train - 699.94400447975) < 1e-6 := by
  sorry

end train_length_proof_l690_690718


namespace smallest_overlap_l690_690213

-- Definitions for percentages of people using smartphones and laptops
def smartphone_users := 0.90
def laptop_users := 0.80

-- The theorem stating the smallest percentage of people who use both
theorem smallest_overlap : ∃ (both_smartphones_and_laptops : ℝ), both_smartphones_and_laptops = 0.70 :=
by
  have sum_users := smartphone_users + laptop_users
  have overlap := sum_users - 1
  use overlap
  sorry

end smallest_overlap_l690_690213


namespace count_two_digit_primes_with_digit_sum_10_l690_690886

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690886


namespace sin2alpha_cos2beta_l690_690101

variable (α β : ℝ)

-- Conditions
def tan_add_eq : Prop := Real.tan (α + β) = -3
def tan_sub_eq : Prop := Real.tan (α - β) = 2

-- Question
theorem sin2alpha_cos2beta (h1 : tan_add_eq α β) (h2 : tan_sub_eq α β) : 
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = -1 / 7 := 
  sorry

end sin2alpha_cos2beta_l690_690101


namespace arithmetic_sequence_eight_terms_l690_690407

theorem arithmetic_sequence_eight_terms 
  (a d : ℤ)
  (seq : Fin 8 → ℤ)
  (h1 : seq 0 = a)
  (h2 : seq 1 = a + d)
  (h3 : seq 2 = a + 2 * d)
  (h4 : seq 3 = a + 3 * d)
  (h5 : seq 4 = a + 4 * d)
  (h6 : seq 5 = a + 5 * d)
  (h7 : seq 6 = a + 6 * d)
  (h8 : seq 7 = a + 7 * d)
  (h_middle_sum : seq 3 + seq 4 = 41)
  (h_first_last_prod : seq 0 * seq 7 = 114) : 
  seq = (fun n => 3 + n * 5) ∨ seq = (fun n => 38 - n * 5) :=
by
  sorry

end arithmetic_sequence_eight_terms_l690_690407


namespace jimmy_bread_packs_needed_l690_690184

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l690_690184


namespace reciprocal_of_repeating_decimal_l690_690655

theorem reciprocal_of_repeating_decimal : ∃ x : ℚ, (x = (36 / 99 : ℚ)) ∧ (x⁻¹ = (11 / 4 : ℚ)) :=
by
  use 36 / 99
  split
  { sorry }
  { sorry }

end reciprocal_of_repeating_decimal_l690_690655


namespace percentage_of_passengers_in_first_class_l690_690215

-- Define the conditions given in the problem
def total_passengers : ℕ := 120
def percentage_female : ℕ := 30
def females_in_coach_class : ℕ := 28
def fraction_male_first_class : ℝ := 1 / 3

-- Define the calculation of the number of female passengers and passengers in first class
def number_female_passengers : ℕ := total_passengers * percentage_female / 100
def number_female_first_class : ℕ := number_female_passengers - females_in_coach_class
def six_div_two_thirds : ℕ := 8 * 3 / 2

def percentage_first_class : ℝ := six_div_two_thirds / total_passengers * 100

-- Theorem statement to prove the percentage of passengers in first class is 30%
theorem percentage_of_passengers_in_first_class :
  percentage_first_class = 30 := by
  sorry

end percentage_of_passengers_in_first_class_l690_690215


namespace ordered_pair_solution_l690_690400

theorem ordered_pair_solution :
  ∃ (x y : ℚ), 3 * x + 4 * y = 1 ∧ 7 * x - y = 6 ∧
               x = 25 / 31 ∧ y = -11 / 31 :=
by
  let x := 25 / 31
  let y := -11 / 31
  use x, y
  split
  {
    show 3 * x + 4 * y = 1
    sorry
  }
  split
  {
    show 7 * x - y = 6
    sorry
  }
  split
  {
    show x = 25 / 31
    sorry
  }
  {
    show y = -11 / 31
    sorry
  }

end ordered_pair_solution_l690_690400


namespace count_prime_two_digit_sum_ten_is_three_l690_690814

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690814


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690922

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690922


namespace count_two_digit_prime_with_digit_sum_10_l690_690871

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690871


namespace even_integer_a_l690_690555

theorem even_integer_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : ∀ n : ℕ, even (⌊a * n + b⌋)) : ∃ k : ℤ, a = 2 * k :=
sorry

end even_integer_a_l690_690555


namespace range_of_a_l690_690662

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l690_690662


namespace proof1_proof2_l690_690685

noncomputable def problem1 : Prop :=
  0.064^(-1 / 3) - (-1 / 8)^0 + 7^(Real.log 2 / Real.log 7) + 0.25^(5 / 2) * 0.5^(-4) = 4

theorem proof1 : problem1 := sorry

noncomputable def problem2 (a b : ℝ) (a_def : a = Real.log 2 / Real.log 10) (b_def : 10^b = 3) : Prop :=
  Real.log (30^(1 / 2)) / Real.log 6 = (b + 1) / (2 * (a + b))

theorem proof2 (a b : ℝ) (a_def : a = Real.log 2 / Real.log 10) (b_def : 10^b = 3) : problem2 a b a_def b_def := sorry

end proof1_proof2_l690_690685


namespace choose_representatives_l690_690286

theorem choose_representatives :
  let females := 3
  let males := 4
  let total_representatives := 3
  ∃ (ways : ℕ), ways = 30 ∧
    (ways = (nat.choose females 1 * nat.choose males 2 + 
             nat.choose females 2 * nat.choose males 1)) :=
by
  sorry

end choose_representatives_l690_690286


namespace math_problem_l690_690120

-- Definition of the function f
def f (x : ℝ) : ℝ := (1/2) * x^3 - (3/2) * x

-- Proof statement
theorem math_problem (
  h1 : ∀ x : ℝ, f (-x) = -f x,
  h2 : f (-1) = 1,
  h3 : ∀ x1 x2 ∈ [-1, 1], abs (f x1 - f x2) ≤ 2
) : ∃ a b c : ℝ, a ≠ 0 ∧ f = a * (λ x, x^3) + b * (λ x, x^2) + c * (λ x, x) :=
sorry

end math_problem_l690_690120


namespace arithmetic_mean_squares_l690_690241

theorem arithmetic_mean_squares (n : ℕ) (h : 0 < n) :
  let S_n2 := (n * (n + 1) * (2 * n + 1)) / 6 
  let A_n2 := S_n2 / n
  A_n2 = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end arithmetic_mean_squares_l690_690241


namespace frustum_similarity_ratios_l690_690654

theorem frustum_similarity_ratios (m : ℝ) :
  (∀ (F1 F2 : frustum_cone),
    -- condition: F1 and F2 are similar frustums
    (similar F1 F2 m) →
    -- assertions:
    -- 1. Lateral Surface Area Ratio
    (ratio_lateral_surface F1 F2 = m^2) ∧
    -- 2. Total Surface Area Ratio
    (ratio_total_surface F1 F2 = m^2) ∧
    -- 3. Volume Ratio
    (ratio_volume F1 F2 = m^3) ∧
    -- 4. Radii of Cross-Sections Ratio
    (ratio_radii_midsection F1 F2 = m)) := sorry

end frustum_similarity_ratios_l690_690654


namespace trapezium_distance_l690_690764

theorem trapezium_distance (a b h : ℝ) (h_area : 190 = (1/2) * (a + b) * h) (h_a : a = 20) (h_b : b = 18) : h = 10 :=
by
  simp [h_a, h_b] at h_area
  linarith

end trapezium_distance_l690_690764


namespace max_gcd_seq_l690_690620

theorem max_gcd_seq : ∃ n ∈ ℕ, ∀ m ∈ ℕ, gcd (99 + 2 * m^2) (99 + 2 * (m+1)^2) <= 11 :=
by sorry

end max_gcd_seq_l690_690620


namespace two_digit_prime_sum_to_ten_count_l690_690844

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690844


namespace find_varphi_l690_690123

theorem find_varphi (φ : ℝ) (hφ : 0 ≤ φ ∧ φ ≤ π) :
  cos (π / 3) = sin (2 * (π / 3) + φ) → φ = π / 6 :=
by
  sorry

end find_varphi_l690_690123


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690950

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l690_690950


namespace factor_t_squared_minus_144_l690_690412

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l690_690412


namespace fox_alice_possible_numbers_l690_690080

-- Definition of the problem conditions.
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by (n d : ℕ) : Prop :=
  n % d = 0

-- Main theorem statement.
theorem fox_alice_possible_numbers :
  {n : ℕ | is_two_digit n ∧
   (divisible_by n 2 ∨ divisible_by n 3 ∨ divisible_by n 4 ∨ divisible_by n 5 ∨ divisible_by n 6) ∧
   ((divisible_by n 2 ∧ ¬divisible_by n 3 ∧ divisible_by n 4 ∧ ¬divisible_by n 5 ∧ divisible_by n 6) ∨
    (divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 4 ∧ ¬divisible_by n 5 ∧ ¬divisible_by n 6) ∨
    (¬divisible_by n 2 ∧ divisible_by n 3 ∧  divisible_by n 4 ∧ divisible_by n 5 ∧ ¬divisible_by n 6))}.card = 8 :=
by
  -- SORRY is used here to indicate the theorem proof is omitted.
  sorry

end fox_alice_possible_numbers_l690_690080


namespace main_l690_690309

lemma perfect_square_1 : ∃ x : ℝ, x^2 = 1^2016 :=
begin
  use 1^1008,
  norm_num,
end

lemma not_perfect_square_2 : ¬ ∃ x : ℝ, x^2 = 2^2017 :=
begin
  intro h,
  cases h with x hx,
  have h1 : x^2 = 2^2016 * 2,
  {
    rw [hx, pow_add, mul_comm],
  },
  have h2 : ∃ y : ℝ, y^2 = 2^2016,
  {
    use x / sqrt 2,
    field_simp [pow_two, mul_comm],
    exact h1.symm,
  },
  suffices : (2 : ℝ) = sqrt 2 * sqrt 2, by norm_num at this,
  rwa [←sqrt_mul (show 2 ≥ 0, by norm_num), mul_self_sqrt (show 2 ≥ 0, by norm_num)] at h2,
end

lemma perfect_square_3 : ∃ x : ℝ, x^2 = 3^2018 :=
begin
  use 3^1009,
  norm_num,
end

lemma perfect_square_4 : ∃ x : ℝ, x^2 = 4^2019 :=
begin
  use 2^2019,
  have h : 4 = 2^2 := rfl,
  rw [h, pow_mul],
  norm_num,
end

lemma perfect_square_5 : ∃ x : ℝ, x^2 = 5^2020 :=
begin
  use 5^1010,
  norm_num,
end

theorem main : ∃ x : ℝ, x^2 = 1^2016 ∧ ¬ ∃ x : ℝ, x^2 = 2^2017 ∧ ∃ x : ℝ, x^2 = 3^2018 ∧ ∃ x : ℝ, x^2 = 4^2019 ∧ ∃ x : ℝ, x^2 = 5^2020 :=
by {
  exact ⟨1^1008, by norm_num, not_perfect_square_2, 3^1009, by norm_num, 2^2019, by norm_num, 5^1010, by norm_num⟩,
}

end main_l690_690309


namespace no_two_champion_teams_l690_690526

theorem no_two_champion_teams
  (T : Type) 
  (M : T -> T -> Prop)
  (superior : T -> T -> Prop)
  (champion : T -> Prop)
  (h1 : ∀ A B, M A B ∨ (∃ C, M A C ∧ M C B) → superior A B)
  (h2 : ∀ A, champion A ↔ ∀ B, superior A B)
  (h3 : ∀ A B, M A B ∨ M B A)
  : ¬ ∃ A B, champion A ∧ champion B ∧ A ≠ B := 
sorry

end no_two_champion_teams_l690_690526


namespace intersection_of_A_and_B_l690_690207

open Set

noncomputable def A : Set ℤ := {1, 3, 5, 7}
noncomputable def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end intersection_of_A_and_B_l690_690207


namespace max_value_quadratic_function_l690_690264

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 6*x - 7

theorem max_value_quadratic_function (t : ℝ) :
  (∃ x, t ≤ x ∧ x ≤ t + 2 ∧ ∀ x' ∈ set.Icc t (t + 2), quadratic_function x' ≤ quadratic_function x) ↔ t ≥ 3 :=
by
  sorry

end max_value_quadratic_function_l690_690264


namespace geometric_sequence_sum_15_l690_690681

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)
variable (h : ∀ k, a (k + 1) = a k * (if k = 1 then 2 else -1))
variable (h1 : S 1 = a 1)
variable (h2 : ∀ n, S (n + 1) = S n + a (n + 1))

theorem geometric_sequence_sum_15 
  (h3 : ∀ n, (1 : ℝ) - (1 / a 2) = 2 / a 3)
  : S 4 = 15 :=
sorry

end geometric_sequence_sum_15_l690_690681


namespace sequences_inequality_l690_690571

noncomputable def sequence_an (a0 : ℝ) : ℕ → ℝ
| 0       := a0
| (n + 1) := (sequence_an a0 n + 1) / 2

noncomputable def sequence_bn (b0 : ℝ) (k : ℝ) : ℕ → ℝ
| 0       := b0
| (n + 1) := (sequence_bn b0 k n) ^ k

theorem sequences_inequality (k : ℝ) (a0 b0 : ℝ) (h1 : 0 < k) (h2 : k < 1/2) (h3 : 0 < a0) (h4 : a0 < 1) (h5 : 0 < b0) (h6 : b0 < 1) :
  ∃ N : ℕ, ∀ n ≥ N, sequence_an a0 n < sequence_bn b0 k n :=
sorry

end sequences_inequality_l690_690571


namespace count_two_digit_primes_with_digit_sum_10_l690_690884

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690884


namespace cheese_initial_weight_l690_690292

theorem cheese_initial_weight (initial_weight : ℕ) 
  (take_bites : ℕ → ℕ × ℕ) 
  (final_state : ℕ × ℕ) 
  (final_weight : ℕ) :
  let final_state := (20, 20)
  ∃ initial_weight, take_bites 3 (initial_weight, 0) = final_state ∧ 
  initial_weight = 680 := 
sorry

end cheese_initial_weight_l690_690292


namespace rectangle_cut_max_pieces_l690_690214

open Classical

theorem rectangle_cut_max_pieces (n : ℕ) (square : Type) (rectangles : Fin n → Set square)
  (parallel_sides : ∀ i, (rectangles i).HasParallelSides) (no_shared_internal_points : ∀ i j, i ≠ j → Disjoint (rectangles i) (rectangles j)) :
  ∃ split_pieces : ℕ, split_pieces ≤ n + 1 :=
sorry

end rectangle_cut_max_pieces_l690_690214


namespace total_amount_eq_444_l690_690714

noncomputable def total_value_in_dollars : ℝ :=
  let A := (1 ^ 2 : ℝ) * 100 * 1
  let B := (∛(125 : ℝ)) * 1
  let C := (200 ^ 2 : ℝ) * 1
  let D := (70 ^ (1 / 3) : ℝ) * 1
  let E := (50 * 2 : ℝ) * 1
  let total_paisa := A + B + C + D + E
  let total_rupees := total_paisa / 100
  let total_dollars := total_rupees / 70
  total_dollars

theorem total_amount_eq_444 (C_share_dollars : ℝ) (hC : C_share_dollars = 3) : total_value_in_dollars ≈ 4.44 :=
by
  have : C = 21000
  sorry

end total_amount_eq_444_l690_690714


namespace smallest_positive_period_l690_690401

theorem smallest_positive_period :
  ∀ (x : ℝ), 5 * Real.sin ((π / 6) - (π / 3) * x) = 5 * Real.sin ((π / 6) - (π / 3) * (x + 6)) :=
by
  sorry

end smallest_positive_period_l690_690401


namespace square_cut_and_reassemble_l690_690394

theorem square_cut_and_reassemble (s a : ℝ) (h1 : 0 < s) (h2 : 0 < a) (h3 : s^2 = 6 * a^2) :
  ∃ (pieces : list (set (ℝ × ℝ))), (
    (∀ part ∈ pieces, measurable part) ∧
    (set.disjoint (set.range pieces)) ∧
    (set.Union pieces = set.univ) ∧
    let assembled_squares := set_of (λ (part : set (ℝ × ℝ)), part ∈ pieces) in
    (∃ (assembled_square1 assembled_square2 assembled_square3 : set (ℝ × ℝ)), 
      (assembled_square1 ∈ assembled_squares ∧ assembled_square2 ∈ assembled_squares ∧ assembled_square3 ∈ assembled_squares) ∧
      (assembled_square1 ∪ assembled_square2 ∪ assembled_square3 = set.univ) ∧
      (∀ (i ∈ [assembled_square1, assembled_square2, assembled_square3]), is_square i a))
  ) :=
sorry

def is_square (s : set (ℝ × ℝ)) (a : ℝ) : Prop :=
  ∃ (x y : ℝ), s = {p | (p.1 - x)^2 ≤ a^2 ∧ (p.2 - y)^2 ≤ a^2}

end square_cut_and_reassemble_l690_690394


namespace num_two_digit_primes_with_digit_sum_10_l690_690789

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690789


namespace integral_f_eq_pi_div_2_add_4_div_3_l690_690562

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Ico (-1 : ℝ) 1 then real.sqrt (1 - x^2) 
  else if x ∈ set.Icc (1 : ℝ) 2 then x^2 - 1 
  else 0

theorem integral_f_eq_pi_div_2_add_4_div_3 :
  ∫ x in -1..2, f x = (real.pi / 2) + (4 / 3) :=
by
  sorry

end integral_f_eq_pi_div_2_add_4_div_3_l690_690562


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690970

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690970


namespace train_speed_l690_690013

theorem train_speed (train_length platform_length total_time : ℕ) 
  (h_train_length : train_length = 150) 
  (h_platform_length : platform_length = 250) 
  (h_total_time : total_time = 8) : 
  (train_length + platform_length) / total_time = 50 := 
by
  -- Proof goes here
  -- Given: train_length = 150, platform_length = 250, total_time = 8
  -- We need to prove: (train_length + platform_length) / total_time = 50
  -- So we calculate
  --  (150 + 250)/8 = 400/8 = 50
  sorry

end train_speed_l690_690013


namespace rational_solution_quadratic_l690_690748

theorem rational_solution_quadratic (m : ℕ) (h_pos : m > 0) : 
  (∃ (x : ℚ), x * x * m + 25 * x + m = 0) ↔ m = 10 ∨ m = 12 :=
by sorry

end rational_solution_quadratic_l690_690748


namespace cheese_original_weight_l690_690300

theorem cheese_original_weight (final_weight : ℕ)
    (after_bites : ∀(n: ℕ) (w: ℕ), n ≤ 3 → (n = 0 → w = final_weight) ∧
                  (n = 1 → w = 40) ∧
                  (n = 2 → w = 100 ∧ 60) ∧
                  (n = 3 → w = 260 ∧ 160) ∧
                  (n = 4 → w = 420 ∧ 260)) :
    final_weight = 680 := by
    sorry

end cheese_original_weight_l690_690300


namespace num_two_digit_primes_with_digit_sum_10_l690_690775

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690775


namespace lice_checks_time_in_hours_l690_690270

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l690_690270


namespace harry_travel_time_l690_690488

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l690_690488


namespace hyperbola_focus_distance_l690_690124

def hyperbola (a b x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

variables (F1 F2 P : ℝ × ℝ) (a b d1 d2 : ℝ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem hyperbola_focus_distance
  (h1 : hyperbola 3 4 (P.1) (P.2))
  (h2 : d1 = distance P F1)
  (h3 : d1 = 3) :
  distance P F2 = 9 :=
sorry

end hyperbola_focus_distance_l690_690124


namespace tangent_line_to_circle_l690_690766

open Real

theorem tangent_line_to_circle (x y : ℝ) :
  ((x - 2) ^ 2 + (y + 1) ^ 2 = 9) ∧ ((x = -1) → (x = -1 ∧ y = 3) ∨ (y = (37 - 8*x) / 15)) :=
by {
  sorry
}

end tangent_line_to_circle_l690_690766


namespace count_valid_subsets_l690_690140

open Finset

-- Define the set A excluding the primes
def A : Finset ℕ := {1, 4, 6, 8, 9, 10, 12}

-- Property definitions
def no_two_consecutive (S : Finset ℕ) : Prop :=
  ∀ (x ∈ S) (y ∈ S), x ≠ y + 1 ∧ y ≠ x + 1

def element_condition (S : Finset ℕ) : Prop :=
  ∀ k (Hk : S.card = k), ∀ (x ∈ S), x ≤ 2 * k

-- Main theorem statement
theorem count_valid_subsets : (A.filter (λ S, no_two_consecutive S ∧ element_condition S)).card = 19 := by
  sorry

end count_valid_subsets_l690_690140


namespace harry_total_travel_time_l690_690491

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l690_690491


namespace symmedian_point_exists_l690_690447

theorem symmedian_point_exists (A B C P X Y Z : Point) 
  (h : Triangle A B C)
  (hx : Proj P (Line B C) = X)
  (hy : Proj P (Line C A) = Y)
  (hz : Proj P (Line A B) = Z):
  is_centroid P (Triangle X Y Z) ↔ is_symmedian_point P h :=
sorry

end symmedian_point_exists_l690_690447


namespace no_integral_roots_l690_690230

theorem no_integral_roots :
  ¬(∃ (x : ℤ), 5 * x^2 + 3 = 40) ∧
  ¬(∃ (x : ℤ), (3 * x - 2)^3 = (x - 2)^3 - 27) ∧
  ¬(∃ (x : ℤ), x^2 - 4 = 3 * x - 4) :=
by sorry

end no_integral_roots_l690_690230


namespace sum_of_digits_of_min_n_l690_690191

theorem sum_of_digits_of_min_n (N : ℕ) (hN : N % 3 = 0) (hP : P(N) < 3/4) : N = 6 → (N.digits.sum = 6) := by
  sorry

end sum_of_digits_of_min_n_l690_690191


namespace problem_equivalent_l690_690959

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690959


namespace trapezoid_midpoints_segment_length_l690_690249

theorem trapezoid_midpoints_segment_length 
  (T : Type) [Trapezoid T] (A B C D E H F G : Point T)
  (h_diagonals_perpendicular : Perpendicular (Diagonal A D) (Diagonal B C))
  (h_midline_equal_5 : length (Midline T E H) = 5) :
  length (Segment F G) = 5 :=
sorry

end trapezoid_midpoints_segment_length_l690_690249


namespace ned_pieces_left_l690_690212

def boxes_bought : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_per_box : ℝ := 6.0
def boxes_left (bought : ℝ) (given : ℝ) : ℝ := bought - given
def total_pieces (boxes : ℝ) (pieces_per_box : ℝ) : ℝ := boxes * pieces_per_box

theorem ned_pieces_left : total_pieces (boxes_left boxes_bought boxes_given) pieces_per_box = 42.0 := by
  sorry

end ned_pieces_left_l690_690212


namespace roots_polynomial_value_l690_690196

theorem roots_polynomial_value (r s t : ℝ) (h₁ : r + s + t = 15) (h₂ : r * s + s * t + t * r = 25) (h₃ : r * s * t = 10) :
  (1 + r) * (1 + s) * (1 + t) = 51 :=
by
  sorry

end roots_polynomial_value_l690_690196


namespace total_wet_surface_area_l690_690670

def cistern_length (L : ℝ) := L = 5
def cistern_width (W : ℝ) := W = 4
def water_depth (D : ℝ) := D = 1.25

theorem total_wet_surface_area (L W D A : ℝ) 
  (hL : cistern_length L) 
  (hW : cistern_width W) 
  (hD : water_depth D) :
  A = 42.5 :=
by
  subst hL
  subst hW
  subst hD
  sorry

end total_wet_surface_area_l690_690670


namespace pythagorean_triple_conditions_l690_690228
open Int

def is_pythagorean_triple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

def is_primitive (a b c : ℤ) : Prop :=
  gcd a b = 1 ∧ gcd a c = 1 ∧ gcd b c = 1

noncomputable def satisfies_conditions (a b c : ℤ) : Prop :=
  (is_primitive a b c) ∧
  ((even a ∧ 4 ∣ a) ∨ (even b ∧ 4 ∣ b)) ∧
  (3 ∣ a ∨ 3 ∣ b) ∧
  (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) ∧
  (odd c ∧ (∃ k : ℤ, c = 4 * k + 1) ∧ ¬ (3 ∣ c) ∧ ¬ (7 ∣ c) ∧ ¬ (11 ∣ c))

theorem pythagorean_triple_conditions {a b c : ℤ} (h : is_pythagorean_triple a b c) :
  satisfies_conditions a b c :=
sorry

end pythagorean_triple_conditions_l690_690228


namespace simplify_abs_expression_l690_690596

/-- Simplify the expression: |-4^3 + 5^2 - 6| and prove the result is equal to 45 -/
theorem simplify_abs_expression :
  |(- 4 ^ 3 + 5 ^ 2 - 6)| = 45 :=
by
  sorry

end simplify_abs_expression_l690_690596


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690975

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690975


namespace factorize_x4_y4_l690_690757

theorem factorize_x4_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) :=
by
  sorry

end factorize_x4_y4_l690_690757


namespace minnows_in_each_prize_bowl_l690_690547

theorem minnows_in_each_prize_bowl :
  let total_minnows := 600 in
  let total_people := 800 in
  let prize_winners := total_people * 15 / 100 in
  let minnows_left_over := 240 in
  let minnows_given_away := total_minnows - minnows_left_over in
  let minnows_per_prize := minnows_given_away / prize_winners in
  minnows_per_prize = 3 :=
by
  sorry

end minnows_in_each_prize_bowl_l690_690547


namespace find_angle_D_l690_690096

theorem find_angle_D (A B C D E F : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 40) 
  (triangle_sum1 : A + B + C + E + F = 180) (triangle_sum2 : D + E + F = 180) : 
  D = 125 :=
by
  -- Only adding a comment, proof omitted for the purpose of this task
  sorry

end find_angle_D_l690_690096


namespace problem_equivalent_l690_690969

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690969


namespace guzman_boxes_l690_690210

theorem guzman_boxes (N d : ℕ) (hN : N = 48) (hd : d = 12) : (N / d) = 4 := by
  rw [hN, hd]
  norm_num
  sorry

end guzman_boxes_l690_690210


namespace locus_of_points_l690_690482

-- Define the points O1, O2, M
variables (O1 O2 M : Point)
-- Define the radii r1, r2 of the circles
variables (r1 r2 : ℝ)
-- Define the ratio constant k
variable (k : ℝ)

-- Lengths of tangents from M to the circles
def tangent_length_1 (M O1 : Point) (r1 : ℝ) : ℝ :=
  sqrt ((M - O1).length_squared - r1^2)

def tangent_length_2 (M O2 : Point) (r2 : ℝ) : ℝ :=
  sqrt ((M - O2).length_squared - r2^2)

-- The set of points M such that the ratio of the tangent lengths is k
def satisfies_ratio (M O1 O2 : Point) (r1 r2 k : ℝ) : Prop :=
  tangent_length_1 M O1 r1 / tangent_length_2 M O2 r2 = k

-- Prove that M forms a circle when k != 1 and a straight line when k = 1
theorem locus_of_points (O1 O2 : Point) (r1 r2 k : ℝ) :
  ∀ M : Point, satisfies_ratio M O1 O2 r1 r2 k → 
  (k ≠ 1 → ∃ C : Point, (M - C).length = some_radius_1) ∧ 
  (k = 1 → ∃ L : Line, M ∈ L) :=
by sorry

end locus_of_points_l690_690482


namespace second_discount_correct_l690_690267

noncomputable def second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : ℝ :=
  let first_discount_amount := first_discount / 100 * original_price
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_correct :
  second_discount_percentage 510 12 381.48 = 15 :=
by
  sorry

end second_discount_correct_l690_690267


namespace exists_quadratic_open_upwards_passing_through_origin_l690_690315

-- Define the general form of a quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Constants for the conditions
constants (a b c : ℝ)

-- Conditions
axiom a_pos : a > 0
axiom passes_through_origin : quadratic_function a b c 0 = 1

-- Goal: Prove that under the given conditions, the quadratic function exists (and hence provide an example of such function).
theorem exists_quadratic_open_upwards_passing_through_origin : 
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 :=
by
  use 1, 0, 1
  split
  { exact zero_lt_one }
  { simp [quadratic_function] }

end exists_quadratic_open_upwards_passing_through_origin_l690_690315


namespace problem_equivalent_l690_690956

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690956


namespace future_value_option_B_correct_l690_690594

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end future_value_option_B_correct_l690_690594


namespace largest_domain_value_l690_690195

def g1 (x : ℝ) := sqrt (2 - x)
def gn (n : ℕ) (x : ℝ) : ℝ := if n = 1 then g1 x else gn (n - 1) (sqrt ((n + 1)^2 - x))

theorem largest_domain_value (M d : ℕ) (hM : M = 4) (hd : d = 25) :
  ∀ n, (∀ x, (gn n x).is_real → n ≤ M) ∧ (domain (gn M) = {d}) :=
  by
    sorry

end largest_domain_value_l690_690195


namespace sum_of_segment_lengths_eq_radius_l690_690711

-- Given definitions and assumptions
variables {R : ℝ} {n : ℕ} (A : ℕ → ℝ × ℝ)

-- Conditions
def semicircle (A : ℕ → ℝ × ℝ) (n : ℕ) :=
  A 0 = (-R, 0) ∧ A (2 * n + 1) = (R, 0) ∧
  ∀ i, 0 < i ∧ i < 2 * n + 1 → dist (A 0) (A i) = i * R / (2 * n + 1) ∧ 
  dist (A (2 * n + 1)) (A i) = (2 * n + 1 - i) * R / (2 * n + 1)

def is_center (O : ℝ × ℝ) :=
  O = (0, 0)

-- Main problem statement
theorem sum_of_segment_lengths_eq_radius
  (O : ℝ × ℝ) (A : ℕ → ℝ × ℝ) (n : ℕ)  (R : ℝ) 
  (h1 : semicircle A n) (h2 : is_center O) :
  (finset.range n).sum (λ k, dist (A k.succ) (O) + dist (A (2 * n + 1 - k.succ)) (O)) = R := 
sorry

end sum_of_segment_lengths_eq_radius_l690_690711


namespace part_one_arithmetic_sequence_part_two_general_formula_l690_690446

def a : ℕ → ℝ
| 1       := 0
| (n+1+1) := 1 / (2 - a (n+1))

def seq1 (n : ℕ) : ℝ := 1 / (a n - 1)

theorem part_one_arithmetic_sequence : ∀ n : ℕ, seq1 n - seq1 (n+1) = -1 :=
by
  sorry

theorem part_two_general_formula : ∀ n : ℕ, a n = (n - 1) / n :=
by
  sorry

end part_one_arithmetic_sequence_part_two_general_formula_l690_690446


namespace two_digit_primes_with_digit_sum_ten_l690_690792

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690792


namespace probability_divisible_by_5_in_T_l690_690560

open_locale big_operators

def T := {n : ℕ | 1 ≤ n ∧ n < 2^50 ∧ nat.popcount n = 3}

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def count {α : Type*} (s : set α) := fintype.card s

theorem probability_divisible_by_5_in_T : 
  (∃ (p q : ℕ), p / q = 12 / 137 ∧ nat.coprime p q ∧ (p + q = 149)) :=
sorry

end probability_divisible_by_5_in_T_l690_690560


namespace cos_halfpi_plus_alpha_f_alpha_l690_690450

theorem cos_halfpi_plus_alpha 
  (h1 : ∀ α, 2 * π - α → cos (2 * π - α) = -4/5) 
  (h2 : ∀ α, 3 * π/2 < α ∧ α < 2 * π) :
  ∀ α, cos (π/2 + α) = 3/5 :=
sorry

theorem f_alpha 
  (h1 : ∀ α, 2 * π - α → cos (2 * π - α) = -4/5) 
  (h2 : ∀ α, 3 * π/2 < α ∧ α < 2 * π) :
  ∀ α, (sin α * tan α * cos α) / (-(cos α)) = -9/20 :=
sorry

end cos_halfpi_plus_alpha_f_alpha_l690_690450


namespace ferry_P_travel_time_l690_690083

-- Definitions of conditions
def speed_P : ℝ := 6 -- speed of ferry P in km/h
def speed_diff_PQ : ℝ := 3 -- speed difference between ferry Q and ferry P in km/h
def travel_longer_Q : ℝ := 2 -- ferry Q travels a route twice as long as ferry P
def time_diff_PQ : ℝ := 1 -- time difference between ferry Q and ferry P in hours

-- Distance traveled by ferry P
def distance_P (t_P : ℝ) : ℝ := speed_P * t_P

-- Distance traveled by ferry Q
def distance_Q (t_P : ℝ) : ℝ := travel_longer_Q * (speed_P * t_P)

-- Speed of ferry Q
def speed_Q : ℝ := speed_P + speed_diff_PQ

-- Time taken by ferry Q
def time_Q (t_P : ℝ) : ℝ := t_P + time_diff_PQ

-- Main theorem statement
theorem ferry_P_travel_time (t_P : ℝ) : t_P = 3 :=
by
  have eq_Q : speed_Q * (time_Q t_P) = distance_Q t_P := sorry
  have eq_P : speed_P * t_P = distance_P t_P := sorry
  sorry

end ferry_P_travel_time_l690_690083


namespace valid_distribution_methods_count_l690_690751

/-- 
  There are 4 teachers (A, B, C, and D) and 3 different schools. 
  Each school must receive at least one teacher.
  Teachers A and B cannot be assigned to the same school.
  Prove that the number of different valid distribution methods is 30.
-/
theorem valid_distribution_methods_count :
  ∃ (f : fin 4 → fin 3), 
    (∀ i : fin 3, ∃ j : fin 4, f j = i) ∧ 
    (f 0 ≠ f 1) ∧ 
    (∃! n, ∃! p, ∃! q, ∃ r, f n ≠ f p ∧ f n ≠ f q ∧ f n ≠ f r ∧ f p ≠ f q ∧ f p ≠ f r ∧ f q ≠ f r) 
  :=
begin
  -- By combinatorial arguments and inclusion-exclusion principle.
  have : 3^4 - (3 * 3 + 3 * 3 * (finset.card {4.choose 3}) + 6 * 2) = 30,
  sorry,
end

end valid_distribution_methods_count_l690_690751


namespace circle_equation_and_distances_l690_690171
-- Import necessary libraries

-- Lean statement for the proof problem
theorem circle_equation_and_distances
    (t : ℝ)
    (A B P : Point)
    (x y : ℝ)
    (l_parametric : x = 1 + (sqrt 2 / 2) * t ∧ y = 2 + (sqrt 2 / 2) * t)
    (circle_polar : ∀ θ, ρ θ = 4 * sin θ)
    (point_P: P = (1, 2))
    (intersects_AB : A ∈ l ∧ B ∈ l ∧ A ∈ C ∧ B ∈ C) :
  (x^2 + y^2 - 4*y = 0) ∧ (|PA| + |PB| = sqrt 14) :=
sorry

end circle_equation_and_distances_l690_690171


namespace find_n_l690_690455

theorem find_n (a b c : ℕ) (n : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : n > 2) 
    (h₃ : (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n))) : n = 4 := 
sorry

end find_n_l690_690455


namespace solve_equation_l690_690231

noncomputable def equation := 
  ∀ x : ℝ, 3 - 7 * (1 - (sin x)^2) * (sin x) - 3 * (sin x)^3 = 0

noncomputable def solution1 (k : ℤ) : ℝ := 
  (π / 2) + 2 * k * π

noncomputable def solution2 (k : ℤ) : ℝ := 
  (-1 : ℝ)^k * (π / 6) + k * π

theorem solve_equation : 
  ∀ (x : ℝ), equation x → 
  (∃ k : ℤ, x = solution1 k) ∨ (∃ k : ℤ, x = solution2 k) :=
  sorry

end solve_equation_l690_690231


namespace distance_P1_P2_l690_690131

/-- Type for 3D points for cleaner type annotations later -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Define the points P1 and P2 with their coordinates -/
def P1 : Point3D := { x := -1, y := 3, z := 5 }
def P2 : Point3D := { x := 2, y := 4, z := -3 }

/-- The Euclidean distance between two 3D points -/
def distance (P1 P2 : Point3D) : ℝ :=
  Real.sqrt ((P2.x - P1.x) ^ 2 + (P2.y - P1.y) ^ 2 + (P2.z - P1.z) ^ 2)

/-- Goal: Prove that the distance between points P1 and P2 is sqrt(74) -/
theorem distance_P1_P2 : distance P1 P2 = Real.sqrt 74 := by
  sorry

end distance_P1_P2_l690_690131


namespace ratio_of_distances_l690_690529

theorem ratio_of_distances (M1 M2 m1 m2 : ℝ) (d1 d2 : ℝ) (log_32_62 := real.log 32.62):
  M1 = m1 + 5 * real.log(32.62 / d1) →
  M2 = m2 + 5 * real.log(32.62 / d2) →
  M1 = 1.45 →
  M2 = -5.53 →
  m1 = -1.45 →
  m2 = -0.73 →
  d1 / d2 = 0.0288 :=
by
  intro h1 h2 h3 h4 h5 h6
  -- the proof part follows here
  sorry

end ratio_of_distances_l690_690529


namespace valid_parameterizations_l690_690258

open Matrix

-- Define the line equation as a predicate
def is_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = -3 * p.1 + 4

-- Define what it means to be a valid parameterization of the line
def is_valid_param (v : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  is_on_line v ∧ (∀ t : ℝ, is_on_line (v.1 + t * d.1, v.2 + t * d.2))

-- Define specific parameterizations given in options
def param_A := (0, 4) + t • (1, -3)
def param_B := (4/3, 0) + t • (3, -1)
def param_C := (1, 1) + t • (5, -15)
def param_D := (-1, 7) + t • (-1/3, 1)
def param_E := (-4, -8) + t • (0.1, -0.3)

-- The statement of the problem in Lean 4
theorem valid_parameterizations :
  is_valid_param (0, 4) (1, -3) ∧
  ¬ is_valid_param (4/3, 0) (3, -1) ∧
  ¬ is_valid_param (1, 1) (5, -15) ∧
  ¬ is_valid_param (-1, 7) (-1/3, 1) ∧
  is_valid_param (-4, -8) (0.1, -0.3) := by
  sorry

end valid_parameterizations_l690_690258


namespace determine_exponent_l690_690127

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem determine_exponent (a : ℝ) (hf : power_function a 4 = 8) : power_function (3/2) = power_function a := by
  sorry

end determine_exponent_l690_690127


namespace ethanol_percentage_in_fuel_B_l690_690371

theorem ethanol_percentage_in_fuel_B 
  (tank_capacity : ℕ)
  (fuel_A_vol : ℕ)
  (ethanol_in_A_percentage : ℝ)
  (ethanol_total : ℝ)
  (ethanol_A_vol : ℝ)
  (fuel_B_vol : ℕ)
  (ethanol_B_vol : ℝ)
  (ethanol_B_percentage : ℝ) 
  (h1 : tank_capacity = 204)
  (h2 : fuel_A_vol = 66)
  (h3 : ethanol_in_A_percentage = 0.12)
  (h4 : ethanol_total = 30)
  (h5 : ethanol_A_vol = fuel_A_vol * ethanol_in_A_percentage)
  (h6 : ethanol_B_vol = ethanol_total - ethanol_A_vol)
  (h7 : fuel_B_vol = tank_capacity - fuel_A_vol)
  (h8 : ethanol_B_percentage = (ethanol_B_vol / fuel_B_vol) * 100) :
  ethanol_B_percentage = 16 :=
by sorry

end ethanol_percentage_in_fuel_B_l690_690371


namespace minimal_pieces_required_for_cubes_l690_690648

theorem minimal_pieces_required_for_cubes 
  (e₁ e₂ n₁ n₂ n₃ : ℕ)
  (h₁ : e₁ = 14)
  (h₂ : e₂ = 10)
  (h₃ : n₁ = 13)
  (h₄ : n₂ = 11)
  (h₅ : n₃ = 6)
  (disassembly_possible : ∀ {x y z : ℕ}, x^3 + y^3 = z^3 → n₁^3 + n₂^3 + n₃^3 = 14^3 + 10^3)
  (cutting_constraints : ∀ d : ℕ, (d > 0) → (d ≤ e₁ ∨ d ≤ e₂) → (d ≤ n₁ ∨ d ≤ n₂ ∨ d ≤ n₃) → (d ≤ 6))
  : ∃ minimal_pieces : ℕ, minimal_pieces = 11 := 
sorry

end minimal_pieces_required_for_cubes_l690_690648


namespace cos_identity_l690_690084

theorem cos_identity (α : ℝ) (h : sin (2 * π / 3 + α) = 1 / 3) : cos (5 * π / 6 - α) = -1 / 3 :=
  sorry

end cos_identity_l690_690084


namespace count_two_digit_primes_with_digit_sum_10_l690_690991

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690991


namespace range_of_c_l690_690089

-- Definitions of p and q based on conditions
def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (c > 1 / 2)

-- The theorem states the required condition on c
theorem range_of_c (c : ℝ) (h : c > 0) :
  ¬(p c ∧ q c) ∧ (p c ∨ q c) ↔ (0 < c ∧ c ≤ 1 / 2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l690_690089


namespace december_sales_fraction_l690_690554

theorem december_sales_fraction (A : ℚ) : 
  let sales_jan_to_nov := 11 * A
  let sales_dec := 5 * A
  let total_sales := sales_jan_to_nov + sales_dec
  (sales_dec / total_sales) = 5 / 16 :=
by
  sorry

end december_sales_fraction_l690_690554


namespace sum_squares_diagonals_parallelogram_sum_squares_diagonals_parallelepiped_l690_690672

-- Problem a
theorem sum_squares_diagonals_parallelogram 
  (A B C D : ℝ × ℝ)
  (parallelogram : is_parallelogram A B C D) :
  let AC := dist A C
      BD := dist B D
      AB := dist A B
      BC := dist B C
      CD := dist C D
      DA := dist D A
  in AC^2 + BD^2 = AB^2 + BC^2 + CD^2 + DA^2 := sorry

-- Problem b
theorem sum_squares_diagonals_parallelepiped 
  (A B C D E F G H : ℝ × ℝ × ℝ)
  (parallelepiped : is_parallelepiped A B C D E F G H) :
  let AG := dist A G
      BH := dist B H
      CE := dist C E
      DF := dist D F
      AB := dist A B
      BC := dist B C
      CD := dist C D
      DA := dist D A
      EF := dist E F
      FG := dist F G
      GH := dist G H
      HE := dist H E
      AE := dist A E
      BF := dist B F
      CG := dist C G
      DH := dist D H
  in AG^2 + BH^2 + CE^2 + DF^2 = AB^2 + BC^2 + CD^2 + DA^2 + EF^2 + FG^2 + GH^2 + HE^2 + AE^2 + BF^2 + CG^2 + DH^2 := sorry

end sum_squares_diagonals_parallelogram_sum_squares_diagonals_parallelepiped_l690_690672


namespace two_digit_prime_sum_to_ten_count_l690_690846

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690846


namespace symmetric_point_l690_690771

-- Define the given conditions
def pointP : (ℤ × ℤ) := (3, -2)
def symmetry_line (y : ℤ) := (y = 1)

-- Prove the assertion that point Q is (3, 4)
theorem symmetric_point (x y1 y2 : ℤ) (hx: x = 3) (hy1: y1 = -2) (hy : symmetry_line 1) :
  (x, 2 * 1 - y1) = (3, 4) :=
by
  sorry

end symmetric_point_l690_690771


namespace convert_cylindrical_to_rectangular_l690_690390

-- Definitions of the conversion from cylindrical to rectangular coordinates
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- The given cylindrical coordinates point
def point_cylindrical : ℝ × ℝ × ℝ := (5, Real.pi / 3, 2)

-- The expected rectangular coordinates result
def expected_point_rectangular : ℝ × ℝ × ℝ := (2.5, 5 * Real.sqrt 3 / 2, 2)

-- The theorem statement to prove
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 5 (Real.pi / 3) 2 = expected_point_rectangular :=
by
  sorry

end convert_cylindrical_to_rectangular_l690_690390


namespace scaled_shifted_area_l690_690206

variable (f : ℝ → ℝ)

def region_area (g : ℝ → ℝ) : ℝ := sorry

theorem scaled_shifted_area (h : region_area f = 15) : region_area (λ x, -2 * f (x + 4)) = 30 :=
sorry

end scaled_shifted_area_l690_690206


namespace count_two_digit_prime_with_digit_sum_10_l690_690870

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l690_690870


namespace vertical_axis_of_residual_plot_is_residuals_l690_690535

-- Definition of a residual plot
def is_residual_plot (v_axis : Type) := v_axis = "residuals"

-- Theorem statement
theorem vertical_axis_of_residual_plot_is_residuals (v_axis : Type) (h : is_residual_plot v_axis) : 
  v_axis = "residuals" := 
by 
  sorry

end vertical_axis_of_residual_plot_is_residuals_l690_690535


namespace hyperbola_const_prod_of_slopes_hyperbola_fixed_point_M_l690_690477

noncomputable def hyperbola_equation
  (a b : ℝ) (h : (2^2) / a^2 - (3^2) / b^2 = 1) (θ : Real.pi / 3) 
  (h_asymptote : b / a = Real.sqrt 3) : Prop :=
  a = 1 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1)

theorem hyperbola_const_prod_of_slopes
  (a b : ℝ) (h : (2^2) / a^2 - (3^2) / b^2 = 1) (θ : Real.pi / 3) 
  (h_asymptote : b / a = Real.sqrt 3) 
  (orig_line : ∀ t : ℝ, slope (0, 0) (t, t * Real.sqrt 3) * slope (0, 0) (-t, -t * Real.sqrt 3) = 3) : 
  ∀ P A B : (ℝ × ℝ), 
  P ≠ A ∧ P ≠ B ∧ A = (x, 3 * (x - 2)) ∧ B = (x, 3 * -(x-2)) → 
  (slope P A) * (slope P B) = 3 := 
sorry

theorem hyperbola_fixed_point_M
  (a b : ℝ) (h : (2^2) / a^2 - (3^2) / b^2 = 1) (θ : Real.pi / 3) 
  (h_asymptote : b / a = Real.sqrt 3) 
  (right_focus : ℝ × ℝ := (2, 0))
  (rotating_line : ∀ k : ℝ, ∀ x1 x2 : ℝ, combine_eq k (x1, k(x1 - 2)) (x2, k(x2 - 2)) = 0)
  (fixed_point : ∃ M : ℝ × ℝ, M = (-1, 0) ∧ (MA_vector M A) • (MB_vector M B) = 0) : 
  ∃ M : ℝ × ℝ, M = (-1, 0) := 
sorry

end hyperbola_const_prod_of_slopes_hyperbola_fixed_point_M_l690_690477


namespace percentage_black_circles_l690_690575

noncomputable def circle_radius : ℕ → ℝ
| 0 => 3
| n + 1 => circle_radius n + 3

def is_black_circle : ℕ → Prop
| 0 => false
| n + 1 => bnot (is_black_circle n)

theorem percentage_black_circles :
  let areas : List ℝ := List.map (fun n => Real.pi * (circle_radius n ^ 2)) [0, 1, 2, 3, 4]
  let total_area := List.foldr (+) 0 areas
  let black_areas := List.foldr (+) 0 (List.map (fun n => if is_black_circle n then areas.get! n else 0) [0, 1, 2, 3, 4])
  (black_areas / total_area) * 100 = 40 := by
    sorry

end percentage_black_circles_l690_690575


namespace find_other_solution_l690_690453

theorem find_other_solution 
  (x : ℚ)
  (h1 : 63 * x^2 + 36 = 100 * x - 9)
  (h2 : x = 5/7) : 
  ∃ y : ℚ, y ≠ 5/7 ∧ 63 * y^2 + 36 = 100 * y - 9 := 
begin
  use 1,
  split,
  { exact ne_of_gt (by norm_num), },
  { norm_num, ring_nf, exact dec_trivial, }
end

end find_other_solution_l690_690453


namespace inequality_condition_is_sufficient_l690_690265

-- Definitions from conditions
def OptionA : Prop := NecessaryCondition
def OptionB : Prop := SufficientCondition
def OptionC : Prop := NecessaryAndSufficientCondition
def OptionD : Prop := NecessaryOrSufficientCondition

-- Lean statement representing the proof problem
theorem inequality_condition_is_sufficient :
  (∃ cond : Prop, cond = OptionB) :=
sorry

end inequality_condition_is_sufficient_l690_690265


namespace percentage_difference_approx_l690_690314

noncomputable def gabrielle_birds := 7 + 5 + 4 + 3 + 6
noncomputable def chase_birds := 4 + 4 + 3 + 2 + 1
noncomputable def bird_difference := gabrielle_birds - chase_birds
noncomputable def bird_percentage_difference := (bird_difference.toFloat / chase_birds.toFloat) * 100

theorem percentage_difference_approx : bird_percentage_difference ≈ 78.57 := by
  sorry

end percentage_difference_approx_l690_690314


namespace odd_function_a_eq_one_l690_690514

-- Given function definition with piecewise components
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + a * x
  else -x^2 + x

-- Condition for odd function: f(-x) = -f(x)
theorem odd_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f(a) (-x) = -f(a) x) → a = 1 :=
by
  intros h
  let x := -1
  have h1: f(a) (-x) = -x^2 - x := by -- Calculation for x < 0
    sorry
  have h2: f(a) x = x^2 + a * x := by -- Calculation for x < 0
    sorry
  have h3: - f(a) x = - (x^2 + a * x) := by -- -f(x) calculation
    rw h2
  have eqn := h x
  rw h1 at eqn
  rw h3 at eqn
  have h4: -x - a * x = -x := by
    sorry
  have := eqn.trans (Eq.symm h4)
  sorry

end odd_function_a_eq_one_l690_690514


namespace charlotte_avg_speed_l690_690022

def distance : ℕ := 60  -- distance in miles
def time : ℕ := 6       -- time in hours

theorem charlotte_avg_speed : (distance / time) = 10 := by
  sorry

end charlotte_avg_speed_l690_690022


namespace cheese_initial_weight_l690_690294

theorem cheese_initial_weight (initial_weight : ℕ) 
  (take_bites : ℕ → ℕ × ℕ) 
  (final_state : ℕ × ℕ) 
  (final_weight : ℕ) :
  let final_state := (20, 20)
  ∃ initial_weight, take_bites 3 (initial_weight, 0) = final_state ∧ 
  initial_weight = 680 := 
sorry

end cheese_initial_weight_l690_690294


namespace triangle_is_right_l690_690734

-- Based on the conditions, we need the distances between points A, B, and C 
variables (A B C : Type)
variables [metric_space A] [metric_space B] [metric_space C]

def distance (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  (dist A B = 5) ∧ (dist B C = 13) ∧ (dist A C = 12)

theorem triangle_is_right (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :
  distance A B C → ∃ (θ : ℝ), θ = 90 :=
by
  sorry

end triangle_is_right_l690_690734


namespace domain_of_function_l690_690250

theorem domain_of_function :
  (∀ x : ℝ, (2 < x ∧ x < 3) ∨ (x > 3) → ∃y, y = 1 / Real.log x ∧ 2.log (x - 2) ≠ 0) → true :=
begin
  sorry
end

end domain_of_function_l690_690250


namespace simplify_fraction_l690_690598

theorem simplify_fraction (a b : ℕ) (ha : a = 216) (hb : b = 4536) : (a / Nat.gcd a b) / (b / Nat.gcd a b) = 1 / 21 := 
by
  rw [ha, hb]
  have hGCD : Nat.gcd 216 4536 = 216 := by
    sorry
  rw [hGCD]
  norm_num

end simplify_fraction_l690_690598


namespace mira_weekly_distance_l690_690209

noncomputable def total_distance_jogging : ℝ :=
  let monday_distance := 4 * 2
  let thursday_distance := 5 * 1.5
  monday_distance + thursday_distance

noncomputable def total_distance_swimming : ℝ :=
  2 * 1

noncomputable def total_distance_cycling : ℝ :=
  12 * 1

noncomputable def total_distance : ℝ :=
  total_distance_jogging + total_distance_swimming + total_distance_cycling

theorem mira_weekly_distance : total_distance = 29.5 := by
  unfold total_distance
  unfold total_distance_jogging
  unfold total_distance_swimming
  unfold total_distance_cycling
  sorry

end mira_weekly_distance_l690_690209


namespace solve_for_x_l690_690437

theorem solve_for_x (x : ℚ) : ((1/3 - x) ^ 2 = 4) → (x = -5/3 ∨ x = 7/3) :=
by
  sorry

end solve_for_x_l690_690437


namespace tg_15_degrees_l690_690257

theorem tg_15_degrees (x : ℝ) (A B C D F : EuclideanSpace ℝ (Fin 2))
  (h_eq_triangle : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = x)
  (h_height : ∃ P, P ∈ line_through B C ∧ dist B P = dist P D ∧ angle A B P = 90)
  (h_extension : dist B F = x)
  (h_connect : angle F D C = 90) :
  Real.tan (15 * (pi / 180)) = 2 - Real.sqrt 3 := 
  sorry

end tg_15_degrees_l690_690257


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l690_690345

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l690_690345


namespace two_digit_primes_with_digit_sum_ten_l690_690795

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l690_690795


namespace gas_cycle_work_done_l690_690677

noncomputable def p0 : ℝ := 10^5
noncomputable def V0 : ℝ := 1

theorem gas_cycle_work_done :
  (3 * Real.pi * p0 * V0 = 942) :=
by
  have h1 : p0 = 10^5 := by rfl
  have h2 : V0 = 1 := by rfl
  sorry

end gas_cycle_work_done_l690_690677


namespace x_intercept_of_line_l690_690419

-- Definition of line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Proposition that the x-intercept of the line 4x + 7y = 28 is (7, 0)
theorem x_intercept_of_line : line_eq 7 0 :=
by
  show 4 * 7 + 7 * 0 = 28
  sorry

end x_intercept_of_line_l690_690419


namespace count_prime_two_digit_sum_ten_is_three_l690_690805

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l690_690805


namespace isha_original_length_l690_690180

variable (current_length sharpened_off : ℕ)

-- Condition 1: Isha's pencil is now 14 inches long
def isha_current_length : current_length = 14 := sorry

-- Condition 2: She sharpened off 17 inches of her pencil
def isha_sharpened_off : sharpened_off = 17 := sorry

-- Statement to prove:
theorem isha_original_length (current_length sharpened_off : ℕ) 
  (h1 : current_length = 14) (h2 : sharpened_off = 17) :
  current_length + sharpened_off = 31 :=
by
  sorry

end isha_original_length_l690_690180


namespace number_of_integer_solutions_l690_690498

theorem number_of_integer_solutions :
  let f : ℤ → ℤ := λ x, (x - 3)^(16 - x^2) in
  (∀ a : ℤ, a^0 = 1) ∧
  (∀ b : ℤ, 1^b = 1) ∧
  (∀ c : ℤ, even c → (-1)^c = 1) →
  ∃ (solutions : Finset ℤ), 
    (∀ x ∈ solutions, f x = 1) ∧
    solutions.card = 3 := 
by
  -- Proof goes here
  sorry

end number_of_integer_solutions_l690_690498


namespace count_two_digit_primes_with_digit_sum_10_l690_690861

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690861


namespace correct_statement_A_l690_690312

theorem correct_statement_A : 
  (∀ (n : ℕ), n ≥ 3 → ∑ k in range n, 360 / n = 360) :=
sorry

end correct_statement_A_l690_690312


namespace partnership_investment_l690_690017

theorem partnership_investment (A B C : ℕ) (x m : ℝ) 
    (H1 : B = 2 * A) 
    (H2 : C = 3 * A) 
    (total_annual_gain A_share : ℝ) 
    (H3 : total_annual_gain = 21000) 
    (H4 : A_share = 7000) 
    (investment_ratio : (A * 12) / (A * 12 + B * 6 + C * (12 - m)) = 1 / 3) :
  m = 8 :=
sorry

end partnership_investment_l690_690017


namespace two_digit_prime_sum_to_ten_count_l690_690842

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690842


namespace tenth_term_of_geometric_sequence_l690_690739

theorem tenth_term_of_geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) (tenth_term : ℚ) :
  a = 5 →
  r = 4 / 3 →
  n = 10 →
  tenth_term = a * r ^ (n - 1) →
  tenth_term = 1310720 / 19683 :=
by sorry

end tenth_term_of_geometric_sequence_l690_690739


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690974

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690974


namespace problem_equivalent_l690_690967

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690967


namespace lean_statement_l690_690002

variables {A B C D I I1 I2 : Point}
variables {r r1 r2 : Real}
variables {S1 S2 : Circle}
variables {φ : Real}

-- Assuming existence of circles and tangencies along with necessary centers and radii
axiom tangent_to_segments_and_circumscribed {S1 S2 : Circle} (D : Point) (S1_tangent_BD: ∀ P, P ∈ Circle.tangent_points S1 -> P ∈ Line BD)
  (S2_tangent_CD: ∀ Q, Q ∈ Circle.tangent_points S2 -> Q ∈ Line CD) (I_center: Center S = I) (I1_center: Center S1 = I1) (I2_center: Center S2 = I2)
  (r_radius: radius S = r) (r1_radius: radius S1 = r1) (r2_radius: radius S2 = r2) : Prop := sorry

axiom angle_def {ABC: Triangle} (φ_def: φ = ∠ ADB) : Prop := sorry

-- Lean statement for the proof problem
theorem lean_statement 
  (ABC: Triangle) (D_on_BC: D ∈ (segment BC))
  (S_tangent : tangent_to_segments_and_circumscribed D (A : Point) (B : Point) (C : Point) (I : Point) (I1 : Point) (I2 : Point) (r : Real) (r1 : Real) (r2 : Real))
  (φ_angle : angle_def (ABC : Triangle) (φ_def : Real)) :
  (I ∈ (segment I1 I2)) ∧ (dist I1 I / dist I I2 = tan²(φ / 2)) ∧ (r = r1 * cos²(φ / 2) + r2 * sin²(φ / 2)) :=
sorry

end lean_statement_l690_690002


namespace part1_expression_for_f_part2_three_solutions_l690_690092

noncomputable def f1 (x : ℝ) := x^2

noncomputable def f2 (x : ℝ) := 8 / x

noncomputable def f (x : ℝ) := f1 x + f2 x

theorem part1_expression_for_f : ∀ x:ℝ, f x = x^2 + 8 / x := by
  sorry  -- This is where the proof would go

theorem part2_three_solutions (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, f x1 = f a ∧ f x2 = f a ∧ f x3 = f a ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 := by
  sorry  -- This is where the proof would go

end part1_expression_for_f_part2_three_solutions_l690_690092


namespace find_a_l690_690541

-- Define the setting of the problem
def total_area := 6
def half_area := total_area / 2
def endpoint1 := (a : ℝ, 0 : ℝ)
def endpoint2 := (5 : ℝ, 2 : ℝ)
def line_splits_area_equally (a : ℝ) : Prop := 
  let slope := (2 - 0) / (5 - a) in
  let y := (λ x : ℝ, slope * (x - a)) in
  let base := 5 - a in
  let triangle_area := 1 / 2 * base * 2 in
  triangle_area = half_area

-- State the theorem to prove
theorem find_a : ∃ a : ℝ, line_splits_area_equally a ∧ a = 2 := 
  by
    sorry

end find_a_l690_690541


namespace bus_dispatch_interval_l690_690217

-- Variables representing the speeds of Xiao Nan and the bus
variable (V_1 V_2 : ℝ)
-- The interval between the dispatch of two buses
variable (interval : ℝ)

-- Stating the conditions in Lean

-- Xiao Nan notices a bus catches up with him every 10 minutes
def cond1 : Prop := ∃ s, s = 10 * (V_1 - V_2)

-- Xiao Yu notices he encounters a bus every 5 minutes
def cond2 : Prop := ∃ s, s = 5 * (V_1 + 3 * V_2)

-- Proof statement
theorem bus_dispatch_interval (h1 : cond1 V_1 V_2) (h2 : cond2 V_1 V_2) : interval = 8 := by
  -- Proof would be provided here
  sorry

end bus_dispatch_interval_l690_690217


namespace convert_cylindrical_to_rectangular_l690_690389

noncomputable theory

open Real

-- Define the cylindrical coordinates
def r : ℝ := 5
def theta : ℝ := π / 3
def z_cylindrical : ℝ := 2

-- Define the expected rectangular coordinates
def x_rect : ℝ := 2.5
def y_rect : ℝ := 5 * sqrt(3) / 2
def z_rect : ℝ := 2

-- Lean 4 statement to verify conversion
theorem convert_cylindrical_to_rectangular
  (r θ z_cylindrical x_rect y_rect z_rect : ℝ)
  (hr : r = 5) (htheta : θ = π / 3) (hz : z_cylindrical = 2)
  (hx : x_rect = 5 * cos (π / 3)) (hy : y_rect = 5 * sin (π / 3)) (hz_rect : z_rect = z_cylindrical) :
  (x_rect, y_rect, z_rect) = (2.5, 5 * sqrt(3) / 2, 2) :=
by { 
  rw [hr, htheta, hz] at *,
  rw cos_pi_div_three at hx, 
  rw sin_pi_div_three at hy,
  exact ⟨hx, hy, hz_rect⟩,
  sorry
}

end convert_cylindrical_to_rectangular_l690_690389


namespace angle_ACB_of_A_parallel_OG_l690_690189

theorem angle_ACB_of_A_parallel_OG : ∀ (A B C D : Point)
  (O : Circle) (G : Point)
  (h_triangle : Triangle ABC)
  (h_center : IsCircumcenter O ABC)
  (h_centroid : IsCentroid G ABC)
  (h_cd_eq_ac : CD = AC)
  (h_cin_bd : C_between B D)
  (h_ad_parallel_og : Parallel AD (\(x : Line) → EulerLine O G))
  (h_angle_bac : ∠ BAC = 40°)
  : ∠ ACB = 70° :=
sorry

end angle_ACB_of_A_parallel_OG_l690_690189


namespace equilateral_triangle_perimeter_l690_690174

theorem equilateral_triangle_perimeter (a : ℝ) (h : a = 10) :
  let side := a in
  let perimeter := 3 * side in
  perimeter = 30 :=
by
  -- Define the side of the triangle
  let side := a
  -- Define the perimeter of the triangle
  let perimeter := 3 * side
  -- State the goal to be proved
  have : perimeter = 30 := sorry
  -- Conclude the proof
  exact this

end equilateral_triangle_perimeter_l690_690174


namespace problem_equivalent_l690_690957

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l690_690957


namespace num_two_digit_primes_with_digit_sum_10_l690_690829

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690829


namespace find_roots_cos_sin_l690_690057

theorem find_roots_cos_sin (x : ℝ) (h1 : x ∈ set.Ioo (-π / 2) 0)
  (h2 : 1 / (Real.cos x)^3 - 1 / (Real.sin x)^3 = 4 * Real.sqrt 2) :
  x = -π / 4 :=
sorry

end find_roots_cos_sin_l690_690057


namespace original_weight_of_cheese_l690_690297

theorem original_weight_of_cheese (initial_pieces final_piece_weight : ℕ) 
  (h1 : initial_pieces = (420, 260))
  (h2 : final_piece_weight = 20)
  (h3 : ∀ n ≥ 0, n ≤ 3 → n * final_piece_weight = initial_pieces.2 ∨ n * final_piece_weight = initial_pieces.1):
  initial_pieces.1 + initial_pieces.2 = 680 :=
by
  apply eq.refl 680

end original_weight_of_cheese_l690_690297


namespace num_two_digit_primes_with_digit_sum_10_l690_690827

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l690_690827


namespace distinct_permutations_mathematics_l690_690747

theorem distinct_permutations_mathematics : 
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  (n.factorial / (freqM.factorial * freqA.factorial * freqT.factorial)) = 4989600 :=
by
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  sorry

end distinct_permutations_mathematics_l690_690747


namespace point_on_line_and_equidistant_l690_690125

theorem point_on_line_and_equidistant {x y : ℝ} :
  (4 * x + 3 * y = 12) ∧ (x = y) ∧ (x ≥ 0) ∧ (y ≥ 0) ↔ x = 12 / 7 ∧ y = 12 / 7 :=
by
  sorry

end point_on_line_and_equidistant_l690_690125


namespace find_g_inv_f_of_minus_5_l690_690724

variables {α β : Type} [Inhabited α] [Inhabited β]

-- Definitions from the conditions
variable (f : α → β)
variable (g : β → α)
hypothesis (h : ∀ x, f (g (x)) = 4 * x - 2 )

-- Definition for demonstrating the problem, i.e., question == correct answer
theorem find_g_inv_f_of_minus_5 : g (f (-5)) = -5 :=
sorry

end find_g_inv_f_of_minus_5_l690_690724


namespace no_such_n_exists_l690_690398

theorem no_such_n_exists (n : ℕ) (h1 : n > 0) (fl_sqrt_n := int.floor (int.sqrt n))
  : ¬ ∃ k : ℤ, (n^2 + 1) / (fl_sqrt_n^2 + 2) = k :=
by
  sorry

end no_such_n_exists_l690_690398


namespace sum_of_center_coordinates_l690_690039

theorem sum_of_center_coordinates : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 = 6*x - 10*y + 24) -> 
  (∃ (cx cy : ℝ), (x^2 - 6*x + y^2 + 10*y = (cx - 3)^2 + (cy + 5)^2 + 58) ∧ (cx + cy = -2)) :=
  sorry

end sum_of_center_coordinates_l690_690039


namespace equilateral_triangle_area_increase_l690_690020

theorem equilateral_triangle_area_increase (A : ℝ) (k : ℝ) (s : ℝ) (s' : ℝ) (A' : ℝ) (ΔA : ℝ) :
  A = 36 * Real.sqrt 3 →
  A = (Real.sqrt 3 / 4) * s^2 →
  s' = s + 3 →
  A' = (Real.sqrt 3 / 4) * s'^2 →
  ΔA = A' - A →
  ΔA = 20.25 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_increase_l690_690020


namespace arithmetic_sequence_a1_d_l690_690459

theorem arithmetic_sequence_a1_d (a_1 a_2 a_3 a_5 d : ℤ)
  (h1 : a_5 = a_1 + 4 * d)
  (h2 : a_1 + a_2 + a_3 = 3)
  (h3 : a_2 = a_1 + d)
  (h4 : a_3 = a_1 + 2 * d) :
  a_1 = -2 ∧ d = 3 :=
by
  have h_a2 : a_2 = 1 := sorry
  have h_a5 : a_5 = 10 := sorry
  have h_d : d = 3 := sorry
  have h_a1 : a_1 = -2 := sorry
  exact ⟨h_a1, h_d⟩

end arithmetic_sequence_a1_d_l690_690459


namespace angle_rotation_l690_690259

theorem angle_rotation (initial_angle : ℝ) (rotation : ℝ) :
  initial_angle = 30 → rotation = 450 → 
  ∃ (new_angle : ℝ), new_angle = 60 :=
by
  sorry

end angle_rotation_l690_690259


namespace harry_travel_time_l690_690489

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l690_690489


namespace surface_area_of_solid_l690_690758

-- Define the setup based on the provided conditions
structure SolidArrangement where
  bottom_row : Fin 5 → Bool
  middle_column : Fin 5 → Bool
  left_columns : Fin 2 → Fin 3 → Bool

-- Given the specific arrangement, prove the total surface area
theorem surface_area_of_solid : 
  ∀ (arr : SolidArrangement),
  (∀ i, arr.bottom_row i = true) ∧
  (∀ i, arr.middle_column i = true) ∧
  (∀ i j, arr.left_columns i j = true) →
  calculate_surface_area arr = 34 := 
by
  sorry -- Proof goes here, but it's omitted

end surface_area_of_solid_l690_690758


namespace correct_transformation_of_95_sq_l690_690666

theorem correct_transformation_of_95_sq : 95^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end correct_transformation_of_95_sq_l690_690666


namespace find_reflection_sum_l690_690168

variable (A B C D E : ℤ × ℤ)
variable (x y : ℤ)

-- Define the given points
def A : ℤ × ℤ := (-4, 0)
def B : ℤ × ℤ := (0, -3)
def C : ℤ × ℤ := (6, 2)

-- Vertex D of parallelogram ABCD
def D := (x, y)

-- Midpoint of diagonal AC and midpoint of diagonal BD
def midpoint (p1 p2 : ℤ × ℤ) : ℤ × ℤ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Reflection of a point P over point A
def reflection (A P : ℤ × ℤ) : ℤ × ℤ := (2 * A.1 - P.1, 2 * A.2 - P.2)

-- Midpoints are equal
axiom midpoints_eq (M : ℤ × ℤ) : 
  midpoint A C = M ∧ midpoint B D = M

-- Coordinates of point D derived from midpoints_eq
axiom D_coords : D = (2, 5)

-- Calculate the reflection point E of point D over A
def E := reflection A D

-- Sum of coordinates of point E
def sum_coords (p : ℤ × ℤ) : ℤ := p.1 + p.2

-- The goal
theorem find_reflection_sum : sum_coords E = -15 :=
sorry

end find_reflection_sum_l690_690168


namespace prob_exactly_two_teams_chosen_prob_dist_and_expected_value_of_xi_l690_690580

-- Definitions and Conditions for Part (1)
def teams : Finset ℕ := {1, 2, 3, 4}

-- Question (1): Probability that exactly two teams are chosen
theorem prob_exactly_two_teams_chosen 
  (choices : Finset (Finset (Fin 4))) 
  (h1 : ∀ c ∈ choices, c ⊆ teams ∧ c.card = 3) 
  (h_eq : ∀ c ∈ choices, (1:ℝ)/4):
  (prob_two_teams : ℝ) :=
sorry

-- Conditions and definitions for Part (2)
def prob_female_german : ℝ := 1 / 3
def prob_male_german : ℝ := 2 / 5

variable (xi : Fin 4 → ℕ)

-- Question (2): Probability distribution and expected value of ξ
theorem prob_dist_and_expected_value_of_xi
  (P0 : ∃ p, p = (2/3) * (3/5) * (3/5) = 6/25)
  (P1 : ∃ p, p = (1/3) * (3/5) * (3/5) + (2/3) * 2 * (2/5) * (3/5) = 11/25)
  (P2 : ∃ p, p = (1/3) * 2 * (2/5) * (3/5) + (2/3) * (3/5) * (3/5) = 4/15)
  (P3 : ∃ p, p = (1/3) * (2/5) * (2/5) = 4/75)
  (E_xi : ∑ i in {0, 1, 2, 3}, i * prob_dist_and_expected_value_of_xi i = 17/15)
  : Proposition :=
sorry

end prob_exactly_two_teams_chosen_prob_dist_and_expected_value_of_xi_l690_690580


namespace new_area_approx_l690_690624

-- Conditions
def width := 8   -- Width of the bathroom in feet
def length := 12 -- Length of the bathroom in feet
def original_area := 96 -- Original area of the bathroom in sq ft

-- Extensions
def rectangle_extension_length := 5 -- Length of the rectangular extension in feet
def semi_circle_diameter := 2 -- Diameter of the semi-circular extension in feet

-- Calculated/assumed values
def pi_approx := 3.14159 -- Approximate value of π
def radius := semi_circle_diameter / 2 -- Radius of the semi-circle in feet
def area_of_rectangular_extension := rectangle_extension_length * width -- Area of the rectangular extension
def area_of_semi_circle := (pi_approx * radius^2) / 2 -- Area of the semi-circular extension

-- New area calculation
def new_area := original_area + area_of_rectangular_extension + area_of_semi_circle

-- Proof that new area is approximately 137.57 sq ft
theorem new_area_approx : new_area ≈ 137.57 := by 
  sorry -- Proof placeholder

end new_area_approx_l690_690624


namespace two_digit_prime_sum_to_ten_count_l690_690848

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l690_690848


namespace num_two_digit_primes_with_digit_sum_10_l690_690779

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l690_690779


namespace barbara_scores_mean_and_above_90_l690_690071

theorem barbara_scores_mean_and_above_90 (
  scores : List ℕ,   -- Given list of scores
  alan_scores : Fin 5 → ℕ, -- Alan's scores
  alan_mean : ℕ -- Alan's mean score
) (h_scores : scores = [80, 82, 85, 86, 90, 92, 95])
  (h_alan_scores : ∀ i, alan_scores i ∈ scores)
  (h_alan_mean : 5 * alan_mean = List.sum (List.of_fn alan_scores))
  (h_alan_mean_value : alan_mean = 87) :
  let barbara_scores := scores.filter (λ s, ∀ i, s ≠ alan_scores i)
  in List.sum barbara_scores = 175 ∧
     (List.sum barbara_scores / 2 = 87.5) ∧
     (List.countp (λ s, s > 90) barbara_scores = 0) :=
by
  sorry

end barbara_scores_mean_and_above_90_l690_690071


namespace count_two_digit_primes_with_digit_sum_10_l690_690851

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690851


namespace correct_average_weight_l690_690242

open Real

theorem correct_average_weight (n : ℕ) (initial_avg : ℝ) (misread_weights : ℝ × ℝ × ℝ) (correct_weights : ℝ × ℝ × ℝ) :
  n = 60 →
  initial_avg = 70.5 →
  misread_weights = (72, 68, 80) →
  correct_weights = (75, 70, 85) →
  (let total_initial_weight := n * initial_avg in
   let misread_sum := misread_weights.1 + misread_weights.2 + misread_weights.3 in
   let correct_sum := correct_weights.1 + correct_weights.2 + correct_weights.3 in
   let correct_total_weight := total_initial_weight - misread_sum + correct_sum in
   let correct_avg := correct_total_weight / n in
   correct_avg) = 70.67 :=
by
  intros n_eq avg_eq misread_eq correct_eq
  let total_initial_weight := n * initial_avg
  let misread_sum := misread_weights.1 + misread_weights.2 + misread_weights.3
  let correct_sum := correct_weights.1 + correct_weights.2 + correct_weights.3
  let correct_total_weight := total_initial_weight - misread_sum + correct_sum
  let correct_avg := correct_total_weight / n
  sorry

end correct_average_weight_l690_690242


namespace son_l690_690671

theorem son's_age (S F : ℕ) (h1 : F = S + 26) (h2 : F + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_l690_690671


namespace ratio_comparison_l690_690601

theorem ratio_comparison (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_lt_n : m < n) :
  (m + 3) / (n + 3) > m / n :=
sorry

end ratio_comparison_l690_690601


namespace count_two_digit_primes_with_digit_sum_10_is_4_l690_690979

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l690_690979


namespace teal_total_sales_l690_690532

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l690_690532


namespace cross_sectional_area_l690_690006

-- Conditions
def edge_length : ℕ := 2

def is_regular_dodecahedron (P : Type) [Set P] :=
RegularDodecahedron P  -- Assume we have a predicate defining a regular dodecahedron

def is_parallel_plane (pl : Type) [Set pl] (P : Type) [Set P] :=
PlaneParallelToOppositeFaces pl P  -- Assume we have a predicate specifying planes parallel to opposite faces

-- The proof problem to state
theorem cross_sectional_area (P : Type) [Set P] (pl : Type) [Set pl]
  (h_dodeca : is_regular_dodecahedron P)
  (h_plane : is_parallel_plane pl P) :
  cross_sectional_area P pl = 65 := by
  sorry

end cross_sectional_area_l690_690006


namespace alyosha_vitia_arrive_simultaneously_at_M_l690_690720

-- Definitions based on given conditions
def distanceNM : ℝ := 20
def alyoshaWalkSpeed : ℝ := 4
def alyoshaCycleSpeed : ℝ := 15
def vitiaWalkSpeed : ℝ := 5
def vitiaCycleSpeed : ℝ := 20

-- Definition of travel time calculations
def alyoshaTravelTime (x : ℝ) : ℝ := (x / alyoshaCycleSpeed) + ((distanceNM - x) / alyoshaWalkSpeed)
def vitiaTravelTime (x : ℝ) : ℝ := (x / vitiaWalkSpeed) + ((distanceNM - x) / vitiaCycleSpeed)

-- Lean statement of the proof problem
theorem alyosha_vitia_arrive_simultaneously_at_M : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ distanceNM ∧ alyoshaTravelTime x = vitiaTravelTime x :=
begin
  use 12,
  split,
  { linarith, },
  split,
  { linarith, },
  -- The main equation to prove
  { sorry }
end

end alyosha_vitia_arrive_simultaneously_at_M_l690_690720


namespace net_percentage_change_l690_690622

variable (P : ℝ) -- Initial price of the stock

def decrease_last_year (P : ℝ) : ℝ := P * (1 - 0.08)
def increase_this_year (new_P : ℝ) : ℝ := new_P * (1 + 0.10)

theorem net_percentage_change (P : ℝ) :
  let new_P := decrease_last_year P in
  let final_P := increase_this_year new_P in
  ((final_P - P) / P) * 100 = 1.2 :=
by
  sorry

end net_percentage_change_l690_690622


namespace quad_eq_pos_neg_root_l690_690261

theorem quad_eq_pos_neg_root (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 2 ∧ x₁ * x₂ = a + 1) ↔ a < -1 :=
by sorry

end quad_eq_pos_neg_root_l690_690261


namespace simplify_expression_l690_690731

variable {R : Type*} [CommRing R] (x y : R)

theorem simplify_expression :
  (x - 2 * y) * (x + 2 * y) - x * (x - y) = -4 * y ^ 2 + x * y :=
by
  sorry

end simplify_expression_l690_690731


namespace problem1_problem2_problem3_l690_690468

-- Defining the function f and its derivative f'
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c
def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x + b

-- Problem 1: Relationship between a and b
theorem problem1 (a b : ℝ) (x : ℝ) (h : f' x a b = 0) : a^2 ≥ 3 * b :=
sorry

-- Problem 2: Values of a and b given extreme values at x = -1 and x = 3
theorem problem2 (a b : ℝ)
  (h1 : f' (-1) a b = 0)
  (h2 : f' 3 a b = 0) :
  a = 3 ∧ b = -9 :=
sorry

-- Problem 3: Range of values for c
theorem problem3 (c : ℝ)
  (h : ∀ x ∈ Icc (-2 : ℝ) 6, f x 3 (-9) c < 2 * c) :
  54 < c :=
sorry

end problem1_problem2_problem3_l690_690468


namespace x_squared_minus_y_squared_l690_690518

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end x_squared_minus_y_squared_l690_690518


namespace power_function_evaluation_l690_690158

theorem power_function_evaluation : 
  (∃ a : ℝ, (2 : ℝ)^a = √2) → (8 : ℝ)^(1 / 2) = 2 * √2 :=
by
  sorry

end power_function_evaluation_l690_690158


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690911

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690911


namespace probability_of_non_yellow_is_19_over_28_l690_690686

def total_beans : ℕ := 4 + 5 + 9 + 10

def non_yellow_beans : ℕ := 4 + 5 + 10

def probability_not_yellow : ℚ := non_yellow_beans / total_beans

theorem probability_of_non_yellow_is_19_over_28 :
  probability_not_yellow = 19 / 28 := by
  sorry

end probability_of_non_yellow_is_19_over_28_l690_690686


namespace identify_x_l690_690431

theorem identify_x (x : ℝ) : 
  (∀ (A : x), (reciprocal_x_not_less_than_1 x ∨ x_does_not_contain_digit_6 x ∨ x_cubed_less_than_221 x) ∧ 
  ¬ (reciprocal_x_not_less_than_1 x ∧ x_does_not_contain_digit_6 x ∧ x_cubed_less_than_221 x)) ∧ 
  (∀ (B : x), (x_is_even x ∨ x_is_prime x ∨ x_is_multiple_of_5 x) ∧
  ¬ (x_is_even x ∧ x_is_prime x ∧ x_is_multiple_of_5 x)) ∧ 
  (∀ (C : x), (x_cannot_be_ratio x ∨ x_less_than_6 x ∨ x_square_of_nat x) ∧ 
  ¬ (x_cannot_be_ratio x ∧ x_less_than_6 x ∧ x_square_of_nat x)) ∧ 
  (∀ (D : x), (x_greater_than_20 x ∨ x_log_at_least_2 x ∨ x_not_less_than_10 x) ∧ 
  ¬ (x_greater_than_20 x ∧ x_log_at_least_2 x ∧ x_not_less_than_10 x)) → 
  x = 25 := 
sorry

end identify_x_l690_690431


namespace sum_of_fx_l690_690117

def f (x : ℝ) : ℝ := x / (1 + x)

theorem sum_of_fx : 
  f 1 + 
  (∑ k in Finset.range 2017, f (k + 2)) + 
  (∑ k in Finset.range 2017, f (1 / (k + 2))) = 
  4033 / 2 := 
by
  sorry

end sum_of_fx_l690_690117


namespace area_sum_triangle_PQR_QMN_l690_690521

theorem area_sum_triangle_PQR_QMN :
  ∀ (P Q R M N : Point),
    is_midpoint M P Q → 
    on_line N P R → 
    length PR = 2 →
    angle PQR = 70 →
    angle QPR = 50 →
    angle RPQ = 60 →
    angle MNQ = 30 →
    area P Q R + 2 * area Q M N = 2 * sin (70 * pi / 180) + 2 * sin (50 * pi / 180) :=
by
  sorry

end area_sum_triangle_PQR_QMN_l690_690521


namespace prob_non_defective_l690_690004

def product_grade :=
  | A
  | B
  | C

def is_defective (grade : product_grade) : Prop :=
  grade = product_grade.B ∨ grade = product_grade.C

def prob_B : ℝ := 0.05
def prob_C : ℝ := 0.03
def prob_defective : ℝ := prob_B + prob_C

theorem prob_non_defective : (1 - prob_defective) = 0.92 :=
by
  sorry

end prob_non_defective_l690_690004


namespace impossible_uniformity_l690_690384

-- Define the dimensions of the board
def board_width : ℕ := 2018
def board_height : ℕ := 2019

-- Define what it means for two squares to be neighbors
def is_neighbor (x1 y1 x2 y2 : ℕ) : Prop :=
(abs (x1 - x2) + abs (y1 - y2) = 1)

-- Define the condition of averaging neighbors
def average_neighbors (board : ℕ → ℕ → ℤ) (x y : ℕ) : ℤ :=
  let sum := (if x > 0 then board (x-1) y else 0) +
             (if y > 0 then board x (y-1) else 0) +
             (if x < board_width - 1 then board (x+1) y else 0) +
             (if y < board_height - 1 then board x (y+1) else 0)
  let count := (if x > 0 then 1 else 0) +
               (if y > 0 then 1 else 0) +
               (if x < board_width - 1 then 1 else 0) +
               (if y < board_height - 1 then 1 else 0)
  sum / count

-- Define a statement that it is not always possible to make the board uniform
theorem impossible_uniformity : ¬ ∀ (board : ℕ → ℕ → ℤ), 
  ∃ (turns : ℕ), -- There exists some finite number of turns 
  ∀ (x y : ℕ), (x < board_width) ∧ (y < board_height) → 
  (let new_value := average_neighbors board x y in 
   (new_value = average_neighbors board x y)) := 
sorry

end impossible_uniformity_l690_690384


namespace find_function_expression_l690_690473

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 - 1) = x^4 + 1) :
  ∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2 :=
sorry

end find_function_expression_l690_690473


namespace movement_down_l690_690538

def point := (ℤ × ℤ)

theorem movement_down (C D : point) (hC : C = (1, 2)) (hD : D = (1, -1)) :
  D = (C.1, C.2 - 3) :=
by
  sorry

end movement_down_l690_690538


namespace count_two_digit_primes_with_digit_sum_10_l690_690999

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l690_690999


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l690_690914

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l690_690914


namespace sculpture_and_base_height_l690_690732

structure Dimensions where
  height : Nat
  feet : Nat := height / 12
  inches : Nat := height % 12
  deriving Repr, BEq

def sculptureHeight : Dimensions :=
  let heightRectPrism := 8
  let heightCylinder := 12 + 3
  let heightPyramid := 10
  Dimensions.mk (heightRectPrism + heightCylinder + heightPyramid)

def baseHeight : Dimensions :=
  Dimensions.mk 10

def combinedHeight (sculpture : Dimensions) (base : Dimensions) : Dimensions :=
  let totalInches := sculpture.height + base.height
  Dimensions.mk totalInches

theorem sculpture_and_base_height :
  combinedHeight sculptureHeight baseHeight = Dimensions.mk (3 * 12 + 7) :=
sorry

end sculpture_and_base_height_l690_690732


namespace max_mass_ethanol_from_glucose_l690_690409

/- Define constants for molar masses -/
def M_glucose : ℚ := 180.2
def M_ethanol : ℚ := 46.1

/- Define the mass of glucose -/
def mass_glucose : ℚ := 15.5 * 1000 -- convert kg to g

/- Number of moles of glucose -/
def moles_glucose : ℚ := mass_glucose / M_glucose

/- Number of moles of ethanol produced given the stoichiometry -/
def moles_ethanol : ℚ := moles_glucose * 2

/- Mass of ethanol produced -/
def mass_ethanol : ℚ := moles_ethanol * M_ethanol

/- Convert the mass of ethanol to kg -/
def mass_ethanol_kg : ℚ := mass_ethanol / 1000

theorem max_mass_ethanol_from_glucose (M_glucose M_ethanol mass_glucose moles_glucose moles_ethanol mass_ethanol mass_ethanol_kg) :
  mass_ethanol_kg = 7.93 := 
sorry

end max_mass_ethanol_from_glucose_l690_690409
