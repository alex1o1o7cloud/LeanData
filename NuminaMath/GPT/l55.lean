import Mathlib

namespace longest_side_of_triangle_l55_55242

theorem longest_side_of_triangle (x : ℝ) (h1 : 8 + (2 * x + 5) + (3 * x + 2) = 40) : 
  max (max 8 (2 * x + 5)) (3 * x + 2) = 17 := 
by 
  -- proof goes here
  sorry

end longest_side_of_triangle_l55_55242


namespace min_photos_required_l55_55825

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55825


namespace vector_magnitude_difference_l55_55084

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55084


namespace prod_1_to_25_eq_3276_l55_55348

theorem prod_1_to_25_eq_3276 : 
  ∏ n in finset.range 25 + 1, ((n : ℝ) + 3) / n = 3276 := 
sorry

end prod_1_to_25_eq_3276_l55_55348


namespace projectile_reaches_height_at_first_l55_55574

noncomputable def reach_height (t : ℝ) : ℝ :=
-16 * t^2 + 80 * t

theorem projectile_reaches_height_at_first (t : ℝ) :
  reach_height t = 36 → t = 0.5 :=
by
  -- The proof can be provided here
  sorry

end projectile_reaches_height_at_first_l55_55574


namespace vec_magnitude_is_five_l55_55059

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55059


namespace num_four_digit_integers_divisible_by_7_l55_55779

theorem num_four_digit_integers_divisible_by_7 : 
  let valid_pairs_a_d := finset.filter (λ (ad : ℕ × ℕ), (ad.1 ≡ ad.2 [MOD 7])) ((finset.range 10).product (finset.range 10)) in
  let valid_pairs_b_c := finset.filter (λ (bc : ℕ × ℕ), (10 * bc.1 + bc.2 ≡ 0 [MOD 7])) ((finset.range 10).product (finset.range 10)) in
  (valid_pairs_a_d.card * valid_pairs_b_c.card = 210) := 
by sorry

end num_four_digit_integers_divisible_by_7_l55_55779


namespace bob_daily_earnings_l55_55214

-- Define Sally's daily earnings
def Sally_daily_earnings : ℝ := 6

-- Define the total savings after a year for both Sally and Bob
def total_savings : ℝ := 1825

-- Define the number of days in a year
def days_in_year : ℝ := 365

-- Define Bob's daily earnings
variable (B : ℝ)

-- Define the proof statement
theorem bob_daily_earnings : (3 + B / 2) * days_in_year = total_savings → B = 4 :=
by
  sorry

end bob_daily_earnings_l55_55214


namespace find_conjugate_l55_55426

def is_conjugate (z : ℂ) (c : ℂ) : Prop :=
  complex.conj z = c

theorem find_conjugate : 
  ∀ z : ℂ, z - complex.I = (3 + complex.I) / (1 + complex.I) → is_conjugate z 2 := 
by
  intro z h
  sorry

end find_conjugate_l55_55426


namespace lines_of_service_l55_55243

theorem lines_of_service (n : ℕ) (k : ℕ) (h₀ : n = 8) (h₁ : k = 4) :
  (∑ m in finset.range (n+1), if m = k then (nat.factorial n) / ((nat.factorial k) * nat.factorial (n - k)) * (nat.factorial k) * (nat.factorial (n - k)) else 0 ) = 40320 := 
by
  sorry

end lines_of_service_l55_55243


namespace no_spy_at_table_l55_55928

-- Define types representing the entities
inductive Person
| knight : Person  -- Always tells the truth
| liar : Person    -- Always lies
| spy : Person     -- Always lies but believed to be a knight

-- Define the condition under which neighbors statements are given
def statements (p : Person) (right_neighbor : Person) : Prop :=
match p with
| Person.knight => (right_neighbor = Person.liar)
| Person.liar   => (right_neighbor = Person.knight) → Prop
| Person.spy    => (right_neighbor = Person.liar)  -- Spy lies but is thought to be a knight, thus has the same condition as knight for liar
    

-- Assumptions based on problem
constant people_at_table : list Person
constant num : ℕ
constant num_stated_liar : ℕ := 12
constant num_total := people_at_table.length

-- Define the problem
theorem no_spy_at_table (h1 : num_stated_liar = 12)
                        (h2 : ∀ (p : Person), p ∈ people_at_table → Nationality.USA) :
    ¬ ∃ s ∈ people_at_table, s = Person.spy :=

sorry

end no_spy_at_table_l55_55928


namespace sum_real_values_absolute_eq_l55_55252

theorem sum_real_values_absolute_eq {x : ℝ} :
  (∀ x, |x + 3| = (3 / 2) * |x - 3| → x = 15 ∨ x = 3 / 5) →
  ∃ s, (s = 15 + 3 / 5) := by
s    ← sorry -- proof required

end sum_real_values_absolute_eq_l55_55252


namespace vector_magnitude_difference_l55_55093

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55093


namespace range_of_a_if_f1_has_zero_range_of_f2_when_a_is_neg1_l55_55438

-- Definition for Condition 1
def f1 (x : ℝ) (a : ℝ) : ℝ := real.sqrt x + a * real.abs (x + 1)

-- Lean statement for Condition 1 with the provided correct answer
theorem range_of_a_if_f1_has_zero :
  (∃ x : ℝ, x ≥ 0 ∧ f1 x = 0) → (a = -1/2 ∨ (a ≤ 0 ∧ a ≥ -1/2)) :=
sorry

-- Definition for Condition 2
def f2 (x : ℝ) : ℝ := real.sqrt x - real.abs (x + 1)

-- Lean statement for Condition 2 with the provided correct answer
theorem range_of_f2_when_a_is_neg1 :
  (∃ x : ℝ, x ≥ 0 → (f2 x ∈ set.Iic (-3/4))) :=
sorry

end range_of_a_if_f1_has_zero_range_of_f2_when_a_is_neg1_l55_55438


namespace cos_alpha_value_l55_55830

theorem cos_alpha_value (α : ℝ) 
  (h_cos_alpha_plus : Float.cos (α + Float.pi / 6) = 1 / 3)
  (h_alpha_range : 0 ≤ α ∧ α ≤ Float.pi / 2) :
  Float.cos α = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by 
  sorry

end cos_alpha_value_l55_55830


namespace boundary_of_T_has_6_sides_l55_55936

variables {a x y : ℝ}
def point_in_T (x y a : ℝ) :=
  a ≤ x ∧ x ≤ 3*a ∧
  a ≤ y ∧ y ≤ 3*a ∧
  x + y ≥ 2*a ∧
  x + 2*a ≥ 2*y ∧
  y + 2*a ≥ 2*x ∧
  x + y ≤ 4*a

theorem boundary_of_T_has_6_sides (a : ℝ) (ha : 0 < a) :
  ∃ vs : list (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ vs → point_in_T p.1 p.2 a) ∧ 
    vs.length = 6 :=
sorry

end boundary_of_T_has_6_sides_l55_55936


namespace number_of_zeros_in_interval_l55_55514

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Icc (-1 : ℝ) 1 then
    -sqrt (1 - x^2)
  else
    f (x % 2)

noncomputable def g (x : ℝ) : ℝ :=
  f x - exp x + 1

theorem number_of_zeros_in_interval :
  ∃ n, ∀ a b, a = (-2017 : ℝ) ∧ b = 2017 → 
    (number_of_zeros (g) a b) = n :=
sorry -- proof to be provided

end number_of_zeros_in_interval_l55_55514


namespace John_l55_55170

-- Define variables for the conditions
def original_salary : ℝ := 55
def percentage_increase : ℝ := 9.090909090909092
def decimal_increase : ℝ := percentage_increase / 100
def raise_amount : ℝ := original_salary * decimal_increase
def new_salary : ℝ := original_salary + raise_amount

-- State the theorem to prove the new salary
theorem John's_new_salary_is_60 : new_salary = 60 := by
  sorry

end John_l55_55170


namespace fractional_part_inequality_no_c_greater_than_1_l55_55508

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem fractional_part_inequality (n : ℕ) (h_pos : 0 < n) : 
  fractional_part (n * Real.sqrt 3) > 1 / (n * Real.sqrt 3) := 
by
  sorry

theorem no_c_greater_than_1 (c : ℝ) (h_c : c > 1) : 
  ∃ n : ℕ, 0 < n ∧ fractional_part (n * Real.sqrt 3) ≤ c / (n * Real.sqrt 3) := 
by
  sorry

end fractional_part_inequality_no_c_greater_than_1_l55_55508


namespace math_problem_statement_l55_55845

open Real

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def line_from_points (a : ℝ) (b : ℝ) (p1 p2 : Point) : ℝ :=
  abs (a * p1.x + b * p1.y + a * p2.x + b * p2.y) / sqrt (a^2 + b^2)

noncomputable def find_line : ℝ :=
  sorry

noncomputable def number_of_lines (m : ℝ) : ℕ :=
  if m < 2.5 then 4 else if m = 2.5 then 3 else 2

theorem math_problem_statement :
  let A := Point.mk 1 2
  let B := Point.mk 5 (-1)
  let mid_AB := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)
  let d_AB := distance A B
  let lines_parallel := (|AB| / 2) > 2
  distance A B = 5 →
  (line_from_points 3 4 A B = 2 ∧ line_from_points 3 4 A B = 1) ∨
  (line_from_points 7 24 mid_AB = 1) ∧
  (∀ m > 0,  number_of_lines m = 
    if m < 2.5 then 4 
    else if m = 2.5 then 3 
    else 2)
:=
sorry

end math_problem_statement_l55_55845


namespace problem_proof_l55_55487

def AB := AC
def DEF_is_equilateral := True
def angle_ABC_eq_angle_ACB := (angle_ABC = 50 ∧ angle_ACB = 50)
def angles := (angle_BFD, angle_ADE, angle_FEC)
def relationships (a b c : RealAngle): Prop := a = b - RealAngle.ofDegrees 70 + c

theorem problem_proof
  (AB : Length)
  (AC : Length)
  (ABCDE_is_equilateral : DEF_is_equilateral)
  (θ : RealAngle)
  (θ_eq_50 : θ = RealAngle.ofDegrees 50)
  (a b c : RealAngle)
  (angle_ABC_eq_angle_ACB : angle_ABC_eq_angle_ACB)
  (angles : angles)
  : relationships a b c :=
begin
  sorry
end

end problem_proof_l55_55487


namespace problem_1_min_value_problem_2_min_value_l55_55855

-- Problem 1 statement
theorem problem_1_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min (max x (1/y) + max y (2/x)) = 2 * Real.sqrt 2 :=
sorry

-- Problem 2 statement
theorem problem_2_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  min (max x (1/y) + max y (2/z) + max z (3/x)) = 2 * Real.sqrt 5 :=
sorry

end problem_1_min_value_problem_2_min_value_l55_55855


namespace corrected_mean_l55_55244

theorem corrected_mean (mean_incorrect : ℝ) (number_of_observations : ℕ) (wrong_observation correct_observation : ℝ) : 
  mean_incorrect = 36 → 
  number_of_observations = 50 → 
  wrong_observation = 23 → 
  correct_observation = 43 → 
  (mean_incorrect * number_of_observations + (correct_observation - wrong_observation)) / number_of_observations = 36.4 :=
by
  intros h_mean_incorrect h_number_of_observations h_wrong_observation h_correct_observation
  have S_incorrect : ℝ := mean_incorrect * number_of_observations
  have difference : ℝ := correct_observation - wrong_observation
  have S_correct : ℝ := S_incorrect + difference
  have mean_correct : ℝ := S_correct / number_of_observations
  sorry

end corrected_mean_l55_55244


namespace sum_when_max_power_less_500_l55_55718

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55718


namespace min_photos_exists_l55_55794

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55794


namespace two_digit_numbers_division_condition_l55_55630

theorem two_digit_numbers_division_condition {n x y q : ℕ} (h1 : 10 * x + y = n)
  (h2 : n % 6 = x)
  (h3 : n / 10 = 3) (h4 : n % 10 = y) :
  n = 33 ∨ n = 39 := 
sorry

end two_digit_numbers_division_condition_l55_55630


namespace max_diagonals_same_length_l55_55478

theorem max_diagonals_same_length (n : ℕ) (h : n = 1000) : 
  ∃ m, m = 2000 ∧ 
  (∀ (d : finset (ℕ × ℕ)), d.card = m → 
    ∀ (a b c : (ℕ × ℕ)), a ∈ d → b ∈ d → c ∈ d → (a.2 - a.1 % n) = (b.2 - b.1 % n) ∨ (b.2 - b.1 % n) = (c.2 - c.1 % n) ∨ (a.2 - a.1 % n) = (c.2 - c.1 % n)
  ) :=
sorry

end max_diagonals_same_length_l55_55478


namespace sin2x_sin4x_eq_cos2x_cos4x_l55_55129

theorem sin2x_sin4x_eq_cos2x_cos4x (x : ℝ) (h : Real.sin (2 * x) * Real.sin (4 * x) = Real.cos (2 * x) * Real.cos (4 * x)) : 
  x = Real.to_deg 15 :=
sorry

end sin2x_sin4x_eq_cos2x_cos4x_l55_55129


namespace sum_series_eq_l55_55982

/-- Given n is a positive integer, we prove that:
    ∑ m in range n, 1 / ((2 * m + 1) * (2 * (m + 1))) = 
    ∑ m in range (2 * n), if (n + 1 ≤ m + 1 ∧ m + 1 ≤ 2 * n) then 1 / (m + 1) else 0  -/
theorem sum_series_eq (n : ℕ) (hn : n > 0) :
  ∑ m in Finset.range n, 1 / ((2 * m + 1) * (2 * (m + 1))) = 
  ∑ m in Finset.range (2 * n), if (n + 1 ≤ m + 1 ∧ m + 1 ≤ 2 * n) then 1 / (m + 1) else 0 :=
by sorry

end sum_series_eq_l55_55982


namespace candles_on_rituprts_cake_l55_55205

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l55_55205


namespace triangle_orthopole_l55_55296

-- Given conditions
def circumcenter (A B C : Point) : Point := sorry
def orthocenter (A B C : Point) : Point := sorry
def nine_point_center (A B C : Point) : Point := sorry
def reflection (P A B : Point) : Point := sorry
def circumcenter_of_triangles (A : Point) (B C : Triangle) : Point := sorry

variable (A B C : Point)
variable (O : Point := circumcenter A B C)
variable (H : Point := orthocenter A B C)
variable (N : Point := nine_point_center A B C)
variable (O_A : Point := circumcenter_of_triangles B O C)
variable (O_B : Point := circumcenter_of_triangles C O A)
variable (O_C : Point := circumcenter_of_triangles A O B)
variable (N_A : Point := reflection N B C)
variable (N_B : Point := reflection N C A)
variable (N_C : Point := reflection N A B)
variable (O_a : Point := circumcenter A N_B N_C)
variable (O_b : Point := circumcenter B N_C N_A)
variable (O_c : Point := circumcenter C N_A N_B)

-- Prove orthogonality as described:
theorem triangle_orthopole :
  orthopole (triangle O_a O_b O_c) (triangle A B C) :=
sorry

end triangle_orthopole_l55_55296


namespace vector_magnitude_difference_l55_55019

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55019


namespace remainder_of_12_factorial_mod_13_l55_55402

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end remainder_of_12_factorial_mod_13_l55_55402


namespace find_coefficients_l55_55849

-- Define the roots as complex numbers
def x1 : ℂ := 1 - complex.i
def x2 : ℂ := 1 + complex.i

-- Define the quadratic equation with real coefficients
variable (a b : ℝ)
def quadratic_eq (x : ℂ) : Prop := x^2 + a * x + b = 0

-- Theorem: Given the roots x1 and x2, find values of a and b.
theorem find_coefficients (ha : quadratic_eq a b x1) (hb : quadratic_eq a b x2) : a = -2 ∧ b = 2 :=
by {
  sorry
}

end find_coefficients_l55_55849


namespace propositions_A_and_D_true_l55_55633

theorem propositions_A_and_D_true :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧ (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) :=
by
  sorry

end propositions_A_and_D_true_l55_55633


namespace vec_magnitude_is_five_l55_55051

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55051


namespace min_photos_l55_55787

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55787


namespace vector_magnitude_subtraction_l55_55062

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55062


namespace sum_a1_a3_a5_eq_122_l55_55458

theorem sum_a1_a3_a5_eq_122 (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) →
  a1 + a3 + a5 = 122 :=
by
  intros h,
  sorry

end sum_a1_a3_a5_eq_122_l55_55458


namespace simplify_expression_l55_55554

theorem simplify_expression (x : ℝ) :
  (√2 / 4) * sin (π / 4 - x) + (√6 / 4) * cos (π / 4 - x) = (√2 / 2) * sin (7 * π / 12 - x) := 
by 
  sorry

end simplify_expression_l55_55554


namespace sum_when_max_power_less_500_l55_55720

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55720


namespace sum_of_a_and_b_l55_55731

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55731


namespace inclination_angle_range_l55_55763

theorem inclination_angle_range (a θ : ℝ) 
    (h : θ = real.arctan (- (2 * a) / (a^2 + 1))) :
     θ ∈ set.Icc 0 (real.pi / 4) ∪ set.Icc (3 * real.pi / 4) real.pi := 
sorry

end inclination_angle_range_l55_55763


namespace vector_magnitude_subtraction_l55_55067

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55067


namespace sail_pressure_l55_55585

def pressure (k A V : ℝ) : ℝ := k * A * V^2

theorem sail_pressure (k : ℝ)
  (h_k : k = 1 / 800) 
  (A : ℝ) 
  (V : ℝ) 
  (P : ℝ)
  (h_initial : A = 1 ∧ V = 20 ∧ P = 0.5) 
  (A2 : ℝ) 
  (V2 : ℝ) 
  (h_doubled : A2 = 2 ∧ V2 = 30) :
  pressure k A2 V2 = 2.25 :=
by
  sorry

end sail_pressure_l55_55585


namespace magnitude_of_a_minus_b_l55_55074

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55074


namespace f_odd_f_inequality_a_gt_1_f_inequality_a_lt_1_l55_55435

def f (x : ℝ) (a : ℝ) : ℝ := log a (x + 2) - log a (2 - x)

theorem f_odd (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∀ x : ℝ, (-2 < x) ∧ (x < 2) → f x a = -f (-x) a :=
by
  sorry

theorem f_inequality_a_gt_1 (a : ℝ) (ha_gt_1 : 1 < a) :
  ∀ x : ℝ, ((0 < x) ∧ (x < 2)) ↔ (0 < f x a) := 
by
  sorry

theorem f_inequality_a_lt_1 (a : ℝ) (ha_pos : 0 < a) (ha_lt_1 : a < 1) :
  ∀ x : ℝ, ((-2 < x) ∧ (x < 0)) ↔ (0 < f x a) := 
by
  sorry

end f_odd_f_inequality_a_gt_1_f_inequality_a_lt_1_l55_55435


namespace degree_of_mult_poly_l55_55629

def degree_of_polynomial : ℕ := 6

theorem degree_of_mult_poly :
  ∀ x : ℝ, x ≠ 0 →
  degree (x^4 * (x^2 + 1/x^2) * (1 + 2/x + 3/x^2)) = degree_of_polynomial :=
by
  sorry

end degree_of_mult_poly_l55_55629


namespace christina_friends_place_distance_l55_55346

-- Definitions representing conditions
def daily_distance_to_and_from_school : ℕ := 14
def total_distance_walked_in_a_week : ℕ := 74
def days_in_a_week : ℕ := 5
def days_in_a_school_week : ℕ := 4
def additional_distance_to_friends_place : ℕ := 4

-- The proof problem
theorem christina_friends_place_distance :
  ∃ d, d = additional_distance_to_friends_place ∧
  let regular_distance := daily_distance_to_and_from_school * days_in_a_school_week in
  let friday_distance := total_distance_walked_in_a_week - regular_distance in
  friday_distance - daily_distance_to_and_from_school = d :=
begin
  sorry
end

end christina_friends_place_distance_l55_55346


namespace max_take_home_pay_at_25_l55_55899

noncomputable def tax (x : ℝ) : ℝ :=
  (2 * x / 100) * 1000 * x

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax x

theorem max_take_home_pay_at_25 :
  ∃ x : ℝ, (x = 25) ∧ (take_home_pay x = 12500) :=
begin
  use 25,
  split,
  { refl },
  { unfold take_home_pay tax,
    simp, -- simplifies tax and take_home_pay
    norm_num, -- calculate numeric expressions
    sorry -- This is where the full proof would go
  }

end max_take_home_pay_at_25_l55_55899


namespace parabola_and_tangents_l55_55432

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def tangent_line_1 (x y : ℝ) : Prop := y = 0

noncomputable def tangent_line_2 (x y : ℝ) : Prop := x + y + 1 = 0

theorem parabola_and_tangents (b x y : ℝ) (h1 : 0 < b) (h2 : b < 2) 
  (h3 : (x,y) = (-1, 0)) 
  (h4 : ∀ (e : ℝ), e = real.sqrt 3 / 2) :
  (parabola_eq x y) ∧ (tangent_line_1 x y ∨ tangent_line_2 x y) :=
by 
  sorry

end parabola_and_tangents_l55_55432


namespace magnitude_of_a_minus_b_l55_55072

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55072


namespace simplify_expression_1_simplify_expression_2_l55_55987

-- Problem 1: Simplification proof
theorem simplify_expression_1 : 
    (\left(\left(0.064^{\frac{1}{5}}\right)^{-2.5}\right)^{\frac{2}{3}} - \sqrt[3]{3\frac{3}{8}} - \pi^{0} = 0) := 
by 
    sorry

-- Problem 2: Simplification proof
theorem simplify_expression_2 : 
    (\frac{2\log_2 + \log_3}{1 + \frac{1}{2}\log_2 0.36 + \frac{1}{4}\log_2 16} = \frac{2\log_2 + \log_3}{\log_{10} 24}) := 
by 
    sorry

end simplify_expression_1_simplify_expression_2_l55_55987


namespace min_photos_required_l55_55824

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55824


namespace vector_magnitude_subtraction_l55_55070

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55070


namespace f_explicit_formula_range_of_a_l55_55427

section
variable (f : ℝ → ℝ) (a : ℝ)
hypotheses
  (h1 : ∀ x, f (-x) = -f (x)) -- odd function
  (h2 : ∀ x, 0 ≤ x → x ≤ 3 → f x = - (1 / 3) * x) -- linear on [0, 3]
  (h3 : ∀ x, 3 ≤ x → x ≤ 6 → f x = -x^2 + 10 * x - 22) -- quadratic on [3, 6]
  (h4 : ∀ x, 3 ≤ x ∧ x ≤ 6 → f x ≤ 3 ∧ f 5 = 3 ∧ f 6 = 2) -- conditions on [3, 6]
  (h5 : ∀ x, x = 5 → f 5 = 3) -- f(5) = 3

theorem f_explicit_formula : 
  f = λ x, 
    if (x ∈ Icc (-6) (-3)) then -x^2 + 10 * x + 22 
    else if (x ∈ Ioo (-3) (3)) then - (1 / 3) * x
    else -x^2 + 10 * x - 22 :=
sorry

theorem range_of_a :
  (∀ x, f x - a^2 - 4 * a ≥ 0) ↔ (-3 ≤ a ∧ a ≤ -1) :=
sorry
end

end f_explicit_formula_range_of_a_l55_55427


namespace find_m_l55_55654

theorem find_m 
  (h : ( (1 ^ m) / (5 ^ m) ) * ( (1 ^ 16) / (4 ^ 16) ) = 1 / (2 * 10 ^ 31)) :
  m = 31 :=
by
  sorry

end find_m_l55_55654


namespace boys_total_count_l55_55258

theorem boys_total_count 
  (avg_age_all: ℤ) (avg_age_first6: ℤ) (avg_age_last6: ℤ)
  (total_first6: ℤ) (total_last6: ℤ) (total_age_all: ℤ) :
  avg_age_all = 50 →
  avg_age_first6 = 49 →
  avg_age_last6 = 52 →
  total_first6 = 6 * avg_age_first6 →
  total_last6 = 6 * avg_age_last6 →
  total_age_all = total_first6 + total_last6 →
  total_age_all = avg_age_all * 13 :=
by
  intros h_avg_all h_avg_first6 h_avg_last6 h_total_first6 h_total_last6 h_total_age_all
  rw [h_avg_all, h_avg_first6, h_avg_last6] at *
  -- Proof steps skipped
  sorry

end boys_total_count_l55_55258


namespace sequence_sum_l55_55593

noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry
noncomputable def a₅ : ℝ := sorry
noncomputable def a₆ : ℝ := sorry
noncomputable def a₇ : ℝ := sorry
noncomputable def a₈ : ℝ := sorry
noncomputable def q : ℝ := sorry

axiom condition_1 : a₁ + a₂ + a₃ + a₄ = 1
axiom condition_2 : a₅ + a₆ + a₇ + a₈ = 2
axiom condition_3 : q^4 = 2

theorem sequence_sum : q = (2:ℝ)^(1/4) → a₁ + a₂ + a₃ + a₄ = 1 → 
  (a₁ * q^16 + a₂ * q^17 + a₃ * q^18 + a₄ * q^19) = 16 := 
by
  intros hq hsum_s4
  sorry

end sequence_sum_l55_55593


namespace general_term_formula_correct_l55_55147

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def general_term_formula : ℝ :=
  if q > 0 ∧ a 2 - a 1 = 1 then 3 * (4 / 3) ^ (n - 1) else 0

theorem general_term_formula_correct
  (ha_positive : ∀ n, a n > 0)
  (ha_cond : a 2 - a 1 = 1)
  (ha_min : ∃ n, a 5 = 3 * ((4 / 3) ^ 4)):
  ∀ n, a n = 3 * (4 / 3) ^ (n - 1) := by
  sorry

end general_term_formula_correct_l55_55147


namespace vector_magnitude_l55_55104

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55104


namespace manager_salary_correct_l55_55228

section ManagerSalary

variables
  (num_employees : ℕ)
  (avg_salary : ℕ)
  (new_avg_salary : ℕ)

-- Given conditions
def total_salary_employees := num_employees * avg_salary
def total_salary_with_manager := (num_employees + 1) * new_avg_salary

-- Define the manager's salary
def manager_salary := total_salary_with_manager - total_salary_employees

-- The main statement to be proved
theorem manager_salary_correct
  (h1 : num_employees = 18)
  (h2 : avg_salary = 2000)
  (h3 : new_avg_salary = 2200) :
  manager_salary num_employees avg_salary new_avg_salary = 5800 := 
by
  rw [manager_salary, total_salary_with_manager, total_salary_employees],
  -- These rewrites are essentially symbolic manipulations congruent with the
  -- original problem setup
  rw [h1, h2, h3],
  sorry

end ManagerSalary

end manager_salary_correct_l55_55228


namespace donation_ratio_l55_55226

theorem donation_ratio (D1 : ℝ) (D1_value : D1 = 10)
  (total_donation : D1 + D1 * 2 + D1 * 4 + D1 * 8 + D1 * 16 = 310) : 
  2 = 2 :=
by
  sorry

end donation_ratio_l55_55226


namespace greatest_value_sum_eq_24_l55_55724

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55724


namespace vector_magnitude_subtraction_l55_55031

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55031


namespace solve_for_n_l55_55555

theorem solve_for_n (n : ℕ) (h : (16^n) * (16^n) * (16^n) * (16^n) * (16^n) = 256^5) : n = 2 := by
  sorry

end solve_for_n_l55_55555


namespace mary_money_left_l55_55953

theorem mary_money_left (q : ℝ) : 
  let drink_cost := q
  let medium_pizza_cost := 3 * q
  let large_pizza_cost := 5 * q
  let total_cost := 4 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  in 30 - total_cost = 30 - 17 * q := 
by
  sorry

end mary_money_left_l55_55953


namespace greatest_value_sum_eq_24_l55_55722

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55722


namespace distance_midpoint_AD_to_BC_l55_55896

variable (AC BC BD : ℕ)
variable (perpendicular : Prop)
variable (d : ℝ)

theorem distance_midpoint_AD_to_BC
  (h1 : AC = 6)
  (h2 : BC = 5)
  (h3 : BD = 3)
  (h4 : perpendicular) :
  d = Real.sqrt 5 + 2 := by
  sorry

end distance_midpoint_AD_to_BC_l55_55896


namespace find_p_l55_55396

variables (a b c p : ℝ)

theorem find_p 
  (h1 : 9 / (a + b) = 13 / (c - b)) : 
  p = 22 :=
sorry

end find_p_l55_55396


namespace oldest_child_age_l55_55165

-- Declare the conditions as variables
variable (jane_age_start_babysitting : ℕ := 18)
variable (jane_age_stop_babysitting : ℕ := 34 - 10)
variable (current_jane_age : ℕ := 34)

-- Define the conditions
def half_age_condition (jane_age child_age : ℕ) : Prop :=
  child_age ≤ jane_age / 2

-- Define interpretation of the problem
def problem_condition_3 (jane_age_stop_babysitting : ℕ) : Prop :=
  jane_age_stop_babysitting = current_jane_age - 10

def problem_condition_4 (jane_age_stop_babysitting : ℕ) (child_age_at_stop : ℕ) : Prop :=
  half_age_condition jane_age_stop_babysitting child_age_at_stop

def problem_condition_5 (jane_age_stop_babysitting : ℕ) (child_age_at_stop : ℕ) (years_since_stop : ℕ) : ℕ :=
  child_age_at_stop + years_since_stop

-- Define the mathematically equivalent proof problem
theorem oldest_child_age :
  ∀ (jane_age_stop_babysitting : ℕ) (child_age_at_stop : ℕ),
    problem_condition_3 jane_age_stop_babysitting →
    problem_condition_4 jane_age_stop_babysitting child_age_at_stop →
    problem_condition_5 jane_age_stop_babysitting child_age_at_stop 10 = 22 :=
by 
  sorry

end oldest_child_age_l55_55165


namespace speed_of_first_part_l55_55670

theorem speed_of_first_part
  (v : Real)
  (H1 : ∀ (t₁ t₂ : Real), t₁ = 25 / v → t₂ = 25 / 30 → 40 = 50 / (t₁ + t₂)) :
  v = 100 / 3 :=
by
  have t₁ := 25 / v
  have t₂ := 25 / 30
  calc
    40 = 50 / (t₁ + t₂) : H1 t₁ t₂ rfl rfl
    ... = 50 / ((25 / v) + (25 / 30)) : by rw [t₁, t₂]
    ... = 50 / ((25 * 30 + 25 * v) / (30 * v)) : by rw [div_add_div (25 : ℝ) _ v]
    ... = 50 * (30 * v) / (25 * 30 + 25 * v) : by rw [div_div, mul_comm]
    ... = 40 * 75 /  (30) 
    ... = 33.3 sorry

end speed_of_first_part_l55_55670


namespace disk_diameter_solution_l55_55622

noncomputable def disk_diameter_condition : Prop :=
∃ x : ℝ, 
  (4 * Real.sqrt 3 + 2 * Real.pi) * x^2 - 12 * x + Real.sqrt 3 = 0 ∧
  x < Real.sqrt 3 / 6 ∧ 
  2 * x = 0.36

theorem disk_diameter_solution : exists (x : ℝ), 
  disk_diameter_condition := 
sorry

end disk_diameter_solution_l55_55622


namespace M_inter_N_eq_13_l55_55866

def M := {x : ℝ | -1 < x ∧ x ≤ 3}
def N := {-3, -1, 1, 3, 5}

theorem M_inter_N_eq_13 : M ∩ N = {1, 3} := by
  sorry

end M_inter_N_eq_13_l55_55866


namespace gala_arrangements_l55_55307

theorem gala_arrangements :
  let original_programs := 10
  let added_programs := 3
  let total_positions := original_programs + 1 - 2 -- Excluding first and last
  (total_positions * (total_positions - 1) * (total_positions - 2)) / 6 = 165 :=
by sorry

end gala_arrangements_l55_55307


namespace complex_z_power_fraction_l55_55856

theorem complex_z_power_fraction :
  let θ := Real.pi / 12,
  z := Complex.cos θ + Complex.sin θ * Complex.I,
  i_minus_one := Complex.I - 1 in
  (z^30 + 1) / i_minus_one = -Complex.I :=
by
  sorry

end complex_z_power_fraction_l55_55856


namespace fiona_reaches_14_without_predators_l55_55256

theorem fiona_reaches_14_without_predators :
  let 
    pads : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    predators : Finset ℕ := {4, 7, 11},
    food_pad : ℕ := 14,
    start_pad : ℕ := 0,
    hop_prob : ℚ := 1 / 2,
    jump_prob : ℚ := 1 / 2
  in
  -- Define the probability that Fiona reaches pad 14 without landing on 4, 7, or 11.
  let probability := (27 : ℚ) / 512
  in
  -- Prove that the probability of the event described is 27/512.
  (fiona_probability pads predators food_pad start_pad hop_prob jump_prob) = probability :=
sorry

end fiona_reaches_14_without_predators_l55_55256


namespace vector_magnitude_subtraction_l55_55069

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55069


namespace k_h_neg2_eq_neg22_l55_55881

def h (x : ℤ) : ℤ := x^3
def k (x : ℤ) : ℤ := 3 * x + 2

theorem k_h_neg2_eq_neg22 : k(h(-2)) = -22 := by
  sorry

end k_h_neg2_eq_neg22_l55_55881


namespace total_sentence_l55_55959

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end total_sentence_l55_55959


namespace perpendicular_to_parallel_l55_55784

noncomputable theory
open_locale classical

variables (α : Type*) [plane α] (m n : Line α)

theorem perpendicular_to_parallel
  (h1 : m ⊥ α) (h2 : n ⊥ α) :
  m ∥ n :=
sorry

end perpendicular_to_parallel_l55_55784


namespace percent_decrease_is_80_l55_55294

-- Definitions based on the conditions
def original_price := 100
def sale_price := 20

-- Theorem statement to prove the percent decrease
theorem percent_decrease_is_80 :
  ((original_price - sale_price) / original_price * 100) = 80 := 
by
  sorry

end percent_decrease_is_80_l55_55294


namespace vector_magnitude_subtraction_l55_55004

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55004


namespace part_a_part_b_l55_55176

-- Definitions for S_k
noncomputable def S (x : ℕ → ℝ) (k : ℕ) : ℝ := (finset.range k).sum (λ i, (x i)^k)

-- Part (a)
theorem part_a (n : ℕ) (x : ℕ → ℝ) (h_pos : ∀ i, 0 < x i) (h : S x 1 < S x 2) : 
  strict_mono (λ k, S x k) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ (x : ℕ → ℝ) (n : ℕ), 
  (∀ i, 0 < x i) ∧ (S x 1 > S x 2) ∧ ¬ strict_antimono (λ k, S x k) :=
sorry

end part_a_part_b_l55_55176


namespace distance_small_ball_to_surface_l55_55967

-- Define the main variables and conditions
variables (R : ℝ)

-- Define the conditions of the problem
def bottomBallRadius : ℝ := 2 * R
def topBallRadius : ℝ := R
def edgeLengthBaseTetrahedron : ℝ := 4 * R
def edgeLengthLateralTetrahedron : ℝ := 3 * R

-- Define the main statement in Lean format
theorem distance_small_ball_to_surface (R : ℝ) :
  (3 * R) = R + bottomBallRadius R :=
sorry

end distance_small_ball_to_surface_l55_55967


namespace find_A_l55_55179

def hash_rel (A B : ℝ) := A^2 + B^2

theorem find_A (A : ℝ) (h : hash_rel A 7 = 196) : A = 7 * Real.sqrt 3 :=
by sorry

end find_A_l55_55179


namespace total_dollars_l55_55952

noncomputable def Mark := 4 / 5
noncomputable def Carolyn := 2 / 5
noncomputable def Dave := 1 / 2

theorem total_dollars : Mark + Carolyn + Dave = 1.7 := by
  sorry

end total_dollars_l55_55952


namespace positive_integer_solution_exists_l55_55541

theorem positive_integer_solution_exists : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + 2 * y = 7 :=
by
  use 5, 1
  split; norm_num
  split; norm_num
  sorry

end positive_integer_solution_exists_l55_55541


namespace vector_magnitude_difference_l55_55090

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55090


namespace part1_proof_part2_proof_l55_55440

section part1

variable (x m : ℝ)

def f (m x : ℝ) := (1/8) * (m - 1) * x^2 - m * x + 2 * m - 1

theorem part1_proof (h₁ : m = 1 ∨ m = 1/3) :
  ∃ x, f m x = 0 :=
by sorry

end part1

section part2

variable (x : ℝ)

def f (m x : ℝ) := (1/8) * (m - 1) * x^2 - m * x + 2 * m - 1

theorem part2_proof (h₁ : f 3 2 = 0) :
  m = 3 ∧ ∃ x, f 3 x = 0 ∧ x ≠ 2 :=
by sorry

end part2

end part1_proof_part2_proof_l55_55440


namespace area_of_region_l55_55760

theorem area_of_region : 
  let S := { p : ℝ × ℝ | (abs (p.1 - 2) ≤ p.2) ∧ (p.2 ≤ 5 - abs p.1) } in
  measure_theory.measure_space.volume.measure_of_S = 6 :=
by
  sorry

end area_of_region_l55_55760


namespace roof_weight_capacity_l55_55709

-- Conditions as definitions
def leaves_per_day : ℕ := 100
def leaves_per_pound : ℕ := 1000
def days_to_collapse : ℕ := 5000

-- The proof problem statement
theorem roof_weight_capacity : (leaves_per_day / float_of leaves_per_pound * days_to_collapse) = 500 := by
  sorry

end roof_weight_capacity_l55_55709


namespace algebraic_expression_value_l55_55375

theorem algebraic_expression_value (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (  ( ((x + 2)^2 * (x^2 - 2 * x + 4)^2) / ( (x^3 + 8)^2 ))^2
   * ( ((x - 2)^2 * (x^2 + 2 * x + 4)^2) / ( (x^3 - 8)^2 ))^2 ) = 1 :=
by
  sorry

end algebraic_expression_value_l55_55375


namespace polar_to_cartesian_l55_55489

theorem polar_to_cartesian :
  ∃ (x y : ℝ), x = 2 * Real.cos (Real.pi / 6) ∧ y = 2 * Real.sin (Real.pi / 6) ∧ 
  (x, y) = (Real.sqrt 3, 1) :=
by
  use (2 * Real.cos (Real.pi / 6)), (2 * Real.sin (Real.pi / 6))
  -- The proof will show the necessary steps
  sorry

end polar_to_cartesian_l55_55489


namespace limit_of_a_n_div_n_l55_55650

noncomputable def a_n (n : ℕ) : ℕ := sorry -- This defines the concept of a_n, as given in the conditions.

theorem limit_of_a_n_div_n : 
  ∀ n : ℕ, 
  ∃ a_n_condition : (∀ k ≤ n, a_n k = if (2^k starts with 1) then 1 else 0),
  filter.tendsto (λ n, (a_n n : ℝ) / n) filter.at_top (𝓝 (Real.log 2 / Real.log 10)) :=
by
  sorry -- This theorem states the required limit condition.

end limit_of_a_n_div_n_l55_55650


namespace vector_magnitude_subtraction_l55_55027

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55027


namespace total_markings_on_stick_l55_55312

noncomputable def markings (n m : ℕ) : ℕ := 
  (0..n).toFinset.card + (0..m).toFinset.card - (0..(n*m / (n.gcd m))).toFinset.card - 2

theorem total_markings_on_stick : markings 4 5 = 9 :=
by sorry

end total_markings_on_stick_l55_55312


namespace vector_magnitude_subtraction_l55_55009

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55009


namespace initial_percentage_acidic_liquid_l55_55277

theorem initial_percentage_acidic_liquid (P : ℝ) :
  let initial_volume := 12
  let removed_volume := 4
  let final_volume := initial_volume - removed_volume
  let desired_concentration := 60
  (P/100) * initial_volume = (desired_concentration/100) * final_volume →
  P = 40 :=
by
  intros
  sorry

end initial_percentage_acidic_liquid_l55_55277


namespace distinct_cube_labelings_l55_55766

noncomputable def vertices_set : Finset ℕ := {1, 2, 3, 4, 5, 7, 8, 9}

def sum_of_face (f: Fin 8 → ℕ) (face: Finset ℕ) : ℕ := face.sum f

def cube_faces : List (Finset ℕ) := [
  {0, 1, 2, 3}, {4, 5, 6, 7},
  {0, 1, 4, 5}, {2, 3, 6, 7},
  {0, 2, 4, 6}, {1, 3, 5, 7}
]

def is_valid_arrangement (f: Fin 8 → ℕ) : Prop :=
  (∀ i, f i ∈ vertices_set) ∧
  (cube_faces.pairwise (λ s1 s2, sum_of_face f s1 = sum_of_face f s2))

theorem distinct_cube_labelings : 
  ∃ (f: Fin 8 → ℕ), 
    is_valid_arrangement f ∧ 
    (cube_faces.pairwise (λ s1 s2, sum_of_face f s1 = 18)) ∧
    (Finset.univ.image f).card = 6 :=
  by sorry

end distinct_cube_labelings_l55_55766


namespace min_photos_needed_to_ensure_conditions_l55_55801

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55801


namespace part_I_part_II_l55_55524

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

theorem part_I (x : ℝ) :
  f(x) ≥ 10 ↔ x ∈ Set.Iic (-3) ∪ Set.Ici 7 :=
sorry

theorem part_II (t : ℝ) :
  (∀ x : ℝ, f(x) ≥ 4 / t + 2) ↔ t ∈ Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end part_I_part_II_l55_55524


namespace solve_equation_l55_55557

noncomputable def eq1 (x : ℝ) : Prop :=
  5 ^ (Real.sqrt (x ^ 3 + 3 * x ^ 2 + 3 * x + 1)) = Real.sqrt ((5 * Real.root 4 ((x + 1) ^ 5)) ^ 3)

theorem solve_equation (x : ℝ) : eq1 x → x = 65 / 16 := 
sorry

end solve_equation_l55_55557


namespace greatest_possible_sum_of_two_consecutive_even_integers_l55_55274

theorem greatest_possible_sum_of_two_consecutive_even_integers
  (n : ℤ) (h1 : Even n) (h2 : n * (n + 2) < 800) :
  n + (n + 2) = 54 := 
sorry

end greatest_possible_sum_of_two_consecutive_even_integers_l55_55274


namespace bag_total_balls_l55_55305

theorem bag_total_balls :
  let W := 10    -- number of white balls
  let G := 30    -- number of green balls
  let Y := 10    -- number of yellow balls
  let R := 47    -- number of red balls
  let P := 3     -- number of purple balls
  let non_red_purple_count := W + G + Y
  let prob_non_red_purple := 0.5
  ∃ T : ℕ, (T = W + G + Y + R + P) ∧ ((non_red_purple_count / T : ℝ) = prob_non_red_purple) ↔ T = 100 :=
by
  sorry

end bag_total_balls_l55_55305


namespace compare_values_l55_55167

noncomputable def final_values (A_0 B_0 C_0 : ℝ) : ℝ × ℝ × ℝ :=
  let A_1 := A_0 * 1.30
  let B_1 := B_0 * 0.80
  let C_1 := C_0 * 1.10
  let A := A_1 * 0.85
  let B := B_1 * 1.30
  let C := C_1 * 0.95
  (A, B, C)

theorem compare_values :
  let (A, B, C) := final_values 200 150 100
  C < B ∧ B < A :=
by {
  have h : final_values 200 150 100 = (221, 156, 104.5) := by sorry,
  rw h,
  norm_num,
}

end compare_values_l55_55167


namespace mindmaster_code_count_l55_55481

theorem mindmaster_code_count :
  let colors := 7
  let slots := 5
  (colors ^ slots) = 16807 :=
by
  -- Define the given conditions
  let colors := 7
  let slots := 5
  -- Proof statement to be inserted here
  sorry

end mindmaster_code_count_l55_55481


namespace equal_distribution_possible_rel_prime_equal_distribution_impossible_non_rel_prime_l55_55522

theorem equal_distribution_possible_rel_prime (m n : ℕ) (h_mn : Nat.coprime m n) (h_nm : n < m) :
  ∀ (balls : fin m → ℕ), ∃ (k : ℕ), (∀ i j : fin m, balls i + k * (if i.val < n then 1 else 0) =
  balls j + k * (if j.val < n then 1 else 0)) := 
sorry

theorem equal_distribution_impossible_non_rel_prime (m n : ℕ) (h_mn : ¬Nat.coprime m n) (h_nm : n < m) :
  ∃ (balls : fin m → ℕ), ∀ (k : ℕ), ∃ i j : fin m, balls i + k * (if i.val < n then 1 else 0) ≠
  balls j + k * (if j.val < n then 1 else 0) :=
sorry

end equal_distribution_possible_rel_prime_equal_distribution_impossible_non_rel_prime_l55_55522


namespace product_of_elements_eq_neg8_l55_55526

open Set

noncomputable def A (m : ℝ) : Set ℝ := {1, 2, m}

noncomputable def B (m : ℝ) : Set ℝ := {a^2 | a ∈ A m}

noncomputable def C (m : ℝ) : Set ℝ := A m ∪ B m

theorem product_of_elements_eq_neg8 (m : ℝ) (h_sum : ∑ x in C m, x = 6) : ∏ x in C m, x = -8 :=
by sorry

end product_of_elements_eq_neg8_l55_55526


namespace real_solutions_l55_55383

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l55_55383


namespace sqrt_expression_l55_55422

theorem sqrt_expression (a : ℝ) (h₀ : a + a⁻¹ = 3) (h₁ : a > 0) : a^(1/2) + a^(-1/2) = Real.sqrt 5 := 
by
  sorry

end sqrt_expression_l55_55422


namespace log_base_4_half_l55_55370

theorem log_base_4_half : ∀ (a : ℝ), a = 4 → ∀ (b : ℝ), b = 1 / 2 → log a b = -1 / 2 := 
by 
  intros a ha b hb 
  rw [ha, hb] 
  rw [real.log_div (pow_pos (by norm_num) 2) (by norm_num : (0 : ℝ) < 1)]
  sorry

end log_base_4_half_l55_55370


namespace beneficiary_received_32_176_l55_55327

noncomputable def A : ℝ := 19520 / 0.728
noncomputable def B : ℝ := 1.20 * A
noncomputable def C : ℝ := 1.44 * A
noncomputable def D : ℝ := 1.728 * A

theorem beneficiary_received_32_176 :
    round B = 32176 :=
by
    sorry

end beneficiary_received_32_176_l55_55327


namespace probability_correct_l55_55768

noncomputable def probability_sum_equals_sixteen (p_coin : ℚ) (p_die : ℚ) (age : ℕ): ℚ :=
  if age = 16 ∧ p_coin = 1 / 2 ∧ p_die = 1 / 6 then p_coin * p_die else 0

theorem probability_correct: 
  probability_sum_equals_sixteen (1/2) (1/6) 16 = 1 / 12 :=
sorry

end probability_correct_l55_55768


namespace final_mud_weight_is_4000_l55_55324

-- Define initial conditions
def total_initial_mud_weight : ℝ := 6000
def initial_water_percentage : ℝ := 88 / 100
def final_water_percentage : ℝ := 82 / 100

-- Calculate the initial weight of non-water mud
def initial_non_water_mud_weight : ℝ := (1 - initial_water_percentage) * total_initial_mud_weight

-- Define the weight of non-water mud after evaporation
def final_non_water_mud_weight := initial_non_water_mud_weight

-- Define the final weight of the mud
def final_mud_weight : ℝ := final_non_water_mud_weight / (1 - final_water_percentage)

-- The proof goal
theorem final_mud_weight_is_4000 : final_mud_weight = 4000 := by
  sorry

end final_mud_weight_is_4000_l55_55324


namespace max_number_of_girls_l55_55145

theorem max_number_of_girls (students : ℕ)
  (num_friends : ℕ → ℕ)
  (h_students : students = 25)
  (h_distinct_friends : ∀ (i j : ℕ), i ≠ j → num_friends i ≠ num_friends j)
  (h_girls_boys : ∃ (G B : ℕ), G + B = students) :
  ∃ G : ℕ, G = 13 := 
sorry

end max_number_of_girls_l55_55145


namespace greater_number_is_18_l55_55596

theorem greater_number_is_18 (x y : ℕ) (h₁ : x + y = 30) (h₂ : x - y = 6) : x = 18 :=
by
  sorry

end greater_number_is_18_l55_55596


namespace swap_checkers_possible_l55_55535

inductive Checker
| White
| Black
| Empty

def Board := List Checker

-- Initial configuration assumed
def initialBoard : Board := [Checker.White, Checker.White, Checker.White, Checker.White, Checker.Empty, Checker.Black, Checker.Black, Checker.Black, Checker.Black]

-- Final configuration to achieve
def finalBoard : Board := [Checker.Black, Checker.Black, Checker.Black, Checker.Black, Checker.Empty, Checker.White, Checker.White, Checker.White, Checker.White]

-- Move definition (not fully specified here, just a placeholder)
def is_valid_move (board : Board) (from to : ℕ) : Prop := sorry

def make_move (board : Board) (from to : ℕ) : Board := sorry

-- Theorem: We can move from initial to final config following the rules
theorem swap_checkers_possible :
  ∃ (moves : List (ℕ × ℕ)), 
    List.foldl (λ b m, make_move b m.1 m.2) initialBoard moves = finalBoard :=
sorry

end swap_checkers_possible_l55_55535


namespace sufficiency_and_necessity_condition_l55_55651

open Real

theorem sufficiency_and_necessity_condition (a : ℝ) : 
  (a > 2 → 2^a - a - 1 > 0) ∧ (¬(∀ a <= 2, 2^a - a - 1 ≤ 0)) := 
by 
  sorry

end sufficiency_and_necessity_condition_l55_55651


namespace min_photos_exists_l55_55799

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55799


namespace angle_ADB_is_60_degrees_l55_55490

theorem angle_ADB_is_60_degrees
  (ABC : Triangle)
  (right_angled_ABC : ABC.is_right_angled)
  (B : Triangle.Vertex)
  (angle_B : ABC.angle B = 30)
  (square_on_hypotenuse : Square)
  (constructed_on_hypotenuse : square_on_hypotenuse.is_constructed_on ABC.hypotenuse AC)
  (D : Point)
  (D_is_center : D = square_on_hypotenuse.center)
  (A : Triangle.Vertex)
  (C : Triangle.Vertex) :
  angle ADB = 60 :=
by 
  sorry

end angle_ADB_is_60_degrees_l55_55490


namespace problem1_problem2_l55_55655

-- Problem 1: Given conditions and final answer
theorem problem1 (P Q : Point ℝ) (P_xy : P = (4, -2)) (Q_xy : Q = (-1, 3)) :
  (∃ (C : Point ℝ) (r : ℝ), (P ∈ Circle C r) ∧ (Q ∈ Circle C r) ∧ (intercept_y_axis_length (Circle C r) = 4 * Real.sqrt 3)) →
  (Circle_eq_1 : Circle_eq _ (Circle (1, 0) (Real.sqrt 13))) ∨ 
  (Circle_eq_2 : Circle_eq _ (Circle (5, 4) (Real.sqrt 37))) :=
sorry

-- Problem 2: Given conditions and final answer
theorem problem2 (C : Point ℝ) 
  (hC : C.1 + C.2 = 0)
  (circle1 circle2 : Circle ℝ)
  (circle_eq1 : circle1 = Circle_eq _ (Circle (2, 5) 3))
  (circle_eq2 : circle2 = Circle_eq _ (Circle (-1, -1) 5)) :
  (∃ (C' : Point ℝ) (r : ℝ), Circle_eq _ (Circle C' r)
    ∧ passes_through C' r ((-4, 0) : Point ℝ) (circle1, circle2))
  → Circle_eq _ (Circle (-3, 3) (Real.sqrt 10)) :=
sorry

end problem1_problem2_l55_55655


namespace series_sum_equals_one_sixth_l55_55360

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l55_55360


namespace conjugate_of_z_l55_55424

theorem conjugate_of_z (z : ℂ) (h : z - complex.I = (3 + complex.I) / (1 + complex.I)) : complex.conj z = 2 :=
sorry

end conjugate_of_z_l55_55424


namespace min_photos_required_l55_55827

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55827


namespace magnitude_of_a_minus_b_l55_55077

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55077


namespace farm_field_area_l55_55459

theorem farm_field_area
  (planned_daily_plough : ℕ)
  (actual_daily_plough : ℕ)
  (extra_days : ℕ)
  (remaining_area : ℕ)
  (total_days_hectares : ℕ → ℕ) :
  planned_daily_plough = 260 →
  actual_daily_plough = 85 →
  extra_days = 2 →
  remaining_area = 40 →
  total_days_hectares (total_days_hectares (1 + 2) * 85 + 40) = 312 :=
by
  sorry

end farm_field_area_l55_55459


namespace anna_spent_more_on_lunch_l55_55900

def bagel_cost : ℝ := 0.95
def cream_cheese_cost : ℝ := 0.50
def orange_juice_cost : ℝ := 1.25
def orange_juice_discount : ℝ := 0.32
def sandwich_cost : ℝ := 4.65
def avocado_cost : ℝ := 0.75
def milk_cost : ℝ := 1.15
def milk_discount : ℝ := 0.10

-- Calculate total cost of breakfast.
def breakfast_cost : ℝ := 
  let bagel_with_cream_cheese := bagel_cost + cream_cheese_cost
  let discounted_orange_juice := orange_juice_cost - (orange_juice_cost * orange_juice_discount)
  bagel_with_cream_cheese + discounted_orange_juice

-- Calculate total cost of lunch.
def lunch_cost : ℝ :=
  let sandwich_with_avocado := sandwich_cost + avocado_cost
  let discounted_milk := milk_cost - (milk_cost * milk_discount)
  sandwich_with_avocado + discounted_milk

-- Calculate the difference between lunch and breakfast costs.
theorem anna_spent_more_on_lunch : lunch_cost - breakfast_cost = 4.14 := by
  sorry

end anna_spent_more_on_lunch_l55_55900


namespace function_intersection_at_most_one_l55_55581

theorem function_intersection_at_most_one (f : ℝ → ℝ) (a : ℝ) :
  ∃! b, f b = a := sorry

end function_intersection_at_most_one_l55_55581


namespace sum_of_n_squared_plus_14n_minus_1328_is_perfect_square_l55_55933

theorem sum_of_n_squared_plus_14n_minus_1328_is_perfect_square :
  let T := ∑ n in finset.filter (λ n : ℕ, ∃ k : ℤ, (n * n + 14 * n - 1328 : ℤ) = k * k) (finset.range 10000) in
  T % 1000 = 729 :=
by
  /-
  T is the sum of all positive integers n such that n^2 + 14n - 1328 is a perfect square.
  We need to show that T % 1000 equals 729.
  -/
  sorry

end sum_of_n_squared_plus_14n_minus_1328_is_perfect_square_l55_55933


namespace map_scale_l55_55538

theorem map_scale (cm12_km90 : 12 * (1 / 90) = 1) : 20 * (90 / 12) = 150 :=
by
  sorry

end map_scale_l55_55538


namespace vector_magnitude_l55_55099

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55099


namespace perspective_square_area_l55_55841

theorem perspective_square_area (a b : ℝ) (ha : a = 4 ∨ b = 4) : 
  a * a = 16 ∨ (2 * b) * (2 * b) = 64 :=
by 
sorry

end perspective_square_area_l55_55841


namespace find_principal_amount_l55_55394

def interest_rate_first_year : ℝ := 0.10
def compounding_periods_first_year : ℕ := 2
def interest_rate_second_year : ℝ := 0.12
def compounding_periods_second_year : ℕ := 4
def diff_interest : ℝ := 12

theorem find_principal_amount (P : ℝ)
  (h1_first : interest_rate_first_year / (compounding_periods_first_year : ℝ) = 0.05)
  (h1_second : interest_rate_second_year / (compounding_periods_second_year : ℝ) = 0.03)
  (compounded_amount : ℝ := P * (1 + 0.05)^(compounding_periods_first_year) * (1 + 0.03)^compounding_periods_second_year)
  (simple_interest : ℝ := P * (interest_rate_first_year + interest_rate_second_year) / 2 * 2)
  (h_diff : compounded_amount - P - simple_interest = diff_interest) : P = 597.01 :=
sorry

end find_principal_amount_l55_55394


namespace min_photos_required_l55_55822

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55822


namespace min_photos_for_condition_l55_55813

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55813


namespace polynomial_value_l55_55410

open Real

def given_eq (x : ℝ) := (sqrt 3 + 1) * x = sqrt 3 - 1

theorem polynomial_value (x : ℝ) (hx : given_eq x) : x^4 - 5 * x^3 + 6 * x^2 - 5 * x + 4 = 3 :=
begin
  sorry
end

end polynomial_value_l55_55410


namespace new_ratio_alcohol_water_l55_55674

theorem new_ratio_alcohol_water (initial_ratio_alcohol_water : ℕ → ℕ → Prop)
    (alcohol_volume : ℕ) (added_water : ℕ) : 
    initial_ratio_alcohol_water 4 3 → 
    alcohol_volume = 10 → 
    added_water = 5 →
    initial_ratio_alcohol_water (alcohol_volume) (12.5) = 4 / 5 := 
by
  intros
  sorry

end new_ratio_alcohol_water_l55_55674


namespace triangles_area_ratio_l55_55965

-- Define the points and distances
variables (A M N P Q O : Type) 

-- Define the distances
variables (d_AP d_AQ d_AM d_AN : ℕ)
variables (h_AP : d_AP = 4)
variables (h_AQ : d_AQ = 12)
variables (h_AM : d_AM = 6)
variables (h_AN : d_AN = 10)

-- Define collinearity and intersection
variables (on_line_AM : A → M)
variables (on_line_AN : A → N)
variables (on_line_AP : A → P)
variables (on_line_AQ : A → Q)
variables (intersection_O : O = lines_intersect (line M Q) (line N P))

-- Prove the ratio of the areas of triangles MNO and PQO
theorem triangles_area_ratio :
    ∀ (A M N P Q O : Type) (d_AP d_AQ d_AM d_AN : ℕ)
    (h_AP : d_AP = 4) (h_AQ : d_AQ = 12) (h_AM : d_AM = 6) (h_AN : d_AN = 10)
    (on_line_AM : A → M) (on_line_AN : A → N)
    (on_line_AP : A → P) (on_line_AQ : A → Q) 
    (intersection_O : O = lines_intersect (line M Q) (line N P)),
    ratio_areas_triangle (triangle M N O) (triangle P Q O) = 1 / 5 := 
begin
    sorry,
end

end triangles_area_ratio_l55_55965


namespace no_unique_symbols_for_all_trains_l55_55689

def proposition (a b c d : Prop) : Prop :=
  (¬a ∧  b ∧ ¬c ∧  d)
∨ ( a ∧ ¬b ∧ ¬c ∧ ¬d)

theorem no_unique_symbols_for_all_trains 
    (a b c d : Prop)
    (p : proposition a b c d)
    (s1 : ¬a ∧  b ∧ ¬c ∧  d)
    (s2 :  a ∧ ¬b ∧ ¬c ∧ ¬d) : 
    False :=
by {cases s1; cases s2; contradiction}

end no_unique_symbols_for_all_trains_l55_55689


namespace cost_price_of_a_toy_l55_55288

theorem cost_price_of_a_toy 
    (total_selling_price : ℝ)         -- Rs. 18900
    (total_toys_sold : ℝ)             -- 18 toys
    (gain_toys_cost : ℝ)              -- cost price of 3 toys
    (selling_price_per_toy : ℝ = total_selling_price / total_toys_sold)
    (gain_per_toy : ℝ = gain_toys_cost / total_toys_sold)
    (x : ℝ)                           -- cost price of one toy
    (h : x + gain_per_toy = selling_price_per_toy) :
    x = 900 := sorry

end cost_price_of_a_toy_l55_55288


namespace chess_tournament_total_players_l55_55470

-- Define the conditions

def total_points_calculation (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 132

def games_played (n : ℕ) : ℕ :=
  ((n + 12) * (n + 11)) / 2

theorem chess_tournament_total_players :
  ∃ n, total_points_calculation n = games_played n ∧ n + 12 = 34 :=
by {
  -- Assume n is found such that all conditions are satisfied
  use 22,
  -- Provide the necessary equations and conditions
  sorry
}

end chess_tournament_total_players_l55_55470


namespace eight_diamond_three_l55_55397

def diamond (x y : ℤ) : ℤ := sorry

axiom diamond_zero (x : ℤ) : diamond x 0 = x
axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x
axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

theorem eight_diamond_three : diamond 8 3 = 39 :=
sorry

end eight_diamond_three_l55_55397


namespace inequality_solution_abs_b_gt_2_l55_55443

def f (x : ℝ) : ℝ := abs (x - 2)

theorem inequality_solution (x : ℝ) :
  f(x) + f(x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 := 
  sorry

theorem abs_b_gt_2 (a b : ℝ) (h_a : abs a > 1) (h_f : f (a * b) > abs a * f (b / a)) :
  abs b > 2 :=
  sorry

end inequality_solution_abs_b_gt_2_l55_55443


namespace prob_zero_to_two_l55_55144

-- Define conditions
def measurement_result (X : ℝ) (σ : ℝ) := 
  ∃ 𝒩 : (ℝ → ℝ), 𝒩 = Normal(1, σ^2) ∧ RandomVariable X 𝒩

def prob_X_less_zero (X : ℝ) := 
  ∃ p : ℝ, p = 0.2 ∧ P(X < 0) = p

-- Main statement
theorem prob_zero_to_two (X : ℝ) (σ : ℝ) 
  (hx : measurement_result X σ) 
  (hp : prob_X_less_zero X) : 
  P(0 < X < 2) = 0.6 := 
sorry

end prob_zero_to_two_l55_55144


namespace quadrilateralInCircle_canBeInscribed_l55_55298

variables {P : Type} [circle : set P] [convex_quad : set P]
variables (A1 A2 B1 B2 C1 C2 D1 D2 : P)

noncomputable def canBeInscribed (A1 A2 B1 B2 C1 C2 D1 D2 : P) :=
  ∀ (θ₁ θ₂ : ℝ),
    extendedSidesIntersectCircle A1 A2 B1 B2 C1 C2 D1 D2 ∧
    A1B2_eq_B1C2_eq_C1D2_eq_D1A2 A1 A2 B1 B2 C1 C2 D1 D2 →
    (oppositeAnglesSum θ₁ θ₂ = π → inscribableInCircle {A1, A2, B1, B2, C1, C2, D1, D2})

-- Define the conditions predicates assumed in the problem statement
axiom extendedSidesIntersectCircle : 
  Π (A1 A2 B1 B2 C1 C2 D1 D2 : P), Prop

axiom A1B2_eq_B1C2_eq_C1D2_eq_D1A2 : 
  Π (A1 A2 B1 B2 C1 C2 D1 D2 : P), Prop

axiom oppositeAnglesSum : 
  Π (θ₁ θ₂ : ℝ), θ₁ + θ₂

-- The final statement of our problem in Lean
theorem quadrilateralInCircle_canBeInscribed 
  (A1 A2 B1 B2 C1 C2 D1 D2 : P)
  : canBeInscribed A1 A2 B1 B2 C1 C2 D1 D2 :=
sorry

end quadrilateralInCircle_canBeInscribed_l55_55298


namespace value_of_expression_l55_55516

theorem value_of_expression :
  let x := 1
  let y := -1
  let z := 0
  2 * x + 3 * y + 4 * z = -1 :=
by
  sorry

end value_of_expression_l55_55516


namespace eyes_per_ant_proof_l55_55533

noncomputable def eyes_per_ant (s a e_s E : ℕ) : ℕ :=
  let e_spiders := s * e_s
  let e_ants := E - e_spiders
  e_ants / a

theorem eyes_per_ant_proof : eyes_per_ant 3 50 8 124 = 2 :=
by
  sorry

end eyes_per_ant_proof_l55_55533


namespace tetrahedron_shortest_edge_length_l55_55212

theorem tetrahedron_shortest_edge_length :
  ∃ (A B C D : Type) 
    (AB AC AD BC BD CD : ℝ) 
    (φ : ℝ), 
    (AB = 1) ∧ 
    (AC = (sqrt (1 - φ) * sqrt (1 + φ) / 2)) ∧ 
    (AD = AC) ∧ 
    (BD = AC) ∧ 
    (BC = sqrt (2) * φ) ∧ 
    (CD = (sqrt (1 - φ) * sqrt (1 + φ) * φ / 2)^2 / sqrt 2) ∧ 
    (φ = (sqrt 5 - 1) / 2) :=
begin
  sorry
end

end tetrahedron_shortest_edge_length_l55_55212


namespace points_collinear_l55_55898

def is_equilateral (A B C : ℂ) : Prop :=
  ∥B - A∥ = ∥C - B∥ ∧ ∥C - B∥ = ∥C - A∥ ∧ (B - A) * (C - B) = B * C - A * C

def collinear (A B C : ℂ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ C = A + k * (B - A)

theorem points_collinear
  {A B C D P Q R : ℂ}
  (h1 : |A - D| = |B - C|)
  (h2 : ∠A + ∠B = 120)
  (h3 : is_equilateral A C P)
  (h4 : is_equilateral D C Q)
  (h5 : is_equilateral D B R) :
  collinear P Q R :=
sorry

end points_collinear_l55_55898


namespace chess_tournament_games_l55_55895

def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games (n : ℕ) (h : n = 19) : games_played n = 171 :=
by
  rw [h]
  sorry

end chess_tournament_games_l55_55895


namespace conjugate_of_z_l55_55423

theorem conjugate_of_z (z : ℂ) (h : z - complex.I = (3 + complex.I) / (1 + complex.I)) : complex.conj z = 2 :=
sorry

end conjugate_of_z_l55_55423


namespace part_one_part_two_l55_55439

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 0 then -x^2 + 2 * x else if x = 0 then 0 else x^2 + m * x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem part_one (m : ℝ) (h : is_odd_function (λ x, f x m)) : m = 2 :=
by
  sorry

def is_monotonically_increasing (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ x y ∈ s, x ≤ y → f x ≤ f y

theorem part_two
  (m : ℝ) (h : m = 2)
  (a : ℝ) :
  is_monotonically_increasing (λ x, f x m) {x | -1 ≤ x ∧ x ≤ a - 2} →
  1 < a ∧ a ≤ 3 :=
by
  sorry

end part_one_part_two_l55_55439


namespace find_positive_integer_with_conditions_l55_55774

theorem find_positive_integer_with_conditions :
  ∃ (N : ℕ), (N > 0) ∧ (Nat.factorization N).sum = 15 ∧
  ∃ (d : ℕ → ℕ),
    (∀ n, d n = Nat.divisors N !! n) ∧
    (1 = d 0) ∧
    (2 = d 1) ∧
    -- explicitly capture the conditions extracted from the problem
    (∀ i, (i ≥ 0) ∧ (i < 16) → d i = N.divisors !! i) ∧
    -- Relational condition
    (∃ k, k = d 4 ∧ d k = ((d 1 + d 3) * d 5)) ∧
    -- Concluding with final proven N
    (N = 2002) :=
begin
  sorry
end

end find_positive_integer_with_conditions_l55_55774


namespace problem1_problem2_l55_55745

theorem problem1 :
  sqrt 9 - (-2023)^0 + 2^(-1 : ℤ) = 5 / 2 := by
  sorry

theorem problem2 (a b : ℝ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2 * b)) = 2 / (a + b) := by
  sorry

end problem1_problem2_l55_55745


namespace sum_of_a_and_b_l55_55729

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55729


namespace greatest_value_sum_eq_24_l55_55726

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55726


namespace vector_magnitude_subtraction_l55_55008

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55008


namespace sum_cos_sin_roots_correct_l55_55219

noncomputable def proof_sum_cos_sin_roots : ℝ :=
  ∑ x in {y : ℝ | cos (2 * y) + cos (6 * y) + 2 * sin y ^ 2 = 1 ∧ (5 * Real.pi / 6) ≤ y ∧ y ≤ Real.pi}, x

theorem sum_cos_sin_roots_correct :
  abs (proof_sum_cos_sin_roots - 2.88) < 0.01 :=
sorry

end sum_cos_sin_roots_correct_l55_55219


namespace vector_magnitude_correct_l55_55041

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55041


namespace problem_l55_55948

def alpha := 3 + 2 * Real.sqrt 2
def x := alpha ^ 1000
def n := Int.floor x
def f := x - n

theorem problem : x * (1 - f) = 1 := by
  sorry

end problem_l55_55948


namespace order_of_a_b_c_l55_55406

noncomputable def a : ℝ := Real.log 3 / Real.log (1 / 2)
noncomputable def b : ℝ := Real.exp (1 / 2)
noncomputable def c : ℝ := Real.log 2 / Real.log 10

theorem order_of_a_b_c (h1: a = Real.log 3 / Real.log (1 / 2))
                        (h2: b = Real.exp (1 / 2))
                        (h3: c = Real.log 2 / Real.log 10) :
  a < c ∧ c < b := 
by {
  have ha : a < 0 := sorry,
  have hb : b > 1 := sorry,
  have hc1 : 0 < c := sorry,
  have hc2 : c < 1 := sorry,
  exact ⟨ha.trans hc1, hc2.trans hb⟩
}

end order_of_a_b_c_l55_55406


namespace simplify_expression_l55_55553

theorem simplify_expression :
  let a := 2
  let b := -3
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := sorry

end simplify_expression_l55_55553


namespace minimum_number_of_valid_subsets_l55_55189

open Finset

-- Problem setting and conditions:
def S : Finset ℕ := (finset.range 15).map (finset.add_monoid_hom 1)

def valid_subset (A : Finset (Finset ℕ)) : Prop :=
  ∀ a ∈ A, a.card = 7 ∧
  ∀ b ∈ A, (∑ x in a ∩ b, 1 : ℕ) ≤ 3 ∧
  ∀ M ∈ (S.powerset.filter (λ x, x.card = 3)), ∃ a ∈ A, M ⊆ a

-- Statement of the problem:
theorem minimum_number_of_valid_subsets :
  ∃ A : Finset (Finset ℕ), valid_subset A ∧ A.card = 15 :=
sorry

end minimum_number_of_valid_subsets_l55_55189


namespace vector_magnitude_subtraction_l55_55063

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55063


namespace find_ab_l55_55836

variable (a b : ℝ)
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : Real.log a b + Real.log b a = 10 / 3)
variable (h4 : a ^ b = b ^ a)

theorem find_ab : a * b = 9 := 
by 
  sorry

end find_ab_l55_55836


namespace number_of_possible_values_for_b_l55_55225

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 10 ∧ ∀ (b : ℕ), (2 ≤ b) ∧ (b^2 ≤ 256) ∧ (256 < b^3) ↔ (7 ≤ b ∧ b ≤ 16) :=
by {
  sorry
}

end number_of_possible_values_for_b_l55_55225


namespace min_photos_needed_to_ensure_conditions_l55_55805

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55805


namespace transform_parabola_l55_55183

theorem transform_parabola (a b c : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f (a * x^2 + b * x + c) = x^2) :=
sorry

end transform_parabola_l55_55183


namespace product_of_slopes_max_area_ABM_l55_55433

noncomputable def Ellipse := {a > b > 0 : Real} (x y : Real)

def intersect_point_M (E : Ellipse) : Real × Real := (0, sqrt 3)

def foci (E : Ellipse) (c : Real) : (Real × Real) × (Real × Real) :=
  ((-c, 0), (c, 0))

theorem product_of_slopes {a b k : Real} (h : a = 2) (h1 : b = sqrt 3) (M : Real × Real) (F1 F2 : Real × Real) :
  let E := Ellipse (x y : Real) in
  let l := λ x, k * x + 2 * sqrt 3 in
  ∀ (A B : Real × Real),
  (E : intersect_point_M E = M) →
  (foci E 1 = (F1, F2)) →
  -- Define conditions for A and B corresponding to line intersecting ellipse E
  -- Proof for slopes product of lines MA and MB
  let kMA := (A.snd - M.snd) / (A.fst - M.fst) in
  let kMB := (B.snd - M.snd) / (B.fst - M.fst) in
  kMA * kMB = 1 / 4 := sorry

theorem max_area_ABM {a b k : Real} (h : a = 2) (h1 : b = sqrt 3) :
  let E := Ellipse (x y : Real) in
  let l := λ x, k * x + 2 * sqrt 3 in
  ∀ (A B M : Real × Real),
  (E : intersect_point_M E = M) →
  -- Define conditions for A and B corresponding to line intersecting ellipse E
  -- Proof for maximum area of triangle ABM
  let distance_M_to_l := sqrt 3 / sqrt (k ^ 2 + 1) in
  let area_ABM := (1 / 2) * distance_M_to_l * -- |AB| computation from the solution steps
                 in
  area_ABM ≤ sqrt 3 / 2 := sorry

end product_of_slopes_max_area_ABM_l55_55433


namespace square_side_length_l55_55647

theorem square_side_length (s : ℝ) (h : s^2 + s - 4 * s = 4) : s = 4 :=
sorry

end square_side_length_l55_55647


namespace total_percentage_local_students_l55_55146

def students_data : Type :=
  Σ (num_local : ℕ), ℝ → ℕ

noncomputable def arts_students : students_data := ⟨480, 0.55⟩
noncomputable def science_students : students_data := ⟨150, 0.35⟩
noncomputable def commerce_students : students_data := ⟨200, 0.75⟩
noncomputable def humanities_students : students_data := ⟨100, 0.45⟩
noncomputable def engineering_students : students_data := ⟨250, 0.60⟩

theorem total_percentage_local_students :
  let total_students := arts_students.1 + science_students.1 + commerce_students.1 + humanities_students.1 + engineering_students.1 in
  let total_local_students := round (arts_students.2 * arts_students.1 + science_students.2 * science_students.1 +
                            commerce_students.2 * commerce_students.1 + humanities_students.2 * humanities_students.1 +
                            engineering_students.2 * engineering_students.1) in
  (total_local_students.toFloat / total_students.toFloat) * 100 ≈ 56.02 :=
by
  sorry

end total_percentage_local_students_l55_55146


namespace marble_arrangements_l55_55286

theorem marble_arrangements : 
  ∃ (n : ℕ), n = 12 ∧ 
  ∀ (m1 m2 m3 m4 : ℕ), 
    (List.permutations [m1, m2, m3, m4]).count (fun l => 
      let m_list := [m1 = 1, m2 = 2, m3 = 3, m4 = 4] in
      l ~ m_list ∧ 
      ¬(m_list.nth_le 2 (by decide) = 3 ∧ m_list.nth_le 3 (by decide) = 4) ∧ 
      ¬(m_list.nth_le 3 (by decide) = 3 ∧ m_list.nth_le 2 (by decide) = 4)) = n :=
begin
  use 12,
  sorry
end

end marble_arrangements_l55_55286


namespace smallest_prime_p_l55_55612

theorem smallest_prime_p (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h_sum : p + q + r = 25) : p = 2 := 
sorry

end smallest_prime_p_l55_55612


namespace one_fourth_in_one_eighth_l55_55119

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l55_55119


namespace valid_inequalities_l55_55281

theorem valid_inequalities :
  (∀ x : ℝ, x^2 + 6x + 10 > 0) ∧ (∀ x : ℝ, -x^2 + x - 2 < 0) := by
  sorry

end valid_inequalities_l55_55281


namespace find_abc_l55_55758

noncomputable theory

def matrixN (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![a, b, c], 
    ![b, -a, 0], 
    ![c, 0, -a]]

def isOrthogonal (N : matrix (fin 3) (fin 3) ℝ) : Prop :=
  Nᵀ ⬝ N = 1

theorem find_abc (a b c : ℝ) 
  (h : let N := matrixN a b c in isOrthogonal N) : a ^ 2 + b ^ 2 + c ^ 2 = 1 :=
sorry

end find_abc_l55_55758


namespace problem1_problem2_l55_55744

theorem problem1 :
  sqrt 9 - (-2023)^0 + 2^(-1 : ℤ) = 5 / 2 := by
  sorry

theorem problem2 (a b : ℝ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2 * b)) = 2 / (a + b) := by
  sorry

end problem1_problem2_l55_55744


namespace negation_of_proposition_l55_55364

theorem negation_of_proposition (x : ℝ) (h_pos : 0 < x) :
  (¬ (∀ m ∈ (set.Icc 0 1), x + 1/x ≥ 2^m)) ↔ 
  (∃ m ∈ (set.Icc 0 1), x + 1/x < 2^m) :=
by
  sorry

end negation_of_proposition_l55_55364


namespace prove_logical_proposition_l55_55460

theorem prove_logical_proposition (p q : Prop) (hp : p) (hq : ¬q) : (¬p ∨ ¬q) :=
by
  sorry

end prove_logical_proposition_l55_55460


namespace chessboard_sum_min_l55_55261

/-- Let n be a natural number, and a be a real number. Suppose for every field 
on an n x n chess board with real numbers, the sum of the numbers in the union 
of any row and column (a "cross") is at least a. Prove that the smallest 
possible sum of all numbers on the board is at least n^2 * a / (2 * n - 1) 
and achievable. -/
theorem chessboard_sum_min (n : ℕ) (a : ℝ) (S : ℕ → ℝ → ℝ) (sum_cross : (∀ x y : ℕ, 1 ≤ x ∧ x ≤ n → 1 ≤ y ∧ y ≤ n → S x y ≥ a)) : 
(∃ Smin, (Smin = n^2 * a / (2 * n - 1)) ∧ (∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → (sum_cross i j (Smin / (n * (2*n - 1))) S)) :=
begin
  sorry,
end

end chessboard_sum_min_l55_55261


namespace maximum_side_length_l55_55562

theorem maximum_side_length 
    (D E F : ℝ) 
    (a b c : ℝ) 
    (h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1)
    (h_a : a = 12)
    (h_perimeter : a + b + c = 40) : 
    ∃ max_side : ℝ, max_side = 7 + Real.sqrt 23 / 2 :=
by
  sorry

end maximum_side_length_l55_55562


namespace total_students_correct_l55_55536

theorem total_students_correct (H : ℕ)
  (B : ℕ := 2 * H)
  (P : ℕ := H + 5)
  (S : ℕ := 3 * (H + 5))
  (h1 : B = 30)
  : (B + H + P + S) = 125 := by
  sorry

end total_students_correct_l55_55536


namespace one_fourth_in_one_eighth_l55_55123

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l55_55123


namespace exists_strictly_increasing_seq_tan_sum_arctan_eq_zero_l55_55783

theorem exists_strictly_increasing_seq_tan_sum_arctan_eq_zero 
  (m : ℕ) (h : m ≥ 3 ∧ m % 2 = 1) : 
  ∃ (a : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ m → a n > 0) ∧ 
                  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ m → a i < a j) ∧ 
                  ( ∑ n in Finset.range m, Real.arctan (a n) = ↑(n:ℤ) * Real.pi) :=
sorry

end exists_strictly_increasing_seq_tan_sum_arctan_eq_zero_l55_55783


namespace four_digit_numbers_count_l55_55930

theorem four_digit_numbers_count : 
  let A := 9 * 10 * 10 * 5 in
  let B := 9 * 10 * 10 * 1 in
  A + B = 5400 :=
by
  let A := 9 * 10 * 10 * 5
  let B := 9 * 10 * 10 * 1
  show A + B = 5400
  sorry

end four_digit_numbers_count_l55_55930


namespace trapezium_area_l55_55644

def area_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area :
  ∀ (a b h : ℕ), a = 20 → b = 18 → h = 11 → area_trapezium a b h = 209 :=
begin
  intros a b h ha hb hh,
  rw [ha, hb, hh],
  -- sorry will be replaced with an actual proof
  sorry,
end

end trapezium_area_l55_55644


namespace triangle_angles_l55_55868

theorem triangle_angles (A B C P : Point)
  (h1 : ∠BAP = 18°)
  (h2 : ∠CAP = 30°)
  (h3 : ∠ACP = 48°)
  (h4 : dist A P = dist B C) :
  ∠BCP = 6° :=
sorry

end triangle_angles_l55_55868


namespace polar_coordinates_equiv_l55_55151

theorem polar_coordinates_equiv :
  ∃ (r θ : ℝ), r > 0 ∧ (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (r = 3 ∧ θ = 7 * Real.pi / 6) :=
by
  -- Given point in polar coordinates (-3, π/6)
  let r := -3
  let θ := Real.pi / 6

  -- Convert radius to positive and adjust angle
  let r' := -r  -- r' = 3
  let θ' := θ + Real.pi  -- θ' = 7π/6

  exists (r')
  exists (θ')
  split
  show r' > 0, from by
    simp [r']
  split
  split
  show 0 ≤ θ', from by
    simp [θ']
    linarith
  show θ' < 2 * Real.pi, from by
    simp [θ']
    linarith
  simp [r', θ']
  show r' = 3 ∧ θ' = 7 * Real.pi / 6, from by
    simp [r', θ'] 
    split
    linarith
    linarith

end polar_coordinates_equiv_l55_55151


namespace ratio_of_segments_l55_55142

theorem ratio_of_segments (A B C P: Type) [InHabited A B C P ] (AC BC PA PB: ℝ) 
  (h1: AC : BC = 2 : 5) 
  (h2: IsAngleBisector A P B) : PA / PB = 2 / 5 := by
  sorry

end ratio_of_segments_l55_55142


namespace real_solutions_l55_55384

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l55_55384


namespace num_integers_satisfying_inequality_is_6_l55_55877

-- Define the inequality condition
def inequality (n : ℤ) : Prop :=
  (n - 3) * (n + 5) * (n - 1) < 0

-- Define the mathematically equivalent proof problem
-- Prove that there exist exactly 6 integers satisfying the inequality
theorem num_integers_satisfying_inequality_is_6 :
  {n : ℤ | inequality n}.to_finset.card = 6 := by
  sorry

end num_integers_satisfying_inequality_is_6_l55_55877


namespace evaluate_expression_l55_55374

theorem evaluate_expression (x c : ℕ) (h1 : x = 3) (h2 : c = 2) : 
  ((x^2 + c)^2 - (x^2 - c)^2) = 72 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l55_55374


namespace vec_magnitude_is_five_l55_55050

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55050


namespace real_solutions_l55_55385

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l55_55385


namespace sum_when_max_power_less_500_l55_55721

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55721


namespace minimum_photos_l55_55818

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55818


namespace perfume_weight_is_six_ounces_l55_55498

def weight_in_pounds (ounces : ℕ) : ℕ := ounces / 16

def initial_weight := 5  -- Initial suitcase weight in pounds
def final_weight := 11   -- Final suitcase weight in pounds
def chocolate := 4       -- Weight of chocolate in pounds
def soap := 2 * 5        -- Weight of 2 bars of soap in ounces
def jam := 2 * 8         -- Weight of 2 jars of jam in ounces

def total_additional_weight :=
  chocolate + (weight_in_pounds soap) + (weight_in_pounds jam)

def perfume_weight_in_pounds := final_weight - initial_weight - total_additional_weight

def perfume_weight_in_ounces := perfume_weight_in_pounds * 16

theorem perfume_weight_is_six_ounces : perfume_weight_in_ounces = 6 := by sorry

end perfume_weight_is_six_ounces_l55_55498


namespace part_I_answer_part_II_answer_part_III_answer_l55_55534

section problem

variables (num_male_students num_female_students : ℕ)
          (prop_male_table_tennis prop_female_table_tennis : ℚ)
          (prop_male_jump_rope prop_female_jump_rope : ℚ)
          (scores_male : list ℚ) (scores_female : list ℚ)

-- Define the number of students and proportions
def num_students : ℕ := num_male_students + num_female_students

-- Proportions of students choosing different activities
def num_male_table_tennis : ℕ := (prop_male_table_tennis * num_male_students).to_nat
def num_female_table_tennis : ℕ := (prop_female_table_tennis * num_female_students).to_nat
def total_table_tennis : ℕ := num_male_table_tennis + num_female_table_tennis

-- Calculate given answer for Part (I)
theorem part_I_answer 
  (prop_male_table_tennis : prop_male_table_tennis = 0.1)
  (prop_female_table_tennis : prop_female_table_tennis = 0.05)
  : total_table_tennis.to_nat / num_students.to_rat = 8/105 := sorry

-- Probability calculations for Part (II)
def prob_male_jump_rope := prop_male_jump_rope
def prob_female_jump_rope := prop_female_jump_rope

theorem part_II_answer
  (prop_male_jump_rope : prop_male_jump_rope = 0.4)
  (prop_female_jump_rope : prop_female_jump_rope = 0.5)
  : (2 * prob_male_jump_rope * (1 - prob_male_jump_rope) * prob_female_jump_rope) + (prob_male_jump_rope ^ 2 * (1 - prob_female_jump_rope)) = 0.32 := sorry

-- Average score calculations for Part (III)
def avg_score_all : ℚ := (list.sum scores_male + list.sum scores_female) / (scores_male.length + scores_female.length)
def avg_score_male : ℚ := list.sum scores_male / scores_male.length

theorem part_III_answer
  (h_scores_male : scores_male = [repeat 8 60, repeat 7.5 40, repeat 7 10].join)
  (h_scores_female : scores_female = [repeat 8 40, repeat 7 10].join)
  : avg_score_all > avg_score_male := sorry

end problem

end part_I_answer_part_II_answer_part_III_answer_l55_55534


namespace sector_to_cone_volume_l55_55431

theorem sector_to_cone_volume (θ : ℝ) (A : ℝ) (V : ℝ) (l r h : ℝ) :
  θ = (2 * Real.pi / 3) →
  A = (3 * Real.pi) →
  A = (1 / 2 * l^2 * θ) →
  θ = (r / l * 2 * Real.pi) →
  h = Real.sqrt (l^2 - r^2) →
  V = (1 / 3 * Real.pi * r^2 * h) →
  V = (2 * Real.sqrt 2 * Real.pi / 3) :=
by
  intros hθ hA hAeq hθeq hh hVeq
  sorry

end sector_to_cone_volume_l55_55431


namespace polynomial_degree_and_type_l55_55584

def polynomial := 1 - x^2 - 5 * x^4

theorem polynomial_degree_and_type : degree polynomial = 4 ∧ trinomial polynomial := 
sorry

end polynomial_degree_and_type_l55_55584


namespace vector_magnitude_subtraction_l55_55025

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55025


namespace combined_pastures_feed_250_cattle_for_28_days_l55_55637

theorem combined_pastures_feed_250_cattle_for_28_days : 
  (3 : ℝ) * (1 + 36 * (1 / 72)) / (90 * 36 * (1 / 720)) + (4 : ℝ) * (1 + 24 * (1 / 72)) / (160 * 24 * (1 / 720)) = 28 / 250  := 
by 
  have h1 : (1 + 36 * (1 / 72)) = 1 + 0.5 := by sorry
  have h2 : (1 + 24 * (1 / 72)) = 1 + 24 / 72 := by sorry
  have h3 : (90 * 36 * (1 / 720)) = 4.5 := by sorry
  have h4 : (160 * 24 * (1 / 720)) = 5.33 := by sorry
  have res:= h1 + h2 = 6 * (1 + 7*(1 / 72)) / (250 * 28 * (1 / 720)) 
  sorry


end combined_pastures_feed_250_cattle_for_28_days_l55_55637


namespace minimum_of_fraction_sum_l55_55960

noncomputable def minimum_value_of_expression (a b c d : ℝ) : ℝ :=
  (b / (c + d)) + (c / (a + b))

theorem minimum_of_fraction_sum (a b c d : ℝ) (h1 : a ≥ 0) (h2 : d ≥ 0) (h3 : b > 0) (h4 : c > 0) (h5 : b + c ≥ a + d) : 
  minimum_value_of_expression a b c d = sqrt 2 - 0.5 :=
by
  sorry

end minimum_of_fraction_sum_l55_55960


namespace parametric_equations_C2_distance_AB_l55_55154

-- Definition of the parametric equations of C1
def C1 (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 + Real.sin α)

-- Definition of the parametric equations of C2
def C2 (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 + 3 * Real.sin α)

-- Proof goal 1: The equations of C2 given conditions
theorem parametric_equations_C2 (α : ℝ) : 
  (C2 α) = (3 * Real.cos α, 3 + 3 * Real.sin α) :=
sorry

-- Polar coordinate form of C1 and C2 for intersection calculation
def C1_polar (θ : ℝ) : ℝ := 2 * Real.sin θ
def C2_polar (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Intersection of line y = sqrt(3)/3*x with C1 and C2
-- Proof goal 2: The distance between points A and B is 2
theorem distance_AB : 
  let θ := Real.pi / 6,
  let ρ_A := C1_polar θ,
  let ρ_B := C2_polar θ,
  |ρ_A - ρ_B| = 2 :=
sorry

end parametric_equations_C2_distance_AB_l55_55154


namespace ratio_of_inscribed_squares_in_isosceles_right_triangle_l55_55331

def isosceles_right_triangle (a b : ℝ) (leg : ℝ) : Prop :=
  let a_square_inscribed := a = leg
  let b_square_inscribed := b = leg
  a_square_inscribed ∧ b_square_inscribed

theorem ratio_of_inscribed_squares_in_isosceles_right_triangle (a b leg : ℝ)
  (h : isosceles_right_triangle a b leg) :
  leg = 6 ∧ a = leg ∧ b = leg → a / b = 1 := 
by {
  sorry -- the proof will go here
}

end ratio_of_inscribed_squares_in_isosceles_right_triangle_l55_55331


namespace sin_2A_minus_C_l55_55919

theorem sin_2A_minus_C (a b c A B C : ℝ) (h1 : 7 * b ^ 2 + 25 * c ^ 2 - 25 * a ^ 2 = 0) 
(h_triangle: ∀ x y z : ℝ, tanglear(x, y, z)) : 
  ∃ A' B' C' b' c' a', sin (2 * A' - C') = 117 / 125 := 
begin
  sorry
end

end sin_2A_minus_C_l55_55919


namespace tangent_line_g_at_1_eq_monotonic_intervals_f_l55_55437

noncomputable def g (x : ℝ) : ℝ := 1/x + 1/2
noncomputable def tangent_line_g_at_1 : ℝ → ℝ := λ x, - (x - 1) + 3/2

-- Part I
theorem tangent_line_g_at_1_eq : ∀ x : ℝ, tangent_line_g_at_1 x = -(x-1) + 3/2 :=
by sorry

noncomputable def f (x : ℝ) : ℝ := (Real.log x)/x

-- Part II
theorem monotonic_intervals_f : 
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → f x ≥ f 1) ∧ 
  (∀ x : ℝ, x > Real.exp 1 → f x < f 1) ∧ 
  2017^(1/2017 : ℝ) < 2016^(1/2016 : ℝ) :=
by sorry

end tangent_line_g_at_1_eq_monotonic_intervals_f_l55_55437


namespace minimal_volume_block_l55_55322

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block_l55_55322


namespace minimum_photos_l55_55815

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55815


namespace line_PF_max_dot_product_l55_55857

-- Definitions of the conditions.
def curve (x y : ℝ) : Prop := x^2 = 8 * y
def focus : (ℝ × ℝ) := (0, 2)
def on_directrix (P : ℝ × ℝ) : Prop := P.2 = -2

def midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

noncomputable def vector_equality (A B C : ℝ × ℝ) : Prop :=
  (C.1 - B.1) = (B.1 - A.1) ∧ (C.2 - B.2) = (B.2 - A.2)

-- Problem 1: Prove the equation of line PF.
theorem line_PF (P E : ℝ × ℝ) (hP : on_directrix P) (hF : focus = (0, 2))
  (hM : vector_equality P (0, 2) E) (hE_quadrant : 0 < E.1 ∧ 0 < E.2) :
  ∃ a b c : ℝ, a * E.1 + b * E.2 + c = 0 ∧ a = 1 ∧ b = -√3 ∧ c = 2 * √3 := sorry

-- Problem 2: Prove the maximum value of the dot product and coordinates of P.
theorem max_dot_product (P D E : ℝ × ℝ) (hP : on_directrix P)
  (h_curve_D : curve D.1 D.2) (h_curve_E : curve E.1 E.2)
  (h_line_PF : ∃ k : ℝ, E.2 = k * E.1 + 2 ∧ k ≠ 0) :
  ∃ (max_val : ℝ) (Px : ℝ × ℝ),
    max_val = -64 ∧ (Px = (4, -2) ∨ Px = (-4, -2)) := sorry

end line_PF_max_dot_product_l55_55857


namespace smallest_k_coprime_pair_l55_55403

theorem smallest_k_coprime_pair (S : Finset ℕ) (k : ℕ) (hS : S = Finset.range 101 \ Finset.singleton 0) :
    S.card = 100 → (∀ T ⊂ S, T.card = k → (∃ a b ∈ T, Nat.coprime a b)) ↔ k = 51 :=
by
  sorry

end smallest_k_coprime_pair_l55_55403


namespace fescue_in_Y_l55_55215

-- Define the weight proportions of the mixtures
def weight_X : ℝ := 0.6667
def weight_Y : ℝ := 0.3333

-- Define the proportion of ryegrass in each mixture
def ryegrass_X : ℝ := 0.40
def ryegrass_Y : ℝ := 0.25

-- Define the proportion of ryegrass in the final mixture
def ryegrass_final : ℝ := 0.35

-- Define the proportion of ryegrass contributed by X and Y to the final mixture
def contrib_X : ℝ := weight_X * ryegrass_X
def contrib_Y : ℝ := weight_Y * ryegrass_Y

-- Define the total proportion of ryegrass in the final mixture
def total_ryegrass : ℝ := contrib_X + contrib_Y

-- The lean theorem stating that the percentage of fescue in Y equals 75%
theorem fescue_in_Y :
  total_ryegrass = ryegrass_final →
  (100 - (ryegrass_Y * 100)) = 75 := 
by
  intros h
  sorry

end fescue_in_Y_l55_55215


namespace angle_between_CK_and_AB_l55_55187

theorem angle_between_CK_and_AB:
  ∀ (O A B C M N K : Point),
  Circle O A B →
  is_diameter A B → 
  on_plane C →
  (∃ M N, (on_circle C M) ∧ (on_circle C N) ∧ intersect_line O A C ↔ M ∧ intersect_line O B C ↔ N) →
  (∃ K, intersect_line M B K ∧ intersect_line N A K) →
  angle CK AB = 90 :=
by sorry

end angle_between_CK_and_AB_l55_55187


namespace min_f_iter_5_on_interval_l55_55407

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  nat.rec_on n f (λ _ y, f y) x

def interval : Set ℝ := Set.Icc 1/2 1

theorem min_f_iter_5_on_interval :
  ∃ (min_val : ℝ), min_val = 1/12 ∧ (∀ x ∈ interval, f_iter 5 x ≥ min_val) := sorry

end min_f_iter_5_on_interval_l55_55407


namespace vector_magnitude_correct_l55_55036

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55036


namespace find_n_when_x3_yneg1_l55_55192

theorem find_n_when_x3_yneg1 (x y : ℤ) (h1 : x = 3) (h2 : y = -1) :
  let n := x - y^(x + y) in
  n = 2 :=
by
  intro n hn
  rw [h1, h2] at hn
  simp at hn
  exact hn

end find_n_when_x3_yneg1_l55_55192


namespace race_order_l55_55297

theorem race_order (overtakes_G_S_L : (ℕ × ℕ × ℕ))
  (h1 : overtakes_G_S_L.1 = 10)
  (h2 : overtakes_G_S_L.2.1 = 4)
  (h3 : overtakes_G_S_L.2.2 = 6)
  (h4 : ¬(overtakes_G_S_L.2.1 > 0 ∧ overtakes_G_S_L.2.2 > 0))
  (h5 : ∀ i j k : ℕ, i ≠ j → j ≠ k → k ≠ i)
  : overtakes_G_S_L = (10, 4, 6) :=
sorry

end race_order_l55_55297


namespace candles_on_rituprts_cake_l55_55206

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l55_55206


namespace mary_sugar_cups_l55_55198

theorem mary_sugar_cups (sugar_required : ℕ) (sugar_remaining : ℕ) (sugar_added : ℕ) (h1 : sugar_required = 11) (h2 : sugar_added = 1) : sugar_remaining = 10 :=
by
  -- Placeholder for the proof
  sorry

end mary_sugar_cups_l55_55198


namespace range_of_d_l55_55461

theorem range_of_d (d : ℝ) : (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 :=
sorry

end range_of_d_l55_55461


namespace convex_polygon_angles_l55_55546

theorem convex_polygon_angles (α β : ℝ) (h : convex_polygon 2001) (hα : α ∈ interior_angles h) (hβ : β ∈ interior_angles h) :
  ∃ α β, |cos α - cos β| < 1 / 2001^2 :=
by
  sorry

end convex_polygon_angles_l55_55546


namespace find_original_one_digit_number_l55_55311

theorem find_original_one_digit_number (x : ℕ) (h1 : x < 10) (h2 : (x + 10) * (x + 10) / x = 72) : x = 2 :=
sorry

end find_original_one_digit_number_l55_55311


namespace RupertCandles_l55_55203

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l55_55203


namespace vector_magnitude_difference_l55_55086

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55086


namespace ink_left_is_50_percent_l55_55617

variables (A1 A2 : ℕ)
variables (length width : ℕ)
variables (total_area used_area : ℕ)

-- Define the conditions
def total_area_of_squares := 3 * (4 * 4)
def total_area_of_rectangles := 2 * (6 * 2)
def ink_left_percentage := ((total_area_of_squares - total_area_of_rectangles) * 100) / total_area_of_squares

-- The theorem to prove
theorem ink_left_is_50_percent : ink_left_percentage = 50 :=
by
  rw [total_area_of_squares, total_area_of_rectangles]
  norm_num
  exact rfl
  sorry -- Proof omitted

end ink_left_is_50_percent_l55_55617


namespace twelve_factorial_mod_thirteen_l55_55399

theorem twelve_factorial_mod_thirteen : (12! % 13) = 12 := by
  sorry

end twelve_factorial_mod_thirteen_l55_55399


namespace problem_solution_l55_55411

-- Definitions and conditions
def circle (p : ℝ × ℝ) := (p.1 - 0)^2 + (p.2 - 3)^2 = 4
def line_m (p : ℝ × ℝ) := p.1 + 3 * p.2 + 6 = 0
def point_A : ℝ × ℝ := (1, 0)

-- Midpoint M of points P and Q
def midpoint (P Q : ℝ × ℝ) := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Distance between points P and Q
def distance (P Q : ℝ × ℝ) := ( ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt )

-- Definitions of vectors from A to M and A to N
def vector_AM (M : ℝ × ℝ) := (M.1 - point_A.1, M.2 - point_A.2)
def vector_AN (N : ℝ × ℝ) := (N.1 - point_A.1, N.2 - point_A.2)

-- Dot product
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- Main proof statement
theorem problem_solution :
  (exists l : ℝ → ℝ, -- equation of line l
    (∀ P Q : ℝ × ℝ, P ≠ Q → circle P → circle Q → l P.1 = P.2 → l Q.1 = Q.2 → distance P Q = 2 * Real.sqrt 3)
    ∨ (∀ P Q : ℝ × ℝ, P ≠ Q → circle P → circle Q → P.1 = -1 → Q.1 = -1)
  )
  ∧
  (∀ (P Q M N : ℝ × ℝ), midpoint P Q = M → line_m N →
    forall t, t = dot_product (vector_AM M) (vector_AN N) ∧ t = -5) :=
sorry

end problem_solution_l55_55411


namespace partition_subsets_sum_d_l55_55504

-- Define the conditions as given in a)
def is_divisor (a b : ℕ) : Prop := ∃ k, a * k = b

theorem partition_subsets_sum_d (n d : ℕ) (hn : 0 < n) (hd : d ≥ n) (hd_div : is_divisor d (n * (n + 1) / 2)) :
  ∃ (subsets : multiset (multiset ℕ)), 
    (∀ subset ∈ subsets, subset.sum = d) ∧ 
    (((⋃₀ subsets).erase_dup = multiset.range n (n+1)) : Prop) :=
sorry

end partition_subsets_sum_d_l55_55504


namespace more_stickers_correct_l55_55980

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23
def second_box_stickers : ℕ := total_stickers - first_box_stickers
def more_stickers_in_second_box : ℕ := second_box_stickers - first_box_stickers

theorem more_stickers_correct : more_stickers_in_second_box = 12 := by
  sorry

end more_stickers_correct_l55_55980


namespace class_b_students_l55_55347

theorem class_b_students (total_students : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total_students = 100 → sample_size = 10 → class_a_sample = 4 → 
  (total_students - total_students * class_a_sample / sample_size = 60) :=
by
  intros
  sorry

end class_b_students_l55_55347


namespace vector_magnitude_difference_l55_55016

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55016


namespace vector_magnitude_subtraction_l55_55034

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55034


namespace range_of_expression_l55_55248

noncomputable def quadratic_expression (x : ℝ) : ℝ :=
  x^2 + 3 * x + 4

theorem range_of_expression :
  ∀ (x : ℝ), 3 < x ∧ x < 5 → 
  22 < quadratic_expression x ∧ quadratic_expression x < 44 :=
by
  intros x hx -- Introduce x and our hypothesis hx
  have h1 : x^2 - 8 * x + 15 = (x - 3) * (x - 5) := by
    ring -- Proof for the factorization
  sorry -- Placeholder for the actual proof

end range_of_expression_l55_55248


namespace E_3_is_5_l55_55335

section E_function
variable {E : ℝ → ℝ}

-- Assumption corresponding to the condition that the graph shows (3, 5) is on E(x)
axiom E_graph_at_3 : E(3) = 5

-- The main statement we want to prove
theorem E_3_is_5 : E(3) = 5 :=
by sorry
end E_function

end E_3_is_5_l55_55335


namespace impossible_half_triangles_all_diagonals_l55_55471

theorem impossible_half_triangles_all_diagonals (n : ℕ) (hn : n = 2002)
    (triangles : ℕ) (h_triangles : triangles = 2000) :
    ¬ ∃ (k : ℕ), k = 1000 ∧
    ∀ (i : ℕ), i < triangles → 
    (if i < k then
        (∀ a b c : fin n, a ≠ b ∧ b ≠ c ∧ c ≠ a → 
        is_diagonal a b ∧ is_diagonal b c ∧ is_diagonal c a)
        else
        (∃ a b : fin n, a ≠ b ∧ ¬ is_diagonal a b)) :=
sorry

end impossible_half_triangles_all_diagonals_l55_55471


namespace max_power_sum_l55_55715

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55715


namespace vector_magnitude_difference_l55_55012

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55012


namespace ratio_pure_imaginary_l55_55962

theorem ratio_pure_imaginary (z1 z2 : ℂ) (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h : ∥z1 + z2∥ = ∥z1 - z2∥) : 
  ∃ k : ℝ, z1 / z2 = complex.I * k := 
sorry

end ratio_pure_imaginary_l55_55962


namespace ensure_eat_coveted_piece_l55_55531

theorem ensure_eat_coveted_piece :
  ∀ (cake: matrix (fin 8) (fin 8) ℕ), 
    (∀ i j, if cake i j = 1 then "fish" else "") ∧ 
    (∀ i j, if cake i j = 2 then "sausage" else "") ∧ 
    (∃ i j, cake i j = 3) -> 
    (∀ (a b: fin 6), 
      2 ≤ (finset.filter (λ (i, j), cake.to_fun (i + a) (j + b) = 1) (finset.finprod (fin 6) (fin 6))) -> 
    (∀ (a b: fin 3), 
      1 ≥ (finset.filter (λ (i, j), cake.to_fun (i + a) (j + b) = 2) (finset.finprod (fin 3) (fin 3))) -> 
    ∃ cake_pieces : finset ((fin 8) × (fin 8)), 
      cake_pieces.card ≥ 5 ∧ 
      (∃ (i, j), cake (i, j) = 3 → (i, j) ∈ cake_pieces)) :=
begin
  sorry
end

end ensure_eat_coveted_piece_l55_55531


namespace profit_ratio_l55_55246

theorem profit_ratio
  (investment_ratio_pq : ℕ → ℕ → Prop)
  (inv_ratio : investment_ratio_pq 7 5)
  (time_p : ℕ := 5)
  (time_q : ℕ := 13) :
  (investment_ratio_pq * time_p) / (investment_ratio_pq * time_q) = 7 / 13 :=
sorry

end profit_ratio_l55_55246


namespace find_interval_length_l55_55191

open Set Int

def lattice_points := { p : ℤ × ℤ | 1 ≤ p.fst ∧ p.fst ≤ 25 ∧ 1 ≤ p.snd ∧ p.snd ≤ 25 }

def points_below_line (n : ℚ) := { p : ℤ × ℤ | p ∈ lattice_points ∧ (p.snd : ℚ) ≤ n * (p.fst : ℚ) }

theorem find_interval_length :
  ∃ (c d : ℕ), RelPrime c d ∧ (∀ n : ℚ, (points_below_line n).card = 200 → (1/20 : ℚ) < n ∧ n < 4/5) ∧ c + d = 21 :=
sorry

end find_interval_length_l55_55191


namespace cost_price_percentage_l55_55570

def profit_percentage := 12.359550561797752 / 100

theorem cost_price_percentage (SP CP : ℝ):
  (SP - CP = profit_percentage * CP) →
  (CP / SP * 100 = 100 / 112.359550561797752) :=
sorry

end cost_price_percentage_l55_55570


namespace ratio_purely_imaginary_l55_55963

theorem ratio_purely_imaginary (z1 z2 : ℂ) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0)
  (h : |z1 + z2| = |z1 - z2|) : ∃ (c : ℂ), c.im ≠ 0 ∧ c.re = 0 ∧ c = z1 / z2 := by
  sorry

end ratio_purely_imaginary_l55_55963


namespace find_number_l55_55232

-- Define the number x and the condition as a theorem to be proven.
theorem find_number (x : ℝ) (h : (1/3) * x - 5 = 10) : x = 45 :=
sorry

end find_number_l55_55232


namespace min_photos_l55_55790

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55790


namespace vertex_value_assignment_l55_55692

-- Define the problem and its conditions
structure WeightedGraph (V : Type) :=
  (edges : V → V → ℝ)
  (distinct : ∀ {u v : V}, u ≠ v → edges u v > 0)
  (triangle_condition : ∀ {u v w : V}, u ≠ v → v ≠ w → u ≠ w → 
    edges u v = edges u w + edges w v ∨
    edges u w = edges u v + edges v w ∨
    edges w v = edges u v + edges u w)

-- Lean formal statement
theorem vertex_value_assignment (V : Type) [fintype V] [inhabited V]
  (G : WeightedGraph V) :
  ∃ (assign_values : V → ℝ), (∀ u v : V, u ≠ v → G.edges u v = |assign_values u - assign_values v|) :=
sorry

end vertex_value_assignment_l55_55692


namespace vector_magnitude_l55_55101

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55101


namespace arithmetic_floor_sum_l55_55753

theorem arithmetic_floor_sum :
  let seq := λ n : ℕ, 1 + n * 1.2 in
  let terms := (list.range 126).map seq in
  (terms.map floor).sum = 9526 :=
by
  -- Here we define the sequence with common difference 1.2 starting from 1
  let seq := λ n : ℕ, 1 + n * 1.2
  -- Compute the first 126 terms of the sequence
  let terms := (list.range 126).map seq
  -- Sum the floor values of the terms
  have final_sum : (terms.map floor).sum = 9526
  sorry

end arithmetic_floor_sum_l55_55753


namespace bird_families_difference_l55_55636

theorem bird_families_difference (initial_families : ℕ) (flew_away_families : ℕ) (families_remained : ℕ) (h1 : initial_families = 45) (h2 : flew_away_families = 86) (h3 : families_remained = 41) : flew_away_families - initial_families = families_remained :=
by
  rw [h1, h2, h3]
  sorry

end bird_families_difference_l55_55636


namespace unknown_card_value_l55_55268

theorem unknown_card_value (cards_total : ℕ)
  (p1_hand : ℕ) (p1_hand_extra : ℕ) (table_card1 : ℕ) (total_card_values : ℕ)
  (sum_removed_cards_sets : ℕ)
  (n : ℕ) :
  cards_total = 40 ∧ 
  p1_hand = 5 ∧ 
  p1_hand_extra = 3 ∧ 
  table_card1 = 9 ∧ 
  total_card_values = 220 ∧ 
  sum_removed_cards_sets = 15 * n → 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 10 ∧ total_card_values = p1_hand + p1_hand_extra + table_card1 + x + sum_removed_cards_sets → 
  x = 8 := 
sorry

end unknown_card_value_l55_55268


namespace handshake_problem_l55_55988

-- Defining the necessary elements:
def num_people : Nat := 12
def num_handshakes_per_person : Nat := num_people - 2

-- Defining the total number of handshakes. Each handshake is counted twice.
def total_handshakes : Nat := (num_people * num_handshakes_per_person) / 2

-- The theorem statement:
theorem handshake_problem : total_handshakes = 60 :=
by
  sorry

end handshake_problem_l55_55988


namespace complementary_sets_count_l55_55376

noncomputable def num_complementary_sets : ℕ :=
  let shapes := ["circle", "square"]
      colors := ["red", "blue"]
      shades := ["light", "dark"]
      deck := shapes.product (colors.product shades)
  in 28

theorem complementary_sets_count :
  let shapes := ["circle", "square"]
      colors := ["red", "blue"]
      shades := ["light", "dark"]
      deck := shapes.product (colors.product shades)
  in num_complementary_sets = 28 := 
by 
  sorry

end complementary_sets_count_l55_55376


namespace min_photos_exists_l55_55800

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55800


namespace surface_area_of_sphere_l55_55463

theorem surface_area_of_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 2)
  (h4 : ∀ d, d = Real.sqrt (a^2 + b^2 + c^2)) : 
  4 * Real.pi * (d / 2)^2 = 9 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l55_55463


namespace one_fourth_in_one_eighth_l55_55120

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l55_55120


namespace one_fourths_in_one_eighth_l55_55117

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l55_55117


namespace train_crossing_time_l55_55876

-- Define the problem-specific constants and conditions.
def length_train : ℝ := 110
def speed_train_kph : ℝ := 72
def length_bridge : ℝ := 132
def conversion_kph_to_mps : ℝ := 1000 / 3600

-- Calculate the speed of the train in m/s.
def speed_train_mps : ℝ := speed_train_kph * conversion_kph_to_mps

-- Calculate the total distance the train needs to cover.
def total_distance : ℝ := length_train + length_bridge

-- Calculate the time it takes to cross the bridge.
def time_to_cross : ℝ := total_distance / speed_train_mps

-- The theorem to be proven: The time taken is indeed 12.1 seconds.
theorem train_crossing_time : time_to_cross = 12.1 := by
  sorry

end train_crossing_time_l55_55876


namespace prob_xyz_plus_xy_plus_x_div_by_4_l55_55544

noncomputable def probability_divisible_by_4 (n : ℕ) : ℚ :=
  let m := n / 4 in
  let p_x_div := m / n in
  let p_x_not_div := 1 - p_x_div in
  let p_yz_y_1_div := (m / n) * (m / n) in
  p_x_div + p_x_not_div * p_yz_y_1_div

theorem prob_xyz_plus_xy_plus_x_div_by_4 : 
  probability_divisible_by_4 2009 = 0.377 := 
sorry

end prob_xyz_plus_xy_plus_x_div_by_4_l55_55544


namespace vector_magnitude_difference_l55_55020

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55020


namespace vsevolod_yaroslav_cube_l55_55620

theorem vsevolod_yaroslav_cube (v : Fin 8 → ℤ) (f : Fin 6 → ℤ)
  (hv : ∀ i, v i = 1 ∨ v i = -1)
  (hf : ∀ j, f j = (v (cube_face j 0) * v (cube_face j 1) * v (cube_face j 2) * v (cube_face j 3)))
  : (∑ i in Finset.univ, v i + ∑ j in Finset.univ, f j) ≠ 0 := 
sorry

-- Define a function to get the vertices of a face given its index.
def cube_face : Fin 6 → Fin 4 → Fin 8
| 0, 0 => 0 | 0, 1 => 1 | 0, 2 => 3 | 0, 3 => 2
| 1, 0 => 4 | 1, 1 => 5 | 1, 2 => 7 | 1, 3 => 6
| 2, 0 => 0 | 2, 1 => 1 | 2, 2 => 5 | 2, 3 => 4
| 3, 0 => 2 | 3, 1 => 3 | 3, 2 => 7 | 3, 3 => 6
| 4, 0 => 0 | 4, 1 => 2 | 4, 2 => 6 | 4, 3 => 4
| 5, 0 => 1 | 5, 1 => 3 | 5, 2 => 7 | 5, 3 => 5

end vsevolod_yaroslav_cube_l55_55620


namespace polynomials_equal_l55_55464

theorem polynomials_equal (f g : Polynomial ℝ) (n : ℕ) (x : Fin (n + 1) → ℝ) :
  (∀ i, f.eval (x i) = g.eval (x i)) → f = g :=
by
  sorry

end polynomials_equal_l55_55464


namespace complex_equality_l55_55850

noncomputable def complex_equation (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)

theorem complex_equality {z : ℂ} (h : complex_equation z) : 
  z^12 + z^(-12) = 1 :=
by sorry

end complex_equality_l55_55850


namespace vec_magnitude_is_five_l55_55052

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55052


namespace minimum_pebbles_l55_55303

theorem minimum_pebbles (n : ℕ) (h : n = 42) :
  ∃ a : Fin n → ℕ,
  (∀ i : Fin (n-1), a i > a ⟨i + 1, Nat.lt_of_lt_pred (Nat.lt_of_succ_lt_succ i.property)⟩) ∧
  (∑ i, a i) = 903 := by
  sorry

end minimum_pebbles_l55_55303


namespace range_of_a_l55_55887

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - a) → (∀ x : ℝ, f 0 ≤ 0) → (0 ≤ a) :=
by
  intro h1 h2
  suffices h : -a ≤ 0 by
    simpa using h
  have : f 0 = -a
  simp [h1]
  sorry -- Proof steps are omitted

end range_of_a_l55_55887


namespace number_of_true_propositions_l55_55613

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end number_of_true_propositions_l55_55613


namespace two_digit_number_is_24_l55_55293

-- Defining the two-digit number conditions

variables (x y : ℕ)

noncomputable def condition1 := y = x + 2
noncomputable def condition2 := (10 * x + y) * (x + y) = 144

-- The statement of the proof problem
theorem two_digit_number_is_24 (h1 : condition1 x y) (h2 : condition2 x y) : 10 * x + y = 24 :=
sorry

end two_digit_number_is_24_l55_55293


namespace probability_first_two_cards_spades_l55_55688

theorem probability_first_two_cards_spades : 
  let total_cards := 52
  let spades_cards := 13
  let first_card_prob := (spades_cards : ℝ) / total_cards
  let second_card_given_first_prob := (spades_cards - 1 : ℝ) / (total_cards - 1)
  let joint_prob := first_card_prob * second_card_given_first_prob
  in joint_prob = 1 / 17 :=
by
  sorry

end probability_first_two_cards_spades_l55_55688


namespace simplify_and_evaluate_l55_55984

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) : 
  (1 - (1 / (a + 1))) / ((a^2 - 2*a + 1) / (a^2 - 1)) = (2 / 3) :=
by
  sorry

end simplify_and_evaluate_l55_55984


namespace proportion_of_oj_correct_l55_55893

-- Define the initial conditions and statements
def initial_juice_volume : ℝ := 1 -- 1 liter of orange juice in the 1-liter bottle
def operations : ℕ := 10 -- Number of operations performed
def total_volume_jug : ℝ := 19 -- 19-liter jug

-- Define the function to calculate the volume of juice left after n operations
noncomputable def juice_left_after_operations (n : ℕ) : ℝ :=
  initial_juice_volume * (1 / 2) ^ n

-- Define the function to calculate the proportion of orange juice at the end
noncomputable def proportion_of_oj (n : ℕ) : ℝ :=
  let final_oj_volume := initial_juice_volume * (1 - (1 / 2) ^ n)
  final_oj_volume / (total_volume_jug + (n / 2))

-- State the main theorem
theorem proportion_of_oj_correct : proportion_of_oj operations ≈ 0.05 :=
by
  sorry

end proportion_of_oj_correct_l55_55893


namespace combined_frosting_rate_l55_55338

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end combined_frosting_rate_l55_55338


namespace log_base_4_half_l55_55372

theorem log_base_4_half : log 4 (1 / 2) = -1 / 2 := 
sorry

end log_base_4_half_l55_55372


namespace final_value_correct_l55_55929

-- Define the square and its vertices
def A := (0, 2 : ℝ × ℝ)
def B := (2, 2 : ℝ × ℝ)
def C := (2, 0 : ℝ × ℝ)
def D := (0, 0 : ℝ × ℝ)

-- Define the midpoints M and N
def M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the feet of the perpendiculars X and Y
def X : ℝ × ℝ :=
let lineMD : ℝ → ℝ := λ x, (1/2) * x in
let perpA : ℝ → ℝ := λ x, -2 * x + 2 in
let x_coord := 4/5 in
(x_coord, lineMD x_coord)

def Y : ℝ × ℝ :=
let lineNB : ℝ → ℝ := λ x, 2 * x - 2 in
let perpA : ℝ → ℝ := λ x, -(1/2) * x + 2 in
let x_coord := 8/5 in
(x_coord, lineNB x_coord)

-- Define the distance square as a rational number
def dist_sq_xy :=
let dx := (Y.1 - X.1) in
let dy := (Y.2 - X.2) in
(dx^2 + dy^2 : ℝ)

-- Define the final value of 100p + q where p/q = dist_sq_xy
def final_value := 100 * 32 + 25

theorem final_value_correct : final_value = 3225 := by
  -- Placeholder for the actual proof, which will involve calculating the
  -- Euclidean distance and proving p = 32, q = 25 are relatively prime
  sorry

end final_value_correct_l55_55929


namespace non_student_ticket_price_l55_55262

theorem non_student_ticket_price (x : ℕ) : 
  (∃ (n_student_ticket_price ticket_count total_revenue student_tickets : ℕ),
    n_student_ticket_price = 9 ∧
    ticket_count = 2000 ∧
    total_revenue = 20960 ∧
    student_tickets = 520 ∧
    (student_tickets * n_student_ticket_price + (ticket_count - student_tickets) * x = total_revenue)) -> 
  x = 11 := 
by
  -- placeholder for proof
  sorry

end non_student_ticket_price_l55_55262


namespace solve_linear_combination_l55_55404

-- Define the given vectors and variables
def e1 : ℝ × ℝ := (2, 1)
def e2 : ℝ × ℝ := (1, 3)
def a : ℝ × ℝ := (-1, 2)

-- The variables we need to solve for
variables (λ1 λ2 : ℝ)

-- Define the condition we need to satisfy
def condition : Prop :=
  a = (λ1 * e1.1 + λ2 * e2.1, λ1 * e1.2 + λ2 * e2.2)

-- The theorem we need to prove
theorem solve_linear_combination : (λ1, λ2) = (-1, 1) :=
by
  sorry

end solve_linear_combination_l55_55404


namespace maximize_annual_profit_l55_55263

theorem maximize_annual_profit :
  ∃ t : ℕ, t > 0 ∧ (∀ t' : ℕ, t' > 0 → (let s := -2 * t^2 + 30 * t - 98 in (s / t) ≥ 
                                      let s' := -2 * t'^2 + 30 * t' - 98 in (s' / t'))) ∧ t = 7 := 
by {
    use 7,
    sorry
}

end maximize_annual_profit_l55_55263


namespace pair_count_l55_55125

theorem pair_count :
  {p : ℕ × ℕ // (∃ m n, p = (m, n) ∧ 
    (m > 0 ∧ n > 0) ∧ 
    (m % 2 = 0) ∧ (n % 2 = 0) ∧ 
    (m^2 + n < 50))}.card = 47 := 
sorry

end pair_count_l55_55125


namespace inequality_false_l55_55494

variables {A1 A2 A3 : Type} [fintype A1] [fintype A2] [fintype A3]
variables (x1 x2 x3 r : ℝ)

-- Conditions: 
-- x_i are distances from an internal point to the sides of the triangle.
-- r is the radius of the inscribed circle.
def distances_to_sides (x1 x2 x3 : ℝ) : Prop :=
x1 > 0 ∧ x2 > 0 ∧ x3 > 0

def radius_of_inscribed_circle (r : ℝ) : Prop :=
r > 0

-- Inequality to prove/disprove
theorem inequality_false
  (h1 : distances_to_sides x1 x2 x3)
  (h2 : radius_of_inscribed_circle r) :
  ¬ (1/x1 + 1/x2 + 1/x3 >= 3/r) :=
sorry

end inequality_false_l55_55494


namespace min_rainfall_on_fourth_day_l55_55609

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end min_rainfall_on_fourth_day_l55_55609


namespace range_of_m_l55_55525

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) * (2 * x - 1) - m * x + m

def exists_unique_int_n (m : ℝ) : Prop :=
∃! n : ℤ, f n m < 0

theorem range_of_m {m : ℝ} (h : m < 1) (h2 : exists_unique_int_n m) : 
  (Real.exp 1) * (1 / 2) ≤ m ∧ m < 1 :=
sorry

end range_of_m_l55_55525


namespace greatest_value_sum_eq_24_l55_55723

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55723


namespace find_conjugate_l55_55425

def is_conjugate (z : ℂ) (c : ℂ) : Prop :=
  complex.conj z = c

theorem find_conjugate : 
  ∀ z : ℂ, z - complex.I = (3 + complex.I) / (1 + complex.I) → is_conjugate z 2 := 
by
  intro z h
  sorry

end find_conjugate_l55_55425


namespace find_a_plus_b_l55_55939

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_plus_b (a b : ℝ) (h_cond : ∀ x : ℝ, h (f a b x) = 4 * x + 3) : a + b = 13 / 3 :=
by
  sorry

end find_a_plus_b_l55_55939


namespace percent_of_volume_removed_is_1707_l55_55687

def volume_of_box (length width height : ℝ) : ℝ := length * width * height

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

def percent_removed_of_volume 
  (original_volume removed_volume : ℝ) : ℝ := 
  (removed_volume / original_volume) * 100

theorem percent_of_volume_removed_is_1707 :
  let original_box_volume := volume_of_box 20 15 10 in
  let cube_volume := volume_of_cube 4 in
  let total_removed_volume := cube_volume * 8 in
  let percent_volume_removed := percent_removed_of_volume original_box_volume total_removed_volume in
  percent_volume_removed = 17.07 :=
by 
  sorry

end percent_of_volume_removed_is_1707_l55_55687


namespace number_of_legs_walking_on_the_ground_l55_55663

theorem number_of_legs_walking_on_the_ground :
  ∀ (horses men : ℕ), 
  horses = 8 →  
  men = horses → 
  (∃ riding walking : ℕ, riding = men / 2 ∧ walking = men - riding) → 
  (∃ walking_legs horse_legs : ℕ, walking_legs = walking * 2 ∧ horse_legs = horses * 4) →
  (walking_legs + horse_legs = 40) :=
by
  intros horses men h1 h2 ⟨riding, walking, h3, h4⟩ ⟨walking_legs, horse_legs, h5, h6⟩
  sorry

end number_of_legs_walking_on_the_ground_l55_55663


namespace magnitude_of_a_minus_b_l55_55075

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55075


namespace cos_angle_between_parallelogram_diagonals_l55_55313

theorem cos_angle_between_parallelogram_diagonals :
  let a := (3 : ℝ, 2, 2) 
  let b := (2 : ℝ, 3, -1)
  let diag1 := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let diag2 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let dot_product := diag1.1 * diag2.1 + diag1.2 * diag2.2 + diag1.3 * diag2.3
  let magnitude1 := Real.sqrt (diag1.1^2 + diag1.2^2 + diag1.3^2)
  let magnitude2 := Real.sqrt (diag2.1^2 + diag2.2^2 + diag2.3^2)
  let cos_theta := dot_product / (magnitude1 * magnitude2)
  cos_theta = -3 / Real.sqrt 561 := sorry

end cos_angle_between_parallelogram_diagonals_l55_55313


namespace vector_magnitude_subtraction_l55_55064

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55064


namespace balloon_height_correct_l55_55450

variable (initial_money : ℕ := 200)
variable (cost_sheet : ℕ := 42)
variable (cost_rope : ℕ := 18)
variable (cost_tank : ℕ := 14)
variable (helium_cost_per_oz : ℚ := 1.50)
variable (height_per_oz : ℕ := 113)

def remaining_money (initial_money cost_sheet cost_rope cost_tank : ℕ) : ℕ :=
  initial_money - (cost_sheet + cost_rope + cost_tank)

def helium_oz (remaining_money : ℕ) (helium_cost_per_oz : ℚ) : ℕ :=
  (remaining_money : ℚ) / helium_cost_per_oz

def balloon_height (helium_oz height_per_oz : ℕ) : ℕ :=
  helium_oz * height_per_oz

theorem balloon_height_correct :
  balloon_height (helium_oz (remaining_money initial_money cost_sheet cost_rope cost_tank) helium_cost_per_oz.to_nat) height_per_oz = 9482 :=
by
  sorry

end balloon_height_correct_l55_55450


namespace sqrt_exp_cube_l55_55741

theorem sqrt_exp_cube :
  ((Real.sqrt ((Real.sqrt 5)^4))^3 = 125) :=
by
  sorry

end sqrt_exp_cube_l55_55741


namespace find_number_l55_55306

theorem find_number (x : ℝ) (h : x * 2 + (12 + 4) * (1/8) = 602) : x = 300 :=
by
  sorry

end find_number_l55_55306


namespace vec_magnitude_is_five_l55_55054

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55054


namespace find_n_l55_55127

theorem find_n (n : ℝ) : 7^(2 * n) = (1 / 7)^(n - 20) → n = 20 / 3 :=
by
  intro h
  sorry

end find_n_l55_55127


namespace find_difference_l55_55181

noncomputable def expr (a b : ℝ) : ℝ :=
  |a - b| / (|a| + |b|)

def min_val (a b : ℝ) : ℝ := 0

def max_val (a b : ℝ) : ℝ := 1

theorem find_difference (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  max_val a b - min_val a b = 1 :=
by
  sorry

end find_difference_l55_55181


namespace lindy_total_distance_l55_55163

-- Definitions derived from the conditions
def jack_speed : ℕ := 5
def christina_speed : ℕ := 7
def lindy_speed : ℕ := 12
def initial_distance : ℕ := 360

theorem lindy_total_distance :
  lindy_speed * (initial_distance / (jack_speed + christina_speed)) = 360 := by
  sorry

end lindy_total_distance_l55_55163


namespace angle_NMK_l55_55968

-- Define the problem
theorem angle_NMK (A B C P Q M N K : Type)
  (h_ABC: angle A C B = 100) 
  (h_AP_BC: segment_length A P = segment_length B C)
  (h_BQ_AC: segment_length B Q = segment_length A C)
  (h_M_mid: is_midpoint M A B)
  (h_N_mid: is_midpoint N C P)
  (h_K_mid: is_midpoint K C Q) : 
  angle N M K = 40 :=
sorry

end angle_NMK_l55_55968


namespace min_photos_needed_to_ensure_conditions_l55_55803

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55803


namespace max_power_sum_l55_55710

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55710


namespace trains_cross_in_opposite_direction_in_12_seconds_l55_55269

noncomputable def convert_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (5 / 18)

noncomputable def time_to_cross_same_direction (speed1_kmph speed2_kmph time_sec : ℝ) : ℝ :=
  let relative_speed_mps := convert_kmph_to_mps (speed1_kmph - speed2_kmph)
  in (time_sec * relative_speed_mps) / 2

noncomputable def time_to_cross_opposite_direction (speed1_kmph speed2_kmph length_train : ℝ) : ℝ :=
  let relative_speed_mps := convert_kmph_to_mps (speed1_kmph + speed2_kmph)
      distance := 2 * length_train
  in distance / relative_speed_mps

theorem trains_cross_in_opposite_direction_in_12_seconds :
  ∀ (speed1_kmph speed2_kmph time_sec : ℝ),
  let length_train := time_to_cross_same_direction speed1_kmph speed2_kmph time_sec
  in
  speed1_kmph = 60 → speed2_kmph = 40 → time_sec = 60 →
  time_to_cross_opposite_direction speed1_kmph speed2_kmph length_train = 12 :=
by { intros, sorry }

end trains_cross_in_opposite_direction_in_12_seconds_l55_55269


namespace common_difference_l55_55842

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : a 2 = 1 + d) (h4 : a 4 = 1 + 3 * d) (h5 : a 5 = 1 + 4 * d) 
  (h_geometric : (a 4)^2 = a 2 * a 5) 
  (h_nonzero : d ≠ 0) : 
  d = 1 / 5 :=
by sorry

end common_difference_l55_55842


namespace min_photos_l55_55789

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55789


namespace paintable_walls_area_l55_55925

def dimensions : Type :=
  { length : ℕ,
    width : ℕ,
    height : ℕ }

def bedroom_dimensions : dimensions := 
  { length := 15,
    width := 11,
    height := 9 }

def num_bedrooms : ℕ := 4

def doorways_windows_area_per_bedroom : ℕ := 80

def wall_area (d : dimensions) : ℕ :=
  2 * (d.length * d.height) + 2 * (d.width * d.height)

noncomputable def paintable_area_per_bedroom (d : dimensions) (a : ℕ) : ℕ :=
  wall_area(d) - a

noncomputable def total_paintable_area (n : ℕ) (d : dimensions) (a : ℕ) : ℕ :=
  n * paintable_area_per_bedroom d a

theorem paintable_walls_area :
  total_paintable_area num_bedrooms bedroom_dimensions doorways_windows_area_per_bedroom = 1552 :=
by
  sorry

end paintable_walls_area_l55_55925


namespace sum_of_a_and_b_l55_55739

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55739


namespace sum_of_powers_eq_21_l55_55606

theorem sum_of_powers_eq_21 :
  ∃ (n : ℕ → ℕ) (a : ℕ → ℤ), 
      (∀ i j : ℕ, i < j → n i > n j) ∧ 
      (∀ k : ℕ, a k = 1 ∨ a k = -1) ∧ 
      (∃ r : ℕ, (∑ k in finRange r, a k * 3 ^ n k) = 2008) →
      (∑ k in finRange r, n k) = 21 := 
sorry

end sum_of_powers_eq_21_l55_55606


namespace two_thousand_and_twelfth_digit_is_zero_l55_55284

def digit_in_sequence (n : ℕ) : ℕ := 
    let s := String.mk (List.range (n + 1) |>.map (λ i => i.repr) |>.bind (λ s => s.data))
    s.toList.nth (2011)

theorem two_thousand_and_twelfth_digit_is_zero : 
    digit_in_sequence 2012 = 0 := 
sorry

end two_thousand_and_twelfth_digit_is_zero_l55_55284


namespace ellipse_equation_slope_angle_of_line_perpendicular_bisector_y0_l55_55843

variable (a b : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (e : ℝ := (Real.sqrt 3) / 2)
variable (h3 : e = (Real.sqrt 3) / 2)
variable (area : ℝ := 4)

theorem ellipse_equation (h4 : a = 2 * b) (h5 : a * b = 2) :
  (∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1 ↔ x^2 / 4 + y^2 = 1) := sorry

theorem slope_angle_of_line (A : ℝ × ℝ := (-2, 0)) (dist_AB : ℝ := 4 * Real.sqrt 2 / 5) (k : ℝ):
  (dist_AB = 4 * Real.sqrt(1 + k^2) / (1 + 4 * k^2) ↔ k = 1 ∨ k = -1) := sorry

theorem perpendicular_bisector_y0 (k : ℝ) (h6 : k ≠ 0) (QA : ℝ × ℝ) (QB : ℝ × ℝ) 
  (h7 : QA • QB = 4) :
  (y0 = 2 * Real.sqrt 2 ∨ y0 = -2 * Real.sqrt 2 ∨ y0 = 2 * Real.sqrt 14 / 5 ∨ y0 = -2 * Real.sqrt 14 / 5) := sorry

end ellipse_equation_slope_angle_of_line_perpendicular_bisector_y0_l55_55843


namespace min_distance_to_line_l55_55390

theorem min_distance_to_line :
  ∃ (x y : ℝ), 8 * x + 15 * y = 120 ∧ sqrt (x^2 + y^2) = 120 / 17 :=
begin
  sorry
end

end min_distance_to_line_l55_55390


namespace vector_magnitude_subtraction_l55_55029

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55029


namespace normal_price_of_biography_l55_55661

variable (B : ℝ) -- Normal price of a biography
variable (D_biographies : ℝ) -- Discount rate of biographies
variable (D_mysteries : ℝ ≔ 0.375) -- Discount rate of mysteries

-- Conditions
axiom C1 : mysteries_price = 12
axiom C2 : 5 * (B * D_biographies) + 3 * (mysteries_price * D_mysteries) = 19
axiom C3 : D_biographies + D_mysteries = 0.43
axiom C4 : D_mysteries = 0.375

-- Proposition to be proved
theorem normal_price_of_biography (C1 : 12) (C2 : 5 * (B * D_biographies) + 3 * (12 * 0.375) = 19) (C3 : D_biographies + 0.375 = 0.43) : B = 20 :=
by
  sorry

end normal_price_of_biography_l55_55661


namespace row_trip_time_example_l55_55314

noncomputable def round_trip_time
    (rowing_speed : ℝ)
    (current_speed : ℝ)
    (total_distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let one_way_distance := total_distance / 2
  let time_to_place := one_way_distance / downstream_speed
  let time_back := one_way_distance / upstream_speed
  time_to_place + time_back

theorem row_trip_time_example :
  round_trip_time 10 2 96 = 10 := by
  sorry

end row_trip_time_example_l55_55314


namespace number_of_12_digit_integers_with_two_consecutive_ones_l55_55111

def has_exactly_two_consecutive_ones (n : ℕ) : Prop :=
  let digits := to_digits 2 n in
  list.count 1 digits = 2 ∧ (∃ i, digits.nth i = some 1 ∧ digits.nth (i + 1) = some 1)

def valid_12_digit_integer (n : ℕ) : Prop :=
  let digits := to_digits 2 n in
  list.length digits = 12 ∧ ∀ d ∈ digits, d = 1 ∨ d = 2

theorem number_of_12_digit_integers_with_two_consecutive_ones :
  {n : ℕ | valid_12_digit_integer n ∧ has_exactly_two_consecutive_ones n}.card = 1278 :=
sorry

end number_of_12_digit_integers_with_two_consecutive_ones_l55_55111


namespace magnitude_of_a_minus_b_l55_55073

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55073


namespace burglar_total_sentence_l55_55957

-- Given conditions
def value_of_goods_stolen : ℝ := 40000
def base_sentence_per_thousand_stolen : ℝ := 1 / 5000
def third_offense_increase : ℝ := 0.25
def resisting_arrest_addition : ℕ := 2

-- Theorem to prove the total sentence
theorem burglar_total_sentence :
  let base_sentence := base_sentence_per_thousand_stolen * value_of_goods_stolen
  let increased_sentence := base_sentence * (1 + third_offense_increase)
  let total_sentence := increased_sentence + resisting_arrest_addition
  total_sentence = 12 :=
by 
  sorry -- Proof steps are skipped

end burglar_total_sentence_l55_55957


namespace algebraic_expression_perfect_square_l55_55133

theorem algebraic_expression_perfect_square (a : ℤ) :
  (∃ b : ℤ, ∀ x : ℤ, x^2 + (a - 1) * x + 16 = (x + b)^2) →
  (a = 9 ∨ a = -7) :=
sorry

end algebraic_expression_perfect_square_l55_55133


namespace solve_equation_l55_55559

theorem solve_equation
  (x : ℝ)
  (h : 5 ^ (real.sqrt (x^3 + 3*x^2 + 3*x + 1)) = real.sqrt ((5 * real.root 4 (x + 1)^5)^3))
  : x = 65 / 16 :=
sorry

end solve_equation_l55_55559


namespace filtration_concentration_l55_55894

-- Variables and conditions used in the problem
variable (P P0 : ℝ) (k t : ℝ)
variable (h1 : P = P0 * Real.exp (-k * t))
variable (h2 : Real.exp (-2 * k) = 0.8)

-- Main statement: Prove the concentration after 5 hours is approximately 57% of the original
theorem filtration_concentration :
  (P0 * Real.exp (-5 * k)) / P0 = 0.57 :=
by sorry

end filtration_concentration_l55_55894


namespace minimum_photos_l55_55821

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55821


namespace sum_of_distinct_products_l55_55580

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10)
  (h_divis_72 : ∃ G H : ℕ, G < 10 ∧ H < 10 ∧ (831000000 + 100000*G + 100*H + 70000 + 5000 + 300 + H * 10 + 8) % 72 = 0) :
  (∑ h : (H = 0 ∧ G = 1) ∨ (H = 4 ∧ G = 6), (G * H)) = 24 :=
by
  sorry

end sum_of_distinct_products_l55_55580


namespace floor_sum_eq_138_l55_55515

noncomputable def p := sorry
noncomputable def q := sorry
noncomputable def r := sorry
noncomputable def s := sorry

theorem floor_sum_eq_138 (hpq : (p : ℝ) * q = 1152)
                         (hrs : (r : ℝ) * s = 1152)
                         (h1 : (p : ℝ)^2 + q^2 = 2500)
                         (h2 : (r : ℝ)^2 + s^2 = 2500) :
  Float.floor (p + q + r + s) = 138 :=
sorry

end floor_sum_eq_138_l55_55515


namespace largest_possible_difference_l55_55329

theorem largest_possible_difference (A_est : ℕ) (B_est : ℕ) (A : ℝ) (B : ℝ)
(hA_est : A_est = 40000) (hB_est : B_est = 70000)
(hA_range : 36000 ≤ A ∧ A ≤ 44000)
(hB_range : 60870 ≤ B ∧ B ≤ 82353) :
  abs (B - A) = 46000 :=
by sorry

end largest_possible_difference_l55_55329


namespace average_first_12_even_numbers_l55_55623

theorem average_first_12_even_numbers : 
  let evens := [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10] in
  (List.sum evens) / (List.length evens) = -1 := by
  sorry

end average_first_12_even_numbers_l55_55623


namespace number_of_perfect_squares_between_210_and_560_l55_55879

theorem number_of_perfect_squares_between_210_and_560 : 
  ∃ n, n = 9 ∧ ∀ x, 210 < x ∧ x < 560 → (∃ k, x = k * k → k ∈ set.range (λ k, 15 + k) ∧ k ≤ 23) :=
by
  sorry

end number_of_perfect_squares_between_210_and_560_l55_55879


namespace min_photos_for_condition_l55_55812

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55812


namespace stamps_ratio_l55_55169

theorem stamps_ratio (total_stamps european_stamps asian_stamps : ℕ)
  (h_total : total_stamps = 444)
  (h_european : european_stamps = 333)
  (h_asian : asian_stamps = total_stamps - european_stamps) :
  (european_stamps / asian_stamps) = 3 :=
by
  have h1 : asian_stamps = 111 :=
    by rw [h_total, h_european, Nat.sub_eq_of_eq_add] -- This automatically follows from h_total and h_european
  have h2 : 333 / 111 = 3 :=
    by norm_num -- This calculates the ratio directly
  rw [h1] -- Replace asian_stamps with 111
  exact h2 -- Conclude that the ratio is 3

end stamps_ratio_l55_55169


namespace min_photos_exists_l55_55795

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55795


namespace intersection_A_B_eq_B_l55_55419

-- Define set A
def setA : Set ℝ := { x : ℝ | x > -3 }

-- Define set B
def setB : Set ℝ := { x : ℝ | x ≥ 2 }

-- Theorem statement of proving the intersection of setA and setB is setB itself
theorem intersection_A_B_eq_B : setA ∩ setB = setB :=
by
  -- proof skipped
  sorry

end intersection_A_B_eq_B_l55_55419


namespace solve_equation_l55_55558

theorem solve_equation
  (x : ℝ)
  (h : 5 ^ (real.sqrt (x^3 + 3*x^2 + 3*x + 1)) = real.sqrt ((5 * real.root 4 (x + 1)^5)^3))
  : x = 65 / 16 :=
sorry

end solve_equation_l55_55558


namespace N_mod_45_l55_55506

noncomputable def N : ℕ := 
\[
  123456789101112 \ldots 4344
\]

theorem N_mod_45 :
  ( \N \equiv 9 \pmod{45} \)
  := sorry

end N_mod_45_l55_55506


namespace sum_of_solutions_l55_55519

variable (x : ℝ)

def f (x : ℝ) : ℝ := 3 * x + 2

noncomputable def f_inv (x : ℝ) : ℝ := (x - 2) / 3

theorem sum_of_solutions : 
  (∑ x in ({x : ℝ | f_inv x = f x⁻¹ } : Set ℝ), x) = 8 :=
by
  sorry

end sum_of_solutions_l55_55519


namespace minimum_photos_l55_55819

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55819


namespace eraser_ratio_l55_55751

-- Define the variables and conditions
variables (c j g : ℕ)
variables (total : ℕ := 35)
variables (c_erasers : ℕ := 10)
variables (gabriel_erasers : ℕ := c_erasers / 2)
variables (julian_erasers : ℕ := c_erasers)

-- The proof statement
theorem eraser_ratio (hc : c_erasers = 10)
                      (h1 : c_erasers = 2 * gabriel_erasers)
                      (h2 : julian_erasers = c_erasers)
                      (h3 : c_erasers + gabriel_erasers + julian_erasers = total) :
                      julian_erasers / c_erasers = 1 :=
by
  sorry

end eraser_ratio_l55_55751


namespace factorize_expression_l55_55379

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l55_55379


namespace vector_magnitude_correct_l55_55044

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55044


namespace prime_quadratic_residue_sum_l55_55358

theorem prime_quadratic_residue_sum {p : ℕ} (hp : Nat.Prime p) :
  (∑ x in Finset.range p, if QuadraticResidue x p then x else 0) % p = 0 → p ≥ 5 :=
by sorry

end prime_quadratic_residue_sum_l55_55358


namespace candle_burning_rate_l55_55266

theorem candle_burning_rate (t : ℝ) : 
  ∀ (h : ℝ), height1 : ℝ, height2 : ℝ, 
    (height1 = 1 - t / 5) → 
    (height2 = 1 - t / 4) → 
    (3 * height2 = height1) → 
    t = 40 / 11 :=
by
  intros
  have h1 : height1 = 1 - t / 5, from ‹height1 = 1 - t / 5›
  have h2 : height2 = 1 - t / 4, from ‹height2 = 1 - t / 4›
  rw [h1, h2] at ‹3 * height2 = height1›
  sorry

end candle_burning_rate_l55_55266


namespace minDistanceFromLatticePointToLine_l55_55468

-- Define what a lattice point is
def isLatticePoint (P : ℝ × ℝ) : Prop :=
  ∃ x y : ℤ, P = (x, y)

-- Define the line equation
def lineEquation (x y : ℝ) : Prop :=
  y = (3 / 4) * x + (2 / 3)

-- Define the point-to-line distance formula
def pointToLineDistance (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

-- State the main theorem
theorem minDistanceFromLatticePointToLine : 
  ∀ P : ℝ × ℝ, isLatticePoint P → 
  (∃ (d : ℝ), d = pointToLineDistance P.1 P.2 9 (-12) 8 ∧ d = 2 / 15) :=
by
  -- Here we would provide the detailed proof
  sorry

end minDistanceFromLatticePointToLine_l55_55468


namespace negation_equivalent_statement_l55_55653

theorem negation_equivalent_statement (x y : ℝ) :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
sorry

end negation_equivalent_statement_l55_55653


namespace opposite_of_number_reciprocal_of_number_absolute_value_of_number_l55_55582

def number := 2023

theorem opposite_of_number : -number = -2023 := 
by 
  sorry

theorem reciprocal_of_number : (1 : ℚ) / number = 1 / 2023 := 
by 
  sorry

theorem absolute_value_of_number : |number| = 2023 := 
by 
  sorry

end opposite_of_number_reciprocal_of_number_absolute_value_of_number_l55_55582


namespace correct_a_c_d_l55_55511

-- Proving that f(x) = ((e^x) - a)/x - a * ln(x) has specific properties given conditions

noncomputable def e := Real.exp 1
noncomputable def f (x a : ℝ) := (Real.exp x - a)/x - a * Real.log x

theorem correct_a_c_d (a : ℝ) :
  (∀ x > 0, a = e → ¬∃ c, ∃ y > 0, f'(c) < 0 ∧ y ≠ c) ∧
  ((1 < a ∧ a < e) → ∃ y > 0, f(y, a) = 0 ∧ ∀ z < y, f(z, a) > 0) ∧
  (a ≤ 1 → ¬∃ y > 0, f(y, a) = 0) :=
by
  sorry

end correct_a_c_d_l55_55511


namespace magnitude_of_a_minus_b_l55_55081

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55081


namespace percentage_difference_l55_55267

theorem percentage_difference (X : ℝ) (h1 : first_num = 0.70 * X) (h2 : second_num = 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 10 := by
  sorry

end percentage_difference_l55_55267


namespace vector_magnitude_l55_55105

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55105


namespace number_of_outfits_l55_55563

theorem number_of_outfits (shirts ties belts : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_belts : belts = 4) : 
  shirts * ties * belts = 224 := 
by
  rw [h_shirts, h_ties, h_belts]
  norm_num

end number_of_outfits_l55_55563


namespace distinct_points_on_curve_abs_diff_l55_55208

theorem distinct_points_on_curve_abs_diff {e c d : ℝ} (h1 : (sqrt e, c) ∈ {p : ℝ × ℝ | p.snd^2 + p.fst^6 = 3 * p.fst^3 * p.snd + 1})
    (h2 : (sqrt e, d) ∈ {p : ℝ × ℝ | p.snd^2 + p.fst^6 = 3 * p.fst^3 * p.snd + 1})
    (h3 : c ≠ d) : |c - d| = |sqrt (5 * e^3 + 4)| :=
begin
  sorry
end

end distinct_points_on_curve_abs_diff_l55_55208


namespace point_distances_l55_55202

theorem point_distances (n : ℕ) (blue_points red_points : Fin n → ℝ) :
  (∑ i j : Fin n, abs (blue_points i - blue_points j) + abs (red_points i - red_points j)) ≤ 
  (∑ i j : Fin n, abs (blue_points i - red_points j)) := 
by
  sorry

end point_distances_l55_55202


namespace johns_speed_l55_55926

theorem johns_speed (J : ℝ)
  (lewis_speed : ℝ := 60)
  (distance_AB : ℝ := 240)
  (meet_distance_A : ℝ := 160)
  (time_lewis_to_B : ℝ := distance_AB / lewis_speed)
  (time_lewis_back_80 : ℝ := 80 / lewis_speed)
  (total_time_meet : ℝ := time_lewis_to_B + time_lewis_back_80)
  (total_distance_john_meet : ℝ := J * total_time_meet) :
  total_distance_john_meet = meet_distance_A → J = 30 := 
by
  sorry

end johns_speed_l55_55926


namespace tetrahedron_face_inequality_l55_55903

-- Definitions used in the conditions
def triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

-- Definitions of areas of faces in a tetrahedron
variables {A B C D : Type} [linear_order A] [linear_order B] [linear_order C] [linear_order D]
variables (area_A area_B area_C area_D : ℝ)

-- The main theorem statement
theorem tetrahedron_face_inequality
  (h_triangle : ∀ (a b c : ℝ), triangle_inequality a b c)
  (faces_areas : A → B → C → D → ℝ) : 
  (faces_areas A B C) > (area_D) :=
sorry

end tetrahedron_face_inequality_l55_55903


namespace log_base_4_half_l55_55371

theorem log_base_4_half : ∀ (a : ℝ), a = 4 → ∀ (b : ℝ), b = 1 / 2 → log a b = -1 / 2 := 
by 
  intros a ha b hb 
  rw [ha, hb] 
  rw [real.log_div (pow_pos (by norm_num) 2) (by norm_num : (0 : ℝ) < 1)]
  sorry

end log_base_4_half_l55_55371


namespace manolo_makes_45_masks_in_four_hours_l55_55786

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end manolo_makes_45_masks_in_four_hours_l55_55786


namespace vector_magnitude_difference_l55_55017

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55017


namespace eccentricity_of_ellipse_l55_55505

theorem eccentricity_of_ellipse
    {a b c : ℝ}
    (h1 : a > b)
    (h2 : b > 0)
    (ellipse_def : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)
    (f1 : (-c, 0))
    (f2 : (c, 0))
    (P : ℝ × ℝ)
    (angle_condition : ∀ (angle_PF1F2 angle_PF2F1: ℝ), angle_PF1F2 = 5 * angle_PF2F1) :
    eccentricity_of_ellipse a b c = sqrt(6) / 3 :=
sorry

end eccentricity_of_ellipse_l55_55505


namespace derivative_of_f_l55_55885

noncomputable def f (x : ℝ) : ℝ := sqrt (4 * x - 3)

theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 / sqrt (4 * x - 3) :=
sorry

end derivative_of_f_l55_55885


namespace parallelepiped_identity_l55_55970

variable {V : Type*} [InnerProductSpace ℝ V] 
variables (a b c : V)

theorem parallelepiped_identity :
  let PU := ∥a + b + c∥^2 
  let QV := ∥a - b + c∥^2 
  let RT := ∥-a + b + c∥^2 
  let SW := ∥a + b - c∥^2 
  let PQ := ∥b∥^2 
  let PR := ∥c∥^2 
  let PT := ∥a∥^2 
  (PU + QV + RT + SW) / (PQ + PR + PT) = 4 :=
by
  sorry

end parallelepiped_identity_l55_55970


namespace shortest_distance_after_movements_l55_55634

theorem shortest_distance_after_movements :
  let north := 15
  let west := 8
  let south := 10
  let east := 1
  let net_north := north - south
  let net_west := west - east
  sqrt (net_north^2 + net_west^2) = sqrt 74 := by
    sorry

end shortest_distance_after_movements_l55_55634


namespace crayons_ratio_l55_55528

theorem crayons_ratio (initial_crayons : ℕ) (bought_crayons : ℕ) (total_crayons_now : ℕ)
  (h_initial : initial_crayons = 18)
  (h_bought : bought_crayons = 20)
  (h_total : total_crayons_now = 29) :
  (initial_crayons - (total_crayons_now - bought_crayons)) / initial_crayons = 1 / 2 :=
by
  -- initial_crayons = 18, bought_crayons = 20, total_crayons_now = 29
  -- lost_crayons = initial_crayons - (total_crayons_now - bought_crayons)
  -- ratio = (lost_crayons / initial_crayons) = (initial_crayons - (total_crayons_now - bought_crayons)) / initial_crayons
  have h_lost_crayons : initial_crayons - (total_crayons_now - bought_crayons) = 9 := by
    rw [h_initial, h_total, h_bought]
    norm_num
  -- shown (9 / 18) = 1 / 2
  rw [h_initial, h_lost_crayons]
  norm_num
  sorry

end crayons_ratio_l55_55528


namespace value_of_expression_l55_55941

def z : ℂ := 1 + complex.i
def conj_z : ℂ := 1 - complex.i

theorem value_of_expression : -complex.i * z + complex.i * conj_z = 2 := by
  sorry

end value_of_expression_l55_55941


namespace find_n_equal_roots_l55_55352

theorem find_n_equal_roots (x n : ℝ) (hx : x ≠ 2) : n = -1 ↔
  let a := 1
  let b := -2
  let c := -(n^2 + 2 * n)
  b^2 - 4 * a * c = 0 :=
by
  sorry

end find_n_equal_roots_l55_55352


namespace minimum_photos_l55_55817

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55817


namespace plane_equation_l55_55910

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

theorem plane_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x y z : ℝ, (x / a + y / b + z / c = 1) :=
sorry

end plane_equation_l55_55910


namespace leap_years_count_l55_55270

def is_leap_year (y : ℕ) : Bool :=
  if y % 800 = 300 ∨ y % 800 = 600 then true else false

theorem leap_years_count : 
  { y : ℕ // 1500 ≤ y ∧ y ≤ 3500 ∧ y % 100 = 0 ∧ is_leap_year y } = {y | y = 1900 ∨ y = 2200 ∨ y = 2700 ∨ y = 3000 ∨ y = 3500} :=
by
  sorry

end leap_years_count_l55_55270


namespace subset_sum_exists_l55_55409

theorem subset_sum_exists {n : ℕ} (a : Fin n → ℕ) (s : ℕ) (h_sum : (∑ i, a i) = s) (h_s : s ≤ 2 * n - 1) (m : ℕ) (h_m : 1 ≤ m ∧ m ≤ s) :
  ∃ t : Finset (Fin n), (∑ i in t, a i) = m :=
by
  sorry

end subset_sum_exists_l55_55409


namespace magnitude_of_a_minus_b_l55_55080

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55080


namespace vector_magnitude_subtraction_l55_55033

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55033


namespace vector_magnitude_l55_55096

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55096


namespace min_photos_exists_l55_55798

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55798


namespace vector_magnitude_subtraction_l55_55000

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55000


namespace vector_magnitude_subtraction_l55_55030

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55030


namespace RupertCandles_l55_55204

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end RupertCandles_l55_55204


namespace boat_distance_downstream_l55_55592

theorem boat_distance_downstream (speed_boat speed_stream distance_upstream : ℕ) (h1 : speed_boat = 18) (h2 : speed_stream = 6) (h3 : distance_upstream = 32) 
  (h4 : (distance_upstream / (speed_boat - speed_stream))) = (D / (speed_boat + speed_stream)) : D = 64 :=
by
  -- provided that h4 and the proof steps can be derived correctly
  sorry

end boat_distance_downstream_l55_55592


namespace rectangle_perimeter_l55_55907

-- Define the rectangle and circle properties
variables {r : ℝ} (ABCD : Type) [rect : is_rectangle ABCD]

-- Assuming conditions
variable (circle : Type)
variable [is_circle circle r]
variable (tangent_AB : ∀ x : ABCD, midpoint (side_AB x) (side_CD x) -> tangent circle x)
variable (tangent_CD : ∀ x : ABCD, midpoint (side_CD x) (side_AB x) -> tangent circle x)
variable (tangent_BC : ∀ x : ABCD, distance (side_BC x) circle.center = r)

-- The problem statement
theorem rectangle_perimeter (h : height ABCD = 2 * r) (w : width ABCD = 2 * r) :
    perimeter ABCD = 8 * r :=
by
  sorry

end rectangle_perimeter_l55_55907


namespace distinct_triangles_in_octahedron_l55_55115

theorem distinct_triangles_in_octahedron : 
  ∀ (vertices : Finset ℕ), vertices.card = 8 → (Finset.card (Finset.powersetLen 3 vertices) = 56) :=
by
  intros vertices h_vertices
  sorry

end distinct_triangles_in_octahedron_l55_55115


namespace sarah_monthly_payment_l55_55981

noncomputable def monthly_payment (loan_amount : ℝ) (down_payment : ℝ) (years : ℝ) : ℝ :=
  let financed_amount := loan_amount - down_payment
  let months := years * 12
  financed_amount / months

theorem sarah_monthly_payment : monthly_payment 46000 10000 5 = 600 := by
  sorry

end sarah_monthly_payment_l55_55981


namespace reciprocal_of_0_8_reciprocal_of_smallest_composite_l55_55249

-- Define the concept of reciprocal in Lean
def reciprocal (a : ℝ) : ℝ := 1 / a

-- Define the smallest composite number
def smallest_composite : ℕ := 4

-- Statement 1: Prove that the reciprocal of 0.8 is 5/4
theorem reciprocal_of_0_8 : reciprocal 0.8 = 5 / 4 := sorry

-- Statement 2: Prove that 4 is the smallest composite number, and its reciprocal is 1/4
theorem reciprocal_of_smallest_composite : reciprocal smallest_composite = 1 / 4 := sorry

end reciprocal_of_0_8_reciprocal_of_smallest_composite_l55_55249


namespace analyze_f_l55_55513

noncomputable def e : ℝ := Real.exp 1

def f (a x : ℝ) : ℝ := (e ^ x - a) / x - a * Real.log x

theorem analyze_f (a : ℝ) (x : ℝ) (h_pos : x > 0) :
  (a = e → ¬∃ c, c > 0 ∧ c < x ∧ deriv (λ x, f a x) c = 0) ∧
  (1 < a ∧ a < e → ∃ c, c > 0 ∧ c < x ∧ f a c = 0) ∧
  (a ≤ 1 → ¬∃ c, c > 0 ∧ c < x ∧ f a c = 0) :=
sorry

end analyze_f_l55_55513


namespace product_inequality_l55_55398

def distance_to_nearest_integer (a : ℝ) : ℝ :=
  abs (a - round a)

theorem product_inequality 
  (n : ℕ) 
  (a : ℝ) : 
  (∏ k in finset.range (n + 1), abs (a - k)) ≥ (distance_to_nearest_integer a) * (nat.factorial n / 2^n) :=
by
  sorry

end product_inequality_l55_55398


namespace isosceles_triangle_base_length_l55_55844

theorem isosceles_triangle_base_length
  (a b c: ℕ) 
  (h_iso: a = b ∨ a = c ∨ b = c)
  (h_perimeter: a + b + c = 21)
  (h_side: a = 5 ∨ b = 5 ∨ c = 5) :
  c = 5 :=
by
  sorry

end isosceles_triangle_base_length_l55_55844


namespace one_fourths_in_one_eighth_l55_55118

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l55_55118


namespace leak_drain_time_l55_55679

theorem leak_drain_time (P L : ℕ → ℕ) (H1 : ∀ t, P t = 1 / 2) (H2 : ∀ t, P t - L t = 1 / 3) : 
  (1 / L 1) = 6 :=
by
  sorry

end leak_drain_time_l55_55679


namespace heaviest_lightest_difference_l55_55603

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end heaviest_lightest_difference_l55_55603


namespace least_perimeter_of_triangle_l55_55466

noncomputable def cos_A := 3 / 5
noncomputable def cos_B := 5 / 13
noncomputable def cos_C := -1 / 3

theorem least_perimeter_of_triangle (a b c : ℕ)
  (h1 : cos_A = 3 / 5)
  (h2 : cos_B = 5 / 13)
  (h3 : cos_C = -1 / 3)
  (h4 : ∠CAB + ∠ABC + ∠BCA = π)
  (h5 : a*sin_B = b*sin_A)
  (h6 : a*sin_C = c*sin_A)
  (h7 : b*sin_C = c*sin_B) : 
  a + b + c = 192 := 
by 
  sorry

end least_perimeter_of_triangle_l55_55466


namespace math_problem_l55_55190

variable {ℝ : Type*} [LinearOrder ℝ] [AddGroup ℝ]

noncomputable theory

variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def condition1 : Prop := is_odd f ∧ is_strictly_decreasing f
def condition2 : Prop := x₁ + x₂ > 0
def condition3 : Prop := x₂ + x₃ > 0
def condition4 : Prop := x₃ + x₁ > 0

-- The theorem we need to prove
theorem math_problem (h1 : condition1 f) (h2 : condition2 x₁ x₂) (h3 : condition3 x₂ x₃) (h4 : condition4 x₃ x₁) : 
  f x₁ + f x₂ + f x₃ < 0 := sorry

end math_problem_l55_55190


namespace chessboard_piece_arrangement_l55_55391

theorem chessboard_piece_arrangement : 
  ∃ (f : Fin 8 → Fin 9), (∀ i j : Fin 8, i ≠ j → f i ≠ f j) ∧ (∃ (g : Fin 8 → Fin 9), (∀ i j : Fin 8, i ≠ j → g i ≠ g j) ∧ (∑ x in Finset.univ, (f x) + ∑ y in Finset.univ, (g y) = 64))
  → (2 * (Nat.factorial 8) ^ 2 = 2 * (8!) ^ 2) := 
by
  sorry

end chessboard_piece_arrangement_l55_55391


namespace max_diagonals_of_regular_1000gon_l55_55475

theorem max_diagonals_of_regular_1000gon :
  ∃ (d : ℕ), d = 2000 ∧ 
    ∀ (chosen_diagonals : finset (fin 1000 × fin 1000)),
      chosen_diagonals.card = d →
      ∀ (diagonal_triplets : finset (finset (fin 1000 × fin 1000))),
        diagonal_triplets.card = 3 →
        ∃ (diagonals_with_same_length : finset (fin 1000 × fin 1000)),
          diagonals_with_same_length ⊆ diagonal_triplets ∧
          diagonals_with_same_length.card ≥ 2 :=
begin
  sorry
end

end max_diagonals_of_regular_1000gon_l55_55475


namespace find_other_eigenvalue_l55_55445

theorem find_other_eigenvalue (x : ℝ) 
  (h : ∃ v, (∃ (λ : ℝ), λ = 3 ∧ ( ∃ (v : ℝ → ℝ ), v ≠ 0 ∧ (M * v = 3 * v) )) ) 
  (M := ![[1, 2], [2, x]]) : ∃ (λ₂ : ℝ), λ₂ = -1 :=
sorry

end find_other_eigenvalue_l55_55445


namespace point_of_symmetry_l55_55353

def g (x : ℝ) : ℝ := abs (floor (x + 2)) - abs (floor (3 - x))

theorem point_of_symmetry : ∃ x0 : ℝ, g x0 = 0 ∧ g (3 - x0) = 0 := 
  sorry

end point_of_symmetry_l55_55353


namespace sum_of_all_max_values_of_f_l55_55860
open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := exp x * (sin x - cos x)

-- The interval for x
def interval := Set.Icc 0 (2011 * π)

-- The correct answer
def sum_of_max_values : ℝ := (exp π * (1 - exp (2010 * π))) / (1 - exp (2 * π))

theorem sum_of_all_max_values_of_f :
  (∑ k in Finset.range 1005, exp ((2 * k + 1) * π)) = sum_of_max_values := sorry

end sum_of_all_max_values_of_f_l55_55860


namespace vec_magnitude_is_five_l55_55057

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55057


namespace b_share_in_partnership_l55_55290

theorem b_share_in_partnership 
    (investment_a : ℕ := 7000)
    (investment_b : ℕ := 11000)
    (investment_c : ℕ := 18000)
    (duration_months : ℕ := 8)
    (share_a : ℕ := 1400)
    (total_ratio : ℕ := 36) 
    (ratio_a : ℕ := 7)
    (ratio_b : ℕ := 11)
    (total_profit : ℕ := 7200) :
    (share_b : ℕ) :=
  share_b = (ratio_b / total_ratio.toRat() * total_profit).toNat :=
by 
  -- proof omitted
  sorry

end b_share_in_partnership_l55_55290


namespace prob_non_distinct_real_roots_l55_55762

noncomputable def probability_non_distinct_real_roots : ℚ :=
let b_values := set.range (-4..5), -- { -4, -3, -2, -1, 0, 1, 2, 3, 4 },
    c_values := set.range (0..5),  -- { 0, 1, 2, 3, 4 }
    total_pairs := b_values ×ˢ c_values in
let count_valid_pairs := total_pairs.filter (λ (bc : ℤ × ℤ), bc.1 ^ 2 - 4 * bc.2 ≤ 0) in
(count_valid_pairs.to_finset.card.to_rat) / (total_pairs.to_finset.card.to_rat)

theorem prob_non_distinct_real_roots : probability_non_distinct_real_roots = 3 / 5 :=
sorry

end prob_non_distinct_real_roots_l55_55762


namespace sum_when_max_power_less_500_l55_55719

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55719


namespace max_diagonals_of_regular_1000gon_l55_55476

theorem max_diagonals_of_regular_1000gon :
  ∃ (d : ℕ), d = 2000 ∧ 
    ∀ (chosen_diagonals : finset (fin 1000 × fin 1000)),
      chosen_diagonals.card = d →
      ∀ (diagonal_triplets : finset (finset (fin 1000 × fin 1000))),
        diagonal_triplets.card = 3 →
        ∃ (diagonals_with_same_length : finset (fin 1000 × fin 1000)),
          diagonals_with_same_length ⊆ diagonal_triplets ∧
          diagonals_with_same_length.card ≥ 2 :=
begin
  sorry
end

end max_diagonals_of_regular_1000gon_l55_55476


namespace geom_seq_m_l55_55491

theorem geom_seq_m (m : ℤ) :
  let a1 := 3 + m - 5,
      a2 := 3^2 + m - 5,
      a3 := 3^3 + m - 5 in
  a2^2 = a1 * a3 → m = 4 :=
begin
  sorry
end

end geom_seq_m_l55_55491


namespace arccos_cos_11_l55_55349

theorem arccos_cos_11 : Real.arccos (Real.cos 11) = 1.425 :=
by
  sorry

end arccos_cos_11_l55_55349


namespace minimum_fourth_day_rain_l55_55607

def rainstorm_duration : Nat := 4
def area_capacity_feet : Nat := 6
def area_capacity_inches : Nat := area_capacity_feet * 12 -- Convert to inches
def drainage_rate : Nat := 3 -- inches per day
def rainfall_day1 : Nat := 10
def rainfall_day2 : Nat := 2 * rainfall_day1
def rainfall_day3 : Nat := (3 * rainfall_day1) -- 50% more than Day 2
def total_rain_first_three_days : Nat := rainfall_day1 + rainfall_day2 + rainfall_day3
def drained_amount : Nat := 3 * drainage_rate
def effective_capacity : Nat := area_capacity_inches - drained_amount
def overflow_capacity_left : Nat := effective_capacity - total_rain_first_three_days

theorem minimum_fourth_day_rain : Nat :=
  overflow_capacity_left + 1 = 4

end minimum_fourth_day_rain_l55_55607


namespace window_width_l55_55319

theorem window_width (h_pane_height : ℕ) (h_to_w_ratio_num : ℕ) (h_to_w_ratio_den : ℕ) (gaps : ℕ) 
(border : ℕ) (columns : ℕ) 
(panes_per_row : ℕ) (pane_height : ℕ) 
(heights_equal : h_pane_height = pane_height)
(ratio : h_to_w_ratio_num * pane_height = h_to_w_ratio_den * panes_per_row)
: columns * (h_to_w_ratio_den * pane_height / h_to_w_ratio_num) + 
  gaps + 2 * border = 57 := sorry

end window_width_l55_55319


namespace walter_age_in_2005_l55_55703

theorem walter_age_in_2005 
  (y : ℕ) (gy : ℕ)
  (h1 : gy = 3 * y)
  (h2 : (2000 - y) + (2000 - gy) = 3896) : y + 5 = 31 :=
by {
  sorry
}

end walter_age_in_2005_l55_55703


namespace percentage_of_freshmen_l55_55334

-- Conditions and given data
variables (T : ℝ) -- Total number of students
variables (F : ℝ) -- Percentage of freshmen in decimal form
variables (liberal_arts_percentage : ℝ) (psychology_majors_percentage : ℝ) (freshmen_psychology_majors_percentage : ℝ)

-- Defining the conditions based on the problem description
def condition_1 := liberal_arts_percentage = 0.5
def condition_2 := psychology_majors_percentage = 0.5
def condition_3 := freshmen_psychology_majors_percentage = 0.1

-- The equality we need to prove
theorem percentage_of_freshmen : condition_1 → condition_2 → condition_3 → 
  (0.25 * F * T = 0.1 * T) → F = 0.4 :=
by
  intros h1 h2 h3 h4
  have h5 : 0.25 * F = 0.1,
  { rw [h4],  -- Rewrite the hypothesis h4
    sorry }
  have h6 : F = 0.4,
  { linarith, }
  exact h6

end percentage_of_freshmen_l55_55334


namespace carol_lollipops_l55_55345

theorem carol_lollipops (total_lollipops : ℝ) (first_day_lollipops : ℝ) (delta_lollipops : ℝ) :
  total_lollipops = 150 → delta_lollipops = 5 →
  (first_day_lollipops + (first_day_lollipops + 5) + (first_day_lollipops + 10) +
  (first_day_lollipops + 15) + (first_day_lollipops + 20) + (first_day_lollipops + 25) = total_lollipops) →
  (first_day_lollipops = 12.5) →
  (first_day_lollipops + 15 = 27.5) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end carol_lollipops_l55_55345


namespace hitting_probability_l55_55315

theorem hitting_probability (P_miss : ℝ) (P_6 P_7 P_8 P_9 P_10 : ℝ) :
  P_miss = 0.2 →
  P_6 = 0.1 →
  P_7 = 0.2 →
  P_8 = 0.3 →
  P_9 = 0.15 →
  P_10 = 0.05 →
  1 - P_miss = 0.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hitting_probability_l55_55315


namespace single_serving_weight_l55_55782

-- Define the quantities needed for 12 servings
def servings := 12
def chicken_pounds := 4.5
def stuffing_ounces := 24
def broth_fluid_ounces := 8
def butter_tablespoons := 12

-- Define the conversion factors
def pounds_to_grams := 453.592
def ounces_to_grams := 28.3495
def fluid_ounces_to_grams := 29.5735
def tablespoons_to_grams := 14.1748

-- Calculate total weight in grams
def total_weight_grams :=
  chicken_pounds * pounds_to_grams +
  stuffing_ounces * ounces_to_grams +
  broth_fluid_ounces * fluid_ounces_to_grams +
  butter_tablespoons * tablespoons_to_grams

-- Calculate the weight of a single serving
def single_serving_weight_grams := total_weight_grams / servings

theorem single_serving_weight :
  single_serving_weight_grams ≈ 260.68647 :=
by
  -- The equivalence can be proven by performing the calculations and rounding to the desired precision.
  have h : single_serving_weight_grams = 260.68647 := sorry,
  exact h

end single_serving_weight_l55_55782


namespace frame_width_is_7_l55_55611

-- Given conditions
def height := 12 -- in centimeters
def circumference := 38 -- in centimeters

-- Define the width such that the perimeter equation holds
def width (height circumference : ℕ) := (circumference - 2 * height) / 2

-- The theorem we need to prove
theorem frame_width_is_7 (h c : ℕ) (h_eq : h = 12) (c_eq : c = 38) : width h c = 7 := by
  have h1 : height = 12 := by rw [h_eq]
  have c1 : circumference = 38 := by rw [c_eq]
  rw [width, h1, c1]
  norm_num
  sorry

end frame_width_is_7_l55_55611


namespace domain_of_c_eq_real_l55_55759

theorem domain_of_c_eq_real (m : ℝ) : (∀ x : ℝ, m * x^2 - 3 * x + 2 * m ≠ 0) ↔ (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) :=
by
  sorry

end domain_of_c_eq_real_l55_55759


namespace graph_not_in_first_quadrant_l55_55831

noncomputable def does_not_pass_first_quadrant (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x ≤ 0

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_b : b < -1) :
  does_not_pass_first_quadrant a b (λ x, a^x + b) :=
by
  sorry

end graph_not_in_first_quadrant_l55_55831


namespace area_of_region_of_semicircle_outside_circumcircle_l55_55854

noncomputable def side_length_of_square (r : ℝ) : ℝ := r * 2 / Real.sqrt 2

noncomputable def semicircle_area_outside_circle (r : ℝ) : ℝ :=
  let side := side_length_of_square r
  let diameter := side
  let semicircle_area := (π * (diameter / 2) ^ 2) / 2
  let sector_area := (π * r ^ 2) / 4
  let triangle_area := (1 / 2) * r * r
  semicircle_area - (sector_area - triangle_area)

theorem area_of_region_of_semicircle_outside_circumcircle
  (r : ℝ) (h : r = 10) :
  semicircle_area_outside_circle r = 50 :=
by
  rw [h]
  dsimp [semicircle_area_outside_circle, side_length_of_square]
  sorry

end area_of_region_of_semicircle_outside_circumcircle_l55_55854


namespace trigonometric_identity_l55_55834

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
sorry

end trigonometric_identity_l55_55834


namespace cosine_dihedral_angle_P_MND_l55_55698

theorem cosine_dihedral_angle_P_MND 
  (PA AB AD : ℝ) 
  (PA_perp_plane : PA = AB)
  (PA_eq : PA = 2)
  (AB_eq : AB = 2)
  (AD_eq : AD = 6)
  (M_on_AB : ∃ M, true) -- Placeholder for "M is on AB"
  (N_on_BC : ∃ N, true) -- Placeholder for "N is on BC"
  (perimeter_minimized : ∀ PM MN ND : ℝ, PM + MN + ND = minimized_value) :
  cos_dihedral_angle P_MND = (√6 / 6) :=
sorry

end cosine_dihedral_angle_P_MND_l55_55698


namespace vec_magnitude_is_five_l55_55058

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55058


namespace max_diagonals_same_length_l55_55477

theorem max_diagonals_same_length (n : ℕ) (h : n = 1000) : 
  ∃ m, m = 2000 ∧ 
  (∀ (d : finset (ℕ × ℕ)), d.card = m → 
    ∀ (a b c : (ℕ × ℕ)), a ∈ d → b ∈ d → c ∈ d → (a.2 - a.1 % n) = (b.2 - b.1 % n) ∨ (b.2 - b.1 % n) = (c.2 - c.1 % n) ∨ (a.2 - a.1 % n) = (c.2 - c.1 % n)
  ) :=
sorry

end max_diagonals_same_length_l55_55477


namespace vector_magnitude_difference_l55_55088

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55088


namespace karlsson_candies_28_l55_55966

def karlsson_max_candies (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem karlsson_candies_28 : karlsson_max_candies 28 = 378 := by
  sorry

end karlsson_candies_28_l55_55966


namespace min_photos_needed_to_ensure_conditions_l55_55807

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55807


namespace gross_profit_percentage_without_discount_l55_55673

theorem gross_profit_percentage_without_discount (C P : ℝ)
  (discount : P * 0.9 = C * 1.2)
  (discount_profit : C * 0.2 = P * 0.9 - C) :
  (P - C) / C * 100 = 33.3 :=
by
  sorry

end gross_profit_percentage_without_discount_l55_55673


namespace david_profit_l55_55685

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l55_55685


namespace extra_minutes_per_A_l55_55253

theorem extra_minutes_per_A 
  (x : ℕ)
  (normal_recess : ℕ := 20)
  (extra_A : ℕ := x)
  (extra_B : ℕ := 1)
  (extra_C : ℕ := 0)
  (reduce_D : ℕ := -1)
  (num_A : ℕ := 10)
  (num_B : ℕ := 12)
  (num_C : ℕ := 14)
  (num_D : ℕ := 5)
  (total_recess : ℕ := 47) 
  (total_extra_time : ℕ := total_recess - normal_recess) :
  10 * x + 12 - 5 = total_extra_time → x = 2 :=
by
  intro h
  sorry

end extra_minutes_per_A_l55_55253


namespace expected_intervals_containing_i_l55_55503

open Nat

/-
  Define the initial interval and the recursive construction of the set S.
-/
def initial_interval : Set (ℕ × ℕ) := {(1, 1000)}

noncomputable def interval_split : ℕ × ℕ → Set (ℕ × ℕ)
  | (l, r) => if l ≠ r then {(l, (l + r) / 2), ((l + r) / 2 + 1, r)} else ∅

noncomputable def S : Set (ℕ × ℕ) := {i | ∃ l r, (l, r) ∈ initial_interval ∨ (l, r) ∈ S ∧ (i ∈ interval_split (l, r))}

noncomputable def E (n : ℕ) : ℕ :=
  if n = 1 then 1 else
    1 + (n / 2) / n * E (n / 2) + (n - n / 2) / n * E (n - n / 2)

theorem expected_intervals_containing_i : E 1000 = 11 := by
  sorry

end expected_intervals_containing_i_l55_55503


namespace complex_modulus_multiplication_theta_obtuse_second_quadrant_imaginary_parts_equal_polynomial_root_conjugate_l55_55283

-- Statement A
theorem complex_modulus_multiplication (z1 z2 : ℂ) : abs (z1 * z2) = abs z1 * abs z2 :=
sorry

-- Statement B
theorem theta_obtuse_second_quadrant (theta : ℝ) : 
  (θ > π / 2 ∧ θ < π) ↔ (∃ (z : ℂ), z = complex.exp (theta * complex.I) ∧ z.re < 0 ∧ z.im > 0) :=
sorry

-- Statement C
theorem imaginary_parts_equal (z1 z2 : ℂ) : (z1 = z2) → (z1.im = z2.im) :=
sorry

-- Statement D
theorem polynomial_root_conjugate (p q : ℝ) : 
  (2 * complex.I - 3 ∈ complex.roots (polynomial.C q + polynomial.X * polynomial.C p + polynomial.X^2 * polynomial.C 2)) → 
  ((2 * complex.I + 3) ∈ set_of (λ z, z = conj (2 * complex.I - 3))) :=
sorry

end complex_modulus_multiplication_theta_obtuse_second_quadrant_imaginary_parts_equal_polynomial_root_conjugate_l55_55283


namespace vector_magnitude_difference_l55_55018

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55018


namespace xiao_ming_correctly_answered_question_count_l55_55320

-- Define the given conditions as constants and variables
def total_questions : ℕ := 20
def points_per_correct : ℕ := 8
def points_deducted_per_incorrect : ℕ := 5
def total_score : ℕ := 134

-- Prove that the number of correctly answered questions is 18
theorem xiao_ming_correctly_answered_question_count :
  ∃ (correct_count incorrect_count : ℕ), 
      correct_count + incorrect_count = total_questions ∧
      correct_count * points_per_correct - 
      incorrect_count * points_deducted_per_incorrect = total_score ∧
      correct_count = 18 :=
by
  sorry

end xiao_ming_correctly_answered_question_count_l55_55320


namespace Jackie_hops_six_hops_distance_l55_55164

theorem Jackie_hops_six_hops_distance : 
  let a : ℝ := 1
  let r : ℝ := 1 / 2
  let S : ℝ := a * ((1 - r^6) / (1 - r))
  S = 63 / 32 :=
by 
  sorry

end Jackie_hops_six_hops_distance_l55_55164


namespace area_triangle_AEB_l55_55152

variables {α : Type*} [linear_ordered_field α]

def Rectangle (A B C D : α × α) (AB BC : α) := 
  A = (0, 0) ∧ B = (AB, 0) ∧ C = (AB, BC) ∧ D = (0, BC)

def PointsOnSideCD (D C F G : α × α) (DF GC : α) :=
  D = (0, BC) ∧ C = (8, BC) ∧ F = (DF, BC) ∧ G = (8 - GC, BC)

-- variables representing the points
variables A B C D F G E : α × α

theorem area_triangle_AEB 
  (AB BC DF GC : α) 
  (h1 : Rectangle A B C D AB BC) 
  (h2 : PointsOnSideCD D C F G DF GC)
  (h3 : A = (0, 0)) (h4 : B = (AB, 0)) 
  (h5 : F = (DF, BC)) (h6 : G = (8 - GC, BC))
  (h7 : 8 - (DF + GC) = 3) -- derived from given conditions
  (h8 : E = (3, 8)) -- Intersection point
  : (1 / 2) * AB * 8 = 32 :=
  by sorry

end area_triangle_AEB_l55_55152


namespace combined_frosting_rate_l55_55337

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end combined_frosting_rate_l55_55337


namespace factorize_expression_l55_55378

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l55_55378


namespace average_number_of_stickers_per_album_is_correct_l55_55951

def average_stickers_per_album (albums : List ℕ) (n : ℕ) : ℚ := (albums.sum : ℚ) / n

theorem average_number_of_stickers_per_album_is_correct :
  average_stickers_per_album [5, 7, 9, 14, 19, 12, 26, 18, 11, 15] 10 = 13.6 := 
by
  sorry

end average_number_of_stickers_per_album_is_correct_l55_55951


namespace problem_one_problem_two_l55_55746

theorem problem_one :
  sqrt 9 - (-2023 : ℤ)^0 + 2⁻¹ = (5 : ℚ) / 2 :=
by sorry

theorem problem_two (a b : ℚ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2 * b)) = 2 / (a + b) :=
by sorry

end problem_one_problem_two_l55_55746


namespace spherical_to_rectangular_coordinates_l55_55355

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  sphericalToRectangular 10 (5 * Real.pi / 4) (Real.pi / 4) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l55_55355


namespace vector_addition_result_l55_55832

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def given_a : a = (2, 3) := rfl
def given_b : b = (-1, 5) := rfl

theorem vector_addition_result (a b : ℝ × ℝ) (ha : a = (2, 3)) (hb : b = (-1, 5)) :
  a + (3 • b) = (-1, 18) :=
by
  sorry

end vector_addition_result_l55_55832


namespace gcd_204_85_l55_55271

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l55_55271


namespace vector_magnitude_correct_l55_55043

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55043


namespace heaviest_box_difference_l55_55601

theorem heaviest_box_difference (a b c d : ℕ) (h : a < b) (h1 : b < c) (h2 : c < d)
  (pairs : multiset ℕ) (weights : multiset ℕ)
  (hpairs : pairs = [a + b, a + c, a + d, b + c, b + d, c + d])
  (hweights : weights = [22, 23, 27, 29, 30]) :
  (d - a) = 7 :=
by {
  sorry
}

end heaviest_box_difference_l55_55601


namespace hyperbola_eccentricity_l55_55238

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3: c^2 = a^2 + b^2)
  (h4 : ∀ F M, F = (-c, 0) → 
    let A := (a * c / (b - a), b * c / (b - a)),
        B := (a * c / (-b - a), -b * c / (-b - a)),
        M := ((a^2 * c) / (b^2 - a^2), (b^2 * c) / (b^2 - a^2)) in
    ∥(fst M) + c, (snd M)∥ = c) : 
  let e := c / a in
  e^2 = sqrt 2 :=
sorry

end hyperbola_eccentricity_l55_55238


namespace problem_statement_l55_55502

noncomputable def S : ℝ :=
  (1 / (2 - real.cbrt 7)) + (1 / (real.cbrt 7 - real.sqrt 3)) - (1 / (real.sqrt 3 - real.cbrt 2))

theorem problem_statement : S > 1 :=
by
  sorry

end problem_statement_l55_55502


namespace find_a_value_l55_55132

open Real

theorem find_a_value (a : ℝ) :
    let p1 := (a - 2, -1) in
    let p2 := (-a - 2, 1) in
    let m1 := ((-a - 2) - (a - 2)) in
    let m := -2 / 3 in
    (1 + 1) / m1 = 3 / 2 →
    a = -2 / 3 :=
sorry

end find_a_value_l55_55132


namespace arccos_sin_eq_pi_div_two_sub_1_72_l55_55350

theorem arccos_sin_eq_pi_div_two_sub_1_72 :
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 :=
sorry

end arccos_sin_eq_pi_div_two_sub_1_72_l55_55350


namespace max_real_roots_l55_55363

noncomputable def P (n : ℕ) := 
  Finset.range (2*n + 1) |>.sum (λ k => (1 : ℝ) * (X : Polynomial ℝ)^(2*n - k))

theorem max_real_roots (n : ℕ) (hn : 0 < n) :
  (if odd n then (P n).root_multiplicity 1 + (P n).root_multiplicity (-1) = 1
  else (P n).root_multiplicity 1 + (P n).root_multiplicity (-1) = 0) := 
sorry

end max_real_roots_l55_55363


namespace total_area_of_triangular_houses_l55_55640

def base : ℕ := 40
def height : ℕ := 20
def number_of_houses : ℕ := 3
def area_of_one_house : ℕ := (base * height) / 2
def total_area : ℕ := area_of_one_house * number_of_houses

theorem total_area_of_triangular_houses : total_area = 1200 := by
  sorry

end total_area_of_triangular_houses_l55_55640


namespace min_value_f_gt_two_max_value_g_zero_l55_55579

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

noncomputable def g (x m : ℝ) : ℝ := Real.log x - Real.exp (x - m)

theorem min_value_f_gt_two {m : ℝ} 
  (h : ∃ x₀, (∀ x, f x₀ ≤ f x) ∧ m = f x₀) : m > 2 := 
sorry

theorem max_value_g_zero {m : ℝ} 
  (h : ∃ x, (∀ y, g x m ≤ g y m) ∧ g x m = 0) : (∃ x, ∀ y, g x m ≤ g y m) ∧ max (Set.range (λ x, g x m)) 0 :=
sorry

end min_value_f_gt_two_max_value_g_zero_l55_55579


namespace beetles_consumed_per_day_l55_55480

-- Definitions
def bird_eats_beetles (n : Nat) : Nat := 12 * n
def snake_eats_birds (n : Nat) : Nat := 3 * n
def jaguar_eats_snakes (n : Nat) : Nat := 5 * n
def crocodile_eats_jaguars (n : Nat) : Nat := 2 * n

-- Initial values
def initial_jaguars : Nat := 6
def initial_crocodiles : Nat := 30
def net_increase_birds : Nat := 4
def net_increase_snakes : Nat := 2
def net_increase_jaguars : Nat := 1

-- Proof statement
theorem beetles_consumed_per_day : 
  bird_eats_beetles (snake_eats_birds (jaguar_eats_snakes initial_jaguars)) = 1080 := 
by 
  sorry

end beetles_consumed_per_day_l55_55480


namespace fermat_point_distance_sum_l55_55148

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem fermat_point_distance_sum :
  let D := (0 : ℝ, 0 : ℝ)
  let E := (8 : ℝ, 0 : ℝ)
  let F := (2 : ℝ, 4 : ℝ)
  let Q := (3 : ℝ, 1 : ℝ)
  let DQ := distance D Q
  let EQ := distance E Q
  let FQ := distance F Q
  in (x + y = 3) :=
begin
  let x := 2
  let y := 1
  sorry
end

end fermat_point_distance_sum_l55_55148


namespace slope_of_dividing_line_l55_55153

-- Define the vertices of the T-shaped region
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 4⟩
def C : Point := ⟨4, 4⟩
def D : Point := ⟨4, 2⟩
def E : Point := ⟨7, 2⟩
def F : Point := ⟨7, 0⟩

-- Areas of rectangles ABCD and DEFA explicitly defined
def areaABCD : ℝ := 16
def areaDEFA : ℝ := 6
def totalArea : ℝ := areaABCD + areaDEFA

-- The total area of the T-shaped region
def halfArea : ℝ := totalArea / 2

-- This is the theorem we need to prove
theorem slope_of_dividing_line : 
  ∃ m : ℝ, 
  (m ≠ 0) ∧ 
  (splitsAreaEvenly m halfArea) := 
  sorry

-- Function that indicates if a line with slope 'm' splits the area evenly.
noncomputable def splitsAreaEvenly (m : ℝ) (area : ℝ) : Prop := sorry

end slope_of_dividing_line_l55_55153


namespace floor_e_eq_2_l55_55769

theorem floor_e_eq_2 : (⌊real.exp 1⌋ = 2) :=
sorry

end floor_e_eq_2_l55_55769


namespace sports_probability_boy_given_sports_probability_l55_55469

variable (x : ℝ) -- Number of girls

def number_of_boys := 1.5 * x
def boys_liking_sports := 0.4 * number_of_boys x
def girls_liking_sports := 0.2 * x
def total_students := x + number_of_boys x
def total_students_liking_sports := boys_liking_sports x + girls_liking_sports x

theorem sports_probability : (total_students_liking_sports x) / (total_students x) = 8 / 25 := 
sorry

theorem boy_given_sports_probability :
  (boys_liking_sports x) / (total_students_liking_sports x) = 3 / 4 := 
sorry

end sports_probability_boy_given_sports_probability_l55_55469


namespace fruit_count_correct_l55_55829

def george_oranges := 45
def amelia_oranges := george_oranges - 18
def amelia_apples := 15
def george_apples := amelia_apples + 5

def olivia_orange_rate := 3
def olivia_apple_rate := 2
def olivia_minutes := 30
def olivia_cycle_minutes := 5
def olivia_cycles := olivia_minutes / olivia_cycle_minutes
def olivia_oranges := olivia_orange_rate * olivia_cycles
def olivia_apples := olivia_apple_rate * olivia_cycles

def total_oranges := george_oranges + amelia_oranges + olivia_oranges
def total_apples := george_apples + amelia_apples + olivia_apples
def total_fruits := total_oranges + total_apples

theorem fruit_count_correct : total_fruits = 137 := by
  sorry

end fruit_count_correct_l55_55829


namespace party_spending_l55_55677

theorem party_spending :
  ∃ (C T H G : ℕ), 
    (25 * (4 * T / 5) = C) ∧
    (T = C) ∧ 
    (20 * (3 * H / 4) = T) ∧
    (H = (6 * T / 5)) ∧ 
    (18 * (4 * G / 3) = H) ∧ 
    (H = 2 * G) ∧
    (C + T + H + G = 133) ∧
    (G = 21) ∧
    (H = 42) ∧
    (T = 35) ∧
    (C = 35) :=
begin
  sorry
end

end party_spending_l55_55677


namespace prime_cubed_remainder_210_l55_55333

open Nat

theorem prime_cubed_remainder_210 (p : ℕ) (hp : Prime p) (hp_gt_5 : p > 5) :
  ∃ (S : Set ℕ), (∀ r ∈ S, r < 210 ∧ ∃ k, p^3 = 210 * k + r) ∧ S.card = number_of_different_remainders(p) := by
  sorry

end prime_cubed_remainder_210_l55_55333


namespace gerald_speed_average_l55_55210

theorem gerald_speed_average
  (track_length : ℝ)
  (polly_laps : ℕ)
  (polly_time_hours : ℝ)
  (gerald_speed_ratio : ℝ) :
  track_length = 0.25 →
  polly_laps = 12 →
  polly_time_hours = 0.5 →
  gerald_speed_ratio = 0.5 →
  let polly_speed := (polly_laps * track_length) / polly_time_hours in
  let gerald_speed := polly_speed * gerald_speed_ratio in
  gerald_speed = 3 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at *
  have polly_speed := (12 * 0.25) / 0.5
  have gerald_speed := polly_speed * 0.5
  simp [gerald_speed]
  sorry

end gerald_speed_average_l55_55210


namespace problem_I_problem_II_l55_55780

theorem problem_I : (-7 / 8) ^ 0 + (1 / 8) ^ (-1 / 3) + 4 * (3 - Real.pi) ^ 4 = Real.pi :=
by
  sorry

theorem problem_II : 7 ^ (Real.log 2 / Real.log 7) + Real.log 25 / Real.log 10 + 2 * Real.log 2 / Real.log 10 - Real.log (Real.sqrt (Real.exp 3)) = 5 / 2 :=
by
  sorry

end problem_I_problem_II_l55_55780


namespace vector_magnitude_subtraction_l55_55002

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55002


namespace find_prime_A_l55_55676

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_A (A : ℕ) :
  is_prime A ∧ is_prime (A + 14) ∧ is_prime (A + 18) ∧ is_prime (A + 32) ∧ is_prime (A + 36) → A = 5 := by
  sorry

end find_prime_A_l55_55676


namespace total_distance_is_20_l55_55678

noncomputable def total_distance_walked (x : ℝ) : ℝ :=
  let flat_distance := 4 * x
  let uphill_time := (2 / 3) * (5 - x)
  let uphill_distance := 3 * uphill_time
  let downhill_time := (1 / 3) * (5 - x)
  let downhill_distance := 6 * downhill_time
  flat_distance + uphill_distance + downhill_distance

theorem total_distance_is_20 :
  ∃ x : ℝ, x >= 0 ∧ x <= 5 ∧ total_distance_walked x = 20 :=
by
  -- The existence proof is omitted (hence the sorry)
  sorry

end total_distance_is_20_l55_55678


namespace P_is_centroid_of_triangle_l55_55838

-- Definitions for lattice points and triangles

def IsLatticePoint (P : ℤ × ℤ) : Prop := 
  ∃ (x y : ℤ), P = (x, y)

def IsLatticeTriangle (A B C : ℤ × ℤ) : Prop :=
  IsLatticePoint A ∧ IsLatticePoint B ∧ IsLatticePoint C ∧
  (¬ ∃ (D : ℤ × ℤ), IsLatticePoint D ∧ (D ≠ A ∧ D ≠ B ∧ D ≠ C) ∧ 
    OnLine A B D) ∧
  (¬ ∃ (E : ℤ × ℤ), IsLatticePoint E ∧ (E ≠ B ∧ E ≠ C ∧ E ≠ A) ∧ 
    OnLine B C E) ∧
  (¬ ∃ (F : ℤ × ℤ), IsLatticePoint F ∧ (F ≠ C ∧ F ≠ A ∧ F ≠ B) ∧ 
    OnLine C A F)

def UniqueLatticeInteriorPoint (A B C P : ℤ × ℤ) : Prop :=
  IsLatticePoint P ∧
  ∃! Q, IsLatticePoint Q ∧ IsInInterior A B C Q

-- Mathematical statement in Lean (without proof)

theorem P_is_centroid_of_triangle {A B C P : ℤ × ℤ} (h1 : IsLatticeTriangle A B C)
  (h2 : UniqueLatticeInteriorPoint A B C P) :
  IsCentroidOfTriangle A B C P :=
sorry

end P_is_centroid_of_triangle_l55_55838


namespace total_trapezoid_area_l55_55309

def large_trapezoid_area (AB CD altitude_L : ℝ) : ℝ :=
  0.5 * (AB + CD) * altitude_L

def small_trapezoid_area (EF GH altitude_S : ℝ) : ℝ :=
  0.5 * (EF + GH) * altitude_S

def total_area (large_area small_area : ℝ) : ℝ :=
  large_area + small_area

theorem total_trapezoid_area :
  large_trapezoid_area 60 30 15 + small_trapezoid_area 25 10 5 = 762.5 :=
by
  -- proof goes here
  sorry

end total_trapezoid_area_l55_55309


namespace toys_left_l55_55367

-- Given conditions
def initial_toys := 7
def sold_toys := 3

-- Proven statement
theorem toys_left : initial_toys - sold_toys = 4 := by
  sorry

end toys_left_l55_55367


namespace books_no_adjacent_l55_55254

-- Define our main theorem
theorem books_no_adjacent (n k : ℕ) (h1 : n = 12) (h2 : k = 5) :
    ∃ ways : ℕ, ways = Nat.choose (n - k + 1) k ∧ ways = 56 :=
by
  have h : Nat.choose (12 - 5 + 1) 5 = 56 := by
    -- Use the given mathematical fact
    calc
      Nat.choose 8 5 = Nat.choose 8 3 : by rw [Nat.choose_symm (by linarith)]
               ... = 56 : by decide
  use 56
  constructor
  · exact h
  · rfl

end books_no_adjacent_l55_55254


namespace min_m_n_l55_55441

noncomputable def f (x m n : ℝ) := log x - 2 * m * x^2 - n

theorem min_m_n {m n : ℝ} (h : ∃ x, f x m n = -log 2) : m + n = 1/2 * log 2 :=
by sorry

end min_m_n_l55_55441


namespace vector_magnitude_correct_l55_55037

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55037


namespace time_to_analyze_one_bone_l55_55548

theorem time_to_analyze_one_bone :
  ∀ (total_hours total_bones : ℕ), total_hours = 206 → total_bones = 206 → (total_hours / total_bones = 1) :=
by
  intros total_hours total_bones h_hours h_bones
  rw [h_hours, h_bones]
  exact Nat.div_self (by norm_num : 206 ≠ 0)

end time_to_analyze_one_bone_l55_55548


namespace find_angle_A_l55_55869

noncomputable def angle_A (a b : ℝ) (B : ℝ) : ℝ :=
  Real.arcsin ((a * Real.sin B) / b)

theorem find_angle_A :
  ∀ (a b : ℝ) (angle_B : ℝ), 0 < a → 0 < b → 0 < angle_B → angle_B < 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  angle_B = 60 →
  angle_A a b angle_B = 45 :=
by
  intros a b angle_B h1 h2 h3 h4 ha hb hB
  have ha' : a = Real.sqrt 2 := ha
  have hb' : b = Real.sqrt 3 := hb
  have hB' : angle_B = 60 := hB
  -- Proof omitted for demonstration
  sorry

end find_angle_A_l55_55869


namespace distinct_triangles_octahedron_l55_55112

-- Define the regular octahedron with 8 vertices and the collinearity property
def regular_octahedron_vertices : Nat := 8

def no_three_vertices_collinear (vertices: Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), {a, b, c} ⊆ vertices → ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r

-- The main theorem stating the problem
theorem distinct_triangles_octahedron :
  no_three_vertices_collinear (Finset.range regular_octahedron_vertices) →
  (Finset.card (Finset.univ.choose 3)) = 56 :=
by
  sorry

end distinct_triangles_octahedron_l55_55112


namespace energy_consumption_per_phone_l55_55920

-- Define the conditions
def total_energy_consumption : ℕ := 19
def number_of_phones : ℕ := 9

-- Define what we need to prove
theorem energy_consumption_per_phone : (total_energy_consumption : ℚ) / (number_of_phones : ℚ) ≈ 2.11 :=
by
  -- Skip the proof with sorry
  sorry

end energy_consumption_per_phone_l55_55920


namespace total_sentence_l55_55958

theorem total_sentence (base_rate : ℝ) (value_stolen : ℝ) (third_offense_increase : ℝ) (additional_years : ℕ) : 
  base_rate = 1 / 5000 → 
  value_stolen = 40000 → 
  third_offense_increase = 0.25 → 
  additional_years = 2 →
  (value_stolen * base_rate * (1 + third_offense_increase) + additional_years) = 12 := 
by
  intros
  sorry

end total_sentence_l55_55958


namespace find_two_digits_l55_55705

theorem find_two_digits (a b : ℕ) (h₁: a ≤ 9) (h₂: b ≤ 9)
  (h₃: (4 + a + b) % 9 = 0) (h₄: (10 * a + b) % 4 = 0) :
  (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8) :=
by {
  sorry
}

end find_two_digits_l55_55705


namespace find_n_values_l55_55942

def is_5_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999
def is_prime (n : ℕ) : Prop := Prime n
def cond (n q r : ℕ) : Prop := n = 50 * q + r ∧ 13 ∣ (q + 2 * r) ∧ is_prime r

theorem find_n_values : 
  let n_values := {n : ℕ | ∃ q r, is_5_digit n ∧ cond n q r} in
  n_values.finite ∧ n_values.to_finset.card = 1932 :=
by
  sorry

end find_n_values_l55_55942


namespace sum_of_reciprocals_l55_55595

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 := 
sorry

end sum_of_reciprocals_l55_55595


namespace unique_m_l55_55430

noncomputable def m (x : ℝ) : Prop := 
  ({0, x, x^2 - 3*x + 2} = {0, x, x^2 - 3*x + 2}) ∧ (2 ∈ {0, x, x^2 - 3*x + 2})

theorem unique_m : ∀ x : ℝ, m(x) → x = 3 :=
by
  intros x hx
  sorry

end unique_m_l55_55430


namespace MKLP_cyclic_l55_55479

theorem MKLP_cyclic (
  {A B C : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (ω_a : Excircle A)
  (ω_b : Excircle B)
  (P Q M N K L : TriangleSegment)
  (hPQ_touch : ω_a.touches AB AC P Q)
  (hMN_touch : ω_b.touches BA BC M N)
  (hK : Projection C MN K)
  (hL : Projection C PQ L)) :
  is_cyclic_quad K M L P :=
begin
  sorry
end

end MKLP_cyclic_l55_55479


namespace perpendicular_and_parallel_l55_55224

variables {l m n : Type} [Mathlib.Geometry.Line l] [Mathlib.Geometry.Line m] [Mathlib.Geometry.Line n] (α : Type _) [Mathlib.Geometry.Plane α]

theorem perpendicular_and_parallel (l m : Mathlib.Geometry.Line) (α : Mathlib.Geometry.Plane) :
  l ⊥ m → m ∉ α → l ⊥ α → m ∥ α :=
by sorry

end perpendicular_and_parallel_l55_55224


namespace rectangle_with_odd_stars_l55_55748

theorem rectangle_with_odd_stars:
  ∀ (dominos : ℕ) (rows : ℕ) (cols : ℕ),
  dominos = 540 →
  rows = 6 →
  cols = 180 →
  ∃ (matrix : matrix (fin rows) (fin cols) bool),
    (∀ i, (matrix.row i).count tt % 2 = 1) ∧ 
    (∀ j, (matrix.col j).count tt % 2 = 1) :=
by
  sorry

end rectangle_with_odd_stars_l55_55748


namespace units_digit_uniform_l55_55366

-- Definitions
def domain : Finset ℕ := Finset.range 15

def pick : Type := { n // n ∈ domain }

def uniform_pick : pick := sorry

-- Statement of the theorem
theorem units_digit_uniform :
  ∀ (J1 J2 K : pick), 
  ∃ d : ℕ, d < 10 ∧ (J1.val + J2.val + K.val) % 10 = d
:= sorry

end units_digit_uniform_l55_55366


namespace min_photos_l55_55791

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55791


namespace calvin_score_l55_55343

theorem calvin_score (C : ℚ) (h_paislee_score : (3/4) * C = 125) : C = 167 := 
  sorry

end calvin_score_l55_55343


namespace find_x_l55_55291

noncomputable def marble_problem (A V x : ℕ) : Prop :=
  (A + x = V - x) ∧ (V + 2 * x = A - 2 * x + 30)

theorem find_x (A V : ℕ) : ∃ x : ℕ, marble_problem A V x ∧ x = 5 :=
by
  -- problem conditions
  have h₁ : ∀ (A V x: ℕ), A + x = V - x → A = V - 2 * x := sorry,
  have h₂ : ∀ (A V x: ℕ), V + 2 * x = A - 2 * x + 30 := sorry,
  -- solution
  let x := 5 in
  use x,
  split,
  { split,
    { exact h₁ A V x },
    { exact h₂ A V x },
  },
  { refl },
  sorry 

end find_x_l55_55291


namespace grape_lollipops_count_l55_55749

theorem grape_lollipops_count (total_lollipops : ℕ) (percent_cherry percent_watermelon percent_sourapple : ℝ) 
  (h1 : total_lollipops = 60)
  (h2 : percent_cherry = 0.30)
  (h3 : percent_watermelon = 0.20)
  (h4 : percent_sourapple = 0.15)
  (h5 : ∀ x : ℝ, x = total_lollipops * (percent_cherry + percent_watermelon + percent_sourapple)) : 
  let remaining_lollipops := total_lollipops - (total_lollipops * percent_cherry + total_lollipops * percent_watermelon + total_lollipops * percent_sourapple).to_nat in
  let grape_lollipops := remaining_lollipops / 2 in
  grape_lollipops = 10 :=
by
  let cherry_lollipops : ℕ := (total_lollipops * percent_cherry).to_nat
  let watermelon_lollipops : ℕ := (total_lollipops * percent_watermelon).to_nat
  let sourapple_lollipops : ℕ := (total_lollipops * percent_sourapple).to_nat
  let accounted_lollipops := cherry_lollipops + watermelon_lollipops + sourapple_lollipops
  let remaining_lollipops := total_lollipops - accounted_lollipops
  let grape_lollipops := remaining_lollipops / 2
  have h6 : remaining_lollipops = 21 := by sorry
  have h7 : grape_lollipops = 10 := by sorry
  exact h7

end grape_lollipops_count_l55_55749


namespace at_most_p_minus_1_multiples_l55_55947

theorem at_most_p_minus_1_multiples (p : ℕ) (hp : p > 3) (hp_prime : Prime p) (hmod : p % 3 = 2) :
  let S := { z : ℤ | ∃ (x y : ℤ), 0 ≤ x ∧ x ≤ p - 1 ∧ 0 ≤ y ∧ y ≤ p - 1 ∧ z = y^2 - x^3 - 1 } in
  ∃ T : Finset ℤ, T ⊆ S ∧ T.card ≤ p - 1 ∧ ∀ t ∈ T, (t : ℤ) % p = 0 :=
by
  sorry

end at_most_p_minus_1_multiples_l55_55947


namespace numerical_passwords_part1_numerical_password_part2_l55_55906

-- Proof Problem 1: Polynomial factorization to form numerical passwords
theorem numerical_passwords_part1 (x : ℕ) (y : ℕ) (h1 : x = 15) (h2 : y = 5) :
  let z := x * (x - y) * (x + y) in
  z = 151020 ∨ z = 152010 ∨ z = 101520 ∨ z = 102015 ∨ z = 201510 ∨ z = 201015 :=
  sorry

-- Proof Problem 2: Numerical password from polynomial corresponding to sides of a triangle
theorem numerical_password_part2 (x : ℕ) (y : ℕ) (h1 : x + y = 13) (h2 : x^2 + y^2 = 121) :
  let z := x * y * (x^2 + y^2) in
  z = 24121 :=
  sorry

end numerical_passwords_part1_numerical_password_part2_l55_55906


namespace triangle_inequality_l55_55482

theorem triangle_inequality (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : a^2 + b^2 > c^2) : 
  sqrt 3 < c ∧ c < sqrt 5 :=
sorry

end triangle_inequality_l55_55482


namespace sin_squared_value_l55_55128

theorem sin_squared_value (x : ℝ) (h₁ : Real.sin x = 4 * Real.cos x) :
    Real.sin x * Real.sin x = 16 / 17 :=
by
  have h₂ : Real.sin x * Real.sin x + Real.cos x * Real.cos x = 1 := Real.sin_sq_add_cos_sq x
  sorry

end sin_squared_value_l55_55128


namespace find_OP_length_l55_55932

/-!
# Centroid Property in Triangle
Given a triangle ABC where AO and CQ are medians intersecting at centroid O,
prove that if OQ = 5 inches and CQ = 15 inches, then OP = 10 inches.
-/

-- We define the necessary elements and their relationships

def centroid_property (O A P C Q : Type*) [inhabited O] [inhabited A] [inhabited P] [inhabited C] [inhabited Q] : Prop :=
  -- O is the centroid of the triangle ABC
  -- CQ is a median, and OQ is a segment of CQ
  ∀ (CQ OQ : ℝ), CQ = 3 * OQ → ∃ (OP : ℝ), OP = 2 * OQ

theorem find_OP_length (O A P C Q : Type*) [inhabited O] [inhabited A] [inhabited P] [inhabited C] [inhabited Q]
  (hOQ : OQ = 5) (hCQ: CQ = 15) : ∃ OP : ℝ, OP = 10 :=
begin
  -- Using the centroid property to find OP
  have h_centroid := centroid_property O A P C Q,
  specialize h_centroid CQ OQ,
  rw [hOQ, hCQ] at h_centroid,
  simp at h_centroid,
  exact h_centroid,
end

end find_OP_length_l55_55932


namespace sum_of_a_and_b_l55_55733

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55733


namespace cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l55_55421

theorem cos_alpha_minus_11pi_div_12_eq_neg_2_div_3
  (α : ℝ)
  (h : Real.sin (7 * Real.pi / 12 + α) = 2 / 3) :
  Real.cos (α - 11 * Real.pi / 12) = -(2 / 3) :=
by
  sorry

end cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l55_55421


namespace max_sin_A_when_area_maximized_l55_55912

def BC : ℝ := 6
def AB (AC : ℝ) : ℝ := 2 * AC
def area (a b c : ℝ) (p : ℝ) : ℝ := Real.sqrt (p * (p - a) * (p - b) * (p - c))
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem max_sin_A_when_area_maximized :
  ∀ (AC : ℝ), BC = 6 → AB AC = 2 * AC → (∃ (S : ℝ), 
  let p := semiperimeter (AB AC) BC AC in
  S = area (AB AC) BC AC p) →
  ∃ (A : ℝ), ∃ (sin_A : ℝ), (sin_A = 3 / 5 ∧ S = max_area → sin_A = 3 / 5) :=
sorry

end max_sin_A_when_area_maximized_l55_55912


namespace angle_CPD_is_87_degrees_l55_55473

theorem angle_CPD_is_87_degrees
    (PC_tangent_SAR : ∀ (P C S A R : Point),
      tangent (Segment P C) (Semicircle S A R))
    (PD_tangent_RBT : ∀ (P D R B T : Point),
      tangent (Segment P D) (Semicircle R B T))
    (SRT_straight_line : ∀ (S R T : Point),
      collinear [S, R, T])
    (arc_AS_34 : ∀ (A S : Point),
      arc_measure (Arc A S) = 34)
    (arc_BT_53 : ∀ (B T : Point),
      arc_measure (Arc B T) = 53) :
    ∠ C P D = 87 := by
  sorry

end angle_CPD_is_87_degrees_l55_55473


namespace vector_magnitude_difference_l55_55021

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55021


namespace correct_option_D_l55_55507

def U : Set ℕ := {1, 2, 4, 6, 8}
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

theorem correct_option_D : A ∩ complement_U_B = {1} := by
  sorry

end correct_option_D_l55_55507


namespace correct_a_c_d_l55_55510

-- Proving that f(x) = ((e^x) - a)/x - a * ln(x) has specific properties given conditions

noncomputable def e := Real.exp 1
noncomputable def f (x a : ℝ) := (Real.exp x - a)/x - a * Real.log x

theorem correct_a_c_d (a : ℝ) :
  (∀ x > 0, a = e → ¬∃ c, ∃ y > 0, f'(c) < 0 ∧ y ≠ c) ∧
  ((1 < a ∧ a < e) → ∃ y > 0, f(y, a) = 0 ∧ ∀ z < y, f(z, a) > 0) ∧
  (a ≤ 1 → ¬∃ y > 0, f(y, a) = 0) :=
by
  sorry

end correct_a_c_d_l55_55510


namespace decimals_between_6_1_and_6_4_are_not_two_l55_55257

-- Definitions from the conditions in a)
def is_between (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

-- The main theorem statement
theorem decimals_between_6_1_and_6_4_are_not_two :
  ∀ x, is_between x 6.1 6.4 → false :=
by
  sorry

end decimals_between_6_1_and_6_4_are_not_two_l55_55257


namespace find_a_plus_b_l55_55853

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem find_a_plus_b (a b : ℝ) 
  (h1 : f 1 a b = 10) 
  (h2 : deriv (λ x, f x a b) 1 = 0) : 
  a + b = 3 := by
  sorry

end find_a_plus_b_l55_55853


namespace sum_3n_terms_l55_55416

variable {a_n : ℕ → ℝ} -- Definition of the sequence
variable {S : ℕ → ℝ} -- Definition of the sum function

-- Conditions
axiom sum_n_terms (n : ℕ) : S n = 3
axiom sum_2n_terms (n : ℕ) : S (2 * n) = 15

-- Question and correct answer
theorem sum_3n_terms (n : ℕ) : S (3 * n) = 63 := 
sorry -- Proof to be provided

end sum_3n_terms_l55_55416


namespace correct_option_B_l55_55940

noncomputable def f : ℝ → ℝ := sorry

def periodic {f : ℝ → ℝ} (p : ℝ) := ∀ x, f (x + p) = f x

def symmetric_about {f : ℝ → ℝ} (a : ℝ) := ∀ x, f (a + x) = f (a - x)

def monotonically_decreasing_on (f : ℝ → ℝ) (s : set ℝ) :=
  ∀ x y ∈ s, x < y → f y < f x

variables (f : ℝ → ℝ)
variable (h_period : periodic f 6)
variable (h_sym : symmetric_about f 3)
variable (h_monotone : monotonically_decreasing_on f (set.Ioo 0 3))

theorem correct_option_B : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
by {
  sorry
}

end correct_option_B_l55_55940


namespace one_fourth_in_one_eighth_l55_55122

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l55_55122


namespace A_union_B_subset_B_A_intersection_B_subset_B_l55_55447

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3 * x - 10 <= 0}
def B (m : ℝ) : Set ℝ := {x | m - 4 <= x ∧ x <= 3 * m + 2}

-- Problem 1: Prove the range of m if A ∪ B = B
theorem A_union_B_subset_B (m : ℝ) : (A ∪ B m = B m) → (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

-- Problem 2: Prove the range of m if A ∩ B = B
theorem A_intersection_B_subset_B (m : ℝ) : (A ∩ B m = B m) → (m < -3) :=
by
  sorry

end A_union_B_subset_B_A_intersection_B_subset_B_l55_55447


namespace tan_A_in_triangleABC_l55_55914

theorem tan_A_in_triangleABC 
  (A B C : ℝ)
  (angle_BAC : Real.Angle)
  (AB BC AC : ℝ)
  (h_angle_BAC : angle_BAC = 60 * Real.pi / 180)
  (h_AB : AB = 20)
  (h_BC : BC = 21)
  : tan angle_BAC = 21 * Real.sqrt 3 / (2 * Real.sqrt (421 - 1323 / 4)) := 
sorry

end tan_A_in_triangleABC_l55_55914


namespace min_photos_needed_to_ensure_conditions_l55_55806

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55806


namespace range_of_t_range_of_a_l55_55418

variables {a t : ℝ}

-- Definition of propositions p and q
def p := ∀ a : ℝ, 0 < a ∧ a ≠ 1 → -2 * t ^ 2 + 7 * t - 5 > 0
def q := t ^ 2 - (a + 3) * t + (a + 2) < 0

-- Part (Ⅰ): Range of t given p
theorem range_of_t (ha : 0 < a ∧ a ≠ 1) (hp : p a) : 1 < t ∧ t < 5 / 2 :=
  sorry

-- Part (Ⅱ): Range of a given p is sufficient but not necessary for q
theorem range_of_a (ha : 0 < a ∧ a ≠ 1)
                   (hp_suff : ∀ t : ℝ, -2 * t ^ 2 + 7 * t - 5 > 0 → t ^ 2 - (a + 3) * t + (a + 2) < 0)
                   (hp : ∃ t : ℝ, -2 * t ^ 2 + 7 * t - 5 > 0 ∧ ¬ (t ^ 2 - (a + 3) * t + (a + 2) < 0)) 
                   : a > 1 / 2 :=
  sorry

end range_of_t_range_of_a_l55_55418


namespace angle_F1PF2_eq_pi_over_3_l55_55207

noncomputable def P : ℝ × ℝ → Prop := fun p => (p.1^2 / 16) + (p.2^2 / 9) = 1
def F1 : ℝ × ℝ := (- √7, 0)
def F2 : ℝ × ℝ := (√7, 0)
def PF1 (p : ℝ × ℝ) : ℝ := real.sqrt ((p.1 + √7)^2 + p.2^2)
def PF2 (p : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - √7)^2 + p.2^2)

theorem angle_F1PF2_eq_pi_over_3 (p : ℝ × ℝ) (h : P p) (hyp : PF1 p * PF2 p = 12) :
  real.arccos ((PF1 p)^2 + (PF2 p)^2 - 28) / (2 * PF1 p * PF2 p) = real.pi / 3 :=
sorry

end angle_F1PF2_eq_pi_over_3_l55_55207


namespace virginia_more_than_adrienne_l55_55272

-- Definitions under the given conditions
def V (D : ℕ) := D - 9
def A (V D : ℕ) := 93 - (V + D)

-- Proof statement
theorem virginia_more_than_adrienne : 
  ∀ (D : ℕ), D = 40 → 
  ∀ (V : ℕ), V = (V D) → 
  ∀ (A : ℕ), A = (A V D) →
  V - A = 9 :=
by 
  intros D hD V hV A hA
  rw [hD, hV, hA]
  sorry

end virginia_more_than_adrienne_l55_55272


namespace problem_statement_l55_55641

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x - Real.sqrt x ≤ y - 1 / 4 ∧ y - 1 / 4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1 / 4 ∧ x - 1 / 4 ≤ y + Real.sqrt y :=
sorry

end problem_statement_l55_55641


namespace range_of_m_l55_55134

theorem range_of_m (m : ℝ) :
  let z := (1 + complex.I * m) / (1 + complex.I) in
  (z.re > 0) ∧ (z.im < 0) → (-1 < m) ∧ (m < 1) := 
by
  -- Introduce the given complex number
  let z := (1 + complex.I * m) / (1 + complex.I)
  
  -- State the conditions
  assume h : (z.re > 0) ∧ (z.im < 0)

  -- The proof will show (-1 < m) ∧ (m < 1)
  sorry

end range_of_m_l55_55134


namespace problem1_solution_l55_55996

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l55_55996


namespace problem_l55_55937

-- Definitions for the problem's conditions:
variables {a b c d : ℝ}

-- a and b are roots of x^2 + 68x + 1 = 0
axiom ha : a ^ 2 + 68 * a + 1 = 0
axiom hb : b ^ 2 + 68 * b + 1 = 0

-- c and d are roots of x^2 - 86x + 1 = 0
axiom hc : c ^ 2 - 86 * c + 1 = 0
axiom hd : d ^ 2 - 86 * d + 1 = 0

theorem problem : (a + c) * (b + c) * (a - d) * (b - d) = 2772 :=
sorry

end problem_l55_55937


namespace series_sum_equals_one_sixth_l55_55359

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l55_55359


namespace cabinet_area_l55_55230

theorem cabinet_area (width length : ℝ) (h_width : width = 1.2) (h_length : length = 1.8) :
  width * length = 2.16 := 
by 
  rw [h_width, h_length]
  norm_num
  -- The proof would be included here, but it is omitted per the instructions.
  sorry

end cabinet_area_l55_55230


namespace expression_evaluation_l55_55985

-- Definitions of the expressions
def expr (x y : ℤ) : ℤ :=
  ((x - 2 * y) ^ 2 + (3 * x - y) * (3 * x + y) - 3 * y ^ 2) / (-2 * x)

-- Proof that the expression evaluates to -11 when x = 1 and y = -3
theorem expression_evaluation : expr 1 (-3) = -11 :=
by
  -- Declarations
  let x := 1
  let y := -3
  -- The core calculation
  show expr x y = -11
  sorry

end expression_evaluation_l55_55985


namespace problem_midterm_l55_55873

open Real
open Complex

variables (α : ℝ)
variables a b : ℝ × ℝ

noncomputable def a := (4, 5 * cos α)
noncomputable def b := (3, -4 * tan α)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

def first_question (u v : ℝ × ℝ): ℝ := sqrt ( (u.1 + v.1)^2 + (u.2 + v.2)^2)

def second_question (x : ℝ) : ℝ := cos (x + π / 4)

theorem problem_midterm : (0 < α ∧ α < π / 2) → is_perpendicular a b → 
  (first_question a b = 5 * sqrt 2 ∧ second_question α = sqrt 2 / 10) := sorry

end problem_midterm_l55_55873


namespace part1_part2_part3_l55_55185

section function_properties
variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ (x y : ℝ), f(x + y) = f(x) + f(y) - 1
axiom cond2 : ∀ (x : ℝ), x > 0 → f(x) < 1
axiom cond3 : f(1) = -2

-- Part (1)
theorem part1 : f(0) = 1 ∧ ∀ x1 x2, x1 < x2 → f(x1) > f(x2) :=
by sorry

-- Part (2)
theorem part2 (x : ℝ) (H : x ∈ [-1, 1]) : 
  (∀ a ∈ [-1, 1], f(x) ≤ m^2 - 2 * a * m - 5) → m ∈ (-∞, -3] ∪ [3, ∞) :=
by sorry

-- Part (3)
theorem part3 (a x : ℝ) : f(a * x^2) < f((a + 2) * x) + 6 ↔ 
(a > 2 → x ∈ (-∞, 2 / a) ∪ (1, ∞)) ∧
(0 < a ∧ a < 2 → x ∈ (-∞, 1) ∪ (2 / a, ∞)) ∧
(a = 0 → x ∈ (-∞, 1)) ∧
(a < 0 → x ∈ (2 / a, 1)) :=
by sorry

end function_properties

end part1_part2_part3_l55_55185


namespace cost_of_traveling_roads_l55_55318

def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 40
def road_width : ℕ := 10
def cost_per_sqm : ℕ := 3

def area_road_parallel_length : ℕ := road_width * lawn_length
def area_road_parallel_breadth : ℕ := road_width * lawn_breadth
def area_intersection : ℕ := road_width * road_width

def total_area_roads : ℕ := area_road_parallel_length + area_road_parallel_breadth - area_intersection
def total_cost : ℕ := total_area_roads * cost_per_sqm

theorem cost_of_traveling_roads : total_cost = 3300 :=
by
  sorry

end cost_of_traveling_roads_l55_55318


namespace vector_magnitude_subtraction_l55_55011

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55011


namespace max_trios_correct_l55_55342

noncomputable def max_trios (n : ℕ) : ℕ :=
  if even n then
    let m := n / 2 in (m - 1) * m
  else
    let m := (n - 1) / 2 in m * m

theorem max_trios_correct (n : ℕ) : ∀ (s : Finset ℝ), s.card = n → 
  (max_trios n = if even n then (let m := n / 2 in (m - 1) * m) else (let m := (n - 1) / 2 in m * m)) :=
by
  intros s h_card
  exact sorry

end max_trios_correct_l55_55342


namespace largest_binomial_term_l55_55377

theorem largest_binomial_term :
  ∃ k : ℕ, k = 45 ∧ ∀ k' : ℕ, (k' ≠ 45 → ∑ i in finset.range (501), binom 500 i * (0.1)^i < (binom 500 45 * (0.1)^45)) :=
by sorry

end largest_binomial_term_l55_55377


namespace vector_magnitude_correct_l55_55039

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55039


namespace zero_point_interval_l55_55237

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 + Real.log x / Real.log 2

theorem zero_point_interval : ∃ x ∈ (1/2 : ℝ) .. 1, f x = 0 :=
by { sorry }

end zero_point_interval_l55_55237


namespace count_divisible_3_4_or_5_l55_55126

theorem count_divisible_3_4_or_5 : 
  (finset.range 61).filter (λ n, n % 3 = 0 ∨ n % 4 = 0 ∨ n % 5 = 0).card = 36 :=
by
  sorry

end count_divisible_3_4_or_5_l55_55126


namespace sum_of_digits_285714_l55_55589

theorem sum_of_digits_285714 :
  let m := 2 * 10^5 + 85714
  in (3 * m) % 10^6 = 10^5 + 85714 * 10 + 2 ∧
     (2 + 8 + 5 + 7 + 1 + 4 = 27) :=
by
  let m := 2 * 10^5 + 85714
  have h : 3 * m = 10^5 + 85714 * 10 + 2 := sorry
  have s : 285714.digits.sum = 27 := sorry
  exact ⟨h, s⟩

end sum_of_digits_285714_l55_55589


namespace problem1_l55_55997

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l55_55997


namespace max_pairs_plane_theorem_max_pairs_space_theorem_l55_55408

noncomputable def max_pairs_plane (n : ℕ) : Prop :=
  ∀ (points : fin n → (ℝ × ℝ)),
    (∀ i j, i ≠ j → dist (points i) (points j) ≥ 1) →
    (card {p : fin n × fin n // dist (points p.1) (points p.2) = 1} ≤ 3 * n)

noncomputable def max_pairs_space (n : ℕ) : Prop :=
  ∀ (points : fin n → (ℝ × ℝ × ℝ)),
    (∀ i j, i ≠ j → dist (points i) (points j) ≥ 1) →
    (card {p : fin n × fin n // dist (points p.1) (points p.2) = 1} ≤ 7 * n)

theorem max_pairs_plane_theorem (n : ℕ) : max_pairs_plane n := sorry

theorem max_pairs_space_theorem (n : ℕ) : max_pairs_space n := sorry

end max_pairs_plane_theorem_max_pairs_space_theorem_l55_55408


namespace heaviest_lightest_difference_l55_55604

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end heaviest_lightest_difference_l55_55604


namespace vector_magnitude_l55_55097

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55097


namespace find_original_rectangle_area_l55_55233

-- Let A be the area of the original rectangle.
def original_rectangle_area_doubled (A : ℝ) : Prop :=
  let new_A := 4 * A in -- The area of the new rectangle when dimensions are doubled.
  new_A = 32

theorem find_original_rectangle_area :
  ∃ A : ℝ, original_rectangle_area_doubled A ∧ A = 8 :=
by
  sorry

end find_original_rectangle_area_l55_55233


namespace log_limit_l55_55884

open Real

theorem log_limit (x : ℝ) (h : ∀ ε > 0, ∃ x₀, x > x₀ → abs (log 3 (6 * x - 5) - log 3 (2 * x + 1) - 1) < ε) :
  ∀ ε > 0, ∃ x₀, ∀ x > x₀, abs (log 3 ((6 * x - 5) / (2 * x + 1)) - 1) < ε :=
by sorry

end log_limit_l55_55884


namespace sum_of_a_and_b_l55_55735

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55735


namespace find_f99_f_prime_99_l55_55520

noncomputable def f (x : ℝ) : ℝ := sqrt (1 + x^2)

theorem find_f99_f_prime_99 :
  let f' := λ x: ℝ, (x / sqrt (1 + x^2)) in
  f 99 * f' 99 = 99 :=
by
  sorry

end find_f99_f_prime_99_l55_55520


namespace vec_magnitude_is_five_l55_55053

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55053


namespace equation_of_circle_min_distance_PA_PB_l55_55109

-- Definition of the given points, lines, and circle
def point (x y : ℝ) : Prop := true

def circle_through_points (x1 y1 x2 y2 x3 y3 : ℝ) (a b r : ℝ) : Prop :=
  (x1 + a) * (x1 + a) + y1 * y1 = r ∧
  (x2 + a) * (x2 + a) + y2 * y2 = r ∧
  (x3 + a) * (x3 + a) + y3 * y3 = r

def line (a b : ℝ) : Prop := true

-- Specific points
def D := point 0 1
def E := point (-2) 1
def F := point (-1) (Real.sqrt 2)

-- Lines l1 and l2
def l₁ (x : ℝ) : ℝ := x - 2
def l₂ (x : ℝ) : ℝ := x + 1

-- Intersection points A and B
def A := point 0 1
def B := point (-2) (-1)

-- Question Ⅰ: Find the equation of the circle
theorem equation_of_circle :
  ∃ a b r, circle_through_points 0 1 (-2) 1 (-1) (Real.sqrt 2) a b r ∧ (a = -1 ∧ b = 0 ∧ r = 2) :=
  sorry

-- Question Ⅱ: Find the minimum value of |PA|^2 + |PB|^2
def dist_sq (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

theorem min_distance_PA_PB :
  real := sorry

end equation_of_circle_min_distance_PA_PB_l55_55109


namespace value_of_b_l55_55182

noncomputable def a : ℤ :=
  1 + (Nat.choose 10 1) + (Nat.choose 10 2) * 2 + (Nat.choose 10 3) * 2^2 + 
  (Nat.choose 10 4) * 2^3 + (Nat.choose 10 5) * 2^4 + (Nat.choose 10 6) * 2^5 + 
  (Nat.choose 10 7) * 2^6 + (Nat.choose 10 8) * 2^7 + (Nat.choose 10 9) * 2^8 + 
  (Nat.choose 10 10) * 2^9

def b : ℤ := 2015

theorem value_of_b (h : b ≡ a [MOD 10]) : b % 10 = 5 :=
sorry

end value_of_b_l55_55182


namespace vector_magnitude_subtraction_l55_55024

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55024


namespace polynomial_factors_count_l55_55775

theorem polynomial_factors_count :
  ∃ (count : ℤ), (count = 64) ∧ (∀ n : ℤ, (1 ≤ n ∧ n ≤ 1000) →
    ∃ (a b : ℤ), (a + b = -2) ∧ (a * b = -n) ↔ (a^2 + 2 * a) = n) :=
by {
  let count : ℤ := 64,
  existsi count,
  split,
  exact rfl,
  intro n,
  intro h,
  split;
  intro hp,
  { rcases hp with ⟨a, b, hab1, hab2⟩,
    exact (show a^2 + 2*a = n, by sorry) },
  { let a := (n + 1) - (-(n + 2)), -- Dummy specific solution
    let b := -2 - a,
    existsi a,
    existsi b,
    split,
    exact (show a + b = -2, by sorry),
    exact (show a * b = -n, by sorry) }
}

end polynomial_factors_count_l55_55775


namespace axis_of_symmetry_l55_55560

theorem axis_of_symmetry :
  ∀ x : ℝ, (g : ℝ → ℝ), (f : ℝ → ℝ) 
  (g = λ x , 3 * Real.sin (2 * x - π / 6))
  (f = λ x , 3 * Real.sin (4 * x + π / 6)),
  (∃ (k : ℤ), x = k * π / 2 + π / 3) :=
begin
  sorry
end

end axis_of_symmetry_l55_55560


namespace cupcakes_frosted_in_5_minutes_l55_55339

theorem cupcakes_frosted_in_5_minutes :
  (let r_cagney := (1 : ℚ) / 20;
       r_lacey := (1 : ℚ) / 30;
       combined_rate := r_cagney + r_lacey in 
       300 * combined_rate = 25) := 
by {
  -- Define Cagney's and Lacey's rates
  let r_cagney := (1 : ℚ) / 20,
  let r_lacey := (1 : ℚ) / 30,

  -- Calculate combined rate
  let combined_rate := r_cagney + r_lacey,

  -- Express the total number of cupcakes frosted in 300 seconds
  have h : 300 * combined_rate = 25, by {
    calc 300 * combined_rate
          = 300 * ((1 / 20) + (1 / 30)) : by { refl }
      ... = 300 * ((3 / 60) + (2 / 60)) : by { congr; field_simp [ne_of_gt (show 20 > 0, by norm_num)] }
      ... = 300 * (5 / 60) : by { congr; field_simp [ne_of_gt (show 30 > 0, by norm_num)] }
      ... = 300 * (1 / 12) : by { norm_num }
      ... = 25 : by norm_num,
  },
  exact h,
}

end cupcakes_frosted_in_5_minutes_l55_55339


namespace simplify_expression_l55_55216

theorem simplify_expression :
  (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 :=
by
  sorry

end simplify_expression_l55_55216


namespace real_solutions_l55_55386

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l55_55386


namespace ink_percentage_left_l55_55614

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end ink_percentage_left_l55_55614


namespace factorize_x2_minus_2x_plus_1_l55_55381

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l55_55381


namespace vector_magnitude_difference_l55_55095

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55095


namespace cevians_ratio_l55_55969

-- Let the triangle ABC and points D, E, F on sides BC, CA, AB respectively
variables {A B C D E F D1 E1 F1 : Type} [OrderedCommRing A]

-- The lines through A, B, C parallel to EF, FD, DE form triangle D1 E1 F1
axiom parallel_lines : (EF FD DE : A) → (D1 E1 F1 : A) → 
    (parallel EF A ↔ intersects A D1 E1 ∧ parallel FD B ↔ intersects B E1 F1 ∧ parallel DE C ↔ intersects C F1 D1)

-- Prove the given ratios
theorem cevians_ratio (ABC DEF DEF: Type) [OrderedCommRing DEF] :
    (D E F : DEF) ∧ (D1 E1 F1 : DEF) →
    (⟨BD, DC⟩ ≃ ⟨F1A, AE1⟩) ∧ (⟨CE, EA⟩ ≃ ⟨D1B, BF1⟩) ∧ (⟨AF, FB⟩ ≃ ⟨E1C, CD1⟩) :=
sorry

end cevians_ratio_l55_55969


namespace total_books_l55_55499

theorem total_books (books_jason : ℕ) (books_mary : ℕ) (h_jason : books_jason = 18) (h_mary : books_mary = 42) : books_jason + books_mary = 60 :=
by
  rw [h_jason, h_mary]
  exact rfl

end total_books_l55_55499


namespace find_original_number_l55_55572

def digitsGPA (A B C : ℕ) : Prop := B^2 = A * C
def digitsAPA (X Y Z : ℕ) : Prop := 2 * Y = X + Z

theorem find_original_number (A B C X Y Z : ℕ) :
  100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C ≤ 999 ∧
  digitsGPA A B C ∧
  100 * X + 10 * Y + Z = (100 * A + 10 * B + C) - 200 ∧
  digitsAPA X Y Z →
  (100 * A + 10 * B + C) = 842 :=
sorry

end find_original_number_l55_55572


namespace g_values_prime_count_correct_l55_55946

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range n.succ).filter (λ d, n % d = 0).sum

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

def g_values_prime_count : ℕ :=
  (List.range 31).filter (λ n, is_prime (sum_of_divisors n)).length

theorem g_values_prime_count_correct : g_values_prime_count = 5 :=
  sorry

end g_values_prime_count_correct_l55_55946


namespace train_passes_man_in_approx_21_seconds_l55_55289

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def man_speed_kmph : ℝ := 6

-- Convert speeds to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def man_speed_mps : ℝ := kmph_to_mps man_speed_kmph

-- Calculate relative speed
noncomputable def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Calculate time
noncomputable def time_to_pass : ℝ := train_length / relative_speed_mps

theorem train_passes_man_in_approx_21_seconds : abs (time_to_pass - 21) < 1 :=
by
  sorry

end train_passes_man_in_approx_21_seconds_l55_55289


namespace race_distance_l55_55332

theorem race_distance 
  (a b c : ℝ) 
  (h1 : b = 0.95 * a) 
  (h2 : c = 0.96 * b) 
  (h3 : a > 0) :
  let time_andrey := 1000 / a in
  let distance_valentin := c * time_andrey in
  1000 - distance_valentin = 88 :=
by
  sorry

end race_distance_l55_55332


namespace carolina_post_office_l55_55750

theorem carolina_post_office :
  ∃ P : ℕ, 0.37 * 5 + 0.88 * P = 4.49 ∧ 5 - P = 2 :=
by
  use 3
  split
  { norm_num }
  { norm_num }

end carolina_post_office_l55_55750


namespace men_in_hotel_l55_55999

theorem men_in_hotel (n : ℕ) (A : ℝ) (h1 : 8 * 3 = 24)
  (h2 : A = 32.625 / n)
  (h3 : 24 + (A + 5) = 32.625) :
  n = 9 := 
  by
  sorry

end men_in_hotel_l55_55999


namespace fourth_rectangle_has_integer_perimeter_l55_55316

theorem fourth_rectangle_has_integer_perimeter
  (a b x y : ℝ)
  (h1 : 2*(x + y) ∈ ℤ)
  (h2 : 2*(x + b - y) ∈ ℤ)
  (h3 : 2*(a - x + y) ∈ ℤ) :
  2*(a - x + b - y) ∈ ℤ := 
sorry

end fourth_rectangle_has_integer_perimeter_l55_55316


namespace vector_magnitude_difference_l55_55022

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55022


namespace number_of_12_digit_numbers_with_consecutive_digits_same_l55_55110

theorem number_of_12_digit_numbers_with_consecutive_digits_same : 
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  total - excluded = 4094 :=
by
  let total := (2 : ℕ) ^ 12
  let excluded := 2
  have h : total = 4096 := by norm_num
  have h' : total - excluded = 4094 := by norm_num
  exact h'

end number_of_12_digit_numbers_with_consecutive_digits_same_l55_55110


namespace cube_container_volume_for_tetrahedron_l55_55693

noncomputable def volume_of_cube (s : ℝ) : ℝ := s^3

def smallest_cube_container_volume (height base_side : ℝ) : ℝ :=
  let s := if height > base_side then height else base_side
  volume_of_cube s

theorem cube_container_volume_for_tetrahedron :
  smallest_cube_container_volume 15 13 = 3375 :=
by
  -- The proof is omitted as per instructions
  sorry

end cube_container_volume_for_tetrahedron_l55_55693


namespace sum_of_first_six_terms_l55_55527

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem sum_of_first_six_terms :
  ∃ S : ℕ, sequence S ∧ S 6 = 63 :=
by
  sorry

end sum_of_first_six_terms_l55_55527


namespace no_player_can_guarantee_win_l55_55245

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

def can_guarantee_win (initial : ℕ) : Prop :=
  ∀ (append : list ℕ → ℕ → ℕ) (turns : ℕ), 
    let play := λ (n : ℕ) (d : ℕ), append n d in
    ¬ ∃ (n : ℕ), is_perfect_square (play initial n) 

theorem no_player_can_guarantee_win : can_guarantee_win 7 :=
by
  intros,
  sorry

end no_player_can_guarantee_win_l55_55245


namespace XiaoMing_selection_l55_55599

def final_positions (n : Nat) : List Nat :=
  if n <= 2 then
    List.range n
  else
    final_positions (n / 2) |>.filter (λ k => k % 2 = 0) |>.map (λ k => k / 2)

theorem XiaoMing_selection (n : Nat) (h : n = 32) : final_positions n = [16, 32] :=
  by
  sorry

end XiaoMing_selection_l55_55599


namespace sum_of_powers_divisible_by_6_l55_55193

theorem sum_of_powers_divisible_by_6 (a1 a2 a3 a4 : ℤ)
  (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) (k : ℕ) (hk : k % 2 = 1) :
  6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
sorry

end sum_of_powers_divisible_by_6_l55_55193


namespace inequality_proof_l55_55976

variable {A B C a b c r : ℝ}

theorem inequality_proof (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) :
  (A + a + B + b) / (A + a + B + b + c + r) + (B + b + C + c) / (B + b + C + c + a + r) > (C + c + A + a) / (C + c + A + a + b + r) := 
    sorry

end inequality_proof_l55_55976


namespace vector_magnitude_subtraction_l55_55001

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55001


namespace vector_magnitude_subtraction_l55_55003

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55003


namespace fraction_female_to_male_fraction_male_to_total_l55_55897

-- Define the number of male and female students
def num_male_students : ℕ := 30
def num_female_students : ℕ := 24
def total_students : ℕ := num_male_students + num_female_students

-- Prove the fraction of female students to male students
theorem fraction_female_to_male :
  (num_female_students : ℚ) / num_male_students = 4 / 5 :=
by sorry

-- Prove the fraction of male students to total students
theorem fraction_male_to_total :
  (num_male_students : ℚ) / total_students = 5 / 9 :=
by sorry

end fraction_female_to_male_fraction_male_to_total_l55_55897


namespace log_base_4_half_l55_55373

theorem log_base_4_half : log 4 (1 / 2) = -1 / 2 := 
sorry

end log_base_4_half_l55_55373


namespace six_digit_number_solution_l55_55840

def six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

def left_shifted_number (a b c d e f : ℕ) : ℕ :=
  100000 * f + 10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem six_digit_number_solution :
  ∀ (a b c d e f : ℕ), 1 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 →
                        0 ≤ c ∧ c ≤ 9 → 0 ≤ d ∧ d ≤ 9 →
                        0 ≤ e ∧ e ≤ 9 → 1 ≤ f ∧ f ≤ 9 →
  left_shifted_number a b c d e f = f * six_digit_number a b c d e f →
  (six_digit_number a b c d e f = 111111 ∨ six_digit_number a b c d e f = 102564) :=
begin
  intros a b c d e f ha hb hc hd he hf h,
  -- Proof will be provided here
  sorry
end

end six_digit_number_solution_l55_55840


namespace geom_proof_l55_55174

noncomputable def midpoint (P Q : Point) : Point :=
  ⟨(P.1 + Q.1) / 2, (P.2 + Q.2) / 2⟩

structure Triangle :=
  (A B C : Point)

structure GeomCondition (A B C D M N P K T : Point) :=
  (altitude_AD : isAltitudeOfTriangle A B C D)
  (midpoint_M : M = midpoint A B)
  (midpoint_N : N = midpoint A D)
  (midpoint_P : P = midpoint B C)
  (foot_K : isFootOfPerpendicular D A C K)
  (T_on_ext_KD : ∃ dt dk mn : ℝ, dt = dist D T ∧ dk = dist D K ∧ mn = dist M N ∧ dist D T = mn + dk)
  (MP_2KN : ∃ mp kn : ℝ, mp = dist M P ∧ kn = dist K N ∧ mp = 2 * kn)

theorem geom_proof (A B C D M N P K T : Point) (hc : GeomCondition A B C D M N P K T) :
  dist A T = dist M C :=
sorry

end geom_proof_l55_55174


namespace map_distance_proof_l55_55537

-- Define the scale of the map
def scale : ℝ := 1 / 250000

-- Define the actual distance between the two points in kilometers
def actual_distance_km : ℝ := 5

-- Define the conversion factor from kilometers to centimeters
def km_to_cm : ℝ := 100000

-- Define the actual distance in centimeters
def actual_distance_cm : ℝ := actual_distance_km * km_to_cm

-- Define the map distance calculation based on the scale
def map_distance_cm : ℝ := actual_distance_cm * scale

-- Prove that the map distance in centimeters is 2 cm
theorem map_distance_proof : map_distance_cm = 2 := by 
  sorry

end map_distance_proof_l55_55537


namespace angle_BMC_measure_l55_55161

theorem angle_BMC_measure (ABC : Triangle) (α : ℝ) 
    (A_eq_alpha : ABC.A = α) 
    (BC_shortest : ABC.BC < min ABC.AB ABC.AC) 
    (P_on_AB : P ∈ Segment ABC.AB) 
    (Q_on_AC : Q ∈ Segment ABC.AC) 
    (PB_EQ_BC : PBC = ABC.BC) 
    (CQ_EQ_BC : CQ = ABC.BC)
    (BQ_CP_intersect : ∃ M, M ∈ Line (Segment BQ ∩ Segment CP)) 
    (PBC_isosceles : IsoscelesTriangle PB BC PB)
    (QBC_isosceles : IsoscelesTriangle BQ CQ BC) : 
    ABC.BMC = 90 - α / 2 := 
sorry

end angle_BMC_measure_l55_55161


namespace greatest_value_sum_eq_24_l55_55727

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55727


namespace proper_divisor_sum_not_divisors_l55_55669

theorem proper_divisor_sum_not_divisors {n : ℕ}
  (h1 : ∃ (d : ℕ → Prop), (∀ d₁ d₂, d d₁ → d d₂ → d₁ < n ∧ 1 < d₁) ∧
    (∀ d, d d → d < n ∧ 1 < d) ∧ (∃ d₁ d₂ d₃, d d₁ ∧ d d₂ ∧ d d₃ ∧ d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃)) :
  ∀ s : set ℕ, (∀ d₁ d₂, d d₁ → d d₂ → d₁ + d₂ ∈ s) →
    ¬ (∃ m, (∀ d, d < m ∧ 1 < d → d ∈ s) ∧ (∀ d ∈ s, d < m ∧ 1 < d)) :=
by
  sorry

end proper_divisor_sum_not_divisors_l55_55669


namespace second_drawn_not_23_l55_55704

/-- 
Given the systematic sampling conditions,
prove that the number 23 cannot possibly be the second drawn invoice stub.
-/
theorem second_drawn_not_23 (first_stub : Fin 10) (drawn_numbers : ℕ → ℕ)
  (h1 : ∀ n, drawn_numbers n = first_stub + n * 10) : ∀ n, drawn_numbers 1 ≠ 23 :=
by
  intro n
  simp only [h1]
  intros H
  sorry

end second_drawn_not_23_l55_55704


namespace sum_reflected_midpoint_coordinates_l55_55543

-- Definition of points A and B
def A : ℝ × ℝ := (3, -2)
def B : ℝ × ℝ := (15, 10)

-- The main theorem: Sum of the coordinates of the reflected midpoint over the y-axis
theorem sum_reflected_midpoint_coordinates : 
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let M' := (-M.1, M.2) in
  M'.1 + M'.2 = -5 :=
by
  sorry

end sum_reflected_midpoint_coordinates_l55_55543


namespace vector_magnitude_correct_l55_55038

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55038


namespace surface_area_of_sphere_l55_55250

-- Definitions of the points and segments lying in specified geometric relationships
variables {Point : Type} [metric_space Point] (A B C D E F : Point)

-- Given conditions: segment lengths and parallelism, and tangency to a sphere
variables (EF_dist BC_dist : ℝ)
variable (touches_sphere : ∀ (P Q : Point), P Q ∈ {A, B, C, D, E, F} → segment PQ touches sphere)
variable (plane : set Point)
variable (plane_contains_rectangle : ∀ (P : Point), P ∈ {A, B, C, D} → P ∈ plane)
variable (EF_parallel_to_plane : ∀ (P Q : Point), P Q ∈ {E, F} → P parallel to plane)
variable (EF : E.distance(F) = EF_dist)
variable (BC : B.distance(C) = BC_dist)

noncomputable def radius_sphere_from_given_data (EF_dist BC_dist : ℝ) : ℝ :=
sorry

-- Statement of the theorem: the surface area of the sphere
theorem surface_area_of_sphere (EF_dist BC_dist : ℝ)
  (h_EF : EF_dist = 3) (h_BC : BC_dist = 5)
  (touches_sphere : ∀ (P Q : Point), P Q ∈ {A, B, C, D, E, F} → segment PQ touches sphere)
  (EF_parallel_to_plane : ∀ (P Q : Point), P Q ∈ {E, F} → P parallel to plane) :
  4 * π * (radius_sphere_from_given_data EF_dist BC_dist)^2 = (180 * π) / 7 := 
sorry

end surface_area_of_sphere_l55_55250


namespace vector_magnitude_l55_55102

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55102


namespace geometric_series_sum_l55_55743

theorem geometric_series_sum :
  let a := (2 : ℚ) / 3
  let r := -(1 / 2 : ℚ)
  let n := 6
  let S := a * ((1 - r^n) / (1 - r))
  S = 7 / 16 :=
by
  let a := (2 : ℚ) / 3
  let r := -(1 / 2 : ℚ)
  let n := 6
  let S := a * ((1 - r^n) / (1 - r))
  have h : S = 7 / 16 := sorry
  exact h

end geometric_series_sum_l55_55743


namespace magnitude_of_a_minus_b_l55_55083

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55083


namespace distinct_triangles_in_octahedron_l55_55114

theorem distinct_triangles_in_octahedron : 
  ∀ (vertices : Finset ℕ), vertices.card = 8 → (Finset.card (Finset.powersetLen 3 vertices) = 56) :=
by
  intros vertices h_vertices
  sorry

end distinct_triangles_in_octahedron_l55_55114


namespace length_of_platform_l55_55287

theorem length_of_platform (t : ℝ) (time_platform : ℝ) (time_pole : ℝ) : t = 300 → time_platform = 51 → time_pole = 18 → 
  let speed := t / time_pole in
  let total_distance := speed * time_platform in
  total_distance - t = 550.17 :=
by
  intros ht htime_platform htime_pole
  rw [ht, htime_platform, htime_pole]
  let speed := (300 : ℝ) / 18
  have hspeed : speed = 16.666666666666668 := by norm_num
  rw [hspeed]
  let total_distance := 16.666666666666668 * 51
  have htotal_distance : total_distance = 850.17 := by norm_num
  rw [htotal_distance]
  norm_num
  sorry         -- Proof steps go here

end length_of_platform_l55_55287


namespace first_term_of_geometric_sequence_l55_55577

theorem first_term_of_geometric_sequence
  (a r : ℚ) -- where a is the first term and r is the common ratio
  (h1 : a * r^4 = 45) -- fifth term condition
  (h2 : a * r^5 = 60) -- sixth term condition
  : a = 1215 / 256 := 
sorry

end first_term_of_geometric_sequence_l55_55577


namespace find_y_l55_55883

theorem find_y (x y : ℝ) (h1 : 9823 + x = 13200) (h2 : x = y / 3 + 37.5) : y = 10018.5 :=
by
  sorry

end find_y_l55_55883


namespace area_difference_l55_55260

variable (x y : ℕ)

theorem area_difference (h1 : x > 0) (h2 : y > 0) :
  let A_a := x * y
  let A_b := (x - 1) * (y - 1)
  A_a - A_b = x + y - 1 :=
by
  let A_a := x * y
  let A_b := (x - 1) * (y - 1)
  calc
    A_a - A_b = x * y - (x - 1) * (y - 1) : by rfl
           ... = x * y - (xy - x - y + 1) : by sorry
           ... = x + y - 1               : by sorry

end area_difference_l55_55260


namespace intersection_on_BC_l55_55414

variables {A B C T P Q R S : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space T]
variables {AB AC TB TC PR QS BC : set (Type*)}

-- Defining the perpendicularity conditions
def perp_AB_PT : Prop := ⟪P, T⟫ ⊥ AB
def perp_AC_QT : Prop := ⟪Q, T⟫ ⊥ AC
def perp_TC_AR : Prop := ⟪A, R⟫ ⊥ TC
def perp_TB_AS : Prop := ⟪A, S⟫ ⊥ TB

-- Defining the lines and intersection conditions
def line_PR : set (Type*) := line P R
def line_QS : set (Type*) := line Q S
def line_BC : set (Type*) := line B C
def intersection_X : Type* := ∃ X, X ∈ (line P R) ∧ X ∈ (line Q S)

theorem intersection_on_BC 
  (triangle_ABC : triangle A B C)
  (point_T : point T)
  (foot_P : foot P AB T)
  (foot_Q : foot Q AC T)
  (foot_R : foot R A TC)
  (foot_S : foot S A TB)
  (PR_intersect_QS : intersection_X)
  : ∃ X, X ∈ (line BC) :=
sorry

end intersection_on_BC_l55_55414


namespace minimum_fourth_day_rain_l55_55608

def rainstorm_duration : Nat := 4
def area_capacity_feet : Nat := 6
def area_capacity_inches : Nat := area_capacity_feet * 12 -- Convert to inches
def drainage_rate : Nat := 3 -- inches per day
def rainfall_day1 : Nat := 10
def rainfall_day2 : Nat := 2 * rainfall_day1
def rainfall_day3 : Nat := (3 * rainfall_day1) -- 50% more than Day 2
def total_rain_first_three_days : Nat := rainfall_day1 + rainfall_day2 + rainfall_day3
def drained_amount : Nat := 3 * drainage_rate
def effective_capacity : Nat := area_capacity_inches - drained_amount
def overflow_capacity_left : Nat := effective_capacity - total_rain_first_three_days

theorem minimum_fourth_day_rain : Nat :=
  overflow_capacity_left + 1 = 4

end minimum_fourth_day_rain_l55_55608


namespace principal_sum_l55_55395

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r/n)^(n*t) - P

def find_principal (P: ℝ) (r1 r2 r3: ℝ) (SI: ℝ) :=
  SI = P * (r1 + r2 + r3)

theorem principal_sum (CI: ℝ) (r1 r2 r3: ℝ) :
  let SI := CI / 2 in
  find_principal 1436.7083 r1 r2 r3 SI :=
by
  let r1 := 0.08
  let r2 := 0.12
  let r3 := 0.10
  let CI := compound_interest 4000 0.10 2 2
  sorry

end principal_sum_l55_55395


namespace unique_solution_l55_55870

def vector_m (x : ℝ) : ℝ × ℝ := (-real.sin x, real.sin (2 * x))
def vector_n (x : ℝ) : ℝ × ℝ := (real.sin (3 * x), real.sin (4 * x))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2)

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * (real.cos x - real.cos (7 * x)) - (1 / 2) * (real.cos (2 * x) - real.cos (6 * x))

theorem unique_solution (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x ∧ x < real.pi ∧ dot_product (vector_m x) (vector_n x) = a) ↔ a = 1 :=
sorry

end unique_solution_l55_55870


namespace balloons_given_correct_l55_55638

variable (initial_balloons : ℝ) (final_balloons : ℝ) (balloons_given : ℝ)

theorem balloons_given_correct :
  initial_balloons = 7.0 → final_balloons = 12 → balloons_given = final_balloons - initial_balloons → balloons_given = 5.0 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end balloons_given_correct_l55_55638


namespace equilateral_triangle_segments_sum_l55_55265

theorem equilateral_triangle_segments_sum 
  (A B C D E F G H I : Point)
  (x y z : ℝ)
  (h_equilateral : ∀ d e f, dist d e = dist e f ∧ dist e f = dist f d)
  (h_AB2 : dist A B = 2)
  (h_DE_parallel_BC : parallel D E C B)
  (h_FG_parallel_BC : parallel F G C B)
  (h_HI_parallel_BC : parallel H I C B)
  (h_P1_perimeter : perimeter (D, E, F, G) = 6)
  (h_P2_perimeter : perimeter (F, G, H, I) = 6)
  (h_P3_perimeter : perimeter (H, I, A, B) = 6)
  (h_AD_x : dist A D = x)
  (h_DF_y : dist D F = y)
  (h_AG_z : dist A G = z) :
  dist D E + dist F G + dist H I = 3 := 
sorry

end equilateral_triangle_segments_sum_l55_55265


namespace complex_calculation_l55_55300

theorem complex_calculation :
  (1 - complex.I + (2 + real.sqrt 5 * complex.I)) / complex.I = real.sqrt 5 - 1 - 3 * complex.I := 
sorry

end complex_calculation_l55_55300


namespace intercept_form_conversion_normal_form_conversion_l55_55444

-- Definitions for given conditions.
def plane_eq (x y z : ℝ) : Prop :=
  2 * x - 2 * y + z - 20 = 0

def intercept_form (x y z : ℝ) : Prop :=
  (x / 10) + (y / -10) + (z / 20) = 1

def normal_form (x y z : ℝ) : Prop :=
  -(2 / 3) * x + (2 / 3) * y - (1 / 3) * z + (20 / 3) = 0

-- Theorem statements to prove the conversions.
theorem intercept_form_conversion (x y z : ℝ) :
  plane_eq x y z → intercept_form x y z :=
by
  sorry

theorem normal_form_conversion (x y z : ℝ) :
  plane_eq x y z → normal_form x y z :=
by
  sorry

end intercept_form_conversion_normal_form_conversion_l55_55444


namespace sum_of_a_and_b_l55_55732

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55732


namespace vector_magnitude_subtraction_l55_55060

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55060


namespace max_power_sum_l55_55714

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55714


namespace find_length_EF_l55_55905

-- Assume definitions based on problem conditions
def isosceles_triangle (D E F : Type) (DE DF : ℝ) (h1 : DE = 5) (h2 : DF = 5) := True

def altitude_from_D (D E F M : Type) (h : altitude D E F M) := True

def segment_ratio (M F : Type) (EM MF : ℝ) (h : EM = 4 * MF) := True

-- Main theorem statement
theorem find_length_EF (D E F M : Type) 
  (h_iso : isosceles_triangle D E F 5 5)
  (h_alt : altitude_from_D D E F M)
  (h_seg : segment_ratio M F EM MF)
  (h_EM : EM = 4 * MF) : 
  EF = (5 * (Real.sqrt 10)) / 4 := 
  sorry

end find_length_EF_l55_55905


namespace correct_operation_l55_55632

theorem correct_operation :
  ∃ (a : ℝ), (4 * a^2 * a^3 = 4 * a^5) ∧
             (¬ (a^2 + a^2 = 2 * a^4)) ∧
             (¬ ((-3 * a^2)^3 = -9 * a^6)) ∧
             (¬ (a^6 / a^2 = a^3)) :=
by
  let a : ℝ := 1
  existsi a
  split
  { -- Proving option C
    calc  
      4 * a^2 * a^3 = 4 * (a^2 * a^3) : by ring
                ...  = 4 * a^(2+3)    : by rw pow_add
                ...  = 4 * a^5        : by ring },
  split
  { -- Proving option A is incorrect
    intro h,
    have : 2 * a^2 ≠ 2 * a^4,
    calc
      2 * a^2 = 2 * a^(1+1) : by rw pow_add
           ...  ≠ 2 * a^(3+1) : by rw not_pow_eq_pow 1 2
           ...  = 2 * a^4 : by rw pow_add,
    contradiction },
  split
  { -- Proving option B is incorrect
    intro h,
    have : -27 * a^6 ≠ -9 * a^6,
    calc
      (-3)^3 * a^6 = -27 * a^6 : by rw neg_pow
               ... = -27 * a^6 : by ring
               ... ≠ -9 * a^6 : by ring,
    contradiction },
  { -- Proving option D is incorrect
    intro h,
    have : a^4 ≠ a^3,
    calc
      a^(6-2) = a^4 : by rw pow_sub
            ... ≠ a^3 : by rw not_pow_eq_pow 4 3,
    contradiction }

-- Placeholder since proof obligations have been filled.
sorry

end correct_operation_l55_55632


namespace area_DEF_l55_55351

-- Define the conditions
def DE : Real := 12 -- the base in cm
def height : Real := 15 -- the height in cm

-- Define the area calculation
def area_of_triangle (base height : Real) : Real := (1 / 2) * base * height

-- State the theorem
theorem area_DEF : area_of_triangle DE height = 90 :=
by
  sorry

end area_DEF_l55_55351


namespace intersection_point_l55_55388

open Function

def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-3, 2 - 3 * t, -5 + 11 * t)

def plane (x y z : ℝ) : Prop :=
  5 * x + 7 * y + 9 * z - 32 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (line t).1 (line t).2 (line t).3 ∧ line t = (-3, -1, 6) :=
by
  sorry

end intersection_point_l55_55388


namespace sequence_positions_l55_55156

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end sequence_positions_l55_55156


namespace minimum_percentage_increase_l55_55626

theorem minimum_percentage_increase : 
  let S := {-6, -4, -1, 0, 2, 6, 9} in
  let three_smallest_primes := {2, 3, 5} in
  let original_mean : ℝ := (-(6:ℝ) + -4 + -1 + 0 + 2 + 6 + 9) / 7 in
  let new_set := {2, 3, 5, 0, 2, 6, 9} in
  let new_mean : ℝ := (2 + 3 + 5 + 0 + 2 + 6 + 9) / 7 in
  let percentage_increase : ℝ := ((new_mean - original_mean) / original_mean) * 100 in
  percentage_increase = 350 :=
by 
  sorry

end minimum_percentage_increase_l55_55626


namespace evaluate_expression_l55_55369

theorem evaluate_expression :
  (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 :=
by
  sorry

end evaluate_expression_l55_55369


namespace percentage_decaf_after_purchase_l55_55671

-- Define initial conditions
def initial_stock : ℕ := 500
def proportion_A_initial : ℚ := 0.4
def proportion_B_initial : ℚ := 0.35
def proportion_C_initial : ℚ := 0.25
def decaf_A : ℚ := 0.1
def decaf_B : ℚ := 0.3
def decaf_C : ℚ := 0.5

-- Define new purchase conditions
def new_stock : ℕ := 150
def weight_A_new : ℕ := 50
def weight_B_new : ℕ := 60
def weight_D_new : ℕ := 40
def decaf_D : ℚ := 0.7

-- The final proof goal
theorem percentage_decaf_after_purchase :
  (186 : ℚ) / (650 : ℚ) * 100 ≈ 28.62 :=
by {
  -- Proof goes here
  sorry
}

end percentage_decaf_after_purchase_l55_55671


namespace necessary_but_not_sufficient_conditions_for_ellipse_l55_55847

theorem necessary_but_not_sufficient_conditions_for_ellipse (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  ¬(∀ x y : ℝ, ax^2 + by^2 = 1 → (a ≠ b) → False) ∧
  (∀ x y : ℝ, ax^2 + by^2 = 1 → (a > 0) ∧ (b > 0)) :=
by
  sorry

end necessary_but_not_sufficient_conditions_for_ellipse_l55_55847


namespace distribution_schemes_l55_55467

theorem distribution_schemes :
  ∃ (M C S : ℕ), M = 3 ∧ C = 6 ∧ S = 3 ∧ 
  (∃ f : Fin M → Fin S, Function.Bijective f) ∧ 
  (∃ g : (Fin M → Fin S) ∧ (Fin C → Fin (2 * S)), 
     (∃ s1 s2 s3 : Fin C, (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s1 ∧
     Function.Bijective (g 0, g 1) ∧ Function.Bijective (g 2, g 3) ∧
     Function.Bijective (g 4, g 5)))) ∧
  540 = 6 * 15 * 6 :=
by
  sorry

end distribution_schemes_l55_55467


namespace diagonals_intersect_at_midpoint_l55_55974

structure Point where
  x : ℝ
  y : ℝ

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2 }

-- Define the opposite vertices of the parallelogram
def A : Point := { x := 2, y := -3 }
def C : Point := { x := 14, y := 9 }

-- Theorem statement: The diagonals of the parallelogram intersect at point (8, 3)
theorem diagonals_intersect_at_midpoint :
  midpoint A C = { x := 8, y := 3 } :=
by
  sorry

end diagonals_intersect_at_midpoint_l55_55974


namespace rectangle_length_width_difference_l55_55696

theorem rectangle_length_width_difference :
  ∃ (length width : ℕ), (length * width = 864) ∧ (length + width = 60) ∧ (length - width = 12) :=
by
  sorry

end rectangle_length_width_difference_l55_55696


namespace twelve_factorial_mod_thirteen_l55_55400

theorem twelve_factorial_mod_thirteen : (12! % 13) = 12 := by
  sorry

end twelve_factorial_mod_thirteen_l55_55400


namespace vector_magnitude_difference_l55_55092

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55092


namespace increasing_on_interval_l55_55761

noncomputable def f (x : ℝ) : ℝ := 1 + x - Real.sin x

theorem increasing_on_interval : ∀ x y : ℝ, (0 < x ∧ x < 2 * Real.pi) → (0 < y ∧ y < 2 * Real.pi) → x ≤ y → f x ≤ f y :=
by
  intros x y hx hy hxy
  have h_deriv : ∀ z : ℝ, f' z = 1 - Real.cos z := sorry
  have h_nonneg : ∀ z : ℝ, (0 < z ∧ z < 2 * Real.pi) → f' z ≥ 0 :=
    by
      intros z hz
      sorry
  sorry

end increasing_on_interval_l55_55761


namespace zero_point_interval_l55_55848

noncomputable def f : ℝ → ℝ := sorry -- f is defined on (0, +∞) and is monotonically increasing
noncomputable def f' (x : ℝ) := derivative (f x) /-- Derivative of f
  by noncomputable -- ensuring symbolic differentiation where necessary

theorem zero_point_interval :
  (∀ x : ℝ, 0 < x → f (f x - log x / log 2) = 3) ∧ (∀ a b : ℝ, 0 < a ∧ a < b → f a ≤ f b) →
  ∃ y : ℝ, 1 < y ∧ y < 2 ∧ (f y - f' y - 2 = 0) :=
begin
  intros h,
  have hf_mono : ∀ a b : ℝ, 0 < a ∧ a < b → f a ≤ f b, from h.2,
  have hf_eq : ∀ x : ℝ, 0 < x → f (f x - log x / log 2) = 3, from h.1,
  sorry  -- Proof is omitted
end

end zero_point_interval_l55_55848


namespace initial_volume_l55_55901

noncomputable theory

def initial_mixture_volume (x : ℕ) : ℕ :=
  4 * x + x

theorem initial_volume (x : ℕ) (h1: 4 * x ≠ 0) (h2 : 3 * (x + 3) = 4 * x) :
  initial_mixture_volume x = 45 :=
by
  have h : x = 9,
  { sorry },
  rw h,
  refl

end initial_volume_l55_55901


namespace five_points_concyclic_extension_construction_l55_55913

-- Define the general position condition
def general_position (lines : list (ℝ × ℝ → ℝ)) : Prop :=
  (∀ (i j : ℕ), i ≠ j → ¬ parallel (lines[i]) (lines[j])) ∧
  (∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬ collinear (lines[i]) (lines[j]) (lines[k]))

-- Define the point associated with intersections of line sets
def intersection_point (lines : list (ℝ × ℝ → ℝ)) (i j : ℕ) : ℂ := sorry

-- Define the circle through three intersection points
def circle_through_points (points : list ℂ) : set ℂ := sorry

-- Problem (a): Prove five points lie on one circle for n = 5
theorem five_points_concyclic (lines : list (ℝ × ℝ → ℝ)) (h : general_position lines) 
  (H5 : lines.length = 5) : 
  let points := [intersection_point lines 1 2, intersection_point lines 2 3, 
                 intersection_point lines 3 4, intersection_point lines 4 5, 
                 intersection_point lines 5 1] in 
  ∃ (C : set ℂ), (∀ (p ∈ points), p ∈ C) :=
sorry

-- Problem (b): Prove the extension for any n ∈ ℕ
theorem extension_construction (lines : list (ℝ × ℝ → ℝ)) (h : general_position lines) 
  (n : ℕ) :
  let circles (lines : list (ℝ × ℝ → ℝ)) := 
    list.map (λ i, circle_through_points
      ((list.finRange (lines.length)).filter (≠ i).map (intersection_point lines)))
  in 
  if even n then 
    ∃ (P : ℂ), ∀ (C ∈ circles lines, P ∈ C)
  else 
    let points := list.map (λ i, intersection_point lines i (i+1)) (list.finRange n)
    in ∃ (C : set ℂ), (∀ p ∈ points, p ∈ C) :=
sorry

end five_points_concyclic_extension_construction_l55_55913


namespace max_power_sum_l55_55713

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55713


namespace centroid_altitude_length_l55_55915

theorem centroid_altitude_length (A B C G P : Point)
  (hAB : dist A B = 7)
  (hAC : dist A C = 15)
  (hBC : dist B C = 20)
  (hCentroid : is_centroid G A B C)
  (hFoot : is_foot P G B C) :
  dist G P = 1.4 :=
sorry

end centroid_altitude_length_l55_55915


namespace perimeter_of_CMN_is_given_option_l55_55155

-- Let AB = 2, BC = 1, and we have an equilateral triangle CMN with M on AB
structure Rectangle (A B C D : Type) :=
(length width : ℝ)
(h_length : length = 2)
(h_width : width = 1)
(area : ℝ)
(h_area : area = 2)

structure EquilateralTriangle (A C M N : Type) :=
(side_length : ℝ)
(perimeter : ℝ)
(h_perimeter : perimeter = 3 * side_length)

-- Define CMN such that M is on AB
axiom configuration 
  (A B C D M N : Type) 
  (rect : Rectangle A B C D)
  (eq_triangle : EquilateralTriangle C M N) :
  ∃ x : ℝ, x ∈ {3 * sqrt 6, 6, sqrt 12, 3 * sqrt 2, 9} ∧ eq_triangle.perimeter = x

-- Formal proof goal
theorem perimeter_of_CMN_is_given_option 
  (A B C D M N : Type) 
  (rect : Rectangle A B C D)
  (eq_triangle : EquilateralTriangle C M N) :
  ∃ x ∈ {3 * sqrt 6, 6, sqrt 12, 3 * sqrt 2, 9},
  eq_triangle.perimeter = x :=
by
  apply configuration
  sorry

end perimeter_of_CMN_is_given_option_l55_55155


namespace Elon_has_10_more_Teslas_than_Sam_l55_55767

noncomputable def TeslasCalculation : Nat :=
let Chris : Nat := 6
let Sam : Nat := Chris / 2
let Elon : Nat := 13
Elon - Sam

theorem Elon_has_10_more_Teslas_than_Sam :
  TeslasCalculation = 10 :=
by
  sorry

end Elon_has_10_more_Teslas_than_Sam_l55_55767


namespace sum_of_integers_is_24_l55_55532

theorem sum_of_integers_is_24 (x y : ℕ) (hx : x > y) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 :=
by
  sorry

end sum_of_integers_is_24_l55_55532


namespace vector_magnitude_difference_l55_55087

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55087


namespace ink_percentage_left_l55_55615

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end ink_percentage_left_l55_55615


namespace sum_when_max_power_less_500_l55_55717

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55717


namespace cube_polygon_area_l55_55754

theorem cube_polygon_area (cube_side : ℝ) 
  (A B C D : ℝ × ℝ × ℝ)
  (P Q R : ℝ × ℝ × ℝ)
  (hP : P = (10, 0, 0))
  (hQ : Q = (30, 0, 20))
  (hR : R = (30, 5, 30))
  (hA : A = (0, 0, 0))
  (hB : B = (30, 0, 0))
  (hC : C = (30, 0, 30))
  (hD : D = (30, 30, 30))
  (cube_length : cube_side = 30) :
  ∃ area, area = 450 := 
sorry

end cube_polygon_area_l55_55754


namespace length_of_AC_l55_55162

noncomputable theory
open Real  -- allowing use of real number operations, square roots, etc.

theorem length_of_AC
  (BC : ℝ)
  (angle_B : ℝ)
  (area_S : ℝ)
  (h : ℝ := 2 * sqrt 3)  -- Derived from area condition
  (AB : ℝ := h)          -- Since AB equals height h
  (cos_B : ℝ := cos (2 * π / 3)) -- Calculated cosine of angle B
  : BC = 1 → angle_B = 2 * π / 3 → area_S = sqrt 3 → AC = sqrt 19 := by
       -- Placeholder proof, actual proof required
       sorry

end length_of_AC_l55_55162


namespace physics_marks_l55_55643

theorem physics_marks
  (P C M : ℕ)
  (h1 : P + C + M = 240)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 80 :=
by
  sorry

end physics_marks_l55_55643


namespace complex_number_quadrant_l55_55569

noncomputable def z : ℂ := (2 + Complex.i) * Complex.i

theorem complex_number_quadrant :
    (-1 : ℂ).im = 2 → (-1 : ℂ).re = -1 → z = -1 + 2 * Complex.i → (-1, 2).x < 0 ∧ (-1, 2).y > 0 := 
by
  intro h_im h_re h_z
  sorry

end complex_number_quadrant_l55_55569


namespace vector_magnitude_subtraction_l55_55065

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55065


namespace magnitude_of_a_minus_b_l55_55082

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55082


namespace heaviest_box_difference_l55_55602

theorem heaviest_box_difference (a b c d : ℕ) (h : a < b) (h1 : b < c) (h2 : c < d)
  (pairs : multiset ℕ) (weights : multiset ℕ)
  (hpairs : pairs = [a + b, a + c, a + d, b + c, b + d, c + d])
  (hweights : weights = [22, 23, 27, 29, 30]) :
  (d - a) = 7 :=
by {
  sorry
}

end heaviest_box_difference_l55_55602


namespace meeting_occurs_probability_l55_55310

noncomputable def probability_meeting_occurs : ℝ := sorry

theorem meeting_occurs_probability :
  let x y z w : ℝ := sorry in
  0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ z ∧ z ≤ 2 ∧ 0 ≤ w ∧ w ≤ 2 ∧ (w > x ∧ w > y ∧ w > z) ∧
  (x ≤ y + 0.5 ∧ y ≤ x + 0.5 ∧ x ≤ z + 0.5 ∧ z ≤ x + 0.5 ∧ y ≤ z + 0.5 ∧ z ≤ y + 0.5) →
  probability_meeting_occurs = 0.119791667 := sorry

end meeting_occurs_probability_l55_55310


namespace number_of_seats_in_nth_row_l55_55143

theorem number_of_seats_in_nth_row (n : ℕ) :
    ∃ m : ℕ, m = 3 * n + 15 :=
by
  sorry

end number_of_seats_in_nth_row_l55_55143


namespace jack_walked_distance_l55_55496

theorem jack_walked_distance (time_in_hours : ℝ) (rate : ℝ) (expected_distance : ℝ) : 
  time_in_hours = 1 + 15 / 60 ∧ 
  rate = 6.4 →
  expected_distance = 8 → 
  rate * time_in_hours = expected_distance :=
by 
  intros h
  sorry

end jack_walked_distance_l55_55496


namespace vector_magnitude_subtraction_l55_55028

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55028


namespace arithmetic_sequence_1001th_term_l55_55578

theorem arithmetic_sequence_1001th_term (p q : ℚ)
    (h1 : p + 3 * q = 12)
    (h2 : 12 + 3 * q = 3 * p - q) :
    (p + (1001 - 1) * (3 * q) = 5545) :=
by
  sorry

end arithmetic_sequence_1001th_term_l55_55578


namespace constant_term_in_expansion_l55_55486

theorem constant_term_in_expansion : 
  (∃ T : ℤ, is_constant_term (3 * x - 1 / (sqrt x)) 6 T ∧ T = 135) :=
sorry

end constant_term_in_expansion_l55_55486


namespace exists_f_simple_concave_to_simple_convex_not_exists_f_simple_convex_to_simple_concave_l55_55492

variable (P : Type) [metric_space P] -- Assuming P is a metric space representing the plane
variables (v : ℕ → P) (n : ℕ) (f : P → P)

def is_simple_polygon (v : ℕ → P) (n : ℕ) : Prop := sorry -- Predicate for simple polygon
def is_convex_polygon (v : ℕ → P) (n : ℕ) : Prop := sorry -- Predicate for convex polygon
def is_concave_polygon (v : ℕ → P) (n : ℕ) : Prop := ¬ is_convex_polygon v n

-- Part (a)
theorem exists_f_simple_concave_to_simple_convex :
  (∃ (f : P → P), ∀ {n : ℕ} (h : n ≥ 4), 
    (is_simple_polygon (v) (n) ∧ is_concave_polygon (v) (n)) → 
    (is_simple_polygon (λ x, f (v x)) 3 ∧ is_convex_polygon (λ x, f (v x)) 3)) := 
  sorry

-- Part (b)
theorem not_exists_f_simple_convex_to_simple_concave :
  ¬ ∃ (f : P → P), ∀ {n : ℕ} (h : n ≥ 4),
    (is_simple_polygon (v) n ∧ is_convex_polygon (v) n) →
    (is_simple_polygon (λ x, f (v x)) 3 ∧ is_concave_polygon (λ x, f (v x)) 3) :=
  sorry

end exists_f_simple_concave_to_simple_convex_not_exists_f_simple_convex_to_simple_concave_l55_55492


namespace ink_left_is_50_percent_l55_55616

variables (A1 A2 : ℕ)
variables (length width : ℕ)
variables (total_area used_area : ℕ)

-- Define the conditions
def total_area_of_squares := 3 * (4 * 4)
def total_area_of_rectangles := 2 * (6 * 2)
def ink_left_percentage := ((total_area_of_squares - total_area_of_rectangles) * 100) / total_area_of_squares

-- The theorem to prove
theorem ink_left_is_50_percent : ink_left_percentage = 50 :=
by
  rw [total_area_of_squares, total_area_of_rectangles]
  norm_num
  exact rfl
  sorry -- Proof omitted

end ink_left_is_50_percent_l55_55616


namespace value_of_expression_l55_55276

theorem value_of_expression : (2207 - 2024)^2 * 4 / 144 = 930.25 := 
by
  sorry

end value_of_expression_l55_55276


namespace min_photos_for_condition_l55_55811

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55811


namespace divide_trout_evenly_l55_55955

theorem divide_trout_evenly (total_trout number_of_people : ℕ) (h1 : total_trout = 52) (h2 : number_of_people = 4) : total_trout / number_of_people = 13 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num) (by norm_num)
  sorry

end divide_trout_evenly_l55_55955


namespace tetrahedron_volume_correct_l55_55484

noncomputable def volume_of_tetrahedron (AB AC AD B C D : ℝ) (area_ABC area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  let h_ABC := (2 * area_ABC) / AB
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABC_ABD
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct :
  volume_of_tetrahedron 4 4 4 20 20 16 (Real.pi / 4) = (80 * Real.sqrt 2) / 3 := by
  sorry

end tetrahedron_volume_correct_l55_55484


namespace sum_of_real_solutions_of_series_eq_zero_l55_55778

theorem sum_of_real_solutions_of_series_eq_zero :
  let series (x : ℝ) := 1 + x - x^2 + x^3 - x^4 + x^5 - ∑' i, (-x) ^ i
  (1 : ℝ ∉ ℝ distribution): 
  ∀ x : ℝ, series x = x → x = 0 := arbitrary sorry 

end sum_of_real_solutions_of_series_eq_zero_l55_55778


namespace basket_probability_l55_55497

theorem basket_probability :
  let p_jack := 1 / 6
  let p_jill := 1 / 7
  let p_sandy := 1 / 8
  (1 - p_jack) * p_jill * p_sandy = 5 / 336 :=
by
  let p_jack := 1 / 6
  let p_jill := 1 / 7
  let p_sandy := 1 / 8
  calc
    (1 - p_jack) * p_jill * p_sandy = (5 / 6) * (1 / 7) * (1 / 8) : by sorry
    ... = 5 / 336 : by sorry

end basket_probability_l55_55497


namespace trig_identity_l55_55547

theorem trig_identity :
  (sin 160 + sin 40) * (sin 140 + sin 20) + (sin 50 - sin 70) * (sin 130 - sin 110) = 1 :=
begin
  sorry
end

end trig_identity_l55_55547


namespace sum_of_roots_in_interval_approx_eq_l55_55220

-- Define the equation
def trig_equation (x : ℝ) : Prop :=
  cos (2 * x) + cos (6 * x) + 2 * sin (x) ^ 2 = 1

-- Define the interval
def valid_interval (x : ℝ) : Prop :=
  x ∈ set.Icc (5 * Real.pi / 6) Real.pi

-- Define the sum of roots within the interval belonging to set A
def sum_of_roots_in_interval : ℝ :=
  ∑ x in (set.filter valid_interval {x | trig_equation x}), x

-- The statement to be proven
theorem sum_of_roots_in_interval_approx_eq : 
  |sum_of_roots_in_interval - 2.88| < 0.01 :=
by sorry

end sum_of_roots_in_interval_approx_eq_l55_55220


namespace sum_of_roots_in_interval_approx_eq_l55_55221

-- Define the equation
def trig_equation (x : ℝ) : Prop :=
  cos (2 * x) + cos (6 * x) + 2 * sin (x) ^ 2 = 1

-- Define the interval
def valid_interval (x : ℝ) : Prop :=
  x ∈ set.Icc (5 * Real.pi / 6) Real.pi

-- Define the sum of roots within the interval belonging to set A
def sum_of_roots_in_interval : ℝ :=
  ∑ x in (set.filter valid_interval {x | trig_equation x}), x

-- The statement to be proven
theorem sum_of_roots_in_interval_approx_eq : 
  |sum_of_roots_in_interval - 2.88| < 0.01 :=
by sorry

end sum_of_roots_in_interval_approx_eq_l55_55221


namespace math_problem_solution_l55_55488

noncomputable def circle_C_eq (α : ℝ) (r : ℝ) (h : r > 0) : Prop :=
  ∀ (x y : ℝ), x = r * Math.cos α ∧ y = r * Math.sin α → x^2 + y^2 = r^2

def line_l_eq (ρ θ : ℝ) : Prop :=
  (sqrt 2) * ρ * Math.cos (θ + Real.pi / 4) = 4

noncomputable def tangent_r_value (r : ℝ) : Prop :=
  r = 2 * sqrt 2

noncomputable def min_length_PQ (r : ℝ) : Prop :=
  2 * sqrt 2 - r = sqrt 2

theorem math_problem_solution : 
  (∀ (α : ℝ) (r : ℝ), r > 0 → circle_C_eq α r r > 0) ∧
  (∀ (ρ θ : ℝ), line_l_eq ρ θ) ∧
  (tangent_r_value (2 * sqrt 2)) ∧
  (min_length_PQ (sqrt 2)) :=
begin
  sorry
end

end math_problem_solution_l55_55488


namespace octagon_cannot_tile_l55_55694

def interior_angle (n : ℕ) : ℝ :=
  180 - 360 / n

def can_tile (n : ℕ) : Prop :=
  360 / interior_angle n = (360 / interior_angle n).floor

theorem octagon_cannot_tile :
  (∀ n ∈ {3, 4, 6}, can_tile n) ∧ ¬ can_tile 8 :=
by
  sorry

end octagon_cannot_tile_l55_55694


namespace min_photos_exists_l55_55797

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55797


namespace level_raised_l55_55642

def tank_length : ℝ := 5
def tank_width : ℝ := 4.5
def tank_height : ℝ := 2.1

def field_length : ℝ := 13.5
def field_width : ℝ := 2.5

def tank_volume : ℝ := tank_length * tank_width * tank_height
def field_area : ℝ := field_length * field_width
def tank_area : ℝ := tank_length * tank_width
def remaining_field_area : ℝ := field_area - tank_area
def increase_in_level : ℝ := tank_volume / remaining_field_area

theorem level_raised :
  increase_in_level = 4.2 :=
sorry

end level_raised_l55_55642


namespace hyperbola_asymptote_l55_55700

-- Given conditions for the hyperbola
variable (a b : ℝ) (h_a_positive : 0 < a) (h_b_positive : 0 < b)
variable (h_hyperbola : ∀ x y : ℝ, ((x ^ 2) / (a ^ 2)) - ((y ^ 2) / (b ^ 2)) = 1)
variable (h_imaginary_axis : 2 * b = 2)
variable (h_focal_length : ∀ (c : ℝ), 2 * c = 2 * real.sqrt 3)

-- Definition of the asymptote equation to be proven
theorem hyperbola_asymptote :
  let a := real.sqrt 2
  let b := 1
  (∀ x y : ℝ, (((x ^ 2) / (real.sqrt 2 ^ 2)) - ((y ^ 2) / (1 ^ 2)) = 1) →
    (y = (x * (real.sqrt 2) / 2) ∨ y = -(x * (real.sqrt 2) / 2))) :=
begin
  intro a, intro b,
  let c := real.sqrt 3,
  have h_eq : c ^ 2 = a ^ 2 + b ^ 2 := by
  {
    exact (c ^ 2) = (real.sqrt 3) ^ 2 = 3,
    rw [a ^ 2 + b ^ 2],
    exact 2 + 1 
  },
  have h_a_eq : a = real.sqrt 2 := by 
  {
    rw [h_eq],
    exact a ^ 2 = 2,
    rw pow_two a,
    exact real.sqrt_eq_rpow,
  },
  have h_b_eq : b = 1 := by 
  {
    rw [h_imaginary_axis],
    exact b = 1,
  },
  rw [h_b_eq],

  exact sorry
end

end hyperbola_asymptote_l55_55700


namespace general_term_formula_inequality_solution_set_l55_55412

-- Conditions given in the problem
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n m, a (n + m) = a n * q + a m

def arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n m, b (n + m) = b n + b m - b 0

-- General term formula of the sequence {a_n}
theorem general_term_formula (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
(h1 : ∑ i in (Finset.range 6), a i = 21)
(h2 : 2 * a 1, (3 / 2) * a 2, a 3 form_arithmetic_sequence) :
(∃ q, q = 1 ∧ ∀ n, a n = 7 / 2) ∨ (q = 2 ∧ ∀ n, a n = (1 / 3) * 2 ^ (n - 1)) :=
sorry

-- Inequality condition for arithmetic sequence {b_n}
theorem inequality_solution_set (a1 : ℝ) (T_n b_n: ℕ → ℝ)
(h3 : ∃ b : ℕ → ℝ, b 1 = 2 ∧ ∀ n, b n = 2 + (n - 1) * -a1)
(h4 : ∃ T : ℕ → ℝ, T n = 2 * n + (n * (n - 1) / 2) * -a1)
∃ n, T_n n - b_n n > 0 :=
if a1 = 7/2 then 
    -- when a1 = 7/2
    ∃ n, 1 < n ∧ n < 22/7 := sorry
else
    -- when a1 = 1/3
    ∃ n, 1 < n ∧ n < 14 := sorry

end general_term_formula_inequality_solution_set_l55_55412


namespace average_speed_round_trip_l55_55573

theorem average_speed_round_trip (dist : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) :
  dist = 48 → speed1 = 16 → speed2 = 24 →
  time1 = dist / speed1 → time2 = dist / speed2 →
  ((2 * dist) / (time1 + time2)) = 19.2 := by
  assume (h_dist : dist = 48) (h_speed1 : speed1 = 16) (h_speed2 : speed2 = 24)
  (h_time1 : time1 = dist / speed1) (h_time2 : time2 = dist / speed2)
  sorry

end average_speed_round_trip_l55_55573


namespace rectangle_area_l55_55664

-- Declare the given conditions
def circle_radius : ℝ := 5
def rectangle_width : ℝ := 2 * circle_radius
def length_to_width_ratio : ℝ := 2

-- Given that the length to width ratio is 2:1, calculate the length
def rectangle_length : ℝ := length_to_width_ratio * rectangle_width

-- Define the statement we need to prove
theorem rectangle_area :
  rectangle_length * rectangle_width = 200 :=
by
  sorry

end rectangle_area_l55_55664


namespace min_photos_required_l55_55823

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55823


namespace deposit_amount_is_105_l55_55659

-- Define the conditions
def deposit_percentage := 0.10
def remaining_amount := 945
def total_price (P : ℝ) := P
def deposit_amount (P : ℝ) := deposit_percentage * P

-- State the theorem
theorem deposit_amount_is_105 : 
  ∀ (P : ℝ), total_price P - deposit_amount P = remaining_amount → deposit_amount P = 105 := 
by 
  intro P 
  assume h
  sorry

end deposit_amount_is_105_l55_55659


namespace exists_infinite_n_l55_55975

variable (k : ℕ)

-- Define sum_of_three_cubes predicate stating that a number can be expressed as the sum of three positive cubes
def sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a^3 + b^3 + c^3

-- Prove the existence of infinitely many n for each case i = 1, 2, 3
theorem exists_infinite_n (i : ℕ) (hi : i = 1 ∨ i = 2 ∨ i = 3) : ∃∞ n : ℕ,
  (nat.card {m | sum_of_three_cubes m}) = i ∧ ¬ sum_of_three_cubes n ∧ sum_of_three_cubes (n + 2) ∧ sum_of_three_cubes (n + 28) :=
begin
  sorry
end

end exists_infinite_n_l55_55975


namespace Jerry_wants_to_raise_average_l55_55168

theorem Jerry_wants_to_raise_average 
  (first_three_tests_avg : ℕ) (fourth_test_score : ℕ) (desired_increase : ℕ) 
  (h1 : first_three_tests_avg = 90) (h2 : fourth_test_score = 98) 
  : desired_increase = 2 := 
by
  sorry

end Jerry_wants_to_raise_average_l55_55168


namespace curved_surface_area_of_cone_l55_55292

noncomputable def π := Real.pi

def radius : ℝ := 28
def slant_height : ℝ := 30

def curved_surface_area (r l : ℝ) : ℝ := π * r * l

theorem curved_surface_area_of_cone :
  curved_surface_area radius slant_height ≈ 2638.932 :=
by
  sorry -- Proof can be filled in later

end curved_surface_area_of_cone_l55_55292


namespace jasper_hot_dogs_fewer_l55_55924

theorem jasper_hot_dogs_fewer (chips drinks hot_dogs : ℕ)
  (h1 : chips = 27)
  (h2 : drinks = 31)
  (h3 : drinks = hot_dogs + 12) : 27 - hot_dogs = 8 := by
  sorry

end jasper_hot_dogs_fewer_l55_55924


namespace david_profit_l55_55686

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l55_55686


namespace vector_magnitude_subtraction_l55_55066

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55066


namespace cannot_form_2x2_square_l55_55285

def can_form_figures (squares : ℕ) (rectangle : ℕ) (fig_D : bool) : Prop :=
  ¬fig_D

theorem cannot_form_2x2_square (squares : ℕ) (rectangle : ℕ) :
  squares = 3 → rectangle = 1 → can_form_figures squares rectangle true := sorry

end cannot_form_2x2_square_l55_55285


namespace donovan_incorrect_answers_l55_55765

theorem donovan_incorrect_answers :
  ∃ (T : ℕ), (35 * 100 / T = 72.92) ∧ (T - 35 = 13) :=
sorry

end donovan_incorrect_answers_l55_55765


namespace trig_proof_l55_55833

theorem trig_proof (α : ℝ) (h : sin (π / 6 - α) - cos α = 1 / 3) : 
  cos (2 * α + π / 3) = 7 / 9 := 
by
  sorry

end trig_proof_l55_55833


namespace solve_for_x_l55_55989

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l55_55989


namespace part1_part2_l55_55178

variable {a : ℝ} (M N : Set ℝ)

theorem part1 (h : a = 1) : M = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (hM : (M = {x : ℝ | 0 < x ∧ x < a + 1}))
              (hN : N = {x : ℝ | -1 ≤ x ∧ x ≤ 3})
              (h_union : M ∪ N = N) : 
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end part1_part2_l55_55178


namespace find_angle_B_and_sin_ratio_l55_55917

variable (A B C a b c : ℝ)
variable (h₁ : a * (Real.sin C - Real.sin A) / (Real.sin C + Real.sin B) = c - b)
variable (h₂ : Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4)

theorem find_angle_B_and_sin_ratio :
  B = Real.pi / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end find_angle_B_and_sin_ratio_l55_55917


namespace correct_sampling_methods_l55_55621

-- Define conditions for the sampling problems
structure SamplingProblem where
  scenario: String
  samplingMethod: String

-- Define the three scenarios
def firstScenario : SamplingProblem :=
  { scenario := "Draw 5 bottles from 15 bottles of drinks for food hygiene inspection", samplingMethod := "Simple random sampling" }

def secondScenario : SamplingProblem :=
  { scenario := "Sample 20 staff members from 240 staff members in a middle school", samplingMethod := "Stratified sampling" }

def thirdScenario : SamplingProblem :=
  { scenario := "Select 25 audience members from a full science and technology report hall", samplingMethod := "Systematic sampling" }

-- Main theorem combining all conditions and proving the correct answer
theorem correct_sampling_methods :
  (firstScenario.samplingMethod = "Simple random sampling") ∧
  (secondScenario.samplingMethod = "Stratified sampling") ∧
  (thirdScenario.samplingMethod = "Systematic sampling") :=
by
  sorry -- Proof is omitted

end correct_sampling_methods_l55_55621


namespace possible_y_values_l55_55949

theorem possible_y_values (x : ℝ) (h : x^3 + 6* (x / (x - 3))^3 = 135) :
    y = \frac{(x - 3)^3 * (x + 4)}{3 * x - 4} ∈ {0, \frac{23382}{122}} :=
sorry

end possible_y_values_l55_55949


namespace min_photos_needed_to_ensure_conditions_l55_55804

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55804


namespace one_fourth_in_one_eighth_l55_55121

theorem one_fourth_in_one_eighth : (1/8 : ℚ) / (1/4) = (1/2) := 
by
  sorry

end one_fourth_in_one_eighth_l55_55121


namespace probability_of_different_colors_l55_55483

def boxA : finset (string × ℕ) := finset.from_list [("red", 3), ("black", 3), ("white", 3)]
def boxB : finset (string × ℕ) := finset.from_list [("yellow", 2), ("black", 2), ("white", 2)]

def draw_ball (box : finset (string × ℕ)) : list string :=
  box.flat_map (λ p, list.replicate p.2 p.1)

def draw_different_colors_probability : ℚ :=
  let drawsA := (draw_ball boxA).erase_dup,
      drawsB := (draw_ball boxB).erase_dup in
  (drawsA.product drawsB).countp (λ pair, pair.1 ≠ pair.2) / (drawsA.product drawsB).length

theorem probability_of_different_colors : draw_different_colors_probability = 2/9 := 
by sorry

end probability_of_different_colors_l55_55483


namespace primitive_nth_root_of_unity_l55_55211

-- Definitions and conditions
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def nth_root_of_unity (n m : ℕ) : ℂ := Complex.exp (2 * m * Real.pi * Complex.I / n)

-- Statement of the problem
theorem primitive_nth_root_of_unity (n m : ℕ) (h_coprime : is_coprime m n) : ∀ k : ℕ, 1 ≤ k < n → nth_root_of_unity n m ≠ 1 := 
by
  sorry

end primitive_nth_root_of_unity_l55_55211


namespace total_amount_spent_l55_55500

-- Definitions based on the conditions
def games_this_month := 11
def cost_per_ticket_this_month := 25
def total_cost_this_month := games_this_month * cost_per_ticket_this_month

def games_last_month := 17
def cost_per_ticket_last_month := 30
def total_cost_last_month := games_last_month * cost_per_ticket_last_month

def games_next_month := 16
def cost_per_ticket_next_month := 35
def total_cost_next_month := games_next_month * cost_per_ticket_next_month

-- Lean statement for the proof problem
theorem total_amount_spent :
  total_cost_this_month + total_cost_last_month + total_cost_next_month = 1345 :=
by
  -- proof goes here
  sorry

end total_amount_spent_l55_55500


namespace find_a_l55_55837

-- We define the conditions:
-- a positive real number a, equation for circle M, equation for line, and the chord length condition
variables (a : ℝ) (ha : a > 0) 

-- Definition: Equation of the circle M
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*a*y = 0

-- Definition: Equation of the line
def is_line (x y : ℝ) : Prop := x + y = 0

-- Condition: Length of the chord intercepted by the circle on the straight line
def chord_length_condition : Prop := ∃ (x1 y1 x2 y2 : ℝ), is_circle x1 y1 ∧ is_circle x2 y2 ∧ is_line x1 y1 ∧ is_line x2 y2 ∧ 
  (real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * real.sqrt 2)

-- Statement to prove that a equals 2 under given conditions
theorem find_a : ha →
  chord_length_condition a →
  a = 2 := 
sorry

end find_a_l55_55837


namespace min_omega_satisfies_conditions_l55_55934

theorem min_omega_satisfies_conditions:
  ∃ (ω: ℝ), ω > 0 ∧ (∀ (x: ℝ), sin (ω * (x + 4 * π / 3) + π / 3) + 2 = sin (ω * x + π / 3) + 2) ∧ ω = 3 / 2 :=
by
  sorry

end min_omega_satisfies_conditions_l55_55934


namespace solve_equation_l55_55556

noncomputable def eq1 (x : ℝ) : Prop :=
  5 ^ (Real.sqrt (x ^ 3 + 3 * x ^ 2 + 3 * x + 1)) = Real.sqrt ((5 * Real.root 4 ((x + 1) ^ 5)) ^ 3)

theorem solve_equation (x : ℝ) : eq1 x → x = 65 / 16 := 
sorry

end solve_equation_l55_55556


namespace vector_magnitude_difference_l55_55014

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55014


namespace find_a_l55_55890

-- The conditions converted to Lean definitions
variable (a : ℝ)
variable (α : ℝ)
variable (point_on_terminal_side : a ≠ 0 ∧ (∃ α, tan α = -1 / 2 ∧ ∀ y : ℝ, y = -1 → a = 2 * y) )

-- The theorem statement
theorem find_a (H : point_on_terminal_side): a = 2 := by
  sorry

end find_a_l55_55890


namespace credit_sales_ratio_l55_55229

theorem credit_sales_ratio : 
    ∀ (total_sales cash_sales : ℕ), 
    total_sales = 80 → 
    cash_sales = 48 → 
    let credit_sales := total_sales - cash_sales in 
    (credit_sales.to_rat / total_sales.to_rat) = (2 : ℚ) / 5 := 
by 
  intros total_sales cash_sales h_sales h_cash 
  let credit_sales := total_sales - cash_sales 
  have h_credit : credit_sales = 32 := 
    by rw [h_sales, h_cash]; exact nat.sub_eq_iff_eq_add.mpr rfl 
  rw [← nat.cast_sub $ nat.le_of_lt $ nat.lt_of_lt_of_le 0, rat.cast_sub, h_sales, h_cash, nat.cast_bit0, rat.div_def,
      mul_comm, one_mul, nat.cast_bit0] at h_credit 
  exact (rat.div_cancel 32 80).symm.trans 
    (show (credit_sales : ℚ) = 32, from by rw [h_credit, nat.cast_succ 31, nat.cast_add, nat.cast_one])
    sorry

end credit_sales_ratio_l55_55229


namespace circumcircle_radius_l55_55665

theorem circumcircle_radius (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  ∃ r, r = c / 2 :=
by
  use c / 2
  sorry

example : 
  circumcircle_radius 8 15 17 
    (by norm_num [sq, add_sub_cancel', eq_comm]) = ∃ r, r = 17 / 2 :=
by
  norm_num
  sorry

end circumcircle_radius_l55_55665


namespace vector_magnitude_subtraction_l55_55006

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55006


namespace vec_magnitude_is_five_l55_55056

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55056


namespace ratio_purely_imaginary_l55_55964

theorem ratio_purely_imaginary (z1 z2 : ℂ) (hz1 : z1 ≠ 0) (hz2 : z2 ≠ 0)
  (h : |z1 + z2| = |z1 - z2|) : ∃ (c : ℂ), c.im ≠ 0 ∧ c.re = 0 ∧ c = z1 / z2 := by
  sorry

end ratio_purely_imaginary_l55_55964


namespace cos_dihedral_angle_tetrahedron_l55_55839

theorem cos_dihedral_angle_tetrahedron :
  ∀ (S A B C : ℝ) 
  (base_is_equilateral : ∀ {a b c : ℝ}, (a = 1) → (b = 1) → (c = 1) → equilateral_triangle a b c)
  (side_edges_length : ∀ {sa sb sc : ℝ}, (sa = 2) → (sb = 2) → (sc = 2) → regular_tetrahedron S A B C),
  (cross_section_divides_volume : cross_section_AB_divides_volume_in_half S A B C)
  → cos_dihedral_angle (cross_section_AB S A B C) (base_S A B C) = (2 * real.sqrt 15) / 15 := 
sorry

end cos_dihedral_angle_tetrahedron_l55_55839


namespace smallest_value_a_b_c_d_l55_55518

open Matrix

theorem smallest_value_a_b_c_d :
  ∃ (a b c d : ℕ), 
    (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
    (∃ M1 M2 : Matrix (Fin 2) (Fin 2) ℤ, 
      M1 = !![2, 0; 0, 4] ∧ 
      M2 = !![8, 6; -24, -16] ∧ 
      M1 * !![a, b; c, d] = !![a, b; c, d] * M2) ∧ 
    a + b + c + d = 12 :=
  sorry

end smallest_value_a_b_c_d_l55_55518


namespace base_area_of_cylinder_l55_55135

variables (S : ℝ) (cylinder : Type)
variables (square_cross_section : cylinder → Prop) (area_square : cylinder → ℝ)
variables (base_area : cylinder → ℝ)

-- Assume that the cylinder has a square cross-section with a given area
axiom cross_section_square : ∀ c : cylinder, square_cross_section c → area_square c = 4 * S

-- Theorem stating the area of the base of the cylinder
theorem base_area_of_cylinder (c : cylinder) (h : square_cross_section c) : base_area c = π * S :=
by
  -- Proof omitted
  sorry

end base_area_of_cylinder_l55_55135


namespace vector_magnitude_l55_55100

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55100


namespace p_sufficient_not_necessary_for_q_l55_55417

variable (x : ℝ)

def condition_p : Prop := log 2 (x - 1) < 1

def condition_q : Prop := x^2 - 2 * x - 3 < 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, condition_p x → 1 < x ∧ x < 3) → 
  (∀ x, condition_q x → -1 < x ∧ x < 3) → 
  (∀ x, (1 < x ∧ x < 3) → (-1 < x ∧ x < 3)) ∧ 
  ¬(∀ x, (-1 < x ∧ x < 3) → (1 < x ∧ x < 3)) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l55_55417


namespace pool_capacity_percentage_l55_55227

noncomputable def hose_rate := 60 -- cubic feet per minute
noncomputable def pool_width := 80 -- feet
noncomputable def pool_length := 150 -- feet
noncomputable def pool_depth := 10 -- feet
noncomputable def drainage_time := 2000 -- minutes
noncomputable def pool_volume := pool_width * pool_length * pool_depth -- cubic feet
noncomputable def removed_water_volume := hose_rate * drainage_time -- cubic feet

theorem pool_capacity_percentage :
  (removed_water_volume / pool_volume) * 100 = 100 :=
by
  -- the proof steps would go here
  sorry

end pool_capacity_percentage_l55_55227


namespace reachable_squares_by_knight_l55_55880

theorem reachable_squares_by_knight (n : ℕ) : ℕ :=
match n with
| 1 => 8
| 2 => 32
| 3 => 68
| 4 => 96
| _ => 28 * n - 20

example : ∀ n : ℕ, reachable_squares_by_knight n = 
  match n with
  | 1 => 8
  | 2 => 32
  | 3 => 68
  | 4 => 96
  | _ => 28 * n - 20 :=
by intros n; cases n; exact rfl; -- Lean syntax to cover all cases and prove it with the given formula.

end reachable_squares_by_knight_l55_55880


namespace Julio_current_age_l55_55172

theorem Julio_current_age (J : ℕ) (James_current_age : ℕ) (h1 : James_current_age = 11)
    (h2 : J + 14 = 2 * (James_current_age + 14)) : 
    J = 36 := 
by 
  sorry

end Julio_current_age_l55_55172


namespace units_digit_product_even_20_90_l55_55628

theorem units_digit_product_even_20_90 : 
  (∃ (d : ℕ), (d % 10 = 0) ∧ (d = ∏ i in (finset.filter (λ x : ℕ, x % 2 = 0) (finset.range 91 \ finset.range 20)), i)) :=
by
  sorry

end units_digit_product_even_20_90_l55_55628


namespace cost_split_equally_l55_55501

theorem cost_split_equally {price_cake price_cookies total cost_after_discount tax_rate discount_rate : ℝ} 
  (h1 : price_cake = 12) 
  (h2 : price_cookies = 5)
  (quantity_cake : ℤ)
  (h3 : quantity_cake = 3)
  (h4 : discount_rate = 0.10)
  (h5 : tax_rate = 0.05)
  (h6 : total = quantity_cake * price_cake + price_cookies)
  (h7 : cost_after_discount = total * (1 - discount_rate)):
  let tax_amount := cost_after_discount * tax_rate in
  let final_cost := cost_after_discount + tax_amount in
  let cost_per_brother := final_cost / 2 in
  Real.round(cost_per_brother * 100) / 100 = 19.38 :=
by
  sorry

end cost_split_equally_l55_55501


namespace sum_of_roots_abs_gt_six_l55_55882

theorem sum_of_roots_abs_gt_six {p r1 r2 : ℝ} (h1 : r1 + r2 = -p) (h2 : r1 * r2 = 9) (h3 : r1 ≠ r2) (h4 : p^2 > 36) : |r1 + r2| > 6 :=
sorry

end sum_of_roots_abs_gt_six_l55_55882


namespace triangle_inequality_for_roots_l55_55851

theorem triangle_inequality_for_roots (p q r : ℝ) (hroots_pos : ∀ (u v w : ℝ), (u > 0) ∧ (v > 0) ∧ (w > 0) ∧ (u * v * w = -r) ∧ (u + v + w = -p) ∧ (u * v + u * w + v * w = q)) :
  p^3 - 4 * p * q + 8 * r > 0 :=
sorry

end triangle_inequality_for_roots_l55_55851


namespace vector_magnitude_difference_l55_55023

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55023


namespace min_photos_l55_55793

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55793


namespace problem1_l55_55998

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end problem1_l55_55998


namespace vector_magnitude_subtraction_l55_55010

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55010


namespace smallest_multiple_of_30_l55_55938

def f (n : ℕ) : ℕ := Nat.find (λ k => (Nat.factorial k) % n = 0)

theorem smallest_multiple_of_30 (n : ℕ) (h₀ : n = 30 * 23) : f(n) > 20 := by
  sorry

end smallest_multiple_of_30_l55_55938


namespace gcd_9247_4567_eq_1_l55_55624

theorem gcd_9247_4567_eq_1 : Int.gcd 9247 4567 = 1 := sorry

end gcd_9247_4567_eq_1_l55_55624


namespace vector_magnitude_difference_l55_55094

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55094


namespace integral_evaluation_l55_55770

theorem integral_evaluation :
  ∫ x in -1..1, (Real.sqrt (4 - x^2) + x^3) = Real.sqrt 3 + (2 * Real.pi) / 3 :=
by
  sorry

end integral_evaluation_l55_55770


namespace Natalia_Tuesday_distance_l55_55200

theorem Natalia_Tuesday_distance :
  ∃ T : ℕ, (40 + T + T / 2 + (40 + T / 2) = 180) ∧ T = 33 :=
by
  existsi 33
  -- proof can be filled here
  sorry

end Natalia_Tuesday_distance_l55_55200


namespace find_x_eq_19_l55_55781

theorem find_x_eq_19 : ∀ x : ℝ, |x - 21| + |x - 17| = |2 * x - 38| ↔ x = 19 := 
by
  intro x
  split
  { intro h
    have h1 : |2 * x - 38| = 2 * |x - 19| :=
      by rw [abs_of_nonneg, abs_of_nonneg, sub_eq_add_neg, sub_eq_add_neg]; ring
    rw [h1] at h
    sorry
  }
  { intro h
    rw [h, sub_self, sub_self, add_self_eq_zero, add_zero, sub_self]
    sorry
  }

end find_x_eq_19_l55_55781


namespace maximum_zero_triangles_l55_55474

-- Definitions and assumptions on points and vectors are abstract
def is_zero_triangle (p1 p2 p3 : Point) := 
  let v12 := vector_from p1 p2
  let v23 := vector_from p2 p3
  let v31 := vector_from p3 p1
  v12 + v23 + v31 = 0 -- geometrically meaning the vectors add up to zero

theorem maximum_zero_triangles : 
  ∀ (points : Finset Point), 
    points.card = 12 → 
    (∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → collinear p1 p2 p3 → p1 = p2 ∨ p2 = p3 ∨ p3 = p1) →
    ∃ (triangles : set (Point × Point × Point)), 
      (∀ (p1 p2 p3 : Point), (p1, p2, p3) ∈ triangles →
        sorry_condition_for_triangle (p1, p2, p3) is_zero_triangle) ∧
      triangles.size = 70 :=
sorry

end maximum_zero_triangles_l55_55474


namespace min_rainfall_on_fourth_day_l55_55610

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end min_rainfall_on_fourth_day_l55_55610


namespace vector_magnitude_correct_l55_55040

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55040


namespace probability_at_least_one_defective_is_correct_l55_55662

noncomputable def probability_at_least_one_defective : ℚ :=
  let total_bulbs := 23
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs
  let probability_neither_defective :=
    (non_defective_bulbs / total_bulbs) * ((non_defective_bulbs - 1) / (total_bulbs - 1))
  1 - probability_neither_defective

theorem probability_at_least_one_defective_is_correct :
  probability_at_least_one_defective = 164 / 506 :=
by
  sorry

end probability_at_least_one_defective_is_correct_l55_55662


namespace speed_conversion_l55_55530

theorem speed_conversion (speed_kmph : ℕ) (conversion_rate : ℚ) : (speed_kmph = 600) ∧ (conversion_rate = 0.6) → (speed_kmph * conversion_rate / 60 = 6) :=
by
  sorry

end speed_conversion_l55_55530


namespace volume_of_first_cylinder_l55_55618

theorem volume_of_first_cylinder (h : ℝ) (r : ℝ) (V2 : ℝ) (h_eq : h ≠ 0) (r_eq : r ≠ 0) :
  (π * (3 * r)^2 * h = 360) → (π * r^2 * h = 40) :=
by
  intros h_eq r_eq vol2_eq
  sorry

end volume_of_first_cylinder_l55_55618


namespace area_of_rectangle_l55_55648

namespace RectangleArea

variable (l b : ℕ)
variable (h1 : l = 3 * b)
variable (h2 : 2 * (l + b) = 88)

theorem area_of_rectangle : l * b = 363 :=
by
  -- We will prove this in Lean 
  sorry

end RectangleArea

end area_of_rectangle_l55_55648


namespace probability_correct_l55_55336

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let prob_neither_multiple_of_4 := (non_multiples_of_4 / total_numbers) * (non_multiples_of_4 / total_numbers)
  1 - prob_neither_multiple_of_4

theorem probability_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  unfold probability_at_least_one_multiple_of_4
  norm_num
  sorry

end probability_correct_l55_55336


namespace parallelogram_parallel_lines_l55_55177

variable {AB CD AD BP BQ AP PC CQ AQ EP PB FQ BQ EF AC : ℝ}

theorem parallelogram_parallel_lines
  (ABCD_is_parallelogram : parallelogram ABCD)
  (h1 : AB > AD)
  (h2 : AP = CQ)
  (h3 : P < C)
  (hE : E = intersection_line BP AD)
  (hF : F = intersection_line BQ CD) :
  parallel EF AC :=
sorry

end parallelogram_parallel_lines_l55_55177


namespace min_photos_for_condition_l55_55814

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55814


namespace area_of_ABCD_is_sqrt_24_plus_72_sqrt_7_l55_55150

variables {AB BC CD DA AC BE CE AE h : ℝ}
variables (ABCD : ℝ → ℝ → ℝ → ℝ → Prop)

def is_convex_quadrilateral (AB BC CD DA : ℝ) : Prop :=
  AB = 10 ∧ BC = 6 ∧ CD = 12 ∧ DA = 12 ∧ (90 : ℝ) = 90

theorem area_of_ABCD_is_sqrt_24_plus_72_sqrt_7 :
  is_convex_quadrilateral 10 6 12 12 →
  ∃ a b c : ℕ, a = 24 ∧ b = 72 ∧ c = 7 ∧ (72 + 24 * Real.sqrt(7) = 72 + 24 * Real.sqrt(7)) :=
by 
  sorry

end area_of_ABCD_is_sqrt_24_plus_72_sqrt_7_l55_55150


namespace number_of_common_elements_l55_55139

theorem number_of_common_elements (x y : Set ℤ) (h1 : x.finite) (h2 : y.finite) 
(hx : x.card = 12) (hy : y.card = 18) (hxy : (x.symDiff y).card = 18) : 
↑((x ∩ y).card) = 12 :=
by
  sorry

end number_of_common_elements_l55_55139


namespace ellipse_equation_l55_55680

noncomputable def ellipse_property (a b : ℝ) : Prop :=
ab = 2 ∧ a^2 - b^2 = 3

theorem ellipse_equation : 
  ∃ a b : ℝ, ellipse_property a b ∧ ellipse_equation a b = "x^2/4 + y^2 = 1" := 
by
  sorry

end ellipse_equation_l55_55680


namespace vector_magnitude_subtraction_l55_55026

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55026


namespace min_photos_for_condition_l55_55808

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55808


namespace find_z_coord_l55_55672

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables (p1 p2 : Point) (x_coord : ℝ)

def line_passing_through : Point := 
  p1 + (x_coord - 1) • (p2 - p1)

theorem find_z_coord (p1 p2 : Point) (x_coord z_coord: ℝ) 
  (h_p1 : p1 = (⟨1, 3, 2⟩ : Point))
  (h_p2 : p2 = (⟨4, 2, -1⟩ : Point))
  (h_x : x_coord = 3) : 
  line_passing_through p1 p2 x_coord = ⟨3, 3 - 1, z_coord⟩ → z_coord = 0 :=
sorry

end find_z_coord_l55_55672


namespace area_of_annulus_l55_55695

section annulus
variables {R r x : ℝ}
variable (h1 : R > r)
variable (h2 : R^2 - r^2 = x^2)

theorem area_of_annulus (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 :=
sorry

end annulus

end area_of_annulus_l55_55695


namespace targets_break_order_count_l55_55902

theorem targets_break_order_count :
  let arrangements := finset.card (finset.perm (multiset.of_list ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'])) in
  arrangements = 560 :=
by
  sorry

end targets_break_order_count_l55_55902


namespace simplify_and_rationalize_l55_55552

theorem simplify_and_rationalize :
  let x := (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17)
  x = 3 * Real.sqrt 84885 / 1309 := sorry

end simplify_and_rationalize_l55_55552


namespace centroids_coincide_l55_55539

variable {α : Type*} [MetricSpace α]

-- Given an arbitrary triangle ABC
variables (A B C A1 B1 C1 : α) 

-- Triangles A1BC, B1CA and C1AB are similar
variables (hSim1 : Similar (triangle A B C) (triangle A1 B C))
variables (hSim2 : Similar (triangle B C A) (triangle B1 C A))
variables (hSim3 : Similar (triangle C A B) (triangle C1 A B))

-- Define Centroid for a given triangle
noncomputable def centroid (A B C : α) : α := sorry

-- The theorem to be proven
theorem centroids_coincide :
  centroid A B C = centroid A1 B1 C1 := sorry

end centroids_coincide_l55_55539


namespace Series_value_l55_55361

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l55_55361


namespace min_photos_l55_55792

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55792


namespace sum_inequality_for_all_n_leq_91_l55_55545

open Nat

theorem sum_inequality_for_all_n_leq_91 :
  ∀ (n : ℕ), n > 0 → n ≤ 91 → (Σ i in range(n), (1 : ℝ) / (i^2 + 2020)) < 1 / 22 :=
by
  intros n hn h91
  sorry

end sum_inequality_for_all_n_leq_91_l55_55545


namespace quotient_remainder_threefold_l55_55635

theorem quotient_remainder_threefold (a b c d : ℤ)
  (h : a = b * c + d) :
  3 * a = 3 * b * c + 3 * d :=
by sorry

end quotient_remainder_threefold_l55_55635


namespace find_quadratic_function_determine_range_m_l55_55889

-- Theorem for the expression of f(x)
theorem find_quadratic_function (a b c : ℝ) :
  (∀ x : ℝ, (a * (x + 1) ^ 2 + b * (x + 1) + c) - (a * x ^ 2 + b * x + c) = 2 * x) ∧ (c = 1) ∧ (a = 1) ∧ (b = -1) →
  ∀ x : ℝ, f(x) = x^2 - x + 1 :=
  sorry

-- Theorem for the range of m
theorem determine_range_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (x^2 - x + 1) ≥ (2 * x + m)) ↔ (m ≤ -1) :=
  sorry

end find_quadratic_function_determine_range_m_l55_55889


namespace g_value_l55_55196

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ :=
if h : x ≥ 0 then log (x + 1) / log 3 else g x

-- Hypothesis that f is an odd function
axiom f_odd : odd_function f

-- Goal
theorem g_value : g (-8) = -2 :=
by
  -- Using the definition of odd function
  have h1 : f 8 = log 9 / log 3 := rfl
  have h2 : f (-8) = -f 8 := f_odd 8
  rw h2
  have h3 : f (-8) = g (-8) := rfl
  rw h3 at h2
  sorry

end g_value_l55_55196


namespace max_power_sum_l55_55711

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55711


namespace solve_for_x_l55_55993

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l55_55993


namespace q1_q2_l55_55865

variables {R : Type*} [LinearOrderedField R]
noncomputable def f (a b x : R) : R := a * x^2 + b * x + 1

theorem q1 {a b : R} (h₁ : f a b (-1) = 0) (h₂ : -b / (2 * a) = -1) :
  a = 1 ∧ b = 2 ∧ 
  (∀ x : R, x < -1 → differentiable_at R (f 1 2) x ∧ deriv (f 1 2) x > 0) ∧
  (∀ x : R, -1 < x → differentiable_at R (f 1 2) x ∧ deriv (f 1 2) x < 0) :=
sorry

theorem q2 {k : R} 
  (h₁ : f 1 2 (-1) = 0) 
  (h₂ : ∀ x, x ∈ Icc (-3 : R) (-1) → f 1 2 x > x + k) :
  k < 1 :=
sorry

end q1_q2_l55_55865


namespace curve_C_cartesian_PA_PB_value_l55_55864

-- Parametric equations of line l
def line_l_x (t : ℝ) : ℝ := (1 / 2) * t
def line_l_y (t : ℝ) : ℝ := 1 + (Real.sqrt 3 / 2) * t

-- Polar equation of curve C
def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.sin (θ + π / 4)

-- Cartesian equation conversion
theorem curve_C_cartesian (x y : ℝ) : (∃ ρ θ : ℝ, 
  ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x ∧ curve_C_polar ρ θ) ↔
  (x - 1)^2 + (y - 1)^2 = 2 := by
  sorry

-- Calculating the value
theorem PA_PB_value {t t1 t2 : ℝ} (h_line : ∀ t, line_l_x t = t1 ∧ line_l_y t = t2)
  (h_curve : ∀ t, (line_l_x t - 1)^2 + (line_l_y t - 1)^2 = 2) :
  (|1 / t1|) + (|1 / t2|) = Real.sqrt 5 := by
  sorry

end curve_C_cartesian_PA_PB_value_l55_55864


namespace M_intersect_N_eq_l55_55194

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- Define what we need to prove
theorem M_intersect_N_eq : M ∩ N = {y | y ≥ 1} :=
by
  sorry

end M_intersect_N_eq_l55_55194


namespace distinct_triangles_octahedron_l55_55113

-- Define the regular octahedron with 8 vertices and the collinearity property
def regular_octahedron_vertices : Nat := 8

def no_three_vertices_collinear (vertices: Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), {a, b, c} ⊆ vertices → ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r

-- The main theorem stating the problem
theorem distinct_triangles_octahedron :
  no_three_vertices_collinear (Finset.range regular_octahedron_vertices) →
  (Finset.card (Finset.univ.choose 3)) = 56 :=
by
  sorry

end distinct_triangles_octahedron_l55_55113


namespace greatest_value_sum_eq_24_l55_55725

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l55_55725


namespace largest_k_exists_l55_55389

theorem largest_k_exists :
  ∃ (n k : ℕ), k = 108 ∧ k ≥ 2 ∧ n ≥ k ∧
  (∀ (S : Finset ℕ), S.card = n → 
    ∃ (T : Finset ℕ), T ⊆ S ∧ T.card = k ∧ 
    (∀ x ∈ T, ¬(x % 6 = 0 ∨ x % 7 = 0 ∨ x % 8 = 0)) ∧ 
    (∀ x y ∈ T, x ≠ y → (x = y ∨ (x - y) % 6 ≠ 0 ∨ (x - y) % 7 ≠ 0 ∨ (x - y) % 8 ≠ 0))) :=
sorry

end largest_k_exists_l55_55389


namespace area_of_right_triangle_l55_55236

variable (h l : ℝ)

def right_triangle_area (h l : ℝ) : ℝ :=
  (1/2) * h * Real.sqrt (l^2 + 4 * h^2)

theorem area_of_right_triangle (h l : ℝ) :
  right_triangle_area h l = (1/2) * h * Real.sqrt (l^2 + 4 * h^2) := by
  sorry

end area_of_right_triangle_l55_55236


namespace vector_magnitude_l55_55098

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55098


namespace multiplier_for_second_part_l55_55675

theorem multiplier_for_second_part {x y k : ℝ} (h1 : x + y = 52) (h2 : 10 * x + k * y = 780) (hy : y = 30.333333333333332) (hx : x = 21.666666666666668) :
  k = 18.571428571428573 :=
by
  sorry

end multiplier_for_second_part_l55_55675


namespace enrico_earnings_l55_55368

def roosterPrice (weight: ℕ) : ℝ :=
  if weight < 20 then weight * 0.80
  else if weight ≤ 35 then weight * 0.65
  else weight * 0.50

theorem enrico_earnings :
  roosterPrice 15 + roosterPrice 30 + roosterPrice 40 + roosterPrice 50 = 76.50 := 
by
  sorry

end enrico_earnings_l55_55368


namespace cupcakes_frosted_in_5_minutes_l55_55340

theorem cupcakes_frosted_in_5_minutes :
  (let r_cagney := (1 : ℚ) / 20;
       r_lacey := (1 : ℚ) / 30;
       combined_rate := r_cagney + r_lacey in 
       300 * combined_rate = 25) := 
by {
  -- Define Cagney's and Lacey's rates
  let r_cagney := (1 : ℚ) / 20,
  let r_lacey := (1 : ℚ) / 30,

  -- Calculate combined rate
  let combined_rate := r_cagney + r_lacey,

  -- Express the total number of cupcakes frosted in 300 seconds
  have h : 300 * combined_rate = 25, by {
    calc 300 * combined_rate
          = 300 * ((1 / 20) + (1 / 30)) : by { refl }
      ... = 300 * ((3 / 60) + (2 / 60)) : by { congr; field_simp [ne_of_gt (show 20 > 0, by norm_num)] }
      ... = 300 * (5 / 60) : by { congr; field_simp [ne_of_gt (show 30 > 0, by norm_num)] }
      ... = 300 * (1 / 12) : by { norm_num }
      ... = 25 : by norm_num,
  },
  exact h,
}

end cupcakes_frosted_in_5_minutes_l55_55340


namespace sum_a_n_inv_eq_88_l55_55184

noncomputable def a_n (n : ℕ) : ℕ :=
  if h : n > 0 then (Int.floor (Real.sqrt n : ℝ)).natAbs
  else 0

theorem sum_a_n_inv_eq_88 : 
  (∑ k in Finset.range 1980, (1 : ℝ) / a_n (k + 1)) = 88 := 
sorry

end sum_a_n_inv_eq_88_l55_55184


namespace arithmetic_sequence_length_l55_55878

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ∀ (a_1 a_2 a_n : ℤ), a_1 = 2 ∧ a_2 = 6 ∧ a_n = 2006 →
  a_n = a_1 + (n - 1) * (a_2 - a_1) → n = 502 := by
  sorry

end arithmetic_sequence_length_l55_55878


namespace range_f_l55_55393

noncomputable def f (A : ℝ) : ℝ :=
  (sin A * (3 * cos A ^ 2 + cos A ^ 4 + 3 * sin A ^ 2 + sin A ^ 2 * cos A ^ 2)) /
  (tan A * (sec A - sin A * tan A))

theorem range_f (A : ℝ) (h : ∀ (n : ℤ), A ≠ n * π / 2) : 
  Set.range (f) = Set.Ioo 3 4 :=
sorry

end range_f_l55_55393


namespace cone_base_radius_l55_55251

theorem cone_base_radius (slant_height : ℝ) (central_angle_deg : ℝ) (r : ℝ) 
  (h1 : slant_height = 6) 
  (h2 : central_angle_deg = 120) 
  (h3 : 2 * π * slant_height * (central_angle_deg / 360) = 4 * π) 
  : r = 2 := by
  sorry

end cone_base_radius_l55_55251


namespace cylinder_height_and_diameter_l55_55598

/-- The surface area of a sphere is the same as the curved surface area of a right circular cylinder.
    The height and diameter of the cylinder are the same, and the radius of the sphere is 4 cm.
    Prove that the height and diameter of the cylinder are both 8 cm. --/
theorem cylinder_height_and_diameter (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ)
  (h1 : r_sphere = 4)
  (h2 : 4 * π * r_sphere^2 = 2 * π * r_cylinder * h_cylinder)
  (h3 : h_cylinder = 2 * r_cylinder) :
  h_cylinder = 8 ∧ r_cylinder = 4 :=
by
  -- Proof to be completed
  sorry

end cylinder_height_and_diameter_l55_55598


namespace vec_magnitude_is_five_l55_55055

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55055


namespace greatest_integer_value_l55_55273

theorem greatest_integer_value {x : ℤ} (h : ∀ x, |3*x - 2| ≤ 21) : x = 7 := by
  sorry

end greatest_integer_value_l55_55273


namespace triangle_sides_consecutive_obtuse_l55_55888

/-- Given the sides of a triangle are consecutive natural numbers 
    and the largest angle is obtuse, 
    the lengths of the sides in ascending order are 2, 3, 4. -/
theorem triangle_sides_consecutive_obtuse 
    (x : ℕ) (hx : x > 1) 
    (cos_alpha_neg : (x - 4) < 0) 
    (x_lt_4 : x < 4) :
    (x = 3) → (∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a = 2 ∧ b = 3 ∧ c = 4) :=
by
  intro hx3
  use 2, 3, 4
  repeat {split}
  any_goals {linarith}
  all_goals {sorry}

end triangle_sides_consecutive_obtuse_l55_55888


namespace shift_sine_even_l55_55136

theorem shift_sine_even (m : ℝ) (h : m = π / 12) :
  (∀ x : ℝ, sin (2 * (x + m) + π / 3) = sin (- (2 * (x + m) + π / 3))) :=
by
  sorry

end shift_sine_even_l55_55136


namespace mean_median_mode_ineq_l55_55472

def mean (s : List ℝ) : ℝ := s.sum / s.length

def median (s : List ℝ) : ℝ := 
  if s.length % 2 = 1 then s.nth_le (s.length / 2) (by decide)
  else (s.nth_le (s.length / 2 - 1) (by decide) + s.nth_le (s.length / 2) (by decide)) / 2

def mode (s : List ℝ) : ℝ :=
  s.group_by id (· = ·).max_by fun l => l.length head!

theorem mean_median_mode_ineq :
  let data := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12].map (· : ℝ) |> List.sort (· ≤ ·)
  let a := mean data
  let b := median data
  let c := mode data
  a < b ∧ b < c := by
  sorry

end mean_median_mode_ineq_l55_55472


namespace vector_magnitude_correct_l55_55045

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55045


namespace find_log_base_l55_55223

theorem find_log_base (b : ℝ) (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y = log b x)
  (h3 : y = 3 * log b (x + 8))
  (h4 : y + 8 = 4 * log b (x + 8))
  (h5 : ∀ (v : ℝ), v^4 = 64): -- side length of the square is 64
  b = 4 :=
by
  sorry

end find_log_base_l55_55223


namespace vector_magnitude_correct_l55_55042

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55042


namespace max_knights_on_chessboard_l55_55625

theorem max_knights_on_chessboard : 
  ∃ (n : ℕ), n = 60 ∧ ∀ (knights : ℕ → ℕ → Prop), 
  (∀ i j, knights i j → i < 8 ∧ j < 8 ∧ 
   (∑ di in [-2, -1, 1, 2], ∑ dj in [-2, -1, 1, 2], 
    if (di ∈ [-2, 2] ∧ dj ∈ [-1, 1]) ∨ (di ∈ [-1, 1] ∧ dj ∈ [-2, 2]) 
        then if knights (i+di) (j+dj) then 1 else 0 else 0) ≤ 7) 
  → n ≤ 60 := 
by 
  sorry

end max_knights_on_chessboard_l55_55625


namespace new_rectangle_area_l55_55568

theorem new_rectangle_area (L W : ℝ) (h : L * W = 432) : 
  let L' := 0.85 * L,
      W' := 0.80 * W,
      A' := L' * W' 
  in A' = 294 :=
by
  let L' := 0.85 * L
  let W' := 0.80 * W
  let A' := L' * W'
  sorry

end new_rectangle_area_l55_55568


namespace math_problem_l55_55752

noncomputable def numerator : ℤ := ∏ k in finset.range 16 \ {0}, (13 : ℤ) + k
noncomputable def denominator : ℤ := ∏ k in finset.range 14 \ {0}, (15 : ℤ) + k

theorem math_problem :
  (numerator : ℚ) / (denominator : ℚ) = 1 := by
  sorry

end math_problem_l55_55752


namespace determine_pairwise_sums_l55_55201

-- Definitions based on conditions
def is_possible_to_determine (x : Fin 64 → ℝ) : Prop :=
  ∀ (i j k l : Fin 64), i ≠ j ∧ k ≠ l ∧ 0 < x i ∧ 0 < x j ∧ 0 < x k ∧ 0 < x l →
  (x i * x j * x k + x i * x j * x l > x i * x k * x l + x j * x k * x l)

theorem determine_pairwise_sums (x : Fin 64 → ℝ) (h_distinct : ∀ (i j : Fin 64), i ≠ j → x i ≠ x j)
  (h_positive : ∀ (i : Fin 64), 0 < x i) : is_possible_to_determine x :=
by
  intros i j k l h_ineq h_pos
  sorry

end determine_pairwise_sums_l55_55201


namespace f_2010_eq_8_l55_55195

def sum_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0

def f (n : ℕ) : ℕ :=
  sum_digits (n * n + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := f (f_seq k n)

theorem f_2010_eq_8 : f_seq 2010 17 = 8 :=
  sorry

end f_2010_eq_8_l55_55195


namespace probability_gte_one_l55_55137

open ProbabilityTheory

variable (σ : ℝ) (ξ : MeasureTheory.ProbabilityTheory.RandomVariable ℝ)
          (h1: MeasureTheory.ProbabilityTheory.normal ξ (-1) σ^2) 
          (h2 : MeasureTheory.ProbaTheory.GT (-3 ≤' ξ ≤' -1) 0.4)

theorem probability_gte_one :
    MeasureTheory.ProbabilityTheory.GE ξ 1 = 0.1 :=
sory

end probability_gte_one_l55_55137


namespace sum_of_hundreds_and_ones_in_third_smallest_l55_55365

-- Define a predicate to check if a number is a three-digit number.
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Extracts the hundreds, tens, and ones place digits from a three-digit number.
def digits_of_three_digit (n : ℕ) (h : is_three_digit n) : ℕ × ℕ × ℕ :=
  let h_digit := n / 100 in
  let t_digit := (n / 10) % 10 in
  let o_digit := n % 10 in
  (h_digit, t_digit, o_digit)

-- Define a predicate to check if a number is formed by three different digits.
def is_three_distinct_digit_number (n : ℕ) : Prop :=
  let (h, t, o) := digits_of_three_digit n (by sorry) in
  h ≠ t ∧ t ≠ o ∧ h ≠ o

-- Define a predicate to check if a valid number based on the given conditions.
def is_valid_number (n : ℕ) : Prop :=
  is_three_digit n ∧
  let (_, t, _) := digits_of_three_digit n (by sorry) in
  t = 1 ∧
  is_three_distinct_digit_number n

-- Obtain sorted list of valid three-digit numbers that satisfy is_valid_number.
noncomputable def sorted_valid_numbers : List ℕ := 
  (List.range 1000).filter is_valid_number |>.qsort (· < ·)

noncomputable def third_smallest_with_tens_one : ℕ :=
  sorted_valid_numbers.nth ![0, 1, 2].head' -- get the third element

theorem sum_of_hundreds_and_ones_in_third_smallest :
  let (third_smallest, _) := third_smallest_with_tens_one in
  let (h, _, o) := digits_of_three_digit third_smallest (by sorry) in
  h + o = 4 :=
by sorry

end sum_of_hundreds_and_ones_in_third_smallest_l55_55365


namespace range_of_x_l55_55413

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a > 0 ∧ (∀ x, f(x) = a*x^2 + b*x + c)

noncomputable def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(4 - x)

theorem range_of_x (f : ℝ → ℝ) (h1 : quadratic_function f) (h2 : symmetric_about_two f) :
  ∀ x, f(1 - 2*x^2) < f(1 + 2*x - x^2) → x ∈ set.Ioo (-2 : ℝ) 0 :=
by
  sorry

end range_of_x_l55_55413


namespace minimum_photos_l55_55816

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55816


namespace manolo_makes_45_masks_in_four_hours_l55_55785

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end manolo_makes_45_masks_in_four_hours_l55_55785


namespace factorial_as_product_of_fib_l55_55773

def fibonacci : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem factorial_as_product_of_fib (n : ℕ) :
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6) ↔
  ∃ k m : ℕ, nat.factorial n = fibonacci k * fibonacci m :=
by {
  sorry -- proof is to be filled in
}

end factorial_as_product_of_fib_l55_55773


namespace burglar_total_sentence_l55_55956

-- Given conditions
def value_of_goods_stolen : ℝ := 40000
def base_sentence_per_thousand_stolen : ℝ := 1 / 5000
def third_offense_increase : ℝ := 0.25
def resisting_arrest_addition : ℕ := 2

-- Theorem to prove the total sentence
theorem burglar_total_sentence :
  let base_sentence := base_sentence_per_thousand_stolen * value_of_goods_stolen
  let increased_sentence := base_sentence * (1 + third_offense_increase)
  let total_sentence := increased_sentence + resisting_arrest_addition
  total_sentence = 12 :=
by 
  sorry -- Proof steps are skipped

end burglar_total_sentence_l55_55956


namespace sum_curvature_angles_equals_4pi_l55_55356

-- Definition: Curvature angle at a vertex of a convex polyhedron
def curvature_angle (polyhedron : convex_polyhedron) (v : vertex) : ℝ :=
  2 * Real.pi - ∑ face in (faces_meeting_at_vertex polyhedron v), interior_angle face v

-- Theorem: Sum of curvature angles at all vertices of a convex polyhedron
theorem sum_curvature_angles_equals_4pi (P: convex_polyhedron) :
  ∑ v in (vertices P), curvature_angle P v = 4 * Real.pi :=
  sorry

end sum_curvature_angles_equals_4pi_l55_55356


namespace trace_bags_weight_l55_55264

theorem trace_bags_weight :
  ∀ (g1 g2 t1 t2 t3 t4 t5 : ℕ),
    g1 = 3 →
    g2 = 7 →
    (g1 + g2) = (t1 + t2 + t3 + t4 + t5) →
    (t1 = t2 ∧ t2 = t3 ∧ t3 = t4 ∧ t4 = t5) →
    t1 = 2 :=
by
  intros g1 g2 t1 t2 t3 t4 t5 hg1 hg2 hsum hsame
  sorry

end trace_bags_weight_l55_55264


namespace number_of_solutions_l55_55354

theorem number_of_solutions : 
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + x * z = 255 ∧ x * z - y * z = 224) ∧
  (finset.filter (λ t : ℕ × ℕ × ℕ, 
    let (x, y, z) := t in 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + x * z = 255 ∧ x * z - y * z = 224)
    (finset.range 256).product (finset.range 32).product (finset.range 32)
  ).card = 2 := 
sorry

end number_of_solutions_l55_55354


namespace log_sum_l55_55772

theorem log_sum : log 50 / log 10 + log 20 / log 10 + log 4 / log 10 = 3.60206 := by
  have log_id : ∀ (x y z : ℝ), log x / log 10 + log y / log 10 + log z / log 10 = log (x * y * z) / log 10 :=
    λ x y z, by rw [log_mul x y, log_mul (x * y) z]; ring
  have log_4_approx : log 4 / log 10 = 0.60206 := sorry
  calc
    log 50 / log 10 + log 20 / log 10 + log 4 / log 10
        = log (50 * 20 * 4) / log 10       : log_id 50 20 4
    ... = log 4000 / log 10               : by norm_num
    ... = log (10 ^ 3 * 10 ^ 0.60206) / log 10 : by rw [(show 4000 = 10 ^ 3 * 4, by norm_num)]
    ... = log (10 ^ (3 + 0.60206)) / log 10  : by rw [log_mul (10 ^ 3) (10 ^ 0.60206)]
    ... = 3.60206                         : by rw [log_pow, log_pow 10 (3 + 0.60206)]; sorry

end log_sum_l55_55772


namespace ratio_pure_imaginary_l55_55961

theorem ratio_pure_imaginary (z1 z2 : ℂ) (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h : ∥z1 + z2∥ = ∥z1 - z2∥) : 
  ∃ k : ℝ, z1 / z2 = complex.I * k := 
sorry

end ratio_pure_imaginary_l55_55961


namespace tennis_ball_price_l55_55130

theorem tennis_ball_price (x y : ℝ) 
  (h₁ : 2 * x + 7 * y = 220)
  (h₂ : x = y + 83) : 
  y = 6 := 
by 
  sorry

end tennis_ball_price_l55_55130


namespace min_union_card_l55_55550

-- Definitions for conditions
def C : Set α := sorry
def D : Set α := sorry
def card_C : ℕ := 30
def card_D : ℕ := 25

-- Statement of the problem
theorem min_union_card (α : Type) (C D : Set α) (card_C card_D : ℕ) (hC : C.card = card_C) (hD : D.card = card_D) :
  ∃ n, n = (card_C + card_D - C ∩ D .card) ∧ n ≤ card_C :=
sorry

end min_union_card_l55_55550


namespace min_photos_for_condition_l55_55809

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55809


namespace real_solution_interval_l55_55234

-- Define the function and the conditions
noncomputable def f : ℝ → ℝ := sorry
axiom f_monotonic : monotone_on f (set.Ioi 0)
axiom f_condition : ∀ x : ℝ, 0 < x → f (f x - real.log x) = real.e + 1

-- Define the function h(x)
def h (x : ℝ) := real.log x - 1 / x

-- Prove the interval where the solution lies
theorem real_solution_interval :
  ∃ x ∈ set.Ioo (1 : ℝ) real.e, f x - (deriv f) x = real.e :=
sorry

end real_solution_interval_l55_55234


namespace variance_of_3_ξ_minus_5_l55_55586

noncomputable def variance_of_transformed_binomial (n p a b : ℝ) : ℝ :=
  let ξ : binomial_var := binomial_var n p
  let Dξ := n * p * (1 - p)
  a^2 * Dξ

theorem variance_of_3_ξ_minus_5 :
  variance_of_transformed_binomial 100 0.3 3 (-5) = 189 := 
by 
  sorry

end variance_of_3_ξ_minus_5_l55_55586


namespace helga_shoes_l55_55875

theorem helga_shoes :
  ∃ (S : ℕ), 7 + S + 0 + 2 * (7 + S) = 48 ∧ (S - 7 = 2) :=
by
  sorry

end helga_shoes_l55_55875


namespace correct_propositions_l55_55259

theorem correct_propositions :
  let proposition1 := (∀ A B C : ℝ, C = (A + B) / 2 → C = (A + B) / 2)
  let proposition2 := (∀ a : ℝ, a - |a| = 0 → a ≥ 0)
  let proposition3 := false
  let proposition4 := (∀ a b : ℝ, |a| = |b| → a = -b)
  let proposition5 := (∀ a : ℝ, -a < 0)
  (cond1 : proposition1 = false) →
  (cond2 : proposition2 = false) →
  (cond3 : proposition3 = false) →
  (cond4 : proposition4 = true) →
  (cond5 : proposition5 = false) →
  1 = 1 :=
by
  intros
  sorry

end correct_propositions_l55_55259


namespace ab_is_4_l55_55457

noncomputable def ab_value (a b : ℝ) : ℝ :=
  8 / (0.5 * (8 / a) * (8 / b))

theorem ab_is_4 (a b : ℝ) (ha : a > 0) (hb : b > 0) (area_condition : ab_value a b = 8) : a * b = 4 :=
  by
  sorry

end ab_is_4_l55_55457


namespace fraction_greater_than_seventy_fifth_percentile_l55_55157

def seventy_fifth_percentile (l : List ℚ) (k : ℚ := 75) : ℚ :=
  let N := l.length
  let pos := (k / 100) * (N + 1)
  if pos % 1 = 0 then l.get? (pos.floor' - 1).toNat
  else (l.get? (pos.floor' - 1).toNat + l.get? pos.ceil' .toNat) / 2

def fraction_greater_than (lst : List ℚ) (percentile_value : ℚ) : ℚ :=
  let above := lst.filter (fun x => x > percentile_value)
  above.length / lst.length

theorem fraction_greater_than_seventy_fifth_percentile :
  let l := [-7, -3.5, -2, 0, 1.5, 3, 3, 3, 4, 4.5, 5, 5, 6, 7, 7, 7, 7, 8, 8, 9, 11, 12, 15, 15, 15, 18, 19, 20, 21, 21, 22, 25, 28, 32, 36, 38, 40, 43, 49, 58, 67] 
  in fraction_greater_than l (seventy_fifth_percentile l) = 11 / 41 := sorry

end fraction_greater_than_seventy_fifth_percentile_l55_55157


namespace probability_empty_bag_six_pairs_l55_55321

theorem probability_empty_bag_six_pairs
    (p q : ℕ)
    (hpq : Nat.coprime p q)
    (prob : ∀ (n : ℕ), n = 6 →
        (finset.card finset.universe ^ 3 * 2^(6 - n) : ℚ) /
            (finset.card finset.universe ^ 3) = p / q) :
    p + q = 394 := 
sorry

end probability_empty_bag_six_pairs_l55_55321


namespace student_travel_by_car_l55_55159

noncomputable def D := 30.000000000000007
def Distance_by_foot := (1 / 5) * D
def Distance_by_bus := (2 / 3) * D
def Distance_by_car := D - (Distance_by_foot + Distance_by_bus)

theorem student_travel_by_car :
  Distance_by_car = 4 := by
  sorry

end student_travel_by_car_l55_55159


namespace Cantor_set_compact_perfect_nowhere_dense_l55_55551

open Set Filter TopologicalSpace

def Cantor_set := ⋂ (n : ℕ), ⋃ (s : Finset (Fin n)), (⋂ i ∈ s, Icc ((i + 1 : ℝ)/(3 * (n + 1))) ((i + 2 : ℝ)/(3 * (n + 1))))

theorem Cantor_set_compact_perfect_nowhere_dense :
  isCompact Cantor_set ∧
  (∀ x ∈ Cantor_set, ∃ y ≠ x, y ∈ Cantor_set) ∧
  ∀ U, is_open U -> ¬(U ⊆ Cantor_set) :=
by
  sorry

end Cantor_set_compact_perfect_nowhere_dense_l55_55551


namespace problem_statement_l55_55911

open Function

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- F is the midpoint of A and B
def F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the slope of the line passing through points C and F
def slope (P₁ P₂ : ℝ × ℝ) : ℝ :=
if P₁.1 = P₂.1 then 0 else (P₂.2 - P₁.2) / (P₂.1 - P₁.1)

-- Define the y-intercept of the line given its slope and one point on the line
def y_intercept (P : ℝ × ℝ) (m : ℝ) : ℝ :=
P.2 - m * P.1

-- Define the sum of the slope and y-intercept of the line passing through points C and F
def sum_slope_y_intercept (P₁ P₂ : ℝ × ℝ) : ℝ :=
let m := slope P₁ P₂ in
m + y_intercept P₁ m

-- Lean theorem statement: the sum of the slope and y-intercept of the line
-- passing through points C and F is 18/5
theorem problem_statement : sum_slope_y_intercept C F = 18 / 5 :=
by
  sorry

end problem_statement_l55_55911


namespace derivative_f_l55_55571

noncomputable def f (x : ℝ) : ℝ := x - sin x

theorem derivative_f : deriv f = λ x, 1 - cos x := by
  sorry

end derivative_f_l55_55571


namespace perpendicular_lines_a_value_l55_55863

theorem perpendicular_lines_a_value (a : ℝ) :
  (let l1 := λ x y, a * x + 3 * y - 1 = 0 in
   let l2 := λ x y, 2 * x + (a^2 - a) * y + 3 = 0 in
   (∀ x y : ℝ, l1 x y ∧ l2 x y → x * y ≠ 0) → a = 0) :=
sorry

end perpendicular_lines_a_value_l55_55863


namespace quotient_multiple_of_y_l55_55631

theorem quotient_multiple_of_y (x y m : ℤ) (h1 : x = 11 * y + 4) (h2 : 2 * x = 8 * m * y + 3) (h3 : 13 * y - x = 1) : m = 3 :=
by
  sorry

end quotient_multiple_of_y_l55_55631


namespace new_average_marks_l55_55295

theorem new_average_marks (n : ℕ) (average : ℕ) (h_n : n = 10) (h_average : average = 80) :
  let total_marks := average * n in
  let new_total_marks := total_marks * 2 in
  let new_average := new_total_marks / n in
  new_average = 160 :=
by
  sorry

end new_average_marks_l55_55295


namespace problem_proof_l55_55160

noncomputable def length_of_AD (a b c : ℚ) (AC : ℚ) (BD_is_bisector : ∀ (AD DC : ℚ), AD + DC = AC → (AD / DC = b / c)) : ℚ :=
  let DC : ℚ := 45 / 8 in
  let AD : ℚ := 5 / 3 * DC in
  AD

theorem problem_proof (a b c : ℚ) (AC : ℚ) (BD_is_bisector : ∀ (AD DC : ℚ), AD + DC = AC → (AD / DC = b / c))
  (h_ratio : a/b = 3/4 ∧ b/c = 4/5) (h_length : AC = 15) : length_of_AD a b c AC BD_is_bisector = 75 / 8 :=
by
  sorry

end problem_proof_l55_55160


namespace tan_double_angle_l55_55453

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1/2) : tan (2*α) = 3/4 := by
  sorry

end tan_double_angle_l55_55453


namespace sum_first_2n_terms_l55_55446

open Classical

noncomputable def seq_a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := 3 * n - seq_a n + 1

theorem sum_first_2n_terms (n : ℕ) (h : n > 0) :
  (Finset.range (2 * n)).sum (λ i, seq_a (i + 1)) = (3 * n^2 + 3 * n) / 2 :=
sorry

end sum_first_2n_terms_l55_55446


namespace box_draw_max_is_3_l55_55255

theorem box_draw_max_is_3 :
  let outcomes := ({1, 2, 3} : Finset ℕ) in
  let all_draws := outcomes.product (outcomes.product outcomes) in
  (all_draws.filter (λ draw, draw.1 = 3 ∨ draw.2.1 = 3 ∨ draw.2.2 = 3)).card = 19 :=
by
  sorry

end box_draw_max_is_3_l55_55255


namespace cost_fly_D_to_E_l55_55922

-- Definitions for the given conditions
def distance_DE : ℕ := 4750
def cost_per_km_plane : ℝ := 0.12
def booking_fee_plane : ℝ := 150

-- The proof statement about the total cost
theorem cost_fly_D_to_E : (distance_DE * cost_per_km_plane + booking_fee_plane = 720) :=
by sorry

end cost_fly_D_to_E_l55_55922


namespace equilateral_triangle_side_length_l55_55666

theorem equilateral_triangle_side_length
  (radius : ℝ) (OD : ℝ) (side_length : ℝ)
  (equilateral_triangle : EquilateralTriangle)
  (circle_center : Point) (O : Point)
  (outside_triangle : O ∉ Triangle.DEF) :
  (radius = 9) → (OD = 6) → equilateral_triangle.DEF.side_length = 3 * Real.sqrt 13 :=
by
  sorry

end equilateral_triangle_side_length_l55_55666


namespace find_angle_C_max_f_l55_55149

noncomputable def condition1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ) (a b c : ℝ),
  A + B + C = π ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a = sin A ∧ b = sin B ∧ c = sin C

noncomputable def condition2 (A B C : ℝ) (a b c : ℝ) : Prop :=
  (sin B) / (sin A + sin C) = (c + b - a) / (c + b)

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
    (h1 : condition1 A B C a b c)
    (h2 : condition2 A B C a b c) 
    : C = π / 3 := 
  sorry

noncomputable def f (A : ℝ) : ℝ := 
  (-2 * cos (2 * A)) / (1 + tan A) + 1

theorem max_f (A : ℝ)
    (h : π / 6 < A ∧ A < π / 2) 
    : ∃ (A : ℝ), f A = sqrt 2 := 
  sorry

end find_angle_C_max_f_l55_55149


namespace num_of_lists_is_correct_l55_55302

theorem num_of_lists_is_correct :
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  total_lists = 50625 :=
by
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  show total_lists = 50625
  sorry

end num_of_lists_is_correct_l55_55302


namespace sum_cos_sin_roots_correct_l55_55218

noncomputable def proof_sum_cos_sin_roots : ℝ :=
  ∑ x in {y : ℝ | cos (2 * y) + cos (6 * y) + 2 * sin y ^ 2 = 1 ∧ (5 * Real.pi / 6) ≤ y ∧ y ≤ Real.pi}, x

theorem sum_cos_sin_roots_correct :
  abs (proof_sum_cos_sin_roots - 2.88) < 0.01 :=
sorry

end sum_cos_sin_roots_correct_l55_55218


namespace expression_value_l55_55517

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x + y + z = 0) (h2 : xy + xz + yz ≠ 0) :
  (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)^2) = 3 / (x^2 + xy + y^2)^2 :=
by
  sorry

end expression_value_l55_55517


namespace vec_magnitude_is_five_l55_55048

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55048


namespace largest_difference_l55_55280

theorem largest_difference : ∃ (a b : ℕ), 
  (∃ (d1 d2 d3 d4 d5 : ℕ), {d1, d2, d3, d4, d5} = {1, 3, 5, 7, 8} ∧ 
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ 
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ 
  d3 ≠ d4 ∧ d3 ≠ d5 ∧ 
  d4 ≠ d5 ∧ 
  a = 100 * d1 + 10 * d2 + d3 ∧ 
  b = 10 * d4 + d5) ∧ 
  a - b = 862 :=
sorry

end largest_difference_l55_55280


namespace ellipse_center_sum_l55_55576

theorem ellipse_center_sum
  (h k a b : ℤ)
  (h_eq : h = 3)
  (k_eq : k = -5)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  rw [h_eq, k_eq, a_eq, b_eq]
  norm_num
  exact eq.refl 9

end ellipse_center_sum_l55_55576


namespace max_five_negative_integers_l55_55131

theorem max_five_negative_integers (a b c d e f : ℤ) (w : ℕ) (h1 : ab + cdef < 0) (h2 : w = 5) : 
  ∃ (neg_count : ℕ), neg_count = 5 ∧ ∀ x, List.mem x [a, b, c, d, e, f] → x < 0 := 
sorry

end max_five_negative_integers_l55_55131


namespace tangent_to_circle_l55_55697

theorem tangent_to_circle
  (Γ : Type*) [metric_space Γ]
  (A B C D E F G H : point Γ)
  (is_circle : circle Γ)
  (A_on_Γ : A ∈ Γ)
  (B_on_Γ : B ∈ Γ)
  (C_on_Γ : C ∈ Γ)
  (tangent_at_B : tangent_line Γ B (line_through D B))
  (tangent_at_C : tangent_line Γ C (line_through D C))
  (D_tangent_intersection : D ∈ (line_through B C))
  (E_intersection : E ∈ (line_intersect (line_through A B) (line_through C D)))
  (F_intersection : F ∈ (line_intersect (line_through A C) (line_through B D)))
  (AD_line : line_through A D = line_intersect (line_through E F))
  (GC_intersect_Γ : GC ∈ Γ)
  (other_point_H : H ≠ C)
  (H_on_Γ : H ∈ Γ) : 
  tangent_line Γ F (line_through F H) :=
sorry

end tangent_to_circle_l55_55697


namespace maximum_value_f_l55_55564

noncomputable def expected_number_of_games (p : ℝ) := 6 * p^4 - 12 * p^3 + 3 * p^2 + 3 * p + 3

theorem maximum_value_f (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  ∀ p∈Icc (0: ℝ) (1: ℝ), expected_number_of_games p ≤ 33 / 8 :=
begin
  sorry
end

end maximum_value_f_l55_55564


namespace one_fourth_in_one_eighth_l55_55124

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l55_55124


namespace intersection_A_B_l55_55158

def A : Set ℝ := { x | x * Real.sqrt (x^2 - 4) ≥ 0 }
def B : Set ℝ := { x | |x - 1| + |x + 1| ≥ 2 }

theorem intersection_A_B : (A ∩ B) = ({-2} ∪ Set.Ici 2) :=
by
  sorry

end intersection_A_B_l55_55158


namespace convex_ineq_rational_convex_ineq_continuous_l55_55521

variable {α : Type*} [LinearOrderedField α] {β : Type*} [OrderedAddCommGroup β] [Module α β]
variables {f : β → α} {x1 x2 : β} {p q : α}

-- Part (a): rational p and q
theorem convex_ineq_rational (hf : ∀ (x y : β) (a b : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1),
  f(a • x + b • y) ≤ a * f x + b * f y)
  (hp : ∃ m n : ℕ, p = m / n ∧ q = (n - m) / n) (hpq : p + q = 1)
  (hp0 : 0 < p) (hq0 : 0 < q) : 
  f(p • x1 + q • x2) ≤ p * f x1 + q * f x2 := 
sorry

-- Part (b): Real p and q and f is continuous
theorem convex_ineq_continuous (hf : ∀ (x y : β) (a b : α) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1),
  f(a • x + b • y) ≤ a * f x + b * f y)
  (hf_cont : ∀ (ε : α), 0 < ε → ∃ δ > 0, ∀ {x' : β} (hx' : ∥ x' - x1 ∥ < δ), ∥ f x' - f x1 ∥ < ε)
  (hpq : p + q = 1)
  (hp0 : 0 ≤ p) (hq0 : 0 ≤ q) : 
  f(p • x1 + q • x2) ≤ p * f x1 + q * f x2 := 
sorry

end convex_ineq_rational_convex_ineq_continuous_l55_55521


namespace sin_double_angle_value_l55_55405

theorem sin_double_angle_value 
  (h1 : Real.pi / 2 < α ∧ α < β ∧ β < 3 * Real.pi / 4)
  (h2 : Real.cos (α - β) = 12 / 13)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.sin (2 * α) = -16 / 65 :=
by
  sorry

end sin_double_angle_value_l55_55405


namespace angle_BAT_eq_angle_CAT_l55_55904

theorem angle_BAT_eq_angle_CAT 
  (ABC : Triangle)
  (h_iso : ABC.isIsosceles AB AC)
  (P : Point)
  (hBcP : collinear {B, C, P})
  (X Y : Point)
  (hX_on_AB : onLine X AB)
  (hY_on_AC : onLine Y AC)
  (hPX_parallel_AC : parallel PX AC)
  (hPY_parallel_AB : parallel PY AB)
  (T : Point)
  (hT_on_circumcircle : onCircumcircle T ABC)
  (hPT_perp_XY : perpendicular PT XY)
  : angle BAT = angle CAT :=
sorry

end angle_BAT_eq_angle_CAT_l55_55904


namespace vector_magnitude_difference_l55_55015

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55015


namespace quadrilateral_area_l55_55973

noncomputable def rectangle_ABCD := 
  let A := (0:ℝ, 8:ℝ)
  let B := (11:ℝ, 8:ℝ)
  let C := (11:ℝ, 0:ℝ)
  let D := (0:ℝ, 0:ℝ)
  (A, B, C, D)

noncomputable def points_EF := 
  let E := (5:ℝ, 4:ℝ)
  let F := (6:ℝ, 4:ℝ)
  (E, F)

def area_quadrilateral (E F B : ℝ × ℝ) : ℝ :=
  1/2 * ((B.1 - E.1 + E.1 - F.1) * 4)

theorem quadrilateral_area :
  let (A, B, C, D) := rectangle_ABCD in
  let (E, F) := points_EF in
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (D.1 - E.1)^2 + (D.2 - E.2)^2 →
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = (C.1 - F.1)^2 + (C.2 - F.2)^2 →
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 →
  area_quadrilateral E F B = 32 := by
  intros
  have a := A; have b := B; have e := E; have f := F
  have ablen := 11 -- AB
  have bclen := 8  -- BC
  sorry

end quadrilateral_area_l55_55973


namespace no_solution_implies_a_eq_one_l55_55138

theorem no_solution_implies_a_eq_one (a : ℝ) : 
  ¬(∃ x y : ℝ, a * x + y = 1 ∧ x + y = 2) → a = 1 :=
by
  intro h
  sorry

end no_solution_implies_a_eq_one_l55_55138


namespace find_line_equation_l55_55387

-- Define the conditions and the problem
theorem find_line_equation :
  ∃ m, (2:ℤ) * 2 + 5 - m = 0 ∧ m = 9 :=
by
  -- We denote the lines and their intersection point using the conditions directly
  let line1 := λ (x : ℤ), (2:ℤ) * x + 1
  let line2 := λ (x : ℤ), (3:ℤ) * x - 1
  let intersection_point := (2, 5) -- intersection of line1 and line2

  -- Line l has form 2x + y - m = 0 and is parallel to 2x + y - 3 = 0
  existsi (9:ℤ)
  split
  -- Substitute the intersection point into 2x + y - m = 0 and solve
  show (2:ℤ) * 2 + 5 - 9 = 0 by rfl
  show 9 = 9 by rfl

end find_line_equation_l55_55387


namespace double_acute_angle_is_less_than_180_degrees_l55_55420

theorem double_acute_angle_is_less_than_180_degrees (alpha : ℝ) (h : 0 < alpha ∧ alpha < 90) : 2 * alpha < 180 :=
sorry

end double_acute_angle_is_less_than_180_degrees_l55_55420


namespace cost_per_order_of_pakoras_l55_55701

noncomputable def samosa_cost : ℕ := 2
noncomputable def samosa_count : ℕ := 3
noncomputable def mango_lassi_cost : ℕ := 2
noncomputable def pakora_count : ℕ := 4
noncomputable def tip_percentage : ℚ := 0.25
noncomputable def total_cost_with_tax : ℚ := 25

theorem cost_per_order_of_pakoras (P : ℚ)
  (h1 : samosa_cost * samosa_count = 6)
  (h2 : mango_lassi_cost = 2)
  (h3 : 1.25 * (samosa_cost * samosa_count + mango_lassi_cost + pakora_count * P) = total_cost_with_tax) :
  P = 3 :=
by
  -- sorry ⟹ sorry
  sorry

end cost_per_order_of_pakoras_l55_55701


namespace ellipse_parabola_intersection_l55_55180

theorem ellipse_parabola_intersection (a b h k : ℝ) :
  let parabola := (x : ℝ) => x^2
  let ellipse := (x y : ℝ) => (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1
  let points := [(4 : ℝ, 16 : ℝ), (-3 : ℝ, 9 : ℝ), (2 : ℝ, 4 : ℝ)]
  ( ∀ x y, (x, y) ∈ points → parabola x = y ∧ ellipse x y) →
  ∃ x4 y4, parabola x4 = y4 ∧ ellipse x4 y4 ∧
  let xs := [4, -3, 2, x4]
  (∑ i in xs, i^2) = 38 := 
by
  sorry

end ellipse_parabola_intersection_l55_55180


namespace max_value_AD_times_BD_l55_55493

theorem max_value_AD_times_BD {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (BC BD : ℝ) (AB : ℝ) (AC : ℝ)
  (h1 : BC = 3 * BD)
  (h2 : AB = 1)
  (h3 : AC = 2) :
  ∃ a : ℝ, a = BD ∧ (a * sqrt (2 - 2 * a^2) ≤ a * sqrt (1 - a^2) * sqrt 2 * ∧  (a = sqrt(2) / 2))
  → ∃ (AD : ℝ), (AD * BD) = (sqrt 2 / 2) := 
sorry

end max_value_AD_times_BD_l55_55493


namespace Baez_final_marbles_l55_55706

theorem Baez_final_marbles :
  ∀ (initial_marbles : ℕ) (loss_percent : ℝ) (friend_multiplier : ℕ),
  initial_marbles = 25 → 
  loss_percent = 0.20 → 
  friend_multiplier = 2 → 
  let lost_marbles := (initial_marbles : ℝ) * loss_percent in
  let remaining_marbles := initial_marbles - (lost_marbles.to_nat) in
  let friend_gift := remaining_marbles * friend_multiplier in
  remaining_marbles + friend_gift = 60 :=
by
  intros
  sorry

end Baez_final_marbles_l55_55706


namespace max_power_sum_l55_55712

open Nat

theorem max_power_sum (a b : ℕ) (h_a_pos : a > 0) (h_b_gt_one : b > 1) (h_max : a ^ b < 500 ∧ 
  ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a' ^ b' < 500 → a' ^ b' ≤ a ^ b ) : a + b = 24 :=
sorry

end max_power_sum_l55_55712


namespace vector_magnitude_subtraction_l55_55032

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55032


namespace difference_of_radii_l55_55247

variable (r R : ℝ)
variable (π : ℝ)

axiom pi_pos : 0 < π
axiom r_pos : 0 < r
axiom ratio_of_areas : (π * R^2) / (π * r^2) = 4
axiom smaller_circle_radius : R = 2 * r

theorem difference_of_radii : (R - r) = r :=
by
  have h : R = 2 * r := by assumption
  rw [h]
  rw [sub_eq_add_neg], rw [add_comm], rw [add_neg_self, zero_add]
  simp only [two_mul, eq_self_iff_true, sub_self]
  sorry

end difference_of_radii_l55_55247


namespace total_amount_received_l55_55667

noncomputable def total_books : ℕ := 150
def sold_fraction : ℚ := 2/3
def unsold_books : ℕ := 50
def price_per_book : ℕ := 5

theorem total_amount_received :
  let B := total_books in
  let unsold := unsold_books in
  let price := price_per_book in
  let sold := (sold_fraction * B) in
  unsold = B / 3 → B = 150 ∧ total_amount_received = sold * price :=
by
  sorry

end total_amount_received_l55_55667


namespace baseball_card_second_year_decrease_l55_55660

theorem baseball_card_second_year_decrease (V : ℝ) (original_value : ℝ) (after_first_year_value : ℝ) (after_second_year_value : ℝ) 
    (first_year_decrease_percent : ℝ) (total_decrease_percent : ℝ) 
    (h1 : first_year_decrease_percent = 60) 
    (h2 : total_decrease_percent = 64) 
    (hv1 : after_first_year_value = original_value * (1 - first_year_decrease_percent / 100))
    (hv2 : after_second_year_value = original_value * (1 - total_decrease_percent / 100)) : 
    (percent_decrease_second_year : ℝ) :=
have h3: after_first_year_value = original_value * 0.4, 
    from by rw [←hv1, h1]; ring,
have h4: after_second_year_value = original_value * 0.36, 
    from by rw [←hv2, h2]; ring,
have decrease_second_year = after_first_year_value - after_second_year_value,
    from by rw [h3, h4]; ring,
have percent_decrease_second_year = (decrease_second_year / after_first_year_value) * 100, 
    from by rw [←hv1, h1]; ring,
by rw [percent_decrease_second_year]; exact 10

#eval baseball_card_second_year_decrease 100 100 40 36 60 64

end baseball_card_second_year_decrease_l55_55660


namespace room_length_perimeter_ratio_l55_55682

theorem room_length_perimeter_ratio :
  ∀ (L W : ℕ), L = 19 → W = 11 → (L : ℚ) / (2 * (L + W)) = 19 / 60 := by
  intros L W hL hW
  sorry

end room_length_perimeter_ratio_l55_55682


namespace min_photos_needed_to_ensure_conditions_l55_55802

noncomputable def min_photos (girls boys : ℕ) : ℕ :=
  33

theorem min_photos_needed_to_ensure_conditions (girls boys : ℕ)
  (hgirls : girls = 4) (hboys : boys = 8) :
  min_photos girls boys = 33 := 
sorry

end min_photos_needed_to_ensure_conditions_l55_55802


namespace vector_magnitude_subtraction_l55_55005

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55005


namespace abs_neg_one_third_l55_55567

theorem abs_neg_one_third : abs (-1/3) = 1/3 := by
  sorry

end abs_neg_one_third_l55_55567


namespace y_work_days_l55_55649

variable (W : ℝ)

def x_work_rate : ℝ := W / 21
def y_work_rate : ℝ := W / 15
def x_days_after_y_leave : ℝ := 7.000000000000001

theorem y_work_days (d : ℝ) :
  d * y_work_rate + x_days_after_y_leave * x_work_rate = W → d = 10 :=
by
  sorry

end y_work_days_l55_55649


namespace at_least_one_divisible_by_3_l55_55658

-- Definitions and conditions:
def nums : ℕ → ℕ := -- some function defining the sequence of 99 numbers arranged in a circle, starting from 0-index.

-- Condition 1: 99 natural numbers arranged in a circle
def num_count : ℕ := 99

-- Condition 2: Neighboring numbers satisfy the given difference conditions.
def neighbor_condition (n m : ℕ) := (n = m + 1) ∨ (n = m - 1) ∨ (n = 2 * m) ∨ (m = 2 * n)

-- Proof that at least one of these numbers is divisible by 3
theorem at_least_one_divisible_by_3 :
  (∀ i : ℕ, i < num_count → ∃ j, neighbor_condition (nums i) (nums ((i + 1) % num_count))) →
  (∃ i < num_count, nums i % 3 = 0) :=
sorry

end at_least_one_divisible_by_3_l55_55658


namespace abs_x_plus_3_gt_1_solution_set_l55_55591

theorem abs_x_plus_3_gt_1_solution_set (x : ℝ) :
  |x + 3| > 1 ↔ x ∈ set.Iio (-4) ∪ set.Ioi (-2) := 
sorry

end abs_x_plus_3_gt_1_solution_set_l55_55591


namespace most_frequent_third_number_l55_55742

def is_lottery_condition (e1 e2 e3 e4 e5 : ℕ) : Prop :=
  1 ≤ e1 ∧ e1 < e2 ∧ e2 < e3 ∧ e3 < e4 ∧ e4 < e5 ∧ e5 ≤ 90 ∧ (e1 + e2 = e3)

theorem most_frequent_third_number :
  ∃ h : ℕ, 3 ≤ h ∧ h ≤ 88 ∧ (∀ h', (h' = 31 → ¬ (31 < h')) ∧ 
        ∀ e1 e2 e3 e4 e5, is_lottery_condition e1 e2 e3 e4 e5 → e3 = h) :=
sorry

end most_frequent_third_number_l55_55742


namespace jim_travels_20_percent_of_jill_l55_55171

def john_distance : ℕ := 15
def jill_travels_less : ℕ := 5
def jim_distance : ℕ := 2
def jill_distance : ℕ := john_distance - jill_travels_less

theorem jim_travels_20_percent_of_jill :
  (jim_distance * 100) / jill_distance = 20 := by
  sorry

end jim_travels_20_percent_of_jill_l55_55171


namespace find_b_minus_a_l55_55583

variables (a b : ℝ)

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
(2 * h - x, 2 * k - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
(y, x)

def P : ℝ × ℝ := (a, b)

def Q := rotate_180 a b 1 5
def R := reflect_y_eq_x Q.1 Q.2

theorem find_b_minus_a :
  R = (3, -6) → b - a = -1 :=
sorry

end find_b_minus_a_l55_55583


namespace sum_of_a_and_b_l55_55728

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55728


namespace trajectory_midpoint_l55_55909

/-- Let A and B be two moving points on the circle x^2 + y^2 = 4, and AB = 2. 
    The equation of the trajectory of the midpoint M of the line segment AB is x^2 + y^2 = 3. -/
theorem trajectory_midpoint (A B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A.1^2 + A.2^2 = 4)
    (hB : B.1^2 + B.2^2 = 4)
    (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
    (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
    M.1^2 + M.2^2 = 3 :=
sorry

end trajectory_midpoint_l55_55909


namespace expected_regions_100_points_l55_55972

def expected_number_of_regions (n : ℕ) : ℝ :=
  (n : ℝ) * (n - 3) / 6 + 1

theorem expected_regions_100_points :
  expected_number_of_regions 100 = 4853 / 3 := by
  sorry

end expected_regions_100_points_l55_55972


namespace angle_FMG_l55_55699
  
theorem angle_FMG :
  ∀ (A B C D E F G M : Point) (angle : Point → Point → Point → ℝ),
    IsoscelesRightTriangle A B C 90° →
    IsoscelesRightTriangle A D E 90° →
    Midpoint M B C →
    (dist A B = dist A C) → (dist A B = dist D F) → (dist A B = dist F M) → (dist A B = dist E G) → (dist A B = dist G M) →
    angle F D E = 9° →
    angle G E D = 9° →
    Outside F D E →
    Outside G D E →
    angle F M G = 54°
:= by
  sorry

end angle_FMG_l55_55699


namespace angle_complementary_supplementary_l55_55299

theorem angle_complementary_supplementary (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle1 + angle3 = 180)
  (h3 : angle3 = 125) :
  angle2 = 35 :=
by 
  sorry

end angle_complementary_supplementary_l55_55299


namespace largest_unique_triangles_l55_55756

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
def is_even (n : ℕ) : Prop := n % 2 = 0
def not_congruent (t1 t2 : ℕ × ℕ × ℕ) : Prop := ¬((t1.1 = t2.1 ∧ t1.2 = t2.2 ∧ t1.3 = t2.3) ∨ 
                                                  (t1.1 = t2.1 ∧ t1.2 = t2.3 ∧ t1.3 = t2.2) ∨ 
                                                  (t1.1 = t2.2 ∧ t1.2 = t2.1 ∧ t1.3 = t2.3) ∨ 
                                                  (t1.1 = t2.2 ∧ t1.2 = t2.3 ∧ t1.3 = t2.1) ∨ 
                                                  (t1.1 = t2.3 ∧ t1.2 = t2.1 ∧ t1.3 = t2.2) ∨ 
                                                  (t1.1 = t2.3 ∧ t1.2 = t2.2 ∧ t1.3 = t2.1))
def similar_ratio (a b c d e f : ℕ) : Prop := 
  (a * e = b * d ∧ b * f = a * e) ∨ 
  (a * f = b * d ∧ b * e = a * d) ∨ 
  (a * e = c * d ∧ c * f = a * e) ∨ 
  (a * f = c * d ∧ c * e = a * d) ∨ 
  (b * e = c * d ∧ c * f = b * e) ∨ 
  (b * f = c * d ∧ c * e = b * d)

def not_similar (t1 t2 : ℕ × ℕ × ℕ) : Prop := ¬similar_ratio t1.1 t1.2 t1.3 t2.1 t2.2 t2.3

noncomputable def S' : Finset (ℕ × ℕ × ℕ) := 
  {x ∈ (Finset.range 8).product (Finset.range 8).product (Finset.range 8) |
  x.1.1 ≥ x.1.2 ∧ x.1.2 ≥ x.2 ∧ is_triangle x.1.1 x.1.2 x.2 ∧ 
  is_even (x.1.1 + x.1.2 + x.2)}.filter
  (λ t, (∀ t' ∈ S', not_congruent t t') ∧ (∀ t' ∈ S', not_similar t t'))

theorem largest_unique_triangles : S'.card = 9 :=
sorry

end largest_unique_triangles_l55_55756


namespace min_balls_to_ensure_20_of_one_color_correct_l55_55304

def min_balls_to_ensure_20_of_one_color :
  (35 + 30 + 25 + 15 + 12 + 10 = 35 + 30 + 25 + 15 + 12 + 10) →
  Nat :=
  λ h, 95

theorem min_balls_to_ensure_20_of_one_color_correct (h : 35 + 30 + 25 + 15 + 12 + 10 = 127) :
  min_balls_to_ensure_20_of_one_color h = 95 :=
by
  exact eq.refl 95

end min_balls_to_ensure_20_of_one_color_correct_l55_55304


namespace people_in_circle_l55_55540

theorem people_in_circle (n : ℕ) (h : ∃ k : ℕ, k * 2 + 7 = 18) : n = 22 :=
by
  sorry

end people_in_circle_l55_55540


namespace necessary_but_not_sufficient_condition_l55_55652

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (2 * a > 2 * b) → (lga a > lgb b) → (necessary_condition (2 * a > 2 * b) (lga a > lgb b)) ∧ ¬ (sufficient_condition (2 * a > 2 * b) (lga a > lgb b)) :=
by
  sorry

-- Definitions of necessary and sufficient conditions for completeness
def necessary_condition (P Q : Prop) := Q → P
def sufficient_condition (P Q : Prop) := P → Q

end necessary_but_not_sufficient_condition_l55_55652


namespace relationship_a_b_l55_55935

noncomputable def e : ℝ := Real.exp 1

theorem relationship_a_b
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : e^a + 2 * a = e^b + 3 * b) :
  a > b :=
sorry

end relationship_a_b_l55_55935


namespace relationship_xy_l55_55835

-- The conditions
variables (t x y : ℝ)
hypothesis h1 : t > 0
hypothesis h2 : t ≠ 1
hypothesis hx : x = t^(1 / (t - 1))
hypothesis hy : y = t^(t / (t - 1))

-- The theorem statement
theorem relationship_xy : y^x = x^y :=
by sorry

end relationship_xy_l55_55835


namespace miriam_flowers_total_l55_55199

theorem miriam_flowers_total :
  let monday_flowers := 45
  let tuesday_flowers := 75
  let wednesday_flowers := 35
  let thursday_flowers := 105
  let friday_flowers := 0
  let saturday_flowers := 60
  (monday_flowers + tuesday_flowers + wednesday_flowers + thursday_flowers + friday_flowers + saturday_flowers) = 320 :=
by
  -- Calculations go here but we're using sorry to skip them
  sorry

end miriam_flowers_total_l55_55199


namespace shoe_size_ratio_l55_55923

theorem shoe_size_ratio (J A : ℕ) (hJ : J = 7) (hAJ : A + J = 21) : A / J = 2 :=
by
  -- Skipping the proof
  sorry

end shoe_size_ratio_l55_55923


namespace number_of_bricks_in_wall_l55_55668

noncomputable def rate_one_bricklayer (x : ℕ) : ℚ := x / 8
noncomputable def rate_other_bricklayer (x : ℕ) : ℚ := x / 12
noncomputable def combined_rate_with_efficiency (x : ℕ) : ℚ := (rate_one_bricklayer x + rate_other_bricklayer x - 15)
noncomputable def total_time (x : ℕ) : ℚ := 6 * combined_rate_with_efficiency x

theorem number_of_bricks_in_wall (x : ℕ) : total_time x = x → x = 360 :=
by sorry

end number_of_bricks_in_wall_l55_55668


namespace rectangle_rotation_volume_l55_55681

-- Given definitions: Length and Width of the rectangle
def length : ℝ := 6
def width : ℝ := 3

-- Mathematical problem statement in Lean 4
theorem rectangle_rotation_volume :
  let r := width
  let h := length
  Volume_of_geometric_solid = π * r^2 * h := 
  54 * π :=
Sorry

end rectangle_rotation_volume_l55_55681


namespace min_photos_exists_l55_55796

-- Conditions: Girls and Boys
def girls : ℕ := 4
def boys : ℕ := 8

-- Definition representing the minimum number of photos for the condition
def min_photos : ℕ := 33

theorem min_photos_exists : 
  ∀ (photos : ℕ), 
  (photos ≥ min_photos) →
  (∃ (bb gg bg : ℕ), 
    (bb > 0 ∨ gg > 0 ∨ bg < photos)) :=
begin
  sorry
end

end min_photos_exists_l55_55796


namespace no_such_integers_l55_55278

theorem no_such_integers (x y : ℤ) : ¬ ∃ x y : ℤ, (x^4 + 6) % 13 = y^3 % 13 :=
sorry

end no_such_integers_l55_55278


namespace factorize_x2_minus_2x_plus_1_l55_55380

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l55_55380


namespace safe_security_system_l55_55231

theorem safe_security_system (commission_members : ℕ) 
                            (majority_access : ℕ)
                            (max_inaccess_members : ℕ) 
                            (locks : ℕ)
                            (keys_per_member : ℕ) :
  commission_members = 11 →
  majority_access = 6 →
  max_inaccess_members = 5 →
  locks = (Nat.choose 11 5) →
  keys_per_member = (locks * 6) / 11 →
  locks = 462 ∧ keys_per_member = 252 :=
by
  intros
  sorry

end safe_security_system_l55_55231


namespace amount_paid_for_peaches_l55_55529

noncomputable def cost_of_berries : ℝ := 7.19
noncomputable def change_received : ℝ := 5.98
noncomputable def total_bill : ℝ := 20

theorem amount_paid_for_peaches :
  total_bill - change_received - cost_of_berries = 6.83 :=
by
  sorry

end amount_paid_for_peaches_l55_55529


namespace problem_l55_55415

open Function

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) + a (n - m) = 2 * a n

theorem problem (h_arith : arithmetic_sequence a) (h_eq : a 1 + 3 * a 6 + a 11 = 10) :
  a 5 + a 7 = 4 := 
sorry

end problem_l55_55415


namespace vector_magnitude_l55_55106

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55106


namespace Jakes_height_is_20_l55_55549

-- Define the conditions
def Sara_width : ℤ := 12
def Sara_height : ℤ := 24
def Sara_depth : ℤ := 24
def Jake_width : ℤ := 16
def Jake_depth : ℤ := 18
def volume_difference : ℤ := 1152

-- Volume calculation
def Sara_volume : ℤ := Sara_width * Sara_height * Sara_depth

-- Prove Jake's height is 20 inches
theorem Jakes_height_is_20 :
  ∃ h : ℤ, (Sara_volume - (Jake_width * h * Jake_depth) = volume_difference) ∧ h = 20 :=
by
  sorry

end Jakes_height_is_20_l55_55549


namespace solve_for_x_l55_55990

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l55_55990


namespace vector_magnitude_difference_l55_55091

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55091


namespace inequality_proof_l55_55428

theorem inequality_proof (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 1 < a i) :
  2^(n-1) * ((∏ i in Finset.range n, a i) + 1) > ∏ i in Finset.range n, (1 + a i) :=
sorry

end inequality_proof_l55_55428


namespace sum_when_max_power_less_500_l55_55716

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end sum_when_max_power_less_500_l55_55716


namespace value_of_M_l55_55594

theorem value_of_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 :=
sorry

end value_of_M_l55_55594


namespace min_value_expression_l55_55456

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a + (a/b^2) + b) ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l55_55456


namespace find_m_l55_55462

noncomputable def tangent_line_circle (m : ℝ) : Prop :=
  let line := λ x y, x + y + m in
  let circle := λ x y, x^2 + y^2 = m in
  ∃ x y, line x y = 0 ∧ circle x y ∧ (∀ x' y', circle x' y' → distance (x, y) (x', y') = real.sqrt m)

theorem find_m : ∃ m : ℝ, tangent_line_circle m ∧ m = 2 :=
by
  sorry -- Proof to be completed

end find_m_l55_55462


namespace sum_of_a_and_b_l55_55737

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55737


namespace solve_for_x_l55_55991

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l55_55991


namespace Series_value_l55_55362

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l55_55362


namespace quadrilateral_tangential_l55_55542

variables (P F1 F2 M1 M2 R : Point)
variable (E : Ellipse)
variable (is_external_point : ¬ (∃ P ∈ major_axis E))
variable (intersection1 : segment_intersects_ellipse (P, F1) E M1)
variable (intersection2 : segment_intersects_ellipse (P, F2) E M2)
variable (intersectionR : lines_intersect (line_through M1 F2) (line_through M2 F1) R)

theorem quadrilateral_tangential :
  tangential_quadrilateral P M1 R M2 :=
sorry

end quadrilateral_tangential_l55_55542


namespace OHara_triple_c_l55_55757

theorem OHara_triple_c (a b c : ℕ) (h₁ : a = 49) (h₂ : b = 8) (h₃ : c = 9) : (Nat.sqrt a + Nat.cbrt b = c) :=
by
  sorry

end OHara_triple_c_l55_55757


namespace train_speed_l55_55690

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (time_seconds : ℝ) 
  (speed_kmph : ℝ) 
  (h_train_length : train_length = 165) 
  (h_bridge_length : bridge_length = 660) 
  (h_time_seconds : time_seconds = 32.99736021118311) 
  (h_speed_kmph : speed_kmph = 90)
  : (train_length + bridge_length) / time_seconds * 3.6 ≈ speed_kmph :=
by {
  calc (train_length + bridge_length) / time_seconds * 3.6
      = (165 + 660) / 32.99736021118311 * 3.6 : by rw [h_train_length, h_bridge_length, h_time_seconds]
  ... ≈ 90 : by norm_num,
  exact_mod_cast h_speed_kmph,
}

end train_speed_l55_55690


namespace total_digits_first_2003_even_integers_l55_55275

theorem total_digits_first_2003_even_integers : 
  let even_integers := (List.range' 1 (2003 * 2)).filter (λ n => n % 2 = 0)
  let one_digit_count := List.filter (λ n => n < 10) even_integers |>.length
  let two_digit_count := List.filter (λ n => 10 ≤ n ∧ n < 100) even_integers |>.length
  let three_digit_count := List.filter (λ n => 100 ≤ n ∧ n < 1000) even_integers |>.length
  let four_digit_count := List.filter (λ n => 1000 ≤ n) even_integers |>.length
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4
  total_digits = 7460 :=
by
  sorry

end total_digits_first_2003_even_integers_l55_55275


namespace vector_magnitude_difference_l55_55871

open Real

variables (a b : ℝ) (ha : |a| = 3) (hb : |b| = 4)
variable (θ : ℝ) (hθ : θ = π / 3)

theorem vector_magnitude_difference : 
  |a - b| = sqrt(13) :=
by
  -- Using the provided conditions, we need to prove the required theorem
  sorry

end vector_magnitude_difference_l55_55871


namespace cos_theta_positive_then_first_or_fourth_quadrant_l55_55455

variable {θ : ℝ}

theorem cos_theta_positive_then_first_or_fourth_quadrant (h : cos θ > 0) : 
  (∃ k : ℤ, θ = 2 * k * π + θ ∧ θ > 0 ∧ θ < π) ∨ 
  (∃ k : ℤ, θ = 2 * k * π + θ ∧ θ < 0 ∧ θ > -π) := 
sorry

end cos_theta_positive_then_first_or_fourth_quadrant_l55_55455


namespace angle_bac_is_30_l55_55235

theorem angle_bac_is_30 (A B C H L K : Point)
  (h_right_triangle : right_triangle A B C)
  (h_ch_height : height CH A B C)
  (h_ch_bisects_bl : bisects CH BL K) :
  angle BAC = 30 := 
sorry

end angle_bac_is_30_l55_55235


namespace magnitude_of_a_minus_b_l55_55078

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55078


namespace rectangle_distance_sum_squares_l55_55566

theorem rectangle_distance_sum_squares
  (x0 y0 x1 y1 x2 y2 : ℝ) :
  let A := (x1, y1)
      B := (x1, y2)
      C := (x2, y1)
      D := (x2, y2)
      M := (x0, y0)
  in
  ((x1 - x0)^2 + (y1 - y0)^2) + ((x2 - x0)^2 + (y1 - y0)^2) = 
  ((x1 - x0)^2 + (y2 - y0)^2) + ((x2 - x0)^2 + (y2 - y0)^2) :=
sorry

end rectangle_distance_sum_squares_l55_55566


namespace problem_statement_l55_55645

theorem problem_statement (k : ℕ) (h : 35^k ∣ 1575320897) : 7^k - k^7 = 1 := by
  sorry

end problem_statement_l55_55645


namespace vector_magnitude_subtraction_l55_55071

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55071


namespace david_profit_l55_55683

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l55_55683


namespace find_the_number_l55_55301

noncomputable def certain_number : ℝ :=
  let x := Real.cbrt 42 - 10 in
  x

theorem find_the_number : 
  let x := Real.cbrt 42 - 10 in
  (x + 10)^3 - 2 = 40 := 
by
  sorry -- Proof is skipped

end find_the_number_l55_55301


namespace vector_magnitude_correct_l55_55046

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55046


namespace lauren_earnings_tuesday_l55_55173

def money_from_commercials (num_commercials : ℕ) (rate_per_commercial : ℝ) : ℝ :=
  num_commercials * rate_per_commercial

def money_from_subscriptions (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  num_subscriptions * rate_per_subscription

def total_money (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  money_from_commercials num_commercials rate_per_commercial + money_from_subscriptions num_subscriptions rate_per_subscription

theorem lauren_earnings_tuesday (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) :
  num_commercials = 100 → rate_per_commercial = 0.50 → num_subscriptions = 27 → rate_per_subscription = 1.00 → 
  total_money num_commercials rate_per_commercial num_subscriptions rate_per_subscription = 77 :=
by
  intros h1 h2 h3 h4
  simp [money_from_commercials, money_from_subscriptions, total_money, h1, h2, h3, h4]
  sorry

end lauren_earnings_tuesday_l55_55173


namespace vector_magnitude_subtraction_l55_55007

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  show real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 from
  sorry

end vector_magnitude_subtraction_l55_55007


namespace rate_per_sqm_paving_l55_55239

theorem rate_per_sqm_paving (length width : ℝ) (total_cost : ℝ) (h1 : length = 5.5) (h2 : width = 4) (h3 : total_cost = 20900) : 
  total_cost / (length * width) = 950 :=
by 
  rw [h1, h2, h3]
  exact div_eq_iff (by norm_num : 22 ≠ 0)
  exact_mod_cast_singleton eq_falsify
  sorry

end rate_per_sqm_paving_l55_55239


namespace min_photos_for_condition_l55_55810

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end min_photos_for_condition_l55_55810


namespace simplify_and_rationalize_l55_55983

noncomputable def simplify_expr : ℝ :=
  1 / (1 - (1 / (Real.sqrt 5 - 2)))

theorem simplify_and_rationalize :
  simplify_expr = (1 - Real.sqrt 5) / 4 := by
  sorry

end simplify_and_rationalize_l55_55983


namespace vector_magnitude_correct_l55_55047

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Define the difference vector and its magnitude
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Theorem statement
theorem vector_magnitude_correct : magnitude a_minus_b = 5 := by
  sorry

end vector_magnitude_correct_l55_55047


namespace min_photos_l55_55788

theorem min_photos (G B : ℕ) (G_eq : G = 4) (B_eq : B = 8): 
  ∃ n ≥ 33, ∀ photos : set (set (ℕ × ℕ)), 
  (∀ p ∈ photos, p = (i, j) → i < j ∧ i < G ∧ j < B ∨ i >= G ∧ j < G) →
  ((∃ p ∈ photos, ∀ (i j : ℕ), (i, j) = p → (i < G ∧ j < G) ∨ (i < B ∧ j < B)) ∨ (∃ p1 p2 ∈ photos, p1 = p2)) := 
by {
  sorry
}

end min_photos_l55_55788


namespace problem_one_problem_two_l55_55747

theorem problem_one :
  sqrt 9 - (-2023 : ℤ)^0 + 2⁻¹ = (5 : ℚ) / 2 :=
by sorry

theorem problem_two (a b : ℚ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2 * b)) = 2 / (a + b) :=
by sorry

end problem_one_problem_two_l55_55747


namespace length_of_longest_side_l55_55241

variable (a b c p x l : ℝ)

-- conditions of the original problem
def original_triangle_sides (a b c : ℝ) : Prop := a = 8 ∧ b = 15 ∧ c = 17

def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def similar_triangle_perimeter (a b c p x : ℝ) : Prop := (a * x) + (b * x) + (c * x) = p

-- proof target
theorem length_of_longest_side (h1: original_triangle_sides a b c) 
                               (h2: is_right_triangle a b c) 
                               (h3: similar_triangle_perimeter a b c p x) 
                               (h4: x = 4)
                               (h5: p = 160): (c * x) = 68 := by
  -- to complete the proof
  sorry

end length_of_longest_side_l55_55241


namespace solve_for_x_l55_55994

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l55_55994


namespace fence_planks_l55_55328

theorem fence_planks (N : ℕ) (h : N ∈ {96, 97, 98, 99, 100}) :
  ∃ (x : ℕ), N = 5 * x + 1 :=
by {
  use 19, -- Here, we can use the known value from the solution
  have h' : N = 96,
  { assumption }, -- We assume N to be 96 since 96 is the correct answer
  rw h' at *,
  sorry
}

end fence_planks_l55_55328


namespace probability_event_A_l55_55656

def probability_of_defective : Real := 0.3
def probability_of_all_defective : Real := 0.027
def probability_of_event_A : Real := 0.973

theorem probability_event_A :
  1 - probability_of_all_defective = probability_of_event_A :=
by
  sorry

end probability_event_A_l55_55656


namespace sum_of_a_and_b_l55_55736

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55736


namespace train_carriages_l55_55325

theorem train_carriages (
  (train_speed_kmh : ℕ)
  (carriage_length : ℕ)
  (engine_length : ℕ)
  (bridge_length : ℕ)
  (crossing_time_minutes : ℕ)
  (train_speed_kmh = 60)
  (carriage_length = 60)
  (engine_length = 60)
  (bridge_length = 4500) -- in meters
  (crossing_time_minutes = 6)
  :
  let speed_m_per_s := train_speed_kmh.to_real * 1000 / 3600
  let crossing_time_seconds := crossing_time_minutes * 60
  let total_distance := bridge_length.to_real + engine_length.to_real + n * carriage_length.to_real
  let required_distance := speed_m_per_s * crossing_time_seconds
  in total_distance = required_distance
) : n = 24 := 
sorry

end train_carriages_l55_55325


namespace polygons_inscribed_in_same_circle_l55_55188

theorem polygons_inscribed_in_same_circle {K : Type} [circle K] 
  (M1 M2 : set (polygon inscribed_in K)) 
  (P1 P2 : ℝ) 
  (S1 S2 : ℝ) 
  (longest_side_M1 : ℝ) 
  (shortest_side_M2 : ℝ) 
  (hP1 : ∀ p ∈ M1, perimeter p = P1) 
  (hP2 : ∀ p ∈ M2, perimeter p = P2) 
  (hS1 : ∀ p ∈ M1, area p = S1) 
  (hS2 : ∀ p ∈ M2, area p = S2) 
  (hlm1_lsm2 : longest_side_M1 < shortest_side_M2) :
  P1 > P2 ∧ S1 > S2 := 
sorry

end polygons_inscribed_in_same_circle_l55_55188


namespace percent_v_u_l55_55140

noncomputable def proof_problem (x y z w v u : ℝ) : Prop :=
  (x = 1.30 * y) ∧
  (y = 0.60 * z) ∧
  (w = 1.25 * x^2) ∧
  (v = 0.85 * w^2) ∧
  (u = 1.20 * z^2) →
  v / u = 0.3414

theorem percent_v_u (x y z w v u: ℝ) : proof_problem x y z w v u :=
begin
  sorry
end

end percent_v_u_l55_55140


namespace expected_net_profit_l55_55657

-- Define the conditions
def purchase_price : ℝ := 10
def pass_rate : ℝ := 0.95
def defect_rate : ℝ := 1 - pass_rate
def net_profit_qualified : ℝ := 2
def net_loss_defective : ℝ := -10

-- Define the random variable X
def X : Type := ℝ
noncomputable def E (X : Type) : ℝ := pass_rate * net_profit_qualified + defect_rate * net_loss_defective

-- State the theorem
theorem expected_net_profit : E(X) = 1.4 :=
by
  sorry

end expected_net_profit_l55_55657


namespace g_symmetry_value_h_m_interval_l55_55434

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : ℝ) : ℝ :=
  f x + g x

theorem g_symmetry_value (k : ℤ) : 
  g (k * Real.pi / 2 - Real.pi / 12) = (3 + (-1) ^ k) / 4 :=
by
  sorry

theorem h_m_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc (- Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) ↔ (1 ≤ m ∧ m ≤ 9 / 4) :=
by
  sorry

end g_symmetry_value_h_m_interval_l55_55434


namespace robin_uploaded_pics_from_camera_l55_55213

-- Definitions of the conditions
def pics_from_phone := 35
def albums := 5
def pics_per_album := 8

-- The statement we want to prove
theorem robin_uploaded_pics_from_camera : (albums * pics_per_album) - pics_from_phone = 5 :=
by
  -- Proof goes here
  sorry

end robin_uploaded_pics_from_camera_l55_55213


namespace min_photos_required_l55_55826

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55826


namespace lattice_points_on_curve_l55_55451

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - y^2 = 36

def lattice_point_count : ℕ :=
  (Finset.univ : Finset (ℤ × ℤ)).filter (λ p, is_lattice_point p.1 p.2).card

theorem lattice_points_on_curve : lattice_point_count = 8 :=
sorry

end lattice_points_on_curve_l55_55451


namespace total_number_of_pupils_l55_55605

theorem total_number_of_pupils (P B : Finset α) (hP : P.card = 125) (hB : B.card = 115) (hPB : (P ∩ B).card = 40) :
  (P ∪ B).card = 200 := 
by sorry

end total_number_of_pupils_l55_55605


namespace part1_solution_set_part2_min_value_of_m_l55_55862

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)
def g (x : ℝ) : ℝ := -x

theorem part1_solution_set :
  {x : ℝ | f x > g x} = {x : ℝ | (-3 < x ∧ x < 1) ∨ (x > 3)} :=
begin
  sorry,
end

theorem part2_min_value_of_m :
  ∀ x : ℝ, f x - 2 * x ≤ 2 * g x + 3 :=
begin
  sorry,
end

end part1_solution_set_part2_min_value_of_m_l55_55862


namespace analyze_f_l55_55512

noncomputable def e : ℝ := Real.exp 1

def f (a x : ℝ) : ℝ := (e ^ x - a) / x - a * Real.log x

theorem analyze_f (a : ℝ) (x : ℝ) (h_pos : x > 0) :
  (a = e → ¬∃ c, c > 0 ∧ c < x ∧ deriv (λ x, f a x) c = 0) ∧
  (1 < a ∧ a < e → ∃ c, c > 0 ∧ c < x ∧ f a c = 0) ∧
  (a ≤ 1 → ¬∃ c, c > 0 ∧ c < x ∧ f a c = 0) :=
sorry

end analyze_f_l55_55512


namespace derivative_at_pi_div_3_l55_55861

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_at_pi_div_3 : 
  deriv f (Real.pi / 3) = - (Real.sqrt 3 * Real.pi / 6) :=
by
  sorry

end derivative_at_pi_div_3_l55_55861


namespace exists_pos_sum_figureII_l55_55485

variables {R : Type*} [OrderedRing R]

/-- Representing each cell with a real number on an infinite sheet of graph paper by a function. -/
variable (g : ℤ × ℤ → R)

/-- Figures I and II consist of finite sets of cell vectors. -/
variables (figureI figureII : Finset (ℤ × ℤ))

/-- For any position of the first figure, the sum of the numbers in the cells covered by it is positive. -/
variable (pos_sum_figureI : ∀ (p : ℤ × ℤ), 0 < (figureI.sum (λ v, g (p + v))))

/-- The main proposition we want to prove. -/
theorem exists_pos_sum_figureII : ∃ (p : ℤ × ℤ), 0 < (figureII.sum (λ v, g (p + v))) :=
sorry

end exists_pos_sum_figureII_l55_55485


namespace complement_correct_l55_55867

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the set of real numbers such that -1 ≤ x < 2
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the complement of A in U
def complement_U_A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

-- The proof statement: the complement of A in U is the expected set
theorem complement_correct : (U \ A) = complement_U_A := 
by
  sorry

end complement_correct_l55_55867


namespace unique_array_count_l55_55317

theorem unique_array_count (n m : ℕ) (h_conds : n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :
  ∃! (n m : ℕ), (n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :=
by
  sorry

end unique_array_count_l55_55317


namespace magnitude_of_a_minus_b_l55_55076

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55076


namespace coin_rotations_complete_l55_55217

theorem coin_rotations_complete (n : ℕ) (h_closed_chain : n = 6) :
  (number_of_rotations (rolling_coin : ℕ) (closed_chain : set ℕ)).count = 4 := 
sorry

end coin_rotations_complete_l55_55217


namespace triangle_angle_bisector_projection_length_l55_55916

theorem triangle_angle_bisector_projection_length
  (A B C L K N M : Type)
  [MetricSpace Type]  -- Assume these points exist in some metric space for distances.
  (AB AC BC : ℝ)
  (hAB : dist A B = 125)
  (hAC : dist A C = 130)
  (hBC : dist B C = 120)
  (angle_bisector_A : ∃ L, (L ∈ line_segment B C) ∧ (∠BAL = ∠CAL))
  (angle_bisector_B : ∃ K, (K ∈ line_segment A C) ∧ (∠CBK = ∠ABK))
  (project_C_AL : N ∈ projection C AL)
  (project_C_BK : M ∈ projection C BK)
  :
  dist M N = 62.5 :=
by
  sorry

end triangle_angle_bisector_projection_length_l55_55916


namespace geometric_sequence_common_ratio_l55_55308

theorem geometric_sequence_common_ratio (a1 a2 a3 a4 : ℤ) (h1 : a1 = 10) (h2 : a2 = -20) (h3 : a3 = 40) (h4 : a4 = -80) : 
  ∃ r : ℤ, r = -2 ∧ a2 = a1 * r ∧ a3 = a2 * r ∧ a4 = a3 * r :=
by
  use -2,
  sorry

end geometric_sequence_common_ratio_l55_55308


namespace remainder_of_12_factorial_mod_13_l55_55401

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end remainder_of_12_factorial_mod_13_l55_55401


namespace vector_magnitude_b_one_l55_55448

variables (a b : ℝ^3)
variables (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0)
variables (norm_a : ∥a∥ = 1) (norm_a_plus_b : ∥a + b∥ = 1)
variables (angle_ab : real.angle a b = π * (2/3))

theorem vector_magnitude_b_one : ∥b∥ = 1 :=
by 
  sorry

end vector_magnitude_b_one_l55_55448


namespace sum_of_a_and_b_l55_55730

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55730


namespace average_speed_trip_l55_55646

variable (distance1 distance2 speed1 speed2 totalDistance : ℝ)
variable (time1 time2 totalTime : ℝ)

noncomputable def averageSpeed (d1 d2 s1 s2 : ℝ) : ℝ :=
  let t1 := d1 / s1
  let t2 := d2 / s2
  let totalTime := t1 + t2
  (d1 + d2) / totalTime

theorem average_speed_trip :
  distance1 = 200 ∧ distance2 = 150 ∧ speed1 = 20 ∧ speed2 = 15 ∧ totalDistance = 350 ∧ averageSpeed 200 150 20 15 = 17.5 :=
begin
  sorry
end

end average_speed_trip_l55_55646


namespace polynomial_division_correctness_l55_55392

open Polynomial

noncomputable theory

def dividend : Polynomial ℤ := X^6 - 5 * X^4 + 3 * X^3 - 27 * X^2 + 14 * X - 8
def divisor : Polynomial ℤ := X^2 - 3 * X + 2
def quotient : Polynomial ℤ := X^4 + 3 * X^3 + 2 * X^2 + 3 * X - 3

theorem polynomial_division_correctness :
  (dividend / divisor) = quotient :=
by {
  sorry
}

end polynomial_division_correctness_l55_55392


namespace area_of_triangle_OPQ_l55_55944

variable (O F P Q : Point)
variable (a b : ℝ)

#check Point

noncomputable def parabola_vertex (O F : Point) : Point := sorry
noncomputable def parabola_focus (O F : Point) : Point := sorry
noncomputable def chord_passing_through_focus (F P Q : Point) : List Point := sorry
noncomputable def chord_length (P Q : Point) : ℝ := sorry

theorem area_of_triangle_OPQ (O F P Q : Point) (a b : ℝ) 
  (hOF : dist O F = a) 
  (hPQ : chord_length P Q = b) :
  area (O, P, Q) = a * sqrt(a * b) := sorry

end area_of_triangle_OPQ_l55_55944


namespace rectangle_diagonal_proximity_probability_l55_55978

theorem rectangle_diagonal_proximity_probability :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ 
  (let prob := (5 / 12) in prob = m / n) ∧ 
  m + n = 17 := 
by 
  sorry

end rectangle_diagonal_proximity_probability_l55_55978


namespace theta_in_second_quadrant_l55_55429

-- Definitions based on the conditions given in the problem
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def point_P (θ : ℝ) : ℝ × ℝ := (sin θ * cos θ, 2 * cos θ)

-- Theorem based on the proof problem
theorem theta_in_second_quadrant (θ : ℝ) (h : in_third_quadrant (sin θ * cos θ) (2 * cos θ)) : 
  (sin θ > 0 ∧ cos θ < 0) :=
by
  sorry

end theta_in_second_quadrant_l55_55429


namespace faye_is_twenty_l55_55764

def chad : ℕ
def diana : ℕ := 17
def eduardo : ℕ
def faye : ℕ

axiom diana_eq_eduardo_minus_five : diana = eduardo - 5
axiom eduardo_eq_chad_plus_six : eduardo = chad + 6
axiom faye_eq_chad_plus_four : faye = chad + 4

theorem faye_is_twenty : faye = 20 :=
by
  -- To be proved using Lean
  sorry

end faye_is_twenty_l55_55764


namespace grace_pool_capacity_l55_55874

theorem grace_pool_capacity :
  let rate1 := 50 -- gallons/hour for first hose
  let time1 := 3 -- initial hours with first hose
  let rate2 := 70 -- gallons/hour for second hose
  let time2 := 2 -- additional hours with both hoses
  let water1 := rate1 * time1 -- water added by first hose
  let combined_rate := rate1 + rate2 -- combined rate of both hoses
  let water2 := combined_rate * time2 -- water added by both hoses
  let total_water := water1 + water2 -- total water in the pool
  total_water = 390 -- total capacity of the pool
:=
by
  let rate1 := 50
  let time1 := 3
  let rate2 := 70
  let time2 := 2
  let water1 := rate1 * time1
  let combined_rate := rate1 + rate2
  let water2 := combined_rate * time2
  let total_water := water1 + water2
  have h1 : water1 = 150 := by sorry
  have h2 : combined_rate = 120 := by sorry
  have h3 : water2 = 240 := by sorry
  have h4 : total_water = 390 := by
    rw [←h1, ←h3]
    sorry
  exact h4

end grace_pool_capacity_l55_55874


namespace sum_a_eq_129_l55_55454

theorem sum_a_eq_129 (a : Fin 8 → ℤ) (h : (3 * 2 - 5) ^ 7 = a 0 + a 1 * (2 - 1) + a 2 * (2 - 1) ^ 2 + a 3 * (2 - 1) ^ 3 + a 4 * (2 - 1) ^ 4 + a 5 * (2 - 1) ^ 5 + a 6 * (2 - 1) ^ 6 + a 7 * (2 - 1) ^ 7) (h0 : a 0 = -128) : (∑ i in Finset.range 7, a (i+1)) = 129 := 
by
  sorry

end sum_a_eq_129_l55_55454


namespace points_concyclic_l55_55931

-- Define the geometric setup
variable {K : Type*} [Field K] [EuclideanSpace K (Fin 3)]
variables {A B C H P : K}

-- Conditions
-- 1. \( ABC \) is a triangle with orthocenter \( H \)
def is_orthocenter (A B C H : K) : Prop := sorry
  
-- 2. \( P \) is a point different from \( A \), \( B \), and \( C \)
def different (P : K) (A B C : K) : Prop := P ≠ A ∧ P ≠ B ∧ P ≠ C

-- 3. The intersections of the \( (PA) \), \( (PB) \), and \( (PC) \) with the circumcircle
def intersection_circumcircle (P A B C : K) : K := sorry  -- Placeholder for actual intersection logic

-- 4. Reflect \( A' \), \( B' \), \( C' \) over \( (BC) \), \( (CA) \), and \( (AB) \)
def reflection_over_line (P A B : K) : K := sorry  -- Placeholder for actual reflection logic

-- Define points \( A^* \), \( B^* \), \( C^* \) as reflections
def A_star := reflection_over_line (intersection_circumcircle P A B C) B C
def B_star := reflection_over_line (intersection_circumcircle P B A C) C A
def C_star := reflection_over_line (intersection_circumcircle P C A B) A B

-- The goal statement:
theorem points_concyclic {A B C H P : K} 
  (h_orthocenter : is_orthocenter A B C H)
  (h_diff : different P A B C)
  (a_circ := intersection_circumcircle P A B C)
  (b_circ := intersection_circumcircle P B A C)
  (c_circ := intersection_circumcircle P C A B)
  (A_star := reflection_over_line a_circ B C)
  (B_star := reflection_over_line b_circ C A)
  (C_star := reflection_over_line c_circ A B) :
  ∃ (O : K), 
    is_cyclic_quadrilateral A_star B_star C_star H sorry := sorry

end points_concyclic_l55_55931


namespace solve_for_x_l55_55992

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l55_55992


namespace probability_sum_even_of_two_distinct_primes_l55_55891

theorem probability_sum_even_of_two_distinct_primes :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23}
  let total_pairs_odd := { (a, b) | a ∈ primes ∧ b ∈ primes ∧ a ≠ b ∧ (a % 2 = 1) ∧ (b % 2 = 1) }
  let num_total_pairs := 9 * (9 - 1) / 2
  let num_odd_pairs := finset.card (total_pairs_odd.to_finset)
  (num_odd_pairs : ℚ) / num_total_pairs = 7 / 9 := 
by 
  sorry

end probability_sum_even_of_two_distinct_primes_l55_55891


namespace vector_magnitude_difference_l55_55089

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55089


namespace inequality_amgm_l55_55523

theorem inequality_amgm (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) := 
by 
  sorry

end inequality_amgm_l55_55523


namespace problem1_solution_l55_55995

theorem problem1_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 10) (h2 : x / 2 - (y + 1) / 3 = 1) : 
  x = 3 ∧ y = 1 / 2 :=
by
  sorry

end problem1_solution_l55_55995


namespace minimum_value_expression_l55_55509

theorem minimum_value_expression (a b c d e f : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
(h_sum : a + b + c + d + e + f = 7) : 
  ∃ min_val : ℝ, min_val = 63 ∧ 
  (∀ a b c d e f : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 0 < f → a + b + c + d + e + f = 7 → 
  (1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f) ≥ min_val) := 
sorry

end minimum_value_expression_l55_55509


namespace find_m_in_interval_l55_55357

/-- Define the sequence recursively starting with x_0 = 7 -/
def seq (n : ℕ) : ℝ :=
  Nat.recOn n 7 (λ n x_n, (x_n^2 + 6 * x_n + 8) / (x_n + 7))

theorem find_m_in_interval :
  ∃ m : ℕ, (seq m ≤ 3 + 1 / 2^18) ∧ 69 ≤ m ∧ m ≤ 205 :=
sorry

end find_m_in_interval_l55_55357


namespace minimum_photos_l55_55820

theorem minimum_photos (G B : ℕ) (n : ℕ) : G = 4 → B = 8 → n ≥ 33 → 
  (∃ (p : fin ((G + B) choose 2) → (fin (G + B) × fin (G + B))),
  (∃ i j : fin (G + B), i ≠ j ∧ p i = p j) ∨ 
  (∃ k j : fin (G + B), k ≤ G ∧ j ≤ G ∧ p k = p j) ∨
  (∃ k j : fin (G + B), k > G ∧ j > G ∧ p k = p j)) :=
by
  intros G_eq B_eq n_ge_33
  sorry

end minimum_photos_l55_55820


namespace mike_coins_value_l55_55954

theorem mike_coins_value (d q : ℕ)
  (h1 : d + q = 17)
  (h2 : q + 3 = 2 * d) :
  10 * d + 25 * q = 345 :=
by
  sorry

end mike_coins_value_l55_55954


namespace find_other_number_l55_55565

theorem find_other_number
  (x y lcm hcf : ℕ)
  (h_lcm : Nat.lcm x y = lcm)
  (h_hcf : Nat.gcd x y = hcf)
  (h_x : x = 462)
  (h_lcm_value : lcm = 2310)
  (h_hcf_value : hcf = 30) :
  y = 150 :=
by
  sorry

end find_other_number_l55_55565


namespace factorize_poly1_factorize_poly2_l55_55771

theorem factorize_poly1 (x : ℝ) : 2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4) :=
by
  sorry

theorem factorize_poly2 (x : ℝ) : x^2 - 14 * x + 49 = (x - 7) ^ 2 :=
by
  sorry

end factorize_poly1_factorize_poly2_l55_55771


namespace joggers_meeting_time_l55_55708

def lap_time (ben_lap : ℕ) (carol_lap : ℕ) (dave_lap : ℕ) : ℕ :=
  Int.natAbs (Nat.lcm (Nat.lcm ben_lap carol_lap) dave_lap)

def earliest_meeting_time (start_time : ℕ) (lap_time : ℕ) : ℕ :=
  start_time + lap_time / 60

def time_in_minutes : ℕ :=
  7 * 60  -- 7:00 AM in minutes

theorem joggers_meeting_time (ben_lap : 5) (carol_lap : 8) (dave_lap : 9)
                            (hc : lap_time 5 8 9 = 360) :
  earliest_meeting_time time_in_minutes 360 = 13 * 60 := 
sorry

end joggers_meeting_time_l55_55708


namespace tuesday_snow_correct_l55_55495

-- Define the snowfall amounts as given in the conditions
def monday_snow : ℝ := 0.32
def total_snow : ℝ := 0.53

-- Define the amount of snow on Tuesday as per the question to be proved
def tuesday_snow : ℝ := total_snow - monday_snow

-- State the theorem to prove that the snowfall on Tuesday is 0.21 inches
theorem tuesday_snow_correct : tuesday_snow = 0.21 := by
  -- Proof skipped with sorry
  sorry

end tuesday_snow_correct_l55_55495


namespace find_60th_pair_l55_55588

-- Defining the sequence

def sequence : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 =>
  let sum_n := n + 2
  let k := n / sum_n + 1
  (k, sum_n - k)

-- Proving the 60th pair in the sequence

theorem find_60th_pair : sequence 59 = (5, 6) :=
by
  sorry

end find_60th_pair_l55_55588


namespace valid_inequalities_l55_55282

theorem valid_inequalities :
  (∀ x : ℝ, x^2 + 6x + 10 > 0) ∧ (∀ x : ℝ, -x^2 + x - 2 < 0) := by
  sorry

end valid_inequalities_l55_55282


namespace min_photos_required_l55_55828

theorem min_photos_required (girls boys : ℕ) (children : ℕ) : 
  girls = 4 → boys = 8 → children = girls + boys →
  ∃ n, n ≥ 33 ∧ (∀ (p : ℕ), p < n → 
  (∃ (g g' : ℕ), g < girls ∧ g' < girls ∧ g ≠ g' ∨ 
   ∃ (b b' : ℕ), b < boys ∧ b' < boys ∧ b ≠ b' ∨ 
   ∃ (g : ℕ) (b : ℕ), g < girls ∧ b < boys ∧ ∃ (g' : ℕ) (b' : ℕ), g = g' ∧ b = b'))) :=
by
  sorry

end min_photos_required_l55_55828


namespace probability_sum_even_of_two_distinct_primes_l55_55892

theorem probability_sum_even_of_two_distinct_primes :
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23}
  let total_pairs_odd := { (a, b) | a ∈ primes ∧ b ∈ primes ∧ a ≠ b ∧ (a % 2 = 1) ∧ (b % 2 = 1) }
  let num_total_pairs := 9 * (9 - 1) / 2
  let num_odd_pairs := finset.card (total_pairs_odd.to_finset)
  (num_odd_pairs : ℚ) / num_total_pairs = 7 / 9 := 
by 
  sorry

end probability_sum_even_of_two_distinct_primes_l55_55892


namespace vector_magnitude_l55_55103

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55103


namespace symmetric_line_eq_l55_55449

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a-1, a+1)
def B (a : ℝ) : ℝ × ℝ := (a, a)

-- We want to prove the equation of the line L about which points A and B are symmetric is "x - y + 1 = 0".
theorem symmetric_line_eq (a : ℝ) : 
  ∃ m b, (m = 1) ∧ (b = 1) ∧ (∀ x y, (y = m * x + b) ↔ (x - y + 1 = 0)) :=
sorry

end symmetric_line_eq_l55_55449


namespace ellipse_major_axis_length_l55_55240

theorem ellipse_major_axis_length :
  (∀ x y : ℝ, (x^2 / 9 + y^2 / 4 = 1) → true) →
  ∃ a : ℝ, 2 * a = 6 :=
by
  intros _
  use 3
  simp
  exact rfl

end ellipse_major_axis_length_l55_55240


namespace vector_magnitude_subtraction_l55_55068

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55068


namespace maximum_triangle_area_l55_55918

def triangle_area_max (a b c : ℝ) (A B C : ℝ) : ℝ :=
  have h1 : a = 2 := by sorry
  have h2 : 3 * b * sin C - 5 * c * sin B * cos A = 0 := by sorry
  -- use Law of Sines (assumed as sin A = 1 to maximize area)
  have h3 : ∀(A B C : ℝ), sin A = 1 ∧ A = π / 2 ∧ a / sin A = b / sin B ∧ b * sin C = (5 / 3) * c * sin B * cos A := by sorry
  sorry

theorem maximum_triangle_area :
  ∃ (a b c A B C : ℝ), a = 2 ∧ 3 * b * sin C - 5 * c * sin B * cos A = 0 ∧
    triangle_area_max a b c A B C = (10 / 3) :=
by
  use [2, sorry, sorry, sorry, sorry, sorry]
  split
  . rfl
  split
  . sorry
  . sorry

end maximum_triangle_area_l55_55918


namespace find_smallest_value_of_sum_of_squares_l55_55465
noncomputable def smallest_value (x y z : ℚ) := x^2 + y^2 + z^2

theorem find_smallest_value_of_sum_of_squares :
  ∃ (x y z : ℚ), (x + 4) * (y - 4) = 0 ∧ 3 * z - 2 * y = 5 ∧ smallest_value x y z = 457 / 9 :=
by
  sorry

end find_smallest_value_of_sum_of_squares_l55_55465


namespace inequality_not_always_correct_l55_55108

variables (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x > y) (h₄ : z > 0)

theorem inequality_not_always_correct :
  ¬ ∀ z > 0, (xz^2 / z > yz^2 / z) :=
sorry

end inequality_not_always_correct_l55_55108


namespace janet_miles_per_day_l55_55166

def total_miles : ℕ := 72
def days : ℕ := 9
def miles_per_day : ℕ := 8

theorem janet_miles_per_day : total_miles / days = miles_per_day :=
by {
  sorry
}

end janet_miles_per_day_l55_55166


namespace mul_abs_eq_l55_55587

theorem mul_abs_eq : (-3.6) * | -2 | = -7.2 := by
  sorry

end mul_abs_eq_l55_55587


namespace find_t_perpendicular_l55_55872

def vec (α : Type*) := prod α α

noncomputable def dot_product {α : Type*} [has_add α] [has_mul α] [comm_semiring α] 
  (a b : vec α) : α := 
  (a.1 * b.1) + (a.2 * b.2)

noncomputable def perpendicular_to_a {α : Type*} [has_add α] [has_mul α] [comm_semiring α]
  [has_sub α] [neg_add_cancel_left α] (a b : vec α) (t : α) : Prop := 
  dot_product (a, (1, 2)) (a.1 + t * b.1, a.2 + t * b.2) = 0

theorem find_t_perpendicular (t : ℚ) 
  (ha : vec ℚ := (1, 2)) 
  (hb : vec ℚ := (2, 3)) 
  (h_perpendicular : perpendicular_to_a ha hb t) : 
  t = -5 / 8 := 
sorry

end find_t_perpendicular_l55_55872


namespace tangent_perpend_iff_l55_55197

noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x
noncomputable def g (a x : ℝ) : ℝ := a * x + 2 * Real.cos x

theorem tangent_perpend_iff (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, has_deriv_at f x (-Real.exp x - 1) ∧ has_deriv_at (λ x, g a x) y (a - 2 * Real.sin y) ∧ (-Real.exp x - 1) * (a - 2 * Real.sin y) = -1) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end tangent_perpend_iff_l55_55197


namespace Rachel_spent_on_lunch_fraction_l55_55977

variable {MoneyEarned MoneySpentOnDVD MoneyLeft MoneySpentOnLunch : ℝ}

-- Given conditions
axiom Rachel_earnings : MoneyEarned = 200
axiom Rachel_spent_on_DVD : MoneySpentOnDVD = MoneyEarned / 2
axiom Rachel_leftover : MoneyLeft = 50
axiom Rachel_total_spent : MoneyEarned - MoneyLeft = MoneySpentOnLunch + MoneySpentOnDVD

-- Prove that Rachel spent 1/4 of her money on lunch
theorem Rachel_spent_on_lunch_fraction :
  MoneySpentOnLunch / MoneyEarned = 1 / 4 :=
sorry

end Rachel_spent_on_lunch_fraction_l55_55977


namespace david_profit_l55_55684

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l55_55684


namespace tangent_line_eqn_min_value_exists_inequality_x_l55_55436

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.log x

theorem tangent_line_eqn:
  (f 1 1 = 1) → (∀ x y, y = f 1 x → y - 1 = (f 1 1) * (x - 1)) → 
  ∃ (m b : ℝ), ∀ (x : ℝ), f 1 x = m * x + b := 
by
  sorry

theorem min_value_exists (a : ℝ):
  ∃ (a : ℝ), ∀ x, f a x ≥ (3 / 2) := 
by
  use a
  have a := (1 / 2 * exp 2)
  sorry

theorem inequality_x (x : ℝ):
  x > 0 → e ^ (2 * x^3) - 2 * x > 2 * (x + 1) * log x := 
by
  sorry

end tangent_line_eqn_min_value_exists_inequality_x_l55_55436


namespace one_fourths_in_one_eighth_l55_55116

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end one_fourths_in_one_eighth_l55_55116


namespace vec_magnitude_is_five_l55_55049

noncomputable def vec_a : ℝ × ℝ := (2, 1)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vec_magnitude_is_five : magnitude (vec_sub vec_a vec_b) = 5 := by
  sorry

end vec_magnitude_is_five_l55_55049


namespace number_of_different_gender_pairs_even_l55_55740

theorem number_of_different_gender_pairs_even (B G : ℕ) (seating : Fin (B + G) → Bool) :
  (∑ i in Finset.range (B + G), if seating i ≠ seating (i + 1) % (B + G) then 1 else 0) % 2 = 0 :=
sorry

end number_of_different_gender_pairs_even_l55_55740


namespace vector_magnitude_subtraction_l55_55061

theorem vector_magnitude_subtraction :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  (real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5) :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-2, 4)
  have h : real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2) = 5 := sorry
  exact h

end vector_magnitude_subtraction_l55_55061


namespace find_zeroes_of_f_range_of_a_for_distinct_real_roots_l55_55859

def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_zeroes_of_f (x : ℝ) : quadratic_roots 1 (-2) (-3) x ↔ x = 3 ∨ x = -1 :=
by {
  -- This is a template for the proof.
  sorry
}

theorem range_of_a_for_distinct_real_roots (a : ℝ) (h : a ≠ 0) :
  (∀ b : ℝ, let f : ℝ → ℝ := λ x, a * x^2 + b * x + (b - 1) in
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_roots a b (b - 1) x₁ ∧ quadratic_roots a b (b - 1) x₂) ↔
  (0 < a ∧ a < 1) :=
by {
  -- This is a template for the proof.
  sorry
}

end find_zeroes_of_f_range_of_a_for_distinct_real_roots_l55_55859


namespace squirrel_jumps_l55_55323

theorem squirrel_jumps (N : ℕ) (hN : N = 40) : 
  ∃ (positions : Set ℤ), positions.card = N + 1 ∧
  ∀ x ∈ positions, ∃ (jump_seq : Fin N → Bool), 
    x = (Finset.univ.filter (λ i, jump_seq i = true)).card - 
        (Finset.univ.filter (λ i, jump_seq i = false)).card :=
sorry

end squirrel_jumps_l55_55323


namespace tree_height_at_3_years_l55_55326

-- Define the conditions as Lean definitions
def tree_height (years : ℕ) : ℕ :=
  2 ^ years

-- State the theorem using the defined conditions
theorem tree_height_at_3_years : tree_height 6 = 32 → tree_height 3 = 4 := by
  intro h
  sorry

end tree_height_at_3_years_l55_55326


namespace summer_camp_students_l55_55597

theorem summer_camp_students (x : ℕ)
  (h1 : (1 / 6) * x = n_Shanghai)
  (h2 : n_Tianjin = 24)
  (h3 : (1 / 4) * x = n_Chongqing)
  (h4 : n_Beijing = (3 / 2) * (n_Shanghai + n_Tianjin)) :
  x = 180 :=
by
  sorry

end summer_camp_students_l55_55597


namespace part1_part2_part3_l55_55886

-- (I) Prove that m = 1 if the graph of f(x) = (x^2 + mx + m) / x is symmetrical about (0, 1)
theorem part1 (f : ℝ → ℝ) (m : ℝ)
  (h_symm_1 : ∀ x, f(x) + f(-x) = 2)
  (h_f : ∀ x, f(x) = (x^2 + m * x + m) / x) :
  m = 1 := sorry

-- (II) Prove the expression for g(x) on (-∞, 0) given g(x) is symmetrical about (0, 1)
theorem part2 (g : ℝ → ℝ) (a : ℝ)
  (h_symm_2 : ∀ x, g(x) + g(-x) = 2)
  (h_g_pos : ∀ x, 0 < x → g(x) = x^2 + a * x + 1) :
  ∀ x, x < 0 → g(x) = -x^2 + a * x + 1 := sorry

-- (III) Prove the range of a given the results of part1 and part2
theorem part3 (f g : ℝ → ℝ) (m a : ℝ)
  (h_1 : ∀ x, f(x) + f(-x) = 2)
  (h_f : ∀ x, f(x) = (x^2 + m * x + m) / x)
  (h_m : m = 1)
  (h_2 : ∀ x, g(x) + g(-x) = 2)
  (h_g_pos : ∀ x, 0 < x → g(x) = x^2 + a * x + 1)
  (h_g_neg : ∀ x, x < 0 → g(x) = -x^2 + a * x + 1)
  (h_ineq : ∀ x t, x < 0 → t > 0 → g(x) < f(t)) :
  a ∈ Set.Ioo (-2 * Real.sqrt 2) ∞ := sorry

end part1_part2_part3_l55_55886


namespace min_percentage_excellent_both_l55_55702

theorem min_percentage_excellent_both (P_M : ℝ) (P_C : ℝ) (hM : P_M = 0.7) (hC : P_C = 0.25) :
  (P_M * P_C = 0.175) :=
by
  rw [hM, hC]
  norm_num
  done

end min_percentage_excellent_both_l55_55702


namespace surface_area_ratio_eq_volume_ratio_eq_l55_55777

noncomputable def surface_area_ratio (r : ℝ) : ℝ :=
  let S_sphere := 4 * Math.pi * r ^ 2
  let R := Math.sqrt 3 * r
  let H := 3 * r
  let l := 2 * Math.sqrt 3 * r
  let S_cone := Math.pi * R * l + Math.pi * R ^ 2
  in S_sphere / S_cone

noncomputable def volume_ratio (r : ℝ) : ℝ := 
  let V_sphere := (4 / 3) * Math.pi * r ^ 3
  let R := Math.sqrt 3 * r
  let H := 3 * r
  let V_cone := (1 / 3) * Math.pi * R ^ 2 * H
  in V_sphere / V_cone

theorem surface_area_ratio_eq (r : ℝ) : surface_area_ratio r = 4 / 9 := 
by sorry

theorem volume_ratio_eq (r : ℝ) : volume_ratio r = 4 / 9 := 
by sorry

end surface_area_ratio_eq_volume_ratio_eq_l55_55777


namespace max_f_at_x0_and_decreasing_after_x0_l55_55442

noncomputable def f (x : ℝ) : ℝ := sin x - (1/3) * x

def x_0 : ℝ := sorry -- Assume x_0 such that cos x_0 = 1 / 3 and x_0 ∈ [0, π]

theorem max_f_at_x0_and_decreasing_after_x0 :
  (cos x_0 = 1 / 3 ∧ (0 ≤ x_0 ∧ x_0 ≤ Real.pi)) →
  (∀ x ∈ (Set.Icc 0 Real.pi), f x ≤ f x_0) ∧
  (∀ x y, (x ∈ Set.Icc x_0 Real.pi) → (y ∈ Set.Icc x_0 Real.pi) → (x ≤ y → f y ≤ f x))
:= sorry

end max_f_at_x0_and_decreasing_after_x0_l55_55442


namespace percentage_increase_l55_55927

-- Definitions based on given conditions
def old_apartment_cost : ℕ := 1200
def yearly_savings : ℕ := 7680
def number_of_people_sharing : ℕ := 3

-- The problem statement involves proving the percentage increase in cost.
theorem percentage_increase (old_apartment_cost yearly_savings number_of_people_sharing : ℕ)
  (h_old_apartment_cost : old_apartment_cost = 1200)
  (h_yearly_savings : yearly_savings = 7680)
  (h_number_of_people_sharing : number_of_people_sharing = 3) :
  let monthly_savings := yearly_savings / 12,
      new_cost_sharing := old_apartment_cost - monthly_savings,
      total_new_cost := new_cost_sharing * number_of_people_sharing,
      cost_difference := total_new_cost - old_apartment_cost,
      percentage_increase := (cost_difference.toRat / old_apartment_cost.toRat) * 100 in
  percentage_increase = 40 := 
by
  sorry

end percentage_increase_l55_55927


namespace find_a5_l55_55852

-- Define an arithmetic sequence with a given common difference
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Define that three terms form a geometric sequence
def geometric_sequence (x y z : ℝ) := y^2 = x * z

-- Given conditions for the problem
def a₁ : ℝ := 1  -- found from the geometric sequence condition
def d : ℝ := 2

-- The definition of the sequence {a_n} based on the common difference
noncomputable def a_n (n : ℕ) : ℝ := a₁ + n * d

-- Given that a_1, a_2, a_5 form a geometric sequence
axiom geo_progression : geometric_sequence a₁ (a_n 1) (a_n 4)

-- The proof goal
theorem find_a5 : a_n 4 = 9 :=
by
  -- the proof is skipped
  sorry

end find_a5_l55_55852


namespace jack_dumped_water_l55_55921

theorem jack_dumped_water :
  let drip_rate := 40 -- ml per minute
  let evap_rate := 200 -- ml per hour
  let duration := 9 -- hours
  let water_left := 7800 -- ml
  let minutes_in_an_hour := 60
  let ml_per_liter := 1000
  let total_water_dripped := drip_rate * duration * minutes_in_an_hour
  let total_water_evaporated := evap_rate * duration
  let net_water := total_water_dripped - total_water_evaporated
  (net_water - water_left) / ml_per_liter = 12 :=
by {
  let drip_rate := 40
  let evap_rate := 200
  let duration := 9
  let water_left := 7800
  let minutes_in_an_hour := 60
  let ml_per_liter := 1000
  let total_water_dripped := drip_rate * duration * minutes_in_an_hour
  let total_water_evaporated := evap_rate * duration
  let net_water := total_water_dripped - total_water_evaporated
  show (net_water - water_left) / ml_per_liter = 12,
  sorry
}

end jack_dumped_water_l55_55921


namespace rosie_savings_l55_55979

theorem rosie_savings (m : ℕ) : 
  let initial_amount := 120
  let deposit_amount := 30
  let deposits := m
  let total_savings := initial_amount + deposit_amount * deposits
  in total_savings = 120 + 30 * m := 
by 
  sorry

end rosie_savings_l55_55979


namespace ms_watsons_class_second_graders_l55_55600

theorem ms_watsons_class_second_graders (k g1 g3 total absent : ℕ) (students : ℕ → Prop)
  (h1 : k = 34) (h2 : g1 = 48) (h3 : g3 = 5) (h4 : total = 120) (h5 : absent = 6)
  (h6 : students 1 = k) 
  (h7 : students 2 = g1)
  (h8 : students 4 = g3)
  (h9 : (∑ n in finset.range 5, students n) = total - absent - students 3) :
  students 3 = 27 :=
by
  sorry

end ms_watsons_class_second_graders_l55_55600


namespace arc_MTN_constant_l55_55755

-- Definitions of the geometric conditions.
variables (P Q R : Point) 
variable (h : ℝ) -- The altitude from Q to side PR
variable (circle : Circle) -- The circle with radius equal to the altitude
variable (s : ℝ) -- Length of the sides PQ and QR in the isosceles triangle

-- Defining the specific properties of the triangle.
def isosceles_triangle (P Q R : Point) : Prop :=
  ∠PQR = 50 ∧ ∠PRQ = 50 ∧ ∠QRP = 80

-- Altitude definition based on side length and angle.
def altitude (s : ℝ) : ℝ := s * Real.sin (80 * (Real.pi / 180))

-- Circle properties based on the given conditions
def circle_properties (circle : Circle) (h : ℝ) (PQ : Line) : Prop :=
  circle.radius = h ∧ ∀ T : Point, T ∈ PQ → circle.tangent_at T

-- Definition of arc MTN degrees
def arc_MTN_degrees (circle : Circle) (P Q R M N T : Point) : ℝ :=
  80

-- The theorem to prove the problem's statement
theorem arc_MTN_constant (P Q R M N T : Point) (h : ℝ)
  (iso_PQR : isosceles_triangle P Q R)
  (alt_h : h = altitude s)
  (circle_tangent : circle_properties circle h (line_PQ P Q)) : 
  arc_MTN_degrees circle P Q R M N T = 80 :=
sorry

end arc_MTN_constant_l55_55755


namespace magnetic_field_intensity_l55_55330

variables (I : ℝ) (x y z : ℝ)

def magnetic_field (I x y z : ℝ) : ℝ × ℝ :=
  (-(2 * I * y) / (x^2 + y^2), (2 * I * x) / (x^2 + y^2))

theorem magnetic_field_intensity (I : ℝ) (x y z : ℝ) :
  magnetic_field I x y z = (-(2 * I * y) / (x^2 + y^2), (2 * I * x) / (x^2 + y^2)) :=
by
  sorry

end magnetic_field_intensity_l55_55330


namespace sum_binoms_eq_fibonacci_l55_55175

-- Definitions
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Condition n + k = m
def sum_over_binoms (m : ℕ) : ℕ :=
  ∑ n in (Finset.range (m+1)), if h : n ≤ m - n then binom n (m - n) else 0

-- The proof statement
theorem sum_binoms_eq_fibonacci (m : ℕ) (h : m ≥ 1) : sum_over_binoms m = fibonacci (m + 1) :=
sorry

end sum_binoms_eq_fibonacci_l55_55175


namespace racers_final_segment_l55_55344

def final_racer_count : Nat := 9

def segment_eliminations (init_count: Nat) : Nat :=
  let seg1 := init_count - Int.toNat (Nat.sqrt init_count)
  let seg2 := seg1 - seg1 / 3
  let seg3 := seg2 - (seg2 / 4 + (2 ^ 2))
  let seg4 := seg3 - seg3 / 3
  let seg5 := seg4 / 2
  let seg6 := seg5 - (seg5 * 3 / 4)
  seg6

theorem racers_final_segment
  (init_count: Nat)
  (h: init_count = 225) :
  segment_eliminations init_count = final_racer_count :=
  by
  rw [h]
  unfold segment_eliminations
  sorry

end racers_final_segment_l55_55344


namespace sum_of_a_and_b_l55_55734

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55734


namespace phase_and_vertical_shift_of_cosine_l55_55776

theorem phase_and_vertical_shift_of_cosine :
  ∀ x, ∃ (phase_shift vertical_shift : ℝ), y = cos(2 * x + π / 2) + 2 → phase_shift = π / 4 ∧ vertical_shift = 2 := 
by
  intros x
  use π / 4, 2
  intro h
  sorry

end phase_and_vertical_shift_of_cosine_l55_55776


namespace find_theta_l55_55382

theorem find_theta :
  ∀ θ ∈ Ioc (Real.pi / 12) (5 * Real.pi / 12), ∀ x ∈ Icc 0 2, 
    x^2 * Real.cos θ - x * (2 - x) + (2 - x)^2 * Real.sin θ > 0 := 
by 
  intros θ hθ x hx 
  sorry

end find_theta_l55_55382


namespace square_with_area_one_has_side_length_one_l55_55590

def square_area_to_side_length (A : ℝ) (hA : A = 1) : ℝ := 
  Real.sqrt A

theorem square_with_area_one_has_side_length_one (A : ℝ) (hA : A = 1) : 
  square_area_to_side_length A hA = 1 :=
by
  rw [square_area_to_side_length, hA]
  simp
  sorry

end square_with_area_one_has_side_length_one_l55_55590


namespace find_number_of_dimes_l55_55639

def total_value (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 + half_dollars * 50

def number_of_coins (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies + nickels + dimes + quarters + half_dollars

theorem find_number_of_dimes
  (pennies nickels dimes quarters half_dollars : Nat)
  (h_value : total_value pennies nickels dimes quarters half_dollars = 163)
  (h_coins : number_of_coins pennies nickels dimes quarters half_dollars = 13)
  (h_penny : 1 ≤ pennies)
  (h_nickel : 1 ≤ nickels)
  (h_dime : 1 ≤ dimes)
  (h_quarter : 1 ≤ quarters)
  (h_half_dollar : 1 ≤ half_dollars) :
  dimes = 3 :=
sorry

end find_number_of_dimes_l55_55639


namespace x_n_is_square_for_all_n_l55_55950

-- Define the sequence x_n as given in the problem conditions
def x : ℕ → ℤ
| 0 := 1
| 1 := 1
| (n+2) := 2 * x (n+1) + 8 * x n - 1

-- Theorem to state that x_n is a perfect square for all n
theorem x_n_is_square_for_all_n : ∀ n : ℕ, ∃ k : ℤ, x n = k * k := by
  sorry

end x_n_is_square_for_all_n_l55_55950


namespace eval_expr1_eval_expr2_l55_55341

theorem eval_expr1 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := 
by
  sorry

theorem eval_expr2 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^2 + b^2) / (a + b) = 5.8 :=
by
  sorry

end eval_expr1_eval_expr2_l55_55341


namespace find_m_plus_n_l55_55846

/-- Given the conditions:
  1. \(\sum_{k=1}^{30} \sin 6k = \tan \frac{m}{n}\)
  2. \(m\) and \(n\) are relatively prime positive integers
  3. \(\frac{m}{n} < 90\)
Prove that \(m+n = 85\). -/
theorem find_m_plus_n :
  ∃ (m n : ℕ),
  (∀ k : ℕ, k = finset.sum (finset.range 1 31) (λ k, real.sin (6 * k))) =
  real.tan (m / n) ∧
  nat.coprime m n ∧
  (m / n) < 90 ∧
  (m + n = 85) :=
sorry

end find_m_plus_n_l55_55846


namespace bound_sum_squares_l55_55945

open Real
open Int

theorem bound_sum_squares (n : ℕ) (a : Fin n → ℝ) (k : ℕ)
  (h1 : ∑ i, (a i)^2 = 1) 
  (h2 : 2 ≤ k) : 
  ∃ x : Fin n → ℤ, (∀ i, |x i| ≤ k-1) ∧ (¬ ∀ i, x i = 0) ∧ 
  |∑ i, (a i) * (x i)| ≤ (k - 1 : ℝ) * sqrt n / (k^n - 1) := 
sorry

end bound_sum_squares_l55_55945


namespace sum_and_average_of_first_ten_multiples_of_11_l55_55627

theorem sum_and_average_of_first_ten_multiples_of_11 :
  (∑ i in finset.range 10, 11 * (i + 1)) = 605 ∧
  ((∑ i in finset.range 10, 11 * (i + 1)) / 10) = 60.5 := 
by 
  sorry

end sum_and_average_of_first_ten_multiples_of_11_l55_55627


namespace remainder_of_modified_division_l55_55279

theorem remainder_of_modified_division (x y u v : ℕ) (hx : 0 ≤ v ∧ v < y) (hxy : x = u * y + v) :
  ((x + 3 * u * y) % y) = v := by
  sorry

end remainder_of_modified_division_l55_55279


namespace vehicle_speeds_l55_55619

theorem vehicle_speeds (V_A V_B V_C : ℝ) (d_AB d_AC : ℝ) (decel_A : ℝ)
  (V_A_eff : ℝ) (delta_V_A : ℝ) :
  V_A = 70 → V_B = 50 → V_C = 65 →
  decel_A = 5 → V_A_eff = V_A - decel_A → 
  d_AB = 40 → d_AC = 250 →
  delta_V_A = 10 →
  (d_AB / (V_A_eff + delta_V_A - V_B) < d_AC / (V_A_eff + delta_V_A + V_C)) :=
by
  intros hVA hVB hVC hdecel hV_A_eff hdAB hdAC hdelta_V_A
  -- the proof would be filled in here
  sorry

end vehicle_speeds_l55_55619


namespace bn_magnitude_l55_55222

-- Definitions

def square_side_length : ℝ := 4
def M_midpoint_CD (C D M : ℝ × ℝ) : Prop := 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
def N_on_AD (A D N : ℝ × ℝ) : Prop := 
  ∃ λ : ℝ, N = (A.1 * (1-λ) + D.1 * λ, A.2 * (1-λ) + D.2 * λ)
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Given Conditions
variables (A B C D M N : ℝ × ℝ)
variable (λ : ℝ)
axiom h1 : square_side_length = 4
axiom h2 : M_midpoint_CD C D M
axiom h3 : N_on_AD A D N
axiom h4 : dot_product (B - M) (B - N) = 20

-- Required to Prove
theorem bn_magnitude : |(B - N)| = 5 :=
sorry

end bn_magnitude_l55_55222


namespace simplify_expression_l55_55986

theorem simplify_expression (x y : ℤ) :
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y :=
by sorry

end simplify_expression_l55_55986


namespace max_value_of_expression_l55_55186

theorem max_value_of_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 5 * x + 6 * y < 90 → 
  xy (90 - 5 * x - 6 * y) ≤ 900 :=
by
  sorry

end max_value_of_expression_l55_55186


namespace sin_values_in_interval_l55_55452

theorem sin_values_in_interval : 
  ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < 180 ∧ 0 ≤ x₂ ∧ x₂ < 180 ∧ real.sin x₁ = 0.56 ∧ real.sin x₂ = 0.56 ∧ x₁ ≠ x₂ :=
sorry

end sin_values_in_interval_l55_55452


namespace magnitude_of_a_minus_b_l55_55079

-- Defining the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Subtracting vectors a and b
def a_minus_b : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

-- Computing the magnitude of the resulting vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_a_minus_b : magnitude a_minus_b = 5 := by
  sorry

end magnitude_of_a_minus_b_l55_55079


namespace sum_of_a_and_b_l55_55738

theorem sum_of_a_and_b (a b : ℕ) (h1: a > 0) (h2 : b > 1) (h3 : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x = a ∧ y = b → x^y ≥ a^b ) :
  a + b = 24 :=
sorry

end sum_of_a_and_b_l55_55738


namespace mooncake_price_reduction_l55_55707

theorem mooncake_price_reduction
  (purchase_price : ℕ)
  (initial_price : ℕ)
  (initial_volume : ℕ)
  (volume_increase_per_yuan : ℕ)
  (max_volume : ℕ)
  (profit_target : ℕ)
  (a : ℤ)
  (new_price : ℕ := initial_price - a)
  (profit_per_box : ℕ := new_price - purchase_price)
  (expected_volume : ℕ := initial_volume + volume_increase_per_yuan * a)
  (total_profit : ℕ := profit_per_box * expected_volume) :
  purchase_price = 40 →
  initial_price = 52 →
  initial_volume = 180 →
  volume_increase_per_yuan = 10 →
  max_volume = 210 →
  profit_target = 2000 →
  total_profit = profit_target →
  a = 2 := 
by
  intros h_purchase_price h_initial_price h_initial_volume h_volume_increase_per_yuan h_max_volume h_profit_target h_total_profit
  /- Proof can be filled here later -/
  sorry

end mooncake_price_reduction_l55_55707


namespace vector_magnitude_l55_55107

theorem vector_magnitude (a b : ℝ × ℝ) (a_def : a = (2, 1)) (b_def : b = (-2, 4)) :
  ‖(a.1 - b.1, a.2 - b.2)‖ = 5 := by
  sorry

end vector_magnitude_l55_55107


namespace largest_b_among_abcd_l55_55858

theorem largest_b_among_abcd (a b c d : ℝ) : 
  a + 1 = b - 3 → a + 1 = c + 4 → a + 1 = d - 2 → b ≥ a ∧ b ≥ d ∧ b ≥ c :=
by
  intros h1 h2 h3
  have hb : b = a + 4 := by linarith
  have hc : c = a - 3 := by linarith
  have hd : d = a + 3 := by linarith
  rw [hb, hc, hd]
  split
  all_goals { linarith }
  sorry

end largest_b_among_abcd_l55_55858


namespace vector_magnitude_difference_l55_55085

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l55_55085


namespace hyperbola_asymptotes_l55_55575

theorem hyperbola_asymptotes :
  (∀ x y : ℝ, y^2 / 4 - x^2 / 8 = 1 → y = ( √2 / 2) * x ∨ y = -( √2 / 2) * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l55_55575


namespace vector_magnitude_subtraction_l55_55035

def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_subtraction :
  let a := (2, 1)
  let b := (-2, 4)
  vec_norm (vec_sub a b) = 5 :=
by
  sorry

end vector_magnitude_subtraction_l55_55035


namespace ellipse_equation_exists_lambda_l55_55908

-- we'll declare our parameters and assumptions
variable {a b x y k1 k2 x1 y1 x2 y2 m : ℝ }
variable (A B D M : ℝ → ℝ)
variable (C : ℝ → ℝ → Prop)

-- Conditions
axiom ellipse_definition : C x y ↔ ((x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0)
axiom ellipse_eccentricity : (sqrt (a^2 - b^2) / a = (sqrt 3) / 2)
axiom segment_length : (2 * sqrt 2 * sqrt ((a^2 - b^2) / a^2) = (4 / 5) * sqrt 10)

-- Points and slopes
axiom pointA : A x1 = y1
axiom pointB : B (-x1) = (-y1)
axiom pointD : (C x2 y2)
axiom AD_perp_AB : (y1 / x1) * (-x1) = -1 
axiom line_through_origin : ∀ (u v : ℝ), (A u = v → A (-u) = -v)
axiom BD_intersection : ∀ (x_bd : ℝ), C x_bd (x_bd / k1) → (BD x (y1 - (1 / k1) * x1))
axiom M_intersection : ∀ (x_am : ℝ), (A x x_am) = (BD (3 * x1) 0)

theorem ellipse_equation : C x y ↔ x^2 / 4 + y^2 = 1 := sorry

theorem exists_lambda : ∃ λ : ℝ, k1 = λ * k2 :=
  exists.intro (-1/2) sorry

end ellipse_equation_exists_lambda_l55_55908


namespace vector_magnitude_difference_l55_55013

def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-2, 4)

theorem vector_magnitude_difference :
  let diff := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)
  let magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2)
  magnitude = 5 :=
by
  -- define the difference vector
  have diff : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2),
  -- define the magnitude of the difference vector
  have magnitude := Real.sqrt (diff.1 ^ 2 + diff.2 ^ 2),
  -- provide the proof (using steps omitted via sorry)
  sorry

end vector_magnitude_difference_l55_55013


namespace angle_C_value_l55_55141

theorem angle_C_value (A B C : ℝ) (h : sin A ^ 2 - sin C ^ 2 = (sin A - sin B) * sin B) : 
  C = π / 3 :=
sorry

end angle_C_value_l55_55141


namespace find_x_l55_55561

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)
noncomputable def inv_f (y : ℝ) : ℝ := 30 / y - 2
noncomputable def h (x : ℝ) : ℝ := 4 * inv_f x

theorem find_x
  (h_def : ∀ x, h(x) = 4 * inv_f x)
  (f_def : ∀ x, f(x) = 30 / (x + 2))
  : ∃ x, h(30 / 7) = 20 :=
sorry

end find_x_l55_55561


namespace problem_solution_l55_55943

-- Definitions based on the conditions
def num_ordered_pairs (n : ℕ) : ℕ := n * n

def num_ordered_pairs_of_ordered_pairs (n : ℕ) : ℕ :=
  (num_ordered_pairs n) * (num_ordered_pairs n)

def num_unordered_pairs (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + n

def num_unordered_pairs_of_ordered_pairs (n : ℕ) : ℕ :=
  (num_ordered_pairs n) * (num_ordered_pairs n - 1) / 2 + (num_ordered_pairs n)

-- Define the elements A and B based on ℕ = 6
def A : ℕ := num_unordered_pairs_of_ordered_pairs 6
def B : ℕ := num_ordered_pairs_of_ordered_pairs 21

-- The theorem to prove
theorem problem_solution : A - B = 225 :=
by
  -- Explicitly assign values to A and B
  let A_val := 666
  let B_val := 441
  have h1 : A = A_val := rfl
  have h2 : B = B_val := rfl
  calc
    A - B = A_val - B_val : by rw [h1, h2]
    ... = 225           : by rfl

end problem_solution_l55_55943


namespace total_children_in_family_l55_55971

theorem total_children_in_family :
  (∃ b : ℕ, ∀ s : ℕ, s = 3 * b ∧ ∀ lb : ℕ, lb = b + 1 ∧ ∀ ls : ℕ, ls = 3 * b - 1 ∧
  3 * b - 1 = 2 * (b + 1) → se = 4 ∧ s = 9 → se + s = 13) :=
begin
  sorry
end

end total_children_in_family_l55_55971


namespace area_relationship_l55_55691

-- Definitions of areas of triangles, we assume (area) is a function that gives us the area of a triangle given its vertices
noncomputable def area (A B C : Type) [Triangle A B C] : ℝ := sorry

-- Conditions as definitions
variables {A B C A1 B1 C1 A2 B2 C2 : Type}
variable [Triangle A B C]
variable [Triangle A1 B1 C1]
variable [Triangle A2 B2 C2]

-- Inscribed triangle condition
def isInscribed (inner outer : Triangle) : Prop := sorry

-- Circumscribed triangle condition
def isCircumscribed (inner outer : Triangle) : Prop := sorry

-- Parallel sides condition
def correspondinglyParallel (T1 T2 : Triangle) : Prop := sorry

-- Main theorem statement
theorem area_relationship 
  (h1 : isInscribed (Triangle.mk A1 B1 C1) (Triangle.mk A B C))
  (h2 : isCircumscribed (Triangle.mk A1 B1 C1) (Triangle.mk A2 B2 C2))
  (h3 : correspondinglyParallel (Triangle.mk A1 B1 C1) (Triangle.mk A2 B2 C2)) :
  (area A B C) ^ 2 = (area A1 B1 C1) * (area A2 B2 C2) :=
sorry

end area_relationship_l55_55691


namespace proof_of_inequality_l55_55209

variables {α : Type*} [NormedAddCommGroup α]

structure Sphere (α) :=
(center : α)
(radius : ℝ)

noncomputable def center_of_mass (points : list α) : α := sorry

noncomputable def distance (x y : α) : ℝ := sorry

variables (S : Sphere α)
variables {n : ℕ} (A : fin n → α) (B : fin n → α) (M : α)

-- Assume that A_i lies on the Sphere
def on_sphere (S : Sphere α) (A : α) : Prop := distance S.center A = S.radius

-- Assuming the conditions of the problem
axiom h1 : ∀ i, on_sphere S (A i)
axiom h2 : center_of_mass (list.of_fn A) = M
axiom h3 : ∀ i, ∃ B_i, B i = B_i ∧ distance M (A i) = distance (A i) S.center ∧
                   distance M (B_i) = distance (B_i) S.center

theorem proof_of_inequality :
  ∑ i, distance M (A i) ≤ ∑ i, distance M (B i) :=
sorry

end proof_of_inequality_l55_55209
