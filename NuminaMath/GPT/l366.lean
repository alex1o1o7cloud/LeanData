import Mathlib

namespace num_pos_divisors_36_l366_366075

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366075


namespace positive_difference_l366_366332

def vertical_drops : List ℕ := [170, 120, 150, 310, 200, 145]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

def median (lst : List ℕ) : ℚ := 
  let sorted := lst.qsort (· < ·)
  let n := sorted.length
  if n % 2 = 1 then
    sorted.nth_le (n / 2) (by simp [Nat.div_lt_self n zero_lt_two])
  else
    ((sorted.nth_le (n / 2 - 1) (by linarith [Nat.div_lt_self n zero_lt_two])) +
     (sorted.nth_le (n / 2) (by linarith [Nat.div_lt_self n zero_lt_two]))) / 2

theorem positive_difference : abs (mean vertical_drops - median vertical_drops) = 22.5 := 
begin
  -- The proof is omitted
  sorry
end

end positive_difference_l366_366332


namespace quiz_answer_key_combinations_l366_366603

theorem quiz_answer_key_combinations : 
  let true_false_combinations := 2^5 in
  let valid_true_false_combinations := true_false_combinations - 2 in
  let multiple_choice_combinations := 4 * 4 in
  valid_true_false_combinations * multiple_choice_combinations = 480 :=
by
  let true_false_combinations := 2^5
  let valid_true_false_combinations := true_false_combinations - 2
  let multiple_choice_combinations := 4 * 4
  have h_valid_true_false := valid_true_false_combinations = 30
  have h_multiple_choice := multiple_choice_combinations = 16
  show valid_true_false_combinations * multiple_choice_combinations = 480
  rw [h_valid_true_false, h_multiple_choice]
  exact mul_eq_mul h_valid_true_false h_multiple_choice sorry

end quiz_answer_key_combinations_l366_366603


namespace unique_real_solution_l366_366934

theorem unique_real_solution (a : ℝ) 
  (h : ∀ x : ℝ, Polynomial.eval x (Polynomial.C x^3 - Polynomial.C a * x^2 - Polynomial.C (2 * a) * x + Polynomial.C (a^2 - 1)) = 0) : 
  a < 3 / 4 :=
by 
  sorry

end unique_real_solution_l366_366934


namespace num_pos_divisors_36_l366_366122

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366122


namespace number_of_divisors_36_l366_366080

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366080


namespace functional_equation_solution_l366_366301

theorem functional_equation_solution {f : ℝ → ℝ} 
  (h1 : ∀ x y : ℝ, f(x * y) + 2 * x = x * f(y) + 3 * f(x))
  (h2 : f(-1) = 7) : 
  f(-1001) = -3493 :=
sorry

end functional_equation_solution_l366_366301


namespace midpoint_bisects_segment_l366_366635

theorem midpoint_bisects_segment 
  (circle : Set Point)
  (A B F C D E G K L : Point)
  (h_circle : is_circle circle)
  (h_midpoint_F : is_midpoint F A B)
  (h_intersect_CD : line_through F (∂ circle) C D)
  (h_intersect_EG : line_through F (∂ circle) E G)
  (h_intersect_CE_AB : intersects (line_through C E) (line_through A B) K)
  (h_intersect_GD_AB : intersects (line_through G D) (line_through A B) L) :
  is_midpoint F K L :=
by
  sorry

end midpoint_bisects_segment_l366_366635


namespace lcm_18_24_l366_366793

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366793


namespace sequence_is_integer_sequence_sequence_perfect_square_l366_366715

open Int

-- Define the sequence {a_n}
def a : ℕ → ℤ
| 1 := 1
| 2 := 4
| (n + 2) := sqrt ((a n) * (a (n + 2)) + 1)

-- Prove that the sequence {a_n} is an integer sequence
theorem sequence_is_integer_sequence (n : ℕ) : ∀ n ≥ 1, ∃ k : ℤ, a n = k := 
by 
  sorry

-- Prove that 2 * a_n * a_{n+1} + 1 is a perfect square
theorem sequence_perfect_square (n : ℕ) (hn : n ≥ 1) : ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k * k :=
by
  sorry

end sequence_is_integer_sequence_sequence_perfect_square_l366_366715


namespace sum_of_arithmetic_sequence_l366_366624

theorem sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) 
  (h1 : ∀ n, S n = n * a₁ + (n - 1) * n / 2 * d)
  (h2 : S 1 / S 4 = 1 / 10) :
  S 3 / S 5 = 2 / 5 := 
sorry

end sum_of_arithmetic_sequence_l366_366624


namespace math_problem_l366_366641

variable (x y : ℝ)

noncomputable def arith_seq_cond : Prop := 2 * x + y = 3

noncomputable def geom_seq_cond : Prop :=
  if 0 ≤ x ∧ x ≤ 1 then
    let b := 2
    y + 3 / b = b / x ∧
    cos (asin (sqrt (1 - x ^ 2))) = x
  else if -1 ≤ x ∧ x < 0 then
    let b := 2 - 2 * x
    y + 3 / b = b / x ∧
    cos (asin (sqrt (1 - x ^ 2))) = x
  else false

noncomputable def valid_answers : Set ℝ :=
  {4, 2 * (sqrt 17 - 3)}

theorem math_problem (hA : arith_seq_cond x y) (hG : geom_seq_cond x y) :
  (x + 1) * (y + 1) ∈ valid_answers := sorry

end math_problem_l366_366641


namespace total_number_of_animals_l366_366703

-- Define the problem conditions
def number_of_cats : ℕ := 645
def number_of_dogs : ℕ := 567

-- State the theorem to be proved
theorem total_number_of_animals : number_of_cats + number_of_dogs = 1212 := by
  sorry

end total_number_of_animals_l366_366703


namespace exists_number_le_kr_l366_366840

open Real

noncomputable def transform (r : ℝ) (a b : ℝ) : Prop :=
  0 < r ∧ 0 < a ∧ 0 < b ∧ 2 * r^2 = a * b

theorem exists_number_le_kr (r : ℝ) (k : ℕ) (h_r_pos : 0 < r) (h_k_nonzero : k > 0) :
  ∃ s ∈ {s : ℝ | ∃ l : List ℝ, List.length l = k^2 ∧ initial_and_operations r ((k^2 - 1)) l ∧ s ∈ l}, s ≤ k * r :=
sorry

/-- A hypothetical function defining the initial condition and the operation applied k^2 - 1 times. -/ 
def initial_and_operations (r : ℝ) (steps : ℕ) (l : List ℝ) : Prop :=
  ∀ i, i < steps → ∃ a b, transform r a b ∧ r = a ∨ r = b

end exists_number_le_kr_l366_366840


namespace smallest_consecutive_integer_l366_366720

theorem smallest_consecutive_integer (n : ℤ) (h : 7 * n + 21 = 112) : n = 13 :=
sorry

end smallest_consecutive_integer_l366_366720


namespace common_ratio_l366_366935

namespace GeometricSeries

-- Definitions
def a1 : ℚ := 4 / 7
def a2 : ℚ := 16 / 49 

-- Proposition
theorem common_ratio : (a2 / a1) = (4 / 7) :=
by
  sorry

end GeometricSeries

end common_ratio_l366_366935


namespace lcm_18_24_eq_72_l366_366816

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366816


namespace trapezoid_problem_l366_366236

open Real

noncomputable def smallest_x_squared (x : ℝ) : ℝ :=
  let AM := 60
  let DM := 12.5
  in AM^2 - DM^2

theorem trapezoid_problem (x : ℝ) (h₁ : AB = 120) (h₂ : CD = 25) (h₃ : AD = x) (h₄ : BC = x + 20) (h₅ : circle_tangent_AD) :
  smallest_x_squared x = 3443.75 :=
by
  sorry

end trapezoid_problem_l366_366236


namespace unit_spheres_intersect_l366_366341

-- Defining the distance "a" calculated through the conditions of the problem.
def unit_spheres_distance : ℝ :=
  4 * Real.cos (4 * Real.pi / 9)

-- Assuming two unit spheres intersect such that their volumes are divided into three equal parts.
theorem unit_spheres_intersect (r : ℝ) (a : ℝ) (h_r : r = 1) (h_a: 0 < a ∧ a < 1)
  (h_eq_parts : 2 * (Real.pi / 6) * (1 - a) * (3 * (1 - a^2) + (1 - a)^2) = (2 * Real.pi / 3)) :
  2 * a = unit_spheres_distance := 
sorry

end unit_spheres_intersect_l366_366341


namespace distance_between_planes_l366_366484

noncomputable def plane1 : ℝ × ℝ × ℝ → Prop :=
  λ p, 2 * p.1 + 4 * p.2 - 4 * p.3 = 10

noncomputable def plane2 : ℝ × ℝ × ℝ → Prop :=
  λ p, 4 * p.1 + 8 * p.2 - 8 * p.3 = 18

theorem distance_between_planes :
  ∃ (d : ℝ), d = 1 / 6 ∧
  ∀ (p : ℝ × ℝ × ℝ), plane2 p → 
  abs (2 * p.1 + 4 * p.2 - 4 * p.3 - 10) / sqrt (2^2 + 4^2 + (-4)^2) = d :=
sorry

end distance_between_planes_l366_366484


namespace number_of_divisors_36_l366_366081

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366081


namespace period_of_f_angle_C_of_triangle_l366_366543

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - π / 6)

theorem period_of_f : ∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x :=
  exists.intro π sorry

theorem angle_C_of_triangle
  {A B C : ℝ}
  (h₀ : 0 < A ∧ A < π)
  (h₁ : 0 < B ∧ B < π)
  (h₂ : 0 < C ∧ C < π)
  (h₃ : A + B + C = π)
  (h₄ : f A = 1)
  (h₅ : Real.sin B ^ 2 + sqrt 2 * Real.sin A * Real.sin C = Real.sin A ^ 2 + Real.sin C ^ 2) :
  C = 5 * π / 12 :=
sorry

end period_of_f_angle_C_of_triangle_l366_366543


namespace michaels_brother_final_amount_l366_366658

theorem michaels_brother_final_amount :
  ∀ (michael_money michael_brother_initial michael_give_half candy_cost money_left : ℕ),
  michael_money = 42 →
  michael_brother_initial = 17 →
  michael_give_half = michael_money / 2 →
  let michael_brother_total := michael_brother_initial + michael_give_half in
  candy_cost = 3 →
  money_left = michael_brother_total - candy_cost →
  money_left = 35 :=
by
  intros michael_money michael_brother_initial michael_give_half candy_cost money_left
  intros h1 h2 h3 michael_brother_total h4 h5
  sorry

end michaels_brother_final_amount_l366_366658


namespace mike_salary_calculation_l366_366494

theorem mike_salary_calculation
  (F : ℝ) (M : ℝ) (new_M : ℝ) (x : ℝ)
  (F_eq : F = 1000)
  (M_eq : M = x * F)
  (increase_eq : new_M = 1.40 * M)
  (new_M_val : new_M = 15400) :
  M = 11000 ∧ x = 11 :=
by
  sorry

end mike_salary_calculation_l366_366494


namespace altitudes_parallel_to_plane_l366_366949

theorem altitudes_parallel_to_plane
  (A B C D A₁ B₁ C₁ : Point)
  (hA : is_altitude A D B C A₁)
  (hB : is_altitude B D A C B₁)
  (hC : is_altitude C D A B C₁) :
  all_lines_parallel_to_plane [line A₁ B₁, line B₁ C₁, line C₁ A₁] :=
sorry

end altitudes_parallel_to_plane_l366_366949


namespace find_f_f_neg3_l366_366539

def f (x : ℝ) : ℝ :=
  if x >= 1 then x + 2 / x - 3
  else log10 (x^2 + 1)

theorem find_f_f_neg3 : f (f (-3)) = 0 := by
  sorry

end find_f_f_neg3_l366_366539


namespace interest_rate_of_second_part_l366_366280

theorem interest_rate_of_second_part
  (total_amount : ℝ)
  (P1 : ℝ)
  (interest1_rate : ℝ)
  (total_interest : ℝ)
  (P2 : ℝ := total_amount - P1)
  (annual_interest_P1 : ℝ := P1 * (interest1_rate / 100))
  (interest2_rate : ℝ) :
  total_amount = 3200 →
  P1 = 800 →
  interest1_rate = 3 →
  total_interest = 144 →
  annual_interest_P1 + P2 * (interest2_rate / 100) = total_interest →
  interest2_rate = 5 := 
by
  intros TotalAmount P1_eq Interest1Rate TotalInterest AnnualInterest_Equation,
  sorry

end interest_rate_of_second_part_l366_366280


namespace smallest_n_for_divisibility_l366_366647

theorem smallest_n_for_divisibility (a₁ a₂ : ℕ) (n : ℕ) (h₁ : a₁ = 5 / 8) (h₂ : a₂ = 25) :
  (∃ n : ℕ, n ≥ 1 ∧ (a₁ * (40 ^ (n - 1)) % 2000000 = 0)) → (n = 7) :=
by
  sorry

end smallest_n_for_divisibility_l366_366647


namespace rectangle_area_l366_366206

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366206


namespace part_a_part_b_l366_366943

noncomputable def reverse (x : ℝ) : ℝ := 
-- Definition of the reverse function r(x) is needed here
sorry

theorem part_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx_fin : ∃ n : ℕ, x * 10^n ∈ ℤ) (hy_fin : ∃ n : ℕ, y * 10^n ∈ ℤ) :
  reverse (x * y) ≤ 10 * reverse x * reverse y :=
sorry

theorem part_b : ∃ (x y : ℝ), (∀ n : ℕ, (x * 10^n ∈ ℤ) ∧ (y * 10^n ∈ ℤ)) ∧ 
  (reverse (x * y) = 10 * reverse x * reverse y) ∧ 
  ((nat.cast (n := ℝ) (num_digits x) ≥ 2015) ∧ (nat.cast (n := ℝ) (num_digits y) ≥ 2015)) :=
sorry

end part_a_part_b_l366_366943


namespace area_of_rectangle_ABCD_l366_366177

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366177


namespace Greg_gold_amount_l366_366552

noncomputable def gold_amounts (G K : ℕ) : Prop :=
  G = K / 4 ∧ G + K = 100

theorem Greg_gold_amount (G K : ℕ) (h : gold_amounts G K) : G = 20 := 
by
  sorry

end Greg_gold_amount_l366_366552


namespace points_lie_on_hyperbola_l366_366946

def point_on_curve (t : ℝ) (ht : t ≠ 0) : ℝ × ℝ :=
(x, y) where
  x = (t^2 + 1) / t
  y = (t^2 - 1) / t

theorem points_lie_on_hyperbola (t : ℝ) (ht : t ≠ 0) :
  let (x, y) := point_on_curve t ht
  in ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * x^2 - b * y^2 = 1) :=
by
  sorry

end points_lie_on_hyperbola_l366_366946


namespace sheela_deposit_l366_366283

def monthly_income : ℝ := 16071.42857142857
def percentage : ℝ := 28 / 100

theorem sheela_deposit : monthly_income * percentage = 4500.00 :=
by
  sorry

end sheela_deposit_l366_366283


namespace rectangle_area_l366_366195

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366195


namespace root_equiv_sum_zero_l366_366956

variable {a b c : ℝ}
variable (h₀ : a ≠ 0)

theorem root_equiv_sum_zero : (1 root_of (a * 1^2 + b * 1 + c = 0)) ↔ (a + b + c = 0) :=
by
  sorry

end root_equiv_sum_zero_l366_366956


namespace area_of_rectangle_ABCD_l366_366181

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366181


namespace lcm_18_24_l366_366806

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366806


namespace lcm_18_24_eq_72_l366_366772

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366772


namespace find_a_l366_366550

theorem find_a (a : ℝ) : 
  let A := {1, 2, 3}
  let B := {x : ℝ | x^2 - (a + 1) * x + a = 0}
  A ∪ B = A → a = 1 ∨ a = 2 ∨ a = 3 :=
by
  intros
  sorry

end find_a_l366_366550


namespace negation_of_universal_prop_l366_366503

variable (a : ℝ)

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, 0 < x → Real.log x = a) ↔ (∃ x : ℝ, 0 < x ∧ Real.log x ≠ a) :=
by
  sorry

end negation_of_universal_prop_l366_366503


namespace mike_total_time_spent_l366_366252

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l366_366252


namespace rectangle_area_l366_366192

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366192


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366360

theorem remainder_3001_3002_3003_3004_3005_mod_17 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366360


namespace possible_values_of_R_l366_366880

-- Define the problem conditions and the result in Lean
theorem possible_values_of_R :
  ∃ R : ℝ, (R = 3 / (4 * real.sqrt 2) ∨ R = 1 / real.sqrt 3) →
  ∀ (S A B C D : ℝ × ℝ × ℝ),
    (S ≠ A → S ≠ B → S ≠ C → S ≠ D → A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D) →
    dist S A = 1 ∧ dist S B = 1 ∧ dist S C = 1 ∧ dist S D = 1 ∧
    dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1 →
    circle R (S + vector (A - S)) ∧ circle R (S + vector (B - S)) ∧
    circle R (S + vector (C - S)) ∧ circle R (S + vector (D - S)) :=
sorry

end possible_values_of_R_l366_366880


namespace solve_quadratic_inequality_l366_366286

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem solve_quadratic_inequality :
  let a := -10
  let b := 4
  let c := 2
  let delta := discriminant a b c in
  delta > 0 ∧ a < 0 →
  let x1 := (1 - Real.sqrt 6) / 5
  let x2 := (1 + Real.sqrt 6) / 5 in
  {x : ℝ | -10 * x^2 + 4 * x + 2 > 0} = {x : ℝ | x1 < x ∧ x < x2} :=
by
  sorry

end solve_quadratic_inequality_l366_366286


namespace area_of_rectangle_l366_366200

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366200


namespace quadratic_to_completed_square_l366_366449

-- Define the given quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 + 2 * x - 2

-- Define the completed square form of the function.
def completed_square_form (x : ℝ) : ℝ := (x + 1)^2 - 3

-- The theorem statement that needs to be proven.
theorem quadratic_to_completed_square :
  ∀ x : ℝ, quadratic_function x = completed_square_form x :=
by sorry

end quadratic_to_completed_square_l366_366449


namespace solution_of_system_l366_366294

def log4 (n : ℝ) : ℝ := log n / log 4

theorem solution_of_system (x y : ℝ) (hx : x + y = 20)
  (hy : log4 x + log4 y = 1 + log4 9) :
  (x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18) :=
by sorry

end solution_of_system_l366_366294


namespace ral_age_is_26_l366_366275

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l366_366275


namespace number_of_12_digit_numbers_with_at_least_two_consecutive_twos_l366_366039

-- Define the Fibonacci sequence with initial conditions F_1 = 1 and F_2 = 1
def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 1
| (n + 1) := fib n + fib (n - 1)

-- Define the number of valid 12-digit numbers with digits 1 or 2 (2 ^ 12)
def total_12_digit_numbers : ℕ := 2 ^ 12

-- Define the number of 12-digit numbers without two consecutive 2's as the (13)th Fibonacci number (fib 13)
def no_two_consecutive_twos : ℕ := fib 13

-- Define the number of 12-digit numbers with at least two consecutive 2's
def numbers_with_two_consecutive_twos : ℕ := total_12_digit_numbers - no_two_consecutive_twos

-- Statement to be proved
theorem number_of_12_digit_numbers_with_at_least_two_consecutive_twos :
  numbers_with_two_consecutive_twos = 3863 :=
by
  -- This is left to be proven
  sorry

end number_of_12_digit_numbers_with_at_least_two_consecutive_twos_l366_366039


namespace most_beneficial_option_l366_366891

-- Defining the conditions
def principal : ℝ := 65000
def annual_rate : ℝ := 0.05

def simple_interest_total (P : ℝ) (r : ℝ) : ℝ := P + (P * r)
def compound_interest_total (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r / n) ^ n

def option_a_total : ℝ := 68380
def option_b_total : ℝ := simple_interest_total principal annual_rate
def option_c_total : ℝ := compound_interest_total principal annual_rate 12
def option_d_total : ℝ := compound_interest_total principal annual_rate 4
def option_e_total : ℝ := compound_interest_total principal annual_rate 2

-- Proving the most beneficial option
theorem most_beneficial_option : option_b_total < option_a_total ∧
                                    option_b_total < option_c_total ∧
                                    option_b_total < option_d_total ∧
                                    option_b_total < option_e_total :=
by
  -- Proof omitted
  sorry

end most_beneficial_option_l366_366891


namespace normal_distribution_properties_l366_366699

theorem normal_distribution_properties (μ σ : ℝ) (x : ℝ) (hx_pos : 0 < σ) :
  (∀ x, (pdf μ σ x > 0) ∧ (∀ ε > 0, ∃ δ > 0, |x - μ| > δ → |pdf μ σ x| < ε)) ∧
  (pdf μ σ x = pdf μ σ (x + 2 * (μ - x))) ∧
  (x = μ → pdf μ σ x = (⨆ (x : ℝ), pdf μ σ x)) :=
begin
  sorry
end

end normal_distribution_properties_l366_366699


namespace num_pos_divisors_36_l366_366126

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366126


namespace lcm_18_24_l366_366749

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366749


namespace part_I_part_II_part_III_l366_366030

noncomputable def f (a x : ℝ) : ℝ := log x - a * x + a / x
noncomputable def f_prime (a x : ℝ) : ℝ := 1 / x - a * (1 + 1 / x^2)

-- (Ⅰ) The tangent line of f(x) at x = 1 passes through (3, 4)
theorem part_I (a : ℝ) : 
  (1 - 2 * a = 2) → (a = -1/2) := 
sorry

-- (Ⅱ) If 0 < a < 1, prove that f(a^2 / 2) > 0
theorem part_II (a : ℝ) : 
  (0 < a) → (a < 1) → (f a (a^2 / 2) > 0) := 
sorry

-- (Ⅲ) When the function f(x) has three distinct zeros, find the range of values for a
theorem part_III (a : ℝ) :
  (0 < a) → (a < 1/2) → 
  (∃ x0 : ℝ, x0 ≠ 1 ∧ x0 ≠ 1 / x0 ∧ f a x0 = 0 ∧ f a 1 = 0 ∧ f a (1 / x0) = 0) :=
sorry

end part_I_part_II_part_III_l366_366030


namespace impossible_to_equalize_sheets_l366_366722

/- Define the conditions -/
variable {α : Type*} [linear_ordered_ring α]

/- The main proposition -/
theorem impossible_to_equalize_sheets (n : ℕ) (sheets : fin n → α)
    (h_diff : ∃ i j, i ≠ j ∧ sheets i ≠ sheets j)
    (h_remove : ∀ k, k ≠ 0 → ∃ i j, i ≠ j ∧ sheets i ≠ sheets j) :
    ¬ (∃ k, ∀ i, sheets i = k) :=
sorry

end impossible_to_equalize_sheets_l366_366722


namespace num_pos_divisors_36_l366_366077

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366077


namespace inclination_angle_of_line_l366_366323

-- Lean definition for the line equation and inclination angle problem
theorem inclination_angle_of_line : 
  ∃ θ : ℝ, (θ ∈ Set.Ico 0 Real.pi) ∧ (∀ x y: ℝ, x + y - 1 = 0 → Real.tan θ = -1) ∧ θ = 3 * Real.pi / 4 :=
sorry

end inclination_angle_of_line_l366_366323


namespace number_of_divisors_36_l366_366084

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366084


namespace ral_current_age_l366_366271

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l366_366271


namespace ant_positions_l366_366337

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  (a + 2 = b) ∧ (b + 2 = c) ∧ (4 * c / c - 2 + 1) = 3 ∧ (4 * c / (c - 4) - 1) = 3

theorem ant_positions (a b c : ℝ) (v : ℝ) (ha : side_lengths a b c) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
by
  sorry

end ant_positions_l366_366337


namespace problem1_problem2_l366_366545

-- Problem 1: Solve the inequality for the function with a = 2
theorem problem1 (x : ℝ) : 
    let a := 2 in 
    let f := λ x, |x + a| + |x + 1/2|
    in f x > 3 ↔ x < -11/4 ∨ x > 1/4 :=
sorry

-- Problem 2: Prove the inequality for the function
theorem problem2 (m a : ℝ) (h : a > 0) :
    let f := λ x, abs (x + a) + abs (x + 1/a)
    in f m + f (-1/m) ≥ 4 :=
sorry

end problem1_problem2_l366_366545


namespace num_pos_divisors_36_l366_366074

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366074


namespace number_of_divisors_of_36_l366_366117

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366117


namespace lcm_18_24_eq_72_l366_366763

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366763


namespace find_specified_time_l366_366866

theorem find_specified_time (x : ℝ)
  (h_dist : 900 = 900)
  (h_slow_time : x + 1 = x + 1)
  (h_fast_time : x - 3 = x - 3)
  (h_speed_rel : (2 : ℝ) * (900 / (x + 1)) = 900 / (x - 3)) :
  (900 / (x + 1)) * 2 = 900 / (x - 3) :=
by
  exact h_speed_rel

end find_specified_time_l366_366866


namespace not_characteristic_of_algorithm_l366_366827

def characteristic_of_algorithm (c : String) : Prop :=
  c = "Abstraction" ∨ c = "Precision" ∨ c = "Finiteness"

theorem not_characteristic_of_algorithm : 
  ¬ characteristic_of_algorithm "Uniqueness" :=
by
  sorry

end not_characteristic_of_algorithm_l366_366827


namespace rectangle_area_l366_366216

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366216


namespace erica_amount_l366_366281

variable (total money : ℕ)
variable (sam amount : ℕ)

theorem erica_amount (total money sam amount : ℕ) (H1 : total = 91) (H2 : sam = 38) : amount = total - sam :=
by
  rw [H1, H2]
  exact dec_trivial

end erica_amount_l366_366281


namespace lcm_18_24_l366_366804

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366804


namespace range_of_a_l366_366147

variable (a : ℝ)
def f (x : ℝ) : ℝ := Real.logBase (3 * a) ((a^2 - 3 * a) * x)

theorem range_of_a (h : ∀ x : ℝ, x < 0 → f a x < f a (x - 1)) : 1 / 3 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l366_366147


namespace lcm_18_24_l366_366788

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366788


namespace number_of_divisors_36_l366_366058

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366058


namespace exists_xy_eq_add_one_and_x4_eq_y4_l366_366995

theorem exists_xy_eq_add_one_and_x4_eq_y4 : 
  ∃ x y : ℝ, x = y + 1 ∧ x^4 = y^4 :=
by
  use [(1 / 2), (-1 / 2)]
  sorry

end exists_xy_eq_add_one_and_x4_eq_y4_l366_366995


namespace Emmanuel_jelly_beans_l366_366896

theorem Emmanuel_jelly_beans :
  ∀ (total : ℕ) (thomas_ratio : ℚ) (ratio_barry : ℕ) (ratio_emmanuel : ℕ),
  total = 200 →
  thomas_ratio = 10 / 100 →
  ratio_barry = 4 →
  ratio_emmanuel = 5 →
  let thomas_share := thomas_ratio * total in
  let remaining := total - thomas_share in
  let total_parts := ratio_barry + ratio_emmanuel in
  let part_value := remaining / total_parts in
  let emmanuel_share := ratio_emmanuel * part_value in
  emmanuel_share = 100 :=
begin
  intros total thomas_ratio ratio_barry ratio_emmanuel h_total h_thomas_ratio h_ratio_barry h_ratio_emmanuel,
  simp [h_total, h_thomas_ratio, h_ratio_barry, h_ratio_emmanuel],
  have ht : thomas_share = 20, by norm_num [h_total, h_thomas_ratio, thomas_share],
  have hr : remaining = 180, by norm_num [remaining, h_total, ht],
  have total_parts : total_parts = 9, by norm_num [h_ratio_barry, h_ratio_emmanuel, total_parts],
  have pv : part_value = 20, by norm_num [part_value, hr, total_parts],
  have es : emmanuel_share = 100, by norm_num [emmanuel_share, h_ratio_emmanuel, pv],
  exact es,
end

end Emmanuel_jelly_beans_l366_366896


namespace envelope_distribution_count_l366_366162

-- Define the constants: number of individuals, number of envelopes, values of envelopes, etc.
constant individuals : Finset ℕ := {0, 1, 2, 3, 4} -- Represents A, B, C, D, E
constant envelopes : Multiset ℕ := {2, 2, 3, 4}   -- The values of the red envelopes

-- Predicate to check if A and B both grab an envelope
def AB_grab_envelopes (AB_grab : ℕ × ℕ) (remaining_envelopes : Multiset ℕ) : Prop :=
  (AB_grab.1 ∈ envelopes) ∧ (AB_grab.2 ∈ envelopes) ∧ (AB_grab.1 ≠ AB_grab.2) ∧ 
  (remaining_envelopes + {AB_grab.1, AB_grab.2} = envelopes)

-- Calculate the number of ways to distribute remaining envelopes among the remaining individuals
def count_remaining_distribution (remaining_envelopes : Multiset ℕ) : ℕ :=
  remaining_envelopes.card.factorial  -- Number of ways to distribute envelopes among 3 remaining people

-- Final theorem to prove the number of valid scenarios
theorem envelope_distribution_count : 
  let n := (individuals.filter (λ x, x ≠ 0 ∧ x ≠ 1)).card in
  let valid_AB_grabs := {(2, 3), (2, 4), (3, 4), (2, 2)} in
  (∑ grab in valid_AB_grabs, count_remaining_distribution (envelopes - {grab.fst, grab.snd})) = 36 :=
by
  sorry


end envelope_distribution_count_l366_366162


namespace largest_possible_A_l366_366373

theorem largest_possible_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 :=
by
  have h3 : A ≤ 10 + 4 := sorry
  exact h3

end largest_possible_A_l366_366373


namespace equation_one_solution_equation_two_solution_l366_366687

theorem equation_one_solution (x : ℝ) (h : 7 * x - 20 = 2 * (3 - 3 * x)) : x = 2 :=
by {
  sorry
}

theorem equation_two_solution (x : ℝ) (h : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1) : x = -1 :=
by {
  sorry
}

end equation_one_solution_equation_two_solution_l366_366687


namespace multiples_of_15_between_25_and_200_l366_366555

theorem multiples_of_15_between_25_and_200 : 
  let S := {x : ℕ | 25 < x ∧ x < 200 ∧ x % 15 = 0} in
  S.card = 12 := 
by 
  sorry

end multiples_of_15_between_25_and_200_l366_366555


namespace janabel_widgets_sold_l366_366611

-- Define the initial conditions of the problem
def first_term : ℕ := 2
def common_difference : ℕ := 4
def number_of_days : ℕ := 15

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_terms (n : ℕ) : ℕ := n * (first_term + nth_term(n)) / 2

-- Theorem stating the sum of the first 15 terms equals 450 widgets
theorem janabel_widgets_sold : sum_terms number_of_days = 450 := by
  sorry

end janabel_widgets_sold_l366_366611


namespace max_value_of_f_l366_366706

def f (x : ℝ) : ℝ := sin x + sin x

theorem max_value_of_f : ∀ x, f(x) = sin x + sin x → ∃ m, m = 2 ∧ ∀ y, f y ≤ m :=
by sorry

end max_value_of_f_l366_366706


namespace common_ratio_infinite_geometric_series_l366_366938

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l366_366938


namespace num_pos_divisors_36_l366_366131

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366131


namespace area_of_rectangle_l366_366201

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366201


namespace alpha_range_l366_366029

-- Given conditions and function
def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

-- Statement to prove the range of α
theorem alpha_range (α : ℝ) (hα : α > 0) (h_range : ∀ x, 0 ≤ x ∧ x ≤ α → 1 ≤ f x ∧ f x ≤ 3 / 2) :
  α ∈ Set.Icc (π / 6) π :=
sorry

end alpha_range_l366_366029


namespace log_sum_implies_value_of_b_l366_366372

theorem log_sum_implies_value_of_b (b : ℝ) (h : 1 / log 3 b + 1 / log 4 b + 1 / log 5 b = 1) : b = 60 :=
sorry

end log_sum_implies_value_of_b_l366_366372


namespace arithmetic_sequence_sum_ratio_l366_366998

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a S : ℕ → ℚ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition_1 (a : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a

def condition_2 (a : ℕ → ℚ) : Prop :=
  (a 5) / (a 3) = 5 / 9

-- Proof statement
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : condition_1 a) (h2 : condition_2 a) (h3 : sum_of_first_n_terms a S) : 
  (S 9) / (S 5) = 1 := 
sorry

end arithmetic_sequence_sum_ratio_l366_366998


namespace find_n_from_exponent_equation_l366_366573

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l366_366573


namespace solution_of_system_l366_366296

def log4 (n : ℝ) : ℝ := log n / log 4

theorem solution_of_system (x y : ℝ) (hx : x + y = 20)
  (hy : log4 x + log4 y = 1 + log4 9) :
  (x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18) :=
by sorry

end solution_of_system_l366_366296


namespace lcm_18_24_l366_366781

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366781


namespace remainder_3001_3005_mod_17_l366_366356

theorem remainder_3001_3005_mod_17 :
  ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17 = 2 := by
  have h1 : 3001 % 17 = 10 := by norm_num
  have h2 : 3002 % 17 = 11 := by norm_num
  have h3 : 3003 % 17 = 12 := by norm_num
  have h4 : 3004 % 17 = 13 := by norm_num
  have h5 : 3005 % 17 = 14 := by norm_num
  calc
    ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17
      = (10 * 11 * 12 * 13 * 14) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 2 : by norm_num

end remainder_3001_3005_mod_17_l366_366356


namespace geometric_mean_of_means_l366_366284

open Finset

-- Define the geometric mean of a set
def geometric_mean (S : Finset ℝ) (h : ∀ x ∈ S, x > 0) : ℝ :=
  (S.prod id) ^ (1 / S.card)

-- Main theorem statement
theorem geometric_mean_of_means (S : Finset ℝ) (h : ∀ x ∈ S, x > 0) :
  geometric_mean S h = geometric_mean (S.powerset.filter (fun T => T ≠ ∅)).product (fun T => geometric_mean T sorry) sorry :=
sorry

end geometric_mean_of_means_l366_366284


namespace centroid_of_triangle_l366_366666

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables {A B C G : V}

theorem centroid_of_triangle (h : A + B + C = 3 * G) : 
  (2/3 : ℝ) • (A + B + C) = G :=
by sorry

end centroid_of_triangle_l366_366666


namespace find_arith_seq_common_diff_l366_366534

-- Let a_n be the nth term of the arithmetic sequence and S_n be the sum of the first n terms
variable {a : ℕ → ℝ} -- arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of first n terms of the sequence

-- Given conditions in the problem
axiom sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3)
axiom sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2)
axiom condition1 : ((S 4) / 12) - ((S 3) / 9) = 1

-- Prove that the common difference d is 6
theorem find_arith_seq_common_diff (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3))
  (sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2))
  (condition1 : (S 4) / 12 - (S 3) / 9 = 1) : 
  d = 6 := 
sorry

end find_arith_seq_common_diff_l366_366534


namespace total_selling_price_l366_366413

theorem total_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (cost3 : ℝ) 
  (profit_percent1 : ℝ) (profit_percent2 : ℝ) (profit_percent3 : ℝ) :
  cost1 = 600 → cost2 = 450 → cost3 = 750 →
  profit_percent1 = 0.08 → profit_percent2 = 0.10 → profit_percent3 = 0.15 →
  (cost1 * (1 + profit_percent1) + cost2 * (1 + profit_percent2) + cost3 * (1 + profit_percent3)) = 2005.50 :=
by
  intros h1 h2 h3 p1 p2 p3
  simp [h1, h2, h3, p1, p2, p3]
  sorry

end total_selling_price_l366_366413


namespace avg_class_size_diff_l366_366420

-- Conditions
def total_students : Nat := 120
def total_teachers : Nat := 6
def class_enrollments : List Nat := [60, 30, 20, 5, 3, 2]

-- Definitions for average class sizes
def t : Float := (class_enrollments.sum.toFloat) / total_teachers.toFloat 
def s : Float := (class_enrollments.map (λ n, n * (n.toFloat / total_students.toFloat))).sum

-- Statement to prove
theorem avg_class_size_diff : t - s = -21.148 := by
  sorry

end avg_class_size_diff_l366_366420


namespace parallel_lines_l366_366892

theorem parallel_lines
  (Γ₁ Γ₂ : Circle)
  (l₁ l₂ : Line)
  (A B₁ C₁ D A₁ B C D₁ : Point)
  (HΓ₁ : A ∈ Γ₁ ∧ B₁ ∈ Γ₁ ∧ C₁ ∈ Γ₁ ∧ D ∈ Γ₁)
  (HΓ₂ : A₁ ∈ Γ₂ ∧ B ∈ Γ₂ ∧ C ∈ Γ₂ ∧ D₁ ∈ Γ₂)
  (Hl₁ : l₁ = LineThroughPoints A D ∧ l₁ = LineThroughPoints B₁ C₁)
  (Hl₂ : l₂ = LineThroughPoints A₁ D₁ ∧ l₂ = LineThroughPoints B C)
  (Hangles1 : ∠ABC = ∠A₁B₁C₁)
  (Hangles2 : ∠BAC = ∠B₁A₁C₁)
  (Hangles3 : ∠ACD = ∠A₁C₁D₁) :
  (Parallel A₁ A D D₁ ∧ Parallel B B₁ C C₁) :=
by
  sorry

end parallel_lines_l366_366892


namespace number_of_divisors_36_l366_366052

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366052


namespace find_n_from_exponent_equation_l366_366572

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l366_366572


namespace miquel_theorem_l366_366948

open EuclideanGeometry

/-- Definition of circumcircles intersecting at a common point (the Miquel point) -/
noncomputable def miquel_point (A B C D E F : Point) : Prop :=
  ∃ P, 
    P ∈ circumcircle A B C ∧
    P ∈ circumcircle C E F ∧
    P ∈ circumcircle B D F ∧
    P ∈ circumcircle A D E

/-- Centers of the circumcircles lying on a single circle passing through the Miquel point -/
noncomputable def centers_on_circle (A B C D E F : Point) : Prop :=
  ∃ O1 O2 O3 O4 : Point,
    O1 = circumcenter A B C ∧
    O2 = circumcenter C E F ∧
    O3 = circumcenter B D F ∧
    O4 = circumcenter A D E ∧
    ∃ P, 
      P ∈ circumcircle A B C ∧
      P ∈ circumcircle C E F ∧
      P ∈ circumcircle B D F ∧
      P ∈ circumcircle A D E ∧
      O1 ≠ O2 ∧ O1 ≠ O3 ∧ O1 ≠ O4 ∧ 
      O2 ≠ O3 ∧ O2 ≠ O4 ∧ 
      O3 ≠ O4 ∧ 
      collinear ({O1, O2, O3, O4} : Set Point) ∧
      P ∈ circumcircle O1 O2 O3 ∧
      P ∈ circumcircle O3 O4 O1

theorem miquel_theorem (A B C D E F : Point) :
  miquel_point A B C D E F ↔ centers_on_circle A B C D E F := 
by sorry

end miquel_theorem_l366_366948


namespace num_pos_divisors_36_l366_366063

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366063


namespace binom_21_14_l366_366004

theorem binom_21_14 :
  nat.choose 21 14 = 116280 :=
begin
  have binom_20_13 : nat.choose 20 13 = 77520 := rfl,
  have binom_20_14 : nat.choose 20 14 = 38760 := rfl,
  rw [← nat.add_sub_assoc (le_of_lt (by norm_num : 13 < 21)),
      nat.add_sub_assoc (le_refl 20),
      ← nat.add_sub_assoc (le_of_lt (by norm_num : 13 < 20))]
    at binom_20_13,
  exact binom_add_one_one 20 13,
  sorry
end

end binom_21_14_l366_366004


namespace probability_red_or_green_l366_366848

variable (P_brown P_purple P_green P_red P_yellow : ℝ)

def conditions : Prop :=
  P_brown = 0.3 ∧
  P_brown = 3 * P_purple ∧
  P_green = P_purple ∧
  P_red = P_yellow ∧
  P_brown + P_purple + P_green + P_red + P_yellow = 1

theorem probability_red_or_green (h : conditions P_brown P_purple P_green P_red P_yellow) :
  P_red + P_green = 0.35 :=
by
  sorry

end probability_red_or_green_l366_366848


namespace equivalent_product_lists_l366_366923

-- Definitions of the value assigned to each letter.
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- We only care about uppercase letters A-Z

def list_product (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * (letter_value c)) 1

-- Given the list MNOP with their products equals letter values.
def MNOP := ['M', 'N', 'O', 'P']
def BJUZ := ['B', 'J', 'U', 'Z']

-- Lean statement to assert the equivalence of the products.
theorem equivalent_product_lists :
  list_product MNOP = list_product BJUZ :=
by
  sorry

end equivalent_product_lists_l366_366923


namespace find_n_l366_366570

theorem find_n (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  unfold pow at h
  sorry

end find_n_l366_366570


namespace count_numbers_in_list_l366_366045

theorem count_numbers_in_list : 
  ∃ (n : ℕ), (list.range n).map (λ k, 165 - 5 * k) = [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45] ∧ n = 25 :=
by {
  sorry
}

end count_numbers_in_list_l366_366045


namespace remainder_when_divided_by_4x_minus_8_l366_366365

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l366_366365


namespace angles_sum_is_210_l366_366222

-- Define the measure of an interior angle of a regular polygon
def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

-- Define the measure of angle ABC in a regular hexagon
def angle_ABC := interior_angle 6

-- Define the measure of angle ABD in a regular square
def angle_ABD := interior_angle 4

-- The sum of the measures of angles ABC and ABD
def sum_of_angles : ℝ := angle_ABC + angle_ABD

-- Proof that the sum of the measures of angles ABC and ABD is 210 degrees
theorem angles_sum_is_210 : sum_of_angles = 210 := by
  -- Just to make sure the term below is recognized
  unfold angle_ABC angle_ABD sum_of_angles interior_angle
  -- This is to avoid reduction errors, would typically not be required here
  simp
  sorry

end angles_sum_is_210_l366_366222


namespace men_handshakes_l366_366336

theorem men_handshakes (n : ℕ) (h : n * (n - 1) / 2 = 435) : n = 30 :=
sorry

end men_handshakes_l366_366336


namespace multiples_of_15_between_25_and_200_l366_366556

theorem multiples_of_15_between_25_and_200 : 
  let S := {x : ℕ | 25 < x ∧ x < 200 ∧ x % 15 = 0} in
  S.card = 12 := 
by 
  sorry

end multiples_of_15_between_25_and_200_l366_366556


namespace mutually_exclusive_events_l366_366600

structure Group :=
  (boys : ℕ)
  (girls : ℕ)

def selection_event_A (g : Group) (selection : finset (ℕ × string)) : Prop :=
    (∃ boy, boy ∈ selection ∧ boy.2 = "girl") ∧
    (∃ girl1 girl2, girl1 ≠ girl2 ∧ girl1 ∈ selection ∧ girl2 ∈ selection ∧ girl1.2 = "girl" ∧ girl2.2 = "girl")

def selection_event_D (g : Group) (selection : finset (ℕ × string)) : Prop :=
    (∃ girl, girl ∈ selection ∧ girl.2 = "girl") ∧
    (∀ boy, boy ∈ selection → boy.2 = "boy")

theorem mutually_exclusive_events (g : Group) (selection : finset (ℕ × string)) :
  selection_event_A g selection ∨ selection_event_D g selection → 
  ¬(selection_event_A g selection ∧ selection_event_D g selection) :=
by
  sorry

end mutually_exclusive_events_l366_366600


namespace division_remainder_l366_366368

def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30
def d (x : ℝ) : ℝ := 4 * x - 8

theorem division_remainder : (∃ q r, p(2) = d(2) * q + r ∧ d(2) ≠ 0 ∧ r = 10) :=
by
  sorry

end division_remainder_l366_366368


namespace num_pos_divisors_36_l366_366134

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366134


namespace rain_probability_l366_366329

theorem rain_probability :
  let p := (3:ℚ) / 4 in
  let q :=  1 - p in
  let prob_no_rain_four_days := q ^ 4 in
  let prob_rain_at_least_once := 1 - prob_no_rain_four_days in
  prob_rain_at_least_once = 255 / 256 :=
by
  sorry

end rain_probability_l366_366329


namespace range_of_a_l366_366585

theorem range_of_a (a : ℝ) : (0 < a ∧ a ≤ Real.exp 1) ↔ ∀ x : ℝ, 0 < x → a * Real.log (a * x) ≤ Real.exp x := 
by 
  sorry

end range_of_a_l366_366585


namespace john_needs_3_basses_l366_366613

def john_number_of_basses (B : ℕ) : Prop :=
  let bass_strings := 4 * B
  let guitar_strings := 6 * (2 * B)
  let eight_string_guitar_count := 2 * B - 3
  let eight_string_guitar_strings := 8 * eight_string_guitar_count
  bass_strings + guitar_strings + eight_string_guitar_strings = 72

theorem john_needs_3_basses :
  ∃ B : ℕ, john_number_of_basses B ∧ B = 3 :=
by
  use 3
  unfold john_number_of_basses
  simp
  sorry

end john_needs_3_basses_l366_366613


namespace length_of_PS_l366_366223

-- Define the problem conditions directly
variables (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
           (PQ : Real) (PR : Real) (cosP : Real)
           (angleP : ℝ)

-- Assume the given conditions
variables (h1 : PQ = 4) (h2 : PR = 8) (h3 : cosP = 1/4)
           (PQR_is_triangle : Triangle P Q R) (PS_bisects_angle_P : Bisects PS angleP)

-- The goal is to prove the specified length of PS
theorem length_of_PS : PQ = 4 → PR = 8 → cosP = 1 / 4 → ∃ PS : ℝ, PS = 4 := by
  sorry

end length_of_PS_l366_366223


namespace five_numbers_property_l366_366829

theorem five_numbers_property :
  let S := {1680, 1692, 1694, 1695, 1696} in
  ∀ (a b ∈ S), a > b → a % (a - b) = 0 := by
  sorry

end five_numbers_property_l366_366829


namespace max_number_of_cubes_l366_366817

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end max_number_of_cubes_l366_366817


namespace volume_of_tetrahedron_l366_366535

theorem volume_of_tetrahedron 
  (A B C D O : Type) 
  (inscribed : ∀ (P : Type), P ∈ {A, B, C, D} → ∃ (r : ℝ), ∀ (P : Type), |P - O| = r)
  (diameter_AD : ∀ (P : Type), P = D → |A - D| = 2 * r)
  (equilateral_ABC : ∀ (e : ℝ > 0), e = 1 → |A - B| = e ∧ |B - C| = e ∧ |C - A| = e)
  (equilateral_BCD : ∀ (e : ℝ > 0), e = 1 → |B - C| = e ∧ |C - D| = e ∧ |D - B| = e) :
  volume_of_tetrahedron A B C D = (√3) / 12 :=
by
  sorry

end volume_of_tetrahedron_l366_366535


namespace parallelogram_area_l366_366438

variable (a b : ℝ^3)
variable (area_ab : ℝ) (h : area_ab = 12)

theorem parallelogram_area :
  ∥(3 • a + 2 • b) × (4 • a - 6 • b)∥ = 312 :=
by
  sorry

end parallelogram_area_l366_366438


namespace area_of_ABCD_l366_366185

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366185


namespace num_pos_divisors_36_l366_366069

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366069


namespace max_stamps_l366_366588

theorem max_stamps (price_per_stamp : ℕ) (available_money : ℕ) (h1 : price_per_stamp = 25) (h2 : available_money = 5000) :
  ∃ (n : ℕ), n = 200 ∧ price_per_stamp * n ≤ available_money ∧ ∀ m, (price_per_stamp * m ≤ available_money) → m ≤ 200 :=
by
  use 200
  split; sorry

end max_stamps_l366_366588


namespace elmer_fuel_cost_savings_l366_366925

theorem elmer_fuel_cost_savings (fuel_eff_prev : ℝ) (cost_gasoline : ℝ) :
  let fuel_eff_new := 1.6 * fuel_eff_prev
      cost_diesel := 1.3 * cost_gasoline
      cost_prev := cost_gasoline
      cost_new := (5 / 8) * cost_diesel
      savings := cost_prev - cost_new 
      savings_percentage := (savings / cost_prev) * 100
  in savings_percentage = 18.75 := by
  sorry

end elmer_fuel_cost_savings_l366_366925


namespace lcm_18_24_eq_72_l366_366810

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366810


namespace lcm_18_24_l366_366755
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366755


namespace median_of_data_set_is_4_l366_366871

/-- A type alias representing our data set. -/
def DataSet := List ℝ

/-- Definition of the average of a list of real numbers. -/
def average (l : DataSet) : ℝ := (l.sum) / (l.length : ℝ)

/-- Definition of the median of a list of real numbers. Note: assumes the list is sorted. -/
def median (l : DataSet) : ℝ :=
if h : l.length % 2 = 1 then l.nthLe (l.length / 2) sorry
else (l.nthLe (l.length / 2 - 1) sorry + l.nthLe (l.length / 2) sorry) / 2

theorem median_of_data_set_is_4 :
  ∀ (x : ℝ), average [1, 2, 3, x, 5, 5] = 4 → median ([1, 2, 3, x, 5, 5].insertion_sort) = 4 :=
by
  intros x h_avg
  sorry

end median_of_data_set_is_4_l366_366871


namespace larger_number_l366_366701

theorem larger_number (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (hcf_eq : hcf = 23) (fact1_eq : factor1 = 13) (fact2_eq : factor2 = 14) : 
  max (hcf * factor1) (hcf * factor2) = 322 := 
by
  sorry

end larger_number_l366_366701


namespace area_ratio_l366_366416

variable {A B C A1 C1 : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]
          [LinearOrder A1] [LinearOrder C1]
          (AC BC : ℝ) (S_ABC : ℝ)
          (h1 : A1 / AC = 0.85)
          (h2 : BC1 / BC = 1.2)

theorem area_ratio :
  (1.2 * 0.85 * S_ABC / S_ABC = 1.02) :=
by 
  sorry

end area_ratio_l366_366416


namespace cost_price_equation_l366_366852

theorem cost_price_equation (x : ℝ) (markup : ℝ) (discount : ℝ) (profit : ℝ) 
  (h_markup : markup = 0.3 * x) 
  (h_discount : discount = 0.1 * (1 + 0.3) * x) 
  (h_profit : profit = 34) :
  0.9 * (1 + 0.3) * x - x = 34 :=
begin
  sorry
end

end cost_price_equation_l366_366852


namespace martin_minimum_fourth_score_l366_366652

/-- Prove that if Martin's scores for the first three quarters are 80%, 78%, and 84%, then the 
minimum score Martin must achieve in the fourth quarter to have an average of at least 85% is 98%. -/
theorem martin_minimum_fourth_score 
  (s₁ s₂ s₃ : ℕ)
  (h1 : s₁ = 80)
  (h2 : s₂ = 78)
  (h3 : s₃ = 84) 
  (average_requirement : ℕ := 85) 
  (quarters : ℕ := 4) :
  let total_needed := quarters * average_requirement in
  let sum_first_three := s₁ + s₂ + s₃ in
  let score_fourth := total_needed - sum_first_three in
  score_fourth = 98 :=
by {
  sorry
}

end martin_minimum_fourth_score_l366_366652


namespace linear_regression_passes_through_sample_center_l366_366705

variables {x y : Type} [Real x] [Real y]
variables {b a : Real}
variables (x̄ ȳ : Real)
variables (ŷ : Real → Real)

-- Definition of the linear regression equation
def regression_equation (x : Real) : Real := b * x + a

-- Condition: given a = ȳ - b * x̄
def a_def : Real := ȳ - b * x̄

-- Main theorem statement
theorem linear_regression_passes_through_sample_center :
  regression_equation b (λ x, b * x + a_def) x̄ = ȳ := by
  sorry

end linear_regression_passes_through_sample_center_l366_366705


namespace greatest_ribbon_length_l366_366430

-- Define lengths of ribbons
def ribbon_lengths : List ℕ := [8, 16, 20, 28]

-- Condition ensures gcd and prime check
def gcd_is_prime (n : ℕ) : Prop :=
  ∃ d : ℕ, (∀ l ∈ ribbon_lengths, d ∣ l) ∧ Prime d ∧ n = d

-- Prove the greatest length that can make the ribbon pieces, with no ribbon left over, is 2
theorem greatest_ribbon_length : ∃ d, gcd_is_prime d ∧ ∀ m, gcd_is_prime m → m ≤ 2 := 
sorry

end greatest_ribbon_length_l366_366430


namespace common_ratio_infinite_geometric_series_l366_366937

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l366_366937


namespace smallest_number_of_disks_l366_366476

-- Definitions for the problem conditions
def file_sizes : List ℝ := List.replicate 5 0.9 ++ List.replicate 15 0.75 ++ List.replicate 20 0.5
def disk_size : ℝ := 1.44
def number_of_files : ℕ := 40

-- Problem statement to prove
theorem smallest_number_of_disks :
  ∃ n, n ≤ 20 ∧ (∀ files_in_disks : List (List ℝ), 
    (∀ disk ∈ files_in_disks, ∑ size in disk, size ≤ disk_size) → 
    ∑ disk in files_in_disks, disk.length = number_of_files → 
    files_in_disks.length = n) :=
sorry

end smallest_number_of_disks_l366_366476


namespace count_even_vs_odd_blocks_l366_366619

def isPositiveInteger (n : ℕ) : Prop := n > 0

def countWordsWithBlocks (n : ℕ) (letters : Finset Char) (ME_blocks MO_blocks : ℕ → Prop) : ℕ :=
  -- Placeholder for the actual function
  sorry


theorem count_even_vs_odd_blocks (n : ℕ) (letters : Finset Char := {'M', 'E', 'O'}) :
  isPositiveInteger n →
  let a := countWordsWithBlocks n letters (λ x, x % 2 = 0) (λ y, y % 2 = 0) in
  let b := countWordsWithBlocks n letters (λ x, x % 2 = 1) (λ y, y % 2 = 1) in
  a > b :=
begin
  intros h,
  -- The proof logic is omitted for brevity
  sorry,
end

end count_even_vs_odd_blocks_l366_366619


namespace lcm_18_24_eq_72_l366_366813

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366813


namespace sum_of_a_plus_b_l366_366264

theorem sum_of_a_plus_b : 
  ∑ (a : ℕ) in {a | ∃ b : ℕ, b * 2^a = 1000} ∪ {b | ∃ a : ℕ, b * 2^a = 1000}, a + b = 881 :=
by
  sorry

end sum_of_a_plus_b_l366_366264


namespace find_face_value_of_bond_l366_366618

theorem find_face_value_of_bond 
  (F : ℝ) 
  (S : ℝ := 4615.384615384615) 
  (h1 : ∃ I : ℝ, I = 0.06 * F) 
  (h2 : ∃ I' : ℝ, I' ≈ 0.065 * S)
  (h3 : I' = I)  
  : F = 5000 :=
by
  sorry

end find_face_value_of_bond_l366_366618


namespace round_3456_to_nearest_hundredth_l366_366670

theorem round_3456_to_nearest_hundredth : Real :=
  let x := 3.456
  prove round_to_nearest_hundredth x = 3.46 by sorry

end round_3456_to_nearest_hundredth_l366_366670


namespace lcm_18_24_l366_366760
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366760


namespace area_of_rectangle_ABCD_l366_366182

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366182


namespace tangents_to_circumcircle_EOF_l366_366259

noncomputable def parallelogram (A B C D : Type*) [add_comm_group A] [vector_space ℝ A] :=
D - C = B - A ∧
C - B = D - A

noncomputable def is_tangent (P Q R : Type*) [inner_product_space ℝ P] :=
let circle_circum := circumcenter P Q R in
let radius := dist circle_circum P in
dist Q circle_circum = radius ∧ dist R circle_circum = radius

variable {A B C D E F O : Type*}

variables [inner_product_space ℝ A] [inner_product_space ℝ B]
  [inner_product_space ℝ C] [inner_product_space ℝ D] 
  [inner_product_space ℝ E] [inner_product_space ℝ F] 
  [inner_product_space ℝ O]

-- Given conditions
variable (h_parallelogram : parallelogram A B C D)
variable (h_points_on_side : ∃ E F, E ∈ line_segment ℝ B F ∧ F ∈ line_segment ℝ B C)
variable (h_diagonals_intersect : ∃ O, affine_independent ℝ ![{A, C, B, D}] O)
variable (h_tangent_AOD : is_tangent A E O ∧ is_tangent D F O)

-- Prove
theorem tangents_to_circumcircle_EOF :
  is_tangent E O F :=
sorry

end tangents_to_circumcircle_EOF_l366_366259


namespace A_days_is_10_l366_366399

noncomputable def workRate_A (A_days : ℝ) : ℝ := 1 / A_days
def workRate_B : ℝ := 1 / 15
def totalWages : ℝ := 3500
def wages_A : ℝ := 2100
def wages_B : ℝ := totalWages - wages_A
def combined_work_rate (A_days : ℝ) : ℝ := workRate_A A_days + workRate_B
def A_work_share (A_days : ℝ) : ℝ := workRate_A A_days / combined_work_rate A_days
def correct_wage_ratio : ℝ := wages_A / totalWages

theorem A_days_is_10 : ∃ A_days : ℝ, A_work_share A_days = correct_wage_ratio ∧ A_days = 10 :=
by 
  sorry

end A_days_is_10_l366_366399


namespace area_of_ABCD_l366_366189

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366189


namespace num_pos_divisors_36_l366_366064

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366064


namespace num_positive_divisors_36_l366_366091

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366091


namespace meaningful_sqrt_range_l366_366590

theorem meaningful_sqrt_range (x : ℝ) (h : 0 ≤ x + 3) : -3 ≤ x :=
by sorry

end meaningful_sqrt_range_l366_366590


namespace range_of_m_l366_366027

variable {x t : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x - 3

def g (x : ℝ) (m : ℝ) (a : ℝ) : ℝ :=
  x^3 + (Real.log x - a * x + 2 + m / 2) * x^2 - 2 * x

theorem range_of_m (a : ℝ) (h : a ≠ 0) (slope_cond : ((-a) / 2 = 1)) (g_mono_cond : ∀ t : ℝ, 1 ≤ t ∧ t ≤ 2 → ¬(∀ x : ℝ, t < x ∧ x < 3 → (3 * x^2 + (m + 4) * x - 2) ≥ 0)) :
  -37/3 < m ∧ m < -9 := 
sorry

end range_of_m_l366_366027


namespace simplify_fraction_l366_366681

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by
  sorry

end simplify_fraction_l366_366681


namespace num_pos_divisors_36_l366_366125

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366125


namespace lcm_18_24_l366_366746

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366746


namespace simplify_polynomial_l366_366683

variable (x : ℝ)

theorem simplify_polynomial :
  (6*x^10 + 8*x^9 + 3*x^7) + (2*x^12 + 3*x^10 + x^9 + 5*x^7 + 4*x^4 + 7*x + 6) =
  2*x^12 + 9*x^10 + 9*x^9 + 8*x^7 + 4*x^4 + 7*x + 6 :=
by
  sorry

end simplify_polynomial_l366_366683


namespace correct_number_of_propositions_l366_366945

-- Define the four propositions as separate definitions

def proposition1 (a b c : ℝ) (hc_ne_zero : c ≠ 0) : Prop :=
  a * c^2 > b * c^2 → a > b

def proposition2 (a b c d : ℝ) (hgt_ab : a > b) (hgt_cd : c > d) : Prop :=
  a + c > b + d

def proposition3 (a b c d : ℝ) (hgt_ab : a > b) (hgt_cd : c > d) : Prop :=
  a * c > b * d

def proposition4 (a b : ℝ) (hgt_ab : a > b) : Prop :=
  1 / a > 1 / b

-- Statement to prove the number of correct propositions

theorem correct_number_of_propositions (a b c d : ℝ) (hc_ne_zero : c ≠ 0) 
  (hgt_ab : a > b) (hgt_cd : c > d) : 
  (proposition1 a b c hc_ne_zero) ∧ 
  (proposition2 a b c d hgt_ab hgt_cd) ∧ 
  ¬ (proposition3 a b c d hgt_ab hgt_cd) ∧ 
  ¬ (proposition4 a b hgt_ab) → 
  2 := 
sorry

end correct_number_of_propositions_l366_366945


namespace eqB_is_quadratic_l366_366381

-- Define the equations given in the problem
def eqA (a b c : ℝ) : ℝ → Prop := λ x, a * x^2 + b * x + c = 0
def eqB : ℝ → Prop := λ x, x^2 = 0
def eqC : ℝ → Prop := λ x, (x + 1) * (x - 1) = x^2 + 2 * x
def eqD : ℝ → Prop := λ x, x + 1 / x = 2

-- Proposition stating that eqB is a quadratic equation
theorem eqB_is_quadratic : ∀ x : ℝ, eqB x → ∃ a b c : ℝ, eqB x = (λ x, a * x^2 + b * x + c = 0) ∧ a ≠ 0 :=
by
  sorry

end eqB_is_quadratic_l366_366381


namespace problem_G2_1_l366_366577

theorem problem_G2_1 (a : ℕ) : (137 / a = 0.1 + 0.0234 + 0.0000234 * repeating_unit) -> a = 1110 := sorry

end problem_G2_1_l366_366577


namespace num_positive_divisors_36_l366_366099

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366099


namespace jesse_blocks_total_l366_366231

-- Define the number of building blocks used for each structure and the remaining blocks
def blocks_building : ℕ := 80
def blocks_farmhouse : ℕ := 123
def blocks_fenced_in_area : ℕ := 57
def blocks_left : ℕ := 84

-- Prove that the total number of building blocks Jesse started with is 344
theorem jesse_blocks_total : blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left = 344 :=
by
  calc
    blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left
      = 80 + 123 + 57 + 84 : by refl
  ... = 260 + 84 : by simp
  ... = 344 : by norm_num

end jesse_blocks_total_l366_366231


namespace area_of_ABCD_l366_366184

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366184


namespace resident_A_water_fee_resident_B_water_usage_resident_C_water_usage_l366_366426

-- Define the water fee function
def water_fee (usage : ℝ) : ℝ :=
  if usage ≤ 10 then usage * 1.5 else 10 * 1.5 + (usage - 10) * 2

-- Define the water usage function
def water_usage (fee : ℝ) : ℝ :=
  if fee ≤ 15 then fee / 1.5 else 10 + (fee - 15) / 2

-- Theorems to state mathematically equivalent proof problems

-- 1. Resident A used 8 cubic meters, the water fee is 12
theorem resident_A_water_fee : water_fee 8 = 12 := sorry

-- 2. Resident B paid 22.8 yuan, the water usage is 13.9 cubic meters
theorem resident_B_water_usage : water_usage 22.8 = 13.9 := sorry

-- 3. Resident C paid m yuan, the water usage is either m/1.5 or 10 + (m - 15)/2
theorem resident_C_water_usage (m : ℝ) : water_usage m = if m ≤ 15 then m / 1.5 else 10 + (m - 15) / 2 := sorry

end resident_A_water_fee_resident_B_water_usage_resident_C_water_usage_l366_366426


namespace chessboard_disk_cover_l366_366404

noncomputable def chessboardCoveredSquares : ℕ :=
  let D : ℝ := 1 -- assuming D is a positive real number; actual value irrelevant as it gets cancelled in the comparison
  let grid_size : ℕ := 8
  let total_squares : ℕ := grid_size * grid_size
  let boundary_squares : ℕ := 28 -- pre-calculated in the insides steps
  let interior_squares : ℕ := total_squares - boundary_squares
  let non_covered_corners : ℕ := 4
  interior_squares - non_covered_corners

theorem chessboard_disk_cover : chessboardCoveredSquares = 32 := sorry

end chessboard_disk_cover_l366_366404


namespace max_a_for_f_decreasing_l366_366578

noncomputable def f (x : ℝ) := Real.cos x - Real.sin x

theorem max_a_for_f_decreasing :
  (∀ x y ∈ [-a, a], x < y → f x ≥ f y) → a ≤ π / 4 :=
sorry

end max_a_for_f_decreasing_l366_366578


namespace lcm_18_24_eq_72_l366_366773

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366773


namespace find_range_of_norm_l366_366161

variable {V : Type*} [inner_product_space ℝ V]

theorem find_range_of_norm (a e : V) 
  (h1 : ∥e∥ = 1) 
  (h2 : inner a e = 2) 
  (h3 : ∀ t : ℝ, ∥a∥^2 ≤ 5 * ∥a + t • e∥) : 
  sqrt 5 ≤ ∥a∥ ∧ ∥a∥ ≤ 2 * sqrt 5 :=
by
  -- Proof omitted
  sorry

end find_range_of_norm_l366_366161


namespace sqrt_sum_simplified_l366_366898

noncomputable def sqrt_sum : ℝ := sqrt 50 + sqrt 32 + sqrt 24

theorem sqrt_sum_simplified : sqrt_sum = 9 * sqrt 2 + 2 * sqrt 6 := 
sorry

end sqrt_sum_simplified_l366_366898


namespace number_of_divisors_of_36_l366_366110

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366110


namespace range_of_a_l366_366983

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x^2 - (x - 3) * exp(x) + 1

def has_two_extreme_points_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
               (∀ x, x1 ≠ x → f a x1 = f a x ∨ f a x2 = f a x)

theorem range_of_a (a : ℝ) : 
  has_two_extreme_points_in_interval a → a ∈ (Set.Ioo (Real.exp 1 / 3) (Real.exp 2 / 6)) :=
by
  sorry

end range_of_a_l366_366983


namespace fixed_point_of_function_l366_366532

def f (a x : ℝ) : ℝ := 4 + 2 * a^(x - 1)

theorem fixed_point_of_function (a : ℝ) : f a 1 = 6 := by
  calculate
  have h1 : a^(1 - 1) = 1, by exact (pow_one a).subst rfl
  show f a 1 = 6
    by simp [f, h1]
  sorry

end fixed_point_of_function_l366_366532


namespace find_rate_of_interest_l366_366689

theorem find_rate_of_interest (
    (A : ℝ) (interest_earned : ℝ) (years : ℕ) (P : ℝ) (r : ℝ)
    (hA : A = 2646)
    (h_interest : interest_earned = 246)
    (hP : P = A - interest_earned)
    (h_years : years = 2)
    (h_formula : A = P * (1 + r / 100) ^ years)
  ) : r = 5 :=
by
  sorry

end find_rate_of_interest_l366_366689


namespace smallest_n_l366_366834

theorem smallest_n 
  {n : ℕ}
  (total_apples : ℕ)
  (red_apples : ℕ)
  (prob_less_than_half : ∏ k in Finset.range n, (red_apples - k) / (total_apples - k) < 0.5) :
  n = 13 :=
by
  let total_apples := 13
  let red_apples := 12
  have h : ∏ k in Finset.range 13, (red_apples - k) / (total_apples - k) < 0.5 := sorry
  exact h

end smallest_n_l366_366834


namespace p_iff_q_l366_366958

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end p_iff_q_l366_366958


namespace num_pos_divisors_36_l366_366129

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366129


namespace area_of_triangle_l366_366012

noncomputable def semi_major_axis := 2
noncomputable def semi_minor_axis := real.sqrt 3
noncomputable def distance_to_focus := 1

-- Point P is on the ellipse x^2/4 + y^2/3 = 1
def on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1

-- Foci of the ellipse
def left_focus := (-distance_to_focus, 0)
def right_focus := (distance_to_focus, 0)

-- Angle ∠F1 P F2 = π / 3
def angle_F1_P_F2 (P : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a = real.pi / 3

-- The area of the triangle F1 P F2
def area_triangle (F1 P F2 : ℝ × ℝ) : ℝ :=
  1 / 2 * real.dist F1 P * real.dist P F2 * real.sin (real.pi / 3)

theorem area_of_triangle (P : ℝ × ℝ) (hP : on_ellipse P) (hangle: angle_F1_P_F2 P) :
  area_triangle left_focus P right_focus = real.sqrt 3 :=
sorry

end area_of_triangle_l366_366012


namespace chess_matches_total_l366_366942

/-- Suppose five students play chess matches against each other. 
Each student plays three matches against each of the other students. 
Prove that the total number of matches played is 30. -/
theorem chess_matches_total (num_students : ℕ) (matches_per_pair : ℕ) 
  (total_matches : ℕ) (total_pairs : ℕ)
  (h1 : num_students = 5) 
  (h2 : matches_per_pair = 3) 
  (h3 : total_pairs = nat.choose num_students 2) 
  (h4 : total_matches = total_pairs * matches_per_pair) :
  total_matches = 30 :=
by {
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end chess_matches_total_l366_366942


namespace rational_root_count_l366_366913

theorem rational_root_count {b_4 b_3 b_2 b_1 : ℚ} :
  (∀ x : ℚ, 16 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + 24 = 0 → 
    (∃ p ∈ {±1, ±2, ±3, ±4, ±6, ±8, ±12, ±24}, ∃ q ∈ {±1, ±2, ±4, ±8, ±16}, x = p/q)) → 
  (finset.card (((finset.image (λ p, finset.image (λ q, p / q) {±1, ±2, ±4, ±8, ±16}).bUnion {±1, ±2, ±3, ±4, ±6, ±8, ±12, ±24}).finset))).card = 40 :=
sorry

end rational_root_count_l366_366913


namespace num_positive_divisors_36_l366_366090

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366090


namespace number_of_terms_in_sequence_l366_366041

-- Definition of the arithmetic sequence parameters
def start_term : Int := 165
def end_term : Int := 45
def common_difference : Int := -5

-- Define a theorem to prove the number of terms in the sequence
theorem number_of_terms_in_sequence :
  ∃ n : Nat, number_of_terms 165 45 (-5) = 25 :=
by
  sorry

end number_of_terms_in_sequence_l366_366041


namespace num_pos_divisors_36_l366_366127

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366127


namespace michaels_brother_final_amount_l366_366657

theorem michaels_brother_final_amount :
  ∀ (michael_money michael_brother_initial michael_give_half candy_cost money_left : ℕ),
  michael_money = 42 →
  michael_brother_initial = 17 →
  michael_give_half = michael_money / 2 →
  let michael_brother_total := michael_brother_initial + michael_give_half in
  candy_cost = 3 →
  money_left = michael_brother_total - candy_cost →
  money_left = 35 :=
by
  intros michael_money michael_brother_initial michael_give_half candy_cost money_left
  intros h1 h2 h3 michael_brother_total h4 h5
  sorry

end michaels_brother_final_amount_l366_366657


namespace remainder_when_divided_by_4x_minus_8_l366_366367

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l366_366367


namespace solve_for_x_l366_366941

theorem solve_for_x : ∃ x : ℝ, ( (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 4) ∧ x = -15 :=
by {
  let x := -15,
  existsi x,
  split,
  sorry,
  refl,
}

end solve_for_x_l366_366941


namespace calculate_value_l366_366441

theorem calculate_value : 15 * (1 / 3) + 45 * (2 / 3) = 35 := 
by
simp -- We use simp to simplify the expression
sorry -- We put sorry as we are skipping the full proof

end calculate_value_l366_366441


namespace round_3456_to_nearest_hundredth_l366_366668

theorem round_3456_to_nearest_hundredth : Real :=
  let x := 3.456
  prove round_to_nearest_hundredth x = 3.46 by sorry

end round_3456_to_nearest_hundredth_l366_366668


namespace sum_frac_ineq_l366_366970

variable (n : Nat) (h_n : n > 3) (x : Fin n → Real) 
variable (h_pos : ∀ i, 0 < x i) (h_prod_one : ∏ i, x i = 1)

theorem sum_frac_ineq : (∑ i, 1 / (1 + x i + x i * x ((i + 1) % n))) > 1 :=
sorry

end sum_frac_ineq_l366_366970


namespace height_of_second_rectangle_l366_366727

theorem height_of_second_rectangle :
  ∃ h : ℝ, (let area1 := 4 * 5 in
            let area2 := 3 * h in
            area1 = area2 + 2) ↔ h = 6 :=
by
  sorry

end height_of_second_rectangle_l366_366727


namespace locus_of_point_P_l366_366979

theorem locus_of_point_P (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  (x^2 + y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2) ↔ 
  ((x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 ∧ x ≠ 2 ∧ x ≠ -2) :=
by
  sorry 

end locus_of_point_P_l366_366979


namespace combined_score_is_75_l366_366434

variable (score1 : ℕ) (total1 : ℕ)
variable (score2 : ℕ) (total2 : ℕ)
variable (score3 : ℕ) (total3 : ℕ)

-- Conditions: Antonette's scores and the number of problems in each test
def Antonette_scores : Prop :=
  score1 = 60 * total1 / 100 ∧ total1 = 15 ∧
  score2 = 85 * total2 / 100 ∧ total2 = 20 ∧
  score3 = 75 * total3 / 100 ∧ total3 = 25

-- Theorem to prove the combined score is 75% (45 out of 60) rounded to the nearest percent
theorem combined_score_is_75
  (h : Antonette_scores score1 total1 score2 total2 score3 total3) :
  100 * (score1 + score2 + score3) / (total1 + total2 + total3) = 75 :=
by sorry

end combined_score_is_75_l366_366434


namespace range_of_m_l366_366499

-- Define the constants used in the problem
def a : ℝ := 0.8
def b : ℝ := 1.2

-- Define the logarithmic inequality problem
theorem range_of_m (m : ℝ) : (a^(b^m) < b^(a^m)) → m < 0 := sorry

end range_of_m_l366_366499


namespace area_of_rectangle_ABCD_l366_366178

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366178


namespace c_negative_l366_366225

theorem c_negative (a b c : ℝ) (h₁ : a + b + c < 0) (h₂ : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 :=
sorry

end c_negative_l366_366225


namespace num_pos_divisors_36_l366_366133

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366133


namespace gcd_8p_18q_l366_366148

theorem gcd_8p_18q (p q : ℕ) (hp : p > 0) (hq : q > 0) (hg : Nat.gcd p q = 9) : Nat.gcd (8 * p) (18 * q) = 18 := 
sorry

end gcd_8p_18q_l366_366148


namespace triangles_with_vertex_A_l366_366257

theorem triangles_with_vertex_A : 
  ∃ (A : Point) (remaining_points : Finset Point), 
    (remaining_points.card = 8) → 
    (∃ (n : ℕ), n = (Nat.choose 8 2) ∧ n = 28) :=
by
  sorry

end triangles_with_vertex_A_l366_366257


namespace emmanuel_jelly_beans_l366_366893

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end emmanuel_jelly_beans_l366_366893


namespace triangle_side_length_l366_366439
  
theorem triangle_side_length (a b c a' d : ℝ)
  (h_angle_A : ∠A = 120)
  (h_angle_bisector : angle_bisector A = a')
  (h_difference : b + c - a = d)
  : a = (d * (a' - d)) / (2 * d - a') :=
by
  sorry

end triangle_side_length_l366_366439


namespace bisect_MK_l366_366169

-- Define the problem 
variables (A B C O M A1 B1 C1 K : Point)
variables (h_acute_triangle : isAcuteTriangle A B C)
variables (h_circumcenter : isCircumcenter O A B C)
variables (h_orthocenter : isOrthocenter M A B C)
variables (h_reflection_A : reflects A1 A (perpendicularBisector B C))
variables (h_reflection_B : reflects B1 B (perpendicularBisector C A))
variables (h_reflection_C : reflects C1 C (perpendicularBisector A B))
variables (h_incenter : isIncenter K A1 B1 C1)

theorem bisect_MK : bisects O M K :=
sorry

end bisect_MK_l366_366169


namespace fraction_result_l366_366952

theorem fraction_result (a b c : ℝ) (h1 : a / 2 = b / 3) (h2 : b / 3 = c / 5) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) :
  (a + b) / (c - a) = 5 / 3 :=
by
  sorry

end fraction_result_l366_366952


namespace geometric_sequence_fourth_term_l366_366314

theorem geometric_sequence_fourth_term (x : ℝ) (h : (3 * x + 3) ^ 2 = x * (6 * x + 6)) :
  (∀ n : ℕ, 0 < n → (x, 3 * x + 3, 6 * x + 6)) = -24 := by
  sorry

end geometric_sequence_fourth_term_l366_366314


namespace lcm_18_24_l366_366799

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366799


namespace sufficient_but_not_necessary_l366_366149

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 1) : (1 / x < 1) ∧ ¬(∀ x : ℝ, 1 / x < 1 → x > 1) :=
by
  split
  . -- Prove that x > 1 implies 1 / x < 1
    sorry
  . -- Prove that 1 / x < 1 does not imply x > 1 in general
    sorry

end sufficient_but_not_necessary_l366_366149


namespace find_n_from_exponent_equation_l366_366571

theorem find_n_from_exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  sorry

end find_n_from_exponent_equation_l366_366571


namespace lcm_18_24_l366_366795

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366795


namespace red_section_not_damaged_l366_366461

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l366_366461


namespace length_of_field_l366_366166

variables (w l : ℝ)
def pond_areas := 6^2 + 5^2 + 4^2 -- Sum of areas of the ponds
def field_area := l * w           -- Area of the field

theorem length_of_field :
  l = 3 * w ∧ pond_areas = 77 ∧ 77 = (1 / 4) * field_area →
  l ≈ 30.39 :=
begin
  sorry
end

end length_of_field_l366_366166


namespace part1_part2_l366_366172

variables {V : Type*} [add_comm_group V] [vector_space ℝ V] {A B C D M N G O : V}

-- Assuming midpoints conditions as data.
variables (midpoint_AD : M = 1⁄2 • (A + D))
variables (midpoint_BC : N = 1⁄2 • (B + C))
variables (midpoint_MN : G = 1⁄2 • (M + N))

-- Part 1: Proving the sum of vectors equals zero vector.
theorem part1 : 
    (G - A) + (G - B) + (G - C) + (G - D) = 0 := sorry

-- Part 2: Proving the vector relation with the centroid.
theorem part2 :
    (O - G) = 1⁄4 • ((O - A) + (O - B) + (O - C) + (O - D)) := sorry

end part1_part2_l366_366172


namespace inequality_example_l366_366265

open Real

theorem inequality_example 
    (x y z : ℝ) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1):
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := 
by 
  sorry

end inequality_example_l366_366265


namespace part1_part2_l366_366962

structure Conditions (k k1 : ℝ) (A M N : ℝ × ℝ) :=
  (k_pos : k > 0)
  (k_neq_one : k ≠ 1)
  (l_formula : ∀ x, (A.snd = k * x + 1))
  (l1_formula : ∀ x, (A.snd = k1 * x + 1))
  (symmetric_formula : ∀ x, (A.snd = k1 * x + 1) = (A.fst = k * (x - 1) + 1))
  (ellipse_formula : ∀ x y, (x^2 / 4 + y^2 = 1) → ((A.fst = x) ∧ (A.snd = y)) ∨ ((M.fst = x) ∧ (M.snd= y)) ∨ ((N.fst = x) ∧ (N.snd= y)))

theorem part1 {k k1 : ℝ} {A M N : ℝ × ℝ} (h_cond : Conditions k k1 A M N) :
  k * k1 = 1 :=
sorry

theorem part2 {k k1 : ℝ} {A M N : ℝ × ℝ} (h_cond : Conditions k k1 A M N) :
  ∃ T : ℝ × ℝ, T = (0, -5 / 3) ∧ (h_cond.ellipse_formula T.fst T.snd) :=
sorry

end part1_part2_l366_366962


namespace geometric_sequence_fourth_term_l366_366316

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end geometric_sequence_fourth_term_l366_366316


namespace Darren_paints_432_feet_l366_366463

theorem Darren_paints_432_feet (t : ℝ) (h : t = 792) (paint_ratio : ℝ) 
  (h_ratio : paint_ratio = 1.20) : 
  let d := t / (1 + paint_ratio)
  let D := d * paint_ratio
  D = 432 :=
by
  sorry

end Darren_paints_432_feet_l366_366463


namespace polynomial_constant_term_q_l366_366240

theorem polynomial_constant_term_q (p q r : Polynomial ℚ)
  (h1 : r = p * q)
  (hp_const : p.eval 0 = 5)
  (hr_const : r.eval 0 = -15) :
  q.eval 0 = -3 :=
sorry

end polynomial_constant_term_q_l366_366240


namespace find_c_for_identical_solutions_l366_366947

theorem find_c_for_identical_solutions :
  ∃ c : ℝ, (∀ x : ℝ, (2 * x^2 = 4 * x + c) ↔ x = 1) ∧ c = -2 :=
begin
  use -2,
  intros x,
  split,
  { intro h,
    have eq1 : 2 * (x^2) = 4 * x + (-2) := h,
    have eq2 : 2 * (x^2) - 4 * x = -2 := calc
      2 * (x^2) - 4 * x = 2 * (x^2 - 2 * x) : by ring
      ... = 2 * ((x - 1)^2 - 1) : by { rw [sub_sq], ring }
      ... = 2 * (x - 1)^2 - 2 : by ring,
    rw ← eq1 at *,
    rw ← eq2,
    simp [sub_eq_zero],
    linarith,
  },
  { intro hx,
    rw hx,
    ring,
  },
end

end find_c_for_identical_solutions_l366_366947


namespace ladder_slip_distance_l366_366845

noncomputable def ladder_slip (h l : ℝ) (x_initial : ℝ) : ℝ :=
  let y_sq := l^2 - (h - x_initial)^2 in
  sqrt (45 + 12 * sqrt y_sq) - 9

theorem ladder_slip_distance :
  ∀ (L : ℝ) (x1 h : ℝ),
  L = 40 → x1 = 9 → h = 6 → 
  ladder_slip 6 (sqrt (40^2 - 9^2)) 9 = sqrt (45 + 12 * sqrt(1519)) - 9 :=
by
  intros
  rw [H, H_1, H_2]
  simplify
  sorry

end ladder_slip_distance_l366_366845


namespace value_of_r_plus_s_l366_366704

noncomputable def line (x: ℝ) : ℝ := -1 / 2 * x + 6

theorem value_of_r_plus_s (r s : ℝ) (P Q T : ℝ × ℝ) (h1 : P = (12, 0)) (h2 : Q = (0, 6)) (h3 : T = (r, s))
  (h4 : T ∈ [ (12, 0) ; (0, 6) ])
  (h5 : (1 / 2) * 12 * 6 = 36)
  (h6 : (1 / 4) * 36 = 9)
  (h7 : s = line r) 
  (h8 : (1 / 2) * 12 * s = 9) :
  r + s = 10.5 := 
by 
  sorry

end value_of_r_plus_s_l366_366704


namespace correct_inequality_l366_366238

open Real

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem correct_inequality
  (h_even : even_function f)
  (h_mono : monotone_increasing_on_nonneg f) :
  f (-2) > f (-1) ∧ f (-1) > f (0) :=
begin
  sorry
end

end correct_inequality_l366_366238


namespace paper_fold_length_l366_366415

theorem paper_fold_length (length_orig : ℝ) (h : length_orig = 12) : length_orig / 2 = 6 :=
by
  rw [h]
  norm_num

end paper_fold_length_l366_366415


namespace min_value_g_geq_6_min_value_g_eq_6_l366_366488

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l366_366488


namespace find_abc_l366_366482

theorem find_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : a^3 + b^3 + c^3 = 2001 → (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) := 
sorry

end find_abc_l366_366482


namespace rectangle_area_l366_366193

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366193


namespace minimum_number_of_tiles_l366_366868

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l366_366868


namespace triangles_sum_length_l366_366636

theorem triangles_sum_length :
  let O := (0, 0)
  let A (n : ℕ) := (let x := n^2 in (x, Real.sqrt x))
  let B (n : ℕ) := (let S := (n * (n + 1) / 2) * (2 / 3) in (S, 0))
  (∑ i in finset.range 2006, triangle_side_length i) = 4022030 / 3 :=
sorry

end triangles_sum_length_l366_366636


namespace num_positive_divisors_36_l366_366098

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366098


namespace area_of_rectangle_l366_366199

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366199


namespace find_pairs_l366_366932

theorem find_pairs (x y : ℕ) (hxy : (x + y) * (x^2 + 9 * y) = Nat.prime_cube p) (hx : x > 0) (hy : y > 0) :
  (x, y) = (2, 5) ∨ (x, y) = (4, 1) := sorry

end find_pairs_l366_366932


namespace remainder_3001_3005_mod_17_l366_366354

theorem remainder_3001_3005_mod_17 :
  ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17 = 2 := by
  have h1 : 3001 % 17 = 10 := by norm_num
  have h2 : 3002 % 17 = 11 := by norm_num
  have h3 : 3003 % 17 = 12 := by norm_num
  have h4 : 3004 % 17 = 13 := by norm_num
  have h5 : 3005 % 17 = 14 := by norm_num
  calc
    ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17
      = (10 * 11 * 12 * 13 * 14) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 2 : by norm_num

end remainder_3001_3005_mod_17_l366_366354


namespace sum_first_40_terms_sequence_l366_366000

theorem sum_first_40_terms_sequence :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n : ℕ, a (n+2) - a n = (-1)^n + 2) ∧ (∑ n in Finset.range 40, a (n+1) = 820) :=
by
  sorry

end sum_first_40_terms_sequence_l366_366000


namespace round_to_hundredth_l366_366674

theorem round_to_hundredth (x : ℝ) (h1 : x = 3.456) (h2 : (x * 100) % 10 = 5) (h3 : (x * 1000) % 10 = 6) : (Real.round (x * 100) / 100) = 3.46 :=
by
  sorry

end round_to_hundredth_l366_366674


namespace problem1_problem2_l366_366246

/-- 
Problem 1: 
Prove that the value of b is 1, given:
1. The function f is defined by f(x) = a * log x + ((1 - a) / 2) * x^2 - b * x,
2. a is a real number and a ≠ 1,
3. The slope of the tangent line to the curve y = f(x) at the point (1, f(1)) is 0.
-/
theorem problem1 (a : ℝ) (h : a ≠ 1) : 
  let f (x : ℝ) := a * log x + ((1 - a) / 2) * x^2 - 1 * x 
  in let f' (x : ℝ) := a / x + (1 - a) * x - (1 : ℝ)
  in f' 1 = 0 → (1 : ℝ) = 1 :=
by
  intros,
  sorry

/-- 
Problem 2:
Prove the range of a is (-sqrt 2 - 1, sqrt 2 - 1) ∪ (1, +∞), given:
1. The function f is defined by f(x) = a * log x + ((1 - a) / 2) * x^2 - x,
2. a is a real number and a ≠ 1,
3. There exists x ∈ [1, +∞) such that f(x) < a / (a - 1).
-/
theorem problem2 (a : ℝ) (h : a ≠ 1) : 
  (
    (∃ x ∈ Icc 1 (real.top),
        let f (x : ℝ) := a * log x + ((1 - a) / 2) * x^2 - 1 * x
        in f x < a / (a - 1) 
    ) →
    a ∈ Ico (-(real.sqrt 2) - 1) (real.sqrt 2 - 1) ∨ a ∈ Ioi (1 : ℝ)
  ) :=
by
  intros,
  sorry

end problem1_problem2_l366_366246


namespace number_of_divisors_of_36_l366_366115

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366115


namespace g_of_f_eq_l366_366638

def f (A B x : ℝ) : ℝ := A * x^2 - B^2
def g (B x : ℝ) : ℝ := B * x + B^2

theorem g_of_f_eq (A B : ℝ) (hB : B ≠ 0) : 
  g B (f A B 1) = B * A - B^3 + B^2 := 
by
  sorry

end g_of_f_eq_l366_366638


namespace ral_current_age_l366_366272

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l366_366272


namespace tangent_properties_l366_366966

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom func_eq : ∀ x, f (x - 2) = f (-x)
axiom tangent_eq_at_1 : ∀ x, (x = 1 → f x = 2 * x + 1)

-- Prove the required results
theorem tangent_properties :
  (deriv f 1 = 2) ∧ (∃ B C, (∀ x, (x = -3) → f x = B -2 * (x + 3)) ∧ (B = 3) ∧ (C = -3)) :=
by
  sorry

end tangent_properties_l366_366966


namespace general_term_formula_l366_366939

theorem general_term_formula (n : ℕ) : 
  ∀ (a : ℕ → ℕ), (a 1 = 1) → (∀ n > 1, a n - a (n-1) = 2^(n-1)) → (a n = 2^n - 1) :=
  by 
  intros a h1 hdif
  sorry

end general_term_formula_l366_366939


namespace prob_return_O_4_steps_prob_distribution_expected_value_l366_366297

theorem prob_return_O_4_steps :
  let p_forward := (3 : ℚ) / 4
  let p_backward := (1 : ℚ) / 4
  (nat.choose 4 2 : ℚ) * (p_forward)^2 * (p_backward)^2 = 27 / 128 := by
  sorry

theorem prob_distribution_expected_value :
  let p_forward := (3 : ℚ) / 4
  let p_backward := (1 : ℚ) / 4
  let prob_X_1 := (nat.choose 5 3 : ℚ) * (p_forward)^3 * (p_backward)^2 + (nat.choose 5 2 : ℚ) * (p_forward)^2 * (p_backward)^3
  let prob_X_3 := (nat.choose 5 4 : ℚ) * (p_forward)^4 * (p_backward)^1 + (nat.choose 5 1 : ℚ) * (p_forward)^1 * (p_backward)^4
  let prob_X_5 := (p_forward)^5 + (p_backward)^5
  let expected_value := 1 * prob_X_1 + 3 * prob_X_3 + 5 * prob_X_5
  prob_X_1 = 45 / 128 ∧ prob_X_3 = 105 / 256 ∧ prob_X_5 = 61 / 256 ∧ expected_value = 355 / 128 := by
  sorry

end prob_return_O_4_steps_prob_distribution_expected_value_l366_366297


namespace units_digit_of_sum_base_8_l366_366493

-- Define the numbers in base 8
def n1 : ℕ := 135
def n2 : ℕ := 157
def n3 : ℕ := 163

-- Function to get the units digit of a base 8 number
def units_digit_base_8 (n : ℕ) : ℕ :=
  n % 8

-- Main theorem to prove
theorem units_digit_of_sum_base_8 : 
  units_digit_base_8 (n1 + n2 + n3) = 6 :=
by 
  -- Conversion of 135, 157, 163 to their corresponding base 10 representations for clarity
  have h1: n1 % 8 = 5 := by norm_num [n1, n2, n3],
  have h2: n2 % 8 = 7 := by norm_num [n1, n2, n3],
  have h3: n3 % 8 = 3 := by norm_num [n1, n2, n3],
  have h_sum: (n1 % 8 + n2 % 8 + n3 % 8) % 8 = 6 := by norm_num [n1, n2, n3, h1, h2, h3],
  rw ←units_digit_base_8,
  exact h_sum

end units_digit_of_sum_base_8_l366_366493


namespace number_of_games_is_15_l366_366830

-- Definition of the given conditions
def total_points : ℕ := 345
def avg_points_per_game : ℕ := 4 + 10 + 9
def number_of_games (total_points : ℕ) (avg_points_per_game : ℕ) := total_points / avg_points_per_game

-- The theorem stating the proof problem
theorem number_of_games_is_15 : number_of_games total_points avg_points_per_game = 15 :=
by
  -- Skipping the proof as only the statement is required
  sorry

end number_of_games_is_15_l366_366830


namespace find_square_area_l366_366730

noncomputable def square_area_parabola_line (b : ℝ) (d : ℝ) : Prop :=
  let line := (x : ℝ) => 2 * x + b in
  let parabola := (x : ℝ) => x^2 in
  let intersection_points := [1 + real.sqrt(1 + b), 1 - real.sqrt(1 + b)] in
  let horizontal_dist := 2 * real.sqrt(b + 1) in
  let vertical_dist := (1 + real.sqrt(1 + b))^2 - (1 - real.sqrt(1 + b))^2 in
  let diag_side := real.sqrt(horizontal_dist^2 + vertical_dist^2) in
  let area := (diag_side / real.sqrt(2))^2 in
  area = 80 ∨ area = 1280

theorem find_square_area : ∃ (b : ℝ) (d : ℝ), square_area_parabola_line b d :=
sorry

end find_square_area_l366_366730


namespace problem_part1_problem_part2_l366_366649

noncomputable def foci := (F1 : ℝ × ℝ, F2 : ℝ × ℝ)

theorem problem_part1 (x y : ℝ) (h : x ≥ real.sqrt 2) :
  ∃ (v : ℝ), v ∈ set.Ici (2 + real.sqrt 10) ∧
  v = (x^2 / 2 - y^2 / 3 = 1) → (x, y) • ((x + real.sqrt 5), y) := 
sorry

theorem problem_part2 (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (h : 
    ∀ P, let d1' := dist P F1, let d2' := dist P F2 in d1' + d2' = const) (h_cos : ∀ P, cos (angle F1 P F2) = -1 / 9) : 
  ∀ P, let (x, y) := P in (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

end problem_part1_problem_part2_l366_366649


namespace cosine_decomposition_l366_366724

noncomputable def b1 := 3 / 4
noncomputable def b2 := 0
noncomputable def b3 := 1 / 4

theorem cosine_decomposition :
  (∀ θ : ℝ, cos θ ^ 3 = b1 * cos θ + b2 * cos (2 * θ) + b3 * cos (3 * θ)) ∧ (b1^2 + b2^2 + b3^2 = 5 / 8) :=
by
  sorry

end cosine_decomposition_l366_366724


namespace number_of_divisors_of_36_l366_366114

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366114


namespace lunch_cost_total_l366_366879

theorem lunch_cost_total (x y : ℝ) (h1 : y = 45) (h2 : x = (2 / 3) * y) : 
  x + y + y = 120 := by
  sorry

end lunch_cost_total_l366_366879


namespace general_formula_of_sequence_l366_366972

noncomputable theory

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 3

theorem general_formula_of_sequence (a : ℕ → ℤ) (h : sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n + 1) - 3 :=
by
  sorry

end general_formula_of_sequence_l366_366972


namespace area_of_rectangle_l366_366202

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366202


namespace count_concave_numbers_is_240_l366_366884

def is_concave_number (n : ℕ) : Prop :=
  ∃ h t u : ℕ, 
    n = 100 * h + 10 * t + u ∧
    1 ≤ h ∧ h ≤ 9 ∧
    0 ≤ t ∧ t ≤ 9 ∧
    0 ≤ u ∧ u ≤ 9 ∧
    h ≠ t ∧ t ≠ u ∧ h ≠ u ∧
    t < h ∧ t < u

def count_concave_numbers : ℕ :=
  (Finset.range 1000).filter is_concave_number |>.card

theorem count_concave_numbers_is_240 : count_concave_numbers = 240 :=
  sorry

end count_concave_numbers_is_240_l366_366884


namespace penalty_kicks_required_l366_366691

theorem penalty_kicks_required (players goalies : ℕ) (h1 : players = 22) (h2 : goalies = 4) : 
  let shots_per_goalie := players - 1 in
  let total_shots := goalies * shots_per_goalie in
  total_shots = 84 := 
by
  sorry

end penalty_kicks_required_l366_366691


namespace residue_of_neg_1237_mod_37_l366_366473

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l366_366473


namespace best_loan_option_l366_366888

def loan_amount : ℝ := 65000
def loan_term : ℝ := 1 -- in years

-- Option a
def option_a_amount : ℝ := 68380

-- Option b: Simple Interest
def option_b_rate : ℝ := 0.05
def option_b_interest : ℝ := loan_amount * option_b_rate
def option_b_amount : ℝ := loan_amount + option_b_interest

-- Option c: Compound Interest Monthly
def option_c_rate : ℝ := 0.05
def option_c_periods : ℝ := 12
def option_c_amount : ℝ := loan_amount * (1 + option_c_rate / option_c_periods) ^ option_c_periods

-- Option d: Compound Interest Quarterly
def option_d_rate : ℝ := 0.05
def option_d_periods : ℝ := 4
def option_d_amount : ℝ := loan_amount * (1 + option_d_rate / option_d_periods) ^ option_d_periods

-- Option e: Compound Interest Semi-Annually
def option_e_rate : ℝ := 0.05
def option_e_periods : ℝ := 2
def option_e_amount : ℝ := loan_amount * (1 + option_e_rate / option_e_periods) ^ option_e_periods

-- Prove that option b is the most advantageous
theorem best_loan_option : option_b_amount = min option_a_amount (min option_c_amount (min option_d_amount option_e_amount)) :=
sorry

end best_loan_option_l366_366888


namespace rectangle_area_l366_366475

theorem rectangle_area (z1 z2 : ℂ) 
  (h1 : z1^2 = 3 + 3 * real.sqrt 10 * complex.I)
  (h2 : z2^2 = 1 + real.sqrt 2 * complex.I) : 
  let A := (real.sqrt 33 / 2) * complex.abs ((complex.re z1 * complex.im z2) - (complex.re z2 * complex.im z1)) in
  A = 0.079 * real.sqrt 33 :=
sorry

end rectangle_area_l366_366475


namespace incorrect_sum_Sn_l366_366330

-- Define the geometric sequence sum formula
def Sn (a r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Define the given values
def S1 : ℕ := 8
def S2 : ℕ := 20
def S3 : ℕ := 36
def S4 : ℕ := 65

-- The main proof statement
theorem incorrect_sum_Sn : 
  ∃ (a r : ℕ), 
  a = 8 ∧ 
  Sn a r 1 = S1 ∧ 
  Sn a r 2 = S2 ∧ 
  Sn a r 3 ≠ S3 ∧ 
  Sn a r 4 = S4 :=
by sorry

end incorrect_sum_Sn_l366_366330


namespace lcm_18_24_l366_366751

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366751


namespace children_playing_both_sports_l366_366597

variable (total_children : ℕ) (T : ℕ) (S : ℕ) (N : ℕ)

theorem children_playing_both_sports 
  (h1 : total_children = 38) 
  (h2 : T = 19) 
  (h3 : S = 21) 
  (h4 : N = 10) : 
  (T + S) - (total_children - N) = 12 := 
by
  sorry

end children_playing_both_sports_l366_366597


namespace area_of_quadrilateral_AEGF_l366_366854

/-
  Given a right triangle ABC with AC = 5 and BC = 12, and a circle circumscribed around the triangle:
  - Points E and G are the midpoints of the arcs AC and BC not containing B and A, respectively.
  - Point F is the midpoint of the arc AB not containing C.
  Prove that the area of quadrilateral AEGF is 169/2.
-/
theorem area_of_quadrilateral_AEGF 
  (A B C E G F : ℝ)
  (hAC : AC = 5) 
  (hBC : BC = 12)
  (circumscribed : circumscribed_around_triangle ABC)
  (midpoints : E = midpoint_of_arc_AC ∧ G = midpoint_of_arc_BC ∧ F = midpoint_of_arc_AB)
  : area_quadrilateral A E G F = 169 / 2 := by sorry

end area_of_quadrilateral_AEGF_l366_366854


namespace estimate_first_year_students_l366_366878

noncomputable def number_of_first_year_students (N : ℕ) : Prop :=
  let p1 := (N - 90) / N
  let p2 := (N - 100) / N
  let p_both := 1 - p1 * p2
  p_both = 20 / N → N = 450

theorem estimate_first_year_students : ∃ N : ℕ, number_of_first_year_students N :=
by
  use 450
  -- sorry added to skip the proof part
  sorry

end estimate_first_year_students_l366_366878


namespace number_of_divisors_of_36_l366_366116

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366116


namespace num_multiples_of_15_l366_366558

theorem num_multiples_of_15 (a b m : ℕ) (h1 : a = 25) (h2 : b = 200) (h3 : m = 15) : 
  ∃ n, n = (b - a) / m + 1 ∧ ∃ k, k = (n - 1) * m + a ∧ k mod m = 0 ∧ a < k ∧ k < b := by
  sorry

end num_multiples_of_15_l366_366558


namespace maximum_b_n_T_l366_366518

/-- Given a sequence {a_n} defined recursively and b_n = a_n / n.
   We need to prove that for all n in positive natural numbers,
   b_n is greater than or equal to T, and the maximum such T is 3. -/
theorem maximum_b_n_T (T : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, n ≥ 1 → a (n + 1) = a n + 2 * n) →
  (∀ n, n ≥ 1 → b n = a n / n) →
  (∀ n, n ≥ 1 → b n ≥ T) →
  T ≤ 3 :=
by
  sorry

end maximum_b_n_T_l366_366518


namespace value_of_M_l366_366326

theorem value_of_M (G A M E: ℕ) (hG : G = 15)
(hGAME : G + A + M + E = 50)
(hMEGA : M + E + G + A = 55)
(hAGE : A + G + E = 40) : 
M = 15 := sorry

end value_of_M_l366_366326


namespace part1_part2a_part2b_l366_366960

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (x : ℝ) := Real.log x

theorem part1 (a : ℝ) (H : ∀ x > 0, f x - g x ≥ f a - g a) : f a * g a = -1 := by
  sorry

theorem part2a (x1 x2 : ℝ) (H1 : 1 < x1) (H2 : x1 < x2) :
  ∃ x0 ∈ Ioo x1 x2, (f x1 - f x2) / (x1 - x2) / ((g x1 - g x2) / (x1 - x2)) = x0 * Real.exp x0 := by
  sorry

theorem part2b (x1 x2 : ℝ) (H1 : 1 < x1) (H2 : x1 < x2) :
  (f x1 - f x2) / (x1 - x2) - (g x1 - g x2) / (x1 - x2) < (f x1 + f x2) / 2 - 1 / Real.sqrt (x1 * x2) := by
  sorry

end part1_part2a_part2b_l366_366960


namespace carolyn_sum_l366_366300

theorem carolyn_sum (n : ℕ) (init_list : list ℕ)
  (h_n : n = 8)
  (h_init_list : init_list = [1, 2, 3, 4, 5, 6, 7, 8])
  (carolyn_moves : list ℕ)
  (h_carolyn_moves : carolyn_moves = [3, 6, 8]):
  carolyn_moves.sum = 17 := by {
  sorry
}

end carolyn_sum_l366_366300


namespace numDyckPaths_l366_366847

/-- A Dyck path of length 2n is a path consisting of n up-steps and n down-steps 
    that never goes below the altitude of the starting point. -/
def isDyckPath (path : List Bool) : Prop :=
  let up_steps := path.filter (λ x => x).length
  let down_steps := path.filter (λ x => ¬x).length
  up_steps = down_steps ∧
  path.foldl (λ (acc : ℤ) (x : Bool) => if x then acc + 1 else acc - 1) 0 ≥ 0

theorem numDyckPaths (n : ℕ) : 
  (finset.card {path : List Bool | path.length = 2*n ∧ isDyckPath path}) = 
  (nat.choose (2*n) n) - (nat.choose (2*n) (n-1)) :=
sorry

end numDyckPaths_l366_366847


namespace lcm_18_24_l366_366796

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366796


namespace max_cubes_in_box_l366_366820

theorem max_cubes_in_box :
  let volume_of_cube := 27 -- volume of each small cube in cubic centimetres
  let dimensions_of_box := (8, 9, 12) -- dimensions of the box in centimetres
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3 -- volume of the box
  volume_of_box / volume_of_cube = 32 := 
by
  let volume_of_cube := 27
  let dimensions_of_box := (8, 9, 12)
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3
  show volume_of_box / volume_of_cube = 32
  sorry

end max_cubes_in_box_l366_366820


namespace remainder_3001_3005_mod_17_l366_366355

theorem remainder_3001_3005_mod_17 :
  ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17 = 2 := by
  have h1 : 3001 % 17 = 10 := by norm_num
  have h2 : 3002 % 17 = 11 := by norm_num
  have h3 : 3003 % 17 = 12 := by norm_num
  have h4 : 3004 % 17 = 13 := by norm_num
  have h5 : 3005 % 17 = 14 := by norm_num
  calc
    ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17
      = (10 * 11 * 12 * 13 * 14) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 2 : by norm_num

end remainder_3001_3005_mod_17_l366_366355


namespace value_of_f_at_7_l366_366016

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x ^ 2 + 5 * x + 3 else -f (-x)

theorem value_of_f_at_7 : f 7 = -66 := by
  sorry

end value_of_f_at_7_l366_366016


namespace tan_trig_identity_l366_366006

noncomputable def given_condition (α : ℝ) : Prop :=
  Real.tan (α + Real.pi / 3) = 2

theorem tan_trig_identity (α : ℝ) (h : given_condition α) :
  (Real.sin (α + (4 * Real.pi / 3)) + Real.cos ((2 * Real.pi / 3) - α)) /
  (Real.cos ((Real.pi / 6) - α) - Real.sin (α + (5 * Real.pi / 6))) = -3 :=
sorry

end tan_trig_identity_l366_366006


namespace range_of_k_l366_366527

theorem range_of_k (k : ℝ) (x : ℝ) : 
  k ≠ 0 ∧ x = 1 → (k^2 * x^2 - 6 * k * x + 8 ≥ 0) ↔ k ∈ Set.Iic(4) ∨ k ∈ Set.Icc(2, 4) :=
sorry

end range_of_k_l366_366527


namespace num_pos_divisors_36_l366_366073

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366073


namespace exponent_equation_l366_366575

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l366_366575


namespace num_pos_divisors_36_l366_366106

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366106


namespace fewest_tiles_to_cover_region_l366_366869

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l366_366869


namespace correct_answer_l366_366711

-- Define the conditions from Step a)
def respecting_objective_laws : Prop := 
  Respecting the law is the basis for fully exerting subjective initiative.  -- Remove spaces for Lean code.

def initiative_of_consciousness : Prop := 
  The invention of new technology is the result of people exercising the initiative of consciousness.  -- Remove spaces for Lean code.

def transformation_of_contradiction : Prop := 
  A bottle of mineral water can clean a car as good as new, showing the need to actively create conditions to achieve the transformation of both sides of a contradiction.  -- Remove spaces for Lean code.

def fundamental_contradiction : Prop := 
  The fundamental contradiction of society is the driving force behind social development.  -- Remove spaces for Lean code.

-- Define the main theorem to be proven
theorem correct_answer (h1: initiative_of_consciousness) (h2: transformation_of_contradiction) : 
  (initiative_of_consciousness ∧ transformation_of_contradiction) :=
begin
  split,
  assumption,
  assumption,
end

end correct_answer_l366_366711


namespace perpendicular_bisector_eqn_line_eqn_l366_366525

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ( (p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2 )

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem perpendicular_bisector_eqn :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (5, 7)
  let M := midpoint A B
  let m_AB := slope A B
  let m_perp := -1 / m_AB
  ∃ a b c : ℝ, (a = 1) ∧ (b = 1) ∧ (c = -8) ∧
    (∀ x y : ℝ, y = m_perp * (x - M.1) + M.2 → a * x + b * y + c = 0) :=
sorry

theorem line_eqn :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (5, 7)
  let P : ℝ × ℝ := (-1, 0)
  slope A B = slope P (3, 5) ∧
  ∃ a b c : ℝ, ((a = 1) ∧ (b = -1) ∧ (c = 1) ∧
  (∀ x y : ℝ, y = slope A B * (x - P.1) + P.2 → a * x + b * y + c = 0)) ∨
  ((a = 5) ∧ (b = -4) ∧ (c = 5) ∧
  (∀ x y : ℝ, y = 5/4 * (x + 1) + 0 → a * x + b * y + c = 0)) :=
sorry

end perpendicular_bisector_eqn_line_eqn_l366_366525


namespace probability_x_plus_one_div_x_minus_one_ge_a_l366_366580

noncomputable def probability_inequality_holds : ℝ :=
  let a_range := set.Icc (0 : ℝ) 5
  let interval_satisfying := set.Icc (0 : ℝ) 2
  (interval_satisfying.volume) / (a_range.volume)

theorem probability_x_plus_one_div_x_minus_one_ge_a :
  (∫ a in 0..5, indicator (λ a, ∀ x > (1 : ℝ), x + (1 / (x - 1)) ≥ a) a) / (5 - 0) = 2 / 5 := 
sorry

end probability_x_plus_one_div_x_minus_one_ge_a_l366_366580


namespace find_polynomials_l366_366931

noncomputable def polynomial_solution (P : ℝ → ℝ) : Prop :=
∀ x : ℝ, P(x^2 - 2 * x) = (P(x - 2))^2

theorem find_polynomials (P : ℝ → ℝ) :
  polynomial_solution P ↔ ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, P(x) = (x - 1)^n :=
by {
  sorry
}

end find_polynomials_l366_366931


namespace train_journey_time_l366_366423

theorem train_journey_time :
  let distance1 := 150
      speed1 := 50
      distance2 := 240
      speed2 := 80
      stop_time := 0.5 -- expressed in hours
  in (distance1 / speed1) + (distance2 / speed2) + stop_time = 6.5 :=
by
  sorry

end train_journey_time_l366_366423


namespace probability_train_there_when_joseph_arrives_l366_366873

variable {t : ℝ} {j : ℝ}

-- Conditions
def train_arrives_between_1_and_3 (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 120
def joseph_arrives_between_1_and_3 (j : ℝ) : Prop := 0 ≤ j ∧ j ≤ 120
def train_waits_for_one_hour (t j : ℝ) : Prop := t ≤ j ∧ j ≤ t + 60

-- Theorem to prove the given probability
theorem probability_train_there_when_joseph_arrives :
  (∀ t j, train_arrives_between_1_and_3 t ∧ joseph_arrives_between_1_and_3 j ∧ train_waits_for_one_hour t j →
   (5 / 8)) := sorry

end probability_train_there_when_joseph_arrives_l366_366873


namespace only_one_line_through_sqrt3_l366_366881

theorem only_one_line_through_sqrt3 (x y : ℝ) (hx : x = 0) :
  ∀ (a b : ℚ), ¬ (line_through (sqrt 3, 0) (a, b) ≠ vertical_line (sqrt 3) ∧ (∃ (p1 p2 : ℤ × ℤ), p1 ≠ p2 ∧ (line_through (sqrt 3, 0) p1 = line_through (sqrt 3, 0) p2))) :=
by sorry

end only_one_line_through_sqrt3_l366_366881


namespace probability_of_two_digit_number_is_1_over_19_l366_366864

-- Define the set and its properties
def set_of_numbers : set ℕ := {x | 50 ≤ x ∧ x ≤ 999}

-- Define the subset of two-digit numbers within the set
def two_digit_numbers : set ℕ := {x | 50 ≤ x ∧ x ≤ 99}

-- Total number of elements in the set
def total_elements : ℕ := 950

-- Number of two-digit elements in the set
def two_digit_count : ℕ := 50

-- Prove that the probability is 1/19
theorem probability_of_two_digit_number_is_1_over_19 :
  (two_digit_count : ℚ) / (total_elements : ℚ) = 1 / 19 := by
  sorry

end probability_of_two_digit_number_is_1_over_19_l366_366864


namespace table_price_l366_366384

theorem table_price :
  ∃ C T : ℝ, (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 72) ∧ (T = 63) :=
by
  sorry

end table_price_l366_366384


namespace lcm_18_24_eq_72_l366_366812

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366812


namespace greatest_prime_factor_expression_l366_366744

open Nat

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def expression : ℕ := (factorial 13 * factorial 12 + factorial 12 * factorial 11 - factorial 11 * factorial 10) / 171

theorem greatest_prime_factor_expression : 
  (∀ p, prime p → divides p expression → p ≤ 19) ∧ 
  prime 19 ∧ divides 19 expression :=
by
  sorry

end greatest_prime_factor_expression_l366_366744


namespace min_polynomial_value_achieves_min_value_l366_366642

noncomputable def polynomial_value (x : ℝ) : ℝ :=
  (17 - x) * (19 - x) * (19 + x) * (17 + x)

theorem min_polynomial_value : ∀ x : ℝ, polynomial_value x ≥ -1296 :=
begin
  intro x,
  sorry
end

theorem achieves_min_value : ∃ x : ℝ, polynomial_value x = -1296 :=
begin
  use 0,
  sorry
end

end min_polynomial_value_achieves_min_value_l366_366642


namespace g_min_value_l366_366491

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l366_366491


namespace rectangle_area_l366_366211

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366211


namespace find_root_of_equation_l366_366904

theorem find_root_of_equation :
  ∃ x : ℝ, (x - 1) + 2 * real.sqrt (x + 3) = 5 ∧ x = 2 :=
by
  use 2
  split
  · -- Proof that (2 - 1) + 2 * real.sqrt (2 + 3) = 5
    sorry
  · -- Proof that x = 2
    rfl

end find_root_of_equation_l366_366904


namespace largest_quantity_l366_366828

-- Defining the quantities
def A : ℝ := (2010 / 2009) + (2010 / 2011)
def B : ℝ := (2010 / 2011) + (2012 / 2011)
def C : ℝ := (2011 / 2010) + (2011 / 2012)

-- The proof statement
theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l366_366828


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366358

theorem remainder_3001_3002_3003_3004_3005_mod_17 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366358


namespace number_of_divisors_36_l366_366083

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366083


namespace geometric_sequence_fourth_term_l366_366315

theorem geometric_sequence_fourth_term (x : ℝ) (h : (3 * x + 3) ^ 2 = x * (6 * x + 6)) :
  (∀ n : ℕ, 0 < n → (x, 3 * x + 3, 6 * x + 6)) = -24 := by
  sorry

end geometric_sequence_fourth_term_l366_366315


namespace sequence_properties_l366_366990

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n+1) = a n * q

noncomputable def arithmetic_sequence (a b c : ℕ → ℝ) := ∀ n : ℕ, 2 * b n = a n + c n

theorem sequence_properties (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a →
  (8 * a 3 * a 4 - a 5 ^ 2 = 0) →
  arithmetic_sequence (λ n, a 2) (λ n, a 4) (λ n, a 6 - 36) →
  (∀ n, a n = 2 ^ n) ∧
  (∀ n : ℕ, ∑ i in range n, (1 / ((- (2 * i - 1)) * (- (2 * i + 1)))) = n / (2 * n + 1)) := by
sorry

end sequence_properties_l366_366990


namespace sin_cos_value_l366_366031

noncomputable def log_function (a x : ℝ) := log a (x - 3) + 2

def fixed_point (x y : ℝ) (P : ℝ × ℝ) := P = (x, y)

def condition_a (a : ℝ) := a > 0 ∧ a ≠ 1

def point_P_on_terminal_side_of_alpha (α x y : ℝ) :=
  -- Here we assume x and y can be computed based on α if necessary
  fixed_point x y (4, 2)

def sin_cos_identity (α : ℝ) :=
  2 * sin α * cos α + (2 * (cos α) ^ 2 - 1)

theorem sin_cos_value : 
  ∀ (a α x y : ℝ), 
    condition_a a ∧
    log_function a x = y + 2 ∧
    fixed_point x y (4, 2) ∧
    point_P_on_terminal_side_of_alpha α 4 2
    → sin_cos_identity α = 7 / 5 := 
begin
  intros,
  sorry
end

end sin_cos_value_l366_366031


namespace lcm_18_24_eq_72_l366_366777

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366777


namespace emmanuel_jelly_beans_l366_366894

theorem emmanuel_jelly_beans (total_jelly_beans : ℕ)
      (thomas_percentage : ℕ)
      (barry_ratio : ℕ)
      (emmanuel_ratio : ℕ)
      (h1 : total_jelly_beans = 200)
      (h2 : thomas_percentage = 10)
      (h3 : barry_ratio = 4)
      (h4 : emmanuel_ratio = 5) :
  let thomas_jelly_beans := (thomas_percentage * total_jelly_beans) / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_jelly_beans
  let total_ratio := barry_ratio + emmanuel_ratio
  let per_part_jelly_beans := remaining_jelly_beans / total_ratio
  let emmanuel_jelly_beans := emmanuel_ratio * per_part_jelly_beans
  emmanuel_jelly_beans = 100 :=
by
  sorry

end emmanuel_jelly_beans_l366_366894


namespace calculate_magnitude_l366_366038

variables {ℝ : Type*}
variables (a b : EuclideanSpace ℝ (fin 2))

open real

noncomputable def vec_a : EuclideanSpace ℝ (fin 2) := ![4, -3]
noncomputable def mag_b : ℝ := 3

theorem calculate_magnitude
  (h1 : a = vec_a)
  (h2 : ∥b∥ = mag_b)
  (h3 : angle a b = 2 * π / 3) :
  ∥2 • a + 3 • b∥ = sqrt 91 :=
sorry

end calculate_magnitude_l366_366038


namespace solve_arcsin_eq_l366_366685

open Real

noncomputable def problem_statement (x : ℝ) : Prop :=
  arcsin (sin x) = (3 * x) / 4

theorem solve_arcsin_eq(x : ℝ) (h : problem_statement x) (h_range: - (2 * π) / 3 ≤ x ∧ x ≤ (2 * π) / 3) : x = 0 :=
sorry

end solve_arcsin_eq_l366_366685


namespace vasya_guaranteed_win_l366_366663

-- Define the game setup
def poly_game (P : ℤ[x]) (queries : List ℤ) : Prop :=
  ∃ (responses : List ℕ),
    (∀ i, responses.nth i = some (polynomial.natDegree (P - polynomial.C (queries.nth i).getD 0))) ∧
    (∃ a b, a ≠ b ∧ responses.nth a = responses.nth b)

-- Define the proof problem
theorem vasya_guaranteed_win : ∀ P : ℤ[x], ∃ queries : List ℤ, length queries = 4 ∧ poly_game P queries := 
by
  -- The specifics of the proof are omitted 
  sorry

end vasya_guaranteed_win_l366_366663


namespace number_of_divisors_of_36_l366_366118

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366118


namespace num_pos_divisors_36_l366_366072

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366072


namespace probability_no_success_l366_366452

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l366_366452


namespace solve_system_l366_366291

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l366_366291


namespace degree_of_polynomial_l366_366741

def polynomial := fun (x : ℝ) => 7 * sin x - 3 * x^5 + 15 + 2 * log x * x^2 + exp x * x^6 - sqrt 5

theorem degree_of_polynomial : degree polynomial = 6 := by
  sorry

end degree_of_polynomial_l366_366741


namespace area_of_ABCD_l366_366187

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366187


namespace partition_condition_l366_366467

def is_partitionable (k : ℕ) : Prop :=
  ∃ A B : Set ℕ, A ∩ B = ∅ ∧ A ∪ B = {1990 + i | i in Finset.range (k + 1)} ∧
  (∑ a in A, a) = ∑ b in B, b

theorem partition_condition (k : ℕ) (h_pos : 0 < k) : 
  (is_partitionable k) ↔ (k % 4 = 3) :=
sorry

end partition_condition_l366_366467


namespace parallelogram_D_coord_l366_366980

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (-1, -2)

-- Define the coordinates of D based on the diagonals bisecting property
def D : ℝ × ℝ := (0, -4)

-- Prove that D is the correct fourth point of the parallelogram ABCD
theorem parallelogram_D_coord : ∃ D : ℝ × ℝ, D = (0, -4) :=
by
  use (0, -4)
  -- Further proof goes here
  sorry

end parallelogram_D_coord_l366_366980


namespace smallest_perimeter_of_triangle_with_consecutive_odd_integers_l366_366822

theorem smallest_perimeter_of_triangle_with_consecutive_odd_integers :
  ∃ (a b c : ℕ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ 
  (a < b) ∧ (b < c) ∧ (c = a + 4) ∧
  (a + b > c) ∧ (b + c > a) ∧ (a + c > b) ∧ 
  (a + b + c = 15) :=
by
  sorry

end smallest_perimeter_of_triangle_with_consecutive_odd_integers_l366_366822


namespace total_students_high_school_l366_366860

theorem total_students_high_school (s10 s11 s12 total_students sample: ℕ ) 
  (h1 : s10 = 600) 
  (h2 : sample = 45) 
  (h3 : s11 = 20) 
  (h4 : s12 = 10) 
  (h5 : sample = s10 + s11 + s12) : 
  total_students = 1800 :=
by 
  sorry

end total_students_high_school_l366_366860


namespace find_natural_solution_l366_366480

theorem find_natural_solution (x y : ℕ) (h : y^6 + 2 * y^3 - y^2 + 1 = x^3) : x = 1 ∧ y = 0 :=
by
  sorry

end find_natural_solution_l366_366480


namespace proof_problem_l366_366511

noncomputable def exists_positive_n (r : ℝ) (k : ℕ) : Prop :=
  r > 1 → ∃ n : ℕ, ∀ m : ℕ, m > n → 
  (∑ (x : ℕ) in finset.Ico (⌊r^(m-1)⌋₊ + 1) (⌊r^m⌋₊ + 1), (x^k)) > r^m

theorem proof_problem (r : ℝ) (k : ℕ) (hr : r > 1) (hk : k > 0) : exists_positive_n r k :=
begin
  rw exists_positive_n,
  intro hr,
  sorry
end

end proof_problem_l366_366511


namespace num_pos_divisors_36_l366_366139

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366139


namespace greatest_x_l366_366743

theorem greatest_x (x : ℕ) (h_pos : 0 < x) (h_ineq : (x^6) / (x^3) < 18) : x = 2 :=
by sorry

end greatest_x_l366_366743


namespace bushes_for_zucchinis_l366_366924

def bushes_yield := 10 -- containers per bush
def container_to_zucchini := 3 -- containers per zucchini
def zucchinis_required := 60 -- total zucchinis needed

theorem bushes_for_zucchinis (hyld : bushes_yield = 10) (ctz : container_to_zucchini = 3) (zreq : zucchinis_required = 60) :
  ∃ bushes : ℕ, bushes = 60 * container_to_zucchini / bushes_yield :=
sorry

end bushes_for_zucchinis_l366_366924


namespace most_beneficial_option_l366_366890

-- Defining the conditions
def principal : ℝ := 65000
def annual_rate : ℝ := 0.05

def simple_interest_total (P : ℝ) (r : ℝ) : ℝ := P + (P * r)
def compound_interest_total (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r / n) ^ n

def option_a_total : ℝ := 68380
def option_b_total : ℝ := simple_interest_total principal annual_rate
def option_c_total : ℝ := compound_interest_total principal annual_rate 12
def option_d_total : ℝ := compound_interest_total principal annual_rate 4
def option_e_total : ℝ := compound_interest_total principal annual_rate 2

-- Proving the most beneficial option
theorem most_beneficial_option : option_b_total < option_a_total ∧
                                    option_b_total < option_c_total ∧
                                    option_b_total < option_d_total ∧
                                    option_b_total < option_e_total :=
by
  -- Proof omitted
  sorry

end most_beneficial_option_l366_366890


namespace victor_hourly_wage_l366_366347

def total_hours_worked (hours_mon : ℕ) (hours_tue : ℕ) := hours_mon + hours_tue
def total_money_earned (money : ℕ) := money
def hourly_wage (hours_worked : ℕ) (money_earned : ℕ) := money_earned / hours_worked

theorem victor_hourly_wage :
  ∀ (h_mon h_tue : ℕ) (money : ℕ), 
  h_mon = 5 → h_tue = 5 → money = 60 → 
  hourly_wage (total_hours_worked h_mon h_tue) money = 6 := 
by
  intros h_mon h_tue money hmon_eq htue_eq money_eq
  simp [hmon_eq, htue_eq, money_eq, total_hours_worked, hourly_wage]
  sorry

end victor_hourly_wage_l366_366347


namespace area_ratio_triangles_l366_366263

/-- Area ratio theorem of triangles PQR and ABC given the division ratios p, q, r. -/
theorem area_ratio_triangles 
  (ABC : Triangle)
  (A1 B1 C1 P Q R : Point)
  (p q r : ℝ)
  (H1 : divides ABC.A ABC.B ABC.C A1 p)
  (H2 : divides ABC.B ABC.C ABC.A B1 q)
  (H3 : divides ABC.C ABC.A ABC.B C1 r)
  (H4 : P = intersection (line ABC.A A1) (line B1 C))
  (H5 : Q = intersection (line ABC.B B1) (line P C1))
  (H6 : R = intersection (line ABC.C C1) (line P B1)) :
  area_ratio (triangle P Q R) (triangle ABC) 
    = ((1 - p * q * r) ^ 2) / ((1 + p + p * q) * (1 + q + q * r) * (1 + r + r * p)) :=
sorry

end area_ratio_triangles_l366_366263


namespace max_cube_edge_length_in_tetrahedron_l366_366858

theorem max_cube_edge_length_in_tetrahedron (edge_length_tetrahedron : ℝ) (h : edge_length_tetrahedron = 6) : 
  ∃ a, is_cube_with_max_edge a ∧ a = sqrt 2 := 
sorry

def is_cube_with_max_edge (a : ℝ) : Prop :=
  ∃ r, r = sqrt 6 / 2 ∧ 3 * a ^ 2 = r ^ 2

end max_cube_edge_length_in_tetrahedron_l366_366858


namespace lcm_18_24_eq_72_l366_366774

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366774


namespace angle_ratio_A_to_B_l366_366258

-- Definitions of points and segments
variable (A B C E L X : Point)
variable (AC BE AL AL_segment BX_segment : Segment)
variable (AX XE : ℝ)
variable (angle_A angle_B : ℝ)

-- Assumptions
axiom AC_contains_E : E ∈ AC
axiom angle_bisector_AL : AL.bisector = true
axiom AL_intersects_BE_at_X : X ∈ BE
axiom AX_eq_XE : AX = XE -- (AX = XE)
axiom AL_eq_BX : AL = BX

-- Theorem to prove
theorem angle_ratio_A_to_B (h1 : E ∈ AC) (h2 : AL.bisector = true) (h3 : X ∈ BE) (h4 : AX = XE) (h5 : AL = BX) : 
  angle_A / angle_B = 2 :=
sorry

end angle_ratio_A_to_B_l366_366258


namespace tan_triple_angle_l366_366146

noncomputable def sine_theta : ℝ := 5 / 13

noncomputable def theta_is_in_first_quadrant : 0 < θ ∧ θ < π / 2 := sorry

theorem tan_triple_angle (θ : ℝ)
  (h1 : sin θ = sine_theta)
  (h2 : θ_is_in_first_quadrant) :
  tan (3 * θ) = 145 / 78 :=
sorry

end tan_triple_angle_l366_366146


namespace ellipse_area_calc_l366_366595

noncomputable def ellipse_area (a b : ℝ) : ℝ :=
  real.pi * a * b

theorem ellipse_area_calc :
  let center := (5 : ℝ, 2 : ℝ)
  let semi_major_axis := 10
  let point_on_ellipse := (13 : ℝ, 6 : ℝ)
  let b := 20 / 3
  in ellipse_area semi_major_axis b = (200 * real.pi) / 3 :=
by
  let center := (5 : ℝ, 2 : ℝ)
  let semi_major_axis := 10
  let point_on_ellipse := (13 : ℝ, 6 : ℝ)
  let b := 20 / 3
  have h : ellipse_area semi_major_axis b = (200 * real.pi) / 3, from sorry
  exact h

end ellipse_area_calc_l366_366595


namespace weight_difference_l366_366307

open Real

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h1 : (W_A + W_B + W_C) / 3 = 50)
  (h2 : W_A = 73)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 53)
  (h4 : (W_B + W_C + W_D + W_E) / 4 = 51) :
  W_E - W_D = 3 := 
sorry

end weight_difference_l366_366307


namespace interior_diagonal_length_correct_l366_366419

structure RectangularSolid where
  x y z : ℝ

def total_surface_area (r : RectangularSolid) : ℝ :=
  2 * (r.x * r.y + r.y * r.z + r.z * r.x)

def total_edge_length (r : RectangularSolid) : ℝ :=
  4 * (r.x + r.y + r.z)

def interior_diagonal_length (r : RectangularSolid) : ℝ :=
  (r.x ^ 2 + r.y ^ 2 + r.z ^ 2).sqrt

theorem interior_diagonal_length_correct (r : RectangularSolid)
  (h_surface_area : total_surface_area r = 34)
  (h_edge_length : total_edge_length r = 28) :
  interior_diagonal_length r = √15 := by
  sorry

end interior_diagonal_length_correct_l366_366419


namespace lcm_18_24_l366_366748

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366748


namespace alternating_sum_l366_366442

theorem alternating_sum : 
  (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19 + 21 - 23 + 25 - 27 + 29 - 31 + 33 - 35 + 37 - 39 + 41 = 21) :=
by
  sorry

end alternating_sum_l366_366442


namespace lcm_18_24_l366_366745

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366745


namespace exists_natural_number_l366_366620

theorem exists_natural_number (a b : ℕ) (hb : b > 1) : 
  ∃ n < b^2, (a^n + n) % b = 0 := 
begin
  sorry
end

end exists_natural_number_l366_366620


namespace area_of_rectangle_l366_366203

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366203


namespace sum_expression_final_value_l366_366625

theorem sum_expression :
  (∑ n in Finset.range 100 \+ 1, (1 : ℝ) / (Real.sqrt (n + Real.sqrt (n^2 - 1/4 : ℝ)))) = 6 + 4 * Real.sqrt 2 :=
sorry

theorem final_value :
  let (a, b, c) := (6, 4, 2) in
  a + b + c = 12 :=
rfl

end sum_expression_final_value_l366_366625


namespace num_positive_divisors_36_l366_366094

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366094


namespace red_section_no_damage_probability_l366_366457

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l366_366457


namespace area_of_rectangle_ABCD_l366_366179

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366179


namespace John_surveyed_total_people_l366_366614

theorem John_surveyed_total_people :
  ∃ P D : ℝ, 
  0 ≤ P ∧ 
  D = 0.868 * P ∧ 
  21 = 0.457 * D ∧ 
  P = 53 :=
by
  sorry

end John_surveyed_total_people_l366_366614


namespace divide_numbers_into_consecutive_products_l366_366919

theorem divide_numbers_into_consecutive_products :
  ∃ (A B : Finset ℕ), A ∪ B = {2, 3, 5, 7, 11, 13, 17} ∧ A ∩ B = ∅ ∧ 
  (A.prod id = 714 ∧ B.prod id = 715 ∨ A.prod id = 715 ∧ B.prod id = 714) :=
sorry

end divide_numbers_into_consecutive_products_l366_366919


namespace num_pos_divisors_36_l366_366060

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366060


namespace lcm_18_24_eq_72_l366_366776

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366776


namespace lcm_18_24_eq_72_l366_366780

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366780


namespace pyramid_volume_calculation_l366_366403

-- Define the conditions of the problem
def cube_edge_length : ℝ := 2
def cube_volume : ℝ := cube_edge_length^3

-- Definitions related to the cuts
def pyramid_base_edge_length : ℝ := cube_edge_length / Real.sqrt 2
def pyramid_base_area : ℝ := pyramid_base_edge_length^2
def pyramid_height : ℝ := cube_edge_length / 2

-- Define the correct answer
def pyramid_volume : ℝ := (1/3) * pyramid_base_area * pyramid_height

-- The theorem statement to be proved
theorem pyramid_volume_calculation : pyramid_volume = 2 / 3 := by
  sorry

end pyramid_volume_calculation_l366_366403


namespace calculate_value_l366_366901

theorem calculate_value : 2 * (75 * 1313 - 25 * 1313) = 131300 := 
by 
  sorry

end calculate_value_l366_366901


namespace number_of_incorrect_statements_l366_366700

-- Definitions of the statements as conditions
def statement_1 : Prop := ∀ (data : List ℝ) (c : ℝ), 
  let data' := data.map (λ x, x + c) in 
  (mean data = mean data') ∧ (variance data = variance data')

def statement_2 : Prop := ∀ (x : ℝ), 
  let y := 5 - 3 * x in 
  (y + 3) = 5 - 3 * (x + 1)

def statement_3 : Prop := ∀ (b a : ℝ) (x y : ℝ), 
  let x̄ := x in 
  let ȳ := y in 
  (ȳ = b * x̄ + a)

def statement_4 : Prop := ∀ (p : ℝ), 
  p = 0.99 → 
  ∀ (n : ℕ), 
  n = 100 → 
  ∃ (k : ℕ), 
  k = 99 ∧
  confidenceTest p n k

-- Problem statement in Lean
theorem number_of_incorrect_statements : (¬statement_1) ∧ (¬statement_2) ∧ statement_3 ∧ (¬statement_4) → 3 := by
  sorry

end number_of_incorrect_statements_l366_366700


namespace problem_230_plus_n_l366_366495

def is_divisible (a b : ℕ) : Prop := b ∣ a

def count_divisible_digits : ℕ :=
  (Finset.range 10).filter (λ n, is_divisible (230 + n) n).card

theorem problem_230_plus_n :
  count_divisible_digits = 3 :=
sorry

end problem_230_plus_n_l366_366495


namespace range_of_x0_l366_366977

theorem range_of_x0 
  (x0 : ℝ)
  (M : ℝ × ℝ)
  (hM : M = (x0, 1))
  (N : ℝ × ℝ)
  (hN_on_circle : N.1 ^ 2 + N.2 ^ 2 = 1)
  (angle_OMN : ℝ)
  (h_angle_OMN : angle_OMN = 30) :
  x0 ∈ Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_x0_l366_366977


namespace cos_A_convex_quadrilateral_l366_366174

theorem cos_A_convex_quadrilateral (ABCD : Quadrilateral)
  (A C : ℝ) (AB CD : ℝ) (AD BC x y : ℝ) (prem : ℝ)
  (h_conv : Convex ABCD)
  (h_angle : ∠A = ∠C)
  (h_AB : AB = 180)
  (h_CD : CD = 180)
  (h_AD_ne_BC : AD ≠ BC)
  (h_prem : AB + BC + CD + DA = 720) :
  cos A = 1 := by
  sorry

end cos_A_convex_quadrilateral_l366_366174


namespace determine_f_2014_l366_366318

open Function

noncomputable def f : ℕ → ℕ :=
  sorry

theorem determine_f_2014
  (h1 : f 2 = 0)
  (h2 : f 3 > 0)
  (h3 : f 6042 = 2014)
  (h4 : ∀ m n : ℕ, f (m + n) - f m - f n ∈ ({0, 1} : Set ℕ)) :
  f 2014 = 671 :=
sorry

end determine_f_2014_l366_366318


namespace original_number_is_80_l366_366838

variable (e : ℝ)

def increased_value := 1.125 * e
def decreased_value := 0.75 * e
def difference_condition := increased_value e - decreased_value e = 30

theorem original_number_is_80 (h : difference_condition e) : e = 80 :=
sorry

end original_number_is_80_l366_366838


namespace total_volume_of_all_cubes_l366_366902

/-- Carl has 4 cubes each with a side length of 3 -/
def carl_cubes_side_length := 3
def carl_cubes_count := 4

/-- Kate has 6 cubes each with a side length of 4 -/
def kate_cubes_side_length := 4
def kate_cubes_count := 6

/-- Total volume of 10 cubes with given conditions -/
theorem total_volume_of_all_cubes : 
  carl_cubes_count * (carl_cubes_side_length ^ 3) + 
  kate_cubes_count * (kate_cubes_side_length ^ 3) = 492 := by
  sorry

end total_volume_of_all_cubes_l366_366902


namespace num_pos_divisors_36_l366_366101

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366101


namespace number_of_divisors_36_l366_366087

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366087


namespace Bernado_wins_n_101_l366_366498

/-- Bernado can win if n = 101, meaning he can turn all points to 0. -/
theorem Bernado_wins_n_101 :
  ∀ (points : Fin 101 → Bool),
  (∃ f : Fin 101 → Bool → Bool, (∀ i, (f i (points i) = points i) ∨ (f i (points i) = ¬points i))) :=
begin
  sorry
end

end Bernado_wins_n_101_l366_366498


namespace oak_trees_remaining_l366_366721

-- Variables representing the initial number of oak trees and the number of cut down trees.
variables (initial_trees cut_down_trees remaining_trees : ℕ)

-- Conditions of the problem.
def initial_trees_condition : initial_trees = 9 := sorry
def cut_down_trees_condition : cut_down_trees = 2 := sorry

-- Theorem representing the proof problem.
theorem oak_trees_remaining (h1 : initial_trees = 9) (h2 : cut_down_trees = 2) :
  remaining_trees = initial_trees - cut_down_trees :=
sorry

end oak_trees_remaining_l366_366721


namespace lemonade_syrup_percentage_after_adjustment_l366_366421

theorem lemonade_syrup_percentage_after_adjustment
  (initial_water : ℚ) (initial_syrup : ℚ) (removed_solution : ℚ) (added_water : ℚ)
  (h_parts : initial_water = 8) (h_syrup : initial_syrup = 7) (h_removed : removed_solution = 2.1428571428571423) (h_added : added_water = 2.1428571428571423) :
  let total_initial_solution := initial_water + initial_syrup,
      total_new_solution := initial_water + initial_syrup - removed_solution + added_water,
      syrup_percentage := (initial_syrup / total_new_solution) * 100
  in syrup_percentage ≈ 46.67 :=
by {
  let total_initial_solution := initial_water + initial_syrup,
  let total_new_solution := initial_water + initial_syrup - removed_solution + added_water,
  let syrup_percentage := (initial_syrup / total_new_solution) * 100,
  have h1 : total_initial_solution = 15, from by linarith [h_parts, h_syrup],
  have h2 : total_new_solution = 15, from by linarith [h1, h_removed, h_added],
  have h3 : syrup_percentage = (initial_syrup / 15) * 100, from by linarith [],
  have h4 : syrup_percentage = 46.66666666666667, from by norm_num [h3, h_syrup],
  exact h4,
  sorry
}

end lemonade_syrup_percentage_after_adjustment_l366_366421


namespace B_additional_days_to_finish_work_l366_366383

-- Definitions of the rates at which A and B individually complete the work
def A_rate : ℝ := 1 / 4
def B_rate : ℝ := 1 / 8

-- Combined rate when A and B work together
def combined_rate : ℝ := A_rate + B_rate

-- Work completed by A and B together in 2 days
def work_done_together : ℝ := 2 * combined_rate

-- Remaining work after 2 days
def remaining_work : ℝ := 1 - work_done_together

-- Time taken by B to finish the remaining work
theorem B_additional_days_to_finish_work : remaining_work / B_rate = 2 := 
by
  show (1 - work_done_together) / B_rate = 2
  show (1 - (2 * (1 / 4 + 1 / 8))) / (1 / 8) = 2
  sorry

end B_additional_days_to_finish_work_l366_366383


namespace rabbit_distributions_l366_366270

/-- There are six rabbits: Peter, Pauline, Flopsie, Mopsie, Cotton-tail, and Silver-ear.
    These rabbits are to be distributed to five different pet stores such that:
    1. No store gets both a parent (Peter or Pauline) and a child (Flopsie, Mopsie, Cotton-tail, or Silver-ear).
    2. At least one store remains empty.
    The total number of ways to distribute the rabbits is 456. -/
theorem rabbit_distributions : 
    let rabbits := ["Peter", "Pauline", "Flopsie", "Mopsie", "Cotton-tail", "Silver-ear"] in
    let parents := ["Peter", "Pauline"] in
    let children := ["Flopsie", "Mopsie", "Cotton-tail", "Silver-ear"] in
    let stores := 5 in
    ∃ (ways : ℕ), ways = 456 ∧
    (ways = 
    let distrib_ways (rabbits : List String) (stores : ℕ) :=
        -- the counting logic to ensure conditions is handled here
        sorry 
    in distrib_ways rabbits stores) := 
456 := 
  sorry

end rabbit_distributions_l366_366270


namespace divisible_by_91_l366_366702

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 202020
  | _ => -- Define the sequence here, ensuring it constructs the number properly with inserted '2's
    sorry -- this might be a more complex function to define

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n :=
  sorry

end divisible_by_91_l366_366702


namespace Gabrielle_sells_8_crates_on_Wednesday_l366_366950

-- Definitions based on conditions from part a)
def crates_sold_on_Monday := 5
def crates_sold_on_Tuesday := 2 * crates_sold_on_Monday
def crates_sold_on_Thursday := crates_sold_on_Tuesday / 2
def total_crates_sold := 28
def crates_sold_on_Wednesday := total_crates_sold - (crates_sold_on_Monday + crates_sold_on_Tuesday + crates_sold_on_Thursday)

-- The theorem to prove the question == answer given conditions
theorem Gabrielle_sells_8_crates_on_Wednesday : crates_sold_on_Wednesday = 8 := by
  sorry

end Gabrielle_sells_8_crates_on_Wednesday_l366_366950


namespace parabola_intersection_lambda_l366_366996

theorem parabola_intersection_lambda 
  (p : ℝ) (h1 : p > 0)
  (h2 : (1, -2*real.sqrt 2), (4, 4*real.sqrt 2) ∈ { (x, y) | y^2 = 2*p*x })
  (h3 : 9 = real.sqrt ((4 - 1)^2 + ((4*real.sqrt 2) - (-2*real.sqrt 2))^2)) :
  ∃ λ : ℝ, λ = 2 :=
by
  sorry

end parabola_intersection_lambda_l366_366996


namespace probability_of_pink_tie_l366_366400

theorem probability_of_pink_tie 
  (black_ties gold_ties pink_ties : ℕ) 
  (h_black : black_ties = 5) 
  (h_gold : gold_ties = 7) 
  (h_pink : pink_ties = 8) 
  (h_total : (5 + 7 + 8) = (black_ties + gold_ties + pink_ties)) 
  : (pink_ties : ℚ) / (black_ties + gold_ties + pink_ties) = 2 / 5 := 
by 
  sorry

end probability_of_pink_tie_l366_366400


namespace rectangle_area_l366_366207

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366207


namespace factory_profit_maximization_l366_366409

-- Definitions based on problem conditions
def total_masks := 50000
def min_type_A_masks := 18000
def type_A_production_per_day := 6000
def type_B_production_per_day := 8000
def type_A_profit_per_mask := 0.5
def type_B_profit_per_mask := 0.3
def days := 8

-- Conditions from part a)
axiom task_conditions (x : ℝ) : 
  1.8 ≤ x ∧ x ≤ 4.2
  ∧ (type_A_production_per_day * days ≥ (x * 10000))
  ∧ (∃ days_type_A : ℝ, days_type_A ≤ days 
     ∧ (type_A_production_per_day * days_type_A = (x * 10000)))

-- Theorem from parts b) and c)
theorem factory_profit_maximization (x : ℝ) (y : ℝ) : 
  task_conditions x → 
  y = (type_A_profit_per_mask * x + type_B_profit_per_mask * (5 - x)) 
  → (y ≤ 2.34)  
  → (x = 4.2)
  → ((type_A_production_per_day * 7 = 42000) ∧ (type_B_production_per_day * 1 = 8000)) 
  ∧ y = 0.2 * x + 1.5 
  ∧ x = 4.2 
  ∧ y = 2.34
  ∧ (1.8 * (6000 / 10000) + 3.2 * (8000 / 10000) = 7) := 
begin
  sorry, -- Proof to be completed.
end

end factory_profit_maximization_l366_366409


namespace red_section_not_damaged_l366_366462

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l366_366462


namespace greatest_prime_factor_l366_366486

theorem greatest_prime_factor : 
  let series_sum := (Finset.range 50).sum (λ n, if n % 2 = 0 then (2 * n + 1) ^ 2 else -(2 * n * (2 * n + 2))) + 99 ^ 2
  in Prime.max_factor series_sum = 11 :=
by
  simp only [Finset.sum, Finset.range, Function.comp, Nat.add_sub_assoc] at series_sum
  let terms : List Int := List.range (2 * 50) |>.map (λ n, if n % 2 = 0 then -(n*(n+2)) else (n+1)^2);
  let last_term : Int := 99^2;
  have : List.sum (terms ++ [last_term]) = series_sum := by sorry;
  exact Prime.max_factor series_sum

end greatest_prime_factor_l366_366486


namespace largest_negative_integer_solution_l366_366487

theorem largest_negative_integer_solution :
  ∃ x : ℤ, x < 0 ∧ 50 * x + 14 % 24 = 10 % 24 ∧ ∀ y : ℤ, (y < 0 ∧ y % 12 = 10 % 12 → y ≤ x) :=
by
  sorry

end largest_negative_integer_solution_l366_366487


namespace arith_seq_common_diff_l366_366153

/-
Given:
- an arithmetic sequence {a_n} with common difference d,
- the sum of the first n terms S_n = n * a_1 + n * (n - 1) / 2 * d,
- b_n = S_n / n,

Prove that the common difference of the sequence {a_n - b_n} is d/2.
-/

theorem arith_seq_common_diff (a b : ℕ → ℚ) (a1 d : ℚ) 
  (h1 : ∀ n, a n = a1 + n * d) 
  (h2 : ∀ n, b n = (a1 + n - 1 * d + n * (n - 1) / 2 * d) / n) : 
  ∀ n, (a n - b n) - (a (n + 1) - b (n + 1)) = d / 2 := 
    sorry

end arith_seq_common_diff_l366_366153


namespace same_polar_coordinate_l366_366017

theorem same_polar_coordinate (r : ℝ) : (r > 0) -> (π/3 = 2 * π - 5 * π / 3) :=
by
  intro hr
  calc
    π/3 = 2 * π - 5 * π / 3 : by sorry

end same_polar_coordinate_l366_366017


namespace johns_trip_distance_is_160_l366_366615

noncomputable def total_distance (y : ℕ) : Prop :=
  y / 2 + 40 + y / 4 = y

theorem johns_trip_distance_is_160 : ∃ y : ℕ, total_distance y ∧ y = 160 :=
by
  use 160
  unfold total_distance
  sorry

end johns_trip_distance_is_160_l366_366615


namespace largest_number_is_B_l366_366912

noncomputable def Option_A := 8.03456
noncomputable def Option_B := 8.034666666... -- Equivalent to 8.034\overline{6}
noncomputable def Option_C := 8.0345454545... -- Equivalent to 8.03\overline{45}
noncomputable def Option_D := 8.034563456... -- Equivalent to 8.0\overline{3456}
noncomputable def Option_E := 8.0345603456... -- Equivalent to 8.\overline{03456}

theorem largest_number_is_B : 
  Option_B > Option_A ∧ 
  Option_B > Option_C ∧ 
  Option_B > Option_D ∧ 
  Option_B > Option_E := 
by {
  sorry
}

end largest_number_is_B_l366_366912


namespace lcm_18_24_eq_72_l366_366811

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366811


namespace neg_P_neither_sufficient_nor_necessary_for_neg_Q_l366_366524

-- Let a be a real number
variable (a : ℝ)

-- Define condition P and Q
def P := a > 0
def Q := a^2 > a

-- Define negation of P and Q
def not_P := ¬ P
def not_Q := ¬ Q

-- The proof problem statement
theorem neg_P_neither_sufficient_nor_necessary_for_neg_Q :
  ¬ (¬ P → ¬ Q) ∧ ¬ (¬ Q → ¬ P) :=
by
  sorry

end neg_P_neither_sufficient_nor_necessary_for_neg_Q_l366_366524


namespace reciprocal_of_F_is_C_l366_366832

open Complex

-- Define the problem conditions
variables (a b : ℝ) (F : ℂ)
hypothesis (h1 : 0 < a) (h2 : 0 < b) (h3 : ∥F∥ > 1) (hF : F = a + bi)

-- Define point C as a complex number that lies in the fourth quadrant inside the unit circle
def pointC : ℂ := cre ⟨_, _⟩  -- Define the specific values for point C

-- State the theorem
theorem reciprocal_of_F_is_C :
  (1 / F) = pointC :=
by
  sorry

end reciprocal_of_F_is_C_l366_366832


namespace lcm_18_24_l366_366761
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366761


namespace count_irrationals_l366_366592

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem count_irrationals :
  let l := [33 / 17, Real.sqrt 3, - (2 ^ (1 / 3 : ℝ)), Real.pi, -3.030030003, 2023] in
  (l.filter is_irrational).length = 3 :=
by
  -- This is a statement only, the proof is omitted using sorry.
  sorry

end count_irrationals_l366_366592


namespace trajectory_equation_max_area_triangle_l366_366604

-- Definition of trajectory Γ
def trajectory (x y : ℝ) : Prop :=
  y ≠ 0 ∧ (x ^ 2 / 4 + y ^ 2 = 1)

-- Statement of the first question
theorem trajectory_equation (x y : ℝ) (h : trajectory x y) : 
  x^2 / 4 + y^2 = 1 := 
  by 
    have h1 : y ≠ 0 := h.left
    have h2 : x^2 / 4 + y^2 = 1 := h.right
    exact h2

-- Statement of the second question
theorem max_area_triangle (b : ℝ) (h : 5 - b^2 > 0) : 
  ∃ (S : ℝ), S = 2/5 * (b + 3) * √(5 - b^2) ∧ 
  S ≤ 16/5 :=
  by 
    use 2/5 * (b + 3) * √(5 - b^2)
    split
    case a => rfl
    case b => sorry

end trajectory_equation_max_area_triangle_l366_366604


namespace exists_symmetry_point_with_distinct_colors_l366_366233

def is_red (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = 19 * a + 85 * b

def is_green (n : ℕ) : Prop :=
  ¬ is_red n

noncomputable def midpoint (n m : ℤ) : ℚ :=
  (n + m) / 2

theorem exists_symmetry_point_with_distinct_colors :
  ∃ (A : ℚ), ∀ (c : ℤ), (is_red (c.to_nat) ∧ is_green ((2 * A - c).to_int)) ∨
                    (is_green (c.to_nat) ∧ is_red ((2 * A - c).to_int)) :=
sorry

end exists_symmetry_point_with_distinct_colors_l366_366233


namespace completing_the_square_l366_366378

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l366_366378


namespace difference_of_roots_eq_one_l366_366905

theorem difference_of_roots_eq_one (p : ℝ) :
  let a := 1
  let b := -(2 * p + 1)
  let c := p * (p + 1)
  let Δ := b^2 - 4 * a * c
  Δ = 1 → (let root1 := (2 * p + 1 + real.sqrt Δ) / 2
           let root2 := (2 * p + 1 - real.sqrt Δ) / 2
           (root1 - root2) = 1) :=
by
  sorry

end difference_of_roots_eq_one_l366_366905


namespace sweets_leftover_candies_l366_366478

theorem sweets_leftover_candies (n : ℕ) (h : n % 8 = 5) : (3 * n) % 8 = 7 :=
sorry

end sweets_leftover_candies_l366_366478


namespace rectangle_area_l366_366196

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366196


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366362

theorem remainder_3001_3002_3003_3004_3005_mod_17 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := 
begin
  sorry
end

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366362


namespace rectangle_area_l366_366218

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366218


namespace B_first_choice_members_l366_366857

/-- Given conditions and variables -/
variables (p q r s t u : ℕ)

/-- Given conditions -/
constants
  (totalMembers : p + q + r + s + t + u = 20) 
  (preferAtoB : p + q + r = 11)
  (preferCtoA : r + s + t = 12)
  (preferBtoC : p + t + u = 14)

/-- Theorem stating the number of members who have B as their first choice is 8 -/
theorem B_first_choice_members : t + u = 8 :=
by
  sorry

end B_first_choice_members_l366_366857


namespace number_of_true_statements_l366_366551

theorem number_of_true_statements : 
  let P := (λ x : ℝ, x^2 > 1), Q := (λ x : ℝ, x > 1) in
  let original_statement := ∀ x, P x → Q x in
  let contrapositive := ∀ x, ¬ Q x → ¬ P x in
  let converse := ∀ x, Q x → P x in
  let negation := ∃ x, P x ∧ ¬ Q x in
  (converse ∧ negation ∧ ¬original_statement ∧ ¬contrapositive) → 
  2 = 2 :=
by sorry

end number_of_true_statements_l366_366551


namespace length_of_chord_proof_l366_366940

noncomputable def length_of_chord : ℝ := 
  let line := { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 1 = 0 }
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 - p.1 + p.2 = 0 }
  let center := (1 / 2, -1 / 2)
  let radius := sqrt 2 / 2
  let distance := 1 / 10
  2 * sqrt (radius^2 - distance^2)

theorem length_of_chord_proof :
  length_of_chord = 7 / 5 := by
  sorry

end length_of_chord_proof_l366_366940


namespace differential_equation_solution_exists_l366_366309

theorem differential_equation_solution_exists (f : ℝ → ℝ) (continuous_diff : Continuous f ∧ Differentiable ℝ f ∧ Differentiable ℝ (deriv f)) 
  (eqn : ∀ x : ℝ, 2007^2 * f x + (deriv (deriv f)) x = 0) :
  ∃ (k l : ℝ), ∀ x : ℝ, f x = l * sin (2007 * x) + k * cos (2007 * x) :=
by
  sorry

end differential_equation_solution_exists_l366_366309


namespace find_C_l366_366261

open Real

-- Define points and their properties.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨1, -2⟩
def B : Point := ⟨7, 2⟩

-- Predicate for a point on the line segment between two other points
def on_segment (C A B : Point) : Prop :=
  ∃ t ∈ Icc (0:ℝ) 1, C = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y)⟩

-- Distance function between two points
def dist (P Q : Point) : ℝ :=
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Proof statement
theorem find_C :
  ∃ C : Point, on_segment C A B ∧ 2 * (dist C A) = dist C B ∧ C = ⟨5, 2 / 3⟩ :=
by
  sorry

end find_C_l366_366261


namespace problem_correct_conclusions_l366_366991

variable (a b : ℝ^3)  -- Define vectors a and b in ℝ^3

-- Given condition: non-zero, non-collinear vectors a and b satisfy |a - b| = |b|
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom noncollinear_ab : ¬collinear a b
axiom condition : ∥a - b∥ = ∥b∥  -- This is the norm (magnitude) condition

-- Use the axiom of real non-negative definiteness and vector operations in ℝ^3
noncomputable def count_correct_conclusions : ℝ :=
if ((2 * ∥b∥^2 > a • b) ∧ (∥2 * a∥ < ∥2 * a - b∥)) then
  2  -- The count of correct conclusions (1 and 4) is exactly 2
else
  (if ((2 * ∥b∥^2 > a • b) ∨ (∥2 * a∥ < ∥2 * a - b∥)) then
     1  -- we might need to change this as per the sidualtison
   else
     0  -- No correct conclusions

-- The theorem to prove
theorem problem_correct_conclusions : count_correct_conclusions a b = 2 := by
  sorry

end problem_correct_conclusions_l366_366991


namespace mike_total_time_spent_l366_366253

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l366_366253


namespace total_hours_correct_l366_366255

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l366_366255


namespace find_f_neg_one_l366_366992

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)

theorem find_f_neg_one (k : ℝ) (h : ∀ (x : ℝ), f k (-x) = -f k x) : f k (-1) = 2 :=
sorry

end find_f_neg_one_l366_366992


namespace cyclic_quadrilateral_theorem_l366_366163

variables {A B C D M N I : Type}
variables [affine_space A C M N] [metric_space C]

-- Definitions based on the given conditions
def cyclic_quadrilateral (A B C D : Type) : Prop := ∃ O, metric_space.circle O A = metric_space.circle O B ∧
  metric_space.circle O B = metric_space.circle O C ∧ metric_space.circle O C = metric_space.circle O D

def incenter_of_triangle (B C M : Type) : Type := I

def points_concyclic (M N B I : Type) : Prop := ∃ O (r : ℝ), 
  (dist O M = r) ∧ (dist O N = r) ∧ (dist O B = r) ∧ (dist O I = r)

def intersection_point (A C B D : Type) : Type := M  

-- The problem statement translated to Lean
theorem cyclic_quadrilateral_theorem 
  (ABCD_cyclic : cyclic_quadrilateral A B C D)
  (AD_eq_BD : dist A D = dist B D)
  (M_is_intersection : intersection_point A C B D = M)
  (N_on_AC : ∃ N, N ∈ line A C ∧ N ≠ M)
  (M_N_B_I_concyclic : points_concyclic M N B (incenter_of_triangle B C M)) :
  dist A N * dist N C = dist C D * dist B N :=
sorry

end cyclic_quadrilateral_theorem_l366_366163


namespace number_of_divisors_36_l366_366051

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366051


namespace range_of_alpha_plus_3beta_l366_366009

theorem range_of_alpha_plus_3beta (α β : ℝ) 
  (h1 : -1 ≤ α + β ∧ α + β ≤ 1)
  (h2 : 1 ≤ α + 2β ∧ α + 2β ≤ 3) : 
  1 ≤ α + 3 * β ∧ α + 3 * β ≤ 7 := 
sorry

end range_of_alpha_plus_3beta_l366_366009


namespace abc_minimizes_variance_l366_366021

-- Definitions based on problem conditions
def population : list ℝ := [c, 3, 3, 8, a, b, 12, 13.7, 18.3, 20]

def median (l : list ℝ) : ℝ := (l[l.length/2 - 1] + l[l.length/2]) / 2
def mean (l : list ℝ) : ℝ := list.sum l / l.length

-- Lean statement, proving that the product abc equals 200 under given conditions
theorem abc_minimizes_variance (c a b : ℝ)
  (h_sorted : list.sort population = [c, 3, 3, 8, a, b, 12, 13.7, 18.3, 20])
  (h_median : median population = 10)
  (h_mean : mean population = 10) :
  a*b*c = 200 :=
by {
  sorry
}

end abc_minimizes_variance_l366_366021


namespace find_a_l366_366696

theorem find_a (a : ℝ) : 
  let line1 := λ (x y : ℝ), 4 * x + 3 * y - 6
      line2 := λ (x y : ℝ), 4 * x + 3 * y + a
      distance := (|(-6 : ℝ) - a| / (real.sqrt (4^2 + 3^2)))
  in distance = 2 → (a = 4 ∨ a = -16) := by
sorry

end find_a_l366_366696


namespace number_of_divisors_36_l366_366050

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366050


namespace cos_alpha_plus_pi_over_3_l366_366964

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h1 : sin α = 1 / 3) (h2 : α ∈ set.Ioc (π/2) π) :
  cos (α + π / 3) = - (2 * sqrt 2 + sqrt 3) / 6 :=
sorry

end cos_alpha_plus_pi_over_3_l366_366964


namespace probability_intersection_intersection_probability_l366_366277

theorem probability_intersection (k : ℝ) (hk : -2 ≤ k ∧ k ≤ 3) :
  (∃ y, ∃ x, x^2 + (y + 2)^2 = 9 ∧ y = k*x + 3) ↔ (k ≤ -4/3 ∨ k ≥ 4/3) :=
sorry

theorem intersection_probability : 
  (∫ k in -2..3, if k ≤ -4/3 ∨ k ≥ 4/3 then 1 else 0) / (∫ k in -2..3, 1) = 7 / 15 :=
sorry

end probability_intersection_intersection_probability_l366_366277


namespace ral_age_is_26_l366_366274

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l366_366274


namespace system_solution_l366_366289

noncomputable def solve_system (x y : ℝ) : Prop :=
  (x + y = 20) ∧ (Real.logBase 4 x + Real.logBase 4 y = 1 + Real.logBase 4 9) ∧
  ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18))

theorem system_solution : ∃ x y : ℝ, solve_system x y :=
  sorry

end system_solution_l366_366289


namespace min_distance_AB_l366_366664

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_AB : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = (8 / 15) * A.1 - 6) ∧ 
    (B.2 = B.1 ^ 2) ∧ 
    distance A.1 A.2 B.1 B.2 = 1334 / 255 :=
sorry

end min_distance_AB_l366_366664


namespace function_is_specific_polynomial_intervals_of_monotonicity_max_min_values_correct_l366_366028

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + 5

-- Given conditions
variable (a b : ℝ)
variable h1 : f' (-1) = 0
variable h2 : f' (3 / 2) = 0

-- The function derived from the conditions should be the specific polynomial
theorem function_is_specific_polynomial 
  (a b : ℝ) 
  (h1 : f' (-1) = 0) 
  (h2 : f' (3 / 2) = 0) : 
  a = -3 ∧ b = -18 := 
by {
  -- Proof goes here
  sorry
}

-- Establish the intervals of monotonicity
theorem intervals_of_monotonicity (a b : ℝ) 
  (h1 : f' (-1) = 0) 
  (h2 : f' (3 / 2) = 0) : 
  (∀ x < -1, f' x > 0) ∧ (∀ x > (3 / 2), f' x > 0) ∧ 
  (∀ x, -1 < x ∧ x < (3 / 2) → f' x < 0) := 
by {
  -- Proof goes here
  sorry
}

-- Establish the maximum and minimum values on the interval [-1,2]
theorem max_min_values_correct (a b : ℝ) 
  (h1 : f' (-1) = 0) 
  (h2 : f' (3 / 2) = 0) : 
  f (-1) = 16 ∧ f (3 / 2) = -61 / 4 :=
by {
  -- Proof goes here
  sorry
}

end function_is_specific_polynomial_intervals_of_monotonicity_max_min_values_correct_l366_366028


namespace y_coordinate_equidistant_l366_366349

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end y_coordinate_equidistant_l366_366349


namespace travel_time_l366_366865

theorem travel_time (v : ℝ) (d : ℝ) (t : ℝ) (hv : v = 65) (hd : d = 195) : t = 3 :=
by
  sorry

end travel_time_l366_366865


namespace lunks_needed_for_apples_l366_366143

theorem lunks_needed_for_apples : (20 : ℕ) * (3 / 5 : ℚ) * (7 / 4 : ℚ) = (21 : ℚ) := by
  -- We need to compute the number of kunks needed for 20 apples.
  -- First compute the number of kunks needed for one apple.
  have h1 : (1 : ℚ) = (5 / 5 : ℚ), by norm_num
  have apple_eq_kunks : (20 : ℚ) * (3 / 5 : ℚ) = (12 : ℚ), by norm_num
  -- Then compute the number of lunks needed for the 12 kunks.
  have kunks_eq_lunks : (12 : ℚ) * (7 / 4 : ℚ) = (21 : ℚ), by norm_num
  -- Combining the results, we get:
  show (20 : ℚ) * (3 / 5 : ℚ) * (7 / 4 : ℚ) = (21 : ℚ), by rw [apple_eq_kunks, kunks_eq_lunks]
  sorry

end lunks_needed_for_apples_l366_366143


namespace number_of_divisors_36_l366_366086

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366086


namespace system_solution_l366_366290

noncomputable def solve_system (x y : ℝ) : Prop :=
  (x + y = 20) ∧ (Real.logBase 4 x + Real.logBase 4 y = 1 + Real.logBase 4 9) ∧
  ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18))

theorem system_solution : ∃ x y : ℝ, solve_system x y :=
  sorry

end system_solution_l366_366290


namespace subsets_with_mean_equal_5_l366_366565

open Finset

noncomputable def original_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem subsets_with_mean_equal_5 : 
  (card (filter (λ t, (t ∈ (original_set.subsets.filter (λ s, s.card = 2))) ∧ 
    (original_set.sum - t.sum) / ( original_set.card - 2) = 5) (original_set.subsets))) = 4 := 
sorry

end subsets_with_mean_equal_5_l366_366565


namespace best_loan_option_l366_366889

def loan_amount : ℝ := 65000
def loan_term : ℝ := 1 -- in years

-- Option a
def option_a_amount : ℝ := 68380

-- Option b: Simple Interest
def option_b_rate : ℝ := 0.05
def option_b_interest : ℝ := loan_amount * option_b_rate
def option_b_amount : ℝ := loan_amount + option_b_interest

-- Option c: Compound Interest Monthly
def option_c_rate : ℝ := 0.05
def option_c_periods : ℝ := 12
def option_c_amount : ℝ := loan_amount * (1 + option_c_rate / option_c_periods) ^ option_c_periods

-- Option d: Compound Interest Quarterly
def option_d_rate : ℝ := 0.05
def option_d_periods : ℝ := 4
def option_d_amount : ℝ := loan_amount * (1 + option_d_rate / option_d_periods) ^ option_d_periods

-- Option e: Compound Interest Semi-Annually
def option_e_rate : ℝ := 0.05
def option_e_periods : ℝ := 2
def option_e_amount : ℝ := loan_amount * (1 + option_e_rate / option_e_periods) ^ option_e_periods

-- Prove that option b is the most advantageous
theorem best_loan_option : option_b_amount = min option_a_amount (min option_c_amount (min option_d_amount option_e_amount)) :=
sorry

end best_loan_option_l366_366889


namespace num_multiples_of_15_l366_366560

theorem num_multiples_of_15 (a b m : ℕ) (h1 : a = 25) (h2 : b = 200) (h3 : m = 15) : 
  ∃ n, n = (b - a) / m + 1 ∧ ∃ k, k = (n - 1) * m + a ∧ k mod m = 0 ∧ a < k ∧ k < b := by
  sorry

end num_multiples_of_15_l366_366560


namespace question_1_question_2_l366_366541

-- Define the function f and its conditions
def f (x : ℝ) (a : ℝ) : ℝ := (Real.log (a * x + 1)) + (x^3) - (x^2) - (a * x)

-- The first problem statement
theorem question_1 (a : ℝ) (h_extremum : is_extremum (f (2/3) a)) : a = 0 :=
sorry

-- Define the function g and its conditions with a fixed a = -1
def g (x : ℝ) : ℝ := (Real.log (1 - x)) + (1 - x) - (1 - x)^2

-- The second problem statement
theorem question_2 (b : ℝ) (has_real_roots : ∃ x : ℝ, g x = b) : b ∈ Iic 0 :=
sorry

end question_1_question_2_l366_366541


namespace common_ratio_l366_366936

namespace GeometricSeries

-- Definitions
def a1 : ℚ := 4 / 7
def a2 : ℚ := 16 / 49 

-- Proposition
theorem common_ratio : (a2 / a1) = (4 / 7) :=
by
  sorry

end GeometricSeries

end common_ratio_l366_366936


namespace round_3456_to_nearest_hundredth_l366_366669

theorem round_3456_to_nearest_hundredth : Real :=
  let x := 3.456
  prove round_to_nearest_hundredth x = 3.46 by sorry

end round_3456_to_nearest_hundredth_l366_366669


namespace points_symmetric_ac_l366_366605

variable (A B C D I J X Y : Point)
variables [InCircle A B C D] [Cyclic ABCD] 
variables [Incenter I ABC] [Incenter J ADC]
variables [DiameterCircle A C X] [DiameterCircle A C Y]
variables [OnSegment IB X] [OnExtension JD Y]
variables [Concyclic B I J D]

theorem points_symmetric_ac 
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : A ≠ D)
  (h5 : AB > BC) (h6 : AD > DC) 
  (h7 : BIC = DID) :
  symmetric_ac X Y :=
sorry

end points_symmetric_ac_l366_366605


namespace find_parabola_equation_l366_366536

open Classical

noncomputable theory

structure Parabola :=
(vertex : ℝ × ℝ)
(focus : ℝ × ℝ)
(equation : ℕ → (ℝ × ℝ) → Prop)

axiom parabola_vertex_origin (P : Parabola) : P.vertex = (0, 0)
axiom parabola_focus_x_axis (P : Parabola) : ∃ x : ℝ, P.focus = (x, 0)
axiom parabola_contains_triangle_vertices 
  (P : Parabola) 
  (A B C : ℝ × ℝ) 
  (hA : P.equation 2 A)
  (hB : P.equation 2 B)
  (hC : P.equation 2 C) 
  : ∃ F : (ℝ × ℝ), F = P.focus ∧ ∃ weight : ℝ, 0 < weight ∧ F = (B.1 + C.1 + A.1) / 3 ∧  F = (B.2 + C.2 + A.2) / 3
axiom line_bc (B C : ℝ × ℝ) : ∃ line_coeff : ℝ × ℝ × ℝ, line_coeff = (4, 1, -20) ∧ line_coeff.1 * B.1 + line_coeff.2 * B.2 + line_coeff.3 = 0 ∧ line_coeff.1 * C.1 + line_coeff.2 * C.2 + line_coeff.3 = 0

theorem find_parabola_equation : ∀ (P : Parabola) A B C : ℝ × ℝ,
  parabola_vertex_origin P →
  parabola_focus_x_axis P →
  parabola_contains_triangle_vertices P A B C (P.equation 2 A) (P.equation 2 B) (P.equation 2 C) →
  line_bc B C →
  P.equation 2 = λ x y, y^2 = 16 * x 
:= sorry

end find_parabola_equation_l366_366536


namespace correct_choice_l366_366431

def PropA : Prop := ∀ x : ℝ, x^2 + 3 < 0
def PropB : Prop := ∀ x : ℕ, x^2 ≥ 1
def PropC : Prop := ∃ x : ℤ, x^5 < 1
def PropD : Prop := ∃ x : ℚ, x^2 = 3

theorem correct_choice : ¬PropA ∧ ¬PropB ∧ PropC ∧ ¬PropD := by
  sorry

end correct_choice_l366_366431


namespace lcm_18_24_l366_366805

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366805


namespace length_of_top_side_l366_366424

def height_of_trapezoid : ℝ := 8
def area_of_trapezoid : ℝ := 72
def top_side_is_shorter (b : ℝ) : Prop := ∃ t : ℝ, t = b - 6

theorem length_of_top_side (b t : ℝ) (h_height : height_of_trapezoid = 8)
  (h_area : area_of_trapezoid = 72) 
  (h_top_side : top_side_is_shorter b)
  (h_area_formula : (1/2) * (b + t) * 8 = 72) : t = 6 := 
by 
  sorry

end length_of_top_side_l366_366424


namespace seating_arrangements_zero_l366_366429

def valid_seating_arrangements (fixed_person : String) (persons : List String) : Nat :=
  let arrangements := 
    persons.permutations.filter (λ perm => 
      -- Alice refuses to sit next to Bob or Carla
      not ((perm[0] = "Alice" ∧ (perm[1] = "Bob" ∨ perm[1] = "Carla")) 
           ∨ (perm[1] = "Alice" ∧ (perm[0] = "Bob" ∨ perm[0] = "Carla")))
      -- Derek refuses to sit next to Eric
      ∧ not ((perm[2] = "Derek" ∧ perm[3] = "Eric") 
             ∨ (perm[3] = "Derek" ∧ perm[2] = "Eric"))
      -- Carla refuses to sit next to Derek
      ∧ not ((perm[1] = "Carla" ∧ perm[2] = "Derek") 
             ∨ (perm[2] = "Carla" ∧ perm[1] = "Derek")))
  arrangements.length

theorem seating_arrangements_zero :
  valid_seating_arrangements "Alice" ["Bob", "Carla", "Derek", "Eric"] = 0 :=
by {
  sorry
}

end seating_arrangements_zero_l366_366429


namespace num_positive_divisors_36_l366_366096

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366096


namespace value_of_expression_l366_366152

theorem value_of_expression (x y : ℤ) (h1 : x = 1) (h2 : y = 630) : 
  2019 * x - 3 * y - 9 = 120 := 
by
  sorry

end value_of_expression_l366_366152


namespace initial_candies_eq_5045_l366_366831

/-- Given the conditions:
   - Day 1: 1249 candies sold.
   - Day 2: 328 more candies sold than Day 1.
   - Day 3: 275 more candies sold than Day 2.
   - 367 candies left at the end.
   Prove the total initial number of candies is 5045.
-/
theorem initial_candies_eq_5045 
  (day1_sales : ℕ := 1249)
  (day2_diff : ℕ := 328)
  (day3_diff : ℕ := 275)
  (candies_left : ℕ := 367) :
  let day2_sales := day1_sales + day2_diff,
      day3_sales := day2_sales + day3_diff,
      total_sold := day1_sales + day2_sales + day3_sales,
      initial_candies := total_sold + candies_left in
  initial_candies = 5045 := by
  sorry

end initial_candies_eq_5045_l366_366831


namespace total_vehicles_in_a_year_l366_366311

theorem total_vehicles_in_a_year 
  (vehicles_per_month_old : ℕ) 
  (capacity_multiplier_new : ℕ) 
  (increase_percentage_new : ℕ) 
  (months_in_a_year : ℕ) 
  (vehicles_per_month_old_eq : vehicles_per_month_old = 2000) 
  (capacity_multiplier_new_eq : capacity_multiplier_new = 2) 
  (increase_percentage_new_eq : increase_percentage_new = 60) 
  (months_in_a_year_eq : months_in_a_year = 12) : 
  let vehicles_per_year_old := vehicles_per_month_old * months_in_a_year,
      vehicles_per_month_new := capacity_multiplier_new * vehicles_per_month_old + (increase_percentage_new / 100) * vehicles_per_month_old,
      vehicles_per_year_new := vehicles_per_month_new * months_in_a_year in
  vehicles_per_year_old + vehicles_per_year_new = 86400 := by
sorry

end total_vehicles_in_a_year_l366_366311


namespace minimum_value_l366_366994

theorem minimum_value {a b : ℝ^3} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| = |a + b|) (h4 : ∠ (a, b) = 2 * real.pi / 3) :
  ∃ t, (∀ t, (|2 * a + t * b|) / |b| ≥ √3) :=
sorry

end minimum_value_l366_366994


namespace no_square_ends_in_2012_l366_366921

theorem no_square_ends_in_2012 : ¬ ∃ a : ℤ, (a * a) % 10 = 2 := by
  sorry

end no_square_ends_in_2012_l366_366921


namespace exists_digit_a_l366_366920

theorem exists_digit_a : 
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (1111 * a - 1 = (a - 1) ^ (a - 2)) :=
by {
  sorry
}

end exists_digit_a_l366_366920


namespace find_m_l366_366023

variables (AB AC AD : ℝ × ℝ)
variables (m : ℝ)

-- Definitions of vectors
def vector_AB : ℝ × ℝ := (-1, 2)
def vector_AC : ℝ × ℝ := (2, 3)
def vector_AD (m : ℝ) : ℝ × ℝ := (m, -3)

-- Conditions
def collinear (B C D : ℝ × ℝ) : Prop := ∃ k : ℝ, B = k • C ∨ C = k • D ∨ D = k • B

-- Main statement to prove
theorem find_m (h1 : vector_AB = (-1, 2))
               (h2 : vector_AC = (2, 3))
               (h3 : vector_AD m = (m, -3))
               (h4 : collinear vector_AB vector_AC (vector_AD m)) :
  m = -16 :=
sorry

end find_m_l366_366023


namespace prove_min_value_inequality_l366_366634

noncomputable def min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 9) : Prop :=
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9

theorem prove_min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 9) :
  min_value_inequality x y z h1 h2 h3 h4 :=
begin
  sorry
end

end prove_min_value_inequality_l366_366634


namespace smallest_of_vasya_numbers_l366_366346

/-
  Given the 10 pairwise sums, we need to prove that the smallest number among the 5 natural numbers 
  considered by Vasya is 60.
-/

theorem smallest_of_vasya_numbers (pairwise_sums : List ℕ) :
  pairwise_sums = [122, 124, 126, 127, 128, 129, 130, 131, 132, 135] → 
  ∃ (a b c d e : ℕ), (List.sorted (≤) [a, b, c, d, e]) ∧ 
  List.sum (List.pairwise (+) [a, b, c, d, e]) = List.sum pairwise_sums ∧ 
  a = 60 := by
  sorry

end smallest_of_vasya_numbers_l366_366346


namespace number_of_fifth_graders_l366_366688

-- Define the conditions given in the problem.
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Derived definitions with the help of the conditions.
def total_seats : ℕ := buses * seats_per_bus
def chaperones_per_grade : ℕ := teachers_per_grade + parents_per_grade
def total_chaperones : ℕ := chaperones_per_grade * grades
def total_sixth_and_seventh_graders : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := total_sixth_and_seventh_graders + total_chaperones
def seats_for_fifth_graders : ℕ := total_seats - seats_taken

-- The final statement to prove the number of fifth graders.
theorem number_of_fifth_graders : seats_for_fifth_graders = 109 :=
by
  sorry

end number_of_fifth_graders_l366_366688


namespace red_balls_count_l366_366401

theorem red_balls_count (R : ℕ) : 
  let total_balls := 60
  let white_balls := 22
  let green_balls := 18
  let yellow_balls := 2
  let purple_balls := 3
  let probability_neither_red_nor_purple := 0.7 in
  (white_balls + green_balls + yellow_balls = 42) ∧ 
  (42 / total_balls = probability_neither_red_nor_purple) →
  R = total_balls - (42 + purple_balls) →
  0 ≤ R :=
by
  sorry

end red_balls_count_l366_366401


namespace valid_arrangement_count_l366_366599

def valid_pairings (men women : list char) : Prop :=
  men.length = 3 ∧ women.length = 3 ∧
  men ≠ ['M1', 'M2', 'M3'] ∧ women ≠ ['W1', 'W2', 'W3'] ∧
  (men.head? ≠ some 'M1' ∨ women.head? ≠ some 'W1') ∧
  (men.tail?.head? ≠ some 'M2' ∨ women.tail?.head? ≠ some 'W2') ∧
  (men.tail?.tail?.head? ≠ some 'M3' ∨ women.tail?.tail?.head? ≠ some 'W3')

theorem valid_arrangement_count : 
  ∃ M W : list char, valid_pairings M W ∧ (∃! p : (list char × list char), p = (M, W)) ∧ M.permutations.length = 6 :=
sorry

end valid_arrangement_count_l366_366599


namespace final_solution_l366_366338

def DE_value_proof (AB AC BC : ℕ) (D_on_AB E_on_AC : Prop)
  (DE_parallel_BC : ∀ D E : (AB ∨ AC), ((∃ I, I is_incenter ∧ I ∈ (DE ∧ DE ∥ BC))) : Prop :=
  AB = 21 ∧ AC = 22 ∧ BC = 20 ∧ D_on_AB ∧ E_on_AC ∧ DE_parallel_BC
  → (∃ (m n : ℕ), m + n = 923 ∧ (fractions.of_ints m n = 860/63))

theorem final_solution (AB AC BC : ℕ) (D_on_AB E_on_AC : Prop)
  (DE_parallel_BC : ∀ D E : (AB ∨ AC), ((∃ I, I is_incenter ∧ I ∈ (DE ∧ DE ∥ BC))) : Prop :
  AB = 21 ∧ AC = 22 ∧ BC = 20 ∧ D_on_AB ∧ E_on_AC ∧ DE_parallel_BC →
  ∃ (m n : ℕ), m + n = 923 ∧ (m.gcd n = 1 ∧ ((m:rat) / n = 860/63)) := sorry

end final_solution_l366_366338


namespace num_pos_divisors_36_l366_366100

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366100


namespace min_value_problem_l366_366632

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : x + y + z = 9) : ℝ :=
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z)

theorem min_value_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : x + y + z = 9) :
  ∃ m : ℝ, m = 9 ∧ minValue x y z h1 h2 h3 h = m :=
begin
  use 9,
  sorry
end

end min_value_problem_l366_366632


namespace extreme_value_of_f_compare_f_and_f_prime_l366_366993

variable {a : ℝ} (x : ℝ) (f : ℝ → ℝ)

theorem extreme_value_of_f (h1 : ∀ x > 0, ∀ a > 0, a ≠ 1 → f(x) + 2 * f(1 / x) = log a x + x / log a + 2 / (x * log a)) 
  (a_pos : a > 0) (a_ne_one: a ≠ 1) : 
  (∀ x, x = 1 → a > 1 → f(1) = 1 / log a) ∧ 
  (∀ x, x = 1 → 0 < a → a < 1 → f(1) = 1 / log a) := sorry

theorem compare_f_and_f_prime (h1 : ∀ x > 0, ∀ a > 0, a ≠ 1 → f(x) + 2 * f(1 / x) = log a x + x / log a + 2 / (x * log a))
  (df : ∀ x, has_deriv_at f ((-1 / (x * log a)) + (1 / (log a))) x)
  (a_pos : a > 0) (a_ne_one: a ≠ 1) : 
  (a > 1 → f(x) > ((-1 / (x * log a)) + (1 / (log a)))) ∧ 
  (0 < a ∧ a < 1 → f(x) < ((-1 / (x * log a)) + (1 / (log a)))) := sorry

end extreme_value_of_f_compare_f_and_f_prime_l366_366993


namespace range_of_k_l366_366623

-- Definitions of M and N
def M := { x : ℝ | -1 ≤ x ∧ x < 2 }
def N (k : ℝ) := { x : ℝ | x ≤ k }

-- Proof problem statement
theorem range_of_k (k : ℝ) : (M ∩ N k ≠ ∅) ↔ k ∈ set.Ici (-1) :=
by
  sorry

end range_of_k_l366_366623


namespace number_of_divisors_of_36_l366_366111

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366111


namespace num_pos_divisors_36_l366_366124

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366124


namespace number_of_divisors_36_l366_366089

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366089


namespace minimum_number_of_tiles_l366_366867

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end minimum_number_of_tiles_l366_366867


namespace cost_per_pizza_l366_366661

theorem cost_per_pizza (total_amount : ℝ) (num_pizzas : ℕ) (H : total_amount = 24) (H1 : num_pizzas = 3) : 
  (total_amount / num_pizzas) = 8 := 
by 
  sorry

end cost_per_pizza_l366_366661


namespace rectangle_area_l366_366210

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366210


namespace lcm_18_24_l366_366750

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366750


namespace num_pos_divisors_36_l366_366130

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366130


namespace area_of_rectangle_l366_366198

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366198


namespace num_pos_divisors_36_l366_366108

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366108


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366357

theorem remainder_3001_3002_3003_3004_3005_mod_17 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366357


namespace mean_difference_is_882_l366_366693

variable (S : ℤ) (N : ℤ) (S_N_correct : N = 1000)

def actual_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 98000) / N

def incorrect_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 980000) / N

theorem mean_difference_is_882 
  (S : ℤ) 
  (N : ℤ) 
  (S_N_correct : N = 1000) 
  (S_in_range : 8200 ≤ S) 
  (S_actual : S + 98000 ≤ 980000) :
  incorrect_mean S N - actual_mean S N = 882 := 
by
  /- Proof steps would go here -/
  sorry

end mean_difference_is_882_l366_366693


namespace path_count_bound_l366_366510

theorem path_count_bound (m n : Nat) :
  let f (m n : Nat) := number_of_paths (rect_grid m n)
  (f m n ≤ 2^(m * n)) :=
sorry

end path_count_bound_l366_366510


namespace lcm_18_24_l366_366759
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366759


namespace find_angle_opposite_c_l366_366168

theorem find_angle_opposite_c
  (a b c : ℝ)
  (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∠ (a + b - c) = 0 :=
begin
  sorry
end

end find_angle_opposite_c_l366_366168


namespace interval_of_increase_l366_366324

-- Define the function and conditions
noncomputable def t (x : ℝ) := real.sqrt (-x^2 + x + 2)
noncomputable def y (x : ℝ) := (1/2) ^ (t x)

theorem interval_of_increase (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 2) :
  (1/2) ≤ x ∧ x ≤ 2 :=
by
  sorry

end interval_of_increase_l366_366324


namespace a_2_value_a_general_formula_l366_366520

open Nat

noncomputable def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 3

def a (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Normally sequences are defined from n = 1
  | 1     => -1
  | (n+1) => 4 * (n + 1) - 2

theorem a_2_value : a 2 = 6 := by
  sorry

theorem a_general_formula : ∀ n : ℕ, a n = 
  if n = 1 then -1 else (if n ≥ 2 then 4 * n - 2 else 0) := by
  sorry

end a_2_value_a_general_formula_l366_366520


namespace lcm_18_24_l366_366753

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366753


namespace power_multiplication_l366_366899

theorem power_multiplication :
  (- (4 / 5 : ℚ)) ^ 2022 * (5 / 4 : ℚ) ^ 2023 = 5 / 4 := 
by {
  sorry
}

end power_multiplication_l366_366899


namespace proof_angle_equality_maximize_ABM_l366_366428

variable (ABC : Triangle) (A B C : Point) (circumcircle : Circle) (tangentA tangentB tangentC : Tangent)
variable (B' E N A' D M : Point)
variable (angle : ABC.angle ABM BAN : Angle)
variable (AC BC : ℝ)

-- Conditions
axiom acute_angled_triangle (t : Triangle) : ∀ {A B C}, acute_angle t A B C
axiom tangents_meet (circ : Circle) (P Q : Point) (tangentP tangentQ : Tangent) : meet (tangent_point circ P) (tangent_point circ Q) = tangent_intersect P Q
axiom midpoint (P Q R : Point) : Midpoint R P Q
axiom tangent_at_intersect (circ : Circle) (P Q : Point) (tangentP : Tangent) : intersect (circumcircle_intersect circ P) (tangentP) = P
axiom tangent_at_point (circ : Circle) (P : Point) (tangentP : Tangent) : tangent_at P circumcircle = P
axiom length_AB_is_one : length AB = 1

-- Proof Goals
theorem proof_angle_equality :
  ∠ABM = ∠BAN := sorry

theorem maximize_ABM :
  ∀ (b : ℝ) (c : ℝ), AC = √2 ∧ BC = √2 := sorry

end proof_angle_equality_maximize_ABM_l366_366428


namespace find_interest_rate_l366_366407

-- Definitions from the conditions
def principal : ℕ := 1050
def time_period : ℕ := 6
def interest : ℕ := 378  -- Interest calculated as Rs. 1050 - Rs. 672

-- Correct Answer
def interest_rate : ℕ := 6

-- Lean 4 statement of the proof problem
theorem find_interest_rate (P : ℕ) (t : ℕ) (I : ℕ) 
    (hP : P = principal) (ht : t = time_period) (hI : I = interest) : 
    (I * 100) / (P * t) = interest_rate :=
by {
    sorry
}

end find_interest_rate_l366_366407


namespace num_pos_divisors_36_l366_366136

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366136


namespace jim_miles_remaining_l366_366387

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (total_miles_eq : total_miles = 1200) (miles_driven_eq : miles_driven = 384) :
  total_miles - miles_driven = 816 :=
by
  sorry

end jim_miles_remaining_l366_366387


namespace lcm_18_24_eq_72_l366_366768

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366768


namespace solve_system_l366_366293

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l366_366293


namespace find_n_l366_366568

theorem find_n (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  unfold pow at h
  sorry

end find_n_l366_366568


namespace decompose_exp2x_as_fourier_series_using_Hermite_l366_366465

noncomputable def Hermite_polynomial (n: ℕ) (x: ℝ) : ℝ := 
  (-1) ^ n * Real.exp (x^2) * (derivative^[n] (λ t, Real.exp (-t^2))) x

-- Define the orthogonality and inner product
def inner_product (u v: ℝ → ℝ) : ℝ :=
  ∫ x in -∞..∞, u x * v x * Real.exp (-x^2)

-- Statement to prove
theorem decompose_exp2x_as_fourier_series_using_Hermite :
  ∀ x : ℝ, @Function.eval ℝ ℝ _ (λ t, Real.exp (2 * t)) x = Real.exp (1:ℝ) * 
    (λ s, ∑' n, @HasSmul.smul ℝ _ s (_root_.Primcodex $n!)) ℝ ×
    λ i, Hermite_polynomial i x :=
sorry

end decompose_exp2x_as_fourier_series_using_Hermite_l366_366465


namespace algebraic_expression_value_l366_366579

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2 * x - 1 = 0) : x^3 - x^2 - 3 * x + 2 = 3 := 
by
  sorry

end algebraic_expression_value_l366_366579


namespace count_numbers_in_list_l366_366046

theorem count_numbers_in_list : 
  ∃ (n : ℕ), (list.range n).map (λ k, 165 - 5 * k) = [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45] ∧ n = 25 :=
by {
  sorry
}

end count_numbers_in_list_l366_366046


namespace number_of_solutions_eq_4_l366_366040

theorem number_of_solutions_eq_4 : 
  ∃ (S : Set ℝ) (hS : ∀ x ∈ S, (x^2 - 7)^2 = 25), S.finite ∧ S.card = 4 :=
begin
  sorry
end

end number_of_solutions_eq_4_l366_366040


namespace amount_b_l366_366833

-- Definitions of the conditions
variables (a b : ℚ) 

def condition1 : Prop := a + b = 1210
def condition2 : Prop := (2 / 3) * a = (1 / 2) * b

-- The theorem to prove
theorem amount_b (h₁ : condition1 a b) (h₂ : condition2 a b) : b = 691.43 :=
sorry

end amount_b_l366_366833


namespace cyclic_quadrilateral_if_parallel_reflections_l366_366843

-- Definitions of quadrilateral, reflections, and cyclic condition
noncomputable def is_parallel (l1 l2 : Line) := ∀ (P Q : Point), P ∈ l1 → Q ∈ l2 → ∃ R : Line, P, Q ∈ R
noncomputable def is_reflection (A B : Point) (l : Line) : Prop := ∃ m : Point, midpoint m A B ∧ m ∈ l
noncomputable def is_cyclic (Q : Quadrilateral) : Prop := ∃ (O : Point), ∀ P ∈ Q.vertices, dist O P = const dist O Q.A

-- Definition of reflection in successive lines
noncomputable def successive_reflections (Q : Quadrilateral) : Quadrilateral :=
  let Q' := {Q with A := reflection Q.A Q.BC, D := reflection Q.D Q.BC}
  let Q'' := {Q' with A := reflection Q'.A Q'.CD, B := reflection Q'.B Q'.CD}
  let Q''' := {Q'' with A := reflection Q''.A Q''.DA'}
  Q'''

-- Main theorem statement
theorem cyclic_quadrilateral_if_parallel_reflections
  (Q : Quadrilateral)
  (Q' := successive_reflections Q)
  (h_parallel : is_parallel Q.AA'' Q.BB'') :
  is_cyclic Q := 
sorry

end cyclic_quadrilateral_if_parallel_reflections_l366_366843


namespace area_of_shaded_design_in_7x7_grid_is_1_point_5_l366_366739

theorem area_of_shaded_design_in_7x7_grid_is_1_point_5 :
  let shaded_design_area := 1.5 in
  shaded_design_area = 1.5 := 
by
  sorry

end area_of_shaded_design_in_7x7_grid_is_1_point_5_l366_366739


namespace minimum_value_l366_366519

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a (n + 1) + a n = (n + 1) * Real.cos (n * Real.pi / 2)

noncomputable def sum_sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0     := a 0
| (n+1) := sum_sequence n + a (n + 1)

noncomputable def problem (a : ℕ → ℝ) (m : ℝ) : Prop :=
  sequence a ∧
  sum_sequence a 2017 + m = 1010 ∧
  a 1 * m > 0

theorem minimum_value (a : ℕ → ℝ) (m : ℝ) :
  problem a m → min ((1 / a 1) + (1 / m)) 2 :=
sorry

end minimum_value_l366_366519


namespace construct_parallel_line_l366_366034

-- Definitions for given conditions
variables {Point : Type} [incidence_geometry Point]

-- Given conditions: segment AB, midpoint F of AB, and point P
variables (A B F P : Point)
variables [hMid : midpoint F A B]

-- Required to prove that we can construct a line PC parallel to AB
theorem construct_parallel_line (A B F P : Point)
  [hMid : midpoint F A B] :
  ∃ C : Point, ∃ line_PC : line Point, on_line PC P ∧ ∀ (l : line Point), on_line l A ∧ on_line l B → parallel line_PC l :=
begin
  sorry
end

end construct_parallel_line_l366_366034


namespace problem_conditions_l366_366320

theorem problem_conditions (m : ℝ) (hf_pow : m^2 - m - 1 = 1) (hf_inc : m > 0) : m = 2 :=
sorry

end problem_conditions_l366_366320


namespace number_of_divisors_36_l366_366054

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366054


namespace total_area_enclosed_l366_366322

noncomputable def area_enclosed_by_arcs (arc_length : ℝ) (octagon_side_length : ℝ) : ℝ := 
  -- Sorry here means we'll skip the actual calculation/proof for now
  sorry

theorem total_area_enclosed (arc_length : ℝ) (octagon_side_length : ℝ) :
  arc_length = 3 * π / 4 → octagon_side_length = 3 →
  area_enclosed_by_arcs arc_length octagon_side_length = 
  54 + 18 * (real.sqrt 2) + 81 * π / 64 - 54 * π / 64 - 18 * π * (real.sqrt 2) / 64 := 
begin
  intros h_arc_length h_octagon_side_length,
  rw [
    h_arc_length,
    h_octagon_side_length
  ],
  -- The actual proof steps would go here
  sorry
end

end total_area_enclosed_l366_366322


namespace percent_saved_is_75_l366_366435

variable (P : ℝ) -- The original price of one ticket
variable (sale_price : ℝ) -- The price paid for 12 tickets during the sale
variable (original_price_12 : ℝ) -- The original price of 12 tickets
variable (amount_saved : ℝ) -- The amount saved by purchasing 12 tickets at the sale price

-- Assume conditions
def condition_1 : sale_price = 3 * P := sorry -- 12 tickets can be purchased for the price of 3 tickets
def condition_2 : original_price_12 = 12 * P := sorry -- The original price of 12 tickets without the sale
def condition_3 : amount_saved = original_price_12 - sale_price := sorry -- Amount saved is the difference between original price and sale price

-- The correct answer
def correct_answer : amount_saved / original_price_12 * 100 = 75 := sorry -- The amount saved is 75% of the original price

-- Main theorem
theorem percent_saved_is_75 : 
  condition_1 → condition_2 → condition_3 → correct_answer := sorry

end percent_saved_is_75_l366_366435


namespace num_multiples_of_15_l366_366559

theorem num_multiples_of_15 (a b m : ℕ) (h1 : a = 25) (h2 : b = 200) (h3 : m = 15) : 
  ∃ n, n = (b - a) / m + 1 ∧ ∃ k, k = (n - 1) * m + a ∧ k mod m = 0 ∧ a < k ∧ k < b := by
  sorry

end num_multiples_of_15_l366_366559


namespace division_remainder_l366_366369

def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30
def d (x : ℝ) : ℝ := 4 * x - 8

theorem division_remainder : (∃ q r, p(2) = d(2) * q + r ∧ d(2) ≠ 0 ∧ r = 10) :=
by
  sorry

end division_remainder_l366_366369


namespace point_on_coordinate_axes_l366_366150

theorem point_on_coordinate_axes (x y : ℝ) (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by sorry

end point_on_coordinate_axes_l366_366150


namespace probability_no_success_l366_366454

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l366_366454


namespace necessary_but_not_sufficient_condition_l366_366566

noncomputable def exponential_inequality (x : ℝ) : Prop := (1 / 3)^x < 1

noncomputable def reciprocal_inequality (x : ℝ) : Prop := 1 / x > 1

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  exponential_inequality x → (reciprocal_inequality x → true := sorry

end necessary_but_not_sufficient_condition_l366_366566


namespace distinct_values_count_l366_366448

def is_odd_pos_less_10 (n : Nat) : Prop := n ∈ {1, 3, 5, 7, 9}

theorem distinct_values_count : 
  ∃ S : Finset Nat, 
    (∀ a b : Nat, 
      is_odd_pos_less_10 a →  
      is_odd_pos_less_10 b → 
      (ab + a + b) ∈ S) ∧ 
    S.card = 10 := 
sorry

end distinct_values_count_l366_366448


namespace lcm_18_24_l366_366786

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366786


namespace Lagrange_interpol_equiv_x_squared_l366_366266

theorem Lagrange_interpol_equiv_x_squared (a b c x : ℝ)
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    c^2 * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
    b^2 * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
    a^2 * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x^2 := 
    sorry

end Lagrange_interpol_equiv_x_squared_l366_366266


namespace find_f_of_f_of_2_l366_366516

def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x + 1 / x else x ^ 2 - 3 * x + 1

theorem find_f_of_f_of_2 : f (f 2) = -3 := 
by
  sorry

end find_f_of_f_of_2_l366_366516


namespace a_sequence_general_formula_sum_b_sequence_l366_366020

def a_sequence (n : ℕ) : ℝ := 1 / 3^n
def S (n : ℕ) : ℝ := 1/2 - 1/2 * a_sequence n
def b_sequence (n : ℕ) : ℝ := a_sequence n * Real.logBase 3 (a_sequence n)
def S_b (n : ℕ) : ℝ := -3 / 4 + (2 * n + 3) / (4 * 3^n)

theorem a_sequence_general_formula (n : ℕ) : a_sequence n = 1 / 3^n := by
  sorry

theorem sum_b_sequence (n : ℕ) : (∑ k in Finset.range n, b_sequence k) = S_b n := by
  sorry

end a_sequence_general_formula_sum_b_sequence_l366_366020


namespace largest_abs_diff_of_solutions_l366_366279

theorem largest_abs_diff_of_solutions (x y : ℝ) 
  (h1 : x^2 + y^2 = 2023) 
  (h2 : (x - 2) * (y - 2) = 3) : 
  ∃ s : ℝ, s = 13 * Real.sqrt 13 ∧ 
    ∀ t : ℝ, t = |x - y| → t ≤ s :=
begin
  sorry
end

end largest_abs_diff_of_solutions_l366_366279


namespace not_all_polynomials_sum_of_cubes_l366_366444

theorem not_all_polynomials_sum_of_cubes :
  ¬ ∀ P : Polynomial ℤ, ∃ Q : Polynomial ℤ, P = Q^3 + Q^3 + Q^3 :=
by
  sorry

end not_all_polynomials_sum_of_cubes_l366_366444


namespace triangle_problem_l366_366158

variables {A B C a b c m : ℝ}

/-- Given conditions and goals to prove:
1. Measure of angle C is π/3
2. Minimum value of m is 2
-/
theorem triangle_problem
  (h_triangle : a^2 + b^2 - c^2 = ab)
  (C_acute : 0 < C ∧ C < π)
  (tan_eq : m / (tan C) = (1 / (tan A) + 1 / (tan B)))
  (h_line : a * (sin A - sin B) + b * (sin B) = c * (sin C))
  (h_acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2) : 
  C = π / 3 ∧ m = 2 :=
by
  sorry

end triangle_problem_l366_366158


namespace rectangle_area_l366_366197

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366197


namespace ratio_of_y_to_x_l366_366582

theorem ratio_of_y_to_x (c x y : ℝ) (hx : x = 0.90 * c) (hy : y = 1.20 * c) :
  y / x = 4 / 3 := 
sorry

end ratio_of_y_to_x_l366_366582


namespace gardener_works_days_l366_366437

theorem gardener_works_days :
  let rose_bushes := 20
  let cost_per_rose_bush := 150
  let gardener_hourly_wage := 30
  let gardener_hours_per_day := 5
  let soil_volume := 100
  let cost_per_soil := 5
  let total_project_cost := 4100
  let total_gardening_days := 4
  (rose_bushes * cost_per_rose_bush + soil_volume * cost_per_soil + total_gardening_days * gardener_hours_per_day * gardener_hourly_wage = total_project_cost) →
  total_gardening_days = 4 :=
by
  intros
  sorry

end gardener_works_days_l366_366437


namespace probability_four_collinear_dots_l366_366602

theorem probability_four_collinear_dots :
  let total_dots := 25
  let total_ways_to_choose_dots := Nat.choose total_dots 4
  let collinear_ways := 16
  (collinear_ways.toRat / total_ways_to_choose_dots.toRat) = (16 : ℚ) / 12650 :=
by
  intros
  let total_dots := 25
  let total_ways_to_choose_dots := Nat.choose total_dots 4
  let collinear_ways := 16
  have collinear_probability : collinear_ways.toRat / total_ways_to_choose_dots.toRat = (16 : ℚ) / 12650 := sorry,
  exact collinear_probability

end probability_four_collinear_dots_l366_366602


namespace distinct_weights_count_l366_366723

theorem distinct_weights_count : 
  ∃ (weights : Finset ℕ) (ways : ℕ → ℕ),
    weights = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
    ways 1 = 1 ∧
    ways 2 = 1 ∧
    ways 3 = 2 ∧
    ways 4 = 2 ∧
    ways 5 = 2 ∧
    ways 6 = 2 ∧
    ways 7 = 2 ∧
    ways 8 = 1 ∧
    ways 9 = 1 ∧
    ways 10 = 1 :=
by {
  let weights := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.toFinset,
  let ways := fun n => match n with
                        | 1 => 1
                        | 2 => 1
                        | 3 => 2
                        | 4 => 2
                        | 5 => 2
                        | 6 => 2
                        | 7 => 2
                        | 8 => 1
                        | 9 => 1
                        | 10 => 1
                        | _ => 0,
  use [weights, ways],
  split, refl,
  repeat { split; refl }
}

end distinct_weights_count_l366_366723


namespace find_num_pennies_l366_366141

def total_value (nickels : ℕ) (dimes : ℕ) (pennies : ℕ) : ℕ :=
  5 * nickels + 10 * dimes + pennies

def num_pennies (nickels_value: ℕ) (dimes_value: ℕ) (total: ℕ): ℕ :=
  total - (nickels_value + dimes_value)

theorem find_num_pennies : 
  ∀ (total : ℕ) (num_nickels : ℕ) (num_dimes: ℕ),
  total = 59 → num_nickels = 4 → num_dimes = 3 → num_pennies (5 * num_nickels) (10 * num_dimes) total = 9 :=
by
  intros
  sorry

end find_num_pennies_l366_366141


namespace inequality_solution_set_l366_366157

theorem inequality_solution_set (a : ℝ) : (∀ x : ℝ, x > 5 ∧ x > a ↔ x > 5) → a ≤ 5 :=
by
  sorry

end inequality_solution_set_l366_366157


namespace proof_problem_l366_366712

-- Necessary types and noncomputable definitions
noncomputable def a_seq : ℕ → ℕ := sorry
noncomputable def b_seq : ℕ → ℕ := sorry

-- The conditions in the problem are used as assumptions
axiom partition : ∀ (n : ℕ), n > 0 → a_seq n < a_seq (n + 1)
axiom b_def : ∀ (n : ℕ), n > 0 → b_seq n = a_seq n + n

-- The mathematical equivalent proof problem stated
theorem proof_problem (n : ℕ) (hn : n > 0) : a_seq n + b_seq n = a_seq (b_seq n) :=
sorry

end proof_problem_l366_366712


namespace num_pos_divisors_36_l366_366135

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366135


namespace exponent_equation_l366_366574

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l366_366574


namespace lcm_18_24_l366_366784

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366784


namespace difference_of_scores_l366_366729

variable {x y : ℝ}

theorem difference_of_scores (h : x / y = 4) : x - y = 3 * y := by
  sorry

end difference_of_scores_l366_366729


namespace sum_of_bases_l366_366173

theorem sum_of_bases (R_1 R_2 : ℕ) 
  (hF1 : (4 * R_1 + 8) / (R_1 ^ 2 - 1) = (3 * R_2 + 6) / (R_2 ^ 2 - 1))
  (hF2 : (8 * R_1 + 4) / (R_1 ^ 2 - 1) = (6 * R_2 + 3) / (R_2 ^ 2 - 1)) : 
  R_1 + R_2 = 21 :=
sorry

end sum_of_bases_l366_366173


namespace inscribed_circle_diameter_l366_366910

-- Define the three sides of the triangle DEF
def DE : ℝ := 13
def DF : ℝ := 14
def EF : ℝ := 15

-- Define the semiperimeter s
def s : ℝ := (DE + DF + EF) / 2

-- Define the area K using Heron's formula
def K : ℝ := Real.sqrt(s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius r of the inscribed circle
def r : ℝ := K / s

-- Define the diameter d of the inscribed circle
def d : ℝ := 2 * r

theorem inscribed_circle_diameter : d = 8 := by
  have DE_nonneg : 0 ≤ DE := by linarith
  have DF_nonneg : 0 ≤ DF := by linarith
  have EF_nonneg : 0 ≤ EF := by linarith
  have s_nonneg : 0 ≤ s := by linarith
  
  rw [DE, DF, EF] at *
  -- Semiperimeter calculation
  have s_calculation : s = 21 := by norm_num
  rw s_calculation at *

  -- Heron's formula calculation
  have area_calculation : K = Real.sqrt (21 * (21 - 13) * (21 - 14) * (21 - 15)) := by simp [K]
  have area_value : K = 84 := by norm_num
  rw area_value at *

  -- Radius calculation
  have radius_calculation : r = 4 := by simp [r, s_calculation, area_value]
  rw radius_calculation at *

  -- Diameter calculation
  have diameter_calculation : d = 8 := by simp [d]
  exact diameter_calculation

end inscribed_circle_diameter_l366_366910


namespace num_pos_divisors_36_l366_366067

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366067


namespace painting_perimeter_l366_366260

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end painting_perimeter_l366_366260


namespace axis_of_symmetry_of_shifted_sine_l366_366680

def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)
def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

theorem axis_of_symmetry_of_shifted_sine : 
  ∃ c : ℝ, g (-Real.pi / 12) = g (-Real.pi / 12 + c) :=
sorry

end axis_of_symmetry_of_shifted_sine_l366_366680


namespace lcm_18_24_l366_366790

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366790


namespace volume_of_63_ounces_l366_366390

variable {V W : ℝ}
variable (k : ℝ)

def directly_proportional (V W : ℝ) (k : ℝ) : Prop :=
  V = k * W

theorem volume_of_63_ounces (h1 : directly_proportional 48 112 k)
                            (h2 : directly_proportional V 63 k) :
  V = 27 := by
  sorry

end volume_of_63_ounces_l366_366390


namespace jason_pokemon_cards_l366_366228

theorem jason_pokemon_cards :
  ∀ (initial_cards trade_benny_lost trade_benny_gain trade_sean_lost trade_sean_gain give_to_brother : ℕ),
  initial_cards = 5 →
  trade_benny_lost = 2 →
  trade_benny_gain = 3 →
  trade_sean_lost = 3 →
  trade_sean_gain = 4 →
  give_to_brother = 2 →
  initial_cards - trade_benny_lost + trade_benny_gain - trade_sean_lost + trade_sean_gain - give_to_brother = 5 :=
by
  intros
  sorry

end jason_pokemon_cards_l366_366228


namespace find_a_l366_366645

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem find_a (a : ℝ) (h : {x | x^2 - 3 * x + 2 = 0} ∩ {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} = {2}) :
  a = -3 ∨ a = -1 :=
by
  sorry

end find_a_l366_366645


namespace max_unique_planes_l366_366422

theorem max_unique_planes (n : ℕ) (h_n : n = 15)
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → p1, p2, p3 are not collinear)
  (h_not_coplanar : ∃ (p : Fin n → Point), ¬ ∀ (i j k : Fin n), Points_coplanar p i j k) : 
  ∃ max_planes : ℕ, max_planes = 455 :=
by
  sorry

end max_unique_planes_l366_366422


namespace possible_values_of_N_l366_366667

noncomputable def totalSurveyParticipants :=
  ∃ N : ℕ, 
      (753 : ℕ) ≤ (9 * N / 14 : ℕ ∧ 
      753 ≤ (7 * N / 12 : ℕ) ∧ 
      (753 / 84 * N / 84 : ℕ) ∧ 
      ((753: ℕ) ≤ N) ((9 * (N / 14) * (N / 12)): ℕ )

theorem possible_values_of_N (k : ℕ) : 753 ≤ N := 753 := 9 * N / 14 := 7 * 753 := N := 9 * N / 7 := 12 :=
  sorry

end possible_values_of_N_l366_366667


namespace solution_set_of_inequality_l366_366531

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  { x : ℝ | |f (x - 2)| > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end solution_set_of_inequality_l366_366531


namespace four_consecutive_integers_plus_one_is_square_l366_366662

theorem four_consecutive_integers_plus_one_is_square (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n ^ 2 + n - 1) ^ 2 := 
by 
  sorry

end four_consecutive_integers_plus_one_is_square_l366_366662


namespace find_a_l366_366567

theorem find_a (a : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 + a * i) * i = -3 + i) : a = 3 :=
by
  sorry

end find_a_l366_366567


namespace problem_1_problem_2_l366_366648

-- First Problem
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∃ x : ℝ, f x - 2 * |x - 7| ≤ 0) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → a ≥ -12 :=
by
  intros
  sorry

-- Second Problem
theorem problem_2 (f : ℝ → ℝ) (a m : ℝ) (h1 : a = 1) 
  (h2 : ∀ x : ℝ, f x + |x + 7| ≥ m) :
  (∀ x : ℝ, f x = 2 * |x - 1| - a) → m ≤ 7 :=
by
  intros
  sorry

end problem_1_problem_2_l366_366648


namespace count_numbers_in_list_l366_366044

theorem count_numbers_in_list : 
  ∃ (n : ℕ), (list.range n).map (λ k, 165 - 5 * k) = [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45] ∧ n = 25 :=
by {
  sorry
}

end count_numbers_in_list_l366_366044


namespace lcm_18_24_l366_366794

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366794


namespace find_purchase_price_minimum_number_of_speed_skating_shoes_l366_366853

/-
A certain school in Zhangjiakou City is preparing to purchase speed skating shoes and figure skating shoes to promote ice and snow activities on campus.

If they buy 30 pairs of speed skating shoes and 20 pairs of figure skating shoes, the total cost is $8500.
If they buy 40 pairs of speed skating shoes and 10 pairs of figure skating shoes, the total cost is $8000.
The school purchases a total of 50 pairs of both types of ice skates, and the total cost does not exceed $8900.
-/

def price_system (x y : ℝ) : Prop :=
  30 * x + 20 * y = 8500 ∧ 40 * x + 10 * y = 8000

def minimum_speed_skating_shoes (x y m : ℕ) : Prop :=
  150 * m + 200 * (50 - m) ≤ 8900

theorem find_purchase_price :
  ∃ x y : ℝ, price_system x y ∧ x = 150 ∧ y = 200 :=
by
  /- Proof goes here -/
  sorry

theorem minimum_number_of_speed_skating_shoes :
  ∃ m, minimum_speed_skating_shoes 150 200 m ∧ m = 22 :=
by
  /- Proof goes here -/
  sorry

end find_purchase_price_minimum_number_of_speed_skating_shoes_l366_366853


namespace red_section_no_damage_probability_l366_366456

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l366_366456


namespace area_of_rectangle_ABCD_l366_366183

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366183


namespace locus_of_vertices_eq_l366_366512

open Real EuclideanGeometry

noncomputable def locus_of_vertices
  (O : Point)
  (r : ℝ)
  (K : Circle O r) :
  set Point :=
{A | ∃ B C D, is_parallelogram A B C D ∧ dist O B < r ∧ dist O D < r ∧ dist O A < r * sqrt 2}

theorem locus_of_vertices_eq
  (O : Point)
  (r : ℝ)
  {K : Circle O r} :
  locus_of_vertices O r K = {P | dist O P < r * sqrt 2} :=
by sorry

end locus_of_vertices_eq_l366_366512


namespace limit_of_sequence_l366_366345

theorem limit_of_sequence (x_n : ℕ → ℝ) (a : ℝ) [Lim n in at_top, x_n n = a] :
  (∀ n : ℕ, x_n n = (2*n - 1) / (3*n + 5)) →
  a = 2/3 :=
by 
  sorry

end limit_of_sequence_l366_366345


namespace max_fraction_l366_366709

theorem max_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1 / 3) : (a, b) = (25, 76) :=
sorry

end max_fraction_l366_366709


namespace parallelogram_area_relation_l366_366241

-- Define a parallelogram in terms of points and the collinear relationships.
structure Parallelogram (A B C D : Type) :=
(side_AB : A ≠ B)
(side_BC : B ≠ C)
(side_CD : C ≠ D)
(side_DA : D ≠ A)
(parallel_AB_CD : Parallel (A, B) (C, D))
(parallel_BC_AD : Parallel (B, C) (D, A))

-- Define the condition for point P on side AB
structure PointOnLine (P A B : Type) :=
(point_on_line : OnLine (P, A, B))

-- Define the condition for the intersection of line through P parallel to BC with diagonal AC
structure LineIntersection (P Q A B C : Type) :=
(intersects_AC : Intersects (P, Q) (A, C))
(parallel_PQ_BC : Parallel (P, Q) (B, C))

-- Define the area of triangles
axiom area_triangle (X Y Z : Type) : ℝ

-- Define the statement to prove
theorem parallelogram_area_relation {A B C D P Q : Type}
  (parallelogram : Parallelogram A B C D)
  (point_P_on_AB : PointOnLine P A B)
  (line_intersection_PQ_AC : LineIntersection P Q A B C) :
  area_triangle D A Q ^ 2 = area_triangle P A Q * area_triangle B C D :=
sorry

end parallelogram_area_relation_l366_366241


namespace lcm_18_24_l366_366762
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366762


namespace ratio_of_distances_l366_366414

theorem ratio_of_distances (d_5 d_4 : ℝ) (h1 : d_5 + d_4 ≤ 26.67) (h2 : d_5 / 5 + d_4 / 4 = 6) : 
  d_5 / (d_5 + d_4) = 1 / 2 :=
sorry

end ratio_of_distances_l366_366414


namespace unused_signs_count_l366_366718

theorem unused_signs_count :
  ∃ x : ℕ, (424^2 - 422^2 = 1688) ∧ (x = 424 - 422) :=
by {
  use 2,
  split,
  {
    calc
      424^2 - 422^2
          = (424 + 422) * (424 - 422) : by rw [pow_two, pow_two, sub_mul, add_mul, add_mul, mul_add, mul_add, sub_add, add_sub_assoc, sub_self, zero_add]
      ... = 846  * 2                    : by norm_num
      ... = 1688                        : by norm_num,
  },
  {
    norm_num,
  }
}

end unused_signs_count_l366_366718


namespace total_time_six_laps_l366_366684

-- Defining the constants and conditions
def total_distance : Nat := 500
def speed_part1 : Nat := 3
def distance_part1 : Nat := 150
def speed_part2 : Nat := 6
def distance_part2 : Nat := total_distance - distance_part1
def laps : Nat := 6

-- Calculating the times based on conditions
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2
def time_per_lap := time_part1 + time_part2
def total_time := laps * time_per_lap

-- The goal is to prove the total time is 10 minutes and 48 seconds (648 seconds)
theorem total_time_six_laps : total_time = 648 :=
-- proof would go here
sorry

end total_time_six_laps_l366_366684


namespace determinant_eq_sum_of_products_l366_366927

theorem determinant_eq_sum_of_products (x y z : ℝ) :
  Matrix.det (Matrix.of ![![1, x + z, y], ![1, x + y + z, y + z], ![1, x + z, x + y + z]]) = x * y + y * z + z * x :=
by
  sorry

end determinant_eq_sum_of_products_l366_366927


namespace odd_function_l366_366826

noncomputable section
def fA (x : ℝ) := -abs x
def fB (x : ℝ) := 2^x + 2^(-x)
def fC (x : ℝ) := Real.log (1 + x) - Real.log (1 - x)
def fD (x : ℝ) := x^3 - 1

theorem odd_function (x : ℝ) : fC (-x) = -fC x := by
  sorry

end odd_function_l366_366826


namespace residue_of_neg_1237_mod_37_l366_366471

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l366_366471


namespace total_quantities_l366_366306

theorem total_quantities (n S S₃ S₂ : ℕ) (h₁ : S = 6 * n) (h₂ : S₃ = 4 * 3) (h₃ : S₂ = 33 * 2) (h₄ : S = S₃ + S₂) : n = 13 :=
by
  sorry

end total_quantities_l366_366306


namespace evaluate_using_horner_l366_366344

def polynomial (x : ℤ) : ℤ :=
  4 * x ^ 5 - 3 * x ^ 4 + 4 * x ^ 3 - 2 * x ^ 2 - 2 * x + 3

def horner_poly (x : ℤ) : ℤ :=
  ((4 * x - 3) * x + 4) * x - 2) * x - 2) * x + 3

theorem evaluate_using_horner :
  horner_poly 3 = 816 ∧ 
  ( /* 5 multiplications and 5 additions */ sorry )
:= sorry

end evaluate_using_horner_l366_366344


namespace simplify_powers_l366_366682

-- Defining the multiplicative rule for powers
def power_mul (x : ℕ) (a b : ℕ) : ℕ := x^(a+b)

-- Proving that x^5 * x^6 = x^11
theorem simplify_powers (x : ℕ) : x^5 * x^6 = x^11 :=
by
  change x^5 * x^6 = x^(5 + 6)
  sorry

end simplify_powers_l366_366682


namespace num_pos_divisors_36_l366_366061

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366061


namespace marble_choices_l366_366728

theorem marble_choices :
  (∃ red green blue : Marble, ∃ yellow : Fin 4, 
     (∀ (g : Fin 3), (image g red deep) ∧ ∀ Marble → Marble → Marble )) → 
  (∃ n : ℕ, n = 7) := by
  sorry

end marble_choices_l366_366728


namespace projectile_reaches_35m_first_at_10_over_7_l366_366313

theorem projectile_reaches_35m_first_at_10_over_7 :
  ∃ (t : ℝ), (y : ℝ) = -4.9 * t^2 + 30 * t ∧ y = 35 ∧ t = 10 / 7 :=
by
  sorry

end projectile_reaches_35m_first_at_10_over_7_l366_366313


namespace system_solution_l366_366288

noncomputable def solve_system (x y : ℝ) : Prop :=
  (x + y = 20) ∧ (Real.logBase 4 x + Real.logBase 4 y = 1 + Real.logBase 4 9) ∧
  ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18))

theorem system_solution : ∃ x y : ℝ, solve_system x y :=
  sorry

end system_solution_l366_366288


namespace lcm_18_24_eq_72_l366_366769

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366769


namespace terminal_side_in_third_quadrant_l366_366005

open Real

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
    θ ∈ Set.Ioo (π : ℝ) (3 * π / 2) := 
sorry

end terminal_side_in_third_quadrant_l366_366005


namespace sum_of_three_digit_numbers_l366_366653

theorem sum_of_three_digit_numbers (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  222 * (a + b + c) ≠ 2021 := 
sorry

end sum_of_three_digit_numbers_l366_366653


namespace odd_function_value_l366_366319

variable (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value : is_odd f → f 2016 = 2 → f (-2016) = -2 :=
by
  intros h_odd h_value
  have h_neg := h_odd 2016
  rw h_value at h_neg
  exact h_neg

end odd_function_value_l366_366319


namespace problem_solution_l366_366997

theorem problem_solution
  (a1 a2 a3: ℝ)
  (a_arith_seq : ∃ d, a1 = 1 + d ∧ a2 = a1 + d ∧ a3 = a2 + d ∧ 9 = a3 + d)
  (b1 b2 b3: ℝ)
  (b_geo_seq : ∃ r, r > 0 ∧ b1 = -9 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -1 = b3 * r) :
  (b2 / (a1 + a3) = -3 / 10) :=
by
  -- Placeholder for the proof, not required in this context
  sorry

end problem_solution_l366_366997


namespace constant_sum_l366_366717

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem constant_sum (a1 d : ℝ) (h : 3 * arithmetic_sequence a1 d 8 = k) :
  ∃ k : ℝ, sum_arithmetic_sequence a1 d 15 = k :=
sorry

end constant_sum_l366_366717


namespace pos_ints_less_than_1500_congruent_to_7_mod_13_are_115_l366_366140

noncomputable def count_congruent_integers : Nat :=
  (List.range 1500).countp (λ n, n ≡ 7 [MOD 13])

theorem pos_ints_less_than_1500_congruent_to_7_mod_13_are_115 :
  count_congruent_integers = 115 :=
sorry

end pos_ints_less_than_1500_congruent_to_7_mod_13_are_115_l366_366140


namespace ratio_of_AD_DC_l366_366593

noncomputable section

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (a b c : ℝ)
variables (AD DC : ℝ)

-- Definitions and conditions
def AB := 6
def BC := 8
def AC := 10
def BD := 6

-- Point D is on segment AC such that BD = 6
def is_on_segment_AC (D : C) : Prop := MetricSpace.dist D A + MetricSpace.dist D C = AC

-- The ratio we need to prove
def ratio_AD_DC (AD DC : ℝ) : ℝ := AD / DC

-- Proof statement
theorem ratio_of_AD_DC (AD_eq : AD = 36/5) (DC_eq : DC = 14/5) : 
    ratio_AD_DC AD DC = 18 / 7 := 
by
  sorry

end ratio_of_AD_DC_l366_366593


namespace y_coordinate_equidistant_l366_366350

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end y_coordinate_equidistant_l366_366350


namespace circle_tangent_to_parabola_axis_l366_366492

theorem circle_tangent_to_parabola_axis (t : ℝ) 
(h_center_on_parabola : ∃ t, (0, t) = (t, 1/2 * t^2))
(h_tangent_to_axis_of_symmetry : abs t = 1/2 * t^2 + 1/2) :
∃ k, k ∈ {1, -1} ∧ ∀ x y, (x - t) ^ 2 + (y - 1/2 * t ^ 2) ^ 2 = k :=
by
  sorry

end circle_tangent_to_parabola_axis_l366_366492


namespace problem_statement_l366_366630

noncomputable def f (x : ℝ) := 3 * x ^ 5 + 4 * x ^ 4 - 5 * x ^ 3 + 2 * x ^ 2 + x + 6
noncomputable def d (x : ℝ) := x ^ 3 + 2 * x ^ 2 - x - 3
noncomputable def q (x : ℝ) := 3 * x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) := 19 * x ^ 2 - 11 * x - 57

theorem problem_statement : (f 1 = q 1 * d 1 + r 1) ∧ q 1 + r 1 = -47 := by
  sorry

end problem_statement_l366_366630


namespace regular_train_pass_time_l366_366410

-- Define the lengths of the trains
def high_speed_train_length : ℕ := 400
def regular_train_length : ℕ := 600

-- Define the observation time for the passenger on the high-speed train
def observation_time : ℕ := 3

-- Define the problem to find the time x for the regular train passenger
theorem regular_train_pass_time :
  ∃ (x : ℕ), (regular_train_length / observation_time) * x = high_speed_train_length :=
by 
  sorry

end regular_train_pass_time_l366_366410


namespace find_m_l366_366903

def s (m : Nat) : ℚ :=
match m with
| 1       => 2
| _       => if m % 3 = 0
             then 2 + s (m / 3)
             else 1 / s (m - 1)

theorem find_m (m : Nat) (h : s m = 25 / 103) : m = 1468 := 
by
  sorry

end find_m_l366_366903


namespace no_solution_or_indeterminate_l366_366906

variables {n : ℕ} -- Number of sides of the polygon
variables {M : Fin n → Point} -- Vertices M_1, M_2, ..., M_n
variables {α : Fin n → ℝ} -- Angles α_1, α_2, ..., α_n
variables {k : Fin n → ℝ} -- Ratios k_1, k_2, ..., k_n

-- The combined transformation conditions
def α_total : ℝ := ∑ i, α i
def k_total : ℝ := ∏ i, k i

-- Lean 4 statement for the problem
theorem no_solution_or_indeterminate :
  (∃ k : ℤ, α_total = 360 * k) ∧ (k_total = 1) ↔ 
  (∀ A1 : Point, (∃ (P : Point), P ≠ A1) ∨ 
   (∃ A1 : Point, ∀ A : Point, A1 = A)) := sorry

end no_solution_or_indeterminate_l366_366906


namespace relationship_among_a_b_c_l366_366504

open Real

noncomputable def a : ℝ := 0.3 ^ 3
noncomputable def b : ℝ := 3 ^ 0.3
noncomputable def c : ℝ := log 3 / log 0.3 -- Using change of base formula for logarithms

theorem relationship_among_a_b_c : c < a ∧ a < b := 
by
  sorry

end relationship_among_a_b_c_l366_366504


namespace cos_double_angle_l366_366502

-- Define the angle α, and the condition that tan(α) = 3
variable (α : ℝ)
axiom tan_alpha_eq : Mathlib.Trigonometry.tan α = 3

-- State the problem: Prove that cos(2α) = -4/5
theorem cos_double_angle : Mathlib.Trigonometry.cos (2 * α) = -4 / 5 :=
by
  -- reminder: this is just a placeholder to define the statement
  sorry

end cos_double_angle_l366_366502


namespace min_value_problem_l366_366631

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : x + y + z = 9) : ℝ :=
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z)

theorem min_value_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : x + y + z = 9) :
  ∃ m : ℝ, m = 9 ∧ minValue x y z h1 h2 h3 h = m :=
begin
  use 9,
  sorry
end

end min_value_problem_l366_366631


namespace probability_computation_l366_366327

open Nat

-- Definitions of permutations and probabilities.
def valid_permutation (n : ℕ) (perm : list ℕ) : Prop :=
  (∀ k < n, ∃ k' > k, perm.indexOf k' = perm.indexOf k + 1 ∨ perm.indexOf k' = perm.indexOf k + 2)

def a (n : ℕ) : ℕ
| 4 := 1
| 5 := 1
| n := 2 * (a (n-1)) + 2 * (n-2) * (a (n-2))

def p (n : ℕ) : ℚ :=
  (a n : ℚ) / (factorial (n-1))

theorem probability_computation :
  let p_10 := p 10 in
  p_10 = 13 / 90 ∧ 100 * 13 + 90 = 1390 := 
by 
  let p_10 := p 10;
  have h1 : p_10 = 13 / 90 := sorry;
  have h2 : 100 * 13 + 90 = 1390 := by norm_num;
  exact ⟨h1, h2⟩

end probability_computation_l366_366327


namespace Cody_fewest_cookies_l366_366395

-- Define the areas of the cookies
def Leah_area : ℝ := 10
def Cody_area : ℝ := 12
def Nina_area : ℝ := 8
def Sam_area : ℝ := 9

-- Define the number of cookies made by Leah with the dough
def Leah_cookies (D : ℝ) : ℝ := D / Leah_area
def Cody_cookies (D : ℝ) : ℝ := D / Cody_area
def Nina_cookies (D : ℝ) : ℝ := D / Nina_area
def Sam_cookies (D : ℝ) : ℝ := D / Sam_area

-- Problem statement
theorem Cody_fewest_cookies (D : ℝ) (h1 : Leah_cookies D = 15) : 
  (Cody_cookies D < Leah_cookies D) ∧ 
  (Cody_cookies D < Nina_cookies D) ∧ 
  (Cody_cookies D < Sam_cookies D) := 
by 
  sorry

end Cody_fewest_cookies_l366_366395


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366361

theorem remainder_3001_3002_3003_3004_3005_mod_17 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := 
begin
  sorry
end

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366361


namespace total_money_is_98_39_l366_366477

def money_eric (ben_dollars : ℝ) : ℝ := ben_dollars - 10
def money_ben (jack_dollars : ℝ) : ℝ := ((jack_dollars / 1.20) - 9) * 1.20
def money_anna (jack_dollars : ℝ) : ℝ := ((2 * (jack_dollars / 1.35)) * 1.35)
def money_jack : ℝ := 26

def total_money (jack_dollars : ℝ) : ℝ :=
  (money_eric (money_ben jack_dollars)) + (money_ben jack_dollars) + jack_dollars + (money_anna jack_dollars)

theorem total_money_is_98_39 : total_money money_jack = 98.39 := by
  sorry

end total_money_is_98_39_l366_366477


namespace find_ordered_pairs_l366_366481

theorem find_ordered_pairs (x y : ℝ) :
  x^2 * y = 3 ∧ x + x * y = 4 → (x, y) = (1, 3) ∨ (x, y) = (3, 1 / 3) :=
sorry

end find_ordered_pairs_l366_366481


namespace base_five_to_decimal_l366_366740

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2 => 2 * 5^0
  | 3 => 3 * 5^1
  | 1 => 1 * 5^2
  | _ => 0

theorem base_five_to_decimal : base5_to_base10 2 + base5_to_base10 3 + base5_to_base10 1 = 42 :=
by sorry

end base_five_to_decimal_l366_366740


namespace sum_of_two_lowest_scores_is_161_l366_366678

noncomputable def sum_of_two_lowest_scores (scores : list ℕ) : ℕ :=
  let sorted_scores := scores.qsort (≤)
  in sorted_scores.head! + (sorted_scores.tail!.head!)

theorem sum_of_two_lowest_scores_is_161 (scores : list ℕ) (h_length : scores.length = 7)
  (h_mean : (scores.sum / 7 : ℝ) = 88) (h_median : scores.nth 3 = some 90)
  (h_mode : scores.count 92 > scores.length / 2) :
  sum_of_two_lowest_scores scores = 161 := sorry

end sum_of_two_lowest_scores_is_161_l366_366678


namespace solve_for_x_l366_366976

variables {R : Type*} [Field R]

theorem solve_for_x
  (a b : R) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x : R, (LinearAlgebra.det ![
    ![x + a, x, x],
    ![x, x + a, x],
    ![x, x, x + a + b]] = 0) ↔ 
  ∃ x : R, x = ((-(a^2 + 2 * a * b) + Real.sqrt (a^4 - 2 * a^3 * b)) / (2 * b)) ∨ 
           x = ((-(a^2 + 2 * a * b) - Real.sqrt (a^4 - 2 * a^3 * b)) / (2 * b)) :=
begin
  sorry
end

end solve_for_x_l366_366976


namespace calculate_weight_l366_366256

theorem calculate_weight (W : ℝ) (h : 0.75 * W + 2 = 62) : W = 80 :=
by
  sorry

end calculate_weight_l366_366256


namespace fraction_calculation_l366_366628

theorem fraction_calculation :
  let a := (4 / 7 : ℚ)
  let b := (5 / 6 : ℚ)
  let c := (3 / 8 : ℚ)
  (a⁻³ * b²) * c⁻¹ = 343 / 346 :=
by
  sorry

end fraction_calculation_l366_366628


namespace monotonicity_of_f_solve_inequality_range_of_m_l366_366986

variable {f : ℝ → ℝ}

-- Question (1)
theorem monotonicity_of_f (h_odd : ∀ x, f (-x) = -f x) (h_f1 : f 1 = 1) 
  (h_pos : ∀ a b ∈ Icc (-1:ℝ) 1, a + b ≠ 0 → (f a + f b) / (a + b) > 0) : 
  ∀ x_1 x_2 ∈ Icc (-1:ℝ) 1, x_1 < x_2 → f x_1 < f x_2 := 
sorry

-- Question (2)
theorem solve_inequality (h_odd : ∀ x, f (-x) = -f x) (h_f1 : f 1 = 1) 
  (h_pos : ∀ a b ∈ Icc (-1:ℝ) 1, a + b ≠ 0 → (f a + f b) / (a + b) > 0) :
  ∀ x ∈ Icc (-3/2:ℝ) (-1), f (x + 1/2) + f (1 / (1 - x)) < 0 := 
sorry

-- Question (3)
theorem range_of_m (h_odd : ∀ x, f (-x) = -f x) (h_f1 : f 1 = 1) 
  (h_pos : ∀ a b ∈ Icc (-1:ℝ) 1, a + b ≠ 0 → (f a + f b) / (a + b) > 0)
  (h_upper : ∀ x a ∈ Icc (-1:ℝ) 1, f x ≤ m^2 - 2*a*m + 1) :
  m ≤ -2 ∨ m = 0 ∨ 2 ≤ m := 
sorry

end monotonicity_of_f_solve_inequality_range_of_m_l366_366986


namespace num_pos_divisors_36_l366_366102

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366102


namespace sum_of_squared_residuals_and_correlation_coefficient_l366_366581

theorem sum_of_squared_residuals_and_correlation_coefficient (x y : List ℝ) :
  (∀ i j, i ≠ j → (x i - x j) * (y i - y j) ≠ 0) →
  (sum_of_squared_residuals x y = 0 ∧ correlation_coefficient x y = 1) :=
sorry

end sum_of_squared_residuals_and_correlation_coefficient_l366_366581


namespace area_of_triangle_l366_366351

theorem area_of_triangle : 
  let L1 : ℝ → ℝ := λ x, 3 * x - 6
  let L2 : ℝ → ℝ := λ x, -2 * x + 14
  let y_intercept_L1 := (0, -6)
  let y_intercept_L2 := (0, 14)
  let intersection := (4, 6)
  let base := 14 - (-6)
  let height := 4
  base * height / 2 = 40 :=
by
  let L1 : ℝ → ℝ := λ x, 3 * x - 6
  let L2 : ℝ → ℝ := λ x, -2 * x + 14
  let y_intercept_L1 := (0, -6)
  let y_intercept_L2 := (0, 14)
  let intersection := (4, 6)
  let base := 14 - (-6)
  let height := 4
  have h1 : base = 20 := rfl
  have h2 : height = 4 := rfl
  have h3 : base * height / 2 = 40 := sorry
  exact h3

end area_of_triangle_l366_366351


namespace find_k_l366_366331

noncomputable def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, ∃ d : ℚ, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
∑ i in finset.range n, a (i + 1)

theorem find_k
  (a : ℕ → ℚ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a 9 = sum_first_n_terms a 4)
  (h3 : a 4 + a k = 0) : k = 10 :=
sorry

end find_k_l366_366331


namespace solution_of_system_l366_366295

def log4 (n : ℝ) : ℝ := log n / log 4

theorem solution_of_system (x y : ℝ) (hx : x + y = 20)
  (hy : log4 x + log4 y = 1 + log4 9) :
  (x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18) :=
by sorry

end solution_of_system_l366_366295


namespace cone_ratio_approx_l366_366513

noncomputable def cone_min_surface_area_ratio (V : ℝ) : ℝ :=
  let r := Real.cbrt (3 * V / Real.pi) in
  let h := 3 * Real.cbrt (V / Real.pi) in
  h / r

theorem cone_ratio_approx (V : ℝ) : 
  cone_min_surface_area_ratio V = 2.08 :=
sorry

end cone_ratio_approx_l366_366513


namespace rectangle_area_l366_366214

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366214


namespace find_a_l366_366984

-- Define the hypotheses
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

noncomputable def a_val (a : ℝ) : ℂ := (a : ℂ) - complex.I
noncomputable def denom_val : ℂ := 1 + complex.I

-- Main statement
theorem find_a (a : ℝ) (h: is_purely_imaginary (a_val a / denom_val)) : a = -1 :=
sorry

end find_a_l366_366984


namespace tangent_of_curve_at_point_l366_366697

def curve (x : ℝ) : ℝ := x^3 - 4 * x

def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

theorem tangent_of_curve_at_point : 
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ tangent_line x y) :=
sorry

end tangent_of_curve_at_point_l366_366697


namespace area_of_rectangle_l366_366388

theorem area_of_rectangle (w d : ℝ) (h_w : w = 4) (h_d : d = 5) : ∃ l : ℝ, (w^2 + l^2 = d^2) ∧ (w * l = 12) :=
by
  sorry

end area_of_rectangle_l366_366388


namespace number_of_divisors_of_36_l366_366119

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366119


namespace number_of_divisors_36_l366_366088

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366088


namespace non_differentiable_counter_example_continuous_implies_differentiable_l366_366392

noncomputable def example_counter_example_f (x : ℝ) : ℝ :=
  if x ∈ Set.uprod (Set.range (Set.univ : Set ℚ)) then 1 else 0

noncomputable def example_g (x : ℝ) : ℝ := 0

def example_a_n (n : ℕ) : ℝ := 1 / (n : ℝ)

theorem non_differentiable_counter_example :
  let f := example_counter_example_f
  let g := example_g
  let a_n := example_a_n
  (∀ x : ℝ, g' x = lim (n → ∞) (f (x + a_n n) - f x) / (a_n n)) →
  ¬ (differentiable f) :=
sorry

theorem continuous_implies_differentiable {f : ℝ → ℝ} {g : ℝ → ℝ}
  (a_n : ℕ → ℝ) (h_a_n : ∀ n, a_n n > 0) (h_a_n_zero : lim (n → ∞) a_n n = 0) :
  (∀ x : ℝ, differentiable g ∧
    (∀ x : ℝ, g' x = lim (n → ∞) (f (x + a_n n) - f x) / (a_n n))) →
  continuous f →
  differentiable f :=
sorry

end non_differentiable_counter_example_continuous_implies_differentiable_l366_366392


namespace isosceles_triangle_count_l366_366607

-- Define the vertices of the segment AB
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (5, 3)

-- Define the set of all possible points (C) on the 7x7 grid
def Points : finset (ℝ × ℝ) :=
  ((finset.range 7).product (finset.range 7))

-- Function to check if triangle ABC is isosceles
def is_isosceles (A B C : (ℝ × ℝ)) : Prop :=
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  let AC := (A.1 - C.1)^2 + (A.2 - C.2)^2 in
  let BC := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
  AB = AC ∨ AB = BC ∨ AC = BC

-- Count the points C that satisfy the condition
def count_isosceles_points (A B : (ℝ × ℝ)) (Points : finset (ℝ × ℝ)) : ℕ :=
  Points.filter (is_isosceles A B).card

theorem isosceles_triangle_count : count_isosceles_points A B Points = 8 := 
  sorry

end isosceles_triangle_count_l366_366607


namespace intersection_M_N_l366_366982

/-- Define the set M as pairs (x, y) such that x + y = 2. -/
def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }

/-- Define the set N as pairs (x, y) such that x - y = 2. -/
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 2 }

/-- The intersection of sets M and N is the single point (2, 0). -/
theorem intersection_M_N : M ∩ N = { (2, 0) } :=
by
  sorry

end intersection_M_N_l366_366982


namespace sum_in_base_b_l366_366650

-- Definitions needed to articulate the problem
def base_b_value (n : ℕ) (b : ℕ) : ℕ :=
  match n with
  | 12 => b + 2
  | 15 => b + 5
  | 16 => b + 6
  | 3146 => 3 * b^3 + 1 * b^2 + 4 * b + 6
  | _  => 0

def s_in_base_b (b : ℕ) : ℕ :=
  base_b_value 12 b + base_b_value 15 b + base_b_value 16 b

theorem sum_in_base_b (b : ℕ) (h : (base_b_value 12 b) * (base_b_value 15 b) * (base_b_value 16 b) = base_b_value 3146 b) :
  s_in_base_b b = 44 := by
  sorry

end sum_in_base_b_l366_366650


namespace total_amount_due_is_correct_l366_366433

-- Define the initial conditions
def initial_amount : ℝ := 350
def first_year_interest_rate : ℝ := 0.03
def second_and_third_years_interest_rate : ℝ := 0.05

-- Define the total amount calculation after three years.
def total_amount_after_three_years (P : ℝ) (r1 : ℝ) (r2 : ℝ) : ℝ :=
  let first_year_amount := P * (1 + r1)
  let second_year_amount := first_year_amount * (1 + r2)
  let third_year_amount := second_year_amount * (1 + r2)
  third_year_amount

theorem total_amount_due_is_correct : 
  total_amount_after_three_years initial_amount first_year_interest_rate second_and_third_years_interest_rate = 397.45 :=
by
  sorry

end total_amount_due_is_correct_l366_366433


namespace domain_of_f_l366_366312

-- Define the function y = 1 / log_base_5(2x - 1)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.log 5 (2 * x - 1)

-- State the conditions for the domain of the function
def condition1 (x : ℝ) : Prop := 2 * x - 1 > 0
def condition2 (x : ℝ) : Prop := Real.log 5 (2 * x - 1) ≠ 0

-- Prove the domain of f is (1 / 2, 1) ∪ (1, ∞)
theorem domain_of_f : {x : ℝ | condition1 x ∧ condition2 x} = {x : ℝ | x > 1 / 2 ∧ x ≠ 1} :=
by
  sorry

end domain_of_f_l366_366312


namespace final_car_speed_l366_366844

theorem final_car_speed :
  ∀ (car_speed : ℕ → ℕ), (∀ n, n ∈ (finset.range 31) → car_speed n = 60 + n + 1) →
  let cars := finset.range 31 in
  let speeds := cars.image car_speed in
  (∃ (remaining_speed : ℕ), remaining_speed ∈ speeds ∧
    ∀ s ∈ speeds.erase remaining_speed, 2 * remaining_speed = s + car_speed (31 - s - 1)) →
  remaining_speed = 76 :=
begin
  intros car_speed hcar_speed,
  let speeds := finset.range 31 |>.image car_speed,
  use 76,
  split,
  { rw finset.mem_image,
    use 15,
    simp [hcar_speed, show 60 + 15 + 1 = 76],
  },
  { intros s hs,
    simp,
    sorry,
  }
end

end final_car_speed_l366_366844


namespace largest_sum_15_l366_366708

theorem largest_sum_15 (a b c : ℕ) (h₁ : a + b + c = 15) : 
  ∃ n l : ℕ, l = 5 ∧ (∀ i, i < l → n + i = (n + i - 1) + 1) ∧ ( ∑ x in range l, n + x) = 15 :=
sorry

end largest_sum_15_l366_366708


namespace Beth_finishes_first_l366_366887

variable (x y : ℝ)

def area_Andy : ℝ := x
def area_Beth : ℝ := (2 / 3) * x
def area_Carlos : ℝ := (1 / 4) * x

def rate_Andy : ℝ := y
def rate_Beth : ℝ := (3 / 4) * y
def rate_Carlos : ℝ := y / 4

def time_Andy : ℝ := area_Andy x / rate_Andy y
def time_Beth : ℝ := area_Beth x / rate_Beth y
def time_Carlos : ℝ := area_Carlos x / rate_Carlos y

theorem Beth_finishes_first :
  time_Beth x y < time_Andy x y ∧ time_Beth x y < time_Carlos x y :=
by sorry

end Beth_finishes_first_l366_366887


namespace area_of_rectangle_ABCD_l366_366180

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l366_366180


namespace michaels_brother_money_end_l366_366654

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l366_366654


namespace find_PF_length_l366_366533

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define point P on the parabola
def P : ℝ × ℝ := (6, 4 * Real.sqrt 3)

-- Define point A on the directrix and on the line AF
def A : ℝ × ℝ := (-2, 4 * Real.sqrt 3)

-- Define the line AF slope condition
def slope_AF (F A : ℝ × ℝ) : Prop := (A.2 - F.2) / (A.1 - F.1) = -Real.sqrt 3

-- The main theorem we want to prove
theorem find_PF_length : 
  (parabola P.1 P.2) → 
  (directrix A.1) → 
  (slope_AF focus A) → 
  ((Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2)) = 8) := 
by 
  intro hp hd hs
  sorry

end find_PF_length_l366_366533


namespace cyclic_quadrilateral_projections_equal_l366_366859

theorem cyclic_quadrilateral_projections_equal
  {A B C D O : Type} [inhabited O] (h_cyclic: cyclic_quadrilateral A B C D O)
  (h_diameter: diameter A C O)
  (h_perpendicular_E: perp_foot C B D E)
  (h_perpendicular_F: perp_foot A B D F)
  : distance B E = distance D F 
:= sorry

end cyclic_quadrilateral_projections_equal_l366_366859


namespace no_such_function_exists_l366_366929

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x ^ 2 - 1996 :=
by
  sorry

end no_such_function_exists_l366_366929


namespace wizard_can_achieve_one_way_traffic_l366_366841

theorem wizard_can_achieve_one_way_traffic (N : ℕ) (roads : fin N → fin N → Prop) (overpass_or_underpass : ∀ i j, i ≠ j → (roads i j ∨ roads j i)) :
  (∃ f : fin N → fin N, ∀ i j, i < j → roads (f i) (f j)) ∧ -- part (a)
  (∃ start stop, (∀ i, roads start i) ∧ (∀ j, ¬roads j stop)) ∧ -- part (b)
  (∃ f : fin N → fin N, ∀ i, f i < f (i + 1)) ∧ -- part (c)
  ∃ permutations, permutations = factorial N -- part (d)
:= sorry

end wizard_can_achieve_one_way_traffic_l366_366841


namespace complete_the_square_l366_366379

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l366_366379


namespace rectangle_area_l366_366208

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366208


namespace lcm_18_24_eq_72_l366_366814

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366814


namespace lcm_18_24_l366_366803

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366803


namespace residue_of_neg_1237_mod_37_l366_366472

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by 
  sorry

end residue_of_neg_1237_mod_37_l366_366472


namespace length_of_first_platform_l366_366874

theorem length_of_first_platform 
  (train_length : ℕ) (first_time : ℕ) (second_platform_length : ℕ) (second_time : ℕ)
  (speed_first : ℕ) (speed_second : ℕ) :
  train_length = 230 → 
  first_time = 15 → 
  second_platform_length = 250 → 
  second_time = 20 → 
  speed_first = (train_length + L) / first_time →
  speed_second = (train_length + second_platform_length) / second_time →
  speed_first = speed_second →
  (L : ℕ) = 130 :=
by
  sorry

end length_of_first_platform_l366_366874


namespace range_a_l366_366003

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem range_a (a : ℝ) :
  (∀ x1 ∈ Ioo (1/2 : ℝ) 2, ∃ x2 ∈ Ioo (1/2 : ℝ) 2, log_base_2 x1 + a = x2^2 + 2) →
  (∀ x1 x2 ∈ Icc (0 : ℝ) 1, a + 3 * x2 > 4 ^ x1 -> False) →
  (13 / 4 <= a) ∧ (a <= 4) :=
by
  sorry

end range_a_l366_366003


namespace lcm_18_24_l366_366756
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366756


namespace amount_donated_to_first_orphanage_l366_366679

theorem amount_donated_to_first_orphanage
    (total_amount_donated : ℕ)
    (amount_given_to_second_orphanage : ℕ)
    (amount_given_to_third_orphanage : ℕ)
    (total_condition : total_amount_donated = 650)
    (second_condition : amount_given_to_second_orphanage = 225)
    (third_condition : amount_given_to_third_orphanage = 250) :
    ∃ amount_given_to_first_orphanage,
        total_amount_donated = amount_given_to_first_orphanage + amount_given_to_second_orphanage + amount_given_to_third_orphanage ∧
        amount_given_to_first_orphanage = 175 :=
by
    use 175
    split
    sorry

end amount_donated_to_first_orphanage_l366_366679


namespace frequency_of_middle_rectangle_l366_366591

theorem frequency_of_middle_rectangle
    (n : ℕ)
    (A : ℕ)
    (h1 : A + (n - 1) * A = 160) :
    A = 32 :=
by
  sorry

end frequency_of_middle_rectangle_l366_366591


namespace elaine_jerry_ratio_l366_366436

theorem elaine_jerry_ratio (j e g : ℕ) (h1 : j = 3) (h2 : g = e / 3) (h3 : j + e + g = 11) : e / j = 2 :=
by {
  sorry,
}

end elaine_jerry_ratio_l366_366436


namespace area_of_ABCD_l366_366190

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366190


namespace number_of_divisors_of_36_l366_366112

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366112


namespace number_of_divisors_36_l366_366053

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366053


namespace rhombus_area_l366_366468

noncomputable def sqrt125 : ℝ := Real.sqrt 125

theorem rhombus_area 
  (p q : ℝ) 
  (h1 : p < q) 
  (h2 : p + 8 = q) 
  (h3 : ∀ a b : ℝ, a^2 + b^2 = 125 ↔ 2*a = p ∧ 2*b = q) : 
  p*q/2 = 60.5 :=
by
  sorry

end rhombus_area_l366_366468


namespace lcm_18_24_eq_72_l366_366765

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366765


namespace dan_initial_money_l366_366908

def money_left : ℕ := 3
def cost_candy : ℕ := 2
def initial_money : ℕ := money_left + cost_candy

theorem dan_initial_money :
  initial_money = 5 :=
by
  -- Definitions according to problem
  let money_left := 3
  let cost_candy := 2

  have h : initial_money = money_left + cost_candy := by rfl
  rw [h]

  -- Show the final equivalence
  show 3 + 2 = 5
  rfl

end dan_initial_money_l366_366908


namespace max_stamps_l366_366587

theorem max_stamps (price_per_stamp : ℕ) (total_money_cents : ℕ) (h_price : price_per_stamp = 25) (h_total_money : total_money_cents = 5000) :
  ∃ n, n ≤ 200 ∧ 25 * n ≤ total_money_cents :=
by
  have h : 5000 / 25 = 200 := by norm_num
  use 200
  simp [h_total_money, h, h_price]
  split
  · norm_num
  · linarith
  done

end max_stamps_l366_366587


namespace rectangle_area_l366_366212

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366212


namespace num_pos_divisors_36_l366_366137

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366137


namespace exists_pair_sum_mod_10_l366_366509

theorem exists_pair_sum_mod_10 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : list.nodup [a1, a2, a3, a4, a5, a6, a7]) (n : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (([a1, a2, a3, a4, a5, a6, a7].nth i).get_or_else 0 + ([a1, a2, a3, a4, a5, a6, a7].nth j).get_or_else 0) % 10 = n % 10 := 
sorry

end exists_pair_sum_mod_10_l366_366509


namespace f_gt_g_l366_366507

def f (n : ℕ) : ℝ := 
  let rec prod (i : ℕ) (acc : ℝ) := 
    if i = 0 then acc 
    else prod (i - 1) (acc * (1 + 1 / (3 * i - 2)))
  prod n 1

def g (n : ℕ) : ℝ := (3 * n + 1 : ℝ)^(1/3)

theorem f_gt_g (n : ℕ) (hn : 0 < n) : f n > g n :=
by
  sorry

end f_gt_g_l366_366507


namespace total_length_figure2_l366_366877

-- Define the initial lengths of each segment in Figure 1.
def initial_length_horizontal1 := 5
def initial_length_vertical1 := 10
def initial_length_horizontal2 := 4
def initial_length_vertical2 := 3
def initial_length_horizontal3 := 3
def initial_length_vertical3 := 5
def initial_length_horizontal4 := 4
def initial_length_vertical_sum := 10 + 3 + 5

-- Define the transformations.
def bottom_length := initial_length_horizontal1
def rightmost_vertical_length := initial_length_vertical1 - 2
def top_horizontal_length := initial_length_horizontal2 - 3
def leftmost_vertical_length := initial_length_vertical1

-- Define the total length in Figure 2 as a theorem to be proved.
theorem total_length_figure2:
  bottom_length + rightmost_vertical_length + top_horizontal_length + leftmost_vertical_length = 24 := by
  sorry

end total_length_figure2_l366_366877


namespace loom_weaving_rate_l366_366851

theorem loom_weaving_rate :
  (119.04761904761905: ℝ) ≠ 0 → 
  (15 / 119.04761904761905) ≈ 0.126 :=
by
  sorry

end loom_weaving_rate_l366_366851


namespace probability_correct_answers_at_least_half_l366_366298

theorem probability_correct_answers_at_least_half :
  let n := 16
  let k := 8
  let p := 3 / 4
  let threshold := 0.999
  binomial_cdf_complement n p k ≤ threshold :=
by
  sorry

end probability_correct_answers_at_least_half_l366_366298


namespace max_stamps_l366_366589

theorem max_stamps (price_per_stamp : ℕ) (available_money : ℕ) (h1 : price_per_stamp = 25) (h2 : available_money = 5000) :
  ∃ (n : ℕ), n = 200 ∧ price_per_stamp * n ≤ available_money ∧ ∀ m, (price_per_stamp * m ≤ available_money) → m ≤ 200 :=
by
  use 200
  split; sorry

end max_stamps_l366_366589


namespace lcm_18_24_eq_72_l366_366775

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366775


namespace lcm_18_24_l366_366785

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366785


namespace lcm_18_24_l366_366782

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366782


namespace hasan_number_ratio_l366_366553

theorem hasan_number_ratio (a b : ℕ) (ha : 1 ≤ a) (ha9 : a ≤ 9) (hb : 0 ≤ b) (hb9 : b ≤ 9) :
  let original_number := 10 * a + b,
      four_digit_number := 1000 * a + 100 * b + 10 * a + b in
  (four_digit_number / original_number) = 101 := 
by
  sorry

end hasan_number_ratio_l366_366553


namespace wolf_and_nobel_prize_laureates_l366_366397

-- Definitions from the conditions
def num_total_scientists : ℕ := 50
def num_wolf_prize_laureates : ℕ := 31
def num_nobel_prize_laureates : ℕ := 29
def num_no_wolf_prize_and_yes_nobel := 3 -- N_W = N_W'
def num_without_wolf_or_nobel : ℕ := num_total_scientists - num_wolf_prize_laureates - 11 -- Derived from N_W' 

-- The statement to be proved
theorem wolf_and_nobel_prize_laureates :
  ∃ W_N, W_N = num_nobel_prize_laureates - (19 - 3) ∧ W_N = 18 :=
  by
    sorry

end wolf_and_nobel_prize_laureates_l366_366397


namespace polynomial_identity_l366_366989

noncomputable def g (a : ℝ) : Polynomial ℝ :=
  Polynomial.X^3 + a * Polynomial.X^2 + Polynomial.X + 20

noncomputable def f (a b c : ℝ) : Polynomial ℝ :=
  Polynomial.X^4 + Polynomial.X^3 + b * Polynomial.X^2 + 50 * Polynomial.X + c

theorem polynomial_identity (a b c : ℝ) (hx : ∃ (r : ℝ), (f a b c) = (g a) * (Polynomial.X - Polynomial.C r)) :
  f a b c .eval 1 = -217 :=
by sorry

end polynomial_identity_l366_366989


namespace inequality_proof_l366_366002

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 1) : 
  (sqrt (a * (1 - a)) / (1 + a)) + (sqrt (b * (1 - b)) / (1 + b)) + (sqrt (c * (1 - c)) / (1 + c)) 
  ≥ 3 * sqrt ((a * b * c) / ((1 - a) * (1 - b) * (1 - c))) :=
sorry

end inequality_proof_l366_366002


namespace num_pos_divisors_36_l366_366103

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366103


namespace red_section_no_damage_probability_l366_366455

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l366_366455


namespace hyperbola_eccentricity_is_2_l366_366176

noncomputable def parabola_focus : ℝ × ℝ := (2, 0)

noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ :=
  let c := sqrt (a^2 + 3) in c / a

theorem hyperbola_eccentricity_is_2 :
  (parabola_focus = (2, 0)) →
  (∀ a : ℝ, a = 1 → hyperbola_eccentricity a = 2) :=
by
  sorry

end hyperbola_eccentricity_is_2_l366_366176


namespace complete_the_square_l366_366380

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l366_366380


namespace lcm_18_24_l366_366754
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366754


namespace lcm_18_24_l366_366783

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366783


namespace min_AP_2BP_3CP_l366_366243

open Point

noncomputable def minimum_value_AP_2BP_3CP (P : Point) (A B C : Point) :=
  dist A P + 2 * dist B P + 3 * dist C P

theorem min_AP_2BP_3CP {A B C : Point} :
  dist A B = 2021 → dist A C = 2022 → dist B C = 2023 →
  ∀ P, minimum_value_AP_2BP_3CP P A B C ≥ 6068 :=
by
  intros hAB hAC hBC P
  sorry

end min_AP_2BP_3CP_l366_366243


namespace incorrect_statement_C_l366_366500

-- Let α and β be two distinct planes
variables {α β : Type} [plane α] [plane β]

-- Let m and n be two distinct lines
variables {m n : Type} [line m] [line n]

-- Assumptions
variables (h1 : m ∥ n) (h2 : m ⊥ α)
variables (h3 : m ⊥ α) (h4 : m ⊥ β)
variables (h5 : m ∥ α) (h6 : α ∩ β = n)
variables (h7 : m ⊥ α) (h8 : m ⊂ β)

theorem incorrect_statement_C: ¬(m ∥ n) :=
by sorry

end incorrect_statement_C_l366_366500


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366359

theorem remainder_3001_3002_3003_3004_3005_mod_17 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366359


namespace shortest_remaining_side_length_l366_366425

noncomputable def triangle_has_right_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem shortest_remaining_side_length {a b : ℝ} (ha : a = 5) (hb : b = 12) (h_right_angle : ∃ c, triangle_has_right_angle a b c) :
  ∃ c, c = 5 :=
by 
  sorry

end shortest_remaining_side_length_l366_366425


namespace num_pos_divisors_36_l366_366123

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366123


namespace chemistry_problem_l366_366470

-- Definition of variables
variables (KOH NH4I HI KI K_NH4_OH2 : ℝ)
variables (reaction1 reaction2 : Prop)

-- Conditions
def initial_conditions := KOH = 1 ∧ NH4I = 1 ∧ HI = 0.5 ∧ 
  (reaction1 ↔ KOH + NH4I = KI + NH3 + H2O) ∧ 
  (reaction2 ↔ 2 * KOH + NH4I + HI = K_NH4_OH2 + KI)

-- Proof statement
theorem chemistry_problem (h : initial_conditions) : 
  KI = 1 ∧ K_NH4_OH2 = 0 ∧ KOH = 0 ∧ HI = 0.5 :=
by
  sorry

end chemistry_problem_l366_366470


namespace club_men_count_l366_366856

theorem club_men_count (M W : ℕ) (h1 : M + W = 30) (h2 : M + (W / 3 : ℕ) = 20) : M = 15 := by
  -- proof omitted
  sorry

end club_men_count_l366_366856


namespace paving_cost_correct_l366_366839

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_m : ℝ := 300
def area (length : ℝ) (width : ℝ) : ℝ := length * width
def cost (area : ℝ) (rate : ℝ) : ℝ := area * rate

theorem paving_cost_correct :
  cost (area length width) rate_per_sq_m = 6187.50 :=
by
  sorry

end paving_cost_correct_l366_366839


namespace polynomial_properties_l366_366909

noncomputable def p (x : ℕ) : ℕ := 2 * x^3 + x + 4

theorem polynomial_properties :
  p 1 = 7 ∧ p 10 = 2014 := 
by
  -- Placeholder for proof
  sorry

end polynomial_properties_l366_366909


namespace perpendicular_line_sin_2theta_l366_366969

theorem perpendicular_line_sin_2theta (θ : ℝ) 
  (h_perpendicular : ∃ l : Line, ∃ (inclination_angle : ℝ), l.inclination_angle = θ ∧ l ⊥ Line.mk 1 2 (-3)) :
  Real.sin (2 * θ) = 4 / 5 :=
sorry

end perpendicular_line_sin_2theta_l366_366969


namespace find_m_l366_366156

theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) : m = 2 :=
sorry

end find_m_l366_366156


namespace lcm_18_24_l366_366787

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366787


namespace max_number_of_cubes_l366_366818

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end max_number_of_cubes_l366_366818


namespace complement_intersection_l366_366651

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) : (U \ (M ∩ N)) = {1, 4} := 
by
  sorry

end complement_intersection_l366_366651


namespace percentage_owning_pets_is_48_l366_366167

/-- 
In a school of 500 students, 80 students own cats, 120 students own dogs, and 40 students own rabbits. 
No student owns more than one type of pet. Prove that the percentage of students who own at least one of the three pets is 48%.
-/

theorem percentage_owning_pets_is_48 
  (total_students : ℕ := 500) 
  (cat_owners : ℕ := 80) 
  (dog_owners : ℕ := 120) 
  (rabbit_owners : ℕ := 40) 
  (no_multiple_pets : ∀ s ∈ {cat_owners, dog_owners, rabbit_owners}, s ≠ total_students) 
  : 
  ((cat_owners + dog_owners + rabbit_owners) / total_students * 100) = 48 := 
  by 
    sorry

end percentage_owning_pets_is_48_l366_366167


namespace charley_pencils_final_count_l366_366445

def charley_initial_pencils := 50
def lost_pencils_while_moving := 8
def misplaced_fraction_first_week := 1 / 3
def lost_fraction_second_week := 1 / 4

theorem charley_pencils_final_count:
  let initial := charley_initial_pencils
  let after_moving := initial - lost_pencils_while_moving
  let misplaced_first_week := misplaced_fraction_first_week * after_moving
  let remaining_after_first_week := after_moving - misplaced_first_week
  let lost_second_week := lost_fraction_second_week * remaining_after_first_week
  let final_pencils := remaining_after_first_week - lost_second_week
  final_pencils = 21 := 
sorry

end charley_pencils_final_count_l366_366445


namespace infinite_series_sum_l366_366446

theorem infinite_series_sum :
  (∑' n : Nat, (4 * n + 1) / ((4 * n - 1)^2 * (4 * n + 3)^2)) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l366_366446


namespace value_of_f_f_2_l366_366154

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then log x / log (1/2) 
else 2^x

theorem value_of_f_f_2 : f (f 2) = 1 / 2 :=
by
  sorry

end value_of_f_f_2_l366_366154


namespace constant_term_expansion_l366_366606

theorem constant_term_expansion :
  let c := (10.choose 2) * 4 in
  c = 180 :=
by
  let c := (10.choose 2) * 4
  sorry

end constant_term_expansion_l366_366606


namespace lcm_18_24_l366_366789

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l366_366789


namespace num_positive_divisors_36_l366_366097

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366097


namespace ral_current_age_l366_366273

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l366_366273


namespace polynomial_product_c_l366_366629

theorem polynomial_product_c (b c : ℝ) (h1 : b = 2 * c - 1) (h2 : (x^2 + b * x + c) = 0 → (∃ r : ℝ, x = r)) :
  c = 1 / 2 :=
sorry

end polynomial_product_c_l366_366629


namespace constant_term_in_expansion_of_fraction_l366_366008

theorem constant_term_in_expansion_of_fraction (
    (∫ x in (0:ℝ)..(2:ℝ), (2*x + 1)) = 6
  ) : (finset.nat.choose 6 4) = 15 :=
by
  -- Definitions for integral and binomial coefficient
  let m := (∫ x in 0..2, (2*x + 1))
  
  -- Ensure m is calculated correctly
  have m_eq : m = 6 := sorry
  
  -- Expand the expression and find the constant term
  have constant_term := finset.nat.choose 6 4
  
  -- Prove the correct answer
  show constant_term = 15

end constant_term_in_expansion_of_fraction_l366_366008


namespace part1_solution_value_of_k_and_a_n_part2i_range_of_k_part2ii_Qn_less_Pn_l366_366971

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 3
| (n+2) := 3 * a (n+1) - 2 * a n

def b (n : ℕ) : ℝ := Real.log2 (a n + 1)

def c (n : ℕ) : ℝ := 1 / (b n * b (n + 1))

def d (n : ℕ) : ℝ := b (n + 3) / (b n * b (n + 1) * (a (n + 1) + 1))

def P (n : ℕ) : ℝ := (Finset.range n).sum c

def Q (n : ℕ) : ℝ := (Finset.range n).sum d

theorem part1_solution_value_of_k_and_a_n :
  exists k : ℝ, (k = 2) ∧ (∀ n, a n = 2^n - 1) :=
sorry

theorem part2i_range_of_k (k : ℝ) (n : ℕ) :
  (P n ≤ k * (n + 4)) → k ≥ 1 / 9 :=
sorry

theorem part2ii_Qn_less_Pn (n : ℕ) : Q n < P n :=
sorry

end part1_solution_value_of_k_and_a_n_part2i_range_of_k_part2ii_Qn_less_Pn_l366_366971


namespace min_value_one_over_a_plus_two_over_b_l366_366036

/-- Given a > 0, b > 0, 2a + b = 1, prove that the minimum value of (1/a) + (2/b) is 8 --/
theorem min_value_one_over_a_plus_two_over_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a) + (2 / b) ≥ 8 :=
sorry

end min_value_one_over_a_plus_two_over_b_l366_366036


namespace lcm_18_24_eq_72_l366_366779

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366779


namespace number_of_elements_in_list_l366_366049

-- Define the arithmetic sequence
def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  ∀ n : ℕ, n < seq.length - 1 → seq[n + 1] - seq[n] = -5

-- Define the given list
def given_list := [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45]

-- The problem statement
theorem number_of_elements_in_list : 
    is_arithmetic_sequence given_list → 
    given_list.length = 25 :=
by
  sorry

end number_of_elements_in_list_l366_366049


namespace sum_of_terms_arithmetic_sequence_l366_366247

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
axiom S_k : S k = 2
axiom S_3k : S (3 * k) = 18

-- The statement to prove
theorem sum_of_terms_arithmetic_sequence : S (4 * k) = 32 := by
  sorry

end sum_of_terms_arithmetic_sequence_l366_366247


namespace overlapping_patches_exist_l366_366402

-- Define the patches and their properties
def coat_area : ℝ := 1
def num_patches : ℕ := 5
def patch_area_lower_bound : ℝ := 1 / 2

-- There exists at least two patches whose overlapping area is at least 1/5
theorem overlapping_patches_exist
  (patches : Fin num_patches → Set ℝ)
  (h_patch_cover : (⋃ i, patches i).measure = coat_area)
  (h_patch_area : ∀ i, (patches i).measure ≥ patch_area_lower_bound) :
  ∃ i j, i ≠ j ∧ (patches i ∩ patches j).measure ≥ 1 / 5 :=
sorry

end overlapping_patches_exist_l366_366402


namespace num_pos_divisors_36_l366_366120

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366120


namespace correct_average_weight_l366_366694

-- Definitions
def initial_average_weight : ℝ := 58.4
def number_of_boys : ℕ := 20
def misread_weight_initial : ℝ := 56
def misread_weight_correct : ℝ := 68

-- Correct average weight
theorem correct_average_weight : 
  let initial_total_weight := initial_average_weight * (number_of_boys : ℝ)
  let difference := misread_weight_correct - misread_weight_initial
  let correct_total_weight := initial_total_weight + difference
  let correct_average_weight := correct_total_weight / (number_of_boys : ℝ)
  correct_average_weight = 59 :=
by
  -- Insert the proof steps if needed
  sorry

end correct_average_weight_l366_366694


namespace num_ways_to_sum_1_l366_366142

theorem num_ways_to_sum_1 :
  ∃ (f : Fin 6 → ℕ), (∀ i, f i = 0 ∨ f i = 1) ∧ 
  ((∑ i, (if f i = 0 then -1 else 1) * ((i + 1) / 7 : ℚ)) = 1) ∧ 
  (count f = 4) :=
sorry

end num_ways_to_sum_1_l366_366142


namespace g_min_value_l366_366490

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l366_366490


namespace socks_different_colors_l366_366164

theorem socks_different_colors :
  let white := 5
  let brown := 3
  let blue := 4
  (white * brown + brown * blue + white * blue) = 47 :=
by
  let white := 5
  let brown := 3
  let blue := 4
  simp [white, brown, blue]
  exact Eq.refl 47

end socks_different_colors_l366_366164


namespace count_values_non_divisible_by_product_l366_366639

noncomputable def product_of_proper_divisors (n : ℕ) : ℕ := 
  ∏ d in (Finset.filter (λ x, x < n ∧ ¬ x = 0 ∧ n % x = 0) (Finset.range n)), d

def does_not_divide (n : ℕ) : Prop := ¬ (n ∣ product_of_proper_divisors n)

theorem count_values_non_divisible_by_product :
  (Finset.card (Finset.filter does_not_divide (Finset.Icc 2 100))) = 31 :=
sorry

end count_values_non_divisible_by_product_l366_366639


namespace total_games_in_season_l366_366304

theorem total_games_in_season (n_teams : ℕ) (games_between_each_team : ℕ) (non_conf_games_per_team : ℕ) 
  (h_teams : n_teams = 8) (h_games_between : games_between_each_team = 3) (h_non_conf : non_conf_games_per_team = 3) :
  let games_within_league := (n_teams * (n_teams - 1) / 2) * games_between_each_team
  let games_outside_league := n_teams * non_conf_games_per_team
  games_within_league + games_outside_league = 108 := by
  sorry

end total_games_in_season_l366_366304


namespace rectangle_area_l366_366205

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366205


namespace solution_set_inequality_inequality_mn_gt_n_div_m_l366_366961

noncomputable def f (x : ℝ) : ℝ := |x + 1|

theorem solution_set_inequality :
  {x : ℝ | f(x + 2) + f(2 * x) ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 0} :=
begin
  sorry
end

theorem inequality_mn_gt_n_div_m (m n : ℝ) (hm : |m| > 1) (hn : |n| > 1) :
  f(m * n) / |m| > f(n / m) :=
begin
  sorry
end

end solution_set_inequality_inequality_mn_gt_n_div_m_l366_366961


namespace lcm_18_24_eq_72_l366_366764

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366764


namespace a_5_equals_31_l366_366548

-- Define the sequence {a_n} recursively
def seq : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * seq n + 1

-- Prove that a_5 equals 31
theorem a_5_equals_31 : seq 5 = 31 := 
by
  sorry

end a_5_equals_31_l366_366548


namespace problem_solution_l366_366505

theorem problem_solution {a b : ℝ} (h : a * b + b^2 = 12) : (a + b)^2 - (a + b) * (a - b) = 24 :=
by sorry

end problem_solution_l366_366505


namespace problem_part1_problem_part2_l366_366391

noncomputable def distinct_remainders_count
    (p : ℕ) [hp : Nat.Prime p]
    (a : Fin p → ℕ) (b : Fin p → ℕ)
    (h_a : ∀ i j, i < j → a i < a j)
    (h_b : ∀ i j, i < j → b i < b j)
    (h_am : ∀ i, a i < p)
    (h_bn : ∀ j, b j < p) : ℕ :=
  (Finset.image (λ (ij : Fin p × Fin p), (a ij.1 + b ij.2) % p) (Finset.univ ×ˢ Finset.univ)).card

theorem problem_part1
    {p m n : ℕ} [hp : Nat.Prime p]
    (a : Fin m → ℕ) (b : Fin n → ℕ)
    (h_a : ∀ i j, i < j → a i < a j)
    (h_b : ∀ i j, i < j → b i < b j)
    (h_am : ∀ i, a i < p)
    (h_bn : ∀ j, b j < p)
    (h_mnp : m + n > p) :
  distinct_remainders_count p a b h_a h_b h_am h_bn = p := sorry

theorem problem_part2
    {p m n : ℕ} [hp : Nat.Prime p]
    (a : Fin m → ℕ) (b : Fin n → ℕ)
    (h_a : ∀ i j, i < j → a i < a j)
    (h_b : ∀ i j, i < j → b i < b j)
    (h_am : ∀ i, a i < p)
    (h_bn : ∀ j, b j < p)
    (h_mnp : m + n ≤ p) :
  distinct_remainders_count p a b h_a h_b h_am h_bn ≥ m + n - 1 := sorry

end problem_part1_problem_part2_l366_366391


namespace mode_of_data_set_is_1_l366_366872

-- Define the conditions
def data_set : List ℕ := [2, 1, 4]
def total_numbers : ℕ := 4
def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

-- State that the average of the set is 2
theorem mode_of_data_set_is_1
  (h1 : average (data_set ++ [1]) = 2) :
  (List.mode (data_set ++ [1]) = some 1) :=
by
  sorry

end mode_of_data_set_is_1_l366_366872


namespace max_stamps_l366_366586

theorem max_stamps (price_per_stamp : ℕ) (total_money_cents : ℕ) (h_price : price_per_stamp = 25) (h_total_money : total_money_cents = 5000) :
  ∃ n, n ≤ 200 ∧ 25 * n ≤ total_money_cents :=
by
  have h : 5000 / 25 = 200 := by norm_num
  use 200
  simp [h_total_money, h, h_price]
  split
  · norm_num
  · linarith
  done

end max_stamps_l366_366586


namespace number_of_terms_in_sequence_l366_366042

-- Definition of the arithmetic sequence parameters
def start_term : Int := 165
def end_term : Int := 45
def common_difference : Int := -5

-- Define a theorem to prove the number of terms in the sequence
theorem number_of_terms_in_sequence :
  ∃ n : Nat, number_of_terms 165 45 (-5) = 25 :=
by
  sorry

end number_of_terms_in_sequence_l366_366042


namespace probability_no_success_l366_366451

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l366_366451


namespace concurrence_of_lines_l366_366348

noncomputable def concurrent_lines (C1 C2 C3 C4 : Circle) 
  (tang1 : tangent_points C1 C2)
  (tang2 : tangent_points C2 C3)
  (tang3 : tangent_points C3 C4)
  (tang4 : tangent_points C4 C1)
  (tang5 : tangent_points C1 C3)
  (tang6 : tangent_points C2 C4) :
  Prop := is_concurrent [connect (tang1) (tang2), connect (tang3) (tang4), connect (tang5) (tang6)]

theorem concurrence_of_lines 
  (C1 C2 C3 C4 : Circle) 
  (tang1 : tangent_points C1 C2)
  (tang2 : tangent_points C2 C3)
  (tang3 : tangent_points C3 C4)
  (tang4 : tangent_points C4 C1)
  (tang5 : tangent_points C1 C3)
  (tang6 : tangent_points C2 C4) :
  concurrent_lines C1 C2 C3 C4 tang1 tang2 tang3 tang4 tang5 tang6 := 
sorry

end concurrence_of_lines_l366_366348


namespace team_points_behind_l366_366221

-- Define the points for Max, Dulce and the condition for Val
def max_points : ℕ := 5
def dulce_points : ℕ := 3
def combined_points_max_dulce : ℕ := max_points + dulce_points
def val_points : ℕ := 2 * combined_points_max_dulce

-- Define the total points for their team and the opponents' team
def their_team_points : ℕ := max_points + dulce_points + val_points
def opponents_team_points : ℕ := 40

-- Proof statement
theorem team_points_behind : opponents_team_points - their_team_points = 16 :=
by
  sorry

end team_points_behind_l366_366221


namespace triangle_bisectors_l366_366842

noncomputable def angles (α γ : ℝ) (hα : α > γ) : Prop :=
∀ (A B C K L M : Type)
  [HasVangle A] [HasVangle C] [HasLinearArithmetic α γ A B C K L M],
  let AK KC AM : ℝ := sorry in
  AK + KC > AM

theorem triangle_bisectors (α γ : ℝ) (hα : α > γ) :
  angles α γ hα :=
by
  undefined -- leave the rest to be defined by future proof

end triangle_bisectors_l366_366842


namespace maximum_area_rectangle_l366_366710

-- Define the conditions
def length (x : ℝ) := x
def width (x : ℝ) := 2 * x
def perimeter (x : ℝ) := 2 * (length x + width x)

-- The proof statement
theorem maximum_area_rectangle (h : perimeter x = 40) : 2 * (length x) * (width x) = 800 / 9 :=
by
  sorry

end maximum_area_rectangle_l366_366710


namespace min_value_of_exp_l366_366033

noncomputable def minimum_value_of_expression (a b : ℝ) : ℝ :=
  (1 - a)^2 + (1 - 2 * b)^2 + (a - 2 * b)^2

theorem min_value_of_exp (a b : ℝ) (h : a^2 ≥ 8 * b) : minimum_value_of_expression a b = 9 / 8 :=
by
  sorry

end min_value_of_exp_l366_366033


namespace mona_cookie_count_l366_366660

theorem mona_cookie_count {M : ℕ} (h1 : (M - 5) + (M - 5 + 10) + M = 60) : M = 20 :=
by
  sorry

end mona_cookie_count_l366_366660


namespace sum_of_d_and_e_l366_366714

theorem sum_of_d_and_e (d e : ℤ) : 
  (∃ d e : ℤ, ∀ x : ℝ, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  sorry

end sum_of_d_and_e_l366_366714


namespace satisfies_equation_l366_366930

noncomputable def f : ℝ → ℝ := λ x, 0

theorem satisfies_equation (x y : ℝ) : 
  |x| * f y + y * f x = f (x * y) + f (x ^ 2) + f (f y) :=
by
  simp [f]
  sorry

end satisfies_equation_l366_366930


namespace sum_first_n_terms_l366_366466

-- Definitions and conditions
def seq (n : ℕ) : ℕ := n^2 + 1

def first_term : Prop := seq 1 = 2
def sum_first_two_terms : Prop := seq 1 + seq 2 = 7
def sum_first_three_terms : Prop := seq 1 + seq 2 + seq 3 = 17

-- The statement to prove
theorem sum_first_n_terms (n : ℕ) 
  (h1 : first_term) 
  (h2 : sum_first_two_terms) 
  (h3 : sum_first_three_terms) : 
  ∑ k in finset.range n.succ, seq k = (n * (n + 1) * (2 * n + 1)) / 6 + n := 
sorry

end sum_first_n_terms_l366_366466


namespace adjacent_numbers_share_digit_l366_366731

theorem adjacent_numbers_share_digit (n : ℕ) :
  ∃ (a b : ℕ), (a, b) ∈ (list.range 20).perm ((list.range 20).map (λ k, n + k)) ∧
  (∀ x ∈ nat.digits 10 a, x ∈ nat.digits 10 b) :=
by sorry

end adjacent_numbers_share_digit_l366_366731


namespace num_positive_divisors_36_l366_366095

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366095


namespace num_pos_divisors_36_l366_366104

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366104


namespace number_of_divisors_36_l366_366056

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366056


namespace lcm_18_24_l366_366747

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366747


namespace find_a_if_perpendicular_l366_366155

-- Define the conditions for the lines being perpendicular
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + a * (a + 1) * y + (a^2 - 1) = 0

def perpendicular_lines (a : ℝ) : Prop :=
  let c1 := a in
  let c2 := 1 in
  let c3 := a + 1 in
  c1 + 2 * c3 = 0 

theorem find_a_if_perpendicular : ∀ a : ℝ, perpendicular_lines a → (a = 0 ∨ a = -3/2) :=
by sorry

end find_a_if_perpendicular_l366_366155


namespace f_continuous_f_strictly_monotone_l366_366621

noncomputable def f : ℝ → ℝ := sorry

axiom has_limits_at_any_point (a : ℝ) : ∃ L : ℝ, tendsto f (𝓝 a) (𝓝 L)
axiom no_local_extrema : ∀ a : ℝ, ¬∃ δ > 0, ∀ x ∈ ball a δ, (f a < f x ∨ f a > f x)

theorem f_continuous : ∀ a : ℝ, continuous_at f a := sorry

theorem f_strictly_monotone : strict_mono f ∨ strict_anti f := sorry

end f_continuous_f_strictly_monotone_l366_366621


namespace f_of_x_l366_366506

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_x (x : ℝ) (h : ∀ (x : ℝ), f(2^x) = x + 1) : f(x) = log x / log 2 + 1 := 
by 
  sorry

end f_of_x_l366_366506


namespace decreasing_on_interval_l366_366583

noncomputable def f (x m : ℝ) : ℝ := - (1 / 3) * x^3 + m * x

theorem decreasing_on_interval (m : ℝ) : 
  (∀ x > 1, f' x m ≤ 0) → m ≤ 1 :=
by
  sorry

-- Helper definition for the derivative:
noncomputable def f' (x m : ℝ) : ℝ := - x^2 + m

end decreasing_on_interval_l366_366583


namespace ral_age_is_26_l366_366276

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l366_366276


namespace find_savings_l366_366837

-- Definitions and conditions
def ratio_income_expenditure : ℝ := 5 / 4
def income : ℝ := 17000
def expenditure (x : ℝ) : ℝ := (income / 5) * 4
def savings (x : ℝ) : ℝ := income - expenditure x

-- Main statement to prove
theorem find_savings (x : ℝ) (h1 : ratio_income_expenditure = 5 / 4) (h2 : income = 17000) :
  savings x = 3400 := by
  sorry

end find_savings_l366_366837


namespace sequence_a_n_l366_366608

noncomputable def a : ℕ → ℝ
| 1       := 2
| (n + 1) := a n + real.log(1 + 1/n)

theorem sequence_a_n (n : ℕ) : a n = 2 + real.log n := sorry

end sequence_a_n_l366_366608


namespace overlapping_wallpaper_area_l366_366726

theorem overlapping_wallpaper_area (total_area : ℕ) (double_layered_area : ℕ) (triple_layered_area : ℕ) : 
  total_area = 300 → double_layered_area = 30 → triple_layered_area = 45 → 
  let double_layered_contribution := 2 * double_layered_area in
  let triple_layered_contribution := 3 * triple_layered_area in
  let extra_counted_double := double_layered_contribution - double_layered_area in
  let extra_counted_triple := triple_layered_contribution - triple_layered_area in
  let A := total_area - extra_counted_double - extra_counted_triple in
  A = 180 :=
by {
  intros,
  simp only *,
  sorry
}

end overlapping_wallpaper_area_l366_366726


namespace quadratic_inequality_l366_366544

variable {a x₁ x₂ : ℝ}

def f (x : ℝ) : ℝ := a * x ^ 2 + 2 * a * x + 4

theorem quadratic_inequality (h₀ : 0 < a) (h₁ : a < 3) (h₂ : x₁ < x₂) (h₃ : x₁ + x₂ = 1 - a) : f x₁ < f x₂ :=
sorry

end quadratic_inequality_l366_366544


namespace mean_greater_than_median_l366_366944

theorem mean_greater_than_median (x : ℕ) (h : 0 < x) : 
  let s := [x, x + 2, x + 4, x + 7, x + 32] in
  let median := s.nthLe (s.length / 2) sorry in
  let mean := s.sum / s.length in
  mean = median + 5 := 
by
  sorry

end mean_greater_than_median_l366_366944


namespace number_of_divisors_of_36_l366_366113

theorem number_of_divisors_of_36 : (nat.divisors 36).card = 9 := by
  sorry

end number_of_divisors_of_36_l366_366113


namespace num_pos_divisors_36_l366_366066

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366066


namespace fraction_relationship_l366_366145

theorem fraction_relationship (a b c : ℚ)
  (h1 : a / b = 3 / 5)
  (h2 : b / c = 2 / 7) :
  c / a = 35 / 6 :=
by
  sorry

end fraction_relationship_l366_366145


namespace B_investment_time_l366_366876

theorem B_investment_time (x : ℝ) (m : ℝ) :
  let A_share := x * 12
  let B_share := 2 * x * (12 - m)
  let C_share := 3 * x * 4
  let total_gain := 18600
  let A_gain := 6200
  let ratio := A_gain / total_gain
  ratio = 1 / 3 →
  A_share = 1 / 3 * (A_share + B_share + C_share) →
  m = 6 := by
sorry

end B_investment_time_l366_366876


namespace gcd_odd_composite_within_20_l366_366742

def is_odd (n : ℕ) : Prop := ¬(even n)
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n
def both_odd_and_composite (n : ℕ) : Prop := n ≤ 20 ∧ is_odd n ∧ is_composite n

theorem gcd_odd_composite_within_20 : gcd 9 15 = 3 :=
by
  sorry

end gcd_odd_composite_within_20_l366_366742


namespace product_of_y_coordinates_l366_366262

theorem product_of_y_coordinates : 
  let Q := λ y : ℝ, (1, y)
  ∃ (y1 y2 : ℝ), ((dist (Q y1) (-4, -3) = 8) ∧ (dist (Q y2) (-4, -3) = 8)) ∧ 
  y1 * y2 = -30 := by
sorry

end product_of_y_coordinates_l366_366262


namespace eccentricity_range_correct_l366_366968

noncomputable def eccentricity_range (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (orthogonal_condition : ℝ) (theta : ℝ) (theta_range : 0 < theta ∧ theta < π / 12) : Set ℝ :=
{e | 1 < e ∧ e < sqrt 2}

theorem eccentricity_range_correct (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
  (orthogonal_condition : ℝ) (theta : ℝ) (theta_range : 0 < theta ∧ theta < π / 12) :
  eccentricity_range a b a_pos b_pos orthogonal_condition theta theta_range = {e | 1 < e ∧ e < sqrt 2} :=
sorry

end eccentricity_range_correct_l366_366968


namespace scheduling_methods_count_l366_366335

theorem scheduling_methods_count :
  let volunteers := ['A', 'B', 'C', 'D']
  let days := 7
  let chosen_days := 4
  let a_before_b (sch : List Char) := sch.indexOf 'A' < sch.indexOf 'B'
  let valid_schedules := { sch : List Char | sch.permutations ∧ (∀ v ∈ set volunteers, v ∈ sch) }
  (filter a_before_b valid_schedules).card = 420 := sorry

end scheduling_methods_count_l366_366335


namespace lcm_18_24_eq_72_l366_366767

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366767


namespace find_x_l366_366396

theorem find_x (x : ℝ) : (3 / 4 * 1 / 2 * 2 / 5) * x = 765.0000000000001 → x = 5100.000000000001 :=
by
  intro h
  sorry

end find_x_l366_366396


namespace residue_of_neg_1237_mod_37_l366_366474

theorem residue_of_neg_1237_mod_37 : (-1237) % 37 = 21 := 
by
  sorry

end residue_of_neg_1237_mod_37_l366_366474


namespace geometric_sequence_fourth_term_l366_366317

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end geometric_sequence_fourth_term_l366_366317


namespace boxes_containing_neither_l366_366229

theorem boxes_containing_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_cards : ℕ)
  (boxes_with_both : ℕ)
  (h1 : total_boxes = 15)
  (h2 : boxes_with_stickers = 8)
  (h3 : boxes_with_cards = 5)
  (h4 : boxes_with_both = 3) :
  (total_boxes - (boxes_with_stickers + boxes_with_cards - boxes_with_both)) = 5 :=
by
  sorry

end boxes_containing_neither_l366_366229


namespace range_of_a_l366_366515

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + x else x - x^2

theorem range_of_a (a : ℝ) (h : f a > f (2 - a)) : a > 1 := 
sorry

end range_of_a_l366_366515


namespace expression_value_l366_366917

theorem expression_value :
  3003 + (1 / 3) * (3002 + (1 / 6) * (3001 + (1 / 9) * (3000 + ... + (1 / (3 * 1000)) * 3))) = 3002.5 :=
sorry

end expression_value_l366_366917


namespace num_positive_divisors_36_l366_366092

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366092


namespace find_polynomials_l366_366933

theorem find_polynomials (P : ℝ[X]) : 
  (16 * P.comp (X^2)) = (P.comp (2 * X))^2 →
  (P = 0 ∨ ∃ n : ℕ, P = monomial n (2^(4 - 2 * n))) :=
begin
  sorry,
end

end find_polynomials_l366_366933


namespace smallest_k_no_real_roots_l366_366821

theorem smallest_k_no_real_roots :
  let a := 3 * (k : ℤ) - 2 in
  let b := -15 in
  let c := 8 in
  (b ^ 2 - 4 * a * c) < 0 ↔ k ≥ 3 :=
by {
  sorry
}

end smallest_k_no_real_roots_l366_366821


namespace number_of_divisors_36_l366_366057

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366057


namespace multiples_of_15_between_25_and_200_l366_366563

theorem multiples_of_15_between_25_and_200 : 
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  in multiples.length = 12 :=
by
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  show multiples.length = 12
  sorry

end multiples_of_15_between_25_and_200_l366_366563


namespace lcm_18_24_l366_366758
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366758


namespace sum_of_squares_l366_366305

noncomputable def sequence_sum (s : ℕ → ℕ) (n : ℕ) :=
  ∑ i in Finset.range n, s i

noncomputable def sequence_sum_squared (s : ℕ → ℕ) (n : ℕ) :=
  ∑ i in Finset.range n, (s i)^2

theorem sum_of_squares (a : ℕ → ℕ) (n : ℕ)
  (h : sequence_sum a n = 2^n - 1) :
  sequence_sum_squared a n = (4^n - 1) / 3 := 
sorry

end sum_of_squares_l366_366305


namespace inequality_f_l366_366479

def f (n : ℕ) : ℕ := sorry  -- definition omitted

theorem inequality_f (n : ℕ) (hn : n ≥ 1) : 
  f(n + 1) ≤ (f(n) + f(n + 2)) / 2 := 
sorry

end inequality_f_l366_366479


namespace binomial_expansion_sum_of_absolute_values_l366_366951

theorem binomial_expansion_sum_of_absolute_values:
  let a := (1 - 2 * x) ^ 7 in
  let a0 := 1 in
  let a1 := -14 * x in
  let a2 := 84 * x ^ 2 in
  let a3 := -280 * x ^ 3 in
  let a4 := 560 * x ^ 4 in
  let a5 := -672 * x ^ 5 in
  let a6 := 448 * x ^ 6 in
  let a7 := -128 * x ^ 7 in
  (\(|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7|\)) = 2187 :=
by 
  sorry

end binomial_expansion_sum_of_absolute_values_l366_366951


namespace correct_choice_B_l366_366024

theorem correct_choice_B (a : ℝ) :
  (x^2 + y^2 - 4 * x + 3 = 0) →
  (∀ {O1 : ℝ × ℝ} {r1 : ℝ}, O1 = (2, 0) ∧ r1 = 1) →
  (x^2 + y^2 - 4 * x - 6 * y + a = 0) →
  (∀ {O2 : ℝ × ℝ} {r2 : ℝ}, O2 = (2, 3) ∧ r2 = Real.sqrt (13 - a)) →
  (distance (2, 0) (2, 3) = 1 + Real.sqrt (13 - a)) →
  a = 9 :=
by
  intros hM hCenterRadiusM hCircleA hCenterRadiusA hDistance
  sorry

end correct_choice_B_l366_366024


namespace dad_caught_more_trouts_l366_366443

-- Definitions based on conditions
def caleb_trouts : ℕ := 2
def dad_trouts : ℕ := 3 * caleb_trouts

-- The proof problem: proving dad caught 4 more trouts than Caleb
theorem dad_caught_more_trouts : dad_trouts = caleb_trouts + 4 :=
by
  sorry

end dad_caught_more_trouts_l366_366443


namespace probability_at_least_one_passes_l366_366528

theorem probability_at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 :=
by sorry

end probability_at_least_one_passes_l366_366528


namespace number_of_true_propositions_is_three_l366_366882

-- Definitions of the conditions
def inverse_proposition (x y : ℝ) : Prop := x + y = 0 → (x = -y)
def negation_congruent_triangles_areas : Prop := ¬ (∀ (Δ1 Δ2 : Triangle), congruent Δ1 Δ2 → area Δ1 = area Δ2)
def proposition_q (q : ℝ) : Prop := q ≤ 1 → (∃ (x : ℝ), (x^2 + 2*x + q = 0))
def contrapositive_equilateral_triangle : Prop := ∀ (T : Triangle), (¬ (angles T).all_equal) → ¬ equilateral T

-- Main proof problem
theorem number_of_true_propositions_is_three :
  (∀ (x y : ℝ), inverse_proposition x y) ∧
  (negation_congruent_triangles_areas = false) ∧
  (∀ (q : ℝ), proposition_q q) ∧
  (contrapositive_equilateral_triangle) →
  (3 = 3) := by
  sorry

end number_of_true_propositions_is_three_l366_366882


namespace sum_of_valid_numbers_is_291_l366_366900

-- Define the range of two-digit numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n < 100}

-- Define the predicate for being greater than 20 but less than 80
def in_range (n : ℕ) := 20 < n ∧ n < 80

-- Define the predicate for being prime
def is_prime (n : ℕ) := Nat.Prime n

-- Define the function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) := (n % 10) * 10 + (n / 10)

-- Define the predicate for being prime when digits are reversed
def is_prime_when_reversed (n : ℕ) := is_prime (reverse_digits n)

-- Collect all numbers satisfying the conditions
def valid_numbers := {n : ℕ | two_digit_numbers n ∧ in_range n ∧ is_prime n ∧ is_prime_when_reversed n}

-- Calculate the sum of these numbers
def sum_valid_numbers := (Finset.filter valid_numbers (Finset.range 100)).sum id

theorem sum_of_valid_numbers_is_291 : sum_valid_numbers = 291 := by
  sorry

end sum_of_valid_numbers_is_291_l366_366900


namespace negation_equivalence_l366_366707

variables (x : ℝ)

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

def has_rational_square (x : ℝ) : Prop := ∃ (q : ℚ), ↑q * ↑q = x * x

def proposition := ∃ (x : ℝ), is_irrational x ∧ has_rational_square x

theorem negation_equivalence :
  (¬ proposition) ↔ ∀ (x : ℝ), is_irrational x → ¬ has_rational_square x :=
by sorry

end negation_equivalence_l366_366707


namespace find_parabola_coefficients_l366_366302

theorem find_parabola_coefficients : 
  ∃ (a b c : ℚ), 
  (∀ x : ℚ, (a * x ^ 2 + b * x + c) = (a * (x - 4) ^ 2 - 1)) ∧ 
  (∀ x : ℚ, (a * x ^ 2 + b * x + c) = (a * (x - 4) ^ 2 - 1) ⟹ 
    (a * 0 ^ 2 + b * 0 + c = -5)) ∧ 
  (∀ x : ℚ, (a * x ^ 2 + b * x + c) = (a * (x - 4) ^ 2 - 1) ⟹ 
    (-16 * a - 1 = -5)) ∧ 
  a = -1 / 4 ∧ 
  b = 2 ∧ 
  c = -5 := 
sorry

end find_parabola_coefficients_l366_366302


namespace number_of_divisors_36_l366_366055

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366055


namespace amount_after_two_years_l366_366928

noncomputable def annual_increase (initial_amount : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + rate) ^ years

theorem amount_after_two_years :
  annual_increase 32000 (1/8) 2 = 40500 :=
by
  sorry

end amount_after_two_years_l366_366928


namespace red_section_not_damaged_l366_366459

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l366_366459


namespace planar_vectors_solution_l366_366547

variables {R : Type*} [inner_product_space ℝ R]

theorem planar_vectors_solution
  (a b c : R)
  (x y : ℝ)
  (h₁ : ⟪a, b⟫ = 0)
  (h₂ : c = x • a + y • b)
  (h₃ : ⟪a, c⟫ > 0)
  (h₄ : ⟪b, c⟫ < 0) :
  x > 0 ∧ y < 0 :=
sorry

end planar_vectors_solution_l366_366547


namespace sin_theta_minus_pi_over_6_tan_theta_plus_pi_over_4_l366_366501

variable (θ : ℝ)
-- condition: Given cos θ = 12/13 and θ in (π, 2π)
def theta_condition : Prop := cos θ = 12 / 13 ∧ θ > π ∧ θ < 2 * π

theorem sin_theta_minus_pi_over_6 (h : theta_condition θ) : sin (θ - π / 6) = -(5 * Real.sqrt 3 + 12) / 26 :=
by
  sorry

theorem tan_theta_plus_pi_over_4 (h : theta_condition θ) : tan (θ + π / 4) = 7 / 17 :=
by
  sorry

end sin_theta_minus_pi_over_6_tan_theta_plus_pi_over_4_l366_366501


namespace michaels_brother_money_end_l366_366656

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l366_366656


namespace round_to_hundredth_l366_366676

theorem round_to_hundredth (x : ℝ) (h1 : x = 3.456) (h2 : (x * 100) % 10 = 5) (h3 : (x * 1000) % 10 = 6) : (Real.round (x * 100) / 100) = 3.46 :=
by
  sorry

end round_to_hundredth_l366_366676


namespace probability_A_in_swimming_pool_l366_366690

theorem probability_A_in_swimming_pool :
  let venues := ["gymnasium", "swimming_pool", "training_hall"],
      volunteers := ["A", "B", "C", "D", "E"] in
  let total_ways := 
    (3 * 6) -- 1, 1, 3 distribution: choose 1 venue for 3, and assign others
    + (3 * 3 * 2) -- 2, 2, 1 distribution: choose 1 venue for 1, assign others
  in let favorable_ways := 
    (3 * 2) -- A and B in group of 3 in swimming pool
    + (3 * 2) -- A and B in one of the groups of 2 in swimming pool
  in let probability := (favorable_ways : ℚ) / total_ways
  in probability = 1 / 3 := sorry

end probability_A_in_swimming_pool_l366_366690


namespace min_absolute_sum_l366_366627

theorem min_absolute_sum (a b c d : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h5 : (⟨a, b⟩ : Matrix (Fin 2) (Fin 2) ℤ) • (⟨a, b⟩ : Matrix (Fin 2) (Fin 2) ℤ) = ⟨9, 0⟩ : Matrix (Fin 2) (Fin 2) ℤ)) :
  ∃ (a b c d : ℤ), |a| + |b| + |c| + |d| = 8 :=
by
  sorry

end min_absolute_sum_l366_366627


namespace solve_system_l366_366292

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l366_366292


namespace joe_new_average_l366_366612

def joe_tests_average (a b c d : ℝ) : Prop :=
  ((a + b + c + d) / 4 = 35) ∧ (min a (min b (min c d)) = 20)

theorem joe_new_average (a b c d : ℝ) (h : joe_tests_average a b c d) :
  ((a + b + c + d - min a (min b (min c d))) / 3 = 40) :=
sorry

end joe_new_average_l366_366612


namespace gcd_polynomials_l366_366985

-- State the problem in Lean 4.
theorem gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * 2 * k) : 
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 :=
by
  sorry

end gcd_polynomials_l366_366985


namespace perimeter_midsegment_l366_366546

-- Variables
variables (a b c : ℝ)

-- Definition of lengths of sides of the original triangle
def lengths (a b c : ℝ) : Prop :=
  a = 8 ∧ b = 10 ∧ c = 12

-- Definition of the perimeter of midsegment triangle
def perimeter_midsegment_triangle (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

-- Theorem statement
theorem perimeter_midsegment (a b c : ℝ) (h : lengths a b c) :
  perimeter_midsegment_triangle a b c = 15 :=
by
  sorry

end perimeter_midsegment_l366_366546


namespace num_pos_divisors_36_l366_366138

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366138


namespace square_position_2010th_l366_366698

def initial_square_position := "ABCD"
def rotate_180 (s : String) : String :=
  match s with
  | "ABCD" => "CDAB"
  | "CDAB" => "DABC"
  | "DABC" => "BADC"
  | "BADC" => "DCBA"
  | "DCBA" => "ABCD"
  | _ => s

def reflect_vertical (s : String) : String :=
  match s with
  | "CDAB" => "DABC"
  | "DABC" => "BADC"
  | "BADC" => "DCBA"
  | "DCBA" => "ABCD"
  | _ => s

theorem square_position_2010th :
  let transformation_sequence := ["ABCD", "CDAB", "DABC", "BADC", "DCBA"]
  let cycle_index := (2010 % 4).nat_abs
  transformation_sequence.nth cycle_index = some "DABC" :=
  sorry

end square_position_2010th_l366_366698


namespace minimal_perimeter_triangle_l366_366916

theorem minimal_perimeter_triangle {A r : ℝ} (hA : 0 < A ∧ A < 180) :
  ∃ (B C : ℝ), B = C ∧ B = (180 - A) / 2 ∧ C = (180 - A) / 2 ∧ ∀ (B' C' : ℝ), let a := r * (Mathlib.Trigonometry.cot (A / 2) + Mathlib.Trigonometry.cot (B' / 2) + Mathlib.Trigonometry.cot (C' / 2)) in 
  0 < B' ∧ 0 < C' ∧ B' + C' + A = 180 → a ≥ r * (Mathlib.Trigonometry.cot (A / 2) + Mathlib.Trigonometry.cot (B / 2) + Mathlib.Trigonometry.cot (C / 2)) :=
by
  sorry

end minimal_perimeter_triangle_l366_366916


namespace set_equality_l366_366549

def P : Set ℝ := { x | x^2 = 1 }

theorem set_equality : P = {-1, 1} :=
by
  sorry

end set_equality_l366_366549


namespace rectangle_area_l366_366209

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l366_366209


namespace total_hours_correct_l366_366254

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l366_366254


namespace find_k_l366_366244

variables {k' : ℝ} {A B C D O : ℝ^3}
variables (vA vB vC vD vO : ℝ^3)
variables {coplanar : ℝ^3 → Prop}

-- Define the vectors from O to other points
def OA := vA - vO
def OB := vB - vO
def OC := vC - vO
def OD := vD - vO

-- The given vector equation condition
def vector_eq : Prop := 4 • OA - 3 • OB + 6 • OC + k' • OD = 0

-- The coplanar condition
def coplanar_points : Prop := coplanar {vA, vB, vC, vD}

-- Proof statement to show k' = -7
theorem find_k' (vector_eq : vector_eq) (coplanar_points : coplanar_points) : k' = -7 :=
sorry

end find_k_l366_366244


namespace round_to_hundredth_l366_366675

theorem round_to_hundredth (x : ℝ) (h1 : x = 3.456) (h2 : (x * 100) % 10 = 5) (h3 : (x * 1000) % 10 = 6) : (Real.round (x * 100) / 100) = 3.46 :=
by
  sorry

end round_to_hundredth_l366_366675


namespace complex_number_sum_l366_366537

noncomputable def x : ℝ := 3 / 5
noncomputable def y : ℝ := -3 / 5

theorem complex_number_sum :
  (x + y) = -2 / 5 := 
by
  sorry

end complex_number_sum_l366_366537


namespace cos_theta_correct_l366_366626

-- Define the direction vector of the line
def direction_vector : ℝ^3 := ⟨4, 5, 8⟩

-- Define the normal vector of the plane
def normal_vector : ℝ^3 := ⟨8, 6, -9⟩

-- Define the dot product function.
noncomputable def dot_product (v w : ℝ^3) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the magnitude function.
noncomputable def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the cosine of theta.
noncomputable def cos_theta : ℝ := -10 / (real.sqrt 105 * real.sqrt 181)

-- The statement to prove
theorem cos_theta_correct : 
  let θ := (dot_product direction_vector normal_vector) / ((magnitude direction_vector) * (magnitude normal_vector)) in
  θ = cos_theta := 
by sorry

end cos_theta_correct_l366_366626


namespace digit_arrangement_count_l366_366175

theorem digit_arrangement_count :
  (∃ digits : Finset ℕ, digits = {2, 2, 4, 8, 0} ∧ 
  (∀ n : ℕ, n ∈ digits → n <= 9)) ∧
  (∀ num : ℕ, num ∈ digits → num ≠ 0 ∧ num ≠ 2) →
  num_valid_permutations digits = 24 :=
by
  sorry

end digit_arrangement_count_l366_366175


namespace sqrt_factorial_expression_l366_366371

theorem sqrt_factorial_expression :
  (sqrt ((5.factorial + 1) * 4.factorial)) ^ 2 = 2904 :=
by
  -- the calculations and proof go here
  sorry

end sqrt_factorial_expression_l366_366371


namespace find_s_2_l366_366242

def t (x : ℝ) : ℝ := 4 * x - 6
def s (y : ℝ) : ℝ := y^2 + 5 * y - 7

theorem find_s_2 : s 2 = 7 := by
  sorry

end find_s_2_l366_366242


namespace num_pos_divisors_36_l366_366068

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366068


namespace sin_C_given_a_36_find_a_b_given_area_6_l366_366609
  
def triangle_ABC_conditions (A B C : Type) [IsTriangle A B C] (a b c : ℕ) (cos_A : ℚ) :=
  c = 13 ∧ cos_A = 5/13

theorem sin_C_given_a_36 (A B C : Type) [IsTriangle A B C] (a b c : ℕ) (cos_A : ℚ) (sin_C : ℚ) :
  triangle_ABC_conditions A B C a b c cos_A →
  a = 36 →
  sin_C = 1/3 :=
by
  intro h_cond h_a36
  sorry

theorem find_a_b_given_area_6 (A B C : Type) [IsTriangle A B C] (a b c : ℕ) (cos_A : ℚ) (area : ℚ) :
  triangle_ABC_conditions A B C a b c cos_A →
  area = 6 →
  a = 4*sqrt 10 ∧ b = 1 :=
by
  intro h_cond h_area
  sorry

end sin_C_given_a_36_find_a_b_given_area_6_l366_366609


namespace locus_of_midpoint_max_distance_l366_366677

-- Parametric equations of the curve C and the line equation in polar coordinates.
variables {α : ℝ} {ρ θ : ℝ}
variables {x : ℝ} {y : ℝ}

-- Midpoint equation in rectangular coordinates.
def midpoint (α : ℝ) : ℝ × ℝ :=
  let x := 1 + cos α in
  let y := sin α in
  (x, y)

-- Rectangular coordinate form of the line l.
def line_eqn (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Circle C in rectangular coordinates.
def circle_eqn (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

axiom midpoint_locus_polar : ∀ (α : ℝ), ∃ (θ ρ : ℝ),
  (midpoint α).fst = ρ * cos θ ∧ 
  (midpoint α).snd = ρ * sin θ ∧ 
  ρ = 2 * cos θ

-- Maximum distance from a point on the curve to the line
def max_distance_to_line : ℝ :=
  2 + (3 * real.sqrt 2 / 2)

-- Proof statements
theorem locus_of_midpoint : ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * cos θ := by
  apply midpoint_locus_polar
  sorry

theorem max_distance : ∃ d : ℝ, d = max_distance_to_line := by
  use max_distance_to_line
  sorry

end locus_of_midpoint_max_distance_l366_366677


namespace simplify_expression_l366_366987

theorem simplify_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ :=
sorry

end simplify_expression_l366_366987


namespace centers_in_same_plane_dual_polyhedron_is_regular_l366_366385

-- Definition of Regular Polyhedron
structure RegularPolyhedron (P : Type*) :=
  (centers : P → P)
  (adjacent_centers_connected : Π v₁ v₂ : P, Bool)
  (is_in_one_plane : Π v : P, Bool)

-- Part (a): Centers of faces around any vertex lie in the same plane.
theorem centers_in_same_plane (P : RegularPolyhedron) (v : P) : 
  P.is_in_one_plane v = true := 
by sorry

-- Part (b): Polyhedron constructed by centers of faces is dual and regular.
structure Polyhedron (P : Type*) := 
  (vertices : P → P)
  (edges : Π v₁ v₂ : P, Bool)
  (faces : Π v : P, Bool)

theorem dual_polyhedron_is_regular (T : RegularPolyhedron) : 
  ∃ T' : RegularPolyhedron, 
    (Π v₁ v₂ : T'.centers, T'.adjacent_centers_connected v₁ v₂ = T.adjacent_centers_connected v₁ v₂) ∧
    (Π v : T'.centers, T'.is_in_one_plane v = T.is_in_one_plane v) :=
by sorry

end centers_in_same_plane_dual_polyhedron_is_regular_l366_366385


namespace tank_filling_time_with_leaks_l366_366417

theorem tank_filling_time_with_leaks (pump_time : ℝ) (leak1_time : ℝ) (leak2_time : ℝ) (leak3_time : ℝ) (fill_time : ℝ)
  (h1 : pump_time = 2)
  (h2 : fill_time = 3)
  (h3 : leak1_time = 6)
  (h4 : leak2_time = 8)
  (h5 : leak3_time = 12) :
  fill_time = 8 := 
sorry

end tank_filling_time_with_leaks_l366_366417


namespace A_takes_4_hours_l366_366850

variables (A B C : ℝ)

-- Given conditions
axiom h1 : 1 / B + 1 / C = 1 / 2
axiom h2 : 1 / A + 1 / C = 1 / 2
axiom h3 : B = 4

-- What we need to prove: A = 4
theorem A_takes_4_hours :
  A = 4 := by
  sorry

end A_takes_4_hours_l366_366850


namespace sum_of_primes_no_solution_congruence_eq_seven_l366_366915

theorem sum_of_primes_no_solution_congruence_eq_seven :
  (∑ q in {p : ℕ | p.prime ∧ ¬ ∃ x : ℤ, 5 * (8 * x + 2) ≡ 7 [ZMOD p]}, q) = 7 :=
sorry

end sum_of_primes_no_solution_congruence_eq_seven_l366_366915


namespace incorrect_correlation_coefficient_range_l366_366432

noncomputable def regression_analysis_conditions 
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) : Prop :=
  non_deterministic_relationship ∧
  correlation_coefficient_range ∧
  perfect_correlation ∧
  correlation_coefficient_sign

theorem incorrect_correlation_coefficient_range
  (non_deterministic_relationship : Prop)
  (correlation_coefficient_range : Prop)
  (perfect_correlation : Prop)
  (correlation_coefficient_sign : Prop) :
  regression_analysis_conditions 
    non_deterministic_relationship 
    correlation_coefficient_range 
    perfect_correlation 
    correlation_coefficient_sign →
  ¬ correlation_coefficient_range :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end incorrect_correlation_coefficient_range_l366_366432


namespace lcm_18_24_eq_72_l366_366809

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366809


namespace area_of_ABCD_l366_366186

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366186


namespace least_pos_int_with_2310_divisors_has_property_m_plus_k_eq_10_l366_366325

theorem least_pos_int_with_2310_divisors_has_property_m_plus_k_eq_10 :
  ∃ (m k : ℕ), (∀ n : ℕ, (number_of_divisors n = 2310) → 
  (∃ m k : ℕ, (n = m * 10^k) ∧ (10 ∣ m → False) ∧ (m + k = 10))) :=
sorry

end least_pos_int_with_2310_divisors_has_property_m_plus_k_eq_10_l366_366325


namespace lcm_18_24_l366_366807

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366807


namespace max_cubes_in_box_l366_366819

theorem max_cubes_in_box :
  let volume_of_cube := 27 -- volume of each small cube in cubic centimetres
  let dimensions_of_box := (8, 9, 12) -- dimensions of the box in centimetres
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3 -- volume of the box
  volume_of_box / volume_of_cube = 32 := 
by
  let volume_of_cube := 27
  let dimensions_of_box := (8, 9, 12)
  let volume_of_box := dimensions_of_box.1 * dimensions_of_box.2 * dimensions_of_box.3
  show volume_of_box / volume_of_cube = 32
  sorry

end max_cubes_in_box_l366_366819


namespace count_irrationals_l366_366220

theorem count_irrationals :
  let numbers := [1 / 6, Real.sqrt 5, 0, Real.cbrt 9, -Real.pi / 3] in
  (numbers.filter (fun x => ¬x.is_rat)).length = 3 :=
by
  sorry

end count_irrationals_l366_366220


namespace limit_fraction_exponential_l366_366440

-- Lean statement translating the proof problem
theorem limit_fraction_exponential :
  tendsto (λ n : ℕ, (2^(n+1) + 3^(n+1)) / (2^n + 3^n)) at_top (𝓝 3) :=
sorry

end limit_fraction_exponential_l366_366440


namespace circumscribed_circles_intersect_at_one_point_l366_366517

theorem circumscribed_circles_intersect_at_one_point
  (A B C D E F : Type*)
  [Geometry A B C D E F]
  (quad : Quadrilateral A B C D)
  (H1 : MeetAt AD BC E)
  (H2 : MeetAt AB DC F) :
  ∃ M, Circumcircle A B E M ∧ Circumcircle A D F M ∧ Circumcircle D C E M ∧ Circumcircle B C F M := by
  -- proof goes here
  sorry

end circumscribed_circles_intersect_at_one_point_l366_366517


namespace equation_of_line_l_distance_AB_is_correct_l366_366011

section MathProof

/- Define the lines and curve -/
def line1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) := x - y + 4 = 0
def line3 (x y : ℝ) := x - 2 * y - 1 = 0
def curve (x y : ℝ) := y^2 + 2 * x = 0

/- Define point P, intersection of line1 and line2 -/
def P : ℝ × ℝ := (-2, 2)

/- Define line l passing through P and perpendicular to line3 -/
def line_l (x y : ℝ) := 2 * x + y + 2 = 0

/- Define points A and B where line l intersects with the curve -/
def A : ℝ × ℝ := (-1/2, -1)
def B : ℝ × ℝ := (-2, 2)

/- Euclidean distance between two points (x1, y1) and (x2, y2) -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

/- Proof statements -/
theorem equation_of_line_l : ∀ x y : ℝ, line_l x y ↔ 2 * x + y + 2 = 0 := sorry

theorem distance_AB_is_correct : distance A B = 3 * (5:ℝ).sqrt / 2 := sorry

end MathProof

end equation_of_line_l_distance_AB_is_correct_l366_366011


namespace double_sum_evaluation_l366_366926

theorem double_sum_evaluation : 
  (∑ m in (Finset.range (Nat.succ m)).filter (λ m, m > 0),
    ∑ n in (Finset.range (Nat.succ n)).filter (λ n, n > 0), 
      (1 : ℝ) / (↑m * ↑n * (↑m + ↑n + 1))) = 2 := 
by sorry

end double_sum_evaluation_l366_366926


namespace equal_chords_of_parabola_l366_366010

theorem equal_chords_of_parabola
  (P Q N M : ℝ × ℝ)
  (p : ℝ)
  (h1 : P.2 ^ 2 = 2 * p * P.1)
  (h2 : Q.2 ^ 2 = 2 * p * Q.1)
  (h3 : N = (0, -p))
  (h4 : P.1 < Q.1)
  (h5 : P.2 = -Q.2)
  (h6 : ∀ x, x ∈ Segment P Q → ∃ N, is_perpendicular (x, N) N)
  (h7 : ∀ y, y ∈ Segment P M → y.1 = P.1) : abs (P.2) = abs (Q.2) :=
by sorry

end equal_chords_of_parabola_l366_366010


namespace logarithm_properties_l366_366394
  
theorem logarithm_properties : (1 / 4) ^ (-2) + (1 / 2) * log 3 6 - log 3 (sqrt 2) = 33 / 2 := 
by
  sorry

end logarithm_properties_l366_366394


namespace ellipse_related_problems_l366_366538

noncomputable def ellipse : Type :=
  {a b : ℝ // a > b ∧ b > 0}

noncomputable def related_circle (C : ellipse) : ℝ × ℝ → Prop :=
  λ (p : ℝ × ℝ), (p.1)^2 + (p.2)^2 = C.val.1^2*C.val.2^2 / (C.val.1^2 + C.val.2^2)

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

theorem ellipse_related_problems (C : ellipse)
  (H_focus_ellipse : (0, 1) = (C.val.1^2 - C.val.2^2).sqrt)
  (H_right_triangle : C.val.2 = (C.val.1^2 - 1).sqrt) :
  (∀ p : ℝ × ℝ, related_circle C p → 
  (∃ m k : ℝ, p.2 = k * p.1 + m ∧ 
    distance (0, 0) 
    (line (p.1 :: _ :: _ :: p.2 :: list.nil)) = (real.sqrt(6)) / 3))
  ∧
  ∀ m : ℝ, m^2 ≥ 0 → 
  ¬(-((real.sqrt 6) / 3) ≤ m ∧ m ≤ (real.sqrt 6) / 3)
  :=
sorry

end ellipse_related_problems_l366_366538


namespace jesse_blocks_total_l366_366230

-- Define the number of building blocks used for each structure and the remaining blocks
def blocks_building : ℕ := 80
def blocks_farmhouse : ℕ := 123
def blocks_fenced_in_area : ℕ := 57
def blocks_left : ℕ := 84

-- Prove that the total number of building blocks Jesse started with is 344
theorem jesse_blocks_total : blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left = 344 :=
by
  calc
    blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left
      = 80 + 123 + 57 + 84 : by refl
  ... = 260 + 84 : by simp
  ... = 344 : by norm_num

end jesse_blocks_total_l366_366230


namespace complete_square_example_l366_366823

theorem complete_square_example :
  ∃ c : ℝ, ∃ d : ℝ, (∀ x : ℝ, x^2 + 12 * x + 4 = (x + c)^2 - d) ∧ d = 32 := by
  sorry

end complete_square_example_l366_366823


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366363

theorem remainder_3001_3002_3003_3004_3005_mod_17 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := 
begin
  sorry
end

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366363


namespace standard_equation_of_circle_l366_366988

theorem standard_equation_of_circle
  (r : ℝ) (h_radius : r = 1)
  (h_center : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x, y) = (a, b))
  (h_tangent_line : ∃ (a : ℝ), 1 = |4 * a - 3| / 5)
  (h_tangent_x_axis : ∃ (a : ℝ), a = 1) :
  (∃ (a b : ℝ), (x-2)^2 + (y-1)^2 = 1) :=
sorry

end standard_equation_of_circle_l366_366988


namespace number_of_sequences_l366_366333

theorem number_of_sequences (n : ℕ) (h : n ≥ 2) : 
  let f : ℕ → ℕ := λ n, 2^(n-1)
  in f n = 2^(n-1) := sorry

end number_of_sequences_l366_366333


namespace y_axis_intersection_at_l366_366411

noncomputable def slope (p1 p2 : (ℝ × ℝ)) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def y_intercept (p1 p2 : (ℝ × ℝ)) : ℝ :=
  p1.2 - slope p1 p2 * p1.1

theorem y_axis_intersection_at (p1 p2 : (ℝ × ℝ)) (hp1 : p1 = (4, 20)) (hp2 : p2 = (-6, -2)) :
  y_intercept p1 p2 = 11.2 :=
by
  sorry

end y_axis_intersection_at_l366_366411


namespace lcm_18_24_l366_366752

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l366_366752


namespace lcm_18_24_eq_72_l366_366771

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366771


namespace possible_remainder_degrees_l366_366824

theorem possible_remainder_degrees (f : Polynomial ℝ) :
  ∃ r : Polynomial ℝ, ∃ q : Polynomial ℝ, 
  f = q * (polynomial.C 5 * polynomial.X ^ 7 - polynomial.C 4 * polynomial.X ^ 3 + polynomial.C 9) + r ∧
  r.degree < 7 :=
begin
  sorry
end

end possible_remainder_degrees_l366_366824


namespace right_triangle_properties_l366_366601

def hypotenuse_and_area (a b : ℕ) (is_right_triangle : a ≠ 0 ∧ b ≠ 0) :
  (hypotenuse : ℕ) × (area : ℕ) := (Math.sqrt (a^2 + b^2), (1 / 2) * a * b)

theorem right_triangle_properties :
  hypotenuse_and_area 30 40 (and.intro (by norm_num) (by norm_num)) = (50, 600) :=
by
  let ⟨h, A⟩ := hypotenuse_and_area 30 40 (and.intro (by norm_num) (by norm_num))
  dsimp only [hypotenuse_and_area] at h A
  norm_cast
  norm_num
  sorry

end right_triangle_properties_l366_366601


namespace number_of_elements_in_list_l366_366048

-- Define the arithmetic sequence
def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  ∀ n : ℕ, n < seq.length - 1 → seq[n + 1] - seq[n] = -5

-- Define the given list
def given_list := [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45]

-- The problem statement
theorem number_of_elements_in_list : 
    is_arithmetic_sequence given_list → 
    given_list.length = 25 :=
by
  sorry

end number_of_elements_in_list_l366_366048


namespace line_passes_quadrants_l366_366007

theorem line_passes_quadrants (a b c : ℝ) (h1 : a * b < 0) (h2 : b * c < 0) :
  ∃ q1 q2 q3, q1 ∈ {1, 2, 3, 4} ∧ q2 ∈ {1, 2, 3, 4} ∧ q3 ∈ {1, 2, 3, 4} ∧
    q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧
    line_passes_through_quadrant (λ x y, a * x + b * y + c = 0) q1 ∧
    line_passes_through_quadrant (λ x y, a * x + b * y + c = 0) q2 ∧
    line_passes_through_quadrant (λ x y, a * x + b * y + c = 0) q3 ∧
    {q1, q2, q3} = {1, 2, 3} :=
sorry

end line_passes_quadrants_l366_366007


namespace Emmanuel_jelly_beans_l366_366895

theorem Emmanuel_jelly_beans :
  ∀ (total : ℕ) (thomas_ratio : ℚ) (ratio_barry : ℕ) (ratio_emmanuel : ℕ),
  total = 200 →
  thomas_ratio = 10 / 100 →
  ratio_barry = 4 →
  ratio_emmanuel = 5 →
  let thomas_share := thomas_ratio * total in
  let remaining := total - thomas_share in
  let total_parts := ratio_barry + ratio_emmanuel in
  let part_value := remaining / total_parts in
  let emmanuel_share := ratio_emmanuel * part_value in
  emmanuel_share = 100 :=
begin
  intros total thomas_ratio ratio_barry ratio_emmanuel h_total h_thomas_ratio h_ratio_barry h_ratio_emmanuel,
  simp [h_total, h_thomas_ratio, h_ratio_barry, h_ratio_emmanuel],
  have ht : thomas_share = 20, by norm_num [h_total, h_thomas_ratio, thomas_share],
  have hr : remaining = 180, by norm_num [remaining, h_total, ht],
  have total_parts : total_parts = 9, by norm_num [h_ratio_barry, h_ratio_emmanuel, total_parts],
  have pv : part_value = 20, by norm_num [part_value, hr, total_parts],
  have es : emmanuel_share = 100, by norm_num [emmanuel_share, h_ratio_emmanuel, pv],
  exact es,
end

end Emmanuel_jelly_beans_l366_366895


namespace common_difference_arithmetic_sequence_l366_366526

theorem common_difference_arithmetic_sequence :
  ∃ d : ℝ, (d ≠ 0) ∧ (∀ (n : ℕ), a_n = 1 + (n-1) * d) ∧ ((1 + 2 * d)^2 = 1 * (1 + 8 * d)) → d = 1 :=
by
  sorry

end common_difference_arithmetic_sequence_l366_366526


namespace area_relation_l366_366235

open Real

def isosceles_triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_relation : 
  let A := isosceles_triangle_area 13 13 10
  let B := isosceles_triangle_area 13 13 24
  in A = B := 
by
  sorry 

end area_relation_l366_366235


namespace rectangle_area_l366_366217

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366217


namespace set_property_l366_366665

theorem set_property {n : ℕ} (hn : n > 16) :
  ∃ S : set ℕ, (∃ (hS : fintype S), fintype.card S = n) ∧
  (∀ A ⊆ S, (∀ (a a' : ℕ), a ∈ A → a' ∈ A → a ≠ a' → a + a' ∉ S) → fintype.card A ≤ 4 * real.sqrt n) :=
sorry

end set_property_l366_366665


namespace distance_to_other_focus_l366_366978

variables {x y : ℝ}

def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 25) - (y^2 / 24) = 1

def distance_to_focus (x y : ℝ) (d : ℝ) : Prop :=
  abs (sqrt (x^2 + (y-7)^2) - d) = 10

theorem distance_to_other_focus
  (x y : ℝ)
  (hP : point_on_hyperbola x y)
  (hD : distance_to_focus x y 11) :
  abs (sqrt (x^2 + (y+7)^2) - 21) = 10 :=
sorry

end distance_to_other_focus_l366_366978


namespace root_in_interval_l366_366897

theorem root_in_interval {a b c : ℝ} (h_a : a ≠ 0) :
  let f := λ x, a * x^2 + b * x + c,
      x₁ := 6.17, y₁ := f x₁,
      x₂ := 6.18, y₂ := f x₂,
      x₃ := 6.19, y₃ := f x₃,
      x₄ := 6.20, y₄ := f x₄ in
  y₁ = -0.03 ∧ y₂ = -0.01 ∧ y₃ = 0.02 ∧ y₄ = 0.04 →
  (∃ x : ℝ, 6.18 < x ∧ x < 6.19 ∧ f x = 0) :=
begin
  -- Sorry placeholder for proof
  sorry
end

end root_in_interval_l366_366897


namespace number_of_terms_in_sequence_l366_366043

-- Definition of the arithmetic sequence parameters
def start_term : Int := 165
def end_term : Int := 45
def common_difference : Int := -5

-- Define a theorem to prove the number of terms in the sequence
theorem number_of_terms_in_sequence :
  ∃ n : Nat, number_of_terms 165 45 (-5) = 25 :=
by
  sorry

end number_of_terms_in_sequence_l366_366043


namespace length_of_dividing_curve_l366_366695

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
          [Triangle A B C a : Type] (a : ℝ) (L : Curve A B C)

-- Noncomputable needs
noncomputable def equilateral_triangle (a : ℝ) := True
noncomputable def curve_divides_triangle_equal_area (L : Curve A B C) := True

theorem length_of_dividing_curve (a : ℝ) (L : Curve A B C)
  (triangle : equilateral_triangle a) 
  (divides : curve_divides_triangle_equal_area L) :
  L.length ≥ (Real.sqrt π) / (2 * Real.sqrt (Real.sqrt 3)) * a := sorry

end length_of_dividing_curve_l366_366695


namespace min_n_for_Tn_l366_366018

theorem min_n_for_Tn (n : ℕ) (h : T_n > 2013) : n ≥ 10 :=
    let a : ℕ → ℕ := λ n, 2 * n - 1
    let b : ℕ → ℕ := λ n, 2^(n - 1)
    let T_n : ℕ → ℕ := λ n, ∑ i in range n, a(b(i)) 
begin
    sorry
end

end min_n_for_Tn_l366_366018


namespace slowest_swimmer_is_daughter_l366_366862

-- Definitions for conditions
inductive Swimmer
| man
| sister
| daughter
| son

def twin : Swimmer → Swimmer
| Swimmer.man := Swimmer.sister
| Swimmer.sister := Swimmer.man
| Swimmer.daughter := Swimmer.son
| Swimmer.son := Swimmer.daughter

-- Handedness definition
inductive Handedness
| left
| right

def handedness : Swimmer → Handedness
| Swimmer.man := Handedness.right
| Swimmer.sister := Handedness.right  -- This could vary, just an example
| Swimmer.daughter := Handedness.left -- This could vary, just an example
| Swimmer.son := Handedness.right     -- This could vary, just an example

-- Assumption on fastest and slowest swimmer being of different handedness
axiom different_handedness (s f : Swimmer) : handedness (twin s) ≠ handedness f

-- Assuming same age for fastest and slowest swimmer
axiom same_age (s f : Swimmer) : s ≠ f → twin s = f

-- Theorem to prove
theorem slowest_swimmer_is_daughter : ∃ s : Swimmer, s = Swimmer.daughter ∧ (∀ f : Swimmer, handedness (twin s) ≠ handedness f ∧ (s ≠ f → twin s = f)) :=
by
  sorry

end slowest_swimmer_is_daughter_l366_366862


namespace log_sqrt3_sixth_root_a_eq_inv_b_l366_366151

variable (a b : ℝ)

-- Use the condition given in the problem
axiom log_a_27_eq_b : Real.logBase a 27 = b 

-- Define the problem statement in Lean 4
theorem log_sqrt3_sixth_root_a_eq_inv_b (h : Real.logBase a 27 = b) :
  Real.logBase (Real.sqrt 3) (Real.root a 6) = 1 / b :=
by
  sorry

end log_sqrt3_sixth_root_a_eq_inv_b_l366_366151


namespace cos_B_value_l366_366594

theorem cos_B_value (A B C a b c : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * a * Real.cos B) :
  Real.cos B = Real.sqrt 3 / 3 := by
  sorry

end cos_B_value_l366_366594


namespace total_shapes_proof_l366_366692

def stars := 50
def stripes := 13

def circles : ℕ := (stars / 2) - 3
def squares : ℕ := (2 * stripes) + 6
def triangles : ℕ := (stars - stripes) * 2
def diamonds : ℕ := (stars + stripes) / 4

def total_shapes : ℕ := circles + squares + triangles + diamonds

theorem total_shapes_proof : total_shapes = 143 := by
  sorry

end total_shapes_proof_l366_366692


namespace sum_of_extrema_of_g_l366_366640

/-- The definition of the function g(x) --/
def g (x : ℝ) : ℝ := |x - 5| + |x - 3| - |3 * x - 15|

/-- The sum of the largest and smallest values of g(x) for 3 ≤ x ≤ 10 --/
theorem sum_of_extrema_of_g : 
  (Real.minimum (g '' {x | 3 ≤ x ∧ x ≤ 10}) + Real.maximum (g '' {x | 3 ≤ x ∧ x ≤ 10}) = -2) :=
sorry

end sum_of_extrema_of_g_l366_366640


namespace root_equiv_sum_zero_l366_366955

variable {a b c : ℝ}
variable (h₀ : a ≠ 0)

theorem root_equiv_sum_zero : (1 root_of (a * 1^2 + b * 1 + c = 0)) ↔ (a + b + c = 0) :=
by
  sorry

end root_equiv_sum_zero_l366_366955


namespace tangent_line_equation_l366_366485

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.log (x + 1)

def point_of_tangency : ℝ × ℝ := (0, 0)

theorem tangent_line_equation :
  let y := 2 * x in
  ∃ (m : ℝ) (b : ℝ), m = 2 ∧ b = 0 ∧ y = m * x + b :=
by
  sorry

end tangent_line_equation_l366_366485


namespace exists_factorial_with_first_digits_2015_l366_366922

theorem exists_factorial_with_first_digits_2015 : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2015 * (10^k) ≤ n! ∧ n! < 2016 * (10^k)) :=
sorry

end exists_factorial_with_first_digits_2015_l366_366922


namespace find_lambda_l366_366037

noncomputable def vec_a : ℝ × ℝ := (-1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, 0)

-- Define dot product of two vectors in ℝ^2
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define orthogonality condition
def orthogonal (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

-- Define the given vectors
def a_minus_b : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)
def two_a_plus_lambda_b (λ : ℝ) : ℝ × ℝ := 
  (2 * vec_a.1 + λ * vec_b.1, 2 * vec_a.2 + λ * vec_b.2)

-- The proof problem statement
theorem find_lambda (λ : ℝ) :
  orthogonal a_minus_b (two_a_plus_lambda_b λ) → λ = 3 :=
by
  -- Proof is to be provided here
  sorry

end find_lambda_l366_366037


namespace lcm_18_24_l366_366792

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366792


namespace number_of_combinations_l366_366232

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}

def follows_pattern (combo : List ℕ) : Prop :=
  ∀ (i : ℕ), i < combo.length - 1 →
    (is_odd (combo.nth_le i sorry) → is_even (combo.nth_le (i + 1) sorry)) ∧
    (is_even (combo.nth_le i sorry) → is_odd (combo.nth_le (i + 1) sorry))

def valid_combination (combo : List ℕ) : Prop :=
  combo.length = 6 ∧ ∀ n ∈ combo, valid_digit n ∧ follows_pattern combo

theorem number_of_combinations : ∃ (n : ℕ), n = 1458 ∧ ∀ (combo : List ℕ), valid_combination combo :=
  sorry

end number_of_combinations_l366_366232


namespace eccentricity_range_slope_PQ_l366_366529

-- Define the necessary components
variable {a b : ℝ} (h_ab : a > b > 0)

-- Define the eccentricity
def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 - (b^2 / a^2))

-- Define the ellipse
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Assume points P and Q are on the ellipse and line PQ passes through the left focus
def on_ellipse (P Q : ℝ × ℝ) (a b : ℝ) : Prop := ellipse P.1 P.2 a b ∧ ellipse Q.1 Q.2 a b ∧ P.1 ≠ Q.1

-- Assume existence of point R on the left directrix such that triangle PQR is an equilateral triangle
def exists_point_R (P Q R : ℝ × ℝ) (a b : ℝ) : Prop :=
  let F := -(real.sqrt (a^2 - b^2), 0) in  -- Left focus
  let d1 := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 in
  let d2 := (R.1 - P.1)^2 + (R.2 - P.2)^2 in
  let d3 := (R.1 - Q.1)^2 + (R.2 - Q.2)^2 in
  d1 = d2 ∧ d2 = d3 ∧ d1 ≠ 0  -- Equilateral triangle condition

-- Prove the range of eccentricity e
theorem eccentricity_range (a b : ℝ) (h_ab : a > b > 0) (PQ_not_vertical : ∀ P Q : ℝ × ℝ, P.1 ≠ Q.1 → on_ellipse P Q a b)
  (equilateral_triangle_condition : ∀ P Q R : ℝ × ℝ, exists_point_R P Q R a b) :
  let e := eccentricity a b in e > real.sqrt(3) / 3 ∧ e < 1 :=
sorry

-- Prove the slope of the line PQ in terms of e
theorem slope_PQ (a b : ℝ) (h_ab : a > b > 0) (PQ_not_vertical : ∀ P Q : ℝ × ℝ, P.1 ≠ Q.1 → on_ellipse P Q a b)
  (equilateral_triangle_condition : ∀ P Q R : ℝ × ℝ, exists_point_R P Q R a b) :
  let e := eccentricity a b in
  let m := λ P Q : ℝ × ℝ, (P.2 - Q.2) / (P.1 - Q.1) in
  ∀ P Q : ℝ × ℝ, P.1 ≠ Q.1 → on_ellipse P Q a b → (m P Q = 1 / real.sqrt (3 * e^2 - 1) ∨ m P Q = -1 / real.sqrt (3 * e^2 - 1)) :=
sorry

end eccentricity_range_slope_PQ_l366_366529


namespace lcm_18_24_l366_366757
  
theorem lcm_18_24 : Nat.lcm 18 24 = 72 :=
by
-- Conditions: interpretations of prime factorizations of 18 and 24
have h₁ : 18 = 2 * 3^2 := by norm_num,
have h₂ : 24 = 2^3 * 3 := by norm_num,
-- Completing proof section
sorry -- skipping proof steps

end lcm_18_24_l366_366757


namespace cryptarithm_solution_exists_l366_366285

theorem cryptarithm_solution_exists :
  ∃ (L E S O : ℕ), L ≠ E ∧ L ≠ S ∧ L ≠ O ∧ E ≠ S ∧ E ≠ O ∧ S ≠ O ∧
  (L < 10) ∧ (E < 10) ∧ (S < 10) ∧ (O < 10) ∧
  (1000 * O + 100 * S + 10 * E + L) +
  (100 * S + 10 * E + L) +
  (10 * E + L) +
  L = 10034 ∧
  ((L = 6 ∧ E = 7 ∧ S = 4 ∧ O = 9) ∨
   (L = 6 ∧ E = 7 ∧ S = 9 ∧ O = 8)) :=
by
  -- The proof is omitted here.
  sorry

end cryptarithm_solution_exists_l366_366285


namespace regular_pentagon_l366_366646

-- Definition for a Convex Pentagon with equal sides
structure ConvexPentagon (A B C D E : Type) :=
(equal_sides : ∀ (a b : Type), a ≠ b → a.length = b.length)
(convex : ∀ (a A' : Type), (a + A') = 180)
(order_of_angles : (∀ {a b c d e : Type}, a.angle ≥ b.angle) ∧ (b.angle ≥ c.angle) ∧ (c.angle ≥ d.angle) ∧ (d.angle ≥ e.angle))

theorem regular_pentagon (ABCDE : ConvexPentagon) : 
  ∃ (A B C D E : Type), A.angle = B.angle ∧ B.angle = C.angle ∧ C.angle = D.angle ∧ D.angle = E.angle :=
sorry

end regular_pentagon_l366_366646


namespace lcm_18_24_eq_72_l366_366770

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366770


namespace number_of_divisors_36_l366_366059

theorem number_of_divisors_36 : Nat.totient 36 = 9 := by
  sorry

end number_of_divisors_36_l366_366059


namespace quentavious_initial_nickels_l366_366269

theorem quentavious_initial_nickels (
  leaves_with_nickels : ℕ,
  pieces_of_gum : ℕ,
  pieces_per_nickel : ℕ)
  (h1 : leaves_with_nickels = 2)
  (h2 : pieces_of_gum = 6)
  (h3 : pieces_per_nickel = 2)
  : leaves_with_nickels + (pieces_of_gum / pieces_per_nickel) = 5 := 
by
  sorry

end quentavious_initial_nickels_l366_366269


namespace probability_density_of_ordinate_y_l366_366165

noncomputable def uniform_density (t : ℝ) : ℝ :=
  if t ∈ Icc (-(Real.pi / 2)) (Real.pi / 2) then (1 / (Real.pi)) else 0

def psi (y : ℝ) : ℝ := Real.arctan (y / 4)

def g (y : ℝ) : ℝ := 4 / (Real.pi * (16 + y^2))

theorem probability_density_of_ordinate_y :
  ∀ y : ℝ, g(y) = (uniform_density (psi y)) * (Real.abs (deriv psi y)) :=
sorry

end probability_density_of_ordinate_y_l366_366165


namespace number_of_divisors_36_l366_366082

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366082


namespace areas_of_quadrilaterals_are_equal_l366_366598

variables {A B C D E F G H W X Y Z : Type} [linear_ordered_field ℝ]
variables (area : Π {A B C D : Type} [linear_ordered_field ℝ], ℝ)

/-- In a cyclic quadrilateral ABCD, let the midpoints of AB, BC, CD, & DA be E, F, G, & H, 
    and the orthocenters of triangles AHE, BEF, CFG, & DGH be W, X, Y, & Z respectively.
    Prove that the areas of quadrilateral ABCD and quadrilateral WXYZ are equal -/
theorem areas_of_quadrilaterals_are_equal
  {A B C D E F G H W X Y Z : Type} [linear_ordered_field ℝ]
  (midpoint_AB : E = midpoint A B)
  (midpoint_BC : F = midpoint B C)
  (midpoint_CD : G = midpoint C D)
  (midpoint_DA : H = midpoint D A)
  (orthocenter_AHE : W = orthocenter A H E)
  (orthocenter_BEF : X = orthocenter B E F)
  (orthocenter_CFG : Y = orthocenter C F G)
  (orthocenter_DGH : Z = orthocenter D G H)
  (cyclic_ABCD : cyclic A B C D)
: area ABCD = area WXYZ := sorry

end areas_of_quadrilaterals_are_equal_l366_366598


namespace infinitely_many_lovely_numbers_no_lovely_number_square_l366_366736

def is_lovely (n : ℕ) : Prop :=
  ∃ k : ℕ, ∃ d : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ k → d i > 0) ∧
    (n = ∏ i in finset.range k, d i) ∧ (∀ i, 1 ≤ i ∧ i ≤ k → (d i)^2 ∣ n + (d i))

theorem infinitely_many_lovely_numbers : ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_lovely n := 
sorry

theorem no_lovely_number_square (n : ℕ) (hn : is_lovely n) : n ≠ x*x ∨ x ≤ 1 :=
sorry

end infinitely_many_lovely_numbers_no_lovely_number_square_l366_366736


namespace area_of_triangle_l366_366025

/-- Given the ellipse equation and points F1, F2, and P with specific properties,
    we want to prove the area of triangle F1 P F2. -/
theorem area_of_triangle {F1 F2 P : ℝ} (a b x y : ℝ)
  (h1 : a = 10) -- semi-major axis (sqrt(100))
  (h2 : b = 8)  -- semi-minor axis (sqrt(64))
  (h3 : x / 10 ^ 2 + y / 8 ^ 2 = 1) -- point P on the ellipse
  (h4 : real.angle F1 P F2 = real.pi / 3) :
  1 / 2 * (20 / 2) ^ 2 * real.sin (real.pi / 3) = 64 * real.sqrt 3 / 3 :=
by sorry

end area_of_triangle_l366_366025


namespace probability_of_at_most_3_tails_in_10_flips_l366_366405

theorem probability_of_at_most_3_tails_in_10_flips :
  ∃ (p : ℚ), p = 11 / 64 ∧
  probability (at_most_k_tails 10 3) = p :=
sorry

def at_most_k_tails (n k : ℕ) : Event :=
  finset.sum (finset.range (k + 1)) (λ i, binomial n i) 

noncomputable instance : ProbabilitySpace :=
  ProbabilitySpace.mk_uniform_of_finite (finset.range (2^10))

def probability (e : Event) : ℚ := e.card / 1024


end probability_of_at_most_3_tails_in_10_flips_l366_366405


namespace slope_of_line_l_intersecting_ellipse_l366_366974

noncomputable def ellipse_equation_proof : Prop :=
  ∃ (a b : ℝ),
    a > b ∧ b > 0 ∧
    let e := (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1 in
    a = sqrt 2 ∧ b = 1 ∧
    (e = (λ (x y : ℝ), x^2 / 2 + y^2 = 1))

theorem slope_of_line_l_intersecting_ellipse : Prop :=
  ∃ (a b : ℝ) (F1 F2 M N : ℝ × ℝ),
    a > b ∧ b > 0 ∧
    let e := (x y : ℝ) → x^2 / a^2 + y^2 / b^2 = 1 in
    let l := (x : ℝ) → (-sqrt 14 / 7) * x + 1 in
    a = sqrt 2 ∧ b = 1 ∧
    slope_of_line_l =
      (λ (P Q : ℝ × ℝ), ∃ (S_F1NQ S_F1MP : ℝ),
        S_F1NQ = 2/3 * S_F1MP → slope_of_line_l = -sqrt 14 / 7)

end slope_of_line_l_intersecting_ellipse_l366_366974


namespace angles_of_triangle_XYZ_acute_l366_366855

-- Let PQR be a triangle
variables (P Q R : Type) [inhabited P] [inhabited Q] [inhabited R]

-- XY is the triangle formed by the tangency points of the incircle within ΔPQR
variables (X Y Z : Type) [inhabited X] [inhabited Y] [inhabited Z]

-- Conditions (the circle touches the sides at the points)
-- Let's consider PQR as angles of the triangle and set:
variables (p q r : ℝ)
-- Note that:
-- p + q + r = 180°

theorem angles_of_triangle_XYZ_acute (hpq : p + q + r = 180) 
: (p/2 + q/2 < 90) ∧ (q/2 + r/2 < 90) ∧ (r/2 + p/2 < 90) := 
sorry

end angles_of_triangle_XYZ_acute_l366_366855


namespace lcm_18_24_l366_366800

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366800


namespace lcm_18_24_eq_72_l366_366815

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366815


namespace rhombus_area_l366_366268

-- Define the parameters given in the problem
namespace MathProof

def perimeter (EFGH : ℝ) : ℝ := 80
def diagonal_EG (EFGH : ℝ) : ℝ := 30

-- Considering the rhombus EFGH with the given perimeter and diagonal
theorem rhombus_area : 
  ∃ (area : ℝ), area = 150 * Real.sqrt 7 ∧ 
  (perimeter EFGH = 80) ∧ 
  (diagonal_EG EFGH = 30) :=
  sorry
end MathProof

end rhombus_area_l366_366268


namespace necessary_but_not_sufficient_condition_l366_366963

variable {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 10
def q : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem necessary_but_not_sufficient_condition (h : ¬p → ¬q) (hm_pos : 0 < m) :
  9 ≤ m := by
  sorry

end necessary_but_not_sufficient_condition_l366_366963


namespace sum_of_union_elements_eq_2a_l366_366013

def A (a : ℤ) : set ℤ := { x | abs (x - a) < a + 1 / 2 }
def B (a : ℤ) : set ℤ := { x | abs x < 2 * a }

theorem sum_of_union_elements_eq_2a (a : ℤ) (ha : 0 < a) :
  (A a ∪ B a).sum = 2 * a :=
sorry

end sum_of_union_elements_eq_2a_l366_366013


namespace female_guests_from_jays_family_l366_366251

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l366_366251


namespace customers_bought_one_melon_l366_366427

theorem customers_bought_one_melon (total_melons : ℕ) (customers_three_melons : ℕ) (customers_two_melons : ℕ) 
 (melons_three_customers : ℕ) (melons_ten_customers : ℕ) 
 (total_customers : ℕ) (customers_one_melon : ℕ) :
 total_melons = 46 →
 customers_three_melons = 3 →
 customers_two_melons = 10 →
 melons_three_customers = 9 →
 melons_ten_customers = 20 →
 total_customers = 46 →
 customers_one_melon = total_customers - melons_three_customers - melons_ten_customers → 
 customers_one_melon = 17 :=
begin
  sorry
end

end customers_bought_one_melon_l366_366427


namespace sets_with_property_P_l366_366521

def property_P (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ) (k : ℝ), (0 < k ∧ k < 1) → ((x, y) ∈ M → (k * x, k * y) ∈ M)

def M1 : Set (ℝ × ℝ) := { p | p.1^2 ≥ p.2 }  -- (x, y) such that x^2 ≥ y

def M2 : Set (ℝ × ℝ) := { p | 2 * p.1^2 + p.2^2 < 1 }  -- (x, y) such that 2x^2 + y^2 < 1

def M3 : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 = 0 }  -- (x, y) such that x^2 + y^2 + 2x + 2y = 0

def M4 : Set (ℝ × ℝ) := { p | p.1^3 + p.2^3 - p.1^2 * p.2 = 0 }  -- (x, y) such that x^3 + y^3 - x^2y = 0

theorem sets_with_property_P :
  (property_P M2) ∧ (property_P M4) ∧ ¬ (property_P M1) ∧ ¬ (property_P M3) :=
by
  -- Proof goes here
  sorry

end sets_with_property_P_l366_366521


namespace tetrahedron_inequality_implies_regular_l366_366733

theorem tetrahedron_inequality_implies_regular
    (A1 A2 A3 A4 P : Type)
    [MetricSpace A1] [MetricSpace A2] [MetricSpace A3] [MetricSpace A4]
    [MetricSpace P]
    (P_in_tetrahedron : ∀ {i j k l : ℕ}, {i, j, k, l} ⊆ {1, 2, 3, 4} → distance P Aᵢ + distance P Aⱼ + distance P Aₖ < distance Aₗ Aᵢ + distance Aₗ Aⱼ + distance Aₗ Aₖ) :
  IsRegularTetrahedron A1 A2 A3 A4 :=
sorry

end tetrahedron_inequality_implies_regular_l366_366733


namespace remainder_3001_3002_3003_3004_3005_mod_17_l366_366364

theorem remainder_3001_3002_3003_3004_3005_mod_17 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := 
begin
  sorry
end

end remainder_3001_3002_3003_3004_3005_mod_17_l366_366364


namespace smallest_area_of_triangle_is_minimum_l366_366622

noncomputable def smallest_area_of_triangle (s : ℝ) : ℝ :=
  let A := (-2 : ℝ, 0, 3)
  let B := (0 : ℝ, 3, 4)
  let C := (s, 0, 0)
  let vector_a_to_b := (2, 3, 1)
  let vector_a_to_c := (s + 2, 0, -3)
  let cross_product := (3, s + 8, -3 * s - 6)
  let area := (1 / 2) * Real.sqrt (9 + (s + 8) ^ 2 + (-3 * s - 6) ^ 2)
  area
   
theorem smallest_area_of_triangle_is_minimum (s : ℝ) :
  (∃ s, smallest_area_of_triangle s = Real.sqrt 27.4 / 2) := by
  sorry

end smallest_area_of_triangle_is_minimum_l366_366622


namespace cylinder_height_l366_366321

-- Define the parameters.
def diameter : ℝ := 10
def radius : ℝ := diameter / 2
def volume : ℝ := 1099.5574287564277

-- The goal is to prove the height is 14 cm given the conditions.
theorem cylinder_height (h : ℝ) : h = volume / (π * radius ^ 2) → h = 14 :=
by
  sorry

end cylinder_height_l366_366321


namespace probability_B_in_middle_l366_366725

theorem probability_B_in_middle (A B C : Type) [fintype A] [fintype B] [fintype C] :
  let total_arrangements := 6
  let favorable_arrangements := 2
  let probability := favorable_arrangements / total_arrangements
  probability = 1 / 3 :=
by 
  repeat { sorry }

end probability_B_in_middle_l366_366725


namespace distance_product_eq_l366_366014

-- Define necessary variables and hypotheses
variables {a b m n : ℝ}
variables {x y : ℝ}

-- Conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ ((x^2 / m) - (y^2 / n) = 1)

-- Defining the foci sharing condition and intersection point
def share_foci (a b m n : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  is_ellipse a b P.1 P.2 ∧ is_hyperbola m n P.1 P.2 ∧
  (F1, F2 are the same for both curves)

-- The theorem to prove
theorem distance_product_eq (a b m n : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ)
  (h_ellipse : is_ellipse a b P.1 P.2)
  (h_hyperbola : is_hyperbola m n P.1 P.2)
  (h_foci : share_foci a b m n F1 F2 P) :
  (let |PF1| := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
       |PF2| := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
   in |PF1| * |PF2| = a - m) :=
sorry

end distance_product_eq_l366_366014


namespace problem_1_problem_2_l366_366248

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -4 ≤ x ∧ x < 2 }
def B : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def P : Set ℝ := { x | x ≤ 0 ∨ x ≥ 5 / 2 }

theorem problem_1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem problem_2 : (U \ B) ∪ P = { x | x ≤ 0 ∨ x ≥ 5 / 2 } :=
sorry

end problem_1_problem_2_l366_366248


namespace num_pos_divisors_36_l366_366062

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366062


namespace max_sum_at_n_7_l366_366973

-- Definitions based on the conditions
variables {a_n : ℕ → ℤ} {d : ℤ} (h_d : d < 0)
def S (n : ℕ) : ℤ := (n * (2 * a_n 1 + (n - 1) * d)) / 2

-- Condition: S_3 = 11 * a_6
theorem max_sum_at_n_7 (h: S 3 = 11 * a_n 6) : S 7 = max (λ n, S n) sorry :=
sorry

end max_sum_at_n_7_l366_366973


namespace concurrency_of_perpendiculars_l366_366643

noncomputable def point : Type := sorry
noncomputable def line : Type := sorry

-- Definitions for perpendicular bisectors
def is_perpendicular_bisector (l : line) (A B : point) : Prop := sorry

-- Definitions for points being noncollinear
def noncollinear (A B C : point) : Prop := sorry

-- Defining the main variables and conditions

variables (A B C D E F : point)
variables (l1 l2 l3 l4 : line)

-- Conditions
axiom cond1 : is_perpendicular_bisector l1 B C ∧ D ∈ l1
axiom cond2 : is_perpendicular_bisector l2 C A ∧ E ∈ l2
axiom cond3 : is_perpendicular_bisector l3 A B ∧ F ∈ l3
axiom cond4 : noncollinear D E F

-- Question to be proved: concurrency of lines through A, B, C perpendicular to EF, FD, DE respectively
def is_perpendicular (l : line) (A B : point) : Prop := sorry
def are_concurrent (l m n : line) : Prop := sorry

theorem concurrency_of_perpendiculars :
  are_concurrent 
    (line_through_perpendicular_to A E F) 
    (line_through_perpendicular_to B F D) 
    (line_through_perpendicular_to C D E) :=
sorry

end concurrency_of_perpendiculars_l366_366643


namespace sqrt_y_cubed_eq_216_l366_366376

theorem sqrt_y_cubed_eq_216 (y : ℝ) (h : (sqrt y)^3 = 216) : y = 36 :=
sorry

end sqrt_y_cubed_eq_216_l366_366376


namespace exponent_equation_l366_366576

theorem exponent_equation (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by sorry

end exponent_equation_l366_366576


namespace rounding_to_nearest_hundredth_l366_366672

-- Definitions
def number_to_round : ℝ := 3.456
def correct_answer : ℝ := 3.46

-- Statement
theorem rounding_to_nearest_hundredth : round_to_nearest_hundredth number_to_round = correct_answer :=
by sorry

end rounding_to_nearest_hundredth_l366_366672


namespace completing_the_square_l366_366377

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l366_366377


namespace lcm_18_24_l366_366801

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366801


namespace bus_lengths_are_equal_l366_366339

noncomputable def distance_A : ℝ := 12 -- Bus A covers 12 km
noncomputable def time_A_minutes : ℝ := 5 / 60 -- Time taken by Bus A in hours
noncomputable def distance_B : ℝ := 18 -- Bus B covers 18 km
noncomputable def time_B_minutes : ℝ := 6 / 60 -- Time taken by Bus B in hours
noncomputable def time_pass_post_A : ℝ := 5 / 3600 -- Time taken by Bus A to pass the post in hours
noncomputable def time_pass_post_B : ℝ := 4 / 3600 -- Time taken by Bus B to pass the post in hours

def speed (distance time : ℝ) : ℝ := distance / time

theorem bus_lengths_are_equal :
  let speed_A := speed distance_A time_A_minutes,
      speed_B := speed distance_B time_B_minutes in
  let length_A := speed_A * time_pass_post_A,
      length_B := speed_B * time_pass_post_B in
  length_A * 1000 = 200 ∧ length_B * 1000 = 200 := 
by
  assume speed_A := speed distance_A time_A_minutes,
        speed_B := speed distance_B time_B_minutes,
        length_A := speed_A * time_pass_post_A,
        length_B := speed_B * time_pass_post_B
  show length_A * 1000 = 200 ∧ length_B * 1000 = 200, from sorry

end bus_lengths_are_equal_l366_366339


namespace num_pos_divisors_36_l366_366071

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366071


namespace correct_number_of_true_statements_l366_366959

-- Given Definitions
def f (a b c x : ℝ) := a * x^2 + b * x + c
def g (a b c x : ℝ) := c * x^2 + b * x + a
def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Constraints
variables (a b c : ℝ)
hypothesis h_ac : a ≠ 0 ∧ c ≠ 0

-- Statements to evaluate
def statement1 (a b c : ℝ) := discriminant a b c < 0 → ∀ x : ℝ, g a b c x > 0
def statement2 (a b c : ℝ) := discriminant a b c = 0 → ∃ x : ℝ, g a b c x = 0
def statement3 (a b c : ℝ) := discriminant a b c > 0 → ¬∀ x : ℝ, g a b c x ≠ 0

-- Number of true statements
def numberOfTrueStatements (a b c : ℝ) : ℕ :=
  (if statement1 a b c then 1 else 0) +
  (if statement2 a b c then 1 else 0) +
  (if statement3 a b c then 1 else 0)

-- Proof statement
theorem correct_number_of_true_statements : 
  numberOfTrueStatements a b c = 1 :=
by
  sorry

end correct_number_of_true_statements_l366_366959


namespace cannot_form_set_l366_366883

-- Conditions used in the proof
def well_defined_collection (X : Set α) := ∀ x : α, x ∈ X ∨ x ∉ X

def difficulty_is_subjective (problems : Set Problem) := ¬well_defined_collection (more_difficult_problems problems)

-- The proof problem statement
theorem cannot_form_set (problems : Set Problem) :
  difficulty_is_subjective problems :=
begin
  sorry
end

end cannot_form_set_l366_366883


namespace math_problem_part1_math_problem_part2_l366_366981

open Set Real

def A : Set ℝ := {x | x^2 - 6 * x < -5}
def B : Set ℝ := {x | 1 < 2^(x-2) ∧ 2^(x-2) ≤ 16}
def f (a x : ℝ) : ℝ := log ((2*a - x) * (x - a - 1))
def C (a : ℝ) : Set ℝ := {x | (2*a - x) * (x - a - 1) > 0}

theorem math_problem_part1 :
  (A ∪ B) = {x : ℝ | 1 < x ∧ x ≤ 6}
  ∧ Aᶜ = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
sorry

theorem math_problem_part2 (a : ℝ) :
  (A ∩ C a) = (C a) → (a ∈ Icc (1 / 2) 1 ∨ a ∈ Ioc 1 (5 / 2)) :=
sorry

end math_problem_part1_math_problem_part2_l366_366981


namespace part_a_part_b_l366_366836

noncomputable def maxEdgesNoTriangles (n : ℕ) : ℕ :=
  if n = 30 then 225 else sorry

theorem part_a : ∀ (G : SimpleGraph (Fin 30)), 
  (∀ v w z : Fin 30, ¬ (G.Adj v w ∧ G.Adj w z ∧ G.Adj z v)) →
    G.edgeFinset.card ≤ maxEdgesNoTriangles 30 := 
by
  intros
  sorry

noncomputable def maxEdgesNoK4 (n : ℕ) : ℕ :=
  if n = 30 then 200 else sorry

theorem part_b : ∀ (G : SimpleGraph (Fin 30)), 
  (∀ (H : Finset (Fin 30)), H.card = 4 → ¬ completeSubgraph G H) →
    G.edgeFinset.card ≤ maxEdgesNoK4 30 :=
by
  intros
  sorry

end part_a_part_b_l366_366836


namespace area_of_ABCD_l366_366188

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l366_366188


namespace coin_toss_probability_l366_366406

-- Define the probability of getting at least two consecutive heads
def probability_two_consecutive_heads : ℚ :=
  1 / 2

-- Define the problem statement
theorem coin_toss_probability :
  let total_outcomes := 2^4,
      unfavorable_outcomes := 8, 
      probability_no_consecutive_heads := unfavorable_outcomes / total_outcomes
  in 1 - probability_no_consecutive_heads = probability_two_consecutive_heads :=
by
  -- The calculations and the proof steps would go here
  sorry

end coin_toss_probability_l366_366406


namespace solve_problem1_solve_problem2_l366_366278

noncomputable def problem1 (m n : ℝ) : Prop :=
  (m + n) ^ 2 - 10 * (m + n) + 25 = (m + n - 5) ^ 2

noncomputable def problem2 (x : ℝ) : Prop :=
  ((x ^ 2 - 6 * x + 8) * (x ^ 2 - 6 * x + 10) + 1) = (x - 3) ^ 4

-- Placeholder for proofs
theorem solve_problem1 (m n : ℝ) : problem1 m n :=
by
  sorry

theorem solve_problem2 (x : ℝ) : problem2 x :=
by
  sorry

end solve_problem1_solve_problem2_l366_366278


namespace num_positive_divisors_36_l366_366093

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end num_positive_divisors_36_l366_366093


namespace area_of_triangle_DEF_eq_480_l366_366159

theorem area_of_triangle_DEF_eq_480 (DE EF DF : ℝ) (h1 : DE = 20) (h2 : EF = 48) (h3 : DF = 52) :
  let s := (DE + EF + DF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  area = 480 :=
by
  sorry

end area_of_triangle_DEF_eq_480_l366_366159


namespace curve_B_is_not_good_l366_366514

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end curve_B_is_not_good_l366_366514


namespace lcm_18_24_l366_366797

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366797


namespace max_price_reduction_l366_366310

theorem max_price_reduction (CP SP : ℝ) (profit_margin : ℝ) (max_reduction : ℝ) :
  CP = 1000 ∧ SP = 1500 ∧ profit_margin = 0.05 → SP - max_reduction = CP * (1 + profit_margin) → max_reduction = 450 :=
by {
  sorry
}

end max_price_reduction_l366_366310


namespace biology_marks_l366_366464

theorem biology_marks (e m p c : ℕ) (a : ℚ) (n : ℕ) (B : ℕ) : 
  e = 72 → m = 60 → p = 35 → c = 62 → a = 62.6 → n = 5 → 
  B = (a * n).to_nat - (e + m + p + c) → B = 84 :=
by
  intros h_e h_m h_p h_c h_a h_n h_B
  calc
    B = (a * n).to_nat - (e + m + p + c) : h_B
    ... = 313 - 229 : by { simp [h_e, h_m, h_p, h_c, h_a, h_n], norm_num }
    ... = 84 : by norm_num

end biology_marks_l366_366464


namespace michaels_brother_money_end_l366_366655

theorem michaels_brother_money_end 
  (michael_money : ℕ)
  (brother_money : ℕ)
  (gives_half : ℕ)
  (buys_candy : ℕ) 
  (h1 : michael_money = 42)
  (h2 : brother_money = 17)
  (h3 : gives_half = michael_money / 2)
  (h4 : buys_candy = 3) : 
  brother_money + gives_half - buys_candy = 35 :=
by {
  sorry
}

end michaels_brother_money_end_l366_366655


namespace collinear_points_l366_366596

variables {O A B C D E F P Q : Type*}

-- Definitions of the conditions
variable [circle : circle O]
variable [diameter : diameter A B O]
variable [diameter : diameter C D O]
variable [on_circle : point_on_circle E O]
variable [on_circle : point_on_circle F O]

-- Definitions of intersection points
def P : Type* := intersection_point (line_through A E) (line_through D F)
def Q : Type* := intersection_point (line_through C E) (line_through B F)

-- The proof statement
theorem collinear_points : collinear {P, O, Q} :=
sorry

end collinear_points_l366_366596


namespace range_of_positive_a_l366_366584

def is_monotonically_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, (aeval x (deriv (λ x, real.exp (a * x + 1) - x * (real.log x - 2)))) < 0

theorem range_of_positive_a (f : ℝ → ℝ) : (0 < a ∧ is_monotonically_decreasing f a) → (a < real.exp (-2)) := 
begin
  intro h,
  sorry
end

end range_of_positive_a_l366_366584


namespace find_n_l366_366569

theorem find_n (n : ℕ) (h : 8^4 = 16^n) : n = 3 :=
by
  unfold pow at h
  sorry

end find_n_l366_366569


namespace ord_contains_at_most_one_prime_factor_l366_366234

noncomputable def Ord (m a : ℕ) : ℕ := -- Definition of order function
  -- Definition implementation (details omitted)
  sorry

theorem ord_contains_at_most_one_prime_factor
  {a m : ℕ} -- Declare a and m as natural numbers (positive integers implicitly)
  (h1 : 0 < a ∧ 0 < m) -- Condition: a and m are positive integers
  (h2 : Ord m a % 2 = 1) -- Condition: Ord_m(a) is odd
  (h3 : ∀ x y : ℕ, -- Condition about x and y
    (x * y ≡ a [MOD m]) → 
    (Ord m x ≤ Ord m a ∧ Ord m y ≤ Ord m a) → 
    (Ord m x ∣ Ord m a ∨ Ord m y ∣ Ord m a)) :
  ∃ p : ℕ, (nat.prime p) ∧ (Ord m a = p ∨ Ord m a = 1) := -- Conclusion: there exists at most one prime factor in Ord_m(a)
sorry

end ord_contains_at_most_one_prime_factor_l366_366234


namespace rectangle_area_l366_366213

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366213


namespace man_was_absent_for_days_l366_366835

theorem man_was_absent_for_days
  (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 10 * x - 2 * y = 216) :
  y = 7 :=
by
  sorry

end man_was_absent_for_days_l366_366835


namespace ratio_of_sides_l366_366885

theorem ratio_of_sides (
  perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 64) :
  (perimeter_triangle / 3) / (perimeter_square / 4) = 1 :=
by
  sorry

end ratio_of_sides_l366_366885


namespace area_of_rectangle_l366_366204

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l366_366204


namespace infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l366_366738

-- Step d): Definitions based on conditions in a)
def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ), 0 < n ∧ n = ∏ i in Finset.range k, d i ∧ ∀ i : Fin k, (d i)^2 ∣ (n + d i)

-- Part (a): There are infinitely many lovely numbers
theorem infinitely_many_lovely_numbers : ∃ (N : ℕ) (p : ℕ → Prop), lovely p ∧ N < p := sorry

-- Part (b): There does not exist a lovely number greater than 1 which is a square of an integer
theorem no_lovely_square_greater_than_one :
  ¬ ∃ n : ℕ, n > 1 ∧ lovely (n^2) := sorry

end infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l366_366738


namespace rectangle_area_l366_366194

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366194


namespace rectangle_area_l366_366215

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l366_366215


namespace find_values_for_opposite_directions_l366_366001

variables (e1 e2 : Vector ℝ)
variables (x y : ℝ)
variable (λ : ℝ)
variables (a b : Vector ℝ)

def non_collinear (u v : Vector ℝ) : Prop := ¬(∃ k : ℝ, u = k • v)

def opposite_directions (a b : Vector ℝ) : Prop := ∃ λ : ℝ, λ < 0 ∧ a = λ • b

theorem find_values_for_opposite_directions (h_non_collinear : non_collinear e1 e2)
    (h_a : a = x • (e1 + e2)) (h_b : b = e1 + y • e2) :
    opposite_directions a b ↔ (x = -1 ∧ y = 1) :=
sorry

end find_values_for_opposite_directions_l366_366001


namespace expansion_constant_term_l366_366015

/-- Given that the expansion of (\sqrt{x} - 1/(24 * x))^n contains a constant term as its fifth term,
we need to prove that n equals 6 and find all the rational terms in the expansion. -/
theorem expansion_constant_term (x : ℝ) (n : ℕ) (h : (∃ a : ℝ, a = sorry)) : 
  -- Prove n = 6
  n = 6 ∧ 
  -- Prove the rational terms in the expansion are x^3 and 15/16
  let T : ℕ → ℝ := λ r, binomial n r * (-1/2)^r * x^((2*n - 3*r)/4)
  ∃ (r₀ r₄ : ℕ), T 0 = x^3 ∧ T 4 = 15/16 := sorry

end expansion_constant_term_l366_366015


namespace percentage_increase_on_sale_l366_366825

theorem percentage_increase_on_sale (P S : ℝ) (hP : P ≠ 0) (hS : S ≠ 0)
  (h_price_reduction : (0.8 : ℝ) * P * S * (1 + (X / 100)) = 1.44 * P * S) :
  X = 80 := by
  sorry

end percentage_increase_on_sale_l366_366825


namespace determine_a10_l366_366447

theorem determine_a10 (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ) 
  (h : (1 - z) ^ a1 * (1 - z^2) ^ a2 * (1 - z^3) ^ a3 * 
       (1 - z^4) ^ a4 * (1 - z^5) ^ a5 * (1 - z^6) ^ a6 * 
       (1 - z^7) ^ a7 * (1 - z^8) ^ a8 * (1 - z^9) ^ a9 * 
       (1 - z^10) ^ a10 ≡ (1 - 3 * z) [z^11]) : a10 = 0 :=
sorry

end determine_a10_l366_366447


namespace red_section_not_damaged_l366_366460

open ProbabilityTheory

noncomputable def bernoulli_p  : ℝ := 2/7
noncomputable def bernoulli_n  : ℕ := 7
noncomputable def no_success_probability : ℝ := (5/7) ^ bernoulli_n

theorem red_section_not_damaged : 
  ∀ (X : ℕ → ℝ), (∀ k, X k = ((7.choose k) * (bernoulli_p ^ k) * ((1 - bernoulli_p) ^ (bernoulli_n - k)))) → 
  (X 0 = no_success_probability) :=
begin
  intros,
  simp [bernoulli_p, bernoulli_n, no_success_probability],
  sorry
end

end red_section_not_damaged_l366_366460


namespace min_value_g_geq_6_min_value_g_eq_6_l366_366489

noncomputable def g (x : ℝ) : ℝ :=
  x + (x / (x^2 + 2)) + (x * (x + 5) / (x^2 + 3)) + (3 * (x + 3) / (x * (x^2 + 3)))

theorem min_value_g_geq_6 : ∀ x > 0, g x ≥ 6 :=
by
  sorry

theorem min_value_g_eq_6 : ∃ x > 0, g x = 6 :=
by
  sorry

end min_value_g_geq_6_min_value_g_eq_6_l366_366489


namespace reciprocal_of_sum_l366_366352

theorem reciprocal_of_sum : (1 / (1 / 3 + 1 / 4)) = 12 / 7 := 
by sorry

end reciprocal_of_sum_l366_366352


namespace find_u_value_l366_366328

theorem find_u_value (u : ℤ) : ∀ (y : ℤ → ℤ), 
  (y 2 = 8) → (y 4 = 14) → (y 6 = 20) → 
  (∀ x, (x % 2 = 0) → (y (x + 2) = y x + 6)) → 
  y 18 = u → u = 56 :=
by
  intros y h2 h4 h6 pattern h18
  sorry

end find_u_value_l366_366328


namespace solve_system_l366_366483

variables {x : ℕ → ℝ} {y : ℝ} {s t : ℝ}  

theorem solve_system :
  (∀ i, x i + x (i + 2) % 5 = y * x (i + 1) % 5) ↔ 
  (∀ i, x i = 0) ∨ 
  (y = 2 ∧ (∀ i, x i = s)) ∨
  (y = (-1 + sqrt 5) / 2 ∨ y = (-1 - sqrt 5) / 2) ∧
  (x 0 = s ∧ x 1 = t ∧ x 2 = -s + y * t ∧ x 3 = - y * s - t ∧ x 4 = y * s - t) :=
sorry

end solve_system_l366_366483


namespace number_of_white_balls_l366_366171

theorem number_of_white_balls (a : ℕ) (h1 : 3 + a ≠ 0) (h2 : (3 : ℚ) / (3 + a) = 3 / 7) : a = 4 :=
sorry

end number_of_white_balls_l366_366171


namespace division_remainder_l366_366370

def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30
def d (x : ℝ) : ℝ := 4 * x - 8

theorem division_remainder : (∃ q r, p(2) = d(2) * q + r ∧ d(2) ≠ 0 ∧ r = 10) :=
by
  sorry

end division_remainder_l366_366370


namespace andy_final_position_l366_366886

open Int

def next_position (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
match direction % 4 with
| 0 => (pos.1, pos.2 + distance)   -- Moving north
| 1 => (pos.1 + distance, pos.2)   -- Moving east
| 2 => (pos.1, pos.2 - distance)   -- Moving south
| _ => (pos.1 - distance, pos.2)   -- Moving west

def final_position_after_moves (start_pos : ℤ × ℤ) (n : ℕ) : ℤ × ℤ :=
(nat.iterate (λ ⟨(pos, direction, dist)⟩, let next_pos := next_position pos direction dist in
  (next_pos, direction + 1, if (direction + 1) % 4 == 0 then dist + 1 else dist)) n
  (start_pos, 0, 1)).1

theorem andy_final_position : final_position_after_moves (10, -10) 2021 = (10, 496) :=
sorry

end andy_final_position_l366_366886


namespace dima_wins_with_perfect_play_l366_366918

theorem dima_wins_with_perfect_play : 
  ∀ (n : ℕ), (∀ m ≤ 10, n + m ≠ 100) → (∃ m ≤ 10, n + m ≥ 100) → (0 ≤ n < 100 → ∃ m ≤ 10, n + m < 100) → (91 ≤ n + m ≤ 100) → false → "Dima wins with perfect play" := 
begin
  sorry,
end

end dima_wins_with_perfect_play_l366_366918


namespace solve_f_2_minus_x_pos_l366_366530

theorem solve_f_2_minus_x_pos (a : ℝ) (h_pos : a > 0) :
  let f := (λ x, (x - 2) * (a * x + 2 * a)) in
  let f_even := (∀ x, f x = f (-x)) in
  (λ x, f (2 - x) > 0) =
  (λ x, x < 0 ∨ x > 4) :=
by
  sorry

end solve_f_2_minus_x_pos_l366_366530


namespace probability_no_success_l366_366453

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l366_366453


namespace lcm_18_24_eq_72_l366_366808

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end lcm_18_24_eq_72_l366_366808


namespace num_pos_divisors_36_l366_366105

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366105


namespace area_of_triangle_OKT_l366_366308

noncomputable theory

variables {A B C M N O K T : Type*} [linear_ordered_field A]
variables (BM : A) (a : A) (hBM : BM = a)
variables (angle_ABC half_angle : A) [lt_one_half : 1 / 2 < 1]
variables (O_center ω_tangent_to_BA ω_tangent_to_BC : Prop)
variables (line_through_M_parallel_to_BC : Prop)
variables (line_KT_tangent_to_ω : Prop)
variables (angle_MTK_eq_half_angle_ABC : angle_ABC = 2 * half_angle)

-- Given conditions
def conditions : Prop :=
  ω_tangent_to_BA ∧ ω_tangent_to_BC ∧ line_through_M_parallel_to_BC ∧
  line_KT_tangent_to_ω ∧ hBM ∧ angle_MTK_eq_half_angle_ABC

-- Question rephrased as a proof problem
theorem area_of_triangle_OKT (h_conditions : conditions) : 
  real_area OKT = a^2 / 2 :=
sorry

end area_of_triangle_OKT_l366_366308


namespace ab_sufficient_not_necessary_ab_not_necessary_sufficient_but_not_necessary_l366_366954

theorem ab_sufficient_not_necessary (a b : ℝ) (h : ab = 1) : a^2 + b^2 ≥ 2 := 
by {
  sorry,
  }

theorem ab_not_necessary (a b : ℝ) (h : a^2 + b^2 ≥ 2) : ¬ (ab = 1) := 
by {
  sorry,
  }

theorem sufficient_but_not_necessary :
(∀ a b, (a, b : ℝ) → (ab = 1 → a^2 + b^2 ≥ 2)) ∧ (∃ a b, a, b : ℝ ∧ a^2 + b^2 ≥ 2 ∧ ab ≠ 1) :=
by {
  sorry,
  }

end ab_sufficient_not_necessary_ab_not_necessary_sufficient_but_not_necessary_l366_366954


namespace largest_variable_l366_366026

theorem largest_variable {x y z w : ℤ} 
  (h1 : x + 3 = y - 4)
  (h2 : x + 3 = z + 2)
  (h3 : x + 3 = w - 1) :
  y > x ∧ y > z ∧ y > w :=
by sorry

end largest_variable_l366_366026


namespace rounding_to_nearest_hundredth_l366_366673

-- Definitions
def number_to_round : ℝ := 3.456
def correct_answer : ℝ := 3.46

-- Statement
theorem rounding_to_nearest_hundredth : round_to_nearest_hundredth number_to_round = correct_answer :=
by sorry

end rounding_to_nearest_hundredth_l366_366673


namespace multiples_of_15_between_25_and_200_l366_366557

theorem multiples_of_15_between_25_and_200 : 
  let S := {x : ℕ | 25 < x ∧ x < 200 ∧ x % 15 = 0} in
  S.card = 12 := 
by 
  sorry

end multiples_of_15_between_25_and_200_l366_366557


namespace range_of_m_l366_366508

variable (m : ℝ)

def p (m : ℝ) : Prop :=
  8 - m > 0 ∧ 2m - 1 > 0 ∧ 8 - m > 2m - 1

def q (m : ℝ) : Prop :=
  (m + 1) * (m - 2) < 0

theorem range_of_m (h : p m ∨ q m) (h' : ¬ (p m ∧ q m)) : 
  m ∈ Set.Ioo (-1 : ℝ) (1 / 2) ∪ Set.Ico (2 : ℝ) (3 : ℝ) := 
sorry

end range_of_m_l366_366508


namespace distance_between_points_l366_366469

theorem distance_between_points :
  let p1 := (3, -2, 5)
  let p2 := (6, -7, 10)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2) = Real.sqrt 59 :=
by
  let p1 := (3, -2, 5)
  let p2 := (6, -7, 10)
  have h1 : p2.1 - p1.1 = 3 := by norm_num
  have h2 : p2.2 - p1.2 = -5 := by norm_num
  have h3 : p2.3 - p1.3 = 5 := by norm_num
  calc
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)
        = Real.sqrt (3^2 + (-5)^2 + 5^2) : by rw [h1, h2, h3]
    ... = Real.sqrt (9 + 25 + 25) : by norm_num
    ... = Real.sqrt 59 : by norm_num

end distance_between_points_l366_366469


namespace multiples_of_15_between_25_and_200_l366_366562

theorem multiples_of_15_between_25_and_200 : 
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  in multiples.length = 12 :=
by
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  show multiples.length = 12
  sorry

end multiples_of_15_between_25_and_200_l366_366562


namespace largest_cosine_value_geometric_triangle_l366_366999

theorem largest_cosine_value_geometric_triangle (a : ℝ) (h : a > 0) :
  let b := a * (Real.sqrt 2),
      c := 2 * a,
      cos_largest := (a^2 + b^2 - c^2) / (2 * a * b) in
  cos_largest = -Real.sqrt 2 / 4 :=
by
  -- This is where the proof would go, but we omit it for now.
  sorry

end largest_cosine_value_geometric_triangle_l366_366999


namespace domain_of_function_l366_366911

open Real

theorem domain_of_function :
  {x : ℝ | 3 - 2 * x - x^2 > 0} = set.Ioo (-3 : ℝ) (1:ℝ) :=
by
  sorry

end domain_of_function_l366_366911


namespace binary_to_base4_conversion_l366_366907

-- Define the binary number as a string
def binary_number : string := "1101101001"

-- Define the expected base 4 number as a string
def expected_base4_number : string := "13201"

-- State the theorem to prove the conversion is correct
theorem binary_to_base4_conversion : 
  convert_to_base4 binary_number = expected_base4_number :=
sorry

end binary_to_base4_conversion_l366_366907


namespace num_pos_divisors_36_l366_366121

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366121


namespace white_squares_in_30th_row_l366_366398

theorem white_squares_in_30th_row :
  ∀ (n : ℕ), 
    (∃ N, 
      (N = 2 * n) ∧ 
      N mod 2 = 0 ∧ 
      (∀ k, 1 ≤ k ∧ k ≤ N → (k = 1 ∨ k = N → k mod 2 = 1) ∧ (1 < k ∧ k < N → k mod 2 ≠ 0)) → 
    n = 30 → N / 2 = 30) :=
by
  sorry

end white_squares_in_30th_row_l366_366398


namespace even_odd_divisors_sum_diff_le_n_l366_366299

def s_n (n : ℕ) (k : ℕ) : ℕ := n / k

def D1 (n : ℕ) : ℕ := ∑ j in (finset.range (n / 2 + 1)).filter (λ j, j > 0), s_n n (2 * j)

def D2 (n : ℕ) : ℕ := ∑ j in (finset.range (n + 1)).filter (λ j, j > 0 ∧ j % 2 = 1), s_n n j

theorem even_odd_divisors_sum_diff_le_n (n : ℕ) : D2 n - D1 n ≤ n := sorry

end even_odd_divisors_sum_diff_le_n_l366_366299


namespace g_at_minus_one_l366_366245

def g (x : ℝ) : ℝ :=
  if x < -2 then 2 * x + 6 else 10 - 3 * x

theorem g_at_minus_one : g (-1) = 13 :=
by
  have h1 : g (-1) = 10 - 3 * (-1) := if_neg (not_lt_of_ge (by norm_num))
  have h2 : 10 - 3 * (-1) = 13 := by norm_num
  rw [h1, h2]
  exact rfl

end g_at_minus_one_l366_366245


namespace female_guests_from_jays_family_l366_366250

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l366_366250


namespace find_annual_interest_rate_l366_366846

noncomputable def annual_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  n * ((A / P) ^ (1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  annual_interest_rate 650 914.6152747265625 12 7 ≈ 0.058410606 :=
by
  sorry

end find_annual_interest_rate_l366_366846


namespace quadratic_function_l366_366393

theorem quadratic_function (f : ℝ → ℝ) (h₁ : ∃ a b : ℝ, f = λ x, a*x^2 + b*x)
  (h₂ : f 0 = 0) (h₃ : ∀ x, f (x + 1) = f x + x + 1) :
  ∀ x, f x = (1 / 2) * x^2 + (1 / 2) * x :=
sorry

end quadratic_function_l366_366393


namespace least_number_increased_by_20_divisible_by_set_l366_366389

theorem least_number_increased_by_20_divisible_by_set (n : ℕ) :
  (∀ m ∈ {52, 84, 114, 133, 221, 379}, (n + 20) % m = 0) ↔ n = 1097897218492 :=
by
  sorry

end least_number_increased_by_20_divisible_by_set_l366_366389


namespace solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l366_366019

theorem solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c ≤ 0 ↔ x ≤ -1 ∨ x ≥ 3) →
  b = -2*a →
  c = -3*a →
  a < 0 →
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) := 
by 
  intro h_root_set h_b_eq h_c_eq h_a_lt_0 
  sorry

end solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l366_366019


namespace number_of_divisors_36_l366_366085

-- Defining the number and its prime factorization
def n : ℕ := 36
def factorization : (ℕ × ℕ) := (2, 2)

-- The number of positive divisors based on the given prime factorization
def number_of_divisors (n : ℕ) (f : ℕ × ℕ) : ℕ :=
  let (a, b) := f
  (a + 1) * (b + 1)

-- Assertion to be proven
theorem number_of_divisors_36 : number_of_divisors n factorization = 9 := by
  -- proof omitted
  sorry

end number_of_divisors_36_l366_366085


namespace part1_solution_part2_solution_l366_366644

-- Definitions for propositions p and q
def p (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- The actual Lean 4 statements
theorem part1_solution (x : ℝ) (m : ℝ) (hm : m = 1) (hp : p m x) (hq : q x) : 2 ≤ x ∧ x < 3 := by
  sorry

theorem part2_solution (m : ℝ) (hm : m > 0) (hsuff : ∀ x, q x → p m x) : (4 / 3) < m ∧ m < 2 := by
  sorry

end part1_solution_part2_solution_l366_366644


namespace part1_part2_l366_366022

variable (m n : ℝ × ℝ × ℝ) -- Define vectors in ℝ³
variable (θ : ℝ) -- Define angle θ

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
noncomputable def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

axiom angle_m : θ = Real.pi / 3
axiom mag_m : magnitude m = 2
axiom unit_n : magnitude n = 1

theorem part1 : dot_product m n = 1 := by
  sorry

theorem part2 : magnitude (m.1 + 2 * n.1, m.2 + 2 * n.2, m.3 + 2 * n.3) = 2 * Real.sqrt 3 := by
  sorry

end part1_part2_l366_366022


namespace tank_full_capacity_l366_366408

-- Define the conditions
def gas_tank_initially_full_fraction : ℚ := 4 / 5
def gas_tank_after_usage_fraction : ℚ := 1 / 3
def used_gallons : ℚ := 18

-- Define the statement that translates to "How many gallons does this tank hold when it is full?"
theorem tank_full_capacity (x : ℚ) : 
  gas_tank_initially_full_fraction * x - gas_tank_after_usage_fraction * x = used_gallons → 
  x = 270 / 7 :=
sorry

end tank_full_capacity_l366_366408


namespace lcm_18_24_l366_366791

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366791


namespace math_problems_equivalence_l366_366540

noncomputable def f (a x : ℝ) := (a^2 - 3*a + 3) * a^x

def Q1_expression (a : ℝ) : Prop :=
  (a^2 - 3*a + 3 = 1) → (∃ a, a = 2)

noncomputable def F (x : ℝ) := 2^x - 2^(-x)

def Q2_parity : Prop :=
  ∀ x, F(-x) = -F(x)

def Q3_inequality_solution_set : set ℝ := { x | -2 < x ∧ x < -1/2 }

def Q3_proof : Prop :=
  ∀ x, log 2 (1 - x) > log 2 (x + 2) ↔ x ∈ Q3_inequality_solution_set

theorem math_problems_equivalence :
  (Q1_expression 2) ∧ Q2_parity ∧ (Q3_proof) :=
begin
  sorry,
end

end math_problems_equivalence_l366_366540


namespace car_value_proof_l366_366227

-- Let's define the variables and the conditions.
def car_sold_value : ℝ := 20000
def sticker_price_new_car : ℝ := 30000
def percent_sold : ℝ := 0.80
def percent_paid : ℝ := 0.90
def out_of_pocket : ℝ := 11000

theorem car_value_proof :
  (percent_paid * sticker_price_new_car - percent_sold * car_sold_value = out_of_pocket) →
  car_sold_value = 20000 := 
by
  intros h
  -- Introduction of any intermediate steps if necessary should just invoke the sorry to indicate the need for proof later
  exact sorry

end car_value_proof_l366_366227


namespace batsman_new_average_l366_366382

def batsman_average_after_16_innings (A : ℕ) (new_avg : ℕ) (runs_16th : ℕ) : Prop :=
  15 * A + runs_16th = 16 * new_avg

theorem batsman_new_average (A : ℕ) (runs_16th : ℕ) (h1 : batsman_average_after_16_innings A (A + 3) runs_16th) : A + 3 = 19 :=
by
  sorry

end batsman_new_average_l366_366382


namespace infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l366_366737

-- Step d): Definitions based on conditions in a)
def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ), 0 < n ∧ n = ∏ i in Finset.range k, d i ∧ ∀ i : Fin k, (d i)^2 ∣ (n + d i)

-- Part (a): There are infinitely many lovely numbers
theorem infinitely_many_lovely_numbers : ∃ (N : ℕ) (p : ℕ → Prop), lovely p ∧ N < p := sorry

-- Part (b): There does not exist a lovely number greater than 1 which is a square of an integer
theorem no_lovely_square_greater_than_one :
  ¬ ∃ n : ℕ, n > 1 ∧ lovely (n^2) := sorry

end infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l366_366737


namespace tile_rectangle_condition_l366_366239

theorem tile_rectangle_condition (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  (∃ q, m = k * q) ∨ (∃ r, n = k * r) :=
sorry

end tile_rectangle_condition_l366_366239


namespace num_pos_divisors_36_l366_366107

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366107


namespace infinitely_many_lovely_numbers_no_lovely_number_square_l366_366735

def is_lovely (n : ℕ) : Prop :=
  ∃ k : ℕ, ∃ d : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ k → d i > 0) ∧
    (n = ∏ i in finset.range k, d i) ∧ (∀ i, 1 ≤ i ∧ i ≤ k → (d i)^2 ∣ n + (d i))

theorem infinitely_many_lovely_numbers : ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_lovely n := 
sorry

theorem no_lovely_number_square (n : ℕ) (hn : is_lovely n) : n ≠ x*x ∨ x ≤ 1 :=
sorry

end infinitely_many_lovely_numbers_no_lovely_number_square_l366_366735


namespace at_least_one_good_point_l366_366965

-- Define the problem in Lean terms
theorem at_least_one_good_point (n : ℕ) (h₁ : 3 * n + 1 = 2017) (h₂ : n ≤ 672) :
  ∃ point : ℕ, point_good point :=
sorry

end at_least_one_good_point_l366_366965


namespace num_pos_divisors_36_l366_366079

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366079


namespace number_of_elements_in_list_l366_366047

-- Define the arithmetic sequence
def is_arithmetic_sequence (seq : List ℝ) : Prop :=
  ∀ n : ℕ, n < seq.length - 1 → seq[n + 1] - seq[n] = -5

-- Define the given list
def given_list := [165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45]

-- The problem statement
theorem number_of_elements_in_list : 
    is_arithmetic_sequence given_list → 
    given_list.length = 25 :=
by
  sorry

end number_of_elements_in_list_l366_366047


namespace remainder_when_divided_by_4x_minus_8_l366_366366

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l366_366366


namespace buratino_coins_l366_366160

theorem buratino_coins (a0 : ℕ) (n k : ℕ) (final_coins : ℕ) :
  a0 = 3 ∧ (∀ i, i ∈ finset.range (99 - 1) → a0 + i * 3 + (if i = n then a0 + n * 3 else 0)) = final_coins ∧
  final_coins = 456 → false := 
by 
  sorry

end buratino_coins_l366_366160


namespace copper_needed_l366_366863

theorem copper_needed (T : ℝ) (lead_percentage : ℝ) (lead_weight : ℝ) (copper_percentage : ℝ) 
  (h_lead_percentage : lead_percentage = 0.25)
  (h_lead_weight : lead_weight = 5)
  (h_copper_percentage : copper_percentage = 0.60)
  (h_total_weight : T = lead_weight / lead_percentage) :
  copper_percentage * T = 12 := 
by
  sorry

end copper_needed_l366_366863


namespace p_iff_q_l366_366957

variables {a b c : ℝ}
def p (a b c : ℝ) : Prop := ∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0
def q (a b c : ℝ) : Prop := a + b + c = 0

theorem p_iff_q (h : a ≠ 0) : p a b c ↔ q a b c :=
sorry

end p_iff_q_l366_366957


namespace michaels_brother_final_amount_l366_366659

theorem michaels_brother_final_amount :
  ∀ (michael_money michael_brother_initial michael_give_half candy_cost money_left : ℕ),
  michael_money = 42 →
  michael_brother_initial = 17 →
  michael_give_half = michael_money / 2 →
  let michael_brother_total := michael_brother_initial + michael_give_half in
  candy_cost = 3 →
  money_left = michael_brother_total - candy_cost →
  money_left = 35 :=
by
  intros michael_money michael_brother_initial michael_give_half candy_cost money_left
  intros h1 h2 h3 michael_brother_total h4 h5
  sorry

end michaels_brother_final_amount_l366_366659


namespace sum_of_initial_N_where_output_is_4_after_six_steps_l366_366374

def modifiedMachine (N : ℕ) : ℕ :=
  if N % 2 = 0 then N / 2 else 5 * N + 2

def applyMachineNTimes (N : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => N
  | n + 1 => applyMachineNTimes (modifiedMachine N) n

theorem sum_of_initial_N_where_output_is_4_after_six_steps :
  (∃ N, applyMachineNTimes N 6 = 4) ∧
  (∀ N1 N2, applyMachineNTimes N1 6 = 4 ∧ applyMachineNTimes N2 6 = 4 → N1 = N2 ∨ N1 + N2 = 256) :=
begin
  sorry
end

end sum_of_initial_N_where_output_is_4_after_six_steps_l366_366374


namespace solve_exp_equation_l366_366686

theorem solve_exp_equation (x : ℝ) (h : 9 = 3^2) (h2: 81 = 3^4) :
  3^x * 9^x = 81^(x - 20) → x = 80 :=
by
  sorry

end solve_exp_equation_l366_366686


namespace root_relation_l366_366032

noncomputable def f (x : ℝ) : ℝ := x + (2 : ℝ) ^ x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x

lemma root_f_x1 (x1 : ℝ) (h1 : f x1 = 0) : 2 ^ x1 + x1 = 0 := h1
lemma root_g_x2 (x2 : ℝ) (h2 : g x2 = 0) : g x2 = 0 := h2

theorem root_relation (x1 x2 : ℝ) (h1 : f x1 = 0) (h2 : g x2 = 0) : x1 < x2 :=
sorry

end root_relation_l366_366032


namespace problem_proof_l366_366861

-- Define the problem parameters
variables
  {A B C D : Type}
  {dist_AB dist_DC : ℝ}
  {radius : ℝ}

-- Conditions given in the problem
axioms 
  (h1 : dist_AB = 20) -- Distance AB is 20 feet.
  (h2 : dist_DC = 12) -- Distance DC is 12 feet.
  (h3 : let AD := dist_AB / 2 in AD = 10) -- D is the midpoint of AB, so AD = 10 feet.
  (h4 : radius = Real.sqrt(AD^2 + dist_DC^2)) -- radius is derived from the Pythagorean theorem.

-- The final proof problem
theorem problem_proof :
  let area := π * radius^2 in
  let circumference := 2 * π * radius in
  area = 244 * π ∧ circumference = 2 * π * Real.sqrt(244) :=
begin
  -- The problem hypothesis and axioms will be used here. 
  sorry
end

end problem_proof_l366_366861


namespace LindasCandies_l366_366249

-- Define the problem conditions and the proof goal
theorem LindasCandies (ChloeCandies : ℕ) (TotalCandies : ℕ) (h1 : ChloeCandies = 28) (h2 : TotalCandies = 62) :
  ∃ LindaCandies : ℕ, LindaCandies = TotalCandies - ChloeCandies ∧ LindaCandies = 34 :=
by
  -- Use the given conditions to determine Linda's candies
  let LindaCandies := TotalCandies - ChloeCandies
  use LindaCandies
  split
  · -- Prove LindaCandies = TotalCandies - ChloeCandies
    rfl
  · -- Prove LindaCandies = 34, using h1 and h2
    rw [h1, h2]
    rfl

end LindasCandies_l366_366249


namespace equilateral_triangle_perimeter_l366_366219

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter_l366_366219


namespace lcm_18_24_l366_366798

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l366_366798


namespace lcm_18_24_eq_72_l366_366778

-- Define the given integers
def a : ℕ := 18
def b : ℕ := 24

-- Define the least common multiple function (LCM)
def lcm (x y : ℕ) : ℕ := x * y / Nat.gcd x y

-- Define the proof statement of the problem, checking if LCM of 18 and 24 is 72
theorem lcm_18_24_eq_72 : lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366778


namespace num_pos_divisors_36_l366_366070

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366070


namespace students_arrival_probability_l366_366342

theorem students_arrival_probability 
  (d e f : ℕ)
  (h1 : n = d - e * Real.sqrt f)
  (h2 : ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ f)
  (h3 : 0.3 = 1 - (2 * (60 - n)^2 / 3600)) 
  : d + e + f = 112
sorry

end students_arrival_probability_l366_366342


namespace radius_of_bowling_ball_l366_366849

-- Definition of the conditions and the question
def density : ℝ := 0.3 -- density in pounds per cubic inch
def weight : ℝ := 16 -- weight in pounds

-- The volume of a sphere formula
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- The main theorem to be proved
theorem radius_of_bowling_ball (ρ : ℝ) (W : ℝ) (hρ : ρ = density) (hW : W = weight) :
  ∃ r : ℝ, r = Real.cbrt (40 / Real.pi) ∧ W = ρ * volume r := 
by
  sorry  -- proof placeholder

end radius_of_bowling_ball_l366_366849


namespace number_of_passwords_l366_366412

def password_spec (password : String) : Prop :=
  (password.length = 9) ∧
  (password.count 'A' = 3) ∧
  (password.count 'a' = 1) ∧
  (password.count 'b' = 1) ∧
  (password.filter Char.isDigit).length = 4 ∧
  (∏ d in password.filter Char.isDigit, Char.toNat d - 48 = 6) ∧  -- "Char.toNat d - 48" converts char digit to actual number
  (∀ i j, i < password.length ∧ j < password.length → 
    (password.get ⟨i, _⟩ = 'A' ∧ password.get ⟨j, _⟩ = 'A' → abs (i - j) > 1)) ∧
  (∀ i j, i < password.length ∧ j < password.length → 
    (password.get ⟨i, _⟩ = 'a' ∧ password.get ⟨j, _⟩ = 'b' → abs (i - j) > 1))

theorem number_of_passwords : ∃ n, n = 13600 ∧
  (∀ s, password_spec s → num_passwords s = n) :=
sorry

end number_of_passwords_l366_366412


namespace red_section_no_damage_probability_l366_366458

noncomputable def probability_no_damage (n : ℕ) (p q : ℚ) : ℚ :=
  (q^n : ℚ)

theorem red_section_no_damage_probability :
  probability_no_damage 7 (2/7) (5/7) = (5/7)^7 :=
by
  simp [probability_no_damage]

end red_section_no_damage_probability_l366_366458


namespace fewest_tiles_to_cover_region_l366_366870

namespace TileCoverage

def tile_width : ℕ := 2
def tile_length : ℕ := 6
def region_width_feet : ℕ := 3
def region_length_feet : ℕ := 4

def region_width_inches : ℕ := region_width_feet * 12
def region_length_inches : ℕ := region_length_feet * 12

def region_area : ℕ := region_width_inches * region_length_inches
def tile_area : ℕ := tile_width * tile_length

def fewest_tiles_needed : ℕ := region_area / tile_area

theorem fewest_tiles_to_cover_region :
  fewest_tiles_needed = 144 :=
sorry

end TileCoverage

end fewest_tiles_to_cover_region_l366_366870


namespace x_is_percent_z_l366_366386

-- Definitions from the conditions
variables {x y z : ℝ}

-- Condition 1: x is 1.30 times y
def cond1 : Prop := x = 1.30 * y

-- Condition 2: y is 0.60 times z
def cond2 : Prop := y = 0.60 * z

-- Proof statement: x is 0.78 times z
theorem x_is_percent_z (h1 : cond1) (h2 : cond2) : x = 0.78 * z :=
by
  sorry

end x_is_percent_z_l366_366386


namespace height_above_center_of_pentagon_l366_366554

/-- 
Given a regular pentagon $ABCDE$ with each side length of 1,
there exists a point $M$ such that the line $AM$ is perpendicular
to the plane $CDM$. Prove that the height of the point $M$ above
the center $O$ of the pentagon is $\frac{\sqrt{3\sqrt{5}+5}}{5}$.
--/

theorem height_above_center_of_pentagon 
  (A B C D E M O F : Point) 
  (hPentagon : regular_pentagon A B C D E)
  (hSideLength : ∀ (P Q : Point), (P, Q) ∈ edges_pentagon → dist P Q = 1)
  (hMidpointF : midpoint F C D)
  (hPerpendicular : ∀ (Q : Point),  
     Q ∈ {P | P ∈ plane O C D → dist A M ⟂ plane O C D}) 
  : height_above_center O M = real_approx 0.76512 := 
sorry

end height_above_center_of_pentagon_l366_366554


namespace max_parts_crescent_moon_l366_366224

theorem max_parts_crescent_moon (n : ℕ) (h : n = 5) : 
  let P := (n^2 + 3 * n) / 2 + 1 in 
  P = 21 := by
  rw [h]
  let P := (5^2 + 3 * 5) / 2 + 1
  sorry

end max_parts_crescent_moon_l366_366224


namespace rectangle_area_l366_366191

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l366_366191


namespace remaining_yellow_marbles_l366_366282

def original_yellow_marbles : ℕ := 86
def taken_yellow_marbles : ℕ := (35 * original_yellow_marbles) / 100

theorem remaining_yellow_marbles : original_yellow_marbles - taken_yellow_marbles = 56 :=
by
  have taken_yellow_marbles := 30
  calc
    86 - 30 = 56 : by norm_num

end remaining_yellow_marbles_l366_366282


namespace f_f_neg3_eq_zero_l366_366542

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 1 then x + (2 / x) - 3 else real.log (x^2 + 1)

theorem f_f_neg3_eq_zero : f (f (-3)) = 0 := 
sorry

end f_f_neg3_eq_zero_l366_366542


namespace julia_picnic_meals_l366_366616

theorem julia_picnic_meals :
  let sandwiches := 4
  let salads := 5
  let choose_salads := Nat.choose salads 3
  let drinks := 3
  sandwiches * choose_salads * drinks = 120 := by
    let sandwiches := 4
    let salads := 5
    let choose_salads := Nat.choose salads 3
    let drinks := 3
    have h1 : sandwiches * choose_salads * drinks = 4 * 10 * 3 := by
      simp [Nat.choose, sandwiches, salads, drinks]
    have h2 : 4 * 10 * 3 = 120 := rfl
    exact eq.trans h1 h2

end julia_picnic_meals_l366_366616


namespace geometric_arithmetic_sequences_l366_366975

-- Problem statement translated into Lean 4:

theorem geometric_arithmetic_sequences :
  (∃ a : ℕ → ℝ, 
    (∀ n, a n = 2 ^ (n-1)) ∧ 
    (a 1 * a 2 * a 3 = 8) ∧ 
    (2 * (a 2 + 2) = (a 1 + 1) + (a 3 + 2))) → 
  (∃ T : ℕ → ℝ, 
    (∀ n, 
      T n = (finset.range n).sum (λ k, 2 ^ (k) + 2 * (k + 1)) ∧
      T n = 2^n + n^2 + n - 1)) :=
by 
  sorry

end geometric_arithmetic_sequences_l366_366975


namespace economy_value_after_two_years_l366_366170

/--
Given an initial amount A₀ = 3200,
that increases annually by 1/8th of itself,
with an inflation rate of 3% in the first year and 4% in the second year,
prove that the value of the amount after two years is 3771.36
-/
theorem economy_value_after_two_years :
  let A₀ := 3200 
  let increase_rate := 1 / 8
  let inflation_rate_year_1 := 0.03
  let inflation_rate_year_2 := 0.04
  let A₁ := A₀ * (1 + increase_rate)
  let V₁ := A₁ * (1 - inflation_rate_year_1)
  let A₂ := V₁ * (1 + increase_rate)
  let V₂ := A₂ * (1 - inflation_rate_year_2)
  V₂ = 3771.36 :=
by
  simp only []
  sorry

end economy_value_after_two_years_l366_366170


namespace exists_n_good_not_nplus1_good_l366_366637

def S (k : ℕ) : ℕ := k.digits.sum -- sum of the digits of k

def is_n_good (n a : ℕ) : Prop :=
  ∃ a_seq : ℕ → ℕ, 
    (∀ i, i < n → a_seq i - S (a_seq i) = a_seq (i + 1)) ∧ 
    a_seq n = a

theorem exists_n_good_not_nplus1_good (n : ℕ) : 
  ∃ a : ℕ, is_n_good n a ∧ ¬ is_n_good (n + 1) a :=
by
  sorry

end exists_n_good_not_nplus1_good_l366_366637


namespace num_pos_divisors_36_l366_366128

theorem num_pos_divisors_36 : 
  let n := 36 in
  (count_divisors n = 9) := 
by
  let prime_factors := [(2, 2), (3, 2)]
  let number_of_divisors := (prime_factors.map (λ p => p.2 + 1)).prod
  have h : 36 = (2^2) * (3^2) := by norm_num
  have num_div : number_of_divisors = 9 := by norm_num
  sorry

end num_pos_divisors_36_l366_366128


namespace average_weight_increase_l366_366375

theorem average_weight_increase
  (N : ℕ)
  (w_left : ℝ)
  (remaining_students : ℕ)
  (avg_weight_remaining : ℝ)
  (total_weight_initial : total_weight_initial = (N * avg_weight_initial))
  (total_weight_remaining : total_weight_remaining = (N * avg_weight_initial) - w_left) :
  avg_weight_remaining = total_weight_remaining / (N-1) → 
  increase_in_avg_weight = (avg_weight_remaining - (total_weight_initial / N)) :=
begin
  sorry
end

end average_weight_increase_l366_366375


namespace compute_a_b_c_l366_366237

noncomputable def root1 (w : ℂ) : ℂ := w + 2 * complex.I
noncomputable def root2 (w : ℂ) : ℂ := w + 7 * complex.I
noncomputable def root3 (w : ℂ) : ℂ := 2 * w - 3

theorem compute_a_b_c (a b c : ℝ) (w : ℂ) 
  (h_w : w = 3 - (9/4) * complex.I) :
  (P : ℂ → ℂ) = λ z, z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) ∧
  (root1 w = w + 2 * complex.I) ∧
  (root2 w = w + 7 * complex.I) ∧
  (root3 w = 2 * w - 3) ∧
  (P(root1 w) = 0) ∧ 
  (P(root2 w) = 0) ∧ 
  (P(root3 w) = 0) →
  a + b + c = /* the calculated value */
  sorry

end compute_a_b_c_l366_366237


namespace num_pos_divisors_36_l366_366076

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366076


namespace one_thirds_in_fraction_l366_366564

theorem one_thirds_in_fraction : (11 / 5) / (1 / 3) = 33 / 5 := by
  sorry

end one_thirds_in_fraction_l366_366564


namespace lcm_18_24_l366_366802

open Nat

/-- The least common multiple of two numbers a and b -/
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

theorem lcm_18_24 : lcm 18 24 = 72 := 
by
  sorry

end lcm_18_24_l366_366802


namespace num_pos_divisors_36_l366_366078

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l366_366078


namespace rounding_to_nearest_hundredth_l366_366671

-- Definitions
def number_to_round : ℝ := 3.456
def correct_answer : ℝ := 3.46

-- Statement
theorem rounding_to_nearest_hundredth : round_to_nearest_hundredth number_to_round = correct_answer :=
by sorry

end rounding_to_nearest_hundredth_l366_366671


namespace probability_not_miss_is_correct_l366_366713

-- Define the probability that Peter will miss his morning train
def p_miss : ℚ := 5 / 12

-- Define the probability that Peter does not miss his morning train
def p_not_miss : ℚ := 1 - p_miss

-- The theorem to prove
theorem probability_not_miss_is_correct : p_not_miss = 7 / 12 :=
by
  -- Proof omitted
  sorry

end probability_not_miss_is_correct_l366_366713


namespace min_value_of_S6_l366_366967

variable {a_1 q : ℝ} (S : ℕ → ℝ)

-- Given conditions
def is_geometric_sequence (S : ℕ → ℝ) (a_1 q : ℝ) : Prop :=
  ∀ n, S n = a_1 * (1 - q ^ n) / (1 - q)

def q_greater_than_one (q : ℝ) : Prop := q > 1

axiom S4_eq_2S2_plus_1 (S : ℕ → ℝ) (a_1 q : ℝ) : 
  is_geometric_sequence S a_1 q → S 4 = 2 * S 2 + 1

-- The proof statement
theorem min_value_of_S6 (S : ℕ → ℝ) (a_1 q : ℝ) :
  is_geometric_sequence S a_1 q →
  q_greater_than_one q →
  S4_eq_2S2_plus_1 S a_1 q →
  ∃ q, S 6 = 2 * Real.sqrt 3 + 3 :=
sorry

end min_value_of_S6_l366_366967


namespace Vikki_take_home_pay_is_correct_l366_366732

noncomputable def Vikki_take_home_pay : ℝ :=
  let hours_worked : ℝ := 42
  let hourly_pay_rate : ℝ := 12
  let gross_earnings : ℝ := hours_worked * hourly_pay_rate

  let fed_tax_first_300 : ℝ := 300 * 0.15
  let amount_over_300 : ℝ := gross_earnings - 300
  let fed_tax_excess : ℝ := amount_over_300 * 0.22
  let total_federal_tax : ℝ := fed_tax_first_300 + fed_tax_excess

  let state_tax : ℝ := gross_earnings * 0.07
  let retirement_contribution : ℝ := gross_earnings * 0.06
  let insurance_cover : ℝ := gross_earnings * 0.03
  let union_dues : ℝ := 5

  let total_deductions : ℝ := total_federal_tax + state_tax + retirement_contribution + insurance_cover + union_dues
  let take_home_pay : ℝ := gross_earnings - total_deductions
  take_home_pay

theorem Vikki_take_home_pay_is_correct : Vikki_take_home_pay = 328.48 :=
by
  sorry

end Vikki_take_home_pay_is_correct_l366_366732


namespace total_distance_proof_l366_366303

-- Define the conditions
def first_half_time := 20
def second_half_time := 30
def average_time_per_kilometer := 5

-- Calculate the total time
def total_time := first_half_time + second_half_time

-- State the proof problem: prove that the total distance is 10 kilometers
theorem total_distance_proof : 
  (total_time / average_time_per_kilometer) = 10 :=
  by sorry

end total_distance_proof_l366_366303


namespace vector_norm_problem_l366_366035

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_norm_problem 
  (a b : V)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ∥2 • a - b∥ = real.sqrt 5) :
  ∥a + 2 • b∥ = real.sqrt 5 :=
sorry

end vector_norm_problem_l366_366035


namespace exponential_inequality_l366_366343

theorem exponential_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end exponential_inequality_l366_366343


namespace num_pos_divisors_36_l366_366132

theorem num_pos_divisors_36 : (Nat.divisors 36).length = 9 := sorry

end num_pos_divisors_36_l366_366132


namespace scenario1_distribution_scenario2_distribution_l366_366875

-- Scenario 1 Conditions
def scenario1_lathes : Nat := 3
def scenario1_failure_prob : ℚ := 0.2

-- Scenario 1 Expected Distribution
def scenario1_p0 : ℚ := 64 / 125
def scenario1_p1 : ℚ := 48 / 125
def scenario1_p2 : ℚ := 12 / 125
def scenario1_p3 : ℚ := 1 / 125

-- Scenario 1 Proof Statement
theorem scenario1_distribution :
    (CalcProbOfX scenario1_lathes scenario1_failure_prob 0 = scenario1_p0) ∧
    (CalcProbOfX scenario1_lathes scenario1_failure_prob 1 = scenario1_p1) ∧
    (CalcProbOfX scenario1_lathes scenario1_failure_prob 2 = scenario1_p2) ∧
    (CalcProbOfX scenario1_lathes scenario1_failure_prob 3 = scenario1_p3) := 
sorry

-- Scenario 2 Conditions
def scenario2_latheA_count : Nat := 2
def scenario2_latheB_count : Nat := 1
def scenario2_failure_prob_A : ℚ := 0.1
def scenario2_failure_prob_B : ℚ := 0.2

-- Scenario 2 Expected Distribution
def scenario2_p0 : ℚ := 0.648
def scenario2_p1 : ℚ := 0.306
def scenario2_p2 : ℚ := 0.044
def scenario2_p3 : ℚ := 0.002

-- Scenario 2 Proof Statement
theorem scenario2_distribution :
    (CalcProbOfX_Scenario2 scenario2_latheA_count scenario2_latheB_count scenario2_failure_prob_A scenario2_failure_prob_B 0 = scenario2_p0) ∧
    (CalcProbOfX_Scenario2 scenario2_latheA_count scenario2_latheB_count scenario2_failure_prob_A scenario2_failure_prob_B 1 = scenario2_p1) ∧
    (CalcProbOfX_Scenario2 scenario2_latheA_count scenario2_latheB_count scenario2_failure_prob_A scenario2_failure_prob_B 2 = scenario2_p2) ∧
    (CalcProbOfX_Scenario2 scenario2_latheA_count scenario2_latheB_count scenario2_failure_prob_A scenario2_failure_prob_B 3 = scenario2_p3) := 
sorry

end scenario1_distribution_scenario2_distribution_l366_366875


namespace find_balls_l366_366334

-- Define the variables for the number of red, yellow, and white balls
variables (x y z : ℚ)

-- State the conditions as hypotheses
def conditions (x y z : ℚ) :=
  x + y + z = 160 ∧ 
  x - (x / 3) + y - (y / 4) + z - (z / 5) = 120 ∧ 
  x - (x / 5) + y - (y / 4) + z - (z / 3) = 116

-- The theorem should state that the number of each colored ball can be found
theorem find_balls (x y z : ℚ) (h : conditions x y z) : x = 45 ∧ y = 40 ∧ z = 75 :=
by {
  have h1 : x + y + z = 160 := h.1,
  have h2 : x - (x / 3) + y - (y / 4) + z - (z / 5) = 120 := h.2,
  have h3 : x - (x / 5) + y - (y / 4) + z - (z / 3) = 116 := h.3,
  -- Normally the proof would continue from here
  sorry
}

end find_balls_l366_366334


namespace smallest_n_for_cookies_l366_366610

theorem smallest_n_for_cookies :
  ∃ n : ℕ, 15 * n - 1 % 11 = 0 ∧ (∀ m : ℕ, 15 * m - 1 % 11 = 0 → n ≤ m) :=
sorry

end smallest_n_for_cookies_l366_366610


namespace right_triangle_of_ratio_and_right_angle_l366_366716

-- Define the sides and the right angle condition based on the problem conditions
variable (x : ℝ) (hx : 0 < x)

-- Variables for the sides in the given ratio
def a := 3 * x
def b := 4 * x
def c := 5 * x

-- The proposition we need to prove
theorem right_triangle_of_ratio_and_right_angle (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by sorry  -- Proof not required as per instructions

end right_triangle_of_ratio_and_right_angle_l366_366716


namespace cyclic_quadrilateral_of_tangent_circles_l366_366523

theorem cyclic_quadrilateral_of_tangent_circles 
  (Γ1 Γ2 Γ3 Γ4 : Circle) (A B C D : Point) 
  (O1 O2 O3 O4 : Point)
  (h1 : tangent_at_point Γ1 Γ2 A)
  (h2 : tangent_at_point Γ2 Γ3 B)
  (h3 : tangent_at_point Γ3 Γ4 C)
  (h4 : tangent_at_point Γ4 Γ1 D)
  (hO1 : center_of Γ1 = O1)
  (hO2 : center_of Γ2 = O2)
  (hO3 : center_of Γ3 = O3)
  (hO4 : center_of Γ4 = O4)
  (r1 r2 r3 r4 : Real)
  (h_dist1 : dist O1 O2 = r1 + r2)
  (h_dist2 : dist O2 O3 = r2 + r3)
  (h_dist3 : dist O3 O4 = r3 + r4)
  (h_dist4 : dist O4 O1 = r4 + r1) :
  cyclic_quadrilateral A B C D :=
sorry

end cyclic_quadrilateral_of_tangent_circles_l366_366523


namespace find_sports_package_channels_l366_366617

-- Defining the conditions
def initial_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def reduce_package_by : ℕ := 10
def supreme_sports_package : ℕ := 7
def final_channels : ℕ := 147

-- Defining the situation before the final step
def channels_after_reduction := initial_channels - channels_taken_away + channels_replaced - reduce_package_by
def channels_after_supreme := channels_after_reduction + supreme_sports_package

-- Prove the original sports package added 8 channels
theorem find_sports_package_channels : ∀ sports_package_added : ℕ,
  sports_package_added + channels_after_supreme = final_channels → sports_package_added = 8 :=
by
  intro sports_package_added
  intro h
  sorry

end find_sports_package_channels_l366_366617


namespace probability_one_piggy_bank_opened_probability_two_piggy_banks_opened_l366_366734

theorem probability_one_piggy_bank_opened : 
  (1:ℚ) / 30 = 1 / 30 :=
begin
  sorry
end

theorem probability_two_piggy_banks_opened : 
  (1:ℚ) / 15 = 1 / 15 :=
begin
  sorry
end

end probability_one_piggy_bank_opened_probability_two_piggy_banks_opened_l366_366734


namespace optimal_strategies_and_value_l366_366340

-- Define the payoff matrix for the two-player zero-sum game
def payoff_matrix : Matrix (Fin 2) (Fin 2) ℕ := ![![12, 22], ![32, 2]]

-- Define the optimal mixed strategies for both players
def optimal_strategy_row_player : Fin 2 → ℚ
| 0 => 3 / 4
| 1 => 1 / 4

def optimal_strategy_column_player : Fin 2 → ℚ
| 0 => 1 / 2
| 1 => 1 / 2

-- Define the value of the game
def value_of_game := (17 : ℚ)

theorem optimal_strategies_and_value :
  (∀ i j, (optimal_strategy_row_player 0 * payoff_matrix 0 j + optimal_strategy_row_player 1 * payoff_matrix 1 j = value_of_game) ∧
           (optimal_strategy_column_player 0 * payoff_matrix i 0 + optimal_strategy_column_player 1 * payoff_matrix i 1 = value_of_game)) :=
by 
  -- sorry is used as a placeholder for the proof
  sorry

end optimal_strategies_and_value_l366_366340


namespace book_pages_l366_366226

noncomputable def totalPages := 240

theorem book_pages : 
  ∀ P : ℕ, 
    (1 / 2) * P + (1 / 4) * P + (1 / 6) * P + 20 = P → 
    P = totalPages :=
by
  intro P
  intros h
  sorry

end book_pages_l366_366226


namespace paper_fold_ratio_l366_366418

theorem paper_fold_ratio (w : ℝ) (hw : 0 < w) :
  let A := 2 * w^2 in
  let B := (1 + Real.sqrt 2) / 2 in
  B / A = (1 + Real.sqrt 2) / 4 :=
by
  sorry

end paper_fold_ratio_l366_366418


namespace inequality_proof_l366_366953

theorem inequality_proof (a b : Real) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_proof_l366_366953


namespace inequality_solution_l366_366287

theorem inequality_solution (x : ℝ) : (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 :=
by
  sorry

end inequality_solution_l366_366287


namespace no_position_swap_l366_366522

def lattice_point := ℂ

def adjacent (A B : lattice_point) : Prop :=
dist A B = 1

def jump (A : lattice_point) (d : lattice_point) : lattice_point :=
A + d

def frog_moves (A B : lattice_point) : list (lattice_point × lattice_point) :=
sorry -- This represents the sequence of jumps according to the rules

noncomputable def finite_moves (A B : lattice_point) :=
∃ (n : ℕ), ∃ (moves : fin n → lattice_point × lattice_point), moves = frog_moves A B

theorem no_position_swap :
∀ (A B : lattice_point),
adjacent A B →
¬ finite_moves A B ∧ (frogs swap positions after finite moves) :=
begin
  sorry
end

end no_position_swap_l366_366522


namespace num_pos_divisors_36_l366_366109

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ (d : ℕ), d ∣ 36 → 1 ≤ d ∧ d ≤ 36 → list.mem d [1, 2, 3, 4, 6, 9, 12, 18, 36]) :=
by
  sorry

end num_pos_divisors_36_l366_366109


namespace number_of_factors_l366_366914

-- Define the initial conditions of the mathematical problem.
def num := 8^2 * 9^3 * 7^5

-- Assertion that we aim to prove (statement only, no proof).
theorem number_of_factors : 
  nat.factors.num.count = 294 := 
sorry

end number_of_factors_l366_366914


namespace num_pos_divisors_36_l366_366065

def prime_factorization (n : ℕ) : list (ℕ × ℕ) := sorry -- Placeholder for prime factorization function

def number_of_divisors (factors : list (ℕ × ℕ)) : ℕ :=
  factors.foldr (λ (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem num_pos_divisors_36 : number_of_divisors [(2, 2), (3, 2)] = 9 :=
by sorry

end num_pos_divisors_36_l366_366065


namespace correct_options_l366_366497

variable (data_A : List ℕ := [3, 5, 6, 7, 7, 8, 8, 8, 9, 10])
variable (data_B : List ℕ := [4, 6, 6, 7, 8, 8, 8, 8, 8, 8])

-- Definition of mode, mean, range, and variance 
def mode (lst : List ℕ) : ℕ :=
(lst.groupBy id).values.map List.length.maximumBy (<).getOrElse 0

def mean (lst : List ℕ) : ℚ :=
(lst.map (λ x => (x : ℚ)).sum / lst.length)

def range (lst : List ℕ) : ℕ :=
lst.maximumBy (<).getOrElse 0 - lst.minimumBy (<).getOrElse 0

def variance (lst : List ℕ) : ℚ :=
let mean_val := mean lst
(lst.map (λ x => (x : ℚ - mean_val) * (x : ℚ - mean_val)).sum / lst.length)

-- Values for Factory A
def x1 := mode data_A
def x2 := mean data_A
def x3 := range data_A
def x4 := variance data_A

-- Values for Factory B
def y1 := mode data_B
def y2 := mean data_B
def y3 := range data_B
def y4 := variance data_B

theorem correct_options : set (ℚ × ℚ × ℕ × ℚ) := {| 
    (x2, y2, x4, y4) | x2 = y2 ∧ x4 > y4 
|}.elem ∅ :=
by
  sorry

end correct_options_l366_366497


namespace cards_to_flip_l366_366719

def card_has_vowel (card : char) : Prop :=
  card = 'A' ∨ card = 'E' ∨ card = 'I' ∨ card = 'O' ∨ card = 'U'

def card_has_even_number (card : char) : Prop :=
  card = '2' ∨ card = '4' ∨ card = '6' ∨ card = '8' ∨ card = '0'

def flip_condition (card : char) : Prop :=
  (card_has_vowel card ∧ ¬(card_has_even_number card)) ∨
  (¬card_has_even_number card ∧ ¬card_has_vowel card)

theorem cards_to_flip (cards : List char) (A, B, four, five : char) : 
  (card_has_vowel A ∧ ¬(card_has_even_number four)) ∧
  (¬(card_has_eve_number four) ∧ ¬card_has_vowel five) :=
  sorry

end cards_to_flip_l366_366719


namespace quadratic_function_properties_l366_366267

noncomputable def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 :=
by
  sorry

end quadratic_function_properties_l366_366267


namespace selection_count_with_gender_constraint_l366_366496

theorem selection_count_with_gender_constraint (m f : ℕ) (total_students : ℕ) (k : ℕ)
    (hm : m = 4) (hf : f = 6) (htotal : total_students = m + f) (hk : k = 3) : 
    ∃ (ways : ℕ), ways = 96 := by
  have total_ways := Nat.choose total_students k
  have male_only_ways := Nat.choose m k
  have female_only_ways := Nat.choose f k
  let mixed_ways := total_ways - male_only_ways - female_only_ways
  use mixed_ways
  rw [hm, hf, htotal, hk]
  norm_num
  have total_ways := (10.choose 3)
  have male_only_ways := (4.choose 3)
  have female_only_ways := (6.choose 3)
  norm_num
  exact congr_arg ((· - ·) total_ways) (male_only_ways + female_only_ways)
  norm_num
  exact 96

end selection_count_with_gender_constraint_l366_366496


namespace lcm_18_24_eq_72_l366_366766

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end lcm_18_24_eq_72_l366_366766


namespace setA_times_setB_equals_desired_l366_366144

def setA : Set ℝ := { x | abs (x - 1/2) < 1 }
def setB : Set ℝ := { x | 1/x ≥ 1 }
def setAB : Set ℝ := { x | (x ∈ setA ∪ setB) ∧ (x ∉ setA ∩ setB) }

theorem setA_times_setB_equals_desired :
  setAB = { x | (-1/2 < x ∧ x ≤ 0) ∨ (1 < x ∧ x < 3/2) } :=
by
  sorry

end setA_times_setB_equals_desired_l366_366144


namespace remainder_3001_3005_mod_17_l366_366353

theorem remainder_3001_3005_mod_17 :
  ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17 = 2 := by
  have h1 : 3001 % 17 = 10 := by norm_num
  have h2 : 3002 % 17 = 11 := by norm_num
  have h3 : 3003 % 17 = 12 := by norm_num
  have h4 : 3004 % 17 = 13 := by norm_num
  have h5 : 3005 % 17 = 14 := by norm_num
  calc
    ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17
      = (10 * 11 * 12 * 13 * 14) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 2 : by norm_num

end remainder_3001_3005_mod_17_l366_366353


namespace episodes_lost_per_season_l366_366450

theorem episodes_lost_per_season (s1 s2 : ℕ) (e : ℕ) (remaining : ℕ) (total_seasons : ℕ) (total_episodes_before : ℕ) (total_episodes_lost : ℕ)
  (h1 : s1 = 12) (h2 : s2 = 14) (h3 : e = 16) (h4 : remaining = 364) 
  (h5 : total_seasons = s1 + s2) (h6 : total_episodes_before = s1 * e + s2 * e) 
  (h7 : total_episodes_lost = total_episodes_before - remaining) :
  total_episodes_lost / total_seasons = 2 := by
  sorry

end episodes_lost_per_season_l366_366450


namespace prove_min_value_inequality_l366_366633

noncomputable def min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 9) : Prop :=
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9

theorem prove_min_value_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 9) :
  min_value_inequality x y z h1 h2 h3 h4 :=
begin
  sorry
end

end prove_min_value_inequality_l366_366633


namespace multiples_of_15_between_25_and_200_l366_366561

theorem multiples_of_15_between_25_and_200 : 
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  in multiples.length = 12 :=
by
  let multiples := list.filter (λ n, 25 < n ∧ n < 200) (list.map (λ n, 15 * n) (list.range 14))
  show multiples.length = 12
  sorry

end multiples_of_15_between_25_and_200_l366_366561
