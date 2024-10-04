import Mathlib

namespace mean_of_first_element_l401_401115

def harmonic_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (k + 1 : ℚ))

def S (n : ℕ) : Finset (Fin n → ℕ) :=
  {σ ∈ Finset.permutations_of_fin n | ∃ i, i > 0 ∧ ∀ j < i, σ j < σ i ∧ ∀ k > 0, σ 0 = k → k < i}

theorem mean_of_first_element (n : ℕ) : 
  ∑ σ in S n, σ 0 / (S n).card = n - (n - 1 : ℚ) / harmonic_sum (n - 1)
:= by
  sorry

end mean_of_first_element_l401_401115


namespace cosine_periodicity_l401_401337

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401337


namespace marys_next_birthday_age_l401_401495

variable (m s d : ℝ)

-- Condition 1: Mary is 30% older than Sally
def mary_older_than_sally := m = 1.3 * s

-- Condition 2: Sally is 50% younger than Danielle
def sally_younger_than_danielle := s = 0.5 * d

-- Condition 3: The sum of their ages is 45 years
def sum_of_ages := m + s + d = 45

-- Proof statement: Prove that Mary's age on her next birthday is 14
theorem marys_next_birthday_age (h1 : mary_older_than_sally m s)
                               (h2 : sally_younger_than_danielle s d)
                               (h3 : sum_of_ages m s d) :
  m + 1 = 14 :=
  Sorry

end marys_next_birthday_age_l401_401495


namespace alice_height_after_growth_l401_401264

/-- Conditions: Bob and Alice were initially the same height. Bob has grown by 25%, Alice 
has grown by one third as many inches as Bob, and Bob is now 75 inches tall. --/
theorem alice_height_after_growth (initial_height : ℕ)
  (bob_growth_rate : ℚ)
  (alice_growth_ratio : ℚ)
  (bob_final_height : ℕ) :
  bob_growth_rate = 0.25 →
  alice_growth_ratio = 1 / 3 →
  bob_final_height = 75 →
  initial_height + (bob_final_height - initial_height) / 3 = 65 :=
by
  sorry

end alice_height_after_growth_l401_401264


namespace other_root_correct_l401_401909

noncomputable def other_root (p : ℝ) : ℝ :=
  let a := 3
  let c := -2
  let root1 := -1
  (-c / a) / root1

theorem other_root_correct (p : ℝ) (h_eq : 3 * (-1) ^ 2 + p * (-1) = 2) : other_root p = 2 / 3 :=
  by
    unfold other_root
    sorry

end other_root_correct_l401_401909


namespace hoseok_basketballs_l401_401608

theorem hoseok_basketballs (v s b : ℕ) (h₁ : v = 40) (h₂ : s = v + 18) (h₃ : b = s - 23) : b = 35 := by
  sorry

end hoseok_basketballs_l401_401608


namespace cyclic_quadrilateral_perimeter_l401_401221

-- Lean statement for the problem:
theorem cyclic_quadrilateral_perimeter (A B C D P Q : Type)
  [IsCyclicQuadrilateral A B C D]
  (H1 : Is_Perpendicular (A, C) (B, D))
  (H2 : IntersectsAt (C, P) (B, D) P)
  (H3 : Q ∈ Segment(C, P))
  (H4 : CQ = AP) :
  Perimeter (Triangle (B, D, Q)) ≥ 2 * Length (A, C) :=
sorry

end cyclic_quadrilateral_perimeter_l401_401221


namespace derivative_at_one_l401_401780

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem derivative_at_one : deriv f 1 = -2 :=
by 
  -- Here we would provide the proof that f'(1) = -2
  sorry

end derivative_at_one_l401_401780


namespace find_swimming_speed_l401_401160

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed_l401_401160


namespace roots_squared_evaluation_l401_401158

noncomputable def polynomial_with_squared_roots (a b c : ℝ) : Polynomial ℝ :=
  Polynomial.monicPolynomial (Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2) (λ x, [a^2, b^2, c^2])

theorem roots_squared_evaluation (a b c : ℝ)
  (h1 : Polynomial.aeval a (Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2) = 0)
  (h2 : Polynomial.aeval b (Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2) = 0)
  (h3 : Polynomial.aeval c (Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2) = 0) :
  (polynomial_with_squared_roots a b c).eval 1 = -15 := by
  sorry

end roots_squared_evaluation_l401_401158


namespace original_price_of_sarees_l401_401177

theorem original_price_of_sarees (P : ℝ) (h : 0.92 * 0.90 * P = 331.2) : P = 400 :=
by
  sorry

end original_price_of_sarees_l401_401177


namespace fixed_point_line_passing_l401_401818

theorem fixed_point_line_passing (m : ℝ) : (∃ x y : ℝ, (m+2)*x + (m-3)*y + 4 = 0) := 
begin
  let x := -4/5,
  let y := 4/5,
  use [x, y],
  sorry, -- this is where the proof steps would go, currently skipped
end

end fixed_point_line_passing_l401_401818


namespace geometric_sum_S_40_l401_401062

variable (S : ℕ → ℝ)

-- Conditions
axiom sum_S_10 : S 10 = 18
axiom sum_S_20 : S 20 = 24

-- Proof statement
theorem geometric_sum_S_40 : S 40 = 80 / 3 :=
by
  sorry

end geometric_sum_S_40_l401_401062


namespace find_a3_l401_401079

-- Define the geometric sequence as a function from ℕ to ℝ
def geometric_sequence (a1 : ℝ) (r : ℝ) : ℕ → ℝ := 
  λ n, a1 * r^n

-- Variables
variable {a1 : ℝ}
variable {a4 : ℝ}

-- The given conditions
axiom a1_eq : a1 = 1
axiom a4_eq : geometric_sequence a1 3 3 = 27

-- The target is to prove that a3 = 9
theorem find_a3 : geometric_sequence a1 3 2 = 9 := 
by 
  sorry

end find_a3_l401_401079


namespace unique_solution_of_fraction_eq_l401_401946

theorem unique_solution_of_fraction_eq (x : ℝ) : (1 / (x - 1) = 2 / (x - 2)) ↔ (x = 0) :=
by
  sorry

end unique_solution_of_fraction_eq_l401_401946


namespace quadratic_coefficients_l401_401589

theorem quadratic_coefficients (x : ℝ) :
  let y := x^2 - 4*x + 5 in
  (1, -4, 5) = (1, -4, 5) :=
by
  sorry

end quadratic_coefficients_l401_401589


namespace largest_number_in_ratio_l401_401185

theorem largest_number_in_ratio (x : ℕ) (h : ((4 * x + 5 * x + 6 * x) / 3 : ℝ) = 20) : 6 * x = 24 := 
by 
  sorry

end largest_number_in_ratio_l401_401185


namespace disproving_iff_l401_401740

theorem disproving_iff (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : (a^2 > b^2) ∧ ¬(a > b) :=
by
  sorry

end disproving_iff_l401_401740


namespace arithmetic_sequence_condition_l401_401379

noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q ≠ 1 → a 0 = 1 → (-3 * a 0 : ℝ) = -a 1 ∧ -a 2 = -3 * a 0 + a 3 →
  S 4 = geometric_sum a q 4 →
  q = -3 → S 4 = -20 := 
by
  intros hq ha1 ha1_arith hS hq_solution
  rw [geometric_sum, ha1, hq_solution]
  sorry

end arithmetic_sequence_condition_l401_401379


namespace perp_lines_iff_m_values_l401_401370

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values_l401_401370


namespace arithmetic_sequence_difference_l401_401620

theorem arithmetic_sequence_difference :
  let a₁ := -3
  let d := 8
  let a := λ n : ℕ, a₁ + (n - 1) * d
  abs (a 1010 - a 1000) = 80 :=
by
  let a₁ := -3
  let d := 8
  let a := λ n : ℕ, a₁ + (n - 1) * d
  sorry

end arithmetic_sequence_difference_l401_401620


namespace number_of_three_digit_numbers_without_zeroes_and_no_permutations_divisible_by_4_l401_401892

theorem number_of_three_digit_numbers_without_zeroes_and_no_permutations_divisible_by_4 :
  (number_of_valid_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
    (h2 : ∀ k ∈ digits 10 n, k ≠ 0)
    (h3 : ∀ m ∈ permutations n, ¬divisible_by_4 m)) = 594 := 
sorry

end number_of_three_digit_numbers_without_zeroes_and_no_permutations_divisible_by_4_l401_401892


namespace percent_increase_first_quarter_l401_401685

theorem percent_increase_first_quarter (S : ℝ) (P : ℝ) :
  (S * 1.75 = (S + (P / 100) * S) * 1.346153846153846) → P = 30 :=
by
  intro h
  sorry

end percent_increase_first_quarter_l401_401685


namespace farmer_turkeys_l401_401232

variable (n c : ℝ)

theorem farmer_turkeys (h1 : n * c = 60) (h2 : (c + 0.10) * (n - 15) = 54) : n = 75 :=
sorry

end farmer_turkeys_l401_401232


namespace maximize_sector_area_l401_401007

noncomputable def sector_angle_max_area (r l : ℝ) (h1 : 2 * r + l = 40) (h2 : S = (1 / 2) * r * l) : ℝ :=
  let max_area := 20 * r - r ^ 2
  let derived_r := 10
  let derived_l := 40 - 2 * derived_r
  let derived_alpha := derived_l / derived_r
  derived_alpha

theorem maximize_sector_area : ∀ (r l : ℝ), (2 * r + l = 40) → (∃ S : ℝ, S = (1 / 2) * r * l → ∃ (α : ℝ), α = sector_angle_max_area r l) → 
  (sector_angle_max_area r l = 2) :=
by
  intros r l h1 h2
  unfold sector_angle_max_area
  sorry

end maximize_sector_area_l401_401007


namespace highest_percentage_without_car_l401_401507

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l401_401507


namespace dancer_count_l401_401236

theorem dancer_count (n : ℕ) : 
  ((n + 5) % 12 = 0) ∧ ((n + 5) % 10 = 0) ∧ (200 ≤ n) ∧ (n ≤ 300) → (n = 235 ∨ n = 295) := 
by
  sorry

end dancer_count_l401_401236


namespace find_AP_PB_l401_401682

def points_on_line (a b: Real) : Prop := ∃ p, p ∈ set.interval a b

variables {A B P A1 B1: Point} 
variables {AA1 PP1 BB1 A1B1 : ℝ}

axiom angles_equal : ∠A = ∠B
axiom AA1_perp_A1B1 : AA1 ⊥ A1B1
axiom PP1_perp_A1B1 : PP1 ⊥ A1B1
axiom BB1_perp_A1B1 : BB1 ⊥ A1B1
axiom length_AA1 : AA1 = 17
axiom length_PP1 : PP1 = 16
axiom length_BB1 : BB1 = 20
axiom length_A1B1 : A1B1 = 12

theorem find_AP_PB : AP + PB = 13 := sorry

end find_AP_PB_l401_401682


namespace matrix_arithmetic_series_l401_401295

theorem matrix_arithmetic_series :
  ∏ (k : ℕ) in (range 50), (λ n => matrix.of !) (k.succ * 2) 0 0 1) = 
  matrix.of! 1 2550 0 1 :=
by
  sorry

end matrix_arithmetic_series_l401_401295


namespace cosine_periodicity_l401_401338

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401338


namespace greatest_value_of_sum_l401_401048

theorem greatest_value_of_sum (x : ℝ) (h : 13 = x^2 + (1/x)^2) : x + 1/x ≤ Real.sqrt 15 :=
sorry

end greatest_value_of_sum_l401_401048


namespace matrix_product_sequence_l401_401299

open Matrix

def mat (x : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, x], ![0, 1]]

theorem matrix_product_sequence :
  (List.prod (List.map mat (List.range (50+1)))).sum (Fin.mk 0 (by simp)) (Fin.mk 1 (by simp)) = 2550 := by
  sorry

end matrix_product_sequence_l401_401299


namespace hyperbola_asymptotes_l401_401017

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (e : ℝ) (he : e = sqrt 7 / 2) (h3 : e = c / a) :
  (c^2 / a^2 = 7 / 4) →
  (1 + (b / a)^2 = c^2 / a^2) → 
  (b / a = sqrt 3 / 2) → 
  ∀ x : ℝ, (|b / a| * x = |sqrt 3 / 2| * x) :=
by
  sorry  -- Proof is omitted

end hyperbola_asymptotes_l401_401017


namespace one_fifth_of_5_times_7_l401_401310

theorem one_fifth_of_5_times_7 : (1 / 5) * (5 * 7) = 7 := by
  sorry

end one_fifth_of_5_times_7_l401_401310


namespace investment_double_l401_401668

theorem investment_double (A : ℝ) (r t : ℝ) (hA : 0 < A) (hr : 0 < r) :
  2 * A ≤ A * (1 + r)^t ↔ t ≥ (Real.log 2) / (Real.log (1 + r)) := 
by
  sorry

end investment_double_l401_401668


namespace multiples_of_six_count_l401_401348

theorem multiples_of_six_count :
  ∃ n : ℕ, n = 4 ∧
    (∀ x : ℕ, (30 ≤ x ∧ x ≤ 50 ∧ x % 6 = 0) ↔ (x = 30 ∨ x = 36 ∨ x = 42 ∨ x = 48)) :=
begin
  sorry
end

end multiples_of_six_count_l401_401348


namespace chris_and_fiona_weight_l401_401702

theorem chris_and_fiona_weight (c d e f : ℕ) (h1 : c + d = 330) (h2 : d + e = 290) (h3 : e + f = 310) : c + f = 350 :=
by
  sorry

end chris_and_fiona_weight_l401_401702


namespace calculate_tough_week_sales_l401_401499

-- Define the conditions
variables (G T : ℝ)
def condition1 := T = G / 2
def condition2 := 5 * G + 3 * T = 10400

-- By substituting and proving
theorem calculate_tough_week_sales (G T : ℝ) (h1 : condition1 G T) (h2 : condition2 G T) : T = 800 := 
by {
  sorry 
}

end calculate_tough_week_sales_l401_401499


namespace range_of_m_l401_401751

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - sqrt 3 * cos (2 * x)

theorem range_of_m : 
  (∀ x : ℝ, 0 < x ∧ x ≤ π / 4 → |f x| < m) → m ≥ sqrt 3 :=
by
  intro h,
  sorry

end range_of_m_l401_401751


namespace proper_subsets_B_l401_401026

theorem proper_subsets_B (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | x^2 + 2*x + 1 = 0})
  (hA_singleton : A = {a})
  (hB : B = {x | x^2 + a*x = 0}) :
  a = -1 ∧ 
  B = {0, 1} ∧
  (∀ S, S ∈ ({∅, {0}, {1}} : Set (Set ℝ)) ↔ S ⊂ B) :=
by
  -- Proof not provided, only statement required.
  sorry

end proper_subsets_B_l401_401026


namespace distinct_values_count_l401_401744

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def distinct_powers (a b : ℕ) : ℝ := 3^((b : ℝ)/(a : ℝ))

theorem distinct_values_count : 
  (S.product S).filter (λ p, p.1 ≠ p.2).image (λ p, distinct_powers p.1 p.2) = 22 := 
begin
  sorry
end

end distinct_values_count_l401_401744


namespace largest_of_19_consecutive_integers_l401_401164

theorem largest_of_19_consecutive_integers (avg : ℤ) (h_avg : avg = 99) :
  let seq := list.range' (avg - 9) 19 in
  list.maximum seq = some 108 :=
by
  have h_seq : seq = list.range' 90 19 := by rw [h_avg]; refl
  have h_max : list.maximum seq = list.maximum (list.range' 90 19) := by rw h_seq
  -- simpler conversion may be necessary for list.maximum
  sorry

end largest_of_19_consecutive_integers_l401_401164


namespace Q_solution_l401_401467

noncomputable def Q (x : ℝ) : ℝ :=
  2 * x^3 - (52 / 3) * x - 2

theorem Q_solution :
  (∀ x : ℝ, Q x = Q 0 + Q 1 * x + Q 3 * x^3) ∧ (Q (-2) = 4) ↔ Q = (λ x, 2 * x^3 - (52 / 3) * x - 2) :=
by
  sorry

end Q_solution_l401_401467


namespace selection_ways_l401_401840

noncomputable def company_reps := list.to_finset ["A1", "A2", "B", "C", "D", "E"]
def company (rep : string) : string :=
  if rep = "A1" ∨ rep = "A2" then "A"
  else if rep = "B" then "B"
  else if rep = "C" then "C"
  else if rep = "D" then "D"
  else "E"

theorem selection_ways :
  let company_reps_set := (company_reps : finset string) in
  let selections := company_reps_set.powerset.filter (λ s, s.card = 3) in
  let valid_selections := selections.filter (λ s, (s.image company).card = 3) in
  valid_selections.card = 16 :=
by {
  sorry
}

end selection_ways_l401_401840


namespace verify_log_expr_l401_401814

noncomputable def calc_log_expr (a m n : ℝ) (h₁ : log a 3 = m) (h₂ : log a 2 = n) : ℝ :=
  a ^ (m + 2 * n)

theorem verify_log_expr (a m n : ℝ) (h₁ : log a 3 = m) (h₂ : log a 2 = n) : calc_log_expr a m n h₁ h₂ = 12 :=
sorry

end verify_log_expr_l401_401814


namespace squares_vs_rectangles_l401_401588

theorem squares_vs_rectangles (sq rect : Type) 
  (hsq : has_finite_bounded_area sq)
  (hrect : has_finite_bounded_area rect)
  (four_right_angles_sq : has_four_right_angles sq)
  (four_right_angles_rect : has_four_right_angles rect)
  (opposite_sides_parallel_equal_sq : has_opposite_sides_parallel_equal sq)
  (opposite_sides_parallel_equal_rect : has_opposite_sides_parallel_equal rect)
  (diagonals_bisect_each_other_sq : has_diagonals_bisect_each_other sq)
  (diagonals_bisect_each_other_rect : has_diagonals_bisect_each_other rect)
  (diagonals_perpendicular_sq : has_diagonals_perpendicular sq)
  (not_diagonals_perpendicular_rect : ¬ has_diagonals_perpendicular rect) :
  ∃ P : Prop, 
    (P = has_diagonals_perpendicular sq ∧ (¬ has_diagonals_perpendicular rect)) :=
begin
  -- proof to be written
  sorry
end

end squares_vs_rectangles_l401_401588


namespace boys_to_girls_ratio_l401_401067

theorem boys_to_girls_ratio (S G B : ℕ) (h : (1/2 : ℚ) * G = (1/3 : ℚ) * S) :
  B / G = 1 / 2 :=
by sorry

end boys_to_girls_ratio_l401_401067


namespace align_second_small_ruler_l401_401981

variables (B : ℝ) (k : ℕ)

-- Conditions
def large_ruler := ∀ n : ℕ, n ∈ ℕ
def first_small_ruler := 11 / 10
def second_small_ruler := 9 / 10
def point_A := 0
def point_B := B
def unit_position (n : ℕ) := (B + first_small_ruler * 3 = k)

-- Statement to prove
theorem align_second_small_ruler (h : 18 ≤ B ∧ B < 19) (h_unit : B + 3.3 = k) 
  (h_div : ∃ (n : ℕ), point_B + n * second_small_ruler = k) :
  ∃ x, x = 7 ∧ (point_B + x * second_small_ruler) ∈ ℕ :=
sorry

end align_second_small_ruler_l401_401981


namespace sugar_added_l401_401646

theorem sugar_added (x : ℝ) :
  let initial_solution := 340
  let initial_water := 0.75 * initial_solution
  let initial_kola := 0.05 * initial_solution
  let initial_sugar := 0.20 * initial_solution
  let added_water := 12
  let added_kola := 6.8
  let total_solution := initial_solution + added_water + added_kola + x
  let percentage_sugar := (initial_sugar + x) / total_solution
  percentage_sugar = 0.1966850828729282 ↔ x ≈ 3.23 :=
by
  sorry

end sugar_added_l401_401646


namespace conditional_probability_l401_401839

noncomputable def P (e : Prop) : ℝ := sorry

variable (A B : Prop)

variables (h1 : P A = 0.6)
variables (h2 : P B = 0.5)
variables (h3 : P (A ∨ B) = 0.7)

theorem conditional_probability :
  (P A ∧ P B) / P B = 0.8 := by
  sorry

end conditional_probability_l401_401839


namespace grains_on_11th_more_than_1_to_9_l401_401660

theorem grains_on_11th_more_than_1_to_9 : 
  let grains_on_square (k : ℕ) := 3 ^ k
  let sum_first_n_squares (n : ℕ) := (3 * (3 ^ n - 1) / (3 - 1))
  grains_on_square 11 - sum_first_n_squares 9 = 147624 :=
by
  sorry

end grains_on_11th_more_than_1_to_9_l401_401660


namespace find_integer_cosine_l401_401327

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401327


namespace lavender_to_coconut_ratio_l401_401865

variable (L : ℕ) -- L is the number of lavender candles

-- Jill makes candles using lavender, coconut, and almond scents
variables (A C: ℝ) -- A is the amount of almond scent per candle, C is the amount of coconut scent per candle

-- Conditions 
-- 1. Each candle uses the same amount of scent
axiom scent_amount_same : ∀ c1 c2 : ℝ, c1 = c2

-- 2. Jill made 10 almond candles and ran out of almond scent 
axiom almond_candles_made : ∀ aC: ℕ, aC = 10 → A * aC = 10 * A

-- 3. Jill ran out of almond scent
axiom out_of_almond_scent : ∀ (total_almond : ℝ), total_almond = 10 * A

-- 4. Jill had one and a half times as much coconut scent as almond scent
axiom coconut_amount : ∀ C A: ℝ, C = 1.5 * 10 * A

-- The ratio of the number of lavender candles to the number of coconut candles is L:15
theorem lavender_to_coconut_ratio (L:ℕ) : L / 15 = \rat.mk L 15 := 
  begin
    sorry
  end

end lavender_to_coconut_ratio_l401_401865


namespace treaty_signed_on_thursday_l401_401655

def initial_day : ℕ := 0  -- 0 representing Monday, assuming a week cycle from 0 (Monday) to 6 (Sunday)
def days_in_week : ℕ := 7

def treaty_day (n : ℕ) : ℕ :=
(n + initial_day) % days_in_week

theorem treaty_signed_on_thursday :
  treaty_day 1000 = 4 :=  -- 4 representing Thursday
by
  sorry

end treaty_signed_on_thursday_l401_401655


namespace resulting_curve_eq_l401_401776

def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

def transformed_curve (x y: ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), 
    is_on_circle x0 y0 ∧ 
    x = x0 ∧ 
    y = 4 * y0

theorem resulting_curve_eq : ∀ (x y : ℝ), transformed_curve x y → (x^2 / 9 + y^2 / 144 = 1) :=
by
  intros x y h
  sorry

end resulting_curve_eq_l401_401776


namespace not_equal_S_n_S_n_plus_1_l401_401872

-- Definition of the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Definition of S(n) as the sum of the digits of 2^n
def S (n : ℕ) : ℕ := 
  sum_of_digits (2^n)

-- Statement to be proved: S(n+1) ≠ S(n) for all n
theorem not_equal_S_n_S_n_plus_1 : ∀ n : ℕ, S(n + 1) ≠ S(n) :=
by
  sorry

end not_equal_S_n_S_n_plus_1_l401_401872


namespace find_n_l401_401329

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401329


namespace sum_odd_minus_even_seq_l401_401203

theorem sum_odd_minus_even_seq :
  (∑ k in Finset.range 1012, (2 * k + 1)) - (∑ k in Finset.range 1011, (2 * (k + 1))) = 22 := by
  sorry

end sum_odd_minus_even_seq_l401_401203


namespace max_weight_of_chocolates_l401_401149

def max_total_weight (chocolates : List ℕ) (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) : ℕ :=
300

theorem max_weight_of_chocolates (chocolates : List ℕ)
  (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) :
  max_total_weight chocolates H_wt H_div = 300 :=
sorry

end max_weight_of_chocolates_l401_401149


namespace OG_formula_OH_formula_GH_formula_GI_formula_HI_formula_angle_AGH_eq_angle_DGO_PAO_formula_PAI_formula_PBC_formula_ratio_areas_formula_IA_IB_IC_product_sum_HA_HB_HC_HA_squared_sum_HA_HB_HC_squared_sum_PA_PB_PC_squared_l401_401436

noncomputable def OG_eq : Real := sorry

theorem OG_formula (R a b c : Real) : 
  OG_eq = (1 / 3) * sqrt(9 * R^2 - (a^2 + b^2 + c^2)) := sorry

noncomputable def OH_eq : Real := sorry

theorem OH_formula (R a b c : Real) : 
  OH_eq = sqrt(9 * R^2 - (a^2 + b^2 + c^2)) := sorry

noncomputable def GH_eq : Real := sorry

theorem GH_formula (R a b c : Real) : 
  GH_eq = (2 / 3) * sqrt(9 * R^2 - (a^2 + b^2 + c^2)) := sorry

noncomputable def GI_eq : Real := sorry

theorem GI_formula (r a b c s : Real) : 
  GI_eq = sqrt(9 * r^2 + 2 * (a^2 + b^2 + c^2) - 3 * s^2) := sorry

noncomputable def HI_eq : Real := sorry

theorem HI_formula (r R a b c : Real) : 
  HI_eq = sqrt(2 * r^2 + 4 * R^2 - (a^2 + b^2 + c^2) / 2) := sorry

noncomputable def angle_AGH : Real := sorry

noncomputable def angle_DGO : Real := sorry

theorem angle_AGH_eq_angle_DGO : 
  angle_AGH = angle_DGO := sorry

noncomputable def PAO_squared : Real := sorry

theorem PAO_formula (R r_a : Real) : 
  PAO_squared = R^2 + 2 * R * r_a := sorry

noncomputable def PAI_eq : Real := sorry

theorem PAI_formula (a A : Real) : 
  PAI_eq = a / cos (A / 2) := sorry

noncomputable def PBC_eq : Real := sorry

theorem PBC_formula (a A : Real) : 
  PBC_eq = a / sin (A / 2) := sorry

noncomputable def ratio_areas : Real := sorry

theorem ratio_areas_formula (a b c : Real) :
  ratio_areas = 1 + (a / (-a + b + c)) + (b / (a - b + c)) + (c / (a + b - c)) := sorry

noncomputable def IA_eq : Real := sorry
noncomputable def IB_eq : Real := sorry
noncomputable def IC_eq : Real := sorry

theorem IA_IB_IC_product (r R : Real) : 
  IA_eq * IB_eq * IC_eq = 4 * r^2 * R := sorry

noncomputable def HA_eq : Real := sorry
noncomputable def HB_eq : Real := sorry
noncomputable def HC_eq : Real := sorry

theorem sum_HA_HB_HC (R r : Real) : 
  HA_eq + HB_eq + HC_eq = 2 * (R + r) := sorry

theorem HA_squared (R a : Real) : 
  HA_eq^2 = 4 * R^2 - a^2 := sorry

theorem sum_HA_HB_HC_squared (a b c : Real) : 
  HA_eq^2 + a^2 = HB_eq^2 + b^2 := HB_eq^2 + b^2 = HC_eq^2 + c^2 := sorry

noncomputable def PA_squared : Real := sorry
noncomputable def PB_squared : Real := sorry
noncomputable def PC_squared : Real := sorry
noncomputable def GA_squared : Real := sorry
noncomputable def GB_squared : Real := sorry
noncomputable def GC_squared : Real := sorry
noncomputable def PG_squared : Real := sorry

theorem sum_PA_PB_PC_squared (PA PB PC GA GB GC PG : Real) : 
  PA^2 + PB^2 + PC^2 = GA^2 + GB^2 + GC^2 + 3 * PG^2 := sorry

end OG_formula_OH_formula_GH_formula_GI_formula_HI_formula_angle_AGH_eq_angle_DGO_PAO_formula_PAI_formula_PBC_formula_ratio_areas_formula_IA_IB_IC_product_sum_HA_HB_HC_HA_squared_sum_HA_HB_HC_squared_sum_PA_PB_PC_squared_l401_401436


namespace problem_statement_l401_401229

noncomputable def given_conditions (M Q E B D : Point)
  (h_circle : circle_through Q E)
  (h_intersects_MQ : circle_intersects B (Line.mk M Q))
  (h_intersects_ME : circle_intersects D (Line.mk M E))
  (h_area_ratio_triangle_BDM_MQE : area_ratio (Triangle.mk B D M) (Triangle.mk M Q E) = 9 / 121)
  (h_area_ratio_triangle_BME_DQM : area_ratio (Triangle.mk B M E) (Triangle.mk D Q M) = 4) : Prop := sorry

theorem problem_statement (M Q E B D : Point)
  (h_circle : circle_through Q E)
  (h_intersects_MQ : circle_intersects B (Line.mk M Q))
  (h_intersects_ME : circle_intersects D (Line.mk M E))
  (h_area_ratio_triangle_BDM_MQE : area_ratio (Triangle.mk B D M) (Triangle.mk M Q E) = 9 / 121)
  (h_area_ratio_triangle_BME_DQM : area_ratio (Triangle.mk B M E) (Triangle.mk D Q M) = 4) :
  (segment_length Q E / segment_length B D = 11 / 3) ∧
  (segment_length B Q / segment_length D E = 5 / 19) := sorry

end problem_statement_l401_401229


namespace find_m_plus_n_l401_401234

def num_fir_trees : ℕ := 4
def num_pine_trees : ℕ := 5
def num_acacia_trees : ℕ := 6

def num_non_acacia_trees : ℕ := num_fir_trees + num_pine_trees
def total_trees : ℕ := num_fir_trees + num_pine_trees + num_acacia_trees

def prob_no_two_acacia_adj : ℚ :=
  (Nat.choose (num_non_acacia_trees + 1) num_acacia_trees * Nat.choose num_non_acacia_trees num_fir_trees : ℚ) /
  Nat.choose total_trees num_acacia_trees

theorem find_m_plus_n : (prob_no_two_acacia_adj = 84/159) -> (84 + 159 = 243) :=
by {
  admit
}

end find_m_plus_n_l401_401234


namespace logans_tower_height_l401_401493

theorem logans_tower_height 
  (h_actual : ℝ)
  (v_actual_liters : ℝ)
  (v_miniature_liters : ℝ)
  (ratio_volumes : v_actual_liters / v_miniature_liters = 1000000)
  (ratio_radii : real.cbrt (v_actual_liters / v_miniature_liters) = 100)
  (h_scaled : h_actual / real.cbrt (v_actual_liters / v_miniature_liters) = 0.6)
  : h_actual = 60 →
    v_actual_liters = 150000 →
    v_miniature_liters = 0.15 →
    h_scaled = 0.6 :=
begin
  sorry
end

end logans_tower_height_l401_401493


namespace grades_1_and_2_percentage_is_32_l401_401179

-- Definitions of the percentages of each grade in Baxter and Dexter
def baxter_percentages : List ℕ := [10, 20, 15, 18, 12, 10, 15]
def dexter_percentages : List ℕ := [14, 12, 18, 11, 10, 20, 15]

-- Number of students in Baxter and Dexter
def baxter_students : ℕ := 150
def dexter_students : ℕ := 180

-- Indices corresponding to grades 1 and 2
def grades_1_and_2 : List ℕ := [1, 2]

-- Calculations for percentages of students in grades 1 and 2
def baxter_grade_1_and_2_percentage : ℕ :=
  List.sum (grades_1_and_2.map (λ idx => List.nthLe baxter_percentages idx (by linarith)))

def dexter_grade_1_and_2_percentage : ℕ :=
  List.sum (grades_1_and_2.map (λ idx => List.nthLe dexter_percentages idx (by linarith)))

def total_grade_1_and_2_percentage : ℕ :=
  (baxter_grade_1_and_2_percentage * baxter_students + dexter_grade_1_and_2_percentage * dexter_students) * 100
    / (baxter_students + dexter_students)

-- Statement to prove that the percentage of students in grades 1 and 2 is 32%
theorem grades_1_and_2_percentage_is_32 :
  total_grade_1_and_2_percentage = 32 := 
sorry

end grades_1_and_2_percentage_is_32_l401_401179


namespace solve_m_l401_401384

noncomputable def findM (m : ℝ) (ellipse : ℝ × ℝ → Prop) (A : ℝ × ℝ) (P : ℝ × ℝ → Prop) 
                        (minDistDiff : ℝ) : Prop :=
  m ∈ (0, 1) ∧ ellipse (fun (x, y) => m * x^2 + y^2 = 4 * m) ∧
  A = (0, 2) ∧ (∀ P, ellipse P) ∧ 
  minDistDiff = -4 / 3 ∧ m = 2 / 9

theorem solve_m :
  findM (2/9) (fun (x, y) => m * x^2 + y^2 = 4 * m) 
        (0, 2) (fun P => P ∈ (fun (x, y) => m*x^2 + y^2 = 4*m))
        (- 4 / 3) :=
  sorry

end solve_m_l401_401384


namespace cross_product_inequality_l401_401534

noncomputable def vector_magnitude (v: ℝ × ℝ × ℝ): ℝ := (v.1^2 + v.2^2 + v.3^2) ^ (1 / 2)

noncomputable def vector_cross_product (a b: ℝ × ℝ × ℝ): ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

noncomputable def vector_magnitude_cross_product (a b: ℝ × ℝ × ℝ): ℝ :=
  vector_magnitude (vector_cross_product a b)

theorem cross_product_inequality
  (a b: ℝ × ℝ × ℝ):
  vector_magnitude_cross_product a b ^ 3 ≤ (3 * real.sqrt 3 / 8) * vector_magnitude a ^ 2 * vector_magnitude b ^ 2 * vector_magnitude (a.1 - b.1, a.2 - b.2, a.3 - b.3) ^ 2 :=
  sorry

end cross_product_inequality_l401_401534


namespace total_arrangements_l401_401154

-- Define the schools
inductive School
| GuangzhouZhixin
| ShenzhenForeignLanguages
| SunYatSenMemorial

open School

-- Prove the total number of different arrangements
theorem total_arrangements : 
  let days := { Monday, Tuesday, Wednesday, Thursday, Friday }
  let schools := { GuangzhouZhixin, ShenzhenForeignLanguages, SunYatSenMemorial }
  let arrange (s1 s2 s3: School) (d1 d2 d3: Prop) := 
    (s1 ∈ schools) ∧ (s2 ∈ schools) ∧ (s3 ∈ schools) ∧
    (d1 ∧ d2 ∧ d3) ∧ 
    (s1 ≠ s2) ∧ (s1 ≠ s3) ∧ (s2 ≠ s3) ∧
    (GuangzhouZhixin < ShenzhenForeignLanguages) ∧ (GuangzhouZhixin < SunYatSenMemorial) in
  (∃ d1 d2 d3 : Prop, arrange GuangzhouZhixin ShenzhenForeignLanguages SunYatSenMemorial d1 d2 d3) →
  (∃ count : ℕ, count = 20) :=
by
  sorry

end total_arrangements_l401_401154


namespace surface_area_of_circumscribed_sphere_l401_401447

-- Definitions and conditions
def V : Point := sorry
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

axiom VA_eq_VB_eq_VC : dist V A = dist V B ∧ dist V B = dist V C ∧ dist V A = sqrt 3
axiom AB_eq_sqrt2 : dist A B = sqrt 2
axiom angle_ACB_pi_div_4 : angle A C B = pi / 4

-- The theorem to be proved
theorem surface_area_of_circumscribed_sphere :
  let r := 3 * sqrt 2 / 4 in
  surface_area (Sphere.mk V r) = (9 * pi) / 2 :=
by 
  sorry

end surface_area_of_circumscribed_sphere_l401_401447


namespace difference_between_numbers_l401_401603

theorem difference_between_numbers (x y : ℕ) 
  (h1 : x + y = 20000) 
  (h2 : y = 7 * x) : y - x = 15000 :=
by
  sorry

end difference_between_numbers_l401_401603


namespace line_plane_relationship_l401_401755

variables (Line Plane : Type) 
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

theorem line_plane_relationship 
  {m : Line} 
  {alpha beta : Plane}
  (halpha_beta : perpendicular alpha beta)
  (hm_alpha : perpendicular m alpha) 
: 
  subset m beta ∨ parallel m beta :=
sorry

end line_plane_relationship_l401_401755


namespace find_ellipse_equation_l401_401010

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
∃ F1 F2 : ℝ × ℝ,
  F1.1 < 0 ∧ F2.1 > 0 ∧
  (1 / a^2) * F1.1^2 + (1 / b^2) * F1.2^2 = 1 ∧ 
  (1 / a^2) * F2.1^2 + (1 / b^2) * F2.2^2 = 1 ∧ 
  ∃ A B : ℝ × ℝ,
    (A.1, A.2) ≠ F1 ∧ (B.1, B.2) ≠ F1 ∧
    (1 / a^2) * A.1^2 + (1 / b^2) * A.2^2 = 1 ∧ 
    (1 / a^2) * B.1^2 + (1 / b^2) * B.2^2 = 1 ∧ 
    let side_len := dist F2 A in
    let h := side_len * sqrt 3 / 2 in
    (1 / 2) * side_len * h = 4 * sqrt 3

theorem find_ellipse_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ ellipse_equation a b ∧ a^2 = 9 ∧ b^2 = 6 :=
sorry

end find_ellipse_equation_l401_401010


namespace road_construction_equation_l401_401979

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l401_401979


namespace mobileChargerProblem_l401_401969

/--
Given the market shares of manufacturers A, B, and C as 0.25, 0.35, and 0.4 respectively,
and given that chargers follow binomial distribution when randomly selecting 3 chargers with 
qualification rates for A, B, and C as 0.7, 0.75, and 0.8 respectively,
prove the following:
1. The probability distribution of the number of chargers produced by Manufacturer A (denoted as X) is:
   P(X=0) = 27/64, P(X=1) = 27/64, P(X=2) = 9/64, P(X=3) = 1/64.
2. The expectation of X, E(X), is 3/4.
3. The variance of X, D(X), is 9/16.
4. The probability that a randomly selected defective charger was produced by Manufacturer A is 30/97.
-/
theorem mobileChargerProblem (pA pB pC pDA pDB pDC : ℝ) :
  pA = 0.25 → pB = 0.35 → pC = 0.4 →
  pDA = 0.70 → pDB = 0.75 → pDC = 0.80 →
  let pX0 := (3/4)^3,
      pX1 := 3 * (1/4) * (3/4)^2,
      pX2 := 3 * (1/4)^2 * (3/4),
      pX3 := (1/4)^3,
      E_X := 3 * (1/4),
      Var_X := 3 * (1/4) * (3/4),
      pE := pA * (1 - pDA) + pB * (1 - pDB) + pC * (1 - pDC),
      pAE := pA * (1 - pDA) / pE
  in
  pX0 = 27/64 ∧ pX1 = 27/64 ∧ pX2 = 9/64 ∧ pX3 = 1/64 ∧
  E_X = 3/4 ∧ Var_X = 9/16 ∧
  pAE = 30/97 :=
by {
  intros,
  -- The proof details would go here.
  sorry
}

end mobileChargerProblem_l401_401969


namespace phase_shift_of_cosine_function_l401_401349

theorem phase_shift_of_cosine_function :
  ∃ C : ℝ, is_phase_shift (λ x : ℝ, 2 * cos (2 * x + π / 3) + 1) C ∧ C = -π / 6 :=
begin
  sorry
end

end phase_shift_of_cosine_function_l401_401349


namespace find_angle_AOB_l401_401947

-- Definitions of the conditions mentioned in the problem
variable {Point : Type} -- Define a type for points
variable {A B C D E O : Point} -- Define the points A, B, C, D, E, and O

-- Assume the existence of rectangles and their properties as per the conditions
variable (rectangles : List (Point × Point × Point × Point × Point))
variable (equal_rectangles : ∀ r ∈ rectangles, (r.fst.dist r.snd = 2 * (r.snd.dist r.thrd))) -- Side ratios are 2:1

-- Assume point D bisects the segment CE and OD || BF
variable (D_bisects_CE : D = (C + E) / 2)
variable (OD_parallel_BF : is_parallel (Line_through O D) (Line_through B F))

-- Define the main theorem based on the question and answer
theorem find_angle_AOB (h : ∀ r ∈ rectangles, (r.fst.dist r.snd = 2 * (r.snd.dist r.thrd))) :
  angle O A B = 135 := by
  sorry

end find_angle_AOB_l401_401947


namespace alternating_sum_1_to_101_l401_401706

open Nat

def alternating_sum_101 : ℤ :=
  (List.range 102).map (λ n, if n % 2 = 0 then -n else n).sum

theorem alternating_sum_1_to_101 : alternating_sum_101 = 51 := by
  sorry

end alternating_sum_1_to_101_l401_401706


namespace factorial_binomial_mod_l401_401150

theorem factorial_binomial_mod (p : ℕ) (hp : Nat.Prime p) : 
  ((Nat.factorial (2 * p)) / (Nat.factorial p * Nat.factorial p)) - 2 ≡ 0 [MOD p] :=
by
  sorry

end factorial_binomial_mod_l401_401150


namespace inclination_angles_determined_l401_401245

noncomputable def right_triangle_inclination (A B C : Point) 
  (h : ℝ) (P1 P2 : Plane) : Prop :=
hypotenuse_in_plane A B P1 ∧ 
right_triangle A B C ∧ 
height_projection C AB P1 h ∧ 
inclination_angle_projection A B C P1 P2

theorem inclination_angles_determined (A B C : Point) (h : ℝ) (P1 P2 : Plane) 
  (hab : hypotenuse_in_plane A B P1) 
  (rtc : right_triangle A B C) 
  (hcp : height_projection C AB P1 h)
  (iap : inclination_angle_projection A B C P1 P2) :
  right_triangle_inclination A B C h P1 P2 :=
by {
  apply and.intro,
  { exact hab, },
  apply and.intro,
  { exact rtc, },
  apply and.intro,
  { exact hcp, },
  { exact iap, }
}

end inclination_angles_determined_l401_401245


namespace set_intersection_complement_l401_401123

variable (U : Set ℝ := Set.univ)
variable (M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)})
variable (N : Set ℝ := {x | 0 < x ∧ x < 2})

theorem set_intersection_complement :
  N ∩ (U \ M) = {x | 0 < x ∧ x ≤ 1} :=
  sorry

end set_intersection_complement_l401_401123


namespace definite_integral_evaluated_l401_401637

theorem definite_integral_evaluated :
  ∫ x in 0..2 * Real.sqrt 2, (x^4) / ((16 - x^2) * Real.sqrt (16 - x^2)) = 20 - 6 * Real.pi := by
  sorry

end definite_integral_evaluated_l401_401637


namespace Juanita_weekday_spending_l401_401035

/- Defining the variables and conditions in the problem -/

def Grant_spending : ℝ := 200
def Sunday_spending : ℝ := 2
def extra_spending : ℝ := 60

-- We need to prove that Juanita spends $0.50 per day from Monday through Saturday on newspapers.

theorem Juanita_weekday_spending :
  (∃ x : ℝ, 6 * 52 * x + 52 * 2 = Grant_spending + extra_spending) -> (∃ x : ℝ, x = 0.5) := by {
  sorry
}

end Juanita_weekday_spending_l401_401035


namespace red_blue_dice_probability_l401_401192

-- Define the possible outcomes for each die
def red_die_outcomes := {1, 2, 3, 4, 5, 6}
def blue_die_outcomes := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the predicate for odd numbers
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the predicate for prime numbers
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Define the successful outcomes
def successful_red_outcomes := {1, 3, 5}
def successful_blue_outcomes := {2, 3, 5, 7}

-- Total possible outcomes
def total_outcomes : ℕ := 6 * 8

-- Successful outcomes for the problem
def successful_outcomes : ℕ := 3 * 4

-- The probability we are interested in
def desired_probability : ℚ := successful_outcomes / total_outcomes

theorem red_blue_dice_probability :
  desired_probability = 1 / 4 :=
by 
  -- Add the proof steps here
  sorry

end red_blue_dice_probability_l401_401192


namespace binomial_coefficient_x3_l401_401453

noncomputable def coefficient_x3_term : ℤ := -1174898049840

theorem binomial_coefficient_x3 (a b : ℤ) (n : ℕ) (h1 : a = 2) (h2 : b = -3) (h3 : n = 20) :
  ( ∑ k in Finset.range (n + 1), Nat.choose n k * a ^ k * b ^ (n - k) ) =  -1174898049840 := 
sorry

end binomial_coefficient_x3_l401_401453


namespace fixed_point_exists_for_any_m_l401_401737

theorem fixed_point_exists_for_any_m (m : ℝ) (c d : ℝ) : 
  (∀ m : ℝ, ∃ (c d : ℝ), d = 9 * 5^2 + m * 5 - 5 * m) → 
  (c, d) = (5, 225) :=
by {
  intros h,
  have h_fixed_point := h m,
  cases h_fixed_point with c_val d_val,
  rw [d_val, mul_assoc] at *,
  exact ⟨rfl, rfl⟩
}

end fixed_point_exists_for_any_m_l401_401737


namespace john_school_year_hours_l401_401094

theorem john_school_year_hours (summer_earnings : ℝ) (summer_hours_per_week : ℝ) (summer_weeks : ℝ) (target_school_earnings : ℝ) (school_weeks : ℝ) :
  summer_earnings = 4000 → summer_hours_per_week = 40 → summer_weeks = 8 → target_school_earnings = 5000 → school_weeks = 25 →
  (target_school_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_weeks) = 16 :=
by
  sorry

end john_school_year_hours_l401_401094


namespace min_distance_l401_401822

theorem min_distance (z : ℂ) (h : abs (z - (2 - I)) = 1) : 
  ∃ w : ℂ, abs (w + 1 - 2 * I) = 3 * real.sqrt 2 - 1 :=
begin
  sorry,
end

end min_distance_l401_401822


namespace roy_cat_finishes_food_on_wednesday_l401_401259

-- Define the conditions
def morning_consumption := (1 : ℚ) / 5
def evening_consumption := (1 : ℚ) / 6
def total_cans := 10

-- Define the daily consumption calculation
def daily_consumption := morning_consumption + evening_consumption

-- Define the day calculation function
def day_cat_finishes_food : String :=
  let total_days := total_cans / daily_consumption
  if total_days ≤ 7 then "certain day within a week"
  else if total_days ≤ 14 then "Wednesday next week"
  else "later"

-- The main theorem to prove
theorem roy_cat_finishes_food_on_wednesday : day_cat_finishes_food = "Wednesday next week" := sorry

end roy_cat_finishes_food_on_wednesday_l401_401259


namespace _l401_401213

noncomputable def locus_of_tangent_intersections (S : Circle) (P : Point) : Set Point :=
{ Q | ∃ (A B : Point), A ≠ B ∧ tangent_at S A ∧ tangent_at S B ∧
secant_through P (A ∩ S) ∧ secant_through P (B ∩ S) ∧
is_intersection_of_tangents Q A B }

noncomputable theorem part_a_locus (S : Circle) (P : Point) :
  locus_of_tangent_intersections S P = (line_through_tangency_points S \ set_of_diameter S) :=
sorry

end _l401_401213


namespace product_neg_int_add_five_l401_401422

theorem product_neg_int_add_five:
  let x := -11 
  let y := -8 
  x * y + 5 = 93 :=
by
  -- Proof omitted
  sorry

end product_neg_int_add_five_l401_401422


namespace sin_alpha_l401_401398

theorem sin_alpha {α : ℝ} 
  {x y : ℝ} 
  (hx : x = 1 / 2) 
  (hy : y = sqrt 3 / 2) 
  (r : ℝ) 
  (hr : r = sqrt (x^2 + y^2)) :
  sin α = y / r := 
sorry

end sin_alpha_l401_401398


namespace polynomial_root_l401_401763

noncomputable def f (x : ℝ) : ℝ := x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5

theorem polynomial_root (a1 a2 a3 a4 a5 : ℤ) :
  (f (ℚ.sqrt 3 + ℚ.sqrt 2) = 0) ∧
  (f 1 + f 3 = 0) →
  f (-1) = 24 :=
by
  sorry

end polynomial_root_l401_401763


namespace tan_sum_identity_l401_401770

variable (α β : Real)
noncomputable def tg (x : Real) := Real.tan x

theorem tan_sum_identity (h : tg α + tg β + tg (- (tg α + tg β) / (1 - tg α * tg β)) = tg α * tg β * (- (tg α + tg β) / (1 - tg α * tg β))) : 
  tg α + tg β + tg (- (tg α + tg β) / (1 - tg α * tg β)) = tg α * tg β * tg (- (tg α + tg β) / (1 - tg α * tg β)) :=
begin
  assumption
end

end tan_sum_identity_l401_401770


namespace case_angle_C_90_case_angle_C_acute_case_angle_C_obtuse_conclusion_conditions_l401_401102

-- Define the problem context with circumcircle and orthocenter
variables {A B C O H: Type}
variables {R : ℝ}

-- Define the conditions: circumcenter, orthocenter, and angle C
variable (circumcenter : A → B → C → O)
variable (orthocenter : A → B → C → H)
variable (circumradius : O → ℝ )
variable (angle_AOB, angle_C, angle_HAB, angle_OAB : ℝ)

-- Case: ∠C = 90°
theorem case_angle_C_90 (h1 : angle_C = 90) :
  AH + BH > 2 * R := sorry

-- Case: ∠C < 90°
theorem case_angle_C_acute (h2 : angle_C < 90) :
  (∀ (O_inside: O ∈ triangle ABC) (H_inside: H ∈ triangle ABC), AH + BH ≥ 2 * R) := sorry

-- Case: ∠C > 90°
theorem case_angle_C_obtuse (h3 : angle_C > 90) :
  (∀ (HAB_gt_OAB : ∠HAB > ∠OAB), AH + BH > AO + BO) := sorry

-- Conclusion: AH + BH > 2R under all conditions
theorem conclusion_conditions (angle_C : ℝ):
  (∀ (θ : ℝ), (θ = 90 ∨ θ < 90 ∨ θ > 90) →
  AH + BH > 2 * R) := sorry

end case_angle_C_90_case_angle_C_acute_case_angle_C_obtuse_conclusion_conditions_l401_401102


namespace find_unbounded_function_l401_401308

theorem find_unbounded_function (f : ℤ → ℤ) (h_unbounded : ∃ x, ∀ y > x, ∃ z, z > y ∧ f(z) > y) 
  (h_condition : ∀ x y : ℤ, f (f x - y) ∣ x - f y) :
  f = λ x, x :=
sorry

end find_unbounded_function_l401_401308


namespace find_a_for_odd_function_l401_401014

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f (x a : ℝ) : ℝ := (2^x - 1) / (2^x + a)

theorem find_a_for_odd_function {a : ℝ} (h : is_odd_function (λ x, f x a)) : a = 1 := 
sorry

end find_a_for_odd_function_l401_401014


namespace roots_complex_nonzero_real_imag_l401_401712

noncomputable def discriminant (a b c : ℂ) : ℂ := b^2 - 4 * a * c

theorem roots_complex_nonzero_real_imag {k : ℝ} (h : 0 < k) :
  let a : ℂ := 10
  let b : ℂ := 5 * complex.I
  let c : ℂ := -k
  in (discriminant a b c).re > 0 → ∀ (z : ℂ), (10 * z^2 + 5 * complex.I * z - k = 0 → 
  (z ≠ 0 ∨ 5 * complex.I ≠ 0) ∧ z.re ≠ 0 ∧ z.im ≠ 0) := by
    intros
    sorry

end roots_complex_nonzero_real_imag_l401_401712


namespace negation_equivalence_l401_401205

theorem negation_equivalence :
  (∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) ∧
  (∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) ∧
  (∃ θ : ℝ, sin θ + cos θ < 1) ↔ (∀ θ : ℝ, sin θ + cos θ ≥ 1) ∧
  (∀ θ : ℝ, sin θ ≤ 1) ↔ (∃ θ : ℝ, sin θ > 1) →
  ¬ ((∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 < 0)) :=
by sorry

end negation_equivalence_l401_401205


namespace problem_statement_l401_401810

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l401_401810


namespace train_length_proof_l401_401250

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 5 / 18
  speed_ms * time_s

theorem train_length_proof : train_length 144 16 = 640 := by
  sorry

end train_length_proof_l401_401250


namespace car_distance_ratio_l401_401634

theorem car_distance_ratio (speed_A time_A speed_B time_B : ℕ) 
  (hA : speed_A = 70) (hTA : time_A = 10) 
  (hB : speed_B = 35) (hTB : time_B = 10) : 
  (speed_A * time_A) / gcd (speed_A * time_A) (speed_B * time_B) = 2 :=
by
  sorry

end car_distance_ratio_l401_401634


namespace six_applications_return_initial_l401_401817

def g (x : ℝ) : ℝ := -2 / x

theorem six_applications_return_initial (x : ℝ) : g (g (g (g (g (g x))))) = x := 
  sorry

example : g (g (g (g (g (g 4))))) = 4 := 
  six_applications_return_initial 4

end six_applications_return_initial_l401_401817


namespace determine_g_l401_401541

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l401_401541


namespace find_k_l401_401433

-- Definitions for the given conditions
def slope_of_first_line : ℝ := 2
def alpha : ℝ := slope_of_first_line
def slope_of_second_line : ℝ := 2 * alpha

-- The proof goal
theorem find_k (k : ℝ) : slope_of_second_line = k ↔ k = 4 := by
  sorry

end find_k_l401_401433


namespace number_of_bottles_of_water_sold_l401_401833

-- Definitions for conditions
def cost_of_cola := 3
def cost_of_juice := 1.5
def cost_of_water := 1
def bottles_of_cola := 15
def bottles_of_juice := 12
def total_earnings := 88

-- Assumption for the earnings equation
def earnings_equation (W : ℝ) : Prop :=
  cost_of_cola * bottles_of_cola + cost_of_juice * bottles_of_juice + cost_of_water * W = total_earnings

-- Statement to prove that \( W = 25 \)
theorem number_of_bottles_of_water_sold (W : ℝ) (h : earnings_equation W) : W = 25 :=
by
  sorry

end number_of_bottles_of_water_sold_l401_401833


namespace seq_a_sum_correct_l401_401378

noncomputable def seq_a : ℕ → ℚ
| 0     := 0  -- traditionally Lean indexes from 0
| 1     := 2
| (n+1) := (2 * n + 1) / (2 ^ (n + 1))

def seq_a_sum (n : ℕ) : ℚ :=
∑ i in Finset.range (n + 1), seq_a i

theorem seq_a_sum_correct (n : ℕ) : seq_a_sum n = 11 / 2 - (2 * n + 5) / (2 ^ n) :=
by
  sorry  -- Proof can be filled in later

end seq_a_sum_correct_l401_401378


namespace complement_of_A_in_U_l401_401791

open Set Nat

noncomputable def U : Set ℕ := {x | x ≥ 3}
noncomputable def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_of_A_in_U : ∁_U A = {3} :=
by
  sorry

end complement_of_A_in_U_l401_401791


namespace area_A1B1C1_l401_401502

variables {Point : Type}
variables [EuclideanSpace ℝ Point]
variables (A B C A1 B1 C1 : Point)
variables (S : ℝ)

-- Conditions
axiom H1 : vectorBetween A B1 = 2 • vectorBetween A B
axiom H2 : vectorBetween B C1 = 2 • vectorBetween B C
axiom H3 : vectorBetween C A1 = 2 • vectorBetween C A
axiom areaABC : ∀ (a b c : Point), triangleArea a b c = S

-- Goal
theorem area_A1B1C1 : triangleArea A1 B1 C1 = 7 * S :=
by
  sorry

end area_A1B1C1_l401_401502


namespace units_digit_of_expression_l401_401995

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def expression := (20 * 21 * 22 * 23 * 24 * 25) / 1000

theorem units_digit_of_expression : units_digit (expression) = 2 :=
by
  sorry

end units_digit_of_expression_l401_401995


namespace sum_of_fractional_parts_l401_401099

def fractionalPart (x : ℚ) : ℚ := x - x.toRat.floor

theorem sum_of_fractional_parts (p : ℕ) [hp_prime : Fact (Nat.Prime p)] (hp_mod : p ≡ 1 [MOD 4]) :
    ∑ k in Finset.range ((p - 1) / 2 + 1), fractionalPart (k^2 / p) = (p - 1) / 4 := by
  sorry

end sum_of_fractional_parts_l401_401099


namespace min_sum_l401_401159

namespace MinimumSum

theorem min_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hc : 98 * m = n^3) : m + n = 42 :=
sorry

end MinimumSum

end min_sum_l401_401159


namespace sum_a_1_to_1000_l401_401893

noncomputable def a : ℕ → ℕ
| 0       := 1
| 1       := 1
| 2       := 1
| (n + 3) := if -4 * (a (n + 2))^3 - 27 * (a (n + 1) * a n)^2 > 0 then 3 else 1

theorem sum_a_1_to_1000 : (Finset.range 1000).sum a = 1000 := by
  sorry

end sum_a_1_to_1000_l401_401893


namespace initial_payment_mr_dubois_l401_401902

-- Definition of the given conditions
def total_cost_of_car : ℝ := 13380
def monthly_payment : ℝ := 420
def number_of_months : ℝ := 19

-- Calculate the total amount paid in monthly installments
def total_amount_paid_in_installments : ℝ := monthly_payment * number_of_months

-- Statement of the theorem we want to prove
theorem initial_payment_mr_dubois :
  total_cost_of_car - total_amount_paid_in_installments = 5400 :=
by
  sorry

end initial_payment_mr_dubois_l401_401902


namespace exists_triangle_area_one_same_color_l401_401289

theorem exists_triangle_area_one_same_color (color : ℝ × ℝ → ℕ) (hcolor : ∀ p : ℝ × ℝ, color p ∈ {1, 2, 3}) :
  ∃ a b c : ℝ × ℝ, color a = color b ∧ color b = color c ∧ color a = color c ∧ 
  (1/2) * abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) = 1 :=
by 
  sorry

end exists_triangle_area_one_same_color_l401_401289


namespace find_k_l401_401060

theorem find_k (k : ℝ) (h : (3, 1) ∈ {(x, y) | y = k * x - 2} ∧ k ≠ 0) : k = 1 :=
by sorry

end find_k_l401_401060


namespace ones_digit_8_pow_40_l401_401987

theorem ones_digit_8_pow_40 : (8^40) % 10 = 6 :=
by {
  have cycle : ∀ n, (8^n) % 10 = [8, 4, 2, 6].cycle n := 
    by intros; revert n; exact sorry,
  calc (8^40) % 10
    = [8, 4, 2, 6].cycle (40 % 4) : by rw (cycle 40)
    ... = 6 : by norm_num
}

end ones_digit_8_pow_40_l401_401987


namespace distance_travelled_by_gavril_l401_401360

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end distance_travelled_by_gavril_l401_401360


namespace angle_EDP_eq_90_l401_401862

-- a) Definitions of triangle and conditions
variables {A B C D E M N P : Type*}
variables (triangle_ABC : IsTriangle A B C)
variables (angle_ABC : angle B A C = 60)
variables (AB BC : Real)
variables (r : Real)
variables (h1 : 5 * AB = 4 * BC)
variables (D_foot : IsFootOfAltitude B D triangle_ABC)
variables (E_foot : IsFootOfAltitude C E triangle_ABC)
variables (M_midpoint : IsMidpoint M B D)
variables (circumcircle_BMC : IsCircumcircleOf B M C)
variables (N_on_AC : IsOnLine N A C)
variables (N_circumcircle : IsOnCircumcircle N circumcircle_BMC)
variables (BN_intersection : IsIntersectionOf BN N A C)
variables (CM_intersection : IsIntersectionOf CM M B C)
variables (P_intersection : BN = CM ∧ BN ∩ CM = P)

-- d) The theorem to be proven
theorem angle_EDP_eq_90
  (angle_ABC : angle B A C = 60)
  (h1 : 5 * AB = 4 * BC)
  (D_foot : IsFootOfAltitude B D triangle_ABC)
  (E_foot : IsFootOfAltitude C E triangle_ABC)
  (M_midpoint : IsMidpoint M B D)
  (circumcircle_BMC : IsCircumcircleOf B M C)
  (N_on_AC : IsOnLine N A C)
  (N_circumcircle : IsOnCircumcircle N circumcircle_BMC)
  (BN_intersection : IsIntersectionOf BN N A C)
  (CM_intersection : IsIntersectionOf CM M B C)
  (P_intersection : BN = CM ∧ BN ∩ CM = P):
  angle E D P = 90 :=
sorry

end angle_EDP_eq_90_l401_401862


namespace problem_1_exists_a_problem_2_values_of_a_l401_401789

open Set

-- Definitions for sets A, B, C
def A (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + 4 * a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Lean statements for the two problems
theorem problem_1_exists_a : ∃ a : ℝ, A a ∩ B = A a ∪ B ∧ a = 1/2 := by
  sorry

theorem problem_2_values_of_a (a : ℝ) : 
  (A a ∩ B ≠ ∅ ∧ A a ∩ C = ∅) → 
  (A a = {-1} → a = -1) ∧ (∀ x, A a = {-1, x} → x ≠ 2 → False) := 
  by sorry

end problem_1_exists_a_problem_2_values_of_a_l401_401789


namespace boundary_length_of_formed_figure_l401_401248

theorem boundary_length_of_formed_figure :
  let side_length := Real.sqrt 144
  let segment_length := side_length / 5
  let radius := segment_length
  let circumference := 2 * Real.pi * radius
  let total_straight_length := segment_length * 8
  let total_boundary_length := circumference + total_straight_length
  (Float.round (total_boundary_length * 10)) / 10 = 34.3 :=
by
  let side_length := Real.sqrt 144
  let segment_length := side_length / 5
  let radius := segment_length
  let circumference := 2 * Real.pi * radius
  let total_straight_length := segment_length * 8
  let total_boundary_length := circumference + total_straight_length
  have h1 : (Float.round (total_boundary_length * 10)) / 10 = 34.3
  sorry

end boundary_length_of_formed_figure_l401_401248


namespace quadratic_roots_interlace_l401_401916

variable (p1 p2 q1 q2 : ℝ)

theorem quadratic_roots_interlace
(h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + p1 * r1 + q1 = 0 ∧ r2^2 + p1 * r2 + q1 = 0)) ∧
  (∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + p2 * s1 + q2 = 0 ∧ s2^2 + p2 * s2 + q2 = 0)) ∧
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
  (a^2 + p1*a + q1 = 0 ∧ b^2 + p2*b + q2 = 0 ∧ c^2 + p1*c + q1 = 0 ∧ d^2 + p2*d + q2 = 0)) := 
sorry

end quadratic_roots_interlace_l401_401916


namespace six_arts_arrangement_l401_401074

theorem six_arts_arrangement (arts : Fin 6 → Char)
  (h1 : arts 2 = '数')
  (h2 : ∃ i : Fin 5, i ≠ 2 ∧ (arts i = '射' ∧ arts (i + 1) = '御') ∨ (arts i = '御' ∧ arts (i + 1) = '射')) :
  (∃ perm : List (Char × Fin 6), (perm.map Prod.fst).perm Arts.sort ∧ perm.length = 6) →
  (arts.perms.length = 36) :=
by
  sorry

end six_arts_arrangement_l401_401074


namespace ellipse_area_l401_401729

-- Define the given ellipse equation and states the goal to prove
theorem ellipse_area : 
  ∃ a b : ℝ, (∀ x y : ℝ, 3 * x^2 + 18 * x + 9 * y^2 - 27 * y + 27 = 0 ↔ 
    ((x + 3) ^ 2) / a ^ 2 + ((y - 1.5) ^ 2) / b ^ 2 = 1) ∧
  (Real.pi * a * b = 2.598 * Real.pi) :=
begin
  -- To be proved, therefore adding sorry
  sorry
end

end ellipse_area_l401_401729


namespace range_of_a_l401_401061

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
by
  sorry -- The proof is omitted as per the instructions.

end range_of_a_l401_401061


namespace magnitude_AB_l401_401452

variable (OA OB : ℝ × ℝ)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem magnitude_AB (hOA : OA = (1, 1)) (hOB : OB = (-1, 3)) :
  Real.sqrt ((AB OA OB).1^2 + (AB OA OB).2^2) = 2 * Real.sqrt 2 :=
by
  rw [hOA, hOB]
  have hAB : AB OA OB = (-2, 2) := by simp [AB]
  rw [hAB]
  simp
  sorry

end magnitude_AB_l401_401452


namespace ratio_A_to_B_l401_401703

theorem ratio_A_to_B (total_weight_X : ℕ) (weight_B : ℕ) (weight_A : ℕ) (h₁ : total_weight_X = 324) (h₂ : weight_B = 270) (h₃ : weight_A = total_weight_X - weight_B):
  weight_A / gcd weight_A weight_B = 1 ∧ weight_B / gcd weight_A weight_B = 5 :=
by
  sorry

end ratio_A_to_B_l401_401703


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401514

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401514


namespace ship_with_highest_no_car_round_trip_percentage_l401_401518

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l401_401518


namespace trigonometric_equation_solution_l401_401208

noncomputable def z (k : ℤ) : ℝ := (4 * k - 1) * π / 8

theorem trigonometric_equation_solution (k : ℤ) (hk : z k ≠ 0) (htg : tan (z k / 2) ≠ 1 ∧ tan (z k / 2) ≠ -1) : 
  cot (z k / 2) - tan (z k / 2) + 4 * acos (2 * sin (z k / 2)) = (4 * tan (z k / 2)) / (tan (z k / 2)^2 - 1) :=
sorry

end trigonometric_equation_solution_l401_401208


namespace sqrt_polynomial_roots_l401_401281

theorem sqrt_polynomial_roots :
  let a := Real.sin (Real.pi / 7)
  let b := Real.sin (2 * Real.pi / 7)
  let c := Real.sin (3 * Real.pi / 7)
  a ^ 2, b ^ 2, and c ^ 2 are roots of (λ x : ℝ, 64 * x ^ 3 - 112 * x ^ 2 + 56 * x - 7) →
  Real.sqrt ((2 - a ^ 2) * (2 - b ^ 2) * (2 - c ^ 2)) = 13 / 8 :=
sorry

end sqrt_polynomial_roots_l401_401281


namespace correct_answer_B_l401_401675

variables {l m : Type} {α β : Type}
variables [has_subset l β] [has_bot α β] [has_bot l β] [has_parallel α β]

-- Option B: If l ⊥ β, and α ‖ β, then l ⊥ α
def correct_proposition_B (l : Type) (β : Type) (α : Type) [has_bot l β] [has_parallel α β] : Prop :=
  l ⊥ β ∧ α ‖ β → l ⊥ α

theorem correct_answer_B (l : Type) (m : Type) (α : Type) (β : Type)
  [has_subset l β] [has_bot α β] [has_parallel α β] :
  correct_proposition_B l β α :=
by sorry

end correct_answer_B_l401_401675


namespace jordan_made_shots_in_sixth_game_l401_401866

-- Define the initial conditions
def initial_made_shots : ℕ := 18
def initial_taken_shots : ℕ := 45
def initial_average : ℚ := 0.4
def additional_taken_shots : ℕ := 15
def final_average : ℚ := 0.45

-- Calculate the final number of made shots
def final_taken_shots : ℕ := initial_taken_shots + additional_taken_shots
def final_made_shots : ℚ := final_taken_shots * final_average

-- Define the Lean proof problem
theorem jordan_made_shots_in_sixth_game : 
  final_made_shots - initial_made_shots = 9 :=
by
sory

end jordan_made_shots_in_sixth_game_l401_401866


namespace theta_perpendicular_m_range_l401_401031

variables (θ m : ℝ)

def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

theorem theta_perpendicular (h1 : θ ∈ Set.Icc 0 Real.pi) 
  (h2 : (vector_a θ).1 * vector_b.1 + (vector_a θ).2 * vector_b.2 = 0) :
  θ = Real.pi / 3 := 
  sorry

theorem m_range (h1 : θ ∈ Set.Icc 0 Real.pi) 
  (h2 : |((2 * vector_a θ.1 - Real.sqrt 3)^2 + (2 * vector_a θ.2 + 1)^2)^0.5| < m) :
  4 < m := 
  sorry

end theta_perpendicular_m_range_l401_401031


namespace find_b_of_bisector_l401_401581

-- Definitions based on the problem conditions
def is_midpoint (P1 P2 M : (ℝ × ℝ)) : Prop :=
  M = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def is_bisector (L : (ℝ × ℝ) → Prop) (P1 P2 : (ℝ × ℝ)) : Prop :=
  ∃ M : (ℝ × ℝ), is_midpoint P1 P2 M ∧ L M

-- Problem statement
theorem find_b_of_bisector :
  (∀ x y b : ℝ, (x + y = b) → is_bisector (λ P, P.1 + P.2 = b) (2, 4) (6, 8) → b = 10) :=
by
  intros x y b H H_bisector
  sorry

end find_b_of_bisector_l401_401581


namespace infinite_div_pairs_l401_401873

theorem infinite_div_pairs {a : ℕ → ℕ} (h_seq : ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001) :
  ∃ (s : ℕ → (ℕ × ℕ)), (∀ n, (s n).2 < (s n).1) ∧ (a ((s n).2) ∣ a ((s n).1)) :=
sorry

end infinite_div_pairs_l401_401873


namespace sum_of_integers_square_greater_272_l401_401597

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l401_401597


namespace value_of_x_squared_plus_reciprocal_squared_l401_401806

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l401_401806


namespace factorize_problem_1_factorize_problem_2_l401_401293

theorem factorize_problem_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := 
by sorry

theorem factorize_problem_2 (x y : ℝ) : 2 * x^3 - 12 * x^2 * y + 18 * x * y^2 = 2 * x * (x - 3 * y)^2 :=
by sorry

end factorize_problem_1_factorize_problem_2_l401_401293


namespace projection_orthogonal_l401_401881

variables (a b : ℝ × ℝ)
variables (v : ℝ × ℝ)
variables (h1 : dot_product a b = 0) -- a and b are orthogonal
variables (h2 : proj a (4, -2) = (1, 2)) -- projection of (4, -2) onto a

-- Theorem statement
theorem projection_orthogonal {a b : ℝ × ℝ} {v : ℝ × ℝ}
  (h1 : dot_product a b = 0)
  (h2 : proj a v = (1, 2)) :
  proj b v = (3, -4) :=
sorry

end projection_orthogonal_l401_401881


namespace apples_b_lighter_than_a_l401_401936

-- Definitions based on conditions
def total_weight : ℕ := 72
def weight_basket_a : ℕ := 42
def weight_basket_b : ℕ := total_weight - weight_basket_a

-- Theorem to prove the question equals the answer given the conditions
theorem apples_b_lighter_than_a : (weight_basket_a - weight_basket_b) = 12 := by
  -- Placeholder for proof
  sorry

end apples_b_lighter_than_a_l401_401936


namespace triangle_EMN_is_isosceles_l401_401605

theorem triangle_EMN_is_isosceles
  (A B C D E O1 O2 M N: Point)
  (h_cyclic: CyclicQuadrilateral A B C D)
  (h_AC_BE: Line A C ∩ Line B D = {E})
  (h_O1_incenter: Incenter O1 A B C)
  (h_O2_incenter: Incenter O2 A B D)
  (h_O1O2_M: Line O1 O2 ∩ Line E B = {M})
  (h_O1O2_N: Line O1 O2 ∩ Line E A = {N}) :
  IsIsoscelesTriangle E M N :=
begin
  sorry,  -- Proof should be provided here
end

end triangle_EMN_is_isosceles_l401_401605


namespace mod_product_example_l401_401156

theorem mod_product_example :
  ∃ m : ℤ, 256 * 738 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 ∧ m = 53 :=
by
  use 53
  sorry

end mod_product_example_l401_401156


namespace geon_wu_run_distance_l401_401745

theorem geon_wu_run_distance
  (circumference : ℕ)
  (geon_wu_speed : ℕ)
  (jia_speed : ℕ)
  (circumference_eq : circumference = 18)
  (geon_wu_speed_eq : geon_wu_speed = 5)
  (jia_speed_eq : jia_speed = 4) :
  ∃ (distance : ℕ), distance = 10 :=
by {
  have H1 : circumference = 18 := circumference_eq,
  have H2 : geon_wu_speed = 5 := geon_wu_speed_eq,
  have H3 : jia_speed = 4 := jia_speed_eq,
  -- Here is where the proof would go
  existsi 10,
  sorry
}

end geon_wu_run_distance_l401_401745


namespace probability_perfect_square_l401_401719

theorem probability_perfect_square (choose_numbers : Finset (Fin 49)) (ticket : Finset (Fin 49))
  (h_choose_size : choose_numbers.card = 6) 
  (h_ticket_size : ticket.card = 6)
  (h_choose_square : ∃ (n : ℕ), (choose_numbers.prod id = n * n))
  (h_ticket_square : ∃ (m : ℕ), (ticket.prod id = m * m)) :
  ∃ T, (1 / T = 1 / T) :=
by
  sorry

end probability_perfect_square_l401_401719


namespace linear_function_quadrants_l401_401427

theorem linear_function_quadrants (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : ¬ ∃ x : ℝ, ∃ y : ℝ, x > 0 ∧ y < 0 ∧ y = b * x - a :=
sorry

end linear_function_quadrants_l401_401427


namespace conditional_probability_l401_401838

noncomputable def P (e : Prop) : ℝ := sorry

variable (A B : Prop)

variables (h1 : P A = 0.6)
variables (h2 : P B = 0.5)
variables (h3 : P (A ∨ B) = 0.7)

theorem conditional_probability :
  (P A ∧ P B) / P B = 0.8 := by
  sorry

end conditional_probability_l401_401838


namespace part1_part2_l401_401015

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x - 1)

theorem part1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 1/2 :=
begin
  sorry
end

theorem part2 (a : ℝ) :
  (∀ x ∈ set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) →
  0 ≤ a ∧ a ≤ 3 :=
begin
  sorry
end

end part1_part2_l401_401015


namespace pace_ratio_l401_401661

variable (P P' D : ℝ)

-- Usual time to reach the office in minutes
def T_usual := 120

-- Time to reach the office on the late day in minutes
def T_late := 140

-- Distance to the office is the same
def office_distance_usual := P * T_usual
def office_distance_late := P' * T_late

theorem pace_ratio (h : office_distance_usual = office_distance_late) : P' / P = 6 / 7 :=
by
  sorry

end pace_ratio_l401_401661


namespace parking_lot_capacity_l401_401072

-- Definitions based on the conditions
def levels : ℕ := 5
def parkedCars : ℕ := 23
def moreCars : ℕ := 62
def capacityPerLevel : ℕ := parkedCars + moreCars

-- Proof problem statement
theorem parking_lot_capacity : levels * capacityPerLevel = 425 := by
  -- Proof omitted
  sorry

end parking_lot_capacity_l401_401072


namespace find_n_l401_401332

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401332


namespace ellipse_and_triangle_max_area_l401_401761

variables {a b x y m t : ℝ}

-- Question 1: Finding the equation of the ellipse
def ellipse_equation (a b : ℝ) (h₀ : a > b > 0) (e : ℝ) (h₁ : e = 1/2) (c : ℝ) (h₂ : c = 1) : Prop :=
  (a = 2) ∧ (b^2 = (a^2 - c^2)) ∧ (c = a * e) ∧ (frac x^2 4 + frac y^2 3 = 1)

-- Question 2: Finding the maximum area of triangle AF'B
def max_area_triangle (l : ℝ → ℝ) (F F' A B : ℝ × ℝ) (h₃ : l F = 0) (h₄ : F = (1, 0)) : Prop :=
  let y1 := (-6 * m) / (3 * m^2 + 4) in
  let y2 := (-9) / (3 * m^2 + 4) in
  let area := (12 * sqrt(m^2 + 1)) / (3 * m^2 + 4) in
  (area = 3)

-- Main theorem incorporating both questions
theorem ellipse_and_triangle_max_area :
  ∀ (a b : ℝ) (h₀ : a > b > 0) (e : ℝ) (h₁ : e = 1/2) (c : ℝ) (h₂ : c = 1) (l : ℝ → ℝ) (F F' A B : ℝ × ℝ) (h₃ : l F = 0) (h₄ : F = (1, 0)),
  ellipse_equation a b h₀ e h₁ c h₂ ∧ max_area_triangle l F F' A B h₃ h₄ :=
by
  sorry

end ellipse_and_triangle_max_area_l401_401761


namespace find_locus_and_distance_l401_401756

-- Definition of the parabola locus E
def locus_eq (x y : ℝ) := y^2 = 8 * x

-- Definition of the midpoint condition
def midpoint (x1 y1 x2 y2 : ℝ) := (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1

-- Definition of the distance |PQ|
def distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Main statement
theorem find_locus_and_distance (x1 y1 x2 y2 : ℝ) (h1 : locus_eq x1 y1) 
  (h2 : locus_eq x2 y2) (h_midpoint : midpoint x1 y1 x2 y2) :
  distance x1 y1 x2 y2 = Real.sqrt 119 / 2 :=
by sorry

end find_locus_and_distance_l401_401756


namespace probability_of_failing_chinese_given_math_l401_401065

-- Definitions of events as probabilities
def P_failed_math : ℝ := 0.25
def P_failed_chinese : ℝ := 0.10
def P_failed_both : ℝ := 0.05

-- The conditional probability formula
def P_conditional := P_failed_both / P_failed_math

-- The theorem to prove
theorem probability_of_failing_chinese_given_math :
  P_conditional = 1 / 5 :=
sorry

end probability_of_failing_chinese_given_math_l401_401065


namespace find_a_tangent_line_eq_l401_401781

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x - 1) * Real.exp x

theorem find_a (a : ℝ) : f 1 (-3) = 0 → a = 1 := by
  sorry

theorem tangent_line_eq (x : ℝ) (e : ℝ) : x = 1 ∧ f 1 x = Real.exp 1 → 
    (4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0) := by
  sorry

end find_a_tangent_line_eq_l401_401781


namespace fourth_student_guess_l401_401558

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  fourth_guess = 525 := 
by
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  show fourth_guess = 525 from sorry

end fourth_student_guess_l401_401558


namespace max_distinct_distances_l401_401870

-- Define the setup with three blocks for each of three letters
def setup : Prop :=
  ∃ seq : list char, 
    seq.length = 9 ∧
    (seq.count 'A' = 3) ∧ 
    (seq.count 'B' = 3) ∧ 
    (seq.count 'C' = 3)

-- Define what it means to have distinct distances for blocks with the same letter
def distinct_distances (seq : list char) (c : char) : set ℕ :=
  { d | ∃ i j, i < j ∧ seq.nth i = some c ∧ seq.nth j = some c ∧ d = j - i }

-- Define the main statement
theorem max_distinct_distances : setup →
  (∃ seq : list char, 
    seq.length = 9 ∧
    (seq.count 'A' = 3) ∧ 
    (seq.count 'B' = 3) ∧ 
    (seq.count 'C' = 3) ∧
    card (distinct_distances seq 'A' ∪ distinct_distances seq 'B' ∪ distinct_distances seq 'C') = 7) :=
by
  -- Placeholder for the proof
  sorry

end max_distinct_distances_l401_401870


namespace exists_real_t_of_modulus_one_l401_401141

theorem exists_real_t_of_modulus_one (z : ℂ) (hz1 : |z| = 1) (hz2 : z ≠ -1) :
  ∃ t : ℝ, z = (1 + complex.I * t) / (1 - complex.I * t) :=
by
  sorry

end exists_real_t_of_modulus_one_l401_401141


namespace a_plus_b_eq_neg1_l401_401055

theorem a_plus_b_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : a + b = -1 :=
by
  sorry

end a_plus_b_eq_neg1_l401_401055


namespace polynomial_evaluation_l401_401269

def polynomial_at (x : ℝ) : ℝ :=
  let f := (7 : ℝ) * x^5 + 12 * x^4 - 5 * x^3 - 6 * x^2 + 3 * x - 5
  f

theorem polynomial_evaluation : polynomial_at 3 = 2488 :=
by
  sorry

end polynomial_evaluation_l401_401269


namespace original_balance_l401_401520

theorem original_balance
  (transfer1 : ℝ)
  (transfer2 : ℝ)
  (service_charge_rate : ℝ)
  (reversed_amount : ℝ)
  (current_balance : ℝ)
  (net_change : ℝ)
  (original_balance : ℝ) :
  transfer1 = 90 →
  transfer2 = 60 →
  service_charge_rate = 0.02 →
  reversed_amount = transfer2 →
  current_balance = 307 →
  net_change = (transfer1 + service_charge_rate * transfer1) - reversed_amount →
  original_balance = current_balance + net_change →
  original_balance = 338.80 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  norm_num
  sorry

end original_balance_l401_401520


namespace find_lowest_temperature_l401_401165

noncomputable def lowest_temperature 
(T1 T2 T3 T4 T5 : ℝ) : ℝ :=
if h : T1 + T2 + T3 + T4 + T5 = 200 ∧ max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 = 50 then
   min (min (min T1 T2) (min T3 T4)) T5
else 
  0

theorem find_lowest_temperature (T1 T2 T3 T4 T5 : ℝ) 
  (h_avg : T1 + T2 + T3 + T4 + T5 = 200)
  (h_range : max (max (max T1 T2) (max T3 T4)) T5 - min (min (min T1 T2) (min T3 T4)) T5 ≤ 50) : 
  lowest_temperature T1 T2 T3 T4 T5 = 30 := 
sorry

end find_lowest_temperature_l401_401165


namespace reading_hours_l401_401896

theorem reading_hours (h : ℕ) (lizaRate suzieRate : ℕ) (lizaPages suziePages : ℕ) 
  (hliza : lizaRate = 20) (hsuzie : suzieRate = 15) 
  (hlizaPages : lizaPages = lizaRate * h) (hsuziePages : suziePages = suzieRate * h) 
  (h_diff : lizaPages = suziePages + 15) : h = 3 :=
by {
  sorry
}

end reading_hours_l401_401896


namespace symmetric_circle_eq_l401_401574

theorem symmetric_circle_eq (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (x^2 + y^2 - 4 * y = 0) :=
sorry

end symmetric_circle_eq_l401_401574


namespace find_m_l401_401003

def vector := ℝ × ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_m (m : ℝ) : 
  let d : vector := (1, -2, 0)
  let n : vector := (m, 3, 6)
  (dot_product n d = 0) → m = 6 :=
by
  sorry

end find_m_l401_401003


namespace perimeter_of_new_figure_is_correct_l401_401153

-- Define the given conditions
def original_horizontal_segments := 16
def original_vertical_segments := 10
def original_side_length := 1
def new_side_length := 2

-- Define total lengths calculations
def total_horizontal_length (new_side_length original_horizontal_segments : ℕ) : ℕ :=
  original_horizontal_segments * new_side_length

def total_vertical_length (new_side_length original_vertical_segments : ℕ) : ℕ :=
  original_vertical_segments * new_side_length

-- Formulate the main theorem
theorem perimeter_of_new_figure_is_correct :
  total_horizontal_length new_side_length original_horizontal_segments + 
  total_vertical_length new_side_length original_vertical_segments = 52 := by
  sorry

end perimeter_of_new_figure_is_correct_l401_401153


namespace inequality_x4_y4_l401_401522

theorem inequality_x4_y4 (x y : ℝ) : x^4 + y^4 + 8 ≥ 8 * x * y := 
by {
  sorry
}

end inequality_x4_y4_l401_401522


namespace solve_for_M_l401_401025

def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 2 * x + y = 2 ∧ x - y = 1 }

theorem solve_for_M : M = { (1, 0) } := by
  sorry

end solve_for_M_l401_401025


namespace concyclic_intersections_of_extended_triangle_l401_401184

/-- Let ABC be a triangle. Suppose there are three lines such that:
1. Each line is parallel to one side of the triangle and at a distance equal to the length of that side.
2. For each side of the triangle, the line parallel to it and the vertex opposite to that side are on different sides of the parallel line.

Let the intersections of these lines with the extended sides of the triangle create six points of intersection.
Prove that these six points are concyclic, i.e., they lie on a single circle. -/
theorem concyclic_intersections_of_extended_triangle (A B C : Point) 
  (h1 : Parallel (Line_through A B) (Line_at_distance (Line_through B C) (length (Line_through B C))))
  (h2 : Parallel (Line_through B C) (Line_at_distance (Line_through C A) (length (Line_through C A))))
  (h3 : Parallel (Line_through C A) (Line_at_distance (Line_through A B) (length (Line_through A B))))
  (h4 : Opposite_sides (Vertex A) (Line_parallel_to_side B C))
  (h5 : Opposite_sides (Vertex B) (Line_parallel_to_side C A))
  (h6 : Opposite_sides (Vertex C) (Line_parallel_to_side A B)) : 
  concyclic_6_points (points_of_intersection (A B C)) := 
sorry

end concyclic_intersections_of_extended_triangle_l401_401184


namespace zain_has_80_coins_l401_401207

theorem zain_has_80_coins (emerie_quarters emerie_dimes emerie_nickels emerie_pennies emerie_half_dollars : ℕ)
  (h_quarters : emerie_quarters = 6) 
  (h_dimes : emerie_dimes = 7)
  (h_nickels : emerie_nickels = 5)
  (h_pennies : emerie_pennies = 10) 
  (h_half_dollars : emerie_half_dollars = 2) : 
  10 + emerie_quarters + 10 + emerie_dimes + 10 + emerie_nickels + 10 + emerie_pennies + 10 + emerie_half_dollars = 80 :=
by
  sorry

end zain_has_80_coins_l401_401207


namespace expression_decreases_by_37_of_64_value_decrease_final_statement_l401_401454

theorem expression_decreases_by_37_of_64 (x y : ℝ) : 
  (0.75 * x) * (0.75 * y)^2 = (27 / 64) * (x * y^2) :=
by
  sorry

theorem value_decrease (x y : ℝ) :
   1 - (27 / 64) = (37 / 64) :=
by
  sorry

theorem final_statement (x y : ℝ) :
   ((1 : ℝ) - (expression_decreases_by_37_of_64 x y)) = (37 / 64) :=
by
  sorry

end expression_decreases_by_37_of_64_value_decrease_final_statement_l401_401454


namespace sum_of_fractions_correct_l401_401271

def sum_of_fractions : ℚ := (4 / 3) + (8 / 9) + (18 / 27) + (40 / 81) + (88 / 243) - 5

theorem sum_of_fractions_correct : sum_of_fractions = -305 / 243 := by
  sorry -- proof to be provided

end sum_of_fractions_correct_l401_401271


namespace find_f_x_l401_401365

theorem find_f_x (f : ℝ → ℝ) 
  (h : ∀ x, f (√x + 1) = x + 2 * √x) 
  (hx : ∀ x, x ≥ 1 → f x = x^2 - 1) : 
  ∀ x, x ≥ 1 → f x = x^2 - 1 :=
sorry

end find_f_x_l401_401365


namespace silas_payment_ratio_l401_401944

theorem silas_payment_ratio (total_bill : ℕ) (tip_rate : ℝ) (friend_payment : ℕ) (S : ℕ) :
  total_bill = 150 →
  tip_rate = 0.10 →
  friend_payment = 18 →
  (S + 5 * friend_payment = total_bill + total_bill * tip_rate) →
  (S : ℝ) / total_bill = 1 / 2 :=
by
  intros h_total_bill h_tip_rate h_friend_payment h_budget_eq
  sorry

end silas_payment_ratio_l401_401944


namespace min_length_ab_l401_401912

noncomputable def minimum_distance : ℝ :=
  let d (x1 x2 : ℝ) :=
    Real.sqrt ((x2 - x1)^2 + (x2^2 - ((5/12) * x1 - 11))^2)
  -- Here we use a dummy value for the supposed minimum distance.
  -- The actual proof to find the minimum distance should be filled in.
  (6311 / 624)

theorem min_length_ab :
  ∃ (x1 x2 : ℝ), (y1 y2 : ℝ), y1 = (5/12) * x1 - 11 ∧ y2 = x2^2 ∧ 
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 6311 / 624) := 
sorry

end min_length_ab_l401_401912


namespace polynomials_identical_l401_401968

-- Required for noncomputable theory and dealing with polynomials
noncomputable theory

open Polynomial

-- Statement of the problem
theorem polynomials_identical
  {f g : Polynomial ℤ} 
  (h : ∃ t : ℤ, (∀ i, |coeff f i| ≤ t / 2 ∧ |coeff g i| ≤ t / 2) ∧ f.eval t = g.eval t) :
  f = g :=
begin
  -- Proof steps omitted
  sorry
end

end polynomials_identical_l401_401968


namespace trigonometric_expression_l401_401000

theorem trigonometric_expression (x : ℝ) (sin_x : ℝ := 3/5) (cos_x : ℝ := -4/5)
  (h1: sin x = sin_x) (h2: cos x = cos_x) :
  (cos(π / 2 + x) * sin(-π - x)) / (cos(π / 2 - x) * sin(9 * π / 2 + x)) = 3 / 4 :=
by
  sorry

end trigonometric_expression_l401_401000


namespace highest_percentage_without_car_l401_401508

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l401_401508


namespace problem_S_equal_102_l401_401919

-- Define the values in Lean
def S : ℕ := 1 * 3^1 + 2 * 3^2 + 3 * 3^3

-- Theorem to prove that S is equal to 102
theorem problem_S_equal_102 : S = 102 :=
by
  sorry

end problem_S_equal_102_l401_401919


namespace ratio_difference_sequence_propositions_true_l401_401083

theorem ratio_difference_sequence_propositions_true :
  (∀ {a : ℕ → ℝ}, (∀ n : ℕ, a(n + 2) / a(n + 1) - a(n + 1) / a(n) = 0) → true) ∧
  (∀ {a : ℕ → ℝ}, (∀ n : ℕ, a n = 2^(n - 1) / n^2) → false) ∧
  (∀ {c : ℕ → ℝ}, (c 1 = 1 ∧ c 2 = 1 ∧ ∀ n, n ≥ 3 → c n = c(n - 1) + c(n - 2)) → true) ∧
  (∀ {a b : ℕ → ℝ}, (∀ n : ℕ, a n = 0 ∨ b n = 1) → false) := sorry

end ratio_difference_sequence_propositions_true_l401_401083


namespace final_solution_percentage_l401_401929

variable (initial_volume replaced_fraction : ℝ)
variable (initial_concentration replaced_concentration : ℝ)

noncomputable
def final_acid_percentage (initial_volume replaced_fraction initial_concentration replaced_concentration : ℝ) : ℝ :=
  let remaining_volume := initial_volume * (1 - replaced_fraction)
  let replaced_volume := initial_volume * replaced_fraction
  let remaining_acid := remaining_volume * initial_concentration
  let replaced_acid := replaced_volume * replaced_concentration
  let total_acid := remaining_acid + replaced_acid
  let final_volume := initial_volume
  (total_acid / final_volume) * 100

theorem final_solution_percentage :
  final_acid_percentage 100 0.5 0.5 0.3 = 40 :=
by
  sorry

end final_solution_percentage_l401_401929


namespace sqrt_product_simplification_l401_401266

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) := 
by 
  sorry

end sqrt_product_simplification_l401_401266


namespace quadratic_equation_original_eq_l401_401998

theorem quadratic_equation_original_eq :
  ∃ (α β : ℝ), (α + β = 3) ∧ (α * β = -6) ∧ (∀ (x : ℝ), x^2 - 3 * x - 6 = 0 → (x = α ∨ x = β)) :=
sorry

end quadratic_equation_original_eq_l401_401998


namespace defective_units_shipped_l401_401851

theorem defective_units_shipped (total_units : ℕ) (type_A_defect_rate type_B_defect_rate : ℝ)
(rework_rate_A rework_rate_B : ℝ) (ship_rate_A ship_rate_B : ℝ)
(h_total_units_positive : 0 < total_units)
(h_A_defect_rate : type_A_defect_rate = 0.07)
(h_B_defect_rate : type_B_defect_rate = 0.08)
(h_A_rework_rate : rework_rate_A = 0.40)
(h_B_rework_rate : rework_rate_B = 0.30)
(h_A_ship_rate : ship_rate_A = 0.03)
(h_B_ship_rate : ship_rate_B = 0.06) :
  let defective_A := total_units * type_A_defect_rate,
      defective_B := total_units * type_B_defect_rate,
      reworked_A := defective_A * rework_rate_A,
      reworked_B := defective_B * rework_rate_B,
      remaining_A := defective_A - reworked_A,
      remaining_B := defective_B - reworked_B,
      shipped_A := remaining_A * ship_rate_A,
      shipped_B := remaining_B * ship_rate_B,
      total_shipped_defective := (shipped_A + shipped_B) / total_units * 100 in
  total_shipped_defective = 0.462 := 
by
  sorry

end defective_units_shipped_l401_401851


namespace parallel_vectors_l401_401362

-- Declaring vectors a and b as given in the problem statement
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Proving that if a is parallel to b, then x = 6
theorem parallel_vectors (x : ℝ) (h : a.1 * b x.2 = b x.1 * a.2) : x = 6 :=
by
  sorry

end parallel_vectors_l401_401362


namespace cube_root_of_2x_minus_23_l401_401396

theorem cube_root_of_2x_minus_23 (x : ℝ) (h : sqrt (2 * x - 1) = 7 ∨ sqrt (2 * x - 1) = -7) : real.cbrt (2 * x - 23) = 3 :=
by
  sorry

end cube_root_of_2x_minus_23_l401_401396


namespace triangle_tangent_half_angle_l401_401860

theorem triangle_tangent_half_angle (a b c : ℝ) (A : ℝ) (C : ℝ)
  (h : a + c = 2 * b) :
  Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3 := 
sorry

end triangle_tangent_half_angle_l401_401860


namespace length_of_wooden_block_l401_401950

theorem length_of_wooden_block (cm_to_m : ℝ := 30 / 100) (base_length : ℝ := 31) :
  base_length + cm_to_m = 31.3 :=
by
  sorry

end length_of_wooden_block_l401_401950


namespace cosine_double_angle_l401_401753

variable (α : Real)

theorem cosine_double_angle (h1 : Real.sin α = 4 / 5) (h2 : α ∈ Ioo (π / 2) π) : Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end cosine_double_angle_l401_401753


namespace intersection_cond_rectangular_to_polar_C1_problem_proof_l401_401456

open Real

def parametric_eq_C1 (α t : ℝ) : ℝ × ℝ := (1 + t * cos α, 1 + t * sin α)

def polar_eq_C2 (θ : ℝ) : ℝ := 4 * cos θ

def rectangular_eq_C2 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x

def point_P : ℝ × ℝ := (1, 1)

def PA (t : ℝ) (α : ℝ) : ℝ := Real.sqrt ((1 + t * cos α - 1)^2 + (1 + t * sin α - 1)^2)

theorem intersection_cond (α : ℝ) (t1 t2 : ℝ) (h1 : t1 + t2 = -2 * (sin α - cos α))
    (h2 : t1 * t2 = -2) (h3 : (PA t1 α)^2 * (PA t2 α)^2 = 1) : 
    α = π / 4 :=
sorry

theorem rectangular_to_polar_C1 (α : ℝ) (h : α = π / 4) : ∀ ρ : ℝ, ∃ θ : ℝ, θ = π / 4 :=
sorry

theorem problem_proof (α : ℝ) (h1 : α ∈ Icc 0 π) (t1 t2 : ℝ) (h2 : t1 + t2 = -2 * (sin α - cos α))
    (h3 : t1 * t2 = -2) (h4 : 1 / (PA t1 α)^2 + 1 / (PA t2 α)^2 = 1) : 
    (∀ x y : ℝ, rectangular_eq_C2 x y) ∧ ∀ ρ : ℝ, ∃ θ : ℝ, θ = π / 4 :=
begin
  split,
  { -- Proof that the rectangular coordinate equation of \(C_{2}\) is \(x^2 + y^2 = 4x\).
    intros x y,
    unfold_rectangular_eq_C2,
    sorry
  },
  { -- Proof that the polar coordinate equation of \(C_{1}\) is \(\theta = \frac{\pi}{4}\).
    apply rectangular_to_polar_C1,
    apply intersection_cond,
    all_goals { assumption }
  }
end

end intersection_cond_rectangular_to_polar_C1_problem_proof_l401_401456


namespace find_g_l401_401542

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l401_401542


namespace period_of_tan_2x_add_pi_l401_401717

theorem period_of_tan_2x_add_pi : ∀ (x : ℝ), ∃ T : ℝ, (∀ x : ℝ, tan (2 * x + π) = tan (2 * (x + T) + π)) ∧ T = π / 2 :=
by
  sorry

end period_of_tan_2x_add_pi_l401_401717


namespace calculation_l401_401856

noncomputable def seq (n : ℕ) : ℕ → ℚ := sorry

axiom cond1 : ∀ (n : ℕ), seq (n + 1) - 2 * seq n = 0
axiom cond2 : ∀ (n : ℕ), seq n ≠ 0

theorem calculation :
  (2 * seq 1 + seq 2) / (seq 3 + seq 5) = 1 / 5 :=
  sorry

end calculation_l401_401856


namespace find_a7_l401_401775

-- Define the arithmetic sequence
def a (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions
def S5 : ℤ := 25
def a2 : ℤ := 3

-- Main Goal: Find a_7
theorem find_a7 (a1 d : ℤ) (h1 : sum_first_n_terms a1 d 5 = S5)
                     (h2 : a a1 d 2 = a2) :
  a a1 d 7 = 13 := 
sorry

end find_a7_l401_401775


namespace problem1_problem2_l401_401699

variable (a m : ℕ)

-- Problem 1
theorem problem1 : (a^6 / a^2) = a^4 := by
  -- Sorry added to skip the proof for now
  sorry

-- Problem 2
theorem problem2 : (m^2 * m^4 - (2 * m^3)^2) = -3 * m^6 := by
  -- Sorry added to skip the proof for now
  sorry

end problem1_problem2_l401_401699


namespace not_divisible_by_121_l401_401084

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 2014)) :=
sorry

end not_divisible_by_121_l401_401084


namespace find_lambda_l401_401414

-- Define vector and point types
structure Vector2D where
  x : ℝ
  y : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define vector subtraction for points to get a vector
def vectorFromPoints (A B : Point) : Vector2D :=
  ⟨B.x - A.x, B.y - A.y⟩

-- Define dot product between vectors.
def dotProduct (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the magnitude of a vector.
def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y)

-- Define the projection of one vector onto another
def projection (v1 v2 : Vector2D) : Vector2D :=
  let c := (dotProduct v1 v2) / (magnitude v2 ^ 2)
  ⟨c * v2.x, c * v2.y⟩

-- Given conditions
def a : Vector2D := ⟨-4, 3⟩
def A : Point := ⟨1, 1⟩
def B : Point := ⟨2, -1⟩

-- Define the vector AB
def AB : Vector2D := vectorFromPoints A B

-- The main statement to prove in Lean 4
theorem find_lambda : ∃ λ : ℝ, projection AB a = λ • a ∧ λ = -2/5 :=
by
  sorry

end find_lambda_l401_401414


namespace Linda_total_sales_l401_401491

theorem Linda_total_sales (necklaces_sold : ℕ) (rings_sold : ℕ) 
    (necklace_price : ℕ) (ring_price : ℕ) 
    (total_sales : ℕ) : 
    necklaces_sold = 4 → 
    rings_sold = 8 → 
    necklace_price = 12 → 
    ring_price = 4 → 
    total_sales = necklaces_sold * necklace_price + rings_sold * ring_price → 
    total_sales = 80 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end Linda_total_sales_l401_401491


namespace sum_odd_divisors_300_l401_401623

theorem sum_odd_divisors_300 :
  let n := 300
  let prime_factorization := (2^2 * 3 * 5^2)
  let sum_odd_divisors (n : ℕ) : ℕ :=
  have odd_divisors := [1, 3, 5, 9, 15, 25, 45, 75, 225]
  odd_divisors.foldl (+) 0
  sum_odd_divisors n = 124 := by
  sorry

end sum_odd_divisors_300_l401_401623


namespace sum_series_l401_401705

theorem sum_series : ∑ i in Finset.range 51, (-1)^((i + 1) : ℤ) * (i + 1) = 51 := by sorry

end sum_series_l401_401705


namespace molecular_weight_NaClO_is_74_44_l401_401727

-- Define the atomic weights
def atomic_weight_Na : Real := 22.99
def atomic_weight_Cl : Real := 35.45
def atomic_weight_O : Real := 16.00

-- Define the calculation of molecular weight
def molecular_weight_NaClO : Real :=
  atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

-- Define the theorem statement
theorem molecular_weight_NaClO_is_74_44 :
  molecular_weight_NaClO = 74.44 :=
by
  -- Placeholder for proof
  sorry

end molecular_weight_NaClO_is_74_44_l401_401727


namespace factorial_fraction_is_integer_l401_401357

theorem factorial_fraction_is_integer (n : ℕ) (h : n > 0) : ∃ k : ℕ, (3 * n)! = k * (6 ^ n * n!) :=
by
  sorry

end factorial_fraction_is_integer_l401_401357


namespace value_of_x_squared_plus_reciprocal_squared_l401_401807

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l401_401807


namespace max_min_of_expression_is_sqrt2_l401_401390

noncomputable def max_min_expression (x y : ℝ) : ℝ := 
  min (min x (1 / y)) (1 / x + y)

theorem max_min_of_expression_is_sqrt2 
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, m = sqrt 2 ∧ max_min_expression x y ≤ m := 
sorry

end max_min_of_expression_is_sqrt2_l401_401390


namespace difference_of_numbers_l401_401961

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : abs (x - y) = 7 :=
sorry

end difference_of_numbers_l401_401961


namespace determinant_of_pascal_submatrix_is_one_l401_401716

def pascal_triangle : ℕ → ℕ → ℕ 
| 0, _ := 1
| _, 0 := 1
| n+1, k+1 := pascal_triangle n k + pascal_triangle n (k+1)

theorem determinant_of_pascal_submatrix_is_one (n m : ℕ) (H : 1 ≤ n) (Hk : 1 ≤ m) :
  ∀ (A : Matrix (Fin n) (Fin m) ℕ), 
  (∀ i j, A i j = pascal_triangle (i + 1) (j + 1)) → 
  |A.det| = 1 := 
by
  sorry

end determinant_of_pascal_submatrix_is_one_l401_401716


namespace rectangle_min_area_l401_401667

theorem rectangle_min_area (l w : ℕ) (h1 : 2 * (l + w) = 60) (h2 : 1 ≤ l) : l * w = 29 :=
by
  have h3 : l + w = 30 := by linarith [h1]
  have w_val : w = 30 - l := by linarith [h3]
  suffices hl_small : l = 1 by
    rw [hl_small, w_val]
    linarith [min, max]
  sorry

end rectangle_min_area_l401_401667


namespace tom_total_dimes_l401_401616

-- Define the original and additional dimes Tom received.
def original_dimes : ℕ := 15
def additional_dimes : ℕ := 33

-- Define the total number of dimes Tom has now.
def total_dimes : ℕ := original_dimes + additional_dimes

-- Statement to prove that the total number of dimes Tom has is 48.
theorem tom_total_dimes : total_dimes = 48 := by
  sorry

end tom_total_dimes_l401_401616


namespace no_treasures_on_island_l401_401457

variables (K G : Prop)
-- Condition that A is either truth-teller or liar is implicitly included
axiom A_is_knight_or_knave : (K ↔ true) ∨ (K ↔ false)
-- A's response "no" to "is K equivalent to G"
axiom A_response : (K ↔ G) = false

theorem no_treasures_on_island (h : A_is_knight_or_knave) (n : A_response) : G = false :=
sorry

end no_treasures_on_island_l401_401457


namespace remainder_of_1997_pow_2000_div_7_l401_401590

theorem remainder_of_1997_pow_2000_div_7 :
  (1997 ^ 2000) % 7 = 4 :=
by
  -- We start by defining our known cycle based on the problem conditions
  have h1 : 1997 % 7 = 2 := by norm_num,
  have h2 : 1997 ^ 2 % 7 = 4 := by norm_num,
  have h3 : 1997 ^ 3 % 7 = 0 := by norm_num,

  -- Determine the position in the cycle: 2000 mod 3 = 2
  have h4 : 2000 % 3 = 2 := by norm_num,

  -- Use the pattern and get the result from the cycle
  have key : (1997 ^ 2000 % 7) = (2 ^ 2000 % 7) := sorry,
  have calc_cycle : (2 ^ 2000) % 7 = 4 := sorry,
  
  exact calc_cycle

end remainder_of_1997_pow_2000_div_7_l401_401590


namespace find_g_l401_401545

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l401_401545


namespace cosine_periodicity_l401_401339

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401339


namespace ship_with_highest_no_car_round_trip_percentage_l401_401517

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l401_401517


namespace annual_earning_l401_401629

variable (monthly_salary : ℕ) (months_in_year : ℕ)
variable (monthly_salary_prop : monthly_salary = 4380)
variable (months_in_year_prop : months_in_year = 12)

theorem annual_earning :
  monthly_salary * months_in_year = 52560 :=
by
  rw [monthly_salary_prop, months_in_year_prop]
  sorry

end annual_earning_l401_401629


namespace find_n_cosine_l401_401313

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401313


namespace smallest_four_digit_palindrome_divisible_by_4_l401_401622

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def smallest_palindrome_divisible_by_4 : ℕ :=
  2112

theorem smallest_four_digit_palindrome_divisible_by_4 :
  is_palindrome smallest_palindrome_divisible_by_4 ∧
  is_four_digit smallest_palindrome_divisible_by_4 ∧
  is_divisible_by_4 smallest_palindrome_divisible_by_4 ∧
  ∀ n : ℕ, is_palindrome n ∧ is_four_digit n ∧ is_divisible_by_4 n → smallest_palindrome_divisible_by_4 ≤ n :=
  by sorry

end smallest_four_digit_palindrome_divisible_by_4_l401_401622


namespace calculate_binom_and_fact_l401_401687

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the factorial function
def fact (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n + 1)) (λ i, i)

-- The main theorem to prove
theorem calculate_binom_and_fact : binom 12 10 * fact 5 = 7920 := by
  sorry

end calculate_binom_and_fact_l401_401687


namespace sum_of_cubes_sign_l401_401285

theorem sum_of_cubes_sign:
  let a := (Real.sqrt 2021) - (Real.sqrt 2020)
      b := (Real.sqrt 2020) - (Real.sqrt 2019)
      c := (Real.sqrt 2019) - (Real.sqrt 2018)
  in a + b + c = 0 →
     a^3 + b^3 + c^3 < 0 :=
by 
  intros a b c h_sum
  simp [a, b, c] at *
  sorry

end sum_of_cubes_sign_l401_401285


namespace find_integer_cosine_l401_401323

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401323


namespace solution_set_inequality_l401_401116

variable {f : ℝ → ℝ}

theorem solution_set_inequality (h1 : f 1 = 1) (h2 : ∀ x : ℝ, f' x < 1/2) :
  {x : ℝ | f (Real.log2 x) > (Real.log2 x + 1) / 2} = set.Ioo 0 2 :=
sorry

end solution_set_inequality_l401_401116


namespace find_a_of_complex_division_l401_401815

theorem find_a_of_complex_division (a : ℝ) : 
  ( (2 + a * complex.I) / (1 + complex.I) = -2 * complex.I ) → a = -2 := 
by
  sorry

end find_a_of_complex_division_l401_401815


namespace symmetric_axis_shifted_function_l401_401404

theorem symmetric_axis_shifted_function :
  ∀ (x : ℝ), (let f := λ x, sqrt 3 * sin (2 * x) - cos (2 * x)
               in ∃ k : ℤ, x = k * (π / 2) + (2 * π / 3)) →
            (let f_shifted := λ x, 2 * sin (2 * (x - π / 3) - π / 6)
            in ∃ k : ℤ, x = (k * (π / 2) + 2 * π / 3)) →
            (x = π / 6) :=
by
  sorry

end symmetric_axis_shifted_function_l401_401404


namespace cube_root_of_expression_l401_401395

theorem cube_root_of_expression (x : ℝ) (h : (2 * x - 1) = 49) : (∛(2 * x - 23)) = 3 :=
by
  sorry

end cube_root_of_expression_l401_401395


namespace find_m_l401_401002

def direction_vector : ℝ × ℝ × ℝ := (1, -2, 0)
def normal_vector (m : ℝ) : ℝ × ℝ × ℝ := (m, 3, 6)

theorem find_m (m : ℝ) :
  let d := direction_vector,
      n := normal_vector m
  in d.1 * n.1 + d.2 * n.2 + d.3 * n.3 = 0 → m = 6 :=
by
  let d := direction_vector
  let n := normal_vector m
  intros h
  sorry

end find_m_l401_401002


namespace range_is_4_mode_is_correct_median_is_22_percentile_80_is_23_l401_401439

-- Define the data set
def temperatures : List ℕ := [22, 21, 20, 20, 22, 23, 24]

-- Prove the range
theorem range_is_4 : (List.maximum temperatures - List.minimum temperatures) = 4 := 
  sorry

-- Prove the mode
theorem mode_is_correct : (List.mode temperatures) = {20, 22} := 
  sorry

-- Prove the median
theorem median_is_22 : (List.median temperatures) = 22 :=
  sorry

-- Prove the 80th percentile
theorem percentile_80_is_23 : (List.percentile temperatures 80) = 23 :=
  sorry

end range_is_4_mode_is_correct_median_is_22_percentile_80_is_23_l401_401439


namespace neighbors_receive_equal_mangoes_l401_401131

-- Definitions from conditions
def total_mangoes : ℕ := 560
def mangoes_sold : ℕ := total_mangoes / 2
def remaining_mangoes : ℕ := total_mangoes - mangoes_sold
def neighbors : ℕ := 8

-- The lean statement
theorem neighbors_receive_equal_mangoes :
  remaining_mangoes / neighbors = 35 :=
by
  -- This is where the proof would go, but we'll leave it with sorry for now.
  sorry

end neighbors_receive_equal_mangoes_l401_401131


namespace marie_socks_problem_l401_401125

theorem marie_socks_problem (x y z : ℕ) : 
  x + y + z = 15 → 
  2 * x + 3 * y + 5 * z = 36 → 
  1 ≤ x → 
  1 ≤ y → 
  1 ≤ z → 
  x = 11 :=
by
  sorry

end marie_socks_problem_l401_401125


namespace fraction_of_male_first_class_l401_401906

theorem fraction_of_male_first_class (total_passengers : ℕ) (percent_female : ℚ) (percent_first_class : ℚ)
    (females_in_coach : ℕ) (h1 : total_passengers = 120) (h2 : percent_female = 0.45) (h3 : percent_first_class = 0.10)
    (h4 : females_in_coach = 46) :
    (((percent_first_class * total_passengers - (percent_female * total_passengers - females_in_coach)))
    / (percent_first_class * total_passengers))  = 1 / 3 := 
by
  sorry

end fraction_of_male_first_class_l401_401906


namespace bumper_cars_number_of_tickets_l401_401460

theorem bumper_cars_number_of_tickets (Ferris_Wheel Roller_Coaster Jeanne_Has Jeanne_Buys : ℕ)
  (h1 : Ferris_Wheel = 5)
  (h2 : Roller_Coaster = 4)
  (h3 : Jeanne_Has = 5)
  (h4 : Jeanne_Buys = 8) :
  Ferris_Wheel + Roller_Coaster + (13 - (Ferris_Wheel + Roller_Coaster)) = 13 - (Ferris_Wheel + Roller_Coaster) :=
by
  sorry

end bumper_cars_number_of_tickets_l401_401460


namespace triangle_inequalities_l401_401458

variable {A B C : ℝ} -- Angles in the triangle
variable {a b c : ℝ} -- Sides of the triangle
variable {p : ℝ} -- Semi-perimeter of the triangle
variable {R r S_triangle : ℝ} -- Circumradius, inradius, and area of the triangle

-- Conditions representing the inequalities
axiom cos_inequality : cos A + cos B + cos C <= 3 / 2
axiom sin_inequality : sin (A / 2) * sin (B / 2) * sin (C / 2) <= 1 / 8
axiom abc_inequality : a * b * c >= 8 * (p - a) * (p - b) * (p - c)
axiom circumradius_inequality : R >= 2 * r
axiom area_inequality : S_triangle <= (1 / 2) * R * p

-- The proposition we need to prove
theorem triangle_inequalities :
  cos A + cos B + cos C <= 3 / 2 ∧
  sin (A / 2) * sin (B / 2) * sin (C / 2) <= 1 / 8 ∧
  a * b * c >= 8 * (p - a) * (p - b) * (p - c) ∧
  R >= 2 * r ∧
  S_triangle <= (1 / 2) * R * p :=
by {
  exact ⟨cos_inequality, sin_inequality, abc_inequality, circumradius_inequality, area_inequality⟩
}

end triangle_inequalities_l401_401458


namespace largest_determinant_l401_401883

open Matrix

noncomputable def u : ℝ^3 := sorry
def v : ℝ^3 := ![3, 2, -2]
def w : ℝ^3 := ![2, -1, 4]

theorem largest_determinant :
  ∃ u : ℝ^3, (∥u∥ = 1) ∧ 
  |u.det v w| = √149 :=
sorry

end largest_determinant_l401_401883


namespace sum_of_integers_satisfying_l401_401601

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l401_401601


namespace find_angle_BAC_l401_401557

noncomputable def obtuse_triangle : Type :=
{ A B C : point | acute_triangle A B C ∧ ¬ is_isosceles A B C }

variables {A B C H I O : point}

def triangle_abc (t : triangl) :=
t.A = A ∧ t.B = B ∧ t.C = C ∧
acute_triangle t.A t.B t.C ∧ ¬ is_isosceles t.A t.B t.C ∧
intersection (altitude t t.A t.B t.C) (altitude t t.B t.C t.A) (altitude t t.C t.A t.B) = H ∧
incenter t t.A t.B t.C = I ∧
circumcenter (triangl (H t.B t.C)) = O ∧
on_line_segment I O A

theorem find_angle_BAC :
  acute_triangle A B C →
  ¬ is_isosceles A B C →
  intersection (altitude A B C) (altitude B C A) (altitude C A B) = H →
  incenter A B C = I →
  circumcenter (BHC H B C) = O →
  on_line_segment I O A →
  ∠BAC = 60 :=
sorry

end find_angle_BAC_l401_401557


namespace corvette_trip_time_percentage_increase_l401_401898

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l401_401898


namespace intersection_at_least_one_element_l401_401027

noncomputable def M (k : ℝ) : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 1 = k * (x + 1))}
noncomputable def N : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (x^2 + y^2 - 2 * y = 0)}

theorem intersection_at_least_one_element (k : ℝ) : ∃ p : ℝ × ℝ, p ∈ M k ∧ p ∈ N :=
begin
  use (-1, 1),
  split,
  { unfold M,
    use [-1, 1],
    split,
    { refl, },
    { linarith, }, },
  { unfold N,
    use [-1, 1],
    split,
    { refl, },
    { norm_num, }, },
end

end intersection_at_least_one_element_l401_401027


namespace initial_workers_l401_401930

theorem initial_workers (W : ℕ) (work1 : ℕ) (work2 : ℕ) :
  (work1 = W * 8 * 30) →
  (work2 = (W + 35) * 6 * 40) →
  (work1 / 30 = work2 / 40) →
  W = 105 :=
by
  intros hwork1 hwork2 hprop
  sorry

end initial_workers_l401_401930


namespace sum_of_exponents_equals_27_l401_401613

theorem sum_of_exponents_equals_27 :
  ∃ (r : ℕ) (n : fin r → ℕ) (a : fin r → ℤ),
    (∀ i j, i < j → n i > n j) ∧
    (∀ i, a i = 1 ∨ a i = -1) ∧
    (∑ i, a i * 3 ^ n i = 1729) →
    (∑ i, n i = 27) :=
by
  sorry

end sum_of_exponents_equals_27_l401_401613


namespace product_of_b_l401_401713

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (y + 4) / 3

theorem product_of_b (b : ℝ) :
  g b 3 = g_inv b (b + 2) → b = 3 := 
by
  sorry

end product_of_b_l401_401713


namespace exist_odd_prime_factor_l401_401377

theorem exist_odd_prime_factor (k : ℕ) (a : ℕ) (p : ℕ → ℕ)
  (hk : 2 ≤ k)
  (hp : ∀ i, i < k → prime (p i) ∧ p i % 2 = 1)
  (hdistinct : ∀ i j, i < k → j < k → i ≠ j → p i ≠ p j)
  (hcoprime : gcd a (List.prod (List.ofFn p k)) = 1)
  : ∃ q, prime q ∧ q % 2 = 1 ∧ (∀ i, i < k → q ≠ p i) ∧ q ∣ (a ^ (List.prod (List.map (λ i, (p i - 1)) (List.range k)) - 1)) :=
sorry

end exist_odd_prime_factor_l401_401377


namespace jenna_round_trip_pay_l401_401090

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l401_401090


namespace quadratic_roots_l401_401155

theorem quadratic_roots (b c : ℝ) :
  is_root (λ x : ℂ, x^2 + b * x + c) (1 - complex.I * real.sqrt 2) ∧
  (∀ (z : ℂ), (polynomial.eval z (x^2 + polynomial.C b * x + polynomial.C c)).im = 0) →
  b = -2 ∧ c = 3 :=
begin
  sorry
end

end quadratic_roots_l401_401155


namespace num_pos_integers_satisfy_condition_l401_401046

def num_valid_n (n : ℕ) : ℕ :=
  if (n + 500) / 50 = (Int.toNat ∘ Real.floor ∘ Nat.sqrt) n then 1 else 0

theorem num_pos_integers_satisfy_condition :
  ∑ n in (Finset.range 1501).filter (λ n, n > 0), num_valid_n n = 2 :=
by
  sorry

end num_pos_integers_satisfy_condition_l401_401046


namespace remainder_of_power_of_five_modulo_500_l401_401690

theorem remainder_of_power_of_five_modulo_500 :
  (5 ^ (5 ^ (5 ^ 2))) % 500 = 25 :=
by
  sorry

end remainder_of_power_of_five_modulo_500_l401_401690


namespace count_valid_4digit_numbers_l401_401039

theorem count_valid_4digit_numbers : 
  let valid_first_two_digits := {2, 3, 7}
  let valid_last_two_digits := {0, 1, 9}
  ∃ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ 
    (n / 1000 ∈ valid_first_two_digits) ∧ 
    ((n / 100) % 10 ∈ valid_first_two_digits) ∧
    (n % 10 ∈ valid_last_two_digits) ∧
    ((n / 10) % 10 ∈ valid_last_two_digits) ∧ 
    ((n % 10) ≠ ((n / 10) % 10)) ∧
    (n.compute_number_of_valid_digits = 54)
  sorry

end count_valid_4digit_numbers_l401_401039


namespace companyA_delivery_timeliness_prob_distribution_of_X_company_choice_for_timeliness_l401_401973

/-- Problem (1) -/
theorem companyA_delivery_timeliness_prob (total_A: ℕ) (delivery_A_75_or_more: ℕ) :
    total_A = 120 → 
    delivery_A_75_or_more = 29 + 47 → 
    (delivery_A_75_or_more : ℚ) / total_A = 19 / 30 := 
by
  intros h1 h2
  sorry

/-- Problem (2) -/
theorem distribution_of_X (total_A total_B satisfaction_A_75_or_more satisfaction_B_75_or_more: ℕ) :
    total_A = 120 → 
    total_B = 80 → 
    satisfaction_A_75_or_more = 24 + 56 → 
    satisfaction_B_75_or_more = 12 + 48 → 
    let P_C := (satisfaction_A_75_or_more : ℚ) / total_A in
    let P_D := (satisfaction_B_75_or_more : ℚ) / total_B in
    (P_C, P_D) = (2/3, 3/4) ∧ 
    (1 - P_C) * (1 - P_D) = 1/12 ∧ 
    (1 - P_C) * P_D + P_C * (1 - P_D) = 5/12 ∧ 
    P_C * P_D = 1/2 ∧ 
    P_X = 17/12 :=
by
  intros h1 h2 h3 h4
  let P_C := (satisfaction_A_75_or_more : ℚ) / total_A
  let P_D := (satisfaction_B_75_or_more : ℚ) / total_B
  sorry

/-- Problem (3) -/
theorem company_choice_for_timeliness (total_A total_B timeliness_A_85_or_more timeliness_B_85_or_more ratings: ℕ) :
    total_A = 120 → 
    total_B = 80 → 
    timeliness_A_85_or_more = 29 → 
    timeliness_B_85_or_more = 16 → 
    (timeliness_A_85_or_more : ℚ) / total_A = 29 / 120 ∧ 
    (timeliness_B_85_or_more : ℚ) / total_B = 16 / 80 := 
by
  intros h1 h2 h3 h4
  sorry

end companyA_delivery_timeliness_prob_distribution_of_X_company_choice_for_timeliness_l401_401973


namespace only_even_and_increasing_function_l401_401256

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b → f a ≤ f b

theorem only_even_and_increasing_function
  (f_A f_B f_C f_D : ℝ → ℝ) :
  f_A = (λ x, Real.cos x) →
  f_B = (λ x, -x^3) →
  f_C = (λ x, (1/2)^Real.abs x) →
  f_D = (λ x, Real.abs (Real.sin x)) →
  (∀ f, (f = f_A ∨ f = f_B ∨ f = f_C ∨ f = f_D) →
        (is_even_function f ∧ is_monotonically_increasing f 0 1) ↔ f = f_D) :=
by
  intros f_A_def f_B_def f_C_def f_D_def
  sorry

end only_even_and_increasing_function_l401_401256


namespace marathon_distance_covered_l401_401662

theorem marathon_distance_covered 
  (n_marathons : ℕ) (km_per_marathon : ℕ) (m_per_marathon : ℕ)
  (km_per_m : ℕ) (total_km : ℕ) (m : ℕ) :
  n_marathons = 15 →
  km_per_marathon = 42 →
  m_per_marathon = 195 →
  km_per_m = 1000 →
  (0 ≤ m ∧ m < km_per_m) →
  m = 15 * m_per_marathon % km_per_m :=
begin
  sorry
end

end marathon_distance_covered_l401_401662


namespace problem1_problem2_l401_401406

theorem problem1 (x : ℝ) (hx : 0 < x ∧ x < 5) : 
  ∃ m : ℝ, (∃ (x : ℝ), 0 < x ∧ x < 5 ∧ (4 / x + 9 / (5 - x) = m)) ∧ ∀ x, (0 < x ∧ x < 5) → (4 / x + 9 / (5 - x) ≥ m) := 
begin
  use 5,
  split,
  { use 2,
    split, linarith,
    split, linarith,
    linarith, },
  { intros x hx,
    linarith, },
end

theorem problem2 : 
  ( |x - 5| + |x + 2| ≤ 9 → -3 ≤ x ∧ x ≤ 6 ) ∧ (-3 ≤ x ∧ x ≤ 6 → |x - 5| + |x + 2| ≤ 9 ) :=
begin
  split,
  { intros h,
    by_cases h1: x < -2,
    { split, by linarith [h1],
      sorry, },
    by_cases h2: x > 5,
    { split, sorry,
      by linarith [h2], },
    split, linarith, linarith, },
  { intros h,
    cases h,
    by_cases h1: x < 5,
    { linarith, },
    by_cases h2: x > -2,
    { linarith, },
    linarith, },
end

end problem1_problem2_l401_401406


namespace waldetrade_wins_on_regular_2014gon_l401_401618

theorem waldetrade_wins_on_regular_2014gon :
  (∃ (p : ℕ), p = 2014 ∧
  ∀ (game : Type), game = {gon : RegularPolygon p, players : list Player, 
  move : Move → Game → Game, valid_move : Move → Game → Prop},
  ∀ (s : Strategy), s = {start_diagonal : ∀ g, valid_move (start_diagonal g) g,
  mirror_strategy : ∀ g m, valid_move m g → valid_move (mirror_strategy m g) g},
  ∃ (p1 p2 : Player), 
  (players game) = [p1, p2] ∧ 
  (∀ g, p1 & p2 take_turns g → 
  (p1 wins g ↔ Waldetrade))) :=
sorry

end waldetrade_wins_on_regular_2014gon_l401_401618


namespace find_d_2017_O_A_l401_401478

def S := {p : ℤ × ℤ | 0 ≤ p.fst ∧ p.fst ≤ 2016 ∧ 0 ≤ p.snd ∧ p.snd ≤ 2016}

def d_2017 (A B : ℤ × ℤ) : ℤ :=
  ((A.fst - B.fst) ^ 2 + (A.snd - B.snd) ^ 2) % 2017

theorem find_d_2017_O_A :
  let A := (5, 5) in
  let B := (2, 6) in
  let C := (7, 11) in
  ∃ O ∈ S, d_2017 O A = 1021 ∧ d_2017 O A = d_2017 O B ∧ d_2017 O A = d_2017 O C :=
  sorry

end find_d_2017_O_A_l401_401478


namespace polynomial_g_l401_401548

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l401_401548


namespace solve_for_x_and_y_l401_401933

def total_selected : ℕ := 50
def freq_group_0_50 : ℕ := 2
def freq_group_50_100 : ℕ := 4
def freq_group_200_250 : ℕ := 8
def freq_group_250_300 : ℕ := 14
def freq_group_300_350 : ℕ := 4

variables (x y : ℕ)
hypothesis h1 : y = 2 * x

def freq_sum : ℕ := freq_group_0_50 + freq_group_50_100 + x + y + freq_group_200_250 + freq_group_250_300 + freq_group_300_350

theorem solve_for_x_and_y (h : total_selected = freq_sum) : x = 6 ∧ y = 12 :=
by
  have h_eq : 50 = 2 + 4 + x + y + 8 + 14 + 4 := sorry
  have h_subst_y : y = 2 * x := sorry
  sorry

end solve_for_x_and_y_l401_401933


namespace soap_box_missing_dimension_l401_401235

theorem soap_box_missing_dimension
  (x : ℕ) -- The missing dimension of the soap box
  (Volume_carton : ℕ := 25 * 48 * 60)
  (Volume_soap_box : ℕ := 8 * x * 5)
  (Max_soap_boxes : ℕ := 300)
  (condition : Max_soap_boxes * Volume_soap_box ≤ Volume_carton) :
  x ≤ 6 := by
sorry

end soap_box_missing_dimension_l401_401235


namespace range_of_m_l401_401021

theorem range_of_m (m : ℝ) :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + m ≤ 0) → 1 < m := by
  sorry

end range_of_m_l401_401021


namespace solve_for_a_l401_401885

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-a * x)

noncomputable def f' (x : ℝ) : ℝ := (f a x + Real.exp (-a * x)).deriv

theorem solve_for_a : (∀ x : ℝ, x * f' a x = x * f' a (-x)) → a = 1 :=
by
  sorry

end solve_for_a_l401_401885


namespace proof_equivalence_l401_401392

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def point_on_circle (x y r : ℝ) : Prop :=
  (x - 5)^2 + y^2 = r^2

def line_tangent_to_circle (x₀ y₀ r : ℝ) : Prop :=
  (x₀ - 5)^2 + y₀^2 = r^2 

def midpoint_property (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

noncomputable def four_lines_condition (r : ℝ) : Prop :=
  (2 < r ∧ r < 4) ∨ r = √7

theorem proof_equivalence :
  ∀ (A B M : ℝ × ℝ) (r : ℝ),
    (∃ (l : ℝ → ℝ), ∀ (x y : ℝ),
        (point_on_parabola x y ∧ ∃ (x₁ y₁ x₂ y₂ : ℝ), point_on_parabola x₁ y₁ ∧ point_on_parabola x₂ y₂ ∧
            (A = (x₁, y₁) ∧ B = (x₂, y₂)) ∧ M = ((x₁ + x₂) / 2, (y₁ + y₂) / 2 ) ∧
            line_tangent_to_circle (fst M) (snd M) r ∧ four_lines_condition r)) →
    (∀ M, fst M = 3) ∧ four_lines_condition r :=
by
  intros
  sorry

end proof_equivalence_l401_401392


namespace Vizing_l401_401966

-- Define a graph structure
structure Graph :=
  (V : Type)
  (E : Type)
  (incidence : E → V × V)
  (adjacent : V → V → Prop := λ x y, ∃ e : E, incidence e = (x, y) ∨ incidence e = (y, x))

-- Assuming definitions for Δ(G) - Maximum degree and χ'(G) - Chromatic index
def maxDegree (G : Graph) : ℕ := sorry
def chromaticIndex (G : Graph) : ℕ := sorry

theorem Vizing (G : Graph) : maxDegree G ≤ chromaticIndex G ∧ chromaticIndex G ≤ maxDegree G + 1 := 
  sorry

end Vizing_l401_401966


namespace range_of_m_l401_401779

theorem range_of_m (x m : ℝ) (h1 : (m - 1) / (x + 1) = 1) (h2 : x < 0) : m < 2 ∧ m ≠ 1 :=
by
  sorry

end range_of_m_l401_401779


namespace positive_real_solutions_count_l401_401283

noncomputable def f (x : ℝ) : ℝ := x^4 + 6 * x^3 + 12 * x^2 + 2027 * x - 1586

theorem positive_real_solutions_count : 
    (∀ x > 0, ((x^4 + 6 * x^3 + 12 * x^2 + 2027 * x - 1586) = 0) ↔ f x = 0) 
    → (∃! x > 0, x^8 + 6 * x^7 + 12 * x^6 + 2027 * x^5 - 1586 * x^4 = 0) :=
by {
  sorry,
}

end positive_real_solutions_count_l401_401283


namespace sum_num_den_cos_alpha_of_parallell_chords_l401_401066

-- This is to state that \(\cos \alpha\) is rational and in lowest terms
def is_rational_in_lowest_terms (x : ℝ) : Prop :=
  ∃ (a b : ℤ), int.gcd a b = 1 ∧ (x = a / b)

theorem sum_num_den_cos_alpha_of_parallell_chords 
  (r α β : ℝ) (hαβ : α + β < Real.pi) 
  (h1 : 2 * r * Real.sin (α / 2) = 2)
  (h2 : 2 * r * Real.sin (β / 2) = 3)
  (h3 : 2 * r * Real.sin ((α + β) / 2) = 4)
  (hcos : is_rational_in_lowest_terms (Real.cos α) ∧ 0 < Real.cos α) :
  (let ⟨a, b, hab⟩ := hcos in a + b = 49) :=
sorry

end sum_num_den_cos_alpha_of_parallell_chords_l401_401066


namespace batches_of_muffins_l401_401263

-- Definitions of the costs and savings
def cost_blueberries_6oz : ℝ := 5
def cost_raspberries_12oz : ℝ := 3
def ounces_per_batch : ℝ := 12
def total_savings : ℝ := 22

-- The proof problem is to show the number of batches Bill plans to make
theorem batches_of_muffins : (total_savings / (2 * cost_blueberries_6oz - cost_raspberries_12oz)) = 3 := 
by 
  sorry  -- Proof goes here

end batches_of_muffins_l401_401263


namespace tank_height_l401_401932

theorem tank_height
  (r_A r_B h_A h_B : ℝ)
  (h₁ : 8 = 2 * Real.pi * r_A)
  (h₂ : h_B = 8)
  (h₃ : 10 = 2 * Real.pi * r_B)
  (h₄ : π * r_A ^ 2 * h_A = 0.56 * (π * r_B ^ 2 * h_B)) :
  h_A = 7 :=
sorry

end tank_height_l401_401932


namespace max_rounds_passed_probability_passing_first_three_rounds_l401_401843

open ProbabilityTheory

-- Definition for fair six-sided die.
def is_fair_dice (s : Set ℕ) := s = {1, 2, 3, 4, 5, 6}

-- Definition of passing condition
def passes_round (n : ℕ) (sum_points : ℕ) := sum_points > 2^n

-- 1. Proving the maximum number of rounds a person can pass
theorem max_rounds_passed : 
  ∀ n, (passes_round n (6 * n)) ↔ n ≤ 4 := 
by sorry

-- Definitions for event A_n and the probability of passing
def A (n : ℕ) := { outcomes | sum outcomes ≤ 2^n }

-- 2. Proving the probability of passing the first three rounds in succession
theorem probability_passing_first_three_rounds (P : ℕ → ℝ) (A : Set (Set ℕ)) :
  (P 1 = 1 - 2 / 6) ∧ (P 2 = 1 - 6 / 36) ∧ (P 3 = 1 - 56 / 216) →
  (P 1 * P 2 * P 3 = 100 / 243) :=
by sorry

end max_rounds_passed_probability_passing_first_three_rounds_l401_401843


namespace combined_surface_area_l401_401244

noncomputable def radius_cylinder : ℝ := 3
noncomputable def height_cylinder : ℝ := 5
noncomputable def radius_cone : ℝ := 3
noncomputable def height_cone : ℝ := 3

noncomputable def lateral_area_cylinder : ℝ := 2 * Real.pi * radius_cylinder * height_cylinder
noncomputable def slant_height_cone : ℝ := Real.sqrt (radius_cone^2 + height_cone^2)
noncomputable def lateral_area_cone : ℝ := Real.pi * radius_cone * slant_height_cone

theorem combined_surface_area : 
  lateral_area_cylinder + lateral_area_cone = 30 * Real.pi + 9 * Real.sqrt(2) * Real.pi :=
by
  sorry

end combined_surface_area_l401_401244


namespace proof_geometric_sequence_first_term_and_number_of_terms_l401_401850

noncomputable def geometric_sequence_first_term_and_number_of_terms 
  (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (a_5 : ℕ) (q : ℕ) (S_n_to_solve : ℕ) : Prop :=
  a_5 = 162 ∧ q = 3 ∧ S_n_to_solve = 242 → a_n 1 = 2 ∧ S_n_to_solve = 5

-- Provide proof for the above statement if required.
theorem proof_geometric_sequence_first_term_and_number_of_terms : 
  geometric_sequence_first_term_and_number_of_terms (λ n, 2 * 3^(n - 1)) (λ n, (2 * 3^n - 2) / (3 - 1)) 162 3 242 :=
by 
  sorry

end proof_geometric_sequence_first_term_and_number_of_terms_l401_401850


namespace pure_imaginary_a_eq_neg2_l401_401487

theorem pure_imaginary_a_eq_neg2 (a : ℝ) (z : ℂ) (h : z = a + 4*complex.I)
  (h1 : (2 - complex.I) * z = (8 - a) * complex.I) : a = -2 :=
sorry

end pure_imaginary_a_eq_neg2_l401_401487


namespace roots_modulus_1_l401_401895

theorem roots_modulus_1 (a b c : ℂ) (h1: ∀ x : ℂ, (x^3 + a * x^2 + b * x + c = 0) → (|x| = 1)) :
  ∀ x : ℂ, (x^3 + |a| * x^2 + |b| * x + |c| = 0) → (|x| = 1) :=
by
  sorry

end roots_modulus_1_l401_401895


namespace distance_travelled_by_gavril_l401_401361

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end distance_travelled_by_gavril_l401_401361


namespace decreasing_interval_l401_401953

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem decreasing_interval : ∀ x ∈ Ioo (-1 : ℝ) 1, f' x < 0 :=
by sorry

end decreasing_interval_l401_401953


namespace xyz_expression_l401_401913

variables (a b c x y z : ℝ)

def p (a b c : ℝ) : ℝ := (a + b + c) / 2

theorem xyz_expression
  (h1 : x^2 + x * y + y^2 = a^2)
  (h2 : y^2 + y * z + z^2 = b^2)
  (h3 : x^2 + x * z + z^2 = c^2) :
  x * y + y * z + x * z = 4 * sqrt ((p a b c) * ((p a b c) - a) * ((p a b c) - b) * ((p a b c) - c) / 3) :=
by
  sorry

end xyz_expression_l401_401913


namespace intersection_M_N_l401_401790

noncomputable def M : set ℝ := { y | ∃ x : ℝ, y = 2^x }
noncomputable def N : set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_M_N :
  M ∩ N = { y : ℝ | 0 < y } :=
begin
  sorry
end

end intersection_M_N_l401_401790


namespace original_time_40_l401_401665

theorem original_time_40
  (S T : ℝ)
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = 0.8 * S * (T + 10)) :
  T = 40 :=
by
  sorry

end original_time_40_l401_401665


namespace problem_statement_l401_401809

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l401_401809


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401513

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401513


namespace determine_g_l401_401538

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l401_401538


namespace units_digit_of_sum_of_squares_l401_401202

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_of_squares :
  let odds := (1 + 2*⟦0, 2052⟧) -- the sequence of the first 2053 odd positive integers
  let squares := odds.map (λ x, x * x)
  let units_digits := squares.map units_digit
  let total_units_digit := (units_digits.sum % 10)
  total_units_digit = 5 := 
begin
  /- Proof content would go here, skipped for now -/
  sorry
end

end units_digit_of_sum_of_squares_l401_401202


namespace game_c_higher_prob_than_game_d_l401_401649

noncomputable def prob_heads : ℚ := 2 / 3
noncomputable def prob_tails : ℚ := 1 / 3

def game_c_winning_prob : ℚ :=
  let prob_first_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_last_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_overlap := prob_heads ^ 5 + prob_tails ^ 5
  prob_first_three + prob_last_three - prob_overlap

def game_d_winning_prob : ℚ :=
  let prob_first_last_two := (prob_heads ^ 2 + prob_tails ^ 2) ^ 2
  let prob_middle_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_overlap_d := 2 * (prob_heads ^ 4 + prob_tails ^ 4)
  prob_first_last_two + prob_middle_three - prob_overlap_d

theorem game_c_higher_prob_than_game_d :
  game_c_winning_prob - game_d_winning_prob = 29 / 81 := 
sorry

end game_c_higher_prob_than_game_d_l401_401649


namespace josh_pays_six_dollars_l401_401867

variables (packs : ℕ) (pieces_per_pack : ℕ) (cost_per_piece : ℕ) (cents_per_dollar : ℕ)

def total_dollars (packs pieces_per_pack cost_per_piece cents_per_dollar : ℕ) : ℕ :=
  (packs * pieces_per_pack * cost_per_piece) / cents_per_dollar

theorem josh_pays_six_dollars 
  (h1 : packs = 3)
  (h2 : pieces_per_pack = 20)
  (h3 : cost_per_piece = 10)
  (h4 : cents_per_dollar = 100) :
  total_dollars packs pieces_per_pack cost_per_piece cents_per_dollar = 6 :=
by
  unfold total_dollars
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end josh_pays_six_dollars_l401_401867


namespace time_duration_away_l401_401133

theorem time_duration_away (h0 m0 h1 m1 : ℕ) (n : ℕ) 
  (h0_ang : h0 * 30 + m0 * (1/2) = 180)
  (m0_ang : m0 * 6 = 0)
  (h1_ang : h1 * 30 + m1 * (1/2) = 180 + 0.5 * n)
  (m1_ang : m1 * 6 = 6 * n)
  (ang_condition_0 : |(180 + 0.5 * n) - 6 * n| = 80)
  (ang_condition_1 : |180 - 5.5 * n| = 80) : 
  n = 29 :=
by
  sorry

end time_duration_away_l401_401133


namespace polynomial_evaluation_l401_401278

-- Define operations using Lean syntax
def star (a b : ℚ) := a + b
def otimes (a b : ℚ) := a - b

-- Define a function to represent the polynomial expression
def expression (a b : ℚ) := star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b)

theorem polynomial_evaluation (a b : ℚ) (ha : a = 5) (hb : b = 3) : expression a b = 435 := by
  sorry

end polynomial_evaluation_l401_401278


namespace find_BC_l401_401435

-- Definitions of the conditions
def angle_A : ℝ := Real.pi / 3  -- π/3 radians == 60 degrees
def sum_of_sides (AB AC : ℝ) : Prop := AB + AC = 10
def area_S (BC : ℝ) (AB AC : ℝ) : Prop := (1 / 2) * AB * AC * Real.sin angle_A = 4 * Real.sqrt 3 

-- The main theorem
theorem find_BC (AB AC BC : ℝ) (h1 : sum_of_sides AB AC) (h2 : area_S BC AB AC) : BC = 2 * Real.sqrt 13 :=
by
  sorry

end find_BC_l401_401435


namespace final_price_after_changes_l401_401587

theorem final_price_after_changes (initial_price : ℝ) (increase1 decrease2 increase3 : ℝ) 
  (h1 : initial_price = 320) (h2 : increase1 = 0.15) (h3 : decrease2 = 0.10) (h4 : increase3 = 0.25) :
  let month1_price := initial_price * (1 + increase1),
      month2_price := month1_price * (1 - decrease2),
      month3_price := month2_price * (1 + increase3)
  in month3_price = 414 := by {
  -- Skipping proof details with "sorry"
  sorry
}

end final_price_after_changes_l401_401587


namespace problem1_solution_problem2_solution_l401_401537

-- Statement for Problem 1
theorem problem1_solution (x : ℝ) : (1 / 2 * (x - 3) ^ 2 = 18) ↔ (x = 9 ∨ x = -3) :=
by sorry

-- Statement for Problem 2
theorem problem2_solution (x : ℝ) : (x ^ 2 + 6 * x = 5) ↔ (x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end problem1_solution_problem2_solution_l401_401537


namespace quadratic_roots_satisfy_condition_l401_401768
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l401_401768


namespace sum_of_solutions_eq_one_l401_401593

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l401_401593


namespace decreasing_interval_l401_401576

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + 3

-- Define the interval
def interval := set.Ici (-1 : ℝ)

-- Define the condition for a function to be decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

-- State the main theorem
theorem decreasing_interval (m : ℝ) :
  (is_decreasing_on (f m) interval) ↔ (-1 ≤ m ∧ m ≤ 0) :=
by 
  sorry

end decreasing_interval_l401_401576


namespace problem_statement_l401_401486

noncomputable def p : Prop := ∀ x, cos (2 * x) = cos (2 * (x + π / 2))
noncomputable def q : Prop := ∀ x, sin (x + π / 3) = sin (2 * π / 6 - x - π / 3)

theorem problem_statement : ¬ q = false :=
by 
  have h1 : ¬(¬(∀ x, sin (x + π / 3) = sin (2 * π / 6 - x - π / 3))),
  { sorry },
  exact h1

end problem_statement_l401_401486


namespace sum_floor_log_base_2_l401_401358

/-- Define the greatest integer function floored
    Given a real number x, ⌊x⌋ is the greatest integer less than or equal to x -/
def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Define the logarithm base 2 function
    Given a real number x, log₂ x is the logarithm of x with base 2 -/
def log_base_2 (x : ℝ) : ℝ := real.log x / real.log 2

/-- Define the summation for the floor of log base 2 from 1 to 32
    Prove the sum is equal to 103 -/
theorem sum_floor_log_base_2 : (∑ k in finset.range 32, floor (log_base_2 (k + 1))) = 103 :=
sorry

end sum_floor_log_base_2_l401_401358


namespace proposition_ab_l401_401521

-- Proposition A: If plane α is in plane β, and plane β is in plane γ, then plane α is parallel to plane γ is false.
def proposition_a (α β γ : Plane) : Prop :=
  (α ∈ β ∧ β ∈ γ) → ¬ (α ∥ γ)

-- Proposition B: If three non-collinear points on plane α are equidistant from plane β, then α is parallel to β is false.
def proposition_b (α β : Plane) [ThreeNonCollinearPoints α] : Prop :=
  (∀ p q r : Point, p ∈ α → q ∈ α → r ∈ α → ¬Collinear p q r → EquidistantFromPlane β {p, q, r}) → ¬ (α ∥ β)

-- Combine both propositions to form the theorem
theorem proposition_ab (α β γ : Plane) [ThreeNonCollinearPoints α] : proposition_a α β γ ∧ proposition_b α β :=
  by
    sorry

end proposition_ab_l401_401521


namespace sec_neg_240_eq_neg_2_l401_401303

theorem sec_neg_240_eq_neg_2 :
  sec (-240 : Real) = -2 :=
by
  have h1 : sec (x : Real) := 1 / cos x
  have h2 : cos (x : Real) = cos (x + 360)
  have h3 : cos (120 : Real) = -1/2
  sorry

end sec_neg_240_eq_neg_2_l401_401303


namespace area_of_AMN_eq_18_l401_401189

-- Definitions and conditions
variables {A B C D M N : Type} [IsMidpoint A D M] [IsMidpoint B C N]

-- Base lengths conditions
variables {AB CD : ℝ} (h1 : AB = 2 * CD)

-- Area of the trapezoid
variable (trapezoid_area : ℝ := 72)

-- Define the area of triangle AMN
noncomputable def area_of_triangle_AMN : ℝ :=
  -- Assuming the result directly:
  ¼ * trapezoid_area

-- The theorem we want to prove
theorem area_of_AMN_eq_18 
  (h_midpoint_AD : IsMidpoint A D M)
  (h_midpoint_BC : IsMidpoint B C N)
  (h_AB_2CD : AB = 2 * CD)
  (h_trapezoid_area : trapezoid_area = 72) :
  area_of_triangle_AMN = 18 :=
by
  -- Leaving proof for later
  sorry

end area_of_AMN_eq_18_l401_401189


namespace system_solution_conditions_l401_401731

theorem system_solution_conditions (α1 α2 α3 α4 : ℝ) :
  (α1 = α4 ∨ α2 = α3) ↔ 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 + x2 = α1 * α2 ∧
    x1 + x3 = α1 * α3 ∧
    x1 + x4 = α1 * α4 ∧
    x2 + x3 = α2 * α3 ∧
    x2 + x4 = α2 * α4 ∧
    x3 + x4 = α3 * α4 ∧
    x1 = x2 ∧
    x2 = x3 ∧
    x1 = α2^2 / 2 ∧
    x3 = α2^2 / 2 ∧
    x4 = α2 * α4 - (α2^2 / 2) ) :=
by sorry

end system_solution_conditions_l401_401731


namespace day_after_exponential_days_l401_401974

noncomputable def days_since_monday (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get! (n % 7)

theorem day_after_exponential_days :
  days_since_monday (2^20) = "Friday" :=
by
  sorry

end day_after_exponential_days_l401_401974


namespace arithmetic_sequence_sum_l401_401451

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a1 : a 1 = 2
axiom a2_a3_sum : a 2 + a 3 = 13

-- The theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) : a (4) + a (5) + a (6) = 42 :=
sorry

end arithmetic_sequence_sum_l401_401451


namespace find_integer_cosine_l401_401325

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401325


namespace area_quadrilateral_XPWQ_l401_401529

-- Definitions based on the conditions
def rectangle (length width : ℝ) := true

def midpoint {A B : ℝ} (point : ℝ) : Prop := point = (A + B) / 2

def area_of_quadrilateral (quad : ℝ) := quad = 37.5

-- Condition instances
axiom XYZ_rect : rectangle 10 5
axiom P_midpoint_YZ : midpoint 2.5
axiom Q_midpoint_ZW : midpoint 2.5

-- The theorem to prove
theorem area_quadrilateral_XPWQ : 
  XYZ_rect ∧ P_midpoint_YZ ∧ Q_midpoint_ZW → area_of_quadrilateral 37.5 :=
begin
  sorry
end

end area_quadrilateral_XPWQ_l401_401529


namespace ellipse_solution_coordinates_G_l401_401760

-- Definitions for the ellipse C
def ellipse_eq (x y a b : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Specification for part (1)
theorem ellipse_solution :
  ∃ (a b : ℝ), ellipse_eq x y a b ∧ a = 4 ∧ b = 2 * Real.sqrt 3 :=
sorry

-- Definitions for additional conditions in part (2)
def on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_D (x1 : ℝ) (m n : ℝ) : Prop :=
  x1 = -2 * Real.sqrt 3 * m / (n - 2 * Real.sqrt 3)

def point_E (x2 : ℝ) (m n : ℝ) : Prop :=
  x2 = 2 * Real.sqrt 3 * m / (n - 2 * Real.sqrt 3)

def points_collinear (A B D : ℝ × ℝ) : Prop :=
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xD, yD) := D in
  yA * xD = yD * xA + yB * xA - yB * xD

-- Specification for part (2)
theorem coordinates_G (m n x1 x2 t: ℝ) (h_ellipse: on_ellipse m n 4 (2 * Real.sqrt 3)) 
  (h_D: point_D x1 m n) (h_E: point_E x2 m n) (h_angle : angle_eq (0, t) (x1, 0) (0, t) (x2, 0)) :
  t = 4 ∨ t = -4 :=
sorry

end ellipse_solution_coordinates_G_l401_401760


namespace bring_to_coincide_l401_401265

-- Define the types for figures and transformations.
structure SymmetricalFigure (α : Type) := 
  (shape : α)
  (symmetry_axis : α → α)
  (is_symmetric : ∀ p : α, symmetry_axis (symmetry_axis p) = p)

def coincide_by_half_turn {α : Type} (fig1 fig2 : SymmetricalFigure α) : Prop :=
  ∃ axis : α → α, 
  ∀ p : α, axis (axis p) = p ∧ (∀ q : α, (fig1.symmetry_axis q) = (fig2.symmetry_axis (axis q)))

-- The statement of the problem
theorem bring_to_coincide {α : Type} (fig1 fig2 : SymmetricalFigure α) 
  (symm_pos : ∀ p : α, fig1.symmetry_axis p = axis (fig2.symmetry_axis p)) :
  coincide_by_half_turn fig1 fig2 :=
by
  sorry

end bring_to_coincide_l401_401265


namespace sufficient_not_necessary_l401_401111

variable (x : ℝ)

theorem sufficient_not_necessary (h : |x| > 0) : (x > 0 ↔ true) :=
by 
  sorry

end sufficient_not_necessary_l401_401111


namespace productivity_after_repair_l401_401135

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end productivity_after_repair_l401_401135


namespace DEF_is_isosceles_right_triangle_l401_401113

noncomputable def midpoint (b c : ℂ) : ℂ := (b + c) / 2

theorem DEF_is_isosceles_right_triangle (A B C D E F : ℂ)
  (hA : A = 0)
  (hF : F = midpoint B C)
  (hD : D = (1 + complex.I) / 2 * B)
  (hE : E = (1 - complex.I) / 2 * C) :
  ∃ θ : ℂ, ∀ z : ℂ, D - F = θ * (E - F) ∧ abs θ = 1 ∧ θ ^ 4 = 1 := by
  sorry

end DEF_is_isosceles_right_triangle_l401_401113


namespace ship_placement_feasibility_l401_401905

-- Define the types and constants
constant grid_size : ℕ := 10

-- Define the ships as tuples (length, quantity)
inductive Ship : Type
| one_one
| one_two
| one_three
| one_four

open Ship

/-- Prove that:
    (a) If you place the ships in descending order, the process can always be completed.
    (b) If you place the ships in ascending order, there can be situations where it is impossible to place the next ship.
-/
theorem ship_placement_feasibility :
  (∀ (grid : ℕ × ℕ), grid = (grid_size, grid_size) → 
    (∀ (ships : list Ship),
      ships = [one_four] ++ replicate 2 one_three ++ replicate 3 one_two ++ replicate 4 one_one →
        (∃ placements : list (ℕ × ℕ), (∀ ship ∈ ships, ∃ placement ∈ placements, valid_placement grid ship placement))) ∧
  (∃ (ships : list Ship),
    ships = replicate 4 one_one ++ replicate 3 one_two ++ replicate 2 one_three ++ [one_four] →
      (∀ placements : list (ℕ × ℕ), ¬ (∀ ship ∈ ships, ∃ placement ∈ placements, valid_placement grid ship placement))) :=
begin
  sorry -- Proof not required
end

-- Define the valid placement function (to be implemented)
-- Checks if a ship can be placed at a given position
constant valid_placement : (ℕ × ℕ) → Ship → (ℕ × ℕ) → Prop

end ship_placement_feasibility_l401_401905


namespace product_of_possible_values_l401_401813

noncomputable def math_problem (x : ℚ) : Prop :=
  |(10 / x) - 4| = 3

theorem product_of_possible_values :
  let x1 := 10 / 7
  let x2 := 10
  (x1 * x2) = (100 / 7) :=
by
  sorry

end product_of_possible_values_l401_401813


namespace intersection_complement_M_and_N_l401_401469
open Set

def U := @univ ℝ
def M := {x : ℝ | x^2 + 2*x - 8 ≤ 0}
def N := {x : ℝ | -1 < x ∧ x < 3}
def complement_M := {x : ℝ | ¬ (x ∈ M)}

theorem intersection_complement_M_and_N :
  (complement_M ∩ N) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_complement_M_and_N_l401_401469


namespace find_n_cosine_l401_401311

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401311


namespace asymptote_intersections_l401_401350

theorem asymptote_intersections :
  let f := λ x : ℝ, (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)
  let vertical_asymptotes := {x : ℝ | x = 3 ∨ x = 6}
  let horizontal_asymptote := 1
  ∀ x ∈ vertical_asymptotes, (x, horizontal_asymptote) ∈ {(3, 1), (6, 1)} :=
by
  sorry

end asymptote_intersections_l401_401350


namespace positive_difference_areas_l401_401569

-- The geometrical definitions
variables {P : Type*} [EuclideanGeometry P]
variables {BSCT : convex_quadrilateral P} (M A : P)

-- Given conditions in the problem
variables (h1 : M ∈ midpoint(S, T))
variables (h2 : collinear(B, T, A))
variables (h3 : collinear(S, C, A))
variables (h4 : length(AB) = 91)
variables (h5 : length(BC) = 98)
variables (h6 : length(CA) = 105)
variables (h7 : perpendicular(AM, BC))

-- redefined problem as Lean theorem statement
theorem positive_difference_areas (h1 : M ∈ midpoint(S, T)) (h2 : collinear(B, T, A))
  (h3 : collinear(S, C, A)) (h4 : length (AB) = 91) (h5 : length (BC) = 98)
  (h6 : length (CA) = 105) (h7 : perpendicular (AM, BC)):
  abs(area (triangle S M C) - area (triangle B M T)) = 336 :=
  sorry

end positive_difference_areas_l401_401569


namespace at_least_three_mismatched_cells_l401_401793

noncomputable section

structure Grid :=
  (cells: fin 1982 → fin 1983 → bool) -- true stands for blue, false stands for red
  (even_rows : ∀ i : fin 1982, even (finset.univ.filter (λ j, cells i j = true)).card)
  (even_columns : ∀ j : fin 1983, even (finset.univ.filter (λ i, cells i j = true)).card)

def mismatched_cells (G1 G2 : Grid) : set (fin 1982 × fin 1983) :=
  { p | G1.cells p.1 p.2 ≠ G2.cells p.1 p.2 }

theorem at_least_three_mismatched_cells (G1 G2 : Grid)
  (h : ∃ p : fin 1982 × fin 1983, G1.cells p.1 p.2 ≠ G2.cells p.1 p.2) :
  ∃ p1 p2 p3 : fin 1982 × fin 1983,
    p1 ∈ mismatched_cells G1 G2 ∧ 
    p2 ∈ mismatched_cells G1 G2 ∧ 
    p3 ∈ mismatched_cells G1 G2 ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end at_least_three_mismatched_cells_l401_401793


namespace improved_productivity_l401_401137

-- Let the initial productivity be a constant
def initial_productivity : ℕ := 10

-- Let the increase factor be a constant, represented as a rational number
def increase_factor : ℚ := 3 / 2

-- The goal is to prove that the current productivity equals 25 trees daily
theorem improved_productivity : initial_productivity + (initial_productivity * increase_factor).toNat = 25 := 
by
  sorry

end improved_productivity_l401_401137


namespace g_50_unique_l401_401479

namespace Proof

-- Define the function g and the condition it should satisfy
variable (g : ℕ → ℕ)
variable (h : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b))

theorem g_50_unique : ∃ (m t : ℕ), m * t = 0 := by
  -- Existence of m and t fulfilling the condition
  -- Placeholder for the proof
  sorry

end Proof

end g_50_unique_l401_401479


namespace jacob_hours_l401_401519

theorem jacob_hours (J : ℕ) (H1 : ∃ (G P : ℕ),
    G = J - 6 ∧
    P = 2 * G - 4 ∧
    J + G + P = 50) : J = 18 :=
by
  sorry

end jacob_hours_l401_401519


namespace orchids_cut_l401_401614

-- defining the initial conditions
def initial_orchids : ℕ := 3
def final_orchids : ℕ := 7

-- the question: prove the number of orchids cut
theorem orchids_cut : final_orchids - initial_orchids = 4 := by
  sorry

end orchids_cut_l401_401614


namespace distance_between_polar_points_l401_401081

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points_l401_401081


namespace quadratic_min_value_l401_401408

theorem quadratic_min_value (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = a * x ^ 2 + b * x + 1)
  (h2 : f' x = 2 * a * x + b)
  (h3 : f' 0 > 0)
  (h4 : b^2 - 4 * a = 0) :
  (min_value : ℝ) := 2 := 
sorry

end quadratic_min_value_l401_401408


namespace integer_n_satisfies_condition_l401_401198

theorem integer_n_satisfies_condition :
  ∃ n : ℤ, 0 ≤ n ∧ n < 251 ∧ (250 * n ≡ 123 [MOD 251]) ∧ (n ≡ 128 [MOD 251]) :=
by
  sorry

end integer_n_satisfies_condition_l401_401198


namespace min_value_period_triangle_sides_l401_401013

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt 3) * sin x * cos x - (cos x) ^ 2 - 1 / 2

theorem min_value_period :
  (∀ x : ℝ, f(x) ≥ -2) ∧ (∃ x : ℝ, f(x) = -2) ∧ (∀ x : ℝ, f(x + π) = f(x)) :=
sorry

theorem triangle_sides (A B C a b c : ℝ)
  (h1 : sin B - 2 * sin A = 0)
  (h2 : c = 3)
  (h3 : f C = 0)
  (h4 : C = π / 3) :
  a = sqrt 3 ∧ b = 2 * sqrt 3 :=
sorry

end min_value_period_triangle_sides_l401_401013


namespace prove_A_annual_savings_l401_401068

noncomputable def employee_A_annual_savings
  (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02) : ℝ :=
  let total_deductions := tax_rate + pension_rate + healthcare_rate
  let Income_after_deductions := A_income * (1 - total_deductions)
  let annual_savings := 12 * Income_after_deductions
  annual_savings

theorem prove_A_annual_savings : 
  ∀ (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02),
  employee_A_annual_savings A_income B_income C_income D_income C_income_val income_ratio tax_rate pension_rate healthcare_rate tax_rate_val pension_rate_val healthcare_rate_val = 232400.16 :=
by
  sorry

end prove_A_annual_savings_l401_401068


namespace concurrency_of_inscribed_squares_l401_401871

open Real -- Assuming we need basic real number operations and trigonometry

noncomputable def triangle (A B C A₁ B₁ C₁ : Point) : Prop := 
  ∃ (K L M N : Point), 
    is_square K L M N ∧ 
    K ∈ line BC ∧ N ∈ line BC ∧ 
    L ∈ line AB ∧ M ∈ line AC ∧ 
    midpoint A₁ K N ∧ 
    ∃ (K' L' M' N' : Point), 
      is_square K' L' M' N' ∧ 
      K' ∈ line AC ∧ N' ∈ line AC ∧ 
      L' ∈ line AB ∧ M' ∈ line BC ∧ 
      midpoint B₁ K' N' ∧ 
      ∃ (K'' L'' M'' N'' : Point), 
        is_square K'' L'' M'' N'' ∧ 
        K'' ∈ line AB ∧ N'' ∈ line AB ∧ 
        L'' ∈ line BC ∧ M'' ∈ line AC ∧ 
        midpoint C₁ K'' N''

theorem concurrency_of_inscribed_squares 
  {A B C A₂ B₂ C₂ : Point} (h : triangle A B C A₂ B₂ C₂) : 
  concurrent (line_through A A₂) (line_through B B₂) (line_through C C₂) :=
sorry

end concurrency_of_inscribed_squares_l401_401871


namespace distinctRemainders_if_and_only_if_even_l401_401639

noncomputable def distinctRemainders (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), 
    (∀ i, a i ≠ 0 ∧ a i < n) ∧ -- 1. Each card has a number from 1 to n-1.
    (∀ i j, i ≠ j → (a i + a j) % n ≠ 0) -- 2. The sums yield different remainders mod n.

theorem distinctRemainders_if_and_only_if_even (n : ℕ) (h : n ≥ 4) :
  distinctRemainders n ↔ n % 2 = 0 := sorry

end distinctRemainders_if_and_only_if_even_l401_401639


namespace angle_A_bounds_altitude_inradius_relation_circumcenter_distance_l401_401638

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
a = (b + c) / 2

theorem angle_A_bounds (a b c : ℝ) (h : triangle_sides a b c) (A : ℝ) :
  0 ≤ A ∧ A ≤ 60 :=
sorry

theorem altitude_inradius_relation (a b c : ℝ) (r h : ℝ) (h1 : triangle_sides a b c) :
  h = 3 * r :=
sorry

theorem circumcenter_distance (a b c R r : ℝ) (h1 : triangle_sides a b c) :
  true :=  -- no straightforward functional definition yet for center distances tucked abstractly

def solve_triangle_problem := 
  angle_A_bounds
  -- invoked next with problem said arguments and further necessary dependencies as needed
  altitude_inradius_relation
  -- invoked equivalently  
  circumcenter_distance
-- troubleshoot and invoke in succession potentially


end angle_A_bounds_altitude_inradius_relation_circumcenter_distance_l401_401638


namespace cos_theta_is_37_over_143_l401_401471

/-- Let θ be the angle between the planes given by the equations:
    Plane1: x - 3y + z - 4 = 0 
    Plane2: 4x - 12y - 3z + 2 = 0.
    We want to prove that cos θ = 37/143. -/
noncomputable def cos_theta_between_planes : ℝ := 
  let n1 := (1, -3, 1) in
  let n2 := (4, -12, -3) in
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3 in
  let magnitude_n1 := real.sqrt (n1.1 ^ 2 + n1.2 ^ 2 + n1.3 ^ 2) in
  let magnitude_n2 := real.sqrt (n2.1 ^ 2 + n2.2 ^ 2 + n2.3 ^ 2) in
  dot_product / (magnitude_n1 * magnitude_n2)

theorem cos_theta_is_37_over_143 :
  cos_theta_between_planes = 37 / 143 :=
sorry

end cos_theta_is_37_over_143_l401_401471


namespace value_of_x_squared_plus_reciprocal_squared_l401_401805

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l401_401805


namespace cost_of_fresh_water_per_day_l401_401438

-- Define the conditions
def cost_per_gallon := 1 -- $1 per gallon
def water_needed_per_person_per_day := 1 / 2 -- 1/2 gallon per person per day
def family_size := 6 -- 6 people in the family

-- Define the theorem to prove the cost for fresh water for a day for the family
theorem cost_of_fresh_water_per_day :
  (family_size * water_needed_per_person_per_day * cost_per_gallon) = 3 := 
by
  calc
    (family_size * water_needed_per_person_per_day * cost_per_gallon)
    = (6 * (1 / 2) * 1) : by rw [family_size, water_needed_per_person_per_day, cost_per_gallon]
    ... = 3 : by norm_num -- 6 * 1/2 * 1 = 3

end cost_of_fresh_water_per_day_l401_401438


namespace xy_value_l401_401920

theorem xy_value (x y : ℝ) (h1 : 2^x = 64^(y + 1)) (h2 : 81^y = 3^(x - 5)) : x * y = -3/2 := 
by
  sorry

end xy_value_l401_401920


namespace cos_alpha2_add_beta_eq_sqrt2_div_2_l401_401053

theorem cos_alpha2_add_beta_eq_sqrt2_div_2
  (α : ℝ) (β : ℝ) (λ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π)
  (hβ : -π / 4 ≤ β ∧ β ≤ π / 4)
  (h1 : (α - π / 2)^3 - cos α - 2 * λ = 0)
  (h2 : 4 * β^3 + sin β * cos β + λ = 0):
  cos (α / 2 + β) = (Real.sqrt 2) / 2 := by
  sorry

end cos_alpha2_add_beta_eq_sqrt2_div_2_l401_401053


namespace no_integer_solutions_l401_401416

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^4 + y^2 = 6 * y - 3 :=
by
  sorry

end no_integer_solutions_l401_401416


namespace f1_even_f2_neither_f3_even_f4_neither_f5_odd_l401_401287

-- Define the functions
def f1 (x : ℝ) : ℝ := if x ≠ 0 then sin x / x else 0
def f2 (x : ℝ) : ℝ := x + cos x
def f3 (x : ℝ) : ℝ := x^2 + cos x
def f4 (x : ℝ) : Option ℝ := if x > 0 then some (log x) else none
def f5 (x : ℝ) : ℝ := x + sin x

-- Define the properties of being even or odd
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

-- Prove the properties for each function
theorem f1_even : is_even f1 := sorry
theorem f2_neither : ¬(is_even f2) ∧ ¬(is_odd f2) := sorry
theorem f3_even : is_even f3 := sorry
theorem f4_neither : ¬(is_even f4) ∧ ¬(is_odd f4) := sorry
theorem f5_odd : is_odd f5 := sorry

end f1_even_f2_neither_f3_even_f4_neither_f5_odd_l401_401287


namespace problem_statement_l401_401483

variables (A B C H P Q R : Point)
variables [triangle ABC] [orthocenter H A B C]
variables [on_circumcircle P A B C]
variables [line_parallel (line BP) (line A Q)]
variables [line_parallel (line CP) (line A R)]
variables [line_intersect Q (line CH)]
variables [line_intersect R (line BH)]

theorem problem_statement : parallel (line Q R) (line A P) :=
sorry

end problem_statement_l401_401483


namespace shortest_distance_to_circle_l401_401199

theorem shortest_distance_to_circle :
  let C : Circle := (x - 12)^2 + (y + 5)^2 = 3^2 in
  shortest_distance (0, 0) C = 10 :=
by
  sorry

end shortest_distance_to_circle_l401_401199


namespace m_range_when_proposition_false_l401_401827

theorem m_range_when_proposition_false :
  (∀ x : ℝ, ¬ (∃ x_0 : ℝ, m * x_0^2 - (m + 3) * x_0 + m ≤ 0)) ↔ (3 < m) :=
begin
  sorry,
end

end m_range_when_proposition_false_l401_401827


namespace first_grab_ceremony_outcomes_l401_401720

theorem first_grab_ceremony_outcomes :
  ∀ (educational_items living_items entertainment_items : ℕ),
  educational_items = 4 → living_items = 3 → entertainment_items = 4 →
  educational_items + living_items + entertainment_items = 11 :=
by
  intros educational_items living_items entertainment_items h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end first_grab_ceremony_outcomes_l401_401720


namespace minimum_possible_value_of_BC_l401_401400

def triangle_ABC_side_lengths_are_integers (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_is_twice_angle_B (A B C : ℝ) : Prop :=
  A = 2 * B

def CA_is_nine (CA : ℕ) : Prop :=
  CA = 9

theorem minimum_possible_value_of_BC
  (a b c : ℕ) (A B C : ℝ) (CA : ℕ)
  (h1 : triangle_ABC_side_lengths_are_integers a b c)
  (h2 : angle_A_is_twice_angle_B A B C)
  (h3 : CA_is_nine CA) :
  ∃ (BC : ℕ), BC = 12 := 
sorry

end minimum_possible_value_of_BC_l401_401400


namespace part1_part2_l401_401144

/-- Part (1): Number of ways to put 7 identical balls into 4 different boxes without any empty boxes. -/
theorem part1 (balls : ℕ) (boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 4) (h_non_empty : 0 < boxes ∧ boxes ≤ balls) :
    (∑ i in finset.Ico 1 (boxes+1), i) = 20 :=
by
    sorry

/-- Part (2): Number of ways to put 7 identical balls into 4 different boxes with the possibility of having empty boxes. -/
theorem part2 (balls : ℕ) (boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 4) :
    (∑ i in finset.range (boxes+1), i) = 120 :=
by
    sorry

end part1_part2_l401_401144


namespace math_problem_l401_401771

open Real -- Open the real number namespace

theorem math_problem (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end math_problem_l401_401771


namespace quadratic_form_m_neg3_l401_401058

theorem quadratic_form_m_neg3
  (m : ℝ)
  (h_exp : m^2 - 7 = 2)
  (h_coef : m ≠ 3) :
  m = -3 := by
  sorry

end quadratic_form_m_neg3_l401_401058


namespace sum_series_l401_401704

theorem sum_series : ∑ i in Finset.range 51, (-1)^((i + 1) : ℤ) * (i + 1) = 51 := by sorry

end sum_series_l401_401704


namespace properties_f_l401_401796

-- Given vectors
def a (x : ℝ) : ℝ × ℝ := (2 * sin x, cos x ^ 2)
def b (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * cos x, 2)

-- Dot product function
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Theorem statement for the proof of required properties
theorem properties_f (x : ℝ) (k : ℤ) :
  (f (x + π) = f x) ∧ 
  (∀ x₁ x₂, x₁ ∈ set.Icc (k * π + π / 6) (2 * π / 3 + k * π) →
   x₂ ∈ set.Icc (k * π + π / 6) (2 * π / 3 + k * π) → x₁ ≤ x₂ → f x₁ ≥ f x₂) ∧
  (∀ x ∈ set.Icc 0 (π / 2), 0 ≤ f x ∧ f x ≤ 3) := sorry

end properties_f_l401_401796


namespace sum_of_first_17_terms_l401_401450

variable {a : ℕ → ℝ}

-- Condition of the problem: a₄ + a₁₄ = 1
def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 14 = 1

-- Sum of the first 17 terms of the arithmetic sequence
def sum_first_17_terms (a : ℕ → ℝ) : ℝ :=
  (17 / 2) * (a 1 + a 17)

-- The final statement we need to prove
theorem sum_of_first_17_terms (h : arithmetic_sequence_condition a) : 
  sum_first_17_terms a = 8.5 :=
by
  sorry

end sum_of_first_17_terms_l401_401450


namespace sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l401_401201

theorem sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100 : 
  (15^25 + 5^25) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_15_pow_25_plus_5_pow_25_mod_100_l401_401201


namespace convex_quadrilateral_transformation_l401_401523

theorem convex_quadrilateral_transformation (ABCD : ConvexQuadrilateral) 
  (h1 : ¬ is_trapezoid ABCD) (h2 : ∀ (s : Side ABCD), ¬ parallel s) :
  ∃ (f : AffineTransformation), 
  ∀ (angles : list Angle), 
    (angles_of_quadrilateral (f.transform_quadrilateral ABCD) angles) → 
    ∃ (i j : ℕ), i ≠ j ∧ angles.nth i = some 90 ∧ angles.nth j = some 90 :=
sorry

end convex_quadrilateral_transformation_l401_401523


namespace consecutive_integers_specific_problem_l401_401431

theorem consecutive_integers (n : ℕ) (hn : n > 1) (hprime : Prime (2^n - 1)) : 
  ¬Prime (2^n + 1) ∧ (2^n + 1) % 2 = 1 ∧ ∃ d, d > 1 ∧ d ∣ (2^n + 1) :=
by
  sorry

-- Specific statement for the given problem
theorem specific_problem : ¬Prime (2^859433 + 1) ∧ (2^859433 + 1) % 2 = 1 ∧ ∃ d, d > 1 ∧ d ∣ (2^859433 + 1) :=
  consecutive_integers 859433 dec_trivial sorry

end consecutive_integers_specific_problem_l401_401431


namespace num_satisfying_integers_l401_401043

theorem num_satisfying_integers : {x : ℤ | (x^2 - 2 * x - 2)^(x + 3) = 1}.toFinset.card = 3 := 
sorry

end num_satisfying_integers_l401_401043


namespace part1_extreme_point_of_f_at_2_part2_range_of_m_l401_401367

def f (a : ℝ) (x : ℝ) := a * x - a / x - 5 * Real.log x
def g (m : ℝ) (x : ℝ) := x^2 - m * x + 4

theorem part1_extreme_point_of_f_at_2 (a : ℝ) (h : deriv (f a) 2 = 0) : a = 2 := sorry

theorem part2_range_of_m (m : ℝ)
  (h1 : ∃ x1 : ℝ, 0 < x1 ∧ x1 < 1 ∧ f 2 x1 ≥ g m 1)
  (h2 : ∀ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 → f 2 x1 ≥ g m x2) :
  ∃ m : ℝ, m ≥ 8 - 5 * Real.log 2 := sorry

end part1_extreme_point_of_f_at_2_part2_range_of_m_l401_401367


namespace base_329_digits_even_l401_401741

noncomputable def base_of_four_digit_even_final : ℕ := 5

theorem base_329_digits_even (b : ℕ) (h1 : b^3 ≤ 329) (h2 : 329 < b^4)
  (h3 : ∀ d, 329 % b = d → d % 2 = 0) : b = base_of_four_digit_even_final :=
by sorry

end base_329_digits_even_l401_401741


namespace polar_to_cartesian_l401_401757

noncomputable def cos_pi_over_3 : ℝ := real.cos (real.pi / 3)
noncomputable def sin_pi_over_3 : ℝ := real.sin (real.pi / 3)

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = real.pi / 3) :
    (r * real.cos θ, r * real.sin θ) = (1, real.sqrt 3) :=
by
  rw [h_r, h_θ, cos_pi_over_3, sin_pi_over_3]
  have h1 : real.cos (real.pi / 3) = 1 / 2 := by sorry
  have h2 : real.sin (real.pi / 3) = real.sqrt 3 / 2 := by sorry
  rw [h1, h2]
  simp
  split
  norm_num
  norm_num
  exact real.sq_sqrt (by norm_num)

end polar_to_cartesian_l401_401757


namespace vertex_of_parabola_l401_401942

theorem vertex_of_parabola : 
  ∀ (x y : ℝ), (y = -x^2 + 3) → (0, 3) ∈ {(h, k) | ∃ (a : ℝ), y = a * (x - h)^2 + k} :=
by
  sorry

end vertex_of_parabola_l401_401942


namespace find_n_cos_eq_l401_401345

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401345


namespace min_value_expression_l401_401884

theorem min_value_expression (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
    ∃ c : ℝ, c = a^2 + b^2 + 2/(a^2) + b/a + 1/(b^2) ∧ c = sqrt 7 :=
by
  sorry

end min_value_expression_l401_401884


namespace lines_parallel_if_perpendicular_to_same_plane_l401_401752

variables (m n : Line) (α : Plane)

-- Define conditions using Lean's logical constructs
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- This would define the condition
def parallel_lines (l1 l2 : Line) : Prop := sorry -- This would define the condition

-- The statement to prove
theorem lines_parallel_if_perpendicular_to_same_plane 
  (h1 : perpendicular_to_plane m α) 
  (h2 : perpendicular_to_plane n α) : 
  parallel_lines m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l401_401752


namespace dots_not_visible_l401_401742

theorem dots_not_visible (visible_dots : List ℕ) (hdice_faces : (∀ d, d ∈ [1, 2, 3, 4, 5, 6])) :
  (∑ i in visible_dots, i) = 26 → (4 * 21) - (∑ i in visible_dots, i) = 58 :=
by 
  sorry

end dots_not_visible_l401_401742


namespace music_stand_cost_proof_l401_401086

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l401_401086


namespace tangent_plane_normal_line_l401_401206

-- Definitions for conditions
def surface (x y z a b c : ℝ) : ℝ := x^2 / a^2 + y^2 / b^2 + z^2 / c^2 - 1

def point_M (a b c : ℝ) : ℝ × ℝ × ℝ := (a / Real.sqrt 2, b / 2, c / 2)

-- Theorems to prove
theorem tangent_plane (a b c : ℝ) : 
  ∀ (x y z : ℝ), 
    surface x y z a b c = 0 → 
    let M := point_M a b c in 
    (M.fst, M.snd.fst, M.snd.snd) = (x, y, z) → 
    (√2 * x) / a + y / b + z / c = 2 :=
sorry

theorem normal_line (a b c : ℝ) : 
  ∀ (x y z : ℝ),
    surface x y z a b c = 0 → 
    let M := point_M a b c in 
    (M.fst, M.snd.fst, M.snd.snd) = (x, y, z) → 
    a / √2 * (x - a / √2) = b * (y - b / 2) = c * (z - c / 2) :=
sorry

end tangent_plane_normal_line_l401_401206


namespace find_m_l401_401766

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l401_401766


namespace f_zeros_f_negative_values_f_minimum_value_f_maximum_value_l401_401403

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem f_zeros :
  (f 1 = 0) ∧ (f 3 = 0) :=
by -- proof steps go here
  sorry

theorem f_negative_values :
  ∀ (x : ℝ), (1 < x ∧ x < 3) → (f x < 0) :=
by -- proof steps go here
  sorry

theorem f_minimum_value :
  ∃ c ∈ set.Icc 0 3, (f c = -1) :=
by -- proof steps go here
  sorry

theorem f_maximum_value :
  (f 0 = 3) ∧ (f 3 = 3) :=
by -- proof steps go here
  sorry

end f_zeros_f_negative_values_f_minimum_value_f_maximum_value_l401_401403


namespace count_positive_solutions_of_eq_l401_401798

theorem count_positive_solutions_of_eq : 
  (∃ x : ℝ, x^2 = -6 * x + 9 ∧ x > 0) ∧ (¬ ∃ y : ℝ, y^2 = -6 * y + 9 ∧ y > 0 ∧ y ≠ -3 + 3 * Real.sqrt 2) :=
sorry

end count_positive_solutions_of_eq_l401_401798


namespace greatest_possible_gcd_value_l401_401279

noncomputable def sn (n : ℕ) := n ^ 2
noncomputable def expression (n : ℕ) := 2 * sn n + 10 * n
noncomputable def gcd_value (a b : ℕ) := Nat.gcd a b 

theorem greatest_possible_gcd_value :
  ∃ n : ℕ, gcd_value (expression n) (n - 3) = 42 :=
sorry

end greatest_possible_gcd_value_l401_401279


namespace program_output_l401_401988

theorem program_output :
  ∃ a b : ℕ, a = 10 ∧ b = a - 8 ∧ a = a - b ∧ a = 8 :=
by
  let a := 10
  let b := a - 8
  let a := a - b
  use a
  use b
  sorry

end program_output_l401_401988


namespace proof_problem_l401_401983

noncomputable def x_i (i : Nat) : ℕ := sorry
def n : ℕ := sorry
def a : ℕ := sorry
def b : ℕ := sorry

def x (i : Nat) : ℕ := if i ≤ n then x_i i else 0

def A : ℕ := ∑ i in range (n + 1), x i * a^i
def B : ℕ := ∑ i in range (n + 1), x i * b^i

def A' : ℕ := ∑ i in range n, x i * a^i
def B' : ℕ := ∑ i in range n, x i * b^i

theorem proof_problem :
  (∀ i, 0 ≤ x i ∧ x i < b) →
  x n > 0 →
  x (n-1) > 0 →
  a > b →
  A' * B < A * B' :=
by sorry

end proof_problem_l401_401983


namespace f_4_1981_eq_l401_401577

def f : ℕ → ℕ → ℕ
| 0, y     => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_4_1981_eq : f 4 1981 = 2 ^ 16 - 3 := sorry

end f_4_1981_eq_l401_401577


namespace andrea_still_needs_rhinestones_l401_401676

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l401_401676


namespace quadratic_segment_length_sum_eq_l401_401409

noncomputable def quadratic_segment_length_sum : ℚ := 
  ∑ n in Finset.range 100.filter (λ n => n > 0), 1 / (n * (n + 2))

theorem quadratic_segment_length_sum_eq :
  ∑ n in Finset.range 100.filter (λ n => n > 0), 1 / (n * (n + 2)) = 7625 / 10302 := 
  by
  sorry

end quadratic_segment_length_sum_eq_l401_401409


namespace fourth_student_guess_l401_401559

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  fourth_guess = 525 := 
by
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  show fourth_guess = 525 from sorry

end fourth_student_guess_l401_401559


namespace problem_statement_l401_401808

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l401_401808


namespace change_in_euros_l401_401096

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def dessert_cost : ℝ := 3
noncomputable def dessert_discount : ℝ := 0.5
noncomputable def side_dish_cost : ℝ := 1.5
noncomputable def side_dish_discount : ℝ := 0.25
noncomputable def euro_to_usd : ℝ := 1.20
noncomputable def payment_in_euros : ℝ := 50
noncomputable def tip_percentage(items_ordered : ℕ) : ℝ :=
  if items_ordered = 1 then 0.05 else
  if items_ordered = 2 then 0.10 else
  if items_ordered = 3 then 0.15 else 0.20

theorem change_in_euros :
  let items_ordered := 4 in -- meal, drink, dessert, side dish
  let total_before_discounts := meal_cost + drink_cost + dessert_cost + side_dish_cost in
  let dessert_discount_amount := dessert_cost * dessert_discount in
  let side_dish_discount_amount := side_dish_cost * side_dish_discount in
  let total_discounts := dessert_discount_amount + side_dish_discount_amount in
  let total_after_discounts := total_before_discounts - total_discounts in
  let total_with_tip := total_after_discounts * (1 + tip_percentage items_ordered) in
  let total_in_euros := total_with_tip / euro_to_usd in
  payment_in_euros - total_in_euros = 34.875 :=
by
  sorry

end change_in_euros_l401_401096


namespace correct_vector_equation_l401_401626

variables {V : Type*} [AddCommGroup V]

variables (A B C: V)

theorem correct_vector_equation : 
  (A - B) - (B - C) = A - C :=
sorry

end correct_vector_equation_l401_401626


namespace inequality_holds_for_all_real_l401_401674

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 1 ≥ 2 * |x| := sorry

end inequality_holds_for_all_real_l401_401674


namespace men_needed_for_new_job_l401_401644

theorem men_needed_for_new_job :
  ∀ (men1 days1 total_men_days new_factor days2 : ℕ),
  men1 = 250 →
  days1 = 16 →
  total_men_days = men1 * days1 →
  new_factor = 3 →
  days2 = 20 →
  ∃ men2 : ℕ, men2 * days2 = new_factor * total_men_days ∧ men2 = 600 :=
by
  intros men1 days1 total_men_days new_factor days2 h_men1 h_days1 h_total_men_days h_new_factor h_days2
  use 600
  split
  {
    rw [h_men1, h_days1, h_total_men_days, h_new_factor, h_days2]
    linarith
  }
  {
    refl
  }

end men_needed_for_new_job_l401_401644


namespace skating_probability_given_skiing_l401_401836

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l401_401836


namespace number_of_intersections_l401_401484

-- Defining the conditions of the problem statement
variable (A_1 B_1 C : Type)
variable [MetricSpace A_1] [MetricSpace B_1] [MetricSpace C]

-- Specific points and lines mentioned in the problem
def angle_A1B1C_right : Prop :=
  ∠A_1 B_1 C = 90

def ratio_CA1_CB1 : Prop :=
  ∀ A_1 C B_1 : ℝ, C A_1 / C B_1 = sqrt 5 + 2

def Ai_on_A1C (i : ℕ) (Ai : A_1) : Prop :=
  ∀ i ≥ 2, Ai ∈ line_through A_1 C ∧ (A_i B_{i-1} ⊥ A_1 C)

def Bi_on_B1C (i : ℕ) (Bi : B_1) : Prop :=
  ∀ i ≥ 2, Bi ∈ line_through B_1 C ∧ (A_i B_i ⊥ B_1 C)

def incircle_Γ1 : Prop :=
  incircle (triangle A_1 B_1 C) Γ_1

def tangent_Γi (i : ℕ) (Γi-1 Γi : Type) [Circle Γi-1] [Circle Γi] : Prop :=
  ∀ i ≥ 2, is_tangent_to Γi-1 Γi ∧ is_tangent_to Γi A_1 C ∧ is_tangent_to Γi B_1 C ∧ (radius Γi < radius Γi-1)

-- The main theorem statement proving the final answer
theorem number_of_intersections : ∃ k : ℕ, count_integers_k (intersect_line_circle A_1 B_{2016} Γ_k) = 4030 :=
sorry

end number_of_intersections_l401_401484


namespace round_table_arrangement_l401_401845

theorem round_table_arrangement :
  ∀ (n : ℕ), n = 10 → (∃ factorial_value : ℕ, factorial_value = Nat.factorial (n - 1) ∧ factorial_value = 362880) := by
  sorry

end round_table_arrangement_l401_401845


namespace total_cost_l401_401897

-- Definitions based on conditions
def regular_price : ℝ := 50
def discount_rate : ℝ := 0.20
def number_of_shirts : ℕ := 6

-- The theorem stating Marlene's total cost
theorem total_cost (regular_price discount_rate : ℝ) (number_of_shirts : ℕ) : ℝ :=
  let discount_amount := discount_rate * regular_price
  let sale_price := regular_price - discount_amount
  let total_cost := sale_price * number_of_shirts
  total_cost = 240 :=
by
  sorry

end total_cost_l401_401897


namespace andrea_rhinestones_needed_l401_401678

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l401_401678


namespace cos_angle_BAO_l401_401075

-- Define rectangle, points, and lengths in the conditions
def Rectangle (A B C D O : Type) : Prop :=
  -- Assume rectangle properties and diagonal intersection
  ∀ (A B C D : Type) (O : Type), 
    (AB = 15 ∧ BC = 20) ∧ 
    ∃ M N, (A = M ∧ B = N ∧ C = M + 20 ∧ D = N + 15 ∧ O = midpoint_of_diagonals AB BC)

-- State the theorem to prove using Lean statement
theorem cos_angle_BAO (A B C D O : Type) (H : Rectangle A B C D O) :
  \cos (\angle BAO) = \frac{3}{5} := 
  by
  sorry

end cos_angle_BAO_l401_401075


namespace find_integer_cosine_l401_401324

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401324


namespace inradius_of_triangle_area_three_times_perimeter_l401_401443

theorem inradius_of_triangle_area_three_times_perimeter (A p s r : ℝ) (h1 : A = 3 * p) (h2 : p = 2 * s) (h3 : A = r * s) (h4 : s ≠ 0) :
  r = 6 :=
sorry

end inradius_of_triangle_area_three_times_perimeter_l401_401443


namespace value_of_x_squared_add_reciprocal_squared_l401_401802

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l401_401802


namespace phone_calls_graph_l401_401182

theorem phone_calls_graph (n m : ℕ) (people : Fin n → Type)
    (phone_call : ∀ i j : Fin n, i ≠ j → Prop) :
    (∀ i j : Fin n, i ≠ j → phone_call i j ∨ phone_call j i → ¬(phone_call i j ∧ phone_call j i)) →
    (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → 
          ∃ k : ℕ, k = 3^m ∧ ∀ (s : Finset (Fin n)) (hs : s.card = n - 2),
            (∀ ⦃i j⦄, i ≠ j → i ∈ s → j ∈ s → phone_call i j ∧ (k = s.card)) →
            3^m = k) →
    n = 5 :=
begin
    sorry
end

end phone_calls_graph_l401_401182


namespace num_integers_diff_squares_l401_401044

theorem num_integers_diff_squares : 
  let count_diff_squares (n : ℕ) := (count (λ m, ∃ a b : ℕ, a^2 - b^2 = m) (range n)) in
  count_diff_squares 1001 = 750 :=
begin
  sorry
end

end num_integers_diff_squares_l401_401044


namespace second_half_takes_200_percent_longer_l401_401900

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l401_401900


namespace sum_of_proper_divisors_30_is_42_l401_401989

def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

-- The set of proper divisors of 30.
def proper_divisors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15}

-- The sum of all proper divisors of 30.
def sum_proper_divisors_30 : ℕ := proper_divisors_30.sum id

theorem sum_of_proper_divisors_30_is_42 : sum_proper_divisors_30 = 42 := 
by
  -- Proof can be filled in here
  sorry

end sum_of_proper_divisors_30_is_42_l401_401989


namespace prove_inequalities_l401_401420

theorem prove_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^3 * b > a * b^3 ∧ a - b / a > b - a / b :=
by
  sorry

end prove_inequalities_l401_401420


namespace FH_eq_2_ma_l401_401528

-- Definitions of Square Construction on the Sides of a Triangle and Median
structure Triangle (α : Type) := 
  (A B C : α) 

structure Square (α : Type) :=
  (A B C D : α)

variables {α : Type} [AddCommGroup α] [Module ℝ α]

-- Definition of Midpoint
def midpoint (A B : α) := (A + B) / 2

-- Definition of the Quadrilateral constructed squares on the sides of the triangle
def squares_constructed_on_sides (Δ : Triangle α) : Prop :=
  ∃ E F G H : α,
    Square.mk Δ.A Δ.B E F ∧ Square.mk Δ.A Δ.C G H

-- Definition of the length of the Median
def median_length (Δ : Triangle α) : ℝ :=
  let M := midpoint Δ.B Δ.C in
  (Δ.A - M).norm

noncomputable def diagonal_square (A B : α) :=
  ((A - B).norm) * nat.sqrt 2

def FH_length {Δ : Triangle α} (squares_cond : squares_constructed_on_sides Δ) (F H : α) : ℝ :=
  (F - H).norm

-- The main theorem statement for the problem
theorem FH_eq_2_ma {Δ : Triangle α}
  (squares_cond : squares_constructed_on_sides Δ)
  (m_a : ℝ) :
  FH_length squares_cond (diagonal_square Δ.A Δ.B) (diagonal_square Δ.A Δ.C) = 2 * median_length Δ := sorry

end FH_eq_2_ma_l401_401528


namespace solve_for_x_l401_401309

theorem solve_for_x (x : ℝ) (h : x ∈ set_of (λ x, (8 / (√(x - 5) - 10) + 2 / (√(x - 5) - 5) + 10 / (√(x - 5) + 5) + 16 / (√(x - 5) + 10) = 0))) : 
  x = 41.67 :=
sorry

end solve_for_x_l401_401309


namespace max_true_statements_l401_401105

theorem max_true_statements (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : 
  (∃ (s : Finset (ℕ → ℝ → ℝ → Prop)), s ⊆ {λ x y z, 1 / x > 1 / y, λ x y z, abs (x^2) < abs (y^2), λ x y z, x > y, λ x y z, 0 < x, λ x y z, 0 < y, λ x y z, abs x > abs y} ∧ 
  s.card > 4 → False) :=
by
  sorry

end max_true_statements_l401_401105


namespace ship_with_highest_no_car_round_trip_percentage_l401_401515

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l401_401515


namespace line_equation_curve_equation_max_distance_l401_401076

/-- Given the parametric equations of line l:
      x = 3 - t
      y = 1 + t
    Prove that the general equation of line l is x + y - 4 = 0.
-/
theorem line_equation (t : ℝ) : ∃ (x y : ℝ), (x = 3 - t) ∧ (y = 1 + t) ∧ (x + y - 4 = 0) :=
by
  sorry

/-- Given the polar coordinate equation of curve C:
      ρ = 2√2 cos(θ - π/4)
    Prove that the Cartesian coordinate equation of curve C is (x - 1)^2 + (y - 1)^2 = 2.
-/
theorem curve_equation (ρ θ : ℝ) (x y : ℝ) :
  (ρ = 2 * Real.sqrt 2 * Real.cos (θ - (Real.pi / 4))) →
  (ρ^2 = x^2 + y^2) →
  (ρ * Real.cos θ = x) →
  (ρ * Real.sin θ = y) →
  ((x - 1)^2 + (y - 1)^2 = 2) :=
by
  sorry

/-- Given the Cartesian coordinate equation of curve C:
      (x - 1)^2 + (y - 1)^2 = 2,
    and the line equation x + y - 4 = 0,
    Prove that the maximum distance from a point on curve C to line l is 2√2.
-/
theorem max_distance (x y d : ℝ) :
  ((x - 1)^2 + (y - 1)^2 = 2) →
  (x + y - 4 = 0) →
  (∃ (α : ℝ), (1 + Real.sqrt 2 * Real.cos α) = x ∧ (1 + Real.sqrt 2 * Real.sin α) = y) →
  (d = Real.abs ((2 * Real.sin (α + Real.pi / 4)) - 2) / Real.sqrt 2) →
  (d = 2 * Real.sqrt 2) :=
by
  sorry

end line_equation_curve_equation_max_distance_l401_401076


namespace parallel_tangents_l401_401482

theorem parallel_tangents
  (ABC : Triangle)
  (Γ : Circle)
  (D E : Point)
  (ω_B ω_C : Circle)
  (X Y : Point)
  (h_inscribed : ABC.inscribed_in Γ)
  (h_D_on_arc_BA : D ∈ Γ.arc_not_containing C)
  (h_E_on_arc_CA : E ∈ Γ.arc_not_containing B)
  (h_DE_parallel_BC : DE ∥ BC)
  (h_ω_B_tangent : ω_B.tangent_internally Γ (Γ.arc_not_containing B) ∧ ω_B.tangent_line (AB) ∧ ω_B.tangent_line (DE))
  (h_ω_C_tangent : ω_C.tangent_internally Γ (Γ.arc_not_containing C) ∧ ω_C.tangent_line (AC) ∧ ω_C.tangent_line (DE))
  (h_X_tangency : ω_B.tangent_point (AB) = X)
  (h_Y_tangency : ω_C.tangent_point (AC) = Y) :
  XY ∥ BC := sorry

end parallel_tangents_l401_401482


namespace repeating_block_length_of_7_over_15_l401_401943

noncomputable def decimal_expansion_repeats (n d : ℚ) :=
  ∃ b, ∃ l : ℕ, (n / d).to_decimal_digits.repeat_cycle_length = l ∧ l = b

theorem repeating_block_length_of_7_over_15 : decimal_expansion_repeats 7 15 ∧ (decimal_expansion_repeats 7 15) = 1 := by
  sorry

end repeating_block_length_of_7_over_15_l401_401943


namespace log_base_0_2_monotonic_decrease_interval_l401_401948

noncomputable def function_monotonic_decrease_interval : Set ℝ :=
  {x : ℝ | x > 3}

theorem log_base_0_2_monotonic_decrease_interval :
  ∀ x : ℝ,
    (x^2 - 2 * x - 3 > 0) → x > 3 → (∃ y : ℝ, y = (Real.log 0.2 (x^2 - 2 * x - 3)) ∧ y ∈ function_monotonic_decrease_interval) := sorry

end log_base_0_2_monotonic_decrease_interval_l401_401948


namespace productivity_after_repair_l401_401134

-- Define the initial productivity and the increase factor.
def original_productivity : ℕ := 10
def increase_factor : ℝ := 1.5

-- Define the expected productivity after the improvement.
def expected_productivity : ℝ := 25

-- The theorem we need to prove.
theorem productivity_after_repair :
  original_productivity * (1 + increase_factor) = expected_productivity := by
  sorry

end productivity_after_repair_l401_401134


namespace sufficient_but_not_necessary_l401_401746

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then m + Real.log2 x else 0

theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 0 → ∃ x ≥ 1, f m x = 0) ∧ (∃ x ≥ 1, f 0 x = 0) :=
by
  sorry

end sufficient_but_not_necessary_l401_401746


namespace find_midpoint_l401_401095

noncomputable def trisect (A B: Point) : Point × Point := sorry

theorem find_midpoint (A B C D E F G H M: Point) (trisect: Point → Point → (Point × Point)) 
  (h1: C ≠ A ∧ C ≠ B ∧ ¬Collinear A B C)
  (trisect_AC: trisect A C = (D, E))
  (trisect_BC: trisect B C = (F, G))
  (H_def: Intersect (CollinearPoints D G) (CollinearPoints E F) = H)
  (M_def: Intersect (CollinearPoints C H) (CollinearPoints A B) = M) :
  Midpoint M A B := 
sorry

end find_midpoint_l401_401095


namespace find_n_l401_401333

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401333


namespace number_of_valid_starting_lineups_l401_401162

theorem number_of_valid_starting_lineups : 
  let players := {Ben, Tom, Dan, Alex, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15}
  let invalidLineup (l : Finset Player) := 
    l.contains Ben ∧ l.contains Tom ∨ 
    l.contains Ben ∧ l.contains Dan ∨ 
    l.contains Tom ∧ l.contains Dan ∨
    l.contains Alex
  let countValidLineups (n : ℕ) := 
    (Finset.powersetLen n players).filter (λ l => l.card = 5 ∧ ¬ invalidLineup l)
  countValidLineups 5.card = 1452 :=
sorry

end number_of_valid_starting_lineups_l401_401162


namespace simplifiedtown_path_difference_l401_401063

/-- In Simplifiedtown, all streets are 30 feet wide. Each enclosed block forms a square with 
each side measuring 400 feet. Sarah runs exactly next to the block on a path that is 400 feet 
from the block's inner edge while Maude runs on the outer edge of the street opposite to 
Sarah. Prove that Maude runs 120 feet more than Sarah for each lap around the block. -/
theorem simplifiedtown_path_difference :
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  maude_lap - sarah_lap = 120 :=
by
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  show maude_lap - sarah_lap = 120
  sorry

end simplifiedtown_path_difference_l401_401063


namespace symmetric_point_yoz_l401_401857

-- Given a point P in 3D space and its reflection with respect to the yoz plane
def point := (ℝ × ℝ × ℝ)

def reflect_y_o_z_plane (P : point) : point :=
  (-P.1, P.2, P.3)

theorem symmetric_point_yoz (P : point) (hP : P = (2, 3, 5)) : reflect_y_o_z_plane P = (-2, 3, 5) :=
by
  sorry

end symmetric_point_yoz_l401_401857


namespace relationship_among_x_y_z_w_l401_401812

theorem relationship_among_x_y_z_w (x y z w : ℝ) (h : (x + y) / (y + z) = (z + w) / (w + x)) :
  x = z ∨ x + y + w + z = 0 :=
sorry

end relationship_among_x_y_z_w_l401_401812


namespace expression_simplification_l401_401723

theorem expression_simplification : 2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := 
by 
  sorry

end expression_simplification_l401_401723


namespace units_digit_2009_2008_plus_2013_l401_401994

theorem units_digit_2009_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 :=
by
  sorry

end units_digit_2009_2008_plus_2013_l401_401994


namespace minimum_chess_pieces_l401_401628

def sum_is_distinct (sums : List ℕ) : Prop :=
  list.nodup sums

theorem minimum_chess_pieces : ∃ (g : ℕ → ℕ → ℕ), 
  (∀ i j, g i j ≥ 0) ∧
  let r := [g 0 0 + g 0 1 + g 0 2, g 1 0 + g 1 1 + g 1 2, g 2 0 + g 2 1 + g 2 2] in
  let c := [g 0 0 + g 1 0 + g 2 0, g 0 1 + g 1 1 + g 2 1, g 0 2 + g 1 2 + g 2 2] in
  sum_is_distinct (r ++ c) ∧
  r.sum = c.sum ∧
  r.sum = 8 := sorry

end minimum_chess_pieces_l401_401628


namespace investment_doubles_in_9_years_l401_401823

noncomputable def years_to_double (initial_amount : ℕ) (interest_rate : ℕ) : ℕ :=
  72 / interest_rate

theorem investment_doubles_in_9_years :
  ∀ (initial_amount : ℕ) (interest_rate : ℕ) (investment_period_val : ℕ) (expected_value : ℕ),
  initial_amount = 8000 ∧ interest_rate = 8 ∧ investment_period_val = 18 ∧ expected_value = 32000 →
  years_to_double initial_amount interest_rate = 9 :=
by
  intros initial_amount interest_rate investment_period_val expected_value h
  sorry

end investment_doubles_in_9_years_l401_401823


namespace value_of_x_squared_add_reciprocal_squared_l401_401803

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l401_401803


namespace larry_result_is_correct_l401_401494

theorem larry_result_is_correct (a b c d e : ℤ) 
  (h1: a = 2) (h2: b = 4) (h3: c = 3) (h4: d = 5) (h5: e = -15) :
  a - (b - (c * (d + e))) = (-17 + e) :=
by 
  rw [h1, h2, h3, h4, h5]
  sorry

end larry_result_is_correct_l401_401494


namespace prove_arithmetic_sequence_properties_l401_401762

noncomputable def arithmetic_sequence :=
{ a_n : ℕ → ℝ // ∃ d a₁, ∀ n, a_n n = a₁ + n * d }

variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(∑ i in range(n), a i)

variables {S : ℕ → ℝ} {a : ℕ → ℝ}
variables (h1 : S 2018 < S 2019) (h2 : S 2019 > S 2020)
variables (hS : ∀ n, S n = sum_first_n_terms a n)
variables (ha : arithmetic_sequence a)

theorem prove_arithmetic_sequence_properties :
  (∀ m, a 1 ≥ a m) ∧ (∀ n, n ≥ 2020 → a n < 0) :=
by sorry

end prove_arithmetic_sequence_properties_l401_401762


namespace intersection_of_sets_l401_401029

theorem intersection_of_sets (x : ℝ) : 
  (x^2 < 1) ∧ (2^x > 1) ↔ (0 < x ∧ x < 1) :=
by 
  sorry

end intersection_of_sets_l401_401029


namespace jill_arrives_before_jack_l401_401085

-- Define the distance to the park
def distance_to_park := 3 -- miles

-- Define Jack's walking speed
def jack_speed := 3 -- miles per hour

-- Define Jill's biking speed
def jill_speed := 12 -- miles per hour

-- Define Jill's delay in starting
def jill_delay := 5 -- minutes

-- Prove Jill arrives 40 minutes before Jack
theorem jill_arrives_before_jack :
  ∀ (jack_travel_time jill_travel_time : ℕ), 
  (jack_travel_time = distance_to_park / jack_speed * 60) → -- Jack's travel time in minutes
  (jill_travel_time = distance_to_park / jill_speed * 60 + jill_delay) → -- Jill's travel time in minutes
  (jack_travel_time - jill_travel_time = 40) :=
by
  intros,
  -- Skipping the proof itself
  sorry

end jill_arrives_before_jack_l401_401085


namespace find_b_of_bisector_l401_401582

-- Definitions based on the problem conditions
def is_midpoint (P1 P2 M : (ℝ × ℝ)) : Prop :=
  M = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def is_bisector (L : (ℝ × ℝ) → Prop) (P1 P2 : (ℝ × ℝ)) : Prop :=
  ∃ M : (ℝ × ℝ), is_midpoint P1 P2 M ∧ L M

-- Problem statement
theorem find_b_of_bisector :
  (∀ x y b : ℝ, (x + y = b) → is_bisector (λ P, P.1 + P.2 = b) (2, 4) (6, 8) → b = 10) :=
by
  intros x y b H H_bisector
  sorry

end find_b_of_bisector_l401_401582


namespace candy_cost_is_correct_l401_401496

def michael_money : ℕ := 42
def brother_initial_money : ℕ := 17

def money_given_to_brother (m : ℕ) : ℕ := m / 2
def brother_money_after_gift (b m : ℕ) : ℕ := b + money_given_to_brother(m)

def brother_money_after_buying_candy (b_after_gift : ℕ) (candy_cost final_money : ℕ) : Prop :=
  b_after_gift - candy_cost = final_money

theorem candy_cost_is_correct :
  brother_money_after_buying_candy (brother_money_after_gift brother_initial_money michael_money) 3 35 :=
by
  sorry

end candy_cost_is_correct_l401_401496


namespace largest_lambda_l401_401376

theorem largest_lambda (n : ℕ) (h : 2 ≤ n)
  (bags : Fin n → ℕ → ℕ)
  (pow_two_weighted : ∀ (i : Fin n) (j : ℕ), ∃ k : ℕ, bags i j = 2^k)
  (equal_total_weight : ∀ (i j : Fin n), (Finset.univ.sum (bags i)) = (Finset.univ.sum (bags j))) :
  ∃ λ, λ = nat.floor (n / 2) + 1 ∧ ∀ (w : ℕ), ∃ (i : Fin n), bags i w ≥ λ :=
sorry

end largest_lambda_l401_401376


namespace matrix_series_product_l401_401300

theorem matrix_series_product :
  (List.foldl (λ (mat_acc : Matrix (Fin 2) (Fin 2) ℚ) (mat : Matrix (Fin 2) (Fin 2) ℚ), mat_acc ⬝ mat)
              (1 : Matrix (Fin 2) (Fin 2) ℚ)
              (List.map (λ a, ![![1, a], ![0, 1]]) (List.range' 2 50).map (λ x, 2 * (x + 1))))
  = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_series_product_l401_401300


namespace sales_tax_calculation_l401_401654

theorem sales_tax_calculation 
  (total_amount_paid : ℝ)
  (tax_rate : ℝ)
  (cost_tax_free : ℝ) :
  total_amount_paid = 30 → tax_rate = 0.08 → cost_tax_free = 12.72 → 
  (∃ sales_tax : ℝ, sales_tax = 1.28) :=
by
  intros H1 H2 H3
  sorry

end sales_tax_calculation_l401_401654


namespace union_of_A_and_B_l401_401383

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def union_AB := {x : ℝ | 1 < x ∧ x ≤ 8}

theorem union_of_A_and_B : A ∪ B = union_AB :=
sorry

end union_of_A_and_B_l401_401383


namespace count_zigzag_paths_l401_401274

-- Define what a zigzag is in the context of crossing exactly 8 white squares and no black squares on an 8x8 chessboard.
def is_zigzag_path (path : list (ℕ × ℕ)) : Prop :=
  path.length = 8 ∧
  (∀ i < 8, (path.nth i).is_some ∧
    let (x, y) := path.nth_le i sorry in
    ((x + y) % 2 = 0)) -- white square condition

-- Define the total number of zigzag paths from one edge of the board to the opposite edge.
def total_zigzag_paths : ℕ :=
  sorry -- Recursive counting implementation

-- The theorem to prove
theorem count_zigzag_paths : total_zigzag_paths = 296 :=
by
  sorry

end count_zigzag_paths_l401_401274


namespace quadratic_function_origin_l401_401640

theorem quadratic_function_origin {a b c : ℝ} :
  (∀ x, y = ax * x + bx * x + c → y = 0 → 0 = c ∧ b = 0) ∨ (c = 0) :=
sorry

end quadratic_function_origin_l401_401640


namespace exists_kite_formation_l401_401910

def point := (ℕ × ℕ)
def is_kite (p1 p2 p3 p4 : point) : Prop :=
  -- Define the properties that make four points (p1, p2, p3, p4) a kite shape here.
  -- For example:
  sorry

def count_kites (points : list point) : ℕ :=
  -- Count the number of kites that can be formed by any four distinct points in the list.
  sorry

theorem exists_kite_formation : ∃ points : list point, points.length = 10 ∧ count_kites points = 5 :=
by {
  sorry
}

end exists_kite_formation_l401_401910


namespace mark_new_phone_plan_cost_l401_401128

noncomputable def total_new_plan_cost (old_plan_cost old_internet_cost old_intl_call_cost : ℝ) (percent_increase_plan percent_increase_internet percent_decrease_intl : ℝ) : ℝ :=
  let new_plan_cost := old_plan_cost * (1 + percent_increase_plan)
  let new_internet_cost := old_internet_cost * (1 + percent_increase_internet)
  let new_intl_call_cost := old_intl_call_cost * (1 - percent_decrease_intl)
  new_plan_cost + new_internet_cost + new_intl_call_cost

theorem mark_new_phone_plan_cost :
  let old_plan_cost := 150
  let old_internet_cost := 50
  let old_intl_call_cost := 30
  let percent_increase_plan := 0.30
  let percent_increase_internet := 0.20
  let percent_decrease_intl := 0.15
  total_new_plan_cost old_plan_cost old_internet_cost old_intl_call_cost percent_increase_plan percent_increase_internet percent_decrease_intl = 280.50 :=
by
  sorry

end mark_new_phone_plan_cost_l401_401128


namespace three_digit_numbers_greater_than_200_in_set_l401_401417

-- Definition of the conditions
def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

def digit_set := {1, 3, 5}

def three_digit_number (hundreds tens ones : ℕ) : ℕ :=
  100 * hundreds + 10 * tens + ones

def is_valid_digit (d : ℕ) : Prop := d ∈ digit_set

-- Formal statement of the problem
theorem three_digit_numbers_greater_than_200_in_set :
  (∃ (count : ℕ), count = 18 ∧ ∀ n, is_three_digit n →
     n > 200 →
     (∃ (h t o : ℕ), n = three_digit_number h t o ∧ is_valid_digit h ∧ is_valid_digit t ∧ is_valid_digit o)) :=
sorry

end three_digit_numbers_greater_than_200_in_set_l401_401417


namespace total_distance_covered_l401_401612

/-- There are three trains A, B, and C with speeds of 110 kmph, 150 kmph, and 180 kmph respectively.
Given that they travel for 11 minutes, the total distance covered by all three trains is 80.652 km. -/
theorem total_distance_covered :
  let time := 11 / 60 in
  let speed_A := 110 in
  let speed_B := 150 in
  let speed_C := 180 in
  let dist_A := speed_A * time in
  let dist_B := speed_B * time in
  let dist_C := speed_C * time in
  dist_A + dist_B + dist_C = 80.652 := sorry

end total_distance_covered_l401_401612


namespace Hannah_van_meet_only_once_l401_401037

-- Define the movements and positions
def speed_Hannah : ℝ := 6  -- Hannah's speed (feet per second)
def speed_van : ℝ := 12  -- Van's speed (feet per second)
def stop_time_van : ℝ := 45  -- Van's stop time at each pail (seconds)
def pail_distance : ℝ := 150  -- Distance between pails (feet)

-- Initial positions at time t = 0
def initial_position_Hannah : ℝ := 0
def initial_position_van : ℝ := pail_distance

-- Function to describe position of Hannah over time
def position_Hannah (t : ℝ) : ℝ := speed_Hannah * t

-- Function to describe the van's movement and stops over time
noncomputable def position_van (t : ℝ) : ℝ := 
    let cycle_time := stop_time_van + pail_distance / speed_van in
    let complete_cycles := (t / cycle_time).to_nat in
    let remaining_time := t % cycle_time in
    let van_moving_time := pail_distance / speed_van in
    if remaining_time < van_moving_time then
      initial_position_van + complete_cycles * pail_distance + remaining_time * speed_van
    else
      initial_position_van + (complete_cycles + 1) * pail_distance

-- Define the total number of meetings between Hannah and the van
def meeting_times : ℕ := 
    if position_Hannah 57.5 = position_van 57.5 then 1 else 0  -- put simply for proof

-- The theorem statement
theorem Hannah_van_meet_only_once : meeting_times = 1 :=
sorry

end Hannah_van_meet_only_once_l401_401037


namespace area_increment_of_closed_curve_l401_401657

theorem area_increment_of_closed_curve (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  ∃ (C : set (ℝ × ℝ)), closed C ∧ s = boundary of some shape such that
  (length (boundary C) = 2 * a * π) ∧ (measure (interior C) = π * a * b + (a - b)^2) :=
sorry

end area_increment_of_closed_curve_l401_401657


namespace power_expression_eval_l401_401050

theorem power_expression_eval (a b : ℝ)
  (h1 : 60^a = 3)
  (h2 : 60^b = 5) :
  12^((1 - a - b) / (2 * (1 - b))) = 2 := 
sorry

end power_expression_eval_l401_401050


namespace continuous_value_at_one_l401_401615

theorem continuous_value_at_one : 
  limit (fun x => (x^3 - 1) / (x^2 - 1)) 1 = 3 / 2 := 
by sorry

end continuous_value_at_one_l401_401615


namespace total_profit_l401_401844

theorem total_profit (x : ℝ) (y : ℝ) (B_profit : ℝ) (A_investment_ratio : ℝ) (B_investment_ratio : ℝ) 
  (profit_ratio : ℝ) :
  A_investment_ratio = 2 / 3 →
  B_investment_ratio = 1 / 2 →
  B_profit = 75000 →
  profit_ratio = B_investment_ratio →
  ∃ (total_profit : ℝ), total_profit = 75000 * 2 :=
by
  intros hA hB hP hR
  use (75000 * 2)
  sorry

end total_profit_l401_401844


namespace find_g_l401_401543

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l401_401543


namespace f_20_equals_97_l401_401119

noncomputable def f_rec (f : ℕ → ℝ) (n : ℕ) := (2 * f n + n) / 2

theorem f_20_equals_97 (f : ℕ → ℝ) (h₁ : f 1 = 2)
    (h₂ : ∀ n : ℕ, f (n + 1) = f_rec f n) : 
    f 20 = 97 :=
sorry

end f_20_equals_97_l401_401119


namespace tangent_line_at_point_l401_401575

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem tangent_line_at_point (x y : ℝ) (slope : ℝ) (tangent : ℝ → ℝ → Prop) :
  (curve 1 = 6) →
  (slope = deriv curve 1) →
  (slope = 9) →
  tangent 1 6 ↔ tangent = λ x y, 9 * x - y - 3 = 0 :=
begin
  sorry
end

end tangent_line_at_point_l401_401575


namespace swap_correct_l401_401413

variables (a b c : ℕ) -- assuming natural numbers for simplicity; generalize if needed
variables (a0 b0 : ℕ)

theorem swap_correct (h1 : a = a0) (h2 : b = b0) :
  let c := a in
  let a := b in
  let b := c in
  a = b0 ∧ b = a0 :=
by {
  sorry
}

end swap_correct_l401_401413


namespace find_ellipse_equation_l401_401949

noncomputable def ellipse_standard_eq (x y : ℝ) (a : ℝ) (b : ℝ) := 
  x^2 / a^2 + y^2 / b^2 - 1

noncomputable def hyperbola_eq (x y : ℝ) := 
  x^2 / 4 - y^2 / 5 - 1

theorem find_ellipse_equation :
  (∃ F1 F2 M N : ℝ × ℝ, 
    let E1_c := hyperbola_eq F1.1 F1.2 = 0 ∧ hyperbola_eq F2.1 F2.2 = 0 ∧ 
               hyperbola_eq M.1 M.2 = 0 ∧ hyperbola_eq N.1 N.2 = 0 in
      F1 = (-3, 0) ∧ F2 = (3, 0) ∧
      ∃ a b, 0 < b ∧ b < a ∧ 
        (ellipse_standard_eq F2.1 F2.2 a b = 0) ∧ 
        ∃ E2: ℝ × ℝ → ℝ, 
          (ellipse_standard_eq M.1 M.2 a b = 0) → 
            ellipse_standard_eq x y (9/2) (3*sqrt(5)/2) = 0
  ) :=
  sorry

end find_ellipse_equation_l401_401949


namespace sum_of_roots_of_quadratic_l401_401735

theorem sum_of_roots_of_quadratic :
  (∑ x in { x : ℝ | x^2 + 2016 * x - 2017 = 0 }, x) = -2016 :=
sorry

end sum_of_roots_of_quadratic_l401_401735


namespace fourth_student_guess_l401_401565

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l401_401565


namespace cube_root_of_neg_27_l401_401698

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end cube_root_of_neg_27_l401_401698


namespace collinear_points_z_l401_401399

open Complex

theorem collinear_points_z (z : ℂ) (hz : abs z = 5) (hcol : collinear ℝ ![(1 : ℂ), (i : ℂ), z]) :
  z = 4 - 3 * I ∨ z = -3 + 4 * I :=
by
  sorry

end collinear_points_z_l401_401399


namespace sum_of_terms_in_geometric_sequence_eq_fourteen_l401_401754

theorem sum_of_terms_in_geometric_sequence_eq_fourteen
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_a1 : a 1 = 1)
  (h_arith : 4 * a 2 = 2 * a 3 ∧ 2 * a 3 - 4 * a 2 = a 4 - 2 * a 3) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_of_terms_in_geometric_sequence_eq_fourteen_l401_401754


namespace infinitely_many_nonintersecting_lines_angle_sum_triangle_l401_401934

-- Definitions of the Poincare plane, lines, and points
noncomputable def Point : Type := sorry
noncomputable def Line : Type := sorry
noncomputable def is_in_poincare_plane (R : Line) : Prop := sorry
noncomputable def is_perpendicular_to (L R : Line) : Prop := sorry
noncomputable def has_center_on (L R : Line) : Prop := sorry
noncomputable def pass_through (L : Line) (P : Point) : Prop := sorry
noncomputable def not_intersect (L1 L2 : Line) : Prop := sorry
noncomputable def not_on (P : Point) (L : Line) : Prop := sorry
noncomputable def is_triangle (A B C : Point) : Prop := sorry
noncomputable def angle_sum (A B C : Point) : ℝ := sorry

-- Assumptions
variables (R : Line) [is_in_poincare_plane R]

-- The first proof problem
theorem infinitely_many_nonintersecting_lines (L : Line) (P : Point) 
  (hL1 : is_perpendicular_to L R ∨ has_center_on L R) 
  (hP : not_on P L) : 
  ∃ (lines_set : set Line), infinite lines_set ∧ ∀ (l : Line), l ∈ lines_set → pass_through l P ∧ not_intersect l L :=
sorry

-- The second proof problem
theorem angle_sum_triangle (A B C : Point) 
  (hABC : is_triangle A B C) : 
  0 < angle_sum A B C ∧ angle_sum A B C < π :=
sorry


end infinitely_many_nonintersecting_lines_angle_sum_triangle_l401_401934


namespace solve_equation_l401_401536

theorem solve_equation (x : ℝ) : 
  (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1 / 3 :=
by sorry

end solve_equation_l401_401536


namespace total_wheels_correct_l401_401848

-- Define the initial state of the garage
def initial_bicycles := 20
def initial_cars := 10
def initial_motorcycles := 5
def initial_tricycles := 3
def initial_quads := 2

-- Define the changes in the next hour
def bicycles_leaving := 7
def cars_arriving := 4
def motorcycles_arriving := 3
def motorcycles_leaving := 2

-- Define the damaged vehicles
def damaged_bicycles := 5  -- each missing 1 wheel
def damaged_cars := 2      -- each missing 1 wheel
def damaged_motorcycle := 1 -- missing 2 wheels

-- Define the number of wheels per type of vehicle
def bicycle_wheels := 2
def car_wheels := 4
def motorcycle_wheels := 2
def tricycle_wheels := 3
def quad_wheels := 4

-- Calculate the state of vehicles at the end of the hour
def final_bicycles := initial_bicycles - bicycles_leaving
def final_cars := initial_cars + cars_arriving
def final_motorcycles := initial_motorcycles + motorcycles_arriving - motorcycles_leaving

-- Calculate the total wheels in the garage at the end of the hour
def total_wheels : Nat := 
  (final_bicycles - damaged_bicycles) * bicycle_wheels + damaged_bicycles +
  (final_cars - damaged_cars) * car_wheels + damaged_cars * 3 +
  (final_motorcycles - damaged_motorcycle) * motorcycle_wheels +
  initial_tricycles * tricycle_wheels +
  initial_quads * quad_wheels

-- The goal is to prove that the total number of wheels in the garage is 102 at the end of the hour
theorem total_wheels_correct : total_wheels = 102 := 
  by
    sorry

end total_wheels_correct_l401_401848


namespace midpoint_sum_l401_401570

-- Defining the coordinates of the endpoints.
def x1 : ℝ := 3
def y1 : ℝ := 4
def x2 : ℝ := 9
def y2 : ℝ := 18

-- Defining the midpoint coordinates.
def midpoint_x : ℝ := (x1 + x2) / 2
def midpoint_y : ℝ := (y1 + y2) / 2

-- Defining the sum of the coordinates of the midpoint.
def sum_of_midpoint_coordinates : ℝ := midpoint_x + midpoint_y

-- The statement we need to prove.
theorem midpoint_sum : sum_of_midpoint_coordinates = 17 := by
  -- Proof goes here
  sorry

end midpoint_sum_l401_401570


namespace maximum_watchman_demand_l401_401606

theorem maximum_watchman_demand (bet_loss : ℕ) (bet_win : ℕ) (x : ℕ) 
  (cond_bet_loss : bet_loss = 100)
  (cond_bet_win : bet_win = 100) :
  x < 200 :=
by
  have h₁ : bet_loss = 100 := cond_bet_loss
  have h₂ : bet_win = 100 := cond_bet_win
  sorry

end maximum_watchman_demand_l401_401606


namespace range_of_a_l401_401624

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → a < g x) : a < 0 := 
by sorry

end range_of_a_l401_401624


namespace sum_solutions_eq_l401_401991

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end sum_solutions_eq_l401_401991


namespace complex_number_in_second_quadrant_l401_401586

open Complex

/--
  Prove that the point corresponding to the complex number 
  z = i / (1 - i) in the complex plane is in the second quadrant.
-/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (i : ℂ) / (1 - i)
  Re z < 0 ∧ Im z > 0 :=
by
  let z := (i : ℂ) / (1 - i)
  sorry

end complex_number_in_second_quadrant_l401_401586


namespace highest_percentage_without_car_l401_401509

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l401_401509


namespace unique_solution_2x_plus_3_eq_11y_l401_401305

theorem unique_solution_2x_plus_3_eq_11y (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) :
  (2^x + 3 = 11^y) ↔ (x = 3 ∧ y = 1) :=
by
  intro h
  split
  {
    -- Handle the case where 2^x + 3 = 11^y
    have : x = 3 := sorry
    have : y = 1 := sorry
    exact ⟨this, this.right⟩
  }
  {
    -- Handle the case where (x, y) = (3, 1)
    rintros ⟨hx3, hy1⟩
    subst hx3
    subst hy1
    norm_num
  }

end unique_solution_2x_plus_3_eq_11y_l401_401305


namespace min_handshakes_l401_401643

theorem min_handshakes (n : ℕ) (h1 : n = 25) 
  (h2 : ∀ (p : ℕ), p < n → ∃ q r : ℕ, q ≠ r ∧ q < n ∧ r < n ∧ q ≠ p ∧ r ≠ p) 
  (h3 : ∃ a b c : ℕ, a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(∃ d : ℕ, (d = a ∨ d = b ∨ d = c) ∧ (¬(a = d ∨ b = d ∨ c = d)) ∧ d < n)) :
  ∃ m : ℕ, m = 28 :=
by
  sorry

end min_handshakes_l401_401643


namespace sum_vectors_n_gon_center_zero_l401_401527

theorem sum_vectors_n_gon_center_zero (n : ℕ) (h_regular : n ≥ 3) :
  let vertices := (0 : ℂ) :: (List.range n).map (λ k, Complex.exp (2 * Complex.pi * Complex.I * k / n)) in
  (vertices.mkArray n).sum = 0 := by
  sorry

end sum_vectors_n_gon_center_zero_l401_401527


namespace matrix_cube_computation_l401_401708

-- Define the original matrix
def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2], ![2, 0]]

-- Define the expected result matrix
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-8, 0], ![0, -8]]

-- State the theorem to be proved
theorem matrix_cube_computation : matrix1 ^ 3 = expected_matrix :=
  by sorry

end matrix_cube_computation_l401_401708


namespace p_adic_valuation_of_factorial_l401_401138

noncomputable def digit_sum (n p : ℕ) : ℕ :=
  -- Definition for sum of digits of n in base p
  sorry

def p_adic_valuation (n factorial : ℕ) (p : ℕ) : ℕ :=
  -- Representation of p-adic valuation of n!
  sorry

theorem p_adic_valuation_of_factorial (n p : ℕ) (hp: p > 1):
  p_adic_valuation n.factorial p = (n - digit_sum n p) / (p - 1) :=
sorry

end p_adic_valuation_of_factorial_l401_401138


namespace relation_among_a_b_c_l401_401388

theorem relation_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = (3 / 5)^4)
  (h2 : b = (3 / 5)^3)
  (h3 : c = Real.log (3 / 5) / Real.log 3) :
  c < a ∧ a < b :=
by
  sorry

end relation_among_a_b_c_l401_401388


namespace sum_of_base_areas_eq_5_l401_401604

-- Define the surface area, lateral area, and the sum of the areas of the two base faces.
def surface_area : ℝ := 30
def lateral_area : ℝ := 25
def sum_base_areas : ℝ := surface_area - lateral_area

-- The theorem statement.
theorem sum_of_base_areas_eq_5 : sum_base_areas = 5 := 
by 
  sorry

end sum_of_base_areas_eq_5_l401_401604


namespace sum_of_solutions_eq_one_l401_401594

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l401_401594


namespace no_intersection_l401_401045

def f₁ (x : ℝ) : ℝ := abs (3 * x + 6)
def f₂ (x : ℝ) : ℝ := -abs (4 * x - 1)

theorem no_intersection : ∀ x, f₁ x ≠ f₂ x :=
by
  sorry

end no_intersection_l401_401045


namespace compare_game_probabilities_l401_401648

noncomputable def probability_heads := 3 / 5
noncomputable def probability_tails := 2 / 5

/-- Probability of winning Game A is the sum of probabilities of getting all heads or all tails in four flips. -/
def win_prob_game_a : ℚ :=
  (probability_heads ^ 4) + (probability_tails ^ 4)

/-- Probability of winning Game B is the product of the probabilities that first two flips are the same and last three flips are the same. -/
def win_prob_game_b : ℚ :=
  ((probability_heads ^ 2) + (probability_tails ^ 2)) * ((probability_heads ^ 3) + (probability_tails ^ 3))

/-- Comparison between the probabilities of winning Game A and Game B. -/
theorem compare_game_probabilities : win_prob_game_a = win_prob_game_b + (6 / 625) :=
by
  sorry

end compare_game_probabilities_l401_401648


namespace product_of_g_xi_l401_401112

noncomputable def x1 : ℂ := sorry
noncomputable def x2 : ℂ := sorry
noncomputable def x3 : ℂ := sorry
noncomputable def x4 : ℂ := sorry
noncomputable def x5 : ℂ := sorry

def f (x : ℂ) : ℂ := x^5 + x^2 + 1
def g (x : ℂ) : ℂ := x^3 - 2

axiom roots_of_f (x : ℂ) : f x = 0 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5

theorem product_of_g_xi : (g x1) * (g x2) * (g x3) * (g x4) * (g x5) = -243 := sorry

end product_of_g_xi_l401_401112


namespace projection_matrix_ratio_l401_401718

theorem projection_matrix_ratio 
  (x y : ℝ)
  (h : (λ (v : ℝ × ℝ), (⟨ (1 / 13) * v.1 - (12 / 13) * v.2, 
                            -(12 / 13) * v.1 + (4 / 13) * v.2 ⟩ : ℝ × ℝ)) (x, y) = (x, y)) : 
  y / x = -4 / 3 :=
by 
  sorry

end projection_matrix_ratio_l401_401718


namespace exists_quadrilateral_equal_parts_and_ratios_l401_401918

theorem exists_quadrilateral_equal_parts_and_ratios : ∃ (ABCD : Type) (M N : ABCD), 
  (ABCD.side_bisects M) ∧ (ABCD.side_divides_in_ratio CD N 2 1) ∧ (ABCD.segment_divides_equal MN) :=
sorry

end exists_quadrilateral_equal_parts_and_ratios_l401_401918


namespace sum_of_cubes_divisible_l401_401151

theorem sum_of_cubes_divisible (a : ℤ) : 
  let sum_cubes := (a - 1)^3 + a^3 + (a + 1)^3 in
  sum_cubes % (3 * a) = 0 ∧ sum_cubes % 9 = 0 :=
by
  sorry

end sum_of_cubes_divisible_l401_401151


namespace compute_radii_sum_l401_401097

def points_on_circle (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) * (dist C D) = (dist A C) * (dist B D)

theorem compute_radii_sum :
  ∃ (r1 r2 : ℝ), points_on_circle (0,0) (-1,-1) (5,2) (6,2) r1
               ∧ points_on_circle (0,0) (-1,-1) (34,14) (35,14) r2
               ∧ r1 > 0
               ∧ r2 > 0
               ∧ r1 < r2
               ∧ r1^2 + r2^2 = 1381 :=
by {
  sorry -- proof not required
}

end compute_radii_sum_l401_401097


namespace sin_approx_ax_tan_approx_ax_arcsin_approx_ax_arctan_approx_ax_sqrt_approx_half_x_l401_401139

-- Part 1
theorem sin_approx_ax (a : ℝ) : tendsto (fun x => (sin (a * x)) / (a * x)) (nhds 0) (nhds 1) :=
sorry

-- Part 2
theorem tan_approx_ax (a : ℝ) : tendsto (fun x => (tan (a * x)) / (a * x)) (nhds 0) (nhds 1) :=
sorry

-- Part 3
theorem arcsin_approx_ax (a : ℝ) : tendsto (fun x => (arcsin (a * x)) / (a * x)) (nhds 0) (nhds 1) :=
sorry

-- Part 4
theorem arctan_approx_ax (a : ℝ) : tendsto (fun x => (arctan (a * x)) / (a * x)) (nhds 0) (nhds 1) :=
sorry

-- Part 5
theorem sqrt_approx_half_x : tendsto (fun x => (sqrt (1 + x) - 1) / (x / 2)) (nhds 0) (nhds 1) :=
sorry

end sin_approx_ax_tan_approx_ax_arcsin_approx_ax_arctan_approx_ax_sqrt_approx_half_x_l401_401139


namespace teams_in_double_round_robin_l401_401440
-- Import the standard math library

-- Lean statement for the proof problem
theorem teams_in_double_round_robin (m n : ℤ) 
  (h : 9 * n^2 + 6 * n + 32 = m * (m - 1) / 2) : 
  m = 8 ∨ m = 32 :=
sorry

end teams_in_double_round_robin_l401_401440


namespace polynomial_g_l401_401546

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l401_401546


namespace num_distinct_lines_with_acute_angle_l401_401787

-- Definitions for the conditions
def is_acoustic_angle (a b : ℤ) := (a ≠ 0 ∧ b ≠ 0) ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)
def valid_set : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}

-- Definition for distinct coefficients picked from valid_set
def distinct_elements (a b c : ℤ) := a ∈ valid_set ∧ b ∈ valid_set ∧ c ∈ valid_set ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c

-- The theorem to prove
theorem num_distinct_lines_with_acute_angle : 
  ∀ (a b c : ℤ), distinct_elements a b c → is_acoustic_angle a b → (∃! lines, count lines = 43) :=
sorry

end num_distinct_lines_with_acute_angle_l401_401787


namespace sequence_term_2008_l401_401888

def sequence_term (n : Nat) : Nat :=
  let k := Nat.floor (Math.sqrt n)
  if n ≤ k * k + k then
    1 + 3 * (n - 1)
  else
    let first_term_of_next_group := 1 + 3 * k * k
    let offset := n - k * k - k
    first_term_of_next_group + 3 * offset

theorem sequence_term_2008 : sequence_term 2008 = 3124 :=
by
  sorry

end sequence_term_2008_l401_401888


namespace rectangular_coordinates_calc_l401_401241

def point_rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
let r := real.sqrt (x^2 + y^2),
    θ := real.acos (x/r) in
(r, θ)

def triple_angle_cos (cosθ : ℝ) : ℝ :=
4 * cosθ^3 - 3 * cosθ

def triple_angle_sin (sinθ : ℝ) : ℝ :=
3 * sinθ - 4 * sinθ^3

def new_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
let r3 := r^3,
    cos3θ := triple_angle_cos (real.cos θ),
    sin3θ := triple_angle_sin (real.sin θ),
    x := r3 * cos3θ,
    y := r3 * sin3θ in
(x, y)

theorem rectangular_coordinates_calc :
  new_rectangular_coordinates 13 (real.acos (12/13)) = (-2197, 31955) :=
by
  simpl # reduce redundancies and break computation
  sorry

end rectangular_coordinates_calc_l401_401241


namespace probability_Y_geq_2_l401_401489

noncomputable def binom (n k : ℕ) : ℚ := nat.choose n k

def bernoulli_pmf (n : ℕ) (p : ℚ) : ℕ → ℚ :=
λ k => binom n k * p^k * (1 - p)^(n - k)

def P_ge (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
bernoulli_pmf n p k + bernoulli_pmf n p (k+1)

theorem probability_Y_geq_2 :
  ∀ (X Y : ℕ → ℚ) (p : ℚ),
  X = bernoulli_pmf 2 p ∧
  Y = bernoulli_pmf 4 p ∧
  P_ge 2 p 1 = 5/9 →
  P_ge 4 p 2 = 11/27 :=
by
  intros X Y p h
  simp [bernoulli_pmf, binom, P_ge] at h
  sorry

end probability_Y_geq_2_l401_401489


namespace remainder_when_4_pow_2023_div_17_l401_401734

theorem remainder_when_4_pow_2023_div_17 :
  ∀ (x : ℕ), (x = 4) → x^2 ≡ 16 [MOD 17] → x^2023 ≡ 13 [MOD 17] := by
  intros x hx h
  sorry

end remainder_when_4_pow_2023_div_17_l401_401734


namespace marissa_initial_ribbon_l401_401126

theorem marissa_initial_ribbon (ribbon_per_box : ℝ) (number_of_boxes : ℝ) (ribbon_left : ℝ) : 
  (ribbon_per_box = 0.7) → (number_of_boxes = 5) → (ribbon_left = 1) → 
  (ribbon_per_box * number_of_boxes + ribbon_left = 4.5) :=
  by
    intros
    sorry

end marissa_initial_ribbon_l401_401126


namespace eccentricity_correct_asymptotes_correct_l401_401018

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 16) = 1

-- Definitions for elements of the problem
def a_sq : ℝ := 9
def b_sq : ℝ := 16
def c_sq : ℝ := a_sq + b_sq
def a : ℝ := real.sqrt a_sq
def b : ℝ := real.sqrt b_sq
def c : ℝ := real.sqrt c_sq

-- The hyperbola's eccentricity
def eccentricity : ℝ := c / a

-- Equations for the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = (b / a) * x
def asymptote2 (x y : ℝ) : Prop := y = -(b / a) * x

-- Statements we need to prove
theorem eccentricity_correct : eccentricity = 5 / 3 :=
by sorry

theorem asymptotes_correct (x y : ℝ) : asymptote1 x y ∨ asymptote2 x y → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by sorry

end eccentricity_correct_asymptotes_correct_l401_401018


namespace z_in_second_quadrant_l401_401941

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 + i^3) = i) : 
  is_second_quadrant z := by
  sorry

end z_in_second_quadrant_l401_401941


namespace like_terms_m_eq_2_l401_401049

theorem like_terms_m_eq_2 (m : ℕ) :
  (∀ (x y : ℝ), 3 * x^m * y^3 = 3 * x^2 * y^3) -> m = 2 :=
by
  intros _
  sorry

end like_terms_m_eq_2_l401_401049


namespace circumcircle_equals_excircle_radius_l401_401077

variables {A B C O I D : Type} [Incenter I ABC] (O : Circumcenter ABC) (AD : Altitude A BC)
          (I_lies_on_OD : LiesOnSegment I O D)

theorem circumcircle_equals_excircle_radius 
  (A B C : Point) (O : Point) (I : Point) (D : Point)
  [CircumcenterTriangle O A B C]
  [IncenterTriangle I A B C]
  [Altitude D A B C]
  (I_on_OD : LiesOn I O D) :
  CircumcenterRadius A B C = ExcircleRadius A B C BC :=
sorry

end circumcircle_equals_excircle_radius_l401_401077


namespace calculate_expression_l401_401691

theorem calculate_expression :
  (0.125: ℝ) ^ 3 * (-8) ^ 3 = -1 := 
by
  sorry

end calculate_expression_l401_401691


namespace true_proposition_is_A_l401_401381

-- Define the propositions
def l1 := ∀ (x y : ℝ), x - 2 * y + 3 = 0
def l2 := ∀ (x y : ℝ), 2 * x + y + 3 = 0
def p : Prop := ¬(l1 ∧ l2 ∧ ¬(∃ (x y : ℝ), x - 2 * y + 3 = 0 ∧ 2 * x + y + 3 = 0 ∧ (1 * 2 + (-2) * 1 ≠ 0)))
def q : Prop := ∃ x₀ : ℝ, (0 < x₀) ∧ (x₀ + 2 > Real.exp x₀)

-- The proof problem statement
theorem true_proposition_is_A : (¬p) ∧ q :=
by
  sorry

end true_proposition_is_A_l401_401381


namespace new_house_cost_l401_401188

theorem new_house_cost (purchase_price : ℕ) (increase_percent : ℕ) (loan_percent : ℕ) (amount_from_sale : ℕ) (new_house_percent : ℕ) :
  purchase_price = 100000 →
  increase_percent = 25 →
  loan_percent = 75 →
  amount_from_sale = 125000 →
  new_house_percent = 25 →
  let new_house_cost := amount_from_sale * 100 / new_house_percent in
  new_house_cost = 500000 :=
by
  intros
  let new_house_cost := amount_from_sale * 100 / new_house_percent
  sorry

end new_house_cost_l401_401188


namespace four_digit_numbers_with_property_l401_401180

def digit_boundaries (a b c d : ℕ) : Prop := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

def digit_sum_27 (a b c d : ℕ) : Prop := a + b + c + d = 27

def divisible_by_27 (n : ℕ) : Prop := n % 27 = 0

theorem four_digit_numbers_with_property :
  ∃ (count : ℕ), count = 75 ∧
  (count = set.size { (a, b, c, d) | 
      digit_boundaries a b c d ∧
      digit_sum_27 a b c d ∧ 
      divisible_by_27 (1000 * a + 100 * b + 10 * c + d) }) :=
begin
  sorry
end

end four_digit_numbers_with_property_l401_401180


namespace number_of_solutions_l401_401284

-- Define the two equations as conditions
def eq1 (x y : ℂ) : Prop := y = (x + 1) ^ 3
def eq2 (x y : ℂ) : Prop := x * y + y = 1

-- State that the system of equations has 2 real and 2 imaginary pairs
theorem number_of_solutions : 
  ∃ (xy_pairs : list (ℂ × ℂ)),
  (∀ xy ∈ xy_pairs, let ⟨x, y⟩ := xy in eq1 x y ∧ eq2 x y)
  ∧ (count_real xy_pairs = 2)
  ∧ (count_imaginary xy_pairs = 2) :=
sorry

end number_of_solutions_l401_401284


namespace race_distance_l401_401441

theorem race_distance 
  (D : ℝ) 
  (A_time : ℝ) (B_time : ℝ) 
  (A_beats_B_by : ℝ) 
  (A_time_eq : A_time = 36)
  (B_time_eq : B_time = 45)
  (A_beats_B_by_eq : A_beats_B_by = 24) :
  ((D / A_time) * B_time = D + A_beats_B_by) -> D = 24 := 
by 
  sorry

end race_distance_l401_401441


namespace music_stand_cost_proof_l401_401087

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end music_stand_cost_proof_l401_401087


namespace chocolate_pieces_l401_401225

theorem chocolate_pieces (total_pieces : ℕ) (michael_portion : ℕ) (paige_portion : ℕ) (mandy_portion : ℕ) 
  (h_total : total_pieces = 60) 
  (h_michael : michael_portion = total_pieces / 2) 
  (h_paige : paige_portion = (total_pieces - michael_portion) / 2) 
  (h_mandy : mandy_portion = total_pieces - (michael_portion + paige_portion)) : 
  mandy_portion = 15 :=
by
  sorry

end chocolate_pieces_l401_401225


namespace no_pentagon_division_l401_401525

theorem no_pentagon_division (k : ℕ)
  (h : 3 * k + 1 = 22) : ¬∃ (p : ℕ), p = 7 ∧ 
  ∃ (f : fin (3 * k + 1) → fin (3 * k + 1 → Prop)),
    (∀ i, ∃ l : list (fin (3 * k + 1)), 
      length l = 5 ∧ (i ∈ l) ∧ (∀ j ∈ l, f i j)) ∧
    (∀ i j, f i j → (i ≠ j → f j i → 
      ¬ (i = j ∨ i = i + 1 ∨ i = i - 1))) :=
begin
  sorry  -- Proof not required
end

end no_pentagon_division_l401_401525


namespace road_building_equation_l401_401976

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end road_building_equation_l401_401976


namespace pupils_in_class_l401_401212

theorem pupils_in_class (n : ℕ) (marks_wrongly_entered : ℕ = 73) (marks_correct : ℕ = 65)
  (average_increase : ℕ = 1/2) (total_increase : marks_wrongly_entered - marks_correct = 8) :
  n * 1/2 = 8 → n = 16 :=
by sorry

end pupils_in_class_l401_401212


namespace ratio_of_nonzero_reals_l401_401474

theorem ratio_of_nonzero_reals (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : (3 - 4 * complex.I) * (c + d * complex.I)).re = 0 : c / d = -4 / 3 := 
sorry

end ratio_of_nonzero_reals_l401_401474


namespace trajectory_curve_fixed_point_exists_l401_401748

variables (F P Q : ℝ × ℝ) (l : ℝ → Prop)
axiom point_F : F = (1, 0)
axiom line_l : l = λ x, x = -1
axiom moving_P : ∀ t : ℝ, ∃ x y : ℝ, P = (x, y)
axiom perp_Q : ∀ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, l Q.1 ∧ Q = (-1, P.2)
axiom vector_condition : ∀ P Q : ℝ × ℝ, (Q.1 - P.1, Q.2 - P.2) • (F.1 - Q.1, F.2 - Q.2) = (F.1 - P.1, F.2 - P.2) • (F.1 - Q.1, F.2 - Q.2)

theorem trajectory_curve (P : ℝ × ℝ) (Q : ℝ × ℝ) 
    (hQ: Q = (-1, P.2)) 
    (hvec : (Q.1 - P.1, Q.2 - P.2) • (F.1 - Q.1, F.2 - Q.2) = (F.1 - P.1, F.2 - P.2) • (F.1 - Q.1, F.2 - Q.2)) :
    P.2^2 = 4 * P.1 :=
sorry

axiom tangent_M_N (k m : ℝ) : ∃ M N : ℝ × ℝ, 
M = (m^2, 2*m) ∧ N = (-1, m - 1 / m)
axiom circle_MN : ∀ M N : ℝ × ℝ, 
(M.1 - 1)^2 + (M.2 - 0)^2 + (N.1 - 1)^2 + (N.2 - 0)^2 = (M.1 - N.1)^2 + (M.2 - N.2)^2 

theorem fixed_point_exists : 
    ∃ E : ℝ × ℝ, E = (1, 0) ∧ 
    ∀ {M N : ℝ × ℝ} (H : tangent_M_N M N), 
    circle_MN M N :=
sorry

end trajectory_curve_fixed_point_exists_l401_401748


namespace cube_root_neg_27_l401_401695

theorem cube_root_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by
  use -3
  split
  · norm_num
  · rfl

end cube_root_neg_27_l401_401695


namespace number_decomposition_l401_401585

theorem number_decomposition : 10101 = 10000 + 100 + 1 :=
by
  sorry

end number_decomposition_l401_401585


namespace joseph_speed_proof_l401_401462

-- The conditions are translated to Lean definitions
variables (t_J : ℝ) (v_K : ℝ) (t_K : ℝ) (d_diff : ℝ)

-- Conditions
def joseph_drives_time := t_J = 2.5
def kyle_speed := v_K = 62
def kyle_drives_time := t_K = 2
def joseph_distance_more := d_diff = 1

-- Derived variables from conditions
def kyle_distance := v_K * t_K
def joseph_distance := kyle_distance + d_diff
def joseph_speed := joseph_distance / t_J

-- The theorem to prove
theorem joseph_speed_proof (h1 : t_J = 2.5) (h2 : v_K = 62) (h3 : t_K = 2) (h4 : d_diff = 1) :
  joseph_speed t_J v_K t_K d_diff = 50 :=
by
  sorry

end joseph_speed_proof_l401_401462


namespace range_of_a_l401_401430

variable (a : ℝ)

def proposition (x : ℝ) : Prop :=
  x^2 + a * x + 9 ≥ 0

theorem range_of_a : (∃ x ∈ set.Icc (1 : ℝ) (2 : ℝ), ¬ proposition a x) ↔ a < -13/2 := sorry

end range_of_a_l401_401430


namespace largest_of_consecutive_even_integers_l401_401602

theorem largest_of_consecutive_even_integers (x : ℤ) (h : 25 * (x + 24) = 10000) : x + 48 = 424 :=
sorry

end largest_of_consecutive_even_integers_l401_401602


namespace reflect_across_x_axis_l401_401846

theorem reflect_across_x_axis (x y : ℝ) : (x, y) = (2, 3) → (x, -y) = (2, -3) := by
  intro h
  rw [h]
  exact rfl

end reflect_across_x_axis_l401_401846


namespace sqrt_domain_l401_401173

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end sqrt_domain_l401_401173


namespace solution_set_abs_inequality_l401_401592

theorem solution_set_abs_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_abs_inequality_l401_401592


namespace sphere_radius_volume_eq_surface_area_l401_401059

theorem sphere_radius_volume_eq_surface_area (r : ℝ) (h₁ : (4 / 3) * π * r^3 = 4 * π * r^2) : r = 3 :=
by
  sorry

end sphere_radius_volume_eq_surface_area_l401_401059


namespace ship_B_has_highest_rt_no_cars_l401_401505

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l401_401505


namespace sum_of_integers_satisfying_l401_401599

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l401_401599


namespace complex_im_part_thm_l401_401009

noncomputable def complex_im_part_proof : Prop :=
  let z := (1 + complex.I) / (1 - complex.I) + complex.I in
  z.im = 2

-- To assert the theorem
theorem complex_im_part_thm : complex_im_part_proof := by 
  sorry

end complex_im_part_thm_l401_401009


namespace maximum_value_of_function_l401_401952

noncomputable def f : ℝ → ℝ := λ x, x^4 - 4*x + 3

theorem maximum_value_of_function :
  ∃ x ∈ set.Icc (-2 : ℝ) 3, ∀ y ∈ set.Icc (-2 : ℝ) 3, f y ≤ f x ∧ f x = 72 :=
sorry

end maximum_value_of_function_l401_401952


namespace C_increases_as_n_increases_l401_401778

theorem C_increases_as_n_increases (e n R r : ℝ) (he : 0 < e) (hn : 0 < n) (hR : 0 < R) (hr : 0 < r) :
  0 < (2 * e * n * R + e * n^2 * r) / (R + n * r)^2 :=
by
  sorry

end C_increases_as_n_increases_l401_401778


namespace sum_a5_a6_a7_l401_401386

variable (a : ℕ → ℝ) (q : ℝ)

-- Assumptions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 1
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = 2

-- The theorem we want to prove
theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 16 := sorry

end sum_a5_a6_a7_l401_401386


namespace integer_pairs_count_l401_401267

/-- Theorem:
Prove that there are exactly 500 pairs of integers (m, n) such that 
1 ≤ m ≤ 1000
and 
3^n < 2^m < 2^{m+1} < 3^{n+1}.
-/
theorem integer_pairs_count : 
  (finset.card { (m, n) : ℕ × ℕ | 1 ≤ m ∧ m ≤ 1000 ∧ 3^n < 2^m ∧ 2^m < 2^(m+1) ∧ 2^(m+1) < 3^(n+1) }) = 500 :=
by
  sorry

end integer_pairs_count_l401_401267


namespace expected_area_of_triangle_XYZ_l401_401890

variables (A B C X Y Z : Type)

noncomputable def expected_area_of_triangle
  (AB_val BC_val AC_val : ℝ)
  (random_X_on_AB random_Y_on_BC random_Z_on_CA : ℕ) : ℝ :=
let s := (AB_val + BC_val + AC_val) / 2 in
let area_ABC := Real.sqrt (s * (s - AB_val) * (s - BC_val) * (s - AC_val)) in
area_ABC / 4

theorem expected_area_of_triangle_XYZ :
  expected_area_of_triangle 8 15 17 1 1 1 = 15 :=
by {
  sorry
}

end expected_area_of_triangle_XYZ_l401_401890


namespace angle_KNL_eq_angle_BCD_l401_401098

theorem angle_KNL_eq_angle_BCD 
    (A B C D K L N : Point)
    (h_cyclic : CyclicQuadrilateral A B C D)
    (h_angle_BAD_lt_90 : ∠BAD < 90)
    (h_K_on_ray_AB : OnRay K A B)
    (h_L_on_ray_AD : OnRay L A D)
    (h_KA_eq_KD : KA = KD)
    (h_LA_eq_LB : LA = LB)
    (h_N_mid_AC : Midpoint N A C)
    (h_angle_BNC_eq_angle_DNC : ∠BNC = ∠DNC) :
    ∠KNL = ∠BCD := 
sorry

end angle_KNL_eq_angle_BCD_l401_401098


namespace find_m_from_distance_l401_401374

theorem find_m_from_distance (m : ℝ) (h : abs (3 + √3 * m - 4) / 2 = 1) : m = √3 ∨ m = -√3 / 3 :=
by
  sorry

end find_m_from_distance_l401_401374


namespace product_of_numbers_l401_401354

theorem product_of_numbers :
  ∃ (a b c : ℚ), a + b + c = 30 ∧
                 a = 2 * (b + c) ∧
                 b = 5 * c ∧
                 a + c = 22 ∧
                 a * b * c = 2500 / 9 :=
by
  sorry

end product_of_numbers_l401_401354


namespace cashier_correct_amount_l401_401651

theorem cashier_correct_amount (y : ℕ) : 
  let quarters_to_dimes_error := 15 * y,
      half_dollars_to_nickels_error := 45 * y
  in quarters_to_dimes_error + half_dollars_to_nickels_error = 60 * y :=
by
  sorry

end cashier_correct_amount_l401_401651


namespace find_forty_percent_of_N_l401_401907

-- Define the conditions
def condition (N : ℚ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14

-- Define the theorem to prove
theorem find_forty_percent_of_N (N : ℚ) (h : condition N) : 0.4 * N = 168 := by
  sorry

end find_forty_percent_of_N_l401_401907


namespace cos_to_sin_shift_l401_401186

theorem cos_to_sin_shift :
  ∀ x : ℝ, sin (2 * (x - (5 * π / 12)) + (π / 2)) = sin (2 * x - (π / 3)) :=
by
  intro x
  have h1 : cos (2 * x) = sin (2 * x + π / 2), by sorry
  have h2 : sin (2 * x + π / 2) = sin (2 * (x - (5 * π / 12)) + π / 2), by sorry
  exact congr_arg sin (eq.trans (congr_arg (λ t, 2 * (x - t) + π / 2) sorry) sorry)

end cos_to_sin_shift_l401_401186


namespace f_value_at_1_2018_l401_401656

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_equiv_1 : ∀ x : ℝ, f(x) + f(1 - x) = 1
axiom f_equiv_2 : ∀ x : ℝ, x ≥ 0 → f(x / 3) = (1 / 2) * f(x)
axiom f_monotonic : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f(x1) ≤ f(x2)

theorem f_value_at_1_2018 : f(1 / 2018) = 1 / 128 := sorry

end f_value_at_1_2018_l401_401656


namespace crocus_bulbs_count_l401_401630

theorem crocus_bulbs_count (C D : ℕ) 
  (h1 : C + D = 55) 
  (h2 : 0.35 * (C : ℝ) + 0.65 * (D : ℝ) = 29.15) :
  C = 22 :=
sorry

end crocus_bulbs_count_l401_401630


namespace find_min_value_l401_401481

def min_value_expr (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 0 < k) : ℝ :=
  (6 * z) / (x + 2 * y + k) + (6 * x) / (2 * z + y + k) + (3 * y) / (x + z + k)

theorem find_min_value (x y z k : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 0 < k) :
  min_value_expr x y z k hx hy hz hk = 4.5 :=
by
  sorry

end find_min_value_l401_401481


namespace distinct_sequences_count_l401_401042

-- Define the original set of letters
def letters : List Char := ['D', 'Y', 'N', 'A', 'M', 'I', 'C']

-- Define the fixed first letter
def first_letter : Char := 'D'

-- Define the possible letters for the last position (excluding 'C')
def last_letters : List Char := ['Y', 'N', 'A', 'M', 'I']

-- Define the problem statement
theorem distinct_sequences_count :
  ∃ (count : ℕ), count = 60 ∧ 
    (∀ (seq : List Char), 
      seq.length = 4 ∧ 
      seq.head = some first_letter ∧ 
      seq.last ≠ some 'C' → 
      count = (last_letters.length * 4 * 3)) :=
sorry

end distinct_sequences_count_l401_401042


namespace a_profit_share_l401_401632

/-- Definitions for the shares of capital -/
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)
def total_profit : ℚ := 2415

/-- The profit share for A, given the conditions on capital subscriptions -/
theorem a_profit_share : a_share * total_profit = 805 := by
  sorry

end a_profit_share_l401_401632


namespace skating_probability_given_skiing_l401_401834

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l401_401834


namespace probability_ge_2_l401_401022

noncomputable def normal_distribution : Type := sorry

variables {ξ : normal_distribution}
variables (σ : ℝ)

axiom given_conditions : (ξ ~ normal_distribution(0, σ^2)) ∧ (P(-2 ≤ ξ ∧ ξ ≤ 0) = 0.2)

theorem probability_ge_2 : P(ξ ≥ 2) = 0.3 := by
  sorry

end probability_ge_2_l401_401022


namespace satisfy_equation_l401_401996

noncomputable def find_ab (p q : ℝ) : set ℝ :=
  { a | a = (real.cbrt (-q/2 + real.sqrt (q^2 / 4 + p^3 / 27))) } ∪
  { b | b = (real.cbrt (-q/2 - real.sqrt (q^2 / 4 + p^3 / 27))) }

theorem satisfy_equation (p q a b : ℝ) :
  (x^3 + p * x + q = x^3 - a^3 - b^3 - 3 * a * b * x) → 
  (a ∈ find_ab p q ∧ b ∈ find_ab p q) :=
by
  sorry

end satisfy_equation_l401_401996


namespace find_n_cosine_l401_401315

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401315


namespace nondecreasing_f_l401_401463

open Finset

noncomputable def f {α : Type} (S : Finset α) (𝓕 : Finset (Finset α))
  (h : ∀ {A B : Finset α}, A ∈ 𝓕 → A ⊆ B → B ⊆ S → B ∈ 𝓕) (t : ℝ) : ℝ :=
  ∑ A in 𝓕, t^(A.card) * (1-t)^(S.card - A.card)

theorem nondecreasing_f {α : Type} {S : Finset α} {𝓕 : Finset (Finset α)}
    (h : ∀ {A B : Finset α}, A ∈ 𝓕 → A ⊆ B → B ⊆ S → B ∈ 𝓕) :
  ∀ p q : ℝ, 0 ≤ p → p ≤ q → q ≤ 1 → f S 𝓕 h p ≤ f S 𝓕 h q :=
by
  sorry

end nondecreasing_f_l401_401463


namespace skating_probability_given_skiing_l401_401835

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l401_401835


namespace problem1_value_problem2_value_l401_401641

-- Problem 1
theorem problem1_value (x : ℚ) (h : x = 1/2) : 
  (2 * x^2 - 5 * x + x^2 + 4 * x - 3 * x^2 - 2) = - 5/2 :=
by {
  rw h,
  sorry
}

-- Problem 2
theorem problem2_value (x y : ℚ) (hx : x = -2) (hy : y = 2 / 3) :
  (1/2 * x - 2 * (x - 1/3 * y^2) + (-3/2 * x + 1/3 * y^2)) = 6 + 4/9 :=
by {
  rw [hx, hy],
  sorry
}

end problem1_value_problem2_value_l401_401641


namespace cube_root_of_neg_27_l401_401697

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end cube_root_of_neg_27_l401_401697


namespace original_fraction_eq_two_thirds_l401_401956

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l401_401956


namespace leak_rate_l401_401038

-- Definitions based on conditions
def initialWater : ℕ := 10   -- 10 cups
def finalWater : ℕ := 2      -- 2 cups
def firstThreeMilesWater : ℕ := 3 * 1    -- 1 cup per mile for first 3 miles
def lastMileWater : ℕ := 3               -- 3 cups during the last mile
def hikeDuration : ℕ := 2    -- 2 hours

-- Proving the leak rate
theorem leak_rate (drunkWater : ℕ) (leakedWater : ℕ) (leakRate : ℕ) :
  drunkWater = firstThreeMilesWater + lastMileWater ∧ 
  (initialWater - finalWater) = (drunkWater + leakedWater) ∧
  hikeDuration = 2 ∧ 
  leakRate = leakedWater / hikeDuration → leakRate = 1 :=
by
  intros h
  sorry

end leak_rate_l401_401038


namespace sequence_periodicity_l401_401023

-- Define the sequence based on given conditions
noncomputable def a : ℕ → ℚ
| 1 := 1
| 2 := 2
| n+3 := a (n+2) / a (n+1)

-- Theorem stating the required proof problem
theorem sequence_periodicity : a 2017 = 1 := by
  sorry

end sequence_periodicity_l401_401023


namespace unique_increasing_sequence_length_l401_401183

theorem unique_increasing_sequence_length :
  (∃ (seq : List ℕ), seq.Nodup ∧ seq.sorted (<) ∧ (2 ^ 225 + 1) / (2 ^ 15 + 1) = seq.foldr (λ b acc, 2 ^ b + acc) 0) →
  ∃ (m : ℕ), m = 241 :=
by 
  intro h
  use 241
  sorry

end unique_increasing_sequence_length_l401_401183


namespace midpoint_sum_l401_401571

-- Defining the coordinates of the endpoints.
def x1 : ℝ := 3
def y1 : ℝ := 4
def x2 : ℝ := 9
def y2 : ℝ := 18

-- Defining the midpoint coordinates.
def midpoint_x : ℝ := (x1 + x2) / 2
def midpoint_y : ℝ := (y1 + y2) / 2

-- Defining the sum of the coordinates of the midpoint.
def sum_of_midpoint_coordinates : ℝ := midpoint_x + midpoint_y

-- The statement we need to prove.
theorem midpoint_sum : sum_of_midpoint_coordinates = 17 := by
  -- Proof goes here
  sorry

end midpoint_sum_l401_401571


namespace functional_equation_solution_l401_401725

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f(2 * f(x) + f(y)) = 2 * x + f(y)) → (f(x) = x) :=
by
  sorry

end functional_equation_solution_l401_401725


namespace constant_term_expansion_l401_401078

theorem constant_term_expansion : 
  let x_term := (x : ℝ)
  let expansion := (x_term + 2 / x_term^2) ^ 6 
  ∃ T : ℝ, T = 60 ∧ 
  ∀ r : ℕ, (choose 6 r) * (2^r) * (x_term^(6 - 3 * r)) = T → 6 - 3 * r = 0 :=
sorry

end constant_term_expansion_l401_401078


namespace alternating_sum_1_to_101_l401_401707

open Nat

def alternating_sum_101 : ℤ :=
  (List.range 102).map (λ n, if n % 2 = 0 then -n else n).sum

theorem alternating_sum_1_to_101 : alternating_sum_101 = 51 := by
  sorry

end alternating_sum_1_to_101_l401_401707


namespace both_girls_given_at_least_one_girl_l401_401231

open Probability

theorem both_girls_given_at_least_one_girl :
  let events := ["GG", "GB", "BG", "BB"]
  in P {x | x ∈ {"GG"}} = 1 / 3 :=
by
  have h_conditions := {"GG", "GB", "BG"}
  have h_probability_set := P (event h_conditions)
  have h_single_event := P {x | x ∈ {"GG"}}
  sorry

end both_girls_given_at_least_one_girl_l401_401231


namespace cos_eq_43_l401_401320

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401320


namespace percentage_discount_offered_is_approximately_4_l401_401246

-- Definitions
def CP := 100.0
def SP_with_discount := 138.0
def SP_no_discount := 143.75
def Discount_amount := SP_no_discount - SP_with_discount
def Percentage_discount := (Discount_amount / SP_no_discount) * 100

-- Problem: Prove that the percentage of discount offered is approximately 4%.
theorem percentage_discount_offered_is_approximately_4 :
  Percentage_discount ≈ 4 := 
by
  sorry

end percentage_discount_offered_is_approximately_4_l401_401246


namespace triangle_solution_l401_401861

theorem triangle_solution {a b : ℝ} {A : ℝ}
  (h_a : a = sqrt 3)
  (h_b : b = 3)
  (h_A : A = π / 6) :
  1.5 < a ∧ a < b →  -- equivalent to b sin A < a < b
  (∃ (k : ℚ), k = 2) := 
sorry

end triangle_solution_l401_401861


namespace evaluate_polynomial_at_neg_two_l401_401722

theorem evaluate_polynomial_at_neg_two : 
  (let x := -2 in x^3 - x^2 + x - 1) = -15 :=
by 
  sorry

end evaluate_polynomial_at_neg_two_l401_401722


namespace sum_is_five_or_negative_five_l401_401393

theorem sum_is_five_or_negative_five (a b c d : ℤ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b * c * d = 14) : 
  (a + b + c + d = 5) ∨ (a + b + c + d = -5) :=
by
  sorry

end sum_is_five_or_negative_five_l401_401393


namespace cosine_periodicity_l401_401335

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401335


namespace corvette_trip_time_percentage_increase_l401_401899

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l401_401899


namespace smallest_n_proof_l401_401355

theorem smallest_n_proof : ∃ (n : ℕ) (n_min : n > 0), (∀ (x : Fin n → ℝ), 
  (∑ i, x i = 100) ∧ (∑ i, (x i) ^ 3 = 10000) → n = 464) :=
by sorry

end smallest_n_proof_l401_401355


namespace count_distinct_ways_to_distribute_balls_l401_401418

theorem count_distinct_ways_to_distribute_balls :
  let balls := 6
  let boxes := 4
  let conditions := {balls, boxes}
  ((finset.univ.powerset.card : nat) (finset.range (balls + boxes - 1)).card / ((boxes - 1).choose 2).card = 9) :=
begin
  sorry
end

end count_distinct_ways_to_distribute_balls_l401_401418


namespace third_week_cases_l401_401497

-- Define the conditions as Lean definitions
def first_week_cases : ℕ := 5000
def second_week_cases : ℕ := first_week_cases / 2
def total_cases_after_three_weeks : ℕ := 9500

-- The statement to be proven
theorem third_week_cases :
  first_week_cases + second_week_cases + 2000 = total_cases_after_three_weeks :=
by
  sorry

end third_week_cases_l401_401497


namespace cube_root_neg_27_l401_401696

theorem cube_root_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by
  use -3
  split
  · norm_num
  · rfl

end cube_root_neg_27_l401_401696


namespace ratio_population_X_to_Z_l401_401216

-- Given definitions
def population_of_Z : ℕ := sorry
def population_of_Y : ℕ := 2 * population_of_Z
def population_of_X : ℕ := 5 * population_of_Y

-- Theorem to prove
theorem ratio_population_X_to_Z : population_of_X / population_of_Z = 10 :=
by
  sorry

end ratio_population_X_to_Z_l401_401216


namespace original_fraction_eq_two_thirds_l401_401957

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l401_401957


namespace sequence_contains_infinitely_many_perfect_squares_l401_401240

open Nat

def a : ℕ → ℕ
| 0       => 1
| (n + 1) => a n + (Nat.floor (Real.sqrt (a n)))

theorem sequence_contains_infinitely_many_perfect_squares :
  ∃ (m : ℕ), ∃ (f : ℕ → ℕ), Injective f ∧ ∀ n, ∃ k, f n = k^2 := sorry

end sequence_contains_infinitely_many_perfect_squares_l401_401240


namespace cos_triangle_inequality_l401_401143

theorem cos_triangle_inequality (α β γ : ℝ) (h_sum : α + β + γ = Real.pi) 
    (h_α : 0 < α) (h_β : 0 < β) (h_γ : 0 < γ) (h_α_lt : α < Real.pi) (h_β_lt : β < Real.pi) (h_γ_lt : γ < Real.pi) : 
    (Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α) ≤ 3 / 4 :=
by
  sorry

end cos_triangle_inequality_l401_401143


namespace eval_f_pi_div_2_max_value_f_min_positive_period_f_l401_401887

def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - Real.cos (2 * x) + 1

theorem eval_f_pi_div_2 : f (Real.pi / 2) = 2 := by sorry

theorem max_value_f : ∃ x, f(x) = Real.sqrt 2 + 1 := by sorry

theorem min_positive_period_f : ∀ x, f (x + Real.pi) = f x := by sorry

end eval_f_pi_div_2_max_value_f_min_positive_period_f_l401_401887


namespace find_abc_l401_401820

open Real

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h1 : a * (b + c) = 154)
  (h2 : b * (c + a) = 164) 
  (h3 : c * (a + b) = 172) : 
  (a * b * c = Real.sqrt 538083) := 
by 
  sorry

end find_abc_l401_401820


namespace scientific_notation_35100_l401_401855

theorem scientific_notation_35100 : 35100 = 3.51 * 10^4 :=
by
  sorry

end scientific_notation_35100_l401_401855


namespace possible_polynomials_l401_401553

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l401_401553


namespace number_of_quadrilaterals_meeting_conditions_l401_401359

-- Define the quadrilateral types
inductive Quadrilateral
| Kite_not_rhombus
| Rectangle
| Scalene_trapezoid
| Rhombus
| General_quadrilateral

-- Define properties of the quadrilaterals
def has_circumcenter (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Kite_not_rhombus => false
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Scalene_trapezoid => false
  | Quadrilateral.Rhombus => true
  | Quadrilateral.General_quadrilateral => false

def symmetric_about_diagonal (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Kite_not_rhombus => false
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Scalene_trapezoid => false
  | Quadrilateral.Rhombus => true
  | Quadrilateral.General_quadrilateral => false

-- Define the main theorem
theorem number_of_quadrilaterals_meeting_conditions : 
  ∃ (n : ℕ), n = 2 ∧ 
    (n = (List.length (List.filter (λ q, (has_circumcenter q) ∧ (symmetric_about_diagonal q)) 
    [Quadrilateral.Kite_not_rhombus, 
     Quadrilateral.Rectangle, 
     Quadrilateral.Scalene_trapezoid, 
     Quadrilateral.Rhombus, 
     Quadrilateral.General_quadrilateral]))) := by
  sorry

end number_of_quadrilaterals_meeting_conditions_l401_401359


namespace find_principal_l401_401669

theorem find_principal (R P : ℝ) (h₁ : (P * R * 10) / 100 = P * R * 0.1)
  (h₂ : (P * (R + 3) * 10) / 100 = P * (R + 3) * 0.1)
  (h₃ : P * 0.1 * (R + 3) - P * 0.1 * R = 300) : 
  P = 1000 := 
sorry

end find_principal_l401_401669


namespace units_digit_product_l401_401692

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_product (a b c : ℕ) :
  units_digit a = 7 → units_digit b = 3 → units_digit c = 9 →
  units_digit ((a * b) * c) = 9 :=
by
  intros h1 h2 h3
  sorry

end units_digit_product_l401_401692


namespace find_x_l401_401204

def list := [9, 3, 6, 3, 7, 3]
def x := 26

def mean (lst : List ℕ) (x : ℕ) : ℚ :=
  (list.foldr (λ a b => a + b) 0 lst + x) / (lst.length + 1)

def mode (lst : List ℕ) : ℕ :=
  3 -- given condition mode is 3

def median (lst : List ℕ) (x : ℕ) : ℕ :=
  let sorted_lst := list.insertionSort lst
  if x <= 3 then 3
  else if x < 6 then x
  else 6

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b

theorem find_x :
  let mean_value := mean list x
  let median_value := median list x
  ∃ x : ℕ, is_arithmetic_progression (mode list) (median_value) (mean_value) :=
sorry

end find_x_l401_401204


namespace lioness_age_l401_401935

theorem lioness_age (H L : ℕ) 
  (h1 : L = 2 * H) 
  (h2 : (H / 2 + 5) + (L / 2 + 5) = 19) : 
  L = 12 :=
sorry

end lioness_age_l401_401935


namespace monotonic_decreasing_interval_l401_401584

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem monotonic_decreasing_interval : ∀ x, (x ∈ Iic 1) ↔ ∀ y, y ∈ Iic 1 → f x ≥ f y := begin
  sorry
end

end monotonic_decreasing_interval_l401_401584


namespace find_m_l401_401004

def vector := ℝ × ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_m (m : ℝ) : 
  let d : vector := (1, -2, 0)
  let n : vector := (m, 3, 6)
  (dot_product n d = 0) → m = 6 :=
by
  sorry

end find_m_l401_401004


namespace sum_of_integers_square_greater_272_l401_401598

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l401_401598


namespace inverse_function_domain_l401_401121

noncomputable def f (x : ℝ) : ℝ := -3 + Real.log (x - 1) / Real.log 2

theorem inverse_function_domain :
  ∀ x : ℝ, x ≥ 5 → ∃ y : ℝ, f x = y ∧ y ≥ -1 :=
by
  intro x hx
  use f x
  sorry

end inverse_function_domain_l401_401121


namespace sequence_term_position_l401_401432

theorem sequence_term_position :
  let a_n := (λ n, Real.sqrt (2 + (n - 1) * 3)) in
  a_n 7 = 2 * Real.sqrt 5 :=
by
  sorry

end sequence_term_position_l401_401432


namespace remaining_balance_is_correct_l401_401424

def total_price (deposit amount sales_tax_rate discount_rate service_charge P : ℝ) :=
  let sales_tax := sales_tax_rate * P
  let price_after_tax := P + sales_tax
  let discount := discount_rate * price_after_tax
  let price_after_discount := price_after_tax - discount
  let total_price := price_after_discount + service_charge
  total_price

theorem remaining_balance_is_correct (deposit : ℝ) (amount_paid : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ)
  (P : ℝ) : deposit = 0.10 * P →
         amount_paid = 110 →
         sales_tax_rate = 0.15 →
         discount_rate = 0.05 →
         service_charge = 50 →
         total_price deposit amount_paid sales_tax_rate discount_rate service_charge P - amount_paid = 1141.75 :=
by
  sorry

end remaining_balance_is_correct_l401_401424


namespace number_of_correct_statements_is_2_l401_401714

def otimes (a b : ℝ) : ℝ := a * (1 - b)

theorem number_of_correct_statements_is_2 :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ a b : ℝ, otimes a b = otimes b a) ∧
  (∀ a : ℝ, otimes 5 a + otimes 6 a = otimes 11 a) ∧
  ¬ (∀ b : ℝ, otimes 3 b = 3 → b = 1) →
  2 = 2 :=
by
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_more_rest,
  cases h_more_rest with h3 h4,
  exact rfl,
  sorry

end number_of_correct_statements_is_2_l401_401714


namespace sequence_sum_l401_401490

def arithmetic_seq (a₀ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₀ + n * d

def geometric_seq (b₀ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₀ * r^(n)

theorem sequence_sum :
  let a : ℕ → ℕ := arithmetic_seq 3 1
  let b : ℕ → ℕ := geometric_seq 1 2
  b (a 0) + b (a 1) + b (a 2) + b (a 3) = 60 :=
  by
    let a : ℕ → ℕ := arithmetic_seq 3 1
    let b : ℕ → ℕ := geometric_seq 1 2
    have h₀ : a 0 = 3 := by rfl
    have h₁ : a 1 = 4 := by rfl
    have h₂ : a 2 = 5 := by rfl
    have h₃ : a 3 = 6 := by rfl
    have hsum : b 3 + b 4 + b 5 + b 6 = 60 := by sorry
    exact hsum

end sequence_sum_l401_401490


namespace oil_level_drop_l401_401210

theorem oil_level_drop (r1 h1 r2 h2 : ℝ) (V_truck V_stationary_initial : ℝ) :
  r1 = 100 ∧ h1 = 25 ∧ r2 = 8 ∧ h2 = 10 ∧ 
  V_truck = real.pi * r2^2 * h2 →
  V_stationary_initial = real.pi * r1^2 * h1 →
  h1 - (V_truck / (real.pi * r1^2)) = 25 - 0.064 :=
by
  sorry

end oil_level_drop_l401_401210


namespace dihedral_angle_problem_l401_401030

noncomputable def sine_of_dihedral_angle (D A B C : ℝ₃) (p₁ p₂ : Plane ℝ₃) : ℝ := sorry

theorem dihedral_angle_problem 
  (D A B C : ℝ₃)
  (h1 : ∠ A C B = 90)
  (h2 : ∠ A B D = 90)
  (h3 : distance A C = distance B C)
  (h4 : ∠ B A D = 30)
  (h5 : ∃ E : ℝ₃, projection C (Plane.mk A B D) = ⟨E, sorry  ⟩ ∧ lies_on E (Line.mk A D)) :
  sine_of_dihedral_angle D A B C (Plane.mk C A B) (Plane.mk A B D) = √6/3 :=
sorry

end dihedral_angle_problem_l401_401030


namespace g_inv_computation_l401_401965

def g : ℕ → ℕ
| 1 := 4
| 2 := 9
| 3 := 11
| 5 := 3
| 7 := 6
| 12 := 2
| _ := 0  -- We need to cover all cases, but for this problem, we care only specific values.

lemma g_inv_exists (y : ℕ) (hy : y ∈ {2, 3, 11}) : ∃ x, g x = y :=
by {
  cases y,
  { use 12, exact rfl },
  { use 5, exact rfl },
  { use 3, exact rfl },
  sorry
}

noncomputable def g_inv (y : ℕ) : ℕ :=
if hy : y ∈ {2, 3, 11} then classical.some (g_inv_exists y hy) else 0

-- Now the proof statement
theorem g_inv_computation : 
  g_inv ((g_inv 2 + g_inv 11) / g_inv 3) = 5 :=
by {
  -- Expand each individual inverse
  have h1 : g_inv 2 = 12, by { unfold g_inv, rw if_pos, exact classical.some_spec (g_inv_exists 2 (by simp)), },
  have h2 : g_inv 11 = 3, by { unfold g_inv, rw if_pos, exact classical.some_spec (g_inv_exists 11 (by simp)), },
  have h3 : g_inv 3 = 5, by { unfold g_inv, rw if_pos, exact classical.some_spec (g_inv_exists 3 (by simp)), },
  
  -- Calculate the inner value
  have h_calc : (g_inv 2 + g_inv 11) / g_inv 3 = 3, by {
    rw [h1, h2, h3],
    norm_num,
  },
  
  -- Expand and apply the outer inverse
  unfold g_inv,
  rw if_pos,
  exact classical.some_spec (g_inv_exists 3 (by simp)),
  sorry
}

end g_inv_computation_l401_401965


namespace count_valid_tuples_l401_401465

variable {b_0 b_1 b_2 b_3 : ℕ}

theorem count_valid_tuples : 
  (∃ b_0 b_1 b_2 b_3 : ℕ, 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧ 
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧ 
    0 ≤ b_3 ∧ b_3 ≤ 99 ∧ 
    5040 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0) ∧ 
    ∃ (M : ℕ), 
    M = 504 :=
sorry

end count_valid_tuples_l401_401465


namespace point_in_region_l401_401501

theorem point_in_region (x y : ℝ) (h : x * y ≥ 0) : (real.sqrt (x * y) ≥ x - 2 * y) ↔ 
  ((x ≥ 0 ∧ y ≥ 0 ∧ y ≥ x / 2) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ y ≤ x / 2)) :=
by
  sorry

end point_in_region_l401_401501


namespace polynomial_form_l401_401726

theorem polynomial_form (P : ℝ → ℝ) (hP: ∀ a : ℝ, P a ∈ ℤ → a ∈ ℤ) : 
    ∃ (p q : ℤ) (hp : p ≠ 0), ∀ x : ℝ, P x = (x + q) / p := 
sorry

end polynomial_form_l401_401726


namespace lattice_point_probability_l401_401273

theorem lattice_point_probability (d : ℝ) :
  (π * d^2 = 1/3) → (d ≈ 0.3) :=
by
  sorry

end lattice_point_probability_l401_401273


namespace value_of_x_squared_add_reciprocal_squared_l401_401800

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l401_401800


namespace cos_alpha_add_pi_over_4_l401_401772

theorem cos_alpha_add_pi_over_4 (x y r : ℝ) (α : ℝ) (h1 : P = (3, -4)) (h2 : r = Real.sqrt (x^2 + y^2)) (h3 : x / r = Real.cos α) (h4 : y / r = Real.sin α) :
  Real.cos (α + Real.pi / 4) = (7 * Real.sqrt 2) / 10 := by
  sorry

end cos_alpha_add_pi_over_4_l401_401772


namespace arithmetic_sequence_solution_l401_401940

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) 
(h1 : d ≠ 0) 
(h2 : a 1 = 2) 
(h3 : a 1 * a 4 = (a 2) ^ 2) :
∀ n, a n = 2 * n :=
by 
  sorry

end arithmetic_sequence_solution_l401_401940


namespace bird_seed_problem_l401_401260

theorem bird_seed_problem : 
  ∀ (cups_per_10parakeets_5days : ℕ) (parakeets_10 : ℕ) (days_5 : ℕ), 
  cups_per_10parakeets_5days = 30 → parakeets_10 = 10 → days_5 = 5 →
  ∃ (cups_needed : ℕ), 
      cups_needed = 60 ∧ 
      cups_needed = (cups_per_10parakeets_5days * (20 / parakeets_10)) := 
by 
  intros cups_per_10parakeets_5days parakeets_10 days_5 h1 h2 h3
  use 60
  split
  exact rfl
  sorry

end bird_seed_problem_l401_401260


namespace find_a_l401_401411

def setA : Set ℤ := {-1, 0, 1}

def setB (a : ℤ) : Set ℤ := {a, a ^ 2}

theorem find_a (a : ℤ) (h : setA ∪ setB a = setA) : a = -1 :=
sorry

end find_a_l401_401411


namespace b_n_formula_l401_401024

-- Define the sequences a_n and b_n according to the conditions
noncomputable def a_n (n : ℕ) : ℝ := 1 / ((n + 1) ^ 2)

noncomputable def b_n (n : ℕ) : ℝ :=
  ∏ i in finset.range (n + 1), (1 - a_n i)

theorem b_n_formula (n : ℕ) : b_n n = (n + 2) / (2 * n + 2) :=
by
  sorry -- Proof goes here

end b_n_formula_l401_401024


namespace smallest_n_factorial_3300_l401_401200

theorem smallest_n_factorial_3300 :
  ∃ (n : ℕ), n > 0 ∧ (3300 ∣ nat.factorial n) ∧ 
  ∀ m, m > 0 ∧ (m < n) → ¬ (3300 ∣ nat.factorial m) := 
begin
  sorry
end

end smallest_n_factorial_3300_l401_401200


namespace matrix_product_sequence_l401_401297

open Matrix

def mat (x : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, x], ![0, 1]]

theorem matrix_product_sequence :
  (List.prod (List.map mat (List.range (50+1)))).sum (Fin.mk 0 (by simp)) (Fin.mk 1 (by simp)) = 2550 := by
  sorry

end matrix_product_sequence_l401_401297


namespace function_identity_l401_401750

theorem function_identity (f : ℕ → ℝ) (α β : ℝ) (h1 : α ≠ β) 
  (h2 : f 1 = (α^2 - β^2) / (α - β)) 
  (h3 : f 2 = (α^3 - β^3) / (α - β)) 
  (h4 : ∀ n, f (n + 2) = (α + β) * f (n + 1) - α * β * f n) : 
  ∀ n, f n = (α^(n+1) - β^(n+1)) / (α - β) :=
by
  sorry

end function_identity_l401_401750


namespace question_inequality_l401_401472

theorem question_inequality
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (cond : a + b ≤ 4) :
  (1 / a + 1 / b) ≥ 1 := 
sorry

end question_inequality_l401_401472


namespace find_range_a_l401_401369

def setA (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def setB (x a : ℝ) : Prop := |x - a| < 5
def real_line (x : ℝ) : Prop := True

theorem find_range_a (a : ℝ) :
  (∀ x, setA x ∨ setB x a) ↔ (-3:ℝ) ≤ a ∧ a ≤ 1 := by
sorry

end find_range_a_l401_401369


namespace inequality_proof_inequality_equality_conditions_l401_401220

theorem inequality_proof
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  (x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 ≤ (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2)) :=
sorry

theorem inequality_equality_conditions
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 = (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2))
  ↔ (x1 = x2 ∧ y1 = y2 ∧ z1 = z2)) :=
sorry

end inequality_proof_inequality_equality_conditions_l401_401220


namespace tangent_line_at_Q_and_through_P_l401_401401

def curve (x : ℝ) : ℝ := 2 * x^2 - x^3
def pointP : (ℝ × ℝ) := (0, -4)
def tangent_line (l : ℝ → ℝ) : Prop := 
  ∃ Q : ℝ × ℝ, 
    (Q.1 = -1 ∧ abs (l Q.1 - Q.2) = 0) ∧
    (∃ m : ℝ, m * (Q.1 - 0) = Q.2 + 4)

theorem tangent_line_at_Q_and_through_P : 
  curve (-1) = 2 * (-1)^2 - (-1)^3 ∧
  tangent_line (fun x => -7 * x - 4) := 
by
  have Q := (-1, curve (-1))
  have h1 : curve (Q.1) = Q.2 := by simp [curve, Q]
  have h2 : pointP.2 = -7 * pointP.1 + Q.2 := by simp [pointP, Q]
  exact ⟨h1, ⟨Q, rfl, ⟨-7, h2⟩⟩⟩

end tangent_line_at_Q_and_through_P_l401_401401


namespace quadratic_roots_satisfy_condition_l401_401769
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l401_401769


namespace find_n_l401_401334

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401334


namespace g_max_value_l401_401280

def g (n : ℕ) : ℕ :=
if n < 15 then n + 15 else g (n - 7)

theorem g_max_value : ∃ N : ℕ, ∀ n : ℕ, g n ≤ N ∧ N = 29 := 
by 
  sorry

end g_max_value_l401_401280


namespace sasha_quarters_l401_401148

theorem sasha_quarters (h₁ : 2.10 = 0.35 * q) : q = 6 := 
sorry

end sasha_quarters_l401_401148


namespace fourth_student_guess_l401_401561

theorem fourth_student_guess :
    let guess1 := 100
    let guess2 := 8 * guess1
    let guess3 := guess2 - 200
    let avg := (guess1 + guess2 + guess3) / 3
    let guess4 := avg + 25
    guess4 = 525 := by
    intros guess1 guess2 guess3 avg guess4
    have h1 : guess1 = 100 := rfl
    have h2 : guess2 = 8 * guess1 := rfl
    have h3 : guess3 = guess2 - 200 := rfl
    have h4 : avg = (guess1 + guess2 + guess3) / 3 := rfl
    have h5 : guess4 = avg + 25 := rfl
    simp [h1, h2, h3, h4, h5]
    sorry

end fourth_student_guess_l401_401561


namespace calculate_area_of_square_field_l401_401937

def area_of_square_field (t: ℕ) (v: ℕ) (d: ℕ) (s: ℕ) (a: ℕ) : Prop :=
  t = 10 ∧ v = 16 ∧ d = v * t ∧ 4 * s = d ∧ a = s^2

theorem calculate_area_of_square_field (t v d s a : ℕ) 
  (h1: t = 10) (h2: v = 16) (h3: d = v * t) (h4: 4 * s = d) 
  (h5: a = s^2) : a = 1600 := by
  sorry

end calculate_area_of_square_field_l401_401937


namespace range_g_l401_401012

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x - π / 6) * cos x + 1 / 2
noncomputable def g (x : ℝ) : ℝ := sin (2 * x + 2 * π / 3)

theorem range_g :
  ∀ x, x ∈ set.Icc (-π / 3) (π / 3) → g x ∈ set.Icc (-√3 / 2) 1 :=
sorry

end range_g_l401_401012


namespace weighted_average_salary_l401_401230

theorem weighted_average_salary :
  let num_managers := 9
  let salary_managers := 4500
  let num_associates := 18
  let salary_associates := 3500
  let num_lead_cashiers := 6
  let salary_lead_cashiers := 3000
  let num_sales_representatives := 45
  let salary_sales_representatives := 2500
  let total_salaries := 
    (num_managers * salary_managers) +
    (num_associates * salary_associates) +
    (num_lead_cashiers * salary_lead_cashiers) +
    (num_sales_representatives * salary_sales_representatives)
  let total_employees := 
    num_managers + num_associates + num_lead_cashiers + num_sales_representatives
  let weighted_avg_salary := total_salaries / total_employees
  weighted_avg_salary = 3000 := 
by
  sorry

end weighted_average_salary_l401_401230


namespace exist_a_b_m_bounds_f_l401_401372

noncomputable def f (x : ℝ) : ℝ :=
(x + 2 * Real.sin x) * (2^(-x) + 1)

theorem exist_a_b_m_bounds_f :
  ∃ a b m : ℝ, ∀ x : ℝ, 0 < x → |f(x) - a * x - b| ≤ m :=
sorry

end exist_a_b_m_bounds_f_l401_401372


namespace find_integer_cosine_l401_401328

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401328


namespace matrix_series_product_l401_401301

theorem matrix_series_product :
  (List.foldl (λ (mat_acc : Matrix (Fin 2) (Fin 2) ℚ) (mat : Matrix (Fin 2) (Fin 2) ℚ), mat_acc ⬝ mat)
              (1 : Matrix (Fin 2) (Fin 2) ℚ)
              (List.map (λ a, ![![1, a], ![0, 1]]) (List.range' 2 50).map (λ x, 2 * (x + 1))))
  = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_series_product_l401_401301


namespace monthly_growth_20_percent_additional_sales_points_l401_401227

section EggProduction

variable (marEggs mayEggs salePerPoint : ℕ)
variable (growthRate : ℝ)

-- Conditions
def eggProdInMar := marEggs = 25000
def eggProdInMay := mayEggs = 36000
def sameGrowthRate := ∃ (x : ℝ), x = growthRate ∧ 2.5 * (1 + growthRate)^2 = 3.6
def salesLimit := salePerPoint = 320000

-- Part 1: Proof of growth rate being 20%
theorem monthly_growth_20_percent
  (h_mar : eggProdInMar marEggs)
  (h_may : eggProdInMay mayEggs)
  (h_growth : sameGrowthRate growthRate) :
  growthRate = 0.2 :=
sorry

-- Part 2: Proof of the number of additional sales points needed being 2
theorem additional_sales_points
  (h_mar : eggProdInMar marEggs)
  (h_may : eggProdInMay mayEggs)
  (h_growth : sameGrowthRate growthRate)
  (h_sales : salesLimit salePerPoint)
  (h_growth_def : growthRate = 0.2) :
  let junEggs := mayEggs * (1 + growthRate) in
  let reqPoints : ℕ := ((junEggs.toFloat / salePerPoint.toFloat).ceil).to_nat in
  let mayPoints : ℕ := ((mayEggs.toFloat / salePerPoint.toFloat).ceil).to_nat in
  reqPoints - mayPoints = 2 :=
sorry

end EggProduction

end monthly_growth_20_percent_additional_sales_points_l401_401227


namespace circles_are_disjoint_l401_401792

noncomputable def positional_relationship_of_circles (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : Prop :=
R₁ + R₂ = d

theorem circles_are_disjoint {R₁ R₂ d : ℝ} (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : positional_relationship_of_circles R₁ R₂ d h₁ h₂ :=
by sorry

end circles_are_disjoint_l401_401792


namespace conditional_probability_l401_401837

noncomputable def P (e : Prop) : ℝ := sorry

variable (A B : Prop)

variables (h1 : P A = 0.6)
variables (h2 : P B = 0.5)
variables (h3 : P (A ∨ B) = 0.7)

theorem conditional_probability :
  (P A ∧ P B) / P B = 0.8 := by
  sorry

end conditional_probability_l401_401837


namespace injective_functions_count_l401_401380

theorem injective_functions_count (m n : ℕ) (h_mn : m ≥ n) (h_n2 : n ≥ 2) :
  ∃ k, k = Nat.choose m n * (2^n - n - 1) :=
sorry

end injective_functions_count_l401_401380


namespace problem_l401_401475

noncomputable def f (A B x : ℝ) : ℝ := A * x^2 + B
noncomputable def g (A B x : ℝ) : ℝ := B * x^2 + A

theorem problem (A B x : ℝ) (h : A ≠ B) 
  (h1 : f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := 
  sorry

end problem_l401_401475


namespace find_m_l401_401001

def direction_vector : ℝ × ℝ × ℝ := (1, -2, 0)
def normal_vector (m : ℝ) : ℝ × ℝ × ℝ := (m, 3, 6)

theorem find_m (m : ℝ) :
  let d := direction_vector,
      n := normal_vector m
  in d.1 * n.1 + d.2 * n.2 + d.3 * n.3 = 0 → m = 6 :=
by
  let d := direction_vector
  let n := normal_vector m
  intros h
  sorry

end find_m_l401_401001


namespace range_of_a_l401_401410

def A := {x : ℝ | |x| >= 3}
def B (a : ℝ) := {x : ℝ | x >= a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a <= -3 :=
sorry

end range_of_a_l401_401410


namespace range_of_f_gt_zero_l401_401426

-- Define the function f given the condition in the problem
def f (x : ℝ) : ℝ := Real.log (-(x)) / Real.log 2

-- State the theorem to be proved
theorem range_of_f_gt_zero : {x : ℝ | f x > 0} = {x : ℝ | x < -1} :=
by
  sorry -- Placeholder for the actual proof

end range_of_f_gt_zero_l401_401426


namespace max_value_y_l401_401999

theorem max_value_y (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) :
  let y := (1 / Real.sin θ - 1) * (1 / Real.cos θ - 1) in
  y ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end max_value_y_l401_401999


namespace total_work_completed_in_days_l401_401214

-- Define the number of days Amit can complete the work
def amit_days : ℕ := 15

-- Define the number of days Ananthu can complete the work
def ananthu_days : ℕ := 90

-- Define the number of days Amit worked
def amit_work_days : ℕ := 3

-- Calculate the amount of work Amit can do in one day
def amit_work_day_rate : ℚ := 1 / amit_days

-- Calculate the amount of work Ananthu can do in one day
def ananthu_work_day_rate : ℚ := 1 / ananthu_days

-- Calculate the total work completed
theorem total_work_completed_in_days :
  amit_work_days * amit_work_day_rate + (1 - amit_work_days * amit_work_day_rate) / ananthu_work_day_rate = 75 :=
by
  -- Placeholder for the proof
  sorry

end total_work_completed_in_days_l401_401214


namespace cost_of_paints_l401_401254

def paintbrush_cost := 1.50
def easel_cost := 12.65
def albert_has := 6.50
def albert_needs := 12

theorem cost_of_paints :
  albert_has + albert_needs - (paintbrush_cost + easel_cost) = 4.35 :=
by
  sorry

end cost_of_paints_l401_401254


namespace pairwise_distance_sum_l401_401127

theorem pairwise_distance_sum (n : ℕ) (blue red : Fin n → ℝ) :
  ∑ i j, abs (blue i - blue j) + ∑ i j, abs (red i - red j) ≤ 
  ∑ i j, abs (blue i - red j) := 
sorry

end pairwise_distance_sum_l401_401127


namespace compute_BSNK_l401_401889

noncomputable def B : ℝ := sorry
noncomputable def S : ℝ := sorry
noncomputable def N : ℝ := sorry
noncomputable def K : ℝ := sorry

axiom condition1 : log 10 (B * S) + log 10 (B * N) = 3
axiom condition2 : log 10 (N * K) + log 10 (N * S) = 4
axiom condition3 : log 10 (K * B) + log 10 (K * S) = 5

theorem compute_BSNK : B * S * N * K = 10^4 := 
by
  sorry

end compute_BSNK_l401_401889


namespace coefficient_of_x2_l401_401715

-- Define the binomial expansion
def binomial_expansion (x : ℝ) : ℝ :=
  (x^2 / 2 - 1 / real.sqrt x) ^ 6

-- Theorem stating the coefficient of the x^2 term
theorem coefficient_of_x2 :
  polynomial.coeff (polynomial.of_real (binomial_expansion x)) 2 = 15/4 :=
sorry

end coefficient_of_x2_l401_401715


namespace find_t_l401_401774

theorem find_t (t : ℝ) : 
  (1 < ∀x, (2 * x - t + 1 < 0)) ∧ (∀x, x ^ 2 + (2 * t - 4) * x + 4 > 0) ↔ (3 < t ∧ t < 4) := by
  sorry

end find_t_l401_401774


namespace M_inter_N_is_1_2_l401_401028

-- Definitions based on given conditions
def M : Set ℝ := { y | ∃ x : ℝ, x > 0 ∧ y = 2^x }
def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Prove intersection of M and N is (1, 2]
theorem M_inter_N_is_1_2 :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end M_inter_N_is_1_2_l401_401028


namespace two_digit_number_unique_solution_l401_401485

theorem two_digit_number_unique_solution
  (a b : ℕ)
  (h1 : 1 ≤ a) (h2 : a < 10)
  (h3 : 0 ≤ b) (h4 : b < 10)
  (h5 : a ≠ b)
  (h6 : (10 * a + b) = (a - b)! * (10 * b + a - 3)) :
  (10 * a + b = 42) :=
begin
  sorry
end

end two_digit_number_unique_solution_l401_401485


namespace digit_B_divisible_by_3_l401_401197

theorem digit_B_divisible_by_3 :
  ∃ B : ℕ, B < 10 ∧ (9 + 5 + 2 + B) % 3 = 0 :=
by
  use 2
  simp
  sorry

end digit_B_divisible_by_3_l401_401197


namespace sequence_bound_l401_401122

def sequence (a : ℕ → ℕ) : Prop :=
a 1 = 2 ∧ ∀ n ≥ 1, a (n + 1) = a n ^ 2 - a n + 1

theorem sequence_bound (a : ℕ → ℕ) (h : sequence a) : ∀ n ≥ 1, a (n + 1) ≥ 2^n + 1 :=
by
  sorry

end sequence_bound_l401_401122


namespace S_part_a_S_part_b_l401_401157

noncomputable def S (x : ℝ) (n : ℕ) : ℝ :=
  n * (n-1) * (x^2 + (n+1)*x + (n+1)*(3*n+2)/12)

theorem S_part_a (x : ℝ) (n : ℕ) (h_n_pos : n > 0) :
  (∑ p in Finset.range (n+1), ∑ q in Finset.range (n+1), if p ≠ q then (x + p) * (x + q) else 0) =
  n * (n-1) * (x^2 + (n+1) * x + (n+1) * (3 * n + 2)/12) := sorry

theorem S_part_b (n : ℕ) :
  (∃ x : ℤ, S (x : ℝ) n = 0) ↔ (∃ k : ℕ, n = 3 * k^2 - 1) := sorry

end S_part_a_S_part_b_l401_401157


namespace compare_abc_l401_401423

open Real

theorem compare_abc (a b c : ℝ) (h1 : 2 ^ a = 3) (h2 : b = log 2 5) (h3 : 3 ^ c = 2) : c < a ∧ a < b := 
by sorry

end compare_abc_l401_401423


namespace angle_EDL_eq_angle_ELD_l401_401858

-- Define the elements present in the problem
variables (A B C D M E N L : Point)
variables (angleA : angle A B C = π / 3)
variables (D_on_AC : OnLine D A C)
variables (M_on_AC : OnLine M A C)
variables (E_on_AB : OnLine E A B)
variables (N_on_AB : OnLine N A B)
variables (DN_perpendicular_AC : PerpendicularBisector D N A C)
variables (EM_perpendicular_AB : PerpendicularBisector E M A B)
variables (L_is_midpoint : Midpoint L M N)

-- Goal to prove
theorem angle_EDL_eq_angle_ELD : 
  angle E D L = angle E L D :=
sorry

end angle_EDL_eq_angle_ELD_l401_401858


namespace problem_statement_l401_401811

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l401_401811


namespace original_fraction_is_two_thirds_l401_401955

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l401_401955


namespace andrea_still_needs_rhinestones_l401_401677

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l401_401677


namespace f_sum_zero_l401_401366

section
variable {x : ℝ}

def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem f_sum_zero (x : ℝ) : f(x) + f(1/x) = 0 :=
by
  sorry
end

end f_sum_zero_l401_401366


namespace sum_of_solutions_l401_401992

theorem sum_of_solutions :
  let eq := (4 * x + 3) * (3 * x - 7) = 0 in
  (is_solution eq (-3/4) ∧ is_solution eq (7/3)) → 
  (-3 / 4 + 7 / 3 = 19 / 12) :=
by 
  intros eq h
  sorry

end sum_of_solutions_l401_401992


namespace pyarelal_loss_l401_401258

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ratio : ℝ) 
  (h1 : Ashok_capital = P / 9)
  (h2 : total_loss = 1200)
  (h3 : ratio = 9)
  : Pyarelal_loss = (ratio / (1 + ratio)) * total_loss :=
by {
    -- suppose the theorem declarations
    let Ashok_capital := P / 9,
    let total_loss := 1200,
    let ratio := 9,

    -- suppose the theorem propositions
    have h1 := Ashok_capital = P / 9,
    have h2 := total_loss = 1200,
    have h3 := ratio = 9,
    sorry
}

end pyarelal_loss_l401_401258


namespace matching_pair_probability_l401_401971

-- Given conditions
def total_gray_socks : ℕ := 12
def total_white_socks : ℕ := 10
def total_socks : ℕ := total_gray_socks + total_white_socks

-- Proof statement
theorem matching_pair_probability (h_grays : total_gray_socks = 12) (h_whites : total_white_socks = 10) :
  (66 + 45) / (total_socks.choose 2) = 111 / 231 :=
by
  sorry

end matching_pair_probability_l401_401971


namespace triangle_HD_HA_ratio_l401_401670

noncomputable def triangle_ratio (a b c : ℝ) (H : ℝ × ℝ) (AD : ℝ) : ℝ :=
  if a = 9 ∧ b = 40 ∧ c = 41 then
    let orthocenter := (0, a) in
    let HD := orthocenter.2 in
    let HA := orthocenter.2 in
    HD / HA
  else 0

theorem triangle_HD_HA_ratio :
  triangle_ratio 9 40 41 (0, 9) 40 = 1 := by
  sorry

end triangle_HD_HA_ratio_l401_401670


namespace solution_proof_l401_401631

theorem solution_proof : 
  ∃ (x y : ℝ), (log (x * y) (y / x) - log y x ^ 2 = 1) ∧ (log 2 (y - x) = 1) ∧ (x = 1 ∧ y = 3) :=
by
  sorry

end solution_proof_l401_401631


namespace polynomial_g_l401_401549

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l401_401549


namespace sum_of_cubes_sign_l401_401286

theorem sum_of_cubes_sign:
  let a := (Real.sqrt 2021) - (Real.sqrt 2020)
      b := (Real.sqrt 2020) - (Real.sqrt 2019)
      c := (Real.sqrt 2019) - (Real.sqrt 2018)
  in a + b + c = 0 →
     a^3 + b^3 + c^3 < 0 :=
by 
  intros a b c h_sum
  simp [a, b, c] at *
  sorry

end sum_of_cubes_sign_l401_401286


namespace lebesgue_decomposition_l401_401104

open MeasureTheory

variables {E : Type*} [measurable_space E]
variable {ν : measure_theory.measure E}
variable {μ : measure_theory.measure E}

-- Define the Lebesgue decomposition theorem
theorem lebesgue_decomposition (μ ν : measure E) [sigma_finite ν] :
  ∃ (f : E → ℝ) 
    (D : set E) 
    (hf : measurable f) 
    (hD : ν D = 0),
    (∀ B ∈ measurable_set, μ B = ∫ x in B, f x ∂ν + μ (B \ D)) ∧
    (∀ (g : E → ℝ) (C : set E) 
      (hg : measurable g) 
      (hC : ν C = 0),
      (∀ B ∈ measurable_set, μ B = ∫ x in B, g x ∂ν + μ (B \ C)) → 
      μ (D \ C ∪ C \ D) = 0 ∧
      ν {x | f x ≠ g x} = 0 ) :=
sorry

end lebesgue_decomposition_l401_401104


namespace sum_of_two_digit_divisors_of_143_mod_eq_5_l401_401109

theorem sum_of_two_digit_divisors_of_143_mod_eq_5 :
    ∑ d in { d | 10 ≤ d ∧ d < 100 ∧ 143 % d = 5 }, d = 115 :=
by
  sorry

end sum_of_two_digit_divisors_of_143_mod_eq_5_l401_401109


namespace ship_B_has_highest_rt_no_cars_l401_401506

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l401_401506


namespace three_digit_number_prime_factors_l401_401711

theorem three_digit_number_prime_factors (A B C : ℕ) (hA : 1 ≤ A) (hC : 1 ≤ C) (hA_C: A ≠ C): 
  (∃ k : ℕ, 99 * (A - C) = 3 * k) ∧ (∃ m : ℕ, 99 * (A - C) = 11 * m) :=
by
  have h : 99 = 3 * 3 * 11 := by norm_num
  sorry

end three_digit_number_prime_factors_l401_401711


namespace max_watercolor_pens_l401_401253

theorem max_watercolor_pens (cost_per_pen total_money : ℝ) (h_cost : cost_per_pen = 1.7) (h_total : total_money = 15) :
  ↑(int.floor (total_money / cost_per_pen)) = (8 : ℝ) :=
by {
  sorry
}

end max_watercolor_pens_l401_401253


namespace cos_eq_43_l401_401322

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401322


namespace broken_line_within_circle_l401_401650

/- Define the points and angles -/
variable {α : Type*} [normed_group α] [normed_space ℝ α]
open real

/-- Define the milestone points on the broken line -/
def A : ℕ → α

/-- Define the distance (length of segments) condition -/
def segment_lengths (n : ℕ) : Prop := ∀ i, (i < n) → dist (A i) (A (i + 1)) = 1

/-- Define the angle condition (in radians) -/
def angle_condition (n : ℕ) : Prop :=
∀ i, (i + 2 < n) → (π / 3) ≤ ∠ (A i) (A (i + 1)) (A (i + 2)) ∧ ∠ (A i) (A (i + 1)) (A (i + 2)) ≤ (2 * π / 3)

/-- Define the main theorem -/
theorem broken_line_within_circle (n : ℕ) (h1 : segment_lengths n) (h2 : angle_condition n) : 
∀ k, (k ≤ n) → dist (A 0) (A k) < 4 :=
sorry

end broken_line_within_circle_l401_401650


namespace cosine_periodicity_l401_401336

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401336


namespace train_passes_jogger_in_30_seconds_l401_401237

-- Definitions of speeds and distances based on given conditions
def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def initial_distance_m : ℝ := 180
def train_length_m : ℝ := 120

-- Conversion from km/h to m/s
def kmph_to_mps (v: ℝ): ℝ := v * (1000 / 3600)

-- Calculations for relative speed and total distance
def jogger_speed_mps : ℝ := kmph_to_mps jogger_speed_kmh
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmh
def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
def total_distance_m : ℝ := initial_distance_m + train_length_m

-- Define the expected time to pass the jogger
noncomputable def time_to_pass_jogger : ℝ := total_distance_m / relative_speed_mps

-- Stating the theorem to be proven
theorem train_passes_jogger_in_30_seconds : time_to_pass_jogger = 30 := sorry

end train_passes_jogger_in_30_seconds_l401_401237


namespace matrix_arithmetic_series_l401_401296

theorem matrix_arithmetic_series :
  ∏ (k : ℕ) in (range 50), (λ n => matrix.of !) (k.succ * 2) 0 0 1) = 
  matrix.of! 1 2550 0 1 :=
by
  sorry

end matrix_arithmetic_series_l401_401296


namespace marathon_training_l401_401187

theorem marathon_training (x : ℝ) (h : 3 * x^4 = 26.3) : x ≈ 1.714 := by
  sorry

end marathon_training_l401_401187


namespace moles_CO2_is_one_l401_401347

noncomputable def moles_CO2_formed (moles_HNO3 moles_NaHCO3 : ℕ) : ℕ :=
  if moles_HNO3 = 1 ∧ moles_NaHCO3 = 1 then 1 else 0

theorem moles_CO2_is_one :
  moles_CO2_formed 1 1 = 1 :=
by
  sorry

end moles_CO2_is_one_l401_401347


namespace possible_polynomials_l401_401551

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l401_401551


namespace combination_identity_remainder_of_power_sum_of_coefficients_sum_of_powers_l401_401625

theorem combination_identity : (nat.choose 10 6) + (nat.choose 10 5) = (nat.choose 11 5) := 
sorry

theorem remainder_of_power : (2^30 - 3) % 7 ≠ 2 := 
sorry

theorem sum_of_coefficients (x : ℝ) (a : ℕ → ℝ) :
  (2 * x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 → 
  (∑ i in finset.range 10, |a (i + 1)|) = 3^10 - 1 :=
sorry

theorem sum_of_powers (n : ℕ) : 
  (∑ i in finset.range (n + 1), 2 ^ i * nat.choose n i) = 2187 → 
  (∑ i in finset.range n, nat.choose n (i + 1)) = 127 :=
sorry

end combination_identity_remainder_of_power_sum_of_coefficients_sum_of_powers_l401_401625


namespace find_n_cos_eq_l401_401344

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401344


namespace functional_characterization_l401_401304

-- Define the Euler's totient function (φ function) as a placeholder.
noncomputable def φ (n : ℕ) : ℕ := sorry

-- Define the function f and its property.
theorem functional_characterization (f : ℕ → ℕ)
  (h : ∀ m n : ℕ, m ≥ n → f(m * φ(n^3)) = f(m) * φ(n^3)) : 
  ∃ b : ℕ, ∀ n : ℕ, f(n) = b * n :=
by
  sorry


end functional_characterization_l401_401304


namespace find_b_l401_401580

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.fst + p2.fst) / 2, (p1.snd + p2.snd) / 2)

theorem find_b (p1 p2 : ℝ × ℝ) (h : p1 = (2, 4) ∧ p2 = (6, 8)) (M : midpoint p1 p2 = (4, 6)) :
  ∃ b : ℝ, ∀ x y : ℝ, (x, y) = M → (x + y = b) := 
begin
  use 10,
  intros x y hxy,
  rw hxy,
  exact rfl,
end

end find_b_l401_401580


namespace all_integer_solutions_form_l401_401915

theorem all_integer_solutions_form (p q : ℤ) (n : ℕ) :
  (p + q * real.sqrt 5 = 1 → (p = 1 ∧ q = 0)) ∧
  (p + q * real.sqrt 5 = 9 + 4 * real.sqrt 5 → (p = 9 ∧ q = 4)) ∧
  (p + q * real.sqrt 5 = (9 + 4 * real.sqrt 5)^n → (p ≥ 0 ∧ q ≥ 0)) :=
sorry
 
end all_integer_solutions_form_l401_401915


namespace not_factorable_into_quadratics_l401_401152

theorem not_factorable_into_quadratics :
  ¬ ∃ (a b c d : ℤ), (a + c = 0 ∧ ad + b * c = 2 ∧ b * d = 2 ∧ ac + b + d = 2 ∧
                     ∀ x : ℝ, (x^4 + 2 * x^2 + 2 * x + 2) = (x^2 + a * x + b) * (x^2 + c * x + d)) :=
by
  intro h
  cases h with a ha
  cases ha with b hb
  cases hb with c hc
  cases hc with d hd
  cases hd with hac habcd
  sorry

end not_factorable_into_quadratics_l401_401152


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401511

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401511


namespace problem1_problem2_problem3_l401_401686

theorem problem1 (p : ℝ) : (-p) ^ 2 * (-p) ^ 3 = -p ^ 5 :=
by sorry

theorem problem2 (a b : ℝ) : (- (1 / 2 * a ^ 2 * b)) ^ 3 = - (1 / 8 * a ^ 6 * b ^ 3) :=
by sorry

theorem problem3 (a b : ℝ) : (-3 * a ^ 2 * b) ^ (-2) = 1 / (9 * a ^ 4 * b ^ 2) :=
by sorry

end problem1_problem2_problem3_l401_401686


namespace triangle_position_after_rolling_l401_401555

def inner_angle_of_hexagon := 120
def inner_angle_of_square := 90
def rotation_per_movement := 150
def total_rotation_after_movements := 450 % 360
def initial_position_of_triangle := "bottom"

theorem triangle_position_after_rolling (
  H1 : inner_angle_of_hexagon = 120,
  H2 : inner_angle_of_square = 90,
  H3 : rotation_per_movement = 150,
  H4 : total_rotation_after_movements % 360 = 90,
  H5 : initial_position_of_triangle = "bottom"
) : "right" = "right" :=
by
  sorry

end triangle_position_after_rolling_l401_401555


namespace soap_economy_order_l401_401249

structure Soap (price_per_gram : ℚ)

noncomputable def tiny : Soap :=
{ price_per_gram := 1 }

noncomputable def normal : Soap :=
{ price_per_gram := (1.4 : ℚ) / (0.75 * 3) }

noncomputable def jumbo : Soap :=
{ price_per_gram := (1.2 * 1.4 : ℚ) / 3 }

theorem soap_economy_order :
  (jumbo.price_per_gram ≤ normal.price_per_gram) ∧ (normal.price_per_gram ≤ tiny.price_per_gram) :=
  sorry

end soap_economy_order_l401_401249


namespace coefficient_x3_expansion_l401_401730

open Polynomial

noncomputable def expansion : Polynomial ℤ := (X - 1) * Polynomial.C 2 * X + Polynomial.C 1)^ 5

theorem coefficient_x3_expansion : coeff expansion 3 = -40 := by
  -- Proof goes here
  sorry

end coefficient_x3_expansion_l401_401730


namespace sin_triple_product_l401_401694

theorem sin_triple_product : 
  sin (10 * real.pi / 180) * sin (50 * real.pi / 180) * sin (70 * real.pi / 180) = 1 / 8 :=
by
  sorry

end sin_triple_product_l401_401694


namespace winnie_keeps_balloons_l401_401627

theorem winnie_keeps_balloons (red white green chartreuse friends total remainder : ℕ) (hRed : red = 17) (hWhite : white = 33) (hGreen : green = 65) (hChartreuse : chartreuse = 83) (hFriends : friends = 10) (hTotal : total = red + white + green + chartreuse) (hDiv : total % friends = remainder) : remainder = 8 :=
by
  have hTotal_eq : total = 198 := by
    sorry -- This would be the computation of 17 + 33 + 65 + 83
  have hRemainder_eq : 198 % 10 = remainder := by
    sorry -- This would involve the computation of the remainder
  exact sorry -- This would be the final proof that remainder = 8, tying all parts together

end winnie_keeps_balloons_l401_401627


namespace area_triangle_constant_equation_of_circle_l401_401449

-- Define the first part of the problem: proving the area of triangle ΔAOB is constant
theorem area_triangle_constant (t : ℝ) (ht : t ≠ 0) :
  let center := (t, (Real.sqrt 3) / t),
      A := (2 * t, 0),
      B := (0, 2 * (Real.sqrt 3) / t) in
  ∃ S : ℝ, S = 2 * Real.sqrt 3 := by
  let center := (t, (Real.sqrt 3) / t)
  let A := (2 * t, 0)
  let B := (0, 2 * (Real.sqrt 3) / t)
  existsi (2 * Real.sqrt 3)
  sorry

-- Define the second part of the problem: finding the equation of the circle
theorem equation_of_circle :
  let center := (1, Real.sqrt 3),
      radius := 2,
      l := line.mk (- Real.sqrt 3 / 3) 4 in
  ∀ (x y : ℝ), (x-1)^2 + (y - Real.sqrt 3)^2 = radius^2 → y = - Real.sqrt 3 / 3 * x + 4 → (x-1)^2 + (y - Real.sqrt 3)^2 = 4 := by
  let center := (1, Real.sqrt 3)
  let radius := 2
  let l := line.mk (- Real.sqrt 3 / 3) 4
  intros x y h1 h2
  existsi ((x - 1)^2 + (y - Real.sqrt 3)^2 = 4)
  sorry

end area_triangle_constant_equation_of_circle_l401_401449


namespace pyramid_circumscribed_sphere_volume_l401_401939

theorem pyramid_circumscribed_sphere_volume 
  (PA ABCD : ℝ) 
  (square_base : Prop)
  (perpendicular_PA_base : Prop)
  (AB : ℝ)
  (PA_val : PA = 1)
  (AB_val : AB = 2) 
  : (∃ (volume : ℝ), volume = (4/3) * π * (3/2)^3 ∧ volume = 9 * π / 2) := 
by
  -- Provided the conditions, we need to prove that the volume of the circumscribed sphere is 9π/2
  sorry

end pyramid_circumscribed_sphere_volume_l401_401939


namespace alice_bob_same_point_after_3_turns_l401_401255

noncomputable def alice_position (t : ℕ) : ℕ := (15 + 4 * t) % 15

noncomputable def bob_position (t : ℕ) : ℕ :=
  if t < 2 then 15
  else (15 - 11 * (t - 2)) % 15

theorem alice_bob_same_point_after_3_turns :
  ∃ t, t = 3 ∧ alice_position t = bob_position t :=
by
  exists 3
  simp only [alice_position, bob_position]
  norm_num
  -- Alice's position after 3 turns
  -- alice_position 3 = (15 + 4 * 3) % 15
  -- bob_position 3 = (15 - 11 * (3 - 2)) % 15
  -- Therefore,
  -- alice_position 3 = 12
  -- bob_position 3 = 12
  sorry

end alice_bob_same_point_after_3_turns_l401_401255


namespace inequality_integral_ln_bounds_l401_401223

-- Define the conditions
variables (x a : ℝ)
variables (hx : 0 < x) (ha : x < a)

-- First part: inequality involving integral
theorem inequality_integral (hx : 0 < x) (ha : x < a) :
  (2 * x / a) < (∫ t in a - x..a + x, 1 / t) ∧ (∫ t in a - x..a + x, 1 / t) < x * (1 / (a + x) + 1 / (a - x)) :=
sorry

-- Second part: to prove 0.68 < ln(2) < 0.71 using the result of the first part
theorem ln_bounds :
  0.68 < Real.log 2 ∧ Real.log 2 < 0.71 :=
sorry

end inequality_integral_ln_bounds_l401_401223


namespace projection_orthogonal_l401_401882

variables (a b : ℝ × ℝ)
variables (v : ℝ × ℝ)
variables (h1 : dot_product a b = 0) -- a and b are orthogonal
variables (h2 : proj a (4, -2) = (1, 2)) -- projection of (4, -2) onto a

-- Theorem statement
theorem projection_orthogonal {a b : ℝ × ℝ} {v : ℝ × ℝ}
  (h1 : dot_product a b = 0)
  (h2 : proj a v = (1, 2)) :
  proj b v = (3, -4) :=
sorry

end projection_orthogonal_l401_401882


namespace minimum_value_l401_401828

theorem minimum_value (x y z : ℝ) (h : x + y + z = 1) : 2 * x^2 + y^2 + 3 * z^2 ≥ 3 / 7 := by
  sorry

end minimum_value_l401_401828


namespace shift_sine_function_l401_401578

theorem shift_sine_function (ω : ℝ) (hω : ω > 0) : 
  (∀n : ℕ, n > 0 → ∃ x : ℝ, (sin (ω * x + (π / 6)) = 0) ∧ (x = n * (π / (2 * ω)))) →
  (∀ x : ℝ, sin (ω * (x - (π / 12))) = sin (ω * x)) := 
by 
  sorry

end shift_sine_function_l401_401578


namespace frame_cover_100x100_l401_401666

theorem frame_cover_100x100 :
  ∃! (cover: (ℕ → ℕ → Prop)), (∀ (n : ℕ) (frame: ℕ → ℕ → Prop),
    (∃ (i j : ℕ), (cover (i + n) j ∧ frame (i + n) j ∧ cover (i - n) j ∧ frame (i - n) j) ∧
                   (∃ (k l : ℕ), (cover k (l + n) ∧ frame k (l + n) ∧ cover k (l - n) ∧ frame k (l - n)))) →
    (∃ (i' j' k' l' : ℕ), cover i' j' ∧ frame i' j' ∧ cover k' l' ∧ frame k' l')) :=
sorry

end frame_cover_100x100_l401_401666


namespace ordered_pizzas_l401_401984

theorem ordered_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) : total_slices / slices_per_pizza = 14 := 
by
  -- These are our given conditions
  rw [h1, h2]
  -- We need to prove that 28 / 2 = 14
  exact Nat.div_eq_of_eq_mul_right (by decide : 0 < 2) rfl

end ordered_pizzas_l401_401984


namespace find_question_mark_l401_401222

noncomputable def c1 : ℝ := (5568 / 87)^(1/3)
noncomputable def c2 : ℝ := (72 * 2)^(1/2)
noncomputable def sum_c1_c2 : ℝ := c1 + c2

theorem find_question_mark : sum_c1_c2 = 16 → 256 = 16^2 :=
by
  sorry

end find_question_mark_l401_401222


namespace distribution_schemes_count_l401_401288

-- Conditions as definitions:
def num_spots : ℕ := 10
def num_schools : ℕ := 4
def distribution : List ℕ := [1, 2, 3, 4]

-- The proof problem as a Lean statement:
theorem distribution_schemes_count :
  (∃ (schools : Finset (Fin num_schools)), schools.card = num_schools) →
  (∑ s in (Finset.range num_schools), (distribution.nth s).get_or_else 0) = num_spots →
  (∃! n, n = 24) :=
by
  intros _ _
  use 24
  split
  exact sorry
  intros y hy
  exact sorry

end distribution_schemes_count_l401_401288


namespace minimum_value_expression_l401_401914

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^4 + y^4 + z^4 = 1) :
  (∑ t in [x, y, z], t^3 / (1 - t^8)) = 9 * Real.root 3 4 / 8 :=
sorry

end minimum_value_expression_l401_401914


namespace sum_divisible_2017_not_2018_l401_401917

def S : ℤ := ∑ k in finset.range 2017, k^2017

theorem sum_divisible_2017_not_2018 :
  (S % 2017 = 0) ∧ (S % 2018 ≠ 0) :=
by
  sorry

end sum_divisible_2017_not_2018_l401_401917


namespace find_cos_B_C_find_side_a_l401_401364

-- Variables and conditions from the problem
variables {A B C : ℝ} {a b c : ℝ}

-- Initial conditions
def conditions (A B C a b c : ℝ) : Prop :=
  B = 2 * C ∧
  sin C = sqrt 7 / 4 ∧
  b * c = 24

-- First part: Proving values of cos B and cos A
theorem find_cos_B_C (A B C a b c : ℝ) 
  (h : conditions A B C a b c) : 
  cos B = 1 / 8 ∧ cos A = 9 / 16 := 
by
  sorry

-- Second part: Proving the length of side a
theorem find_side_a (A B C a b c : ℝ) 
  (h : conditions A B C a b c) : 
  a = 5 := 
by
  sorry

end find_cos_B_C_find_side_a_l401_401364


namespace find_n_l401_401330

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401330


namespace find_numbers_with_identical_digits_l401_401728

theorem find_numbers_with_identical_digits :
  ∃ x ∈ {9, 18, 27, 36, 45, 54, 63, 72, 81}, 
  ∃ a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    12345679 * x = 111111111 * a :=
by
  sorry

end find_numbers_with_identical_digits_l401_401728


namespace regular_octagon_AEC_angle_l401_401986

theorem regular_octagon_AEC_angle (ABCDEFGH : Type) [h: regular_octagon ABCDEFGH] :
  degree_measure_of_angle_AEC ABCDEFGH = 22.5 := 
sorry

end regular_octagon_AEC_angle_l401_401986


namespace convert_to_rectangular_form_l401_401277

noncomputable def rectangular_form (z : ℂ) : ℂ :=
  let e := Complex.exp (13 * Real.pi * Complex.I / 6)
  3 * e

theorem convert_to_rectangular_form :
  rectangular_form (3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 * (Complex.cos (Real.pi / 6)) + 3 * Complex.I * (Complex.sin (Real.pi / 6))) :=
by
  sorry

end convert_to_rectangular_form_l401_401277


namespace probability_not_blue_l401_401825

theorem probability_not_blue (odds_blue : ℕ × ℕ) (h : odds_blue = (5, 6)) :
  let total := odds_blue.1 + odds_blue.2 in
  let not_blue := odds_blue.2 in
  (not_blue : ℚ) / (total : ℚ) = 6 / 11 :=
by
  simp [h]
  sorry

end probability_not_blue_l401_401825


namespace sequence_an_correct_l401_401385

noncomputable def sequence_a (n : ℕ) : ℚ := 
  if n = 1 then 1 
  else -2 / ((2 * n - 1) * (2 * n - 3))

noncomputable def sequence_S (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ k, sequence_a (k + 1))

theorem sequence_an_correct (n : ℕ) :
  (sequence_a 1 = 1) ∧
  (∀ n, n ≥ 2 → sequence_a n = (2 * sequence_S n ^ 2) / (2 * sequence_S n - 1)) ∧
  (sequence_a n = 
    if n = 1 then 1 
    else -2 / ((2 * n - 1) * (2 * n - 3))) :=
by sorry

end sequence_an_correct_l401_401385


namespace least_integer_divisibility_l401_401659

theorem least_integer_divisibility :
  ∃ N : ℕ,
  (∀ k : ℕ, k ∈ {1, 2, ..., 26, 30} → k ∣ N) ∧
  ¬ (27 ∣ N) ∧ ¬ (28 ∣ N) ∧ ¬ (29 ∣ N) ∧
  ∀ M : ℕ, 
    (∀ k : ℕ, k ∈ {1, 2, ..., 26, 30} → k ∣ M) ∧
    ¬ (27 ∣ M) ∧ ¬ (28 ∣ M) ∧ ¬ (29 ∣ M) →
    N ≤ M :=
exists.intro 1225224000 (by
  sorry)

end least_integer_divisibility_l401_401659


namespace even_increasing_func_implies_decreasing_l401_401773

-- Define an even function
def even_function (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

-- Define a function that is increasing on (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) := ∀ x1 x2, 0 < x1 ∧ x1 < x2 → f(x1) < f(x2)

-- Main theorem statement
theorem even_increasing_func_implies_decreasing (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_incr_pos : increasing_on_pos f) :
  ∀ x1 x2, x1 < 0 ∧ x2 < 0 ∧ x1 < x2 → f(x1) > f(x2) :=
begin
  sorry
end

end even_increasing_func_implies_decreasing_l401_401773


namespace dice_probability_l401_401218

theorem dice_probability : 
  let total_outcomes := 36
  let favorable_outcomes := 6
  in (favorable_outcomes / total_outcomes : ℝ) = 1 / 6 :=
by
  sorry

end dice_probability_l401_401218


namespace find_integer_cosine_l401_401326

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l401_401326


namespace find_k_l401_401819

theorem find_k (m : ℝ) (h : ∃ A B : ℝ, (m^3 - 24*m + 16) = (m^2 - 8*m) * (A*m + B) ∧ A - 8 = -k ∧ -8*B = -24) : k = 5 :=
sorry

end find_k_l401_401819


namespace start_net_profit_from_4th_year_profit_equivalence_and_choice_l401_401642

-- Define the initial investment, annual rental income, and renovation costs
def initial_investment := 810000
def annual_rental_income := 300000
def first_year_renovation_cost := 10000
def renovation_cost_increment := 20000

-- Function to calculate the renovation cost for a given year
def renovation_cost (n : ℕ) : ℕ :=
  first_year_renovation_cost + renovation_cost_increment * (n - 1)

-- Function to calculate the net profit for a given year
def net_profit (n : ℕ) : ℕ :=
  annual_rental_income - renovation_cost n

-- 1. Prove that the developer starts to make a net profit from the 4th year
theorem start_net_profit_from_4th_year : ∀ n, n < 4 → net_profit n < initial_investment := by
  sorry

-- 2. Considering two options, prove the profit equivalence and choice
def selling_price_option_one := 460000
def selling_price_option_two := 100000
def average_annual_profit_option_one := ∀ n, (annual_rental_income * n - initial_investment - ∑ k in range(n), (renovation_cost k)) / n
def total_net_profit_option_two (n : ℕ) : ℕ := annual_rental_income * n - initial_investment - ∑ k in range(n), (renovation_cost k)

theorem profit_equivalence_and_choice :
  ∀ n, average_annual_profit_option_one n = total_net_profit_option_two n / n → selling_price_option_one > selling_price_option_two := by
  sorry

end start_net_profit_from_4th_year_profit_equivalence_and_choice_l401_401642


namespace interval_length_of_m_640_points_l401_401878

def lattice_points_in_set (n : ℕ) : set (ℕ × ℕ) := { p | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n }

def count_points_in_set_on_or_below_line (T : set (ℕ × ℕ)) (m : ℚ) (c : ℚ) : ℕ :=
  T.count (λ p, (p.snd : ℚ) ≤ m * (p.fst : ℚ) + c)

theorem interval_length_of_m_640_points (n : ℕ) (c : ℚ) (T : set (ℕ × ℕ)) :
  T = lattice_points_in_set n →
  ∃ (m1 m2 : ℚ) (a b : ℕ), 
    count_points_in_set_on_or_below_line T m1 c = 640 ∧ 
    count_points_in_set_on_or_below_line T m2 c = 640 ∧
    (m2 - m1) = (a : ℚ) / (b : ℚ) ∧ 
    Nat.gcd a b = 1 ∧
    (a + b) = 94 :=
by
  sorry

end interval_length_of_m_640_points_l401_401878


namespace ratio_nora_to_tamara_savings_l401_401554

def total_debt : ℕ := 40
def lulu_savings : ℕ := 6
def nora_savings : ℕ := 5 * lulu_savings
def remaining_money_divided : ℕ := 2
def total_savings : ℕ := total_debt + 3 * remaining_money_divided

theorem ratio_nora_to_tamara_savings
  (total_debt = 40)
  (lulu_savings = 6)
  (nora_savings = 5 * lulu_savings)
  (remaining_money_divided = 2)
  (total_savings = total_debt + 3 * remaining_money_divided) :
  nora_savings / (total_savings - (lulu_savings + nora_savings)) = 3 :=
by
  sorry

end ratio_nora_to_tamara_savings_l401_401554


namespace defective_units_shipped_l401_401852

theorem defective_units_shipped (total_units : ℕ) (type_A_defect_rate type_B_defect_rate : ℝ)
(rework_rate_A rework_rate_B : ℝ) (ship_rate_A ship_rate_B : ℝ)
(h_total_units_positive : 0 < total_units)
(h_A_defect_rate : type_A_defect_rate = 0.07)
(h_B_defect_rate : type_B_defect_rate = 0.08)
(h_A_rework_rate : rework_rate_A = 0.40)
(h_B_rework_rate : rework_rate_B = 0.30)
(h_A_ship_rate : ship_rate_A = 0.03)
(h_B_ship_rate : ship_rate_B = 0.06) :
  let defective_A := total_units * type_A_defect_rate,
      defective_B := total_units * type_B_defect_rate,
      reworked_A := defective_A * rework_rate_A,
      reworked_B := defective_B * rework_rate_B,
      remaining_A := defective_A - reworked_A,
      remaining_B := defective_B - reworked_B,
      shipped_A := remaining_A * ship_rate_A,
      shipped_B := remaining_B * ship_rate_B,
      total_shipped_defective := (shipped_A + shipped_B) / total_units * 100 in
  total_shipped_defective = 0.462 := 
by
  sorry

end defective_units_shipped_l401_401852


namespace matrix_product_sequence_l401_401298

open Matrix

def mat (x : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![1, x], ![0, 1]]

theorem matrix_product_sequence :
  (List.prod (List.map mat (List.range (50+1)))).sum (Fin.mk 0 (by simp)) (Fin.mk 1 (by simp)) = 2550 := by
  sorry

end matrix_product_sequence_l401_401298


namespace correct_systematic_sampling_l401_401972

-- Define conditions mentioned in the problem
def population_size : ℕ := 102
def sample_size : ℕ := 9
def interval (total remaining_groups: ℕ) : ℕ := total / remaining_groups

-- Exclude 3 individuals condition
def remaining_population := population_size - 3

-- Proving that the correct sampling method is to exclude 3 individuals
-- and then divide the remaining population into 9 groups with an interval of 11.
theorem correct_systematic_sampling :
  remaining_population = 99 ∧
  (99 % sample_size = 0) ∧
  interval 99 sample_size = 11 ->
  (exclude_3_divide_99_interval_11) :=
by simp [population_size, sample_size, remaining_population, interval]; norm_num; split; sorry

end correct_systematic_sampling_l401_401972


namespace percent_defective_units_shipped_for_sale_correct_l401_401853

noncomputable def percentage_defective_units_shipped_for_sale : ℝ := 
let units_produced := 100 in
let type_A_defective := 0.07 * units_produced in
let type_A_reworked := 0.40 * type_A_defective in
let type_A_remaining := type_A_defective - type_A_reworked in
let type_A_shipped := 0.03 * type_A_remaining in
let type_B_defective := 0.08 * units_produced in
let type_B_reworked := 0.30 * type_B_defective in
let type_B_remaining := type_B_defective - type_B_reworked in
let type_B_shipped := 0.06 * type_B_remaining in
let total_defective_shipped := type_A_shipped + type_B_shipped in
(total_defective_shipped / units_produced) * 100

theorem percent_defective_units_shipped_for_sale_correct :
  percentage_defective_units_shipped_for_sale = 0.462 := by
  sorry

end percent_defective_units_shipped_for_sale_correct_l401_401853


namespace trajectory_length_eq_five_sqrt_two_plus_five_l401_401161

noncomputable def coordDistance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem trajectory_length_eq_five_sqrt_two_plus_five :
  ∀ (C : ℝ × ℝ), 0 ≤ C.1 ∧ C.1 ≤ 10 ∧ 0 ≤ C.2 ∧ C.2 ≤ 10 ∧ 
    coordDistance C (1, 3) = coordDistance C (6, 9) →
    ∃ L : ℝ, L = 5 * (Real.sqrt 2 + 1) :=
begin
  sorry
end

end trajectory_length_eq_five_sqrt_two_plus_five_l401_401161


namespace Deepak_wife_meet_time_l401_401635

theorem Deepak_wife_meet_time :
  ∀ (circumference : ℝ) (deepak_speed_km_hr wife's_speed_km_hr : ℝ),
  circumference = 1000 → deepak_speed_km_hr = 20 → wife's_speed_km_hr = 16 →
  let deepak_speed_m_min := deepak_speed_km_hr * 1000 / 60,
      wife_speed_m_min := wife's_speed_km_hr * 1000 / 60,
      combined_speed_m_min := deepak_speed_m_min + wife_speed_m_min,
      meet_time_min := circumference / combined_speed_m_min
  in meet_time_min = 1000 / (333.33 + 266.67) :=
by
  intros circumference deepak_speed_km_hr wife's_speed_km_hr h1 h2 h3;
  let deepak_speed_m_min := deepak_speed_km_hr * 1000 / 60;
  let wife_speed_m_min := wife's_speed_km_hr * 1000 / 60;
  let combined_speed_m_min := deepak_speed_m_min + wife_speed_m_min;
  let meet_time_min := circumference / combined_speed_m_min;
  sorry

end Deepak_wife_meet_time_l401_401635


namespace original_fraction_is_two_thirds_l401_401954

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l401_401954


namespace prob_sunny_l401_401672

variables (A B C : Prop) 
variables (P : Prop → ℝ)

-- Conditions
axiom prob_A : P A = 0.45
axiom prob_B : P B = 0.2
axiom mutually_exclusive : P A + P B + P C = 1

-- Proof problem
theorem prob_sunny : P C = 0.35 :=
by sorry

end prob_sunny_l401_401672


namespace train_car_count_l401_401681

noncomputable def totalTrainCars (carsPer15Sec : ℕ) (totalTimeSec : ℕ) : ℕ :=
  (carsPer15Sec * (totalTimeSec / 15))

theorem train_car_count (h1 : 10 = 10) (h2 : 3 * 60 + 30 = 210) : totalTrainCars 10 210 = 140 := 
by
  -- Define the rate of cars passing per second
  have rate_cars_per_sec : ℚ := 10 / 15
  -- Define the total time in seconds
  have total_time_sec : ℕ := 210
  -- Calculate the total number of cars
  have total_cars := rate_cars_per_sec * total_time_sec
  -- Simplify to find the total number of cars equals 140
  simp only [total_cars]
  norm_num
  sorry

end train_car_count_l401_401681


namespace july_savings_l401_401262

theorem july_savings (january: ℕ := 100) (total_savings: ℕ := 12700) :
  let february := 2 * january
  let march := 2 * february
  let april := 2 * march
  let may := 2 * april
  let june := 2 * may
  let july := 2 * june
  let total := january + february + march + april + may + june + july
  total = total_savings → july = 6400 := 
by
  sorry

end july_savings_l401_401262


namespace octagon_area_l401_401100

variable {P : Type} [plane_geom : Geometry P] 

-- Definitions for the regular octagon and midpoints
variable (A B C D E F G H I J K : P)
variable (h_octagon : regular_octagon A B C D E F G H)
variable (h_I : midpoint I A B)
variable (h_J : midpoint J D E)
variable (h_K : midpoint K G H)

-- Given condition: area of triangle IJK is 144
axiom area_triangle_IJK : area (triangle I J K) = 144

-- Proof statement
theorem octagon_area : area (octagon A B C D E F G H) = 1152 := 
sorry

end octagon_area_l401_401100


namespace holiday_rush_increase_l401_401958

theorem holiday_rush_increase (O : Real) (x : Real) :
  let new_output := 1.10 * O
  let increased_output := new_output * (1 + x / 100)
  (increased_output * (1 - 0.3506) = O) →
  x ≈ 39.986 := by
  intros
  sorry

end holiday_rush_increase_l401_401958


namespace find_trigonometric_identity_l401_401006

-- Define the conditions for the given problem:
variables (a b c S : ℝ) (A : ℝ)

-- Assume the area of the triangle is given by:
-- S = a^2 - (b - c)^2
axiom area_condition : S = a^2 - (b - c)^2

-- Prove that given these conditions, the value of
-- sin A / (1 - cos A) equals 4.
theorem find_trigonometric_identity (h : S = a^2 - (b - c)^2) : 
  ∀ (A : ℝ), b ≠ 0 → c ≠ 0 → cos A ≠ 1 → sin A / (1 - cos A) = 4 :=
by
  sorry

end find_trigonometric_identity_l401_401006


namespace determine_g_l401_401540

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l401_401540


namespace ratio_of_area_to_perimeter_l401_401689

-- Define essential elements from the conditions
def equilateral_triangle_side := 10

-- Define auxiliary functions based on the conditions and the typical calculations outlined in the solution.
def altitude_of_equilateral_triangle (a : ℝ) : ℝ :=
  let half_side := a / 2
  real.sqrt (a^2 - half_side^2)

def area_of_equilateral_triangle (a : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * a * h

def perimeter_of_equilateral_triangle (a : ℝ) : ℝ :=
  3 * a

-- The main statement proving the ratio.
theorem ratio_of_area_to_perimeter : 
  let h := altitude_of_equilateral_triangle equilateral_triangle_side in
  let area := area_of_equilateral_triangle equilateral_triangle_side h in
  let perimeter := perimeter_of_equilateral_triangle equilateral_triangle_side in
  area / perimeter = (5 * real.sqrt 3) / 6 :=
by 
  sorry

end ratio_of_area_to_perimeter_l401_401689


namespace relationship_between_x_y_l401_401821

theorem relationship_between_x_y (x y : ℝ) (h1 : x^2 - y^2 > 2 * x) (h2 : x * y < y) : x < y ∧ y < 0 := 
sorry

end relationship_between_x_y_l401_401821


namespace determine_arg_range_l401_401777

noncomputable def arg_range (a b : ℝ) (z : ℂ) : Prop :=
  arg (z - b + complex.of_real b * I) ∈
  set.Icc (real.arctan ((a + b) / (a - b)))
          (real.pi + real.arctan ((a - b) / (a + b)))

theorem determine_arg_range (z a b: ℂ)
  (h1 : complex.arg (z + a + a * I) = real.pi / 4)
  (h2 : complex.arg (z - a - a * I) = 5 * real.pi / 4)
  (ha : a.re > 0)
  (hb : b.re > 0)
  (h_ineq: a.re > b.re) :
  arg_range a.re b.re z := sorry

end determine_arg_range_l401_401777


namespace fare_for_90_miles_l401_401680

noncomputable def fare_cost (miles : ℕ) (base_fare cost_per_mile : ℝ) : ℝ :=
  base_fare + cost_per_mile * miles

theorem fare_for_90_miles (base_fare : ℝ) (cost_per_mile : ℝ)
  (h1 : base_fare = 30)
  (h2 : fare_cost 60 base_fare cost_per_mile = 150)
  (h3 : cost_per_mile = (150 - base_fare) / 60) :
  fare_cost 90 base_fare cost_per_mile = 210 :=
  sorry

end fare_for_90_miles_l401_401680


namespace max_min_sum_of_f_l401_401782

def f (x : ℝ) : ℝ := (x^2 - 2 * x) * Real.sin (x - 1) + x + 1

-- Lean statement for the proof problem
theorem max_min_sum_of_f (M m : ℝ) :
  (∃ (M m : ℝ), (∀ x ∈ Icc (-1:ℝ) 3, f x ≤ M) ∧ (∀ x ∈ Icc (-1:ℝ) 3, m ≤ f x)) →
  M + m = 4 :=
by sorry

end max_min_sum_of_f_l401_401782


namespace probability_of_winning_second_draw_given_first_is_1_5_l401_401951

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω]
variable (card_deck : Set Ω)
variable (event_A1 : Set Ω) -- drawing a card numbered 5 or 6 on first draw
variable (event_A2 : Set Ω) -- drawing a card numbered 5 or 6 on second draw

-- Card deck consists of numbers 1 to 6
def card_deck_elements : Set Ω := {1, 2, 3, 4, 5, 6}

-- Event A1: Winning on the first draw
def A1 := {5, 6}

-- Event A2: Winning on the second draw
def A2 := {5, 6}

/-- Given that a prize is won on the first draw, 
the probability of winning a prize on the second draw is 1/5. -/
theorem probability_of_winning_second_draw_given_first_is_1_5 
  (h₁ : event_A1 = A1)
  (h₂ : event_A2 = A2)
  (cards_drawn_without_replacement : True) :
  (Finset.card event_A2 / Finset.card card_deck) = 1 / 5 := by
  sorry

end probability_of_winning_second_draw_given_first_is_1_5_l401_401951


namespace find_triples_of_positive_integers_l401_401307

theorem find_triples_of_positive_integers (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn_pos : 0 < n) 
  (equation : p * (p + 3) + q * (q + 3) = n * (n + 3)) : 
  (p = 3 ∧ q = 2 ∧ n = 4) :=
sorry

end find_triples_of_positive_integers_l401_401307


namespace Gage_total_cubes_l401_401034

theorem Gage_total_cubes 
  (Grady_red_cubes : ℕ := 20)
  (Grady_blue_cubes : ℕ := 15)
  (Gage_initial_red_cubes : ℕ := 10)
  (Gage_initial_blue_cubes : ℕ := 12)
  (Grady_red_given_ratio : ℚ := 2/5)
  (Grady_blue_given_ratio : ℚ := 1/3):
  let Grady_red_given_to_Gage := Grady_red_given_ratio * Grady_red_cubes
      Grady_blue_given_to_Gage := Grady_blue_given_ratio * Grady_blue_cubes
      Gage_total_red_cubes := Gage_initial_red_cubes + Grady_red_given_to_Gage
      Gage_total_blue_cubes := Gage_initial_blue_cubes + Grady_blue_given_to_Gage
      Gage_total_cubes := Gage_total_red_cubes + Gage_total_blue_cubes
  in Gage_total_cubes = 35 :=
by
  intros
  let Grady_red_given_to_Gage := (2/5 : ℚ) * (20 : ℚ)
  let Grady_blue_given_to_Gage := (1/3 : ℚ) * (15 : ℚ)
  let Gage_total_red_cubes := (10 : ℚ) + Grady_red_given_to_Gage
  let Gage_total_blue_cubes := (12 : ℚ) + Grady_blue_given_to_Gage
  let Gage_total_cubes := Gage_total_red_cubes + Gage_total_blue_cubes
  have : Gage_total_cubes = 35, by sorry
  exact this

end Gage_total_cubes_l401_401034


namespace necessary_work_to_determine_proportions_l401_401070

def student_test_analysis : Prop :=
  ∃ (n : ℕ) (A1 A2 A3 A4 A5 : set ℕ),
    n = 800 ∧
    A1 = {x | x ≥ 120} ∧
    A2 = {x | 90 ≤ x ∧ x < 120} ∧
    A3 = {x | 75 ≤ x ∧ x < 90} ∧
    A4 = {x | 60 ≤ x ∧ x < 75} ∧
    A5 = {x | x < 60} ∧
    (conduct_frequency_distribution n [A1, A2, A3, A4, A5])

axiom conduct_frequency_distribution 
  (n : ℕ) (score_ranges : list (set ℕ)) : Prop

theorem necessary_work_to_determine_proportions
  : student_test_analysis := sorry

end necessary_work_to_determine_proportions_l401_401070


namespace ants_meet_at_q_one_l401_401874

noncomputable def ant_meeting_problem (q : ℚ) : Prop :=
  ∀ (n : ℕ), n > 0 →
  ∃ (ε ε' : Fin n → ℂ),
    ε ∈ {1, -1, Complex.i, -Complex.i} ∧
    ε' ∈ {1, -1, Complex.i, -Complex.i} ∧
    (∑ i in Finset.range n, ε i * q^i : ℂ) =
    (∑ i in Finset.range n, ε' i * q^i : ℂ) ∧
    ε ≠ ε'

theorem ants_meet_at_q_one : ∀ q : ℚ, 0 < q → ant_meeting_problem q → q = 1 := 
begin
  intros q hq h,
  sorry
end

end ants_meet_at_q_one_l401_401874


namespace sandy_final_fish_l401_401146

theorem sandy_final_fish :
  let Initial_fish := 26
  let Bought_fish := 6
  let Given_away_fish := 10
  let Babies_fish := 15
  let Final_fish := Initial_fish + Bought_fish - Given_away_fish + Babies_fish
  Final_fish = 37 :=
by
  sorry

end sandy_final_fish_l401_401146


namespace taller_building_height_l401_401975

theorem taller_building_height
  (H : ℕ) -- H is the height of the taller building
  (h_ratio : (H - 36) / H = 5 / 7) -- heights ratio condition
  (h_diff : H > 36) -- height difference must respect physics
  : H = 126 := sorry

end taller_building_height_l401_401975


namespace eccentricity_of_hyperbola_l401_401945

def hyperbola_eccentricity (a b c e : ℝ) : Prop :=
  a = 1 ∧ b = sqrt 3 ∧ c = sqrt (a^2 + b^2) ∧ e = c / a

theorem eccentricity_of_hyperbola : ∃ e : ℝ, ∀ a b c : ℝ, 
  (a = 1 → b = sqrt 3 → c = sqrt (a^2 + b^2) → e = c / a → e = 2) :=
begin
  use 2,
  intros a b c h1 h2 h3 h4,
  rw [h1, h2, h3],
  simp,
  exact h4,
end

end eccentricity_of_hyperbola_l401_401945


namespace max_x_minus_y_l401_401894

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^3 + y^3) = x + y) : x - y ≤ (Real.sqrt 2 / 2) :=
by {
  sorry
}

end max_x_minus_y_l401_401894


namespace parallel_to_a_l401_401671

-- Define the initial vector a
def a : ℝ × ℝ := (-5, 4)

-- Define the scalar multiple condition
def is_scalar_multiple (u v : ℝ × ℝ) (k : ℝ) : Prop :=
  u = (k * v.1, k * v.2)

-- Define the candidate vector and scalar
variables (k : ℝ)
def candidate : ℝ × ℝ := (-5 * k, 4 * k)

-- The theorem we want to prove
theorem parallel_to_a : ∃ k : ℝ, is_scalar_multiple candidate a k :=
  sorry

end parallel_to_a_l401_401671


namespace Chris_had_before_birthday_l401_401270

-- Define the given amounts
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Define the total birthday money received
def birthday_money : ℕ := grandmother_money + aunt_uncle_money + parents_money

-- Define the amount of money Chris had before his birthday
def money_before_birthday (total_now birthday_money : ℕ) : ℕ := total_now - birthday_money

-- Proposition to prove
theorem Chris_had_before_birthday : money_before_birthday total_money_now birthday_money = 159 := by
  sorry

end Chris_had_before_birthday_l401_401270


namespace fibonacci_expr_result_l401_401211

noncomputable def fibonacci_sequence : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := fibonacci_sequence (n + 1) + fibonacci_sequence n

def is_fibonacci_between (x : ℕ) (low high : ℕ) : Prop :=
  ∃ n, x = fibonacci_sequence n ∧ low < x ∧ x < high

theorem fibonacci_expr_result :
  ∀ a b: ℕ, 
  is_fibonacci_between a 49 61 → 
  is_fibonacci_between b 59 71 →
  a^3 + b^2 = 170096 := 
by
  sorry

end fibonacci_expr_result_l401_401211


namespace unique_handshakes_conference_l401_401967

theorem unique_handshakes_conference (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 5):
  let total_people := n * m in
  let handshakes_per_person := total_people - 1 - (m - 1) in
  let total_handshakes := total_people * handshakes_per_person / 2 in
  total_handshakes = 250 :=
by
  -- Define the specific numbers of people and companies
  rw [h₁, h₂]
  -- Simplify the expression for total_people
  have total_people_eq : total_people = 25 := by
    rw [mul_eq, h₁, h₂]
    exact rfl
  -- Simplify the expression for handshakes_per_person from the total_peoples calculation
  have handshakes_per_person_eq : handshakes_per_person = 20 := by
    rw [total_people_eq]
    simp
  -- Simplify the expression for total_handshakes from the total_handshakes per person calculation
  have total_handshakes_eq : total_handshakes = 25 * 20 / 2 := by
    rw [handshakes_per_person_eq]
    simp
  -- The final simplified answer should be 250
  exact rfl

end unique_handshakes_conference_l401_401967


namespace periodic_sequence_values_l401_401178

theorem periodic_sequence_values (a : ℝ) :
  (∀ n : ℕ, ∃ k : ℕ, ∀ m : ℕ, m > k → x_(n+m) = x_n) ↔ (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end periodic_sequence_values_l401_401178


namespace minFuseLength_l401_401849

namespace EarthquakeRelief

def fuseLengthRequired (distanceToSafety : ℕ) (speedOperator : ℕ) (burningSpeed : ℕ) (lengthFuse : ℕ) : Prop :=
  (lengthFuse : ℝ) / (burningSpeed : ℝ) > (distanceToSafety : ℝ) / (speedOperator : ℝ)

theorem minFuseLength 
  (distanceToSafety : ℕ := 400) 
  (speedOperator : ℕ := 5) 
  (burningSpeed : ℕ := 12) : 
  ∀ lengthFuse: ℕ, 
  fuseLengthRequired distanceToSafety speedOperator burningSpeed lengthFuse → lengthFuse > 96 := 
by
  sorry

end EarthquakeRelief

end minFuseLength_l401_401849


namespace geometric_sequence_general_term_l401_401405

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n
  
theorem geometric_sequence_general_term 
  (h_geo : is_geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  a n = 3 * 2^(n-3) :=
by sorry

end geometric_sequence_general_term_l401_401405


namespace find_range_a_l401_401785

theorem find_range_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, (1 ≤ x ∧ x ≤ 2) → (2 ≤ y ∧ y ≤ 3) → (xy ≤ a*x^2 + 2*y^2)) ↔ (-1/2 ≤ a) :=
sorry

end find_range_a_l401_401785


namespace quadratic_roots_difference_l401_401171

theorem quadratic_roots_difference (p q : ℤ) (h1 : ∀ x : ℝ, 2 * x^2 - 11 * x + 5 = 0 → x ∈ {5, 0.5})
  (h2 : is_integer q) (h3 : ∀ prime r, r * r ∣ p → false) : p + q = 83 := sorry

end quadratic_roots_difference_l401_401171


namespace max_diff_white_black_triangles_l401_401219

-- Define the conditions
def convex_N_gon (N : ℕ) : Prop := N ≥ 3

def triangulated (N : ℕ) (diagonals : set (ℕ × ℕ)) : Prop := 
  ∃ (triangles : set (fin N.succ) × (fin N.succ) × (fin N.succ)),
  (∀ (d ∈ diagonals), ∃ t1 t2 ∈ triangles, d ∈ t1 ∪ t2 ∧ t1 ≠ t2)

def correctly_colored (N : ℕ) (triangles : set (fin N.succ) × (fin N.succ) × (fin N.succ)) 
  (coloring : (fin N.succ) × (fin N.succ) × (fin N.succ) → bool) : Prop := 
  ∀ (t1 t2 ∈ triangles), (t1.1 = t2.2 ∨ t1.2 = t2.2 ∨ t1.3 = t2.2) → coloring t1 ≠ coloring t2

-- State the theorem
theorem max_diff_white_black_triangles (N : ℕ) (diagonals : set (ℕ × ℕ)) 
  (triangles : set (fin N.succ) × (fin N.succ) × (fin N.succ))
  (coloring : (fin N.succ) × (fin N.succ) × (fin N.succ) → bool) 
  (h_convex : convex_N_gon N) 
  (h_triangulated : triangulated N diagonals) 
  (h_correctly_colored : correctly_colored N triangles coloring) :
  let k := N / 3 in
  (N % 3 = 0 ∨ N % 3 = 2 → ∃ w b : ℕ, w - b = k) ∧ (N % 3 = 1 → ∃ w b : ℕ, w - b = k - 1) :=
sorry

end max_diff_white_black_triangles_l401_401219


namespace fourth_student_guess_l401_401563

theorem fourth_student_guess :
    let guess1 := 100
    let guess2 := 8 * guess1
    let guess3 := guess2 - 200
    let avg := (guess1 + guess2 + guess3) / 3
    let guess4 := avg + 25
    guess4 = 525 := by
    intros guess1 guess2 guess3 avg guess4
    have h1 : guess1 = 100 := rfl
    have h2 : guess2 = 8 * guess1 := rfl
    have h3 : guess3 = guess2 - 200 := rfl
    have h4 : avg = (guess1 + guess2 + guess3) / 3 := rfl
    have h5 : guess4 = avg + 25 := rfl
    simp [h1, h2, h3, h4, h5]
    sorry

end fourth_student_guess_l401_401563


namespace intersection_M_N_l401_401412

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2 * x - 3 ≤ 0}
def intersection : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l401_401412


namespace jenna_round_trip_pay_l401_401091

-- Definitions based on conditions
def rate : ℝ := 0.40
def one_way_distance : ℝ := 400
def round_trip_distance : ℝ := 2 * one_way_distance

-- Theorem based on the question and correct answer
theorem jenna_round_trip_pay : round_trip_distance * rate = 320 := by
  sorry

end jenna_round_trip_pay_l401_401091


namespace rose_needs_more_money_l401_401530

def budget : ℝ := 35.00
def sales_tax_rate : ℝ := 0.07
def shipping_cost : ℝ := 8.00
def paintbrush_price : ℝ := 2.40
def paints_price : ℝ := 9.20
def easel_price : ℝ := 6.50
def canvas_price : ℝ := 12.25
def drawing_pad_price : ℝ := 4.75
def discount_rate : ℝ := 0.15

def total_item_cost : ℝ := paintbrush_price + paints_price + easel_price + canvas_price + drawing_pad_price

noncomputable def discounted_total : ℝ := total_item_cost * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * sales_tax_rate
noncomputable def total_with_tax : ℝ := discounted_total + tax_amount
noncomputable def total_cost : ℝ := total_with_tax + shipping_cost

def additional_money_needed : ℝ := if total_cost > budget then total_cost - budget else 0

theorem rose_needs_more_money : total_cost = 39.93 ∧ additional_money_needed = 4.93 := by
  sorry

end rose_needs_more_money_l401_401530


namespace hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l401_401908

-- Definitions for the given problem
def one_hundred_million : ℕ := 100000000
def ten_million : ℕ := 10000000
def one_million : ℕ := 1000000
def ten_thousand : ℕ := 10000

-- Proving the statements
theorem hundred_million_is_ten_times_ten_million :
  one_hundred_million = 10 * ten_million :=
by
  sorry

theorem one_million_is_hundred_times_ten_thousand :
  one_million = 100 * ten_thousand :=
by
  sorry

end hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l401_401908


namespace fourth_power_sum_l401_401419

theorem fourth_power_sum
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 19.5 := 
sorry

end fourth_power_sum_l401_401419


namespace exists_multiple_with_sum_of_digits_eq_n_l401_401480

theorem exists_multiple_with_sum_of_digits_eq_n (n : ℕ) (h : n ≥ 1) : 
  ∃ m : ℕ, (m % n = 0) ∧ (nat.sum_of_digits m = n) :=
sorry

end exists_multiple_with_sum_of_digits_eq_n_l401_401480


namespace probability_B_not_losing_l401_401617

variable (P_A_w P_A_nl P_B_w P_B_nl p : ℝ)

-- Given conditions
def A_w := P_A_w = 0.4
def A_nl := P_A_nl = 0.8

-- Define the proof problem
theorem probability_B_not_losing :
  A_w → A_nl → (P_B_nl = (P_B_w + p)) → (p = P_A_nl - P_A_w) → 
  (P_B_w = 1 - P_A_nl) → 
  P_B_nl = 0.6 :=
by
  sorry

end probability_B_not_losing_l401_401617


namespace find_n_cosine_l401_401314

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401314


namespace percent_defective_units_shipped_for_sale_correct_l401_401854

noncomputable def percentage_defective_units_shipped_for_sale : ℝ := 
let units_produced := 100 in
let type_A_defective := 0.07 * units_produced in
let type_A_reworked := 0.40 * type_A_defective in
let type_A_remaining := type_A_defective - type_A_reworked in
let type_A_shipped := 0.03 * type_A_remaining in
let type_B_defective := 0.08 * units_produced in
let type_B_reworked := 0.30 * type_B_defective in
let type_B_remaining := type_B_defective - type_B_reworked in
let type_B_shipped := 0.06 * type_B_remaining in
let total_defective_shipped := type_A_shipped + type_B_shipped in
(total_defective_shipped / units_produced) * 100

theorem percent_defective_units_shipped_for_sale_correct :
  percentage_defective_units_shipped_for_sale = 0.462 := by
  sorry

end percent_defective_units_shipped_for_sale_correct_l401_401854


namespace find_n_cos_eq_l401_401343

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401343


namespace problem_statement_l401_401363

theorem problem_statement (m n : ℤ) (h : 3 * m - n = 1) : 9 * m ^ 2 - n ^ 2 - 2 * n = 1 := 
by sorry

end problem_statement_l401_401363


namespace roots_distinct_and_expression_integer_l401_401928

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x - 1

theorem roots_distinct_and_expression_integer :
  (∀ a b c : ℝ, f(a) = 0 → f(b) = 0 → f(c) = 0 →
    a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
  (∀ a b c : ℝ, f(a) = 0 → f(b) = 0 → f(c) = 0 →
    ∃ k : ℤ, (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a) ∧ 
      k = ( (a^1982 - b^1982) / (a - b) + (b^1982 - c^1982) / (b - c) + (c^1982 - a^1982) / (c - a) )) :=
by
  sorry

end roots_distinct_and_expression_integer_l401_401928


namespace initial_decaf_percent_l401_401658

theorem initial_decaf_percent (x : ℝ) : 
  let initialStock := 400
      newlyBoughtCoffee := 100
      newDecafPercent := 60 / 100
      totalStock := initialStock + newlyBoughtCoffee
      totalDecafPercent := 28.000000000000004 / 100
      initialDecafCoffee := (x / 100) * initialStock
      newDecafCoffee := newDecafPercent * newlyBoughtCoffee
      totalDecafCoffee := totalDecafPercent * totalStock
      remainingInitialDecaf := totalDecafCoffee - newDecafCoffee in
  (remainingInitialDecaf / initialStock) * 100 = 20 :=
by
  sorry

end initial_decaf_percent_l401_401658


namespace gcd_108_45_l401_401194

theorem gcd_108_45 : gcd 108 45 = 9 :=
by {
  -- Euclidean algorithm steps
  have step1 : 108 = 45 * 2 + 18 := by norm_num,
  have step2 : 45 = 18 * 2 + 9 := by norm_num,
  have step3 : 18 = 9 * 2 + 0 := by norm_num,
  -- Successive subtraction verification steps
  have sub1 : 108 - 45 = 63 := by norm_num,
  have sub2 : 63 - 45 = 18 := by norm_num,
  have sub3 : 45 - 18 = 27 := by norm_num,
  have sub4 : 27 - 18 = 9 := by norm_num,
  have sub5 : 18 - 9 = 9 := by norm_num,
  -- Intermediate results validate the final GCD
  apply nat.gcd_eq_of_sub_eq_right 108 45,
  sorry -- this is to skip the proof
}

end gcd_108_45_l401_401194


namespace circle_center_radius_l401_401239

theorem circle_center_radius (x y : ℝ) :
  (x^2 + y^2 + 4 * x - 6 * y = 11) →
  ∃ (h k r : ℝ), h = -2 ∧ k = 3 ∧ r = 2 * Real.sqrt 6 ∧
  (x+h)^2 + (y+k)^2 = r^2 :=
by
  sorry

end circle_center_radius_l401_401239


namespace ellipse_hyperbola_same_foci_l401_401169

theorem ellipse_hyperbola_same_foci (k : ℝ) (h1 : k > 0) :
  (∀ (x y : ℝ), (x^2 / 9 + y^2 / k^2 = 1) ↔ (x^2 / k - y^2 / 3 = 1)) → k = 2 :=
by
  sorry

end ellipse_hyperbola_same_foci_l401_401169


namespace matrix_series_product_l401_401302

theorem matrix_series_product :
  (List.foldl (λ (mat_acc : Matrix (Fin 2) (Fin 2) ℚ) (mat : Matrix (Fin 2) (Fin 2) ℚ), mat_acc ⬝ mat)
              (1 : Matrix (Fin 2) (Fin 2) ℚ)
              (List.map (λ a, ![![1, a], ![0, 1]]) (List.range' 2 50).map (λ x, 2 * (x + 1))))
  = ![![1, 2550], ![0, 1]] :=
by
  sorry

end matrix_series_product_l401_401302


namespace find_n_cosine_l401_401316

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401316


namespace max_competitors_l401_401238

theorem max_competitors (P1 P2 P3 : ℕ → ℕ → ℕ)
(hP1 : ∀ i, 0 ≤ P1 i ∧ P1 i ≤ 7)
(hP2 : ∀ i, 0 ≤ P2 i ∧ P2 i ≤ 7)
(hP3 : ∀ i, 0 ≤ P3 i ∧ P3 i ≤ 7)
(hDistinct : ∀ i j, i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :
  ∃ n, n ≤ 64 ∧ ∀ k, k < n → (∀ i j, i < k → j < k → i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :=
sorry

end max_competitors_l401_401238


namespace find_m_l401_401767

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l401_401767


namespace min_value_frac_sum_l401_401749

variable {a b c : ℝ}

theorem min_value_frac_sum (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b + b * c + c * a = 1) : 
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
  sorry

end min_value_frac_sum_l401_401749


namespace total_journey_length_l401_401531

theorem total_journey_length (y : ℚ)
  (h1 : y * 1 / 4 + 30 + y * 1 / 7 = y) : 
  y = 840 / 17 :=
by 
  sorry

end total_journey_length_l401_401531


namespace exists_unique_triangle_perimeters_l401_401619

open Classical

variables {O A B C : Point}

structure Line :=
  (origin : Point)
  (angle : Real)

noncomputable def in_half_line (P : Point) (L : Line) : Prop :=
  -- Hypothetical definition of being on a half-line
  sorry

noncomputable def perimeter (P1 P2 P3 : Point) : Real :=
  dist P1 P2 + dist P2 P3 + dist P3 P1

theorem exists_unique_triangle_perimeters
  (Ox Oy Oz : Line)
  (p : Real) (hp : p > 0) :
  ∃! (A B C : Point), in_half_line A Ox ∧ in_half_line B Oy ∧ in_half_line C Oz ∧
  perimeter O A B = 2 * p ∧ perimeter O B C = 2 * p ∧ perimeter O C A = 2 * p :=
begin
  sorry
end

end exists_unique_triangle_perimeters_l401_401619


namespace find_n_cos_eq_l401_401342

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401342


namespace race_distance_l401_401841

def race_distance_problem (V_A V_B T : ℝ) : Prop :=
  V_A * T = 218.75 ∧
  V_B * T = 193.75 ∧
  V_B * (T + 10) = 218.75 ∧
  T = 77.5

theorem race_distance (D : ℝ) (V_A V_B T : ℝ) 
  (h1 : V_A * T = D) 
  (h2 : V_B * T = D - 25) 
  (h3 : V_B * (T + 10) = D) 
  (h4 : V_A * T = 218.75) 
  (h5 : T = 77.5) 
  : D = 218.75 := 
by 
  sorry

end race_distance_l401_401841


namespace matrix_arithmetic_series_l401_401294

theorem matrix_arithmetic_series :
  ∏ (k : ℕ) in (range 50), (λ n => matrix.of !) (k.succ * 2) 0 0 1) = 
  matrix.of! 1 2550 0 1 :=
by
  sorry

end matrix_arithmetic_series_l401_401294


namespace infinitely_many_superabundant_l401_401464

noncomputable def sigma (n : ℕ) : ℕ :=
  (finset.range (n+1)).filter (λ d, n % d = 0).sum id

def superabundant (m : ℕ) : Prop :=
  ∀ k : ℕ, k < m → (sigma m) / m > (sigma k) / k

theorem infinitely_many_superabundant : ∃ᶠ m in filter.at_top, superabundant m :=
sorry

end infinitely_many_superabundant_l401_401464


namespace area_of_lemniscate_l401_401268

-- Define the polar equation for the Bernoulli lemniscate
def lemniscate (r a θ : ℝ) : Prop := r^2 = a^2 * cos (2 * θ)

-- Statement of the problem
theorem area_of_lemniscate (a : ℝ) : 
  ∫ (θ : ℝ) in 0..(2 * Real.pi), (1 / 2) * (a^2 * cos (2 * θ)) dθ = a^2 :=
sorry

end area_of_lemniscate_l401_401268


namespace ship_B_has_highest_rt_no_cars_l401_401503

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l401_401503


namespace distinct_x_intercepts_count_l401_401041

theorem distinct_x_intercepts_count :
  let f : ℝ → ℝ := λ x, (x - 5) * (x^2 + 6 * x + 10)
  ∃! x, f x = 0 := 
sorry

end distinct_x_intercepts_count_l401_401041


namespace value_of_x_squared_add_reciprocal_squared_l401_401801

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l401_401801


namespace taxi_fare_total_distance_l401_401824

theorem taxi_fare_total_distance (initial_fare additional_fare : ℝ) (total_fare : ℝ) (initial_distance additional_distance : ℝ) :
  initial_fare = 10 ∧ additional_fare = 1 ∧ initial_distance = 1/5 ∧ (total_fare = 59) →
  (total_distance = initial_distance + additional_distance * ((total_fare - initial_fare) / additional_fare)) →
  total_distance = 10 := 
by 
  sorry

end taxi_fare_total_distance_l401_401824


namespace dimes_max_diff_l401_401911

-- Definitions and conditions
def num_coins (a b c : ℕ) : Prop := a + b + c = 120
def coin_values (a b c : ℕ) : Prop := 5 * a + 10 * b + 50 * c = 1050
def dimes_difference (a1 a2 b1 b2 c1 c2 : ℕ) : Prop := num_coins a1 b1 c1 ∧ num_coins a2 b2 c2 ∧ coin_values a1 b1 c1 ∧ coin_values a2 b2 c2 ∧ a1 = a2 ∧ c1 = c2

-- Theorem statement
theorem dimes_max_diff : ∃ (a b1 b2 c : ℕ), dimes_difference a a b1 b2 c c ∧ b1 - b2 = 90 :=
by sorry

end dimes_max_diff_l401_401911


namespace complex_sum_l401_401056

theorem complex_sum (
  A : ℂ := 3 + 2 * complex.I,
  B : ℂ := -5,
  C : ℂ := 1 - 2 * complex.I,
  D : ℂ := 3 + 5 * complex.I
) : A + B + C + D = 2 + 5 * complex.I :=
by
  sorry

end complex_sum_l401_401056


namespace triangle_perimeter_eq_eight_l401_401442

theorem triangle_perimeter_eq_eight (A r s p : ℝ) (h₁ : A = 4 * r) (h₂ : A = r * s) : p = 8 :=
by
  -- Definitions of the variables
  let s := 4
  let p := 2 * s
  have : p = 8 := by
    -- Computations required
    show 2 * 4 = 8 from rfl
  exact this

end triangle_perimeter_eq_eight_l401_401442


namespace trapezoid_diagonal_intersection_eqdistance_l401_401251

variable (A B C D P Q O : Type)
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable [metric_space P] [metric_space Q] [metric_space O]

theorem trapezoid_diagonal_intersection_eqdistance
  (trapezoid_ABCD : is_trapezoid A B C D)
  (P_on_AD : P ∈ line_segment A D)
  (Q_on_BC : Q ∈ line_segment B C)
  (angle_eq_APB_CPD : ∠APB = ∠CPD)
  (angle_eq_AQB_CQD : ∠AQB = ∠CQD)
  (O_inter_diags : is_intersection_of_diagonals O A B C D)
  : dist P O = dist Q O := 
sorry

end trapezoid_diagonal_intersection_eqdistance_l401_401251


namespace min_employees_needed_l401_401257

theorem min_employees_needed
  (W A S : Finset ℕ)
  (hW : W.card = 120)
  (hA : A.card = 150)
  (hS : S.card = 100)
  (hWA : (W ∩ A).card = 50)
  (hAS : (A ∩ S).card = 30)
  (hWS : (W ∩ S).card = 20)
  (hWAS : (W ∩ A ∩ S).card = 10) :
  (W ∪ A ∪ S).card = 280 :=
by
  sorry

end min_employees_needed_l401_401257


namespace inequality_l401_401033

noncomputable def a : ℝ := log 3 (sqrt 3)
noncomputable def b : ℝ := log 4 (sqrt 3)
noncomputable def c : ℝ := (0.3 : ℝ)⁻¹

theorem inequality : c > a ∧ a > b := by
  sorry

end inequality_l401_401033


namespace total_weight_of_nuts_l401_401652

theorem total_weight_of_nuts :
  let almonds := 0.14
      pecans := 0.38
      walnuts := 0.22
      cashews := 0.47
      pistachios := 0.29
      brazil_nuts_oz := 6.0
      macadamia_nuts_oz := 4.5
      hazelnuts_oz := 7.3
      ounce_to_kg := 0.0283495
      brazil_nuts := brazil_nuts_oz * ounce_to_kg
      macadamia_nuts := macadamia_nuts_oz * ounce_to_kg
      hazelnuts := hazelnuts_oz * ounce_to_kg
  in almonds + pecans + walnuts + cashews + pistachios + brazil_nuts + macadamia_nuts + hazelnuts = 2.1128216 :=
sorry

end total_weight_of_nuts_l401_401652


namespace simplify_vector_expression_l401_401795

variables (a b : Type) [vector_space ℝ a] [vector_space ℝ b]

theorem simplify_vector_expression (a b : ℝ) : 
  (1/2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end simplify_vector_expression_l401_401795


namespace find_a_l401_401391

theorem find_a (a : ℝ) : 
  (∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
  ∃ y, y ≤ 6 ∧ 
  (∀ x, x^2 + y^2 = a^2 ∧ 
  x^2 + y^2 + a * y - 6 = 0)) → 
  a = 2 ∨ a = -2 :=
by sorry

end find_a_l401_401391


namespace inequality_am_gm_l401_401524

theorem inequality_am_gm 
  (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := 
by 
  sorry

end inequality_am_gm_l401_401524


namespace scientific_notation_of_116_million_l401_401830

theorem scientific_notation_of_116_million : 116000000 = 1.16 * 10^7 :=
sorry

end scientific_notation_of_116_million_l401_401830


namespace ken_gets_back_18_60_l401_401683

theorem ken_gets_back_18_60 :
  let steak_price := 7
  let sale_discount := 0.5
  let steak_quantity := 2
  let eggs_price := 3
  let milk_price := 4
  let bagels_price := 6
  let sales_tax_rate := 0.07
  let payment := 20 + 10 + 10 + 3 + 0.75
  let steak_cost := steak_price + steak_price * sale_discount
  let other_items_cost := eggs_price + milk_price + bagels_price
  let subtotal := steak_cost + other_items_cost
  let sales_tax := subtotal * sales_tax_rate
  let total_cost := subtotal + sales_tax
  in payment - total_cost = 18.60 :=
by
  sorry

end ken_gets_back_18_60_l401_401683


namespace senate_seating_l401_401647

-- Definitions for the problem
def num_ways_of_seating (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  if h : num_democrats = 6 ∧ num_republicans = 4 then
    5! * (finset.card (finset.powerset_len 4 (finset.range 6))) * 4!
  else
    0

-- The proof statement
theorem senate_seating : num_ways_of_seating 6 4 = 43200 :=
by {
  -- Placeholder for proof
  sorry
}

end senate_seating_l401_401647


namespace conditional_probability_P_B_given_A_l401_401743

open Classical

noncomputable def S : Finset ℕ := {1, 2, 3, 4, 5}

def eventA : Set (ℕ × ℕ) := {p | p.1 + p.2 ∈ S ∧ (p.1 + p.2) % 2 = 0 ∧ p.1 < p.2}
def eventB : Set (ℕ × ℕ) := {p | p.1 ∈ {2, 4} ∧ p.2 ∈ {2, 4} ∧ p.1 < p.2}

def P_eventA : ℝ := ↑((3.choose 2 + 2.choose 2) : ℕ) / (5.choose 2)
def P_eventB : ℝ := ↑((2.choose 2) : ℕ) / (5.choose 2)

theorem conditional_probability_P_B_given_A : 
  P (eventB) * P (eventA) =
  (2.choose 2 : ℝ) * (5.choose 2) :=
sorry

end conditional_probability_P_B_given_A_l401_401743


namespace road_building_equation_l401_401977

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end road_building_equation_l401_401977


namespace vector_identity_l401_401747

def vec_a : ℝ × ℝ := (2, 2)
def vec_b : ℝ × ℝ := (-1, 3)

theorem vector_identity : 2 • vec_a - vec_b = (5, 1) := by
  sorry

end vector_identity_l401_401747


namespace find_b_l401_401579

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.fst + p2.fst) / 2, (p1.snd + p2.snd) / 2)

theorem find_b (p1 p2 : ℝ × ℝ) (h : p1 = (2, 4) ∧ p2 = (6, 8)) (M : midpoint p1 p2 = (4, 6)) :
  ∃ b : ℝ, ∀ x y : ℝ, (x, y) = M → (x + y = b) := 
begin
  use 10,
  intros x y hxy,
  rw hxy,
  exact rfl,
end

end find_b_l401_401579


namespace fill_with_L_blocks_l401_401196

variable (m n k : ℕ)
variable (h_m : m > 1) (h_n : n > 1) (h_k : k > 1)
variable (h_div : m * n * k % 3 = 0)

theorem fill_with_L_blocks :
  ∃ (L : ℕ × ℕ × ℕ → Prop),
    (∀ x y z, L (x, y, z) → x < m ∧ y < n ∧ z < k) ∧
    (∑ x in finset.range m, ∑ y in finset.range n, ∑ z in finset.range k, if L (x, y, z) then 1 else 0) = m * n * k :=
sorry

end fill_with_L_blocks_l401_401196


namespace exists_fifth_degree_poly_neg_roots_pos_derivative_roots_l401_401459

theorem exists_fifth_degree_poly_neg_roots_pos_derivative_roots :
  ∃ P : ℝ[X], P.degree = 5 ∧ 
             (∀ x, (P.derivative.eval x = 0 → x > 0)) ∧ 
             (∀ x, (P.eval x = 0 → x < 0)) ∧ 
             (∃ x, P.eval x = 0) ∧ 
             (∃ x, P.derivative.eval x = 0) :=
by
  sorry

end exists_fifth_degree_poly_neg_roots_pos_derivative_roots_l401_401459


namespace persons_initially_l401_401938

theorem persons_initially (n : ℕ) 
  (avg_increase : n * 2.5 = 20)
  (new_weight : 90)
  (old_weight : 70)
  (weight_difference : new_weight - old_weight = 20) : n = 8 := 
by
  sorry

end persons_initially_l401_401938


namespace find_n_cos_eq_l401_401341

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401341


namespace area_of_triangle_formula_l401_401927

theorem area_of_triangle_formula (
    { R r A B C : ℝ }
    (h1: 0 < R)
    (h2: 0 < r)
    (h_A: 0 < A ∧ A < π)
    (h_B: 0 < B ∧ B < π)
    (h_C: 0 < C ∧ C < π)
    (h_sum: A + B + C = π)) :
    let T := R * r * (sin A + sin B + sin C) in
    let T_alt := 2 * R^2 * sin A * sin B * sin C in
    (T = T ∧ T_alt = T_alt) := by
    sorry

end area_of_triangle_formula_l401_401927


namespace daniel_biked_more_l401_401167

def miles_biked_after_4_hours_more (speed_plain_daniel : ℕ) (speed_plain_elsa : ℕ) (time_plain : ℕ) 
(speed_hilly_daniel : ℕ) (speed_hilly_elsa : ℕ) (time_hilly : ℕ) : ℕ :=
(speed_plain_daniel * time_plain + speed_hilly_daniel * time_hilly) - 
(speed_plain_elsa * time_plain + speed_hilly_elsa * time_hilly)

theorem daniel_biked_more : miles_biked_after_4_hours_more 20 18 3 16 15 1 = 7 :=
by
  sorry

end daniel_biked_more_l401_401167


namespace min_positive_announcements_l401_401684

theorem min_positive_announcements (x y : ℕ) (h1 : y * (y - 1) + (x - y) * (x - y - 1) = 62)
  (ha : x * (x - 1) = 132) : y = 5 := by
  have hx : x = 12 := by {
    -- Solving quadratic equation x^2 - x - 132 = 0
    sorry
  }
  have hquad : 2 * y^2 - 24 * y + 132 = 62 := by {
    -- Begin manipulating the given equation and solve the quadratic formula
    sorry
  }
  have hy_values : y = 7 ∨ y = 5 := by {
    -- Solution of simplified quadratic equation and its roots
    sorry
  }
  exact hy_values.elim (λ h : y = 7, by { have := h, contradiction }) (λ h : y = 5, by { exact h })

end min_positive_announcements_l401_401684


namespace range_of_a_l401_401407

theorem range_of_a (a : ℝ) : 
  (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ x1 ∈ set.Ioi a ∧ x2 ∈ set.Ioi a ∧ ((x1 : ℝ)^2 - (a+1)*x1 + a < 0) ∧ ((x2 : ℝ)^2 - (a+1)*x2 + a < 0) ∧ 
                  (∀ x : ℤ, (x : ℝ)^2 - (a+1)*x + a < 0 → x = x1 ∨ x = x2)) ↔ 
  (a ∈ set.Icc (-2 : ℝ) (-1 : ℝ) ∨ a ∈ set.Ioc (3 : ℝ) (4 : ℝ)) :=
sorry

end range_of_a_l401_401407


namespace johns_father_age_l401_401461

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age_l401_401461


namespace express_in_scientific_notation_l401_401132

-- Definitions based on problem conditions
def GDP_first_quarter : ℝ := 27017800000000

-- Main theorem statement that needs to be proved
theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), (GDP_first_quarter = a * 10 ^ b) ∧ (a = 2.70178) ∧ (b = 13) :=
by
  sorry -- Placeholder to indicate the proof is omitted

end express_in_scientific_notation_l401_401132


namespace lines_MN_pass_through_excenter_l401_401492

-- Definition for tangency and points basics
variables {A B C D E F P Q M N I_A : Point} {ABC : Triangle}

-- Conditions given:
--    - PQ is tangent to the incircle of triangle ABC at T
--    - P lies on AB, Q lies on AC
--    - AM = BP and AN = CQ
-- We need to show that MN passes through the excenter IA of triangle ABC 

noncomputable def tangent_line_condition (ABC : Triangle) (P Q : Point) := 
  tangent_to_incircle PQ ABC ∧ 
  lies_on_side PQ PQ AB AC

noncomputable def point_relation_condition (A B C P Q M N : Point) := 
  lies_on_side P AB ∧ 
  lies_on_side Q AC ∧ 
  AM = BP ∧ 
  AN = CQ

theorem lines_MN_pass_through_excenter
  (ABC : Triangle) (PQ : Line) (P Q M N I_A : Point) 
  (tangent_condition : tangent_line_condition ABC P Q)
  (point_relation : point_relation_condition A B C P Q M N) : 
  passes_through MN I_A :=
sorry

end lines_MN_pass_through_excenter_l401_401492


namespace andrea_rhinestones_needed_l401_401679

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l401_401679


namespace derivative_y_eq_l401_401732

noncomputable def y (x : ℝ) : ℝ := 
  (3 / 2) * Real.log (Real.tanh (x / 2)) + Real.cosh x - (Real.cosh x) / (2 * (Real.sinh x)^2)

theorem derivative_y_eq :
  (deriv y x) = (Real.cosh x)^4 / (Real.sinh x)^3 :=
sorry

end derivative_y_eq_l401_401732


namespace limit_fraction_nk_over_an_l401_401633

theorem limit_fraction_nk_over_an (k : ℕ) (a : ℝ) (h : 1 < a) : 
  Filter.Tendsto (fun n : ℕ => (n^k) / (a^n)) Filter.atTop (𝓝 0) := 
sorry

end limit_fraction_nk_over_an_l401_401633


namespace fraction_boxes_loaded_by_day_crew_l401_401215

variables {D W_d : ℝ}

theorem fraction_boxes_loaded_by_day_crew
  (h1 : ∀ (D W_d: ℝ), D > 0 → W_d > 0 → ∃ (D' W_n : ℝ), (D' = 0.5 * D) ∧ (W_n = 0.8 * W_d))
  (h2 : ∃ (D W_d : ℝ), ∀ (D' W_n : ℝ), (D' = 0.5 * D) → (W_n = 0.8 * W_d) → 
        (D * W_d / (D * W_d + D' * W_n)) = (5 / 7)) :
  (∃ (D W_d : ℝ), D > 0 → W_d > 0 → (D * W_d)/(D * W_d + 0.5 * D * 0.8 * W_d) = (5/7)) := 
  sorry 

end fraction_boxes_loaded_by_day_crew_l401_401215


namespace meeting_point_l401_401282

theorem meeting_point (A J : ℝ × ℝ) (hA : A = (8, -3)) (hJ : J = (2, 7)) :
  let midpoint : ℝ × ℝ := ((A.1 + J.1) / 2, (A.2 + J.2) / 2) in midpoint = (5, 2) :=
by
  sorry

end meeting_point_l401_401282


namespace second_half_takes_200_percent_longer_l401_401901

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l401_401901


namespace car_wheels_l401_401831

theorem car_wheels (num_cars : ℕ) (wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  -- Using the given conditions to prove the statement
  rw [h1, h2]
  norm_num

end car_wheels_l401_401831


namespace midpoint_DE_on_CF_l401_401863

open EuclideanGeometry

-- Defining initial points for the parallelogram ABCD
variables (A B C D E F M : Point)
-- Assuming the properties of the points and the parallelogram
variables (parallelogram_ABCD : Parallelogram A B C D)
          (E_on_bisector_A : E ∈ bisector A B D)
          (F_on_bisector_C : F ∈ bisector C B D)
          (O_mid_BF_lies_on_AE : midpoint O B F ∧ O ∈ lineThrough A E)

-- Defining midpoint of DE
def midpoint_DE_lies_on_CF (mid_DE_lies_CF : midpoint M D E ∈ lineThrough C F) : Prop :=
  sorry

-- The main statement
theorem midpoint_DE_on_CF :
  parallelogram A B C D →
  E ∈ bisector A B D →
  F ∈ bisector C B D →
  midpoint O B F ∧ O ∈ lineThrough A E →
  midpoint M D E ∈ lineThrough C F :=
sorry

end midpoint_DE_on_CF_l401_401863


namespace proof_ARML_value_l401_401875

noncomputable def ARML_value (A R M L : ℝ) : Prop :=
  ARML = 256 := sorry

theorem proof_ARML_value
    (A R M L : ℝ) (h1 : log 2 (A * L) + log 2 (A * M) = 6)
    (h2 : log 2 (M * L) + log 2 (M * R) = 8)
    (h3 : log 2 (R * A) + log 2 (R * L) = 10) : ARML_value A R M L :=
  sorry

end proof_ARML_value_l401_401875


namespace sum_of_solutions_l401_401993

theorem sum_of_solutions :
  let eq := (4 * x + 3) * (3 * x - 7) = 0 in
  (is_solution eq (-3/4) ∧ is_solution eq (7/3)) → 
  (-3 / 4 + 7 / 3 = 19 / 12) :=
by 
  intros eq h
  sorry

end sum_of_solutions_l401_401993


namespace ezekiel_first_day_distance_l401_401724

noncomputable def distance_first_day (total_distance second_day_distance third_day_distance : ℕ) :=
  total_distance - (second_day_distance + third_day_distance)

theorem ezekiel_first_day_distance:
  ∀ (total_distance second_day_distance third_day_distance : ℕ),
  total_distance = 50 →
  second_day_distance = 25 →
  third_day_distance = 15 →
  distance_first_day total_distance second_day_distance third_day_distance = 10 :=
by
  intros total_distance second_day_distance third_day_distance h1 h2 h3
  sorry

end ezekiel_first_day_distance_l401_401724


namespace symmetric_point_l401_401375

theorem symmetric_point (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : |y| = 3) : 
  (2, -3) = (-x, -y) :=
sorry

end symmetric_point_l401_401375


namespace find_smallest_prime_with_properties_l401_401352

-- Definitions
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100
def has_three_as_tens_digit (n : ℕ) : Prop := (n / 10) % 10 = 3
def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)
def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ nat.prime n

-- Statement of the problem
theorem find_smallest_prime_with_properties :
  ∃ n : ℕ, is_two_digit n ∧ is_prime n ∧ has_three_as_tens_digit n ∧ is_composite (reverse_digits n) ∧
           ∀ m : ℕ, is_two_digit m → is_prime m → has_three_as_tens_digit m → is_composite (reverse_digits m) → n <= m := 
begin
  sorry -- Proof goes here.
end

end find_smallest_prime_with_properties_l401_401352


namespace sharon_distance_to_mothers_house_l401_401926

noncomputable def total_distance (x : ℝ) :=
  x / 240

noncomputable def adjusted_speed (x : ℝ) :=
  x / 240 - 1 / 4

theorem sharon_distance_to_mothers_house (x : ℝ) (h1 : x / 240 = total_distance x) 
(h2 : adjusted_speed x = x / 240 - 1 / 4) 
(h3 : 120 + 120 * x / (x - 60) = 330) : 
x = 140 := 
by 
  sorry

end sharon_distance_to_mothers_house_l401_401926


namespace hyperbola_eccentricity_l401_401709

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptotes_l1: ∀ x : ℝ, y = (b / a) * x)
  (h_asymptotes_l2: ∀ x : ℝ, y = -(b / a) * x)
  (h_focus: c^2 = a^2 + b^2)
  (h_symmetric: ∀ m : ℝ, m = -c / 2 ∧ (m, (b * c) / (2 * a)) ∈ { p : ℝ × ℝ | p.2 = -(b / a) * p.1 }) :
  (c / a) = 2 := sorry

end hyperbola_eccentricity_l401_401709


namespace square_area_in_segment_l401_401071

-- Define various parameters and conditions
def radius : ℝ := 2 * Real.sqrt 3 + Real.sqrt 17
def angle : ℝ := 60 / 180 * Real.pi  -- Convert degrees to radians

-- Main problem statement: Prove that the area of the inscribed square is 1
theorem square_area_in_segment : ∃ (x : ℝ), (0 < x) ∧ (x^2 = 1) :=
  sorry

end square_area_in_segment_l401_401071


namespace line_BC_eq_line_AD_eq_altitude_AD_eq_l401_401847

noncomputable def point (x y : ℝ) := (x, y)

def A : ℝ × ℝ := point 2 1
def B : ℝ × ℝ := point (-2) 3
def C : ℝ × ℝ := point (-3) 0

def line_eq (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem line_BC_eq : ∃ k b, (∀ x y, ((x, y) = B ∨ (x, y) = C) → line_eq k b x y) ∧ 
  (∀ k' b', (∀ x y, ((x, y) = B ∨ (x, y) = C) → line_eq k' b' x y) → k' = 3 ∧ b' = 9) :=
begin
  sorry
end

theorem line_AD_eq : ∃ k b, (∀ x y, ((x, y) = A) → line_eq k b x y) ∧ 
  k = -1/3 ∧ b = 5/3 :=
begin
  sorry
end

theorem altitude_AD_eq : ∀ x y, ((x, y) = A ∨ ∃ k b, (∀ x' y', ((x', y') = B ∨ (x', y') = C) → line_eq k b x' y') ∧ 
  (line_eq (-1/3) (5/3) x y)) → 
  x + 3 * y - 5 = 0 :=
begin
  sorry
end

end line_BC_eq_line_AD_eq_altitude_AD_eq_l401_401847


namespace systematic_sampling_correct_l401_401611

-- Definitions for the conditions
def total_products := 60
def group_count := 5
def products_per_group := total_products / group_count

-- systematic sampling condition: numbers are in increments of products_per_group
def systematic_sample (start : ℕ) (count : ℕ) : List ℕ := List.range' start products_per_group count

-- Given sequences
def A : List ℕ := [5, 10, 15, 20, 25]
def B : List ℕ := [5, 12, 31, 39, 57]
def C : List ℕ := [5, 17, 29, 41, 53]
def D : List ℕ := [5, 15, 25, 35, 45]

-- Correct solution defined
def correct_solution := [5, 17, 29, 41, 53]

-- Problem Statement
theorem systematic_sampling_correct :
  systematic_sample 5 group_count = correct_solution :=
by
  sorry

end systematic_sampling_correct_l401_401611


namespace find_f_3_l401_401816

-- Defining the function f with the given condition
def f : ℝ → ℝ := sorry

-- The main theorem stating the required proof problem
theorem find_f_3 : (∀ x : ℝ, f (10^x) = x) → f 3 = log 10 3 :=
sorry

end find_f_3_l401_401816


namespace largest_element_in_set_l401_401052

theorem largest_element_in_set (a : ℝ) (h : a = 3) : 
  let elements := { -3 * a, 4 * a, 24 / a, a^2, 1 } in
  ∃ (x : ℝ), x ∈ elements ∧ ∀ y ∈ elements, y ≤ x ∧ x = 4 * a :=
sorry

end largest_element_in_set_l401_401052


namespace jenna_round_trip_pay_l401_401089

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l401_401089


namespace max_quarters_l401_401925

theorem max_quarters (q : ℕ) (h1 : q + q + q / 2 = 20): q ≤ 11 :=
by
  sorry

end max_quarters_l401_401925


namespace train_cross_platform_time_l401_401645

def train_length : ℝ := 300
def platform_length : ℝ := 550
def signal_pole_time : ℝ := 18

theorem train_cross_platform_time :
  let speed : ℝ := train_length / signal_pole_time
  let total_distance : ℝ := train_length + platform_length
  let crossing_time : ℝ := total_distance / speed
  crossing_time = 51 :=
by
  sorry

end train_cross_platform_time_l401_401645


namespace sarahs_profit_l401_401532

/-- Sarah's profit calculation. -/
theorem sarahs_profit : 
  let loaves := 60,
  morning_price := 3.00,
  afternoon_price := 1.50,
  evening_price := 1.00,
  cost := 1.00,
  morning_sales := (1 / 3) * loaves,
  remaining_after_morning := loaves - morning_sales,
  afternoon_sales := (3 / 4) * remaining_after_morning,
  remaining_after_afternoon := remaining_after_morning - afternoon_sales,
  evening_sales := remaining_after_afternoon,
  total_revenue := (morning_sales * morning_price) + (afternoon_sales * afternoon_price) + (evening_sales * evening_price),
  total_cost := loaves * cost,
  profit := total_revenue - total_cost 
  in 
  profit = 55 := 
by
  sorry

end sarahs_profit_l401_401532


namespace five_digit_even_numbers_count_l401_401040

theorem five_digit_even_numbers_count : 
  (∃ (f : Fin 5 → Fin 5), 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (f 4 = 1 ∨ f 4 = 3) ∧ 
    ∀ i, f i ≠ f 4) → 
  48 :=
by
  sorry

end five_digit_even_numbers_count_l401_401040


namespace laundry_ratio_l401_401904

variables (kilos_two_days_ago kilos_yesterday kilos_today : ℕ)
variables (earning_per_kilo total_earning : ℕ)

-- Conditions
def condition_1 : earning_per_kilo = 2 := rfl
def condition_2 : kilos_two_days_ago = 5 := rfl
def condition_3 : kilos_yesterday = kilos_two_days_ago + 5 := rfl
def condition_4 : total_earning = 70 := rfl
def condition_5 : 2 * kilos_two_days_ago + 2 * (kilos_two_days_ago + 5) + 2 * kilos_today = total_earning := rfl

-- Theorem statement
theorem laundry_ratio (h1 : earning_per_kilo = 2)
    (h2 : kilos_two_days_ago = 5)
    (h3 : kilos_yesterday = kilos_two_days_ago + 5)
    (h4 : total_earning = 70)
    (h5 : 2 * kilos_two_days_ago + 2 * (kilos_two_days_ago + 5) + 2 * kilos_today = total_earning) :
    kilos_today / kilos_yesterday = 2 :=
sorry

end laundry_ratio_l401_401904


namespace min_distance_point_to_line_l401_401429

theorem min_distance_point_to_line 
  (m n : ℝ) (h : 4 * m + 3 * n = 10) : m^2 + n^2 ≥ 4 := 
begin
  sorry
end

end min_distance_point_to_line_l401_401429


namespace inverse_function_of_f_l401_401170

noncomputable def f (x : ℝ) : ℝ := x^2

theorem inverse_function_of_f :
  ∀ x, x > 4 → f⁻¹(x) = -√x :=
by
  sorry

end inverse_function_of_f_l401_401170


namespace area_of_square_l401_401500

variables (A B C D E F : Type)
variables [InnerProductSpace ℝ E]

noncomputable def is_square (ABCD : E → Prop) : Prop :=
  ∃ (x : ℝ), ∀ (A B C D : E), 
  dist A B = x ∧
  dist B C = x ∧
  dist C D = x ∧
  dist D A = x ∧
  ∠ A B C = 90 ∧
  ∠ B C D = 90 ∧
  ∠ C D A = 90 ∧
  ∠ D A B = 90

variables (ABCD : E → Prop) [h : is_square ABCD]

noncomputable def side_length (x : ℝ) (E F : E) :=
  dist E F = x / sqrt 2

noncomputable def find_area_of_square (x : ℝ) : ℝ :=
  x * x

theorem area_of_square (BE EF FD : E) (b_dist : dist E F = 20) (f_dist : dist F B = 20) :
  ∃ (x : ℝ), find_area_of_square x = 800 :=
begin
  sorry
end

end area_of_square_l401_401500


namespace probability_lt_8000_correct_l401_401166

noncomputable def num_pairs : ℕ := nat.choose 5 2

def distances : list ℕ := [
  6300, 6609, 5944, 2850, -- Distances involving Bangkok
  11535, 5989, 13714, -- Distances involving Cape Town
  7240, 3876, -- Distances involving Honolulu
  5959 -- Distance involving London
  -- No need for Tokyo entries as they are already considered in previous pairs
]

def count_lt_8000 (l : list ℕ) : ℕ := l.countp (λ d, d < 8000)

def probability : ℚ := (count_lt_8000 distances) / num_pairs

theorem probability_lt_8000_correct :
  probability = 7 / 10 :=
by
  sorry

end probability_lt_8000_correct_l401_401166


namespace bulbs_remaining_l401_401093

theorem bulbs_remaining
    (led_initial : ℕ) (inc_initial : ℕ)
    (led_used : ℕ) (inc_used : ℕ)
    (led_to_alex_ratio : ℚ) (inc_to_bob_ratio : ℚ)
    (led_to_charlie_ratio : ℚ) (inc_to_charlie_ratio : ℚ)
    (led_initial = 24) (inc_initial = 16)
    (led_used = 10) (inc_used = 6)
    (led_to_alex_ratio = 1/2) (inc_to_bob_ratio = 1/4)
    (led_to_charlie_ratio = 1/5) (inc_to_charlie_ratio = 3/10) :
    let led_remaining_initial := led_initial - led_used in
    let inc_remaining_initial := inc_initial - inc_used in
    let led_remaining_alex := led_remaining_initial - (led_to_alex_ratio * led_remaining_initial).nat_abs in
    let inc_remaining_bob := inc_remaining_initial - (inc_to_bob_ratio * inc_remaining_initial).nat_abs in
    let led_remaining_charlie := led_remaining_alex - (led_to_charlie_ratio * led_remaining_alex).nat_abs in
    let inc_remaining_charlie := inc_remaining_bob - (inc_to_charlie_ratio * inc_remaining_bob).nat_abs in
    led_remaining_charlie = 6 ∧ inc_remaining_charlie = 6 := by
  sorry

end bulbs_remaining_l401_401093


namespace second_integer_in_sequence_l401_401962

theorem second_integer_in_sequence (h₀ : 68 * 4 + 2 = 274) : 
  ∃ n : ℤ, 4 * n + 2 = 274 ∧ n = 68 := 
by {
  use 68,
  split,
  exact h₀,
  refl,
}

end second_integer_in_sequence_l401_401962


namespace sequence_contains_infinitely_many_powers_of_3_l401_401526

theorem sequence_contains_infinitely_many_powers_of_3 :
  ∃ᶠ n in (at_top : Filter ℕ), ∃ k : ℕ, a_n = (3 : ℕ)^k := by
  let a_n := λ n : ℕ, ⌊(n : ℝ) * Real.sqrt 2⌋
  sorry

end sequence_contains_infinitely_many_powers_of_3_l401_401526


namespace find_AD_length_l401_401445

variables (A B C D O : Point)
variables (BO OD AO OC AB AD : ℝ)

def quadrilateral_properties (BO OD AO OC AB : ℝ) (O : Point) : Prop :=
  BO = 3 ∧ OD = 9 ∧ AO = 5 ∧ OC = 2 ∧ AB = 7

theorem find_AD_length (h : quadrilateral_properties BO OD AO OC AB O) : AD = Real.sqrt 151 :=
by
  sorry

end find_AD_length_l401_401445


namespace exists_p_for_q_l401_401535

noncomputable def sqrt_56 : ℝ := Real.sqrt 56
noncomputable def sqrt_58 : ℝ := Real.sqrt 58

theorem exists_p_for_q (q : ℕ) (hq : q > 0) (hq_ne_1 : q ≠ 1) (hq_ne_3 : q ≠ 3) :
  ∃ p : ℤ, sqrt_56 < (p : ℝ) / q ∧ (p : ℝ) / q < sqrt_58 :=
by sorry

end exists_p_for_q_l401_401535


namespace average_weight_entire_class_l401_401609

def avg_weight_of_section (num_students : ℕ) (avg_weight : ℝ) : ℝ :=
  num_students * avg_weight

def total_weight_of_class 
  (num_students_A num_students_B num_students_C num_students_D : ℕ) 
  (avg_weight_A avg_weight_B avg_weight_C avg_weight_D : ℝ) : ℝ :=
  avg_weight_of_section num_students_A avg_weight_A +
  avg_weight_of_section num_students_B avg_weight_B +
  avg_weight_of_section num_students_C avg_weight_C +
  avg_weight_of_section num_students_D avg_weight_D

theorem average_weight_entire_class :
  let num_students_A := 40
      num_students_B := 25
      num_students_C := 35
      num_students_D := 20
      avg_weight_A := 50
      avg_weight_B := 40
      avg_weight_C := 55
      avg_weight_D := 45
  in
  total_weight_of_class num_students_A num_students_B num_students_C num_students_D avg_weight_A avg_weight_B avg_weight_C avg_weight_D / 
  (num_students_A + num_students_B + num_students_C + num_students_D) = 48.54 :=
by
  let num_students_A := 40
  let num_students_B := 25
  let num_students_C := 35
  let num_students_D := 20
  let avg_weight_A := 50
  let avg_weight_B := 40
  let avg_weight_C := 55
  let avg_weight_D := 45
  let total_weight := total_weight_of_class num_students_A num_students_B num_students_C num_students_D avg_weight_A avg_weight_B avg_weight_C avg_weight_D
  let total_students := (num_students_A + num_students_B + num_students_C + num_students_D)
  have h : total_weight / total_students = 48.54 := sorry
  exact h

end average_weight_entire_class_l401_401609


namespace number_of_arrangements_l401_401797

theorem number_of_arrangements (letters : List Char) (arrangements : List (List Char)) : 
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] ∧ 
  ∀ arrangement ∈ arrangements, arrangement.length = 4 ∧ 
    arrangement.head? = some 'D' ∧ 
    'E' ∈ arrangement ∧ 
    arrangement.nodup → 
  arrangements.length = 60 :=
by
  sorry

end number_of_arrangements_l401_401797


namespace express_2011_with_digit_1_l401_401446

theorem express_2011_with_digit_1 :
  ∃ (a b c d e: ℕ), 2011 = a * b - c * d + e - f + g ∧
  (a = 1111 ∧ b = 1111) ∧ (c = 111 ∧ d = 11111) ∧ (e = 1111) ∧ (f = 111) ∧ (g = 11) ∧
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g) :=
sorry

end express_2011_with_digit_1_l401_401446


namespace evaluate_exponent_l401_401428

theorem evaluate_exponent (m n : ℤ) (h1 : m = 3) (h2 : n = 2) : (-n)^m = -8 :=
  by
  sorry

end evaluate_exponent_l401_401428


namespace sum_of_all_possible_two_digit_values_of_d_l401_401108

theorem sum_of_all_possible_two_digit_values_of_d :
  (∑ d in {d : ℕ | d ∣ (143 - 5) ∧ 10 ≤ d ∧ d < 100}, d) = 115 :=
by
  sorry

end sum_of_all_possible_two_digit_values_of_d_l401_401108


namespace prove_g_satisfies_l401_401016

def f (x : ℝ) : ℝ := if x.is_rat then 1 else 0

def g (x : ℝ) : ℝ := abs x

theorem prove_g_satisfies (x : ℝ) : x - f x ≤ g x :=
by
  sorry

end prove_g_satisfies_l401_401016


namespace fourth_student_guess_l401_401564

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l401_401564


namespace cos_eq_43_l401_401318

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401318


namespace cube_root_of_expression_l401_401394

theorem cube_root_of_expression (x : ℝ) (h : (2 * x - 1) = 49) : (∛(2 * x - 23)) = 3 :=
by
  sorry

end cube_root_of_expression_l401_401394


namespace new_mean_correct_l401_401163

-- Define the original condition data
def initial_mean : ℝ := 42
def total_numbers : ℕ := 60
def discard1 : ℝ := 50
def discard2 : ℝ := 60
def increment : ℝ := 2

-- A function representing the new arithmetic mean
noncomputable def new_arithmetic_mean : ℝ :=
  let initial_sum := initial_mean * total_numbers
  let sum_after_discard := initial_sum - (discard1 + discard2)
  let sum_after_increment := sum_after_discard + (increment * (total_numbers - 2))
  sum_after_increment / (total_numbers - 2)

-- The theorem statement
theorem new_mean_correct : new_arithmetic_mean = 43.55 :=
by 
  sorry

end new_mean_correct_l401_401163


namespace infection_equation_l401_401664

-- Given conditions
def initially_infected : Nat := 1
def total_after_two_rounds : ℕ := 81
def avg_infect_per_round (x : ℕ) : ℕ := x

-- Mathematically equivalent proof problem
theorem infection_equation (x : ℕ) 
  (h1 : initially_infected = 1)
  (h2 : total_after_two_rounds = 81)
  (h3 : ∀ (y : ℕ), initially_infected + avg_infect_per_round y + (avg_infect_per_round y)^2 = total_after_two_rounds):
  (1 + x)^2 = 81 :=
by
  sorry

end infection_equation_l401_401664


namespace ratio_of_combined_semi_to_circle_l401_401980

-- Definition of the situation involving the circles and semicircles
def combined_area_ratio (r : ℝ) : ℝ :=
  let semi_area := (π * (r / 2)^2) / 2
  let combined_semi_area := 2 * semi_area
  let circle_area := π * r^2
  combined_semi_area / circle_area

-- Assert the required ratio
theorem ratio_of_combined_semi_to_circle (r : ℝ) (h : r ≠ 0) : combined_area_ratio r = 1 / 4 :=
  by sorry

end ratio_of_combined_semi_to_circle_l401_401980


namespace circles_externally_tangent_l401_401032

theorem circles_externally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : c₁ = (0, 8)) (h₂ : c₂ = (-6, 0)) (h₃ : r₁ = 6) (h₄ : r₂ = 2) :
  dist c₁ c₂ > r₁ + r₂ :=
by
  rw [dist_eq, h₁, h₂]
  norm_num
  sorry

end circles_externally_tangent_l401_401032


namespace limit_quotient_l401_401826

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def partial_sum (n : ℕ) : ℝ := -3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1

theorem limit_quotient (h : ∀ n : ℕ, S n = partial_sum n) :
  (tendsto (λ n, (a n) / (3 * (n:ℝ))) at_top (𝓝 (-2))) := 
sorry

end limit_quotient_l401_401826


namespace combined_salaries_of_A_C_D_E_l401_401960

-- definitions based on conditions
def b_salary : ℕ := 5000
def avg_salary_of_all : ℕ := 8200

-- theorem to prove
theorem combined_salaries_of_A_C_D_E : 
  let total_salary_of_all := avg_salary_of_all * 5 in
  let combined_salaries := total_salary_of_all - b_salary in
  combined_salaries = 36000 :=
by
  sorry

end combined_salaries_of_A_C_D_E_l401_401960


namespace sum_of_integers_eq_l401_401168

-- We define the conditions
variables (x y : ℕ)
-- The conditions specified in the problem
def diff_condition : Prop := x - y = 16
def prod_condition : Prop := x * y = 63

-- The theorem stating that given the conditions, the sum is 2*sqrt(127)
theorem sum_of_integers_eq : diff_condition x y → prod_condition x y → x + y = 2 * Real.sqrt 127 :=
by
  sorry

end sum_of_integers_eq_l401_401168


namespace harrys_sister_stamps_l401_401903

/-- We define the number of stamps collected by Harry's sister and the total stamps collected -/
variables (S H_total : ℕ)
variables (h1 : Harry_total = 240)
variables (h2 : Harry_total = 3 * S)

/-- Proof statement: Harry's sister collected 60 stamps -/
theorem harrys_sister_stamps : 
  ∃ S, Harry_total = 240 ∧ Harry_total = 3 * S ∧ 4 * S = 240 := sorry

end harrys_sister_stamps_l401_401903


namespace squares_in_region_l401_401799

theorem squares_in_region : 
  let region := { (x, y) | 0 ≤ y ∧ y ≤ 2 * x ∧ 0 ≤ x ∧ x ≤ 6 }
  (number_of_squares (region : Set (ℕ × ℕ))) = 94 :=
by
  let region := { (x, y) | 0 ≤ y ∧ y ≤ 2 * x ∧ 0 ≤ x ∧ x ≤ 6 }
  exact 94 -- sorry, this represents the proof is omitted

end squares_in_region_l401_401799


namespace cosine_periodicity_l401_401340

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l401_401340


namespace cos_eq_43_l401_401317

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401317


namespace infinite_solutions_x2_minus_7y2_eq_1_l401_401142

theorem infinite_solutions_x2_minus_7y2_eq_1 :
  ∃ f : ℕ → ℕ × ℕ, (∀ n, let (x, y) := f n in x^2 - 7 * y^2 = 1) ∧ (function.injective f) :=
sorry

end infinite_solutions_x2_minus_7y2_eq_1_l401_401142


namespace remainder_pow_mod_l401_401621

theorem remainder_pow_mod (base : ℤ) (exp : ℤ) (modulus : ℤ) (h_base : base = 17) (h_exp : exp = 1988) (h_modulus : modulus = 23) : base^exp % modulus = 1 := 
by
  sorry

end remainder_pow_mod_l401_401621


namespace smallest_k_l401_401114

-- Define the set S as a finite set of numbers from 1 to 50
def S : Finset ℕ := Finset.range 51 \ {0}

-- Define the property P that checks if (a + b) divides a * b
def P (a b : ℕ) : Prop := (a + b) ∣ (a * b)

-- Define the main theorem statement which needs to be proven
theorem smallest_k (k : ℕ) : k = 39 ↔ ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = k → ∃ a b ∈ T, a ≠ b ∧ P a b := 
by
  -- Proof goes here
  sorry

end smallest_k_l401_401114


namespace max_value_of_y_l401_401997

theorem max_value_of_y (x : ℝ) (h₁ : 0 < x) (h₂ : x < 4) : 
  ∃ y : ℝ, (y = x * (8 - 2 * x)) ∧ (∀ z : ℝ, z = x * (8 - 2 * x) → z ≤ 8) :=
sorry

end max_value_of_y_l401_401997


namespace sum_of_integers_satisfying_l401_401600

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l401_401600


namespace fractions_addition_l401_401693

theorem fractions_addition : (1 / 6 - 5 / 12 + 3 / 8) = 1 / 8 :=
by
  sorry

end fractions_addition_l401_401693


namespace choir_members_max_l401_401228

theorem choir_members_max (s x : ℕ) (h1 : s * x < 147) (h2 : s * x + 3 = (s - 3) * (x + 2)) : s * x = 84 :=
sorry

end choir_members_max_l401_401228


namespace jess_height_l401_401864

variable (Jana_height Kelly_height Jess_height : ℕ)

-- Conditions
axiom Jana_height_eq : Jana_height = 74
axiom Jana_taller_than_Kelly : Jana_height = Kelly_height + 5
axiom Kelly_shorter_than_Jess : Kelly_height = Jess_height - 3

-- Prove Jess's height
theorem jess_height : Jess_height = 72 := by
  -- Proof goes here
  sorry

end jess_height_l401_401864


namespace points_concyclic_l401_401759

theorem points_concyclic
  (A B C O D E X Y : Type)
  [Triangle A B C]
  (h1 : AB ≠ AC)
  (h2 : circumcenter O A B C)
  (h3 : angle_bisector A B C BAC D)
  (h4 : reflection D (midpoint B C) E)
  (h5 : perpendicular_through D BC AO X)
  (h6 : perpendicular_through E BC AD Y)
  : concyclic B X C Y :=
sorry

end points_concyclic_l401_401759


namespace value_of_x_squared_plus_reciprocal_squared_l401_401804

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l401_401804


namespace probability_first_spade_second_ace_l401_401191

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end probability_first_spade_second_ace_l401_401191


namespace robotics_club_neither_l401_401498

theorem robotics_club_neither (total students programming electronics both: ℕ) 
  (h1: total = 120)
  (h2: programming = 80)
  (h3: electronics = 50)
  (h4: both = 15) : 
  total - ((programming - both) + (electronics - both) + both) = 5 :=
by
  sorry

end robotics_club_neither_l401_401498


namespace smallest_n_correct_l401_401353

noncomputable def smallest_n : ℕ :=
  let factors := [(1, 48), (2, 24), (3, 16), (4, 12), (6, 8)] in
  factors.map (λ (A, B) => 5 * B + A).min

theorem smallest_n_correct : smallest_n = 31 :=
  by
  sorry

end smallest_n_correct_l401_401353


namespace find_n_cos_eq_l401_401346

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l401_401346


namespace find_b_l401_401736

noncomputable def tangent_line_is_tangent (b : ℝ) : Prop :=
  ∃ m : ℝ, 
    (m ≠ 0) ∧ 
    (m = 1) ∧ 
    (b = 1) ∧ 
    (- (1/2) + (1/m) = 1/2) ∧ 
    (-(1/2) * m + ln m = -(1/2))

theorem find_b :
  tangent_line_is_tangent 1 :=
sorry

end find_b_l401_401736


namespace find_p5_l401_401117

noncomputable def p : ℚ[X] := sorry

theorem find_p5 (p : ℚ[X])
  (hp1 : p.eval 1 = 2 / 1^3)
  (hp2 : p.eval 2 = 2 / 2^3)
  (hp3 : p.eval 3 = 2 / 3^3)
  (hp4 : p.eval 4 = 2 / 4^3) :
  p.eval 5 = -463 / 750 :=
sorry

end find_p5_l401_401117


namespace problem_solution_l401_401794

-- Definitions based on conditions given in the problem statement
def validExpression (n : ℕ) : ℕ := 
  sorry -- Placeholder for function defining valid expressions

def T (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else validExpression n

def R (n : ℕ) : ℕ := T n % 4

def computeSum (k : ℕ) : ℕ := 
  (List.range k).map R |>.sum

-- Lean theorem statement to be proven
theorem problem_solution : 
  computeSum 1000001 = 320 := 
sorry

end problem_solution_l401_401794


namespace cos_eq_43_l401_401319

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401319


namespace coloring_ways_l401_401047

theorem coloring_ways : 
  ∃ f : ℕ → ℕ, 
    (∀ n, 2 ≤ n ∧ n ≤ 10 → f n ∈ {0, 1, 2}) ∧ 
    (∀ n, 2 ≤ n ∧ n ≤ 10 → ∀ d, d ∣ n ∧ d < n → f n ≠ f d) ∧ 
    (f 2 ≠ f 3 → 96 = 192 / 2)  ∧ 
    (f 3 ≠ f 4 → 96 = 192 / 2)  ∧ 
    (∀ n, n ∈ {2, 3, 5, 7} → 96 = 192 / 2) :=
sorry

end coloring_ways_l401_401047


namespace smallest_a1_value_l401_401886

noncomputable def a : ℕ → ℝ
| 1       => x  -- where x is the value of a1
| (n + 1) => 13 * a n - 2 * (n + 1)

theorem smallest_a1_value : ∃ x > 0, ∀ a' : ℕ → ℝ, (∀ n > 1, a' n = 13 * a' (n - 1) - 2 * n) → a' 1 = x :=
by
  use 13/36
  sorry

end smallest_a1_value_l401_401886


namespace find_n_l401_401331

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l401_401331


namespace max_angle_APB_l401_401011

noncomputable theory

variables {θ : ℝ} {x y : ℝ}

def circle_C (x y : ℝ) := (x - 3)^2 + y^2 = 1
def circle_M (x y θ : ℝ) := (x - 3 - 3 * cos θ)^2 + (y - 3 * sin θ)^2 = 1

theorem max_angle_APB (θ : ℝ) :
  ∀ P, (circle_M P.1 P.2 θ) →
  ∀ A B, (lying_on_line_through P A B) →
  (circle_C A.1 A.2) → (circle_C B.1 B.2) →
  angle P A B ≤ (π / 3)
  sorry

end max_angle_APB_l401_401011


namespace smallest_sum_l401_401891

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  (∀ A B C D : ℕ, 
    5 * A = 25 * A - 27 * B ∧
    5 * B = 15 * A - 16 * B ∧
    3 * C = 25 * C - 27 * D ∧
    3 * D = 15 * C - 16 * D) ∧
  a = 4 ∧ b = 3 ∧ c = 27 ∧ d = 22 ∧ a + b + c + d = 56

theorem smallest_sum : problem_statement :=
  sorry

end smallest_sum_l401_401891


namespace eccentricity_eq_half_ellipse_eq_l401_401101

-- Define the conditions and attributes of the ellipse
variables {a b c : ℝ} (a_pos : a > 0) (b_pos : b > 0) (b_a : b < a)
variables {C : set (ℝ × ℝ)} (hC : C = {p | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1})

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-real.sqrt (a^2 - b^2), 0)
noncomputable def F₂ : ℝ × ℝ := (real.sqrt (a^2 - b^2), 0)

-- Define the point M
variables {M : ℝ × ℝ} (M_on_C : M ∈ C) (M_perp : M.1 = real.sqrt (a^2 - b^2))
variables {N : ℝ × ℝ} (N_on_C : N ∈ C) (MN_slope : (M.2 - 0) / (M.1 - (-M.1)) = 3/4)

-- Define the distances condition
variables {dist_sum : ℝ} (dist_sum_eq : dist_sum = dist M F₁ + dist M F₂) (dist_sum_4 : dist_sum = 4)

-- Prove that the eccentricity is 1/2
theorem eccentricity_eq_half : real.sqrt (1 - b^2 / a^2) = 1 / 2 :=
sorry

-- Prove that the equation of the ellipse is x^2/4 + y^2/3 = 1
theorem ellipse_eq : a = 2 → b = real.sqrt 3 → C = {p | p.1 ^ 2 / 4 + p.2 ^ 2 / 3 = 1} :=
sorry

end eccentricity_eq_half_ellipse_eq_l401_401101


namespace sum_property_l401_401784

def f (x : ℝ) : ℝ := x + sin (π * x) - 3

theorem sum_property :
  ∑ k in (finset.range 4027).map (λ k, k + 1) {x // true} → (ℝ) (λ k, f (k / 2014)) = -8054 :=
by
  sorry

end sum_property_l401_401784


namespace angle_between_e_and_S_star_l401_401193

noncomputable def angle_between_line_and_plane (S S_star : Plane) (m e : Line) : angle :=
  if S ∩ S_star = m ∧ planeAngle S S_star = 45 ∧ inPlane e S ∧ lineAngle e m = 45 then 30 else sorry

theorem angle_between_e_and_S_star
  (S S_star : Plane)
  (m e : Line)
  (h1 : S ∩ S_star = m)
  (h2 : planeAngle S S_star = 45)
  (h3 : inPlane e S)
  (h4 : lineAngle e m = 45) :
  angle_between_line_and_plane S S_star m e = 30 :=
by
  sorry

end angle_between_e_and_S_star_l401_401193


namespace magnitude_of_2a_minus_b_l401_401415

-- Vector, dot product, and magnitude definitions could be assumed to be included within Mathlib.

variable {V : Type*} [InnerProductSpace ℝ V]
variable (a b : V)
variable (h_dot : ⟪a, b⟫ = 0)
variable (h_norm_a : ∥a∥ = 1)
variable (h_norm_b : ∥b∥ = 2)

theorem magnitude_of_2a_minus_b : ∥2 • a - b∥ = 2 * Real.sqrt 2 := by
  sorry

end magnitude_of_2a_minus_b_l401_401415


namespace smallest_n_satisfying_ratio_l401_401275

-- Definitions and conditions from problem
def sum_first_n_odd_numbers_starting_from_3 (n : ℕ) : ℕ := n^2 + 2 * n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)

theorem smallest_n_satisfying_ratio :
  ∃ n : ℕ, n > 0 ∧ (sum_first_n_odd_numbers_starting_from_3 n : ℚ) / (sum_first_n_even_numbers n : ℚ) = 49 / 50 ∧ n = 51 :=
by
  use 51
  exact sorry

end smallest_n_satisfying_ratio_l401_401275


namespace determine_g_l401_401539

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l401_401539


namespace cos_eq_43_l401_401321

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l401_401321


namespace triangle_ABC_properties_l401_401437

theorem triangle_ABC_properties
  (A B C : Type)
  [MetricSpace B] [MetricSpace C]
  (angle_A : ℝ) (tan_C : ℝ) (AC : ℝ)
  (h1 : angle_A = 90)
  (h2 : tan_C = 8)
  (h3 : AC = 160) :
  let AB := (1280 * real.sqrt 65) / 65,
      perimeter := (1440 * real.sqrt 65 + 10400) / 65 in
  (∀ (AB' : ℝ), AB' = AB) ∧
  (∀ (perimeter' : ℝ), perimeter' = perimeter) :=
by {
  sorry
}

end triangle_ABC_properties_l401_401437


namespace locus_of_point_H_const_value_AQ_AP_by_OR_l401_401448

-- Define the points M and N, and other necessary points and lines
def M := (-2 * Real.sqrt 2, 0)
def N := (2 * Real.sqrt 2, 0)
def A := (-4, 0)
def O := (0, 0)

-- Define the locus of point H and its equation
def E := {H : ℝ × ℝ | H.2 ≠ 0 ∧ (H.1 ^ 2 / 16 + H.2 ^ 2 / 8 = 1)}

-- Preliminary definitions for the line l and determining points P, Q, R
def l (k: ℝ) : (ℝ × ℝ) → Prop := λ (p : ℝ × ℝ), p.2 = k * (p.1 + 4)
def l' (k: ℝ) : (ℝ × ℝ) → Prop := λ (p : ℝ × ℝ), p.2 = k * p.1

theorem locus_of_point_H :
  (∀ (x y : ℝ), y ≠ 0 → (x, y) ∈ E ↔ (x^2 / 16 + y^2 / 8 = 1)) :=
sorry

theorem const_value_AQ_AP_by_OR (k : ℝ) (hk : k ≠ 0) :
  let Q := (0, 4 * k),
      P := (4 - 8 * k ^ 2) / (1 + 2 * k ^ 2),
      R := 16 / (1 + 2 * k ^ 2),
      AQ := Real.sqrt (16 + 16 * k ^ 2),
      AP := 8 * Real.sqrt(1 + k ^ 2) / (1 + 2 * k ^ 2),
      OR := Real.sqrt(16 * (1 + k ^ 2) / (1 + 2 * k ^ 2))
  in AQ * AP / OR^2 = 2 :=
sorry

end locus_of_point_H_const_value_AQ_AP_by_OR_l401_401448


namespace monotonic_intervals_of_f_range_of_a_l401_401120

noncomputable def f (x : ℝ) (a : ℝ) := log x - 2 * a * x + 2 * a

noncomputable def g (x : ℝ) (a : ℝ) := x * (log x - 2 * a * x + 2 * a) + a * x^2 - x

theorem monotonic_intervals_of_f (a : ℝ) :
  (a ≤ 0 ∧ ∀ x > 0, f x a ≥ f (x + 1) a) ∨
  (a > 0 ∧ ∀ x ∈ Ioo 0 (1 / (2 * a)), f x a ≥ f (x + 1) a ∧ ∀ x > 1 / (2 * a), f x a < f (x + 1) a) :=
sorry

theorem range_of_a (a : ℝ) (h : ∀ x > 0, g x a ≤ g 1 a) : a > 1 / 2 :=
sorry

end monotonic_intervals_of_f_range_of_a_l401_401120


namespace complex_conjugate_quadrant_l401_401008

theorem complex_conjugate_quadrant 
  (z : ℂ)
  (hz : z = (1 + 2*complex.I) / complex.I) :
  let conj_z := conj z
  in conj_z.re > 0 ∧ conj_z.im > 0 :=
by 
  sorry

end complex_conjugate_quadrant_l401_401008


namespace tetrahedron_properties_l401_401351

noncomputable def volume_tetrahedron (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A1
  let (x2, y2, z2) := A2
  let (x3, y3, z3) := A3
  let (x4, y4, z4) := A4
  (1 / 6) * abs (
    x1 * (y2 * (z3 - z4) + y3 * (z4 - z2) + y4 * (z2 - z3)) - 
    x2 * (y1 * (z3 - z4) + y3 * (z4 - z1) + y4 * (z1 - z3)) +
    x3 * (y1 * (z2 - z4) + y2 * (z4 - z1) + y4 * (z1 - z2)) - 
    x4 * (y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2))
  )

noncomputable def height_tetrahedron (A1 A2 A3: ℝ × ℝ × ℝ) (V: ℝ): ℝ :=
  let (x1, y1, z1) := A1
  let (x2, y2, z2) := A2
  let (x3, y3, z3) := A3
  let cross_product := ( (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1),
                         (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1),
                         (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) )
  let area := (1 / 2) * real.sqrt (cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2)
  (3 * V) / area

theorem tetrahedron_properties :
  let A1 := (1, 2, -3)
  let A2 := (1, 0, 1)
  let A3 := (-2, -1, 6)
  let A4 := (0, -5, -4)
  volume_tetrahedron A1 A2 A3 A4 = 16 ∧ height_tetrahedron A1 A2 A3 (volume_tetrahedron A1 A2 A3 A4) = 8 * real.sqrt (2 / 3) :=
by
  sorry

end tetrahedron_properties_l401_401351


namespace piecewise_continuous_b_zero_l401_401739

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem piecewise_continuous_b_zero :
  (∀ x : ℝ, (¬ x > 2 → 3 * 2 + b = 2 + 4) → b = 0) :=
begin
  intros x h,
  have h1 : 3 * 2 + b = 6 := by { exact (h (by linarith)) },
  linarith,
end

end piecewise_continuous_b_zero_l401_401739


namespace victoria_speed_l401_401982

def victoria_speed_in_m_per_s (distance_in_km : ℝ) (time_in_min : ℝ) : ℝ :=
  (distance_in_km * 1000) / (time_in_min * 60)

theorem victoria_speed (distance_in_km : ℝ) (time_in_min : ℝ) (h_dist : distance_in_km = 11.4) (h_time : time_in_min = 2) :
  victoria_speed_in_m_per_s distance_in_km time_in_min = 95 :=
by
  rw [h_dist, h_time]
  simp [victoria_speed_in_m_per_s]
  norm_num
  sorry

end victoria_speed_l401_401982


namespace man_mass_l401_401209

theorem man_mass (length breadth sinking_depth density : ℝ) 
  (length_eq : length = 3) 
  (breadth_eq : breadth = 2) 
  (sinking_depth_eq : sinking_depth = 0.01) 
  (density_eq : density = 1000) : 
  length * breadth * sinking_depth * density = 60 :=
by 
  -- substitute the given values into the formula
  rw [length_eq, breadth_eq, sinking_depth_eq, density_eq]
  -- perform the final computation
  norm_num
  sorry

end man_mass_l401_401209


namespace quadratic_expression_value_l401_401389

theorem quadratic_expression_value (a : ℝ) (h : a^2 - 2 * a - 3 = 0) : a^2 - 2 * a + 1 = 4 :=
by 
  -- Proof omitted for clarity in this part
  sorry 

end quadratic_expression_value_l401_401389


namespace systematic_sampling_student_number_l401_401842

theorem systematic_sampling_student_number 
  (total_students : ℕ)
  (sample_size : ℕ)
  (interval_between_numbers : ℕ)
  (student_17_in_sample : ∃ n, 17 = n ∧ n ≤ total_students ∧ n % interval_between_numbers = 5)
  : ∃ m, m = 41 ∧ m ≤ total_students ∧ m % interval_between_numbers = 5 := 
sorry

end systematic_sampling_student_number_l401_401842


namespace exists_not_prime_P_l401_401663

noncomputable def general_formula (n : ℕ) : ℕ :=
  n * (n - 1) + 41

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_not_prime_P : ∃ n, ¬ is_prime (general_formula 41) :=
by {
  use 41,
  unfold general_formula,
  show ¬ is_prime 1681,
  sorry
}

end exists_not_prime_P_l401_401663


namespace karl_total_income_correct_l401_401868

noncomputable def price_of_tshirt : ℝ := 5
noncomputable def price_of_pants : ℝ := 4
noncomputable def price_of_skirt : ℝ := 6
noncomputable def price_of_refurbished_tshirt : ℝ := price_of_tshirt / 2

noncomputable def discount_for_skirts (n : ℕ) : ℝ := (n / 2) * 2 * price_of_skirt * 0.10
noncomputable def discount_for_tshirts (n : ℕ) : ℝ := (n / 5) * 5 * price_of_tshirt * 0.20
noncomputable def discount_for_pants (n : ℕ) : ℝ := 0 -- accounted for in quantity

noncomputable def sales_tax (amount : ℝ) : ℝ := amount * 0.08

noncomputable def total_income : ℝ := 
  let tshirt_income := 8 * price_of_tshirt + 7 * price_of_refurbished_tshirt - discount_for_tshirts 15
  let pants_income := 6 * price_of_pants - discount_for_pants 6
  let skirts_income := 12 * price_of_skirt - discount_for_skirts 12
  let income_before_tax := tshirt_income + pants_income + skirts_income
  income_before_tax + sales_tax income_before_tax

theorem karl_total_income_correct : total_income = 141.80 :=
by
  sorry

end karl_total_income_correct_l401_401868


namespace intersection_of_sets_l401_401764

open Set

theorem intersection_of_sets : 
  let A := {-1, 1, 2, 4}
  let B := {-1, 0, 2}
  A ∩ B = {-1, 2} := by
    sorry

end intersection_of_sets_l401_401764


namespace correct_statements_l401_401476

def f (x : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d
def f_prime (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem correct_statements (b c d : ℝ) :
  (∃ x : ℝ, f x b c d = 4 ∧ f_prime x b c = 0) ∧
  (∃ x : ℝ, f x b c d = 0 ∧ f_prime x b c = 0) :=
by
  sorry

end correct_statements_l401_401476


namespace improved_productivity_l401_401136

-- Let the initial productivity be a constant
def initial_productivity : ℕ := 10

-- Let the increase factor be a constant, represented as a rational number
def increase_factor : ℚ := 3 / 2

-- The goal is to prove that the current productivity equals 25 trees daily
theorem improved_productivity : initial_productivity + (initial_productivity * increase_factor).toNat = 25 := 
by
  sorry

end improved_productivity_l401_401136


namespace range_of_b_for_points_on_circle_l401_401005

theorem range_of_b_for_points_on_circle :
  ∀ (b : ℝ), 
  (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0 ∧ |y - x - b| = 2 * sqrt 2) →
  -2 ≤ b ∧ b ≤ 2 :=
begin
  sorry
end

end range_of_b_for_points_on_circle_l401_401005


namespace necessary_but_not_sufficient_l401_401567

-- Definitions used in the conditions
def x (x : ℝ) : Prop := x < 2
def y (x : ℝ) : Prop := x^2 - 2 * x < 0

-- Statement to prove that x < 2 is a necessary but not sufficient condition for x^2 - 2x < 0
theorem necessary_but_not_sufficient (x : ℝ) (h : x < 2) : ¬ (y x ↔ (x x)) :=
by
  sorry

end necessary_but_not_sufficient_l401_401567


namespace evaluate_fraction_l401_401291

theorem evaluate_fraction : (3 / (1 - 3 / 4) = 12) := by
  have h : (1 - 3 / 4) = 1 / 4 := by
    sorry
  rw [h]
  sorry

end evaluate_fraction_l401_401291


namespace largest_divisor_of_consecutive_even_product_l401_401877

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), (n > 0) → ∃ d : ℕ, d = 48 ∧ d ∣ (8 * n * (n + 1) * (n + 2)) :=
by
  intro n hn
  use 48
  split
  sorry

end largest_divisor_of_consecutive_even_product_l401_401877


namespace nikki_irises_l401_401176

theorem nikki_irises (initial_roses add_roses: ℕ) 
  (ratio_iris_roses: ℕ × ℕ) 
  (h_ratio: ratio_iris_roses = (2, 5)) 
  (h_initial_roses: initial_roses = 25) 
  (h_add_roses: add_roses = 20) : 
  (nat.ceil ((2 * (initial_roses + add_roses) / 5 : ℚ)) = 18) :=
by
  sorry

end nikki_irises_l401_401176


namespace fourth_student_guess_l401_401562

theorem fourth_student_guess :
    let guess1 := 100
    let guess2 := 8 * guess1
    let guess3 := guess2 - 200
    let avg := (guess1 + guess2 + guess3) / 3
    let guess4 := avg + 25
    guess4 = 525 := by
    intros guess1 guess2 guess3 avg guess4
    have h1 : guess1 = 100 := rfl
    have h2 : guess2 = 8 * guess1 := rfl
    have h3 : guess3 = guess2 - 200 := rfl
    have h4 : avg = (guess1 + guess2 + guess3) / 3 := rfl
    have h5 : guess4 = avg + 25 := rfl
    simp [h1, h2, h3, h4, h5]
    sorry

end fourth_student_guess_l401_401562


namespace quadratic_root_one_is_minus_one_l401_401425

theorem quadratic_root_one_is_minus_one (m : ℝ) (h : ∃ x : ℝ, x = -1 ∧ m * x^2 + x - m^2 + 1 = 0) : m = 1 :=
by
  sorry

end quadratic_root_one_is_minus_one_l401_401425


namespace problem1_problem2_problem3_problem4_l401_401700

theorem problem1 : 24 - (-16) + (-25) - 32 = -17 := by
  sorry

theorem problem2 : (-1 / 2) * 2 / 2 * (-1 / 2) = 1 / 4 := by
  sorry

theorem problem3 : -2^2 * 5 - (-2)^3 * (1 / 8) + 1 = -18 := by
  sorry

theorem problem4 : ((-1 / 4) - (5 / 6) + (8 / 9)) / (-1 / 6)^2 + (-2)^2 * (-6)= -31 := by
  sorry

end problem1_problem2_problem3_problem4_l401_401700


namespace circle_line_intersection_length_l401_401371

theorem circle_line_intersection_length :
  ∀ (x y : ℝ), (x-1)^2 + y^2 = 1 → x - 2 * y + 1 = 0 → |AB| = 2 * sqrt 5 / 5 :=
by
  intros x y h₁ h₂
  sorry

end circle_line_intersection_length_l401_401371


namespace sum_solutions_eq_l401_401990

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end sum_solutions_eq_l401_401990


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401512

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l401_401512


namespace calculate_f_at_5_l401_401488

def f (x : ℝ) : ℝ := log (x - 3) + log x

theorem calculate_f_at_5 : f 5 = 1 := 
by sorry

end calculate_f_at_5_l401_401488


namespace sanjay_homework_fraction_l401_401147

theorem sanjay_homework_fraction (x : ℚ) :
  (2 * x + 1) / 3 + 4 / 15 = 1 ↔ x = 3 / 5 :=
by
  sorry

end sanjay_homework_fraction_l401_401147


namespace find_k_l401_401064

def is_black (x y : ℕ) : Prop := (x + y) % 2 = 0

def 5x5_black_cells (shift_x shift_y : ℕ) : ℕ :=
  (List.range 5).bind (λ x, (List.range 5).filter (λ y, is_black (x + shift_x) (y + shift_y))).length

def 5x5_white_cells (shift_x shift_y : ℕ) : ℕ :=
  25 - (5x5_black_cells shift_x shift_y)

theorem find_k (S : finset (ℕ × ℕ)) (k : ℕ) (hS : S.card = 25) 
  (hprob : (finset.filter (λ s, 5x5_black_cells (prod.fst s) (prod.snd s) > 5x5_white_cells (prod.fst s) < 5).card) / S.card = 0.48) :
  k = 9 :=
sorry

end find_k_l401_401064


namespace fourth_student_guess_l401_401560

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  fourth_guess = 525 := 
by
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  show fourth_guess = 525 from sorry

end fourth_student_guess_l401_401560


namespace marco_needs_at_least_n_tables_n_tables_are_sufficient_l401_401721

variables (n : ℕ) -- let n be the number of stickers and friends

-- Part (i) statement
theorem marco_needs_at_least_n_tables (h : ∀ i, 1 ≤ i → i ≤ n → ∃ P : set ℕ, (∀ j, j ≠ i → j ∈ P) ∧ i ∉ P):
  ∃ m, m ≥ n := 
sorry

-- Part (ii) statement
theorem n_tables_are_sufficient (h : ∀ i, 1 ≤ i → i ≤ n → ∃ P : set ℕ, (∀ j, j ≠ i → j ∈ P) ∧ i ∉ P):
  ∀ (seating : ℕ → set ℕ), (∀ t, t ≤ n → seating t = { x | ∃ i, x ≠ i }) → 
  ∀ t, t ≤ n → ¬ ∃ P₁ P₂, P₁ ≠ P₂ ∧ P₁ ∪ P₂ = {1, 2, ..., n} :=
sorry

end marco_needs_at_least_n_tables_n_tables_are_sufficient_l401_401721


namespace sum_of_integers_square_greater_272_l401_401596

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l401_401596


namespace mathematics_equivalent_proof_l401_401106

noncomputable def distinctRealNumbers (a b c d : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d

theorem mathematics_equivalent_proof (a b c d : ℝ)
  (H₀ : distinctRealNumbers a b c d)
  (H₁ : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 :=
sorry

end mathematics_equivalent_proof_l401_401106


namespace find_n_l401_401733

theorem find_n : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 12345 [MOD 7] ∧ n = 4 :=
by
  sorry

end find_n_l401_401733


namespace part1_part2_l401_401701

theorem part1 : 2 * (-1)^3 - (-2)^2 / 4 + 10 = 7 := by
  sorry

theorem part2 : abs (-3) - (-6 + 4) / (-1 / 2)^3 + (-1)^2013 = -14 := by
  sorry

end part1_part2_l401_401701


namespace a_n_plus_1_is_geometric_general_term_formula_l401_401788

-- Define the sequence a_n.
def a : ℕ → ℤ
| 0       => 0  -- a_0 is not given explicitly, we start the sequence from 1.
| (n + 1) => if n = 0 then 1 else 2 * a n + 1

-- Prove that the sequence {a_n + 1} is a geometric sequence.
theorem a_n_plus_1_is_geometric : ∃ r : ℤ, ∀ n : ℕ, (a (n + 1) + 1) / (a n + 1) = r := by
  sorry

-- Find the general formula for a_n.
theorem general_term_formula : ∃ f : ℕ → ℤ, ∀ n : ℕ, a n = f n := by
  sorry

end a_n_plus_1_is_geometric_general_term_formula_l401_401788


namespace sqrt_meaningful_range_l401_401175

theorem sqrt_meaningful_range (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 1)) ↔ x ≥ 1 :=
by 
  sorry

end sqrt_meaningful_range_l401_401175


namespace no_primes_in_sequence_l401_401758

noncomputable def Q : ℕ :=
  ∏ p in Finset.filter Nat.Prime (Finset.range 68), p

def sequence : List ℕ :=
  (List.range' 3 65).map (λ m => Q + m)

def isPrimeSequence := ∀ n ∈ sequence, ¬Nat.Prime n

theorem no_primes_in_sequence : isPrimeSequence := by
  sorry

end no_primes_in_sequence_l401_401758


namespace cost_per_dozen_approx_l401_401923

def chicken_initial_cost := 80
def feed_cost_per_week := 1
def total_weeks := 81
def total_cost := chicken_initial_cost + feed_cost_per_week * total_weeks
def eggs_per_week := 4 * 3  -- 4 chickens, each produces 3 eggs per week
def total_eggs_produced := eggs_per_week * total_weeks
def dozens_of_eggs := total_weeks
def cost_per_dozen := total_cost / dozens_of_eggs

theorem cost_per_dozen_approx :
  abs(cost_per_dozen - 1.99) < 1e-2 := by
  sorry

end cost_per_dozen_approx_l401_401923


namespace solve_inequality_l401_401261

theorem solve_inequality (x : ℝ) : (1 + x) / 3 < x / 2 → x > 2 := 
by {
  sorry
}

end solve_inequality_l401_401261


namespace stock_AB_increase_factor_l401_401036

-- Define the conditions as mathematical terms
def stock_A_initial := 300
def stock_B_initial := 300
def stock_C_initial := 300
def stock_C_final := stock_C_initial / 2
def total_final := 1350
def AB_combined_initial := stock_A_initial + stock_B_initial
def AB_combined_final := total_final - stock_C_final

-- The statement to prove that the factor by which stocks A and B increased in value is 2.
theorem stock_AB_increase_factor :
  AB_combined_final / AB_combined_initial = 2 :=
  by
    sorry

end stock_AB_increase_factor_l401_401036


namespace minimum_value_f_on_positive_reals_l401_401387

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * log (x + sqrt (1 + x^2)) + 3

theorem minimum_value_f_on_positive_reals (a b : ℝ) 
  (h₁ : ∀ x < 0, f a b x ≤ 10)
  (h₂ : ∃ x < 0, f a b x = 10)
  : ∃ x > 0, f a b x = -4 :=
sorry

end minimum_value_f_on_positive_reals_l401_401387


namespace functional_equation_sum_l401_401054

theorem functional_equation_sum :
  (∀ a b : ℕ, f (a + b) = f a * f b) ∧ f 1 = 2 → 
  (∑ k in finset.range 2012, f (k + 2) / f (k + 1)) = 4024 :=
sorry

end functional_equation_sum_l401_401054


namespace calculate_area_l401_401688

noncomputable def totalArea : ℝ :=
  6 * (2 * (π / 9) + 4 * sin (6 * (π / 9)) / 3)

theorem calculate_area : totalArea = (8 * π / 3) + 8 * Real.sqrt 3 :=
by
  sorry

end calculate_area_l401_401688


namespace find_g1_l401_401373

open Function

-- Definitions based on the conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem find_g1 (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0) 
  (h2 : g (-1) = 1) 
  : g 1 = -3 :=
sorry

end find_g1_l401_401373


namespace triangle_BC_is_220_l401_401829

-- Given definitions and conditions
variables (A B C X : Type) {AB AC : ℕ}
variable (hAB : AB = 101)
variable (hAC : AC = 131)
variables (x y : ℕ) -- BX = y, CX = x such that x and y are integer lengths
variable (hx : x = 90)
variable (hy : y = 130)

-- Prove BC = 220 given the conditions
theorem triangle_BC_is_220 (hAX_eq_AB : (A - X) = AB) (hBX_int : ∃ y : ℕ, BX = y) (hCX_int : ∃ x : ℕ, CX = x) :
  let BC := x + y
  BC = 220 :=
by
  rw [hx, hy]
  exact rfl

end triangle_BC_is_220_l401_401829


namespace boys_girls_relationship_l401_401069

theorem boys_girls_relationship (b g : ℕ)
  (h1 : ∀ n : ℕ, n > 0 → (nth_boy_girls (n + ((2:ℕ) * (n - 1)) = 2 * n + 4))
  (h2 : nth_boy_girls b = g) :
  b = (g - 4) / 2 :=
by
  sorry

end boys_girls_relationship_l401_401069


namespace directrix_of_parabola_l401_401020

-- Define the given conditions
def parabola_focus_on_line (p : ℝ) := ∃ (x y : ℝ), y^2 = 2 * p * x ∧ 2 * x + 3 * y - 8 = 0

-- Define the statement to be proven
theorem directrix_of_parabola (p : ℝ) (h: parabola_focus_on_line p) : 
   ∃ (d : ℝ), d = -4 := 
sorry

end directrix_of_parabola_l401_401020


namespace turnover_threshold_l401_401356

-- Definitions based on the problem conditions
def valid_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def daily_turnover (x : ℝ) : ℝ := 20 * (10 - x) * (50 + 8 * x)

-- Lean 4 statement equivalent to mathematical proof problem
theorem turnover_threshold (x : ℝ) (hx : valid_domain x) (h_turnover : daily_turnover x ≥ 10260) :
  x ≥ 1 / 2 ∧ x ≤ 2 :=
sorry

end turnover_threshold_l401_401356


namespace acute_angle_formed_by_lines_l401_401556

noncomputable def slope (l : ℝ × ℝ × ℝ) : ℝ :=
if h : l.2 ≠ 0 then -l.1 / l.2 else 0

noncomputable def inclination_angle (m : ℝ) : ℝ :=
if m = 0 then 0 else real.arctan m

noncomputable def angle_between (θ1 θ2 : ℝ) : ℝ :=
abs (θ1 - θ2)

theorem acute_angle_formed_by_lines : 
  let l1 := (√3, -1, 1)
  let l2 := (1, 0, -5)
  let m1 := slope l1
  let m2 := slope l2
  let θ1 := inclination_angle m1
  let θ2 := inclination_angle m2
in angle_between θ1 θ2 = real.to_radians 30 :=
begin
  sorry
end

end acute_angle_formed_by_lines_l401_401556


namespace rug_area_is_24_l401_401242

def length_floor : ℕ := 12
def width_floor : ℕ := 10
def strip_width : ℕ := 3

theorem rug_area_is_24 :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := 
by
  sorry

end rug_area_is_24_l401_401242


namespace problem_statement_l401_401786

def is_ideal_circle (circle : ℝ × ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, (circle P = 0 ∧ circle Q = 0) ∧ (abs (l P) = 1 ∧ abs (l Q) = 1)

noncomputable def line_l (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 - 12

noncomputable def circle_D (p : ℝ × ℝ) : ℝ := (p.1 - 4) ^ 2 + (p.2 - 4) ^ 2 - 16

theorem problem_statement : is_ideal_circle circle_D line_l :=
sorry  -- The proof would go here

end problem_statement_l401_401786


namespace max_even_integers_l401_401673

theorem max_even_integers (a1 a2 a3 a4 a5 a6 : ℕ) (h1: a1 > 0) (h2: a2 > 0) 
  (h3: a3 > 0) (h4: a4 > 0) (h5: a5 > 0) (h6: a6 > 0) (hProd: a1 * a2 * a3 * a4 * a5 * a6 % 2 = 1) : 
  ∃ k : ℕ, k = 0 ∧ card {i | ∃ j, i = j ∧ (j = a1 ∨ j = a2 ∨ j = a3 ∨ j = a4 ∨ j = a5 ∨ j = a6) ∧ j % 2 = 0} = k := 
by
  sorry

end max_even_integers_l401_401673


namespace fabian_walks_l401_401292

variable (D : ℝ)
variable (h1 : 3 * D + 3 * D = 30)

theorem fabian_walks (h2 : h1) : D = 5 := by
  sorry

end fabian_walks_l401_401292


namespace unique_position_assignments_l401_401247

theorem unique_position_assignments (n : ℕ) :
    n = 16 → (16 * 15 * 14 * 13 * 12 = 524160) :=
by
  intro h
  rw [h]
  norm_num

end unique_position_assignments_l401_401247


namespace right_triangle_decagon_pentagon_l401_401082

theorem right_triangle_decagon_pentagon (A B C D : Type) [right_triangle A B C] 
  (BC_eq_2AB : BC = 2 * AB) (D_is_bisector : is_internal_angle_bisector A B C D) :
  ∃ (r : ℝ), r = AB ∧
  side_of_decagon BD (circle_with_radius r) ∧
  side_of_pentagon AD (circle_with_radius r) := 
sorry

end right_triangle_decagon_pentagon_l401_401082


namespace find_abc_sum_l401_401103

theorem find_abc_sum {U : Type} 
  (a b c : ℕ)
  (ha : a = 26)
  (hb : b = 1)
  (hc : c = 32)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  a + b + c = 59 :=
by
  sorry

end find_abc_sum_l401_401103


namespace road_construction_equation_l401_401978

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l401_401978


namespace four_cycle_in_town_graph_l401_401444

noncomputable
def exists_four_cycle (G : SimpleGraph (Fin 7)) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin 7), 
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v4 ∧ v4 ≠ v1 ∧ 
    G.Adj v1 v2 ∧ G.Adj v2 v3 ∧ G.Adj v3 v4 ∧ G.Adj v4 v1

theorem four_cycle_in_town_graph (G : SimpleGraph (Fin 7)) 
  (h : ∀ (v : Fin 7), 3 ≤ G.degree v) : exists_four_cycle G :=
sorry

end four_cycle_in_town_graph_l401_401444


namespace triangle_side_length_l401_401859

-- Definitions based on problem conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variables (AC BC AD AB CD : ℝ)

-- Conditions from the problem
axiom h1 : BC = 2 * AC
axiom h2 : AD = (1 / 3) * AB

-- Theorem statement to be proved
theorem triangle_side_length (h1 : BC = 2 * AC) (h2 : AD = (1 / 3) * AB) : CD = 2 * AD :=
sorry

end triangle_side_length_l401_401859


namespace general_term_arithmetic_sequence_max_sum_arithmetic_sequence_l401_401765

open nat

-- Given an arithmetic sequence with a_2 = 1 and a_5 = -5.
-- Prove that the general term a_n is -2n + 5.
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : a 2 = 1) (h2 : a 5 = -5) : 
  ∃ (d a₁ : ℤ), d = -2 ∧ a₁ = 3 ∧ ∀ n, a n = a₁ + (n - 1) * d := 
sorry

-- Given an arithmetic sequence with a_1 = 3 and d = -2.
-- Prove that the maximum value of the sum S_n of the first n terms is 4.
theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ)
  (h1 : a₁ = 3) (h2 : d = -2) : 
  ∃ n : ℕ, S n = n * a₁ + (n * (n - 1)) / 2 * d ∧ 
  (∀ m : ℕ, S m ≤ 4) ∧ S 2 = 4 := 
sorry

end general_term_arithmetic_sequence_max_sum_arithmetic_sequence_l401_401765


namespace determine_f_980_1980_l401_401964
open Nat

def f : ℕ → ℕ → ℕ
| x, y := if x = y then x else (x + y) * f x y / y

theorem determine_f_980_1980 :
  ∀ f : ℕ → ℕ → ℕ,
  (∀ x : ℕ, f x x = x) →
  (∀ x y : ℕ, f x y = f y x) →
  (∀ x y : ℕ, (x + y) * f x y = f x (x + y) * y) →
  f 980 1980 = 97020 :=
begin
  intros f h1 h2 h3,
  -- proof to be filled in
  sorry
end

end determine_f_980_1980_l401_401964


namespace trivia_game_probability_l401_401252

/-
Theorem: The probability that the player wins the trivia game by guessing at least three 
out of four questions correctly is 13/256.
Conditions:
1. Each game consists of 4 multiple-choice questions.
2. Each question has 4 choices.
3. A player wins if they answer at least 3 out of 4 questions correctly.
4. The player guesses on each question.

We need to prove that the probability of winning is 13/256.
-/
theorem trivia_game_probability : 
  let p : ℚ := 1 / 4 in
  let prob_all_correct : ℚ := p^4 in
  let prob_three_correct : ℚ := (p^3) * (3 / 4) * 4 in
  prob_all_correct + prob_three_correct = 13 / 256 :=
by
  sorry

end trivia_game_probability_l401_401252


namespace place_value_ratio_l401_401455

theorem place_value_ratio : 
  let number := 86549.2047 in
  let digit6_place_value := 1000 in  -- thousands place
  let digit2_place_value := 0.1 in  -- tenths place
  digit6_place_value / digit2_place_value = 10000 := 
by
  -- Proof goes here
  sorry

end place_value_ratio_l401_401455


namespace cube_root_square_root_exp_l401_401272

theorem cube_root_square_root_exp : 
  (real.cbrt ((-2.0: ℝ) ^ 3) - real.sqrt 4.0 + (real.sqrt 3.0) ^ 0) = -3 :=
by
  sorry

end cube_root_square_root_exp_l401_401272


namespace ticket_queue_correct_l401_401963

-- Define the conditions
noncomputable def ticket_queue_count (m n : ℕ) (h : n ≥ m) : ℕ :=
  (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))

-- State the theorem
theorem ticket_queue_correct (m n : ℕ) (h : n ≥ m) :
  ticket_queue_count m n h = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by
  sorry

end ticket_queue_correct_l401_401963


namespace proj_b_l401_401879

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end proj_b_l401_401879


namespace pen_cost_l401_401290

def pencil_cost : ℝ := 1.60
def elizabeth_money : ℝ := 20.00
def num_pencils : ℕ := 5
def num_pens : ℕ := 6

theorem pen_cost (pen_cost : ℝ) : 
  elizabeth_money - (num_pencils * pencil_cost) = num_pens * pen_cost → 
  pen_cost = 2 :=
by 
  sorry

end pen_cost_l401_401290


namespace range_of_x_range_of_a_l401_401382

variable (a x : ℝ)

-- Define proposition p: x^2 - 3ax + 2a^2 < 0
def p (a x : ℝ) : Prop := x^2 - 3 * a * x + 2 * a^2 < 0

-- Define proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- First theorem: Prove the range of x when a = 2 and p ∨ q is true
theorem range_of_x (h : p 2 x ∨ q x) : 2 < x ∧ x < 4 := 
by sorry

-- Second theorem: Prove the range of a when ¬p is necessary but not sufficient for ¬q
theorem range_of_a (h : ∀ x, q x → p a x) : 3/2 ≤ a ∧ a ≤ 2 := 
by sorry

end range_of_x_range_of_a_l401_401382


namespace find_g_l401_401544

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l401_401544


namespace hyperbola_sum_eq_l401_401710

-- Definitions based on conditions
def center : ℝ × ℝ := (3, -1)
def focus : ℝ × ℝ := (3 + Real.sqrt 53, -1)
def vertex : ℝ × ℝ := (7, -1)

-- Variables for the calculations
def h := center.1
def k := center.2
def a := |h - vertex.1|
def c := |h - focus.1|
def b := Real.sqrt (c^2 - a^2)

-- Statement of the problem
theorem hyperbola_sum_eq : h + k + a + b = 6 + Real.sqrt 37 :=
by
  -- Placeholder for actual proof
  sorry

end hyperbola_sum_eq_l401_401710


namespace al_sandwiches_count_l401_401568

noncomputable def total_sandwiches (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

noncomputable def prohibited_combinations (bread_forbidden_combination cheese_forbidden_combination : ℕ) : ℕ := 
  bread_forbidden_combination + cheese_forbidden_combination

theorem al_sandwiches_count (bread meat cheese : ℕ) 
  (bread_forbidden_combination cheese_forbidden_combination : ℕ) 
  (h1 : bread = 5) 
  (h2 : meat = 7) 
  (h3 : cheese = 6) 
  (h4 : bread_forbidden_combination = 5) 
  (h5 : cheese_forbidden_combination = 6) : 
  total_sandwiches bread meat cheese - prohibited_combinations bread_forbidden_combination cheese_forbidden_combination = 199 :=
by
  sorry

end al_sandwiches_count_l401_401568


namespace problem_statement_l401_401922

def f(x : ℝ) : ℝ := Real.cos x + |Real.sin x|

-- Definition for an even function
def is_even(f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)

-- Definition for a monotonically decreasing function on an interval
def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f(x1) > f(x2)

-- Proof problem stating the conditions and conclusions
theorem problem_statement :
  is_even f ∧ 
  ¬(∀ x, f(x + π) = f(x)) ∧ 
  is_monotonically_decreasing f (π / 4) π ∧ 
  (¬ ∃ x, f(x) = 2) :=
by {
  sorry
}

end problem_statement_l401_401922


namespace sum_of_two_digit_divisors_of_143_mod_eq_5_l401_401110

theorem sum_of_two_digit_divisors_of_143_mod_eq_5 :
    ∑ d in { d | 10 ≤ d ∧ d < 100 ∧ 143 % d = 5 }, d = 115 :=
by
  sorry

end sum_of_two_digit_divisors_of_143_mod_eq_5_l401_401110


namespace sqrt_meaningful_range_l401_401174

theorem sqrt_meaningful_range (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 1)) ↔ x ≥ 1 :=
by 
  sorry

end sqrt_meaningful_range_l401_401174


namespace exists_integers_x_y_z_l401_401140

theorem exists_integers_x_y_z (n : ℕ) : 
  ∃ x y z : ℤ, (x^2 + y^2 + z^2 = 3^(2^n)) ∧ (Int.gcd x (Int.gcd y z) = 1) :=
sorry

end exists_integers_x_y_z_l401_401140


namespace highest_percentage_without_car_l401_401510

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l401_401510


namespace square_plot_area_l401_401233

theorem square_plot_area
  (cost_per_foot : ℕ)
  (total_cost : ℕ)
  (s : ℕ)
  (area : ℕ)
  (h1 : cost_per_foot = 55)
  (h3 : total_cost = 3740)
  (h4 : total_cost = 4 * s * cost_per_foot)
  (h5 : area = s * s) :
  area = 289 := sorry

end square_plot_area_l401_401233


namespace mona_grouped_before_first_group_l401_401129

def mona_game (total_groups players_per_group unique_players prev_group_count other_group_count : ℕ) : ℕ :=
  let total_player_slots := total_groups * players_per_group in
  let repeated_players := total_player_slots - unique_players in
  let repeated_in_other_group := repeated_players - other_group_count in
  repeated_in_other_group

theorem mona_grouped_before_first_group :
  mona_game 9 4 33 2 1 = 2 :=
by
  -- The proof will be provided here
  sorry

end mona_grouped_before_first_group_l401_401129


namespace sum_x_coords_Q3_l401_401243

/-- Define a regular 50-gon Q_1 --/
structure Regular50gon where
  vertices : Fin 50 → ℝ × ℝ -- 50 vertices as pairs of (x, y) coordinates

/-- Given Q_1, Q_2, and Q_3 as described, prove the sum of x-coordinates of Q_3 is 1010 --/
theorem sum_x_coords_Q3 (Q1 : Regular50gon) (sum_x_Q1 : (Fin 50 → ℝ × ℝ) → ℝ := λ v, Finset.univ.sum (λ i, (v i).1)) (h1 : sum_x_Q1 Q1.vertices = 1010) :
    let Q2_vertices := (λ i, (Q1.vertices i, Q1.vertices (i + 1) % 50)).map (λ xy, ((xy.1.1 + xy.2.1) / 2, (xy.1.2 + xy.2.2) / 2))
    let Q2 : Regular50gon := {vertices := Q2_vertices }
    let Q3_vertices := (λ i, (Q2.vertices i, Q2.vertices (i + 1) % 50)).map (λ xy, ((xy.1.1 + xy.2.1) / 2, (xy.1.2 + xy.2.2) / 2))
    let Q3 : Regular50gon := {vertices := Q3_vertices }
    sum_x_Q1 Q3.vertices = 1010 :=
by sorry

end sum_x_coords_Q3_l401_401243


namespace four_numbers_are_perfect_squares_l401_401959

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem four_numbers_are_perfect_squares (a b c d : ℕ) (h1 : is_perfect_square (a * b * c))
                                                      (h2 : is_perfect_square (a * c * d))
                                                      (h3 : is_perfect_square (b * c * d))
                                                      (h4 : is_perfect_square (a * b * d)) : 
                                                      is_perfect_square a ∧
                                                      is_perfect_square b ∧
                                                      is_perfect_square c ∧
                                                      is_perfect_square d :=
by
  sorry

end four_numbers_are_perfect_squares_l401_401959


namespace plane_satisfies_conditions_l401_401124

noncomputable def PlaneEquation (A B C D : ℤ) (x y z : ℝ) := A * x + B * y + C * z + D = 0

theorem plane_satisfies_conditions : 
  ∃ (A B C D : ℤ), A > 0 ∧ 
  gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  PlaneEquation A B C D x y z = (y + 2 * z + 7 = 0) ∧
  (∃ (x y z : ℝ), PlaneEquation x y z * (x + y + 2 * z - 4) = 0 ∧ PlaneEquation x y z * (2 * x - y + z - 1) = 0) ∧
  (∀ (x y z : ℝ), PlaneEquation A B C D 2 2 (-2) = 5 / (√6)) :=
begin
  existsi [0, 1, 2, 7],
  split,
  { exact zero_lt_one },
  split,
  { apply Int.gcd_gcd_gcd,
    repeat {
      repeat {
        rw abs_eq_self.mpr (le_of_lt (Int.zero_lt_one)) } },
    exact one_pos },
  split,
  { sorry }, -- actual proof skipped
  split,
  { sorry }, -- actual proof skipped
  { sorry }, -- actual proof skipped
end

end plane_satisfies_conditions_l401_401124


namespace sum_of_midpoint_coordinates_l401_401572

theorem sum_of_midpoint_coordinates :
  let A := (3 : ℝ, 4 : ℝ)
  let B := (9 : ℝ, 18 : ℝ)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  midpoint.1 + midpoint.2 = 17 :=
by {
  let A := (3 : ℝ, 4 : ℝ);
  let B := (9 : ℝ, 18 : ℝ);
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2);
  sorry
}

end sum_of_midpoint_coordinates_l401_401572


namespace problem_statement_l401_401738

-- Define the given expression as a function
def expression (n : ℕ) : ℚ := (n^2 + 2)! / (n! ^ (n + 2))

-- State the theorem
theorem problem_statement : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → (expression n).denom = 1 := 
begin 
  intro n, 
  assume h, 
  sorry 
end

end problem_statement_l401_401738


namespace denominator_exceeds_numerator_by_263_l401_401876

def G : ℚ := 736 / 999

theorem denominator_exceeds_numerator_by_263 : 999 - 736 = 263 := by
  -- Since 736 / 999 is the simplest form already, we simply state the obvious difference
  rfl

end denominator_exceeds_numerator_by_263_l401_401876


namespace tangent_line_value_l401_401019

-- Definitions based on conditions
def circle_center := (1 : ℝ, 0 : ℝ)
def circle_radius := 2
def line (a : ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Theorem statement based on question and correct answer
theorem tangent_line_value (a : ℝ) (h : a > 0)
  (tangent : ∀ p ∈ line a, dist p circle_center = circle_radius) : a = 3 :=
sorry

end tangent_line_value_l401_401019


namespace compute_fg_l401_401421

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg_l401_401421


namespace sign_of_f_l401_401368

theorem sign_of_f (a b c R r : ℝ) (h : a ≤ b ∧ b ≤ c) (C : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (htri : a + b > c ∧ b + c > a ∧ c + a > b)
  (hR : 2 * R = a * b * c / √((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))) -- Formula for circumradius in terms of sides
  (hr : 2 * r = 4 * R * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.sin (C / 2))) -- Expression for inradius r in terms of R and angles
  (hA : A = real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (hB : B = real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (hC : C = real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (hf : f = a + b - 2 * R - 2 * r) :
  ((C ≥ real.pi / 3 ∧ C < real.pi / 2) → f > 0) ∧
  (C = real.pi / 2 → f = 0) ∧
  ((C > real.pi / 2 ∧ C < real.pi) → f < 0) := sorry

end sign_of_f_l401_401368


namespace b_is_positive_integer_l401_401591

noncomputable def a : ℕ → ℚ
| 0       := 1
| (n + 1) := (1 / 2) * a n + (1 / (4 * a n))

def b (n : ℕ) : ℚ :=
  sqrt (2 / (2 * (a n)^2 - 1))

theorem b_is_positive_integer (n : ℕ) (h : n > 1) : ∃ k : ℤ, k > 0 ∧ b n = k := 
sorry

end b_is_positive_integer_l401_401591


namespace count_valid_quadratics_l401_401130

theorem count_valid_quadratics :
  let quadratics := { f : ℤ × ℤ × ℤ // let (a, b, c) := f in
                      a ≠ 0 ∧ b ≠ c ∧
                      -10 ≤ b ∧ b ≤ 10 ∧ -10 ≤ c ∧ c ≤ 10 ∧
                      (b + c) % 2 = 0 ∧  -- Vertex x-coordinate integer
                      -10 ≤ -a * (c - b) ^ 2 / 4 ∧ -a * (c - b) ^ 2 / 4 ≤ 10 -- Vertex y-coordinate integer constraint
                    }
  in
  quadratics.card = 510 :=
begin
  sorry
end

end count_valid_quadratics_l401_401130


namespace parabola_intersection_count_l401_401276

noncomputable def count_intersections : Nat := 2912

theorem parabola_intersection_count :
  let parabolas : List (ℤ × ℤ) := 
    (List.product 
      [-3, -2, -1, 0, 1, 2, 3].toList 
      [-4, -3, -2, -1, 1, 2, 3, 4].toList),
      P := parabolas.length
  ∃ (n : Nat), 
  P - 2 * choose (P - 2) 2 - 7 * 12 = n → 
  n = count_intersections :=
by 
  sorry

end parabola_intersection_count_l401_401276


namespace log_relation_l401_401402

theorem log_relation (x y z : ℝ) (h1 : log 3 (log (1/2) (log 2 x)) = 0)
  (h2 : log 3 (log (1/3) (log 3 y)) = 0)
  (h3 : log 3 (log (1/5) (log 5 z)) = 0) :
  z < x ∧ x < y :=
sorry

end log_relation_l401_401402


namespace bat_wings_area_l401_401921

theorem bat_wings_area :
  let E := (0, 0 : ℚ × ℚ),
      F := (3, 0 : ℚ × ℚ),
      A := (3, 4 : ℚ × ℚ),
      D := (0, 4 : ℚ × ℚ),
      C := (2, 4 : ℚ × ℚ),
      B := (3, 3 : ℚ × ℚ) in
  let Z := (12/5, 12/5 : ℚ × ℚ) in
  let area_ECZ := 1/2 * abs (0 * 4 + 2 * 12/5 + 12/5 * 0 - (0 * 2 + 4 * 12/5 + 12/5 * 0)) in
  let area_FZB := 1/2 * abs (3 * 3 + 12/5 * 0 + 3 * 12/5 - (0 * 12/5 + 3 * 3 + 12/5 * 3)) in
  area_ECZ + area_FZB = 12/5 :=
by {
  let E := (0, 0 : ℚ × ℚ),
  let F := (3, 0 : ℚ × ℚ),
  let A := (3, 4 : ℚ × ℚ),
  let D := (0, 4 : ℚ × ℚ),
  let C := (2, 4 : ℚ × ℚ),
  let B := (3, 3 : ℚ × ℚ),
  let Z := (12/5, 12/5 : ℚ × ℚ),
  let area_ECZ := 1/2 * abs (0 * 4 + 2 * 12/5 + 12/5 * 0 - (0 * 2 + 4 * 12/5 + 12/5 * 0)),
  let area_FZB := 1/2 * abs (3 * 3 + 12/5 * 0 + 3 * 12/5 - (0 * 12/5 + 3 * 3 + 12/5 * 3)),
  have : area_ECZ + area_FZB = 12/5,
  sorry
}

end bat_wings_area_l401_401921


namespace ship_B_has_highest_rt_no_cars_l401_401504

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l401_401504


namespace solve_for_x_l401_401217

theorem solve_for_x (x : ℝ) (h : (x / 5) + 3 = 4) : x = 5 :=
by
  sorry

end solve_for_x_l401_401217


namespace probability_first_spade_second_ace_l401_401190

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end probability_first_spade_second_ace_l401_401190


namespace probability_horizontal_distance_at_least_one_lemma_l401_401468

noncomputable def probability_horizontal_distance_at_least_one (T : set (ℝ × ℝ)) (side_length : ℝ) : ℝ :=
if (side_length = 2) ∧ (∀ x ∈ T, (0 ≤ x.1 ∧ x.1 ≤ 2) ∧ (0 ≤ x.2 ∧ x.2 ≤ 2)) then
  1 / 2
else
  0

theorem probability_horizontal_distance_at_least_one_lemma :
  let T := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 } in
  probability_horizontal_distance_at_least_one T 2 = 1 / 2 :=
by {
  sorry
}

end probability_horizontal_distance_at_least_one_lemma_l401_401468


namespace ant_position_after_2019_moves_l401_401636

-- Definitions of given conditions
def bipartite (V : Type) [Fintype V] (E : V → V → Prop) : Prop :=
  ∃ (U : V → bool), ∀ v w : V, E v w → U v ≠ U w

variables (V : Type) [Fintype V] (E : V → V → Prop)

-- Start is a specific vertex of type V
variable (Start : V)

-- Q is a specific vertex of type V
variable (Q : V)

-- Labeling function
variable (label : V → bool)

-- Given conditions
axiom network_bipartite : bipartite V E
axiom label_Start : label Start = tt
axiom label_Q : label Q = ff

-- The proof problem statement
theorem ant_position_after_2019_moves
  (hv : Start ≠ Q)
  (lcons : ∀ (v w : V), E v w → label v ≠ label w)
  (steps : 2019 % 2 = 1) :
  ∃ (v : V), (Start --[E]-->^[2019] v) ∧ v = Q := sorry

end ant_position_after_2019_moves_l401_401636


namespace sum_of_midpoint_coordinates_l401_401573

theorem sum_of_midpoint_coordinates :
  let A := (3 : ℝ, 4 : ℝ)
  let B := (9 : ℝ, 18 : ℝ)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  midpoint.1 + midpoint.2 = 17 :=
by {
  let A := (3 : ℝ, 4 : ℝ);
  let B := (9 : ℝ, 18 : ℝ);
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2);
  sorry
}

end sum_of_midpoint_coordinates_l401_401573


namespace sum_distinct_integers_l401_401473

theorem sum_distinct_integers (a b c d e : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h : (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120) :
    a + b + c + d + e = 13 := by
  sorry

end sum_distinct_integers_l401_401473


namespace F_final_image_correct_l401_401583

-- Define possible orientations of the letter F
inductive Orientation
| original
| rotated_90
| reflected_y
| rotated_180

-- Base and stem orientation
structure F_Position where
  base : String -- Possible values: "positive x-axis", "negative x-axis", "positive y-axis", "negative y-axis"
  stem : String -- Possible values: "positive x-axis", "negative x-axis", "positive y-axis", "negative y-axis"

-- Initial position of the letter F
def initial_position : F_Position :=
  ⟨"negative x-axis", "negative y-axis"⟩

-- Transformation functions
def rotate_90 (pos : F_Position) : F_Position :=
  ⟨if pos.base = "negative x-axis" then "negative y-axis" else "positive x-axis",
    if pos.stem = "negative y-axis" then "positive x-axis" else "negative y-axis"⟩

def reflect_y (pos : F_Position) : F_Position :=
  ⟨pos.base, if pos.stem = "positive x-axis" then "negative x-axis" else "positive x-axis"⟩

def rotate_180 (pos : F_Position) : F_Position :=
  ⟨if pos.base = "negative y-axis" then "positive y-axis" else pos.base,
    if pos.stem = "negative x-axis" then "positive x-axis" else pos.stem⟩

-- Final transformation function chain
def final_position : F_Position :=
  rotate_180 (reflect_y (rotate_90 initial_position))

-- Statement to prove
theorem F_final_image_correct : final_position = ⟨"positive y-axis", "positive x-axis"⟩ :=
sorry

end F_final_image_correct_l401_401583


namespace light_off_combinations_l401_401607

theorem light_off_combinations (k n m : ℕ) (h1 : k = 300) (h2 : n = 2020) (h3 : m = 1710):
  ∃ ways : ℕ, ways = nat.choose 1710 300 :=
by {
  have h := nat.choose_eq_formula m k,
  rw [h1, h2, h3],
  exact h
}

end light_off_combinations_l401_401607


namespace difference_in_price_l401_401226

-- Definitions based on the given conditions
def price_with_cork : ℝ := 2.10
def price_cork : ℝ := 0.05
def price_without_cork : ℝ := price_with_cork - price_cork

-- The theorem proving the given question and correct answer
theorem difference_in_price : price_with_cork - price_without_cork = price_cork :=
by
  -- Proof can be omitted
  sorry

end difference_in_price_l401_401226


namespace polynomial_g_l401_401547

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l401_401547


namespace prob_z_minus_3i_real_prob_a_b_in_circle_l401_401970

-- Definitions for the problem
def roll_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def complex_plane_prob_z_real (a b : ℕ) (z : ℂ) : Prop := b = 3
def complex_plane_prob_in_circle (a b : ℕ) : Prop := (a - 2) ^ 2 + b ^ 2 ≤ 9

-- Statement 1: Proving the probability of "z - 3i is a real number" is 1/6
theorem prob_z_minus_3i_real :
  (1 / roll_die.card : ℚ) = 1/6 := sorry

-- Statement 2: Proving the probability of "(a, b) satisfies (a - 2)^2 + b^2 ≤ 9" is 1/4
theorem prob_a_b_in_circle :
    (let valid_cases := 
        { (a, b) | a ∈ roll_die ∧ b ∈ roll_die ∧ complex_plane_prob_in_circle a b } in
        valid_cases.card / (roll_die.card * roll_die.card) : ℚ) = 1 / 4 := sorry

end prob_z_minus_3i_real_prob_a_b_in_circle_l401_401970


namespace range_of_m_l401_401783

noncomputable def f (a x : ℝ) : ℝ := -x^3 + 3 * a * x^2 - 4

def g (a x m : ℝ) : ℝ := f a x + m * x

theorem range_of_m (a b m : ℝ) (h1 : f a b = -7 / 2) (h2 : ∀ x ∈ set.Icc 0 2, deriv (g a x m) x ≤ 0) :
  m ≤ -3 / 4 :=
sorry

end range_of_m_l401_401783


namespace radius_of_inscribed_circle_l401_401224

theorem radius_of_inscribed_circle (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 10) (hc : c = 20)
  (h : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))) :
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by
  -- Statements and conditions are setup, but the proof is omitted.
  sorry

end radius_of_inscribed_circle_l401_401224


namespace condo_cats_l401_401832

theorem condo_cats (x y : ℕ) (h1 : 2 * x + y = 29) : 6 * x + 3 * y = 87 := by
  sorry

end condo_cats_l401_401832


namespace joey_run_time_one_way_l401_401092

noncomputable def time_to_run_one_way : ℝ := 
let t := 1 in
let avg_speed := 3 in
let return_speed := 6.000000000000002 in
let total_distance := 4 in
let return_time := 2 / return_speed in
let total_time := t + return_time in
avg_speed = total_distance / total_time

theorem joey_run_time_one_way : time_to_run_one_way = 1 := by
  sorry

end joey_run_time_one_way_l401_401092


namespace expression_equiv_l401_401985

theorem expression_equiv :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end expression_equiv_l401_401985


namespace unbounded_region_exists_opposite_point_l401_401533

-- Definitions
variable {Line : Type} [Nonempty Line] (divides_plane : Line → Point → Prop)
variable (A : Point) (lines : Set Line)

-- Conditions
variables (non_parallel : ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ ∃ p : Point, divides_plane l1 p ∧ divides_plane l2 p)
variables (marked_point_inside : ∃ l ∈ lines, ¬ divides_plane l A) 

-- Proof problem
theorem unbounded_region_exists_opposite_point :
  (∃ unbounded_part : Set Point, A ∈ unbounded_part ∧ ∀ part ∈ unbounded_part, ∀ l ∈ lines, ¬ divides_plane l part) ↔
  (∃ B : Point, ∀ l ∈ lines, divides_plane l A ↔ ¬ divides_plane l B) :=
sorry

end unbounded_region_exists_opposite_point_l401_401533


namespace jenna_round_trip_pay_l401_401088

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end jenna_round_trip_pay_l401_401088


namespace valid_lengths_of_polygon_DUPLAK_l401_401080

theorem valid_lengths_of_polygon_DUPLAK
  (area_DRAK : ℕ) (area_DUPE : ℕ) (area_DUPLAK : ℕ)
  (hDRAK : area_DRAK = 44)
  (hDUPE : area_DUPE = 64)
  (hDUPLAK : area_DUPLAK = 92) :
  ∃ (DR DE DU DK PL LA : ℕ),
    (DR * DE = 16) ∧ (DR * DK = 44) ∧ (DU * DE = 64) ∧ 
    ((DK - DE) = LA) ∧ ((DU - DR) = PL) ∧
    ({(DR, DE, DU, DK, PL, LA)} = 
      { (1, 16, 4, 44, 3, 28), (2, 8, 8, 22, 6, 14), (4, 4, 16, 11, 12, 7) }) :=
sorry

end valid_lengths_of_polygon_DUPLAK_l401_401080


namespace fill_blank_correct_l401_401931

def sentence := "Sometimes, " ++ _ ++ " we show our gratitude to a person is reflected in the kind of food we serve him or her."

def options := ["when", "whether", "why", "how"]

def correct_option := "how"

theorem fill_blank_correct :
  ∃ option ∈ options, option = correct_option :=
by
  existsi correct_option
  split
  . simp [options]
  . refl

-- Skip proof
sorry

end fill_blank_correct_l401_401931


namespace necessarily_negative_b_plus_3b_squared_l401_401145

theorem necessarily_negative_b_plus_3b_squared
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 1) :
  b + 3 * b^2 < 0 :=
sorry

end necessarily_negative_b_plus_3b_squared_l401_401145


namespace triangle_dimensions_l401_401073

-- Define the problem in Lean 4
theorem triangle_dimensions (a m : ℕ) (h₁ : a = m + 4)
  (h₂ : (a + 12) * (m + 12) = 10 * a * m) : 
  a = 12 ∧ m = 8 := 
by
  sorry

end triangle_dimensions_l401_401073


namespace kate_age_correct_l401_401869

def total_age (kate_age maggie_age sue_age : ℕ) : ℕ := kate_age + maggie_age + sue_age

variable (kate_age : ℕ)

-- Conditions
def maggie_age : ℕ := 17
def sue_age : ℕ := 12
def combined_age : ℕ := 48

-- Proposition
theorem kate_age_correct : kate_age = 19 :=
by
  -- State the equivalence given conditions
  have h : total_age kate_age maggie_age sue_age = combined_age := by sorry
  -- Using the conditions to prove kate_age = 19
  have h_sum : maggie_age + sue_age = 29 := by sorry
  -- Subtract Maggie's and Sue's age from total_age to find Kate's age
  calc kate_age 
    = combined_age - (maggie_age + sue_age) : by sorry
    = 19 : by sorry

end kate_age_correct_l401_401869


namespace knights_count_l401_401610

theorem knights_count (n : ℕ) (h₁ : ∀ i : ℕ, 0 < i ∧ i < 61 →
                    (∃ k : ℕ, 0 < k ∧ k < 6 ∧ (knight i → k ≤ n ∧ liar i → k ≥ n)))
                    (h₂ : ∀ i : ℕ, knight i ∨ liar i) : 
  ∑ i in range 60, (if knight i then 1 else 0) = 40 := 
sorry

end knights_count_l401_401610


namespace pears_total_correct_l401_401924

noncomputable def pickedPearsTotal (sara_picked tim_picked : Nat) : Nat :=
  sara_picked + tim_picked

theorem pears_total_correct :
    pickedPearsTotal 6 5 = 11 :=
  by
    sorry

end pears_total_correct_l401_401924


namespace proj_b_l401_401880

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end proj_b_l401_401880


namespace possible_polynomials_l401_401552

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l401_401552


namespace polynomial_zero_value_l401_401477

noncomputable def P (n : ℕ) : polynomial ℝ := sorry

theorem polynomial_zero_value (P : polynomial ℝ) (n : ℕ)
  (hdeg : P.degree = n) 
  (hvals : ∀ k ∈ { 1, 2, 4, ..., 2^n }, P.eval k = 1 / k) : 
  P.eval 0 = 2 - 1 / 2^n := 
sorry

end polynomial_zero_value_l401_401477


namespace sum_of_solutions_eq_one_l401_401595

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l401_401595


namespace Catalan_quotient_integer_l401_401195

theorem Catalan_quotient_integer (n : ℕ) : 
  ∃ (k : ℕ), k = (nat.factorial (2 * n)) / (nat.factorial n * nat.factorial n * (n + 1)) :=
begin
  sorry
end

end Catalan_quotient_integer_l401_401195


namespace subsequences_divisibility_condition_l401_401306

theorem subsequences_divisibility_condition (n : ℕ) (a : ℕ → ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → i + 1 ∣ 2 * (∑ j in Finset.range i, a j)) →
  (a = (λ k, k + 1) ∨ a = (λ k, if k = 0 then 2 else if k = 1 then 1 else k + 1)) :=
sorry

end subsequences_divisibility_condition_l401_401306


namespace selection_count_l401_401653

theorem selection_count :
  let english_only := 3
  let japanese_only := 2
  let bilingual := 2
  let total_english_ways := (Nat.choose 3 3) + (Nat.choose 3 2 * Nat.choose 2 1)
  let total_japanese_ways := (Nat.choose 2 2) + (Nat.choose 2 1 * Nat.choose 2 1)
  let total_selection_ways := total_english_ways * total_japanese_ways
  total_selection_ways = 27 :=
by
  sorry

end selection_count_l401_401653


namespace sqrt_domain_l401_401172

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end sqrt_domain_l401_401172


namespace sum_of_distances_from_focus_to_intersections_l401_401470

theorem sum_of_distances_from_focus_to_intersections
    (a b c : ℝ) :
    let d := -(a + b + c),
        focus := (0, 1 / 4 : ℝ × ℝ)
    in
    {dist_1 := abs (a^2 - (1 / 4)),
     dist_2 := abs (b^2 - (1 / 4)),
     dist_3 := abs (c^2 - (1 / 4)),
     dist_4 := abs (d^2 - (1 / 4))}
    (a = 5 ∧ a^2 = 25) ∧ (b = 1 ∧ b^2 = 1) ∧ (c = -6 ∧ c^2 = 36)
    → abs (a^2 - (1 / 4)) + abs (b^2 - (1 / 4)) + abs (c^2 - (1 / 4)) + abs (d^2 - (1 / 4)) = 61.5 :=
begin
    intro h,
    sorry
end

end sum_of_distances_from_focus_to_intersections_l401_401470


namespace any_point_has_barycentric_coords_unique_barycentric_coords_l401_401118

variables {A1 A2 A3 X : Type} 
variables [barycentric_space A1 A2 A3 X]

-- Define the barycentric coordinates m1, m2, m3 such that m1 + m2 + m3 = 1
variables (m1 m2 m3 : ℝ)
variable [h1 : m1 + m2 + m3 = 1]

--Proof that any point X has barycentric coordinates
theorem any_point_has_barycentric_coords : ∃ (m1 m2 m3 : ℝ), X = m1 • A1 + m2 • A2 + m3 • A3 :=
by sorry

--Proof that barycentric coordinates are unique
theorem unique_barycentric_coords (m1 m2 m3 m1' m2' m3' : ℝ) 
    (h2 : X = m1 • A1 + m2 • A2 + m3 • A3)
    (h3 : X = m1' • A1 + m2' • A2 + m3' • A3)
    (h4 : m1 + m2 + m3 = 1)
    (h5 : m1' + m2' + m3' = 1) : 
    m1 = m1' ∧ m2 = m2' ∧ m3 = m3' :=
by sorry

end any_point_has_barycentric_coords_unique_barycentric_coords_l401_401118


namespace tan_half_pos_of_second_quadrant_l401_401057

-- Given definition: θ is an angle in the second quadrant
def second_quadrant (θ : ℝ) : Prop :=
  π / 2 < θ ∧ θ < π

-- Proof that if θ is an angle in the second quadrant, then tan(θ / 2) is positive
theorem tan_half_pos_of_second_quadrant {θ : ℝ} (h : second_quadrant θ) : 
  Real.tan (θ / 2) > 0 :=
by
  sorry

end tan_half_pos_of_second_quadrant_l401_401057


namespace find_m_l401_401181

theorem find_m
  (x y : ℝ)
  (h1 : 100 = 300 * x + 200 * y)
  (h2 : 120 = 240 * x + 300 * y)
  (h3 : ∃ m : ℝ, 50 * 3 = 150 * x + m * y):
  ∃ m : ℝ, m = 450 :=
by
  sorry

end find_m_l401_401181


namespace ship_with_highest_no_car_round_trip_percentage_l401_401516

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l401_401516


namespace possible_polynomials_l401_401550

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l401_401550


namespace cube_root_of_2x_minus_23_l401_401397

theorem cube_root_of_2x_minus_23 (x : ℝ) (h : sqrt (2 * x - 1) = 7 ∨ sqrt (2 * x - 1) = -7) : real.cbrt (2 * x - 23) = 3 :=
by
  sorry

end cube_root_of_2x_minus_23_l401_401397


namespace sum_of_all_possible_two_digit_values_of_d_l401_401107

theorem sum_of_all_possible_two_digit_values_of_d :
  (∑ d in {d : ℕ | d ∣ (143 - 5) ∧ 10 ≤ d ∧ d < 100}, d) = 115 :=
by
  sorry

end sum_of_all_possible_two_digit_values_of_d_l401_401107


namespace triangle_angle_B_max_value_l401_401434

theorem triangle_angle_B_max_value (a b c : ℝ) 
  (h : ∀ x, (deriv (λ x, (1 / 3 : ℝ) * x^3 + b * x^2 + (a^2 + c^2 + real.sqrt 3 * a * c) * x) x = 0 → false)) :
  ∠B = 5 * real.pi / 6 :=
begin
  sorry
end

end triangle_angle_B_max_value_l401_401434


namespace fourth_student_guess_l401_401566

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l401_401566


namespace tan_add_pi_over_3_l401_401051

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by
  sorry

end tan_add_pi_over_3_l401_401051


namespace find_n_cosine_l401_401312

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l401_401312


namespace product_of_chords_is_correct_l401_401466

theorem product_of_chords_is_correct :
  let P := (Complex.ofReal 3)
  let Q := (Complex.ofReal (-3))
  let ω := Complex.exp (Real.pi * Complex.I / 5)
  let D1 := 3 * ω
  let D2 := 3 * ω^2
  let D3 := 3 * ω^3
  P = 3 ∧ Q = -3 ∧
  D1 = 3 * Complex.exp (Real.pi * Complex.I / 5) ∧
  D2 = 3 * (Complex.exp (Real.pi * Complex.I / 5))^2 ∧
  D3 = 3 * (Complex.exp (Real.pi * Complex.I / 5))^3 →
  (let PD1 := Complex.abs (P - D1)
   let PD2 := Complex.abs (P - D2)
   let PD3 := Complex.abs (P - D3)
   let QD1 := Complex.abs (Q - D1)
   let QD2 := Complex.abs (Q - D2)
   let QD3 := Complex.abs (Q - D3)
  in PD1 * PD2 * PD3 * QD1 * QD2 * QD3 = 3645) := 
begin
  intros P Q ω D1 D2 D3 h,
  sorry
end

end product_of_chords_is_correct_l401_401466
