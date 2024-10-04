import Mathlib

namespace savings_by_end_of_2019_l396_396633

variable (income_monthly : ℕ → ℕ) (expenses_monthly : ℕ → ℕ)
variable (initial_savings : ℕ)

noncomputable def total_income : ℕ :=
  (income_monthly 9 + income_monthly 10 + income_monthly 11 + income_monthly 12) * 4

noncomputable def total_expenses : ℕ :=
  (expenses_monthly 9 + expenses_monthly 10 + expenses_monthly 11 + expenses_monthly 12) * 4

noncomputable def final_savings (initial_savings : ℕ) (total_income : ℕ) (total_expenses : ℕ) : ℕ :=
  initial_savings + total_income - total_expenses

theorem savings_by_end_of_2019 :
  (income_monthly 9 = 55000) →
  (income_monthly 10 = 45000) →
  (income_monthly 11 = 10000) →
  (income_monthly 12 = 17400) →
  (expenses_monthly 9 = 40000) →
  (expenses_monthly 10 = 20000) →
  (expenses_monthly 11 = 5000) →
  (expenses_monthly 12 = 2000) →
  initial_savings = 1147240 →
  final_savings initial_savings total_income total_expenses = 1340840 :=
by
  intros h_income_9 h_income_10 h_income_11 h_income_12
         h_expenses_9 h_expenses_10 h_expenses_11 h_expenses_12
         h_initial_savings
  rw [final_savings, total_income, total_expenses]
  rw [h_income_9, h_income_10, h_income_11, h_income_12]
  rw [h_expenses_9, h_expenses_10, h_expenses_11, h_expenses_12]
  rw h_initial_savings
  sorry

end savings_by_end_of_2019_l396_396633


namespace avg_length_remaining_ropes_l396_396320

theorem avg_length_remaining_ropes :
  ∀ (lengths : list ℕ) (lengths1 : list ℕ),
    lengths.length = 9 ∧
    (∑ x in lengths, x) / 9 = 90 ∧
    lengths1.length = 3 ∧
    (∑ x in lengths1, x) / 3 = 70 ∧
    list.zipWith (λ x y, (x : ℕ, y : ℕ)) lengths1 [2, 3, 5].map (λ r, r * 21) = lengths1 →
    (∑ x in (lengths.diff lengths1), x) / 6 = 100 := sorry

end avg_length_remaining_ropes_l396_396320


namespace problem1_problem2_l396_396960

-- Definitions of conditions
variables {A B C : ℝ} {a b c : ℝ}
variable (h1 : c = 2)
variable (h2 : C = real.pi / 3)  -- 60 degrees in radians

-- First problem: Find value of (a + b) / (sin A + sin B)
theorem problem1 : (a + b) / (real.sin A + real.sin B) = 4 * real.sqrt 3 / 3 :=
begin
  sorry
end

-- Second problem: Given a + b = ab, find the area of triangle ABC
theorem problem2 (h3 : a + b = a * b) : 
  (1 / 2) * a * b * real.sin C = real.sqrt 3 :=
begin
  sorry
end

end problem1_problem2_l396_396960


namespace ratio_of_books_to_bookmarks_l396_396477

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem ratio_of_books_to_bookmarks (books bookmarks : ℕ) (h_books : books = 72) (h_bookmarks : bookmarks = 16) : 
  let common_divisor := gcd books bookmarks in
  (books / common_divisor = 9) ∧ (bookmarks / common_divisor = 2) :=
by
  sorry

end ratio_of_books_to_bookmarks_l396_396477


namespace number_of_subsets_containing_7_l396_396572

-- Define the set
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the subset condition
def contains_7 (A : Set ℕ) : Prop := 7 ∈ A

-- Define the count of subsets containing 7
def count_subsets_containing_7 : ℕ := 
  (Set.powerset S).filter contains_7).card

-- The theorem statement
theorem number_of_subsets_containing_7 : count_subsets_containing_7 = 64 := by
  sorry

end number_of_subsets_containing_7_l396_396572


namespace trajectory_and_focus_proof_l396_396257

open Real

variable {x₀ y₀ x y : ℝ}

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def foot_of_perpendicular (M N : ℝ × ℝ) : Prop := N.1 = M.1 ∧ N.2 = 0

def point_P_conditions (N M P : ℝ × ℝ) : Prop := 
  P.1 - N.1 = 0 ∧ P.2 = sqrt 2 * M.2 ∧ N.1 = M.1 ∧ N.2 = 0

def line_x_eq_neg3 (Q : ℝ × ℝ) : Prop := Q.1 = -3

def vector_dot (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

def vector_OP_dot_vector_PQ_eq1 (O P Q : ℝ × ℝ) : Prop := 
  vector_dot (O.1, O.2) (P.1 - Q.1, P.2 - Q.2) = 1

def left_focus (F : ℝ × ℝ) : Prop := F = (-1, 0)

theorem trajectory_and_focus_proof :
  ∀ (M N P Q F : ℝ × ℝ), 
    ellipse M.1 M.2 →
    foot_of_perpendicular M N →
    point_P_conditions N M P →
    line_x_eq_neg3 Q →
    vector_OP_dot_vector_PQ_eq1 (0, 0) P Q →
    left_focus F →
    P.1^2 + P.2^2 = 2 ∧ 
    (let line_PF_perp_OQ (F' O' : ℝ × ℝ) : Prop := 
      sorry in
      line_PF_perp_OQ P F) :=
sorry

end trajectory_and_focus_proof_l396_396257


namespace calc_value_l396_396023

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l396_396023


namespace fractional_part_subtraction_l396_396235

noncomputable def t := 1 / (2 - real.sqrt 3)
noncomputable def a := t - real.floor t
noncomputable def b := -t - real.floor (-t)

theorem fractional_part_subtraction : 
  (1 / (2 * b) - 1 / a = 1 / 2) :=
by sorry

end fractional_part_subtraction_l396_396235


namespace angle_B_leq_60_l396_396969

variable (A B C M H : Type) [Triangle A B C] [AcuteAngled A B C]
variable (angle_B : Angle A B C)
variable (altitude_AH : Altitude A H C) (median_BM : Median B M)

theorem angle_B_leq_60 (h : height_altitude_AH = median_BM) : angle_B ≤ 60 :=
by
  sorry

end angle_B_leq_60_l396_396969


namespace total_amount_correct_l396_396003

noncomputable def rupees_to_paisa (r : ℝ): ℝ := 100 * r

def w_gets (W : ℝ) : ℝ := W
def x_gets (W : ℝ) : ℝ := 0.75 * W
def y_gets (W : ℝ) : ℝ := 0.45 * W
def z_gets (W : ℝ) : ℝ := 0.30 * W

def total_amount (W : ℝ) : ℝ :=
  w_gets W + x_gets W + y_gets W + z_gets W

theorem total_amount_correct (W : ℝ) (hy : y_gets W = 36) : total_amount W = 200 := 
  by have hw : W = 36 / 0.45 := by sorry
     have total := 2.50 * W
     show total = 200
     sorry

end total_amount_correct_l396_396003


namespace problem_statement_l396_396998

noncomputable def g (x : ℝ) : ℝ := sorry

theorem problem_statement :
  (∀ x, g(x) ≥ 0) ∧ (∀ a b, g(a) * g(b) = g(a * b)) →
  g(1) = 1 ∧ (∀ a ≠ 0, g(1/a) = 1 / g(a)) ∧ (∀ a, g(a) = (g(a^3))^(1/3)) := 
by 
  intros h
  have h₁ : ∀ x, g x ≥ 0 := h.1
  have h₂ : ∀ a b, g(a) * g(b) = g(a * b) := h.2

  -- To be proved:
  -- Prove g(1) = 1
  -- Prove ∀ a ≠ 0, g(1/a) = 1 / g(a)
  -- Prove ∀ a, g(a) = (g(a^3))^(1/3)
  sorry  -- Proof is omitted

end problem_statement_l396_396998


namespace total_notes_l396_396407

theorem total_notes (total_amount : ℤ) (num_50_notes : ℤ) (value_50 : ℤ) (value_500 : ℤ) (total_notes : ℤ) :
  total_amount = num_50_notes * value_50 + (total_notes - num_50_notes) * value_500 → 
  total_amount = 10350 → num_50_notes = 77 → value_50 = 50 → value_500 = 500 → total_notes = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_notes_l396_396407


namespace orthocenter_of_triangle_l396_396222

theorem orthocenter_of_triangle (A B C M : Point) 
  (hMAB_MCB : ∠MAB = ∠MCB) 
  (hMBA_MCA : ∠MBA = ∠MCA) : 
  is_orthocenter M A B C :=
sorry

end orthocenter_of_triangle_l396_396222


namespace imaginary_part_neg_one_l396_396094

noncomputable def complex_imaginary_part : ℂ := (2 * complex.I^3) / (1 - complex.I)

theorem imaginary_part_neg_one : complex.im complex_imaginary_part = -1 :=
by sorry

end imaginary_part_neg_one_l396_396094


namespace Sam_age_l396_396287

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l396_396287


namespace abel_arrives_earlier_by_225_minutes_l396_396006

noncomputable def abel_time : ℝ :=
  let first_part := 200 / 50
  let second_part := 100 / 40
  let third_part := (1000 - 200 - 100) / 50
  let break := 0.5
  first_part + second_part + third_part + break

noncomputable def alice_time : ℝ :=
  let total_distance := 1000 + 30
  total_distance / 40 - 1

theorem abel_arrives_earlier_by_225_minutes :
  (alice_time - abel_time) * 60 = 225 := by
  sorry

end abel_arrives_earlier_by_225_minutes_l396_396006


namespace distance_between_first_and_last_tree_l396_396082

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h₁ : n = 8)
  (h₂ : d = 75)
  : (d / ((4 - 1) : ℕ)) * (n - 1) = 175 := sorry

end distance_between_first_and_last_tree_l396_396082


namespace gretchen_socks_probability_l396_396910

theorem gretchen_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let choose_colors := Nat.choose colors (colors - 1)
  let choose_pair_color := Nat.choose (colors - 1) 1
  let ways_to_choose := choose_colors * choose_pair_color * (2 ^ 3)
  let probability := (ways_to_choose : ℚ) / total_combinations
  in probability = 10 / 63 := by
  sorry

end gretchen_socks_probability_l396_396910


namespace original_problem_theorem_l396_396243

-- Define the third sequence in general
def third_sequence_term (j : ℕ) (n : ℕ) (hj : 1 ≤ j ∧ j ≤ n-2) : ℕ :=
  4 * j + 4

-- Define the term in the last sequence given the provided recurrence relation
noncomputable def last_sequence_term (n : ℕ) : ℕ :=
  (n + 1) * 2^(n - 2)

-- Main theorem combining both parts
theorem original_problem_theorem (n j : ℕ) (hn : 1 ≤ n) (hj : 1 ≤ j ∧ j ≤ n-2) :
  third_sequence_term j n hj = 4 * j + 4 ∧ last_sequence_term n = (n + 1) * 2^(n - 2) :=
begin
  split,
  { -- Proof for the third sequence term is not needed
    sorry },
  { -- Proof for the term in the last sequence is not needed
    sorry },
end

end original_problem_theorem_l396_396243


namespace integer_pairs_satisfy_equation_l396_396562

theorem integer_pairs_satisfy_equation :
  ∃ (S : Finset (ℤ × ℤ)), S.card = 5 ∧ ∀ (m n : ℤ), (m, n) ∈ S ↔ m^2 + n = m * n + 1 :=
by
  sorry

end integer_pairs_satisfy_equation_l396_396562


namespace find_constant_and_max_term_l396_396892

noncomputable def expansion := (1/x + 6*x)^n
noncomputable def sum_binomial_coeff := ∑ k in finset.range (n + 1), binomial n k * (1/x)^k * (6*x)^(n - k)

theorem find_constant_and_max_term (n : ℕ) (h : 2^n = 256) :
  (n = 8 ∧ nat.choose 8 6 * 2^6 = 1792) ∧
  (∃ r, r = 4 ∧ nat.choose 8 4 * 2^4 * x^(-8 / 3) = 1120 * x^(-8 / 3)) :=
by
  sorry

end find_constant_and_max_term_l396_396892


namespace part_I_part_II_l396_396165

noncomputable def f (x : ℝ) := |x - 2| - |2 * x + 1|

theorem part_I :
  { x : ℝ | f x ≤ 0 } = { x : ℝ | x ≤ -3 ∨ x ≥ (1 : ℝ) / 3 } :=
by
  sorry

theorem part_II :
  ∀ x : ℝ, f x - 2 * m^2 ≤ 4 * m :=
by
  sorry

end part_I_part_II_l396_396165


namespace three_digit_cubes_divisible_by_16_l396_396580

theorem three_digit_cubes_divisible_by_16 :
  (count (λ n : ℕ, 4 * n = n ∧ (100 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 999)) {n | 1 ≤ n ∧ n ≤ 2}) = 1 :=
sorry

end three_digit_cubes_divisible_by_16_l396_396580


namespace solve_system_l396_396123

theorem solve_system : ∀ (x y : ℝ), x + 2 * y = 1 ∧ 2 * x + y = 2 → x + y = 1 :=
by
  intros x y h
  cases h with h₁ h₂
  sorry

end solve_system_l396_396123


namespace find_m_l396_396904

theorem find_m (x p q m : ℝ) 
    (h1 : 4 * p^2 + 9 * q^2 = 2) 
    (h2 : (1/2) * x + 3 * p * q = 1) 
    (h3 : ∀ x, x^2 + 2 * m * x - 3 * m + 1 ≥ 1) :
    m = -3 ∨ m = 1 :=
sorry

end find_m_l396_396904


namespace area_of_each_small_concave_quadrilateral_l396_396789

noncomputable def inner_diameter : ℝ := 8
noncomputable def outer_diameter : ℝ := 10
noncomputable def total_area_covered_by_annuli : ℝ := 112.5
noncomputable def pi : ℝ := 3.14

theorem area_of_each_small_concave_quadrilateral (inner_diameter outer_diameter total_area_covered_by_annuli pi: ℝ)
    (h1 : inner_diameter = 8)
    (h2 : outer_diameter = 10)
    (h3 : total_area_covered_by_annuli = 112.5)
    (h4 : pi = 3.14) :
    (π * (outer_diameter / 2) ^ 2 - π * (inner_diameter / 2) ^ 2) * 5 - total_area_covered_by_annuli / 4 = 7.2 := 
sorry

end area_of_each_small_concave_quadrilateral_l396_396789


namespace kit_costs_more_l396_396782

-- Defining the individual prices of the filters and the kit price
def price_filter1 := 16.45
def price_filter2 := 14.05
def price_filter3 := 19.50
def kit_price := 87.50

-- Calculating the total price of the filters if bought individually
def total_individual_price := (2 * price_filter1) + (2 * price_filter2) + price_filter3

-- Calculate the amount saved
def amount_saved := total_individual_price - kit_price

-- The theorem to show the amount saved 
theorem kit_costs_more : amount_saved = -7.00 := by
  sorry

end kit_costs_more_l396_396782


namespace angle_terminal_side_equiv_l396_396012

theorem angle_terminal_side_equiv (α : ℝ) (k : ℤ) :
  (∃ k : ℤ, α = 30 + k * 360) ↔ (∃ β : ℝ, β = 30 ∧ α % 360 = β % 360) :=
by
  sorry

end angle_terminal_side_equiv_l396_396012


namespace nine_digit_not_perfect_square_l396_396440

theorem nine_digit_not_perfect_square (D : ℕ) (h1 : 100000000 ≤ D) (h2 : D < 1000000000)
  (h3 : ∀ c : ℕ, (c ∈ D.digits 10) → (c ≠ 0)) (h4 : D % 10 = 5) :
  ¬ ∃ A : ℕ, D = A ^ 2 := 
sorry

end nine_digit_not_perfect_square_l396_396440


namespace count_n_with_product_zero_l396_396105

theorem count_n_with_product_zero : 
  (Finset.filter (λ n, 1 ≤ n ∧ n ≤ 2016 ∧ 
    ∃ k, 0 ≤ k ∧ k < n ∧ (2 + Real.cos (4 * Real.pi * k / n) + Real.sin (4 * Real.pi * k / n) * Complex.i) ^ n = 1) 
      (Finset.range (2017))).card = 504 :=
sorry

end count_n_with_product_zero_l396_396105


namespace an_arithmetic_seq_sum_cn_l396_396977

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {c : ℕ → ℝ}
variable {T : ℕ → ℝ}

axiom pos_seq : ∀ n : ℕ, n > 0 → a n > 0
axiom sum_seq : ∀ n : ℕ, n > 0 → S n = ∑ i in finset.range n, a (i + 1)
axiom an_def : ∀ n : ℕ, n > 0 → a n = 2 * real.sqrt (S n) - 1
axiom geom_seq_b : ∀ n : ℕ, n > 0 → b n = 2 - 1 / 2^(n - 1)

theorem an_arithmetic_seq :
  (∀ n : ℕ, n > 0 → a n = 2 * n - 1) :=
sorry

axiom cn_def : ∀ n : ℕ, n > 0 → c n = a n * b n

theorem sum_cn :
  (∀ n : ℕ, n > 0 → (∑ i in finset.range n, c (i + 1)) = 2 * n^2 + (2 * n + 3) / 2^(n - 1) - 6) :=
sorry

end an_arithmetic_seq_sum_cn_l396_396977


namespace trajectory_midpoint_l396_396533

variable (P : ℝ × ℝ)
variable (A : ℝ × ℝ)
variable (M : ℝ × ℝ)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 16

def midpoint (A P : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + P.1) / 2, (A.2 + P.2) / 2)

theorem trajectory_midpoint
  (hA : A = (12, 0))
  (hP : on_circle P) :
  ∃ (M : ℝ × ℝ), M = midpoint A P ∧ (M.1 - 6)^2 + M.2^2 = 4 :=
begin
  sorry
end

end trajectory_midpoint_l396_396533


namespace solution_set_for_f_l396_396183

theorem solution_set_for_f (
  f : ℝ → ℝ,
  h : ∀ x, f x = 3 - 2 * x
) : { x : ℝ | |f (x + 1) + 2| ≤ 3 } = set.Icc 0 3 :=
by
  sorry

end solution_set_for_f_l396_396183


namespace linear_approx_threshold_correct_l396_396716

noncomputable def linear_approx_threshold : ℝ := 3 - 2 * Real.sqrt 2

theorem linear_approx_threshold_correct :
  ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 2) → (∃ (k : ℝ), | 3 - (x + 2 / x) | ≤ k ∧ k = linear_approx_threshold) :=
by
  intros x hx
  sorry

end linear_approx_threshold_correct_l396_396716


namespace num_subsets_containing_7_l396_396565

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem num_subsets_containing_7 : (S.filter (λ s => 7 ∈ s)).card = 64 := by
  sorry

end num_subsets_containing_7_l396_396565


namespace classroom_student_count_l396_396339

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l396_396339


namespace general_term_formula_l396_396554

def sequence (n : ℕ) : ℚ :=
  match n with
  | 1 => -1 / 2
  | 2 => 1 / 4
  | 3 => -1 / 8
  | 4 => 1 / 16
  | _ => 0

theorem general_term_formula (n : ℕ) : sequence (n) = (-1) ^ (n + 1) / 2 ^ n := by
  sorry

end general_term_formula_l396_396554


namespace slope_difference_l396_396616

open Real

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  if p1.1 = p2.1 then 0
  else (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_difference :
  let A := (4, 10)
  let B := (0, 0)
  let C := (12, 0)
  let D := midpoint A B
  let E := midpoint B C
  slope A D - slope C E = 2.5 :=
by
  let A := (4, 10)
  let B := (0, 0)
  let C := (12, 0)
  let D := midpoint A B
  let E := midpoint B C
  have hD : D = (2, 5) := rfl
  have hE : E = (6, 0) := rfl
  have h_slope_AD : slope A D = 2.5 := rfl
  have h_slope_CE : slope C E = 0 := rfl
  calc
    slope A D - slope C E = 2.5 - 0 := by rw [h_slope_AD, h_slope_CE]
    ... = 2.5 := by norm_num

end slope_difference_l396_396616


namespace select_at_least_one_first_class_part_l396_396391

theorem select_at_least_one_first_class_part (total_parts first_class second_class selected_parts : ℕ) 
  (h_total : total_parts = 20) (h_first_class : first_class = 16) 
  (h_second_class : second_class = 4) (h_selected_parts : selected_parts = 3) :
  (Nat.choose first_class 1 * Nat.choose second_class (selected_parts - 1 ) +
  (Nat.choose first_class 2 * Nat.choose second_class (selected_parts - 2) + 
  Nat.choose first_class selected_parts)) = 1136 :=
by
  rw [h_total, h_first_class, h_second_class, h_selected_parts]
  sorry

end select_at_least_one_first_class_part_l396_396391


namespace original_function_is_l396_396858

noncomputable def translated_function (x : ℝ) : ℝ :=
  (x - 4)^2 + 2 * (x - 4) - 2 + 3

theorem original_function_is (x : ℝ) :
  let y := translated_function 1 in
  let slope := -4 in
  4 * 1 + y - 8 = 0 →
  ∃ b c, (translated_function x) = (x + 3)^2 + b * (x + 3) + c ∧ b = 2 ∧ c = -2 :=
sorry

end original_function_is_l396_396858


namespace g_one_eq_l396_396252

theorem g_one_eq {a b c d : ℝ} (ha : 1 < a) (hb : a < b) (hc : b < c) (hd : c < d)
  (g f : ℝ → ℝ)
  (h_f : f = λ x, x^4 + a * x^3 + b * x^2 + c * x + d)
  (h_g : g = λ x, (x - 1 / (roots f).nth 0) * (x - 1 / (roots f).nth 1) * (x - 1 / (roots f).nth 2) * (x - 1 / (roots f).nth 3))
  (hx4f : ∀ x, f x = (x - p) * (x - q) * (x - r) * (x - s)) :
  g 1 = (1 + a + b + c + d) / d := 
sorry

end g_one_eq_l396_396252


namespace construct_x1_4_axis_l396_396346

-- Define the conditions and the point P with its coordinates
variables (P P' P'': ℝ) 
  (plane1 plane2: Type) 
  (dihedral_angle: ℝ)
  (Cone: Type) 

-- State the problem conditions
constants 
  (is_second_projection_plane : plane2)
  (point_P : P = (P', P''))
  (fourth_dihedral_angle : dihedral_angle = 60)

-- Define the theorem statement
theorem construct_x1_4_axis
  (h1 : is_second_projection_plane)
  (h2 : point_P)
  (h3 : fourth_dihedral_angle):
  ∃ (x1_4_axis : Type), 
    tangent_plane (Cone) (x1_4_axis) ∧ 
    intersects_projection_plane (plane2) (x1_4_axis) :=
sorry

end construct_x1_4_axis_l396_396346


namespace probability_neither_mix_l396_396784

noncomputable def total_buyers : ℕ := 100
noncomputable def cake_buyers : ℕ := 50
noncomputable def muffin_buyers : ℕ := 40
noncomputable def both_buyers : ℕ := 19

theorem probability_neither_mix :
  let neither_buyers := total_buyers - (cake_buyers + muffin_buyers - both_buyers)
  let probability := (neither_buyers : ℝ) / total_buyers
  probability = 29 / 100 :=
by 
  let neither_buyers := total_buyers - (cake_buyers + muffin_buyers - both_buyers)
  let probability := (neither_buyers : ℝ) / total_buyers
  have h : neither_buyers = 29 := by sorry
  rw h
  norm_num

end probability_neither_mix_l396_396784


namespace reduced_price_is_correct_l396_396763

variable (P : ℝ)
variable (reduced_price_per_dozen : ℝ)

-- Conditions
def original_price_per_dozen := P
def reduction_percentage := 0.4
def price_after_reduction := (1 - reduction_percentage) * original_price_per_dozen
def dozens_of_bananas_for_40_rs : ℝ := 40 / price_after_reduction
def additional_bananas_in_dozens := 66 / 12 -- 66 bananas more = 5.5 dozens

-- Proof statement
theorem reduced_price_is_correct
  (h1 : reduced_price_per_dozen = price_after_reduction)
  (h2 : additional_bananas_in_dozens = dozens_of_bananas_for_40_rs - 40 / original_price_per_dozen)
  : reduced_price_per_dozen = 0.6 * 12.12 :=
by
  sorry

end reduced_price_is_correct_l396_396763


namespace fixed_point_of_gf_l396_396359

noncomputable def f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : ℝ → ℝ :=
  λ x, a^x + 1

noncomputable def g (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : ℝ → ℝ :=
  λ x, a^(x-2) + 1

theorem fixed_point_of_gf (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : g a h₀ h₁ (f a h₀ h₁ 0) = 2 :=
by
  sorry

end fixed_point_of_gf_l396_396359


namespace binom_7_2_eq_21_l396_396048

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396048


namespace Alex_shirt_count_l396_396429

variables (Ben_shirts Joe_shirts Alex_shirts : ℕ)

-- Conditions from the problem
def condition1 := Ben_shirts = 15
def condition2 := Ben_shirts = Joe_shirts + 8
def condition3 := Joe_shirts = Alex_shirts + 3

-- Statement to prove
theorem Alex_shirt_count : condition1 ∧ condition2 ∧ condition3 → Alex_shirts = 4 :=
by
  intros h
  sorry

end Alex_shirt_count_l396_396429


namespace bear_problem_l396_396459

-- Definitions of the variables
variables (W B Br : ℕ)

-- Given conditions
def condition1 : B = 2 * W := sorry
def condition2 : B = 60 := sorry
def condition3 : W + B + Br = 190 := sorry

-- The proof statement
theorem bear_problem : Br - B = 40 :=
by
  -- we would use the given conditions to prove this statement
  sorry

end bear_problem_l396_396459


namespace equilateral_triangle_properties_l396_396081

-- Define the main context and parameters
def equilateral_triangle (n : ℕ) (a b c : ℝ) : Prop :=
∀ rhombus (v1 v2 v3 v4 : ℝ), v1 + v3 = v2 + v4 

def nodes_distance (n : ℕ) (a b c : ℝ) (r : ℝ) : Prop :=
r = if a = b ∧ b = c then 0
    else if a < b ∧ b < c then 1
    else if (a < b ∧ b = c ∨ a = b ∧ b < c) ∧ even n then sqrt 3 / 2
    else if (a < b ∧ b = c ∨ a = b ∧ b < c) ∧ ¬ (even n) then 1 / (2 * n) * sqrt (3 * n^2 + 1)
    else 0 -- default case, should not occur in well-formed problem

def total_sum (n : ℕ) (a b c : ℝ) (S : ℝ) : Prop :=
S = 1 / 6 * n * (n + 1) * (a + b + c)

-- Main theorem statement combining all above conditions
theorem equilateral_triangle_properties (n : ℕ) (a b c r S : ℝ) :
  equilateral_triangle n a b c →
  nodes_distance n a b c r →
  total_sum n a b c S :=
begin
  intros h1 h2 h3,
  exact ((h1, h2), h3),
end

end equilateral_triangle_properties_l396_396081


namespace irreducible_fraction_l396_396109

theorem irreducible_fraction (n : ℕ) : 
  irreducible (2 * n^2 + 11 * n - 18) (n + 7) ↔ 
    (n % 3 = 0) ∨ (n % 3 = 1) :=
sorry

end irreducible_fraction_l396_396109


namespace infinite_rational_points_on_circle_l396_396689

noncomputable def exists_infinitely_many_rational_points_on_circle : Prop :=
  ∃ f : ℚ → ℚ × ℚ, (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                   (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m)

theorem infinite_rational_points_on_circle :
  ∃ (f : ℚ → ℚ × ℚ), (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                     (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m) := sorry

end infinite_rational_points_on_circle_l396_396689


namespace difference_in_percentage_l396_396417

noncomputable def principal : ℝ := 600
noncomputable def timePeriod : ℝ := 10
noncomputable def interestDifference : ℝ := 300

theorem difference_in_percentage (R D : ℝ) (h : 60 * (R + D) - 60 * R = 300) : D = 5 := 
by
  -- Proof is not provided, as instructed
  sorry

end difference_in_percentage_l396_396417


namespace ratio_problem_l396_396938

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396938


namespace geometric_property_l396_396207

noncomputable theory

variables {A B C O P Q : Type*}
variables [inner_product_space ℝ (triangle ABC)]
variables (h₁ : is_acute (triangle ABC))
variables (h₂ : length (segment AB) > length (segment AC))
variables (h₃ : tangent_to (circumcircle (triangle ABC)) A P)
variables (h₄ : Q ∈ segment (segment OP))
variables (h₅ : angle O B Q = angle O P B)

theorem geometric_property :
  length (segment AC) * real.sqrt (length (segment QB)) = 
  length (segment AB) * real.sqrt (length (segment QC)) :=
sorry

end geometric_property_l396_396207


namespace part1_part2_part3_l396_396544

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin (π / 6 * x) * cos (π / 6 * x)

noncomputable def g (t : ℝ) : ℝ :=
  (f (t + 1) - f t) / (t + 1 - t)

theorem part1 : g 0 = (Real.sqrt 3) / 2 := by
  sorry

noncomputable def derived_g (t : ℝ) : ℝ :=
  -sin (π / 3 * t - π / 3)

theorem part2 : ∀ t, g t = derived_g t := by
  sorry

theorem part3 : ∀ t ∈ set.Icc (-3 / 2) (3 / 2), 
  derived_g t ∈ set.Icc (-1) (1 / 2) := by
  sorry

end part1_part2_part3_l396_396544


namespace problem_solution_l396_396122

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end problem_solution_l396_396122


namespace cookie_ratio_l396_396824

theorem cookie_ratio (K : ℕ) (h1 : K / 2 + K + 24 = 33) : 24 / K = 4 :=
by {
  sorry
}

end cookie_ratio_l396_396824


namespace circles_intersection_correct_statements_l396_396327

-- Definitions of the circles
def circle_O1 (x y : ℝ) := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y = 0

-- Line of the chord AB derived from the given circles
def chord_line (x y : ℝ) := x - y = 0

-- Perpendicular bisector of segment AB
def perpendicular_bisector (x y : ℝ) := x + y - 1 = 0

-- Length of common chord AB (incorrect statement C)
def length_common_chord_AB : ℝ := real.sqrt 2

-- Maximum distance from a point P on circle O1 to the chord line
def max_distance_P_to_chord : ℝ := (real.sqrt 2) / 2 + 1

-- The proof problem statement
theorem circles_intersection_correct_statements :
  (∀ x y : ℝ, circle_O1 x y ∧ circle_O2 x y → chord_line x y) ∧
  (∀ x y : ℝ, ∃ xmid ymid : ℝ, xmid = 0 ∧ ymid = 1 ∧ perpendicular_bisector x y) ∧
  -- statement C is false, hence we don't require the length proof here
  (∀ x y : ℝ, ∀ P : ℝ, circle_O1 x y → max_distance_P_to_chord = (real.sqrt 2) / 2 + 1) 
 := 
sorry

end circles_intersection_correct_statements_l396_396327


namespace avg_difference_students_teachers_l396_396413

/-- Let t be the average number of students per teacher
    and s be the average number of students per student
    in a school with 120 students and 5 teachers where
    the class enrollments are 60, 30, 20, 5, and 5. Prove
    that the difference t - s is equal to -17.25. --/
theorem avg_difference_students_teachers (t s : ℕ → ℚ) :
    ∀ (students teachers : ℕ) (classes : List ℕ),
    students = 120 →
    teachers = 5 →
    classes = [60, 30, 20, 5, 5] →
    t = (60 + 30 + 20 + 5 + 5) / 5 →
    s = 60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 5 * (5 / 120) + 5 * (5 / 120) →
    t - s = -17.25 :=
by
  intros students teachers classes h_students h_teachers h_classes h_t h_s
  sorry

end avg_difference_students_teachers_l396_396413


namespace sum_lowest_highest_l396_396856

variable (scores : List ℕ)
variable (h_len : scores.length = 4)
variable (h_mean : scores.sum = 320)
variable (h_median : (scores[1] + scores[2]) / 2 = 83)
variable (h_mode : scores.count 85 = 2)
variable (h_distinct : ∀ i j, i ≠ j → scores[i] ≠ scores[j] ∨ i = 2 ∨ i = 3 ∨ j = 2 ∨ j = 3)

theorem sum_lowest_highest : scores.min + scores.max = 152 := by
  sorry

end sum_lowest_highest_l396_396856


namespace card_selection_probability_l396_396781

theorem card_selection_probability :
  let n := 6
  let k := 3
  let total_ways := Nat.choose n k
  let favorable_ways := Nat.choose 4 2
  total_ways > 0 → Probability (favorable_ways / total_ways) = 3/10 := 
by
  let n := 6
  let k := 3
  have total_ways : ℕ := Nat.choose n k
  have favorable_ways : ℕ := Nat.choose 4 2
  have prob := favorable_ways / total_ways
  guard prob = 3/10
  sorry

end card_selection_probability_l396_396781


namespace ratio_expression_value_l396_396923

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396923


namespace albert_mary_age_ratio_l396_396009

theorem albert_mary_age_ratio
  (A M B : ℕ)
  (h1 : A = 4 * B)
  (h2 : M = A - 14)
  (h3 : B = 7)
  :
  A / M = 2 := 
by sorry

end albert_mary_age_ratio_l396_396009


namespace value_of_k_l396_396534

noncomputable def find_k (k : ℝ) : Prop :=
  let eq : Polynomial := C 8 * X^2 + C (6 * k) * X + C (2 * k + 1)
  let roots : Finset ℝ := {r | polynomial.eval r eq = 0}.to_finset
  ∃ θ : ℝ, roots = finset.insert ₀.{term_of ℝ} (θ.cos) θ.sin ∅ ∨
    roots = finset.insert ₀.{term_of ℝ} (θ.sin) θ.cos ∅ ∧
    ∀ (k_val : ℝ), k = k_val → k_val = -10 / 9

theorem value_of_k : ∃ k : ℝ, find_k k :=
sorry

end value_of_k_l396_396534


namespace problem_solution_l396_396121

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end problem_solution_l396_396121


namespace option_c_same_function_l396_396584

theorem option_c_same_function :
  ∀ (x : ℝ), x ≠ 0 → (1 + (1 / x) = u ↔ u = 1 + (1 / (1 + 1 / x))) :=
by sorry

end option_c_same_function_l396_396584


namespace cesaro_sum_51_l396_396595

-- Given: The Cesaro sum of the sequence (a_1, a_2, ..., a_{50}) is 300
-- Definition of summation for the sequence (a_1, a_2, ..., a_{50})
def cesaro_sum_50 (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∑ k in range n, ∑ i in range (k + 1), a i) / n = 300

-- Let's define the new sequence (2, a_1, a_2, ..., a_{50})
def new_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 2 else a (n-1)

-- Definition of the sum of the partial sums for the new sequence
def partial_sum_51 (a : ℕ → ℕ) : ℕ :=
  ∑ k in range 51, ∑ i in range (k + 1), new_sequence a i

-- Theorem to prove: the Cesaro sum of the new sequence is 296
theorem cesaro_sum_51 (a : ℕ → ℕ) (h : cesaro_sum_50 a 50) : partial_sum_51 a / 51 = 296 :=
by
  sorry

end cesaro_sum_51_l396_396595


namespace largest_of_set_l396_396433

theorem largest_of_set : 
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  c = 2 ∧ (d < b ∧ b < a ∧ a < c) := by
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  sorry

end largest_of_set_l396_396433


namespace polynomial_coeff_a2_l396_396552

theorem polynomial_coeff_a2 (a : ℕ) (x : ℝ) :
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ,
  (x^2 + x^10) = a₀ + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 +
  a₄ * (x+1)^4 + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 +
  a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) →
  a₂ = 46 :=
begin
  sorry
end

end polynomial_coeff_a2_l396_396552


namespace friends_seating_position_l396_396690

theorem friends_seating_position (p_A p_B p_C p_D p_E p_F : ℕ)
  (h_initial : {p_A, p_B, p_C, p_D, p_E, p_F} = {1, 2, 3, 4, 5, 6})
  (h_bea : p_B' = p_B + 1)
  (h_ceci : p_C' = p_C)
  (h_dee_fay : p_D' = p_F ∧ p_F' = p_D)
  (h_edie_end : p_E' = p_E - 2 ∧ (p_E' = 1 ∨ p_E' = 6))
  (h_ada_return : p_A' ∈ {p_F - 1, p_F + 1}) :
  p_A = 4 :=
by sorry

end friends_seating_position_l396_396690


namespace count_subsets_containing_7_l396_396568

def original_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def subsets_containing_7 (s : Set ℕ) : Prop :=
  7 ∈ s

theorem count_subsets_containing_7 :
  {s : Set ℕ | s ⊆ original_set ∧ subsets_containing_7 s}.finite.card = 64 :=
sorry

end count_subsets_containing_7_l396_396568


namespace perpendicular_c_a_l396_396149

variables (a b c : ℝ^3)
variables (h1 : |a| = 1) (h2 : |b| = 2) (h3 : c = a + b) (h4 : a • b = -1)

theorem perpendicular_c_a : c • a = 0 :=
by {
  sorry
}

end perpendicular_c_a_l396_396149


namespace tangent_line_eq_dot_product_const_l396_396155

/-- A circle centered at (3, 4) with radius 2. -/
def circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

/-- A line passing through point A(1, 0). -/
def line_passing_through_A (l : ℝ → ℝ) : Prop :=
  l 1 = 0

/-- The equation of line l_2. -/
def line_l2 (x y : ℝ) : Prop :=
  x + 2 * y + 2 = 0

/-- To be proven: Given circle C and conditions, 
    the line l_1 tangent to C must have specific equations. -/
theorem tangent_line_eq (l : ℝ → ℝ) :
  (∀ x y, circle x y → line_passing_through_A l) →
  (∀ x y, (l x = y) ∧ ((x - 3)^2 + (y - 4)^2 = 4) → (l x - y = k) →
  (k = 0 → x = 1) ∨ (k ≠ 0 → 3*x - 4*y - 3 = 0)) := sorry

/-- To be proven: Given lines l1 and l2 intersecting with the circle,
    the dot product of vectors AM and AN is constant. -/
theorem dot_product_const (l : ℝ → ℝ) (k : ℝ) :
  (∀ x y, circle x y ∧ line_passing_through_A l) →
  (∃ x y, (l x = y) ∧ (line_l2 x y) ∧ ((x-3)^2 + (y-4)^2 = 4) →
  let M := ((k^2 + 4*k + 3)/(1 + k^2), (4*k^2 + 2*k)/(1 + k^2)),
      N := ((2*k-2)/(2*k+1), -3*k/(2*k+1)) in
  (6 : ℝ)) := sorry

end tangent_line_eq_dot_product_const_l396_396155


namespace bc_fraction_ad_l396_396674

theorem bc_fraction_ad
  (B C E A D : Type)
  (on_AD : ∀ P : Type, P = B ∨ P = C ∨ P = E)
  (AB BD AC CD DE EA: ℝ)
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : DE = 2 * EA)

  : ∃ BC AD: ℝ, BC = 1 / 12 * AD := 
sorry -- Proof is omitted

end bc_fraction_ad_l396_396674


namespace perpendicular_vectors_lambda_l396_396173

theorem perpendicular_vectors_lambda (λ : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (λ, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → λ = -2 := by
  intros
  sorry

end perpendicular_vectors_lambda_l396_396173


namespace min_value_l396_396147

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) := 
sorry

end min_value_l396_396147


namespace math_problem_l396_396309

variables (x y : ℝ)

noncomputable def question_value (x y : ℝ) : ℝ := (2 * x - 5 * y) / (5 * x + 2 * y)

theorem math_problem 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (cond : (5 * x - 2 * y) / (2 * x + 3 * y) = 1) : 
  question_value x y = -5 / 31 :=
sorry

end math_problem_l396_396309


namespace probability_neither_square_nor_cube_nor_fourth_l396_396334

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_fourth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n
def is_perfect_eighth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k * k * k = n
def is_perfect_twelfth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k * k * k * k * k * k * k = n

theorem probability_neither_square_nor_cube_nor_fourth :
  let range := (1 : ℕ) to 200;
  let count_perfect_squares := 14;
  let count_perfect_cubes := 5;
  let count_perfect_fourth_powers := 3;
  let overlap_sixth := 2;
  let overlap_eighth := 1;
  let overlap_twelfth := 1;
  let unique_special_numbers := count_perfect_squares + count_perfect_cubes + count_perfect_fourth_powers - overlap_sixth - overlap_eighth - overlap_twelfth;
  let not_special_count := (range.end - range.start + 1) - unique_special_numbers;
  (not_special_count : ℚ) / (200 : ℚ) = 91 / 100 :=
by
  sorry

end probability_neither_square_nor_cube_nor_fourth_l396_396334


namespace prove_fx_le_x2_l396_396260

def condition1 (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 → f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2)

def condition2 (f : ℝ → ℝ) (M : ℝ) : Prop :=
  M > 0 ∧ ∀ x, 0 ≤ x ∧ x ≤ 1 → |f(x)| ≤ M

theorem prove_fx_le_x2 (f : ℝ → ℝ) (M : ℝ) :
  (∀ x y, condition1 f x y) →
  condition2 f M →
  ∀ x, x ≥ 0 → f(x) ≤ x^2 :=
by { intros h1 h2 x hx, sorry }

end prove_fx_le_x2_l396_396260


namespace smallest_percent_increase_between_7_and_8_l396_396790

def values : List ℕ := [100, 300, 600, 800, 1500, 3000, 4500, 7000, 10000, 15000, 30000, 45000, 75000, 150000, 300000]

def percentIncrease (v1 v2 : ℕ) : Float := 
  ((v2 - v1) / v1.toFloat) * 100

theorem smallest_percent_increase_between_7_and_8 :
  ∀ (i j : ℕ), 
    (i, j) ∈ [(2, 3), (4, 5), (7, 8), (12, 13), (14, 15)] → 
    percentIncrease (values.get! (i - 1)) (values.get! (j - 1)) ≥ percentIncrease (values.get! 6) (values.get! 7) :=
by
  sorry

end smallest_percent_increase_between_7_and_8_l396_396790


namespace min_sum_weights_l396_396599

theorem min_sum_weights (S : ℕ) (h1 : S > 280) (h2 : S % 70 = 30) : S = 310 :=
sorry

end min_sum_weights_l396_396599


namespace length_segment_AB_l396_396551

-- Definitions
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus (p : ℝ) : (ℝ × ℝ) := (p/2, 0)
def line_l (p x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - p/2)
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2
def distance_origin (s : ℝ × ℝ) : ℝ := Real.sqrt (s.fst^2 + s.snd^2)
def point_M (p : ℝ) : (ℝ × ℝ) := (-p/2, -(Real.sqrt 3 / 3) * p)
def OM_dist (p : ℝ) : Prop := distance_origin (point_M p) = (2 * Real.sqrt 21) / 3

-- Theorem and proof statement
theorem length_segment_AB (p : ℝ) : 
  parabola p ∧ focus p ∧ directrix p ∧ OM_dist p →
  p = 4 → (∑ x in (roots (λ x, x^2 - 28*x + 4)), x) + p = 32 :=
by
  intros hp
  sorry

end length_segment_AB_l396_396551


namespace diagonals_of_convex_polygon_l396_396820

theorem diagonals_of_convex_polygon (n : ℕ) (hn : n = 25) : 
  let total_diagonals := (n * (n - 3)) / 2
  in total_diagonals = 275 :=
by 
  sorry

end diagonals_of_convex_polygon_l396_396820


namespace triangle_ABM_perimeter_2_l396_396673

-- Definitions of the given conditions
structure Triangle :=
(A B C : Point)

def perimeter (A B C : Point) : ℝ :=
  dist A B + dist B C + dist C A

def is_on_ray (P Q R : Point) : Prop :=
  ∃ k : ℝ, k > 0 ∧ Q = P + k • R

def is_on_segment (P Q : Point) (M : Point) : Prop :=
  dist P M + dist M Q = dist P Q

-- The problem: Prove that the perimeter of one of the triangles ABM or ACM is 2
theorem triangle_ABM_perimeter_2 (A B C X Y M : Point)
    (h_triangle : Triangle)
    (h_perimeter : perimeter A B C = 4)
    (h_AX : dist A X = 1)
    (h_AY : dist A Y = 1)
    (h_X_on_ray_AB : is_on_ray A B X)
    (h_Y_on_ray_AC : is_on_ray A C Y)
    (h_M_intersection : is_on_segment B C M ∧ is_on_segment X Y M) :
  perimeter A B M = 2 :=
sorry

end triangle_ABM_perimeter_2_l396_396673


namespace part_a_part_b_l396_396382

open EuclideanGeometry

noncomputable def rotational_homothety_center_p1 (A B A1 B1 P : Point) (h : ¬ (A = B ∨ A = A1 ∨ A = B1 ∨ A = P ∨ B = A1 ∨ B = B1 ∨ B = P ∨ A1 = B1 ∨ A1 = P ∨ B1 = P))
  (intersect : line_thru A B ∩ line_thru A1 B1 = {P}) : Point :=
Classical.some sorry

theorem part_a (A B A1 B1 P : Point) (h : ¬ (A = B ∨ A = A1 ∨ A = B1 ∨ A = P ∨ B = A1 ∨ B = B1 ∨ B = P ∨ A1 = B1 ∨ A1 = P ∨ B1 = P))
  (intersect : line_thru A B ∩ line_thru A1 B1 = {P}) (O : Point) :
  (O ∈ circumcircle A P A1) ∧ (O ∈ circumcircle B P B1) → O = rotational_homothety_center_p1 A B A1 B1 P h intersect :=
sorry

noncomputable def rotational_homothety_center_p2 (A B C : Point)
  : Point :=
Classical.some sorry

theorem part_b (A B C : Point) (O : Point) :
  (O ∈ circle_tangent_at_point A B C) ∧ (O ∈ circle_tangent_at_point C B A) → O = rotational_homothety_center_p2 A B C :=
sorry

end part_a_part_b_l396_396382


namespace average_A_B_l396_396706

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end average_A_B_l396_396706


namespace sqrt_inequality_l396_396256

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := 
by
  sorry

end sqrt_inequality_l396_396256


namespace matrix_product_l396_396453

-- Define matrix A
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 3, 2], ![1, -3, 4]]

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, 0], ![2, 0, 4], ![3, 0, 1]]

-- Define the expected result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![9, 6, -1], ![12, 0, 14], ![7, 3, -8]]

-- The statement to prove
theorem matrix_product : A * B = C :=
by
  sorry

end matrix_product_l396_396453


namespace sin_double_angle_l396_396538

-- Defining the conditions
variable (α : ℝ)
variable (m : ℝ)

-- Condition: The terminal side intersects the unit circle at point A(m, sqrt(3) * m)
def unit_circle_point (m : ℝ) : Prop :=
  (m ^ 2 + (√3 * m) ^ 2 = 1)

-- Main Statement: Given conditions, prove sin 2α = sqrt(3) / 2
theorem sin_double_angle
  (h1 : unit_circle_point m) 
  (m_nonzero : m ≠ 0) :
  Real.sin (2 * α) = √3 / 2 :=
sorry

end sin_double_angle_l396_396538


namespace problem_statement_l396_396329

-- Define the two circles in terms of their equations
def O₁ : Set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 2 * p.1 = 0 }
def O₂ : Set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 + 2 * p.1 - 4 * p.2 = 0 }

-- Define the intersection points A and B
def intersect_points (O₁ O₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := O₁ ∩ O₂

-- Definitions for points A, B, and conditions on these points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ
axiom A_in_intersect : A ∈ intersect_points O₁ O₂
axiom B_in_intersect : B ∈ intersect_points O₁ O₂

-- Prove the statements given the conditions
theorem problem_statement :
  (∀ P : ℝ × ℝ, P ∈ O₁ → (∀ x y : ℝ, x = y → ∃ l : Polynomial ℝ, l.eval (x, y) = l.eval P)) ∧ 
  (A.x + B.x) / 2 = 0 ∧
  (A.y + B.y) / 2 = 1 ∧
  ∃ (AB_length : ℝ), AB_length = sqrt 2 / 2 ∧
  ∃ (max_dist : ℝ), max_dist = (sqrt 2 / 2) + 1 := 
sorry

end problem_statement_l396_396329


namespace eggs_remainder_and_full_cartons_l396_396007

def abigail_eggs := 48
def beatrice_eggs := 63
def carson_eggs := 27
def carton_size := 15

theorem eggs_remainder_and_full_cartons :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  ∃ (full_cartons left_over : ℕ),
    total_eggs = full_cartons * carton_size + left_over ∧
    left_over = 3 ∧
    full_cartons = 9 :=
by
  sorry

end eggs_remainder_and_full_cartons_l396_396007


namespace function_domain_l396_396619

theorem function_domain (x : ℝ) : x ≠ 3 → ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end function_domain_l396_396619


namespace probability_correct_l396_396778

noncomputable def probability_two_blue_one_red_one_green 
  (total_red : ℕ) (total_blue : ℕ) (total_green : ℕ) (total_marbles : ℕ) : ℚ :=
  if total_red = 15 ∧ total_blue = 9 ∧ total_green = 6 ∧ total_marbles = 30 then
    (2 * (15 / 30 * 9 / 29 * 8 / 28 * 6 / 27) + 
     2 * (15 / 30 * 9 / 29 * 6 / 28 * 8 / 27) + 
     2 * (15 / 30 * 6 / 29 * 9 / 28 * 8 / 27)).reduce
  else 0

theorem probability_correct : 
  probability_two_blue_one_red_one_green 15 9 6 30 = 5 / 812 :=
  by exact sorry

end probability_correct_l396_396778


namespace savings_by_december_l396_396630

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l396_396630


namespace ratio_problem_l396_396942

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396942


namespace original_recipe_calls_for_4_tablespoons_l396_396447

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end original_recipe_calls_for_4_tablespoons_l396_396447


namespace can_realize_991_threes_and_6_twos_cannot_realize_8_eights_985_sixes_and_8_threes_l396_396606

-- Definition of the convex polygon M with 1994 sides
def M := Polygon 1994

-- Condition that 997 diagonals are drawn such that each vertex connects to exactly one diagonal
def has_997_diagonals_drawn (p: Polygon 1994) :=
∀ v ∈ vertices(p), ∃! d ∈ diagonals(p), connects d v

-- Definition of the length of a diagonal
def diagonal_length (d: Diagonal) (p: Polygon 1994) : ℕ :=
let parts := divide_perimeter d p
in min parts.1.length parts.2.length

-- First problem: Prove the sequence with 991 threes and 6 twos can be realized
theorem can_realize_991_threes_and_6_twos
  (h: has_997_diagonals_drawn M) :
  ∃ (seq : List ℕ), seq = replicate 991 3 ++ replicate 6 2 :=
by sorry

-- Second problem: Prove the sequence with 4 eights, 985 sixes, and 8 threes cannot be realized
theorem cannot_realize_8_eights_985_sixes_and_8_threes
  (h: has_997_diagonals_drawn M) :
  ¬ (∃ (seq : List ℕ), seq = replicate 4 8 ++ replicate 985 6 ++ replicate 8 3) :=
by sorry

end can_realize_991_threes_and_6_twos_cannot_realize_8_eights_985_sixes_and_8_threes_l396_396606


namespace distinct_pairs_count_l396_396318

theorem distinct_pairs_count :
  ∃ (S : Finset (ℕ × ℕ)), 
  S.card > 3 ∧ 
  ∀ (pair : ℕ × ℕ), pair ∈ S → 
  let ⟨b1, b2⟩ := pair in 
  (10 * b1 + 10 * b2 = 100) := 
by
  sorry

end distinct_pairs_count_l396_396318


namespace value_of_M_N_P_Q_l396_396209

-- Define the conditions
variables (vertices : Finset ℕ)
variables (0 2 3 : ℕ)
variables (M N P Q R : ℕ)
variables (prime : ℕ → Prop)
variables (is_prime_edge_sum : ℕ → ℕ → Prop)

-- Assume all vertices are distinct elements from the set {0, 1, 2, 3, 4, 5, 6, 7}
axiom vertices_condition : vertices = {0, 1, 2, 3, 4, 5, 6, 7}

-- Known vertices 
axiom vertex0 : 0 ∈ vertices
axiom vertex2 : 2 ∈ vertices
axiom vertex3 : 3 ∈ vertices

-- Each number is labeled on one vertex and each pair sum of connected vertices is prime
axiom prime_sum_condition : 
  ∀ (a b : ℕ), a ∈ vertices → b ∈ vertices → is_prime_edge_sum a b → prime (a + b)

-- Problem proof statement
theorem value_of_M_N_P_Q : M + N + P + Q = 18 :=
sorry

end value_of_M_N_P_Q_l396_396209


namespace increase_in_circle_area_l396_396598

theorem increase_in_circle_area (r : ℝ) (original_radius : r = 8) (new_radius : r + 2 = 10) : 
  (π * (10 ^ 2) - π * (8 ^ 2)) = 36 * π :=
by
  have original_circumference : 2 * π * r = 16 * π := by
    rw [original_radius]
    norm_num
  calc
    π * (10 ^ 2) - π * (8 ^ 2)
        = π * 100 - π * 64 : by norm_num
    ... = π * (100 - 64) : by ring
    ... = π * 36 : by norm_num
    ... = 36 * π : by ring

end increase_in_circle_area_l396_396598


namespace domain_of_sqrt_quadratic_l396_396847

def quadratic_expr (x : ℝ) : ℝ := -8 * x^2 + 10 * x + 3

theorem domain_of_sqrt_quadratic :
  {x : ℝ | √quadratic_expr x ∈ ℝ } = set.Icc (-1/2) (3/4) :=
by
  -- Proof goes here
  sorry

end domain_of_sqrt_quadratic_l396_396847


namespace arithmetic_sequence_ratio_l396_396833

theorem arithmetic_sequence_ratio (a x : ℝ) (h_seq : ∀ n, n ∈ {0, 1, 3, 6}) :
  (a + n * x) / (a + 6 * x) = 1 / 4 := by
sorry

end arithmetic_sequence_ratio_l396_396833


namespace complex_number_solution_l396_396156

open Complex

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : z = 1 + I :=
sorry

end complex_number_solution_l396_396156


namespace min_rolls_to_ensure_duplicate_sum_l396_396752

theorem min_rolls_to_ensure_duplicate_sum :
  (∀ (rolls : List (ℕ × ℕ × ℕ)), (∀ roll ∈ rolls, 1 ≤ roll.1 ∧ roll.1 ≤ 6 ∧ 1 ≤ roll.2 ∧ roll.2 ≤ 6 ∧ 1 ≤ roll.3 ∧ roll.3 ≤ 6) →
    (3 ≤ (rolls.map (λ roll, roll.1 + roll.2 + roll.3)).length ∧ 
    (rolls.map (λ roll, roll.1 + roll.2 + roll.3)).length ≤ 18) →
    (∃ (d : ℕ), d ∈ ((rolls.map (λ roll, roll.1 + roll.2 + roll.3)) : Finset ℕ) →
    nat.card (Finset.of_list (rolls.map (λ roll, roll.1 + roll.2 + roll.3))) > 16) →
    rolls.length ≥ 17) := sorry

end min_rolls_to_ensure_duplicate_sum_l396_396752


namespace inverse_exponential_l396_396532

theorem inverse_exponential :
  (∃ a : ℝ, ∀ x : ℝ, (a > 0) ∧ (a ≠ 1) → (2010 = a^1) → (∃ f_inv : ℝ → ℝ, ∀ y : ℝ, f_inv (a^y) = y)) :=
begin
  let a := 2010,
  sorry
end

end inverse_exponential_l396_396532


namespace binom_7_2_eq_21_l396_396053

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396053


namespace red_segments_equal_green_segments_l396_396745

theorem red_segments_equal_green_segments
  (m n : ℕ)
  (tokens : Fin (2 * m) → Fin (2 * n) → Prop)
  (is_red : Prop)
  (is_green : Prop)
  (equal_rows : ∀ i : Fin (2 * m), ∃ r g : ℕ, (r = n) ∧ (g = n) ∧ ∀ j : Fin (2 * n), (tokens i j = is_red ∨ tokens i j = is_green))
  (equal_columns : ∀ j : Fin (2 * n), ∃ r g : ℕ, (r = m) ∧ (g = m) ∧ ∀ i : Fin (2 * m), (tokens i j = is_red ∨ tokens i j = is_green))
  (red_segment : (Fin (2 * m) × Fin (2 * n)) → (Fin (2 * m) × Fin (2 * n)) → Prop)
  (green_segment : (Fin (2 * m) × Fin (2 * n)) → (Fin (2 * m) × Fin (2 * n)) → Prop) :
  (∀ i : Fin (2 * m), ∀ j : Fin (2 * n - 1), red_segment (i, j) (i, ⟨j + 1, sorry⟩) ↔ tokens i j = is_red ∧ tokens i ⟨j + 1, sorry⟩ = is_red) →
  (∀ i : Fin (2 * m - 1), ∀ j : Fin (2 * n), red_segment (i, j) (⟨i + 1, sorry⟩, j) ↔ tokens i j = is_red ∧ tokens ⟨i + 1, sorry⟩ j = is_red) →
  (∀ i : Fin (2 * m), ∀ j : Fin (2 * n - 1), green_segment (i, j) (i, ⟨j + 1, sorry⟩) ↔ tokens i j = is_green ∧ tokens i ⟨j + 1, sorry⟩ = is_green) →
  (∀ i : Fin (2 * m - 1), ∀ j : Fin (2 * n), green_segment (i, j) (⟨i + 1, sorry⟩, j) ↔ tokens i j = is_green ∧ tokens ⟨i + 1, sorry⟩ j = is_green) →
  (∃ red_count green_count : ℕ, red_count = green_count) :=
sorry

end red_segments_equal_green_segments_l396_396745


namespace h_inch_approx_l396_396319

noncomputable def h_cm : ℝ := 14.5 - 2 * 1.7
noncomputable def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
noncomputable def h_inch : ℝ := cm_to_inch h_cm

theorem h_inch_approx : abs (h_inch - 4.37) < 1e-2 :=
by
  -- The proof is omitted
  sorry

end h_inch_approx_l396_396319


namespace necessary_condition_x_squared_minus_x_lt_zero_l396_396862

theorem necessary_condition_x_squared_minus_x_lt_zero (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧ ((-1 < x ∧ x < 1) → ¬ (x^2 - x < 0)) :=
by
  sorry

end necessary_condition_x_squared_minus_x_lt_zero_l396_396862


namespace lemons_for_lemonade_l396_396589

theorem lemons_for_lemonade (L G L₁ G₁ : ℕ) (h_ratio : L/G = L₁/G₁) (h_LG : L = 40) (h_G : G = 50) (h_G₁ : G₁ = 15) : L₁ = 12 :=
by
  have h1 : 40 / 50 = L₁ / 15 := by rw [h_ratio, h_LG, h_G, h_G₁]
  have h2 : 4 / 5 = L₁ / 15 := by norm_num at h1
  have h3 : 4 * 15 = 5 * L₁ := by cross_mul h2
  have h4 : 60 = 5 * L₁ := by norm_num at h3
  have h5 : L₁ = 12 := by linarith
  exact h5

end lemons_for_lemonade_l396_396589


namespace semicircle_perimeter_l396_396767

def π : ℝ := Real.pi

theorem semicircle_perimeter (r : ℝ) (h : r = 4.8) : 
  (π * r + 2 * r ≈ 24.672) :=
by 
  -- assuming an approximate value for π
  have approx_π : π ≈ 3.14, from sorry,
  have P : π * 4.8 + 2 * 4.8 ≈ 24.672, from sorry,
  -- finishing proof
  sorry

end semicircle_perimeter_l396_396767


namespace probability_both_hit_l396_396224

-- Define the probabilities of hitting the target for shooters A and B.
def prob_A_hits : ℝ := 0.7
def prob_B_hits : ℝ := 0.8

-- Define the independence condition (not needed as a direct definition but implicitly acknowledges independence).
axiom A_and_B_independent : true

-- The statement we want to prove: the probability that both shooters hit the target.
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.56 :=
by
  -- Placeholder for proof
  sorry

end probability_both_hit_l396_396224


namespace solve_system_l396_396124

theorem solve_system : ∀ (x y : ℝ), x + 2 * y = 1 ∧ 2 * x + y = 2 → x + y = 1 :=
by
  intros x y h
  cases h with h₁ h₂
  sorry

end solve_system_l396_396124


namespace sum_g35_l396_396659

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (y : ℝ) : ℝ := (real.sqrt y) ^ 2 - real.sqrt y + 2

theorem sum_g35 (x : ℝ) (hx : f x = 35) : (g (f (real.sqrt 9.5)) + g (f (-real.sqrt 9.5))) = 23 :=
by
  have h_f_x : f (real.sqrt 9.5) = 35 := sorry
  have h_f_minus_x : f (-real.sqrt 9.5) = 35 := sorry
  have h_g_f_x : g (f (real.sqrt 9.5)) = 9.5 - real.sqrt 9.5 + 2 := sorry
  have h_g_f_minus_x : g (f (-real.sqrt 9.5)) = 9.5 + real.sqrt 9.5 + 2 := sorry
  sorry

end sum_g35_l396_396659


namespace contrapositive_l396_396707

theorem contrapositive (x : ℝ) (h : x^2 ≥ 1) : x ≥ 0 ∨ x ≤ -1 :=
sorry

end contrapositive_l396_396707


namespace animals_arrangement_equivalence_l396_396315
open Nat

/-- Define the factorial function. -/
def factorial : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * factorial n

/-- Define the problem conditions. -/
def pigs := 5
def rabbits := 3
def dogs := 2
def chickens := 6
def cages := 16

/-- Define the total number of animals. -/
def total_animals := pigs + rabbits + dogs + chickens

/-- Define the ways to arrange each group and the calculations. -/
def group_arrangements := factorial 4
def pigs_arrangements := factorial pigs
def rabbits_arrangements := factorial rabbits
def dogs_arrangements := factorial dogs
def chickens_arrangements := factorial chickens
def total_arrangements := group_arrangements * pigs_arrangements * rabbits_arrangements * dogs_arrangements * chickens_arrangements

/-- The theorem statement. -/
theorem animals_arrangement_equivalence : total_arrangements = 12441600 := by
  sorry

end animals_arrangement_equivalence_l396_396315


namespace problem1_problem2_l396_396160

noncomputable def function1 (x : ℝ) (a : ℝ) (b : ℝ) := (1 / 3) * x^3 + a * x^2 + b * x

theorem problem1 (a b : ℝ) : 
  (∀ x, function1 x 1 b ≥ 0 ↔ b ≥ 1) ∧ 
  (b < 1 → (∀ x, (function1 x 1 b).derivative = x^2 + 2 * x + b) → 
    ∃ (n : ℝ), (function1 n 1 b ≥ 0) ∧ 
                ∀ x, (x < n - (1 : ℝ) - (real.sqrt (1 - b)) ∨ 
                       x > n - (1 : ℝ) + (real.sqrt (1 - b)) 
                → function1 x 1 b > 0)) :=
sorry

noncomputable def function2 (x : ℝ) (a : ℝ) := (1 / 3) * x^3 + a * x^2 + (-a) * x

theorem problem2 (a : ℝ) :
  function2 1 a = 1 / 3 → 
  (∀ x, (x > 0) → (x < 1 / 2) → 
    (∃ y, (x^2 / (1 - 2 * x) = y) ∧ (function2 y a = 0)) → 
     false) → (a ∈ set.Iic 0 ) :=
sorry

end problem1_problem2_l396_396160


namespace T_is_a_line_l396_396179

open Complex

noncomputable def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ (5 + 3 * I) * z ∈ ℝ}

theorem T_is_a_line : ∃ m b : ℝ, T = {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ x = m * y + b} :=
by
  sorry

end T_is_a_line_l396_396179


namespace log_sqrt8_512sqrt8_l396_396486

theorem log_sqrt8_512sqrt8 : log (sqrt 8) (512 * sqrt 8) = 7 := sorry

end log_sqrt8_512sqrt8_l396_396486


namespace total_hours_worked_l396_396798

-- Definitions from conditions
def hours_in_a_day := 24
def hours_on_software := 24
def hours_on_helping_users := 17
def maintenance_percentage := 0.35
def rnd_percentage := 0.27
def marketing_percentage := 0.15
def multitasking_employees := 3
def additional_tasks_hours := 12
def total_multitask_hours := hours_in_a_day

-- Theorem to prove
theorem total_hours_worked : 
  hours_on_software + hours_on_helping_users + (maintenance_percentage + rnd_percentage + marketing_percentage) * hours_in_a_day + additional_tasks_hours - 
  (total_multitask_hours) = 36 :=
by
  -- placeholders for steps in the solution
  have h1 : maintenance_percentage + rnd_percentage + marketing_percentage = 0.77 := by linarith
  have h2 : (maintenance_percentage + rnd_percentage + marketing_percentage) * hours_in_a_day = 18.48 := by norm_num
  have h3 : hours_on_software + hours_on_helping_users + 18.48 + additional_tasks_hours = 71.48 := by norm_num
  exact sorry

end total_hours_worked_l396_396798


namespace arrange_in_ascending_order_l396_396648

def a : ℝ := 0.3^2
def b : ℝ := 2^(0.5)
def c : ℝ := Real.log 4 / Real.log 2

theorem arrange_in_ascending_order :
  a < b ∧ b < c := by
  sorry

end arrange_in_ascending_order_l396_396648


namespace ratio_of_sums_l396_396247

-- Definitions based on the conditions in the problem
variables {a_1 a_2 a_3 q : ℝ}

def S (n : ℕ) := a_1 * (1 - q^n) / (1 - q)

-- Conditions
axiom identical_roots_condition : a_3^2 = 4 * a_1 * a_2
axiom common_ratio_condition : q^3 = 4

-- The main theorem to prove
theorem ratio_of_sums : (S 9) / (S 3) = 21 :=
by
  sorry

end ratio_of_sums_l396_396247


namespace double_counted_toddlers_l396_396019

def number_of_toddlers := 21
def missed_toddlers := 3
def billed_count := 26

theorem double_counted_toddlers : 
  ∃ (D : ℕ), (number_of_toddlers + D - missed_toddlers = billed_count) ∧ D = 8 :=
by
  sorry

end double_counted_toddlers_l396_396019


namespace equal_intercepts_l396_396322

theorem equal_intercepts (a : ℝ) (h : (a + 1) ≠ 0) :
  (a - 2 = (a - 2) / (a + 1)) → (a = 2 ∨ a = 0) :=
begin
  intro heq,
  set x_intercept := (a - 2) / (a + 1),
  set y_intercept := a - 2,
  have heq' : y_intercept = x_intercept := heq,
  rw ← heq' at *,
  sorry -- Proof steps omitted
end

end equal_intercepts_l396_396322


namespace domain_of_f_range_of_f_no_monotonically_increasing_a_l396_396898

def u (a x : ℝ) : ℝ := x^2 - 2*a*x + 3
def f (a x : ℝ) : ℝ := Real.logBase (1 / 2) (u a x)

theorem domain_of_f (a : ℝ) : 
  (∀ x : ℝ, u a x > 0) ↔ (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) :=
by 
  sorry

theorem range_of_f (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (a ≤ -Real.sqrt 3 ∨ a ≥ Real.sqrt 3) :=
by 
  sorry

theorem no_monotonically_increasing_a (a : ℝ) :
  ¬ (∀ x ∈ Set.Ioo (-∞ : ℝ) 2, Deriv.differentiableOn ℝ (f a) (Set.Ioo (-∞ : ℝ) 2) 
  ∧ (∀ x ∈ Set.Ioo (-∞ : ℝ) 2, Deriv.deriv (f a) x > 0)) :=
by 
  sorry

end domain_of_f_range_of_f_no_monotonically_increasing_a_l396_396898


namespace ratio_problem_l396_396947

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396947


namespace find_angle_A_l396_396959

theorem find_angle_A (A B : ℝ) (a b : ℝ) (h1 : b = 2 * a * Real.sin B) (h2 : a ≠ 0) :
  A = 30 ∨ A = 150 :=
by
  sorry

end find_angle_A_l396_396959


namespace number_of_true_statements_l396_396836

theorem number_of_true_statements 
  (m n : ℝ)
  (prop : |m| > |n| → m^2 > n^2)
  (contra : m^2 ≤ n^2 → |m| ≤ |n|)
  (conv : m^2 > n^2 → |m| > |n|)
  (inv : |m| ≤ |n| → m^2 ≤ n^2) :
  prop ∧ contra ∧ conv ∧ inv :=
sorry

end number_of_true_statements_l396_396836


namespace combination_7_2_l396_396027

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396027


namespace count_integers_with_digit_sum_16_l396_396913

theorem count_integers_with_digit_sum_16 :
  {n : ℕ | 300 ≤ n ∧ n < 500 ∧ (nat.digits 10 n).sum = 16}.to_finset.card = 13 :=
sorry

end count_integers_with_digit_sum_16_l396_396913


namespace avg_first_300_terms_l396_396071

def seq_term (n : ℕ) : ℤ := (-1)^(n+1) * 2 * n

theorem avg_first_300_terms : (∑ i in Finset.range 300, seq_term i) / 300 = -1 := 
by
  sorry

end avg_first_300_terms_l396_396071


namespace trapezoid_condition_l396_396283

-- Define the problem statement in Lean
theorem trapezoid_condition (A B C D M N : Point) (h1 : M = midpoint A B) (h2 : N = midpoint C D)
  (h3 : area (quadrilateral A M N B) = area (quadrilateral M N C D)) : 
  parallel (line A D) (line B C) :=
sorry

end trapezoid_condition_l396_396283


namespace binomial_7_2_l396_396040

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396040


namespace intersection_points_3_l396_396476

def eq1 (x y : ℝ) : Prop := (x - y + 3) * (2 * x + 3 * y - 9) = 0
def eq2 (x y : ℝ) : Prop := (2 * x - y + 2) * (x + 3 * y - 6) = 0

theorem intersection_points_3 :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y) ∧
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    eq1 x1 y1 ∧ eq2 x1 y1 ∧ 
    eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    eq1 x3 y3 ∧ eq2 x3 y3 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :=
sorry

end intersection_points_3_l396_396476


namespace angle_and_length_theorem_l396_396812

-- Define the conditions and statement of the theorem

variables {A B C L M N : Type}
variable [metric_space A]
variable [metric_space B]
variable [metric_space C]
variable [metric_space L]
variable [metric_space M]
variable [metric_space N]

variable (AL : A -> L -> Prop)
variable (BM : B -> M -> Prop)
variable (CN : C -> N -> Prop)

variable (angle_bisectors : ∀ {A B C : Type}, AL A L ∧ BM B M ∧ CN C N)
variable (angle_eq : ∀ {A N M L}, angle (AN ∧ NM) = angle (AL ∧ LC))

-- Define the theorem to be proved
theorem angle_and_length_theorem : 
  angle ACB = 120 ∧ (NM^2 + NL^2 = ML^2) :=
begin
  -- Here the proof would go
  sorry
end

end angle_and_length_theorem_l396_396812


namespace decreasing_log_function_l396_396325

open Real

noncomputable def f (a x : ℝ) : ℝ := log a (6 - a * x)

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x > f y

theorem decreasing_log_function (a : ℝ) : 
  (1 < a ∧ a ≤ 3) ↔ is_decreasing_on (f a) (set.Ioo 0 2) :=
by
  sorry

end decreasing_log_function_l396_396325


namespace problem1_problem2_l396_396521

-- Problem 1 Statement
theorem problem1 (f h g : ℝ → ℝ) (D : set ℝ) (Hf : ∀ x ∈ D, f x = x^2 + x)
  (Hh : ∀ x ∈ D, h x = -x^2 + x) (Hg : ∀ x ∈ D, f x ≥ g x ∧ g x ≥ h x) :
  ∀ x, g x = x :=
sorry

-- Problem 2 Statement
theorem problem2 (f h g : ℝ → ℝ) (D : set ℝ) (k : ℝ) (Hf : ∀ x ∈ D, f x = x^2 + x + 2)
  (Hh : ∀ x ∈ D, h x = x - 1/x) (Hg : ∀ x ∈ D, g x = k * x + 1) (Hrange : ∀ x ∈ D, f x ≥ g x ∧ g x ≥ h x) :
  1 ≤ k ∧ k ≤ 3 :=
sorry

end problem1_problem2_l396_396521


namespace unique_arrangement_l396_396853

def chair := Fin 5 

structure Arrangement := 
  (positions : chair → chair)
  (no_fixed_point : ∀ (i : chair), positions i ≠ i)
  (no_adjacent : ∀ (i : chair), positions i ≠ (i + 1) % 5 ∧ positions i ≠ (i + 4) % 5)
  (person1_to_5 : positions 0 = 4)

theorem unique_arrangement : ∃! (arr : Arrangement), arr.positions 0 = 4 ∧ arr.no_fixed_point ∧ arr.no_adjacent := sorry

end unique_arrangement_l396_396853


namespace problem_solution_l396_396264

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem problem_solution
  (k : ℕ)
  (a : ℝ)
  (h1 : (∑ i in Finset.range 20, abs ((k + i) - a)) = 360)
  (h2 : (∑ i in Finset.range 20, abs ((k + i) - a^2)) = 345) :
  a = -0.5 ∨ a = 1.5 :=
sorry

end problem_solution_l396_396264


namespace part_a_part_b_l396_396392

def fake_coin_min_weighings_9 (n : ℕ) : ℕ :=
  if n = 9 then 2 else 0

def fake_coin_min_weighings_27 (n : ℕ) : ℕ :=
  if n = 27 then 3 else 0

theorem part_a : fake_coin_min_weighings_9 9 = 2 := by
  sorry

theorem part_b : fake_coin_min_weighings_27 27 = 3 := by
  sorry

end part_a_part_b_l396_396392


namespace temperature_equivalence_l396_396341

theorem temperature_equivalence (x : ℝ) (h : x = (9 / 5) * x + 32) : x = -40 :=
sorry

end temperature_equivalence_l396_396341


namespace units_digit_33_exp_l396_396837

def units_digit_of_power_cyclic (base exponent : ℕ) (cycle : List ℕ) : ℕ :=
  cycle.get! (exponent % cycle.length)

theorem units_digit_33_exp (n : ℕ) (h1 : 33 = 1 + 4 * 8) (h2 : 44 = 4 * 11) :
  units_digit_of_power_cyclic 33 (33 * 44 ^ 44) [3, 9, 7, 1] = 3 :=
by
  sorry

end units_digit_33_exp_l396_396837


namespace inequalities_always_true_l396_396669

theorem inequalities_always_true (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) 
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ (x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b) := by
  sorry

end inequalities_always_true_l396_396669


namespace GEB_100th_term_l396_396316

-- Definitions based on conditions from part (a)

def is_increasing (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n < seq (n+1)

def diff_seq (seq : ℕ → ℕ) : ℕ → ℕ :=
  λ n, seq (n+1) - seq n

def is_growing (seq : ℕ → ℕ) : Prop :=
  is_increasing (diff_seq seq)

def positive_integers_exclusive (seq : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → (¬ ∃ n : ℕ, seq n = i) → (∃ n : ℕ, diff_seq seq n = i)

-- The GEB sequence satisfies the above properties
def GEB_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 7
  | _ => sorry  -- Further definition skipped

-- Lean statement to verify the 100th term of the GEB sequence using conditions
theorem GEB_100th_term :
  is_increasing GEB_sequence ∧
  is_growing GEB_sequence ∧
  positive_integers_exclusive GEB_sequence →
  GEB_sequence 99 = 5764 :=
by
  sorry

end GEB_100th_term_l396_396316


namespace classroom_students_count_l396_396336

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l396_396336


namespace distance_between_vertices_l396_396091
noncomputable theory

def hyperbola_eq : (ℝ → ℝ → ℝ) := λ x y, 16 * y ^ 2 - 32 * y - 4 * x ^ 2 - 24 * x + 84

theorem distance_between_vertices :
  ∃ a : ℝ, (∀ x y : ℝ, hyperbola_eq x y = 0) ∧ (2 * real.sqrt 6.5 = a) :=
sorry

end distance_between_vertices_l396_396091


namespace combination_7_2_l396_396032

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396032


namespace arithmetic_progression_conditions_l396_396846

theorem arithmetic_progression_conditions (a d : ℝ) :
  let x := a
  let y := a + d
  let z := a + 2 * d
  (y^2 = (x^2 * z^2)^(1/2)) ↔ (d = 0 ∨ d = a * (-2 + Real.sqrt 2) ∨ d = a * (-2 - Real.sqrt 2)) :=
by
  intros
  sorry

end arithmetic_progression_conditions_l396_396846


namespace gunthers_monthly_payment_l396_396558

theorem gunthers_monthly_payment (total_amount : ℕ) (term_years : ℕ) 
  (no_interest : true) : 
  total_amount = 9000 → term_years = 5 → (total_amount / (term_years * 12)) = 150 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end gunthers_monthly_payment_l396_396558


namespace parallelogram_base_l396_396845

theorem parallelogram_base (A h : ℝ) (A_eq : A = 416) (h_eq : h = 16) : 
  let b := A / h in b = 26 :=
by
  sorry

end parallelogram_base_l396_396845


namespace savings_by_december_l396_396628

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l396_396628


namespace _l396_396876

def Ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def TriangleArea (A B O : (ℝ × ℝ)) := 
  1/2 * (A.1 - O.1) * (B.2 - O.2) = (real.sqrt 2) / 4

def LineThroughPointWithSlope (P : (ℝ × ℝ)) (k : ℝ) := ∀ x y : ℝ, y = k * x + P.2

noncomputable theorem proof_problem 
  (a b k λ μ : ℝ)
  (O : (ℝ × ℝ)) 
  (A : (ℝ × ℝ := (1, 0)))
  (B : (ℝ × ℝ))
  (P : (ℝ × ℝ := (0, 1)))
  (S T : (ℝ × ℝ))
  (ellipse_condition : Ellipse a b)
  (triangle_area_condition : TriangleArea A B O)
  (line_l_condition : LineThroughPointWithSlope P k)
  (intersection_points : ∃ M N : (ℝ × ℝ), ∃ y₁ y₂ : ℝ, y₁ ≠ y₂)
  (circle_diameter_condition : ∀ M N: ℝ, M * N + y₁ * y₂ = 0)
  : (x^2 + 2y^2 = 1) ∧ 
    (y = real.sqrt 2 * x + 1) ∧ 
    ((λ + μ) ∈ set.Ioo (real.sqrt 2) 2) :=
sorry

end _l396_396876


namespace prove_k_eq_one_l396_396253

theorem prove_k_eq_one 
  (n m k : ℕ) 
  (h_positive : 0 < n)  -- implies n, and hence n-1, n+1, are all positive
  (h_eq : (n-1) * n * (n+1) = m^k): 
  k = 1 := 
sorry

end prove_k_eq_one_l396_396253


namespace solve_equation_1_solve_equation_2_l396_396852

theorem solve_equation_1 (x : ℝ) : 5 * x^2 - 10 = 0 ↔ x = real.sqrt 2 ∨ x = -real.sqrt 2 :=
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * (x - 4)^2 = 375 ↔ x = 4 + 5 * real.sqrt 5 ∨ x = 4 - 5 * real.sqrt 5 :=
by sorry

end solve_equation_1_solve_equation_2_l396_396852


namespace solution_set_of_inequality_l396_396535

-- Define the even function and its properties
def even_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f(x) = f(-x)

-- Define increasing on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f(x) ≤ f(y)

-- Define the problem statement
theorem solution_set_of_inequality (f : ℝ → ℝ) :
  even_function f →
  increasing_on_nonneg f →
  { x : ℝ | f (2 * x - 1) < f (1 / 3) } = { x : ℝ | 1 / 3 < x ∧ x < 2 / 3 } :=
by
  intros h_even h_increasing
  sorry

end solution_set_of_inequality_l396_396535


namespace sum_distances_between_l396_396621

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

theorem sum_distances_between (A B D : ℝ × ℝ)
  (hB : B = (0, 5))
  (hD : D = (8, 0))
  (hA : A = (20, 0)) :
  21 < distance A D + distance B D ∧ distance A D + distance B D < 22 :=
by
  sorry

end sum_distances_between_l396_396621


namespace domain_of_f_l396_396710

-- Define the function
def f (x : ℝ) : ℝ := real.sqrt (x^3 - 1)

-- State the theorem
theorem domain_of_f : ∀ x : ℝ, x^3 - 1 ≥ 0 ↔ x ≥ 1 := by
  sorry

end domain_of_f_l396_396710


namespace alex_shirts_4_l396_396432

/-- Define the number of new shirts Alex, Joe, and Ben have. -/
def shirts_of_alex (alex_shirts : ℕ) (joe_shirts : ℕ) (ben_shirts : ℕ) : Prop :=
  joe_shirts = alex_shirts + 3 ∧ ben_shirts = joe_shirts + 8 ∧ ben_shirts = 15

theorem alex_shirts_4 {alex_shirts : ℕ} :
  ∃ joe_shirts ben_shirts, shirts_of_alex alex_shirts joe_shirts ben_shirts ∧ alex_shirts = 4 :=
by
  have joe_shirts := 4 + 3 by rfl
  have ben_shirts := 7 + 8 by rfl
  use joe_shirts, ben_shirts
  split
  . exact ⟨rfl, rfl, rfl⟩
  . exact rfl
  sorry

end alex_shirts_4_l396_396432


namespace disproves_proposition_specific_disproof_example_l396_396503

theorem disproves_proposition (a b : ℤ) (h : a^2 > b^2) : ¬ (a > b) :=
  sorry

-- Specific example for Option C
theorem specific_disproof_example : disproves_proposition (-3) (-2) :=
begin
  show ¬ ((-3) > (-2)),
  exact lt_irrefl 3,
  sorry
end

end disproves_proposition_specific_disproof_example_l396_396503


namespace probability_of_cosine_negative_is_correct_l396_396135

-- Define the conditions of the arithmetic sequence
variables (a_n S_n : ℕ → ℝ) (d : ℝ)
noncomputable def S (n : ℕ) := (n * (a_n(1) + a_n(n))) / 2

-- Define the given conditions
axiom S_4_eq_pi : S 4 = Real.pi
axiom a_4_eq_2a_2 : a_n 4 = 2 * a_n 2
axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d

-- Define the probability function
noncomputable def probability_of_negative_cosine : ℝ := 
  let count := (list.range 30).countp (λ n, let a := a_n (n + 1) in
    ∃ k : ℤ, (Real.pi / 2 + 2 * ↑k * Real.pi < a ∧ a < 3 * Real.pi / 2 + 2 * ↑k * Real.pi)) in
  count / 30

-- Prove that the probability is 14/30
theorem probability_of_cosine_negative_is_correct :
  probability_of_negative_cosine a_n = 14 / 30 := sorry

end probability_of_cosine_negative_is_correct_l396_396135


namespace expression_equals_19_l396_396749

def a : ℂ := 3 - complex.I
def b : ℂ := 2 + complex.I
def c : ℂ := -1 + 2 * complex.I

theorem expression_equals_19 : 3 * a + 4 * b - 2 * c = 19 := by
  sorry

end expression_equals_19_l396_396749


namespace intersection_complement_M_N_l396_396170

noncomputable def M := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}
noncomputable def N := {x : ℝ | ∃ y : ℝ, y = sqrt (3 - x^2)}
noncomputable def M_complement := {y : ℝ | y < -1}
noncomputable def intersection := {x : ℝ | -sqrt 3 ≤ x ∧ x < -1}

theorem intersection_complement_M_N :
  {x : ℝ | x ∈ M_complement} ∩ N = {-sqrt 3, -1} :=
sorry

end intersection_complement_M_N_l396_396170


namespace convex_quadrilateral_inequality_l396_396607

theorem convex_quadrilateral_inequality 
  {A B C D O K L M N : Type} 
  [convex_quadrilateral A B C D] 
  (hO : inside_quadrilateral O A B C D)
  (hK : point_on_segment K A B)
  (hL : point_on_segment L B C)
  (hM : point_on_segment M C D)
  (hN : point_on_segment N D A)
  (hParallelogram1 : parallelogram O K B L)
  (hParallelogram2 : parallelogram O N D M)
  (S1 S2 : ℝ)
  (hS1 : area_quadrilateral O N A K = S1)
  (hS2 : area_quadrilateral O L C M = S2)
  (S : ℝ)
  (hS : area_quadrilateral A B C D = S) :
  S ≥ S1 + S2 + 2 * real.sqrt (S1 * S2) :=
by
  sorry

end convex_quadrilateral_inequality_l396_396607


namespace one_plus_one_invertible_ring_is_field_l396_396230

variables {A : Type*} [fintype A] [ring A] [nontrivial A]
variable (n : ℕ)
variable (h_card : fintype.card A = n)
variable (h_n_ge_3 : n ≥ 3)
variable (h_squares : (finset.univ.filter (λ b : A, ∃ a : A, a^2 = b)).card = (n + 1) / 2)

-- Proving that 1 + 1 is invertible in the ring
theorem one_plus_one_invertible :
  is_unit (1 + 1 : A) :=
sorry

-- Proving that the ring is a field
theorem ring_is_field :
  is_field A :=
sorry

end one_plus_one_invertible_ring_is_field_l396_396230


namespace axis_of_symmetry_cosine_l396_396323

theorem axis_of_symmetry_cosine (k : ℤ) :
  ∃ x : ℝ, 2 * x + π / 6 = k * π ∧ x = -π / 12 :=
begin
  use -π / 12,
  split,
  {
    have h1 : 2 * (-π / 12) = -π / 6 := by ring,
    linarith,
  },
  refl,
end

end axis_of_symmetry_cosine_l396_396323


namespace intersect_at_diametrically_opposite_points_l396_396825

-- Define the basic geometric setting and properties
def circle (center : Point) (radius : ℝ) : set Point := 
  {p | dist p center = radius}

def tangent (p : Point) (c : set Point) : Prop := 
  ∃ center radius, c = circle center radius ∧ dist p center = radius

-- Define the problem conditions
variable (c c1 c2 : set Point)
variable (A B P Q : Point)
variable [tangent A c ∧ tangent B c]
variable [tangent P c1 ∧ tangent Q c2]
variable (ℓ : set Point)  -- internal tangent line to c1 and c2
variable [∀ p ∈ ℓ, p ∈ tangent P c1 ∨ p ∈ tangent Q c2]

-- Statement of the theorem
theorem intersect_at_diametrically_opposite_points :
  let AP := line_through A P
  let BQ := line_through B Q
  ∃ M N ∈ c, M ≠ N ∧ 
  M ∈ AP ∧ N ∈ BQ ∧ diametrically_opposite M N :=
sorry

end intersect_at_diametrically_opposite_points_l396_396825


namespace log_probability_probability_solution_l396_396161

theorem log_probability 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = Real.log x / Real.log 3)
  (x0 : set.Icc (1 : ℝ) 27) :
  ℝ :=
begin
  sorry
end

theorem probability_solution :
  log_probability (λ x, Real.log x / Real.log 3) (by { intro x, rw Real.log, field_simp, }) set.Icc.mk 27 (one_le_of_lt (by norm_num : 1 < 27)) = (3 / 13) := 
begin
  sorry
end

end log_probability_probability_solution_l396_396161


namespace incorrect_negation_l396_396809

theorem incorrect_negation :
  (¬(∃ (x : ℤ), (∃ (k : ℤ), x = 3 * k) → ∃ (x : ℤ), ¬(x % 2 = 1)) ∧
  ¬(∀ (q : Type) [quadrilateral q], ∃ c : circle, ∀ v : vertex q, v ∈ c → ¬(∀ (v : vertex q), ∃ c : circle, ∀ v : vertex q, v ∈ c)) ∧
  ¬(∃ (t : Type) [triangle t], (∃ (t : t, is_equilateral t)) → ∀ (t : Type) [triangle t], ¬(is_equilateral t)) ∧
  (¬(∃ (x : ℝ), x^2 + 2*x + 2 ≤ 0) /\ (∀ (x : ℝ), x^2 + 2*x + 2 > 0))

end incorrect_negation_l396_396809


namespace bus_driver_total_compensation_l396_396783

-- Define the regular rate
def regular_rate : ℝ := 16

-- Define the number of regular hours
def regular_hours : ℕ := 40

-- Define the overtime rate as 75% higher than the regular rate
def overtime_rate : ℝ := regular_rate * 1.75

-- Define the total hours worked in the week
def total_hours_worked : ℕ := 48

-- Calculate the overtime hours
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Calculate the total compensation
def total_compensation : ℝ :=
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

-- Theorem to prove that the total compensation is $864
theorem bus_driver_total_compensation : total_compensation = 864 := by
  -- Proof is omitted
  sorry

end bus_driver_total_compensation_l396_396783


namespace remainder_when_3x_7y_5z_div_31517_l396_396263

theorem remainder_when_3x_7y_5z_div_31517
  (x y z : ℕ)
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  (3 * x + 7 * y - 5 * z) % 31517 = ((69 * (x / 23) + 203 * (y / 29) - 185 * (z / 37) + 72) % 31517) := 
sorry

end remainder_when_3x_7y_5z_div_31517_l396_396263


namespace general_seq_a_general_seq_T_max_terms_sequence_b_l396_396726

-- Define the sequence a_n and its partial sum S_n
def partial_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 2 * (a n) - 1

-- Define the general sequence a_n
def seq_a (a : ℕ → ℕ) : Prop := 
∀ n : ℕ, n ≥ 1 → (a (n) = 2^ (n - 1))

-- Define the sequence T_n
def seq_T (T a : ℕ → ℕ) : Prop := 
∀ n : ℕ, T n = (2 / 5) * (1 - (-4)^n)

-- Condition for sequence b_n
def seq_b (b : ℕ → ℕ) (a m : ℕ) : Prop := 
∀ n : ℕ, b n = b (n-1) + 1 ∧ 
(log_of_2(2) + (Σ (1 ≤ i ≤ m, log_of_2 (1 + 1 / b i))) = log_of_2(log_of_2(a m)))

-- Maximum number of terms and sum of sequence b_n
def max_terms_sum (b : ℕ → ℕ) : ℕ × ℕ := (9, 63)

-- Lean 4 Theorem statements
theorem general_seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (Seq_a : seq_a a) (Partial_sum : partial_sum S a) : 
a n = 2^(n-1) := sorry

theorem general_seq_T (a T : ℕ → ℕ) (Seq_a : seq_a a) (Seq_T : seq_T T a) : 
T n = (2 / 5) * (1 - (-4)^n) := sorry

theorem max_terms_sequence_b (a b : ℕ → ℕ) (Seq_b : seq_b b a) : 
max_terms_sum b = (9, 63) := sorry

end general_seq_a_general_seq_T_max_terms_sequence_b_l396_396726


namespace alex_shirts_4_l396_396430

/-- Define the number of new shirts Alex, Joe, and Ben have. -/
def shirts_of_alex (alex_shirts : ℕ) (joe_shirts : ℕ) (ben_shirts : ℕ) : Prop :=
  joe_shirts = alex_shirts + 3 ∧ ben_shirts = joe_shirts + 8 ∧ ben_shirts = 15

theorem alex_shirts_4 {alex_shirts : ℕ} :
  ∃ joe_shirts ben_shirts, shirts_of_alex alex_shirts joe_shirts ben_shirts ∧ alex_shirts = 4 :=
by
  have joe_shirts := 4 + 3 by rfl
  have ben_shirts := 7 + 8 by rfl
  use joe_shirts, ben_shirts
  split
  . exact ⟨rfl, rfl, rfl⟩
  . exact rfl
  sorry

end alex_shirts_4_l396_396430


namespace classroom_student_count_l396_396338

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l396_396338


namespace binom_7_2_eq_21_l396_396052

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396052


namespace intervals_of_monotonicity_range_of_a_l396_396896

noncomputable def f (x : ℝ) (a : ℝ) := Real.log (x + 1) - (a * x) / (1 - x)

theorem intervals_of_monotonicity : 
  ∀ x : ℝ, -1 < x ∧ x < 1 ∧ a = 1 → 
  (   (∀ x, -1 < x ∧ x < 0 → (f x 1 > 0)) 
   ∨ (∀ x, 3 < x → (f x 1 > 0)) 
   ∨ (∀ x, 0 < x ∧ x < 1 → (f x 1 < 0)) 
   ∨ (∀ x, 1 < x ∧ x < 3 → (f x 1 < 0))). 
sorry

theorem range_of_a (h : ∀ x, -1 < x ∧ x < 1 → f x a ≤ 0) : a = 1 :=
sorry

end intervals_of_monotonicity_range_of_a_l396_396896


namespace part1_part2_part3_l396_396515

noncomputable def a (n : ℕ) : ℚ := (1/4)^n

def b (n : ℕ) : ℚ := 3 * (Int.log (a (n : ℕ)) $\frac{1}{4}$) - 2

def c (n : ℕ) : ℚ := a n * b n

def S (n : ℕ) : ℚ := (List.range n).sum (λ i, c (i + 1))

theorem part1 (hn : ℕ) : ∀ n : ℕ, b (n + 1).succ = b (n + 1) + 3 :=
by sorry

theorem part2 (n : ℕ) : S n = (2 / 3) - ((12 * n + 8) / 3) * ((1 / 4) ^ (n + 1)) :=
by sorry

theorem part3 (m : ℚ) : (∀ n : ℕ, c n ≤ (1/4) * m^2 + m - 1) → (m ≥ 1 ∨ m ≤ -5) :=
by sorry

end part1_part2_part3_l396_396515


namespace num_true_propositions_l396_396877

-- Define the arithmetic sequence
def arith_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the propositions
def seq1_increasing (a₁ d : ℕ) (d_pos: d > 0) : Prop :=
  ∀ n m : ℕ, n < m → arith_seq a₁ d n < arith_seq a₁ d m

def seq2_increasing (a₁ d : ℕ) (d_pos: d > 0) : Prop :=
  ∀ n m : ℕ, n < m → n * arith_seq a₁ d n < m * arith_seq a₁ d m

def seq3_decreasing (a₁ d : ℕ) (d_pos: d > 0) : Prop :=
  ∀ n m : ℕ, n < m → arith_seq a₁ d n / n > arith_seq a₁ d m / m

def seq4_increasing (a₁ d : ℕ) (d_pos: d > 0) : Prop :=
  ∀ n m : ℕ, n < m → (arith_seq a₁ d n + 3 * n * d) < (arith_seq a₁ d m + 3 * m * d)

-- The main proof statement
theorem num_true_propositions (a₁ d : ℕ) (h1: seq1_increasing a₁ d (by linarith))
  (h2: ¬ seq2_increasing a₁ d (by linarith)) (h3: ¬ seq3_decreasing a₁ d (by linarith))
  (h4: seq4_increasing a₁ d (by linarith)) : 2 := by
  sorry

end num_true_propositions_l396_396877


namespace correct_mark_up_percent_l396_396408

def list_price : ℝ := 100
def purchase_price : ℝ := list_price * 0.70
def marked_price : ℝ := 133.33
def selling_price (x : ℝ) : ℝ := 0.75 * x
def profit_condition (x : ℝ) : Prop := selling_price x - purchase_price = 0.30 * selling_price x

theorem correct_mark_up_percent :
  ∃ x, x = marked_price ∧ profit_condition x :=
by
  use 133.33
  unfold profit_condition selling_price purchase_price list_price
  have h_eq : 0.75 * 133.33 - 70 = 0.30 * (0.75 * 133.33), from sorry
  exact ⟨rfl, h_eq⟩

end correct_mark_up_percent_l396_396408


namespace seventh_data_entry_value_l396_396409

theorem seventh_data_entry_value :
  ∀ (results : List ℝ) (a b : ℕ) (average total_sum : ℝ),
  results.length = 15 →
  total_sum = 60 * 15 →
  (∀ (first_set second_set last_set : List ℝ),
    first_set.length = 7 ∧ second_set.length = 6 ∧ last_set.length = 6 →
    results = first_set ++ second_set ++ last_set →
    (first_set.sum / 7 = 56) →
    (second_set.sum / 6 = 63) →
    (last_set.sum / 6 = 66) →
    last_set.sum = second_set.sum + first_set.nth_le 6 sorry →
    results.nth_le 6 sorry = 18) :=
begin
  sorry
end

end seventh_data_entry_value_l396_396409


namespace ice_bag_cost_correct_l396_396014

def total_cost_after_discount (cost_small cost_large : ℝ) (num_bags num_small : ℕ) (discount_rate : ℝ) : ℝ :=
  let num_large := num_bags - num_small
  let total_cost_before_discount := num_small * cost_small + num_large * cost_large
  let discount := discount_rate * total_cost_before_discount
  total_cost_before_discount - discount

theorem ice_bag_cost_correct :
  total_cost_after_discount 0.80 1.46 30 18 0.12 = 28.09 :=
by
  sorry

end ice_bag_cost_correct_l396_396014


namespace coefficient_of_x_pow_31_l396_396021

theorem coefficient_of_x_pow_31 :
  coeff (expand_polynomial (\sum_{i=0}^{30} x^i * (\sum_{j=0}^{17} x^j) ^ 2)) 31 = -737 :=
sorry

end coefficient_of_x_pow_31_l396_396021


namespace probability_sqrt_two_digit_lt_7_l396_396369

theorem probability_sqrt_two_digit_lt_7 : 
  let two_digit_set := Finset.Icc 10 99
  let favorable_set := Finset.Icc 10 48
  (favorable_set.card : ℚ) / two_digit_set.card = 13 / 30 :=
by sorry

end probability_sqrt_two_digit_lt_7_l396_396369


namespace binomial_7_2_l396_396056

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396056


namespace largest_modulus_of_z_l396_396255

noncomputable def complex_largest_modulus (a b c z : ℂ) (r : ℝ) : Prop :=
  |a| = r ∧ |c| = r ∧ r > 0 ∧ |b| = 2 * r ∧ az^2 + bz + c = 0

theorem largest_modulus_of_z (a b c z : ℂ) (r : ℝ)
  (h : complex_largest_modulus a b c z r) : |z| ≤ 1 + Real.sqrt 2 := 
sorry

end largest_modulus_of_z_l396_396255


namespace savings_by_december_l396_396629

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l396_396629


namespace math_problem_l396_396020

theorem math_problem :
  (Int.ceil ((18: ℚ) / 5 * (-25 / 4)) - Int.floor ((18 / 5) * Int.floor (-25 / 4))) = 4 := 
by
  sorry

end math_problem_l396_396020


namespace area_of_square_field_l396_396005

-- defining the total length of wire and the number of times it rounds the field
def total_length : ℕ := 15840
def rounds : ℕ := 15

-- defining the side length and area of the square
def side_length (total_length : ℕ) (rounds : ℕ) : ℕ := total_length / (4 * rounds)
def area (s : ℕ) : ℕ := s * s

-- statement of the problem
theorem area_of_square_field :
  let s := side_length total_length rounds in
  area s = 69696 :=
by
  let s := side_length total_length rounds
  have : s = 264 := by sorry
  show area s = 69696 from by sorry

end area_of_square_field_l396_396005


namespace canoe_total_weight_l396_396276

def total_people (num_people : ℕ) (dog_factor : ℚ) : ℚ := dog_factor * num_people

def total_person_weight (num_people : ℕ) (weight_per_person : ℕ) : ℕ := num_people * weight_per_person

def dog_weight (person_weight : ℕ) (dog_ratio : ℚ) : ℚ := dog_ratio * person_weight

def total_canoe_weight (person_weight : ℕ) (dog_weight : ℚ) : ℚ := person_weight + dog_weight

theorem canoe_total_weight :
  let
    people_wt := total_person_weight 4 140 -- total weight of 4 people
    dog_wt := dog_weight 140 (1 / 4 : ℚ) -- weight of the dog
    total_wt := total_canoe_weight people_wt dog_wt -- total weight in the canoe
  in total_wt = 595 := by
sorry

end canoe_total_weight_l396_396276


namespace liars_count_possible_l396_396967

noncomputable def num_liars (answers : Finset Nat) (num_people : Nat) : Prop :=
  answers = Finset.range 1 101 ∧ ∃ k, k > 0 ∧ k < num_people ∧ (num_people - k = 1 ∨ num_people - k = 2)

theorem liars_count_possible (answers : Finset Nat) (num_people : Nat) (num_knights : Nat) :
  answers = Finset.range 1 101 →
  num_knights > 0 →
  num_liars answers num_people → (num_people - num_knights = 99 ∨ num_people - num_knights = 100) :=
sorry

end liars_count_possible_l396_396967


namespace trapezoid_median_l396_396803

noncomputable def median_trapezoid (base₁ base₂ height : ℝ) : ℝ :=
(base₁ + base₂) / 2

theorem trapezoid_median (b_t : ℝ) (a_t : ℝ) (h_t : ℝ) (a_tp : ℝ) 
  (h_eq : h_t = 16) (a_eq : a_t = 192) (area_tp_eq : a_tp = a_t) : median_trapezoid h_t h_t h_t = 12 :=
by
  have h_t_eq : h_t = 16 := by sorry
  have a_t_eq : a_t = 192 := by sorry
  have area_tp : a_tp = 192 := by sorry
  sorry

end trapezoid_median_l396_396803


namespace fermat_coprime_l396_396760

theorem fermat_coprime (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 :=
sorry

end fermat_coprime_l396_396760


namespace perpendicular_lines_l396_396888

def vec3 := ℝ × ℝ × ℝ

def dot_product (a b : vec3) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem perpendicular_lines :
  let a : vec3 := (1, 2, -2)
  let b : vec3 := (-2, 3, 2)
  dot_product a b = 0 :=
by
  -- proof to be written
  sorry

end perpendicular_lines_l396_396888


namespace binom_7_2_eq_21_l396_396049

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396049


namespace exists_root_in_interval_l396_396835

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 3

theorem exists_root_in_interval : ∃ c ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  sorry

end exists_root_in_interval_l396_396835


namespace largest_fraction_sum_l396_396951

theorem largest_fraction_sum (a b c d : ℕ) (h : {a, b, c, d} = {3, 4, 6, 7}) :
  (∃ n₁ d₁ n₂ d₂ : ℕ, n₁ ≠ d₁ ∧ n₂ ≠ d₂ ∧ {n₁, d₁} ∪ {n₂, d₂} = {a, b, c, d} ∧
   (n₁ / d₁ + n₂ / d₂ : ℝ) = 23 / 6) :=
begin
  sorry
end

end largest_fraction_sum_l396_396951


namespace export_volume_scientific_notation_l396_396422

theorem export_volume_scientific_notation :
  (234.1 * 10^6) = (2.341 * 10^8) := 
sorry

end export_volume_scientific_notation_l396_396422


namespace muffins_originally_baked_l396_396436

theorem muffins_originally_baked :
  let monday := 1
  let tuesday := monday + 1
  let wednesday := tuesday + 1
  let thursday := wednesday + 1
  let friday := thursday + 1
  let total_brought := monday + tuesday + wednesday + thursday + friday
  let leftover := 7 in
  (total_brought + leftover) = 22 := 
by
  let monday := 1
  let tuesday := monday + 1
  let wednesday := tuesday + 1
  let thursday := wednesday + 1
  let friday := thursday + 1
  let total_brought := monday + tuesday + wednesday + thursday + friday
  let leftover := 7
  have : total_brought = 15 := rfl
  have : total_brought + leftover = 22
  show (total_brought + leftover) = 22
  sorry

end muffins_originally_baked_l396_396436


namespace ratio_expression_value_l396_396918

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396918


namespace seq_property_question_l396_396132

noncomputable def seq (n : Nat) : ℝ := 
  if n = 1 then 1 else
  if n = 2 then Real.sqrt 3 else
  seq_aux n  -- define seq_aux following the recurrence relation

-- auxiliary definition to follow the recurrence relation
noncomputable def seq_aux (n : Nat) : ℝ :=
  seq_aux_rec n
  where
    seq_aux_rec : Nat → ℝ 
    | 0        := 1
    | 1        := Real.sqrt 3
    | (k + 2)  := Real.sqrt ((2 * (seq k) - 2 * (seq (k - 1)) + 1) + (seq (k + 1))^2)

theorem seq_property : ∀ (n : ℕ), n ≥ 2 → 
  (seq (n + 1))^2 - (seq n)^2 = 2 * (seq n) - 2 * (seq (n - 1)) + 1 := sorry

theorem question : (seq 2023)^2 - 2 * (seq 2022) = 2022 :=
  sorry

end seq_property_question_l396_396132


namespace frequency_count_third_group_l396_396412

theorem frequency_count_third_group 
  (x n : ℕ)
  (h1 : n = 420 - x)
  (h2 : x / (n:ℚ) = 0.20) :
  x = 70 :=
by sorry

end frequency_count_third_group_l396_396412


namespace solution_set_of_inequality_l396_396996

variable {α : Type*} [LinearOrder α]

def is_decreasing (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (domain_cond : ∀ x, x ∈ Set.Ioo (-2 : ℝ) 2 → x ∈ Set.Ioo (-2 : ℝ) 2)
  : { x | x > 0 ∧ x < 1 } = { x | f x > f (2 - x) } :=
by {
  sorry
}

end solution_set_of_inequality_l396_396996


namespace find_a_l396_396153

-- Define the lines and conditions
def line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, a * x + 2 * y + 6 = 0

def line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x + (a - 1) * y + (a^2 - 1) = 0

-- Main theorem
theorem find_a (a : ℝ) : (∀ x y, line1 a x y → ∀ x y, line2 a x y → 
  (¬ (line1 a = line2 a))) → a = -1 :=
by
  sorry

end find_a_l396_396153


namespace triangle_area_l396_396205

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1/2 * a * b * sin C

theorem triangle_area
  (a b c : ℝ)
  (A B C : ℝ)
  (hb : b = 2)
  (hB : B = π / 3)
  (h_relation : c * sin A = sqrt 3 * a * cos C) :
  area_of_triangle a b c A B C = sqrt 3 := 
sorry

end triangle_area_l396_396205


namespace midpoint_on_nine_point_circle_l396_396881

open Locale Classical
noncomputable theory

variables {A B C D E H F K M : Type} 

-- Define the points and properties in the setup
variables (circumcircle : Set A)
variables [geometry A]
variables (AE : segment A E)
variables (circum_circle_of_triangle : circumcircle_of (triangle A B C) = circumcircle)
variables (orthocenter : H = orthocenter_of (triangle A B C))
variables (D_on_circumcircle : D ∈ circumcircle ∧ H ∈ line_segment E D)
variables (F_mid_HD : midpoint F H D)
variables (nine_point_circle : Set A)
variables (nine_point_circle_property : nine_point_circle = nine_point_circle_of (triangle A B C))

-- Prove that the midpoint of HD lies on the nine-point circle of triangle ABC
theorem midpoint_on_nine_point_circle :
  F ∈ nine_point_circle :=
sorry

end midpoint_on_nine_point_circle_l396_396881


namespace sixth_largest_number_l396_396844

theorem sixth_largest_number : 
  ∃ num_list : List Nat, 
  (∀ n ∈ num_list, n / 1000 ≠ 0 ∧ (Set.toFinset (Nat.digits n) = {1, 3, 0, 5}) ) 
  ∧ List.nthLe (num_list.insertionSort (≥) ↔ List.sort (≥) num_list) 5 (by simp [List.length_eq, sorry]) = 5013 := sorry

end sixth_largest_number_l396_396844


namespace classroom_students_count_l396_396337

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l396_396337


namespace intersection_eq_l396_396662

-- Defining sets M and N based on the given conditions
def setM : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0 }
def setN : Set ℤ := {-2, -1, 0, 1, 2}

-- The proof goal is to show the intersection M ∩ N equals {0, 1, 2}
theorem intersection_eq : 
  (setM ∩ (setN : Set ℝ)) = { x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end intersection_eq_l396_396662


namespace cos_75_cos_15_minus_sin_75_sin_195_l396_396851

noncomputable def cos_diff_identity : ℝ :=
  let alpha := 75 * (Real.pi / 180)
  let beta := 15 * (Real.pi / 180)
  let gamma := 195 * (Real.pi / 180)
  cos alpha * cos beta - sin alpha * sin gamma

theorem cos_75_cos_15_minus_sin_75_sin_195 :
  cos_diff_identity = 1 / 2 :=
by
  sorry

end cos_75_cos_15_minus_sin_75_sin_195_l396_396851


namespace correct_options_for_f_l396_396166

noncomputable def f (x : ℝ) : ℝ := 2 * abs (sin x + cos x) - sin (2 * x)

theorem correct_options_for_f :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π → T ≠ 2 * π) → 
  (∀ x, f x = f (2 * π - x)) ∧ 
  (∀ x, x = π / 4 → ∀ y, f y = f (π / 4 - x + y)) ∧ 
  (∀ x, ∃ m, m = 1 ∧ ∀ y, m ≤ f y) ∧
  (∀ x, (π / 4) ≤ x ∧ x ≤ (π / 2) → monotone_on f (set.Icc (π / 4) (π / 2))) :=
by
  sorry

end correct_options_for_f_l396_396166


namespace sin_add_pi_over_4_l396_396585

theorem sin_add_pi_over_4 (α : ℝ) (h1 : cos α = -4/5) (h2 : π < α ∧ α < 3*π/2) : 
  sin (α + π/4) = -7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_add_pi_over_4_l396_396585


namespace trig_identity_second_quadrant_l396_396377

theorem trig_identity_second_quadrant (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0) : 
  (|sin α| / sin α) - (cos α / |cos α|) = 2 :=
by
  sorry

end trig_identity_second_quadrant_l396_396377


namespace bird_nest_trip_analysis_l396_396361

theorem bird_nest_trip_analysis :
  let distA_X := 15 * 300 * 2,
      distA_Z := 10 * 400 * 2,
      distB_Y := 20 * 500 * 2,
      distB_Z := 5 * 600 * 2,
      timeA_X := 15 * 30,
      timeA_Z := 10 * 40,
      timeB_Y := 20 * 60,
      timeB_Z := 5 * 50,

      total_dist_A := distA_X + distA_Z,
      total_dist_B := distB_Y + distB_Z,
      total_dist := total_dist_A + total_dist_B,

      total_time_A := timeA_X + timeA_Z,
      total_time_B := timeB_Y + timeB_Z,
      total_time := total_time_A + total_time_B,
      total_hours := total_time.toFloat / 60
  in total_dist = 43000 ∧ total_hours ≈ 38.33 :=
by
  -- Proof will go here
  sorry

end bird_nest_trip_analysis_l396_396361


namespace sequence_properties_l396_396872

-- Define the sequences and their properties based on given conditions
-- Condition 1: a_{n+1}^2 = 2S_n + n + 4
-- Condition 2: a_2 - 1, a_3, and a_7 form a geometric sequence {b_n}

noncomputable def a (n : ℕ) : ℕ := n + 1
noncomputable def b (n : ℕ) : ℕ := 2^n
noncomputable def c (n : ℕ) : ℤ := (-1)^n * a n * b n
noncomputable def T (n : ℕ) : ℤ := -(3 * n + 2) / (9 : ℤ) * (-2)^(n+1) - 2 / (9 : ℤ)

theorem sequence_properties
  (S : ℕ → ℕ)
  (h₁ : ∀ n, (a n + 1)^2 = 2 * S n + n + 4)
  (h₂ : ∃ q, a 2 - 1 = a 3 / q ∧ a 7 = a 3 * q^4) :
  (∀ n, a n = n + 1) ∧ 
  (∀ n, b n = 2^n) ∧ 
  (∀ n, ∑ i in range (n+1), c i = T n) := by 
  sorry

end sequence_properties_l396_396872


namespace difficult_vs_easy_problems_l396_396613

-- Defining the conditions given in the problem
variables (x1 x2 x3 y12 y13 y23 z : ℕ)

-- Define the hypotheses based on the problem conditions
def hypothesis1 : Prop := x1 + x2 + x3 + y12 + y13 + y23 + z = 100
def hypothesis2 : Prop := x1 + y12 + y13 + z = 60
def hypothesis3 : Prop := x2 + y12 + y23 + z = 60
def hypothesis4 : Prop := x3 + y13 + y23 + z = 60

-- The statement to prove
theorem difficult_vs_easy_problems
  (h1 : hypothesis1)
  (h2 : hypothesis2)
  (h3 : hypothesis3)
  (h4 : hypothesis4) :
  x1 + x2 + x3 - z = 20 :=
sorry

end difficult_vs_easy_problems_l396_396613


namespace binomial_7_2_eq_21_l396_396041

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396041


namespace megan_operations_reach_below_five_l396_396695

def operation (n : Nat) : Nat :=
  (3 * n) / 2

theorem megan_operations_reach_below_five :
  let rec apply_operations (count : Nat) (current : Nat) : Nat :=
    if current < 5 then count
    else apply_operations (count + 1) (operation current)
  in apply_operations 0 243 = 20 :=
sorry

end megan_operations_reach_below_five_l396_396695


namespace berengere_contribution_is_zero_l396_396017

theorem berengere_contribution_is_zero
  (cost_of_pastry : ℝ)
  (amount_in_dollars : ℝ)
  (exchange_rate : ℝ)
  (emily_has_enough : (amount_in_dollars / exchange_rate) ≥ cost_of_pastry) :
  ∃ berengere_contribution : ℝ, berengere_contribution = 0 := 
by 
  have emily_euros := amount_in_dollars / exchange_rate
  have no_need_for_berengere := emily_euros ≥ cost_of_pastry
  use 0
  sorry

-- Define the conditions
def condition_1 := cost_of_pastry = 8
def condition_2 := amount_in_dollars = 10
def condition_3 := exchange_rate = 1.2

-- Plug in conditions to the theorem
example : berengere_contribution_is_zero 8 10 1.2
  (by { rw [div_eq_mul_inv, div_eq_mul_inv], norm_num, exact le_of_lt }) :=
by sorry

end berengere_contribution_is_zero_l396_396017


namespace num_three_digit_perfect_cubes_divisible_by_16_l396_396577

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l396_396577


namespace sequence_5th_term_l396_396609

theorem sequence_5th_term (a b c : ℚ) (h1 : a = 1 / 4 * (4 + b)) (h2 : b = 1 / 4 * (a + 40)) (h3 : 40 = 1 / 4 * (b + c)) : 
  c = 2236 / 15 := 
by 
  sorry

end sequence_5th_term_l396_396609


namespace max_modulus_z_l396_396661

noncomputable def max_modulus_of_z : ℂ := 4

theorem max_modulus_z (z : ℂ) (hz : complex.abs z = 1) : 
  ∃ θ : ℝ, complex.abs (z + 2 * real.sqrt 2 + complex.I) = max_modulus_of_z := sorry 

end max_modulus_z_l396_396661


namespace highest_power_10_and_sum_powers_4_6_l396_396834

theorem highest_power_10_and_sum_powers_4_6 (n : ℕ) (h : n = 20) : 
  (∃ k : ℕ, k = highest_power_of_base_in_factorial 10 n ∧ k = 4) ∧
  (∃ l : ℕ, l = highest_power_of_base_in_factorial 4 n ∧ ∃ m : ℕ, m = highest_power_of_base_in_factorial 6 n ∧ l + m = 17) :=
by 
  sorry

end highest_power_10_and_sum_powers_4_6_l396_396834


namespace largest_fraction_l396_396757

theorem largest_fraction :
  let f1 := (2 : ℚ) / 3
  let f2 := (3 : ℚ) / 4
  let f3 := (2 : ℚ) / 5
  let f4 := (11 : ℚ) / 15
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 :=
by
  sorry

end largest_fraction_l396_396757


namespace total_cookies_l396_396694

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1: num_people = 4) (h2: cookies_per_person = 22) : total_cookies = 88 :=
by
  sorry

end total_cookies_l396_396694


namespace combination_7_2_l396_396028

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396028


namespace projection_divides_same_ratio_l396_396675

variables {R : Type*} [linear_ordered_field R]

/-- Define the points on line 'l' and their projections on line 'm'. --/
variables (A B C A₁ B₁ C₁ : R)

/-- Define the projections to meet on the line 'm'. --/
def projection_on_line_m (l m : set (R × R)) (A B C A₁ B₁ C₁ : R × R) : Prop :=
  ∃ (proj : (R × R) → (R × R)),
    (proj A = A₁ ∧ proj B = B₁ ∧ proj C = C₁) ∧
    (proj l ⊆ m ∧ ∀ p ∈ l, proj p ∈ m)

/-- Define the ratio of segments. --/
def divides_in_ratio (P Q R: R) (ratio: ℚ) :=
  ∃ (k: R), P + k * ratio = Q ∧ Q + k * (1 - ratio) = R

theorem projection_divides_same_ratio (hline_l : set (R × R)) (hline_m : set (R × R))
  (hproj : projection_on_line_m hline_l hline_m (A, 0) (B, 0) (C, 0) (A₁, 1) (B₁, 1) (C₁, 1))
  (hdiv : divides_in_ratio (0: R) (B: R) (C: R) (2/5)) :
  divides_in_ratio (A₁: R) (B₁: R) (C₁: R) (2/5) :=
sorry

end projection_divides_same_ratio_l396_396675


namespace probability_one_from_each_l396_396010

theorem probability_one_from_each (cards : Finset (Fin 9)) 
  (alex tommy : Finset (Fin 9)) 
  (h_card_count : cards.card = 9)
  (h_alex : alex.card = 4)
  (h_tommy : tommy.card = 5)
  (h_alex_union_tommy : alex ∪ tommy = cards)
  (h_disjoint : Disjoint alex tommy) :
  probability (λ (x : Fin 9 × Fin 9), x.1 ∈ alex ∧ x.2 ∈ tommy ∨ x.1 ∈ tommy ∧ x.2 ∈ alex) 
    (Finset.powersetLen 2 cards).toFinset = 5 / 9 :=
by
  sorry

end probability_one_from_each_l396_396010


namespace solution_set_l396_396536

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (hf1 : f 1 = 1)
variable (hf' : ∀ x, (f x)' < 1 / 2)

-- Define the final property to prove
theorem solution_set :
  { x : ℝ | f (x^2) < x^2 / 2 + 1 / 2 } = { x : ℝ | x < -1 ∨ x > 1 } :=
sorry

end solution_set_l396_396536


namespace combinatorial_identity_l396_396143

theorem combinatorial_identity :
  (nat.choose 22 5) = 26334 := by
  have h1 : nat.choose 20 3 = 1140 := sorry
  have h2 : nat.choose 20 4 = 4845 := sorry
  have h3 : nat.choose 20 5 = 15504 := sorry
  sorry

end combinatorial_identity_l396_396143


namespace number_of_subsets_containing_7_l396_396571

-- Define the set
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the subset condition
def contains_7 (A : Set ℕ) : Prop := 7 ∈ A

-- Define the count of subsets containing 7
def count_subsets_containing_7 : ℕ := 
  (Set.powerset S).filter contains_7).card

-- The theorem statement
theorem number_of_subsets_containing_7 : count_subsets_containing_7 = 64 := by
  sorry

end number_of_subsets_containing_7_l396_396571


namespace width_of_room_l396_396724

theorem width_of_room (length : ℝ) (total_cost : ℝ) (rate : ℝ) (width : ℝ) : 
  length = 5.5 ∧ total_cost = 12375 ∧ rate = 600 ∧ width = 3.75 → 
  width * length = total_cost / rate :=
by
  intros h
  cases h with h_len h_rest
  cases h_rest with h_tot_cost h_rest
  cases h_rest with h_rate h_width
  rw [h_len, h_tot_cost, h_rate, h_width]
  norm_num
  sorry

end width_of_room_l396_396724


namespace ratio_problem_l396_396944

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396944


namespace Marc_watch_episodes_l396_396268

theorem Marc_watch_episodes : ∀ (episodes per_day : ℕ), episodes = 50 → per_day = episodes / 10 → (episodes / per_day) = 10 :=
by
  intros episodes per_day h1 h2
  sorry

end Marc_watch_episodes_l396_396268


namespace values_for_a_distances_for_a_eq_15_l396_396771

noncomputable def distances (a : ℝ) (h : 0 < a ∧ a < 100) : ℝ × ℝ × ℝ :=
let x := (225 - a) / 5
let y := (75 + 3a) / 5
let z := 30 + a / 5
in (x, y, z)

theorem values_for_a (a : ℝ) (h : 0 < a ∧ a < 100) :
  let (x, y, z) := distances a h
  in x + y = 2 * z ∧ z + y = x + a ∧ x + z = 75 :=
by
  sorry

theorem distances_for_a_eq_15 :
  distances 15 (by norm_num) = (42, 24, 33) :=
by
  sorry

end values_for_a_distances_for_a_eq_15_l396_396771


namespace weighted_avg_sales_increase_l396_396416

section SalesIncrease

/-- Define the weightages for each category last year. -/
def w_e : ℝ := 0.4
def w_c : ℝ := 0.3
def w_g : ℝ := 0.3

/-- Define the percent increases for each category this year. -/
def p_e : ℝ := 0.15
def p_c : ℝ := 0.25
def p_g : ℝ := 0.35

/-- Prove that the weighted average percent increase in sales this year is 0.24 or 24%. -/
theorem weighted_avg_sales_increase :
  ((w_e * p_e) + (w_c * p_c) + (w_g * p_g)) / (w_e + w_c + w_g) = 0.24 := 
by
  sorry

end SalesIncrease

end weighted_avg_sales_increase_l396_396416


namespace part1_part2_l396_396547

def f (x : ℝ) : ℝ := x^2 - 1

theorem part1 (m x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (ineq : 4 * m^2 * |f x| + 4 * f m ≤ |f (x-1)|) : 
    -1/2 ≤ m ∧ m ≤ 1/2 := 
sorry

theorem part2 (x1 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) : 
    (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = |2 * f x2 - a * x2|) →
    (0 ≤ a ∧ a ≤ 3/2 ∨ a = 3) := 
sorry

end part1_part2_l396_396547


namespace hyperbola_problem_l396_396660

theorem hyperbola_problem 
  (x y m : ℝ)
  (P F1 F2 : ℝ × ℝ)
  (hyperbola_eq : ∃ x y, x^2 / 9 - y^2 / m = 1)
  (perpendicular_cond : PF1 • PF2 = 0)
  (directrix_cond : ∃ y, y^2 = 16 * x ∧ passes_through_focus) :
  |P - F1| * |P - F2| = 14 :=
by
  sorry

end hyperbola_problem_l396_396660


namespace integral_eval_l396_396490

open Real

noncomputable def integral_problem : Prop :=
  integral (λ x : ℝ, x^2 * sin x + sqrt (4 - x^2)) (-2) 2 = 2 * π

theorem integral_eval : integral_problem := 
  by sorry

end integral_eval_l396_396490


namespace charles_dickens_born_on_l396_396700

noncomputable def day_of_the_week_of_birth : String := 
let leap_years := 36
let regular_years := 114
let days_backward := regular_years * 1 + leap_years * 2
let days_mod := days_backward % 7
if days_mod = 0 then "Monday"
else if days_mod = 1 then "Sunday"
else if days_mod = 2 then "Saturday"
else if days_mod = 3 then "Friday"
else if days_mod = 4 then "Thursday"
else if days_mod = 5 then "Wednesday"
else "Tuesday"

theorem charles_dickens_born_on : day_of_the_week_of_birth = "Wednesday" := 
sorry

end charles_dickens_born_on_l396_396700


namespace train_cross_time_after_detachment_l396_396419

-- We specify the conditions as Lean definitions
def number_of_boggies_initial := 12
def length_of_each_boggy := 15 -- in meters
def time_to_cross_initial := 18 -- in seconds
def number_of_boggies_detached := 1

-- Calculate the total length of the train initially
def total_length_initial := number_of_boggies_initial * length_of_each_boggy

-- Calculate the speed of the train
def speed_of_train := total_length_initial / time_to_cross_initial

-- Calculate the new number of boggies after detachment
def number_of_boggies_final := number_of_boggies_initial - number_of_boggies_detached

-- Calculate the new total length of the train
def total_length_final := number_of_boggies_final * length_of_each_boggy

-- Calculate the new time to cross the telegraph post
def new_time_to_cross := total_length_final / speed_of_train

-- The problem is to prove that the new time to cross the telegraph post is 16.5 seconds
theorem train_cross_time_after_detachment :
  new_time_to_cross = 16.5 :=
by
  sorry

end train_cross_time_after_detachment_l396_396419


namespace num_three_digit_perfect_cubes_divisible_by_16_l396_396579

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l396_396579


namespace frosting_cupcakes_l396_396451

noncomputable def rate_cagney := 1 / 25  -- Cagney's rate in cupcakes per second
noncomputable def rate_lacey := 1 / 20  -- Lacey's rate in cupcakes per second

noncomputable def break_time := 30      -- Break time in seconds
noncomputable def work_period := 180    -- Work period in seconds before a break
noncomputable def total_time := 600     -- Total time in seconds (10 minutes)

noncomputable def combined_rate := rate_cagney + rate_lacey -- Combined rate in cupcakes per second

-- Effective work time after considering breaks
noncomputable def effective_work_time :=
  total_time - (total_time / work_period) * break_time

-- Total number of cupcakes frosted in the effective work time
noncomputable def total_cupcakes := combined_rate * effective_work_time

theorem frosting_cupcakes : total_cupcakes = 48 :=
by
  sorry

end frosting_cupcakes_l396_396451


namespace meeting_equation_correct_l396_396379

-- Define the conditions
def distance : ℝ := 25
def time : ℝ := 3
def speed_Xiaoming : ℝ := 4
def speed_Xiaogang (x : ℝ) : ℝ := x

-- The target equation derived from conditions which we need to prove valid.
theorem meeting_equation_correct (x : ℝ) : 3 * (speed_Xiaoming + speed_Xiaogang x) = distance :=
by
  sorry

end meeting_equation_correct_l396_396379


namespace can_place_28_dominoes_l396_396401

theorem can_place_28_dominoes (total_pieces : ℕ) (total_squares : ℕ) (board : matrix (fin 8) (fin 8) ℕ) :
  total_pieces = 28 ∧ total_squares = 64 ∧ (∀ i j, board i j ∈ {0, 1}) ∧
  ∀ (placement : (fin 8) × (fin 8) → option (fin 28)) (piece_covered: (fin 28) → fin 8 × fin 8 × fin 8 × fin 8),
    (∀ p, ∃! (i : (fin 8) × (fin 8)), (placement i = some p ∧ ∃ (r1 c1 r2 c2: fin 8), piece_covered p = (r1, c1, r2, c2) ∧ placement (r1, c1) = some p ∧ placement (r2, c2) = some p)) →
    (∀ (p1 p2 : fin 28), p1 ≠ p2 → (∀ (x1 x2 x3 x4 : fin 8), (piece_covered p1 = (x1, x2, x3, x4)) → ∀ (y1 y2 y3 y4 : fin 8), (piece_covered p2 = (y1, y2, y3, y4)) → (x1, x2) ≠ (y1, y2) ∧ (x3, x4) ≠ (y3, y4))) →
    ∃ (placement_strategy: matrix (fin 8) (fin 8) (option (fin 28))), 
      (∀ (i j), (placement_strategy i j = some p → board i j = 1) ∧ (placement_strategy i j = none → board i j = 0)) :=
sorry

end can_place_28_dominoes_l396_396401


namespace spider_total_distance_l396_396415

theorem spider_total_distance :
  let a := -3
  let b := -8
  let c := 0
  let d := 7
  let dist_ab := abs (b - a)
  let dist_bc := abs (c - b)
  let dist_cd := abs (d - c)
  dist_ab + dist_bc + dist_cd = 20 := by
  let a := -3
  let b := -8
  let c := 0
  let d := 7
  let dist_ab := abs (b - a)
  let dist_bc := abs (c - b)
  let dist_cd := abs (d - c)
  sorry

end spider_total_distance_l396_396415


namespace Arthur_total_distance_l396_396443

/-- Arthur walks 8 blocks south and then 10 blocks west. Each block is one-fourth of a mile.
How many miles did Arthur walk in total? -/
theorem Arthur_total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length_miles : ℝ) :
  blocks_south = 8 ∧ blocks_west = 10 ∧ block_length_miles = 1/4 →
  (blocks_south + blocks_west) * block_length_miles = 4.5 :=
by
  intro h
  have h1 : blocks_south = 8 := h.1
  have h2 : blocks_west = 10 := h.2.1
  have h3 : block_length_miles = 1 / 4 := h.2.2
  sorry

end Arthur_total_distance_l396_396443


namespace chriss_fishing_times_l396_396816

def fish_problem (C : ℕ) : Prop :=
  let BrianFishingTwice : C*2 = 2*C,
      BrianFishPerTrip : 400,
      ChrisFishPerTrip : (7/5:ℝ) * BrianFishPerTrip,
      TotalFish : BrianFishingTwice * BrianFishPerTrip + C * ChrisFishPerTrip = 13600 in
  C

theorem chriss_fishing_times (C : ℕ) (h : fish_problem C): C = 10 := by
  sorry

end chriss_fishing_times_l396_396816


namespace prove_one_correct_conclusion_l396_396140

-- Define the polynomials M and N
def M (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 2
def N (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the algebraic expressions for conclusions
def expression1 (x : ℝ) : ℝ := 13 * x / (x^2 - 3 * x - 1)
def expression2 (x : ℝ) (a : ℝ) : ℝ := M x - N x a
def expression3 (x : ℝ) (a : ℝ) : ℝ := M x * N x a

theorem prove_one_correct_conclusion :
  (M 2 = 0 → expression1 2 = 26 / 3) →
  (M (-1/2) = 0 → expression1 (-1/2) = 26 / 3) →
  (∀ a, a = -3 → x ≥ 4 → expression2 x a < -14) →
  (∀ a, a = 0 → ∃ x1 x2 : ℝ, expression3 x1 a = 0 ∧ expression3 x2 a = 0 ∧ x1 ≠ x2) →
  (1 = 1) :=
by
  sorry

end prove_one_correct_conclusion_l396_396140


namespace compute_value_of_expression_l396_396070

noncomputable def sum_of_fractions : ℚ :=
  (∑ i in Finset.range 20, (i + 1) / 7) - (∑ i in Finset.range 10, (i + 1) / 3)

theorem compute_value_of_expression : sum_of_fractions = 35 / 3 := sorry

end compute_value_of_expression_l396_396070


namespace probability_greater_than_30_l396_396681

theorem probability_greater_than_30 :
  let digits := {1, 2, 3, 4, 5} in
  let total_events := (Finset.unorderedPairs digits).card in
  let favorable_events := (Finset.unorderedPairs digits).filter (λ (x : ℕ × ℕ), 10 * x.1 + x.2 > 30 ∨ 10 * x.2 + x.1 > 30) in
  favorable_events.card / total_events = 3 / 5 :=
by
  let digits := {1, 2, 3, 4, 5}
  let total_events := (Finset.unorderedPairs digits).card
  let favorable_events := (Finset.unorderedPairs digits).filter (λ (x : ℕ × ℕ), 10 * x.1 + x.2 > 30 ∨ 10 * x.2 + x.1 > 30)
  have h_card : favorable_events.card = 12 := sorry
  have h_total : total_events = 20 := sorry
  calc
    favorable_events.card / total_events
        = 12 / 20 : by rw [h_card, h_total]
    ... = 3 / 5 : by norm_num_ratio

end probability_greater_than_30_l396_396681


namespace ratio_of_areas_l396_396479

theorem ratio_of_areas (s : ℝ) (n : ℕ) (S : ℝ) (N : ℕ) 
  (h₀ : s = 2) (h₁ : n = 8) 
  (h₂ : N = 1) 
  (h₃ : S = n * s) 
  (h₄ : S = 3 * sqrt 3 * (S / 3)^2) 
  : (n * (sqrt 3 / 4) * s^2) / ((sqrt 3 / 4) * (S / 3)^2) = 1 / 8 :=
sorry

end ratio_of_areas_l396_396479


namespace no_valid_three_digit_number_with_digit_sum_27_l396_396175

def is_valid_digit (d : ℕ) : Prop := d < 10

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ := 
  let d2 := n % 10 in
  let d1 := (n / 10) % 10 in
  let d0 := n / 100 in
  d0 + d1 + d2

def valid_three_digit_numbers (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  digit_sum n = 27 ∧
  is_even ((n / 10) % 10) ∧
  is_even (n % 10)

theorem no_valid_three_digit_number_with_digit_sum_27 (n : ℕ) :
  valid_three_digit_numbers n → false :=
by sorry

end no_valid_three_digit_number_with_digit_sum_27_l396_396175


namespace savings_correct_l396_396634

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l396_396634


namespace intersection_M_N_l396_396261

noncomputable def M := {x : ℕ | x < 6}
noncomputable def N := {x : ℕ | x^2 - 11 * x + 18 < 0}
noncomputable def intersection := {x : ℕ | x ∈ M ∧ x ∈ N}

theorem intersection_M_N : intersection = {3, 4, 5} := by
  sorry

end intersection_M_N_l396_396261


namespace compare_expressions_l396_396463

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l396_396463


namespace EquivalentTrapezoid_l396_396231

open EuclideanGeometry

variables {P : Type*} [MetricSpace P] [NormedAddTorsor ℝ P]

-- Assumptions and conditions
variables (A B C D A1 B1 C1 D1 : P)
variable (circumcircleABC : Circle)
variable (circumcircleBCD : Circle)

-- Define the non-isosceles trapezoid ABCD
noncomputable def is_non_isosceles_trapezoid  (A B C D : P) : Prop :=
Trapezoid A B C D ∧ ¬(IsoscelesTrapezoid A B C D)

-- Define A1 as the intersection of circumcircle of triangle BCD and line AC
noncomputable def A1_def (A C A1 : P) (circumcircleBCD : Circle) : Prop :=
A1 ∈ circumcircleBCD ∧ Line A C ∧ A1 ≠ C

-- Define B1, C1, D1 similarly
noncomputable def B1_def (B D B1 : P) (circumcircleCDA : Circle) : Prop :=
B1 ∈ circumcircleCDA ∧ Line B D ∧ B1 ≠ D

noncomputable def C1_def (C A C1 : P) (circumcircleDAB : Circle) : Prop :=
C1 ∈ circumcircleDAB ∧ Line C A ∧ C1 ≠ A

noncomputable def D1_def (D B D1 : P) (circumcircleABC : Circle) : Prop :=
D1 ∈ circumcircleABC ∧ Line D B ∧ D1 ≠ B

theorem EquivalentTrapezoid 
  (h1 : is_non_isosceles_trapezoid A B C D)
  (h2 : A1_def A C A1 circumcircleBCD)
  (h3 : B1_def B D B1 circumcircleCDA)
  (h4 : C1_def C A C1 circumcircleDAB)
  (h5 : D1_def D B D1 circumcircleABC) : 
  A1B1C1D1
:=
begin
  sorry
end 

end EquivalentTrapezoid_l396_396231


namespace triangle_area_side_l396_396195

theorem triangle_area_side (A : ℝ) (b : ℝ) (area : ℝ) (hA : A = 60) (hb : b = 1) (harea : area = (Real.sqrt 3 / 4)) :
  ∃ (a : ℝ), a = 3 :=
by
  use 3
  sorry

end triangle_area_side_l396_396195


namespace ab_cd_l396_396765

theorem ab_cd {a b c d : ℕ} {w x y z : ℕ}
  (hw : Prime w) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (horder : w < x ∧ x < y ∧ y < z)
  (hprod : w^a * x^b * y^c * z^d = 660) :
  (a + b) - (c + d) = 1 :=
by
  sorry

end ab_cd_l396_396765


namespace part1_part2_part3_l396_396682

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- 1. Prove that B ⊆ A implies m ≤ 3
theorem part1 (m : ℝ) : B m ⊆ A → m ≤ 3 := sorry

-- 2. Prove that the number of non-empty proper subsets of A when x ∈ ℤ is 254
theorem part2 : ∀ x ∈ (Set.univ : Set ℤ), x ∈ {-2, -1, 0, 1, 2, 3, 4, 5} → ∃! n, n = 254 := sorry

-- 3. Prove that A ∩ B = ∅ implies m < 2 or m > 4
theorem part3 (m : ℝ) : A ∩ B m = ∅ → m < 2 ∨ m > 4 := sorry

end part1_part2_part3_l396_396682


namespace winning_candidate_percentage_l396_396776

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 7636) (h3 : v3 = 11628) :
  ((v3: ℝ) / (v1 + v2 + v3)) * 100 = 57 := by
  sorry

end winning_candidate_percentage_l396_396776


namespace circumcenter_perpendicular_l396_396620

open EuclideanGeometry

-- Define the given problem in Lean 4
theorem circumcenter_perpendicular {A B O C D E F M N K P : Point} :
  (∃ (circleO : Circle O) 
     (diameterAB : Diameter A B circleO) 
     (chordCD : Chord C D circleO) 
     (perpAB_CD : Perpendicular AB CD)
     (pointE : LiesOn E circleO)
     (intersectionF : IntersectsAt CE AB F)
     (perpendicularFM : Perpendicular FM AD)
     (intersectsM : IntersectsAt FM AD M)
     (perpendicularFN : Perpendicular FN AE)
     (intersectsN : IntersectsAt FN AE N)
     (perpendicularBP : Perpendicular BP AB)
     (intersectsP : IntersectsAt BP DE P)
     (circumcenterK : Circumcenter K \triangle AMN)),
  Perpendicular KF FP :=
sorry

end circumcenter_perpendicular_l396_396620


namespace num_three_digit_perfect_cubes_divisible_by_16_l396_396578

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l396_396578


namespace katie_earnings_l396_396388

-- Define the constants for the problem
def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

-- Define the total earnings calculation
def total_necklaces : Nat := bead_necklaces + gemstone_necklaces
def total_earnings : Nat := total_necklaces * cost_per_necklace

-- Statement of the proof problem
theorem katie_earnings : total_earnings = 21 := by
  sorry

end katie_earnings_l396_396388


namespace shaded_shape_area_l396_396750

/-- Define the coordinates and the conditions for the central square and triangles in the grid -/
def grid_size := 10
def central_square_side := 2
def central_square_area := central_square_side * central_square_side

def triangle_base := 5
def triangle_height := 5
def triangle_area := (1 / 2) * triangle_base * triangle_height

def number_of_triangles := 4
def total_triangle_area := number_of_triangles * triangle_area

def total_shaded_area := total_triangle_area + central_square_area

theorem shaded_shape_area : total_shaded_area = 54 :=
by
  -- We have defined each area component and summed them to the total shaded area.
  -- The statement ensures that the area of the shaded shape is equal to 54.
  sorry

end shaded_shape_area_l396_396750


namespace degree_of_P_l396_396366

-- Define the polynomial
def P : Polynomial ℝ := 3 * X^6 + 7 + 15 * X^5 + 22 * X^3 - 2 * Real.pi * X^6 + Real.sqrt 5 * X^2 - 11

-- Formulate the theorem for the degree of the polynomial
theorem degree_of_P : P.degree = 6 := 
by 
  -- need to proof here
  sorry

end degree_of_P_l396_396366


namespace equal_intercepts_line_eq_l396_396712

theorem equal_intercepts_line_eq (P : ℝ × ℝ) (eq_intercepts : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y : ℝ, x + y = a ∧ (2, 3) ∈ P)):
  ∃ L : ℝ × ℝ → Prop, (∀ (x y : ℝ), L (x, y) ↔ x + y - 5 = 0 ∨ 3 * x - 2 * y = 0) :=
sorry

end equal_intercepts_line_eq_l396_396712


namespace telephone_number_problem_l396_396418

theorem telephone_number_problem
  (digits : Finset ℕ)
  (A B C D E F G H I J : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_distinct : [A, B, C, D, E, F, G, H, I, J].Nodup)
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_DEF_consecutive_odd : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
  (h_GHIJ_consecutive_even : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
  (h_sum_ABC : A + B + C = 15) :
  A = 9 :=
by
  sorry

end telephone_number_problem_l396_396418


namespace area_ratio_l396_396248

noncomputable def ratio_area_triangles (A1 A2 A3 : Point) : ℚ :=
  let A4 := A1
  let A5 := A2
  let B1 := midpoint A1 A2
  let B2 := midpoint A2 A3
  let B3 := midpoint A3 A4
  let C1 := midpoint A1 B1
  let C2 := midpoint A2 B2
  let C3 := midpoint A3 B3
  
  let D1 := intersection (line_through A1 (C2)) (line_through B1 (A3))
  let D2 := intersection (line_through A2 (C3)) (line_through B2 (A4))
  let D3 := intersection (line_through A3 (C4)) (line_through B3 (A5))
  
  let E1 := intersection (line_through A1 (B2)) (line_through C1 (A3))
  let E2 := intersection (line_through A2 (B3)) (line_through C2 (A4))
  let E3 := intersection (line_through A3 (B4)) (line_through C3 (A5))

  let area_triangle (P Q R : Point) : ℚ := sorry
  
  (area_triangle D1 D2 D3) / (area_triangle E1 E2 E3)

theorem area_ratio (A1 A2 A3 : Point) :
  ratio_area_triangles A1 A2 A3 = 25 / 49 :=
by
  sorry

end area_ratio_l396_396248


namespace remainder_13_pow_2000_mod_1000_l396_396372

theorem remainder_13_pow_2000_mod_1000 :
  (13^2000) % 1000 = 1 := 
by 
  sorry

end remainder_13_pow_2000_mod_1000_l396_396372


namespace tita_can_hop_only_finitely_l396_396740

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then finset.max' {p : ℕ | p ∣ n ∧ is_prime p} (finset.nonempty_of_ne_empty h)
  else 1
def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then finset.min' {p : ℕ | p ∣ n ∧ is_prime p} (finset.nonempty_of_ne_empty h)
  else 1

def hop (n : ℕ) : ℕ := largest_prime_factor n + smallest_prime_factor n

theorem tita_can_hop_only_finitely (k : ℕ) (hk: k > 12) : 
  ∃ N : ℕ, ∀ n ≥ N, ¬ (∃ m : ℕ, hop m = n) :=
sorry

end tita_can_hop_only_finitely_l396_396740


namespace van_distance_covered_l396_396804

noncomputable def distance_covered (V : ℝ) := 
  let D := V * 6
  D

theorem van_distance_covered : ∃ (D : ℝ), ∀ (V : ℝ), 
  (D = 288) ∧ (D = distance_covered V) ∧ (D = 32 * 9) :=
by
  sorry

end van_distance_covered_l396_396804


namespace sam_age_l396_396295

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l396_396295


namespace tan_five_pi_over_four_eq_one_l396_396069

theorem tan_five_pi_over_four_eq_one : Real.tan (5 * Real.pi / 4) = 1 :=
by sorry

end tan_five_pi_over_four_eq_one_l396_396069


namespace jogging_track_circumference_l396_396721

noncomputable def suresh_speed_km_hr : ℝ := 4.5
noncomputable def wife_speed_km_hr : ℝ := 3.75
noncomputable def meeting_time_min : ℝ := 5.28

def suresh_speed_km_min : ℝ := suresh_speed_km_hr / 60
def wife_speed_km_min : ℝ := wife_speed_km_hr / 60

def total_distance_km : ℝ := (suresh_speed_km_min + wife_speed_km_min) * meeting_time_min

theorem jogging_track_circumference :
  total_distance_km = 0.7257 :=
by
  sorry

end jogging_track_circumference_l396_396721


namespace cubic_roots_inequality_l396_396237

theorem cubic_roots_inequality (a b c : ℝ) (h : ∃ (α β γ : ℝ), (x : ℝ) → x^3 + a * x^2 + b * x + c = (x - α) * (x - β) * (x - γ)) :
  3 * b ≤ a^2 :=
sorry

end cubic_roots_inequality_l396_396237


namespace log2_6_gt_2_sqrt_5_l396_396461

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l396_396461


namespace probability_hits_10_ring_l396_396794

-- Definitions based on conditions
def total_shots : ℕ := 10
def hits_10_ring : ℕ := 2

-- Theorem stating the question and answer equivalence.
theorem probability_hits_10_ring : (hits_10_ring : ℚ) / total_shots = 0.2 := by
  -- We are skipping the proof with 'sorry'
  sorry

end probability_hits_10_ring_l396_396794


namespace shortest_tangent_segment_length_l396_396991

-- Define the circles C1 and C2
def C1 (x y : ℝ) := (x - 12)^2 + y^2 = 64
def C2 (x y : ℝ) := (x + 18)^2 + y^2 = 100

-- Define the centers and radii
def center_C1 := (12 : ℝ, 0 : ℝ)
def radius_C1 := 8
def center_C2 := (-18 : ℝ, 0 : ℝ)
def radius_C2 := 10

-- Define the length AB
def length_AB := 30

-- Define the point D that divides AB in the ratio 4:5
def point_D := (-4/3 : ℝ, 0 : ℝ)

-- Define the distances PD and QD
def distance_PD := 40 / 3
def distance_QD := 50 / 3

-- Define the length of the shortest line segment PQ
def length_PQ := 190 / 3

-- The final theorem statement
theorem shortest_tangent_segment_length :
  ∃ PQ : ℝ, PQ = 190 / 3 ∧
  (∀ (P Q : ℝ × ℝ), 
    C1 P.1 P.2 → C2 Q.1 Q.2 → 
    segment PQ ∧ tangent_to C1 P ∧ tangent_to C2 Q → length PQ = 190 / 3) :=
sorry

end shortest_tangent_segment_length_l396_396991


namespace avg_weight_A_and_B_l396_396704

-- Definitions for the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- Statement to be proved
theorem avg_weight_A_and_B : condition1 A B C ∧ condition2 A B C ∧ condition3 B → (A + B) / 2 = 40 :=
begin
  intros h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2,
  sorry -- Proof goes here
end

end avg_weight_A_and_B_l396_396704


namespace find_x_interval_l396_396089

theorem find_x_interval (x : ℝ) : 
  (x + 1) / (x + 3) ≤ 3 ↔ x ∈ set.Ico (-4 : ℝ) (-3 : ℝ) :=
by
  sorry

end find_x_interval_l396_396089


namespace perfect_squares_less_than_5000_have_ones_digit_5_6_or_7_l396_396563

theorem perfect_squares_less_than_5000_have_ones_digit_5_6_or_7 {n : ℕ} :
  (finset.filter (λ x : ℕ, x < 5000) (finset.image (λ x, x * x) (finset.range 71))).card = 22 :=
sorry

end perfect_squares_less_than_5000_have_ones_digit_5_6_or_7_l396_396563


namespace ratio_expression_value_l396_396931

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396931


namespace part1_part2_l396_396164

noncomputable def f (a m : ℝ) (x : ℝ) := a^x + (1-m) * a^(-x)
noncomputable def g (a t : ℝ) (x : ℝ) := Real.log t (2^(2*x) + 2^(-2*x) - t * (a^x - a^(-x)))

theorem part1 (a m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a m 0 = 0) (h4 : f a m (-1) = -3 / 2) :
  m = 2 ∧ a = 2 :=
  sorry

theorem part2 (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1)
    (h3 : ∀ x, 1 ≤ x ∧ x ≤ Real.log 2 3 → g 2 t x ≤ 0) :
    t ∈ Ioo 0 1 :=
  sorry

end part1_part2_l396_396164


namespace max_real_part_of_z_plus_w_l396_396310

noncomputable theory
open Complex

-- Definitions based on the conditions
variables {z w : ℂ}
variables {a b c d : ℝ}

example
  (hz : abs z = 3)
  (hw : abs w = 2)
  (hc : z * conj w + conj z * w = 2) :
  ℝ :=
begin
  -- Meet the specific conditions as described without solving.
  sorry
end

theorem max_real_part_of_z_plus_w 
  (hz : abs z = 3)
  (hw : abs w = 2)
  (hc : z * conj w + conj z * w = 2) :
  ∃ (a b c d : ℝ), (z = 3 * (a + b * I)) ∧ (w = 2 * (c + d * I)) ∧ (3 * a + 2 * c = sqrt 13) :=
sorry

end max_real_part_of_z_plus_w_l396_396310


namespace midpoint_translation_l396_396680

theorem midpoint_translation (A B : ℝ × ℝ) (C D : ℝ) :
  A = (5, 3) →
  B = (-9, 9) →
  C = 1 →
  D = 2 →
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let M2 := (M1.1 + 3, M1.2 - 4) in
  M2 = (C, D) :=
by
  intros
  sorry

end midpoint_translation_l396_396680


namespace calculate_expression_l396_396454

theorem calculate_expression :
  (Real.sqrt 2 - 3)^0 - Real.sqrt 9 + |(-2: ℝ)| + ((-1/3: ℝ)⁻¹)^2 = 9 :=
by
  sorry

end calculate_expression_l396_396454


namespace inequality_proof_l396_396250

variables {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
  (↑((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) ^ (1 / 3 : ℝ)) ≤ ((a + b + c) / 3) ^ 2 + 1 :=
by
  sorry

end inequality_proof_l396_396250


namespace trigonometric_identity_l396_396120

theorem trigonometric_identity (α : ℝ) (h1 : sin(α + π / 4) = 3 / 5) (h2 : π / 4 < α ∧ α < 3 * π / 4) :
  cos(α) = -√2 / 10 :=
  sorry

end trigonometric_identity_l396_396120


namespace worker_b_time_l396_396378

theorem worker_b_time (T_B : ℝ) : 
  (1 / 10) + (1 / T_B) = 1 / 6 → T_B = 15 := by
  intro h
  sorry

end worker_b_time_l396_396378


namespace num_subsets_containing_7_l396_396566

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem num_subsets_containing_7 : (S.filter (λ s => 7 ∈ s)).card = 64 := by
  sorry

end num_subsets_containing_7_l396_396566


namespace choose_seven_starters_with_quadruplets_l396_396672

theorem choose_seven_starters_with_quadruplets :
  let total_players := 18
  let quadruplets := 4
  let remaining_players := total_players - quadruplets
  let starters := 7
  let additional_starters := starters - quadruplets
  combinatorics.choose remaining_players additional_starters = 364 :=
by
  have total_players := 18
  have quadruplets := 4
  have remaining_players := total_players - quadruplets
  have starters := 7
  have additional_starters := starters - quadruplets
  calc
  combinatorics.choose remaining_players additional_starters = 364 : sorry

end choose_seven_starters_with_quadruplets_l396_396672


namespace find_prices_possible_plans_cost_effective_plan_l396_396227

-- Part (1): Finding prices of A and B type devices
theorem find_prices (x y : ℝ) (h1 : 3 * x - 2 * y = 16) (h2 : 2 * x + 6 = 3 * y) : x = 12 ∧ y = 10 :=
sorry

-- Part (2): Possible purchasing plans within the budget
theorem possible_plans (m : ℕ) (h1 : 12 * m + 10 * (10 - m) ≤ 110) : m ∈ {0, 1, 2, 3, 4, 5} :=
sorry

-- Part (3): Most cost-effective purchasing plan
theorem cost_effective_plan (m : ℕ)
  (h1 : 240 * m + 180 * (10 - m) ≥ 2040)
  (h2 : 12 * m + 10 * (10 - m) ≤ 110) :
  m = 4 :=
sorry

end find_prices_possible_plans_cost_effective_plan_l396_396227


namespace total_earnings_from_peaches_l396_396663

-- Definitions of the conditions
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def price_per_peach_friends : ℝ := 2
def peaches_sold_to_relatives : ℕ :=  4
def price_per_peach_relatives : ℝ := 1.25
def peaches_for_self : ℕ := 1

-- We aim to prove the following statement
theorem total_earnings_from_peaches :
  (peaches_sold_to_friends * price_per_peach_friends) +
  (peaches_sold_to_relatives * price_per_peach_relatives) = 25 := by
  -- proof goes here
  sorry

end total_earnings_from_peaches_l396_396663


namespace keychain_arrangement_count_l396_396612

theorem keychain_arrangement_count :
  let keys := {1, 2, 3, 4, 5, 6} in
  let house_key := 1 in
  let car_key := 2 in
  let shed_key := 3 in
  let remaining_keys := {4, 5, 6} in
  ∃ arrangements : Finset (Finset Nat), 
    (∀ k ∈ arrangements, k ∈ {keys} ∨ k = {house_key, car_key, shed_key, remaining_keys}) ∧
    arrangements.card = 36 :=
by
  sorry

end keychain_arrangement_count_l396_396612


namespace colonization_combinations_l396_396917

theorem colonization_combinations :
  ∑ a in finset.range 9,
   (∑ b in finset.range 8,
   if 2 * a + b = 18 then (nat.choose 8 a) * (nat.choose 7 b) else 0) = 497 :=
by
  sorry

end colonization_combinations_l396_396917


namespace value_of_y_l396_396838

theorem value_of_y (x y : ℤ) (h1 : 1.5 * (x : ℝ) = 0.25 * (y : ℝ)) (h2 : x = 24) : y = 144 :=
  sorry

end value_of_y_l396_396838


namespace line_bisects_area_and_perimeter_proof_l396_396593

theorem line_bisects_area_and_perimeter_proof (ABC : Triangle) (DE : Line) (D E : Point) :
  bisects_area DE ABC ∧ bisects_perimeter DE ABC ∧ D ∈ (side AB ABC) ∧ E ∈ (side AC ABC)  →
  passes_through (line DE) (incenter ABC) :=
by
  sorry

end line_bisects_area_and_perimeter_proof_l396_396593


namespace ratio_expression_value_l396_396930

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396930


namespace integral_exp_plus_2x_l396_396488

theorem integral_exp_plus_2x : 
  ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 := 
  sorry

end integral_exp_plus_2x_l396_396488


namespace domain_shifted_function_l396_396955

def domain_fx := set.Icc (0 : ℝ) 2
def domain_fx_plus_1 := set.Icc (-1 : ℝ) 1

theorem domain_shifted_function (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_fx → f x = f (x + 1)) →
  (∀ y, y ∈ domain_fx_plus_1 → f (y + 1) = f y) :=
by
  intros h y hy
  sorry

end domain_shifted_function_l396_396955


namespace circles_intersection_correct_statements_l396_396326

-- Definitions of the circles
def circle_O1 (x y : ℝ) := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y = 0

-- Line of the chord AB derived from the given circles
def chord_line (x y : ℝ) := x - y = 0

-- Perpendicular bisector of segment AB
def perpendicular_bisector (x y : ℝ) := x + y - 1 = 0

-- Length of common chord AB (incorrect statement C)
def length_common_chord_AB : ℝ := real.sqrt 2

-- Maximum distance from a point P on circle O1 to the chord line
def max_distance_P_to_chord : ℝ := (real.sqrt 2) / 2 + 1

-- The proof problem statement
theorem circles_intersection_correct_statements :
  (∀ x y : ℝ, circle_O1 x y ∧ circle_O2 x y → chord_line x y) ∧
  (∀ x y : ℝ, ∃ xmid ymid : ℝ, xmid = 0 ∧ ymid = 1 ∧ perpendicular_bisector x y) ∧
  -- statement C is false, hence we don't require the length proof here
  (∀ x y : ℝ, ∀ P : ℝ, circle_O1 x y → max_distance_P_to_chord = (real.sqrt 2) / 2 + 1) 
 := 
sorry

end circles_intersection_correct_statements_l396_396326


namespace simplify_expression_l396_396301

theorem simplify_expression (a1 a2 a3 a4 : ℝ) (h1 : 1 - a1 ≠ 0) (h2 : 1 - a2 ≠ 0) (h3 : 1 - a3 ≠ 0) (h4 : 1 - a4 ≠ 0) :
  1 + a1 / (1 - a1) + a2 / ((1 - a1) * (1 - a2)) + a3 / ((1 - a1) * (1 - a2) * (1 - a3)) + 
  (a4 - a1) / ((1 - a1) * (1 - a2) * (1 - a3) * (1 - a4)) = 
  1 / ((1 - a2) * (1 - a3) * (1 - a4)) :=
by
  sorry

end simplify_expression_l396_396301


namespace domain_of_f_squared_l396_396954

theorem domain_of_f_squared (f : ℝ → ℝ) (hf : ∀ x, 1 ≤ x ∧ x ≤ 2 → x ∈ set.univ) :
  set.range (λ x, f (x^2)) = {y | (∃ x, -real.sqrt 2 ≤ x ∧ x ≤ -1 ∧ y = f(x^2)) ∨ 
                                   (∃ x, 1 ≤ x ∧ x ≤ real.sqrt 2 ∧ y = f(x^2)) } :=
sorry

end domain_of_f_squared_l396_396954


namespace marlon_bunnies_count_l396_396665

def initial_bunnies : Nat := 30
def fraction_given_away : Rational := 2/5
def kittens_per_bunny : Nat := 2

noncomputable def bunnies_after_giving_away : Nat :=
  initial_bunnies - (fraction_given_away * initial_bunnies : Rational).natValue

noncomputable def new_kittens : Nat :=
  bunnies_after_giving_away * kittens_per_bunny

noncomputable def total_bunnies : Nat :=
  bunnies_after_giving_away + new_kittens

theorem marlon_bunnies_count : total_bunnies = 54 := by
  sorry

end marlon_bunnies_count_l396_396665


namespace perpendicular_line_equation_l396_396491

theorem perpendicular_line_equation 
  (A : ℝ × ℝ) (L : ℝ → ℝ → Prop) (x y : ℝ) :
  A = (3, 2) →
  L = λ x y, 4 * x + 5 * y - 8 = 0 →
  (5 * x - 4 * y - 7 = 0) ∧ L A.1 A.2 = 0 →
  (∃ m : ℝ, ∀ x y, 5 * x - 4 * y + m = 0 → L x y = 0 → ∃ x y, (x, y) = A ∧ 5 * x - 4 * y - 7 = 0) :=
by sorry

end perpendicular_line_equation_l396_396491


namespace find_n_l396_396192

-- Define the operation ø
def op (x w : ℕ) : ℚ := (2^x : ℚ) / (2^w : ℚ)

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : op (op n 2) 3 = 2) : n = 6 := sorry

end find_n_l396_396192


namespace herd_total_cows_l396_396403

theorem herd_total_cows (n : ℕ) (h1 : (1 / 3 : ℚ) * n + (1 / 5 : ℚ) * n + (1 / 6 : ℚ) * n + 19 = n) : n = 63 :=
sorry

end herd_total_cows_l396_396403


namespace triangle_A_l396_396987

variables {a b c : ℝ}
variables (A B C : ℝ) -- Represent vertices
variables (C1 C2 A1 A2 B1 B2 A' B' C' : ℝ)

-- Definition of equilateral triangle
def is_equilateral_trig (x y z : ℝ) : Prop :=
  dist x y = dist y z ∧ dist y z = dist z x

-- Given conditions
axiom ABC_equilateral : is_equilateral_trig A B C
axiom length_cond_1 : dist A1 A2 = a ∧ dist C B1 = a ∧ dist B C2 = a
axiom length_cond_2 : dist B1 B2 = b ∧ dist A C1 = b ∧ dist C A2 = b
axiom length_cond_3 : dist C1 C2 = c ∧ dist B A1 = c ∧ dist A B2 = c

-- Additional constructions
axiom A'_construction : is_equilateral_trig A' B2 C1
axiom B'_construction : is_equilateral_trig B' C2 A1
axiom C'_construction : is_equilateral_trig C' A2 B1

-- The final proof goal
theorem triangle_A'B'C'_equilateral : is_equilateral_trig A' B' C' :=
sorry

end triangle_A_l396_396987


namespace equation_permutations_l396_396507

theorem equation_permutations {eq : List Char} (h_eq : eq = ['e', 'q', 'u', 'a', 't', 'i', 'o', 'n']) :
  ∃ n : ℕ, n = 480 ∧ 
  let letters := ['e', 'a', 't', 'i', 'o', 'n'],
      selections := Nat.choose letters.length 3,
      total_permutations := selections * 4! 
  in total_permutations = 480 :=
by
  sorry

end equation_permutations_l396_396507


namespace largest_sum_at_cube_vertex_l396_396324

theorem largest_sum_at_cube_vertex (opposite_faces_sum_7 : ∀ (f1 f2 : ℕ), (f1 + f2 = 7) → 
                                  (f1 = 1 → f2 = 6) ∧ (f1 = 2 → f2 = 5) ∧ (f1 = 3 → f2 = 4)) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
                  (a + b + c = 14) ∧ 
                  (a + opposite_faces_sum_7 a b = 7) ∧ 
                  (b + opposite_faces_sum_7 b c = 7) :=
sorry

end largest_sum_at_cube_vertex_l396_396324


namespace simplify_and_sum_of_exponents_l396_396300

-- Define the given expression
def radicand (x y z : ℝ) : ℝ := 40 * x ^ 5 * y ^ 7 * z ^ 9

-- Define what cube root stands for
noncomputable def cbrt (a : ℝ) := a ^ (1 / 3 : ℝ)

-- Define the simplified expression outside the cube root
noncomputable def simplified_outside_exponents (x y z : ℝ) : ℝ := x * y * z ^ 3

-- Define the sum of the exponents outside the radical
def sum_of_exponents_outside (x y z : ℝ) : ℝ := (1 + 1 + 3 : ℝ)

-- Statement of the problem in Lean
theorem simplify_and_sum_of_exponents (x y z : ℝ) :
  sum_of_exponents_outside x y z = 5 :=
by 
  sorry

end simplify_and_sum_of_exponents_l396_396300


namespace product_of_four_consecutive_integers_is_perfect_square_l396_396286

-- Define the main statement we want to prove
theorem product_of_four_consecutive_integers_is_perfect_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  -- Proof is omitted
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l396_396286


namespace necessary_and_sufficient_condition_l396_396510

variable (a b : ℝ)

-- Define the conditions
def condition_1 : Prop := a > 0
def condition_2 : Prop := a ≠ 1 

-- Define the implications
def implication_1 : Prop := (a^b > 1) → ((a - 1) * b > 0)
def implication_2 : Prop := ((a - 1) * b > 0) → (a^b > 1)

-- Define the final necessary and sufficient condition statement
theorem necessary_and_sufficient_condition 
  (h1 : condition_1) 
  (h2 : condition_2) : implication_1 ∧ implication_2 := 
  by
    sorry

end necessary_and_sufficient_condition_l396_396510


namespace part1_part2_l396_396901

noncomputable def f (x a : ℝ) := Real.exp x - a * x^2

theorem part1 (x : ℝ) (hx : 0 ≤ x) :
  f x 1 ≥ 1 :=
sorry

theorem part2 (h : ∀ x : ℝ, 0 < x → (f x a) = 0 → a = Real.exp 2 / 4) :
  ∃ a : ℝ, ∀ x : ℝ, (0 < x → (f x a) = 0) → a = Real.exp 2 / 4 :=
by { use Real.exp 2 / 4, exact h }

end part1_part2_l396_396901


namespace find_angle_C_l396_396658

-- Define the obtuse triangle ABC with circumcenter O and the required conditions
variables (A B C O D : Type)
           [Triangle ABC]
           (circumcenter O ABC)
           (BD : Segment B D)
           (DC : Segment D C)

-- Define given conditions
def obtuse_triangle : Prop := ∃ A B C : Type, Triangle ABC ∧ ∠ABC = 15 ∧ ∠BAC > 90
def intersection : Prop := ∃ A O D : Type, Segment A O ∧ Segment D (B C) ∧ intersects AO (BC) at D
def equation_condition : Prop := OD^2 + OC * DC = OC^2

-- Define angles
variables (α : ℝ) (r : ℝ)

-- Lean statement for proving the angle ∠ C
theorem find_angle_C (h1: obtuse_triangle) (h2: ∠ABC = 15) (h3: ∠BAC > 90) 
 (h4: intersection) (h5: equation_condition) : ∠ACB = 35 :=
by
  sorry

end find_angle_C_l396_396658


namespace emily_coloring_books_l396_396480

variable (initial_books : ℕ) (given_away : ℕ) (total_books : ℕ) (bought_books : ℕ)

theorem emily_coloring_books :
  initial_books = 7 →
  given_away = 2 →
  total_books = 19 →
  initial_books - given_away + bought_books = total_books →
  bought_books = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end emily_coloring_books_l396_396480


namespace irreducible_fraction_l396_396110

theorem irreducible_fraction (n : ℕ) :
  irreducible (2 * n^2 + 11 * n - 18) (n + 7) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
sorry

end irreducible_fraction_l396_396110


namespace total_vehicles_in_lanes_l396_396000

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l396_396000


namespace isosceles_triangle_perimeter_l396_396186

theorem isosceles_triangle_perimeter :
  ∃ P : ℕ, (P = 15 ∨ P = 18) ∧ ∀ (a b c : ℕ), (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 4 ∨ b = 4 ∨ c = 4) → ((a = 7 ∨ a = 4) ∧ (b = 7 ∨ b = 4) ∧ (c = 7 ∨ c = 4)) ∧ P = a + b + c :=
by
  sorry

end isosceles_triangle_perimeter_l396_396186


namespace cash_received_l396_396202

noncomputable def total_contribution_A := 25
noncomputable def total_contribution_B := 36
noncomputable def total_contribution_C := 38
noncomputable def first_week_winnings := 1100
noncomputable def second_week_winnings := 73000
noncomputable def ticket_cost := 561
noncomputable def remaining_to_share := first_week_winnings - ticket_cost
noncomputable def remaining_ratio_A := 1
noncomputable def remaining_ratio_B := 2
noncomputable def remaining_ratio_C := 4
noncomputable def remaining_total_ratio := remaining_ratio_A + remaining_ratio_B + remaining_ratio_C
noncomputable def share_A := (remaining_to_share * remaining_ratio_A) / remaining_total_ratio
noncomputable def share_B := (remaining_to_share * remaining_ratio_B) / remaining_total_ratio
noncomputable def share_C := (remaining_to_share * remaining_ratio_C) / remaining_total_ratio
noncomputable def net_contribution_A := (25 * 1100 / (25 + 36 + 38)) - share_A
noncomputable def net_contribution_B := (36 * 1100 / (25 + 36 + 38)) - share_B
noncomputable def net_contribution_C := (38 * 1100 / (25 + 36 + 38)) - share_C

theorem cash_received (A B C: ℝ) : 
  A = 26135.89 ∧ B = 32052.34 ∧ C = 14811.77 :=
begin
  sorry
end

end cash_received_l396_396202


namespace jogging_track_circumference_l396_396720

noncomputable def suresh_speed_km_hr : ℝ := 4.5
noncomputable def wife_speed_km_hr : ℝ := 3.75
noncomputable def meeting_time_min : ℝ := 5.28

def suresh_speed_km_min : ℝ := suresh_speed_km_hr / 60
def wife_speed_km_min : ℝ := wife_speed_km_hr / 60

def total_distance_km : ℝ := (suresh_speed_km_min + wife_speed_km_min) * meeting_time_min

theorem jogging_track_circumference :
  total_distance_km = 0.7257 :=
by
  sorry

end jogging_track_circumference_l396_396720


namespace find_x_l396_396590

theorem find_x (x y : ℝ) (h1 : 0.65 * x = 0.20 * y)
  (h2 : y = 617.5 ^ 2 - 42) : 
  x = 117374.3846153846 :=
by
  sorry

end find_x_l396_396590


namespace binomial_7_2_l396_396062

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396062


namespace caitlin_draws_pairs_probability_l396_396818

def caitlin_probability : ℚ :=
  let total_ways := nat.choose 10 6 in
  let favorable_ways := nat.choose 5 2 * nat.choose 3 2 * 1 * 1 in
  favorable_ways / total_ways

theorem caitlin_draws_pairs_probability :
  caitlin_probability = 1 / 7 :=
by {
  sorry
}

end caitlin_draws_pairs_probability_l396_396818


namespace find_valid_pairs_l396_396141

theorem find_valid_pairs :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 →
  (∃ C : ℤ, ∀ (n : ℕ), 0 < n → (a^n + b^(n+9)) % 13 = C % 13) ↔
  (a, b) = (1, 1) ∨ (a, b) = (4, 4) ∨ (a, b) = (10, 10) ∨ (a, b) = (12, 12) := 
by
  sorry

end find_valid_pairs_l396_396141


namespace part_B_part_C_l396_396860

noncomputable def a : ℝ := Real.exp (-4 / 5)
noncomputable def b : ℝ := 13 / 25
noncomputable def c : ℝ := 5 / 9

theorem part_B : a < b := 
by { 
  sorry,
}

theorem part_C : a < c := 
by { 
  sorry,
}

end part_B_part_C_l396_396860


namespace alex_mother_age_proof_l396_396670

-- Define the initial conditions
def alex_age_2004 : ℕ := 7
def mother_age_2004 : ℕ := 35
def initial_year : ℕ := 2004

-- Define the time variable and the relationship conditions
def years_after_2004 (x : ℕ) : Prop :=
  let alex_age := alex_age_2004 + x
  let mother_age := mother_age_2004 + x
  mother_age = 2 * alex_age

-- State the theorem to be proved
theorem alex_mother_age_proof : ∃ x : ℕ, years_after_2004 x ∧ initial_year + x = 2025 :=
by
  sorry

end alex_mother_age_proof_l396_396670


namespace tom_spends_total_cost_l396_396356

theorem tom_spends_total_cost :
  (let total_bricks := 1000
       half_bricks := total_bricks / 2
       full_price := 0.50
       half_price := full_price / 2
       cost_half := half_bricks * half_price
       cost_full := half_bricks * full_price
       total_cost := cost_half + cost_full
   in total_cost = 375) := 
by
  let total_bricks := 1000
  let half_bricks := total_bricks / 2
  let full_price := 0.50
  let half_price := full_price / 2
  let cost_half := half_bricks * half_price
  let cost_full := half_bricks * full_price
  let total_cost := cost_half + cost_full
  show total_cost = 375 from sorry

end tom_spends_total_cost_l396_396356


namespace flooring_cost_correct_l396_396411

noncomputable def floorCost : ℝ := 
  let largeRectangleArea := 5.5 * 3.75   -- Larger rectangular area
  let smallRectangleArea := 2.5 * 1.5    -- Smaller rectangular area
  let costA := largeRectangleArea * 600  -- Cost for material A
  let costB := smallRectangleArea * 450  -- Cost for material B
  costA + costB                          -- Total cost

theorem flooring_cost_correct :
  floorCost = 14062.50 :=
by
  unfold floorCost
  apply congrArg2 (· + ·) 
  {
    have h1 : 5.5 * 3.75 = 20.625 := by norm_num
    have h2 : 20.625 * 600 = 12375 := by norm_num
    exact h2
  }
  {
    have h1 : 2.5 * 1.5 = 3.75 := by norm_num
    have h2 : 3.75 * 450 = 1687.5 := by norm_num
    exact h2
  }
  have h3 : 12375 + 1687.5 = 14062.5 := by norm_num
  exact h3

end flooring_cost_correct_l396_396411


namespace complex_root_modulus_one_implies_divisibility_l396_396242

theorem complex_root_modulus_one_implies_divisibility (n : ℕ) (z : ℂ) (h : z^(n+1) - z^n - 1 = 0) (hz : |z| = 1) :
  (n + 2) % 6 = 0 ↔ ∃ z : ℂ, z^(n + 1) - z^n - 1 = 0 ∧ |z| = 1 :=
  sorry

end complex_root_modulus_one_implies_divisibility_l396_396242


namespace total_yellow_marbles_l396_396269

theorem total_yellow_marbles 
  (Mary Joan Tim Lisa : ℕ)
  (hMary : Mary = 9)
  (hJoan : Joan = 3)
  (hTim : Tim = 5)
  (hLisa : Lisa = 7) : Mary + Joan + Tim + Lisa = 24 := by
  rw [hMary, hJoan, hTim, hLisa]
  norm_num
  sorry

end total_yellow_marbles_l396_396269


namespace last_two_digits_A_pow_20_l396_396986

/-- 
Proof that for any even number A not divisible by 10, 
the last two digits of A^20 are 76.
--/
theorem last_two_digits_A_pow_20 (A : ℕ) (h_even : A % 2 = 0) (h_not_div_by_10 : A % 10 ≠ 0) : 
  (A ^ 20) % 100 = 76 :=
by
  sorry

end last_two_digits_A_pow_20_l396_396986


namespace fastest_route_to_Sherbourne_l396_396348

/-- Given the distance and travel times of different train lines,
prove that the fastest route for Harsha to reach Sherbourne from Forest Grove
is via the Blue Line, and it takes her 2 hours. -/
theorem fastest_route_to_Sherbourne :
  let total_distance := 200
  let forest_grove_distance := total_distance / 5
  let remaining_distance := total_distance - forest_grove_distance
  let blue_line_speed := (total_distance / (5 / 2)) -- 80 km/h
  let red_line_speed := (total_distance / (6 / 2)) -- 66.67 km/h
  let green_line_speed := (total_distance / (7 / 2)) -- 57.14 km/h
  let blue_line_time := remaining_distance / blue_line_speed
  let red_line_time := remaining_distance / red_line_speed
  let green_line_time := remaining_distance / green_line_speed
  in blue_line_time = 2 ∧ blue_line_time = min red_line_time (min blue_line_time green_line_time) :=
by
  /- Proof steps would go here -/
  sorry

end fastest_route_to_Sherbourne_l396_396348


namespace integral_abs_sin_minus_cos_l396_396734

theorem integral_abs_sin_minus_cos :
  ∫ x in 0..π, |sin x - cos x| = 2 * sqrt 2 :=
by
  sorry

end integral_abs_sin_minus_cos_l396_396734


namespace lorelei_roses_l396_396213

theorem lorelei_roses :
  let red_flowers := 12
  let pink_flowers := 18
  let yellow_flowers := 20
  let orange_flowers := 8
  let lorelei_red := (50 / 100) * red_flowers
  let lorelei_pink := (50 / 100) * pink_flowers
  let lorelei_yellow := (25 / 100) * yellow_flowers
  let lorelei_orange := (25 / 100) * orange_flowers
  lorelei_red + lorelei_pink + lorelei_yellow + lorelei_orange = 22 :=
by
  sorry

end lorelei_roses_l396_396213


namespace unique_prime_range_start_l396_396201

theorem unique_prime_range_start (N : ℕ) (hN : N = 220) (h1 : ∀ n, N ≥ n → n ≥ 211 → ¬Prime n) (h2 : Prime 211) : N - 8 = 212 :=
by
  sorry

end unique_prime_range_start_l396_396201


namespace derivative_at_1_eq_3_l396_396167

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + x

-- Define the derivative of the function f
noncomputable def f' (x : ℝ) : ℝ := DifferentiableAt.deriv (λ x, 2 * Real.log x + x) x

-- Statement of the theorem to prove f'(1) = 3
theorem derivative_at_1_eq_3 : f' 1 = 3 :=
sorry

end derivative_at_1_eq_3_l396_396167


namespace find_fraction_a1_d_l396_396614

-- Define the conditions for the arithmetic sequence
variable (a_1 d : ℝ)
variable (S : ℕ → ℝ)
variable (n : ℕ)

-- Sum of the first n terms in an arithmetic sequence
def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

-- Given condition
axiom given_condition : arithmetic_sum a_1 d 10 = 4 * arithmetic_sum a_1 d 5

theorem find_fraction_a1_d : a_1 / d = 1 / 2 :=
by {
  -- Sorry serves as a placeholder for steps of the proof
  sorry
}

end find_fraction_a1_d_l396_396614


namespace count_subsets_containing_7_l396_396570

def original_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def subsets_containing_7 (s : Set ℕ) : Prop :=
  7 ∈ s

theorem count_subsets_containing_7 :
  {s : Set ℕ | s ⊆ original_set ∧ subsets_containing_7 s}.finite.card = 64 :=
sorry

end count_subsets_containing_7_l396_396570


namespace sign_eq_unit_step_diff_l396_396907

def unit_step (x : ℝ) : ℝ := if x >= 0 then 1 else 0

def sign (x : ℝ) : ℝ := if x > 0 then 1 else if x = 0 then 0 else -1

theorem sign_eq_unit_step_diff (x : ℝ) : sign x = unit_step x - unit_step (-x) :=
  sorry

end sign_eq_unit_step_diff_l396_396907


namespace area_correct_l396_396769

noncomputable def area_bounded_curves : ℝ := sorry

theorem area_correct :
  ∃ S, S = area_bounded_curves ∧ S = 12 * pi + 16 := sorry

end area_correct_l396_396769


namespace number_of_solutions_l396_396501

theorem number_of_solutions (x : ℤ) : (x^2 < 7 * x) → x ∈ {1, 2, 3, 4, 5, 6} :=
sorry

end number_of_solutions_l396_396501


namespace range_of_a_minus_b_l396_396511

theorem range_of_a_minus_b (a b : ℝ) (h1 : a^2 - 2 * a * b - 3 * b^2 = 1) (h2 : -1 ≤ Real.log2 (a + b) ∧ Real.log2 (a + b) ≤ 1) :
  1 ≤ a - b ∧ a - b ≤ 5/4 :=
by sorry

end range_of_a_minus_b_l396_396511


namespace quadratic_inequality_solution_l396_396107

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 - 6 * x - 16 > 0) ↔ (x < -2 ∨ x > 8) :=
sorry

end quadratic_inequality_solution_l396_396107


namespace solution_of_x_l396_396866

theorem solution_of_x (x y z : ℝ) (h_nonzero : y * z ≠ 0)
  (h_set_eq : {2 * x, 3 * z, x * y} = {y, 2 * x^2, 3 * x * z}) :
  x = 1 := 
by {
  sorry
}

end solution_of_x_l396_396866


namespace complex_numbers_on_same_circle_l396_396139

theorem complex_numbers_on_same_circle
  (a1 a2 a3 a4 a5 : ℂ)
  (q : ℂ)
  (s : ℝ) (h_s : |s| ≤ 2)
  (h1 : a2 / a1 = q)
  (h2 : a3 / a2 = q)
  (h3 : a4 / a3 = q)
  (h4 : a5 / a4 = q)
  (h5 : (a1 + a2 + a3 + a4 + a5) = 4 * (1 / a1 + 1 / a2 + 1 / a3 + 1 / a4 + 1 / a5))
  : ∃ (r : ℝ), ∀ (i : ℕ), |[a1, a2, a3, a4, a5].nthLe i sorry| = r := sorry

end complex_numbers_on_same_circle_l396_396139


namespace base9_digit_divisible_by_13_l396_396854

theorem base9_digit_divisible_by_13 :
    ∃ (d : ℕ), (0 ≤ d ∧ d ≤ 8) ∧ (13 ∣ (2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4)) :=
by
  sorry

end base9_digit_divisible_by_13_l396_396854


namespace equation_of_line_polar_to_cartesian_range_of_distances_to_line_l396_396961

theorem equation_of_line_polar_to_cartesian {rho theta : ℝ} (h_C : rho * (cos theta)^2 = 2 * sin theta)
  {Mx My : ℝ} (h_M : Mx = 2*sqrt(2) ∧ My = pi/4) :
  ∃ (x y : ℝ), y = 2*x - 2 := sorry

theorem range_of_distances_to_line {a b : ℝ} (h_ellipse : a^2 / 3 + b^2 / 4 = 1)
  (h_line : 2 * a - b - 2 = 0) :
  ∃ (d_min d_max : ℝ), d_min = 0 ∧ d_max = 6 * sqrt(5) / 5 := sorry

end equation_of_line_polar_to_cartesian_range_of_distances_to_line_l396_396961


namespace distinct_ellipses_eccentricity_count_l396_396118

def ellipse_equation (a b : ℕ) : Prop := a > b ∧ a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}

noncomputable def eccentricity (a b : ℕ) : ℝ := real.sqrt (1 - (b^2 / a^2))

def count_distinct_eccentricities : ℕ :=
  (finset.univ.product finset.univ).filter (λ (p : ℕ × ℕ), ellipse_equation p.1 p.2)
  .image (λ p, eccentricity p.1 p.2)
  .card

theorem distinct_ellipses_eccentricity_count : count_distinct_eccentricities = 11 := 
by sorry

end distinct_ellipses_eccentricity_count_l396_396118


namespace largest_prime_factor_9604_l396_396095

theorem largest_prime_factor_9604 :
  let n := 9604
  let a := 64
  let b := 136
  let factors_64 := 2^6
  let factors_136 := 2^3 * 17
  n = a * b ∧ a = factors_64 ∧ b = factors_136 → ∃ p, p = 17 ∧ (∀ q, prime q → q ∣ n → q ≤ 17) :=
by
  sorry

end largest_prime_factor_9604_l396_396095


namespace geometric_sequence_a6a7_l396_396973

theorem geometric_sequence_a6a7 (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n+1) = q * a n)
  (h1 : a 4 * a 5 = 1)
  (h2 : a 8 * a 9 = 16) : a 6 * a 7 = 4 :=
sorry

end geometric_sequence_a6a7_l396_396973


namespace sum_of_ages_l396_396839

theorem sum_of_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
sorry

end sum_of_ages_l396_396839


namespace As_share_of_profit_l396_396762

theorem As_share_of_profit (A_investment : ℕ) (A_months : ℕ) (B_investment : ℕ) (B_months : ℕ) (total_profit : ℕ) :
  A_investment = 300 →
  A_months = 12 →
  B_investment = 200 →
  B_months = 6 →
  total_profit = 100 →
  let A_investment_months := A_investment * A_months,
      B_investment_months := B_investment * B_months,
      total_investment_months := A_investment_months + B_investment_months,
      A_share_of_profit := (A_investment_months * total_profit) / total_investment_months
  in A_share_of_profit = 75 :=
by
  intros
  let A_investment_months := A_investment * A_months
  let B_investment_months := B_investment * B_months
  let total_investment_months := A_investment_months + B_investment_months
  let A_share_of_profit := (A_investment_months * total_profit) / total_investment_months
  sorry

end As_share_of_profit_l396_396762


namespace probability_two_red_two_blue_l396_396780

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let total_ways_to_choose_4 := Nat.choose total_marbles 4
  let ways_to_choose_2_red := Nat.choose red_marbles 2
  let ways_to_choose_2_blue := Nat.choose blue_marbles 2
  (ways_to_choose_2_red * ways_to_choose_2_blue : ℚ) / total_ways_to_choose_4 = 56 / 147 := 
by {
  sorry
}

end probability_two_red_two_blue_l396_396780


namespace quadratic_inequality_l396_396502

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 16 :=
by sorry

end quadratic_inequality_l396_396502


namespace problem_part1_problem_part2_l396_396531

noncomputable def f (x : ℝ) (h : x >= 0) : ℝ := x^2 - 4 * x

theorem problem_part1 (f : ℝ -> ℝ)
  (h1 : ∀ x, f(-x) = -f(x))
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 4 * x) :
  f(-3) + f(-2) + f(3) = 4 := 
  sorry
  
theorem problem_part2 (f : ℝ -> ℝ)
  (h1 : ∀ x, f(-x) = -f(x))
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 4 * x) :
  (∀ x, f x =
    if x ≥ 0 then x^2 - 4 * x 
    else - x^2 - 4 * x)
  ∧  (inter : set.Icc (-real.infinity) (-2) ∫ (2 : ℝ) (+real.infinity))
    := 
  sorry

end problem_part1_problem_part2_l396_396531


namespace find_a_l396_396171

theorem find_a (A B : Set ℤ) (a b : ℤ)
  (hA : A = {-1, a})
  (hB : B = {3^a, b})
  (hUnion : A ∪ B = {-1, 0, 1}) :
  a = 0 := by
  sorry

end find_a_l396_396171


namespace diagonals_of_convex_polygon_l396_396821

theorem diagonals_of_convex_polygon (n : ℕ) (hn : n = 25) : 
  let total_diagonals := (n * (n - 3)) / 2
  in total_diagonals = 275 :=
by 
  sorry

end diagonals_of_convex_polygon_l396_396821


namespace triangle_problem_l396_396882

variables {a b c A B C : ℝ}

-- The conditions
def triangle_conditions (a b c A B C : ℝ) : Prop :=
  c * Real.cos A + (√3) * c * Real.sin A - b - a = 0

-- The problem statement to prove
theorem triangle_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : triangle_conditions a b c A B C) :
  C = 60 * Real.pi / 180 ∧ (c = 1 → 0.5 * a * b * Real.sin C ≤ sqrt 3 / 4) :=
by
  sorry

end triangle_problem_l396_396882


namespace smallest_N_satisfying_2_pow_n_gt_n_sq_l396_396497

theorem smallest_N_satisfying_2_pow_n_gt_n_sq :
  ∃ N : ℕ, ∀ n ∈ {N, N + 1, N + 2, N + 3, N + 4}, 2^n > n^2 ∧ N = 5 :=
by
  existsi 5
  intros n hn
  fin_cases hn
  all_goals 
    { 
      sorry 
    }

end smallest_N_satisfying_2_pow_n_gt_n_sq_l396_396497


namespace spadesuit_evaluation_l396_396474

def spadesuit (a b : ℝ) : ℝ := (3 * a^2) / b - (b^2) / a

theorem spadesuit_evaluation :
  spadesuit 4 (spadesuit 2 3) = -0.5 → 
  spadesuit 4 (-0.5) = -96.0625 →
  spadesuit (-96.0625) 5 = 5541.3428 :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end spadesuit_evaluation_l396_396474


namespace solution_l396_396136

variables (a b c : ℤ → ℝ)

def seq_conditions := ∀ n : ℤ,
  a n ≥ (1 / 2) * (b (n + 1) + c (n - 1)) ∧
  b n ≥ (1 / 2) * (c (n + 1) + a (n - 1)) ∧
  c n ≥ (1 / 2) * (a (n + 1) + b (n - 1))

def initial_conditions : Prop :=
  a 0 = 26 ∧ b 0 = 6 ∧ c 0 = 2004

theorem solution (h1 : seq_conditions a b c) (h2 : initial_conditions):
  a 2005 = 2004 ∧ b 2005 = 26 ∧ c 2005 = 6 :=
sorry

end solution_l396_396136


namespace paths_from_A_to_B_paths_from_A_to_B_via_C_l396_396603

section

variables (A B C : ℕ × ℕ)
variables (m n p q : ℕ)

def shortest_paths (A B : ℕ × ℕ) : ℕ :=
  let (a1, a2) := A in
  let (b1, b2) := B in
  Nat.choose (b1 + b2 - a1 - a2) (b1 - a1)

theorem paths_from_A_to_B (A B : ℕ × ℕ) (hA : A = (0, 0)) (hB : B = (4, 4)) :
  shortest_paths A B = 70 := sorry

theorem paths_from_A_to_B_via_C (A B C : ℕ × ℕ)
  (hA : A = (0, 0)) (hB : B = (4, 4)) (hC : C = (2, 2)) :
  shortest_paths A C * shortest_paths C B = 36 := sorry

end

end paths_from_A_to_B_paths_from_A_to_B_via_C_l396_396603


namespace find_b_l396_396087

theorem find_b (x : ℝ) (b : ℝ) :
  (∃ t u : ℝ, (bx^2 + 18 * x + 9) = (t * x + u)^2 ∧ u^2 = 9 ∧ 2 * t * u = 18 ∧ t^2 = b) →
  b = 9 :=
by
  sorry

end find_b_l396_396087


namespace prime_divides_iff_in_prime_factorization_l396_396688

noncomputable theory
open_locale classical

open Nat

-- We are stating the theorem.
theorem prime_divides_iff_in_prime_factorization (p n : ℕ) (hp : Prime p) :
  (p ∣ n) ↔ (p ∈ Nat.factors n) := 
sorry

end prime_divides_iff_in_prime_factorization_l396_396688


namespace twenty_four_game_l396_396550

theorem twenty_four_game : 8 / (3 - 8 / 3) = 24 := 
by
  sorry

end twenty_four_game_l396_396550


namespace add_fractions_simplified_l396_396819

theorem add_fractions_simplified (a b c d : ℕ) (h1 : a = 7) (h2 : b = 12) (h3 : c = 11) (h4 : d = 16):
    (a : ℚ) / b + c / d = 61 / 48 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end add_fractions_simplified_l396_396819


namespace ln_increasing_l396_396076

noncomputable def ln_increasing_interval (x : ℝ) : Prop :=
  ln (6 + x - x^2) = y ∧ (-2 < x ∧ x < 1/2)

theorem ln_increasing (x : ℝ) (hx : (-2 < x ∧ x < 1/2)) :
  ∀ x₁ x₂ : ℝ, (-2 < x₁ ∧ x₁ < 1/2) ∧ (-2 < x₂ ∧ x₂ < 1/2) ∧ x₁ < x₂ →
  ln (6 + x₁ - x₁^2) < ln (6 + x₂ - x₂^2) :=
sorry

end ln_increasing_l396_396076


namespace slope_CD_eq_one_l396_396624

theorem slope_CD_eq_one (k x1 x2 : ℝ) (h_k_gt_zero : k > 0) (h_x1_pos : x1 > 0) (h_x2_pos : x2 > 0) 
  (h_A : k * x1 = exp (x1 - 1)) (h_B : k * x2 = exp (x2 - 1)) :
  (ln x2 - ln x1) / (x2 - x1) = 1 :=
sorry

end slope_CD_eq_one_l396_396624


namespace crayon_selection_l396_396736

theorem crayon_selection (total_crayons selected_crayons remaining_crayons mandatory_crayons : ℕ) :
  total_crayons = 15 → 
  selected_crayons = 6 → 
  mandatory_crayons = 2 → 
  remaining_crayons = total_crayons - mandatory_crayons →
  combinatorial.choose (remaining_crayons) (selected_crayons - mandatory_crayons) = 715 :=
by intros; dsimp at *; sorry

end crayon_selection_l396_396736


namespace square_of_length_graph_v_l396_396308

-- Define linear functions p(x), q(x), r(x)
noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry
noncomputable def r : ℝ → ℝ := sorry

-- Define u(x) and v(x)
noncomputable def u (x : ℝ) : ℝ := max (max (p x) (q x)) (r x)
noncomputable def v (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

-- Condition for u(x) graph segments
axiom u_conditions : 
  ∀ x, 
    if x < -1 then u x = 6 - (1) * (x + 4)
    else if x < 4 then u x = 3 + (0.6) * (x + 1)
    else u x = 6

-- Prove the length squared of v(x) graph from -4 to 4
theorem square_of_length_graph_v :
  let l1 := real.sqrt (9 + 9),
      l2 := real.sqrt (25 + 9)
  in (l1 + l2) ^ 2 = 18 + 34 + 6 * real.sqrt 68 := sorry

end square_of_length_graph_v_l396_396308


namespace total_worth_of_stock_l396_396001

theorem total_worth_of_stock (X : ℝ) :
  (0.30 * 0.10 * X + 0.40 * -0.05 * X + 0.30 * -0.10 * X = -500) → X = 25000 :=
by
  intro h
  -- Proof to be completed
  sorry

end total_worth_of_stock_l396_396001


namespace triangle_inequality_for_segments_l396_396671

theorem triangle_inequality_for_segments
  (ABC_eq_triangle : EquilateralTriangle ABC)
  (H_angles : ∀. angle(A'B C') > 120 ∧ angle(B' C A') > 120 ∧ angle(C' A B') > 120)
  (H_side_equalities : AB' = AC' ∧ BC' = BA' ∧ CA' = CB')
  : (AB' < BC' + CA') ∧ (BC' < CA' + AB') ∧ (CA' < AB' + BC') := 
sorry

end triangle_inequality_for_segments_l396_396671


namespace divide_polynomials_l396_396676

theorem divide_polynomials (n : ℕ) (h : ∃ (k : ℤ), n^2 + 3*n + 51 = 13 * k) : 
  ∃ (m : ℤ), 21*n^2 + 89*n + 44 = 169 * m := by
  sorry

end divide_polynomials_l396_396676


namespace f_of_g_of_3_eq_10_l396_396651

def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

theorem f_of_g_of_3_eq_10 : f (g 3) = 10 :=
by
  sorry

end f_of_g_of_3_eq_10_l396_396651


namespace square_free_condition_l396_396843

/-- Define square-free integer -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Define the problem in Lean -/
theorem square_free_condition (p : ℕ) (hp : p ≥ 3 ∧ Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q ∧ q < p → square_free (p - (p / q) * q)) ↔
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13 := by
  sorry

end square_free_condition_l396_396843


namespace quadratic_has_two_distinct_real_roots_iff_l396_396957

theorem quadratic_has_two_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6 * x - m = 0 ∧ y^2 - 6 * y - m = 0) ↔ m > -9 :=
by 
  sorry

end quadratic_has_two_distinct_real_roots_iff_l396_396957


namespace factorize_expression_l396_396086

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l396_396086


namespace area_cyclic_quadrilateral_ABCD_l396_396125

-- Define the cyclic quadrilateral with given side lengths
variables (AB BC CD DA : ℝ) (cyclic : AB = 2 ∧ BC = 6 ∧ CD = 4 ∧ DA = 4)

-- Define the problem to prove
theorem area_cyclic_quadrilateral_ABCD (h : cyclic): (area : ℝ) := 
  ∃ A : ℝ, ∃ sinA : ℝ, 
  sinA = Real.sin 120 ∧ -- because we found in the solution that A = 120 degrees
  area = 16 * sinA ∧ -- from the area calculation step
  area = 8 * Real.sqrt 3 := 
sorry

end area_cyclic_quadrilateral_ABCD_l396_396125


namespace angle_measure_x_l396_396210

theorem angle_measure_x
    (angle_CBE : ℝ)
    (angle_EBD : ℝ)
    (angle_ABE : ℝ)
    (sum_angles_TRIA : ∀ a b c : ℝ, a + b + c = 180)
    (sum_straight_ANGLE : ∀ a b : ℝ, a + b = 180) :
    angle_CBE = 124 → angle_EBD = 33 → angle_ABE = 19 → x = 91 :=
by
    sorry

end angle_measure_x_l396_396210


namespace exists_composite_number_among_sequence_l396_396988

theorem exists_composite_number_among_sequence (m n : ℕ) (h_mn_ge_two : m ≥ 2) (h_n_ge_two : n ≥ 2)
    (h_gcd_m_n_minus_one : gcd m (n - 1) = 1) (h_gcd_m_n : gcd m n = 1) :
    ∃ i : ℕ, i < m ∧ ¬ nat.prime (seq_of_a_i m n i) := 
sorry

def seq_of_a_i (m n : ℕ) : ℕ → ℕ
| 0     := mn + 1
| (k+1) := n * (seq_of_a_i k) + 1

end exists_composite_number_among_sequence_l396_396988


namespace smallest_addition_to_palindrome_l396_396498

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_addition_to_palindrome : ∃ k : ℕ, k > 0 ∧ is_palindrome (91237 + k) ∧ (91237 + k).digits.length = 5 ∧ (∀ m, m > 0 ∧ m < k → ¬is_palindrome (91237 + m) ∨ (91237 + m).digits.length ≠ 5) ∧ k = 892 :=
sorry

end smallest_addition_to_palindrome_l396_396498


namespace james_meat_sales_l396_396225

theorem james_meat_sales
  (beef_pounds : ℕ)
  (pork_pounds : ℕ)
  (meat_per_meal : ℝ)
  (meal_price : ℝ)
  (total_meat : ℝ)
  (number_of_meals : ℝ)
  (total_money : ℝ)
  (h1 : beef_pounds = 20)
  (h2 : pork_pounds = beef_pounds / 2)
  (h3 : meat_per_meal = 1.5)
  (h4 : meal_price = 20)
  (h5 : total_meat = beef_pounds + pork_pounds)
  (h6 : number_of_meals = total_meat / meat_per_meal)
  (h7 : total_money = number_of_meals * meal_price) :
  total_money = 400 := by
  sorry

end james_meat_sales_l396_396225


namespace triangle_area_proof_l396_396601

def triangle_area (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

theorem triangle_area_proof :
  ∀ (a b c : ℝ) (C : ℝ),
    a = 1 →
    c = 2 →
    Real.cos C = 0.25 →
    b = 2 →
      triangle_area a b C = (Real.sqrt 15) / 4 :=
begin
  intros a b c C ha hb hc h_cos,
  sorry
end

end triangle_area_proof_l396_396601


namespace sam_age_l396_396294

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l396_396294


namespace parallel_symmetry_l396_396519

theorem parallel_symmetry (ABC : Triangle)
    (scalene : ABC.isScalene) 
    (A1 B1 C1 : Point)
    (incircle_tangency : ABC.incircle.tangentPoints = (A1, B1, C1))
    (A2 B2 C2 : Point)
    (symmetry_angles : A2 = reflect_point A1 (bisector ABC.∠A) ∧ B2 = reflect_point B1 (bisector ABC.∠B) ∧ C2 = reflect_point C1 (bisector ABC.∠C)):
    parallel A2 C2 ABC.AC :=
by
  sorry

end parallel_symmetry_l396_396519


namespace geometric_sum_4_terms_l396_396974

theorem geometric_sum_4_terms 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 243) 
  (hq : ∀ n, a (n + 1) = a n * q) 
  : a 1 * (1 - q^4) / (1 - q) = 120 := 
sorry

end geometric_sum_4_terms_l396_396974


namespace expected_value_sum_expected_value_diff_l396_396893

noncomputable def expected_value (X : Type) [Add X] [Sub X] [Zero X] : X → ℝ := sorry 

variables (X Y : ℝ)

-- Conditions
axiom expected_value_X : expected_value X = 3
axiom expected_value_Y : expected_value Y = 2
axiom linearity_of_expectation_sum : ∀ X Y, expected_value (X + Y) = expected_value X + expected_value Y
axiom linearity_of_expectation_diff : ∀ X Y, expected_value (X - Y) = expected_value X - expected_value Y

-- Theorem statements
theorem expected_value_sum : expected_value (X + Y) = 5 := by
  apply linearity_of_expectation_sum
  rw [expected_value_X, expected_value_Y]
  sorry

theorem expected_value_diff : expected_value (X - Y) = 1 := by
  apply linearity_of_expectation_diff
  rw [expected_value_X, expected_value_Y]
  sorry

end expected_value_sum_expected_value_diff_l396_396893


namespace pleasant_goat_paths_l396_396015

-- Define the grid points A, B, and C
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := { x := 0, y := 0 }
def C : Point := { x := 3, y := 3 }  -- assuming some grid layout
def B : Point := { x := 1, y := 1 }

-- Define a statement to count the number of shortest paths
def shortest_paths_count (A B C : Point) : ℕ := sorry

-- Proving the shortest paths from A to C avoiding B is 81
theorem pleasant_goat_paths : shortest_paths_count A B C = 81 := 
sorry

end pleasant_goat_paths_l396_396015


namespace B_pow_5_eq_rB_plus_sI_B_pow_5_spec_l396_396993

noncomputable theory
open Matrix

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ! [![2, 3], ![4, -1]]

def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem B_pow_5_eq_rB_plus_sI :
  ∃ r s : ℝ, (B ^ 5) = r • B + s • I := by
  sorry

theorem B_pow_5_spec :
  (∃ r s : ℝ, B ^ 5 = r • B + s • I) → (B ^ 5 = 1125 • B + 1243 • I) := by
  intro h
  cases h with r hs
  cases hs with s hs
  exact hs.symm

end B_pow_5_eq_rB_plus_sI_B_pow_5_spec_l396_396993


namespace count_paths_l396_396915

theorem count_paths (m n : ℕ) : (n + m).choose m = (n + m).choose n :=
by
  sorry

end count_paths_l396_396915


namespace sum_of_squared_residuals_eq_zero_correlation_coefficient_eq_one_or_neg_one_l396_396594

variables {n : ℕ}
variables {x y : fin n → ℝ} -- Sample points (x_i, y_i)

noncomputable def regression_line (x y : fin n → ℝ) : fin n → ℝ :=
  let b := ((finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x) * (y i - finset.univ.mean y))) /
            (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x)^2))) in
  let a := finset.univ.mean y - b * finset.univ.mean x in
  λ i, b * x i + a

def sum_squared_residuals (x y : fin n → ℝ) : ℝ :=
  finset.univ.sum (λ i : fin n, (y i - regression_line x y i)^2)

def correlation_coefficient (x y : fin n → ℝ) : ℝ :=
  let num := finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x) * (y i - finset.univ.mean y)) in
  let den_x := finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x)^2) in
  let den_y := finset.univ.sum (λ i : fin n, (y i - finset.univ.mean y)^2) in
  num / (den_x.sqrt * den_y.sqrt)

theorem sum_of_squared_residuals_eq_zero (x y : fin n → ℝ) :
  (∀ i, y i = regression_line x y i) →
  sum_squared_residuals x y = 0 :=
by
  intro h
  have : ∀ i, y i - regression_line x y i = 0 := 
    by intro i; rw h i
  simp_rw [this, sub_self, pow_two, zero_mul, finset.sum_const_zero]

theorem correlation_coefficient_eq_one_or_neg_one (x y : fin n → ℝ) (hx : ∀ i, y i = regression_line x y i) :
  correlation_coefficient x y = 1 ∨ correlation_coefficient x y = -1 :=
by
  have hssr : sum_squared_residuals x y = 0 :=
    sum_of_squared_residuals_eq_zero x y hx
  calc
    correlation_coefficient x y
        = (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x) * (y i - finset.univ.mean y))) / (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x)^2)).sqrt : sorry -- Definition of correlation_coefficient
    ... = (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x) * regression_line x y i)) / (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x)^2)).sqrt : sorry -- Using h
    ... = ((finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x) * (∑ x y))) / (finset.univ.sum (λ i : fin n, (x i - finset.univ.mean x)^2)).sqrt : sorry -- From regression line relation
    ... = 1 ∨ ... = -1 : sorry -- Using the given formula and simplifying


end sum_of_squared_residuals_eq_zero_correlation_coefficient_eq_one_or_neg_one_l396_396594


namespace smallest_magnitude_z_theorem_l396_396657

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem smallest_magnitude_z_theorem : 
  ∃ z : ℂ, (Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) ∧
  smallest_magnitude_z z = 36 / Real.sqrt 97 := 
sorry

end smallest_magnitude_z_theorem_l396_396657


namespace irreducible_fraction_l396_396111

theorem irreducible_fraction (n : ℕ) :
  irreducible (2 * n^2 + 11 * n - 18) (n + 7) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
sorry

end irreducible_fraction_l396_396111


namespace geometric_sum_eval_l396_396542

noncomputable def f (n : ℕ) := ∑ i in Finset.range (n+1), 2^(3 * i + 1)

theorem geometric_sum_eval (n : ℕ) : f(n) = (2 / 7) * (8^(n + 1) - 1) :=
by
  let f := (n : ℕ) := ∑ i in Finset.range (n+1), 2^(3 * i + 1)
  sorry


end geometric_sum_eval_l396_396542


namespace mod_inverse_sum_l396_396829

theorem mod_inverse_sum :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 1 := by
  sorry

end mod_inverse_sum_l396_396829


namespace somu_fathers_age_ratio_l396_396305

noncomputable def somus_age := 16

def proof_problem (S F : ℕ) : Prop :=
  S = 16 ∧ 
  (S - 8 = (1 / 5) * (F - 8)) ∧
  (S / F = 1 / 3)

theorem somu_fathers_age_ratio (S F : ℕ) : proof_problem S F :=
by
  sorry

end somu_fathers_age_ratio_l396_396305


namespace smallest_multiple_6_15_l396_396103

theorem smallest_multiple_6_15 (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) :
  ∃ (b : ℕ), (b > 0) ∧ (b % 6 = 0) ∧ (b % 15 = 0) ∧ (∀ x : ℕ, (x > 0) ∧ (x % 6 = 0) ∧ (x % 15 = 0) → x ≥ b) :=
sorry

end smallest_multiple_6_15_l396_396103


namespace find_a4_l396_396647

theorem find_a4 (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n - 3) : a 4 = -8 :=
by {
  sorry
}

end find_a4_l396_396647


namespace binomial_22_5_computation_l396_396145

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end binomial_22_5_computation_l396_396145


namespace complex_pair_solutions_l396_396100

theorem complex_pair_solutions :
  ∃ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 ∧ (∃ n : ℕ, n ∈ finset.range 24 ∧ a = exp(2 * π * I * n / 24) ∧ b = exp(-16 * π * I * n / 24)) :=
by sorry

end complex_pair_solutions_l396_396100


namespace final_answer_l396_396652

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y : ℝ, g (g x + y) = g (x^3 - y) + 6 * g x * y

def possible_values_of_g2 : Finset ℝ := {0, 8}

def t : ℝ := possible_values_of_g2.sum id

def n : ℕ := possible_values_of_g2.card

theorem final_answer : n * t = 16 := 
by 
  -- Definition and calculation are done according to given problem and solution
  sorry

end final_answer_l396_396652


namespace exists_x0_f_x0_add_f_double_prime_x0_eq_zero_l396_396249

variable (f : ℝ → ℝ)
variable [IsTwiceDifferentiable f]
variable (h1 : ∀ x, f x ≥ -1 ∧ f x ≤ 1) -- f : ℝ → [-1,1]
variable (h2 : f 0 ^ 2 + (deriv f 0) ^ 2 = 4) -- f(0)^2 + f'(0)^2 = 4

theorem exists_x0_f_x0_add_f_double_prime_x0_eq_zero :
  ∃ x0 : ℝ, f x0 + deriv^2 f x0 = 0 := by
  sorry

end exists_x0_f_x0_add_f_double_prime_x0_eq_zero_l396_396249


namespace coeff_x21_is_385_l396_396617

-- Define the polynomials
def P1 (x : ℕ → ℕ → ℤ) := (1 - x^21) / (1 - x)
def P2 (x : ℕ → ℕ → ℤ) := (1 - x^11)^2 / (1 - x)^2
def series_sum (x : ℕ → ℕ → ℕ) := ∑ k in range (k+2), nat.choose (k+2) 2 * x^k

-- Define the main function that calculates the coefficient
def coefficient_x21 : ℕ := 
  let term1 : ℕ := (nat.choose 23 2);  -- coefficient for x^21
  let term2 : ℕ := 2 * (nat.choose 12 2); -- two ways to get x^10 * x^11
  let term3 : ℕ := 0; -- coefficient for x^{-1} is 0
  term1 + term2 + term3 -- summing all contributions

-- Prove that the coefficient of x^{21} is 385
theorem coeff_x21_is_385 : coefficient_x21 = 385 := 
  by 
  {
    -- Unfold and apply the definitions
    unfold coefficient_x21,
    unfold let term1,
    unfold let term2,
    unfold let term3,
    -- The actual calculation
    change 253 + 132 + 0 = 385,
    -- Verified by reasoning
    refl
  }

end coeff_x21_is_385_l396_396617


namespace three_digit_cubes_divisible_by_16_l396_396581

theorem three_digit_cubes_divisible_by_16 :
  (count (λ n : ℕ, 4 * n = n ∧ (100 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 999)) {n | 1 ≤ n ∧ n ≤ 2}) = 1 :=
sorry

end three_digit_cubes_divisible_by_16_l396_396581


namespace range_sin_alpha_plus_sin_beta_l396_396588

theorem range_sin_alpha_plus_sin_beta (α β : ℝ)
  (h : cos α + sin β = 1 / 2) :
  set.range (λ α β, sin α + sin β) = set.Icc (1 / 2 - real.sqrt 2) (1 + real.sqrt 3 / 2) :=
sorry

end range_sin_alpha_plus_sin_beta_l396_396588


namespace bus_speed_excluding_stoppages_l396_396084

variable (v : ℝ) -- Speed of the bus excluding stoppages

-- Conditions
def bus_stops_per_hour := 45 / 60 -- 45 minutes converted to hours
def effective_driving_time := 1 - bus_stops_per_hour -- Effective time driving in an hour

-- Given Condition
def speed_including_stoppages := 12 -- Speed including stoppages in km/hr

theorem bus_speed_excluding_stoppages 
  (h : effective_driving_time * v = speed_including_stoppages) : 
  v = 48 :=
sorry

end bus_speed_excluding_stoppages_l396_396084


namespace binomial_7_2_eq_21_l396_396047

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396047


namespace square_side_is_8_l396_396701

-- Definitions based on problem conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 16
def rectangle_area : ℝ := rectangle_width * rectangle_length

def square_side_length (s : ℝ) : Prop := s^2 = rectangle_area

-- The theorem we need to prove
theorem square_side_is_8 (s : ℝ) : square_side_length s → s = 8 := by
  -- Proof to be filled in
  sorry

end square_side_is_8_l396_396701


namespace three_digit_multiples_of_3_and_11_l396_396177

theorem three_digit_multiples_of_3_and_11 : 
  ∃ n, n = 27 ∧ ∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 33 = 0 ↔ ∃ k, x = 33 * k ∧ 4 ≤ k ∧ k ≤ 30 :=
by
  sorry

end three_digit_multiples_of_3_and_11_l396_396177


namespace coby_speed_idaho_nevada_l396_396460

def coby_travel (Distance_WI Distance_IN Speed_WI Total_Time : ℕ) : Prop :=
  let Time_WI := Distance_WI / Speed_WI in
  let Time_IN := Total_Time - Time_WI in
  let Speed_IN := Distance_IN / Time_IN in
  Speed_IN = 50

theorem coby_speed_idaho_nevada :
  coby_travel 640 550 80 19 :=
by
  unfold coby_travel
  sorry

end coby_speed_idaho_nevada_l396_396460


namespace natural_numbers_divisible_by_6_l396_396127

theorem natural_numbers_divisible_by_6 :
  {n : ℕ | 2 ≤ n ∧ n ≤ 88 ∧ 6 ∣ n} = {n | n = 6 * k ∧ 1 ≤ k ∧ k ≤ 14} :=
by
  sorry

end natural_numbers_divisible_by_6_l396_396127


namespace work_alone_days_l396_396386

theorem work_alone_days (d : ℝ) (p q : ℝ) (h1 : q = 10) (h2 : 2 * (1/d + 1/q) = 0.3) : d = 20 :=
by
  sorry

end work_alone_days_l396_396386


namespace problem_1_problem_2_l396_396520
open BigOperators

-- Define the sequence {a_n} and the sequence condition S_n = 2a_n - n
def S (n : ℕ) : ℕ := 2 * a n - n
def a : ℕ → ℕ
| 1     := 1
| (n+2) := 2 * a (n + 1) + 1

-- 1. Prove that {a_n + 1} is a geometric sequence
theorem problem_1 : ∀ n ≥ 1, is_geometric_sequence (λ n, a n + 1) :=
by sorry

-- 2. Given b_n = 2^n / (a_n * a_(n+1)), find the sum T_n
def b (n : ℕ) : ℚ := (2 ^ n : ℚ) / ((a n) * (a (n + 1)))

def T (n : ℕ) : ℚ := ∑ i in finset.range n, b i

theorem problem_2 : ∀ n : ℕ, T n = 1 - 1 / (2 ^ (n + 1) - 1) :=
by sorry

end problem_1_problem_2_l396_396520


namespace extreme_point_at_1_l396_396545

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (2 * a^3 - a^2) * Real.log x - (a^2 + 2 * a - 1) * x

theorem extreme_point_at_1 (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ ∀ x > 0, deriv (f a) x = 0 →
  a = -1) := sorry

end extreme_point_at_1_l396_396545


namespace books_count_l396_396351

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end books_count_l396_396351


namespace domain_of_f_l396_396709

-- Define the function f
def f (x : ℝ) : ℝ := Real.sqrt (Real.log (3 - x) / Real.log (1/2))

-- Define the condition for the domain
def condition (x : ℝ) : Prop := 0 < 3 - x ∧ 3 - x <= 1

-- State the theorem about the domain of the function
theorem domain_of_f : {x : ℝ | condition x} = set.Ico 2 3 := by
  sorry

end domain_of_f_l396_396709


namespace value_of_f_2_3_l396_396830

noncomputable def f (x y : ℝ) : ℝ := sorry

theorem value_of_f_2_3 (f : ℝ → ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x, y) + 3 * f(8 - x, 4 - y) = x + y) :
  f 2 3 = 2 :=
sorry

end value_of_f_2_3_l396_396830


namespace each_child_play_time_l396_396814

theorem each_child_play_time (n_children : ℕ) (game_time : ℕ) (children_per_game : ℕ)
  (h1 : n_children = 8) (h2 : game_time = 120) (h3 : children_per_game = 2) :
  ((children_per_game * game_time) / n_children) = 30 :=
  sorry

end each_child_play_time_l396_396814


namespace an_general_formula_bn_sum_formula_l396_396525

noncomputable def an_formula : ℕ → ℤ :=
  assume (n : ℕ), 2 * n - 12

theorem an_general_formula (n : ℕ) :
  ∀ n, (∃ d : ℤ, ∃ a_1 : ℤ, an_formula 3 = a_1 + 2 * d ∧ an_formula 6 = a_1 + 5 * d) →
    an_formula n = 2 * n - 12 :=
by sorry

noncomputable def bn_formula (n : ℕ) : ℤ :=
  -8 * n

theorem bn_sum_formula (n : ℕ) :
  ∀ n, (∃ a_1 a_2 a_3 : ℤ, an_formula 1 = a_1 ∧ an_formula 2 = a_2 ∧ an_formula 3 = a_3 ∧ bn_formula 1 = -8 ∧ bn_formula 2 = a_1 + a_2 + a_3) →
    bn_formula n = -8 * n :=
by sorry

end an_general_formula_bn_sum_formula_l396_396525


namespace ratio_problem_l396_396945

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396945


namespace Sam_age_l396_396288

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l396_396288


namespace savings_by_end_of_2019_l396_396631

variable (income_monthly : ℕ → ℕ) (expenses_monthly : ℕ → ℕ)
variable (initial_savings : ℕ)

noncomputable def total_income : ℕ :=
  (income_monthly 9 + income_monthly 10 + income_monthly 11 + income_monthly 12) * 4

noncomputable def total_expenses : ℕ :=
  (expenses_monthly 9 + expenses_monthly 10 + expenses_monthly 11 + expenses_monthly 12) * 4

noncomputable def final_savings (initial_savings : ℕ) (total_income : ℕ) (total_expenses : ℕ) : ℕ :=
  initial_savings + total_income - total_expenses

theorem savings_by_end_of_2019 :
  (income_monthly 9 = 55000) →
  (income_monthly 10 = 45000) →
  (income_monthly 11 = 10000) →
  (income_monthly 12 = 17400) →
  (expenses_monthly 9 = 40000) →
  (expenses_monthly 10 = 20000) →
  (expenses_monthly 11 = 5000) →
  (expenses_monthly 12 = 2000) →
  initial_savings = 1147240 →
  final_savings initial_savings total_income total_expenses = 1340840 :=
by
  intros h_income_9 h_income_10 h_income_11 h_income_12
         h_expenses_9 h_expenses_10 h_expenses_11 h_expenses_12
         h_initial_savings
  rw [final_savings, total_income, total_expenses]
  rw [h_income_9, h_income_10, h_income_11, h_income_12]
  rw [h_expenses_9, h_expenses_10, h_expenses_11, h_expenses_12]
  rw h_initial_savings
  sorry

end savings_by_end_of_2019_l396_396631


namespace youngest_and_oldest_sum_l396_396982

theorem youngest_and_oldest_sum (ages : Fin 6 → ℕ) 
  (h_mean : (∑ i, ages i) = 60) 
  (h_median : (ages 2 + ages 3) / 2 = 11) :
  ages 0 + ages 5 = 38 :=
sorry

end youngest_and_oldest_sum_l396_396982


namespace find_second_discount_l396_396331

theorem find_second_discount 
    (list_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (h₁ : list_price = 65)
    (h₂ : final_price = 57.33)
    (h₃ : first_discount = 0.10)
    (h₄ : (list_price - (first_discount * list_price)) = 58.5)
    (h₅ : final_price = 58.5 - (second_discount * 58.5)) :
    second_discount = 0.02 := 
by
  sorry

end find_second_discount_l396_396331


namespace binomial_7_2_l396_396064

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396064


namespace diagonals_in_25_sided_polygon_l396_396822

/-
  Proof Problem: Prove that the number of diagonals in a convex polygon with 25 sides is 275.
-/

theorem diagonals_in_25_sided_polygon : ∀ (n : ℕ), n = 25 → (n * (n - 3)) / 2 = 275 :=
by
  intros n h
  rw h
  sorry

end diagonals_in_25_sided_polygon_l396_396822


namespace Marc_watch_episodes_l396_396267

theorem Marc_watch_episodes : ∀ (episodes per_day : ℕ), episodes = 50 → per_day = episodes / 10 → (episodes / per_day) = 10 :=
by
  intros episodes per_day h1 h2
  sorry

end Marc_watch_episodes_l396_396267


namespace ratio_problem_l396_396937

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396937


namespace pqrs_sum_eq_236_l396_396499

theorem pqrs_sum_eq_236 :
  ∃ (x y : ℝ) (p q r s : ℕ), x + y = 5 ∧ x * y = 6 ∧ x = (p + q * Real.sqrt r) / s ∧
  p + q + r + s = 236 := by
  sorry

end pqrs_sum_eq_236_l396_396499


namespace angle_A_is_pi_div_3_triangle_area_l396_396148

-- Define the conditions of the triangle and the given equations
variables (A B C a b c : ℝ)
variables (h_bc : (b - c) ^ 2 = a ^ 2 - b * c)
variables (h_a : a = 2) (h_sinC : sin C = 2 * sin B)

-- Prove the measure of angle A is π / 3
theorem angle_A_is_pi_div_3 (h_bc : (b - c) ^ 2 = a ^ 2 - b * c) : A = π / 3 := 
sorry

-- Prove the area of the triangle
theorem triangle_area (h_a : a = 2) 
  (h_sinC : sin C = 2 * sin B) 
  (h_A : A = π / 3) : 
  let b := sqrt (4 / 3),
      c := 2 * b in
  (1 / 2) * b * c * sin A = (2 * sqrt 3) / 3 := 
sorry

end angle_A_is_pi_div_3_triangle_area_l396_396148


namespace isosceles_triangle_existence_l396_396472

theorem isosceles_triangle_existence
  (a b m_a m_b k1 k2 k3 k4 : ℝ)
  (h1 : a + b = k1)
  (h2 : m_a + m_b = k2)
  (h3 : b - a = k3)
  (h4 : m_a - m_b = k4)
  (h5 : m_a * a = m_b * b) :
  ∃ (A B C : ℝ × ℝ), 
    (let Δ := {A, B, C} in 
     ((dist A B = dist A C ∧ dist B C > 0) ∨ 
      (dist A C = dist B C ∧ dist A B > 0))) :=
sorry

end isosceles_triangle_existence_l396_396472


namespace find_x_l396_396625

def sequence (a : ℕ → ℤ) : Prop :=
  (a 5 = 2) ∧ (a 6 = 5) ∧ (a 7 = 7) ∧ (a 8 = 12) ∧ (a 9 = 19) ∧ (a 10 = 31) ∧
  (∀ n, n ≥ 5 → a (n + 1) = a n + a (n - 1) + a (n - 2))

theorem find_x (a : ℕ → ℤ) (h : sequence a) : 
  a 3 + a 4 + a 5 = a 6 → a 2 + a 3 + a 4 = a 5 → a 1 + a 2 + a 3 = a 4 → a 0 + a 1 + a 2 = a 3 →
  a 0 = -13 :=
by
  intro h1 h2 h3 h4
  sorry

end find_x_l396_396625


namespace three_digit_cubes_divisible_by_16_l396_396574

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l396_396574


namespace mary_initials_l396_396667

theorem mary_initials (total_characters : ℕ)
  (characters_initial_A : total_characters / 3)
  (characters_initial_B : (total_characters - characters_initial_A) / 4)
  (characters_initial_C : ((total_characters - characters_initial_A) - characters_initial_B) / 5)
  (remaining_characters := total_characters - characters_initial_A - characters_initial_B - characters_initial_C)
  (E F D : ℕ)
  (H1 : D = 3 * E)
  (H2 : E = F / 2)
  (H3 : D + E + F = remaining_characters) :
  D = 24 := by
  sorry

end mary_initials_l396_396667


namespace line_perpendicular_to_plane_l396_396596

theorem line_perpendicular_to_plane (e : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) :
  e = (2, 3, -1) ∧ n = (-1, -3 / 2, 1 / 2) → 
  (e.1 * n.1 + e.2 * n.2 + e.3 * n.3 = 0) :=
by
  intros h
  cases h
  simp [h_left, h_right]
  sorry

end line_perpendicular_to_plane_l396_396596


namespace part1_solution_set_part2_range_of_a_l396_396168

noncomputable def f (x : ℝ) : ℝ := abs (4 * x - 1) - abs (x + 2)

-- Part 1: Prove the solution set of f(x) < 8 is -9 / 5 < x < 11 / 3
theorem part1_solution_set : {x : ℝ | f x < 8} = {x : ℝ | -9 / 5 < x ∧ x < 11 / 3} :=
sorry

-- Part 2: Prove the range of a such that the inequality has a solution
theorem part2_range_of_a (a : ℝ) : (∃ x : ℝ, f x + 5 * abs (x + 2) < a^2 - 8 * a) ↔ (a < -1 ∨ a > 9) :=
sorry

end part1_solution_set_part2_range_of_a_l396_396168


namespace isosceles_right_triangle_square_ratio_l396_396831

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := Real.sqrt 2 / 2

theorem isosceles_right_triangle_square_ratio :
  x / y = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_square_ratio_l396_396831


namespace range_of_a_l396_396131

noncomputable def is_decreasing (f : ℕ+ → ℝ) : Prop :=
  ∀ (n m : ℕ+), n < m → f n > f m

def sequence_a_n (a : ℝ) : ℕ+ → ℝ
| ⟨n, hn⟩ := if n ≤ 6 then (1 - 3 * a) * ↑n + 10 * a else a ^ (↑n - 7)

theorem range_of_a (a : ℝ) (h : is_decreasing (sequence_a_n a)) : (1 / 3) < a ∧ a < (5 / 8) :=
sorry

end range_of_a_l396_396131


namespace count_permutation_multiples_of_11_l396_396912

-- Define a predicate to check if a number is within the range and some permutation is multiple of 11
def permutation_is_multiple_of_11 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (perm : ℕ), perm ∈ (Nat.permutations n) ∧ perm % 11 = 0

-- Main theorem statement
theorem count_permutation_multiples_of_11 :
  ∃ (count : ℕ), count = 16200 ∧ ∀ n, permutation_is_multiple_of_11 n → True :=
sorry

end count_permutation_multiples_of_11_l396_396912


namespace prob_subscribe_newspaper_A_prob_subscribe_newspaper_B_prob_not_subscribe_either_l396_396962

theorem prob_subscribe_newspaper_A (families_selected : ℕ) (prob_subscribe_A : ℝ) (prob_not_subscribe_A : ℝ) : 
  (families_selected = 4) → 
  (prob_subscribe_A = 0.3) → 
  (prob_not_subscribe_A = 0.7) → 
  P(exactly 3 of these 4 families subscribe to Newspaper A) = 0.0756 :=
by
  intros
  sorry

theorem prob_subscribe_newspaper_B (families_selected : ℕ) (prob_subscribe_B : ℝ) : 
  (families_selected = 4) → 
  (prob_subscribe_B = 0.6) → 
  P(at most 3 of these 4 families subscribe to Newspaper B) = 0.8704 :=
by
  intros
  sorry

theorem prob_not_subscribe_either (families_selected : ℕ) (prob_not_subscribe : ℝ) : 
  (families_selected = 4) → 
  (prob_not_subscribe = 0.3) → 
  P(exactly 2 of these 4 families do not subscribe to either Newspaper A or B) = 0.2646 :=
by
  intros
  sorry

end prob_subscribe_newspaper_A_prob_subscribe_newspaper_B_prob_not_subscribe_either_l396_396962


namespace value_of_expression_l396_396924

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396924


namespace pieces_in_grid_l396_396777

theorem pieces_in_grid (n : ℕ) (h : n ≥ 2) (k : ℕ) 
  (cond : ∀ subgrid : fin 2 × fin 2, 2 × 2 subgrid contains exactly 2 pieces):
  k = 0 ∨ k = (n * n) / 2 :=
by sorry

end pieces_in_grid_l396_396777


namespace triangle_properties_l396_396220

noncomputable theory

variables (a b c : ℝ)
variables (A B C : ℝ)
variable (R : ℝ) -- circumradius
variables (m n : ℝ × ℝ)

-- Conditions
def vector_m := (a / 2, c / 2)
def vector_n := (Real.cos C, Real.cos A)
def dot_product_condition := vector_n.1 * vector_m.1 + vector_n.2 * vector_m.2 = b * Real.cos B
def cos_frac_condition := Real.cos ((A - C) / 2) = sqrt 3 * Real.sin A
def magnitude_m := sqrt (a ^ 2 / 4 + c ^ 2 / 4) = sqrt 5

-- Proof Statement
theorem triangle_properties (h_dot : dot_product_condition)
    (h_cos : cos_frac_condition)
    (h_mag : magnitude_m)
: B = Real.pi / 3 ∧ (1/2) * a * b = 2 * sqrt 3 := sorry

end triangle_properties_l396_396220


namespace combination_7_2_l396_396033

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396033


namespace sum_of_last_two_digits_of_fibonacci_factorial_series_l396_396374

def last_two_digits (n : Nat) : Nat :=
  n % 100

def relevant_factorials : List Nat := [
  last_two_digits (Nat.factorial 1),
  last_two_digits (Nat.factorial 1),
  last_two_digits (Nat.factorial 2),
  last_two_digits (Nat.factorial 3),
  last_two_digits (Nat.factorial 5),
  last_two_digits (Nat.factorial 8)
]

def sum_last_two_digits : Nat :=
  relevant_factorials.sum

theorem sum_of_last_two_digits_of_fibonacci_factorial_series :
  sum_last_two_digits = 5 := by
  sorry

end sum_of_last_two_digits_of_fibonacci_factorial_series_l396_396374


namespace marc_watching_episodes_l396_396266

theorem marc_watching_episodes :
  ∀ (n : ℕ) (f : ℝ),
  n = 50 → f = 1/10 → 
  n / (n * f) = 10 := 
by
  intro n f hn hf
  rw [hn, hf]
  norm_num
  sorry

end marc_watching_episodes_l396_396266


namespace halve_second_column_l396_396098

-- Definitions of given matrices
variable (f g h i : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := ![![f, g], ![h, i]])
variable (N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, (1/2)]])

-- Proof statement to be proved
theorem halve_second_column (hf : f ≠ 0) (hh : h ≠ 0) : N * A = ![![f, (1/2) * g], ![h, (1/2) * i]] := by
  sorry

end halve_second_column_l396_396098


namespace irreducible_fraction_l396_396108

theorem irreducible_fraction (n : ℕ) : 
  irreducible (2 * n^2 + 11 * n - 18) (n + 7) ↔ 
    (n % 3 = 0) ∨ (n % 3 = 1) :=
sorry

end irreducible_fraction_l396_396108


namespace binomial_7_2_eq_21_l396_396044

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396044


namespace parabola_properties_l396_396330

-- Given conditions
theorem parabola_properties :
  let parabola : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ y^2 = 3 * x}
  let focus : ℝ × ℝ := (3 / 4, 0)
  let directrix : ℝ -> Prop := λ x, x = -3 / 4
  let line_through_focus (m : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x = m * y + 3 / 4}
  let A : ℝ × ℝ := (/* x1 to be determined from parabola and line intersection */, /* y1 where y1 > 0 */)
  let B : ℝ × ℝ := (/* x2 to be determined from parabola and line intersection */, /* y2 where y2 < 0 */)

  -- Options to be proven
  (angle_A1FB_90 : ∃ A1 B1 : ℝ × ℝ, A1.1 = -3 / 4 ∧ A1.2 = A.2 ∧ B1.1 = -3 / 4 ∧ B1.2 = B.2 ∧ (∠ A1 F B1 = 90)) ∧
  (line_MB_parallel_x : ∃ M : ℝ × ℝ, M.1 = -3 / 4 ∧ A.1 = M.1 ∧ line_through_focus(slope_between M B) = 0) ∧
  (min_value_AF_BF : ∀ AF BF : ℝ, |AF| * |BF| = 9 / 4)
  :=
sorry

end parabola_properties_l396_396330


namespace ratio_expression_value_l396_396921

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396921


namespace consecutive_numbers_even_count_l396_396684

def percentage_of_evens (n : ℕ) := 13 * n / 25

theorem consecutive_numbers_even_count (n : ℕ) (h1 : 52 * n / 100 = percentage_of_evens n) :
    percentage_of_evens 25 = 13 := 
begin
    sorry
end

end consecutive_numbers_even_count_l396_396684


namespace prove_statement_II_must_be_true_l396_396395

-- Definitions of the statements
def statement_I (d : ℕ) : Prop := d = 5
def statement_II (d : ℕ) : Prop := d ≠ 6
def statement_III (d : ℕ) : Prop := d = 7
def statement_IV (d : ℕ) : Prop := d ≠ 8

-- Condition: Exactly three of these statements are true and one is false
def exactly_three_true (P Q R S : Prop) : Prop :=
  (P ∧ Q ∧ R ∧ ¬S) ∨ (P ∧ Q ∧ ¬R ∧ S) ∨ (P ∧ ¬Q ∧ R ∧ S) ∨ (¬P ∧ Q ∧ R ∧ S)

-- Problem statement
theorem prove_statement_II_must_be_true (d : ℕ) (h : exactly_three_true (statement_I d) (statement_II d) (statement_III d) (statement_IV d)) : 
  statement_II d :=
by
  -- proof goes here
  sorry

end prove_statement_II_must_be_true_l396_396395


namespace consecutive_even_numbers_l396_396685

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end consecutive_even_numbers_l396_396685


namespace harkamal_total_payment_l396_396559

-- Define the prices of different fruits per kg
def price_of_grapes_per_kg: ℝ := 70
def price_of_mangoes_per_kg: ℝ := 55
def price_of_apples_per_kg: ℝ := 40
def price_of_oranges_per_kg: ℝ := 30

-- Define the quantity of each fruit purchased
def quantity_of_grapes: ℝ := 9
def quantity_of_mangoes: ℝ := 9
def quantity_of_apples: ℝ := 5
def quantity_of_oranges: ℝ := 6

-- Define the discount rate for grapes and apples
def discount_rate: ℝ := 0.10

-- Define the sales tax rate
def sales_tax_rate: ℝ := 0.06

-- Define the total amount Harkamal has to pay (which we need to prove)
def total_amount_to_pay: ℝ := 1507.32

theorem harkamal_total_payment :
  let total_price := (quantity_of_grapes * price_of_grapes_per_kg) +
                     (quantity_of_mangoes * price_of_mangoes_per_kg) +
                     (quantity_of_apples * price_of_apples_per_kg) +
                     (quantity_of_oranges * price_of_oranges_per_kg) in
  let discount := (quantity_of_grapes * price_of_grapes_per_kg * discount_rate) +
                  (quantity_of_apples * price_of_apples_per_kg * discount_rate) in
  let price_after_discount := total_price - discount in
  let sales_tax := price_after_discount * sales_tax_rate in
  let final_price := price_after_discount + sales_tax in
  final_price = total_amount_to_pay := by 
    sorry

end harkamal_total_payment_l396_396559


namespace total_weight_of_canoe_l396_396275

-- Define conditions as constants
constant total_people_capacity : ℕ := 6
constant proportion_with_dog : ℝ := 2/3
constant weight_per_person : ℝ := 140
constant dog_weight_proportion : ℝ := 1/4

-- Define the theorem to prove
theorem total_weight_of_canoe : 
  let number_of_people_with_dog := proportion_with_dog * total_people_capacity
  let total_people_weight := number_of_people_with_dog * weight_per_person
  let dog_weight := dog_weight_proportion * weight_per_person
  total_people_weight + dog_weight = 595 := 
by 
  sorry

end total_weight_of_canoe_l396_396275


namespace log2_6_gt_2_sqrt_5_l396_396462

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l396_396462


namespace nancy_small_gardens_l396_396271

theorem nancy_small_gardens (total_seeds big_garden_seeds small_garden_seed_count : ℕ) 
    (h1 : total_seeds = 52) 
    (h2 : big_garden_seeds = 28) 
    (h3 : small_garden_seed_count = 4) : 
    (total_seeds - big_garden_seeds) / small_garden_seed_count = 6 := by 
    sorry

end nancy_small_gardens_l396_396271


namespace jogging_track_circumference_l396_396723

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end jogging_track_circumference_l396_396723


namespace greatest_perimeter_l396_396226

def isosceles_triangle (base height : ℝ) : Prop :=
  ∃ a b c : ℝ, a = height ∧ b = base / 2 ∧ c^2 = a^2 + b^2

def divide_base (base : ℝ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → base / n = 1

def perimeter (k : ℕ) (height : ℝ) : ℝ :=
  1 + Real.sqrt (height^2 + k^2) + Real.sqrt (height^2 + (k + 1)^2)

theorem greatest_perimeter :
  isosceles_triangle 10 12 →
  divide_base 10 10 →
  ∃ k : ℕ, k < 5 ∧ Real.floor (perimeter k 12 * 100) / 100 = 26.65 :=
by
  intros h_triangle h_division
  sorry

end greatest_perimeter_l396_396226


namespace erica_income_l396_396481

def fish_price_per_kg : ℝ := 20
def fish_trawled_past_months : ℝ := 80
def fish_trawled_today : ℝ := 2 * fish_trawled_past_months
def total_fish_trawled : ℝ := fish_trawled_past_months + fish_trawled_today
def total_income : ℝ := total_fish_trawled * fish_price_per_kg

theorem erica_income (p : ℝ) (fpast : ℝ) (ftoday : ℝ) (tf : ℝ) (ti : ℝ) :
  p = 20 ∧
  fpast = 80 ∧
  ftoday = 2 * fpast ∧
  tf = fpast + ftoday ∧
  ti = tf * p →
  ti = 4800 :=
by
  intro h
  rcases h with ⟨price_def, past_def, today_def, total_def, income_def⟩
  rw [price_def, past_def, today_def, total_def, income_def]
  norm_num
  sorry

end erica_income_l396_396481


namespace eleven_different_remainders_l396_396414

theorem eleven_different_remainders (A : Fin 100 → Fin 100) (h : ∀ i j : Fin 100, i ≠ j → A i ≠ A j) :
  ∃ r : Finset (Fin 100), r.card ≥ 11 ∧ ∀ n : Fin 100, (∑ i in Finset.range n.succ, A i) % 100 ∈ r :=
  sorry

end eleven_different_remainders_l396_396414


namespace trajectory_equations_perimeter_triangle_MNB_l396_396138

-- Definition of circles and point N
def CircleA (x y : ℝ) := (x + 2)^2 + y^2 = 1
def CircleB (x y : ℝ) := (x - 2)^2 + y^2 = 49
def PointN : ℝ × ℝ := (2, 5 / 3)

-- Trajectory equations of the center of the moving circle P
def ellipse1 (x y : ℝ) := x^2 / 16 + y^2 / 12 = 1
def ellipse2 (x y : ℝ) := x^2 / 9 + y^2 / 5 = 1

-- Perimeter of triangle MNB
def triangle_perimeter (M B N : ℝ × ℝ) : ℝ := 
  let dMN := (M.1 - N.1)^2 + (M.2 - N.2)^2
  let dMB := (M.1 - B.1)^2 + (M.2 - B.2)^2
  let dNB := (N.1 - B.1)^2 + (N.2 - B.2)^2
  sqrt dMN + sqrt dMB + sqrt dNB

-- Theorem statements
theorem trajectory_equations (x y : ℝ) : 
  (CircleA x y ∧ CircleB x y ∧ (exists r, TangentCircleP x y r)) → 
  (ellipse1 x y ∨ ellipse2 x y) :=
sorry

theorem perimeter_triangle_MNB :
  let B := (2, 0) in
  let M := some_point_on_trajectory -- Assume we have M by intersection condition
  ∃ M, triangle_perimeter M B PointN = 16 / 3 :=
sorry

end trajectory_equations_perimeter_triangle_MNB_l396_396138


namespace arrangement_with_one_between_l396_396737

theorem arrangement_with_one_between (people : Finset ℕ) (A B : ℕ) (h_size : people.card = 5) (h_dist : ∀ (arr : List ℕ), (A ≠ B ∧ (arr.indexOf A ∈ (1 :: 2 :: List.nil) → arr.indexOf B = arr.indexOf A + 2 ∨ arr.indexOf B = arr.indexOf A - 2))) : 
  (number_of_arrangements people A B = 36) :=
sorry

end arrangement_with_one_between_l396_396737


namespace value_of_expression_l396_396926

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396926


namespace solution_set_f_pos_l396_396528

-- Definition of odd function and conditions given in the problem
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def f (x : ℝ) : ℝ := if x > 0 then Real.log x / Real.log 2 else if x < 0 then -Real.log (-x) / Real.log 2 else 0

theorem solution_set_f_pos (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  {x : ℝ | f x > 0} = (Set.Ioi 1) ∪ (Set.Iio 0 ∩ (Set.Ici (-1))) :=
by
  sorry

end solution_set_f_pos_l396_396528


namespace height_of_pole_l396_396799

def lean_angle : ℝ := 85
def cable_length_to_base : ℝ := 4
def leah_height : ℝ := 1.75
def leah_walk_distance : ℝ := 3

theorem height_of_pole 
  (lean_angle = 85)
  (cable_length_to_base = 4)
  (leah_height = 1.75)
  (leah_walk_distance = 3) : 
  (height_of_pole = 7) :=
by
  sorry

end height_of_pole_l396_396799


namespace females_with_advanced_degrees_eq_90_l396_396963

-- define the given constants
def total_employees : ℕ := 360
def total_females : ℕ := 220
def total_males : ℕ := 140
def advanced_degrees : ℕ := 140
def college_degrees : ℕ := 160
def vocational_training : ℕ := 60
def males_with_college_only : ℕ := 55
def females_with_vocational_training : ℕ := 25

-- define the main theorem to prove the number of females with advanced degrees
theorem females_with_advanced_degrees_eq_90 :
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 90 :=
by
  sorry

end females_with_advanced_degrees_eq_90_l396_396963


namespace blue_sequins_per_row_l396_396980

theorem blue_sequins_per_row : 
  ∀ (B : ℕ),
  (6 * B) + (5 * 12) + (9 * 6) = 162 → B = 8 :=
by
  intro B
  sorry

end blue_sequins_per_row_l396_396980


namespace find_j_l396_396240

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_j
  (a b c : ℤ)
  (h1 : f a b c 2 = 0)
  (h2 : 200 < f a b c 10 ∧ f a b c 10 < 300)
  (h3 : 400 < f a b c 9 ∧ f a b c 9 < 500)
  (j : ℤ)
  (h4 : 1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1)) :
  j = 36 := sorry

end find_j_l396_396240


namespace maximum_black_squares_l396_396259

theorem maximum_black_squares (n : ℕ) (h : n ≥ 2) : 
  (n % 2 = 0 → ∃ b : ℕ, b = (n^2 - 4) / 2) ∧ 
  (n % 2 = 1 → ∃ b : ℕ, b = (n^2 - 1) / 2) := 
by sorry

end maximum_black_squares_l396_396259


namespace head_start_l396_396764

theorem head_start 
    (v_A v_B L : ℝ) 
    (h_speed : v_A = (20/16) * v_B) : 
    let x := 1 - (16/20) in
    (L / v_A = (L - x * L) / v_B) ↔ x = 1 / 5 :=
by sorry

end head_start_l396_396764


namespace jindra_gray_fields_counts_l396_396387

-- Definitions for the problem setup
noncomputable def initial_gray_fields: ℕ := 7
noncomputable def rotation_90_gray_fields: ℕ := 8
noncomputable def rotation_180_gray_fields: ℕ := 4

-- Statement of the theorem to be proved
theorem jindra_gray_fields_counts:
  initial_gray_fields = 7 ∧
  rotation_90_gray_fields = 8 ∧
  rotation_180_gray_fields = 4 := by
  sorry

end jindra_gray_fields_counts_l396_396387


namespace hyperbola_iff_ab_pos_l396_396773

theorem hyperbola_iff_ab_pos (a b x y : ℝ) :
  (ab > 0 ↔ (∃ (x y : ℝ), ax^2 - by^2 = 1)) :=
sorry

end hyperbola_iff_ab_pos_l396_396773


namespace sum_of_coordinates_l396_396448

-- Define a function h that satisfies the given conditions
variable (h : ℝ → ℝ)

-- Conditions
axiom h_condition : ∀ x : ℝ, x ≠ -2 → x ≠ 2 → h(x) = if x = -2 then 3 else if x = 2 then 3 else h x
axiom intersection_condition : ∀ a : ℝ, h(a) = h(a - 4)

-- The proof statement
theorem sum_of_coordinates (a : ℝ) (b : ℝ) (ha : h a = b) (ha_minus_4 : h (a - 4) = b) 
  (h_intersects : h 2 = 3 ∧ h (-2) = 3) : a + b = 5 :=
by
  sorry

end sum_of_coordinates_l396_396448


namespace exit_forest_strategy_l396_396258

/-- A strategy ensuring the parachutist will exit the forest with a path length of less than 2.5l -/
theorem exit_forest_strategy (l : Real) : 
  ∃ (path_length : Real), path_length < 2.5 * l :=
by
  use 2.278 * l
  sorry

end exit_forest_strategy_l396_396258


namespace complex_pair_solutions_l396_396099

theorem complex_pair_solutions :
  ∃ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 ∧ (∃ n : ℕ, n ∈ finset.range 24 ∧ a = exp(2 * π * I * n / 24) ∧ b = exp(-16 * π * I * n / 24)) :=
by sorry

end complex_pair_solutions_l396_396099


namespace sin_increasing_on_interval_l396_396435

theorem sin_increasing_on_interval :
  ∃ a b : ℝ, a = -π / 2 ∧ b = π / 2 ∧ (∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → sin x ≤ sin y) :=  sorry

end sin_increasing_on_interval_l396_396435


namespace order_a_b_c_d_l396_396859

-- Conditions
def a : ℝ := Real.logBase 0.5 5
def b : ℝ := Real.logBase 0.5 3
def c : ℝ := Real.logBase 3 2
def d : ℝ := 2 ^ 0.3

-- Proof statement
theorem order_a_b_c_d : a < b ∧ b < c ∧ c < d :=
by
  sorry

end order_a_b_c_d_l396_396859


namespace num_divisible_by_18_30_40_l396_396176

-- Define the LCM function (not provided in standard libraries)
def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

-- Main proof statement
theorem num_divisible_by_18_30_40 (lo hi : ℕ) (h1 : lo = 1000) (h2 : hi = 3000) : 
  (Finset.card (Finset.filter (λ x, x % (lcm (lcm 18 30) 40) = 0) (Finset.Icc lo hi))) = 6 :=
by
  sorry

end num_divisible_by_18_30_40_l396_396176


namespace new_pressure_of_nitrogen_gas_l396_396016

variable (p1 p2 v1 v2 k : ℝ)

theorem new_pressure_of_nitrogen_gas :
  (∀ p v, p * v = k) ∧ (p1 = 8) ∧ (v1 = 3) ∧ (p1 * v1 = k) ∧ (v2 = 7.5) →
  p2 = 3.2 :=
by
  intro h
  sorry

end new_pressure_of_nitrogen_gas_l396_396016


namespace sin_3x_sin_x_solutions_l396_396564

open Real

theorem sin_3x_sin_x_solutions :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (3 * x) = sin x) ∧ s.card = 7 := 
by sorry

end sin_3x_sin_x_solutions_l396_396564


namespace max_subway_riders_l396_396964

theorem max_subway_riders:
  ∃ (P F : ℕ), P + F = 251 ∧ (1 / 11) * P + (1 / 13) * F = 22 := sorry

end max_subway_riders_l396_396964


namespace f_2015_eq_neg_cos_l396_396119

noncomputable def f : ℕ → (ℝ → ℝ)
| 1 := λ x, Real.cos x
| n + 1 := λ x, (f n)' x

theorem f_2015_eq_neg_cos : ∀ x, f 2015 x = -Real.cos x := by
  sorry

end f_2015_eq_neg_cos_l396_396119


namespace count_right_angle_triangles_in_cube_l396_396112

theorem count_right_angle_triangles_in_cube (vertices : Fin 8 → ℝ × ℝ × ℝ) (cube : ∀ (v1 v2: Fin 8), EuclideanDistance (vertices v1) (vertices v2) = 1) :
  ∃ n, n = _ := -- Placeholder for the actual combinatorial count of right-angled triangles
sorry

end count_right_angle_triangles_in_cube_l396_396112


namespace casey_pumping_time_l396_396458

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end casey_pumping_time_l396_396458


namespace trigonometric_identity_solution_l396_396761

theorem trigonometric_identity_solution
  (x : ℝ) (n : ℤ) (k : ℤ) :
  (4 * (sin x)^3 * cos (3 * x) + 4 * (cos x)^3 * sin (3 * x) = 3 * sin (2 * x)) ↔
  (∃ n : ℤ, x = (π / 6) * (2 * n + 1)) ∨ (∃ k : ℤ, x = k * π) :=
by
  have h_cos_3x : ∀ (x : ℝ), cos (3 * x) = 4 * (cos x)^3 - 3 * cos x,
  { intro x, exact cos_triple_angle x }
  have h_sin_3x : ∀ (x : ℝ), sin (3 * x) = 3 * sin x - 4 * (sin x)^3,
  { intro x, exact sin_triple_angle x }
  have h_sin_2x : ∀ (x : ℝ), sin (2 * x) = 2 * sin x * cos x,
  { intro x, exact sin_double_angle x }
  have h_cos_2x : ∀ (x : ℝ), cos (2 * x) = (cos x)^2 - (sin x)^2,
  { intro x, exact cos_double_angle x }
  have h_pythagorean : ∀ (x : ℝ), (sin x)^2 + (cos x)^2 = 1,
  { intro x, exact sin_square_add_cos_square x }
  sorry -- Proof omitted

end trigonometric_identity_solution_l396_396761


namespace complement_of_M_in_U_l396_396992

-- Define U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define M based on the given condition
def M : Set ℕ := {x ∈ U | Real.log 2 (x^2 - 3 * x + 4) = 1}

-- Define the complement of M in U
def C_UM : Set ℕ := {x ∈ U | ¬(x ∈ M)}

-- The theorem statement
theorem complement_of_M_in_U :
  C_UM = {3, 4, 5} :=
sorry

end complement_of_M_in_U_l396_396992


namespace ratio_problem_l396_396939

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396939


namespace count_valid_pairs_l396_396492

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d ≠ 0

def is_valid_pair (a b : ℕ) : Prop :=
  a + b = 500 ∧ has_no_zero_digit a ∧ has_no_zero_digit b

theorem count_valid_pairs : 
  (finset.univ.filter (λ (a : ℕ), is_valid_pair a (500 - a))).card = 249 :=
sorry

end count_valid_pairs_l396_396492


namespace value_of_expression_l396_396928

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396928


namespace fraction_pow_rule_l396_396452

theorem fraction_pow_rule :
  (5 / 7)^4 = 625 / 2401 :=
by
  sorry

end fraction_pow_rule_l396_396452


namespace min_frac_sum_geometric_sequence_l396_396128

theorem min_frac_sum_geometric_sequence :
  ∀ {a : ℕ+ → ℝ} (a₁ : ℝ) (q : ℝ) (m n : ℕ+), 
  (∀ k, a k = a₁ * q ^ (k - 1)) →
  (∀ (m n : ℕ+), sqrt (a m * a n) = 4 * a₁) →
  a 7 = a 6 + 2 * a 5 →
  m + n = 6 →
  (1 / m.1 : ℝ) + 4 / (n.1 : ℝ) ≥ 3 / 2 :=
by
  sorry

end min_frac_sum_geometric_sequence_l396_396128


namespace max_value_at_e_f_inequal_e_f_prime_midpoint_l396_396548

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_at_e : f e = 1 / e := by
  sorry

theorem f_inequal_e (x : ℝ) (h1 : 0 < x) (h2 : x < e) : f (e + x) > f (e - x) := by
  sorry

theorem f_prime_midpoint (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < e) (h3 : e < x2) :
  let x0 := (x1 + x2) / 2
  f' (x0) < 0 := by
  sorry

end max_value_at_e_f_inequal_e_f_prime_midpoint_l396_396548


namespace maximum_value_of_w_l396_396865

theorem maximum_value_of_w 
  (p q : ℝ)
  (h1 : 2 * p - q ≥ 0)
  (h2 : 3 * q - 2 * p ≥ 0)
  (h3 : 6 - 2 * q ≥ 0) :
  (∃ w : ℝ, w = sqrt (2 * p - q) + sqrt (3 * q - 2 * p) + sqrt (6 - 2 * q) ∧ w ≤ 3 * sqrt 2) ∧ 
  ∃ p q : ℝ, 2 * p - q = 2 ∧ 3 * q - 2 * p = 2 ∧ 6 - 2 * q = 2 ∧ p = 2 ∧ q = 2 :=
begin
  sorry -- proof can be filled later
end

end maximum_value_of_w_l396_396865


namespace area_of_BCE_eq_region_l396_396655

open_locale classical

variable {Point : Type}
variables {A B C D E F : Point}
variables (AC BC AB CA : ℝ)
variables (R S : ℝ)  -- radii
variables (line_through_C : Point → Point → Prop)
variables (is_isosceles_right_triangle : Point → Point → Point → Prop)
variables (semicircle : Point → ℝ → Prop)
variables (quadrant : Point → ℝ → Prop)
variables (area : Point → Point → Point → ℝ)
variables (region : Point → Point → Point → Point → ℝ)

hypothesis hyp : is_isosceles_right_triangle A B C
hypothesis sem1 : semicircle A AB
hypothesis sem2 : semicircle C CA
hypothesis quad : quadrant C CA
hypothesis line : line_through_C D E
hypothesis intersect_D : line C D intersects arc BD
hypothesis intersect_E : line C E intersects arc BE
hypothesis intersect_F : line C F intersects arc BF

theorem area_of_BCE_eq_region :
  area B C E = region D F arc BD arc BF := by
sorry

end area_of_BCE_eq_region_l396_396655


namespace ratio_expression_value_l396_396934

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396934


namespace not_arithmetic_seq_geometric_conditions_sum_ineq_conditions_l396_396728

open Real

-- Definitions according to the given conditions
def a_seq (a : ℕ → ℝ) (k : ℝ) : Prop := ∀ n, a (n + 1) = k * a n + n

def b_seq (a b : ℕ → ℝ) : Prop := ∀ n, b n = a n - (2/3) * n + (4/9)

-- Prove a_n is not arithmetic under given conditions
theorem not_arithmetic_seq (a : ℕ → ℝ) (k : ℝ) (h1 : a 1 = 1) (ha : a_seq a k) : ¬∀ d, ∀ n, a (n + 1) = a n + d :=
sorry

-- Determine conditions on a_1 for b_n to be geometric when k = -1/2
theorem geometric_conditions (a b : ℕ → ℝ) (h1 : a_seq a (-1/2)) (h2 : b_seq a b) :
  ∀ a1, a 1 = a1 → (∀ q : ℝ, (∀ n, b (n + 1) = q * b n) ↔ a1 ≠ 2/9) :=
sorry

-- Prove range of a_1 for sum S_n to satisfy given inequalities
theorem sum_ineq_conditions (a b : ℕ → ℝ) (h1 : a_seq a (-1/2)) (h2 : b_seq a b) :
  ∃ a1, a 1 = a1 ∧ ∀ n, ∑ i in range (n + 1), b i ≥ 1/3 ∧ ∑ i in range (n + 1), b i ≤ 2/3 :=
exists.intro (8/9) sorry

end not_arithmetic_seq_geometric_conditions_sum_ineq_conditions_l396_396728


namespace trig_identity_example_l396_396390

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) -
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l396_396390


namespace log_sqrt8_512sqrt8_l396_396487

theorem log_sqrt8_512sqrt8 : log (sqrt 8) (512 * sqrt 8) = 7 := sorry

end log_sqrt8_512sqrt8_l396_396487


namespace three_digit_cubes_divisible_by_16_l396_396575

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l396_396575


namespace cost_price_of_one_meter_l396_396004

constant totalMeters : ℕ
constant totalSellingPrice : ℤ
constant profitPerMeter : ℤ

axiom h1 : totalMeters = 85
axiom h2 : totalSellingPrice = 8925
axiom h3 : profitPerMeter = 20

theorem cost_price_of_one_meter : (totalSellingPrice - totalMeters * profitPerMeter) / totalMeters = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cost_price_of_one_meter_l396_396004


namespace appointment_plans_count_l396_396970

-- We define the conditions in Lean.

def volunteers : List String := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
def tasks : List String := ["translation", "tour guide", "etiquette", "driving"]
def restrictedVolunteers : List String := ["Xiao Zhang", "Xiao Zhao"]
def unrestrictedVolunteers : List String := ["Xiao Li", "Xiao Luo", "Xiao Wang"]

-- We formalize the statement that the total number of different appointment plans is 15.

theorem appointment_plans_count : 
  ∃ (count : ℕ), count = 15 ∧ 
  (let 
     case1 := 3; -- When both Xiao Zhang and Xiao Zhao are selected
     case2 := 12 -- When only one of Xiao Zhang or Xiao Zhao is selected
  in case1 + case2 = count) := 
  sorry

end appointment_plans_count_l396_396970


namespace divide_segment_with_compass_and_straightedge_l396_396742

theorem divide_segment_with_compass_and_straightedge 
  (A B C : Point) (n : ℕ) (h : n > 0)
  (AB : LineSegment A B) (AC : Ray A C)
  (A_i : Finₓ n → Point) (h_eq_segments : ∀ i, distance A_i.succ.val A_i ≤ distance A_i A)
  (D : Point) (h_end : A_i n = D) 
  (DB : LineSegment D B) (P_i : Finₓ (n-1) → Point)
  (h_parallel : ∀ i, is_parallel (LineSegment A_i.val D) (LineSegment P_i.val B)) :
  ∀ i j, distance P_i.val P_j.val = distance P_0.val A := sorry

end divide_segment_with_compass_and_straightedge_l396_396742


namespace infinite_sum_of_reciprocals_l396_396073

noncomputable def set_B := { n : ℕ | ∀ p : ℕ, p.prime → p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 }

theorem infinite_sum_of_reciprocals (p q : ℕ) (hpq_rel_prime : Nat.coprime p q)
  (h_sum_eq : ∑' n in set_B, 1 / n = (35 : ℚ) / 8) : 
  p + q = 43 :=
sorry

end infinite_sum_of_reciprocals_l396_396073


namespace kevin_hop_distance_l396_396640

theorem kevin_hop_distance :
  (1/4) + (3/16) + (9/64) + (27/256) + (81/1024) + (243/4096) = 3367 / 4096 := 
by
  sorry 

end kevin_hop_distance_l396_396640


namespace largest_share_received_l396_396855

theorem largest_share_received (total_profit : ℕ) (ratio_1 ratio_2 ratio_3 ratio_4 : ℕ) :
  total_profit = 45000 ∧ ratio_1 = 1 ∧ ratio_2 = 4 ∧ ratio_3 = 4 ∧ ratio_4 = 6 →
  let total_parts := ratio_1 + ratio_2 + ratio_3 + ratio_4 in
  let value_per_part := total_profit / total_parts in
  let largest_share := ratio_4 * value_per_part in
  largest_share = 18000 :=
by
  intro h
  cases h with hp hr
  have total_parts_eq : ratio_1 + ratio_2 + ratio_3 + ratio_4 = 15, {
    rw [hr.1, hr.2, hr.2, hr.3],
    norm_num
  }
  have value_per_part_eq : total_profit / total_parts = 3000, {
    rw [←hp, total_parts_eq],
    norm_num
  }
  have largest_share_eq : ratio_4 * value_per_part = 18000, {
    rw [hr.3, value_per_part_eq],
    norm_num
  }
  exact largest_share_eq

end largest_share_received_l396_396855


namespace sequence_sum_fraction_l396_396489

/--
Evaluate the fraction formed by the sum of the sequence 4, 8, 12, ..., 44 
and the sum of the sequence 4, 8, 12, ..., 68.
-/
theorem sequence_sum_fraction :
  let numerator_sum := (11 * (4 + 44)) / 2
  let denominator_sum := (17 * (4 + 68)) / 2
  (numerator_sum / denominator_sum).simplify_fractions = 22 / 51 :=
by
  sorry

end sequence_sum_fraction_l396_396489


namespace tree_count_in_yard_l396_396602

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end tree_count_in_yard_l396_396602


namespace real_solution_if_z_real_imaginary_solution_if_z_pure_imaginary_l396_396653

variables (m : ℝ) (z : ℂ)

def complex_expression (m : ℝ) : ℂ :=
  (2 + complex.I) * m^2 - 3 * (1 + complex.I) * m - 2 * (1 - complex.I)

theorem real_solution_if_z_real :
  (z = complex_expression m ∧ z.im = 0) → (m = 1 ∨ m = 2) :=
sorry

theorem imaginary_solution_if_z_pure_imaginary :
  (z = complex_expression m ∧ z.re = 0 ∧ z.im ≠ 0) → (m = -1 / 2) :=
sorry

end real_solution_if_z_real_imaginary_solution_if_z_pure_imaginary_l396_396653


namespace interval_increase_log_03_neg_x2_plus_4x_l396_396077

noncomputable def interval_of_increase (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

noncomputable def log_base (b t : ℝ) : ℝ :=
  Real.log t / Real.log b

theorem interval_increase_log_03_neg_x2_plus_4x :
  interval_of_increase (λ x, log_base 0.3 (-x^2 + 4*x)) [2, 4) :=
sorry

end interval_increase_log_03_neg_x2_plus_4x_l396_396077


namespace area_of_triangle_max_perimeter_l396_396219

noncomputable theory

variables {a b c : ℝ} {A B C : ℝ}
variables (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
variables (h : 2 * a * Real.cos B + Real.sqrt 3 * b = 2 * c)

-- Question (1): Prove the area of triangle ABC is 2sqrt(3) or sqrt(3).
theorem area_of_triangle (area : ℝ) : 
  ∃ area, (area = Real.sqrt 3 ∨ area = 2 * Real.sqrt 3) :=
sorry

-- Question (2): Prove the maximum perimeter is 2 + 2(sqrt(6) + sqrt(2)).
theorem max_perimeter (P : ℝ) :
  ∃ P, P = 2 + 2 * (Real.sqrt 6 + Real.sqrt 2) :=
sorry

end area_of_triangle_max_perimeter_l396_396219


namespace tom_spends_total_cost_l396_396357

theorem tom_spends_total_cost :
  (let total_bricks := 1000
       half_bricks := total_bricks / 2
       full_price := 0.50
       half_price := full_price / 2
       cost_half := half_bricks * half_price
       cost_full := half_bricks * full_price
       total_cost := cost_half + cost_full
   in total_cost = 375) := 
by
  let total_bricks := 1000
  let half_bricks := total_bricks / 2
  let full_price := 0.50
  let half_price := full_price / 2
  let cost_half := half_bricks * half_price
  let cost_full := half_bricks * full_price
  let total_cost := cost_half + cost_full
  show total_cost = 375 from sorry

end tom_spends_total_cost_l396_396357


namespace lengths_of_catheti_l396_396966

-- Define a right triangle with incenter I and geocenter G
variables {A B C I G : Type} [normed_field A]
          [normed_space ℝ A] [normed_field B]
          [normed_space ℝ B] [normed_field C]
          [normed_space ℝ C]

-- Conditions for incenter I and geocenter G
def rectangle_triangle (A B C I G : Type) :=
  is_right_triangle A B C ∧
  incenter A B C I ∧
  geocenter A B C G ∧
  parallel (IG) (BC) ∧
  distance I G = 10

-- The theorem to prove the lengths of the two catheti are 90 cm and 120 cm
theorem lengths_of_catheti (A B C I G : Type) [normed_field A]
          [normed_space ℝ A] [normed_field B]
          [normed_space ℝ B] [normed_field C]
          [normed_space ℝ C]
          (h : rectangle_triangle A B C I G) :
  ∃ (a b : ℝ), a = 90 ∧ b = 120 :=
sorry

end lengths_of_catheti_l396_396966


namespace constant_term_of_polynomial_l396_396883

noncomputable def a := ∫ x in 0..Real.pi, (Real.sin x - 1 + 2 * Real.cos (x / 2) ^ 2)

theorem constant_term_of_polynomial (h : a = 2) :
  let p := Polynomial.C a * Polynomial.root_x 1 - Polynomial.C (1 / a)
  let q := p ^ 6 * (Polynomial.x ^ 2 + Polynomial.C 2)
  Polynomial.coeff q 0 = -332 := sorry

end constant_term_of_polynomial_l396_396883


namespace combination_7_2_l396_396030

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396030


namespace minibus_children_count_l396_396303

theorem minibus_children_count
  (total_seats : ℕ)
  (seats_with_3_children : ℕ)
  (seats_with_2_children : ℕ)
  (children_per_seat_3 : ℕ)
  (children_per_seat_2 : ℕ)
  (h_seats_count : total_seats = 7)
  (h_seats_distribution : seats_with_3_children = 5 ∧ seats_with_2_children = 2)
  (h_children_per_seat : children_per_seat_3 = 3 ∧ children_per_seat_2 = 2) :
  seats_with_3_children * children_per_seat_3 + seats_with_2_children * children_per_seat_2 = 19 :=
by
  sorry

end minibus_children_count_l396_396303


namespace sum_op_two_triangles_l396_396311

def op (a b c : ℕ) : ℕ := 2 * a - b + c

theorem sum_op_two_triangles : op 3 7 5 + op 6 2 8 = 22 := by
  sorry

end sum_op_two_triangles_l396_396311


namespace eval_expression_l396_396733

theorem eval_expression : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end eval_expression_l396_396733


namespace original_price_l396_396399

theorem original_price (SP : ℝ) (gain_percent : ℝ) (P : ℝ) : SP = 1080 → gain_percent = 0.08 → SP = P * (1 + gain_percent) → P = 1000 :=
by
  intro hSP hGainPercent hEquation
  sorry

end original_price_l396_396399


namespace marc_watching_episodes_l396_396265

theorem marc_watching_episodes :
  ∀ (n : ℕ) (f : ℝ),
  n = 50 → f = 1/10 → 
  n / (n * f) = 10 := 
by
  intro n f hn hf
  rw [hn, hf]
  norm_num
  sorry

end marc_watching_episodes_l396_396265


namespace find_n_l396_396500

-- Define the function d(n) which counts the number of positive divisors of n
def d (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ i, i > 0 ∧ n % i = 0).card

-- Main statement with conditions
theorem find_n (n : ℕ) (hn : 0 < n) (h : d(n)^3 = 4 * n) : n = 2 ∨ n = 128 ∨ n = 2000 :=
sorry

end find_n_l396_396500


namespace no_solution_for_inequality_l396_396958

theorem no_solution_for_inequality (a : ℝ) : ∀ x : ℝ, (x^2 + a * x + a - 2 > 0) → (a ∈ ∅) := 
by 
  sorry

end no_solution_for_inequality_l396_396958


namespace sum_of_solutions_sum_of_all_solutions_l396_396850

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 64) : x = 16 ∨ x = 0 :=
begin
  sorry
end

theorem sum_of_all_solutions : (∀ x, (x-8)^2=64 → x=16 ∨ x=0) → 16 + 0 = 16 :=
by { intro h, refl }

end sum_of_solutions_sum_of_all_solutions_l396_396850


namespace area_cyclic_quadrilateral_ABCD_l396_396126

-- Define the cyclic quadrilateral with given side lengths
variables (AB BC CD DA : ℝ) (cyclic : AB = 2 ∧ BC = 6 ∧ CD = 4 ∧ DA = 4)

-- Define the problem to prove
theorem area_cyclic_quadrilateral_ABCD (h : cyclic): (area : ℝ) := 
  ∃ A : ℝ, ∃ sinA : ℝ, 
  sinA = Real.sin 120 ∧ -- because we found in the solution that A = 120 degrees
  area = 16 * sinA ∧ -- from the area calculation step
  area = 8 * Real.sqrt 3 := 
sorry

end area_cyclic_quadrilateral_ABCD_l396_396126


namespace find_c_value_l396_396793

theorem find_c_value 
  (c : ℝ)
  (h₀ : 0 < c)
  (h₁ : c < 6)
  (H : ∀ Q P S (Q : ℝ × ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ),
    Q = (c, 0) ∧ P = (0, c) ∧ S = (6, c - 6) →
    let area_QOP := (1/2) * c * c in
    let area_QRS := (1/2) * ((6 - c) * (c - 6)) in
    (area_QRS / area_QOP) = 4/25) : 
  c = 30 / 7 :=
sorry

end find_c_value_l396_396793


namespace largest_angle_in_pentagon_l396_396608

-- Define the angles of the pentagon
variables (C D E : ℝ) 

-- Given conditions
def is_pentagon (A B C D E : ℝ) : Prop :=
  A = 75 ∧ B = 95 ∧ D = C + 10 ∧ E = 2 * C + 20 ∧ A + B + C + D + E = 540

-- Prove that the measure of the largest angle is 190°
theorem largest_angle_in_pentagon (C D E : ℝ) : 
  is_pentagon 75 95 C D E → max 75 (max 95 (max C (max (C + 10) (2 * C + 20)))) = 190 :=
by 
  sorry

end largest_angle_in_pentagon_l396_396608


namespace cone_volume_from_sector_l396_396800

theorem cone_volume_from_sector (r : ℝ) (θ : ℝ) (h' : ℝ) (V : ℝ) (cond1 : θ = 3 / 4 * 2 * π) (cond2 : r = 4)
  (cond3 : h' = 4) (circumference : 2 * π ≠ 0) :
  V = 1 / 3 * π * (3) * (3) * (sqrt 7) :=
  by
    -- let arc length (3/4 of full circle circumference) form the base circumference of the cone
    let base_circumference := cond1 * cond2,
    have base_radius := 3, -- since 2πr = 6π, implies r = 3
    have slant_height := cond3,
    have cone_height := sqrt 7,
    -- V = 1/3 π r^2 h
    have vol_cone := 1 / 3 * π * (base_radius) * (base_radius) * cone_height,
    sorry

end cone_volume_from_sector_l396_396800


namespace log_inequality_m_condition_l396_396389

theorem log_inequality_m_condition (a : ℝ) (m : ℝ) (h : a > 1) : 
  (m = 2) ↔ (∃ x > 1, log x 2 + log 2 x ≥ m) ∧ (m < 2 → ∀ y > 1, log y 2 + log 2 y ≥ m) :=
by sorry

end log_inequality_m_condition_l396_396389


namespace Jackie_has_more_apples_l396_396423

def Adam_apples : Nat := 9
def Jackie_apples : Nat := 10

theorem Jackie_has_more_apples : Jackie_apples - Adam_apples = 1 := by
  sorry

end Jackie_has_more_apples_l396_396423


namespace equal_angles_in_parallelogram_l396_396869

variables {A B C D P : Type}
variables [geometry A B C D P]  -- Assuming we have some structure that includes points and angles

theorem equal_angles_in_parallelogram
  (parallelogram : is_parallelogram A B C D)
  (P_inside : is_inside P A B C D)
  (angle_eq : ∠ P B C = ∠ P D C) :
  ∠ P A B = ∠ P C B :=
sorry

end equal_angles_in_parallelogram_l396_396869


namespace repeating_decimal_fraction_equiv_l396_396376

def repeating_decimal_to_fraction := 0.35 -- 0.35 part before the repeating part

def repeating_part := 247

theorem repeating_decimal_fraction_equiv :
  let y := (0.35 + (repeat_to_infinite repeating_part / 1000)) in
  y = 3518950 / 999900 := 
by
  sorry

end repeating_decimal_fraction_equiv_l396_396376


namespace telephone_numbers_containing_12_l396_396916

theorem telephone_numbers_containing_12 :
  let six_digits := fin 10 -> fin 10 -> fin 10 -> fin 10 -> fin 10 -> fin 10 -> Prop
  let contains_12 (n : fin 10 → fin 6 → nat) := exists i, n 1 i = 1 ∧ n 2 (i+1) = 2
  (finset.filter contains_12 (finset.range 10^6)).card = 49401 :=
begin
  sorry
end

end telephone_numbers_containing_12_l396_396916


namespace expected_value_of_square_of_distinct_colors_digit_sum_l396_396699

theorem expected_value_of_square_of_distinct_colors_digit_sum :
  (let n : ℕ := 10,
       S := -- the exact mathematical expression for the expected value of the square of the number of distinct colors
             (1 / n ^ n) * (
               (10 * 9 * 8 ^ n +
                10 * (1 - 2 * 10) * 9 ^ n +
                10 ^ 12)
             )
    in S.digits.sum) = 55 :=
by {
  -- Proof will be placed by users
  sorry
}

end expected_value_of_square_of_distinct_colors_digit_sum_l396_396699


namespace min_value_quadratic_l396_396375

theorem min_value_quadratic : ∃ x : ℝ, (x^2 - 14 * x + 45) ≤ y ∀ y : ℝ → ∀ t : ℝ, (t^2 - 14 * t + 45 ≥ x^2 - 14 * x + 45) :=
  by
    sorry

end min_value_quadratic_l396_396375


namespace MaxPointsTiffanyCanGet_l396_396739

/-- Tiffany attends the carnival and plays ring toss. We formalize the conditions and prove that the maximum total points she can get for all three games is 53. -/
theorem MaxPointsTiffanyCanGet :
  ∃ (initial_money game_cost max_games rings_per_game red_points green_points blue_points blue_success_rate red_hits green_hits blue_hits time_per_game : ℕ),
  initial_money = 3 ∧
  game_cost = 1 ∧
  max_games = 3 ∧
  rings_per_game = 5 ∧
  red_points = 2 ∧
  green_points = 3 ∧
  blue_points = 5 ∧
  blue_success_rate = 10 ∧
  red_hits = 4 ∧
  green_hits = 5 ∧
  blue_hits = 1 ∧
  time_per_game = 1 ∧ 
  (initial_money / game_cost = max_games) ∧
  (red_hits * red_points + green_hits * green_points + blue_hits * blue_points + rings_per_game * blue_points = 53) := 
begin
  sorry
end

end MaxPointsTiffanyCanGet_l396_396739


namespace order_exponentials_l396_396509

theorem order_exponentials (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) : 
  a < a^(a^a) ∧ a^(a^a) < a^(sqrt a) ∧ a^(sqrt a) < a^a := 
by
  sorry

end order_exponentials_l396_396509


namespace part1_part2_l396_396543

noncomputable def f (a c x : ℝ) : ℝ :=
  if x >= c then a * Real.log x + (x - c) ^ 2
  else a * Real.log x - (x - c) ^ 2

theorem part1 (a c : ℝ)
  (h_a : a = 2 * c - 2)
  (h_c_gt_0 : c > 0)
  (h_f_geq : ∀ x, x ∈ (Set.Ioi c) → f a c x >= 1 / 4) :
    a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
  sorry

theorem part2 (a c x1 x2 : ℝ)
  (h_a_lt_0 : a < 0)
  (h_c_gt_0 : c > 0)
  (h_x1 : x1 = Real.sqrt (- a / 2))
  (h_x2 : x2 = c)
  (h_tangents_intersect : deriv (f a c) x1 * deriv (f a c) x2 = -1) :
    c >= 3 * Real.sqrt 3 / 2 :=
  sorry

end part1_part2_l396_396543


namespace range_of_a_l396_396514

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + a

theorem range_of_a 
  (f : ℝ → ℝ → ℝ)
  (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x a ≥ 0) : 
  3 ≤ a :=
sorry

end range_of_a_l396_396514


namespace bridget_profit_l396_396817

/- Definitions based on conditions -/

def total_loaves : ℕ := 60
def morning_sales_loaves : ℕ := total_loaves * 2 / 5
def morning_sales_revenue : ℕ := morning_sales_loaves * 3

def remaining_loaves_after_morning : ℕ := total_loaves - morning_sales_loaves
def afternoon_sales_loaves : ℕ := remaining_loaves_after_morning / 2
def afternoon_sales_revenue : ℕ := afternoon_sales_loaves * 3 / 2

def remaining_loaves_after_afternoon : ℕ := remaining_loaves_after_morning - afternoon_sales_loaves
def late_afternoon_sales_loaves : ℕ := remaining_loaves_after_afternoon * 2 / 3
def late_afternoon_sales_revenue : ℕ := late_afternoon_sales_loaves * 2

def total_revenue : ℕ := morning_sales_revenue + afternoon_sales_revenue + late_afternoon_sales_revenue

def cost_per_loaf : ℕ := 1
def total_cost_loaves : ℕ := total_loaves * cost_per_loaf
def operational_cost : ℕ := 10
def total_cost : ℕ := total_cost_loaves + operational_cost

def profit : ℕ := total_revenue - total_cost

/- Theorem stating Bridget's profit -/

theorem bridget_profit : profit = 53 :=
by
  unfold profit total_revenue morning_sales_revenue afternoon_sales_revenue late_afternoon_sales_revenue
  unfold total_cost total_cost_loaves operational_cost
  rw [morning_sales_revenue, afternoon_sales_revenue, late_afternoon_sales_revenue]
  rw [total_cost_loaves, operational_cost]
  sorry

end bridget_profit_l396_396817


namespace factor_expression_l396_396508

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) :=
by sorry

end factor_expression_l396_396508


namespace simplify_expression_l396_396115

theorem simplify_expression :
  ((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4) = 125 / 12 := by
  sorry

end simplify_expression_l396_396115


namespace remainder_sequences_mod_1000_l396_396849

theorem remainder_sequences_mod_1000 :
  ∃ m, (m = 752) ∧ (m % 1000 = 752) ∧ 
  (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ 6 → (a i) - i % 2 = 1), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 6 → a i ≤ a j) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 6 → 1 ≤ a i ∧ a i ≤ 1500)
  ) := by
    -- proof would go here
    sorry

end remainder_sequences_mod_1000_l396_396849


namespace jeff_cats_count_l396_396981

theorem jeff_cats_count :
  let initial_cats := 20
  let found_monday := 2 + 3
  let found_tuesday := 1 + 2
  let adopted_wednesday := 4 * 2
  let adopted_thursday := 3
  let found_friday := 3
  initial_cats + found_monday + found_tuesday - adopted_wednesday - adopted_thursday + found_friday = 20 := by
  sorry

end jeff_cats_count_l396_396981


namespace bob_cleans_in_15_minutes_l396_396807

-- Define Alice's cleaning time
def alice_cleaning_time : ℝ := 40

-- Define the ratio of Bob's cleaning time to Alice's cleaning time
def bob_ratio : ℝ := 3 / 8

-- Define Bob's cleaning time based on Alice's cleaning time and the given ratio
def bob_cleaning_time : ℝ := alice_cleaning_time * bob_ratio

-- Theorem statement
theorem bob_cleans_in_15_minutes : bob_cleaning_time = 15 :=
by
  -- sorry is a placeholder for the proof
  sorry

end bob_cleans_in_15_minutes_l396_396807


namespace value_of_expression_l396_396927

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396927


namespace min_value_f_l396_396897

noncomputable def f (r x : ℝ) : ℝ := r * x - x ^ r + (1 - r)

-- Given that r is a rational number and 0 < r < 1
variables (r : ℝ) [hr_rat : Rational r] (hr1 : 0 < r) (hr2 : r < 1)

-- We need to prove that the minimum value of f(x) for x > 0 is 0, and it occurs at x = 1
theorem min_value_f (x : ℝ) (hx : 0 < x) : min (f r x) = 0 :=
by sorry

end min_value_f_l396_396897


namespace solve_for_x_l396_396691

theorem solve_for_x :
  ∀ x : ℝ, x > 0 → (x ^ (Real.log10 x) = x ^ 5 / 10000) → (x = 10 ∨ x = 10000) :=
by
  intro x
  intro hx
  intro h
  sorry

end solve_for_x_l396_396691


namespace min_value_of_h_min_value_of_b_plus_4_over_a_l396_396885

-- Part (1)
def f (x : ℝ) : ℝ := 2 * x - 2
def g (x : ℝ) : ℝ := x ^ 2 + 4 * x - 5
def h (x : ℝ) : ℝ := max (f x) (g x)

theorem min_value_of_h : ∀ x : ℝ, h x ≥ -8 :=
sorry

-- Part (2)
variables (a b : ℝ)
def f' (x : ℝ) : ℝ := a * x - 2
def g' (x : ℝ) : ℝ := x ^ 2 + b * x - 5

theorem min_value_of_b_plus_4_over_a 
  (h: ∀ x > 0, (f' a x) * (g' a b x) ≥ 0) 
  (ha : a > 0) : b + 4 / a ≥ 2 * Real.sqrt 5 :=
sorry

end min_value_of_h_min_value_of_b_plus_4_over_a_l396_396885


namespace simplest_quadratic_radical_l396_396011

theorem simplest_quadratic_radical (a : ℝ) : 
  let optionA := -sqrt 3,
      optionB := sqrt 20,
      optionC := sqrt (1 / 2),
      optionD := sqrt (a ^ 2)
  in optionA = -sqrt 3 :=
by
  sorry

end simplest_quadratic_radical_l396_396011


namespace sum_first_5_arithmetic_l396_396383

theorem sum_first_5_arithmetic (u : ℕ → ℝ) (h : u 3 = 0) : 
  (u 1 + u 2 + u 3 + u 4 + u 5) = 0 :=
sorry

end sum_first_5_arithmetic_l396_396383


namespace Luke_mowing_lawns_l396_396664

theorem Luke_mowing_lawns (L : ℕ) (h1 : 18 + L = 27) : L = 9 :=
by
  sorry

end Luke_mowing_lawns_l396_396664


namespace distance_between_feet_of_perpendiculars_eq_area_over_radius_l396_396284
noncomputable def area (ABC : Type) : ℝ := sorry
noncomputable def circumradius (ABC : Type) : ℝ := sorry

theorem distance_between_feet_of_perpendiculars_eq_area_over_radius
  (ABC : Type)
  (area_ABC : ℝ)
  (R : ℝ)
  (h_area : area ABC = area_ABC)
  (h_radius : circumradius ABC = R) :
  ∃ (m : ℝ), m = area_ABC / R := sorry

end distance_between_feet_of_perpendiculars_eq_area_over_radius_l396_396284


namespace tournament_trio_l396_396446

theorem tournament_trio
  (n : ℕ)
  (h_n : n ≥ 3)
  (match_result : Fin n → Fin n → Prop)
  (h1 : ∀ i j : Fin n, i ≠ j → (match_result i j ∨ match_result j i))
  (h2 : ∀ i : Fin n, ∃ j : Fin n, match_result i j)
:
  ∃ (A B C : Fin n), match_result A B ∧ match_result B C ∧ match_result C A :=
by
  sorry

end tournament_trio_l396_396446


namespace alex_shirts_4_l396_396431

/-- Define the number of new shirts Alex, Joe, and Ben have. -/
def shirts_of_alex (alex_shirts : ℕ) (joe_shirts : ℕ) (ben_shirts : ℕ) : Prop :=
  joe_shirts = alex_shirts + 3 ∧ ben_shirts = joe_shirts + 8 ∧ ben_shirts = 15

theorem alex_shirts_4 {alex_shirts : ℕ} :
  ∃ joe_shirts ben_shirts, shirts_of_alex alex_shirts joe_shirts ben_shirts ∧ alex_shirts = 4 :=
by
  have joe_shirts := 4 + 3 by rfl
  have ben_shirts := 7 + 8 by rfl
  use joe_shirts, ben_shirts
  split
  . exact ⟨rfl, rfl, rfl⟩
  . exact rfl
  sorry

end alex_shirts_4_l396_396431


namespace radius_of_inscribed_circle_l396_396610

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end radius_of_inscribed_circle_l396_396610


namespace binomial_7_2_l396_396035

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396035


namespace probability_positive_root_eq_half_l396_396678

-- Define the conditions of the problem

def in_interval (a : ℝ) : Prop := -3 ≤ a ∧ a ≤ 3

def has_positive_root (a : ℝ) : Prop :=
∃ x : ℝ, x > 0 ∧ x^2 + 3*x + a = 0

def probability_condition : ℝ := (3 : ℝ) / 6

-- Prove the theorem
theorem probability_positive_root_eq_half :
  (∃ a : ℝ, in_interval a ∧ has_positive_root a) →
  probability_condition = 1 / 2 :=
by
  sorry

end probability_positive_root_eq_half_l396_396678


namespace estimated_sales_l396_396890

theorem estimated_sales (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) (h1 : x1 + x2 + x3 + x4 = 18)
  (h2 : y1 + y2 + y3 + y4 = 14) (h3 : ∀ x, ∃ y, y = 0.8 * x + -0.1) : False :=
begin
  -- given conditions
  let x_avg := (x1 + x2 + x3 + x4) / 4,
  let y_avg := (y1 + y2 + y3 + y4) / 4,
  have h4 : x_avg = 4.5, from calc
    x_avg = (x1 + x2 + x3 + x4) / 4 : by refl
        ... = 18 / 4                 : by rw h1
        ... = 4.5                    : by norm_num,
  have h5 : y_avg = 3.5, from calc
    y_avg = (y1 + y2 + y3 + y4) / 4 : by refl
        ... = 14 / 4                : by rw h2
        ... = 3.5                   : by norm_num,
  -- regression line verification
  have a := 3.5 - 0.8 * 4.5,
  have y_estimate := 0.8 * 6 - 0.1,
  have correct_estimate := y_estimate = 4.7,

  -- assert that the correct estimate is 4.7
  sorry
end

end estimated_sales_l396_396890


namespace dartboard_distributions_l396_396808

/-- Alice throws five identical darts. Each hits one of four identical dartboards on the wall. 
    Prove that the number of distinct lists of counts of darts on each board (sorted from greatest to least) is 6. -/
theorem dartboard_distributions : 
  (number_of_distinct_lists (4, 5)) = 6 :=
sorry

/-- Helper function for counting distinct lists -/
def number_of_distinct_lists (boards_darts : ℕ × ℕ) : ℕ :=
  let (boards, darts) := boards_darts
  if darts = 0 then 1 else
    multiset.powerset_len boards (multiset.repeat 1 darts).to_powerset


end dartboard_distributions_l396_396808


namespace triangle_properties_l396_396600

noncomputable def triangle {A B C : Type} [MetricSpace A] [AddGroup B] [AddGroup C]
(ABC : Triangle ℝ) (a b c : ℝ) (cosB : ℝ) :=
  (a = 2) ∧ (b + c = 7) ∧ (cosB = -1 / 4)

theorem triangle_properties : 
  ∃ (b : ℝ) (S : ℝ), triangle ABC 2 b (7 - b) (-1 / 4) ∧ b = 4 ∧
  S = (1 / 2) * 2 * (7 - b) * Real.sin (Real.acos (-1 / 4)) ∧ 
  S = (3 * Real.sqrt 15) / 4 := by sorry

end triangle_properties_l396_396600


namespace sum_of_roots_g_eq_3006_l396_396405

noncomputable def g (x : ℝ) : ℝ := 3 * x - 3 / x

theorem sum_of_roots_g_eq_3006 : 
  let T := ∑ x in {x : ℝ | g x = 3006}.to_finset in
  T = 1002 :=
 by
  sorry

end sum_of_roots_g_eq_3006_l396_396405


namespace ivan_erema_meeting_time_l396_396979

theorem ivan_erema_meeting_time (R : ℝ) (I F E : ℝ) 
  (h1 : ∀ t, Ivan_distance(t) = R * (4/9) + I * t) 
  (h2 : ∀ t, Foma_distance(t) = F * t)
  (h3 : ∀ t, Erema_distance(t) = R - E * t)
  (h4 : F = (4/9) * R + I)
  (h5 : F + E = 2/3 * R) : 
  ∃ t, t = 2.5 ∧ Ivan_distance(t) + Erema_distance(t) = R := 
begin
  existsi 2.5, 
  split,
  { -- prove t = 2.5
    simp,
    exact rfl,
  },
  { -- prove Ivan_distance(t) + Erema_distance(t) = R
    have d_ivan := h1 2.5,
    have d_erema := h3 2.5,
    -- combining distances to show they equal R
    rw [d_ivan, d_erema], 
    -- simplifying, using relevant equations for the rates
    simp only [add_comm, sub_self, add_right_inj, mul_add, R],
    -- providing justification
    ring,
  },
  sorry
end

end ivan_erema_meeting_time_l396_396979


namespace mural_width_l396_396984

theorem mural_width (l p r c t w : ℝ) (h₁ : l = 6) (h₂ : p = 4) (h₃ : r = 1.5) (h₄ : c = 10) (h₅ : t = 192) :
  4 * 6 * w + 10 * (6 * w / 1.5) = 192 → w = 3 :=
by
  intros
  sorry

end mural_width_l396_396984


namespace augmented_matrix_solution_l396_396952

theorem augmented_matrix_solution (m n : ℝ) (x y : ℝ)
  (h1 : m * x = 6) (h2 : 3 * y = n) (hx : x = -3) (hy : y = 4) :
  m + n = 10 :=
by
  sorry

end augmented_matrix_solution_l396_396952


namespace mike_books_l396_396352

theorem mike_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  ∃ (mike_books : ℕ), mike_books = total_books - tim_books :=
by {
  use 20,
  rw [h1, h2],
  norm_num,
  sorry
}

end mike_books_l396_396352


namespace zero_point_in_interval_l396_396541

noncomputable def f (x : ℝ) : ℝ := log x - 3 / real.exp 1

theorem zero_point_in_interval :
  (∃ c : ℝ, e < c ∧ c < e^2 ∧ f c = 0) :=
by
  have : f e < 0 := by
    calc f e = log e - 3 / real.exp 1 : by sorry
         ... = 1 - 3 / real.exp 1 : by sorry
         ... < 0 : by sorry
  have : f (e^2) > 0 := by
    calc f (e^2) = log (e^2) - 3 / real.exp 1 : by sorry
              ... = 2 - 3 / real.exp 1 : by sorry
              ... > 0 : by sorry
  sorry

end zero_point_in_interval_l396_396541


namespace ratio_expression_value_l396_396920

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396920


namespace range_of_a_l396_396106

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → a ∈ set.Ioo (-1 : ℝ) 0 :=
by
  sorry

end range_of_a_l396_396106


namespace equation_of_regression_line_correct_l396_396889

-- Declare the known values
def m : ℝ := 1.2
def point := (4 : ℝ, 5 : ℝ)

-- Definition of the regression line
def regression_line (x : ℝ) (b : ℝ) : ℝ := m * x + b

-- Proposition that needs to be proved
theorem equation_of_regression_line_correct (b : ℝ) (H : regression_line 4 b = 5) : b = 0.2 :=
sorry

end equation_of_regression_line_correct_l396_396889


namespace area_of_interior_triangle_l396_396349

theorem area_of_interior_triangle (a b c : ℝ) (A B C : ℝ) (ha : a = sqrt 36) (hb : b = sqrt 64) (hc : c = sqrt 100) (h_diag : A = a^2) (h_diag' : B = b^2) (h_eq : C = A + B) :
  1 / 2 * a * b = 24 :=
by
  have ha := ha.symm
  have hb := hb.symm
  have hc := hc.symm
  rw [ha, hb, hc]
  have h_Pythagorean : 6 ^ 2 + 8 ^ 2 = 100 := by ring
  have h_eq_diag : C = 6 ^ 2 + 8 ^ 2 := by rwa h_eq
  have h_diag_val : 10 ^ 2 = 100 := by norm_num
  rw [←h_eq_diag, h_diag_val]
  calc
    1 / 2 * 6 * 8 = 1 / 2 * 48 := by ring
    ... = 24 := by norm_num


end area_of_interior_triangle_l396_396349


namespace eating_cupcakes_correct_l396_396741

noncomputable def num_girls_eating_two_cupcakes 
  (total_girls : ℕ) (average_cupcakes : ℚ) (no_cupcakes_girls : ℕ) (total_cupcakes : ℕ) : ℕ :=
  if h : (average_cupcakes * total_girls = total_cupcakes) ∧
          (total_girls - no_cupcakes_girls).natCast >= 0 ∧
          (2 * (total_girls - no_cupcakes_girls) = (total_cakes + no_cupcakes_girls + 2))
  then ‹(total_girls - no_cupcakes_girls - 2)›
  else 0

theorem eating_cupcakes_correct :
  num_girls_eating_two_cupcakes 12 (3/2) 2 18 = 8 :=
by
  sorry

end eating_cupcakes_correct_l396_396741


namespace abs_neg_five_l396_396393

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l396_396393


namespace apples_difference_l396_396079

variable (D C total_diff : ℕ)

theorem apples_difference :
  (∀ (D C : ℕ), C = 15 → D + C = 50 → D - C = total_diff) → total_diff = 20 :=
by
  intros h
  have h1 := h 35 15 (by rfl) (by rfl)
  exact h1

end apples_difference_l396_396079


namespace max_jars_in_crate_l396_396402

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end max_jars_in_crate_l396_396402


namespace general_term_l396_396627

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0  -- To handle zero-indexing, not part of the natural sequence
  else if n = 1 then 3
  else 3 * sequence (n-1) - 4

theorem general_term (n : ℕ) (h : n > 0) : sequence n = 3^(n-1) + 2 :=
  sorry

end general_term_l396_396627


namespace ratio_expression_value_l396_396932

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396932


namespace sequence_arithmetic_l396_396714

variable (a b : ℕ → ℤ)

theorem sequence_arithmetic :
  a 0 = 3 →
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →
  b 3 = -2 →
  b 10 = 12 →
  a 8 = 3 :=
by
  intros h1 ha hb3 hb10
  sorry

end sequence_arithmetic_l396_396714


namespace mod_exp_l396_396373

theorem mod_exp (n : ℕ) : (5^303) % 11 = 4 :=
  by sorry

end mod_exp_l396_396373


namespace four_digit_numbers_divisible_by_17_and_even_l396_396561

theorem four_digit_numbers_divisible_by_17_and_even :
  ∃ n, (n = 265 ∧
          (∀ k, 59 ≤ k ∧ k ≤ 588 → (17 * k) % 2 = 0 → 17 * k ∈ finset.range 9999 \ finset.range 1000)) :=
sorry

end four_digit_numbers_divisible_by_17_and_even_l396_396561


namespace zero_function_l396_396088

noncomputable def f : ℝ → ℝ := sorry

theorem zero_function :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro h
  sorry

end zero_function_l396_396088


namespace avg_weight_A_and_B_l396_396703

-- Definitions for the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- Statement to be proved
theorem avg_weight_A_and_B : condition1 A B C ∧ condition2 A B C ∧ condition3 B → (A + B) / 2 = 40 :=
begin
  intros h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2,
  sorry -- Proof goes here
end

end avg_weight_A_and_B_l396_396703


namespace sequence_general_term_l396_396553

noncomputable def a : ℕ → ℚ
| 0     := 1/2
| (n+1) := a n + 1/(n^2 + 3*n + 2)

theorem sequence_general_term (n : ℕ) (n_pos : 0 < n):
  a n = n / (n + 1) := by
  sorry

end sequence_general_term_l396_396553


namespace sum_of_x_and_reciprocal_eq_3_5_l396_396540

theorem sum_of_x_and_reciprocal_eq_3_5
    (x : ℝ)
    (h : x^2 + (1 / x^2) = 10.25) :
    x + (1 / x) = 3.5 := 
by
  sorry

end sum_of_x_and_reciprocal_eq_3_5_l396_396540


namespace unique_solution_condition_l396_396181

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l396_396181


namespace find_x_from_collinearity_l396_396174

variable (x : ℝ)
def vector_a : ℝ × ℝ := (8, 1/2 * x)
def vector_b : ℝ × ℝ := (x, 1)

def is_collinear (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 - u.2 * v.1 = 0

theorem find_x_from_collinearity (hx : 0 < x) :
  is_collinear (vector_a x - 2 • vector_b x) (2 • vector_a x + vector_b x) → x = 4 :=
by
  intro h
  sorry

end find_x_from_collinearity_l396_396174


namespace dominoes_form_rectangle_l396_396744

theorem dominoes_form_rectangle :
  ∃ (rectangle : list (list (nat × nat))),
    (forall (r : list (nat × nat)), r ∈ rectangle -> length r = 4) ∧
    (forall (c : list (nat × nat)), (list.transpose rectangle) ∈ c -> length c = 3) ∧
    (∀ r ∈ rectangle, list.sum (list.map prod.fst r) = 4) ∧
    (∀ c, c ∈ list.transpose rectangle -> list.sum (list.map prod.fst c) = 3) ∧
    length rectangle = 3 := sorry

end dominoes_form_rectangle_l396_396744


namespace cost_price_percentage_l396_396708

theorem cost_price_percentage (MP CP : ℝ) (discount gain : ℝ) 
  (h_discount : discount = 0.16)
  (h_gain : gain = 0.3125)
  (h_SP_eq : CP * (1 + gain) = MP * (1 - discount)) :
  CP / MP = 0.64 :=
by
  rw [h_discount, h_gain] at h_SP_eq
  have h1 : MP * 0.84 = CP * 1.3125,
  { exact h_SP_eq }
  have h2 : CP / MP = 0.84 / 1.3125,
  { field_simp at h1,
    exact_mod_cast h1.symm }
  have h3 : 0.84 / 1.3125 = 0.64,
  { norm_num }
  rw h3
  -- sorry step to conclude the proof here, it is correct by calculation
  sorry

end cost_price_percentage_l396_396708


namespace intersection_and_union_when_m_eq_3_range_of_m_when_union_eq_A_l396_396989

-- Definitions and conditions
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m - 1 ≤ x ∧ x ≤ 2 * m + 1 }

-- Proof formulation for part 1
theorem intersection_and_union_when_m_eq_3 : A ∩ B 3 = { x | 2 ≤ x ∧ x ≤ 5 } ∧ (Aᶜ ∪ B 3) = { x | x < -2 ∨ x ≥ 2 } :=
sorry

-- Proof formulation for part 2
theorem range_of_m_when_union_eq_A : {m | ∀ x, ((A ∪ B m) = A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) } :=
sorry

end intersection_and_union_when_m_eq_3_range_of_m_when_union_eq_A_l396_396989


namespace casey_pumping_time_l396_396457

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end casey_pumping_time_l396_396457


namespace probability_sum_greater_than_10_eq_2_5_l396_396622

noncomputable def probability_of_sum_greater_than_10 : ℝ :=
  let interval := set.Icc (0 : ℝ) 10
  let event := set.Ioo 6 10
  (set.measure_of event interval).toReal

theorem probability_sum_greater_than_10_eq_2_5 :
  probability_of_sum_greater_than_10 = (2 / 5) :=
sorry

end probability_sum_greater_than_10_eq_2_5_l396_396622


namespace find_n_l396_396842

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∃ k d : ℕ, prime k ∧ k ∣ n ∧ d ∣ n ∧ d > 1 ∧ n = k^2 + d^2) → 
  n = 8 ∨ n = 20 :=
by sorry

end find_n_l396_396842


namespace jogging_track_circumference_l396_396722

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end jogging_track_circumference_l396_396722


namespace radius_of_inscribed_circle_l396_396611

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end radius_of_inscribed_circle_l396_396611


namespace unsolvable_triangle_conditions_l396_396204

theorem unsolvable_triangle_conditions :
  ∃ (a b c : ℝ) (α β γ : ℝ),
  (a < b ∧ b < c) ∧ (a + d = b ∧ b + d = c) ∧
  (α + β + γ = 180) ∧ (α < β ∧ β < γ) ∧ ((area_of_triangle a b c = 50) ∧ (circumscribed_radius a b c = 10)) → false :=
by
  sorry

end unsolvable_triangle_conditions_l396_396204


namespace savings_by_end_of_2019_l396_396632

variable (income_monthly : ℕ → ℕ) (expenses_monthly : ℕ → ℕ)
variable (initial_savings : ℕ)

noncomputable def total_income : ℕ :=
  (income_monthly 9 + income_monthly 10 + income_monthly 11 + income_monthly 12) * 4

noncomputable def total_expenses : ℕ :=
  (expenses_monthly 9 + expenses_monthly 10 + expenses_monthly 11 + expenses_monthly 12) * 4

noncomputable def final_savings (initial_savings : ℕ) (total_income : ℕ) (total_expenses : ℕ) : ℕ :=
  initial_savings + total_income - total_expenses

theorem savings_by_end_of_2019 :
  (income_monthly 9 = 55000) →
  (income_monthly 10 = 45000) →
  (income_monthly 11 = 10000) →
  (income_monthly 12 = 17400) →
  (expenses_monthly 9 = 40000) →
  (expenses_monthly 10 = 20000) →
  (expenses_monthly 11 = 5000) →
  (expenses_monthly 12 = 2000) →
  initial_savings = 1147240 →
  final_savings initial_savings total_income total_expenses = 1340840 :=
by
  intros h_income_9 h_income_10 h_income_11 h_income_12
         h_expenses_9 h_expenses_10 h_expenses_11 h_expenses_12
         h_initial_savings
  rw [final_savings, total_income, total_expenses]
  rw [h_income_9, h_income_10, h_income_11, h_income_12]
  rw [h_expenses_9, h_expenses_10, h_expenses_11, h_expenses_12]
  rw h_initial_savings
  sorry

end savings_by_end_of_2019_l396_396632


namespace polynomial_solution_l396_396871

theorem polynomial_solution (A : ℝ) (n : ℤ) (P : polynomial ℝ) :
  (2 ≤ n ∧ n ≤ 19) →
  (P.eval (P.eval (P.eval x)) = A * x^n + 19 * x + 99) →
  (A = 0 → P = polynomial.C (root_of 3 19) * polynomial.X + polynomial.C (99 / (root_of (2 / 3) 19 + root_of (1 / 3) 19 + 1))) ∧
  (A ≠ 0 → ¬ ∃ P, P.eval (P.eval (P.eval x)) = A * x^n + 19 * x + 99) :=
by sorry

end polynomial_solution_l396_396871


namespace binomial_7_2_l396_396065

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396065


namespace binom_7_2_eq_21_l396_396054

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396054


namespace binomial_7_2_eq_21_l396_396042

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396042


namespace kristy_baked_cookies_l396_396642

theorem kristy_baked_cookies (C : ℕ) :
  (C - 3) - 8 - 12 - 16 - 6 - 14 = 10 ↔ C = 69 := by
  sorry

end kristy_baked_cookies_l396_396642


namespace convex_2k_vertices_l396_396965

theorem convex_2k_vertices (k : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ 50)
    (P : Finset (EuclideanSpace ℝ (Fin 2)))
    (hP : P.card = 100) (M : Finset (EuclideanSpace ℝ (Fin 2)))
    (hM : M.card = k) : 
  ∃ V : Finset (EuclideanSpace ℝ (Fin 2)), V.card = 2 * k ∧ ∀ m ∈ M, m ∈ convexHull ℝ V :=
by
  sorry

end convex_2k_vertices_l396_396965


namespace geometry_problem_l396_396129

theorem geometry_problem (
  h1 : ∃ p : ℝ × ℝ, (2 * p.1 - p.2 - 3 = 0) ∧ (4 * p.1 - 3 * p.2 - 5 = 0),
  h2 : ∃ l : ℝ → ℝ, (∀ pt : ℝ × ℝ, (x+y−2) = 0 → l(pt.1) = pt.2) ∧ 
                    (∀ pt : ℝ × ℝ, p = pt → l(pt.1) = pt.2),
  h3 : (∃ C : ℝ × ℝ × ℝ, (C.1 - 1)^2 + (C.2) = C.3) ∧ 
        (∀ x : ℝ, x.a = 3 → ∃ pt : ℝ × ℝ, pt = (3,0))
  h4 : ∃ r : ℝ, ∃ p : ℝ × ℝ, 2 / sqrt(2) = r
) : (∀ p : ℝ × ℝ, p = (1, 0)) ∧
     (∀ l : ℝ → ℝ, l = x - 1) ∧ 
     (∃ C : ℝ × ℝ × ℝ, C = (x - 3, x^2, (x^2 + y^2=4)) :=
begin
  sorry
end

end geometry_problem_l396_396129


namespace diagonals_in_25_sided_polygon_l396_396823

/-
  Proof Problem: Prove that the number of diagonals in a convex polygon with 25 sides is 275.
-/

theorem diagonals_in_25_sided_polygon : ∀ (n : ℕ), n = 25 → (n * (n - 3)) / 2 = 275 :=
by
  intros n h
  rw h
  sorry

end diagonals_in_25_sided_polygon_l396_396823


namespace domain_of_g_l396_396093

noncomputable def g (x : ℝ) : ℝ := real.sqrt (2 - real.sqrt (4 - real.sqrt (5 - x)))

theorem domain_of_g :
  (∀ x, g x ∈ ℝ ↔ -11 ≤ x ∧ x ≤ 5) :=
by
  sorry

end domain_of_g_l396_396093


namespace find_omega_find_g_min_max_l396_396163

open Real

def f (ω x : ℝ) := sin (ω * x - π / 6) + sin (ω * x - π / 2)
def g (x : ℝ) := sqrt 3 * sin (x - π / 12)

theorem find_omega (ω : ℝ) (h₀ : 0 < ω) (h₁ : ω < 3) (h₂ : f ω (π / 6) = 0) : ω = 2 :=
sorry

theorem find_g_min_max : 
  ∃ (min max : ℝ), min = -3 / 2 ∧ max = sqrt 3 ∧
    ∀ (x : ℝ), -π / 4 ≤ x ∧ x ≤ 3 * π / 4 → (g x ≥ min ∧ g x ≤ max) :=
sorry

end find_omega_find_g_min_max_l396_396163


namespace opposite_of_inv_23_l396_396335

theorem opposite_of_inv_23 : ∀ x : ℚ, x = 1 / 23 → -x = -1 / 23 :=
by
  intro x h
  rw h
  apply neg_eq_neg_of_eq
  refl

end opposite_of_inv_23_l396_396335


namespace find_a_and_intervals_and_extrema_l396_396597

def f (a x : ℝ) := a * x^2 + 2 * x - (4 / 3) * Real.log x

def f_prime (a x : ℝ) := 2 * a * x + 2 - (4 / 3) * (1 / x)

theorem find_a_and_intervals_and_extrema :
  (f_prime a 1 = 0) →
  a = -(1 / 3) ∧
  (∀ x, 0 < x → x < 1 → f_prime (-1 / 3) x < 0) ∧ 
  (∀ x, 1 < x → x < 2 → f_prime (-1 / 3) x > 0) ∧ 
  (∀ x, 2 < x → f_prime (-1 / 3) x < 0) ∧ 
  f (-1 / 3) 1 = 5 / 3 ∧ 
  f (-1 / 3) 2 = (8 / 3) - (4 / 3) * Real.log 2 :=
by
  sorry

end find_a_and_intervals_and_extrema_l396_396597


namespace element_with_mass_percentage_is_nitrogen_l396_396097

section DinitrogenTrioxide
variable (M_N : ℝ) (M_O : ℝ) (mass_percentage : ℝ)
variable (m_mass_N2O3 : ℝ)

def dinitrogen_trioxide : Bool :=
  let M_N2O3 := 2 * M_N + 3 * M_O
  let mass_N := (mass_percentage / 100) * M_N2O3
  abs (mass_N - 2 * M_N) < 0.05

theorem element_with_mass_percentage_is_nitrogen
  (M_N : 14.01) (M_O : 16.00) (mass_percentage : 36.84) :
  dinitrogen_trioxide M_N M_O mass_percentage = True :=
  by
    -- The detailed proof steps would go here to show the element with 36.84% is nitrogen
    sorry
end DinitrogenTrioxide

end element_with_mass_percentage_is_nitrogen_l396_396097


namespace smallest_positive_period_and_max_value_tan_beta_value_l396_396899

noncomputable def f (x : ℝ) := (Real.sin (2 * x - π / 4)) + (Real.cos (2 * x - 3 * π / 4))

theorem smallest_positive_period_and_max_value :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ M, ∀ x, f x ≤ M) ∧ (∀ x, f x = 2) :=
sorry

theorem tan_beta_value {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hfα : f α = Real.sqrt 2) (hcos : Real.cos (α + β) = 1 / 3) :
  Real.tan β = (9 - 4 * Real.sqrt 2) / 7 :=
sorry

end smallest_positive_period_and_max_value_tan_beta_value_l396_396899


namespace calc_value_l396_396022

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l396_396022


namespace no_zero_digit_pairs_count_l396_396495

def has_no_zero_digit (n : ℕ) : Prop :=
  ¬ (n.toDigits 10).any (λ d, d = 0)

theorem no_zero_digit_pairs_count : 
  (∃ n : ℕ, (∀ a b : ℕ, a + b = 500 → a > 0 → b > 0 → has_no_zero_digit a → has_no_zero_digit b → n = 410)) :=
by
  sorry

end no_zero_digit_pairs_count_l396_396495


namespace butterfly_theorem_proj_l396_396743

/-- Define points and givens for the setup -/
variables {A B O E F F' : ℝ}

/-- Assume F' is the symmetric point of F with respect to O -/
def symmetric_point (F O : ℝ) : ℝ := 2*O - F

/-- Assume the invariance of cross-ratio under projective transformation from problem 30.2(b) -/
axiom projective_transform_cross_ratio_invariance 
  {a b o e f' : ℝ} : (a / b) = (f' / o) → (a / o) = (f' / b)

/-- Butterfly Problem: Assuming the conditions above, E is the symmetric point of F with respect to O -/
theorem butterfly_theorem_proj (h1 : F' = symmetric_point F O)
  (h2 : projective_transform_cross_ratio_invariance (A, B, O, E) (B, A, F', O)) :
  E = F' := 
  sorry

end butterfly_theorem_proj_l396_396743


namespace probability_of_odd_after_removal_is_122_over_720_l396_396468

noncomputable def probability_odd_face_after_removal : ℚ :=
let total_dots := 36 in
let prob_remove_from : ℕ → ℚ := λ n, n / total_dots in
let prob_no_removal : ℕ → ℚ := λ n, (total_dots - n) / total_dots * (total_dots - 1 - n) / (total_dots - 1) in
let prob_single_removal : ℕ → ℚ := λ n, 2 * (n / total_dots) * (total_dots - 1 - n) / (total_dots - 1) in
let prob_odd_face : ℕ → ℚ := λ n, if n = 7 then prob_no_removal n else prob_single_removal n in
let odd_faces := [1, 3, 5, 7] in
let total_prob := (1 / 8) * (prob_odd_face 1 + prob_odd_face 3 + prob_odd_face 5 + prob_odd_face 7) in
total_prob

theorem probability_of_odd_after_removal_is_122_over_720 :
  probability_odd_face_after_removal = 122 / 720 :=
by
  sorry

end probability_of_odd_after_removal_is_122_over_720_l396_396468


namespace distance_between_lighthouses_l396_396223

variable (a : ℝ)
variable (AC BC : ℝ) (angleACB : ℝ)

-- Conditions
def condition1 := AC = a
def condition2 := BC = a
def condition3 := angleACB = 120 * Real.pi / 180

-- Cosine theorem and distance calculation
theorem distance_between_lighthouses (AC BC : ℝ) (angleACB : ℝ) (h1 : AC = a) (h2 : BC = a) (h3 : angleACB = 120 * Real.pi / 180) : 
  let AB := sqrt (AC^2 + BC^2 - 2 * AC * BC * Real.cos angleACB)
  in AB = sqrt 3 * a
by
  sorry

end distance_between_lighthouses_l396_396223


namespace order_alpha_beta_gamma_l396_396117

open Real

theorem order_alpha_beta_gamma (α β γ : ℝ) (hα₀ : 0 < α) (hα₁ : α < π / 2)
  (hβ₀ : 0 < β) (hβ₁ : β < π / 2)
  (hγ₀ : 0 < γ) (hγ₁ : γ < π / 2)
  (hα : cot α = α) 
  (hβ : sin (cot β) = β) 
  (hγ : cot (sin γ) = γ) : 
  β < α ∧ α < γ :=
  sorry

end order_alpha_beta_gamma_l396_396117


namespace Montoya_budget_spent_on_food_l396_396313

-- Define the fractions spent on groceries and going out to eat
def groceries_fraction : ℝ := 0.6
def eating_out_fraction : ℝ := 0.2

-- Define the total fraction spent on food
def total_food_fraction (g : ℝ) (e : ℝ) : ℝ := g + e

-- The theorem to prove
theorem Montoya_budget_spent_on_food : total_food_fraction groceries_fraction eating_out_fraction = 0.8 := 
by
  -- the proof will go here
  sorry

end Montoya_budget_spent_on_food_l396_396313


namespace floor_sqrt_99_l396_396482

theorem floor_sqrt_99 : ⌊real.sqrt 99⌋ = 9 := 
by {
  -- Definitions related to the conditions
  let n := 9,
  let m := n + 1,
  have h1 : n ^ 2 = 81 := by norm_num,
  have h2 : m ^ 2 = 100 := by norm_num,

  -- Conditions from the problem
  have h3 : n ^ 2 < 99 := by norm_num,
  have h4 : 99 < m ^ 2 := by norm_num,

  -- Conclude the answer
  have h5 : (n : ℝ) < real.sqrt 99 := by sorry,
  have h6 : real.sqrt 99 < (m : ℝ) := by sorry,
  
  -- Final conclusion using the floor property
  exact int.floor_of_nonneg_of_le_ceil _ h5 h6,
}

end floor_sqrt_99_l396_396482


namespace solve_fraction_equation_l396_396692

theorem solve_fraction_equation (x : ℚ) :
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 := 
by
  sorry

end solve_fraction_equation_l396_396692


namespace find_k_l396_396191

theorem find_k (k x y : ℝ) (h_ne_zero : k ≠ 0) (h_x : x = 4) (h_y : y = -1/2) (h_eq : y = k / x) : k = -2 :=
by
  -- This is where the proof would go
  sorry

end find_k_l396_396191


namespace sequence_converges_to_zero_l396_396746

noncomputable theory

def a : ℕ → ℝ 
| 0 := arbitrary ℝ -- arbitrary initial value a_0
| (n + 1) := Real.sin (a n) -- recursive definition a_{n+1} = sin(a_n)

theorem sequence_converges_to_zero : ∃ L : ℝ, (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - L| < ε) ∧ L = 0 :=
by 
  -- Proof omitted
  sorry

end sequence_converges_to_zero_l396_396746


namespace cartesian_equation_of_C_length_of_chord_l396_396975

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

noncomputable def find_intersection_points (A B : ℝ × ℝ) : ℝ := abs (fst A - fst B)

theorem cartesian_equation_of_C (ρ θ : ℝ) : 
  (∃ ρ θ, ρ * sin(θ) ^ 2 = 8 * cos(θ)) → ∃ x y, y ^ 2 = 8 * x :=
sorry

theorem length_of_chord (t : ℝ) 
  (h1 : t = 8) (h2 : t = -8 / 3) : find_intersection_points (8, 0) (-8 / 3, 0) = 32 / 3 :=
sorry

end cartesian_equation_of_C_length_of_chord_l396_396975


namespace scout_troop_profit_l396_396796

-- Defining the basic conditions as Lean definitions
def num_bars : ℕ := 1500
def cost_rate : ℚ := 3 / 4 -- rate in dollars per bar
def sell_rate : ℚ := 2 / 3 -- rate in dollars per bar

-- Calculate total cost, total revenue, and profit
def total_cost : ℚ := num_bars * cost_rate
def total_revenue : ℚ := num_bars * sell_rate
def profit : ℚ := total_revenue - total_cost

-- The final theorem to be proved
theorem scout_troop_profit : profit = -125 := by
  sorry

end scout_troop_profit_l396_396796


namespace marble_count_calculation_l396_396583

theorem marble_count_calculation (y b g : ℕ) (x : ℕ)
  (h1 : y = 2 * x)
  (h2 : b = 3 * x)
  (h3 : g = 4 * x)
  (h4 : g = 32) : y + b + g = 72 :=
by
  sorry

end marble_count_calculation_l396_396583


namespace solution_set_inequality_l396_396187

theorem solution_set_inequality (m : ℝ) (x : ℝ) 
  (h : 3 - m < 0) : (2 - m) * x + m > 2 ↔ x < 1 :=
by
  sorry

end solution_set_inequality_l396_396187


namespace area_of_square_l396_396008

theorem area_of_square 
  (a : ℝ)
  (h : 4 * a = 28) :
  a^2 = 49 :=
sorry

end area_of_square_l396_396008


namespace total_money_made_from_jerseys_l396_396314

def price_per_jersey : ℕ := 76
def jerseys_sold : ℕ := 2

theorem total_money_made_from_jerseys : price_per_jersey * jerseys_sold = 152 := 
by
  -- The actual proof steps will go here
  sorry

end total_money_made_from_jerseys_l396_396314


namespace no_integral_root_l396_396282

theorem no_integral_root (f : ℤ[X]) (k : ℕ) (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬k ∣ f.eval i) : 
  ¬∃ n : ℤ, f.eval n = 0 :=
begin
  sorry -- Proof to be provided
end

end no_integral_root_l396_396282


namespace fraction_of_work_left_l396_396380

theorem fraction_of_work_left (a_days b_days : ℕ) (together_days : ℕ) 
    (h_a : a_days = 15) (h_b : b_days = 20) (h_together : together_days = 4) : 
    (1 - together_days * ((1/a_days : ℚ) + (1/b_days))) = 8/15 := by
  sorry

end fraction_of_work_left_l396_396380


namespace binomial_7_2_l396_396063

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396063


namespace range_of_m_l396_396188

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem range_of_m (m : ℝ) (h : second_quadrant (m-3) (m-2)) : 2 < m ∧ m < 3 :=
sorry

end range_of_m_l396_396188


namespace angle_PFA_90_ratio_AF_FB_5_1_prism_volume_l396_396770

variables (A B C A1 B1 C1 F N P : Type) [right_triangular_prism A B C A1 B1 C1]
-- Constants satisfying the problem statements
variable (AC_diameter : ∃ S : Sphere, S.diameter = segment A C ∧ S.intersects_segment_at F N ∧ F ≠ A ∧ F ≠ B ∧ N ≠ B ∧ N ≠ C)
variable (Intersections : ∃ P, segment C1 F ∩ segment A1 N = {P})
variable (h_A1N : segment_length A1 N = 7)
variable (h_C1P : segment_length C1 P = 6)
variable (h_AB : segment_length A B = 6)

-- Part (a) - Proving the angle PFA is 90 degrees
theorem angle_PFA_90 : angle P F A = 90 := by
  sorry

-- Part (b) - Proving the ratio AF : FB = 5 : 1
theorem ratio_AF_FB_5_1 : ratio (segment_length A F) (segment_length F B) = 5 := by
  sorry

-- Part (c) - Proving the volume of the prism
theorem prism_volume : prism_volume A B C A1 B1 C1 = 21 * sqrt 10 := by
  sorry

end angle_PFA_90_ratio_AF_FB_5_1_prism_volume_l396_396770


namespace popsicle_count_l396_396211

theorem popsicle_count (total_popsicles cherry_popsicles banana_popsicles grape_popsicles : ℕ) 
    (h1 : total_popsicles = 17) 
    (h2 : cherry_popsicles = 13) 
    (h3 : banana_popsicles = 2) 
    (h4 : grape_popsicles = total_popsicles - cherry_popsicles - banana_popsicles) 
    : grape_popsicles = 2 := 
by
  rw [h1, h2, h3] at h4
  exact h4.symm

end popsicle_count_l396_396211


namespace value_range_of_f_in_interval_l396_396735

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem value_range_of_f_in_interval : 
  ∀ x, (2 ≤ x ∧ x ≤ 4) → (1/2 ≤ f x ∧ f x ≤ 2/3) := 
by
  sorry

end value_range_of_f_in_interval_l396_396735


namespace sequence_a_general_term_sum_T_sequence_c_l396_396873

open Nat

noncomputable def sequence_a (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n-2)

noncomputable def sum_S (n : ℕ) : ℝ := ∑ k in range n, sequence_a k

noncomputable def sequence_b (n : ℕ) : ℝ := 4 - 2 * n

noncomputable def sequence_c (n : ℕ) : ℝ := (sequence_b n) / (sequence_a n)

noncomputable def sum_T (n : ℕ) : ℝ := ∑ k in range n, sequence_c k

theorem sequence_a_general_term (n : ℕ) (hn : n > 0) : sequence_a n = 2^(n-2) :=
by
  sorry

theorem sum_T_sequence_c (n : ℕ) (hn : n > 0) : sum_T n = (8 * n / 2^n) :=
by
  sorry

end sequence_a_general_term_sum_T_sequence_c_l396_396873


namespace categorical_variables_l396_396895

-- Define the given variables as types
inductive Variable
| Smoking
| Gender
| ReligiousBelief
| Nationality

-- Define what makes a variable categorical
def isCategorical : Variable → Prop
| Variable.Smoking := true  -- Smoking could be considered categorical in broader classification
| Variable.Gender := true
| Variable.ReligiousBelief := true
| Variable.Nationality := true

-- State the theorem
theorem categorical_variables :
  isCategorical Variable.Gender ∧
  isCategorical Variable.ReligiousBelief ∧
  isCategorical Variable.Nationality :=
by
  -- proof steps are skipped
  sorry

end categorical_variables_l396_396895


namespace binomial_7_2_l396_396058

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396058


namespace value_of_a_l396_396948

theorem value_of_a (x a : ℤ) (h : x = 3 ∧ x^2 = a) : a = 9 :=
sorry

end value_of_a_l396_396948


namespace simson_line_bisects_segment_l396_396605

open EuclideanGeometry

variables (P Q R S H K : Point)
variables (c : Circle)
variables (hPQRS : P ∈ c ∧ Q ∈ c ∧ R ∈ c ∧ S ∈ c)
variables (hPSR : ∠ P S R = π / 2)
variables (hH : Perpendicular Q P R H)
variables (hK : Perpendicular Q R S K)
variables (h_collinear : Collinear_segment_segment P Q R S)

theorem simson_line_bisects_segment (P Q R S H K : Point) 
  (c : Circle) 
  (hPQRS : P ∈ c ∧ Q ∈ c ∧ R ∈ c ∧ S ∈ c) 
  (hPSR : ∠ P S R = π / 2) 
  (hH : Perpendicular Q P R H) 
  (hK : Perpendicular Q R S K) 
  (h_collinear: Collinear_segment_segment P Q R S): 
  Bisection HK SQ := 
sorry

end simson_line_bisects_segment_l396_396605


namespace amy_baked_l396_396439

-- Definition of total muffins brought from Monday to Friday
def muffins_brought_to_school (monday : ℕ) : ℕ :=
  let tuesday := monday + 1 in
  let wednesday := tuesday + 1 in
  let thursday := wednesday + 1 in
  let friday := thursday + 1 in
  monday + tuesday + wednesday + thursday + friday

-- Definition of muffins left on Saturday
def muffins_left := 7

-- Theorem stating total muffins Amy baked
theorem amy_baked (original_muffins : ℕ) : original_muffins = muffins_brought_to_school 1 + muffins_left := by
  sorry

end amy_baked_l396_396439


namespace numberOfComplexOrderedPairs_l396_396102

noncomputable def complexOrderedPairs : Set (ℂ × ℂ) :=
  {p | let a := p.1, b := p.2 in a^4 * b^6 = 1 ∧ a^8 * b^3 = 1}

theorem numberOfComplexOrderedPairs : Fintype.card complexOrderedPairs = 12 :=
by
  sorry

end numberOfComplexOrderedPairs_l396_396102


namespace steven_peach_apple_difference_l396_396638

theorem steven_peach_apple_difference :
  ∀ (peaches apples : ℕ), peaches = 17 → apples = 16 → peaches - apples = 1 :=
by
  intros peaches apples h_peaches h_apples
  rw [h_peaches, h_apples]
  exact Nat.sub_self 1

end steven_peach_apple_difference_l396_396638


namespace isosceles_triangle_area_l396_396718

open Real

noncomputable def area_of_isosceles_triangle (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

theorem isosceles_triangle_area :
  ∃ (b : ℝ) (l : ℝ), h = 8 ∧ (2 * l + b = 32) ∧ (area_of_isosceles_triangle b h = 48) :=
by
  sorry

end isosceles_triangle_area_l396_396718


namespace quadratic_passing_points_l396_396152

theorem quadratic_passing_points : 
  ∃ f : ℝ → ℝ, 
    (∀ x, f x = -x^2 + 4) ∧ 
    (f 2 = 0) ∧ (f 0 = 4) ∧ (f (-2) = 0) := 
by
  use (λ x : ℝ, -x^2 + 4)
  split
  { intro x, refl }
  repeat { split; norm_num }

end quadratic_passing_points_l396_396152


namespace alex_shirts_l396_396426

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l396_396426


namespace expression_is_4045_l396_396713

theorem expression_is_4045 :
  ∃ q : ℕ, (4035 + 1) ∣ (2022 + 1) * 2022 ∧
  𝔽 2023 2022 - 𝔽 2022 2023 = 𝔽 4045 q ∧ 
  Nat.gcd 2022 q == 1 :=
by
  sorry

end expression_is_4045_l396_396713


namespace floor_sqrt_99_eq_9_l396_396485

theorem floor_sqrt_99_eq_9 :
  ∀ x, 81 ≤ x ∧ x < 100 → floor (real.sqrt 99) = 9 :=
by
  sorry

end floor_sqrt_99_eq_9_l396_396485


namespace day_200th_of_year_N_minus_1_is_Wednesday_l396_396976

-- Define the basic conditions given in the problem
def day_of_year_N (d : ℕ) : nat := (d % 7)
def day_of_week (day : nat) : Prop :=
  day_of_year_N day = 1   -- 1 represents Wednesday

-- Assume the given conditions
axiom condition_400th_day_of_N_is_Wednesday : day_of_week 400
axiom condition_300th_day_of_N_plus_2_is_Wednesday : day_of_week (300 + 2 * 365 + 1) -- considering 1 leap year

-- Define the year calculations as derived and reasoned in the problem
def day_200th_of_N_minus_1 (d : ℕ) : nat :=
  (d - 365) % 7

-- The statement to prove
theorem day_200th_of_year_N_minus_1_is_Wednesday :
  day_of_week (day_200th_of_N_minus_1 1) :=
sorry

end day_200th_of_year_N_minus_1_is_Wednesday_l396_396976


namespace sum_of_log_divisors_eq_540_l396_396731

theorem sum_of_log_divisors_eq_540 (n : ℕ) (h : ∑ i in (finset.range (n+1)).product (finset.range (n+1)), real.log10 (2^i.1 * 3^i.2) = 540) : n = 10 := 
sorry

end sum_of_log_divisors_eq_540_l396_396731


namespace number_of_elements_in_intersection_l396_396555

theorem number_of_elements_in_intersection :
  let A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let B := {p : ℝ × ℝ | p.2 = p.1}
  (A ∩ B).finite.to_finset.card = 2 :=
by
  sorry

end number_of_elements_in_intersection_l396_396555


namespace ratio_of_numbers_l396_396732

-- Definitions for the conditions
variable (S L : ℕ)

-- Given conditions
def condition1 : Prop := S + L = 44
def condition2 : Prop := S = 20
def condition3 : Prop := L = 6 * S

-- The theorem to be proven
theorem ratio_of_numbers (h1 : condition1 S L) (h2 : condition2 S) (h3 : condition3 S L) : L / S = 6 := 
  sorry

end ratio_of_numbers_l396_396732


namespace hyperbola_eccentricity_l396_396868

-- Define the conditions
variables {a b : ℝ} (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) (h_b_gt_a : b > a > 0)
variables {l1 l2 : ℝ}

-- Define the additional conditions related to the points and arithmetic progression
variables {F A B O : ℝ} (h_perpendicular : ∀ x : ℝ, x ∉ F)
          (h_oa_ob_ap : ∀ oa ab ob : ℝ, oa + ob = 2 * ab ∧ ab = (ob - oa))

-- The goal is to prove
theorem hyperbola_eccentricity : ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l396_396868


namespace log_function_monotonically_decreasing_l396_396719

open Real

/-- Proving the interval in which the function y = log_{1/3}(x^2 - 2x - 3) is monotonically decreasing. -/
theorem log_function_monotonically_decreasing :
  {x : ℝ | x < -1 ∨ x > 3} ⊆
  {x : ℝ | ∀ x' > 3, log (1/3) (x^2 - 2*x - 3) = log (1/3) (x'^2 - 2*x' - 3) → x = x'} :=
sorry

end log_function_monotonically_decreasing_l396_396719


namespace value_b_2_100_l396_396650

section
open BigOperators

def sequence (b : ℕ → ℕ) : Prop :=
  (b 1 = 2) ∧ ∀ n : ℕ, b (2 * n) = (n + 1) * b n

theorem value_b_2_100 (b : ℕ → ℕ) (h : sequence b) : 
  b (2^100) = 2^2 * ∏ k in finset.range 100, (2^k + 1) := 
sorry

end

end value_b_2_100_l396_396650


namespace problem_division_l396_396364

theorem problem_division :
  (3^2 + 5^2) / (3^-2 + 5^-2) = 225 := by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 3^-2 = 1 / 9 := by norm_num
  have h4 : 5^-2 = 1 / 25 := by norm_num
  calc
    (3^2 + 5^2) / (3^-2 + 5^-2)
        = (9 + 25) / (1/9 + 1/25) : by rw [h1, h2, h3, h4]
    ... = 34 / (1/9 + 1/25)       : by norm_num
    ... = 34 / (25/225 + 9/225)   : by norm_num
    ... = 34 / (34/225)           : by norm_num
    ... = 34 * 225 / 34           : by field_simp
    ... = 225                     : by norm_num

end problem_division_l396_396364


namespace equal_lateral_edges_iff_sphere_inscribed_l396_396677

structure TruncatedPyramid (V : Type) [MetricSpace V] :=
(base1 base2 : Finset V)
(lateralFaces : List (Set V))
(num_faces : lateralFaces.length = 3)
(trapezoidFaces : ∀ f ∈ lateralFaces, ∃ a b c d : V, [a, b, c, d] ⊆ f ∧ IsTrapezoid [a, b, c, d])

def SphereInscribedAroundTruncatedPyramid (V : Type) [MetricSpace V] (P : TruncatedPyramid V) :=
∃ O : V, ∀ v ∈ P.base1 ∪ P.base2, dist O v = r

theorem equal_lateral_edges_iff_sphere_inscribed {V : Type} [MetricSpace V] 
  (P : TruncatedPyramid V) :
  (∀ f ∈ P.lateralFaces, ∃ a b c d : V, IsIsoscelesTrapezoid [a, b, c, d]) ↔ 
  SphereInscribedAroundTruncatedPyramid V P :=
sorry

end equal_lateral_edges_iff_sphere_inscribed_l396_396677


namespace binomial_7_2_l396_396038

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396038


namespace value_mn_l396_396549

-- Define conditions
variables (m n : ℝ)
-- First condition: perpendicular lines
def perp_condition : Prop := m - 2 * n = 0

-- Second condition: shortest distance from a point to a circle
def distance_condition : Prop := real.sqrt ((2 - m)^2 + (5 - n)^2) - 1 = 3

-- Theorem: value of mn
theorem value_mn (h1 : perp_condition m n) (h2 : distance_condition m n) : m * n = 2 :=
by sorry

end value_mn_l396_396549


namespace total_charge_for_trip_l396_396983

-- Definitions based on the conditions
def initial_fee : ℝ := 2.0
def additional_charge_per_increment : ℝ := 0.35
def increment_distance : ℝ := 2 / 5
def trip_distance : ℝ := 3.6

-- Statement to prove the total charge for a trip of 3.6 miles
theorem total_charge_for_trip : ((trip_distance / increment_distance) * additional_charge_per_increment) + initial_fee = 5.15 := by
  sorry

end total_charge_for_trip_l396_396983


namespace smallest_product_of_digits_4_5_6_7_l396_396279

theorem smallest_product_of_digits_4_5_6_7 : ∃ (a b : ℕ), 
  (a / 10 ∈ {4, 5} ∧ a % 10 ∈ {6, 7} ∧
  b / 10 ∈ {4, 5} ∧ b % 10 ∈ {6, 7} ∧
  a ≠ b ∧ a * b = 2622) :=
sorry

end smallest_product_of_digits_4_5_6_7_l396_396279


namespace probability_last_digit_8_l396_396114

theorem probability_last_digit_8 :
  let S := Finset.range 100
  let event_occurs (a b : ℕ) : Prop := (3^a + 7^b) % 10 = 8
  let possible_pairs := S.product S
  let favorable_pairs := possible_pairs.filter (λ (ab : ℕ × ℕ), event_occurs ab.1 ab.2)
  let probability := favorable_pairs.card.toRat / possible_pairs.card.toRat
  probability = 3 / 16 := 
sorry

end probability_last_digit_8_l396_396114


namespace percent_women_surveryed_equal_40_l396_396203

theorem percent_women_surveryed_equal_40
  (W M : ℕ) 
  (h1 : W + M = 100)
  (h2 : (W / 100 * 1 / 10 : ℚ) + (M / 100 * 1 / 4 : ℚ) = (19 / 100 : ℚ))
  (h3 : (9 / 10 : ℚ) * (W / 100 : ℚ) + (3 / 4 : ℚ) * (M / 100 : ℚ) = (1 - 19 / 100 : ℚ)) :
  W = 40 := 
sorry

end percent_women_surveryed_equal_40_l396_396203


namespace consecutive_even_numbers_l396_396686

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end consecutive_even_numbers_l396_396686


namespace domain_of_f_l396_396711

-- Define the function
def f (x : ℝ) : ℝ := real.sqrt (x^3 - 1)

-- State the theorem
theorem domain_of_f : ∀ x : ℝ, x^3 - 1 ≥ 0 ↔ x ≥ 1 := by
  sorry

end domain_of_f_l396_396711


namespace cost_of_bricks_l396_396355

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end cost_of_bricks_l396_396355


namespace prove_properties_l396_396775

variable {α : Type*} [LinearOrderedField α]

-- Given conditions
def close_numbers (a : List α) (n : α) :=
  n > 1 ∧ (∀ x ∈ a, x < (a.sum / (n - 1)))

-- Given sum
def sum_of_list (a : List α) : α := a.sum

-- To prove: All numbers are positive
def all_positive (a : List α) :=
  ∀ x ∈ a, x > 0

-- To prove: First two numbers are greater than the third
def first_two_greater_than_third (a : List α) :=
  nth_le a 0 (by sorry) + nth_le a 1 (by sorry) > nth_le a 2 (by sorry)

-- To prove: First two numbers greater than the sum divided by (n - 1)
def first_two_greater_than_sum_div (a : List α) (n : α) :=
  nth_le a 0 (by sorry) + nth_le a 1 (by sorry) > sum_of_list a / (n - 1)

-- Properties to prove in Lean
theorem prove_properties (a : List α) (n : α)
  (h_close : close_numbers a n) :
  all_positive a ∧
  (length a > 2 → first_two_greater_than_third a) ∧
  first_two_greater_than_sum_div a n := by
  sorry

end prove_properties_l396_396775


namespace f_has_property_P_b_f_monotonic_intervals_g_range_of_m_l396_396246

-- Define the function f and its derivative
def f (x : ℝ) := real.log x + ((b + 2) / (x + 1))
def f_derivative (x : ℝ) := (x^2 - b*x + 1) / (x * (x + 1)^2)

-- Define the property P
def P (a : ℝ) (f_derivative : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x > 1, h x > 0) ∧ ∀ x, f_derivative x = h(x) * (x^2 - a*x + 1)

-- Part (1)
-- (i) f has property P(b)
theorem f_has_property_P_b (x : ℝ) (b : ℝ) (hx : x > 1) : 
  P b f_derivative := sorry

-- (ii) Monotonic intervals of f based on b
theorem f_monotonic_intervals (x : ℝ) (b : ℝ) : 
  (∀ x > 1, if b ≤ 2 then Ψ_monotonically_increasing else (∃ c, x < c ∧ x > c ∧ Ψ_decreasing x < c ∧ Ψ_increasing x > c) := sorry

-- Part (2)
-- Given g(x) with property P(2) and given conditions, find range of m
def g (x : ℝ) : ℝ := sorry  -- Suppose g is some function satisfying P(2)
def g_derivative (x : ℝ) := (x-1)^2 * h x  -- h(x) is positive on (1, +∞)

theorem g_range_of_m (x1 x2 : ℝ) (hx1 : x1 > 1) (hx2 : x2 > 1) (h_ineq : x1 < x2) :
  (∀ m ∈ Ioo 0 1, let α := m * x1 + (1 - m) * x2;
                  let β := (1 - m) * x1 + m * x2 in
                  α > 1 ∧ β > 1 ∧ |g(α) - g(β)| < |g(x1) - g(x2)| ) := sorry

end f_has_property_P_b_f_monotonic_intervals_g_range_of_m_l396_396246


namespace probability_point_between_lines_l396_396470

theorem probability_point_between_lines {x y : ℝ} :
  (∀ x, y = -2 * x + 8) →
  (∀ x, y = -3 * x + 8) →
  0.33 = 0.33 :=
by
  intro hl hm
  sorry

end probability_point_between_lines_l396_396470


namespace small_cubes_have_no_faces_colored_l396_396384

theorem small_cubes_have_no_faces_colored :
  ∀ (k : ℕ), (k = 4) → (k^3 = 64) → (f = (k - 2)^3) → (f = 8) :=
begin
  intros k hk hkc hf,
  rw hk at hf,
  simp at hf,
  linarith,
end

end small_cubes_have_no_faces_colored_l396_396384


namespace kira_can_win_for_any_N_above_100_l396_396450

noncomputable def kira_winning_strategy (N : ℕ) : Prop :=
  N > 100 → ∃ k : ℕ, k > 1 ∧ (N % k = 0 ∨ kira_winning_strategy (N - k) ∧ ∀ m > 1, m ≠ k → kira_winning_strategy (N - m))

theorem kira_can_win_for_any_N_above_100 (N : ℕ) (hN : N > 100) : kira_winning_strategy N :=
sorry

end kira_can_win_for_any_N_above_100_l396_396450


namespace non_monotonic_iff_range_l396_396190

theorem non_monotonic_iff_range (c : ℝ) (m : ℝ) :
  (∃ x : ℝ, f' x * (m < x ∧ x < m + 1) < 0) ↔ (0 < m ∧ m < 1/2) ∨ (1 < m ∧ m < 2) :=
by
  let f := λ x : ℝ, 2 * log x + x ^ 2 - 5 * x + c
  let f' := λ x : ℝ, (2 / x) + 2 * x - 5
  sorry

end non_monotonic_iff_range_l396_396190


namespace combination_7_2_l396_396031

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396031


namespace volume_of_pyramid_l396_396990

variables (a d : ℕ) (M A B C D : Type) [IsSquare ABCD]  
  (MA : segment M A) (MB : segment M B) (MC : segment M C) (AD : segment A D)

def side_length (AD : segment A D) := (4 : ℝ) * real.sqrt 2

theorem volume_of_pyramid (h1 : MA.length = a) (h2 : MB.length = a + 2)(h3 : MC.length = a + 4) (h4 : d = 2) : 
  volume (MABC_d (side_length AD) d) = 64 / 3 :=
sorry

end volume_of_pyramid_l396_396990


namespace area_approximation_l396_396537

-- Define the function and its properties
variable (f : ℝ → ℝ)
variable (cont_f : ∀ x ∈ Icc 0 1, continuous_at f x)
variable (nonneg_f : ∀ x ∈ Icc 0 1, 0 ≤ f x)
variable (bounded_f : ∀ x ∈ Icc 0 1, f x ≤ 1)

-- Define the random points and their counting
variable (N : ℕ)
variable (x : ℕ → ℝ)
variable (y : ℕ → ℝ)
variable (x_uniform : ∀ i, 0 < x i ∧ x i ≤ 1)
variable (y_uniform : ∀ i, 0 < y i ∧ y i ≤ 1)
variable (N1 : ℕ)
variable (count_points : N1 = (finset.univ.filter (λ i, y i ≤ f(x i))).card)

-- Main statement
theorem area_approximation :
  ∫ x in 0..1, f x ≈ (N1 : ℝ) / (N : ℝ) :=
sorry

end area_approximation_l396_396537


namespace find_sam_age_l396_396290

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l396_396290


namespace count_valid_pairs_l396_396493

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d ≠ 0

def is_valid_pair (a b : ℕ) : Prop :=
  a + b = 500 ∧ has_no_zero_digit a ∧ has_no_zero_digit b

theorem count_valid_pairs : 
  (finset.univ.filter (λ (a : ℕ), is_valid_pair a (500 - a))).card = 249 :=
sorry

end count_valid_pairs_l396_396493


namespace centroid_of_parabola_l396_396090

open Real

-- Definition of the region bounded by the parabola and the coordinate axes
def region (a : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ (sqrt p.1 + sqrt p.2 = sqrt a)}

-- First moments about the y-axis
def I1 (a : ℝ) : ℝ :=
  ∫ x in 0..a, x * (a - x)^2 / a dx

-- Second moments about the y-axis (area of the region)
def I2 (a : ℝ) : ℝ :=
  ∫ x in 0..a, (a - x)^2 / a dx

-- Centroid calculation
def centroid_x (a : ℝ) : ℝ := I1 a / I2 a
def centroid_y (a : ℝ) : ℝ := I1 a / I2 a

theorem centroid_of_parabola (a : ℝ) (h : 0 ≤ a) :
  let x_C := centroid_x a
  let y_C := centroid_y a
  (x_C = a / 5) ∧ (y_C = a / 5) :=
by 
  sorry

end centroid_of_parabola_l396_396090


namespace find_second_number_l396_396345

theorem find_second_number 
  (k : ℕ)
  (h_k_is_1 : k = 1)
  (h_div_1657 : ∃ q1 : ℕ, 1657 = k * q1 + 10)
  (h_div_x : ∃ q2 : ℕ, ∀ x : ℕ, x = k * q2 + 7 → x = 1655) 
: ∃ x : ℕ, x = 1655 :=
by
  sorry

end find_second_number_l396_396345


namespace find_b_eq_3_l396_396466

def Point : Type := ℝ × ℝ

def triangleArea (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

def verticalArea (b : ℝ) : ℝ :=
  let P := (0, 4)
  let Q := (0, 0)
  let R := (6, 0)
  let side1 := (b, 0)
  let side2 := (b, (-2 / 3) * b + 4)
  triangleArea Q side1 side2

def P := (0, 4) : Point
def Q := (0, 0) : Point
def R := (6, 0) : Point

-- Problem statement: find the value of b such that the area divided by the vertical line x = b are equal.
theorem find_b_eq_3 : verticalArea 3 = 6 :=
  sorry

end find_b_eq_3_l396_396466


namespace divide_milk_l396_396080

theorem divide_milk : (3 / 5 : ℚ) = 3 / 5 := by {
    sorry
}

end divide_milk_l396_396080


namespace sin_cos_prod_eq_l396_396587

variable (θ a : ℝ)

-- Assuming the given conditions
axiom condition1 : 0 < θ ∧ θ < π / 2  -- θ is an acute angle
axiom condition2 : sin (2 * θ) = a  -- sin 2θ = a

-- The goal is to prove that sin θ * cos θ = a / 2 under the given conditions
theorem sin_cos_prod_eq : sin θ * cos θ = a / 2 :=
by
  sorry

end sin_cos_prod_eq_l396_396587


namespace books_count_l396_396350

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end books_count_l396_396350


namespace binomial_7_2_l396_396068

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396068


namespace binomial_7_2_l396_396039

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396039


namespace binomial_7_2_l396_396057

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396057


namespace probability_sqrt_two_digit_lt_7_l396_396368

theorem probability_sqrt_two_digit_lt_7 : 
  let two_digit_set := Finset.Icc 10 99
  let favorable_set := Finset.Icc 10 48
  (favorable_set.card : ℚ) / two_digit_set.card = 13 / 30 :=
by sorry

end probability_sqrt_two_digit_lt_7_l396_396368


namespace percentage_of_singles_l396_396199

/-- In a baseball season, Lisa had 50 hits. Among her hits were 2 home runs, 
2 triples, 8 doubles, and 1 quadruple. The rest of her hits were singles. 
What percent of her hits were singles? --/
theorem percentage_of_singles
  (total_hits : ℕ := 50)
  (home_runs : ℕ := 2)
  (triples : ℕ := 2)
  (doubles : ℕ := 8)
  (quadruples : ℕ := 1)
  (non_singles := home_runs + triples + doubles + quadruples)
  (singles := total_hits - non_singles) :
  (singles : ℚ) / (total_hits : ℚ) * 100 = 74 := by
  sorry

end percentage_of_singles_l396_396199


namespace probability_sqrt_lt_7_of_random_two_digit_number_l396_396370

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end probability_sqrt_lt_7_of_random_two_digit_number_l396_396370


namespace solution_l396_396002

def list_price_replica_jersey : ℝ := 80
def list_price_soccer_ball : ℝ := 40
def list_price_soccer_cleats : ℝ := 100

def min_regular_discount_replica_jersey : ℝ := 0.30 * list_price_replica_jersey
def min_regular_discount_soccer_ball : ℝ := 0.40 * list_price_soccer_ball
def min_regular_discount_soccer_cleats : ℝ := 0.20 * list_price_soccer_cleats

def additional_discount_replica_jersey : ℝ := 0.20 * list_price_replica_jersey
def additional_discount_soccer_ball : ℝ := 0.25 * list_price_soccer_ball
def additional_discount_soccer_cleats : ℝ := 0.15 * list_price_soccer_cleats

def lowest_possible_sale_price_replica_jersey : ℝ :=
  let initial_sale_price := list_price_replica_jersey - min_regular_discount_replica_jersey
  initial_sale_price - additional_discount_replica_jersey

def lowest_possible_sale_price_soccer_ball : ℝ :=
  let initial_sale_price := list_price_soccer_ball - min_regular_discount_soccer_ball
  initial_sale_price - additional_discount_soccer_ball

def lowest_possible_sale_price_soccer_cleats : ℝ :=
  let initial_sale_price := list_price_soccer_cleats - min_regular_discount_soccer_cleats
  initial_sale_price - additional_discount_soccer_cleats

def combined_lowest_possible_sale_price : ℝ :=
  lowest_possible_sale_price_replica_jersey + lowest_possible_sale_price_soccer_ball +
  lowest_possible_sale_price_soccer_cleats

def total_list_price : ℝ :=
  list_price_replica_jersey + list_price_soccer_ball + list_price_soccer_cleats

def R_percent : ℝ :=
  (combined_lowest_possible_sale_price / total_list_price) * 100

theorem solution : R_percent ≈ 54.09 := by
  -- The proof is omitted.
  sorry

end solution_l396_396002


namespace sam_dimes_l396_396679

theorem sam_dimes (dimes_original dimes_given : ℕ) :
  dimes_original = 9 → dimes_given = 7 → dimes_original + dimes_given = 16 :=
by
  intros h1 h2
  sorry

end sam_dimes_l396_396679


namespace average_A_B_l396_396705

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end average_A_B_l396_396705


namespace mutually_exclusive_event_count_is_one_l396_396506

-- Define the events
def event_1 (bag : list ℕ) : Prop := ∃ w, w ∈ bag ∧ w = 1 ∨ w = 2  -- At least one white ball
def event_2 (bag : list ℕ) : Prop := ∃ w, w ∈ bag ∧ w = 2           -- All white balls
def event_3 (bag : list ℕ) : Prop := ∃ r w, r ∈ bag ∧ w ∈ bag ∧ r = 1 ∧ w = 1 -- At least one red ball and at least one white ball
def event_4 (bag : list ℕ) : Prop := ∃ w, w ∈ bag ∧ w = 1           -- Exactly one white ball
def event_5 (bag : list ℕ) : Prop := ∃ r, r ∈ bag ∧ r = 2           -- All red balls
def event_6 (bag : list ℕ) : Prop := ∃ r, r ∈ bag ∧ r = 3           -- Exactly two red balls

-- Define the mutually exclusive events count
def mutually_exclusive_event_count : ℕ :=
if (event_1 [0, 1, 1, 1] ∧ ¬event_4 [0, 1, 1, 1]) -- At least one white ball and exactly one white ball
    ∨ (event_2 [0, 1, 1, 1] ∧ ¬event_5 [0, 1, 1, 1]) -- All white balls and all red balls
    ∨ (event_3 [0, 1, 1, 1] ∧ ¬event_6 [0, 1, 1, 1]) -- At least one red ball and exactly two red balls 
    ∨ (event_4 [0, 1, 1, 1] ∧ ¬event_3 [0, 1, 1, 1]) -- Exactly one white ball and exactly two red balls
then 1 else 0

-- Theorem to prove
theorem mutually_exclusive_event_count_is_one : mutually_exclusive_event_count = 1 :=
  by trivial

end mutually_exclusive_event_count_is_one_l396_396506


namespace muffins_originally_baked_l396_396437

theorem muffins_originally_baked :
  let monday := 1
  let tuesday := monday + 1
  let wednesday := tuesday + 1
  let thursday := wednesday + 1
  let friday := thursday + 1
  let total_brought := monday + tuesday + wednesday + thursday + friday
  let leftover := 7 in
  (total_brought + leftover) = 22 := 
by
  let monday := 1
  let tuesday := monday + 1
  let wednesday := tuesday + 1
  let thursday := wednesday + 1
  let friday := thursday + 1
  let total_brought := monday + tuesday + wednesday + thursday + friday
  let leftover := 7
  have : total_brought = 15 := rfl
  have : total_brought + leftover = 22
  show (total_brought + leftover) = 22
  sorry

end muffins_originally_baked_l396_396437


namespace remainder_when_divided_l396_396496

-- Define the polynomial to be divided
def poly := (λ x : ℝ, x^5 + 3*x^2 + 1)

-- Define the divisor
def divisor := (λ x : ℝ, (x - 1)^2)

-- State that the remainder of dividing poly by divisor is 8x - 3
theorem remainder_when_divided (x : ℝ) : ∃ (q : ℝ → ℝ), poly x = divisor x * q x + 8*x - 3 :=
by sorry

end remainder_when_divided_l396_396496


namespace transmission_correctness_l396_396216

namespace Transmission

variable {α β : ℝ}
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Single transmission scheme: Receiving 1, 0, 1 in sequence
def single_transmission_prob : ℝ := (1 - α) * (1 - β)^2

-- Triple transmission scheme: Receiving 1, 0, 1 in sequence when sending 1
def triple_transmission_prob_seq : ℝ := β * (1 - β)^2

-- Triple transmission scheme: Decoding as 1 when sending 1
def triple_transmission_prob_decode_1 : ℝ := 3 * β * (1 - β)^2 + (1 - β)^3

-- Probability comparison for triple vs single transmission when sending 0
def triple_transmission_prob_decode_0 : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3
def single_transmission_prob_0 : ℝ := 1 - α

theorem transmission_correctness :
  (single_transmission_prob == (1 - α) * (1 - β)^2) ∧
  (triple_transmission_prob_seq == β * (1 - β)^2) ∧
  (triple_transmission_prob_decode_1 == 3 * β * (1 - β)^2 + (1 - β)^3) ∧
  (0 < α ∧ α < 0.5 → triple_transmission_prob_decode_0 > single_transmission_prob_0) :=
by
  sorry

end Transmission

end transmission_correctness_l396_396216


namespace sum_bounds_l396_396687

noncomputable def S : ℝ := ∑ i in finset.range(2014), 1 / (i * real.sqrt(i + 1) + (i + 1) * real.sqrt(i))

theorem sum_bounds :
  (43 / 44) < S ∧ S < (44 / 45) :=
begin
  sorry
end

end sum_bounds_l396_396687


namespace scale_model_height_l396_396811

theorem scale_model_height (scale_ratio : ℝ) (actual_height : ℝ) (desired_height : ℝ) : 
  scale_ratio = 25 ∧ actual_height = 1454 ∧ desired_height = 58 →
  Float.round (actual_height / scale_ratio) = desired_height :=
by
  intros h
  have h_scale_ratio : scale_ratio = 25 := h.1
  have h_actual_height : actual_height = 1454 := h.2.1
  have h_desired_height : desired_height = 58 := h.2.2
  sorry

end scale_model_height_l396_396811


namespace sum_of_valid_n_values_l396_396465

open Nat

theorem sum_of_valid_n_values : 
  ∑ k in ({1, 2, 4, 8} : Finset ℕ), (k ^ 2 + 1) = 89 := by
  -- proof omitted
  sorry

end sum_of_valid_n_values_l396_396465


namespace class_mean_l396_396198

def student_scores (students1: ℕ) (mean1: ℕ) (students2: ℕ) (mean2: ℕ) : ℕ :=
  (students1 * mean1 + students2 * mean2) / (students1 + students2)

theorem class_mean {students1 students2 : ℕ} {mean1 mean2 : ℕ}
  (h1 : students1 = 40)
  (h2 : mean1 = 85)
  (h3 : students2 = 10)
  (h4 : mean2 = 80) :
  student_scores students1 mean1 students2 mean2 = 84 :=
by
  simp [student_scores, h1, h2, h3, h4]
  sorry

end class_mean_l396_396198


namespace frank_defeated_enemies_l396_396759

theorem frank_defeated_enemies
  (E : ℕ)
  (h1 : ∀ (E : ℕ), E * 9 + 8 = 62)
  : E = 6 :=
begin
  have h2 : 6 * 9 + 8 = 62, by norm_num,
  exact eq_of_heq (HEq.trans h1 (HEq.symm h2)),
  sorry

end frank_defeated_enemies_l396_396759


namespace sum_square_geq_one_third_l396_396995

variable (a b c : ℝ)

theorem sum_square_geq_one_third (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end sum_square_geq_one_third_l396_396995


namespace point_of_tangency_l396_396953

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

theorem point_of_tangency (a : ℝ) (h1 : ∀ x : ℝ, f x a - f (-x) a = 0) 
  (h2 : ∃ x0 : ℝ, f' x0 a = 3 / 2) : ∃ x0 : ℝ, x0 = Real.log ((3 + Real.sqrt 17) / 4) :=
sorry

end point_of_tangency_l396_396953


namespace score_order_l396_396200

variable (A B C D : ℕ)

-- Condition 1: B + D = A + C
axiom h1 : B + D = A + C
-- Condition 2: A + B > C + D + 10
axiom h2 : A + B > C + D + 10
-- Condition 3: D > B + C + 20
axiom h3 : D > B + C + 20
-- Condition 4: A + B + C + D = 200
axiom h4 : A + B + C + D = 200

-- Question to prove: Order is Donna > Alice > Brian > Cindy
theorem score_order : D > A ∧ A > B ∧ B > C :=
by
  sorry

end score_order_l396_396200


namespace ellipse_equation_no_point_P_l396_396158

noncomputable def C1_ellipse (x y : ℝ) (a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def line_l (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def C2_circle (x y r : ℝ) : Prop := 
  (x - 3)^2 + (y - 3)^2 = r^2

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : e = 1/2) :
  ∃ (C₁ : ℝ → ℝ → Prop), 
    (a = 4) ∧ (b = 2 * √3) ∧ (C₁ = C1_ellipse x y a b) ∧ 
    (C₁ x y = (x^2 / 16) + (y^2 / 12) = 1) := sorry

theorem no_point_P (a b r : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : e = 1/2) 
  (h4 : line_l (-2) 0) (h5 : 2 * √2) (h6 : r > 0) :
  ¬ ∃ (P : ℝ × ℝ), 
    C2_circle (P.fst) (P.snd) r ∧ 
    (∃ (F₁ F₂ : ℝ × ℝ), 
      F₁ = (-2, 0) ∧ F₂ = (2, 0) ∧ 
      |(P.fst + 2)^2 + (P.snd)^2| = (2 * √3 / 3) * |(P.fst - 2)^2 + (P.snd)^2|) := sorry

end ellipse_equation_no_point_P_l396_396158


namespace zero_of_function_l396_396637

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

theorem zero_of_function (x : ℝ) (h : f x = 0) (x1 x2 : ℝ)
  (h1 : -1 < x1 ∧ x1 < x)
  (h2 : x < x2 ∧ x2 < 2) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end zero_of_function_l396_396637


namespace ratio_problem_l396_396946

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396946


namespace orange_juice_savings_l396_396774

theorem orange_juice_savings :
  ∀ (initial_oranges_per_50 : ℝ) (initial_liters : ℝ) (desired_liters : ℝ) (percent_decrease : ℝ),
    initial_oranges_per_50 = 30 →
    initial_liters = 50 →
    desired_liters = 20 →
    percent_decrease = 0.1 →
    let initial_oranges_for_20 := (initial_oranges_per_50 / initial_liters) * desired_liters in
    let oranges_for_first_10 := initial_oranges_for_20 / 2 in
    let oranges_for_next_10 := oranges_for_first_10 * (1 - percent_decrease) in
    let total_oranges_needed := oranges_for_first_10 + oranges_for_next_10 in
    let oranges_saved := initial_oranges_for_20 - total_oranges_needed in
    oranges_saved = 0.6 :=
begin
  sorry,
end

end orange_juice_savings_l396_396774


namespace pipe_tank_fill_time_l396_396278

/-- 
Given:
1. Pipe A fills the tank in 2 hours.
2. The leak empties the tank in 4 hours.
Prove: 
The tank is filled in 4 hours when both Pipe A and the leak are working together.
 -/
theorem pipe_tank_fill_time :
  let A := 1 / 2 -- rate at which Pipe A fills the tank (tank per hour)
  let L := 1 / 4 -- rate at which the leak empties the tank (tank per hour)
  let net_rate := A - L -- net rate of filling the tank
  net_rate > 0 → (1 / net_rate) = 4 := 
by
  intros
  sorry

end pipe_tank_fill_time_l396_396278


namespace triangle_area_of_perimeter_12_l396_396421

noncomputable def integral_triangle (a b c : ℕ) : Prop :=
  a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_area_of_perimeter_12 (a b c : ℕ) (h : integral_triangle a b c) :
  ∃ A : ℝ, A = 6 :=
begin
  sorry
end

end triangle_area_of_perimeter_12_l396_396421


namespace kelvin_hops_minimum_l396_396639

theorem kelvin_hops_minimum : 
  let total_hops := (∑ i in Finset.range 10, i * (10 - i)) + 11
  in total_hops = 176 :=
by
  sorry

end kelvin_hops_minimum_l396_396639


namespace trisecting_angle_not_divide_into_equal_parts_middle_segment_not_always_smallest_l396_396285

theorem trisecting_angle_not_divide_into_equal_parts
  (ABC : Triangle)
  (A B C : Point)
  (A_1 A_2 : Point)
  (hA_2_between_A_1_C : A_1 ≠ A_2 ∧ A_2 ≠ C ∧ collinear A_2 A_1 C)
  (hangle_BAC : Angle B A C = α)
  (hangle_trisection : (Angle B A A_1 = α / 3) ∧ (Angle A_1 A A_2 = α / 3) ∧ (Angle A_2 A C = α / 3)) :
  ¬(segment_eq_within_collinear_points (Segment B A_1) (Segment A_1 A_2) (Segment A_2 C)) :=
sorry

theorem middle_segment_not_always_smallest
  (ABC : Triangle)
  (A B C : Point)
  (A_1 A_2 : Point)
  (hA_2_between_A_1_C : A_1 ≠ A_2 ∧ A_2 ≠ C ∧ collinear A_2 A_1 C)
  (hangle_BAC : Angle B A C = α)
  (hangle_trisection : (Angle B A A_1 = α / 3) ∧ (Angle A_1 A A_2 = α / 3) ∧ (Angle A_2 A C = α / 3)) :
  ¬(∀ x y z : Segment, segment_eq_within_collinear_points (Segment B A_1) y z → middle_is_smallest y) :=
sorry

end trisecting_angle_not_divide_into_equal_parts_middle_segment_not_always_smallest_l396_396285


namespace complement_union_correct_l396_396908

-- Defining the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_correct : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l396_396908


namespace B_determination_num_possible_sets_A_l396_396696

open Set

def f (x : ℤ) : ℤ := |x| + 1

theorem B_determination {A : Set ℤ} (hA : A = {-1, 0, 1}) (hB : ∀ y, (∃ x ∈ A, f x = y) → y ∈ (B : Set ℤ)) :
  B = {1, 2} :=
by
  sorry

theorem num_possible_sets_A (B : Set ℤ) (hB : B = {1, 2}) :
  ∃ S : Finset (Set ℤ), (∀ s ∈ S, ∀ x ∈ s, f x ∈ B) ∧ S.card = 7 :=
by
  sorry

end B_determination_num_possible_sets_A_l396_396696


namespace find_first_day_sale_l396_396406

/-- A grocer wants to find out the sale for the first day based on a given average sales target and known sales figures. -/
theorem find_first_day_sale
  (average_sale : ℤ)
  (sales_day2 sales_day3 sales_day4 sales_day5 sales_day6 : ℤ)
  (total_days : ℤ) :
  average_sale = 625 →
  total_days = 5 →
  sales_day2 = 927 →
  sales_day3 = 855 →
  sales_day4 = 230 →
  sales_day5 = 562 →
  sales_day6 = 741 →
  ∃ sales_day1 : ℤ, sales_day1 = 551 :=
begin
  intros h_average h_days h_day2 h_day3 h_day4 h_day5 h_day6,
  use 551,
  calc
    551 = (625 * 5) - (927 + 855 + 230 + 562 + 741) : sorry
          ... = 551 : sorry
end

end find_first_day_sale_l396_396406


namespace count_subsets_containing_7_l396_396569

def original_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def subsets_containing_7 (s : Set ℕ) : Prop :=
  7 ∈ s

theorem count_subsets_containing_7 :
  {s : Set ℕ | s ⊆ original_set ∧ subsets_containing_7 s}.finite.card = 64 :=
sorry

end count_subsets_containing_7_l396_396569


namespace point_closer_to_F_l396_396218

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)

def side_length (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def point_in_triangle (T : Triangle) (Q : Point) : Prop :=
  sorry -- definition to determine if point Q is within triangle T

def closer_to (T : Triangle) (Q : Point) (P : Point) : Prop :=
  side_length Q P < min (side_length Q T.A) (side_length Q T.B)

def area_probability (T : Triangle) : ℝ :=
  1 / 6

theorem point_closer_to_F
  (D E F : Point)
  (T : Triangle := {A := D, B := E, C := F})
  (h1 : side_length D E = 7)
  (h2 : side_length E F = 6)
  (h3 : side_length F D = 8)
  (Q : Point)
  (hQ : point_in_triangle T Q)
  : closer_to T Q F → area_probability T = 1 / 6 :=
by
  sorry

end point_closer_to_F_l396_396218


namespace smallest_solution_of_equation_l396_396078

theorem smallest_solution_of_equation : 
    ∃ x : ℝ, x*|x| = 3 * x - 2 ∧ 
            ∀ y : ℝ, y*|y| = 3 * y - 2 → x ≤ y :=
sorry

end smallest_solution_of_equation_l396_396078


namespace no_zero_digit_pairs_count_l396_396494

def has_no_zero_digit (n : ℕ) : Prop :=
  ¬ (n.toDigits 10).any (λ d, d = 0)

theorem no_zero_digit_pairs_count : 
  (∃ n : ℕ, (∀ a b : ℕ, a + b = 500 → a > 0 → b > 0 → has_no_zero_digit a → has_no_zero_digit b → n = 410)) :=
by
  sorry

end no_zero_digit_pairs_count_l396_396494


namespace ratio_problem_l396_396941

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396941


namespace problem_l396_396234

theorem problem (S_n : ℕ → ℝ) (n : ℕ)
  (h1 : S_n n = ∑ k in finset.range n, 1 / (real.sqrt (3 * k + 1) + real.sqrt (3 * k + 4)))
  (h2 : S_n n = 5) : 
  n = 85 :=
sorry

end problem_l396_396234


namespace solution_set_inequality_l396_396729

theorem solution_set_inequality (x : ℝ) : 
  5 * x + 1 ≥ 3 * x - 5 ↔ x ≥ -3 :=
begin
  sorry
end

end solution_set_inequality_l396_396729


namespace binomial_coefficient_12_4_l396_396828

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_coefficient_12_4 : binomial_coefficient 12 4 = 495 := by
  sorry

end binomial_coefficient_12_4_l396_396828


namespace savings_correct_l396_396636

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l396_396636


namespace part1_part2_l396_396878

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - x - a * log x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / x - exp (x - 1)
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := (f a x + g b x - |f a x - g b x|) / 2

-- Problem 1: Prove that 0 < a ≤ 3 if f(x) is monotonically increasing on (1, +∞).
theorem part1 (a : ℝ) : (∀ x : ℝ, 1 < x → (4 * x - 1 - a / x) ≥ 0) → (0 < a ∧ a ≤ 3) :=
sorry

-- Problem 2: Prove that there exists a unique positive real number a such that h(x) has exactly two zeros.
theorem part2 : ∃! (a : ℝ), 0 < a ∧ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ h a 1 x1 = 0 ∧ h a 1 x2 = 0 :=
sorry

end part1_part2_l396_396878


namespace ratio_expression_value_l396_396933

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396933


namespace problem_f2014_l396_396867

noncomputable theory
open Real

def f : ℝ → ℝ := sorry

axiom f_4 : f 4 = 2 - sqrt 3
axiom f_recursive : ∀ x : ℝ, f (x + 2) = -1 / f x

theorem problem_f2014 : f 2014 = -(2 + sqrt 3) := sorry

end problem_f2014_l396_396867


namespace find_x_value_l396_396556

theorem find_x_value (x : ℝ) (a b c : ℝ × ℝ × ℝ) 
  (h_a : a = (1, 1, x)) 
  (h_b : b = (1, 2, 1)) 
  (h_c : c = (1, 1, 1)) 
  (h_cond : (c - a) • (2 • b) = -2) : 
  x = 2 := 
by 
  -- the proof goes here
  sorry

end find_x_value_l396_396556


namespace tangency_circumcircle_l396_396643

noncomputable theory

variables {A B C D X Y : Point}
variables (BC : Line) 
variables (ω1 ω2 : Circle)
variables (circumcircle_XDY : Circle)

-- D is an arbitrary point on BC
axiom h1 : D ∈ BC

-- Circles ω1 and ω2 pass through A and D, with specific tangency properties
axiom h2 : A ∈ ω1 ∧ D ∈ ω1
axiom h3 : A ∈ ω2 ∧ D ∈ ω2
axiom h4 : tangent_line (A, B) ω1
axiom h5 : tangent_line (A, C) ω2

-- BX is the second tangent from B to ω1, and CY the second tangent from C to ω2
axiom h6 : second_tangent (B, X) ω1
axiom h7 : second_tangent (C, Y) ω2

-- circumcircle of triangle XDY
axiom h8 : circumcircle circumcircle_XDY = Circle (triangle_circum_center (X, D, Y)) (triangle_circum_radius (X, D, Y))

theorem tangency_circumcircle : tangent circumcircle_XDY BC := 
sorry

end tangency_circumcircle_l396_396643


namespace floor_sqrt_99_l396_396483

theorem floor_sqrt_99 : ⌊real.sqrt 99⌋ = 9 := 
by {
  -- Definitions related to the conditions
  let n := 9,
  let m := n + 1,
  have h1 : n ^ 2 = 81 := by norm_num,
  have h2 : m ^ 2 = 100 := by norm_num,

  -- Conditions from the problem
  have h3 : n ^ 2 < 99 := by norm_num,
  have h4 : 99 < m ^ 2 := by norm_num,

  -- Conclude the answer
  have h5 : (n : ℝ) < real.sqrt 99 := by sorry,
  have h6 : real.sqrt 99 < (m : ℝ) := by sorry,
  
  -- Final conclusion using the floor property
  exact int.floor_of_nonneg_of_le_ceil _ h5 h6,
}

end floor_sqrt_99_l396_396483


namespace cages_used_l396_396795

theorem cages_used (total_puppies sold_puppies puppies_per_cage remaining_puppies needed_cages additional_cage total_cages: ℕ) 
  (h1 : total_puppies = 36) 
  (h2 : sold_puppies = 7) 
  (h3 : puppies_per_cage = 4) 
  (h4 : remaining_puppies = total_puppies - sold_puppies) 
  (h5 : needed_cages = remaining_puppies / puppies_per_cage) 
  (h6 : additional_cage = if (remaining_puppies % puppies_per_cage = 0) then 0 else 1) 
  (h7 : total_cages = needed_cages + additional_cage) : 
  total_cages = 8 := 
by 
  sorry

end cages_used_l396_396795


namespace windmill_pivot_infinitely_often_l396_396254

def is_finite_set (S : Set Point) : Prop := finite S ∧ 2 ≤ S.card
def no_three_collinear (S : Set Point) : Prop := ∀ P Q R ∈ S, ¬collinear {P, Q, R}
def windmill (P : Point) (S : Set Point) (line : Line) : Prop := 
    ∀ Q ∈ S, Q ≠ P → ∃ next_line, ∃ next_point ∈ S, next_point ≠ Q ∧ next_point = P

theorem windmill_pivot_infinitely_often (S : Set Point) (h_finite : is_finite_set S)
    (h_no_collinear : no_three_collinear S) (P : Point) (hP : P ∈ S) :
    ∃ line, windmill P S line := 
sorry

end windmill_pivot_infinitely_often_l396_396254


namespace ratio_volumes_l396_396410

-- Conditions
variables {a h : ℝ} (h_pos : 0 < h) (a_pos : 0 < a)

-- Define the volumes
def V_cone : ℝ := (1/3) * π * (a^2) * h
def V_prism : ℝ := 6 * (a^2) * h

-- The theorem to prove
theorem ratio_volumes : V_cone / V_prism = π / 18 :=
by sorry

end ratio_volumes_l396_396410


namespace triangle_inequality_cosine_l396_396281

theorem triangle_inequality_cosine {a b c A B C p : ℝ}
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (ha : 0 < A ∧ A < real.pi)
  (hb : 0 < B ∧ B < real.pi)
  (hc : 0 < C ∧ C < real.pi)
  (ha1 : a = b * real.cos C + c * real.cos B)
  (hb1 : b = a * real.cos C + c * real.cos A)
  (hc1 : c = a * real.cos B + b * real.cos A)
  (hp : p = (a + b + c) / 2) :
  a * real.cos A + b * real.cos B + c * real.cos C ≤ p :=
sorry

end triangle_inequality_cosine_l396_396281


namespace determine_m_l396_396546

noncomputable def f (m x : ℝ) : ℝ := m * x^3 + 3 * (m - 1) * x^2 - m^2 + 1

theorem determine_m (m : ℝ) (hm : m > 0) 
  (hdecreasing : ∀ x ∈ Ioo (0 : ℝ) (4 : ℝ), deriv (f m) x < 0) : m = 1 / 3 :=
begin
  sorry
end

end determine_m_l396_396546


namespace tangent_lines_to_circle_passing_through_M_maximum_area_of_triangle_ABC_l396_396137

-- Defining the conditions
def point_M := (-6, -5)
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 6 * y - 3 = 0
def center := (-2, 3)
def radius := 4
def point_N := (1, 3)

-- Lean statement for Question 1
theorem tangent_lines_to_circle_passing_through_M :
  ∃ (k : ℝ), (k = 3 / 4 ∧ (λ x y, y + 5 = k * (x + 6)) ∨ k = 1 / 0 ∧ (λ x y, x = -6))
:= sorry

-- Lean statement for Question 2
theorem maximum_area_of_triangle_ABC :
  ∃ (k : ℝ), ((k = 2 * sqrt 2 ∨ k = -2 * sqrt 2) ∧ ∀ A B C, 
  line_AB (N : ℝ × ℝ) (k : ℝ → ℝ) (A B : ℝ × ℝ),
  intersect_circle circle_C A B →
  triangle_area A B C = 8)
:= sorry

end tangent_lines_to_circle_passing_through_M_maximum_area_of_triangle_ABC_l396_396137


namespace binomial_7_2_eq_21_l396_396046

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396046


namespace find_k_l396_396233

-- Definitions corresponding to the conditions

def parabola_f (x : ℝ) : ℝ := 4 * x
def inverse_proportion (k x : ℝ) : ℝ := k / x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def intersects (P : ℝ × ℝ) (k : ℝ) (C : ℝ → ℝ) : Prop :=
  P.2 = C P.1 ∧ P.2 = inverse_proportion k P.1

-- Main statement
theorem find_k (k : ℝ) (P : ℝ × ℝ) : (0 < k) →
  focus (1, 0) →
  intersects P k (λ x, √(4 * x)) →
  P.1 = 1 →
  P.2 = 2 →
  k = 2 :=
by
  sorry

end find_k_l396_396233


namespace grandparents_cans_ratio_l396_396026

-- Definitions for the given conditions
def earnings_per_can : ℝ := 0.25
def cans_home : ℕ := 12
def cans_neighbor : ℕ := 46
def cans_dad_office : ℕ := 250
def savings : ℝ := 43

-- To prove:
theorem grandparents_cans_ratio :
  let total_earnings := 2 * savings,
      total_cans := total_earnings / earnings_per_can,
      known_cans := cans_home + cans_neighbor + cans_dad_office,
      grandparents_cans := total_cans - known_cans,
      ratio := grandparents_cans / cans_home
  in ratio = 3 := by
  sorry

end grandparents_cans_ratio_l396_396026


namespace integer_solutions_l396_396903

def P (n : ℤ) : ℤ := n^3 - n^2 - 5 * n + 2

theorem integer_solutions (n : ℤ) : P(n)^2 = (p : ℤ)^2 ∧ Prime p ↔ n = -3 ∨ n = -1 :=
by
  sorry

end integer_solutions_l396_396903


namespace passengers_at_third_station_l396_396802

theorem passengers_at_third_station 
  (initial_passengers : ℕ) 
  (drop_fraction_first : ℚ) 
  (pickup_first : ℕ) 
  (drop_fraction_second : ℚ)
  (pickup_second: ℕ)
  (final_passengers: ℕ):
  initial_passengers = 288 →
  drop_fraction_first = 1/3 →
  pickup_first = 280 →
  drop_fraction_second = 1/2 →
  pickup_second = 12 →
  final_passengers = 248 →
  let passengers_after_first_station := (initial_passengers * (1 - drop_fraction_first).toInt) + pickup_first in
  let passengers_after_second_station := (passengers_after_first_station * (1 - drop_fraction_second).toInt) + pickup_second in
  passengers_after_second_station = final_passengers :=
by
  intros initial_passengers_eq drop_fraction_first_eq pickup_first_eq drop_fraction_second_eq pickup_second_eq final_passengers_eq
  simp [initial_passengers_eq, drop_fraction_first_eq, pickup_first_eq, drop_fraction_second_eq, pickup_second_eq, final_passengers_eq]
  sorry

end passengers_at_third_station_l396_396802


namespace evaluate_expression_l396_396840

theorem evaluate_expression :
  -2 ^ 2005 + (-2) ^ 2006 + 2 ^ 2007 - 2 ^ 2008 = 2 ^ 2005 :=
by
  -- The following proof is left as an exercise.
  sorry

end evaluate_expression_l396_396840


namespace value_of_expression_l396_396929

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396929


namespace wooden_box_length_l396_396805

theorem wooden_box_length :
  let box_volume := 8 * 7 * 6 * (1 : ℝ)   -- cm^3
  let total_boxes := 1000000
  let total_volume_cm3 := total_boxes * box_volume
  let total_volume_m3 := total_volume_cm3 / (100^3)
  let box_length := real.cbrt total_volume_m3
  box_length ≈ 6.93 :=
by
  sorry

end wooden_box_length_l396_396805


namespace m_factorial_n_value_l396_396193

theorem m_factorial_n_value (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h : (m + n)! / n! = 5040) : m! * n = 144 := 
sorry

end m_factorial_n_value_l396_396193


namespace sum_of_naturals_eq_91_l396_396194

theorem sum_of_naturals_eq_91 (n : ℕ) (h : ∑ i in Finset.range (n + 1), i = 91) : n = 13 := 
by sorry

end sum_of_naturals_eq_91_l396_396194


namespace probability_satisfies_condition_l396_396296

noncomputable def probability_geometric_area (S : Set (ℝ × ℝ)) (A : ℝ) : ℝ :=
  (Set.volume S) / A

theorem probability_satisfies_condition :
  let unit_square := Set.Icc 0 1 ×ˢ Set.Icc 0 1
  let region := { p : ℝ × ℝ | p.1 ∈ Set.Icc 0 1 ∧ p.2 ∈ Set.Icc 0 1 ∧ p.2 ≥ sqrt (1 - p.1 ^ 2) }
  probability_geometric_area region 1 = 1 - (π / 4) :=
by
  -- This is where the proof would go
  sorry

end probability_satisfies_condition_l396_396296


namespace find_a7_l396_396615

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions: a is an arithmetic sequence and the sum of the first 13 terms is 52.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of an arithmetic sequence.
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * ((a 0) + (a (n - 1))) / 2

axiom sum_condition : sum_of_first_n_terms a 13 = 52

theorem find_a7 (h : is_arithmetic_sequence a) : a 6 = 4 :=
by sorry

end find_a7_l396_396615


namespace ratio_problem_l396_396936

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396936


namespace rahul_savings_after_expenditures_l396_396113

theorem rahul_savings_after_expenditures (salary : ℕ) (house_rent_percent : ℝ) (education_percent : ℝ) (clothes_percent : ℝ)
  (h_salary : salary = 2125)
  (h_house_rent : house_rent_percent = 0.20)
  (h_education : education_percent = 0.10)
  (h_clothes : clothes_percent = 0.10) :
  let house_rent := house_rent_percent * salary in
  let salary_after_house_rent := salary - house_rent in
  let education_cost := education_percent * salary_after_house_rent in
  let salary_after_education := salary_after_house_rent - education_cost in
  let clothes_cost := clothes_percent * salary_after_education in
  let final_salary := salary_after_education - clothes_cost in
  final_salary = 1377 :=
by
  sorry

end rahul_savings_after_expenditures_l396_396113


namespace hypotenuse_of_right_triangle_l396_396367

theorem hypotenuse_of_right_triangle (a b : ℕ) (ha : a = 140) (hb : b = 336) :
  Nat.sqrt (a * a + b * b) = 364 := by
  sorry

end hypotenuse_of_right_triangle_l396_396367


namespace part_1_part_2_l396_396504

-- Given definitions and conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom a1 : a 1 = 1
axiom S_cond : ∀ n ≥ 2, (S n)^2 = a n * (S n - 1/2)

noncomputable def is_arithmetic_seq (f : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, f (n + 1) = f n + d

-- Prove (1): {1/S_n} is an arithmetic sequence and find the expression for S_n
theorem part_1 :
  is_arithmetic_seq (λ n, 1 / S n) ∧ S = λ n, 1 / (2 * n - 1) := sorry

-- Define b_n and Prove (2): T_n = (2n-3) * 2^(n+1) + 6
def b_n (n : ℕ) : ℝ := 2^n / S n

def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n i

theorem part_2 : T_n = λ n, (2 * n - 3) * 2^(n+1) + 6 := sorry

end part_1_part_2_l396_396504


namespace expected_positions_after_transpositions_l396_396302

theorem expected_positions_after_transpositions :
  let balls : Fin 6 → ℕ := λ _, 1 in
  let swap_adjacent (balls : Fin 6 → ℕ) (i : Fin 6) : Fin 6 → ℕ :=
    λ j, if j = i ∨ j = (i + 1) % 6 then balls ((j + 1) % 6) else balls j in
  ∑ i : Fin 6, pairwise_distinct (swap_adjacent (swap_adjacent balls i) i) = 3 :=
sorry

end expected_positions_after_transpositions_l396_396302


namespace count_valid_sequences_l396_396645

-- Define the vertices of the pentagon
def P_vertices : List (Int × Int) := [(0,0), (5,0), (0,4), (3,3), (-1,3)]

-- Define the transformations as functions
def rotate_90 (p : Int × Int) : Int × Int := (-p.2, p.1)
def rotate_180 (p : Int × Int) : Int × Int := (-p.1, -p.2)
def rotate_270 (p : Int × Int) : Int × Int := (p.2, -p.1)
def translate (v : Int × Int) (p : Int × Int) : Int × Int := (p.1 + v.1, p.2 + v.2)

-- Define the transformation sequences and their effect on the pentagon
def transformations (t : Int) : List (Int × Int) → List (Int × Int)
| []           => []
| (p :: pts)   => match t with
  | 0 => rotate_90 p :: transformations 0 pts
  | 1 => rotate_180 p :: transformations 1 pts
  | 2 => rotate_270 p :: transformations 2 pts
  | 3 => translate (5, 0) p :: transformations 3 pts
  | 4 => translate (0, 4) p :: transformations 4 pts
  | _ => p :: transformations t pts

-- Prove that there are 9 valid sequences of three transformations
theorem count_valid_sequences : 
  let valid_sequences := (List.range 5).product ((List.range 5).product (List.range 5)).filter (λ seq,
    transformations seq.fst.fst (transformations seq.fst.snd (transformations seq.snd P_vertices)) = P_vertices)
  in valid_sequences.length = 9 := by sorry

end count_valid_sequences_l396_396645


namespace num_ways_to_assign_guests_is_17640_l396_396442

noncomputable def num_ways_to_assign_guests : ℕ :=
  let case1 := Nat.factorial 7 in
  let case2 := Nat.choose 7 5 * Nat.factorial 5 * Nat.choose 2 1 * Nat.factorial 2 in
  let case3 := Nat.choose 7 4 * Nat.factorial 4 * Nat.choose 3 1 * Nat.choose 3 3 in
  case1 + case2 + case3

theorem num_ways_to_assign_guests_is_17640 :
  num_ways_to_assign_guests = 17640 :=
by
  unfold num_ways_to_assign_guests
  sorry

end num_ways_to_assign_guests_is_17640_l396_396442


namespace find_third_number_l396_396185

-- Define the conditions
def equation1_valid : Prop := (5 * 3 = 15) ∧ (5 * 2 = 10) ∧ (2 * 1000 + 3 * 100 + 5 = 1022)
def equation2_valid : Prop := (9 * 2 = 18) ∧ (9 * 4 = 36) ∧ (4 * 1000 + 2 * 100 + 9 = 3652)

-- The theorem to prove
theorem find_third_number (h1 : equation1_valid) (h2 : equation2_valid) : (7 * 2 = 14) ∧ (7 * 5 = 35) ∧ (5 * 1000 + 2 * 100 + 7 = 547) :=
by 
  sorry

end find_third_number_l396_396185


namespace part1_part2_part3_l396_396516

def pointM (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Part 1
theorem part1 (m : ℝ) (h : 2 * m + 3 = 0) : pointM m = (-5 / 2, 0) :=
  sorry

-- Part 2
theorem part2 (m : ℝ) (h : 2 * m + 3 = -1) : pointM m = (-3, -1) :=
  sorry

-- Part 3
theorem part3 (m : ℝ) (h1 : |m - 1| = 2) : pointM m = (2, 9) ∨ pointM m = (-2, 1) :=
  sorry

end part1_part2_part3_l396_396516


namespace sum_of_prime_factors_of_1386_l396_396754

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : list ℕ := 
  if h : n = 0 then [] else 
  list.filter (λ p, is_prime p ∧ p ∣ n) (list.range (n + 1))

def smallest_prime_factor (n : ℕ) : ℕ :=
  (prime_factors n).head!

def largest_prime_factor (n : ℕ) : ℕ :=
  (prime_factors n).reverse.head!

theorem sum_of_prime_factors_of_1386 : 
  smallest_prime_factor 1386 + largest_prime_factor 1386 = 13 :=
by
  -- proof goes here
  sorry

end sum_of_prime_factors_of_1386_l396_396754


namespace problem_statement_l396_396328

-- Define the two circles in terms of their equations
def O₁ : Set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 2 * p.1 = 0 }
def O₂ : Set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 + 2 * p.1 - 4 * p.2 = 0 }

-- Define the intersection points A and B
def intersect_points (O₁ O₂ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := O₁ ∩ O₂

-- Definitions for points A, B, and conditions on these points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ
axiom A_in_intersect : A ∈ intersect_points O₁ O₂
axiom B_in_intersect : B ∈ intersect_points O₁ O₂

-- Prove the statements given the conditions
theorem problem_statement :
  (∀ P : ℝ × ℝ, P ∈ O₁ → (∀ x y : ℝ, x = y → ∃ l : Polynomial ℝ, l.eval (x, y) = l.eval P)) ∧ 
  (A.x + B.x) / 2 = 0 ∧
  (A.y + B.y) / 2 = 1 ∧
  ∃ (AB_length : ℝ), AB_length = sqrt 2 / 2 ∧
  ∃ (max_dist : ℝ), max_dist = (sqrt 2 / 2) + 1 := 
sorry

end problem_statement_l396_396328


namespace cost_of_bricks_l396_396354

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end cost_of_bricks_l396_396354


namespace not_excellent_195_max_M_for_GM_is_827_l396_396950

def isExcellentNumber (M : ℕ) : Prop :=
  ∃ A B : ℕ, 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧
    (A % 10 + B % 10 = 8) ∧ (A / 10 = B / 10) ∧ (M = A * B)

def P (M : ℕ) (A B : ℕ) : ℕ := A % 10 + B % 10
def Q (M : ℕ) (A B : ℕ) : ℕ := abs (A - B)
def G (M : ℕ) (A B : ℕ) : ℕ := (P M A B) / (Q M A B)

theorem not_excellent_195 : ¬ isExcellentNumber 195 :=
by { sorry }

theorem max_M_for_GM_is_827 : ∃ M A B : ℕ, isExcellentNumber M ∧ G M A B % 8 = 0 ∧ M = 882727 :=
by { sorry }

end not_excellent_195_max_M_for_GM_is_827_l396_396950


namespace intersection_area_is_5_over_7_l396_396133

-- Given initial square with side length 1
def M0 : ℝ := 1

-- Given intersection area we need to prove
def intersection_area := (5:ℝ) / 7

-- Lean statement: proving the area of the intersection of all polygons equals 5/7
theorem intersection_area_is_5_over_7 :
  -- Condition: Sequence of polygons starting with square of area 1
  ∀ (M : ℕ → ℝ), 
  -- M 0 refers to initial square with area 1
  M 0 = M0 → 
  -- Next iteration involves removing triangles from the corners as described
  ∀ k, 
  -- Area formula adapting based on the process iteratively
  M (k + 1) = M k - 4 * (1 / 2 * (M k) ^ 2 / 9) / 9 →
  -- Prove the final intersection area is 5/7
  (∀ n, M n) = intersection_area :=
sorry

end intersection_area_is_5_over_7_l396_396133


namespace max_value_a4_b2_c2_d2_l396_396238

theorem max_value_a4_b2_c2_d2
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  a^4 + b^2 + c^2 + d^2 ≤ 100 :=
sorry

end max_value_a4_b2_c2_d2_l396_396238


namespace car_speed_second_hour_l396_396730

theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (avg_speed : ℝ)
  (hours : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_first_hour : ℝ)
  (distance_second_hour : ℝ) :
  speed_first_hour = 90 →
  avg_speed = 75 →
  hours = 2 →
  total_time = hours →
  total_distance = avg_speed * total_time →
  distance_first_hour = speed_first_hour * 1 →
  distance_second_hour = total_distance - distance_first_hour →
  distance_second_hour / 1 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end car_speed_second_hour_l396_396730


namespace height_tank_C_l396_396312

noncomputable def radius (C: ℂ) (π : ℂ) := 
    C / (2 * π)

theorem height_tank_C :
  ∀ (C_C C_B : ℕ) (h_B : ℕ) (π : ℝ),
  C_C = 8 →
  C_B = 10 →
  h_B = 8 →
  (radius C_C π) ^ 2 * h_C = 0.8 * (radius C_B π) ^ 2 * h_B →
  h_C = 10 :=
begin
  intros C_C C_B h_B π hCirC hCirB htankB hCapacity,
  sorry
end

end height_tank_C_l396_396312


namespace binomial_7_2_l396_396061

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396061


namespace binomial_7_2_l396_396034

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396034


namespace sister_height_on_birthday_l396_396273

theorem sister_height_on_birthday (previous_height : ℝ) (growth_rate : ℝ)
    (h_previous_height : previous_height = 139.65)
    (h_growth_rate : growth_rate = 0.05) :
    previous_height * (1 + growth_rate) = 146.6325 :=
by
  -- Proof omitted
  sorry

end sister_height_on_birthday_l396_396273


namespace find_three_numbers_l396_396321

theorem find_three_numbers (x y z : ℝ) 
  (h1 : x - y = 12) 
  (h2 : (x + y) / 4 = 7) 
  (h3 : z = 2 * y) 
  (h4 : x + z = 24) : 
  x = 20 ∧ y = 8 ∧ z = 16 := by
  sorry

end find_three_numbers_l396_396321


namespace find_m_plus_n_l396_396994

theorem find_m_plus_n (AB AC BC : ℕ) (RS : ℚ) (m n : ℕ) 
  (hmn_rel_prime : Nat.gcd m n = 1)
  (hAB : AB = 1995)
  (hAC : AC = 1994)
  (hBC : BC = 1993)
  (hRS : RS = m / n) :
  m + n = 997 :=
sorry

end find_m_plus_n_l396_396994


namespace function_domain_l396_396075

open Set

noncomputable def domain_of_function : Set ℝ :=
  {x | x ≠ 2}

theorem function_domain :
  domain_of_function = {x : ℝ | x ≠ 2} :=
by sorry

end function_domain_l396_396075


namespace max_squares_covered_l396_396786

theorem max_squares_covered :
  ∀ (n : ℕ), 
    ∀ (checkerboard : ℕ → ℕ → Prop), 
      (∀ i j : ℕ, checkerboard i j → (1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n)) →
        ∀ (card : ℝ → ℝ → ℝ → ℝ → Prop), 
          (∀ x y : ℝ, card x y 2 45 → checkerboard (floor x) (floor y)) →
            ∃ n, n = 16 := sorry

end max_squares_covered_l396_396786


namespace probability_sqrt_lt_7_of_random_two_digit_number_l396_396371

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end probability_sqrt_lt_7_of_random_two_digit_number_l396_396371


namespace max_angle_BAC_l396_396471

-- Define the basic geometric setup
variables (A B C A' B' D : Type) [triangle A B C] -- Assume A', B', and D are points
variables (m_a s_b : ℝ) -- lengths of altitude and median

-- Define the geometric properties of the triangle
axiom altitude_from_A (hA : altitude_from A) : length hA = m_a
axiom median_from_B (hB : median_from B) : length hB = s_b

-- Statement of the problem: Prove the maximum value of angle ∠BAC
theorem max_angle_BAC :
  ∃ (A₁ A₂ : Point), (angle BAC) = max (angle B A₁ C) (angle B A₂ C) :=
sorry

end max_angle_BAC_l396_396471


namespace cost_per_notebook_before_discount_l396_396473

-- Definitions and conditions
def total_spent : ℝ := 56
def original_prices : list (ℝ × ℕ) := [(30, 1), (2, 3), (1.5, 2), (15, 1)]
def discount_rate : ℝ := 0.10
def total_notebooks : ℕ := 5

-- The property stating the correct answer equivalence
theorem cost_per_notebook_before_discount :
  (let 
    original_total := (original_prices.map (λ p, p.1 * p.2.to_real)).sum,
    discount := original_total * discount_rate,
    discounted_total := original_total - discount,
    cost_notebooks := total_spent - discounted_total,
    cost_per_notebook := cost_notebooks / total_notebooks.to_real
  in cost_per_notebook = 1.48) := sorry

end cost_per_notebook_before_discount_l396_396473


namespace probability_of_sum_l396_396505

def is_fair_ten_sided_die (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def number_of_outcomes : ℕ := 10^4

def favorable_outcomes : ℕ := 180  -- from the detailed solution steps indicating total favorable sums

theorem probability_of_sum (d1 d2 d3 d4 : ℕ) 
    (h1 : is_fair_ten_sided_die d1) 
    (h2 : is_fair_ten_sided_die d2) 
    (h3 : is_fair_ten_sided_die d3) 
    (h4 : is_fair_ten_sided_die d4) : 
    ∃ (p : ℚ), p = (favorable_outcomes : ℚ) / (number_of_outcomes : ℚ) ∧ p = 9 / 500 := 
begin
  sorry
end

end probability_of_sum_l396_396505


namespace complex_quadrant_l396_396208

-- Definitions based on the conditions
def i : ℂ := complex.I

-- Statement
theorem complex_quadrant :
  let Z := i * (1 + i) in
  Z.re < 0 ∧ Z.im > 0 :=
by
  sorry

end complex_quadrant_l396_396208


namespace map_scale_calculation_l396_396815

theorem map_scale_calculation :
  let distance_gs_nw : ℝ := 55 * 1.5,
      distance_nw_rd : ℝ := 50 * 2,
      distance_rd_mad : ℝ := 60 * 3,
      total_distance : ℝ := distance_gs_nw + distance_nw_rd + distance_rd_mad,
      distance_on_map : ℝ := 5
  in total_distance = 362.5 ∧ distance_on_map / total_distance = 0.0138 :=
by
  let distance_gs_nw : ℝ := 55 * 1.5,
      distance_nw_rd : ℝ := 50 * 2,
      distance_rd_mad : ℝ := 60 * 3,
      total_distance : ℝ := distance_gs_nw + distance_nw_rd + distance_rd_mad,
      distance_on_map : ℝ := 5
  sorry

end map_scale_calculation_l396_396815


namespace expected_value_max_l396_396863

def E_max_x_y_z (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) : ℚ :=
  (4 * (1/6) + 5 * (1/3) + 6 * (1/4) + 7 * (1/6) + 8 * (1/12))

theorem expected_value_max (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) :
  E_max_x_y_z x y z h1 h2 h3 h4 = 17 / 3 := 
sorry

end expected_value_max_l396_396863


namespace sum_of_reciprocals_AF_BF_l396_396971

-- Define the Cartesian plane with the curve C and the line l
def curve_C (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def line_l (k x : ℝ) : ℝ := k * (x - 1)

-- Define the conditions in terms of the intersection points and the focus
variable (k x1 x2 : ℝ)
axiom focus_F : (1, 0)
axiom intersection_A : curve_C (x1) = (x1, line_l k x1)
axiom intersection_B : curve_C (x2) = (x2, line_l k x2)
axiom quadratic_relation : k^2 * x1^2 - (4 + 2 * k^2) * x1 + k^2 = 0 ∧ k^2 * x2^2 - (4 + 2 * k^2) * x2 + k^2 = 0

-- Define the goal in terms of the proof statement
theorem sum_of_reciprocals_AF_BF : (1 / (x1 + 1) + 1 / (x2 + 1)) = 1 := 
by
  sorry

end sum_of_reciprocals_AF_BF_l396_396971


namespace range_of_a_l396_396654

variable (a : ℝ)
def A := Set.Ico (-2 : ℝ) 4
def B := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (h : B a ⊆ A) : 0 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l396_396654


namespace money_first_day_l396_396478

-- Define the total mushrooms
def total_mushrooms : ℕ := 65

-- Define the mushrooms picked on the second day
def mushrooms_day2 : ℕ := 12

-- Define the mushrooms picked on the third day
def mushrooms_day3 : ℕ := 2 * mushrooms_day2

-- Define the price per mushroom
def price_per_mushroom : ℕ := 2

-- Prove that the amount of money made on the first day is $58
theorem money_first_day : (total_mushrooms - mushrooms_day2 - mushrooms_day3) * price_per_mushroom = 58 := 
by
  -- Skip the proof
  sorry

end money_first_day_l396_396478


namespace range_of_mn_l396_396159

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_mn (m n : ℝ)
  (h₁ : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4)
  (h₂ : ∀ z, -5 ≤ z ∧ z ≤ 4 → ∃ x, f x = z ∧ m ≤ x ∧ x ≤ n) :
  1 ≤ m + n ∧ m + n ≤ 7 :=
by
  sorry

end range_of_mn_l396_396159


namespace three_digit_cubes_divisible_by_16_l396_396582

theorem three_digit_cubes_divisible_by_16 :
  (count (λ n : ℕ, 4 * n = n ∧ (100 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 999)) {n | 1 ≤ n ∧ n ≤ 2}) = 1 :=
sorry

end three_digit_cubes_divisible_by_16_l396_396582


namespace sqrt_sub_one_gt_one_l396_396826

theorem sqrt_sub_one_gt_one (h : 2 < real.sqrt 5 ∧ real.sqrt 5 < 3) : real.sqrt 5 - 1 > 1 := sorry

end sqrt_sub_one_gt_one_l396_396826


namespace triangle_problem_l396_396360

/-- Given triangle ABC with the following properties and conditions:
- AC = 600
- BC = 400
- Points K and L are located on AC and AB respectively such that AK = CK, and CL is the angle bisector of angle C
- P is the intersection point of BK and CL
- M is a point on BK such that K is the midpoint of PM
- AM = 240
Prove that LP equals 96. -/
theorem triangle_problem (A B C K L P M : Type) [Point A] [Point B] [Point C] [Point K] [Point L] [Point P] [Point M]
  (h1 : AC = 600)
  (h2 : BC = 400)
  (h3 : K ∈ segment A C)
  (h4 : L ∈ segment A B)
  (h5 : AK = CK)
  (h6 : CL is angle_bisector_of ∠C)
  (h7 : BK ∩ CL = P)
  (h8 : K is midpoint_of segment PM)
  (h9 : AM = 240) :
  LP = 96 := 
sorry

end triangle_problem_l396_396360


namespace amy_baked_l396_396438

-- Definition of total muffins brought from Monday to Friday
def muffins_brought_to_school (monday : ℕ) : ℕ :=
  let tuesday := monday + 1 in
  let wednesday := tuesday + 1 in
  let thursday := wednesday + 1 in
  let friday := thursday + 1 in
  monday + tuesday + wednesday + thursday + friday

-- Definition of muffins left on Saturday
def muffins_left := 7

-- Theorem stating total muffins Amy baked
theorem amy_baked (original_muffins : ℕ) : original_muffins = muffins_brought_to_school 1 + muffins_left := by
  sorry

end amy_baked_l396_396438


namespace compare_expressions_l396_396464

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l396_396464


namespace bill_new_win_percentage_l396_396018

theorem bill_new_win_percentage :
  ∀ (initial_games : ℕ) (initial_win_percentage : ℚ) (additional_games : ℕ) (losses_in_additional_games : ℕ),
  initial_games = 200 →
  initial_win_percentage = 0.63 →
  additional_games = 100 →
  losses_in_additional_games = 43 →
  ((initial_win_percentage * initial_games + (additional_games - losses_in_additional_games)) / (initial_games + additional_games)) * 100 = 61 := 
by
  intros initial_games initial_win_percentage additional_games losses_in_additional_games h1 h2 h3 h4
  sorry

end bill_new_win_percentage_l396_396018


namespace sum_of_coefficients_eq_minus_36_l396_396717

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem sum_of_coefficients_eq_minus_36 
  (a b c : ℝ)
  (h_min : ∀ x, quadratic a b c x ≥ -36)
  (h_points : quadratic a b c (-3) = 0 ∧ quadratic a b c 5 = 0)
  : a + b + c = -36 :=
sorry

end sum_of_coefficients_eq_minus_36_l396_396717


namespace part1_part2_l396_396524

def set_A := {x : ℝ | x^2 + 2*x - 8 = 0}
def set_B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem part1 (a : ℝ) (h : a = 1) : 
  (set_A ∩ set_B a) = {-4} := by
  sorry

theorem part2 (a : ℝ) : 
  (set_A ∩ (set_B a) = set_B a) → (a < -1 ∨ a > 3) := by
  sorry

end part1_part2_l396_396524


namespace num_nurses_l396_396344

theorem num_nurses (total : ℕ) (ratio_doctors : ℕ) (ratio_nurses : ℕ) 
  (h1 : total = 200) (h2 : ratio_doctors = 4) (h3 : ratio_nurses = 6) : 
  let nurses := total * ratio_nurses / (ratio_doctors + ratio_nurses) in
  nurses = 120 :=
by
  -- definitions and assumptions
  let nurses := total * ratio_nurses / (ratio_doctors + ratio_nurses)
  have h4 : nurses = 200 * 6 / (4 + 6) := by rw [h1, h2, h3]
  have h5 : nurses = 120 := by norm_num at h4
  exact h5

end num_nurses_l396_396344


namespace original_price_per_brick_l396_396358

-- Definitions translated from the given problem conditions
variable (num_bricks : Nat) (discounted_bricks : Nat) (full_price_bricks : Nat) (total_cost : ℝ)
variable (P : ℝ) -- the original price per brick
variable (discount_rate : ℝ := 0.5) -- 50% discount rate

-- Conditions from the problem
def conditions : Prop :=
  num_bricks = 1000 ∧
  discounted_bricks = 500 ∧
  full_price_bricks = 500 ∧
  total_cost = 375 ∧
  total_cost = (discounted_bricks * (discount_rate * P)) + (full_price_bricks * P)

-- The main statement we need to prove
theorem original_price_per_brick (h : conditions num_bricks discounted_bricks full_price_bricks total_cost P discount_rate) : P = 0.50 :=
by {
  sorry,
}


end original_price_per_brick_l396_396358


namespace largest_divisor_of_product_l396_396751

theorem largest_divisor_of_product (n : ℕ) (h : n % 3 = 0) : ∃ d, d = 288 ∧ ∀ n (h : n % 3 = 0), d ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) := 
sorry

end largest_divisor_of_product_l396_396751


namespace max_even_integers_l396_396806

theorem max_even_integers (x : Fin 7 → ℕ) (h_pos : ∀ i, x i > 0) (h_prod_odd : (∏ i, x i) % 2 = 1) : 
  (∃ i, x i % 2 = 0) → False :=
by sorry

end max_even_integers_l396_396806


namespace num_occupied_third_floor_rooms_l396_396228

-- Definitions based on conditions
def first_floor_rent : Int := 15
def second_floor_rent : Int := 20
def third_floor_rent : Int := 2 * first_floor_rent
def rooms_per_floor : Int := 3
def monthly_earnings : Int := 165

-- The proof statement
theorem num_occupied_third_floor_rooms : 
  let total_full_occupancy_cost := rooms_per_floor * first_floor_rent + rooms_per_floor * second_floor_rent + rooms_per_floor * third_floor_rent
  let revenue_difference := total_full_occupancy_cost - monthly_earnings
  revenue_difference / third_floor_rent = 1 → rooms_per_floor - revenue_difference / third_floor_rent = 2 :=
by
  sorry

end num_occupied_third_floor_rooms_l396_396228


namespace range_of_a_l396_396715

noncomputable def f (x : ℝ) := 4 * x^3 - 3 * x

def f_prime (x : ℝ) := 12 * x^2 - 3

theorem range_of_a (a : ℝ) 
  (h1 : a < -1 / 2)
  (h2 : a + 2 > -1 / 2)
  (h3 : 4 * a^3 - 3 * a < 1) :
  a ∈ Ioo (-5 / 2) (-1 / 2) :=
begin
  sorry
end

end range_of_a_l396_396715


namespace maximal_word_length_l396_396810

theorem maximal_word_length (n : ℕ) :
  ∃ w : List Char, 
    (∀ i, i < w.length - 1 → w.nth_le i (sorry) ≠ w.nth_le (i + 1) (sorry)) ∧
    (¬ ∃ a b : Char, a ≠ b ∧ 
      ((List.filter (λ x, x = a ∨ x = b) w) = [a, b, a, b] ∨
       List.filter (λ x, x = a ∨ x = b) w = [b, a, b, a])) ∧
      w.length = 2 * n - 1 :=
sorry

end maximal_word_length_l396_396810


namespace average_candies_correct_l396_396343

def candy_counts : List ℕ := [16, 22, 30, 26, 18, 20]
def num_members : ℕ := 6
def total_candies : ℕ := List.sum candy_counts
def average_candies : ℕ := total_candies / num_members

theorem average_candies_correct : average_candies = 22 := by
  -- Proof is omitted, as per instructions
  sorry

end average_candies_correct_l396_396343


namespace mike_books_l396_396353

theorem mike_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  ∃ (mike_books : ℕ), mike_books = total_books - tim_books :=
by {
  use 20,
  rw [h1, h2],
  norm_num,
  sorry
}

end mike_books_l396_396353


namespace value_of_expression_l396_396925

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l396_396925


namespace number_of_subsets_containing_7_l396_396573

-- Define the set
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the subset condition
def contains_7 (A : Set ℕ) : Prop := 7 ∈ A

-- Define the count of subsets containing 7
def count_subsets_containing_7 : ℕ := 
  (Set.powerset S).filter contains_7).card

-- The theorem statement
theorem number_of_subsets_containing_7 : count_subsets_containing_7 = 64 := by
  sorry

end number_of_subsets_containing_7_l396_396573


namespace exists_line_intersecting_all_segments_l396_396404

theorem exists_line_intersecting_all_segments 
  (segments : List (ℝ × ℝ)) 
  (h1 : ∀ (P Q R : (ℝ × ℝ)), P ∈ segments → Q ∈ segments → R ∈ segments → ∃ (L : ℝ × ℝ → Prop), L P ∧ L Q ∧ L R) :
  ∃ (L : ℝ × ℝ → Prop), ∀ (S : (ℝ × ℝ)), S ∈ segments → L S :=
by
  sorry

end exists_line_intersecting_all_segments_l396_396404


namespace alex_shirts_l396_396425

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l396_396425


namespace convex_n_gon_divisible_by_3_l396_396788

theorem convex_n_gon_divisible_by_3 
  (n : ℕ) 
  (convex_ngon : Type) 
  [is_convex_ngon : is_convex_polygon convex_ngon n] 
  (triangulated : ∀ {d : diagonal}, non_intersecting_diagonals convex_ngon d) 
  (odd_triangles_convergence : ∀ (v : vertex convex_ngon), 
    odd (count_triangles_converging_at v triangular_division)) 
  : 
  n % 3 = 0 := 
sorry

namespace mynamespace
-- Add necessary definitions for is_convex_polygon, diagonal, 
-- non_intersecting_diagonals, vertex, count_triangles_converging_at, etc. here

-- Helper states/structures
structure convex_ngon (n : ℕ) := 
(vertices : fin n → fin n → Prop)

class is_convex_polygon (P : convex_ngon) (n : ℕ) := 
(property_one : ∀ (a b c : fin n), -- some convex property) 
(property_two : ∀ (d : diagonal P), non_intersecting_diagonals P d)

structure vertex (P : convex_ngon) := 
-- Vertex properties and triangulation property definitions here

structure diagonal (P : convex_ngon) := 
-- Diagonal properties here

def count_triangles_converging_at 
  (v : vertex P) 
  (triangular_division : Type) 
  : nat := 
sorry

def odd (a : nat) : Prop := a % 2 = 1

open is_convex_polygon
open convex_ngon
end mynamespace

end convex_n_gon_divisible_by_3_l396_396788


namespace nancy_age_l396_396272

variable (n g : ℕ)

theorem nancy_age (h1 : g = 10 * n) (h2 : g - n = 45) : n = 5 :=
by
  sorry

end nancy_age_l396_396272


namespace even_function_condition_odd_functions_sufficient_not_necessary_l396_396182

variable {R : Type} [AddGroup R] [H0: OrderedRing R]
variable {f g : R → R}

theorem even_function_condition (hf_even : ∀ x, f (-x) = f x) (hg_even : ∀ x, g (-x) = g x) :
  (∀ x, (f x * g x) = (f (-x) * g (-x))) :=
by
  sorry

theorem odd_functions (f g : R → R) (hf_odd : ∀ x, f (-x) = - f x) (hg_odd : ∀ x, g (-x) = - g x):
  (∀ x, f (0 * x) = 0) → (∀ x , g (0 * x ) = 0 )
by 
    sorry

theorem sufficient_not_necessary (hf_even : ∀ x, f (-x) = f x) (hg_even : ∀ x, g (-x) = g x) :
  (∀ x, f (-x) = - f x → ∀ x , g (-x) = - g x): A 
by
  exact A

  
 
end even_function_condition_odd_functions_sufficient_not_necessary_l396_396182


namespace sin_add_alpha_cos_sub_two_alpha_l396_396884

variable (α : ℝ)

theorem sin_add_alpha (h1 : α ∈ Ioo (π / 2) π) (h2 : sin α = sqrt 5 / 5) :
  sin (π / 4 + α) = - (sqrt 10) / 10 :=
sorry

theorem cos_sub_two_alpha (h1 : α ∈ Ioo (π / 2) π) (h2 : sin α = sqrt 5 / 5) :
  cos (5 * π / 6 - 2 * α) = - (4 + 3 * sqrt 3) / 10 :=
sorry

end sin_add_alpha_cos_sub_two_alpha_l396_396884


namespace polynomial_has_given_roots_l396_396756

theorem polynomial_has_given_roots : 
  (∀ x : ℝ, x = sqrt (8 + sqrt 13) ∨ x = -sqrt (8 + sqrt 13) ∨ x = sqrt (8 - sqrt 13) ∨ x = -sqrt (8 - sqrt 13) → x^4 - 16 * x^2 + 51 = 0) :=
sorry

end polynomial_has_given_roots_l396_396756


namespace solve_equation_l396_396693

theorem solve_equation :
  ∀ (x : ℝ), x * (3 * x + 6) = 7 * (3 * x + 6) → (x = 7 ∨ x = -2) :=
by
  intro x
  sorry

end solve_equation_l396_396693


namespace binomial_7_2_l396_396036

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396036


namespace solve_for_a_l396_396529

theorem solve_for_a (x a : ℤ) (h1 : x = 3) (h2 : x + 2 * a = -1) : a = -2 :=
by
  sorry

end solve_for_a_l396_396529


namespace find_m_of_odd_number_sequence_l396_396104

theorem find_m_of_odd_number_sequence : 
  ∃ m : ℕ, m > 1 ∧ (∃ a : ℕ, a = m * (m - 1) + 1 ∧ a = 2023) ↔ m = 45 :=
by
    sorry

end find_m_of_odd_number_sequence_l396_396104


namespace non_triangular_areas_relation_l396_396396

noncomputable def sides : ℕ × ℕ × ℕ := (13, 14, 15)

def semi_perimeter (a b c : ℕ) : ℕ := (a + b + c) / 2

def triangle_area (a b c : ℕ) (s : ℕ) : ℝ := 
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def circumradius (a b c : ℕ) (Δ : ℝ) : ℝ :=
  (a * b * c : ℝ) / (4 * Δ)

def circle_area (R : ℝ) : ℝ :=
  Real.pi * R^2

def C (circle_area : ℝ) : ℝ :=
  circle_area / 2

def non_triangular_areas_sum (A B : ℝ) (C : ℝ) : Prop :=
  A + B + 84 = C

theorem non_triangular_areas_relation :
  ∃ (A B C Δ : ℝ) (a b c : ℕ), 
    let s := semi_perimeter a b c in
    let Δ := triangle_area a b c s in 
    let R := circumradius a b c Δ in
    let circle_area := circle_area R in 
    let C := C circle_area in
    non_triangular_areas_sum A B C
:=
by
  sorry

end non_triangular_areas_relation_l396_396396


namespace circle_passing_points_l396_396304

variable (d : ℝ)
variable (A B C K L M : EuclideanGeometry.Point)
variable (A' : EuclideanGeometry.Point)
variable (circle : EuclideanGeometry.Point → ℝ → EuclideanGeometry.Circle)

-- Conditions
def equilateral_triangle (A B C : EuclideanGeometry.Point) (d : ℝ) : Prop :=
  EuclideanGeometry.dist A B = d ∧ EuclideanGeometry.dist B C = d ∧ EuclideanGeometry.dist C A = d

def equidistant_points (A B C K L M : EuclideanGeometry.Point) (d : ℝ) : Prop :=
  EuclideanGeometry.dist A B = d ∧ EuclideanGeometry.dist B C = d ∧ EuclideanGeometry.dist C A = d ∧
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ EuclideanGeometry.angle A K B = θ ∧ EuclideanGeometry.angle B L C = θ ∧ EuclideanGeometry.angle C M A = θ

def shorter_arcs (A B C K L M : EuclideanGeometry.Point) : Prop :=
  -- assumption of shorter arcs division into parts
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = EuclideanGeometry.angle B K C ∧ θ₂ = EuclideanGeometry.angle C L A ∧ θ₃ = EuclideanGeometry.angle A M B ∧ 
  0 < θ₁ ∧ 0 < θ₂ ∧ 0 < θ₃ ∧ θ₁ + θ₂ + θ₃ = 2 * π

-- Theorem
theorem circle_passing_points (d : ℝ) (A B C K L M A' : EuclideanGeometry.Point)
  (h_eq_triangle: equilateral_triangle A B C d)
  (h_equidistant: equidistant_points A B C K L M d)
  (h_shorter_arcs: shorter_arcs A B C K L M): 
  EuclideanGeometry.circle K (EuclideanGeometry.dist K L) M ∧ EuclideanGeometry.circle K (EuclideanGeometry.dist K L) A' :=
by
  sorry

end circle_passing_points_l396_396304


namespace course_selection_schemes_l396_396792

-- Define the problem parameters and conditions
def num_courses : ℕ := 4
def num_students : ℕ := 4
def num_empty_courses : ℕ := 2
def num_filled_courses := num_courses - num_empty_courses

-- The formal statement of the problem
theorem course_selection_schemes : 
  ∃(schemes : ℕ), schemes = 18 ∧
  num_courses = 4 ∧
  num_students = 4 ∧
  each_student_chooses_one_course num_students num_courses ∧
  exactly_two_courses_without_students num_courses num_empty_courses := 
sorry

-- Definitions representing conditions
def each_student_chooses_one_course (students courses : ℕ) : Prop := 
  students = courses

def exactly_two_courses_without_students (courses empty_courses : ℕ) : Prop :=
  empty_courses = 2

end course_selection_schemes_l396_396792


namespace find_smallest_m_l396_396649

theorem find_smallest_m (b : ℝ) (hb : b = Real.pi / 2010) : ∃ m : ℕ, m > 0 ∧ 
  (2 * (Finset.range m).sum (λ k, Real.cos ((k + 1)^2 * b) * Real.sin ((k + 1) * b))).denom = 1 ∧ m = 67 :=
by
  sorry

end find_smallest_m_l396_396649


namespace remainder_correct_l396_396753

def dividend : ℕ := 165
def divisor : ℕ := 18
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem remainder_correct {d q r : ℕ} (h1 : d = dividend) (h2 : q = quotient) (h3 : r = divisor * q) : d = 165 → q = 9 → 165 = 162 + remainder :=
by { sorry }

end remainder_correct_l396_396753


namespace proj_matrix_eq_Q_l396_396475

noncomputable def Q : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    (1 / 14 : ℚ), (3 / 14 : ℚ), (-2 / 14 : ℚ),
    (3 / 14 : ℚ), (9 / 14 : ℚ), (-6 / 14 : ℚ),
    (-2 / 14 : ℚ), (-6 / 14 : ℚ), (4 / 14 : ℚ)
  ]

theorem proj_matrix_eq_Q (v : Fin 3 → ℚ) :
  let u : Fin 3 → ℚ := fun i => 
    if i = 0 then 1 else if i = 1 then 3 else -2
  in
  (Q ⬝ v) =  
    let dot_product (a b : Fin 3 → ℚ) : ℚ := 
      Finset.univ.sum (fun i => a i * b i)
    let scalar := dot_product v u / dot_product u u
    fun i =>
      scalar * u i 
:= by sorry

end proj_matrix_eq_Q_l396_396475


namespace problem_I_problem_II_l396_396162
open Real

-- Definitions based on conditions
def f (x : ℝ) := a * x^3 + b * x
def g (x : ℝ) := f x + 3 * x - x^2 - 3
def t (x : ℝ) := c / x^2 + log x

-- Problem (I): Proving the analytic expression and the monotonic decreasing interval
theorem problem_I (a b : ℝ) 
  (h_tangent : 27 * a + b = 24) 
  (h_extreme : 3 * a + b = 0) : 
  f = (fun x => x^3 - 3 * x) ∧ 
  ∀ x ∈ Icc (-1 : ℝ) 1, deriv f x ≤ 0 := sorry

-- Problem (II): Proving the range for the real number c
theorem problem_II {c : ℝ}
  (h_conditions : ∀ (x1 x2 : ℝ), x1 ∈ Icc (1/3 : ℝ) 2 → x2 ∈ Icc (1/3 : ℝ) 2 → x1 * t x1 ≥ g x2) : 
  1 ≤ c := sorry

end problem_I_problem_II_l396_396162


namespace initial_average_mark_of_class_l396_396702

theorem initial_average_mark_of_class (A : ℝ) :
  (∃ (total_students : ℕ) (excluded_students : ℕ) (remaining_students : ℕ) 
     (excluded_avg : ℝ) (remaining_avg : ℝ),
      total_students = 20 ∧ 
      excluded_students = 5 ∧ 
      remaining_students = 15 ∧ 
      excluded_avg = 50 ∧ 
      remaining_avg = 90 ∧ 
      total_students * A = remaining_students * remaining_avg + excluded_students * excluded_avg) →
  A = 80 :=
by
  intro h
  obtain ⟨total_students, excluded_students, remaining_students, excluded_avg, remaining_avg, 
          h1, h2, h3, h4, h5, equation⟩ := h
  have : 20 * A = 1350 + 250 := equation
  have : 20 * A = 1600 := by simp [this]
  have A_eq : A = 80 := by linarith
  exact A_eq

end initial_average_mark_of_class_l396_396702


namespace Sam_age_l396_396289

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l396_396289


namespace binomial_7_2_eq_21_l396_396045

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396045


namespace sugar_mixture_problem_l396_396914

theorem sugar_mixture_problem :
  ∃ x : ℝ, (9 * x + 7 * (63 - x) = 0.9 * (9.24 * 63)) ∧ x = 41.724 :=
by
  sorry

end sugar_mixture_problem_l396_396914


namespace solution_set_of_inequality_l396_396340

theorem solution_set_of_inequality :
  { x : ℝ | x > 0 ∧ x < 1 } = { x : ℝ | 1 / x > 1 } :=
by
  sorry

end solution_set_of_inequality_l396_396340


namespace chord_length_l396_396096

theorem chord_length : 
  ∀ (a b r : ℝ), a = 1 → b = 2 → r = 5 → 
  (∀ x y, y = 3 * x → (x + 1) ^ 2 + (y - 2) ^ 2 = r^2 → 
   ∃ A B, distance A B = 3 * √10) := by
  sorry

end chord_length_l396_396096


namespace ratio_expression_value_l396_396935

theorem ratio_expression_value (A B C : ℚ) (hA : A = 3 * B / 2) (hC : C = 5 * B / 2) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by sorry

end ratio_expression_value_l396_396935


namespace sqrt_inequality_l396_396239

theorem sqrt_inequality (a : ℕ → ℝ) (n : ℕ) (hn : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i) :
  (∑ i in range (n + 1), sqrt (∑ j in range (n + 1 - i), a (i + j + 1))) 
  ≥ sqrt (∑ i in range (n + 1), (i + 1)^2 * a (i + 1)) :=
sorry

end sqrt_inequality_l396_396239


namespace arithmetic_sequence_properties_l396_396887

noncomputable def arithmeticSeq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_properties (a₁ d : ℕ) (n : ℕ) (h1 : d = 2)
  (h2 : (a₁ + d)^2 = a₁ * (a₁ + 3 * d)) :
  (a₁ = 2) ∧ (∃ S, S = (n * (2 * a₁ + (n - 1) * d)) / 2 ∧ S = n^2 + n) :=
by 
  sorry

end arithmetic_sequence_properties_l396_396887


namespace lockers_after_ghosts_l396_396968

theorem lockers_after_ghosts : 
  ∀ (n : ℕ), (n = 1000) →
  ∃ (open_lockers_count : ℕ), 
  (open_lockers_count = 31) :=
begin
  intros,
  have h_lockers : n = 1000 := by assumption,
  existsi 31,
  sorry
end

end lockers_after_ghosts_l396_396968


namespace length_of_intercepted_segment_l396_396623

-- Define the parametric line
def line (t : ℝ) : ℝ × ℝ := (1 + t, 1 + 2 * t)

-- Define the polar equation of the curve
def curve (θ : ℝ) : ℝ := 2 * real.cos θ

-- Define the proof statement
theorem length_of_intercepted_segment : 
  let intersection_points := {p : ℝ × ℝ | ∃ t θ, p = line t ∧ p = (curve θ * real.cos θ, curve θ * real.sin θ)} in
  ∃ A B ∈ intersection_points, 
  ∥A - B∥ = 4 * real.sqrt 5 / 5 :=
sorry

end length_of_intercepted_segment_l396_396623


namespace min_distance_from_origin_l396_396879

-- Define the condition of the problem
def condition (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 4 = 0

-- Statement of the problem in Lean 4
theorem min_distance_from_origin (x y : ℝ) (h : condition x y) : 
  ∃ m : ℝ, m = Real.sqrt (x^2 + y^2) ∧ m = Real.sqrt 13 - 3 := 
sorry

end min_distance_from_origin_l396_396879


namespace problem_part1_problem_part2_l396_396184

section
variable (x y : ℝ)
variable (hx : x > 0) (hy : y > 0)

def op (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem problem_part1 : op 7 4 = 11 / 29 := by
  sorry

theorem problem_part2 : op 2 (op 7 4) = 23 / 17 := by
  have h1 : op 7 4 = 11 / 29 := by admit -- Assuming result from part1
  sorry

end

end problem_part1_problem_part2_l396_396184


namespace main_theorem_l396_396074

variable {R : Type} [LinearOrderedField R] (f : R → R) (a x₁ x₂ : R)

def increasing (f : R → R) (s : Set R) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def even (f : R → R) : Prop :=
  ∀ ⦃x⦄, f x = f (-x)

theorem main_theorem (h_incr : increasing f (Set.Iio a))
    (h_even : even (λ x, f (x + a)))
    (h1 : x₁ < a) (h2 : a < x₂) (h3 : abs (x₁ - a) < abs (x₂ - a)) :
  f (2 * a - x₁) > f (2 * a - x₂) :=
by sorry

end main_theorem_l396_396074


namespace transform_to_zero_set_l396_396656

def S (p : ℕ) : Finset ℕ := Finset.range p

def P (p : ℕ) (x : ℕ) : ℕ := 3 * x ^ ((2 * p - 1) / 3) + x ^ ((p + 1) / 3) + x + 1

def remainder (n p : ℕ) : ℕ := n % p

theorem transform_to_zero_set (p k : ℕ) (hp : Nat.Prime p) (h_cong : p % 3 = 2) (hk : 0 < k) :
  (∃ n : ℕ, ∀ i ∈ S p, remainder (P p i) p = n) ∨ (∃ n : ℕ, ∀ i ∈ S p, remainder (i ^ k) p = n) ↔
  Nat.gcd k (p - 1) > 1 :=
sorry

end transform_to_zero_set_l396_396656


namespace mary_walking_speed_l396_396666

-- Definitions based on the conditions:
def distance_sharon (t : ℝ) : ℝ := 6 * t
def distance_mary (x t : ℝ) : ℝ := x * t
def total_distance (x t : ℝ) : ℝ := distance_sharon t + distance_mary x t

-- Lean statement to prove that the speed x is 4 given the conditions
theorem mary_walking_speed (x : ℝ) (t : ℝ) (h1 : t = 0.3) (h2 : total_distance x t = 3) : x = 4 :=
by
  sorry

end mary_walking_speed_l396_396666


namespace odd_function_ln_negx_l396_396241

theorem odd_function_ln_negx (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_positive : ∀ x, x > 0 → f x = Real.log x) :
  ∀ x, x < 0 → f x = -Real.log (-x) :=
by 
  intros x hx_neg
  have hx_pos : -x > 0 := by linarith
  rw [← h_positive (-x) hx_pos, h_odd x]
  sorry

end odd_function_ln_negx_l396_396241


namespace Alex_shirt_count_l396_396427

variables (Ben_shirts Joe_shirts Alex_shirts : ℕ)

-- Conditions from the problem
def condition1 := Ben_shirts = 15
def condition2 := Ben_shirts = Joe_shirts + 8
def condition3 := Joe_shirts = Alex_shirts + 3

-- Statement to prove
theorem Alex_shirt_count : condition1 ∧ condition2 ∧ condition3 → Alex_shirts = 4 :=
by
  intros h
  sorry

end Alex_shirt_count_l396_396427


namespace blue_tetrahedron_volume_correct_l396_396398

-- Define the side length of the cube
def side_length : ℝ := 7

-- Define the volume of the cube based on its side length
def cube_volume : ℝ := side_length ^ 3

-- Define the area of the base of each corner tetrahedron
def base_area : ℝ := 0.5 * side_length * side_length

-- Define the height of each corner tetrahedron, which equals the side length of the cube
def height : ℝ := side_length

-- Define the volume of each corner tetrahedron
def corner_tetrahedron_volume : ℝ := (1/3) * base_area * height

-- There are four such corner tetrahedra in the cube
def total_corner_tetrahedra_volume : ℝ := 4 * corner_tetrahedron_volume

-- Define the volume of the blue vertices tetrahedron by subtracting corner tetrahedra volumes from the cube's volume
def blue_tetrahedron_volume : ℝ := cube_volume - total_corner_tetrahedra_volume

-- The Lean statement proving the target volume
theorem blue_tetrahedron_volume_correct : 
  blue_tetrahedron_volume = 114.3332 := 
by
  sorry

end blue_tetrahedron_volume_correct_l396_396398


namespace sum_first_1000_terms_sequence_l396_396072

-- Definition of the sequence sum
def sequenceSum (n : ℕ) : ℕ := 
  let total_blocks := (1 + 1) + (3 * 1 + 1) + (3 * 2 + 1) + ... + (3 * 43 + 1) 
  -- Sum of terms in the first 44 blocks 
  let sum_first_990_terms := (44 * (3 * 44 - 1)) / 2
  -- Adding remaining 10 terms which are all 3's
  let sum_remaining_10_terms := 10 * 3
  sum_first_990_terms + sum_remaining_10_terms

theorem sum_first_1000_terms_sequence : 
  sequenceSum 1000 = 2912 := 
by {
  sorry
}

end sum_first_1000_terms_sequence_l396_396072


namespace Tim_income_percentage_less_than_Juan_l396_396668

-- Definitions for the problem
variables (T M J : ℝ)

-- Conditions based on the problem
def condition1 : Prop := M = 1.60 * T
def condition2 : Prop := M = 0.80 * J

-- Goal statement
theorem Tim_income_percentage_less_than_Juan :
  condition1 T M ∧ condition2 M J → T = 0.50 * J :=
by sorry

end Tim_income_percentage_less_than_Juan_l396_396668


namespace find_length_of_crease_l396_396467

theorem find_length_of_crease
  (DEF : Type) [linear_ordered_field DEF]
  (D E F D' R S : DEF)
  (ED' : DEF := 2)
  (D'F : DEF := 3)
  (equilateral_triangle_DEF : true) -- Equilateral triangle assumption
  (folding_condition : true) -- Folding condition assumption
  :
  ∃ RS : DEF, RS = 9/5 := by
  sorry

end find_length_of_crease_l396_396467


namespace middle_integer_is_n_minus_1_l396_396949

theorem middle_integer_is_n_minus_1 (n : ℤ) :
  let lst := [n + 3, n - 9, n - 4, n + 6, n - 1].sort in
  lst.nth 2 = some (n - 1) :=
by
  sorry

end middle_integer_is_n_minus_1_l396_396949


namespace eleven_billion_in_scientific_notation_l396_396618

namespace ScientificNotation

def Yi : ℝ := 10 ^ 8

theorem eleven_billion_in_scientific_notation : (11 * (10 : ℝ) ^ 9) = (1.1 * (10 : ℝ) ^ 10) :=
by 
  sorry

end ScientificNotation

end eleven_billion_in_scientific_notation_l396_396618


namespace tan_beta_rational_iff_square_integer_l396_396307

theorem tan_beta_rational_iff_square_integer (p q : ℤ) (hpq : q ≠ 0) (tan_alpha_def : tan α = p / q) :
  (∃ β : ℚ, tan (2 * β) = tan (3 * α)) ↔ ∃ k : ℤ, k^2 = p^2 + q^2 :=
sorry

end tan_beta_rational_iff_square_integer_l396_396307


namespace choir_group_members_l396_396787

theorem choir_group_members (first second third num_absent : ℕ) (choir_total : ℕ) (fourth : ℕ) :
  first = 22 →
  second = 33 →
  third = 36 →
  fourth = second - 3 →
  num_absent = 7 →
  choir_total = 162 →
  ∃ fifth : ℕ, fifth = choir_total - num_absent - (first + second + third + fourth) ∧ fifth = 34 :=
by
  intros h_first h_second h_third h_fourth h_num_absent h_choir_total
  use choir_total - num_absent - (first + second + third + fourth)
  split
  · sorry
  · sorry

end choir_group_members_l396_396787


namespace apples_per_box_l396_396347

-- Defining the given conditions
variable (apples_per_crate : ℤ)
variable (number_of_crates : ℤ)
variable (rotten_apples : ℤ)
variable (number_of_boxes : ℤ)

-- Stating the facts based on given conditions
def total_apples := apples_per_crate * number_of_crates
def remaining_apples := total_apples - rotten_apples

-- The statement to prove
theorem apples_per_box 
    (hc1 : apples_per_crate = 180)
    (hc2 : number_of_crates = 12)
    (hc3 : rotten_apples = 160)
    (hc4 : number_of_boxes = 100) :
    (remaining_apples apples_per_crate number_of_crates rotten_apples) / number_of_boxes = 20 := 
sorry

end apples_per_box_l396_396347


namespace cylinder_area_ratio_l396_396189

theorem cylinder_area_ratio (r h : ℝ) (h_eq : h = 2 * r * Real.sqrt π) :
  let S_lateral := 2 * π * r * h
  let S_total := S_lateral + 2 * π * r^2
  S_total / S_lateral = 1 + (1 / (2 * Real.sqrt π)) := by
sorry

end cylinder_area_ratio_l396_396189


namespace Lakota_spent_l396_396229

-- Define the conditions
def U : ℝ := 9.99
def Mackenzies_cost (N : ℝ) : ℝ := 3 * N + 8 * U
def cost_of_Lakotas_disks (N : ℝ) : ℝ := 6 * N + 2 * U

-- State the theorem
theorem Lakota_spent (N : ℝ) (h : Mackenzies_cost N = 133.89) : cost_of_Lakotas_disks N = 127.92 :=
by
  sorry

end Lakota_spent_l396_396229


namespace train_length_calculation_l396_396420

theorem train_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (platform_length_m : ℝ) (train_length_m: ℝ) : speed_kmph = 45 → time_seconds = 51.99999999999999 → platform_length_m = 290 → train_length_m = 360 :=
by
  sorry

end train_length_calculation_l396_396420


namespace sam_age_l396_396293

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l396_396293


namespace number_of_zeros_in_99_l396_396394

theorem number_of_zeros_in_99 : (nat.count_digit 0 99) = 2 :=
by sorry

end number_of_zeros_in_99_l396_396394


namespace area_triangle_AEF_l396_396221

theorem area_triangle_AEF (ABC : Triangle) (BC : Line)
  (F : Point) (AB : Line) (D : Point) (DB : Line) (E : Point)
  (hF_midpoint : F = midpoint(BC))
  (hABC_area : area(ABC) = 120)
  (hD_midpoint : D = midpoint(AB))
  (hE_midpoint : E = midpoint(DB)) :
  area(triangle(A, E, F)) = 45 :=
sorry

end area_triangle_AEF_l396_396221


namespace negation_of_universal_l396_396857

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ¬ (∀ x : ℤ, x^3 < 1) ↔ ∃ x : ℤ, x^3 ≥ 1 :=
by
  sorry

end negation_of_universal_l396_396857


namespace find_a_plus_b_l396_396698

open Function

theorem find_a_plus_b (a b : ℝ) (f g h : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x - b)
  (h_g : ∀ x, g x = -4 * x - 1)
  (h_h : ∀ x, h x = f (g x))
  (h_h_inv : ∀ y, h⁻¹ y = y + 9) :
  a + b = -9 := 
by
  -- Proof goes here.
  sorry

end find_a_plus_b_l396_396698


namespace digit_product_equality_l396_396748

theorem digit_product_equality (x y z : ℕ) (hx : x = 3) (hy : y = 7) (hz : z = 1) :
  x * (10 * x + y) = 111 * z :=
by
  -- Using hx, hy, and hz, the proof can proceed from here
  sorry

end digit_product_equality_l396_396748


namespace Claire_plans_to_buy_five_cookies_l396_396025

theorem Claire_plans_to_buy_five_cookies :
  let initial_amount := 100
  let latte_cost := 3.75
  let croissant_cost := 3.50
  let days := 7
  let cookie_cost := 1.25
  let remaining_amount := 43
  let daily_expense := latte_cost + croissant_cost
  let weekly_expense := daily_expense * days
  let total_spent := initial_amount - remaining_amount
  let cookie_spent := total_spent - weekly_expense
  let cookies := cookie_spent / cookie_cost
  cookies = 5 :=
by {
  sorry
}

end Claire_plans_to_buy_five_cookies_l396_396025


namespace max_a_value_l396_396997

noncomputable def f (x a : ℝ) := Real.exp (2 * x) + a
noncomputable def g (x : ℝ) := Real.exp x + x

theorem max_a_value 
  (x : ℕ → ℝ) -- Sequence of x_i
  (a : ℝ) -- Parameter a
  (h1 : ∀ i, 1 ≤ i → i ≤ 2023 → x i ∈ Set.Icc (-1) 1)
  (h2 : (∑ i in Finset.range 2022, f (x i) a) + g (x 2023) = (∑ i in Finset.range 2022, g (x i)) + f (x 2023) a) : 
  a ≤ (Real.exp 2 - Real.exp 1 - 1) / 2021 :=
begin
  sorry
end

end max_a_value_l396_396997


namespace замок_permutations_ротор_permutations_топор_permutations_колокол_permutations_l396_396560

theorem замок_permutations : ∏ i in (list.range 5).map (λ x, x + 1) = 120 := sorry

theorem ротор_permutations : ∏ i in (list.range 5).map (λ x, x + 1) / ((∏ i in (list.range 2).map (λ x, x + 1)) * (∏ i in (list.range 2).map (λ x, x + 1))) = 30 := sorry

theorem топор_permutations : ∏ i in (list.range 5).map (λ x, x + 1) / (∏ i in (list.range 2).map (λ x, x + 1)) = 60 := sorry

theorem колокол_permutations : 
  ∏ i in (list.range 7).map (λ x, x + 1) / 
  ((∏ i in (list.range 2).map (λ x, x + 1)) * (∏ i in (list.range 3).map (λ x, x + 1)) * (∏ i in (list.range 2).map (λ x, x + 1))) = 210 := sorry

end замок_permutations_ротор_permutations_топор_permutations_колокол_permutations_l396_396560


namespace OK_eq_KB_l396_396875

-- Relevant geometric definitions and constructions
variable {O A B C E K : Point}
variable (circle : Circle)
variable [tangentOA : Tangent circle O A]
variable [tangentOB : Tangent circle O B]
variable (rayAC : Ray A C) [parallel : Parallel rayAC (Line O B)]
variable (segmentOC : Segment O C)
variable [intersectionE : Intersection segmentOC circle E]
variable [intersectionK : Intersection (Line A E) (Line O B) K]

-- The theorem to be proven
theorem OK_eq_KB : dist O K = dist K B := sorry

end OK_eq_KB_l396_396875


namespace odd_function_solution_l396_396956

theorem odd_function_solution (a : ℝ) :
  (∀ x : ℝ, ln ((a * (-x) - 1) / (2 * (-x) + 1)) = -ln ((a * x - 1) / (2 * x + 1))) ↔ a = 2 := 
by
  sorry

end odd_function_solution_l396_396956


namespace water_park_admission_l396_396317

def adult_admission_charge : ℝ := 1
def child_admission_charge : ℝ := 0.75
def children_accompanied : ℕ := 3
def total_admission_charge (adults : ℝ) (children : ℝ) : ℝ := adults + children

theorem water_park_admission :
  let adult_charge := adult_admission_charge
  let children_charge := children_accompanied * child_admission_charge
  total_admission_charge adult_charge children_charge = 3.25 :=
by sorry

end water_park_admission_l396_396317


namespace calculate_angle_hat_ead_l396_396523

-- Define the geometric setup
variables (A B C D E F G : Type)
-- Assume the points form a symmetrical heptagon with the given distances
variables [metric_space α] [normed_group α] [normed_space ℝ α]
variables (A B C D E F G : α)
-- Add the conditions
axiom distances : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E F ∧ dist E F = dist F G ∧ dist F G = dist G A ∧ dist G A = dist A B
axiom collinear1 : ∃ (l : α), A ∈ l ∧ B ∈ l ∧ F ∈ l ∧ D ∈ l
axiom collinear2 : ∃ (m : α), A ∈ m ∧ G ∈ m ∧ C ∈ m ∧ E ∈ m

-- Prove the angle calculation
theorem calculate_angle_hat_ead : angle E A D = π / 7 := by
  sorry

end calculate_angle_hat_ead_l396_396523


namespace percentage_of_two_is_point_eight_l396_396785

theorem percentage_of_two_is_point_eight (p : ℝ) : (p / 100) * 2 = 0.8 ↔ p = 40 := 
by
  sorry

end percentage_of_two_is_point_eight_l396_396785


namespace largest_tile_size_l396_396381

theorem largest_tile_size (length width : ℕ) (h1 : length = 378) (h2 : width = 525) :
  ∃ tile_size, tile_size = Nat.gcd length width ∧ tile_size = 21 :=
by
  use Nat.gcd length width
  split
  . rfl
  . rw [←Nat.gcd_comm, Nat.gcd_eq_right_iff_dvd.mpr]
    -- proof of gcd being 21 is left as sorry
    sorry

end largest_tile_size_l396_396381


namespace cone_volume_from_sector_l396_396801

theorem cone_volume_from_sector (r : ℝ) (θ : ℝ) (h' : ℝ) (V : ℝ) (cond1 : θ = 3 / 4 * 2 * π) (cond2 : r = 4)
  (cond3 : h' = 4) (circumference : 2 * π ≠ 0) :
  V = 1 / 3 * π * (3) * (3) * (sqrt 7) :=
  by
    -- let arc length (3/4 of full circle circumference) form the base circumference of the cone
    let base_circumference := cond1 * cond2,
    have base_radius := 3, -- since 2πr = 6π, implies r = 3
    have slant_height := cond3,
    have cone_height := sqrt 7,
    -- V = 1/3 π r^2 h
    have vol_cone := 1 / 3 * π * (base_radius) * (base_radius) * cone_height,
    sorry

end cone_volume_from_sector_l396_396801


namespace find_relation_between_m_and_n_l396_396539

section

-- Definitions of the ellipse parameters and eccentricity condition
def ellipse (a b : ℝ) (condition : a > b ∧ b > 0 ∧ a > 0 ∧ (a^2 = b^2 + (b * (3 / sqrt 6))^2)) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

-- Definition of the given circle and its tangency condition
def circle_tangent (b : ℝ) : Prop :=
  (1 - 0 + sqrt 2 - 1) / sqrt 2 = b

-- Slope conditions 
def slope_conditions (k1 k2 k3 : ℝ) (N : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  k1 + k3 = 2 * k2 ∧ k2 = 1 → (P.1 - N.1) / (P.1 - N.1) = (P.2 - N.2) / (P.1 - N.1)

-- Theorem statement putting all the conditions and the result together
theorem find_relation_between_m_and_n :
  ∀ a b k1 k2 k3 : ℝ, 
    ellipse a b (And.intro ((And.intro (lt_trans (nat.cast_pos.mpr zero_lt_one) 
    (by norm_num)) (lt_trans (nat.cast_pos.mpr zero_lt_one) (by norm_num)))) 
    (by norm_num)) ∧
        circle_tangent b ∧ 
        slope_conditions k1 k2 k3 (3, 2) (m, n) →
        m - n - 1 = 0 :=
sorry

end

end find_relation_between_m_and_n_l396_396539


namespace john_spent_at_candy_store_l396_396911

noncomputable def johns_allowance : ℝ := 2.40
noncomputable def arcade_spending : ℝ := (3 / 5) * johns_allowance
noncomputable def remaining_after_arcade : ℝ := johns_allowance - arcade_spending
noncomputable def toy_store_spending : ℝ := (1 / 3) * remaining_after_arcade
noncomputable def remaining_after_toy_store : ℝ := remaining_after_arcade - toy_store_spending
noncomputable def candy_store_spending : ℝ := remaining_after_toy_store

theorem john_spent_at_candy_store : candy_store_spending = 0.64 := by sorry

end john_spent_at_candy_store_l396_396911


namespace number_of_triples_satisfying_conditions_l396_396848

-- Definitions for the conditions
def is_triple (a b c : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) c = 21 ∧ Nat.lcm (Nat.lcm a b) c = 3^17 * 7^15

-- Theorem statement capturing the initial problem conditions and expected result
theorem number_of_triples_satisfying_conditions :
  { t : (ℕ × ℕ × ℕ) // is_triple t.1 t.2 t.3}.toFinset.card = 8064 :=
sorry

end number_of_triples_satisfying_conditions_l396_396848


namespace max_b_c_l396_396518

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end max_b_c_l396_396518


namespace cupcakes_per_box_l396_396297

theorem cupcakes_per_box (total_baked : ℕ) (left_at_home : ℕ) (total_boxes : ℕ) 
  (h_baked : total_baked = 53) (h_left : left_at_home = 2) (h_boxes : total_boxes = 17) : 
  (total_baked - left_at_home) / total_boxes = 3 :=
by
  rw [h_baked, h_left, h_boxes]
  simp
  norm_num
  sorry

end cupcakes_per_box_l396_396297


namespace am_gm_inequality_minimum_value_of_function_l396_396527

variable {a b x y : Real}

theorem am_gm_inequality (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : x > 0) (h5 : y > 0) :
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

theorem minimum_value_of_function {
  (h1 : ∀ x, 0 < x → x < 1/2 → (2/x + 9/(1-2*x)) ≥ 25) 
  (h2 : (2/(1/5) + 9/(1-2*(1/5))) = 25) 
  (h3 : 0 < (1/5)) 
  (h4 : (1/5) < 1/2) :
  ∃ x ∈ (0,1/2), (2 / x + 9 / (1 - 2 * x)) = 25 :=
sorry

end am_gm_inequality_minimum_value_of_function_l396_396527


namespace bug_at_A_after_7_meters_l396_396813

-- Define the probability of being at vertex A after n meters
noncomputable def a : ℕ → ℚ
| 0       := 1
| (n + 1) := (1 / 3) * (1 - a n)

-- Prove the probability of being at vertex A after crawling 7 meters is 182 / 729
theorem bug_at_A_after_7_meters : a 7 = 182 / 729 := 
sorry

end bug_at_A_after_7_meters_l396_396813


namespace max_magnitude_vector_sum_l396_396206

open Real

-- Define the points A and B
def A : ℝ × ℝ := (sqrt 3, 1)

-- Define the function that computes magnitude of a vector
def magnitude (x y : ℝ) : ℝ := sqrt (x * x + y * y)

-- Prove the statement
theorem max_magnitude_vector_sum :
  ∃ (B : ℝ × ℝ), (magnitude B.1 B.2 = 1) ∧ 
  let max_sum := magnitude A.1 A.2 + 1 in
  max_sum = 3 :=
by
  use (0, 1) -- Choose B on the unit circle
  have hB : magnitude (0) (1) = 1 := by simp [magnitude]; norm_num
  have hA : magnitude A.1 A.2 = 2 := by 
    rw [A]; simp [magnitude]; 
    rw [sq_sqrt]; norm_num; linarith [real.sqrt_pos.mpr zero_lt_three]
  rw [<-hA]
  exact ⟨hB, by rfl⟩

end max_magnitude_vector_sum_l396_396206


namespace checkerboard_tiling_l396_396298

theorem checkerboard_tiling (m n k : ℕ) : (∃ f : ℕ × ℕ → option (ℕ × ℕ), 
  (∀ x y, x < m → y < n → ∃ x' y', f (x, y) = some (x', y') ∨ f (x', y') = some (x, y)) →
  (∀ x y, f (x, y) = some (x + 1, y) ∨ f (x, y) = some (x, y + 1) ∨ f (x, y) = some (x - 1, y) ∨ f (x, y) = some (x, y - 1))) ↔ 
  (k ∣ m ∨ k ∣ n) := 
sorry

end checkerboard_tiling_l396_396298


namespace binomial_7_2_l396_396060

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396060


namespace roots_prod_eq_29_l396_396244

open Polynomial

noncomputable def root1 := RootOf (X^3 - 15 * X^2 + 25 * X - 12) 0
noncomputable def root2 := RootOf (X^3 - 15 * X^2 + 25 * X - 12) 1
noncomputable def root3 := RootOf (X^3 - 15 * X^2 + 25 * X - 12) 2

theorem roots_prod_eq_29 : (1 + root1) * (1 + root2) * (1 + root3) = 29 := by
  -- proof goes here
  sorry

end roots_prod_eq_29_l396_396244


namespace ratio_expression_value_l396_396919

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396919


namespace concurrency_of_lines_l396_396513

theorem concurrency_of_lines
  (O X Y : Point)
  (C C' : Circle)
  (A B Z : Point)
  (h1 : C.center = O)
  (h2 : C'.center = X)
  (h3 : X ∈ interior(C))
  (h4 : tangent_to C C' A)
  (h5 : Y ∈ interior(C))
  (h6 : tangent_to C ⟨center := Y, radius := (dist Y B)⟩ B)
  (h7 : tangent_to C' ⟨center := Y, radius := (dist Y B)⟩ Z) :
  concurrent_lines (line_through X B) (line_through Y A) (line_through O Z) :=
sorry

end concurrency_of_lines_l396_396513


namespace measure_of_PBC_l396_396196

noncomputable def triangle_ABC (A B C P : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited P] : Prop :=
  ∃ (∠ ABC : ℝ) (∠ ACB : ℝ) (∠ PAC : ℝ) (∠ PCB : ℝ) (∠ PBC : ℝ),
    ∠ ABC = 40 ∧
    ∠ ACB = 40 ∧
    ∠ PAC = 20 ∧
    ∠ PCB = 30 ∧
    ∠ PBC = 20

theorem measure_of_PBC (A B C P : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited P] :
  triangle_ABC A B C P → ∃ (∠ PBC : ℝ), ∠ PBC = 20 :=
by
  intro h
  use 20
  exact sorry

end measure_of_PBC_l396_396196


namespace melanie_bread_pieces_l396_396270

theorem melanie_bread_pieces :
  ∀ (slices : ℕ), slices = 2 → (∀ (n : ℕ), slices * (2 ^ 2) = 8)
:=
begin
  intros slices slices_eq,
  cases slices_eq,
  intros n,
  simp,
end

end melanie_bread_pieces_l396_396270


namespace isosceles_trapezoid_properties_l396_396434

def isosceles_trapezoid.Axisymmetric (T : Type) : Prop := 
-- Definition of axisymmetric for isosceles trapezoid T goes here.
sorry

def isosceles_trapezoid.Diagonals_Equal (T : Type) : Prop := 
-- Definition that diagonals of isosceles trapezoid T are equal.
sorry

def isosceles_trapezoid.Base_Angles_Equal (T : Type) : Prop := 
-- Definition that base angles of isosceles trapezoid T are equal.
sorry

def isosceles_trapezoid.Complementary_Angles (T : Type) : Prop := 
-- Definition that two sets of angles in isosceles trapezoid T are complementary.
sorry

theorem isosceles_trapezoid_properties
  (T : Type) [IsoscelesTrapezoid T]
  (H1 : isosceles_trapezoid.Axisymmetric T)
  (H2 : isosceles_trapezoid.Diagonals_Equal T)
  (H3 : isosceles_trapezoid.Base_Angles_Equal T)
  (H4 : isosceles_trapezoid.Complementary_Angles T) : 
  (H1 ∧ H2 ∧ ¬H3 ∧ H4) ∨ (H1 ∧ H2 ∧ H3 ∧ ¬H4) ∨ (H1 ∧ ¬H2 ∧ H3 ∧ H4) ∨ (¬H1 ∧ H2 ∧ H3 ∧ H4) := 
sorry

end isosceles_trapezoid_properties_l396_396434


namespace ratio_expression_value_l396_396922

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l396_396922


namespace geom_seq_a1_l396_396212

-- Define a geometric sequence.
def geom_seq (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q ^ n

-- Given conditions
def a2 (a : ℕ → ℝ) : Prop := a 1 = 2 -- because a2 = a(1) in zero-indexed
def a5 (a : ℕ → ℝ) : Prop := a 4 = -54 -- because a5 = a(4) in zero-indexed

-- Prove that a1 = -2/3
theorem geom_seq_a1 (a : ℕ → ℝ) (a1 q : ℝ) (h_geom : geom_seq a a1 q)
  (h_a2 : a2 a) (h_a5 : a5 a) : a1 = -2 / 3 :=
by
  sorry

end geom_seq_a1_l396_396212


namespace fill_grid_diagonals_sum_l396_396978

theorem fill_grid_diagonals_sum :
  ∀ n : ℕ,
  (∃ (fill_function : ℕ × ℕ → ℤ),
    (∀ i j : ℕ, i < n → j < n →
      (sum (λ k, fill_function (i + k, j)) (range n) = 1) ∧
      (sum (λ k, fill_function (i, j + k)) (range n) = 1))) ↔
  (n % 2 = 1 → possible) ∧ (n % 2 = 0 → impossible) :=
sorry

end fill_grid_diagonals_sum_l396_396978


namespace sum_series_eq_half_l396_396083

theorem sum_series_eq_half : 
  (∑ k in (Set.Ioi (0 : ℕ)), (3 ^ (2 ^ k) / (9 ^ (2 ^ k) - 1))) = 1/2 :=
sorry

end sum_series_eq_half_l396_396083


namespace general_term_formula_maximum_area_l396_396522

-- Define the points on the log base 1/3 function
def points_on_log_function (n : ℕ) (hn : n > 0) : ℕ → ℝ × ℝ
| n := (real.exp (real.logb (1 / 3) n), n)

-- Define the sequence a_n
def sequence_a_n (n : ℕ) (hn : n > 0) : ℝ := (1 / 3) ^ n

-- Area of the triangle OAnMn
def triangle_area (n : ℕ) (hn : n > 0) : ℝ :=
  1 / 2 * sequence_a_n n hn * n

-- Prove that the general term formula for the sequence a_n is (1/3)^n
theorem general_term_formula (n : ℕ) (hn : n > 0) :
  ∀ k, points_on_log_function k hn = (sequence_a_n k hn, k) :=
sorry

-- Prove that the maximum area among the triangles is 1/6
theorem maximum_area (n : ℕ) (hn : n > 0) :
  ∃ k, k ≤ n ∧ triangle_area k hn = 1 / 6 :=
sorry

end general_term_formula_maximum_area_l396_396522


namespace savings_correct_l396_396635

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l396_396635


namespace num_subsets_containing_7_l396_396567

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem num_subsets_containing_7 : (S.filter (λ s => 7 ∈ s)).card = 64 := by
  sorry

end num_subsets_containing_7_l396_396567


namespace bandits_cannot_have_equal_coins_l396_396985

-- Define the bandits and coins
inductive Bandit
| B1 | W1 | B2 | W2 | B3 | W3

-- Function to represent the number of coins each bandit has
def coin_count : Bandit → ℕ
| Bandit.B1 => 6
| _ => 0

-- Function to represent the adjacency of bandits
def adjacent (b1 b2 : Bandit) : Prop :=
  match b1, b2 with
  | Bandit.B1, Bandit.W1 => true
  | Bandit.W1, Bandit.B2 => true
  | Bandit.B2, Bandit.W2 => true
  | Bandit.W2, Bandit.B3 => true
  | Bandit.B3, Bandit.W3 => true
  | Bandit.W3, Bandit.B1 => true
  | _, _ => false

-- Invariant: Total number of coins held by black bandits is even
def black_banded (b : Bandit) : bool :=
  match b with
  | Bandit.B1 | Bandit.B2 | Bandit.B3 => true
  | _ => false

-- Prove that it is impossible for all bandits to have the same number of coins
theorem bandits_cannot_have_equal_coins :
  ∀ moves : (Bandit → ℕ) → (Bandit → ℕ),
  (∀ b : Bandit, ∃ k : ℕ, moves (coin_count) b = k ∧ moves (coin_count) b = moves (coin_count) (b.prev) + k) →
  ¬(∃ c : ℕ, ∀ b : Bandit, moves (coin_count) b = c) :=
by
  sorry

end bandits_cannot_have_equal_coins_l396_396985


namespace parabola_tangent_normal_l396_396827

theorem parabola_tangent_normal (x y : ℝ) (A : ℝ × ℝ) :
  let f := λ x, x^2 + 4 * x + 2 in
  A = (1, 7) →
  f 1 = 7 →
  deriv f 1 = 6 →
  (∀ x, y = 6 * x + 1) →
  (∀ x y, x + 6 * y - 43 = 0) :=
by
  intros f A H1 H2 H3 H4; sorry

end parabola_tangent_normal_l396_396827


namespace sequence_integer_terms_count_l396_396727

theorem sequence_integer_terms_count :
  ∃ n, ∀ i, (i < n → (9720 / (3^i)) ∈ ℤ) ∧ (¬ (9720 / (3^n)) ∈ ℤ) :=
begin
  use 6,
  intros i,
  split,
  { intro hi,
    have h1 : 9720 = 2^3 * 3^5 * 5 := by norm_num,
    have h2 : ∀ k, k < 6 → 9720 / (3^k) ∈ ℤ := by
    { intro k, 
      intro hk, 
      norm_cast,
      exact int.coe_nat_dvd.mpr (nat.pow_div_of_le h1 (le_of_lt hk)) },
    exact h2 i hi, },
  { intro hn,
    have h3 : ¬ (9720 / (3^6) ∈ ℤ) := by norm_cast,
    exact h3, },
end

end sequence_integer_terms_count_l396_396727


namespace minimize_sum_of_reciprocals_l396_396874

def dataset : List ℝ := [2, 4, 6, 8]

def mean : ℝ := 5
def variance: ℝ := 5

theorem minimize_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : mean * a + variance * b = 1) : 
  (1 / a + 1 / b) = 20 :=
sorry

end minimize_sum_of_reciprocals_l396_396874


namespace ellipse_C2_equation_slopes_product_constant_l396_396832

-- Definitions for the given conditions
def ellipse_C1 (x y : ℝ) : Prop := (x ^ 2) / 2 + (y ^ 2) = 1
def ellipse_C2 (x y : ℝ) (a b : ℝ) : Prop := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1
def is_focus (sqrt_5 : ℝ) : Prop := sqrt_5 = real.sqrt 5
def midpoint_AB (x y : ℝ) : Prop := x = 2 ∧ y = -1
def vector_eq (x0 y0 x1 y1 x2 y2 : ℝ) : Prop := x0 = x1 + 2 * x2 ∧ y0 = y1 + 2 * y2

-- The proof problem statement
theorem ellipse_C2_equation {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
    (focus_condition : is_focus (real.sqrt 5))
    (midpoint_condition : midpoint_AB 2 (-1))
    : ellipse_C2 x y (real.sqrt 10) (real.sqrt 5) :=
sorry

theorem slopes_product_constant {x0 y0 x1 y1 x2 y2 : ℝ} :
    vector_eq x0 y0 x1 y1 x2 y2 →
    ellipse_C2 x0 y0 (real.sqrt 10) (real.sqrt 5) →
    ellipse_C1 x1 y1 →
    ellipse_C1 x2 y2 →
    let k1 := y1 / x1
    let k2 := y2 / x2 in
    k1 * k2 = -1 / 2 :=
sorry

end ellipse_C2_equation_slopes_product_constant_l396_396832


namespace circle_intersects_simple_curve_l396_396232

theorem circle_intersects_simple_curve
  (Γ : Set ℝ → ℝ) -- Γ: simple curve in ℝ (parametrized by ℝ?)
  (circle : Set (ℝ × ℝ)) -- circle in ℝ²
  (r : ℝ) -- radius r
  (ell : ℝ) -- length ell
  (k : ℕ) -- integer k
  (hΓ : ∀ t₁ t₂, t₁ ≠ t₂ → Γ t₁ ≠ Γ t₂) -- Γ is a simple curve
  (hΓ_in_circle : ∀ t, Γ t ∈ ball 0 r) -- Γ lies inside a circle of radius r
  (hΓ_rectifiable : rectifiable Γ) -- Γ is rectifiable
  (hΓ_length : length Γ = ell) -- Length of Γ is ell
  (h_ell_gt_krpi : ell > k * r * π) :
  ∃ circle : Set (ℝ × ℝ), 
    radius circle = r ∧ 
    (∃ (points : Finset ℝ), points.card ≥ k + 1 ∧ ∀ t ∈ points, Γ t ∈ circle) :=
sorry

end circle_intersects_simple_curve_l396_396232


namespace find_n_l396_396146

theorem find_n (x : ℝ) (n : ℕ) (hx1 : log 10 (sin x) + log 10 (cos x) = -2)
  (hx2 : log 10 (sin x + cos x) = 1/2 * (log 10 n - 2)) : n = 102 :=
by
  sorry

end find_n_l396_396146


namespace division_of_fractions_l396_396365

theorem division_of_fractions :
  (10 / 21) / (4 / 9) = 15 / 14 :=
by
  -- Proof will be provided here 
  sorry

end division_of_fractions_l396_396365


namespace exists_root_in_interval_l396_396891

theorem exists_root_in_interval
    (a b c x₁ x₂ : ℝ)
    (h₁ : a * x₁^2 + b * x₁ + c = 0)
    (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
    ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
sorry

end exists_root_in_interval_l396_396891


namespace value_range_f_l396_396342

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem value_range_f : Set.set_range (λ x : {x // 0 ≤ x ∧ x ≤ 3}, f x) = Set.Icc 0 4 := by
  sorry

end value_range_f_l396_396342


namespace quadrant_of_theta_l396_396180

theorem quadrant_of_theta (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin θ < 0) : (0 < θ ∧ θ < π/2) ∨ (3*π/2 < θ ∧ θ < 2*π) :=
by
  sorry

end quadrant_of_theta_l396_396180


namespace min_value_x_add_2y_l396_396154

theorem min_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end min_value_x_add_2y_l396_396154


namespace three_digit_cubes_divisible_by_16_l396_396576

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l396_396576


namespace base8_357_plus_base13_4CD_eq_1084_l396_396841

def C := 12
def D := 13

def base8_357 := 3 * (8^2) + 5 * (8^1) + 7 * (8^0)
def base13_4CD := 4 * (13^2) + C * (13^1) + D * (13^0)

theorem base8_357_plus_base13_4CD_eq_1084 :
  base8_357 + base13_4CD = 1084 :=
by
  sorry

end base8_357_plus_base13_4CD_eq_1084_l396_396841


namespace binomial_7_2_eq_21_l396_396043

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l396_396043


namespace floor_sqrt_99_eq_9_l396_396484

theorem floor_sqrt_99_eq_9 :
  ∀ x, 81 ≤ x ∧ x < 100 → floor (real.sqrt 99) = 9 :=
by
  sorry

end floor_sqrt_99_eq_9_l396_396484


namespace stations_between_l396_396738

theorem stations_between (n : ℕ) (h : n * (n - 1) / 2 = 306) : n - 2 = 25 := 
by
  sorry

end stations_between_l396_396738


namespace binomial_7_2_l396_396055

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396055


namespace triangle_inequality_l396_396236

variables (a b c S : ℝ) (S_def : S = (a + b + c) / 2)

theorem triangle_inequality 
  (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) 
  ≤ 3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) :=
sorry

end triangle_inequality_l396_396236


namespace simplify_expression_result_l396_396299

noncomputable def simplify_expression (α : ℝ) : ℝ :=
  (sin (4 * α)) / (4 * (sin (π / 4 + α))^2 * tan (π / 4 - α))

theorem simplify_expression_result (α : ℝ) : simplify_expression α = sin (2 * α) :=
by
  sorry

end simplify_expression_result_l396_396299


namespace intersection_A_B_l396_396880

def A (x : ℝ) : Prop := 4 * x - 8 < 0
def B (x : ℝ) : Prop := 9 ^ x < 3

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | x < 1 / 2} :=
by sorry

end intersection_A_B_l396_396880


namespace casey_pumping_minutes_l396_396456

theorem casey_pumping_minutes :
  let pump_rate := 3
  let corn_rows := 4
  let corn_plants_per_row := 15
  let water_needed_per_corn_plant := 0.5
  let num_pigs := 10
  let water_needed_per_pig := 4
  let num_ducks := 20
  let water_needed_per_duck := 0.25
  let total_water_needed := (corn_rows * corn_plants_per_row * water_needed_per_corn_plant) +
                            (num_pigs * water_needed_per_pig) +
                            (num_ducks * water_needed_per_duck)
  let minutes_needed := total_water_needed / pump_rate
  in minutes_needed = 25 :=
by 
  sorry

end casey_pumping_minutes_l396_396456


namespace line_polar_eq_and_intersections_l396_396972

theorem line_polar_eq_and_intersections
  (t : ℝ)
  (rho theta x y : ℝ)
  (h_param_l : x = 2 + (1 / 2) * t ∧ y = (sqrt 3 / 2) * t)
  (h_polar_c : rho = 4 * cos theta)
  (h_cartesian_c : x^2 + y^2 - 4 * x = 0) :
  (sqrt 3 * rho * cos theta - rho * sin theta - 2 * sqrt 3 = 0) ∧
  ((x, y) = (1, -sqrt 3) ∨ (x, y) = (3, sqrt 3)) ∧
  ((rho, theta) = (2, 5 * pi / 3) ∨ (rho, theta) = (2 * sqrt 3, pi / 6)) :=
sorry

end line_polar_eq_and_intersections_l396_396972


namespace alex_shirts_l396_396424

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l396_396424


namespace focal_chord_length_perpendicular_l396_396902

theorem focal_chord_length_perpendicular (x1 y1 x2 y2 : ℝ)
  (h_parabola : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_perpendicular : x1 = x2) :
  abs (y1 - y2) = 4 :=
by sorry

end focal_chord_length_perpendicular_l396_396902


namespace sin_alpha_value_cos_2alpha_plus_pi_over_3_value_l396_396530

section
  -- Given conditions
  variable (α : ℝ) (h1 : π / 2 < α) (h2 : α < π)
  variable (h3 : sin (α / 2) + cos (α / 2) = (3 * real.sqrt 5) / 5)

  -- Statement of the first proof problem
  theorem sin_alpha_value : sin α = 4 / 5 := by
   sorry

  -- Given the value of sin α from the first theorem and the same α interval
  variable (h4 : sin α = 4 / 5)

  -- Statement of the second proof problem
  theorem cos_2alpha_plus_pi_over_3_value : cos (2 * α + π / 3) = (24 * real.sqrt 3 - 7) / 50 := by
   sorry
end

end sin_alpha_value_cos_2alpha_plus_pi_over_3_value_l396_396530


namespace min_sum_of_p_q_r_l396_396725

/-- Proof the minimum value of p + q + r is 9 given the conditions
  that the pairwise gcds of five positive integers include 2, 3,
  4, 5, 6, 7, 8, p, q, and r in some order. -/
theorem min_sum_of_p_q_r :
  ∃ (p q r : ℕ), (∀ (a b c d e : ℕ),
    {gcd a b, gcd a c, gcd a d, gcd a e, gcd b c, gcd b d, gcd b e, gcd c d, gcd c e, gcd d e} = {2, 3, 4, 5, 6, 7, 8, p, q, r}) →
    p + q + r = 9 :=
by
  sorry

end min_sum_of_p_q_r_l396_396725


namespace consecutive_numbers_even_count_l396_396683

def percentage_of_evens (n : ℕ) := 13 * n / 25

theorem consecutive_numbers_even_count (n : ℕ) (h1 : 52 * n / 100 = percentage_of_evens n) :
    percentage_of_evens 25 = 13 := 
begin
    sorry
end

end consecutive_numbers_even_count_l396_396683


namespace ratio_n_over_p_l396_396469

theorem ratio_n_over_p (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0) 
  (h4 : ∃ r1 r2 : ℝ, r1 + r2 = -p ∧ r1 * r2 = m ∧ 3 * r1 + 3 * r2 = -m ∧ 9 * r1 * r2 = n) :
  n / p = -27 := 
by
  sorry

end ratio_n_over_p_l396_396469


namespace number_of_even_factors_l396_396591

def is_even_factor (m : ℕ) (n : ℕ) : Prop :=
  m ∣ n ∧ m % 2 = 0

theorem number_of_even_factors (a b c : ℕ) (p q r : ℕ) 
  (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 2)
  (n : ℕ)
  (h₄ : n = 2 ^ a * 3 ^ b * 5 ^ c) :
  ∃ k : ℕ, k = 27 ∧ (∀ m, m ∣ n → m % 2 = 0 → m ∈ {1, 2, ..., 27}) :=
sorry

end number_of_even_factors_l396_396591


namespace graph_properties_l396_396557

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

theorem graph_properties :
  (∀ x, x ≠ 1 → f x = (x-2)*(x-3)/(x-1)) ∧
  (∃ x, f x = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε) ∧
  ((∀ ε > 0, ∃ M > 0, ∀ x > M, f x > ε) ∧ (∀ ε > 0, ∃ M < 0, ∀ x < M, f x < -ε)) := sorry

end graph_properties_l396_396557


namespace find_y_when_z_is_three_l396_396697

theorem find_y_when_z_is_three
  (k : ℝ) (y z : ℝ)
  (h1 : y = 3)
  (h2 : z = 1)
  (h3 : y ^ 4 * z ^ 2 = k)
  (hc : z = 3) :
  y ^ 4 = 9 :=
sorry

end find_y_when_z_is_three_l396_396697


namespace y_days_to_complete_work_l396_396768

-- Define the work rates and work durations.
variable (x_work_rate : ℕ → ℝ) -- work rate of x given the number of days to complete the work
variable (y_work_rate : ℕ → ℝ) -- work rate of y given the number of days to complete the work
variable (x_days : ℕ := 40) -- days x takes to complete the work
variable (y_days : ℕ) -- days y will take to complete the work
variable (x_partial_days : ℕ := 8) -- days x works initially
variable (y_partial_days : ℕ := 28) -- days y works to complete the remaining work
variable (unit_work : ℝ := 1) -- total amount of work considered as 1 unit

-- Define the work rate functions
def x_work_rate (d : ℕ) : ℝ := unit_work / d
def y_work_rate (d : ℕ) : ℝ := unit_work / d

-- Define the total work done by x in the initial period
def x_work_done (r : ℕ → ℝ) (d : ℕ) (days : ℕ) : ℝ := r(d) * days

-- Define the remaining work after x's initial work
def remaining_work (total : ℝ) (partial : ℝ) : ℝ := total - partial

-- Define the work rate of y for the remaining work
def y_calculated_work_rate (remaining : ℝ) (days : ℕ) : ℝ := remaining / days

-- Define the proof statement
theorem y_days_to_complete_work :
  let x_rate := x_work_rate(x_days)
  let x_work := x_work_done(x_work_rate, x_days, x_partial_days)
  let remaining := remaining_work(unit_work, x_work)
  let y_rate := y_calculated_work_rate(remaining, y_partial_days)
  y_days = unit_work / y_rate :=
by
  sorry

end y_days_to_complete_work_l396_396768


namespace largest_sequence_sum_45_l396_396332

theorem largest_sequence_sum_45 
  (S: ℕ → ℕ)
  (h_S: ∀ n, S n = n * (n + 1) / 2)
  (h_sum: ∃ m: ℕ, S m = 45):
  (∃ k: ℕ, k ≤ 9 ∧ S k = 45) ∧ (∀ m: ℕ, S m ≤ 45 → m ≤ 9) :=
by
  sorry

end largest_sequence_sum_45_l396_396332


namespace value_of_m_l396_396217

def pyramid_base : list Int := [6, 12, 10]

def pyramid_layer (xs : list Int) : list Int :=
  match xs with
  | (x :: y :: rest) => (x + y) :: pyramid_layer (y :: rest)
  | _ => []

def number_wall_pyramid (base : list Int) : list (list Int) :=
  match base with
  | [] => []
  | [x] => [[x]]
  | _ => base :: number_wall_pyramid (pyramid_layer base)

def top_block (pyramid : list (list Int)) : Int :=
  match pyramid with
  | [] => 0
  | x :: xs => match x.reverse.reverse with
                | [] => 0
                | y :: _ => y

theorem value_of_m : top_block (number_wall_pyramid (6 :: -28 :: 12 :: 10 :: [])) = 36 := by
  sorry

end value_of_m_l396_396217


namespace distance_from_point_to_line_is_correct_l396_396092

-- Given points in 3D space
def point_a : EuclideanSpace ℝ (Fin 3) := ![2, -3, 4]
def point_b : EuclideanSpace ℝ (Fin 3) := ![0, 3, -1]
def point_c : EuclideanSpace ℝ (Fin 3) := ![3, 0, 2]

-- Direction vector of the line passing through point_b and point_c
def direction_vector : EuclideanSpace ℝ (Fin 3) := point_c - point_b

-- Calculate the distance from point_a to the line defined by points point_b and point_c
noncomputable def distance_point_to_line (a b c : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let v := c - b in
  let t := -(dot_product (a - point_b) v) / (norm_sq v) in
  norm (point_b + t • v - a)

-- Expected result
def expected_distance : ℝ := Real.sqrt (1097 / 81)

-- The main statement:
theorem distance_from_point_to_line_is_correct :
  distance_point_to_line point_a point_b point_c = expected_distance := by
  sorry

end distance_from_point_to_line_is_correct_l396_396092


namespace range_of_m_l396_396169

theorem range_of_m (m x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : (1 - 2 * m) / x1 < (1 - 2 * m) / x2) : m < 1 / 2 :=
sorry

end range_of_m_l396_396169


namespace hyperbola_chord_length_l396_396151

open Real

theorem hyperbola_chord_length
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_eccentricity : sqrt (1 + (b ^ 2) / (a ^ 2)) = sqrt 5)
  (center_x center_y r : ℝ)
  (h_circle : ((x - center_x) ^ 2 + (y - center_y) ^ 2 = 1))
  (asymptote_slope : ℝ)
  (intersect_points : (Set.Point : ℝ × ℝ))
  (A B : set (ℝ × ℝ)) :
  (abs (chord_length A B) = (4 * sqrt 5) / 5) := 
sorry

end hyperbola_chord_length_l396_396151


namespace numberOfComplexOrderedPairs_l396_396101

noncomputable def complexOrderedPairs : Set (ℂ × ℂ) :=
  {p | let a := p.1, b := p.2 in a^4 * b^6 = 1 ∧ a^8 * b^3 = 1}

theorem numberOfComplexOrderedPairs : Fintype.card complexOrderedPairs = 12 :=
by
  sorry

end numberOfComplexOrderedPairs_l396_396101


namespace intersection_M_N_l396_396906

def M := {x : ℝ | -2 < x ∧ x < 3}
def N : set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} :=
by
-- proof omitted
sorry

end intersection_M_N_l396_396906


namespace ratio_of_inscribed_to_circumscribed_l396_396886

theorem ratio_of_inscribed_to_circumscribed (a : ℝ) :
  let r' := a * Real.sqrt 6 / 12
  let R' := a * Real.sqrt 6 / 4
  r' / R' = 1 / 3 := by
  sorry

end ratio_of_inscribed_to_circumscribed_l396_396886


namespace binomial_7_2_l396_396066

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396066


namespace find_sam_age_l396_396292

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l396_396292


namespace a_100_equals_1226_l396_396626

noncomputable def a_series : ℕ → ℕ
| 0     := 1
| (n+1) := if (n + 1) % 2 = 0 then a_series n + (-1)^(n.div2) else a_series (n-1) + n.div2

theorem a_100_equals_1226 : a_series 99 = 1226 := by
  sorry

end a_100_equals_1226_l396_396626


namespace binomial_7_2_l396_396037

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396037


namespace Alex_shirt_count_l396_396428

variables (Ben_shirts Joe_shirts Alex_shirts : ℕ)

-- Conditions from the problem
def condition1 := Ben_shirts = 15
def condition2 := Ben_shirts = Joe_shirts + 8
def condition3 := Joe_shirts = Alex_shirts + 3

-- Statement to prove
theorem Alex_shirt_count : condition1 ∧ condition2 ∧ condition3 → Alex_shirts = 4 :=
by
  intros h
  sorry

end Alex_shirt_count_l396_396428


namespace train_usual_time_l396_396363

-- Variables for train's usual speed and usual time.
variable (S T : ℝ)

-- Definitions corresponding to conditions.
-- Train's new speed is 5/6 of its usual speed.
def new_speed := (5 / 6) * S

-- Let T' be the time the train takes when it is 10 minutes late.
def T' := T + 10

-- Establish the relationship between usual time (T) and late time (T') due to speed change.
def time_relation := T' = (6 / 5) * T

-- The goal is to prove that the usual time T is 60 minutes.
theorem train_usual_time (h1 : T' = T + 10) (h2 : T' = (6 / 5) * T) : T = 60 := by
  sorry

end train_usual_time_l396_396363


namespace find_a_l396_396157

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) 
  (h₂ : (1 : ℝ) + Real.log 1 / Real.log a) = 1
  (h₃ : ∀ x, 1 + Real.log x / Real.log a =
    ((1 / (x * Real.log a)) * (x - 1) + 1) ∧ 
    (1 / Real.log a ≠ 0) → 
    ((0 : ℝ) = 0) - 1 /(Real.log a)* (-1)) :
  a = Real.exp 1 := sorry

end find_a_l396_396157


namespace starting_player_wins_by_following_sequence_l396_396362

-- Defining the game state and critical numbers
def game_state (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 100
def critical_numbers : list ℕ := [1, 12, 23, 34, 45, 56, 67, 78, 89, 100]

-- Defining the main goal: the starting player wins by following the sequence to 100
theorem starting_player_wins_by_following_sequence :
  ∀ current_sum : ℕ,
    game_state current_sum →
    (∃ next_sum : ℕ, next_sum ∈ critical_numbers ∧ next_sum ≤ 100) →
    ∃ win : ℕ, win = 100 :=
by
  intros current_sum h1 h2
  -- The proof strategy would go here
  sorry

end starting_player_wins_by_following_sequence_l396_396362


namespace large_square_area_l396_396444

variable {s S : ℝ} -- side lengths of the small and large squares

def squarePerimeter (side : ℝ) : ℝ := 4 * side
def squareArea (side : ℝ) : ℝ := side * side

-- Conditions
def perimeter_condition : Prop := squarePerimeter S = squarePerimeter s + 80
def shaded_area_condition : Prop := squareArea S - squareArea s = 880

-- Theorem statement
theorem large_square_area :
  (∃ s S : ℝ, perimeter_condition ∧ shaded_area_condition) →
  squareArea S = 1024 :=
sorry

end large_square_area_l396_396444


namespace max_valid_subset_cardinality_l396_396905

def set_S : Finset ℕ := Finset.range 1998 \ {0}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → (x + y) % 117 ≠ 0

theorem max_valid_subset_cardinality :
  ∃ (A : Finset ℕ), is_valid_subset A ∧ 995 = A.card :=
sorry

end max_valid_subset_cardinality_l396_396905


namespace canoe_total_weight_l396_396277

def total_people (num_people : ℕ) (dog_factor : ℚ) : ℚ := dog_factor * num_people

def total_person_weight (num_people : ℕ) (weight_per_person : ℕ) : ℕ := num_people * weight_per_person

def dog_weight (person_weight : ℕ) (dog_ratio : ℚ) : ℚ := dog_ratio * person_weight

def total_canoe_weight (person_weight : ℕ) (dog_weight : ℚ) : ℚ := person_weight + dog_weight

theorem canoe_total_weight :
  let
    people_wt := total_person_weight 4 140 -- total weight of 4 people
    dog_wt := dog_weight 140 (1 / 4 : ℚ) -- weight of the dog
    total_wt := total_canoe_weight people_wt dog_wt -- total weight in the canoe
  in total_wt = 595 := by
sorry

end canoe_total_weight_l396_396277


namespace hyperbola_condition_l396_396214

variables (a b : ℝ)
def e1 : (ℝ × ℝ) := (2, 1)
def e2 : (ℝ × ℝ) := (2, -1)

theorem hyperbola_condition (h1 : e1 = (2, 1)) (h2 : e2 = (2, -1)) (p : ℝ × ℝ)
  (h3 : p = (2 * a + 2 * b, a - b)) :
  4 * a * b = 1 :=
sorry

end hyperbola_condition_l396_396214


namespace min_score_guarantees_payoff_l396_396385

-- Defining the probability of a single roll being a six
def prob_single_six : ℚ := 1 / 6 

-- Defining the event of rolling two sixes independently
def prob_two_sixes : ℚ := prob_single_six * prob_single_six

-- Defining the score of two die rolls summing up to 12
def is_score_twelve (a b : ℕ) : Prop := a + b = 12

-- Proving the probability of Jim scoring 12 in two rolls guarantees some monetary payoff.
theorem min_score_guarantees_payoff :
  (prob_two_sixes = 1/36) :=
by
  sorry

end min_score_guarantees_payoff_l396_396385


namespace incorrect_statements_l396_396758

-- Define basic properties for lines and their equations.
def point_slope_form (y y1 x x1 k : ℝ) : Prop := (y - y1) = k * (x - x1)
def intercept_form (x y a b : ℝ) : Prop := x / a + y / b = 1
def distance_to_origin_on_y_axis (k b : ℝ) : ℝ := abs b
def slope_intercept_form (y m x c : ℝ) : Prop := y = m * x + c

-- The conditions specified in the problem.
variables (A B C D : Prop)
  (hA : A ↔ ∀ (y y1 x x1 k : ℝ), ¬point_slope_form y y1 x x1 k)
  (hB : B ↔ ∀ (x y a b : ℝ), intercept_form x y a b)
  (hC : C ↔ ∀ (k b : ℝ), distance_to_origin_on_y_axis k b = abs b)
  (hD : D ↔ ∀ (y m x c : ℝ), slope_intercept_form y m x c)

theorem incorrect_statements : ¬ B ∧ ¬ C ∧ ¬ D :=
by
  -- Intermediate steps would be to show each statement B, C, and D are false.
  sorry

end incorrect_statements_l396_396758


namespace arithmetic_geometric_means_l396_396526

theorem arithmetic_geometric_means (a b : ℝ) (h1 : 2 * a = 1 + 2) (h2 : b^2 = (-1) * (-16)) : a * b = 6 ∨ a * b = -6 :=
by
  sorry

end arithmetic_geometric_means_l396_396526


namespace second_sample_number_l396_396604

theorem second_sample_number (class_size sample_size initial : ℕ) (h_class : class_size = 60) (h_sample : sample_size = 6) (h_initial : initial = 4) :
  (initial + (class_size / sample_size) = 14) :=
by
  have h_inter : class_size / sample_size = 10 := by
    rw [h_class, h_sample]
    exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

  rw [h_initial, h_inter]
  norm_num

end second_sample_number_l396_396604


namespace integer_powers_of_reciprocal_sum_l396_396245

variable (x: ℝ)

theorem integer_powers_of_reciprocal_sum (hx : x ≠ 0) (hx_int : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ k : ℤ, x^n + 1/x^n = k :=
by
  sorry

end integer_powers_of_reciprocal_sum_l396_396245


namespace price_increase_and_decrease_l396_396797

theorem price_increase_and_decrease (P : ℝ) (x : ℝ) 
  (h1 : 0 < P) 
  (h2 : (P * (1 - (x / 100) ^ 2)) = 0.81 * P) : 
  abs (x - 44) < 1 :=
by
  sorry

end price_increase_and_decrease_l396_396797


namespace range_of_slope_on_circle_l396_396517

theorem range_of_slope_on_circle (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (k : set ℝ), k = (λ x y, x y / (x + 2)) ⟨[-(real.sqrt 3) / 3, (real.sqrt 3) / 3]⟩ :=
sorry

end range_of_slope_on_circle_l396_396517


namespace correct_statements_l396_396894

theorem correct_statements (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x, a^x ∈ set.univ) ∧ (∀ x, x^2 ∈ set.univ) ∧ 
  ¬(function.inverse (λ x : ℝ, 2^x) (λ x : ℝ, log 3 x)) ∧
  (∀ x, (3^|x|) ∈ set.Ici 1) :=
by
  admit

end correct_statements_l396_396894


namespace part_I_part_II_l396_396172

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (cos x, sin x)

def norm_sq (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem part_I (x : ℝ) (h : x ∈ Icc 0 (π / 2)) (h_ab: norm_sq (a x) = norm_sq (b x)) : x = π / 6 :=
by
  sorry

theorem part_II : ∃ x ∈ Icc 0 (π / 2), f x = 3 / 2 :=
by
  sorry

end part_I_part_II_l396_396172


namespace sequence_general_terms_l396_396861

-- Define the function f(x)
def f (x : Real) : Real := 2 * sin (π / 3 * x + π / 6)

-- Define the set M
def M : Set Real := {x | abs (f x) = 2 ∧ x > 0}

-- Define the sequence a_n
def a (n : ℕ) := 3 * n - 2

-- Define the initial value of sequence b_n
def b₁ : ℕ := 1

-- Define the recursive sequence b_n
def b : ℕ → ℕ
| 1     := b₁
| (n+1) := b n + a (2^n)

-- The theorem statement proving the general term of sequences a_n and b_n
theorem sequence_general_terms (n : ℕ) (hn : n > 0) :
  a n = 3 * n - 2 ∧ b n = 3 * 2^n - 2 * n - 3 :=
by
  sorry

end sequence_general_terms_l396_396861


namespace ratio_problem_l396_396943

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l396_396943


namespace casey_pumping_minutes_l396_396455

theorem casey_pumping_minutes :
  let pump_rate := 3
  let corn_rows := 4
  let corn_plants_per_row := 15
  let water_needed_per_corn_plant := 0.5
  let num_pigs := 10
  let water_needed_per_pig := 4
  let num_ducks := 20
  let water_needed_per_duck := 0.25
  let total_water_needed := (corn_rows * corn_plants_per_row * water_needed_per_corn_plant) +
                            (num_pigs * water_needed_per_pig) +
                            (num_ducks * water_needed_per_duck)
  let minutes_needed := total_water_needed / pump_rate
  in minutes_needed = 25 :=
by 
  sorry

end casey_pumping_minutes_l396_396455


namespace sphere_radius_l396_396306

theorem sphere_radius (r_A r_B : ℝ) (h₁ : r_A = 40) (h₂ : (4 * π * r_A^2) / (4 * π * r_B^2) = 16) : r_B = 20 :=
  sorry

end sphere_radius_l396_396306


namespace ellipse_parameters_correct_l396_396441

noncomputable theory
open_locale real

structure EllipseParameters where
  a b h k : ℝ

def ellipse_foci_and_point (f1 f2 : ℝ × ℝ) (pass_point : ℝ × ℝ) : EllipseParameters :=
  let d1 := real.sqrt ((pass_point.1 - f1.1) ^ 2 + (pass_point.2 - f1.2) ^ 2)
  let d2 := real.sqrt ((pass_point.1 - f2.1) ^ 2 + (pass_point.2 - f2.2) ^ 2)
  let major_axis_length := d1 + d2

  let c := real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) / 2
  let a := major_axis_length / 2
  let b := real.sqrt (a^2 - c^2)

  let h := (f1.1 + f2.1) / 2
  let k := (f1.2 + f2.2) / 2
  EllipseParameters.mk a b h k

theorem ellipse_parameters_correct :
  ellipse_foci_and_point (1, 1) (7, 1) (0, 8) = EllipseParameters.mk (6 * real.sqrt 2) (3 * real.sqrt 7) 4 1 :=
  sorry

end ellipse_parameters_correct_l396_396441


namespace restore_original_text_l396_396772

def russian_alphabet : List Char := ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

def received_words : List String := ["ГЪЙ", "АЭЁ", "БПРК", "ЕЖЩЮ", "НМЬЧ", "СЫЛЗ", "ШДУ", "ЦХОТ", "ЯФВИ"]

-- Function to check if a given character can replace another character based on position constraint
def can_replace (original : Char) (replacement : Char) (alphabet : List Char) : Bool :=
  let original_idx := alphabet.indexOf original
  let replacement_idx := alphabet.indexOf replacement
  abs (original_idx - replacement_idx) ≤ 2

-- Main theorem statement
theorem restore_original_text :
  ∃ words : List String, words = ["БЫК", "ВЯЗ", "ГНОЙ", "ДИЧЬ", "ПЛЮЩ", "СЪЁМ", "ЦЕХ", "ШУРФ", "ЭТАЖ"] ∧
  ∀ (received : String), received ∈ received_words →
  ∃ (original : String), original ∈ words ∧ 
  (∀ (i : ℕ), i < original.length → i < received.length → 
    can_replace (original.get i) (received.get i) russian_alphabet) :=
sorry

end restore_original_text_l396_396772


namespace factorize_expression_l396_396085

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l396_396085


namespace complement_A_U_l396_396262

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := U \ A

-- Theorem: The complement of A with respect to U is {2, 4}
theorem complement_A_U : C_U_A = {2, 4} := by
  sorry

end complement_A_U_l396_396262


namespace points_on_the_same_sphere_l396_396864

theorem points_on_the_same_sphere 
    (n : ℕ)
    (h1 : n ≥ 5) 
    (points : Fin n → ℝ × ℝ × ℝ)
    (h2 : ∀ A B C D E, 
            ∃! W : point ∧ 
            (W ∈ sphere A B C D)
            → (W ∈ (sphere A B C D) ∨ (W in_sphere_or_inside sphere A B C D points))
            ∧ (coplanar F G H I. F G H I ∈ points ᶻT)  :=
    (sphere_through_4 : ∀ s : sphere points,
        all_points_on_same_sphere A B C D n points _, by {
  sorry

end points_on_the_same_sphere_l396_396864


namespace cube_expression_l396_396586

theorem cube_expression {x : ℝ} (h : sqrt (x - 3) = 3) : (x - 3)^3 = 729 :=
by
  -- Proof steps omitted
  sorry

end cube_expression_l396_396586


namespace largest_of_three_l396_396755

theorem largest_of_three (a b c : ℝ) (h₁ : a = 43.23) (h₂ : b = 2/5) (h₃ : c = 21.23) :
  max (max a b) c = a :=
by
  sorry

end largest_of_three_l396_396755


namespace chess_tournament_schedule_l396_396747

theorem chess_tournament_schedule :
  let num_players := 3
  let games_per_match := 3
  let total_games := num_players * num_players * games_per_match
  let rounds := total_games / 3
  let schedules := Nat.factorial rounds / (Nat.factorial games_per_match * Nat.factorial games_per_match * Nat.factorial games_per_match)
  rounds = 9 ∧ schedules = 1680
:=
by
  let num_players := 3
  let games_per_match := 3
  let total_games := num_players * num_players * games_per_match
  let rounds := total_games / 3
  let schedules := Nat.factorial rounds / (Nat.factorial games_per_match * Nat.factorial games_per_match * Nat.factorial games_per_match)
  have h1 : rounds = 9 := by norm_num
  have h2 : schedules = 1680 := by norm_num
  exact ⟨h1, h2⟩

end chess_tournament_schedule_l396_396747


namespace prime_and_congruence_l396_396644

-- Given conditions
variables (n d : ℤ) (hn : n > 1) (hd : d > 1) (h_gcd : Int.gcd n (Nat.factorial d) = 1)

-- Prove the statement
theorem prime_and_congruence (n d : ℤ) (hn : n > 1) (hd : d > 1) 
  (h_gcd : Int.gcd n (Nat.factorial d) = 1) : 
  (Nat.Prime n ∧ Nat.Prime (n + d)) ↔ (Nat.factorial d * d * (Nat.factorial (n - 1) + 1) + n * (Nat.factorial d - 1) ≡ 0 [ZMOD (n * (n + d))) := 
by sorry

end prime_and_congruence_l396_396644


namespace quincy_age_l396_396641

variable (Kiarra Bea Job Figaro Harry Charles Vivian Quincy : ℕ)

theorem quincy_age (hK : Kiarra = 30)
  (h1 : Kiarra = 2 * Bea)
  (h2 : Job = 3 * Bea)
  (h3 : Figaro = Job + 7)
  (h4 : Harry = Figaro / 2)
  (h5 : Charles = Harry - 4)
  (h6 : Vivian = Charles / 3)
  (h7 : Quincy = (Job + Figaro) / 2) :
  Quincy = 48.5 := sorry

end quincy_age_l396_396641


namespace range_of_x_l396_396900

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3*x - 2)) : 1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l396_396900


namespace flowers_sold_l396_396116

theorem flowers_sold (lilacs roses gardenias : ℕ) 
  (h1 : lilacs = 10)
  (h2 : roses = 3 * lilacs)
  (h3 : gardenias = lilacs / 2) : 
  lilacs + roses + gardenias = 45 :=
by
  sorry

end flowers_sold_l396_396116


namespace combinatorial_identity_l396_396142

theorem combinatorial_identity :
  (nat.choose 22 5) = 26334 := by
  have h1 : nat.choose 20 3 = 1140 := sorry
  have h2 : nat.choose 20 4 = 4845 := sorry
  have h3 : nat.choose 20 5 = 15504 := sorry
  sorry

end combinatorial_identity_l396_396142


namespace candidate_defeat_margin_l396_396445

theorem candidate_defeat_margin
  (total_votes : ℕ)
  (invalid_votes : ℕ)
  (valid_votes : ℕ)
  (defeated_votes : ℕ)
  (winning_votes : ℕ)
  (defeat_margin : ℕ)
  (H1 : total_votes = 90_083)
  (H2 : invalid_votes = 83)
  (H3 : valid_votes = total_votes - invalid_votes)
  (H4 : defeated_votes = (45 * valid_votes) / 100)
  (H5 : winning_votes = (55 * valid_votes) / 100)
  (H6 : defeat_margin = winning_votes - defeated_votes)
  : defeat_margin = 9_000 := 
  by 
    sorry

end candidate_defeat_margin_l396_396445


namespace binom_7_2_eq_21_l396_396050

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396050


namespace general_term_sum_of_b_l396_396215

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℝ := n * (n + 1) / 2
def a (n : ℕ) : ℝ := if n = 0 then 0 else S n - S (n - 1)

-- Prove the general term formula for the sequence a_n is n
theorem general_term (n : ℕ) : a n = n := sorry

-- Define the sequence b_n
def b (n : ℕ) : ℝ := a n / 2^n

-- Define the sum of the first n terms T_n for the sequence b_n
def T (n : ℕ) : ℝ := (finset.range n).sum (λ k, b (k + 1))

-- Prove the sum of the first n terms T_n, for the sequence b_n
theorem sum_of_b (n : ℕ) : T n = 2 - (2 + n) / 2^n := sorry

end general_term_sum_of_b_l396_396215


namespace subset_range_a_l396_396512

def setA : Set ℝ := { x | (x^2 - 4 * x + 3) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (2^(1 - x) + a) ≤ 0 ∧ (x^2 - 2*(a + 7)*x + 5) ≤ 0 }

theorem subset_range_a (a : ℝ) : setA ⊆ setB a ↔ -4 ≤ a ∧ a ≤ -1 := 
  sorry

end subset_range_a_l396_396512


namespace hyperbola_chord_length_l396_396150

open Real

theorem hyperbola_chord_length
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_eccentricity : sqrt (1 + (b ^ 2) / (a ^ 2)) = sqrt 5)
  (center_x center_y r : ℝ)
  (h_circle : ((x - center_x) ^ 2 + (y - center_y) ^ 2 = 1))
  (asymptote_slope : ℝ)
  (intersect_points : (Set.Point : ℝ × ℝ))
  (A B : set (ℝ × ℝ)) :
  (abs (chord_length A B) = (4 * sqrt 5) / 5) := 
sorry

end hyperbola_chord_length_l396_396150


namespace binomial_7_2_l396_396067

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l396_396067


namespace ratio_problem_l396_396940

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l396_396940


namespace solution_l396_396178

noncomputable def find_x (S : Real) (hS : S = 1 + 3 * x + 5 * x ^ 2 + 7 * x ^ 3 + 9 * x ^ 4 + ...) := 
  x

theorem solution (x : Real) (h : 1 + 3 * x + 5 * x ^ 2 + 7 * x ^ 3 + 9 * x ^ 4 + ... = 16) : 
  x = 5 / 8 :=
sorry

end solution_l396_396178


namespace distinct_positive_factors_count_l396_396766

-- Define x and y as odd prime numbers
def isPrime (n : ℕ) : Prop := ∀ m ∣ n, m = 1 ∨ m = n
def isOdd (n : ℕ) : Prop := n % 2 = 1
def distinctOddPrimes (x y : ℕ) : Prop := isPrime x ∧ isPrime y ∧ isOdd x ∧ isOdd y ∧ x < y

-- Define 2xy
def two_x_y (x y : ℕ) : ℕ := 2 * x * y

-- Define the theorem statement
theorem distinct_positive_factors_count (x y : ℕ) (h : distinctOddPrimes x y) : 
  (2 * x * y).nat_factors.length = 8 := 
sorry

end distinct_positive_factors_count_l396_396766


namespace total_weight_of_canoe_l396_396274

-- Define conditions as constants
constant total_people_capacity : ℕ := 6
constant proportion_with_dog : ℝ := 2/3
constant weight_per_person : ℝ := 140
constant dog_weight_proportion : ℝ := 1/4

-- Define the theorem to prove
theorem total_weight_of_canoe : 
  let number_of_people_with_dog := proportion_with_dog * total_people_capacity
  let total_people_weight := number_of_people_with_dog * weight_per_person
  let dog_weight := dog_weight_proportion * weight_per_person
  total_people_weight + dog_weight = 595 := 
by 
  sorry

end total_weight_of_canoe_l396_396274


namespace calculation_l396_396024

theorem calculation :
  2 * Real.sin (Float.pi / 3) + (1 / 2) ^ (-2: ℤ) + Real.abs (2 - Real.sqrt 3) - Real.sqrt 9 = 3 := by
  sorry

end calculation_l396_396024


namespace count_k_f_divisible_l396_396130
open Nat

def is_primitive_root (r p : ℕ) : Prop :=
  (1 < r < p) ∧ (∀ k : ℕ, k < p → r^k % p ≠ 1) ∧ (r^(p-1) % p = 1)

def f (k p : ℕ) : ℕ :=
  ∑ r in range (p-1), if is_primitive_root r p then r^k % p else 0

def is_f_divisible_by_p (k p : ℕ) : Prop :=
  f k p % p = 0

theorem count_k_f_divisible :
  ∀ p, p = 73 →
  ∃ n, n = 1841 ∧
  ∃ k_values, (∀ k ∈ k_values, k < 2015 ∧ is_f_divisible_by_p k p) ∧
  length k_values = n :=
by 
  intros p hp
  use 1841
  sorry

end count_k_f_divisible_l396_396130


namespace number_of_subsets_of_M_l396_396333

def M : Set ℕ := {1, 2, 3}

theorem number_of_subsets_of_M : 
  ∃ n : ℕ, M.finite ∧ M.card = n ∧ (2^n = 8) :=
by
  sorry

end number_of_subsets_of_M_l396_396333


namespace combination_7_2_l396_396029

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l396_396029


namespace binomial_22_5_computation_l396_396144

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end binomial_22_5_computation_l396_396144


namespace brownies_in_pan_l396_396779

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan_l396_396779


namespace solve_problem_l396_396909

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def valid_set (a b c : ℕ) : Prop :=
  (a + b + c = 22) ∧ distinct_digits a b c

def is_valid_solution (s : set (set ℕ)) : Prop :=
  s = {{9, 5, 8}, {9, 6, 7}}

theorem solve_problem (a b c : ℕ) (h1 : valid_set a b c) : {a, b, c} ∈ {{9, 5, 8}, {9, 6, 7}} :=
  sorry

end solve_problem_l396_396909


namespace binom_7_2_eq_21_l396_396051

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l396_396051


namespace question1_question2_l396_396870

-- Part (1)
theorem question1 (a b c : ℝ):
  (a ≠ 0) → 
  (M = { x | -1/2 < x ∧ x < 2 }) → 
  (f 0 = 2) → 
  (f = (λ x, -2 * x^2 + 3 * x + 2)) 
:=
sorry

-- Part (2)
theorem question2 (a b c : ℝ) (m : ℝ):
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (M = (-∞, m) ∪ (m, ∞)) →
  (m < 0) →
  (min₂ ((b / c) - 2 * m) = 4)
:=
sorry

end question1_question2_l396_396870


namespace product_even_of_permuted_integers_l396_396251

theorem product_even_of_permuted_integers :
  ∀ (a b : Fin 25 → ℤ),
    (∀ i j, i ≠ j → a i ≠ a j ∧ b i ≠ b j) →  -- Ensuring distinctness is not necessary as they are permutations
    (∃ σ : Fin 25 → Fin 25, ∀ i, b (σ i) = a i) →
    Even (∏ i, a i - b i) :=
by
  intros a b hdistinct hperm
  sorry

end product_even_of_permuted_integers_l396_396251


namespace isosceles_triangle_and_sin_cos_range_l396_396197

theorem isosceles_triangle_and_sin_cos_range 
  (A B C : ℝ) (a b c : ℝ) 
  (hA_pos : 0 < A) (hA_lt_pi_div_2 : A < π / 2) (h_triangle : a * Real.cos B = b * Real.cos A) :
  (A = B ∧
  ∃ x, x = Real.sin B + Real.cos (A + π / 6) ∧ (1 / 2 < x ∧ x ≤ 1)) :=
by
  sorry

end isosceles_triangle_and_sin_cos_range_l396_396197


namespace oil_tank_depth_l396_396400

theorem oil_tank_depth (L r A : ℝ) (h : ℝ) (L_pos : L = 8) (r_pos : r = 2) (A_pos : A = 16) :
  h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

end oil_tank_depth_l396_396400


namespace calculate_raised_beds_l396_396449

def num_planks_per_bed (height width length plank_width : ℕ) : ℕ :=
  let side_planks := (length / plank_width) * (height / (plank_width / 2))
  let end_planks := (2 * (width / plank_width))
  in side_planks + end_planks

def num_raised_beds (total_planks plank_per_bed : ℕ) : ℕ := total_planks / plank_per_bed

theorem calculate_raised_beds
  (height width length total_planks plank_width : ℕ)
  (h_height : height = 2)
  (h_width : width = 2)
  (h_length : length = 8)
  (h_total_planks : total_planks = 50)
  (h_plank_width : plank_width = 1) :
  num_raised_beds total_planks (num_planks_per_bed height width length plank_width) = 10 :=
by
  rw [h_height, h_width, h_length, h_total_planks, h_plank_width]
  simp [num_planks_per_bed, num_raised_beds]
  sorry

end calculate_raised_beds_l396_396449


namespace player_A_success_l396_396280

/-- Representation of the problem conditions --/
structure GameState where
  coins : ℕ
  boxes : ℕ
  n_coins : ℕ 
  n_boxes : ℕ 
  arrangement: ℕ → ℕ 
  (h_coins : coins ≥ 2012)
  (h_boxes : boxes = 2012)
  (h_initial_distribution : (∀ b, arrangement b ≥ 1))
  
/-- The main theorem for player A to ensure at least 1 coin in each box --/
theorem player_A_success (s : GameState) : 
  s.coins ≥ 4022 → (∀ b, s.arrangement b ≥ 1) :=
by
  sorry

end player_A_success_l396_396280


namespace all_edges_equal_l396_396134

variable (A B C D O_a O_b O_c O_d : Type)
variables [Tetrahedron A B C D]
variables [ExcircleCenter O_a B C D]
variables [ExcircleCenter O_b A C D]
variables [ExcircleCenter O_c A B D]
variables [ExcircleCenter O_d A B C]

theorem all_edges_equal (h1 : TrihedralAngleRight O_a B C D)
                         (h2 : TrihedralAngleRight O_b A C D)
                         (h3 : TrihedralAngleRight O_c A B D)
                         (h4 : TrihedralAngleRight O_d A B C) :
                         (AllEdgesEqual A B C D) := 
sorry

end all_edges_equal_l396_396134


namespace total_weight_of_envelopes_l396_396013

theorem total_weight_of_envelopes :
  (8.5 * 880 / 1000) = 7.48 :=
by
  sorry

end total_weight_of_envelopes_l396_396013


namespace find_sam_age_l396_396291

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l396_396291


namespace find_certain_number_l396_396592

-- Define the given operation a # b
def sOperation (a b : ℝ) : ℝ :=
  a * b - b + b^2

-- State the theorem to find the value of the certain number
theorem find_certain_number (x : ℝ) (h : sOperation 3 x = 48) : x = 6 :=
sorry

end find_certain_number_l396_396592


namespace max_area_convex_pentagon_l396_396397

theorem max_area_convex_pentagon 
  (A B C D E : ℂ)
  (hA : abs A = 1)
  (hB : abs B = 1)
  (hC : abs C = 1)
  (hD : abs D = 1)
  (hE : abs E = 1)
  (h_perp : (re C * re D - im C * im D) = 0):
  ∃ (A B C D E : ℂ), 
    (abs A = 1) ∧
    (abs B = 1) ∧
    (abs C = 1) ∧
    (abs D = 1) ∧
    (abs E = 1) ∧
    ((re A * re C - im A * im C) = 0) ∧
    area ABCDE = 1 + 3 * real.sqrt (3) / 4 :=
sorry

end max_area_convex_pentagon_l396_396397


namespace lite_soda_bottles_count_l396_396791

theorem lite_soda_bottles_count
  (total_bottles : ℕ)
  (regular_soda : ℕ)
  (diet_soda : ℕ)
  (h_total_bottles : total_bottles = 110)
  (h_regular_soda : regular_soda = 57)
  (h_diet_soda : diet_soda = 26) :
  total_bottles - (regular_soda + diet_soda) = 27 := by
  have h_step1 : regular_soda + diet_soda = 83, by
    rw [h_regular_soda, h_diet_soda]
    refl
  have h_step2 : total_bottles - 83 = 27, by
    rw [h_total_bottles]
    refl
  rw h_step1
  exact h_step2

end lite_soda_bottles_count_l396_396791


namespace find_b_eq_neg_half_l396_396999

def h (x : ℝ) : ℝ :=
  if x ≤ 1 then -x else 3 * x - 6

theorem find_b_eq_neg_half : 
  ∀ (b : ℝ), b < 0 → h (h (h (-0.5))) = h (h (h b)) → b = -0.5 :=
by
  sorry

end find_b_eq_neg_half_l396_396999


namespace arithmetic_seq_a11_l396_396646

theorem arithmetic_seq_a11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 21 = 105) : a 11 = 5 :=
sorry

end arithmetic_seq_a11_l396_396646


namespace binomial_7_2_l396_396059

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l396_396059
