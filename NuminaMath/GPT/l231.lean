import Mathlib

namespace pascal_triangle_fifth_number_l231_231749

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231749


namespace pascal_triangle_fifth_number_l231_231685

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231685


namespace pascal_fifth_number_l231_231746

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231746


namespace binomial_distribution_probability_l231_231061

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k
noncomputable def probability_of_X_eq_1 (n : ℕ) (p : ℚ) : ℚ := binomial n 1 * p * (1 - p)^(n - 1)

theorem binomial_distribution_probability
  (n : ℕ) (p : ℚ)
  (h1 : (n : ℚ) * p = 6)
  (h2 : (n : ℚ) * p * (1 - p) = 3) :
  probability_of_X_eq_1 n p = 3 * 2^(-10) :=
by
  sorry

end binomial_distribution_probability_l231_231061


namespace gcd_of_B_is_2_l231_231852

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231852


namespace pascal_fifteen_four_l231_231789

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231789


namespace sin_pi_minus_alpha_l231_231467

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi - α) = -1/3) : Real.sin α = -1/3 :=
sorry

end sin_pi_minus_alpha_l231_231467


namespace gcd_of_B_is_2_l231_231842

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231842


namespace custard_slice_price_l231_231098

noncomputable def price_of_custard_slice (x : ℝ) : Prop :=
  let pumpkin_slices := 4 * 8
  let pumpkin_revenue := 32 * 5
  let custard_slices := 5 * 6
  let total_revenue := 340
  in 160 + 30 * x = total_revenue

theorem custard_slice_price : price_of_custard_slice 6 :=
by
  let pumpkin_slices := 4 * 8
  let pumpkin_revenue := 32 * 5
  let custard_slices := 5 * 6
  let total_revenue := 340
  let x := 6
  show 160 + 30 * x = 340
  sorry

end custard_slice_price_l231_231098


namespace eval_f_a_plus_1_l231_231477

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the condition
axiom a : ℝ

-- State the theorem to be proven
theorem eval_f_a_plus_1 : f (a + 1) = a^2 + 2*a + 1 :=
by
  sorry

end eval_f_a_plus_1_l231_231477


namespace gcd_of_B_l231_231937

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231937


namespace equilateral_triangle_l231_231462

theorem equilateral_triangle 
  (a b c : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_inequality : ∑ cycle (λ x, 1/x * real.sqrt (1/x.succ + 1/x.succ.succ)) (1/a) ≥ 3/2 * real.sqrt(Π cycle (λ x, 1/x + 1/x.succ) (1/a)))
  : a = b ∧ b = c :=
sorry

end equilateral_triangle_l231_231462


namespace proof_problem_l231_231437

theorem proof_problem (x : ℝ) : (0 < x ∧ x < 5) → (x^2 - 5 * x < 0) ∧ (|x - 2| < 3) :=
by
  sorry

end proof_problem_l231_231437


namespace least_positive_integer_with_leading_six_and_fraction_property_l231_231215

theorem least_positive_integer_with_leading_six_and_fraction_property :
  ∃ (x : ℕ), (x % 10 ≠ 0) ∧ («leading_digit» x = 6) ∧ ((x - «leading_digit» x * 10^(x.digits.length - 1)) * 25 = x) ∧ x = 625 :=
by
  sorry

end least_positive_integer_with_leading_six_and_fraction_property_l231_231215


namespace gcd_of_B_is_two_l231_231887

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231887


namespace pascal_fifth_element_15th_row_l231_231715

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231715


namespace pascal_row_fifth_number_l231_231572

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231572


namespace time_to_distance_l231_231226

theorem time_to_distance (D : ℝ) (h1 : 0 < D) :
  let speed1 := 400
  let speed2 := 250
  let combined_speed := speed1 + speed2
  combined_speed = 650 →
  T = D / 650 :=
begin
  sorry,
end

end time_to_distance_l231_231226


namespace pascal_triangle_15_4_l231_231612

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231612


namespace pascal_triangle_fifth_number_l231_231643

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231643


namespace memorable_phone_numbers_count_l231_231352

def is_memorable (d1 d2 d3 d4 d5 d6 d7 : ℕ) : Prop := 
  (d1 = d4 ∧ d2 = d5 ∧ d3 = d6) ∨ 
  (d1 = d5 ∧ d2 = d6 ∧ d3 = d7)

theorem memorable_phone_numbers_count : 
  (∑ d1 in Finset.range 10, 
    ∑ d2 in Finset.range 10, 
      ∑ d3 in Finset.range 10, 
        ∑ d4 in Finset.range 10, 
          ∑ d5 in Finset.range 10,
            ∑ d6 in Finset.range 10, 
              ∑ d7 in Finset.range 10, 
                if is_memorable d1 d2 d3 d4 d5 d6 d7 then 1 else 0) = 19990 :=
begin
  sorry
end

end memorable_phone_numbers_count_l231_231352


namespace geometric_sequence_product_l231_231106

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 * a 11 = 8) :
  a 2 * a 8 = 4 :=
sorry

end geometric_sequence_product_l231_231106


namespace yearly_return_of_1500_investment_is_27_percent_l231_231306

-- Definitions based on conditions
def combined_yearly_return (x : ℝ) : Prop :=
  let investment1 := 500
  let investment2 := 1500
  let total_investment := investment1 + investment2
  let combined_return := 0.22 * total_investment
  let return_from_500 := 0.07 * investment1
  let return_from_1500 := combined_return - return_from_500
  x / 100 * investment2 = return_from_1500

-- Theorem statement to be proven
theorem yearly_return_of_1500_investment_is_27_percent : combined_yearly_return 27 :=
by sorry

end yearly_return_of_1500_investment_is_27_percent_l231_231306


namespace pascal_15_5th_number_l231_231776

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231776


namespace total_number_of_red_and_white_jelly_beans_in_fishbowl_l231_231822

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end total_number_of_red_and_white_jelly_beans_in_fishbowl_l231_231822


namespace greatest_two_digit_prod_12_l231_231287

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231287


namespace eccentricity_ellipse_l231_231191

-- Definitions directly from the problem statement
def x (θ : ℝ) : ℝ := 5 * Real.cos θ
def y (θ : ℝ) : ℝ := 4 * Real.sin θ

-- The mathematically equivalent proof problem in Lean 4
theorem eccentricity_ellipse (θ : ℝ) : 
  let a := 5; b := 4; c := Real.sqrt (a^2 - b^2) in
  c / a = 3 / 5 := 
by 
  sorry

end eccentricity_ellipse_l231_231191


namespace smallest_cube_sum_l231_231159

theorem smallest_cube_sum (n : ℕ) (n_cube : n = 6) (sum_cubes : n^3 = 1^3 + 6^3 + 5^3) : 
  216 = n^3 :=
by { have h : n^3 = 216 := by linarith, rw [h], exact sum_cubes }

end smallest_cube_sum_l231_231159


namespace pascal_triangle_15_4_l231_231615

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231615


namespace Pascal_triangle_fifth_number_l231_231664

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231664


namespace gcd_B_is_2_l231_231905

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231905


namespace proof_problem_l231_231980

variables (a b c : ℤ)
variables (h1 : a < 0) (h2 : b > 0) (h3 : c > 0) (h4 : b < c)

theorem proof_problem :
  (ac < bc) ∧ (ab < ac) ∧ (a + b < b + c) :=
sorry

end proof_problem_l231_231980


namespace simplification_problem_l231_231294

theorem simplification_problem :
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73 / 89 :=
  sorry

end simplification_problem_l231_231294


namespace f_one_f_inequality_l231_231469

noncomputable def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x, 0 < x → ∃ y, y = f x
axiom f_multiplicative : ∀ x y, 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom f_half : f (1/2) = 1
axiom f_decreasing : ∀ x y, 0 < x ∧ x < y → f x > f y

theorem f_one : f 1 = 0 :=
sorry

theorem f_inequality : ∀ x ∈ set.Icc (-1 : ℝ) 0, f (-x) + f (3 - x) ≥ -2 :=
sorry

end f_one_f_inequality_l231_231469


namespace pascal_row_fifth_number_l231_231582

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231582


namespace original_number_of_movies_l231_231343

/-- Suppose a movie buff owns movies on DVD, Blu-ray, and digital copies in a ratio of 7:2:1.
    After purchasing 5 more Blu-ray movies and 3 more digital copies, the ratio changes to 13:4:2.
    She owns movies on no other medium.
    Prove that the original number of movies in her library before the extra purchase was 390. -/
theorem original_number_of_movies (x : ℕ) (h1 : 7 * x != 0) 
  (h2 : 2 * x != 0) (h3 : x != 0)
  (h4 : 7 * x / (2 * x + 5) = 13 / 4)
  (h5 : 7 * x / (x + 3) = 13 / 2) : 10 * x = 390 :=
by
  sorry

end original_number_of_movies_l231_231343


namespace arithmetic_mean_l231_231447

theorem arithmetic_mean (n : ℕ) (hn : n > 1) : 
  let a := 2 * (1 + 1 / n) + (1 - 1 / n) + (n - 3) * 1 in
  a / n = 1 + 1 / (n ^ 2) :=
by 
  sorry

end arithmetic_mean_l231_231447


namespace dodecahedron_interior_diagonals_l231_231040

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231040


namespace fifth_number_in_pascals_triangle_l231_231590

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231590


namespace total_matches_round_robin_l231_231351

/-- A round-robin chess tournament is organized in two groups with different numbers of players. 
Group A consists of 6 players, and Group B consists of 5 players. 
Each player in each group plays every other player in the same group exactly once. 
Prove that the total number of matches is 25. -/
theorem total_matches_round_robin 
  (nA : ℕ) (nB : ℕ) 
  (hA : nA = 6) (hB : nB = 5) : 
  (nA * (nA - 1) / 2) + (nB * (nB - 1) / 2) = 25 := 
  by
    sorry

end total_matches_round_robin_l231_231351


namespace GCD_of_set_B_is_2_l231_231946

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231946


namespace parallel_proof_l231_231136

noncomputable theory
open EuclideanGeometry

variables (A B C H_A H_B H_C D E : Point)

def triangle (A B C : Point) : Prop :=
  ∃ (AB BC CA : Line), line_through AB A B ∧ line_through BC B C ∧ line_through CA C A

def is_altitude_foot (A B C H_A : Point) : Prop :=
  ∃ (AB BC : Line), 
    line_through AB A B ∧ line_through BC B C ∧ perpendicular AB (line_through H_A B C)

def is_projection (H_A D AB : Point) : Prop :=
  perpendicular (line_through H_A D) AB

def is_projection (H_A E BC : Point) : Prop :=
  perpendicular (line_through H_A E) BC

theorem parallel_proof 
  (h_triangle : triangle A B C)
  (h_H_A : is_altitude_foot A B C H_A)
  (h_H_B : is_altitude_foot B A C H_B)
  (h_H_C : is_altitude_foot C A B H_C)
  (h_proj_AB : is_projection H_A D (line_through A B))
  (h_proj_BC : is_projection H_A E (line_through B C)) :
  parallel (line_through D E) (line_through H_B H_C) :=
sorry

end parallel_proof_l231_231136


namespace find_y_l231_231552

theorem find_y (y : ℝ) (hy : 0 < y) 
  (h : (Real.sqrt (12 * y)) * (Real.sqrt (6 * y)) * (Real.sqrt (18 * y)) * (Real.sqrt (9 * y)) = 27) : 
  y = 1 / 2 := 
sorry

end find_y_l231_231552


namespace max_min_sum_2023_l231_231193

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + real.sqrt 3

theorem max_min_sum_2023 :
  let I := set.Icc (-2023 : ℝ) 2023 in
  (∀ M N : ℝ, 
    (M = set.image f I).max ∧ (N = set.image f I).min → 
    M + N = 2 * real.sqrt 3)
:= sorry

end max_min_sum_2023_l231_231193


namespace martha_initial_bottles_l231_231361

def initial_bottles_in_fridge (R : ℕ) : Prop :=
  R + 6 = 10 → R = 4

theorem martha_initial_bottles:
  ∃ R : ℕ, initial_bottles_in_fridge R :=
by
  use 4
  intro h
  rw [Nat.add_comm] at h
  exact Nat.add_right_cancel h
  sorry

end martha_initial_bottles_l231_231361


namespace infinite_integer_solutions_l231_231485

theorem infinite_integer_solutions (x y : ℤ) (h : x + y = 1) : ∃ᶠ x, ∃ y, x + y = 1 :=
by
  sorry

end infinite_integer_solutions_l231_231485


namespace greatest_two_digit_product_12_l231_231276

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231276


namespace Pascal_triangle_fifth_number_l231_231654

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231654


namespace greater_difference_validity_l231_231096

theorem greater_difference_validity {a b c d : ℕ} :
  (a + b ≠ 0) ∧ (c + d ≠ 0) →
  let diffA := abs ((a * d - b * c) / ((a + b) * (c + d))) in
  let diffC := abs ((a * (b + c) - c * (a + b)) / ((a + b) * (b + c))) in
  let diffB := abs ((a * (a + b) - c * (c + d)) / ((c + d) * (a + b))) in
  let diffD := abs ((a * (a + c) - b * (a + d)) / ((b + d) * (a + c))) in
  diffA > diffB ∧ diffA > diffC ∧ diffA > diffD :=
sorry

end greater_difference_validity_l231_231096


namespace greatest_common_divisor_of_B_l231_231916

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231916


namespace gcd_of_B_l231_231932

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231932


namespace multiples_of_6_or_8_under_201_not_both_l231_231523

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231523


namespace juniper_bones_ratio_l231_231826

noncomputable def juniper_initial_bones : ℕ := 4
noncomputable def juniper_final_bones : ℕ := 6
noncomputable def bones_stolen : ℕ := 2

def ratio_of_bones (initial_bones : ℕ) (final_bones : ℕ) (stolen_bones : ℕ) : ℕ :=
  let x := final_bones + stolen_bones - initial_bones
  let bones_after_given := initial_bones + x
  bones_after_given / initial_bones

theorem juniper_bones_ratio (initial_bones : ℕ) (final_bones : ℕ) (stolen_bones : ℕ) :
  initial_bones = 4 → final_bones = 6 → stolen_bones = 2 →
  ratio_of_bones initial_bones final_bones stolen_bones = 2 :=
by
  intros h1 h2 h3
  have h_initial_bones : initial_bones = 4 := h1
  have h_final_bones : final_bones = 6 := h2
  have h_stolen_bones : stolen_bones = 2 := h3
  unfold ratio_of_bones
  rw [h_initial_bones, h_final_bones, h_stolen_bones]
  sorry

end juniper_bones_ratio_l231_231826


namespace find_pairs_l231_231987

def digit_sum (m : ℕ) : ℕ :=
  m.digits.sum -- Function to compute the sum of digits of m

theorem find_pairs (a b : ℕ)
    (h1 : a > 0)
    (h2 : b > 0)
    (h3 : digit_sum (a^(b+1)) = a^b) :
    (a = 1 ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) := 
begin
  sorry
end

end find_pairs_l231_231987


namespace base7_digits_of_143_l231_231107

theorem base7_digits_of_143 : ∃ d1 d2 d3 : ℕ, (d1 < 7 ∧ d2 < 7 ∧ d3 < 7) ∧ (143 = d1 * 49 + d2 * 7 + d3) ∧ (d1 = 2 ∧ d2 = 6 ∧ d3 = 3) :=
by
  sorry

end base7_digits_of_143_l231_231107


namespace gcd_of_B_is_two_l231_231881

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231881


namespace gcd_of_sum_of_four_consecutive_integers_l231_231876

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231876


namespace project_over_budget_l231_231310

theorem project_over_budget (total_budget : ℝ) (months : ℕ) (actual_spent : ℝ) (monthly_alloc : ℝ)
  (expected_spent : ℝ) (difference : ℝ) : 
  total_budget = 12600 ∧ months = 12 ∧ actual_spent = 6580 ∧ 
  monthly_alloc = total_budget / months ∧ expected_spent = monthly_alloc * 6 ∧ 
  difference = actual_spent - expected_spent → 
  difference = 280 :=
by 
  intros h,
  let ⟨h1, h2, h3, h4, h5, h6⟩ := h,
  sorry

end project_over_budget_l231_231310


namespace greatest_common_divisor_of_B_l231_231921

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231921


namespace find_pq_l231_231457

noncomputable def p_and_q (p q : ℝ) := 
  (Complex.I * 2 - 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0} ∧ 
  - (Complex.I * 2 + 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0}

theorem find_pq : ∃ (p q : ℝ), p_and_q p q ∧ p + q = 38 :=
by
  sorry

end find_pq_l231_231457


namespace number_of_interior_diagonals_of_dodecahedron_l231_231001

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l231_231001


namespace dodecahedron_interior_diagonals_l231_231009

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231009


namespace pascal_triangle_fifth_number_l231_231683

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231683


namespace area_of_shaded_region_l231_231414

theorem area_of_shaded_region : 
  let line1 := (λ x : ℝ, - (1 / 10) * x + 3)
      line2 := (λ x : ℝ, - x + 5) in
  (∫ x in (0 : ℝ)..5, line1 x - line2 x) = 5 / 4 := by sorry

end area_of_shaded_region_l231_231414


namespace gcd_B_is_2_l231_231913

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231913


namespace pascal_fifth_number_l231_231741

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231741


namespace reasonable_reasoning_l231_231298

theorem reasonable_reasoning (s1 : Prop) (s2 : Prop) (s3 : Prop) (s4 : Prop) (analogy : Prop) (inductive : Prop) (deductive : Prop)
  (h1 : s1 = analogy)
  (h2 : s2 = inductive)
  (h3 : s3 = inductive)
  (h4 : s4 = deductive) :
  (s1 ∧ s2 ∧ s3) = (analogy ∧ inductive ∧ inductive) :=
by
  sorry

end reasonable_reasoning_l231_231298


namespace dodecahedron_interior_diagonals_l231_231038

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231038


namespace diagonal_length_of_rectangular_prism_l231_231349

-- Define the dimensions of the rectangular prism
variables (a b c : ℕ) (a_pos : a = 12) (b_pos : b = 15) (c_pos : c = 8)

-- Define the theorem statement
theorem diagonal_length_of_rectangular_prism : 
  ∃ d : ℝ, d = Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) ∧ d = Real.sqrt 433 := 
by
  -- Note that the proof is intentionally omitted
  sorry

end diagonal_length_of_rectangular_prism_l231_231349


namespace pascal_triangle_fifth_number_l231_231682

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231682


namespace problem1_problem2_problem3_problem4_l231_231461

variable {V : Type*} [innerProductSpace ℝ V]

variables (a b : V)
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 3) (angle_ab : real_angle a b = π * 2 / 3)

-- (1) ∀ a b, ∥a∥ = 2 → ∥b∥ = 3 → real_angle a b = π * 2 / 3 → inner_product a b = -3
theorem problem1 : inner_product a b = -3 := by sorry

-- (2) ∀ a b, ∥a∥ = 2 → ∥b∥ = 3 → norm_sq a - norm_sq b = -5
theorem problem2 : ∀ (a b : V), ∥a∥ = 2 → ∥b∥ = 3 → ∥a∥ ^ 2 - ∥b∥ ^ 2 = -5 := by sorry

-- (3) ∀ a b, ∥a∥ = 2 → ∥b∥ = 3 → inner_product a b = -3 → inner_product (2 • a - b) (a + 3 • b) = -34
theorem problem3 : inner_product (2 • a - b) (a + 3 • b) = -34 := by sorry

-- (4) ∀ a b, ∥a∥ = 2 → ∥b∥ = 3 → inner_product a b = -3 → ∥a + b∥ = √7
theorem problem4 : ∥a + b∥ = real.sqrt 7 := by sorry

end problem1_problem2_problem3_problem4_l231_231461


namespace gcd_B_is_2_l231_231907

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231907


namespace sqrt_7_estimate_l231_231402

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate_l231_231402


namespace Pascal_triangle_fifth_number_l231_231655

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231655


namespace binomial_sum_even_terms_l231_231405

theorem binomial_sum_even_terms :
  (Finset.range 51).sum (λ k, binomial 101 (2 * k)) = -2^75 :=
by
  sorry

end binomial_sum_even_terms_l231_231405


namespace collinear_ABC_DE_parallel_BC_l231_231988

theorem collinear_ABC_DE_parallel_BC (A B C D E P U V Q : Point) :
  -- Given conditions
  D ∈ Segment A B →
  E ∈ Segment A C →
  Parallel (Line D E) (Line B C) →
  P ∈ Triangle A D E →
  U ∈ (LineSegment P B) ∩ (LineSegment D E) → 
  V ∈ (LineSegment P C) ∩ (LineSegment D E) →
  Q ≠ P ∧ Q ∈ Circumcircle (Triangle P D V) ∧ Q ∈ Circumcircle (Triangle P E U) →
  -- To show
  Collinear {A, P, Q} := by
  sorry

end collinear_ABC_DE_parallel_BC_l231_231988


namespace pascal_row_fifth_number_l231_231584

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231584


namespace greatest_two_digit_product_12_l231_231253

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231253


namespace knights_can_attack_on_3x3_board_l231_231154

noncomputable def prob_knights_attacking (n : ℕ): ℚ :=
  if n = 3 then 209 / 256 else 0

theorem knights_can_attack_on_3x3_board :
  prob_knights_attacking 3 = 209 / 256 :=
by sorry

end knights_can_attack_on_3x3_board_l231_231154


namespace eccentricity_of_ellipse_l231_231556

theorem eccentricity_of_ellipse (A B C : Type*) (a c : ℝ) 
  (hABC : ∀ (A B C : Type*), ∠ C = 90° ∧ ∠ A = 30°) 
  (hEllipse : focus_at A B ∧ passes_through C) : e = √3 - 1 := 
sorry

end eccentricity_of_ellipse_l231_231556


namespace Pascal_triangle_fifth_number_l231_231658

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231658


namespace pascal_fifth_element_15th_row_l231_231707

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231707


namespace find_AB_l231_231100

theorem find_AB (ABCD : Rectangle) (P : Point) (BC : Line) (BP CP : ℝ) 
  (h1 : BP = 18) (h2 : CP = 12) (h3 : tan (angle_APD P ABCD) = 4) : AB = 19 :=
sorry

end find_AB_l231_231100


namespace triangle_area_ratio_l231_231348

theorem triangle_area_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let area_C := (1/2) * (x / 8) * y,
      area_D := (1/2) * (y / 4) * x in
    area_C / area_D = (1/2) :=
by sorry

end triangle_area_ratio_l231_231348


namespace f_g_minus_g_f_l231_231179

def f (x : ℝ) : ℝ := 8 * x - 12
def g (x : ℝ) : ℝ := x / 4 + 3

theorem f_g_minus_g_f (x : ℝ) : f (g x) - g (f x) = 12 := 
by sorry

end f_g_minus_g_f_l231_231179


namespace dodecahedron_interior_diagonals_l231_231018

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231018


namespace ellipse_and_line_properties_l231_231450

noncomputable def ellipse (a b : ℝ) := {p : ℝ × ℝ // (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def line (m c : ℝ) : ℝ × ℝ → Prop
| (x, y) => y = m * (x - 1)

theorem ellipse_and_line_properties :
  ∃ a b : ℝ, a > b ∧ 
             a > 0 ∧ b > 0 ∧ 
             (a^2 = 4 ∧ b^2 = 3) ∧
             (a^2 - b^2 = (1/4) * a^2) ∧
             (let e : ℝ := sqrt 2,
              b = sqrt 3 ∧ 
              ∀ (x y : ℝ), line e 0 (x, y) ↔ (x, y) ∈ ellipse a b ∧ 
              (let ⟨x1, y1⟩ := ((x, y) : ℝ × ℝ),
                (x1 + x1 = 16 / 11 ∧ x1 * x1 = -4 / 11) ∧
                (y1 + y1 = _) ∧ 
                (by sorry) ∧
                (abs (y1 - y1) = 36 / 11) ∧
                (area (⟨0, 0⟩, (x1, y1), (x2, y2)) = 6 * sqrt 6 / 11) ))

end ellipse_and_line_properties_l231_231450


namespace fifth_number_in_pascal_row_l231_231807

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231807


namespace pascal_triangle_fifth_number_l231_231750

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231750


namespace evaluate_expr_l231_231320

-- Define the imaginary unit i
def i := Complex.I

-- Define the expressions for the proof
def expr1 := (1 + 2 * i) * i ^ 3
def expr2 := 2 * i ^ 2

-- The main statement we need to prove
theorem evaluate_expr : expr1 + expr2 = -i :=
by 
  sorry

end evaluate_expr_l231_231320


namespace unique_intersection_l231_231827

noncomputable def f (x : ℝ) : ℝ := x^3 - 9 * x^2 + 27 * x - 14

theorem unique_intersection :
    ∃! a b : ℝ, ((b = f a) ∧ (a = f b)) ∧ f a = a ∧ a = 2 :=
begin
  sorry
end

end unique_intersection_l231_231827


namespace pascal_triangle_fifth_number_l231_231642

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231642


namespace unobstructed_sight_l231_231473

-- Define the curve C as y = 2x^2
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define point A and point B
def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (3, a)

-- Statement of the problem
theorem unobstructed_sight {a : ℝ} (h : ∀ x : ℝ, 0 ≤ x → x ≤ 3 → 4 * x - 2 ≥ 2 * x^2) : a < 10 :=
sorry

end unobstructed_sight_l231_231473


namespace f_two_eq_three_halves_l231_231470

noncomputable def log2 (x : ℝ) : ℝ := real.log x / real.log 2

-- Define the function f(x) according to the given condition.
noncomputable def f (x : ℝ) : ℝ :=
  1 + f (1/2) * log2 x

-- The theorem to prove that f(2) = 3/2
theorem f_two_eq_three_halves : f 2 = 3/2 := 
  sorry

end f_two_eq_three_halves_l231_231470


namespace language_class_probability_l231_231204

theorem language_class_probability :
  (∀ (total_students german_students chinese_students selected_students : ℕ),
    total_students = 30
    → german_students = 22
    → chinese_students = 19
    → selected_students = 2
    → (1 - (Nat.choose (german_students - (german_students + chinese_students - total_students)) 2 +
            Nat.choose (chinese_students - (german_students + chinese_students - total_students)) 2) /
          Nat.choose total_students selected_students) = 352 / 435) :=
begin
  sorry
end

end language_class_probability_l231_231204


namespace count_multiples_6_or_8_not_both_l231_231513

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231513


namespace pascal_triangle_fifth_number_l231_231756

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231756


namespace least_positive_difference_l231_231168

/--
Given Sequence A is a geometric sequence starting at 3 and stopping before exceeding 300.
Sequence B is an arithmetic sequence starting at 10 and stopping before exceeding 300.

Prove that the least positive difference between a number selected from Sequence A and a number selected from Sequence B is 2.
-/
theorem least_positive_difference (A B : List ℕ) (r : ℕ) (d : ℕ) :
  (A = [a in A | a < 300 ∧ ∃ k, a = 3 * r^k]) →
  (B = [b in B | b < 300 ∧ ∃ k, b = 10 + k * d]) →
  ∃ a ∈ A, ∃ b ∈ B, a - b = 2 ∨ b - a = 2 := 
by
  sorry

end least_positive_difference_l231_231168


namespace convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l231_231386

theorem convert_deg_to_rad1 : 780 * (Real.pi / 180) = (13 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad2 : -1560 * (Real.pi / 180) = -(26 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad3 : 67.5 * (Real.pi / 180) = (3 * Real.pi) / 8 := sorry
theorem convert_rad_to_deg1 : -(10 * Real.pi / 3) * (180 / Real.pi) = -600 := sorry
theorem convert_rad_to_deg2 : (Real.pi / 12) * (180 / Real.pi) = 15 := sorry
theorem convert_rad_to_deg3 : (7 * Real.pi / 4) * (180 / Real.pi) = 315 := sorry

end convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l231_231386


namespace pascal_fifth_number_in_row_15_l231_231623

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231623


namespace pascal_triangle_row_fifth_number_l231_231725

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231725


namespace pascal_fifth_number_in_row_15_l231_231625

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231625


namespace whisky_replacement_l231_231339

variable (V : ℝ) (x : ℝ)

theorem whisky_replacement (h_condition : 0.40 * V - 0.40 * x + 0.19 * x = 0.26 * V) : 
  x = (2 / 3) * V := 
sorry

end whisky_replacement_l231_231339


namespace find_d_from_sine_wave_conditions_l231_231373

theorem find_d_from_sine_wave_conditions (a b d : ℝ) (h1 : d + a = 4) (h2 : d - a = -2) : d = 1 :=
by {
  sorry
}

end find_d_from_sine_wave_conditions_l231_231373


namespace divide_hypothetical_day_l231_231338

theorem divide_hypothetical_day (h : 100000 = 2^5 * 5^5) : 
  ∃ (n m : ℕ), n * m = 100000 ∧ n > 0 ∧ m > 0 :=
begin
  sorry
end

end divide_hypothetical_day_l231_231338


namespace problem_statement_l231_231981

variable {a b c d k : ℝ}

theorem problem_statement (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_pos : 0 < k)
    (h_sum_ab : a + b = k)
    (h_sum_cd : c + d = k^2)
    (h_roots1 : ∀ x, x^2 - 4*a*x - 5*b = 0 → x = c ∨ x = d)
    (h_roots2 : ∀ x, x^2 - 4*c*x - 5*d = 0 → x = a ∨ x = b) : 
    a + b + c + d = k + k^2 :=
sorry

end problem_statement_l231_231981


namespace x_plus_y_equals_six_l231_231132

theorem x_plus_y_equals_six (x y : ℝ) (h₁ : y - x = 1) (h₂ : y^2 = x^2 + 6) : x + y = 6 :=
by
  sorry

end x_plus_y_equals_six_l231_231132


namespace pascal_triangle_fifth_number_l231_231695

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231695


namespace negative_number_among_options_l231_231365

theorem negative_number_among_options : 
  ∃ (x : ℤ), x < 0 ∧ 
    (x = |-4| ∨ x = -(-4) ∨ x = (-4)^2 ∨ x = -4^2)
:= by
  use -16
  split
  {
    -- prove that -16 is negative
    linarith
  }
  {
    -- prove that -16 is one of the options
    right; right; right
    norm_num
  }

end negative_number_among_options_l231_231365


namespace imaginary_part_of_complex_l231_231196

theorem imaginary_part_of_complex :
  let i := Complex.I
  let z := 10 * i / (3 + i)
  z.im = 3 :=
by
  sorry

end imaginary_part_of_complex_l231_231196


namespace pascal_fifth_number_in_row_15_l231_231620

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231620


namespace measure_angle_A_l231_231488

-- Define the conditions of the triangle and midpoint etc.
variables (A B C D : Type) [EuclideanGeometry A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Further configurations on points and segments in the triangle
variables {AB AC BD BC : ℝ}
variables {area_ABC : ℝ}
variables [triangle_ABC : Triangle A B C]
variables [midpoint_D : Midpoint D A C]

-- Given conditions
variables (AB_eq_3 : AB = 3)
variables (BD_eq_BC : BD = BC)
variables (area_ABC_eq_3 : area_ABC = 3)
variables (midpoint_DAC : Midpoint D A C)

-- Proof Objective
theorem measure_angle_A (h1 : AB_eq_3) (h2 : BD_eq_BC) (h3 : area_ABC_eq_3) (h4 : midpoint_DAC) :
  ∃ θ, θ = π / 4 :=
by
  -- Sorry to skip proof steps
  sorry

end measure_angle_A_l231_231488


namespace hyperbola_Γ1_exists_l231_231077

-- Definitions based on conditions
def ellipse_Γ2 (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

def foci_Γ2 : set (ℝ × ℝ) :=
  {(-3, 0), (3, 0)}

def vertices_Γ2 : set (ℝ × ℝ) :=
  {(-5, 0), (5, 0)}

def directrix_condition (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_Γ2 x y ∧ (p = (x + 3, y) ∨ p = (x - 3, y))

-- Theorem statement
theorem hyperbola_Γ1_exists :
  (∃ (a b : ℝ), a^2 = 15 ∧ b^2 = 10) →
  (∀ x y : ℝ, (x, y) ∈ vertices_Γ2 → directrix_condition (x, y)) →
  (∃ x y : ℝ, x^2 / 15 - y^2 / 10 = 1) :=
by {
  intros,
  sorry
}

end hyperbola_Γ1_exists_l231_231077


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231964

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231964


namespace smallest_positive_real_l231_231419

theorem smallest_positive_real (x : ℝ) (h₁ : ∃ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 4) : x = 29 / 5 :=
by
  sorry

end smallest_positive_real_l231_231419


namespace books_in_library_l231_231214

theorem books_in_library (initial : ℕ) (taken_out : ℕ) (brought_back : ℕ) : initial = 336 → taken_out = 124 → brought_back = 22 → initial - taken_out + brought_back = 234 :=
by
  intro h_initial h_taken_out h_brought_back
  rw [h_initial, h_taken_out, h_brought_back]
  norm_num
  sorry

end books_in_library_l231_231214


namespace consecutive_numbers_count_count_ways_with_consecutive_l231_231060

theorem consecutive_numbers_count :
  ∀ (s : finset ℕ), s.card = 6 → s ⊆ (finset.range 49).map (λ x, x + 1) →
  ∃ i j, i ≠ j ∧ |(s.to_list.nth_le i (by sorry) - s.to_list.nth_le j (by sorry))| = 1 :=
begin
    sorry
end

theorem count_ways_with_consecutive :
  (nat.choose 49 6) - (nat.choose 44 6) = 6924764 :=
begin
    linarith [nat.choose_eq_factorial_div_factorial (44+5) 5, nat.factorial_eq1_iff],
    sorry
end

end consecutive_numbers_count_count_ways_with_consecutive_l231_231060


namespace pascal_fifth_number_in_row_15_l231_231634

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231634


namespace min_balls_needed_l231_231326

theorem min_balls_needed (bag : Finset (String × Nat)) :
  (∃ r b y bw : ℕ, r + b + y + bw = 70 ∧ r = 20 ∧ b = 20 ∧ y = 20 ∧ bw = 10) →
  ∃ m : ℕ, (m = 38) ∧ (∀ (draw : Finset (String × Nat)), draw.card = m → 
  ∃ c : String, (c = "red" ∨ c = "blue" ∨ c = "yellow") ∧ ∃ n: ℕ, (n ≥ 10) ∧ draw.sum (λ x, if x.1 = c then x.2 else 0) = n) :=
by
  sorry

end min_balls_needed_l231_231326


namespace greatest_common_divisor_of_B_l231_231926

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231926


namespace greatest_two_digit_product_is_12_l231_231245

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231245


namespace pascal_triangle_fifth_number_l231_231757

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231757


namespace dodecahedron_interior_diagonals_l231_231032

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231032


namespace greatest_two_digit_product_is_12_l231_231246

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231246


namespace xy_positive_l231_231069

theorem xy_positive (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 :=
sorry

end xy_positive_l231_231069


namespace ramu_repair_cost_correct_l231_231167

def ramu_spent_on_repairs : ℝ := 12000

theorem ramu_repair_cost_correct (initial_cost selling_price profit_percent : ℝ) 
  (h1 : initial_cost = 34000)
  (h2 : selling_price = 65000)
  (h3 : profit_percent = 41.30434782608695) 
  : 
  let total_cost := initial_cost + ramu_spent_on_repairs in
  let profit := selling_price - total_cost in
  profit_percent = (profit / total_cost) * 100 :=
sorry

end ramu_repair_cost_correct_l231_231167


namespace greatest_common_divisor_of_B_l231_231902

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231902


namespace GCD_of_set_B_is_2_l231_231941

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231941


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231971

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231971


namespace greatest_two_digit_product_12_l231_231273

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231273


namespace fifth_number_in_pascals_triangle_l231_231597

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231597


namespace number_of_paths_l231_231380

theorem number_of_paths (total_steps right_steps up_steps : ℕ) 
    (h1 : total_steps = 10) 
    (h2 : right_steps = 6) 
    (h3 : up_steps = 4) : (Nat.choose total_steps up_steps) = 210 :=
by
  -- Given the conditions
  have h4 : total_steps = right_steps + up_steps, from sorry,
  sorry

end number_of_paths_l231_231380


namespace fifth_number_in_pascal_row_l231_231805

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231805


namespace pascal_triangle_fifth_number_l231_231649

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231649


namespace symmetry_additional_squares_l231_231109

structure Square extends Set (Fin 4 × Fin 6)  -- Define a Square in a 4x6 grid

def initial_shaded : Square := 
  { (1, 2), (3, 1), (4, 4), (6, 1) }

def min_additional_squares_for_symmetry (initial : Square) : Nat :=
  8  -- This is the minimum number of additional squares needed.

theorem symmetry_additional_squares :
  min_additional_squares_for_symmetry initial_shaded = 8 :=
by sorry

end symmetry_additional_squares_l231_231109


namespace concyclic_points_l231_231127

-- Definitions
variable {ABC : Type}
variables {A B C D P M H: ABC} 
variables [IsAcuteTriangle A B C] (hACgtBC : AC > BC)
variables [IsOrthocenter H A B C]
variables [FootOfPerpendicular P B]
variables [Midpoint M A B]
variables [IntersectCircles D ABC (triangleCircumcircle H P C)]

-- Statement of the theorem
theorem concyclic_points (A B C D P M H: ABC) [IsAcuteTriangle A B C] [IsOrthocenter H A B C]
  [FootOfPerpendicular P B] [Midpoint M A B] [IntersectCircles D ABC (triangleCircumcircle H P C)]
  (hACgtBC : AC > BC) : CyclicQuad D P M B :=
sorry

end concyclic_points_l231_231127


namespace michael_classes_selection_l231_231149

theorem michael_classes_selection :
  (nat.choose 9 3 = 84) :=
by sorry

end michael_classes_selection_l231_231149


namespace complex_number_in_fourth_quadrant_l231_231071

theorem complex_number_in_fourth_quadrant :
  let z1 := (1 - Complex.i)
  let z2 := (3 + Complex.i)
  let z := z1 * z2
  (z.re > 0) ∧ (z.im < 0) :=
by
  let z1 := (1 - Complex.i)
  let z2 := (3 + Complex.i)
  let z := z1 * z2
  -- Proof will be inserted here
  sorry

end complex_number_in_fourth_quadrant_l231_231071


namespace incorrect_statement_D_l231_231301

theorem incorrect_statement_D
  (passes_through_center : ∀ (x_vals y_vals : List ℝ), ∃ (regression_line : ℝ → ℝ), 
    regression_line (x_vals.sum / x_vals.length) = (y_vals.sum / y_vals.length))
  (higher_r2_better_fit : ∀ (r2 : ℝ), r2 > 0 → ∃ (residual_sum_squares : ℝ), residual_sum_squares < (1 - r2))
  (slope_interpretation : ∀ (x : ℝ), (0.2 * x + 0.8) - (0.2 * (x - 1) + 0.8) = 0.2)
  (chi_squared_k2 : ∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), (k > 0) → 
    ∃ (confidence : ℝ), confidence > 0) :
  ¬(∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), k > 0 → 
    ∃ (confidence : ℝ), confidence < 0) :=
by
  sorry

end incorrect_statement_D_l231_231301


namespace sequence_characterization_l231_231409

theorem sequence_characterization (a : ℕ → ℕ)
  (h1 : ∀ n, a n ≤ n * real.sqrt n)
  (h2 : ∀ m n, m ≠ n → (a m - a n) % (m - n) = 0) :
  (∀ n, a n = 1) ∨ (∀ n, a n = n) :=
sorry

end sequence_characterization_l231_231409


namespace calculate_expr_l231_231374

noncomputable def expr := sqrt 6 * (sqrt 2 + sqrt 3) - 3 * sqrt (1 / 3)
theorem calculate_expr : expr = sqrt 3 + 3 * sqrt 2 := by
  sorry

end calculate_expr_l231_231374


namespace gcd_of_B_is_two_l231_231882

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231882


namespace a_2009_eq_neg_6_l231_231486

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then 3
  else if n = 2 then 6
  else sequence (n - 1) - sequence (n - 2)

theorem a_2009_eq_neg_6 : sequence 2009 = -6 := by
  sorry

end a_2009_eq_neg_6_l231_231486


namespace greatest_multiple_of_30_less_than_1000_l231_231238

theorem greatest_multiple_of_30_less_than_1000 : ∃ (n : ℕ), n < 1000 ∧ n % 30 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 30 = 0 → m ≤ n := 
by 
  use 990
  sorry

end greatest_multiple_of_30_less_than_1000_l231_231238


namespace pascal_triangle_fifth_number_l231_231648

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231648


namespace pascal_triangle_15_4_l231_231610

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231610


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231968

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231968


namespace ellipse_standard_equation_l231_231468

theorem ellipse_standard_equation :
  let c := 2 in
  let point := (2 * Real.sqrt 3, Real.sqrt 3) in
  ∃ a : ℝ,
    (a = 4) ∧
    (∀ x y : ℝ, (x, y) = point → (x^2 / (a^2) + y^2 / (a^2 - c^2) = 1)) ∧
    (a ≠ 0) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1 →
                   x^2 / (a^2) + y^2 / (a^2 - c^2) = 1 
                   )) :=
by
  sorry

end ellipse_standard_equation_l231_231468


namespace pascal_triangle_row_fifth_number_l231_231730

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231730


namespace pascal_triangle_fifth_number_l231_231694

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231694


namespace proof_math_problem_lean_l231_231540

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231540


namespace dodecahedron_interior_diagonals_l231_231025

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231025


namespace triangle_angle_inequality_l231_231448

noncomputable def solve : Prop :=
  ∀ (A B C : ℝ), 
      (A < B ∧ B < C) ∧ (A + B + C = π) →
      (1 - (Real.sin A / Real.sin B) ≥ (2 * (Real.tan (A / 2))^2 * (Real.tan (B / 2))^2))

theorem triangle_angle_inequality : solve :=
  sorry

end triangle_angle_inequality_l231_231448


namespace pascal_fifth_number_in_row_15_l231_231631

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231631


namespace true_propositions_l231_231979

-- Definitions for lines and planes and their relationships
variables {Line Plane : Type}
variables (a b : Line) (α β : Plane)

-- Definitions of relationships
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l₁ l₂ : Line) : Prop := sorry
def perp_planes (p₁ p₂ : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Propositions
def proposition1 := (perp_line_plane a α) ∧ (line_in_plane a β) → perp_planes α β
def proposition2 := (parallel_line_plane a α) ∧ (perp_planes α β) → perp_line_plane a β
def proposition3 := (perp_line_plane a β) ∧ (perp_planes α β) → parallel_line_plane a α
def proposition4 := (perp_line_plane a α) ∧ (perp_line_plane b α) → parallel_lines a b

-- Correctness of propositions
def correct_propositions : Prop :=
  proposition1 ∧ proposition4 ∧ ¬proposition2 ∧ ¬proposition3

-- Final statement of the theorem
theorem true_propositions :
  correct_propositions := 
  sorry

end true_propositions_l231_231979


namespace remainder_polynomial_l231_231128

theorem remainder_polynomial (Q : ℚ[X]) :
  Q.eval 10 = 35 ∧ Q.eval 35 = 10 →
  ∃ a b : ℚ, Q = (fun x => (x - 10) * (x - 35) * (R : ℚ[X]) + C a * X + C b) ∧
               a = -1 ∧ b = 45 := 
by
  intro h
  have h1 : Q.eval 10 = 35, from h.1
  have h2 : Q.eval 35 = 10, from h.2
  sorry

end remainder_polynomial_l231_231128


namespace number_of_interior_diagonals_of_dodecahedron_l231_231000

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l231_231000


namespace find_nat_numbers_l231_231411

theorem find_nat_numbers (a b : ℕ) (h : 1 / (a - b) = 3 * (1 / (a * b))) : a = 6 ∧ b = 2 :=
sorry

end find_nat_numbers_l231_231411


namespace equation_relationship_l231_231192

variable (x y : ℝ)

-- Defining the condition as a hypothesis
def condition : Prop := (1/2) * x + 3 = 2 * y

-- The theorem to prove the condition
theorem equation_relationship : condition x y :=
begin
  -- proof is not required
  sorry
end

end equation_relationship_l231_231192


namespace dodecahedron_interior_diagonals_l231_231023

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231023


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231967

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231967


namespace gcd_of_B_is_2_l231_231833

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231833


namespace radius_of_tangent_circle_l231_231331

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l231_231331


namespace pascal_triangle_row_fifth_number_l231_231719

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231719


namespace gcd_of_B_is_2_l231_231843

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231843


namespace photo_arrangements_l231_231355

-- The description of the problem conditions translated into definitions
def num_positions := 6  -- Total positions (1 teacher + 5 students)

def teacher_positions := 4  -- Positions where teacher can stand (not at either end)

def student_permutations : ℕ := Nat.factorial 5  -- Number of ways to arrange 5 students

-- The total number of valid arrangements where the teacher does not stand at either end
def total_valid_arrangements : ℕ := teacher_positions * student_permutations

-- Statement to be proven
theorem photo_arrangements:
  total_valid_arrangements = 480 :=
by
  sorry

end photo_arrangements_l231_231355


namespace greatest_two_digit_with_product_12_l231_231265

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231265


namespace fifth_number_in_pascals_triangle_l231_231598

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231598


namespace greatest_two_digit_prod_12_l231_231282

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231282


namespace integral_abs_eq_4_l231_231379

noncomputable def integral_value : ℝ :=
  ∫ x in 0..4, |x - 2|

theorem integral_abs_eq_4 : integral_value = 4 :=
by
  sorry

end integral_abs_eq_4_l231_231379


namespace greatest_integer_100y_l231_231995

noncomputable def y : ℝ := (∑ n in finset.range 1 31, real.cos (n * real.pi / 180)) / (∑ n in finset.range 1 31, real.sin (n * real.pi / 180))

theorem greatest_integer_100y : ⌊100 * y⌋ = 373 := sorry

end greatest_integer_100y_l231_231995


namespace pascal_triangle_fifth_number_l231_231669

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231669


namespace greatest_common_divisor_of_B_l231_231892

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231892


namespace pascal_fifth_element_15th_row_l231_231706

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231706


namespace pq_plus_qr_plus_rp_cubic_1_l231_231126

theorem pq_plus_qr_plus_rp_cubic_1 (p q r : ℝ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + p * r + q * r = -2)
  (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -6 :=
by
  sorry

end pq_plus_qr_plus_rp_cubic_1_l231_231126


namespace sequence_general_formula_l231_231110

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 2 = 2)
    (h3 : ∀ n, a (n + 2) = a n + 2) :
    ∀ n, a n = n := by
  sorry

end sequence_general_formula_l231_231110


namespace gcd_of_B_l231_231939

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231939


namespace real_part_reciprocal_l231_231986

noncomputable def realPartOfReciprocal (z : ℂ) (h₁ : z.im ≠ 0) (h₂ : abs z = 2) : ℝ :=
  ((1 : ℂ) / (2 - z)).re

theorem real_part_reciprocal (z : ℂ) (h₁ : z.im ≠ 0) (h₂ : abs z = 2) : realPartOfReciprocal z h₁ h₂ = 1 / 4 := by
  sorry

end real_part_reciprocal_l231_231986


namespace multiples_6_8_not_both_l231_231529

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231529


namespace num_common_divisors_l231_231059

theorem num_common_divisors (a b : ℕ) (h_a : a = 48) (h_b : b = 60) :
  (∃ n, n = 6) ↔ (∀ d, d ∣ a ∧ d ∣ b ↔ d ∈ {1, 2, 3, 4, 6, 12}) :=
by
  sorry

end num_common_divisors_l231_231059


namespace Pascal_triangle_fifth_number_l231_231665

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231665


namespace cos_angle_between_vectors_l231_231489

theorem cos_angle_between_vectors (a b : ℝ^3)
    (h₁ : ‖a‖ = 5) (h₂ : ‖b‖ = 12) (h₃ : ‖a + b‖ = 13) :
    real.cos (real.angle a b) = 0 := 
sorry

end cos_angle_between_vectors_l231_231489


namespace count_integers_satisfy_inequality_l231_231503

theorem count_integers_satisfy_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.count = 15 := 
sorry

end count_integers_satisfy_inequality_l231_231503


namespace point_is_centroid_l231_231369

variables {A B C P D E F : Type*}
variables [triangle : Simplex A B C]  -- Implicitly defining a triangle ABC
variables [pointP : Inside P triangle] -- P is a point inside the triangle
variables [ap : extension A P] [bp : extension B P] [cp : extension C P]
variables [d : Intersect AP triangle ] [e : Intersect BP triangle] [f : Intersect CP triangle]
variables (S: TriangleAreaSimplex)

-- Given conditions
def S_condition (h1: S APF = 1) (h2: S BPD = 1) (h3: S CPE = 1): P_centroid := sorry

theorem point_is_centroid 
    (h1 : S APF = 1) 
    (h2 : S BPD = 1) 
    (h3 : S CPE = 1) :
    is_centroid P :=
    S_condition h1 h2 h3

end point_is_centroid_l231_231369


namespace area_ratio_inequality_l231_231197

-- Definitions of the sides of the triangle and the areas of triangles
variables {a b c : ℝ} (t_ABC t_PQR : ℝ)

-- Assume the points of tangency and the properties of tangents from vertices to points of tangency
variables {AR AQ BR BP CP CQ : ℝ}

-- Conditions for the sides to be touched by the incircle
axiom tangents_equal : AR = AQ ∧ BR = BP ∧ CP = CQ

-- Definition of areas in terms of side lengths
def area_triangle (a b c : ℝ) : ℝ := sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

-- Let t_ABC and t_PQR be the areas of triangles ABC and PQR respectively
axiom area_ABC : t_ABC = area_triangle a b c

-- The actual proof statement
theorem area_ratio_inequality (h : t_PQR = (b + c - a)^2 / (4 * b * c) * t_ABC
                                 + (c + a - b)^2 / (4 * a * c) * t_ABC
                                 + (a + b - c)^2 / (4 * a * b) * t_ABC) :
  t_PQR ≤ 1 / 4 * t_ABC :=
sorry

end area_ratio_inequality_l231_231197


namespace pascal_fifteen_four_l231_231784

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231784


namespace find_a_l231_231436

theorem find_a 
  (a : ℝ) 
  (f : ℝ → ℝ := λ x, |a * x + 1|)
  (h : ∀ x, -2 ≤ x ∧ x ≤ 1 → f x ≤ 3) :
  a = 2 := by sorry

end find_a_l231_231436


namespace pascal_triangle_fifth_number_l231_231650

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231650


namespace fifth_number_in_pascal_row_l231_231798

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231798


namespace gcd_of_B_is_2_l231_231846

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231846


namespace subsets_containing_5_l231_231544

theorem subsets_containing_5 {α : Type} (s : set α) (h : s = {1, 2, 3, 4, 5}) :
  (set.subset s).count (λ x, 5 ∈ x) = 16 :=
sorry

end subsets_containing_5_l231_231544


namespace fifth_number_in_pascal_row_l231_231811

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231811


namespace log2_S2012_l231_231070

noncomputable def a : ℕ → ℤ
| 0     => 0  -- not defined for n = 0, but added to avoid pattern match failure
| 1     => 2
| (n+1) => 2^(n+1) + 2^n - a n

noncomputable def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, a (i+1))

theorem log2_S2012 :
  log 2 (S 2012 + 2) = 2013 := by
  sorry

end log2_S2012_l231_231070


namespace largest_rectangle_area_in_region_l231_231329

noncomputable def radius := 1
noncomputable def chord_length := Real.sqrt 3
noncomputable def max_rectangle_area := Real.sqrt 3 / 2

theorem largest_rectangle_area_in_region :
  ∀ (C : Set (ℝ × ℝ)) (chord : Set (ℝ × ℝ)), 
    (∀ p ∈ C, ∥p∥ = radius) → 
    (∀ (p ∈ chord), p ∈ C ∧ ∥p - C.midpoint (C.Centre, chord.Centre) ∥ = chord_length / 2) →
    (∃ (R : Set (ℝ × ℝ)), R ⊆ (C ∩ {p | ∀ (q ∈ chord), (p, q) ∈ chord})
      ∧ Rectangle R 
      ∧ area R = max_rectangle_area) := 
sorry

end largest_rectangle_area_in_region_l231_231329


namespace circle_radius_l231_231383

noncomputable def radius_of_circle (PQ : ℝ) (PQR_is_right_iso : PQ = 2) (tangent_to_axes_and_PQ : Prop) : ℝ :=
2 + Mathlib.sqrt 2

theorem circle_radius {PQ : ℝ} (PQR_is_right_iso : PQ = 2) (tangent_to_axes_and_PQ : Prop) :
  radius_of_circle PQ PQR_is_right_iso tangent_to_axes_and_PQ = 2 + Mathlib.sqrt 2 :=
sorry

end circle_radius_l231_231383


namespace greatest_two_digit_with_product_12_l231_231257

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231257


namespace race_time_difference_l231_231362

-- Definitions based on the given conditions
def alice_speed : ℝ := 5  -- minutes per mile
def bob_speed : ℝ := 7    -- minutes per mile
def race_distance : ℝ := 12  -- miles

-- The statement in Lean 4
theorem race_time_difference : 
  (alice_speed * race_distance) + 24 = bob_speed * race_distance :=
by 
  -- The proof is omitted as it's not required
  sorry

end race_time_difference_l231_231362


namespace gcd_B_eq_two_l231_231860

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231860


namespace find_angle_DCA_l231_231815

-- Definitions of the given elements and conditions
variables {A B C D : Type} [euclidean_plane A]

-- Lean does not directly handle Euclidean geometry, but we can use angle definitions and trigonometric formulas.
-- The following angles and lengths are based on the given conditions which need to be proved.

def problem_condition (A B C D : Type) [euclidean_plane A] : Prop :=
  ∃ (ABC : triangle A B C) (DAC : isosceles_triangle A D C),
  angle B = pi / 3 ∧
  side_length A C = sqrt 3 ∧
  point_on_line D A B ∧
  side_length B D = 1 ∧
  side_length D A = side_length D C

def solution (A B C D : Type) [euclidean_plane A] : Prop :=
  ∃ θ : real, 
  problem_condition A B C D ∧
  (angle D C A = π / 6 ∨ angle D C A = π / 18)

theorem find_angle_DCA (A B C D : Type) [euclidean_plane A] :
  solution A B C D :=
sorry

end find_angle_DCA_l231_231815


namespace find_order_amount_l231_231359

noncomputable def unit_price : ℝ := 100

def discount_rate (x : ℕ) : ℝ :=
  if x < 250 then 0
  else if x < 500 then 0.05
  else if x < 1000 then 0.10
  else 0.15

theorem find_order_amount (T : ℝ) (x : ℕ)
    (hx : x = 980) (hT : T = 88200) :
  T = unit_price * x * (1 - discount_rate x) :=
by
  rw [hx, hT]
  sorry

end find_order_amount_l231_231359


namespace find_g_inv_f_of_10_l231_231065

-- Given functions and their inverses
variable {X Y : Type} [Nonempty X] [Nonempty Y]
variable (f : X → Y) (g : X → Y) (f_inv : Y → X) (g_inv : Y → X)

-- Conditions
variable (h1 : ∀ x : X, f_inv (g x) = x^2 - 2)
variable (h2 : Function.Bijective g)

-- The theorem statement
theorem find_g_inv_f_of_10 :
  g_inv (f 10) = 2 * Real.sqrt 3 :=
  sorry

end find_g_inv_f_of_10_l231_231065


namespace children_l231_231206

variable (C : ℝ) -- Define the weight of a children's book

theorem children's_book_weight :
  (9 * 0.8 + 7 * C = 10.98) → C = 0.54 :=
by  
sorry

end children_l231_231206


namespace dodecahedron_interior_diagonals_l231_231019

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231019


namespace ice_cream_volume_l231_231198

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem ice_cream_volume :
  let r_cone := 3
  let r_sphere := 3
  let h_cone := 12
  volume_of_cone r_cone h_cone + volume_of_sphere r_sphere = 72 * π :=
by 
  let r_cone := 3
  let r_sphere := 3
  let h_cone := 12
  sorry

end ice_cream_volume_l231_231198


namespace proof_math_problem_lean_l231_231538

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231538


namespace pascal_triangle_fifth_number_l231_231681

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231681


namespace pascal_triangle_fifth_number_l231_231687

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231687


namespace pascal_fifteen_four_l231_231794

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231794


namespace pascal_15_5th_number_l231_231767

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231767


namespace gcd_of_B_is_2_l231_231849

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231849


namespace cube_fitting_problem_l231_231316

theorem cube_fitting_problem :
  ∀ (L W H V_cube : ℕ), 
  L = 8 → W = 9 → H = 12 → V_cube = 27 →
  (L * W * H / V_cube = 32) :=
by
  intros L W H V_cube hL hW hH hV_cube
  rw [hL, hW, hH, hV_cube]
  norm_num
  sorry

end cube_fitting_problem_l231_231316


namespace gcd_of_B_is_two_l231_231890

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231890


namespace at_least_one_ge_two_l231_231125

theorem at_least_one_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 :=
sorry

end at_least_one_ge_two_l231_231125


namespace greatest_common_divisor_of_B_l231_231927

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231927


namespace cards_flipping_minimum_l231_231323

noncomputable def min_flips_to_determine_positions (grid : Matrix ℕ 4 4) : ℕ :=
sorry

theorem cards_flipping_minimum :
  ∀ (grid : Matrix ℕ 4 4), 
  (∀ i j, (1 <= grid i j ∧ grid i j <= 16) ∧ 
          ((i > 0 ∧ grid i j = grid (i - 1) j + 1) ∨
           (i < 3 ∧ grid i j = grid (i + 1) j + 1) ∨
           (j > 0 ∧ grid i j = grid i (j - 1) + 1) ∨
           (j < 3 ∧ grid i j = grid i (j + 1) + 1))) →
  min_flips_to_determine_positions grid = 8 :=
sorry

end cards_flipping_minimum_l231_231323


namespace fifth_number_in_pascal_row_l231_231810

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231810


namespace dodecahedron_interior_diagonals_l231_231043

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231043


namespace real_distance_between_cities_l231_231188

-- Condition: the map distance between Goteborg and Jonkoping
def map_distance_cm : ℝ := 88

-- Condition: the map scale
def map_scale_km_per_cm : ℝ := 15

-- The real distance to be proven
theorem real_distance_between_cities :
  (map_distance_cm * map_scale_km_per_cm) = 1320 := by
  sorry

end real_distance_between_cities_l231_231188


namespace pascal_triangle_fifth_number_l231_231684

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231684


namespace fifth_number_in_pascal_row_l231_231799

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231799


namespace pascal_triangle_fifth_number_l231_231651

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231651


namespace pascal_triangle_15_4_l231_231608

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231608


namespace gcd_of_B_l231_231938

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231938


namespace greatest_two_digit_product_12_l231_231250

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231250


namespace gcd_of_B_is_2_l231_231845

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231845


namespace gcd_of_B_is_2_l231_231844

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231844


namespace pascal_triangle_15_4_l231_231611

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231611


namespace geometric_sequence_product_l231_231571

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_cond : a 2 * a 4 = 16) : a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64 :=
by
  sorry

end geometric_sequence_product_l231_231571


namespace gcd_B_is_2_l231_231909

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231909


namespace dodecahedron_interior_diagonals_l231_231054

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231054


namespace gcd_of_B_is_2_l231_231853

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231853


namespace pascal_fifth_number_l231_231742

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231742


namespace pascal_triangle_fifth_number_l231_231647

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231647


namespace symmedian_of_triangle_l231_231318

-- The tangent at point B to the circumscribed circle S of triangle ABC intersects line AC at point K
-- A second tangent KD is drawn from point K to circle S
-- Prove that BD is the symmedian of triangle ABC

theorem symmedian_of_triangle (A B C K D : Point) (S : Circle) (hCircle : Circle.ContainTriangle S A B C)
  (hTangent1 : IsTangent S B (LineThrough B K)) 
  (hIntersect : K ∈ LineThrough A C) 
  (hTangent2 : IsTangent S D (LineThrough K D)) :
  IsSymmedian B D A B C :=
begin
  sorry
end

end symmedian_of_triangle_l231_231318


namespace dodecahedron_interior_diagonals_l231_231037

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231037


namespace count_multiples_6_or_8_not_both_l231_231510

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231510


namespace gcd_of_B_is_2_l231_231850

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231850


namespace total_climbing_time_l231_231823

theorem total_climbing_time :
  let a := 30
  let d := 10
  let n := 8
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 520 :=
by
  let a := 30
  let d := 10
  let n := 8
  let S := (n / 2) * (2 * a + (n - 1) * d)
  sorry

end total_climbing_time_l231_231823


namespace gcd_of_B_l231_231929

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231929


namespace profit_is_35_percent_l231_231309

def cost_price (C : ℝ) := C
def initial_selling_price (C : ℝ) := 1.20 * C
def second_selling_price (C : ℝ) := 1.50 * C
def final_selling_price (C : ℝ) := 1.35 * C

theorem profit_is_35_percent (C : ℝ) : 
    final_selling_price C - cost_price C = 0.35 * cost_price C :=
by
    sorry

end profit_is_35_percent_l231_231309


namespace proof_math_problem_lean_l231_231541

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231541


namespace pascal_triangle_15_4_l231_231606

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231606


namespace angle_ABC_eq_120_l231_231816

variable (A B C K L N : Type)
variable [IsTriangle A B C]
variable [IsAngleBisector B K A C]
variable [IsAngleBisector C L A B]
variable [OnSegment N B K]
variable [Parallel LN AC]
variable [EqualLengths NK LN]

theorem angle_ABC_eq_120 : ∠ABC = 120 := by
  sorry

end angle_ABC_eq_120_l231_231816


namespace pascal_row_fifth_number_l231_231575

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231575


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231975

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231975


namespace gcd_B_eq_two_l231_231867

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231867


namespace problem1_l231_231319

theorem problem1 (x : ℝ) (hx : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 := 
sorry

end problem1_l231_231319


namespace complement_of_P_in_U_l231_231458

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U_l231_231458


namespace gcd_B_is_2_l231_231904

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231904


namespace gcd_elements_of_B_l231_231959

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231959


namespace greatest_common_divisor_of_B_l231_231896

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231896


namespace unique_integers_exist_l231_231382

theorem unique_integers_exist (p : ℕ) (hp : p > 1) : 
  ∃ (a b c : ℤ), b^2 - 4*a*c = 1 - 4*p ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end unique_integers_exist_l231_231382


namespace sequence_contains_2018_l231_231422

noncomputable def g (x : ℕ) : ℕ :=
  if x = 0 then 1
  else x.divisors.filter (λ d, d % 2 = 1).max' (begin
    obtain ⟨d, hd⟩ := nat.exists_odd_dvd x,
    use d,
    simp [nat.dvd_of_mem_divisors hd],
  end)

noncomputable def f (x : ℕ) : ℕ :=
  if even x then x / 2 + x / g x else 2 ^ ((x + 1) / 2)

theorem sequence_contains_2018 :
  ∃ n : ℕ, x n = 2018 ∧ (∀ m : ℕ, x m = 2018 → m = n) :=
sorry

noncomputable def x : ℕ → ℕ
| 0     := 1
| (n+1) := f (x n)


end sequence_contains_2018_l231_231422


namespace pascal_triangle_fifth_number_l231_231748

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231748


namespace gcd_of_sum_of_four_consecutive_integers_l231_231875

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231875


namespace sqrt_eq_cond_l231_231396

theorem sqrt_eq_cond (a b : ℝ) (k : ℕ) (hk : k > 0) :
  sqrt (a^2 + (k * b)^2) = a + k * b → (a * k * b = 0 ∧ a + k * b ≥ 0) :=
by
  sorry

end sqrt_eq_cond_l231_231396


namespace pascal_triangle_15_4_l231_231605

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231605


namespace function_monotonically_increasing_iff_range_of_a_l231_231074

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem function_monotonically_increasing_iff_range_of_a (a : ℝ) :
  (∀ x, (deriv (f a) x) ≥ 0) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
by
  sorry

end function_monotonically_increasing_iff_range_of_a_l231_231074


namespace pascal_triangle_fifth_number_l231_231671

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231671


namespace gcd_B_is_2_l231_231910

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231910


namespace length_of_platform_l231_231325

theorem length_of_platform (length_of_train time_to_cross_platform time_to_cross_pole : ℕ)
  (h_train : length_of_train = 750)
  (h_platform_time : time_to_cross_platform = 65)
  (h_pole_time : time_to_cross_pole = 30) :
  let speed_of_train := length_of_train / time_to_cross_pole in
  let total_distance := speed_of_train * time_to_cross_platform in
  total_distance - length_of_train = 875 :=
by
  sorry

end length_of_platform_l231_231325


namespace future_ages_equation_l231_231824

-- Defining the ages of Joe and James with given conditions
def joe_current_age : ℕ := 22
def james_current_age : ℕ := 12

-- Defining the condition that Joe is 10 years older than James
lemma joe_older_than_james : joe_current_age = james_current_age + 10 := by
  unfold joe_current_age james_current_age
  simp

-- Defining the future age condition equation and the target years y.
theorem future_ages_equation (y : ℕ) :
  2 * (joe_current_age + y) = 3 * (james_current_age + y) → y = 8 := by
  unfold joe_current_age james_current_age
  intro h
  linarith

end future_ages_equation_l231_231824


namespace smallest_integer_solution_exists_l231_231291

theorem smallest_integer_solution_exists (x : ℤ) (h : x < 3 * x - 15) : x ≥ 8 :=
by {
  have h1: 2 * x > 15 := sorry,
  have h2: x > 7.5 := sorry,
  have h3: x ≥ 8 := sorry,
  exact h3,
}

end smallest_integer_solution_exists_l231_231291


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231965

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231965


namespace pascal_fifth_number_in_row_15_l231_231624

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231624


namespace slope_product_l231_231483

   -- Define the hyperbola
   def hyperbola (x y : ℝ) : Prop := x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

   -- Define the slope calculation for points P, M, N on the hyperbola
   def slopes (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (Real.sqrt 5 + 1) / 2 = ((yP - y0) * (yP + y0)) / ((xP - x0) * (xP + x0)) := sorry
  
   -- Theorem to show the required relationship
   theorem slope_product (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (yP^2 - y0^2) / (xP^2 - x0^2) = (Real.sqrt 5 + 1) / 2 := sorry
   
end slope_product_l231_231483


namespace greatest_common_divisor_of_B_l231_231925

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231925


namespace multiples_of_6_or_8_under_201_not_both_l231_231522

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231522


namespace probability_of_sum_17_l231_231087

noncomputable def prob_sum_dice_is_seventeen : ℚ :=
1 / 72

theorem probability_of_sum_17 :
  let dice := finset.product (finset.product finset.univ finset.univ) finset.univ in
  let event := dice.filter (λ (x : ℕ × (ℕ × ℕ)), x.1 + x.2.1 + x.2.2 = 17) in
  (event.card : ℚ) / (dice.card : ℚ) = prob_sum_dice_is_seventeen :=
by
  sorry

end probability_of_sum_17_l231_231087


namespace correct_operation_l231_231297

theorem correct_operation :
  (2 * a - a ≠ 2) ∧ ((a - 1) * (a - 1) ≠ a ^ 2 - 1) ∧ (a ^ 6 / a ^ 3 ≠ a ^ 2) ∧ ((-2 * a ^ 3) ^ 2 = 4 * a ^ 6) :=
by
  sorry

end correct_operation_l231_231297


namespace pascal_fifth_element_15th_row_l231_231713

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231713


namespace coefficient_of_monomial_l231_231186

theorem coefficient_of_monomial : 
  ∀ (x y : ℝ), 
  coefficient ([-(9/4), 2] * x^2 * y) = - (9 / 4) := 
begin 
  sorry 
end

end coefficient_of_monomial_l231_231186


namespace pascal_15_5th_number_l231_231778

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231778


namespace distance_incenters_ACD_BCD_l231_231111

noncomputable def distance_between_incenters (AC : ℝ) (angle_ABC : ℝ) (angle_BAC : ℝ) : ℝ :=
  -- Use the given conditions to derive the distance value
  -- Skipping the detailed calculations, denoted by "sorry"
  sorry

theorem distance_incenters_ACD_BCD :
  distance_between_incenters 1 (30 : ℝ) (60 : ℝ) = 0.5177 := sorry

end distance_incenters_ACD_BCD_l231_231111


namespace pascal_triangle_fifth_number_l231_231751

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231751


namespace cover_midpoints_l231_231818

theorem cover_midpoints (M : Set ℝ) (a b c : ℝ) :
  (∀ x ∈ M, x ∈ Set.Icc a (a+1) ∨ x ∈ Set.Icc b (b+1) ∨ x ∈ Set.Icc c (c+1)) →
  ∃ S : ℕ, S = 6 ∧ 
  ∀ (p q ∈ M), ∃ x y : ℝ, (x = (p+q)/2 ∧ y = x + 1) ∧ 
  (x ∈ Set.Icc a (a+1) ∨ x ∈ Set.Icc b (b+1) ∨ x ∈ Set.Icc c (c+1) ∨
   x ∈ Set.Icc (a/2 + b/2) ((a/2 + b/2) + 1) ∨
   x ∈ Set.Icc (a/2 + c/2) ((a/2 + c/2) + 1) ∨
   x ∈ Set.Icc (b/2 + c/2) ((b/2 + c/2) + 1)) :=
by {
  sorry
}

end cover_midpoints_l231_231818


namespace tangent_length_l231_231102

-- Define given conditions
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def A := Point.mk 0 0
def B := Point.mk 10 0
def C := Point.mk 20 0
def D := Point.mk 30 0
def G := Point.mk 50 0  -- Position on x-axis beyond D for constraining problem

def circle_O := Circle.mk A 10
def circle_N := Circle.mk B 10
def circle_P := Circle.mk D 10

-- Angle condition for tangency: angle between AG and PG is 90 degrees
theorem tangent_length :
  let AG := Point.mk 50 (50 * real.sqrt 6) in
  dist A G = 20 * real.sqrt 6 :=
by
  -- Using sorry here to skip the proof as per instructions
  sorry

end tangent_length_l231_231102


namespace multiples_count_l231_231517

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231517


namespace pascal_15_5th_number_l231_231779

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231779


namespace num_zeros_F_unit_circle_l231_231418

noncomputable def F (z : ℂ) : ℂ := z^8 - 4 * z^5 + z^2 - 1

theorem num_zeros_F_unit_circle : (FT (f := F)).box_count (ast_lemma 𝟙) = 5 := 
sorry

end num_zeros_F_unit_circle_l231_231418


namespace dodecahedron_interior_diagonals_l231_231005

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231005


namespace pascal_triangle_fifth_number_l231_231760

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231760


namespace sum_first_pq_l231_231205

-- Given conditions transformed into Lean definitions
variables {a d : ℚ} (p q : ℕ) (h₁ : p ≠ q)
-- The sum of the first 'n' terms in an arithmetic sequence can be expressed as
def arithmetic_sum (n : ℕ) : ℚ :=
  (d / 2) * n^2 + (2 * a - d) / 2 * n

-- Conditions from the problem
axiom sum_first_p (hp : ℕ) (hq : ℕ) : arithmetic_sum hp = hq
axiom sum_first_q (hq : ℕ) (hp : ℕ) : arithmetic_sum hq = hp

-- Proving the required result
theorem sum_first_pq (hp hq : ℕ) (h₁ : hp ≠ hq) : arithmetic_sum (hp + hq) = -(hp + hq) :=
  sorry

end sum_first_pq_l231_231205


namespace dodecahedron_interior_diagonals_l231_231016

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231016


namespace max_people_round_table_l231_231324

theorem max_people_round_table (n : ℕ) (h : 8 ≤ n * n) :
  ∀ (arrangement : ℕ → ℕ), (∀ i < 8, arrangement (i + 1) % n ≠ arrangement i % n) → 
  8 ≤ n * n :=
begin
  sorry,
end

end max_people_round_table_l231_231324


namespace pascal_triangle_15_4_l231_231617

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231617


namespace num_integers_satisfy_inequality_l231_231497

theorem num_integers_satisfy_inequality :
  {n : ℤ | (n+5) * (n-9) ≤ 0}.to_finset.card = 15 :=
by
  sorry

end num_integers_satisfy_inequality_l231_231497


namespace popton_bus_l231_231157

def Hoopit := { toes_per_hand : ℕ := 3, hands : ℕ := 4 }
def Neglart := { toes_per_hand : ℕ := 2, hands : ℕ := 5 }

theorem popton_bus (total_toes : ℕ) (num_neglarts : ℕ) (total_neglart_toes : ℕ) (hoopit_toes_per_hand : ℕ) (hoopit_hands : ℕ) (expected_hoopit_students : ℕ) :
  total_toes = 164 → num_neglarts = 8 →
  total_neglart_toes = num_neglarts * (Neglart.toes_per_hand * Neglart.hands) →
  hoopit_toes_per_hand = Hoopit.toes_per_hand →
  hoopit_hands = Hoopit.hands →
  (total_toes - total_neglart_toes) / (hoopit_toes_per_hand * hoopit_hands) = expected_hoopit_students :=
by
  intros ht nt tnt htp hh
  have neglart_total_toes : total_neglart_toes = 80, from
    by simp [Neglart.toes_per_hand, Neglart.hands]; linarith
  rw [neglart_total_toes] at ht
  have hoopit_toes := 164 - 80
  have hoopit_students := hoopit_toes / (htp * hh)
  simp [Hoopit.toes_per_hand, hoopit_toes, hoopit_students]; linarith
  sorry

end popton_bus_l231_231157


namespace multiples_six_or_eight_not_both_l231_231534

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231534


namespace car_rental_cost_l231_231143

theorem car_rental_cost
  (rent_per_day : ℝ) (cost_per_mile : ℝ) (days_rented : ℕ) (miles_driven : ℝ)
  (h1 : rent_per_day = 30)
  (h2 : cost_per_mile = 0.25)
  (h3 : days_rented = 5)
  (h4 : miles_driven = 500) :
  rent_per_day * days_rented + cost_per_mile * miles_driven = 275 := 
  by
  sorry

end car_rental_cost_l231_231143


namespace pascal_15_5th_number_l231_231766

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231766


namespace GCD_of_set_B_is_2_l231_231940

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231940


namespace sum_of_distances_gt_half_perimeter_l231_231165

variables {A B C M : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (d1 d2 d3 a b c : ℝ)

theorem sum_of_distances_gt_half_perimeter (h1 : d1 + d2 > c) 
                                           (h2 : d1 + d3 > b) 
                                           (h3 : d2 + d3 > a) : 
  d1 + d2 + d3 > (a + b + c) / 2 := 
sorry

end sum_of_distances_gt_half_perimeter_l231_231165


namespace pascal_triangle_fifth_number_l231_231753

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231753


namespace pascal_triangle_fifth_number_l231_231698

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231698


namespace greatest_common_divisor_of_B_l231_231920

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231920


namespace probability_frieda_edges_within_three_hops_l231_231428

constant Grid : Type
constant State : Grid -> Type
constant center : State Grid -> Prop
constant edge : State Grid -> Prop
constant transition_prob : (State Grid) -> (State Grid) -> ℝ

-- Frieda starts in center and the probability of reaching the edge within 3 hops is 13/16
theorem probability_frieda_edges_within_three_hops (C1 C2 : State Grid)
  (cC1 : center C1)
  (cC2 : center C2)
  (edges : Finset (State Grid))
  (edge_cond : ∀ E ∈ edges, edge E)
  (hop_prob : ∀ v w, transition_prob v w = 1/4) :
  (∑ E in edges, transition_prob C1 E ^ 3) + (∑ E in edges, transition_prob C2 E ^ 3) = 13 / 16 :=
sorry

end probability_frieda_edges_within_three_hops_l231_231428


namespace jessica_guess_l231_231820

-- Step a: Define the conditions
def bags : ℕ := 3
def red_jellybeans_bag : ℕ := 24
def white_jellybeans_bag : ℕ := 18

-- Step c: Define the mathematical equivalent problem
theorem jessica_guess :
  let total_jellybeans_bag := red_jellybeans_bag + white_jellybeans_bag in
  let total_jellybeans_fishbowl := total_jellybeans_bag * bags in
  total_jellybeans_fishbowl = 126 :=
by
  sorry

end jessica_guess_l231_231820


namespace sum_binom_mod_2017_sq_a_sum_binom_mod_2017_sq_b_l231_231161

theorem sum_binom_mod_2017_sq_a :
    (∑ k in finset.range 1008 + 1, k * nat.choose 2017 k) % 2017^2 = 0 :=
sorry

theorem sum_binom_mod_2017_sq_b :
    (∑ k in finset.range 504 + 1, (-1)^k * nat.choose 2017 k) % 2017^2 = 3 * (2^2016 - 1) :=
sorry

end sum_binom_mod_2017_sq_a_sum_binom_mod_2017_sq_b_l231_231161


namespace parallel_planes_by_skew_lines_l231_231354

-- Definitions of planes and lines
variables (Plane Line : Type) [has_mem Line Plane]
variable  (Parallel : Line → Plane → Prop)

-- Definitions of skew lines
variable  (Skew : Line → Line → Prop)

-- Given two planes α and β
variables (α β : Plane)

-- Sufficient condition: two skew lines that meet the required properties
theorem parallel_planes_by_skew_lines (a b : Line) :
  a ∈ α ∧ b ∈ β ∧ Skew a b ∧ Parallel a β ∧ Parallel b α → (∀ (a b : Line), Skew a b → Parallel a β → Parallel b α → (a ∈ α ∧ b ∈ β → α = β)) :=
sorry

end parallel_planes_by_skew_lines_l231_231354


namespace pascal_fifth_number_l231_231737

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231737


namespace dodecahedron_interior_diagonals_l231_231012

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231012


namespace pascal_triangle_row_fifth_number_l231_231729

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231729


namespace pascal_15_5th_number_l231_231771

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231771


namespace work_together_days_l231_231308

theorem work_together_days (W : ℝ) (W_AB : ℝ) (W_A : ℝ) :
  (W_AB * 5 = W) →
  (W_A * 10 = W) →
  (W / W_AB = 5) :=
by
  intros h1 h2
  have h3 : W = W_AB * 5 := h1
  have h4 : W = W_A * 10 := h2
  sorry

end work_together_days_l231_231308


namespace gcd_of_B_is_2_l231_231840

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231840


namespace gcd_B_is_2_l231_231912

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231912


namespace Pascal_triangle_fifth_number_l231_231660

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231660


namespace max_exponent_sum_l231_231440

theorem max_exponent_sum (k : ℕ) (a : ℝ) (h_a_pos : a > 0) 
  (k_values : list ℕ) (h_sum : k_values.sum = k) (h_all_nat : ∀ k_i ∈ k_values, k_i ∈ ℕ) 
  (h_r_bounds : 1 ≤ k_values.length ∧ k_values.length ≤ k) : 
  ∃ k_1 k_2 ... (k_values.length terms), 
  a^(k_1) + a^(k_2) + ... + a^(k_values.length) = (k_values.sum) ≤ max (k * a) (a^(k)) :=
sorry

end max_exponent_sum_l231_231440


namespace commutative_otimes_l231_231108

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem commutative_otimes (a b : ℝ) : otimes a b = otimes b a :=
by
  /- The proof will go here, but we omit it and use sorry. -/
  sorry

end commutative_otimes_l231_231108


namespace trigonometric_identity_l231_231433

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : θ > -π/2)
  (h2 : θ < 0)
  (h3 : Real.tan θ = -2) :
  (Real.sin θ)^2 / (Real.cos (2 * θ) + 2) = 4 / 7 :=
sorry

end trigonometric_identity_l231_231433


namespace last_even_distribution_l231_231328

theorem last_even_distribution (n : ℕ) (h : n = 590490) :
  ∃ k : ℕ, (k ≤ n ∧ (n = 3^k + 3^k + 3^k) ∧ (∀ m : ℕ, m < k → ¬(n = 3^m + 3^m + 3^m))) ∧ k = 1 := 
by 
  sorry

end last_even_distribution_l231_231328


namespace weight_of_11_25m_rod_l231_231067

noncomputable def weight_per_meter (total_weight : ℝ) (length : ℝ) : ℝ :=
  total_weight / length

def weight_of_rod (weight_per_length : ℝ) (length : ℝ) : ℝ :=
  weight_per_length * length

theorem weight_of_11_25m_rod :
  let total_weight_8m := 30.4
  let length_8m := 8.0
  let length_11_25m := 11.25
  let weight_per_length := weight_per_meter total_weight_8m length_8m
  weight_of_rod weight_per_length length_11_25m = 42.75 :=
by sorry

end weight_of_11_25m_rod_l231_231067


namespace count_ball_distributions_l231_231545

theorem count_ball_distributions : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → (∀ (dist : ℕ → ℕ), (sorry: Prop))) := sorry

end count_ball_distributions_l231_231545


namespace greatest_two_digit_with_product_12_l231_231270

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231270


namespace pascal_fifteen_four_l231_231780

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231780


namespace starting_lineups_count_l231_231377

theorem starting_lineups_count :
  ∀ (players : Finset ℕ),
  players.card = 15 →
  ∃ Ace Zeppo, Ace ∈ players ∧ Zeppo ∈ players ∧ Ace ≠ Zeppo →
  (let remaining_players := (players.erase Ace).erase Zeppo in
  (remaining_players.card = 13 ∧ ∃ chosen4, chosen4 ⊆ remaining_players ∧ chosen4.card = 4) →
  Finset.choose 4 remaining_players.card = 715) := 
by
  intros players h_players_card h_ace_zeppo h_remaining_players
  sorry

end starting_lineups_count_l231_231377


namespace pascal_fifth_number_l231_231738

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231738


namespace max_n_cells_l231_231353

theorem max_n_cells (n : ℕ) (cond : ∀ (r c : list (ℕ × ℕ)), r.length = n ∧ c.length = n ∧
    (∀ i, i < n → ∃ j, (i, j) ∈ r ∧ (i, j) ∈ c ∧ (r.length * c.length ≥ n)) → n ≤ 7 := 
begin
  sorry,
end


end max_n_cells_l231_231353


namespace pascal_row_fifth_number_l231_231573

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231573


namespace proof_math_problem_lean_l231_231542

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231542


namespace park_area_calculation_l231_231350

def scale := 300 -- miles per inch
def short_diagonal := 10 -- inches
def real_length := short_diagonal * scale -- miles
def park_area := (1/2) * real_length * real_length -- square miles

theorem park_area_calculation : park_area = 4500000 := by
  sorry

end park_area_calculation_l231_231350


namespace cars_on_river_road_l231_231315

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 40) (h2 : B * 3 = C) : C = 60 := 
sorry

end cars_on_river_road_l231_231315


namespace simplify_expression_l231_231174

theorem simplify_expression : 
  (1 / (1 / (1 / 3) ^ 1 + 1 / (1 / 3) ^ 2 + 1 / (1 / 3) ^ 3 + 1 / (1 / 3) ^ 4)) = 1 / 120 :=
by
  sorry

end simplify_expression_l231_231174


namespace multiples_count_l231_231514

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231514


namespace angle_of_inclination_l231_231413

theorem angle_of_inclination (x y : ℝ) (h : x + √3 * y - 3 = 0) : ∃ θ : ℝ, θ = 150 :=
by
  sorry

end angle_of_inclination_l231_231413


namespace day_is_Friday_l231_231101

-- Definitions for days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Definitions for the conditions from the problem
def firstBrother_statement1 (d : Day) : Prop :=
  d ≠ Day.Sunday

def secondBrother_statement1 (d : Day) : Prop :=
  d = Day.Monday

def firstBrother_statement2 (d : Day) (lies_on : Day → Prop) : Prop :=
  lies_on (d.succ)

def secondBrother_statement2 (d : Day) (lied_yesterday : Day → Prop) : Prop :=
  lied_yesterday (d.pred)

-- The theorem stating today is Friday under given conditions
theorem day_is_Friday
  (Tweedledee_lies : Day → Prop)
  (Lion_lied : Day → Prop)
  (h1 : ∀ d, firstBrother_statement1 d)
  (h2 : ∀ d, secondBrother_statement1 d)
  (h3 : ∀ d, firstBrother_statement2 d Tweedledee_lies)
  (h4 : ∀ d, secondBrother_statement2 d Lion_lied) :
  ∃ d, d = Day.Friday :=
sorry

end day_is_Friday_l231_231101


namespace dodecahedron_interior_diagonals_l231_231011

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231011


namespace which_is_linear_system_l231_231299

-- Conditions for the problem
def A_eq1 (x y : ℝ) : Prop := x + y = 1
def A_eq2 (y z : ℝ) : Prop := y + z = 2

def B_eq1 (x y : ℝ) : Prop := x * y = 2
def B_eq2 (x y : ℝ) : Prop := x + y = 1

def C_eq1 (x y : ℝ) : Prop := x + y = 5
def C_eq2 (y : ℝ) : Prop := y = 2

def D_eq1 (x y : ℝ) : Prop := x - 1/y = 2
def D_eq2 (x y : ℝ) : Prop := x + 2 * y = 1

-- Definition of a linear equation in the form of ax + by = c
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- The main theorem statement
theorem which_is_linear_system : 
  (is_linear C_eq1 ∧ is_linear C_eq2) ∧ 
  (¬ (is_linear A_eq1 ∧ is_linear A_eq2)) ∧ 
  (¬ (is_linear B_eq1 ∧ is_linear B_eq2)) ∧ 
  (¬ (is_linear D_eq1 ∧ is_linear D_eq2)) := sorry

end which_is_linear_system_l231_231299


namespace find_f_f3_l231_231476

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 1 - 2^x else x^2 - x - 5

theorem find_f_f3 : f (f 3) = -1 :=
by
  sorry

end find_f_f3_l231_231476


namespace eliot_has_210_l231_231313

noncomputable def eliot_account_balance (A E : ℝ) : Prop :=
  A > E ∧
  A - E = (1 / 12) * (A + E) ∧
  1.10 * A = 1.20 * E + 21

theorem eliot_has_210 : ∃ (E : ℝ), eliot_account_balance (13 / 11 * E) E ∧ E = 210 :=
by
  use 210
  split
  { unfold eliot_account_balance
    sorry }
  { rfl }

end eliot_has_210_l231_231313


namespace dice_probability_sum_17_l231_231081

theorem dice_probability_sum_17 :
  let s : Finset (ℕ × ℕ × ℕ) := 
    (Finset.range 6).image (λ x, (x + 1, x + 1, x + 1))
  ∀ (d1 d2 d3 : ℕ), 
  d1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6} → 
  (d1 + d2 + d3 = 17 ↔ (d1, d2, d3) ∈ s) → 
  s.card = 1 / 72 := 
begin
  sorry
end

end dice_probability_sum_17_l231_231081


namespace pascal_fifth_element_15th_row_l231_231705

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231705


namespace trace_bag_weight_l231_231225

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l231_231225


namespace pascal_fifth_number_in_row_15_l231_231622

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231622


namespace tan_identity_equality_l231_231459

theorem tan_identity_equality
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 :=
by
  sorry

end tan_identity_equality_l231_231459


namespace mod_sum_correct_l231_231063

theorem mod_sum_correct (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
    (h1 : a * b * c ≡ 1 [MOD 7])
    (h2 : 5 * c ≡ 2 [MOD 7])
    (h3 : 6 * b ≡ 3 + b [MOD 7]) :
    (a + b + c) % 7 = 4 := sorry

end mod_sum_correct_l231_231063


namespace valid_set_of_points_l231_231442

/-- A finite set of points in the plane where no three points are collinear, and for any three points A, B, C 
in the set, the orthocenter of triangle ABC is also in the set. Such sets must either be the set of 
four vertices of a rectangle, the set of three vertices of a right triangle, or the set of three vertices of an acute triangle along with its orthocenter. -/
theorem valid_set_of_points
    (M : Finset Point)
    (h1 : ∀ A B C ∈ M, ¬ collinear A B C)
    (h2 : ∀ A B C ∈ M, orthocenter A B C ∈ M) :
    (exists (a b c d : Point), M = {a, b, c, d} ∧ is_rectangle a b c d) ∨
    (exists (a b c : Point), M = {a, b, c} ∧ is_right_triangle a b c) ∨
    (exists (a b c H : Point), M = {a, b, c, H} ∧ is_acute_triangle a b c ∧ H = orthocenter a b c) :=
sorry

end valid_set_of_points_l231_231442


namespace GCD_of_set_B_is_2_l231_231945

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231945


namespace dodecahedron_interior_diagonals_l231_231045

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231045


namespace gcd_B_eq_two_l231_231856

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231856


namespace greatest_common_divisor_of_B_l231_231918

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231918


namespace range_of_a_l231_231455

variable (x a : ℝ)

def p : Prop := 4 / (x - 1) ≤ -1
def q : Prop := x ^ 2 - x < a ^ 2 - a

theorem range_of_a :
  (∃ x : ℝ, p x) →
  (∃ x : ℝ, q x) →
  (∀ x : ℝ, ¬ q x → ¬ p x) →
  (0 ≤ a ∧ a ≤ 1 ∧ a ≠ 1/2) :=
by
  sorry

end range_of_a_l231_231455


namespace greatest_common_divisor_of_B_l231_231924

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231924


namespace gcd_elements_of_B_l231_231954

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231954


namespace jessica_guess_l231_231819

-- Step a: Define the conditions
def bags : ℕ := 3
def red_jellybeans_bag : ℕ := 24
def white_jellybeans_bag : ℕ := 18

-- Step c: Define the mathematical equivalent problem
theorem jessica_guess :
  let total_jellybeans_bag := red_jellybeans_bag + white_jellybeans_bag in
  let total_jellybeans_fishbowl := total_jellybeans_bag * bags in
  total_jellybeans_fishbowl = 126 :=
by
  sorry

end jessica_guess_l231_231819


namespace dodecahedron_interior_diagonals_l231_231048

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231048


namespace domain_of_f_eq_domain_of_f_cos_l231_231189

variable (k : ℤ)

def domain_of_f_cos : Set ℝ :=
  {x | ∃ k : ℤ, x ∈ Icc (2 * k * Real.pi - Real.pi / 6) (2 * k * Real.pi + 2 * Real.pi / 3)}

def domain_of_f : Set ℝ :=
  Icc (-1 / 2) 1

theorem domain_of_f_eq_domain_of_f_cos :
  (∀ x ∈ domain_of_f_cos k, (∃ y ∈ domain_of_f, y = Real.cos x)) ↔
  (domain_of_f = Icc (-1 / 2) 1) :=
sorry

end domain_of_f_eq_domain_of_f_cos_l231_231189


namespace heavy_tailed_permutations_count_l231_231443

/-- A permutation is heavy-tailed if the sum of the first three numbers is less than the sum of 
    the last three numbers and the third number is even. -/
def heavy_tailed (p : Perm (Fin 6)) : Prop :=
  p 0 + p 1 + p 2 < p 3 + p 4 + p 5 ∧ (p 2) % 2 = 0

open Finset

/-- The number of heavy-tailed permutations of the set {1, 2, 3, 4, 5, 6} -/
theorem heavy_tailed_permutations_count : 
  (Finset.univ.filter heavy_tailed).card = 140 := sorry

end heavy_tailed_permutations_count_l231_231443


namespace complement_M_eq_45_l231_231999

open Set Nat

/-- Define the universal set U and the set M in Lean -/
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

def M : Set ℕ := {x | 6 % x = 0 ∧ x ∈ U}

/-- Lean theorem statement for the complement of M in U -/
theorem complement_M_eq_45 : (U \ M) = {4, 5} :=
by
  sorry

end complement_M_eq_45_l231_231999


namespace dodecahedron_interior_diagonals_l231_231002

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231002


namespace gcd_of_B_is_2_l231_231847

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231847


namespace sum_of_integers_division_remainder_l231_231123

theorem sum_of_integers_division_remainder :
  let S := (List.sum
    (List.filter (λ n, ∃ m : ℤ, n^2 + 12 * n - 3021 = m^2) 
      (List.range 2000))) -- arbitrary large bound to search for solutions
  in
  S % 1000 = 749 := 
by
  -- Let S be defined as the sum of those integers n such that n^2 + 12n - 3021 = m^2
  let S := List.sum
    (List.filter (λ n, ∃ m : ℤ, n^2 + 12 * n - 3021 = m^2) 
      (List.range 2000));
  -- Proving the final result
  have h : S = 1749 := sorry;
  have mod_result : 1749 % 1000 = 749 := by norm_num;
  exact mod_result

end sum_of_integers_division_remainder_l231_231123


namespace pascal_triangle_fifth_number_l231_231680

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231680


namespace pascal_fifteen_four_l231_231787

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231787


namespace pascal_triangle_fifth_number_l231_231754

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231754


namespace polynomial_problem_l231_231138

noncomputable def p (x : ℤ) : ℤ := sorry -- The actual polynomial p(x)
noncomputable def q (x : ℤ) : ℤ := sorry -- The actual polynomial q(x)

theorem polynomial_problem :
  (x^8 - 50 * x^4 + 25) = (p(x) * q(x)) →
  polynomial.monic p ∧ polynomial.monic q →
  (¬ (p.degree = 0) ∧ ¬ (q.degree = 0)) →
  (p.coeff p.natDegree = 1 ∧ q.coeff q.natDegree = 1) →
  p(1) + q(1) = 12 :=
by
  sorry

end polynomial_problem_l231_231138


namespace multiples_6_8_not_both_l231_231530

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231530


namespace fifth_number_in_pascal_row_l231_231800

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231800


namespace resized_height_l231_231400

-- Define original dimensions
def original_width : ℝ := 4.5
def original_height : ℝ := 3

-- Define new width
def new_width : ℝ := 13.5

-- Define new height to be proven
def new_height : ℝ := 9

-- Theorem statement
theorem resized_height :
  (new_width / original_width) * original_height = new_height :=
by
  -- The statement that equates the new height calculated proportionately to 9
  sorry

end resized_height_l231_231400


namespace pascal_15_5th_number_l231_231773

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231773


namespace GCD_of_set_B_is_2_l231_231943

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231943


namespace intersection_point_l231_231288

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 10

noncomputable def slope_perp : ℝ := -1/3

noncomputable def line_perp (x : ℝ) : ℝ := slope_perp * x + (2 - slope_perp * 3)

theorem intersection_point : 
  ∃ (x y : ℝ), y = line1 x ∧ y = line_perp x ∧ x = -21 / 10 ∧ y = 37 / 10 :=
by
  sorry

end intersection_point_l231_231288


namespace polynomial_lt_factorial_l231_231453

theorem polynomial_lt_factorial (A B C : ℝ) : ∃N : ℕ, ∀n : ℕ, n > N → An^2 + Bn + C < n! := 
by
  sorry

end polynomial_lt_factorial_l231_231453


namespace terminating_decimal_values_l231_231423

theorem terminating_decimal_values (k : ℤ) (hk : 10 ≤ k ∧ k ≤ 230) :
  ∃ n : ℕ, n = 6 ∧ (∀ k, 10 ≤ k ∧ k ≤ 230 → (k / 330 : ℚ).denom ≤ 5 ↔ (k % 33 = 0)) :=
by sorry

end terminating_decimal_values_l231_231423


namespace greatest_two_digit_product_12_l231_231255

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231255


namespace dodecahedron_interior_diagonals_l231_231015

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231015


namespace percentage_reduction_l231_231203

theorem percentage_reduction (S P : ℝ) (h : S - (P / 100) * S = S / 2) : P = 50 :=
by
  sorry

end percentage_reduction_l231_231203


namespace dodecahedron_interior_diagonals_l231_231017

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231017


namespace gcd_B_eq_two_l231_231862

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231862


namespace pascal_fifth_element_15th_row_l231_231701

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231701


namespace pascal_fifteen_four_l231_231785

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231785


namespace greatest_two_digit_prod_12_l231_231283

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231283


namespace pascal_row_fifth_number_l231_231585

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231585


namespace num_integers_satisfy_inequality_l231_231499

theorem num_integers_satisfy_inequality :
  {n : ℤ | (n+5) * (n-9) ≤ 0}.to_finset.card = 15 :=
by
  sorry

end num_integers_satisfy_inequality_l231_231499


namespace pascal_triangle_fifth_number_l231_231637

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231637


namespace dodecahedron_interior_diagonals_l231_231028

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231028


namespace multiples_count_l231_231516

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231516


namespace total_blue_marbles_l231_231146

noncomputable def total_blue_marbles_collected_by_friends : ℕ := 
  let jenny_red := 30 in
  let jenny_blue := 25 in
  let mary_red := 2 * jenny_red in
  let anie_red := mary_red + 20 in
  let anie_blue := 2 * jenny_blue in
  let mary_blue := anie_blue / 2 in
  jenny_blue + mary_blue + anie_blue

theorem total_blue_marbles (jenny_red jenny_blue mary_red anie_red mary_blue anie_blue : ℕ) :
  jenny_red = 30 → 
  jenny_blue = 25 → 
  mary_red = 2 * jenny_red → 
  anie_red = mary_red + 20 → 
  anie_blue = 2 * jenny_blue → 
  mary_blue = anie_blue / 2 → 
  jenny_blue + mary_blue + anie_blue = 100 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4, h5, h6],
  norm_num,
end

end total_blue_marbles_l231_231146


namespace pascal_row_fifth_number_l231_231576

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231576


namespace problem1_problem2_problem3_l231_231460

open Real

def f (x a : ℝ) : ℝ := (x^2 - 4) * (x - a)

theorem problem1 (a : ℝ): 
  ∀ x : ℝ, deriv (λ x, f x  a) x = 3 * x^2 - 2 * a * x - 4 := by
  intro x
  sorry

theorem problem2 : 
  let f (x: ℝ ) := x^3 - (1/2)*x^2 - 4*x + 2 in 
  ∃ x_max x_min : ℝ, 
  x_max ∈ Icc (-2 : ℝ) (2 : ℝ) ∧ f x_max = (9 / 2) ∧
  x_min ∈ Icc (-2 : ℝ) (2 : ℝ) ∧ f x_min = - (50 / 27) := by
  have x₁ := -1 
  have x₂ := 2 
  have x₃ := -2 
  have x₄ := (4:ℤ)/3 
  let fx₁ := (9 : ℝ ) / (2 : ℝ )
  let fx₂ := (0 : ℝ)
  let fx₃ := (0 : ℝ)
  let fx₄ := - (50 : ℝ ) / 27
  use x₁ , x₄ 
  split 
  {
    dsimp 
    sorry,
  }
  split
  {
    dsimp 
    sorry
  }
  split 
  {
    dsimp 
    sorry
  }
  split 
  {
    dsimp
    sorry
  }

theorem problem3 (a : ℝ): 
  (∀ x ∈ Icc (-∞: ℝ) (-2 : ℝ), deriv (λ x , f x a) x ≥ 0) ∧ 
  (∀ x ∈ Icc (2: ℝ) (+∞ : ℝ), deriv (λ x , f x a) x ≥ 0) ↔ 
  (-2: ℝ) ≤ a ∧ a ≤ (2: ℝ) := by
  intro x
  have P : ∀ x , deriv (λ x, f x  a) x = 3 * x^2 - 2 * a * x - 4 := by 
    intro x
    sorry
  split 
  {
    intro
    dsimp
    split 
    {
      dsimp
      sorry
    }
    split 
    {
      dsimp
      sorry
    }
  }
  split 
  {
    intro 
    split 
    {
      intro 
      dsimp 
      sorry
    }
    split 
    {
      intro 
      dsimp
      sorry
    }
  }

end problem1_problem2_problem3_l231_231460


namespace pascal_triangle_row_fifth_number_l231_231718

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231718


namespace fg_minus_gf_l231_231180

def f (x : ℝ) : ℝ := 8 * x - 12
def g (x : ℝ) : ℝ := x / 4 + 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 12 := 
by
  sorry

end fg_minus_gf_l231_231180


namespace monotonicity_of_g_l231_231480

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb a (|x + 1|)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.logb a (- (3 / 2) * x^2 + a * x)

theorem monotonicity_of_g (a : ℝ) (h : 0 < a ∧ a ≠ 1) (h0 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x a < 0) :
  ∀ x : ℝ, 0 < x ∧ x ≤ a / 3 → (g x a) < (g (x + ε) a) := 
sorry


end monotonicity_of_g_l231_231480


namespace sum_tens_units_digit_9_pow_1001_l231_231293

-- Define a function to extract the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define a function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (last_two_digits n) / 10

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := (last_two_digits n) % 10

-- The main theorem
theorem sum_tens_units_digit_9_pow_1001 :
  tens_digit (9 ^ 1001) + units_digit (9 ^ 1001) = 9 :=
by
  sorry

end sum_tens_units_digit_9_pow_1001_l231_231293


namespace minimum_value_proof_l231_231337

-- Conditions of the problem are defined as Lean hypotheses
def hyperbola_equation (a b : ℝ) := ∀ (x y : ℝ), ((x^2) / (a^2)) - ((y^2) / (b^2)) = 1

-- Definitions based on the given problem
def relation_b_a (a b : ℝ) : Prop := (b = sqrt 3 * a)
def eccentricity_relation (a e : ℝ) (c : ℝ) (h : c = a * e) : Prop := true

-- The minimum value to be proven
def minimum_expression (a e b : ℝ) : ℝ := (a^2 + e) / b

-- The Lean statement of the proof
theorem minimum_value_proof (a e b c : ℝ) 
(hyperbola_eq : hyperbola_equation a b) 
(h_rel : relation_b_a a b) 
(eccentricity_rel : eccentricity_relation a e c (by sorry)) 
(h_ecc : e = 2) 
(h_a : a = sqrt 2) : minimum_expression a e b = 2 * sqrt 6 / 3 := 
by 
  unfold minimum_expression
  rw [h_ecc, h_a, h_rel]
  admit -- the proof is left as an exercise

end minimum_value_proof_l231_231337


namespace pascal_15_5th_number_l231_231775

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231775


namespace pascal_fifth_number_l231_231735

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231735


namespace greatest_two_digit_with_product_12_l231_231258

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231258


namespace total_blue_marbles_l231_231147

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end total_blue_marbles_l231_231147


namespace num_integers_satisfying_inequality_l231_231501

theorem num_integers_satisfying_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.finite.toFinset.card = 15 :=
by
  sorry

end num_integers_satisfying_inequality_l231_231501


namespace negative_number_among_options_l231_231366

theorem negative_number_among_options : 
  ∃ (x : ℤ), x < 0 ∧ 
    (x = |-4| ∨ x = -(-4) ∨ x = (-4)^2 ∨ x = -4^2)
:= by
  use -16
  split
  {
    -- prove that -16 is negative
    linarith
  }
  {
    -- prove that -16 is one of the options
    right; right; right
    norm_num
  }

end negative_number_among_options_l231_231366


namespace dice_probability_sum_17_l231_231080

theorem dice_probability_sum_17 :
  let s : Finset (ℕ × ℕ × ℕ) := 
    (Finset.range 6).image (λ x, (x + 1, x + 1, x + 1))
  ∀ (d1 d2 d3 : ℕ), 
  d1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6} → 
  (d1 + d2 + d3 = 17 ↔ (d1, d2, d3) ∈ s) → 
  s.card = 1 / 72 := 
begin
  sorry
end

end dice_probability_sum_17_l231_231080


namespace fifth_number_in_pascal_row_l231_231808

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231808


namespace gcd_times_xyz_is_square_l231_231134

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_times_xyz_is_square_l231_231134


namespace pascal_triangle_fifth_number_l231_231686

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231686


namespace part_a_l231_231829

noncomputable def f (x : ℝ) : ℝ := sorry

theorem part_a (f : ℝ → ℝ)
  (h1 : ∀ x > 0, ∀ n, f (n * x) ≤ f ((n + 1) * x))
  (h2 : ∀ x ∈ Icc (0 : ℝ) 1, continuous_at f x) :
  ¬ ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y :=
sorry

end part_a_l231_231829


namespace greatest_common_divisor_of_B_l231_231899

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231899


namespace pascal_triangle_fifth_number_l231_231696

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231696


namespace gcd_elements_of_B_l231_231962

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231962


namespace dodecahedron_interior_diagonals_l231_231020

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231020


namespace pascal_triangle_fifth_number_l231_231673

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231673


namespace count_integers_satisfy_inequality_l231_231505

theorem count_integers_satisfy_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.count = 15 := 
sorry

end count_integers_satisfy_inequality_l231_231505


namespace distance_from_point_to_line_is_correct_l231_231415

-- Definitions from problem conditions
def point := (2 : ℝ, 4 : ℝ, 6 : ℝ)
def line (t : ℝ) : ℝ × ℝ × ℝ := (8 + 4 * t, 9 + 3 * t, 9 - 3 * t)

-- Definition of the distance function
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2

-- The proof statement
theorem distance_from_point_to_line_is_correct :
  ∃ t : ℝ, let closest_point := line t,
           let dist := distance point closest_point in
           dist = 2 * Real.sqrt 41 :=
by
  sorry

end distance_from_point_to_line_is_correct_l231_231415


namespace gcd_of_B_is_2_l231_231848

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231848


namespace rational_solutions_quad_eq_iff_k_eq_4_l231_231394

theorem rational_solutions_quad_eq_iff_k_eq_4 (k : ℕ) (hk : 0 < k) : 
  (∃ x : ℚ, x^2 + 24/k * x + 9 = 0) ↔ k = 4 :=
sorry

end rational_solutions_quad_eq_iff_k_eq_4_l231_231394


namespace new_marbles_found_l231_231825

theorem new_marbles_found : ∀ (lost found_additional : ℕ), lost = 8 → found_additional = 2 → (lost + found_additional) = 10 :=
by
  intros lost found_additional hlost hfound_additional
  rw [hlost, hfound_additional]
  exact rfl

end new_marbles_found_l231_231825


namespace pascal_fifteen_four_l231_231783

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231783


namespace subsets_and_proper_subsets_l231_231142

namespace SubsetProof

open Set

def M : Set ℕ := {1, 2, 3}

theorem subsets_and_proper_subsets :
  ∃ subsets proper_subsets : Set (Set ℕ),
  subsets = {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}} ∧
  proper_subsets = {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}} :=
by
  let subsets := (𝒫 M).toFinset
  let proper_subsets := (subsets.erase M).toFinset
  exists subsets proper_subsets
  split
  · rw [subset_def, toFinset_eq, Finset.mem_def, toFinset_mem_equiv, Set.toFinset_univ]
    finish
  · rw [subset_def, toFinset_eq, Finset.erase_eq, Finset.mem_def, toFinset_mem_equiv, Set.toFinset_univ]
    finish

#check subsets_and_proper_subsets -- Check the statement

end SubsetProof

end subsets_and_proper_subsets_l231_231142


namespace greatest_two_digit_with_product_12_l231_231269

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231269


namespace pascal_fifth_element_15th_row_l231_231711

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231711


namespace vacant_seat_count_l231_231093

def total_seats : ℕ := 600
def filled_percentage : ℚ := 0.50
def discounted_percentage : ℚ := 0.30

def filled_seats : ℕ := (filled_percentage * total_seats).toInt
def discounted_seats : ℕ := (discounted_percentage * total_seats).toInt
def regular_seats : ℕ := total_seats - discounted_seats

def vacant_discounted_seats : ℕ := 0
def vacant_regular_seats : ℕ := regular_seats - (filled_seats - discounted_seats)

theorem vacant_seat_count :
  vacant_discounted_seats = 0 ∧ vacant_regular_seats = 300 := by
  sorry

end vacant_seat_count_l231_231093


namespace spheres_in_parallelepiped_l231_231115

theorem spheres_in_parallelepiped (AB A1D1 CC1 AB_less_sqrt2 A1D1_more_sqrt2 CC1_6 :
  ∃ σ₁ σ₂ : Type, 
         touching_each_other σ₁ σ₂
     ∧ touching_faces σ₁ AB A1D1 A A A
     ∧ touching_faces σ₂ A1D1 B1 C1 B1
     ∧ AB = 6 - sqrt 2 
     ∧ A1D1 = 6 + sqrt 2 
     ∧ CC1 = 6 
     ) :
    distance_between_centers σ₁ σ₂ = 4 
∧ min_total_volume σ₁ σ₂ = (136 / 3 - 16 * sqrt 2) * pi 
∧ max_total_volume σ₁ σ₂ = 64 / 3 * pi
:=
by
  sorry

end spheres_in_parallelepiped_l231_231115


namespace percentage_juan_to_jason_l231_231312

variables {T J M K Jn : ℝ}

-- Given conditions
def condition1 : M = 1.60 * T := sorry
def condition2 : T = 0.50 * J := sorry
def condition3 : K = 1.20 * M := sorry
def condition4 : Jn = 0.70 * K := sorry

-- The goal is to prove the percentage relationship
theorem percentage_juan_to_jason :
  (J / Jn) * 100 = 148.81 :=
begin
  -- Using the given conditions
  have h1 : M = 1.60 * T := condition1,
  have h2 : T = 0.50 * J := condition2,
  have h3 : K = 1.20 * M := condition3,
  have h4 : Jn = 0.70 * K := condition4,
  -- Proving the required relation
  sorry
end

end percentage_juan_to_jason_l231_231312


namespace evaluate_expression_l231_231403

theorem evaluate_expression :
  let c := (-2 : ℚ)
  let x := (2 : ℚ) / 5
  let y := (3 : ℚ) / 5
  let z := (-3 : ℚ)
  c * x^3 * y^4 * z^2 = (-11664) / 78125 := by
  sorry

end evaluate_expression_l231_231403


namespace greatest_common_divisor_of_B_l231_231894

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231894


namespace pascal_fifth_element_15th_row_l231_231703

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231703


namespace f_periodic_l231_231998

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom domain_ℝ : ∀ x : ℝ, x ∈ ℝ
axiom f_odd : ∀ x : ℝ, f(-x - 1) = -f(x - 1)
axiom f_even : ∀ x : ℝ, f(-x + 1) = f(x + 1)
axiom f_interval : ∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 → f(x) = -Real.exp x

-- To prove
theorem f_periodic : ∀ x : ℝ, f(2x) = f(2x + 8) := 
sorry

end f_periodic_l231_231998


namespace planes_intersect_necessary_not_sufficient_l231_231463

noncomputable def planes_and_lines (α β : Type) (m n : Type): Prop :=
  (α ≠ β ∧ m ⊥ α ∧ n ⊥ β) → 
  (∃ p : α ∩ β, ∀ q ∈ α ∩ β, q = p) →
  (n ∩ m = ∅ → α ∩ β = ∅)

-- Statement: Given the conditions, the planes intersect is necessary but not sufficient for the lines to be skew.
theorem planes_intersect_necessary_not_sufficient 
  (α β : Type) (m n : Type)
  (h : planes_and_lines α β m n) : 
  (∃ p : α ∩ β, ∀ q ∈ α ∩ β, q = p) ↔ (n ∩ m = ∅) :=
begin
  sorry
end

end planes_intersect_necessary_not_sufficient_l231_231463


namespace pascal_fifth_number_l231_231732

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231732


namespace Pascal_triangle_fifth_number_l231_231663

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231663


namespace ellipse_focus_ratio_l231_231976

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) := {p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def foci (a b : ℝ) : set (ℝ × ℝ) :=
  let c := Real.sqrt (a^2 - b^2) in {(-c, 0), (c, 0)}

theorem ellipse_focus_ratio :
  let a := 2
  let b := 1
  let e := ellipse a b
  let f1 := (-Real.sqrt (a^2 - b^2), 0)
  let f2 := (Real.sqrt (a^2 - b^2), 0)
  ∀ P : ℝ × ℝ,
    P ∈ e →
    (∃ M : ℝ × ℝ, M.1 = 0 ∧ M = ((P.1 + f1.1) / 2, (P.2 + f1.2) / 2)) →
    |P.fst - f2.fst| / |P.fst - f1.fst| = 1/7 := by
  intros a b e f1 f2 P P_in_e midpoint
  sorry

end ellipse_focus_ratio_l231_231976


namespace pascal_fifth_number_in_row_15_l231_231627

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231627


namespace gcd_B_is_2_l231_231906

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231906


namespace greatest_two_digit_product_is_12_l231_231243

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231243


namespace compare_expressions_l231_231434

theorem compare_expressions :
  let a := 2 ^ (Real.log 3 / Real.log 4)
  let b := Real.log 8 / Real.log 4
  let c := 3 ^ 0.6
  (b < a) ∧ (a < c) :=
by
  sorry

end compare_expressions_l231_231434


namespace greatest_two_digit_product_12_l231_231272

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231272


namespace range_of_k_l231_231555

theorem range_of_k 
  (h : ∀ x : ℝ, (k^2 - 2*k + 3/2)^x < (k^2 - 2*k + 3/2)^(1 - x) ↔ x ∈ Ioi (1/2)) :
  1 - Real.sqrt 2 / 2 < k ∧ k < 1 + Real.sqrt 2 / 2 :=
by sorry

end range_of_k_l231_231555


namespace product_A_mod_p_l231_231990

theorem product_A_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (hodd : p % 2 = 1) :
  let A := {a | a < p ∧ ¬(∃ t, (t * t) % p = a) ∧ ¬(∃ t, (t * t) % p = (4 - a) % p)}
  let prodA := Finset.prod (Finset.filter (λ a, a ∈ A) (Finset.range p)) (λ x => x)
  prodA % p = 2 :=
by
  sorry

end product_A_mod_p_l231_231990


namespace number_of_white_towels_l231_231303

-- Definitions based on the conditions
def green_towels := 35
def given_away := 34
def remaining_towels := 22

-- Theorem statement
theorem number_of_white_towels : 
    (total_towels : ℕ) (white_towels : ℕ) 
    (H1 : total_towels = remaining_towels + given_away)
    (H2 : white_towels = total_towels - green_towels) 
    : white_towels = 21 := 
by
  sorry

end number_of_white_towels_l231_231303


namespace radius_of_tangent_circle_l231_231330

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l231_231330


namespace fifth_number_in_pascals_triangle_l231_231589

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231589


namespace fifth_number_in_pascal_row_l231_231809

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231809


namespace complex_algebra_l231_231431

theorem complex_algebra (a b : ℝ) (i : ℂ) (hi : i = complex.I) (h : (a + i) / i = 1 + b * i) :
  a + b = 0 :=
sorry

end complex_algebra_l231_231431


namespace solution_to_inequality_l231_231479

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x
  else x^2

theorem solution_to_inequality (x : ℝ) :
  (f(x^2) > f(3 - 2x)) ↔ (x ∈ Iio (-3) ∪ Ioo 1 3) := by
  sorry

end solution_to_inequality_l231_231479


namespace gcd_12345_6789_l231_231234

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l231_231234


namespace GCD_of_set_B_is_2_l231_231948

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231948


namespace gcd_12345_6789_l231_231233

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l231_231233


namespace original_revenue_l231_231358

theorem original_revenue (current_revenue : ℝ) (percentage_decrease : ℝ) (original_revenue : ℝ) : 
  current_revenue = 42.0 ∧ percentage_decrease = 39.130434782608695 / 100 →
  original_revenue = current_revenue / (1 - percentage_decrease) →
  original_revenue ≈ 68.97 := 
by sorry

end original_revenue_l231_231358


namespace A_inf_l231_231465

noncomputable def f : ℝ → ℝ := sorry -- Specify that f is noncomputable

def A : set ℝ := {a | f a > a^2}

theorem A_inf (h1 : ∀ x : ℝ, (f x)^2 ≤ 2 * x^2 * f (x / 2)) (h2 : A.nonempty) : set.infinite A :=
by
  sorry

end A_inf_l231_231465


namespace pascal_fifth_number_in_row_15_l231_231632

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231632


namespace fifth_number_in_pascals_triangle_l231_231602

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231602


namespace no_valid_triple_exists_l231_231408

theorem no_valid_triple_exists :
  ∀ (a b c : ℤ), a ≥ 2 → b ≥ 3 → c ≥ 1 → (log a b = c^3) → (a + b + c = 3000) → false :=
by
  sorry

end no_valid_triple_exists_l231_231408


namespace total_number_of_red_and_white_jelly_beans_in_fishbowl_l231_231821

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end total_number_of_red_and_white_jelly_beans_in_fishbowl_l231_231821


namespace evaluate_expression_l231_231404

theorem evaluate_expression :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end evaluate_expression_l231_231404


namespace polynomial_simplification_l231_231173

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) = 
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 :=
by
  sorry

end polynomial_simplification_l231_231173


namespace pascal_fifth_element_15th_row_l231_231712

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231712


namespace Pascal_triangle_fifth_number_l231_231662

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231662


namespace pascal_fifteen_four_l231_231792

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231792


namespace pascal_fifteen_four_l231_231781

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231781


namespace function_range_is_minus_two_to_zero_l231_231208

noncomputable def function_range : set ℝ :=
{y : ℝ | ∃ x : ℝ, y = 2 * sin x * cos x - 1}

theorem function_range_is_minus_two_to_zero : function_range = set.Icc (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end function_range_is_minus_two_to_zero_l231_231208


namespace point_in_second_quadrant_l231_231570

noncomputable def complex_number := (5 * Complex.I) / (2 - Complex.I)

theorem point_in_second_quadrant : 
  let z := complex_number in z.re < 0 ∧ z.im > 0 :=
by
  sorry

end point_in_second_quadrant_l231_231570


namespace fifth_number_in_pascals_triangle_l231_231603

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231603


namespace problem_statements_l231_231481

def f (x : ℝ) : ℝ := | Real.cos x | * Real.sin x

theorem problem_statements :
  (f (2014 * Real.pi / 3) = - Real.sqrt 3 / 4) ∧
  ∀ x1 x2, (| f x1 | = | f x2 | → (x1 = x2 + k * Real.pi ∧ k ∈ ℤ)) ∨
  (∀ x, x ∈ Icc (- Real.pi / 4) (Real.pi / 4) → f.deriv x >= 0) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x, f (- x - Real.pi / 2) = f x) :=
sorry

end problem_statements_l231_231481


namespace pascal_row_fifth_number_l231_231577

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231577


namespace count_multiples_6_or_8_not_both_l231_231512

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231512


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231969

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231969


namespace gcd_B_eq_two_l231_231857

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231857


namespace compare_AC_and_CD_CB_l231_231474

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 2) = 1
def right_vertex_C : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def is_symmetric (p q : ℝ × ℝ) : Prop := p.1 = -q.1 ∧ p.2 = -q.2
def perpendicular_to_x_axis (A D : ℝ × ℝ) : Prop := D.1 = A.1 ∧ D.2 = 0
def line_eq (A B D : ℝ × ℝ) : Prop :=
  D.2 = ((B.2 - right_vertex_C.2) / (B.1 - right_vertex_C.1)) * (D.1 - right_vertex_C.1) + right_vertex_C.2

-- Main theorem statement
theorem compare_AC_and_CD_CB (A B D C : ℝ × ℝ) (α : ℝ)
  (hA_on_ellipse : ellipse A.1 A.2)
  (hC_eq_right_vertex : C = right_vertex_C)
  (hB_sym_to_A : is_symmetric A B)
  (hD_perpendicular : perpendicular_to_x_axis A D)
  (hD_on_BC : line_eq A B D)
  : (A.1 - C.1)^2 + (A.2 - C.2)^2 < ((D.1 - C.1)^2 + (D.2 - C.2)^2) * ((B.1 - C.1)^2 + (B.2 - C.2)^2) := 
  sorry

end compare_AC_and_CD_CB_l231_231474


namespace pascal_triangle_fifth_number_l231_231674

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231674


namespace pascal_triangle_fifth_number_l231_231761

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231761


namespace dodecahedron_interior_diagonals_l231_231046

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231046


namespace probability_valid_arrangement_l231_231439

theorem probability_valid_arrangement : 
  let n := 3;
  let C3 := (Nat.Catalan n);
  let totalArrangements := (Nat.choose (2 * n) n);
  let validProbability := C3 / totalArrangements;
  validProbability = 1 / 4 :=
by
  sorry

end probability_valid_arrangement_l231_231439


namespace num_integers_satisfying_inequality_l231_231502

theorem num_integers_satisfying_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.finite.toFinset.card = 15 :=
by
  sorry

end num_integers_satisfying_inequality_l231_231502


namespace find_a_l231_231076

theorem find_a (a : ℝ) : (∀ (x : ℝ), -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  assume h : ∀ (x : ℝ), -1 < x ∧ x < 2 ↔ |a * x + 2| < 6
  sorry

end find_a_l231_231076


namespace pascal_fifth_number_in_row_15_l231_231633

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231633


namespace fifth_number_in_pascals_triangle_l231_231600

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231600


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231972

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231972


namespace greatest_two_digit_product_12_l231_231279

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231279


namespace pascal_fifth_element_15th_row_l231_231714

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231714


namespace pascal_triangle_fifth_number_l231_231646

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231646


namespace dodecahedron_interior_diagonals_l231_231052

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231052


namespace pascal_15_5th_number_l231_231770

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231770


namespace pascal_triangle_15_4_l231_231618

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231618


namespace greatest_common_divisor_of_B_l231_231922

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231922


namespace dodecahedron_interior_diagonals_l231_231053

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231053


namespace greatest_two_digit_prod_12_l231_231285

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231285


namespace pascal_row_fifth_number_l231_231574

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231574


namespace inscribed_circle_radius_eq_four_l231_231095

theorem inscribed_circle_radius_eq_four
  (A p s r : ℝ)
  (hA : A = 2 * p)
  (hp : p = 2 * s)
  (hArea : A = r * s) :
  r = 4 :=
by
  -- Proof would go here.
  sorry

end inscribed_circle_radius_eq_four_l231_231095


namespace parabola_focus_l231_231187

theorem parabola_focus :
  let y_eq := λ x : ℝ, 2 * x^2 in
  (∃ (focus_x focus_y : ℝ), (focus_x = 0 ∧ focus_y = 1/8)) :=
by
  let y_eq : ℝ → ℝ := λ x, 2 * x^2
  use 0, 1/8
  split
  sorry

end parabola_focus_l231_231187


namespace gcd_of_sum_of_four_consecutive_integers_l231_231871

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231871


namespace gcd_of_B_is_2_l231_231854

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231854


namespace average_speed_l231_231152

theorem average_speed (D : ℝ) :
  let time_by_bus := D / 80
  let time_walking := D / 16
  let time_cycling := D / 120
  let total_time := time_by_bus + time_walking + time_cycling
  let total_distance := 2 * D
  total_distance / total_time = 24 := by
  sorry

end average_speed_l231_231152


namespace pascal_triangle_fifth_number_l231_231690

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231690


namespace imaginary_part_l231_231441

theorem imaginary_part (z : ℂ) (h : z / (2 - complex.I) = complex.I) : z.im = 2 :=
sorry

end imaginary_part_l231_231441


namespace count_bad_arrangements_l231_231201

def is_bad_arrangement (arr : list ℕ) : Prop :=
  ¬ (∀ n ∈ (list.range 21).map (+1), ∃ sublist : list ℕ,
    (selections sublist arr) ∧ (sublist.sum = n))

def are_equivalent (arr1 arr2 : list ℕ) : Prop :=
  ∃ k, (arr2 = rotate_arr arr1 k) ∨ (arr2 = rotate_arr (reverse_arr arr1) k)

theorem count_bad_arrangements : ∃ (bad_arrangements : set (list ℕ)), 
  (∀ arr ∈ bad_arrangements, is_bad_arrangement arr) ∧
  (∃ (unique_bad_arrangements : set (list ℕ)),
    ∀ arr1 arr2 ∈ unique_bad_arrangements,
    are_equivalent arr1 arr2 → arr1 = arr2 ∧ 
    unique_bad_arrangements.card = 3) := sorry

end count_bad_arrangements_l231_231201


namespace f_neg_expression_l231_231983

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then x^2 - 2*x + 3 else sorry

-- Define f by cases: for x > 0 and use the property of odd functions to conclude the expression for x < 0.

theorem f_neg_expression (x : ℝ) (h : x < 0) : f x = -x^2 - 2*x - 3 :=
by
  sorry

end f_neg_expression_l231_231983


namespace probability_of_sum_17_l231_231086

noncomputable def prob_sum_dice_is_seventeen : ℚ :=
1 / 72

theorem probability_of_sum_17 :
  let dice := finset.product (finset.product finset.univ finset.univ) finset.univ in
  let event := dice.filter (λ (x : ℕ × (ℕ × ℕ)), x.1 + x.2.1 + x.2.2 = 17) in
  (event.card : ℚ) / (dice.card : ℚ) = prob_sum_dice_is_seventeen :=
by
  sorry

end probability_of_sum_17_l231_231086


namespace sum_of_coordinates_of_X_l231_231124

theorem sum_of_coordinates_of_X 
  (X Y Z : ℝ × ℝ)
  (h1 : dist X Z / dist X Y = 1 / 2)
  (h2 : dist Z Y / dist X Y = 1 / 2)
  (hY : Y = (1, 7))
  (hZ : Z = (-1, -7)) :
  (X.1 + X.2) = -24 :=
sorry

end sum_of_coordinates_of_X_l231_231124


namespace f_g_minus_g_f_l231_231178

def f (x : ℝ) : ℝ := 8 * x - 12
def g (x : ℝ) : ℝ := x / 4 + 3

theorem f_g_minus_g_f (x : ℝ) : f (g x) - g (f x) = 12 := 
by sorry

end f_g_minus_g_f_l231_231178


namespace GCD_of_set_B_is_2_l231_231944

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231944


namespace count_valid_quads_eq_five_l231_231417

open Nat

-- Define the conditions
def is_valid_quad (a b c d : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧
  ∃ k, (ab = k) ∧ (cd = k) ∧ (k = a + b + c + d - 3)

-- Define the theorem to prove the number of valid quadruples
theorem count_valid_quads_eq_five : 
  {quad | ∃ a b c d, is_valid_quad a b c d} = 5 :=
by
  sorry

end count_valid_quads_eq_five_l231_231417


namespace greatest_two_digit_product_is_12_l231_231244

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231244


namespace find_num_roots_l231_231451

variable {f : ℝ → ℝ}

noncomputable def num_roots (a b : ℝ) (f : ℝ → ℝ) := (finset.Icc a b).filter (λ x, f x = 0).card

theorem find_num_roots
  (hx : ∀ x, f (-x) = -f x)
  (hper : ∀ x, f (x + 4) = f x)
  (h3 : f 3 = 0) :
  num_roots 0 10 f = 11 :=
sorry

end find_num_roots_l231_231451


namespace range_of_x_l231_231464

variable {m x : ℝ}

theorem range_of_x
  (h : ∀ m : ℝ, m ≠ 0 → |2 * m - 1| + |1 - m| ≥ |m| * (|x - 1| - |2 * x + 3|)) :
  x ∈ set.Iic (-3) ∪ set.Ici (-1) :=
sorry

end range_of_x_l231_231464


namespace castle_lego_ratio_l231_231119

def total_legos : ℕ := 500
def legos_put_back : ℕ := 245
def legos_missing : ℕ := 5
def legos_used : ℕ := total_legos - legos_put_back - legos_missing
def ratio (a b : ℕ) : ℚ := a / b

theorem castle_lego_ratio : ratio legos_used total_legos = 1 / 2 :=
by
  unfold ratio legos_used total_legos legos_put_back legos_missing
  norm_num

end castle_lego_ratio_l231_231119


namespace graph_n_plus_k_odd_l231_231566

-- Definitions and assumptions
variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable (n k : ℕ)
variable (hG : Fintype.card V = n)
variable (hCond : ∀ (S : Finset V), S.card = k → (G.commonNeighborsFinset S).card % 2 = 1)

-- Goal
theorem graph_n_plus_k_odd :
  (n + k) % 2 = 1 :=
sorry

end graph_n_plus_k_odd_l231_231566


namespace pascal_fifth_element_15th_row_l231_231700

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231700


namespace trajectory_and_coordinates_l231_231569

-- Define the conditions and the problem statement
theorem trajectory_and_coordinates :
  (∀ (x y : ℝ), ((y / (x + 2)) * (y / (x - 2)) = -3 / 4 → (x^2 / 4 + y^2 / 3 = 1)) ∧
  (∀ (x1 y1 x2 y2 m : ℝ), (-- Intersection points and areas
    (x1^2 / 4 + y1^2 / 3 = 1) ∧ (y1 = x1 - 1) ∧ 
    (x2^2 / 4 + y2^2 / 3 = 1) ∧ (y2 = x2 - 1) ∧ 
    ((1 / 2) * (24 / 7) * (abs (m - 1) / (sqrt 2)) = 6 * sqrt 2) 
    → (m = 8 ∨ m = -6))) :=
begin
  sorry
end

end trajectory_and_coordinates_l231_231569


namespace lights_all_off_after_one_round_l231_231212

theorem lights_all_off_after_one_round :
  ∀ (n : ℕ) (rooms : fin n → bool),
  n = 20 →
  (∃ i : fin n, i = 0 ∧ rooms i = tt) →
  (∃ k : ℕ, k = 10 ∧ (∃ on_set off_set : fin n → bool, 
    (∀ i, rooms i = tt → on_set i = true) ∧ 
    (∀ i, rooms i = ff → off_set i = true) ∧ 
    (card (finset.univ.filter on_set) = k) ∧ 
    (card (finset.univ.filter off_set) = k))) →
  (∀ i : fin n, 
    let remaining_on := card (finset.univ.filter (λ j, j ≠ i ∧ rooms j = tt)),
        remaining_off := card (finset.univ.filter (λ j, j ≠ i ∧ rooms j = ff)) in 
    (remaining_on < remaining_off) → rooms i = ff) →
  (∀ i : fin n, rooms i = ff) := 
by {
  sorry
}

end lights_all_off_after_one_round_l231_231212


namespace pascal_triangle_fifth_number_l231_231676

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231676


namespace greatest_two_digit_product_12_l231_231278

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231278


namespace pascal_triangle_row_fifth_number_l231_231731

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231731


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231974

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231974


namespace pascal_15_5th_number_l231_231769

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231769


namespace magnitude_w_eq_2_l231_231985

noncomputable def s : ℝ := sorry
noncomputable def w : ℂ := sorry

def abs_s_lt_3 (s : ℝ) : Prop := |s| < 3
def condition_w_s (w : ℂ) (s : ℝ) : Prop := w + 2 / w = s

theorem magnitude_w_eq_2 (h_abs_s : abs_s_lt_3 s) (h_w_condition : condition_w_s w s) : |w| = 2 := 
sorry

end magnitude_w_eq_2_l231_231985


namespace linear_function_not_in_fourth_quadrant_l231_231425

theorem linear_function_not_in_fourth_quadrant (b : ℝ) (h : b ≥ 0) :
  ∀ x : ℝ, 2 * x + b < 0 → x ≤ 0 :=
begin
  sorry,
end

end linear_function_not_in_fourth_quadrant_l231_231425


namespace gcd_times_xyz_is_square_l231_231135

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_times_xyz_is_square_l231_231135


namespace find_ab_squared_l231_231989

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

theorem find_ab_squared (a b : ℝ) (h1 : f(a) = 1) (h2 : f(b) = 19) : (a + b)^2 = 4 := by
  sorry

end find_ab_squared_l231_231989


namespace greatest_common_divisor_of_B_l231_231900

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231900


namespace min_value_frac_2_over_a_plus_3_over_b_l231_231444

theorem min_value_frac_2_over_a_plus_3_over_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end min_value_frac_2_over_a_plus_3_over_b_l231_231444


namespace angle_between_vectors_eq_l231_231412

-- Vectors u and v defined as given in the conditions of the problem
def u : ℝ × ℝ × ℝ := (3, -2, 2)
def v : ℝ × ℝ × ℝ := (2, -3, 1)

-- Proof statement: to prove the angle θ (in degrees) between vector u and vector v is 
-- that arccos of the given value
theorem angle_between_vectors_eq :
  let dot_product := (3 * 2) + (-2 * -3) + (2 * 1)
      norm_u := real.sqrt ((3:ℝ)^2 + (-2)^2 + (2)^2)
      norm_v := real.sqrt ((2:ℝ)^2 + (-3)^2 + (1)^2)
      cos_theta := dot_product / (norm_u * norm_v)
  real.arccos cos_theta = real.arccos (real.sqrt 14 / real.sqrt 17) := 
by
  sorry

end angle_between_vectors_eq_l231_231412


namespace central_angles_l231_231334

-- Define the conditions given in the problem
def prob_region_A : ℝ := 1 / 8
def prob_region_B : ℝ := 1 / 12

-- Define what we want to prove
theorem central_angles :
  ∃ (θ_A θ_B : ℝ), 
  (prob_region_A = θ_A / 360) ∧ 
  (prob_region_B = θ_B / 360) ∧ 
  (θ_A = 45) ∧ 
  (θ_B = 30) := 
by {
  -- Construct the values that satisfy the conditions
  use [45, 30],
  -- Prove each part of the conjunction
  split; norm_num,
}

end central_angles_l231_231334


namespace dodecahedron_interior_diagonals_l231_231004

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231004


namespace find_x_for_A_l231_231831

theorem find_x_for_A (x : ℝ) : 
  let A := {-3, x + 2, x^2 - 4 * x} in
  5 ∈ A ↔ (x = -1 ∨ x = 5) :=
by
  let A := {-3, x + 2, x^2 - 4 * x}
  sorry

end find_x_for_A_l231_231831


namespace total_volume_removed_prisms_l231_231387

/-- Let the initial prism be a rectangular prism with dimensions 2 units by 2 units by 3 units.
    After slicing off the corners to transform each of the larger faces into regular hexagons,
    the total volume of the removed parts (small pyramids) is calculated. -/
theorem total_volume_removed_prisms {a b c : ℝ} (ha : a = 2) (hb : b = 2) (hc : c = 3) :
  let total_volume_removed := 8 * (1/3 * (sqrt 3 / 9) * 2)
  in total_volume_removed = 16 * sqrt 3 / 27 :=
by sorry

end total_volume_removed_prisms_l231_231387


namespace two_a_sq_minus_six_b_plus_one_l231_231064

theorem two_a_sq_minus_six_b_plus_one (a b : ℝ) (h : a^2 - 3 * b = 5) : 2 * a^2 - 6 * b + 1 = 11 := by
  sorry

end two_a_sq_minus_six_b_plus_one_l231_231064


namespace dodecahedron_interior_diagonals_l231_231036

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231036


namespace pascal_triangle_15_4_l231_231613

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231613


namespace find_xyz_value_l231_231177

noncomputable def xyz_value (x y z : ℝ) : ℝ :=
  x * y * z

theorem find_xyz_value (x y z : ℝ) (h1 : x * y = 27 * Real.cbrt 3) (h2 : x * z = 45 * Real.cbrt 3) (h3 : y * z = 18 * Real.cbrt 3) (h4 : x = 2 * y) :
  xyz_value x y z = 108 * Real.sqrt 3 :=
by
  sorry

end find_xyz_value_l231_231177


namespace courses_selection_l231_231160

theorem courses_selection :
  let choose_two_out_of_four := Nat.choose 4 2 in
  (choose_two_out_of_four * choose_two_out_of_four = 36) ∧
  ((choose_two_out_of_four * choose_two_out_of_four - choose_two_out_of_four) / (choose_two_out_of_four * choose_two_out_of_four) = 5 / 6) :=
by
  sorry

end courses_selection_l231_231160


namespace gcd_elements_of_B_l231_231963

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231963


namespace gcd_of_sum_of_four_consecutive_integers_l231_231870

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231870


namespace polynomial_coefficient_sum_equality_l231_231139

theorem polynomial_coefficient_sum_equality :
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
    (∀ x : ℝ, (2 * x + 1)^4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
    (a₀ - a₁ + a₂ - a₃ + a₄ = 1) :=
by
  intros
  sorry

end polynomial_coefficient_sum_equality_l231_231139


namespace man_half_father_age_in_years_l231_231342

theorem man_half_father_age_in_years
  (M F Y : ℕ) 
  (h1: M = (2 * F) / 5) 
  (h2: F = 25) 
  (h3: M + Y = (F + Y) / 2) : 
  Y = 5 := by 
  sorry

end man_half_father_age_in_years_l231_231342


namespace limit_n_b_n_l231_231390

noncomputable def M (x : ℝ) : ℝ := x - x^2 / 3

def b_n (n : ℕ) : ℝ :=
  let rec iterate (k : ℕ) (y : ℝ) : ℝ :=
    if k = 0 then y else iterate (k - 1) (M y)
  iterate n (23 / n)

theorem limit_n_b_n : 
  tendsto (fun n => n * b_n n) at_top (𝓝 (69 / 20)) := by
  sorry

end limit_n_b_n_l231_231390


namespace squirrel_group_exists_l231_231399

theorem squirrel_group_exists :
  ∃ (group : Finset ℕ), group.card = 3 ∧ 
    ∀ (i j : ℕ), i ∈ group → j ∈ group → (i ≠ j → ¬ (throws_cone_at i j)) :=
sorry

end squirrel_group_exists_l231_231399


namespace watermelon_weights_total_l231_231150

theorem watermelon_weights_total :
  let M := 12 in
  let C := 1.5 * M in
  let J := 0.5 * C in
  let E := 0.75 * J in
  let S := max (E + 3) (2 * M) in
  M + C + J + E + S = 69.75 :=
by {
  sorry
}

end watermelon_weights_total_l231_231150


namespace pascal_triangle_row_fifth_number_l231_231728

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231728


namespace multiples_of_6_or_8_under_201_not_both_l231_231524

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231524


namespace number_above_157_is_133_l231_231295

theorem number_above_157_is_133 :
  ∀ (k : ℕ), k * k ≥ 157 ∧ (k - 1) * (k - 1) < 157 →
  let row_start := (k - 1) * (k - 1) + 1 in
  let pos_157 := 157 - row_start + 1 in
  157 ∈ range (row_start + pos_157 - 1) →
  let prev_row_start := (k - 2) * (k - 2) + 1 in
  let pos_above_157 := prev_row_start + pos_157 - 1 in
  pos_above_157 = 133 :=
begin
  intros k hk row_start pos_157 h157 prev_row_start pos_above_157,
  sorry
end

end number_above_157_is_133_l231_231295


namespace pascal_fifteen_four_l231_231788

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231788


namespace gcd_elements_of_B_l231_231961

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231961


namespace trace_bag_weight_l231_231222

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l231_231222


namespace factorize_a3_minus_4ab2_l231_231407

theorem factorize_a3_minus_4ab2 (a b : ℝ) : a^3 - 4 * a * b^2 = a * (a + 2 * b) * (a - 2 * b) :=
by
  -- Proof is omitted; write 'sorry' as a placeholder
  sorry

end factorize_a3_minus_4ab2_l231_231407


namespace max_num_triangles_for_right_triangle_l231_231156

-- Define a right triangle on graph paper
def right_triangle (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n

-- Define maximum number of triangles that can be formed within the triangle
def max_triangles (n : ℕ) : ℕ :=
  if h : n = 7 then 28 else 0  -- Given n = 7, the max number is 28

-- Define the theorem to be proven
theorem max_num_triangles_for_right_triangle :
  right_triangle 7 → max_triangles 7 = 28 :=
by
  intro h
  -- Proof goes here
  sorry

end max_num_triangles_for_right_triangle_l231_231156


namespace dodecahedron_interior_diagonals_l231_231050

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231050


namespace gcd_of_B_is_2_l231_231835

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231835


namespace f_derivative_l231_231073

noncomputable def f (x : ℝ) : ℝ := 2*x^3 + 5*x + c

theorem f_derivative (c : ℝ) : (λ x : ℝ, 6*x^2 + 5) = (λ x : ℝ, (f x).derivative) :=
by {
    sorry
}

end f_derivative_l231_231073


namespace log_inequality_l231_231163

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem log_inequality (a b c : ℝ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : 2 ≤ c) :
  log_base (b + c) (a^2) + log_base (a + c) (b^2) + log_base (a + b) (c^2) ≥ 3 :=
  by
  sorry

end log_inequality_l231_231163


namespace range_of_x_f_greater_f_2x_minus_1_l231_231140

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.log (Real.sqrt (x^2 + 1) + x) + x^2 - x * Real.sin x

theorem range_of_x_f_greater_f_2x_minus_1 :
  {x : ℝ | f x > f (2 * x - 1)} = set.Ioo (1 / 3 : ℝ) 1 :=
by
  sorry

end range_of_x_f_greater_f_2x_minus_1_l231_231140


namespace pascal_15_5th_number_l231_231777

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231777


namespace area_nine_times_l231_231813

theorem area_nine_times 
  (AB AC BC : ℝ)
  (h1 : AB = 15)
  (h2 : AC = 9)
  (h3 : BC = 13)
  (AB' AC' BC' : ℝ)
  (h4 : AB' = 3 * AB)
  (h5 : AC' = 3 * AC)
  (h6 : BC' = BC):
  let areaABC := sorry in
  let areaA'B'C' := sorry in
  areaA'B'C' = 9 * areaABC := sorry

end area_nine_times_l231_231813


namespace gcd_12345_6789_eq_3_l231_231230

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l231_231230


namespace sequence_general_formula_specific_values_a1_specific_values_a2_specific_values_a3_l231_231446

-- Defining the sequence {a_n} and its sum S_n
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := (a n ^ 2 - 2 * a n + 2) / (2 * a n)

-- Conditions: sequence {a_n} such that S_n = (a_n^2 - 2a_n + 2) / (2a_n)
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n a = (a n ^ 2 - 2 * a n + 2) / (2 * a n)

-- Proving that a_n = sqrt(2n+1) - sqrt(2n-1) for all n in N_+
theorem sequence_general_formula (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) :=
sorry

-- Verifying specific values for a_1, a_2, a_3
theorem specific_values_a1 (a : ℕ → ℝ) (h : sequence a) : a 1 = Real.sqrt 3 - 1 :=
sorry

theorem specific_values_a2 (a : ℕ → ℝ) (h : sequence a) : a 2 = Real.sqrt 5 - Real.sqrt 3 :=
sorry

theorem specific_values_a3 (a : ℕ → ℝ) (h : sequence a) : a 3 = Real.sqrt 7 - Real.sqrt 5 :=
sorry

end sequence_general_formula_specific_values_a1_specific_values_a2_specific_values_a3_l231_231446


namespace pascal_triangle_fifth_number_l231_231672

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231672


namespace length_of_AB_l231_231104

noncomputable def isosceles_triangle (A B C : ℝ) : Prop := A = B ∨ B = C ∨ C = A

variables (A B C D : ℝ)
variables (h1 : isosceles_triangle A B C)
variables (h2 : ∠A = 2 * ∠B)
variables (h3 : A + B + C = 26)
variables (h4 : B + C + D = 21)
variables (h5 : D = 9)

theorem length_of_AB : A = 10 :=
sorry

end length_of_AB_l231_231104


namespace dodecahedron_interior_diagonals_l231_231021

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231021


namespace infinite_almost_square_quadruples_l231_231229

def almost_square (n : ℕ) : Prop :=
  ∃ a b : ℕ, (n = a * b) ∧ (|a - b| ≤ 1)

theorem infinite_almost_square_quadruples :
  ∃ᶠ (m : ℕ) in at_top, ∀ k ∈ ({4 * m^4 - 1, 4 * m^4, 4 * m^4 + 1, 4 * m^4 + 2} : set ℕ), almost_square k :=
by {
  sorry
}

end infinite_almost_square_quadruples_l231_231229


namespace gcd_12345_6789_eq_3_l231_231232

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l231_231232


namespace sin_sum_to_product_l231_231406

theorem sin_sum_to_product (x : Real) : 
  sin (3 * x) + sin (9 * x) = 2 * sin (6 * x) * cos (3 * x) :=
sorry

end sin_sum_to_product_l231_231406


namespace greatest_two_digit_prod_12_l231_231281

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231281


namespace dodecahedron_interior_diagonals_l231_231006

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231006


namespace pascal_triangle_fifth_number_l231_231697

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231697


namespace correct_calculation_l231_231296

theorem correct_calculation : (-3) + (-9) = -12 :=
by
  have hA : ¬ (0 - (-5) = -5) := by linarith
  have hB : ¬ (-2 / (1/3) * 3 = -2) := by field_simp; linarith
  have hC : ¬ ((-1/5) / 5 = -1) := by field_simp; linarith
  have hD : (-3) + (-9) = -12 := by linarith
  exact hD

end correct_calculation_l231_231296


namespace fifth_number_in_pascals_triangle_l231_231591

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231591


namespace inequality_not_always_holds_l231_231435

theorem inequality_not_always_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  ¬(∀ a b, (a > 0) → (b > 0) → (a + b = 2) → sqrt a + sqrt b ≤ sqrt 2) :=
begin
  intro h,
  have h1 := h 1 1,
  specialize h1 (by norm_num) (by norm_num) (by norm_num),
  norm_num at h1,
  exact not_le_of_gt zero_lt_two h1,
end

end inequality_not_always_holds_l231_231435


namespace count_valid_N_l231_231057

theorem count_valid_N :
  {N : ℕ | N > 300 ∧
          ((1000 ≤ 4 * N ∧ 4 * N < 10000) +
           (1000 ≤ N - 300 ∧ N - 300 < 10000) +
           (1000 ≤ N + 45 ∧ N + 45 < 10000) +
           (1000 ≤ 2 * N ∧ 2 * N < 10000)).card = 2}.card = 5410 := sorry

end count_valid_N_l231_231057


namespace incorrect_statement_C_l231_231302

/-- Define the propositions involved in statement C --/
variables (p q : Prop)

-- Prove that statement C is incorrect
theorem incorrect_statement_C : ¬ ((¬ p ∧ ¬ q) → (¬ p ∧ ¬ q)) :=
sorry

end incorrect_statement_C_l231_231302


namespace r_of_r_r_r_of_15_l231_231991

def r (θ : ℝ) : ℝ := 1 / (1 - θ)

theorem r_of_r_r_r_of_15 : r (r (r (r (15)))) = -1 / 14 := 
by 
  sorry

end r_of_r_r_r_of_15_l231_231991


namespace pascal_15_5th_number_l231_231768

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231768


namespace tetrahedron_volume_correct_l231_231812

def volume_of_tetrahedron {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : A) (CD : C) (AB_EQ : distance AB = 1) (CD_EQ : distance CD = sqrt 3)
  (DISTANCE_BETWEEN : distance_between_lines AB CD = 2)
  (ANGLE_BETWEEN : angle_between_lines AB CD = π / 3) : ℝ :=
  1 / 2

theorem tetrahedron_volume_correct :
  ∀ (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB : A) (CD : C) (AB_EQ : distance AB = 1) (CD_EQ : distance CD = sqrt 3)
  (DISTANCE_BETWEEN : distance_between_lines AB CD = 2)
  (ANGLE_BETWEEN : angle_between_lines AB CD = π / 3),
  volume_of_tetrahedron AB CD AB_EQ CD_EQ DISTANCE_BETWEEN ANGLE_BETWEEN = 1 / 2 :=
by
  intros
  sorry

end tetrahedron_volume_correct_l231_231812


namespace find_x_if_perpendicular_l231_231492

-- Define vectors a and b in the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x - 5, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The Lean theorem statement equivalent to the math problem
theorem find_x_if_perpendicular (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 2 := by
  sorry

end find_x_if_perpendicular_l231_231492


namespace fifth_number_in_pascals_triangle_l231_231601

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231601


namespace dodecahedron_interior_diagonals_l231_231003

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231003


namespace sum_binary_flip_leq_l231_231421

def binary_flip (n : ℕ) : ℕ :=
  let binary_str := Nat.toDigits 2 n
  let flipped_str := binary_str.map (λ d, if d = 0 then 1 else 0)
  Nat.ofDigits 2 flipped_str

theorem sum_binary_flip_leq (n : ℕ) (n_pos : n > 0) : 
  ∑ k in Finset.range n.succ, binary_flip k ≤ n * n / 4 :=
by
  sorry

end sum_binary_flip_leq_l231_231421


namespace circle_radius_l231_231333

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l231_231333


namespace num_integers_satisfying_inequality_l231_231500

theorem num_integers_satisfying_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.finite.toFinset.card = 15 :=
by
  sorry

end num_integers_satisfying_inequality_l231_231500


namespace equal_probability_selection_l231_231429

theorem equal_probability_selection :
  ∀ (n m k : ℕ), n = 2010 → m = 2000 → k = 50 →
  (∀ i : ℕ, i < m → (choose n i / choose n m) = (k / m)) →
  ∀ j : ℕ, j < n → (choose (n - 10) k / choose n k) = 5 / 201 :=
by
  sorry

end equal_probability_selection_l231_231429


namespace three_digit_number_second_digit_l231_231356

theorem three_digit_number_second_digit (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (100 * a + 10 * b + c) - (a + b + c) = 261 → b = 7 :=
by sorry

end three_digit_number_second_digit_l231_231356


namespace dodecahedron_interior_diagonals_l231_231013

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231013


namespace fifth_number_in_pascal_row_l231_231803

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231803


namespace greatest_two_digit_with_product_12_l231_231266

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231266


namespace fraction_base12_l231_231397

-- Define the conversion from base-12 to base-10 for specific values
def base12_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 10 => 12
  | _  => n

-- Define the conversion from base-10 to base-12 for specific values
def base10_to_base12 (n : ℕ) : ℕ :=
  match n with
  | 12 => 10
  | n  => n

theorem fraction_base12 : 
  (1 / 4 * 20 = 6) →
  (1 / 5 * base12_to_base10 10 = 2.4) →
  base10_to_base12 (floor (2.4)) = 2 ∧ base10_to_base12 (modf (2.4)) = 2 / 5 :=
by
    sorry

end fraction_base12_l231_231397


namespace N_plus_K_l231_231384

noncomputable def P (z : ℂ) : ℂ := z^4 + 6 * z^3 + 2 * z^2 + 4 * z + 1

def g (z : ℂ) : ℂ := 2 * complex.I * complex.conj z

-- Roots of the polynomial P
variables (z1 z2 z3 z4 : ℂ)
-- Assuming z1, z2, z3, z4 are roots of P
axiom roots_P : ∀ z, P z = 0 ↔ (z = z1 ∨ z = z2 ∨ z = z3 ∨ z = z4)

-- Transformed polynomial R
def R (z : ℂ) : ℂ := z^4 + (6 * complex.I) * z^3 + (-8 : ℂ) * z^2 + (4 * complex.I) * z + 16

-- N and K (coefficients of z^2 and constant term in R respectively)
def N : ℂ := -8  -- As calculated from the transformation
def K : ℂ := 16  -- As calculated from the transformation

theorem N_plus_K : N + K = 8 := by
  sorry

end N_plus_K_l231_231384


namespace dodecahedron_interior_diagonals_l231_231022

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231022


namespace pascal_triangle_row_fifth_number_l231_231720

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231720


namespace total_history_and_maths_l231_231558

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end total_history_and_maths_l231_231558


namespace dodecahedron_interior_diagonals_l231_231033

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231033


namespace greatest_common_divisor_of_B_l231_231893

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231893


namespace pascal_triangle_row_fifth_number_l231_231717

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231717


namespace polynomial_remainder_l231_231290

noncomputable def p : Polynomial ℤ := 3 * X^4 + 13 * X^3 + 5 * X^2 - 10 * X + 20
noncomputable def d : Polynomial ℤ := X^2 + 5 * X + 1
noncomputable def r : Polynomial ℤ := -68 * X + 8

theorem polynomial_remainder :
  ∃ q : Polynomial ℤ, p = d * q + r ∧ r.degree < d.degree :=
sorry

end polynomial_remainder_l231_231290


namespace arrange_pieces_on_chessboard_l231_231817

-- Definitions of chessboard dimensions and piece counts
def chessboard_size : Nat := 8
def white_pieces : Nat := 16
def black_pieces : Nat := 16

-- Define what it means to be a neighbor on a chessboard
def is_neighbor (x1 y1 x2 y2 : Nat) : Prop := 
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ 
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1)) ∨
  (x1 = x2 + 1 ∨ x1 = x2 - 1 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1))

-- Problem statement
theorem arrange_pieces_on_chessboard :
  ∃ (W B : Fin 64 → bool), 
    (∑ i, if W i then 1 else 0) = white_pieces ∧ 
    (∑ i, if B i then 1 else 0) = black_pieces ∧ 
    (∀ i, (∑ j, if is_neighbor i.1 i.2 j.1 j.2 ∧ W j then 1 else 0) = 
           (∑ j, if is_neighbor i.1 i.2 j.1 j.2 ∧ B j then 1 else 0)) :=
sorry

end arrange_pieces_on_chessboard_l231_231817


namespace find_common_ratio_l231_231471

noncomputable def a_n (n : ℕ) (q : ℚ) : ℚ :=
  if n = 1 then 1 / 8 else (q^(n - 1)) * (1 / 8)

theorem find_common_ratio (q : ℚ) :
  (a_n 4 q = -1) ↔ (q = -2) :=
by
  sorry

end find_common_ratio_l231_231471


namespace isosceles_triangle_side_length_l231_231367

noncomputable def equilateral_triangle_side_length : ℝ := 2

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s^2

noncomputable def isosceles_triangle_area (A : ℝ) : ℝ :=
  A / 4

noncomputable def isosceles_triangle_height (base : ℝ) (area : ℝ) : ℝ :=
  2 * area / base

noncomputable def isosceles_triangle_congruent_side (base : ℝ) (height : ℝ) : ℝ :=
  real.sqrt (base/2)^2 + height^2

theorem isosceles_triangle_side_length
  (s : ℝ) (A_equilateral : ℝ) (A_isosceles : ℝ) (base : ℝ) (height : ℝ) (side : ℝ) :
  s = equilateral_triangle_side_length →
  A_equilateral = equilateral_triangle_area s →
  A_isosceles = isosceles_triangle_area A_equilateral →
  base = s →
  height = isosceles_triangle_height base A_isosceles →
  side = isosceles_triangle_congruent_side base height →
  side = real.sqrt (19) / 4 :=
by
  intro h_s h_A_equilateral h_A_isosceles h_base h_height h_side
  rw [h_s, h_base] at *
  sorry

end isosceles_triangle_side_length_l231_231367


namespace count_numbers_with_zero_l231_231058

theorem count_numbers_with_zero : 
  ∀ n ≤ 3050, (count (λ x, (x.to_digits ∈ ['0'])) (list.range (n+1)) = 763) := 
sorry

end count_numbers_with_zero_l231_231058


namespace lowest_score_l231_231185

theorem lowest_score (μ σ top5score : ℝ) (mean : μ = 60) (std_dev : σ = 10) 
  (top_5 : top5score = zscore 1.645 σ + μ) (within_2_std : zscore 2 σ + μ ≥ top5score) :
  77 ∈ ℤ :=
by sorry

-- Definitions
def zscore (z : ℝ) (σ : ℝ) : ℝ := z * σ

end lowest_score_l231_231185


namespace greatest_common_divisor_of_B_l231_231903

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231903


namespace pascal_fifth_number_in_row_15_l231_231630

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231630


namespace pascal_fifth_number_l231_231743

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231743


namespace find_a_for_exact_three_solutions_l231_231410

theorem find_a_for_exact_three_solutions :
  ∃ a, (a = 9 ∨ a = 121 ∨ a = ((17 * Real.sqrt 2 - 1) / 2)^2) ∧
       (∀ (x y : ℝ), (x = -|y - Real.sqrt a| + 6 - Real.sqrt a ∧ 
                      (|x| - 8)^2 + (|y| - 15)^2 = 289) →
       ... -- the condition ensuring that the solution count must be exactly 3
sorry

end find_a_for_exact_three_solutions_l231_231410


namespace count_multiples_6_or_8_not_both_l231_231508

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231508


namespace pascal_15_5th_number_l231_231774

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231774


namespace fourth_ball_black_probability_l231_231327

-- Definitions from the conditions
def total_balls : ℕ := 6
def black_balls : ℕ := 3
def red_balls : ℕ := 3

theorem fourth_ball_black_probability :
  (4 : ℕ) ∈ Finset.range (total_balls + 1) →
  (black_balls + red_balls = total_balls) →
  (red_balls = 3) →
  (black_balls = 3) →
  sorry 
  -- The probability that the fourth ball selected is black is 1/2 
  -- Possible implementation could leverage definitions and requisite libraries.

end fourth_ball_black_probability_l231_231327


namespace gcd_of_B_is_two_l231_231884

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231884


namespace trace_bag_weight_l231_231224

-- Definitions for the given problem
def weight_gordon_bag1 := 3
def weight_gordon_bag2 := 7
def total_weight_gordon := weight_gordon_bag1 + weight_gordon_bag2

noncomputable def weight_trace_one_bag : ℕ :=
  sorry

-- Theorem for what we need to prove
theorem trace_bag_weight :
  total_weight_gordon = 10 ∧
  weight_trace_one_bag = total_weight_gordon / 5 :=
sorry

end trace_bag_weight_l231_231224


namespace cos_B_correct_max_area_triangle_correct_l231_231088

noncomputable def cos_B (A B : ℕ) (a b : ℕ) : ℝ :=
  if A = 60 ∧ a = 3 ∧ b = 2 then 
    (by sorry : ℝ)
  else 
    0

noncomputable def max_area_triangle (A : ℕ) (a : ℕ) : ℝ :=
  if A = 60 ∧ a = 3 then 
    (by sorry : ℝ)
  else 
    0

theorem cos_B_correct : cos_B 60 0 3 2 = real.sqrt 6 / 3 := sorry

theorem max_area_triangle_correct : max_area_triangle 60 3 = (9 * real.sqrt 3) / 4 := sorry

end cos_B_correct_max_area_triangle_correct_l231_231088


namespace probability_two_correct_letters_l231_231216

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of derangement of n
noncomputable def derangement (n : ℕ) : ℕ :=
  factorial n * (List.range n).sum (λ k, (-1 : ℤ) ^ k / (factorial k) : ℚ).ceil.toNat

theorem probability_two_correct_letters :
  let total_people := 7
  let num_ways_two_correct := Nat.choose total_people 2
  let derangements_five := derangement 5
  let total_favorable := num_ways_two_correct * derangements_five
  let total_distributions := factorial total_people
  total_favorable / total_distributions = 11 / 60 :=
by
  sorry

end probability_two_correct_letters_l231_231216


namespace pascal_fifth_number_l231_231745

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231745


namespace common_divisor_l231_231202

theorem common_divisor (d : ℕ) (h1 : 30 % d = 3) (h2 : 40 % d = 4) : d = 9 :=
by 
  sorry

end common_divisor_l231_231202


namespace greatest_integer_not_exceeding_100y_l231_231993

noncomputable def y : ℝ := (∑ n in Finset.range 30, Real.cos (n + 1) * (Real.pi / 180)) / (∑ n in Finset.range 30, Real.sin (n + 1) * (Real.pi / 180))

theorem greatest_integer_not_exceeding_100y :
  ⌊100 * y⌋ = 173 :=
by
  sorry

end greatest_integer_not_exceeding_100y_l231_231993


namespace pascal_triangle_fifth_number_l231_231638

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231638


namespace gcd_7_nplus2_8_2nplus1_l231_231236

theorem gcd_7_nplus2_8_2nplus1 : 
  ∃ d : ℕ, (∀ n : ℕ, d ∣ (7^(n+2) + 8^(2*n+1))) ∧ (∀ n : ℕ, d = 57) :=
sorry

end gcd_7_nplus2_8_2nplus1_l231_231236


namespace pascal_fifth_number_l231_231747

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231747


namespace parallel_vectors_orthogonal_vectors_min_k_l231_231491

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)
noncomputable def vector_c (m t : ℝ) : ℝ × ℝ := (1 + (t^2 + 1) * (-2), 2 + (t^2 + 1) * m)
noncomputable def vector_d (k t : ℝ) : ℝ × ℝ := (-k * 1 + (1 / t) * (-2), -k * 2 + (1 / t) * m)
noncomputable def vector_x (m t : ℝ) : ℝ × ℝ := (1 + (t^2 + 1) * (-2), 2 + (t^2 + 1) * m)
noncomputable def vector_y (k t : ℝ) : ℝ × ℝ := (-k * 1 + (1 / t) * (-2), -k * 2 + (1 / t) * 1)

theorem parallel_vectors (m : ℝ) (h : vector_a = vector_b m) : m = -4 := by
  sorry

theorem orthogonal_vectors (m : ℝ) (h : vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) : m = 1 := by
  sorry

theorem min_k (k t : ℝ) (h : t > 0) (m : ℝ) (hx : vector_x m t) (hy : vector_y k t) (hperp : hx.1 * hy.1 + hx.2 * hy.2 = 0) : k ≥ 2 := by
  sorry

end parallel_vectors_orthogonal_vectors_min_k_l231_231491


namespace gcd_elements_of_B_l231_231955

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231955


namespace gcd_B_is_2_l231_231914

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231914


namespace correct_scientific_notation_l231_231376

def scientific_notation (n : ℝ) : ℝ × ℝ := 
  (4, 5)

theorem correct_scientific_notation : scientific_notation 400000 = (4, 5) :=
by {
  sorry
}

end correct_scientific_notation_l231_231376


namespace triangle_angle_C_min_value_2a_plus_b_l231_231113

-- Part (1): Triangle Angle Condition
theorem triangle_angle_C (a b c : ℝ) (h : (a + b + c) * (a + b - c) = a * b) : 
  ∃ C : ℝ, cos C = -1/2 ∧ 0 < C ∧ C < pi := 
sorry

-- Part (2): Minimum value of 2a + b
theorem min_value_2a_plus_b (a b c d : ℝ) (h : C = 2 * pi / 3) (hCD : CD = 2): 
  ∀ a b : ℝ, 
  ((a + b + c) * (a + b - c) = a * b) →
  (angle_bisector_intersects_D : CD = 2) →
  2 * a + b ≥ 6 + 4 * sqrt 2 := 
sorry

end triangle_angle_C_min_value_2a_plus_b_l231_231113


namespace ordered_triples_count_l231_231507

theorem ordered_triples_count (x y z : ℝ)
  (h1 : x^2 + y^2 + z^2 = 9)
  (h2 : x^4 + y^4 + z^4 = 33)
  (h3 : x * y * z = -4) :
  ∃ (S : finset (ℝ × ℝ × ℝ)), S.card = 12 ∧ ∀ t ∈ S, let (x, y, z) := t in
    x^2 + y^2 + z^2 = 9 ∧ x^4 + y^4 + z^4 = 33 ∧ x * y * z = -4 :=
begin
  sorry
end

end ordered_triples_count_l231_231507


namespace pascal_fifth_number_in_row_15_l231_231635

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231635


namespace pascal_fifth_number_l231_231739

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231739


namespace gcd_of_B_is_two_l231_231886

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231886


namespace GCD_of_set_B_is_2_l231_231951

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231951


namespace gcd_of_sum_of_four_consecutive_integers_l231_231878

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231878


namespace three_point_one_two_six_as_fraction_l231_231304

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction_l231_231304


namespace total_history_and_maths_l231_231559

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end total_history_and_maths_l231_231559


namespace dice_probability_sum_17_l231_231082

-- Problem: Prove the probability that the sum of the face-up integers is 17 when three standard 6-faced dice are rolled is 1/24.

def probability_sum_17 (dice_rolls : ℕ → ℕ) (n : ℕ) : ℝ :=
  let probability_6 := 1 / 6  in
  let probability_case_A := (6 * (probability_6^3))  -- Case where one die shows 6 and other two sum to 11
  let probability_case_B := (3 * (probability_6^3))  -- Case where two dice show 6 and third shows 5
  probability_case_A + probability_case_B

theorem dice_probability_sum_17 : probability_sum_17 = 1 / 24 :=
by
  sorry

end dice_probability_sum_17_l231_231082


namespace fish_lifespan_proof_l231_231221

def hamster_lifespan : ℝ := 2.5

def dog_lifespan : ℝ := 4 * hamster_lifespan

def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_proof :
  fish_lifespan = 12 := 
  by
  sorry

end fish_lifespan_proof_l231_231221


namespace round_robin_tournament_teams_l231_231094

theorem round_robin_tournament_teams (n : ℕ) (h_match_count : 34) (h_withdraw: 2 * 3) (h_play: (n - 2) * (n - 3) / 2 = 28) : n = 10 := 
by
  have h1 : (n - 2) * (n - 3) = 56 := by
    rw [← nat.choose_two_eq h_play]
    exact (mul_comm (n - 3) (n - 2)).symm
  have h2 : (n - 2) * (n - 3) = 56 := by sorry
  have h3 : nat.main_eq (8 * (n - 10 ) = 80) := by sorry
  sorry

end round_robin_tournament_teams_l231_231094


namespace gcd_of_B_is_2_l231_231839

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231839


namespace cd_perpendicular_to_plane_a1oc_sine_dihedral_angle_b_a1c_d_l231_231317

variables {A B C D E O A_1 : Type}

structure RightAngledTrapezoid (A B C D : Type) :=
  (AD_parallel_BC : ∀ {a b c d : A B C D}, d ∥ c)
  (angle_BAD_right : ∀ {a b : A B}, angle a b = π / 2)
  (AB_one : ∀ {a b : A B}, dist a b = 1)
  (BC_one : ∀ {b c : B C}, dist b c = 1)
  (AD_two : ∀ {a d : A D}, dist a d = 2)
  (midpoint_E : ∀ {a d : A D}, E = midpoint a d)
  (intersection_O : ∀ {a c b e : A C B E}, O = intersection a c b e)

theorem cd_perpendicular_to_plane_a1oc {A B C D E O A_1 : Type} 
  [RightAngledTrapezoid A B C D]
  (CD : ∀ {c d : C D}, c ⊥ d)
  (plane_A1OC : ∀ {a_1 o c : A_1 O C}, plane a_1 o c) :
  CD ⊥ plane_A1OC :=
sorry

theorem sine_dihedral_angle_b_a1c_d {A B C D E O A_1 : Type} 
  [RightAngledTrapezoid A B C D]
  (plane_A1BE_perp_BCDE : ∀ {a_1 b e : A_1 B E}, plane a_1 b e ⊥ plane B C D E)
  : real.sin (dihedral_angle B A_1 C D) =
sorry

end cd_perpendicular_to_plane_a1oc_sine_dihedral_angle_b_a1c_d_l231_231317


namespace angle_a_sin_cos_relation_polygon_sides_march_days_l231_231105

theorem angle_a (AEP_angle EAP_angle: ℕ) (h1: AEP_angle = 85) (h2: EAP_angle = 65) 
  : 180 - AEP_angle - EAP_angle = 30 := by sorry

theorem sin_cos_relation (a_deg b_deg: ℝ) (ha: a_deg = 30) (h1 : sin (a_deg + 210) = cos b_deg) 
  (h2 : 90 < b_deg ∧ b_deg < 180) : b_deg = 150 := by sorry

theorem polygon_sides (b_angle: ℝ) (hb: b_angle = 150) 
  : 360 / (180 - b_angle) = 12 := by sorry

theorem march_days (n_day k_day: ℕ) (h1: n_day = 12) (h2 : k_day > 20 ∧ k_day < 25)
  (h_friday: n_day = 5) (h_wednesday: k_day = 3)
  (h_days: k_day - n_day = 12) : k_day = 24 := by sorry

end angle_a_sin_cos_relation_polygon_sides_march_days_l231_231105


namespace dodecahedron_interior_diagonals_l231_231024

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231024


namespace circles_separated_l231_231454

/-- Given the equations of two circles C₁ and C₂ and the condition m > 3,
    prove that the two circles are separated. -/
theorem circles_separated (m : ℝ) (hm : m > 3) :
  let C₁ := λ x y : ℝ, x^2 + y^2 - 2*m*x + m^2 = 4
  let C₂ := λ x y : ℝ, x^2 + y^2 + 2*x - 2*m*y = 8 - m^2
  ∃ d : ℝ, d = real.sqrt ((m + 1)^2 + m^2) ∧ d > (2 + 3) :=
by {
  let C₁ := λ x y : ℝ, x^2 + y^2 - 2*m*x + m^2 = 4,
  let C₂ := λ x y : ℝ, x^2 + y^2 + 2*x - 2*m*y = 8 - m^2,
  existsi real.sqrt ((m + 1)^2 + m^2),
  split,
  { refl },
  { sorry }
}

end circles_separated_l231_231454


namespace GCD_of_set_B_is_2_l231_231949

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231949


namespace gcd_12345_6789_eq_3_l231_231231

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l231_231231


namespace matchstick_combinations_l231_231213

theorem matchstick_combinations : 
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (x, y) ∈ s, 4 * x + 3 * y = 200) ∧ 
    (∀ (x, y) ∈ s, x ∈ ℕ ∧ y ∈ ℕ) ∧ 
    s.card = 16 := 
by
  sorry

end matchstick_combinations_l231_231213


namespace percentage_of_stock_is_40_l231_231099

-- Define the conditions from the problem
def income : ℝ := 15000
def investment : ℝ := 37500

-- Define the percentage of stock P, which we aim to prove equals 40.
def P : ℝ := (income * 100) / investment

-- The theorem states that given the above conditions, P equals 40.
theorem percentage_of_stock_is_40 : P = 40 := by
  -- Proof is not required, so we use 'sorry' to skip it
  sorry

end percentage_of_stock_is_40_l231_231099


namespace number_of_true_propositions_l231_231494

/-- Defining the complex number z --/
def z : ℂ := 2 / (1 + I)

/-- Defining the propositions --/
def P1 : Prop := conj z = 1 + I
def P2 : Prop := z.re = 1
def P3 : Prop := (z.re, z.im) ⬝ (1, 1) = 0 -- Dot product of corresponding vectors
def P4 : Prop := complex.abs z = real.sqrt 2

/-- Proving that there are exactly three true propositions among P1, P2, P3, P4 --/
theorem number_of_true_propositions : (¬ P1) ∧ P2 ∧ P3 ∧ P4 :=
by
  -- Placeholder for the steps to prove the propositions
  sorry

end number_of_true_propositions_l231_231494


namespace GCD_of_set_B_is_2_l231_231947

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231947


namespace minimum_translation_to_odd_l231_231391

/-- Determining the minimum value of translation such that resulting function is odd --/
theorem minimum_translation_to_odd (a1 a2 a3 a4 x : ℝ) (h_det : a1 * a4 - a2 * a3 = sqrt 3 * sin x - cos x) (varphi : ℝ) (h_varphi_pos : 0 < varphi) :
  (∃ k : ℤ, varphi + π / 6 = k * π ∧ varphi = (5 * π) / 6) :=
sorry

end minimum_translation_to_odd_l231_231391


namespace relationship_of_squares_and_products_l231_231176

theorem relationship_of_squares_and_products (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) : 
  x^2 > ax ∧ ax > b^2 :=
by
  sorry

end relationship_of_squares_and_products_l231_231176


namespace greatest_two_digit_product_is_12_l231_231242

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231242


namespace skateboards_and_bicycles_total_l231_231144

def number_of_skateboards (k : ℕ) : ℕ := 7 * k
def number_of_bicycles (k : ℕ) : ℕ := 4 * k

theorem skateboards_and_bicycles_total :
  ∃ (k : ℕ), number_of_skateboards k - number_of_bicycles k = 12 ∧
             number_of_skateboards k + number_of_bicycles k = 44 :=
begin
  sorry,
end

end skateboards_and_bicycles_total_l231_231144


namespace triangle_BX_in_terms_of_sides_l231_231360

-- Define the triangle with angles and points
variables {A B C : ℝ}
variables {AB AC BC : ℝ}
variables (X Y : ℝ) (AZ : ℝ)

-- Add conditions as assumptions
variables (angle_A_bisector : 2 * A = (B + C)) -- AZ is the angle bisector of angle A
variables (angle_B_lt_C : B < C) -- angle B < angle C
variables (point_XY : X / AB = Y / AC ∧ X = Y) -- BX = CY and angles BZX = CZY

-- Define the statement to be proved
theorem triangle_BX_in_terms_of_sides :
    BX = CY →
    (AZ < 1 ∧ AZ > 0) →
    A + B + C = π → 
    BX = (BC * BC) / (AB + AC) :=
sorry

end triangle_BX_in_terms_of_sides_l231_231360


namespace multiples_6_8_not_both_l231_231526

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231526


namespace pascal_fifteen_four_l231_231790

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231790


namespace product_of_five_consecutive_integers_not_square_l231_231170

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l231_231170


namespace inequality_not_true_l231_231438

variable {x y : ℝ}

theorem inequality_not_true (h : x > y) : ¬(-3 * x + 6 > -3 * y + 6) :=
by
  sorry

end inequality_not_true_l231_231438


namespace seventh_observation_value_l231_231184

theorem seventh_observation_value (avg6 : ℝ) (new_avg : ℝ) (sum6 : ℕ) (sum7 : ℕ) 
  (h1 : avg6 = 12) 
  (h2 : sum6 = 6 * avg6)
  (h3 : new_avg = avg6 - 1)
  (h4 : sum7 = 7 * new_avg) :
  sum7 - sum6 = 5 :=
by {
  have h5 : sum6 = 72,
  { rw [h1, h2], norm_num },
  have h6 : new_avg = 11,
  { rw [h1, h3], norm_num },
  have h7 : sum7 = 77,
  { rw [h6, h4], norm_num },
  rw [h5, h7],
  norm_num,
}

end seventh_observation_value_l231_231184


namespace pascal_triangle_fifth_number_l231_231699

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231699


namespace total_lucky_stars_l231_231182

theorem total_lucky_stars : 
  (∃ n : ℕ, 10 * n + 6 = 116 ∧ 4 * 8 + (n - 4) * 12 = 116) → 
  116 = 116 := 
by
  intro h
  obtain ⟨n, h1, h2⟩ := h
  sorry

end total_lucky_stars_l231_231182


namespace area_of_trapezoid_l231_231166

-- Define the conditions of the trapezoid
variables 
  (AD AB BC h : ℕ) 
  (ABpos BCpos hpos : ℕ → Prop)

-- Set the given conditions
def is_trapezoid (AD AB BC h : ℕ) : Prop := 
  ABpos AB ∧ BCpos BC ∧ hpos h ∧ AD = 15 ∧ AB = 50 ∧ BC = 20 ∧ h = 12

-- The main statement to prove the area
theorem area_of_trapezoid
  (AD AB BC h : ℕ)
  (trapezoid : is_trapezoid AD AB BC h) :
  let CD := (Math.sqrt (AB - h) ^ 2 + BC ^ 2) + AB + (Math.sqrt (BC - h) ^ 2 + AD ^ 2) in
  (1/2) * (AB + CD) * h = 750 :=
by
  sorry -- This is where the proof would go, but it is skipped as instructed.

end area_of_trapezoid_l231_231166


namespace dodecahedron_interior_diagonals_l231_231014

theorem dodecahedron_interior_diagonals :
  let vertices := 20
  let faces_meet_at_vertex := 3
  let interior_diagonals := (vertices * (vertices - faces_meet_at_vertex - 1)) / 2
  interior_diagonals = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l231_231014


namespace pascal_triangle_fifth_number_l231_231644

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231644


namespace pascal_triangle_row_fifth_number_l231_231722

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231722


namespace pascal_triangle_15_4_l231_231607

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231607


namespace dodecahedron_interior_diagonals_l231_231051

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231051


namespace gcd_of_sum_of_four_consecutive_integers_l231_231873

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231873


namespace line_intersects_x_axis_at_3_0_l231_231372

theorem line_intersects_x_axis_at_3_0 : ∃ (x : ℝ), ∃ (y : ℝ), 2 * y + 5 * x = 15 ∧ y = 0 ∧ (x, y) = (3, 0) :=
by
  sorry

end line_intersects_x_axis_at_3_0_l231_231372


namespace fifth_number_in_pascals_triangle_l231_231593

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231593


namespace roots_equation_l231_231978

variables (m n : ℝ)
variables (α β γ δ : ℝ)

-- Conditions
def polynomial1 (x : ℝ) : Prop := x^2 + m * x - 1 = 0
def polynomial2 (x : ℝ) : Prop := x^2 + n * x - 1 = 0

-- Roots of the polynomials
axiom alpha_root : polynomial1 α
axiom beta_root : polynomial1 β
axiom gamma_root : polynomial2 γ
axiom delta_root : polynomial2 δ

theorem roots_equation :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = (m - n) ^ 2 :=
sorry

end roots_equation_l231_231978


namespace circumcircles_concurrent_l231_231370

variable {α : Type*} [EuclideanGeometry α]

theorem circumcircles_concurrent (O A B C X Y Z : α)
  (hO_mid_AX : midpoint O A X)
  (hO_mid_BY : midpoint O B Y)
  (hO_mid_CZ : midpoint O C Z) :
  ∃ M : α, M ∈ circ_circle B C X ∧ M ∈ circ_circle C A Y ∧ M ∈ circ_circle A B Z ∧ M ∈ circ_circle X Y Z := 
sorry

end circumcircles_concurrent_l231_231370


namespace greatest_common_divisor_of_B_l231_231919

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231919


namespace pascal_fifth_number_in_row_15_l231_231629

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231629


namespace pascal_triangle_fifth_number_l231_231675

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231675


namespace dodecahedron_interior_diagonals_l231_231010

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231010


namespace ratio_hearts_to_diamonds_l231_231346

theorem ratio_hearts_to_diamonds :
  ∀ (S D H C : ℕ), 
  C = 6 ∧ 
  S + D + H + C = 13 ∧ 
  D = 2 * S ∧
  S + C = 7 ∧ 
  D + H = 6 → 
  H / gcd H D = 2 ∧ D / gcd H D = 1 :=
by {
  intros S D H C,
  assume hC hSum hD hBlack hRed,
  have hC6 : C = 6 := hC,
  have hSum13 : S + D + H + C = 13 := hSum,
  have hD2S : D = 2 * S := hD,
  have hSplusC : S + C = 7 := hBlack,
  have hDplusH : D + H = 6 := hRed,
  have gcd_exists : gcd H D > 0 := nat.gcd_pos_of_pos_right D (by sorry),
  exact ⟨nat.div_eq_of_eq_mul_right gcd_exists (by sorry), nat.div_eq_of_eq_mul_left gcd_exists (by sorry)⟩
}

end ratio_hearts_to_diamonds_l231_231346


namespace fifth_number_in_pascals_triangle_l231_231596

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231596


namespace pascal_fifth_number_l231_231733

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231733


namespace average_speed_difference_l231_231207

noncomputable def v_R : Float := 56.44102863722254
noncomputable def distance : Float := 750
noncomputable def t_R : Float := distance / v_R
noncomputable def t_P : Float := t_R - 2
noncomputable def v_P : Float := distance / t_P

theorem average_speed_difference : v_P - v_R = 10 := by
  sorry

end average_speed_difference_l231_231207


namespace pascal_triangle_15_4_l231_231614

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231614


namespace gcd_of_B_is_2_l231_231832

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231832


namespace fifth_number_in_pascal_row_l231_231804

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231804


namespace pascal_15_5th_number_l231_231765

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231765


namespace greatest_integer_100y_l231_231994

noncomputable def y : ℝ := (∑ n in finset.range 1 31, real.cos (n * real.pi / 180)) / (∑ n in finset.range 1 31, real.sin (n * real.pi / 180))

theorem greatest_integer_100y : ⌊100 * y⌋ = 373 := sorry

end greatest_integer_100y_l231_231994


namespace altitudes_sum_equals_two_sum_sides_l231_231097

noncomputable section

-- Defining the basic structure and properties of an acute angled triangle
variables {A B C D E F H_A H_B H_C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
variables (triangle : Triangle A B C) (midpoints : Midpoint D E F) 
variables (altitudes : Altitude H_A H_B H_C)

-- Prove the relationship given in the problem
theorem altitudes_sum_equals_two_sum_sides
  (h_a h_b h_c : ℝ)
  (a b c : ℝ)
  (h1 : is_midpoint D B C)
  (h2 : is_midpoint E C A)
  (h3 : is_midpoint F A B)
  (h4 : is_foot H_A A BC)
  (h5 : is_foot H_B B CA)
  (h6 : is_foot H_C C AB)
  (h7 : right_angle A H_A B)
  (h8 : right_angle B H_B C)
  (h9 : right_angle C H_C A)
  (h10 : altitude_length A H_A = h_a)
  (h11 : altitude_length B H_B = h_b)
  (h12 : altitude_length C H_C = h_c)
  (h13 : side_length B C = a)
  (h14 : side_length C A = b)
  (h15 : side_length A B = c)
  (H_acute : acute_triangle A B C) :
  h_a + h_b + h_c = 2 * (a + b + c) := sorry

end altitudes_sum_equals_two_sum_sides_l231_231097


namespace pascal_triangle_fifth_number_l231_231639

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231639


namespace greatest_common_divisor_of_B_l231_231917

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231917


namespace greatest_two_digit_prod_12_l231_231286

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231286


namespace g_f_neg3_l231_231982

def f (x : ℤ) : ℤ := x^3 - 1
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 1

theorem g_f_neg3 : g (f (-3)) = 2285 :=
by
  -- provide the proof here
  sorry

end g_f_neg3_l231_231982


namespace trace_bag_weight_l231_231223

-- Define the weights of Gordon's bags
def gordon_bag1_weight : ℕ := 3
def gordon_bag2_weight : ℕ := 7

-- Define the number of Trace's bags
def trace_num_bags : ℕ := 5

-- Define what we are trying to prove: the weight of one of Trace's shopping bags
theorem trace_bag_weight :
  (gordon_bag1_weight + gordon_bag2_weight) = (trace_num_bags * 2) :=
by
  sorry

end trace_bag_weight_l231_231223


namespace pascal_triangle_15_4_l231_231609

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231609


namespace dodecahedron_interior_diagonals_l231_231027

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231027


namespace gcd_of_B_is_two_l231_231888

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231888


namespace isosceles_base_lines_l231_231200
open Real

theorem isosceles_base_lines {x y : ℝ} (h1 : 7 * x - y - 9 = 0) (h2 : x + y - 7 = 0) (hx : x = 3) (hy : y = -8) :
  (x - 3 * y - 27 = 0) ∨ (3 * x + y - 1 = 0) :=
sorry

end isosceles_base_lines_l231_231200


namespace dodecahedron_interior_diagonals_l231_231030

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231030


namespace area_of_triangle_l231_231089

theorem area_of_triangle {a c : ℝ} (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = 60) :
    (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end area_of_triangle_l231_231089


namespace Pascal_triangle_fifth_number_l231_231667

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231667


namespace greatest_two_digit_product_12_l231_231275

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231275


namespace greatest_two_digit_with_product_12_l231_231263

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231263


namespace non_squarefree_exists_ap_gp_seq_l231_231427

theorem non_squarefree_exists_ap_gp_seq (m : ℕ) (h_pos : m > 0) : 
  (∃ (a : ℕ → ℤ), (∀ n, a (n + 1) ≡ a n + (a 1 - a 0) [MOD m]) ∧ 
                    (∀ n, a (n + 1) ≡ a n * (a 1 / a 0) [MOD m]) ∧ 
                    (∀ n, a n ≠ a (n+1))) → 
  ¬ is_squarefree m := 
sorry

end non_squarefree_exists_ap_gp_seq_l231_231427


namespace fg_minus_gf_l231_231181

def f (x : ℝ) : ℝ := 8 * x - 12
def g (x : ℝ) : ℝ := x / 4 + 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 12 := 
by
  sorry

end fg_minus_gf_l231_231181


namespace gcd_B_eq_two_l231_231863

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231863


namespace pascal_row_fifth_number_l231_231586

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231586


namespace greatest_two_digit_with_product_12_l231_231261

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231261


namespace greatest_two_digit_with_product_12_l231_231260

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231260


namespace greatest_two_digit_product_12_l231_231252

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231252


namespace graph_shift_correct_l231_231194

def g (x : ℝ) : ℝ :=
if x >= -4 ∧ x <= -1 then x + 3
else if x >= -1 ∧ x <= 1 then real.sqrt (1 - x^2) + 1
else if x >= 1 ∧ x <= 4 then -x + 3
else 0  -- Outside the given range.

theorem graph_shift_correct : 
  ∃ (x : ℝ), g x = 2 ∧ (g (x) + 2 = 4) := 
begin
  use 0,
  split,
  {
    -- This would be the proof that g(0) = 2
    sorry
  },
  {
    -- This would be the proof that g(0) + 2 = 4
    sorry
  }
end

end graph_shift_correct_l231_231194


namespace five_common_divisors_product_l231_231381

theorem five_common_divisors_product :
  let L := [48, 64, -18, 162, 144]
  ∃ (d1 d2 d3 d4 d5 : ℕ),
      (∀ n ∈ L, d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧ d4 ∣ n ∧ d5 ∣ n) ∧
      [d1, d2, d3, d4, d5] = [1, 2, 3, 6, 3] ∧
      d1 * d2 * d3 * d4 * d5 = 108 := 
by
  let L := [48, 64, -18, 162, 144]
  existsi (1 : ℕ)
  existsi (2 : ℕ)
  existsi (3 : ℕ)
  existsi (6 : ℕ)
  existsi (3 : ℕ)
  simp
  sorry

end five_common_divisors_product_l231_231381


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231970

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231970


namespace dodecahedron_interior_diagonals_l231_231047

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231047


namespace ways_to_reach_10_steps_l231_231210

def ways_to_reach_step : ℕ → ℕ
| 0       := 1  -- to ensure the base case simplicity
| 1       := 1
| 2       := 2
| 3       := 4
| (n + 1) := ways_to_reach_step n + ways_to_reach_step (n - 1) + ways_to_reach_step (n - 2)

theorem ways_to_reach_10_steps : ways_to_reach_step 10 = 274 :=
by
  sorry

end ways_to_reach_10_steps_l231_231210


namespace bela_wins_l231_231091

noncomputable def has_winning_strategy (n : ℕ) : Prop :=
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ n ∧ ¬ Nat.Prime (⌈x⌉_nat) → x + 1.5 ≤ n → 
   (∃ y : ℝ, 0 ≤ y ∧ y ≤ n ∧ x + 1.5 ≤ y ∧ x ≠ y ∧ ¬ Nat.Prime (⌈y⌉_nat)))

theorem bela_wins (n : ℕ) (h : n > 5) : 
  has_winning_strategy n ∨ (∃ x : ℝ, 0 ≤ x ∧ x ≤ n ∧ Nat.Prime (⌈x⌉_nat)) → false :=
begin
  sorry
end

end bela_wins_l231_231091


namespace multiples_of_6_or_8_under_201_not_both_l231_231520

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231520


namespace pascal_triangle_fifth_number_l231_231641

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231641


namespace greatest_two_digit_product_12_l231_231248

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231248


namespace parallelepiped_problem_l231_231344

open Real 

-- Define that vectors u, v, w forming the parallelepiped
variables (u v w : ℝ^3) (α : ℝ)

-- The condition that scaling factor α ≠ 1
axiom alpha_non_one : α ≠ 1

-- Vectors representing the vertices of the parallelepiped
def A := (0 : ℝ^3)
def B := v
def D := w
def E := α • u
def C := v + w
def F := α • u + v
def G := α • u + v + w
def H := α • u + w

-- Squared distances
def AG2 := ∥G - A∥^2
def BH2 := ∥H - B∥^2
def CE2 := ∥E - C∥^2
def DF2 := ∥F - D∥^2

def AB2 := ∥B - A∥^2
def AD2 := ∥D - A∥^2
def AE2 := ∥E - A∥^2

-- The proof problem
theorem parallelepiped_problem : 
  (AG2 + BH2 + CE2 + DF2) / (AB2 + AD2 + AE2) = 4 :=
sorry

end parallelepiped_problem_l231_231344


namespace tan_Y_value_l231_231565

namespace RightTriangle

-- Define the sides and the hypotenuse of the right triangle
variables (XY : ℝ) (YZ : ℝ) (XZ : ℝ)

-- Conditions on the sides
axiom XY_length : XY = 24
axiom YZ_length : YZ = 26
axiom XZ_length : XZ = sqrt (YZ ^ 2 - XY ^ 2)

-- Define the tangent of the angle Y
noncomputable def tan_Y (XY XZ : ℝ) : ℝ := XZ / XY

-- State the theorem to be proven
theorem tan_Y_value : tan_Y 24 10 = 5 / 12 :=
by 
  rw [tan_Y, XY_length, XZ_length]
  norm_num
  sorry

end RightTriangle

end tan_Y_value_l231_231565


namespace proof_math_problem_lean_l231_231539

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231539


namespace pascal_row_fifth_number_l231_231581

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231581


namespace multiples_count_l231_231515

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231515


namespace multiples_six_or_eight_not_both_l231_231532

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231532


namespace total_points_mismatch_l231_231305

theorem total_points_mismatch :
  let bruce_goals : ℕ := 4,
      michael_goals := 2 * bruce_goals,
      jack_goals := bruce_goals - 1,
      sarah_goals := jack_goals / 2,
      total_football_goals := bruce_goals + michael_goals + jack_goals + sarah_goals,

      andy_points : ℕ := 22,
      lily_points := andy_points + 18,
      total_basketball_points := andy_points + lily_points,

      combined_total_points := total_football_goals + total_basketball_points
  in combined_total_points ≠ 130 := by {
  sorry -- no proof required
}

end total_points_mismatch_l231_231305


namespace pascal_fifth_number_l231_231734

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231734


namespace pascal_row_fifth_number_l231_231587

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231587


namespace solve_inequalities_factorize_expression_no_solution_fractional_eq_simplify_evaluate_expr_l231_231375

open Int
open Real

-- Part 1: Inequalities
theorem solve_inequalities (x : ℝ) : 
(2 * x - 1) / 3 - (5 * x + 1) / 2 ≤ 1 ∧ (5 * x - 1) < 3 * (x + 1) → -1 ≤ x ∧ x < 2 := 
sorry

-- Part 2: Factorization
theorem factorize_expression (a b : ℝ) : 
-b^3 + 4 * a * b^2 - 4 * a^2 * b = -b * (b - 2 * a)^2 := 
sorry

-- Part 3: Fractional equation
theorem no_solution_fractional_eq (x : ℝ) : 
x / (x - 1) - 1 = 3 / ((x - 1) * (x + 2)) → false := 
sorry

-- Part 4: Simplification and evaluation
theorem simplify_evaluate_expr (x : ℝ) : 
(1 - x^2 / (x^2 + x)) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1 :=
begin
  intro h,
  have h1 : x = 2, from h,
  sorry
end

end solve_inequalities_factorize_expression_no_solution_fractional_eq_simplify_evaluate_expr_l231_231375


namespace dodecahedron_interior_diagonals_l231_231029

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231029


namespace greatest_two_digit_product_12_l231_231274

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231274


namespace pascal_fifth_element_15th_row_l231_231710

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231710


namespace inscribed_circle_diameter_l231_231393

noncomputable def diameter_inscribed_circle (side_length : ℝ) : ℝ :=
  let s := (3 * side_length) / 2
  let K := (Real.sqrt 3 / 4) * (side_length ^ 2)
  let r := K / s
  2 * r

theorem inscribed_circle_diameter (side_length : ℝ) (h : side_length = 10) :
  diameter_inscribed_circle side_length = (10 * Real.sqrt 3) / 3 :=
by
  rw [h]
  simp [diameter_inscribed_circle]
  sorry

end inscribed_circle_diameter_l231_231393


namespace max_excellent_courses_l231_231568

section CourseEvaluation

-- Defining the structure for a course
structure Course :=
  (views : ℕ) -- Number of views
  (score : ℕ) -- Expert score

-- Defining the "not inferior" relation
def not_inferior (A B : Course) : Prop :=
  (A.views > B.views) ∨ (A.score > B.score)

-- Defining the condition for an excellent course
def excellent_course (A : Course) (others : List Course) : Prop :=
  ∀ B ∈ others, not_inferior A B

-- Proving the maximum number of excellent courses
theorem max_excellent_courses (courses : List Course) (h_length : courses.length = 5) :
  ∃ (excellent_courses : List Course), excellent_courses.length = 5 ∧
    ∀ (A ∈ excellent_courses), excellent_course A courses :=
sorry

end CourseEvaluation

end max_excellent_courses_l231_231568


namespace pascal_triangle_fifth_number_l231_231758

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231758


namespace pascal_triangle_fifth_number_l231_231762

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231762


namespace multiples_six_or_eight_not_both_l231_231536

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231536


namespace greatest_integer_not_exceeding_100y_l231_231992

noncomputable def y : ℝ := (∑ n in Finset.range 30, Real.cos (n + 1) * (Real.pi / 180)) / (∑ n in Finset.range 30, Real.sin (n + 1) * (Real.pi / 180))

theorem greatest_integer_not_exceeding_100y :
  ⌊100 * y⌋ = 173 :=
by
  sorry

end greatest_integer_not_exceeding_100y_l231_231992


namespace pascal_triangle_fifth_number_l231_231689

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231689


namespace nth_term_of_sequence_l231_231493

theorem nth_term_of_sequence (n : Nat) (hn : n > 0) : (10^(n-1) : Nat) = (nth_term n) where
  nth_term n := match n with
    | 0 => 10^0
    | _ => 10^n
sorry

end nth_term_of_sequence_l231_231493


namespace gcd_of_sum_of_four_consecutive_integers_l231_231869

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231869


namespace dodecahedron_interior_diagonals_l231_231041

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231041


namespace gcd_of_B_l231_231936

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231936


namespace pascal_fifth_element_15th_row_l231_231704

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231704


namespace sum_of_complex_numbers_nonzero_sum_of_reciprocal_complex_numbers_nonzero_l231_231996

noncomputable def z_i : ℂ → Prop := sorry

variables {α : ℝ} {n : ℕ} {z : Fin n → ℂ}

-- Condition: non-zero complex numbers lying in the half-plane α < arg z < α + π
def in_half_plane (α : ℝ) (z : ℂ) : Prop :=
  z ≠ 0 ∧ α < Complex.arg z ∧ Complex.arg z < α + π

-- We define a list of such complex numbers
def complex_numbers_in_half_plane (α : ℝ) (z : Fin n → ℂ) : Prop :=
  ∀ i, in_half_plane α (z i)

-- Prove: z₁ + ... + zₙ ≠ 0
theorem sum_of_complex_numbers_nonzero (h : complex_numbers_in_half_plane α z) :
  (∑ i, z i) ≠ 0 := 
sorry

-- Prove: 1/z₁ + ... + 1/zₙ ≠ 0
theorem sum_of_reciprocal_complex_numbers_nonzero (h : complex_numbers_in_half_plane α z) :
  (∑ i, (z i)⁻¹) ≠ 0 :=
sorry

end sum_of_complex_numbers_nonzero_sum_of_reciprocal_complex_numbers_nonzero_l231_231996


namespace pascal_fifth_number_l231_231736

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231736


namespace gcd_of_B_l231_231928

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231928


namespace pascal_triangle_row_fifth_number_l231_231726

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231726


namespace multiples_6_8_not_both_l231_231527

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231527


namespace series_1_divergent_series_2_absolutely_convergent_series_3_conditionally_convergent_l231_231117

noncomputable theory

open Complex Topology

def series_1 : (ℕ → Complex) := λ n, (3 - 2 * Complex.I) / (1 + Complex.sqrt n)

def series_2 : (ℕ → Complex) := λ n, (↑n * (3 * Complex.I - 1) ^ n) / 5 ^ n

def series_3 : (ℕ → Complex) := λ n, (-1) ^ n * (1 / (2 * ↑n - 1) + Complex.I / (2 * ↑n + 1))

theorem series_1_divergent : ¬ (Summable series_1) :=
sorry

theorem series_2_absolutely_convergent : Summable (λ n, Complex.abs (series_2 n)) :=
sorry

theorem series_3_conditionally_convergent : (Summable series_3) ∧ ¬ (Summable (λ n, Complex.abs (series_3 n))) :=
sorry

end series_1_divergent_series_2_absolutely_convergent_series_3_conditionally_convergent_l231_231117


namespace payment_to_C_l231_231307

-- Work rates definition
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 8
def combined_work_rate_A_B : ℚ := work_rate_A + work_rate_B
def combined_work_rate_A_B_C : ℚ := 1 / 3

-- C's work rate calculation
def work_rate_C : ℚ := combined_work_rate_A_B_C - combined_work_rate_A_B

-- Payment calculation
def total_payment : ℚ := 3200
def C_payment_ratio : ℚ := work_rate_C / combined_work_rate_A_B_C
def C_payment : ℚ := total_payment * C_payment_ratio

-- Theorem stating the result
theorem payment_to_C : C_payment = 400 := by
  sorry

end payment_to_C_l231_231307


namespace a_n_formula_b_n_sum_formula_l231_231449

section arithmetic_sequence

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Lean uses zero-based index, adjust accordingly
  | _ => 2^(n-1)

theorem a_n_formula (n : ℕ) (hn : n ≠ 0) : a n = 2^(n-1) :=
  sorry

-- Definition of the sequence {b_n}
def b (n : ℕ) : ℕ := 2*n - 1 + a n

-- Sum of the first n terms of {b_n}
def S (n : ℕ) : ℕ := (List.range n).sum b

theorem b_n_sum_formula (n : ℕ) : S n = n^2 + 2^n - 1 :=
  sorry

end arithmetic_sequence

end a_n_formula_b_n_sum_formula_l231_231449


namespace gcd_of_B_is_2_l231_231855

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231855


namespace gcd_elements_of_B_l231_231952

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231952


namespace common_area_of_rectangle_and_circle_l231_231347

theorem common_area_of_rectangle_and_circle :
  let width := 6
  let height := 2
  let radius := 5
  let area_rectangle := width * height
  sqrt (width * width + height * height) <= 2 * radius → 
  area_rectangle = 12 := 
by 
  let width := 6
  let height := 2
  let radius := 5
  let area_rectangle := width * height
  intro h
  have : area_rectangle = 12 := by norm_num
  exact this

end common_area_of_rectangle_and_circle_l231_231347


namespace greatest_two_digit_product_12_l231_231251

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231251


namespace gcd_of_B_is_two_l231_231891

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231891


namespace gcd_elements_of_B_l231_231957

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231957


namespace pascal_triangle_fifth_number_l231_231759

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231759


namespace product_of_five_consecutive_integers_not_square_l231_231169

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : a > 0) :
  ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) :=
by
  sorry

end product_of_five_consecutive_integers_not_square_l231_231169


namespace angle_AEB_l231_231116

-- Definitions based on the conditions
structure Square (V : Type) :=
(A B C D : V) [affine_space V ℝ]
(h_sq : distance A B = distance B C ∧ 
        distance B C = distance C D ∧ 
        distance C D = distance D A ∧
        angle A B C = π/2 ∧ 
        angle B C D = π/2 ∧
        angle C D A = π/2 ∧ 
        angle D A B = π/2)

noncomputable def is_equilateral_trian {V : Type} [affine_space V ℝ] (A B C : V) : Prop :=
distance A B = distance B C ∧ distance B C = distance C A ∧ 
angle A B C = π/3 ∧ angle B C A = π/3 ∧ angle C A B = π/3

-- Theorem statement: Given the conditions, proving the measure of angle AEB is 150 degrees
theorem angle_AEB {V : Type} [affine_space V ℝ] 
  (A B C D E : V) 
  (h_sq : Square V)
  (h_equi: is_equilateral_trian D E C ): 
  angle A E B = 5 * π / 6 := 
sorry

end angle_AEB_l231_231116


namespace min_red_chips_l231_231560

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ w / 3) 
  (h2 : b ≤ r / 4) 
  (h3 : w + b ≥ 72) :
  72 ≤ r :=
by
  sorry

end min_red_chips_l231_231560


namespace pascal_15_5th_number_l231_231764

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231764


namespace arc_length_of_curve_l231_231416

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, sqrt (1 + (deriv f x) ^ 2)

-- Conditions from the problem
def curve (x : ℝ) : ℝ :=
  (1/2) * log ((exp (2 * x) + 1) / (exp (2 * x) - 1))

-- Theorem statement proving the arc length
theorem arc_length_of_curve :
  arc_length curve 1 2 = (1/2) * log (exp (4:ℝ) + 1) - 1 :=
by
  sorry

end arc_length_of_curve_l231_231416


namespace gate_perimeter_l231_231335

theorem gate_perimeter (r : ℝ) (theta : ℝ) (h1 : r = 2) (h2 : theta = π / 2) :
  let arc_length := (3 / 4) * (2 * π * r)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 :=
by
  simp [h1, h2]
  sorry

end gate_perimeter_l231_231335


namespace gcd_of_sum_of_four_consecutive_integers_l231_231868

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231868


namespace fifth_number_in_pascals_triangle_l231_231595

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231595


namespace dodecahedron_interior_diagonals_l231_231039

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231039


namespace dodecahedron_interior_diagonals_l231_231035

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231035


namespace pascal_row_fifth_number_l231_231578

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231578


namespace gcd_of_B_l231_231930

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231930


namespace pascal_fifth_number_l231_231740

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231740


namespace greatest_common_divisor_of_B_l231_231901

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231901


namespace total_trees_after_planting_l231_231211

theorem total_trees_after_planting
  (initial_walnut_trees : ℕ) (initial_oak_trees : ℕ) (initial_maple_trees : ℕ)
  (plant_walnut_trees : ℕ) (plant_oak_trees : ℕ) (plant_maple_trees : ℕ) :
  (initial_walnut_trees = 107) →
  (initial_oak_trees = 65) →
  (initial_maple_trees = 32) →
  (plant_walnut_trees = 104) →
  (plant_oak_trees = 79) →
  (plant_maple_trees = 46) →
  initial_walnut_trees + plant_walnut_trees +
  initial_oak_trees + plant_oak_trees +
  initial_maple_trees + plant_maple_trees = 433 :=
by
  intros
  sorry

end total_trees_after_planting_l231_231211


namespace pascal_fifth_number_in_row_15_l231_231628

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231628


namespace greatest_two_digit_prod_12_l231_231280

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231280


namespace range_of_a_l231_231445

theorem range_of_a (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) (h3 : ∃ (s : set ℕ), s.card = n ∧ ∀ x ∈ s, ∃ y, x = ⌊a * y⌋) :
  1 + (1 / n) ≤ a ∧ a < 1 + (1 / (n - 1)) :=
sorry

end range_of_a_l231_231445


namespace dodecahedron_interior_diagonals_l231_231049

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231049


namespace find_x_l231_231547

theorem find_x (x : ℝ) : 
  16^(x + 2) = 300 + 12 * 16^x → 
  x = Real.log (150/122) / Real.log 16 :=
by
  intro h
  sorry

end find_x_l231_231547


namespace multiples_count_l231_231518

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231518


namespace hall_width_to_length_ratio_l231_231209

def width (w l : ℝ) : Prop := w * l = 578
def length_width_difference (w l : ℝ) : Prop := l - w = 17

theorem hall_width_to_length_ratio (w l : ℝ) (hw : width w l) (hl : length_width_difference w l) : (w / l = 1 / 2) :=
by
  sorry

end hall_width_to_length_ratio_l231_231209


namespace compare_log_values_l231_231378

noncomputable def log2 := Real.log 2

lemma log2_pos : 0 < log2 :=
by sorry -- log 2 is positive

lemma log2_lt_one : log2 < 1 :=
by sorry -- log 2 is less than 1

lemma log2_sq_pos : 0 < log2^2 :=
by sorry -- square of a positive number is positive

lemma log2_sq_lt_log2 : log2^2 < log2 :=
by sorry -- Given 0 < log2 < 1, this implies log2^2 < log2.

lemma log_log2_neg : Real.log log2 < 0 :=
by sorry -- log of a number between 0 and 1 is negative

theorem compare_log_values :
  ∀ x ∈ {log2, log2^2, Real.log log2}, x = log2 ∨ x = log2^2 ∨ x = Real.log log2 →
  (∀ y ∈ {log2, log2^2, Real.log log2}, log2 ≤ y) ∧ 
  (∀ y ∈ {log2, log2^2, Real.log log2}, Real.log log2 ≥ y) :=
by {
  intros x hx, split,
  { intros y hy, apply le_of_lt, exact log2_lt_one, }, -- Proving log2 is the largest
  { intros y hy, apply le_of_lt, exact log_log2_neg } -- Proving log(log2) is the smallest
}

end compare_log_values_l231_231378


namespace yellow_highlighters_l231_231557

def highlighters (pink blue yellow total : Nat) : Prop :=
  (pink + blue + yellow = total)

theorem yellow_highlighters (h : highlighters 3 5 y 15) : y = 7 :=
by 
  sorry

end yellow_highlighters_l231_231557


namespace multiples_six_or_eight_not_both_l231_231535

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231535


namespace dice_probability_sum_17_l231_231079

theorem dice_probability_sum_17 :
  let s : Finset (ℕ × ℕ × ℕ) := 
    (Finset.range 6).image (λ x, (x + 1, x + 1, x + 1))
  ∀ (d1 d2 d3 : ℕ), 
  d1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d3 ∈ {1, 2, 3, 4, 5, 6} → 
  (d1 + d2 + d3 = 17 ↔ (d1, d2, d3) ∈ s) → 
  s.card = 1 / 72 := 
begin
  sorry
end

end dice_probability_sum_17_l231_231079


namespace pascal_triangle_fifth_number_l231_231677

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231677


namespace beehive_cell_surface_description_beehive_cell_total_surface_area_l231_231398

structure BeehiveCell where
  s : ℝ  -- side of the base
  h : ℝ  -- height of the cell
  θ : ℝ   -- angle between TO and TV

def surface_description (cell : BeehiveCell) : Prop :=
  (∃ TUVW TWXY TYZU: Set (Real × Real), ∀ x ∈ [TUVW, TWXY, TYZU], is_rhombus x) ∧
  (∀ A B UV, [ABVU, CBVW, ..., remain 4 trapezoids named appropriately], ∀ x ∈ [ABVU, CBVW, ... remain 4 trapezoids], is_trapezoid x)

def total_surface_area (cell : BeehiveCell) : ℝ :=
  6 * cell.s * cell.h - (9 * (cell.s ^ 2)) / (2 * tan cell.θ) + (3 * (cell.s ^ 2) * Real.sqrt 3) / (2 * sin cell.θ)

theorem beehive_cell_surface_description (cell : BeehiveCell) : surface_description cell :=
  sorry

theorem beehive_cell_total_surface_area (cell : BeehiveCell) : 
  total_surface_area cell = 6 * cell.s * cell.h - (9 * (cell.s ^ 2)) / (2 * tan cell.θ) + (3 * (cell.s ^ 2) * Real.sqrt 3) / (2 * sin cell.θ) :=
  sorry

end beehive_cell_surface_description_beehive_cell_total_surface_area_l231_231398


namespace count_multiples_6_or_8_not_both_l231_231509

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231509


namespace max_students_divide_equal_pen_pencil_l231_231314

theorem max_students_divide_equal_pen_pencil : Nat.gcd 2500 1575 = 25 := 
by
  sorry

end max_students_divide_equal_pen_pencil_l231_231314


namespace greatest_two_digit_prod_12_l231_231284

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l231_231284


namespace beautiful_equals_pairs_plus_one_l231_231131

variable (n : ℕ)
variable (n_ge_2 : n ≥ 2)

-- Definition of a beautiful arrangement for numbers [0, 1, ..., n]
def beautiful_arrangement (arr : List ℕ) : Prop :=
  ∀ (a b c d : ℕ), a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a + c = b + d → ¬ are_chords_intersecting arr a c b d

-- Number of beautiful arrangements of {0, 1, ..., n}
def beautiful_count (n : ℕ) : ℕ :=
  sorry

-- Number of pairs (x, y) where x + y ≤ n and gcd(x, y) = 1
def pairs_count (n : ℕ) : ℕ :=
  sorry

-- Proposition to prove
theorem beautiful_equals_pairs_plus_one : beautiful_count n = pairs_count n + 1 :=
  sorry

end beautiful_equals_pairs_plus_one_l231_231131


namespace fifth_number_in_pascals_triangle_l231_231599

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231599


namespace greatest_int_less_than_neg_17_div_4_l231_231237

theorem greatest_int_less_than_neg_17_div_4 :
  ∃ z : ℤ, z < -17 / 4 ∧ ∀ w : ℤ, w < -17 / 4 → w ≤ z :=
begin
  use -5,
  split,
  { -- -5 < -17 / 4
    have h : (-5 : ℚ) < -17 / 4, from by norm_num,
    exact_mod_cast h, },
  { -- ∀ w : ℤ, w < -17 / 4 → w ≤ -5
    sorry,
  }
end

end greatest_int_less_than_neg_17_div_4_l231_231237


namespace sin_value_in_triangle_l231_231814

   noncomputable def sin_a (a b : ℝ) (cos_b : ℝ) : ℝ :=
     let sin_b := real.sqrt (1 - cos_b^2)
     (a * sin_b) / b
   
   theorem sin_value_in_triangle (a b : ℝ) (cos_b : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : cos_b = 4 / 5) :
     sin_a a b cos_b = 9 / 10 :=
   by
     simp [sin_a, h1, h2, h3]
     sorry
   
end sin_value_in_triangle_l231_231814


namespace number_of_participants_l231_231553

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants_l231_231553


namespace greatest_two_digit_with_product_12_l231_231264

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231264


namespace total_blue_marbles_l231_231148

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end total_blue_marbles_l231_231148


namespace complement_of_angle_is_correct_l231_231549

def degree_minute_to_degree (deg : ℝ) (min : ℝ) : ℝ :=
  deg + min / 60

def complement (α : ℝ) : ℝ :=
  90 - α

theorem complement_of_angle_is_correct :
  let α := degree_minute_to_degree 20 18 in
  complement α = 69.7 := by
{
  let α := degree_minute_to_degree 20 18
  have h : α = 20 + 18 / 60 := rfl
  show complement α = 69.7, from sorry
}

end complement_of_angle_is_correct_l231_231549


namespace greatest_two_digit_product_12_l231_231254

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231254


namespace gcd_of_B_is_two_l231_231880

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231880


namespace boat_speed_still_water_l231_231340

variable (V_b V_s t : ℝ)

-- Conditions given in the problem
axiom speedOfStream : V_s = 13
axiom timeRelation : ∀ t, (V_b + V_s) * t = 2 * (V_b - V_s) * t

-- The statement to be proved
theorem boat_speed_still_water : V_b = 39 :=
by
  sorry

end boat_speed_still_water_l231_231340


namespace a_range_l231_231478

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else if x < 0 then Real.log (-x) / (-Real.log 2) else 0

theorem a_range (a : ℝ) (h : f a > f (-a)) : a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioi (1 : ℝ) :=
sorry

end a_range_l231_231478


namespace find_distance_EF_l231_231122

-- Definitions based on conditions
def rectangle (A B C D : Point) : Prop := 
  collinear A B C D ∧ 
  dist A C = dist B D ∧ 
  dist A B > dist B C

noncomputable def distance (P Q : Point) : ℝ := sorry

def AB := {A B : Point // dist A B > dist B C}
def BC := {A B : Point // dist B C < dist A B}

-- Coordinates for points
variable (a b : ℝ)

def pointE (A B E : Point) : Prop := 
  distance A E = 8 * sqrt 5 ∧
  distance B E = 12 * sqrt 5

def pointF (C D : Point) : Prop := 
  distance C D = 20 * sqrt 5 ∧ 
  midpoint F C D

-- Given and to prove
variable (A B C D E F : Point)

theorem find_distance_EF (a b : ℝ) : 
    rectangle A B C D ∧ 
    distance A E = 8 * sqrt 5 ∧ 
    distance B E = 12 * sqrt 5 ∧ 
    midpoint F C D → 
    EF = 4 * sqrt 35 ∧ 
    (4 + 35 = 39) :=
sorry

end find_distance_EF_l231_231122


namespace nancy_target_amount_l231_231151

theorem nancy_target_amount {rate : ℝ} {hours : ℝ} (h1 : rate = 28 / 4) (h2 : hours = 10) : 28 / 4 * 10 = 70 :=
by
  sorry

end nancy_target_amount_l231_231151


namespace neg_p_l231_231997

variable {α : Type}
variable (x : α)

def p (x : Real) : Prop := ∀ x : Real, x > 1 → x^2 - 1 > 0

theorem neg_p : ¬( ∀ x : Real, x > 1 → x^2 - 1 > 0) ↔ ∃ x : Real, x > 1 ∧ x^2 - 1 ≤ 0 := 
by 
  sorry

end neg_p_l231_231997


namespace exists_K_p_l231_231121

noncomputable def constant_K_p (p : ℝ) (hp : p > 1) : ℝ :=
  (p * p) / (p - 1)

theorem exists_K_p (p : ℝ) (hp : p > 1) :
  ∃ K_p > 0, ∀ x y : ℝ, |x|^p + |y|^p = 2 → (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by
  use constant_K_p p hp
  sorry

end exists_K_p_l231_231121


namespace unpainted_region_area_l231_231227

def board1_width : ℝ := 5
def board2_width : ℝ := 7
def intersect_angle : ℝ := 45 / 180 * Real.pi

theorem unpainted_region_area :
  ∀ (board1 board2 : ℝ), board1 = board1_width ∧ board2 = board2_width →
  ∀ (angle : ℝ), angle = intersect_angle →
  (board1 * board2 = 35) :=
by
  intros
  sorry

end unpainted_region_area_l231_231227


namespace pascal_fifth_element_15th_row_l231_231702

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231702


namespace tangent_k_value_one_common_point_range_l231_231484

namespace Geometry

-- Definitions:
def line (k : ℝ) : ℝ → ℝ := λ x => k * x - 3 * k + 2
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4
def is_tangent (k : ℝ) : Prop := |-2 * k + 3| / (Real.sqrt (k^2 + 1)) = 2
def has_only_one_common_point (k : ℝ) : Prop :=
  (1 / 2 < k ∧ k <= 5 / 2) ∨ (k = 5 / 12)

-- Theorem statements:
theorem tangent_k_value : ∀ k : ℝ, is_tangent k → k = 5 / 12 := sorry

theorem one_common_point_range : ∀ k : ℝ, has_only_one_common_point k → k ∈
  Set.union (Set.Ioc (1 / 2) (5 / 2)) {5 / 12} := sorry

end Geometry

end tangent_k_value_one_common_point_range_l231_231484


namespace option_D_is_negative_l231_231363

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end option_D_is_negative_l231_231363


namespace gcd_of_B_is_2_l231_231838

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231838


namespace ny_mets_fans_count_l231_231090

-- Define the known ratios and total fans
def ratio_Y_to_M (Y M : ℕ) : Prop := 3 * M = 2 * Y
def ratio_M_to_R (M R : ℕ) : Prop := 4 * R = 5 * M
def total_fans (Y M R : ℕ) : Prop := Y + M + R = 330

-- Define what we want to prove
theorem ny_mets_fans_count (Y M R : ℕ) (h1 : ratio_Y_to_M Y M) (h2 : ratio_M_to_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_count_l231_231090


namespace circumcircle_radius_of_triangle_l231_231158

theorem circumcircle_radius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 4)
  (h_angle_ABC : angle_ABC = 120) :
  ∃ (R : ℝ), R = 4 := by
  sorry

end circumcircle_radius_of_triangle_l231_231158


namespace gcd_elements_of_B_l231_231960

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231960


namespace two_b_minus_a_is_13_l231_231385

def binary_representation (n : ℕ) : string := "11111101" -- The binary representation of 253
def a : ℕ := 1 -- Number of zeros in 11111101
def b : ℕ := 7 -- Number of ones in 11111101

theorem two_b_minus_a_is_13 : 2 * b - a = 13 := 
by
  sorry

end two_b_minus_a_is_13_l231_231385


namespace multiples_count_l231_231519

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l231_231519


namespace multiples_six_or_eight_not_both_l231_231533

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231533


namespace dodecahedron_interior_diagonals_l231_231034

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231034


namespace polynomial_coeff_sum_abs_l231_231321

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
    (2 * x - 1)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 242 := by 
  sorry

end polynomial_coeff_sum_abs_l231_231321


namespace Pascal_triangle_fifth_number_l231_231656

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231656


namespace gcd_of_sum_of_four_consecutive_integers_l231_231877

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231877


namespace dodecahedron_interior_diagonals_l231_231026

-- Define the number of vertices and faces in a dodecahedron
def dodecahedron_vertices : ℕ := 20
def dodecahedron_faces : ℕ := 12

-- Each pentagonal face has 5 vertices
def vertices_per_face : ℕ := 5

-- Each vertex connects to other vertices by edges on three adjacent faces
def adjacent_faces_per_vertex : ℕ := 3

-- Total potential connections per vertex
def potential_connections_per_vertex : ℕ := dodecahedron_vertices - 1

-- Define interior diagonals as segments connecting vertices not lying on the same face
noncomputable def interior_diagonals (vertices pentagons faces_per_vertex potential_connections adjacent_faces : ℕ) : ℕ :=
  let internal_connections := potential_connections - (adjacent_faces + vertices_per_face - 2)
  (vertices * internal_connections) / 2

theorem dodecahedron_interior_diagonals :
  interior_diagonals dodecahedron_vertices vertices_per_face adjacent_faces_per_vertex 
                  potential_connections_per_vertex vertices_per_face = 120 :=
  sorry

end dodecahedron_interior_diagonals_l231_231026


namespace greatest_two_digit_product_is_12_l231_231241

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231241


namespace minimum_angle_of_inclined_line_with_plane_l231_231164

theorem minimum_angle_of_inclined_line_with_plane
  (l : Line) (φ : Plane) (A : Point) (α : ℝ)
  (intersects_at : l.intersects φ A)
  (angle_with_plane : l.angle_with φ = α) :
  ∀ (a : Line), (a ∈ φ) → ∃ β : ℝ, line.angle_with a = β → α ≤ β :=
by
  sorry

end minimum_angle_of_inclined_line_with_plane_l231_231164


namespace gcd_B_eq_two_l231_231866

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231866


namespace greatest_two_digit_product_is_12_l231_231240

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231240


namespace gcd_of_B_l231_231935

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231935


namespace gcd_B_is_2_l231_231915

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231915


namespace oxen_a_grazing_l231_231311

theorem oxen_a_grazing (a b c rent cost_c : ℝ) (x : ℝ) : 
  a * 7 + b + c = rent → c = cost_c → x = 17 :=
by 
  intro h1 h2
  have h3 : a * 7 + b + cost_c = rent, from h1,
  have h4 : c = 15 * 3, from h2,
  sorry

end oxen_a_grazing_l231_231311


namespace isosceles_triangle_largest_angle_l231_231567

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : is_triangle A B C) (h2 : A = B) (h3 : C = 30) : ∃ D, D = 120 :=
by
  sorry

end isosceles_triangle_largest_angle_l231_231567


namespace seven_digit_palindromes_l231_231495

theorem seven_digit_palindromes : 
  let digits := {1, 1, 4, 4, 6, 6, 6} in 
  (count_palindromes digits = 6) :=
by
  sorry

end seven_digit_palindromes_l231_231495


namespace gcd_of_B_l231_231931

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231931


namespace dodecahedron_interior_diagonals_l231_231031

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l231_231031


namespace a_perp_a_add_b_l231_231490

def vector (α : Type*) := α × α

def a : vector ℤ := (2, -1)
def b : vector ℤ := (1, 7)

def dot_product (v1 v2 : vector ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def add_vector (v1 v2 : vector ℤ) : vector ℤ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def perpendicular (v1 v2 : vector ℤ) : Prop :=
  dot_product v1 v2 = 0

theorem a_perp_a_add_b :
  perpendicular a (add_vector a b) :=
by {
  sorry
}

end a_perp_a_add_b_l231_231490


namespace Pascal_triangle_fifth_number_l231_231653

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231653


namespace greatest_two_digit_product_12_l231_231249

-- Definition of a two-digit whole number
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Definition of the digit product condition
def digits_product (n : ℕ) (p : ℕ) : Prop := ∃ (d1 d2 : ℕ), d1 * d2 = p ∧ n = 10 * d1 + d2

-- The main theorem stating the greatest two-digit number whose digits multiply to 12 is 62
theorem greatest_two_digit_product_12 : ∀ (n : ℕ), is_two_digit (n) → digits_product (n) 12 → n <= 62 :=
by {
    sorry -- Proof of the theorem
}

end greatest_two_digit_product_12_l231_231249


namespace inequality_may_not_hold_l231_231062

theorem inequality_may_not_hold (a b c : ℝ) (h : a > b) : (c < 0) → ¬ (a/c > b/c) := 
sorry

end inequality_may_not_hold_l231_231062


namespace gcd_elements_of_B_l231_231956

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231956


namespace area_of_common_region_l231_231219

noncomputable def common_area_three_arcs : ℝ :=
  let r := 6
  let θ := real.pi / 3 -- 120 degrees in radians
  let triangle_side_length := 2 * r
  let area_triangle := (real.sqrt 3) / 4 * (triangle_side_length ^ 2)
  let sector_area := 1 / 3 * real.pi * (r ^ 2)
  let triangle_sector_area := 1 / 2 * (r ^ 2) * real.sin θ
  let segment_area := sector_area - triangle_sector_area
  3 * triangle_sector_area + area_triangle - (3 * segment_area)

theorem area_of_common_region :
  common_area_three_arcs = 63 * real.sqrt 3 - 36 * real.pi :=
by
  sorry

end area_of_common_region_l231_231219


namespace standard_eq_line_l_cartesian_eq_curve_C_max_distance_P_to_l_l231_231141

-- Definitions for the parametric equation of line l
def parametric_eq_line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Definitions for the polar equation of curve C
def polar_eq_curve_C (θ : ℝ) : ℝ :=
  Real.sqrt (12 / (3 * Real.cos θ ^ 2 + 4 * Real.sin θ ^ 2))

-- Lean 4 formalization
theorem standard_eq_line_l : ∀ t : ℝ, parametric_eq_line_l t = (x, y) → x - y - 2 = 0 :=
sorry

theorem cartesian_eq_curve_C : ∀ (ρ : ℝ) (θ : ℝ),
  (x, y) = polar_to_cartesian ρ θ → ρ = polar_eq_curve_C θ → x ^ 2 / 4 + y ^ 2 / 3 = 1 :=
sorry

theorem max_distance_P_to_l : 
  ∃ θ₀ : ℝ, θθ₀,
  ∀ P : ℝ × ℝ, P ∈ ellipse 4 3 → max_dist P parametric_eq_line_l = Real.sqrt 14 / 2 + Real.sqrt 2 :=
sorry

end standard_eq_line_l_cartesian_eq_curve_C_max_distance_P_to_l_l231_231141


namespace pascal_fifteen_four_l231_231782

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231782


namespace pascal_triangle_row_fifth_number_l231_231716

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231716


namespace g_neg_eq_g_l231_231984

variable {x : ℝ}

def g (x : ℝ) : ℝ := (x ^ 2 + 3 * x + 2) / (x ^ 2 - 1)

theorem g_neg_eq_g (hx : x^2 ≠ 1) : g (-x) = g x := by
  -- Proof is omitted
  sorry

end g_neg_eq_g_l231_231984


namespace GCD_of_set_B_is_2_l231_231950

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231950


namespace greatest_two_digit_with_product_12_l231_231268

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231268


namespace pascal_triangle_fifth_number_l231_231670

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231670


namespace Pascal_triangle_fifth_number_l231_231659

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231659


namespace pascal_fifteen_four_l231_231791

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231791


namespace num_integers_satisfy_inequality_l231_231498

theorem num_integers_satisfy_inequality :
  {n : ℤ | (n+5) * (n-9) ≤ 0}.to_finset.card = 15 :=
by
  sorry

end num_integers_satisfy_inequality_l231_231498


namespace sin_15_eq_l231_231420

-- Definition of sine and angle in radians
def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / 180

theorem sin_15_eq : Real.sin (deg_to_rad 15) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end sin_15_eq_l231_231420


namespace no_positive_integers_l231_231546

theorem no_positive_integers (m : ℕ) (hm : 2 ≤ m) :
  ∀ (x : ℕ → ℕ), (∀ i, i < m → 0 < x i) → (∀ i j, i < j → x i < x j) →
  (∑ i in finset.range m, (x i)⁻³ : ℝ) ≠ 1 :=
sorry

end no_positive_integers_l231_231546


namespace num_subcommittees_from_seven_l231_231496

theorem num_subcommittees_from_seven :
  (nat.choose 7 3) = 35 :=
by
  sorry

end num_subcommittees_from_seven_l231_231496


namespace sin_double_angle_l231_231472

variable (α : ℝ) (x y : ℝ)
variables (P : ℝ × ℝ)

-- Conditions
def unit_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 = 1

def point_P_condition (P : ℝ × ℝ) : Prop :=
  P.1 = 1 / 2

def alpha_condition (α : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, α = (π / 3 + 2 * k * π) ∨ α = (-π / 3 + 2 * k * π)

-- Question
theorem sin_double_angle (α : ℝ) (P : ℝ × ℝ) (h1 : unit_circle P) (h2 : point_P_condition P) (h3 : alpha_condition α P) :
  sin (π / 2 + α) = 1 / 2 :=
sorry

end sin_double_angle_l231_231472


namespace greatest_two_digit_with_product_12_l231_231262

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231262


namespace product_of_five_consecutive_integers_not_square_l231_231171

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : 0 < a) :
  ¬ is_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) := by
sorry

end product_of_five_consecutive_integers_not_square_l231_231171


namespace logarithm_graph_passes_through_point_l231_231550

theorem logarithm_graph_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : log a 1 = 0 :=
by
  sorry

end logarithm_graph_passes_through_point_l231_231550


namespace terminating_decimal_count_l231_231424

theorem terminating_decimal_count (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 499) :
  (∀ k, n = 7 * k → ∃ m : ℕ, ((m : ℚ) / 700).denom = 1) :=
sorry

end terminating_decimal_count_l231_231424


namespace man_l231_231341

theorem man's_age (x : ℕ) : 6 * (x + 6) - 6 * (x - 6) = x → x = 72 :=
by
  sorry

end man_l231_231341


namespace pascal_fifth_number_in_row_15_l231_231621

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231621


namespace anne_distance_l231_231068

theorem anne_distance :
  let speed := 2
  let time := 3
  let distance := speed * time
  in distance = 6 :=
by
  let speed := 2
  let time := 3
  let distance := speed * time
  show distance = 6 from sorry

end anne_distance_l231_231068


namespace pascal_row_fifth_number_l231_231579

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231579


namespace pascal_triangle_15_4_l231_231616

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231616


namespace pascal_fifth_number_l231_231744

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l231_231744


namespace multiples_of_6_or_8_under_201_not_both_l231_231525

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231525


namespace algebraic_expression_value_l231_231066

theorem algebraic_expression_value (a b : ℤ) (h : 2 * (-3) - a + 2 * b = 0) : 2 * a - 4 * b + 1 = -11 := 
by {
  sorry
}

end algebraic_expression_value_l231_231066


namespace gcd_elements_of_B_l231_231953

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231953


namespace cape_may_multiple_l231_231388

theorem cape_may_multiple :
  ∃ x : ℕ, 26 = x * 7 + 5 ∧ x = 3 :=
by
  sorry

end cape_may_multiple_l231_231388


namespace six_coins_not_sum_to_14_l231_231175

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem six_coins_not_sum_to_14 (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 ∈ coin_values) (h2 : a2 ∈ coin_values) (h3 : a3 ∈ coin_values) (h4 : a4 ∈ coin_values) (h5 : a5 ∈ coin_values) (h6 : a6 ∈ coin_values) : a1 + a2 + a3 + a4 + a5 + a6 ≠ 14 := 
sorry

end six_coins_not_sum_to_14_l231_231175


namespace Pascal_triangle_fifth_number_l231_231661

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231661


namespace pascal_fifteen_four_l231_231795

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231795


namespace Pascal_triangle_fifth_number_l231_231652

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231652


namespace fifth_number_in_pascals_triangle_l231_231592

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231592


namespace solve_equation_l231_231395

theorem solve_equation (x : ℚ) :
  (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) → x = -3 / 2 :=
by
  sorry

end solve_equation_l231_231395


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231966

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231966


namespace pascal_fifth_element_15th_row_l231_231709

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231709


namespace find_p_q_r_sum_l231_231487

def A (p : ℝ) : set ℝ := {x | x^2 - p*x - 2 = 0}
def B (q r : ℝ) : set ℝ := {x | x^2 + q*x + r = 0}
def union_prop (p q r : ℝ) : Prop := A p ∪ B q r = {-2, 1, 5}
def intersection_prop (p q r : ℝ) : Prop := A p ∩ B q r = {-2}

theorem find_p_q_r_sum (p q r : ℝ) (h_union : union_prop p q r) (h_inter : intersection_prop p q r) : p + q + r = -14 :=
by
  sorry

end find_p_q_r_sum_l231_231487


namespace gcd_of_sum_of_four_consecutive_integers_l231_231879

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231879


namespace pascal_fifteen_four_l231_231793

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231793


namespace multiples_6_8_not_both_l231_231531

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231531


namespace vendor_throws_away_correct_percent_l231_231357

def initial_apples : ℕ := 100

def day1_sold (initial : ℕ) : ℕ := (55.5 / 100) * initial
def day1_remaining (initial : ℕ) : ℕ := initial - day1_sold initial
def day1_thrown (remaining : ℕ) : ℕ := (1 / 3) * remaining

def day2_sold (remaining : ℕ) : ℕ := (47 / 100) * remaining
def day2_remaining (remaining : ℕ) : ℕ := remaining - day2_sold remaining
def day2_thrown (remaining : ℕ) : ℕ := (35 / 100) * remaining

def day3_sold (remaining : ℕ) : ℕ := (62.5 / 100) * remaining
def day3_remaining (remaining : ℕ) : ℕ := remaining - day3_sold remaining
def day3_thrown (remaining : ℕ) : ℕ := (1 / 2) * remaining

def day4_sold (remaining : ℕ) : ℕ := (28.7 / 100) * remaining
def day4_remaining (remaining : ℕ) : ℕ := remaining - day4_sold remaining
def day4_thrown (remaining : ℕ) : ℕ := (20.1 / 100) * remaining

def total_thrown : ℕ :=
  let after_day1 := day1_remaining initial_apples
  let thrown_day1 := day1_thrown after_day1
  let after_day2 := day2_remaining (after_day1 - thrown_day1)
  let thrown_day2 := day2_thrown after_day2
  let after_day3 := day3_remaining (after_day2 - thrown_day2)
  let thrown_day3 := day3_thrown after_day3
  let after_day4 := day4_remaining (after_day3 - thrown_day3)
  let thrown_day4 := day4_thrown after_day4
  thrown_day1 + thrown_day2 + thrown_day3 + thrown_day4

def percentage (part whole : ℕ) : ℕ := (part * 100) / whole

theorem vendor_throws_away_correct_percent : percentage total_thrown initial_apples = 22.53 := sorry

end vendor_throws_away_correct_percent_l231_231357


namespace greatest_two_digit_with_product_12_l231_231271

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231271


namespace gcd_of_B_l231_231933

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231933


namespace gcd_of_B_is_2_l231_231841

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231841


namespace conditional_probability_l231_231432

variable (P : Set → Rat)
variable (A B : Set)

axiom P_A : P A = 3 / 5
axiom P_AB : P (A ∩ B) = 3 / 10

theorem conditional_probability 
  (h_cond : P A ≠ 0) : P B * P A = 3 / 10 → (P (A ∩ B) = 3 / 10) → P (A) = 3 / 5 → P B = 1 / 2 :=
by
  sorry

end conditional_probability_l231_231432


namespace gcd_of_B_is_two_l231_231883

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231883


namespace inversion_preserves_tangency_or_transforms_tangent_entities_l231_231162

-- Definitions taken from the problem conditions
noncomputable def angle_between_circles (c1 c2 : Circle) : Angle := sorry

noncomputable def angle_between_circle_and_line (c : Circle) (l : Line) : Angle := sorry

-- Statement of the theorem to be proven
theorem inversion_preserves_tangency_or_transforms_tangent_entities
  (circle : Circle) (line : Line)
  (inverted_circle : (Circle → Circle)) (inverted_line : (Line → Line))
  (tangent : Tangent circle line) :
  Tangent (inverted_circle circle) (inverted_line line) ∨
  TangentCircles (inverted_circle circle) (inverted_circle circle) ∨
  ParallelLines (inverted_line line) (inverted_line line) := 
sorry

end inversion_preserves_tangency_or_transforms_tangent_entities_l231_231162


namespace parabola_vertex_range_l231_231078

def parabola_vertex_in_first_quadrant (m : ℝ) : Prop :=
  ∃ v : ℝ × ℝ, v = (m, m - 1) ∧ 0 < m ∧ 0 < (m - 1)

theorem parabola_vertex_range (m : ℝ) (h_vertex : parabola_vertex_in_first_quadrant m) :
  1 < m :=
by
  sorry

end parabola_vertex_range_l231_231078


namespace total_subjects_l231_231371

theorem total_subjects (n T : ℕ) (h1 : ∀ m, average m n = 76)
  (h2 : average (first_n_marks 5 n) 5 = 74) (h3 : last_mark n = 86) :
  n = 6 := 
sorry

end total_subjects_l231_231371


namespace pascal_triangle_row_fifth_number_l231_231724

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231724


namespace pascal_fifteen_four_l231_231786

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l231_231786


namespace fifth_number_in_pascals_triangle_l231_231594

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231594


namespace fifth_number_in_pascal_row_l231_231806

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231806


namespace calculate_tax_l231_231392

noncomputable def cadastral_value : ℝ := 3000000 -- 3 million rubles
noncomputable def tax_rate : ℝ := 0.001        -- 0.1% converted to decimal
noncomputable def tax : ℝ := cadastral_value * tax_rate -- Tax formula

theorem calculate_tax : tax = 3000 := by
  sorry

end calculate_tax_l231_231392


namespace perimeter_of_triangle_l231_231466

noncomputable def ellipse_perimeter (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) : ℝ :=
  let a := 2
  let c := 1
  2 * a + 2 * c

theorem perimeter_of_triangle (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) :
  ellipse_perimeter x y h = 6 :=
by 
  sorry

end perimeter_of_triangle_l231_231466


namespace number_of_correct_conclusions_l231_231153

theorem number_of_correct_conclusions
  (a b c : ℕ)
  (h1 : (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) 
  (conclusion1 : (a^b - b^c) % 2 = 1 ∧ (b^c - c^a) % 2 = 1 ∧ (c^a - a^b) % 2 = 1)
  (conclusion4 : ¬ ∃ a b c : ℕ, (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713) :
  ∃ n : ℕ, n = 2 :=
by
  sorry

end number_of_correct_conclusions_l231_231153


namespace pascal_row_fifth_number_l231_231580

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231580


namespace incorrect_statement_for_linear_function_l231_231426

theorem incorrect_statement_for_linear_function :
  let f := λ x : ℝ, x + 2 in
  ¬(∀ x > 2, f x < 4) :=
by
  let f := λ x : ℝ, x + 2
  sorry

end incorrect_statement_for_linear_function_l231_231426


namespace f_six_equals_twenty_two_l231_231092

-- Definitions as per conditions
variable (n : ℕ) (f : ℕ → ℕ)

-- Conditions of the problem
-- n is a natural number greater than or equal to 3
-- f(n) satisfies the properties defined in the given solution
axiom f_base : f 1 = 2
axiom f_recursion {k : ℕ} (hk : k ≥ 1) : f (k + 1) = f k + (k + 1)

-- Goal to prove
theorem f_six_equals_twenty_two : f 6 = 22 := sorry

end f_six_equals_twenty_two_l231_231092


namespace original_concentration_A_l231_231218

-- Definitions of initial conditions and parameters
def mass_A : ℝ := 2000 -- 2 kg in grams
def mass_B : ℝ := 3000 -- 3 kg in grams
def pour_out_A : ℝ := 0.15 -- 15% poured out from bottle A
def pour_out_B : ℝ := 0.30 -- 30% poured out from bottle B
def mixed_concentration1 : ℝ := 27.5 -- 27.5% concentration after first mix
def pour_out_restored : ℝ := 0.40 -- 40% poured out again

-- Using the calculated remaining mass and concentration to solve the proof
theorem original_concentration_A (x y : ℝ) 
  (h1 : 300 * x + 900 * y = 27.5 * (300 + 900)) 
  (h2 : (1700 * x + 300 * 27.5) * 0.4 / (2000 * 0.4) + (2100 * y + 900 * 27.5) * 0.4 / (3000 * 0.4) = 26) : 
  x = 20 :=
by 
  -- Skipping the proof. The proof should involve solving the system of equations.
  sorry

end original_concentration_A_l231_231218


namespace greatest_common_divisor_of_B_l231_231923

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l231_231923


namespace Pascal_triangle_fifth_number_l231_231657

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231657


namespace gcd_of_B_is_2_l231_231851

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l231_231851


namespace fifth_number_in_pascal_row_l231_231797

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231797


namespace supplementary_angles_side_false_l231_231300

theorem supplementary_angles_side_false :
  ¬ (∀ (a b : ℝ), a + b = 180 → (a ≠ b → ∃ side, false)) :=
by
  sorry

end supplementary_angles_side_false_l231_231300


namespace convex_cyclic_quad_min_area_100m_n_l231_231828

theorem convex_cyclic_quad_min_area_100m_n (ABCD : Type)
  [convex ABCD]
  [cyclic ABCD]
  (AC BD : ℝ)
  (H_AC : AC = 4)
  (H_BD : BD = 5)
  (H_perp : ∀ A B C D : ABCD, ⟦A, B⟧ ⊥ ⟦C, D⟧) :
  ∃ m n : ℕ, ∀ (m n : ℕ), λ = 90 / 41 ∧ 100 * m + n = 9041 := sorry

end convex_cyclic_quad_min_area_100m_n_l231_231828


namespace multiples_of_6_or_8_under_201_not_both_l231_231521

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l231_231521


namespace pascal_triangle_row_fifth_number_l231_231727

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231727


namespace gcd_of_B_l231_231934

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l231_231934


namespace GCD_of_set_B_is_2_l231_231942

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l231_231942


namespace pascal_triangle_fifth_number_l231_231636

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231636


namespace gcd_B_eq_two_l231_231864

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231864


namespace gcd_12345_6789_l231_231235

theorem gcd_12345_6789 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l231_231235


namespace gcd_of_B_is_two_l231_231885

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231885


namespace fifth_number_in_pascal_row_l231_231801

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231801


namespace problem_a_problem_b_l231_231137

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define when a pair (a, b) is n-good
def is_ngood (n a b : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ P a b m - P a b k → n ∣ m - k

-- Define when a pair (a, b) is very good
def is_verygood (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ (infinitely_many_n : ℕ), is_ngood n a b

-- Problem (a): Find a pair (1, -51^2) which is 51-good but not very good
theorem problem_a : ∃ a b : ℤ, a = 1 ∧ b = -(51^2) ∧ is_ngood 51 a b ∧ ¬is_verygood a b := 
by {
  sorry
}

-- Problem (b): Show that all 2010-good pairs are very good
theorem problem_b : ∀ a b : ℤ, is_ngood 2010 a b → is_verygood a b := 
by {
  sorry
}

end problem_a_problem_b_l231_231137


namespace eccentricity_of_ellipse_l231_231475

noncomputable def eccentricity_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (d1 d2 : ℝ) (c : ℝ) := 
  (d1 = a - c) ∧ (d2 = a + c) ∧ (d1 + 2 * c = d2) → (c = a / 2) → (a > 0) → (b > 0) → (2 * c = a) → 
  let e := c / a in e = 1 / 2

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (d1 d2 : ℝ) (c : ℝ) :
  eccentricity_problem a b h1 h2 d1 d2 c := by 
  sorry

end eccentricity_of_ellipse_l231_231475


namespace lines_passing_through_two_points_l231_231506

/-- Definition for the number of lines passing through k points in the hexagonal grid -/
def lines_passing_through_k_points (k : ℕ) : ℕ :=
  if k = 3 then 15
  else if k = 4 then 6
  else if k = 5 then 3
  else 0

/-- Main problem statement -/
theorem lines_passing_through_two_points : 
  let total_pairs := nat.choose 19 2,
      correction_3 := 15 * nat.choose 3 2,
      correction_4 := 6 * nat.choose 4 2,
      correction_5 := 3 * nat.choose 5 2 in
  total_pairs - correction_3 - correction_4 - correction_5 = 60 := 
by
  let total_pairs := nat.choose 19 2,
      correction_3 := 15 * nat.choose 3 2,
      correction_4 := 6 * nat.choose 4 2,
      correction_5 := 3 * nat.choose 5 2 in
  have h1 : total_pairs = 171 := by sorry,
  have h2 : correction_3 = 45 := by sorry,
  have h3 : correction_4 = 36 := by sorry,
  have h4 : correction_5 = 30 := by sorry,
  calc 
    total_pairs - correction_3 - correction_4 - correction_5
        = 171 - 45 - 36 - 30 : by rw [h1, h2, h3, h4]
    ... = 60 : by norm_num

end lines_passing_through_two_points_l231_231506


namespace pascal_triangle_fifth_number_l231_231688

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231688


namespace square_perimeter_l231_231183

theorem square_perimeter (A : ℝ) (hA : A = 325) : ∃ P : ℝ, P = 20 * real.sqrt 13 ∧ (∃ s : ℝ, s = (real.sqrt A) ∧ P = 4 * s) :=
by {
  let s := (real.sqrt A),
  have hs : s = real.sqrt 325, { rw hA },
  let P := 4 * s,
  have hP : P = 20 * real.sqrt 13,
  {
    rw [hs, real.sqrt_mul (show 0 ≤ 325, by norm_num), real.sqrt_mul (show 0 ≤ 25, by norm_num)],
    norm_num,
  },
  use P,
  exact ⟨hP, ⟨s, hs, rfl⟩⟩,
}

end square_perimeter_l231_231183


namespace pascal_triangle_15_4_l231_231604

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231604


namespace fifth_number_in_pascal_row_l231_231796

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231796


namespace pascal_triangle_fifth_number_l231_231645

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231645


namespace seating_arrangement_l231_231336

theorem seating_arrangement (m e d i : ℕ) (total_employees : ℕ) 
  (h1 : m = 5) (h2 : e = 2) (h3 : d = 4) (h4 : i = 2) (h5 : total_employees = 13) : 
  (fact (total_employees - 1) / (fact m * fact e * fact d * fact i)) - 
  (fact (total_employees - 2) / (fact m * fact e * fact d * fact (i - 1))) = 55440 := 
  by sorry

end seating_arrangement_l231_231336


namespace sum_of_first_twelve_multiples_of_18_l231_231292

-- Given conditions
def sum_of_first_n_positives (n : ℕ) : ℕ := n * (n + 1) / 2

def first_twelve_multiples_sum (k : ℕ) : ℕ := k * (sum_of_first_n_positives 12)

-- The question to prove
theorem sum_of_first_twelve_multiples_of_18 : first_twelve_multiples_sum 18 = 1404 :=
by
  sorry

end sum_of_first_twelve_multiples_of_18_l231_231292


namespace pascal_row_fifth_number_l231_231583

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l231_231583


namespace dodecahedron_interior_diagonals_l231_231007

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231007


namespace range_x_greater_g_condition_l231_231452

variables {R : Type} [LinearOrderedField R]

-- Define odd function f and its derivative
variable (f : R → R)
variable (f' : R → R)
variable (h_odd_f : ∀ x, f (-x) = -f x)
variable (h_deriv_f : ∀ x, has_deriv_at f (f' x) x)

-- Given condition: for x ∈ (0, +∞), xf'(x) < f(-x)
variable (h_condition : ∀ x, 0 < x → x * f' x < f (-x))

-- Define g function
def g (x : R) := x * f x

-- Theorem to prove the range of x such that g(1) > g(1 - 2x)
theorem range_x_greater_g_condition :
  ∀ (x : R), g 1 > g (1 - 2 * x) ↔ x < 0 ∨ x > 1 :=
begin
  sorry
end

end range_x_greater_g_condition_l231_231452


namespace first_number_positive_from_initial_pair_l231_231114

theorem first_number_positive_from_initial_pair :
  ∀ (x y : ℝ), (x > 0) ∧ (1 ≤ y) ∧ 
  (∀ (x y : ℝ), (x > 0) ∧ (1 ≤ y) → ((x, y - 1) ∨ (x + y, y + 1) ∨ (x, x * y) ∨ (1 / x, y))) →
  ∀ p : ℝ × ℝ, p ∈ { initial_pair := (1,1) :
     set (ℝ × ℝ) | ∀ q : ℝ × ℝ, (q = (x, y - 1) ∧ x > 0 ) ∨ (q = (x + y, y + 1) ∧ x > 0) ∨
      (q = (x, x * y) ∧ x > 0 ) ∨ (q = (1 / x, y) ∧ x > 0 ) } → 
  fst p > 0 :=
sorry

end first_number_positive_from_initial_pair_l231_231114


namespace triangle_DEF_sum_of_possible_values_of_EF_l231_231112

noncomputable def DE := 100 * Real.sqrt 2
noncomputable def DF := 100
noncomputable def angle_E := 45

theorem triangle_DEF_sum_of_possible_values_of_EF :
  ∃ (EF : ℝ), (angle_E = 45) ∧ (DE = 100 * Real.sqrt 2) ∧ (DF = 100) ∧ (EF = 100 * Real.sqrt 3) :=
by
  use (100 * Real.sqrt 3)
  simp [angle_E, DE, DF]
  sorry

end triangle_DEF_sum_of_possible_values_of_EF_l231_231112


namespace pascal_triangle_fifth_number_l231_231755

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231755


namespace greatest_common_divisor_of_B_l231_231897

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231897


namespace intersection_of_A_and_B_l231_231456

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | abs (x^2 - 1) ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = A :=
sorry

end intersection_of_A_and_B_l231_231456


namespace multiples_six_or_eight_not_both_l231_231537

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l231_231537


namespace max_value_18_l231_231129

section Problem

def S : Finset ℕ := Finset.range 50 \ {0}

def valid_arrangement (l : List ℕ) : Prop := 
  ∀ (i : Fin l.length), (l.nthLe i (by simp)) * (l.nthLe ((i + 1) % l.length) (by simp)) < 100

theorem max_value_18 : ∃ l : List ℕ, l.toFinset ⊆ S ∧ l.length = 18 ∧ valid_arrangement l :=
sorry

end Problem

end max_value_18_l231_231129


namespace distance_between_x_intercepts_l231_231228

noncomputable def line_eq (slope : ℝ) (x y : ℝ) : ℝ := slope * x - ( slope * 4 - 6)

theorem distance_between_x_intercepts :
  let L1 := line_eq 2 in
  let L2 := line_eq 6 in 
  let x1 := (- L1 0 0) / 2 in
  let x2 := (- L2 0 0) / 6 in
  abs (x2 - x1) = 2 :=
by
  -- Definitions for points and lines
  let L1 := line_eq 2
  let L2 := line_eq 6 
  let x1 := (-(6 - L1 0 0)) / 2
  let x2 := (-(6 - L2 0 0)) / 6
  -- The proof is omitted
  sorry

end distance_between_x_intercepts_l231_231228


namespace dodecahedron_interior_diagonals_l231_231055

theorem dodecahedron_interior_diagonals :
  ∀ (dodecahedron : Type) (has_12_faces : ∃ (f : dodecahedron → Prop), ∃ F : finset dodecahedron, F.card = 12 ∧ ∀ f ∈ F, ∃! (p : dodecahedron) → Prop, f p) 
    (has_20_vertices : fintype.card dodecahedron = 20) 
    (three_faces_per_vertex : ∀ v : dodecahedron, ∃! F : finset dodecahedron, F.card = 3 ∧ ∀ f ∈ F, (v ∈ f)) 
    (not_common_face : ∀ v w : dodecahedron, v ≠ w → (∃ f₁ f₂, f₁ ≠ f₂ ∧ ¬ (v ∈ f₁ ∧ w ∈ f₁) ∧ ¬ (v ∈ f₂ ∧ w ∈ f₂) ∧ (f₁ ∉ [f₂]))),
  130 :=
by
  -- formalize the proof steps here
  sorry

end dodecahedron_interior_diagonals_l231_231055


namespace unicorns_fly_and_some_magical_are_unicorns_implies_some_flying_are_magical_l231_231563

variable (Unicorn : Type) (MagicalCreature : Type) (FlyingCreature : Type)
variable (u_is_uc : Unicorn → Prop)
variable (m_is_uc : MagicalCreature → Prop)
variable (f_is_fc : FlyingCreature → Prop)

-- Conditions
variable (all_unicorns_can_fly : ∀ u : Unicorn, f_is_fc u)
variable (some_magical_are_unicorns : ∃ m : MagicalCreature, u_is_uc m)

-- Statement II: Some flying creatures are magical creatures
theorem unicorns_fly_and_some_magical_are_unicorns_implies_some_flying_are_magical :
  (∃ f : FlyingCreature, m_is_uc f) :=
sorry

end unicorns_fly_and_some_magical_are_unicorns_implies_some_flying_are_magical_l231_231563


namespace dodecagon_radius_l231_231368

/-- A convex dodecagon inscribed in a circle with alternating side lengths sqrt(2) and sqrt(24) -/
def dodecagon_inscribed (r : ℝ) : Prop :=
  r > 0 ∧ 
  (∀ i : ℕ, i < 12 → (
    (i % 2 = 0 → sqrt(2) = 2 * r * sin ((π * 1) / 12)) ∧ 
    (i % 2 = 1 → sqrt(24) = 2 * r * sin ((π * 1) / 12))
  ))

theorem dodecagon_radius :
  ∃ r, dodecagon_inscribed r ∧ r = sqrt(38) :=
by
  sorry

end dodecagon_radius_l231_231368


namespace multiples_6_8_not_both_l231_231528

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l231_231528


namespace ratio_MN_l231_231548

variables (Q P R M N : ℝ)

def satisfies_conditions (Q P R M N : ℝ) : Prop :=
  M = 0.40 * Q ∧
  Q = 0.25 * P ∧
  R = 0.60 * P ∧
  N = 0.50 * R

theorem ratio_MN (Q P R M N : ℝ) (h : satisfies_conditions Q P R M N) : M / N = 1 / 3 :=
by {
  sorry
}

end ratio_MN_l231_231548


namespace count_multiples_6_or_8_not_both_l231_231511

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l231_231511


namespace sin_angle_HAC_proof_l231_231322

noncomputable def sin_angle_HAC (AB AD AE BC EH : ℝ) : ℝ :=
  let HA := real.sqrt (EH ^ 2 + AD ^ 2)
  let AC := real.sqrt (AB ^ 2 + BC ^ 2)
  let HC := EH
  let cos_HAC := (HA ^ 2 - AC ^ 2 - HC ^ 2) / (-2 * AC * HC)
  real.sqrt (1 - cos_HAC ^ 2)

theorem sin_angle_HAC_proof :
  sin_angle_HAC 1 1 1 2 3 = real.sqrt (14 / 15) :=
by
  sorry

end sin_angle_HAC_proof_l231_231322


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l231_231973

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l231_231973


namespace gcd_B_is_2_l231_231908

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231908


namespace pascal_triangle_fifth_number_l231_231678

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231678


namespace gcd_of_B_is_2_l231_231837

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231837


namespace product_mod_eq_l231_231289

theorem product_mod_eq :
  (1497 * 2003) % 600 = 291 := 
sorry

end product_mod_eq_l231_231289


namespace ellipse_eq_triangle_area_l231_231118

-- Definitions based on the conditions given
def f1 : ℝ × ℝ := (-1, 0)
def f2 : ℝ × ℝ := (1, 0)
def c : ℝ := 1 -- half the distance between the foci
def a : ℝ := 2 -- semi-major axis length
def b : ℝ := real.sqrt (a ^ 2 - c ^ 2) -- semi-minor axis length

-- Proof for question (1) - Ellipse equation
theorem ellipse_eq : ∃ a b : ℝ, (a = 2) ∧ (b = sqrt 3) ∧ ∀ x y : ℝ, ((x^2) / (a^2) + (y^2) / (b^2) = 1) :=
sorry

-- Definitions for question (2)
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  (fst P)^2 / 4 + (snd P)^2 / 3 = 1
  
def p1 : ℝ × ℝ := (λ P : ℝ × ℝ, P)

-- Proof for question (2) - Area of triangle PF1F2
theorem triangle_area (P : ℝ × ℝ) (angle_P : ℝ) (h1 : point_on_ellipse P) (h2 : angle_P = 120) : 
  let (x, y) := P in ∃ area : ℝ, area = 3 * real.sqrt 3 :=
sorry

end ellipse_eq_triangle_area_l231_231118


namespace gcd_of_sum_of_four_consecutive_integers_l231_231874

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231874


namespace dice_probability_sum_17_l231_231083

-- Problem: Prove the probability that the sum of the face-up integers is 17 when three standard 6-faced dice are rolled is 1/24.

def probability_sum_17 (dice_rolls : ℕ → ℕ) (n : ℕ) : ℝ :=
  let probability_6 := 1 / 6  in
  let probability_case_A := (6 * (probability_6^3))  -- Case where one die shows 6 and other two sum to 11
  let probability_case_B := (3 * (probability_6^3))  -- Case where two dice show 6 and third shows 5
  probability_case_A + probability_case_B

theorem dice_probability_sum_17 : probability_sum_17 = 1 / 24 :=
by
  sorry

end dice_probability_sum_17_l231_231083


namespace find_X_l231_231195

-- Define the initial terms and conditions
def row_first_term : ℤ := 25
def row_intersection_term : ℤ := 11
def column_bottom_term : ℤ := 11

-- Define the common difference for the row's arithmetic sequence
def row_common_difference : ℚ := (row_intersection_term - row_first_term) / 3
-- Verify this calculation aligns with the problem statement
#eval row_common_difference -- Should output -(14/3)

def X : ℤ := row_first_term + 6 * row_common_difference

-- The proof objective is to show that X = -3
theorem find_X : X = -3 := by
  -- Calculations and validations would be done here
  sorry

end find_X_l231_231195


namespace pascal_triangle_row_fifth_number_l231_231721

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231721


namespace ratio_of_areas_l231_231103

noncomputable section

open Real

variables {A B C D E F : Point}
variables (AB AC AD CF : ℝ)
variables (AB_pos : AB = 115)
   (AC_pos : AC = 115)
   (AD_pos : AD = 38)
   (CF_pos : CF = 77)

theorem ratio_of_areas : 
  ∃ (r : ℝ), r = (19 / 96) ∧ 
  let BD := AB - AD in
  let CE := 192 in
  let BE := 38 in
  let EF := 115 in
  let DE := 115 in
  r = (EF / DE) * (CE / BE) * (sin(CEF_angle) / sin(BED_angle)) :=
by
  let BD := AB - AD
  let CE := 192
  let BE := 38
  let EF := 115
  let DE := 115
  sorry

end ratio_of_areas_l231_231103


namespace find_a_l231_231072

variables (x y : ℝ) (a : ℝ)

-- Condition 1: Original profit equation
def original_profit := y - x = x * (a / 100)

-- Condition 2: New profit equation with 5% cost decrease
def new_profit := y - 0.95 * x = 0.95 * x * ((a + 15) / 100)

theorem find_a (h1 : original_profit x y a) (h2 : new_profit x y a) : a = 185 :=
sorry

end find_a_l231_231072


namespace dodecahedron_interior_diagonals_l231_231044

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231044


namespace product_of_five_consecutive_integers_not_square_l231_231172

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (h : 0 < a) :
  ¬ is_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) := by
sorry

end product_of_five_consecutive_integers_not_square_l231_231172


namespace proof_math_problem_lean_l231_231543

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l231_231543


namespace gcd_B_eq_two_l231_231858

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231858


namespace arc_length_of_circle_l231_231561

/-- 
In a circle with a radius of 2, the arc length corresponding to a central angle 
of 60 degrees is 2π/3.
-/
theorem arc_length_of_circle (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 60 * real.pi / 180) :
  r * θ = 2 * real.pi / 3 :=
by
  sorry

end arc_length_of_circle_l231_231561


namespace apartments_with_one_resident_l231_231564

theorem apartments_with_one_resident :
  let T := 1000
  let P_occupied := 0.92
  let P_ge_2 := 0.65
  let occupied_apartments := P_occupied * T
  let apartments_with_ge_2_residents := P_ge_2 * occupied_apartments
  let apartments_with_one_resident := occupied_apartments - apartments_with_ge_2_residents
  apartments_with_one_resident = 322 :=
by
  -- Definitions
  let T := 1000
  let P_occupied := 0.92
  let P_ge_2 := 0.65
  let occupied_apartments := P_occupied * T
  let apartments_with_ge_2_residents := P_ge_2 * occupied_apartments
  let apartments_with_one_resident := occupied_apartments - apartments_with_ge_2_residents
  -- The statement to be proven
  exact rfl

end apartments_with_one_resident_l231_231564


namespace monotonic_intervals_of_h_range_of_a_l231_231482

def f (x : ℝ) : ℝ := x * Real.log x - x + 1
def g (x : ℝ) : ℝ := x^2 - 2 * Real.log x - 1
def h (x : ℝ) : ℝ := 4 * f x - g x

theorem monotonic_intervals_of_h :
  (∀ x, 0 < x → x < 1 → ∀ y, y = h x → h x < h y) ∧
  (∀ x, 1 < x → ∀ y, y = h x → h x > h y) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → a * f x ≤ g x) ↔ (a ≤ 4) :=
sorry

end monotonic_intervals_of_h_range_of_a_l231_231482


namespace max_unique_frequencies_21_persons_l231_231562

theorem max_unique_frequencies_21_persons :
  ∃ m n : ℕ, m + n = 21 ∧ m * n = 110 :=
begin
  use [11, 10],
  split,
  { exact rfl },
  { exact rfl },
end

end max_unique_frequencies_21_persons_l231_231562


namespace pascal_triangle_fifth_number_l231_231640

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l231_231640


namespace circle_radius_l231_231332

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l231_231332


namespace petya_win_condition_l231_231155

def petya_can_always_win (n : ℕ) : Prop :=
  ∀ initial_position, (∃ s, ∃ direction, (move token by s in direction results in 0)) 

theorem petya_win_condition (n : ℕ) (h : n > 1) :
  petya_can_always_win n ↔ (∃ k : ℕ, n = 2^k) :=
by sorry

end petya_win_condition_l231_231155


namespace triangle_percentage_is_correct_l231_231345

def side_length := real
def square_area (s : side_length) := s ^ 2
def triangle_area (s : side_length) := (real.sqrt 3 / 4) * s ^ 2
def pentagon_area (s : side_length) := square_area s + triangle_area s
def triangle_percentage (s : side_length) := 
    (triangle_area s) / (pentagon_area s) * 100

theorem triangle_percentage_is_correct (s : side_length) :
    triangle_percentage s = (100 * real.sqrt 3 - 75) / 13 :=
    sorry

end triangle_percentage_is_correct_l231_231345


namespace first_worker_time_budget_l231_231220

theorem first_worker_time_budget
  (total_time : ℝ := 1)
  (second_worker_time : ℝ := 1 / 3)
  (third_worker_time : ℝ := 1 / 3)
  (x : ℝ) :
  x + second_worker_time + third_worker_time = total_time → x = 1 / 3 :=
by
  sorry

end first_worker_time_budget_l231_231220


namespace gcd_B_eq_two_l231_231861

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231861


namespace gcd_B_eq_two_l231_231865

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231865


namespace gcd_B_eq_two_l231_231859

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l231_231859


namespace option_D_is_negative_l231_231364

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end option_D_is_negative_l231_231364


namespace greatest_two_digit_with_product_12_l231_231259

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231259


namespace greatest_two_digit_product_12_l231_231277

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l231_231277


namespace dodecahedron_interior_diagonals_l231_231042

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end dodecahedron_interior_diagonals_l231_231042


namespace ant_second_turn_time_l231_231430

noncomputable def time_for_second_turn (initial_time: ℝ) (number_of_rays: ℕ) : ℝ :=
initial_time * (Math.sqrt(3) / 2) ^ number_of_rays

theorem ant_second_turn_time (O : Point) (l : Fin 12 → Ray O) (initial_time : ℝ) :
  time_for_second_turn initial_time 12 = initial_time * (729 / 4096) :=
by
  sorry

end ant_second_turn_time_l231_231430


namespace Pascal_triangle_fifth_number_l231_231666

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l231_231666


namespace gcd_of_B_is_2_l231_231836

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231836


namespace pascal_triangle_fifth_number_l231_231752

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231752


namespace equilateral_hyperbola_eccentricity_l231_231190

theorem equilateral_hyperbola_eccentricity (a b c : ℝ) (h_equilateral : a = b) (h_rel : c^2 = a^2 + b^2) : 
  e = real.sqrt 2 :=
by
  have h1 : c = a * real.sqrt 2 := by sorry
  have h2 : e = c / a := by sorry
  rw [h1, h_equilateral]
  -- additional steps skipped
  sorry

end equilateral_hyperbola_eccentricity_l231_231190


namespace pascal_triangle_fifth_number_l231_231668

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231668


namespace greatest_two_digit_product_is_12_l231_231247

theorem greatest_two_digit_product_is_12 : 
  ∃ (n : ℕ), (∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12 ∧ 10 ≤ n ∧ n < 100) ∧ 
              ∀ (m : ℕ), (∃ (e1 e2 : ℕ), m = 10 * e1 + e2 ∧ e1 * e2 = 12 ∧ 10 ≤ m ∧ m < 100) → m ≤ n :=
sorry

end greatest_two_digit_product_is_12_l231_231247


namespace required_CO2_l231_231056

noncomputable def moles_of_CO2_required (Mg CO2 MgO C : ℕ) (hMgO : MgO = 2) (hC : C = 1) : ℕ :=
  if Mg = 2 then 1 else 0

theorem required_CO2
  (Mg CO2 MgO C : ℕ)
  (hMgO : MgO = 2)
  (hC : C = 1)
  (hMg : Mg = 2)
  : moles_of_CO2_required Mg CO2 MgO C hMgO hC = 1 :=
  by simp [moles_of_CO2_required, hMg]

end required_CO2_l231_231056


namespace greatest_two_digit_with_product_12_l231_231256

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l231_231256


namespace pascal_triangle_fifth_number_l231_231679

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l231_231679


namespace fifth_number_in_pascal_row_l231_231802

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l231_231802


namespace gcd_of_B_is_two_l231_231889

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l231_231889


namespace ella_max_book_price_l231_231401

/--
Given that Ella needs to buy 20 identical books and her total budget, 
after deducting the $5 entry fee, is $195. Each book has the same 
cost in whole dollars, and an 8% sales tax is applied to the price of each book. 
Prove that the highest possible price per book that Ella can afford is $9.
-/
theorem ella_max_book_price : 
  ∀ (n : ℕ) (B T : ℝ), n = 20 → B = 195 → T = 1.08 → 
  ∃ (p : ℕ), (↑p ≤ B / T / n) → (9 ≤ p) := 
by 
  sorry

end ella_max_book_price_l231_231401


namespace pascal_triangle_fifth_number_l231_231691

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231691


namespace circle_segment_proof_l231_231830

/--
Let A, B, C, and D be points on a circle such that AB = 13 and CD = 21.
Point P is on segment AB with AP = 7, and Q is on segment CD with CQ = 8.
The line through P and Q intersects the circle at points X and Y.
Given PQ = 30, prove that the length of segment XY is approximately 34.77.
-/
theorem circle_segment_proof (A B C D P Q X Y : ℝ) 
  (h1 : A ∈ circle) 
  (h2 : B ∈ circle) 
  (h3 : C ∈ circle) 
  (h4 : D ∈ circle) 
  (h5 : A < B)
  (h6 : C < D)
  (h7 : P ∈ Icc A B) 
  (h8 : Q ∈ Icc C D) 
  (h9 : dist A B = 13)
  (h10 : dist C D = 21)
  (h11 : dist A P = 7) 
  (h12 : dist C Q = 8)
  (h13 : dist P Q = 30)
: dist X Y = 34.77 := 
sorry

end circle_segment_proof_l231_231830


namespace pascal_fifth_element_15th_row_l231_231708

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l231_231708


namespace greatest_number_divides_l231_231239

noncomputable def gcd_problem : ℕ :=
let a := 263 - 7
let b := 935 - 7
let c := 1383 - 7
in Nat.gcd (Nat.gcd a b) c

theorem greatest_number_divides : gcd_problem = 32 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end greatest_number_divides_l231_231239


namespace pascal_triangle_row_fifth_number_l231_231723

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l231_231723


namespace muffin_cost_ratio_l231_231120

-- Define the costs of items
variables (m b : ℝ)

-- Conditions given in the problem
axiom Kristy_cost : 5 * m + 4 * b
axiom Tim_cost_multiple : 3 * (5 * m + 4 * b)
axiom Tim_direct_cost : 3 * m + 20 * b

-- Theorem to prove that the cost of a muffin is 2/3 the cost of a banana
theorem muffin_cost_ratio (h1 : 3 * (5 * m + 4 * b) = 3 * m + 20 * b) : m = (2 * b) / 3 :=
by {
  sorry
}


end muffin_cost_ratio_l231_231120


namespace gcd_of_sum_of_four_consecutive_integers_l231_231872

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l231_231872


namespace gcd_B_is_2_l231_231911

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l231_231911


namespace probability_of_sum_17_l231_231085

noncomputable def prob_sum_dice_is_seventeen : ℚ :=
1 / 72

theorem probability_of_sum_17 :
  let dice := finset.product (finset.product finset.univ finset.univ) finset.univ in
  let event := dice.filter (λ (x : ℕ × (ℕ × ℕ)), x.1 + x.2.1 + x.2.2 = 17) in
  (event.card : ℚ) / (dice.card : ℚ) = prob_sum_dice_is_seventeen :=
by
  sorry

end probability_of_sum_17_l231_231085


namespace five_a_plus_five_b_eq_neg_twenty_five_thirds_l231_231389

variable (g f : ℝ → ℝ)
variable (a b : ℝ)
axiom g_def : ∀ x, g x = 3 * x + 5
axiom g_inv_rel : ∀ x, g x = (f⁻¹ x) - 1
axiom f_def : ∀ x, f x = a * x + b
axiom f_inv_def : ∀ x, f⁻¹ (f x) = x

theorem five_a_plus_five_b_eq_neg_twenty_five_thirds :
    5 * a + 5 * b = -25 / 3 :=
sorry

end five_a_plus_five_b_eq_neg_twenty_five_thirds_l231_231389


namespace no_real_roots_range_l231_231554

theorem no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_range_l231_231554


namespace pascal_triangle_fifth_number_l231_231692

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231692


namespace gcd_elements_of_B_l231_231958

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l231_231958


namespace find_other_leg_of_right_triangle_l231_231199

noncomputable def hypotenuse (a : ℕ) (b : ℕ) : ℚ :=
  real.sqrt (a^2 + b^2)

theorem find_other_leg_of_right_triangle (a : ℕ) (h : ℚ) (b : ℚ) :
  a = 5 → h = 4 → (2 * a * b) = (2 * hypotenuse a b * h) → b = 20 / 3 :=
by
  intro ha h4 area_eq
  sorry

end find_other_leg_of_right_triangle_l231_231199


namespace pascal_15_5th_number_l231_231772

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l231_231772


namespace complex_sum_eq_500_l231_231133

theorem complex_sum_eq_500 (x : ℂ) (hx1 : x ^ 1001 = 1) (hx2 : x ≠ 1) :
  (∑ k in Finset.range 1 1000, x^((4 * k + 2) : my) / (x^((2 * k + 1) : my) - 1)) = 500 :=
begin
  sorry
end

end complex_sum_eq_500_l231_231133


namespace count_integers_satisfy_inequality_l231_231504

theorem count_integers_satisfy_inequality :
  {n : ℤ | (n + 5) * (n - 9) ≤ 0}.count = 15 := 
sorry

end count_integers_satisfy_inequality_l231_231504


namespace green_peaches_per_basket_l231_231217

-- Definition of the conditions
def has_red_peaches (n : ℕ) : Prop := n = 4
def total_peaches (n : ℕ) : Prop := n = 7
def number_of_baskets (n : ℕ) : Prop := n = 1

-- Proof problem statement
theorem green_peaches_per_basket (r g t b : ℕ) 
  (hr : has_red_peaches r) 
  (ht : total_peaches t) 
  (hb : number_of_baskets b) : 
  g = t - r := by
  -- conditions imply the following
  rw [hr, ht, hb]
  sorry

end green_peaches_per_basket_l231_231217


namespace pascal_triangle_fifth_number_l231_231693

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l231_231693


namespace infinitely_many_terms_in_interval_l231_231130

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), 1 / (i + 1 : ℝ)

theorem infinitely_many_terms_in_interval (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) (h3 : b ≤ 1) :
  ∃ᶠ n in at_top, (S n - ⌊S n⌋) ∈ set.Ioo a b := 
sorry

end infinitely_many_terms_in_interval_l231_231130


namespace compute_alpha_l231_231977

theorem compute_alpha (α β : ℂ) (h1 : α + β ∈ ℝ+) (h2 : i * (2 * α - β) ∈ ℝ+) (hβ : β = 4 + 3 * i) : α = 2 - 3 * i :=
sorry

end compute_alpha_l231_231977


namespace greatest_common_divisor_of_B_l231_231898

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231898


namespace dodecahedron_interior_diagonals_l231_231008

def is_dodecahedron (G : Type) := 
  ∃ (vertices : set G) (faces : set (set G)), 
    (vertices.card = 20) ∧ 
    (∀ f ∈ faces, f.card = 5) ∧
    (∃ faces_inter, (∀ v ∈ vertices, faces_inter v = {f ∈ faces | v ∈ f}.card = 3))

def num_interior_diagonals (G : Type) [is_dodecahedron G] : ℕ :=
  170

theorem dodecahedron_interior_diagonals (G : Type) [is_dodecahedron G] :
  num_interior_diagonals G = 170 :=
sorry

end dodecahedron_interior_diagonals_l231_231008


namespace find_c_l231_231551

theorem find_c (b c : ℝ) (h : 4 * (λ x, (x + 2) * (x + b)) = (λ x, x^2 + c * x + 12)) : c = 14 :=
sorry

end find_c_l231_231551


namespace pascal_fifth_number_in_row_15_l231_231626

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l231_231626


namespace gcd_of_B_is_2_l231_231834

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l231_231834


namespace dice_probability_sum_17_l231_231084

-- Problem: Prove the probability that the sum of the face-up integers is 17 when three standard 6-faced dice are rolled is 1/24.

def probability_sum_17 (dice_rolls : ℕ → ℕ) (n : ℕ) : ℝ :=
  let probability_6 := 1 / 6  in
  let probability_case_A := (6 * (probability_6^3))  -- Case where one die shows 6 and other two sum to 11
  let probability_case_B := (3 * (probability_6^3))  -- Case where two dice show 6 and third shows 5
  probability_case_A + probability_case_B

theorem dice_probability_sum_17 : probability_sum_17 = 1 / 24 :=
by
  sorry

end dice_probability_sum_17_l231_231084


namespace total_blue_marbles_l231_231145

noncomputable def total_blue_marbles_collected_by_friends : ℕ := 
  let jenny_red := 30 in
  let jenny_blue := 25 in
  let mary_red := 2 * jenny_red in
  let anie_red := mary_red + 20 in
  let anie_blue := 2 * jenny_blue in
  let mary_blue := anie_blue / 2 in
  jenny_blue + mary_blue + anie_blue

theorem total_blue_marbles (jenny_red jenny_blue mary_red anie_red mary_blue anie_blue : ℕ) :
  jenny_red = 30 → 
  jenny_blue = 25 → 
  mary_red = 2 * jenny_red → 
  anie_red = mary_red + 20 → 
  anie_blue = 2 * jenny_blue → 
  mary_blue = anie_blue / 2 → 
  jenny_blue + mary_blue + anie_blue = 100 :=
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4, h5, h6],
  norm_num,
end

end total_blue_marbles_l231_231145


namespace fifth_number_in_pascals_triangle_l231_231588

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l231_231588


namespace normal_distribution_expectation_l231_231075

noncomputable def ξ : ℝ := sorry

axiom ξ_distrib : ξ ≠ 0 ∧ ξ = 2

theorem normal_distribution_expectation : E ξ = (2 : ℝ) := 
  by
  sorry

end normal_distribution_expectation_l231_231075


namespace greatest_common_divisor_of_B_l231_231895

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l231_231895


namespace greatest_two_digit_with_product_12_l231_231267

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l231_231267


namespace pascal_triangle_fifth_number_l231_231763

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l231_231763


namespace pascal_triangle_15_4_l231_231619

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l231_231619
