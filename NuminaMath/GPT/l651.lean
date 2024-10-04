import Mathlib

namespace dark_squares_exceed_light_squares_by_one_l651_651030

theorem dark_squares_exceed_light_squares_by_one 
  (m n : ℕ) (h_m : m = 9) (h_n : n = 9) (h_total_squares : m * n = 81) :
  let dark_squares := 5 * 5 + 4 * 4
  let light_squares := 5 * 4 + 4 * 5
  dark_squares - light_squares = 1 :=
by {
  sorry
}

end dark_squares_exceed_light_squares_by_one_l651_651030


namespace positive_number_among_given_options_l651_651856

theorem positive_number_among_given_options :
  (0 <= 0) ∧ (5 > 0) ∧ (-1 / 2 < 0) ∧ (-Real.sqrt 2 < 0) → (∀ x, x = 0 ∨ x = (-1/2) ∨ x = (-Real.sqrt 2) → (x <= 0)) :=
by
  intros h
  obtain ⟨hz, hp, hn1, hn2⟩ := h
  intros x hx
  cases hx with hx₀ hx₁
  {
    rw [hx₀],
    exact hz
  }
  {
    cases hx₁ with hx₁ hx₂
    {
      rw [hx₁],
      exact hn1
    }
    {
      rw [hx₂],
      exact hn2
    }
  }

end positive_number_among_given_options_l651_651856


namespace problem1_problem2_l651_651559

-- Definition for the first problem
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- First Lean 4 statement for 2^n + 3 = x^2
theorem problem1 (n : ℕ) (h : isPerfectSquare (2^n + 3)) : n = 0 :=
sorry

-- Second Lean 4 statement for 2^n + 1 = x^2
theorem problem2 (n : ℕ) (h : isPerfectSquare (2^n + 1)) : n = 3 :=
sorry

end problem1_problem2_l651_651559


namespace same_remainder_division_l651_651776

theorem same_remainder_division {a m b : ℤ} (r c k : ℤ) 
  (ha : a = b * c + r) (hm : m = b * k + r) : b ∣ (a - m) :=
by
  sorry

end same_remainder_division_l651_651776


namespace circle_center_sum_radius_eq_neg_nine_l651_651274

noncomputable def circle_center_sum_radius (x y : ℝ) : ℝ :=
  let a := -5
  let b := -6
  let r := 2
  a + b + r

theorem circle_center_sum_radius_eq_neg_nine (x y : ℝ) :
  (x^2 + 12 * y + 57 = -y^2 - 10 * x) → circle_center_sum_radius x y = -9 :=
by
  intro h
  unfold circle_center_sum_radius
  rw [add_assoc, add_comm (-6)]
  exact rfl

end circle_center_sum_radius_eq_neg_nine_l651_651274


namespace smallest_factorial_5_4_l651_651978

theorem smallest_factorial_5_4 (n : ℕ) (h₁ : n = 4) :
  ∃ m : ℕ, (∑ i in (range (nat.log 5 (m!)+1)), m / 5 ^ i) = n ∧
           (∑ i in (range (nat.log 5 ((m+1)!)+1)), (m+1) / 5 ^ i) > n := 
begin
  -- Proof would go here.
  sorry
end

end smallest_factorial_5_4_l651_651978


namespace elderly_in_sample_l651_651029

variable (A E M : ℕ)
variable (total_employees : ℕ)
variable (total_young : ℕ)
variable (sample_size_young : ℕ)
variable (sampling_ratio : ℚ)
variable (sample_elderly : ℕ)

axiom condition_1 : total_young = 160
axiom condition_2 : total_employees = 430
axiom condition_3 : M = 2 * E
axiom condition_4 : A + M + E = total_employees
axiom condition_5 : sampling_ratio = sample_size_young / total_young
axiom sampling : sample_size_young = 32
axiom elderly_employees : sample_elderly = 18

theorem elderly_in_sample : sample_elderly = sampling_ratio * E := by
  -- Proof steps are not provided
  sorry

end elderly_in_sample_l651_651029


namespace problem1_problem2_l651_651604

variable (α : ℝ) (tan_alpha_eq_one_over_three : Real.tan α = 1 / 3)

-- For the first proof problem
theorem problem1 : (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by sorry

-- For the second proof problem
theorem problem2 : Real.cos α ^ 2 - Real.sin (2 * α) = 3 / 10 :=
by sorry

end problem1_problem2_l651_651604


namespace distance_to_nearest_edge_l651_651044

-- Definitions of conditions
def wall_width : ℝ := 27
def picture_width : ℝ := 5

-- Theorem statement
theorem distance_to_nearest_edge : 
  let y := (wall_width - picture_width) / 2 in y = 11 := 
  by
  -- The proof would go here, but is omitted as per instructions
  sorry

end distance_to_nearest_edge_l651_651044


namespace projections_relationship_l651_651921

theorem projections_relationship (a b r : ℝ) (h : r ≠ 0) :
  (∃ α β : ℝ, a = r * Real.cos α ∧ b = r * Real.cos β ∧ (Real.cos α)^2 + (Real.cos β)^2 = 1) → (a^2 + b^2 = r^2) :=
by
  sorry

end projections_relationship_l651_651921


namespace seunghyeon_pizza_diff_l651_651308

theorem seunghyeon_pizza_diff (S Y : ℕ) (h : S - 2 = Y + 7) : S - Y = 9 :=
by {
  sorry
}

end seunghyeon_pizza_diff_l651_651308


namespace circle_distance_and_radius_l651_651851

-- Define the given circle equation
def circle (x y : ℝ) := x^2 + y^2 - 6*x + 8*y - 18

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The center of the circle given by completing the square
def center : ℝ × ℝ := (3, -4)

-- The known point
def point : ℝ × ℝ := (-3, 4)

-- The radius obtained by completing the square
def radius : ℝ := real.sqrt 43

theorem circle_distance_and_radius:
  distance center point = 10 ∧ radius = real.sqrt 43 :=
by
  sorry

end circle_distance_and_radius_l651_651851


namespace subset_solution_l651_651718

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l651_651718


namespace all_faces_rhombuses_l651_651654

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l651_651654


namespace find_a_l651_651233

-- Lean definitions for the conditions
def C1 (x y a : ℝ) : Prop := x^2 + (y - 1)^2 = a^2
def C2 (ρ θ : ℝ) : Prop := ρ = 4 * real.cos θ
def C3 (θ α₀ : ℝ) : Prop := θ = α₀ ∧ real.tan α₀ = 2

theorem find_a (x y ρ θ a α₀ : ℝ) (hC1 : C1 x y a) (hC2 : C2 ρ θ) (hC3 : C3 θ α₀) :
  a = 1 :=
by
  sorry

end find_a_l651_651233


namespace exists_sequence_unbounded_positive_integers_l651_651303

open Nat

theorem exists_sequence_unbounded_positive_integers (a : ℕ → ℕ) :
  (∀ n, a n ≤ a (n + 1)) ∧ (∀ n, ∃ m, a m > n) ∧
  (∃ M, ∀ n ≥ M, (¬ prime (n + 1)) → 
    (∀ p, prime p → p ∣ (factorial n + 1) → p > n + a n)) :=
sorry

end exists_sequence_unbounded_positive_integers_l651_651303


namespace boys_camp_problem_l651_651221

noncomputable def total_boys_in_camp : ℝ :=
  let schoolA_fraction := 0.20
  let science_fraction := 0.30
  let non_science_boys := 63
  let non_science_fraction := 1 - science_fraction
  let schoolA_boys := (non_science_boys / non_science_fraction)
  schoolA_boys / schoolA_fraction

theorem boys_camp_problem : total_boys_in_camp = 450 := 
by
  sorry

end boys_camp_problem_l651_651221


namespace sqrt_of_sixteen_l651_651814

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l651_651814


namespace ages_total_l651_651905

-- Define the variables and conditions
variables (A B C : ℕ)

-- State the conditions
def condition1 (B : ℕ) : Prop := B = 14
def condition2 (A B : ℕ) : Prop := A = B + 2
def condition3 (B C : ℕ) : Prop := B = 2 * C

-- The main theorem to prove
theorem ages_total (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 B C) : A + B + C = 37 :=
by
  sorry

end ages_total_l651_651905


namespace find_line_eq_l651_651166

-- Given conditions
variable (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]

-- Points and parameters
variable (point_A : A) (point_B : B) (point_C : (ℝ × ℝ))

-- Coordinates of points
variable (xa ya xb yb xc yc : ℝ)
variable (ha : point_A = (xa, ya))
variable (hb : point_B = (xb, yb))
variable (hc : point_C = (xc, yc))

noncomputable def line_equation (k : ℝ) : ℝ → ℝ → Prop := 
  λ x y, k*x - y - k + 2 = 0

theorem find_line_eq : 
  ∀ (l : ℝ → ℝ → Prop), 
  (∀ x y, l x y ↔ line_equation (1/2) x y ∨ line_equation (-1/6) x y) →
  (∀ p, abs ((xa - xc) * (snd p - yc) - (ya - yc) * (fst p - xc)) / sqrt ((xa - xc)^2 + (ya - yc)^2) = 
        abs ((xb - xc) * (snd p - yc) - (yb - yc) * (fst p - xc)) / sqrt ((xb - xc)^2 + (yb - yc)^2)) → 
  ( ∃ k : ℝ, l = line_equation k ∨ l = line_equation (-1/6) ) :=
by sorry

end find_line_eq_l651_651166


namespace alyosha_cube_problem_l651_651465

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651465


namespace area_of_shape_l651_651137

theorem area_of_shape (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  (x - 2 * Real.cos α)^2 + (y - 2 * Real.sin α)^2 = 16 →
  ∃ A : ℝ, A = 32 * Real.pi :=
by
  sorry

end area_of_shape_l651_651137


namespace solve_cubic_eq_l651_651312

theorem solve_cubic_eq (x : ℂ) :
  (x^3 + 4 * x^2 * real.sqrt 3 + 12 * x + 8 * real.sqrt 3) + (x + 2 * real.sqrt 3) = 0 ↔
  (x = -2 * real.sqrt 3 ∨ x = -2 * real.sqrt 3 + complex.I ∨ x = -2 * real.sqrt 3 - complex.I) :=
by
  sorry

end solve_cubic_eq_l651_651312


namespace nth_row_equation_l651_651295

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1) ^ 2 - n ^ 2 := 
sorry

end nth_row_equation_l651_651295


namespace alyosha_cube_problem_l651_651471

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651471


namespace min_value_inverse_sum_l651_651330

theorem min_value_inverse_sum (a m n : ℝ) (a_pos : 0 < a) (a_ne_one : a ≠ 1) (mn_pos : 0 < m * n) :
  (a^(1-1) = 1) ∧ (m + n = 1) → (1/m + 1/n) = 4 :=
by
  sorry

end min_value_inverse_sum_l651_651330


namespace range_of_alpha_in_first_quadrant_l651_651984

theorem range_of_alpha_in_first_quadrant (α : ℝ) (k : ℤ) (h : cos α ≤ sin α) : 
  ∃ (k : ℤ), 2 * k * π + π / 4 ≤ α ∧ α < 2 * k * π + π / 2 := 
  sorry

end range_of_alpha_in_first_quadrant_l651_651984


namespace cost_price_article_l651_651864

-- Define the variables and conditions
variables (cost_price selling_price1 selling_price2 : ℝ)

-- Conditions given in the problem
def condition1 : Prop := selling_price1 = 0.90 * cost_price
def condition2 : Prop := selling_price2 = 1.10 * cost_price
def condition3 : Prop := selling_price2 = selling_price1 + 50

-- The theorem statement to prove
theorem cost_price_article : 
  condition1 ∧ condition2 ∧ condition3 → cost_price = 250 :=
by 
  sorry

end cost_price_article_l651_651864


namespace square_root_of_16_is_pm_4_l651_651821

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l651_651821


namespace president_and_committee_count_l651_651229

theorem president_and_committee_count (total_people : ℕ)
    (people : Finset ℕ) (h_size : people.card = total_people) :
    total_people = 10 → 
    (∃ president ∈ people, ∀ committee ⊂ people \ {president}, committee.card = 3 → 
    ↑(Finset.choose_cardinality (people.erase president) 3) = 84 → 
    (total_people * (9.choose 3) = 840)) :=
begin
  sorry,
end

end president_and_committee_count_l651_651229


namespace remainder_17_pow_63_mod_7_l651_651384

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651384


namespace range_of_a_l651_651178

variable (a : ℝ)

def a_n (n : ℕ) : ℝ :=
if n = 1 then a else 4 * ↑n + (-1 : ℝ) ^ n * (8 - 2 * a)

theorem range_of_a (h : ∀ n : ℕ, n > 0 → a_n a n < a_n a (n + 1)) : 3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l651_651178


namespace shoes_difference_l651_651786

theorem shoes_difference : 
  ∀ (Scott_shoes Anthony_shoes Jim_shoes : ℕ), 
  Scott_shoes = 7 → 
  Anthony_shoes = 3 * Scott_shoes → 
  Jim_shoes = Anthony_shoes - 2 → 
  Anthony_shoes - Jim_shoes = 2 :=
by
  intros Scott_shoes Anthony_shoes Jim_shoes 
  intros h1 h2 h3 
  sorry

end shoes_difference_l651_651786


namespace integer_values_count_l651_651315

noncomputable def perimeter_abe := 10 * Real.pi

def valid_n (n : ℝ) : Prop :=
  (5 * Real.pi < n) ∧ (n < 20 * Real.pi)

def integer_values_for_n (lower upper : ℝ) : ℕ :=
  Nat.floor upper - Nat.ceil lower + 1

theorem integer_values_count :
  integer_values_for_n (5 * Real.pi) (20 * Real.pi) = 47 :=
by
  sorry

end integer_values_count_l651_651315


namespace remainder_17_pow_63_mod_7_l651_651378

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651378


namespace seven_digit_palindromes_count_l651_651952

theorem seven_digit_palindromes_count : 
  let digits := [0, 1, 1, 8, 8, 8, 8] in 
  let palindrome_num := 
    (digits.erase 0).permutations.filter (λ d, d.take 3 = d.reverse.take 3) |>.length in 
  palindrome_num = 3 := sorry

end seven_digit_palindromes_count_l651_651952


namespace circle_center_l651_651322

theorem circle_center : 
  ∃ (h k : ℝ), (h, k) = (1, -2) ∧ 
    ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - h)^2 + (y - k)^2 = 9 :=
by
  sorry

end circle_center_l651_651322


namespace max_number_of_soap_boxes_l651_651903

noncomputable def carton_volume : ℕ := 25 * 42 * 60
noncomputable def soap_box_volume : ℕ := 7 * 12 * 5
noncomputable def diagonal_soap_box_base : ℝ := Real.sqrt (12 ^ 2 + 5 ^ 2)
def carton_base_diag_fits : bool := diagonal_soap_box_base ≤ Real.sqrt (25 ^ 2 + 42 ^ 2)
def max_soap_boxes := carton_volume / soap_box_volume

theorem max_number_of_soap_boxes (h : carton_base_diag_fits = tt) : max_soap_boxes = 150 := 
sorry

end max_number_of_soap_boxes_l651_651903


namespace total_handshakes_l651_651526

-- Define the conditions as Lean definitions
def team_members : ℕ := 7
def referees : ℕ := 3

-- Define the proof problem
theorem total_handshakes : 
  let inter_team_handshakes := team_members * team_members in
  let total_players := team_members + team_members in
  let referee_handshakes := total_players * referees in
  inter_team_handshakes + referee_handshakes = 91 := 
by
  let inter_team_handshakes := team_members * team_members
  let total_players := team_members + team_members
  let referee_handshakes := total_players * referees
  sorry

end total_handshakes_l651_651526


namespace triangle_proof_l651_651250

noncomputable def triangle_condition (a b c sinA sinB : ℝ) (CosC : ℝ → ℝ) (SinB : ℝ → ℝ) (CosB : ℝ → ℝ) (pi : ℝ) : Prop :=
  (b = 6) ∧ 
  (b * CosC C + c * SinB B = a) ∧ 
  (sin A = sin (B + C)) ∧ 
  (CosB B = CosB B * SinB B)

theorem triangle_proof (a b c sinA sinB : ℝ) (CosC : ℝ → ℝ) (SinB : ℝ → ℝ) (CosB : ℝ → ℝ) (pi : ℝ) :
  triangle_condition a b c sinA sinB CosC SinB CosB pi →
  (a + 2*b) / (sinA + 2*sinB) = 6 * Real.sqrt 2 :=
by 
  sorry

end triangle_proof_l651_651250


namespace staff_discount_l651_651439

theorem staff_discount (d : ℝ) : 
  let first_discount := 0.65
      second_discount := 0.60
      price_after_first_discount := d * (1 - first_discount)
      result_price := price_after_first_discount * (1 - second_discount) 
  in result_price = d * 0.14 :=
by 
  let first_discount := 0.65;
  let second_discount := 0.60;
  let price_after_first_discount := d * (1 - first_discount);
  let result_price := price_after_first_discount * (1 - second_discount);
  show result_price = d * 0.14;
  sorry

end staff_discount_l651_651439


namespace angle_B_is_72_degrees_l651_651691

theorem angle_B_is_72_degrees (A B C D E : Type) 
  [Point A] [Point B] [Point C] [Point D] [Point E] 
  (AB AC BD DC BE EA DE : Real)
  (h1 : AB = AC) 
  (h2 : BD = 2 * DC)
  (h3 : BE = 2 * EA)
  (h4 : DE = EA + DC) : 
  angle A B C = 72 :=
by 
  sorry

end angle_B_is_72_degrees_l651_651691


namespace seq_diff_five_consec_odd_avg_55_l651_651795

theorem seq_diff_five_consec_odd_avg_55 {a b c d e : ℤ} 
    (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) (h5: e % 2 = 1)
    (h6: b = a + 2) (h7: c = a + 4) (h8: d = a + 6) (h9: e = a + 8)
    (avg_5_seq : (a + b + c + d + e) / 5 = 55) : 
    e - a = 8 := 
by
    -- proof part can be skipped with sorry
    sorry

end seq_diff_five_consec_odd_avg_55_l651_651795


namespace mark_total_votes_l651_651763

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l651_651763


namespace digit_one_occurrences_l651_651253

theorem digit_one_occurrences : 
  let count_tens := 10 in
  let count_units := 10 in
  let count_hundreds := 100 in
  count_tens + count_units + count_hundreds = 120 :=
by
  let count_tens := 10 
  let count_units := 10 
  let count_hundreds := 100 
  show count_tens + count_units + count_hundreds = 120
  sorry

end digit_one_occurrences_l651_651253


namespace sec_of_7pi_over_4_l651_651966

-- Define given angle in radians
def angle := (7 * Real.pi) / 4

-- Define the required secant function using the given relationship to cosine
def sec (x : ℝ) : ℝ := 1 / Real.cos x

-- State the theorem to be proved in Lean
theorem sec_of_7pi_over_4 : sec angle = Real.sqrt 2 := 
sorry

end sec_of_7pi_over_4_l651_651966


namespace period_and_interval_area_triangle_ABC_l651_651641

-- Definitions for vectors
def m (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 1 / 2)
def f (x : ℝ) : ℝ := let v := (m x).1 + (n x).1
                      let u := (m x).2 + (n x).2
                      v * (m x).1 + u * (m x).2

-- Definitions for the triangle conditions
def a := 2 * Real.sqrt 3
def b := 4
def b' := 2 -- Based on the solution where b = 2

-- Question 1: Smallest positive period T and monotonically increasing interval
theorem period_and_interval : (∀ (x ∈ ℝ), x = x + π) ∧ (∀ k : ℤ, ∃! x : ℝ, x ∈ [k * π - π / 6, k * π + π / 3]) :=
  sorry

-- Question 2: Area of the triangle
theorem area_triangle_ABC (A : ℝ) (hA : 0 < A ∧ A < π / 2) : 
  f A = 3 → 
  let c := b' in
  let b := 2 in
  let area := 1 / 2 * b * c * Real.sin A in
  area = 2 * Real.sqrt 3 :=
  sorry

end period_and_interval_area_triangle_ABC_l651_651641


namespace find_a4_and_a2023_l651_651294

noncomputable theory

-- Given conditions
def initial_terms (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 1 / 2 ∧ a 3 = 2 / 7

def sequence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 / a n + 1 / a (n + 2) = 2 / a (n + 1)

-- Define the sequence
def a (n : ℕ) : ℝ := sorry

-- Prove the questions based on the conditions
theorem find_a4_and_a2023 : 
  (initial_terms a) ∧ (sequence_relation a) → 
  a 4 = 1 / 5 ∧ a 2023 = 2 / 6067 :=
by
  sorry

end find_a4_and_a2023_l651_651294


namespace height_of_pillar_at_E_l651_651520

open Real

-- Definition for the points A, B, C, heights of the pillars
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (6, 0, 0)
def C : ℝ × ℝ × ℝ := (3, 3 * sqrt 3, 0)
def hA : ℝ := 12
def hB : ℝ := 9
def hC : ℝ := 10

def Q := (6, 0, 9)
def R := (3, 3 * sqrt 3, 10)

-- Given points on the solar panel
def P := (0, 0, 12)

-- Calculate the vectors PQ and PR
def PQ := (6, 0, -3)
def PR := (3, 3 * sqrt 3, -2)

-- Calculate the normal vector of the plane
def normal_vector := (9 * sqrt 3, 9, 18 * sqrt 3)

-- Equation of the Plane
def d : ℝ := 216 * sqrt 3

noncomputable def plane_eq (x y z : ℝ) : Prop :=
  (9 * sqrt 3) * x + 9 * y + (18 * sqrt 3) * z = d

-- Coordinates for point E
def E := (0, -6 * sqrt 3, 17)

-- The Problem Statement
theorem height_of_pillar_at_E : plane_eq E.1 E.2 E.3 :=
by sorry

end height_of_pillar_at_E_l651_651520


namespace only_pair_2_2_satisfies_l651_651105

theorem only_pair_2_2_satisfies :
  ∀ a b : ℕ, (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by sorry

end only_pair_2_2_satisfies_l651_651105


namespace find_value_of_g_zero_l651_651280

theorem find_value_of_g_zero (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g(g(x + y)) = g(x) * g(y) + g(x) + g(y) + x * y) :
  g 0 = 0 :=
sorry

end find_value_of_g_zero_l651_651280


namespace circumradii_ratio_half_inradii_inequality_l651_651596

theorem circumradii_ratio_half (ABC : Triangle) (BM : Line) 
  (h1 : ABC.BC = 2 * ABC.AB) 
  (h2 : IsAngleBisector ABC.BM) :
  (circumradius (triangle ABM) / circumradius (triangle CBM) = 1 / 2) :=
sorry

theorem inradii_inequality (ABC : Triangle) (BM : Line) 
  (h1 : ABC.BC = 2 * ABC.AB)
  (h2 : IsAngleBisector ABC.BM) :
  (3 / 4 < inradius (triangle ABM) / inradius (triangle CBM) 
      ∧ inradius (triangle ABM) / inradius (triangle CBM) < 1) :=
sorry

end circumradii_ratio_half_inradii_inequality_l651_651596


namespace min_triangle_value_l651_651065

/-- Define the given numbers -/
def nums : List ℝ := [1.2, 3.7, 6.5, 2.9, 4.6]

/-- Define a function to calculate the average of three numbers -/
def avg3 (a b c : ℝ) : ℝ := (a + b + c) / 3

/-- Given an arrangement of the numbers in five circles and three squares, 
    determine the minimum possible value in the triangle following the conditions -/
theorem min_triangle_value : 
  ∃ (arrangement : List ℝ), arrangement.length = 5 ∧ arrangement.perm nums ∧
  let s1 := avg3 arrangement[0] arrangement[1] arrangement[2],
      s2 := avg3 arrangement[1] arrangement[2] arrangement[3],
      s3 := avg3 arrangement[2] arrangement[3] arrangement[4]
  in (avg3 s1 s2 s3) = 3.1 :=
sorry

end min_triangle_value_l651_651065


namespace subset_a_eq_1_l651_651722

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651722


namespace check_extra_postage_l651_651447

variable (l h : ℕ)

def requires_extra_postage (l h : ℕ) : Prop :=
  (l / h.toRat < 1.5) ∨ (l / h.toRat > 2.8)

noncomputable def count_extra_postage_envelopes : ℕ :=
  let envelopes := [(7, 5), (10, 4), (8, 5), (14, 5)]
  envelopes.count (λ (e : ℕ × ℕ), requires_extra_postage e.fst e.snd)

theorem check_extra_postage :
  count_extra_postage_envelopes = 1 :=
by
  sorry

end check_extra_postage_l651_651447


namespace fare_for_x_gt_5_fare_for_six_l651_651826

def fare (x : ℝ) : ℝ :=
  if x ≤ 3 then 10
  else if x ≤ 5 then 10 + 1.3 * (x - 3)
  else 10 + 2 * 1.3 + 2.4 * (x - 5)

theorem fare_for_x_gt_5 (x : ℝ) (hx : x > 5) : fare x = 2.4 * x + 0.6 := by
  sorry

theorem fare_for_six : fare 6 = 15 := by
  sorry

end fare_for_x_gt_5_fare_for_six_l651_651826


namespace solve_AlyoshaCube_l651_651507

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651507


namespace find_b_l651_651407

-- Definitions based on the given conditions
def good_point (a b : ℝ) (φ : ℝ) : Prop :=
  a + (b - a) * φ = 2.382 ∨ b - (b - a) * φ = 2.382

theorem find_b (b : ℝ) (φ : ℝ := 0.618) :
  good_point 2 b φ → b = 2.618 ∨ b = 3 :=
by
  sorry

end find_b_l651_651407


namespace seventeen_power_sixty_three_mod_seven_l651_651365

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651365


namespace probability_sum_is_five_l651_651884

theorem probability_sum_is_five : 
  let balls := {1, 2, 3, 4}
  let outcomes := ({(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)} : Finset (ℕ × ℕ))
  let favorable := ({(1, 4), (2, 3)} : Finset (ℕ × ℕ))
  (favorable.card / outcomes.card : ℝ) = (1 / 3 : ℝ) :=
by
  sorry

end probability_sum_is_five_l651_651884


namespace floor_trig_sum_l651_651728

theorem floor_trig_sum :
  Int.floor (Real.sin 1) + Int.floor (Real.cos 2) + Int.floor (Real.tan 3) +
  Int.floor (Real.sin 4) + Int.floor (Real.cos 5) + Int.floor (Real.tan 6) = -4 := by
  sorry

end floor_trig_sum_l651_651728


namespace probability_two_girls_given_one_l651_651862

/-- Let A be the event "both children are girls" and B the event "at least one of the children is a girl". 
    Given the conditions, prove that the probability of A given B is 1/3. -/
theorem probability_two_girls_given_one:
  (P : Set (Set (fin 2))) 
  (A : Event P := {s | s = {0, 0}})
  (B : Event P := {s | s ∈ {1, 0} ∨ s ∈ {0, 1} ∨ s ∈ {0, 0}}):
  conditionalProb P A B = 1 / 3 :=
by
  sorry

end probability_two_girls_given_one_l651_651862


namespace irrational_sqrt_3_l651_651409

theorem irrational_sqrt_3 :
  let A := (1 / 2 : ℝ)
  let B := (0.2 : ℝ)
  let C := (-5 : ℝ)
  let D := (real.sqrt 3)
  irrational D ∧ (¬ irrational A) ∧ (¬ irrational B) ∧ (¬ irrational C) :=
by
  sorry

end irrational_sqrt_3_l651_651409


namespace max_real_part_l651_651792

noncomputable def complex_number_problem (z w : ℂ) : Prop :=
  |z| = 1 ∧ |w| = 1 ∧ (z * conj w + conj z * w = 2)

theorem max_real_part (z w : ℂ) (h : complex_number_problem z w) :
  real_part (z + w) ≤ 2 := 
sorry

end max_real_part_l651_651792


namespace solve_problem_l651_651891

noncomputable def proof_problem (x y : ℕ) : Prop :=
  (10 ≤ x) ∧ (x ≤ 99) ∧
  (10 ≤ y) ∧ (y ≤ 99) ∧
  (y = x + 21) ∧
  (100y + x = 100x + y + 2058) →
  (10 ≤ x) ∧ (x ≤ 78)

theorem solve_problem (x y : ℕ) : proof_problem x y :=
  by
  sorry

end solve_problem_l651_651891


namespace cost_of_5_pound_bag_is_2_l651_651900

-- Define costs of each type of bag
def cost_10_pound_bag : ℝ := 20.40
def cost_25_pound_bag : ℝ := 32.25
def least_total_cost : ℝ := 98.75

-- Define the total weight constraint
def min_weight : ℕ := 65
def max_weight : ℕ := 80
def weight_25_pound_bags : ℕ := 75

-- Given condition: The total purchase fulfils the condition of minimum cost
def total_cost_3_bags_25 : ℝ := 3 * cost_25_pound_bag
def remaining_cost : ℝ := least_total_cost - total_cost_3_bags_25

-- Prove the cost of the 5-pound bag is $2.00
theorem cost_of_5_pound_bag_is_2 :
  ∃ (cost_5_pound_bag : ℝ), cost_5_pound_bag = remaining_cost :=
by
  sorry

end cost_of_5_pound_bag_is_2_l651_651900


namespace rob_planned_reading_time_l651_651783

def planned_reading_time (actual_pages : ℕ) (minutes_per_page : ℕ) (actual_fraction : ℚ) : ℚ :=
  let actual_time := actual_pages * minutes_per_page
  (actual_time / actual_fraction) / 60

theorem rob_planned_reading_time (actual_pages : ℕ) (minutes_per_page : ℕ) (actual_fraction : ℚ) : 
  actual_pages = 9 ∧ minutes_per_page = 15 ∧ actual_fraction = 3/4 →
  planned_reading_time 9 15 (3/4) = 3 := 
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h3, h4]
  sorry

end rob_planned_reading_time_l651_651783


namespace initial_candies_l651_651523

-- Define the conditions
def candies_given_older_sister : ℕ := 7
def candies_given_younger_sister : ℕ := 6
def candies_left : ℕ := 15

-- Conclude the initial number of candies
theorem initial_candies : (candies_given_older_sister + candies_given_younger_sister + candies_left) = 28 := by
  sorry

end initial_candies_l651_651523


namespace mark_total_votes_l651_651757

theorem mark_total_votes (h1 : 70% = 0.70) (h2 : 100000 : ℕ) (h3 : twice := 2)
  (votes_first_area : ℕ := 0.70 * 100000)
  (votes_remaining_area : ℕ := twice * votes_first_area)
  (total_votes : ℕ := votes_first_area + votes_remaining_area) : 
  total_votes = 210000 := 
by
  sorry

end mark_total_votes_l651_651757


namespace neither_necessary_nor_sufficient_l651_651016

theorem neither_necessary_nor_sufficient (x : ℝ) :
  ¬ ((-1 < x ∧ x < 2) → (|x - 2| < 1)) ∧ ¬ ((|x - 2| < 1) → (-1 < x ∧ x < 2)) :=
by
  sorry

end neither_necessary_nor_sufficient_l651_651016


namespace natural_numbers_condition_l651_651970

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_numbers_condition (n : ℕ) (p1 p2 : ℕ)
  (hp1_prime : is_prime p1) (hp2_prime : is_prime p2)
  (hn : n = p1 ^ 2) (hn72 : n + 72 = p2 ^ 2) :
  n = 49 ∨ n = 289 :=
  sorry

end natural_numbers_condition_l651_651970


namespace distinct_fractions_count_sum_of_distinct_fractions_l651_651455

-- Define the conditions as a set of pairs
def domino_tiles : Set (ℕ × ℕ) := 
  { (a, b) | a ≤ 6 ∧ b ≤ 6 ∧ a ≤ b } \ { (0, 0) }

-- Define the fractions
def domino_fractions : Set ℚ := 
  { (a, b) | (a, b) ∈ domino_tiles }.image (λ (p : ℕ × ℕ), p.1 / p.2)

-- Part (a): Number of distinct values
theorem distinct_fractions_count : domino_fractions.to_finset.card = 13 :=
  sorry

-- Part (b): Sum of distinct values
theorem sum_of_distinct_fractions : (domino_fractions.to_finset : Set ℚ).Sum nat.cast = 13 / 2 :=
  sorry

end distinct_fractions_count_sum_of_distinct_fractions_l651_651455


namespace cats_at_rescue_center_l651_651643

theorem cats_at_rescue_center : 
  ∀ (num_puppies : ℕ) (weight_per_puppy weight_per_cat weight_difference : ℝ),
  num_puppies = 4 →
  weight_per_puppy = 7.5 →
  weight_per_cat = 2.5 →
  weight_difference = 5 →
  (weight_per_puppy * num_puppies + weight_difference) / weight_per_cat = 14 :=
by
  intros num_puppies weight_per_puppy weight_per_cat weight_difference
  assume h_num_puppies h_weight_per_puppy h_weight_per_cat h_weight_difference
  have h1: weight_per_puppy * num_puppies = 30 := by rw [h_num_puppies, h_weight_per_puppy]; norm_num
  have h2: (weight_per_puppy * num_puppies + weight_difference) = 35 := by rw [h1, h_weight_difference]; norm_num
  have h3: (weight_per_puppy * num_puppies + weight_difference) / weight_per_cat = 14 := by rw [h2, h_weight_per_cat]; norm_num
  exact h3

end cats_at_rescue_center_l651_651643


namespace qiuqiu_servings_l651_651961

-- Define the volume metrics
def bottles : ℕ := 1
def cups_per_bottle_kangkang : ℕ := 4
def foam_expansion : ℕ := 3
def foam_fraction : ℚ := 1 / 2

-- Calculate the effective cup volume under Qiuqiu's serving method
def beer_fraction_per_cup_qiuqiu : ℚ := 1 / 2 + (1 / foam_expansion) * foam_fraction

-- Calculate the number of cups Qiuqiu can serve from one bottle
def qiuqiu_cups_from_bottle : ℚ := cups_per_bottle_kangkang / beer_fraction_per_cup_qiuqiu

-- The theorem statement
theorem qiuqiu_servings :
  qiuqiu_cups_from_bottle = 6 := by
  sorry

end qiuqiu_servings_l651_651961


namespace event_impossible_l651_651423

def drawing_a_pie := "an action of creating a representation on a surface"
def satisfying_hunger := "fulfilling a biological need"
def drawing_a_pie_to_satisfy_hunger : Prop :=
  ∀ (draw : String) (hunger : String), (draw = drawing_a_pie) ∧ (hunger = satisfying_hunger) → False

theorem event_impossible : drawing_a_pie_to_satisfy_hunger := 
by
  intros draw hunger h
  cases h with h1 h2
  sorry

end event_impossible_l651_651423


namespace sandyMoreTokens_l651_651785

noncomputable def numberOfTokensSandyBuys := 3_000_000
noncomputable def numberOfSiblings := 7
noncomputable def keptPercentage := 0.40
noncomputable def additionalTokensBought := 500_000
noncomputable def fractionOfTokensKept := (keptPercentage * numberOfTokensSandyBuys).toNat
noncomputable def totalTokensKept := fractionOfTokensKept + additionalTokensBought
noncomputable def remainingTokens := numberOfTokensSandyBuys - fractionOfTokensKept
noncomputable def tokensPerSibling := (remainingTokens / numberOfSiblings).toNat
noncomputable def tokensMoreThanSibling := totalTokensKept - tokensPerSibling

theorem sandyMoreTokens : tokensMoreThanSibling = 1_442_858 := by
  sorry

end sandyMoreTokens_l651_651785


namespace part_I_part_II_l651_651781

-- Definitions for part (I)
theorem part_I (x : ℝ) (n : ℕ) (h : n ≥ 2) :
  n * ((1 + x)^(n - 1) - 1) = (∑ k in Finset.range (n + 1), if h₁ : k ≥ 2 then (↑k : ℝ) * Nat.choose n k * x^(k-1) else 0) :=
by
  sorry

-- Definitions for part (II)
noncomputable def b_n (n : ℕ) : ℝ := 
  (n * (n^2 + 1) * (2^n - 2^(n-1))) / (n * (3^(n-1)))

theorem part_II :
  is_greatest (set.range b_n) (b_n 5) :=
by
  sorry

end part_I_part_II_l651_651781


namespace general_formula_lambda_range_mn_exist_l651_651616

-- Define the arithmetic sequence and its sum
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d
def sum_seq (a d : ℝ) (n : ℕ) : ℝ := n * a + (n * (n - 1) / 2) * d

-- Define the conditions
axiom S2 : sum_seq 1 2 2 = 4
axiom S5 : sum_seq 1 2 5 = 25

-- 1. Prove the general formula for the arithmetic sequence
theorem general_formula (n : ℕ) : arithmetic_seq 1 2 n = 2 * n - 1 := sorry

-- Define the sequences {a_n} and {b_n}
def a_seq (n : ℕ) : ℝ := 2 * n - 1
def b_seq (n : ℕ) : ℝ := 1 / (a_seq n - a_seq (n + 1))

-- Sum of first n terms of sequence {b_n}
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b_seq (i + 1)

-- 2. Prove the range of λ
def lambda_condition (λ : ℝ) : Prop := ∀ n : ℕ, λ * T n < n + 8 * (-1) ^ n
theorem lambda_range (λ : ℝ) (h : lambda_condition λ) : λ < -21 := sorry

-- 3. Prove the existence of m and n
def geometric_seq (a b c : ℝ) : Prop := b^2 = a * c

theorem mn_exist : geometric_seq (T 1) (T 2) (T 12) := sorry

end general_formula_lambda_range_mn_exist_l651_651616


namespace book_price_distribution_l651_651882

theorem book_price_distribution :
  ∃ (x y z: ℤ), 
  x + y + z = 109 ∧
  (34 * x + 27.5 * y + 17.5 * z : ℝ) = 2845 ∧
  (x - y : ℤ).natAbs ≤ 2 ∧ (y - z).natAbs ≤ 2 := 
sorry

end book_price_distribution_l651_651882


namespace min_value_of_b_minus_a_l651_651173

noncomputable def f (x : ℝ) : ℝ :=
1 + x - x^2 / 2 + x^3 / 3 - x^4 / 4 + ∑ i in (finset.range 2016).filter (λ n, n > 4), (-1)^i * x^i / i

noncomputable def g (x : ℝ) : ℝ :=
1 - x + x^2 / 2 - x^3 / 3 + x^4 / 4 + ∑ i in (finset.range 2016).filter (λ n, n > 4), (-1)^(i+1) * x^i / i

noncomputable def F (x : ℝ) : ℝ :=
f (x + 3) * g (x - 4)

theorem min_value_of_b_minus_a :
  ∃ (a b : ℤ), (∀ x, F x = 0 → a ≤ x ∧ x ≤ b) ∧ b - a = 9 :=
sorry

end min_value_of_b_minus_a_l651_651173


namespace decreased_area_of_equilateral_triangle_l651_651521

theorem decreased_area_of_equilateral_triangle 
    (A : ℝ) (hA : A = 100 * Real.sqrt 3) 
    (decrease : ℝ) (hdecrease : decrease = 6) :
    let s := Real.sqrt (4 * A / Real.sqrt 3)
    let s' := s - decrease
    let A' := (s' ^ 2 * Real.sqrt 3) / 4
    A - A' = 51 * Real.sqrt 3 :=
by
  sorry

end decreased_area_of_equilateral_triangle_l651_651521


namespace distance_X_to_AD_l651_651680

open Real

theorem distance_X_to_AD (s : ℝ) (h : 0 < s) :
  let X := (s / 4, s / 2) in
  let intercept_distance := X.1 in
  intercept_distance = s / 4 :=
begin
  -- Definitions
  let midpoint_AB := (s / 2, 0) in
  let midpoint_BC := (s, s / 2) in
  let semicircle_AB := λ (x y : ℝ), (x - s / 2) ^ 2 + y ^ 2 = (s / 2) ^ 2 in
  let semicircle_BC := λ (x y : ℝ), (x - s) ^ 2 + (y - s / 2) ^ 2 = (s / 2) ^ 2 in

  -- Prove X is the intersection of the equations
  have HX_eq_AB : semicircle_AB X.1 X.2 := by sorry,
  have HX_eq_BC : semicircle_BC X.1 X.2 := by sorry,

  -- Verify the computed distance
  have intercept_distance_def : intercept_distance = X.1 := rfl,
  rw intercept_distance_def,

  -- Final equality
  ring,
  norm_num,
end

end distance_X_to_AD_l651_651680


namespace problem1_problem2_l651_651218

open Real

variables {A B C : ℝ} {a b c : ℝ}

-- Condition: In $\triangle ABC$, sides opposite to angles A, B, C respectively, and $\cos A = \frac{1}{3}$.
axiom cos_A_eq_one_third : cos A = 1 / 3

-- Problem (1): Prove $\sin^2 \frac{B+C}{2} + \cos 2A = -\frac{1}{9}$ given the conditions.
theorem problem1 (h1 : A + B + C = π) (cos_A_eq_one_third : cos A = 1 / 3) :
    sin^2 ((B + C) / 2) + cos (2 * A) = -1 / 9 :=
sorry

-- Additional condition for problem (2): $a = \sqrt{3}$.
axiom a_eq_sqrt_three : a = sqrt 3

-- Problem (2): Prove that $bc \leq 9 / 4$ and that the maximum value of $bc$ is $9 / 4$.
theorem problem2 (h1 : A + B + C = π) 
                 (cos_A_eq_one_third : cos A = 1 / 3) 
                 (a_eq_sqrt_three : a = sqrt 3) :
    ∃ b c, a = sqrt 3 ∧ b = c ∧ b * c = 9 / 4 :=
sorry

end problem1_problem2_l651_651218


namespace original_cube_volume_l651_651771

theorem original_cube_volume
  (a : ℝ)
  (h : (a + 2) * (a - 1) * a = a^3 + 14) :
  a^3 = 64 :=
by
  sorry

end original_cube_volume_l651_651771


namespace alyosha_cube_cut_l651_651500

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651500


namespace line_intersects_circle_l651_651807

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

noncomputable def line_eq (x y k : ℝ) : Prop := y - 1 = k * (x - 1)

theorem line_intersects_circle (k : ℝ) :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq x y k :=
by {
  use 1,
  use 1,
  split,
  sorry, -- proof that (1, 1) satisfies the circle equation
  sorry  -- proof that (1, 1) satisfies the line equation
}

end line_intersects_circle_l651_651807


namespace second_markdown_percentage_l651_651916

theorem second_markdown_percentage 
  (P : ℝ) 
  (h1 : 0.80 * P > 0)
  (h2 : 0.72 * P > 0) 
  (h3 : 0 < P) :
  ∃ (X : ℝ), (1 - X / 100) * 0.80 * P = 0.72 * P ∧ X = 10 :=
begin
  sorry
end

end second_markdown_percentage_l651_651916


namespace focus_of_parabola_x2_eq_neg_4y_l651_651323

theorem focus_of_parabola_x2_eq_neg_4y :
  (∀ x y : ℝ, x^2 = -4 * y → focus = (0, -1)) := 
sorry

end focus_of_parabola_x2_eq_neg_4y_l651_651323


namespace cube_decomposition_l651_651478

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651478


namespace find_mt_l651_651742

noncomputable def g : ℝ → ℝ := sorry

axiom g_property :
  ∀ (x y : ℝ), g ((x - y)^2 + 1) = g(x)^2 - 2 * x * g(y) + y^2 + 1

theorem find_mt :
  let m := {y : ℝ | g 2 = y}.to_finset.card,
      t := {y : ℝ | g 2 = y}.to_finset.sum (λ x, x)
  in m * t = 10 := sorry

end find_mt_l651_651742


namespace mr_slinkums_shipments_l651_651919

theorem mr_slinkums_shipments 
  (T : ℝ) 
  (h : (3 / 4) * T = 150) : 
  T = 200 := 
sorry

end mr_slinkums_shipments_l651_651919


namespace city_mayor_reform_l651_651014

-- Define what it means for cities to be connected
variables (N : ℕ) (neighbors : Finset (Finset ℕ))
  -- Assume neighbors are always pairs of distinct cities
  (connections : ∀ (c₁ c₂ : ℕ), c₁ ∈ neighbors c₂ ↔ (c₂, c₁) ∈ neighbors ∧ c₁ ≠ c₂)

-- Define the conditions for traveling between cities
variables (path_exists : ∀ (c₁ c₂ : ℕ), ∃ (path : list ℕ), c₁ ∈ path ∧ c₂ ∈ path ∧ ∀ c ∈ path, c ∈ (neighbors (head path)))
  (no_return_via_diff_roads : ∀ (path : list ℕ), ∀ (c₁ ∈ path) (c₂ ∈ path), (c₂ ≠ head path → c₁ ≠ c₂))

-- Define the reform conditions
variables (reform : ℕ → ℕ)
  -- After the reform, neighboring mayors still govern neighboring cities
  (neighbor_after_reform : ∀ (c₁ c₂ : ℕ), c₁ ∈ neighbors c₂ → reform c₁ ∈ neighbors (reform c₂))

theorem city_mayor_reform (N : ℕ) (neighbors : Finset (Finset ℕ))
  (connections : ∀ (c₁ c₂ : ℕ), c₁ ∈ neighbors c₂ ↔ (c₂, c₁) ∈ neighbors ∧ c₁ ≠ c₂)
  (path_exists : ∀ (c₁ c₂ : ℕ), ∃ (path : list ℕ), c₁ ∈ path ∧ c₂ ∈ path ∧ ∀ c ∈ path, c ∈ (neighbors (head path)))
  (no_return_via_diff_roads : ∀ (path : list ℕ), ∀ (c₁ c₂ : ℕ), c₁ ∈ path → c₂ ∈ path → (c₂ ≠ head path → c₁ ≠ c₂))
  (reform : ℕ → ℕ)
  (neighbor_after_reform : ∀ (c₁ c₂ : ℕ), c₁ ∈ neighbors c₂ → reform c₁ ∈ neighbors (reform c₂)) :
  ∃ (c : ℕ), reform c = c ∨ ∃ (c₁ c₂ : ℕ), c₁ ∈ neighbors c₂ ∧ reform c₁ = c₂ ∧ reform c₂ = c₁ :=
by sorry

end city_mayor_reform_l651_651014


namespace alyosha_cube_problem_l651_651467

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651467


namespace coefficient_x4_of_product_is_25_l651_651849

-- Define polynomials as polynomial terms in the corresponding conditions
def poly1 : Polynomial ℤ := -5 * Polynomial.X^3 - 5 * Polynomial.X^2 - 7 * Polynomial.X + 1
def poly2 : Polynomial ℤ := -Polynomial.X^3 - 6 * Polynomial.X + 1

-- State the theorem
theorem coefficient_x4_of_product_is_25 : 
  (Polynomial.coeff (poly1 * poly2) 4 = 25) :=
by
  sorry

end coefficient_x4_of_product_is_25_l651_651849


namespace element_subset_a_l651_651710

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l651_651710


namespace smallest_integer_with_12_factors_l651_651391

theorem smallest_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∏ (d ∈ (List.range k).filter (λ n, k % n = 0), 1) = 12) ∧ ∀ m : ℕ, (m > 0 ∧ (∏ (d ∈ (List.range m).filter (λ n, m % n = 0), 1) = 12) → k ≤ m) ∧ k = 60 :=
sorry

end smallest_integer_with_12_factors_l651_651391


namespace unique_factorization_l651_651697

theorem unique_factorization (m : ℕ) (h₀ : m % 4 = 2) :
  ∀ a b c d : ℕ, m = a * b → m = c * d → 0 < a - b ∧ a - b < (sqrt (5 + 4 * sqrt (4 * m + 1))) → 0 < c - d ∧ c - d < (sqrt (5 + 4 * sqrt (4 * m + 1))) → a = c ∧ b = d :=
by
  sorry

end unique_factorization_l651_651697


namespace count_5_letter_words_with_vowels_l651_651198

def is_vowel (ch : Char) : Prop := ch = 'A' ∨ ch = 'E'

def valid_letters : List Char := ['A', 'B', 'C', 'D', 'E']

def is_valid_word (word : List Char) : Prop := 
  word.length = 5 ∧ word.all (λ ch => ch ∈ valid_letters)

def contains_vowel (word : List Char) : Prop :=
  ∃ ch ∈ word, is_vowel ch

theorem count_5_letter_words_with_vowels : 
  let total_words := 5^5 in
  let consonant_words := 3^5 in
  total_words - consonant_words = 2882 :=
  by
    sorry

end count_5_letter_words_with_vowels_l651_651198


namespace average_practice_hours_l651_651023

noncomputable def total_hours_weekdays : ℕ := 2 * 5
noncomputable def hours_weekend : ℕ := 11
noncomputable def total_hours_week : ℕ := total_hours_weekdays + hours_weekend
noncomputable def average_hours_per_day : ℚ := total_hours_week / 7

theorem average_practice_hours :
  average_hours_per_day = 3 := 
  by
    rw [total_hours_week, total_hours_weekdays]
    norm_num
    exact (by norm_num : 21 / 7 = 3)
 
end average_practice_hours_l651_651023


namespace subset_a_eq_1_l651_651705

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651705


namespace a_2008_is_2_l651_651245

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 7
  else (sequence (n - 2) * sequence (n - 1)) % 10

theorem a_2008_is_2 : sequence 2008 = 2 :=
  by sorry

end a_2008_is_2_l651_651245


namespace find_x_exists_unique_l651_651997

theorem find_x_exists_unique (n : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ x = p * q * r) (h3 : 11 ∣ x) : x = 59048 :=
sorry

end find_x_exists_unique_l651_651997


namespace geometry_problem_l651_651676

noncomputable def reflection_point (C A1 B1 : Point) : Point := sorry -- Definition placeholder
noncomputable def orthocenter (A B C : Point) : Point := sorry -- Definition placeholder
noncomputable def circumcenter (A B C : Point) : Point := sorry -- Definition placeholder
noncomputable def altitude_foot (A B C : Point) : Point := sorry -- Definition placeholder
noncomputable def is_concyclic (A B C D : Point) : Prop := sorry -- Definition placeholder

variable (A B C : Point)

theorem geometry_problem (H O A1 B1 C1 C2 : Point)
  (acute_triangle : IsAcuteAngledTriangle A B C)
  (H_def : H = orthocenter A B C)
  (O_def : O = circumcenter A B C)
  (A1_def : A1 = altitude_foot A B C)
  (B1_def : B1 = altitude_foot B A C)
  (C1_def : C1 = altitude_foot C A B)
  (C2_def : C2 = reflection_point C A1 B1):
  is_concyclic H O C1 C2 := sorry

end geometry_problem_l651_651676


namespace cube_decomposition_l651_651476

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651476


namespace total_trees_in_gray_areas_l651_651036

theorem total_trees_in_gray_areas (white_region_first : ℕ) (white_region_second : ℕ)
    (total_first : ℕ) (total_second : ℕ)
    (h1 : white_region_first = 82) (h2 : white_region_second = 82)
    (h3 : total_first = 100) (h4 : total_second = 90) :
  (total_first - white_region_first) + (total_second - white_region_second) = 26 := by
  sorry

end total_trees_in_gray_areas_l651_651036


namespace sum_of_intersections_l651_651331

theorem sum_of_intersections :
  let y_expr := λ x : ℝ, x^3 - 4*x + 3
  let line_eq := λ x y : ℝ, x + 5*y = 5
  ∃ x1 x2 x3 y1 y2 y3 : ℝ,
    y_expr x1 = y1 ∧ line_eq x1 y1 ∧
    y_expr x2 = y2 ∧ line_eq x2 y2 ∧
    y_expr x3 = y3 ∧ line_eq x3 y3 ∧
    (x1 + x2 + x3 = 0 ∧ y1 + y2 + y3 = 3) :=
by
  sorry

end sum_of_intersections_l651_651331


namespace chinese_number_representation_l651_651237

theorem chinese_number_representation :
  ∀ (祝 贺 华 杯 赛 : ℕ),
  祝 = 4 → 贺 = 8 → 
  华 ≠ 杯 ∧ 华 ≠ 赛 ∧ 杯 ≠ 赛 ∧ 华 ≠ 祝 ∧ 华 ≠ 贺 ∧ 杯 ≠ 祝 ∧ 杯 ≠ 贺 ∧ 赛 ≠ 祝 ∧ 赛 ≠ 贺 → 
  华 ≥ 1 ∧ 华 ≤ 9 → 杯 ≥ 1 ∧ 杯 ≤ 9 → 赛 ≥ 1 ∧ 赛 ≤ 9 → 
  华 * 100 + 杯 * 10 + 赛 = 7632 :=
begin
  sorry
end

end chinese_number_representation_l651_651237


namespace find_x3_l651_651347

theorem find_x3 (x1 x2 x3 : ℝ) (hx1 : x1 = 2) (hx2 : x2 = 8) (hx2_gt_1 : 1 < x2) (hx1_lt_x2 : x1 < x2):
  let f := λ x : ℝ, x^2 in
  let A := (x1, f x1) in
  let B := (x2, f x2) in
  let xC := x1 + 1/9 * (x2 - x1) in
  let yC := f xC in
  yC = x3^2 ∧ 1 < x3 →
  x3 = 8/3 :=
sorry

end find_x3_l651_651347


namespace sin_cos_value_sin_plus_cos_value_l651_651130

noncomputable def given_condition (θ : ℝ) : Prop := 
  (Real.tan θ + 1 / Real.tan θ = 2)

theorem sin_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ * Real.cos θ = 1 / 2 :=
sorry

theorem sin_plus_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ + Real.cos θ = Real.sqrt 2 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 :=
sorry

end sin_cos_value_sin_plus_cos_value_l651_651130


namespace boys_running_speed_l651_651027
-- Import the necessary libraries

-- Define the input conditions:
def side_length : ℝ := 50
def time_seconds : ℝ := 80
def conversion_factor_meters_to_kilometers : ℝ := 1000
def conversion_factor_seconds_to_hours : ℝ := 3600

-- Define the theorem:
theorem boys_running_speed :
  let perimeter := 4 * side_length
  let distance_kilometers := perimeter / conversion_factor_meters_to_kilometers
  let time_hours := time_seconds / conversion_factor_seconds_to_hours
  distance_kilometers / time_hours = 9 :=
by
  sorry

end boys_running_speed_l651_651027


namespace camila_hikes_number_l651_651072

-- Definitions based on the conditions
def camila_hikes : ℕ := C
def amanda_hikes := 8 * camila_hikes
def steven_hikes := amanda_hikes + 15
def additional_hikes := 4 * 16  -- Camila plans 4 hikes a week for 16 weeks

-- Statement to be proved
theorem camila_hikes_number : ∃ (C : ℕ), camila_hikes + additional_hikes = steven_hikes ∧ C = 7 :=
by
  existsi 7
  unfold camila_hikes amanda_hikes steven_hikes additional_hikes
  simp
  sorry

end camila_hikes_number_l651_651072


namespace marble_ratio_l651_651766

theorem marble_ratio :
  ∀ (E M S : ℕ), 
    M + E = S - 5 ∧ S = 50 ∧ M = 30 → M / E = 2 :=
by
  intros E M S 
  intro h
  cases h with h1 h2
  cases h2 with h2_1 h2_2
  sorry

end marble_ratio_l651_651766


namespace janina_must_sell_21_pancakes_l651_651259

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l651_651259


namespace percentage_passed_all_subjects_l651_651677

-- Define the percentages
def total_students : ℝ := 100
def percentage_failed_hindi : ℝ := 30
def percentage_failed_english : ℝ := 45
def percentage_failed_math : ℝ := 25
def percentage_failed_science : ℝ := 40
def percentage_failed_hindi_english : ℝ := 12
def percentage_failed_hindi_math : ℝ := 15
def percentage_failed_hindi_science : ℝ := 18
def percentage_failed_english_math : ℝ := 20
def percentage_failed_english_science : ℝ := 22
def percentage_failed_math_science : ℝ := 24
def percentage_failed_all_subjects : ℝ := 10

-- Define the statement to prove
theorem percentage_passed_all_subjects :
  let failed_one_or_more := 
    percentage_failed_hindi + percentage_failed_english + percentage_failed_math + 
    percentage_failed_science - (percentage_failed_hindi_english + percentage_failed_hindi_math + 
    percentage_failed_hindi_science + percentage_failed_english_math + 
    percentage_failed_english_science + percentage_failed_math_science) + 
    percentage_failed_all_subjects in
  let passed_all := total_students - failed_one_or_more in
  passed_all = 61 := 
by {
  sorry
}

end percentage_passed_all_subjects_l651_651677


namespace line_through_midpoint_of_chord_of_circle_l651_651165

theorem line_through_midpoint_of_chord_of_circle :
  let P := (1, 1)
  let C := { p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9 }
  let L := { p : ℝ × ℝ | 2 * p.1 - p.2 - 1 = 0 }
  (∃ MN : set (ℝ × ℝ), P ∈ MN ∧ ∃ p1 p2 ∈ C, p1 ≠ p2 ∧ MN = { p | ∃ λ, p = (λ * p1.1 + (1 - λ) * p2.1, λ * p1.2 + (1 - λ) * p2.2) })
    → ∃ p : ℝ × ℝ, p ∈ L :=
begin
  sorry
end

end line_through_midpoint_of_chord_of_circle_l651_651165


namespace remainder_17_pow_63_mod_7_l651_651355

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651355


namespace tan_sub_sin_eq_sq3_div2_l651_651935

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l651_651935


namespace max_unique_problems_l651_651799

namespace OlympiadProblem

-- Definitions matching the problem conditions:
def num_grades : Nat := 6
def problems_per_grade : Nat := 7
def unique_per_grade : Nat := 4
def common_problems : Nat := 3

/-- 
Statement to prove:
Given 6 grades and each grade's problem set having 7 problems where 4 are unique to the grade, 
and 3 problems are shared among all grades, the maximum number of unique problems included 
in the olympiad is 27.
-/
theorem max_unique_problems : (num_grades * unique_per_grade) + common_problems = 27 := 
by
  simp [num_grades, unique_per_grade, common_problems]
  sorry

end OlympiadProblem

end max_unique_problems_l651_651799


namespace intervals_of_monotonicity_no_tangent_through_origin_l651_651591

noncomputable def f (x m : ℝ) : ℝ := (1/2)*x^2 + Real.log x - m*x

def derivative_f (x m : ℝ) : ℝ := x - 1/x - m

def discriminant (m : ℝ) : ℝ := m^2 - 4

-- Lean statement to prove intervals of monotonicity
theorem intervals_of_monotonicity (m : ℝ) (hm : 0 < m) : 
  (0 < m ∧ m < 2) → (∀ x > 0, derivative_f x m > 0) ∨
  (m = 2) → (∃ x : ℝ, derivative_f x m = 0) ∧ (∀ x > 0, derivative_f x m = 0) ∨
  (2 < m) → (derivative_f x m = 0) ∧ 
  (∃ a b : ℝ, a < b ∧ derivative_f a m > 0 ∧ derivative_f b m < 0 ∧ 
  ∀ x, (0 < x ∧ x < a) ∨ (x > b) → derivative_f x m > 0 ∧ (a < x ∧ x < b → derivative_f x m < 0)) :=
sorry

-- Lean statement to prove no tangent line through the origin
theorem no_tangent_through_origin (m : ℝ) (hm : 0 < m) :
  ¬(∃ x0 : ℝ, f x0 m = (x0 - 1/x0 - m) * x0) :=
sorry

end intervals_of_monotonicity_no_tangent_through_origin_l651_651591


namespace subset_a_eq_1_l651_651724

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651724


namespace initial_blue_balls_l651_651002

theorem initial_blue_balls (B : ℕ) 
  (h1 : 18 - 3 = 15) 
  (h2 : (B - 3) / 15 = 1 / 5) : 
  B = 6 :=
by sorry

end initial_blue_balls_l651_651002


namespace college_enrollment_1995_l651_651889

noncomputable def enrollment_1991_A := 2000
noncomputable def enrollment_1991_B := 1000

noncomputable def enrollment_1992_A := enrollment_1991_A * 1.25
noncomputable def enrollment_1992_B := enrollment_1991_B * 1.10

noncomputable def enrollment_1993_A := enrollment_1992_A * 1.15
noncomputable def enrollment_1993_B := enrollment_1992_B * 1.20

noncomputable def enrollment_1994_A := enrollment_1993_A * 1.10
noncomputable def enrollment_1994_B := enrollment_1993_B * 1.10
noncomputable def enrollment_1994_A_afterC := enrollment_1994_A * 0.95
noncomputable def enrollment_1994_B_afterC := enrollment_1994_B * 0.95

noncomputable def enrollment_1994_C := 300
noncomputable def enrollment_1995_C := enrollment_1994_C * 1.50

noncomputable def total_enrollment_1995 := enrollment_1994_A_afterC + enrollment_1994_B_afterC + enrollment_1995_C

noncomputable def total_enrollment_1991 := enrollment_1991_A + enrollment_1991_B

noncomputable def percent_change := (total_enrollment_1995 - total_enrollment_1991) / total_enrollment_1991 * 100

theorem college_enrollment_1995 : 
  total_enrollment_1995 = 4833.775 ∧ percent_change = 61.1258333 :=
by 
  sorry

end college_enrollment_1995_l651_651889


namespace exists_pal_number_with_n_digits_l651_651349

def is_pal (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 ∧ ∃ k : ℕ, (∑ d in n.digits 10, d * d) = k * k

theorem exists_pal_number_with_n_digits (n : ℕ) (h : n > 1) : ∃ m : ℕ, is_pal m ∧ m.digits 10.length = n :=
by
  sorry

end exists_pal_number_with_n_digits_l651_651349


namespace complement_intersection_l651_651168

open Set

variable {x : ℝ}

def A : Set ℝ := {x | log 2 (3 - x) ≤ 2}
def B : Set ℝ := {x | abs (x - 3) ≤ 2}

theorem complement_intersection :
  (A ∩ B)ᶜ = {x | x < 1 ∨ x ≥ 3} :=
by
  sorry

end complement_intersection_l651_651168


namespace exists_q_r_polynomials_l651_651419

theorem exists_q_r_polynomials (n : ℕ) (p : Polynomial ℝ) 
  (h_deg : p.degree = n) 
  (h_monic : p.leadingCoeff = 1) :
  ∃ q r : Polynomial ℝ, 
    q.degree = n ∧ r.degree = n ∧ 
    (∀ x : ℝ, q.eval x = 0 → r.eval x = 0) ∧
    (∀ y : ℝ, r.eval y = 0 → q.eval y = 0) ∧
    q.leadingCoeff = 1 ∧ r.leadingCoeff = 1 ∧ 
    p = (q + r) / 2 := 
sorry

end exists_q_r_polynomials_l651_651419


namespace sqrt_16_eq_pm_4_l651_651817

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l651_651817


namespace championship_titles_l651_651124

theorem championship_titles {S T : ℕ} (h_S : S = 4) (h_T : T = 3) : S^T = 64 := by
  rw [h_S, h_T]
  norm_num

end championship_titles_l651_651124


namespace systematic_sampling_correct_l651_651514

theorem systematic_sampling_correct :
  ∃ (seq : List ℕ), 
  seq = [6, 18, 30, 42, 54] ∧ 
  (∃ (n k : ℕ), ∃ (students : List ℕ), n = 60 ∧ k = 12 ∧ 
    students = List.range (n + 1) ∧ 
    ∀ (i : ℕ), i < 5 → List.nth (seq) i = List.nth (students) (6 + i * k)) :=
begin
  sorry
end

end systematic_sampling_correct_l651_651514


namespace prob_abd_together_l651_651866

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

lemma perm_count_of_8 : (8.fact : ℝ) = 40320 :=
by norm_num

lemma perm_count_of_6 : (6.fact : ℝ) = 720 :=
by norm_num

lemma perm_count_of_3 : (3.fact : ℝ) = 6 :=
by norm_num

theorem prob_abd_together :
  let total_arrangements := (8.fact : ℝ),
      group_arrangements := (6.fact : ℝ) * (3.fact : ℝ) in
  group_arrangements / total_arrangements = 1 / 9.3333 :=
by {
  let total_arrangements := 40320,
  let group_arrangements := 720 * 6,
  have eq1 : total_arrangements = 8.fact := by norm_num,
  have eq2 : group_arrangements = 6.fact * 3.fact := by norm_num,
  have eq3 : (6.fact : ℝ) * (3.fact : ℝ) = 4320 := by norm_num,
  let probability := group_arrangements / total_arrangements,
  have : probability = 1 / 9.3333333333333... := by        calc
     probability = 4320 / 40320 : by norm_cast
              ... = 1 / 9.333333333333333... : by norm_num,
  exact this,
}

end prob_abd_together_l651_651866


namespace number_of_pairs_l651_651973

open_locale big_operators

/-- 
  Prove that the number of pairs of integers (a, b) 
  such that 1 ≤ a < b ≤ 57 
  and a^2 % 57 < b^2 % 57 
  is 738.
-/
theorem number_of_pairs (count_pairs : ℕ) :
  count_pairs = 738 :=
begin
  sorry
end

end number_of_pairs_l651_651973


namespace fraction_of_pelicans_moved_l651_651120

-- Conditions
variables (P : ℕ)
variables (n_Sharks : ℕ := 60) -- Number of sharks in Pelican Bay
variables (n_Pelicans_original : ℕ := 2 * P) -- Twice the original number of Pelicans in Shark Bite Cove
variables (n_Pelicans_remaining : ℕ := 20) -- Number of remaining Pelicans in Shark Bite Cove

-- Proof to show fraction that moved
theorem fraction_of_pelicans_moved (h : 2 * P = n_Sharks) : (P - n_Pelicans_remaining) / P = 1 / 3 :=
by {
  sorry
}

end fraction_of_pelicans_moved_l651_651120


namespace sqrt_of_sixteen_l651_651815

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l651_651815


namespace correct_options_on_ellipse_l651_651156

-- Condition 1: M is a point on the ellipse C: (x^2 / 8) + (y^2 / 4) = 1.
-- Condition 2: F_1 and F_2 are the left and right foci of the ellipse.

theorem correct_options_on_ellipse
  (M : ℝ × ℝ)
  (F₁ F₂ : ℝ × ℝ)
  (a b : ℝ)
  (h_ellipse1 : a = 2 * sqrt 2)
  (h_ellipse2 : b = 2)
  (h_point_on_ellipse : (M.1 ^ 2 / 8) + (M.2 ^ 2 / 4) = 1)
  (h_foci : F₁ = (-2, 0) ∧ F₂ = (2, 0)) :
  ((sqrt 2 / 2) = 2 / (2 * sqrt 2)) ∧
  (2 * b = 4) ∧
  (1 / 2 * (2 * 2) * b = 4) :=
by
  -- All these proofs are omitted with sorry
  sorry

end correct_options_on_ellipse_l651_651156


namespace increasing_F_f_additive_towards_f_sum_f_n_additive_towards_f_sum_l651_651751

variable {f : ℝ → ℝ} {x : ℝ} {x1 x2 : ℝ} {n : ℕ} {x_vals : Fin n → ℝ}

-- Given conditions
theorem increasing_F (h : ∀ x > 0, deriv f x > f x / x) : ∀ x > 0, deriv (λ x, f x / x) x > 0 :=
sorry

theorem f_additive_towards_f_sum (h : ∀ x > 0, deriv f x > f x / x) : ∀ x1 x2 > 0, f x1 + f x2 < f (x1 + x2) :=
sorry

theorem f_n_additive_towards_f_sum (n : ℕ) (h : ∀ x > 0, deriv f x > f x / x) : ∀ x_vals : Fin n → ℝ, (∀ i, x_vals i > 0) → (∑ i, f (x_vals i)) < f (∑ i, x_vals i)
 :=
sorry

end increasing_F_f_additive_towards_f_sum_f_n_additive_towards_f_sum_l651_651751


namespace probability_product_greater_than_zero_l651_651842

def interval : Set ℝ := {x : ℝ | -30 ≤ x ∧ x ≤ 15}

theorem probability_product_greater_than_zero :
  let prob (S : Set ℝ) : ℝ := (measure_theory.volume S) / (measure_theory.volume interval) in
  let negative_set := {x : ℝ | -30 ≤ x ∧ x < 0} in
  let positive_set := {x : ℝ | 0 < x ∧ x ≤ 15} in
  prob (negative_set) = 2 / 3 ∧
  prob (positive_set) = 1 / 3 ∧
  ((prob (negative_set) * prob (negative_set)) + (prob (positive_set) * prob (positive_set)) = 5 / 9) :=
by
  sorry

end probability_product_greater_than_zero_l651_651842


namespace subset_a_eq_1_l651_651702

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651702


namespace log_change_of_base_l651_651302

theorem log_change_of_base (a b N : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : N > 0) (h4 : a ≠ 1) (h5 : b ≠ 1) :
  Real.logN b N = (Real.logN a N) / (Real.logN a b) :=
sorry

end log_change_of_base_l651_651302


namespace problem1_problem2_l651_651194

-- Definitions of vectors
def vec_a (θ : Real) : Real × Real := (2 * Real.cos θ, Real.sin θ)
def vec_b : Real × Real := (1, 2)

-- Problem statement 1
theorem problem1 (θ : Real) (h : Real.sin θ = 4 * Real.cos θ) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (2 * Real.sin θ + Real.cos θ) = 10 / 9 := by
  sorry

-- Definitions specific to problem 2
def vec_scal_mult (s : Real) (v : Real × Real) : Real × Real := (s * v.1, s * v.2)
def vec_add (v1 v2 : Real × Real) : Real × Real := (v1.1 + v2.1, v1.2 + v2.2)
def vec_dot (v1 v2 : Real × Real) : Real := v1.1 * v2.1 + v1.2 * v2.2

-- Problem statement 2
theorem problem2 (θ : Real) (hθ : θ = Real.pi / 4) (t : Real)
  (h : vec_dot (vec_add (vec_scal_mult 2 (vec_a θ)) (vec_scal_mult (-t) vec_b))
    (vec_add (vec_scal_mult (Real.sqrt 2) (vec_a θ)) vec_b) = 0) :
  t = Real.sqrt 2 := by
  sorry

end problem1_problem2_l651_651194


namespace normal_line_at_x0_is_correct_l651_651110

noncomputable def curve (x : ℝ) : ℝ := x^(2/3) - 20

def x0 : ℝ := -8

def normal_line_equation (x : ℝ) : ℝ := 3 * x + 8

theorem normal_line_at_x0_is_correct : 
  ∃ y0 : ℝ, curve x0 = y0 ∧ y0 = curve x0 ∧ normal_line_equation x0 = y0 :=
sorry

end normal_line_at_x0_is_correct_l651_651110


namespace alyosha_cube_problem_l651_651466

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651466


namespace seventeen_power_sixty_three_mod_seven_l651_651361

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651361


namespace problem_may_not_be_equal_l651_651922

-- Define the four pairs of expressions
def expr_A (a b : ℕ) := (a + b) = (b + a)
def expr_B (a : ℕ) := (3 * a) = (a + a + a)
def expr_C (a b : ℕ) := (3 * (a + b)) ≠ (3 * a + b)
def expr_D (a : ℕ) := (a ^ 3) = (a * a * a)

-- State the theorem stating that the expression in condition C may not be equal
theorem problem_may_not_be_equal (a b : ℕ) : (3 * (a + b)) ≠ (3 * a + b) :=
by
  sorry

end problem_may_not_be_equal_l651_651922


namespace digit_sum_of_product_l651_651804

def digits_after_multiplication (a b : ℕ) : ℕ :=
  let product := a * b
  let units_digit := product % 10
  let tens_digit := (product / 10) % 10
  tens_digit + units_digit

theorem digit_sum_of_product :
  digits_after_multiplication 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909 = 9 :=
by 
  -- proof goes here
sorry

end digit_sum_of_product_l651_651804


namespace probability_of_different_cousins_name_l651_651513

theorem probability_of_different_cousins_name :
  let total_letters := 19
  let amelia_letters := 6
  let bethany_letters := 7
  let claire_letters := 6
  let probability := 
    2 * ((amelia_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)) +
         (amelia_letters / (total_letters : ℚ)) * (claire_letters / (total_letters - 1 : ℚ)) +
         (claire_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)))
  probability = 40 / 57 := sorry

end probability_of_different_cousins_name_l651_651513


namespace inclination_angle_of_intersected_line_l651_651661

noncomputable def inclination_angle (m : ℝ) : ℝ :=
15 -- Assume we need to define inclination angle function to give the problem context

theorem inclination_angle_of_intersected_line (l1 l2 : LinearMap ℝ (ℝ × ℝ))
  (m_intercept_len : ℝ)
  (h_l1 : l1 = LinearMap.proj (^1, -1 + 1))
  (h_l2 : l2 = LinearMap.proj (^1, -1 + 3))
  (h_len : m_intercept_len = 2 * Real.sqrt 2) :
  inclination_angle m = 15 ∨ inclination_angle m = 75 := by
  sorry

end inclination_angle_of_intersected_line_l651_651661


namespace probability_green_ball_l651_651540

/-- 
Given three containers with specific numbers of red and green balls, 
and the probability of selecting each container being equal, 
the probability of picking a green ball when choosing a container randomly is 7/12.
-/
theorem probability_green_ball :
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  (green_I + green_II + green_III) = 7 / 12 :=
by 
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  have : (green_I + green_II + green_III) = (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) := by rfl
  have : (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) = (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) := by rfl
  have : (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) = (1 / 9 + 2 / 9 + 1 / 4) := by rfl
  have : (1 / 9 + 2 / 9 + 1 / 4) = (4 / 36 + 8 / 36 + 9 / 36) := by rfl
  have : (4 / 36 + 8 / 36 + 9 / 36) = 21 / 36 := by rfl
  have : 21 / 36 = 7 / 12 := by rfl
  rfl

end probability_green_ball_l651_651540


namespace area_of_triangle_QCA_l651_651091

/-- Given points Q, A, and C, determine the area of triangle QCA in terms of p -/
theorem area_of_triangle_QCA (p : ℝ) : 
  let Q := (0 : ℝ, 15 : ℝ)
      A := (3 : ℝ, 15 : ℝ)
      C := (0 : ℝ, p : ℝ) in
  let base := (A.1 - Q.1)
      height := (Q.2 - C.2) in
  (1 / 2) * base * height = (45 / 2) - (3 * p / 2) :=
by 
{ 
  simp [Q, A, C, base, height],
  sorry
}

end area_of_triangle_QCA_l651_651091


namespace batsman_average_after_25th_innings_l651_651885

theorem batsman_average_after_25th_innings (A : ℝ) (h_pre_avg : (25 * (A + 3)) = (24 * A + 80))
  : A + 3 = 8 := 
by
  sorry

end batsman_average_after_25th_innings_l651_651885


namespace frog_jumps_2_stones_100_times_frog_skips_incrementally_100_times_l651_651832

-- Definitions and Lean statements based on the conditions

def stones := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def start := 1

-- Part (a)
theorem frog_jumps_2_stones_100_times :
  (start + 2 * 100) % 10 = 1 :=
by
  sorry

-- Part (b)
theorem frog_skips_incrementally_100_times :
  (start + (∑ i in Finset.range 101, i) % 10) = 1 :=
by
  sorry

end frog_jumps_2_stones_100_times_frog_skips_incrementally_100_times_l651_651832


namespace max_min_f_range_m_l651_651177

noncomputable def f (x : ℝ) : ℝ := 2 * sin ((π/4) + x) ^ 2 - sqrt 3 * cos (2 * x)

theorem max_min_f (x : ℝ) (h : x ∈ Icc (π / 4) (π / 2)) :
  f x ≤ 3 ∧ f x ≥ 2 :=
sorry

theorem range_m (m : ℝ) :
  (∀ x ∈ Icc (π / 4) (π / 2), abs (f x - m) < 2) ↔ (1 < m ∧ m < 4) :=
sorry

end max_min_f_range_m_l651_651177


namespace find_A_and_B_l651_651794

theorem find_A_and_B : ∃ (A B : ℕ), A ≠ 0 ∧ 111 * A * A - 99 * A = B ∧ (111 * A * A + 111 * A = 111 * (1000 * A + B)) ∧ A = 9 ∧ B = 0 :=
by
  -- We declare the values for A and B based on the solution
  let A := 9
  let B := 0
  -- We check that A and B meet the conditions
  use [A, B]
  -- We will now provide the conditions
  split
  { -- A ≠ 0
    show A ≠ 0,
    exact dec_trivial
  },
  split
  { -- 111 * A * A - 99 * A = B
    show 111 * A * A - 99 * A = B,
    calc
      111 * A * A - 99 * A = 111 * 9 * 9 - 99 * 9 : by refl
      ... = 8100 - 891
      ... = 0 : by refl
  },
  split
  { -- 111 * A * A + 111 * A = 111 * (1000 * A + B)
    show 111 * A * A + 111 * A = 111 * (1000 * A + B),
    calc
      111 * A * A + 111 * A = 111 * 9 * 9 + 111 * 9 : by refl
      ... = 8100 + 999
      ... = 111 * (1000 * 9 + 0) : by refl
  },
  split
  { -- A = 9
    show A = 9,
    exact rfl
  },
  { -- B = 0
    show B = 0,
    exact rfl
  }

end find_A_and_B_l651_651794


namespace non_integer_sum_exists_l651_651743

theorem non_integer_sum_exists (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ M : ℕ, ∀ n : ℕ, n > M → ¬ ∃ t : ℤ, (k + 1/2)^n + (l + 1/2)^n = t := 
sorry

end non_integer_sum_exists_l651_651743


namespace find_angle_A_max_area_triangle_l651_651617

-- Definitions
def vector_m (A : ℝ) : ℝ × ℝ := (Real.sin A, 1 / 2)
def vector_n (A : ℝ) : ℝ × ℝ := (3, Real.sin A + Real.sqrt 3 * Real.cos A)
def colinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 - u.2 * v.1 = 0

-- Problem statement
theorem find_angle_A (A : ℝ) (h_colinear : colinear (vector_m A) (vector_n A)) : A = π / 3 :=
sorry

def area_of_triangle (b c A : ℝ) : ℝ := 1 / 2 * b * c * Real.sin A

theorem max_area_triangle (BC A : ℝ) (h_BC : BC = 2) (h_A : A = π / 3) :
  ∃ max_S, max_S = Real.sqrt 3 ∧ ∀ S, S = area_of_triangle BC BC A → S ≤ max_S :=
sorry

end find_angle_A_max_area_triangle_l651_651617


namespace ab_less_than_pq_l651_651836

variables {A B C M P Q : Point}
variables {AB PQ : ℝ}

-- Geometric constraints
variables (is_midpoint : M = midpoint A B)
variables (is_isosceles : isosceles_triangle A B C)
variables (line_through_M : line_through M intersects_side P_and_extension Q)
variables (M_between_PQ : between M P Q)

theorem ab_less_than_pq (h : is_midpoint) (h1 : is_isosceles) (h2 : line_through_M) (h3 : M_between_PQ) : AB < PQ :=
sorry

end ab_less_than_pq_l651_651836


namespace determine_y_l651_651544

theorem determine_y (y : ℕ) : (8^5 + 8^5 + 2 * 8^5 = 2^y) → y = 17 := 
by {
  sorry
}

end determine_y_l651_651544


namespace sum_of_divisors_360_l651_651397

theorem sum_of_divisors_360 : 
  (∑ d in (finset.filter (λ x, 360 % x = 0) (finset.range (360 + 1))), d) = 1170 :=
sorry

end sum_of_divisors_360_l651_651397


namespace trihedral_no_regular_triangle_coplanar_vectors_l651_651693

section Part1

-- Definitions
variable (P A B C M N K : Type) 
variable (trihedralAngle : P → A → B → C → Prop)
variable (sectionPlane : P → A → B → C → Type)
variable (regularTriangle : M → N → K → Prop)
variable (liesOnRay : P → A → M → Prop)

-- Question: Is it true that in the cross-section of any trihedral angle by a plane, 
-- a regular triangle can be obtained?

theorem trihedral_no_regular_triangle (h1 : right_angle P A C)
                                      (h2 : right_angle P B C)
                                      (h3 : ∠ABP = 30)
                                      (h4 : liesOnRay P A M)
                                      (h5 : liesOnRay P B N)
                                      (h6 : liesOnRay P C K)
                                      (h7 : regularTriangle M N K) : False :=
sorry

end Part1

section Part2

-- Definitions
variable (a b c m n p : Type)
variable (vec_a : a)
variable (vec_b : b)
variable (vec_c : c)
variable (vec_m : -3 * vec_a + 4.5 * vec_b - 7 * vec_c)
variable (vec_n : vec_a - 2 * vec_b + 3 * vec_c)
variable (vec_p : -2 * vec_a + vec_b - 2 * vec_c)

-- Given:
variable (non_coplanar : ¬coplanar vec_a vec_b vec_c)

-- Question: Prove that the vectors \( \vec{m}, \vec{n}, \) and \( \vec{p} \) are coplanar.

theorem coplanar_vectors (h1 : 2 * vec_m + 4 * vec_n = vec_p) : coplanar vec_m vec_n vec_p :=
sorry

end Part2

end trihedral_no_regular_triangle_coplanar_vectors_l651_651693


namespace second_degree_polynomial_inequality_l651_651561

def P (u v w x : ℝ) : ℝ := u * x^2 + v * x + w

theorem second_degree_polynomial_inequality 
  (u v w : ℝ) (h : ∀ a : ℝ, 1 ≤ a → P u v w (a^2 + a) ≥ a * P u v w (a + 1)) :
  u > 0 ∧ w ≤ 4 * u :=
by
  sorry

end second_degree_polynomial_inequality_l651_651561


namespace readers_literary_works_l651_651671

theorem readers_literary_works (T SF B LW : ℕ) (hT: T = 150) (hSF: SF = 120) (hB: B = 60) :
  LW = 150 - (SF - B) :=
by
  rw [hT, hSF, hB]
  sorry

end readers_literary_works_l651_651671


namespace product_of_digits_base9_l651_651354

theorem product_of_digits_base9 (n : ℕ) (h : n = 7654) :
  let digits := [1, 1, 4, 3, 1] in digits.prod = 12 :=
by {
  sorry
}

end product_of_digits_base9_l651_651354


namespace solve_equation1_solve_equation2_l651_651789

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  2 * x^2 = 3 * (2 * x + 1)

-- Define the solution set for the first equation
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2

-- Prove that the solutions for the first equation are correct
theorem solve_equation1 (x : ℝ) : equation1 x ↔ solution1 x :=
by
  sorry

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  3 * x * (x + 2) = 4 * x + 8

-- Define the solution set for the second equation
def solution2 (x : ℝ) : Prop :=
  x = -2 ∨ x = 4 / 3

-- Prove that the solutions for the second equation are correct
theorem solve_equation2 (x : ℝ) : equation2 x ↔ solution2 x :=
by
  sorry

end solve_equation1_solve_equation2_l651_651789


namespace inscribed_circle_radius_l651_651787

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (θ : ℝ) (tangent : ℝ) :
    θ = π / 3 →
    R = 5 →
    tangent = (5 : ℝ) * (Real.sqrt 2 - 1) →
    r * (1 + Real.sqrt 2) = R →
    r = 5 * (Real.sqrt 2 - 1) := 
by sorry

end inscribed_circle_radius_l651_651787


namespace radioactive_half_life_l651_651117

theorem radioactive_half_life :
  ∀ (a : ℝ) (t : ℝ), a * (1 - 0.08) ^ t = a / 2 → t = log 0.5 / log 0.92 :=
by
  intros a t h
  sorry

end radioactive_half_life_l651_651117


namespace angle_equality_l651_651297

-- Define the geometrical setup (circle, points, tangents, midpoint)
variables {Circle : Type} [metric_space Circle] [normed_add_torsor ℝ Circle]
variables (O A K L P Q M : Circle)
variables (h_circle : is_circle O)  -- Hypothesis that O is the center of a circle

-- Conditions
variable (h_A : A ∈ extension_of_chord K L)
variable (h_tangents: is_tangent_from A P ∧ is_tangent_from A Q)
variable (h_midpoint : midpoint P Q M)

-- Theorem statement
theorem angle_equality :
  ∠M K O = ∠M L O :=
sorry  -- Proof not required

end angle_equality_l651_651297


namespace six_congruent_circles_area_l651_651788

theorem six_congruent_circles_area : 
  let r := 10 in
  let radius_C := 30 in
  let area_C := π * radius_C^2 in
  let area_six_circles := 6 * π * r^2 in
  let K := area_C - area_six_circles in
  ⌊K⌋ = 942 := 
by
  -- The proof would go here
  sorry

end six_congruent_circles_area_l651_651788


namespace sector_area_max_angle_l651_651612

theorem sector_area_max_angle (r : ℝ) (θ : ℝ) (h : 0 < r ∧ r < 10) 
  (H : 2 * r + r * θ = 20) : θ = 2 :=
by
  sorry

end sector_area_max_angle_l651_651612


namespace penguin_giraffe_ratio_l651_651668

theorem penguin_giraffe_ratio:
  ∀ (A : ℕ),
  (0.04 * A = 2) →
  (5 ≤ A) →
  let num_giraffes := 5 in
  let num_penguins := 0.20 * A in
  num_penguins / num_giraffes = 2 :=
by
  assume (A : ℕ) (h1 : 0.04 * A = 2) (h2 : 5 ≤ A),
  let num_giraffes := 5 in
  let num_penguins := 0.20 * A in
  sorry

end penguin_giraffe_ratio_l651_651668


namespace trapezium_area_calc_l651_651007

-- Define the lengths of the parallel sides and the distance between them
def side1 : ℝ := 10
def side2 : ℝ := 18
def distance : ℝ := 15

-- Define the area of the trapezium using the given formula
def trapeziumArea (a b h : ℝ) : ℝ := 1 / 2 * (a + b) * h

-- State the theorem that the area of the trapezium is 210 square centimeters
theorem trapezium_area_calc : trapeziumArea side1 side2 distance = 210 := by
  simp
  sorry

end trapezium_area_calc_l651_651007


namespace alyosha_cube_problem_l651_651469

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651469


namespace gen_term_b_gen_term_a_sum_S_n_l651_651595

-- Definitions based on conditions
def a_seq (n : ℕ) : ℕ → ℕ
| 0 := 2
| (n+1) := (2 * (n + 1) * (a_seq n)) / n

def b_seq (n : ℕ) : ℕ → ℕ
| 0 := 2
| (n+1) := 2 * b_seq n

-- Proofs of properties of the sequences
theorem gen_term_b (n : ℕ) : b_seq n = 2^n :=
sorry

theorem gen_term_a (n : ℕ) : a_seq n = n * 2^n :=
sorry

theorem sum_S_n (n : ℕ) : 
  ∑ i in finset.range n, a_seq i = (n-1) * 2^(n+1) + 2 :=
sorry

end gen_term_b_gen_term_a_sum_S_n_l651_651595


namespace bin_expected_value_l651_651428

theorem bin_expected_value (m : ℕ) (h : (21 - 4 * m) / (7 + m) = 1) : m = 3 := 
by {
  sorry
}

end bin_expected_value_l651_651428


namespace problem_statement_l651_651159

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem problem_statement (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_deriv : ∀ x, deriv f x = f' x)
  (h_f_neg1 : f (-1) = 0)
  (h_condition : ∀ x, x > 0 → x * f' x - f x > 0) :
  {x : ℝ | f x > 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (1 < x ∧ x < ∞)} :=
sorry

end problem_statement_l651_651159


namespace fifty_percent_of_x_l651_651653

variable (x : ℝ)

theorem fifty_percent_of_x (h : 0.40 * x = 160) : 0.50 * x = 200 :=
by
  sorry

end fifty_percent_of_x_l651_651653


namespace sum_first_2015_terms_l651_651142

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {OA OB OC : ℝ → ℝ}

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a n = a m + (n - m) * (a 1 - a 0)

axiom sum_of_arithmetic_sequence (a : ℕ → ℝ) : is_arithmetic_sequence a → ∀ n, S n = (n / 2) * (a 1 + a n)

axiom given_conditions (a : ℕ → ℝ) (OA OB OC : ℝ → ℝ) :
  ∀ n, a n = S n / n →
  (OA = a 3 • OB + a 2013 • OC) →
  collinear OA OB OC →
  a 3 + a 2013 = 1

theorem sum_first_2015_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (OA OB OC : ℝ → ℝ) [h : is_arithmetic_sequence a] :
  (OA = a 3 • OB + a 2013 • OC) →
  collinear OA OB OC →
  S 2015 = 2015 / 2 :=
by
  have h₁ : a 3 + a 2013 = 1 := given_conditions a OA OB OC
  have h₂ : a 1 + a 2015 = 1 := by sorry -- Use arithmetic sequence properties
  show S 2015 = 2015 / 2 by sorry -- Use sum of arithmetic sequence

end sum_first_2015_terms_l651_651142


namespace third_vertex_coordinates_l651_651843

/-- Given two vertices of an obtuse triangle at (8, 6) and (0, 0),
    and the third vertex lying on the negative y-axis,
    if the area of the triangle is 24 square units, then the coordinates
    of the third vertex are (0, -4.8). -/
theorem third_vertex_coordinates :
  ∃ (x y : ℝ), (x, y) = (0, -4.8) ∧
  (let base := real.sqrt ((8 - 0)^2 + (6 - 0)^2) in
   let height := 24 * 2 / base in
   base = 10 ∧ height = 4.8) :=
by 
  use [0, -4.8]
  split
  · rfl
  · dsimp
    split
    · rw [sub_zero, sub_zero, sqr, sqr, add, real.sqrt_eq_rpow]
      norm_num
    · norm_num

end third_vertex_coordinates_l651_651843


namespace angle_in_third_quadrant_l651_651662

theorem angle_in_third_quadrant (α : ℝ) (h : cos α < 0 ∧ tan α > 0) : 
  ∃ q : ℕ, q = 3 ∧ α ∈ set.Ioo (π:ℝ) (3 * π / 2) :=
by 
  sorry

end angle_in_third_quadrant_l651_651662


namespace problem_statement_l651_651210

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem problem_statement (x : ℝ) (hx : 3 ≤ x) :
  log_base 5 (log_base 2 (log_base 3 x)) = 0 -> x ^ (-1 / 3) = 1 / real.cbrt 9 :=
by
  intro h
  sorry

end problem_statement_l651_651210


namespace cos_alpha_plus_pi_over_6_l651_651128

theorem cos_alpha_plus_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hcosα : real.cos α = √3 / 3) :
  real.cos (α + π / 6) = (3 - ↑(real.sqrt 6)) / 6 :=
by sorry

end cos_alpha_plus_pi_over_6_l651_651128


namespace combined_length_of_trains_l651_651834

def length_of_train (speed_kmhr : ℕ) (time_sec : ℕ) : ℚ :=
  (speed_kmhr : ℚ) / 3600 * time_sec

theorem combined_length_of_trains :
  let L1 := length_of_train 300 33
  let L2 := length_of_train 250 44
  let L3 := length_of_train 350 28
  L1 + L2 + L3 = 8.52741 := by
  sorry

end combined_length_of_trains_l651_651834


namespace smallest_yellow_marbles_exists_l651_651755

noncomputable def smallest_yellow_marbles : ℕ :=
  ∃ n : ℕ, (n % 10 = 0) ∧ 
  (let blue_marbs := (n / 2) in
  let red_marbs := (n / 5) in
  let green_marbs := (2 * n / 5) in
  let yellow_marbs := n - (blue_marbs + red_marbs + green_marbs) in
  yellow_marbs = 2)

theorem smallest_yellow_marbles_exists : smallest_yellow_marbles :=
begin
  use 10,
  split,
  { refl },
  { simp,
    ring,
  },
  { ring_nf }
end

end smallest_yellow_marbles_exists_l651_651755


namespace find_huabei_number_l651_651239

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end find_huabei_number_l651_651239


namespace square_root_of_16_is_pm_4_l651_651822

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l651_651822


namespace difference_of_a_and_b_l651_651307

theorem difference_of_a_and_b :
  (∃ (a b : ℕ) (ha : a ∈ finset.range 27) (hb : b ∈ finset.range 27),
     a ≠ b ∧ a * b = finset.sum (finset.range 27) - a - b ∧ |a - b| = 6) :=
sorry

end difference_of_a_and_b_l651_651307


namespace ribbon_length_l651_651434

theorem ribbon_length (circumference height loops : ℝ) (h1 : circumference = 6) (h2 : height = 18) (h3 : loops = 6) :
  let vertical_rise := height / loops in
  let diagonal_length_per_loop := Real.sqrt (vertical_rise^2 + circumference^2) in
  let total_length := loops * diagonal_length_per_loop in
  total_length = 18 * Real.sqrt 5 := 
by
  simp only [h1, h2, h3]
  sorry

end ribbon_length_l651_651434


namespace angle_equality_l651_651678

noncomputable theory

open EuclideanGeometry

variables {A B C D E F G : Point}

-- Conditions
variable (h1 : ConvexQuadrilateral A B C D)
variable (h2 : ∃ θ : ℝ, ∠ A B C = θ ∧ ∠ A D C = θ)
variable (h3 : Collinear A C E)
variable (h4 : LineSegment B E A C G)
variable (h5 : LineSegment D G B F)

-- Goal
theorem angle_equality (h1 : ConvexQuadrilateral A B C D)
  (h2 : ∃ θ : ℝ, ∠ A B C = θ ∧ ∠ A D C = θ)
  (h3 : Collinear A C E)
  (h4 : LineSegment B E A C G)
  (h5 : LineSegment D G B F) : ∠ A B F = ∠ D A E :=
sorry

end angle_equality_l651_651678


namespace cindy_olaf_earnings_l651_651941
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l651_651941


namespace find_p_of_intersection_l651_651143

theorem find_p_of_intersection 
  (p : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / 8 + y^2 / 2 = 1) ↔ (y^2 = 2 * p * x)) 
  (h2 : ∀ A B : ℝ × ℝ, (A ≠ B) ∧ (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2)) : 
  p = 1 / 4 := 
by 
  sorry

end find_p_of_intersection_l651_651143


namespace general_formula_for_a_n_sum_of_b_n_l651_651727

noncomputable def S_n (n : ℕ) (a : ℕ → ℕ) : ℕ := ∑ i in Finset.range n, a i

theorem general_formula_for_a_n (n : ℕ) (a : ℕ → ℕ) :
  a 1 = 3 → 
  (∀ n, a (n + 1) = 2 * S_n n a + 3) →
  a n = 3^n :=
sorry

theorem sum_of_b_n (n : ℕ) (a b : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ n, a n = 3 ^ n) →
  (∀ n, b n = (2 * n - 1) * a n) →
  (∀ n, T n = ∑ i in Finset.range n, b i) →
  T n = 3 + (n-1) * 3^(n+1) :=
sorry

end general_formula_for_a_n_sum_of_b_n_l651_651727


namespace smallest_number_after_operations_l651_651829

theorem smallest_number_after_operations :
  ∀ (nums : List ℕ), (nums.length = 101 ∧ ∀ k, k < 101 → nums.get k = (k + 1) ^ 2) →
  (∃ m, m = 1 ∧ achievable_after_operations nums 100 m) :=
by
  intros nums h
  sorry

end smallest_number_after_operations_l651_651829


namespace tangent_through_P_tangent_through_M_l651_651636

def circle_eq (x y : ℝ) := (x - 1)^2 + (y - 2)^2 = 4

def point_P := (Real.sqrt 2 + 1, 2 - Real.sqrt 2)
def point_M := (3, 1)

theorem tangent_through_P :
  ∃ m b : ℝ, m = 1 ∧ b = 2*Real.sqrt 2 - 1 ∧ (∀ x y : ℝ, (y - (2 - Real.sqrt 2) = m * (x - (Real.sqrt 2 + 1))) ↔ (x - y + 1 - 2*Real.sqrt 2 = 0)) :=
sorry

theorem tangent_through_M :
  ∃ m1 m2  b1 b2 : ℝ, m1 = 0 ∧ b1 = 3 ∧ m2 = 3/4 ∧ b2 = -5/4 ∧ 
    (∀ x y : ℝ, (x = 3) → (x - 3 = 0)) ∧ 
    (∀ x y : ℝ, (y - 1 = m2 * (x - 3)) → (3*x - 4*y - 5 = 0)) ∧ 
    ∃ l : ℝ, l = 1 :=
sorry

end tangent_through_P_tangent_through_M_l651_651636


namespace cube_decomposition_l651_651473

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651473


namespace group_members_count_l651_651863

theorem group_members_count (n: ℕ) (total_paise: ℕ) (condition1: total_paise = 3249) :
  (n * n = total_paise) → n = 57 :=
by
  sorry

end group_members_count_l651_651863


namespace equal_parallelogram_faces_are_rhombuses_l651_651657

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l651_651657


namespace tan_sin_difference_l651_651932

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l651_651932


namespace jake_final_amount_l651_651254

def initial_amount : ℝ := 5000
def motorcycle_cost (initial: ℝ) : ℝ := 0.35 * initial
def remaining_after_motorcycle (initial: ℝ) (motorcycle: ℝ) : ℝ := initial - motorcycle

def concert_ticket_cost (remaining: ℝ) : ℝ := 0.25 * remaining
def remaining_after_concert (remaining: ℝ) (concert: ℝ) : ℝ := remaining - concert

def convert_to_euros (remaining: ℝ) (rate: ℝ) : ℝ := remaining / rate
def hotel_cost (remaining: ℝ) : ℝ := 0.15 * remaining
def remaining_after_hotel (remaining: ℝ) (hotel: ℝ) : ℝ := remaining - hotel

def invest_stock_market (remaining: ℝ) : ℝ := 0.40 * remaining
def remaining_after_investment (remaining: ℝ) (investment: ℝ) : ℝ := remaining - investment

def investment_loss (investment: ℝ) : ℝ := 0.20 * investment
def value_after_loss (investment: ℝ) (loss: ℝ) : ℝ := investment - loss

def total_remaining_euros (remaining: ℝ) (investment_value: ℝ) : ℝ := remaining + investment_value
def convert_to_usd (remaining_euros: ℝ) (rate: ℝ) : ℝ := remaining_euros * rate

theorem jake_final_amount :
  let initial := initial_amount in
  let motorcycle := motorcycle_cost initial in
  let remaining1 := remaining_after_motorcycle initial motorcycle in
  let concert := concert_ticket_cost remaining1 in
  let remaining2 := remaining_after_concert remaining1 concert in
  let rate1 := 1.1 in
  let euros := convert_to_euros remaining2 rate1 in
  let hotel := hotel_cost euros in
  let remaining3 := remaining_after_hotel euros hotel in
  let rate2 := 1.15 in
  let investment := invest_stock_market remaining3 in
  let remaining4 := remaining_after_investment remaining3 investment in
  let loss := investment_loss investment in
  let investment_value := value_after_loss investment loss in
  let final_euros := total_remaining_euros remaining4 investment_value in
  let usd := convert_to_usd final_euros (1 / rate2) in
  usd = 1506.82 :=
by
  sorry

end jake_final_amount_l651_651254


namespace total_performance_orders_l651_651670

-- Define the total number of different performance orders given the conditions.
theorem total_performance_orders : 
  ∃ (n : ℕ), 
  n = (
    let c21 := (nat.choose 2 1) in
    let c63 := (nat.choose 6 3) in
    let a44 := (fintype.card (fintype.perms (fin 4))) in
    let c62 := (nat.choose 6 2) in
    let a22 := (fintype.card (fintype.perms (fin 2))) in
    let a32 := (fintype.card (fintype.perms (fin 3))) in
    c21 * c63 * a44 + c62 * a22 * a32
  ) = 1140 :=
begin
  -- Placeholder for proof. We need to prove the calculations are correct.
  sorry
end

end total_performance_orders_l651_651670


namespace four_digit_number_l651_651064

theorem four_digit_number (grid : matrix (fin 6) (fin 6) nat) (regions : fin 9 → set (fin 6 × fin 6))
  (N : ℕ → ℕ)
  (h_reg1: ∀ i, (regions i).card = N i)
  (h_reg2: ∀ i, ∀ (x y : fin 6 × fin 6), x ∈ regions i → y ∈ regions i → (x ≠ y) → grid x ≠ grid y)
  (h_reg_filled: ∀ i, ∀ x ∈ regions i, 1 ≤ grid x ∧ grid x ≤ N i)
  (h_H : grid ⟨0, 0⟩ = 1) -- Assume positions are given here
  (h_B : grid ⟨0, 1⟩ = 2)
  (h_A : regions 0 = {(⟨0, 0⟩, (0, 1))}) -- Example regions
  (h_C : grid ⟨1, 1⟩ = 5)
  (h_D : grid ⟨1, 2⟩ = 2)
  : let ABCD := grid ⟨0, 0⟩ * 1000 + grid ⟨0, 1⟩ * 100 + grid ⟨1, 1⟩ * 10 + grid ⟨1, 2⟩ in
    ABCD = 4252 := 
by
  sorry
 
end four_digit_number_l651_651064


namespace polygon_with_5_sides_l651_651729

def is_cond_1 (b x : ℝ) := b ≤ x ∧ x ≤ 3 * b
def is_cond_2 (b y : ℝ) := b ≤ y ∧ y ≤ 3 * b
def is_cond_3 (b x y : ℝ) := x + y ≥ 2 * b
def is_cond_4 (b x y : ℝ) := x + 2 * b ≥ y
def is_cond_5 (b x y : ℝ) := y + 2 * b ≥ x
def is_cond_6 (b x y : ℝ) := x + y ≤ 4 * b

def T (b x y : ℝ) : Prop :=
  is_cond_1 b x ∧
  is_cond_2 b y ∧
  is_cond_3 b x y ∧
  is_cond_4 b x y ∧
  is_cond_5 b x y ∧
  is_cond_6 b x y

theorem polygon_with_5_sides (b : ℝ) (pos_b : 0 < b) :
  ∃ polygon, polygon.has_5_sides ∧
             ∀ (x y : ℝ), T b x y ↔ (x, y) ∈ polygon :=
sorry

end polygon_with_5_sides_l651_651729


namespace lines_divide_plane_l651_651672

theorem lines_divide_plane (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ parts : ℕ, parts = 50 :=
by 
  use 50
  sorry

end lines_divide_plane_l651_651672


namespace contains_C4_of_edges_bound_l651_651644

noncomputable def G_n (n : ℕ) := SimpleGraph
def e (G : SimpleGraph) : ℝ := G.edge_count

theorem contains_C4_of_edges_bound {n : ℕ} (Gn : SimpleGraph) :
  e(Gn) ≥ n * real.sqrt n / 2 + n / 4 → ∃ C4 : SimpleGraph, C4 = Cycle 4 := 
by sorry

end contains_C4_of_edges_bound_l651_651644


namespace longest_representation_l651_651119

theorem longest_representation (n : ℕ) (a : ℕ → ℕ) (k : ℕ) (h_sum : n = (list.range k).sum (λ i, a i))
  (h_decreasing : ∀ i < k - 1, a i > a (i + 1))
  (h_divisible : ∀ i < k - 1, a i % a (i + 1) = 0) :
  (n = 1992 → k = 6 ∧ a 0 = 1992 ∧ a 1 = 996 ∧ a 2 = 498 ∧ a 3 = 249 ∧ a 4 = 83 ∧ a 5 = 1) :=
begin
  sorry
end

end longest_representation_l651_651119


namespace subset_a_eq_1_l651_651725

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651725


namespace outer_perimeter_of_fence_l651_651346

def square_fence_perimeter (num_posts : ℕ) (width_post feet_between_posts : ℝ) : ℝ := 
  if num_posts = 24 ∧ width_post = (4 / 12) ∧ feet_between_posts = 5 then
    let num_corner_posts := 4 in
    let non_corner_posts := (num_posts - num_corner_posts) in
    let posts_per_side := non_corner_posts / 4 in
    let total_posts_per_side := posts_per_side + 2 in
    let gaps_between_posts := total_posts_per_side - 1 in
    let side_length := (gaps_between_posts * feet_between_posts) + (total_posts_per_side * width_post) in
    4 * side_length
  else 0

theorem outer_perimeter_of_fence :
  square_fence_perimeter 24 (4 / 12) 5 = 129 + 1 / 3 := 
  by
  sorry

end outer_perimeter_of_fence_l651_651346


namespace increase_by_percentage_proof_l651_651019

def initial_number : ℕ := 150
def percentage_increase : ℝ := 0.4
def final_number : ℕ := 210

theorem increase_by_percentage_proof :
  initial_number + (percentage_increase * initial_number) = final_number :=
by
  sorry

end increase_by_percentage_proof_l651_651019


namespace cube_decomposition_l651_651474

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651474


namespace presidency_meeting_ways_l651_651449

def numWaysToChooseRepresentatives : ℕ :=
  4 * (Nat.choose 5 3) * (5 * 5 * 5)

theorem presidency_meeting_ways : numWaysToChooseRepresentatives = 5000 :=
by
  unfold numWaysToChooseRepresentatives
  rw [Nat.choose_eq_factorial_div_factorial (le_refl 2)]
  sorry

end presidency_meeting_ways_l651_651449


namespace tangent_circle_radius_l651_651431

theorem tangent_circle_radius (r1 r2 d : ℝ) (h1 : r2 = 2) (h2 : d = 5) (tangent : abs (r1 - r2) = d ∨ r1 + r2 = d) :
  r1 = 3 ∨ r1 = 7 :=
by
  sorry

end tangent_circle_radius_l651_651431


namespace angle_HAE_equilateral_hexagon_l651_651926

theorem angle_HAE_equilateral_hexagon (A B C F G H I : Type)
  [EquilateralTriangle ABC] [RegularHexagon BCF GHI]
  (BC_common_side : is_common_side B C ABC BCF GHI) :
  ∠ HAE = 150 := 
sorry

end angle_HAE_equilateral_hexagon_l651_651926


namespace measure_of_arc_KT_measure_of_arc_KA_l651_651669

variables (Q : Type) [Field Q]
variables (angle_TAK angle_KAT : ℚ)

def measure_arc_KT := 2 * angle_TAK
def measure_arc_KA := 2 * angle_KAT

theorem measure_of_arc_KT (h_TAK : angle_TAK = 60) : measure_arc_KT Q angle_TAK = 120 := by
  sorry

theorem measure_of_arc_KA (h_KAT : angle_KAT = 30) : measure_arc_KA Q angle_KAT = 60 := by
  sorry

end measure_of_arc_KT_measure_of_arc_KA_l651_651669


namespace cube_decomposition_l651_651475

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651475


namespace measure_angle_CED_l651_651981

theorem measure_angle_CED (t : ℝ) (h : t = 40) : x = 30 :=
  let BAC := t
  let BCA := 180 - 3 * BAC
  let CDE := 90
  let DCE := BCA
  let CED := 180 - CDE - DCE
  have step1 : BCA = 180 - 3 * BAC := by sorry
  have step2 : DCE = BCA := by sorry
  have step3 : CED = 180 - CDE - DCE := by sorry
  have step4 : x = 3 * BAC - 90 := by sorry
  show x = 30 := by
    rw [← step4, h]
    sorry

end measure_angle_CED_l651_651981


namespace ramesh_profit_percentage_l651_651779

noncomputable def profit_percentage (cost price transport installation selling: ℝ) : ℝ :=
  ((selling - (cost + transport + installation)) / (cost + transport + installation)) * 100

theorem ramesh_profit_percentage : 
  let P := 15625
  let cost := 12500
  let transport := 125
  let installation := 250
  let selling_price := 19200
  profit_percentage cost transport installation selling_price ≈ 49.13 :=
by
  -- Definitions based on problem description
  let total_cost := cost + transport + installation
  let profit := selling_price - total_cost
  have P_calc : P = 12500 / 0.80 := by sorry
  have total_cost_calc : total_cost = 12500 + 125 + 250 := by sorry
  have profit_calc : profit = 19200 - total_cost := by sorry
  have profit_percentage_calc : profit_percentage cost transport installation selling_price ≈ 49.13 := by sorry
  exact (profit_percentage_calc : profit_percentage cost transport installation selling_price ≈ 49.13)

end ramesh_profit_percentage_l651_651779


namespace total_time_round_trip_l651_651420

def boat_speed : ℝ := 9 -- km/h
def stream_speed : ℝ := 1.5 -- km/h
def distance : ℝ := 105 -- km

def speed_downstream (v_boat v_stream : ℝ) : ℝ := v_boat + v_stream
def speed_upstream (v_boat v_stream : ℝ) : ℝ := v_boat - v_stream

def time_taken (d v : ℝ) : ℝ := d / v

theorem total_time_round_trip :
  time_taken distance (speed_downstream boat_speed stream_speed) +
  time_taken distance (speed_upstream boat_speed stream_speed) = 24 := by
  sorry

end total_time_round_trip_l651_651420


namespace train_length_is_120_l651_651458

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let total_distance := speed_ms * time_s
  total_distance - bridge_length_m

theorem train_length_is_120 :
  length_of_train 70 13.884603517432893 150 = 120 :=
by
  sorry

end train_length_is_120_l651_651458


namespace distinct_connected_stamps_l651_651980

theorem distinct_connected_stamps (n : ℕ) : 
  ∃ d : ℕ → ℝ, 
    d (n+1) = 1 / 4 * (1 + Real.sqrt 2)^(n + 3) + 1 / 4 * (1 - Real.sqrt 2)^(n + 3) - 2 * n - 7 / 2 :=
sorry

end distinct_connected_stamps_l651_651980


namespace coeff_of_x2_in_expansion_l651_651543

theorem coeff_of_x2_in_expansion :
  let T_r (r : ℕ) := (5.choose r) * (2 : ℤ)^(5-r) * (-1) ^ r * (x : ℤ)^(5 - (3 / 2) * r)
  (coeff_5 := ∑ r in finset.range 6, T_r r)
  ((2 * x - (1 / real.sqrt x)) ^ 5).coeff 2 = 80 :=
by
  sorry

end coeff_of_x2_in_expansion_l651_651543


namespace cube_cut_problem_l651_651494

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651494


namespace johns_remaining_money_l651_651267

theorem johns_remaining_money :
  ∀ (saved : ℕ) (ticket_cost : ℕ), saved = 2925 → ticket_cost = 1200 → saved - ticket_cost = 1725 := 
begin
  intros saved ticket_cost hs ht,
  rw [hs, ht],
  norm_num,
end

end johns_remaining_money_l651_651267


namespace primary_objective_of_group_relocation_l651_651339

variable (conditions : Type*)
variable (industrial_relocation : conditions → Prop)
variable (regional_cooperation : conditions → Prop)
variable (simultaneous_transformation : conditions → Prop)
variable (optimized_chain_layout : conditions → Prop)
variable (environmental_protection : conditions → Prop)
variable (has_footwear_industry : conditions → Prop)
variable (enterprise_cluster : conditions → Prop)
variable (facilitates_connections : conditions → Prop)
variable (reduces_costs : conditions → Prop)
variable (improves_efficiency_profitability : conditions → Prop)
variable (achieves_economies_scale : conditions → Prop)

theorem primary_objective_of_group_relocation
  (h_industrial_relocation : industrial_relocation conditions)
  (h_regional_cooperation : regional_cooperation conditions)
  (h_simultaneous_transformation : simultaneous_transformation conditions)
  (h_optimized_chain_layout : optimized_chain_layout conditions)
  (h_environmental_protection : environmental_protection conditions)
  (h_has_footwear_industry : has_footwear_industry conditions)
  (h_enterprise_cluster : enterprise_cluster conditions)
  (h_facilitates_connections : facilitates_connections conditions)
  (h_reduces_costs : reduces_costs conditions)
  (h_improves_efficiency_profitability : improves_efficiency_profitability conditions) :
  achieves_economies_scale conditions := 
sorry

end primary_objective_of_group_relocation_l651_651339


namespace cube_cut_problem_l651_651495

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651495


namespace pastries_sold_l651_651427

theorem pastries_sold
  (P : ℕ)
  (daily_avg_sales : ℕ := 20 * 2 + 10 * 4)
  (bread_today_sales : ℕ := 25 * 4)
  (diff : ℕ := 48)
  (today_sales := 2 * P + bread_today_sales)
  (h : today_sales - daily_avg_sales = diff) :
  P = 14 :=
by {
  unfold daily_avg_sales at h,
  unfold bread_today_sales at today_sales h,
  unfold diff at h,
  unfold today_sales at h,
  sorry
}

end pastries_sold_l651_651427


namespace janina_cover_expenses_l651_651262

theorem janina_cover_expenses : 
  ∀ (rent supplies price_per_pancake : ℕ), 
    rent = 30 → 
    supplies = 12 → 
    price_per_pancake = 2 → 
    (rent + supplies) / price_per_pancake = 21 := 
by 
  intros rent supplies price_per_pancake h_rent h_supplies h_price_per_pancake 
  rw [h_rent, h_supplies, h_price_per_pancake]
  sorry

end janina_cover_expenses_l651_651262


namespace sum_of_divisors_360_l651_651395

-- Define the function to calculate the sum of divisors
def sum_of_divisors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

-- Statement of the problem
theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 :=
by
  sorry

end sum_of_divisors_360_l651_651395


namespace Mahdi_find_all_sets_l651_651293

-- Define the main entities and the set operations
variable {α : Type} (sets : Finset (Finset α)) (n : ℕ)
variable (A B : Finset α)
variable (find_set_info : ∀ (A B : Finset α), {I : Finset α // I = A ∩ B} × {U : Finset α // U = A ∪ B})

-- The main theorem as a Lean statement
theorem Mahdi_find_all_sets (h_card_sets : sets.card = 100)
    (h_find_set_info : ∀ A B ∈ sets, A ≠ B → find_set_info A B) :
    ∃ steps : ℕ, steps = 100 ∧ ∀ S ∈ sets, S ∈ sets := sorry

end Mahdi_find_all_sets_l651_651293


namespace find_lambda_l651_651151

-- Definitions
variables {A B C D : Type} [add_comm_group A] [vector_space ℝ A]
variables (DA DB CB : A)
variables (collinear : ∀ {u v w : A}, ∃ a b c : ℝ, a • u + b • v + c • w = 0)

-- The main statement to prove
theorem find_lambda (λ : ℝ) (h : DA = 2 * λ • DB + 3 • CB) (h_collinear: collinear A B C) :
  λ = 1 / 2 := 
sorry

end find_lambda_l651_651151


namespace cartesian_equation_of_C2_and_type_max_min_AB_l651_651682

-- Definitions for curve C_1 in Cartesian coordinates
def curve_C1 (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Definitions for curve C_2 in polar coordinates
def curve_C2_polar (θ : ℝ) : ℝ :=
  8 * Real.cos (θ - Real.pi / 3)

-- Conversion from polar to Cartesian
noncomputable def curve_C2_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 4 * Real.sqrt 3 * y

-- Theorem statement 1: Cartesian equation of C_2 and type of curve.
theorem cartesian_equation_of_C2_and_type :
  ∃ (x y : ℝ), curve_C2_cartesian x y ∧ (x^2 + y^2 = 4 * x + 4 * Real.sqrt 3 * y) :=
sorry

-- Theorem statement 2: Maximum and minimum values of |AB|
theorem max_min_AB (α : ℝ) :
  ∃ (t₁ t₂ : ℝ), (curve_C1 t₁ α = curve_C1 t₂ α ∧
    Real.abs (t₁ - t₂) = Real.sqrt (12 * (Real.sin α)^2 + 52)) ∧
    (Real.sin α = 0 → Real.abs (t₁ - t₂) = 2 * Real.sqrt 13) ∧
    (Real.sin α = 1 ∨ Real.sin α = -1 → Real.abs (t₁ - t₂) = 8) :=
sorry

end cartesian_equation_of_C2_and_type_max_min_AB_l651_651682


namespace cricket_player_average_increase_l651_651033

theorem cricket_player_average_increase (total_innings initial_innings next_run : ℕ) (initial_average desired_increase : ℕ) 
(h1 : initial_innings = 10) (h2 : initial_average = 32) (h3 : next_run = 76) : desired_increase = 4 :=
by
  sorry

end cricket_player_average_increase_l651_651033


namespace sum_cubic_polynomial_l651_651034

noncomputable def q : ℤ → ℤ := sorry  -- We use a placeholder definition for q

theorem sum_cubic_polynomial :
  q 3 = 2 ∧ q 8 = 22 ∧ q 12 = 10 ∧ q 17 = 32 →
  (q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 + q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18) = 272 :=
sorry

end sum_cubic_polynomial_l651_651034


namespace element_subset_a_l651_651708

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l651_651708


namespace mathematics_trick_l651_651928

theorem mathematics_trick (x y : ℕ) (h : x + y = 99999) : 
    x + y + (99999 - x) = 199998 := 
by 
  calc
    x + y + (99999 - x) = y + 99999 : by rw [add_assoc, add_sub_cancel']
    ... = 99999 + y : by rw [add_comm]
    ... = 199998 : by rw [add_comm _ 99999, Nat.add_sub_of_le h]

end mathematics_trick_l651_651928


namespace even_function_f_l651_651614

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - x^2 else -(-x)^3 - (-x)^2

theorem even_function_f (x : ℝ) (h : ∀ x ≤ 0, f x = x^3 - x^2) :
  (∀ x, f x = f (-x)) ∧ (∀ x > 0, f x = -x^3 - x^2) :=
by
  sorry

end even_function_f_l651_651614


namespace find_x_geometric_sequence_l651_651558

theorem find_x_geometric_sequence :
  ∃ x : ℝ, x ≠ 0 ∧ 
    let frac_part := x - (⌊x⌋ : ℝ) in
    frac_part ≠ 0 ∧ ⌊x⌋ ≠ 0 ∧
    frac_part * x = (⌊x⌋ : ℝ)^2 ∧
    x = (1 + Real.sqrt 5) / 2 :=
sorry

end find_x_geometric_sequence_l651_651558


namespace no_asymptote_intersection_l651_651974

-- Define the rational function
def rational_function (x : ℝ) : ℝ :=
  (x^2 - 9*x + 20) / (x^2 - 9*x + 21)

-- Define the condition for horizontal asymptote y = 1
def horizontal_asymptote : Prop :=
  ∀ (x : ℝ), abs(x) > 1 → abs(rational_function(x) - 1) < ε

-- Prove that there is no intersection of vertical and horizontal asymptotes
theorem no_asymptote_intersection :
  ¬∃ (x : ℝ), (rational_function x = 1) ∧ vertical_asymptote x :=
sorry

end no_asymptote_intersection_l651_651974


namespace expression_meaning_l651_651353

variable (m n : ℤ) -- Assuming m and n are integers for the context.

theorem expression_meaning : 2 * (m - n) = 2 * (m - n) := 
by
  -- It simply follows from the definition of the expression
  sorry

end expression_meaning_l651_651353


namespace solve_AlyoshaCube_l651_651508

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651508


namespace find_n_l651_651485

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651485


namespace linda_color_choices_l651_651679

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem linda_color_choices : combination 8 3 = 56 :=
  by sorry

end linda_color_choices_l651_651679


namespace const_expr_l651_651225

open EuclideanGeometry

-- Definitions of the geometric objects and points
variables (a b : Line) (D C A B A1 B1 : Point)

-- Conditions outlined from the problem statement
axiom LinesIntersectAtD : ∃ (p : Point), p ∈ a ∧ p ∈ b
axiom PointCOnPlane : ∃ (p : Point), p ≠ D ∧ p ∉ a ∧ p ∉ b
axiom LineThroughC : ∃ l, l ∋ C ∧ l ∋ A ∧ l ∋ B
axiom PerpendicularsFromAAndB : ∀ (p : Point), p ∈ LineThrough C D → Perpendicular A p ∧ Perpendicular B p

-- The mathematically equivalent proof statement
theorem const_expr (h : (LinesIntersectAtD) ∧ (PointCOnPlane) ∧ (LineThroughC) ∧ (PerpendicularsFromAAndB)) :
  ∀ (A A1 B B1 : Point), 
    let AA1 := distance A A1,
    let BB1 := distance B B1 in
  (AA1 ≠ 0 ∧ BB1 ≠ 0) → 
  (∀ (α β : ℝ), α = 1 / AA1 ∧ β = 1 / BB1 → (α - β) = constant_function) :=
sorry

end const_expr_l651_651225


namespace problem_l651_651989

theorem problem 
  (a b A B : ℝ)
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
by sorry

end problem_l651_651989


namespace cube_decomposition_l651_651477

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651477


namespace president_and_committee_l651_651227

theorem president_and_committee :
  ∃ n : ℕ, n = 10 * (Nat.choose 9 3) :=
begin
  use 840,
  sorry
end

end president_and_committee_l651_651227


namespace solve_AlyoshaCube_l651_651506

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651506


namespace range_of_m_l651_651184

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ (x y : ℝ), 3 < m ∧ m < 7/3 → (x^2 / (m + 3) + y^2 / (7 * m - 3) = 1)
def q (m : ℝ) : Prop := ∀ (x : ℝ), (5 - 2 * m) > 1 → (5 - 2 * m)^x

-- Main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≤ -3 ∨ (3/7 ≤ m ∧ m < 2)) :=
begin
  sorry
end

end range_of_m_l651_651184


namespace remainder_17_pow_63_mod_7_l651_651376

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651376


namespace janina_must_sell_21_pancakes_l651_651258

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l651_651258


namespace distance_swim_against_current_l651_651051

-- Definitions based on problem conditions
def swimmer_speed_still_water : ℝ := 4 -- km/h
def water_current_speed : ℝ := 1 -- km/h
def time_swimming_against_current : ℝ := 2 -- hours

-- Calculation of effective speed against the current
def effective_speed_against_current : ℝ :=
  swimmer_speed_still_water - water_current_speed

-- Proof statement
theorem distance_swim_against_current :
  effective_speed_against_current * time_swimming_against_current = 6 :=
by
  -- By substituting values from the problem,
  -- effective_speed_against_current * time_swimming_against_current = 3 * 2
  -- which equals 6.
  sorry

end distance_swim_against_current_l651_651051


namespace pen_cost_l651_651264

theorem pen_cost (P : ℝ)
  (book_cost : ℝ := 25)
  (ruler_cost : ℝ := 1)
  (total_paid : ℝ := 50)
  (change_received : ℝ := 20)
  (total_spent : ℝ := total_paid - change_received) : P = 4 :=
by
  have h : 25 + P + 1 = total_spent := sorry
  linarith

end pen_cost_l651_651264


namespace discriminant_nonnegative_l651_651536

theorem discriminant_nonnegative (x : ℤ) (h : x^2 * (25 - 24 * x^2) ≥ 0) : x = 0 ∨ x = 1 ∨ x = -1 :=
by sorry

end discriminant_nonnegative_l651_651536


namespace geometric_sequence_first_term_l651_651092

theorem geometric_sequence_first_term (a b c : ℕ) 
    (h1 : 16 = a * (2^3)) 
    (h2 : 32 = a * (2^4)) : 
    a = 2 := 
sorry

end geometric_sequence_first_term_l651_651092


namespace lambda_range_l651_651180

open Real

theorem lambda_range (x : ℝ) (n : ℕ) (h : 1 ≤ n) (hx : x ≤ λ) :
  (x^2 + 1/2*x - (1/2)^n ≥ 0) → λ ≤ -1 := sorry

end lambda_range_l651_651180


namespace michael_dinner_cost_l651_651291

def dinner_cost (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ := total / (1 + tax_rate + tip_rate)

theorem michael_dinner_cost (T : ℝ) (θ τ : ℝ) (hT : T = 40) (hθ : θ = 0.096) (hτ : τ = 0.18) :
  dinner_cost T θ τ = 31 := by
  sorry

end michael_dinner_cost_l651_651291


namespace find_n_l651_651484

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651484


namespace boy_speed_l651_651026

theorem boy_speed (d : ℝ) (v₁ v₂ : ℝ) (t₁ t₂ l e : ℝ) :
  d = 2 ∧ v₂ = 8 ∧ l = 7 / 60 ∧ e = 8 / 60 ∧ t₁ = d / v₁ ∧ t₂ = d / v₂ ∧ t₁ - t₂ = l + e → v₁ = 4 :=
by
  sorry

end boy_speed_l651_651026


namespace cylinder_radius_in_cone_l651_651914

theorem cylinder_radius_in_cone (d_cone h_cone d_cylinder r : ℝ) 
  (H1 : d_cone = 12) 
  (H2 : h_cone = 18) 
  (H3 : d_cylinder = 2 * r) 
  (H4 : (h_cone - d_cylinder) / r = 3) : 
  r = 18 / 5 :=
by 
  have H5 : h_cone - 2 * r = 3 * r := by linarith [H3, H4]
  have H6 : h_cone = 5 * r := by linarith [H5]
  linarith [H1, H2, H6] 

end cylinder_radius_in_cone_l651_651914


namespace intersection_of_A_and_B_l651_651186

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = x + 1 }

theorem intersection_of_A_and_B :
  A ∩ B = {2, 3, 4} :=
sorry

end intersection_of_A_and_B_l651_651186


namespace kramer_boxes_per_minute_l651_651269

theorem kramer_boxes_per_minute :
  ∀ (boxes_per_case cases cases_per_time : ℕ), 
  (boxes_per_case = 5) → (cases = 240) → (cases_per_time = 2) →      
  (boxes_per_case * cases) / (cases_per_time * 60) = 10 := 
by 
  intros boxes_per_case cases cases_per_time h1 h2 h3
  simp [h1, h2, h3]
  sorry

end kramer_boxes_per_minute_l651_651269


namespace least_value_of_g_l651_651083

noncomputable def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 1

theorem least_value_of_g : ∃ x : ℝ, ∀ y : ℝ, g y ≥ g x ∧ g x = -2 := by
  sorry

end least_value_of_g_l651_651083


namespace trajectory_of_m1_minimum_area_passes_through_fixed_point_l651_651634

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the trajectory of midpoint M1
def trajectory_m1 (x y : ℝ) : Prop := y^2 = 2 * (x - 1)

-- Define the focus point
def focus : (ℝ × ℝ) := (1, 0)

-- Define the fixed point
def fixed_point : (ℝ × ℝ) := (3, 0)

theorem trajectory_of_m1 : ∃ x y, parabola x y 
  ∧ trajectory_m1 x y := 
sorry

theorem minimum_area : ∃ (M1 M2 : ℝ × ℝ), 
  let dist := λ P Q, real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) in
  dist (focus) M1 * dist (focus) M2 / 2 = 4 :=
sorry

theorem passes_through_fixed_point : 
  ∀ M1 M2 : ℝ × ℝ, 
  let line_eq (p1 p2 : ℝ × ℝ) : Prop := 
    (p1.2 - p2.2) * (focus.1 - p1.1) = (focus.2 - p1.2) * (p1.1 - p2.1) in
  line_eq M1 M2 →
  ∃ k : ℝ, k ≠ 1 ∧ fixed_point ∈ ({(x, y) | y * k^2 + (x - 3) * k + y = 0}) :=
sorry

end trajectory_of_m1_minimum_area_passes_through_fixed_point_l651_651634


namespace seventeen_power_sixty_three_mod_seven_l651_651364

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651364


namespace find_xy_sum_diff_solve_linear_system_custom_operation_l651_651012

/-- Understanding application -/
theorem find_xy_sum_diff :
  (∀ (x y : ℝ), 2 * x + 3 * y = 6 ∧ 3 * x + 2 * y = 4 → x + y = 2 ∧ x - y = -2) :=
by
  intros x y h
  cases' h with h1 h2
  -- Assuming h1: 2 * x + 3 * y = 6 and h2: 3 * x + 2 * y = 4
  sorry

/-- Solve the equation system -/
theorem solve_linear_system :
  (∀ (x y : ℝ), 2024 * x + 2025 * y = 2023 ∧ 2022 * x + 2023 * y = 2021 → x = 2 ∧ y = -1) :=
by
  intros x y h
  cases' h with h1 h2
  -- Assuming h1: 2024 * x + 2025 * y = 2023 and h2: 2022 * x + 2023 * y = 2021
  sorry

/-- Extension and enhancement -/
theorem custom_operation :
  (∀ (a b c : ℝ), (2 * a + 4 * b + c = 15) ∧ (3 * a + 7 * b + c = 27) → a + b + c = 3) →
  (∀ (a b c : ℝ), 1 * a + 1 * b + c = 3) :=
by 
  intros h a b c h1
  -- Assuming h1: (2 * a + 4 * b + c = 15) and h2: (3 * a + 7 * b + c = 27)
  sorry

end find_xy_sum_diff_solve_linear_system_custom_operation_l651_651012


namespace largest_k_inequality_l651_651111

noncomputable def k : ℚ := 39 / 2

theorem largest_k_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b + c)^3 ≥ (5 / 2) * (a^3 + b^3 + c^3) + k * a * b * c := 
sorry

end largest_k_inequality_l651_651111


namespace keiko_speed_correct_l651_651268

noncomputable def keiko_speed_proof : ℝ :=
let inner_radius := 15
let track_width := 12
let time_difference := 72
let outer_radius := inner_radius + track_width
let C1 := 2 * (Real.pi * inner_radius)
let C2 := 2 * (Real.pi * outer_radius)
let extra_distance := C2 - C1
let keiko_speed := extra_distance / time_difference
in keiko_speed

theorem keiko_speed_correct : keiko_speed_proof = Real.pi / 3 :=
by
  sorry

end keiko_speed_correct_l651_651268


namespace correct_statement_d_l651_651001

theorem correct_statement_d (x : ℝ) : 2 * (x + 1) = x + 7 → x = 5 :=
by
  sorry

end correct_statement_d_l651_651001


namespace adam_money_left_l651_651462

def initial_money : ℝ := 5
def spent_on_game : ℝ := 2
def spent_on_snack : ℝ := 1.5
def money_from_mom : ℝ := 3
def money_from_grandfather : ℝ := 2
def allowance : ℝ := 5
def cost_of_toy : ℝ := 0.75

theorem adam_money_left : 
  initial_money 
  - spent_on_game 
  - spent_on_snack 
  + money_from_mom 
  + money_from_grandfather 
  + allowance 
  - cost_of_toy 
  = 10.75 := 
begin
  sorry
end

end adam_money_left_l651_651462


namespace min_sum_factors_of_9_fac_l651_651798

theorem min_sum_factors_of_9_fac : 
  ∃ x y z w : ℕ, x * y * z * w = 9! ∧ x + y + z + w = 69 := 
sorry

end min_sum_factors_of_9_fac_l651_651798


namespace area_of_triangle_ABC_l651_651246

variables (A B C D : ℝ)
variables (h_ABCD : true) (h_parallel : true) (h_CDtoAB : CD = 3 * AB) (h_areaTrap : (1/2) * (AB + CD) * height = 30)
variables (height : ℝ)

noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * AB * height

theorem area_of_triangle_ABC :
  area_triangle_ABC = 7.5 :=
by 
  sorry

end area_of_triangle_ABC_l651_651246


namespace distance_between_centers_of_tangent_circles_l651_651189

theorem distance_between_centers_of_tangent_circles
  (R r d : ℝ) (h1 : R = 8) (h2 : r = 3) (h3 : d = R + r) : d = 11 :=
by
  -- Insert proof here
  sorry

end distance_between_centers_of_tangent_circles_l651_651189


namespace expression_1_eq_neg_5_expression_2_eq_6_div_5_l651_651129

variables {α : Type*} [Real α] (α : α)

-- Given condition
def tan_alpha (α : α) := 2

-- Equivalent proof problems

theorem expression_1_eq_neg_5 (h : tan α = 2) :
  (2 * sin α + cos α) / (sin α - 3 * cos α) = -5 :=
sorry

theorem expression_2_eq_6_div_5 (h : tan α = 2) :
  sin α * (sin α + cos α) = 6 / 5 :=
sorry

end expression_1_eq_neg_5_expression_2_eq_6_div_5_l651_651129


namespace subset_solution_l651_651716

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l651_651716


namespace polynomial_degree_condition_l651_651417

noncomputable def polynomial_solutions (α β : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f(α * x + β) = f(x)

theorem polynomial_degree_condition {n : ℕ} {α : ℝ} (hα : α^n = 1) :
  ∃ f : ℝ → ℝ, (∃ a_0 : ℝ, a_0 ≠ 0) ∧
  (∀ (a : ℕ) (x : ℝ), f x = ∑ i in range (n + 1), (λ k, (finset.card (finset.filter (λ (j : ℕ), k = j) (finset.range (n + 1)))) * x ^ (n - k))) ∧ 
  polynomial_solutions α β f :=
sorry

end polynomial_degree_condition_l651_651417


namespace convert_mps_to_kmph_l651_651867

theorem convert_mps_to_kmph (v_mps : ℝ) (conversion_factor : ℝ) : v_mps = 22 → conversion_factor = 3.6 → v_mps * conversion_factor = 79.2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end convert_mps_to_kmph_l651_651867


namespace cake_slices_l651_651888

theorem cake_slices (S : ℕ) (h : 347 * S = 6 * 375 + 526) : S = 8 :=
sorry

end cake_slices_l651_651888


namespace rearrangement_time_l651_651266

theorem rearrangement_time (letters_in_name : ℕ) (rate : ℕ) : letters_in_name = 6 → rate = 15 → (letters_in_name.factorial / rate) / 60 = 0.8 :=
by
  intros h1 h2
  have h_factorial := Nat.factorial_eq h1
  rw [h_factorial]
  have h_rate := Nat.cast_eq_of_eq h2
  rw [h_rate]
  sorry

end rearrangement_time_l651_651266


namespace transform_equation_l651_651920

theorem transform_equation (x : ℝ) : (5 = 3 * x - 2) → (5 + 2 = 3 * x) :=
begin
  intro h,
  linarith,
end

end transform_equation_l651_651920


namespace calculate_dividend_l651_651296

theorem calculate_dividend : 
  ∃ dividend : ℤ, 
    let divisor := 9 in 
    let quotient := 9 in 
    let remainder := 2 in 
    dividend = divisor * quotient + remainder :=
begin
  use 83,
  let divisor := 9,
  let quotient := 9,
  let remainder := 2,
  show 83 = divisor * quotient + remainder,
  calc
    83 = 9 * 9 + 2 : by simp [divisor, quotient, remainder]
end

end calculate_dividend_l651_651296


namespace no_finite_prime_set_l651_651546

open Nat

-- Definition for sum of squares from 2 to n
def sum_of_squares (n : ℕ) : ℕ :=
  if n < 2 then 0 else (finset.range n).filter (λ m, m >= 2).sum (λ m, m^2)

-- Theorem statement
theorem no_finite_prime_set (S : finset ℕ) (h : ∀ p ∈ S, nat.prime p) :
  ∃ n ≥ 2, (∀ p ∈ S, ¬ p ∣ sum_of_squares n) :=
sorry

end no_finite_prime_set_l651_651546


namespace min_value_on_curve_l651_651615

theorem min_value_on_curve
  (x y : ℝ)
  (h : x^2 / 9 + y^2 = 1) :
  ∃ θ ∈ Icc (0 : ℝ) (2 * Real.pi),
  ∃ φ : ℝ, x + 2 * Real.sqrt 3 * y = -Real.sqrt 21 :=
by sorry

end min_value_on_curve_l651_651615


namespace sum_of_odd_integers_between_400_600_l651_651853

theorem sum_of_odd_integers_between_400_600 :
  ∑ k in (finset.Icc 401 599).filter (λ x, x % 2 = 1), k = 50000 :=
by sorry

end sum_of_odd_integers_between_400_600_l651_651853


namespace polynomial_problem_l651_651747

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem polynomial_problem (f_nonzero : ∀ x, f x ≠ 0) 
  (h1 : ∀ x, f (g x) = f x * g x)
  (h2 : g 3 = 10)
  (h3 : ∃ a b, g x = a * x + b) :
  g x = 2 * x + 4 :=
sorry

end polynomial_problem_l651_651747


namespace range_of_a_l651_651811

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - 1 < 3) ∧ (x - a < 0) → (x < a)) → (a ≤ 2) :=
by
  intro h
  sorry

end range_of_a_l651_651811


namespace remainder_17_pow_63_mod_7_l651_651358

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651358


namespace parallel_AC_A1C1_l651_651726

theorem parallel_AC_A1C1
  (A B C H A1 C1 : Point) 
  (triangleABC : Triangle A B C)
  (altitudeBH : Altitude B H triangleABC)
  (intersectionA1 : A1 ∈ Segment B A ∧ H ∈ Line A1 A)
  (intersectionC1 : C1 ∈ Segment B C ∧ H ∈ Line C1 C) :
  Parallel Line A C Line A1 C1 := 
sorry

end parallel_AC_A1C1_l651_651726


namespace right_triangle_inequality_l651_651226

theorem right_triangle_inequality (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : b > a) (h3 : b / a < 2) :
  a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) > 4 / 9 :=
by
  sorry

end right_triangle_inequality_l651_651226


namespace tangency_parallel_equiv_l651_651590

variables (A B C D : Point) (circle1 : Circle)

-- Defining the convex quadrilateral and tangent line conditions
def convex_quadrilateral (A B C D : Point) : Prop := 
  -- Add the precise geometric definition here
  sorry

def tangent_line_circle (line : Line) (circle : Circle) : Prop := 
  -- Definition for tangency of a line to a circle
  sorry

-- Given conditions
variables (h1 : convex_quadrilateral A B C D)
          (h2 : tangent_line_circle (Line.mk C D) (circle1))
          (circle2 : Circle)
          (h3 : diameter circle2 = Segment.mk C D)
          (h4 : diameter circle1 = Segment.mk A B)

-- Goal to prove the equivalence
theorem tangency_parallel_equiv :
  (parallel (Line.mk B C) (Line.mk A D)) ↔ tangent_line_circle (Line.mk A B) (circle2) :=
sorry

end tangency_parallel_equiv_l651_651590


namespace alyosha_cube_cut_l651_651504

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651504


namespace equal_parallelogram_faces_are_rhombuses_l651_651656

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l651_651656


namespace sqrt_16_eq_pm_4_l651_651818

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l651_651818


namespace part_1_part_2_l651_651197

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 - 1

theorem part_1 : 
  (∀ (m n : ℝ), (n = f m 1) ∧ (f' m 1 = 1) → m = 2 / 3 ∧ n = -1 / 3) :=
sorry

theorem part_2 : 
  (∀ (m : ℝ), m = 2 / 3 →
   ∃ k : ℕ, k = 2008 ∧ ∀ x ∈ Icc (-1 : ℝ) 3, f m x ≤ (k - 1993)) :=
sorry

end part_1_part_2_l651_651197


namespace number_of_rows_l651_651937

-- Definitions of the conditions
def total_students : ℕ := 23
def students_in_restroom : ℕ := 2
def students_absent : ℕ := 3 * students_in_restroom - 1
def students_per_desk : ℕ := 6
def fraction_full (r : ℕ) := (2 * r) / 3

-- The statement we need to prove 
theorem number_of_rows : (total_students - students_in_restroom - students_absent) / (students_per_desk * 2 / 3) = 4 :=
by
  sorry

end number_of_rows_l651_651937


namespace smaller_cube_size_l651_651898

theorem smaller_cube_size
  (original_cube_side : ℕ)
  (number_of_smaller_cubes : ℕ)
  (painted_cubes : ℕ)
  (unpainted_cubes : ℕ) :
  original_cube_side = 3 → 
  number_of_smaller_cubes = 27 → 
  painted_cubes = 26 → 
  unpainted_cubes = 1 →
  (∃ (side : ℕ), side = original_cube_side / 3 ∧ side = 1) :=
by
  intros h1 h2 h3 h4
  use 1
  have h : 1 = original_cube_side / 3 := sorry
  exact ⟨h, rfl⟩

end smaller_cube_size_l651_651898


namespace visible_diagonal_angle_l651_651402

noncomputable def α : ℝ := 144.73555555555555 -- approximation of 144° 44' 08'' in degrees

structure Cube := 
  (A B C D A1 B1 C1 D1 : ℝ×ℝ×ℝ)

-- Define the surface condition
def on_surface (M : ℝ×ℝ×ℝ) (K : Cube) : Prop :=
  (M = K.A ∨ M = K.B ∨ M = K.C ∨ M = K.D ∨ M = K.A1 ∨ M = K.C1 ∨ M = K.B1 ∨ M = K.D1)
  ∨ (∀ (x y z : ℝ), (M = (x, 0, 0) ∨ M = (x, 0, 1) ∨ M = (x, 1, 0) ∨ M = (x, 1, 1) ∨
                      M = (0, y, 0) ∨ M = (0, y, 1) ∨ M = (1, y, 0) ∨ M = (1, y, 1) ∨
                      M = (0, 0, z) ∨ M = (0, 1, z) ∨ M = (1, 0, z) ∨ M = (1, 1, z)))

theorem visible_diagonal_angle (K : Cube) (M : ℝ×ℝ×ℝ) (hM : on_surface M K) :
  90 ≤ ∠ (K.A, M, K.C1) ∧ ∠ (K.A, M, K.C1) < α := sorry

end visible_diagonal_angle_l651_651402


namespace least_multiple_75_with_digit_product_l651_651352

theorem least_multiple_75_with_digit_product (n : Nat) : 
  (n = 75375) → 
  (∃ m : Nat, m < n ∧ m % 75 = 0 ∧ (m.digits.product % 75 = 0)) → 
  False :=
sorry

end least_multiple_75_with_digit_product_l651_651352


namespace simplify_polynomial_l651_651309

theorem simplify_polynomial (r : ℝ) :
  (2 * r ^ 3 + 5 * r ^ 2 - 4 * r + 8) - (r ^ 3 + 9 * r ^ 2 - 2 * r - 3)
  = r ^ 3 - 4 * r ^ 2 - 2 * r + 11 :=
by sorry

end simplify_polynomial_l651_651309


namespace color_circles_correct_l651_651013

-- Definitions indicating the presence of five circles and the restriction on their coloring.
def CircleColor : Type := {c : Fin 5 // true}
def Color : Type := {c : Fin 3 // true}

-- Indicating connectedness between circles
def connected (a b : CircleColor) : Prop := sorry

-- Predicate that checks whether the coloring of two circles differs.
def different_colors (a b : CircleColor) (coloring : CircleColor → Color) : Prop :=
  coloring a ≠ coloring b

-- Main statement for the proof problem
theorem color_circles_correct : 
  ∃ (coloring : CircleColor → Color), (∀ (a b : CircleColor), connected a b → different_colors a b coloring) 
  ∧ (finset.card {c : CircleColor → Color // ∀ (a b : CircleColor), connected a b → different_colors a b c} = 36) :=
sorry

end color_circles_correct_l651_651013


namespace polynomial_remainder_zero_l651_651568

theorem polynomial_remainder_zero 
  (x : Polynomial ℝ) :
  (Polynomial.mod_by_monic (x ^ 1012) ((x ^ 2 - 1) * (x + 1))) = 0 :=
sorry

end polynomial_remainder_zero_l651_651568


namespace age_of_other_man_l651_651319

-- Definitions of the given conditions
def average_age_increase (avg_men : ℕ → ℝ) (men_removed women_avg : ℕ) : Prop :=
  avg_men 8 + 2 = avg_men 6 + women_avg / 2

def one_man_age : ℕ := 24
def women_avg : ℕ := 30

-- Statement of the problem to prove
theorem age_of_other_man (avg_men : ℕ → ℝ) (other_man : ℕ) :
  average_age_increase avg_men 24 women_avg →
  other_man = 20 :=
sorry

end age_of_other_man_l651_651319


namespace distance_to_place_l651_651908

-- Definitions based on the conditions
def rowing_speed : ℝ := 5
def current_speed : ℝ := 1
def total_time : ℝ := 1

-- Effective speeds downstream and upstream
def speed_downstream := rowing_speed + current_speed
def speed_upstream := rowing_speed - current_speed

-- Time equations
def time_downstream (D : ℝ) := D / speed_downstream
def time_upstream (D : ℝ) := D / speed_upstream

theorem distance_to_place :
  ∃ (D : ℝ), time_downstream D + time_upstream D = 1 ∧ D = 2.4 :=
by
  sorry

end distance_to_place_l651_651908


namespace pyramid_surface_area_l651_651046

noncomputable def total_surface_area_of_pyramid 
  (s h : ℝ) (square_base : s = 8) (height : h = 9) : ℝ :=
  let base_area := s^2
  let mid_to_center := s / 2
  let slant_height := real.sqrt (h^2 + mid_to_center^2)
  let lateral_face_area := (s * slant_height) / 2
  in base_area + 4 * lateral_face_area

theorem pyramid_surface_area : 
  total_surface_area_of_pyramid 8 9 8 9 = 64 + 16 * real.sqrt 97 :=
by 
  unfold total_surface_area_of_pyramid 
  sorry

end pyramid_surface_area_l651_651046


namespace remainder_17_pow_63_mod_7_l651_651380

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651380


namespace y_coordinate_of_P_l651_651700

open Real

noncomputable def A : : ℝ × ℝ := (-3, 0)
noncomputable def B : ℝ × ℝ := (-2, 2)
noncomputable def C : ℝ × ℝ := (2, 2)
noncomputable def D : ℝ × ℝ := (3, 0)

theorem y_coordinate_of_P (P : ℝ × ℝ) (PA PD PB PC : ℝ) :
  PA = dist P A →
  PD = dist P D →
  PB = dist P B →
  PC = dist P C →
  PA + PD = 10 →
  PB + PC = 10 →
  ∃ a b c d : ℕ, a = 32 ∧ b = 8 ∧ c = 21 ∧ d = 5 ∧ P.2 = ( (-a : ℝ) + b * sqrt c) / d ∧ (a + b + c + d = 66) :=
by
  sorry

end y_coordinate_of_P_l651_651700


namespace find_k_l651_651193

noncomputable def veca : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def vecb : ℝ × ℝ × ℝ := (-1, 0, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_k (k : ℝ) :
  dot_product (k • veca + vecb) (2 • veca - vecb) = 0 ↔ k = 7 / 5 :=
by
  sorry

end find_k_l651_651193


namespace train_cross_time_l651_651053

open Real

theorem train_cross_time :
  ∀ (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ),
    train_length = 330 →
    train_speed_kmph = 25 →
    man_speed_kmph = 2 →
    (train_length / (((train_speed_kmph + man_speed_kmph) * 1000 / 3600))) = 44 := by
  intros train_length train_speed_kmph man_speed_kmph h1 h2 h3
  dsimp
  rw [h1, h2, h3]
  -- Convert km/h to m/s: 25 km/h = 25 * (1000 / 3600) m/s etc.
  have relative_speed_mps : (25 + 2) * (1000 / 3600) = 27 * (5 / 18) := by
    norm_num
  rw relative_speed_mps
  norm_num
  sorry

end train_cross_time_l651_651053


namespace min_pencils_in_pile_l651_651444

open Nat

theorem min_pencils_in_pile : lcm 3 4 = 12 :=
by sorry

end min_pencils_in_pile_l651_651444


namespace total_chairs_in_cafe_l651_651429

-- Definitions based on conditions
def num_indoor_tables : ℕ := 9
def min_chairs_per_indoor_table : ℕ := 6
def num_outdoor_tables : ℕ := 11
def min_chairs_per_outdoor_table : ℕ := 3
def total_customers : ℕ := 35
def indoor_customers : ℕ := 18
def outdoor_customers : ℕ := total_customers - indoor_customers
def one_chair_per_customer : bool := true

-- Proof problem statement in Lean
theorem total_chairs_in_cafe :
  let total_chairs := (num_indoor_tables * min_chairs_per_indoor_table) +
                      (num_outdoor_tables * min_chairs_per_outdoor_table)
  in total_chairs = 87 :=
by
  sorry

end total_chairs_in_cafe_l651_651429


namespace ab_cd_eq_cd_l651_651589

theorem ab_cd_eq_cd 
  (a b c d : ℕ) 
  (h1 : a * b ∣ c * d) 
  (h2 : c + d ∣ a + b) 
  (h3 : d ≥ c ^ 2 - c) : 
  {a, b} = {c, d} := 
sorry

end ab_cd_eq_cd_l651_651589


namespace range_of_a_l651_651750

noncomputable def f (a x : ℝ) := x + a ^ 2 / x
noncomputable def g (x : ℝ) := x - Real.log x

theorem range_of_a (a : ℝ) (hx1 hx2 : ℝ) (h1 : 1 ≤ hx1 ∧ hx1 ≤ Real.exp 1) (h2 : 1 ≤ hx2 ∧ hx2 ≤ Real.exp 1) :
  (a > 0) →
  (∀ x1 x2 ∈ Icc (1 : ℝ) (Real.exp 1), f a x1 ≥ g x2) →
  a ≥ Real.sqrt (Real.exp 1 - 2) :=
by
  sorry

end range_of_a_l651_651750


namespace determine_scalar_l651_651096

theorem determine_scalar (u v w : ℝ^3) (k : ℝ) (h1 : u + v + w = 0) 
  (h2 : k * (v × u) + v × w + w × u = 0) : k = 2 := 
by
  sorry

end determine_scalar_l651_651096


namespace not_in_range_of_g_l651_651735

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else undefined

theorem not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g(x) ≠ 0 :=
by sorry

end not_in_range_of_g_l651_651735


namespace line_always_passes_through_fixed_point_l651_651182

theorem line_always_passes_through_fixed_point (k : ℝ) : 
  ∀ x y, y + 2 = k * (x + 1) → (x = -1 ∧ y = -2) :=
by
  sorry

end line_always_passes_through_fixed_point_l651_651182


namespace tony_paint_necessary_gallons_l651_651345

noncomputable def paint_required (n h d : ℝ) (A_coverage: ℝ) : ℝ :=
  let r := d / 2
  let area := n * (2 * Real.pi * r * h)
  Float.ceil (area / A_coverage)

theorem tony_paint_necessary_gallons : paint_required 16 18 10 350 = 26 :=
  by
    sorry

end tony_paint_necessary_gallons_l651_651345


namespace remainder_17_pow_63_mod_7_l651_651359

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651359


namespace number_of_boundaries_l651_651886

def total_runs : ℕ := 120
def sixes : ℕ := 4
def runs_per_six : ℕ := 6
def percentage_runs_by_running : ℚ := 0.60
def runs_per_boundary : ℕ := 4

theorem number_of_boundaries :
  let runs_by_running := (percentage_runs_by_running * total_runs : ℚ)
  let runs_by_sixes := (sixes * runs_per_six)
  let runs_by_boundaries := (total_runs - runs_by_running - runs_by_sixes : ℚ)
  (runs_by_boundaries / runs_per_boundary) = 6 := by
  sorry

end number_of_boundaries_l651_651886


namespace tangents_sum_of_isometric_tangent_circles_l651_651948

theorem tangents_sum_of_isometric_tangent_circles 
  (Γ : Circle) 
  (M : Point) 
  (A B C : Point) 
  (H : tangents_to_three_isometric_circles Γ M A B C) :
  (MA = MB + MC) ∨ (MB = MA + MC) ∨ (MC = MA + MB) :=
sorry

-- Definitions used in the statement
def Circle := { c : Point // same_distance_from_center }

def Point := (ℝ × ℝ)

def tangents_to_three_isometric_circles 
  (Γ : Circle) 
  (M : Point) 
  (A B C : Point) : Prop := 
true -- This would entail the geometric setup and proof of tangency, equality of radii, etc. 

def MA := distance M A
def MB := distance M B
def MC := distance M C

def distance (p1 p2 : Point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

end tangents_sum_of_isometric_tangent_circles_l651_651948


namespace addition_and_rounding_l651_651463

def round_to_thousandth (x : ℝ) : ℝ :=
  (Real.toRat x * 1000).round / 1000

theorem addition_and_rounding:
  round_to_thousandth (75.126 + 8.0034) = 83.129 :=
by
  sorry

end addition_and_rounding_l651_651463


namespace rhombus_area_l651_651840

theorem rhombus_area
  (side_length : ℝ)
  (h₀ : side_length = 2 * Real.sqrt 3)
  (tri_a_base : ℝ)
  (tri_b_base : ℝ)
  (h₁ : tri_a_base = side_length)
  (h₂ : tri_b_base = side_length) :
  ∃ rhombus_area : ℝ,
    rhombus_area = 8 * Real.sqrt 3 - 12 :=
by
  sorry

end rhombus_area_l651_651840


namespace sum_of_repeating_decimal_digits_l651_651328

theorem sum_of_repeating_decimal_digits (n : ℕ) (digits : Fin n → ℕ) :
  let x := (1 : ℚ) / (81 ^ 2)
  let dec_rep := digits
  (∀ k, dec_rep = ((0.012345679 : ℚ) ^ 2).nth_digit k) →
  (∑ i in Finset.range n, digits i) = 720 :=
by
  sorry

end sum_of_repeating_decimal_digits_l651_651328


namespace greatest_integer_less_than_l651_651155

theorem greatest_integer_less_than (M : ℕ)
  (h: Σ (s : Finset ℕ), ∑ i in s, (1 : ℚ) / (nat.factorial i * nat.factorial (21 - i))
      = (M : ℚ) / (1 * nat.factorial 20)) :
  ⌊ (M : ℚ) / 100 ⌋ = 499 :=
begin
  sorry
end

end greatest_integer_less_than_l651_651155


namespace soccer_team_substitutions_l651_651918

theorem soccer_team_substitutions : 
  let a : ℕ → ℕ := λ n, match n with
                         | 0 => 1
                         | 1 => 11 * 11
                         | 2 => 11^3 * 10
                         | 3 => 11^4 * 10 * 9
                         | _ => 0
  in (a 0 + a 1 + a 2 + a 3) % 1000 = 122 :=
by
  -- proof will go here
  sorry

end soccer_team_substitutions_l651_651918


namespace salad_percentage_less_l651_651011

def tom_rate : ℝ := 2 / 3
def tammy_rate : ℝ := 3 / 2
def combined_rate : ℝ := tom_rate + tammy_rate
def total_salad : ℝ := 65
def time_to_chop : ℝ := total_salad / combined_rate

def tom_chopped : ℝ := time_to_chop * tom_rate
def tammy_chopped : ℝ := time_to_chop * tammy_rate
def difference : ℝ := tammy_chopped - tom_chopped
def percentage_difference : ℝ := (difference / tammy_chopped) * 100

theorem salad_percentage_less :
  percentage_difference = 55.56 := by
  sorry

end salad_percentage_less_l651_651011


namespace subset_solution_l651_651717

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l651_651717


namespace symmetric_center_of_transformed_function_l651_651215

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * cos (x) ^ 2

noncomputable def g (x : ℝ) : ℝ := 2 * sin ((2 / 3) * x - π / 4) - 1

theorem symmetric_center_of_transformed_function :
  ∃ (x₀ : ℝ), (∃ (k : ℤ), x₀ = (3 / 2) * (k : ℝ) * π + (3 * π / 8)) ∧ g ((x₀ + π / 8) / 3) = -1 :=
by
  sorry

end symmetric_center_of_transformed_function_l651_651215


namespace problem_l651_651281

theorem problem (n : ℤ) (h1 : 0 ≤ n ∧ n < 37)
  (h2 : 5 * n ≡ 1 [MOD 37]) :
  ((2^n)^3 - 2) ≡ 1 [MOD 37] := 
sorry

end problem_l651_651281


namespace xy_sum_l651_651648

theorem xy_sum : ∀ (x y : ℚ), (1 / x) + (1 / y) = 4 ∧ (1 / x) - (1 / y) = -6 → x + y = -4 / 5 := 
by 
  intros x y h
  cases h with h1 h2
  sorry

end xy_sum_l651_651648


namespace find_min_value_l651_651972

theorem find_min_value :
  ∀ x : ℝ, (0 < x) → (x < 1/3) → (f x = 3/x + 1/(1 - 3*x)) → ∃ y : ℝ, y = 16 :=
by
  sorry

end find_min_value_l651_651972


namespace sin_cos_inequality_l651_651988

open Real

theorem sin_cos_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * π) :
  (sin (x - π / 6) > cos x) ↔ (π / 3 < x ∧ x < 4 * π / 3) :=
by sorry

end sin_cos_inequality_l651_651988


namespace total_budget_l651_651784

theorem total_budget (s_ticket : ℕ) (s_drinks_food : ℕ) (k_ticket : ℕ) (k_drinks : ℕ) (k_food : ℕ) 
  (h1 : s_ticket = 14) (h2 : s_drinks_food = 6) (h3 : k_ticket = 14) (h4 : k_drinks = 2) (h5 : k_food = 4) : 
  s_ticket + s_drinks_food + k_ticket + k_drinks + k_food = 40 := 
by
  sorry

end total_budget_l651_651784


namespace find_m_value_l651_651332

theorem find_m_value (m : ℝ) :
  let P := (0, 1 - m)
  let Q := (0, m^2 - 3)
  (P.snd = -Q.snd) → m = -1 :=
by
  let P := (0, 1 - m)
  let Q := (0, m^2 - 3)
  assume h : P.snd = -Q.snd
  sorry

end find_m_value_l651_651332


namespace sphere_radius_eq_l651_651593

-- Definitions based on the problem conditions
variables (a : ℝ) (P A B C E M : Point)
variables 
  (hAB : segment A B)
  (hAC : segment A C)
  (hPABC_reg : regular_tetrahedron P A B C)
  (hE_mid : midpoint E A B)
  (hM_mid : midpoint M A C)

-- Theorem statement proving the radius of the sphere passing through points C, E, M, and P
theorem sphere_radius_eq {R : ℝ} :
  sphere_radius_through_points P A B C E M = R :=
sorry

end sphere_radius_eq_l651_651593


namespace tan_of_acute_angle_l651_651603

theorem tan_of_acute_angle (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 4 * (Real.sin A)^2 - 4 * Real.sin A * Real.cos A + (Real.cos A)^2 = 0) :
  Real.tan A = 1 / 2 :=
by
  sorry

end tan_of_acute_angle_l651_651603


namespace N_subset_M_l651_651187

-- Definitions of sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x * x - x < 0 }

-- Proof statement: N is a subset of M
theorem N_subset_M : N ⊆ M :=
sorry

end N_subset_M_l651_651187


namespace choir_membership_l651_651333

theorem choir_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 8 = 3) (h3 : n ≥ 100) (h4 : n ≤ 200) :
  n = 123 ∨ n = 179 :=
by
  sorry

end choir_membership_l651_651333


namespace total_earnings_l651_651939

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end total_earnings_l651_651939


namespace remainder_17_pow_63_mod_7_l651_651356

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651356


namespace intersection_area_l651_651839

noncomputable def circle1_radius : ℝ := 3
noncomputable def circle2_radius : ℝ := 3
noncomputable def circle1_center : (ℝ, ℝ) := (3, 0)
noncomputable def circle2_center : (ℝ, ℝ) := (0, 3)

theorem intersection_area :
  let r1 := circle1_radius
  let r2 := circle2_radius
  let c1 := circle1_center
  let c2 := circle2_center in
  (interior_intersection_area r1 c1 r2 c2) = (9/2) * Real.pi - 9 := 
sorry

end intersection_area_l651_651839


namespace M_union_N_eq_M_l651_651284

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs (p.1 * p.2) = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end M_union_N_eq_M_l651_651284


namespace sequence_length_l651_651954

theorem sequence_length (n : ℕ) (a1 : ℤ) (d : ℤ) (a_last : ℤ)
  (h1 : a1 = 1) (h2 : d = 3) (h3 : a_last = 46) :
  n = ((a_last - a1) / d) + 1 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sequence_length_l651_651954


namespace intersection_A_B_l651_651639

def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { x | 3^x < 1 }

theorem intersection_A_B (A B : Set ℝ) :
  A = { x | x < 1 } → B = { x | 3^x < 1 } → 
  A ∩ B = { x | x < 0 } :=
by
  intro hA hB
  sorry

end intersection_A_B_l651_651639


namespace mark_increase_reading_time_l651_651756

theorem mark_increase_reading_time : 
  (let hours_per_day := 2
   let days_per_week := 7
   let desired_weekly_hours := 18
   let current_weekly_hours := hours_per_day * days_per_week
   let increase_per_week := desired_weekly_hours - current_weekly_hours
   increase_per_week = 4) :=
by
  let hours_per_day := 2
  let days_per_week := 7
  let desired_weekly_hours := 18
  let current_weekly_hours := hours_per_day * days_per_week
  let increase_per_week := desired_weekly_hours - current_weekly_hours
  have h1 : current_weekly_hours = 14 := by norm_num
  have h2 : increase_per_week = desired_weekly_hours - current_weekly_hours := rfl
  have h3 : increase_per_week = 18 - 14 := by rw [h2, h1]
  have h4 : increase_per_week = 4 := by norm_num
  exact h4

end mark_increase_reading_time_l651_651756


namespace smallest_positive_period_range_f_in_interval_l651_651628

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ (∀ T', 0 < T' → T' < T → ∀ x, f (x + T') ≠ f x) := by
  sorry

theorem range_f_in_interval :
  Set.range (fun x => f x) ∩ Set.Icc 0 (Real.pi / 2) = set.Icc (0 : ℝ) (1/2 + Real.sqrt 3 / 4) := by
  sorry

end smallest_positive_period_range_f_in_interval_l651_651628


namespace ratio_of_radii_of_touching_circles_l651_651838

theorem ratio_of_radii_of_touching_circles
  (r R : ℝ) (A B C D : ℝ) (h1 : A + B + C = D)
  (h2 : 3 * A = 7 * B)
  (h3 : 7 * B = 2 * C)
  (h4 : R = D / 2)
  (h5 : B = R - 3 * A)
  (h6 : C = R - 2 * A)
  (h7 : r = 4 * A)
  (h8 : R = 6 * A) :
  R / r = 3 / 2 := by
  sorry

end ratio_of_radii_of_touching_circles_l651_651838


namespace find_b120_l651_651088

-- Define the sequence based on given conditions
def b : ℕ → ℚ
| 1 := 2
| 2 := 3
| (n + 3) := (2 - b (n + 2)) / (3 * b (n + 1))

-- Statement we need to prove
theorem find_b120 : b 120 = 2 / 3 := by
  sorry

end find_b120_l651_651088


namespace no_adjacent_birch_prob_l651_651901

-- Define the main statement of the problem.
theorem no_adjacent_birch_prob :
  let m := 6 in
  let n := 143 in
  let P := (6 : ℚ) / 143 in
  ∀ (total_trees : ℕ) (non_birch_trees : ℕ) (birch_trees : ℕ) 
    (ways_all : ℕ) (ways_no_adjacent : ℕ),
  total_trees = 15 → 
  non_birch_trees = 9 →
  birch_trees = 6 →
  ways_all = (Nat.choose 15 6) →
  ways_no_adjacent = (Nat.choose 10 6) →
  (ways_no_adjacent : ℚ) / ways_all = P →
  m + n = 149 := by
  intros total_trees non_birch_trees birch_trees ways_all ways_no_adjacent h1 h2 h3 h4 h5 h6 h7
  sorry

end no_adjacent_birch_prob_l651_651901


namespace expected_value_twelve_sided_die_l651_651460

theorem expected_value_twelve_sided_die :
  let faces := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
      n := 12,
      a := 2,
      l := 24,
      S := (n / 2) * (a + l)
  in (S / n = 13) :=
by
  -- faces: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
  -- n: 12
  -- a: 2
  -- l: 24
  -- S: (n / 2) * (a + l)
  -- expected value: S / n = 13
  sorry

end expected_value_twelve_sided_die_l651_651460


namespace element_subset_a_l651_651707

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l651_651707


namespace min_value_75_l651_651188

def min_value (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value_75 
  (x y z : ℝ) 
  (h1 : (x + 5) * (y - 5) = 0) 
  (h2 : (y + 5) * (z - 5) = 0) 
  (h3 : (z + 5) * (x - 5) = 0) :
  min_value x y z = 75 := 
sorry

end min_value_75_l651_651188


namespace computation_l651_651532

theorem computation : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end computation_l651_651532


namespace perpendicular_condition_centroid_coordinates_l651_651170

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 0}
def B : Point := {x := 4, y := 0}
def C (c : ℝ) : Point := {x := 0, y := c}

def vec (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y}

def dot_product (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y

theorem perpendicular_condition (c : ℝ) (h : dot_product (vec A (C c)) (vec B (C c)) = 0) :
  c = 2 ∨ c = -2 :=
by
  -- proof to be filled in
  sorry

theorem centroid_coordinates (c : ℝ) (h : c = 2 ∨ c = -2) :
  (c = 2 → Point.mk 1 (2 / 3) = Point.mk 1 (2 / 3)) ∧
  (c = -2 → Point.mk 1 (-2 / 3) = Point.mk 1 (-2 / 3)) :=
by
  -- proof to be filled in
  sorry

end perpendicular_condition_centroid_coordinates_l651_651170


namespace derivative_at_1_l651_651632

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x - 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*x - 2

theorem derivative_at_1 : f_derivative 1 = 3 := by
  sorry

end derivative_at_1_l651_651632


namespace find_x_exists_unique_l651_651999

theorem find_x_exists_unique (n : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ x = p * q * r) (h3 : 11 ∣ x) : x = 59048 :=
sorry

end find_x_exists_unique_l651_651999


namespace total_votes_proof_l651_651760

variable (total_voters first_area_percent votes_first_area votes_remaining_area votes_total : ℕ)

-- Define conditions
def first_area_votes_condition : Prop :=
  votes_first_area = (total_voters * first_area_percent) / 100

def remaining_area_votes_condition : Prop :=
  votes_remaining_area = 2 * votes_first_area

def total_votes_condition : Prop :=
  votes_total = votes_first_area + votes_remaining_area

-- Main theorem to prove
theorem total_votes_proof (h1: first_area_votes_condition) (h2: remaining_area_votes_condition) (h3: total_votes_condition) :
  votes_total = 210000 :=
by
  sorry

end total_votes_proof_l651_651760


namespace b_completes_work_in_24_days_l651_651859

theorem b_completes_work_in_24_days 
  (a_and_b_work_rate : ℝ)
  (a_work_rate : ℝ)
  (work_rate_combined : a_and_b_work_rate = 1 / 12)
  (work_rate_a : a_work_rate = 1 / 24) : 
  (b_work_rate : ℝ) 
  (time_b: b_work_rate = 1 / 24) :=
begin
  have b_work_rate_def : b_work_rate = a_and_b_work_rate - a_work_rate := sorry,
  have time_b_def : 1 / b_work_rate = 24 := sorry,
  exact time_b_def
end

end b_completes_work_in_24_days_l651_651859


namespace smallest_w_l651_651209

def fact_936 : ℕ := 2^3 * 3^1 * 13^1

theorem smallest_w (w : ℕ) (h_w_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (12^2 ∣ 936 * w) → w = 36 :=
by
  sorry

end smallest_w_l651_651209


namespace a_eq_b_when_n_is_60_l651_651139

-- Definitions and conditions
def side_length_a (n : ℕ) (h : n > 2) : ℝ := 60 / n
def side_length_b (n : ℕ) (h : n > 2) : ℝ := 67 / (n + 7)

-- Statement of the theorem
theorem a_eq_b_when_n_is_60 {n : ℕ} (h : n > 2) : side_length_a n h = side_length_b n h ↔ n = 60 :=
by sorry

end a_eq_b_when_n_is_60_l651_651139


namespace rotor_permutations_l651_651093

-- Define the factorial function for convenience
def fact : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * fact n

-- The main statement to prove
theorem rotor_permutations : (fact 5) / ((fact 2) * (fact 2)) = 30 := by
  sorry

end rotor_permutations_l651_651093


namespace activity_popularity_order_l651_651524

theorem activity_popularity_order (dodgeball movie magic_show : ℚ)
  (h_dodgeball : dodgeball = 13 / 40)
  (h_movie : movie = 3 / 10)
  (h_magic_show : magic_show = 9 / 20) :
  [("Magic Show", magic_show), ("Dodgeball", dodgeball), ("Movie", movie)].map Prod.fst = ["Magic Show", "Dodgeball", "Movie"] :=
by
  -- Convert fractions to common denominator for easier comparison
  have h_movie' : movie = 12 / 40 := by rw [h_movie, rat.div_num_den_div (3 * 4) 10, rat.mk_eq_div_mul]
  have h_magic_show' : magic_show = 18 / 40 := by rw [h_magic_show, rat.div_num_den_div (9 * 2) 20, rat.mk_eq_div_mul]
  -- Compare and order fractions manually
  have compare_fractions : 18 / 40 > 13 / 40 ∧ 13 / 40 > 12 / 40 := sorry
  -- Using the comparison to decide the order
  exact sorry

end activity_popularity_order_l651_651524


namespace initial_percentage_of_alcohol_l651_651020

theorem initial_percentage_of_alcohol 
  (P: ℝ)
  (h_condition1 : 18 * P / 100 = 21 * 17.14285714285715 / 100) : 
  P = 20 :=
by 
  sorry

end initial_percentage_of_alcohol_l651_651020


namespace range_of_x_l651_651805

theorem range_of_x (x : ℝ) : (6 - 2 * x) ≠ 0 ↔ x ≠ 3 := 
by {
  sorry
}

end range_of_x_l651_651805


namespace seventeen_power_sixty_three_mod_seven_l651_651362

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651362


namespace find_x_l651_651206

theorem find_x
  (x : ℕ)
  (h1 : x % 7 = 0)
  (h2 : x > 0)
  (h3 : x^2 > 144)
  (h4 : x < 25) : x = 14 := 
  sorry

end find_x_l651_651206


namespace cafeteria_initial_apples_l651_651321

theorem cafeteria_initial_apples (a p k : Nat) (handed_out : a = 41) (pies : p = 2) (per_pie : k = 5) :
  a + (p * k) = 51 :=
by
  -- Given condition how to distribute apples
  rw [handed_out, pies, per_pie]
  -- Calculate apples used in pies
  simp [Nat.add_comm, Nat.mul_comm]
  -- Final initial apples count
  sorry

end cafeteria_initial_apples_l651_651321


namespace value_of_f_2017_pi_over_3_l651_651623

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f(x + Real.pi) = f(x) + Real.cos x
axiom condition2 (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi) : f(x) = -1

theorem value_of_f_2017_pi_over_3 : f(2017 * Real.pi / 3) = -1 :=
by sorry

end value_of_f_2017_pi_over_3_l651_651623


namespace overall_weighted_defective_shipped_percentage_l651_651911

theorem overall_weighted_defective_shipped_percentage
  (defective_A : ℝ := 0.06) (shipped_A : ℝ := 0.04) (prod_A : ℝ := 0.30)
  (defective_B : ℝ := 0.09) (shipped_B : ℝ := 0.06) (prod_B : ℝ := 0.50)
  (defective_C : ℝ := 0.12) (shipped_C : ℝ := 0.07) (prod_C : ℝ := 0.20) :
  prod_A * defective_A * shipped_A + prod_B * defective_B * shipped_B + prod_C * defective_C * shipped_C = 0.00510 :=
by
  sorry

end overall_weighted_defective_shipped_percentage_l651_651911


namespace num_ways_choose_starters_l651_651024

-- Given conditions as definitions
def num_players : ℕ := 12
def quadruplets : Finset ℕ := {0, 1, 2, 3}  -- Representing Betty, Barbara, Brenda, and Bethany
def starters_needed : ℕ := 5
def quadruplets_in_lineup : ℕ := 2

-- The proof statement
theorem num_ways_choose_starters (n : ℕ) (quad : Finset ℕ) (r : ℕ) (q : ℕ) :
  n = 12 ∧ quad = {0, 1, 2, 3} ∧ r = 5 ∧ q = 2 →
  (Finset.card (Finset.choose 2 quad) * Finset.card (Finset.choose (r - q) (Finset.range (n - Finset.card quad)))) = 336 :=
by
  intros h
  sorry

end num_ways_choose_starters_l651_651024


namespace pow_mod_seventeen_l651_651387

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651387


namespace fin_pos_integers_exists_l651_651991

theorem fin_pos_integers_exists (a : ℕ) (h : a ≥ 9) :
  ∃ (S : Finset ℕ), ∀ n,
    n ∈ S → tau(n) = a ∧ n ∣ phi(n) + sigma(n) :=
begin
  sorry
end

end fin_pos_integers_exists_l651_651991


namespace simplify_fraction_l651_651311

variable {a b c : ℝ}
variable {α β γ : ℝ}

-- Define the angles and sides conditions as a relevant context
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem simplify_fraction
  (h_triangle : is_triangle a b c)
  (h_angles : α + β + γ = π) :
  (α * Real.cos α + b * Real.cos β - c * Real.cos γ) / (α * Real.cos α - b * Real.cos β + c * Real.cos γ)
  = Real.tan γ * Real.cot β := 
begin
  sorry
end

end simplify_fraction_l651_651311


namespace man_l651_651003

-- Define the conditions
def speed_downstream : ℕ := 8
def speed_upstream : ℕ := 4

-- Define the man's rate in still water
def rate_in_still_water : ℕ := (speed_downstream + speed_upstream) / 2

-- The target theorem
theorem man's_rate_in_still_water : rate_in_still_water = 6 := by
  -- The statement is set up. Proof to be added later.
  sorry

end man_l651_651003


namespace cosine_value_l651_651611

noncomputable def cosine_of_angle (x y : ℤ) : ℚ :=
  x / Real.sqrt (x^2 + y^2)

theorem cosine_value (x y : ℤ) (h : x = -4 ∧ y = 3) : 
  cosine_of_angle x y = -4 / 5 := 
by
  -- Provide proof steps here
  sorry

end cosine_value_l651_651611


namespace rowing_distance_l651_651912

theorem rowing_distance
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (total_time : ℝ)
  (D : ℝ)
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : total_time = 15)
  (h4 : D / (rowing_speed + current_speed) + D / (rowing_speed - current_speed) = total_time) :
  D = 72 := 
sorry

end rowing_distance_l651_651912


namespace sum_of_divisors_360_l651_651398

theorem sum_of_divisors_360 : 
  (∑ d in (finset.filter (λ x, 360 % x = 0) (finset.range (360 + 1))), d) = 1170 :=
sorry

end sum_of_divisors_360_l651_651398


namespace find_bn_bound_an_bn_limit_ln_n_an_l651_651089

noncomputable def a_seq (n : ℕ) : ℝ :=
  ∫ θ in -Real.pi / 6 .. Real.pi / 6, Real.exp (n * Real.sin θ)

noncomputable def b_seq (n : ℕ) : ℝ :=
  ∫ θ in -Real.pi / 6 .. Real.pi / 6, Real.exp (n * Real.sin θ) * Real.cos θ

theorem find_bn (n : ℕ) : b_seq n = (2 / n) * Real.sinh (n / 2) := 
  sorry

theorem bound_an_bn (n : ℕ) : b_seq n ≤ a_seq n ∧ a_seq n ≤ (2 / Real.sqrt 3) * b_seq n := 
  sorry

theorem limit_ln_n_an : Real.lim (λ n, (Real.ln (n * a_seq n) / n)) = 1 / 2 := 
  sorry

end find_bn_bound_an_bn_limit_ln_n_an_l651_651089


namespace zero_not_in_range_of_g_l651_651732

def g (x : ℝ) : ℤ :=
  if x > -3 then
    Int.ceil (2 / (x + 3))
  else 
    Int.floor (2 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ x : ℝ, g x = 0 :=
sorry

end zero_not_in_range_of_g_l651_651732


namespace max_regions_by_n_lines_max_regions_prop_l651_651535

theorem max_regions_by_n_lines (n : ℕ) : ℕ :=
  let z : ℕ → ℕ
    | 0     => 1
    | k + 1 => z k + (k + 1)
  z n

theorem max_regions_prop (n : ℕ) : max_regions_by_n_lines n = 1 + (n * (n + 1)) / 2 := by
  induction n with k hk
  case zero =>
    show max_regions_by_n_lines 0 = 1 + (0 * (0 + 1)) / 2
    simp [max_regions_by_n_lines]
  case succ k ih =>
    show max_regions_by_n_lines (k + 1) = 1 + ((k + 1) * (k + 2)) / 2
    simp [max_regions_by_n_lines, ih]
    sorry

end max_regions_by_n_lines_max_regions_prop_l651_651535


namespace math_proof_problem_l651_651650

variables (m n : Type) (α β : Type)
variables [non_coincident_lines m n] [non_coincident_planes α β]

-- Assumptions: m and n are non-coincident lines, and α and β are non-coincident planes
axiom non_coincident_lines (l1 l2 : Type) : Prop
axiom non_coincident_planes (p1 p2 : Type) : Prop

def Statement1 : Prop := forall m n α, (parallel_to_plane m α ∧ parallel_to_plane n α) → (¬ intersects m n)
def Statement2 : Prop := forall m n α, (perpendicular_to_plane m α ∧ perpendicular_to_plane n α) → (parallel_lines m n)
def Statement3 : Prop := forall m n α β, (perpendicular_planes α β ∧ perpendicular_lines m n ∧ perpendicular_to_plane m α) → (perpendicular_to_plane n β)
def Statement4 : Prop := forall m n α, (perpendicular_projections_in_plane m n α) → (perpendicular_lines m n)

theorem math_proof_problem (m n : Type) (α β : Type)
  [non_coincident_lines m n]
  [non_coincident_planes α β]
  : ∃! S, S ∈ { Statement1, Statement2, Statement3, Statement4 } ∧ S := 
sorry

end math_proof_problem_l651_651650


namespace sum_of_roots_of_z7_l651_651808

noncomputable def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ * real.pi / 180)

theorem sum_of_roots_of_z7 :
  let z := -1 / real.sqrt 2 + complex.I / real.sqrt 2 in
  ∃ (θs : list ℝ), (∀ θ ∈ θs, 0 ≤ θ ∧ θ < 360) ∧ 
  θs.sum = 945 ∧ 
  θs.length = 7 ∧ 
  (∀ θ ∈ θs, cis θ ^ 7 = z) := sorry

end sum_of_roots_of_z7_l651_651808


namespace hyperbola_eccentricity_l651_651179

-- Definitions based on conditions
def hyperbola (x y : ℝ) (a : ℝ) := x^2 / a^2 - y^2 / 5 = 1

-- Main theorem
theorem hyperbola_eccentricity (a : ℝ) (c : ℝ) (h_focus : c = 3) (h_hyperbola : hyperbola 0 0 a) (focus_condition : c^2 = a^2 + 5) :
  c / a = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l651_651179


namespace remainder_of_17_pow_63_mod_7_l651_651369

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651369


namespace triangle_area_l651_651249

theorem triangle_area
  (a b : ℝ)
  (C : ℝ)
  (h₁ : a = 2)
  (h₂ : b = 3)
  (h₃ : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l651_651249


namespace problem1_problem2_l651_651936

theorem problem1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 :=
  sorry

theorem problem2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 :=
  sorry

end problem1_problem2_l651_651936


namespace four_student_committee_l651_651982

axiom num_students : ℕ
axiom num_committee : ℕ
axiom condition : num_committee = (8 * 7) * (Nat.choose 6 2)

theorem four_student_committee : num_committee = 840 :=
by
  rw condition
  sorry

end four_student_committee_l651_651982


namespace prop_logic_example_l651_651664

theorem prop_logic_example (p q : Prop) (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by {
  sorry
}

end prop_logic_example_l651_651664


namespace value_of_7x_minus_3y_l651_651869

theorem value_of_7x_minus_3y (x y : ℚ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := 
sorry

end value_of_7x_minus_3y_l651_651869


namespace real_roots_equation_l651_651855

theorem real_roots_equation :
  ¬(∃ x : ℝ, x ≥ -1 ∧ sqrt(x+1) = -1) ∧
  (∃ x : ℝ, x = -real.cbrt 2) ∧
  ¬(∃ x : ℝ, x ≠ 1 ∧ x / (x^2 - 1) = 1 / (x^2 - 1)) ∧
  ¬(∃ x : ℝ, (x-1)^2 = -2) :=
by
  -- proof goes here
  sorry

end real_roots_equation_l651_651855


namespace find_x_with_conditions_l651_651995

theorem find_x_with_conditions (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1)
  (h2 : (nat.factors x).to_finset.card = 3) (h3 : 11 ∈ (nat.factors x).to_finset) :
  x = 59048 := 
by {
  sorry
}

end find_x_with_conditions_l651_651995


namespace alyosha_cube_cut_l651_651499

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651499


namespace polynomial_sum_eq_3_l651_651316

noncomputable def x : ℂ := sorry

axiom x_condition : x^2020 - 3 * x - 1 = 0
axiom x_not_one : x ≠ 1

theorem polynomial_sum_eq_3 : (λ n : ℕ => x^n).sum (Finset.range 2020) = 3 :=
by sorry

end polynomial_sum_eq_3_l651_651316


namespace sufficient_but_not_necessary_condition_for_a_gt_b_gt_zero_l651_651605

variable (a b : ℝ)

-- Conditions
axiom condition_a_gt_b_gt_zero : a > b ∧ b > 0

-- The proof goal
theorem sufficient_but_not_necessary_condition_for_a_gt_b_gt_zero :
  (sqrt (a - 1) > sqrt (b - 1)) → (a > b ∧ b > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_a_gt_b_gt_zero_l651_651605


namespace smallest_value_of_a_b_l651_651272

theorem smallest_value_of_a_b :
  ∃ (a b : ℤ), (∀ x : ℤ, ((x^2 + a*x + 20) = 0 ∨ (x^2 + 17*x + b) = 0) → x < 0) ∧ a + b = -5 :=
sorry

end smallest_value_of_a_b_l651_651272


namespace square_root_of_16_is_pm_4_l651_651823

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l651_651823


namespace expressions_not_equivalent_l651_651086

theorem expressions_not_equivalent (x : ℝ) (h₁ : x^2 + 1 ≠ 0) (h₂ : x^2 + 2x + 1 ≠ 0) :
  (x^2 + x + 1) / (x^2 + 1) ≠ (x + 1)^2 / (x^2 + 2x + 1) :=
by
  sorry

end expressions_not_equivalent_l651_651086


namespace missing_digit_in_115th_rising_number_l651_651539

theorem missing_digit_in_115th_rising_number :
  ∀ n : ℕ, n = 115 →
  let rising_numbers := { number : ℕ | ∃ (digits : Finset ℕ), (digits.card = 6) ∧ (∀ (d1 d2 : ℕ), d1 < d2 → d1 ∈ digits → d2 ∈ digits) ∧ (number ∈ digits) } in
  let sorted_rising_numbers := (Finset.sort (· < ·) rising_numbers) in
  ∃ missing_digit : ℕ, missing_digit ∈ ({1,2,3,4,5,6,7,8} : Finset ℕ) \ (Finset.of_list (sorted_rising_numbers.get ⟨n, sorry⟩)) :=
sorry

end missing_digit_in_115th_rising_number_l651_651539


namespace B_fraction_l651_651304

theorem B_fraction (A_s B_s C_s : ℕ) (h1 : A_s = 600) (h2 : A_s = (2 / 5) * (B_s + C_s))
  (h3 : A_s + B_s + C_s = 1800) :
  B_s / (A_s + C_s) = 1 / 6 :=
by
  sorry

end B_fraction_l651_651304


namespace QR_length_l651_651325

noncomputable theory

variables (x y k : ℝ)

def ellipse : Prop := (x^2 / 2002^2 + y^2 / 1949^2 = 1)

def line_slope_ab : ℝ := k
def line_slope_cd : ℝ := - (1949^2 / (2002^2 * k))

def triangle_pqr (OA OC QR : ℝ) (angleAOC : ℝ) : Prop :=
  (OA = 2002) ∧ (OC = 1949) ∧ (QR = 53) ∧ (OA * OC * sin angleAOC = 2002 * 1949) ∧
  (abs (angleAOC - π/2) = abs angleAOC)

theorem QR_length (QR : ℝ) (angleAOC : ℝ) :
  ∀ (OA OC : ℝ), triangle_pqr OA OC QR angleAOC → QR = 53 :=
by
  intros OA OC h
  cases h with hoA hoC
  sorry

end QR_length_l651_651325


namespace area_shaded_region_is_negative_68_pi_l651_651242

noncomputable def radius_larger_circle : ℝ := 10
noncomputable def distance_center_to_tangency : ℝ := 4
noncomputable def radius_smaller_circles : ℝ := real.sqrt(84)
noncomputable def area_smaller_circles : ℝ := 2 * real.pi * (radius_smaller_circles ^ 2)
noncomputable def area_larger_circle : ℝ := real.pi * (radius_larger_circle ^ 2)
noncomputable def area_shaded_region : ℝ := area_larger_circle - area_smaller_circles

theorem area_shaded_region_is_negative_68_pi :
  area_shaded_region = -68 * real.pi := by
   -- sorry will be replaced by the actual proof 
  sorry

end area_shaded_region_is_negative_68_pi_l651_651242


namespace sum_first_50_terms_l651_651185

def sequence (n : ℕ) : ℤ := (-1)^n * (4*n - 3)

theorem sum_first_50_terms : (Finset.range 50).sum (λ n, sequence n) = 100 :=
by
  sorry

end sum_first_50_terms_l651_651185


namespace opposite_seven_is_minus_seven_l651_651335

theorem opposite_seven_is_minus_seven :
  ∃ x : ℤ, 7 + x = 0 ∧ x = -7 := 
sorry

end opposite_seven_is_minus_seven_l651_651335


namespace find_area_fifth_rect_l651_651907

variable (x y n : ℕ)
variable (k m : ℕ → ℕ)
variable (a b c d : ℕ)

-- Given areas of the rectangles AEIB, EIFJ, IJDB, and GJHD respectively
variable (area_AEIB : a = k (y - n))
variable (area_EIFJ : b = (m k) (y - n))
variable (area_IJDB : d = (x - m k) n)
variable (area_GJHD : c = m k (y - n))

theorem find_area_fifth_rect : (x * y - x * n) = xy - xn := by
  sorry -- proof is not required

end find_area_fifth_rect_l651_651907


namespace at_most_four_points_with_equal_pairwise_distances_l651_651214

-- Definition: in_space n means n points are in space
def in_space (n : ℕ) : Prop := n ≥ 1

-- Definition: equal_pairwise_distances means all pairwise distances are equal
def equal_pairwise_distances (points : List (ℝ × ℝ × ℝ)) : Prop :=
  ∀ i j, i ≠ j → dist points[i] points[j] = dist points[0] points[1]

theorem at_most_four_points_with_equal_pairwise_distances :
  ∀ (n : ℕ) (points : List (ℝ × ℝ × ℝ)), in_space n → equal_pairwise_distances points → n ≤ 4 :=
by
  intros n points h_space h_equal
  sorry

end at_most_four_points_with_equal_pairwise_distances_l651_651214


namespace prism_x_eq_3600_l651_651450

/-- Given that a right rectangular prism whose surface area and volume are numerically equal has edge lengths log2 x, log5 x, and log6 x,
prove that x = 3600
-/
theorem prism_x_eq_3600 {x : ℝ} (h : (2 * ((log x / log 2) * (log x / log 5) + (log x / log 2) * (log x / log 6) + (log x / log 5) * (log x / log 6)) = (log x / log 2) * (log x / log 5) * (log x / log 6))) : x = 3600 :=
sorry

end prism_x_eq_3600_l651_651450


namespace subset_a_eq_1_l651_651721

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651721


namespace rhind_papyrus_prob_l651_651876

theorem rhind_papyrus_prob (a₁ a₂ a₃ a₄ a₅ : ℝ) (q : ℝ) 
  (h_geom_seq : a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3 ∧ a₅ = a₁ * q^4)
  (h_loaves_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 93)
  (h_condition : a₁ + a₂ = (3/4) * a₃) 
  (q_gt_one : q > 1) :
  a₃ = 12 :=
sorry

end rhind_papyrus_prob_l651_651876


namespace angle_B_is_60_l651_651692

noncomputable def triangle_with_centroid (a b c : ℝ) (GA GB GC : ℝ) : Prop :=
  56 * a * GA + 40 * b * GB + 35 * c * GC = 0

theorem angle_B_is_60 {a b c GA GB GC : ℝ} (h : 56 * a * GA + 40 * b * GB + 35 * c * GC = 0) :
  ∃ B : ℝ, B = 60 :=
sorry

end angle_B_is_60_l651_651692


namespace all_faces_rhombuses_l651_651655

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l651_651655


namespace range_of_g_l651_651541

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ - x + 1

theorem range_of_g : ∀ y, y ∈ set.range (g) ↔ (0 ≤ y ∧ y < 1) :=
sorry

end range_of_g_l651_651541


namespace prism_has_12_vertices_l651_651567

theorem prism_has_12_vertices (B : Type) (V : Type) [Fintype B] [Fintype V]
  (b1 b2 : B) (rects : Fin 6 → B × V → B × V)
  (hex : B → Fin 6 → V) (parallel_cong_bases : b1 ≠ b2 ∧ ∃ p1 p2, parallel p1 p2 ∧ congruent p1 p2) 
  (rect_sides : ∀ (i : Fin 6), ∃ a b c d, rects i = (a, b), (b, c), (c, d), (d, a)) :
  Fintype.card V = 12 :=
by
  sorry

end prism_has_12_vertices_l651_651567


namespace pile_weight_replacement_l651_651769

open List

variable {α : Type*} [LinearOrderedAddCommMonoid α]

def sum_k_heaviest (l : List α) (k : ℕ) : α :=
(sum (take k (sort (flip (· ≤ ·)) l)))

theorem pile_weight_replacement 
    (pile1 pile2 : List α)
    (x : α)
    (hx : 0 < x)
    (h_weight_eq : pile1.sum = pile2.sum)
    (h_k_heaviest : ∀ k : ℕ, k ≤ pile1.length → k ≤ pile2.length → 
        sum_k_heaviest pile1 k ≤ sum_k_heaviest pile2 k) :
    sum (map (λ w => min w x) pile1) ≥ sum (map (λ w => min w x) pile2) := by 
  sorry

end pile_weight_replacement_l651_651769


namespace cube_cut_problem_l651_651489

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651489


namespace find_k_l651_651183

theorem find_k (k : ℝ) (h_pos : k > 0)
  (h_eq : (y = k * x) ∩ C = {A, B}) 
  (AB : dist A B = (2 / 5) * sqrt 5) : k = 1 / 2 := by
  /-
  Given the conditions:
  1. The line equation y = kx with k > 0.
  2. The circle equation (x-2)^2 + y^2 = 1.
  3. The intersection of the line and circle are points A and B.
  4. The distance AB is given by 2/5 * sqrt 5.
  
  We need to prove that: k = 1 / 2.
  -/
  sorry

end find_k_l651_651183


namespace polygon_max_sides_l651_651212

theorem polygon_max_sides (n : ℕ) (h : (n - 2) * 180 < 2005) : n ≤ 13 :=
by {
  sorry
}

end polygon_max_sides_l651_651212


namespace arithmetic_signs_l651_651924

theorem arithmetic_signs :
  (∃ A B C D E, 
     A = "÷" ∧ B = "=" ∧ C = "×" ∧ D = "+" ∧ E = "-" ∧
     (4 A 2 B 2) ∧ (8 B 4 C 2) ∧ (2 D 3 B 5) ∧ (4 B 5 E 1)) :=
by
  sorry

end arithmetic_signs_l651_651924


namespace ratio_AA1_BB1_division_height_proportions_l651_651865

-- Part (a) statement
theorem ratio_AA1_BB1_division (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (A B C A1 B1 : Point) (hA1 : LineSegment B C) 
  (hB1 : LineSegment A C) (hRatio1 : A1 ∈ hA1 ∧ BA1 / A1C = 1 / p) 
  (hRatio2 : B1 ∈ hB1 ∧ AB1 / B1C = 1 / q) :
  AA1 / AO = (1 + p) / q := 
sorry

-- Part (b) statement
theorem height_proportions (a1 b1 c d : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0)
  (hc : c > 0) (hd : d > 0)
  (hRelation : a1 = (1 + p + q) / (1 + p) ∧ b1 = (1 + p + q) / (1 + q) ∧ c = (1 + p + q) * d) :
  1 / a1 + 1 / b1 = 1 / c + 1 / d := 
sorry

end ratio_AA1_BB1_division_height_proportions_l651_651865


namespace sub_frac_pow_eq_l651_651528

theorem sub_frac_pow_eq :
  7 - (2 / 5)^3 = 867 / 125 := by
  sorry

end sub_frac_pow_eq_l651_651528


namespace sqrt_decimals_l651_651689

theorem sqrt_decimals :
  (sqrt 0.0625 = 0.25) →
  (sqrt 0.625 ≈ 0.791) →
  (sqrt 6.25 = 2.5) ∧ (sqrt 62.5 ≈ 7.91) :=
by
  intros h1 h2
  sorry

end sqrt_decimals_l651_651689


namespace product_of_d_for_common_root_l651_651113

theorem product_of_d_for_common_root 
    (f : Polynomial ℝ := Polynomial.C 4 + Polynomial.C 3 * X + Polynomial.C 2 * X^2 + X^3)
    (g : Polynomial ℝ := Polynomial.C 3 + d * X + X^2)
    (d : ℝ) :
  (∀ α : ℝ, f.eval α = 0 ∧ g.eval α = 0) → ∃ ds : Set ℝ, ds = {0, 2} ∧ ∏ x in ds, x = 0 := 
sorry

end product_of_d_for_common_root_l651_651113


namespace solve_trig_eq_l651_651116

theorem solve_trig_eq (x : ℝ) (k : ℤ) (h1 : cos(2 * x) ≠ 0) (h2 : sin(2 * x) ≠ 0) :
    (tan(2 * x))^3 + (cot(2 * x))^3 + 6 * (arcsin(2 * x)) = 8 * (sin(4 * x))⁻³ ↔
    (2 * x = (2 * π * k + π / 3) ∨ 2 * x = (2 * π * k - π / 3)) := sorry

end solve_trig_eq_l651_651116


namespace range_of_f_max_value_of_omega_l651_651131

noncomputable def f (ω x : ℝ) : ℝ :=
  4 * cos (ω * x - π / 6) * sin (ω * x) - cos (2 * ω * x + π)

theorem range_of_f (ω : ℝ) (hω : ω > 0) : 
  ∀ x : ℝ, 1 - sqrt 3 ≤ f(ω, x) ∧ f(ω, x) ≤ 1 + sqrt 3 :=
sorry

theorem max_value_of_omega (ω : ℝ) (hω : ω > 0)
  (h_increasing : ∀ (x y : ℝ), - (3 * π / 2) ≤ x ∧ x ≤ y ∧ y ≤ π / 2 → f(ω, x) ≤ f(ω, y)) :
  ω ≤ 1 / 6 :=
sorry

end range_of_f_max_value_of_omega_l651_651131


namespace cost_per_person_l651_651951

-- Definitions based on conditions
def totalCost : ℕ := 13500
def numberOfFriends : ℕ := 15

-- Main statement
theorem cost_per_person : totalCost / numberOfFriends = 900 :=
by sorry

end cost_per_person_l651_651951


namespace parabola_vertex_distance_l651_651955

noncomputable def vertex_distance : ℝ :=
  let p1 := (0, 4) -- vertex of the first parabola
  let p2 := (0, -1) -- vertex of the second parabola
  Real.dist p1.snd p2.snd

theorem parabola_vertex_distance :
  let p := (λ x y : ℝ, (real.sqrt (x^2 + y^2) + abs (y - 3))) in
  (∀ x y, p x y = 5 → Real.dist 4 (-1) = 5)
:= by
  intro p x y hp
  have h1 : Real.dist 4 (-1) = abs (4 - (-1)) := Real.dist_eq.abs_sub
  rw [abs_of_nonneg] at h1
  exact h1
  linarith
  sorry

end parabola_vertex_distance_l651_651955


namespace above_265_is_234_l651_651401

namespace PyramidArray

-- Definition of the pyramid structure and identifying important properties
def is_number_in_pyramid (n : ℕ) : Prop :=
  ∃ k : ℕ, (k^2 - (k - 1)^2) / 2 ≥ n ∧ (k^2 - (k - 1)^2) / 2 < n + (2 * k - 1)

def row_start (k : ℕ) : ℕ :=
  (k - 1)^2 + 1

def row_end (k : ℕ) : ℕ :=
  k^2

def number_above (n : ℕ) (r : ℕ) : ℕ :=
  row_start r + ((n - row_start (r + 1)) % (2 * (r + 1) - 1))

theorem above_265_is_234 : 
  (number_above 265 16) = 234 := 
sorry

end PyramidArray

end above_265_is_234_l651_651401


namespace flowers_not_roses_percentage_l651_651830

def percentage_non_roses (roses tulips daisies : Nat) : Nat :=
  let total := roses + tulips + daisies
  let non_roses := total - roses
  (non_roses * 100) / total

theorem flowers_not_roses_percentage :
  percentage_non_roses 25 40 35 = 75 :=
by
  sorry

end flowers_not_roses_percentage_l651_651830


namespace vectors_not_coplanar_l651_651062
-- Import the entire Mathlib for necessary components

-- Define the vectors
def a : ℝ × ℝ × ℝ := (6, 3, 4)
def b : ℝ × ℝ × ℝ := (-1, -2, -1)
def c : ℝ × ℝ × ℝ := (2, 1, 2)

-- Define a function to compute the scalar triple product (determinant)
def scalar_triple_product (a b c : ℝ × ℝ × ℝ) : ℝ :=
  (a.1 * (b.2 * c.3 - b.3 * c.2)) -
  (a.2 * (b.1 * c.3 - b.3 * c.1)) +
  (a.3 * (b.1 * c.2 - b.2 * c.1))

-- The statement to prove that the vectors are not coplanar
theorem vectors_not_coplanar : scalar_triple_product a b c ≠ 0 :=
by
  unfold scalar_triple_product
  -- Computation of the determinant shows it's -6
  change 6 * ((-2 * 2) - (-1 * 1)) -
         3 * ((-1 * 2) - (-1 * 2)) +
         4 * ((-1 * 1) - (-2 * 2)) ≠ 0
  norm_num
  -- Conclude with  -6 ≠ 0
  exact ne_of_eq_of_ne rfl (by norm_num)

end vectors_not_coplanar_l651_651062


namespace president_and_committee_l651_651228

theorem president_and_committee :
  ∃ n : ℕ, n = 10 * (Nat.choose 9 3) :=
begin
  use 840,
  sorry
end

end president_and_committee_l651_651228


namespace alyosha_cube_cut_l651_651497

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651497


namespace calculate_star_value_l651_651844

def custom_operation (a b : ℕ) : ℕ :=
  (a + b)^3

theorem calculate_star_value : custom_operation 3 5 = 512 :=
by
  sorry

end calculate_star_value_l651_651844


namespace probability_convex_pentagon_l651_651549

theorem probability_convex_pentagon (h_points : Finset ℕ) (h_size : h_points.card = 8) :
  let total_chords := (Finset.choose 2 h_points.card)
  let total_ways := (Finset.choose 5 total_chords)
  let favorable_outcomes := (Finset.choose 5 h_points.card)
  let probability := favorable_outcomes / total_ways
  probability = (1 : ℚ) / 1755 :=
begin
  sorry
end

end probability_convex_pentagon_l651_651549


namespace find_b_l651_651620

-- Define the variables involved
variables (a b : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := 2 * a + 1 = 1
def condition_2 : Prop := b + a = 3

-- Theorem statement to prove b = 3 given the conditions
theorem find_b (h1 : condition_1 a) (h2 : condition_2 a b) : b = 3 := by
  sorry

end find_b_l651_651620


namespace walking_speed_l651_651043

theorem walking_speed : 
  ∀ (v : ℝ), (∀ (half_distance total_distance : ℝ), half_distance = total_distance / 2 → total_distance = 26.67  → 
  ∀ (time1 time2 total_time : ℝ), time1 = half_distance / v → time2 = half_distance / 4 → total_time = time1 + time2 → total_time = 6)
  → v = 5 :=
by
  intros v h
  have h1 := h (26.67 / 2) 26.67 
  simp only [div_div_eq_div_mul, div_self] at h1
  sorry

end walking_speed_l651_651043


namespace max_distance_correct_l651_651351

def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

noncomputable def sphere1_center : ℝ × ℝ × ℝ := (-2, -10, 5)
noncomputable def sphere1_radius : ℝ := 19

noncomputable def sphere2_center : ℝ × ℝ × ℝ := (12, 8, -16)
noncomputable def sphere2_radius : ℝ := 87

noncomputable def max_distance_between_spheres : ℝ :=
  dist sphere1_center sphere2_center + sphere1_radius + sphere2_radius

theorem max_distance_correct : max_distance_between_spheres = 137 := by
  sorry

end max_distance_correct_l651_651351


namespace janina_cover_expenses_l651_651256

noncomputable def rent : ℝ := 30
noncomputable def supplies : ℝ := 12
noncomputable def price_per_pancake : ℝ := 2
noncomputable def total_expenses : ℝ := rent + supplies

theorem janina_cover_expenses : total_expenses / price_per_pancake = 21 := 
by
  calc
    total_expenses / price_per_pancake 
    = (rent + supplies) / price_per_pancake : by rfl
    ... = 42 / 2 : by norm_num
    ... = 21 : by norm_num

end janina_cover_expenses_l651_651256


namespace sufficient_but_not_necessary_condition_l651_651017

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 + x = 0) : (x = -1 → x^2 + x = 0) ∧ (∃ y, y ≠ -1 ∧ y^2 + y = 0) :=
by
  -- Sufficient condition: If x = -1, then the equation x^2 + x = 0 holds.
  have suff : x = -1 → x^2 + x = 0 := by
    intro hx
    rw [hx]
    norm_num,

  -- Not necessary condition: There exists some y different from -1 such that y^2 + y = 0 holds.
  have not_necess : ∃ y, y ≠ -1 ∧ y^2 + y = 0 :=
    ⟨0, by norm_num⟩,

  -- Combine both results
  exact ⟨suff, not_necess⟩

end sufficient_but_not_necessary_condition_l651_651017


namespace x_plus_2y_equals_2_l651_651652

theorem x_plus_2y_equals_2 (x y : ℝ) (h : |x + 3| + (2 * y - 5)^2 = 0) : x + 2 * y = 2 := 
sorry

end x_plus_2y_equals_2_l651_651652


namespace binomial_coeff_sum_l651_651161

theorem binomial_coeff_sum 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h1 : (1 - 2 * 0 : ℝ)^(7) = a_0 + a_1 * 0 + a_2 * 0^2 + a_3 * 0^3 + a_4 * 0^4 + a_5 * 0^5 + a_6 * 0^6 + a_7 * 0^7)
  (h2 : (1 - 2 * 1 : ℝ)^(7) = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5 + a_6 * 1^6 + a_7 * 1^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 := 
sorry

end binomial_coeff_sum_l651_651161


namespace worker_efficiency_l651_651461

theorem worker_efficiency (W_p W_q : ℚ) 
  (h1 : W_p = 1 / 24) 
  (h2 : W_p + W_q = 1 / 14) :
  (W_p - W_q) / W_q * 100 = 40 :=
by
  sorry

end worker_efficiency_l651_651461


namespace angle_B_and_C_l651_651696

-- Let ABC be a triangle with \(\angle A = 60^\circ\)
-- Points M, N, K lie on BC, AC, AB respectively.
-- BK = KM = MN = NC.
-- AN = 2AK.
-- Prove that \(\angle B = 75^\circ\) and \(\angle C = 45^\circ\).

theorem angle_B_and_C (A B C M N K : Type) [triangle A B C]
  (angle_A_eq_60 : angle A = 60)
  (point_conditions : (M ∈ line(B, C)) ∧ (N ∈ line(A, C)) ∧ (K ∈ line(A, B)))
  (length_conditions : distance(B, K) = distance(K, M) ∧ distance(K, M) = distance(M, N) ∧ distance(M, N) = distance(N, C))
  (AN_eq_2AK : distance(A, N) = 2 * distance(A, K)):
  angle B = 75 ∧ angle C = 45 := sorry

end angle_B_and_C_l651_651696


namespace fractional_sides_l651_651211

variable {F : ℕ} -- Number of fractional sides
variable {D : ℕ} -- Number of diagonals

theorem fractional_sides (h1 : D = 2 * F) (h2 : D = F * (F - 3) / 2) : F = 7 :=
by
  sorry

end fractional_sides_l651_651211


namespace find_radius_c_l651_651858

noncomputable def rectangle_circle_radius (PQRS : Prop) (C D : Type) (r : ℝ) (c : ℝ) : Prop :=
  ∃ (PQ RS QR PS : ℝ), PQRS ∧
   (∀ (C_center : C) (D_center : D),
      (PQRS ∧ r = 9 ∧ (∀ (x : ℝ), x ∈ [PQ, RS, QR, PS]) ∧
      (touches (circle C_center 9) PQRS) ∧
      (touches (circle D_center c) QRPS) ∧
      (touches (circle C_center 9) (circle D_center c))))

theorem find_radius_c (PQRS : Prop) (C D : Type) (r : ℝ) (c : ℝ) :
  rectangle_circle_radius PQRS C D r c → c = 4 := by
    sorry

end find_radius_c_l651_651858


namespace chinese_number_representation_l651_651236

theorem chinese_number_representation :
  ∀ (祝 贺 华 杯 赛 : ℕ),
  祝 = 4 → 贺 = 8 → 
  华 ≠ 杯 ∧ 华 ≠ 赛 ∧ 杯 ≠ 赛 ∧ 华 ≠ 祝 ∧ 华 ≠ 贺 ∧ 杯 ≠ 祝 ∧ 杯 ≠ 贺 ∧ 赛 ≠ 祝 ∧ 赛 ≠ 贺 → 
  华 ≥ 1 ∧ 华 ≤ 9 → 杯 ≥ 1 ∧ 杯 ≤ 9 → 赛 ≥ 1 ∧ 赛 ≤ 9 → 
  华 * 100 + 杯 * 10 + 赛 = 7632 :=
begin
  sorry
end

end chinese_number_representation_l651_651236


namespace number_of_math_books_l651_651846

-- Definitions for conditions
variables (M H : ℕ)

-- Given conditions as a Lean proposition
def conditions : Prop :=
  M + H = 80 ∧ 4 * M + 5 * H = 368

-- The theorem to prove
theorem number_of_math_books (M H : ℕ) (h : conditions M H) : M = 32 :=
by sorry

end number_of_math_books_l651_651846


namespace part2_l651_651176

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem part2 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f 1 x1 = f 1 x2) : x1 + x2 > 2 := by
  have f_x1 := h2
  sorry

end part2_l651_651176


namespace continuous_stripe_probability_l651_651963

-- Define the concept of the cube's faces being striped or clear.
def face_state : Type := bool  -- true represents striped, false represents clear

-- Define the total number of faces.
def num_faces : Nat := 6

-- The total number of possible combinations of face states.
noncomputable def total_combinations : Nat := Nat.pow 2 num_faces

-- The number of favorable outcomes where a continuous stripe encircles the cube.
def favorable_outcomes : Nat := 6

-- The probability of a continuous stripe encircling the cube.
noncomputable def probability_continuous_stripe : ℚ := favorable_outcomes.toRat / total_combinations.toRat

-- The theorem to prove the probability of a continuous stripe encircling the cube is 3/32.
theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 32 := by 
  sorry

end continuous_stripe_probability_l651_651963


namespace maximum_modest_number_l651_651213

-- Definitions and Conditions
def is_modest (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  5 * a = b + c + d ∧
  d % 2 = 0

def G (a b c d : ℕ) : ℕ :=
  (1000 * a + 100 * b + 10 * c + d - (1000 * c + 100 * d + 10 * a + b)) / 99

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_divisible_by_3 (abc : ℕ) : Prop :=
  abc % 3 = 0

-- Theorem statement
theorem maximum_modest_number :
  ∃ a b c d : ℕ, is_modest a b c d ∧ is_divisible_by_11 (G a b c d) ∧ is_divisible_by_3 (100 * a + 10 * b + c) ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 3816 := 
sorry

end maximum_modest_number_l651_651213


namespace find_x_exists_unique_l651_651998

theorem find_x_exists_unique (n : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ x = p * q * r) (h3 : 11 ∣ x) : x = 59048 :=
sorry

end find_x_exists_unique_l651_651998


namespace quadratic_function_coefficient_not_zero_l651_651651

theorem quadratic_function_coefficient_not_zero (m : ℝ) : (∀ x : ℝ, (m-2)*x^2 + 2*x - 3 ≠ 0) → m ≠ 2 :=
by
  intro h
  by_contra h1
  exact sorry

end quadratic_function_coefficient_not_zero_l651_651651


namespace rectangular_prism_paths_A_to_C1_l651_651018

def rectangular_prism_paths (rect: Set (Set String)) (A C1: String) : ℕ :=
  -- fictitious function representing counting valid paths
  sorry

theorem rectangular_prism_paths_A_to_C1 (rect: Set (Set String)) (A C1: String) (H1: rectangular_prism_rect rect) (H2: symmetric_about_center rect A C1):
  rectangular_prism_paths rect A C1 = 18 := sorry

end rectangular_prism_paths_A_to_C1_l651_651018


namespace circles_tangent_and_through_point_l651_651944

lemma circle_locus_and_line_slope {x y m : ℝ} :
  (∀ Q : ℝ × ℝ, Q = (-√2, 0) → 
   (∃ c1 c2 : ℝ, ∀ P : ℝ × ℝ, (P.1^2 + P.2^2 - 2 * √2 * P.1 - 10 = 0) → 
   ((P.1^2 + P.2^2 - 2 * √2 * P.1 - 10 = 0) ∧ 
    ∃ f g : ℝ, ((f^2 / 3 + g^2 = 1) ∧ 
    (∀ (x y : ℝ), (y = √3 * x + m) → 
    (let A : ℝ × ℝ := (x, y) in 
     let B : ℝ × ℝ := (x, y) in 
      ∀ (AB_mid : ℝ × ℝ), 
      (AB_mid = ((-√3/10 * m), (m/10))) → 
      (y - (m / 10) = -√3 / 3 * ((x) + (3√3 / 10 * m))) → 
      ∃ m' : ℝ, m' = 5 / 6 ∧ 
                (y = √3 * x + m'))))))
by 
sorries

theorem circles_tangent_and_through_point :
  ∀ Q : ℝ × ℝ, Q = (-√2, 0) →
  ∀ P1 P2 : ℝ,
    (x^2 + y^2 - 2 * √2 * x - 10 = 0) →
    (x^2 / 3 + y^2 = 1) →
    ∀ p1 p2 : ℝ,
      (p1 / 2 + p2 = 2 * √2) →
        (y = √3 * x + 5 / 6)
by
  sorry

end circles_tangent_and_through_point_l651_651944


namespace alyosha_cube_cut_l651_651502

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651502


namespace necessary_but_not_sufficient_condition_l651_651739

variable (x : ℝ)

theorem necessary_but_not_sufficient_condition (h1 : 2 - x ≥ 0) :
  (|x - 1| ≤ 1) → (h1).necessary_but_not_sufficient := by
  sorry

end necessary_but_not_sufficient_condition_l651_651739


namespace goldfish_growth_solution_l651_651930

def goldfish_growth_problem : Prop :=
  ∃ n : ℕ, 
    (∀ k, (k < n → 3 * (5:ℕ)^k ≠ 243 * (3:ℕ)^k)) ∧
    3 * (5:ℕ)^n = 243 * (3:ℕ)^n

theorem goldfish_growth_solution : goldfish_growth_problem :=
sorry

end goldfish_growth_solution_l651_651930


namespace trajectory_center_circle_equation_circle_equation_1_l651_651681

noncomputable def R : ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

def condition1 : Prop := R^2 - b^2 = 2
def condition2 : Prop := R^2 - a^2 = 3

theorem trajectory_center (h1 : condition1) (h2 : condition2) : b^2 - a^2 = 1 :=
sorry

def distance_condition : Prop := |b - a| = 1

theorem circle_equation (h1 : condition1) (h2 : condition2) (h3 : distance_condition) : 
  (a = 0 ∧ b = 1 ∧ R = sqrt 3) ∨ (a = 0 ∧ b = -1 ∧ R = sqrt 3) :=
sorry

theorem circle_equation_1 (h1 : condition1) (h2 : condition2) (h3 : distance_condition) : 
  (∀ (x y: ℝ), (x^2 + (y - 1)^2 = 3) ∨ (x^2 + (y + 1)^2 = 3)) :=
sorry

end trajectory_center_circle_equation_circle_equation_1_l651_651681


namespace sqrt_16_eq_pm_4_l651_651819

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l651_651819


namespace water_heater_ratio_l651_651845

variable (Wallace_capacity : ℕ) (Catherine_capacity : ℕ)
variable (Wallace_fullness : ℚ := 3/4) (Catherine_fullness : ℚ := 3/4)
variable (total_water : ℕ := 45)

theorem water_heater_ratio :
  Wallace_capacity = 40 →
  (Wallace_fullness * Wallace_capacity : ℚ) + (Catherine_fullness * Catherine_capacity : ℚ) = total_water →
  ((Wallace_capacity : ℚ) / (Catherine_capacity : ℚ)) = 2 :=
by
  sorry

end water_heater_ratio_l651_651845


namespace cube_cut_problem_l651_651491

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651491


namespace constant_term_binomial_expansion_l651_651796

theorem constant_term_binomial_expansion :
  let f (x : ℚ) := (2 * x - 1 / (2 * x)) ^ 6 in
  ∃ c : ℚ, (∀ x : ℚ, f x = c + P(x)) ∧ c = -20 :=
by
  let f (x : ℚ) := (2 * x - 1 / (2 * x)) ^ 6
  let c := -20
  have H1 : ∀ x, f x = c + P(x) := sorry
  exact ⟨c, H1, rfl⟩

end constant_term_binomial_expansion_l651_651796


namespace sets_in_proportion_l651_651000

/-- Check if four line segments are in proportion by verifying if 
the product of the extremes equals the product of the means. -/
def in_proportion (a b c d : ℕ) : Prop :=
  a * d = b * c

/-- Determine the sets of line segments that are in proportion -/
theorem sets_in_proportion :
  ({5, 15, 3, 9} ⊆ {4, 6, 8, 10, 3, 4, 5, 6, 5, 15, 3, 9, 8, 6, 2, 1}) ∧
  (in_proportion 5 15 3 9) :=
by
  sorry

end sets_in_proportion_l651_651000


namespace janina_cover_expenses_l651_651261

theorem janina_cover_expenses : 
  ∀ (rent supplies price_per_pancake : ℕ), 
    rent = 30 → 
    supplies = 12 → 
    price_per_pancake = 2 → 
    (rent + supplies) / price_per_pancake = 21 := 
by 
  intros rent supplies price_per_pancake h_rent h_supplies h_price_per_pancake 
  rw [h_rent, h_supplies, h_price_per_pancake]
  sorry

end janina_cover_expenses_l651_651261


namespace find_n_l651_651483

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651483


namespace bc_inequality_l651_651299

theorem bc_inequality (a b c : ℝ) (h1 : sqrt a ≥ sqrt (b * c)) (h2 : sqrt (b * c) ≥ sqrt a - c) :
  b * c ≥ b + c :=
sorry

end bc_inequality_l651_651299


namespace meet_time_difference_computation_l651_651241

noncomputable def meet_time_difference
  (green_height : ℕ) 
  (stata_height : ℕ) 
  (distance_between : ℕ) 
  (zipline_speed : ℕ) : ℕ :=
  let x := (4/3) * (distance_between.to_real / ((4/3) + (3/4)).to_real) in
  let y := (3/4) * (distance_between.to_real / ((4/3) + (3/4)).to_real) in
  let ben_distance := Real.sqrt (x^2 + green_height^2) in
  let jerry_distance := Real.sqrt (y^2 + stata_height^2) in
  ((ben_distance / zipline_speed) - (jerry_distance / zipline_speed)).to_real.to_nat

theorem meet_time_difference_computation : 
  meet_time_difference 160 90 120 10 = 740 :=
by
  sorry

end meet_time_difference_computation_l651_651241


namespace incenter_inequality_l651_651015

theorem incenter_inequality {A B C Fa Fb Fc O : Type} 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace Fa] [MetricSpace Fb] [MetricSpace Fc] [MetricSpace O] 
  (hA : A → MetricSpace)
  (hB : B → MetricSpace)
  (hC : C → MetricSpace)
  (hFa : Fa → A)
  (hFb : Fb → B)
  (hFc : Fc → C)
  (hO : O = Center (InCircle A B C)) 
  (hFa_def: IsIntersectionOfAngleBisectors A B C Fa) 
  (hFb_def: IsIntersectionOfAngleBisectors B A C Fb) 
  (hFc_def: IsIntersectionOfAngleBisectors C A B Fc) :
  (dist O A / dist O Fa) + (dist O B / dist O Fb) + (dist O C / dist O Fc) ≥ 3 := by
  sorry

end incenter_inequality_l651_651015


namespace sum_binary_digits_l651_651300

theorem sum_binary_digits (n : ℕ) : 
    n - (Nat.div n 2 + Nat.div n 4 + Nat.div n 8 + Nat.div n 16 + ...) = 
    (Nat.digits 2 n).sum := 
    sorry

end sum_binary_digits_l651_651300


namespace number_of_integers_become_one_after_9_operations_is_55_l651_651118

-- Define the operation that specifies how the integer changes in each step
def operation (n : ℕ) : ℕ :=
if n % 2 = 0 then n / 2 else n + 1

-- Define a function that performs the operations recursively for a given number of steps
def perform_operations : ℕ → ℕ → ℕ
| 0, n := n
| (steps + 1), n := perform_operations steps (operation n)

-- Define a function to count the number of integers that become 1 after a specified number of operations
def count_integers_become_one (steps : ℕ) : ℕ :=
((List.range (steps * 10)).filter (λ n => perform_operations steps n = 1)).length

theorem number_of_integers_become_one_after_9_operations_is_55 :
  count_integers_become_one 9 = 55 :=
sorry

end number_of_integers_become_one_after_9_operations_is_55_l651_651118


namespace binomial_coefficient_expression_l651_651533

theorem binomial_coefficient_expression :
  (binomial (- 1 / 3) 2023 * 4^2023) / (binomial 4046 2023) =
  (-1)^(2022) * (2^4046 / 3^2023) * (nat.factorial 6053 / (nat.factorial 3030 * nat.factorial 4046)) :=
by
  sorry

end binomial_coefficient_expression_l651_651533


namespace positive_irrational_less_than_one_l651_651059

theorem positive_irrational_less_than_one : 
  ∃! (x : ℝ), 
    (x = (Real.sqrt 6) / 3 ∧ Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = -(Real.sqrt 3) / 3 ∧ Irrational x ∧ x < 0) ∨ 
    (x = 1 / 3 ∧ ¬Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = Real.pi / 3 ∧ Irrational x ∧ x > 1) :=
by
  sorry

end positive_irrational_less_than_one_l651_651059


namespace not_sufficient_nor_necessary_condition_l651_651167

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_increasing_for_nonpositive (f : ℝ → ℝ) : Prop :=
∀ x y, x ≤ 0 → y ≤ 0 → x < y → f x < f y

theorem not_sufficient_nor_necessary_condition
  {f : ℝ → ℝ}
  (hf_even : is_even_function f)
  (hf_incr : is_increasing_for_nonpositive f)
  (x : ℝ) :
  (6/5 < x ∧ x < 2) → ¬((1 < x ∧ x < 7/4) ↔ (f (Real.log (2 * x - 2) / Real.log 2) > f (Real.log (2 / 3) / Real.log (1 / 2)))) :=
sorry

end not_sufficient_nor_necessary_condition_l651_651167


namespace inequality_abc_l651_651602

theorem inequality_abc 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c ≤ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
by sorry

end inequality_abc_l651_651602


namespace highest_water_level_changes_on_tuesday_l651_651825

def water_levels : List (String × Float) :=
  [("Monday", 0.03), ("Tuesday", 0.41), ("Wednesday", 0.25), ("Thursday", 0.10),
   ("Friday", 0.0), ("Saturday", -0.13), ("Sunday", -0.2)]

theorem highest_water_level_changes_on_tuesday :
  ∃ d : String, d = "Tuesday" ∧ ∀ d' : String × Float, d' ∈ water_levels → d'.snd ≤ 0.41 := by
  sorry

end highest_water_level_changes_on_tuesday_l651_651825


namespace probability_of_at_least_one_dollar_l651_651887

-- Defining the conditions: number of each type of coin
def num_pennies : ℕ := 3
def num_nickels : ℕ := 5
def num_dimes : ℕ := 4
def num_quarters : ℕ := 2
def total_coins : ℕ := num_pennies + num_nickels + num_dimes + num_quarters
def num_drawn_coins : ℕ := 8

-- The probability to prove
def prob_at_least_one_dollar (successes total_outcomes: ℕ) : ℚ :=
  (successes : ℚ) / (total_outcomes : ℚ)

-- Total outcomes for drawing 8 coins without replacement from 14
noncomputable def total_outcomes : ℕ := nat.choose total_coins num_drawn_coins

-- Calculated number of successful outcomes
def successful_outcomes : ℕ := 1596

-- Proof statement: Probability that the value of drawn coins is at least $1
theorem probability_of_at_least_one_dollar :
  prob_at_least_one_dollar successful_outcomes total_outcomes = 1596 / 3003 := by
  sorry

end probability_of_at_least_one_dollar_l651_651887


namespace multiply_correct_l651_651557

theorem multiply_correct : 2.4 * 0.2 = 0.48 := by
  sorry

end multiply_correct_l651_651557


namespace triangle_area_solution_l651_651608

-- Define the triangle ABC with given conditions and properties
noncomputable def triangle_area_problem : ℝ :=
  let CE := 2
  let angle_BAC := 60
  let A := (0, 2) -- Point A is positioned at (0,2)
  let B := (sqrt 3, 0) -- B is positioned to make the given configuration
  let C := (0, 0)
  let area := 1 / 2 * 4 * 2 -- 1/2 * base * height
  in area

-- Lean theorem stating the problem with the expected result
theorem triangle_area_solution : triangle_area_problem = 4 := 
by {
  sorry
}

end triangle_area_solution_l651_651608


namespace cyclic_quadrilateral_projection_sum_eq_l651_651778

/-
Quadrilateral ABCD is inscribed in a circle.
Diagonals AC and BD intersect at point P.
The projections of P onto the sides AB, BC, CD, and DA are E, F, G, and H respectively.
We need to prove EH + FG = EF + HG.
-/
theorem cyclic_quadrilateral_projection_sum_eq
  (A B C D P E F G H : Point) 
  (h1 : InscribedQuadrilateral A B C D)
  (h2 : Line A C ∩ Line B D = {P})
  (h3 : Projection P A B E)
  (h4 : Projection P B C F)
  (h5 : Projection P C D G)
  (h6 : Projection P D A H) :
  distance E H + distance F G = distance E F + distance H G :=
sorry

end cyclic_quadrilateral_projection_sum_eq_l651_651778


namespace mark_total_votes_l651_651764

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l651_651764


namespace remainder_17_pow_63_mod_7_l651_651379

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651379


namespace obtuse_triangle_of_ratio_l651_651675

noncomputable def triangle_angles (total_angle : ℝ) (ratios : list ℝ) : list ℝ :=
  ratios.map (λ r, r * total_angle / (ratios.sum))

theorem obtuse_triangle_of_ratio {α β γ : ℝ} (hαβγ : α / β = 2 / 3) (hβγ : β / γ = 3 / 7) 
                                (h_sum : α + β + γ = 180) : 
  max α (max β γ) > 90 :=
by
  have ratios : list ℝ := [2, 3, 7]
  have angles := triangle_angles 180 ratios
  have h1 : angles.sum = 180 := sorry
  have h2 : angles = [30, 45, 105] := sorry
  show max 30 (max 45 105) > 90
  sorry

end obtuse_triangle_of_ratio_l651_651675


namespace difference_between_correct_and_girls_answer_l651_651902

theorem difference_between_correct_and_girls_answer :
  let n := 134 in
  let correct_answer := n * 43 in
  let girls_answer := n * 34 in
  correct_answer - girls_answer = 1206 :=
by
  sorry

end difference_between_correct_and_girls_answer_l651_651902


namespace parabola_opens_upwards_l651_651337

-- The condition defining the parabolic curve equation 
def parabolic_curve (x : ℝ) : ℝ := 2 * (x + 3)^2 - 3

-- The main theorem stating that the curve opens upwards
theorem parabola_opens_upwards : ∀ x : ℝ, (2 > 0) → ∃ d, d = parabolic_curve x ∧ d ≥ -3 :=
by
  intro x h
  exists (parabolic_curve x)
  split
  · refl
  · sorry

end parabola_opens_upwards_l651_651337


namespace ratio_of_times_l651_651430

theorem ratio_of_times (A_work_time B_combined_rate : ℕ) 
  (h1 : A_work_time = 6) 
  (h2 : (1 / (1 / A_work_time + 1 / (B_combined_rate / 2))) = 2) :
  (B_combined_rate : ℝ) / A_work_time = 1 / 2 :=
by
  -- below we add the proof part which we will skip for now with sorry.
  sorry

end ratio_of_times_l651_651430


namespace solve_AlyoshaCube_l651_651512

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651512


namespace polynomial_double_root_l651_651446

theorem polynomial_double_root (b₄ b₃ b₂ b₁ t : ℤ) :
  ∀ s : ℤ, (polynomial.of_fn [1, b₄, b₃, b₂, b₁, 24]).eval t = 0 ∧
  polynomial.derivative (polynomial.of_fn [1, b₄, b₃, b₂, b₁, 24]).eval t = 0 ↔
  t ∈ {-4, -3, -2, -1, 1, 2, 3, 4} := 
by sorry

end polynomial_double_root_l651_651446


namespace exist_two_numbers_with_GCD_and_LCM_l651_651957

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem exist_two_numbers_with_GCD_and_LCM :
  ∃ A B : ℕ, GCD A B = 21 ∧ LCM A B = 3969 ∧ ((A = 21 ∧ B = 3969) ∨ (A = 147 ∧ B = 567)) :=
by
  sorry

end exist_two_numbers_with_GCD_and_LCM_l651_651957


namespace find_b_l651_651640

variables {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def is_on_hyperbola (P : R × R) : Prop := P.1 ^ 2 - (P.2 ^ 2) / 2 = 1
def is_symmetric (P Q : R × R) (b : R) : Prop := Q.2 = -Q.1 + b + (P.2 + P.1)
def is_on_parabola (P : R × R) : Prop := P.2 ^ 2 = 8 * P.1
def midpoint (P Q : R × R) : R × R := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Main statement
theorem find_b (P Q : R × R) (b : R) :
  is_on_hyperbola P →
  is_on_hyperbola Q →
  is_symmetric P Q b →
  is_on_parabola (midpoint P Q) →
  b = 0 ∨ b = 6 :=
by sorry

end find_b_l651_651640


namespace problem_statement_l651_651625

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * cos (ω * x + π / 6)

theorem problem_statement (ω : ℝ) (hx : ω = 2 ∨ ω = -2) :
  f ω (π / 3) = -3 ∨ f ω (π / 3) = 0 := by
  unfold f
  cases hx with
  | inl w_eq => sorry
  | inr w_eq => sorry

end problem_statement_l651_651625


namespace triangle_RS_length_l651_651666

/-- Given triangle XYZ with side lengths XY = 8, YZ = 9, and XZ = 10,
let XK be an altitude, and let M and N be points on sides XZ and XY
respectively, such that YM and ZN are angle bisectors intersecting XK
at points S and R respectively. Then, the length of RS is 1.8. -/
theorem triangle_RS_length
  (X Y Z K M N S R : Type*) [metric_space X] [metric_space Y] [metric_space Z] [metric_space K]
  [metric_space M] [metric_space N] [metric_space S] [metric_space R]
  (XY YZ XZ : ℝ)
  (hXY : XY = 8) (hYZ : YZ = 9) (hXZ : XZ = 10)
  (h_altitude_XK : altitude X K XZ)
  (h_point_M : on_segment M X Z)
  (h_point_N : on_segment N X Y)
  (h_bisector_YM : angle_bisector X Y M)
  (h_bisector_ZN : angle_bisector X Z N)
  (h_intersect_S : intersects_at S X K YM)
  (h_intersect_R : intersects_at R X K ZN) :
  distance R S = 1.8 := sorry

end triangle_RS_length_l651_651666


namespace remainder_of_17_pow_63_mod_7_l651_651372

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651372


namespace subset_solution_l651_651719

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l651_651719


namespace max_ab_l651_651286

theorem max_ab (a b : ℝ) (f : ℝ → ℝ) (h : f = λ x, a * x + b)
    (h_condition : ∀ x ∈ Icc 0 1, abs (f x) ≤ 1) : ab ≤ 1/4 := 
sorry

end max_ab_l651_651286


namespace equivalent_statements_l651_651857

variables (P Q : Prop)

theorem equivalent_statements : (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by
  -- Proof goes here
  sorry

end equivalent_statements_l651_651857


namespace product_value_l651_651084

theorem product_value :
  (\(\prod_{n=3}^{12} \left(1 - \frac{1}{n^3}\right) = \frac{11}{6}\) := 
sorry

end product_value_l651_651084


namespace monotonic_increasing_iff_l651_651174

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + 1 / x

theorem monotonic_increasing_iff (a : ℝ) :
  (∀ x : ℝ, 1 < x → f a x ≥ f a 1) ↔ a ≥ 1 :=
by
  sorry

end monotonic_increasing_iff_l651_651174


namespace pebble_product_sum_l651_651847

theorem pebble_product_sum:
  ( n : ℕ ) ( h : n = 1001 ) : 
  ( ∃ product_sum : ℕ, 
      ( ∀ a b : ℕ, a + b = n → a * b + pebble_product_sum = (n * (n - 1)) / 2 ) 
     ) :=
begin
  -- given n = 1001
  let n : ℕ := 1001,
  existsi (500500 : ℕ),
  -- proof will be established here
  sorry,
end

end pebble_product_sum_l651_651847


namespace zero_not_in_range_of_g_l651_651731

def g (x : ℝ) : ℤ :=
  if x > -3 then
    Int.ceil (2 / (x + 3))
  else 
    Int.floor (2 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ x : ℝ, g x = 0 :=
sorry

end zero_not_in_range_of_g_l651_651731


namespace largest_invertible_interval_for_g_l651_651350

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 2

theorem largest_invertible_interval_for_g :
  ∃ I : Set ℝ, (1 ∈ I ∧ (∀ x1 x2 ∈ I, g x1 = g x2 → x1 = x2) ∧
  (∀ x ∈ I, ∃ y ∈ I, g y = x) ∧
  (∀ x ∈ I, g x = 3 * (x - 1)^2 - 5) ∧
  I = Set.Iic 1) :=
sorry

end largest_invertible_interval_for_g_l651_651350


namespace scalar_square_of_vector_a_magnitude_of_vector_a_l651_651569

def vector_a : ℝ × ℝ × ℝ := (2, -1, -2)

theorem scalar_square_of_vector_a :
  let (a1, a2, a3) := vector_a in
  a1^2 + a2^2 + a3^2 = 9 := by
  sorry

theorem magnitude_of_vector_a :
  let (a1, a2, a3) := vector_a in
  real.sqrt (a1^2 + a2^2 + a3^2) = 3 := by
  sorry

end scalar_square_of_vector_a_magnitude_of_vector_a_l651_651569


namespace find_lambda_l651_651149

variables {A B C D : Type} [add_comm_group D] [vector_space ℝ D]
variables (A B C : D)

-- Condition: Points A, B, and C are collinear
def collinear (A B C : D) : Prop := ∃ (r s : ℝ), A = r • B + s • C

-- Given: vector equation
def vector_eq (A B C D : D) (λ : ℝ) :=
  (D - A : D) = 2 * λ • (D - B) + 3 • (C - B)

-- The main theorem to prove
theorem find_lambda (A B C D : D) (h_collinear : collinear A B C) (h_vector_eq : vector_eq A B C D λ) :
  λ = 1 / 2 :=
sorry

end find_lambda_l651_651149


namespace sum_of_divisors_360_l651_651399

theorem sum_of_divisors_360 : 
  (∑ d in (finset.filter (λ x, 360 % x = 0) (finset.range (360 + 1))), d) = 1170 :=
sorry

end sum_of_divisors_360_l651_651399


namespace range_of_a_l651_651624

theorem range_of_a 
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x < 0, f x = a^x)
  (h2 : ∀ x ≥ 0, f x = (a - 3) * x + 4 * a)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  0 < a ∧ a ≤ 1 / 4 :=
sorry

end range_of_a_l651_651624


namespace each_car_selected_exactly_three_times_l651_651048

-- Definition to state the number of cars and clients and their selections.
def cars : ℕ := 10
def clients : ℕ := 15

-- Definition to state the number of times each car is selected.
def selections := list ℕ

-- The total number of selections made by clients, and the expected total number of selections.
axiom selections_made_by_clients (s : selections) : s.length = cars → (∑ x in s, x) = clients * 2

-- Proof statement that each car was selected exactly 3 times.
theorem each_car_selected_exactly_three_times (s : selections) :
  s.length = cars → (∑ x in s, x) = 30 → (∀ x ∈ s, x = 3) :=
sorry

end each_car_selected_exactly_three_times_l651_651048


namespace surrounding_circles_radius_l651_651893

theorem surrounding_circles_radius (r : ℝ) : 
  let center_inner_circle := 2
  let center_outer_circle := r
  (∃ (ABC : Triangle), ABC.isEquilateral ∧ ABC.hasSides 2r ∧ 
   O.B == 2 + r ∧
   ABC.altitude == (O.B^2 == OA^2 + AB^2/4)) → 
  r = 2 * sqrt 3 + 4 :=
by {
  sorry
}

end surrounding_circles_radius_l651_651893


namespace remainder_349_div_13_l651_651405

theorem remainder_349_div_13 : 349 % 13 = 11 := 
by 
  sorry

end remainder_349_div_13_l651_651405


namespace pow_mod_seventeen_l651_651385

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651385


namespace even_integers_between_l651_651645

theorem even_integers_between :
  let a := 12 / 3
  let b := 50 / 2
  ∃ n : Nat, n = 10 ∧ ∀ x, x > a ∧ x ≤ b ∧ x % 2 = 0 → x ∈ {6, 8, 10, 12, 14, 16, 18, 20, 22, 24} :=
by
  let a := 4
  let b := 25
  use 10
  sorry

end even_integers_between_l651_651645


namespace excluded_angle_sum_1680_degrees_l651_651913

theorem excluded_angle_sum_1680_degrees (sum_except_one : ℝ) (h : sum_except_one = 1680) : 
  (180 - (1680 % 180)) = 120 :=
by
  have mod_eq : 1680 % 180 = 60 := by sorry
  rw [mod_eq]

end excluded_angle_sum_1680_degrees_l651_651913


namespace number_of_terminal_zeros_in_product_is_two_l651_651201

theorem number_of_terminal_zeros_in_product_is_two
  (h1 : 75 = 5^2 * 3)
  (h2 : 180 = 2^2 * 3^2 * 5) :
  ∃ n : ℕ, n = 2 ∧ ∀ (a b : ℕ), (75 * 180 = a * 5^b) → n = min (nat.factorization 5 a) (nat.factorization 2 a) :=
sorry

end number_of_terminal_zeros_in_product_is_two_l651_651201


namespace mark_total_votes_l651_651758

theorem mark_total_votes (h1 : 70% = 0.70) (h2 : 100000 : ℕ) (h3 : twice := 2)
  (votes_first_area : ℕ := 0.70 * 100000)
  (votes_remaining_area : ℕ := twice * votes_first_area)
  (total_votes : ℕ := votes_first_area + votes_remaining_area) : 
  total_votes = 210000 := 
by
  sorry

end mark_total_votes_l651_651758


namespace janina_must_sell_21_pancakes_l651_651260

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l651_651260


namespace even_factors_count_l651_651748

noncomputable def n : ℕ := 2^4 * 3^3 * 5^2 * 7

theorem even_factors_count : 
  (∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1 → 
   ∃ k, k = 2^a * 3^b * 5^c * 7^d ∧ k ∣ n) → 
  finset.card { x : ℕ | x ∣ n ∧ x % 2 = 0 } = 96 := 
by
  sorry

end even_factors_count_l651_651748


namespace cube_cut_problem_l651_651492

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651492


namespace max_marks_l651_651780

-- Define the conditions
def passing_marks (M : ℕ) : ℕ := 40 * M / 100

def Ravish_got_marks : ℕ := 40
def marks_failed_by : ℕ := 40

-- Lean statement to prove
theorem max_marks (M : ℕ) (h : passing_marks M = Ravish_got_marks + marks_failed_by) : M = 200 :=
by
  sorry

end max_marks_l651_651780


namespace second_order_derivative_l651_651976

open Real

noncomputable def x (t : ℝ) : ℝ := sinh t ^ 2
noncomputable def y (t : ℝ) : ℝ := 1 / cosh t ^ 2

theorem second_order_derivative:
  ∀ t : ℝ, 
  let y_xx := (2 / (cosh t ^ 6)) 
  in derivative (λ x, y_xx) t = y_xx := 
by
  sorry

end second_order_derivative_l651_651976


namespace trajectory_eq_midpoint_dot_product_val_l651_651598

-- Problem 1

def circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 16
def point_A : ℝ × ℝ := (10, 0)
def midpoint_of_AP (x y x₀ y₀ : ℝ) : Prop := x = (x₀ + 10) / 2 ∧ y = y₀ / 2

theorem trajectory_eq_midpoint :
  ∀ x y : ℝ, ∃ x₀ y₀ : ℝ, circle x₀ y₀ → midpoint_of_AP x y x₀ y₀ → (x - 6)^2 + (y - 1)^2 = 4 :=
by sorry

-- Problem 2

def line_l (x y k : ℝ) : Prop := k * x - y - 10 * k = 0
def intersects_circle (x₁ y₁ x₂ y₂ k : ℝ) : Prop := (circle x₁ y₁ ∧ line_l x₁ y₁ k) ∧ (circle x₂ y₂ ∧ line_l x₂ y₂ k)

theorem dot_product_val :
  ∀ (x₁ y₁ x₂ y₂ k : ℝ), intersects_circle x₁ y₁ x₂ y₂ k → 
  (let AM := (x₁ - 10, y₁); AN := (x₂ - 10, y₂) in 
   AM.1 * AN.1 + AM.2 * AN.2) = 48 :=
by sorry

end trajectory_eq_midpoint_dot_product_val_l651_651598


namespace sum_of_midpoint_coordinates_l651_651114

-- Define the coordinates of the endpoints
def point1 : (ℤ × ℤ) := (8, -2)
def point2 : (ℤ × ℤ) := (-4, 10)

-- Define the function to calculate the midpoint of two points
def midpoint (p1 p2 : (ℤ × ℤ)) : (ℤ × ℤ) :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

-- Calculate the sum of the coordinates of the midpoint
def sum_of_coordinates (p : (ℤ × ℤ)) : ℤ :=
  let (x, y) := p
  x + y

-- The Lean statement to prove
theorem sum_of_midpoint_coordinates :
  sum_of_coordinates (midpoint point1 point2) = 6 :=
by
  -- We do not provide the proof here, add sorry to skip it.
  sorry

end sum_of_midpoint_coordinates_l651_651114


namespace not_in_range_of_g_l651_651734

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else undefined

theorem not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g(x) ≠ 0 :=
by sorry

end not_in_range_of_g_l651_651734


namespace part1_part2_l651_651192

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (-Real.sin x, 2)
def b (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

-- Define the function f
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Problem (1)
theorem part1 : f (π / 6) = Real.sqrt 3 - 1 / 2 :=
by
  sorry

-- Problem (2)
def g (x : ℝ) : ℝ := 
  (Real.sin (π + x) + 4 * Real.cos (2 * π - x)) / 
  (Real.sin (π / 2 - x) - 4 * Real.sin (-x))

theorem part2 (h : f(x) = 0) : g x = 2 / 9 :=
by
  sorry

end part1_part2_l651_651192


namespace find_m_l651_651987

open Real

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) (m : ℝ) : ℝ :=
  2 * cos (ω * x + ϕ) + m

theorem find_m (ω ϕ : ℝ) (hω : 0 < ω)
  (symmetry : ∀ t : ℝ,  f (π / 4 - t) ω ϕ m = f t ω ϕ m)
  (f_π_8 : f (π / 8) ω ϕ m = -1) :
  m = -3 ∨ m = 1 := 
sorry

end find_m_l651_651987


namespace customers_at_start_l651_651056

def initial_customers (X : ℕ) : Prop :=
  let first_hour := X + 3
  let second_hour := first_hour - 6
  second_hour = 12

theorem customers_at_start {X : ℕ} : initial_customers X → X = 15 :=
by
  sorry

end customers_at_start_l651_651056


namespace find_x_with_conditions_l651_651994

theorem find_x_with_conditions (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1)
  (h2 : (nat.factors x).to_finset.card = 3) (h3 : 11 ∈ (nat.factors x).to_finset) :
  x = 59048 := 
by {
  sorry
}

end find_x_with_conditions_l651_651994


namespace part1_hire_candidate_A_part2_hire_candidate_B_l651_651032

-- Conditions for the problem
def candidate_A_written_test_score : ℝ := 85
def candidate_A_interview_score : ℝ := 75
def candidate_B_written_test_score : ℝ := 60
def candidate_B_interview_score : ℝ := 95

-- Part (1) Equivalent
theorem part1_hire_candidate_A :
  let average_A := (candidate_A_written_test_score + candidate_A_interview_score) / 2
  let average_B := (candidate_B_written_test_score + candidate_B_interview_score) / 2
  average_A = 80 ∧ average_B = 77.5 → "A" = "A" :=
by
  let average_A := (candidate_A_written_test_score + candidate_A_interview_score) / 2
  let average_B := (candidate_B_written_test_score + candidate_B_interview_score) / 2
  have h1 : average_A = 80 := by sorry
  have h2 : average_B = 77.5 := by sorry
  exact ⟨h1, h2⟩

-- Part (2) Equivalent
theorem part2_hire_candidate_B :
  let weighted_average_A := (candidate_A_written_test_score * 0.4 + candidate_A_interview_score * 0.6)
  let weighted_average_B := (candidate_B_written_test_score * 0.4 + candidate_B_interview_score * 0.6)
  weighted_average_A = 79 ∧ weighted_average_B = 81 → "B" = "B" :=
by
  let weighted_average_A := (candidate_A_written_test_score * 0.4 + candidate_A_interview_score * 0.6)
  let weighted_average_B := (candidate_B_written_test_score * 0.4 + candidate_B_interview_score * 0.6)
  have h1 : weighted_average_A = 79 := by sorry
  have h2 : weighted_average_B = 81 := by sorry
  exact ⟨h1, h2⟩

end part1_hire_candidate_A_part2_hire_candidate_B_l651_651032


namespace max_value_of_S_a_l651_651418

noncomputable def max_S (n : ℕ) (m : ℝ) (x : fin n → ℝ) : ℝ :=
  ∑ i in finset.range n, ∑ j in finset.Ico 0 i, x i * x j

theorem max_value_of_S_a {n : ℕ} (h_n : 2 < n) {m : ℝ} (x : fin n → ℝ) 
  (h_x_neg : ∀ i, x i < 0) (h_sum_x : ∑ i, x i = m) :
  max_S n m x ≤ (n - 1) * m^2 / n :=
sorry

end max_value_of_S_a_l651_651418


namespace find_n_l651_651486

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651486


namespace train_cars_estimate_l651_651525

noncomputable def train_cars_count (total_time_secs : ℕ) (delay_secs : ℕ) (cars_counted : ℕ) (count_time_secs : ℕ): ℕ := 
  let rate_per_sec := cars_counted / count_time_secs
  let cars_missed := delay_secs * rate_per_sec
  let cars_in_remaining_time := rate_per_sec * (total_time_secs - delay_secs)
  cars_missed + cars_in_remaining_time

theorem train_cars_estimate :
  train_cars_count 210 15 8 20 = 120 :=
sorry

end train_cars_estimate_l651_651525


namespace exists_large_coeff_polynomials_l651_651097

noncomputable def large_coeff_poly_existence : Prop :=
  ∃ (P Q : Polynomial ℤ), 
      (∃ a ∈ P.coeffs, |a| > 2015) ∧ 
      (∃ b ∈ Q.coeffs, |b| > 2015) ∧ 
      (∀ c ∈ (P * Q).coeffs, |c| ≤ 1)

theorem exists_large_coeff_polynomials :
  large_coeff_poly_existence :=
sorry

end exists_large_coeff_polynomials_l651_651097


namespace trig_identity_l651_651078

theorem trig_identity :
  (cos (45 * π / 180))^2 + (tan (60 * π / 180)) * (cos (30 * π / 180)) = 2 :=
by
  -- Define the trigonometric values based on the conditions
  have h1 : cos (45 * π / 180) = sqrt 2 / 2 := by sorry
  have h2 : tan (60 * π / 180) = sqrt 3 := by sorry
  have h3 : cos (30 * π / 180) = sqrt 3 / 2 := by sorry
  -- Use the values to show the identity
  calc (cos (45 * π / 180))^2 + (tan (60 * π / 180)) * (cos (30 * π / 180))
       = (sqrt 2 / 2)^2 + sqrt 3 * (sqrt 3 / 2) : by rw [h1, h2, h3]
   ... = 1 / 2 + 3 / 2 : by sorry
   ... = 2 : by sorry

end trig_identity_l651_651078


namespace train_length_l651_651457

theorem train_length (L : ℕ) (V : ℕ) (platform_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
    (h1 : V = L / time_pole) 
    (h2 : V = (L + platform_length) / time_platform) :
    L = 300 := 
by 
  -- The proof can be filled here
  sorry

end train_length_l651_651457


namespace power_of_prime_divisors_l651_651564

theorem power_of_prime_divisors (n : ℕ) (h_n : n ≥ 1) 
  (h_divisors : ∃ (d : ℕ → ℕ) (k : ℕ), 
    d 1 = 1 ∧ (∀ (i : ℕ), 1 ≤ i ∧ i < k → d i < d (i + 1)) ∧ 
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ k - 2 → d i ∣ (d (i + 1) + d (i + 2)))) : 
  ∃ (p : ℕ), nat.prime p ∧ ∃ (m : ℕ), n = p ^ m :=
sorry -- Proof not required as per the instructions

end power_of_prime_divisors_l651_651564


namespace James_age_is_47_5_l651_651068

variables (James_Age Mara_Age : ℝ)

def condition1 : Prop := James_Age = 3 * Mara_Age - 20
def condition2 : Prop := James_Age + Mara_Age = 70

theorem James_age_is_47_5 (h1 : condition1 James_Age Mara_Age) (h2 : condition2 James_Age Mara_Age) : James_Age = 47.5 :=
by
  sorry

end James_age_is_47_5_l651_651068


namespace t_shaped_slope_correct_l651_651232

def point := (ℝ × ℝ)

def vertices : list point := [(0,0), (0,5), (2,5), (2,2), (7,2), (7,0)]

def area_rectangle (p1 p2 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  abs ((x2 - x1) * (y2 - y1))

def area_T_shape (vertices : list point) : ℝ :=
  let rect1 := area_rectangle (vertices.nth 0) (vertices.nth 2)
  let rect2 := area_rectangle (vertices.nth 2) (vertices.nth 5)
  rect1 + rect2

def slope_of_line_through_origin_dividing_area_in_half (vertices : list point) : ℝ :=
  let total_area := area_T_shape vertices
  let half_area := total_area / 2
  -- Let's assume this is the slope we find by calculation
  4 / 5

theorem t_shaped_slope_correct : 
  slope_of_line_through_origin_dividing_area_in_half vertices = 4 / 5 :=
sorry

end t_shaped_slope_correct_l651_651232


namespace subset_A_B_l651_651714

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l651_651714


namespace medication_price_reduction_l651_651962

theorem medication_price_reduction (m x : ℝ) :
  let y := m * (1 - x) * (1 - x) in y = m * (1 - x)^2 :=
by 
  sorry

end medication_price_reduction_l651_651962


namespace no_inverse_for_congruent_triangle_angles_l651_651827

theorem no_inverse_for_congruent_triangle_angles :
  ∀ (T1 T2 : Triangle), 
  (congruent T1 T2 → corresponding_angles_equal T1 T2) ∧
  (∀ (T1 T2 : Triangle), corresponding_angles_equal T1 T2 → ¬ congruent T1 T2) := 
by 
  intros T1 T2 
  split 
  { intros h 
    exact corresponding_angles_equal_of_congruent h } 
  { intros h h'
    have := congruence_implies_corresponding_angles h' 
    exact h this }

end no_inverse_for_congruent_triangle_angles_l651_651827


namespace triangle_area_axis_asymptotes_parabola_hyperbola_l651_651563

theorem triangle_area_axis_asymptotes_parabola_hyperbola :
  let parabola := y^2 = 8 * x
  let hyperbola := (x^2 / 8) - (y^2 / 4) = 1
  let axis := x = -2
  let asymptote1 := y = (√2 / 2) * x
  let asymptote2 := y = -(√2 / 2) * x
  ∃ triangle_area: ℝ, 
    triangle_area = 2 * √2 :=
by
  -- The proof steps are omitted
  sorry

end triangle_area_axis_asymptotes_parabola_hyperbola_l651_651563


namespace probability_n_spades_in_13_draws_l651_651583

theorem probability_n_spades_in_13_draws (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 13) :
  let total_ways := Nat.choose 52 13,
      ways_to_choose_spades := Nat.choose 13 n,
      ways_to_choose_non_spades := Nat.choose 39 (13 - n),
      favorable_outcomes := ways_to_choose_spades * ways_to_choose_non_spades in
  Pn = favorable_outcomes / total_ways :=
sorry

end probability_n_spades_in_13_draws_l651_651583


namespace mark_total_votes_l651_651765

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l651_651765


namespace total_weight_of_fish_l651_651037

theorem total_weight_of_fish (fry : ℕ) (survival_rate : ℚ) 
  (first_catch : ℕ) (first_avg_weight : ℚ) 
  (second_catch : ℕ) (second_avg_weight : ℚ)
  (third_catch : ℕ) (third_avg_weight : ℚ)
  (total_weight : ℚ) :
  fry = 100000 ∧ 
  survival_rate = 0.95 ∧ 
  first_catch = 40 ∧ 
  first_avg_weight = 2.5 ∧ 
  second_catch = 25 ∧ 
  second_avg_weight = 2.2 ∧ 
  third_catch = 35 ∧ 
  third_avg_weight = 2.8 ∧ 
  total_weight = fry * survival_rate * 
    ((first_catch * first_avg_weight + 
      second_catch * second_avg_weight + 
      third_catch * third_avg_weight) / 100) / 10000 →
  total_weight = 24 :=
by
  sorry

end total_weight_of_fish_l651_651037


namespace proof_problem_l651_651931

def log_4 (x : ℝ) : ℝ := Real.log x / Real.log 4
def log_16 (x : ℝ) : ℝ := Real.log x / Real.log 16

noncomputable def problem : ℝ :=
  (∑ k in Finset.range 10, log_4 (2^(k.succ^2))) * (∑ k in Finset.range 50, log_16 (64^k.succ))

theorem proof_problem :
  problem = 2062.5 :=
sorry

end proof_problem_l651_651931


namespace area_of_triangle_on_ellipse_l651_651610

theorem area_of_triangle_on_ellipse
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h1 : P.1^2 / 25 + P.2^2 / 9 = 1)
  (h2 : let (a, b) := F1 in let (c, d) := F2 in (a - c)^2 + (b - d)^2 = (2 * 4)^2)
  (h3 : ∀ {x y z : ℝ}, angle (F1.1 - P.1, F1.2 - P.2) (F2.1 - P.1, F2.2 - P.2) = π / 3):
  sqrt 3 :=
sorry

end area_of_triangle_on_ellipse_l651_651610


namespace find_p_q_for_b_find_formula_and_check_m_type_l651_651573

-- Define the notation for sequence to be used.
def sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) = p * a n + q

-- Problem Part (I)
theorem find_p_q_for_b (p q : ℝ) (b : ℕ → ℝ) 
  (hb_def : ∀ n : ℕ, b n = 2 * n) : 
  (∀ n, b (n + 1) = p * b n + q) → 
  (p = 1 ∧ q = 2) := 
  sorry

-- Problem Part (II)
theorem find_formula_and_check_m_type 
  (c : ℕ → ℝ) 
  (hc1 : c 1 = 1) 
  (hc_rec : ∀ n : ℕ, c (n + 1) - c n = 2^n)   
  (hc_formula: ∀ n, c n = 2^n - 1): 
  (∃ p q : ℝ, ∀ n, c (n + 1) = p * c n + q) :=
  sorry

end find_p_q_for_b_find_formula_and_check_m_type_l651_651573


namespace EM_parallel_AC_l651_651147

variables {Point : Type} [AffineSpace Point]

-- Define A, B, C, D, E, M as points
variables (A B C D E M : Point)

-- Define the condition that ABCD is an isosceles trapezoid with AB parallel to CD
def isoscelesTrapezoid (A B C D : Point) [AffineSpace Point] : Prop :=
  ∃ (l1 l2 l3 : Line),
    l1.contains A ∧ l1.contains B ∧
    l2.contains C ∧ l2.contains D ∧
    l1.parallel l2 ∧
    ∃ (midC D_eq : Segment), midpointC D_eq = midpointA B_eq

-- Define the condition that E is the projection of D onto AB
def isProjection (E D A B : Point) [AffineSpace Point] : Prop :=
  isOrthogonal (Line.mk A B) (Perpendicular.mk A E D)

-- Define the condition that M is the midpoint of the diagonal BD
def isMidpoint (M B D : Point) [AffineSpace Point] : Prop :=
  midpoint B D = Some M

-- State the theorem with these conditions
theorem EM_parallel_AC
  (h1: isoscelesTrapezoid A B C D)
  (h2: isProjection E D A B)
  (h3: isMidpoint M B D) :
  parallelSegment M E A C :=
sorry

end EM_parallel_AC_l651_651147


namespace apple_pie_slices_bought_l651_651313

-- Define the constants for item prices and quantities
def cupcake_price := 2
def doughnut_price := 1
def apple_pie_price := 2
def cookie_price := 0.60

def num_cupcakes := 5
def num_doughnuts := 6
def num_cookies := 15

-- Define the total amount spent
def total_spent := 33

-- Define the sum of the costs calculated directly
def sum_cupcakes := num_cupcakes * cupcake_price
def sum_doughnuts := num_doughnuts * doughnut_price
def sum_cookies := num_cookies * cookie_price

-- Define the total spent on other items
def sum_other_items := sum_cupcakes + sum_doughnuts + sum_cookies

-- Calculate the amount spent on apple pie
def amount_spent_on_apple_pie := total_spent - sum_other_items

-- Calculate the number of apple pie slices bought
def num_apple_pie_slices := amount_spent_on_apple_pie / apple_pie_price

-- Theorem stating the number of apple pie slices Sophie bought
theorem apple_pie_slices_bought : num_apple_pie_slices = 4 :=
by 
  -- Definitions are used according to provided conditions, and the proof frame is created
  -- sorry to be replaced with the actual proof
  sorry

end apple_pie_slices_bought_l651_651313


namespace can_obtain_factorial_18_l651_651334

noncomputable def factorial (n : ℕ) : ℕ :=
nat.factorial n

def coprime (a b : ℕ) : Prop :=
nat.gcd a b = 1

def can_replace (a : ℕ) (d : ℕ) : Prop :=
10 ≤ d ∧ d ≤ 20 ∧ coprime a d

def replace_sequence (start : ℕ) (end_seq : ℕ) : Prop :=
∃ (n : ℕ) (seq : fin n → ℕ), 
  seq 0 = start ∧
  seq (fin.last n) = end_seq ∧ 
  ∀ i : fin (n - 1), 
    can_replace (seq i.val) ((seq (i + 1).val) - (seq i.val))

theorem can_obtain_factorial_18 :
  replace_sequence 1 (factorial 18) :=
sorry

end can_obtain_factorial_18_l651_651334


namespace total_tires_l651_651578

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l651_651578


namespace total_wheels_l651_651576

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l651_651576


namespace base_six_to_ten_2154_l651_651453

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  2 * 6^3 + 1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_six_to_ten_2154 :
  convert_base_six_to_ten 2154 = 502 :=
by
  sorry

end base_six_to_ten_2154_l651_651453


namespace not_all_squares_congruent_to_each_other_l651_651415

variable (Square : Type) [Equiangular Square] [Rectangle Square] [RegularPolygon Square] [Similar Square]

def congruent (a b : Square) : Prop := sorry -- Define the congruence relation

theorem not_all_squares_congruent_to_each_other (s1 s2 : Square) (h1 : s1 ≠ s2) : ¬ congruent s1 s2 :=
sorry

end not_all_squares_congruent_to_each_other_l651_651415


namespace remainder_17_pow_63_mod_7_l651_651374

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651374


namespace circle_center_radius_sum_l651_651275

theorem circle_center_radius_sum (x y a b r : ℝ) :
  (x^2 + 10*x + y^2 + 12*y + 57 = 0) →
  (x + 5)^2 + (y + 6)^2 = 4 →
  a = -5 →
  b = -6 →
  r = 2 →
  a + b + r = -9 :=
by
  intro h_eq h_standard_form ha hb hr
  have h_ab : a + b = -11, from sorry
  have h_ab_r := h_ab + hr
  exact h_ab_r

end circle_center_radius_sum_l651_651275


namespace complex_modulus_l651_651202

open Complex

theorem complex_modulus (z : ℂ) (h : (1 + I) * (1 - z) = 1) : abs z = sqrt 2 / 2 :=
sorry

end complex_modulus_l651_651202


namespace subset_a_eq_1_l651_651701

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651701


namespace lines_are_skew_l651_651106

def line1 (a t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * t, 3 + 4 * t, a + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * u, 2 + 2 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) :
  ¬(∃ t u : ℝ, line1 a t = line2 u) ↔ a ≠ 5 / 3 :=
sorry

end lines_are_skew_l651_651106


namespace unique_solution_eq_condition_l651_651094

theorem unique_solution_eq_condition (p q : ℝ) :
  (∃! x : ℝ, (2 * x - 2 * p + q) / (2 * x - 2 * p - q) = (2 * q + p + x) / (2 * q - p - x)) ↔ (p = 3 * q / 4 ∧ q ≠ 0) :=
  sorry

end unique_solution_eq_condition_l651_651094


namespace length_of_segment_QZ_l651_651687

-- Define the given conditions in terms of Lean.
variables {Point : Type} [linear_ordered_field Point]
variables (A B Y Z M Q : Point)

-- Conditions: AB // YZ, M is the midpoint of YZ, AZ = 54, BQ = 18, MQ = 30
variables (h_parallel : ∃ k : Point, A = Y + k * B) -- AB // YZ implies a linear combination
variables (h_midpoint : 2 * M = Y + Z) -- M midpoint of YZ
variables (h_AZ : AZ = 54)
variables (h_BQ : BQ = 18)
variables (h_MQ : MQ = 30)

-- Proof Goal: Length of segment QZ is 33.75
theorem length_of_segment_QZ : QZ = 33.75 :=
sorry

end length_of_segment_QZ_l651_651687


namespace intersection_points_l651_651234

def parametric_eq (t : Real) : Real × Real :=
  (t, 4 - t)

def polar_eq (theta : Real) : Real :=
  4 * Real.cos theta

theorem intersection_points :
  (∃ (t : Real), ∃ (theta1 theta2 : Real),
    polar_eq theta1 = 4 ∧ parametric_eq t = (4 * Real.cos theta1, 4 - 4 * Real.cos theta1) ∧
    (theta1 = 0 ∧ 4 * Real.cos theta1 = 4)) ∧
  (∃ (t : Real), ∃ (theta1 theta2 : Real),
    polar_eq theta2 = 2 * Real.sqrt 2 ∧ parametric_eq t = (2 * Real.sqrt 2 * Real.cos theta2, 4 - 2 * Real.sqrt 2 * Real.cos theta2) ∧
    (theta2 = Real.pi / 4 ∧ 2 * Real.sqrt 2 * Real.cos theta2 = 2 * Real.sqrt 2)) :=
sorry

end intersection_points_l651_651234


namespace price_percentage_combined_assets_l651_651005

variable (A B P : ℝ)

-- Conditions
axiom h1 : P = 1.20 * A
axiom h2 : P = 2 * B

-- Statement
theorem price_percentage_combined_assets : (P / (A + B)) * 100 = 75 := by
  sorry

end price_percentage_combined_assets_l651_651005


namespace find_lambda_l651_651148

variables {A B C D : Type} [add_comm_group D] [vector_space ℝ D]
variables (A B C : D)

-- Condition: Points A, B, and C are collinear
def collinear (A B C : D) : Prop := ∃ (r s : ℝ), A = r • B + s • C

-- Given: vector equation
def vector_eq (A B C D : D) (λ : ℝ) :=
  (D - A : D) = 2 * λ • (D - B) + 3 • (C - B)

-- The main theorem to prove
theorem find_lambda (A B C D : D) (h_collinear : collinear A B C) (h_vector_eq : vector_eq A B C D λ) :
  λ = 1 / 2 :=
sorry

end find_lambda_l651_651148


namespace remainder_problem_l651_651530

def rem (x y : ℚ) := x - y * (⌊x / y⌋ : ℤ)

theorem remainder_problem :
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  rem x y = (-19 : ℚ) / 63 :=
by
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  sorry

end remainder_problem_l651_651530


namespace part1_max_bc_part2_f_range_l651_651690

-- Definition of problem (1)
noncomputable def max_bc (b c : ℝ) (θ : ℝ) : Prop := 
  ∃ (a : ℝ), a = 4 ∧ 8 = b * c * Real.cos θ ∧ a^2 = b^2 + c^2 - 2 * b * c * (Real.cos θ)

theorem part1_max_bc : 
  max_bc 4 4 (Real.pi / 3) :=
sorry

-- Definition of problem (2)
noncomputable def f (θ : ℝ) : ℝ := 
  sqrt 3 * Real.sin (2 * θ) + Real.cos (2 * θ) - 1

theorem part2_f_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ Real.pi / 3) :
  0 ≤ f θ ∧ f θ ≤ 1 :=
sorry

end part1_max_bc_part2_f_range_l651_651690


namespace probability_distribution_40_rings_l651_651123

noncomputable def probability_after_40_rings : ℚ := 1 / 9

theorem probability_distribution_40_rings :
  let players_start := {players : Fin 4 → ℕ // ∀ i, players i = 1},
      ring_event := λ (players : Fin 4 → ℕ) (i : Fin 4) (j : Fin 4), 
                   if players i > 0 then (players i - 1, players j + 1)
                   else (players i, players j)
  in probability_after_40_rings = 1 / 9 :=
sorry

end probability_distribution_40_rings_l651_651123


namespace prob_inequalities_l651_651660

variables {Ω : Type*} [measurable_space Ω] (P : measure_theory.measure Ω)
variables (A B : set Ω)

theorem prob_inequalities (hA : measurable_set A) (hB : measurable_set B) :
  (P (A ∩ B) ≤ P A * P B ↔ P A = 0 ∨ P B = 0 ∨ P (A | B) ≤ P A ∨ P (B | A) ≤ P B) ∧ 
  (P (A ∩ B) ≥ P A * P B ↔ P A = 0 ∨ P B = 0 ∨ P (A | B) ≥ P A ∨ P (B | A) ≥ P B) :=
sorry

end prob_inequalities_l651_651660


namespace NationalDayDiscount_l651_651960

def spendVouchers (spending : ℕ) : ℕ :=
  (spending / 100) * 20

theorem NationalDayDiscount (initial_spending : ℕ) (h : initial_spending = 16000) :
  let total_voucher_value := spendVouchers initial_spending / (1 - 0.2) in
  let total_value := initial_spending + total_voucher_value in
  total_value = 20000 := 
by
  sorry

end NationalDayDiscount_l651_651960


namespace trapezoids_more_than_parallelograms_by_five_l651_651063

-- Define the parallel relationships:
variable (A B C D E F G H I J K : Type)
variable (Line : Type)
variable (parallel : Line → Line → Prop)

-- Given conditions
axiom h1 : parallel A B E F
axiom h2 : parallel A B G H
axiom h3 : parallel A B D C
axiom h4 : parallel A D I J
axiom h5 : parallel A J I K
axiom h6 : parallel A J B C

-- Conclusion to prove
theorem trapezoids_more_than_parallelograms_by_five : (trapezoids : ℕ) > (parallelograms : ℕ) :=
sorry

end trapezoids_more_than_parallelograms_by_five_l651_651063


namespace pow_mod_seventeen_l651_651390

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651390


namespace index_diff_of_sexes_l651_651574

theorem index_diff_of_sexes (n k : ℕ) (hn : n = 25) (hk : k = 8) : 
  (n - k) / n - (k / n) = 9 / 25 := 
by 
  rw [hn, hk] 
  sorry

end index_diff_of_sexes_l651_651574


namespace m_n_value_l651_651633

theorem m_n_value (m n : ℝ)
  (h1 : m * (-1/2)^2 + n * (-1/2) - 1/m < 0)
  (h2 : m * 2^2 + n * 2 - 1/m < 0)
  (h3 : m < 0)
  (h4 : (-1/2 + 2 = -n/m))
  (h5 : (-1/2) * 2 = -1/m^2) :
  m - n = -5/2 :=
sorry

end m_n_value_l651_651633


namespace tan_sin_difference_l651_651933

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end tan_sin_difference_l651_651933


namespace sum_of_divisors_360_l651_651396

-- Define the function to calculate the sum of divisors
def sum_of_divisors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

-- Statement of the problem
theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 :=
by
  sorry

end sum_of_divisors_360_l651_651396


namespace bernie_final_postcards_l651_651929

-- Define the initial conditions
def initial_postcards : ℕ := 18
def sold_euros : ℕ := 6
def sold_pounds : ℕ := 3
def sold_dollars : ℕ := 2

def euro_price : ℝ := 10
def pound_price : ℝ := 12
def dollar_price : ℝ := 15

def euro_to_dollar : ℝ := 1.20
def pound_to_dollar : ℝ := 1.35
def yen_to_dollar : ℝ := 1 / 110

def new_postcard_price : ℝ := 8
def yen_postcard_price : ℕ := 800

-- Prove that Bernie has 26 postcards after all his transactions
theorem bernie_final_postcards : 
  let initial_postcards : ℕ := 18
  let sold_postcards : ℕ := 6 + 3 + 2
  let euro_earnings : ℝ := sold_euros * euro_price
  let pound_earnings : ℝ := sold_pounds * pound_price
  let dollar_earnings : ℝ := sold_dollars * dollar_price
  let total_earnings_dollars : ℝ := (euro_earnings * euro_to_dollar)
                                     + (pound_earnings * pound_to_dollar)
                                     + dollar_earnings
  let spending_dollars : ℝ := total_earnings_dollars * 0.70
  let new_postcards : ℕ := floor (spending_dollars / new_postcard_price)
  let remaining_dollars : ℝ := total_earnings_dollars - (new_postcards * new_postcard_price)
  let remaining_yen : ℝ := remaining_dollars * 110
  let additional_postcards : ℕ := floor (remaining_yen / yen_postcard_price)
  in 
    initial_postcards - sold_postcards + new_postcards + additional_postcards = 26 :=
sorry

end bernie_final_postcards_l651_651929


namespace cos_ratio_value_l651_651171

-- Given Definitions
variables {a b x y : ℝ}
def ellipse (x y : ℝ) (a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a b : ℝ) := (a^2 - b^2) / a^2 = 3 / 4

-- The point P is on the ellipse
def point_on_ellipse (x y a b : ℝ) := ellipse x y a b ∧ a > b ∧ b > 0

-- The slopes of the lines PA and PB
def tan_alpha (x y a : ℝ) := y / (x + a)
def tan_beta (x y a : ℝ) := y / (x - a)
def cos_ratio (α β : ℝ) := (Real.cos (α - β)) / (Real.cos (α + β))

-- The final theorem statement
theorem cos_ratio_value (x y a b : ℝ) (P_on_ellipse : point_on_ellipse x y a b) 
  (e : eccentricity a b) :
  cos_ratio (tan_alpha x y a) (tan_beta x y a) = 3 / 5 :=
sorry

end cos_ratio_value_l651_651171


namespace average_minutes_of_lecture_heard_l651_651667

def total_attendees : ℕ := 200
def lecture_duration : ℕ := 90
def percentage_listened_entire : ℕ := 30
def percentage_slept_through : ℕ := 5
def remainder_attendees := total_attendees * (100 - percentage_listened_entire - percentage_slept_through) / 100
def percentage_heard_half_of_remainder : ℕ := 40
def percentage_heard_quarter_of_remainder := 100 - percentage_heard_half_of_remainder

theorem average_minutes_of_lecture_heard :
  (percentage_listened_entire * total_attendees / 100 * lecture_duration
   + percentage_slept_through * total_attendees / 100 * 0
   + percentage_heard_half_of_remainder * remainder_attendees / 100 * (lecture_duration / 2)
   + percentage_heard_quarter_of_remainder * remainder_attendees / 100 * (lecture_duration / 4)
  ) / total_attendees = 47.5 :=
by
  sorry

end average_minutes_of_lecture_heard_l651_651667


namespace min_value_expr_l651_651278

noncomputable def min_expr (a b c k m n : ℝ) : ℝ := 
  (k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a

theorem min_value_expr (a b c k m n : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) :
  min_expr (k:=k) (m:=m) (n:=n) a b c k m n ≥ 6 * real.sqrt (real.sqrt (k * m * n)) :=
sorry

end min_value_expr_l651_651278


namespace measure_AX_l651_651240

-- Definitions based on conditions
def circle_radii (r_A r_B r_C : ℝ) : Prop :=
  r_A - r_B = 6 ∧
  r_A - r_C = 5 ∧
  r_B + r_C = 9

-- Theorem statement
theorem measure_AX (r_A r_B r_C : ℝ) (h : circle_radii r_A r_B r_C) : r_A = 10 :=
by
  sorry

end measure_AX_l651_651240


namespace alyosha_cube_problem_l651_651472

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651472


namespace minimum_distance_parabola_line_l651_651164

noncomputable def distance_point_line (x y : ℝ) : ℝ := 
  |4 * x + 3 * y - 8| / real.sqrt (4 ^ 2 + 3 ^ 2)

def parabola_y (x : ℝ) : ℝ := - x ^ 2

theorem minimum_distance_parabola_line :
  ∃ m : ℝ, parabola_y m = -m^2 ∧ distance_point_line m (parabola_y m) = 4 / 3 :=
sorry

end minimum_distance_parabola_line_l651_651164


namespace all_terms_are_integers_l651_651341

open Nat

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 143 ∧ ∀ n ≥ 2, a (n + 1) = 5 * (Finset.range n).sum a / n

theorem all_terms_are_integers (a : ℕ → ℕ) (h : seq a) : ∀ n : ℕ, 1 ≤ n → ∃ k : ℕ, a n = k := 
by
  sorry

end all_terms_are_integers_l651_651341


namespace find_n_l651_651481

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651481


namespace bamboo_node_volume_5_l651_651803

theorem bamboo_node_volume_5 {a_1 d : ℚ} :
  (a_1 + (a_1 + d) + (a_1 + 2 * d) + (a_1 + 3 * d) = 3) →
  ((a_1 + 6 * d) + (a_1 + 7 * d) + (a_1 + 8 * d) = 4) →
  (a_1 + 4 * d = 67 / 66) :=
by sorry

end bamboo_node_volume_5_l651_651803


namespace number_of_solutions_l651_651112

noncomputable def f (x : ℝ) : ℝ := x / 50
noncomputable def g (x : ℝ) : ℝ := Real.cos x

theorem number_of_solutions : 
  (Set.filter (λ x : ℝ, f x = g x) (Set.Icc (-50 : ℝ) 50)).card = 31 :=
by sorry

end number_of_solutions_l651_651112


namespace f_at_1_f_expr_l651_651158

noncomputable def f : ℝ → ℝ := sorry

axiom hx : ∀ x : ℝ, f (Real.exp x) = x + 2

theorem f_at_1 : f 1 = 2 := by
  have h := hx 0
  show f 1 = 2
  calc
    f 1 = f (Real.exp 0) := by rw Real.exp_zero
    ... = 0 + 2 := h
    ... = 2 := by ring

theorem f_expr (x : ℝ) : f x = Real.log x + 2 := by
  sorry

end f_at_1_f_expr_l651_651158


namespace cube_decomposition_l651_651480

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651480


namespace inequality_proof_l651_651135

def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) (h : 0 < x ∧ x < π / 2) : True
axiom f_derivative (x : ℝ) (h : 0 < x ∧ x < π / 2) : differentiable_at ℝ f x
axiom f_inequality (x : ℝ) (h : 0 < x ∧ x < π / 2) : f' x * sin x < f x * cos x

theorem inequality_proof : sqrt 3 * f (π / 4) > sqrt 2 * f (π / 3) :=
sorry

end inequality_proof_l651_651135


namespace sum_a_t_l651_651986

theorem sum_a_t (a : ℝ) (t : ℝ) 
  (h₁ : a = 6)
  (h₂ : t = a^2 - 1) : a + t = 41 :=
by
  sorry

end sum_a_t_l651_651986


namespace total_amount_shared_l651_651432

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
axiom condition1 : a = (1 / 3 : ℝ) * (b + c)
axiom condition2 : b = (2 / 7 : ℝ) * (a + c)
axiom condition3 : a = b + 15

-- The proof statement
theorem total_amount_shared : a + b + c = 540 :=
by
  -- We assume these axioms are declared and noncontradictory
  sorry

end total_amount_shared_l651_651432


namespace not_in_range_g_zero_l651_651737

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l651_651737


namespace longest_line_segment_square_in_sector_l651_651437

noncomputable def pie_diameter : ℝ := 12
noncomputable def num_sectors : ℕ := 3

theorem longest_line_segment_square_in_sector :
  let r := pie_diameter / 2 in
  let θ := (2 * Real.pi) / num_sectors in
  let l := 2 * r * Real.sqrt (1 - (Real.cos θ / 2)^2) in
  l^2 = 108 :=
by
  let r := pie_diameter / 2
  let θ := (2 * Real.pi) / num_sectors
  let l := 2 * r * Real.sqrt (1 - (Real.cos θ / 2)^2)
  have l_val : l = 6 * Real.sqrt 3 := sorry
  have h : l^2 = (6 * Real.sqrt 3)^2 := by rw [l_val]
  rw [Real.mul_self_sqrt (by norm_num), mul_comm, mul_assoc, ← Real.pow_two] at h
  exact h

end longest_line_segment_square_in_sector_l651_651437


namespace find_k_value_l651_651571

theorem find_k_value :
  ∃ k : ℝ, (∀ x : ℝ, 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5) ∧ 
          (∃ a b : ℝ, b - a = 8 ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5)) ∧ 
          k = 9 / 4 :=
sorry

end find_k_value_l651_651571


namespace determine_f_l651_651448

-- Given conditions
def vertex_condition (d e f : ℝ) : Prop := ∀ x : ℝ, y : ℝ, (x - 3 = 0) → (y = -5) → (y = d * x^2 + e * x + f)
def point_condition (d e f : ℝ) : Prop := d * 25 + e * 5 + f = -1

-- The main theorem stating that f = 4
theorem determine_f (d e f : ℝ) (h1 : vertex_condition d e f) (h2 : point_condition d e f) : f = 4 :=
sorry

end determine_f_l651_651448


namespace irrational_sqrt_3_l651_651412

theorem irrational_sqrt_3 : 
  (∀ (x : ℝ), (x = 1 / 2 → ¬irrational x) ∧ (x = 0.2 → ¬irrational x) ∧ (x = -5 → ¬irrational x) ∧ (x = sqrt 3 → irrational x)) :=
by
  sorry

end irrational_sqrt_3_l651_651412


namespace sqrt_16_eq_pm_4_l651_651816

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l651_651816


namespace all_tell_truth_at_same_time_l651_651080

-- Define the probabilities of each person telling the truth.
def prob_Alice := 0.7
def prob_Bob := 0.6
def prob_Carol := 0.8
def prob_David := 0.5

-- Prove that the probability that all four tell the truth at the same time is 0.168.
theorem all_tell_truth_at_same_time :
  prob_Alice * prob_Bob * prob_Carol * prob_David = 0.168 :=
by
  sorry

end all_tell_truth_at_same_time_l651_651080


namespace prob_grid_entirely_black_l651_651883

noncomputable theory

-- Define the initial grid and its properties
structure Grid :=
  (grid : fin 4 → fin 4 → bool) -- 4x4 grid of bool values representing black (true) or white (false)

def is_adjacent (i j k l : fin 4) : Prop :=
  (i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ (j = l ∧ (i = k + 1 ∨ i = k - 1))

def white_to_black (grid : fin 4 → fin 4 → bool) (i j : fin 4) : bool :=
  if ∃ (k l : fin 4), is_adjacent i j k l ∧ grid k l then true else grid i j

def rotate_90 (grid: fin 4 → fin 4 → bool) : fin 4 → fin 4 → bool := 
  λ i j, grid (3 - j) i

def grid_all_black (grid : fin 4 → fin 4 → bool) : Prop :=
  ∀ i j : fin 4, grid i j

theorem prob_grid_entirely_black : 
  (probability ( ∀ (g : Grid), grid_all_black (λ i j, white_to_black (rotate_90 g.grid) i j))) < (ε)
:= sorry

end prob_grid_entirely_black_l651_651883


namespace oldest_child_age_l651_651320

-- Average age of 7 children is 8 years old
def average_age_seven_children (ages : Fin 7 → ℕ) : Prop :=
  (∑ i, ages i) = 7 * 8

-- Each child has a different age
def different_ages (ages : Fin 7 → ℕ) : Prop :=
  ∀ i j, i ≠ j → ages i ≠ ages j

-- Difference of two years in the ages of any two consecutive children
def consecutive_age_difference (ages : Fin 7 → ℕ) : Prop :=
  ∀ i : Fin 6, ages (⟨i.1 + 1, Nat.succ_lt_succ_iff.mp i.2⟩) = ages i + 2

-- To prove that the age of the oldest child is 14 years
theorem oldest_child_age (ages : Fin 7 → ℕ) 
  (avg : average_age_seven_children ages)
  (diff : different_ages ages)
  (consec : consecutive_age_difference ages) :
  ages (⟨6, by decide⟩) = 14 :=
by
  sorry

end oldest_child_age_l651_651320


namespace bernardo_wins_at_5_l651_651754

theorem bernardo_wins_at_5 :
  (∀ N : ℕ, (16 * N + 900 < 1000) → (920 ≤ 16 * N + 840) → N ≥ 5)
    ∧ (5 < 10 ∧ 16 * 5 + 900 < 1000 ∧ 920 ≤ 16 * 5 + 840) := by
{
  sorry
}

end bernardo_wins_at_5_l651_651754


namespace trapezoid_not_parallelogram_is_minor_premise_l651_651688

theorem trapezoid_not_parallelogram_is_minor_premise 
  (square_is_parallelogram : ∀ (S : Type), square S → parallelogram S)
  (trapezoid_not_parallelogram : ∀ (T : Type), trapezoid T → ¬parallelogram T) :
  ∃ (minor_premise : ∀ (T : Type), trapezoid T → ¬parallelogram T), minor_premise = trapezoid_not_parallelogram :=
by
  sorry

end trapezoid_not_parallelogram_is_minor_premise_l651_651688


namespace find_ax5_plus_by5_l651_651153

variable (a b x y : ℝ)

-- Conditions
axiom h1 : a * x + b * y = 3
axiom h2 : a * x^2 + b * y^2 = 7
axiom h3 : a * x^3 + b * y^3 = 16
axiom h4 : a * x^4 + b * y^4 = 42

-- Theorem (what we need to prove)
theorem find_ax5_plus_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end find_ax5_plus_by5_l651_651153


namespace calculateDifferentialSavings_l651_651927

/-- 
Assumptions for the tax brackets and deductions/credits.
-/
def taxBracketsCurrent (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 15 / 100
  else if income ≤ 45000 then
    15000 * 15 / 100 + (income - 15000) * 42 / 100
  else
    15000 * 15 / 100 + (45000 - 15000) * 42 / 100 + (income - 45000) * 50 / 100

def taxBracketsProposed (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 12 / 100
  else if income ≤ 45000 then
    15000 * 12 / 100 + (income - 15000) * 28 / 100
  else
    15000 * 12 / 100 + (45000 - 15000) * 28 / 100 + (income - 45000) * 50 / 100

def standardDeduction : ℕ := 3000
def childrenCredit (num_children : ℕ) : ℕ := num_children * 1000

def taxableIncome (income : ℕ) : ℕ :=
  income - standardDeduction

def totalTaxLiabilityCurrent (income num_children : ℕ) : ℕ :=
  (taxBracketsCurrent (taxableIncome income)) - (childrenCredit num_children)

def totalTaxLiabilityProposed (income num_children : ℕ) : ℕ :=
  (taxBracketsProposed (taxableIncome income)) - (childrenCredit num_children)

def differentialSavings (income num_children : ℕ) : ℕ :=
  totalTaxLiabilityCurrent income num_children - totalTaxLiabilityProposed income num_children

/-- 
Statement of the Lean 4 proof problem.
-/
theorem calculateDifferentialSavings : differentialSavings 34500 2 = 2760 :=
by
  sorry

end calculateDifferentialSavings_l651_651927


namespace least_divisible_except_two_consecutives_l651_651904

theorem least_divisible_except_two_consecutives :
  ∃ N : ℕ, N = 180180 ∧ (∀ m ∈ (finset.range 28).image (λ x, x + 1), m ∣ N) ∧ ¬ (28 ∣ N) ∧ ¬ (29 ∣ N) := 
by 
  -- Proof omitted
  sorry

end least_divisible_except_two_consecutives_l651_651904


namespace total_votes_proof_l651_651762

variable (total_voters first_area_percent votes_first_area votes_remaining_area votes_total : ℕ)

-- Define conditions
def first_area_votes_condition : Prop :=
  votes_first_area = (total_voters * first_area_percent) / 100

def remaining_area_votes_condition : Prop :=
  votes_remaining_area = 2 * votes_first_area

def total_votes_condition : Prop :=
  votes_total = votes_first_area + votes_remaining_area

-- Main theorem to prove
theorem total_votes_proof (h1: first_area_votes_condition) (h2: remaining_area_votes_condition) (h3: total_votes_condition) :
  votes_total = 210000 :=
by
  sorry

end total_votes_proof_l651_651762


namespace model_height_l651_651098

noncomputable def H_actual : ℝ := 50
noncomputable def A_actual : ℝ := 25
noncomputable def A_model : ℝ := 0.025

theorem model_height : 
  let ratio := (A_actual / A_model)
  ∃ h : ℝ, h = H_actual / (Real.sqrt ratio) ∧ h = 5 * Real.sqrt 10 := 
by 
  sorry

end model_height_l651_651098


namespace cube_decomposition_l651_651479

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l651_651479


namespace number_of_homework_situations_l651_651343

theorem number_of_homework_situations (teachers students : ℕ) (homework_options : students = 4 ∧ teachers = 3) :
  teachers ^ students = 81 :=
by
  sorry

end number_of_homework_situations_l651_651343


namespace probability_pair_sum_l651_651436

theorem probability_pair_sum
  (m n : ℕ)
  (deck_size : ℕ := 40)
  (removed_pair : ℕ := 2)
  (remaining_cards : ℕ := deck_size - removed_pair)
  (total_ways_to_draw_two_cards : ℕ := remaining_cards * (remaining_cards - 1) / 2)
  (pair_from_unaffected_numbers : ℕ := 9 * 6)
  (pair_from_removed_number : ℕ := 1)
  (total_ways_to_form_pair : ℕ := pair_from_unaffected_numbers + pair_from_removed_number)
  (probability : ℚ := total_ways_to_form_pair / total_ways_to_draw_two_cards)
  (gcd_mn : Nat.gcd m n = 1)
  (prob_fraction : Rational := (55, 703))
  (sum_m_n : ℕ := prob_fraction.1 + prob_fraction.2) :
  m + n = 758 :=
sorry

end probability_pair_sum_l651_651436


namespace unique_k_linear_equation_l651_651647

theorem unique_k_linear_equation :
  (∀ x y k : ℝ, (2 : ℝ) * x^|k| + (k - 1) * y = 3 → (|k| = 1 ∧ k ≠ 1) → k = -1) :=
by
  sorry

end unique_k_linear_equation_l651_651647


namespace average_weight_estimate_l651_651222

noncomputable def average_weight (female_students male_students : ℕ) (avg_weight_female avg_weight_male : ℕ) : ℝ :=
  (female_students / (female_students + male_students) : ℝ) * avg_weight_female +
  (male_students / (female_students + male_students) : ℝ) * avg_weight_male

theorem average_weight_estimate :
  average_weight 504 596 49 57 = (504 / 1100 : ℝ) * 49 + (596 / 1100 : ℝ) * 57 :=
by
  sorry

end average_weight_estimate_l651_651222


namespace dice_sum_probability_l651_651403

theorem dice_sum_probability : 
  ∃ n : ℕ, ((∀ d : ℕ, d ∈ finset.range(1, 7) → True) ∧ n = nat.choose 7 4) ∧
  ((∃ (k : ℕ), ∀ j : ℕ, j ∈ finset.range(1, 7) → ∀ m, 1 ≤ m ∧ m ≤ 6 → 5 * k + j + m = 8) →
  ∃ n : ℕ, n = 35) :=
begin
    sorry
end

end dice_sum_probability_l651_651403


namespace initial_flour_amount_l651_651126

theorem initial_flour_amount
  (batch_flour : ℕ)
  (baked_batches : ℕ)
  (future_batches : ℕ)
  (total_initial_flour : ℕ) :
  batch_flour = 2 ∧
  baked_batches = 3 ∧
  future_batches = 7 ∧
  total_initial_flour = 20 :=
begin
  sorry
end

end initial_flour_amount_l651_651126


namespace lines_form_isosceles_triangle_l651_651095

-- Define the lines
def line1 : ℝ → ℝ := λ x, 4 * x + 3
def line2 : ℝ → ℝ := λ x, -4 * x + 3
def line3 : ℝ → ℝ := λ x, -3

-- Define the proof statement
theorem lines_form_isosceles_triangle : 
  ∃ (A B C : ℝ × ℝ), 
    A = (0, line1 0) ∧
    B = (-3 / 2, line3 (-3 / 2)) ∧
    C = (3 / 2, line3 (3 / 2)) ∧
    ∃ (a b c : ℝ),
    a = dist A B ∧ b = dist A C ∧ c = dist B C ∧
    a = b ∧ a ≠ c ∧ b ≠ c :=
by 
  let A := (0, line1 0)
  let B := (-3 / 2, line3 (-3 / 2))
  let C := (3 / 2, line3 (3 / 2))
  have hA : A = (0, 3) := rfl
  have hB : B = (-3 / 2, -3) := rfl
  have hC : C = (3 / 2, -3) := rfl
  have ha : dist A B = 3 * real.sqrt 17 / 2 := sorry
  have hb : dist A C = 3 * real.sqrt 17 / 2 := sorry
  have hc : dist B C = 3 := sorry
  use [A, B, C]
  use [3 * real.sqrt 17 / 2, 3 * real.sqrt 17 / 2, 3]
  exact ⟨rfl, rfl, rfl, ⟨rfl, rfl, rfl, ⟨rfl, by norm_num⟩⟩⟩


end lines_form_isosceles_triangle_l651_651095


namespace perimeter_of_quadrilateral_l651_651231

-- Define the quadrilateral with given properties
variable (A B C D : Point)
variable (AB BC CD : ℝ)
variable (right_angle_B : angle B = π/2)
variable (BD_perpendicular_CD : perpendicular BD CD)
variable (AB_eq_15 : AB = 15)
variable (BC_eq_20 : BC = 20)
variable (CD_eq_9 : CD = 9)

-- Define the result to be proved
def perimeter_of_ABCD : ℝ :=
  AB + BC + CD + (BD_length BD CD)

theorem perimeter_of_quadrilateral :
  perimeter_of_ABCD AB BC CD = 44 + sqrt 481 := by
  sorry

end perimeter_of_quadrilateral_l651_651231


namespace circles_externally_tangent_l651_651802

-- Define the data for circle 1
def circle1_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 1

-- Define the data for circle 2
def circle2_center : ℝ × ℝ := (0, 3)
def circle2_radius : ℝ := 2

-- The distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- The theorem to prove the positional relationship
theorem circles_externally_tangent :
  distance circle1_center circle2_center = circle1_radius + circle2_radius :=
by
  sorry

end circles_externally_tangent_l651_651802


namespace arrangement_count_l651_651946

theorem arrangement_count (white_cubes : ℕ) (black_cubes : ℕ) (cube_size : ℕ) : 
  white_cubes = 9 ∧ black_cubes = 18 ∧ cube_size = 3 → 
  ∃ count : ℕ, count = 60480 :=
by
  intro h
  use 60480
  sorry

end arrangement_count_l651_651946


namespace fraction_special_phone_numbers_l651_651066

def valid_phone_numbers : ℕ := (7 * 9 * 10^5)

def special_phone_numbers : ℕ := 10^5

theorem fraction_special_phone_numbers :
  (special_phone_numbers : ℚ) / valid_phone_numbers = 1 / 63 := 
by
  sorry

end fraction_special_phone_numbers_l651_651066


namespace sum_scores_with_three_ways_l651_651224

def valid_score (c u i : ℕ) : Prop := 
  c + u + i = 30 ∧ 
  7 * c + 3 * u = S ∧
  c ≥ 0 ∧ c ≤ 30 ∧ 
  u ≥ 0 ∧ u ≤ 30 ∧ 
  i ≥ 0

def score_count_exactly_three (S : ℕ) : ℕ :=
  ∑ c in finset.range 31, 
    ∑ u in finset.range 31, 
      if valid_score c u (30 - c - u) then 1 else 0

theorem sum_scores_with_three_ways :
  let scores := finset.range (210 + 1) in
  let scores_with_three_ways := scores.filter (λ S, score_count_exactly_three S = 3) in
  ∑ s in scores_with_three_ways, s = 189 :=
sorry

end sum_scores_with_three_ways_l651_651224


namespace cut_out_area_l651_651582

theorem cut_out_area (x : ℝ) (h1 : x * (x - 10) = 1575) : 10 * x - 10 * 10 = 450 := by
  -- Proof to be filled in here
  sorry

end cut_out_area_l651_651582


namespace circle_center_radius_sum_l651_651276

theorem circle_center_radius_sum (x y a b r : ℝ) :
  (x^2 + 10*x + y^2 + 12*y + 57 = 0) →
  (x + 5)^2 + (y + 6)^2 = 4 →
  a = -5 →
  b = -6 →
  r = 2 →
  a + b + r = -9 :=
by
  intro h_eq h_standard_form ha hb hr
  have h_ab : a + b = -11, from sorry
  have h_ab_r := h_ab + hr
  exact h_ab_r

end circle_center_radius_sum_l651_651276


namespace cosine_angle_BHD_l651_651683

variables {a : ℝ}
-- Define point coordinates assuming dimensions of the solid, length CD = a
variables {D H G F B : ℝ}

-- Conditions of the angles given in the problem
def angle_DHG := 30 -- in degrees
def angle_FHB := 45 -- in degrees

-- Variables for the angles in radians for calculation
noncomputable def angle_DHG_rad : ℝ := angle_DHG * (π / 180)
noncomputable def angle_FHB_rad : ℝ := angle_FHB * (π / 180)

-- Definition of the main goal to prove
theorem cosine_angle_BHD (a : ℝ) (angle_DHG_rad = π / 6) (angle_FHB_rad = π / 4) :
  ∃ (cos_BHD : ℝ), cos_BHD = 1 / 4 :=
by
  sorry

end cosine_angle_BHD_l651_651683


namespace remainder_17_pow_63_mod_7_l651_651360

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651360


namespace find_negative_number_l651_651517

theorem find_negative_number :
  ∃ x : ℤ, (x = -|5|) ∧ x < 0 ∧ (-(-3) ≥ 0) ∧ (1/2 > 0) ∧ (0 = 0) :=
by
  use (-|5|)
  split; 
  { 
    sorry 
  }

end find_negative_number_l651_651517


namespace remainder_17_pow_63_mod_7_l651_651382

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651382


namespace midpoint_locus_l651_651990

open Real EuclideanGeometry

-- Definition of a circle with center O and radius r
structure Circle :=
(center : Point)
(radius : ℝ)

-- Define the point P inside the circle with distance r/2 from O
variables (O P : Point) (r : ℝ)
hypothesis_distance : dist O P = r / 2

theorem midpoint_locus (K : Circle) (H1 : K.center = O) (H2 : K.radius = r) :
    ∀ (M : Point), (∃ (A B : Point), dist O A = r ∧ dist O B = r ∧ M = midpoint A B ∧ collinear {A, P, B}) ↔
    dist M (midpoint O P) = r / 4 := 
by 
    -- The proof body is omitted
    sorry

end midpoint_locus_l651_651990


namespace problem_AD_l651_651145

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

open Real

theorem problem_AD :
  (∀ x, 0 < x ∧ x < π / 4 → f x < f (x + 0.01) ∧ g x < g (x + 0.01)) ∧
  (∃ x, x = π / 4 ∧ f x + g x = 1 / 2 + sqrt 2) :=
by
  sorry

end problem_AD_l651_651145


namespace circle_center_sum_radius_eq_neg_nine_l651_651273

noncomputable def circle_center_sum_radius (x y : ℝ) : ℝ :=
  let a := -5
  let b := -6
  let r := 2
  a + b + r

theorem circle_center_sum_radius_eq_neg_nine (x y : ℝ) :
  (x^2 + 12 * y + 57 = -y^2 - 10 * x) → circle_center_sum_radius x y = -9 :=
by
  intro h
  unfold circle_center_sum_radius
  rw [add_assoc, add_comm (-6)]
  exact rfl

end circle_center_sum_radius_eq_neg_nine_l651_651273


namespace factor_expression_l651_651101

theorem factor_expression (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) /
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) =
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) :=
by
  sorry

end factor_expression_l651_651101


namespace reciprocal_of_fraction_l651_651806

noncomputable def fraction := (Real.sqrt 5 + 1) / 2

theorem reciprocal_of_fraction :
  (fraction⁻¹) = (Real.sqrt 5 - 1) / 2 :=
by
  -- proof steps
  sorry

end reciprocal_of_fraction_l651_651806


namespace find_original_major_radius_l651_651454

noncomputable def original_major_radius (b : ℝ) := 2 * b

theorem find_original_major_radius (minor_radius : ℝ) 
  (volume_conserved : true) 
  (proportional_axes : original_major_radius minor_radius = 2 * minor_radius) 
  (minor_radius_val : minor_radius = 4 * real.cbrt 3) : 
  original_major_radius minor_radius = 8 * real.cbrt 3 :=
by
  sorry

end find_original_major_radius_l651_651454


namespace angle_between_lines_is_arctan_one_third_l651_651599

theorem angle_between_lines_is_arctan_one_third
  (l1 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (l2 : ∀ x y : ℝ, x - y - 2 = 0)
  : ∃ θ : ℝ, θ = Real.arctan (1 / 3) := 
sorry

end angle_between_lines_is_arctan_one_third_l651_651599


namespace alyosha_cube_cut_l651_651498

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651498


namespace central_angle_equilateral_cone_l651_651897

-- Definitions
def is_equilateral_triangle (T : Type) [metric_space T] (triangle : set T) : Prop := sorry

def slant_height (R : ℝ) : ℝ := 2 * R

def central_angle (R : ℝ) (h : ℝ) : ℝ := (2 * real.pi * R) / h

-- Proposition to prove
theorem central_angle_equilateral_cone (R : ℝ) (h : ℝ) (triangle : set ℝ) 
  (h_eq : h = slant_height R) 
  (cross_section : is_equilateral_triangle ℝ triangle) :
  central_angle R h = real.pi :=
by
  sorry

end central_angle_equilateral_cone_l651_651897


namespace number_of_rectangles_in_5x5_grid_l651_651199

-- Number of ways to choose k elements from a set of n elements
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def points_in_each_direction : ℕ := 5
def number_of_rectangles : ℕ :=
  binomial points_in_each_direction 2 * binomial points_in_each_direction 2

-- Lean statement to prove the problem
theorem number_of_rectangles_in_5x5_grid :
  number_of_rectangles = 100 :=
by
  -- begin Lean proof
  sorry

end number_of_rectangles_in_5x5_grid_l651_651199


namespace find_a_intervals_of_monotonicity_find_b_l651_651622

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  6 * Real.log x - a * x^2 - 7 * x + b

theorem find_a (h_extremum : ∀ a b, deriv (λ x, f x a b) 2 = 0) : 
  ∃ a, a = -1 := 
sorry

theorem intervals_of_monotonicity (h_a : ∀ a, a = -1) :
  ∃ (U V W : set ℝ), 
    (∀ x, x ∈ U ↔ 0 < x ∧ x < 3 / 2) ∧
    (∀ x, x ∈ V ↔ x < 2 ∧ x > 3 / 2) ∧ 
    (∀ x, x ∈ W ↔ x > 2) ∧
    (∀ x, U x → deriv (λ x, f x (-1) b) x > 0) ∧ 
    (∀ x, V x → deriv (λ x, f x (-1) b) x < 0) ∧ 
    (∀ x, W x → deriv (λ x, f x (-1) b) x > 0) :=
sorry

theorem find_b (h_extremum : ∀ a b, deriv (λ x, f x a b) 2 = 0) 
  (h_a : ∀ a, a = -1)
  (h_ln2 : Real.log 2 = 0.693) (h_ln15 : Real.log (3 / 2) = 0.405) :
  ∃ b, (33 / 4 - 6 * Real.log (3 / 2)) < b ∧ b < (10 - 6 * Real.log 2) :=
sorry

end find_a_intervals_of_monotonicity_find_b_l651_651622


namespace fraction_identity_l651_651196

variable {n : ℕ}

theorem fraction_identity
  (h1 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → 1 / (n * (n + 1)) = 1 / n - 1 / (n + 1)))
  (h2 : ∀ n : ℕ, (n ≠ 0 → n ≠ 1 → n ≠ 2 → 1 / (n * (n + 1) * (n + 2)) = 1 / (2 * n * (n + 1)) - 1 / (2 * (n + 1) * (n + 2))))
  : 1 / (n * (n + 1) * (n + 2) * (n + 3)) = 1 / (3 * n * (n + 1) * (n + 2)) - 1 / (3 * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end fraction_identity_l651_651196


namespace rate_of_stream_l651_651025

theorem rate_of_stream : 
  ∀ (v : ℝ), 
  (boat_speed_still : ℝ) (distance_downstream : ℝ) (time_downstream : ℝ), 
  boat_speed_still = 16 → 
  distance_downstream = 147 → 
  time_downstream = 7 → 
  distance_downstream = (boat_speed_still + v) * time_downstream → 
  v = 5 :=
by
  intros v boat_speed_still distance_downstream time_downstream 
  intros h1 h2 h3 h4
  sorry

end rate_of_stream_l651_651025


namespace root_magnitude_conditions_l651_651205

theorem root_magnitude_conditions (p : ℝ) (h : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = -p) ∧ (r1 * r2 = -12)) :
  (∃ r1 r2 : ℝ, (r1 ≠ r2) ∧ |r1| > 2 ∨ |r2| > 2) ∧ (∀ r1 r2 : ℝ, (r1 + r2 = -p) ∧ (r1 * r2 = -12) → |r1| * |r2| ≤ 14) :=
by
  -- Proof of the theorem goes here
  sorry

end root_magnitude_conditions_l651_651205


namespace number_of_valid_permutations_l651_651964

-- Define the concept of permutations
def is_valid_permutation (perm : List ℕ) : Prop :=
  perm.length = 8 ∧            -- ensure the permutation has 8 elements
  ∀ i, i < 8 →                 -- for each initial position
    perm[i] ≠ i ∧              -- new position is not the same as old position
    perm[i] ≠ (i + 1) % 8 ∧    -- new position is not adjacent to old position
    perm[i] ≠ (i + 7) % 8      -- new position is not opposite to old position

-- Define the set of all valid permutations
def valid_permutations : List (List ℕ) :=
  (List.permutations (List.range 8)).filter is_valid_permutation

-- Assert the number of valid permutations
theorem number_of_valid_permutations : valid_permutations.length = 6 := by
  sorry

end number_of_valid_permutations_l651_651964


namespace pencils_sold_per_rupee_initially_l651_651909

theorem pencils_sold_per_rupee_initially :
  ∃ P : ℝ, 
    (0.7 * P = 1.3 * 10.77) ∧ 
    P = 20 :=
by
  use 20
  split
  calc
    0.7 * 20 = 14   : by norm_num
    ...      = 1.3 * 10.77 : by norm_num
  exact rfl

end pencils_sold_per_rupee_initially_l651_651909


namespace number_of_arrangements_is_48_l651_651021

noncomputable def number_of_arrangements (students : List String) (boy_not_at_ends : String) (adjacent_girls : List String) : Nat :=
  sorry

theorem number_of_arrangements_is_48 : number_of_arrangements ["A", "B1", "B2", "G1", "G2", "G3"] "B1" ["G1", "G2", "G3"] = 48 :=
by
  sorry

end number_of_arrangements_is_48_l651_651021


namespace non_existence_of_complex_numbers_l651_651959

theorem non_existence_of_complex_numbers 
  (a b c : ℂ) (h : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) :
  ¬ (∀ (k l m : ℤ), (|k| + |l| + |m| >= 1996) → (|1 + ↑k * a + ↑l * b + ↑m * c| > 1 / h)) :=
sorry

end non_existence_of_complex_numbers_l651_651959


namespace rectangle_area_l651_651243

theorem rectangle_area (AB AD AE : ℝ) (S_trapezoid S_triangle : ℝ) (perim_triangle perim_trapezoid : ℝ)
  (h1 : AD - AB = 9)
  (h2 : S_trapezoid = 5 * S_triangle)
  (h3 : perim_triangle + 68 = perim_trapezoid)
  (h4 : S_trapezoid + S_triangle = S_triangle * 6)
  (h5 : perim_triangle = AB + AE + (AE - AB))
  (h6 : perim_trapezoid = AB + AD + AE + (2 * (AD - AE))) :
  AD * AB = 3060 := by
  sorry

end rectangle_area_l651_651243


namespace josie_wins_20x20_chessboard_l651_651695

theorem josie_wins_20x20_chessboard :
  ∀ (board : fin 20 × fin 20 → option bool) 
    (josie_turn : fin 20 × fin 20 → fin 20 × fin 20 → option bool → option bool)
    (ross_turn : fin 20 × fin 20 → option bool → option bool),
  (∀ (board : fin 20 × fin 20 → option bool),  -- Initial game board
  ∃ (win_strategy : (fin 20 × fin 20 → option bool) → bool),
  win_strategy board)
    :=
by
  -- Formal statement, no proof needed.
  sorry

end josie_wins_20x20_chessboard_l651_651695


namespace no_solution_for_n_ge_10_l651_651790

open Nat

theorem no_solution_for_n_ge_10 (n : ℕ) (h : n ≥ 10) : ¬ (n ≤ n! - 4^n ∧ n! - 4^n ≤ 4 * n) := 
sorry

end no_solution_for_n_ge_10_l651_651790


namespace correct_statement_C_l651_651413

theorem correct_statement_C (hA : "3.70 has the same precision as 3.7" = false)
      (hB : "0.200 is accurate to 0.1" = false)
      (hC : "4.0 × 10^3 is accurate to the hundreds place" = true)
      (hD : "5938 is accurate to the tens place as 5940" = false) : 
      "4.0 × 10^3 is accurate to the hundreds place" = true :=
by
  exact hC

end correct_statement_C_l651_651413


namespace find_matrix_M_l651_651565

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℝ :=
  ![\[2, 0, 7\], 
    \[3, 5, -1\], 
    \[-8, -2, 4\]]

def vec_i : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[1\], 
    \[0\], 
    \[0\]]

def vec_j : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[0\], 
    \[1\], 
    \[0\]]

def vec_k : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[0\], 
    \[0\], 
    \[1\]]

def vec_ri : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[2\], 
    \[3\], 
    \[-8\]]

def vec_rj : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[0\], 
    \[5\], 
    \[-2\]]

def vec_rk : Matrix (Fin 3) (Fin 1) ℝ :=
  ![\[7\], 
    \[-1\], 
    \[4\]]

-- Statement to prove
theorem find_matrix_M :
  M * vec_i = vec_ri ∧
  M * vec_j = vec_rj ∧
  M * vec_k = vec_rk :=
  by
    sorry

end find_matrix_M_l651_651565


namespace simplify_and_evaluate_expression_l651_651310

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℤ), x = 2 → y = -1 → (x - 2 * y)^2 - (x - 3 * y) * (x + 3 * y) - 4 * y^2 = 17 :=
by {
  assume x y (hx : x = 2) (hy : y = -1),
  sorry
}

end simplify_and_evaluate_expression_l651_651310


namespace train_passes_platform_in_time_l651_651054

noncomputable def train_speed_km_per_hr := 54 -- speed in km/hr
noncomputable def train_speed_m_per_s := (54 * 5 / 18 : ℝ) -- convert speed to m/s
noncomputable def platform_length_m := 210.0168 -- length of platform in meters
noncomputable def time_to_pass_man_s := 20 -- time to pass the man in seconds

noncomputable def train_length_m := (20 * train_speed_m_per_s : ℝ) -- length of the train
noncomputable def total_distance_m := train_length_m + platform_length_m -- total distance to cover
noncomputable def time_to_pass_platform_s := total_distance_m / train_speed_m_per_s -- time to pass the platform

theorem train_passes_platform_in_time :
    time_to_pass_platform_s = 34.00112 :=
by
    sorry

end train_passes_platform_in_time_l651_651054


namespace subset_A_B_l651_651715

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l651_651715


namespace remaining_black_cards_l651_651022

theorem remaining_black_cards 
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)
  (cards_taken_out : ℕ)
  (h1 : total_cards = 52)
  (h2 : black_cards = 26)
  (h3 : red_cards = 26)
  (h4 : cards_taken_out = 5) :
  black_cards - cards_taken_out = 21 := 
by {
  sorry
}

end remaining_black_cards_l651_651022


namespace log_base_2_inequality_l651_651875

theorem log_base_2_inequality (a b : ℝ) (h : a > b) : 
  (¬ (a > 0 ∧ b > 0 ∧ log 2 a > log 2 b)) → (log 2 a > log 2 b → a > b) := 
by 
  sorry

end log_base_2_inequality_l651_651875


namespace judah_to_shelby_ratio_l651_651074

variable (M : ℕ)

-- Carter scores 4 goals per game
def Carter_avg_goals : ℕ := 4

-- Shelby scores half as many goals per game as Carter
def Shelby_avg_goals : ℕ := Carter_avg_goals / 2

-- Judah scores three less than a certain multiple of Shelby's goals per game
def Judah_avg_goals : ℕ := M * Shelby_avg_goals - 3

-- Total goals scored by the team per game is 7
def team_total_goals : ℕ := Carter_avg_goals + Shelby_avg_goals + Judah_avg_goals

theorem judah_to_shelby_ratio : team_total_goals = 7 → Judah_avg_goals / Shelby_avg_goals = 1 / 2 :=
by
  intro h
  have Carter_avg_goals_value : Carter_avg_goals = 4 := rfl
  have Shelby_avg_goals_value : Shelby_avg_goals = 2 := rfl
  have team_goal_eq : 4 + 2 + Judah_avg_goals = 7 := by {
    rw [Carter_avg_goals_value, Shelby_avg_goals_value],
    exact h
  }
  have Judah_avg_goals_value : Judah_avg_goals = 1 := by {
    rw [Nat.add_assoc, Nat.add_sub_assoc, Nat.add_zero, team_goal_eq],
    trivial
  }
  rw [Shelby_avg_goals_value, Judah_avg_goals_value]
  exact sorry

end judah_to_shelby_ratio_l651_651074


namespace janina_cover_expenses_l651_651255

noncomputable def rent : ℝ := 30
noncomputable def supplies : ℝ := 12
noncomputable def price_per_pancake : ℝ := 2
noncomputable def total_expenses : ℝ := rent + supplies

theorem janina_cover_expenses : total_expenses / price_per_pancake = 21 := 
by
  calc
    total_expenses / price_per_pancake 
    = (rent + supplies) / price_per_pancake : by rfl
    ... = 42 / 2 : by norm_num
    ... = 21 : by norm_num

end janina_cover_expenses_l651_651255


namespace pipe_A_fill_time_correct_l651_651057

-- Definitions from the conditions
def initial_fullness : ℝ := 4 / 5
def pipe_B_empty_time : ℝ := 6
def combined_time : ℝ := 12

-- The target or proof statement
def pipe_A_fill_time : ℝ := 60 / 11

theorem pipe_A_fill_time_correct :
  let fill_rate_A := 1 / pipe_A_fill_time in
  let empty_rate_B := 1 / pipe_B_empty_time in
  let combined_rate := fill_rate_A - empty_rate_B in
  combined_rate = 1 / (5 * combined_time) :=
sorry

end pipe_A_fill_time_correct_l651_651057


namespace train_speed_l651_651459

-- Define the conditions
def time_to_cross_pole : ℝ := 36  -- in seconds
def length_of_train : ℝ := 700   -- in meters

-- Define the conversion factor
def m_s_to_km_h : ℝ := 3.6

-- Define the given speed
def speed_in_km_h : ℝ := 70

-- State the theorem to prove
theorem train_speed :
  (length_of_train / time_to_cross_pole * m_s_to_km_h) ≈ speed_in_km_h :=
sorry

end train_speed_l651_651459


namespace dayan_sequence_20th_term_l651_651317

theorem dayan_sequence_20th_term (a : ℕ → ℕ) (h1 : a 0 = 0)
    (h2 : a 1 = 2) (h3 : a 2 = 4) (h4 : a 3 = 8) (h5 : a 4 = 12)
    (h6 : a 5 = 18) (h7 : a 6 = 24) (h8 : a 7 = 32) (h9 : a 8 = 40) (h10 : a 9 = 50)
    (h_even : ∀ n : ℕ, a (2 * n) = 2 * n^2) :
  a 20 = 200 :=
  sorry

end dayan_sequence_20th_term_l651_651317


namespace find_abs_xyz_l651_651745

variables {x y z : ℝ}

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem find_abs_xyz
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : distinct x y z)
  (h3 : x + 1 / y = 2)
  (h4 : y + 1 / z = 2)
  (h5 : z + 1 / x = 2) :
  |x * y * z| = 1 :=
by sorry

end find_abs_xyz_l651_651745


namespace minimum_possible_value_of_M_l651_651556

open Finset

theorem minimum_possible_value_of_M :
  ∀ (grid : Matrix (Fin 21) (Fin 21) ℕ),
    (∀ i j, grid i j ∈ range (1, 441 + 1)) →
    (∀ i, ∃ max_row, ∃ min_row, max_row = grid i (0:Fin 21) ∧ min_row = grid i (0:Fin 21)) →
    (∀ j, ∃ max_col, ∃ min_col, max_col = grid (0:Fin 21) j ∧ min_col = grid (0:Fin 21) j) →
    let row_diffs := image (λ i, (find_max grid i - find_min grid i)) univ in
    let col_diffs := image (λ j, (find_max grid j - find_min grid j)) univ in
    let M := row_diffs ∪ col_diffs in
    ∃ m, m ∈ M ∧ m = 230 :=
by
  -- Definitions of find_max and find_min
  let find_max : Matrix (Fin 21) (Fin 21) ℕ → Fin 21 → ℕ := sorry
  let find_min : Matrix (Fin 21) (Fin 21) ℕ → Fin 21 → ℕ := sorry

  -- Outline of the proof should go here
  sorry

end minimum_possible_value_of_M_l651_651556


namespace value_of_f_neg1_l651_651279

def f (x : ℝ) : ℝ := if x >= 0 then 2^x + 2 * x + b else -(2^(-x) + 2 * (-x) + b)

theorem value_of_f_neg1 (b : ℝ) : f (-1) = -4 :=
sorry

end value_of_f_neg1_l651_651279


namespace subset_solution_l651_651720

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l651_651720


namespace statues_painted_l651_651958

-- Definitions based on the conditions provided in the problem
def paint_remaining : ℚ := 1/2
def paint_per_statue : ℚ := 1/4

-- The theorem that answers the question
theorem statues_painted (h : paint_remaining = 1/2 ∧ paint_per_statue = 1/4) : 
  (paint_remaining / paint_per_statue) = 2 := 
sorry

end statues_painted_l651_651958


namespace proof_problem_l651_651287

noncomputable def validate_k_and_t (k t : ℚ) : Prop :=
  let quadratic_x := k * x^2 + (3 - 3 * k) * x + (2 * k - 6) = 0
  let quadratic_y := (k + 3) * y^2 - 15 * y + t = 0
  -- Conditions for k
  (∃ m : ℤ, quadratic_x.discriminant = m^2) ∧
  -- Conditions for y roots
  (∃ y₁ y₂ : ℤ, y₁ > 0 ∧ y₂ > 0 ∧ quadratic_y.eval y₁ = 0 ∧ quadratic_y.eval y₂ = 0 ∧ y₁^2 + y₂^2 = minimum (λ (y ∈ ℤ), y₁^2 + y₂^2))

theorem proof_problem (k t : ℚ) (h₁ : k = 0 ∨ k = 3/4 ∨ k = -3/2 ∨ k = -1/2) (h₂ : t = 15) : validate_k_and_t k t :=
  sorry

end proof_problem_l651_651287


namespace sum_of_fractions_l651_651079

theorem sum_of_fractions :
  ∑ k in Finset.range 10, (k + 1) / 5 = 11 := 
by
  have h_sum : ∑ k in Finset.range 10, (k + 1) = 55 := 
    by sorry -- Sum of first 10 positive integers
  have h_div : 55 / 5 = 11 := 
    by norm_num
  rw [sum_div, h_sum, h_div]
  simp

end sum_of_fractions_l651_651079


namespace smallest_nat_div_7_and_11_l651_651519

theorem smallest_nat_div_7_and_11 (n : ℕ) (h1 : n > 1) (h2 : n % 7 = 1) (h3 : n % 11 = 1) : n = 78 :=
by
  sorry

end smallest_nat_div_7_and_11_l651_651519


namespace multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l651_651699

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem multiple_of_4 : ∃ k : ℕ, y = 4 * k := by
  sorry

theorem multiple_of_8 : ∃ k : ℕ, y = 8 * k := by
  sorry

theorem not_multiple_of_16 : ¬ ∃ k : ℕ, y = 16 * k := by
  sorry

theorem multiple_of_24 : ∃ k : ℕ, y = 24 * k := by
  sorry

end multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l651_651699


namespace alyosha_cube_problem_l651_651470

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651470


namespace integer_abs_vals_less_than_4_l651_651566

-- Define the finite set of integers with an absolute value less than 4
def intSet := {x : ℤ | abs x < 4}

-- Define the list of integers with an absolute value less than 4
def intList : List ℤ := [-3, -2, -1, 0, 1, 2, 3]

-- Define properties to validate
def numIntegers : Prop := intList.length = 7

def sumIntegers : Prop := intList.sum = 0

def productIntegers : Prop := intList.foldr (*) 1 = 0

-- Combine them into a single proposition
def proof : Prop := numIntegers ∧ sumIntegers ∧ productIntegers

theorem integer_abs_vals_less_than_4 : proof := sorry

end integer_abs_vals_less_than_4_l651_651566


namespace Chris_average_speed_l651_651058

-- Define the initial odometer reading
def initial_odometer := 2332

-- Define the final odometer reading
def final_odometer := 2772

-- Define the hours ridden on the first and second days
def hours_day1 := 5
def hours_day2 := 7

-- Define the total time ridden
def total_time := hours_day1 + hours_day2

-- Define the total distance travelled
def total_distance := final_odometer - initial_odometer

-- Define the average speed as total distance divided by total time
def average_speed := total_distance.toReal / total_time.toReal

-- The mathematical problem to be proved
theorem Chris_average_speed : average_speed = 36.67 := by
  sorry

end Chris_average_speed_l651_651058


namespace income_of_deceased_is_correct_l651_651004

-- Definitions based on conditions
def family_income_before_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def family_income_after_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def income_of_deceased (total_before: ℝ) (total_after: ℝ) : ℝ := total_before - total_after

-- Given conditions
def avg_income_before : ℝ := 782
def avg_income_after : ℝ := 650
def num_members_before : ℕ := 4
def num_members_after : ℕ := 3

-- Mathematical statement
theorem income_of_deceased_is_correct : 
  income_of_deceased (family_income_before_death avg_income_before num_members_before) 
                     (family_income_after_death avg_income_after num_members_after) = 1178 :=
by
  sorry

end income_of_deceased_is_correct_l651_651004


namespace minimum_people_who_like_both_l651_651298

open Nat

theorem minimum_people_who_like_both (total : ℕ) (mozart : ℕ) (bach : ℕ)
  (h_total: total = 100) (h_mozart: mozart = 87) (h_bach: bach = 70) :
  ∃ x, x = mozart + bach - total ∧ x ≥ 57 :=
by
  sorry

end minimum_people_who_like_both_l651_651298


namespace line_intersects_circle_l651_651635

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (2 * t, 1 + 4 * t)

noncomputable def polar_circle (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin θ

theorem line_intersects_circle :
  ∃ t θ, parametric_line t = (ρ * Real.cos θ, ρ * Real.sin θ)
    ∧ ρ = polar_circle θ
    ∧ ρ^2 = (0 - parametric_line t.1)^2 + (Real.sqrt 2 - parametric_line t.2)^2
    ∧ 2 * (ρ * Real.cos θ)^2 + (ρ * Real.sin θ)^2 = 2 * Real.sqrt 2 * Real.sin θ
    ∧ abs(- Real.sqrt 2 + 1) / Real.sqrt 5 < Real.sqrt 2 :=
sorry

end line_intersects_circle_l651_651635


namespace find_x_with_conditions_l651_651993

theorem find_x_with_conditions (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1)
  (h2 : (nat.factors x).to_finset.card = 3) (h3 : 11 ∈ (nat.factors x).to_finset) :
  x = 59048 := 
by {
  sorry
}

end find_x_with_conditions_l651_651993


namespace functional_eq_solution_l651_651104

-- Define the conditions
variables (f g : ℕ → ℕ)

-- Define the main theorem
theorem functional_eq_solution :
  (∀ n : ℕ, f n + f (n + g n) = f (n + 1)) →
  ( (∀ n, f n = 0) ∨ 
    (∃ (n₀ c : ℕ), 
      (∀ n < n₀, f n = 0) ∧ 
      (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
      (∀ n < n₀ - 1, ∃ ck : ℕ, g n = ck) ∧
      g (n₀ - 1) = 1 ∧
      ∀ n ≥ n₀, g n = 0 ) ) := 
by
  intro h
  /- Proof goes here -/
  sorry

end functional_eq_solution_l651_651104


namespace janina_cover_expenses_l651_651263

theorem janina_cover_expenses : 
  ∀ (rent supplies price_per_pancake : ℕ), 
    rent = 30 → 
    supplies = 12 → 
    price_per_pancake = 2 → 
    (rent + supplies) / price_per_pancake = 21 := 
by 
  intros rent supplies price_per_pancake h_rent h_supplies h_price_per_pancake 
  rw [h_rent, h_supplies, h_price_per_pancake]
  sorry

end janina_cover_expenses_l651_651263


namespace base7_product_l651_651975

-- Definitions of the numbers in base 7
def a : ℕ := nat.of_digits 7 [5, 2, 3]  -- 325_7 in base 10
def b : ℕ := nat.of_digits 7 [6]        -- 6_7 in base 10

-- Expected result of the product in base 7
def expected_result : ℕ := nat.of_digits 7 [4, 2, 6, 2]  -- 2624_7 in base 10

-- The theorem stating the product of a and b should be equal to the expected result
theorem base7_product :
  (a * b) = expected_result :=
begin
  sorry
end

end base7_product_l651_651975


namespace correct_choice_l651_651515

-- Definitions from conditions
def A := 2 * Real.sqrt 3 ⊆ {x : ℝ | x < 4}
def B := 2 * Real.sqrt 3 ∈ {x : ℝ | x < 4}
def C := {2 * Real.sqrt 3} ∈ {x : ℝ | x < 4}
def D := {2 * Real.sqrt 3} ⊆ {x : ℝ | x < 3}

-- Theorem statement: proving that B is the correct answer
theorem correct_choice : B = true ∧ A = false ∧ C = false ∧ D = false := by
  sorry

end correct_choice_l651_651515


namespace magic_8_ball_probability_l651_651265

theorem magic_8_ball_probability :
  let n := 6
      k := 3
      p_pos := 1/3
      p_neg := 2/3
  in (nat.choose n k * (p_pos^k * p_neg^(n-k))) = 160 / 729 :=
by
  sorry

end magic_8_ball_probability_l651_651265


namespace distance_from_M_to_AB_l651_651251

-- Definitions
variable (A B C A_1 B_1 M : Point)
variable (triangle_ABC : Triangle A B C)
variable (AA1_bisector : IsAngleBisector A A_1 B C)
variable (BB1_bisector : IsAngleBisector B B_1 A C)
variable (M_on_A1B1 : OnSegment M A_1 B_1)

-- Theorem Statement
theorem distance_from_M_to_AB (A B C M A_1 B_1 : Point)
  [Triangle A B C]
  [IsAngleBisector A A_1 B C]
  [IsAngleBisector B B_1 A C]
  [OnSegment M A_1 B_1] :
  distance_from_point_to_line M (line_through A B) = 
  distance_from_point_to_line M (line_through A C) + 
  distance_from_point_to_line M (line_through B C) :=
sorry

end distance_from_M_to_AB_l651_651251


namespace problem_equivalence_l651_651271

variable {S : Type*} [fintype S]
variable {f : set S → ℝ} (h_f : ∀ X Y, X ⊆ Y → f X ≥ f Y)

theorem problem_equivalence :
  (∀ X Y : set S, f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y) ↔
  (∀ a ∈ (univ : set S), ∀ X Y, X ⊆ Y → f (X ∪ {a}) - f X ≥ f (Y ∪ {a}) - f Y) := sorry

end problem_equivalence_l651_651271


namespace find_huabei_number_l651_651238

theorem find_huabei_number :
  ∃ (hua bei sai : ℕ), 
    (hua ≠ 4 ∧ hua ≠ 8 ∧ bei ≠ 4 ∧ bei ≠ 8 ∧ sai ≠ 4 ∧ sai ≠ 8) ∧
    (hua ≠ bei ∧ hua ≠ sai ∧ bei ≠ sai) ∧
    (1 ≤ hua ∧ hua ≤ 9 ∧ 1 ≤ bei ∧ bei ≤ 9 ∧ 1 ≤ sai ∧ sai ≤ 9) ∧
    ((100 * hua + 10 * bei + sai) = 7632) :=
sorry

end find_huabei_number_l651_651238


namespace remaining_water_after_45_days_l651_651441

def initial_water : ℝ := 500
def daily_loss : ℝ := 1.2
def days : ℝ := 45

theorem remaining_water_after_45_days :
  initial_water - daily_loss * days = 446 := by
  sorry

end remaining_water_after_45_days_l651_651441


namespace minimum_value_m_ineq_proof_l651_651630

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem minimum_value_m (x₀ : ℝ) (m : ℝ) (hx : f x₀ ≤ m) : 4 ≤ m := by
  sorry

theorem ineq_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 4) : 3 ≤ 3 / b + 1 / a := by
  sorry

end minimum_value_m_ineq_proof_l651_651630


namespace remainder_17_pow_63_mod_7_l651_651377

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651377


namespace point_P_is_diametrically_opposite_l651_651421

noncomputable def circle (α : Type) := {c : α | c = c}

variables {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α] [has_dist α]
variables (ω : circle α) (A B C D K P : α)

def is_midpoint (A K D : α) : Prop := dist A K = 2 * dist A D
def is_bisected_angle (A B C D : α) : Prop := ∃ l : α, angle A B l = angle l B C ∧ D ∈ l ∧ D ∈ ω
def line_intersects_circle_again (K C P : α) (ω : circle α) : Prop := ∃ Q ∈ ω, Q ≠ C ∧ K + (K - C) = P

theorem point_P_is_diametrically_opposite (h1: D ∈ ω)(h2: is_midpoint A K D)(h3: is_bisected_angle A B C D)
(h4: line_intersects_circle_again K C P ω) : ∀ B C, ∃ P', P' ∈ ω ∧ (P' = P) ∧ is_diametrically_opposite P' A :=
sorry

end point_P_is_diametrically_opposite_l651_651421


namespace millet_more_than_half_on_wednesday_l651_651292

theorem millet_more_than_half_on_wednesday :
    ∀ (initial_millet : ℝ) (initial_other : ℝ) (millet_add : ℝ) (other_add : ℝ) (millet_eaten : ℝ) (other_eaten : ℝ),
    initial_millet = 0.3 ∧ initial_other = 0.7 ∧
    millet_add = 0.3 ∧ other_add = 0.7 ∧
    millet_eaten = 0.2 ∧ other_eaten = 0.5 →
    let remaining_millet_day := λ n, (millet_add / (1 - (1 - millet_eaten))) * (1 - (1 - millet_eaten)^n) in
    remaining_millet_day 3 > 0.5 * (remaining_millet_day 3 + other_add) :=
by
  intro initial_millet initial_other millet_add other_add millet_eaten other_eaten
  dsimp at *
  sorry

end millet_more_than_half_on_wednesday_l651_651292


namespace find_roots_sum_l651_651658

noncomputable def f : ℝ → ℝ := sorry -- Definition of f

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom symmetric_about_one : ∀ x : ℝ, f (2 - x) = f x
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.logBase 3 x

theorem find_roots_sum : ((∑ x in {x | f x + 4 = f 0 ∧ 0 < x ∧ x < 10}, x) = 30) := by
  -- Proof to be filled in
  sorry

end find_roots_sum_l651_651658


namespace solve_AlyoshaCube_l651_651509

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651509


namespace binomial_coeff_max_7th_term_l651_651686

theorem binomial_coeff_max_7th_term (n : ℕ) (hn : 0 < n) : 
  (nat.choose n 5 ≤ nat.choose n 6) → 
  (nat.choose n 6 ≤ nat.choose n 7) → 
  11 ≤ n ∧ n ≤ 13 :=
sorry

end binomial_coeff_max_7th_term_l651_651686


namespace seventeen_power_sixty_three_mod_seven_l651_651363

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651363


namespace proof_problem_l651_651152

def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0
def q : Prop := ∀ (a b : ℝ), (a^2 < b^2) → (a < b)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : p ∧ ¬ q := by
  exact ⟨h₁, h₂⟩

end proof_problem_l651_651152


namespace remainder_of_17_pow_63_mod_7_l651_651371

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651371


namespace f_increasing_on_interval_1_inf_f_min_value_2_6_f_max_value_2_6_l651_651629

open Set Real

namespace FunctionProperties

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

-- Prove monotonicity on (1, +∞)
theorem f_increasing_on_interval_1_inf : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 < f x2 := 
by
  sorry

-- Prove the minimum value on [2, 6]
theorem f_min_value_2_6 : is_glb (f '' (Icc 2 6)) (5 / 2) := 
by
  sorry

-- Prove the maximum value on [2, 6]
theorem f_max_value_2_6 : is_lub (f '' (Icc 2 6)) (37 / 6) := 
by
  sorry

end FunctionProperties

end f_increasing_on_interval_1_inf_f_min_value_2_6_f_max_value_2_6_l651_651629


namespace sin_theta_is_3_over_5_l651_651235

-- Definitions of the conditions
section
variable (A B C D M N : Point)
variable (AB BC CD DA : Line)
variable (θ α : ℝ)

-- Conditions
variable (is_square: is_square A B C D)
variable (midpoint_BC: midpoint M B C)
variable (midpoint_CD: midpoint N C D)
variable (angle_BAM: angle A B M = α)
variable (angle_DAN: angle A D N = α)
variable (hypotenuse_AM: hypotenuse A M (sqrt 5 / 2))

-- Proof problem statement
theorem sin_theta_is_3_over_5 :
  sin θ = 3 / 5 :=
sorry
end

end sin_theta_is_3_over_5_l651_651235


namespace pow_mod_seventeen_l651_651389

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651389


namespace least_value_of_q_minus_p_l651_651674

variables (y p q : ℝ)

/-- Triangle side lengths -/
def BC := y + 7
def AC := y + 3
def AB := 2 * y + 1

/-- Given conditions for triangle inequalities and angle B being the largest -/
def triangle_inequality_conditions :=
  (y + 7 + (y + 3) > 2 * y + 1) ∧
  (y + 7 + (2 * y + 1) > y + 3) ∧
  ((y + 3) + (2 * y + 1) > y + 7)

def angle_largest_conditions :=
  (2 * y + 1 > y + 3) ∧
  (2 * y + 1 > y + 7)

/-- Prove the least possible value of q - p given the conditions -/
theorem least_value_of_q_minus_p
  (h1 : triangle_inequality_conditions y)
  (h2 : angle_largest_conditions y)
  (h3 : 6 < y)
  (h4 : y < 8) :
  q - p = 2 := sorry

end least_value_of_q_minus_p_l651_651674


namespace ceil_sqrt_sum_l651_651550

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end ceil_sqrt_sum_l651_651550


namespace weight_equivalence_l651_651828

def weight_bigcircle : ℝ := 1 -- Let's assume the weight of \bigcirc as a scalar in ℝ for simplicity.

def weight_circle : ℝ := (2 / 5) * weight_bigcircle

def weight_15circles := 15 * weight_circle

theorem weight_equivalence : weight_15circles = 6 * weight_bigcircle :=
by
  sorry

end weight_equivalence_l651_651828


namespace proof_problem_correct_l651_651665

noncomputable def proof_problem
  (a b c A B C : ℝ)
  (h1 : b * (b - real.sqrt 3 * c) = (a - c) * (a + c))
  (h2 : B > real.pi / 2 ∧ B < real.pi) : Prop :=
  (A = real.pi / 6) ∧ (a = 1 / 2 → b - real.sqrt 3 * c ∈ set.Ioo (-1 / 2) (1 / 2))

theorem proof_problem_correct : ∀ a b c A B C, 
  proof_problem a b c A B C :=
by 
  intros a b c A B C
  sorry

end proof_problem_correct_l651_651665


namespace find_original_volume_l651_651464

theorem find_original_volume
  (V : ℝ)
  (h1 : V - (3 / 4) * V = (1 / 4) * V)
  (h2 : (1 / 4) * V - (3 / 4) * ((1 / 4) * V) = (1 / 16) * V)
  (h3 : (1 / 16) * V = 0.2) :
  V = 3.2 :=
by 
  -- Proof skipped, as the assistant is instructed to provide only the statement 
  sorry

end find_original_volume_l651_651464


namespace volume_of_regular_hexagonal_pyramid_l651_651572

noncomputable def hexagonalPyramidVolume (a r : ℝ) : ℝ :=
  (sqrt 3 / 2) * a^2 * r

theorem volume_of_regular_hexagonal_pyramid (a r : ℝ) :
  ∃ V, V = hexagonalPyramidVolume a r := 
  by sorry

end volume_of_regular_hexagonal_pyramid_l651_651572


namespace ceil_sqrt_sum_l651_651551

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end ceil_sqrt_sum_l651_651551


namespace subset_a_eq_1_l651_651703

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651703


namespace diff_between_sams_sum_and_alexs_sum_l651_651306

theorem diff_between_sams_sum_and_alexs_sum :
  let Sum_S := ∑ i in (Finset.range 101), i
  let Sum_A := 10 * 0 + 20 * 20 + 40 * 20 + 60 * 20 + 80 * 20 + 100 * 10
  (Sum_S - Sum_A = 50) :=
by
  sorry

end diff_between_sams_sum_and_alexs_sum_l651_651306


namespace log_cos_sum_val_l651_651570

noncomputable def log_cos_sum : ℝ :=
  log 4 (cos (π / 5)) + log 4 (cos (2 * π / 5))

theorem log_cos_sum_val : log_cos_sum = -1 := 
  sorry

end log_cos_sum_val_l651_651570


namespace zero_not_in_range_of_g_l651_651730

def g (x : ℝ) : ℤ :=
  if x > -3 then
    Int.ceil (2 / (x + 3))
  else 
    Int.floor (2 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ x : ℝ, g x = 0 :=
sorry

end zero_not_in_range_of_g_l651_651730


namespace seventeen_power_sixty_three_mod_seven_l651_651366

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end seventeen_power_sixty_three_mod_seven_l651_651366


namespace area_of_triangle_ABC_l651_651609

-- Define a right triangle ΔABC with vertices A, B, and C
variable {A B C D E : Point} -- Points in the triangle setup
variable {AD BE : Line} -- Medians AD and BE

-- Conditions
axiom medians_perpendicular (h1 : IsMedian AD A B C) (h2 : IsMedian BE B A C) 
            (h3: Perpendicular AD BE) : 
            ∃ G : Point, IsCentroid G A B C

axiom triangle_right (h : IsRightTriangle A B C) : True

axiom AD_length : length AD = 25
axiom BE_length : length BE = 35

-- The proof statement
theorem area_of_triangle_ABC (h1 : IsMedian AD A B C)
                             (h2 : IsMedian BE B A C) 
                             (h3 : Perpendicular AD BE)
                             (h : IsRightTriangle A B C)
                             (AD_length : length AD = 25) 
                             (BE_length : length BE = 35) : 
                             area A B C = 5250 / 9 :=
begin
  sorry
end

end area_of_triangle_ABC_l651_651609


namespace angle_value_l651_651400

theorem angle_value (y : ℝ) (h1 : 2 * y + 140 = 360) : y = 110 :=
by {
  -- Proof will be written here
  sorry
}

end angle_value_l651_651400


namespace hawks_first_half_score_l651_651220

variable (H1 H2 E : ℕ)

theorem hawks_first_half_score (H1 H2 E : ℕ) 
  (h1 : H1 + H2 + E = 120)
  (h2 : E = H1 + H2 + 16)
  (h3 : H2 = H1 + 8) :
  H1 = 22 :=
by
  sorry

end hawks_first_half_score_l651_651220


namespace arithmetic_mean_is_five_sixths_l651_651768

theorem arithmetic_mean_is_five_sixths :
  let a := 3 / 4
  let b := 5 / 6
  let c := 7 / 8
  (a + c) / 2 = b := sorry

end arithmetic_mean_is_five_sixths_l651_651768


namespace perimeter_of_triangle_JKL_l651_651049

theorem perimeter_of_triangle_JKL :
  ∀ (P Q R S T U J K L : ℝ×ℝ) 
    (height : ℝ) 
    (side : ℝ), 
    P = (0, 0) → Q = (side, 0) → 
    R = (side / 2, (side * real.sqrt 3) / 2) → 
    T = (0, height) → 
    U = (side, height) → 
    S = (side / 2, height - ((side * real.sqrt 3) / 2)) → 
    J = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) → 
    K = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) → 
    L = ((P.1 + T.1) / 2, (P.2 + T.2) / 2) →
    height = 20 → 
    side = 10 →
    dist J K + dist K L + dist L J = (5 / 2) * (2 * real.sqrt 5 + 2 * real.sqrt 7 + real.sqrt 13) :=
begin
  sorry
end

end perimeter_of_triangle_JKL_l651_651049


namespace sqrt_of_sixteen_l651_651813

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l651_651813


namespace cost_of_pears_l651_651767

theorem cost_of_pears 
  (initial_amount : ℕ := 55) 
  (left_amount : ℕ := 28) 
  (banana_count : ℕ := 2) 
  (banana_price : ℕ := 4) 
  (asparagus_price : ℕ := 6) 
  (chicken_price : ℕ := 11) 
  (total_spent : ℕ := 27) :
  initial_amount - left_amount - (banana_count * banana_price + asparagus_price + chicken_price) = 2 := 
by
  sorry

end cost_of_pears_l651_651767


namespace find_x_with_conditions_l651_651992

theorem find_x_with_conditions (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1)
  (h2 : (nat.factors x).to_finset.card = 3) (h3 : 11 ∈ (nat.factors x).to_finset) :
  x = 59048 := 
by {
  sorry
}

end find_x_with_conditions_l651_651992


namespace ceil_sqrt_sum_eq_24_l651_651553

theorem ceil_sqrt_sum_eq_24:
  1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 →
  5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 →
  15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 →
  Int.ceil (Real.sqrt 3) + Int.ceil (Real.sqrt 27) + Int.ceil (Real.sqrt 243) = 24 :=
by
  intros h1 h2 h3
  have h1_ceil := Real.ceil_sqrt_of_lt_of_gt h1.left h1.right
  have h2_ceil := Real.ceil_sqrt_of_lt_of_gt h2.left h2.right
  have h3_ceil := Real.ceil_sqrt_of_lt_of_gt h3.left h3.right
  simp [h1_ceil, h2_ceil, h3_ceil]
  sorry

end ceil_sqrt_sum_eq_24_l651_651553


namespace subset_A_B_l651_651711

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l651_651711


namespace polynomial_expansion_equality_l651_651649

theorem polynomial_expansion_equality
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : ℤ)
  (polynomial_expansion : (2 * x - 3) ^ 12 = a_0 + a_1 * (x - 1) + a_2 * (x - 1) ^ 2 +
    a_3 * (x - 1) ^ 3 + a_4 * (x - 1) ^ 4 + a_5 * (x - 1) ^ 5 +
    a_6 * (x - 1) ^ 6 + a_7 * (x - 1) ^ 7 + a_8 * (x - 1) ^ 8 +
    a_9 * (x - 1) ^ 9 + a_{10} * (x - 1) ^ 10 + a_{11} * (x - 1) ^ 11 +
    a_{12} * (x - 1) ^ 12) :
  (a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 + a_{10} - a_{11} + a_{12} = 3 ^ 12) ∧
  (a_1 / 2 + a_2 / 2 ^ 2 + a_3 / 2 ^ 3 + a_4 / 2 ^ 4 + a_5 / 2 ^ 5 +
   a_6 / 2 ^ 6 + a_7 / 2 ^ 7 + a_8 / 2 ^ 8 + a_9 / 2 ^ 9 +
   a_{10} / 2 ^ 10 + a_{11} / 2 ^ 11 + a_{12} / 2 ^ 12 = -1) := sorry

end polynomial_expansion_equality_l651_651649


namespace T_1007_mod_9_l651_651575

def T : ℕ → ℕ 
| 0 := 1  -- assigning base cases
| 1 := 2  -- there are exactly two sequences for length 1: "C" and "D"
| n + 1 := 
  let c₁ := T n in -- sequences ending in one "C"
  let c₂ := T n in -- sequences ending in two "C"s
  let d₁ := T n in -- sequences ending in one "D"
  let d₂ := T n in -- sequences ending in two "D"s
  c₁ + c₂ + d₁ + d₂

theorem T_1007_mod_9 : (T 1007) % 9 = 0 := 
sorry

end T_1007_mod_9_l651_651575


namespace smallest_positive_x_l651_651404

theorem smallest_positive_x
  (x : ℕ)
  (h1 : x % 3 = 2)
  (h2 : x % 7 = 6)
  (h3 : x % 8 = 7) : x = 167 :=
by
  sorry

end smallest_positive_x_l651_651404


namespace solve_problem_l651_651115

noncomputable def problem_statement : Prop :=
  3 * 2.2 * ((3.6^2 * 0.48 * log 2.50) / (sqrt 0.12 * sin 0.09 * real.log 0.5)) = -720.72

theorem solve_problem : problem_statement :=
sorry

end solve_problem_l651_651115


namespace difference_of_place_values_l651_651850

theorem difference_of_place_values :
  let n := 54179759
  let pos1 := 10000 * 7
  let pos2 := 10 * 7
  pos1 - pos2 = 69930 := by
  sorry

end difference_of_place_values_l651_651850


namespace min_swindler_ministers_l651_651685

theorem min_swindler_ministers (T D H : ℕ) 
  (h1 : T = 100) 
  (h2 : ∀ (g : Finset ℕ), g.card = 10 → ∃ (d ∈ g), d < H) : 
  D ≥ 91 :=
sorry

end min_swindler_ministers_l651_651685


namespace percentage_is_25_l651_651890

theorem percentage_is_25 : 
  ∀ (P : ℝ), ∀ (x : ℝ), x = 180 → (P * x = 50 - 5) → P = 0.25 :=
by
  intros P x hx eq
  have h: x = 180 := hx
  have h_eq: P * 180 = 50 - 5 := eq
  sorry

end percentage_is_25_l651_651890


namespace distinct_c_values_count_l651_651580

theorem distinct_c_values_count :
  (∀ c ∈ set.Icc (0 : ℝ) 1500, ∃ x : ℝ, 6 * floor x + 3 * ceil x = c) →
  set.card (set.Icc (0 : ℝ) 1500 ∩ {c | ∃ x : ℝ, 6 * floor x + 3 * ceil x = c}) = 167 := sorry

end distinct_c_values_count_l651_651580


namespace area_inside_C_but_outside_A_and_B_l651_651942

-- Definitions for circles with given radii and conditions
def circle_A := { center : ℝ × ℝ, radius : ℝ := 2 }
def circle_B := { center : ℝ × ℝ, radius : ℝ := 1 }
def circle_C := { center : ℝ × ℝ, radius : ℝ := 1 }

-- Condition 1: Circle A and Circle B are tangent to each other
axiom tangent_A_B : dist (circle_A.center) (circle_B.center) = circle_A.radius + circle_B.radius

-- Condition 2: Circle C is tangent to Circle A at the midpoint of the segment connecting the centers of A and B
axiom tangent_C_A_at_midpoint : dist (circle_A.center, circle_C.center) = 1

theorem area_inside_C_but_outside_A_and_B : (π - ((π / 2) - 1)) = π / 2 + 1 :=
by
  sorry

end area_inside_C_but_outside_A_and_B_l651_651942


namespace rectangular_solid_surface_area_l651_651548

theorem rectangular_solid_surface_area (a b c : ℕ) (h₁ : Prime a ∨ ∃ p : ℕ, Prime p ∧ a = p + (p + 1))
                                         (h₂ : Prime b ∨ ∃ q : ℕ, Prime q ∧ b = q + (q + 1))
                                         (h₃ : Prime c ∨ ∃ r : ℕ, Prime r ∧ c = r + (r + 1))
                                         (h₄ : a * b * c = 399) :
  2 * (a * b + b * c + c * a) = 422 := 
sorry

end rectangular_solid_surface_area_l651_651548


namespace roots_are_simplified_sqrt_form_l651_651977

theorem roots_are_simplified_sqrt_form : 
  ∃ m p n : ℕ, gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 ∧
    (∀ x : ℝ, (3 * x^2 - 8 * x + 1 = 0) ↔ 
    (x = (m : ℝ) + (Real.sqrt n)/(p : ℝ) ∨ x = (m : ℝ) - (Real.sqrt n)/(p : ℝ))) ∧
    n = 13 :=
by
  sorry

end roots_are_simplified_sqrt_form_l651_651977


namespace polynomial_bound_l651_651138

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (hP : ∀ x : ℝ, |x| < 1 → |P x a b c d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l651_651138


namespace correct_number_of_true_propositions_l651_651518

noncomputable def true_proposition_count : ℕ := 1

theorem correct_number_of_true_propositions (a b c : ℝ) :
    (∀ a b : ℝ, (a > b) ↔ (a^2 > b^2) = false) →
    (∀ a b : ℝ, (a > b) ↔ (a^3 > b^3) = true) →
    (∀ a b : ℝ, (a > b) → (|a| > |b|) = false) →
    (∀ a b c : ℝ, (a > b) → (a*c^2 ≤ b*c^2) = false) →
    (true_proposition_count = 1) :=
by
  sorry

end correct_number_of_true_propositions_l651_651518


namespace amount_c_is_1600_l651_651006

-- Given conditions
def total_money : ℕ := 2000
def ratio_b_c : (ℕ × ℕ) := (4, 16)

-- Define the total_parts based on the ratio
def total_parts := ratio_b_c.fst + ratio_b_c.snd

-- Define the value of each part
def value_per_part := total_money / total_parts

-- Calculate the amount for c
def amount_c_gets := ratio_b_c.snd * value_per_part

-- Main theorem stating the problem
theorem amount_c_is_1600 : amount_c_gets = 1600 := by
  -- Proof would go here
  sorry

end amount_c_is_1600_l651_651006


namespace distance_point_to_line_l651_651301

variables {a b c x₀ y₀ : ℝ}

theorem distance_point_to_line :
  ∀ (a b c x₀ y₀ : ℝ), 
  (a ≠ 0 ∨ b ≠ 0) →
  (d : ℝ) =
  (abs (a * x₀ + b * y₀ + c)) / (sqrt ((a ^ 2) + (b ^ 2))) :=
sorry

end distance_point_to_line_l651_651301


namespace dad_contribution_is_correct_l651_651073

noncomputable def carl_savings_weekly : ℕ := 25
noncomputable def savings_duration_weeks : ℕ := 6
noncomputable def coat_cost : ℕ := 170

-- Total savings after 6 weeks
noncomputable def total_savings : ℕ := carl_savings_weekly * savings_duration_weeks

-- Amount used to pay bills in the seventh week
noncomputable def bills_payment : ℕ := total_savings / 3

-- Money left after paying bills
noncomputable def remaining_savings : ℕ := total_savings - bills_payment

-- Amount needed from Dad
noncomputable def dad_contribution : ℕ := coat_cost - remaining_savings

theorem dad_contribution_is_correct : dad_contribution = 70 := by
  sorry

end dad_contribution_is_correct_l651_651073


namespace tan_of_acute_angle_l651_651157

-- Definitions and conditions
def is_acute (α : Real) : Prop := 0 < α ∧ α < π / 2
def condition (α : Real) : Prop := cos (π / 2 + α) = -3 / 5

-- Theorem statement
theorem tan_of_acute_angle (α : Real) (h₁ : is_acute α) (h₂ : condition α) : tan α = 3 / 4 := 
  sorry

end tan_of_acute_angle_l651_651157


namespace find_x_exists_unique_l651_651996

theorem find_x_exists_unique (n : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p q r : ℕ, p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ x = p * q * r) (h3 : 11 ∣ x) : x = 59048 :=
sorry

end find_x_exists_unique_l651_651996


namespace part1_part2_l651_651949

-- Define the problem conditions
variables {R r : ℝ} (h : R > r)
variables (P B C A : ℂ) -- Using complex numbers to represent geometric points
variables (hP : abs P = r) (hB : abs B = R)
variables (hC : ∃ t : ℝ, C = B + t * (P - B) ∧ abs C = R)
variables (hA : ∃ l : line, l.contains P ∧ l.is_perpendicular (B - P) ∧ (l.intersects_circle r).nonempty ∧ A ∈ l.intersects_circle r)

-- Proof for part (1)
theorem part1 (hABC : BC^2 + CA^2 + AB^2 = 6 * R^2 + 2 * r^2) : 
  BC^2 + CA^2 + AB^2 = 6 * R^2 + 2 * r^2 :=
sorry

-- Proof for part (2)
theorem part2 (Q : ℂ) (hQ : midpoint P B = Q) : 
  locus Q center = P / 2 ∧ radius = R / 2 :=
sorry

end part1_part2_l651_651949


namespace smallest_expression_l651_651740

def x : ℝ := 10^(-2024)

theorem smallest_expression : x / 4 = 10^(-2025) ∧ 
                              10^(-2025) < 4 + x ∧ 
                              10^(-2025) < 4 - x ∧ 
                              10^(-2025) < 4 * x ∧ 
                              10^(-2025) < 4 / x :=
by 
  sorry

end smallest_expression_l651_651740


namespace positive_diff_is_correct_l651_651925

-- Setting up the conditions for the arithmetic sequence
def a : ℕ → ℤ
| 0        := 3
| (n + 1)  := a n + 8

-- Defining the condition for the geometric sequence
def g : ℕ → ℤ
| 0        := 2
| (n + 1)  := g n * 5

-- Defining the specific terms
def a_100 : ℤ := a 99
def g_4 : ℤ := g 3

-- Calculating the positive difference
def positive_difference : ℤ := abs (a_100 - g_4)

-- The theorem statement
theorem positive_diff_is_correct : positive_difference = 545 := 
by 
  -- Proof steps go here
  sorry

end positive_diff_is_correct_l651_651925


namespace weighted_average_percentage_l651_651772

theorem weighted_average_percentage :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 112
  let bag3_total := 130
  let bag4_popped := 68
  let bag4_total := 90
  let bag5_popped := 82
  let bag5_total := 100

  -- Calculate percentage for each bag
  let perc1 := (bag1_popped : Float) / bag1_total * 100
  let perc2 := (bag2_popped : Float) / bag2_total * 100
  let perc3 := (bag3_popped : Float) / bag3_total * 100
  let perc4 := (bag4_popped : Float) / bag4_total * 100
  let perc5 := (bag5_popped : Float) / bag5_total * 100

  -- Calculate weighted values for each bag
  let weighted1 := perc1 * bag1_total
  let weighted2 := perc2 * bag2_total
  let weighted3 := perc3 * bag3_total
  let weighted4 := perc4 * bag4_total
  let weighted5 := perc5 * bag5_total

  -- Sum the weighted values
  let weighted_sum := weighted1 + weighted2 + weighted3 + weighted4 + weighted5

  -- Total number of kernels
  let total_kernels := bag1_total + bag2_total + bag3_total + bag4_total + bag5_total

  -- Calculate weighted average percentage
  let weighted_avg := weighted_sum / total_kernels

  weighted_avg ≈ 75.03 := 
by sorry

end weighted_average_percentage_l651_651772


namespace sum_y_z_is_84_percent_of_x_l651_651208

open_locale classical

noncomputable theory

variables (x y z : ℝ)

-- Conditions from the problem
variable (h1 : 0.20 * (x - y) = 0.14 * (x + y))
variable (h2 : 0.25 * (x - z) = 0.10 * (y + z))

-- Goal: The sum of y and z is approximately 84% of x.
theorem sum_y_z_is_84_percent_of_x (x y z : ℝ) (h1 : 0.20 * (x - y) = 0.14 * (x + y)) (h2 : 0.25 * (x - z) = 0.10 * (y + z)) :
  (y + z) ≈ 0.84 * x :=
sorry

end sum_y_z_is_84_percent_of_x_l651_651208


namespace square_root_of_16_is_pm_4_l651_651820

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l651_651820


namespace smallest_integer_with_12_factors_l651_651392

theorem smallest_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∏ (d ∈ (List.range k).filter (λ n, k % n = 0), 1) = 12) ∧ ∀ m : ℕ, (m > 0 ∧ (∏ (d ∈ (List.range m).filter (λ n, m % n = 0), 1) = 12) → k ≤ m) ∧ k = 60 :=
sorry

end smallest_integer_with_12_factors_l651_651392


namespace real_roots_polynomial_l651_651288

variable {n : ℕ}
variable {α β : Fin n → ℝ}
variable {λ : ℝ}

-- Given conditions
def cond1 (α : Fin n → ℝ) : Prop := ∑ j, (α j)^2 < 1
def cond2 (β : Fin n → ℝ) : Prop := ∑ j, (β j)^2 < 1

-- Definitions
def A (α : Fin n → ℝ) : ℝ := Real.sqrt (1 - ∑ j, (α j)^2)
def B (β : Fin n → ℝ) : ℝ := Real.sqrt (1 - ∑ j, (β j)^2)
def W (α β : Fin n → ℝ) : ℝ := (1 / 2) * (1 - ∑ j, (α j) * (β j))^2

-- Main theorem statement
theorem real_roots_polynomial (hα : cond1 α) (hβ : cond2 β) :
  (∀ x : ℝ, (x^n + λ * (x^(n-1) + x^(n-2) + ... + x^3 + W α β * x^2 + A α * B β * x + 1) = 0) → 
  ∀ y : ℝ, y ∈ Roots) ↔ λ = 0 := 
sorry

end real_roots_polynomial_l651_651288


namespace children_on_bus_after_events_l651_651426

-- Definition of the given problem parameters
def initial_children : Nat := 21
def got_off : Nat := 10
def got_on : Nat := 5

-- The theorem we want to prove
theorem children_on_bus_after_events : initial_children - got_off + got_on = 16 :=
by
  -- This is where the proof would go, but we leave it as sorry for now
  sorry

end children_on_bus_after_events_l651_651426


namespace cindy_olaf_earnings_l651_651940
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l651_651940


namespace ellipse_properties_and_angle_l651_651172

variables {b : ℝ}
variables {x₀ y₀ : ℝ}
variables {A B M N E C G : ℝ × ℝ}

-- Given conditions
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1
def focus : ℝ × ℝ := (real.sqrt 3, 0)
def y_axis_intersection_A : ℝ × ℝ := (0, 1)
def y_axis_intersection_B : ℝ × ℝ := (0, -1)
def arbitrary_point_M : Prop := ellipse x₀ y₀ ∧ x₀ ≠ 0
def perpendicular_N : ℝ × ℝ := (0, y₀)
def midpoint_E (M N : ℝ × ℝ) : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
def intersection_C (A E : ℝ × ℝ) : ℝ × ℝ := (x₀ / (1 - y₀), -1)
def midpoint_G (B C : ℝ × ℝ) : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def origin : ℝ × ℝ := (0, 0)

-- Proof problem
theorem ellipse_properties_and_angle :
  (∀ (b > 0), ∃ (eq : ellipse 0 y_axis_intersection_A.2),
    (b = 1) ∧ (eq = ellipse (0) 1) ∧ (real.sqrt (1 - b^2 / 4) = real.sqrt 3 / 2) ∧
    let O := origin,
        E := midpoint_E M (perpendicular_N),
        G := midpoint_G y_axis_intersection_B (intersection_C y_axis_intersection_A E) in
    (O ≠ E ∧ E ≠ G ∧ O ≠ G) ∧
    let dot_product := E.1 * G.1 + E.2 * G.2 in
    dot_product = 0 ∧ (∃ (angle : ℝ), angle = 90)) := sorry

end ellipse_properties_and_angle_l651_651172


namespace locus_of_intersection_point_l651_651592

noncomputable def line (a b : ℝ) := { x : ℝ | a * x + b = 0 }

variables {A B C D P Q M : ℝ → ℝ}
variable {e : line 1 0}
variable {AP BQ : ℝ}
variable (k b c d : ℝ)

-- Assumptions 
-- Point A: (0, 0)
-- Point B: (b, 0)
-- Segment CD: parallel to e
-- AP = x from A
-- BQ = 2 * AP from B
-- P = intersection of line PC with x-axis
-- Q = intersection of line QD with x-axis

theorem locus_of_intersection_point (P_moves : {x : ℝ | P x ∈ e}) :
  ( ∃ M : ℝ → ℝ,
    (∀ x : ℝ, M x = x * (k / (b + 2 * c - d)) + (b * k / (b + 2 * c - d))) ∨ 
    ( ∀ x : ℝ, M x = -b)) :=
sorry

end locus_of_intersection_point_l651_651592


namespace josh_500_coins_impossible_l651_651880

theorem josh_500_coins_impossible : ¬ ∃ (x y : ℕ), x + y ≤ 500 ∧ 36 * x + 6 * y + (500 - x - y) = 3564 := 
sorry

end josh_500_coins_impossible_l651_651880


namespace cos_alpha_minus_pi_over_3_cos_2alpha_minus_pi_over_6_l651_651749

variables {α : ℝ} 
variables (hα : 0 < α ∧ α < π / 2)
variables (hcos : cos (α + π / 6) = 3 / 5)

theorem cos_alpha_minus_pi_over_3 : cos (α - π / 3) = 4 / 5 :=
sorry

theorem cos_2alpha_minus_pi_over_6 : cos (2 * α - π / 6) = 24 / 25 :=
sorry

end cos_alpha_minus_pi_over_3_cos_2alpha_minus_pi_over_6_l651_651749


namespace evaluate_expression_l651_651100

theorem evaluate_expression (x y : ℝ) (h1 : x = 3) (h2 : y = 0) : y * (y - 3 * x) = 0 :=
by sorry

end evaluate_expression_l651_651100


namespace probability_of_green_face_l651_651947

theorem probability_of_green_face (total_faces green_faces purple_faces : Nat)
  (h1 : total_faces = 6)
  (h2 : green_faces = 3)
  (h3 : purple_faces = 3)
  (h4 : green_faces + purple_faces = total_faces) :
  (green_faces : ℚ) / (total_faces : ℚ) = 1 / 2 :=
by
    rw [h2, h1]
    norm_num
    sorry

end probability_of_green_face_l651_651947


namespace tan_7pi_over_4_l651_651102

theorem tan_7pi_over_4 : real.tan (7 * real.pi / 4) = -1 :=
by
  sorry

end tan_7pi_over_4_l651_651102


namespace num_elements_P_plus_Q_l651_651277

noncomputable def P : Set ℝ := {0, 2, 5}
noncomputable def Q : Set ℝ := {1, 2, 6}
noncomputable def P_plus_Q : Set ℝ := {x | ∃ a b, a ∈ P ∧ b ∈ Q ∧ x = a + b}

theorem num_elements_P_plus_Q : (P_plus_Q.to_finset.card = 8) :=
by 
  -- Ensuring the Lean statement typechecks
  sorry

end num_elements_P_plus_Q_l651_651277


namespace number_of_smaller_cubes_l651_651039

theorem number_of_smaller_cubes 
  (volume_large_cube : ℝ)
  (volume_small_cube : ℝ)
  (surface_area_difference : ℝ)
  (h1 : volume_large_cube = 216)
  (h2 : volume_small_cube = 1)
  (h3 : surface_area_difference = 1080) :
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3))^2 = surface_area_difference ∧ n = 216 :=
by
  sorry

end number_of_smaller_cubes_l651_651039


namespace total_votes_proof_l651_651761

variable (total_voters first_area_percent votes_first_area votes_remaining_area votes_total : ℕ)

-- Define conditions
def first_area_votes_condition : Prop :=
  votes_first_area = (total_voters * first_area_percent) / 100

def remaining_area_votes_condition : Prop :=
  votes_remaining_area = 2 * votes_first_area

def total_votes_condition : Prop :=
  votes_total = votes_first_area + votes_remaining_area

-- Main theorem to prove
theorem total_votes_proof (h1: first_area_votes_condition) (h2: remaining_area_votes_condition) (h3: total_votes_condition) :
  votes_total = 210000 :=
by
  sorry

end total_votes_proof_l651_651761


namespace dot_product_parallel_vectors_l651_651162

noncomputable theory

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 3) (hb : ‖b‖ = 5) (h_parallel : ∃ k : ℝ, a = k • b)

theorem dot_product_parallel_vectors :
  a ⋅ b = 15 ∨ a ⋅ b = -15 :=
sorry

end dot_product_parallel_vectors_l651_651162


namespace melted_mixture_weight_l651_651416

theorem melted_mixture_weight
    (Z C : ℝ)
    (ratio_eq : Z / C = 9 / 11)
    (zinc_weight : Z = 33.3) :
    Z + C = 74 :=
by
  sorry

end melted_mixture_weight_l651_651416


namespace ellipse_properties_l651_651619

noncomputable theory

variables {x y a b k λ : ℝ}

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2) / a

def is_tangent (x y c : ℝ) : Prop := 
  x - y + c = 0

theorem ellipse_properties (h1 : a > b > 0)
  (h2 : eccentricity a b = Real.sqrt 2 / 2)
  (h3 : b = 1)
  (h4 : is_tangent x y (Real.sqrt 2)) :
  (ellipse_equation (Real.sqrt 2) 1 x y) ∧ 
  (0 < λ ∧ λ < 4 / 3 ∨ -4 / 3 < λ ∧ λ < 0) :=
sorry

end ellipse_properties_l651_651619


namespace someone_answered_yes_l651_651770

def Person : Type := { x : ℕ // x < 2021 }

def is_knight : Person -> Prop
def is_liar : Person -> Prop
def sees_more_liars : Person -> Prop

axiom non_empty_knights : ∃ p : Person, is_knight p
axiom non_empty_liars : ∃ p : Person, is_liar p

axiom knight_or_liar (p : Person) : is_knight p ∨ is_liar p

axiom visibility (p : Person) : 
  ∀ k : ℕ, k ∈ (1 : ℕ)..12 -> Person

axiom knight_truth (p : Person) : 
  is_knight p → sees_more_liars p = false 

axiom liar_lies (p : Person) : 
  is_liar p → sees_more_liars p = true 

theorem someone_answered_yes :
  ∃ p : Person, sees_more_liars p = true := 
sorry

end someone_answered_yes_l651_651770


namespace min_pipes_needed_l651_651445

theorem min_pipes_needed (h : ℝ) : 
  let V_large := (real.pi * (8 ^ 2) * h)
  (∀ n : ℕ, n * (real.pi * (1.5 ^ 2) * h) < V_large → n < 29) :=
begin
  sorry
end

end min_pipes_needed_l651_651445


namespace symmetric_axis_parabola_expression_parabola_vertex_l651_651824

noncomputable def parabola_expression (a h : ℝ) : linear_simp := fun x => a * (x + h)^2

theorem symmetric_axis_parabola_expression (a : ℝ) :
  (∃ h, -h = -2 ∧ -3 = a * (1 + h)^2) → a = -1 / 3 ∧ h = 2 ∧ (∀ x, parabola_expression a h x = - (1 / 3 : ℝ) * (x + 2)^2) :=
begin
  intro h,
  have h_eq_2 : h = 2, by linarith [h],
  have a_eq_neg_one_third : a = -1 / 3, by nlinarith [h],
  split, assumption, assumption,
  intro x, calc
    parabola_expression a h x
        = a * (x + h)^2 : rfl
    ... = -1 / 3 * (x + 2)^2 : by congr,
end

theorem parabola_vertex (a h : ℝ) (h_eq_2 : h = 2) :
  (∀ x, parabola_expression a h x = - (1 / 3 : ℝ) * (x + 2)^2)  →
  a * (1 + h)^2 = -3 →
  -2 = -h ∧ (-2, 0) = (h, 0) :=
begin
  intros hx_eq hx_at_1,
  split,
  linarith,
  have h_eq_2 : h = 2, by linarith,
  have hx_eq_2 : (∀ x, parabola_expression a h x = - (1 / 3 : ℝ) * (x + 2)^2)  → parabola_expression a h (-2) = 0 := perdtodo,
 sorry,
 sorry,
 sorry
end


end symmetric_axis_parabola_expression_parabola_vertex_l651_651824


namespace problem_l651_651631

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x + 2
noncomputable def f' (a x : ℝ) : ℝ := a * (Real.log x + 1) + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x - x^2 - (a + 2) * x + a

theorem problem (a x : ℝ) (h : 1 ≤ x) (ha : 0 < a) : f' a x < x^2 + (a + 2) * x + 1 :=
by
  sorry

end problem_l651_651631


namespace find_principal_l651_651456

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℕ) (h : SI = 4016.25) (hR : R = 5) (hT : T = 5) : 
  let P := 16065 in 
  SI = P * R * T / 100 :=
by
  sorry

end find_principal_l651_651456


namespace paperboy_deliveries_l651_651042

-- Define the sequence E_n
def E : ℕ → ℕ
| 0 := 1        -- arbitrary base case for zero houses
| 1 := 2
| 2 := 4
| 3 := 8
| 4 := 15
| (n+5) := E n + E (n+1) + E (n+2) + E (n+3)

-- Prove that E 12 = 2873
theorem paperboy_deliveries : E 12 = 2873 := by
  sorry

end paperboy_deliveries_l651_651042


namespace theta_third_quadrant_l651_651985

theorem theta_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  π < θ ∧ θ < 3 * π / 2 :=
by 
  sorry

end theta_third_quadrant_l651_651985


namespace pow_mod_seventeen_l651_651388

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651388


namespace certain_number_eq_0_08_l651_651881

theorem certain_number_eq_0_08 (x : ℝ) (h : 1 / x = 12.5) : x = 0.08 :=
by
  sorry

end certain_number_eq_0_08_l651_651881


namespace imaginary_part_of_z_squared_l651_651587

-- Let i be the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number (1 - 2i)
def z : ℂ := 1 - 2 * i

-- Define the expanded form of (1 - 2i)^2
def z_squared : ℂ := z^2

-- State the problem of finding the imaginary part of (1 - 2i)^2
theorem imaginary_part_of_z_squared : (z_squared).im = -4 := by
  sorry

end imaginary_part_of_z_squared_l651_651587


namespace finite_repeated_values_l651_651451

-- Define the Fibonacci type sequence
def fibonacci_seq (A B : ℕ) : ℕ → ℤ
| 0     := A
| 1     := B
| (n+2) := fibonacci_seq n + fibonacci_seq (n + 1)

-- Prove that we can choose A and B such that the sequence has finitely many repeated values
theorem finite_repeated_values (A B : ℕ) :
  ∃ (S : set ℤ) (n : ℕ), 
  (∀ (i j : ℕ), i ≠ j → fibonacci_seq A B i ≠ fibonacci_seq A B j ∨ (fibonacci_seq A B i ∈ S ∧ i < n)) :=
sorry

end finite_repeated_values_l651_651451


namespace square_area_from_perimeter_l651_651314

theorem square_area_from_perimeter (P : ℕ) (h : P = 80) : ∃ A : ℕ, A = 400 :=
by
  -- let side length be s
  let s := P / 4
  -- let area be A
  let A := s * s
  -- we know P = 80
  have h1 : P = 80 := h
  -- therefore s = 80 / 4 = 20
  have s_eq_20 : s = 20 :=
    by
      rw [h1]
      norm_num
  -- and thus A = 20 * 20 = 400
  have A_eq_400 : A = 400 :=
    by
      rw [s_eq_20]
      norm_num
  -- hence, the required A exists and equals 400
  exact ⟨A, A_eq_400⟩


end square_area_from_perimeter_l651_651314


namespace angular_measures_of_arcs_l651_651252

def triangle_ABC (α β γ : ℝ) (C_right : γ = 90) (B_angle : β = 40) (sum_angles : α + β + γ = 180) : Prop :=
  α = 50 ∧ β = 40 ∧ γ = 90

def arc_measure (α β γ : ℝ) (D_intersect : γ = 90) (arc_AD : γ - 2 * α) (arc_DE : γ - (γ - 2 * α)) : Prop :=
  arc_AD = 80 ∧ arc_DE = 10

theorem angular_measures_of_arcs :
  ∀ α β γ : ℝ, triangle_ABC α β γ 90 40 (α + β + γ = 180) → arc_measure α β γ 90 (90 - 2 * α) (90 - (90 - 2 * α)) :=
by
  intros α β γ h_tri
  sorry

end angular_measures_of_arcs_l651_651252


namespace sum_of_a_and_b_l651_651070

theorem sum_of_a_and_b (a b : ℕ) (h1 : a > 0) (h2 : b > 1) (h3 : a^b < 500) (h_max : ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a'^b' ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l651_651070


namespace cube_cut_problem_l651_651496

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651496


namespace find_k_l651_651638

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

noncomputable def C(n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_k (k : ℕ) (h : k ∈ A) : (2 / 5 : ℚ) = ((10 - k) * (k - 1)) / C 10 2 → (k = 4 ∨ k = 7) :=
by
  intro h1
  sorry

end find_k_l651_651638


namespace polynomial_couples_l651_651968

noncomputable def P (C : ℝ) (x : ℝ) : ℝ := 2 * C * x^2 + 2 * (C + 1) * x + 1
noncomputable def Q (x : ℝ) : ℝ := 2 * x * (x + 1)

theorem polynomial_couples (P Q : ℝ → ℝ)
  (h : ∀ x : ℝ, Q x ≠ 0 ∧ Q (x + 1) ≠ 0 ∧ (P x / Q x - P (x + 1) / Q (x + 1) = 1 / (x * (x + 2)))) :
  ∃ C : ℝ, (∀ x : ℝ, P x = 2 * C * x^2 + 2 * (C + 1) * x + 1) ∧ (∀ x : ℝ, Q x = 2 * x * (x + 1)) :=
begin
  sorry
end

end polynomial_couples_l651_651968


namespace number_of_pieces_of_paper_l651_651075

def three_digit_number_with_unique_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n / 100 ≠ (n / 10) % 10 ∧ n / 100 ≠ n % 10 ∧ (n / 10) % 10 ≠ n % 10

theorem number_of_pieces_of_paper (n : ℕ) (k : ℕ) (h1 : three_digit_number_with_unique_digits n) (h2 : 2331 = k * n) : k = 9 :=
by
  sorry

end number_of_pieces_of_paper_l651_651075


namespace nineteen_cards_divisible_by_eleven_l651_651588

theorem nineteen_cards_divisible_by_eleven :
  ∃ (cards : Fin 19 → ℕ), (∀ i, cards i ≠ 0 ∧ cards i < 10) ∧ 
                            ((List.alternating_sum (Fin 19) cards) % 11 = 0) := 
by
  sorry

end nineteen_cards_divisible_by_eleven_l651_651588


namespace round_trip_time_correct_l651_651443

def travel_time (distance: ℝ) (speed: ℝ) : ℝ := distance / speed

def total_round_trip_time : ℝ :=
  let time_AB_car := travel_time 40 80
  let time_AB_train := travel_time 60 120
  let time_AB_bike := travel_time 20 20
  let time_BA_car := travel_time 40 120
  let time_BA_train := travel_time 60 96
  let time_BA_bike := travel_time 20 20
  (time_AB_car + time_AB_train + time_AB_bike) + (time_BA_car + time_BA_train + time_BA_bike)

theorem round_trip_time_correct : abs (total_round_trip_time - 3.9583) < 0.0001 := by sorry

end round_trip_time_correct_l651_651443


namespace not_in_range_g_zero_l651_651738

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l651_651738


namespace number_of_correct_statements_l651_651621

theorem number_of_correct_statements (x a b : ℝ)
  (h1 : x^2 + 3 > 2x)
  (h2 : ¬ (a^5 + b^5 ≥ a^3 b^2 + a^2 b^3))
  (h3 : a^2 + b^2 ≥ 2 * (a - b - 1)) : 
  ( (x^2 + 3 > 2x) 
    ∧ (¬ (a^5 + b^5 ≥ a^3 b^2 + a^2 b^3)) 
    ∧ (a^2 + b^2 ≥ 2 * (a - b - 1))
  ) → 2 
:= by
  sorry

end number_of_correct_statements_l651_651621


namespace alyosha_cube_cut_l651_651503

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651503


namespace find_m_l651_651613

theorem find_m
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ)
  (h₁ : a = (1, 2, -2))
  (h₂ : b = (-2, 3, m))
  (h₃ : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :
  m = 2 :=
sorry

end find_m_l651_651613


namespace power_sum_equality_l651_651793

theorem power_sum_equality (x : ℂ) (h : x^(2018) - 2*(x^2) + 1 = 0) (h1 : x ≠ 1) : 
  x^(2017) + x^(2016) + ... + x^2 + 1 = 4 :=
sorry

end power_sum_equality_l651_651793


namespace remainder_17_pow_63_mod_7_l651_651375

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651375


namespace largest_of_three_l651_651414

theorem largest_of_three (a b c : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 4) : max a (max b c) = 8 := 
sorry

end largest_of_three_l651_651414


namespace sunny_candles_per_cake_l651_651791

/-- 
Sunny bakes 8 cakes, gives away 2 cakes, and uses a total of 36 candles. The candles are distributed equally among the remaining cakes. We need to prove that Sunny puts 6 candles on each cake. 
-/
theorem sunny_candles_per_cake :
  ∀ (cakes_baked cakes_given away cakes_left total_candles candles_per_cake : ℕ),
  cakes_baked = 8 →
  cakes_given_away = 2 →
  total_candles = 36 →
  cakes_left = cakes_baked - cakes_given_away →
  candles_per_cake = total_candles / cakes_left →
  candles_per_cake = 6 :=
by
  intros cakes_baked cakes_given_away cakes_left total_candles candles_per_cake 
  intro h1 h2 h3 h4 h5
  rw [h1] at h4
  rw [h2] at h4
  rw [h4] at h5
  rw [h3] at h5
  norm_num at h5
  exact h5
  
# Print sunny_candles_per_cake -- Uncomment to print the theorem 


end sunny_candles_per_cake_l651_651791


namespace tiles_needed_for_classroom_l651_651894

theorem tiles_needed_for_classroom : 
  ∀ (length width : ℕ), length = 624 → width = 432 → 
  ∃ (tiles : ℕ), tiles = 117 :=
by
  intros length width h_length h_width
  let g := Nat.gcd length width
  have h1 : length = 624, from h_length
  have h2 : width = 432, from h_width
  have h_gcd : g = Nat.gcd 624 432 := by
    rw [h1, h2]
  let area := (length * width) / (g * g)
  use area
  sorry

end tiles_needed_for_classroom_l651_651894


namespace probability_sum_468_theorem_l651_651438

def die1 : list ℕ := [1, 2, 2, 3, 3, 3]
def die2 : list ℕ := [1, 1, 2, 6, 7, 7]

noncomputable def favorable_sums : ℕ → ℕ → bool
| a, b => (a + b = 4) ∨ (a + b = 6) ∨ (a + b = 8)

noncomputable def probability_sum_468 (d1 d2 : list ℕ) : ℚ :=
  let favorable_pairs := d1.product d2 |>.filter (λ p => favorable_sums p.fst p.snd)
  (favorable_pairs.length : ℚ) / (d1.length * d2.length : ℚ)

theorem probability_sum_468_theorem :
  probability_sum_468 die1 die2 = 5 / 18 :=
sorry

end probability_sum_468_theorem_l651_651438


namespace find_ω_φ_l651_651338

-- Definition of the problem variables and functions
def function_form (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Given conditions
def period_condition (x1 x2 T : ℝ) : Prop :=
  x2 - x1 = T / 4

def angular_frequency (ω T : ℝ) : Prop :=
  ω = 2 * Real.pi / T

def phase_shift_condition (ω φ x : ℝ) : Prop :=
  (ω * x + φ) = Real.pi / 2

-- Statement of the proof problem
theorem find_ω_φ : 
  ∀ x1 x2 (ω φ : ℝ),
  period_condition x1 x2 (8 : ℝ) → 
  angular_frequency ω (8 : ℝ) →
  phase_shift_condition ω φ (1 : ℝ) →
  ω = Real.pi / 4 ∧ φ = Real.pi / 4 :=
by
  intros
  -- The proof will be filled in here
  sorry

end find_ω_φ_l651_651338


namespace scientific_notation_correct_l651_651555

theorem scientific_notation_correct :
  (0.00000428 : ℝ) = 4.28 * 10^(-6) :=
sorry

end scientific_notation_correct_l651_651555


namespace x_coordinate_second_point_l651_651247

theorem x_coordinate_second_point (m n : ℝ) 
(h₁ : m = 2 * n + 5)
(h₂ : m + 2 = 2 * (n + 1) + 5) : 
  (m + 2) = 2 * n + 7 :=
by sorry

end x_coordinate_second_point_l651_651247


namespace solve_for_m_l651_651207

theorem solve_for_m : 
  ∀ m : ℝ, (3 * (-2) + 5 = -2 - m) → m = -1 :=
by
  intros m h
  sorry

end solve_for_m_l651_651207


namespace bins_of_soup_l651_651547

theorem bins_of_soup (total_bins : ℝ) (bins_of_vegetables : ℝ) (bins_of_pasta : ℝ) 
(h1 : total_bins = 0.75) (h2 : bins_of_vegetables = 0.125) (h3 : bins_of_pasta = 0.5) :
  total_bins - (bins_of_vegetables + bins_of_pasta) = 0.125 := by
  -- proof
  sorry

end bins_of_soup_l651_651547


namespace independent_events_probability_l651_651008

variables (A B : Type) (P : Set A → ℚ)
-- Conditions
variables (hA : P {a | a = a} = 5/7)
variables (hB : P {b | b = b} = 2/5)
variables (indep : ∀ (A B : Set A), P (A ∩ B) = P A * P B)

-- Statement
theorem independent_events_probability (A B : Set A) (P : Set A → ℚ)
  (hA : P A = 5 / 7)
  (hB : P B = 2 / 5)
  (indep : P (A ∩ B) = P A * P B) :
  P (A ∩ B) = 2 / 7 :=
by sorry

end independent_events_probability_l651_651008


namespace element_subset_a_l651_651706

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l651_651706


namespace coprime_product_mod_120_l651_651282

theorem coprime_product_mod_120 {a : ℕ} :
  let n := 120 in
  ∃ product_mod : ℕ, product_mod ≡ a [MOD n] ∧
    product_mod = ∏ i in (Finset.filter (λ x => Nat.gcd x n = 1) (Finset.range n)), i :=
sorry

end coprime_product_mod_120_l651_651282


namespace coin_flip_probability_l651_651870

theorem coin_flip_probability :
    let p := (1/2 : ℚ) in
    (p * p * p * p * (1 - p) = 1 / 32) :=
by
  let p := (1/2 : ℚ)
  show p * p * p * p * (1 - p) = 1 / 32
  sorry

end coin_flip_probability_l651_651870


namespace DE_passes_through_fixed_point_l651_651146

-- Definition of a triangle with a given circumcircle
variables {A B C : Point} (circumcircle : Circle)

-- Hypothesis: |AC| < |AB|
axiom AC_lt_AB : dist A C < dist A B

-- Definition of D as a variable point on the short arc AC, excluding A
variables {D : Point}
axiom D_on_short_arc_AC : D ∈ circumcircle ∧ D ≠ A ∧ is_on_short_arc A C D circumcircle

-- Definition of E as the reflection of A with respect to the internal bisector of ∠BDC
variables {E : Point}
axiom E_reflection_A : is_reflection E A (internal_bisector (angle B D C))

-- Hypothesis to express the passage through a fixed point
theorem DE_passes_through_fixed_point :
  ∃ F : Point, ∀ (D : Point), D ∈ circumcircle ∧ D ≠ A ∧ is_on_short_arc A C D circumcircle →
  let E := reflect A (internal_bisector (angle B D C)) in line_through D E ∋ F := sorry

end DE_passes_through_fixed_point_l651_651146


namespace rigged_coin_probability_l651_651896

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1 / 2) (h2 : 20 * (p ^ 3) * ((1 - p) ^ 3) = 1 / 12) :
  p = (1 - Real.sqrt 0.86) / 2 :=
by
  sorry

end rigged_coin_probability_l651_651896


namespace find_lambda_l651_651150

-- Definitions
variables {A B C D : Type} [add_comm_group A] [vector_space ℝ A]
variables (DA DB CB : A)
variables (collinear : ∀ {u v w : A}, ∃ a b c : ℝ, a • u + b • v + c • w = 0)

-- The main statement to prove
theorem find_lambda (λ : ℝ) (h : DA = 2 * λ • DB + 3 • CB) (h_collinear: collinear A B C) :
  λ = 1 / 2 := 
sorry

end find_lambda_l651_651150


namespace population_estimation_correct_l651_651099

-- Definition of the initial population and doubling rate conditions
def initial_population (year: ℕ) := if year = 2020 then 250 else 0
def doubling_rate (years_between_doubling: ℕ) := years_between_doubling = 20

-- Proof statement
theorem population_estimation_correct :
  (∀ year, initial_population 2020 = 250 ∧ doubling_rate 20) →
  (year_in_which_population_reaches ≈ 5000) = 2100 :=
sorry

end population_estimation_correct_l651_651099


namespace man_distance_from_start_l651_651910

noncomputable def distance_from_start (west_distance north_distance : ℝ) : ℝ :=
  Real.sqrt (west_distance^2 + north_distance^2)

theorem man_distance_from_start :
  distance_from_start 10 10 = Real.sqrt 200 :=
by
  sorry

end man_distance_from_start_l651_651910


namespace sqrt_of_sixteen_l651_651812

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l651_651812


namespace value_depletion_rate_l651_651040

theorem value_depletion_rate (P F : ℝ) (t : ℝ) (r : ℝ) (h₁ : P = 1100) (h₂ : F = 891) (h₃ : t = 2) (decay_formula : F = P * (1 - r) ^ t) : r = 0.1 :=
by 
  sorry

end value_depletion_rate_l651_651040


namespace correct_relation_l651_651516

-- Definitions of sets and elements
variable {α : Type*} -- introducing a type α
variable (a b : α)
variable (s : set α)

-- Main statement
theorem correct_relation : a ∈ {a, b} :=
  sorry

end correct_relation_l651_651516


namespace tangent_circle_arc_angle_l651_651348

theorem tangent_circle_arc_angle (A O B C : Point) (hTangentB : is_tangent A B O) 
  (hTangentC : is_tangent A C O) (hCircleBOC : on_circle B O C) 
  (hArcRatio : arc_length B C / arc_length C B' = 3 / 5) :
  ∠BAC = 67.5 :=
sorry

end tangent_circle_arc_angle_l651_651348


namespace prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l651_651801

theorem prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29 
  (n : ℕ) (h1 : Prime n) (h2 : 20 < n) (h3 : n < 30) (h4 : n % 8 = 5) : n = 29 := 
by
  sorry

end prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l651_651801


namespace total_selling_price_correct_l651_651917

-- Define the conditions
def metres_of_cloth : ℕ := 500
def loss_per_metre : ℕ := 5
def cost_price_per_metre : ℕ := 41
def selling_price_per_metre : ℕ := cost_price_per_metre - loss_per_metre
def expected_total_selling_price : ℕ := 18000

-- Define the theorem
theorem total_selling_price_correct : 
  selling_price_per_metre * metres_of_cloth = expected_total_selling_price := 
by
  sorry

end total_selling_price_correct_l651_651917


namespace club_organizing_teams_l651_651895

theorem club_organizing_teams (total_members seniors : ℕ) (condition1 : total_members = 12) (condition2 : seniors = 5) :
  let non_seniors := total_members - seniors,
      total_teams := Nat.choose total_members 5,
      teams_with_no_seniors := Nat.choose non_seniors 5,
      teams_with_one_senior := seniors * Nat.choose non_seniors 4
  in total_teams - (teams_with_no_seniors + teams_with_one_senior) = 596 :=
by
  sorry

end club_organizing_teams_l651_651895


namespace cube_cut_problem_l651_651493

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651493


namespace interval_proof_l651_651542

noncomputable def valid_interval (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y

theorem interval_proof : ∀ x : ℝ, valid_interval x ↔ (0 ≤ x ∧ x < 4) :=
by
  sorry

end interval_proof_l651_651542


namespace coordinates_of_A_and_B_area_of_trapezoid_MNBA_area_of_quadrilateral_VPOQ_l651_651336

-- Define the parabola
def parabola (x : ℝ) : ℝ := -x^2 + 16

-- Define the x-intercepts (A and B)
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the points M and N when y = 7
def M : ℝ × ℝ := (-3, 7)
def N : ℝ × ℝ := (3, 7)

-- Define the points P and Q when y = -33
def P : ℝ × ℝ := (-7, -33)
def Q : ℝ × ℝ := (7, -33)

-- Define the vertex of the parabola
def V : ℝ × ℝ := (0, 16)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Lean 4 theorem statements
theorem coordinates_of_A_and_B : A = (-4, 0) ∧ B = (4, 0) := by
  sorry

theorem area_of_trapezoid_MNBA : 
  let lengthMN := (N.1 - M.1).abs
  let lengthAB := (B.1 - A.1).abs
  let height := (M.2 - A.2).abs
  (1 / 2) * (lengthMN + lengthAB) * height = 49 := by
  sorry

theorem area_of_quadrilateral_VPOQ : 
  let baseVO := (V.2 - O.2).abs
  let heightVP := (P.1 - V.1).abs
  let areaVOP := (1 / 2) * baseVO * heightVP
  (2 * areaVOP) = 112 := by
  sorry

end coordinates_of_A_and_B_area_of_trapezoid_MNBA_area_of_quadrilateral_VPOQ_l651_651336


namespace graph_symmetric_about_x_2_l651_651121

variables {D : Set ℝ} {f : ℝ → ℝ}

theorem graph_symmetric_about_x_2 (h : ∀ x ∈ D, f (x + 1) = f (-x + 3)) : 
  ∀ x ∈ D, f (x) = f (4 - x) :=
by
  sorry

end graph_symmetric_about_x_2_l651_651121


namespace complex_division_l651_651160

theorem complex_division (i : ℂ) (h : i ^ 2 = -1) : (3 - 4 * i) / i = -4 - 3 * i :=
by
  sorry

end complex_division_l651_651160


namespace slope_of_chord_AB_l651_651892

/-- Definition of a point --/
structure Point := (x : ℝ) (y : ℝ)

/-- Definition of a line --/
structure Line := (slope : ℝ) (intercept : ℝ)

/-- Given conditions --/
def P : Point := ⟨2, -2⟩
def parabola (x y : ℝ) : Prop := x^2 = -2*y

/-- Theorem statement: existence and uniqueness of the slope of chord AB --/
theorem slope_of_chord_AB :
  (∃ A B : Point, parabola A.x A.y ∧ parabola B.x B.y ∧
    Line.through_two_points A P ∧ Line.through_two_points B P ∧
    ∃ (s1 s2 : ℝ), complementary_inclination P A B s1 s2 ∧
    chord_of_slope A B 2) :=
sorry

/-- Complementary inclination definition: two slopes whose angles sum to 90 degrees --/
def complementary_inclination (P A B : Point) (slope1 slope2 : ℝ) : Prop :=
slope1 * slope2 = -1

/-- Chord with a given slope --/
def chord_of_slope (A B : Point) (slope : ℝ) : Prop :=
(A.y - B.y = slope * (A.x - B.x))

/-- Line through two points --/
def Line.through_two_points (A B : Point) : Prop :=
∃ (m : ℝ), (A.y - B.y = m * (A.x - B.x)) still uncertain

end slope_of_chord_AB_l651_651892


namespace problem_statement_l651_651597

variables {α : Type*} [has_mul α] (※ : α → α → α) (a b c : α)

-- Definitions given in conditions
def op_assoc : Prop := ∀ a b c : α, a ※ (b ※ c) = (a ※ b) * c
def op_self : Prop := ∀ a : α, a ※ a = 1

theorem problem_statement (x p q : ℚ) (h₀ : op_assoc ※) (h₁ : op_self ※) (gcd : p.nat_abs.gcd q.nat_abs = 1) (h₂ : (2016 ※ (6 ※ x) = 100) → x = p / q) : p + q = 529 :=
sorry

end problem_statement_l651_651597


namespace not_in_range_of_g_l651_651733

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else undefined

theorem not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g(x) ≠ 0 :=
by sorry

end not_in_range_of_g_l651_651733


namespace sum_of_divisors_360_l651_651394

-- Define the function to calculate the sum of divisors
def sum_of_divisors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

-- Statement of the problem
theorem sum_of_divisors_360 : sum_of_divisors 360 = 1170 :=
by
  sorry

end sum_of_divisors_360_l651_651394


namespace gu_gu_number_count_l651_651698

open Int

-- Necessary definitions and conditions
def is_gu_gu_number (p : ℤ) (a : ℤ) : Prop :=
  p ∣ ∏ i in finset.range (nat_abs p + 1), (i^3 - a * i - 1)

def gu_gu_count (p : ℤ) : ℤ :=
  (finset.range (nat_abs p)).count (λ a, is_gu_gu_number p a)

theorem gu_gu_number_count (p : ℤ) (hp_prime : nat.prime (nat_abs p)) (hp_mod : p % 3 = 2) :
  gu_gu_count p = (2 * p - 1) / 3 := 
  sorry

end gu_gu_number_count_l651_651698


namespace irrational_sqrt_3_l651_651410

theorem irrational_sqrt_3 :
  let A := (1 / 2 : ℝ)
  let B := (0.2 : ℝ)
  let C := (-5 : ℝ)
  let D := (real.sqrt 3)
  irrational D ∧ (¬ irrational A) ∧ (¬ irrational B) ∧ (¬ irrational C) :=
by
  sorry

end irrational_sqrt_3_l651_651410


namespace option_d_correct_factorization_l651_651327

theorem option_d_correct_factorization (x : ℝ) : 
  -8 * x ^ 2 + 8 * x - 2 = -2 * (2 * x - 1) ^ 2 :=
by 
  sorry

end option_d_correct_factorization_l651_651327


namespace cosine_varphi_rhombus_cosine_varphi_rectangle_l651_651136

-- Define the setup and geometric conditions based on the given problem
def parallelogram_conditions (A B C D M N : Point) (alpha beta : Plane) :=
  IsParallelogram ABCD ∧
  A ∈ MN ∧ B ∈ alpha ∧ C ∈ alpha ∧ D ∈ alpha ∧
  Distance A B = 2 * Distance A D ∧
  Angle D A N = 45 ∧
  Angle B A D = 60

-- Define the problem for the rhombus case
theorem cosine_varphi_rhombus (A B C D M N : Point) (alpha beta : Plane) 
  (h : parallelogram_conditions A B C D M N alpha beta) : 
  CosineVarphi alpha beta = 4 - Real.sqrt 3 :=
sorry

-- Define the problem for the rectangle case
theorem cosine_varphi_rectangle (A B C D M N : Point) (alpha beta : Plane) 
  (h : parallelogram_conditions A B C D M N alpha beta) : 
  CosineVarphi alpha beta = Real.sqrt 3 - 1 :=
sorry

end cosine_varphi_rhombus_cosine_varphi_rectangle_l651_651136


namespace arithmetic_sequence_general_term_and_k_l651_651684

theorem arithmetic_sequence_general_term_and_k (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77) :
  (∀ n : ℕ, a n = (2 * n + 3) / 3) ∧ (∃ k : ℕ, a k = 13 ∧ k = 18) := 
by
  sorry

end arithmetic_sequence_general_term_and_k_l651_651684


namespace problem1_problem2_l651_651181

/-
Define the constants and conditions given in the problem
-/
def l_line (x y : ℝ) : Prop := y = sqrt 3 * x - 2 * sqrt 3
def x_line (a c x : ℝ) : Prop := x = a^2 / c
def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def condition_abc (a b c : ℝ) : Prop := a > b ∧ b > 0

/-
Define the equations to be proven
-/
def ellipse_equation : Prop :=
  ∀ a b c : ℝ, l_line 0 2 ∧ x_line a c 3 ∧ condition_abc a b c → 
  ellipse 6 2

def max_lambda_value (lambda : ℝ) : Prop :=
  ∀ t : ℝ, ∃ y1 y2 : ℝ, (4 * t) / (3 + t^2) = y1 + y2 ∧ (-2) / (3 + t^2) = y1 * y2 ∧ 
  (2 * sqrt 6 * sqrt (1 + t^2)) / (3 + t^2) < sqrt 3 →
  lambda = sqrt 3

/-
Formalizing the final statements to be proven
-/
theorem problem1 (a b c : ℝ) : 
  l_line 0 2 ∧ x_line a c 3 ∧ condition_abc a b c → 
  ellipse 6 2 :=
sorry

theorem problem2 (lambda : ℝ) : ∀ t : ℝ, 
  ∃ y1 y2 : ℝ, (4 * t) / (3 + t^2) = y1 + y2 ∧ (-2) / (3 + t^2) = y1 * y2 ∧ 
  (2 * sqrt 6 * sqrt (1 + t^2)) / (3 + t^2) < sqrt 3 →
  max_lambda_value lambda :=
sorry

end problem1_problem2_l651_651181


namespace find_p_l651_651175

def f (x : ℝ) (p : ℝ) : ℝ :=
  if x < 2 then 2^x + 1 else x^2 + p * x

theorem find_p (p : ℝ) (h : f (f 0 p) p = 5 * p) : p = 4 / 3 :=
by
  sorry

end find_p_l651_651175


namespace geometric_sequence_sum_l651_651753

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_ratio_ne_one : q ≠ 1)
  (S : ℕ → ℝ) (h_a1 : S 1 = 1) (h_S4_eq_5S2 : S 4 - 5 * S 2 = 0) :
  S 5 = 31 :=
sorry

end geometric_sequence_sum_l651_651753


namespace number_of_men_l651_651067

theorem number_of_men (M W C : ℕ) 
  (h1 : M + W + C = 10000)
  (h2 : C = 2500)
  (h3 : C = 5 * W) : 
  M = 7000 := 
by
  sorry

end number_of_men_l651_651067


namespace det_N_pow_5_l651_651606

variable (N : Matrix n n ℝ) (h : det N = 3)

theorem det_N_pow_5 : det (N ^ 5) = 243 :=
by sorry

end det_N_pow_5_l651_651606


namespace hypocycloid_theorem_l651_651038

noncomputable def prove_hypocycloids_and_radii (x y : ℝ → ℝ) : Prop :=
  (∀ t, (deriv (deriv x) t + deriv y t + 6 * (x t) = 0) ∧ (deriv (deriv y) t - deriv x t + 6 * (y t) = 0)) →
  (deriv x 0 = 0 ∧ deriv y 0 = 0) →
  (∃ (c : ℝ) (R : ℝ) (t : ℝ), (x t = (c-1)*R * cos(t) + R * cos((c-1)*t)) ∧ (y t = (c-1)*R * sin(t) - R * sin((c-1)*t))) ∧
  (c = 5 / 2 ∨ c = 5 / 3)

-- The statement of the theorem
theorem hypocycloid_theorem : ∃ (x y : ℝ → ℝ), prove_hypocycloids_and_radii x y :=
sorry

end hypocycloid_theorem_l651_651038


namespace points_at_fixed_distance_in_space_form_spherical_surface_l651_651342

theorem points_at_fixed_distance_in_space_form_spherical_surface
  (P : Type) [metric_space P] (c : P) (r : ℝ) (h : r > 0) :
  {p : P | dist p c = r} = {p : P | ∃ (x y z : ℝ), p = (x, y, z) ∧ dist (x, y, z) (0, 0, 0) = r} :=
by
  sorry

end points_at_fixed_distance_in_space_form_spherical_surface_l651_651342


namespace find_tangent_line_l651_651618

-- Define the circle and the given point
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2
def pointN := (1 : ℝ, 2 : ℝ)

-- Define the tangent line equation to be proved
def tangent_line (x y : ℝ) : Prop := x + y - 3 = 0

-- Prove that the tangent line passing through the point (1,2) has the equation x + y - 3 = 0
theorem find_tangent_line :
  ∃ (m b : ℝ), (∀ x y : ℝ, tangent_line x y ↔ y = -1 * x + b) ∧ 
               tangent_line 1 2 ∧ 
               (∀ p : prod ℝ ℝ, (circle p.1 p.2) → tangent_line p.1 p.2) := 
sorry

end find_tangent_line_l651_651618


namespace enclosed_area_l651_651848

theorem enclosed_area (x y : ℝ) :
  (|5 * x| + |6 * y| = 30) ∧ (x^2 + y^2 ≤ 25) → 
  ∃ A, A = 25 * real.pi :=
by
  sorry

end enclosed_area_l651_651848


namespace intervals_of_monotonic_increase_range_of_f_on_interval_l651_651627

open Real

noncomputable def f (x: ℝ) : ℝ := sin (2 * x - π / 3)

theorem intervals_of_monotonic_increase (k : ℤ) :
  {x : ℝ | -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π}.nonempty :=
sorry

theorem range_of_f_on_interval :
  ∀ x ∈ (Icc 0 (π / 2)), -sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 :=
sorry

end intervals_of_monotonic_increase_range_of_f_on_interval_l651_651627


namespace ceil_sqrt_sum_eq_24_l651_651552

theorem ceil_sqrt_sum_eq_24:
  1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 →
  5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 →
  15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 →
  Int.ceil (Real.sqrt 3) + Int.ceil (Real.sqrt 27) + Int.ceil (Real.sqrt 243) = 24 :=
by
  intros h1 h2 h3
  have h1_ceil := Real.ceil_sqrt_of_lt_of_gt h1.left h1.right
  have h2_ceil := Real.ceil_sqrt_of_lt_of_gt h2.left h2.right
  have h3_ceil := Real.ceil_sqrt_of_lt_of_gt h3.left h3.right
  simp [h1_ceil, h2_ceil, h3_ceil]
  sorry

end ceil_sqrt_sum_eq_24_l651_651552


namespace range_of_a_l651_651775

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  a > 1

-- Translate the problem to a Lean 4 statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Set.Icc (-2 : ℝ) 1 ∪ Set.Ici 2 :=
by
  sorry

end range_of_a_l651_651775


namespace evaluate_number_l651_651965

theorem evaluate_number (n : ℝ) (h : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24) : n = 1 / 3 :=
by
  sorry

end evaluate_number_l651_651965


namespace outer_boundary_diameter_l651_651050

theorem outer_boundary_diameter (d_pond : ℝ) (w_picnic : ℝ) (w_track : ℝ)
  (h_pond_diam : d_pond = 16) (h_picnic_width : w_picnic = 10) (h_track_width : w_track = 4) :
  2 * (d_pond / 2 + w_picnic + w_track) = 44 :=
by
  -- We avoid the entire proof, we only assert the statement in Lean
  sorry

end outer_boundary_diameter_l651_651050


namespace trains_crossing_time_correct_l651_651872

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end trains_crossing_time_correct_l651_651872


namespace probability_of_red_given_red_l651_651133

noncomputable theory
open_locale big_operators

variables {n m : ℕ}

-- Definitions based on conditions in the problem
axiom condition1 : 1 - ((m * (m-1)) / (n * (n-1)) : ℚ) = 3 / 5
axiom condition2 : (6 * (m / n : ℚ)) = 4

-- The problem statement in Lean
theorem probability_of_red_given_red :
  ( (m - 2) * ((m - 1) / (n - 1) : ℚ) + ((n - m) - 1) * ((2 / 3) : ℚ) ) / (m + n - 1) = 1 / 5 :=
sorry

end probability_of_red_given_red_l651_651133


namespace find_f_2007_l651_651329

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2007 (h1 : ∀ x y > 0, f (x * y) = f x + f y)
    (h2 : f (1007 / 1024) = 1) : f 2007 = -1 :=
sorry

end find_f_2007_l651_651329


namespace minimum_value_of_2_pow_a_plus_4_pow_b_l651_651163

theorem minimum_value_of_2_pow_a_plus_4_pow_b (a b : ℝ) (h : a + 2 * b = -3) : 2^a + 4^b = sqrt 2 / 2 :=
sorry

end minimum_value_of_2_pow_a_plus_4_pow_b_l651_651163


namespace shaded_region_area_l651_651031

-- Conditions
def O : Point := sorry  -- Center of the circle
def R : Real := 3       -- Radius of the circle
def A : Point := sorry  -- Point A on the rectangle OABC
def B : Point := sorry  -- Point B on the rectangle OABC
def C : Point := sorry  -- Point C on the rectangle OABC
def OA : Real := 2      -- Side length OA of the rectangle
def AB : Real := 1      -- Side length AB of the rectangle
def D : Point := sorry  -- Point D on the circle where AB is extended past B
def E : Point := sorry  -- Point E on the circle where CB is extended past B

-- Proof to be completed
theorem shaded_region_area : 
  ∀ (O A B C D E : Point) (R OA AB : Real), 
  (R = 3) → 
  (OA = 2) → 
  (AB = 1) → 
  (inside_circle R O D) → 
  (inside_circle R O E) → 
  (area_shaded BD BE arc_DE = 6.23) :=
sorry

end shaded_region_area_l651_651031


namespace second_intersection_points_circle_l651_651141

noncomputable theory

variables {Sphere : Type*} {Circle : Type*} {Point : Type*}

-- Define the conditions
structure sphere (S : Type*) :=
(center : Point)
(radius : ℝ)

structure circle (S : Type*) :=
(center : Point)
(radius : ℝ)
(inside : sphere S)

-- Given conditions
variables (sphere1 : sphere Sphere) (circle_S : circle Sphere) (P : Point)
(hP : P ≠ sphere1.center)

-- Statement to prove
theorem second_intersection_points_circle :
  ∀ Q ∈ circle_S, ∃ circle' : circle Sphere,
  (∃ X Y : Point, line_through P X ∧ line_through P Y ∧ 
   X ∈ sphere1 ∧ Y ∈ sphere1 ∧ Q ∈ circle_S ∧
   Y ≠ Q ∧ Y ∈ circle'.inside) :=
sorry

end second_intersection_points_circle_l651_651141


namespace arithmetic_mean_correct_l651_651140

variable (n : ℕ) (hn : n > 1)

def set_of_numbers (i : ℕ) : ℝ :=
  if i = 0 then 1 + 1 / n else 1

noncomputable def arithmetic_mean (n : ℕ) (hn : n > 1) : ℝ :=
  (∑ i in Finset.range n, set_of_numbers n i) / n

theorem arithmetic_mean_correct (n : ℕ) (hn : n > 1) :
  arithmetic_mean n hn = 1 + 1 / n^2 := by
  sorry

end arithmetic_mean_correct_l651_651140


namespace solve_AlyoshaCube_l651_651511

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651511


namespace andrei_kolya_ages_l651_651522

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + (n / 1000)

theorem andrei_kolya_ages :
  ∃ (y1 y2 : ℕ), (sum_of_digits y1 = 2021 - y1) ∧ (sum_of_digits y2 = 2021 - y2) ∧ (y1 ≠ y2) ∧ ((2022 - y1 = 8 ∧ 2022 - y2 = 26) ∨ (2022 - y1 = 26 ∧ 2022 - y2 = 8)) :=
by
  sorry

end andrei_kolya_ages_l651_651522


namespace eval_g_l651_651082

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem eval_g : 3 * g 2 + 4 * g (-4) = 327 := 
by
  sorry

end eval_g_l651_651082


namespace subset_a_eq_1_l651_651704

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651704


namespace find_n_l651_651482

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651482


namespace extremum_of_function_l651_651326

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem extremum_of_function :
  (∀ x, f x ≥ -Real.exp 1) ∧ (f 1 = -Real.exp 1) ∧ (∀ M, ∃ x, f x > M) :=
by
  sorry

end extremum_of_function_l651_651326


namespace rectangle_diagonals_not_perpendicular_l651_651923

theorem rectangle_diagonals_not_perpendicular
  (rectangle : Type)
  (has_diagonals_equal : ∀ (r : rectangle), diagonals_are_equal r)
  (has_all_right_angles : ∀ (r : rectangle), all_angles_right r)
  (is_symmetrical : ∀ (r : rectangle), symmetrical r)
  (diagonals_not_necessarily_perpendicular : ∀ (r : rectangle), ¬ diagonals_are_perpendicular r) :
  ∃ (r : rectangle), ¬ diagonals_are_perpendicular r :=
by
  sorry

end rectangle_diagonals_not_perpendicular_l651_651923


namespace find_m_l651_651340

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 19 ∧
  a 2 = 98 ∧
  ∀ n, a (n + 2) = a n - 2 / (a n + 1)

theorem find_m (a : ℕ → ℝ) (h : sequence a) : a 933 = 0 :=
by
  sorry

end find_m_l651_651340


namespace subset_a_eq_1_l651_651723

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l651_651723


namespace amy_total_spending_l651_651527

def initial_tickets : ℕ := 33
def cost_per_ticket : ℝ := 1.50
def additional_tickets : ℕ := 21
def total_cost : ℝ := 81.00

theorem amy_total_spending :
  (initial_tickets * cost_per_ticket + additional_tickets * cost_per_ticket) = total_cost := 
sorry

end amy_total_spending_l651_651527


namespace alyosha_cube_problem_l651_651468

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l651_651468


namespace compute_expression_l651_651534

theorem compute_expression : 12 * (1 / 26) * 52 * 4 = 96 :=
by
  sorry

end compute_expression_l651_651534


namespace find_a_l651_651584

def expression (x a : ℝ) := (x + a)^2 * (2 * x - 1 / x)^5

theorem find_a (x a : ℝ) (h : ¬∃ n b : ℝ, b ≠ 0 ∧ expression x a = b * x^3) :
  a = 1 ∨ a = -1 :=
sorry

end find_a_l651_651584


namespace intersection_of_sets_l651_651879

open Set

theorem intersection_of_sets :
  let M := {2, 3, 4}
  let N := {0, 2, 3, 5}
  M ∩ N = {2, 3} :=
by {
  sorry
}

end intersection_of_sets_l651_651879


namespace dodecahedron_interior_diagonals_l651_651646

-- Definitions based on conditions
def dodecahedron (V : Type) [Fintype V] [DecidableEq V] : Prop :=
  Fintype.card V = 20 ∧ 
  ∀ (v : V), ∃ (adjacent : Finset V), adjacent.card = 3 ∧ 
  ∀ u ∈ adjacent, u ≠ v

-- Statement to prove
theorem dodecahedron_interior_diagonals {V : Type} [Fintype V] [DecidableEq V] 
  (h : dodecahedron V) : 
  let count_diagonals := (Fintype.card V * (Fintype.card V - 3)) / 2 
  in count_diagonals = 160 :=
by {
  sorry
}

end dodecahedron_interior_diagonals_l651_651646


namespace gain_percent_is_correct_l651_651861

noncomputable def gain_percent (CP SP : ℝ) : ℝ :=
  let gain := SP - CP
  (gain / CP) * 100

theorem gain_percent_is_correct :
  gain_percent 930 1210 = 30.11 :=
by
  sorry

end gain_percent_is_correct_l651_651861


namespace rational_root_neg_one_third_l651_651560

def P (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_neg_one_third : P (-1/3) = 0 :=
by
  have : (-1/3 : ℚ) ≠ 0 := by norm_num
  sorry

end rational_root_neg_one_third_l651_651560


namespace coin_flip_probability_l651_651305

theorem coin_flip_probability :
  let p : ℚ := 3
  let q : ℚ := 5 in
  (p / q).denom = q ∧ (p / q).num = p ∧ p.gcd q = 1 →
  p + q = 8 := 
by
  intros
  sorry

end coin_flip_probability_l651_651305


namespace inscribed_square_side_length_l651_651782

noncomputable def right_triangle := ∃ (A B C : Type),
  right_triangle A B C ∧ 
  dist A B = 6 ∧ 
  dist B C = 8 ∧ 
  dist A C = 10

noncomputable def square_inscribed := ∃ (X Y Z W : Type),
  square X Y Z W ∧
  inscribed_in_triangle X Y Z W (∑ AB : (dist A B), (dist B C), (dist A C))

theorem inscribed_square_side_length
  (h_triangle: right_triangle A B C)
  (h_square: square_inscribed X Y Z W) :
  ∃ s, s = (120 : ℝ) / (37 : ℝ) :=
sorry

end inscribed_square_side_length_l651_651782


namespace perpendicular_condition_l651_651642

variables (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)
def vector_a_b : ℝ × ℝ := (1 + 3, m - 2)

theorem perpendicular_condition : (vector_a_b.fst * vector_b.fst + vector_a_b.snd * vector_b.snd = 0) ↔ (m = 8) :=
sorry

end perpendicular_condition_l651_651642


namespace total_earnings_l651_651938

theorem total_earnings :
  (15 * 2) + (12 * 1.5) = 48 := by
  sorry

end total_earnings_l651_651938


namespace find_n_l651_651538

def sequence_t : ℕ → ℚ
| 1 := 1/3
| n := if n % 2 = 0 then 1 + sequence_t (n / 2) else 1 / sequence_t (n - 1)

theorem find_n (n : ℕ) (h : sequence_t n = 27 / 91) : n = 67 :=
by
  sorry

end find_n_l651_651538


namespace hypotenuse_length_50_l651_651852

theorem hypotenuse_length_50 (a b : ℕ) (h₁ : a = 14) (h₂ : b = 48) :
  ∃ c : ℕ, c = 50 ∧ c = Nat.sqrt (a^2 + b^2) :=
by
  sorry

end hypotenuse_length_50_l651_651852


namespace number_of_packages_sold_l651_651061

noncomputable def supplier_charges (P : ℕ) : ℕ :=
  if P ≤ 10 then 25 * P
  else 250 + 20 * (P - 10)

theorem number_of_packages_sold
  (supplier_received : ℕ)
  (percent_to_X : ℕ)
  (percent_to_Y : ℕ)
  (percent_to_Z : ℕ)
  (per_package_price : ℕ)
  (discount_percent : ℕ)
  (discount_threshold : ℕ)
  (P : ℕ)
  (h_received : supplier_received = 1340)
  (h_to_X : percent_to_X = 15)
  (h_to_Y : percent_to_Y = 15)
  (h_to_Z : percent_to_Z = 70)
  (h_full_price : per_package_price = 25)
  (h_discount : discount_percent = 4 * per_package_price / 5)
  (h_threshold : discount_threshold = 10)
  (h_calculation : supplier_charges P = supplier_received) : P = 65 := 
sorry

end number_of_packages_sold_l651_651061


namespace number_of_men_l651_651659

variable (W D X : ℝ)

theorem number_of_men (M_eq_2W : M = 2 * W)
  (wages_40_women : 21600 = 40 * W * D)
  (men_wages : 14400 = X * M * 20) :
  X = (2 / 3) * D :=
  by
  sorry

end number_of_men_l651_651659


namespace solve_AlyoshaCube_l651_651510

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651510


namespace annual_income_correct_l651_651108

-- Definitions based on conditions in the given problem
def investment : ℝ := 6800
def dividend_rate : ℝ := 0.20
def price_per_share : ℝ := 136
def face_value_per_share : ℝ := 100  -- Assuming a typical face value since it’s not explicitly stated

-- Theorem statement
theorem annual_income_correct :
  let number_of_shares := investment / price_per_share in
  let annual_income_per_share := dividend_rate * face_value_per_share in
  let total_annual_income := number_of_shares * annual_income_per_share in
  total_annual_income = 1000 :=
by
  sorry

end annual_income_correct_l651_651108


namespace minimum_vector_sum_is_3_l651_651600

noncomputable def minimum_vector_sum : Real :=
  let dist (a : ℝ) (θ : ℝ) : ℝ := 
    sqrt ((2 * cos θ - 5)^2 + (2 * sin θ - 6 + a)^2)
  Inf (dist 6 0)

theorem minimum_vector_sum_is_3 : minimum_vector_sum = 3 :=
  sorry

end minimum_vector_sum_is_3_l651_651600


namespace calculate_f_pi_minus_2_plus_f_pi_l651_651663

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  a * (Real.sin x)^2 + b * (Real.tan x) + 1

theorem calculate_f_pi_minus_2_plus_f_pi (a b : ℝ) (h : f 2 a b = 5) :
  f (Real.pi - 2) a b + f Real.pi a b = -2 := by
  sorry

end calculate_f_pi_minus_2_plus_f_pi_l651_651663


namespace XT_is_correct_l651_651877

-- Define the problem with the given conditions
def rectangle (a b : ℝ) := a * b
def pyramid_volume (base_height top_height : ℝ) := (top_height / base_height)^3
def distance (x y : ℝ) := x + y

noncomputable def XT (base_height top_height : ℝ) : ℝ :=
  by
    let v_ratio : ℝ := pyramid_volume base_height top_height
    let diag_AC : ℝ := 25   -- sqrt(15^2 + 20^2)
    let a_prime_b_prime := (2 / 3) * 15
    let a_prime_c_prime := (2 / 3) * diag_AC
    let r : ℝ :=  sqrt(diag_AC^2 + top_height ^2)
    let height_dist := 11.111 -- exact value needed can be calculated
    let XT_val : ℝ := height_dist + 20
    exact XT_val

-- Problem statement
theorem XT_is_correct (h_base h_top2 : ℝ) (vol_ratio : ℝ) (ab_length bc_length : ℝ) 
  (diag_AC_length : ℝ) (a_prime_b_prime_length a_prime_c_prime_length height : ℝ)
  (expected_val_XT : ℝ) : 
  XT_2 h_base h_top2 = expected_val_XT :=
  sorry

end XT_is_correct_l651_651877


namespace cosine_eq_half_l651_651837

variables {α : Type*} [InnerProductSpace ℝ α]

-- Conditions
variables {A B C E F : α}
variables (h1 : dist A B = 2) (h2 : dist E F = 2) (h3 : dist B C = 8) (h4 : dist A C = real.sqrt 77)
variables (midpoint_B : midpoint ℝ E F = B)
variables (dot_prod_eq : ⟪A - B, A - E⟫ + ⟪A - C, A - F⟫ = 3)

-- Target
theorem cosine_eq_half : real.cos (angle (E - F) (B - C)) = 1 / 2 :=
sorry

end cosine_eq_half_l651_651837


namespace find_coordinates_l651_651190

noncomputable def centroid (a b c : ℝ^3) : ℝ^3 :=
  (a + b + c) / 3

variables (a b c : ℝ^3)
variable (P : ℝ^3)
variable (M : ℝ^3)
variable (N : ℝ^3)

theorem find_coordinates (a b c : ℝ^3) (M := centroid a b c)
  (N := centroid b c a) 
  (P := x a + y b + z c)
  (h : P - M = 2 * (P - N)) :
  x = -2/9 ∧ y = 4/9 ∧ z = 5/9 :=
sorry

end find_coordinates_l651_651190


namespace four_digit_perfect_square_exists_l651_651103

theorem four_digit_perfect_square_exists (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) (h3 : 101 * x + 100 = y^2) : 
  ∃ n, n = 8281 ∧ n = y^2 ∧ (((n / 100) : ℕ) = ((n % 100) : ℕ) + 1) :=
by 
  sorry

end four_digit_perfect_square_exists_l651_651103


namespace tan_sub_sin_eq_sq3_div2_l651_651934

noncomputable def tan_60 := Real.tan (Real.pi / 3)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def result := (tan_60 - sin_60)

theorem tan_sub_sin_eq_sq3_div2 : result = Real.sqrt 3 / 2 := 
by
  -- Proof might go here
  sorry

end tan_sub_sin_eq_sq3_div2_l651_651934


namespace love_cycle_l651_651835

variable (Men : Type) [Fintype Men] [DecidableEq Men]
variable (Girls : Type) [Fintype Girls] [DecidableEq Girls]

-- Assume there are three young men and three girls.
variables (Kolya Petya Yura : Men)
variables (Tanya Zina Galya : Girls)

-- Definition of who loves whom.
variable (loves : Men → Girls)

-- Conditions as per the problem statement
axiom Kolya_condition : loves Kolya = loves (loves (loves Tanya))
axiom Petya_condition : loves Petya = loves (loves (loves Zina))
axiom Zina_not_loves_Yura : ¬ (loves Zina = Yura)

-- Each person loves exactly one other person.
axiom loves_one_to_one : ∀ m1 m2 : Men, loves m1 = loves m2 → m1 = m2
axiom every_girl_loved : ∀ g : Girls, ∃ m : Men, loves m = g

-- Proving the love cycle relationships:
theorem love_cycle :
  (loves Kolya = Galya) ∧ (loves Galya = Petya) ∧ (loves Petya = Tanya) ∧ 
  (loves Tanya = Yura) ∧ (loves Yura = Zina) ∧ (loves Zina = Kolya) := 
sorry

end love_cycle_l651_651835


namespace solve_AlyoshaCube_l651_651505

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l651_651505


namespace total_students_l651_651047

theorem total_students (S : ℕ) (h1 : S / 2 / 2 = 250) : S = 1000 :=
by
  sorry

end total_students_l651_651047


namespace simple_interest_is_500_l651_651868

-- Definitions based on the conditions
def principal (P : ℝ) : Prop := P = 10000
def rate (R : ℝ) : Prop := R = 0.05
def time (T : ℝ) : Prop := T = 1

-- Simple Interest Formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Theorem to prove that the simple interest is 500 given the conditions
theorem simple_interest_is_500 (P R T : ℝ) (hP : principal P) (hR : rate R) (hT : time T) :
  simple_interest P R T = 500 :=
by {
  unfold principal at hP,
  unfold rate at hR,
  unfold time at hT,
  rw [hP, hR, hT],
  -- Proof would go here
  sorry
}

end simple_interest_is_500_l651_651868


namespace nylon_cord_length_l651_651035

theorem nylon_cord_length
  (A B C D : Point)
  (AB AC : ℝ)
  (TotalDistance : ℝ)
  (tree : Point)
  (north_of_tree : A = Point.north_of(tree))
  (east_of_A : B = Point.east_of(A, 10))
  (southeast_of_A : C = Point.southeast_of(A, 20))
  (runs_from_A_to_D : TotalDistance = 60)
  (dog_path: TotalDistance = AB + AC + dist C D)
  : dist C D = 30 := by sorry

end nylon_cord_length_l651_651035


namespace quadrilateral_ratio_l651_651270

theorem quadrilateral_ratio (AB CD AD BC IA IB IC ID : ℝ)
  (h_tangential : AB + CD = AD + BC)
  (h_IA : IA = 5)
  (h_IB : IB = 7)
  (h_IC : IC = 4)
  (h_ID : ID = 9) :
  AB / CD = 35 / 36 :=
by
  -- Proof will be provided here
  sorry

end quadrilateral_ratio_l651_651270


namespace union_A_B_eq_l651_651154

noncomputable def A := {x : ℝ | x ≥ -3}
noncomputable def B := {x : ℝ | 1 < x ∧ x < 3}

theorem union_A_B_eq : (A ∪ B) = Ico -3 (3 : ℝ) ∪ Icc 3 (∞) :=
by sorry

end union_A_B_eq_l651_651154


namespace log_addition_l651_651554

theorem log_addition :
  log 5 45 + log 5 20 = 2 + 2 * log 5 2 + 2 * log 5 3 := 
sorry

end log_addition_l651_651554


namespace alpha_eq_beta_l651_651874

-- Definitions of positive irrational numbers
def is_irrational (α : ℝ) : Prop := α ∉ ℚ 
def positive (α : ℝ) : Prop := α > 0

-- Given that α and β are positive irrational numbers and the given condition holds for all positive x,
-- we need to prove that α = β.
theorem alpha_eq_beta (α β : ℝ) (h_irrational_α : is_irrational α) (h_irrational_β : is_irrational β)
  (h_positive_α : positive α) (h_positive_β : positive β)
  (h_eq_condition : ∀ (x : ℝ), x > 0 → ⌊α * ⌊β * x⌋⌋ = ⌊β * ⌊α * x⌋⌋) : α = β :=
sorry

end alpha_eq_beta_l651_651874


namespace ratio_of_volumes_l651_651878

theorem ratio_of_volumes (a : ℝ) (h : a = real.sqrt 2) :
  let R := ((real.sqrt 3) * a) / real.sqrt 2,
      r := a / (2 * real.sqrt 6),
      V := (4 / 3) * real.pi * R^3,
      V' := (4 / 3) * real.pi * r^3 in
  V / V' = 27 :=
sorry

end ratio_of_volumes_l651_651878


namespace cube_cut_problem_l651_651490

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l651_651490


namespace angle_property_l651_651967

theorem angle_property (θ : ℝ) (hθ1 : θ > π / 2) (hθ2 : θ < π) :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → (x^3 * real.cos θ + x * (1 - x) - (1 - x)^3 * real.sin θ < 0) :=
sorry

end angle_property_l651_651967


namespace matrix_solution_correct_l651_651107

-- Define the matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![-1/2, -6],
  ![-1/3, 7/3]
]

-- Define vectors v1 and v2
def v1 : Vector (Fin 2) ℚ := ![2, -1]
def v2 : Vector (Fin 2) ℚ := ![1, 4]

-- Define vectors w1 and w2
def w1 : Vector (Fin 2) ℚ := ![5, -3]
def w2 : Vector (Fin 2) ℚ := ![2, 9]

-- The theorem stating the conditions
theorem matrix_solution_correct :
  (N.mul_vec v1 = w1) ∧ (N.mul_vec v2 = w2) :=
by {
  sorry
}

end matrix_solution_correct_l651_651107


namespace alyosha_cube_cut_l651_651501

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l651_651501


namespace find_p_l651_651344

theorem find_p (p q r : ℕ) (prime_p : nat.prime p) (prime_q : nat.prime q) (prime_r : nat.prime r)
  (h1 : p + q = r) (h2 : 1 < p) (h3 : p < q) : p = 2 :=
sorry

end find_p_l651_651344


namespace total_tires_l651_651579

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l651_651579


namespace algebraic_expression_value_l651_651585

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = 9) : a^2 - 3 * a * b + b^2 = 19 :=
sorry

end algebraic_expression_value_l651_651585


namespace similar_triangle_shortest_side_l651_651915

theorem similar_triangle_shortest_side (a b c : ℕ) (H1 : a^2 + b^2 = c^2) (H2 : a = 15) (H3 : c = 34) (H4 : b = Int.sqrt 931) : 
  ∃ d : ℝ, d = 3 * Int.sqrt 931 ∧ d = 102  :=
by
  sorry

end similar_triangle_shortest_side_l651_651915


namespace laurent_series_l651_651087

noncomputable def f (z : ℂ) : ℂ := z * sin (π * z / (z - 1))

theorem laurent_series (z : ℂ) (h : z ≠ 1) :
  f(z) = - ∑ n in (Finset.range 100), (-1)^n * π^(2*n+1) / (2*(n:ℂ)+1)! / (z-1)^(2*n) -
         ∑ n in (Finset.range 100), (-1)^n * π^(2*n+1) / (2*(n:ℂ)+1)! / (z-1)^(2*n+1) :=
sorry

end laurent_series_l651_651087


namespace general_equation_of_line_rectangular_equation_of_curve_max_distance_curve_to_line_l651_651244

-- Defining the line l through its parametric equations
def line_parametric (t : ℝ) : ℝ × ℝ := (3 - t, 1 + t)

-- Defining the polar equation of the curve C
def curve_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ - Real.pi / 4)

-- Theorem to state the general equation of line l in rectangular coordinates
theorem general_equation_of_line : ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ t : ℝ, let (x, y) := line_parametric t in a * x + b * y + c = 0 := by
  sorry

-- Theorem to state the rectangular coordinate equation of curve C
theorem rectangular_equation_of_curve :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

-- Theorem to find the maximum distance from the points on curve C to line l
theorem max_distance_curve_to_line :
  ∃ (d : ℝ), (d = 2 * Real.sqrt 2) ∧ (
    ∀ (x y : ℝ),
      ((x - 1)^2 + (y - 1)^2 = 2) →
      ∃ t : ℝ, let (x_l, y_l) := line_parametric t in d = (Real.sqrt ((x - x_l)^2 + (y - y_l)^2))
  ) :=
by
  sorry

end general_equation_of_line_rectangular_equation_of_curve_max_distance_curve_to_line_l651_651244


namespace total_opponents_runs_l651_651452

theorem total_opponents_runs (team_scores : List ℕ) (opponent_scores : List ℕ) :
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  ∃ lost_games won_games opponent_lost_scores opponent_won_scores,
    lost_games = [1, 3, 5, 7, 9, 11] ∧
    won_games = [2, 4, 6, 8, 10, 12] ∧
    (∀ (t : ℕ), t ∈ lost_games → ∃ o : ℕ, o = t + 1 ∧ o ∈ opponent_scores) ∧
    (∀ (t : ℕ), t ∈ won_games → ∃ o : ℕ, o = t / 2 ∧ o ∈ opponent_scores) ∧
    opponent_scores = opponent_lost_scores ++ opponent_won_scores ∧
    opponent_lost_scores = [2, 4, 6, 8, 10, 12] ∧
    opponent_won_scores = [1, 2, 3, 4, 5, 6] →
  opponent_scores.sum = 63 :=
by
  sorry

end total_opponents_runs_l651_651452


namespace net_difference_in_expenditure_l651_651009

variable (P Q : ℝ)
-- Condition 1: Price increased by 25%
def new_price (P : ℝ) : ℝ := P * 1.25

-- Condition 2: Purchased 72% of the originally required amount
def new_quantity (Q : ℝ) : ℝ := Q * 0.72

-- Definition of original expenditure
def original_expenditure (P Q : ℝ) : ℝ := P * Q

-- Definition of new expenditure
def new_expenditure (P Q : ℝ) : ℝ := new_price P * new_quantity Q

-- Statement of the proof problem.
theorem net_difference_in_expenditure
  (P Q : ℝ) : new_expenditure P Q - original_expenditure P Q = -0.1 * original_expenditure P Q := 
by
  sorry

end net_difference_in_expenditure_l651_651009


namespace volume_of_lemon_juice_correct_l651_651435

-- Definitions of given conditions translated into Lean 4:
def height_of_glass : ℝ := 8
def diameter_of_glass : ℝ := 4
def ratio_lemon_juice_to_water : ℝ := 1 / 5
def fraction_full : ℝ := 1 / 3
def pi : ℝ := Real.pi

-- Radius is defined based on diameter:
def radius_of_glass : ℝ := diameter_of_glass / 2

-- Height of the liquid in the glass is one-third of the total height:
def height_of_liquid : ℝ := fraction_full * height_of_glass

-- Calculate the volume of the liquid in the glass:
def volume_of_lemonade : ℝ := pi * (radius_of_glass ^ 2) * height_of_liquid

-- Calculate the volume of lemon juice in the glass:
def volume_of_lemon_juice : ℝ := volume_of_lemonade * (1 / (1 + ratio_lemon_juice_to_water))

-- The expected volume of lemon juice to the nearest hundredth:
def expected_volume_of_lemon_juice : ℝ := (16 / 9) * pi

theorem volume_of_lemon_juice_correct : (volume_of_lemon_juice ≈ 5.59) :=
by
  -- Define the approximation operator to compare floating-point values
  def (x ≈ y) : Prop := abs (x - y) < 0.01
  sorry

end volume_of_lemon_juice_correct_l651_651435


namespace area_quad_reflection_doubled_l651_651134

noncomputable def quadratic_reflection_area (A B C D P : Point) (S : ℝ) : Prop :=
  let M := midpoint A B
  let N := midpoint B C
  let K := midpoint C D
  let L := midpoint D A
  let X := reflect P M
  let Y := reflect P N
  let Z := reflect P K
  let T := reflect P L
  area_quadrilateral X Y Z T = 2 * S

-- Main theorem statement
theorem area_quad_reflection_doubled (A B C D P : Point) (S : ℝ) 
  (h: convex_quadrilateral A B C D ∧ area_quadrilateral A B C D = S) :
  quadratic_reflection_area A B C D P S :=
sorry

end area_quad_reflection_doubled_l651_651134


namespace cos_add_sub_cos_l651_651607

theorem cos_add_sub_cos (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2)
  (h₃ : Real.sin α = sqrt 5 / 5) (h₄ : Real.cos β = 3 * sqrt 10 / 10) :
  Real.cos (α + β) = sqrt 2 / 2 ∧ Real.cos (α - β) = 7 * sqrt 2 / 10 :=
by
  sorry

end cos_add_sub_cos_l651_651607


namespace unique_two_digit_factors_l651_651200

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def factors (n : ℕ) (a b : ℕ) : Prop := a * b = n

theorem unique_two_digit_factors : 
  ∃! (a b : ℕ), is_two_digit a ∧ is_two_digit b ∧ factors 1950 a b :=
by sorry

end unique_two_digit_factors_l651_651200


namespace square_completion_form_l651_651773

theorem square_completion_form (x k m: ℝ) (h: 16*x^2 - 32*x - 512 = 0):
  (x + k)^2 = m ↔ m = 65 :=
by
  sorry

end square_completion_form_l651_651773


namespace remainder_17_pow_63_mod_7_l651_651373

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l651_651373


namespace image_of_center_after_transform_l651_651943

structure Point where
  x : ℤ
  y : ℤ

def reflect_across_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def translate_right (p : Point) (units : ℤ) : Point :=
  { x := p.x + units, y := p.y }

def transform_point (p : Point) : Point :=
  translate_right (reflect_across_x p) 5

theorem image_of_center_after_transform :
  transform_point {x := -3, y := 4} = {x := 2, y := -4} := by
  sorry

end image_of_center_after_transform_l651_651943


namespace compute_ratio_l651_651077

noncomputable def problem_statement := 
  (let x := 1722 in 
   let y := 1715 in 
   let a := 1729 in 
   let b := 1708 in 
   (x - y) * (x + y) / ((a - b) * (a + b)) = 1 / 3)

theorem compute_ratio : problem_statement := 
by
  sorry

end compute_ratio_l651_651077


namespace cost_of_seven_games_equals_175_l651_651841

-- Define the conditions
def total_cost_of_two_games : ℤ := 50
def cost_of_one_game : ℤ := total_cost_of_two_games / 2
def number_of_games : ℤ := 7

-- Prove that the cost of seven video games is $175
theorem cost_of_seven_games_equals_175 : number_of_games * cost_of_one_game = 175 := by
  calc
    number_of_games * cost_of_one_game = 7 * (50 / 2) : by rfl
    ... = 7 * 25 : by rfl
    ... = 175 : by rfl

end cost_of_seven_games_equals_175_l651_651841


namespace ratio_of_heights_is_correct_l651_651045

def original_cone_info := 
(circumference_base : ℝ)
(height_original : ℝ)
(volume_shorter : ℝ)
(height_ratio : ℚ)

def cone_properties (info : original_cone_info) : Prop :=
  info.circumference_base = 20 * Real.pi ∧ 
  info.height_original = 24 ∧ 
  info.volume_shorter = 500 * Real.pi ∧
  ∃ (r : ℝ), -- there exists a radius r for this cone
    2 * Real.pi * r = info.circumference_base  
    ∧ info.height_ratio = Rational.mk 5 8

theorem ratio_of_heights_is_correct (info : original_cone_info) 
  (h : cone_properties info) : 
  info.height_ratio = 5 / 8 :=
sorry

end ratio_of_heights_is_correct_l651_651045


namespace partition_necessary_sufficient_l651_651746

-- Define a convex quadrilateral with a specified angle condition
structure ConvexQuadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)
  (convex : ∀ (i j k : Fin 4), (j, k) ≠ (i, i) → 
            let (v₁, v₂, v₃) := (vertices i, vertices j, vertices k) in
            let (x₁, y₁) := v₂ - v₁, (x₂, y₂) := v₃ - v₁ in 
            x₁ * y₂ - y₁ * x₂ ≠ 0)
  (obtuse_angle : (D_angle) ∧ (∀ (v ∈ {A B C D} \ D), IsAcuteAngle v))

-- Define what it means to partition into obtuse triangles
def partition_is_obtuse_triangles 
  (quad : ConvexQuadrilateral) (n : ℕ) :=
  ∃ (triangles : Fin n → Fin 4 → ℝ × ℝ), 
  ∀ i, (IsTriangle (triangles i)) ∧ (IsObtuseTriangle (triangles i))

-- Main theorem statement
theorem partition_necessary_sufficient {quad : ConvexQuadrilateral} (n : ℕ) :
  partition_is_obtuse_triangles quad n ↔ n ≥ 4 := 
sorry

end partition_necessary_sufficient_l651_651746


namespace A_time_to_complete_job_l651_651424

def rA (rA rB rC : ℚ) := rA
def rB (rA rB rC : ℚ) := rB
def rC (rA rB rC : ℚ) := rC

axiom work_rate :
  ∃ (rA rB rC : ℚ), 
  rA + rB = 1/3 ∧ 
  rB + rC = 1/6 ∧ 
  rA + rC = 5/18 ∧ 
  rA + rB + rC = 2/3

theorem A_time_to_complete_job :
  ∃ (days : ℚ), work_rate ∧ days = 4.5 :=
sorry

end A_time_to_complete_job_l651_651424


namespace forgotten_angles_correct_l651_651076

theorem forgotten_angles_correct (n : ℕ) (h1 : (n - 2) * 180 = 2520) (h2 : 2345 + 175 = 2520) : 
  ∃ a b : ℕ, a + b = 175 :=
by
  sorry

end forgotten_angles_correct_l651_651076


namespace solve_equation_l651_651090

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by
  sorry

end solve_equation_l651_651090


namespace neg_p_sufficient_but_not_necessary_for_q_l651_651144

variable {x : ℝ}

def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

theorem neg_p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, ¬ p x → q x) ∧ ¬ (∀ x : ℝ, q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_q_l651_651144


namespace janina_cover_expenses_l651_651257

noncomputable def rent : ℝ := 30
noncomputable def supplies : ℝ := 12
noncomputable def price_per_pancake : ℝ := 2
noncomputable def total_expenses : ℝ := rent + supplies

theorem janina_cover_expenses : total_expenses / price_per_pancake = 21 := 
by
  calc
    total_expenses / price_per_pancake 
    = (rent + supplies) / price_per_pancake : by rfl
    ... = 42 / 2 : by norm_num
    ... = 21 : by norm_num

end janina_cover_expenses_l651_651257


namespace principal_amount_approx_l651_651854

theorem principal_amount_approx (R : ℝ) (T : ℝ) (diff : ℝ) (P : ℝ) :
  R = 0.10 → T = 4 → diff = 64.10 →
  (diff = (P * (1 + R)^T - P) - (P * R * T)) → P = 60.24 := 
by
  intros hR hT hdiff hdiff_def
  -- Establish intermediate definitions
  have hSI: SI = P * R * T := by sorry
  have hCI: CI = P * (1 + R)^T - P := by sorry
  -- Calculate CI - SI and set to diff
  have hdiff_eq: diff = P * ((1 + R)^T - 1) - (P * R * T) := by sorry
  -- Substitute known values and solve for P
  have hP : P = 60.24 := by sorry
  exact hP


end principal_amount_approx_l651_651854


namespace pow_mod_seventeen_l651_651386

theorem pow_mod_seventeen sixty_three :
  17^63 % 7 = 6 := by
  have h : 17 % 7 = 3 := by norm_num
  have h1 : 17^63 % 7 = 3^63 % 7 := by rw [pow_mod_eq_of_mod_eq h] 
  norm_num at h1
  rw [h1]
  sorry

end pow_mod_seventeen_l651_651386


namespace difference_even_number_sums_l651_651871

open Nat

def sum_of_even_numbers (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem difference_even_number_sums :
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  sum_B - sum_A = 2100 :=
by
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  show sum_B - sum_A = 2100
  sorry

end difference_even_number_sums_l651_651871


namespace subset_A_B_l651_651712

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l651_651712


namespace log_transform_l651_651203

theorem log_transform (x : ℝ) (h : log 7 (x + 6) = 2) : log 13 (x - 1) = log 13 42 := by
  sorry

end log_transform_l651_651203


namespace distinct_terms_count_l651_651081

theorem distinct_terms_count (x y z : ℕ) :
  let expr := 3 * (x + y + z)^2008 - 2 * (x - y - z)^2008 
  in count_distinct_terms expr = 2018045 :=
sorry

end distinct_terms_count_l651_651081


namespace triangle_ADK_area_l651_651248

theorem triangle_ADK_area 
  (A B C D K : Type) 
  (is_midpoint_D : midpoint D A B)
  (is_midpoint_K : midpoint K B C) 
  (area_ABC : area A B C = 50) : 
  area A D K = 12.5 := 
  sorry

end triangle_ADK_area_l651_651248


namespace mixed_gender_groups_l651_651809

theorem mixed_gender_groups (boys girls : ℕ) (h_boys : boys = 28) (h_girls : girls = 4) :
  ∃ groups : ℕ, (groups ≤ girls) ∧ (groups * 2 ≤ boys) ∧ groups = 4 :=
by
   sorry

end mixed_gender_groups_l651_651809


namespace pine_sample_count_l651_651440

variable (total_saplings : ℕ)
variable (pine_saplings : ℕ)
variable (sample_size : ℕ)

theorem pine_sample_count (h1 : total_saplings = 30000) (h2 : pine_saplings = 4000) (h3 : sample_size = 150) :
  pine_saplings * sample_size / total_saplings = 20 := 
sorry

end pine_sample_count_l651_651440


namespace cole_round_trip_time_l651_651945

/-- Prove that the total round trip time is 2 hours given the conditions -/
theorem cole_round_trip_time :
  ∀ (speed_to_work : ℝ) (speed_back_home : ℝ) (time_to_work_min : ℝ),
  speed_to_work = 50 → speed_back_home = 110 → time_to_work_min = 82.5 →
  ((time_to_work_min / 60) * speed_to_work + (time_to_work_min * speed_to_work / speed_back_home) / 60) = 2 :=
by
  intros
  sorry

end cole_round_trip_time_l651_651945


namespace mark_up_price_l651_651442

noncomputable def list_price : ℝ := 100
noncomputable def purchase_price (list_price : ℝ) : ℝ := list_price * 0.7
noncomputable def marked_price (x : ℝ) := x
noncomputable def selling_price (marked_price : ℝ) : ℝ := marked_price * 0.8
noncomputable def profit (selling_price purchase_price : ℝ) : ℝ := selling_price - purchase_price
noncomputable def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

theorem mark_up_price (x : ℝ) :
  let L := list_price in
  let PP := purchase_price L in
  let SP := selling_price x in
  profit SP PP = desired_profit SP → x = 125 := 
by
  sorry

end mark_up_price_l651_651442


namespace percentage_increase_area_rectangle_l651_651873

theorem percentage_increase_area_rectangle (L W : ℕ) 
    (h1: L' = 1.20 * L) 
    (h2: W' = 1.20 * W) : 
    (1.44 - 1) * 100 = 44 :=
by
    sorry

end percentage_increase_area_rectangle_l651_651873


namespace remainder_of_17_pow_63_mod_7_l651_651367

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651367


namespace subset_A_B_l651_651713

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l651_651713


namespace find_f_prime_at_1_l651_651132

def f (x : ℝ) (f_prime_at_1 : ℝ) : ℝ := x^2 + 3 * x * f_prime_at_1

theorem find_f_prime_at_1 (f_prime_at_1 : ℝ) :
  (∀ x, deriv (λ x => f x f_prime_at_1) x = 2 * x + 3 * f_prime_at_1) → 
  deriv (λ x => f x f_prime_at_1) 1 = -1 := 
by
exact sorry

end find_f_prime_at_1_l651_651132


namespace sin_alpha_sol_cos_2alpha_pi4_sol_l651_651127

open Real

-- Define the main problem conditions
def cond1 (α : ℝ) := sin (α + π / 3) + sin α = 9 * sqrt 7 / 14
def range (α : ℝ) := 0 < α ∧ α < π / 3

-- Define the statement for the first problem
theorem sin_alpha_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) : sin α = 2 * sqrt 7 / 7 := 
sorry

-- Define the statement for the second problem
theorem cos_2alpha_pi4_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) (h3 : sin α = 2 * sqrt 7 / 7) : 
  cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := 
sorry

end sin_alpha_sol_cos_2alpha_pi4_sol_l651_651127


namespace value_of_expression_l651_651217

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 2 * x + 5 = 9) : 3 * x^2 + 3 * x - 7 = -1 :=
by
  -- The proof would go here
  sorry

end value_of_expression_l651_651217


namespace find_a_plus_b_l651_651800

theorem find_a_plus_b (a b : ℤ) (k : ℝ)
  (h1 : k = a + Real.sqrt b)
  (h2 : ∀ k : ℝ, Real.log 5 (k + 4) - Real.log 5 k = 0.5) :
  a + b = 6 :=
sorry

end find_a_plus_b_l651_651800


namespace volume_tetrahedron_at_most_one_third_cube_l651_651052

noncomputable def volume_of_tetrahedron (vertices : Fin 4 → (ℝ × ℝ × ℝ)) : ℝ :=
  let ⟨v0, v1, v2, v3⟩ := (vertices 0, vertices 1, vertices 2, vertices 3)
  let matrix := Matrix.of ![
    ![v1.1 - v0.1, v2.1 - v0.1, v3.1 - v0.1],
    ![v1.2 - v0.2, v2.2 - v0.2, v3.2 - v0.2],
    ![v1.3 - v0.3, v2.3 - v0.3, v3.3 - v0.3]
  ] 
  (1 / 6) * abs (matrix.det)

def cube_side_length : ℝ := 1

theorem volume_tetrahedron_at_most_one_third_cube (T : Fin 4 → (ℝ × ℝ × ℝ)) (hT : ∀ i, ∀ j, (T i).1 ∈ Set.Icc 0 cube_side_length ∧ (T i).2 ∈ Set.Icc 0 cube_side_length ∧ (T i).3 ∈ Set.Icc 0 cube_side_length) :
  volume_of_tetrahedron T ≤ (1 / 3) * (cube_side_length ^ 3) :=
by
  sorry

end volume_tetrahedron_at_most_one_third_cube_l651_651052


namespace solution_set_of_inequality_system_l651_651810

theorem solution_set_of_inequality_system :
  { x : ℝ | 1 + x > -1 ∧ 4 - 2x ≥ 0 } = { x : ℝ | -2 < x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_of_inequality_system_l651_651810


namespace number_of_valid_strings_l651_651219

-- Define the characters
inductive Char
| a | b | c | d | e

open Char

def isVowel (ch : Char) : Bool :=
  ch = a ∨ ch = e

def isConsonant (ch : Char) : Bool :=
  ch = b ∨ ch = c ∨ ch = d

-- Define the conditions
def valid_string (s : List Char) : Prop :=
  s.length = 6 ∧ 
  (isConsonant s.head ∧ isConsonant s.ilast) ∧
  (List.countp isVowel s = 2) ∧
  (¬∃ i, i < 5 ∧ (s.nth i = some a ∧ s.nth (i+1) = some e ∨ s.nth i = some e ∧ s.nth (i+1) = some a)) ∧
  (∀ i, i < 5 → s.nth i ≠ s.nth (i+1))

-- State the theorem
theorem number_of_valid_strings : ∃ (s : List Char), valid_string s ∧ List.count valid_string s = 648 :=
sorry

end number_of_valid_strings_l651_651219


namespace minimize_expr_at_6_l651_651122

def expr (c : ℝ) : ℝ :=
  (3 / 4) * c^2 - 9 * c + 13

theorem minimize_expr_at_6 : ∀ c : ℝ, expr c = expr 6 → c = 6 := 
  by 
    sorry

end minimize_expr_at_6_l651_651122


namespace irrational_sqrt_3_l651_651411

theorem irrational_sqrt_3 : 
  (∀ (x : ℝ), (x = 1 / 2 → ¬irrational x) ∧ (x = 0.2 → ¬irrational x) ∧ (x = -5 → ¬irrational x) ∧ (x = sqrt 3 → irrational x)) :=
by
  sorry

end irrational_sqrt_3_l651_651411


namespace num_people_got_on_bus_l651_651069

-- Definitions based on the conditions
def initialNum : ℕ := 4
def currentNum : ℕ := 17
def peopleGotOn (initial : ℕ) (current : ℕ) : ℕ := current - initial

-- Theorem statement
theorem num_people_got_on_bus : peopleGotOn initialNum currentNum = 13 := 
by {
  sorry -- Placeholder for the proof
}

end num_people_got_on_bus_l651_651069


namespace magnitude_of_a_l651_651191

def vector (X Y : Type) := (X, Y)

def dot_product (v1 v2 : vector ℝ ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

def magnitude (v : vector ℝ ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_a :
  ∀ (n : ℝ), (dot_product (1, n) (-1, n) = 0) → magnitude (1, n) = real.sqrt 2 :=
by
  intro n,
  intro h_perpendicular,
  sorry

end magnitude_of_a_l651_651191


namespace correct_calculation_l651_651408

theorem correct_calculation : sqrt 2 / sqrt (1 / 2) = 2 :=
by
  sorry

end correct_calculation_l651_651408


namespace solve_conjugate_l651_651169
open Complex

-- Problem definition:
def Z (a : ℝ) : ℂ := ⟨a, 1⟩  -- Z = a + i

def conj_Z (a : ℝ) : ℂ := ⟨a, -1⟩  -- conjugate of Z

theorem solve_conjugate (a : ℝ) (h : Z a + conj_Z a = 4) : conj_Z 2 = 2 - I := by
  sorry

end solve_conjugate_l651_651169


namespace triangle_AC_length_l651_651774

theorem triangle_AC_length 
  (D E : Point) 
  (A B C : Point)
  (hAD_EC : dist A D = dist E C)
  (hAB : dist A B = 7)
  (hBE : dist B E = 2)
  (hBD_ED : dist B D = dist E D)
  (hBDC_DEB : ∠ B D C = ∠ D E B) 
  : dist A C = 12 := 
sorry

end triangle_AC_length_l651_651774


namespace find_n_l651_651488

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651488


namespace people_seated_around_table_l651_651223

theorem people_seated_around_table (n : ℕ) (h1 : (n - 1)! = 144) (h2 : n ≤ 6) : n = 5 :=
by 
  sorry

end people_seated_around_table_l651_651223


namespace indicated_angle_is_72_degrees_l651_651797

-- Define the problem parameters
def num_sides_of_polygon : ℕ := 6
def num_trapezoids : ℕ := 5

-- The sum of the interior angles of a polygon with n sides
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The figure describes
def angle_sum_of_hexagon : ℝ := sum_interior_angles num_sides_of_polygon

-- Each trapezoid contributes two identical angles
def num_contributing_angles : ℕ := num_trapezoids * 2

-- The measure of one indicated angle θ
def indicated_angle : ℝ := angle_sum_of_hexagon / num_contributing_angles

-- The statement to prove
theorem indicated_angle_is_72_degrees : indicated_angle = 72 :=
by
  sorry

end indicated_angle_is_72_degrees_l651_651797


namespace div_by_7_but_not_8_eq_div_by_8_l651_651060

theorem div_by_7_but_not_8_eq_div_by_8 :
  (∑ n in Finset.range 56000, (if (n + 1) % 7 = 0 ∧ (n + 1) % 8 ≠ 0 then 1 else 0)) =
  (∑ n in Finset.range 56000, (if (n + 1) % 8 = 0 then 1 else 0)) :=
by
  sorry

end div_by_7_but_not_8_eq_div_by_8_l651_651060


namespace compare_powers_l651_651125

theorem compare_powers :
  let a1 := real.exp (real.log 4 / 4)
  let a2 := real.exp (real.log 5 / 5)
  let a3 := real.exp (real.log 16 / 16)
  let a4 := real.exp (real.log 25 / 25)
  a1 > a2 ∧ a2 > a3 ∧ a3 >= a4 := 
by
  -- Proof omitted
  sorry

end compare_powers_l651_651125


namespace cubic_polynomial_sum_l651_651744

-- Define the roots and their properties according to Vieta's formulas
variables {p q r : ℝ}
axiom root_poly : p * q * r = -1
axiom pq_sum : p * q + p * r + q * r = -3
axiom roots_sum : p + q + r = 0

-- Define the target equality to prove
theorem cubic_polynomial_sum :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 3 :=
by
  sorry

end cubic_polynomial_sum_l651_651744


namespace area_inside_C_outside_A_B_l651_651531

/-- Define the radii of circles A, B, and C --/
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

/-- Define the condition of tangency and overlap --/
def circles_tangent_at_one_point (r1 r2 : ℝ) : Prop :=
  r1 = r2 

def circle_C_tangent_to_A_B (rA rB rC : ℝ) : Prop :=
  rA = 1 ∧ rB = 1 ∧ rC = 2 ∧ circles_tangent_at_one_point rA rB

/-- Statement to be proved: The area inside circle C but outside circles A and B is 2π --/
theorem area_inside_C_outside_A_B (h : circle_C_tangent_to_A_B radius_A radius_B radius_C) : 
  π * radius_C^2 - π * (radius_A^2 + radius_B^2) = 2 * π :=
by
  sorry

end area_inside_C_outside_A_B_l651_651531


namespace two_questions_exactly13_l651_651833

open Finset

variables (N A B C W R : ℕ)
variables (exactly_two_correct : ℕ) 

-- Defining the conditions
def conditions := 
  N = 40 ∧
  A = 10 ∧
  B = 13 ∧
  C = 15 ∧
  W = 15 ∧
  R = 1

-- The theorem we need to prove
theorem two_questions_exactly13 (h: conditions N A B C W R) : 
exactly_two_correct = 13 :=
by
  unfold conditions at h
  sorry

end two_questions_exactly13_l651_651833


namespace compare_m_n_l651_651601

theorem compare_m_n (b m n : ℝ) :
  m = -3 * (-2) + b ∧ n = -3 * (3) + b → m > n :=
by
  sorry

end compare_m_n_l651_651601


namespace remainder_17_pow_63_mod_7_l651_651381

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651381


namespace find_n_l651_651487

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l651_651487


namespace dot_product_expression_max_value_of_dot_product_l651_651195

variable (x : ℝ)
variable (k : ℤ)
variable (a : ℝ × ℝ := (Real.cos x, -1 + Real.sin x))
variable (b : ℝ × ℝ := (2 * Real.cos x, Real.sin x))

theorem dot_product_expression :
  (a.1 * b.1 + a.2 * b.2) = 2 - 3 * (Real.sin x)^2 - (Real.sin x) := 
sorry

theorem max_value_of_dot_product :
  ∃ (x : ℝ), 2 - 3 * (Real.sin x)^2 - (Real.sin x) = 9 / 4 ∧ 
  (Real.sin x = -1/2 ∧ 
  (x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 11 * Real.pi / 6 + 2 * k * Real.pi)) := 
sorry

end dot_product_expression_max_value_of_dot_product_l651_651195


namespace coefficient_x2_term_in_expansion_l651_651109

theorem coefficient_x2_term_in_expansion : 
  let general_term (r : ℕ) := (-2)^r * 3^(6-r) * (Nat.choose 6 r) * x^(6-r) 
  in ∑ r in Finset.range 7, general_term r = 2160 := 
sorry

end coefficient_x2_term_in_expansion_l651_651109


namespace inverse_of_A_is_zero_if_det_is_zero_l651_651971

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ := 
  !![5, 10; 
    -3, -6]

theorem inverse_of_A_is_zero_if_det_is_zero : 
  det A = 0 → inverse A = 0 :=
by
  sorry

end inverse_of_A_is_zero_if_det_is_zero_l651_651971


namespace num_valid_x_l651_651953

theorem num_valid_x : ∃ n : ℕ, 
  (∀ x : ℕ, (10 ≤ x ∧ x < 100) → (10 ≤ 2 * x ∧ 2 * x < 100) → (10 ≤ 3 * x ∧ 3 * x < 100) → (100 ≤ 4 * x) → x) 
  ∧ n = 9 :=
by
  sorry

end num_valid_x_l651_651953


namespace remainder_of_17_pow_63_mod_7_l651_651368

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651368


namespace one_tail_in_three_tosses_l651_651406

open Probability

-- Define the fairness of the coin
def fair_coin : ProbSpace :=
{ space := bool,
  prob := λ b, 1/2 }

-- Define the experiment of tossing the coin 3 times
def coin_toss_experiment : ProbSpace :=
  vector_measure_of (λ _, fair_coin) 3

-- Define the specific event occurrence of exactly one tail and rest heads in 3 tosses
def one_tail_and_rest_heads : set (vector bool 3) :=
  {v | (v.to_list.filter id).length = 2}

-- State the theorem to prove the desired probability
theorem one_tail_in_three_tosses (S : ProbSpace) :
  (@measure S _ coin_toss_experiment one_tail_and_rest_heads) = 3/8 :=
sorry

end one_tail_in_three_tosses_l651_651406


namespace cos_half_alpha_beta_l651_651586

theorem cos_half_alpha_beta (α β : ℝ) (hα : π/2 < α) (hα_lt : α < π) 
  (hβ : 0 < β) (hβ_lt : β < π/2) 
  (h1 : Real.sin(α/2 - β) = 2/3) (h2 : Real.sin(α - β/2 + π/2) = -1/9) :
  Real.cos((α + β) / 2) = 7 * Real.sqrt 5 / 27 := 
  sorry

end cos_half_alpha_beta_l651_651586


namespace Marty_combination_count_l651_651290

theorem Marty_combination_count :
  let num_colors := 4
  let num_methods := 3
  num_colors * num_methods = 12 :=
by
  let num_colors := 4
  let num_methods := 3
  sorry

end Marty_combination_count_l651_651290


namespace count_property_P_subsets_l651_651285

def has_property_P (A : Finset ℕ) : Prop :=
  ∃ a b c, a + b = 3 * c ∧ A = {a, b, c}

noncomputable def count_subsets_with_property_P : ℕ :=
  (Finset.powersetLen 3 (Finset.range 101)).filter has_property_P |>.card

theorem count_property_P_subsets : count_subsets_with_property_P = 1600 := by
  sorry

end count_property_P_subsets_l651_651285


namespace sqrt_inequality_l651_651422

theorem sqrt_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
  sqrt (7 * x + 3) + sqrt (7 * y + 3) + sqrt (7 * z + 3) ≤ 7 := by 
  sorry

end sqrt_inequality_l651_651422


namespace maximum_value_at_2001_l651_651979
noncomputable def a_n (n : ℕ) : ℝ := n^2 / (1.001^n)

theorem maximum_value_at_2001 : ∃ n : ℕ, n = 2001 ∧ ∀ k : ℕ, a_n k ≤ a_n 2001 := by
  sorry

end maximum_value_at_2001_l651_651979


namespace total_wheels_l651_651577

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l651_651577


namespace face_value_of_share_l651_651041

theorem face_value_of_share (FV : ℝ) (dividend_percent : ℝ) (interest_percent : ℝ) (market_value : ℝ) :
  dividend_percent = 0.09 → 
  interest_percent = 0.12 →
  market_value = 33 →
  (0.09 * FV = 0.12 * 33) → FV = 44 :=
by
  intros
  sorry

end face_value_of_share_l651_651041


namespace mark_total_votes_l651_651759

theorem mark_total_votes (h1 : 70% = 0.70) (h2 : 100000 : ℕ) (h3 : twice := 2)
  (votes_first_area : ℕ := 0.70 * 100000)
  (votes_remaining_area : ℕ := twice * votes_first_area)
  (total_votes : ℕ := votes_first_area + votes_remaining_area) : 
  total_votes = 210000 := 
by
  sorry

end mark_total_votes_l651_651759


namespace possible_values_of_reciprocal_sum_l651_651283

theorem possible_values_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x^2 + y^2 = 1) :
  ∃ lb, ∃ ub, lb = 8 ∧ (∀ v, 8 ≤ v) ∧ (∀ z, (z ≥ 8) → (z ≤ v)) :=
sorry

end possible_values_of_reciprocal_sum_l651_651283


namespace great_great_grandmother_age_l651_651950

def darcie_age : ℚ := 4
def darcie_and_mother_relation : ℚ := 1 / 6
def mother_and_grandmother_relation : ℚ := 4 / 5
def grandmother_and_great_grandfather_relation : ℚ := 3 / 4
def great_grandfather_and_great_great_grandmother_relation : ℚ := 7 / 10

theorem great_great_grandmother_age :
    let mother_age := darcie_age / darcie_and_mother_relation in
    let grandmother_age := mother_age / mother_and_grandmother_relation in
    let great_grandfather_age := grandmother_age / grandmother_and_great_grandfather_relation in
    let great_great_grandmother_age := great_grandfather_age / great_grandfather_and_great_great_grandmother_relation in
    great_great_grandmother_age = 400 / 7 :=
by
    sorry

end great_great_grandmother_age_l651_651950


namespace exists_distinct_subset_sum_2020_l651_651831

theorem exists_distinct_subset_sum_2020 :
  ∃ S : Finset ℕ, S.card < ∞ ∧ (∀ a b ∈ S, a ≠ b → a ≠ b) ∧ (Finset.card (S.powerset.filter (λ T, T.sum = 2020)) = 2020) := sorry

end exists_distinct_subset_sum_2020_l651_651831


namespace skew_implies_non_intersect_non_intersect_not_implies_skew_l651_651425

variables (l1 l2 : Line)
def skew (l1 l2 : Line) : Prop := -- definition of skew lines, should be provided
def non_intersect (l1 l2 : Line) : Prop := -- definition of non-intersecting, should be provided

theorem skew_implies_non_intersect :
  skew l1 l2 → non_intersect l1 l2 := sorry

theorem non_intersect_not_implies_skew :
  non_intersect l1 l2 → ¬ skew l1 l2 := sorry

end skew_implies_non_intersect_non_intersect_not_implies_skew_l651_651425


namespace continuous_random_variable_formula_l651_651562

variables {b a x r : ℝ}

def pdf (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 / (b - a) then b / (1 + a * x)^2 else 0

theorem continuous_random_variable_formula (b a r_i : ℝ) (h0 : 0 < b)
  (h1 : 0 ≤ a) (h2 : 0 ≤ r_i) (h3 : r_i ≤ 1) :
  ∃ x_i : ℝ, x_i = r_i / (b - a * r_i) :=
by
  sorry

end continuous_random_variable_formula_l651_651562


namespace milk_volume_after_operations_l651_651055

noncomputable def milk_remaining : ℝ :=
  let initial_volume := 100 in
  let remove_replace volume := (volume - 15) * (volume / initial_volume) in
  let final_volume := (remove_replace (remove_replace (remove_replace (remove_replace initial_volume)))) in
  final_volume

theorem milk_volume_after_operations : milk_remaining = 52.200625 := by
  sorry

end milk_volume_after_operations_l651_651055


namespace president_and_committee_count_l651_651230

theorem president_and_committee_count (total_people : ℕ)
    (people : Finset ℕ) (h_size : people.card = total_people) :
    total_people = 10 → 
    (∃ president ∈ people, ∀ committee ⊂ people \ {president}, committee.card = 3 → 
    ↑(Finset.choose_cardinality (people.erase president) 3) = 84 → 
    (total_people * (9.choose 3) = 840)) :=
begin
  sorry,
end

end president_and_committee_count_l651_651230


namespace prism_properties_correct_l651_651673

-- We define what it means for each option to correctly describe a prism.
def option_A : Prop := "Only two faces are parallel"
def option_B : Prop := "All faces are parallel"
def option_C : Prop := "All faces are parallelograms"
def option_D : Prop := "Two pairs of faces are parallel, and each lateral edge is also parallel to each other"

-- Then we state that option D correctly describes the properties of a prism.
theorem prism_properties_correct : option_D := by
  sorry

end prism_properties_correct_l651_651673


namespace age_difference_l651_651010

theorem age_difference {A B C : ℕ} (h : A + B = B + C + 15) : A - C = 15 := 
by 
  sorry

end age_difference_l651_651010


namespace find_f_107_5_l651_651752

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x, f x = f (-x)
axiom func_eq : ∀ x, f (x + 3) = - (1 / f x)
axiom cond_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x

theorem find_f_107_5 : f 107.5 = 1 / 10 := by {
  sorry
}

end find_f_107_5_l651_651752


namespace unique_real_solution_k_values_l651_651581

theorem unique_real_solution_k_values :
  (∀ x k, (3*x + 6)*(x - 4) = -40 + k*x → 
         (∃! x, (3*x^2 - (6 + k)*x + 16 = 0)) → 
         (k = -6 + 8*sqrt 3 ∨ k = -6 - 8*sqrt 3)) :=
sorry

end unique_real_solution_k_values_l651_651581


namespace toothpicks_needed_l651_651906

def count_triangles (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def total_toothpicks_if_unique (n : ℕ) : ℕ :=
  3 * count_triangles n

def shared_toothpicks (n : ℕ) : ℕ :=
  (total_toothpicks_if_unique n) / 2

def boundary_toothpicks (n : ℕ) : ℕ :=
  3 * n

def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + boundary_toothpicks n

theorem toothpicks_needed (n : ℕ) (h : n = 2004) : total_toothpicks n = 3021042 :=
by {
  rw h,
  have h1 : count_triangles 2004 = 2010020 := rfl,
  have h2 : total_toothpicks_if_unique 2004 = 6030060 := rfl,
  have h3 : shared_toothpicks 2004 = 3015030 := rfl,
  have h4 : boundary_toothpicks 2004 = 6012 := rfl,
  show 3015030 + 6012 = 3021042,
  exact rfl,
}

end toothpicks_needed_l651_651906


namespace number_of_terms_arithmetic_sequence_l651_651529

theorem number_of_terms_arithmetic_sequence :
  let a := 2
  let d := 4
  let an := 46
  n = (an - a) / d + 1
  2 + (n - 1) * 4 = 46
  n = 12 :=
sorry

end number_of_terms_arithmetic_sequence_l651_651529


namespace not_in_range_g_zero_l651_651736

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l651_651736


namespace distance_A_beats_B_l651_651860

theorem distance_A_beats_B 
  (A_time : ℝ) (A_distance : ℝ) (B_time : ℝ) (B_distance : ℝ)
  (hA : A_distance = 128) (hA_time : A_time = 28)
  (hB : B_distance = 128) (hB_time : B_time = 32) :
  (A_distance - (B_distance * (A_time / B_time))) = 16 :=
by
  sorry

end distance_A_beats_B_l651_651860


namespace polynomial_coefficient_sum_l651_651983

theorem polynomial_coefficient_sum :
  ∀ (a0 a1 a2 a3 a4 a5 : ℤ), 
  (3 - 2 * x)^5 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 → 
  a0 + a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 = 233 :=
by
  sorry

end polynomial_coefficient_sum_l651_651983


namespace razorback_shop_tshirts_l651_651318

theorem razorback_shop_tshirts (T : ℕ) (h : 215 * T = 4300) : T = 20 :=
by sorry

end razorback_shop_tshirts_l651_651318


namespace gcd_of_polynomial_and_multiple_l651_651204

theorem gcd_of_polynomial_and_multiple (b : ℕ) (hb : 714 ∣ b) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end gcd_of_polynomial_and_multiple_l651_651204


namespace cricket_player_average_l651_651433

theorem cricket_player_average
  (A : ℕ)
  (h1 : 8 * A + 96 = 9 * (A + 8)) :
  A = 24 :=
by
  sorry

end cricket_player_average_l651_651433


namespace remainder_17_pow_63_mod_7_l651_651383

theorem remainder_17_pow_63_mod_7 : (17^63) % 7 = 6 := 
by
  sorry

end remainder_17_pow_63_mod_7_l651_651383


namespace find_k_l651_651626

theorem find_k (k : ℝ) 
  (hf : ∀ x : ℝ, f x = log (2 ^ x + 1) / log 2 + k * x) 
  (h : f(-1) = f(1)) : k = -1 / 2 :=
by sorry

end find_k_l651_651626


namespace remainder_17_pow_63_mod_7_l651_651357

theorem remainder_17_pow_63_mod_7 :
  (17 ^ 63) % 7 = 6 :=
by {
  -- Given that 17 ≡ 3 (mod 7)
  have h1 : 17 % 7 = 3 := by norm_num,
  
  -- We need to show that (3 ^ 63) % 7 = 6.
  have h2 : (17 ^ 63) % 7 = (3 ^ 63) % 7 := by {
    rw ← h1,
    exact pow_mod_eq_mod_pow _ _ _
  },
  
  -- Now it suffices to show that (3 ^ 63) % 7 = 6
  have h3 : (3 ^ 63) % 7 = 6 := by {
    rw pow_eq_pow_mod 6, -- 63 = 6 * 10 + 3, so 3^63 = (3^6)^10 * 3^3
    have : 3 ^ 6 % 7 = 1 := by norm_num,
    rw [this, one_pow, one_mul, pow_mod_eq_pow_mod],
    exact_pow [exact_mod [norm_num]],
    exact rfl,
  },
  
  -- Combine both results
  exact h2 ▸ h3
}

end remainder_17_pow_63_mod_7_l651_651357


namespace no_integer_roots_of_odd_p0_p1_l651_651289

-- Define the polynomial p(x)
def polynomial (n : ℕ) (a : ℕ → ℤ) (x : ℤ) : ℤ :=
  ∑ i in Finset.range (n + 1), a i * x ^ (n - i)

-- Assume that a_n and the sum of coefficients a_0 + a_1 + ... + a_n are both odd
variables {n : ℕ} {a : ℕ → ℤ}
axiom a_n_odd : is_odd (a n)
axiom sum_coeffs_odd : is_odd (∑ i in Finset.range (n + 1), a i)

-- Define the statement that p(x) has no integer roots
theorem no_integer_roots_of_odd_p0_p1 (x : ℤ) : polynomial n a x ≠ 0 :=
by sorry

end no_integer_roots_of_odd_p0_p1_l651_651289


namespace car_actual_miles_l651_651028

-- Define the condition that the faulty odometer skips the digit '6'.
def skips_six (n : ℕ) : Prop := ¬ (6 ∈ n.digits 10)

-- Define the corrected mileage function assuming 'skips_six'.
noncomputable def correct_mileage (r : ℕ) : ℕ :=
  let rec aux n acc :=
    if n = 0 then acc
    else if skips_six acc then aux (n - 1) (acc + 1)
         else aux n (acc + 1)
  in aux r 0

-- Define the odometer reading
def odometer_reading : ℕ := 9008

-- State the theorem
theorem car_actual_miles : correct_mileage odometer_reading = 6287 :=
begin
  sorry
end

end car_actual_miles_l651_651028


namespace ones_digit_of_power_l651_651956

theorem ones_digit_of_power (n : ℕ) : 
  (13 ^ (13 * (12 ^ 12)) % 10) = 9 :=
by
  sorry

end ones_digit_of_power_l651_651956


namespace element_subset_a_l651_651709

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l651_651709


namespace find_roots_l651_651537

theorem find_roots {n : ℕ} {a_2 a_3 ... a_n : ℂ} 
  (hn : 0 < n)
  (p : Polynomial ℂ := Polynomial.monomial n 1 + Polynomial.monomial (n-1) n + Polynomial.monomial (n-2) a_2 + ... + Polynomial.C a_n)
  (roots : Fin n.succ → ℂ)
  (hroots : Multiset.map (λ r, r.1) (Multiset.ofMap (Polynomial.rootSet p ℂ)) = Multiset.ofFinsupp (Finsupp.onFinset (Finₙ.elems n) (λ i, Coeffs roots i)))
  (sum_mag_16 : Σ (i : Finₙ n), (complex.abs (roots.val i)) ^ 16 = n) :
  ∀ i : Finₙ n, roots i = -1 :=
by
  sorry

end find_roots_l651_651537


namespace converse_binomial_divisibility_l651_651777

open Nat

theorem converse_binomial_divisibility (n : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → n ∣ binomial n k) → Nat.Prime n :=
by
  sorry

end converse_binomial_divisibility_l651_651777


namespace sequence_a_10_l651_651637

theorem sequence_a_10 : 
  (∀ n : ℕ, a (n + 1) = a n + 2^n ∧ a 1 = 1) → a 10 = 1023 := 
by
  sorry

end sequence_a_10_l651_651637


namespace inequality_solution_l651_651216

variable {α : Type*} [LinearOrderedField α]
variable (a b x : α)

theorem inequality_solution (h1 : a < 0) (h2 : b = -a) :
  0 < x ∧ x < 1 ↔ ax^2 + bx > 0 :=
by sorry

end inequality_solution_l651_651216


namespace find_locus_of_Y_l651_651594

noncomputable def locus_of_Y (AB : line) (B : point) (k : real) : set point :=
  let semicircle := sorry -- Define the semicircle with diameter AB
  let homothety := sorry -- Define the rotational homothety centered at B
  homothety image semicircle

theorem find_locus_of_Y (AB : line) (A B : point) (k : real)
  (semicircle : set point)
  (Hsemicircle : semicircle = sorry -- Semicircle with diameter AB)
  (X Y : point)
  (HXY : ∀ X ∈ semicircle, ∃ Y, Y ∈ ray XA ∧ XY = k * XB) :
  ∀ X ∈ semicircle, Y ∈ locus_of_Y AB B k :=
begin
  apply sorry,
end

end find_locus_of_Y_l651_651594


namespace perfect_squares_perfect_square_plus_one_l651_651969

theorem perfect_squares : (∃ n : ℕ, 2^n + 3 = (x : ℕ)^2) ↔ n = 0 ∨ n = 3 :=
by
  sorry

theorem perfect_square_plus_one : (∃ n : ℕ, 2^n + 1 = (x : ℕ)^2) ↔ n = 3 :=
by
  sorry

end perfect_squares_perfect_square_plus_one_l651_651969


namespace smallest_n_divisor_not_factorial_square_l651_651393

theorem smallest_n_divisor_not_factorial_square :
  ∃ (n : ℕ), 100 ≤ n ∧ n <= 999 ∧ 
  let S_n := n * (n + 1) / 2 in
  let factorial_squared := (Nat.factorial n) ^ 2 in
  ¬ (S_n ∣ factorial_squared) ∧
  ∀ m, 100 ≤ m ∧ m < n →
  let S_m := m * (m + 1) / 2 in
  let factorial_squared_m := (Nat.factorial m) ^ 2 in
  S_m ∣ factorial_squared_m
:=
begin
  -- We state that there exists an integer n
  -- with all required properties.
  use 100,
  -- Prove required properties for n = 100.
  sorry
end

end smallest_n_divisor_not_factorial_square_l651_651393


namespace part1_part2_l651_651694

noncomputable section

variable (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (S_n : ℕ → ℝ) (T_n : ℕ → ℝ)

-- Conditions
axiom a1 : ∀ n : ℕ, S_n n = 2 * (a_n n) - 2 * n
axiom a2 : ∀ n : ℕ, b_n n = Real.log (a_n n + 2) / Real.log 2
axiom a3 : ∀ n : ℕ, T_n n = ∑ i in Finset.range n, 1 / (b_n i * b_n (i + 1))

-- Part (1): Prove the general formula for a_n
theorem part1 : ∀ n : ℕ, a_n n = 2 ^ (n + 1) - 2 :=
sorry

-- Part (2): Determine the range of possible values for a
theorem part2 (a : ℝ) : (∀ n : ℕ, T_n n < a) ↔ a ≥ 1 / 2 :=
sorry

end part1_part2_l651_651694


namespace largest_fraction_l651_651545

theorem largest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 5)
                                          (h2 : f2 = 3 / 6)
                                          (h3 : f3 = 5 / 10)
                                          (h4 : f4 = 7 / 15)
                                          (h5 : f5 = 8 / 20) : 
  (f2 = 1 / 2 ∨ f3 = 1 / 2) ∧ (f2 ≥ f1 ∧ f2 ≥ f4 ∧ f2 ≥ f5) ∧ (f3 ≥ f1 ∧ f3 ≥ f4 ∧ f3 ≥ f5) := 
by
  sorry

end largest_fraction_l651_651545


namespace smallest_even_five_digit_tens_place_l651_651324

theorem smallest_even_five_digit_tens_place :
  ∃ (n : ℕ), n = 13586 ∧ (n / 10 % 10 = 8) :=
by 
  existsi (13586 : ℕ)
  split
  · rfl
  · rfl
  sorry

end smallest_even_five_digit_tens_place_l651_651324


namespace max_volume_of_cube_l651_651071

theorem max_volume_of_cube (S : ℝ) (hS : S > 0) :
  (∃ l : ℝ, ∀ (W H : ℝ), (W H = l) → ( W * H ≤ (l ^ 2) / 16 ) ) →
  (∃ l : ℝ, ∀ (W H D : ℝ), (W H + WH + HD + W D = S) → (W * H * D ≤ (S / 6) ^ (3 / 2))) :=
sorry

end max_volume_of_cube_l651_651071


namespace triangle_construction_possible_l651_651085

theorem triangle_construction_possible {m_a m_b s_c : ℝ} (h₁ : m_a < 2 * s_c) (h₂ : m_b < 2 * s_c) :
  ∃ (A B C : Type), Triangle A B C := sorry

end triangle_construction_possible_l651_651085


namespace exist_polynomials_unique_polynomials_l651_651741

-- Problem statement: the function 'f'
variable (f : ℝ → ℝ → ℝ → ℝ)

-- Condition: f(w, w, w) = 0 for all w ∈ ℝ
axiom f_ww_ww_ww (w : ℝ) : f w w w = 0

-- Statement for existence of A, B, C
theorem exist_polynomials (f : ℝ → ℝ → ℝ → ℝ)
  (hf : ∀ w : ℝ, f w w w = 0) : 
  ∃ A B C : ℝ → ℝ → ℝ → ℝ, 
  (∀ w : ℝ, A w w w + B w w w + C w w w = 0) ∧ 
  ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x) :=
sorry

-- Statement for uniqueness of A, B, C
theorem unique_polynomials (f : ℝ → ℝ → ℝ → ℝ) 
  (A B C A' B' C' : ℝ → ℝ → ℝ → ℝ)
  (hf: ∀ w : ℝ, f w w w = 0)
  (h1 : ∀ w : ℝ, A w w w + B w w w + C w w w = 0)
  (h2 : ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))
  (h3 : ∀ w : ℝ, A' w w w + B' w w w + C' w w w = 0)
  (h4 : ∀ x y z : ℝ, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) : 
  A = A' ∧ B = B' ∧ C = C' :=
sorry

end exist_polynomials_unique_polynomials_l651_651741


namespace eight_step_paths_board_l651_651899

theorem eight_step_paths_board (P Q : ℕ) (hP : P = 0) (hQ : Q = 7) : 
  ∃ (paths : ℕ), paths = 70 :=
by
  sorry

end eight_step_paths_board_l651_651899


namespace remainder_of_17_pow_63_mod_7_l651_651370

theorem remainder_of_17_pow_63_mod_7 :
  17^63 % 7 = 6 :=
by {
  -- Condition: 17 ≡ 3 (mod 7)
  have h : 17 % 7 = 3 := by norm_num,
  -- Use the periodicity established in the powers of 3 modulo 7 to prove the statement
  -- Note: Leaving the proof part out as instructed
  sorry
}

end remainder_of_17_pow_63_mod_7_l651_651370
