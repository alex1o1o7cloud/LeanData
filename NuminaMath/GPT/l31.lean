import Mathlib

namespace limit_proof_l31_31771

noncomputable def f (x : ℝ) : ℝ := 2 * log (3 * x) + 8 * x

theorem limit_proof : (∀ x : ℝ, x > 0 → f(x) = 2 * log (3 * x) + 8 * x) → 
  (∀ g : ℝ → ℝ, g = λ x, (f(1-2*x) - f(1)) / x → 
  (∃ l : ℝ, is_limit g 0 l ∧ l = -20)) :=
sorry

end limit_proof_l31_31771


namespace opposite_neg_fraction_l31_31952

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31952


namespace courtyard_length_l31_31608

theorem courtyard_length (width_of_courtyard : ℝ) (brick_length_cm brick_width_cm : ℝ) (total_bricks : ℕ) (H1 : width_of_courtyard = 16) (H2 : brick_length_cm = 20) (H3 : brick_width_cm = 10) (H4 : total_bricks = 20000) :
  ∃ length_of_courtyard : ℝ, length_of_courtyard = 25 := 
by
  -- variables and hypotheses
  let brick_length_m := brick_length_cm / 100
  let brick_width_m := brick_width_cm / 100
  let area_one_brick := brick_length_m * brick_width_m
  let total_area := total_bricks * area_one_brick
  have width_of_courtyard_val : width_of_courtyard = 16 := H1
  have brick_length_cm_val : brick_length_cm = 20 := H2
  have brick_width_cm_val : brick_width_cm = 10 := H3
  have total_bricks_val : total_bricks = 20000 := H4
  let length_of_courtyard := total_area / width_of_courtyard
  have length_courtyard_val : length_of_courtyard = 25 := sorry
  use length_of_courtyard,
  exact length_courtyard_val sorry

end courtyard_length_l31_31608


namespace inequality_induction_l31_31171

theorem inequality_induction (n : ℕ) (h : n > 0) : 
  (  ∏ i in (finset.range n), ((2 * i + 1) / (2 * (i + 1))) ) ≤ 1 / (real.sqrt (3 * n)) :=
by
  sorry

end inequality_induction_l31_31171


namespace number_of_chords_number_of_chords_l31_31054

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31054


namespace isosceles_triangle_l31_31835

-- Define the properties and conditions given in the problem
def midpoint (O B C : Point) : Prop := dist O B = dist O C
def lineSegment (A O : Point) : Set Point := { T | (|dist A T| = 2 * |dist T O|) }

-- Define the trajectory equation for point T
def trajectory (T : Point) : Prop :=
  (T.x^2 / 9 + T.y^2 / (9 / 2) = 1) ∧ (T.y ≠ 0)

-- Main theorem to prove that triangle MPR is isosceles
theorem isosceles_triangle
  (A B C O M N P Q R : Point)
  (BC_eq : dist B C = 3 * sqrt 2)
  (perimeter_ABC : dist A B + dist A C + dist B C = 6 + 3 * sqrt 2)
  (O_midpoint : midpoint O B C)
  (T_on_AO : T ∈ lineSegment A O)
  (M_on_OC : ∃ m, M = (m, 0))
  (N_on_OC : ∃ n, N = (1 / m, 0))
  (OM_ON_eq1 : dist O M * dist O N = 1)
  (QM_not_parallel : true) -- Noting that line QM is not parallel to coordinate axes.
  (intersect_PQ : ∃ (x₁ y₁ x₂ y₂ : ℝ), lineThrough M intersects E at P (x₁, y₁) and Q (x₂, y₂))
  (intersect_QR : ∃ (x₃ y₃ : ℝ), lineThrough Q intersects E at R (x₃, y₃))
  : isosceles (triangle M P R) :=
sorry

end isosceles_triangle_l31_31835


namespace find_f_50_l31_31120

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x * y
axiom f_20 : f 20 = 10

theorem find_f_50 : f 50 = 25 :=
by
  sorry

end find_f_50_l31_31120


namespace point_P_in_third_quadrant_l31_31744

-- Points A, B, C and definitions
def point_A := (2, 3)
def point_B := (5, 4)
def point_C := (7, 10)

-- Vector definitions
def vec_AB := (point_B.1 - point_A.1, point_B.2 - point_A.2)
def vec_AC := (point_C.1 - point_A.1, point_C.2 - point_A.2)

-- Define the vector equation
def vec_AP := λ (λ : ℝ) => (vec_AB.1 + λ * vec_AC.1, vec_AB.2 + λ * vec_AC.2)

-- Define the condition for point P to be in the third quadrant
def in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The theorem to prove
theorem point_P_in_third_quadrant (λ : ℝ) : in_third_quadrant (5 + 5*λ, 4 + 7*λ) ↔ λ < -1 :=
sorry

end point_P_in_third_quadrant_l31_31744


namespace tims_initial_cans_l31_31543
noncomputable theory

-- Definitions extracted from conditions
def initial_cans (x : ℕ) : ℕ := x
def after_jeff (x : ℕ) : ℕ := x - 6
def after_buying_more (x : ℕ) : ℕ := after_jeff x + (after_jeff x / 2)

-- Statement of the problem
theorem tims_initial_cans (x : ℕ) (h : after_buying_more x = 24) : x = 22 :=
by
  sorry

end tims_initial_cans_l31_31543


namespace range_of_k_l31_31714

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| - |x - 2| > k) → k < -3 := 
by 
  intros h 
  have h1 : (∀ x : ℝ, |x + 1| - |x - 2| ≤ 3) := 
    by 
      intro x 
      apply abs_sub_le_iff.mpr; norm_num
  have h2 : (∀ x : ℝ, -3 ≤ |x + 1| - |x - 2|) := 
    by 
      intro x 
      apply neg_le_abs_sub_iff_le.mpr; norm_num
  sorry

end range_of_k_l31_31714


namespace three_digit_numbers_contain_7_8_9_l31_31379

theorem three_digit_numbers_contain_7_8_9 : 
  let total_three_digit_numbers := 900
  let without_7_8_9 := 6 * 7 * 7
  in total_three_digit_numbers - without_7_8_9 = 606 :=
by
  sorry

end three_digit_numbers_contain_7_8_9_l31_31379


namespace max_elements_in_set_l31_31876

theorem max_elements_in_set (S : Finset ℕ) (hS : ∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → 
  ∃ (k : ℕ) (c d : ℕ), c < d ∧ c ∈ S ∧ d ∈ S ∧ a + b = c^k * d) :
  S.card ≤ 48 :=
sorry

end max_elements_in_set_l31_31876


namespace Sonja_oil_used_l31_31505

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l31_31505


namespace second_number_less_than_first_by_16_percent_l31_31593

variable (X : ℝ)

theorem second_number_less_than_first_by_16_percent
  (h1 : X > 0)
  (first_num : ℝ := 0.75 * X)
  (second_num : ℝ := 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 16 := by
  sorry

end second_number_less_than_first_by_16_percent_l31_31593


namespace ratio_eq_two_l31_31799

theorem ratio_eq_two (a b c d : ℤ) (h1 : b * c + a * d = 1) (h2 : a * c + 2 * b * d = 1) : 
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end ratio_eq_two_l31_31799


namespace max_sections_with_5_lines_l31_31832

theorem max_sections_with_5_lines (n : ℕ) (h₀ : n = 0 → sections n = 1)
  (h₁ : n = 1 → sections n = 2) (h₂ : n = 2 → sections n = 4)
  (h₃ : n = 3 → sections n = 7) (h₄ : n = 4 → sections n = 11) :
  sections 5 = 16 :=
sorry

end max_sections_with_5_lines_l31_31832


namespace perpendicular_planes_l31_31863

variables (b c : Line) (α β : Plane)
axiom line_in_plane (b : Line) (α : Plane) : Prop -- b ⊆ α
axiom line_parallel_plane (c : Line) (α : Plane) : Prop -- c ∥ α
axiom lines_are_skew (b c : Line) : Prop -- b and c could be skew
axiom planes_are_perpendicular (α β : Plane) : Prop -- α ⊥ β
axiom line_perpendicular_plane (c : Line) (β : Plane) : Prop -- c ⊥ β

theorem perpendicular_planes (hcα : line_in_plane c α) (hcβ : line_perpendicular_plane c β) : planes_are_perpendicular α β := 
sorry

end perpendicular_planes_l31_31863


namespace smallest_square_l31_31963

theorem smallest_square 
  (a b : ℕ) 
  (h1 : 15 * a + 16 * b = m ^ 2) 
  (h2 : 16 * a - 15 * b = n ^ 2)
  (hm : m > 0) 
  (hn : n > 0) : 
  min (15 * a + 16 * b) (16 * a - 15 * b) = 481 ^ 2 := 
sorry

end smallest_square_l31_31963


namespace bricks_painted_white_l31_31254

/-- The total number of bricks painted white in a stack of 180 bricks with two sides against the wall. -/
theorem bricks_painted_white (b : ℕ) (h : b = 180) : ∃ n : ℕ, n = 96 :=
by
  use 96
  trivial

end bricks_painted_white_l31_31254


namespace shortest_side_of_similar_triangle_l31_31639

theorem shortest_side_of_similar_triangle 
  (a b : ℝ) (hyp_a b' hyp_b' : ℝ)
  (h_a : a = 30) 
  (h_b : hyp_a = 34) 
  (h_sim : hyp_b' = 102)
  (h_right_1 : a^2 + b^2 = hyp_a^2) 
  (h_right_2 : ∃ k, b' = k * b) 
  : b' = 48 :=
by 
  have b := b * 1,                   -- Initial definition
  rw h_right_2 b at *,              -- Use condition of similarity
  sorry                              -- Skip actual proof

end shortest_side_of_similar_triangle_l31_31639


namespace hclo4_required_l31_31375

theorem hclo4_required (m_naoh m_koh : ℕ) (h_naoh : m_naoh = 1) (h_koh : m_koh = 0.5) : 
  m_naoh + m_koh = 1.5 :=
by {
  rw [h_naoh, h_koh],
  norm_num,
}

end hclo4_required_l31_31375


namespace remainder_of_sum_l31_31715

theorem remainder_of_sum (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = 7145) (h2 : n2 = 7146)
  (h3 : n3 = 7147) (h4 : n4 = 7148) (h5 : n5 = 7149) :
  ((n1 + n2 + n3 + n4 + n5) % 8) = 7 :=
by sorry

end remainder_of_sum_l31_31715


namespace not_collinear_c1_c2_l31_31595

noncomputable def a : ℝ × ℝ × ℝ := (1, 0, 1)
noncomputable def b : ℝ × ℝ × ℝ := (-2, 3, 5)
noncomputable def c1 : ℝ × ℝ × ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2, a.3 + 2 * b.3)
noncomputable def c2 : ℝ × ℝ × ℝ := (3 * a.1 - b.1, 3 * a.2 - b.2, 3 * a.3 - b.3)

def are_collinear (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (γ : ℝ), u = (γ * v.1, γ * v.2, γ * v.3)

theorem not_collinear_c1_c2 : ¬are_collinear c1 c2 :=
by {
  -- Placeholder for the proof
  sorry -- The actual proof is omitted
}

end not_collinear_c1_c2_l31_31595


namespace nancy_child_support_l31_31470

theorem nancy_child_support (income : ℕ → ℕ) (total_owed paid total_support : ℕ) : 
  (∀ i : ℕ, i < 3 → income i = 30000) →
  (∀ i : ℕ, 3 ≤ i → i < 7 → income i = 36000) →
  paid = total_support - total_owed →
  total_support = (0.3 * 30000 * 3 + 0.3 * 36000 * 4 : ℕ) →
  total_owed = 69000 →
  paid = 1200 :=
by
  sorry

end nancy_child_support_l31_31470


namespace find_number_l31_31934

theorem find_number (n : ℕ) (h1 : n % 5 = 0) (h2 : 70 ≤ n ∧ n ≤ 90) (h3 : Nat.Prime n) : n = 85 := 
sorry

end find_number_l31_31934


namespace equation_1_solution_equation_2_solution_l31_31103

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end equation_1_solution_equation_2_solution_l31_31103


namespace number_of_numbers_is_three_l31_31088

theorem number_of_numbers_is_three :
  ∃ (n : ℕ), n ≠ 0 ∧ (∀ (a : fin n → ℝ) (i : fin n), a i = (∑ j in finset.univ \ {i}, a j) / 2) → n = 3 :=
begin
  sorry
end

end number_of_numbers_is_three_l31_31088


namespace problem_statement_l31_31306

theorem problem_statement (x y : ℝ) (h : x + 4 * y - 3 = 0) : 2^x * 16^y = 8 :=
by
  sorry

end problem_statement_l31_31306


namespace num_chords_l31_31044

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31044


namespace train_stops_time_l31_31199

/-- Given the speeds of a train excluding and including stoppages, 
calculate the stopping time in minutes per hour. --/
theorem train_stops_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 48)
  (h2 : speed_including_stoppages = 40) :
  ∃ minutes_stopped : ℝ, minutes_stopped = 10 :=
by
  sorry

end train_stops_time_l31_31199


namespace problem1_problem2_l31_31004

namespace TriangleProofs

-- Problem 1: Prove that A + B = π / 2
theorem problem1 (a b c : ℝ) (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.cos B))
  (h2 : n = (b, Real.cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  (h_neq : m ≠ n)
  : A + B = Real.pi / 2 :=
sorry

-- Problem 2: Determine the range of x
theorem problem2 (A B : ℝ) (x : ℝ) 
  (h : A + B = Real.pi / 2) 
  (hx : x * Real.sin A * Real.sin B = Real.sin A + Real.sin B) 
  : 2 * Real.sqrt 2 ≤ x :=
sorry

end TriangleProofs

end problem1_problem2_l31_31004


namespace problem_l31_31448

def g_n (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

theorem problem (f : ℝ → ℝ) (h : ∀ n x, f x = g_n n x) :
  (∃ xs : Finset ℝ, ∀ x ∈ xs, x ∈ Icc 0 (2 * Real.pi) ∧
    8 * g_n 5 x - 5 * g_n 3 x = 3 * g_n 1 x ∧ xs.card = 5) :=
begin
  sorry
end

end problem_l31_31448


namespace at_least_sixty_dancers_l31_31969

theorem at_least_sixty_dancers
  (performances : ℕ)
  (participants_per_performance : ℕ)
  (pairs_limit : ∀ p1 p2 : ℕ, p1 ≠ p2 → ∃! performance : ℕ, performance ≤ performances ∧
     p1 ∈ {x | x < performances} ∧ p2 ∈ {x | x < performances} ∧ ∃! y : ℕ, y ∈ {1, 2, ..., participants_per_performance})
  (total_performances : performances = 40)
  (total_participants : participants_per_performance = 10) :
  ∃ (dancers : ℕ), dancers ≥ 60 :=
begin
  sorry
end

end at_least_sixty_dancers_l31_31969


namespace value_of_item_l31_31170

theorem value_of_item (a b m p : ℕ) (h : a ≠ b) (eq_capitals : a * x + m = b * x + p) : 
  x = (p - m) / (a - b) :=
by
  sorry

end value_of_item_l31_31170


namespace ax_eq_bc_l31_31005

variable {A B C M B1 C1 X : Point}
variable {b c : Line}
variable [IsTriangle A B C]
variable [IsAngleEq A 45]
variable [IsMedian A M]
variable [IsSymmetric b AM B B1]
variable [IsSymmetric c AM C C1]
variable [IntersectionPoint b c X]

theorem ax_eq_bc :
  AX = BC :=
sorry

end ax_eq_bc_l31_31005


namespace cube_divided_by_planes_l31_31638

theorem cube_divided_by_planes : 
  ∀ (cube : Type), 
  (∃ (planes : Type), 
    (faces_planes : cube → planes) →
    (number_of_parts : ℕ),
    number_of_parts = 27 
  ) := 
sorry

end cube_divided_by_planes_l31_31638


namespace number_of_chords_number_of_chords_l31_31055

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31055


namespace reciprocal_neg_2023_l31_31129

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l31_31129


namespace max_valid_pairs_l31_31107

-- Define the conditions: distinct ordered pairs of nonnegative integers
def distinct_pairs (n : ℕ) : Prop := 
  ∃ (a b : Fin n → ℕ), (∀ i j : Fin n, a i ≠ a j ∨ b i ≠ b j ∨ i = j)

-- Define the predicate for pairs satisfying the given condition
def valid_pair (a b : Fin 100 → ℕ) (i j : Fin 100) : Prop :=
  1 ≤ i.val + 1 ∧ i.val + 1 < j.val + 1 ∧ j.val + 1 ≤ 100 ∧ 
  |a i * b j - a j * b i| = 1

-- Define the main theorem statement
theorem max_valid_pairs :
  ∀ (a b : Fin 100 → ℕ), distinct_pairs 100 → 
  (∃ N, (∀ i j, valid_pair a b i j → N) ∧ N ≤ 197) :=
by
  sorry

end max_valid_pairs_l31_31107


namespace terminating_decimals_count_l31_31720

theorem terminating_decimals_count :
  (∃ count : ℕ, count = 166 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → (∃ m : ℕ, n = 3 * m)) :=
sorry

end terminating_decimals_count_l31_31720


namespace solve_equation_l31_31910

theorem solve_equation (x : ℝ) (h : x ≥ 2) :
  (sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 10 - 8 * sqrt (x - 2)) = 3) ↔ x = 44.25 :=
sorry

end solve_equation_l31_31910


namespace number_is_seven_l31_31476

theorem number_is_seven (x : ℝ) (h : x^2 + 120 = (x - 20)^2) : x = 7 := 
by
  sorry

end number_is_seven_l31_31476


namespace total_legs_of_animals_l31_31033

def num_kangaroos := 23
def num_goats := 3 * num_kangaroos
def legs_per_kangaroo := 2
def legs_per_goat := 4

def total_legs := (num_kangaroos * legs_per_kangaroo) + (num_goats * legs_per_goat)

theorem total_legs_of_animals : total_legs = 322 := by
  sorry

end total_legs_of_animals_l31_31033


namespace find_initial_E_l31_31698

-- Define the variables
variables (E J Jo : ℕ)

-- Define the problem's conditions
def total_volume (E J Jo : ℕ) : Prop := E + J + Jo = 1200
def equal_after_transfer (E J Jo : ℕ) : Prop := 
  let J' := J + 200 in
  let E' := E + 100 in
  let Jo' := Jo - 300 in
  E' = J' ∧ J' = Jo' ∧ E' = Jo'

-- The main theorem
theorem find_initial_E (E J Jo : ℕ) (h1 : total_volume E J Jo) (h2 : equal_after_transfer E J Jo) : E = 300 :=
by
  sorry

end find_initial_E_l31_31698


namespace lcm_ge_n_times_a1_l31_31869

theorem lcm_ge_n_times_a1 (n : ℕ) (a : Fin n → ℕ) (h : ∀ i j : Fin n, i < j → a i < a j) (hn : ∀ i : Fin n, 0 < a i) :
  nat.lcm (Finset.univ.image a) ≥ n * a 0 := sorry

end lcm_ge_n_times_a1_l31_31869


namespace evaluate_trig_expression_l31_31701

theorem evaluate_trig_expression :
  (Real.tan (π / 18) - Real.sqrt 3) * Real.sin (2 * π / 9) = -1 :=
by
  sorry

end evaluate_trig_expression_l31_31701


namespace circle_equation_unique_external_tangency_l31_31733

variable (r x y a b: ℝ)

-- Conditions for part (1)
def radius : ℝ := 5
def line_eq (a b : ℝ) := a - b + 10 = 0
def passes_through (a b : ℝ) := (x - a)^2 + (y - b)^2 = radius^2

theorem circle_equation (a b : ℝ) (h1 : a - b + 10 = 0) (h2 : (x = -5) → (y = 0) → (x + 5 + a)^2 + (b - y)^2 = radius^2)
  : ((x = -10) ∧ (y = 0) ∨ (x = -5) ∧ (y = 5)) :=
sorry

-- Conditions for part (2)
def dist_center_to_line : ℝ := 5 * Real.sqrt 2
def external_tangency_cond (r : ℝ) := r = dist_center_to_line - radius

theorem unique_external_tangency (r : ℝ) (h : r = dist_center_to_line - radius)
  : (∀ (x y : ℝ), (x^2 + y^2 = r^2) → ∃! (a b : ℝ), passes_through a b = radius^2 ∧ (x - a)^2 + (y - b)^2 = (r + radius)^2) :=
sorry

end circle_equation_unique_external_tangency_l31_31733


namespace segment_length_and_area_l31_31001

variable {R : Type} [LinearOrderedField R]

structure Trapezoid (R : Type) [LinearOrderedField R] :=
(AB CD : R)
(parallel : AB = 15 ∧ CD = 24)

structure Midpoints (R : Type) [LinearOrderedField R] :=
(M N P : R)
(midpoints : M = (15 + 14) / 2 ∧ N = (24 + 14) / 2 ∧ P = 15 / 2)

theorem segment_length_and_area {R : Type} [LinearOrderedField R] 
  (trapezoid : Trapezoid R)
  (midpoints : Midpoints R) :
  (MN : R) = 4.5 ∧ (area_MNP : R) = 15.75 := 
  sorry

end segment_length_and_area_l31_31001


namespace systematic_sampling_eighth_group_l31_31630

theorem systematic_sampling_eighth_group (total_students : ℕ) (groups : ℕ) (group_size : ℕ)
(start_number : ℕ) (group_number : ℕ)
(h1 : total_students = 480)
(h2 : groups = 30)
(h3 : group_size = 16)
(h4 : start_number = 5)
(h5 : group_number = 8) :
  (group_number - 1) * group_size + start_number = 117 := by
  sorry

end systematic_sampling_eighth_group_l31_31630


namespace smallest_int_divisible_by_pow2_l31_31093

theorem smallest_int_divisible_by_pow2 (n : ℕ) (h : 0 < n) :
  ∃ k : ℕ, k > (sqrt 3 + 1)^(2 * n) ∧ k % (2^(n + 1)) = 0 :=
by
  sorry

end smallest_int_divisible_by_pow2_l31_31093


namespace generate_any_2022_tuple_l31_31879

-- Definitions of operations on tuples
def tuple_add (v w : Fin 2022 → ℤ) : Fin 2022 → ℤ := 
  fun i => v i + w i

def tuple_max (v w : Fin 2022 → ℤ) : Fin 2022 → ℤ := 
  fun i => max (v i) (w i)

-- Hypotheses
def can_generate_all_tuples (s : List (Fin 2022 → ℤ)) : Prop := 
  ∀ t : Fin 2022 → ℤ, ∃ steps : List (Fin 2022 → ℤ), 
    List.Sublist steps s ∧ 
    ∃ f : steps.length → Fin 2022 → ℤ, 
      ∃ g : steps.length → steps.length → Fin 2022 → ℤ, 
      (∀ i, (g i) = (tuple_add f i) ∨ (g i) = (tuple_max f i)) ∧ 
      (g (steps.length - 1) = t)

-- Statement: Prove that Lucy can generate any 2022-tuple with 3 initial tuples
theorem generate_any_2022_tuple :
  ∃ initial_tuples : List (Fin 2022 → ℤ), 
  List.length initial_tuples = 3 ∧ 
  can_generate_all_tuples initial_tuples :=
sorry

end generate_any_2022_tuple_l31_31879


namespace coconut_flavored_jelly_beans_l31_31137

-- Define the conditions
def total_jelly_beans : ℕ := 4000
def fraction_red : ℚ := 3 / 4
def fraction_coconut_flavored : ℚ := 1 / 4

-- Calculate the number of red jelly beans
def red_jelly_beans : ℕ := (fraction_red * total_jelly_beans).toNat

-- The final proof statement
theorem coconut_flavored_jelly_beans :
  (fraction_coconut_flavored * red_jelly_beans).toNat = 750 := by
  -- Proof goes here
  sorry

end coconut_flavored_jelly_beans_l31_31137


namespace range_of_a_l31_31523

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*x + a

theorem range_of_a :
  ∀ a : ℝ, 
  (∃ x1 : ℝ, x1 ∈ Ioo (-2 : ℝ) 0 ∧ f a x1 = 0) ∧
  (∃ x2 : ℝ, x2 ∈ Ioo (2 : ℝ) 3 ∧ f a x2 = 0) → 
  -3 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l31_31523


namespace solve_inequality_l31_31501

open Nat

def A (n k : ℕ) : ℕ := fact n / fact (n - k)

theorem solve_inequality (x : ℕ) (hx1 : 2 < x) (hx2 : x ≤ 9) :
  A 9 x > 6 * A 9 (x - 2) ↔ x ∈ {3, 4, 5, 6, 7} :=
by
  sorry

end solve_inequality_l31_31501


namespace maximize_area_triangle_at_sqrt5_l31_31316

noncomputable def circle (a : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 1}

def line : set (ℝ × ℝ) := 
  {p | p.2 = 3 * p.1}

def points_of_intersection (a : ℝ) : set (ℝ × ℝ) := 
  circle a ∩ line

def area_triangle (a : ℝ) : ℝ :=
  let center := (a, a)
  let d := a / real.sqrt (5)
  let chord_length := real.sqrt (2 - (a ^ 2 / 5))
  1 / 2 * chord_length * d

theorem maximize_area_triangle_at_sqrt5 : 
  ∀ a > 0, area_triangle a ≤ area_triangle (real.sqrt 5) := 
sorry

end maximize_area_triangle_at_sqrt5_l31_31316


namespace max_elements_in_T_l31_31868

-- Definition of the problem
def is_valid_set (T : Set ℕ) : Prop :=
  (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a + b) % 5 ≠ 0) ∧
  T ⊆ {n | n ∈ Finset.range 60 ∧ n ≥ 1}

-- The proposition we must prove
theorem max_elements_in_T : 
  ∃ (T : Set ℕ), is_valid_set T ∧ (Finset.card (T.to_finset) = 24) :=
sorry

end max_elements_in_T_l31_31868


namespace number_of_chords_number_of_chords_l31_31057

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31057


namespace exponential_function_has_inverse_l31_31576

noncomputable def has_inverse {α β : Type*} (f : α → β) :=
  ∃ g : β → α, function.left_inverse g f ∧ function.right_inverse g f

theorem exponential_function_has_inverse :
  has_inverse (λ x : ℝ, 2^x) :=
sorry

end exponential_function_has_inverse_l31_31576


namespace integral_eval_l31_31284

open Real

theorem integral_eval : ∫ x in -π/2..π/2, (2*x - sin x) = 0 :=
by
  sorry

end integral_eval_l31_31284


namespace courtyard_is_25_meters_long_l31_31613

noncomputable def courtyard_length (width : ℕ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) : ℝ :=
  let brick_area := brick_length * brick_width
  let total_area := num_bricks * brick_area
  total_area / width

theorem courtyard_is_25_meters_long (h_width : 16 = 16)
  (h_brick_length : 0.20 = 0.20)
  (h_brick_width: 0.10 = 0.10)
  (h_num_bricks: 20_000 = 20_000)
  (h_total_area: 20_000 * (0.20 * 0.10) = 400) :
  courtyard_length 16 0.20 0.10 20_000 = 25 := by
        sorry

end courtyard_is_25_meters_long_l31_31613


namespace multiple_statements_l31_31916

theorem multiple_statements (c d : ℤ)
  (hc4 : ∃ k : ℤ, c = 4 * k)
  (hd8 : ∃ k : ℤ, d = 8 * k) :
  (∃ k : ℤ, d = 4 * k) ∧
  (∃ k : ℤ, c + d = 4 * k) ∧
  (∃ k : ℤ, c + d = 2 * k) :=
by
  sorry

end multiple_statements_l31_31916


namespace proof_of_triangle_properties_l31_31782

noncomputable def triangle_c (A : ℝ) (a b : ℝ) : ℝ :=
  let A_rad := Real.pi / 3
  let cos_A := Real.cos A_rad
  let equation := a ^ 2 - b ^ 2 + 2 * b * Real.cos A_rad
  Classical.choose (exists_root_eq_quadratic equation -12 0)

noncomputable def triangle_ratio (a b c : ℝ) (A : ℝ) : ℝ :=
  let sin_A := Real.sin (Real.pi / 3)
  let sin_B := 1 * (a / c) * (Real.sin (Real.arcsin (a / (2 * (a/c))))
  let sin_C := c * (a / b) * (Real.sin (Real.arcsin (a / (2 * (a/b))))
  (a + b + c) / (sin_A + sin_B + sin_C)

theorem proof_of_triangle_properties :
  let a := sqrt 13
  let b := 1
  let A := 60
  let c := triangle_c A a b
  c = 4 ∧ triangle_ratio a b c A = 2 * sqrt 39 / 3 := 
  by sorry

end proof_of_triangle_properties_l31_31782


namespace ordering_l31_31303

noncomputable def a : ℝ := 1 / (Real.exp 0.6)
noncomputable def b : ℝ := 0.4
noncomputable def c : ℝ := Real.log 1.4 / 1.4

theorem ordering : a > b ∧ b > c :=
by
  have ha : a = 1 / (Real.exp 0.6) := rfl
  have hb : b = 0.4 := rfl
  have hc : c = Real.log 1.4 / 1.4 := rfl
  sorry

end ordering_l31_31303


namespace nine_points_chords_l31_31065

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31065


namespace difference_between_eights_l31_31115

theorem difference_between_eights (value_tenths : ℝ) (value_hundredths : ℝ) (h1 : value_tenths = 0.8) (h2 : value_hundredths = 0.08) : 
  value_tenths - value_hundredths = 0.72 :=
by 
  sorry

end difference_between_eights_l31_31115


namespace find_principal_l31_31594

noncomputable def principal (P : ℝ) (r : ℝ) : Prop :=
  (P * (1 + r) ^ 2 = 8000) ∧ (P * (1 + r) ^ 3 = 9261)

theorem find_principal : ∃ P : ℝ, ∃ r : ℝ, principal P r ∧ P ≈ 5967.91 :=
by
  sorry

end find_principal_l31_31594


namespace triangle_properties_l31_31242

-- Definitions of sides of the triangle
def a : ℕ := 15
def b : ℕ := 11
def c : ℕ := 18

-- Definition of the triangle inequality theorem in the context
def triangle_inequality (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Perimeter calculation
def perimeter (x y z : ℕ) : ℕ :=
  x + y + z

-- Stating the proof problem
theorem triangle_properties : triangle_inequality a b c ∧ perimeter a b c = 44 :=
by
  -- Start the process for the actual proof that will be filled out
  sorry

end triangle_properties_l31_31242


namespace relation_among_a_b_c_l31_31302

def a : ℝ := 6 ^ 0.7
def b : ℝ := Real.log 0.6 / Real.log 7  -- using the change of base formula
def c : ℝ := Real.log 0.7 / Real.log 0.6

theorem relation_among_a_b_c : a > c ∧ c > b := by
  sorry

end relation_among_a_b_c_l31_31302


namespace aziz_parents_in_america_l31_31656

/-- 
Given that Aziz's parents moved to America in 1982, the current year is 2021, and Aziz just celebrated his 36th birthday,
prove that Aziz's parents had been living in America for 3 years before he was born.
-/
theorem aziz_parents_in_america (year_parents_moved : ℕ) (current_year : ℕ) (aziz_age : ℕ)
  (h_move : year_parents_moved = 1982)
  (h_current : current_year = 2021)
  (h_age : aziz_age = 36) :
  current_year - aziz_age - year_parents_moved = 3 :=
by
  -- conditions
  rw [h_move, h_current, h_age]
  -- calculation 
  sorry

end aziz_parents_in_america_l31_31656


namespace riverside_theme_parks_adjustment_plans_l31_31817

/-
In my city, we are building the happiest city with a plan to construct 7 riverside theme parks along the Wei River.
To enhance the city's quality and upgrade the park functions, it is proposed to reduce the number of riverside theme parks by 2.
The theme parks at both ends of the river are not to be adjusted, and two adjacent riverside theme parks cannot be adjusted simultaneously.
The number of possible adjustment plans is 6.
-/

theorem riverside_theme_parks_adjustment_plans :
  let total_parks := 7
  let end_parks := 2
  let remaining_parks := total_parks - end_parks
  let adjustments_needed := 2
  let adjacent_pairs := 4
  nat.choose remaining_parks adjustments_needed - adjacent_pairs = 6 := 
by {
  sorry
}

end riverside_theme_parks_adjustment_plans_l31_31817


namespace num_chords_l31_31040

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31040


namespace smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l31_31995

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l31_31995


namespace nine_points_circle_chords_l31_31072

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31072


namespace hyperbola_eccentricity_l31_31353

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l31_31353


namespace serves_probability_l31_31421

variable (p : ℝ) (hpos : 0 < p) (hneq0 : p ≠ 0)

def ExpectedServes (p : ℝ) : ℝ :=
  p + 2 * p * (1 - p) + 3 * (1 - p) ^ 2

theorem serves_probability (h : ExpectedServes p > 1.75) : 0 < p ∧ p < 1 / 2 :=
  sorry

end serves_probability_l31_31421


namespace matrix_calculation_l31_31859

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l31_31859


namespace common_divisors_count_l31_31790

theorem common_divisors_count : 
  let a := 45;
  let b := 75;
  let gcd := Nat.gcd a b;
  let divisors := Nat.divisors gcd;
  divisors.card = 4 := by
  {
    let a := 45;
    let b := 75;
    let gcd := Nat.gcd a b;
    let divisors := Nat.divisors gcd;
    have h1 : a = 3^2 * 5 := by norm_num,
    have h2 : b = 3 * 5^2 := by norm_num,
    have h_gcd : gcd = 3 * 5 := by
    {
      rw [Nat.gcd_def, h1, h2],
      norm_num,
    },
    rw h_gcd at divisors,
    have h_divisors : divisors = [1, 3, 5, 15] := by
    {
      rw [Nat.divisors_def, h_gcd],
      norm_num,
    },
    simp only [h_divisors, List.card, List.length],
  }

end common_divisors_count_l31_31790


namespace angle_AGP_right_angle_l31_31527

variables {A B C D E F P G : Point}
variables (h_incircle_touch : incircle_touch_triangle ABC D E F)
variables (h_parallel : parallel (line_through A P) (line_through B C))
variables (h_intersect : intersects AD incircle_at G)

theorem angle_AGP_right_angle :
  angle A G P = 90 :=
sorry

end angle_AGP_right_angle_l31_31527


namespace alice_monthly_salary_l31_31245

-- Definitions for given conditions
def commission_rate : ℝ := 0.02
def sales : ℝ := 2500
def savings : ℝ := 29
def savings_rate : ℝ := 0.10

-- Intermediate calculations from the problem
def commission : ℝ := commission_rate * sales
def total_earnings : ℝ := 10 * savings

-- The problem to be proven
theorem alice_monthly_salary : ∃ S : ℝ, total_earnings = S + commission ∧ S = 240 :=
by
  have commission_calc : commission = 50 := by
    unfold commission
    norm_num
  have total_earnings_calc : total_earnings = 290 := by
    unfold total_earnings
    norm_num
  use (total_earnings - commission)
  split
  case left => 
    norm_num at total_earnings_calc commission_calc
    rw [total_earnings_calc, commission_calc]
    ring
  case right => 
    norm_num

end alice_monthly_salary_l31_31245


namespace cabin_charges_per_night_l31_31693

theorem cabin_charges_per_night 
  (total_lodging_cost : ℕ)
  (hostel_cost_per_night : ℕ)
  (hostel_days : ℕ)
  (total_cabin_days : ℕ)
  (friends_sharing_expenses : ℕ)
  (jimmy_lodging_expense : ℕ) 
  (total_cost_paid_by_jimmy : ℕ) :
  total_lodging_cost = total_cost_paid_by_jimmy →
  hostel_cost_per_night = 15 →
  hostel_days = 3 →
  total_cabin_days = 2 →
  friends_sharing_expenses = 3 →
  jimmy_lodging_expense = 75 →
  ∃ cabin_cost_per_night, cabin_cost_per_night = 45 :=
by
  sorry

end cabin_charges_per_night_l31_31693


namespace socks_expected_value_l31_31144

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l31_31144


namespace find_y_l31_31557

theorem find_y : ∃ y : ℝ, (7 / 3) * y = 42 ∧ y = 18 :=
by
  use 18
  split
  · norm_num
  · norm_num

end find_y_l31_31557


namespace f_f_3_eq_651_over_260_l31_31755

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (2 + x⁻¹))

/-- Prove that f(f(3)) = 651/260 -/
theorem f_f_3_eq_651_over_260 : f (f (3)) = 651 / 260 := 
sorry

end f_f_3_eq_651_over_260_l31_31755


namespace range_of_m_l31_31445

theorem range_of_m (m : ℝ) : 
  (∃ (θ : ℝ), 0 < θ ∧ θ < π ∧ sin θ = cos θ + m) ↔ (-√2 / 2 < m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l31_31445


namespace tim_initial_soda_l31_31546

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end tim_initial_soda_l31_31546


namespace num_two_digit_values_l31_31023

def sumOfDigits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem num_two_digit_values (n : ℕ) :
  (10 ≤ n ∧ n < 100) →
  (sumOfDigits (sumOfDigits n) = 3) →
  (finset.univ.filter (λ x, 10 ≤ x ∧ x < 100 ∧ sumOfDigits (sumOfDigits x) = 3)).card = 10 :=
sorry

end num_two_digit_values_l31_31023


namespace ellipse_problem_l31_31853

variables {a b c x y : ℝ}

def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def foci (a b c : ℝ) : Prop :=
  c = sqrt (a^2 - b^2)

def arithmetic_sequence (a b c : ℝ) : Prop :=
  a^2 + c^2 = 2 * b^2

def distances (P F1 A F2 B : ℝ) : Prop :=
  P / A + F1 / B = 4

theorem ellipse_problem 
  (h_ellipse : ellipse a b x y) 
  (h_foci : foci a b c) 
  (h_seq : arithmetic_sequence a b c) 
  (h_dist : distances (|P F1|) (|A F1|) (|P F2|) (|B F2|)) :
  (|P F1| / |A F1| + |P F2| / |B F2| = 4)  :=
sorry

end ellipse_problem_l31_31853


namespace exists_cube_with_prime_points_l31_31489

def is_prime (n : ℕ) : Prop := sorry -- Assume we have some prime-checking predicate

def prime_points_in_cube (x y z : ℕ) (s : ℕ) : ℕ :=
  (finset.range s).fold 0 (λ acc i,
    (finset.range s).fold acc (λ acc j,
      (finset.range s).fold acc (λ acc k,
        if is_prime (x + i) ∧ is_prime (y + j) ∧ is_prime (z + k) then
          acc + 1
        else
          acc)))

theorem exists_cube_with_prime_points :
  ∃ (x y z : ℕ), prime_points_in_cube x y z 2014 = 2014 :=
sorry

end exists_cube_with_prime_points_l31_31489


namespace calculate_sample_std_dev_l31_31640

-- Define the sample data
def sample_weights : List ℝ := [125, 124, 121, 123, 127]

-- Define the mean calculation
def sample_mean (weights : List ℝ) : ℝ :=
  (weights.foldl (+) 0) / weights.length

-- Define the variance calculation
def sample_variance (weights : List ℝ) (mean : ℝ) : ℝ :=
  (weights.foldl (λ acc x => acc + (x - mean) ^ 2) 0) / weights.length

-- Define the standard deviation calculation
def sample_std_dev (variance : ℝ) : ℝ :=
  Real.sqrt variance

-- The main theorem we want to prove
theorem calculate_sample_std_dev :
  let weights := sample_weights in
  let mean := sample_mean weights in
  let variance := sample_variance weights mean in
  sample_std_dev variance = 2 :=
by
  sorry

end calculate_sample_std_dev_l31_31640


namespace table_coverage_percentage_l31_31159

def A := 204  -- Total area of the runners
def T := 175  -- Area of the table
def A2 := 24  -- Area covered by exactly two layers of runner
def A3 := 20  -- Area covered by exactly three layers of runner

theorem table_coverage_percentage : 
  (A - 2 * A2 - 3 * A3 + A2 + A3) / T * 100 = 80 := 
by
  sorry

end table_coverage_percentage_l31_31159


namespace exists_team_cycle_l31_31912

def tournament (teams : Type) := ∀ (A B : teams), (A ≠ B) → (A beats B ∨ B beats A)

theorem exists_team_cycle (teams : Type) [fintype teams] [inhabited teams]
  {score : teams → ℕ}
  (total_teams : finset teams)
  (round_robin_tournament : tournament teams)
  (equal_scores : ∃ A B : teams, A ≠ B ∧ score A = score B)
  :
  ∃ A B C : teams, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (A beats B ∧ B beats C ∧ C beats A) :=
sorry

end exists_team_cycle_l31_31912


namespace maximum_y_coordinate_on_graph_of_r_eq_sin_3theta_l31_31710

theorem maximum_y_coordinate_on_graph_of_r_eq_sin_3theta :
  ∃ θ : ℝ, ∀ θ', r θ = sin (3 * θ) ∧ y θ = r θ * sin θ ∧ y θ' = r θ' * sin θ' → 
    y θ ≤ y θ' → y θ = 1 :=
by { sorry }

end maximum_y_coordinate_on_graph_of_r_eq_sin_3theta_l31_31710


namespace john_money_left_l31_31432

-- Define the initial amount of money John had
def initial_amount : ℝ := 200

-- Define the fraction of money John gave to his mother
def fraction_to_mother : ℝ := 3 / 8

-- Define the fraction of money John gave to his father
def fraction_to_father : ℝ := 3 / 10

-- Prove that the amount of money John had left is $65
theorem john_money_left : 
    let money_given_to_mother := fraction_to_mother * initial_amount,
        money_given_to_father := fraction_to_father * initial_amount,
        total_given_away := money_given_to_mother + money_given_to_father,
        money_left := initial_amount - total_given_away
    in money_left = 65 := sorry

end john_money_left_l31_31432


namespace alix_has_15_more_chocolates_than_nick_l31_31881

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l31_31881


namespace variable_relation_l31_31636

variable (a : ℝ) (y : ℝ)

theorem variable_relation (h : y = 3 * a) : (a ≠ 0 → y ≠ 0) :=
by
  intros h₀
  rw h
  intro h₁
  exact h₀ (mul_eq_zero.mp h₁).2

end variable_relation_l31_31636


namespace system_solution_l31_31747

theorem system_solution (x y : ℝ) (h1 : 4 * x - y = 3) (h2 : x + 6 * y = 17) : x + y = 4 :=
by
  sorry

end system_solution_l31_31747


namespace years_ago_same_average_l31_31518

section committee_age

variables {O N T X : ℝ} (h1 : N = O - 40) (h2 : T + O = T + 9 * X + N)

/-- The number of years ago when the average age of 10 committee members was the same. -/
theorem years_ago_same_average : X = 40 / 9 := by
  -- Using provided conditions to prove X = 40 / 9
  have eq1 : T + O = T + 9 * X + (O - 40) := by rw [h1]
  have eq2 : T + O = T + 9 * X + O - 40 := by rw [h1]
  have eq3 : 9 * X = 40 := by linarith
  have eq4 : X = 40 / 9 := by linarith
  exact eq4
  sorry

end committee_age

end years_ago_same_average_l31_31518


namespace length_cut_XY_l31_31642

theorem length_cut_XY (a x : ℝ) (h1 : 4 * a = 100) (h2 : a + a + 2 * x = 56) : x = 3 :=
by { sorry }

end length_cut_XY_l31_31642


namespace average_speed_l31_31104

-- Define the conditions
def distance1 := 350 -- miles
def time1 := 6 -- hours
def distance2 := 420 -- miles
def time2 := 7 -- hours

-- Define the total distance and total time (excluding break)
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- Define the statement to prove
theorem average_speed : 
  (total_distance / total_time : ℚ) = 770 / 13 := by
  sorry

end average_speed_l31_31104


namespace chocolate_difference_l31_31886

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l31_31886


namespace probability_positive_difference_ge_three_l31_31165

open Finset Nat

theorem probability_positive_difference_ge_three :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (s.card.choose 2)
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  probability = 15 / 28 :=
by
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := s.card.choose 2
  have total_pairs_eq : total_pairs = 28 := by decide
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  have favorable_pairs_eq : favorable_pairs = 15 := by decide
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  have probability_eq : probability = 15 / 28 := by
    rw [favorable_pairs_eq, total_pairs_eq, ←nat_cast_add, ←rat.div_eq_div_iff]
    norm_num
  exact probability_eq

end probability_positive_difference_ge_three_l31_31165


namespace sum_of_sequence_l31_31524

theorem sum_of_sequence (n : ℕ) (h : n = 30):
  let a_n := λ n, n^2 * (cos (n * π / 3))^2 - (sin (n * π / 3))^2,
      S_n := ∑ k in range n, a_n k in
  S_n = 470 := 
by
  sorry

end sum_of_sequence_l31_31524


namespace hexagonal_tessellation_color_count_l31_31268

noncomputable def hexagonal_tessellation : SimpleGraph ℕ :=
{ adj := λ n m, (m = n + 1 ∨ m = n - 1 ∨ 
                 m = n + 2 ∨ m = n - 2 ∨
                 m = n + 3 ∨ m = n - 3),
  sym := by {
    intros a b h,
    cases h;
    simp [h],
  },
  loopless := by {
    intro a,
    simp,
  }
}

theorem hexagonal_tessellation_color_count :
  ∃ k, k = 4 ∧ ∀ f : ℕ → Fin k, (∀ {x y}, hexagonal_tessellation.adj x y → f x ≠ f y) := 
begin
  use 4,
  split,
  { refl },
  { intros f h,
    sorry,
  }
end

end hexagonal_tessellation_color_count_l31_31268


namespace equal_roots_of_quadratic_eq_l31_31978

theorem equal_roots_of_quadratic_eq (k : ℝ) : (3 * k^2 - 144 = 0) → (k = 12 ∨ k = -12) :=
by
sintro h
sorry

end equal_roots_of_quadratic_eq_l31_31978


namespace fraction_division_l31_31562

theorem fraction_division :
  (1/4) / 2 = 1/8 :=
by
  sorry

end fraction_division_l31_31562


namespace domain_of_function_l31_31682

theorem domain_of_function :
  (∀ x : ℝ, 1 - x ≥ 0 → 2 * x ^ 2 - 3 * x - 2 ≠ 0 → x ≤ 1 ∧ x ≠ -1 / 2 ∧ x ≠ 2) →
  ∀ x : ℝ, x ∈ (-∞ : Set ℝ, -1 / 2] ∪ Set.Ioo (-1 / 2) 1 :=
by
  intro h x
  sorry

end domain_of_function_l31_31682


namespace glove_probability_correct_l31_31809

noncomputable def glove_probability : ℚ :=
  let red_pair := ("r1", "r2") -- pair of red gloves
  let black_pair := ("b1", "b2") -- pair of black gloves
  let white_pair := ("w1", "w2") -- pair of white gloves
  let all_pairs := [
    (red_pair.1, red_pair.2), 
    (black_pair.1, black_pair.2), 
    (white_pair.1, white_pair.2),
    (red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
    (red_pair.2, black_pair.1), (red_pair.2, white_pair.1),
    (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)
  ]
  let valid_pairs := [(red_pair.1, black_pair.2), (red_pair.1, white_pair.2), 
                      (red_pair.2, black_pair.1), (red_pair.2, white_pair.1), 
                      (black_pair.1, white_pair.2), (black_pair.2, white_pair.1)]
  (valid_pairs.length : ℚ) / (all_pairs.length : ℚ)

theorem glove_probability_correct :
  glove_probability = 2 / 5 := 
by
  sorry

end glove_probability_correct_l31_31809


namespace football_club_balance_l31_31621

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l31_31621


namespace find_annual_interest_rate_first_account_l31_31648

noncomputable def annualInterestRateOfFirstAccount : ℝ := 0.065

theorem find_annual_interest_rate_first_account :
  ∀ (totalInvestment totalInterest firstAccountInvestment secondAccountInterestRate : ℝ),
  totalInvestment = 9000 ∧
  secondAccountInterestRate = 0.08 ∧
  totalInterest = 678.87 ∧
  firstAccountInvestment = totalInvestment - 6258.0 →
  totalInterest = (firstAccountInvestment * annualInterestRateOfFirstAccount) + (6258.0 * secondAccountInterestRate) →
  annualInterestRateOfFirstAccount ≈ 0.065 :=
by
  intros totalInvestment totalInterest firstAccountInvestment secondAccountInterestRate h1 h2
  sorry

end find_annual_interest_rate_first_account_l31_31648


namespace log_condition_l31_31598

noncomputable def condition_necessary_and_not_sufficient (m a : ℝ) : Prop :=
  (m - 1) * (a - 1) > 0 → log a m > 0 ∧ log a m > 0 → (m - 1) * (a - 1) > 0

theorem log_condition (m a : ℝ) :
  (m - 1) * (a - 1) > 0 ↔ log a m > 0 :=
sorry

end log_condition_l31_31598


namespace trigonometric_expression_equals_one_l31_31581

theorem trigonometric_expression_equals_one :
  (sin 22 * cos 8 + cos 158 * cos 98) / (sin 23 * cos 7 + cos 157 * cos 97) = 1 := sorry

end trigonometric_expression_equals_one_l31_31581


namespace slope_of_parametric_line_l31_31134

theorem slope_of_parametric_line :
  (∃ t : ℝ, (2 + 3 * t, 1 - t) = (x, y)) → (y = -1 / 3 * x + 5 / 3) → 
  ((y - (1 - t)) / (x - (2 + 3 * t)) = -1 / 3) :=
by
  intro h1 h2
  unfold slope
  sorry

end slope_of_parametric_line_l31_31134


namespace conjugate_of_z_l31_31381

theorem conjugate_of_z : 
  let i := complex.i in
  let z := 1 / (1 + i) in 
  complex.conj z = ((1 : ℂ) / 2) + ((1 : ℂ) / 2) * i := 
by
  sorry

end conjugate_of_z_l31_31381


namespace negation_proof_l31_31531

theorem negation_proof :
  (¬ ∀ x : ℝ, exp x - 2 * sin x + 4 ≤ 0) ↔ (∃ x : ℝ, exp x - 2 * sin x + 4 > 0) := 
by
  sorry

end negation_proof_l31_31531


namespace football_club_balance_l31_31622

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l31_31622


namespace OI_eq_IH_l31_31850

theorem OI_eq_IH {A B C D E F O I H : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  [Inhabited E] [Inhabited F] [Inhabited O] [Inhabited I] [Inhabited H]
  (hD : A → B → D) (hE : A → B → E) (hF : C → A → B → F) 
  (hO : A → B → C → O) (hI : A → B → C → I) (hH : A → B → C → H) 
  (h_cond : ∀ (CF AD BE : ℝ), CF / AD + CF / BE = 2) :
  dist O I = dist I H := 
sorry

end OI_eq_IH_l31_31850


namespace shortest_distance_to_circle_from_origin_l31_31570

theorem shortest_distance_to_circle_from_origin :
  let C := {p : ℝ × ℝ | (p.1 - 9)^2 + (p.2 - 4)^2 = 56}
  in infi (λ (p : ℝ × ℝ), (p.1^2 + p.2^2)^(1/2)) - 2 * real.sqrt 14 = real.sqrt 97 - 2 * real.sqrt 14 :=
by
  let C := {p : ℝ × ℝ | (p.1 - 9)^2 + (p.2 - 4)^2 = 56}
  show infi (λ (p : ℝ × ℝ), (p.1^2 + p.2^2)^(1/2)) - 2 * real.sqrt 14 = real.sqrt 97 - 2 * real.sqrt 14
  sorry

end shortest_distance_to_circle_from_origin_l31_31570


namespace function_value_at_9_l31_31220

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-2, 0] then 3^x - 1 else 0  -- This captures only part of the definition given.

lemma odd_function (x : ℝ) : f (-x) = -f (x) := sorry

lemma periodic_function (x : ℝ) : f (x - 2) = f (x + 2) := sorry

theorem function_value_at_9 : f 9 = 2 / 3 :=
by
  have h1 : f 9 = f 1 := sorry
  have h2 : f 1 = -f (-1) := odd_function 1
  have h3 : f (-1) = 3^(-1) - 1 := sorry
  rw [h2, h3]
  norm_num
  sorry

end function_value_at_9_l31_31220


namespace arithmetic_pascal_l31_31030

-- Define the structure and properties of the Pascal triangle
def PascalTriangle (n k : ℕ) : ℕ := Nat.binomial n k

theorem arithmetic_pascal (s m : ℕ) 
  (h1: ∀ (k : ℕ), k ≠ 0 ∧ k ≠ m → PascalTriangle s k = 0) 
  (h2: ∀ (n k : ℕ), PascalTriangle s k = PascalTriangle (n + 1) k) 
  (h3: ∀ (n k : ℕ), PascalTriangle n (k - 1) + PascalTriangle n k = PascalTriangle (n + 1) k) 
  (h4: ∀ (n k : ℕ), ∃ (P : ℕ), PascalTriangle n k = P * PascalTriangle 0 0): 
  ∀ (n k : ℕ), PascalTriangle (n + 1) k = (P : ℕ) * PascalTriangle 0 0 :=
begin
  sorry
end

end arithmetic_pascal_l31_31030


namespace transformed_coords_of_point_l31_31757

noncomputable def polar_to_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def transformed_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let new_r := r ^ 3
  let new_θ := (3 * Real.pi / 2) * θ
  polar_to_rectangular_coordinates new_r new_θ

theorem transformed_coords_of_point (r θ : ℝ)
  (h_r : r = Real.sqrt (8^2 + 6^2))
  (h_cosθ : Real.cos θ = 8 / 10)
  (h_sinθ : Real.sin θ = 6 / 10)
  (coords_match : polar_to_rectangular_coordinates r θ = (8, 6)) :
  transformed_coordinates r θ = (-600, -800) :=
by
  -- The proof goes here
  sorry

end transformed_coords_of_point_l31_31757


namespace sum_inequality_l31_31872

theorem sum_inequality {n : ℕ} {a : ℕ → ℝ} (hpos : ∀ i ≤ n + 1, 0 < a i) :
  (∑ i in Finset.range n, a i) * (∑ i in Finset.range n, a (i + 1)) ≥
  (∑ i in Finset.range n, (a i * a (i + 1)) / (a i + a (i + 1))) * (∑ i in Finset.range n, (a i + a (i + 1))) :=
sorry

end sum_inequality_l31_31872


namespace percentage_less_than_m_plus_d_l31_31195

variable (m d : ℝ)

def is_symmetric_distribution (distribution : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, distribution m = distribution (2 * m - x)

def within_one_standard_deviation (distribution : ℝ → ℝ) : Prop := 
  distribution (m + d) - distribution (m - d) = 0.68

theorem percentage_less_than_m_plus_d (distribution : ℝ → ℝ) 
  (h1 : is_symmetric_distribution distribution) 
  (h2 : within_one_standard_deviation distribution) : 
  distribution (m + d) - distribution m = 0.84 :=
sorry

end percentage_less_than_m_plus_d_l31_31195


namespace value_of_a_l31_31784

theorem value_of_a (a : ℝ) (M N : set ℝ)
  (hM : M = {x | x - a = 0})
  (hN : N = {x | a * x - 1 = 0})
  (h_inter : M ∩ N = N)
  : a = 0 ∨ a = 1 ∨ a = -1 :=
sorry

end value_of_a_l31_31784


namespace min_of_f_l31_31311

noncomputable def f (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ k in Finset.range(3^n), abs ((∑ i in Finset.univ, (fin.val_tp2 k.nat_mod 3 i) * (x i)) - 1)

theorem min_of_f (n : ℕ) (x : Fin n → ℝ) (hx : ∀ k, x k = 1/(n+1)) : 
  f n x = ∑ k in Finset.range(3^n), abs ((∑ i in Finset.univ, (fin.val_tp2 k.nat_mod 3 i) / (n+1)) - 1) :=
sorry

end min_of_f_l31_31311


namespace sum_of_roots_of_quadratic_l31_31182

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h : a ≠ 0)
    (eq_quadratic : ∀ x, a * x^2 + b * x + c = 0) :
    (b = -16) → (a = 1) → (c = 4) → (-b / a) = 16 :=
by
  intros h_b h_a h_c
  rw [h_b, h_a]
  linarith
  sorry

end sum_of_roots_of_quadratic_l31_31182


namespace second_candy_cost_l31_31606

theorem second_candy_cost 
  (C : ℝ) 
  (hp := 25 * 8 + 50 * C = 75 * 6) : 
  C = 5 := 
  sorry

end second_candy_cost_l31_31606


namespace expected_socks_pairs_l31_31147

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l31_31147


namespace sum_of_consecutive_integers_l31_31964

theorem sum_of_consecutive_integers (x : ℤ) (h1 : x * (x + 1) + x + (x + 1) = 156) (h2 : x + 1 < 20) : x + (x + 1) = 23 :=
by
  sorry

end sum_of_consecutive_integers_l31_31964


namespace total_sheets_of_paper_l31_31462

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l31_31462


namespace possible_values_of_angle_B_l31_31021

theorem possible_values_of_angle_B (O H : Point) (A B C : Point) (R b : ℝ) 
  (circumcenter_triangle : is_circumcenter O A B C) (orthocenter_triangle : is_orthocenter H A B C) 
  (BO_BH_equal : dist B O = dist B H) 
  (angle_B_in_degrees : ℝ) :
  angle_B_in_degrees = 60 ∨ angle_B_in_degrees = 120 :=
by
  sorry

end possible_values_of_angle_B_l31_31021


namespace combined_yearly_return_percentage_l31_31213

-- Given conditions
def investment1 : ℝ := 500
def return_rate1 : ℝ := 0.07
def investment2 : ℝ := 1500
def return_rate2 : ℝ := 0.15

-- Question to prove
theorem combined_yearly_return_percentage :
  let yearly_return1 := investment1 * return_rate1
  let yearly_return2 := investment2 * return_rate2
  let total_yearly_return := yearly_return1 + yearly_return2
  let total_investment := investment1 + investment2
  ((total_yearly_return / total_investment) * 100) = 13 :=
by
  -- skipping the proof
  sorry

end combined_yearly_return_percentage_l31_31213


namespace solve_quadratic_l31_31100

def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic :
  (∃ p q : ℝ, (∀ x : ℂ, (quadratic_eq 5 (-7) 20 x) -> x = p + q * complex.I ∨ x = p - q * complex.I) ∧ 
  (p + q * q = 421 / 100)) := 
sorry

end solve_quadratic_l31_31100


namespace kaleb_books_count_l31_31013

/-- Kaleb's initial number of books. -/
def initial_books : ℕ := 34

/-- Number of books Kaleb sold. -/
def sold_books : ℕ := 17

/-- Number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Prove the number of books Kaleb has now. -/
theorem kaleb_books_count : initial_books - sold_books + new_books = 24 := by
  sorry

end kaleb_books_count_l31_31013


namespace number_of_female_students_l31_31119

variable (n m : ℕ)

theorem number_of_female_students (hn : n ≥ 0) (hm : m ≥ 0) (hmn : m ≤ n) : n - m = n - m :=
by
  sorry

end number_of_female_students_l31_31119


namespace number_of_solutions_l31_31684

-- Defining the conditions as hypotheses
def satisfies_conditions (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  ab + 2 * b * c = 92 ∧ ac + 3 * b * c = 39

-- Stating the theorem
theorem number_of_solutions : 
  (finset.filter satisfies_conditions (finset.product (finset.range 93) (finset.product (finset.range 93) (finset.range 65)))).card = 3 :=
sorry

end number_of_solutions_l31_31684


namespace donna_babysitting_hours_l31_31280

theorem donna_babysitting_hours 
  (total_earnings : ℝ)
  (dog_walking_hours : ℝ)
  (dog_walking_rate : ℝ)
  (dog_walking_days : ℝ)
  (card_shop_hours : ℝ)
  (card_shop_rate : ℝ)
  (card_shop_days : ℝ)
  (babysitting_rate : ℝ)
  (days : ℝ)
  (total_dog_walking_earnings : ℝ := dog_walking_hours * dog_walking_rate * dog_walking_days)
  (total_card_shop_earnings : ℝ := card_shop_hours * card_shop_rate * card_shop_days)
  (total_earnings_dog_card : ℝ := total_dog_walking_earnings + total_card_shop_earnings)
  (babysitting_hours : ℝ := (total_earnings - total_earnings_dog_card) / babysitting_rate) :
  total_earnings = 305 → dog_walking_hours = 2 → dog_walking_rate = 10 → dog_walking_days = 5 →
  card_shop_hours = 2 → card_shop_rate = 12.5 → card_shop_days = 5 →
  babysitting_rate = 10 → babysitting_hours = 8 :=
by
  intros
  sorry

end donna_babysitting_hours_l31_31280


namespace bottle_caps_found_l31_31244

theorem bottle_caps_found
  (caps_current : ℕ) 
  (caps_earlier : ℕ) 
  (h_current : caps_current = 32) 
  (h_earlier : caps_earlier = 25) :
  caps_current - caps_earlier = 7 :=
by 
  sorry

end bottle_caps_found_l31_31244


namespace KW_price_percentage_l31_31668

noncomputable def combined_assets_percentage (P A B : ℝ) : ℝ :=
  let combinedAssets := (P / 1.60) + (P / 2.00)
  in (P / combinedAssets) * 100

theorem KW_price_percentage (P A B : ℝ) 
  (h1 : P = 1.60 * A)
  (h2 : P = 2.00 * B) :
  combined_assets_percentage P A B ≈ 88.89 :=
by 
  sorry

end KW_price_percentage_l31_31668


namespace domain_of_function_l31_31681

theorem domain_of_function :
  {x : ℝ | 2 + x - x^2 > 0 ∧ |x| - x ≠ 0} = set.Ioo (-1 : ℝ) 0 := 
begin
  sorry
end

end domain_of_function_l31_31681


namespace x_equals_32_l31_31737

def find_x (L : list ℕ) (x : ℕ) : Prop :=
  L = [23, 28, 30, x, 34, 39] ∧
  (L.nth 2).get_or_else 0 + (L.nth 3).get_or_else 0 = 62 

theorem x_equals_32 : ∃ x, find_x [23, 28, 30, x, 34, 39] x ∧ x = 32 :=
by
  sorry

end x_equals_32_l31_31737


namespace range_of_t_l31_31319

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  ∃ t : ℝ, (t = a^2 - a*b + b^2) ∧ (1/3 ≤ t ∧ t ≤ 3) :=
sorry

end range_of_t_l31_31319


namespace solve_equation_l31_31689

theorem solve_equation (x : ℝ) :
  (x^2 + 2*x + 1 = abs (3*x - 2)) ↔ 
  (x = (-7 + Real.sqrt 37) / 2) ∨ 
  (x = (-7 - Real.sqrt 37) / 2) :=
by
  sorry

end solve_equation_l31_31689


namespace krishan_money_l31_31592

theorem krishan_money (R G K : ℕ) (h₁ : 7 * G = 17 * R) (h₂ : 7 * K = 17 * G) (h₃ : R = 686) : K = 4046 :=
  by sorry

end krishan_money_l31_31592


namespace find_roots_l31_31716

theorem find_roots (a b c d x : ℝ) (h₁ : a + d = 2015) (h₂ : b + c = 2015) (h₃ : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := 
sorry

end find_roots_l31_31716


namespace points_on_circumcircle_of_orthocenter_symmetry_l31_31483

noncomputable def orthocenter_symmetry (ABC : Triangle) (H A_1 B_1 C_1 : Point) : Prop :=
  let A := ABC.vertex_A in
  let B := ABC.vertex_B in
  let C := ABC.vertex_C in
  H = ABC.orthocenter ∧
  A_1 = H.symmetric_over BC ∧
  B_1 = H.symmetric_over CA ∧
  C_1 = H.symmetric_over AB ∧
  A_1 ∈ ABC.circumcircle ∧
  B_1 ∈ ABC.circumcircle ∧
  C_1 ∈ ABC.circumcircle

variable (ABC : Triangle)
variable (H A_1 B_1 C_1 : Point)

theorem points_on_circumcircle_of_orthocenter_symmetry :
  orthocenter_symmetry ABC H A_1 B_1 C_1 :=
sorry

end points_on_circumcircle_of_orthocenter_symmetry_l31_31483


namespace mall_spending_l31_31467

-- Define the given conditions
def cost_per_movie := 24 : ℤ
def num_movies := 3 : ℤ
def total_movie_cost := num_movies * cost_per_movie

def cost_per_bag_of_beans := (125 / 100) : ℚ
def num_bags_of_beans := 20 : ℤ
def total_bean_cost := num_bags_of_beans * cost_per_bag_of_beans

def total_spent := 347 : ℤ

-- Define the total for movies and beans combined
def total_movies_and_beans_cost := total_movie_cost + (num_bags_of_beans * cost_per_bag_of_beans).toRat

-- The final statement to be proved
theorem mall_spending : (total_spent : ℚ) - total_movies_and_beans_cost = 250 := by
  sorry

end mall_spending_l31_31467


namespace tim_initial_soda_l31_31545

-- Define the problem
def initial_cans (x : ℕ) : Prop :=
  let after_jeff_takes := x - 6
  let after_buying_more := after_jeff_takes + after_jeff_takes / 2
  after_buying_more = 24

-- Theorem stating the problem in Lean 4
theorem tim_initial_soda (x : ℕ) (h: initial_cans x) : x = 22 :=
by
  sorry

end tim_initial_soda_l31_31545


namespace chords_from_nine_points_l31_31077

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31077


namespace a4_is_15_l31_31970

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * sequence (n - 1) + 1

theorem a4_is_15 : sequence 4 = 15 := by
  sorry

end a4_is_15_l31_31970


namespace leap_year_march_has_five_mondays_l31_31106

theorem leap_year_march_has_five_mondays 
    (is_leap_year : ℕ -> Prop)
    (feb_has_29_days : ∀ (year : ℕ), is_leap_year year → 29 = 29)
    (five_sundays_in_feb : ∀ (year : ℕ), is_leap_year year → (∃ (days : list ℕ), 
        days.length = 5 ∧ ∀ d, d ∈ days → d % 7 = 0 ∧ d ≤ 29)) : 
    ∀ (year : ℕ), is_leap_year year → (∃ (days : list ℕ), 
        days.length = 5 ∧ ∀ (monday : ℕ

    monday ∈ days → ∀ m, ∀ i < 31, (1 % 31️⃣) = (m + 1) % 🗓10)) :=
by
    sorry

end leap_year_march_has_five_mondays_l31_31106


namespace opposite_of_neg_one_over_2023_l31_31959

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31959


namespace work_done_by_force_field_is_zero_l31_31297

noncomputable def work_done (x y a m : ℝ) (t : ℝ → ℝ) : ℝ :=
  let F_x := m * (x + y)
  let F_y := -m * x
  ∮ (F_x * (-a * (Real.sin ∘ t) t) + F_y * (-a * (Real.cos ∘ t) t))

theorem work_done_by_force_field_is_zero (m a : ℝ) (t : ℝ) :
  let x := a * Real.cos t
  let y := -a * Real.sin t
  work_done x y a m t = 0 :=
by
  sorry

end work_done_by_force_field_is_zero_l31_31297


namespace pastries_left_l31_31299

def pastries_baked : ℕ := 4 + 29
def pastries_sold : ℕ := 9

theorem pastries_left : pastries_baked - pastries_sold = 24 :=
by
  -- assume pastries_baked = 33
  -- assume pastries_sold = 9
  -- prove 33 - 9 = 24
  sorry

end pastries_left_l31_31299


namespace base8_add_sub_example_l31_31565

theorem base8_add_sub_example : 
  let b : ℕ := 8,
  let n1 := 2 * b + 4,
  let n2 := 5 * b + 3,
  let n3 := 1 * b + 7,
  let sum := n1 + n2,
  let result := sum - n3
  in natDigits b result = [0, 6] := 
by 
  let b : ℕ := 8 in
  let n1 := 2 * b + 4 in
  let n2 := 5 * b + 3 in
  let n3 := 1 * b + 7 in
  let sum := n1 + n2 in
  let result := sum - n3 in
  have h1 : natDigits b n1 = [4, 2] := by sorry,
  have h2 : natDigits b n2 = [3, 5] := by sorry,
  have h3 : natDigits b n3 = [7, 1] := by sorry,
  have h4 : natDigits b sum = [7, 7] := by sorry,
  have h5 : natDigits b result = [0, 6] := by sorry,
  exact h5

end base8_add_sub_example_l31_31565


namespace parabola_chord_dot_product_l31_31410

theorem parabola_chord_dot_product:
  (∃ (y_1 y_2 t : ℝ), (y_1 + y_2 = 4 * t) ∧ (y_1 * y_2 = -4) ∧
   (let x_1 := t * y_1 + 1 in let x_2 := t * y_2 + 1 in
    (x_1 * x_2 + y_1 * y_2 = -3))) :=
sorry

end parabola_chord_dot_product_l31_31410


namespace limit_expression_l31_31206

theorem limit_expression :
  (Real.Lim (fun x => ((x^2 + 3*x + 2)^2) / (x^3 + 2*x^2 - x - 2)) (-1) = 0) :=
sorry

end limit_expression_l31_31206


namespace polygon_area_limit_is_correct_l31_31272

-- Define the initial polygon P0 and its characteristics 
def P₀ : convex_poly.to_simple_poly := 
  convex_poly.to_simple_poly.mk (Simplex 2 (Equilateral 1))

-- Define the function that generates P_{n+1} from P_n
def next_polygon (P_n : convex_poly.to_simple_poly) : convex_poly.to_simple_poly :=
  P_n.cut_corners_one_third

-- Define the sequence of polygons {P_n}
noncomputable def P_seq : ℕ → convex_poly.to_simple_poly
| 0       => P₀
| (n + 1) => next_polygon (P_seq n)

-- Define and compute the area of the polygons in the sequence
noncomputable def polygon_area (P : convex_poly.to_simple_poly) : ℝ :=
  convex_poly.area P

-- Define the limit of the area of {P_n} as n approaches infinity
noncomputable def area_limit : ℝ :=
  real.lim (polygon_area ∘ P_seq)

-- Statement of the problem: Prove that the limit of the areas is sqrt(3)/7
theorem polygon_area_limit_is_correct : area_limit = real.sqrt 3 / 7 :=
sorry

end polygon_area_limit_is_correct_l31_31272


namespace opposite_of_neg_frac_l31_31945

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31945


namespace max_base_isosceles_triangle_l31_31194

theorem max_base_isosceles_triangle (R x : ℝ) :
  (∃ y : ℝ, 
    is_isosceles_triangle_in_semicircle R x y ∧ 
    y = 2 * R - x) → 
  x = (4 / 3) * R :=
sorry

-- Definitions:
def is_isosceles_triangle_in_semicircle (R x y : ℝ) : Prop :=
  ∃ (A B C : E), 
    isosceles_triangle A B C ∧ 
    on_diameter A C R ∧ 
    is_chord B C R

def isosceles_triangle (A B C : E) : Prop :=
  dist B A = dist B C

def on_diameter (A C : E) (R : ℝ) : Prop :=
  dist A C = 2 * R

def is_chord (B C : E) (R : ℝ) : Prop :=
  dist B C < 2 * R

end max_base_isosceles_triangle_l31_31194


namespace chords_from_nine_points_l31_31050

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31050


namespace mindy_tax_rate_l31_31037

variables (M : ℝ) -- Mork's income
variables (r : ℝ) -- Mindy's tax rate

-- Conditions
def Mork_tax_rate := 0.45 -- 45% tax rate
def Mindx_income := 4 * M -- Mindy earned 4 times as much as Mork
def combined_tax_rate := 0.21 -- Combined tax rate is 21%

-- Equation derived from the conditions
def combined_tax_rate_eq := (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.21

theorem mindy_tax_rate : combined_tax_rate_eq M r → r = 0.15 :=
by
  intros conditional_eq
  sorry

end mindy_tax_rate_l31_31037


namespace find_hyperbola_m_l31_31355

theorem find_hyperbola_m (m : ℝ) :
  (∃ (a b : ℝ), a^2 = 16 ∧ b^2 = m ∧ (sqrt (1 + m / 16) = 5 / 4)) → m = 9 :=
by
  intro h
  sorry

end find_hyperbola_m_l31_31355


namespace sufficient_but_not_necessary_condition_for_increasing_on_interval_l31_31690

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def abs_fn (a x : ℝ) : ℝ := abs (x - a)

theorem sufficient_but_not_necessary_condition_for_increasing_on_interval :
  (is_increasing_on_interval (abs_fn 2) 2 (real.infty)) ∧
  (∀ a, is_increasing_on_interval (abs_fn a) 2 (real.infty) → a ≤ 2) :=
sorry

end sufficient_but_not_necessary_condition_for_increasing_on_interval_l31_31690


namespace area_JLK_l31_31423

-- Definitions and conditions
variables (G H I J K L : Type) [is_triangle G H I]
variables (J_midpoint : is_midpoint J G H)
variables (K_midpoint : is_midpoint K G I)
variables (L_midpoint : is_midpoint L G I)
constant area_GHI : ℝ := 120
def area_triangle (A B C : Type) : ℝ := sorry

-- Statement to prove
theorem area_JLK (G H I J K L : Type) [is_triangle G H I]
  [J_midpoint : is_midpoint J G H]
  [K_midpoint : is_midpoint K G I]
  [L_midpoint : is_midpoint L G I]
  (area_GHI : area_triangle G H I = 120) :
  area_triangle J L K = 30 := sorry

end area_JLK_l31_31423


namespace divisible_by_7_divisibility_rule_7_divisibility_rule_13_l31_31210

open Nat

-- Proof statement for part (a)
theorem divisible_by_7 (abcdef abc def : ℕ) (h1 : abcdef = 1000 * abc + def) (h2 : (abc - def) % 7 = 0) : abcdef % 7 = 0 := sorry

-- Proof statement for part (b)
theorem divisibility_rule_7 (N : ℕ) : (alternating sum of N's digits in groups of three from right to left) % 7 = 0 ↔ N % 7 = 0 := sorry

-- Proof statement for part (c)
theorem divisibility_rule_13 (N : ℕ) : (alternating sum of N's digits in groups of three from right to left) % 13 = 0 ↔ N % 13 = 0 := sorry

end divisible_by_7_divisibility_rule_7_divisibility_rule_13_l31_31210


namespace hyperbola_eccentricity_l31_31357

variable (m n : ℝ)
variable (m_neg : m < 0)
variable (n_pos : 0 < n)
variable (h_asymptote : ∀ x y : ℝ, (y = sqrt 2 * x ∨ y = -sqrt 2 * x) → true)

theorem hyperbola_eccentricity (h_hyperbola : ∀ x y : ℝ, (x^2 / m + y^2 / n = 1) → true) :
  ∃ e : ℝ, e = sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l31_31357


namespace find_complex_solutions_l31_31287

noncomputable def complex_sol : set ℂ :=
  {z | ∃ a b : ℝ, z = complex.mk a b ∧
                   (z^2 = complex.mk (-81 : ℝ) (-48 : ℝ)) ∧
                   (a = real.sqrt ((-81 + real.sqrt 8865) / 2)) ∧
                   (b = -24 / a)}

theorem find_complex_solutions (z : ℂ) :
  z^2 = complex.mk (-81) (-48) → z ∈ complex_sol := by
  sorry

end find_complex_solutions_l31_31287


namespace cone_circumference_l31_31228

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_circumference (V h r C : ℝ) (hV : V = 27 * π) (hh : h = 9)
  (hV_formula : V = volume_of_cone r h) (hC : C = 2 * π * r) : 
  C = 6 * π :=
by 
  sorry

end cone_circumference_l31_31228


namespace slope_y_intercept_product_l31_31933

theorem slope_y_intercept_product (m b : ℝ) (hm : m = -1/2) (hb : b = 4/5) : -1 < m * b ∧ m * b < 0 :=
by
  sorry

end slope_y_intercept_product_l31_31933


namespace Aziz_parents_years_in_America_l31_31659

noncomputable def years_lived_in_America_before_Aziz : ℕ := 
  let birth_year := 2021 - 36  -- Aziz's birth year
  in birth_year - 1982  -- Years lived in America before Aziz's birth

theorem Aziz_parents_years_in_America :
  years_lived_in_America_before_Aziz = 3 :=
  by
    simp [years_lived_in_America_before_Aziz]
    sorry

end Aziz_parents_years_in_America_l31_31659


namespace number_of_triangles_l31_31526

def countTriangles : ℕ :=
  let k_range := List.range' (-15) 31 -- [-15, -14, ..., 14, 15]
  let intersections := {t : ℤ × ℤ × ℤ | ∃ k ∈ k_range, ∃ x, y = k ∧ y = 2 * x + 3 * k ∧ y = -2 * x + 3 * k ∧ side_length = √3}
  intersections.count

theorem number_of_triangles : countTriangles = 806 := by
  sorry

end number_of_triangles_l31_31526


namespace greatest_possible_value_of_a_l31_31926

noncomputable def greatest_a : ℤ :=
  let f (a : ℤ) (x : ℤ) := x^2 + a * x = -18
  if ∃ x : ℤ, f 19 x then 19 else 0

theorem greatest_possible_value_of_a (a : ℤ) (x : ℤ) (h : x^2 + a * x = -18) (h_pos : a > 0) :
  a ≤ greatest_a := by
  sorry

end greatest_possible_value_of_a_l31_31926


namespace star_value_l31_31390

-- Define the operation &
def and_operation (a b : ℕ) : ℕ := (a + b) * (a - b)

-- Define the operation star
def star_operation (c d : ℕ) : ℕ := and_operation c d + 2 * (c + d)

-- The proof problem
theorem star_value : star_operation 8 4 = 72 :=
by
  sorry

end star_value_l31_31390


namespace find_p_l31_31324

theorem find_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 :=
sorry

end find_p_l31_31324


namespace savings_prediction_l31_31295

noncomputable def monthly_savings (x : ℝ) : ℝ :=
  0.3 * x - 0.4

theorem savings_prediction (x : ℝ) : monthly_savings 7 = 1.7 :=
by 
  calc
    monthly_savings 7 = 0.3 * 7 - 0.4 : rfl
                    ... = 2.1 - 0.4 : by norm_num
                    ... = 1.7 : by norm_num

end savings_prediction_l31_31295


namespace find_m_plus_n_l31_31334

noncomputable def quadratic_equation (m : ℝ) : Polynomial ℂ := Polynomial.C 2 + Polynomial.C m * Polynomial.X + Polynomial.X^2

theorem find_m_plus_n
  (m n : ℝ)
  (h1 : quadratic_equation m).eval (1 + Complex.i * n) = 0
  (hm : m ∈ ℝ)
  (hn : n ∈ Set.Ioi 0) :
  m + n = -1 :=
sorry

end find_m_plus_n_l31_31334


namespace max_sections_with_5_lines_l31_31827

theorem max_sections_with_5_lines : ∃ (n : ℕ), n = 16 ∧
  ∀ (rectangle : Type) (line_segment : Type) 
  (draw_lines : rectangle → line_segment → ℕ), 
  draw_lines (r : rectangle) (l : line_segment) = 5 → 
  sections_created_by_lines (r : rectangle) (l : line_segment) = 16 :=
begin
  sorry
end

end max_sections_with_5_lines_l31_31827


namespace expected_pairs_socks_l31_31151

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l31_31151


namespace max_sections_with_5_lines_l31_31833

theorem max_sections_with_5_lines (n : ℕ) (h₀ : n = 0 → sections n = 1)
  (h₁ : n = 1 → sections n = 2) (h₂ : n = 2 → sections n = 4)
  (h₃ : n = 3 → sections n = 7) (h₄ : n = 4 → sections n = 11) :
  sections 5 = 16 :=
sorry

end max_sections_with_5_lines_l31_31833


namespace stan_caught_13_pieces_l31_31108

theorem stan_caught_13_pieces (S: ℕ) :
  (Tabitha Julie Carlos: ℕ) 
    (H1: Tabitha = 22)
    (H2: Julie = Tabitha / 2)
    (H3: Carlos = 2 * S)
    (H_total: Tabitha + S + Julie + Carlos = 72)
    : S = 13 :=
    by
    sorry

end stan_caught_13_pieces_l31_31108


namespace find_point_M_l31_31474

theorem find_point_M (y : ℝ) :
  (∃ y0, y0 = 0 ∨ y0 = -5 / 3 ∧ y = y0) ↔ 
  ∃ (k1 k2 : ℝ), 
  -- Conditions from the problem
  let x := (Real.sqrt 3 / 2)
  ∧ y = (k1^2 / 2 - k1^2 + k1 * x) 
  ∧ y = (k2^2 / 2 - k2^2 + k2 * x)
  ∧ k1 + k2 = (Real.sqrt 3)
  ∧ k1 * k2 = 2 * y :=
sorry

end find_point_M_l31_31474


namespace evaluate_f_at_1_l31_31340

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem evaluate_f_at_1 : f 1 = 6 := 
  sorry

end evaluate_f_at_1_l31_31340


namespace solve_for_m_l31_31719

noncomputable def operation (a b c x y : ℝ) := a * x + b * y + c * x * y

theorem solve_for_m (a b c : ℝ) (h1 : operation a b c 1 2 = 3)
                              (h2 : operation a b c 2 3 = 4) 
                              (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), operation a b c x m = x) :
  ∃ (m : ℝ), m = 4 :=
sorry

end solve_for_m_l31_31719


namespace oil_used_l31_31503

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l31_31503


namespace min_gumballs_to_ensure_four_same_color_l31_31629

/-- A structure to represent the number of gumballs of each color. -/
structure Gumballs :=
(red : ℕ)
(white : ℕ)
(blue : ℕ)
(green : ℕ)

def gumball_machine : Gumballs := { red := 10, white := 9, blue := 8, green := 6 }

/-- Theorem to state the minimum number of gumballs required to ensure at least four of any color. -/
theorem min_gumballs_to_ensure_four_same_color 
  (g : Gumballs) 
  (h1 : g.red = 10)
  (h2 : g.white = 9)
  (h3 : g.blue = 8)
  (h4 : g.green = 6) : 
  ∃ n, n = 13 := 
sorry

end min_gumballs_to_ensure_four_same_color_l31_31629


namespace half_angle_quadrant_l31_31749

-- Let α be an angle in the first quadrant, meaning 2kπ < α < π/2 + 2kπ for some integer k.
variable (k : ℤ) (α : ℝ)
hypothesis (hα : (2 * k * Real.pi) < α ∧ α < (Real.pi / 2) + (2 * k * Real.pi))

-- We want to show that α / 2 is either in the first or third quadrant.
theorem half_angle_quadrant :
  (∃ i : ℤ, (2 * i * Real.pi) < (α / 2) ∧ (α / 2) < (Real.pi / 4) + (2 * i * Real.pi)) ∨
  (∃ i : ℤ, (Real.pi + 2 * i * Real.pi) < (α / 2) ∧ (α / 2) < (5 * Real.pi / 4) + (2 * i * Real.pi)) :=
sorry

end half_angle_quadrant_l31_31749


namespace minimum_a_for_f_leq_one_range_of_a_for_max_value_l31_31772

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * log x - (1 / 3) * a * x^3 + 2 * x

theorem minimum_a_for_f_leq_one :
  ∀ {a : ℝ}, (a > 0) → (∀ x : ℝ, f a x ≤ 1) → (a ≥ 3) :=
sorry

theorem range_of_a_for_max_value :
  ∀ {a : ℝ}, (a > 0) → (∃ B : ℝ, ∀ x : ℝ, f a x ≤ B) ↔ (0 < a ∧ a ≤ (3 / 2) * exp 3) :=
sorry

end minimum_a_for_f_leq_one_range_of_a_for_max_value_l31_31772


namespace max_gold_coins_l31_31579

theorem max_gold_coins (n : ℤ) (h₁ : ∃ k : ℤ, n = 13 * k + 3) (h₂ : n < 150) : n ≤ 146 :=
by {
  sorry -- Proof not required as per instructions
}

end max_gold_coins_l31_31579


namespace exists_vector_pointing_from_origin_l31_31270

-- Define the parameterized line
def line_param (t : ℝ) : ℝ × ℝ :=
  (3 * t + 1, t + 1)

-- Define the parallel vector
def parallel_vector (k : ℝ) : ℝ × ℝ :=
  (3 * k, k)

-- Define the existence of the needed vector
theorem exists_vector_pointing_from_origin : 
  ∃ (a b : ℝ),
    ∃ (t : ℝ),
      (a, b) = line_param t ∧ 
      (a, b) = parallel_vector 3 :=
begin
  use [9, 3], -- Provide specific values for a and b
  use 2, -- Provide the specific value for t
  split,
  { -- Prove the first part: (a, b) = line_param t
    unfold line_param,
    norm_num,
  },
  { -- Prove the second part: (a, b) = parallel_vector 3
    unfold parallel_vector,
    norm_num,
  },
end

end exists_vector_pointing_from_origin_l31_31270


namespace valid_range_and_distances_l31_31158

variables a x y z : ℝ

theorem valid_range_and_distances (h1: x + y = 4 * z)
                                   (h2: z + y = x + a)
                                   (h3: x + z = 85)
                                   (a_pos: 0 < a)
                                   (a_lt_68: a < 68) :
  0 < a ∧ a < 68 ∧
  (a = 5 → x = 60 ∧ y = 40 ∧ z = 25) :=
begin
  sorry
end

end valid_range_and_distances_l31_31158


namespace matrix_power_minus_l31_31856

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l31_31856


namespace distance_between_lines_l1_l2_l31_31116

-- Definitions of the equations of the lines
def l1 (x y : ℝ) : Prop := x - y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 3 * y + 2 = 0

-- Definition of the distance between two parallel lines
def distance_between_parallel_lines
(a b c1 c2 : ℝ) : ℝ :=
(abs (c2 - c1)) / (Real.sqrt (a^2 + b^2))

-- Constants based on the conditions of the problem
def a : ℝ := 3
def b : ℝ := -3
def c1 : ℝ := 18
def c2 : ℝ := 2

-- The theorem statement, proving the distance is as given in the problem
theorem distance_between_lines_l1_l2 :
  distance_between_parallel_lines a b c1 c2 = 8 * Real.sqrt 2 / 3 :=
by sorry

end distance_between_lines_l1_l2_l31_31116


namespace smallest_a1_pos_l31_31875

noncomputable def sequence (a : ℕ → ℚ) : ℕ → ℚ
| 0     := a 0
| (n+1) := 11 * sequence n - (n + 1)

theorem smallest_a1_pos {
  a₀ : ℚ
  (h₀ : 0 < a₀)
  (hpos : ∀ n, 0 < sequence (λ n, n.rec_on a₀ (λ n a, 11 * a - n))) :
  (∃ m n : ℕ, nat.coprime m n ∧ a₀ = m / n ∧ m + n = 121) :=
sorry

end smallest_a1_pos_l31_31875


namespace solution_l31_31253

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∀ (O A B C D E F G : Point),
  let O := circumcircle A B C in
  tangent (D, A) O →
  ∠DBA = ∠ABC →
  E ∈ O ∧ (E ≠ A ∧ E ≠ B ∧ E ≠ C) →
  B ∈ line (D, E) ∧ B ≠ E →
  F ∈ O ∧ (F ≠ A ∧ F ≠ B ∧ F ≠ C) →
  BF ∥ EC →
  G = intersection_point_extension (C, F) (D, A) →
  AG = AD
  
theorem solution : problem_statement :=
begin
  sorry
end

end solution_l31_31253


namespace elective_schemes_total_l31_31232

def num_elective_schemes (total_courses : ℕ) (exclusive_courses : ℕ) (remaining_courses : ℕ) (select_courses : ℕ) : ℕ :=
  let none_exclusive := Nat.choose remaining_courses select_courses
  let one_exclusive := (Nat.choose exclusive_courses 1) * (Nat.choose remaining_courses (select_courses - 1))
  none_exclusive + one_exclusive

theorem elective_schemes_total : 
  num_elective_schemes 10 3 7 3 = 98 := 
by
  unfold num_elective_schemes
  simp [Nat.choose, factorial, -Nat.factorial_succ, -Nat.succ_sub]
  sorry

end elective_schemes_total_l31_31232


namespace find_a_l31_31305

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x > 0 then log x else x + ∫ t in set.Ioc 0 a, (3 * t^2)

theorem find_a (a : ℝ) (h : f a (f a 1) = 1) : a = 1 :=
by 
  sorry

end find_a_l31_31305


namespace trapezoid_problem_l31_31090

theorem trapezoid_problem (b h x : ℝ) 
  (hb : b > 0)
  (hh : h > 0)
  (h_ratio : (b + 90) / (b + 30) = 3 / 4)
  (h_x_def : x = 150 * (h / (x - 90) - 90))
  (hx2 : x^2 = 26100) :
  ⌊x^2 / 120⌋ = 217 := sorry

end trapezoid_problem_l31_31090


namespace power_function_solution_l31_31801

theorem power_function_solution (f : ℝ → ℝ) (alpha : ℝ)
  (h₀ : ∀ x, f x = x ^ alpha)
  (h₁ : f (1 / 8) = 2) :
  f (-1 / 8) = -2 :=
sorry

end power_function_solution_l31_31801


namespace total_artworks_created_l31_31032

/-- Conditions: Liam is teaching a class with 24 students and 36 art kits.
  12 students share 1 kit each, and 12 students share 2 kits each. 
  From the art kits distributed, 8 students make 5 artworks each, 
  10 students make 6 artworks each, and 6 students make 7 artworks each. 
  We need to prove that the total number of artistic works created 
  by the whole class is 142.
-/
theorem total_artworks_created (students : ℕ) (art_kits : ℕ)
  (students_share1_kit : ℕ) (students_share2_kits : ℕ)
  (group1 : ℕ) (artworks1 : ℕ) (group2 : ℕ) (artworks2 : ℕ)
  (group3 : ℕ) (artworks3 : ℕ) :
  students = 24 →
  art_kits = 36 →
  students_share1_kit = 12 →
  students_share2_kits = 12 →
  group1 = 8 → artworks1 = 5 →
  group2 = 10 → artworks2 = 6 →
  group3 = 6 → artworks3 = 7 →
  (group1 * artworks1 + group2 * artworks2 + group3 * artworks3) = 142 :=
by {
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10,
  sorry
}

end total_artworks_created_l31_31032


namespace max_subset_card_l31_31855

def subset_T (n : ℕ) : set ℕ := {x | x ∈ finset.range n}

def pairwise_coprime (s : set ℕ) :=
  ∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → (a + b) % 9 ≠ 0

theorem max_subset_card (T : set ℕ) (hT : T ⊆ subset_T 60) (hPairwise : pairwise_coprime T) :
  ∃ S ⊆ T, S.card = 34 := 
sorry

end max_subset_card_l31_31855


namespace journey_time_l31_31528

-- Conditions
def initial_speed : ℝ := 80  -- miles per hour
def initial_time : ℝ := 5    -- hours
def new_speed : ℝ := 50      -- miles per hour
def distance : ℝ := initial_speed * initial_time

-- Statement
theorem journey_time :
  distance / new_speed = 8.00 :=
by
  sorry

end journey_time_l31_31528


namespace DE_length_l31_31819

noncomputable def DE (EF : ℝ) (angleD : ℝ) : ℝ :=
  EF / Real.tan angleD

theorem DE_length (EF : ℝ) (angleD degrees : ℝ) (hD : angleD = 25 * Real.pi / 180) (hE : angleD + degrees = Real.pi / 2) :
  (DE EF angleD) ≈ 19.3 :=
by
  rw [DE, hD]
  sorry

end DE_length_l31_31819


namespace angel_score_l31_31824

theorem angel_score
    (beth_score : ℕ) 
    (jan_score : ℕ) 
    (judy_score : ℕ) 
    (point_difference : ℕ)
    (first_team_total : beth_score + jan_score = 22)
    (point_differential_condition : beth_score + jan_score = judy_score + point_difference + 3) :
    ∃ (angel_score : ℕ), judy_score + angel_score = 19 ∧ angel_score = 11 :=
by {
  intro h,
  use 11,
  split,
  { exact (by linarith), },
  { refl, },
}

end angel_score_l31_31824


namespace find_vector_a_l31_31781

noncomputable def vector_a (x y : ℝ) : ℝ × ℝ × ℝ :=
  (x, y, 1)

def A := (0, 2, 3 : ℝ)
def B := (-2, 1, 6 : ℝ)
def C := (1, -1, 5 : ℝ)

def AB : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def AC : ℝ × ℝ × ℝ :=
  (C.1 - A.1, C.2 - A.2, C.3 - A.3)

def is_perpendicular (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

theorem find_vector_a :
  ∃ x y : ℝ, vector_a x y = (1, 1, 1) ∧
  is_perpendicular (vector_a x y) AB ∧
  is_perpendicular (vector_a x y) AC :=
begin
  sorry
end

end find_vector_a_l31_31781


namespace h_at_2_l31_31384

noncomputable def h (x : ℝ) : ℝ := 
(x + 2) * (x - 1) * (x + 4) * (x - 3) - x^2

theorem h_at_2 : 
  h (-2) = -4 ∧ h (1) = -1 ∧ h (-4) = -16 ∧ h (3) = -9 → h (2) = -28 := 
by
  intro H
  sorry

end h_at_2_l31_31384


namespace reciprocal_of_neg_2023_l31_31131

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l31_31131


namespace cost_of_hair_dye_l31_31491

/-
Rebecca runs a hair salon. She charges $30 for haircuts, $40 for perms, and $60 for dye jobs, but she has to buy a box of hair dye for some amount to dye every head of hair. Today, she has four haircuts, one perm, and two dye jobs scheduled. If she makes $50 in tips, she will have $310 at the end of the day. Prove that the cost of a box of hair dye is $10.
-/
theorem cost_of_hair_dye
  (haircut_cost : ℕ)
  (perm_cost : ℕ)
  (dyejob_cost : ℕ)
  (haircuts_scheduled : ℕ)
  (perms_scheduled : ℕ)
  (dyejobs_scheduled : ℕ)
  (tips : ℕ)
  (final_amount : ℕ)
  (total_revenue_without_dye_cost : ℕ := (haircut_cost * haircuts_scheduled) + (perm_cost * perms_scheduled) + (dyejob_cost * dyejobs_scheduled) + tips)
  (total_deduction : ℕ := total_revenue_without_dye_cost - final_amount)
  (num_dye_jobs : ℕ := dyejobs_scheduled)
  (box_cost : ℕ := total_deduction / num_dye_jobs) : box_cost = 10 := 
sorry

#eval cost_of_hair_dye 30 40 60 4 1 2 50 310 -- Expecting the theorem to hold valid proofs

end cost_of_hair_dye_l31_31491


namespace real_part_of_diff_times_i_l31_31798

open Complex

def z1 : ℂ := (4 : ℂ) + (29 : ℂ) * I
def z2 : ℂ := (6 : ℂ) + (9 : ℂ) * I

theorem real_part_of_diff_times_i :
  re ((z1 - z2) * I) = -20 := 
sorry

end real_part_of_diff_times_i_l31_31798


namespace chords_from_nine_points_l31_31049

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31049


namespace range_of_k_l31_31327

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

noncomputable def g (x : ℝ) : ℝ := x^2 - 1

noncomputable def h (x : ℝ) : ℝ := x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g (k * x + k / x) < g (x^2 + 1 / x^2 + 1)) ↔ (-3 / 2 < k ∧ k < 3 / 2) :=
by
  sorry

end range_of_k_l31_31327


namespace word_combination_count_l31_31428

theorem word_combination_count : 
  ∀ (alphabet : Type) [has_cardinal alphabet] [fintype alphabet], 
  fintype.card alphabet = 26 → 
  (number_of_words : ℕ) = (fintype.card alphabet)^3 → 
  number_of_words = 17576 :=
begin
  intros alphabet h1 h2 h3 h4,
  sorry
end

end word_combination_count_l31_31428


namespace population_difference_correct_l31_31977

-- Define initial populations of cities A, B, and C
variables {A B C : ℝ}

-- Condition: The total population of city A and city B is 5000 more than the total population of city B and city C
def population_condition : Prop := A + B = B + C + 5000

-- Define the annual growth rates
def growth_rate_A : ℝ := 0.03
def growth_rate_C : ℝ := 0.02

-- Define the population of cities after 2 years
def population_A_after_2_years (A : ℝ) : ℝ := A * (1 + growth_rate_A)^2
def population_C_after_2_years (C : ℝ) : ℝ := C * (1 + growth_rate_C)^2

-- Define the required difference in population after 2 years
def population_difference_after_2_years (A C : ℝ) : ℝ :=
  population_A_after_2_years A - population_C_after_2_years C

-- Prove the required difference in population is 0.0205A + 5202
theorem population_difference_correct
  (A C : ℝ)
  (h : population_condition) : population_difference_after_2_years A C = 0.0205 * A + 5202 :=
sorry

end population_difference_correct_l31_31977


namespace sum_of_cubes_inequality_l31_31972

variable (n : ℕ) (a : ℕ → ℕ)

-- Define the conditions in Lean
def a_0_eq_0 : Prop := a 0 = 0
def a_sequence_condition : Prop := ∀ k < n, 0 ≤ a (k + 1) - a k ∧ a (k + 1) - a k ≤ 1

-- Define the inequality
def inequality_lemma : Prop := (∑ k in Finset.range (n + 1), (a k) ^ 3) ≤ (∑ k in Finset.range (n + 1), a k) ^ 2

-- The theorem statement in Lean 4
theorem sum_of_cubes_inequality (h0 : a_0_eq_0 a) (h_seq : a_sequence_condition n a) : inequality_lemma n a :=
by
  sorry

end sum_of_cubes_inequality_l31_31972


namespace monotonic_intervals_range_of_k_k_greater_than_gx0_l31_31773

-- Condition: The function definition
def f (x : ℝ) (k : ℝ) : ℝ := log x + k / x

-- Problem 1: The monotonic intervals
theorem monotonic_intervals (x : ℝ) (h : 0 < x) : 
  (x < 1 → f x 1 < f 1 1) ∧ (1 < x → f 1 1 < f x 1) := 
sorry

-- Problem 2: The range of k
theorem range_of_k (x : ℝ) (h : 0 < x) (hx : f x k ≥ 2 + (1 - Real.e) / x) : 
  k ≥ 1 := 
sorry

-- Condition: The function g definition
def g (x : ℝ) (k : ℝ) : ℝ := f x k - k / x + 1

-- Problem 3: Prove k > g(x0)
theorem k_greater_than_gx0 (x1 x2 k : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  let x0 := (x1 + x2) / 2 
  in k > (f x0 k) - (k / x0) + 1 := 
sorry

end monotonic_intervals_range_of_k_k_greater_than_gx0_l31_31773


namespace proposition_1_valid_l31_31654

noncomputable def proposition_1_circle_line_intersection_chord_length : Prop :=
  let circle_eq : ∀ x y : ℝ, (x + 2)^2 + (y + 1)^2 = 4
  let line_eq : ∀ x y : ℝ, x - 2 * y = 0
  let chord_length : ℝ := 2
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ 
    circle_eq x₂ y₂ ∧ 
    line_eq x₁ y₁ ∧ 
    line_eq x₂ y₂ ∧ 
    (dist (x₁, y₁) (x₂, y₂) = chord_length)

theorem proposition_1_valid : proposition_1_circle_line_intersection_chord_length := 
by {
  -- Proof goes here.
  sorry 
}

end proposition_1_valid_l31_31654


namespace chocolate_difference_l31_31884

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l31_31884


namespace reciprocal_neg_2023_l31_31128

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l31_31128


namespace irrational_pi_l31_31655

theorem irrational_pi :
  (∀ r : ℝ, r = 0.33333333… → ¬ irrational r) ∧
  (∀ r : ℝ, r = 4 → ¬ irrational r) ∧
  (∀ r : ℝ, r = 22 / 7 → ¬ irrational r) →
  irrational π :=
begin
  sorry
end

end irrational_pi_l31_31655


namespace solve_trigonometric_equation_l31_31500

-- Definition of the problem
theorem solve_trigonometric_equation :
  ∀ x : ℝ, (sin x) ^ 4 + (cos x) ^ 4 = 2 * sin (2 * x) → 
  (∃ x₁ x₂ : ℝ, 
      (x₁ ≈ 13/180*π / 21 / 60*π / 19 / 3600*π ∧ x₂ ≈ 76/180*π / 38 / 60*π / 41 / 3600*π)) :=
by {
    -- Add necessary proof or conditions here
    sorry
}

end solve_trigonometric_equation_l31_31500


namespace total_animal_sightings_l31_31398

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l31_31398


namespace length_of_diagonal_EG_of_regular_octagon_l31_31711

theorem length_of_diagonal_EG_of_regular_octagon (a : ℝ) (h : a = 10) :
  let r := a
  let diagonal_length := 10 * real.sqrt (2 + real.sqrt 2)
  in ∃ EG : ℝ, EG = diagonal_length :=
sorry

end length_of_diagonal_EG_of_regular_octagon_l31_31711


namespace spring_festival_scientific_notation_l31_31255

noncomputable def scientific_notation := (260000000: ℝ) = (2.6 * 10^8)

theorem spring_festival_scientific_notation : scientific_notation :=
by
  -- proof logic goes here
  sorry

end spring_festival_scientific_notation_l31_31255


namespace nine_points_circle_chords_l31_31067

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31067


namespace number_of_chords_l31_31085

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31085


namespace find_x_solutions_l31_31291

theorem find_x_solutions :
  { x : ℝ | (9^x + 32^x) / (15^x + 24^x) = 4 / 3 } =
  { real.logb (3 / 2) (3 / 4), real.logb 3 4 } :=
sorry

end find_x_solutions_l31_31291


namespace mike_max_marks_l31_31587

theorem mike_max_marks
  (M : ℝ)
  (h1 : 0.30 * M = 234)
  (h2 : 234 = 212 + 22) : M = 780 := 
sorry

end mike_max_marks_l31_31587


namespace triangle_points_condition_l31_31363

theorem triangle_points_condition
  (α β γ k : ℝ)
  (inside_triangle : Type)
  (x y z : inside_triangle → ℝ)
  (M : inside_triangle) :
  (∀ M₁ M₂ : inside_triangle, 
    α * x M₁ + β * y M₁ + γ * z M₁ = k ∧
    α * x M₂ + β * y M₂ + γ * z M₂ = k → 
    ∀ t ∈ (set.Icc 0 1), α * x ((1 - t) * M₁ + t * M₂) + β * y ((1 - t) * M₁ + t * M₂) + γ * z ((1 - t) * M₁ + t * M₂) = k) →
  (∀ M₁ M₂ M₃ : inside_triangle,
    α * x M₁ + β * y M₁ + γ * z M₁ = k ∧
    α * x M₂ + β * y M₂ + γ * z M₂ = k ∧
    α * x M₃ + β * y M₃ + γ * z M₃ = k ∧ 
    ¬(collinear M₁ M₂ M₃) → 
    ∀ M : inside_triangle, α * x M + β * y M + γ * z M = k) →
  ∃ (S : set inside_triangle), 
    (S = ∅ ∨ 
     (∃ M₁ M₂, S = {M | ∃ t ∈ (set.Icc 0 1), M = (1 - t) * M₁ + t * M₂}) ∨ 
     (S = set.univ)) ∧ 
    ∀ M, M ∈ S ↔ α * x M + β * y M + γ * z M = k := 
sorry

end triangle_points_condition_l31_31363


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l31_31502

-- Define the first theorem
theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the second theorem
theorem solve_quadratic_2 (x : ℝ) : 25*x^2 - 36 = 0 ↔ x = 6/5 ∨ x = -6/5 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the third theorem
theorem solve_quadratic_3 (x : ℝ) : x^2 + 10*x + 21 = 0 ↔ x = -3 ∨ x = -7 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the fourth theorem
theorem solve_quadratic_4 (x : ℝ) : (x-3)^2 + 2*x*(x-3) = 0 ↔ x = 3 ∨ x = 1 := 
by {
  -- We assume this proof is provided
  sorry
}

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l31_31502


namespace isosceles_trapezoid_ratio_proof_l31_31015

-- Definitions and conditions
variables (A B C D N K L : Point)
variables (circle : Circle)
variables (AB CD : Line)
variables (isosceles_trapezoid : Quadrilateral)
variables (tangent_to_circle_A : Tangent circle A)
variables (tangent_to_circle_B : Tangent circle B)
variables (tangent_to_circle_C : Tangent circle C)
variables (tangent_to_circle_D : Tangent circle D)
variable (tangent_to_circle_AD : Tangent circle AD)
variable (tangent_point_N : Touches circle AD N)
variable (NC_touches_K : MeetAgain circle N C K)
variable (NB_touches_L : MeetAgain circle N B L)

-- Proving the statement
theorem isosceles_trapezoid_ratio_proof
  (h1 : isosceles_trapezoid A B C D)
  (h2 : AB.parallel CD)
  (h3 : TangentToCircleQuadrilateral A B C D circle)
  (h4 : tangent_point_N)
  (h5 : NC_touches_K)
  (h6 : NB_touches_L) :
  (|NB| / |BL|) + (|NC| / |CK|) = 10 :=
sorry

end isosceles_trapezoid_ratio_proof_l31_31015


namespace vectors_parallel_implies_fraction_l31_31371

theorem vectors_parallel_implies_fraction (α : ℝ) :
  let a := (Real.sin α, 3)
  let b := (Real.cos α, 1)
  (a.1 / b.1 = 3) → (Real.sin (2 * α) / (Real.cos α) ^ 2 = 6) :=
by
  sorry

end vectors_parallel_implies_fraction_l31_31371


namespace rectangle_count_in_region_l31_31378
-- Import the entire Mathlib library for necessary definitions and tools

-- Define the bounded region 
structure Region :=
  (x_max : ℕ)
  (y_max : ℕ → ℕ)

def bounded_region : Region :=
  { x_max := 4,
    y_max := λ x, if x < 4 then 3 * x else 12 }

-- Define the lattice points and rectangles within the region
def count_rectangles (region: Region) : ℕ :=
  ∑ x1 in range region.x_max,
    ∑ y1 in range (region.y_max x1 + 1),
      ∑ x2 in range (x1 + 1, region.x_max + 1),
        ∑ y2 in range (y1 + 1, region.y_max x2 + 1), 1

-- Statement to prove the equivalence
theorem rectangle_count_in_region : count_rectangles bounded_region = 62 :=
by sorry

end rectangle_count_in_region_l31_31378


namespace spherical_to_rectangular_l31_31676

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 15 ∧ θ = π / 4 ∧ φ = π / 6 →
  let x := ρ * sin φ * cos θ in
  let y := ρ * sin φ * sin θ in
  let z := ρ * cos φ in
  (x, y, z) = (15 * sqrt 2 / 4, 15 * sqrt 2 / 4, 15 * sqrt 3 / 2) :=
by
  intro ρ θ φ
  rintro ⟨rfl, rfl, rfl⟩
  simp [Real.sin_pi_over_six, Real.cos_pi_over_four, Real.cos_pi_over_six]
  sorry

end spherical_to_rectangular_l31_31676


namespace expression_positive_iff_intervals_l31_31683

theorem expression_positive_iff_intervals (x : ℝ) :
  (x + 1)*(x - 3) > 0 ↔ x ∈ (-∞, -1) ∪ (3, ∞) := by
  sorry

end expression_positive_iff_intervals_l31_31683


namespace opposite_of_neg_one_div_2023_l31_31937

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31937


namespace complex_exponentiation_l31_31741

noncomputable theory

open Complex

theorem complex_exponentiation (a b : ℝ) (i : ℂ) (h₁ : z1 = -1 + 3 * i)
    (h₂ : z2 = a + b * (i ^ 3)) (h₃ : z1 = z2) : b^a = -1/3 :=
by
  sorry

end complex_exponentiation_l31_31741


namespace solve_coin_problem_l31_31814

def coin_problem : Prop :=
  ∃ (x y z : ℕ), 
  1 * x + 2 * y + 5 * z = 71 ∧ 
  x = y ∧ 
  x + y + z = 31 ∧ 
  x = 12 ∧ 
  y = 12 ∧ 
  z = 7

theorem solve_coin_problem : coin_problem :=
  sorry

end solve_coin_problem_l31_31814


namespace parallelogram_area_l31_31224

variable (base height : ℝ) (tripled_area_factor original_area new_area : ℝ)

theorem parallelogram_area (h_base : base = 6) (h_height : height = 20)
    (h_tripled_area_factor : tripled_area_factor = 9)
    (h_original_area_calc : original_area = base * height)
    (h_new_area_calc : new_area = original_area * tripled_area_factor) :
    original_area = 120 ∧ tripled_area_factor = 9 ∧ new_area = 1080 := by
  sorry

end parallelogram_area_l31_31224


namespace greater_savings_on_hat_l31_31235

theorem greater_savings_on_hat (savings_shoes spent_shoes savings_hat sale_price_hat : ℝ) 
  (h1 : savings_shoes = 3.75)
  (h2 : spent_shoes = 42.25)
  (h3 : savings_hat = 1.80)
  (h4 : sale_price_hat = 18.20) :
  ((savings_hat / (sale_price_hat + savings_hat)) * 100) > ((savings_shoes / (spent_shoes + savings_shoes)) * 100) :=
by
  sorry

end greater_savings_on_hat_l31_31235


namespace evaluate_log_expression_l31_31700

theorem evaluate_log_expression :
  (log 2 96 / log 48 2 - log 3 162 / log 27 3 + log 2 192 / log 24 2) =
  (26 + 18 * log 2 3 + 2 * (log 2 3) ^ 2 - 4 * log 3 2) :=
by
  sorry

end evaluate_log_expression_l31_31700


namespace product_of_inserted_numbers_l31_31839

theorem product_of_inserted_numbers (a b : ℝ) (r : ℝ) 
  (h1 : a = 5 * r) 
  (h2 : b = 5 * r^2) 
  (h3 : 5 * r^3 = 40) 
: a * b = 200 := 
begin
  sorry
end

end product_of_inserted_numbers_l31_31839


namespace product_of_numbers_eq_zero_l31_31713

theorem product_of_numbers_eq_zero (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 1) 
  (h3 : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := 
by
  sorry

end product_of_numbers_eq_zero_l31_31713


namespace polynomial_roots_fraction_sum_l31_31673

theorem polynomial_roots_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 12) 
  (h2 : ab + ac + bc = 20) 
  (h3 : abc = 3) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 328 / 9 := 
by 
  sorry

end polynomial_roots_fraction_sum_l31_31673


namespace arcsin_inequality_l31_31350

-- Given condition: f(x) = ax^2 + bx + c
-- where a > 0 and f(x+1) = f(1-x)
variables {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Assuming a > 0
axiom pos_a : a > 0

-- Given symmetric condition
axiom sym_f : ∀ x : ℝ, f(x + 1) = f(1 - x)

-- The proof statement
theorem arcsin_inequality : 
  f (Real.arcsin (1 / 3)) > f (Real.arcsin (2 / 3)) := 
sorry

end arcsin_inequality_l31_31350


namespace intersection_point_is_neg3_l31_31849

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 9 * x + 15

theorem intersection_point_is_neg3 :
  ∃ a b : ℝ, (f a = b) ∧ (f b = a) ∧ (a, b) = (-3, -3) := sorry

end intersection_point_is_neg3_l31_31849


namespace isosceles_triangle_A1B1M_l31_31475

/-
We need to define the geometrical configurations and prove that 
triangle A1 B1 M is isosceles with the angle at vertex M equal to 120°.

- ABC is an arbitrary triangle.
- A1, B1, and C1 are vertices of equilateral triangles constructed on the sides BC, AC, and AB, respectively.
- A1 and A, B1 and B are on opposite sides of BC and AC, respectively.
- C1 and C are on the same side of AB.
- M is the center of triangle ABC1.
-/

variables {A B C M A1 B1 C1 : Type}

axiom eq_equilateral_triangle_BC_A1 {BC A1 : Type} :
  is_equilateral_triangle BC A1 -- axiom representing BC_A1 is an equilateral triangle

axiom eq_equilateral_triangle_AC_B1 {AC B1 : Type} :
  is_equilateral_triangle AC B1 -- axiom representing AC_B1 is an equilateral triangle

axiom eq_equilateral_triangle_AB_C1 {AB C1 : Type} :
  is_equilateral_triangle AB C1 -- axiom representing AB_C1 is an equilateral triangle

axiom opposite_side_BC_A1 : opposite_side BC A1 A
axiom opposite_side_AC_B1 : opposite_side AC B1 B
axiom same_side_AB_C1 : same_side AB C1 C

axiom is_centroid_M_ABC1 {M : Type} : is_centroid M ABC1 -- axiom representing M is the centroid of ABC1

theorem isosceles_triangle_A1B1M : is_isosceles A1 B1 M ∧ angle A1 M B1 = 120 := sorry

end isosceles_triangle_A1B1M_l31_31475


namespace largest_prime_divisor_to_test_l31_31986

theorem largest_prime_divisor_to_test
  (n : ℤ) (h₁ : n ≥ 500) (h₂ : n ≤ 550) : ∃ p : ℕ, prime p ∧ p ≤ 23 ∧ p = 23 :=
by
  sorry

end largest_prime_divisor_to_test_l31_31986


namespace incorrect_method_of_proving_locus_l31_31577

def is_on_locus (P : Point) (center : Point) (radius : Real) : Prop :=
  dist P center = radius

def satisfies_condition (P : Point) (center : Point) (radius : Real) : Prop :=
  dist P center = radius

theorem incorrect_method_of_proving_locus :
  let center : Point := arbitrary Point
  let r : Real := arbitrary Real

  let statementA : Prop := ∀ P : Point, is_on_locus P center r ↔ satisfies_condition P center r
  let statementB : Prop := ∀ P : Point, ¬satisfies_condition P center r → ¬is_on_locus P center r ∧ is_on_locus P center r → satisfies_condition P center r
  let statementC : Prop := ∀ P : Point, (satisfies_condition P center r → is_on_locus P center r) ∧ (is_on_locus P center r → satisfies_condition P center r)
  let statementD : Prop := ∀ P : Point, ¬is_on_locus P center r → ¬satisfies_condition P center r ∧ (satisfies_condition P center r → ¬is_on_locus P center r)
  let statementE : Prop := ∀ P : Point, (satisfies_condition P center r → is_on_locus P center r) ∧ (¬satisfies_condition P center r → ¬is_on_locus P center r)

  ¬statementD :=
begin
  sorry
end

end incorrect_method_of_proving_locus_l31_31577


namespace minimum_travel_time_l31_31898

noncomputable def min_travel_time (A B : ℝ × ℝ) (border : ℝ × ℝ) (speed_meadow speed_wasteland : ℝ)
    (d_A_border d_B_border : ℝ) (distance_AB : ℝ) : ℝ :=
  let min_time := fun (C : ℝ × ℝ) =>
    let AC := real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    let CB := real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    (AC / speed_meadow) + (CB / speed_wasteland)
  in
  min_time (0, 0) -- This would be replaced by an argmin procedure to find point C

-- Example fixed coordinates for simplicity
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (24, -4)
def border : ℝ × ℝ := (0, 0)
def speed_meadow : ℝ := 6
def speed_wasteland : ℝ := 3
def d_A_border : ℝ := 8
def d_B_border : ℝ := 4
def distance_AB : ℝ := 24

theorem minimum_travel_time : 
  min_travel_time A B border speed_meadow speed_wasteland d_A_border d_B_border distance_AB = 4.89 :=
by
  sorry

end minimum_travel_time_l31_31898


namespace at_least_twelve_mutual_friends_l31_31404

-- Let G be a graph with 42 vertices representing people in the club
variables (G : SimpleGraph (Fin 42))

-- Condition 1: The club has 42 people
-- Implicit in the setup with G having 42 vertices

-- Condition 2: Any two people have at least ten mutual friends
def mutual_friend_condition := ∀ (u v : Fin 42), u ≠ v → (G.commonNeighbors u v).card ≥ 10

-- Question: Prove that there are at least two people who have at least twelve mutual friends
theorem at_least_twelve_mutual_friends
  (h : mutual_friend_condition G) :
  ∃ u v : Fin 42, u ≠ v ∧ (G.commonNeighbors u v).card ≥ 12 :=
sorry

end at_least_twelve_mutual_friends_l31_31404


namespace bisect_CF_l31_31974

theorem bisect_CF {A B C D E G F H : Type} [AffinePlane Type] 
  (AB DC : Line) (T : Trapezoid A B C D)
  (M : Point) (E: Point) (line_M : M = (intersect (Line AC) (Line BD)) )
  (line_E: E = (intersect (Line M) (Line AD parallel_to Line AB)) )
  (line_G: G = (intersect (Line D) (Line AB parallel_to Line BC)) )
  (line_F: F = (intersect (Line E) (Line AB parallel_to Line BC)) )
  (line_H: H = (intersect (Line AC) (Line DG))) :
  CF bisects GH := sorry

end bisect_CF_l31_31974


namespace exists_line_through_exactly_two_points_l31_31811

theorem exists_line_through_exactly_two_points {n : ℕ} (h : 2 ≤ n) 
  (points : Fin n → ℝ × ℝ) 
  (not_collinear : ∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)) :
  ∃ (i j: Fin n), i ≠ j ∧ (∀ k : Fin n, k ≠ i → k ≠ j → ¬ collinear (points i) (points j) (points k)) :=
sorry

-- Helper function (need to be defined) to determine if three points are collinear
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop := 
  (p2.snd - p1.snd) * (p3.fst - p1.fst) = (p3.snd - p1.snd) * (p2.fst - p1.fst)

end exists_line_through_exactly_two_points_l31_31811


namespace sum_of_powers_l31_31449

theorem sum_of_powers (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) :
  ∑ i in Finset.range (n + 1), i^m = 
  (1 / (m + 1) : ℚ) * ((n + 1) * ∑ k in Finset.range m.succ, Nat.choose m k * n^k - 
  ∑ j in Finset.range (m - 1).succ, Nat.choose (m + 1) j * ∑ i in Finset.range (n + 1), i^j) := by
  sorry

end sum_of_powers_l31_31449


namespace choose_officers_count_l31_31408

-- Define the group of 8 people
def group : Finset (Fin 8) := Finset.univ

-- Define the problem statement in Lean
theorem choose_officers_count :
  (group.card) * (Finset.erase group (Fin 8).zero).card * (Finset.erase (Finset.erase group (Fin 8).zero) (Fin 8).succ).card = 336 := 
by
  -- We use card to represent the cardinality
  rw [Finset.card_univ, Finset.card_erase_of_mem, Finset.card_erase_of_mem, Finset.card_univ, Finset.card_univ]
  swap, exact (Finset.mem_univ (Fin 8).succ) 
  swap, exact (Finset.mem_univ (Fin 8).zero)
  sorry

end choose_officers_count_l31_31408


namespace dot_product_solve_for_x_l31_31785

theorem dot_product_solve_for_x (x : ℝ) : (let a := (1, -1 : ℝ × ℝ) in
                                          let b := (2, x : ℝ × ℝ) in
                                          a.1 * b.1 + a.2 * b.2 = 1) → x = 1 := 
by
  intros
  sorry

end dot_product_solve_for_x_l31_31785


namespace safe_numbers_count_10000_l31_31718

def is_p_safe (p n : ℕ) : Prop := 
  ∀ k : ℕ, (n ≠ k * p + 2) ∧ (n ≠ k * p + 1) ∧ (n ≠ k * p) ∧ (n ≠ k * p - 1) ∧ (n ≠ k * p - 2)

def count_safe (p1 p2 p3 N : ℕ) : ℕ := 
  (Finset.range (N + 1)).filter (λ n, is_p_safe p1 n ∧ is_p_safe p2 n ∧ is_p_safe p3 n).card

theorem safe_numbers_count_10000 :
  count_safe 7 11 13 10000 = 3196 :=
by {
  sorry
}

end safe_numbers_count_10000_l31_31718


namespace chocolate_difference_l31_31885

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l31_31885


namespace allowable_sandwich_combinations_l31_31519

-- Definitions based on the conditions:
def bread := {type1, type2, type3}
def meat := {turkey, roast_beef, meat3, meat4, meat5}
def cheese := {swiss, cheese2, cheese3, cheese4}
def valid_sandwich (b : bread) (m : meat) (c : cheese) : Prop :=
  ¬((m = turkey ∧ c = swiss) ∨ (m = roast_beef ∧ b = rye))

-- Statement to prove the final result
theorem allowable_sandwich_combinations : 
  (3 * 5 * 4) - (3 * 1) - (1 * 4) = 53 :=
sorry

end allowable_sandwich_combinations_l31_31519


namespace number_of_correct_statements_l31_31249

theorem number_of_correct_statements :
  let A_equal_to_obtuse : Prop := ∀ (A B : Type), ∀ (a b : A) (c : B), vector_dot_product a b c < 0 → angle_obtuse a b c
  let geometric_sequence_condition : Prop := ∀ (Sn : ℕ → ℝ), (∀ n, n ≥ 6 → (Sn 2, Sn 4 - Sn 2, Sn 6 - Sn 4) form_geometric_sequence) → false
  let sequence_sum : Prop := ∀ (an : ℕ → ℝ) (Sn : ℕ → ℝ), (∀ n, an n = 1 / (n * (n + 2))) → (∀ n, Sn n = (n + 1) / (2 * (n + 2))) → false
  let tan_sum_condition : Prop := ∀ (A B C : Type) (a : A) (b c : B), obtuse_triangle a b c → tan_sum a b c < 0
  in 2 = count_true [A_equal_to_obtuse, (¬ geometric_sequence_condition), (¬ sequence_sum), tan_sum_condition] sorry

end number_of_correct_statements_l31_31249


namespace eccentricity_of_hyperbola_l31_31758

theorem eccentricity_of_hyperbola (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2)
  (h2 : ∃ (k : ℝ), b = k * a)
  (h3 : k = sqrt 2) :
  let e := (sqrt (c^2 - a^2)) / a in
  e = (sqrt 6) / 2 := 
by
  sorry

end eccentricity_of_hyperbola_l31_31758


namespace positive_difference_prob_l31_31168

/-- Probability that the positive difference between two randomly chosen numbers from 
the set {1, 2, 3, 4, 5, 6, 7, 8} is 3 or greater -/
theorem positive_difference_prob :
  (let S := {1, 2, 3, 4, 5, 6, 7, 8}
       in (S.powerset.filter (λ s => s.card = 2)).card.filter (λ s => (s.to_list.head! - s.to_list.tail.head!).nat_abs >= 3).card /
           (S.powerset.filter (λ s => s.card = 2)).card = 15 / 28) := 
begin
  sorry
end

end positive_difference_prob_l31_31168


namespace sum_of_roots_eq_six_l31_31185

theorem sum_of_roots_eq_six (a b c : ℤ) (h : a = 1 ∧ b = -6 ∧ c = 8) :
  let sum_of_roots := -b / a in
  sum_of_roots = 6 := by
  sorry

end sum_of_roots_eq_six_l31_31185


namespace trapezium_area_l31_31706

theorem trapezium_area (a b h : ℕ) (Area : ℕ) 
  (ha : a = 26) (hb : b = 18) (hh : h = 15) : Area = 330 :=
  by
    have hsum : a + b = 44 := by rw [ha, hb]; exact rfl
    have A_calc : Area = (1 / 2) * (a + b) * h := by sorry
    have A_calc' : Area = 22 * 15 := by sorry
    exact rfl

end trapezium_area_l31_31706


namespace triangle_area_l31_31409

-- Conditions
def is_right_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A = 90 ∧ angle_B + angle_C = 90

def is_isosceles_right_triangle (A B C : Type) (angle_B angle_C : ℝ) : Prop :=
  angle_B = angle_C ∧ is_right_triangle A B C 90 angle_B angle_C

def hypotenuse_length (A C : Type) (length : ℝ) : Prop :=
  length = 10 * real.sqrt 2

def leg_length (x : ℝ) : Prop :=
  x = 10

-- Theorem to prove
theorem triangle_area (A B C : Type) (angle_B angle_C : ℝ) (AC_length : ℝ)
  (h1 : is_isosceles_right_triangle A B C angle_B angle_C)
  (h2 : hypotenuse_length A C AC_length)
  (h3 : leg_length 10):
  let AB := 10 in
  let BC := 10 in
  let area := (1/2) * AB * BC in
  area = 50 :=
by 
  sorry

end triangle_area_l31_31409


namespace proof_problem_l31_31724

theorem proof_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) :
  (a < 1 → b > 2) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y - 1 / (2 * x) - 2 / y = 3 / 2 → x + y ≥ 3) :=
by
  sorry

end proof_problem_l31_31724


namespace total_animal_sightings_l31_31399

def A_Jan := 26
def A_Feb := 3 * A_Jan
def A_Mar := A_Feb / 2

theorem total_animal_sightings : A_Jan + A_Feb + A_Mar = 143 := by
  sorry

end total_animal_sightings_l31_31399


namespace emphasis_on_key_parts_l31_31649

def company_focuses_on_core_business (core_business : ℕ) (total_business : ℕ) : Prop :=
  core_business = 20 * total_business / 100

def core_drives_majority (core_business : ℕ) (majority_business : ℕ) : Prop :=
  majority_business = 80 * (core_business + majority_business) / 100

def objective_improves_efficiency : Prop := true

theorem emphasis_on_key_parts 
  (core_business total_business : ℕ) 
  (h1 : company_focuses_on_core_business core_business total_business)
  (majority_business : ℕ)
  (h2 : core_drives_majority core_business majority_business)
  (h3 : objective_improves_efficiency)
  : Prop := 
  "Emphasis should be placed on the function of key parts."

end emphasis_on_key_parts_l31_31649


namespace football_club_balance_l31_31623

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l31_31623


namespace sum_fraction_to_limit_l31_31900

theorem sum_fraction_to_limit (n : ℕ) (hn : n > 0) :
  (∑ i in Finset.range n, (1 : ℚ) / (((2 * i + 1) : ℕ) * ((2 * i + 3) : ℕ))) = (n : ℚ) / (2 * n + 1) :=
sorry

end sum_fraction_to_limit_l31_31900


namespace range_of_f_minus_x_l31_31929

theorem range_of_f_minus_x (f : ℝ → ℝ)
  (h_f : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → f(x) ≥ x)
  (hf_values : ∀ x : ℝ, ∃ y : ℝ, (x, y) ∈ {(-4, -4), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4)} ∧ f(x) = y) :
  set.range (λ x, f(x) - x) = set.Icc 0 1 := sorry

end range_of_f_minus_x_l31_31929


namespace pedestrians_at_most_twice_collinear_l31_31540

-- Definition of vectors and related operations can be imported from Mathlib, if needed.

variable {a b c d : vector3}  -- Assuming vector3 is defined in Mathlib for 3-dimensional vectors.
variable (t : ℝ)  -- Time can be represented as real numbers.

-- Initial conditions
def initial_not_collinear (b d : vector3) : Prop := (b ∧ d) ≠ 0

-- Definition for relative position vectors
def v (t : ℝ) (a b : vector3) : vector3 := t • a + b
def w (t : ℝ) (c d : vector3) : vector3 := t • c + d

-- Definition for collinearity condition
def collinear_at_t (t : ℝ) (a b c d : vector3) : Prop := 
  (v t a b ∧ w t c d) = 0

-- Main theorem statement translating the mathematical problem
theorem pedestrians_at_most_twice_collinear (a b c d: vector3)
  (h_initial : initial_not_collinear b d) :
  ∃ at_most_two_times, at_most_two_times = ∀ t1 t2 : ℝ, 
    collinear_at_t t1 a b c d ∧ collinear_at_t t2 a b c d → (t1 = t2) ∨ (card ({t1, t2}) ≤ 2) :=
sorry

end pedestrians_at_most_twice_collinear_l31_31540


namespace five_lines_max_sections_l31_31831

theorem five_lines_max_sections (n : ℕ) (h : n = 5):
    max_sections n = 16 :=
sorry

end five_lines_max_sections_l31_31831


namespace probability_ge_3_l31_31162

open Set

def num_pairs (s : Set ℕ) (n : ℕ) : ℕ :=
  (s.Subset (Filter fun x y => abs (x - y) ≥ n)).Card

def undesirable_pairs : Set (ℕ × ℕ) :=
  {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), 
   (1,3), (2,4), (3,5), (4,6), (5,7), (6,8)}

def total_pairs : ℕ := choose 8 2

def prob_diff_ge_3 : ℚ :=
  1 - (undesirable_pairs.toList.length : ℚ) / total_pairs

theorem probability_ge_3 : prob_diff_ge_3 = 15 / 28 := by
  sorry

end probability_ge_3_l31_31162


namespace downhill_speed_is_correct_l31_31215

variable (V_d : ℝ)

noncomputable def uphill_speed : ℝ := 30 -- car's uphill speed in km/hr
noncomputable def downhill_distance : ℝ := 50 -- distance traveled downhill in km
noncomputable def uphill_distance : ℝ := 100 -- distance traveled uphill in km
noncomputable def average_speed : ℝ := 37.89 -- average speed in km/hr
noncomputable def total_distance : ℝ := uphill_distance + downhill_distance -- total distance traveled in km
noncomputable def T_uphill : ℝ := uphill_distance / uphill_speed -- time taken to travel uphill in hr
noncomputable def T_downhill : ℝ := downhill_distance / V_d -- time taken to travel downhill in hr

-- Main statement
theorem downhill_speed_is_correct :
  average_speed = total_distance / (T_uphill + T_downhill) →
  V_d ≈ 79.97 :=
sorry

end downhill_speed_is_correct_l31_31215


namespace remainder_div_x11_x15_l31_31022

-- Definitions and conditions based on the problem
def Q (x : ℚ) : ℚ

axiom div_x_15_r (x: ℚ) : Q 15 = 9
axiom div_x_11_r (x: ℚ) : Q 11 = 2

-- The theorem that we need to prove
theorem remainder_div_x11_x15 : ∃ r s : ℚ, (∀ x : ℚ, Q x = (x - 11) * (x - 15) * (M x) + r * x + s) ∧ 
                                              (r = 7 / 4) ∧ (s = -69 / 4) :=
by
  sorry

end remainder_div_x11_x15_l31_31022


namespace socks_cleaning_possible_l31_31851

-- Definitions based on the conditions
variables (k n r : ℕ) (hk : 0 < k) (hn : 0 < n) (hr : 0 < r) (h_r_lt_n : r < n)

-- The theorem that matches the problem statement
theorem socks_cleaning_possible (h : k * n + r) :
  (r ≥ k ∧ n > k + r) ↔ 
  ∃ (black white : List ℕ), 
    black.length = k * n + r ∧ 
    white.length = k * n + r ∧ 
    ∀ (ls : List ℕ), ls.length < 2 * n → 
    (∃ (blacks whites : ℕ), ls = List.replicate blacks 0 ++ List.replicate whites 1 ∧ blacks = n ∧ whites = n) → False :=
by
  sorry

end socks_cleaning_possible_l31_31851


namespace nine_points_chords_l31_31061

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31061


namespace math_problem_l31_31298

noncomputable def z_star (y : ℝ) : ℕ :=
  floor (y / 2) * 2
  
noncomputable def w_star (x : ℝ) : ℕ :=
  if even (ceil x + 1) then nat.ceil x + 2 else nat.ceil x + 1

theorem math_problem (x y : ℝ) (h1 : x = 3.25) (h2 : y = 12.5) :
  let zs := z_star y in
  let ws := w_star x in
  let result := (6.32 - zs) * (ws - x) in
  result = -9.94 :=
by
  sorry

end math_problem_l31_31298


namespace g_formula_l31_31731

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def g : ℕ → ℕ
| 0     := 3
| (n+1) := f (g n)

theorem g_formula (n : ℕ) : g n = 2^(n + 1) - 1 :=
sorry

end g_formula_l31_31731


namespace time_to_empty_tank_l31_31632

-- Definitions of initial conditions
def tank_capacity : ℝ := 5040
def leak_rate : ℝ := tank_capacity / 6
def inlet_rate : ℝ := 3.5 * 60

-- Theorem statement: Given the above definitions, prove the time to empty the tank is 8 hours
theorem time_to_empty_tank : tank_capacity / (leak_rate - inlet_rate) = 8 :=
by sorry

end time_to_empty_tank_l31_31632


namespace geometric_sequence_sum_l31_31417

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = a n * r) 
    (h1 : a 1 + a 2 = 40) 
    (h2 : a 3 + a 4 = 60) : 
    a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l31_31417


namespace cost_of_history_book_l31_31561

theorem cost_of_history_book (total_books : ℕ) (cost_math_book : ℕ) (total_price : ℕ) (num_math_books : ℕ) (num_history_books : ℕ) (cost_history_book : ℕ) 
    (h_books_total : total_books = 90)
    (h_cost_math : cost_math_book = 4)
    (h_total_price : total_price = 396)
    (h_num_math_books : num_math_books = 54)
    (h_num_total_books : num_math_books + num_history_books = total_books)
    (h_total_cost : num_math_books * cost_math_book + num_history_books * cost_history_book = total_price) : cost_history_book = 5 := by 
  sorry

end cost_of_history_book_l31_31561


namespace trigonometric_expression_evaluation_l31_31669

theorem trigonometric_expression_evaluation :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 3 / Real.cos (70 * Real.pi / 180) = -4 :=
by
  sorry

end trigonometric_expression_evaluation_l31_31669


namespace opposite_of_neg_one_over_2023_l31_31956

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31956


namespace table_max_height_l31_31836

noncomputable def heron (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def height (area : ℝ) (base : ℝ) : ℝ :=
  (2 * area) / base

theorem table_max_height :
  let DE := 25
  let EF := 28
  let FD := 31
  let area := heron DE EF FD
  let h_DE := height area DE
  let h_EF := height area EF
  let h_FD := height area FD
  42 * real.sqrt 77 = area → -- Area of triangle DEF using Heron's formula
  h_EF = (84 * real.sqrt 77) / 28 → -- Height from EF
  h_DE = (84 * real.sqrt 77) / 25 → -- Height from DE
  h_FD = (84 * real.sqrt 77) / 31 → -- Height from FD
  h' = (h_DE * h_EF) / (h_DE + h_EF) →
  h' = (4 * real.sqrt 77) / 53 :=
by
  sorry

end table_max_height_l31_31836


namespace ellipse_eccentricity_l31_31389

theorem ellipse_eccentricity (b : ℝ) (h_b : b = 1) (a : ℝ) (h_a : a = 2) : 
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = Real.sqrt 3 / 2 :=
by
  sorry

end ellipse_eccentricity_l31_31389


namespace total_weight_of_peppers_l31_31788

def green_peppers := 0.3333333333333333
def red_peppers := 0.4444444444444444
def yellow_peppers := 0.2222222222222222
def orange_peppers := 0.7777777777777778

theorem total_weight_of_peppers :
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 :=
by
  sorry

end total_weight_of_peppers_l31_31788


namespace sum_of_four_smallest_divisors_eq_11_l31_31293

noncomputable def common_divisors_sum : ℤ :=
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  let smallest_four := common_divisors.take 4
  smallest_four.sum

theorem sum_of_four_smallest_divisors_eq_11 :
  common_divisors_sum = 11 := by
  sorry

end sum_of_four_smallest_divisors_eq_11_l31_31293


namespace collinear_A_E_F_l31_31816

variables {O O' A B C D E F : Point}
variables {r R : ℝ}
variables [Circle O] [LineSegment O A B] [Perpendicular AOB OC]
variables [Circle O'] [TangentCircleO OB D O' E] [InternallyTangent O O' F]

theorem collinear_A_E_F (h1: Perpendicular AOB OC) 
                        (h2: TangentCircleO OB D O' E) 
                        (h3: InternallyTangent O O' F) : 
                        Collinear A E F :=
  sorry

end collinear_A_E_F_l31_31816


namespace james_heavy_lifting_time_l31_31841

/-- Suppose James experiences acute pain from an injury for 3 days.
    It requires 5 times as long as the acute pain period to heal fully.
    James waits an additional 3 days after healing before starting light exercises.
    The light exercise period is 2 weeks.
    The moderate exercise period after light exercises is 1 week.
    James must wait an additional 3 weeks after moderate exercise before starting heavy lifting.
    Prove that the total time before James can return to heavy lifting is 60 days. -/
theorem james_heavy_lifting_time : 
  let acute_pain := 3
  let healing_factor := 5
  let additional_rest := 3
  let light_exercise_weeks := 2
  let moderate_exercise_weeks := 1
  let final_wait_weeks := 3
  let total_time :=
    acute_pain * healing_factor + additional_rest
    + light_exercise_weeks * 7
    + moderate_exercise_weeks * 7
    + final_wait_weeks * 7
  in total_time = 60 :=
by
  sorry

end james_heavy_lifting_time_l31_31841


namespace angle_opposite_c_exceeds_l31_31396

theorem angle_opposite_c_exceeds (a b : ℝ) (c : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 2) (h_c : c >= 4) : 
  C >= 120 := 
sorry

end angle_opposite_c_exceeds_l31_31396


namespace problem_1_problem_2_problem_3_l31_31664

-- Problem (1)
theorem problem_1 : (choose 100 2 + choose 100 97) / perm 101 3 = 1 / 6 := 
sorry

-- Problem (2)
theorem problem_2 : ∑ k in (range 10).filter (fun x => x ≥ 3), choose k 3 = 330 := 
sorry

-- Problem (3)
theorem problem_3 (n m : ℕ) (h: m ≤ n) : (choose (n+1) m) / (choose n m) - (choose n (n-m+1)) / (choose n (n-m)) = 1 :=
sorry

end problem_1_problem_2_problem_3_l31_31664


namespace solid_circles_2006_l31_31239

noncomputable def circlePattern : Nat → Nat
| n => (2 + n * (n + 3)) / 2

theorem solid_circles_2006 :
  ∃ n, circlePattern n < 2006 ∧ circlePattern (n + 1) > 2006 ∧ n = 61 :=
by
  sorry

end solid_circles_2006_l31_31239


namespace geometric_sequence_sum_ratio_l31_31309

noncomputable def S (a₁ : ℝ) (n : ℕ) (q : ℝ) : ℝ :=
a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a₁ : ℝ) (n : ℕ) :
  let q := -5 in
  (S a₁ (n + 1) q) / (S a₁ n q) = -4 := 
by
  sorry

end geometric_sequence_sum_ratio_l31_31309


namespace sin_alpha_add_pi_over_2_l31_31330

noncomputable def P : ℝ × ℝ := (2, 1)

theorem sin_alpha_add_pi_over_2 :
  let α := real.arccos ((P.1 : ℝ) / real.sqrt (P.1 ^ 2 + P.2 ^ 2))
  sin (α + real.pi / 2) = 2 * real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_add_pi_over_2_l31_31330


namespace max_sections_with_5_lines_l31_31834

theorem max_sections_with_5_lines (n : ℕ) (h₀ : n = 0 → sections n = 1)
  (h₁ : n = 1 → sections n = 2) (h₂ : n = 2 → sections n = 4)
  (h₃ : n = 3 → sections n = 7) (h₄ : n = 4 → sections n = 11) :
  sections 5 = 16 :=
sorry

end max_sections_with_5_lines_l31_31834


namespace find_number_l31_31603

def single_digit (n : ℕ) : Prop := n < 10
def greater_than_zero (n : ℕ) : Prop := n > 0
def less_than_two (n : ℕ) : Prop := n < 2

theorem find_number (n : ℕ) : 
  single_digit n ∧ greater_than_zero n ∧ less_than_two n → n = 1 :=
by
  sorry

end find_number_l31_31603


namespace usual_time_to_catch_bus_l31_31560

variable (S T T_miss : ℝ)
variable (usual_speed : S)
variable (usual_time : T)
variable (missed_time : T_miss)

-- Condition 1: Missing the bus by 5 minutes walking with 3/5 of usual speed
axiom time_relation : T_miss = T + 5

-- Condition 2: Distance traveled is the same in both scenarios
axiom distance_relation : S * T = (3/5) * S * T_miss

theorem usual_time_to_catch_bus : usual_time = 7.5 :=
by
  -- These two axioms provided above should lead us to the proof that T = 7.5
  sorry

end usual_time_to_catch_bus_l31_31560


namespace identify_counterfeit_max_l31_31214

def maxCoinsIdentified (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem identify_counterfeit_max (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ n, f n = 2 * n^2 + 1) →
  ∀ m, (∃ k, k ≤ n ∧ m = f k) ↔ m = maxCoinsIdentified n :=
begin
  sorry
end

end identify_counterfeit_max_l31_31214


namespace number_of_chords_l31_31086

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31086


namespace sequence_sum_l31_31877

-- Definitions for the sequence and sum given conditions
def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else -(a n) + 1 - (1 / (2 ^ n))
def T_n (S : ℕ → ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (Finset.range n).sum S

-- Stating the theorem
theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n : ℕ, n ≠ 0 → S n = -a n + 1 - 1 / (2 ^ n)) →
  (∀ n : ℕ, n ≠ 0 → a n = (n : ℝ) / (2 ^ (n + 1))) →
  (∀ n : ℕ, T n = n - 2 + (n + 4) / (2 ^ (n + 1))) :=
by
  intros hS ha
  sorry

end sequence_sum_l31_31877


namespace teddy_pillows_correct_l31_31512

noncomputable def teddy_pillows : ℕ :=
  let fluffy_foam_per_pillow := 2
  let microfiber_per_pillow := 2.5
  let cotton_fabric_per_pillow := 1.75
  let total_fluffy_foam := 6000
  let total_microfiber := 4000
  let total_cotton_fabric := 3000
  let max_pillows_fluffy_foam := total_fluffy_foam / fluffy_foam_per_pillow
  let max_pillows_microfiber := total_microfiber / microfiber_per_pillow
  let max_pillows_cotton_fabric := total_cotton_fabric / cotton_fabric_per_pillow
  nat.floor (min (min max_pillows_fluffy_foam max_pillows_microfiber) max_pillows_cotton_fabric)

theorem teddy_pillows_correct : teddy_pillows = 1600 :=
by
  sorry

end teddy_pillows_correct_l31_31512


namespace range_of_a_l31_31761

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≥ 0 then exp x - a * x else exp (-x) - a * (-x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 ∨ (f a x = 0 → False)) →
  ( ∃ (x₁ x₂ x₃ x₄ : ℝ), f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  a ∈ set.Ioi (exp 1) :=
begin
  sorry
end

end range_of_a_l31_31761


namespace equation_1_solution_equation_2_solution_l31_31102

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end equation_1_solution_equation_2_solution_l31_31102


namespace ellipse_focus_point_sum_l31_31867

noncomputable def is_on_ellipse (P : Point) : Prop :=
  P.x^2 / 25 + P.y^2 / 16 = 1

noncomputable def is_focus (F₁ F₂ : Point) (a b : ℝ) : Prop :=
  F₁.x = -sqrt (a^2 - b^2) ∧ F₁.y = 0 ∧
  F₂.x = sqrt (a^2 - b^2) ∧ F₂.y = 0

theorem ellipse_focus_point_sum
  (P F₁ F₂ : Point)
  (hP : is_on_ellipse P)
  (hF : is_focus F₁ F₂ 5 (4 : ℝ)) :
  dist P F₁ + dist P F₂ = 10 :=
sorry

end ellipse_focus_point_sum_l31_31867


namespace minimum_tan_alpha_is_one_l31_31821

noncomputable def minimum_tan_alpha (k b : ℝ) (hk : k > 0) : ℝ :=
  let b_squared := (k^2 + 1) / 2
  let tan_alpha := b_squared / k
  tan_alpha

theorem minimum_tan_alpha_is_one {k b : ℝ} (hk : k > 0)
  (h_chord : (b^2 + k^2) = 2 * sqrt(2) * sqrt(k^2 + 1))
  (h_tan : b^2 = (k^2 + 1) / 2) :
  minimum_tan_alpha k b hk = 1 ∧ (k = 1) :=
sorry

end minimum_tan_alpha_is_one_l31_31821


namespace Aziz_parents_years_in_America_l31_31658

noncomputable def years_lived_in_America_before_Aziz : ℕ := 
  let birth_year := 2021 - 36  -- Aziz's birth year
  in birth_year - 1982  -- Years lived in America before Aziz's birth

theorem Aziz_parents_years_in_America :
  years_lived_in_America_before_Aziz = 3 :=
  by
    simp [years_lived_in_America_before_Aziz]
    sorry

end Aziz_parents_years_in_America_l31_31658


namespace solve_inequality_l31_31704

theorem solve_inequality (x : ℝ) : 
  1 / (x^2 + 2) > 4 / x + 21 / 10 ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end solve_inequality_l31_31704


namespace find_abc_l31_31288

theorem find_abc (a b c : ℝ) (t : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_le_one : a ≤ 1 ∧ b ≤ 1 ∧ c ≤ 1)
  (h_satisfy : 
    min (sqrt ((ab + 1) / (abc))) 
        (min (sqrt ((bc + 1) / (abc))) 
             (sqrt ((ac + 1) / (abc)))) 
    =
    sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c)) :
  (a = 1 / (-t^2 + t + 1) ∧ b = t ∧ c = 1 - t ∧ 1/2 ≤ t ∧ t < 1) ∨
  (perm (a, b, c) (1 / (-t^2 + t + 1), t, 1 - t) ∧ 1/2 ≤ t ∧ t < 1) :=
sorry

end find_abc_l31_31288


namespace sum_of_roots_eq_six_l31_31187

theorem sum_of_roots_eq_six (a b c : ℝ) (h_eq : a = 1 ∧ b = -6 ∧ c = 8) : 
  let sum_of_roots := -(b / a) in 
  sum_of_roots = 6 := 
by
  sorry

end sum_of_roots_eq_six_l31_31187


namespace correct_calculation_iff_l31_31575

theorem correct_calculation_iff :
  (sqrt 3 * sqrt 3 = 3) ∧
  ¬(3 * sqrt 3 - 2 * sqrt 3 = 1) ∧
  ¬(sqrt 27 + sqrt 3 = 9) ∧
  ¬(sqrt 3 + sqrt 3 = sqrt 6) := 
by
  sorry

end correct_calculation_iff_l31_31575


namespace reciprocal_of_neg_2023_l31_31133

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l31_31133


namespace polynomial_divisibility_l31_31722

theorem polynomial_divisibility (a : ℤ) : 
  (∀x : ℤ, x^2 - x + a ∣ x^13 + x + 94) → a = 2 := 
by 
  sorry

end polynomial_divisibility_l31_31722


namespace min_cos_y_plus_sin_x_l31_31936

theorem min_cos_y_plus_sin_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.cos x = Real.sin (3 * x))
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) - Real.cos (2 * x)) :
  ∃ (v : ℝ), v = -1 - Real.sqrt (2 + Real.sqrt 2) / 2 :=
sorry

end min_cos_y_plus_sin_x_l31_31936


namespace max_value_of_expr_l31_31917

open Classical
open Real

theorem max_value_of_expr 
  (x y : ℝ) 
  (h₁ : 0 < x) 
  (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  ∃ a b c d : ℝ, 
    (x^2 + 2 * x * y + 3 * y^2 = 20 + 10 * sqrt 3) ∧ 
    (a = 20) ∧ 
    (b = 10) ∧ 
    (c = 3) ∧ 
    (d = 2) := 
sorry

end max_value_of_expr_l31_31917


namespace weekly_tax_percentage_is_zero_l31_31878

variables (daily_expense : ℕ) (daily_revenue_fries : ℕ) (daily_revenue_poutine : ℕ) (weekly_net_income : ℕ)

def weekly_expense := daily_expense * 7
def weekly_revenue := daily_revenue_fries * 7 + daily_revenue_poutine * 7
def weekly_total_income := weekly_net_income + weekly_expense
def weekly_tax := weekly_total_income - weekly_revenue

theorem weekly_tax_percentage_is_zero
  (h1 : daily_expense = 10)
  (h2 : daily_revenue_fries = 12)
  (h3 : daily_revenue_poutine = 8)
  (h4 : weekly_net_income = 56) :
  weekly_tax = 0 :=
by sorry

end weekly_tax_percentage_is_zero_l31_31878


namespace number_of_chords_l31_31081

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31081


namespace courtyard_is_25_meters_long_l31_31612

noncomputable def courtyard_length (width : ℕ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) : ℝ :=
  let brick_area := brick_length * brick_width
  let total_area := num_bricks * brick_area
  total_area / width

theorem courtyard_is_25_meters_long (h_width : 16 = 16)
  (h_brick_length : 0.20 = 0.20)
  (h_brick_width: 0.10 = 0.10)
  (h_num_bricks: 20_000 = 20_000)
  (h_total_area: 20_000 * (0.20 * 0.10) = 400) :
  courtyard_length 16 0.20 0.10 20_000 = 25 := by
        sorry

end courtyard_is_25_meters_long_l31_31612


namespace opposite_of_neg_one_over_2023_l31_31960

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31960


namespace reciprocal_of_neg_2023_l31_31132

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l31_31132


namespace Berry_Temperature_Friday_l31_31256

theorem Berry_Temperature_Friday (temps : Fin 6 → ℝ) (avg_temp : ℝ) (total_days : ℕ) (friday_temp : ℝ) :
  temps 0 = 99.1 → 
  temps 1 = 98.2 →
  temps 2 = 98.7 →
  temps 3 = 99.3 →
  temps 4 = 99.8 →
  temps 5 = 98.9 →
  avg_temp = 99 →
  total_days = 7 →
  friday_temp = (avg_temp * total_days) - (temps 0 + temps 1 + temps 2 + temps 3 + temps 4 + temps 5) →
  friday_temp = 99 :=
by 
  intros h0 h1 h2 h3 h4 h5 h_avg h_days h_friday
  sorry

end Berry_Temperature_Friday_l31_31256


namespace minimum_stars_needed_l31_31138

theorem minimum_stars_needed 
  (sheets : ℕ)
  (stars_per_minute : ℕ) 
  (distinct : ℕ → Prop) 
  (h_sheets : sheets = 7)
  (h_stars_per_minute : stars_per_minute = 4) :
  (∃ m : ℕ, m = 28 ∧ ∀ n : ℕ, n ∈ {0, 1, 2, 3, 4, 5, 6, 7} → distinct n) 
    :=
sorry

end minimum_stars_needed_l31_31138


namespace min_value_quadratic_l31_31191

theorem min_value_quadratic (x : ℝ) : -2 * x^2 + 8 * x + 5 ≥ -2 * (2 - x)^2 + 13 :=
by
  sorry

end min_value_quadratic_l31_31191


namespace axis_of_symmetry_and_values_of_a_b_l31_31346

noncomputable def func (a b x : ℝ) : ℝ :=
  a * sin x * cos x - sqrt 3 * a * cos x ^ 2 + (sqrt 3 / 2) * a + b

theorem axis_of_symmetry_and_values_of_a_b (a b : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x : ℝ, 2 ≤ func a b x) (h₂ : ∀ x : ℝ, func a b x ≤ 4) :
  (∀ k : ℤ, ∃ x : ℝ, x = 5 * π / 12 + k * π / 2) ∧ a = 1 ∧ b = 3 :=
by
  sorry

end axis_of_symmetry_and_values_of_a_b_l31_31346


namespace points_symmetric_orthocenter_lie_on_circumcircle_l31_31485

open EuclideanGeometry

noncomputable def symmetric_point (H : Point) (P Q : Line) : Point := sorry

theorem points_symmetric_orthocenter_lie_on_circumcircle (ABC : Triangle) (H : Point) (A1 B1 C1 : Point)
  (hH_to_BC : A1 = symmetric_point H ABC.BC)
  (hH_to_CA : B1 = symmetric_point H ABC.CA)
  (hH_to_AB : C1 = symmetric_point H ABC.AB)
  (h_pepend1 : ⊥_on_line ABC.AB H ABC.CH)
  (h_pepend2 : ⊥_on_line ABC.BC H ABC.AH)
  (isosceles_AC1H : isosceles_triangle ABC.A H C1): 

  PointsOnCircumcircle A1 B1 C1 ABC := sorry

end points_symmetric_orthocenter_lie_on_circumcircle_l31_31485


namespace problem2008_l31_31440

theorem problem2008 
    (a b : ℕ) 
    (h_rel_prime : Nat.coprime a b) 
    (a_pos : a > 0) 
    (b_pos : b > 0) 
    (x : Fin 2007 → ℝ) 
    (h_nonneg : ∀ i, 0 ≤ x i) 
    (h_sum_pi : ∑ i, x i = Real.pi) :
    (∑ i, Real.sin (x i) ^ 2 = (Real.pi ^ 2) / 2007) → 
    a + b = 2008 :=
sorry

end problem2008_l31_31440


namespace canal_cross_section_area_l31_31114

theorem canal_cross_section_area
  (a b h : ℝ)
  (H1 : a = 12)
  (H2 : b = 8)
  (H3 : h = 84) :
  (1 / 2) * (a + b) * h = 840 :=
by
  rw [H1, H2, H3]
  sorry

end canal_cross_section_area_l31_31114


namespace finite_steps_iff_power_of_2_l31_31246

-- Define the conditions of the problem
def S (k n : ℕ) : ℕ := (k * (k + 1) / 2) % n

-- Define the predicate to check if the game finishes in finite number of steps
def game_completes (n : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i : ℕ, i < n → S (k + i) n ≠ S k n

-- The main statement to prove
theorem finite_steps_iff_power_of_2 (n : ℕ) : game_completes n ↔ ∃ t : ℕ, n = 2^t :=
sorry  -- Placeholder for the proof

end finite_steps_iff_power_of_2_l31_31246


namespace reduce_4128_over_4386_to_lowest_terms_l31_31202

noncomputable def reduced_fraction := Rat.mk 4128 4386

theorem reduce_4128_over_4386_to_lowest_terms : reduced_fraction = Rat.mk 295 313 := by
  -- Proof omitted; this statement asserts the equality of the two fractions.
  sorry

end reduce_4128_over_4386_to_lowest_terms_l31_31202


namespace non_empty_subsets_l31_31680

theorem non_empty_subsets :
  let S := { s : Set ℕ | s ≠ ∅ ∧ ∀ a ∈ s, ∀ b ∈ s, ∀ c ∈ s, abs (a - b) ≠ 1 ∧ abs (b - c) ≠ 1 ∧ abs (a - c) ≠ 2 } in
  let valid_subsets := { s : Set ℕ | ∃ k, s = { x ∈ {1, 2, 3, ..., 10} | x ≥ 2 * k } } in
  ∃ f : S → valid_subsets, 
  Finset.card (Finset.univ.filter (λ s, ∀ a ∈ s, ∀ b ∈ s, ∀ c ∈ s, abs(a - b) ≠ 1 ∧ abs(b - c) ≠ 1 ∧ abs(a - c) ≠ 2 ∧ ∃ k, ∀ x ∈ s, x ≥ 2 * k)) = 40 :=
begin
  sorry -- Proof goes here
end

end non_empty_subsets_l31_31680


namespace sum_of_roots_eq_six_l31_31183

theorem sum_of_roots_eq_six (a b c : ℤ) (h : a = 1 ∧ b = -6 ∧ c = 8) :
  let sum_of_roots := -b / a in
  sum_of_roots = 6 := by
  sorry

end sum_of_roots_eq_six_l31_31183


namespace expected_value_of_T_l31_31510

noncomputable def expected_adjacent_boy_girl_pairs (boys girls : ℕ) (arrangement : List (ℕ × ℕ)) : ℕ :=
  let total_adjacent_pairs := List.length arrangement
  let boy_girl_prob := (boys / total_adjacent_pairs.toFloat) * (girls / (total_adjacent_pairs - 1).toFloat)
  let girl_boy_prob := (girls / total_adjacent_pairs.toFloat) * (boys / (total_adjacent_pairs - 1).toFloat)
  (boy_girl_prob + girl_boy_prob) * total_adjacent_pairs

theorem expected_value_of_T :
  ∀ (boys girls : ℕ),
  boys = 10 →
  girls = 8 →
  ∀ (arrangement : List (ℕ × ℕ)),
  List.length arrangement = 18 →
  expected_adjacent_boy_girl_pairs boys girls arrangement = 9 :=
by
  intros boys girls hb hg arrangement hlen
  have : boys = 10 := hb
  have : girls = 8 := hg
  sorry

end expected_value_of_T_l31_31510


namespace coefficient_expansion_x2_is_60_l31_31521

noncomputable def coefficient_x2 := 
  (1 - (1 / 2) * x) * (1 + 2 * Real.sqrt x) ^ 5

theorem coefficient_expansion_x2_is_60 :
  polynomial.coeff (polynomial.expand (λ x, coefficient_x2) x^2) = 60 := 
sorry

end coefficient_expansion_x2_is_60_l31_31521


namespace code_DEG_value_l31_31614

-- Definitions from conditions
def base7_digits : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def decode_base7 (s : List Char) : Nat :=
  s.enum.zipWith (· * (7 ^ ·.fst)) (·.snd) 
  |>.sumBy (λ c, base7_digits.indexOf c.snd * (7 ^ c.fst))

-- Assertion to prove the problem statement
theorem code_DEG_value :
  decode_base7 ['D', 'E', 'G'] = 69 := by
  sorry

end code_DEG_value_l31_31614


namespace expected_socks_pairs_l31_31150

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l31_31150


namespace driving_time_to_beach_l31_31840

theorem driving_time_to_beach (total_trip_time : ℝ) (k : ℝ) (x : ℝ)
  (h1 : total_trip_time = 14)
  (h2 : k = 2.5)
  (h3 : total_trip_time = (2 * x) + (k * (2 * x))) :
  x = 2 := by 
  sorry

end driving_time_to_beach_l31_31840


namespace quadrilateral_probability_l31_31247

def total_shapes : ℕ := 6
def quadrilateral_shapes : ℕ := 3

theorem quadrilateral_probability : (quadrilateral_shapes : ℚ) / (total_shapes : ℚ) = 1 / 2 :=
by
  sorry

end quadrilateral_probability_l31_31247


namespace simplify_product_correct_l31_31497

noncomputable def simplify_product : Prop :=
  10 * (15 / 8) * (-28 / 45) * (3 / 5) = -7 / 4

theorem simplify_product_correct : simplify_product := 
begin
  sorry
end

end simplify_product_correct_l31_31497


namespace positive_difference_C_D_l31_31273

def C : ℕ :=
  (finset.range 20).sum (λ n, (2*n + 2) * (2*n + 3)) + 40

def D : ℕ :=
  (finset.range 19).sum (λ n, (2*n + 3) * (2*n + 4)) + 2

theorem positive_difference_C_D :
  |C - D| = 361 := by
  sorry

end positive_difference_C_D_l31_31273


namespace number_of_chords_number_of_chords_l31_31052

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31052


namespace alix_more_chocolates_than_nick_l31_31889

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l31_31889


namespace opposite_of_neg_one_div_2023_l31_31939

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31939


namespace sum_equals_two_l31_31871

-- Condition that a and b are positive integers
variables (a b : ℕ) (ha : 0 < a) (hb : 0 < b)

-- Definition of the general term
def f (k n : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2^n)

-- Define the sum S_{a,b}
def S (a b : ℕ) : ℚ :=
  (Finset.range (a + 1)).sum (λ i, f b (b + i)) +
  (Finset.range (b + 1)).sum (λ i, f a (a + i))

-- The theorem stating that the given sum equals 2
theorem sum_equals_two : S a b = 2 := by
  sorry

end sum_equals_two_l31_31871


namespace katy_books_l31_31848

theorem katy_books (june july aug : ℕ) (h1 : june = 8) (h2 : july = 2 * june) (h3 : june + july + aug = 37) :
  july - aug = 3 :=
by sorry

end katy_books_l31_31848


namespace multiple_of_6_is_multiple_of_3_l31_31967

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) : (∃ k : ℕ, n = 6 * k) → (∃ m : ℕ, n = 3 * m) :=
by
  sorry

end multiple_of_6_is_multiple_of_3_l31_31967


namespace altitudes_concur_acute_triangle_l31_31488

theorem altitudes_concur_acute_triangle (A B C : Type) [EuclideanGeometry A B C]
  (h_acute : is_acute (triangle A B C)) :
  ∃ H, is_orthocenter H (triangle A B C) :=
sorry

end altitudes_concur_acute_triangle_l31_31488


namespace books_sold_on_Monday_l31_31846

theorem books_sold_on_Monday :
  ∀ (total_stock : ℕ) (percent_unsold : ℝ) 
    (sold_tuesday sold_wednesday sold_thursday sold_friday : ℕ), 
    total_stock = 1400 →
    percent_unsold = 71.28571428571429 →
    sold_tuesday = 50 →
    sold_wednesday = 64 →
    sold_thursday = 78 →
    sold_friday = 135 →
    let unsold_books := total_stock * (percent_unsold / 100) in
    let books_sold_tuesday_to_friday := sold_tuesday + sold_wednesday + sold_thursday + sold_friday in
    let books_sold_on_monday := total_stock - unsold_books.to_nat - books_sold_tuesday_to_friday in
    books_sold_on_monday = 75 :=
begin
  intros,
  sorry
end

end books_sold_on_Monday_l31_31846


namespace expected_socks_pairs_l31_31149

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l31_31149


namespace largest_constant_k_l31_31708

theorem largest_constant_k {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (K : ℝ), (∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → 
    sqrt (ab/c) + sqrt (bc/a) + sqrt (ac/b) ≥ K * sqrt (a + b + c)) ↔ K = sqrt 3 := by
  sorry

end largest_constant_k_l31_31708


namespace pyramid_sum_height_width_l31_31110

def height : ℕ := 500 + 20
def width : ℕ := height + 234

theorem pyramid_sum_height_width : height + width = 1274 :=
by
  sorry

end pyramid_sum_height_width_l31_31110


namespace minimum_trucks_needed_l31_31907

theorem minimum_trucks_needed (total_weight : ℝ) (box_weight : ℕ → ℝ) 
  (n : ℕ) (H_total_weight : total_weight = 10) 
  (H_box_weight : ∀ i, box_weight i ≤ 1) 
  (truck_capacity : ℝ) 
  (H_truck_capacity : truck_capacity = 3) : 
  n = 5 :=
by {
  sorry
}

end minimum_trucks_needed_l31_31907


namespace max_sections_with_5_lines_l31_31826

theorem max_sections_with_5_lines : ∃ (n : ℕ), n = 16 ∧
  ∀ (rectangle : Type) (line_segment : Type) 
  (draw_lines : rectangle → line_segment → ℕ), 
  draw_lines (r : rectangle) (l : line_segment) = 5 → 
  sections_created_by_lines (r : rectangle) (l : line_segment) = 16 :=
begin
  sorry
end

end max_sections_with_5_lines_l31_31826


namespace bicycle_total_distance_l31_31921

noncomputable def front_wheel_circumference : ℚ := 4/3
noncomputable def rear_wheel_circumference : ℚ := 3/2
noncomputable def extra_revolutions : ℕ := 25

theorem bicycle_total_distance :
  (front_wheel_circumference * extra_revolutions + (rear_wheel_circumference * 
  ((front_wheel_circumference * extra_revolutions) / (rear_wheel_circumference - front_wheel_circumference))) = 300) := sorry

end bicycle_total_distance_l31_31921


namespace least_value_of_expression_l31_31179

theorem least_value_of_expression : ∃ (x y : ℝ), (2 * x - y + 3)^2 + (x + 2 * y - 1)^2 = 295 / 72 := sorry

end least_value_of_expression_l31_31179


namespace expected_socks_to_pair_l31_31141

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l31_31141


namespace opposite_neg_fraction_l31_31954

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31954


namespace find_added_value_l31_31393

theorem find_added_value (N : ℕ) (V : ℕ) (H : N = 1280) :
  ((N + V) / 125 = 7392 / 462) → V = 720 :=
by 
  sorry

end find_added_value_l31_31393


namespace regression_equation_error_probability_after_cloud_cost_increase_to_maintain_quality_l31_31666

-- Defining the given data
def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [692, 962, 1334, 2091, 3229]

-- Given calculated sums
def sum_ln_y : ℝ := 36.33
def sum_x_ln_y : ℝ := 112.85

-- Defined probabilities and error distributions
def initial_prob : ℝ := 0.6827
def initial_sigma_sq (m : ℝ) := 4 / m
def new_sigma_sq (m : ℝ) := 1 / m

-- Proof Statements

theorem regression_equation : 
  let x_avg := (x.sum) / (x.length : ℝ)
  let y_ln := y.map log
  let y_avg := (y_ln.sum) / (y.length : ℝ)
  let sum_x := x.sum
  let sum_x_sq := (x.map (λ xi, xi^2)).sum
  let n := (x.length : ℝ)
  let b := (sum_x_ln_y - (sum_x * y_avg)) / (sum_x_sq - (n * x_avg^2))
  let a := y_avg - b * x_avg
  (b = 0.386) ∧ (a = 6.108) ∧ (∀ xi, 1 ≤ xi → xi ≤ 5 → ⟨λ xi, exp(b * xi + a)⟩) := by
  sorry

theorem error_probability_after_cloud (m : ℝ) : 
  m = 4 → 
  let σ := sqrt (new_sigma_sq m)
  P(-1 < ε < 1) = 0.9545 := by
  sorry

theorem cost_increase_to_maintain_quality (initial_m : ℝ) (new_m : ℝ) : 
  initial_m = 4 →
  (sqrt (initial_sigma_sq initial_m)) = 1 → 
  (sqrt (new_sigma_sq new_m) = 1) → 
  new_m = 1 ∧ (initial_m - new_m = 3) := by
  sorry

end regression_equation_error_probability_after_cloud_cost_increase_to_maintain_quality_l31_31666


namespace west_1000_move_l31_31797

def eastMovement (d : Int) := d  -- east movement positive
def westMovement (d : Int) := -d -- west movement negative

theorem west_1000_move : westMovement 1000 = -1000 :=
  by
    sorry

end west_1000_move_l31_31797


namespace alix_has_15_more_chocolates_than_nick_l31_31882

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l31_31882


namespace present_age_of_B_l31_31806

theorem present_age_of_B :
  ∃ (A B : ℕ), (A + 20 = 2 * (B - 20)) ∧ (A = B + 10) ∧ (B = 70) :=
by
  sorry

end present_age_of_B_l31_31806


namespace sam_time_to_cover_distance_l31_31905

/-- Define the total distance between points A and B as the sum of distances from A to C and C to B -/
def distance_A_to_C : ℕ := 600
def distance_C_to_B : ℕ := 400
def speed_sam : ℕ := 50
def distance_A_to_B : ℕ := distance_A_to_C + distance_C_to_B

theorem sam_time_to_cover_distance :
  let time := distance_A_to_B / speed_sam
  time = 20 := 
by
  sorry

end sam_time_to_cover_distance_l31_31905


namespace sin_value_f_range_l31_31786

-- Definitions based on the conditions
def a (x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin (x + π / 6), -2)

def b (x : ℝ) : ℝ × ℝ :=
  (2, (Real.sqrt 3 / 2) - 2 * Real.cos x)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- First proof: Given a ⊥ b, prove sin(x + 4π/3) = -1/4
theorem sin_value (x : ℝ) (h : perpendicular (a x) (b x)) : 
  Real.sin (x + 4 * π / 3) = -1 / 4 :=
sorry

-- Definitions for the function f and its range
def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Second proof: Given x ∈ [0, π], prove range of f(x) is [-6 - sqrt 3, 3 sqrt 3]
theorem f_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) :
  -6 - Real.sqrt 3 ≤ f x ∧ f x ≤ 3 * Real.sqrt 3 :=
sorry

end sin_value_f_range_l31_31786


namespace raja_medicines_percentage_l31_31902

theorem raja_medicines_percentage : 
  let I := 37500
  let household_items := 0.35 * I
  let clothes := 0.20 * I
  let savings := 15000
  let medicines_amount := I - (household_items + clothes + savings)
  let M := medicines_amount / I
  M * 100 = 5 :=
by
  let I := 37500
  let household_items := 0.35 * I
  let clothes := 0.20 * I
  let savings := 15000
  let medicines_amount := I - (household_items + clothes + savings)
  let M := medicines_amount / I
  have hM : M = 0.05 := by sorry
  have hM_percentage : M * 100 = 5 := by
    rw [hM]
    norm_num
  exact hM_percentage

end raja_medicines_percentage_l31_31902


namespace find_a3_a4_a5_l31_31825

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 2 * a n

noncomputable def sum_first_three (a : ℕ → ℝ) : Prop :=
a 0 + a 1 + a 2 = 21

theorem find_a3_a4_a5 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : sum_first_three a) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end find_a3_a4_a5_l31_31825


namespace ribbons_left_l31_31468

theorem ribbons_left {initial_ribbons : ℕ} {morning_ribbons : ℕ} {afternoon_ribbons : ℕ} 
  (h_initial : initial_ribbons = 38) 
  (h_morning : morning_ribbons = 14) 
  (h_afternoon : afternoon_ribbons = 16) : 
  initial_ribbons - (morning_ribbons + afternoon_ribbons) = 8 := 
by 
  rw [h_initial, h_morning, h_afternoon]
  norm_num

end ribbons_left_l31_31468


namespace mary_pokemon_cards_l31_31034

theorem mary_pokemon_cards (initial_cards : ℕ) (torn_cards : ℕ) (sam_gift : ℕ) (alex_gift : ℕ) 
  (h_initial : initial_cards = 123) (h_torn : torn_cards = 18) (h_sam : sam_gift = 56) (h_alex : alex_gift = 35) : 
  initial_cards - torn_cards + sam_gift + alex_gift = 196 :=
  by
  rw [h_initial, h_torn, h_sam, h_alex]
  -- expected to use basic arithmetic next which is skipped here
  sorry

end mary_pokemon_cards_l31_31034


namespace student_avg_greater_actual_avg_l31_31238

theorem student_avg_greater_actual_avg
  (x y z : ℝ)
  (hxy : x < y)
  (hyz : y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end student_avg_greater_actual_avg_l31_31238


namespace opposite_of_neg_one_over_2023_l31_31955

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31955


namespace merchant_marked_price_l31_31633

theorem merchant_marked_price (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) (h1 : P = 0.7 * L) (h2 : S = 0.8 * M) (h3 : S - P = 0.3 * S) : M = 1.25 * L :=
  sorry

end merchant_marked_price_l31_31633


namespace penny_dime_same_probability_l31_31511

theorem penny_dime_same_probability : 
  let outcomes : ℕ := 2^4 in 
  let successful_outcomes : ℕ := 2^3 in 
  successful_outcomes / outcomes = 1 / 2 := 
by 
  sorry

end penny_dime_same_probability_l31_31511


namespace num_chords_l31_31042

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31042


namespace find_y_l31_31556

theorem find_y : ∃ y : ℝ, (7 / 3) * y = 42 ∧ y = 18 :=
by
  use 18
  split
  · norm_num
  · norm_num

end find_y_l31_31556


namespace find_a_max_perimeter_l31_31325

-- Define the constants and variables for sides and angles
variables (R : ℝ) (a b c : ℝ) (A B C : ℝ)

/-- Given that in triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively,
if 4 * sin A - b * sin B = c * sin (A - B). We need to find the value of a. -/
theorem find_a (h1 : 4 * Real.sin A - b * Real.sin B = c * Real.sin (A - B))
  (h2 : a = 2 * R * Real.sin A)
  (h3 : b = 2 * R * Real.sin B)
  (h4 : c = 2 * R * Real.sin C) :
  a = 4 :=
by {
  sorry
}

/-- If the area of triangle ABC is sqrt(3) * (b^2 + c^2 - a^2) / 4,
we need to find the maximum perimeter of triangle ABC. -/
theorem max_perimeter (S : ℝ) 
  (h1 : S = sqrt 3 * (b ^ 2 + c ^ 2 - a ^ 2) / 4)
  (h2 : a = 2 * R * Real.sin A)
  (h3 : b = 2 * R * Real.sin B)
  (h4 : c = 2 * R * Real.sin C)
  (h5 : b + c ≥ a):
  S = 12 :=
by {
  sorry
}

end find_a_max_perimeter_l31_31325


namespace problem_solver_l31_31804

def Problem :=
  ∃ (A B C D F : Type) (AC : ℝ),
  (D is_midpoint AC) ∧
  (is_perpendicular AB AC) ∧
  (is_perpendicular AF BC) ∧
  (BD = 2 ∧ DC = 2 ∧ FC = 2) ∧
  (AB = AF) →
  AC = 4

theorem problem_solver : Problem := sorry

end problem_solver_l31_31804


namespace distance_between_closest_points_l31_31262

theorem distance_between_closest_points
  (circle1_center : ℂ := (3, 3))
  (circle2_center : ℂ := (20, 15))
  (radius1 : ℝ := 3) 
  (radius2 : ℝ := 15) :
  let d_centers := complex.abs (circle2_center - circle1_center)
  in d_centers - (radius1 + radius2) = ℝ.sqrt 433 - 18 :=
by
  sorry

end distance_between_closest_points_l31_31262


namespace part1_domain_part1_odd_function_part2_c_and_a_l31_31726

variable (a c : ℝ)

def f (x : ℝ) := (x^2 + (3*a + 1)*x + c) / (x + a)

theorem part1_domain (a : ℝ) (h : a = 0) :
    ∀ x, x ≠ 0 → f x = x + c/x + 1 := by 
  sorry

theorem part1_odd_function (h : ∀ x, x ≠ 0 → f x = x + c/x + 1) : 
    ¬ ∃ c : ℝ, ∀ x, f (-x) = - f x := by 
  sorry

theorem part2_c_and_a (hx : f 1 = 3)
  (hI: ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ (x^2 + (3*a + 1)*x + 1 = 0)) :
    c = 1 ∧ (a > 1/3 ∧ a ≠ 1/2):= by 
  sorry

end part1_domain_part1_odd_function_part2_c_and_a_l31_31726


namespace log_func_increasing_iff_l31_31441

theorem log_func_increasing_iff (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (∀ x y : ℝ, x < y → (log a (3/4))^x < (log a (3/4))^y) ↔ (3 / 4 < a ∧ a < 1) :=
sorry

end log_func_increasing_iff_l31_31441


namespace zero_zero_solution_l31_31373

theorem zero_zero_solution (x y : ℝ) :
  x = 0 ∧ y = 0 → x^2 * (y + y^2) = y^3 + x^4 :=
by
  intros h
  cases h
  subst h_left
  subst h_right
  simp
  sorry

end zero_zero_solution_l31_31373


namespace complex_magnitude_l31_31443

theorem complex_magnitude (z w : ℂ) (hz : |z| = 1) (hw : |w| = 3) (hzw : |z + w| = 2) :
  ∣ (1/z + 1/w) ∣ = 2/3 :=
by
  sorry

end complex_magnitude_l31_31443


namespace rectangular_equation_of_curve_shortest_distance_point_l31_31820

noncomputable def curve_polar_eq : ℝ → ℝ := λ θ, 2 * Real.sin θ

def line_eq (x : ℝ) : ℝ := -Real.sqrt 3 * x + 5

def curve_rect_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

def distance_to_line (x y : ℝ) : ℝ := (Real.abs (-Real.sqrt 3 * x + y - 5)) / Real.sqrt (1 + 3)

theorem rectangular_equation_of_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : 
  ∃ x y, curve_rect_eq x y ∧ curve_polar_eq θ = Real.sqrt (x^2 + y^2) ∧ (2 * Real.sin θ = y) := sorry

theorem shortest_distance_point : ∃ (x y : ℝ), 
  curve_rect_eq x y ∧ distance_to_line x y = 
  distance_to_line (Real.sqrt(3)/2) (3/2) := sorry

end rectangular_equation_of_curve_shortest_distance_point_l31_31820


namespace opposite_of_neg_one_div_2023_l31_31942

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31942


namespace exists_ten_numbers_in_intervals_l31_31479

theorem exists_ten_numbers_in_intervals :
  ∃ (x : ℕ → ℝ), (∀ i, 0 ≤ x i ∧ x i < 1) ∧ 
    (∀ i, i ≥ 1 → x i < x (i - 1) % 1 / i) :=
begin
  sorry
end

end exists_ten_numbers_in_intervals_l31_31479


namespace wheel_radii_l31_31965

variable (r_front r_rear : ℝ)

-- Definitions of conditions
def radius_relation : Prop := r_rear = 2 * r_front
def front_circumference : ℝ := 2 * Real.pi * r_front + 5
def rear_circumference : ℝ := 4 * Real.pi * r_front - 5
def revolutions_relation : Prop := (1500 / front_circumference) = (1875 / rear_circumference)

-- Hypotheses based on conditions
def conditions (r_front r_rear : ℝ) :=
  radius_relation r_front r_rear ∧
  (revolutions_relation r_front)

-- Statement to be proven
theorem wheel_radii (r_front r_rear : ℝ) :
  conditions r_front r_rear →
  r_front = 15 / (2 * Real.pi) ∧ r_rear = 2 * (15 / (2 * Real.pi)) :=
sorry

end wheel_radii_l31_31965


namespace square_perimeter_ratio_l31_31517

theorem square_perimeter_ratio (x y : ℝ)
(h : (x / y) ^ 2 = 16 / 25) : (4 * x) / (4 * y) = 4 / 5 :=
by sorry

end square_perimeter_ratio_l31_31517


namespace chess_tournament_problem_l31_31403

variable (m : ℕ)

def num_women := m
def num_men := 3 * m
def total_players := num_women m + num_men m
def total_matches := total_players m * (total_players m - 1) / 2
def total_wins := 5 * (m * (4 * m - 1)) / 10

theorem chess_tournament_problem (m : ℕ) (h : num_women m + num_men m = 4 * m)
  (total_matches m = total_wins m) (h2 : (m * (4 * m - 1)) % 10 = 0) : m = 4 := by
  sorry

end chess_tournament_problem_l31_31403


namespace marble_weight_l31_31471

-- Define the conditions
def condition1 (m k : ℝ) : Prop := 9 * m = 5 * k
def condition2 (k : ℝ) : Prop := 4 * k = 120

-- Define the main goal, i.e., proving m = 50/3 given the conditions
theorem marble_weight (m k : ℝ) 
  (h1 : condition1 m k) 
  (h2 : condition2 k) : 
  m = 50 / 3 := by 
  sorry

end marble_weight_l31_31471


namespace false_proposition_B_l31_31096

def circle (a : ℝ) := { p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = a^2 }

def line_tangent (a : ℝ) : Prop :=
    let center := (a, 0) in
    let radius := |a| in
    let dist := (abs (1 + 3 : ℝ))/(sqrt(1 + (sqrt 3)^2)) in
    dist ≠ radius

theorem false_proposition_B (a : ℝ) (hA : |a| = 1) (hC : circle a (2, 0))
    (hD : ∀ x : ℝ, x - 1 = 0 → line_tangent a) :
    ¬ line_tangent a :=
by
  sorry

end false_proposition_B_l31_31096


namespace product_d_e_l31_31508

-- Define the problem: roots of the polynomial x^2 + x - 2
def roots_of_quadratic : Prop :=
  ∃ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0)

-- Define the condition that both roots are also roots of another polynomial
def roots_of_higher_poly (α β : ℚ) : Prop :=
  (α^7 - 7 * α^3 - 10 = 0 ) ∧ (β^7 - 7 * β^3 - 10 = 0)

-- The final proposition to prove
theorem product_d_e :
  ∀ α β: ℚ, (α^2 + α - 2 = 0) ∧ (β^2 + β - 2 = 0) → (α^7 - 7 * α^3 - 10 = 0) ∧ (β^7 - 7 * β^3 - 10 = 0) → 7 * 10 = 70 := 
by sorry

end product_d_e_l31_31508


namespace solution_set_for_inequality_l31_31793

theorem solution_set_for_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decreasing : ∀ ⦃x y⦄, 0 < x → x < y → f y < f x)
  (h_f_neg3 : f (-3) = 1) :
  { x | f x < 1 } = { x | x < -3 ∨ 3 < x } := 
by
  -- TODO: Prove this theorem
  sorry

end solution_set_for_inequality_l31_31793


namespace find_m_given_solution_set_l31_31763

theorem find_m_given_solution_set :
  (∀ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) 1 → x^2 + 3 * (1 : ℝ) * x - 4 < 0) → (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Ioo (-4 : ℝ) 1 → x^2 + 3m * x - 4 < 0) → m = 1) :=
by
  intros h1 h2
  have h3 : ∀ x, x ∈ Set.Ioo (-4 : ℝ) 1 → x^2 + 3 * (1 : ℝ) * x - 4 < 0 := h1
  have h4 : m = 1 := sorry
  exact h4

end find_m_given_solution_set_l31_31763


namespace football_club_balance_l31_31626

/-- A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. Prove that the final balance is $60 million. -/
theorem football_club_balance :
  let initial_balance := 100
  let income_from_sales := 10 * 2
  let expenditure_on_purchases := 15 * 4
  let final_balance := initial_balance + income_from_sales - expenditure_on_purchases
  final_balance = 60 :=
by
  simp only [initial_balance, income_from_sales, expenditure_on_purchases, final_balance]
  sorry

end football_club_balance_l31_31626


namespace range_of_g_l31_31276

def g (x : ℝ) : ℝ := sin x ^ 4 - 2 * sin x * cos x + cos x ^ 4

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ sin (2 * x) ∧ sin (2 * x) ≤ 1 →
  -1/2 ≤ g x ∧ g x ≤ 1 :=
sorry

end range_of_g_l31_31276


namespace triangle_identity_l31_31725

theorem triangle_identity (a b c : ℝ) (B: ℝ) (hB: B = 120) :
    a^2 + a * c + c^2 - b^2 = 0 :=
by
  sorry

end triangle_identity_l31_31725


namespace lance_more_pebbles_l31_31259

-- Given conditions
def candy_pebbles : ℕ := 4
def lance_pebbles : ℕ := 3 * candy_pebbles

-- Proof statement
theorem lance_more_pebbles : lance_pebbles - candy_pebbles = 8 :=
by
  sorry

end lance_more_pebbles_l31_31259


namespace num_axisymmetric_and_centrally_symmetric_shapes_l31_31248

-- Definitions of shapes and their symmetry properties
def is_axisymmetric (s : Type) : Prop := sorry
def is_centrally_symmetric (s : Type) : Prop := sorry

def isosceles_triangle : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry
def square : Type := sorry
def circle : Type := sorry

axiom isosceles_triangle_props :
  is_axisymmetric isosceles_triangle ∧ ¬ is_centrally_symmetric isosceles_triangle

axiom equilateral_triangle_props :
  is_axisymmetric equilateral_triangle ∧ ¬ is_centrally_symmetric equilateral_triangle

axiom rectangle_props :
  is_axisymmetric rectangle ∧ is_centrally_symmetric rectangle

axiom square_props :
  is_axisymmetric square ∧ is_centrally_symmetric square

axiom circle_props :
  is_axisymmetric circle ∧ is_centrally_symmetric circle

-- The theorem statement
theorem num_axisymmetric_and_centrally_symmetric_shapes :
  let shapes := [isosceles_triangle, equilateral_triangle, rectangle, square, circle] in
  (shapes.count (λ s, is_axisymmetric s ∧ is_centrally_symmetric s) = 3) :=
by {
  sorry
}

end num_axisymmetric_and_centrally_symmetric_shapes_l31_31248


namespace distance_and_midpoint_l31_31812

-- Define the points
def point_1 : ℝ × ℝ := (0, 0)
def point_2 : ℝ × ℝ := (12, -9)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Statement that needs to be proven
theorem distance_and_midpoint :
  distance point_1 point_2 = 15 ∧ midpoint point_1 point_2 = (6, -4.5) :=
by
  sorry

end distance_and_midpoint_l31_31812


namespace solve_inequality_l31_31932

-- Defining the function f with its domain and the given conditions
def f (x : ℝ) : ℝ := sorry

-- Defining the proposition to ensure function's symmetry about x = 2
axiom f_symmetry : ∀ x, f(x) + f(4 - x) = 2

-- Defining the domain
def domain := {x : ℝ | (x ∈ set.Ico (-1 : ℝ) 2) ∨ (x ∈ set.Icc 2 5)}

-- The inequality we need to solve
def inequality (x : ℝ) : Prop := f(x) - f(4 - x) > -2

-- The expected solution set
def solution_set := {x : ℝ | (x ∈ set.Ico (-1 : ℝ) (1/2)) ∨ (x ∈ set.Icc 2 5)}

-- The main theorem statement
theorem solve_inequality : ∀ x ∈ domain, inequality x → x ∈ solution_set := sorry

end solve_inequality_l31_31932


namespace smallest_value_N_div_a3_possible_values_a3_l31_31446

-- Given the conditions
def is_set_of_natural_numbers (a : List ℕ) : Prop :=
  a.length = 10 ∧ (∀ i j, i < j → a[i] < a[j])

def lcm_set (a : List ℕ) : ℕ :=
  a.foldl Nat.lcm 1

-- Proof statement for (a)
theorem smallest_value_N_div_a3 (a : List ℕ) (N : ℕ) 
  (h1 : is_set_of_natural_numbers a)
  (h2 : N = lcm_set a) :
  N / a[2] = 8 :=
sorry

-- Proof statement for (b)
theorem possible_values_a3 (a : List ℕ) (N : ℕ)
  (h1 : is_set_of_natural_numbers a)
  (h2 : N = lcm_set a) :
  (N / a[0] = 10) →
  (a[2] ∈ (Finset.range 1000).filter (λ x, x = 315 ∨ x = 630 ∨ x = 945)) :=
sorry

end smallest_value_N_div_a3_possible_values_a3_l31_31446


namespace rocket_speed_soaring_l31_31231

theorem rocket_speed_soaring
    (soar_time : ℕ)
    (plummet_time : ℕ)
    (plummet_distance : ℕ)
    (total_average_speed : ℕ)
    (soar_distance : ℕ)
    (total_time := soar_time + plummet_time)
    (total_distance := total_average_speed * total_time)
    (soar_speed := soar_distance / soar_time)
    (plummet_time_eq : plummet_time = 3)
    (soar_time_eq : soar_time = 12)
    (plummet_distance_eq : plummet_distance = 600)
    (total_average_speed_eq : total_average_speed = 160)
    (soar_distance_eq : soar_distance = total_distance - plummet_distance) :
    soar_speed = 150 := 
by
    rw [soar_time_eq, plummet_time_eq, plummet_distance_eq, total_average_speed_eq, soar_distance_eq]
    sorry

end rocket_speed_soaring_l31_31231


namespace prime_intersect_even_l31_31854

-- Definitions for prime numbers and even numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Sets P and Q
def P : Set ℕ := { n | is_prime n }
def Q : Set ℕ := { n | is_even n }

-- Proof statement
theorem prime_intersect_even : P ∩ Q = {2} :=
by
  sorry

end prime_intersect_even_l31_31854


namespace exists_zero_sum_row_l31_31808

noncomputable def row_sum (rows : List (List Int)) : List Int :=
  rows.foldl (λ acc row => List.zipWith (+) acc row) (List.repeat 0 (rows.head?.length.getD 0))

theorem exists_zero_sum_row (n : Int) (table : List (List Int)) (corrupted_table : List (List Int))
  (h_table : ∃ seqs, table = seqs ∧ (∀ row ∈ seqs, List.length row = n ∧ ∀ x ∈ row, x = 1 ∨ x = -1))
  (h_corrupt : ∃ z_seqs, corrupted_table = z_seqs ∧ (∀ row ∈ z_seqs, List.length row = n ∧ (∀ x ∈ row, x = 1 ∨ x = -1 ∨ x = 0))) :
  ∃ rows ⊆ corrupted_table, row_sum rows = List.repeat 0 n :=
  sorry

end exists_zero_sum_row_l31_31808


namespace problem_statement_l31_31318

-- Definitions corresponding to the problem conditions
def Point (R : Type) := prod R R

-- Given points A and B
def A : Point ℝ := (-3, 0)
def B : Point ℝ := (3, 0)
def F : Point ℝ := (5, 0)
def P : Point ℝ := (4, 2)

-- Line equation parameters and slopes condition
def lines_intersect_at_M (M : Point ℝ) :=
  let (x, y) := M in
  (y / (x + 3)) * (y / (x - 3)) = (16 / 9) ∧ x ≠ 3 ∧ x ≠ -3

-- Trajectory of point M (hyperbola equation)
def hyperbola_equation (x y : ℝ) := (x^2 / 9) - (y^2 / 16) = 1

-- The minimum value of |MP| + |MF|
def minimum_value_MP_plus_MF (M : Point ℝ) :=
  | dist M P + dist M F = sqrt 85 - 6

-- Full problem statement
theorem problem_statement :
  ∃ M : Point ℝ, lines_intersect_at_M M
  ∧ hyperbola_equation M.1 M.2
  ∧ minimum_value_MP_plus_MF M :=
sorry

end problem_statement_l31_31318


namespace alix_has_15_more_chocolates_than_nick_l31_31883

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end alix_has_15_more_chocolates_than_nick_l31_31883


namespace value_of_p_l31_31586

theorem value_of_p (m n p : ℝ) (h1 : m = 6 * n + 5) (h2 : m + 2 = 6 * (n + p) + 5) : p = 1 / 3 :=
by
  sorry

end value_of_p_l31_31586


namespace part_a_l31_31707

variable (x y z : ℝ)

def vector_field_A : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ :=
  λ p, (p.1 * p.2^2, p.1^2 * p.2, p.3^3)

def div_vector_field_A (p : ℝ × ℝ × ℝ) : ℝ :=
  (partial (λ a => a * (p.2)^2) p.1 1 0 0) + (partial (λ a => (p.1)^2 * a) p.2 0 1 0) + (partial (λ a => a^3) p.3 0 0 1)

theorem part_a : (div_vector_field_A (1, -1, 3)) = 29 :=
by
  unfold div_vector_field_A
  -- speculate the partial derivatives explicitly if needed
  sorry

end part_a_l31_31707


namespace expected_socks_to_pair_l31_31139

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l31_31139


namespace total_children_count_l31_31982

theorem total_children_count (boys girls : ℕ) (hb : boys = 40) (hg : girls = 77) : boys + girls = 117 := by
  sorry

end total_children_count_l31_31982


namespace circumcircles_tangent_l31_31652

variables {A B C E F H P l S T : Point}
variables {α β γ : Angle}

-- Define the conditions
axiom h1 : ∀ (A B C : Triangle), is_acute_angled A B C
axiom h2 : altitude B E A C
axiom h3 : altitude C F A B
axiom h4 : intersection_altitudes A B C E F = H
axiom h5 : perpendicular H (line_through E F)
axiom h6 : line_through A parallel_to (line_through B C) = l
axiom h7 : intersection_line_perpendicular_from H to E F at l = P
axiom h8 : angle_bisector_between_lines l (line_through H P) at B C = (S, T)

-- Declare the theorem
theorem circumcircles_tangent
  (hABC : Circle)
  (hPST : Circle)
  (tangent : Tangent hABC hPST) :
  tangent (circumcircle A B C) (circumcircle P S T) :=
sorry

end circumcircles_tangent_l31_31652


namespace tan_sum_eq_neg_one_cos_diff_eq_seven_sqrt_two_over_ten_l31_31723

theorem tan_sum_eq_neg_one (α β : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Mathlib.Real.tan α + Mathlib.Real.tan β = 5)
  (h6 : Mathlib.Real.tan α * Mathlib.Real.tan β = 6) :
  Mathlib.Real.tan (α + β) = -1 := 
by
  sorry

theorem cos_diff_eq_seven_sqrt_two_over_ten (α β : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Mathlib.Real.tan α + Mathlib.Real.tan β = 5)
  (h6 : Mathlib.Real.tan α * Mathlib.Real.tan β = 6) :
  Mathlib.Real.cos (α - β) = (7 * Mathlib.Real.sqrt 2) / 10 := 
by
  sorry

end tan_sum_eq_neg_one_cos_diff_eq_seven_sqrt_two_over_ten_l31_31723


namespace golden_flower_probability_l31_31968

/-- 
The probability of revealing at least one golden flower in three games of smashing three
golden eggs, where the probability of revealing a golden flower with each smash is 1/2
and the outcomes are independent, is 511/512.
-/
theorem golden_flower_probability :
  let p := (1 / 2 : ℚ) in
  let P_at_least_one := 1 - (1 - (3 / 8 + 3 / 8 + 1 / 8))^3 in
  P_at_least_one = 511 / 512 :=
by
  let p := (1 / 2 : ℚ)
  let P_at_least_one := 1 - (1 - (3 / 8 + 3 / 8 + 1 / 8))^3
  have h : P_at_least_one = 511 / 512 := by sorry
  exact h

end golden_flower_probability_l31_31968


namespace construct_equilateral_triangle_l31_31780

open EuclideanGeometry

-- Definitions of the three parallel lines and point A
variable (d₁ d₂ d₃ : Line) (A : Point)

-- Assumptions that the lines are parallel and A is on the middle line
axiom parallel_lines : Parallel d₁ d₂ ∧ Parallel d₂ d₃
axiom A_on_d₂ : OnLine A d₂

-- The proof statement: There exist points B and C on d₁ and d₃, respectively, such that triangle ABC is equilateral
theorem construct_equilateral_triangle :
  ∃ (B : Point) (C : Point), OnLine B d₁ ∧ OnLine C d₃ ∧ equilateral_triangle A B C := by
  sorry

end construct_equilateral_triangle_l31_31780


namespace add_to_37_eq_52_l31_31650

theorem add_to_37_eq_52 (x : ℕ) (h : 37 + x = 52) : x = 15 := by
  sorry

end add_to_37_eq_52_l31_31650


namespace stickers_distribution_l31_31374

theorem stickers_distribution : ∃ (n : ℕ), n = 126 ∧ 
    ∃ (f : Fin 5 → ℕ), (∑ i, f i) = 10 ∧ (∀ i, (1 ≤ f i)) :=
by
  use 126
  split
  . refl
  sorry

end stickers_distribution_l31_31374


namespace smallest_n_value_l31_31261

theorem smallest_n_value :
  let r := 10
      g := 8
      b := 9
      y := 12
      p := 18
  in ∃ n : ℕ,
  (r * x = g * y) ∧ (g * y = b * z) ∧ (b * z = y * w) ∧ (y * w = p * n) ∧ n = 20 :=
by
  sorry

end smallest_n_value_l31_31261


namespace monotonic_increasing_interval_l31_31924

theorem monotonic_increasing_interval (f : ℝ → ℝ) :
  (∀ x, deriv f x = x * (1 - x)) → (∀ x, 0 ≤ x ∧ x ≤ 1 → monotone_on f (Icc 0 1)) :=
by
  intro h_deriv
  sorry

end monotonic_increasing_interval_l31_31924


namespace number_of_qualified_days_l31_31880

def month_number_letters : List (Nat × Nat) :=
  [(1, 7), (2, 8), (3, 5), (4, 5), (5, 3), (6, 4), (7, 4), (8, 6), (9, 9), (10, 7), (11, 8), (12, 8)]

def days_in_month (is_leap_year : Bool) : Nat → Nat
  | 1  => 31
  | 2  => if is_leap_year then 29 else 28
  | 3  => 31
  | 4  => 30
  | 5  => 31
  | 6  => 30
  | 7  => 31
  | 8  => 31
  | 9  => 30
  | 10 => 31
  | 11 => 30
  | 12 => 31
  | _  => 0

theorem number_of_qualified_days (is_leap_year : Bool) : 
  (Σ (mn : month_number_letters), mn.snd > mn.fst) = 121 :=
by
sorry

end number_of_qualified_days_l31_31880


namespace remaining_paint_fraction_l31_31217

theorem remaining_paint_fraction :
  ∀ (initial_paint : ℝ) (half_usage : ℕ → ℝ → ℝ),
    initial_paint = 2 →
    half_usage 0 (2 : ℝ) = 1 →
    half_usage 1 (1 : ℝ) = 0.5 →
    half_usage 2 (0.5 : ℝ) = 0.25 →
    half_usage 3 (0.25 : ℝ) = (0.25 / initial_paint) := by
  sorry

end remaining_paint_fraction_l31_31217


namespace smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l31_31996

theorem smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 11 = 0) ∧ (num_even_digits n = 2) ∧ (num_odd_digits n = 2) ∧ (n = 1469) :=
by
  sorry

-- Auxiliary definitions for counting even and odd digits can be provided
def num_even_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 0)

def num_odd_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 1)

def to_digits (n : ℕ) : list ℕ :=
  if n < 10 then [n] else to_digits (n / 10) ++ [n % 10]

end smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l31_31996


namespace pyramid_height_l31_31209

theorem pyramid_height (α β S : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hS : 0 < S) :
  ∃ H : ℝ, H = sin β * sqrt (S * cos α / abs (cos (α + β))) :=
by sorry

end pyramid_height_l31_31209


namespace parabola_focus_point_distance_parabola_equation_line_through_focus_midpoint_l31_31775

theorem parabola_focus_point_distance (p m : ℝ) (h_p_pos : p > 0) 
    (h_point_on_parabola : (3, m) ∈ { (x, y) | y^2 = 2 * p * x })
    (h_distance_to_focus : sqrt ((3 - p / 2)^2 + m^2) = 4) :
    p = 2 := sorry

theorem parabola_equation : 
    (p : ℝ) (h_p : parabola_focus_point_distance p (h_p_pos := by sorry) 
    \( m_proof_parabola p \)) :
    parabola_eq_p := y^2 = 4 * x := sorry

theorem line_through_focus_midpoint (A B : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_focus : ∀ (x1 y1 x2 y2 : ℝ), y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ F = (1,0)) 
    (h_midpoint_coord : (A.snd + B.snd) / 2 = -1) :
    y = -2 * (x - 1) :=
    2 * x + y - 2 = 0 := sorry

end parabola_focus_point_distance_parabola_equation_line_through_focus_midpoint_l31_31775


namespace opposite_of_neg_one_over_2023_l31_31957

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31957


namespace alix_more_chocolates_than_nick_l31_31888

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l31_31888


namespace toothpick_250_stage_l31_31928

-- Define the arithmetic sequence for number of toothpicks at each stage
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

-- The proof statement for the 250th stage
theorem toothpick_250_stage : toothpicks 250 = 1001 :=
  by
  sorry

end toothpick_250_stage_l31_31928


namespace remainder_division_P_by_D_l31_31993

def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15
def D (x : ℝ) := 4 * x - 8

theorem remainder_division_P_by_D :
  let remainder := P 2 % D 2
  remainder = 31 :=
by
  -- Proof will be inserted here, but currently skipped
  sorry

end remainder_division_P_by_D_l31_31993


namespace radius_inner_circle_l31_31643

theorem radius_inner_circle (s : ℝ) (n : ℕ) (d : ℝ) (r : ℝ) :
  s = 4 ∧ n = 16 ∧ d = s / 4 ∧ ∀ k, k = d / 2 → r = (Real.sqrt (s^2 / 4 + k^2) - k) / 2 
  → r = Real.sqrt 4.25 / 2 :=
by
  sorry

end radius_inner_circle_l31_31643


namespace vertex_in_fourth_quadrant_l31_31391

noncomputable theory

def vertex_quadrant (f: ℝ → ℝ) (g: ℝ → ℝ) (cond: ∀ x : ℝ, g x ≠ 0): Prop :=
  ∃ x := - (k + 1)/2, (k < -1) ∧ (x > 0) ∧ (f x < 0)

theorem vertex_in_fourth_quadrant (k : ℝ) 
  (h : ∀ x : ℝ, x^2 - 2*x - k ≠ 0): 
  vertex_quadrant (λ x, x^2 + (k + 1)*x + k) (λ x, x^2 - 2*x - k) h :=
begin
  sorry
end

end vertex_in_fourth_quadrant_l31_31391


namespace max_sections_with_5_lines_l31_31828

theorem max_sections_with_5_lines : ∃ (n : ℕ), n = 16 ∧
  ∀ (rectangle : Type) (line_segment : Type) 
  (draw_lines : rectangle → line_segment → ℕ), 
  draw_lines (r : rectangle) (l : line_segment) = 5 → 
  sections_created_by_lines (r : rectangle) (l : line_segment) = 16 :=
begin
  sorry
end

end max_sections_with_5_lines_l31_31828


namespace khali_shovels_snow_l31_31437

theorem khali_shovels_snow :
  let section1_length := 30
  let section1_width := 3
  let section1_depth := 1
  let section2_length := 15
  let section2_width := 2
  let section2_depth := 0.5
  let volume1 := section1_length * section1_width * section1_depth
  let volume2 := section2_length * section2_width * section2_depth
  volume1 + volume2 = 105 :=
by 
  sorry

end khali_shovels_snow_l31_31437


namespace red_m_and_m_ratio_l31_31260

theorem red_m_and_m_ratio :
  ∀ (initial_green : ℕ) (initial_red : ℕ) (eaten_green : ℕ) (added_yellow : ℕ)
    (prob_green : ℚ),
  initial_green = 20 →
  initial_red = 20 →
  eaten_green = 12 →
  added_yellow = 14 →
  prob_green = 0.25 →
  let remaining_green := initial_green - eaten_green in
  let total_m_and_ms := remaining_green / prob_green in
  let remaining_red := total_m_and_ms - remaining_green - added_yellow in
  let eaten_red := initial_red - remaining_red in
  eaten_red / initial_red = 1 / 2 :=
by 
  intros initial_green initial_red eaten_green added_yellow prob_green 
         initial_green_eq initial_red_eq eaten_green_eq added_yellow_eq prob_green_eq 
         remaining_green total_m_and_ms remaining_red eaten_red,
  sorry

end red_m_and_m_ratio_l31_31260


namespace wall_width_l31_31160

theorem wall_width (bricks : ℝ) (brick_length brick_width brick_height wall_length wall_thickness wall_volume : ℝ) (h_bricks : bricks = 242.42424242424244) (h_brick_dims : brick_length = 25 ∧ brick_width = 11 ∧ brick_height = 6) (h_wall_dims : wall_length = 800 ∧ wall_thickness = 5) (h_wall_vol : wall_volume = 400000) :
  ∃ w : ℝ, w = 100 :=
by 
  -- Conditions and definitions
  have h1 : bricks = 242.42424242424244 := h_bricks,
  have h2 : brick_length = 25 := h_brick_dims.1,
  have h3 : brick_width = 11 := h_brick_dims.2.1,
  have h4 : brick_height = 6 := h_brick_dims.2.2,
  have h5 : wall_length = 800 := h_wall_dims.1,
  have h6 : wall_thickness = 5 := h_wall_dims.2,
  -- Volume calculations (brick and wall)
  have brick_volume : ℝ := brick_length * brick_width * brick_height,
  have wall_volume_calculated : ℝ := wall_length * wall_thickness * 100, -- assuming width is 100
  have wall_volume_correct : wall_volume_calculated = wall_volume,
  -- Deriving the width
  use (wall_volume / (wall_length * wall_thickness)),
  -- Concluding width is 100 cm
  have brick_total_volume : ℝ := bricks * brick_volume,
  have : wall_volume = brick_total_volume,
  sorry

end wall_width_l31_31160


namespace opposite_of_neg_one_div_2023_l31_31938

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31938


namespace quadratic_factorization_l31_31736

-- Given condition definitions
def quadratic_eq (p q : ℝ) (x : ℝ) : Prop :=
  2 * x ^ 2 + p * x + q = 0

def root1 : ℝ := -2
def root2 : ℝ := 3 / 2

-- Lean statement for the proof problem
theorem quadratic_factorization (p q : ℝ) :
  quadratic_eq p q x → 
  (x + 2) * (2 * x - 3) = 0 :=
sorry

end quadratic_factorization_l31_31736


namespace angle_MKR_60_l31_31805

/- Define a triangle with specified angles and properties, and prove that angle MKR is 60 degrees. -/
theorem angle_MKR_60 
  (P Q R K M : Type) 
  (angle_P : angle P = 120)
  (angle_Q : angle Q = 30)
  (angle_R : angle R = 30)
  (is_altitude_PK : is_altitude PK)
  (is_median_QM : is_median QM):
  angle MKR = 60 := sorry

end angle_MKR_60_l31_31805


namespace abs_tan_45_eq_sqrt3_factor_4x2_36_l31_31211

theorem abs_tan_45_eq_sqrt3 : abs (1 - Real.sqrt 3) + Real.tan (Real.pi / 4) = Real.sqrt 3 := 
by 
  sorry

theorem factor_4x2_36 (x : ℝ) : 4 * x ^ 2 - 36 = 4 * (x + 3) * (x - 3) := 
by 
  sorry

end abs_tan_45_eq_sqrt3_factor_4x2_36_l31_31211


namespace probability_intersection_interval_l31_31124

theorem probability_intersection_interval (PA PB p : ℝ) (hPA : PA = 5 / 6) (hPB : PB = 3 / 4) :
  0 ≤ p ∧ p ≤ 3 / 4 :=
sorry

end probability_intersection_interval_l31_31124


namespace range_of_function_cos2_sinx_cosx_l31_31966

theorem range_of_function_cos2_sinx_cosx :
  (∀ x ∈ set.Icc (-π / 6) (π / 4), 
    ∃ y, y = cos x ^ 2 + (sqrt 3) * (sin x) * (cos x) ∧ y ∈ set.Icc 0 ((sqrt 3 + 1) / 2)) :=
sorry

end range_of_function_cos2_sinx_cosx_l31_31966


namespace Miss_Adamson_paper_usage_l31_31466

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l31_31466


namespace alcohol_quantity_l31_31635

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 2 / 5) (h2 : A / (W + 10) = 2 / 7) : A = 10 :=
by
  sorry

end alcohol_quantity_l31_31635


namespace area_of_triangle_l31_31665

noncomputable def proof_problem : ℝ :=
  let r : ℝ := 5
  let a := 300
  let b := 3700 / 9
  let q1q2 := r * (2 + (sqrt 370) / 3)
  let area :=
    (sqrt 3 / 4) * (q1q2 ^ 2)
  sqrt a + sqrt b

theorem area_of_triangle :
  let a := 300
  let b := 3700 / 9
  ∃ (Q_1 Q_2 Q_3 : ℝ), sqrt a + sqrt b = proof_problem :=
by
  sorry

end area_of_triangle_l31_31665


namespace no_real_roots_x_squared_ax_b_plus_2_l31_31365

theorem no_real_roots_x_squared_ax_b_plus_2
  (a b : ℤ)
  (h₁ : ∃ k₁ : ℤ, k₁ * k₁ = a^2 - 4*b)
  (h₂ : ∃ k₂ : ℤ, k₂ * k₂ = a^2 - 4*(b+1))
  (h₃ : ∀ k₁ : ℤ, k₁ * k₁ = a^2 - 4*b → k₁ ∈ ℤ)
  (h₄ : ∀ k₂ : ℤ, k₂ * k₂ = a^2 - 4*(b+1) → k₂ ∈ ℤ) :
  ∀ (a b : ℤ), ¬∃ x : ℝ, x^2 + (a : ℝ) * x + (b + 2 : ℤ) = 0 :=
begin
  sorry -- Proof required here
end

end no_real_roots_x_squared_ax_b_plus_2_l31_31365


namespace smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l31_31994

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l31_31994


namespace ratio_DE_EF_l31_31422

-- Define the points and segments ratios
variables {A B C D E F : Type} [vector_space Type V]
variables (a b c d e f : V)
variables (AD_DB BE_EC : ℚ)

-- Conditions
def AD_DB := (3 : ℚ) / 5
def BE_EC := (3 : ℚ) / 5

-- Define the situation in the problem
variables {D_on_AB : d = (3 : ℚ / 5) • a + (2 : ℚ / 5) • b}
variables {E_on_BC : e = (3 : ℚ / 5) • b + (2 : ℚ / 5) • c}
variables {F_intersect : ∃ f, f = 3 • e - 2 • d ∧ f = e + ½ • (c - e)}

-- The proof problem
theorem ratio_DE_EF : AD_DB = 3 / 5 → BE_EC = 3 / 5 → 
  (∃ D_on_AB E_on_BC F_intersect), DE / EF = 1 / 2 :=
  begin
    sorry  -- Proof steps to be filled in
  end

end ratio_DE_EF_l31_31422


namespace arccos_cos_eq_l31_31258

theorem arccos_cos_eq (a : ℝ) (h : a = 11) : Real.arccos (Real.cos a) = 11 - 4 * Real.pi :=
by
  rw h
  -- we should manipulate the period of cosine and arccos logic here
  -- skipped proof as instructed
  sorry

end arccos_cos_eq_l31_31258


namespace elements_in_M_l31_31117

def is_element_of_M (x y : ℕ) : Prop :=
  x + y ≤ 1

def M : Set (ℕ × ℕ) :=
  {p | is_element_of_M p.fst p.snd}

theorem elements_in_M :
  M = { (0,0), (0,1), (1,0) } :=
by
  -- Proof would go here
  sorry

end elements_in_M_l31_31117


namespace find_remainder_remainder_is_five_l31_31089

theorem find_remainder (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 95) (h2 : divisor = 15) (h3 : quotient = 6) : 
  remainder = dividend - divisor * quotient := by
  -- proofs can be filled in here
  sorry

theorem remainder_is_five : find_remainder 95 15 6 5 := by
  sorry

end find_remainder_remainder_is_five_l31_31089


namespace number_of_identical_domain_and_range_l31_31337

-- Define the four functions
def f1 (x : ℝ) : ℝ := 1 - x
def f2 (x : ℝ) : ℝ := 2x - 1
def f3 (x : ℝ) : ℝ := x^2 - 1
def f4 (x : ℝ) : ℝ := 5 / x

-- Define predicates for domain and range of being identical
def domain_range_identical (f : ℝ → ℝ) : Prop :=
  (∀ y, ∃ x, f x = y) ∧ (∀ y, ∃ x, f x = y)
-- Note: In Lean, 'domain_range_identical' means that every real number y can be achieved by some x, which speaks to both domain and range in simple ℝ → ℝ functions.

-- The final theorem statement
theorem number_of_identical_domain_and_range : 3 = 
  (if domain_range_identical f1 then 1 else 0) +
  (if domain_range_identical f2 then 1 else 0) +
  (if domain_range_identical f3 then 1 else 0) +
  (if domain_range_identical f4 then 1 else 0) :=
by
  -- Using this placeholder to allow the code to compile
  sorry

end number_of_identical_domain_and_range_l31_31337


namespace matchsticks_left_l31_31696

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l31_31696


namespace wendy_total_profit_l31_31990

theorem wendy_total_profit:
  let day1_morning_apples_profit := 40 * 1.50 * 0.20,
      day1_morning_oranges_profit := 30 * 1 * 0.25,
      day1_morning_bananas_profit := 10 * 0.75 * 0.15,
      day1_afternoon_apples_profit := 50 * 1.40 * 0.20,
      day1_afternoon_oranges_profit := 40 * 0.95 * 0.25,
      day1_afternoon_bananas_profit := 20 * 0.60 * 0.15,
      day2_morning_apples_profit := 40 * 1.45 * 0.18,
      day2_morning_oranges_profit := 30 * 1 * 0.23,
      day2_morning_bananas_profit := 10 * 0.70 * 0.14,
      day2_afternoon_apples_profit := 50 * 1.30 * 0.15,
      day2_afternoon_oranges_profit := 40 * 0.90 * 0.20,
      day2_afternoon_bananas_profit := 20 * 0.50 * 0.10,
      unsold_bananas_profit := 20 * 0.75 * 0.50 * 0.15,
      unsold_oranges_profit := 10 * 1 * 0.30 * 0.25,
      day1_total_profit := day1_morning_apples_profit + day1_morning_oranges_profit + day1_morning_bananas_profit + day1_afternoon_apples_profit + day1_afternoon_oranges_profit + day1_afternoon_bananas_profit,
      day2_total_profit := day2_morning_apples_profit + day2_morning_oranges_profit + day2_morning_bananas_profit + day2_afternoon_apples_profit + day2_afternoon_oranges_profit + day2_afternoon_bananas_profit,
      total_unsold_profit := unsold_bananas_profit + unsold_oranges_profit,
      total_profit := day1_total_profit + day2_total_profit + total_unsold_profit
  in
  total_profit = 84.07 :=
sorry

end wendy_total_profit_l31_31990


namespace sue_driving_days_l31_31915

-- Define the conditions as constants or variables
def total_cost : ℕ := 2100
def sue_payment : ℕ := 900
def sister_days : ℕ := 4
def total_days_in_week : ℕ := 7

-- Prove that the number of days Sue drives the car (x) equals 3
theorem sue_driving_days : ∃ x : ℕ, x = 3 ∧ sue_payment * sister_days = x * (total_cost - sue_payment) := 
by
  sorry

end sue_driving_days_l31_31915


namespace complex_number_quadrant_l31_31274

theorem complex_number_quadrant (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let z := Complex.ofReal (Real.cos (3 * Real.pi / 2 - θ)) + Complex.I * Complex.ofReal (Real.sin (π + θ))
  Real.Re z < 0 ∧ Real.Im z < 0 :=
sorry

end complex_number_quadrant_l31_31274


namespace sequences_and_sum_l31_31764

theorem sequences_and_sum (S : ℕ → ℕ) (a b : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - 2)
  (h2 : b 3 = a 2)
  (h3 : b 2 + b 6 = 10) :
  (∀ n, a n = 2^n) ∧
  (∀ n, b n = n + 1) ∧
  (∀ n, (∑ k in range n, a k * (2 * b k - 3)) = (2 * n - 3) * 2^(n+1) + 6) :=
  sorry

end sequences_and_sum_l31_31764


namespace second_and_third_finish_job_together_in_8_days_l31_31541

theorem second_and_third_finish_job_together_in_8_days
  (x y : ℕ)
  (h1 : 1/24 + 1/x + 1/y = 1/6) :
  1/x + 1/y = 1/8 :=
by sorry

end second_and_third_finish_job_together_in_8_days_l31_31541


namespace minimize_divided_equilateral_triangle_area_l31_31803

theorem minimize_divided_equilateral_triangle_area (a : ℝ) (hx : a > 0) :
  (∃ x : ℝ, x = a / 2 ∧ (∀ y : ℝ, y ≠ a / 2 → inner_triangle_area a y > inner_triangle_area a (a / 2))) :=
sorry

noncomputable def inner_triangle_area (a : ℝ) (x : ℝ) : ℝ :=
  (a^2 * real.sqrt 3 / 4) - (3 * real.sqrt 3 / 4) * x * (a - x)

end minimize_divided_equilateral_triangle_area_l31_31803


namespace equidistant_points_on_line_quadrants_l31_31362

theorem equidistant_points_on_line_quadrants :
  ∃ (p : ℝ × ℝ), (4 * p.1 + 3 * p.2 = 12) ∧ (abs p.1 = abs p.2) ∧
    ((p.1 > 0 ∧ p.2 > 0) ∨ (p.1 > 0 ∧ p.2 < 0)) ∧ ¬(p.1 < 0 ∧ p.2 < 0) ∧ ¬(p.1 < 0 ∧ p.2 > 0) := 
begin
  sorry
end

end equidistant_points_on_line_quadrants_l31_31362


namespace smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l31_31997

theorem smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 11 = 0) ∧ (num_even_digits n = 2) ∧ (num_odd_digits n = 2) ∧ (n = 1469) :=
by
  sorry

-- Auxiliary definitions for counting even and odd digits can be provided
def num_even_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 0)

def num_odd_digits (n : ℕ) : ℕ :=
  (to_digits n).count (λ d, d % 2 = 1)

def to_digits (n : ℕ) : list ℕ :=
  if n < 10 then [n] else to_digits (n / 10) ++ [n % 10]

end smallest_four_digit_divisible_by_11_with_two_even_two_odd_digits_l31_31997


namespace ratio_of_areas_l31_31991

theorem ratio_of_areas (side_length : ℝ) (h : side_length = 6) :
  let area_triangle := (side_length^2 * Real.sqrt 3) / 4
  let area_square := side_length^2
  (area_triangle / area_square) = Real.sqrt 3 / 4 :=
by
  sorry

end ratio_of_areas_l31_31991


namespace final_balance_is_60_million_l31_31618

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l31_31618


namespace sum_possible_values_of_k_l31_31415

theorem sum_possible_values_of_k (j k : ℕ) (h : (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 4)) (hj : 0 < j) (hk : 0 < k) :
  {x : ℕ | (1 / (j : ℚ) + 1 / (x : ℚ) = 1 / 4) ∧ 0 < x}.sum id = 51 :=
sorry

end sum_possible_values_of_k_l31_31415


namespace number_of_chords_l31_31080

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31080


namespace hyperbola_product_slopes_constant_l31_31852

theorem hyperbola_product_slopes_constant (a b x0 y0 : ℝ) (h_a : a > 0) (h_b : b > 0) (hP : (x0 / a) ^ 2 - (y0 / b) ^ 2 = 1) (h_diff_a1_a2 : x0 ≠ a ∧ x0 ≠ -a) :
  (y0 / (x0 + a)) * (y0 / (x0 - a)) = b^2 / a^2 :=
by sorry

end hyperbola_product_slopes_constant_l31_31852


namespace mrs_franklin_needs_more_valentines_l31_31469

theorem mrs_franklin_needs_more_valentines (valentines_have : ℝ) (students : ℝ) : valentines_have = 58 ∧ students = 74 → students - valentines_have = 16 :=
by
  sorry

end mrs_franklin_needs_more_valentines_l31_31469


namespace john_money_left_l31_31433

-- Define the initial amount of money John had
def initial_amount : ℝ := 200

-- Define the fraction of money John gave to his mother
def fraction_to_mother : ℝ := 3 / 8

-- Define the fraction of money John gave to his father
def fraction_to_father : ℝ := 3 / 10

-- Prove that the amount of money John had left is $65
theorem john_money_left : 
    let money_given_to_mother := fraction_to_mother * initial_amount,
        money_given_to_father := fraction_to_father * initial_amount,
        total_given_away := money_given_to_mother + money_given_to_father,
        money_left := initial_amount - total_given_away
    in money_left = 65 := sorry

end john_money_left_l31_31433


namespace proof_triangle_xyz_ZU_l31_31553

noncomputable def triangle_xyz_ZU (XY YZ ZX : ℕ) (WY WZ : ℝ) (ZU : ℝ) : Prop :=
  XY = 13 ∧ YZ = 30 ∧ ZX = 26 ∧ 
  -- angle bisector theorem to find WY and WZ
  let ratio := XY / ZX in
  let WY := YZ * ratio / (1 + ratio) in
  let WZ := YZ / (1 + ratio) in
  -- given the calculations from the angle bisector theorem
  WY = 10 ∧ WZ = 20 →
  -- ZU should be 10 as derived from the similarity arguments  
  ZU = 10

theorem proof_triangle_xyz_ZU : triangle_xyz_ZU 13 30 26 10 20 10 :=
begin
  sorry
end

end proof_triangle_xyz_ZU_l31_31553


namespace product_two_sides_gt_product_diameters_l31_31487

variable {a b c : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variable (triangle_ineq : a + b > c)

noncomputable def Δ := 
  let s := (a + b + c) / 2 
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def r := 2 * Δ / (a + b + c)

noncomputable def R := (a * b * c) / (4 * Δ)

theorem product_two_sides_gt_product_diameters 
  (h₁ : a + b > c) 
  (h₂ : b + c > a) 
  (h₃ : c + a > b) : 
  a * b > 4 * r * R := 
by 
  sorry

end product_two_sides_gt_product_diameters_l31_31487


namespace Alex_age_l31_31558

theorem Alex_age : ∃ (x : ℕ), (∃ (y : ℕ), x - 2 = y^2) ∧ (∃ (z : ℕ), x + 2 = z^3) ∧ x = 6 := by
  sorry

end Alex_age_l31_31558


namespace cone_circumference_l31_31230

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def base_circumference (r : ℝ) : ℝ :=
  2 * π * r

theorem cone_circumference (V h : ℝ) (hV : V = 27 * π) (hh : h = 9) :
  ∃ C, C = 6 * π :=
by
  -- Definitions
  let r := sqrt (V * 3 / (π * h))
  have h_r : r = 3 := by
    field_simp [hV, hh]
    norm_num
  -- Circumference calculation
  use base_circumference r
  rw [h_r]
  norm_num
  -- Demonstrate resulting circumference
  rfl
  sorry

end cone_circumference_l31_31230


namespace total_money_shared_l31_31653

theorem total_money_shared 
  (A B C D total : ℕ) 
  (h1 : A = 3 * 15)
  (h2 : B = 5 * 15)
  (h3 : C = 6 * 15)
  (h4 : D = 8 * 15)
  (h5 : A = 45) :
  total = A + B + C + D → total = 330 :=
by
  sorry

end total_money_shared_l31_31653


namespace tom_clothing_total_l31_31551

def total_clothing (load1 load2 load3 : ℕ) : ℕ := load1 + load2 + load3

theorem tom_clothing_total : ∃ (t : ℕ), t = total_clothing 18 9 9 ∧ t = 36 := 
by
  exists 36
  simp [total_clothing]
  sorry

end tom_clothing_total_l31_31551


namespace triangle_cross_section_l31_31192

-- Definitions for the given conditions
inductive Solid
| Prism
| Pyramid
| Frustum
| Cylinder
| Cone
| TruncatedCone
| Sphere

-- The theorem statement of the proof problem
theorem triangle_cross_section (s : Solid) (cross_section_is_triangle : Prop) : 
  cross_section_is_triangle →
  (s = Solid.Prism ∨ s = Solid.Pyramid ∨ s = Solid.Frustum ∨ s = Solid.Cone) :=
sorry

end triangle_cross_section_l31_31192


namespace air_conditioned_percentage_l31_31588

theorem air_conditioned_percentage (x : ℕ) (h1 : 0 < x) 
  (h2 : ∃ (r a ar : ℕ), r = (3 / 4 : ℝ) * x ∧ a = (3 / 5 : ℝ) * x ∧ ar = (2 / 3 : ℝ) * a ∧ ar = (2 / 5 : ℝ) * x) :
  (\(\boxed(80%))office) :=
by {
  sorry
}

end air_conditioned_percentage_l31_31588


namespace sum_of_roots_eq_six_l31_31186

theorem sum_of_roots_eq_six (a b c : ℝ) (h_eq : a = 1 ∧ b = -6 ∧ c = 8) : 
  let sum_of_roots := -(b / a) in 
  sum_of_roots = 6 := 
by
  sorry

end sum_of_roots_eq_six_l31_31186


namespace total_sheets_of_paper_l31_31459

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l31_31459


namespace SR_sq_ge_AS_mul_CR_l31_31439

variable {Point : Type}
variable [Inhabited Point]

noncomputable def isRhombus (A B C D : Point) : Prop :=
  true -- This will be the definition of a rhombus.

noncomputable def angleBAD_60 (A B C D : Point) (h : isRhombus A B C D) : Prop :=
  true -- This corresponds to the given condition ∠BAD = 60°.

noncomputable def inside (P : Point) (Δ : Point → Point → Point → Prop) : Prop :=
  true -- This represents a point being inside a triangle.

variable {A B C D S R : Point}

axiom rhombus_ABCD : isRhombus A B C D
axiom angle_BAD_60 : angleBAD_60 A B C D rhombus_ABCD
axiom S_in_ABD : inside S (λ A B D , true) -- should represent S inside triangle ABD
axiom R_in_DBC : inside R (λ D B C , true) -- should represent R inside triangle DBC
axiom angle_SBR_60 : true -- should represent ∠SBR = 60°
axiom angle_RDS_60 : true -- should represent ∠RDS = 60°

theorem SR_sq_ge_AS_mul_CR :
  ∀ (A B C D S R : Point),
    isRhombus A B C D →
    angleBAD_60 A B C D rhombus_ABCD →
    inside S (λ A B D , true) →
    inside R (λ D B C , true) →
    angle_SBR_60 →
    angle_RDS_60 →
    true := -- should represent the desired inequality SR^2 >= AS ⋅ CR
    sorry

end SR_sq_ge_AS_mul_CR_l31_31439


namespace exist_non_intersecting_triangles_l31_31730

open Set

/-- Given 300 points on a plane such that no three points are collinear, prove that there exist 100 pairwise non-intersecting triangles with vertices among these points. -/
theorem exist_non_intersecting_triangles (points : Set (ℝ × ℝ))
  (h_card : points.card = 300)
  (h_no_collinear : ∀ (a b c : ℝ × ℝ), a ∈ points → b ∈ points → c ∈ points → collinear ℝ {a, b, c} → a = b ∨ b = c ∨ a = c) :
  ∃ (triangles : Finset (Finset (ℝ × ℝ))), 
    triangles.card = 100 ∧ 
    (∀ t ∈ triangles, t.card = 3 ∧ ∀ p q ∈ t, p ≠ q) ∧ 
    (∀ t₁ t₂ ∈ triangles, t₁ ≠ t₂ → disjoint t₁ t₂) :=
sorry

/-- Definition of collinearity for three points in the plane -/
def collinear (K : Type*) [Field K] [AddCommGroup K] [VectorSpace K K] (s : Set (K × K)) : Prop :=
∃ (a b : K × K) (k : K), s ⊆ {p | ∃ t : K, p = a + t • b}

end exist_non_intersecting_triangles_l31_31730


namespace two_B_lt_A_plus_C_l31_31307

variable (n : ℕ) (R r x : ℝ) (B A C : ℝ)

-- Conditions
-- A convex n-gon has both a circumscribed circle and an inscribed circle.
-- Let B be the area of the n-gon.
-- Let A be the area of the circumscribed circle.
-- Let C be the area of the inscribed circle.
def convex_ngon_conditions : Prop :=
  A = Real.pi * R^2 ∧
  C = Real.pi * r^2 ∧
  B = (1 / 2) * x * r ∧
  x < 2 * Real.pi * R

-- Proving the main inequality
theorem two_B_lt_A_plus_C (h : convex_ngon_conditions n R r x B A C) : 2 * B < A + C := by
  sorry

end two_B_lt_A_plus_C_l31_31307


namespace ratio_KL_eq_3_over_5_l31_31277

theorem ratio_KL_eq_3_over_5
  (K L : ℤ)
  (h : ∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    (K : ℝ) / (x + 3) + (L : ℝ) / (x^2 - 3 * x) = (x^2 - x + 5) / (x^3 + x^2 - 9 * x)):
  (K : ℝ) / (L : ℝ) = 3 / 5 :=
by
  sorry

end ratio_KL_eq_3_over_5_l31_31277


namespace sum_of_products_leq_one_third_l31_31009

theorem sum_of_products_leq_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 :=
sorry

end sum_of_products_leq_one_third_l31_31009


namespace eleven_million_scientific_notation_l31_31109

-- Definition of the scientific notation condition and question
def scientific_notation (a n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ k : ℤ, n = 10 ^ k

-- The main theorem stating that 11 million can be expressed as 1.1 * 10^7
theorem eleven_million_scientific_notation : scientific_notation 1.1 (10 ^ 7) :=
by 
  -- Adding sorry to skip the proof
  sorry

end eleven_million_scientific_notation_l31_31109


namespace triangle_isosceles_from_segments_l31_31662

theorem triangle_isosceles_from_segments (A B C : Type*) [linear_ordered_field A] 
  (h1 : Segment A A₁ = Segment B B₁ ∧ Segment B B₁ = Segment C C₁)
  (h2 : Segment A A₂ = Segment B B₂ ∧ Segment B B₂ = Segment C C₂)
  (h3 : Segment A A₃ = Segment B B₃ ∧ Segment B B₃ = Segment C C₃)
  (h_eq : ∀ (s : Segment A), s ∈ { Segment A A₁, Segment A A₂, Segment A A₃, 
                                    Segment B B₁, Segment B B₂, Segment B B₃,
                                    Segment C C₁, Segment C C₂, Segment C C₃ } → 
                                    ∃ t ∈ { Segment A A₁, Segment A A₂, Segment A A₃, 
                                            Segment B B₁, Segment B B₂, Segment B B₃,
                                            Segment C C₁, Segment C C₂, Segment C C₃ }, 
                                    s = t) :
triangle_is_isosceles A :=
sorry

end triangle_isosceles_from_segments_l31_31662


namespace solve_for_x_l31_31766
noncomputable theory

variables {x y z a b d : ℝ}
variables (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0)
variables (h1 : xy / (x + y) = a)
variables (h2 : xz / (x + z) = b)
variables (h3 : yz / (y - z) = d)

theorem solve_for_x (h1 : xy / (x + y) = a) (h2 : xz / (x + z) = b) (h3 : yz / (y - z) = d) (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) : 
  x = ab / (a + b) :=
sorry

end solve_for_x_l31_31766


namespace proof_propositions_l31_31745

def proposition_p (a b c : ℝ) : Prop :=
a > b ∧ (∀ c, ac^2 > bc^2 → a > b) ∧ ¬ (ac^2 ≤ bc^2 ∧ a > b)

def proposition_q (A B C a b c : ℝ) : Prop :=
∠ C > ∠ B ↔ sin C > sin B

theorem proof_propositions (a b c A B C : ℝ) :
  proposition_p a b c ∧ proposition_q A B C :=
by
  sorry

end proof_propositions_l31_31745


namespace abs_diff_of_two_numbers_l31_31135

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 :=
by
  sorry

end abs_diff_of_two_numbers_l31_31135


namespace solution_set_of_inequality_l31_31759

noncomputable def satisfies_condition (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f(x) + x * deriv (deriv f) x > 0

theorem solution_set_of_inequality {f : ℝ → ℝ} (h_dom : ∀ x, 0 < x → 0 < f x)
  (h_cond : ∀ x, 0 < x → satisfies_condition f x) :
  { x : ℝ | (x - 1) * f (x^2 - 1) < f (x + 1) } = set.Ioo 1 2 :=
by
  sorry

end solution_set_of_inequality_l31_31759


namespace matchsticks_left_l31_31697

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l31_31697


namespace expected_socks_to_pair_l31_31142

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l31_31142


namespace find_f_l31_31756

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f (x : ℝ) :
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = (1 - x^2) / (1 + x^2)) →
  f x = (2 * x) / (1 + x^2) :=
by
  intros h
  specialize h ((1 - x) / (1 + x))
  specialize h rfl
  exact sorry

end find_f_l31_31756


namespace alix_more_chocolates_than_nick_l31_31887

theorem alix_more_chocolates_than_nick :
  let nick_chocolates := 10
  let initial_alix_chocolates := 3 * nick_chocolates
  let after_mom_took_chocolates := initial_alix_chocolates - 5
  after_mom_took_chocolates - nick_chocolates = 15 := by
sorry

end alix_more_chocolates_than_nick_l31_31887


namespace rhombus_longer_diagonal_l31_31925

theorem rhombus_longer_diagonal (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_area : (d1 * d2) / 2 = 110) : d2 = 20 :=
by
  sorry

end rhombus_longer_diagonal_l31_31925


namespace cube_surface_division_and_minimal_distance_l31_31092

theorem cube_surface_division_and_minimal_distance :
  (∃ (p : ℝ^3 → Prop), (∀ cube_vertices: ℝ^3, p cube_vertices)) ∧ (∃! (O : ℝ^3), ∀ P : ℝ^3, (∑ v ∈ cube_vertices, dist P v) ≤ (∑ v ∈ cube_vertices, dist O v)) :=
by
  sorry

end cube_surface_division_and_minimal_distance_l31_31092


namespace cookie_bags_l31_31913

theorem cookie_bags (B : ℕ) (h1 : ∀ bags, total_cookies_in_bags bags = 7 * bags)
                    (h2 : ∀ boxes, total_cookies_in_boxes boxes = 12 * boxes)
                    (h3 : 8 * 12 = 7 * B + 33) : B = 9 := 
sorry

end cookie_bags_l31_31913


namespace socks_expected_value_l31_31146

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l31_31146


namespace quadratic_equation_with_roots_l31_31800

example (a b : ℝ) (h1 : (a + b) = 10) (h2 : (a * b) = 225) : 
  polynomial (x : ℝ) := 
by {
  have h_eq : polynomial := polynomial.C (-a - b) + polynomial.C (ab),
  have h_subst : h_eq == (polynomial (x : ℝ)) ( -10 + 225),
  exact h_subst sorry, 
}

Another:


theorem quadratic_equation_with_roots (a b : ℝ)
  (arithmetic_mean : (a + b) = 10)
  (geometric_mean : (a * b) = 225) :
  ∃ (x : polynomial ℝ), x = polynomial.C (-a - b) * x + polynomial.C (ab) :=
begin 
  use polynomial.C (-10) * x + polynomial.C 225, 
  rw [h1,h2],
  sorry, 
   end

Notes: Importing polynomial, RW constructions would be useful for the proof and fixing appropriately.
.export techniques require x components in polynomial library in lean(mathlib)


end quadratic_equation_with_roots_l31_31800


namespace nine_points_circle_chords_l31_31068

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31068


namespace hh3_eq_6582_l31_31028

def h (x : ℤ) : ℤ := 3 * x^2 + 5 * x + 4

theorem hh3_eq_6582 : h (h 3) = 6582 :=
by
  sorry

end hh3_eq_6582_l31_31028


namespace expected_pairs_socks_l31_31153

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l31_31153


namespace not_possible_to_color_l31_31007

theorem not_possible_to_color (f : ℕ → ℕ) (c1 c2 c3 : ℕ) :
  ∃ (x : ℕ), 1 < x ∧ f 2 = c1 ∧ f 4 = c1 ∧ 
  ∀ (a b : ℕ), 1 < a → 1 < b → f a ≠ f b → (f (a * b) ≠ f a ∧ f (a * b) ≠ f b) → 
  false :=
sorry

end not_possible_to_color_l31_31007


namespace number_of_chords_number_of_chords_l31_31056

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31056


namespace cubic_roots_l31_31499

open Real

theorem cubic_roots (x1 x2 x3 : ℝ) (h1 : x1 * x2 = 1)
  (h2 : 3 * x1^3 + 2 * sqrt 3 * x1^2 - 21 * x1 + 6 * sqrt 3 = 0)
  (h3 : 3 * x2^3 + 2 * sqrt 3 * x2^2 - 21 * x2 + 6 * sqrt 3 = 0)
  (h4 : 3 * x3^3 + 2 * sqrt 3 * x3^2 - 21 * x3 + 6 * sqrt 3 = 0) :
  (x1 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x1 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) := 
sorry

end cubic_roots_l31_31499


namespace smallest_positive_period_intervals_of_monotonic_increase_max_min_values_l31_31339

def f (x : ℝ) : ℝ := sin (2 * x + π / 3) - sqrt 3 * sin (2 * x - π / 6)

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem intervals_of_monotonic_increase : ∃ (k : ℤ), ∀ x, 
  (2 * x + π / 3 >= -π / 2 + 2 * k * π ∧ 2 * x + π / 3 <= π / 2 + 2 * k * π) 
  ↔ (x >= -7 * π / 12 + k * π ∧ x <= -π / 12 + k * π) := sorry

theorem max_min_values : ∃ (x_max x_min : ℝ), 
  x_max ∈ [-π / 6, π / 3] ∧ x_min ∈ [-π / 6, π / 3] ∧ 
  (f x_max = 2 ∧ f x_min = -sqrt 3) ∧ 
  (x_max = -π / 12 ∧ x_min = π / 3) := sorry

end smallest_positive_period_intervals_of_monotonic_increase_max_min_values_l31_31339


namespace find_x_solutions_l31_31292

theorem find_x_solutions :
  { x : ℝ | (9^x + 32^x) / (15^x + 24^x) = 4 / 3 } =
  { real.logb (3 / 2) (3 / 4), real.logb 3 4 } :=
sorry

end find_x_solutions_l31_31292


namespace find_point_A_l31_31893

theorem find_point_A :
  (∃ A : ℤ, A + 2 = -2) ∨ (∃ A : ℤ, A - 2 = -2) → (∃ A : ℤ, A = 0 ∨ A = -4) :=
by
  sorry

end find_point_A_l31_31893


namespace intercepts_of_line_l31_31173

theorem intercepts_of_line (x y : ℝ) : 
  (x + 6 * y + 2 = 0) → (x = -2) ∧ (y = -1 / 3) :=
by
  sorry

end intercepts_of_line_l31_31173


namespace triangle_is_right_l31_31765

theorem triangle_is_right :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (1/2, real.sqrt 3 / 2)
  (vector.dot 
    (C.1 - A.1, C.2 - A.2) 
    (C.1 - B.1, C.2 - B.2) = 0) :=
by
  sorry

end triangle_is_right_l31_31765


namespace segment_length_AB_l31_31480

theorem segment_length_AB (x y : ℝ) (PR : ℝ) (P_ratio Q_ratio R_ratio_PQ : ℝ) (PR_length : ℝ) :
  x = (3 / 4) * (9 + y) →
  (x + 9) / y = (4 / 5) →
  PR = 1 →
  R_ratio_PQ = 1 / 2 →
  PR_length = 3 →
  9 = PR_length * (1 + R_ratio_PQ) →
  AB : ℝ := x + 9 + y →
  AB = 567 := 
sorry

end segment_length_AB_l31_31480


namespace probability_no_touch_outer_edge_l31_31890

theorem probability_no_touch_outer_edge :
  let total_squares := 100 in
  let perimeter_squares := 36 in
  let inner_squares := total_squares - perimeter_squares in
  let probability := inner_squares / total_squares in
  probability = (16 : ℚ) / 25 := 
by {
  have h1 : total_squares = 10 * 10 := by simp [total_squares],
  have h2 : perimeter_squares = (10 + 10 + (10 - 2) + (10 - 2)) := by simp [perimeter_squares],
  have h3 : inner_squares = total_squares - perimeter_squares := by simp [inner_squares, h1, h2],
  have h4 : probability = inner_squares / total_squares := by simp [probability],
  norm_num,
  rw [h1, h2] at *,
  norm_num at *,
  exact h4,
  sorry
}

end probability_no_touch_outer_edge_l31_31890


namespace gymnastics_team_square_formation_l31_31222

theorem gymnastics_team_square_formation :
  let n := 48 in let s1 := 49 in let s2 := 36 in
  (∃ a, n + a = s1) ∧ (∃ r, n - r = s2) :=
by
  let n := 48
  let s1 := 49
  let s2 := 36
  have h1 : ∃ a, n + a = s1 := ⟨1, by simp⟩
  have h2 : ∃ r, n - r = s2 := ⟨12, by simp⟩
  exact ⟨h1, h2⟩
  sorry

end gymnastics_team_square_formation_l31_31222


namespace expected_socks_pairs_l31_31148

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l31_31148


namespace k_less_than_two_l31_31361

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l31_31361


namespace root_of_unity_l31_31688

noncomputable def tan (x : ℝ) : ℝ := Real.tan x

theorem root_of_unity (n : ℤ) (h : 0 ≤ n ∧ n < 14) : 
    (Complex.ofReal (tan (Real.pi / 7)) + Complex.I) / 
    (Complex.ofReal (tan (Real.pi / 7)) - Complex.I) = 
    Complex.exp(Complex.I * (10 * Real.pi / 14)) := 
sorry

end root_of_unity_l31_31688


namespace total_boxes_l31_31157

variable (N_initial : ℕ) (N_nonempty : ℕ) (N_new_boxes : ℕ)

theorem total_boxes (h_initial : N_initial = 7) 
                     (h_nonempty : N_nonempty = 10)
                     (h_new_boxes : N_new_boxes = N_nonempty * 7) :
  N_initial + N_new_boxes = 77 :=
by 
  have : N_initial = 7 := h_initial
  have : N_new_boxes = N_nonempty * 7 := h_new_boxes
  have : N_nonempty = 10 := h_nonempty
  sorry

end total_boxes_l31_31157


namespace num_integer_solutions_eq_3_l31_31335

theorem num_integer_solutions_eq_3 :
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((2 * x^2) + (x * y) + (y^2) - x + 2 * y + 1 = 0 ↔ (x, y) ∈ S)) ∧ 
  S.card = 3 :=
sorry

end num_integer_solutions_eq_3_l31_31335


namespace chords_from_nine_points_l31_31048

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31048


namespace minimum_norm_of_linear_combination_l31_31321

open Complex Real

-- Defining the complex number ω
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

-- Main theorem statement
theorem minimum_norm_of_linear_combination {a b c : ℤ} (h : a * b * c = 60) (hω : ω ≠ 1) (hω3 : ω ^ 3 = 1) : 
  ∃ a b c : ℤ, h ∧ |(a : ℂ) + (b : ℂ) * ω + (c : ℂ) * ω ^ 2| = √3 :=
by 
  sorry

end minimum_norm_of_linear_combination_l31_31321


namespace nine_points_circle_chords_l31_31071

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31071


namespace Heather_delay_l31_31914

noncomputable def find_start_time : ℝ :=
  let d := 15 -- Initial distance between Stacy and Heather in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := H + 1 -- Stacy's speed in miles/hour
  let d_H := 5.7272727272727275 -- Distance Heather walked when they meet
  let t_H := d_H / H -- Time Heather walked till they meet in hours
  let d_S := S * t_H -- Distance Stacy walked till they meet in miles
  let total_distance := d_H + d_S -- Total distance covered when they meet in miles
  let remaining_distance := d - total_distance -- Remaining distance Stacy covers alone before Heather starts in miles
  let t_S := remaining_distance / S -- Time Stacy walked alone in hours
  let minutes := t_S * 60 -- Convert time Stacy walked alone to minutes
  minutes -- Result in minutes

theorem Heather_delay : find_start_time = 24 := by
  sorry -- Proof of the theorem

end Heather_delay_l31_31914


namespace find_x_solutions_l31_31290

theorem find_x_solutions (x : ℝ) (h : (9^x + 32^x) / (15^x + 24^x) = 4 / 3) : x = -1 ∨ x = 1 :=
sorry

end find_x_solutions_l31_31290


namespace total_weight_of_lifts_l31_31406

theorem total_weight_of_lifts
  (F S : ℕ)
  (h1 : F = 600)
  (h2 : 2 * F = S + 300) :
  F + S = 1500 := by
  sorry

end total_weight_of_lifts_l31_31406


namespace initial_percentage_rise_l31_31234

-- Definition of the conditions
def final_price_gain (P : ℝ) (x : ℝ) : Prop :=
  P * (1 + x / 100) * 0.9 * 0.85 = P * 1.03275

-- The statement to be proven
theorem initial_percentage_rise (P : ℝ) (x : ℝ) : final_price_gain P x → x = 35.03 :=
by
  sorry -- Proof to be filled in

end initial_percentage_rise_l31_31234


namespace union_is_equivalent_l31_31017

def A (x : ℝ) : Prop := x ^ 2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem union_is_equivalent (x : ℝ) :
  (A x ∨ B x) ↔ (-2 ≤ x ∧ x < 4) :=
sorry

end union_is_equivalent_l31_31017


namespace prime_arithmetic_sequence_sum_l31_31539

open Nat

theorem prime_arithmetic_sequence_sum :
  ∃! (s : Finset ℕ), (∀ x ∈ s, Prime x) ∧ (∃ d, d = 6 ∧ ∀ x y ∈ s, x ≠ y → abs (x - y) = d) ∧ s.sum id = 85 := 
by
  sorry

end prime_arithmetic_sequence_sum_l31_31539


namespace number_of_chords_number_of_chords_l31_31053

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31053


namespace range_a_range_reciprocal_sum_l31_31348

noncomputable def f (x a : ℝ) := x * abs (x - 2 * a) + a^2 - 3 * a

theorem range_a (a : ℝ) :
  ∃ x1 x2 x3 : ℝ, (x1 < x2 ∧ x2 < x3) ∧ (f x1 a = 0) ∧ (f x2 a = 0) ∧ (f x3 a = 0) ↔
  (3/2 < a ∧ a < 3) :=
begin
  sorry
end

theorem range_reciprocal_sum (a x1 x2 x3 : ℝ) (h : 3/2 < a ∧ a < 3) :
  x1 < x2 ∧ x2 < x3 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 →
  ∃ k : ℝ, k = (1/x1 + 1/x2 + 1/x3) ∧ (2*(sqrt 2 + 1)/3 < k) :=
begin
  sorry
end

end range_a_range_reciprocal_sum_l31_31348


namespace mandy_yoga_time_l31_31424

theorem mandy_yoga_time:
  ∀ (gym bicycle yoga: ℕ), (gym / bicycle = 2 / 3) ∧ (yoga / (gym + bicycle) = 2 / 3) ∧ (bicycle = 18) →
  yoga = 20 :=
by
  intro gym bicycle yoga
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end mandy_yoga_time_l31_31424


namespace sienna_marble_sharing_l31_31099

theorem sienna_marble_sharing (x : ℤ) :
  (150 - x) = 3 * (90 + x) →
  x = 30 :=
by
  intros h,
  sorry

end sienna_marble_sharing_l31_31099


namespace chords_from_nine_points_l31_31078

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31078


namespace tangent_line_equation_l31_31748

noncomputable section

-- Define the given point A
def A := (-1 : ℝ, 2 : ℝ)

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the point A on the parabola
def A_on_parabola : Prop := A.snd = parabola A.fst

-- Calculate the derivative of the parabola
def parabola_slope (x : ℝ) : ℝ := 4 * x

-- Determine the slope of the tangent at A
def tangent_slope_at_A : ℝ := parabola_slope A.fst

-- Define the tangent line equation
def tangent_line (x : ℝ) (y : ℝ) : Prop := y = -4 * x - 2

-- Convert to standard form
def standard_form (x y : ℝ) : Prop := 4 * x + y + 2 = 0

-- The theorem to prove
theorem tangent_line_equation : standard_form ∧ ∀ x y, tangent_line x y → standard_form x y :=
by
  sorry

end tangent_line_equation_l31_31748


namespace karlsson_weight_l31_31207

variable {F K M : ℕ}

theorem karlsson_weight (h1 : F + K = M + 120) (h2 : K + M = F + 60) : K = 90 := by
  sorry

end karlsson_weight_l31_31207


namespace abs_add_gt_abs_sub_l31_31746

variables {a b : ℝ}

theorem abs_add_gt_abs_sub (h : a * b > 0) : |a + b| > |a - b| :=
sorry

end abs_add_gt_abs_sub_l31_31746


namespace find_l_l31_31447

variables (a b c l : ℤ)
def g (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_l :
  g a b c 2 = 0 →
  60 < g a b c 6 ∧ g a b c 6 < 70 →
  80 < g a b c 9 ∧ g a b c 9 < 90 →
  6000 * l < g a b c 100 ∧ g a b c 100 < 6000 * (l + 1) →
  l = 5 :=
sorry

end find_l_l31_31447


namespace part_a_independent_of_x_l31_31031

theorem part_a_independent_of_x (p : ℝ) :
  (∀ x : ℝ, (sin x)^6 + (cos x)^6 + p * ((sin x)^4 + (cos x)^4) = (1 + p) - ((3 + 2 * p) / 4)) :=
sorry

end part_a_independent_of_x_l31_31031


namespace street_sweeper_routes_l31_31237

def num_routes (A B C : Type) :=
  -- Conditions: Starts from point A, 
  -- travels through all streets exactly once, 
  -- and returns to point A.
  -- Correct Answer: Total routes = 12
  2 * 6 = 12

theorem street_sweeper_routes (A B C : Type) : num_routes A B C := by
  -- The proof is omitted as per instructions
  sorry

end street_sweeper_routes_l31_31237


namespace total_sheets_of_paper_l31_31460

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l31_31460


namespace number_of_functions_satisfying_equation_l31_31789

/-- 
There are exactly 2 functions f : ℝ → ℝ satisfying the 
functional equation f(x)f(y)f(z) = 12 * f(x * y * z) - 16 * (x * y * z)
for every real numbers x, y, z.
-/
theorem number_of_functions_satisfying_equation :
  {f : ℝ → ℝ | ∀ x y z, f(x) * f(y) * f(z) = 12 * f(x * y * z) - 16 * (x * y * z)}.to_finset.card = 2 :=
sorry

end number_of_functions_satisfying_equation_l31_31789


namespace collinear_KED_l31_31473

variable (A B C D E N M K : Type)
variable [has_lt A] [has_lt B] [has_lt C] [has_lt D] [has_lt E] [has_lt N] [has_lt M] [has_lt K]

-- Assume ABCD is a rhombus
def is_rhombus (ABCD : A × B × C × D) : Prop := sorry

-- E is a point on AC such that E ≠ A and E ≠ C
def is_point_on_AC (E : A × C) : Prop := sorry
def E_distinct (E A C : Type) : Prop := (E ≠ A) ∧ (E ≠ C)

-- N is a point on AB such that AE = NE
def is_point_on_AB (N : A × B) : Prop := sorry
def same_distance_AE_NE (A E N : Type) : Prop := sorry

-- M is a point on BC such that CE = ME
def is_point_on_BC (M : B × C) : Prop := sorry
def same_distance_CE_ME (C E M : Type) : Prop := sorry

-- K is the intersection of AM and CN
def intersection_AM_CN (K : Type) : Prop := sorry

-- Prove points K, E, and D are collinear
theorem collinear_KED (A B C D E N M K : Type)
  [has_lt A] [has_lt B] [has_lt C] [has_lt D] [has_lt E] [has_lt N] [has_lt M] [has_lt K]
  (h_rhombus : is_rhombus (A, B, C, D))
  (h_on_AC : is_point_on_AC E)
  (h_E_distinct : E_distinct E A C)
  (h_on_AB : is_point_on_AB N)
  (h_same_AE_NE : same_distance_AE_NE A E N)
  (h_on_BC : is_point_on_BC M)
  (h_same_CE_ME : same_distance_CE_ME C E M)
  (h_intersection : intersection_AM_CN K) :
  is_collinear K E D := sorry

end collinear_KED_l31_31473


namespace EY_eq_FZ_l31_31787

open Set

variables {A B C D E F X Y Z : Point}
variable (incircle : Circle)
variable (triangle : Triangle) (triangleABC : triangle = Triangle.mk A B C)
variable (touches_inc : incircle.tangency_points A B C → (D, E, F))
variable (AD_line : line A D) (AX_XD : segment A X → segment X D → AX_XD = true)
variable (BX_line : line B X) (CX_line : line C X)
variable (Y_intersect : circle_point_intersection incircle B X Y)
variable (Z_intersect : circle_point_intersection incircle C X Z)

theorem EY_eq_FZ :
  ∀ ⦃A B C D E F X Y Z : Point⦄ incircle triangle triangleABC touches_inc AD_line AX_XD BX_line CX_line Y_intersect Z_intersect,
    distance E Y = distance F Z :=
sorry

end EY_eq_FZ_l31_31787


namespace sum_adjacent_to_49_l31_31599

noncomputable def sum_of_adjacent_divisors : ℕ :=
  let divisors := [5, 7, 35, 49, 245]
  -- We assume an arrangement such that adjacent pairs to 49 are {35, 245}
  35 + 245

theorem sum_adjacent_to_49 : sum_of_adjacent_divisors = 280 := by
  sorry

end sum_adjacent_to_49_l31_31599


namespace delta_five_is_zero_l31_31674

def v (n : ℕ) : ℕ := n^4 + 2 * n^2

def Δ (k : ℕ) (v : ℕ → ℕ) : ℕ → ℕ
| 0     := λ n, v n
| (k+1) := λ n, Δ k v (n+1) - Δ k v n

theorem delta_five_is_zero (v : ℕ → ℕ) (Δ : ℕ → (ℕ → ℕ) → ℕ → ℕ) (n : ℕ) : Δ 5 v n = 0 := by
sory

end delta_five_is_zero_l31_31674


namespace num_valid_colorings_l31_31263

/-- Define a 4x4 chessboard with cells indexed by (i, j) where i, j are from 1 to 4. -/
def chessboard := Fin 4 × Fin 4

/-- Define a coloring of the chessboard as a set of cells. -/
def coloring := set chessboard

/-- Predicate for valid coloring: 
  each row and each column has exactly 2 black squares. --/
def is_valid_coloring (coloring : coloring) : Prop :=
  ∀ i : Fin 4, (finset.univ.filter (λ j, (i, j) ∈ coloring)).card = 2 ∧ 
               (finset.univ.filter (λ j, (j, i) ∈ coloring)).card = 2

/-- Main theorem stating the number of valid colorings is 90. -/
theorem num_valid_colorings : (finset.univ.filter is_valid_coloring).card = 90 :=
by sorry

end num_valid_colorings_l31_31263


namespace remainder_of_5032_div_28_l31_31992

theorem remainder_of_5032_div_28 : 5032 % 28 = 20 :=
by
  sorry

end remainder_of_5032_div_28_l31_31992


namespace num_chords_l31_31043

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31043


namespace points_on_circumcircle_of_orthocenter_symmetry_l31_31484

noncomputable def orthocenter_symmetry (ABC : Triangle) (H A_1 B_1 C_1 : Point) : Prop :=
  let A := ABC.vertex_A in
  let B := ABC.vertex_B in
  let C := ABC.vertex_C in
  H = ABC.orthocenter ∧
  A_1 = H.symmetric_over BC ∧
  B_1 = H.symmetric_over CA ∧
  C_1 = H.symmetric_over AB ∧
  A_1 ∈ ABC.circumcircle ∧
  B_1 ∈ ABC.circumcircle ∧
  C_1 ∈ ABC.circumcircle

variable (ABC : Triangle)
variable (H A_1 B_1 C_1 : Point)

theorem points_on_circumcircle_of_orthocenter_symmetry :
  orthocenter_symmetry ABC H A_1 B_1 C_1 :=
sorry

end points_on_circumcircle_of_orthocenter_symmetry_l31_31484


namespace infinitely_many_n_f_n_plus_1_gt_f_n_infinitely_many_n_f_n_plus_1_lt_f_n_l31_31596

def f (n : ℕ) : ℚ := 
  if n = 0 then 0 else (1 / n) * ((List.range (n+1)).map (λ k, (n / (k +1))).sum)

theorem infinitely_many_n_f_n_plus_1_gt_f_n : ∃ᶠ n in at_top, f (n + 1) > f n :=
sorry

theorem infinitely_many_n_f_n_plus_1_lt_f_n : ∃ᶠ n in at_top, f (n + 1) < f n :=
sorry

end infinitely_many_n_f_n_plus_1_gt_f_n_infinitely_many_n_f_n_plus_1_lt_f_n_l31_31596


namespace nine_points_chords_l31_31059

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31059


namespace number_of_chords_l31_31083

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31083


namespace interval_range_l31_31387

noncomputable def f (x : ℝ) := x^3 - 3 * x

def has_maximum_on (a : ℝ) : Prop :=
  a < 6 - a^2 ∧ a < -1 ∧ 6 - a^2 > -1 ∧ 6 - a^2 ≤ 2

theorem interval_range (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo a (6 - a^2) ∧ f x = f (-1)) → -sqrt 7 < a ∧ a ≤ -2 :=
by
  sorry

end interval_range_l31_31387


namespace min_value_xyz_l31_31729

theorem min_value_xyz (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 := 
by
  sorry

end min_value_xyz_l31_31729


namespace solve_inequality_l31_31911

theorem solve_inequality (x : ℝ) : 
  -2 < (x^2 - 18*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 18*x + 35) / (x^2 - 4*x + 8) < 2 ↔ 
  3 < x ∧ x < 17 / 3 :=
by
  sorry

end solve_inequality_l31_31911


namespace tan_cos_addition_l31_31265

noncomputable def tan_30 : ℝ := 1 / real.sqrt 3
noncomputable def cos_30 : ℝ := real.sqrt 3 / 2
noncomputable def sin_30 : ℝ := 1 / 2

theorem tan_cos_addition  : tan_30 + 4 * cos_30 = (7 * real.sqrt 3) / 3 := 
by 
  sorry

end tan_cos_addition_l31_31265


namespace radar_placement_and_coverage_ring_l31_31985

theorem radar_placement_and_coverage_ring 
  (r d : ℝ) (n : ℕ) (width : ℝ) (π : Real.pi) 
  (h_radius : r = 37) 
  (h_n : n = 9) 
  (h_width : width = 24) :
  let distance := (35 / Real.sin (20 * Real.pi / 180)) in
  let outer_radius := (35 / Real.tan (20 * Real.pi / 180) + 12) in
  let inner_radius := (35 / Real.tan (20 * Real.pi / 180) - 12) in
  let ring_area := π * (outer_radius ^ 2 - inner_radius ^ 2) in
  distance = 35 / Real.sin (20 * Real.pi / 180) ∧
  ring_area = 1680 * π / Real.tan (20 * Real.pi / 180) :=
by
  sorry

end radar_placement_and_coverage_ring_l31_31985


namespace parabola_function_expression_l31_31331

noncomputable def vertex_parabola (a b x₀ y₀ : ℝ) : (ℝ × ℝ) → ℝ
| (x, y) := y - (a * (x - x₀)^2 + b)

theorem parabola_function_expression :
  ∃ a : ℝ, (vertex_parabola a 2 (-3) 2 (1, -14) = 0) ∧ ∀ x, vertex_parabola (-1) 2 (-3) 2 (x, (-(x + 3)^2 + 2)) = 0 :=
by
  sorry

end parabola_function_expression_l31_31331


namespace Jeff_has_20_trucks_l31_31843

theorem Jeff_has_20_trucks
  (T C : ℕ)
  (h1 : C = 2 * T)
  (h2 : T + C = 60) :
  T = 20 :=
sorry

end Jeff_has_20_trucks_l31_31843


namespace smallest_n_correct_l31_31452

open Real

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | ∃ (m : ℝ) (r : ℝ), (r > 0 ∧ r < 1/1000) ∧ (m = (n + r)^3)}

theorem smallest_n_correct : smallest_n = 19 := by
  sorry

end smallest_n_correct_l31_31452


namespace find_f_half_l31_31735

-- Definition of the power function f with an unknown exponent α
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Given that the power function passes through the point (4, 2)
def condition (α : ℝ) : Prop := power_function α 4 = 2

-- The statement we want to prove
theorem find_f_half (α : ℝ) (hα : condition α) : power_function α (1 / 2) = Real.sqrt 2 / 2 := by
  sorry

end find_f_half_l31_31735


namespace find_x_l31_31796

theorem find_x (x : ℝ) (hx : x > 0) (h : Real.sqrt (12*x) * Real.sqrt (5*x) * Real.sqrt (7*x) * Real.sqrt (21*x) = 21) : 
  x = 21 / 97 :=
by
  sorry

end find_x_l31_31796


namespace acute_triangle_count_l31_31721

noncomputable def number_of_acute_triangles (x : ℤ) : ℕ :=
  if 14 < x ∧ x < 34 ∧ x < 26 ∧ (24 < x → x^2 < 676) ∧ (x ≤ 24 → 476 < x^2)
  then 1 else 0

theorem acute_triangle_count : 
  (∑ x in (finset.range 35), number_of_acute_triangles x) = 4 :=
by
  sorry

end acute_triangle_count_l31_31721


namespace min_ab_l31_31753

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + b + 3 = a * b) : 9 ≤ a * b :=
sorry

end min_ab_l31_31753


namespace chords_from_nine_points_l31_31079

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31079


namespace inverse_function_log_l31_31388

theorem inverse_function_log {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ y : ℝ, f y = real.log y / real.log a) (h4 : f 2 = 1) :
  ∀ x : ℝ, f x = real.log x / real.log 2 := 
by
  sorry

end inverse_function_log_l31_31388


namespace number_of_ways_l31_31094

def number_of_ways_to_place_balls : ℕ :=
  15

def boxes_numbered : list ℕ :=
  [1, 2, 3]

def min_balls_in_each_box (box_num : ℕ) : ℕ :=
  box_num

theorem number_of_ways (total_balls : ℕ) (box_ids : list ℕ)
  (min_balls : ℕ → ℕ) : 
  total_balls = 15 ∧ box_ids = [1, 2, 3] ∧ (∀ box ∈ box_ids, min_balls box = box) → 
  (number_of_ways_to_place_balls total_balls box_ids min_balls = 91) := 
by {
  sorry
}

end number_of_ways_l31_31094


namespace polynomial_conditions_l31_31296

def q (x : ℝ) : ℝ := -3/5 * x^2 - 3/5 * x + 36/5

theorem polynomial_conditions :
  q (-4) = 0 ∧ q (3) = 0 ∧ q (6) = -18 := by
  sorry

end polynomial_conditions_l31_31296


namespace sum_of_remainders_l31_31251

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : 
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := 
by 
  sorry

end sum_of_remainders_l31_31251


namespace tan_E_tan_F_l31_31961

theorem tan_E_tan_F (E F : ℝ) (hE : E ∈ Ioo 0 (Real.pi / 2)) (hF : F ∈ Ioo 0 (Real.pi / 2)) (HM HD : ℝ) (hHM : HM = 10) (hHD : HD = 18) :
  Real.tan E * Real.tan F = 2.8 :=
by
  sorry

end tan_E_tan_F_l31_31961


namespace original_price_of_shirt_l31_31641

variable (P : ℝ)
variable (discount_price : ℝ := 560)
variable (discount_percentage : ℝ := 0.40)

theorem original_price_of_shirt : P = 933.33 :=
by
  have paid_percentage : ℝ := 1 - discount_percentage
  have original_price : ℝ := discount_price / paid_percentage
  have original_price_equals_P : original_price = P := by sorry
  show P = 933.33 from sorry

end original_price_of_shirt_l31_31641


namespace find_real_number_l31_31712

theorem find_real_number (a b : ℕ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + b * i)^3.im = 107 * i) 
  (h3 : 3 * a^2 * b - b^3 = 107) :
  (a = 6 ∧ b = 1) → (a^3 - 3 * a * b^2 = 198) := 
by
  intros
  subst_vars
  sorry

end find_real_number_l31_31712


namespace solve_for_M_l31_31380

theorem solve_for_M (a b M : ℝ) (h : (a + 2 * b) ^ 2 = (a - 2 * b) ^ 2 + M) : M = 8 * a * b :=
by sorry

end solve_for_M_l31_31380


namespace num_chords_l31_31039

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31039


namespace circle_tangent_l31_31927

theorem circle_tangent (center : Point) (line : Line) (radius : ℝ) (x y : ℝ) :
  center = (2, -1) → line = {pt : Point | pt.1 + pt.2 = 5} → radius = 2 * Real.sqrt 2 →
  (x - 2)^2 + (y + 1)^2 = 8 := by
  intros hc hl hr
  sorry

end circle_tangent_l31_31927


namespace simplify_expression_to_fraction_l31_31498

theorem simplify_expression_to_fraction : 
  (1 / (1 / (1/2)^2 + 1 / (1/2)^3 + 1 / (1/2)^4 + 1 / (1/2)^5)) = 1/60 :=
by 
  have h1 : 1 / (1/2)^2 = 4 := by sorry
  have h2 : 1 / (1/2)^3 = 8 := by sorry
  have h3 : 1 / (1/2)^4 = 16 := by sorry
  have h4 : 1 / (1/2)^5 = 32 := by sorry
  have h5 : 4 + 8 + 16 + 32 = 60 := by sorry
  have h6 : 1 / 60 = 1/60 := by sorry
  sorry

end simplify_expression_to_fraction_l31_31498


namespace bottle_caps_before_vacation_l31_31035

theorem bottle_caps_before_vacation (x : ℝ) (h1 : x + 0.4 * x - 0.2 * (x + 0.4 * x) = x + 21) : x = 175 :=
by
  have h2 : 1.12 * x = x + 21 := by
    calc
      x + 0.4 * x - 0.2 * (x + 0.4 * x)
          = x + 0.4 * x - 0.2 * 1.4 * x : by rw [← mul_add, smul_eq_mul]
      ... = x + 0.4 * x - 0.28 * x : by norm_num
      ... = x + 0.4 * x - 0.28 * x : by norm_num
      ... = 1.12 * x : by ring
  sorry

end bottle_caps_before_vacation_l31_31035


namespace angie_carlos_probability_l31_31252

theorem angie_carlos_probability :
  (∃ (positions : list string), 
    positions.length = 5 ∧ 
    {a, b, c, d, e} = {Angie, Bridget, Carlos, Diego, Eliza} ∧ 
    position Angie positions = n ∧ 
    (position Carlos positions = (n + 2) % 5 ∨ position Carlos positions = (n - 2) % 5) 
    /
    (5! = 5 permutations)) = 1/2 := by
  sorry

end angie_carlos_probability_l31_31252


namespace find_coefficient_l31_31975

theorem find_coefficient :
  let b := 2
  in (3.choose 0 * (2:ℤ)^(3+2-0) + 3.choose 1 * (2:ℤ)^3 * 5 + 3.choose 2 * 2) = 446 :=
by
  sorry

end find_coefficient_l31_31975


namespace nine_points_chords_l31_31060

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31060


namespace find_n_l31_31533

theorem find_n (a b n : ℕ) (k l m : ℤ) 
  (ha : a % n = 2) 
  (hb : b % n = 3) 
  (h_ab : a > b) 
  (h_ab_mod : (a - b) % n = 5) : 
  n = 6 := 
sorry

end find_n_l31_31533


namespace sum_of_roots_eq_six_l31_31188

theorem sum_of_roots_eq_six (a b c : ℝ) (h_eq : a = 1 ∧ b = -6 ∧ c = 8) : 
  let sum_of_roots := -(b / a) in 
  sum_of_roots = 6 := 
by
  sorry

end sum_of_roots_eq_six_l31_31188


namespace part_a_part_b_l31_31314

-- Definitions and conditions
variable {A B C D E F K L M A₁ : Type}
variable (hABC : AcuteTriangle A B C)
variable (hIncircle : IncircleTouchesTriangle A B C D E F)
variable (hAngleBisector : AngleBisector A D K E L F)
variable (hAltitude : IsAltitude A A₁ B C)
variable (hMidpoint : IsMidpoint M B C)

-- Question 1: Prove that BK and CL are perpendicular to the angle bisector of ∠BAC.
theorem part_a
  (hAcute : AcuteTriangle A B C)
  (hIncircle : IncircleTouchesTriangle A B C D E F)
  (hAngleBisector : AngleBisector A D K E L F)
  (hAltitude : IsAltitude A A₁ B C)
  (hMidpoint : IsMidpoint M B C) :
  Perpendicular (Line B K) (AngleBisector A D K E L F) ∧ Perpendicular (Line C L) (AngleBisector A D K E L F) :=
sorry

-- Question 2: Prove that A₁KML is a cyclic quadrilateral.
theorem part_b
  (hAcute : AcuteTriangle A B C)
  (hIncircle : IncircleTouchesTriangle A B C D E F)
  (hAngleBisector : AngleBisector A D K E L F)
  (hAltitude : IsAltitude A A₁ B C)
  (hMidpoint : IsMidpoint M B C) :
  CyclicQuadrilateral A₁ K M L :=
sorry

end part_a_part_b_l31_31314


namespace length_of_AB_l31_31920

theorem length_of_AB
  (O A B C : Type) [MetricSpace O]
  (O1 O2 O3 : MetricSpace O)
  (r3 : ℝ)
  (hO3_radius : r3 = 13)
  (hO1_tangent : ∀ (x : O), dist x A = r3)
  (hO2_tangent : ∀ (y : O), dist y B = r3)
  (hO1_passes_O : ∀ (x : O), dist x O = 2 * r3)
  (hO2_passes_O : ∀ (y : O), dist y O = 2 * r3)
  (hO1_intersect_O2 : dist O C = 12) :
  dist A B = 10 := sorry

end length_of_AB_l31_31920


namespace modified_monotonous_count_correct_l31_31271

def modified_monotonous_count : ℕ :=
  -- Define the total count of modified monotonous positive integers satisfying the problem constraints
  2976

theorem modified_monotonous_count_correct : 
  ∃ n : ℕ, n = modified_monotonous_count ∧ 
  n = 2976 := 
by
  use modified_monotonous_count
  split
  . rfl
  . rfl

end modified_monotonous_count_correct_l31_31271


namespace rebecca_charge_for_dye_job_l31_31492

def charges_for_services (haircuts per perms per dye_jobs hair_dye_per_dye_job tips : ℕ) : ℕ := 
  4 * 30 + 1 * 40 + 2 * (dye_jobs - hair_dye_per_dye_job) + tips

theorem rebecca_charge_for_dye_job 
  (haircuts: ℕ) (perms: ℕ) (hair_dye_per_dye_job: ℕ) (tips: ℕ) (end_of_day_amount: ℕ) : 
  haircuts = 4 → perms = 1 → hair_dye_per_dye_job = 10 → tips = 50 → 
  end_of_day_amount = 310 → 
  ∃ D: ℕ, D = 60 := 
by
  sorry

end rebecca_charge_for_dye_job_l31_31492


namespace max_distinct_license_plates_l31_31402

noncomputable def max_license_plates : ℕ :=
  100000

theorem max_distinct_license_plates (digits : Fin 10) (plates : Fin 100000 → Fin 10 → digits)
  (h : ∀ (i j : Fin 100000), i ≠ j → ∃ (k : Fin 6), plates i k ≠ plates j k) :
  max_license_plates = 100000 :=
by
  sorry

end max_distinct_license_plates_l31_31402


namespace collinear_A_S_T_l31_31874

noncomputable theory
open_locale classical

variables (A B C D E F P Q R K L S T : Point)

-- Conditions
def incircle_touches (triangle : Triangle Point)
  (D E F : Point) : Prop := 
  touches_incircle triangle A B C D E F

def a_excircle_touches (triangle : Triangle Point)
  (P Q R : Point) : Prop := 
  touches_a_excircle triangle A B C P Q R

def perpendicular_through_A (line_perp : Line Point) (K L : Point) : Prop :=
  perpendicular_to_BC_through_A triangle line_perp K L

def intersection_points (LD KP EF QR : Line Point)
  (S T : Point) : Prop :=
  intersects_at LD EF S ∧ intersects_at KP QR T

-- Conjecture: A, S, and T are collinear
theorem collinear_A_S_T (triangle : Triangle Point) (D E F P Q R K L S T : Point)
  (line_perp : Line Point) (LD KP EF QR : Line Point) :
  incircle_touches triangle D E F →
  a_excircle_touches triangle P Q R →
  perpendicular_through_A line_perp K L →
  intersection_points LD KP EF QR S T →
  collinear A S T :=
sorry

end collinear_A_S_T_l31_31874


namespace ana_wins_probability_l31_31012

noncomputable def probability_ana_wins : ℚ :=
  (1 / 2) ^ 5 / (1 - (1 / 2) ^ 5)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 31 :=
by
  sorry

end ana_wins_probability_l31_31012


namespace condition_for_z_eqn_l31_31559

theorem condition_for_z_eqn
  (n : ℕ)
  (h : 2 ≤ n)
  (x : Fin (n + 1) → ℝ) -- x_0, x_1, ..., x_n
  (y : Fin (n + 1) → ℝ) -- y_0, y_1, ..., y_n
  (z : Fin (n + 1) → ℂ := λ k, ⟨x k, y k⟩) -- z_k = x_k + i*y_k
  : (z 0)^2 = ∑ i in Finset.finRange (n+1) \ {0}, (z i)^2 ↔ (x 0)^2 ≤ ∑ i in Finset.finRange (n+1) \ {0}, (x i)^2 := by
  sorry

end condition_for_z_eqn_l31_31559


namespace opposite_of_neg_one_over_2023_l31_31958

theorem opposite_of_neg_one_over_2023 : 
  ∃ x : ℚ, (-1 / 2023) + x = 0 ∧ x = 1 / 2023 :=
by
  use (1 / 2023)
  split
  · norm_num
  · refl

end opposite_of_neg_one_over_2023_l31_31958


namespace line_through_point_p_intersects_ellipse_l31_31329

open Real

theorem line_through_point_p_intersects_ellipse
    (a b : ℝ)
    (h1 : 0 < a^2 + b^2)
    (h2 : a^2 + b^2 < 3) :
    ∃ (m c : ℝ), {p : ℝ × ℝ | p.1 = m * p.2 + c} ∩ {p : ℝ × ℝ | (p.1)^2 / 4 + (p.2)^2 / 3 = 1} = 
    {p1, p2 : ℝ × ℝ | {p1, p2} ≠ ∅ ∧ p1 ≠ p2} := 
by
  sorry

end line_through_point_p_intersects_ellipse_l31_31329


namespace other_root_l31_31894

theorem other_root (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + m * x - 5 = 0 → (x = 1 ∨ x = -5 / 3)) :=
by {
  sorry
}

end other_root_l31_31894


namespace opposite_neg_fraction_l31_31953

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31953


namespace opposite_of_neg_frac_l31_31948

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31948


namespace sum_of_real_solutions_l31_31795

theorem sum_of_real_solutions (b : ℝ) (hb : b > 0) :
  ∃! x : ℝ, (sqrt (b - sqrt (b + 2 * x)) = x) ∧ x = sqrt (b - 1) - 1 :=
sorry

end sum_of_real_solutions_l31_31795


namespace sequence_condition_required_sum_l31_31983

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 4
  else ((n^2 : ℚ) / ((n-1)^2 : ℚ))

theorem sequence_condition (n : ℕ) (hn : n > 2) :
  (∏ i in Finset.range (n + 1), sequence (i + 1)) = (n+1)^2 := sorry

theorem required_sum : sequence 3 + sequence 5 = 61 / 16 := sorry

end sequence_condition_required_sum_l31_31983


namespace compute_N_l31_31111

def abs_value (a : ℝ) : ℝ :=
if a > 0 then a
else if a = 0 then 0
else -a

theorem compute_N : let N := abs_value 5 + abs_value (3 - 8) - abs_value (-4) in N = 6 :=
by
  sorry

end compute_N_l31_31111


namespace sandbox_perimeter_l31_31434

def sandbox_width : ℝ := 5
def sandbox_length := 2 * sandbox_width
def perimeter (length width : ℝ) := 2 * (length + width)

theorem sandbox_perimeter : perimeter sandbox_length sandbox_width = 30 := 
by
  sorry

end sandbox_perimeter_l31_31434


namespace maximum_rooks_10x10_l31_31208

-- Definitions (conditions)
def board_size : ℕ := 10

def is_attacked (board : List (List Bool)) (rook_pos : (ℕ × ℕ)) : List (List Bool) :=
  board.map.with_index (λ row i, row.map.with_index (λ cell j, cell || (i = rook_pos.1 || j = rook_pos.2)))

def rook_positions_valid (positions : List (ℕ × ℕ)) : Prop :=
  positions.length ≤ board_size ∧
  positions.all (λ pos, pos.1 < board_size ∧ pos.2 < board_size)

def attacked_board (positions : List (ℕ × ℕ)) : List (List Bool) :=
  positions.foldl (is_attacked) (List.replicate board_size (List.replicate board_size false))

def remaining_rook_conditions (positions : List (ℕ × ℕ)) : Prop :=
  ∀ pos ∈ positions,
    let new_positions := positions.erase pos in
    ∃ i j, attacked_board new_positions !! i !! j = false

-- The statement (translation)
theorem maximum_rooks_10x10 : ∃ k : ℕ, 
  (k ≤ board_size * board_size ∧ ∃ positions : List (ℕ × ℕ), rook_positions_valid positions ∧ remaining_rook_conditions positions ∧ k = 16) := sorry

end maximum_rooks_10x10_l31_31208


namespace material_for_7_quilts_l31_31845

theorem material_for_7_quilts (x : ℕ) (h1 : ∀ y : ℕ, y = 7 * x) (h2 : 36 = 12 * x) : 7 * x = 21 := 
by 
  sorry

end material_for_7_quilts_l31_31845


namespace hyperbola_other_asymptote_l31_31477

theorem hyperbola_other_asymptote 
    (asymptote1 : ∀ x : ℝ, asymptote1_eq x = 4 * x)
    (foci_x_coord : ∀ y : ℝ, foci_eq (3, y)):
    ∃ b : ℝ, ∀ x : ℝ, other_asymptote_eq x = -4 * x + b :=
by
  sorry

abbreviation asymptote1_eq (x : ℝ) := (4 : ℝ) * x
abbreviation foci_eq (c : ℝ × ℝ) := c.1 = 3
abbreviation other_asymptote_eq (x : ℝ) := -4 * x + 24

end hyperbola_other_asymptote_l31_31477


namespace min_value_of_frac_sum_l31_31352

noncomputable def min_frac_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) := 
  m + n = 3 → 
  ∃ z, z = (4 / m) + (1 / n) ∧ z = 3

theorem min_value_of_frac_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : m + n = 3) : 
  min_frac_sum m n hm hn := 
begin
  sorry
end

end min_value_of_frac_sum_l31_31352


namespace remainder_of_polynomial_division_l31_31278

-- Definitions of the conditions
def poly1 := x^3 - 1
def poly2 := x^6 - 1
def poly3 := x^2 - 1
def dividend := (x^6 - 1) * (x^2 - 1)

-- The theorem to be proven
theorem remainder_of_polynomial_division : dividend % poly1 = poly3 := by sorry

end remainder_of_polynomial_division_l31_31278


namespace nine_points_chords_l31_31064

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31064


namespace domain_and_range_of_g_l31_31628

variable (f : ℝ → ℝ)
variable (h_domain : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ∈ set.Icc (0:ℝ) (1:ℝ))

noncomputable def g (x : ℝ) : ℝ := 1 - f (x + 2)

theorem domain_and_range_of_g : 
 (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → g f h_domain x ∈ set.Icc (0:ℝ) (1:ℝ)) ∧ 
 (∀ y : ℝ → ℝ, y = g f h_domain → ∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → y x = g f h_domain x) :=
sorry

end domain_and_range_of_g_l31_31628


namespace tree_planting_activity_l31_31281

variables (trees_first_group trees_second_group people_first_group people_second_group : ℕ)
variable (average_trees_per_person_first_group average_trees_per_person_second_group : ℕ)

theorem tree_planting_activity :
  trees_first_group = 12 →
  trees_second_group = 36 →
  people_second_group = people_first_group + 6 →
  average_trees_per_person_first_group = trees_first_group / people_first_group →
  average_trees_per_person_second_group = trees_second_group / people_second_group →
  average_trees_per_person_first_group = average_trees_per_person_second_group →
  people_first_group = 3 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tree_planting_activity_l31_31281


namespace compute_M_value_l31_31670

theorem compute_M_value : 
  let M := (101^2 + 100^2 - 99^2 - 98^2 + 97^2 
           + 96^2 - 95^2 - 94^2 + 93^2 + 92^2
           - 91^2 - 90^2 + 89^2 + 88^2 - 87^2
           - 86^2 + 85^2 + 84^2 - 83^2 - 82^2
           + 81^2 + 80^2 - 79^2 - 78^2 + 77^2
           + 76^2 - 75^2 - 74^2 + 73^2 + 72^2
           - 71^2 - 70^2 + 69^2 + 68^2 - 67^2
           - 66^2 + 65^2 + 64^2 - 63^2 - 62^2
           + 61^2 + 60^2 - 59^2 - 58^2 + 57^2
           + 56^2 - 55^2 - 54^2 + 53^2 + 52^2
           - 51^2 - 50^2 + 49^2 + 48^2 - 47^2
           - 46^2 + 45^2 + 44^2 - 43^2 - 42^2
           + 41^2 + 40^2 - 39^2 - 38^2 + 37^2
           + 36^2 - 35^2 - 34^2 + 33^2 + 32^2
           - 31^2 - 30^2 + 29^2 + 28^2 - 27^2
           - 26^2 + 25^2 + 24^2 - 23^2 - 22^2
           + 21^2 + 20^2 - 19^2 - 18^2 + 17^2
           + 16^2 - 15^2 - 14^2 + 13^2 + 12^2
           - 11^2 - 10^2 + 9^2 + 8^2 - 7^2
           - 6^2 + 5^2 + 4^2 - 3^2 - 2^2) 
  in 
  M = 5304 := 
sorry

end compute_M_value_l31_31670


namespace reciprocal_of_sum_is_correct_l31_31180

theorem reciprocal_of_sum_is_correct : 1 / (\frac{1}{4} + \frac{1}{5}) = \frac{20}{9} :=
by
  sorry

end reciprocal_of_sum_is_correct_l31_31180


namespace reciprocal_of_negative_2023_l31_31125

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l31_31125


namespace eccentricity_range_of_hyperbola_l31_31776

noncomputable def eccentricity_range (m : ℝ) (h : -2 ≤ m ∧ m ≤ -1) : set ℝ :=
  {e | e = (sqrt (4 - m) / 2)}

theorem eccentricity_range_of_hyperbola :
  ∀ (m : ℝ) (h : -2 ≤ m ∧ m ≤ -1),
  eccentricity_range m h = set.Icc (sqrt 5 / 2) (sqrt 6 / 2) :=
  sorry

end eccentricity_range_of_hyperbola_l31_31776


namespace cosine_sum_formula_l31_31751

theorem cosine_sum_formula
  (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 4 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end cosine_sum_formula_l31_31751


namespace shortest_distance_to_circle_from_origin_l31_31568

-- Definitions derived from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 - 18 * x + y^2 - 8 * y + 153 = 0

-- Statement we need to prove
theorem shortest_distance_to_circle_from_origin :
  ∃ d : ℝ, d = (Real.sqrt 97 - Real.sqrt 44) ∧
  ∀ (x y : ℝ), circle_equation x y → Real.dist (0, 0) (9, 4) = Real.sqrt 97 ∧
  ∀ (x y : ℝ), circle_equation x y → Real.dist (0, 0) (x, y) = d + Real.sqrt 44 :=
sorry

end shortest_distance_to_circle_from_origin_l31_31568


namespace sum_distances_eq_nine_l31_31891

theorem sum_distances_eq_nine
  (r : ℝ) (triangle : Type) (A B C D : point ℝ)
  (equilateral : is_equilateral_triangle triangle A B C)
  (inscribed : is_inscribed_triangle r A B C)
  (AD_eq_9 : distance A D = 9) :
  distance B D + distance C D = 9 :=
sorry

end sum_distances_eq_nine_l31_31891


namespace sin_seven_pi_over_six_l31_31212

theorem sin_seven_pi_over_six : sin (7 * Real.pi / 6) = -1 / 2 := by
  sorry

end sin_seven_pi_over_six_l31_31212


namespace equal_roots_quadratic_l31_31760

theorem equal_roots_quadratic (k : ℝ) : (∃ (x : ℝ), x*(x + 2) + k = 0 ∧ ∀ y z, (y, z) = (x, x)) → k = 1 :=
sorry

end equal_roots_quadratic_l31_31760


namespace jellybean_guess_l31_31922

noncomputable def proof_problem (x : ℝ) : Prop :=
  let second_guess := 8 * x in
  let third_guess := 8 * x - 200 in
  let first_three_avg := (x + second_guess + third_guess) / 3 in
  let fourth_guess := first_three_avg + 25 in
  fourth_guess = 525 → x = 100

theorem jellybean_guess : ∃ x : ℝ, proof_problem x :=
begin
  use 100,
  unfold proof_problem,
  intro h,
  linarith,
end

end jellybean_guess_l31_31922


namespace tan_cos_addition_l31_31264

noncomputable def tan_30 : ℝ := 1 / real.sqrt 3
noncomputable def cos_30 : ℝ := real.sqrt 3 / 2
noncomputable def sin_30 : ℝ := 1 / 2

theorem tan_cos_addition  : tan_30 + 4 * cos_30 = (7 * real.sqrt 3) / 3 := 
by 
  sorry

end tan_cos_addition_l31_31264


namespace true_discount_double_time_l31_31383

theorem true_discount_double_time (FV PV : ℝ) (TD R T : ℝ) 
  (h₁ : FV = 110) 
  (h₂ : TD = 10)
  (h₃ : PV = FV - TD)
  (h₄ : TD = (PV * R * T) / 100)
  : 
  let TD' := (PV * R * (2 * T)) / 100 in
  TD' = 20 :=
by 
  sorry

end true_discount_double_time_l31_31383


namespace expected_value_bound_l31_31870

noncomputable theory

variables {R : ℝ × ℝ → ℝ} {X Y : ℝ → ℝ} [measure_space ℝ]

/-- R is a symmetric non-negative definite function on ℝ² -/
def symmetric_nonneg_definite (R : ℝ × ℝ → ℝ) : Prop :=
∀ x y : ℝ, R (x, y) = R (y, x) ∧ R (x, x) ≥ 0

/-- X is a random variable such that E[√R(X, X)] < ∞ -/
def random_variable_cond (R : ℝ × ℝ → ℝ) (X : ℝ → ℝ) [measure_space ℝ] : Prop :=
𝔼 (λ ω, (R ((X ω), (X ω))).sqrt) < ∞

/-- Y is an independent copy of the variable X -/
def independent_copy (X Y : ℝ → ℝ) [measure_space ℝ] : Prop :=
independent X Y

/-- Main Theorem -/
theorem expected_value_bound (R : ℝ × ℝ → ℝ) (X Y : ℝ → ℝ) [measure_space ℝ] 
  (h1 : symmetric_nonneg_definite R) 
  (h2 : random_variable_cond R X) 
  (h3 : independent_copy X Y) : 
  0 ≤ 𝔼 (λ ω, R ((X ω), (Y ω))) ∧ 𝔼 (λ ω, R ((X ω), (Y ω))) < ∞ := 
sorry

end expected_value_bound_l31_31870


namespace trig_identity_proof_l31_31580

theorem trig_identity_proof (α : ℝ) :
  cos (real.pi / 2 - 2 * α) * 
  (cos (real.pi / 6 - 2 * α) / sin (real.pi / 6 - 2 * α)) * 
  (sin (4 * real.pi / 3 - 2 * α) / cos (4 * real.pi / 3 - 2 * α)) * 
  (2 * cos (4 * α) - 1) = -sin (6 * α) :=
by
  sorry

end trig_identity_proof_l31_31580


namespace value_of_M_l31_31687

theorem value_of_M (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 :=
by
  sorry

end value_of_M_l31_31687


namespace nine_points_circle_chords_l31_31066

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31066


namespace probability_ge_3_l31_31163

open Set

def num_pairs (s : Set ℕ) (n : ℕ) : ℕ :=
  (s.Subset (Filter fun x y => abs (x - y) ≥ n)).Card

def undesirable_pairs : Set (ℕ × ℕ) :=
  {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), 
   (1,3), (2,4), (3,5), (4,6), (5,7), (6,8)}

def total_pairs : ℕ := choose 8 2

def prob_diff_ge_3 : ℚ :=
  1 - (undesirable_pairs.toList.length : ℚ) / total_pairs

theorem probability_ge_3 : prob_diff_ge_3 = 15 / 28 := by
  sorry

end probability_ge_3_l31_31163


namespace greatest_divisor_l31_31200

theorem greatest_divisor (n : ℕ) (h1 : 1428 % n = 9) (h2 : 2206 % n = 13) : n = 129 :=
sorry

end greatest_divisor_l31_31200


namespace find_line_equation_l31_31333

noncomputable def equation_of_line 
  (A : ℝ × ℝ) 
  (O_center : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_length : ℝ) 
  (k : ℝ) 
  (l : ℝ → ℝ) : Prop :=
  A = (4, 0) ∧ 
  O_center = (-3, 1) ∧ 
  radius = 2 ∧ 
  chord_length = 2 * sqrt 3 ∧ 
  k = 7 / 24 ∧ 
  l = λ (x : ℝ), k * x - 4 * k

theorem find_line_equation 
  (A O_center : ℝ × ℝ) 
  (radius chord_length k : ℝ) 
  (l : ℝ → ℝ) :
  equation_of_line A O_center radius chord_length k l →
  ∃ (k : ℝ), (A = (4, 0) ∧ O_center = (-3, 1) ∧ radius = 2 ∧ chord_length = 2 * sqrt 3 ∧ k ≠ 0) →
  (l = λ (x : ℝ), (7 * x - 28) / 24) :=
begin
  sorry
end

end find_line_equation_l31_31333


namespace probability_red_ball_10th_draw_l31_31815

-- Definitions for conditions in the problem
def total_balls : ℕ := 10
def red_balls : ℕ := 2

-- Probability calculation function
def probability_of_red_ball (total : ℕ) (red : ℕ) : ℚ :=
  red / total

-- Theorem statement: Given the conditions, the probability of drawing a red ball on the 10th attempt is 1/5
theorem probability_red_ball_10th_draw :
  probability_of_red_ball total_balls red_balls = 1 / 5 :=
by
  sorry

end probability_red_ball_10th_draw_l31_31815


namespace reciprocal_of_negative_2023_l31_31127

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l31_31127


namespace smallest_positive_integer_23n_mod_5678_mod_11_l31_31998

theorem smallest_positive_integer_23n_mod_5678_mod_11 :
  ∃ n : ℕ, 0 < n ∧ 23 * n % 11 = 5678 % 11 ∧ ∀ m : ℕ, 0 < m ∧ 23 * m % 11 = 5678 % 11 → n ≤ m :=
by
  sorry

end smallest_positive_integer_23n_mod_5678_mod_11_l31_31998


namespace range_of_m_l31_31367

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2*x + m = 0}

theorem range_of_m (m : ℝ) : (A ∪ B m = A) ↔ m ∈ Set.Ici 1 :=
by
  sorry

end range_of_m_l31_31367


namespace count_double_digit_sum_to_three_is_ten_l31_31026

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def double_digit_sum_to_three (count : ℕ) : Prop :=
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 → 
    (sum_of_digits (sum_of_digits x) = 3 → x ∞∣∑_x count ∧) → count = 10 

theorem count_double_digit_sum_to_three_is_ten : 
  ∃ count, double_digit_sum_to_three count ∧ count = 10 := sorry

end count_double_digit_sum_to_three_is_ten_l31_31026


namespace segment_CM_length_eq_sqrt_13_l31_31507

open Real

def side_length : ℝ := 3
def square_area := side_length * side_length
def part_area := square_area / 3
def height_BM := 2 -- Derived from the problem context

theorem segment_CM_length_eq_sqrt_13 :
  ∃ CM : ℝ, CM = sqrt (side_length^2 + height_BM^2) ∧ CM = sqrt 13 :=
begin
  use sqrt (side_length^2 + height_BM^2),
  split,
  { refl },
  { norm_num }
end

end segment_CM_length_eq_sqrt_13_l31_31507


namespace probability_positive_difference_ge_three_l31_31166

open Finset Nat

theorem probability_positive_difference_ge_three :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (s.card.choose 2)
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  probability = 15 / 28 :=
by
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := s.card.choose 2
  have total_pairs_eq : total_pairs = 28 := by decide
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  have favorable_pairs_eq : favorable_pairs = 15 := by decide
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  have probability_eq : probability = 15 / 28 := by
    rw [favorable_pairs_eq, total_pairs_eq, ←nat_cast_add, ←rat.div_eq_div_iff]
    norm_num
  exact probability_eq

end probability_positive_difference_ge_three_l31_31166


namespace base7_number_divisibility_l31_31113

theorem base7_number_divisibility (x : ℕ) (h : 0 ≤ x ∧ x ≤ 6) :
  (5 * 343 + 2 * 49 + x * 7 + 4) % 29 = 0 ↔ x = 6 := 
by
  sorry

end base7_number_divisibility_l31_31113


namespace find_m_of_conditions_l31_31762

-- Define the power function f(x) = x^α
def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

-- State the conditions
axiom cond1 : ∃ α : ℝ, f 2 α = Real.sqrt 2
axiom cond2 : ∃ m : ℝ, f m (1 / 2) = 4

-- Prove that m = 16 given the conditions
theorem find_m_of_conditions : 
  ∃ m : ℝ, (∃ α : ℝ, f 2 α = Real.sqrt 2) ∧ (f m (1/2) = 4) → m = 16 := 
by
  sorry

end find_m_of_conditions_l31_31762


namespace final_balance_is_60_million_l31_31620

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l31_31620


namespace matrix_calculation_l31_31861

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l31_31861


namespace number_of_students_playing_cricket_l31_31091

theorem number_of_students_playing_cricket
    (total_students : ℕ)
    (students_play_football : ℕ)
    (students_play_both : ℕ)
    (students_neither : ℕ)
    (C : ℕ) :
    total_students = 410 →
    students_play_football = 325 →
    students_play_both = 140 →
    students_neither = 50 →
    C = 175 :=
begin
    sorry
end

end number_of_students_playing_cricket_l31_31091


namespace difference_of_squares_550_450_l31_31189

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end difference_of_squares_550_450_l31_31189


namespace sum_of_even_term_coefficients_rational_terms_expansion_correct_l31_31336

noncomputable def binomial_sum_even_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem sum_of_even_term_coefficients
  (n : ℕ) (h : n * (n - 1) = 72):
  binomial_sum_even_terms n = 256 :=
sorry

noncomputable def rational_terms_in_expansion (n r : ℕ) (x : ℝ) : ℝ :=
  if r = 0 then x^3 else
  if r = 3 then -672 * x^(-1) else
  if r = 6 then -5376 * x^(-5) else
  if r = 9 then -512 * x^(-9) else 0

theorem rational_terms_expansion_correct
  (x : ℝ):
  {r | rational_terms_in_expansion 9 r x ≠ 0} = {x^3, -672 * x^(-1), -5376 * x^(-5), -512 * x^(-9)} :=
sorry

end sum_of_even_term_coefficients_rational_terms_expansion_correct_l31_31336


namespace problem_I_solution_set_problem_II_range_a_l31_31269

-- Problem (I)
-- Given f(x) = |x-1|, g(x) = 2|x+1|, and a=1, prove that the inequality f(x) - g(x) > 1 has the solution set (-1, -1/3)
theorem problem_I_solution_set (x: ℝ) : abs (x - 1) - 2 * abs (x + 1) > 1 ↔ -1 < x ∧ x < -1 / 3 := 
by sorry

-- Problem (II)
-- Given f(x) = |x-1|, g(x) = 2|x+a|, prove that if 2f(x) + g(x) ≤ (a + 1)^2 has a solution for x,
-- then a ∈ (-∞, -3] ∪ [1, ∞)
theorem problem_II_range_a (a x: ℝ) (h : ∃ x, 2 * abs (x - 1) + 2 * abs (x + a) ≤ (a + 1) ^ 2) : 
  a ≤ -3 ∨ a ≥ 1 := 
by sorry

end problem_I_solution_set_problem_II_range_a_l31_31269


namespace elvis_ralph_matchsticks_l31_31694

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l31_31694


namespace area_triangle_QPO_is_quarter_of_trapezoid_area_l31_31002

variables {A B C D P Q O : Type}
variables [has_area ABCD : ℝ] [has_area QPO : ℝ]

-- Define the conditions
def is_trapezoid (ABCD : ℝ) (AB CD : ℝ) : Prop :=
  AB = 2 * CD ∧ -- AB and CD are proportional as defined
  -- Medians DP and CQ bisects the trapezoid at points P, Q, and intersect at O.

def area_of_trapezoid (trapezoid_area : ℝ) : ℝ :=
  trapezoid_area

def area_of_triangle (triangle_area : ℝ) : ℝ :=
  triangle_area

-- Lean statement requiring proof
theorem area_triangle_QPO_is_quarter_of_trapezoid_area
  (ABCD : ℝ) (AB CD : ℝ) (k : ℝ) 
  (h : is_trapezoid ABCD AB CD) : 
  area_of_triangle QPO = k / 4 :=
  sorry

end area_triangle_QPO_is_quarter_of_trapezoid_area_l31_31002


namespace sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l31_31122

noncomputable def sum_of_consecutive_triplets (a : Fin 12 → ℕ) (i : Fin 12) : ℕ :=
a i + a ((i + 1) % 12) + a ((i + 2) % 12)

theorem sum_of_consecutive_at_least_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i ≥ 20 :=
by
  sorry

theorem sum_of_consecutive_greater_than_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i > 20 :=
by
  sorry

end sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l31_31122


namespace chords_from_nine_points_l31_31074

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31074


namespace propositionA_necessary_for_propositionB_l31_31672

variable {α : Type*} [TopologicalSpace α] [NormedGroup α] [NormedSpace ℝ α]
variable {f : ℝ → α}
variable {x : ℝ}

-- Proposition A: f'(x) = 0
def propA (f : ℝ → α) (x : ℝ) : Prop :=
  deriv f x = 0

-- Proposition B: f(x) has an extremum at x = c
def has_extremum (f : ℝ → α) (c : ℝ) : Prop :=
  ∃ x, (((∀ y, x < y → f y ≥ f x) ∨ (∀ y, y < x → f y ≥ f x)) ∨
        ((∀ y, x < y → f y ≤ f x) ∨ (∀ y, y < x → f y ≤ f x))) 

theorem propositionA_necessary_for_propositionB (f : ℝ → α) (c : ℝ) :
  has_extremum f c → propA f c :=
by
  intros h,
  sorry

end propositionA_necessary_for_propositionB_l31_31672


namespace minimum_point_transformed_function_l31_31930

def original_function (x : ℝ) : ℝ := 2 * |x| - 3

def transformed_function (x : ℝ) : ℝ := 2 * |x + 3| - 7

theorem minimum_point_transformed_function :
  ∃ (x y : ℝ), x = -3 ∧ y = -7 ∧ transformed_function x = y :=
by
  -- Exists the tuple representing the minimum point of the new graph.
  use (-3), (-7)
  -- State the conditions for the x and y values and the equation.
  simp [transformed_function]
  sorry

end minimum_point_transformed_function_l31_31930


namespace length_of_DB_l31_31416

open EuclideanGeometry

theorem length_of_DB (A B C D : Point) (angle_RIGHT : ∀ X Y Z, ∠X Y Z = 90° )
  (h1 : ∠A B C = 90°) (h2 : ∠A D B = 90°) (h3 : dist A C = 20) (h4 : dist A D = 4) :
  dist D B = 8 := 
by sorry

end length_of_DB_l31_31416


namespace area_PQR_greater_than_2_over_9_area_ABC_l31_31895

theorem area_PQR_greater_than_2_over_9_area_ABC
  (A B C P Q R : Type) 
  [triangle A B C] 
  (P_on_AB : P ∈ segment A B) 
  (Q_on_AB : Q ∈ segment A B) 
  (R_on_AC : R ∈ segment A C) 
  (perimeter_division : divides_perimeter_eq A B C P Q R) :
    area (triangle P Q R) > (2 / 9) * area (triangle A B C) :=
sorry

end area_PQR_greater_than_2_over_9_area_ABC_l31_31895


namespace tims_initial_cans_l31_31544
noncomputable theory

-- Definitions extracted from conditions
def initial_cans (x : ℕ) : ℕ := x
def after_jeff (x : ℕ) : ℕ := x - 6
def after_buying_more (x : ℕ) : ℕ := after_jeff x + (after_jeff x / 2)

-- Statement of the problem
theorem tims_initial_cans (x : ℕ) (h : after_buying_more x = 24) : x = 22 :=
by
  sorry

end tims_initial_cans_l31_31544


namespace points_symmetric_orthocenter_lie_on_circumcircle_l31_31486

open EuclideanGeometry

noncomputable def symmetric_point (H : Point) (P Q : Line) : Point := sorry

theorem points_symmetric_orthocenter_lie_on_circumcircle (ABC : Triangle) (H : Point) (A1 B1 C1 : Point)
  (hH_to_BC : A1 = symmetric_point H ABC.BC)
  (hH_to_CA : B1 = symmetric_point H ABC.CA)
  (hH_to_AB : C1 = symmetric_point H ABC.AB)
  (h_pepend1 : ⊥_on_line ABC.AB H ABC.CH)
  (h_pepend2 : ⊥_on_line ABC.BC H ABC.AH)
  (isosceles_AC1H : isosceles_triangle ABC.A H C1): 

  PointsOnCircumcircle A1 B1 C1 ABC := sorry

end points_symmetric_orthocenter_lie_on_circumcircle_l31_31486


namespace sum_sines_tangent_l31_31323

/-- Given the sum of sines of multiples of 7 degrees from 1 to 25 is equal to the 
    tangent of the fraction p/q, where p and q are relatively prime positive 
    integers satisfying p/q < 90 degrees, prove that p + q = 6. -/
theorem sum_sines_tangent (p q : ℕ) (hpq_coprime : Nat.coprime p q) 
  (hpq_lt_90 : (p : ℝ) / q < 90) :
  (∑ k in Finset.range 25, Real.sin (7 * (k + 1) * Real.pi / 180)) = Real.tan (p * Real.pi / (q * 180))
  → p + q = 6 :=
by
  assume H
  sorry

end sum_sines_tangent_l31_31323


namespace find_100th_term_l31_31678

noncomputable def b (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, Real.cos (k + 1)

theorem find_100th_term (h : ∀ n : ℕ, b n > 0.5) : ∃! n : ℕ, n = 629 ∧ b 629 > 0.5 :=
begin
  sorry
end

end find_100th_term_l31_31678


namespace angleC_equals_40_of_angleA_40_l31_31818

-- Define an arbitrary quadrilateral type and its angle A and angle C
structure Quadrilateral :=
  (angleA : ℝ)  -- angleA is in degrees
  (angleC : ℝ)  -- angleC is in degrees

-- Given condition in the problem
def quadrilateral_with_A_40 : Quadrilateral :=
  { angleA := 40, angleC := 0 } -- Initialize angleC as a placeholder

-- Theorem stating the problem's claim
theorem angleC_equals_40_of_angleA_40 :
  quadrilateral_with_A_40.angleA = 40 → quadrilateral_with_A_40.angleC = 40 :=
by
  sorry  -- Proof is omitted for brevity

end angleC_equals_40_of_angleA_40_l31_31818


namespace expected_pairs_socks_l31_31152

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l31_31152


namespace equation_of_tangent_line_l31_31342

noncomputable def f (x : ℝ) : ℝ := 2 * x - 2 / x - 2 * Real.log x

theorem equation_of_tangent_line :
  let x0 := 1
      p0 := (1 : ℝ, f 1)
      m := (deriv f 1)
  in m = 2 ∧ p0 = (1, 0) ∧ ∀ x, f 1 + m * (x - 1) = 2 * x - 2 :=
by
  let x0 := 1
  let p0 := (1 : ℝ, f 1)
  let m := (deriv f 1)
  show m = 2 ∧ p0 = (1, 0) ∧ ∀ x, f 1 + m * (x - 1) = 2 * x - 2
  sorry

end equation_of_tangent_line_l31_31342


namespace T_shaped_tiling_impossible_l31_31481

theorem T_shaped_tiling_impossible :
  ¬ ∃ (f : Fin (10 * 10) → Fin (10 * 10) → Prop), 
    (∀ i j, (f i j) → (∃ b : Fin 4, 
      T_shaped (i, j) b ∧ 
      ∀ c : Fin 4, 
        if c = b then True 
        else f (next_i i c) (next_j j c))) := 
sorry

end T_shaped_tiling_impossible_l31_31481


namespace projection_vector_l31_31370

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (h1 : ∥b∥ = 1) (h2 : ⟪a, b⟫ = 0)

theorem projection_vector :
  (proj b (a - 2 • b)) = -2 • b :=
by
  sorry

end projection_vector_l31_31370


namespace paco_ate_15_sweet_cookies_l31_31896

theorem paco_ate_15_sweet_cookies (initial_sweet_cookies sweet_cookies_left : ℕ) (hyp1 : initial_sweet_cookies = 22) (hyp2 : sweet_cookies_left = 7) : 
  initial_sweet_cookies - sweet_cookies_left = 15 := 
by 
  -- Using the given conditions
  rw [hyp1, hyp2]
  -- Performing the calculation
  exact Nat.sub_eq_iff_eq_add.mpr rfl

end paco_ate_15_sweet_cookies_l31_31896


namespace johns_money_left_l31_31431

noncomputable def johns_initial_amount : ℝ := 200
noncomputable def fraction_given_to_mother : ℝ := 3 / 8
noncomputable def fraction_given_to_father : ℝ := 3 / 10

noncomputable def amount_given_to_mother := fraction_given_to_mother * johns_initial_amount
noncomputable def amount_given_to_father := fraction_given_to_father * johns_initial_amount
noncomputable def total_given := amount_given_to_mother + amount_given_to_father
noncomputable def amount_left := johns_initial_amount - total_given

theorem johns_money_left : amount_left = 65 :=
by
    have h_mother : amount_given_to_mother = 75 := by norm_num [amount_given_to_mother, fraction_given_to_mother, johns_initial_amount]
    have h_father : amount_given_to_father = 60 := by norm_num [amount_given_to_father, fraction_given_to_father, johns_initial_amount]
    have h_total : total_given = 135 := by norm_num [total_given, h_mother, h_father]
    show amount_left = 65 from by norm_num [amount_left, h_total, johns_initial_amount]

end johns_money_left_l31_31431


namespace height_difference_l31_31538

variable (S J H B : ℕ)

theorem height_difference :
  J = S + 207 → 
  H = S + 252 → 
  B = J + 839 → 
  B - S = 1046 := by
  intros h1 h2 h3
  rw [h1] at h3
  rw [← add_assoc] at h3
  linarith

#eval height_difference

end height_difference_l31_31538


namespace chords_from_nine_points_l31_31076

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31076


namespace chord_length_l31_31000

-- Definition of the curve and the line in polar coordinates.
def curve (θ : ℝ) : ℝ := 4 * Real.sin θ
def line (θ : ℝ) : Prop := θ = Real.pi / 4

-- The length of the chord cut by the curve on the line.
theorem chord_length :
  ∀ θ (ρ : ℝ), curve θ = ρ → line θ → (ρ = 4 * Real.sin (Real.pi / 4)) →
  Real.sqrt ((0 - 2)^2 + (0 - 2)^2) = 2 * Real.sqrt 2 := sorry

end chord_length_l31_31000


namespace smallest_positive_period_l31_31341

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := 
sorry

end smallest_positive_period_l31_31341


namespace unique_zero_iff_a_in_range_l31_31347

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_zero_iff_a_in_range (a : ℝ) :
  (∃ x0 : ℝ, f a x0 = 0 ∧ (∀ x1 : ℝ, f a x1 = 0 → x1 = x0) ∧ x0 > 0) ↔ a < -2 :=
by sorry

end unique_zero_iff_a_in_range_l31_31347


namespace part_a_part_b_l31_31866

-- Definition of the problem's conditions
variables {F1 F2 F3 : Type}
variables {l1 l2 l3 : Type}
variables {W : Type}
variables {J1 J2 J3 : Type}
variables (circle_of_similarity : Type)

-- Theorem for Part (a)
theorem part_a :
  (l1 := F1) → (l2 := F2) → (l3 := F3) → 
  (W ∈ l1 ∧ W ∈ l2 ∧ W ∈ l3) → 
  (W ∈ circle_of_similarity) :=
sorry

-- Theorem for Part (b)
theorem part_b :
  (l1 := F1) → (l2 := F2) → (l3 := F3) → 
  (J1 ∈ l1 ∧ J1 ∈ circle_of_similarity ∧ J1 ≠ W) →
  (J2 ∈ l2 ∧ J2 ∈ circle_of_similarity ∧ J2 ≠ W) →
  (J3 ∈ l3 ∧ J3 ∈ circle_of_similarity ∧ J3 ≠ W) →
  (∀ l1' l2' l3', 
     (l1' := F1) → (l2' := F2) → (l3' := F3) →
     (J1 = J1' ∧ J2 = J2' ∧ J3 = J3')) :=
sorry

end part_a_part_b_l31_31866


namespace eugene_total_pencils_l31_31282

-- Define the initial number of pencils Eugene has
def initial_pencils : ℕ := 51

-- Define the number of pencils Joyce gives to Eugene
def pencils_from_joyce : ℕ := 6

-- Define the expected total number of pencils
def expected_total_pencils : ℕ := 57

-- Theorem to prove the total number of pencils Eugene has
theorem eugene_total_pencils : initial_pencils + pencils_from_joyce = expected_total_pencils := 
by sorry

end eugene_total_pencils_l31_31282


namespace nine_points_circle_chords_l31_31069

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31069


namespace odd_numbers_divisibility_l31_31482

theorem odd_numbers_divisibility 
  (a b c : ℤ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : (ab - 1) % 4 = 0 ∨ (bc - 1) % 4 = 0 ∨ (ca - 1) % 4 = 0 := 
sorry

end odd_numbers_divisibility_l31_31482


namespace maximize_sequence_l31_31777

theorem maximize_sequence (n : ℕ) (an : ℕ → ℝ) (h : ∀ n, an n = (10/11)^n * (3 * n + 13)) : 
  (∃ n_max, (∀ m, an m ≤ an n_max) ∧ n_max = 6) :=
by
  sorry

end maximize_sequence_l31_31777


namespace decreasing_interval_cos_2theta_l31_31338

noncomputable theory
open Real

def f (x : ℝ) : ℝ := 4 * sin x * (cos x - sin x) + 3

theorem decreasing_interval (x : ℝ) (h1 : 0 < x ∧ x < π) : 
  ((π / 8 : ℝ) ≤ x ∧ x ≤ (5 * π / 8 : ℝ)) ↔ (deriv f x < 0) :=
sorry

theorem cos_2theta (θ : ℝ) (h2 : 0 ≤ θ ∧ θ ≤ π) (h3 : range (λ x : ℝ, f x) = set.Icc 0 (2 * sqrt 2 + 1)) :
  cos (2 * θ) = - (sqrt 7 + 1) / 4 :=
sorry

end decreasing_interval_cos_2theta_l31_31338


namespace safety_rent_a_car_cost_per_mile_l31_31097

/-
Problem:
Prove that the cost per mile for Safety Rent-a-Car is 0.177 dollars, given that the total cost of renting an intermediate-size car for 150 miles is the same for Safety Rent-a-Car and City Rentals, with their respective pricing schemes.
-/

theorem safety_rent_a_car_cost_per_mile :
  let x := 21.95
  let y := 18.95
  let z := 0.21
  (x + 150 * real_safety_per_mile) = (y + 150 * z) ↔ real_safety_per_mile = 0.177 :=
by
  sorry

end safety_rent_a_car_cost_per_mile_l31_31097


namespace probability_ge_3_l31_31161

open Set

def num_pairs (s : Set ℕ) (n : ℕ) : ℕ :=
  (s.Subset (Filter fun x y => abs (x - y) ≥ n)).Card

def undesirable_pairs : Set (ℕ × ℕ) :=
  {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), 
   (1,3), (2,4), (3,5), (4,6), (5,7), (6,8)}

def total_pairs : ℕ := choose 8 2

def prob_diff_ge_3 : ℚ :=
  1 - (undesirable_pairs.toList.length : ℚ) / total_pairs

theorem probability_ge_3 : prob_diff_ge_3 = 15 / 28 := by
  sorry

end probability_ge_3_l31_31161


namespace short_sleeves_count_l31_31537

variables (S L : ℕ)

theorem short_sleeves_count :
  S + L = 36 ∧ L - S = 24 → S = 6 :=
by
  intro h
  cases' h with h1 h2
  -- Introduce equations from h1 and h2
  have h3 : 2 * L = 60 :=
    calc
      2 * (L) = (S + L) + (L - S) : by linarith
      ... = 36 + 24 : by linarith
      ... = 60 : by linarith
  -- Solve for L
  have h4 : L = 30 := by linarith
  -- Substitute back to find S
  have h5 : S = 36 - L := by linarith
  rw [h4] at h5
  assumption

end short_sleeves_count_l31_31537


namespace isosceles_right_triangle_exists_l31_31862

noncomputable def is_isosceles_right_triangle (P Q R : Point) : Prop :=
  let dPQ := dist P Q
  let dPR := dist P R
  let dQR := dist Q R
  dPQ = dPR ∧ dQR = (sqrt 2) * dPQ

theorem isosceles_right_triangle_exists
  (A B C D E M : Point)
  (ABC_isosceles_right : is_isosceles_right_triangle A B C)
  (ADE_isosceles_right : is_isosceles_right_triangle A D E)
  (non_congruent : ¬ (is_congruent_triangle A B C A D E))
  (AC_gt_AE : dist A C > dist A E)
  (rotation : ∀ θ : ℝ, rotate (A, E) θ = (triangle A D E)) :
  ∃ M : Point, on_segment E C M ∧ is_isosceles_right_triangle B M D :=
sorry

end isosceles_right_triangle_exists_l31_31862


namespace Miss_Adamson_paper_usage_l31_31465

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l31_31465


namespace problem_f_2017_l31_31768

noncomputable def f : ℝ → ℝ
| x := if h : 0 ≤ x ∧ x ≤ 2 then sin (real.pi * x / 6) else 2 * f (x - 2)

theorem problem_f_2017 :
  f 2017 = 2^1007 :=
sorry

end problem_f_2017_l31_31768


namespace train_speed_l31_31646

-- Define the lengths and time as given in the conditions
def length_train : ℝ := 110
def length_bridge : ℝ := 340
def time_crossing : ℝ := 26.997840172786177

-- Define the total distance covered
def total_distance : ℝ := length_train + length_bridge

-- Define the expected speed
def expected_speed : ℝ := total_distance / time_crossing

-- The theorem stating that the speed of the train is 16.669 m/s
theorem train_speed :
  expected_speed = 450 / 26.997840172786177 :=
sorry

end train_speed_l31_31646


namespace number_is_18_l31_31554

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end number_is_18_l31_31554


namespace least_perimeter_of_triangle_l31_31530

theorem least_perimeter_of_triangle (a b : ℕ) (a_eq : a = 33) (b_eq : b = 42) (c : ℕ) (h1 : c + a > b) (h2 : c + b > a) (h3 : a + b > c) : a + b + c = 85 :=
sorry

end least_perimeter_of_triangle_l31_31530


namespace all_numbers_are_rational_l31_31691

noncomputable def is_rational (x : ℚ) := x

theorem all_numbers_are_rational :
  is_rational (Real.sqrt (2^2)) ∧
  is_rational (Real.cbrt 0.512) ∧
  is_rational (Real.sqrt ((0.04)⁻¹)) ∧
  is_rational (Real.cbrt (-8)) ∧
  is_rational (Real.cbrt (-8) * Real.sqrt ((0.04)⁻¹)) :=
by
  split, sorry,
  split, sorry,
  split, sorry,
  split, sorry, sorry

end all_numbers_are_rational_l31_31691


namespace cement_used_tess_street_l31_31903

-- Define the given conditions
def cement_used_lexi_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Define the statement to prove the amount of cement used for Tess's street
theorem cement_used_tess_street : total_cement_used - cement_used_lexi_street = 5.1 :=
by
  sorry

end cement_used_tess_street_l31_31903


namespace cube_volume_from_surface_area_l31_31976

theorem cube_volume_from_surface_area (s : ℝ) (h : 6 * s^2 = 54) : s^3 = 27 :=
sorry

end cube_volume_from_surface_area_l31_31976


namespace jenna_age_l31_31844

theorem jenna_age (D J : ℕ) (h1 : J = D + 5) (h2 : J + D = 21) (h3 : D = 8) : J = 13 :=
by
  sorry

end jenna_age_l31_31844


namespace tan_30_deg_l31_31742

theorem tan_30_deg (x y : ℝ) (h : (x, y) ≠ (0, 0)) (h_angle : true) : y / x = sqrt 3 / 3 :=
by
  sorry

end tan_30_deg_l31_31742


namespace find_G_14_l31_31444

noncomputable def G (x : ℝ) : ℝ := sorry

lemma G_at_7 : G 7 = 20 := sorry

lemma functional_equation (x : ℝ) (hx: x ^ 2 + 8 * x + 16 ≠ 0) : 
  G (4 * x) / G (x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16) := sorry

theorem find_G_14 : G 14 = 96 := sorry

end find_G_14_l31_31444


namespace volume_ratio_tetrahedron_l31_31226

/-- Given a regular tetrahedron ABCD, a plane perpendicular to an edge AB, and a point K that divides AB in the
    ratio 1:4, the ratio of the volumes of the resulting parts is 4:121. -/
theorem volume_ratio_tetrahedron (A B C D K : Point) (h_tetra : regular_tetrahedron A B C D)
(h_perpendicular_plane : plane_perpendicular_to_edge (A, B) K)
(h_division_ratio : divides_edge_in_ratio (A, B) K (1/5) (4/5)) :
ratio_of_volumes (tetrahedron_part1 A B C D K) (tetrahedron_part2 A B C D K) = 4 / 121 :=
by
  sorry

end volume_ratio_tetrahedron_l31_31226


namespace find_smallest_prime_l31_31320

open Nat

-- Define the prime numbers and natural numbers according to the conditions
variables {a b c : ℕ}

-- Attack the main problem definition
theorem find_smallest_prime 
  (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h2 : Prime a) 
  (h3 : Prime b) 
  (h4 : Prime c)
  (h5 : Prime (a + b - c ))
  (h6 : Prime (a + c - b ))
  (h7 : Prime (b + c - a ))
  (h8 : Prime (a + b + c )) 
  : min (min (min (min (min (min a b) c) (a + b - c)) (a + c - b)) (b + c - a)) (a + b + c) = 3 :=
sorry

end find_smallest_prime_l31_31320


namespace pure_alcohol_to_add_l31_31582

def initial_volume := 6
def initial_percentage := 0.25
def final_percentage := 0.50

theorem pure_alcohol_to_add (x : ℝ) :
  (initial_percentage * initial_volume + x) / (initial_volume + x) = final_percentage ↔ x = 3 :=
by
  sorry

end pure_alcohol_to_add_l31_31582


namespace eval_sum_tan_equals_half_of_89_l31_31702

noncomputable def eval_sum_tan : ℝ :=
  ∑ x in (finset.range 89).map (λ x, x+1), 1 / (1 + real.tan (x * real.pi / 180))

theorem eval_sum_tan_equals_half_of_89 : eval_sum_tan = 89/2 :=
by
  sorry

end eval_sum_tan_equals_half_of_89_l31_31702


namespace simplify_expression_l31_31699

noncomputable def sqrt' (x : ℝ) : ℝ := Real.sqrt x

theorem simplify_expression :
  (3 * sqrt' 8 / (sqrt' 2 + sqrt' 3 + sqrt' 7)) = (sqrt' 2 + sqrt' 3 - sqrt' 7) := 
by
  sorry

end simplify_expression_l31_31699


namespace probability_of_selecting_cubes_l31_31615

/-- Define the probabilities and conditions for unit cubes in the larger cube -/
def unitCubes : ℕ := 125
def doublePainted : ℕ := 8
def unpainted : ℕ := 83
def totalWays : ℕ := (unitCubes * (unitCubes - 1)) / 2
def successfulOutcomes : ℕ := doublePainted * unpainted
def probability := rat.mk successfulOutcomes totalWays

/-- Probability that one of two selected unit cubes has exactly two painted faces while
 the other unit cube has no painted faces -/
theorem probability_of_selecting_cubes :
  probability = rat.ofInt 332 / rat.ofInt 3875 :=
sorry

end probability_of_selecting_cubes_l31_31615


namespace find_x_solutions_l31_31289

theorem find_x_solutions (x : ℝ) (h : (9^x + 32^x) / (15^x + 24^x) = 4 / 3) : x = -1 ∨ x = 1 :=
sorry

end find_x_solutions_l31_31289


namespace smallest_integral_area_of_circle_l31_31918

noncomputable def A (r : ℝ) : ℝ := π * r^2

noncomputable def C (r : ℝ) : ℝ := 2 * π * r

lemma integral_area_greater_than_circumference (r : ℝ) : (π * r^2 > 2 * π * r) → r > 2 :=
begin
  intro h,
  have h1 : r * (r - 2) > 0,
  { rw mul_comm at h,
    exact (lt_div_iff (pi_pos)).mp ((div_lt_iff (pi_pos)).mpr h) },
  exact gt_of_not_le (@not_or_distrib ℝ r 0 2).mpr (not_le_of_gt h1)
end

theorem smallest_integral_area_of_circle : ∃ r : ℝ, r > 2 ∧ (28 < π * r^2 ∧ π * r^2 < 30) :=
begin
  use 3,
  split,
  { linarith },
  split,
  { linarith [pi_pos],
    norm_num,
    linarith [real.pi_pos] },
  { norm_num,
    linarith }
end

end smallest_integral_area_of_circle_l31_31918


namespace positive_real_solutions_count_eq_two_l31_31377

def P (x : ℝ) : ℝ := x^10 - 4*x^9 + 6*x^8 + 878*x^7 - 3791*x^6

theorem positive_real_solutions_count_eq_two : 
  ∃ S : set ℝ, (∀ x ∈ S, 0 < x) ∧ (∀ x, x ∈ S → P x = 0) ∧ card S = 2 :=
sorry

end positive_real_solutions_count_eq_two_l31_31377


namespace exists_a_bc_l31_31197

-- Definitions & Conditions
def satisfies_conditions (a b c : ℤ) : Prop :=
  - (b + c) - 10 = a ∧ (b + 10) * (c + 10) = 1

-- Theorem Statement
theorem exists_a_bc : ∃ (a b c : ℤ), satisfies_conditions a b c := by
  -- Substitute the correct proof below
  sorry

end exists_a_bc_l31_31197


namespace intersection_correct_l31_31020

variable (x : ℝ)

def M : Set ℝ := { x | x^2 > 4 }
def N : Set ℝ := { x | x^2 - 3 * x ≤ 0 }
def NM_intersection : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem intersection_correct :
  {x | (M x) ∧ (N x)} = NM_intersection :=
sorry

end intersection_correct_l31_31020


namespace magic_triangle_max_S_l31_31405

theorem magic_triangle_max_S :
  ∃ (S : ℕ), (∀ (a b c d e f g h i : ℕ), 
    {a, b, c, d, e, f, g, h, i} = {10, 11, 12, 13, 14, 15, 16, 17, 18} ∧
    a + b + c = S ∧
    d + e + f = S ∧
    g + h + i = S) ∧
  S = 42 :=
by sorry

end magic_triangle_max_S_l31_31405


namespace div_eq_of_scaled_div_eq_l31_31794

theorem div_eq_of_scaled_div_eq (h : 29.94 / 1.45 = 17.7) : 2994 / 14.5 = 17.7 := 
by
  sorry

end div_eq_of_scaled_div_eq_l31_31794


namespace measure_angle_YNM_l31_31420

-- Define the triangle \( \triangle XYZ \)
variables {X Y Z M N : Type}
variables (angle X angle Z : ℝ) (Y M N : X)

-- Condition statements in Lean
def angle_X := 70
def angle_Z := 50
def MY_eq_YN : Bool := M = N -- MY = YN simplified as point equality for this particular Lean proof

-- Theorem: Prove the measure of \( \angle YNM \)
theorem measure_angle_YNM (h : MY_eq_YN) : (angle X + angle Z + 60 = 180) :=
  sorry

end measure_angle_YNM_l31_31420


namespace miles_hiked_first_day_l31_31987

theorem miles_hiked_first_day (total_distance remaining_distance : ℕ)
  (h1 : total_distance = 36)
  (h2 : remaining_distance = 27) :
  total_distance - remaining_distance = 9 :=
by
  sorry

end miles_hiked_first_day_l31_31987


namespace venki_trip_time_l31_31172

theorem venki_trip_time :
  let distance_from_speed_time (speed : ℝ) (time : ℝ) := speed * time in
  let midway (d : ℝ) := d / 2 in
  let total_distance := (distance_from_speed_time 45 4.444444444444445) * 2 in
  let travel_time (d : ℝ) (s : ℝ) := d / s in
  travel_time total_distance 80 = 5 :=
by
  let distance_from_speed_time (speed : ℝ) (time : ℝ) := speed * time
  let midway (d : ℝ) := d / 2
  let total_distance := (distance_from_speed_time 45 4.444444444444445) * 2
  let travel_time (d : ℝ) (s : ℝ) := d / s
  show travel_time total_distance 80 = 5
  sorry

end venki_trip_time_l31_31172


namespace opposite_of_neg_frac_l31_31947

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31947


namespace probability_white_ball_l31_31578
open Classical

theorem probability_white_ball (B : Finset ℕ) (w r : ℕ) (B_count : B.card = 3)
  (w_count : w = 1) (r_count : r = 2) : 
  (w + r = B.card) → 
  (w / B.card) = 1 / 3 :=
by 
  intros h
  rw [h, w_count, r_count]
  norm_num
  sorry

end probability_white_ball_l31_31578


namespace cot_240_eq_neg_sqrt3_div_3_l31_31286

theorem cot_240_eq_neg_sqrt3_div_3 : Real.cot (240 * Real.pi / 180) = -Real.sqrt 3 / 3 :=
  by
  sorry

end cot_240_eq_neg_sqrt3_div_3_l31_31286


namespace chords_from_nine_points_l31_31073

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31073


namespace county_population_percentage_l31_31522

theorem county_population_percentage 
    (percent_less_than_20000 : ℝ)
    (percent_20000_to_49999 : ℝ) 
    (h1 : percent_less_than_20000 = 35) 
    (h2 : percent_20000_to_49999 = 40) : 
    percent_less_than_20000 + percent_20000_to_49999 = 75 := 
by
  sorry

end county_population_percentage_l31_31522


namespace products_equal_l31_31597

theorem products_equal (n : ℕ) (h : n ≥ 2) :
  ∃ (x : Fin n → ℝ), (∀ i, x i ≠ 1) ∧ (∏ i, x i) = ∏ i, (1 / (1 - x i)) :=
by sorry

end products_equal_l31_31597


namespace expected_pairs_socks_l31_31154

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l31_31154


namespace Kendra_sites_on_Wednesday_l31_31014

theorem Kendra_sites_on_Wednesday :
  (Kendra_number_of_sites_mon : ℕ) (Kendra_average_birds_mon : ℕ) (Kendra_number_of_sites_tue : ℕ) 
  (Kendra_average_birds_tue : ℕ) (Kendra_average_birds_wed : ℕ) (Kendra_overall_average_birds : ℕ) 
  (Kendra_number_of_sites_mon = 5) (Kendra_average_birds_mon = 7)
  (Kendra_number_of_sites_tue = 5) (Kendra_average_birds_tue = 5)
  (Kendra_average_birds_wed = 8) (Kendra_overall_average_birds = 7) :
  ∃ (W : ℕ), (60 + 8 * W) = 7 * (10 + W) ∧ W = 10 :=
by
  sorry

end Kendra_sites_on_Wednesday_l31_31014


namespace share_of_c_l31_31196

variable (a b c : ℝ)

theorem share_of_c (h1 : a + b + c = 427) (h2 : 3 * a = 7 * c) (h3 : 4 * b = 7 * c) : c = 84 :=
  by
  sorry

end share_of_c_l31_31196


namespace divisor_is_20_l31_31204

theorem divisor_is_20 (D q1 q2 q3 : ℕ) :
  (242 = D * q1 + 11) ∧
  (698 = D * q2 + 18) ∧
  (940 = D * q3 + 9) →
  D = 20 :=
by
  sorry

end divisor_is_20_l31_31204


namespace ratio_AE_EC_l31_31394

/-- Given a triangle ABC with AB = 6, BC = 8, AC = 10, and a point E on AC such that BE = 6, 
    the ratio AE:EC is 18:7. -/
theorem ratio_AE_EC (A B C E : Type) 
  (dist_AB : ℝ) (dist_BC : ℝ) (dist_AC : ℝ) (dist_BE : ℝ) 
  (on_segment : E ∈ line_segment A C) 
  (side_AB : dist_AB = 6)
  (side_BC : dist_BC = 8)
  (side_AC : dist_AC = 10)
  (side_BE : dist_BE = 6)
  (ratio : ℝ) (AE EC : ℝ) :
  AE / EC = ratio ↔ ratio = 18 / 7 :=
begin
  sorry,
end

end ratio_AE_EC_l31_31394


namespace car_cost_l31_31651

def initial_savings : ℕ := 14500
def charge_per_trip : ℚ := 1.5
def percentage_groceries_earnings : ℚ := 0.05
def number_of_trips : ℕ := 40
def total_value_of_groceries : ℕ := 800

theorem car_cost (initial_savings charge_per_trip percentage_groceries_earnings number_of_trips total_value_of_groceries : ℚ) :
  initial_savings + (charge_per_trip * number_of_trips) + (percentage_groceries_earnings * total_value_of_groceries) = 14600 := 
by
  sorry

end car_cost_l31_31651


namespace area_of_triangle_eq_zero_l31_31118

noncomputable def area_of_triangle_with_roots : Real :=
  let a b c : Real := sorry -- assume roots are real and distinct
  let p := (a + b + c) / 2
  let K := Real.sqrt(p * (p - a) * (p - b) * (p - c))
  K

theorem area_of_triangle_eq_zero :
  ∀ a b c : Real, (Polynomial.real_roots (Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6) = [a, b, c]) →
  let p := (a + b + c) / 2
  let K := Real.sqrt(p * (p - a) * (p - b) * (p - c))
  K = 0 :=
by
  intro a b c h_roots
  -- The roots should satisfy the polynomial equations and conditions.
  have h1 : a + b + c = 6 := by sorry
  have h2 : a * b + a * c + b * c = 11 := by sorry
  have h3 : a * b * c = 6 := by sorry
  let p := (a + b + c) / 2
  let K := Real.sqrt(p * (p - a) * (p - b) * (p - c))
  show K = 0 from sorry

end area_of_triangle_eq_zero_l31_31118


namespace cake_division_l31_31547

theorem cake_division {cakes children : ℕ} (h_cakes : cakes = 9) (h_children : children = 4)
  (h_cake_parts : ∀ c, c < cakes → c ≤ 1) :
  ∃ (division : list (ℕ × ℕ)), 
    (∀ ch, ch < children → 
      (∃ parts : list ℕ, parts.length = cakes ∧ 
      (∀ p, p ∈ parts → p = 2 ∨ p = 1) ∧ 
      parts.sum = (2 * cakes) ∧ 
      (∃ (one_quarter_pieces : ℕ), one_quarter_pieces = 1 ∧ parts.length = 1/4 * cakes))) :=
by sorry

end cake_division_l31_31547


namespace ellipse_eccentricity_l31_31529

noncomputable def eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :=
  let c := sqrt (a^2 - b^2)
  c / a

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h_arithmetic : (a - sqrt (a^2 - b^2), 2 * sqrt (a^2 - b^2), a + sqrt (a^2 - b^2)) = (λ x y z, (x + y) / 2 = (y + z) / 2) ):
  eccentricity a b h1 h2 = 1 / 2 := 
sorry

end ellipse_eccentricity_l31_31529


namespace courtyard_length_l31_31609

theorem courtyard_length (width_of_courtyard : ℝ) (brick_length_cm brick_width_cm : ℝ) (total_bricks : ℕ) (H1 : width_of_courtyard = 16) (H2 : brick_length_cm = 20) (H3 : brick_width_cm = 10) (H4 : total_bricks = 20000) :
  ∃ length_of_courtyard : ℝ, length_of_courtyard = 25 := 
by
  -- variables and hypotheses
  let brick_length_m := brick_length_cm / 100
  let brick_width_m := brick_width_cm / 100
  let area_one_brick := brick_length_m * brick_width_m
  let total_area := total_bricks * area_one_brick
  have width_of_courtyard_val : width_of_courtyard = 16 := H1
  have brick_length_cm_val : brick_length_cm = 20 := H2
  have brick_width_cm_val : brick_width_cm = 10 := H3
  have total_bricks_val : total_bricks = 20000 := H4
  let length_of_courtyard := total_area / width_of_courtyard
  have length_courtyard_val : length_of_courtyard = 25 := sorry
  use length_of_courtyard,
  exact length_courtyard_val sorry

end courtyard_length_l31_31609


namespace lines_intersect_and_not_perpendicular_l31_31532

theorem lines_intersect_and_not_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + 3 * y + a = 0 ∧ 3 * x - 2 * y + 1 = 0) ∧ 
  ¬ (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = 3 / 2 ∧ k1 ≠ k2 ∧ k1 * k2 = -1) :=
by
  sorry

end lines_intersect_and_not_perpendicular_l31_31532


namespace chords_from_nine_points_l31_31046

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31046


namespace airsickness_related_to_gender_l31_31813

def a : ℕ := 28
def b : ℕ := 28
def c : ℕ := 28
def d : ℕ := 56
def n : ℕ := 140

def contingency_relation (a b c d n K2 : ℕ) : Prop := 
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  K2 > 3841 / 1000

-- Goal statement for the proof
theorem airsickness_related_to_gender :
  contingency_relation a b c d n 3888 :=
  sorry

end airsickness_related_to_gender_l31_31813


namespace profit_percentage_correct_tea_profit_percentage_l31_31240

def total_cost (weight1 weight2 cost1 cost2 : ℕ) : ℕ :=
  weight1 * cost1 + weight2 * cost2

def total_sale (total_weight sale_price : ℕ) : ℕ :=
  total_weight * sale_price

def profit (sale cost : ℕ) : ℕ :=
  sale - cost

def profit_percentage (profit cost : ℕ) : ℕ :=
  (profit * 100) / cost

theorem profit_percentage_correct : profit_percentage 560 1600 = 35 := by
  sorry

theorem tea_profit_percentage :
  let weight1 := 80 in
  let weight2 := 20 in
  let cost1 := 15 in
  let cost2 := 20 in
  let sale_price := 21.6.to_nat in
  let total_weight := weight1 + weight2 in
  let total_cost := total_cost weight1 weight2 cost1 cost2 in
  let total_sale := total_sale total_weight sale_price in
  let profit := profit total_sale total_cost in
  profit_percentage profit total_cost = 35 := by
  sorry

end profit_percentage_correct_tea_profit_percentage_l31_31240


namespace proof_problem_l31_31451

def p (a b : ℝ) : Prop := a > b → 1 / a < 1 / b
def q (a b : ℝ) : Prop := 1 / (a * b) < 0 → a * b < 0

theorem proof_problem (a b : ℝ) (hp : ¬ p a b) (hq : q a b) :
  (¬ p a b ∧ q a b) ∨ (¬ p a b ∧ ¬ q a b) ∨ (p a b ∧ ¬ q a b) = 1 := by
  sorry

end proof_problem_l31_31451


namespace final_balance_is_60_million_l31_31619

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l31_31619


namespace a_18_value_l31_31312

variable (a : ℕ → ℚ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a_rec (n : ℕ) (hn : 2 ≤ n) : 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)

theorem a_18_value : a 18 = 26 / 9 :=
sorry

end a_18_value_l31_31312


namespace opposite_of_neg_frac_l31_31946

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31946


namespace opposite_of_neg_one_div_2023_l31_31940

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31940


namespace opposite_of_neg_nine_l31_31123

theorem opposite_of_neg_nine : ∃ x : ℤ, -9 + x = 0 ∧ x = 9 :=
by
  use 9
  split
  . exact rfl
  . exact rfl

end opposite_of_neg_nine_l31_31123


namespace merry_boxes_on_saturday_l31_31456

variables (x : ℕ) (left_boxes : ℕ) (sold_apples : ℕ) (sunday_boxes : ℕ)
variables (box_apples : ℕ) (total_apples_sold : ℕ)

-- Conditions
variables (h1 : sunday_boxes = 25)
variables (h2 : box_apples = 10)
variables (h3 : total_apples_sold = 720)
variables (h4 : left_boxes = 3)

theorem merry_boxes_on_saturday :
  x = 69 :=
by
  have sunday_apples : ℕ := sunday_boxes * box_apples,
  have saturday_apples : ℕ := total_apples_sold - sunday_apples,
  have saturday_boxes : ℕ := saturday_apples / box_apples,
  have sold_boxes : ℕ := saturday_boxes - left_boxes,
  have initial_saturday_boxes : ℕ := sold_boxes + sunday_boxes,
  exact calc
    x = initial_saturday_boxes : by sorry
    ... = 69 : by sorry

end merry_boxes_on_saturday_l31_31456


namespace range_of_f_l31_31563

noncomputable def f (x : ℝ) : ℝ := (floor (2 * x : ℝ)) - x

theorem range_of_f : set.range f = set.Icc (-0.5 : ℝ) (0.5 : ℝ) :=
by sorry

end range_of_f_l31_31563


namespace find_angle_A_max_area_l31_31838

-- Definition of the problem conditions
variables {A B C : ℝ}
variables {a b c : ℝ}

-- First condition: given in the problem statement
def condition : Prop :=
  (sin (2 * B) / (sqrt 3 * cos (B + C) - cos C * sin B)) = (2 * b / c)

-- First question: Finding A
theorem find_angle_A (h : condition) : A = 2 * Real.pi / 3 :=
sorry

-- Second question: Finding the maximum area with given specific values
theorem max_area (h : condition) (hA : A = 2 * Real.pi / 3) (ha : a = sqrt 3) : 
  (let area : ℝ := 1 / 2 * b * c * sin A in area <= sqrt 3 / 4) :=
sorry

end find_angle_A_max_area_l31_31838


namespace angle_ABC_60_degrees_l31_31418

variables (A B C D E : Type) [EuclideanGeometry ℝ]

-- Define that ABCDE forms a regular pentagon
def is_regular_pentagon (ABCDE : set Type) : Prop := 
∃ (s : ℝ), (∀ (p q : ABCDE), p ≠ q → dist p q = s ∧ 
∀ (p q r : ABCDE), p ≠ q ∧ q ≠ r ∧ p ≠ r → 
∠ p q r = 108 * (π/180))

-- Define that ∠ABC = 2 ∠DBE
def angle_relationship (A B C D E : Type) [EuclideanGeometry ℝ]
(∠ABC ∠DBE : AngleType ℝ) : Prop := 
∠ABC = 2 * ∠DBE

-- Finally state what we want to prove
theorem angle_ABC_60_degrees (A B C D E : Type)
  [EuclideanGeometry ℝ] 
  (ABCDE : set Type)
  (h1 : is_regular_pentagon ABCDE)
  (h2 : angle_relationship A B C D E ∠ABC ∠DBE) :
  ∠ABC = 60 * (π/180) :=
begin
  sorry
end

end angle_ABC_60_degrees_l31_31418


namespace total_rulers_is_21_l31_31535

def initial_rulers : ℝ := 11
def rulers_added_first : ℝ := 14
def rulers_removed : ℝ := 7
def rulers_added_second : ℝ := 3.5
def total_rulers (initial added1 removed added2 : ℝ) : ℝ := initial + added1 - removed + added2
def total_whole_rulers (initial added1 removed added2 : ℝ) : Int := Int.floor (total_rulers initial added1 removed added2)

theorem total_rulers_is_21 :
  total_whole_rulers initial_rulers rulers_added_first rulers_removed rulers_added_second = 21 := by
  sorry

end total_rulers_is_21_l31_31535


namespace circle_equation_l31_31605

noncomputable def circle_center_x : ℝ := sqrt 14 / 4
noncomputable def circle_center_y : ℝ := sqrt 14 / 12
noncomputable def circle_radius_squared : ℝ := 7 / 16

theorem circle_equation :
  (x - circle_center_x) ^ 2 + (y - circle_center_y) ^ 2 = circle_radius_squared :=
sorry

end circle_equation_l31_31605


namespace reversed_number_increase_l31_31637

theorem reversed_number_increase (a b c : ℕ) 
  (h1 : a + b + c = 10) 
  (h2 : b = a + c)
  (h3 : a = 2 ∧ b = 5 ∧ c = 3) :
  (c * 100 + b * 10 + a) - (a * 100 + b * 10 + c) = 99 :=
by
  sorry

end reversed_number_increase_l31_31637


namespace does_not_pass_through_third_quadrant_l31_31931

noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

theorem does_not_pass_through_third_quadrant (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  ¬ ∃ x, f a b x < 0 ∧ x < 0 := sorry

end does_not_pass_through_third_quadrant_l31_31931


namespace problem1_problem2_problem3_l31_31236

-- Problem 1
theorem problem1
  (monday : ℝ := -27.8)
  (tuesday : ℝ := -70.3)
  (wednesday : ℝ := 200)
  (thursday : ℝ := 138.1)
  (friday : ℝ := -8)
  (sunday : ℝ := 188)
  (total : ℝ := 458) :
  let profit_saturday := total - (monday + tuesday + wednesday + thursday + friday + sunday) in
  profit_saturday = 38 :=
by {
  -- Solution here
  sorry
}

-- Problem 2
theorem problem2
  (monday : ℝ := -27.8)
  (tuesday : ℝ := -70.3)
  (wednesday : ℝ := 200)
  (thursday : ℝ := 138.1)
  (sunday : ℝ := 188)
  (total : ℝ := 458)
  (friday_earn_more : ℝ := 10) :
  let friday := (total - (monday + tuesday + wednesday + thursday + sunday) - friday_earn_more) / 2 in
  let saturday := friday + friday_earn_more in
  saturday = 20 :=
by {
  -- Solution here
  sorry
}

-- Problem 3
theorem problem3
  (monday : ℝ := -27.8)
  (tuesday : ℝ := -70.3)
  (wednesday : ℝ := 200)
  (thursday : ℝ := 138.1)
  (sunday : ℝ := 188)
  (total : ℝ := 458) :
  let total_excl_fri_sat := total - (monday + tuesday + wednesday + thursday + sunday) in
  ∃ (friday saturday : ℝ), friday < 0 ∧ saturday > 0 ∧ saturday > total_excl_fri_sat :=
by {
  -- Solution here
  sorry
}

end problem1_problem2_problem3_l31_31236


namespace order_of_magnitude_l31_31779

-- Definitions
variables (x : ℝ) (hx1 : 0.9 < x) (hx2 : x < 1.0) (y : ℝ) (z : ℝ)
-- Conditions
def y : ℝ := x^x
def z : ℝ := x ^ (x^x)

-- Theorem statement
theorem order_of_magnitude (h1 : 0.9 < x) (h2 : x < 1.0) : x < z ∧ z < y :=
by
  -- Proof details go here
  sorry

end order_of_magnitude_l31_31779


namespace problem_3_equals_answer_l31_31663

variable (a : ℝ)

theorem problem_3_equals_answer :
  (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 :=
by
  sorry

end problem_3_equals_answer_l31_31663


namespace find_total_pupils_l31_31542

-- Define the conditions for the problem
def diff1 : ℕ := 85 - 45
def diff2 : ℕ := 79 - 49
def diff3 : ℕ := 64 - 34
def total_diff : ℕ := diff1 + diff2 + diff3
def avg_increase : ℕ := 3

-- Assert that the number of pupils n satisfies the given conditions
theorem find_total_pupils (n : ℕ) (h_diff : total_diff = 100) (h_avg_inc : avg_increase * n = total_diff) : n = 33 :=
by
  sorry

end find_total_pupils_l31_31542


namespace sum_of_corners_9x9_grid_l31_31513

theorem sum_of_corners_9x9_grid : 
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  show topLeft + topRight + bottomLeft + bottomRight = 164
  sorry
}

end sum_of_corners_9x9_grid_l31_31513


namespace coefficient_of_x9_in_polynomial_is_240_l31_31923

-- Define the polynomial (1 + 3x - 2x^2)^5
noncomputable def polynomial : ℕ → ℝ := (fun x => (1 + 3*x - 2*x^2)^5)

-- Define the term we are interested in (x^9)
def term := 9

-- The coefficient we want to prove
def coefficient := 240

-- The goal is to prove that the coefficient of x^9 in the expansion of (1 + 3x - 2x^2)^5 is 240
theorem coefficient_of_x9_in_polynomial_is_240 : polynomial 9 = coefficient := sorry

end coefficient_of_x9_in_polynomial_is_240_l31_31923


namespace opposite_neg_fraction_l31_31949

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31949


namespace coordinates_after_transformation_l31_31822

theorem coordinates_after_transformation :
  ∀ (x y : ℤ), x = -2 → y = 5 → (x - 1, y + 3) = (-3, 8) := by
  intros x y hx hy
  -- From conditions
  rw [hx, hy]
  -- Calculations and transformations
  simp
  -- With the transformations, the resulting coordinates should equal the answer
  sorry

end coordinates_after_transformation_l31_31822


namespace find_m_l31_31750

-- Definitions based on the given conditions
def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

def a := (1 : ℝ, 2 : ℝ)
def b (m : ℝ) := (2 * m - 1, -1)

-- The statement to be proven
theorem find_m (m : ℝ) (h : vector_perpendicular a (b m)) : m = 3 / 2 :=
by
  sorry

end find_m_l31_31750


namespace matrix_power_minus_l31_31858

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l31_31858


namespace sum_of_f_is_negative_l31_31732

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_of_f_is_negative (x₁ x₂ x₃ : ℝ)
  (h1: x₁ + x₂ < 0)
  (h2: x₂ + x₃ < 0) 
  (h3: x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := 
sorry

end sum_of_f_is_negative_l31_31732


namespace correct_calculation_l31_31574

theorem correct_calculation (a b : ℝ) : 
  ¬(3 * a + b = 3 * a * b) ∧ 
  ¬(a^2 + a^2 = a^4) ∧ 
  ¬((a - b)^2 = a^2 - b^2) ∧ 
  ((-3 * a)^2 = 9 * a^2) :=
by
  sorry

end correct_calculation_l31_31574


namespace aziz_parents_in_america_l31_31657

/-- 
Given that Aziz's parents moved to America in 1982, the current year is 2021, and Aziz just celebrated his 36th birthday,
prove that Aziz's parents had been living in America for 3 years before he was born.
-/
theorem aziz_parents_in_america (year_parents_moved : ℕ) (current_year : ℕ) (aziz_age : ℕ)
  (h_move : year_parents_moved = 1982)
  (h_current : current_year = 2021)
  (h_age : aziz_age = 36) :
  current_year - aziz_age - year_parents_moved = 3 :=
by
  -- conditions
  rw [h_move, h_current, h_age]
  -- calculation 
  sorry

end aziz_parents_in_america_l31_31657


namespace number_of_girls_l31_31087

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 16) : total_children - boys = 44 := by
  sorry

end number_of_girls_l31_31087


namespace necessarily_true_l31_31233

def statements (d : ℕ) : Prop × Prop × Prop × Prop := 
  (d = 5, d ≠ 6, d = 7, d ≠ 8)

theorem necessarily_true (d : ℕ) :
  (∃ a b c d' ∈ ({1, 2, 3, 4} : set ℕ), 
    3 ∈ {a, b, c} 
  ∧ (a = 1 ↔ d = 5) 
  ∧ (b = 2 ↔ d ≠ 6) 
  ∧ (c = 3 ↔ d = 7) 
  ∧ (d' = 4 ↔ d ≠ 8)) → 
  d ≠ 6 := 
by 
  sorry

end necessarily_true_l31_31233


namespace quotient_of_division_l31_31472

theorem quotient_of_division (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181) (h2 : divisor = 20) (h3 : remainder = 1) 
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 9 :=
by
  sorry -- proof goes here

end quotient_of_division_l31_31472


namespace derivative_at_zero_l31_31727

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  x * (x + 1) * (2 * x + 1) * (3 * x + 1) * ... * (n * x + 1)

theorem derivative_at_zero (n : ℕ) : deriv (f n) 0 = 1 :=
by {
  sorry
}

end derivative_at_zero_l31_31727


namespace shortest_distance_curve_to_line_l31_31973

open Real

def curve (x : ℝ) : ℝ := log (2 * x - 1)

def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

theorem shortest_distance_curve_to_line :
  ∃ (d_min : ℝ), ∀ x : ℝ, x > 0.5 → 
  ∃ y : ℝ, y = curve x ∧ 
  (line x y → abs (2 * x - log (2 * x - 1) + 3) / sqrt 5 = d_min) :=
by
  sorry

end shortest_distance_curve_to_line_l31_31973


namespace opposite_neg_fraction_l31_31950

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31950


namespace theta_value_l31_31792

theorem theta_value (θ : ℝ) (k : ℤ) :
  (sin (2 * θ) = 1) ∧ (sqrt 2 * cos θ + 1 ≠ 0) ↔ (∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 4) :=
by
  sorry

end theta_value_l31_31792


namespace geometric_series_properties_l31_31675

theorem geometric_series_properties :
  let a := 3
  let r := 1 / 4
  let S := a / (1 - r)
  (S = 4) -> 
  (¬ (S > 4) ∧ ¬ (S < 4) ∧ 
   (∀ ε > 0, ∃ N, ∀ n ≥ N, |a * r ^ n - 0| < ε) ∧
   ¬ (∀ ε > 0, |S - 5| < ε) ∧ 
   (∃ L, ∀ ε > 0, |S - L| < ε)) :=
by
  sorry

end geometric_series_properties_l31_31675


namespace triangle_is_isosceles_or_right_angled_l31_31908

variables {α β γ : ℝ} -- Angles of the triangle

theorem triangle_is_isosceles_or_right_angled
  (h_angle_sum : α + β + γ = π)
  (h_ratio_condition : (tan α / tan β) = (sin α ^ 2 / sin β ^ 2)) :
  α = β ∨ γ = π / 2 :=
sorry

end triangle_is_isosceles_or_right_angled_l31_31908


namespace curve_equation_minimum_QM_l31_31743

-- Definition of points A and B and the condition for moving point P satisfying |PA| = √2|PB|
def pointA : ℝ × ℝ := (-1, 2)
def pointB : ℝ × ℝ := (0, 1)

def satisfiesCondition (P : ℝ × ℝ) : Prop :=
  dist P pointA = real.sqrt 2 * dist P pointB

-- Proving the equation of curve C
theorem curve_equation (P : ℝ × ℝ) (h : satisfiesCondition P) :
  (P.1 - 1)^2 + P.2^2 = 4 := sorry

-- Definition of line l1 and the point Q on the line
def line_l1 (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

-- Minimum value of |QM|
theorem minimum_QM (Q : ℝ × ℝ) (M : ℝ × ℝ) (hQ : line_l1 Q.1 Q.2) 
  (hM : (M.1 - 1)^2 + M.2^2 = 4 ∧ line_l1 Q.1 Q.2) : 
  ∀ q : ℝ, dist Q M = real.sqrt 5 := sorry

end curve_equation_minimum_QM_l31_31743


namespace paintings_on_last_page_in_rearrangement_l31_31906

theorem paintings_on_last_page_in_rearrangement 
  (albums_initial : ℕ) 
  (pages_per_album : ℕ) 
  (paintings_per_page_initial : ℕ) 
  (paintings_per_page_new : ℕ) 
  (albums_rearranged_full : ℕ) 
  (pages_per_almost_album : ℕ) :
  albums_initial = 10 → 
  pages_per_album = 36 → 
  paintings_per_page_initial = 8 → 
  paintings_per_page_new = 9 → 
  albums_rearranged_full = 6 → 
  pages_per_almost_album = 28 → 
  (let total_paintings := albums_initial * pages_per_album * paintings_per_page_initial,
       total_pages_filled := albums_rearranged_full * pages_per_album + pages_per_almost_album,
       paintings_remaining := total_paintings - total_pages_filled * paintings_per_page_new
   in paintings_remaining) % paintings_per_page_new = 0 :=
by 
   intros hadd_pages hadd_books hadd_pics hnew_pics hadd_fully_used hpages_almost_more;
   sorry

end paintings_on_last_page_in_rearrangement_l31_31906


namespace five_lines_max_sections_l31_31830

theorem five_lines_max_sections (n : ℕ) (h : n = 5):
    max_sections n = 16 :=
sorry

end five_lines_max_sections_l31_31830


namespace range_of_a_l31_31774

open Real

theorem range_of_a (a : ℝ) :
    (∀ θ ∈ Icc 0 (π / 2), 
        sqrt 2 * (2 * a + 3) * cos (θ - π / 4) + 6 / (sin θ + cos θ) - 2 * sin (2 * θ) < 3 * a + 6) 
    → a > 3 :=
by
  sorry

end range_of_a_l31_31774


namespace positive_difference_prob_l31_31169

/-- Probability that the positive difference between two randomly chosen numbers from 
the set {1, 2, 3, 4, 5, 6, 7, 8} is 3 or greater -/
theorem positive_difference_prob :
  (let S := {1, 2, 3, 4, 5, 6, 7, 8}
       in (S.powerset.filter (λ s => s.card = 2)).card.filter (λ s => (s.to_list.head! - s.to_list.tail.head!).nat_abs >= 3).card /
           (S.powerset.filter (λ s => s.card = 2)).card = 15 / 28) := 
begin
  sorry
end

end positive_difference_prob_l31_31169


namespace five_lines_max_sections_l31_31829

theorem five_lines_max_sections (n : ℕ) (h : n = 5):
    max_sections n = 16 :=
sorry

end five_lines_max_sections_l31_31829


namespace smallest_prime_dividing_sum_of_powers_l31_31999

theorem smallest_prime_dividing_sum_of_powers :
  ∃ p : ℕ, prime p ∧ p ∣ (4^13 + 6^15) ∧ (∀ q : ℕ, prime q ∧ q ∣ (4^13 + 6^15) → p ≤ q) ∧ p = 2 :=
by
  sorry

end smallest_prime_dividing_sum_of_powers_l31_31999


namespace length_of_CD_in_cyclic_quad_l31_31095

theorem length_of_CD_in_cyclic_quad (A B C D O : Point)
  (h_inscribed : CyclicQuadrilateral A B C D)
  (h_diameter : distance A D = 4)
  (h_AB : distance A B = 1)
  (h_BC : distance B C = 1) :
  distance C D = 7/2 := 
sorry

end length_of_CD_in_cyclic_quad_l31_31095


namespace max_bricks_truck_can_carry_l31_31981

-- Define the truck's capacity in terms of bags of sand and bricks
def max_sand_bags := 50
def max_bricks := 400
def sand_to_bricks_ratio := 8

-- Define the current number of sand bags already on the truck
def current_sand_bags := 32

-- Define the number of bricks equivalent to a given number of sand bags
def equivalent_bricks (sand_bags: ℕ) := sand_bags * sand_to_bricks_ratio

-- Define the remaining capacity in terms of bags of sand
def remaining_sand_bags := max_sand_bags - current_sand_bags

-- Define the maximum number of additional bricks the truck can carry
def max_additional_bricks := equivalent_bricks remaining_sand_bags

-- Prove the number of additional bricks the truck can carry is 144
theorem max_bricks_truck_can_carry : max_additional_bricks = 144 := by
  sorry

end max_bricks_truck_can_carry_l31_31981


namespace sum_integers_between_neg12_and_3_l31_31572

theorem sum_integers_between_neg12_and_3 : ∑ i in finset.Icc (-12) 3, i = -72 := by
  sorry

end sum_integers_between_neg12_and_3_l31_31572


namespace gcf_120_180_240_l31_31177

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l31_31177


namespace amount_used_to_pay_l31_31098

noncomputable def the_cost_of_football : ℝ := 9.14
noncomputable def the_cost_of_baseball : ℝ := 6.81
noncomputable def the_change_received : ℝ := 4.05

theorem amount_used_to_pay : 
    (the_cost_of_football + the_cost_of_baseball + the_change_received) = 20.00 := 
by
  sorry

end amount_used_to_pay_l31_31098


namespace angle_between_sum_is_pi_over_6_l31_31783

open Real EuclideanSpace

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := sqrt (u.1^2 + u.2^2)
  let norm_v := sqrt (v.1^2 + v.2^2)
  arccos (dot_product / (norm_u * norm_v))

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (1/2 * cos (π / 3), 1/2 * sin (π / 3))

theorem angle_between_sum_is_pi_over_6 :
  angle_between_vectors (a.1 + 2 * b.1, a.2 + 2 * b.2) b = π / 6 :=
by
  sorry

end angle_between_sum_is_pi_over_6_l31_31783


namespace problem_statement_l31_31366

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else sequence (n - 1) + sequence (n - 2) - sequence (n - 3) + 1

theorem problem_statement :
  (∑ k in Finset.range 2016, 1 / (sequence (k + 1) : ℝ)) < 3 :=
by
  sorry

end problem_statement_l31_31366


namespace gcf_120_180_240_l31_31178

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l31_31178


namespace relation_between_x_y_q_z_l31_31105

variable (a c e f : ℝ)
variable {x y q z : ℝ}

-- Given conditions
def condition1 := a^(3 * x) = e ∧ c^(4 * q) = e
def condition2 := c^(2 * y) = f ∧ a^(5 * z) = f

-- The theorem to prove
theorem relation_between_x_y_q_z (h1 : condition1) (h2 : condition2) : 3 * y = 10 * q := by
  sorry

end relation_between_x_y_q_z_l31_31105


namespace elvis_ralph_matchsticks_l31_31695

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l31_31695


namespace janice_office_floor_l31_31426

theorem janice_office_floor 
    (F : ℕ) 
    (up_times : 5) 
    (down_times : 3) 
    (total_flights : 24) 
    (h1 : 5 * F + 3 * F = total_flights) : 
    F = 3 := 
by 
  sorry

end janice_office_floor_l31_31426


namespace count_double_digit_sum_to_three_is_ten_l31_31025

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def double_digit_sum_to_three (count : ℕ) : Prop :=
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 → 
    (sum_of_digits (sum_of_digits x) = 3 → x ∞∣∑_x count ∧) → count = 10 

theorem count_double_digit_sum_to_three_is_ten : 
  ∃ count, double_digit_sum_to_three count ∧ count = 10 := sorry

end count_double_digit_sum_to_three_is_ten_l31_31025


namespace find_reciprocal_point_l31_31823

theorem find_reciprocal_point (F : ℂ) (hF: F = 3 + 4 * Complex.I) (hF_outside: Complex.abs F > 1) :
  let recF := 1 / F in
  (∃ (x : ℂ), (x.re = recF.re) ∧ (x.im = recF.im) ∧ x = (1 / 2) - (1 / 2) * Complex.I) :=
by
  sorry

end find_reciprocal_point_l31_31823


namespace smallest_a_l31_31181

def is_undefined_mod (a n : ℕ) : Prop := 
  Nat.gcd a n ≠ 1

theorem smallest_a:
  ∃ a : ℕ, a > 0 ∧ is_undefined_mod a 77 ∧ is_undefined_mod a 91 ∧ 
           ∀ b : ℕ, b > 0 ∧ is_undefined_mod b 77 ∧ is_undefined_mod b 91 → a ≤ b :=
begin
  use 7,
  split,
  { exact Nat.succ_pos 6 },
  split,
  { unfold is_undefined_mod,
    suffices : Nat.gcd 7 77 = 7, 
    { rwa [this] },
    exact Nat.gcd_comm 7 77 },
  split,
  { unfold is_undefined_mod,
    suffices : Nat.gcd 7 91 = 7, 
    { rwa [this] },
    exact Nat.gcd_comm 7 91 },
  { intros b hb,
    cases hb with hb_pos hb_cond,
    cases hb_cond with hb_mod_77 hb_cond_91,
    unfold is_undefined_mod at *,
    have : Nat.gcd b 77 > 1, by linarith,
    have : Nat.gcd b 91 > 1, by linarith,
    sorry,
  }
end

end smallest_a_l31_31181


namespace find_hyperbola_m_l31_31356

theorem find_hyperbola_m (m : ℝ) :
  (∃ (a b : ℝ), a^2 = 16 ∧ b^2 = m ∧ (sqrt (1 + m / 16) = 5 / 4)) → m = 9 :=
by
  intro h
  sorry

end find_hyperbola_m_l31_31356


namespace prod_op_result_l31_31016

-- Define set A and set B as provided.
def A : Set ℝ := {x | true } -- Simplified since the set is not well-defined.
def B : Set ℝ := {y | ∃ x, y = 2 ^ x ∧ x > 0 }

-- Define the product operation given in the problem.
def prod_op (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- The final theorem we want to prove.
theorem prod_op_result : prod_op A B = [0,1] ∪ (2, +∞) :=
by
  -- Proof goes here.
  sorry

end prod_op_result_l31_31016


namespace ceiling_relation_l31_31283

theorem ceiling_relation :
  let r1 := (25 / 11 : ℝ) - ⌈35 / 19⌉ in
  let r2 := (35 / 11 : ℝ) + ⌈209 / 35⌉ in
  ⌈r1⌉ / ⌈r2⌉ = (1 / 10 : ℝ) :=
by
  have h1 : ⌈35 / 19⌉ = 2, from sorry
  have h2 : ⌈209 / 35⌉ = 6, from sorry
  have h3 : 25 / 11 - 2 ≈ 3 / 11, from sorry
  have h4 : 35 / 11 + 6 ≈ 101 / 11, from sorry
  have h5 : ⌈3 / 11⌉ = 1, from sorry
  have h6 : ⌈101 / 11⌉ = 10, from sorry
  sorry -- full proof would go here

end ceiling_relation_l31_31283


namespace Jacqueline_gave_Jane_l31_31425

def total_fruits (plums guavas apples : ℕ) : ℕ :=
  plums + guavas + apples

def fruits_given_to_Jane (initial left : ℕ) : ℕ :=
  initial - left

theorem Jacqueline_gave_Jane :
  let plums := 16
  let guavas := 18
  let apples := 21
  let left := 15
  let initial := total_fruits plums guavas apples
  fruits_given_to_Jane initial left = 40 :=
by
  sorry

end Jacqueline_gave_Jane_l31_31425


namespace line_and_circle_distance_l31_31734

theorem line_and_circle_distance (a b : ℝ) (h : a^2 + b^2 = 1) : 
  let d := 1 / (Real.sqrt (a^2 + b^2))
  in d = 1 :=
by
  sorry

end line_and_circle_distance_l31_31734


namespace courtyard_is_25_meters_long_l31_31611

noncomputable def courtyard_length (width : ℕ) (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) : ℝ :=
  let brick_area := brick_length * brick_width
  let total_area := num_bricks * brick_area
  total_area / width

theorem courtyard_is_25_meters_long (h_width : 16 = 16)
  (h_brick_length : 0.20 = 0.20)
  (h_brick_width: 0.10 = 0.10)
  (h_num_bricks: 20_000 = 20_000)
  (h_total_area: 20_000 * (0.20 * 0.10) = 400) :
  courtyard_length 16 0.20 0.10 20_000 = 25 := by
        sorry

end courtyard_is_25_meters_long_l31_31611


namespace vendor_pepsi_volume_l31_31218

theorem vendor_pepsi_volume 
    (liters_maaza : ℕ)
    (liters_sprite : ℕ)
    (num_cans : ℕ)
    (h1 : liters_maaza = 40)
    (h2 : liters_sprite = 368)
    (h3 : num_cans = 69)
    (volume_pepsi : ℕ)
    (total_volume : ℕ)
    (h4 : total_volume = liters_maaza + liters_sprite + volume_pepsi)
    (h5 : total_volume = num_cans * n)
    (h6 : 408 % num_cans = 0) :
  volume_pepsi = 75 :=
sorry

end vendor_pepsi_volume_l31_31218


namespace find_points_PQ_l31_31892

-- Define the points A, B, M, and E in 3D space
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨10, 0, 0⟩
def M : Point := ⟨5, 5, 0⟩
def E : Point := ⟨0, 0, 10⟩

-- Define the lines AB and EM
def line_AB (t : ℝ) : Point := ⟨10 * t, 0, 0⟩
def line_EM (s : ℝ) : Point := ⟨5 * s, 5 * s, 10 - 10 * s⟩

-- Define the points P and Q
def P (t : ℝ) : Point := line_AB t
def Q (s : ℝ) : Point := line_EM s

-- Define the distance function in 3D space
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- The main theorem
theorem find_points_PQ (t s : ℝ) (h1 : t = 0.4) (h2 : s = 0.8) :
  (P t = ⟨4, 0, 0⟩) ∧ (Q s = ⟨4, 4, 2⟩) ∧
  (distance (P t) (Q s) = distance (line_AB 0.4) (line_EM 0.8)) :=
by
  sorry

end find_points_PQ_l31_31892


namespace inverse_proportion_first_third_quadrant_l31_31358

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l31_31358


namespace cone_circumference_l31_31227

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_circumference (V h r C : ℝ) (hV : V = 27 * π) (hh : h = 9)
  (hV_formula : V = volume_of_cone r h) (hC : C = 2 * π * r) : 
  C = 6 * π :=
by 
  sorry

end cone_circumference_l31_31227


namespace k_less_than_two_l31_31360

theorem k_less_than_two
    (x : ℝ)
    (k : ℝ)
    (y : ℝ)
    (h : y = (2 - k) / x)
    (h1 : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) : k < 2 :=
by
  sorry

end k_less_than_two_l31_31360


namespace painter_total_collection_l31_31644

noncomputable def totalAmountPainterCollected : ℝ := 88.5

theorem painter_total_collection :
  let south_seq := (list.range 25).map (λ n, 5 + 7 * n),
      north_seq := (list.range 25).map (λ n, 2 + 7 * n),
      south_less_than_100 := (south_seq.filter (λ n, n < 100)).sum (λ n, (n.toString.length : ℝ)),
      south_more_than_100 := (south_seq.filter (λ n, n ≥ 100)).sum (λ n, (n.toString.length : ℝ) * 0.5),
      north_less_than_100 := (north_seq.filter (λ n, n < 100)).sum (λ n, (n.toString.length : ℝ)),
      north_more_than_100 := (north_seq.filter (λ n, n ≥ 100)).sum (λ n, (n.toString.length : ℝ) * 0.5)
  in
  south_less_than_100 + south_more_than_100 + north_less_than_100 + north_more_than_100 = totalAmountPainterCollected :=
by
  let south_seq := (list.range 25).map (λ n, 5 + 7 * n)
  let north_seq := (list.range 25).map (λ n, 2 + 7 * n)
  let south_less_than_100 := (south_seq.filter (λ n, n < 100)).sum (λ n, (n.toString.length : ℝ))
  let south_more_than_100 := (south_seq.filter (λ n, n ≥ 100)).sum (λ n, (n.toString.length : ℝ) * 0.5)
  let north_less_than_100 := (north_seq.filter (λ n, n < 100)).sum (λ n, (n.toString.length : ℝ))
  let north_more_than_100 := (north_seq.filter (λ n, n ≥ 100)).sum (λ n, (n.toString.length : ℝ) * 0.5)
  have h_eq_sum : south_less_than_100 + south_more_than_100 + north_less_than_100 + north_more_than_100 = 88.5 := sorry
  have h_eq_constant : totalAmountPainterCollected = 88.5 := rfl
  rw h_eq_constant at h_eq_sum
  exact h_eq_sum

end painter_total_collection_l31_31644


namespace no_rational_solution_of_odd_quadratic_l31_31029

theorem no_rational_solution_of_odd_quadratic (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ x : ℚ, a * x^2 + b * x + c = 0 :=
sorry

end no_rational_solution_of_odd_quadratic_l31_31029


namespace courtyard_length_l31_31610

theorem courtyard_length (width_of_courtyard : ℝ) (brick_length_cm brick_width_cm : ℝ) (total_bricks : ℕ) (H1 : width_of_courtyard = 16) (H2 : brick_length_cm = 20) (H3 : brick_width_cm = 10) (H4 : total_bricks = 20000) :
  ∃ length_of_courtyard : ℝ, length_of_courtyard = 25 := 
by
  -- variables and hypotheses
  let brick_length_m := brick_length_cm / 100
  let brick_width_m := brick_width_cm / 100
  let area_one_brick := brick_length_m * brick_width_m
  let total_area := total_bricks * area_one_brick
  have width_of_courtyard_val : width_of_courtyard = 16 := H1
  have brick_length_cm_val : brick_length_cm = 20 := H2
  have brick_width_cm_val : brick_width_cm = 10 := H3
  have total_bricks_val : total_bricks = 20000 := H4
  let length_of_courtyard := total_area / width_of_courtyard
  have length_courtyard_val : length_of_courtyard = 25 := sorry
  use length_of_courtyard,
  exact length_courtyard_val sorry

end courtyard_length_l31_31610


namespace cyclist_wait_time_l31_31631

open_locale classical

def hiker_speed_kph : ℝ := 4
def cyclist_speed_kph : ℝ := 24
def cyclist_wait_time_minute : ℝ := 5

noncomputable def hiker_speed_kpm : ℝ := hiker_speed_kph / 60
noncomputable def cyclist_speed_kpm : ℝ := cyclist_speed_kph / 60

noncomputable def distance_cyclist_travels : ℝ :=
  cyclist_speed_kpm * cyclist_wait_time_minute

noncomputable def time_for_hiker_to_catch_up : ℝ :=
  distance_cyclist_travels / hiker_speed_kpm

theorem cyclist_wait_time : time_for_hiker_to_catch_up = 30 :=
sorry

end cyclist_wait_time_l31_31631


namespace max_value_of_expression_l31_31332

-- Define the conditions of the problem
variable (m n : ℕ) -- Natural numbers m and n
variable (even_numbers : Finset ℕ) (odd_numbers : Finset ℕ) -- Sets of even and odd numbers

-- Conditions
def is_distinct_even (a : ℕ) : Prop := a % 2 = 0
def is_distinct_odd (a : ℕ) : Prop := a % 2 = 1
def sum_is_2015 : Prop := even_numbers.sum + odd_numbers.sum = 2015
def correct_form_of_sets : Prop := 
  even_numbers.card = m ∧
  odd_numbers.card = n ∧
  ∀ x ∈ even_numbers, is_distinct_even x ∧
  ∀ y ∈ odd_numbers, is_distinct_odd y

-- Prove the required maximum value
theorem max_value_of_expression : 
  correct_form_of_sets even_numbers odd_numbers →
  sum_is_2015 even_numbers odd_numbers →
  20 * m + 15 * n ≤ 1105 :=
begin
  sorry
end

end max_value_of_expression_l31_31332


namespace solve_trig_eqns_l31_31454

theorem solve_trig_eqns (x : Real) :
  (sin x = a) ∧ (sin (2 * x) = b) ∧ (sin (3 * x) = c) ↔ 
  (∃ k : Int, x = k * Real.pi / 2) ∨ 
  (∃ (m : Int), x = ±(2 * Real.pi / 3) + 2 * Real.pi * m) ∨ 
  (∃ n : Int, x = 2 * Real.pi * n / 5) := 
sorry

end solve_trig_eqns_l31_31454


namespace remainder_when_y_divided_by_48_l31_31509

theorem remainder_when_y_divided_by_48 (y : ℤ) 
  (h1 : 2 + y ≡ 8 [MOD 16]) 
  (h2 : 4 + y ≡ 16 [MOD 64]) 
  (h3 : 6 + y ≡ 36 [MOD 216]) : 
  y % 48 = 44 := 
sorry

end remainder_when_y_divided_by_48_l31_31509


namespace lana_extra_flowers_l31_31438

theorem lana_extra_flowers (tulips roses used total extra : ℕ) 
  (h1 : tulips = 36) 
  (h2 : roses = 37) 
  (h3 : used = 70) 
  (h4 : total = tulips + roses) 
  (h5 : extra = total - used) : 
  extra = 3 := 
sorry

end lana_extra_flowers_l31_31438


namespace tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3_l31_31267

-- Define the necessary trigonometric values
def tan_30 : ℝ := Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)
def cos_30 : ℝ := Real.cos (Real.pi / 6)

-- The main theorem statement
theorem tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3 :
  tan_30 + 4 * cos_30 = 7 * Real.sqrt 3 / 3 := by 
  sorry

end tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3_l31_31267


namespace ab_value_l31_31525

theorem ab_value (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ∧ (∀ y : ℝ, (x = 0 ∧ (y = 5 ∨ y = -5)))))
  (h2 : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (∀ x : ℝ, (y = 0 ∧ (x = 8 ∨ x = -8))))) :
  |a * b| = Real.sqrt 867.75 :=
by
  sorry

end ab_value_l31_31525


namespace find_min_max_AC_l31_31899

noncomputable def min_length_AC (A B C D : Point) (AB BC CD DA : ℝ) (hAB : AB = 2) (hBC : BC = 7) (hCD : CD = 5) (hDA : DA = 12) : ℝ :=
if h : (AC : ℝ) ≥ 7 then AC else 7

noncomputable def max_length_AC (A B C D : Point) (AB BC CD DA : ℝ) (hAB : AB = 2) (hBC : BC = 7) (hCD : CD = 5) (hDA : DA = 12) : ℝ :=
if h : (AC : ℝ) ≤ 9 then AC else 9

theorem find_min_max_AC (A B C D : Point) (AB BC CD DA : ℝ) (hAB : AB = 2) (hBC : BC = 7) (hCD : CD = 5) (hDA : DA = 12) :
  (min_length_AC A B C D AB BC CD DA hAB hBC hCD hDA, max_length_AC A B C D AB BC CD DA hAB hBC hCD hDA) = (7, 9) :=
sorry

end find_min_max_AC_l31_31899


namespace original_wire_length_is_correct_l31_31250

-- Define lengths of the pieces
def length_piece1 : ℕ := 14
def length_piece2 : ℕ := 16

-- Define the condition that one piece is 2 ft longer than the other
def length_difference : Prop := (length_piece2 = length_piece1 + 2)

-- Define the original length
def original_length : ℕ := length_piece1 + length_piece2

-- Prove that the original length is 30 ft given the conditions
theorem original_wire_length_is_correct : original_length = 30 := by
  -- Conditions
  have h1 : length_piece2 = length_piece1 + 2 := rfl
  have h2 : length_piece1 = 14 := rfl
  have h3 : length_piece2 = 16 := rfl

  -- Calculate the original length
  have h_orig : original_length = length_piece1 + length_piece2 := rfl
  
  -- Substitute known values and prove
  calc 
  original_length 
      = 14 + 16 := by rw [h2, h3]
  ... = 30 := by norm_num

end original_wire_length_is_correct_l31_31250


namespace hyperbola_eccentricity_l31_31354

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l31_31354


namespace tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3_l31_31266

-- Define the necessary trigonometric values
def tan_30 : ℝ := Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6)
def cos_30 : ℝ := Real.cos (Real.pi / 6)

-- The main theorem statement
theorem tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3 :
  tan_30 + 4 * cos_30 = 7 * Real.sqrt 3 / 3 := by 
  sorry

end tan_30_plus_4_cos_30_eq_7_sqrt_3_over_3_l31_31266


namespace num_chords_l31_31041

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31041


namespace triangle_LCM_is_isosceles_l31_31520

variables {A B C K L M O O' : Type*}
variables [incircle ω] [excircle Ω]
variables (AL : line A L) (K' : point_on_circle K ω) (M' : point_on_circle M Ω)
variables (AB AC BC : line A B, line A C, line B C)
variables (BK CM : line B K, line C M)

-- Define the given conditions
axiom touches_ω_AB : touches_circle_line ω AB
axiom touches_ω_AC : touches_circle_line ω AC
axiom touches_Ω_AC : touches_circle_line Ω AC
axiom touches_Ω_extension_AB : touches_circle_line Ω (extended_line AB)
axiom tangency_point_L : tangency_point L BC ω Ω

-- Prove the statement
theorem triangle_LCM_is_isosceles (parallel_KB_CM : BK.is_parallel_to CM) 
: (is_isosceles LCM) := 
sorry

end triangle_LCM_is_isosceles_l31_31520


namespace team_ranking_l31_31677

-- Definitions of the experience levels
variable (Experience : Type) [LinearOrder Experience]

-- The experience levels of David, Emma, and Frank
variables (David_E : Experience) (Emma_E : Experience) (Frank_E: Experience)

-- Statements about the experience levels
def stmt1 : Prop := Emma_E < Frank_E
def stmt2 : Prop := David_E > Frank_E
def stmt3 : Prop := ∀ x, (x = David_E) ∨ (x = Emma_E) ∨ (x = Frank_E) → x <= Frank_E

-- Exactly one of the statements is true
variable (h : (stmt1 ∧ ¬stmt2 ∧ ¬stmt3) ∨ (¬stmt1 ∧ stmt2 ∧ ¬stmt3) ∨ (¬stmt1 ∧ ¬stmt2 ∧ stmt3))

-- Statement to prove the ranking order
theorem team_ranking : David_E > Emma_E ∧ Emma_E > Frank_E :=
by
  sorry

end team_ranking_l31_31677


namespace Miss_Adamson_paper_usage_l31_31464

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l31_31464


namespace Vermont_clicked_on_185_ads_l31_31989

-- Definitions for the number of ads on each web page
def ads_on_first := 18
def ads_on_second := 2 * ads_on_first
def ads_on_third := ads_on_second + 32
def ads_on_fourth := (5 / 8) * ads_on_second -- Note: We'll round in the calculations.
def ads_on_fifth := ads_on_third + 15
def ads_on_sixth := (ads_on_first + ads_on_second + ads_on_third) - 42

-- Total number of ads calculated
def total_ads := ads_on_first + ads_on_second + ads_on_third + Nat.round (ads_on_fourth) + ads_on_fifth + ads_on_sixth

-- Fraction of ads clicked
def fraction_clicked := 3 / 5

-- Total number of ads clicked (rounded to the nearest whole number)
def ads_clicked : Nat := Nat.round (fraction_clicked * total_ads)

-- Proof statement
theorem Vermont_clicked_on_185_ads : ads_clicked = 185 :=
by
  sorry

end Vermont_clicked_on_185_ads_l31_31989


namespace number_of_chords_l31_31084

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31084


namespace solution_to_quadratic_inequality_l31_31279

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 - 5 * x > 9

theorem solution_to_quadratic_inequality (x : ℝ) : quadratic_inequality x ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_to_quadratic_inequality_l31_31279


namespace brinley_animals_count_l31_31692

theorem brinley_animals_count :
  let snakes := 100
  let arctic_foxes := 80
  let leopards := 20
  let bee_eaters := 10 * ((snakes / 2) + (2 * leopards))
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 481340 := by
  sorry

end brinley_animals_count_l31_31692


namespace largest_house_number_l31_31847

-- Define the sum of the digits of John's phone number
def phone_number_digits_sum : ℕ := 4 + 3 + 1 + 7 + 8 + 2

-- Condition: the sum of John's phone number digits
lemma phone_number_sum : phone_number_digits_sum = 25 := by
  unfold phone_number_digits_sum
  norm_num

-- Define the house number as a list of digits
def house_number_digits : list ℕ := [9, 8, 7, 1, 0]

-- Prove the sum of the house number digits is 25
lemma house_number_sum : house_number_digits.sum = 25 := by
  unfold house_number_digits
  norm_num

-- Prove the digits in the house number are distinct
lemma house_number_distinct : house_number_digits.nodup := by
  unfold house_number_digits
  norm_num

-- Define the house number
def house_number : ℕ := 98710

-- Combine proofs to show the largest possible house number
theorem largest_house_number :
  list.sum house_number_digits = phone_number_digits_sum ∧
  house_number_digits.nodup ∧
  house_number = 98710 :=
by
  unfold house_number house_number_digits phone_number_digits_sum
  norm_num
  exact ⟨rfl, house_number_distinct, rfl⟩

end largest_house_number_l31_31847


namespace exists_sum_geq_3_999_l31_31740

noncomputable def sequence (x : ℕ → ℝ) (n : ℕ) : Prop := 
  x 0 = 1 ∧ ∀ (k : ℕ), 0 < x (k + 1) ∧ x (k + 1) ≤ x k

theorem exists_sum_geq_3_999 (x : ℕ → ℝ) (hn : sequence x) : 
  ∃ n ≥ 1, ∑ i in Finset.range n, (x i) ^ 2 / (x (i + 1)) ≥ 3.999 := 
sorry

end exists_sum_geq_3_999_l31_31740


namespace weight_of_a_l31_31591

variables (a b c d e : ℝ)

theorem weight_of_a (h1 : (a + b + c) / 3 = 80)
                    (h2 : (a + b + c + d) / 4 = 82)
                    (h3 : e = d + 3)
                    (h4 : (b + c + d + e) / 4 = 81) :
  a = 95 :=
by
  sorry

end weight_of_a_l31_31591


namespace chords_from_nine_points_l31_31075

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l31_31075


namespace largest_of_five_consecutive_non_primes_under_40_l31_31709

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n 

theorem largest_of_five_consecutive_non_primes_under_40 :
  ∃ x, (x > 9) ∧ (x + 4 < 40) ∧ 
       (¬ is_prime x) ∧
       (¬ is_prime (x + 1)) ∧
       (¬ is_prime (x + 2)) ∧
       (¬ is_prime (x + 3)) ∧
       (¬ is_prime (x + 4)) ∧
       (x + 4 = 36) :=
sorry

end largest_of_five_consecutive_non_primes_under_40_l31_31709


namespace maximum_t_equilateral_l31_31395

-- Given conditions definition
variables {A B C D E F : Point}
variable [Triangle : Triangle ABC]
variable [Angle60 : Angle A = 60]
variable [AngleBisectorAD : AngleBisector A D B C]
variable [PerpendicularDE : Perpendicular DE AB]
variable [PerpendicularDF : Perpendicular DF AC]

-- Definition for areas of triangles DEF and ABC
def area_DEF : ℝ := Area DEF
def area_ABC : ℝ := Area ABC

-- Definition for ratio t
def t : ℝ := area_DEF / area_ABC

-- The mathematical goal to prove
theorem maximum_t_equilateral (h : is_maximum t) : is_equilateral ABC :=
sorry

end maximum_t_equilateral_l31_31395


namespace chords_from_nine_points_l31_31047

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31047


namespace bekahs_reading_l31_31661

def pages_per_day (total_pages read_pages days_left : ℕ) : ℕ :=
  (total_pages - read_pages) / days_left

theorem bekahs_reading :
  pages_per_day 408 113 5 = 59 := by
  sorry

end bekahs_reading_l31_31661


namespace expected_socks_to_pair_l31_31140

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l31_31140


namespace solve_for_x_l31_31566

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 3029) : x = 200.4 :=
by
  sorry

end solve_for_x_l31_31566


namespace no_prime_difference_in_sequence_l31_31006

open Nat

theorem no_prime_difference_in_sequence (k : ℤ) :
  ¬ (∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ (4 + 10 * k).natAbs = p1 - p2) :=
by
  sorry

end no_prime_difference_in_sequence_l31_31006


namespace number_is_18_l31_31555

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end number_is_18_l31_31555


namespace probability_of_at_least_2_girls_equals_specified_value_l31_31514

def num_combinations (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def probability_at_least_2_girls : ℚ :=
  let total_committees := num_combinations 24 5
  let all_boys := num_combinations 14 5
  let one_girl_four_boys := num_combinations 10 1 * num_combinations 14 4
  let at_least_2_girls := total_committees - (all_boys + one_girl_four_boys)
  at_least_2_girls / total_committees

theorem probability_of_at_least_2_girls_equals_specified_value :
  probability_at_least_2_girls = 2541 / 3542 := 
sorry

end probability_of_at_least_2_girls_equals_specified_value_l31_31514


namespace rationalize_denominator_l31_31490

theorem rationalize_denominator :
  exists A B C D : ℤ, D > 0 ∧ ¬ ( ∃ p : ℤ, nat.prime p ∧ p ∣ B ) ∧
  ( A * sqrt B + C ) / D = ( 5 + sqrt 2 ) / 4 ∧ 
  A + B + C + D = 12 :=
begin
  sorry
end

end rationalize_denominator_l31_31490


namespace find_distance_A_to_B_l31_31897

-- Definitions of all involved conditions and entities
constant PersonA_StartTime : ℕ := 0   -- Time in minutes, 8:00 is represented as 0
constant PersonB_StartTime : ℕ := 20  -- Time in minutes, 8:20 is represented as 20
constant PersonC_StartTime : ℕ := 30  -- Time in minutes, 8:30 is represented as 30

-- Person C travels for 10 minutes
constant PersonC_TravelTime : ℕ := 10

-- Distance from Person C to point B after 10 minutes
constant PersonC_to_B_Distance : ℕ := 2015 -- in meters

-- They all travel at the same speed
constant TravelSpeed : ℕ

-- Definition of the distance between points A and B
noncomputable def Distance_A_to_B : ℕ := 2418 -- in meters

-- Theorem statement
theorem find_distance_A_to_B :
  ∀ (distanceAB: ℕ),
    (PersonA_StartTime + 40 = PersonB_StartTime + (distanceAB / TravelSpeed * 1 / 2))
    ∧ (PersonC_StartTime + PersonC_TravelTime = distanceAB - PersonC_to_B_Distance) 
    → distanceAB = 2418 :=
begin
  sorry
end

end find_distance_A_to_B_l31_31897


namespace cone_circumference_l31_31229

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

noncomputable def base_circumference (r : ℝ) : ℝ :=
  2 * π * r

theorem cone_circumference (V h : ℝ) (hV : V = 27 * π) (hh : h = 9) :
  ∃ C, C = 6 * π :=
by
  -- Definitions
  let r := sqrt (V * 3 / (π * h))
  have h_r : r = 3 := by
    field_simp [hV, hh]
    norm_num
  -- Circumference calculation
  use base_circumference r
  rw [h_r]
  norm_num
  -- Demonstrate resulting circumference
  rfl
  sorry

end cone_circumference_l31_31229


namespace total_sheets_of_paper_l31_31463

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l31_31463


namespace inequality_form_l31_31703

variable (x : ℝ)

theorem inequality_form :
  x + 4 < 10 ↔ "x plus 4 is less than 10" := sorry

end inequality_form_l31_31703


namespace polygon_sides_from_diagonals_l31_31385

theorem polygon_sides_from_diagonals (D : ℕ) (hD : D = 16) : 
  ∃ n : ℕ, 2 * D = n * (n - 3) ∧ n = 7 :=
by
  use 7
  simp [hD]
  norm_num
  sorry

end polygon_sides_from_diagonals_l31_31385


namespace shortest_distance_to_circle_from_origin_l31_31567

-- Definitions derived from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 - 18 * x + y^2 - 8 * y + 153 = 0

-- Statement we need to prove
theorem shortest_distance_to_circle_from_origin :
  ∃ d : ℝ, d = (Real.sqrt 97 - Real.sqrt 44) ∧
  ∀ (x y : ℝ), circle_equation x y → Real.dist (0, 0) (9, 4) = Real.sqrt 97 ∧
  ∀ (x y : ℝ), circle_equation x y → Real.dist (0, 0) (x, y) = d + Real.sqrt 44 :=
sorry

end shortest_distance_to_circle_from_origin_l31_31567


namespace problem_statement_l31_31221

noncomputable def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := derivative f x

theorem problem_statement
  (h : ∀ x ∈ Ioo (0 : ℝ) (Real.pi / 2), sin x * f_prime x > cos x * f x) :
  sqrt 3 * f (Real.pi / 6) < f (Real.pi / 3) :=
by {
  sorry
}

end problem_statement_l31_31221


namespace quadratic_factorization_l31_31285

open Complex

theorem quadratic_factorization :
  ∀ (x : ℂ), 2 * x^2 - 4 * x + 5 = (sqrt 2 * x - sqrt 2 + sqrt 3 * Complex.i) * (sqrt 2 * x - sqrt 2 - sqrt 3 * Complex.i) :=
by
  intro x
  sorry

end quadratic_factorization_l31_31285


namespace caden_coins_l31_31257

/-- Caden's coin jar problem statement -/
theorem caden_coins :
  ∀ (P N D Q : ℕ), 
    2 * Q = D → 
    5 * N = D → 
    P = 120 → 
    120 * 0.01 + N * 0.05 + D * 0.10 + Q * 0.25 = 8 →
    P / N = 3 / 1 :=
by sorry

end caden_coins_l31_31257


namespace socks_expected_value_l31_31145

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l31_31145


namespace num_two_digit_values_l31_31024

def sumOfDigits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem num_two_digit_values (n : ℕ) :
  (10 ≤ n ∧ n < 100) →
  (sumOfDigits (sumOfDigits n) = 3) →
  (finset.univ.filter (λ x, 10 ≤ x ∧ x < 100 ∧ sumOfDigits (sumOfDigits x) = 3)).card = 10 :=
sorry

end num_two_digit_values_l31_31024


namespace scheduled_conference_games_total_l31_31515

def number_of_teams_in_A := 7
def number_of_teams_in_B := 5
def games_within_division (n : Nat) : Nat := n * (n - 1)
def interdivision_games := 7 * 5
def rivalry_games := 7

theorem scheduled_conference_games_total : 
  let games_A := games_within_division number_of_teams_in_A
  let games_B := games_within_division number_of_teams_in_B
  let total_games := games_A + games_B + interdivision_games + rivalry_games
  total_games = 104 :=
by
  sorry

end scheduled_conference_games_total_l31_31515


namespace julia_gold_watch_percentage_l31_31436

def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def total_watches_before_gold : ℕ := silver_watches + bronze_watches
def total_watches_after_gold : ℕ := 88
def gold_watches : ℕ := total_watches_after_gold - total_watches_before_gold
def percentage_gold_watches : ℚ := (gold_watches : ℚ) / (total_watches_after_gold : ℚ) * 100

theorem julia_gold_watch_percentage :
  percentage_gold_watches = 9.09 := by
  sorry

end julia_gold_watch_percentage_l31_31436


namespace center_of_symmetry_g_l31_31345

-- Define the given functions and symmetry condition
def f (x : ℝ) := Real.sin (2 * x + π / 6)
def symmetry_line := π / 12

-- Define the conjecture for the center of symmetry of g(x)
def g (x : ℝ) := Real.cos (2 * x)
def center_of_symmetry := (π / 4, 0 : ℝ)

theorem center_of_symmetry_g {x : ℝ} :
  (f (π / 6 - x) = Real.sin (π / 3 - 2 * x + π / 6) →
   Real.sin (π / 2 - 2 * x) = Real.cos (2 * x) →
   g (x) = Real.cos (2 * x)) →
  (π / 4, 0) ∈ set_of (λ (p : ℝ × ℝ), p.1 ∈ set_of (λ x, ∃ k : ℤ, x = k * π / 2 + π / 4) ∧ p.2 = 0) :=
by
  intros h
  sorry

end center_of_symmetry_g_l31_31345


namespace inequality_am_gm_l31_31754

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
by sorry

end inequality_am_gm_l31_31754


namespace number_of_chords_l31_31082

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l31_31082


namespace sum_of_distances_equals_60_l31_31027

def distance (a b : ℝ × ℝ) : ℝ := 
  real.sqrt ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2)

def triangle_vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := 
  ((0, 0), (8, 0), (1, 7))

def point_p : ℝ × ℝ := (3, 4)

def m := 5
def n := 1
def p := 41
def q := 13

theorem sum_of_distances_equals_60 :
  let A := (0, 0)
      B := (8, 0)
      C := (1, 7)
      P := (3, 4)
      AP := distance A P
      BP := distance B P
      CP := distance C P in
  AP + BP + CP = m + real.sqrt p + n * real.sqrt q ∧ m + n + p + q = 60 :=
by {
  sorry
}

end sum_of_distances_equals_60_l31_31027


namespace forty_percent_jacqueline_candy_l31_31300

def fred_candy : ℕ := 12
def uncle_bob_candy : ℕ := fred_candy + 6
def total_fred_uncle_bob_candy : ℕ := fred_candy + uncle_bob_candy
def jacqueline_candy : ℕ := 10 * total_fred_uncle_bob_candy

theorem forty_percent_jacqueline_candy : (40 * jacqueline_candy) / 100 = 120 := by
  sorry

end forty_percent_jacqueline_candy_l31_31300


namespace gain_is_25_percent_l31_31584

variable (purchasePrice : ℝ) (sellingPrice : ℝ) (waterPercentage : ℝ)

def percentage_gain (purchasePrice sellingPrice : ℝ) : ℝ :=
  ((sellingPrice - purchasePrice) / purchasePrice) * 100

theorem gain_is_25_percent
  (h1 : purchasePrice = 12)
  (h2 : waterPercentage = 0.2)
  (h3 : sellingPrice = 15) :
  percentage_gain purchasePrice sellingPrice = 25 := by
  sorry

end gain_is_25_percent_l31_31584


namespace words_between_CZYEB_and_XCEDA_l31_31301

-- Mapping from letters to their numeric values
def letter_to_number : Char → ℕ
| 'A' := 0
| 'B' := 1
| 'C' := 2
| 'D' := 3
| 'E' := 4
| 'X' := 5
| 'Y' := 6
| 'Z' := 7
| _ := sorry  -- This handles unexpected characters

-- Function to convert a 5-letter word to its equivalent base-8 number
def word_to_base8 (word : List Char) : ℕ :=
  word.enum.sum (λ ⟨i, c⟩, letter_to_number c * 8^i)

-- Function to count the words between two given words in lexicographic order
def count_words_between (word1 word2 : List Char) : ℕ :=
  let n1 := word_to_base8 word1
  let n2 := word_to_base8 word2
  n2 - n1 - 1

-- The proof problem
theorem words_between_CZYEB_and_XCEDA :
  count_words_between ['C', 'Z', 'Y', 'E', 'B'] ['X', 'C', 'E', 'D', 'A'] = 9590 :=
by
  sorry

end words_between_CZYEB_and_XCEDA_l31_31301


namespace opposite_neg_fraction_l31_31951

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l31_31951


namespace range_k_l31_31686

-- Define the function k(x)
def k (x : ℝ) : ℝ := (3 * x + 5) / (x - 4)

-- State the theorem about the range of k(x)
theorem range_k : (Set.range k) = { y : ℝ | y ≠ 3 } :=
by
  sorry

end range_k_l31_31686


namespace at_most_one_negative_l31_31988

theorem at_most_one_negative (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (a < 0 ∧ b >= 0 ∧ c >= 0) ∨ (a >= 0 ∧ b < 0 ∧ c >= 0) ∨ (a >= 0 ∧ b >= 0 ∧ c < 0) ∨ 
  (a >= 0 ∧ b >= 0 ∧ c >= 0) :=
sorry

end at_most_one_negative_l31_31988


namespace football_club_balance_l31_31624

/-- A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. Prove that the final balance is $60 million. -/
theorem football_club_balance :
  let initial_balance := 100
  let income_from_sales := 10 * 2
  let expenditure_on_purchases := 15 * 4
  let final_balance := initial_balance + income_from_sales - expenditure_on_purchases
  final_balance = 60 :=
by
  simp only [initial_balance, income_from_sales, expenditure_on_purchases, final_balance]
  sorry

end football_club_balance_l31_31624


namespace angle_ADB_fixed_l31_31315

section
variables (a b : ℝ) (h1 : a > b > 0) (e : ℝ := 1/2)
def ellipse_equation (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def triangle_area (y : ℝ) : Prop := (1/2 * 1 * y * 2) = 3/2

noncomputable def ellipse_parameters : Prop :=
∃ (a b : ℝ), e = sqrt(1 - b^2 / a^2) ∧ ellipse_equation 1 (3/2)

lemma find_ellipse_equation (a b : ℝ) (h1 : a > b > 0) (h2 : ellipse_parameters) :
  ellipse_equation 1 (3/2) :=
sorry

variables (x1 y1 x2 y2 : ℝ) (h2 : (2/7, 0)) (h3 : ellipse_equation x1 y1) (h4 : ellipse_equation x2 y2)
variables (D : ℝ × ℝ) (h5 : D = (a, 0)) (h6 : ADB x1 y1 x2 y2)

noncomputable def ADB (x1 y1 x2 y2 : ℝ) : Prop :=
let A := (x1, y1), B := (x2, y2) in
let DA := (x1 - a, y1), DB := (x2 - a, y2) in
DA.1 * DB.1 + DA.2 * DB.2 = 0

theorem angle_ADB_fixed (A B : ℝ × ℝ) (D : ℝ × ℝ) (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (hD : D = (a, 0)) (hAngle : ADB x1 y1 x2 y2) : 
  ADB x1 y1 x2 y2 :=
sorry
end

end angle_ADB_fixed_l31_31315


namespace sum_of_common_divisors_l31_31155

theorem sum_of_common_divisors : 
  let d := { n ∈ {60, 120, -30, 180, 240} | ∀ x, x ∣ n } in
  (1  + 2 + 3 + 5 + 6 = 17) := by
  sorry

end sum_of_common_divisors_l31_31155


namespace binomial_coefficient_times_two_l31_31174

theorem binomial_coefficient_times_two : 2 * Nat.choose 8 5 = 112 := 
by 
  -- The proof is omitted here
  sorry

end binomial_coefficient_times_two_l31_31174


namespace gcf_120_180_240_is_60_l31_31175

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l31_31175


namespace exists_division_of_triangle_into_congruent_parts_l31_31008

theorem exists_division_of_triangle_into_congruent_parts :
  ∃ (T : Triangle), (is_right_triangle T) ∧ (∃ (P1 P2 : Point), divides_into_congruent_triangles T P1 P2 3 2) :=
by
  sorry

end exists_division_of_triangle_into_congruent_parts_l31_31008


namespace largest_prime_factor_of_sum_l31_31219

theorem largest_prime_factor_of_sum (seq : List ℤ) (h_seq : seq.length > 0)
  (h_digits_cyclic : ∀ i, seq[i % seq.length] % 10 = (seq[(i + 1) % seq.length] / 1000) ∧
                          (seq[i % seq.length] / 10 % 10) = (seq[(i + 1) % seq.length] / 100 % 10) ∧
                          (seq[i % seq.length] / 100 % 10) = (seq[(i + 1) % seq.length] / 10 % 10)) : 
  101 ∣ seq.sum := 
sorry

end largest_prime_factor_of_sum_l31_31219


namespace nine_points_circle_chords_l31_31070

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l31_31070


namespace chords_from_nine_points_l31_31045

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31045


namespace b_3_pow_100_l31_31864

-- Define the sequence b_n
def b : ℕ → ℕ
| 1       := 2
| (3 * n) := n * b n
| _       := 0  -- This case should not happen based on the problem's conditions

-- We start by defining the above equivalency within a theorem setting.
theorem b_3_pow_100 :
  b (3^100) = 2 * 3^99 := by
  -- Proof would go here
  sorry

end b_3_pow_100_l31_31864


namespace average_score_assigned_day_l31_31400

theorem average_score_assigned_day (n : ℕ) (p : ℝ) (q : ℝ) (m_avg : ℝ) (total_avg : ℝ)
  (h1 : n = 100)
  (h2 : p = 0.70)
  (h3 : q = 0.95)
  (h4 : total_avg = 67) :
  (70 * A + 30 * q ) / 100 = total_avg →
  A = 55 :=
begin
  sorry
end

end average_score_assigned_day_l31_31400


namespace reflection_x_intercept_l31_31935

theorem reflection_x_intercept :
  let original_line := λ x : ℝ, 2 * x - 6,
      reflected_line := λ x : ℝ, -2 * x - 6
  in (∃ x : ℝ, reflected_line x = 0) ↔ x = -3 :=
by
  let original_line := λ x : ℝ, 2 * x - 6
  let reflected_line := λ x : ℝ, -2 * x - 6
  have h : (∃ x : ℝ, reflected_line x = -6) ↔ (x = -3),
    from sorry
  exact h

end reflection_x_intercept_l31_31935


namespace tangent_line_eq_range_of_m_l31_31343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 + a * x^2 - 2 * x - 1

theorem tangent_line_eq (a : ℝ) :
  (∀ x : ℝ, f a x = (1 / 3) * x^3 + a * x^2 - 2 * x - 1) →
  ∃ (m b : ℝ), (m = -2) ∧ (b = -1) ∧ (∀ (y : ℝ), y = m * x + b → (2 * x + y + 1 = 0)) :=
begin
  sorry -- Proof not required
end

theorem range_of_m (a : ℝ) :
  f a 1 - f a 0 = 0 →
  a = 1 / 2 →
  ∃ (m : ℝ), m ∈ Ioo (-13/6) (7/3) :=
begin
  sorry -- Proof not required
end

end tangent_line_eq_range_of_m_l31_31343


namespace sqrt_meaningful_range_l31_31392

theorem sqrt_meaningful_range (x : ℝ) : (∃ y, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by sorry

end sqrt_meaningful_range_l31_31392


namespace decrease_percent_in_revenue_l31_31136

variable (T C : ℝ)

-- Assume the conditions
def original_revenue := T * C
def new_tax_rate := 0.8 * T
def new_consumption := 1.1 * C
def new_revenue := new_tax_rate * new_consumption

-- Prove that the decrease percent in the revenue is 12%
theorem decrease_percent_in_revenue : 
  (original_revenue - new_revenue) / original_revenue * 100 = 12 := 
by
  -- Definitions derived from conditions
  have h1 : original_revenue = T * C := rfl
  have h2 : new_revenue = 0.88 * T * C := by rw [new_tax_rate, new_consumption]; ring
  
  -- Calculate decrease in revenue
  have h3 : original_revenue - new_revenue = (T * C) - (0.88 * T * C) := by rw [h1, h2]
  have h4 : original_revenue - new_revenue = 0.12 * T * C := by ring
  
  -- Calculate percent decrease
  have h5 : (original_revenue - new_revenue) / original_revenue = 0.12 := 
    by rw [h4, h1]; field_simp [T * C ≠ 0]
  have h6 : (original_revenue - new_revenue) / original_revenue * 100 = 0.12 * 100 := by rw [h5]
  
  -- Conclude the proof
  exact h6.trans (by norm_num)

end decrease_percent_in_revenue_l31_31136


namespace log_sum_range_l31_31326

theorem log_sum_range {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h : Real.log (x + y) / Real.log 2 = Real.log x / Real.log 2 + Real.log y / Real.log 2) :
  4 ≤ x + y :=
by
  sorry

end log_sum_range_l31_31326


namespace rashmi_bus_stop_distance_l31_31589

theorem rashmi_bus_stop_distance :
  let t := 11 / 6 in
  let d := 10 in
  d = (5 * (t + 10 / 60)) ∧ d = (6 * (t - 10 / 60)) :=
by
  sorry

end rashmi_bus_stop_distance_l31_31589


namespace trigonometric_identity_l31_31495

theorem trigonometric_identity (α : ℝ) (hα : α = 20) : 
  cos α + cos (3 * α) + cos (5 * α) + cos (7 * α) = 1 / 2 := 
by 
  sorry

end trigonometric_identity_l31_31495


namespace find_diameter_of_wheel_l31_31243

noncomputable def diameter_of_wheel : ℝ := sorry

theorem find_diameter_of_wheel :
  let total_distance := 3520
  let number_of_revolutions := 40.03639672429481
  let pi := Real.pi
  ∃ d : ℝ, (total_distance / number_of_revolutions = pi * d) ∧ (d ≈ 27.993) := sorry

end find_diameter_of_wheel_l31_31243


namespace required_number_l31_31573

-- Define the main variables and conditions
variables {i : ℂ} (z : ℂ)
axiom i_squared : i^2 = -1

-- State the theorem that needs to be proved
theorem required_number (h : z + (4 - 8 * i) = 1 + 10 * i) : z = -3 + 18 * i :=
by {
  -- the exact steps for the proof will follow here
  sorry
}

end required_number_l31_31573


namespace find_number_l31_31382

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number_l31_31382


namespace initial_speed_of_cyclist_l31_31225

def walking_speed : ℝ := 5 -- walking speed is 5 km/h
def delay : ℝ := 4 + 24 / 60 -- cyclist starts 4 hours and 24 minutes later
def burst_distance : ℝ := 8 -- cyclist travels 8 km before the inner tube bursts
def change_time : ℝ := 10 / 60 -- time taken to change the inner tube is 10 minutes, converted to hours
def increased_speed_factor : ℝ := 2 -- the cyclist speeds up by 2 km/h after the stop

-- We are to prove that the initial speed of the cyclist is 16 km/h given the conditions

theorem initial_speed_of_cyclist :
  ∃ x : ℝ, 
    (∀ t : ℝ, 
      ((t*x = walking_speed * (t + delay) + 22) ∧ 
       (burst_distance + (t - burst_distance / x - change_time) * (x + increased_speed_factor) = walking_speed * (t + delay) + 22)) ↔ 
       x = 16) :=
by
  sorry

end initial_speed_of_cyclist_l31_31225


namespace value_of_a_l31_31802

-- Definition of the function and the point
def graph_function (x : ℝ) : ℝ := -x^2
def point_lies_on_graph (a : ℝ) : Prop := (a, -9) ∈ {p : ℝ × ℝ | p.2 = graph_function p.1}

-- The theorem stating that if the point (a, -9) lies on the graph of y = -x^2, then a = ±3
theorem value_of_a (a : ℝ) (h : point_lies_on_graph a) : a = 3 ∨ a = -3 :=
by 
  sorry

end value_of_a_l31_31802


namespace opposite_of_neg_frac_l31_31943

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31943


namespace opposite_of_neg_one_div_2023_l31_31941

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end opposite_of_neg_one_div_2023_l31_31941


namespace arrange_COMMUNICATION_l31_31685

theorem arrange_COMMUNICATION : 
  let n := 12
  let o_count := 2
  let i_count := 2
  let n_count := 2
  let m_count := 2
  let total_repeats := o_count * i_count * n_count * m_count
  n.factorial / (o_count.factorial * i_count.factorial * n_count.factorial * m_count.factorial) = 29937600 :=
by sorry

end arrange_COMMUNICATION_l31_31685


namespace train_pass_time_correct_l31_31201

-- Define the length of the train
def train_length : ℝ := 50

-- Define the speed of the train in kmph
def speed_kmph : ℝ := 36

-- Conversion factor from kmph to m/s
def conversion_factor (kmph: ℝ) : ℝ :=
  kmph * 1000 / 3600

-- Define the speed of the train in m/s using the conversion factor
def speed_mps : ℝ :=
  conversion_factor speed_kmph

-- Given the length of the train and its speed, calculate the time to pass the telegraph post
def time_to_pass_telegraph_post : ℝ :=
  train_length / speed_mps

-- The proof statement: the time to pass the telegraph post equals 5 seconds
theorem train_pass_time_correct : time_to_pass_telegraph_post = 5 := by
  sorry

end train_pass_time_correct_l31_31201


namespace average_marks_failed_l31_31919

variable {TotalCandidates : ℕ}
variable {PassedCandidates : ℕ}
variable {TotalAverage : ℝ}
variable {PassedAverage : ℝ}
variable {FailedAverage : ℝ}
variable {FailedCandidates : ℕ}

-- Conditions extracted from the problem
def conditions : Prop :=
  TotalCandidates = 120 ∧
  PassedCandidates = 100 ∧
  TotalAverage = 35 ∧
  PassedAverage = 39 ∧
  FailedCandidates = TotalCandidates - PassedCandidates

-- Expression of the total marks
def total_marks (TotalCandidates : ℕ) (TotalAverage : ℝ) : ℝ :=
  TotalCandidates * TotalAverage

def total_marks_passed (PassedCandidates : ℕ) (PassedAverage : ℝ) : ℝ :=
  PassedCandidates * PassedAverage

def total_marks_failed (FailedCandidates : ℕ) (FailedAverage : ℝ) : ℝ :=
  FailedCandidates * FailedAverage

-- Proof goal
theorem average_marks_failed :
  conditions →
  total_marks TotalCandidates TotalAverage =
  total_marks_passed PassedCandidates PassedAverage +
  total_marks_failed (TotalCandidates - PassedCandidates) FailedAverage →
  FailedAverage = 15 :=
by
  intros h_conditions h_equation
  -- Since we're skipping the proof details, we add sorry
  sorry

end average_marks_failed_l31_31919


namespace farmer_profit_is_40_l31_31617

def seeds_per_ear := 4
def selling_price_per_ear := 0.1
def cost_per_bag := 0.5
def seeds_per_bag := 100
def ears_sold := 500

noncomputable def calculate_profit (seeds_per_ear : ℕ) (selling_price_per_ear : ℝ) (cost_per_bag : ℝ) (seeds_per_bag : ℕ) (ears_sold : ℕ) : ℝ :=
  let seeds_needed := ears_sold * seeds_per_ear
  let bags_needed := seeds_needed / seeds_per_bag
  let total_cost := bags_needed * cost_per_bag
  let total_revenue := ears_sold * selling_price_per_ear
  total_revenue - total_cost

theorem farmer_profit_is_40 :
  calculate_profit seeds_per_ear selling_price_per_ear cost_per_bag seeds_per_bag ears_sold = 40 := 
sorry

end farmer_profit_is_40_l31_31617


namespace total_bricks_in_wall_l31_31401

theorem total_bricks_in_wall :
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  (rows.sum = 80) := 
by
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  sorry

end total_bricks_in_wall_l31_31401


namespace closest_percentage_change_l31_31011

noncomputable def total_price_before_tax : Float :=
  12.99 + 10.99 + 6.99 + 4.99 + 3.99 + 1.99

noncomputable def tax_rate : Float :=
  0.10

noncomputable def total_price_including_tax : Float :=
  total_price_before_tax * (1 + tax_rate)

noncomputable def initial_payment : Float :=
  50.0

noncomputable def change_received : Float :=
  initial_payment - total_price_including_tax

noncomputable def percentage_change : Float :=
  (change_received / initial_payment) * 100

theorem closest_percentage_change :
  percentage_change ≈ 7.74 :=
by
  -- Proof omitted
  sorry

end closest_percentage_change_l31_31011


namespace mike_marbles_l31_31457

theorem mike_marbles (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 8) 
  (h2 : given = 4)
  (h3 : final = original - given) : 
  final = 4 :=
by sorry

end mike_marbles_l31_31457


namespace transformation_correct_l31_31548

theorem transformation_correct:
  ∀ x : ℝ, 
  2 * sin (x) = 2 * sin (3 * (x - ( -π / 6) )) :=
by
  sorry

end transformation_correct_l31_31548


namespace pat_donut_selections_l31_31478

theorem pat_donut_selections : ∃ (n : ℕ), n = 10 :=
  let g' := 0
  let c' := 0
  let p' := 0
  let s' := 0
  have h : g' + c' + p' + s' = 2 := by sorry
  have binomial_calc := (5.choose 3) = 10 := by sorry
  ⟨10, binomial_calc⟩

end pat_donut_selections_l31_31478


namespace average_last_4_matches_l31_31590

theorem average_last_4_matches 
  (avg_10 : ℝ) (avg_6 : ℝ) (result : ℝ)
  (h1 : avg_10 = 38.9)
  (h2 : avg_6 = 42)
  (h3 : result = 34.25) :
  let total_runs_10 := avg_10 * 10
  let total_runs_6 := avg_6 * 6
  let total_runs_4 := total_runs_10 - total_runs_6
  let avg_4 := total_runs_4 / 4
  avg_4 = result :=
  sorry

end average_last_4_matches_l31_31590


namespace student_rank_from_right_l31_31645

theorem student_rank_from_right (n m : ℕ) (h1 : n = 8) (h2 : m = 20) : m - (n - 1) = 13 :=
by
  sorry

end student_rank_from_right_l31_31645


namespace count_squares_in_H_l31_31971

def H : set (ℤ × ℤ) := { p | let (x, y) := p in 2 ≤ |x| ∧ |x| ≤ 8 ∧ 2 ≤ |y| ∧ |y| ≤ 8 }

theorem count_squares_in_H : 
  (∃ n, n = 9 + 4 + 1 ∧ ∀ square ∈ finset.filter (λ s, 5 ≤ (geometry.square_side_length s)) (geometry.squares_with_vertices_in_set H), true) :=
sorry

end count_squares_in_H_l31_31971


namespace sum_possible_values_of_k_l31_31414

theorem sum_possible_values_of_k (j k : ℕ) (h : (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 4)) (hj : 0 < j) (hk : 0 < k) :
  {x : ℕ | (1 / (j : ℚ) + 1 / (x : ℚ) = 1 / 4) ∧ 0 < x}.sum id = 51 :=
sorry

end sum_possible_values_of_k_l31_31414


namespace sum_of_solutions_l31_31534

theorem sum_of_solutions (a b c : ℝ) (x : ℝ) :
  let quadratic_eqn := λ x, x^2 - 6 * x - 8 - 4 * x - 20
  roots_sum := -((-10) / 1)
  quadratic_eqn x = 0 → roots_sum = 10 :=
by
  sorry

end sum_of_solutions_l31_31534


namespace Z_is_1_5_decades_younger_l31_31962

theorem Z_is_1_5_decades_younger (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end Z_is_1_5_decades_younger_l31_31962


namespace chords_from_nine_points_l31_31051

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l31_31051


namespace distance_A_B_l31_31429

theorem distance_A_B (D : ℝ) 
  (rowing_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : rowing_speed + stream_speed) 
  (upstream_speed : rowing_speed - stream_speed) 
  (time_downstream time_upstream : ℝ) 
  (dist_eq_downstream : D = 6 * time_downstream) 
  (dist_eq_upstream : D = 4 * time_upstream) 
  (time_eq : time_downstream + time_upstream = 5):
  D = 12 := 
by
  -- Conditions
  have rowing_speed := 5
  have stream_speed := 1
  have total_time := 5
  have downstream_speed := rowing_speed + stream_speed
  have upstream_speed := rowing_speed - stream_speed

  -- Define intermediate variables and equations use in the proof
  have T1 := time_downstream
  have T2 := time_upstream
  have dist_eq_down := dist_eq_downstream
  have dist_eq_up := dist_eq_upstream

  -- Show D = 12 using the given conditions
  sorry

end distance_A_B_l31_31429


namespace remainder_of_x_plus_2_pow_2022_l31_31564

theorem remainder_of_x_plus_2_pow_2022 (x : ℂ) :
  ∃ r : ℂ, ∃ q : ℂ, (x + 2)^2022 = q * (x^2 - x + 1) + r ∧ (r = x) :=
by
  sorry

end remainder_of_x_plus_2_pow_2022_l31_31564


namespace find_f_and_extrema_l31_31728

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * real.sqrt 3 * real.cos (ω * x) * real.sin (ω * x) - 2 * (real.cos (ω * x))^2 + 1

theorem find_f_and_extrema (ω : ℝ) (hω : ω > 0)
  (hT : ∀ x, f ω (x + π) = f ω x) :
  (∀ x, f ω x = 2 * real.sin (2 * x - π / 6)) ∧
  (∀ x, x ∈ Icc (0 : ℝ) (π / 2) → 
    (f ω x ≤ 2 ∧ f ω x ≥ -1) ∧ 
    (f ω 0 = -1 ∧ f ω (π / 3) = 2)) :=
sorry

end find_f_and_extrema_l31_31728


namespace find_m_l31_31369

noncomputable theory
  
-- Define the vectors
def a (m : ℝ) : ℝ × ℝ := (m, 1)
def b : ℝ × ℝ := (2, 1)

-- Condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  (∃ k : ℝ, u = k • v)

-- Given condition
def given_condition (m : ℝ) : Prop :=
  parallel (a m - 2 • b) b

-- Goal: Prove that given the condition, m = 2
theorem find_m (m : ℝ) : given_condition m → m = 2 :=
by
  sorry

end find_m_l31_31369


namespace inequality_and_equality_l31_31496

variables {x y z : ℝ}

theorem inequality_and_equality (x y z : ℝ) :
  (x^2 + y^4 + z^6 >= x * y^2 + y^2 * z^3 + x * z^3) ∧ (x^2 + y^4 + z^6 = x * y^2 + y^2 * z^3 + x * z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end inequality_and_equality_l31_31496


namespace horizontal_axis_represents_independent_variable_l31_31193

-- Define what it means for points on the horizontal axis usually represent
def points_on_horizontal_axis (x : Type) : Prop :=
  x = "Independent variable"

-- Main theorem statement
theorem horizontal_axis_represents_independent_variable :
  points_on_horizontal_axis "Independent variable" :=
by 
  -- This is where the proof would go
  sorry

end horizontal_axis_represents_independent_variable_l31_31193


namespace determine_p_l31_31679

theorem determine_p (p : ℝ) (h : (2 * p - 1) * (-1)^2 + 2 * (1 - p) * (-1) + 3 * p = 0) : p = 3 / 7 := by
  sorry

end determine_p_l31_31679


namespace angle_B_is_2pi_over_3_eq_a_1_or_3_l31_31003

-- Definitions based on the conditions
variables (A B C : ℝ) (a b c : ℝ)
hypothesis (triangle_inequality : A + B + C = π)
hypothesis (cos_ratio : cos B / cos C = -b / (2 * a + c))

-- Question 1: Prove that B = 2π / 3 given the condition
theorem angle_B_is_2pi_over_3 
  (h : cos B / cos C = -b / (2 * a + c)) 
  (triangle_inequality : A + B + C = π) : 
  B = 2 * π / 3 :=
sorry

-- Definitions for further conditions in Part 2
variable (B_2pi_over_3 : B = 2 * π / 3)
variable (b_value : b = sqrt 13)
variable (sum_a_c : a + c = 4)

-- Question 2: Prove that for the given values, a is either 1 or 3
theorem eq_a_1_or_3 
  (h_cos_ratio : cos B / cos C = -b / (2 * a + c)) 
  (triangle_inequality : A + B + C = π) 
  (B_2pi_over_3 : B = 2 * π / 3) 
  (b_value : b = sqrt 13) 
  (sum_a_c : a + c = 4) : 
  a = 1 ∨ a = 3 :=
sorry

end angle_B_is_2pi_over_3_eq_a_1_or_3_l31_31003


namespace total_sheets_of_paper_l31_31461

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l31_31461


namespace distance_between_A_and_B_l31_31660

theorem distance_between_A_and_B :
  let A := (0, 0)
  let B := (-10, 24)
  dist A B = 26 :=
by
  sorry

end distance_between_A_and_B_l31_31660


namespace symmetric_colors_different_at_8281_div_2_l31_31667

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2_l31_31667


namespace num_chords_l31_31038

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l31_31038


namespace reconstruct_triangle_l31_31493

-- Definitions of the lines l_b and l_c containing the bisectors of angles B and C
variable (l_b l_c : Line)

-- Definition of point L_1 as the base of the bisector of angle A
variable (L1 : Point)

-- The statement to be proved: Given the lines l_b, l_c and the point L1, we can reconstruct the triangle ABC
theorem reconstruct_triangle (l_b l_c : Line) (L1 : Point) : ∃ (A B C : Point), triangle ABC ∧
  (bisector l_b B) ∧
  (bisector l_c C) ∧
  (base_bisector L1 A) :=
  sorry

end reconstruct_triangle_l31_31493


namespace solve_first_equation_solve_second_equation_l31_31101

noncomputable def roots_eq1_solve (x : ℝ) := x = -1 + sqrt 2 / 2 ∨ x = -1 - sqrt 2 / 2

noncomputable def roots_eq2_solve (x : ℝ) := x = -3 + sqrt 14 ∨ x = -3 - sqrt 14

theorem solve_first_equation (x : ℝ) : 2*x^2 + 4*x + 1 = 0 → roots_eq1_solve x :=
sorry

theorem solve_second_equation (x : ℝ) : x^2 + 6*x = 5 → roots_eq2_solve x :=
sorry

end solve_first_equation_solve_second_equation_l31_31101


namespace find_angle_A_range_ratio_l31_31837

variables {A B C a b c : ℝ}

-- Condition 1 given in the problem
def cond1 (A B C a b c : ℝ) : Prop :=
  2 * cos A * (c * cos B + b * cos C) = a

-- Condition 2 given in the problem
def cond2 (C a b c : ℝ) : Prop :=
  sqrt 3 * sin C - b = c - a * cos C

-- Proving angle A given condition 1
theorem find_angle_A (h : cond1 A B C a b c) : A = π / 3 := by
  sorry

-- Finding the range of (bc - c^2) / a^2 given that the triangle is acute
theorem range_ratio (h : cond1 A B C a b c) (ha : A < π / 2) (hb : B < π / 2) (hc : C < π / 2) :
  ∃ k, k = (b * c - c^2) / a^2 ∧ -2 / 3 < k ∧ k < 1 / 3 := by
  sorry

end find_angle_A_range_ratio_l31_31837


namespace find_p_plus_s_l31_31865

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem find_p_plus_s (p q r s : ℝ) (h : p * q * r * s ≠ 0) 
  (hg : ∀ x : ℝ, g p q r s (g p q r s x) = x) : p + s = 0 := 
by 
  sorry

end find_p_plus_s_l31_31865


namespace general_formula_sum_inequality_l31_31313

-- Condition: Sequence α{a_n}
variable {a : ℕ+ → ℝ} 

-- Condition: S_n is the sum of the first n terms
variable {S : ℕ+ → ℝ}

-- Given condition in problem: 
axiom condition_a {n : ℕ+} : a n = (S n) / n + 2 * n - 2

-- Given condition in problem: 
axiom condition_b: S 2 = 6

-- Correct answers as targets to prove

-- General formula for the sequence ∀ n, a n = 4n - 3
theorem general_formula : ∀ n : ℕ+, a n = 4 * n - 3 :=
sorry

-- Sum inequality result 
theorem sum_inequality : ∀ n : ℕ+, 
  ((∑ i in (finset.range n), 1 / S (i + 1)) < (5/3)) :=
sorry

end general_formula_sum_inequality_l31_31313


namespace matrix_power_minus_l31_31857

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end matrix_power_minus_l31_31857


namespace circle_radius_l31_31275

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 4*x - 2*y = 1) → ∃ r : ℝ, r = sqrt 6 :=
by
  intro h
  exists sqrt 6
  sorry

end circle_radius_l31_31275


namespace water_depth_proof_l31_31205

def water_depth_after_cube (a : ℝ) (h₁ : 0 < a) (h₂ : a ≤ 50) : ℝ :=
  if 49 ≤ a then 50
  else if 9 ≤ a then a + 1
  else (10 / 9) * a

theorem water_depth_proof (a : ℝ) (h₁ : 0 < a) (h₂ : a ≤ 50) :
  water_depth_after_cube a h₁ h₂ =
    if 49 ≤ a then 50
    else if 9 ≤ a then a + 1
    else (10 / 9) * a :=
sorry

end water_depth_proof_l31_31205


namespace four_digit_composite_l31_31627

theorem four_digit_composite (abcd : ℕ) (h : 1000 ≤ abcd ∧ abcd < 10000) :
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≥ 2 ∧ m * n = (abcd * 10001) :=
by
  sorry

end four_digit_composite_l31_31627


namespace fish_farming_problem_l31_31549

-- Definitions and given conditions
def fish_masses : List ℝ := [1.0, 1.2, 1.5, 1.8]
def fish_frequencies : List ℕ := [4, 5, 8, 3]

-- Helper function to calculate median of a sample
def median (masses : List ℝ) (freqs : List ℕ) : ℝ :=
  let samples := List.join (List.map (fun (mass, freq) => List.replicate freq mass) (List.zip masses freqs))
  let sorted_samples := List.sort (· ≤ ·) samples
  let n := List.length sorted_samples
  if n % 2 = 0 then
    (sorted_samples.get! (n / 2 - 1) + sorted_samples.get! (n / 2)) / 2
  else
    sorted_samples.get! (n / 2)

-- Helper function to calculate sample mean
def sample_mean (masses : List ℝ) (freqs : List ℕ) : ℝ :=
  let num := List.sum (List.map (fun (mass, freq) => mass * freq) (List.zip masses freqs))
  let denom := List.sum freqs
  num / denom

-- Capture-recapture estimation
def total_fish_in_pond (marked_initially n_marked n_caught : ℕ) : ℕ :=
  (n_caught * marked_initially) / n_marked

-- Definition for total mass calculation
def total_mass_of_fish (mean_mass : ℝ) (total_fish : ℕ) : ℝ := mean_mass * total_fish

theorem fish_farming_problem :
  median fish_masses fish_frequencies = 1.5 ∧
  sample_mean fish_masses fish_frequencies = 1.37 ∧
  total_mass_of_fish 1.37 (total_fish_in_pond 20 2 100) = 1370 :=
by
  sorry

end fish_farming_problem_l31_31549


namespace find_f_2023_l31_31328

def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (3 + x)

theorem find_f_2023 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)) 
  (h2 : ∀ x : ℝ, f (1 - x) = f (3 + x)) : 
  f 2023 = 2 :=
sorry

end find_f_2023_l31_31328


namespace football_club_balance_l31_31625

/-- A football club has a balance of $100 million. The club then sells 2 of its players at $10 million each, and buys 4 more at $15 million each. Prove that the final balance is $60 million. -/
theorem football_club_balance :
  let initial_balance := 100
  let income_from_sales := 10 * 2
  let expenditure_on_purchases := 15 * 4
  let final_balance := initial_balance + income_from_sales - expenditure_on_purchases
  final_balance = 60 :=
by
  simp only [initial_balance, income_from_sales, expenditure_on_purchases, final_balance]
  sorry

end football_club_balance_l31_31625


namespace angle_reduction_l31_31516

theorem angle_reduction (θ : ℝ) : θ = 1303 → ∃ k : ℤ, θ = 360 * k - 137 := 
by  
  intro h 
  use 4 
  simp [h] 
  sorry

end angle_reduction_l31_31516


namespace g_nested_result_l31_31450

def g (n : ℕ) : ℕ :=
if n < 5 then
  n^2 + 1
else
  2 * n + 3

theorem g_nested_result : g (g (g 3)) = 49 := by
sorry

end g_nested_result_l31_31450


namespace dividend_and_shares_l31_31602

noncomputable def companyData : Type :=
struct Expected_Earnings : ℝ Expected_Dividends : ℝ Extra_Earnings : ℝ Corporate_Tax : ℝ Market_Price : ℝ Discount : ℝ Actual_Earnings : ℝ Shares_Owned : ℕ

def c : companyData := 
{ Expected_Earnings := 0.80, Expected_Dividends := 0.40, Extra_Earnings := 0.04, Corporate_Tax := 0.15, Market_Price := 20, Discount := 0.05, Actual_Earnings := 1.10, Shares_Owned := 500 }

def total_dividend (c: companyData) : ℝ := 
  let additional_dividend := (c.Actual_Earnings - c.Expected_Earnings) / 0.10 * c.Extra_Earnings
  let total_dividend_before_tax := c.Expected_Dividends + additional_dividend
  let tax := c.Corporate_Tax * c.Actual_Earnings
  let total_dividend_after_tax := total_dividend_before_tax - tax
  total_dividend_after_tax

def total_shares (c: companyData) : ℕ := 
  let total_dividend := total_dividend c * c.Shares_Owned
  let discounted_price := c.Market_Price * (1 - c.Discount)
  let num_shares := total_dividend / discounted_price
  num_shares.to_nat

theorem dividend_and_shares (c : companyData) : 
  total_dividend c = 0.355 ∧ total_shares c = 9 := 
  sorry

end dividend_and_shares_l31_31602


namespace physical_fitness_test_l31_31810

theorem physical_fitness_test (x : ℝ) (hx : x > 0) :
  (1000 / x - 1000 / (1.25 * x) = 30) :=
sorry

end physical_fitness_test_l31_31810


namespace train_length_l31_31647

theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (speed_mps : ℝ) (length_train : ℝ) : 
  speed_kmph = 90 → 
  time_seconds = 6 → 
  speed_mps = (speed_kmph * 1000 / 3600) →
  length_train = (speed_mps * time_seconds) → 
  length_train = 150 :=
by
  intros h_speed h_time h_speed_mps h_length
  sorry

end train_length_l31_31647


namespace range_of_m_l31_31351

def f (x : ℝ) : ℝ := x^3 - 3*x + 1
def g (x m : ℝ) : ℝ := (1 / 2) ^ x - m

theorem range_of_m (m : ℝ) : (∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc 0 2, f x1 ≥ g x2 m) → m ≥ 5 / 4 :=
by
  sorry

end range_of_m_l31_31351


namespace find_angle_l31_31368

variables {V : Type*} [inner_product_space ℝ V]

def angle_between (a b : V) : ℝ :=
  real.arccos ((inner_product_space.inner a b) / (∥a∥ * ∥b∥))

theorem find_angle (a b : V) 
  (h1 : ∥a + b∥ = ∥a∥)
  (h2 : ∥a + b∥ = ∥b∥)
  (h3 : ∥a∥ = ∥b∥) :
  angle_between b (a - b) = 5 * real.pi / 6 := 
begin
  sorry
end

end find_angle_l31_31368


namespace boy_usual_time_l31_31203

theorem boy_usual_time (R T : ℝ) (h : R * T = (7 / 6) * R * (T - 2)) : T = 14 :=
by
  sorry

end boy_usual_time_l31_31203


namespace trajectory_eq_min_RP_l31_31310

-- Define the conditions
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4
def on_OM (OM OP : ℝ) : Prop := OM * OP = 12

-- Define the polar equation of the trajectory
theorem trajectory_eq (ρ θ : ℝ) (h : line_l ρ θ) (OM OP : ℝ) (h_OM_OP: on_OM OM OP) : 
  ρ = 3 * Real.cos θ :=
sorry

-- Minimum value of |RP| theorem
theorem min_RP (θ : ℝ) (R P : ℝ × ℝ) (h_R_line_l : R.1 = 4) : 
  let x := P.1
  let y := P.2
  let RP := Real.sqrt ((x - R.1)^2 + (y - R.2)^2)
  in RP ≥ 1 :=
sorry

end trajectory_eq_min_RP_l31_31310


namespace range_a_l31_31769

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + x else -3 * x

theorem range_a (a : ℝ) : 
  a * (f a - f (-a)) > 0 ↔ a ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 2 ∞ :=
sorry

end range_a_l31_31769


namespace socks_expected_value_l31_31143

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l31_31143


namespace problem1_problem2_l31_31317

section complex_numbers

variable (a : ℝ)
def z1 : ℂ := ⟨a, -2⟩
def z2 : ℂ := ⟨3, 4⟩
def product := z1 * z2

-- Problem 1: Prove that if z1 * z2 is pure imaginary, then a = -8/3.
theorem problem1 : (product.re = 0) → a = -8 / 3 := by
  sorry

-- Problem 2: Prove that if z1 * z2 is in the fourth quadrant, then -8/3 < a < 3/2.
theorem problem2 : (product.re > 0) ∧ (product.im < 0) → (-8 / 3 < a) ∧ (a < 3 / 2) := by
  sorry

end complex_numbers

end problem1_problem2_l31_31317


namespace reciprocal_neg_2023_l31_31130

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l31_31130


namespace chris_is_14_l31_31112

-- Definitions from the given conditions
variables (a b c : ℕ)
variables (h1 : (a + b + c) / 3 = 10)
variables (h2 : c - 4 = a)
variables (h3 : b + 5 = (3 * (a + 5)) / 4)

theorem chris_is_14 (h1 : (a + b + c) / 3 = 10) (h2 : c - 4 = a) (h3 : b + 5 = (3 * (a + 5)) / 4) : c = 14 := 
sorry

end chris_is_14_l31_31112


namespace Sonja_oil_used_l31_31506

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l31_31506


namespace cos_angle_difference_l31_31752

variable (A B : ℝ)

theorem cos_angle_difference :
  (sin A + sin B = 3 / 2) →
  (cos A + cos B = 1) →
  cos (A - B) = 5 / 8 :=
by
  intros hsin hcos
  sorry

end cos_angle_difference_l31_31752


namespace monotone_f_range_of_m_l31_31344

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := 2 * x / (x + 1)

-- Monotonicity of the function f(x) for x > 0
theorem monotone_f : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 < f x2 :=
by
  intros x1 x2 hx1 h
  show f x1 < f x2
  have h_eq : f x1 - f x2 = (2 / (x2 + 1) - 2 / (x1 + 1)) := sorry
  have h_lt : (2 / (x2 + 1) - 2 / (x1 + 1)) < 0 := sorry
  sorry

-- Range of m for the inequality f(2m-1) > f(1-m)
theorem range_of_m (m : ℝ) : f (2 * m - 1) > f (1 - m) ↔ m ∈ set.Ioo (2 / 3) 1 :=
by
  intro hm
  show m ∈ set.Ioo (2 / 3) 1
  have h_ineq : 2 * m - 1 > 1 - m ↔ m > 2 / 3 := sorry
  have h_restrict : (2 / 3) < m ∧ m < 1 := sorry
  sorry

end monotone_f_range_of_m_l31_31344


namespace find_other_number_l31_31583

theorem find_other_number
  (B : ℕ)
  (hcf_condition : Nat.gcd 24 B = 12)
  (lcm_condition : Nat.lcm 24 B = 396) :
  B = 198 :=
by
  sorry

end find_other_number_l31_31583


namespace total_sheets_of_paper_l31_31458

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l31_31458


namespace solution_set_of_inequality_l31_31767

noncomputable def f (x : ℝ) : ℝ := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem solution_set_of_inequality :
  (∀ x : ℝ, f (3 * x + 1) + f x > 0 ↔ x > -1/4) :=
by
  -- conditions
  have domain_f : ∀ x : ℝ, true := λ x, by trivial,
  have odd_f : ∀ x : ℝ, f (-x) = -f x := sorry,
  have increasing_f : ∀ x y : ℝ, x < y → f x < f y := sorry,
  
  sorry

end solution_set_of_inequality_l31_31767


namespace dual_polyhedron_is_regular_l31_31671

-- Declarations of the necessary conditions

-- A convex polyhedron
constant ConvexPolyhedron : Type
-- A regular polyhedron
constant RegularPolyhedron : Type
-- A function to find centers of faces of a regular polyhedron
constant centers_of_faces : RegularPolyhedron → ConvexPolyhedron

-- The given condition: There exists a convex polyhedron whose vertices are the centers of the faces
def dual_polyhedron (P : RegularPolyhedron) : ConvexPolyhedron :=
  centers_of_faces P

-- The theorem to prove: The dual polyhedron is regular
noncomputable def is_regular_polyhedron (P : ConvexPolyhedron) : Prop := sorry
constant is_regular_original : RegularPolyhedron → Prop

theorem dual_polyhedron_is_regular (P : RegularPolyhedron)
  (hP : is_regular_original P) : is_regular_polyhedron (dual_polyhedron P) :=
sorry

end dual_polyhedron_is_regular_l31_31671


namespace sum_of_roots_eq_six_l31_31184

theorem sum_of_roots_eq_six (a b c : ℤ) (h : a = 1 ∧ b = -6 ∧ c = 8) :
  let sum_of_roots := -b / a in
  sum_of_roots = 6 := by
  sorry

end sum_of_roots_eq_six_l31_31184


namespace initial_water_amount_l31_31842

variable (W : ℝ) 

theorem initial_water_amount (h1 : W - 4 ≥ 0) 
    (h2 : (let total_mixture := 2 + 4 * (W - 4) in 
           2 / total_mixture = 0.04)) : 
    W = 16 := 
by
  sorry

end initial_water_amount_l31_31842


namespace union_M_N_eq_l31_31778

def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N : Set ℝ := {0, 4}

theorem union_M_N_eq : M ∪ N = Set.Icc 0 4 := 
  by
    sorry

end union_M_N_eq_l31_31778


namespace coffee_shop_brewed_cups_in_week_l31_31607

theorem coffee_shop_brewed_cups_in_week 
    (weekday_rate : ℕ) (weekend_rate : ℕ)
    (weekday_hours : ℕ) (saturday_hours : ℕ) (sunday_hours : ℕ)
    (num_weekdays : ℕ) (num_saturdays : ℕ) (num_sundays : ℕ)
    (h1 : weekday_rate = 10)
    (h2 : weekend_rate = 15)
    (h3 : weekday_hours = 5)
    (h4 : saturday_hours = 6)
    (h5 : sunday_hours = 4)
    (h6 : num_weekdays = 5)
    (h7 : num_saturdays = 1)
    (h8 : num_sundays = 1) :
    (weekday_rate * weekday_hours * num_weekdays) + 
    (weekend_rate * saturday_hours * num_saturdays) + 
    (weekend_rate * sunday_hours * num_sundays) = 400 := 
by
  sorry

end coffee_shop_brewed_cups_in_week_l31_31607


namespace num_paths_in_grid_l31_31376

-- Define the parameters and the condition
def grid_size_7_8 (cols rows : ℕ) := cols = 7 ∧ rows = 8

-- Define the movement constraints
def valid_move_constraints (cols rows : ℕ) := 
  grid_size_7_8 cols rows ∧ (∀ x y, (x <= cols) ∧ (y <= rows))

-- Proving the number of unique paths from A to B is 6435
theorem num_paths_in_grid (cols rows : ℕ) (h : grid_size_7_8 cols rows) :
  ∃ paths : ℕ, paths = Nat.choose (cols + rows) rows ∧ paths = 6435 :=
by
  use Nat.choose (cols + rows) rows
  have h1 : cols + rows = 15, from sorry
  have h2 : rows = 8, from sorry
  have h3 : Nat.choose 15 8 = 6435, from sorry
  exact ⟨Nat.choose 15 8, h3⟩

end num_paths_in_grid_l31_31376


namespace tim_initial_balls_correct_l31_31904

-- Defining the initial number of balls Robert had
def robert_initial_balls : ℕ := 25

-- Defining the final number of balls Robert had
def robert_final_balls : ℕ := 45

-- Defining the number of balls Tim had initially
def tim_initial_balls := 40

-- Now, we state the proof problem:
theorem tim_initial_balls_correct :
  robert_initial_balls + (tim_initial_balls / 2) = robert_final_balls :=
by
  -- This is the part where you typically write the proof.
  -- However, we put sorry here because the task does not require the proof itself.
  sorry

end tim_initial_balls_correct_l31_31904


namespace prob_four_heads_in_ten_flips_l31_31550

theorem prob_four_heads_in_ten_flips :
  let p := (3 : ℚ) / 7
  let q := (4 : ℚ) / 7
  let n := 10
  let k := 4
  ∑ (x : ℕ) in finset.Icc 4 4, nat.choose 10 4 * p^4 * q^6 = (69,874,560 : ℚ) / 282,576,201 :=
by
  let p := (3 : ℚ) / 7
  let q := (4 : ℚ) / 7
  sorry

end prob_four_heads_in_ten_flips_l31_31550


namespace probability_3_one_color_1_another_l31_31601

theorem probability_3_one_color_1_another :
  let black_balls := 10
  let white_balls := 9
  let total_balls := black_balls + white_balls
  let total_ways := Nat.choose total_balls 4
  let ways_3_black_1_white := Nat.choose black_balls 3 * Nat.choose white_balls 1
  let ways_1_black_3_white := Nat.choose black_balls 1 * Nat.choose white_balls 3
  let favorable_ways := ways_3_black_1_white + ways_1_black_3_white
  let probability := favorable_ways / total_ways
in probability = (160 : ℚ) / 323 := sorry

end probability_3_one_color_1_another_l31_31601


namespace dice_composite_probability_l31_31435

theorem dice_composite_probability (m n : ℕ) (h : Nat.gcd m n = 1) :
  (∃ m n : ℕ, (m * 36 = 29 * n) ∧ Nat.gcd m n = 1) → m + n = 65 :=
by {
  sorry
}

end dice_composite_probability_l31_31435


namespace Q1_Q2_l31_31901

noncomputable def polynomial_exists (n : ℕ) : Prop :=
  ∃ P : ℕ → ℝ → ℝ, (∀ t : ℝ, 2 * real.cos (n * t) = P n (2 * real.cos t)) ∧
    (∀ m, (m = 0 ∨ m = 1) → ∃! f : ℝ → ℝ, f = (λ x, x ^ m - 2)) ∧
    (∀ k, k ≥ 2 → P k = λ x, - (P (k - 2) x) + x * (P (k - 1) x))

theorem Q1 (n : ℕ) : polynomial_exists n :=
sorry

noncomputable def rational_cos (α : ℚ) : Prop :=
  let r := real.cos (α * real.pi) in
  r = 0 ∨ r = 1 / 2 ∨ r = -1 / 2 ∨ r = 1 ∨ r = -1 ∨ irrational r

theorem Q2 (α : ℚ) : rational_cos α :=
sorry

end Q1_Q2_l31_31901


namespace oil_used_l31_31504

theorem oil_used (total_weight : ℕ) (ratio_oil_peanuts : ℕ) (ratio_total_parts : ℕ) 
  (ratio_peanuts : ℕ) (ratio_parts : ℕ) (peanuts_weight : ℕ) : 
  ratio_oil_peanuts = 2 → 
  ratio_peanuts = 8 → 
  ratio_total_parts = 10 → 
  ratio_parts = 20 →
  peanuts_weight = total_weight / ratio_total_parts →
  total_weight = 20 → 
  2 * peanuts_weight = 4 :=
by sorry

end oil_used_l31_31504


namespace find_interest_rate_l31_31294

-- Defining the conditions
def P : ℝ := 5000
def A : ℝ := 5302.98
def t : ℝ := 1.5
def n : ℕ := 2

-- Statement of the problem in Lean 4
theorem find_interest_rate (P A t : ℝ) (n : ℕ) (hP : P = 5000) (hA : A = 5302.98) (ht : t = 1.5) (hn : n = 2) : 
  ∃ r : ℝ, r * 100 = 3.96 :=
sorry

end find_interest_rate_l31_31294


namespace find_a_n_l31_31372

theorem find_a_n (a : ℕ → ℕ) :
  (∀ t : ℝ, (∑ n : ℕ, (a n) * (t ^ n) / (Nat.factorial n : ℝ))
              = (∑ n : ℕ, (2 ^ n) * (t ^ n) / (Nat.factorial n : ℝ))^2
                * (∑ n : ℕ, (3 ^ n) * (t ^ n) / (Nat.factorial n : ℝ))^2) →
  ∀ n : ℕ, a n = 10 ^ n :=
begin
  sorry
end

end find_a_n_l31_31372


namespace minimum_a_l31_31873

noncomputable def f (x a : ℝ) := Real.exp x * (x^3 - 3 * x + 3) - a * Real.exp x - x

theorem minimum_a (a : ℝ) : (∃ x, x ≥ -2 ∧ f x a ≤ 0) ↔ a ≥ 1 - 1 / Real.exp 1 :=
by
  sorry

end minimum_a_l31_31873


namespace difference_of_squares_550_450_l31_31190

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end difference_of_squares_550_450_l31_31190


namespace triplet_difference_after_2016_seconds_l31_31984

-- Define initial triplet
def initial_triplet := (20, 1, 6)

-- Define the transformation rule
def transform_triplet (t : (ℕ × ℕ × ℕ)) : (ℕ × ℕ × ℕ) :=
  (t.2.1 + t.2.2, t.1 + t.2.2, t.1 + t.2.1)

-- Prove that after 2016 seconds, the difference remains 19
theorem triplet_difference_after_2016_seconds :
  let final_triplet := (iterate transform_triplet 2016 initial_triplet)
  in abs ((max final_triplet.1 (max final_triplet.2.1 final_triplet.2.2)) - (min final_triplet.1 (min final_triplet.2.1 final_triplet.2.2))) = 19 :=
by {
  -- Proof goes here
  sorry
}

end triplet_difference_after_2016_seconds_l31_31984


namespace problem_solution_l31_31019

variable {a b c d : ℝ}
variable (h_a : a = 4 * π / 3)
variable (h_b : b = 10 * π)
variable (h_c : c = 62)
variable (h_d : d = 30)

theorem problem_solution : (b * c) / (a * d) = 15.5 :=
by
  rw [h_a, h_b, h_c, h_d]
  -- Continued steps according to identified solution steps
  -- and arithmetic operations.
  sorry

end problem_solution_l31_31019


namespace card_d_total_percent_change_l31_31717

noncomputable def card_d_initial_value : ℝ := 250
noncomputable def card_d_percent_changes : List ℝ := [0.05, -0.15, 0.30, -0.10, 0.20]

noncomputable def final_value (initial_value : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_value

theorem card_d_total_percent_change :
  let final_val := final_value card_d_initial_value card_d_percent_changes
  let total_percent_change := ((final_val - card_d_initial_value) / card_d_initial_value) * 100
  total_percent_change = 25.307 := by
  sorry

end card_d_total_percent_change_l31_31717


namespace shortest_distance_to_circle_from_origin_l31_31569

theorem shortest_distance_to_circle_from_origin :
  let C := {p : ℝ × ℝ | (p.1 - 9)^2 + (p.2 - 4)^2 = 56}
  in infi (λ (p : ℝ × ℝ), (p.1^2 + p.2^2)^(1/2)) - 2 * real.sqrt 14 = real.sqrt 97 - 2 * real.sqrt 14 :=
by
  let C := {p : ℝ × ℝ | (p.1 - 9)^2 + (p.2 - 4)^2 = 56}
  show infi (λ (p : ℝ × ℝ), (p.1^2 + p.2^2)^(1/2)) - 2 * real.sqrt 14 = real.sqrt 97 - 2 * real.sqrt 14
  sorry

end shortest_distance_to_circle_from_origin_l31_31569


namespace find_prime_p_l31_31455

noncomputable def concatenate (q r : ℕ) : ℕ :=
q * 10 ^ (r.digits 10).length + r

theorem find_prime_p (q r p : ℕ) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp : Nat.Prime p)
  (h : concatenate q r + 3 = p^2) : p = 5 :=
sorry

end find_prime_p_l31_31455


namespace sum_possible_values_for_k_l31_31412

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end sum_possible_values_for_k_l31_31412


namespace middle_number_between_52_and_certain_number_l31_31536

theorem middle_number_between_52_and_certain_number :
  ∃ n, n > 52 ∧ (∀ k, 52 ≤ k ∧ k ≤ n → ∃ l, k = 52 + l) ∧ (n = 52 + 16) :=
sorry

end middle_number_between_52_and_certain_number_l31_31536


namespace total_children_l31_31156

-- Define the problem conditions
variable (ages : List ℕ)
variable (S : ℕ)
variable (oldest_age : ℕ)
variable (excluded : ℕ)
variable (arithmetic_seq : List ℕ)

-- Define the constraints based on the conditions
def conditions : Prop :=
  S = 50 ∧           -- Total sum of ages is 50
  oldest_age = 13 ∧  -- The oldest child is 13 years old
  10 ∈ ages ∧        -- One of them is 10 years old
  S = (10 + (List.sum arithmetic_seq)) ∧          -- Excluding one 10-year-old, sum of remaining is 40
  arithmetic_seq.head? = some excluded ∧           -- The smallest age in sequence
  (List.length ages = List.length arithmetic_seq + 1) ∧ -- Total number of children is one more than arithmetic sequence length
  is_arith_seq (List.map (. - excluded) arithmetic_seq) -- Remaining ages form an arithmetic sequence

-- Define the relation stating the number of children
def number_of_children (n : ℕ) : Prop :=
  List.length ages = n

-- Theorem to prove
theorem total_children : ∃ n : ℕ, conditions ages S oldest_age excluded arithmetic_seq ∧ number_of_children ages n := 
sorry

end total_children_l31_31156


namespace solve_for_radius_l31_31453

theorem solve_for_radius
  (V A S : ℝ)
  (r k : ℝ)
  (h1 : V = (4 / 3) * π * r^3 = S)
  (h2 : A = 4 * π * r^2 = S)
  (h3 : k * r = S):
  r = 3 := 
by
  sorry

end solve_for_radius_l31_31453


namespace least_number_of_cans_l31_31616

theorem least_number_of_cans (Maaza Pepsi Sprite : ℕ) (h1 : Maaza = 157) (h2 : Pepsi = 173) (h3 : Sprite = 389) (gcd_Maaza_Pepsi : Nat.gcd 157 173 = 1) (gcd_Maaza_Pepsi_Sprite : Nat.gcd (Nat.gcd 157 173) 389 = 1) : Maaza + Pepsi + Sprite = 719 :=
by
  rw [h1, h2, h3]
  exact rfl

end least_number_of_cans_l31_31616


namespace number_of_digits_of_a_times_10n_l31_31791

noncomputable def number_of_integer_digits (x : ℝ) : ℕ :=
  (Real.floor (Real.log10 x) + 1).toNat

theorem number_of_digits_of_a_times_10n
  (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : a < 10) (h3 : 0 < n) :
  number_of_integer_digits (a * (10:ℝ)^n) = n + 1 :=
by
  sorry

end number_of_digits_of_a_times_10n_l31_31791


namespace conjugate_of_z_l31_31442

def z : ℂ := 2 / (-1 + complex.I)

theorem conjugate_of_z : conj z = -1 + complex.I := 
sorry

end conjugate_of_z_l31_31442


namespace find_vector_d_l31_31121

def line_eq (x : ℝ) : ℝ := (4 * x - 7) / 3

def parameterized_form (t : ℝ) (d : ℝ × ℝ) : ℝ × ℝ :=
 (4, 2) + t • d

def distance_condition (t : ℝ) (d : ℝ × ℝ) : Prop :=
  dist (parameterized_form t d) (4, 2) = t

theorem find_vector_d (d : ℝ × ℝ) :
  line_eq 4 = 2 ∧ distance_condition t d → d = (3 / 5, 4 / 5) :=
by
  sorry

end find_vector_d_l31_31121


namespace smallest_odd_divisors_l31_31571

-- Define integer 360 and its properties
def n := 360
def prime_factors_360 : Prop := (∀ p: ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5)
def divisors_count_360 : Prop := 
  ∀ d: ℕ, d ∣ n → (∃ k: ℕ, d = 2^k ∨ d = 3^k ∨ d = 5^k ∨ d = (2^a) * (3^b) ∨ d = (2^a) * (5^c) ∨ d = (3^b) * (5^c) ∨ d = (2^a) * (3^b) * (5^c))

-- Hannissuetstatt smallest odd number having the same divisor count
def smallest_odd_with_24_divisors : ℕ := 31185

theorem smallest_odd_divisors (h : ∀ d: ℕ,  d ∣ n → (∃ k: ℕ, d = 2^k ∨ d = 3^k ∨ d = 5^k ∨ d = (2^a) * (3^b) ∨ d = (2^a) * (5^c) ∨ d = (3^b) * (5^c) ∨ d = (2^a) * (3^b) * (5^c))) :
smallest_odd_with_24_divisors = 31185 := 
sorry

end smallest_odd_divisors_l31_31571


namespace number_of_chords_number_of_chords_l31_31058

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l31_31058


namespace inverse_proportion_first_third_quadrant_l31_31359

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l31_31359


namespace max_area_of_triangle_AMN_l31_31980

noncomputable def parabola (y : ℝ) : ℝ := (y^2) / 4

def line (x b : ℝ) : ℝ := x + b

def triangle_area (b : ℝ) : ℝ := 
  2 * abs(5 + b) * real.sqrt(1 - b)

theorem max_area_of_triangle_AMN : 
  ∃ b : ℝ, (∀ x b : ℝ, y = line x b) ∧ 
           (y^2 = 4 * x) ∧ 
           b = -1 → 
           triangle_area b = 8 * real.sqrt 2 := 
begin
  sorry
end

end max_area_of_triangle_AMN_l31_31980


namespace cost_of_salt_per_pound_is_120_20_l31_31552

noncomputable def total_cost_of_salt 
  (flour_needed : ℕ)
  (flour_bag_weight : ℕ)
  (flour_bag_cost : ℕ) 
  (salt_needed : ℕ) 
  (ticket_price : ℕ) 
  (tickets_sold : ℕ) 
  (promotion_cost : ℕ) 
  (total_revenue_made : ℕ)
  (final_amount : ℕ) : ℝ :=
  let flour_cost := (flour_needed / flour_bag_weight) * flour_bag_cost
  let revenue := tickets_sold * ticket_price
  let total_expenses := promotion_cost + flour_cost + final_amount
  let cost_of_salt := (revenue - total_expenses) / salt_needed
  cost_of_salt

theorem cost_of_salt_per_pound_is_120_20 :
  total_cost_of_salt 500 50 20 10 20 500 1000 8798 = 120.20 :=
by
  sorry

end cost_of_salt_per_pound_is_120_20_l31_31552


namespace positive_difference_prob_l31_31167

/-- Probability that the positive difference between two randomly chosen numbers from 
the set {1, 2, 3, 4, 5, 6, 7, 8} is 3 or greater -/
theorem positive_difference_prob :
  (let S := {1, 2, 3, 4, 5, 6, 7, 8}
       in (S.powerset.filter (λ s => s.card = 2)).card.filter (λ s => (s.to_list.head! - s.to_list.tail.head!).nat_abs >= 3).card /
           (S.powerset.filter (λ s => s.card = 2)).card = 15 / 28) := 
begin
  sorry
end

end positive_difference_prob_l31_31167


namespace reciprocal_of_negative_2023_l31_31126

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l31_31126


namespace opposite_of_neg_frac_l31_31944

theorem opposite_of_neg_frac :
  ∀ x : ℝ, x = - (1 / 2023) → -x = 1 / 2023 :=
by
  intros x hx
  rw hx
  norm_num

end opposite_of_neg_frac_l31_31944


namespace sum_possible_values_for_k_l31_31413

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end sum_possible_values_for_k_l31_31413


namespace melanie_dimes_l31_31036

theorem melanie_dimes (original_dimes dad_dimes mom_dimes total_dimes : ℕ) :
  original_dimes = 7 →
  mom_dimes = 4 →
  total_dimes = 19 →
  (total_dimes = original_dimes + dad_dimes + mom_dimes) →
  dad_dimes = 8 :=
by
  intros h1 h2 h3 h4
  sorry -- The proof is omitted as instructed.

end melanie_dimes_l31_31036


namespace matrix_calculation_l31_31860

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end matrix_calculation_l31_31860


namespace general_seq_a_sum_seq_a_b_l31_31738

variable {a_n : ℕ → ℤ}  -- Define the arithmetic sequence
variable {b_n : ℕ → ℤ}  -- Define the geometric sequence
variable {S_n : ℕ → ℤ}  -- Define the sum of the first n terms of a_n + b_n 

-- Conditions of the problem
axiom condition1 : a_n 5 = 9
axiom condition2 : a_n 7 = 13
axiom condition3 : b_n = λ n, 2^(n-1)

-- Correct answers
theorem general_seq_a :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
sorry

theorem sum_seq_a_b (n : ℕ) :
  (S_n n = ∑ i in range (n+1), (a_n i + b_n i) = n^2 + 2^n - 1) :=
sorry

end general_seq_a_sum_seq_a_b_l31_31738


namespace tangent_line_interval_l31_31349

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem tangent_line_interval (x0 : ℝ) (x0_gt_0: 0 < x0) (x0_lt_1: x0 < 1) :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ tangent f x0 (g m)) → x0 ∈ Set.Ioo (Real.sqrt 2) (Real.sqrt 3) := sorry

end tangent_line_interval_l31_31349


namespace angle_AEC_length_AE_l31_31411

variable (Point Circle Line : Type) [DivisionRing Circle]

noncomputable def midpoint (D O C : Point) : Point := sorry
noncomputable def extend (A D : Point) : Line := sorry
noncomputable def intersection (line : Line) (circle : Circle) : Point := sorry
noncomputable def tangent (P A : Point) (circle : Circle) : Prop := sorry
noncomputable def is_midpoint (D O C : Point) : Prop := sorry
noncomputable def angle_at_point (A P B : Point) : ℝ := sorry
noncomputable def dist (A B : Point) : ℝ := sorry

-- Variables given in the problem
variables (P A O B C D E : Point)
variables (circle O : Circle)
variables (PO : Line)
variable (PA_value : ℝ := 2 * Real.sqrt 3)
variable (angle_APB_value : ℝ := 30 * Real.pi / 180)

-- Defining the conditions
axiom tangent_PA : tangent P A circle
axiom intersect_PO : ∃ (B C : Point), PO intersects circle at B and C
axiom midpoint_D : is_midpoint D O C
axiom extend_AD_E : extend A D intersects circle at E
axiom PA_eq : dist P A = PA_value
axiom angle_APB_eq : angle_at_point A P B = angle_APB_value

-- Proof of the goals
theorem angle_AEC : ∠ A E C = (60 : ℝ) * Real.pi / 180 := sorry
theorem length_AE : dist A E = 2 + 2 * Real.sqrt 5 := sorry


end angle_AEC_length_AE_l31_31411


namespace sin_ratio_value_l31_31397

variable (a b A B : ℝ)
variable (sin : ℝ → ℝ)
variable (sin_sq := λ x, sin x * sin x)
variable (triangle_sides : 3 * a = 2 * b)

theorem sin_ratio_value (h : triangle_sides) : (2 * sin_sq B - sin_sq A) / sin_sq A = 7 / 2 := by
  sorry

end sin_ratio_value_l31_31397


namespace function_odd_domain_of_f_range_of_f_l31_31198

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem function_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intro x
  trivial

theorem range_of_f : ∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  intro y
  sorry

end function_odd_domain_of_f_range_of_f_l31_31198


namespace additional_charge_per_segment_l31_31427

-- Definitions for the conditions
def initial_fee : ℝ := 2.25
def total_charge : ℝ := 5.85
def distance_traveled : ℝ := 3.6
def segment_length : ℝ := 2 / 5

-- The statement to prove
theorem additional_charge_per_segment :
  (total_charge - initial_fee) = 9 * 0.40 :=
begin
  -- Definitions for clarity and calculations
  let number_of_segments := distance_traveled / segment_length,
  have h1 : number_of_segments = 9,
  { sorry },

  let total_distance_charge := total_charge - initial_fee,
  have h2 : total_distance_charge = 3.60,
  { sorry },

  have h3 : total_distance_charge = number_of_segments * 0.40,
  { sorry },

  show 3.60 = 9 * 0.40, from h3
end

end additional_charge_per_segment_l31_31427


namespace peanuts_in_box_l31_31585

theorem peanuts_in_box (original_peanuts added_peanuts total_peanuts : ℕ) (h1 : original_peanuts = 10) (h2 : added_peanuts = 8) (h3 : total_peanuts = original_peanuts + added_peanuts) : total_peanuts = 18 := 
by {
  sorry
}

end peanuts_in_box_l31_31585


namespace maximize_profit_l31_31604

noncomputable def annual_profit (x a : ℝ) : ℝ :=
  (x - 3 - a) * (11 - x)^2

theorem maximize_profit :
  ∀ a : ℝ, 1 ≤ a ∧ a ≤ 3 → 
    (if 1 ≤ a ∧ a ≤ 2 then 
       annual_profit 7 a = 16 * (4 - a)
     else if 2 < a ∧ a ≤ 3 then 
       annual_profit ((17 + 2 * a) / 3) a = (8 - a)^3 
     else 
       False) :=
by
  intro a h
  cases h.left
  · sorry
  · cases h.right
    · sorry
    · sorry

end maximize_profit_l31_31604


namespace ellipse_equation_minimum_distance_l31_31739

-- Define the conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2) / (a^2) + (y^2) / (b^2) = 1)

def eccentricity (a c : ℝ) : Prop :=
  c = a / 2

def focal_distance (c : ℝ) : Prop :=
  2 * c = 4

def foci_parallel (F1 A B C D : ℝ × ℝ) : Prop :=
  let ⟨x1, y1⟩ := F1;
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (yA - y1) / (xA - x1) = (yC - y1) / (xC - x1) ∧ 
  (yB - y1) / (xB - x1) = (yD - y1) / (xD - x1)

def orthogonal_vectors (A C B D : ℝ × ℝ) : Prop :=
  let ⟨xA, yA⟩ := A;
  let ⟨xC, yC⟩ := C;
  let ⟨xB, yB⟩ := B;
  let ⟨xD, yD⟩ := D;
  (xC - xA) * (xD - xB) + (yC - yA) * (yD - yB) = 0

-- Prove equation of ellipse E
theorem ellipse_equation (a b : ℝ) (x y : ℝ) (c : ℝ)
  (h1 : ellipse a b x y)
  (h2 : eccentricity a c)
  (h3 : focal_distance c) :
  (a = 4) ∧ (b^2 = 12) ∧ (x^2 / 16 + y^2 / 12 = 1) :=
sorry

-- Prove minimum value of |AC| + |BD|
theorem minimum_distance (A B C D : ℝ × ℝ)
  (F1 : ℝ × ℝ)
  (h1 : foci_parallel F1 A B C D)
  (h2 : orthogonal_vectors A C B D) :
  |(AC : ℝ)| + |(BD : ℝ)| = 96 / 7 :=
sorry

end ellipse_equation_minimum_distance_l31_31739


namespace metal_wasted_is_three_fourths_l31_31634

noncomputable def metal_wasted (w : ℝ) : ℝ :=
  let area_rectangle := 2 * w^2 in
  let radius_circle := w / 2 in
  let area_circle := π * (radius_circle^2) in
  let side_square := w * (Real.sqrt 2) / 2 in
  let area_square := side_square^2 in
  area_rectangle - area_circle + area_circle - area_square

theorem metal_wasted_is_three_fourths (w : ℝ) :
  metal_wasted w = 3 / 4 * (2 * w^2) :=
by
  let area_rectangle := 2 * w^2
  let radius_circle := w / 2
  let area_circle := π * (radius_circle^2)
  let side_square := w * (Real.sqrt 2) / 2
  let area_square := side_square^2
  have h1 : metal_wasted w = area_rectangle - area_circle + area_circle - area_square := rfl
  calc
    metal_wasted w
        = area_rectangle - area_circle + area_circle - area_square : h1
    ... = area_rectangle - area_square : by rw [sub_add_eq_sub_sub, add_sub_cancel']
    ... = 2 * w^2 - w^2 / 2 : by rw [←sq, ←sq, Real.sqrt_sq (le_of_lt (half_pos zero_lt_one)), Real.sqrt_sq (le_of_lt zero_lt_two)]
    ... = (4 * w^2) / 2 - w^2 / 2 : by norm_num [mul_div_cancel_left w zero_lt_two]
    ... = (4 * w^2 - w^2) / 2 : by ring
    ... = 3 * w^2 / 2 : by ring
    ... = 3 / 4 * (2 * w^2) : by ring


end metal_wasted_is_three_fourths_l31_31634


namespace f_at_1_f_inequality_solution_l31_31308

-- Define the function f and its properties
def f (x : ℝ) : ℝ := sorry -- Assume the existence of such a function

-- Properties of the function
axiom f_pos_domain : ∀ x > 0, f x ≥ 0
axiom f_mul_add : ∀ x y > 0, f (x * y) = f x + f y
axiom f_at_4 : f 4 = 12
axiom f_pos_gt_1 : ∀ x > 1, f x > 0

-- Statement a: f(1) = 0
theorem f_at_1 : f 1 = 0 :=
sorry

-- Statement d: Solution set of f(x+3) - f(2/x) < 6 is (0,1)
theorem f_inequality_solution (x : ℝ) (hx : 0 < x ∧ x < 1) :
  f (x + 3) - f (2 / x) < 6 :=
sorry

end f_at_1_f_inequality_solution_l31_31308


namespace inequality_solution_l31_31705

theorem inequality_solution :
  {x : ℝ | (x^2 + 5 * x) / ((x - 3) ^ 2) ≥ 0} = {x | x < -5} ∪ {x | 0 ≤ x ∧ x < 3} ∪ {x | x > 3} :=
by
  sorry

end inequality_solution_l31_31705


namespace find_sin_angle_HAD_l31_31018

-- Definitions and conditions
variables {A B C D E F G H : ℝ × ℝ × ℝ}
variables (AB_length AD_length AE_length : ℝ)

-- Conditions
def rectangular_prism : Prop := 
  ∃ (AB AD AE : ℝ), AB = 1 ∧ AD = 2 ∧ AE = 3

-- Function to calculate sin of angle HAD in a rectangular prism
noncomputable def sin_angle_HAD 
  (A H D : ℝ × ℝ × ℝ) : ℝ :=
let HA := (A.1 - H.1, A.2 - H.2, A.3 - H.3),
    HD := (D.1 - H.1, D.2 - H.2, D.3 - H.3),
    dot_product := HA.1 * HD.1 + HA.2 * HD.2 + HA.3 * HD.3,
    HA_norm := real.sqrt (HA.1^2 + HA.2^2 + HA.3^2),
    HD_norm := real.sqrt (HD.1^2 + HD.2^2 + HD.3^2),
    cos_theta := dot_product / (HA_norm * HD_norm),
    sin_theta_squared := 1 - cos_theta^2 in
  real.sqrt sin_theta_squared

-- Main theorem to prove
theorem find_sin_angle_HAD 
  (h: rectangular_prism) : 
  sin_angle_HAD (0, 0, 0) (0, 2, 3) (2, 0, 0) = real.sqrt (52 / 221) :=
sorry

end find_sin_angle_HAD_l31_31018


namespace johns_money_left_l31_31430

noncomputable def johns_initial_amount : ℝ := 200
noncomputable def fraction_given_to_mother : ℝ := 3 / 8
noncomputable def fraction_given_to_father : ℝ := 3 / 10

noncomputable def amount_given_to_mother := fraction_given_to_mother * johns_initial_amount
noncomputable def amount_given_to_father := fraction_given_to_father * johns_initial_amount
noncomputable def total_given := amount_given_to_mother + amount_given_to_father
noncomputable def amount_left := johns_initial_amount - total_given

theorem johns_money_left : amount_left = 65 :=
by
    have h_mother : amount_given_to_mother = 75 := by norm_num [amount_given_to_mother, fraction_given_to_mother, johns_initial_amount]
    have h_father : amount_given_to_father = 60 := by norm_num [amount_given_to_father, fraction_given_to_father, johns_initial_amount]
    have h_total : total_given = 135 := by norm_num [total_given, h_mother, h_father]
    show amount_left = 65 from by norm_num [amount_left, h_total, johns_initial_amount]

end johns_money_left_l31_31430


namespace Dexter_and_Sam_on_same_team_l31_31807

theorem Dexter_and_Sam_on_same_team {n k t : ℕ} (students : Fin n)
  (Dexter Sam : Fin n) (split_teams : ∀ i : Fin t, Finset (Fin k)) :
  n = 12 ∧ k = 6 ∧ Dexter ∈ (Finset.univ : Finset (Fin n)) ∧ Sam ∈ (Finset.univ : Finset (Fin n)) → 
  (∑ i in Finset.range t, if Dexter ∈ split_teams i ∧ Sam ∈ split_teams i then 1 else 0) = 210 := 
by
  intros h
  -- prove based on the given conditions
  sorry

end Dexter_and_Sam_on_same_team_l31_31807


namespace joe_eggs_around_park_l31_31010

variable (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ)

def joe_eggs (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ) : Prop :=
  total_eggs = club_house_eggs + town_hall_garden_eggs + park_eggs

theorem joe_eggs_around_park (h1 : total_eggs = 20) (h2 : club_house_eggs = 12) (h3 : town_hall_garden_eggs = 3) :
  ∃ park_eggs, joe_eggs total_eggs club_house_eggs town_hall_garden_eggs park_eggs ∧ park_eggs = 5 :=
by
  sorry

end joe_eggs_around_park_l31_31010


namespace evaluate_f_f_neg2_l31_31304

def f (x : ℝ) : ℝ :=
  if x < 0 then x*x else 2^x - 2

theorem evaluate_f_f_neg2 : f (f (-2)) = 14 := 
  by
    sorry

end evaluate_f_f_neg2_l31_31304


namespace nine_points_chords_l31_31063

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31063


namespace arithmetic_value_l31_31979

theorem arithmetic_value : (8 * 4) + 3 = 35 := by
  sorry

end arithmetic_value_l31_31979


namespace total_donation_correct_l31_31494

-- Define the donations to each orphanage
def first_orphanage_donation : ℝ := 175.00
def second_orphanage_donation : ℝ := 225.00
def third_orphanage_donation : ℝ := 250.00

-- State the total donation
def total_donation : ℝ := 650.00

-- The theorem statement to be proved
theorem total_donation_correct :
  first_orphanage_donation + second_orphanage_donation + third_orphanage_donation = total_donation :=
by
  sorry

end total_donation_correct_l31_31494


namespace lines_intersection_point_l31_31223

theorem lines_intersection_point :
  let line1 (s : Real) := (1 : Real, 2 : Real) + s * (3, -7)
  let line2 (t : Real) := (-5 : Real, 3 : Real) + t * (5, -8)
  ∃ s t, line1(s) = (7, -12) ∧ line2(t) = (7, -12) := 
begin
  sorry
end

end lines_intersection_point_l31_31223


namespace simplify_and_evaluate_l31_31909

variable (x y : ℝ)

theorem simplify_and_evaluate (h : x / y = 3) : 
  (1 + y^2 / (x^2 - y^2)) * (x - y) / x = 3 / 4 :=
by
  sorry

end simplify_and_evaluate_l31_31909


namespace gcf_120_180_240_is_60_l31_31176

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l31_31176


namespace nine_points_chords_l31_31062

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l31_31062


namespace regression_line_correct_l31_31407

open Real

def dataset : List (ℝ × ℝ) := [(1, 4), (2, 7), (3, 10), (4, 13)]

def linear_regression (data : List (ℝ × ℝ)) : ℝ × ℝ :=
  let n := data.length
  let x_bar := (data.map Prod.fst).sum / n
  let y_bar := (data.map Prod.snd).sum / n
  let numerator := (data.map (λ ⟨x, y⟩ => x * y)).sum - n * x_bar * y_bar
  let denominator := (data.map (λ ⟨x, _⟩ => x^2)).sum - n * x_bar ^ 2
  let b := numerator / denominator
  let a := y_bar - b * x_bar
  (b, a)

theorem regression_line_correct : linear_regression dataset = (3, 1) :=
  sorry

end regression_line_correct_l31_31407


namespace polar_and_rectangular_coordinate_equations_l31_31419

noncomputable def polar_coordinate_equation (θ : ℝ) : ℝ :=
  2 * Real.cos (θ - 3 * Real.pi / 4)

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1/2 + Real.sqrt 2 / 4)^2 + (y - Real.sqrt 2 / 4)^2 = 1/4

theorem polar_and_rectangular_coordinate_equations :
  (∀ θ, ρ = polar_coordinate_equation θ) ∧ 
  (∀ x y, Q_midpoint (x y) = trajectory_equation x y) :=
sorry

end polar_and_rectangular_coordinate_equations_l31_31419


namespace charity_ticket_sales_l31_31216

theorem charity_ticket_sales
  (x y p : ℕ)
  (h1 : x + y = 200)
  (h2 : x * p + y * (p / 2) = 3501)
  (h3 : x = 3 * y) :
  150 * 20 = 3000 :=
by
  sorry

end charity_ticket_sales_l31_31216


namespace remaining_red_cards_l31_31600

theorem remaining_red_cards (total_red_cards red_cards_taken_out : ℕ) (h1 : total_red_cards = 26) (h2 : red_cards_taken_out = 10) :
  total_red_cards - red_cards_taken_out = 16 :=
by
  rw [h1, h2]
  exact rfl

end remaining_red_cards_l31_31600


namespace cubic_eq_roots_l31_31386

theorem cubic_eq_roots (x1 x2 x3 : ℕ) (P : ℕ) 
  (h1 : x1 + x2 + x3 = 10) 
  (h2 : x1 * x2 * x3 = 30) 
  (h3 : x1 * x2 + x2 * x3 + x3 * x1 = P) : 
  P = 31 := by
  sorry

end cubic_eq_roots_l31_31386


namespace angles_sum_360_angles_sum_540_angles_sum_n_l31_31322

-- Given that AB is parallel to CD
variable (AB_parallel_CD : Prop)

-- First problem
theorem angles_sum_360 (AB_parallel_CD : AB_parallel_CD) :
  ∀ (angle1 angle2 angle3 : ℝ), angle1 + angle2 + angle3 = 360 :=

by
  sorry

-- Second problem
theorem angles_sum_540 (AB_parallel_CD : AB_parallel_CD) :
  ∀ (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ), angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 540 :=
by
  sorry

-- Third problem
theorem angles_sum_n (AB_parallel_CD : AB_parallel_CD) :
  ∀ n, (∑ i in (finset.range n), (λ i, (angle i))) = 180 * (n - 1) :=
by
  sorry

end angles_sum_360_angles_sum_540_angles_sum_n_l31_31322


namespace area_of_triangle_PFM_l31_31364

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 4 * P.1

-- Define the point and the focus F
def F : ℝ × ℝ := (1, 0)
def M (P : ℝ × ℝ) : ℝ × ℝ := (P.1, 0)

-- Define the distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

-- Define the area of triangle PMF based on base and height
def area_triangle (P M F : ℝ × ℝ) : ℝ :=
  abs (P.1 - M.1) * abs (P.2 - F.2) / 2

-- Define the main theorem statement
theorem area_of_triangle_PFM (P : ℝ × ℝ) 
  (hP : parabola P)
  (hPF : distance P F = 4) :
  area_triangle P (M P) F = 3 * real.sqrt 3 :=
sorry

end area_of_triangle_PFM_l31_31364


namespace probability_positive_difference_ge_three_l31_31164

open Finset Nat

theorem probability_positive_difference_ge_three :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (s.card.choose 2)
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  probability = 15 / 28 :=
by
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := s.card.choose 2
  have total_pairs_eq : total_pairs = 28 := by decide
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  have favorable_pairs_eq : favorable_pairs = 15 := by decide
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  have probability_eq : probability = 15 / 28 := by
    rw [favorable_pairs_eq, total_pairs_eq, ←nat_cast_add, ←rat.div_eq_div_iff]
    norm_num
  exact probability_eq

end probability_positive_difference_ge_three_l31_31164


namespace intervals_strictly_increasing_max_min_values_l31_31770

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

theorem intervals_strictly_increasing (k : ℤ) :
  ∀ x ∈ set.Icc (k * π - 3 * π / 8) (k * π + π / 8), 
  (f x ≥ f (x - 0.0001 * (k * π - x))) :=
sorry

theorem max_min_values : 
  (∃ x ∈ set.Icc 0 (π / 2), f x = sqrt 2 + 1) ∧
  (∃ x ∈ set.Icc 0 (π / 2), f x = 0) :=
sorry

end intervals_strictly_increasing_max_min_values_l31_31770


namespace bridge_length_correct_l31_31241

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 255 := by
  sorry

end bridge_length_correct_l31_31241
