import Mathlib

namespace find_a_and_b_l601_601349

theorem find_a_and_b (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {2, 3}) 
  (hB : B = {x | x^2 + a * x + b = 0}) 
  (h_intersection : A ∩ B = {2}) 
  (h_union : A ∪ B = A) : 
  (a + b = 0) ∨ (a + b = 1) := 
sorry

end find_a_and_b_l601_601349


namespace exp_to_rect_form_l601_601289

open Complex Real

-- Define the problem conditions
def euler_formula (θ : ℝ) : Complex := Complex.exp (θ * Complex.i) = cos θ + Complex.i * sin θ

def pi_six_cos : cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry

def pi_six_sin : sin (Real.pi / 6) = 1 / 2 := by sorry

-- The theorem to prove
theorem exp_to_rect_form : 
  (sqrt 3 : ℂ) * Complex.exp ((13 * Real.pi / 6) * Complex.i) = 
  (3 / 2 : ℂ) + (Complex.i * (sqrt 3 / 2 : ℂ)) :=
by 
  have h1 := euler_formula (13 * Real.pi / 6)
  -- The proof is skipped
  sorry

end exp_to_rect_form_l601_601289


namespace moving_circle_trajectory_line_tangent_fixed_point_l601_601054

/-- Given a moving circle above the x-axis that is tangent to the x-axis and externally tangent 
to the circle x^2 + y^2 - 2y = 0, prove that the center of the moving circle follows the equation 
x^2 = 4y, y > 0. --/
theorem moving_circle_trajectory (x y : ℝ) (hy : y > 0) 
(h_tangent : (sqrt (x^2 + (y - 1)^2) - 1 = y)) : x^2 = 4y :=
sorry

/-- Given a point P(a, b) (a ≠ 0, b ≠ 0), and if two tangents PM and PN to the curve x^2 = 4y can 
be drawn from P, and the line MN connecting the tangent points M and N is perpendicular to OP, 
then prove that line MN passes through the fixed point (0, 2).
--/
theorem line_tangent_fixed_point (a b x1 x2 y1 y2 : ℝ)
(ha : a ≠ 0) (hb : b ≠ 0)
(h_curve : ∀ (x : ℝ), x^2 = 4 * (1/4) * x^2)
(h_tangent_points : y1 = (1/4) * x1^2 ∧ y2 = (1/4) * x2^2)
(h_perpendicular : (x1, y1) ≠ (x2, y2) ∧ (a, b) ∉ (x1, y1) ∧ (a, b) ∉ (x2, y2) ∧
 - (1/x1) = b/(a - 0))
(h_slope : b = -2) : (∀ x y, (y - 2) = 0) :=
sorry

end moving_circle_trajectory_line_tangent_fixed_point_l601_601054


namespace sin_cos_theta_identity_l601_601415

theorem sin_cos_theta_identity
  (a b θ : ℝ)
  (h : (sin θ) ^ 4 / a + (cos θ) ^ 4 / b = 1 / (a + b)) :
  (sin θ) ^ 8 / a ^ 3 + (cos θ) ^ 8 / b ^ 3 = 1 / (a + b) ^ 3 :=
by
  sorry

end sin_cos_theta_identity_l601_601415


namespace trapezoid_properties_proof_l601_601775

noncomputable def trapezoid_inscribed_circle_diagonal_and_area 
  (r : ℝ) (smaller_base : ℝ) (chord_length : ℝ) 
  (l_diag : ℝ) (area : ℝ) : Prop :=
  ∃ (length_diagonal : ℝ) (area_trapezoid : ℝ),
    (length_diagonal = 5) ∧ (area_trapezoid = (975 * real.sqrt(3)) / 196) ∧
    -- Conditions
    r = real.sqrt(7) ∧
    smaller_base = 4 ∧
    chord_length = 5

theorem trapezoid_properties_proof :
  trapezoid_inscribed_circle_diagonal_and_area (real.sqrt 7) 4 5 5 ((975 * real.sqrt(3)) / 196) :=
by
  exists 5
  exists (975 * real.sqrt 3) / 196
  repeat { exact rfl }
  sorry

end trapezoid_properties_proof_l601_601775


namespace ladder_distance_l601_601676

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601676


namespace complex_quadrant_l601_601925

def z1 : ℂ := 1 + 2 * Complex.I
def z2 : ℂ := 1 - Complex.I
def z : ℂ := z1 + z2

theorem complex_quadrant :
  ∃ (x y : ℝ), z = x + y * Complex.I ∧ x > 0 ∧ y > 0 :=
by
  use 2, 1
  simp [z, z1, z2]
  split
  · refl
  · split
    · linarith
    · linarith
  sorry -- proof here

end complex_quadrant_l601_601925


namespace determinant_eq_l601_601804

theorem determinant_eq :
  forall (a b : ℝ), 
  matrix.det ![
    ![1, sin (a - b), sin a],
    ![sin (a - b), 1, sin b],
    ![sin a, sin b, 1]
  ] = 1 - sin a ^ 2 - sin b ^ 2 - sin (a - b) ^ 2 + 2 * sin a * sin b * sin (a - b) := 
by
  intros
  sorry

end determinant_eq_l601_601804


namespace paige_initial_pencils_l601_601512

theorem paige_initial_pencils (used_pencils : ℕ) (current_pencils : ℕ) : 
  used_pencils = 3 → 
  current_pencils = 91 → 
  (used_pencils + current_pencils = 94) :=
by
  intros h_used h_current
  rw [h_used, h_current]
  exact rfl

end paige_initial_pencils_l601_601512


namespace modular_inverse_sum_l601_601797

theorem modular_inverse_sum (a b c d e f g h : ℤ) :
  (2^9 ≡ a [MOD 11]) →
  (2^8 ≡ b [MOD 11]) →
  (2^7 ≡ c [MOD 11]) →
  (2^6 ≡ d [MOD 11]) →
  (2^5 ≡ e [MOD 11]) →
  (2^4 ≡ f [MOD 11]) →
  (2^3 ≡ g [MOD 11]) →
  (2^2 ≡ h [MOD 11]) →
  (a + b + c + d + e + f + g + h ≡ 8 [MOD 11]) :=
  sorry

end modular_inverse_sum_l601_601797


namespace probability_of_vowel_initials_l601_601027

/-- In a class with 26 students, each student has unique initials that are double letters
    (i.e., AA, BB, ..., ZZ). If the vowels are A, E, I, O, U, and W, then the probability of
    randomly picking a student whose initials are vowels is 3/13. -/
theorem probability_of_vowel_initials :
  let total_students := 26
  let vowels := ['A', 'E', 'I', 'O', 'U', 'W']
  let num_vowels := 6
  let probability := num_vowels / total_students
  probability = 3 / 13 :=
by
  sorry

end probability_of_vowel_initials_l601_601027


namespace convert_to_rectangular_form_l601_601287

noncomputable def polar_expression : ℂ := √3 * complex.exp (13 * real.pi * complex.I / 6)
noncomputable def rectangular_form : ℂ := (3 / 2) + (√3 / 2) * complex.I

theorem convert_to_rectangular_form :
  polar_expression = rectangular_form :=
by 
  sorry

end convert_to_rectangular_form_l601_601287


namespace collinear_points_sum_l601_601303

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end collinear_points_sum_l601_601303


namespace boundary_even_length_l601_601734

theorem boundary_even_length (m n : ℕ) (hm : m = 2010) (hn : n = 2011)
  (cover : ∀i j, (i < m) ∧ (j < n) → domino_covering.exists_some_domino_placedᵒvertically_or_horizontally (i, j) ) :
  even (boundary_length */
    domino_covering.boundary_between_horizontal_and_vertical (domino_covering.board_cover m n cover)) :=
sorry

end boundary_even_length_l601_601734


namespace selling_price_l601_601555

-- Define the terms
def cp : ℝ := 768 / 1.20
def loss_price : ℝ := 448
def profit_percent : ℝ := 0.20

-- Define the conditions
def condition_profit_loss (sp cp loss_price : ℝ) : Prop :=
  sp - cp = cp - loss_price

def condition_profit_percent (sp cp : ℝ) : Prop :=
  sp = cp * (1 + profit_percent)

-- Define the main theorem
theorem selling_price (sp cp : ℝ) (loss_price : ℝ) (profit_percent : ℝ) 
  (h_cp : cp = 768 / 1.20)
  (h_loss : loss_price = 448)
  (h_profit : profit_percent = 0.20)
  (h_profit_loss : condition_profit_loss sp cp loss_price)
  (h_profit_percent : condition_profit_percent 768 cp) :
  sp = 832 := sorry

end selling_price_l601_601555


namespace evaluate_complex_fraction_l601_601310

theorem evaluate_complex_fraction :
  (⌈(23 / 8 : ℚ) - ⌈32 / 19⌉⌉ / ⌈32 / 8 + ⌈(8 * 19) / 32⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end evaluate_complex_fraction_l601_601310


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601177

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601177


namespace max_marked_nodes_in_6x6_grid_l601_601777

theorem max_marked_nodes_in_6x6_grid : 
  ∃ (markable_nodes : ℕ), 
  (∀ (grid : Fin 6 × Fin 6 → bool), 
  let marked_nodes := 
    {node : Fin 7 × Fin 7 | 
      (∃ shaded unshaded, 
        (shaded = (adjacent_cells node grid).filter id) ∧
        (unshaded = (adjacent_cells node grid).filter (λ x, ¬ x)) ∧
        shaded.card = unshaded.card) ∧
        (adjacent_cells node grid).card = 4
    } in
    marked_nodes.card ≤ markable_nodes) ∧
  markable_nodes = 45 :=
begin
  sorry
end

def adjacent_cells (node : Fin 7 × Fin 7) (grid : Fin 6 × Fin 6 → bool) : list bool :=
  sorry

end max_marked_nodes_in_6x6_grid_l601_601777


namespace evaluate_expression_l601_601311

theorem evaluate_expression : (3 : ℚ) / (1 - (2 : ℚ) / 5) = 5 := sorry

end evaluate_expression_l601_601311


namespace factorized_sum_is_33_l601_601126

theorem factorized_sum_is_33 (p q r : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 21 * x + 110 = (x + p) * (x + q))
  (h2 : ∀ x : ℤ, x^2 - 23 * x + 132 = (x - q) * (x - r)) : 
  p + q + r = 33 := by
  sorry

end factorized_sum_is_33_l601_601126


namespace integral_greatest_int_equals_sum_l601_601490

noncomputable def greatest_int_le (u : ℝ) : ℤ := int.floor u

variable (f : ℝ → ℝ)
variable (x : ℝ)
variable [continuous f]

theorem integral_greatest_int_equals_sum (h : x > 1) :
  ∫ (u : ℝ) in 1..x, (greatest_int_le u : ℝ) * ((greatest_int_le u + 1) : ℝ) * f u = 
  2 * ∑ (i : ℤ) in finset.range ⌊x⌋.succ, i * ∫ (u : ℝ) in (i : ℝ)..x, f u := 
sorry

end integral_greatest_int_equals_sum_l601_601490


namespace vehicle_restriction_today_is_thursday_l601_601572

theorem vehicle_restriction_today_is_thursday :
  (∀ day : ℕ, 1 ≤ day ∧ day ≤ 5 → ∃ day : ℕ, day = 4 → ¬(day = today.E)) ∧  -- vehicle E is restricted on Thursday
  ∃ day : ℕ, day = today.B - 1 → ¬(day = ⟨1, 7⟩) ∧ ¬(day = ⟨7, 7⟩) ∧        -- vehicle B was restricted yesterday, so not Monday or Sunday
  ∃ days : list ℕ, days = [today.A, today.C] ∧ (∀ day ∈ days, day = today → (today + 4) ∈ days) ∧ -- vehicles A and C free for next 4 days
  ∃ day : ℕ, day = (today + 1) → day = today.E ∧                             -- vehicle E can be on the road tomorrow
  ∃ days : list ℕ, days.length ≥ 4 ∧ days ⊆ [today.A, today.B, today.C, today.D, today.E] 
  → today = 4 := -- today is Thursday
sorry

end vehicle_restriction_today_is_thursday_l601_601572


namespace rationalize_simplify_l601_601109

theorem rationalize_simplify :
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 :=
by
  sorry

end rationalize_simplify_l601_601109


namespace unique_X_Y_l601_601470

structure Triangle (A B C : Type) :=
(A_dist : A ≠ B ∧ A ≠ C ∧ B ≠ C)

structure Point (A : Type) :=
(coord : A)

variable {A : Type} [DecidableEq A]

def is_isosceles {A : Type} [metric_space A] (T : Triangle A) (P1 P2 : Point A) : Prop :=
dist T.A P1.coord = dist T.A P2.coord

def is_similar {A : Type} (T1 T2 : Triangle A) : Prop :=
-- definition for similarity goes here

theorem unique_X_Y (A B C X Y : A) 
  [metric_space A]
  (T : Triangle A)
  (is_iso_ABX : is_isosceles T (Point.mk A X) (Point.mk A B))
  (is_iso_ACY : is_isosceles T (Point.mk C Y) (Point.mk C A))
  (similarity : is_similar (Triangle.mk A B X) (Triangle.mk A C Y))
  (equidistance : dist C X = dist B Y) :
  ∃! (X Y : A), 
    is_isosceles (Triangle.mk A B X) (Point.mk A X) (Point.mk A B) ∧ 
    is_isosceles (Triangle.mk A C Y) (Point.mk C Y) (Point.mk C A) ∧ 
    is_similar (Triangle.mk A B X) (Triangle.mk A C Y) ∧ 
    dist C X = dist B Y :=
by 
  sorry

end unique_X_Y_l601_601470


namespace smallest_a_has_50_perfect_squares_l601_601859

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601859


namespace find_number_eq_seven_point_five_l601_601198

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601198


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601183

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601183


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601429

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601429


namespace number_of_real_solutions_l601_601819

theorem number_of_real_solutions :
  (∃! x : ℝ, 3 ^ (3 * x^3 - 9 * x^2 + 15 * x - 5) = 1) :=
sorry

end number_of_real_solutions_l601_601819


namespace number_of_pupils_l601_601591

-- Define the conditions.
variables (n : ℕ) -- Number of pupils in the class.

-- Axioms based on the problem statement.
axiom marks_difference : 67 - 45 = 22
axiom avg_increase : (1 / 2 : ℝ) * n = 22 

-- The theorem we need to prove.
theorem number_of_pupils : n = 44 := by
  -- Proof will go here.
  sorry

end number_of_pupils_l601_601591


namespace find_number_l601_601191

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601191


namespace work_done_one_depth_l601_601768

variable (V p0 γ h : ℝ)

theorem work_done_one_depth (V p0 γ h : ℝ) : 
  let W := p0 * V * Real.log ((γ * h + p0) / p0) in 
  W = p0 * V * Real.log ((γ * h + p0) / p0) := 
begin
  sorry,
end

end work_done_one_depth_l601_601768


namespace ladder_base_distance_l601_601657

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601657


namespace max_possible_area_l601_601981

-- Definitions and conditions
def Circle (r : ℝ) := { p : ℝ × ℝ // p.fst^2 + p.snd^2 ≤ r^2 }
def line := { p : ℝ × ℝ // p.snd = 0 }
def tangent_to_line_at_point (c : Circle r) (ℓ : line) (B : ℝ × ℝ) := exists (t : ℝ), (B.snd = 0) ∧ (p ∈ c) ∧ (p.snd = B.snd)

-- The proof statement
theorem max_possible_area :
  let c1 := Circle 2
  let c2 := Circle 4
  let c3 := Circle 6 
  let c4 := Circle 8 in
  ∃ (B : ℝ × ℝ) (ℓ : line), 
    tangent_to_line_at_point c1 ℓ B ∧
    tangent_to_line_at_point c2 ℓ B ∧
    tangent_to_line_at_point c3 ℓ B ∧
    tangent_to_line_at_point c4 ℓ B ∧
    (area_of_region_T c1 c2 c3 c4 = 84 * π) := 
sorry

end max_possible_area_l601_601981


namespace range_of_m_l601_601128

noncomputable def f (x m : ℝ) : ℝ := -x^2 - 4 * m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → f x1 m ≥ f x2 m) ↔ m ≥ -1 := 
sorry

end range_of_m_l601_601128


namespace magnitude_complex_expression_l601_601438

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601438


namespace g_minimum_l601_601393

noncomputable theory

open Real

def f (x : ℝ) (a c : ℝ) : ℝ := a * x ^ 2 - (1 / 2) * x + c

axiom f_conditions (a c : ℝ) : 
  (f 1 a c = 0) ∧ (∀ x : ℝ, f x a c ≥ 0)

example : ∃ (a c : ℝ), f_conditions (1 / 4) (1 / 4) :=
by 
  use (1 / 4)
  use (1 / 4)
  sorry

def g (x : ℝ) (a c m : ℝ) : ℝ := 4 * (f x a c) - m * x

theorem g_minimum : 
  ∃ m : ℝ, (m = 3) ∧ (∀ x ∈ set.Icc m (m + 2), 
  g x (1 / 4) (1 / 4) m > -5 → g x (1 / 4) (1 / 4) m = -5) :=
by 
  use 3
  split
  { -- proof that m = 3
    refl
  }
  {
    -- proof that g has minimum value -5 in (Icc 3 (3 + 2))
    sorry
  }

end g_minimum_l601_601393


namespace ladder_base_distance_l601_601651

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601651


namespace sequence_general_term_l601_601403

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
 (h₂ : ∀ n : ℕ, sqrt (a (n + 1)) - sqrt (a n) = 3) :
 ∀ n : ℕ, a n = (3 * n - 2)^2 := 
by
  sorry

end sequence_general_term_l601_601403


namespace equal_prob_first_ace_l601_601045

/-
  Define the problem:
  In a 4-player card game with a 32-card deck containing 4 aces,
  prove that the probability of each player drawing the first ace is 1/8.
-/

namespace CardGame

def deck : list ℕ := list.range 32

def is_ace (card : ℕ) : Prop := card % 8 = 0

def player_turn (turn : ℕ) : ℕ := turn % 4

def first_ace_turn (deck : list ℕ) : ℕ :=
deck.find_index is_ace

theorem equal_prob_first_ace :
  ∀ (deck : list ℕ) (h : deck.cardinality = 32) (h_ace : ∑ (card ∈ deck) (is_ace card) = 4),
  ∀ (player : ℕ), player < 4 → (∃ n < 32, first_ace_turn deck = some n ∧ player_turn n = player) →
  (deck.countp is_ace) / 32 = 1 / 8 :=
by sorry

end CardGame

end equal_prob_first_ace_l601_601045


namespace distance_from_wall_l601_601635

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601635


namespace calc_expression_l601_601091

namespace MathProblem

-- Define the sides of the triangle
def PQ : ℝ := 7
def PR : ℝ := 8
def QR : ℝ := 5

-- Define the angles (in degrees)
noncomputable def P : ℝ := sorry -- introduce the proper definition
noncomputable def Q : ℝ := sorry -- introduce the proper definition
noncomputable def R : ℝ := sorry -- introduce the proper definition

-- Main theorem statement
theorem calc_expression :
  (cos ((P - Q) / 2) / sin (R / 2)) - (sin ((P - Q) / 2) / cos (R / 2)) = 16 / 7 :=
sorry

end MathProblem

end calc_expression_l601_601091


namespace smallest_natural_number_with_50_squares_in_interval_l601_601868

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601868


namespace KLMN_is_parallelogram_l601_601228

variable (A B C D P K L M N : Point)
variables (hConvex : ConvexQuadrilateral A B C D)
variables (hIntAngleBisectors : 
  ∃ (K : Point), IsAngleBisector (AngleBisector A P B) A B ∧ 
  ∃ (L : Point), IsAngleBisector (AngleBisector B P C) B C ∧ 
  ∃ (M : Point), IsAngleBisector (AngleBisector C P D) C D ∧ 
  ∃ (N : Point), IsAngleBisector (AngleBisector D P A) D A)

variables (hPerpBisectorIntersections : IntersectPerpBisectorsDiagonal P A C B D)

theorem KLMN_is_parallelogram : IsParallelogram K L M N :=
sorry

end KLMN_is_parallelogram_l601_601228


namespace math_problem_l601_601336

noncomputable def proof_problem (x y z : ℚ) : Prop :=
  (x + 1/3 * z = y) ∧
  (y + 1/3 * x = z) ∧
  (z - x = 10) →
  x = 10 ∧ y = 50/3 ∧ z = 20

theorem math_problem : ∃ (x y z : ℚ), proof_problem x y z :=
by
  use [10, 50/3, 20]
  dsimp [proof_problem]
  split
  { sorry }
  split
  { sorry }
  { sorry }

end math_problem_l601_601336


namespace ladder_base_distance_l601_601729

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601729


namespace age_difference_is_eight_l601_601488

theorem age_difference_is_eight (A B k : ℕ)
  (h1 : A = B + k)
  (h2 : A - 1 = 3 * (B - 1))
  (h3 : A = 2 * B + 3) :
  k = 8 :=
by sorry

end age_difference_is_eight_l601_601488


namespace smallest_a_with_50_squares_l601_601903


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601903


namespace smallest_natural_number_with_50_squares_in_interval_l601_601866

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601866


namespace min_colors_for_grid_coloring_l601_601482

theorem min_colors_for_grid_coloring : ∃c : ℕ, c = 4 ∧ (∀ (color : ℕ × ℕ → ℕ), 
  (∀ i j : ℕ, i < 5 ∧ j < 5 → 
     ((i < 4 → color (i, j) ≠ color (i+1, j+1)) ∧ 
      (j < 4 → color (i, j) ≠ color (i+1, j-1))) ∧ 
     ((i > 0 → color (i, j) ≠ color (i-1, j-1)) ∧ 
      (j > 0 → color (i, j) ≠ color (i-1, j+1)))) → 
  c = 4) :=
sorry

end min_colors_for_grid_coloring_l601_601482


namespace last_two_digits_condition_l601_601545

-- Define the function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Given numbers
def n1 := 122
def n2 := 123
def n3 := 125
def n4 := 129

-- The missing number
variable (x : ℕ)

theorem last_two_digits_condition : 
  last_two_digits (last_two_digits n1 * last_two_digits n2 * last_two_digits n3 * last_two_digits n4 * last_two_digits x) = 50 ↔ last_two_digits x = 1 :=
by 
  sorry

end last_two_digits_condition_l601_601545


namespace annie_diorama_time_l601_601275

theorem annie_diorama_time (P B : ℕ) (h1 : B = 3 * P - 5) (h2 : B = 49) : P + B = 67 :=
sorry

end annie_diorama_time_l601_601275


namespace possible_positional_relationships_l601_601492

theorem possible_positional_relationships (a b x₁ y₁ : ℝ) (hP : x₁^2 + y₁^2 = 9)
    (hQ : (a - x₁)^2 + (b - y₁)^2 = 1) :
    (Exists (λ c: ℝ, c = 2 ∨ c = 3 ∨ c = 4)) :=
by
  sorry

end possible_positional_relationships_l601_601492


namespace Robin_cut_off_23_53_percent_l601_601517

variable (original_hair_length : ℝ)
variable (new_hair_length : ℝ)

def percentage_cut_off (original_hair_length new_hair_length : ℝ) : ℝ :=
  ((original_hair_length - new_hair_length) / original_hair_length) * 100

theorem Robin_cut_off_23_53_percent :
  original_hair_length = 17 → new_hair_length = 13 →
  percentage_cut_off original_hair_length new_hair_length ≈ 23.53 :=
by
  intros h1 h2
  sorry

end Robin_cut_off_23_53_percent_l601_601517


namespace general_term_a_n_sum_c_n_l601_601918

-- Conditions
def is_arithmetic_sequence (n : ℕ) (a S : ℕ → ℕ) : Prop :=
  ∀ n ≥ 2, S n + n = 2 * a n ∧ S (n - 1) + (n - 1) = 2 * a (n - 1)

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * (Int.log2 (1 + a n)).toNat - 1

-- Problem 1: General term formula for sequence (a_n)
theorem general_term_a_n (a S : ℕ → ℕ)
  (h_arith : is_arithmetic_sequence a S) :
  ∀ n, a n = 2^n - 1 :=
sorry

-- Problem 2: Sum of sequence (c_n)
theorem sum_c_n (a S : ℕ → ℕ)
  (h_arith : is_arithmetic_sequence a S)
  (h_b_n : ∀ n, b_n a n = 2 * n - 1) :
  ∑ i in range 100, (b_n a (i + 1)) - ∑ i in range 7, a (i + 1) + 7 = 11202 :=
sorry

end general_term_a_n_sum_c_n_l601_601918


namespace overall_average_score_l601_601239

theorem overall_average_score (first_6_avg last_4_avg : ℝ) (n_first n_last n_total : ℕ) 
    (h_matches : n_first + n_last = n_total)
    (h_first_avg : first_6_avg = 41)
    (h_last_avg : last_4_avg = 35.75)
    (h_n_first : n_first = 6)
    (h_n_last : n_last = 4)
    (h_n_total : n_total = 10) :
    ((first_6_avg * n_first + last_4_avg * n_last) / n_total) = 38.9 := by
  sorry

end overall_average_score_l601_601239


namespace triangle_angle_bisector_eq_l601_601024

theorem triangle_angle_bisector_eq (A B C D E : Point)
(h_triangle: Triangle A B C)
(h_bisector_1: AngleBisector B D A C)
(h_bisector_2: AngleBisector C E A B)
(h_eq_bisectors: length B D = length C E):
length A B = length A C :=
begin
  sorry
end

end triangle_angle_bisector_eq_l601_601024


namespace sum_distances_circumcircle_sum_distances_unrestricted_l601_601380

-- Define the problem statement for the first question
theorem sum_distances_circumcircle (n : ℕ) (n_pos : 0 < n) (A : Fin n → ℂ) (P : ℂ) (hA : ∀ k : Fin n, ∥A k∥ = 1) (hP : ∥P∥ = 1) :
  (∑ k in Finset.univ, complex.abs (A k - P)) ≤ 2 * real.csc (π / (2 * n)) ∧ 
  (∑ k in Finset.univ, complex.abs (A k - P)) ≥ 2 * real.cot (π / (2 * n)) := sorry

-- Define the problem statement for the second question
theorem sum_distances_unrestricted (n : ℕ) (n_pos : 0 < n) (A : Fin n → ℂ) (P : ℂ) (hA : ∀ k : Fin n, ∥A k∥ = 1) :
  (∑ k in Finset.univ, complex.abs (A k - P)) ≥ n := sorry

end sum_distances_circumcircle_sum_distances_unrestricted_l601_601380


namespace find_divisor_l601_601510

theorem find_divisor (x : ℕ) (h : 172 = 10 * x + 2) : x = 17 :=
sorry

end find_divisor_l601_601510


namespace instantaneous_velocity_at_3_l601_601236

-- Definitions based on the conditions.
def displacement (t : ℝ) := 2 * t ^ 3

-- The statement to prove.
theorem instantaneous_velocity_at_3 : (deriv displacement 3) = 54 := by
  sorry

end instantaneous_velocity_at_3_l601_601236


namespace num_9_digit_integers_l601_601009

theorem num_9_digit_integers : ∃ n : ℕ, n = 9 * 10^8 := by
  use 900000000
  sorry

end num_9_digit_integers_l601_601009


namespace max_fleas_on_board_l601_601985

theorem max_fleas_on_board : ∃ max_fleas : ℕ, max_fleas = 40 ∧ 
  (∀ t : nat, t ≤ 60 → 
    (∀ (pos1 pos2 : fin 10 × fin 10), 
      (flea_position t pos1 ∧ flea_position t pos2) → pos1 ≠ pos2)) :=
sorry

end max_fleas_on_board_l601_601985


namespace triangle_angle_obtuse_l601_601999

theorem triangle_angle_obtuse (A B C D : Type) [n : nonempty (triangle A B C)] :
  D ≠ B ∧ AD / DC = AB / BC → ∠ ACB > 90° :=
by
  sorry

end triangle_angle_obtuse_l601_601999


namespace complement_A_in_U_l601_601950

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_A_in_U : (U \ A) = {3, 9} := 
by sorry

end complement_A_in_U_l601_601950


namespace inequality_holds_for_gt_sqrt2_l601_601106

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l601_601106


namespace chestnuts_distribution_l601_601264

theorem chestnuts_distribution:
  ∃ (chestnuts_Alya chestnuts_Valya chestnuts_Galya : ℕ),
    chestnuts_Alya + chestnuts_Valya + chestnuts_Galya = 70 ∧
    4 * chestnuts_Valya = 3 * chestnuts_Alya ∧
    6 * chestnuts_Galya = 7 * chestnuts_Alya ∧
    chestnuts_Alya = 24 ∧
    chestnuts_Valya = 18 ∧
    chestnuts_Galya = 28 :=
by {
  sorry
}

end chestnuts_distribution_l601_601264


namespace card_distribution_remainders_l601_601461

theorem card_distribution_remainders (m n : ℕ) (cards : Fin (m * n) → ℕ) 
  (boys_cards girls_cards : Fin m → Fin n → Fin (m * n)) :
  (∀ i, boys_cards i = cards (Fin.castSucc i)) →
  (∀ j, girls_cards j = cards (Fin.castSucc j + m)) →
  (∀ i j, (cards i + cards j) % (m * n) ≠ (cards i + cards j) % (m * n) → i ≠ j) →
  (∃ (m = 1) ∨ (n = 1)) :=
sorry

end card_distribution_remainders_l601_601461


namespace baron_munchausen_l601_601169

def Point : Type := ℝ × ℝ

structure Snake (points : list Point) := 
  (segment_angles_equal : ∀ (i j : ℕ), (i < points.length - 1) → (j < points.length - 1) → angle (points.nth_le i sorry) (points.nth_le (i+1) sorry) = angle (points.nth_le j sorry) (points.nth_le (j+1) sorry))
  (neighboring_segments_different_half_planes : ∀ (i : ℕ), (1 ≤ i) → (i < points.length - 1) → half_plane (points.nth_le i sorry) (points.nth_le (i-1) sorry) ≠ half_plane (points.nth_le i sorry) (points.nth_le (i+1) sorry))

theorem baron_munchausen (points : list Point) (h : points.length = 6) : 
  ∃ (snakes : list (Snake points)), snakes.length = 6 :=
sorry

end baron_munchausen_l601_601169


namespace sphere_cross_section_area_l601_601829

theorem sphere_cross_section_area (R : ℝ) (d : ℝ) (hR : R = 3) (hd : d = 2) : 
    let r := Real.sqrt (R^2 - d^2) in
    let area := Real.pi * r^2 in
    area = 5 * Real.pi := 
  by
  -- Introduce the values
  rw [hR, hd]
  -- Let r be the radius of the cross-section circle
  let r := Real.sqrt (3^2 - 2^2)
  have hr : r = Real.sqrt 5 := by
    -- Calculation details:
    calc
      r = Real.sqrt (3^2 - 2^2) : by rw [hR, hd]
      ... = Real.sqrt (9 - 4)    : by norm_num
      ... = Real.sqrt 5         : by norm_num
  -- Hence the area of the cross-section circle
  let area := Real.pi * r^2
  have harea : area = 5 * Real.pi := by
    calc
      area = Real.pi * (Real.sqrt 5)^2 : by rw hr
      ...  = Real.pi * 5               : by norm_num
      ...  = 5 * Real.pi               : by ring
  exact harea

end sphere_cross_section_area_l601_601829


namespace ladder_base_distance_l601_601670

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601670


namespace num_good_sets_of_2004_l601_601241

-- Define the value of each card numbered k as 2^k
def card_value (k : ℕ) : ℕ := 2 ^ k

-- Define the set of card values we have in the deck
def card_values : Set ℕ := (Finset.range 11).image card_value

-- Calculate the number of good sets of cards
def good_sets_count (sum_value : ℕ) : ℕ :=
  if H : sum_value < 2 ^ 11 then
    -- Given the constraint 2004 < 2 ^ 11, evaluate the sum series
    let f (x : ℕ) : ℕ := ∑ i in (Finset.range 1003), 2005 - 2 * i
    f sum_value
  else
    0

theorem num_good_sets_of_2004 : good_sets_count 2004 = 1006009 :=
  by
    -- Providing the proof directly results in the desired count
    rw [good_sets_count]
    -- Assumes sum calculation is correct from solution
    sorry

end num_good_sets_of_2004_l601_601241


namespace largest_n_unique_k_l601_601582

theorem largest_n_unique_k : ∃ n : ℕ, (∀ k : ℤ, (8 / 15 : ℚ) < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (7 / 13 : ℚ) → k = unique_k) ∧ n = 112 :=
sorry

end largest_n_unique_k_l601_601582


namespace ladder_base_distance_l601_601656

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601656


namespace nonzero_terms_expansion_l601_601956

def poly1 (x : ℝ) : ℝ := (x - 2) * (3 * x ^ 2 - 2 * x + 5)
def poly2 (x : ℝ) : ℝ := 4 * (x ^ 3 + x ^ 2 - 3 * x)

def expanded_poly : ℝ → ℝ
| x => (3 * x ^ 3 - 8 * x ^ 2 + 9 * x - 10) + (4 * x ^ 3 + 4 * x ^ 2 - 12 * x)

def final_poly : ℝ → ℝ
| x => 7 * x ^ 3 - 4 * x ^ 2 - 3 * x - 10

theorem nonzero_terms_expansion : 
  ∀ (x : ℝ), (poly1 x + poly2 x) = final_poly x → 
  (∃ n : ℕ, n = 4) :=
begin
  intro x,
  intro h,
  use 4,
  sorry
end

end nonzero_terms_expansion_l601_601956


namespace sum_of_solutions_l601_601519

theorem sum_of_solutions : 
  (∃ x, 3^(x^2 + 6*x + 9) = 27^(x + 3)) → (∀ x₁ x₂, (3^(x₁^2 + 6*x₁ + 9) = 27^(x₁ + 3) ∧ 3^(x₂^2 + 6*x₂ + 9) = 27^(x₂ + 3)) → x₁ + x₂ = -3) :=
sorry

end sum_of_solutions_l601_601519


namespace problem_1_problem_2_l601_601396

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem problem_1 (k : ℤ) :
  (∀ x, f(x + Real.pi) = f(x)) ∧
  ((k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4) → Real.deriv f x > 0) ∧
  ((k * Real.pi + Real.pi / 4 < x ∧ x < k * Real.pi + 3 * Real.pi / 4) → Real.deriv f x < 0) := 
sorry

variables (a b c A B C : ℝ)
variables (acute_triangle : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2)
variables (cos_B_eq : Real.cos B = 1 / 3)
variables (c_eq : c = Real.sqrt 6)
variables (f_half_C_eq : f (C / 2) = -1 / 4)

theorem problem_2 (acute_triangle : acute_triangle) (cos_B_eq : cos_B_eq) (c_eq : c_eq) (f_half_C_eq : f_half_C_eq) :
  b = 8 / 3 :=
sorry

end problem_1_problem_2_l601_601396


namespace sum_a1_a5_l601_601919

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 5 = 11 :=
sorry

end sum_a1_a5_l601_601919


namespace angle_A_is_120_degrees_area_of_triangle_l601_601478

-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively, and given
axiom ABC : Type
axiom a b c : Real
axiom A B C : Real
axiom side_opposite : (a = | |- opposite_angle A) (b = | |- opposite_angle B) (c = | |- opposite_angle C)
axiom angle_sum : (0 < A + B + C < 180)

-- This angle and relation between trigonometric functions are given
axiom trig_identity : cos (B - C) - 2 * sin B * sin C = 1 / 2
axiom side_a : a = sqrt 3
axiom side_c : c = 2 * cos C

-- We need to prove these two statements
theorem angle_A_is_120_degrees : A = 120 :=
by
  sorry

theorem area_of_triangle : area ABC = (3 - sqrt 3) / 4 :=
by
  sorry

end angle_A_is_120_degrees_area_of_triangle_l601_601478


namespace count_odd_numbers_in_range_l601_601011

-- Define the properties of odd numbers and their bounds
def odd (n : Nat) : Prop := n % 2 = 1

def greaterThan (n m : Nat) : Prop := n > m

def lessThan (n m : Nat) : Prop := n < m

-- Define the range and count the odd numbers within those bounds
def countOddNumbers (a b : Nat) : Nat :=
  Nat.card { x // odd x ∧ greaterThan x a ∧ lessThan x b }

theorem count_odd_numbers_in_range :
  countOddNumbers 215 500 = 142 := 
sorry

end count_odd_numbers_in_range_l601_601011


namespace ratio_of_roosters_to_hens_l601_601153

theorem ratio_of_roosters_to_hens
  (total_chickens : ℕ) (roosters : ℕ) (hens : ℕ)
  (h_total : total_chickens = 9000)
  (h_roosters : roosters = 6000)
  (h_hens : hens = total_chickens - roosters) :
  roosters / hens = 2 :=
by {
  subst_vars,
  sorry
}

end ratio_of_roosters_to_hens_l601_601153


namespace john_needs_total_planks_l601_601345

theorem john_needs_total_planks : 
  let large_planks := 12
  let small_planks := 17
  large_planks + small_planks = 29 :=
by
  sorry

end john_needs_total_planks_l601_601345


namespace find_f_at_2_l601_601385

variables {f g : ℝ → ℝ}
variables {a : ℝ}

-- f is an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- g is an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- main theorem statement
theorem find_f_at_2 (h_odd_f : is_odd f) (h_even_g : is_even g) 
  (h_fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2)
  (h_g2_eq_a : g 2 = a) (h_a_gt_zero : 0 < a) (h_a_ne_one : a ≠ 1) :
  f 2 = 15 / 4 :=
begin
  sorry
end

end find_f_at_2_l601_601385


namespace factor_x4_plus_16_l601_601314

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end factor_x4_plus_16_l601_601314


namespace smallest_number_divide_perfect_cube_l601_601175

theorem smallest_number_divide_perfect_cube (n : ℕ):
  n = 450 → (∃ m : ℕ, n * m = k ∧ ∃ k : ℕ, k ^ 3 = n * m) ∧ (∀ m₂ : ℕ, (n * m₂ = l ∧ ∃ l : ℕ, l ^ 3 = n * m₂) → m ≤ m₂) → m = 60 :=
by
  sorry

end smallest_number_divide_perfect_cube_l601_601175


namespace solution_set_of_inequality_l601_601147

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality_l601_601147


namespace last_digit_odd_second_digit_even_l601_601116

theorem last_digit_odd_second_digit_even (a b : ℕ) (n = 10 * a + b) (hb : b ∈ {1, 3, 5, 7, 9}) :
  let last_two_digits := (n * n) % 100
  in (last_two_digits / 10) % 2 = 0 :=
sorry

end last_digit_odd_second_digit_even_l601_601116


namespace min_y1_at_45_degrees_no_min_max_y2_l601_601335

noncomputable def y1 (x : ℝ) : ℝ := (1 / Real.tan x) + (1 / Real.cot x)
noncomputable def y2 (x : ℝ) : ℝ := (1 / Real.tan x) - (1 / Real.cot x)

-- Lean doesn't support degrees in its standard library, so we convert 45 degrees to radians
def forty_five_degrees_in_radians : ℝ := Real.pi / 4

theorem min_y1_at_45_degrees :
  ∀ x : ℝ, y1 x ≥ y1 forty_five_degrees_in_radians := sorry

theorem no_min_max_y2 :
  ∀ val : ℝ, ∃ x : ℝ, y2 x = val := sorry

end min_y1_at_45_degrees_no_min_max_y2_l601_601335


namespace determine_intersections_l601_601284

open Real EuclideanGeometry

noncomputable def numberOfIntersections (A B : Point) (r l : ℝ) : ℕ :=
  let dist_AB := dist A B
  let radius_k2 := sqrt (l^2 + r^2)
  if dist_AB < r + radius_k2 then 2
  else if dist_AB = r + radius_k2 then 1
  else 0

theorem determine_intersections (A B : Point) (r l : ℝ) :
  ∃ (n : ℕ), 
    n = numberOfIntersections A B r l ∧ 
    ((dist A B < r + sqrt (l^2 + r^2) → n = 2) ∧ 
     (dist A B = r + sqrt (l^2 + r^2) → n = 1) ∧ 
     (dist A B > r + sqrt (l^2 + r^2) → n = 0)) := 
by
  let dist_AB := dist A B
  let radius_k2 := sqrt (l^2 + r^2)
  exists numberOfIntersections A B r l
  split
  rfl
  split <;> intro h
  specialize numberOfIntersections A B r l
  simp [dist_AB, radius_k2, numberOfIntersections] at h
  exact h
  sorry

end determine_intersections_l601_601284


namespace find_smallest_a_l601_601852

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601852


namespace maximize_profit_l601_601740

noncomputable def price_per_ton (x : ℝ) : ℝ := 24200 - (1 / 5) * x^2

noncomputable def production_cost (x : ℝ) : ℝ := 50000 + 200 * x

noncomputable def profit (x : ℝ) : ℝ := (price_per_ton x) * x - (production_cost x)

theorem maximize_profit : 
  let x := 200 in 
  x = 200 ∧ profit x = 3150000 := 
by 
  -- let x := 200
  have h1 : x = 200 := rfl,
  -- compute profit(200)
  have h2 : profit 200 = 3150000, by sorry,
  exact ⟨h1, h2⟩

end maximize_profit_l601_601740


namespace find_y_l601_601334

theorem find_y : ∃ y : ℝ, sqrt (4 - 2 * y) = 9 → y = -38.5 := by
  sorry

end find_y_l601_601334


namespace octagon_area_inscribed_in_circle_l601_601782

theorem octagon_area_inscribed_in_circle (r : ℝ) (oct_area : ℝ) :
  r = 3 → oct_area = 18 * real.sqrt 2 :=
by
  intro hr
  rw hr
  sorry

end octagon_area_inscribed_in_circle_l601_601782


namespace correct_propositions_count_l601_601299

-- Definitions for the propositions
def prop1 : Prop := true    -- A triangle can have at most three axes of symmetry (always true for equilateral and isosceles triangles).
def prop2 (a b c : ℝ) : Prop := (a^2 + b^2 ≠ c^2) → ¬((a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2))
-- prop3 is incorrect as per problem, we don't need to define it
def prop4 (A B C : ℝ) (h : A + B + C = 180) : Prop := (A < 90) ∧ (B < 90) ∨ (A < 90) ∧ (C < 90) ∨ (B < 90) ∧ (C < 90)

-- Main theorem statement
theorem correct_propositions_count (a b c A B C : ℝ) (h1 : prop1) (h2 : prop2 a b c) (h4 : prop4 A B C) :
  exactly_two_of_four h1 false false h4 :=
sorry

-- Helper definition to state that exactly two out of four propositions are true
def exactly_two_of_four (p1 p2 p3 p4 : Prop) : Prop :=
  (p1 ∧ p4 ∧ ¬p2 ∧ ¬p3) ∨ (p1 ∧ p3 ∧ ¬p2 ∧ ¬p4) ∨ (p1 ∧ p2 ∧ ¬p3 ∧ ¬p4) ∨ (p2 ∧ p3 ∧ ¬p1 ∧ ¬p4) ∨ 
  (p2 ∧ p4 ∧ ¬p1 ∧ ¬p3) ∨ (p3 ∧ p4 ∧ ¬p1 ∧ ¬p2)

end correct_propositions_count_l601_601299


namespace infinite_series_value_l601_601281

-- Define the series using necessary conditions
def series_term (n : ℕ) : ℝ :=
  (n^4 + 5 * n^2 + 8 * n + 15) / (2^n * (n^4 + 9))

def infinite_series : ℝ :=
  ∑' (n : ℕ) (hn : n ≥ 2), series_term n

-- State the problem as a theorem in Lean 4
theorem infinite_series_value :
  infinite_series = 1 / 2 + small_term :=
sorry

end infinite_series_value_l601_601281


namespace proper_fraction_cubed_numerator_triples_denominator_add_three_l601_601328

theorem proper_fraction_cubed_numerator_triples_denominator_add_three
  (a b : ℕ)
  (h1 : a < b)
  (h2 : (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b) : 
  a = 2 ∧ b = 9 :=
by
  sorry

end proper_fraction_cubed_numerator_triples_denominator_add_three_l601_601328


namespace base_distance_l601_601616

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601616


namespace convert_to_rectangular_form_l601_601288

noncomputable def polar_expression : ℂ := √3 * complex.exp (13 * real.pi * complex.I / 6)
noncomputable def rectangular_form : ℂ := (3 / 2) + (√3 / 2) * complex.I

theorem convert_to_rectangular_form :
  polar_expression = rectangular_form :=
by 
  sorry

end convert_to_rectangular_form_l601_601288


namespace stamp_collection_total_l601_601050

theorem stamp_collection_total (F O : Finset ℕ) (Neither : ℕ) 
  (hF : F.card = 90) 
  (hO : O.card = 50) 
  (hFO : (F ∩ O).card = 20) 
  (hNeither : Neither = 80): 
  F.card + O.card - (F ∩ O).card + Neither = 200 :=
by
  rw [hF, hO, hFO, hNeither]
  norm_num

end stamp_collection_total_l601_601050


namespace ladder_base_distance_l601_601607

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601607


namespace triangle_perimeter_l601_601998

variable (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]

theorem triangle_perimeter {XYZ : Triangle X Y Z}
  (h1 : XYZ.angles.XYZ = XYZ.angles.YXZ)
  (h2 : XYZ.sides.XZ = 8)
  (h3 : XYZ.sides.YZ = 6) : XYZ.perimeter = 20 :=
  sorry

end triangle_perimeter_l601_601998


namespace base_from_wall_l601_601649

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601649


namespace find_number_l601_601190

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601190


namespace find_number_divided_by_3_equals_subtracted_5_l601_601207

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601207


namespace find_a_find_point_P_l601_601405
 
noncomputable def l₁ (x y a : ℝ) : Prop := 2 * x - y + a = 0
noncomputable def l₂ (x y : ℝ) : Prop := -4 * x + 2 * y + 1 = 0
noncomputable def l₃ (x y : ℝ) : Prop := x + y - 1 = 0

noncomputable def distance_between_lines (a : ℝ) : ℝ :=
  (abs (-2 * a - 1)) / (sqrt 20)

theorem find_a (a : ℝ) (h : a > 0) (h_dist : distance_between_lines a = (7 * sqrt 5) / 10) : a = 3 :=
  sorry

noncomputable def distance_point_to_line (x y a b c : ℝ): ℝ := 
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem find_point_P (m n : ℝ) (h₁ : m > 0 ∧ n > 0)
    (h₂ : distance_point_to_line m n 2 (-1) 3 = 1/2 * distance_point_to_line m n (-4) 2 1)
    (h₃ : distance_point_to_line m n 2 (-1) 3 / distance_point_to_line m n 1 1 (-1) = sqrt 2 / sqrt 5):
    (m, n) = (1/9, 37/18) :=
  sorry

end find_a_find_point_P_l601_601405


namespace smallest_a_has_50_perfect_squares_l601_601871

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601871


namespace area_to_be_painted_l601_601111

variable (h_wall : ℕ) (l_wall : ℕ)
variable (h_window : ℕ) (l_window : ℕ)
variable (h_door : ℕ) (l_door : ℕ)

theorem area_to_be_painted :
  ∀ (h_wall : ℕ) (l_wall : ℕ) (h_window : ℕ) (l_window : ℕ) (h_door : ℕ) (l_door : ℕ),
  h_wall = 10 → l_wall = 15 →
  h_window = 3 → l_window = 5 →
  h_door = 2 → l_door = 3 →
  (h_wall * l_wall) - ((h_window * l_window) + (h_door * l_door)) = 129 :=
by
  intros
  sorry

end area_to_be_painted_l601_601111


namespace valid_numbers_count_valid_numbers_sum_l601_601231

def set_A : Set ℕ := {1, 4, 7}
def set_B : Set ℕ := {2, 5, 8}

def valid_digits : Set ℕ := set_A ∪ set_B

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (∀ d ∈ (n.digits 10), d ∈ valid_digits) ∧
  (n % 3 ≠ 0)

def count_valid_numbers : ℕ :=
  (3^4 + 3^4 + 4 * 3^3 + 4 * 3^3)

def total_sum_of_digits : ℕ :=
  405 * 36  -- we assumed that the pairs count was correctly verified

theorem valid_numbers_count :
  ∃ (n : ℕ), is_valid_number n ∧ n = 810 :=
by
  have h : 3^4 + 3^4 + 4 * 3^3 + 4 * 3^3 = 810 := by norm_num
  exact ⟨count_valid_numbers, sorry, h⟩

theorem valid_numbers_sum :
  ∃ (s : ℕ), s = 14580 :=
by
  have h : 405 * 36 = 14580 := by norm_num
  exact ⟨total_sum_of_digits, h⟩

end valid_numbers_count_valid_numbers_sum_l601_601231


namespace ladder_base_distance_l601_601694

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601694


namespace problem_distribution_count_l601_601266

theorem problem_distribution_count : 12^6 = 2985984 := 
by
  sorry

end problem_distribution_count_l601_601266


namespace ladder_base_distance_l601_601668

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601668


namespace ladder_base_distance_l601_601605

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601605


namespace smallest_a_with_50_squares_l601_601909


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601909


namespace find_general_solution_l601_601320

noncomputable def general_solution_of_differential_eq {C₁ C₂ C₃ : ℝ} (y : ℝ → ℝ) : Prop :=
  y''' + 4 * y'' + 13 * y' = 0 →
  y = λ x, C₁ + C₂ * exp (-2 * x) * cos (3 * x) + C₃ * exp (-2 * x) * sin (3 * x)

theorem find_general_solution (h : ∀ x, y''' x + 4 * y'' x + 13 * y' x = 0) :
  ∃ C₁ C₂ C₃ : ℝ, ∀ x : ℝ, y x = C₁ + C₂ * exp (-2 * x) * cos (3 * x) + C₃ * exp (-2 * x) * sin (3 * x) :=
sorry

end find_general_solution_l601_601320


namespace find_number_divided_by_3_equals_subtracted_5_l601_601210

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601210


namespace domain_of_function_l601_601125

def sqrt (x : ℝ) : ℝ := Real.sqrt x -- definition of square root
def log_base (b : ℝ) (x : ℝ) : ℝ := if 0 < x ∧ x ≠ 1 then Real.log x / Real.log b else 0 -- definition of logarithm with base b

theorem domain_of_function : 
  { x : ℝ | sqrt ((x - 1) / (2 * x)) - log_base 2 (4 - x^2) ∈ ℝ } = { x | -2 < x ∧ x < 0 } ∪ { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end domain_of_function_l601_601125


namespace sum_of_possible_x_values_l601_601524

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l601_601524


namespace sqrt_two_in_A_l601_601078

noncomputable def A : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }

theorem sqrt_two_in_A : sqrt(2) ∈ A :=
by sorry

end sqrt_two_in_A_l601_601078


namespace fireworks_display_l601_601771

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l601_601771


namespace ladder_base_distance_l601_601652

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601652


namespace probability_three_primes_in_seven_dice_l601_601795

def prime_probability (n : ℕ) : ℚ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then (2 / 5) else (3 / 5)

def primes_in_dice (dice : List ℕ) : ℚ :=
  let num_primes := dice.count (λ x => x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7)
  if num_primes = 3 then (35 : ℚ) * (2 / 5) ^ 3 * (3 / 5) ^ 4 else 0

theorem probability_three_primes_in_seven_dice :
  primes_in_dice [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7] = (9072 / 31250) :=
sorry

end probability_three_primes_in_seven_dice_l601_601795


namespace part_a_part_b_part_c_part_d_l601_601088

-- Let P_{k, l}(n) be the number of partitions of a number n into no more than k parts, each of which does not exceed l.

-- Defining the proof function types
def P (k l n : ℕ) : ℕ := sorry

-- a) P_{k, l}(n) - P_{k, l-1}(n) = P_{k-1, l}(n-l)
theorem part_a (k l n : ℕ) : P(k, l, n) - P(k, l - 1, n) = P(k - 1, l, n - l) :=
by sorry

-- b) P_{k, l}(n) - P_{k-1, l}(n) = P_{k, l-1}(n-k)
theorem part_b (k l n : ℕ) : P(k, l, n) - P(k - 1, l, n) = P(k, l - 1, n - k) :=
by sorry

-- c) P_{k, l}(n) = P_{l, k}(n)
theorem part_c (k l n : ℕ) : P(k, l, n) = P(l, k, n) :=
by sorry

-- d) P_{k, l}(n) = P_{k, l}(kl-n)
theorem part_d (k l n : ℕ) : P(k, l, n) = P(k, l, k * l - n) :=
by sorry

end part_a_part_b_part_c_part_d_l601_601088


namespace min_period_of_f_l601_601550

noncomputable def f (x : ℝ) : ℝ :=
  (sin x + cos x + (sin x)^2 * (cos x)^2) / (2 - sin (2 * x))

theorem min_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π :=
sorry

end min_period_of_f_l601_601550


namespace sum_of_solutions_l601_601521

theorem sum_of_solutions : 
  (∃ x, 3^(x^2 + 6*x + 9) = 27^(x + 3)) → (∀ x₁ x₂, (3^(x₁^2 + 6*x₁ + 9) = 27^(x₁ + 3) ∧ 3^(x₂^2 + 6*x₂ + 9) = 27^(x₂ + 3)) → x₁ + x₂ = -3) :=
sorry

end sum_of_solutions_l601_601521


namespace smallest_natural_with_50_perfect_squares_l601_601883

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601883


namespace ladder_distance_l601_601703

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601703


namespace find_number_divided_by_3_equals_subtracted_5_l601_601208

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601208


namespace keith_apples_correct_l601_601506

def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def total_apples : ℕ := 16
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_apples_correct : keith_apples = 6 := by
  -- the actual proof would go here
  sorry

end keith_apples_correct_l601_601506


namespace total_hours_worked_l601_601498

variable (A B C D E T : ℝ)

theorem total_hours_worked (hA : A = 12)
  (hB : B = 1 / 3 * A)
  (hC : C = 2 * B)
  (hD : D = 1 / 2 * E)
  (hE : E = A + 3)
  (hT : T = A + B + C + D + E) : T = 46.5 :=
by
  sorry

end total_hours_worked_l601_601498


namespace circumcenter_line_perpendicular_l601_601538

-- Given a trapezoid ABCD with AD longer than BC
variables {A B C D : Point}
variable [Trapezoid A B C D]
variable (hAD_gt_BC : AD.length > BC.length)

-- Extensions of AB and CD intersect at a point P
variable {P : Point}
variable [IntersectsABCD A B C D P]

-- There is a point Q on segment AD such that BQ = CQ
variable {Q : Point}
variable [OnSegment Q A D]
variable (hBQ_eq_CQ : BQ.length = CQ.length)

-- Define O1 and O2 as the circumcenters of triangles AQC and BQD respectively
variable {O1 O2 : Point}
variable [Circumscribed A Q C O1]
variable [Circumscribed B Q D O2]

-- Prove that the line connecting O1 and O2 is perpendicular to line PQ
theorem circumcenter_line_perpendicular {A B C D P Q O1 O2 : Point}
  (trapezoid : Trapezoid A B C D)
  (intersect : IntersectsABCD A B C D P)
  (on_segment : OnSegment Q A D)
  (eq_lengths : BQ.length = CQ.length)
  (circum1 : Circumscribed A Q C O1)
  (circum2 : Circumscribed B Q D O2)
  : Perpendicular (Line O1 O2) (Line P Q) := sorry

end circumcenter_line_perpendicular_l601_601538


namespace focus_parabola_y_eq_4x2_l601_601122

theorem focus_parabola_y_eq_4x2 :
  ∀ (x y : ℝ), y = 4 * x^2 → ∃ p : ℝ, p = 1 / 16 ∧ (x, y) = (0, p) :=
by
  intros x y hy
  use (1 / 16)
  split
  { refl }
  { suffices hr : y = 4 * x ^ 2, rw [hy, hr], norm_num }

end focus_parabola_y_eq_4x2_l601_601122


namespace tangent_to_incircle_through_D_l601_601066

-- Definitions of the problem conditions
variables {A B C D : Type} [triangle : triangle A B C] (AD : angle_bisector A B C D)
          (l : tangent_line_to_circumcircle A B C A)

-- The theorem statement
theorem tangent_to_incircle_through_D :
  ∀ (l' : line) (is_parallel : ∀ (P Q : Point), is_parallel_to l' l), tangent_to_incircle l' := sorry

end tangent_to_incircle_through_D_l601_601066


namespace fruit_eating_contest_l601_601308

variable (Sam_apple Zoe_apple Mark_apple Anne_apple : ℕ)
variable (Lily_banana John_banana Beth_banana Chris_banana : ℕ)

-- Given conditions
def apple_contest :=
  Sam_apple = 5 ∧ Zoe_apple = 2 ∧ Mark_apple = 1 ∧ Anne_apple = 3

def banana_contest :=
  Lily_banana = 6 ∧ John_banana = 4 ∧ Beth_banana = 1 ∧ Chris_banana = 2

-- Prove the differences
theorem fruit_eating_contest (h1 : apple_contest Sam_apple Zoe_apple Mark_apple Anne_apple)
                              (h2 : banana_contest Lily_banana John_banana Beth_banana Chris_banana) :
  (Sam_apple - Mark_apple = 4) ∧ (Lily_banana - Beth_banana = 5) :=
by
  simp [apple_contest, banana_contest] at h1 h2
  cases h1 with h1a h1b
  cases h1b with h1c h1d
  cases h1d with h1e h1f
  cases h2 with h2a h2b
  cases h2b with h2c h2d
  cases h2d with h2e h2f
  rw [h1a, h1c, h2a, h2c]
  split
  · exact rfl
  · exact rfl

end fruit_eating_contest_l601_601308


namespace sqrt_conjecture_l601_601508

theorem sqrt_conjecture (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + (1 / (n + 2)))) = ((n + 1) * Real.sqrt (1 / (n + 2))) :=
sorry

end sqrt_conjecture_l601_601508


namespace find_four_digit_numbers_l601_601818

def isFourDigitNumber (n : ℕ) : Prop := (1000 ≤ n) ∧ (n < 10000)

noncomputable def solveABCD (AB CD : ℕ) : ℕ := 100 * AB + CD

theorem find_four_digit_numbers :
  ∀ (AB CD : ℕ),
    isFourDigitNumber (solveABCD AB CD) →
    solveABCD AB CD = AB * CD + AB ^ 2 →
      solveABCD AB CD = 1296 ∨ solveABCD AB CD = 3468 :=
by
  intros AB CD h1 h2
  sorry

end find_four_digit_numbers_l601_601818


namespace winning_strategy_for_A_l601_601028

-- Definitions of conditions
def board_size : ℕ := 1994

inductive Move
| horizontal : Move
| vertical : Move

inductive Player
| A : Player
| B : Player

structure KnightGame :=
(start : (ℕ × ℕ))
(moves : Player → Move)
(restriction : ∀ p : Player, match moves p with
                             | Move.horizontal => p = Player.A
                             | Move.vertical   => p = Player.B
                             end)
(invalid_moves : set (ℕ × ℕ))

-- Definition based on the given problem
noncomputable def A_has_winning_strategy : Prop :=
  ∃ strategy : (ℕ × ℕ) × (Player → (ℕ × ℕ → ((ℕ × ℕ) × (ℕ × ℕ)))),
    -- Ensure the starting position and moves conform to the game rules
    strategy.1 = (1, 1) ∧
    (strategy.2 Player.A).1 = (3, 2) ∧
    (strategy.2 Player.A).2 = (1, 1) →
    ∀ moves_made : ℕ,
      ((strategy.2 Player.A moves_made).fst.fst = 1 ∧
       (strategy.2 Player.B moves_made).fst.fst = 1 →
       (strategy.2 Player.A moves_made).snd.fst = 1 ∧
       (strategy.2 Player.B moves_made).snd.fst = 1)

-- Statement to be proved
theorem winning_strategy_for_A : A_has_winning_strategy := sorry

end winning_strategy_for_A_l601_601028


namespace number_of_valid_N_l601_601279

-- Define the digit constraints
def is_valid_base4_digit (a : ℕ) := 0 ≤ a ∧ a < 4
def is_valid_base3_digit (c : ℕ) := 0 ≤ c ∧ c < 3

-- Define the main problem statement
theorem number_of_valid_N :
  let valid_N_count := (finset.range 4).sum (λ a,
    (finset.range 4).sum (λ b,
      (finset.range 3).sum (λ c,
        (finset.range 3).sum (λ d,
          if 36 * a + 9 * b - 33 * c - 11 * d = 0 then 1 else 0)))) in
  valid_N_count = 10 :=  -- replace 10 with the actual count after verification
sorry

end number_of_valid_N_l601_601279


namespace base_from_wall_l601_601647

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601647


namespace smallest_a_with_50_perfect_squares_l601_601886

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601886


namespace number_of_isosceles_triangles_l601_601997

noncomputable def angle_measure (P Q R : Type) [inner_product_space ℝ P] (PQ PR PS : ℝ) : ℝ :=
  sorry

noncomputable def isosceles_triangle {P Q R : Type} [inner_product_space ℝ P]
  (angle : ℝ) (PQ PR : ℝ) : Prop :=
  PQ = PR ∧ angle = 60

noncomputable def bisects_angle {P Q R : Type} [inner_product_space ℝ P]
  (PS PQ PR : ℝ) : Prop :=
  sorry
  
noncomputable def parallel {P Q : Type} [inner_product_space ℝ P]
  (SU PQ : ℝ) : Prop :=
  sorry

theorem number_of_isosceles_triangles {P Q R S T U : Type} [inner_product_space ℝ P]
  (PQ PR PS PQ_PR PS_bis QS_bis SU_par : ℝ) :
  isosceles_triangle 60 PQ PR → 
  bisects_angle PS PQ PR →
  bisects_angle ST QS PR →
  parallel SU PQ →
  PQ = PR →
  ∃ (n : ℕ), n = 5 :=
begin
  sorry
end

end number_of_isosceles_triangles_l601_601997


namespace ones_digit_seven_consecutive_integers_l601_601554

theorem ones_digit_seven_consecutive_integers (k : ℕ) (hk : k % 5 = 1) :
  (k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10 = 0 :=
by
  sorry

end ones_digit_seven_consecutive_integers_l601_601554


namespace ladder_distance_from_wall_l601_601710

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601710


namespace base_distance_l601_601617

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601617


namespace fourth_vertex_of_tetrahedron_l601_601251

noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p
  let (x2, y2, z2) := q
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem fourth_vertex_of_tetrahedron (x y z : ℤ) :
  distance (1,0,3) (5,1,2) = distance (1,0,3) (x,y,z) ∧
  distance (1,0,3) (5,1,2) = distance (5,1,2) (x,y,z) ∧
  distance (1,0,3) (5,1,2) = distance (4,0,6) (x,y,z) →
  (x, y, z) = (7, -8, 12) := 
by
  -- Proof of the theorem would go here
  sorry

end fourth_vertex_of_tetrahedron_l601_601251


namespace time_to_raise_object_l601_601253

-- Define conditions
def radius : ℝ := 50 -- in cm
def speed : ℝ := 4 -- in revolutions per minute
def raise_height : ℝ := 100 -- in cm

-- Constants involved
def circumference := 2 * Real.pi * radius
def seconds_per_minute : ℝ := 60
def revolutions_per_second : ℝ := speed / seconds_per_minute
def time_per_revolution : ℝ := 1 / revolutions_per_second

-- The final proof statement
theorem time_to_raise_object : time_per_revolution * (raise_height / circumference) = 15 / Real.pi :=
by sorry

end time_to_raise_object_l601_601253


namespace ladder_distance_from_wall_l601_601719

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601719


namespace intersection_points_count_l601_601805

def fractional_part (x : ℝ) : ℝ := x - floor x

def circle_eq (x y : ℝ) : Prop := (fractional_part x)^2 + (y - 1)^2 = fractional_part x

def line_eq (x y : ℝ) : Prop := y = (1/3) * x

theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), 
    (∀ p ∈ points, circle_eq p.1 p.2 ∧ line_eq p.1 p.2) ∧ points.card = 14 :=
sorry

end intersection_points_count_l601_601805


namespace inverse_B_squared_l601_601493

-- Defining the inverse matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 0, 1]

-- Theorem to prove that the inverse of B^2 is a specific matrix
theorem inverse_B_squared :
  (B_inv * B_inv) = !![9, -6; 0, 1] :=
  by sorry


end inverse_B_squared_l601_601493


namespace magnitude_expression_l601_601437

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601437


namespace smallest_natural_number_with_50_squares_in_interval_l601_601862

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601862


namespace phi_tau_equality_l601_601502

def is_coprime (a b : ℕ) : Prop := gcd a b = 1

def phi (n : ℕ) : ℕ :=
  finset.card { m | m < n ∧ is_coprime m n }

def tau (n : ℕ) : ℕ :=
  finset.card { d | d ∣ n }

def has_two_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p.prime ∧ q.prime ∧ p ≠ q ∧ n = p^(nat.prime_exp n p) * q^(nat.prime_exp n q)

theorem phi_tau_equality {n : ℕ} (h1 : n > 0) (h2 : has_two_distinct_prime_factors n) :
  phi (tau n) = tau (phi n) → ∃ t r : ℕ, nat.prime r ∧ n = 2^(t-1) * 3^(r-1) :=
sorry

end phi_tau_equality_l601_601502


namespace smallest_a_has_50_perfect_squares_l601_601877

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601877


namespace ladder_distance_l601_601706

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601706


namespace negation_of_implication_l601_601551

-- Define the propositions p and q as boolean values
variables p q : Prop 

-- State the theorem to assert that the negation of (p → q) is (¬ p → ¬ q)
theorem negation_of_implication (p q : Prop) : ¬ (p → q) ↔ (¬ p ∨ ¬ q) :=
sorry

end negation_of_implication_l601_601551


namespace x_of_x35x_div_by_18_l601_601333

theorem x_of_x35x_div_by_18 (x : ℕ) (h₁ : 18 = 2 * 9) (h₂ : (2 * x + 8) % 9 = 0) (h₃ : ∃ k : ℕ, x = 2 * k) : x = 8 :=
sorry

end x_of_x35x_div_by_18_l601_601333


namespace AP_bisects_CT_l601_601361

open_locale classical

variables (P B C T O A S R : Type) [AddCommGroup A]
variables (circle : set A) (diameter : A → A → A) (on_tangent : A → A → Prop)
variables (tangent : A → A → Prop) (perpendicular : A → A → Prop) (project_onto : A → A → A)

-- Definitions from conditions
def cntr_circle (c : A) := c = O
def diameter_of_circle (a b : A) := diameter A B = diameter a b
def point_on_tangent (p b : A) := on_tangent p b
def tangent_from_point (p c : A) := tangent p c
def perpendicular_projection (c t : A) := project_onto c T = t

-- Definitions of points S and R
def intersection_AP_CT := S
def intersection_PB_AC := R

-- Proof statement
theorem AP_bisects_CT (h1 : cntr_circle O)
    (h2 : diameter_of_circle A B)
    (h3 : point_on_tangent P B)
    (h4 : tangent_from_point P C)
    (h5 : perpendicular_projection C T)
    (h6 : intersection_AP_CT = S)
    (h7 : intersection_PB_AC = R) : 
    is_midpoint S C T :=
sorry

end AP_bisects_CT_l601_601361


namespace ladder_base_distance_l601_601655

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601655


namespace least_number_to_subtract_l601_601226

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (k : ℕ) (hk : 42398 % 15 = k) : k = 8 :=
by
  sorry

end least_number_to_subtract_l601_601226


namespace ladder_base_distance_l601_601665

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601665


namespace lifting_ratio_after_gain_l601_601793

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l601_601793


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601184

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601184


namespace inequality_solution_l601_601942

def f (x : ℝ) : ℝ := 2016^x + log 2016 (sqrt (x^2 + 1) + x) - 2016^(-x) + 2

theorem inequality_solution (x : ℝ) : f (3*x + 1) + f x > 4 ↔ -1/4 < x := 
sorry

end inequality_solution_l601_601942


namespace range_of_m_l601_601397

-- Given function f
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Given function g
def g (x m : ℝ) : ℝ := f x - m

-- Condition: g has three zeros in the interval [-3/2, 3]
def has_three_zeros_in_interval (m : ℝ) : Prop :=
  (∃ ξ1 ξ2 ξ3 : ℝ, -3/2 ≤ ξ1 ∧ ξ1 < ξ2 ∧ ξ2 < ξ3 ∧ ξ3 ≤ 3 ∧ g ξ1 m = 0 ∧ g ξ2 m = 0 ∧ g ξ3 m = 0)

-- Proof problem: Finding the range of m
theorem range_of_m :
  {m : ℝ | has_three_zeros_in_interval m} = set.Ico (9 / 8) 2 :=
by sorry

end range_of_m_l601_601397


namespace smallest_a_has_50_perfect_squares_l601_601874

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601874


namespace john_mission_total_time_l601_601075

theorem john_mission_total_time :
  let first_mission := 5 + 0.60 * 5 in
  let second_mission := first_mission * 0.50 in
  let third_mission := 
    max (second_mission * 2) (first_mission * (1 - 0.20)) in
  let fourth_mission := 3 + 0.50 * third_mission in
  first_mission + second_mission + third_mission + fourth_mission = 24.6 := 
by 
  sorry

end john_mission_total_time_l601_601075


namespace a_n_formula_Sn_formula_l601_601920

-- Define the sequences and conditions
def a (n : ℕ) : ℕ := 2*n - 1
def b (n : ℕ) : ℕ := 2^(n-1)

-- Define the parameters based on the given conditions
axiom a_conditions : a 5 = 9 ∧ a 7 = 13

-- Define the sum of the first n terms of the sequence {a_n + b_n}
def S (n : ℕ) : ℕ :=
  ∑ i in finset.range n, (a (i + 1) + b (i + 1))

-- Theorem statements to be proved
theorem a_n_formula : ∀ n : ℕ, n > 0 → a n = 2*n - 1 :=
by sorry

theorem Sn_formula : ∀ n : ℕ, S n = n^2 + 2^n - 1 :=
by sorry

end a_n_formula_Sn_formula_l601_601920


namespace ladder_base_distance_l601_601606

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601606


namespace complex_exp_identity_l601_601419

theorem complex_exp_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_exp_identity_l601_601419


namespace arithmetic_sequence_sum_neq_l601_601923

theorem arithmetic_sequence_sum_neq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
    (h_arith : ∀ n, a (n + 1) = a n + d)
    (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
    (h_abs_eq : abs (a 3) = abs (a 9))
    (h_d_neg : d < 0) : S 5 ≠ S 6 := by
  sorry

end arithmetic_sequence_sum_neq_l601_601923


namespace trig_identity_l601_601600

theorem trig_identity : 
  sin (10 * real.pi / 180) * cos (70 * real.pi / 180) - cos (10 * real.pi / 180) * cos (20 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_l601_601600


namespace ladder_base_distance_l601_601726

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601726


namespace pebbles_distribution_l601_601568

theorem pebbles_distribution (n : ℕ) : 
  (let first_share := n / 2 + 1 in
   let remaining_after_first := n - first_share in
   let second_share := remaining_after_first / 3 in
   let third_share := 2 * second_share in
   remaining_after_first - second_share = third_share)
    → False := 
by
  sorry

end pebbles_distribution_l601_601568


namespace ladder_distance_l601_601709

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601709


namespace line_intersects_xz_plane_at_l601_601327

variables (p1 p2 p3 : ℝ × ℝ × ℝ)

def direction_vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def parametric_line (p : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + t * d.1, p.2 + t * d.2, p.3 + t * d.3)

theorem line_intersects_xz_plane_at :
  ∃ t, parametric_line (2, -1, 3) (1, -1, 1) t = (1, 0, 2) :=
by
  use -1
  sorry

end line_intersects_xz_plane_at_l601_601327


namespace sum_of_possible_x_values_l601_601523

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l601_601523


namespace ladder_distance_from_wall_l601_601714

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601714


namespace smallest_a_with_50_perfect_squares_l601_601897

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601897


namespace smallest_a_with_50_perfect_squares_l601_601900

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601900


namespace tan_Y_in_right_triangle_l601_601826

theorem tan_Y_in_right_triangle:
  ∀ (XY YZ XZ : ℝ),
  XY = 30 →
  YZ = 37 →
  XZ = Real.sqrt (YZ^2 - XY^2) →
  tan Y = XZ / XY :=
by sorry

end tan_Y_in_right_triangle_l601_601826


namespace full_price_tickets_count_l601_601595

def num_tickets_reduced := 5400
def total_tickets := 25200
def num_tickets_full := 5 * num_tickets_reduced

theorem full_price_tickets_count :
  num_tickets_reduced + num_tickets_full = total_tickets → num_tickets_full = 27000 :=
by
  sorry

end full_price_tickets_count_l601_601595


namespace part1_part2_l601_601928

variables (x y : ℝ)
def A : ℝ := x^2 + x * y + 3 * y
def B : ℝ := x^2 - x * y
def C : ℝ := -3 * x^2 + 3 * x * y + 6 * y

theorem part1 : 3 * A - B = 2 * x^2 + 4 * x * y + 9 * y := by
  sorry

theorem part2 (h : A - (1 / 3) * C = 2 * x^2 + y) : A + (1 / 3) * C = 2 * x * y + 5 * y := by
  sorry

end part1_part2_l601_601928


namespace probability_after_second_shot_probability_distribution_and_expectation_l601_601213

section AirplaneShooting

-- Conditions
def prob_hit_I := 1 / 6
def prob_hit_II := 1 / 3
def prob_hit_III := 1 / 2

-- Function that calculates P(X = 2)
def prob_shoot_down_after_second : ℚ := (5 / 6) * prob_hit_I + (prob_hit_II ^ 2)

-- Function that calculates the X
def prob_X : ℕ → ℚ
| 1 => prob_hit_I
| 2 => prob_shoot_down_after_second
| 3 => 1 / 3
| 4 => 1 / 4
| _ => 0 -- Assumption that X > 4 has zero probability

-- Mathematical expectation E[X]
def expectation_X : ℚ := 1 * prob_X 1 + 2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4

theorem probability_after_second_shot :
  prob_shoot_down_after_second = 1 / 4 := sorry

theorem probability_distribution_and_expectation :
  (∀ x ∈ [1, 2, 3, 4], prob_X x = [1 / 6, 1 / 4, 1 / 3, 1 / 4].nth (x - 1).get_or_else 0) ∧
  expectation_X = 8 / 3 := sorry

end AirplaneShooting

end probability_after_second_shot_probability_distribution_and_expectation_l601_601213


namespace sum_real_imag_parts_eq_l601_601967

noncomputable def z (a b : ℂ) : ℂ := a / b

theorem sum_real_imag_parts_eq (z : ℂ) (h : z * (2 + I) = 2 * I - 1) : 
  (z.re + z.im) = 1 / 5 :=
sorry

end sum_real_imag_parts_eq_l601_601967


namespace line_slope_abs_value_l601_601576

theorem line_slope_abs_value :
  ∃ a : ℝ, 
  abs a = 33 / 15 ∧ 
  (∀ x y : ℝ, (0, 20) = (x, y)) ∧ 
  (∀ x y : ℝ, (7, 13) = (x, y)) ∧ 
  (∀ x y : ℝ, (4, 0) = (x, y)) ∧ 
  (∀ r : ℝ, r = 4) ∧ 
  (line_through (4, 0) passes_through_circles_dividing_area r (0, 20), 
                                            (7, 13)) :=
by
  sorry

end line_slope_abs_value_l601_601576


namespace part_I_part_II_l601_601000

-- Condition definitions:
def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

-- Part I: Prove m = 1
theorem part_I (m : ℝ) : (∀ x : ℝ, f (x + 2) m ≥ 0) ↔ m = 1 :=
by
  sorry

-- Part II: Prove a + 2b + 3c ≥ 9
theorem part_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end part_I_part_II_l601_601000


namespace ladder_distance_l601_601707

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601707


namespace last_two_digits_of_product_squared_l601_601597

def mod_100 (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_product_squared :
  mod_100 ((301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2) = 76 := 
by
  sorry

end last_two_digits_of_product_squared_l601_601597


namespace order_of_f_values_l601_601355

def f (x: ℝ) : ℝ := 2 ^ x - 2 ^ (-x)
def a : ℝ := (7 / 9) ^ (-1 / 4)
def b : ℝ := (9 / 7) ^ (1 / 5)
def c : ℝ := Real.log 7 / Real.log 9

theorem order_of_f_values : f(c) < f(a) ∧ f(a) < f(b) :=
by {
  -- Proof to be filled in here
  sorry
}

end order_of_f_values_l601_601355


namespace smallest_natural_with_50_perfect_squares_l601_601885

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601885


namespace sum_of_powers_of_i_l601_601150

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end sum_of_powers_of_i_l601_601150


namespace probability_first_ace_equal_l601_601034

theorem probability_first_ace_equal (num_cards : ℕ) (num_aces : ℕ) (num_players : ℕ)
  (h1 : num_cards = 32) (h2 : num_aces = 4) (h3 : num_players = 4) :
  ∀ player : ℕ, player ∈ {1, 2, 3, 4} → (∃ positions : list ℕ, (∀ n ∈ positions, n % num_players = player - 1)) → 
  (positions.length = 8) →
  let P := 1 / 8 in
  P = 1 / num_players :=
begin
  sorry
end

end probability_first_ace_equal_l601_601034


namespace base_from_wall_l601_601642

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601642


namespace ladder_base_distance_l601_601602

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601602


namespace andy_wrong_questions_l601_601274

-- Definitions of variables and conditions
variables (a b c d : ℕ) 

-- Given conditions
def condition1 := a + b = c + d
def condition2 := a + d = b + c + 6
def condition3 := c = 7

-- Theorem stating the problem and the expected result
theorem andy_wrong_questions : condition1 ∧ condition2 ∧ condition3 → a = 10 :=
by {
  intros,
  sorry
}

end andy_wrong_questions_l601_601274


namespace find_number_l601_601193

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601193


namespace selling_price_per_set_l601_601516

theorem selling_price_per_set (P : ℝ) 
  (initial_outlay : ℝ := 10000)
  (cost_per_set : ℝ := 20)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  P = 50 :=
by
  let revenue := num_sets * P
  let manufacturing_cost := initial_outlay + (num_sets * cost_per_set)
  have h : profit = revenue - manufacturing_cost, by sorry
  have eq1 : profit = revenue - manufacturing_cost, from h
  -- solving for P
  have eq2 : 5000 = 500 * P - (10000 + 500 * 20), by sorry
  have eq3 : 5000 = 500 * P - 20000, by sorry
  have eq4 : 25000 = 500 * P, by sorry
  have eq5 : P = 50, by sorry
  exact eq5

end selling_price_per_set_l601_601516


namespace greatest_possible_remainder_l601_601411

-- State the problem formally in Lean 4 statement

theorem greatest_possible_remainder (y : ℕ) : 
  ∃ r, r < 11 ∧ r = y % 11 ∧ r = 10 :=
begin
  sorry
end

end greatest_possible_remainder_l601_601411


namespace bugs_meeting_time_l601_601164

/-- Two circles with radii 7 inches and 3 inches are tangent at a point P. 
Two bugs start crawling at the same time from point P, one along the larger circle 
at 4π inches per minute, and the other along the smaller circle at 3π inches per minute. 
Prove they will meet again after 14 minutes and determine how far each has traveled.

The bug on the larger circle will have traveled 28π inches.
The bug on the smaller circle will have traveled 42π inches.
-/
theorem bugs_meeting_time
  (r₁ r₂ : ℝ) (v₁ v₂ : ℝ)
  (h₁ : r₁ = 7) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) :
  ∃ t d₁ d₂, t = 14 ∧ d₁ = 28 * Real.pi ∧ d₂ = 42 * Real.pi := by
  sorry

end bugs_meeting_time_l601_601164


namespace cartesian_equation_of_circle_tangent_line_slope_l601_601402

theorem cartesian_equation_of_circle (ρ : ℝ) (hρ : ρ = 2) : ∀ x y : ℝ, x^2 + y^2 = 4 :=
by
  -- The statement of the theorem
  sorry

theorem tangent_line_slope (k : ℝ) : kx + y + 3 = 0 ∧ x^2 + y^2 = 4 → k = ±(sqrt(5)/2) :=
by
  -- The statement of the theorem
  sorry

end cartesian_equation_of_circle_tangent_line_slope_l601_601402


namespace coeff_x3_in_expansion_l601_601534

theorem coeff_x3_in_expansion :
  let f : ℕ → ℂ := λ n, if n = 3 then 30 else 0 in
  (∀ (x : ℂ), x ≠ 0 → (2 * x - 1) * (x⁻¹ + x)^6 = ∑ k, f k * x^k) :=
by
  sorry

end coeff_x3_in_expansion_l601_601534


namespace equality_condition_l601_601305

theorem equality_condition (a b c : ℝ) (h : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
    sqrt (a^2 + b^2 + c^2) = a + b + c ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ c = 0) ∨ (b = 0 ∧ c = 0) :=
by
  sorry

end equality_condition_l601_601305


namespace constant_term_expansion_l601_601830

theorem constant_term_expansion :
  (∃ t : ℤ, is_constant_term (λ (x : ℂ), (x - 2 / x) ^ 6) t ∧ t = -160) :=
by
  sorry

end constant_term_expansion_l601_601830


namespace find_lambda_and_projection_l601_601410

variables (λ : ℝ)
variables (a b : ℝ × ℝ)

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2 in
  let v_norm_sq := v.1 * v.1 + v.2 * v.2 in
  let scalar := dot_product / v_norm_sq in
  (scalar * v.1, scalar * v.2)

theorem find_lambda_and_projection
  (h_perpendicular : perpendicular (-2, λ) (1, 1)) :
  λ = 2 ∧ proj ((-2, λ) - (1, 1)) (1, 1) = (-1, -1) :=
begin
  sorry
end

end find_lambda_and_projection_l601_601410


namespace carina_coffee_l601_601800

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee_l601_601800


namespace magnitude_complex_expression_l601_601442

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601442


namespace part_a_l601_601599

noncomputable def omega := set.Ioo (0: ℝ) 1
def sigma_algebra := borel omega
def P: measure ℝ := volume.restrict omega
def A (n : ℕ) : set ℝ := set.Ioo 0 (1 / n : ℝ)
def occurs_infinitely_often : set ℝ := {ω | ∀ n : ℕ, ∃ m ≥ n, ω ∈ A m}

theorem part_a : P occurs_infinitely_often = 0 := sorry

end part_a_l601_601599


namespace base_from_wall_l601_601645

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601645


namespace max_sum_of_squares_l601_601089

theorem max_sum_of_squares :
  ∃ (x : Fin 10 → ℤ),
    (∀ i, x i ≠ 0) ∧
    (∀ i, -1 ≤ x i ∧ x i ≤ 2) ∧
    (∑ i, x i = 11) ∧
    (∑ i, x i * x i = 31) :=
by
  sorry

end max_sum_of_squares_l601_601089


namespace cost_of_one_of_the_shirts_l601_601337

theorem cost_of_one_of_the_shirts
    (total_cost : ℕ) 
    (cost_two_shirts : ℕ) 
    (num_equal_shirts : ℕ) 
    (cost_of_shirt : ℕ) :
    total_cost = 85 → 
    cost_two_shirts = 20 → 
    num_equal_shirts = 3 → 
    cost_of_shirt = (total_cost - 2 * cost_two_shirts) / num_equal_shirts → 
    cost_of_shirt = 15 :=
by
  intros
  sorry

end cost_of_one_of_the_shirts_l601_601337


namespace limit_S_n_l601_601129

def a (n : ℕ) := 1 / (n * (n + 1))

def S (n : ℕ) := ∑ i in Finset.range (n + 1), a i

theorem limit_S_n : filter.tendsto S filter.at_top (nhds 1) := sorry

end limit_S_n_l601_601129


namespace jordan_width_l601_601801

-- Definitions based on conditions
def area_of_carols_rectangle : ℝ := 15 * 20
def jordan_length_feet : ℝ := 6
def feet_to_inches (feet: ℝ) : ℝ := feet * 12
def jordan_length_inches : ℝ := feet_to_inches jordan_length_feet

-- Main statement
theorem jordan_width :
  ∃ w : ℝ, w = 300 / 72 :=
sorry

end jordan_width_l601_601801


namespace stairs_ways_to_10_l601_601158

def a : ℕ → ℕ
| 0     := 0   -- By convention, 0 ways to reach step 0
| 1     := 1
| 2     := 2
| (n+3) := a n + a (n + 1) + a (n + 2)

theorem stairs_ways_to_10 :
  a 10 = 89 :=
sorry

end stairs_ways_to_10_l601_601158


namespace complete_work_together_in_days_l601_601218

noncomputable def a_days := 16
noncomputable def b_days := 6
noncomputable def c_days := 12

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem complete_work_together_in_days :
  let combined_rate := (work_rate a_days) + (work_rate b_days) + (work_rate c_days)
  let days_to_complete := 1 / combined_rate
  days_to_complete = 3.2 :=
  sorry

end complete_work_together_in_days_l601_601218


namespace find_number_eq_seven_point_five_l601_601199

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601199


namespace decrease_percent_in_revenue_l601_601596

theorem decrease_percent_in_revenue
  (T C : ℝ)
  (h_pos_T : 0 < T)
  (h_pos_C : 0 < C)
  (h_new_tax : T_new = 0.80 * T)
  (h_new_consumption : C_new = 1.20 * C) :
  let original_revenue := T * C
  let new_revenue := 0.80 * T * 1.20 * C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 4 := by
sorry

end decrease_percent_in_revenue_l601_601596


namespace convert_to_rectangular_form_l601_601286

noncomputable def polar_expression : ℂ := √3 * complex.exp (13 * real.pi * complex.I / 6)
noncomputable def rectangular_form : ℂ := (3 / 2) + (√3 / 2) * complex.I

theorem convert_to_rectangular_form :
  polar_expression = rectangular_form :=
by 
  sorry

end convert_to_rectangular_form_l601_601286


namespace andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l601_601273

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that André wins the book is 1/4. -/
theorem andre_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let probability := (black_balls : ℚ) / total_balls
  probability = 1 / 4 := 
by 
  sorry

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that Dalva wins the book is 1/4. -/
theorem dalva_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let andre_white := (3 / 4 : ℚ)
  let bianca_white := (2 / 3 : ℚ)
  let carlos_white := (1 / 2 : ℚ)
  let probability := andre_white * bianca_white * carlos_white * (black_balls / (total_balls - 3))
  probability = 1 / 4 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that André wins the book is 5/14. -/
theorem andre_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_first_black := (black_balls : ℚ) / total_balls
  let andre_fifth_black := (((6 / 8 : ℚ) * (5 / 7 : ℚ) * (4 / 6 : ℚ) * (3 / 5 : ℚ)) * black_balls / (total_balls - 4))
  let probability := andre_first_black + andre_fifth_black
  probability = 5 / 14 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that Dalva wins the book is 1/7. -/
theorem dalva_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_white := (6 / 8 : ℚ)
  let bianca_white := (5 / 7 : ℚ)
  let carlos_white := (4 / 6 : ℚ)
  let dalva_black := (black_balls / (total_balls - 3))
  let probability := andre_white * bianca_white * carlos_white * dalva_black
  probability = 1 / 7 := 
by 
  sorry

end andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l601_601273


namespace max_digit_sum_24_hour_format_l601_601242

theorem max_digit_sum_24_hour_format : 
  ∃ sum, (sum = 24) ∧ (∀ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → sum_digits h + sum_digits m ≤ sum) :=
sorry

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

end max_digit_sum_24_hour_format_l601_601242


namespace Aarti_work_days_l601_601262

theorem Aarti_work_days (x : ℕ) : (3 * x = 24) → x = 8 := by
  intro h
  linarith

end Aarti_work_days_l601_601262


namespace range_of_sum_of_zeros_l601_601392

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x else 1 - x / 2

noncomputable def F (x : ℝ) (m : ℝ) : ℝ :=
  f (f x + 1) + m

def has_zeros (F : ℝ → ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ m = 0 ∧ F x₂ m = 0

theorem range_of_sum_of_zeros (m : ℝ) :
  has_zeros F m →
  ∃ (x₁ x₂ : ℝ), F x₁ m = 0 ∧ F x₂ m = 0 ∧ (x₁ + x₂) ≥ 4 - 2 * Real.log 2 := sorry

end range_of_sum_of_zeros_l601_601392


namespace relationship_P_Q_l601_601491

open Real

-- Define the conditions of the problem
def s (a b c : ℝ) : ℝ := (a + b + c) / 2

def area_isosceles (a b c : ℝ) : ℝ := sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

def area_right_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Define the specific triangles
def area_P : ℝ := area_isosceles 17 17 16
def area_Q : ℝ := area_right_triangle 15 20

-- Theorem to prove the relationship
theorem relationship_P_Q : area_P = (4/5) * area_Q := by
  sorry

end relationship_P_Q_l601_601491


namespace fireworks_display_l601_601772

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l601_601772


namespace ladder_base_distance_l601_601724

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601724


namespace collinearity_condition_l601_601533

-- Definitions of the given conditions

structure Circle (α : Type) where
  center : α
  radius : ℝ

variables {α : Type} [EuclideanGeometry α]

-- Given conditions
variable (ω : Circle α)
variable (O T : α)
variable (B C B' C' K H K' H' : α)

-- Supplementary properties
variable (isTangent : TangentFromPoint T ω B TC)
variable (secondIntersectionB : SecondIntersectionPoint O B ω B')
variable (secondIntersectionC : SecondIntersectionPoint O C ω C')
variable (angleBisectorConditionK : OnAngleBisector BC O K')
variable (perpendicularK : ⟂ (LineSegment K K') BC)
variable (angleBisectorConditionH : OnAngleBisector CB O H')
variable (perpendicularH : ⟂ (LineSegment H H') BC)

theorem collinearity_condition :
  Collinear [K, H', B'] ↔ Collinear [H, K', C'] :=
sorry

end collinearity_condition_l601_601533


namespace number_of_bad_arrangements_eq_2_l601_601139

def is_bad_arrangement (arr : List ℕ) : Prop :=
  ∀ n : ℕ, n ∈ Finset.range 21 → ∃ (subarr : List ℕ), subarr ≠ [] 
  ∧ (subarr.sum = n ∧ (∀ i j, List.take (j - i) (List.drop i arr) = subarr → List.drop i arr ≠ subarr))

theorem number_of_bad_arrangements_eq_2 : 
  ∃ bad_arrangements : Finset (List ℕ), bad_arrangements.card = 2 ∧ 
  (∀ p, p ∈ bad_arrangements → is_bad_arrangement p) :=
sorry

end number_of_bad_arrangements_eq_2_l601_601139


namespace frac_addition_l601_601585

theorem frac_addition :
  (3 / 5) + (2 / 15) = 11 / 15 :=
sorry

end frac_addition_l601_601585


namespace min_max_f_l601_601831

theorem min_max_f (a b x y z t : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hxz : x + z = 1) (hyt : y + t = 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hz : 0 ≤ z) (ht : 0 ≤ t) :
  1 ≤ ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ∧
  ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ≤ 2 :=
sorry

end min_max_f_l601_601831


namespace cell_division_relationship_l601_601588

noncomputable def number_of_cells_after_divisions (x : ℕ) : ℕ :=
  2^x

theorem cell_division_relationship (x : ℕ) : 
  number_of_cells_after_divisions x = 2^x := 
by 
  sorry

end cell_division_relationship_l601_601588


namespace slope_angle_range_l601_601364

noncomputable def curve (x : ℝ) : ℝ := x^3 - real.sqrt 3 * x + 3

noncomputable def slope_angle_of_tangent (x : ℝ) : ℝ := real.arctan (3 * x^2 - real.sqrt 3)

theorem slope_angle_range : 
  (∀ x : ℝ, 0 ≤ slope_angle_of_tangent x ∧ slope_angle_of_tangent x < π/2) ∨ 
  (∀ x : ℝ, 2 * π / 3 ≤ slope_angle_of_tangent x ∧ slope_angle_of_tangent x < π) :=
sorry

end slope_angle_range_l601_601364


namespace probability_at_least_one_common_element_l601_601077

-- Definitions based on the conditions (set and subsets)
def U : Finset ℕ := {1, 2, 3, 4}

-- Cardinality of the subsets (to represent uniform selection of subsets)
def all_subsets : Finset (Finset ℕ) := U.powerset

-- Definition of the total number of possible pairs of subsets
def total_pairs : ℕ := all_subsets.card * all_subsets.card

-- Probability calculation
noncomputable def probability_common_element : ℚ :=
  1 - (all_subsets.sum (λ A, 2^(4 - A.card) * (1 / total_pairs)))

-- Theorem statement
theorem probability_at_least_one_common_element :
  probability_common_element = 175 / 256 :=
sorry

end probability_at_least_one_common_element_l601_601077


namespace find_number_l601_601186

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601186


namespace find_a_l601_601019

noncomputable def function_decreasing_interval (f : ℝ → ℝ) (a : ℝ) := ∀ x, x ≤ 2 → (deriv (λ x, x^2 + 2 * a * x + 1) x) < 0

theorem find_a (f : ℝ → ℝ) (a : ℝ) (h : function_decreasing_interval f a) : a = -2 :=
by
  -- Proof goes here.
  sorry

end find_a_l601_601019


namespace max_value_of_g_l601_601127

def g (n : ℕ) : ℕ :=
  if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ m, ∀ n, g n ≤ m ∧ (∃ k, g k = m) :=
by
  use 29
  sorry

end max_value_of_g_l601_601127


namespace watermelon_cost_is_100_rubles_l601_601741

theorem watermelon_cost_is_100_rubles :
  (∀ (x y k m n : ℕ) (a b : ℝ),
    x + y = k →
    n * a = m * b →
    n * a + m * b = 24000 →
    n = 120 →
    m = 30 →
    k = 150 →
    a = 100) :=
by
  intros x y k m n a b
  intros h1 h2 h3 h4 h5 h6
  have h7 : 120 * a = 30 * b, from h2
  have h8 : 120 * a + 30 * b = 24000, from h3
  have h9 : 120 * a = 12000, from sorry
  have h10 : a = 100, from sorry
  exact h10

end watermelon_cost_is_100_rubles_l601_601741


namespace sequence_general_term_l601_601541

-- Define the sequence a n according to the general term formula
def sequence (n : ℕ) : ℚ := (2 * n + 1) / (n + 1)

-- Define the expected values for specific terms in the sequence
def expected_vals : list ℚ := [3/2, 5/3, 7/4, 9/5]

-- The main theorem statement checking the sequence's correctness up to the first four terms
theorem sequence_general_term :
  (sequence 1 = 3/2) ∧
  (sequence 2 = 5/3) ∧
  (sequence 3 = 7/4) ∧
  (sequence 4 = 9/5) := 
by
  -- Must fill in the proof later.
  -- This is a placeholder to ensure the Lean code builds successfully.
  sorry

end sequence_general_term_l601_601541


namespace berries_from_seventh_bush_l601_601504

theorem berries_from_seventh_bush :
  ∃ n1 n2 n3 n4 n5 n6 n7 : ℕ,
    n1 = 2 ∧
    n2 = 3 ∧
    n3 = (n1 + n2) * 3 ∧
    n4 = (n2 + n3) * 4 ∧
    n5 = (n3 + n4) * 5 ∧
    n6 = (n4 + n5) * 6 ∧
    n7 = (n5 + n6) * 7 ∧
    n7 = 24339 := 
by
  use [2, 3, 15, 72, 435, 3042, 24339]
  repeat { split }
  all_goals { sorry }

end berries_from_seventh_bush_l601_601504


namespace smallest_a_has_50_perfect_squares_l601_601854

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601854


namespace range_of_a_for_monotonically_decreasing_f_l601_601968

open Real

noncomputable def f (a : ℝ) (x : ℝ) := x^2 + 2 * x + a * log x

theorem range_of_a_for_monotonically_decreasing_f :
  (∀ x ∈ Ioo 0 1, deriv (f a) x ≤ 0) → a ≤ -4 :=
by
  assume h : ∀ x ∈ Ioo 0 1, deriv (f a) x ≤ 0
  -- Proof to be filled in
  sorry

end range_of_a_for_monotonically_decreasing_f_l601_601968


namespace verify_squaring_method_l601_601526

theorem verify_squaring_method (x : ℝ) :
  ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 :=
by
  sorry

end verify_squaring_method_l601_601526


namespace smallest_a_with_50_perfect_squares_l601_601887

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601887


namespace prove_cos_half_angle_l601_601352

noncomputable def given_conditions (α : ℝ) : Prop := 
  sin α = 4 / 5 ∧ 0 < α ∧ α < π / 2

theorem prove_cos_half_angle (α : ℝ) (h : given_conditions α) : cos (α / 2) = 2 * sqrt 5 / 5 := 
by sorry

end prove_cos_half_angle_l601_601352


namespace maximum_area_ΔABC_exists_l601_601007

/-
Problem statement:
Given a triangle ΔABC, where AB = 2 and AC² + BC² = 10,
prove that the maximum possible area of ΔABC is 2.
-/

noncomputable def max_area_ΔABC (A B C : ℝ × ℝ) : Prop :=
let AB := ((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt in
let AC := ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt in
let BC := ((C.1 - B.1)^2 + (C.2 - B.2)^2).sqrt in
AB = 2 ∧ AC^2 + BC^2 = 10 ∧ 
(∃ (K : ℝ), K ≥ 0 ∧ (AB * K) = 2)

theorem maximum_area_ΔABC_exists (A B C : ℝ × ℝ) :
  max_area_ΔABC A B C → 
  ∃ K, K ≤ 2 ∧ K = (2 * (B.1 - A.1).abs * (C.2 - 0).abs)/2 := 
sorry

end maximum_area_ΔABC_exists_l601_601007


namespace arrangement_count_l601_601563

-- Definitions corresponding to the given problem conditions
def numMathBooks : Nat := 3
def numPhysicsBooks : Nat := 2
def numChemistryBooks : Nat := 1
def totalArrangements : Nat := 2592

-- Statement of the theorem
theorem arrangement_count :
  ∃ (numM numP numC : Nat), 
    numM = 3 ∧ 
    numP = 2 ∧ 
    numC = 1 ∧ 
    (numM + numP + numC = 6) ∧ 
    allMathBooksAdjacent ∧ 
    physicsBooksNonAdjacent → 
    totalArrangements = 2592 :=
by
  sorry

end arrangement_count_l601_601563


namespace jack_jill_meet_distance_l601_601069

theorem jack_jill_meet_distance : 
  ∀ (total_distance : ℝ) (uphill_distance : ℝ) (headstart : ℝ) 
  (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ), 
  total_distance = 12 → 
  uphill_distance = 6 → 
  headstart = 1 / 4 → 
  jack_speed_up = 12 → 
  jack_speed_down = 18 → 
  jill_speed_up = 14 → 
  jill_speed_down = 20 → 
  ∃ meet_position : ℝ, meet_position = 15.75 :=
by
  sorry

end jack_jill_meet_distance_l601_601069


namespace ladder_distance_l601_601698

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601698


namespace function_conditions_l601_601315

noncomputable def c : ℝ := sorry

def f (x : ℝ) : ℝ := c * |x|

theorem function_conditions (c_nonneg : 0 ≤ c) (c_le_two : c ≤ 2) :
  (∀ x : ℝ, function_conditions := ((0 < x) ∧ (x < 1)) →  f x < 2) ∧
  ∀ x y : ℝ, max (f (x + y)) (f (x - y)) = f x + f y :=
begin
  sorry
end

end function_conditions_l601_601315


namespace equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601030

section
variable [decidable_eq ℕ]
variable (deck_size: ℕ := 32)
variable (num_aces: ℕ := 4)
variable (players: Π (i: fin 4), ℕ := λ i, 1)
variable [uniform_dist: Probability_Mass_Function (fin deck_size)] 

-- Part (a): Probabilities for each player to get the first Ace
noncomputable def player1_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player2_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player3_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player4_prob (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_of_first_ace :
  player1_prob deck = 1/8 ∧
  player2_prob deck = 1/8 ∧
  player3_prob deck = 1/8 ∧
  player4_prob deck = 1/8 :=
sorry

-- Part (b): Modify rules to deal until Ace of Spades
noncomputable def player_prob_ace_of_spades (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_with_ace_of_spades :
  ∀(p: fin 4), player_prob_ace_of_spades deck = 1/4 :=
sorry
end

end equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601030


namespace area_inequality_l601_601065

-- Definition of points within a triangle
structure Triangle (α : Type _) :=
(A B C K L M N R F : α)

variables {α : Type _} [MetricSpace α] (T : Triangle α)

-- Definitions of each area
variables (E1 E2 E3 E4 E5 E6 E : ℝ)
variables (hE1 : E1 = area T.A T.M T.R)
variables (hE2 : E2 = area T.C T.K T.R)
variables (hE3 : E3 = area T.B T.K T.F)
variables (hE4 : E4 = area T.A T.L T.F)
variables (hE5 : E5 = area T.B T.M T.N)
variables (hE6 : E6 = area T.C T.L T.N)
variables (hE : E = area T.A T.B T.C)

-- Inequality to prove
theorem area_inequality :
  E ≥ 8 * (E1 * E2 * E3 * E4 * E5 * E6)^(1/6) := 
sorry

end area_inequality_l601_601065


namespace chess_tournament_total_players_l601_601455
-- Import the necessary library

-- Define the conditions and the problem statement
theorem chess_tournament_total_players :
  Exists (λ T : ℕ, 
    (∀ (n : ℕ), 
      (T = n + 8) → 
      (n * (n - 1) + 56 = (n + 8) * (n + 7) / 2) → 
      (n > 8) →
      T = 21
    )
  ) :=
sorry

end chess_tournament_total_players_l601_601455


namespace ladder_base_distance_l601_601732

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601732


namespace common_difference_of_arithmetic_sequence_l601_601922

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def S_n (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (a1 d : ℕ) (h1 : a_n a1 d 3 = 8) (h2 : S_n a1 d 6 = 54) : d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l601_601922


namespace smallest_a_with_50_squares_l601_601908


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601908


namespace number_of_distinct_arrangements_l601_601954

theorem number_of_distinct_arrangements : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
      first_position := 'A'
      specific_position := 2
      specific_letter := 'F'
      arrange (n : ℕ) (k : ℕ) := list.permutations (list.filter (λ c, c ≠ 'A' ∧ c ≠ 'F') letters) 
       in
  (k : ℕ) (∀ k = specific_position → specific_letter) (letters.length = 7) → (first_position = 'A') 
       → list.all (arrange letters) (λ l, list.nodup l) 
       → let remaining := ['B', 'C', 'D', 'E', 'G'] 
         in
  ((remaining.length).choose(3) * permute 3) = 60 := 
by
  sorry

end number_of_distinct_arrangements_l601_601954


namespace magnitude_eq_2sqrt2_l601_601424

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601424


namespace collinear_points_sum_l601_601301

theorem collinear_points_sum (p q : ℝ) (h1 : 2 = p) (h2 : q = 4) : p + q = 6 :=
by 
  rw [h1, h2]
  sorry

end collinear_points_sum_l601_601301


namespace committee_probability_l601_601530

theorem committee_probability :
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  specific_committees / total_committees = 64 / 211 := 
by
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  have h_total_committees : total_committees = 593775 := by sorry
  have h_boys_choose : boys_choose = 816 := by sorry
  have h_girls_choose : girls_choose = 220 := by sorry
  have h_specific_committees : specific_committees = 179520 := by sorry
  have h_probability : specific_committees / total_committees = 64 / 211 := by sorry
  exact h_probability

end committee_probability_l601_601530


namespace base_distance_l601_601622

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601622


namespace count_possible_values_l601_601057

open Nat

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def is_valid_addition (A B C D : ℕ) : Prop :=
  ∀ x y z w v u : ℕ, 
  (x = A) ∧ (y = B) ∧ (z = C) ∧ (w = D) ∧ (v = B) ∧ (u = D) →
  (A + C = D) ∧ (A + D = B) ∧ (B + B = D) ∧ (D + D = C)

theorem count_possible_values : ∀ (A B C D : ℕ), 
  distinct_digits A B C D → is_valid_addition A B C D → num_of_possible_D = 4 :=
by
  intro A B C D hd hv
  sorry

end count_possible_values_l601_601057


namespace triangle_property_l601_601589

-- Define the conditions for each option

-- Condition for Option A: angles in a ratio of 3:4:5
def condition_A (A B C : ℝ) : Prop :=
  (∠ A :ℝ) : (∠ B :ℝ) : (∠ C :ℝ) = 3 : 4 : 5

-- Condition for Option B: sum of angles
def condition_B (A B C : ℝ) : Prop :=
  (∠ A + ∠ B = ∠ C)

-- Condition for Option C: a^2 - b^2 = c^2
def condition_C (a b c : ℝ) : Prop :=
  (a^2 - b^2 = c^2)

-- Condition for Option D: sides in the ratio of 6:8:10
def condition_D (a b c : ℝ) : Prop :=
  (a = 6 * c) ∧ (b = 8 * c)

-- Define right triangle property
def right_triangle (A B C: ℝ) : Prop :=
  (A = 90 ∨ B = 90 ∨ C = 90)

-- Main theorem statement
theorem triangle_property :
  ( ∀A B C, condition_A A B C → ¬ right_triangle A B C ) ∧
  ( ∀A B C, condition_B A B C → right_triangle A B C ) ∧
  ( ∀a b c, condition_C a b c → right_triangle a b c ) ∧
  ( ∀a b c, condition_D a b c → right_triangle a b c ) :=
by
  -- sorry can be used here as the proof is not required
  sorry

end triangle_property_l601_601589


namespace sum_of_five_consecutive_odd_integers_l601_601176

theorem sum_of_five_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 8) = 156) :
  n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390 :=
by
  sorry

end sum_of_five_consecutive_odd_integers_l601_601176


namespace circle_arrangement_max_l601_601112

theorem circle_arrangement_max (nums : List ℝ) (h_distinct : nums.nodup) (h_circle : ∀ (x : ℕ), nums.get? x = (nums.get? (x - 1 + nums.length) % nums.length * nums.get? ((x + 1) % nums.length))) :
  nums.length ≤ 6 :=
sorry

end circle_arrangement_max_l601_601112


namespace volume_of_sphere_l601_601254

theorem volume_of_sphere (d : ℝ) (A : ℝ) (h1 : d = 1) (h2 : A = π) :
  let r := 1,
      R := sqrt 2 in
  (4 / 3) * π * R^3 = (8 * sqrt 2 * π) / 3 :=
by 
  sorry

end volume_of_sphere_l601_601254


namespace equal_prob_first_ace_l601_601040

theorem equal_prob_first_ace (deck : List ℕ) (players : Fin 4) (h_deck_size : deck.length = 32)
  (h_distinct : deck.nodup) (h_aces : ∀ _i, deck.filter (λ card, card = 1 ).length = 4)
  (h_shuffled : ∀ (card : ℕ), card ∈ deck → card ∈ (range 32)) :
  ∀ (player : Fin 4), let positions := List.range' (player + 1) (32 / 4) * 4 + player;
  (∀ (pos : ℕ), pos ∈ positions → deck.nth pos = some 1) →
  P(player) = 1 / 8 :=
by
  sorry

end equal_prob_first_ace_l601_601040


namespace a_10_equals_256_l601_601912

-- Define the sequence {a_n}, and the partial sum S_n.
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given condition: S_n is the sum of the first n terms of the sequence {a_n}
axiom sum_first_n : ∀ n : ℕ, S n = ∑ i in finset.range n, a i

-- Given condition: S_n = 2^(n-1)
axiom S_n_def : ∀ n : ℕ, S n = 2^(n-1)

theorem a_10_equals_256 : a 10 = 256 :=
by
  sorry

end a_10_equals_256_l601_601912


namespace slope_AB_is_minus_one_l601_601932

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := -1 }

-- Define the slope function
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

-- Define the proof statement
theorem slope_AB_is_minus_one : slope A B = -1 :=
by
  -- Steps for proof can be filled in here
  sorry

end slope_AB_is_minus_one_l601_601932


namespace find_c_l601_601418

variable (a b c : ℕ)

theorem find_c (h1 : a = 9) (h2 : b = 2) (h3 : Odd c) (h4 : a + b > c) (h5 : a - b < c) (h6 : b + c > a) (h7 : b - c < a) : c = 9 :=
sorry

end find_c_l601_601418


namespace log7_18_l601_601444

theorem log7_18 (a b : ℝ) (h1 : log 10 2 = a) (h2 : log 10 3 = b) : log 7 18 = (2 * a + 4 * b) / (1 + 2 * a) := 
by
  sorry

end log7_18_l601_601444


namespace area_of_quadrilateral_l601_601445

theorem area_of_quadrilateral (θ : ℝ) (sin_θ : Real.sin θ = 4/5) (b1 b2 : ℝ) (h: ℝ) (base1 : b1 = 14) (base2 : b2 = 20) (height : h = 8) : 
  (1 / 2) * (b1 + b2) * h = 136 := by
  sorry

end area_of_quadrilateral_l601_601445


namespace tau_n7_over_tau_n_l601_601341

-- Definition of τ(n) which gives the number of positive divisors of n
def tau (n : ℕ) : ℕ := (n.factors.to_finset.card : ℕ) -- Placeholder definition, replace with appropriate definition

-- Given problem conditions
variables (n : ℕ) (hn : 0 < n) (h : tau (n^2) / tau n = 3)

-- The theorem statement
theorem tau_n7_over_tau_n : tau (n^7) / tau n = 29 :=
by
  sorry

end tau_n7_over_tau_n_l601_601341


namespace find_multiple_l601_601481

-- Definitions based on the conditions provided
def mike_chocolate_squares : ℕ := 20
def jenny_chocolate_squares : ℕ := 65
def extra_squares : ℕ := 5

-- The theorem to prove the multiple
theorem find_multiple : ∃ (multiple : ℕ), jenny_chocolate_squares = mike_chocolate_squares * multiple + extra_squares ∧ multiple = 3 := by
  sorry

end find_multiple_l601_601481


namespace twenty_two_is_three_good_twenty_three_is_three_good_twenty_four_is_three_good_even_k_good_implies_k_plus_one_good_smallest_2023_bad_number_l601_601495

-- Definition of k-good number
def is_k_good (k m : ℕ) : Prop :=
  ∃ (a : Fin k.succ → ℕ) (c : Fin k.succ → ℕ), m = ∑ i, ((-1)^a i) * 2^(c i)

-- Determining whether 22, 23, 24 are 3-good numbers
theorem twenty_two_is_three_good : is_k_good 3 22 := sorry
theorem twenty_three_is_three_good : is_k_good 3 23 := sorry
theorem twenty_four_is_three_good : is_k_good 3 24 := sorry

-- If m is an even number and a k-good number, then m is a (k+1)-good number
-- and m / 2 is a k-good number.
theorem even_k_good_implies_k_plus_one_good {m k : ℕ} 
  (hm_even : even m) (hm_good : is_k_good k m) : 
  is_k_good (k + 1) m ∧ is_k_good k (m / 2) := sorry

-- The smallest 2023-bad number
theorem smallest_2023_bad_number : ∀ m, (m < (2^4047 + 1) / 3) → is_k_good 2023 m := sorry

end twenty_two_is_three_good_twenty_three_is_three_good_twenty_four_is_three_good_even_k_good_implies_k_plus_one_good_smallest_2023_bad_number_l601_601495


namespace ellipse_major_axis_length_l601_601321

theorem ellipse_major_axis_length :
  ∀ (x y : ℝ),
  (x^2 / 4) + (y^2 / 9) = 1 → 6 =
  let a := 3 in
  let major_axis_length := 2 * a in
  major_axis_length :=
begin
  intros x y h,
  let a := 3,
  let major_axis_length := 2 * a,
  have h1: 6 = major_axis_length,
  { sorry },  -- Proof steps would go here
  exact h1,
end

end ellipse_major_axis_length_l601_601321


namespace teacher_buys_total_21_pens_l601_601119

def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5
def total_pens : Nat := num_black_pens + num_blue_pens + num_red_pens

theorem teacher_buys_total_21_pens : total_pens = 21 := 
by
  unfold total_pens num_black_pens num_blue_pens num_red_pens
  rfl -- reflexivity (21 = 21)

end teacher_buys_total_21_pens_l601_601119


namespace watermelon_cost_100_l601_601749

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l601_601749


namespace find_number_eq_seven_point_five_l601_601200

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601200


namespace find_smallest_a_l601_601847

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601847


namespace carry_sum_l601_601340

def a (n : ℕ) : ℕ :=
-- definition for a(n), the number of carries when adding 2017 and n*2017
-- Here we assume a(n) is already defined or a placeholder.
sorry

theorem carry_sum :
  ∑ i in finset.range (10 ^ 2017), a (i + 1) = 10 * (10 ^ 2017 - 1) / 9 := 
by 
sorry

end carry_sum_l601_601340


namespace ladder_distance_from_wall_l601_601717

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601717


namespace investment_amount_l601_601270

noncomputable def total_investment (A T : ℝ) : Prop :=
  (0.095 * T = 0.09 * A + 2750) ∧ (T = A + 25000)

theorem investment_amount :
  ∃ T, ∀ A, total_investment A T ∧ T = 100000 :=
by
  sorry

end investment_amount_l601_601270


namespace total_hangers_l601_601975

theorem total_hangers (pink green blue yellow orange purple red : ℕ) 
  (h_pink : pink = 7)
  (h_green : green = 4)
  (h_blue : blue = green - 1)
  (h_yellow : yellow = blue - 1)
  (h_orange : orange = 2 * pink)
  (h_purple : purple = yellow + 3)
  (h_red : red = purple / 2) :
  pink + green + blue + yellow + orange + purple + red = 37 :=
sorry

end total_hangers_l601_601975


namespace geometric_sequence_a5_l601_601936

theorem geometric_sequence_a5 {a : ℕ → ℝ} 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := 
sorry

end geometric_sequence_a5_l601_601936


namespace reciprocal_of_neg_five_l601_601144

theorem reciprocal_of_neg_five : (1 / (-5 : ℝ)) = -1 / 5 := 
by
  sorry

end reciprocal_of_neg_five_l601_601144


namespace smallest_a_with_50_squares_l601_601906


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601906


namespace second_difference_is_quadratic_l601_601475

theorem second_difference_is_quadratic (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, (f (n + 2) - 2 * f (n + 1) + f n) = 2) :
  ∃ (a b : ℝ), ∀ (n : ℕ), f n = n^2 + a * n + b :=
by
  sorry

end second_difference_is_quadratic_l601_601475


namespace total_fireworks_l601_601769

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l601_601769


namespace ladder_base_distance_l601_601654

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601654


namespace min_value_l601_601369

theorem min_value (a b c x y z : ℝ) (h1 : a + b + c = 1) (h2 : x + y + z = 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  ∃ val : ℝ, val = -1 / 4 ∧ ∀ a b c x y z : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ x → 0 ≤ y → 0 ≤ z → a + b + c = 1 → x + y + z = 1 → (a - x^2) * (b - y^2) * (c - z^2) ≥ val :=
sorry

end min_value_l601_601369


namespace smallest_b_factors_l601_601834

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l601_601834


namespace smallest_a_has_50_perfect_squares_l601_601876

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601876


namespace line_bisect_circle_is_x_plus_2y_minus_5_l601_601547

-- Define the circle equation
def circle (x y : ℝ) := x^2 + y^2 - 2*x - 4*y = 0

-- Create a predicate for the line bisecting the circle and being parallel to another line
def bisect_and_parallel (l : ℝ → ℝ → Prop) (x y : ℝ) :=
  (∀ (x y : ℝ), circle x y ↔ x^2 + y^2 - 2*x - 4*y = 0) ∧
  (l x y ↔ ∃ b : ℝ, l x y ↔ x + 2*y + b = 0)

-- The theorem statement to be proved
theorem line_bisect_circle_is_x_plus_2y_minus_5 :
  ∃ l, (bisect_and_parallel l 1 2) → (∀ (x y : ℝ), l x y ↔ x + 2*y - 5 = 0) :=
by {
  sorry
}

end line_bisect_circle_is_x_plus_2y_minus_5_l601_601547


namespace p_sufficient_but_not_necessary_for_q_l601_601474

def sequence (a : ℕ → ℤ) : Prop := a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) + a n = 2^(n + 1) + 2^n

def p (a : ℕ → ℤ) : Prop := ∀ n : ℕ, a (n + 1) + a n = 2^(n + 1) + 2^n

def q (a : ℕ → ℤ) : Prop := ∃ r : ℤ, ∀ n m : ℕ, m ≠ 0 → (a (n + m) - 2^(n + m)) = r * (a n - 2^n)

theorem p_sufficient_but_not_necessary_for_q (a : ℕ → ℤ) (h_seq : sequence a):
  p a → q a ∧ ¬(q a → p a) :=
by 
  intro p,
  sorry -- Proof

end p_sufficient_but_not_necessary_for_q_l601_601474


namespace visits_exactly_two_friends_l601_601812

theorem visits_exactly_two_friends (a_visits b_visits c_visits vacation_period : ℕ) (full_period days : ℕ)
(h_a : a_visits = 4)
(h_b : b_visits = 5)
(h_c : c_visits = 6)
(h_vacation : vacation_period = 30)
(h_full_period : full_period = Nat.lcm (Nat.lcm a_visits b_visits) c_visits)
(h_days : days = 360)
(h_start_vacation : ∀ n, ∃ k, n = k * vacation_period + 30):
  ∃ n, n = 24 :=
by {
  sorry
}

end visits_exactly_two_friends_l601_601812


namespace losing_probability_l601_601118

theorem losing_probability : 
  ∀ (n : ℕ), n = 6 → (let losing_outcomes := 2 in 
                      let total_outcomes := 6 in 
                      (losing_outcomes / total_outcomes : ℚ) = 1 / 3) := 
by
  intros
  sorry

end losing_probability_l601_601118


namespace find_number_eq_seven_point_five_l601_601201

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601201


namespace value_of_product_l601_601417

theorem value_of_product (x : ℝ) (h : sqrt (6 + x) + sqrt (21 - x) = 8) : 
  (6 + x) * (21 - x) = 1369 / 4 := 
sorry

end value_of_product_l601_601417


namespace gross_revenue_is_47_l601_601590

def total_net_profit : ℤ := 44
def babysitting_profit : ℤ := 31
def lemonade_stand_expense : ℤ := 34

def gross_revenue_from_lemonade_stand (P_t P_b E : ℤ) : ℤ :=
  P_t - P_b + E

theorem gross_revenue_is_47 :
  gross_revenue_from_lemonade_stand total_net_profit babysitting_profit lemonade_stand_expense = 47 :=
by
  sorry

end gross_revenue_is_47_l601_601590


namespace min_sqrt_leq_sqrt3_l601_601814

theorem min_sqrt_leq_sqrt3 (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  min (m^(1/(n:ℝ))) (n^(1/(m:ℝ))) ≤ 3^(1/3) :=
sorry

end min_sqrt_leq_sqrt3_l601_601814


namespace arithmetic_sequence_sum_l601_601924

theorem arithmetic_sequence_sum 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 = 12) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28) :=
sorry

end arithmetic_sequence_sum_l601_601924


namespace trigonometric_identity_l601_601913

theorem trigonometric_identity (x : ℝ) (hx : sin x = 2 * cos x) :
  (sin x) ^ 2 - 2 * (sin x) * (cos x) + 3 * (cos x) ^ 2 = 3 / 5 := by
  sorry

end trigonometric_identity_l601_601913


namespace number_of_rows_of_desks_is_8_l601_601456

-- Definitions for the conditions
def first_row_desks : ℕ := 10
def desks_increment : ℕ := 2
def total_desks : ℕ := 136

-- Definition for the sum of an arithmetic series
def arithmetic_series_sum (n a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- The proof problem statement
theorem number_of_rows_of_desks_is_8 :
  ∃ n : ℕ, arithmetic_series_sum n first_row_desks desks_increment = total_desks ∧ n = 8 :=
by
  sorry

end number_of_rows_of_desks_is_8_l601_601456


namespace number_of_girls_more_than_boys_l601_601751

theorem number_of_girls_more_than_boys
  (total_students : ℕ)
  (number_of_boys : ℕ)
  (h1 : total_students = 485)
  (h2 : number_of_boys = 208) :
  (total_students - number_of_boys) - number_of_boys = 69 := 
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end number_of_girls_more_than_boys_l601_601751


namespace ladder_base_distance_l601_601722

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601722


namespace trip_duration_60_mph_l601_601149

noncomputable def time_at_new_speed (initial_time : ℚ) (initial_speed : ℚ) (new_speed : ℚ) : ℚ :=
  initial_time * (initial_speed / new_speed)

theorem trip_duration_60_mph :
  time_at_new_speed (9 / 2) 70 60 = 5.25 := 
by
  sorry

end trip_duration_60_mph_l601_601149


namespace angle_BCM_in_pentagon_l601_601463

-- Definitions of the conditions
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), ∃ (angle : ℝ), angle = 108)

structure EquilateralTriangle (A B M : Type) :=
  (is_equilateral : ∀ (x y : Type), ∃ (angle : ℝ), angle = 60)

-- Problem statement
theorem angle_BCM_in_pentagon (A B C D E M : Type) (P : Pentagon A B C D E) (T : EquilateralTriangle A B M) :
  ∃ (angle : ℝ), angle = 66 :=
by
  sorry

end angle_BCM_in_pentagon_l601_601463


namespace collinear_points_sum_l601_601304

theorem collinear_points_sum (p q : ℝ) 
  (h1 : p = 2) (h2 : q = 4) 
  (collinear : ∃ (s : ℝ), 
     (2, p, q) = (2, s*p, s*q) ∧ 
     (p, 3, q) = (s*p, 3, s*q) ∧ 
     (p, q, 4) = (s*p, s*q, 4)): 
  p + q = 6 := by
  sorry

end collinear_points_sum_l601_601304


namespace score_of_tenth_game_must_be_at_least_l601_601476

variable (score_5 average_9 average_10 score_10 : ℤ)
variable (H1 : average_9 > score_5 / 5)
variable (H2 : average_10 > 18)
variable (score_6 score_7 score_8 score_9 : ℤ)
variable (H3 : score_6 = 23)
variable (H4 : score_7 = 14)
variable (H5 : score_8 = 11)
variable (H6 : score_9 = 20)
variable (H7 : average_9 = (score_5 + score_6 + score_7 + score_8 + score_9) / 9)
variable (H8 : average_10 = (score_5 + score_6 + score_7 + score_8 + score_9 + score_10) / 10)

theorem score_of_tenth_game_must_be_at_least :
  score_10 ≥ 29 :=
by
  sorry

end score_of_tenth_game_must_be_at_least_l601_601476


namespace polynomial_divisibility_l601_601329

noncomputable def P (x : ℂ) : ℂ := x^66 + x^55 + x^44 + x^33 + x^22 + x^11 + 1
noncomputable def Q (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

-- Stating the proof problem as a Lean 4 theorem without solving it.
theorem polynomial_divisibility : ∀ (x : ℂ), Q(x) = 0 → P(x) = 0 :=
by
  -- by using the conditions, the proof will follow from the properties of the nth roots of unity
  sorry

end polynomial_divisibility_l601_601329


namespace ladder_distance_from_wall_l601_601712

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601712


namespace m_values_l601_601342

def operation_odot (a b : ℝ) : ℝ := 
  if a - b ≤ 1 then a else b

def f (x : ℝ) : ℝ := operation_odot (2^(x + 1)) (1 - x)

def g (x : ℝ) : ℝ := x^2 - 6 * x

theorem m_values (m : ℝ) (h1 : f x is_decreasing_on (set.Ioo m (m + 1)))
  (h2 : g x is_decreasing_on (set.Ioo m (m + 1)))
  (h3 : m ∈ set_of (-1, 0, 1, 3)) :
  m = 0 ∨ m = 1 :=
sorry

end m_values_l601_601342


namespace equation_has_two_solutions_l601_601553

theorem equation_has_two_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^x1 = x1^2 - 2*x1 - a ∧ a^x2 = x2^2 - 2*x2 - a :=
sorry

end equation_has_two_solutions_l601_601553


namespace probability_first_ace_equal_l601_601035

theorem probability_first_ace_equal (num_cards : ℕ) (num_aces : ℕ) (num_players : ℕ)
  (h1 : num_cards = 32) (h2 : num_aces = 4) (h3 : num_players = 4) :
  ∀ player : ℕ, player ∈ {1, 2, 3, 4} → (∃ positions : list ℕ, (∀ n ∈ positions, n % num_players = player - 1)) → 
  (positions.length = 8) →
  let P := 1 / 8 in
  P = 1 / num_players :=
begin
  sorry
end

end probability_first_ace_equal_l601_601035


namespace no_real_roots_range_l601_601020

theorem no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 :=
by
  sorry

end no_real_roots_range_l601_601020


namespace probability_positive_product_l601_601168

def set_of_integers : set ℤ := {-7, -3, 2, 8, -1}
def favorable_outcomes : ℕ := 4
def total_outcomes : ℕ := 10

theorem probability_positive_product : (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 2 / 5 := by
  sorry

end probability_positive_product_l601_601168


namespace polynomial_expansion_derived_sum_l601_601910

theorem polynomial_expansion_derived_sum :
  (∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ),
  (2*x - 1)^6 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6) →
  (6 : ℝ) * (2*1 - 1)^5 * (2 : ℝ) = ∑ n in finset.range(7), (n * (a_n * (1 : ℝ)^n)) :=
begin
  sorry
end

end polynomial_expansion_derived_sum_l601_601910


namespace three_digit_numbers_with_diff_3_l601_601413

/--
The number of three-digit numbers where the difference between consecutive digits is 3
is equal to 3.
-/
theorem three_digit_numbers_with_diff_3 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 3)
  (hB : B = A + 3) (hC : C = A + 6) (hB_range : 1 ≤ B ∧ B ≤ 9) (hC_range : 1 ≤ C ∧ C ≤ 9) :
  {n : ℕ // n < 1000 ∧ n ≥ 100 ∧ (A * 100 + B * 10 + C) = n}.card = 3 := 
sorry

end three_digit_numbers_with_diff_3_l601_601413


namespace magnitude_complex_expression_l601_601443

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601443


namespace elena_recipe_multiple_l601_601822

variable (butter_per_flour : ℕ → ℕ → ℚ)
variable [decidable_eq ℚ]

/-- Given that the original recipe calls for 5 ounces of butter for each 
7 cups of flour and that Elena's recipe uses 12 ounces of butter for 28 cups of flour, 
we want to prove that Elena makes 2.4 times the original recipe. -/
theorem elena_recipe_multiple 
  (h1 : butter_per_flour 5 7 = 5 / 7)
  (h2 : butter_per_flour 12 28 = 12 / 28) :
  (12 / 5 : ℚ) = 2.4 :=
  sorry

end elena_recipe_multiple_l601_601822


namespace graph_shift_upwards_by_two_l601_601542

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Set.Icc (-3) 0 then -2 - x
else if x ∈ Set.Icc 0 2 then Real.sqrt (4 - (x - 2)^2) - 2
else if x ∈ Set.Icc 2 3 then 2 * (x - 2)
else 0

noncomputable def g (x : ℝ) : ℝ := f x + 2

theorem graph_shift_upwards_by_two : 
  ∀ (x : ℝ), g x = f x + 2 := 
by {
  intro x,
  unfold g,
  exact rfl,
}

end graph_shift_upwards_by_two_l601_601542


namespace halfway_between_one_nine_and_one_eleven_l601_601324

theorem halfway_between_one_nine_and_one_eleven : 
  (1/9 + 1/11) / 2 = 10/99 :=
by sorry

end halfway_between_one_nine_and_one_eleven_l601_601324


namespace yesterday_tomorrow_is_friday_l601_601451

-- Defining the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to go to the next day
def next_day : Day → Day
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday

-- Function to go to the previous day
def previous_day : Day → Day
| Sunday    => Saturday
| Monday    => Sunday
| Tuesday   => Monday
| Wednesday => Tuesday
| Thursday  => Wednesday
| Friday    => Thursday
| Saturday  => Friday

-- Proving the statement
theorem yesterday_tomorrow_is_friday (T : Day) (H : next_day (previous_day T) = Thursday) : previous_day (next_day (next_day T)) = Friday :=
by
  sorry

end yesterday_tomorrow_is_friday_l601_601451


namespace find_n_from_binomial_terms_l601_601344

theorem find_n_from_binomial_terms (x a : ℕ) (n : ℕ) 
  (h1 : n.choose 1 * x^(n-1) * a = 56) 
  (h2 : n.choose 2 * x^(n-2) * a^2 = 168) 
  (h3 : n.choose 3 * x^(n-3) * a^3 = 336) : 
  n = 5 :=
by
  sorry

end find_n_from_binomial_terms_l601_601344


namespace parallel_lines_distance_l601_601003

theorem parallel_lines_distance :
  let l1 : ℝ → ℝ → ℝ := λ x y, 3 * x + 4 * y - (3 / 4)
  let l2 : ℝ → ℝ → ℝ := λ x y, 12 * x + 16 * y + 37
  let A := 12
  let B := 16
  let C1 := -3
  let C2 := 37
  let d := λ (A B C1 C2 : ℝ), |C1 - C2| / Real.sqrt (A^2 + B^2)
  d A B C1 C2 = 2 :=
by
  sorry

end parallel_lines_distance_l601_601003


namespace minimum_value_of_A_l601_601323

noncomputable def A (n : ℕ) (x y z : Fin n → ℝ) : ℝ :=
  let sum_of : List ℝ → ℝ := List.foldr (· + ·) 0
  let sn := sum_of (List.ofFn (Fin n) z)
  (Real.sqrt ((sum_of (List.ofFn (Fin n) (λ i => x i) - 948)^2) + 
    (sum_of (List.ofFn (Fin n) (λ i => y i) - 1185)^2 + z 0^2)) +
  sum_of (List.ofFn (λ m, Real.sqrt (z (Nat.pred m)^2 + x (Nat.pred (n - m))^2 + y (Nat.pred (n - m))^2))) +
  Real.sqrt ((1264 - sum_of (List.ofFn (Fin n) (λ i => z i)))^2 + x n.pred^2 + y n.pred^2)

theorem minimum_value_of_A (n : ℕ) (x y z : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) (hy : ∀ i, 0 ≤ y i) (hz : ∀ i, 0 ≤ z i) :
  (A n x y z) = 1975 :=
sorry

end minimum_value_of_A_l601_601323


namespace exp_to_rect_form_l601_601290

open Complex Real

-- Define the problem conditions
def euler_formula (θ : ℝ) : Complex := Complex.exp (θ * Complex.i) = cos θ + Complex.i * sin θ

def pi_six_cos : cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry

def pi_six_sin : sin (Real.pi / 6) = 1 / 2 := by sorry

-- The theorem to prove
theorem exp_to_rect_form : 
  (sqrt 3 : ℂ) * Complex.exp ((13 * Real.pi / 6) * Complex.i) = 
  (3 / 2 : ℂ) + (Complex.i * (sqrt 3 / 2 : ℂ)) :=
by 
  have h1 := euler_formula (13 * Real.pi / 6)
  -- The proof is skipped
  sorry

end exp_to_rect_form_l601_601290


namespace total_soccer_balls_donated_l601_601764

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end total_soccer_balls_donated_l601_601764


namespace find_e_l601_601807

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) (h1 : 3 + d + e + f = -6)
  (h2 : - f / 3 = -6)
  (h3 : 9 = f)
  (h4 : - d / 3 = -18) : e = -72 :=
by
  sorry

end find_e_l601_601807


namespace ladder_distance_l601_601704

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601704


namespace number_of_sides_of_polygon_l601_601973

theorem number_of_sides_of_polygon (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l601_601973


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601428

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601428


namespace find_a2_plus_b2_l601_601971

theorem find_a2_plus_b2 (a b : ℝ) :
  (∀ x, |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x| ≤ 11)
  → a^2 + b^2 = 50 :=
by
  sorry

end find_a2_plus_b2_l601_601971


namespace magnitude_complex_expression_l601_601440

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601440


namespace base_distance_l601_601621

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601621


namespace find_AB_l601_601468

noncomputable def Triangle : Type := 
  { a b c : ℝ // a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180 }

variables (ABC DEF : Triangle)

/-- Defining the first triangle with specific side lengths and angles. --/
def triangle_ABC : Triangle := {
  a := 120,
  b := 6,
  c := 54, -- remaining angle for the triangle to sum to 180 degrees
  sorry := sorry
}

/-- Defining the second triangle with specific side lengths and angles. --/
def triangle_DEF : Triangle := {
  a := 120,
  b := 4,
  c := 102,
  sorry := sorry
}

theorem find_AB (ABC DEF : Triangle) (h1 : ABC.a = 120) (h2 : ABC.b = 6) (h3 : ABC.c = 54)
(h4 : DEF.a = 120) (h5 : DEF.b = 4) (h6 : DEF.c = 102) : 
similar (ABC) (DEF) → (ABC.b = 8) :=
begin
  sorry
end

end find_AB_l601_601468


namespace sin_minus_cos_proof_l601_601376

noncomputable def sin_minus_cos (α : Real) : Real := 
-Real.sqrt 17 / 3

theorem sin_minus_cos_proof (α : Real) (h1 : cos α + sin α = 1 / 3) (h2 : 0 ≤ cos α) (h3 : cos α ≤ 1) (h4 : -1 ≤ sin α) (h5 : sin α ≤ 0):
  sin α - cos α = sin_minus_cos α :=
sorry

end sin_minus_cos_proof_l601_601376


namespace find_b_l601_601841

-- Defining the problem conditions
variables {a b c d : ℝ}
variable (i : ℂ)
variable (z w : ℂ)

-- Given conditions in the problem
def has_four_non_real_roots (f : ℂ → ℂ) :=
  ∀ x, f x = (x - z) * (x - w) * (x - conj(z)) * (x - conj(w))

-- Additional given conditions
def product_of_two_roots : z * w = (7 + 4 * i) := by sorry
def sum_of_conjugate_roots : conj(z) + conj(w) = (-2 + 5 * i) := by sorry

-- Main statement to prove
theorem find_b (h_i : i^2 = -1) (h_roots: has_four_non_real_roots (λ x, x^4 + a*x^3 + b*x^2 + c*x + d))
    (h_prod: product_of_two_roots)
    (h_sum_conj: sum_of_conjugate_roots) :
  b = 43 := by 
sorry

end find_b_l601_601841


namespace train_length_l601_601750

theorem train_length (v_kmph : ℝ) (t_s : ℝ) (L_p : ℝ) (L_t : ℝ) : 
  (v_kmph = 72) ∧ (t_s = 15) ∧ (L_p = 250) →
  L_t = 50 :=
by
  intro h
  sorry

end train_length_l601_601750


namespace least_n_for_distance_ge_100_l601_601489

structure Point2D :=
(x : ℝ)
(y : ℝ)

def is_equilateral (A B C : Point2D) : Prop :=
dist A B = dist B C ∧ dist B C = dist C A

noncomputable def A (n : ℕ) : Point2D :=
match n with
| 0 => ⟨0, 0⟩
| n+1 => sorry -- a function to calculate based on previous points and properties

noncomputable def B (n : ℕ) : Point2D :=
let x_n := match n with
           | 0 => 0
           | n+1 => (A (n+1)).x
           in Point2D.mk x_n (Real.sqrt x_n)

theorem least_n_for_distance_ge_100 : ∃ n : ℕ, (A n).x ≥ 100 ∧ ∀ m < n, (A m).x < 100 :=
begin
  sorry -- the proof will show that the least such n is 17.
end

end least_n_for_distance_ge_100_l601_601489


namespace smallest_divisor_is_10_l601_601598

/-- Define the prime factorization of 2880. -/
def factor_2880 : ℕ := 2^5 * 3^2 * 5

/-- A function to compute the smallest factor resulting in a perfect square when dividing 2880. -/
def smallest_perfect_square_divisor (n : ℕ) : Prop :=
  (factor_2880 % n = 0) ∧ 
  (∃ m : ℕ, m^2 = factor_2880 / n)

theorem smallest_divisor_is_10 : smallest_perfect_square_divisor 10 :=
by
  -- Assuming factor_2880 is correctly defined as 2880 = 2^5 * 3^2 * 5
  have h1 : factor_2880 = 2880 := rfl
  -- Proving 10 divides 2880
  have h2 : factor_2880 % 10 = 0 := sorry
  -- Proving 2880 / 10 = 288 is a perfect square
  have h3 : ∃ m : ℕ, m^2 = factor_2880 / 10 := sorry
  exact ⟨h2, h3⟩

end smallest_divisor_is_10_l601_601598


namespace ladder_base_distance_l601_601730

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601730


namespace exp_to_rect_form_l601_601291

open Complex Real

-- Define the problem conditions
def euler_formula (θ : ℝ) : Complex := Complex.exp (θ * Complex.i) = cos θ + Complex.i * sin θ

def pi_six_cos : cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry

def pi_six_sin : sin (Real.pi / 6) = 1 / 2 := by sorry

-- The theorem to prove
theorem exp_to_rect_form : 
  (sqrt 3 : ℂ) * Complex.exp ((13 * Real.pi / 6) * Complex.i) = 
  (3 / 2 : ℂ) + (Complex.i * (sqrt 3 / 2 : ℂ)) :=
by 
  have h1 := euler_formula (13 * Real.pi / 6)
  -- The proof is skipped
  sorry

end exp_to_rect_form_l601_601291


namespace card_distribution_remainders_l601_601460

theorem card_distribution_remainders (m n : ℕ) (cards : Fin (m * n) → ℕ) 
  (boys_cards girls_cards : Fin m → Fin n → Fin (m * n)) :
  (∀ i, boys_cards i = cards (Fin.castSucc i)) →
  (∀ j, girls_cards j = cards (Fin.castSucc j + m)) →
  (∀ i j, (cards i + cards j) % (m * n) ≠ (cards i + cards j) % (m * n) → i ≠ j) →
  (∃ (m = 1) ∨ (n = 1)) :=
sorry

end card_distribution_remainders_l601_601460


namespace range_of_m_l601_601407

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | x^2 - 5*x - 14 ≤ 0},
      B := {x : ℝ | m + 1 < x ∧ x < 2 * m - 1} in
  (A ∪ B = A) ↔ (m ∈ Iic 4) :=
by
  sorry

end range_of_m_l601_601407


namespace log_relation_l601_601958

theorem log_relation (a b : ℝ) 
  (h₁ : a = Real.log 1024 / Real.log 16) 
  (h₂ : b = Real.log 32 / Real.log 2) : 
  a = 1 / 2 * b := 
by 
  sorry

end log_relation_l601_601958


namespace find_x_solution_l601_601827

theorem find_x_solution :
  ∃ x, 2 ^ (x / 2) * (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x)) = 6 ∧
       x = 2 * Real.log 1.5 / Real.log 2 := by
  sorry

end find_x_solution_l601_601827


namespace total_expenditure_correct_l601_601047

-- Definitions for conditions
def length := 20 -- in meters
def width := 15 -- in meters
def height := 5 -- in meters
def cost_per_square_meter := 30 -- Rs. per square meter

-- Definition for area calculations
def A_floor := length * width
def A_long_walls := 2 * (length * height)
def A_short_walls := 2 * (width * height)
def A_total := A_floor + A_long_walls + A_short_walls

-- Definition for total expenditure
def total_expenditure := A_total * cost_per_square_meter

-- The theorem statement
theorem total_expenditure_correct : total_expenditure = 19500 := by {
  sorry
}

end total_expenditure_correct_l601_601047


namespace ladder_base_distance_l601_601667

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601667


namespace inclination_angle_l601_601388

noncomputable theory

-- Given point P
def P : ℝ × ℝ := (0, real.sqrt 3)

-- Line l with angle of inclination α passing through P
def parametric_line (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * real.cos α, real.sqrt 3 + t * real.sin α)

-- Polar equation of circle C translated to Cartesian form
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - real.sqrt 3)^2 = 5

-- Main theorem to prove the inclination angle α
theorem inclination_angle (α : ℝ) :
  (∀ (t₁ t₂ : ℝ),
    circle_C (parametric_line α t₁).1 (parametric_line α t₁).2 ∧
    circle_C (parametric_line α t₂).1 (parametric_line α t₂).2 →
    abs (t₁ - t₂) = real.sqrt 2) →
  (α = real.pi / 4 ∨ α = 3 * real.pi / 4) := by
  sorry

end inclination_angle_l601_601388


namespace base_distance_l601_601619

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601619


namespace area_difference_square_rectangle_l601_601156

theorem area_difference_square_rectangle :
  ∀ (P_sq P_rect l_rect : ℝ), 
  P_sq = 52 → 
  P_rect = 52 → 
  l_rect = 15 → 
  let s := P_sq / 4 in 
  let w_rect := (P_rect / 2) - l_rect in 
  (s * s - l_rect * w_rect) = 4 :=
by
  intros P_sq P_rect l_rect hP_sq hP_rect hl_rect s w_rect,
  sorry

end area_difference_square_rectangle_l601_601156


namespace reciprocal_of_2023_l601_601557

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l601_601557


namespace existence_of_points_l601_601840

theorem existence_of_points (n : ℕ) (h1 : n ≥ 3) :
  ∃ P : fin n → ℝ × ℝ,
    (∀ i j : fin n, i ≠ j → (let d := real.sqrt ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 in irrational d)) ∧
    (∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k →
      let area := 1 / 2 * abs ((P i).1 * (P j).2 + (P j).1 * (P k).2 + (P k).1 * (P i).2 - (P j).1 * (P i).2 - (P k).1 * (P j).2 - (P i).1 * (P k).2)
      in rational area) :=
begin
  sorry
end

end existence_of_points_l601_601840


namespace average_final_value_l601_601735

def compound_return (r1 r2 r3 r4 : ℝ) : ℝ :=
  (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4)

theorem average_final_value :
  let final_A := compound_return 0.10 0.05 (-0.15) 0.25 in
  let final_B := compound_return 0.08 (-0.12) (-0.05) 0.20 in
  let final_C := compound_return (-0.05) 0.10 0.15 (-0.08) in
  let final_D := compound_return 0.18 0.04 (-0.10) (-0.02) in
  let final_E := compound_return 0.03 (-0.18) 0.22 0.11 in
  (final_A + final_B + final_C + final_D + final_E) / 5 = 1.119251 := by
  sorry

end average_final_value_l601_601735


namespace base8_satisfies_l601_601332

noncomputable def check_base (c : ℕ) : Prop := 
  ((2 * c ^ 2 + 4 * c + 3) + (1 * c ^ 2 + 5 * c + 6)) = (4 * c ^ 2 + 2 * c + 1)

theorem base8_satisfies : check_base 8 := 
by
  -- conditions: (243_c, 156_c, 421_c) translated as provided
  -- proof is skipped here as specified
  sorry

end base8_satisfies_l601_601332


namespace ladder_distance_l601_601674

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601674


namespace Mark_candies_l601_601093

theorem Mark_candies (peter_candies : ℕ) (john_candies : ℕ) (equal_share : ℕ) 
  (h_peter : peter_candies = 25) 
  (h_john : john_candies = 35) 
  (h_share : equal_share = 30) : 
  let total_candies := equal_share * 3 in
  let mark_candies := total_candies - (peter_candies + john_candies) in
  mark_candies = 30 :=
by
  sorry

end Mark_candies_l601_601093


namespace valid_paths_count_l601_601507

-- Define the coordinates of points of interest
structure Point where
  x : ℕ
  y : ℕ

def A : Point := { x := 0, y := 4 }
def B : Point := { x := 20, y := 0 }
def C : Point := { x := 12, y := 2 }
def D : Point := { x := 12, y := 3 }

-- Function to compute the binomial coefficient (combinations)
def binom (n k : ℕ) : ℕ :=
  Nat_choose n k

-- Total number of paths from A to B without restrictions
def total_paths (start end : Point) : ℕ :=
  binom (end.x - start.x + (start.y - end.y)) (start.y - end.y)

-- Number of paths from A to C
def paths_A_to_C : ℕ :=
  total_paths A C

-- Number of paths from D to B
def paths_D_to_B : ℕ :=
  total_paths D B

-- Number of restricted paths passing through forbidden segment CD
def restricted_paths : ℕ :=
  paths_A_to_C * paths_D_to_B

-- Total valid paths from A to B avoiding forbidden segment
def valid_paths : ℕ :=
  total_paths A B - restricted_paths

theorem valid_paths_count : valid_paths = 861 := by
  sorry

end valid_paths_count_l601_601507


namespace segment_length_AB_l601_601917

-- Define the line l with parameter equations
def line_l (t : ℝ) : ℝ × ℝ := (t, t - 2)

-- Define the curve C with parameter equations
def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the functions which represent the line and curve
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0
def curve_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the proof problem stating that the length of segment AB is 2√2
theorem segment_length_AB : ∀ A B : ℝ × ℝ, 
  (∃ t : ℝ, A = line_l t) → (∃ θ : ℝ, A = curve_C θ) →
  (∃ t : ℝ, B = line_l t) → (∃ θ : ℝ, B = curve_C θ) →
  (A ≠ B) → ∥A - B∥ = 2 * Real.sqrt 2 :=
by
  intros A B hAline hAcurve hBline hBcurve hABneq
  sorry
  -- Proof would go here, utilizing the provided conditions.


end segment_length_AB_l601_601917


namespace minimum_S_sqrt3_l601_601931

def minimum_S (x y z : ℝ) : ℝ :=
  (x * y) / z + (y * z) / x + (z * x) / y

theorem minimum_S_sqrt3 {x y z : ℝ} (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : x^2 + y^2 + z^2 = 1) :
  ∃ S : ℝ, S = sqrt 3 ∧ ∀ t : ℝ, (minimum_S x y z) ≥ t → t = sqrt 3 :=
begin
  use sqrt 3,
  sorry -- Proof omitted
end

end minimum_S_sqrt3_l601_601931


namespace shaded_area_square_semicircles_l601_601115

theorem shaded_area_square_semicircles :
  let side_length := 2
  let radius_circle := side_length * Real.sqrt 2 / 2
  let area_circle := Real.pi * radius_circle^2
  let area_square := side_length^2
  let area_semicircle := Real.pi * (side_length / 2)^2 / 2
  let total_area_semicircles := 4 * area_semicircle
  let shaded_area := total_area_semicircles - area_circle
  shaded_area = 4 :=
by
  sorry

end shaded_area_square_semicircles_l601_601115


namespace parallel_vectors_cosine_l601_601404

variable (α : ℝ)

def a := (Real.cos (Real.pi / 3 + α), 1)
def b := (1, 4)

theorem parallel_vectors_cosine : 
  (a α).1 / (a α).2 = (b α).1 / (b α).2 → 
  Real.cos (Real.pi / 3 - 2 * α) = 7 / 8 := 
by
  intro h
  -- actual proof omitted
  sorry

end parallel_vectors_cosine_l601_601404


namespace learn_at_least_537_words_l601_601012

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words_l601_601012


namespace set_M_properties_l601_601948

theorem set_M_properties (α β γ : ℂ) (M : Set ℂ) 
  (h1 : M = {α, β, γ})
  (h2 : ∀ x y ∈ M, x * y ∈ M)
  (h3 : ∀ z ∈ M, z^2 ∈ M) : 
  M = {-1, 0, 1} := 
sorry

end set_M_properties_l601_601948


namespace part_a_l601_601221

-- Lean 4 statement equivalent to Part (a)
theorem part_a (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : n^2 ≤ x) : 
  n * Real.sqrt (x - n^2) ≤ x / 2 := 
sorry

-- Lean 4 statement equivalent to Part (b)
noncomputable def find_xyz : ℕ × ℕ × ℕ :=
  ((2, 8, 18) : ℕ × ℕ × ℕ)

end part_a_l601_601221


namespace ladder_base_distance_l601_601666

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601666


namespace total_prime_factors_l601_601223

theorem total_prime_factors : 
  (∃ (n : ℕ), n = (4 ^ 11) * (7 ^ 7) * (11 ^ 2)) → 
  ∃ (k : ℕ), k = 31 :=
begin
  sorry
end

end total_prime_factors_l601_601223


namespace coefficient_x2y3_l601_601467

-- Define the general binomial term
noncomputable def binomial_coeff (n k : ℕ) : ℤ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Expansion of (x - 2y)^4
noncomputable def binomial_term (n : ℕ) (x y : ℤ) (r : ℕ) : ℤ :=
  binomial_coeff n r * x^(n-r) * (-2*y)^r

-- The specific term we are looking for is x^2 * y^3
noncomputable def target_term (x y : ℤ) : ℤ :=
  -40 * x^2 * y^3

-- Main statement
theorem coefficient_x2y3 : ∀ (x y : ℤ),
  let term := target_term x y in
  term = -40 * x^2 * y^3 :=
  sorry

end coefficient_x2y3_l601_601467


namespace smallest_a_has_50_perfect_squares_l601_601855

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601855


namespace estimate_total_score_l601_601528

def score_increase (t : ℝ) (k P : ℝ) : ℝ := k * P / (1 + real.log (t + 1))

theorem estimate_total_score :
  ∀ (P : ℝ), (∀ (k : ℝ), score_increase 60 k P = P / 6 → 
    let k := (1 + real.log 61) / 6 in
    let f100 := score_increase 100 k P in
    P = 400 →
    P + f100 ≈ 460) :=
by
  intros P k h k_val f100_val mock_score
  sorry

end estimate_total_score_l601_601528


namespace ellipse_foci_theorem_l601_601374

noncomputable def ellipse_foci_and_point_max_triangle_area (m n : ℝ) : Prop :=
  let F1 := (-3, 0)
  let F2 := (3, 0)
  let ellipse_eq := (x y : ℝ) -> (x^2 / m + y^2 / n = 1)
  ∃ P : ℝ × ℝ, 
    (P ∈ ellipse_eq) ∧ 
    (angle F1 P F2 = 2 * Real.pi / 3) ∧ 
    (P forms maximum area triangle with F1 F2)

theorem ellipse_foci_theorem : ellipse_foci_and_point_max_triangle_area 12 3 := 
sorry

end ellipse_foci_theorem_l601_601374


namespace count_specific_integers_l601_601955

def contains_digits (n : ℕ) (d1 d2 : ℕ) : Prop :=
  d1 ∈ (n.digits 10) ∧ d2 ∈ (n.digits 10)

theorem count_specific_integers : 
  ∃ count : ℕ, count = 14 ∧ ∀ n : ℕ, 700 ≤ n ∧ n ≤ 1500 → contains_digits n 2 5 → n ∈ (700..1500) :=
begin
  use 14,
  sorry,
end

end count_specific_integers_l601_601955


namespace translate_f_to_g_l601_601574

-- Define the original function
def f (x : ℝ) : ℝ := log 2 (3 * x + 2) - 1

-- Define the translated function
def g (x : ℝ) : ℝ := log 2 (3 * x - 4)

-- State that translating the function f by 1 upward and 2 to the right yields g
theorem translate_f_to_g : ∀ x : ℝ, g x = log 2 (3 * (x - 2) + 2) := sorry

end translate_f_to_g_l601_601574


namespace transformed_function_l601_601573

-- Define the initial function
def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

-- Define the function after left shift by π/4
def f_shifted_left (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4 + Real.pi / 4)

-- Define the new function after compressing the horizontal axis to half
def f_new (x : ℝ) : ℝ := Real.cos (2 * x)

theorem transformed_function :
  (∀ x : ℝ, f_shifted_left (x/2) = f_new x) :=
by
  -- Proof is required here
  sorry

end transformed_function_l601_601573


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601427

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601427


namespace max_points_in_M_l601_601062

-- Definitions for the region M and conditions for the point sets A and B.
def region_M (x y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 2 - x ∧ 0 ≤ x ∧ x ≤ 2

def group_A_constraint (S : Finset (ℝ × ℝ)) : Prop :=
  S.sum (λ p, p.1) ≤ 6

def group_B_constraint (S : Finset (ℝ × ℝ)) : Prop :=
  S.sum (λ p, p.2) ≤ 6

-- The theorem statement
theorem max_points_in_M (k : ℕ) (points : Finset (ℝ × ℝ)) (h : ∀ p ∈ points, region_M p.1 p.2) :
  k ≤ 11 :=
sorry

end max_points_in_M_l601_601062


namespace ladder_distance_l601_601681

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601681


namespace num_valid_seat_permutations_l601_601307

/-- 
  The number of ways eight people can switch their seats in a circular 
  arrangement such that no one sits in the same, adjacent, or directly 
  opposite chair they originally occupied is 5.
-/
theorem num_valid_seat_permutations : 
  ∃ (σ : Equiv.Perm (Fin 8)), 
  (∀ i : Fin 8, σ i ≠ i) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 7 then i + 1 else 0) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 8 / 2 then (i + 8 / 2) % 8 else (i - 8 / 2) % 8) :=
  sorry

end num_valid_seat_permutations_l601_601307


namespace smallest_a_with_50_perfect_squares_l601_601896

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601896


namespace find_g_inv_l601_601017

noncomputable def g (x : ℝ) : ℝ :=
  (x^7 - 1) / 4

noncomputable def g_inv_value : ℝ :=
  (51 / 32)^(1/7)

theorem find_g_inv (h : g (g_inv_value) = 19 / 128) : g_inv_value = (51 / 32)^(1/7) :=
by
  sorry

end find_g_inv_l601_601017


namespace smallest_a_with_50_perfect_squares_l601_601889

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601889


namespace magnitude_complex_expression_l601_601439

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601439


namespace smallest_natural_number_with_50_squares_in_interval_l601_601867

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601867


namespace ladder_distance_l601_601701

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601701


namespace greatest_of_consecutive_integers_with_sum_39_l601_601580

theorem greatest_of_consecutive_integers_with_sum_39 :
  ∃ x : ℤ, x + (x + 1) + (x + 2) = 39 ∧ max (max x (x + 1)) (x + 2) = 14 :=
by
  sorry

end greatest_of_consecutive_integers_with_sum_39_l601_601580


namespace find_second_number_l601_601121

theorem find_second_number (x : ℕ) : 
  ((20 + 40 + 60) / 3 = 4 + ((x + 10 + 28) / 3)) → x = 70 :=
by {
  -- let lhs = (20 + 40 + 60) / 3
  -- let rhs = 4 + ((x + 10 + 28) / 3)
  -- rw rhs at lhs,
  -- value the lhs and rhs,
  -- prove x = 70
  sorry
}

end find_second_number_l601_601121


namespace perp_134_implies_2_perp_234_implies_1_l601_601081

variables {Plane Line : Type} 
variables (m n : Line) (α β : Plane)

-- Define perpendicular relations
def perp (a b : Type) : Prop := sorry -- Define this adequately

-- ①③④ => ②
theorem perp_134_implies_2 (h1 : perp m n) (h3 : perp n β) (h4 : perp m α) : perp α β :=
by sorry

-- ②③④ => ①
theorem perp_234_implies_1 (h2 : perp α β) (h3 : perp n β) (h4 : perp m α) : perp m n :=
by sorry

end perp_134_implies_2_perp_234_implies_1_l601_601081


namespace smallest_a_has_50_perfect_squares_l601_601873

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601873


namespace distance_from_wall_l601_601637

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601637


namespace ellipses_same_focal_length_l601_601406

noncomputable def focal_length (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

def C1 : ∀ x y : ℝ, (x^2) / 12 + (y^2) / 4 = 1 := sorry
def C2 : ∀ x y : ℝ, (x^2) / 16 + (y^2) / 8 = 1 := sorry

theorem ellipses_same_focal_length :
  focal_length (2 * real.sqrt 3) 2 = focal_length 4 (2 * real.sqrt 2) :=
by
  have a1 : ℝ := 2 * real.sqrt 3
  have b1 : ℝ := 2
  have a2 : ℝ := 4
  have b2 : ℝ := 2 * real.sqrt 2

  show focal_length a1 b1 = focal_length a2 b2
  sorry

end ellipses_same_focal_length_l601_601406


namespace smallest_a_has_50_perfect_squares_l601_601858

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601858


namespace distance_from_O_to_plane_ABC_l601_601499

-- Structure to hold the vertices of the tetrahedron
structure Tetrahedron :=
  (S A B C : Point)
  (base_ABC_isosceles_right : ∃ D, is_midpoint D A B ∧ dist A D = dist B D ∧ dist A B = 2)
  (SA_eq_SB_eq_SC : dist S A = 2 ∧ dist S B = 2 ∧ dist S C = 2)
  (circumcenter_O_exists : ∃ (O : Point), sphere O S A B C)

-- Lean theorem statement
theorem distance_from_O_to_plane_ABC :
  ∀ (T : Tetrahedron), 
  ∃ (O : Point), distance_from_O_to_plane T.O T.ABC = √3 / 3 :=
by 
  intros,
  sorry

end distance_from_O_to_plane_ABC_l601_601499


namespace polynomial_degree_is_5_l601_601817

def polynomial := 3 - 7*x^2 + (1/2)*x^5 + 4*real.sqrt 3 * x^5 - 11 

theorem polynomial_degree_is_5 : polynomial.natDegree = 5 := 
by 
  sorry

end polynomial_degree_is_5_l601_601817


namespace jury_selection_duration_is_two_l601_601483

variable (jury_selection_days : ℕ) (trial_days : ℕ) (deliberation_days : ℕ)

axiom trial_lasts_four_times_jury_selection : trial_days = 4 * jury_selection_days
axiom deliberation_is_six_full_days : deliberation_days = (6 * 24) / 16
axiom john_spends_nineteen_days : jury_selection_days + trial_days + deliberation_days = 19

theorem jury_selection_duration_is_two : jury_selection_days = 2 :=
by
  sorry

end jury_selection_duration_is_two_l601_601483


namespace fraction_of_shaded_area_l601_601752

def total_area (length width : ℕ) : ℕ := length * width
def quarter_area (area : ℕ) : ℕ := area / 4
def shaded_area (quarter : ℕ) : ℕ := quarter / 2

theorem fraction_of_shaded_area 
  (length : ℕ) (width : ℕ) 
  (total_area = length * width) 
  (quarter_area = total_area / 4) 
  (shaded_area = quarter_area / 2) :
  (shaded_area : ℚ) / total_area = 1 / 8 := 
sorry

end fraction_of_shaded_area_l601_601752


namespace trailing_zeroes_500_l601_601816

-- Define the function to count the number of trailing zeroes in n!
def trailing_zeroes (n : Nat) : Nat :=
  let rec count_factors (k : Nat) (acc : Nat) : Nat :=
    if k > n then acc
    else count_factors (k * 5) (acc + n / k)
  count_factors 5 0

-- Prove that the number of trailing zeroes in 500! is 124
theorem trailing_zeroes_500 : trailing_zeroes 500 = 124 :=
  by
  sorry

end trailing_zeroes_500_l601_601816


namespace base_from_wall_l601_601648

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601648


namespace geometric_sequence_condition_l601_601082

variable {a : ℕ → ℝ}

-- Definitions based on conditions in the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The statement translating the problem
theorem geometric_sequence_condition (q : ℝ) (a : ℕ → ℝ) (h : is_geometric_sequence a q) : ¬((q > 1) ↔ is_increasing_sequence a) :=
  sorry

end geometric_sequence_condition_l601_601082


namespace louie_took_home_pie_l601_601789

theorem louie_took_home_pie (h_pie_left: ℚ) (h_people: ℕ) (h_equal_split: h_pie_left = 12 / 13 ∧ h_people = 4):
  ∃ (x : ℚ), x = 3 / 13 :=
begin
  rcases h_equal_split with ⟨h_pie_left_eq, h_people_eq⟩,
  use (h_pie_left / h_people),
  have h_pie_left_val : h_pie_left = 12 / 13 := h_pie_left_eq,
  have h_people_val : h_people = 4 := h_people_eq,
  rw [←h_pie_left_val, ←h_people_val],
  norm_num,
  apply rat.ext,
  norm_num,
end

end louie_took_home_pie_l601_601789


namespace initial_bacteria_count_l601_601532

theorem initial_bacteria_count (doubling_time : ℕ) (initial_time : ℕ) (initial_bacteria : ℕ) 
(final_bacteria : ℕ) (doubling_rate : initial_time / doubling_time = 8 ∧ final_bacteria = 524288) : 
  initial_bacteria = 2048 :=
by
  sorry

end initial_bacteria_count_l601_601532


namespace min_value_of_f_l601_601394

noncomputable def f (x: ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem min_value_of_f : ∃ x : ℝ, f x = 3 / 2 ∧ ∀ y : ℝ, f y ≥ 3 / 2 :=
begin
  -- proof is not required
  sorry
end

end min_value_of_f_l601_601394


namespace star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l601_601389

def star (x y : ℤ) := (x + 2) * (y + 2) - 2

-- Statement A: commutativity
theorem star_comm : ∀ x y : ℤ, star x y = star y x := 
by sorry

-- Statement B: distributivity over addition
theorem star_distrib_over_add : ¬(∀ x y z : ℤ, star x (y + z) = star x y + star x z) :=
by sorry

-- Statement C: special case
theorem star_special_case : ¬(∀ x : ℤ, star (x - 2) (x + 2) = star x x - 2) :=
by sorry

-- Statement D: identity element
theorem star_no_identity : ¬(∃ e : ℤ, ∀ x : ℤ, star x e = x ∧ star e x = x) :=
by sorry

-- Statement E: associativity
theorem star_not_assoc : ¬(∀ x y z : ℤ, star (star x y) z = star x (star y z)) :=
by sorry

end star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l601_601389


namespace sin_sum_inequality_ABC_l601_601452

noncomputable theory

theorem sin_sum_inequality_ABC
  (A B C : ℝ)
  (h₁ : A + B + C = π)
  (h₂ : 0 < A ∧ A < π / 2)
  (h₃ : 0 < B ∧ B < π / 2)
  (h₄ : 0 < C ∧ C < π / 2) :
  sin (3 * A) + sin (3 * B) + sin (3 * C) ≤ (3 * Real.sqrt 3) / 2 := 
sorry

end sin_sum_inequality_ABC_l601_601452


namespace ladder_base_distance_l601_601653

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601653


namespace graph_of_equation_l601_601214

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 :=
by sorry

end graph_of_equation_l601_601214


namespace mnCardCondition_l601_601458

def distinct_remainders (m n : ℕ) (cards : Fin (m * n) → ℕ) : Prop :=
  ∀ i j k l : Fin (m * n), (i ≠ k ∨ j ≠ l) → (cards i + cards j) % (m * n) ≠ (cards k + cards l) % (m * n)

theorem mnCardCondition (m n : ℕ) (cards : Fin (m * n) → ℕ) : 
  distinct_remainders m n cards ↔ (n = 1 ∨ m = 1) :=
begin
  sorry
end

end mnCardCondition_l601_601458


namespace length_of_trapezoid_median_l601_601776

-- Define the conditions as variables
variable (h : ℝ) -- Altitude
variable (area : ℝ) -- Common area of both shapes

-- Given conditions in the problem
def triangle_base : ℝ := 24
def trapezoid_base : ℝ := triangle_base / 2

-- Theorem statement to prove that the median of the trapezoid is 12 inches
theorem length_of_trapezoid_median (h : ℝ) :
  (∃ (area : ℝ), (1/2) * triangle_base * h = area ∧ trapezoid_base * h = area) →
  (trapezoid_base = 12) :=
by
  sorry

end length_of_trapezoid_median_l601_601776


namespace problem_I_problem_II_l601_601398

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + a|

-- Problem (I)
theorem problem_I (x : ℝ) : f x 2 > 6 ↔ x ∈ Set.Ioo (-∞:ℝ) (-3) ∪ Set.Ioo 3 ∞ :=
sorry

-- Problem (II)
theorem problem_II (a : ℝ) : 
  (∃ x : ℝ, f x a < a^2 - 1) ↔ a ∈ Set.Ioo (-∞:ℝ) (-1 - Real.sqrt 2) ∪ Set.Ioo (1 + Real.sqrt 2) ∞ :=
sorry

end problem_I_problem_II_l601_601398


namespace angle_CDE_is_90_degrees_l601_601059

theorem angle_CDE_is_90_degrees
  (A B C D E : Type)
  [has_angle A] [has_angle B] [has_angle C] [has_angle D] [has_angle E]
  (angle_A_right : angle A = 90)
  (angle_B_right : angle B = 90)
  (angle_C_right : angle C = 90)
  (angle_AEB_45 : angle (AEB) = 45)
  (angle_BED_eq_BDE : angle (BED) = angle (BDE))
  : angle (CDE) = 90 :=
  by sorry

end angle_CDE_is_90_degrees_l601_601059


namespace smallest_integer_larger_than_sqrt5_plus_sqrt3_pow6_l601_601174

noncomputable def sqrt (x : ℝ) := x^(1/2)

theorem smallest_integer_larger_than_sqrt5_plus_sqrt3_pow6 :
  let expr := (sqrt 5 + sqrt 3)^6 in
  ∃ (n : ℤ), n = 3323 ∧ (n : ℝ) > expr ∧ ∀ m : ℤ, m > expr → m ≥ n :=
by {
  let expr := (sqrt 5 + sqrt 3)^6,
  have : 3323 = 3323 := rfl,
  have : (3323 : ℝ) > expr := by sorry,
  have : ∀ m : ℤ, m > expr → m ≥ 3323 := by sorry,
  exact ⟨3323, this, this, this⟩
}

end smallest_integer_larger_than_sqrt5_plus_sqrt3_pow6_l601_601174


namespace alicia_tax_correct_l601_601267

theorem alicia_tax_correct :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let basic_tax_rate := 0.01
  let additional_tax_rate := 0.0075
  let basic_tax := basic_tax_rate * hourly_wage_cents
  let excess_amount_cents := (hourly_wage_dollars - 20) * 100
  let additional_tax := additional_tax_rate * excess_amount_cents
  basic_tax + additional_tax = 28.75 := 
by
  sorry

end alicia_tax_correct_l601_601267


namespace janet_hours_exterminator_l601_601479

theorem janet_hours_exterminator (h : ℕ) : 
  let earnings_exterminator := 70 * h in
  let earnings_sculptures := 20 * 5 + 20 * 7 in
  let total_earnings := earnings_exterminator + earnings_sculptures in
  total_earnings = 1640 → h = 20 :=
by
  intros
  sorry

end janet_hours_exterminator_l601_601479


namespace marshmallows_needed_l601_601785

theorem marshmallows_needed (total_campers : ℕ) (fraction_boys : ℚ) (fraction_girls : ℚ)
  (boys_toast_fraction : ℚ) (girls_toast_fraction : ℚ) :
  total_campers = 96 →
  fraction_boys = 2 / 3 →
  fraction_girls = 1 / 3 →
  boys_toast_fraction = 1 / 2 →
  girls_toast_fraction = 3 / 4 →
  let boys := (fraction_boys * total_campers).natAbs,
      girls := (fraction_girls * total_campers).natAbs,
      boys_toast := (boys_toast_fraction * boys).natAbs,
      girls_toast := (girls_toast_fraction * girls).natAbs in
  boys_toast + girls_toast = 56 := by
sorrry

end marshmallows_needed_l601_601785


namespace simplify_expression_l601_601381

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) :
  ( ( (p - 7) / (9 - r) ) * ( (q - 8) / (7 - p) ) * ( (r - 9) / (8 - q) ) ) = -1 := 
by 
  sorry

end simplify_expression_l601_601381


namespace smallest_a_with_50_squares_l601_601904


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601904


namespace area_of_regular_inscribed_octagon_l601_601779

-- Definitions from conditions
def radius : ℝ := 3

-- Problem statement to prove
theorem area_of_regular_inscribed_octagon 
  (h : ∀ (n : ℕ), n = 8 → (radius : ℝ) = 3) :
  ∃ (A : ℝ), A = 18 * √2 := sorry

end area_of_regular_inscribed_octagon_l601_601779


namespace david_average_speed_l601_601222

theorem david_average_speed (d t : ℚ) (h1 : d = 49 / 3) (h2 : t = 7 / 3) :
  (d / t) = 7 :=
by
  rw [h1, h2]
  norm_num

end david_average_speed_l601_601222


namespace total_cups_l601_601142

theorem total_cups (b f s : ℕ) (ratio_bt_f_s : b / s = 1 / 5) (ratio_fl_b_s : f / s = 8 / 5) (sugar_cups : s = 10) :
  b + f + s = 28 :=
sorry

end total_cups_l601_601142


namespace f_neg_gt_1_f_decreasing_l601_601083

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (x y : ℝ) : f(x + y) = f(x) * f(y)
axiom f_pos_lt_1 (x : ℝ) : 0 < x → 0 < f(x) ∧ f(x) < 1

theorem f_neg_gt_1 {x : ℝ} (h : x < 0) : f(x) > 1 := sorry

theorem f_decreasing {x₁ x₂ : ℝ} (h : x₁ < x₂) : f(x₁) > f(x₂) := sorry

end f_neg_gt_1_f_decreasing_l601_601083


namespace part_1_part_2_part_3_l601_601939

section
variables (f g : ℝ → ℝ) (a b : ℝ) (n : ℕ)
variables (cond_f_tangent : ∀ (x : ℝ), f x = ln(1 + x) - a * x)
variables (cond_tangent_slope : ∀ (x : ℝ), x = -1/2 → deriv f x = 1)
variables (cond_f_g : ∀ x, g x = b * (exp x - x))
variables (cond_f_g_le : ∀ x, f x ≤ g x)
variables (cond_f_at_zero : f 0 = 0)
variables (cond_g_at_zero : g 0 = b)

theorem part_1 : a = 1 ∧ (∀ x, f(x) ≤ 0 ∧ (∀ x, f(x) ≤ f(0))) :=
  sorry

theorem part_2 : ∀ n : ℕ, 0 < n → 1 + (∑ i in finset.range n, (1/(i+1))) > real.log (n + 1) :=
  sorry

theorem part_3 : 0 ≤ b ∧ ∀ x, f x ≤ g x :=
  sorry

end

end part_1_part_2_part_3_l601_601939


namespace line_equation_parallel_with_tangent_l601_601513

def point := ℝ × ℝ

def curve (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

def derivative (x : ℝ) : ℝ := 6 * x - 4

noncomputable def equation_of_line (P M : point) (f : ℝ → ℝ) : ℝ × ℝ × ℝ :=
  let slope := derivative (fst M)
  let (x1, y1) := P
  let (x2, y2) := M
  (slope, -1, slope * (x1 + 1) - y1)

theorem line_equation_parallel_with_tangent
  (P M : point) (f : ℝ → ℝ)
  (hP : P = (-1, 2))
  (hM : M = (1, 1))
  (hf : ∀ x, f x = 3 * x^2 - 4 * x + 2)
  : equation_of_line P M f = (2, -1, 4) :=
by
  sorry

end line_equation_parallel_with_tangent_l601_601513


namespace find_water_amounts_l601_601165

-- Define the capacities of the containers
def capacity_larger : ℕ := 144
def capacity_smaller : ℕ := 100

-- Define the initial amounts of water in the containers
def initial_water_larger (x : ℕ) : Prop := x ≤ capacity_larger
def initial_water_smaller (y : ℕ) : Prop := y ≤ capacity_smaller

-- Define the conditions given in the problem as properties
def condition1 (x y : ℕ) : Prop := (x + (4 * y) / 5 = capacity_larger) ∧ ((1 * y) / 5 = initial_water_smaller (y / 5))
def condition2 (x y : ℕ) : Prop := (y + (5 * x) / 12 = capacity_smaller) ∧ ((7 * x) / 12 = initial_water_larger ((7 * x) / 12))

-- Define the main goal: finding x and y satisfying all conditions
theorem find_water_amounts (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 96 ∧ y = 60 := by
  sorry

end find_water_amounts_l601_601165


namespace range_of_a_l601_601969

def inequality_system_has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (x + a ≥ 0) ∧ (1 - 2 * x > x - 2)

theorem range_of_a (a : ℝ) : inequality_system_has_solution a ↔ a > -1 :=
by
  sorry

end range_of_a_l601_601969


namespace second_term_of_geometric_series_l601_601272

theorem second_term_of_geometric_series 
  (a : ℝ) (r : ℝ) (S : ℝ) :
  r = 1 / 4 → S = 40 → S = a / (1 - r) → a * r = 7.5 :=
by
  intros hr hS hSum
  sorry

end second_term_of_geometric_series_l601_601272


namespace average_value_log2_l601_601813

theorem average_value_log2 : 
  ∃ (M : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ set.Icc 1 (2^2018) ∧ x₂ ∈ set.Icc 1 (2^2018) ∧ 
    (f x₁ = real.log x₁ / real.log 2) ∧ (f x₂ = real.log x₂ / real.log 2) ∧ 
    (f x₁ + f x₂) / 2 = M) ↔ M = 1009 :=
begin
  sorry
end

end average_value_log2_l601_601813


namespace spherical_surface_area_in_tetrahedron_l601_601362

-- Define the parameters and conditions
variables (a : ℝ) (R : ℝ)

-- Define the radius R based on the edge length a of the tetrahedron
def radius_of_sphere_in_tetrahedron (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2

-- Define the area of the spherical surface located inside the tetrahedron with edge length a
def spherical_surface_area (a : ℝ) : ℝ :=
  let h := radius_of_sphere_in_tetrahedron a
  2 * Real.pi * h * (a * (Real.sqrt 2 / 2 - 1 / 2 * Real.sqrt (2 / 3)))

-- The theorem we want to state and eventually prove
theorem spherical_surface_area_in_tetrahedron (a : ℝ) :
  spherical_surface_area a = (Real.pi * a^2 / 6) * (2 * Real.sqrt 3 - 3):=
sorry

end spherical_surface_area_in_tetrahedron_l601_601362


namespace ladder_distance_from_wall_l601_601718

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601718


namespace angles_identity_l601_601375
open Real

theorem angles_identity (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2))
  (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end angles_identity_l601_601375


namespace ladder_base_distance_l601_601691

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601691


namespace magnitude_complex_expression_l601_601441

theorem magnitude_complex_expression 
  (z : ℂ) (hz : z = 1 + complex.i) :
  complex.abs (complex.i * z + 3 * complex.conj z) = 2 * real.sqrt 2 := by
sorry

end magnitude_complex_expression_l601_601441


namespace transformation_1_transformation_2_l601_601994

theorem transformation_1 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq1 : 5 * x + 2 * y = 0) : 
  5 * x' + 3 * y' = 0 := 
sorry

theorem transformation_2 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq2 : x^2 + y^2 = 1) : 
  4 * x' ^ 2 + 9 * y' ^ 2 = 1 := 
sorry

end transformation_1_transformation_2_l601_601994


namespace total_soccer_balls_donated_l601_601762

-- Defining the conditions
def elementary_classes_per_school := 4
def middle_classes_per_school := 5
def schools := 2
def soccer_balls_per_class := 5

-- Proving the total number of soccer balls donated
theorem total_soccer_balls_donated : 
  (soccer_balls_per_class * (schools * (elementary_classes_per_school + middle_classes_per_school))) = 90 := 
by
  -- Using the given conditions to compute the numbers
  let total_classes_per_school := elementary_classes_per_school + middle_classes_per_school
  let total_classes := total_classes_per_school * schools
  let total_soccer_balls := soccer_balls_per_class * total_classes
  -- Prove the equivalence
  show total_soccer_balls = 90
  from sorry

end total_soccer_balls_donated_l601_601762


namespace distinct_exponentiation_values_l601_601824

theorem distinct_exponentiation_values : 
  let a := 3^(3^(3^3))
  let b := 3^((3^3)^3)
  let c := ((3^3)^3)^3
  let d := 3^((3^3)^(3^2))
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → 
  ∃ n, n = 3 := 
sorry

end distinct_exponentiation_values_l601_601824


namespace diagonal_bisects_important_rectangles_l601_601766

theorem diagonal_bisects_important_rectangles (N : ℕ) : 
  let square := set.univ.prod set.univ : set (ℕ × ℕ),
    diagonals := finset.range (2 * N),
    D := {p : ℕ × ℕ | p.1 = p.2 ∧ p.1 < N},
  in
  2 * (Σ r in important_rectangles, finset.card r) = Σ r in set.univ, finset.card r :=
sorry

end diagonal_bisects_important_rectangles_l601_601766


namespace theater_ticket_area_l601_601258

theorem theater_ticket_area
  (P width : ℕ)
  (hP : P = 28)
  (hwidth : width = 6)
  (length : ℕ)
  (hlength : 2 * (length + width) = P) :
  length * width = 48 :=
by
  sorry

end theater_ticket_area_l601_601258


namespace keith_attended_games_l601_601566

-- Definitions based on the given conditions
def total_games : ℕ := 8
def missed_games : ℕ := 4

-- The proof goal: Keith's attendance
def attended_games : ℕ := total_games - missed_games

-- Main statement to prove the total games Keith attended
theorem keith_attended_games : attended_games = 4 := by
  -- Sorry is a placeholder for the proof
  sorry

end keith_attended_games_l601_601566


namespace quadratic_solution_l601_601525

theorem quadratic_solution (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
sorry

end quadratic_solution_l601_601525


namespace base_from_wall_l601_601640

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601640


namespace range_a_part1_range_a_part2_l601_601947

def A (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0
def B (x a : ℝ) : Prop := x = x^2 - 4*x + a
def C (x a : ℝ) : Prop := x^2 - a*x - 4 ≤ 0

def p (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a
def q (a : ℝ) : Prop := ∀ x : ℝ, A x → C x a

theorem range_a_part1 : ¬(p a) → a > 6 := sorry

theorem range_a_part2 : p a ∧ q a → 0 ≤ a ∧ a ≤ 6 := sorry

end range_a_part1_range_a_part2_l601_601947


namespace apartments_greater_than_scales_l601_601465

variables (K A P C : ℕ)

-- Define the conditions
def number_of_house := K
def number_of_apartments (house : number_of_house) := K
def number_of_aquariums (apartment : number_of_apartments house) := A
def number_of_fish (aquarium : number_of_aquariums apartment) := P
def number_of_scales (fish : number_of_fish aquarium) := C

-- Given conditions
axiom fish_in_each_house (h : number_of_house) : K * A * P > A * P * C

theorem apartments_greater_than_scales : K > C :=
by {
  sorry
}

end apartments_greater_than_scales_l601_601465


namespace athletes_meet_distance_l601_601162

theorem athletes_meet_distance
  (v1 v2 x : ℝ)
  (h1 : 300 / v1 = (x - 300) / v2)
  (h2 : (x + 100) / v1 = (x - 100) / v2) :
  x = 500 := 
begin
  -- proof steps go here
  sorry
end

end athletes_meet_distance_l601_601162


namespace distance_from_wall_l601_601626

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601626


namespace charge_Y_correct_l601_601842

def charge_X : ℝ := 1.25
def charge_Y : ℝ

theorem charge_Y_correct (h : 40 * charge_Y = 40 * charge_X + 60) : charge_Y = 2.75 :=
by
  have : 40 * charge_Y = 40 * 1.25 + 60 := h
  sorry

end charge_Y_correct_l601_601842


namespace limit_r_l601_601087

noncomputable def L (m : ℝ) : ℝ := (m - Real.sqrt (m^2 + 24)) / 2

noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem limit_r (h : ∀ m : ℝ, m ≠ 0) : Filter.Tendsto r (nhds 0) (nhds (-1)) :=
sorry

end limit_r_l601_601087


namespace ladder_distance_l601_601677

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601677


namespace percentage_of_girls_after_changes_l601_601983

theorem percentage_of_girls_after_changes :
  let boys_classA := 15
  let girls_classA := 20
  let boys_classB := 25
  let girls_classB := 35
  let boys_transferAtoB := 3
  let girls_transferAtoB := 2
  let boys_joiningA := 4
  let girls_joiningA := 6

  let boys_classA_after := boys_classA - boys_transferAtoB + boys_joiningA
  let girls_classA_after := girls_classA - girls_transferAtoB + girls_joiningA
  let boys_classB_after := boys_classB + boys_transferAtoB
  let girls_classB_after := girls_classB + girls_transferAtoB

  let total_students := boys_classA_after + girls_classA_after + boys_classB_after + girls_classB_after
  let total_girls := girls_classA_after + girls_classB_after 

  (total_girls / total_students : ℝ) * 100 = 58.095 := by
  sorry

end percentage_of_girls_after_changes_l601_601983


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601179

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601179


namespace perpendiculars_pass_through_single_point_l601_601823

open EuclideanGeometry

-- Definitions of geometric constructions:
variables {A B C M N K : Point}

def isEquilateral (P Q R : Point) : Prop := dist P Q = dist Q R ∧ dist Q R = dist R P

def isPerpendicular (l1 l2 : Line) : Prop := angle (direction l1) (direction l2) = π/2

def ninePointCenter (A B C : Point) : Point := sorry

theorem perpendiculars_pass_through_single_point 
  (h_eq1 : isEquilateral A M B)
  (h_eq2 : isEquilateral B N C)
  (h_eq3 : isEquilateral C K A)
  (M_mid : exists M_mid' : Point, ∀ p : Point, midpoint M N p ↔ p = M_mid')
  (N_mid : exists N_mid' : Point, ∀ p : Point, midpoint N K p ↔ p = N_mid')
  (K_mid : exists K_mid' : Point, ∀ p : Point, midpoint K M p ↔ p = K_mid')
  (h_perp_MN : ∃ l_CA : Line, ∀ q : Point, isPerpendicular l_CA (Line.mk q A) ∧ midpoint M N q)
  (h_perp_NK : ∃ l_AB : Line, ∀ s : Point, isPerpendicular l_AB (Line.mk s B) ∧ midpoint N K s)
  (h_perp_KM : ∃ l_BC : Line, ∀ t : Point, isPerpendicular l_BC (Line.mk t C) ∧ midpoint K M t) :
  ∃ P : Point, P = ninePointCenter A B C ∧ 
               ∀ m : Point, (isPerpendicular (Line.mk m P) (Line.mk M N) ∧ midpoint M N m) ∨
                            (isPerpendicular (Line.mk m P) (Line.mk N K) ∧ midpoint N K m) ∨
                            (isPerpendicular (Line.mk m P) (Line.mk K M) ∧ midpoint K M m) :=
sorry

end perpendiculars_pass_through_single_point_l601_601823


namespace magnitude_eq_2sqrt2_l601_601423

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601423


namespace range_m_intersect_l601_601940

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3*a*x - 1

def has_extremum_at (f : ℝ → ℝ) (c : ℝ) : Prop := ∃ a, by simp [f, deriv]; exact deriv f c = 0

theorem range_m_intersect
  (a x m : ℝ)
  (h1 : a ≠ 0)
  (h2 : has_extremum_at (λ x, x^3 - 3*a*x - 1) (-1))
  (h3 : ∃ (k : ℕ), (λ y, ∃ x, x^3 - 3*a*x - 1 = y) ⁻¹' ({m} : set ℝ) = k) :
  m ∈ Ioo (-3 : ℝ) (1 : ℝ) :=
sorry

end range_m_intersect_l601_601940


namespace smallest_natural_with_50_perfect_squares_l601_601880

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601880


namespace base_from_wall_l601_601639

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601639


namespace base_distance_l601_601623

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601623


namespace mean_of_remaining_four_numbers_l601_601120

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h1 : (a + b + c + d + 106) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.5 := 
sorry

end mean_of_remaining_four_numbers_l601_601120


namespace smallest_a_has_50_perfect_squares_l601_601861

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601861


namespace nat_representation_l601_601306

theorem nat_representation (k : ℕ) : ∃ n r : ℕ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ k = 3 * n + r :=
by
  sorry

end nat_representation_l601_601306


namespace maximum_triangle_area_l601_601581

-- Definitions based on given conditions
def triangle_sides (a b c : ℝ) : Prop :=
  0 < a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 2 ≤ c ∧ c ≤ 3

-- Function to calculate the area of a right triangle
def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Proposition stating the maximum area
theorem maximum_triangle_area {a b c : ℝ} (h : triangle_sides a b c) : 
  ∃ a b, 0 < a ∧ a ≤ 1 ∧ 1 ≤ b ∧ b ≤ 2 ∧ 
  (∀ c, 2 ≤ c ∧ c ≤ 3 → ∃ c, c = (Real.sqrt (a^2 + b^2))) ∧ 
  triangle_area a b = 1 :=
begin
  use [1, 2],
  split, linarith,
  split, linarith, 
  split, linarith, 
  split, linarith, 
  intros c hc,
  use sqrt (1 + 4),
  split, apply sqrt_nonneg,
  rw [one_mul, add_comm, sqrt_eq_rpow],
  exact @triangle_area 1 2 sorry,
end

end maximum_triangle_area_l601_601581


namespace find_a_10_l601_601559

variable (a : ℕ → ℕ)
variable (AS : ℕ → ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (n : ℕ)
variable (d : ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) := ∃ (d : ℕ), ∀ n, a(n+1) = a(n) + d

def S_n (a : ℕ → ℕ) (n : ℕ) := n + (n * (n-1)) / 2 * d
def sqrt_S_n_arithmetic (S : ℕ → ℕ) := ∀ n, ∃ b (m : ℕ), (sqrt S n = b n)

theorem find_a_10 
  (h1 : a 1 = 1)
  (h2 : is_arithmetic_sequence a)
  (h3 : sqrt_S_n_arithmetic S)
  : a 10 = 19
:= 
sorry

end find_a_10_l601_601559


namespace rhombus_area_correct_l601_601124

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 80 120 = 4800 :=
by 
  -- the proof is skipped by including sorry
  sorry

end rhombus_area_correct_l601_601124


namespace series_sum_gt_n_min_k_gt_2_pow_n_l601_601593

-- Part (a)
theorem series_sum_gt_n (n : ℕ) (h_pos : 0 < n) : 
  (∑ k in Finset.range (2*n) | k > 0, 1 / (2^k : ℝ)) > n := 
sorry

-- Part (b)
theorem min_k_gt_2_pow_n (n : ℕ) (h_pos : 0 < n) : 
  ∃ k : ℕ, k ≥ 2 ∧ (∑ i in Finset.range k | i > 1, 1 / (i : ℝ)) > n ∧ k > 2^n := 
sorry

end series_sum_gt_n_min_k_gt_2_pow_n_l601_601593


namespace line_equation_l601_601319

theorem line_equation {x y : ℝ} (h : (x = 1) ∧ (y = -3)) :
  ∃ c : ℝ, x - 2 * y + c = 0 ∧ c = 7 :=
by
  sorry

end line_equation_l601_601319


namespace smallest_natural_with_50_perfect_squares_l601_601882

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601882


namespace base_from_wall_l601_601646

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601646


namespace polar_to_cartesian_l601_601140

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * sin θ) : 
  ∃ x y : ℝ, x^2 + (y - 2)^2 = 4 ∧ x = ρ * cos θ ∧ y = ρ * sin θ := 
sorry

end polar_to_cartesian_l601_601140


namespace smallest_a_with_50_perfect_squares_l601_601891

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601891


namespace convert_to_rectangular_form_l601_601293

theorem convert_to_rectangular_form (θ : ℝ) (r : ℝ) (hθ : θ = 13 * Real.pi / 6) (hr : r = Real.sqrt 3) :
  r * Real.exp (Complex.I * θ) = (3 / 2 : ℝ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end convert_to_rectangular_form_l601_601293


namespace option_one_better_than_option_two_l601_601739

/-- Define the probability of winning in the first lottery option (drawing two red balls from a box
containing 4 red balls and 2 white balls). -/
def probability_option_one : ℚ := 2 / 5

/-- Define the probability of winning in the second lottery option (rolling two dice and having at least one die show a four). -/
def probability_option_two : ℚ := 11 / 36

/-- Prove that the probability of winning in the first lottery option is greater than the probability of winning in the second lottery option. -/
theorem option_one_better_than_option_two : probability_option_one > probability_option_two :=
by sorry

end option_one_better_than_option_two_l601_601739


namespace find_y_l601_601464

noncomputable def y_value (BD DC AE : ℝ) (h1 : BD = 4) (h2 : DC = 6) (h3 : AE = 3) : ℝ :=
let y := 4.5 in y

theorem find_y (BD DC AE y : ℝ) (hBD : BD = 4) (hDC : DC = 6) (hAE : AE = 3) (h_similar : ∃ (A B C D E : Type), 
  acute_triangle A B C ∧ 
  Altitude BD AC A B ∧ 
  Altitude AE BC A B ∧ 
  SegmentLength BD 4 ∧ 
  SegmentLength DC 6 ∧ 
  SegmentLength AE 3 ∧ 
  SegmentLength EB y ∧ 
  (similar_triangles (triangle A E B) (triangle B D C) ∧
  (Triangle BDC.right_angle ∧ Triangle AEB.right_angle ∧ ∠ABC = ∠BCD))) : 
  y = 4.5 :=
sorry

end find_y_l601_601464


namespace distance_from_wall_l601_601630

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601630


namespace constant_term_of_binomial_expansion_l601_601535

theorem constant_term_of_binomial_expansion :
  let f := (2 * x - 1 / x)^6 in
  (constant_term f) = -160 :=
sorry

end constant_term_of_binomial_expansion_l601_601535


namespace curve_and_line_properties_l601_601390

theorem curve_and_line_properties :
  (∀ θ, let ρ := 4 * cos θ / sin θ ^ 2 in y = ρ * sin θ ∧ x = ρ * cos θ)
  ∧ (∃ α t, 0 ≤ α ∧ α < π ∧ x = t * cos α ∧ y = 1 + t * sin α ∧ 1 = t * cos α ∧ 0 = 1 + t * sin α)
  → (∀ x y, y^2 = 4 * x)
  ∧ (∃ t₁ t₂, abs (t₁ - t₂) = 8) :=
by
  intros
  sorry

end curve_and_line_properties_l601_601390


namespace num_integers_same_remainder_l601_601297

theorem num_integers_same_remainder (n : ℤ) : 
  (150 < n ∧ n < 250 ∧ n % 7 = n % 9) → (∃ l, l.card = 12 ∧ ∀ i ∈ l, 150 < i ∧ i < 250 ∧ i % 7 = i % 9) :=
begin
  sorry
end

end num_integers_same_remainder_l601_601297


namespace a_and_b_together_finish_in_40_days_l601_601219

theorem a_and_b_together_finish_in_40_days (D : ℕ) 
    (W : ℕ)
    (day_with_b : ℕ)
    (remaining_days_a : ℕ)
    (a_alone_days : ℕ)
    (a_b_together : D = 40)
    (ha : (remaining_days_a = 15) ∧ (a_alone_days = 20) ∧ (day_with_b = 10))
    (work_done_total : 10 * (W / D) + 15 * (W / a_alone_days) = W) :
    D = 40 := 
    sorry

end a_and_b_together_finish_in_40_days_l601_601219


namespace percent_palindromes_contain_7_l601_601758

def palindrome (n : ℕ) : Prop :=
  n.toString.reverse = n.toString

def is_between (n : ℕ) (a b : ℕ) : Prop :=
  a ≤ n ∧ n ≤ b

def contains_seven (n : ℕ) : Prop :=
  ('7' : Char) ∈ n.toString.toList

def three_digit_palindrome (n : ℕ) : Prop :=
  palindrome n ∧ 100 ≤ n ∧ n < 1000

def count_palindromes (f : ℕ → Prop) : ℕ :=
  Finset.card (Finset.filter f (Finset.range 1000))

def percentage (subset_count total_count : ℕ) : ℚ :=
  (subset_count : ℚ) / (total_count : ℚ) * 100

theorem percent_palindromes_contain_7 :
  percentage (count_palindromes (λ n, three_digit_palindrome n ∧ contains_seven n))
             (count_palindromes three_digit_palindrome) = 10 := 
sorry

end percent_palindromes_contain_7_l601_601758


namespace smallest_fraction_l601_601215

theorem smallest_fraction :
  let a := 5/12
  let b := 7/17
  let c := 20/41
  let d := 125/252
  let e := 155/312
  b < a ∧ b < c ∧ b < d ∧ b < e :=
begin
  sorry
end

end smallest_fraction_l601_601215


namespace months_decreasing_l601_601509

noncomputable def stock_decrease (m : ℕ) : Prop :=
  2 * m + 2 * 8 = 18

theorem months_decreasing (m : ℕ) (h : stock_decrease m) : m = 1 :=
by
  exact sorry

end months_decreasing_l601_601509


namespace ConstructPrism_l601_601285

noncomputable theory
open_locale classical

-- Definition of the midpoints as given points in the conditions
variables {A A₁ B C C₁ B₁ K L M N P Q : Point}

-- Given conditions: images of midpoints are provided
def MidpointAA1 := Midpoint(A, A₁) = K
def MidpointBC := Midpoint(B, C) = L
def MidpointCC1 := Midpoint(C, C₁) = M
def MidpointA1C1 := Midpoint(A₁, C₁) = N

-- Defining the problem statement
theorem ConstructPrism (h1 : MidpointAA1)
                       (h2 : MidpointBC)
                       (h3 : MidpointCC1)
                       (h4 : MidpointA1C1)
                       (h5 : Midpoint(K, M) = P)
                       (h6 : PQ = PN ∧ Extend(NP, PQ) = Q)
                       (h7 : ParallelThrough(Q, N, KM) ∧ ParallelThrough(K, M, QN))
                       (h8 : Extend(CL, LB) = B)
                       (h9 : ParallelThrough(A₁, B, Q) ∧ ParallelThrough(C₁, C, B₁)): 
  PrismVertices(A, B, C, A₁, B₁, C₁) :=
sorry

end ConstructPrism_l601_601285


namespace measure_angle_ABC_l601_601052

-- Define vectors in space
def vector_AB : ℝ × ℝ × ℝ := (2, 4, 0)
def vector_BC : ℝ × ℝ × ℝ := (-1, 3, 0)

-- Define the angle calculation function
noncomputable def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
  let magnitude_v1 := Real.sqrt (v1.1^2 + v1.2^2 + v1.3^2)
  let magnitude_v2 := Real.sqrt (v2.1^2 + v2.2^2 + v2.3^2)
  dot_product / (magnitude_v1 * magnitude_v2)

-- Prove that the measure of angle ABC is 135°
theorem measure_angle_ABC : (Real.arccos (cos_angle (-vector_AB) vector_BC)) = Real.pi * (3 / 4) :=
by {
  -- Vector definitions
  let vec_BA := (-vector_AB.1, -vector_AB.2, -vector_AB.3),
  -- dot product calculation
  let dot_product := vec_BA.1 * vector_BC.1 + vec_BA.2 * vector_BC.2 + vec_BA.3 * vector_BC.3,
  let magnitude_BA := Real.sqrt (vec_BA.1^2 + vec_BA.2^2 + vec_BA.3^2),
  let magnitude_BC := Real.sqrt (vector_BC.1^2 + vector_BC.2^2 + vector_BC.3^2),
  let cos_angle := dot_product / (magnitude_BA * magnitude_BC),
  -- Solving for the angle
  have h_cos_angle : cos_angle = -Real.sqrt(2) / 2, sorry,
  rw h_cos_angle,
  norm_num
}

end measure_angle_ABC_l601_601052


namespace pentagon_perimeter_l601_601172

-- Define the points and the distances between them as per the problem conditions.
def point : Type := ℝ × ℝ

def F : point := (0, 0)
def G : point := (0, -2)
def H : point := (2, -2)
def I : point := (2 + 1 / sqrt(2), -2 + 1 / sqrt(2))
def J : point := (0 + 1.5, 0 - 1.5)

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def FG : ℝ := 2
def GH : ℝ := 2
def HI : ℝ := sqrt 2
def IJ : ℝ := sqrt 4.5
def FJ : ℝ := sqrt 4.5

def perimeter : ℝ := FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : perimeter = 4 + 4 * sqrt 2 := by
  sorry

end pentagon_perimeter_l601_601172


namespace ladder_base_distance_l601_601664

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601664


namespace sphere_surface_area_l601_601970

theorem sphere_surface_area 
  (a b c : ℝ) 
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = 2)
  (h_spherical_condition : ∃ R : ℝ, ∀ (x y z : ℝ), x^2 + y^2 + z^2 = (2 * R)^2) :
  4 * Real.pi * ((3 / 2)^2) = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_l601_601970


namespace distance_between_trees_l601_601453

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ)
  (h_yard : yard_length = 400) (h_trees : num_trees = 26) : 
  (yard_length / (num_trees - 1)) = 16 :=
by
  sorry

end distance_between_trees_l601_601453


namespace two_digit_primes_count_from_set_l601_601414

open Finset Nat

def digit_set : Finset ℕ :=
  {2, 3, 7, 8, 9} 

def is_two_digit_prime_formed (tens units : ℕ) : Prop :=
  tens ≠ units ∧
  tens ∈ digit_set ∧
  units ∈ digit_set ∧
  prime (10 * tens + units)

theorem two_digit_primes_count_from_set :
  card (filter (λ (n : ℕ), ∃ tens units, is_two_digit_prime_formed tens units ∧ n = 10 * tens + units)
    (Ico 10 100)) = 7 :=
sorry

end two_digit_primes_count_from_set_l601_601414


namespace intersection_condition_l601_601944

-- Define the function f(x) = x^2 + x - 2
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the piecewise function g(x)
def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0
  else -f x

theorem intersection_condition (a b : ℝ) (h_a : 0 < a) (h_a_ub : a < 3) :
  (2 * a < b) ∧ (b < (1/4) * (a + 1)^2 + 2) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
  g x1 = a * x1 + b ∧ g x2 = a * x2 + b ∧ g x3 = a * x3 + b :=
sorry

end intersection_condition_l601_601944


namespace find_the_number_l601_601229

theorem find_the_number (n : ℚ) (h : 8 * n - 4 = 17) : n = 21 / 8 :=
by 
suffices : n = 21 / 8,
  rw this,
  norm_num,
  repeat {sorry}

end find_the_number_l601_601229


namespace find_third_circle_radius_l601_601163

-- Define the context of circles and their tangency properties
variable (A B : ℝ → ℝ → Prop) -- Centers of circles
variable (r1 r2 : ℝ) -- Radii of circles

-- Define conditions from the problem
def circles_are_tangent (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) : Prop :=
  ∀ x y : ℝ, A x y → B (x + 7) y ∧ r1 = 2 ∧ r2 = 5

def third_circle_tangent_to_others_and_tangent_line (A B : ℝ → ℝ → Prop) (r3 : ℝ) : Prop :=
  ∃ D : ℝ → ℝ → Prop, ∀ x y : ℝ, D x y →
  ((A (x + r3) y ∧ B (x - r3) y) ∧ (r3 > 0))

theorem find_third_circle_radius (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) :
  circles_are_tangent A B r1 r2 →
  (∃ r3 : ℝ, r3 = 1 ∧ third_circle_tangent_to_others_and_tangent_line A B r3) :=
by
  sorry

end find_third_circle_radius_l601_601163


namespace problem_1_problem_2_l601_601945

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h_symmetry : ∃ k : ℤ, a = k * Real.pi / 2) : g (2 * a) = 1 / 2 := by
  sorry

-- Proof Problem 2
theorem problem_2 (x : ℝ) (h_range : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  ∃ y : ℝ, y = h x ∧ 1/2 ≤ y ∧ y ≤ 2 := by
  sorry

end problem_1_problem_2_l601_601945


namespace sum_of_interior_numbers_l601_601300

def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

theorem sum_of_interior_numbers :
  sum_interior 8 + sum_interior 9 + sum_interior 10 = 890 :=
by
  sorry

end sum_of_interior_numbers_l601_601300


namespace ladder_distance_from_wall_l601_601711

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601711


namespace ratio_of_areas_l601_601123

variables {Point : Type*} [AffineSpace ℝ Point]

-- Define the points A, B, C, D, E, K, L, M, N
variables (A B C D E K L M N : Point)

-- Define the areas of triangles
variables (S_ABC S_ADC S_KLMN : ℝ)

-- Conditions
variable (h1 : collinear A C D ∧ collinear B C D)  -- AC and BD intersect at E
variable (h2 : ∃ E : Point, line_through E A ∧ line_through E B)
variable (h3 : S_ABC = S_ADC)
variable (h4 : parallel_through E A D K ∧ parallel_through E D C L ∧ 
                parallel_through E C B M ∧ parallel_through E B A N)

-- Define areas 
variable (h5 : S_ABC > 0 ∧ S_ADC > 0 ∧ S_KLMN > 0)
variable (h6 : area_eq_ratio S_KLMN S_ABC)

-- The ratio
theorem ratio_of_areas :
  S_KLMN / S_ABC = 1 :=
sorry

end ratio_of_areas_l601_601123


namespace min_b_for_factorization_l601_601836

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l601_601836


namespace triangle_properties_l601_601477

theorem triangle_properties (A B C a b c : ℝ) (h1 : a * Real.tan C = 2 * c * Real.sin A)
  (h2 : C > 0 ∧ C < Real.pi)
  (h3 : a / Real.sin A = c / Real.sin C) :
  C = Real.pi / 3 ∧ (1 / 2 < Real.sin (A + Real.pi / 6) ∧ Real.sin (A + Real.pi / 6) ≤ 1) →
  (Real.sqrt 3 / 2 < Real.sin A + Real.sin B ∧ Real.sin A + Real.sin B ≤ Real.sqrt 3) :=
by
  intro h4
  sorry

end triangle_properties_l601_601477


namespace magnitude_expression_l601_601434

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601434


namespace smallest_a_has_50_perfect_squares_l601_601872

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601872


namespace fraction_zero_implies_x_equals_2_l601_601022

theorem fraction_zero_implies_x_equals_2 (x : ℝ) :
  (x - 2 = 0) ∧ (x + 5 ≠ 0) → (x = 2) := 
by
  intro h
  cases h with numerator_zero denominator_not_zero
  -- Proof goes here
  sorry

end fraction_zero_implies_x_equals_2_l601_601022


namespace first_column_digit_l601_601570

def is_three_digit_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

theorem first_column_digit (m n : ℕ) (h3m : is_three_digit_number (3 ^ m))
  (h7n : is_three_digit_number (7 ^ n)) : (∃ d : ℕ, first_digit (3 ^ m) = d ∧ first_digit (7 ^ n) = d) ↔ d = 3 := 
sorry

-- Helper function to extract the first digit of a number
def first_digit (x : ℕ) : ℕ :=
  x / (10 ^ (x.digits 10).length.pred)

end first_column_digit_l601_601570


namespace magnitude_eq_2sqrt2_l601_601420

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601420


namespace simplify_sqrt_expression_l601_601113

variable (a : ℝ) (h : 0 ≤ a)

theorem simplify_sqrt_expression : (sqrt (a ^ (1 / 2) * sqrt (a ^ (1 / 2) * sqrt a))) = a ^ (1 / 2) :=
by
  sorry

end simplify_sqrt_expression_l601_601113


namespace sum_of_solutions_l601_601520

theorem sum_of_solutions : 
  (∃ x, 3^(x^2 + 6*x + 9) = 27^(x + 3)) → (∀ x₁ x₂, (3^(x₁^2 + 6*x₁ + 9) = 27^(x₁ + 3) ∧ 3^(x₂^2 + 6*x₂ + 9) = 27^(x₂ + 3)) → x₁ + x₂ = -3) :=
sorry

end sum_of_solutions_l601_601520


namespace magnitude_eq_2sqrt2_l601_601422

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601422


namespace no_solution_tan_eq_neg_tan_tan_l601_601412

noncomputable theory

open Real

theorem no_solution_tan_eq_neg_tan_tan (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ arctan 500) :
  ¬ (tan x = -tan (tan x)) :=
by sorry

end no_solution_tan_eq_neg_tan_tan_l601_601412


namespace seq_is_integer_sequence_two_seq_mul_plus_one_is_perfect_square_l601_601145

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  if n = 2 then 4 else seq (n - 1) * seq (n + 1) + 1

theorem seq_is_integer_sequence : ∀ n : ℕ, ∃ k : ℕ, seq n = k :=
sorry

theorem two_seq_mul_plus_one_is_perfect_square : ∀ n : ℕ, ∃ k : ℕ, 2 * seq n * seq (n + 1) + 1 = k * k :=
sorry

end seq_is_integer_sequence_two_seq_mul_plus_one_is_perfect_square_l601_601145


namespace find_parabola_equation_l601_601935

def parabola (p : ℝ) : Prop := ∃ (y : ℝ), y^2 = 2 * p * 3

theorem find_parabola_equation (p : ℝ) (hp : p > 0) (hM : parabola p)
  (hMF : |3 + p / 2| = 2 * p) : 2 * p = 4 :=
by
  sorry

# To verify that hMF leads to the equation y^2 = 4*x, we need to prove p = 2
  sorry

end find_parabola_equation_l601_601935


namespace sum_of_x_coords_of_A_l601_601160

open Real

-- Definitions identified from the conditions
def B := (0 : ℝ, 0 : ℝ)
def C := (200 : ℝ, 0 : ℝ)
def D := (600 : ℝ, 400 : ℝ)
def E := (610 : ℝ, 410 : ℝ)

def area_ABC := 3000
def area_ADE := 6000

-- The Lean statement of the proof problem
theorem sum_of_x_coords_of_A :
  ∃ (A : ℝ × ℝ), (∑ (a ∈ (A)), a.fst) = 400 ∧
    (∃ a b : ℝ, A = (a, b) ∧
      (area_ABC = 3000 ∧ area_ADE = 6000 ∧
        B = (0, 0) ∧ C = (200, 0) ∧ D = (600, 400) ∧ E = (610, 410))) :=
sorry

end sum_of_x_coords_of_A_l601_601160


namespace base_from_wall_l601_601644

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601644


namespace min_magnitude_l601_601951

noncomputable def cos_deg (d : ℝ) : ℝ := real.cos (d * real.pi / 180)
noncomputable def sin_deg (d : ℝ) : ℝ := real.sin (d * real.pi / 180)

def vector_a := (cos_deg 25, sin_deg 25)
def vector_b := (sin_deg 20, cos_deg 20)
def vector_u (t : ℝ) := (vector_a.1 + t * vector_b.1, vector_a.2 + t * vector_b.2)

def magnitude_squared (p : ℝ × ℝ) : ℝ := p.1 * p.1 + p.2 * p.2

theorem min_magnitude : ∃ t : ℝ, magnitude_squared (vector_u t) = (real.sqrt 2 / 2) :=
sorry

end min_magnitude_l601_601951


namespace sum_of_digits_of_t_l601_601085

theorem sum_of_digits_of_t (n : ℕ) (k : ℕ) (h1 : n > 0)
  (h2 : ∀ n, (∑ i in List.range (n+1), (Nat.floor (i / 5) + Nat.floor (i / 25)) = k))
  (h3 : ∀ n, (∑ i in List.range ((3*n)+1), (Nat.floor (i / 5) + Nat.floor (i / 25)) = 4 * k)) :
  (digits (10 + 11 + 15 + 16)).sum = 7 := by
    sorry


end sum_of_digits_of_t_l601_601085


namespace find_number_eq_seven_point_five_l601_601197

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601197


namespace magnitude_eq_2sqrt2_l601_601421

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601421


namespace bishop_placement_count_l601_601074

/-- Define the positions of the kings and the condition of the bishops not checking each other. -/
def whiteKing : (ℕ × ℕ) := (1, 1)
def blackKing : (ℕ × ℕ) := (4, 4)

def bishop_not_check (bishop : (ℕ × ℕ)) (king : (ℕ × ℕ)) : Prop :=
  bishop.1 ≠ king.1 ∧ bishop.2 ≠ king.2 ∧ abs (bishop.1 - king.1) ≠ abs (bishop.2 - king.2)

/-- Define the chessboard dimensions and valid positions for placing the bishops. -/
def board_size : ℕ := 4
def valid_positions : List (ℕ × ℕ) := 
  List.product (List.range board_size) (List.range board_size) 
  |>.filter (λ p, p ≠ whiteKing ∧ p ≠ blackKing)

/-- Count the number of valid bishop configurations. -/
def valid_bishop_configurations : ℕ :=
  (valid_positions.filter (λ b, bishop_not_check b blackKing)).length *
  (valid_positions.filter (λ w, bishop_not_check w whiteKing)).length

/-- Main theorem: Count the total number of valid configurations for the bishops given the conditions. -/
theorem bishop_placement_count : valid_bishop_configurations = 876 := 
sorry

end bishop_placement_count_l601_601074


namespace estimated_values_correct_l601_601737

def total_sample_mean (m n : ℕ) (x y : ℝ) : ℝ :=
  (m * x + n * y) / (m + n)

def total_sample_variance (m n : ℕ) (s1 s2 : ℝ) (x y ω : ℝ) : ℝ :=
  (m / (m + n : ℝ)) * (s1 + (x - ω) ^ 2) + (n / (m + n : ℝ)) * (s2 + (y - ω) ^ 2)

theorem estimated_values_correct :
  ∀ (m n : ℕ) (x y s1 s2 : ℝ),
    m = 100 ∧ n = 100 ∧ x = 170 ∧ y = 160 ∧ s1 = 22 ∧ s2 = 38 →
    total_sample_mean m n x y = 165 ∧ total_sample_variance m n s1 s2 x y 165 = 55 :=
by {
  intros m n x y s1 s2 h,
  cases h with h1 h,
  cases h with h2 h,
  cases h with h3 h,
  cases h with h4 h,
  cases h with h5 h6,
  have h_mean : total_sample_mean m n x y = 165 := by {
    dsimp [total_sample_mean],
    rw [h1, h2, h3, h4],
    norm_num,
  },
  have h_variance : total_sample_variance m n s1 s2 x y 165 = 55 := by {
    dsimp [total_sample_variance],
    rw [h1, h2, h3, h4, h5, h6],
    norm_num,
  },
  exact ⟨h_mean, h_variance⟩,
}

end estimated_values_correct_l601_601737


namespace largest_prime_factor_of_T_l601_601283

theorem largest_prime_factor_of_T (T : ℕ) : 
  (∀ (s₁ s₂ : list ℕ), 
    s₁ = [312, 123, 231] →
    s₂ = [231, 312, 123] →
    T = list.sum s₁ + list.sum s₂) →
  nat.prime 37 ∧ ∃ p, nat.prime p ∧ p > 37 ∧ p ∣ T → false :=
by
  sorry

end largest_prime_factor_of_T_l601_601283


namespace amount_left_for_gas_and_maintenance_l601_601487

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l601_601487


namespace circles_intersect_range_l601_601141

def circle1_radius := 3
def circle2_radius := 5

theorem circles_intersect_range : 2 < d ∧ d < 8 :=
by
  let r1 := circle1_radius
  let r2 := circle2_radius
  have h1 : d > r2 - r1 := sorry
  have h2 : d < r2 + r1 := sorry
  exact ⟨h1, h2⟩

end circles_intersect_range_l601_601141


namespace voltage_relationship_l601_601952

variables (x y z : ℝ) -- Coordinates representing positions on the lines
variables (I R U : ℝ) -- Representing current, resistance, and voltage respectively

-- Conditions translated into Lean
def I_def := I = 10^x
def R_def := R = 10^(-2 * y)
def U_def := U = 10^(-z)
def coord_relation := x + z = 2 * y

-- The final theorem to prove V = I * R under given conditions
theorem voltage_relationship : I = 10^x → R = 10^(-2 * y) → U = 10^(-z) → (x + z = 2 * y) → U = I * R :=
by 
  intros hI hR hU hXYZ
  sorry

end voltage_relationship_l601_601952


namespace find_magnitude_of_angle_find_value_of_a_l601_601025

variables {A B C : ℝ} {a b c : ℝ} {S_ABC : ℝ} 
variables (A_eq_pi_div_3 : A = π / 3) (b_eq_3 : b = 3) (S_ABC_eq_3sqrt3 : S_ABC = 3 * real.sqrt 3)
variables (vec_a : ℝ × ℝ) (vec_b : ℝ × ℝ)
variables (cos_A : ℝ → ℝ) (cos_B : ℝ → ℝ) (sin_C : ℝ) (sin_B : ℝ) (sin_A : ℝ → ℝ)
variables (parallel : vec_a = vec_b)

def calc_angle_A : Prop :=
  vec_a = (cos A, cos B) ∧
  vec_b = (a, 2 * c - b) ∧
  parallel ∧
  A = π / 3

def calc_side_a : Prop :=
  vec_a = (cos A, cos B) ∧
  vec_b = (a, 2 * c - b) ∧
  parallel ∧
  b = 3 ∧
  S_ABC = 3 * real.sqrt 3 ∧
  A = π / 3 ∧
  a = real.sqrt 13

theorem find_magnitude_of_angle (h : calc_angle_A) : A = π / 3 :=
  sorry

theorem find_value_of_a (h : calc_side_a) : a = real.sqrt 13 :=
  sorry

end find_magnitude_of_angle_find_value_of_a_l601_601025


namespace bug_total_crawl_distance_l601_601234

theorem bug_total_crawl_distance :
  let start := 3
  let point1 := -4
  let point2 := 7
  let end := -1
  abs (point1 - start) + abs (point2 - point1) + abs (end - point2) = 26 :=
by
  let start := 3
  let point1 := -4
  let point2 := 7
  let end := -1
  have h1 : abs (point1 - start) = abs (-4 - 3) := rfl
  have h2 : abs (point2 - point1) = abs (7 - -4) := rfl
  have h3 : abs (end - point2) = abs (-1 - 7) := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end bug_total_crawl_distance_l601_601234


namespace no_solution_ggx_eq_3_l601_601811

def g (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x^2 - 2 * x else 2 * x - 4

theorem no_solution_ggx_eq_3 :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → g (g x) ≠ 3 :=
begin
  sorry
end

end no_solution_ggx_eq_3_l601_601811


namespace geometric_sequence_a3_l601_601060

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 4 = 8)
  (h3 : ∀ k : ℕ, a (k + 1) = a k * q) : a 3 = 4 :=
sorry

end geometric_sequence_a3_l601_601060


namespace percent_typing_used_l601_601220

-- Definitions based on conditions
def sheet_width : ℕ := 20
def sheet_length : ℕ := 30
def margin_side : ℕ := 2
def margin_top_bottom : ℕ := 3

-- Derived calculations based on conditions
def typing_width : ℕ := sheet_width - 2 * margin_side
def typing_length : ℕ := sheet_length - 2 * margin_top_bottom
def area_typing : ℕ := typing_width * typing_length
def area_total : ℕ := sheet_width * sheet_length

-- Statement to prove
theorem percent_typing_used : 
  (area_typing.to_rat / area_total.to_rat) * 100 = 64 := by
  sorry

end percent_typing_used_l601_601220


namespace function_decreasing_l601_601915

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3 * a
  else a ^ x

theorem function_decreasing : ∀ (a : ℝ), (0 < a) ∧ (a ≠ 1) →
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) ↔ (a ∈ set.Ico (1/3) 1) :=
by
  -- Proof should go here.
  sorry

end function_decreasing_l601_601915


namespace smallest_a_with_50_squares_l601_601902


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601902


namespace ladder_base_distance_l601_601658

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601658


namespace cost_price_first_batch_min_selling_price_second_batch_l601_601237

variable (x y : ℕ)
variable (first_batch_cost : ℕ := 4000)
variable (second_batch_cost : ℕ := 5400)
variable (quantity_ratio : ℚ := 1.5)
variable (price_difference : ℕ := 5)
variable (first_batch_selling_price : ℕ := 70)
variable (desired_profit : ℕ := 4060)

theorem cost_price_first_batch (h : (5400 / (x - 5) : ℚ) = 1.5 * (4000 / x)) : x = 50 := 
sorry

theorem min_selling_price_second_batch 
  (h : (first_batch_selling_price - 50) * (4000 / 50) + (y - 45) * (5400 / 45) ≥ desired_profit)
  (h_int : y ∈ ℤ) : y ≥ 66 := 
sorry

end cost_price_first_batch_min_selling_price_second_batch_l601_601237


namespace inequality_proof_l601_601515

theorem inequality_proof 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 > 0) 
  (h2 : a2 > 0) 
  (h3 : a3 > 0)
  (h4 : a4 > 0):
  (a1 + a3) / (a1 + a2) + 
  (a2 + a4) / (a2 + a3) + 
  (a3 + a1) / (a3 + a4) + 
  (a4 + a2) / (a4 + a1) ≥ 4 :=
by
  sorry

end inequality_proof_l601_601515


namespace journey_second_part_speed_l601_601014

theorem journey_second_part_speed
  (S1 : ℕ) (D : ℕ) (T : ℕ) (t1 : ℕ)
  (hS1 : S1 = 40)
  (hD : D = 240)
  (hT : T = 5)
  (ht1 : t1 = 3) :
  let d1 := S1 * t1 in -- distance of the first part
  let d2 := D - d1 in -- distance of the second part
  let t2 := T - t1 in -- time of the second part
  d2 / t2 = 60 :=
by
  intros
  sorry

end journey_second_part_speed_l601_601014


namespace parallelogram_diagonals_lengths_find_t_value_l601_601056

structure Point2D where
  x : ℝ
  y : ℝ

def vectorSub (v1 v2 : Point2D) : Point2D :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

def vectorAdd (v1 v2 : Point2D) : Point2D :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

def dotProduct (v1 v2 : Point2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

def magnitude (v : Point2D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2)

noncomputable def t_solution : ℝ :=
  -11 / 5

theorem parallelogram_diagonals_lengths :
    let A := Point2D.mk (-1) (-2)
    let B := Point2D.mk 2 3
    let C := Point2D.mk (-2) (-1)
    let AB := Point2D.mk 3 5
    let AC := Point2D.mk (-1) 1
    let OC := Point2D.mk (-2) (-1)
    magnitude (vectorAdd AB AC) = 2 * Real.sqrt 10 ∧
    magnitude (vectorSub AB AC) = 4 * Real.sqrt 2 :=
  by
    sorry

theorem find_t_value :
    let A := Point2D.mk (-1) (-2)
    let B := Point2D.mk 2 3
    let C := Point2D.mk (-2) (-1)
    let AB := Point2D.mk 3 5
    let OC := Point2D.mk (-2) (-1)
    dotProduct (vectorSub AB (vectorScalarMul t_solution OC)) OC = 0 ∧
    t_solution = -11 / 5 :=
  by
    sorry

end parallelogram_diagonals_lengths_find_t_value_l601_601056


namespace heat_engine_efficiency_l601_601245

theorem heat_engine_efficiency :
  ∀ (η : ℚ) (T1 T2 T3 P1 P2 P3 V1 V2 V3 : ℚ),
    (η = (1 - Real.log 2) / 5) * 100 ∧
    T2 = 2 * T1 ∧
    T3 = T1 ∧
    T1 > 0 ∧
    P1 > 0 ∧ P2 > 0 ∧ P3 > 0 ∧
    V1 > 0 ∧ V2 > 0 ∧ V3 > 0  →
    η = 6.14 := sorry

end heat_engine_efficiency_l601_601245


namespace watermelon_cost_is_100_rubles_l601_601743

theorem watermelon_cost_is_100_rubles :
  (∀ (x y k m n : ℕ) (a b : ℝ),
    x + y = k →
    n * a = m * b →
    n * a + m * b = 24000 →
    n = 120 →
    m = 30 →
    k = 150 →
    a = 100) :=
by
  intros x y k m n a b
  intros h1 h2 h3 h4 h5 h6
  have h7 : 120 * a = 30 * b, from h2
  have h8 : 120 * a + 30 * b = 24000, from h3
  have h9 : 120 * a = 12000, from sorry
  have h10 : a = 100, from sorry
  exact h10

end watermelon_cost_is_100_rubles_l601_601743


namespace find_number_divided_by_3_equals_subtracted_5_l601_601206

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601206


namespace amount_left_for_gas_and_maintenance_l601_601486

def monthly_income : ℤ := 3200
def rent : ℤ := 1250
def utilities : ℤ := 150
def retirement_savings : ℤ := 400
def groceries_eating_out : ℤ := 300
def insurance : ℤ := 200
def miscellaneous : ℤ := 200
def car_payment : ℤ := 350

def total_expenses : ℤ :=
  rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment

theorem amount_left_for_gas_and_maintenance : monthly_income - total_expenses = 350 :=
by
  -- Proof is omitted
  sorry

end amount_left_for_gas_and_maintenance_l601_601486


namespace distance_from_wall_l601_601628

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601628


namespace tan_XOY_l601_601255

-- declare conditions
def circle_centered_at_O (O : Point) (C : Circle) : Prop := C.center = O
def square_and_triangle_inscribed (C : Circle) (s : Square) (t : Triangle) : Prop :=
  inscribed s C ∧ inscribed t C
def AB_parallel_QR (s : Square) (t : Triangle) : Prop :=
  s.AB ∥ t.QR
def sides_PQ_PR_meet_AB_at (t : Triangle) (s : Square) (X Y : Point) : Prop :=
  meet_at t.PQ s.AB X ∧ meet_at t.PR s.AB Y

-- declare the statement
theorem tan_XOY (s : Square) (t : Triangle) (C : Circle) (O : Point) (X Y : Point)
  (h1 : circle_centered_at_O O C) 
  (h2 : square_and_triangle_inscribed C s t)
  (h3 : AB_parallel_QR s t)
  (h4 : sides_PQ_PR_meet_AB_at t s X Y) :
  tan (∠ X O Y) = (sqrt 6 - sqrt 3) / sqrt 2 := sorry

end tan_XOY_l601_601255


namespace age_problem_l601_601756

theorem age_problem
  (D M : ℕ)
  (h1 : M = D + 45)
  (h2 : M - 5 = 6 * (D - 5)) :
  D = 14 ∧ M = 59 := by
  sorry

end age_problem_l601_601756


namespace smallest_a_has_50_perfect_squares_l601_601875

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601875


namespace triangle_equilateral_if_arithmetic_sequences_l601_601996

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences_l601_601996


namespace triangle_area_equality_l601_601753

theorem triangle_area_equality (A B C D X Y Z T : ℝ×ℝ)
  (h_square : is_square A B C D)
  (h_intersect : is_intersection A B X ∧ is_intersection B C Y ∧ 
                 is_extension_intersection A D Z ∧ is_extension_intersection C D T)
  (h_parallel : is_parallel A B C D)
  (h_transversal : is_transversal Z T A B C D) :
  area_triangle C X Z = area_triangle A Y T := by
srry

end triangle_area_equality_l601_601753


namespace find_smallest_a_l601_601851

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601851


namespace smallest_natural_number_with_50_squares_in_interval_l601_601863

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601863


namespace find_number_l601_601192

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601192


namespace parabola_point_value_k_l601_601759

theorem parabola_point_value_k :
  let point_of_intersection := (4, -1)
  let x_intercepts_line1 := (3, 0)
  let x_intercepts_line2 := (6, 0)
  let parabola := λ x, (1 / 2) * (x - 3) * (x - 6)
  -- Condition
  (parabola 10 = 14) := sorry

end parabola_point_value_k_l601_601759


namespace find_a_from_conditions_l601_601295

-- Define the Cartesian equation of curve C
def cartesian_equation_of_curve (a : ℝ) : Prop :=
  ∀ (x y : ℝ), y^2 = 2 * a * x

-- Define the standard equation of line l
def standard_equation_of_line : Prop :=
  ∀ (x y : ℝ), y = x - 2

-- Main theorem: Given conditions, show a = 1
theorem find_a_from_conditions (a : ℝ) (a_pos : a > 0) :
  (∀ (ρ θ : ℝ), ρ * sin (2 * θ) = 2 * a * cos θ) →
  (∃ (P : ℝ × ℝ), P = (-2, -4) ∧ standard_equation_of_line) →
  (|PA| × |PB| = |AB|^2) →
  a = 1 :=
by
  sorry

end find_a_from_conditions_l601_601295


namespace y_coord_is_13_div_3_l601_601578

noncomputable def y_coord_equidistant (y : ℝ) : Prop :=
    dist (0, y) (3, 0) = dist (0, y) (5, 6)

theorem y_coord_is_13_div_3 : ∃ y : ℝ, y_coord_equidistant y ∧ y = 13 / 3 :=
by
  use (13 / 3)
  split
  all_goals sorry

end y_coord_is_13_div_3_l601_601578


namespace Alex_sandwich_count_l601_601265

theorem Alex_sandwich_count :
  let meats := 10
  let cheeses := 9
  let sandwiches := meats * (cheeses.choose 2)
  sandwiches = 360 :=
by
  -- Here start your proof
  sorry

end Alex_sandwich_count_l601_601265


namespace base_from_wall_l601_601643

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601643


namespace hannah_card_sum_l601_601953

theorem hannah_card_sum :
  ∀ y : ℝ, (90 < y ∧ y < 180) →
    ∃ (sin_y : ℝ) (cos_y : ℝ) (tan_y : ℝ),
      sin_y = Real.sin y ∧
      cos_y = Real.cos y ∧
      tan_y = Real.tan y ∧
      (∀ a b c : ℝ, a = sin_y ∨ a = cos_y ∨ a = tan_y ∧ 
        b = sin_y ∨ b = cos_y ∨ b = tan_y ∧ 
        c = sin_y ∨ c = cos_y ∨ c = tan_y → 
        ∃ onlyAliceIdentifies : ℝ, 
          onlyAliceIdentifies = sin_y ∧
          ∀ z : ℝ, (90 < z ∧ z < 180) →
             Real.sin z ∈ {1}) :=
sorry

end hannah_card_sum_l601_601953


namespace equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601033

section
variable [decidable_eq ℕ]
variable (deck_size: ℕ := 32)
variable (num_aces: ℕ := 4)
variable (players: Π (i: fin 4), ℕ := λ i, 1)
variable [uniform_dist: Probability_Mass_Function (fin deck_size)] 

-- Part (a): Probabilities for each player to get the first Ace
noncomputable def player1_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player2_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player3_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player4_prob (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_of_first_ace :
  player1_prob deck = 1/8 ∧
  player2_prob deck = 1/8 ∧
  player3_prob deck = 1/8 ∧
  player4_prob deck = 1/8 :=
sorry

-- Part (b): Modify rules to deal until Ace of Spades
noncomputable def player_prob_ace_of_spades (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_with_ace_of_spades :
  ∀(p: fin 4), player_prob_ace_of_spades deck = 1/4 :=
sorry
end

end equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601033


namespace find_number_l601_601188

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601188


namespace pts_on_circ_l601_601472

theorem pts_on_circ (A B C A₁ B₁ P T : Point) (hABC_right : is_right_triangle A B C)
  (h_inc_A₁ : incircle_touch_BC_at_A₁ A B C) (h_inc_B₁ : incircle_touch_AC_at_B₁ A B C)
  (h_midline_PT : midline_parallel_to_AB_intersects_circumcircle PT)
  (hA1 : pt_on_line A₁ BC) (hB1 : pt_on_line B₁ AC) :
  cyclic_points P T A₁ B₁ :=
sorry

end pts_on_circ_l601_601472


namespace find_ff_of_five_half_l601_601938

noncomputable def f (x : ℝ) : ℝ :=
if x <= 1 then 2^x - 2 else Real.log x / Real.log 2

theorem find_ff_of_five_half : f (f (5/2)) = -1/2 := by
  sorry

end find_ff_of_five_half_l601_601938


namespace sqrt_condition_sqrt_not_meaningful_2_l601_601021

theorem sqrt_condition (x : ℝ) : 1 - x ≥ 0 ↔ x ≤ 1 := 
by
  sorry

theorem sqrt_not_meaningful_2 : ¬(1 - 2 ≥ 0) :=
by
  sorry

end sqrt_condition_sqrt_not_meaningful_2_l601_601021


namespace equal_prob_first_ace_l601_601039

theorem equal_prob_first_ace (deck : List ℕ) (players : Fin 4) (h_deck_size : deck.length = 32)
  (h_distinct : deck.nodup) (h_aces : ∀ _i, deck.filter (λ card, card = 1 ).length = 4)
  (h_shuffled : ∀ (card : ℕ), card ∈ deck → card ∈ (range 32)) :
  ∀ (player : Fin 4), let positions := List.range' (player + 1) (32 / 4) * 4 + player;
  (∀ (pos : ℕ), pos ∈ positions → deck.nth pos = some 1) →
  P(player) = 1 / 8 :=
by
  sorry

end equal_prob_first_ace_l601_601039


namespace ladder_distance_from_wall_l601_601716

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601716


namespace ladder_base_distance_l601_601695

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601695


namespace max_value_4x_plus_y_l601_601497

theorem max_value_4x_plus_y (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (u : ℝ), (∃ (x y : ℝ), 16 * x^2 + y^2 + 4 * x * y = 3 ∧ u = 4 * x + y) → u ≤ M :=
by
  use 2
  sorry

end max_value_4x_plus_y_l601_601497


namespace find_smallest_a_l601_601848

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601848


namespace ladder_base_distance_l601_601723

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601723


namespace find_smallest_a_l601_601849

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601849


namespace sector_area_angle_1_sector_max_area_l601_601937

-- The definition and conditions
variable (c : ℝ) (r l : ℝ)

-- 1) Proof that the area of the sector when the central angle is 1 radian is c^2 / 18
-- given 2r + l = c
theorem sector_area_angle_1 (h : 2 * r + l = c) (h1: l = r) :
  (1/2 * l * r = (c^2 / 18)) :=
by sorry

-- 2) Proof that the central angle that maximizes the area is 2 radians and the maximum area is c^2 / 16
-- given 2r + l = c
theorem sector_max_area (h : 2 * r + l = c) :
  ∃ l r, 2 * r = l ∧ 1/2 * l * r = (c^2 / 16) :=
by sorry

end sector_area_angle_1_sector_max_area_l601_601937


namespace cards_max_l601_601480

theorem cards_max (budget : ℝ) (price1 : ℝ) (price2 : ℝ) (discount_threshold : ℕ) :
  budget = 10 ∧ price1 = 0.75 ∧ price2 = 0.70 ∧ discount_threshold = 10 →
  ∃ n : ℕ, n ≤ 13 ∧
          (n ≤ discount_threshold → price1 * n ≤ budget) ∧
          (n > discount_threshold → 7.5 + price2 * (n - discount_threshold) ≤ budget) :=
begin
  sorry
end

end cards_max_l601_601480


namespace fraction_red_surface_area_eq_3_over_4_l601_601240

-- Define the larger cube made of smaller cubes.
structure Cube :=
  (side_length : ℕ)
  (num_cubes : ℕ)

-- Define the color distribution.
structure ColorDistribution :=
  (red_cubes : ℕ)
  (blue_cubes : ℕ)

-- Conditions
def larger_cube : Cube := ⟨4, 64⟩
def color_dist : ColorDistribution := ⟨32, 32⟩
def blue_per_face : ℕ := 4

-- Theorem statement
theorem fraction_red_surface_area_eq_3_over_4 :
  let total_surface_area := 6 * (larger_cube.side_length ^ 2)
  let blue_faces := blue_per_face * 6
  let red_faces := total_surface_area - blue_faces in
  (red_faces : ℚ) / (total_surface_area : ℚ) = 3 / 4 := by
  sorry

end fraction_red_surface_area_eq_3_over_4_l601_601240


namespace min_cost_correct_l601_601844

noncomputable def min_cost_to_feed_group : ℕ :=
  let main_courses := 50
  let salads := 30
  let soups := 15
  let price_salad := 200
  let price_soup_main := 350
  let price_salad_main := 350
  let price_all_three := 500
  17000

theorem min_cost_correct : min_cost_to_feed_group = 17000 :=
by
  sorry

end min_cost_correct_l601_601844


namespace perimeter_of_rectangle_abcdef_677_l601_601110

-- Definitions of rhombus and rectangle
variables (A B C D P Q R S : Type)
variables [AddCommGroup A] [AffineAddGroup A]

/-- Rhombus PQRS inscribed in rectangle ABCD -/
def inscribed_rhombus_in_rectangle (A B C D P Q R S : A) :=
  affine_add_subgroup.right_dist_eq_left_dist Q R A D ∧
  affine_add_subgroup.right_dist_eq_left_dist P S B C ∧
  affine_add_subgroup.right_dist_eq_left_dist R S A B ∧
  affine_add_subgroup.right_dist_eq_left_dist P Q C D ∧
  dist P B = 15 ∧
  dist B Q = 20 ∧
  dist P R = 30 ∧
  dist Q S = 40

/-- Perimeter of a rectangle -/
def perimeter_rectangle (A B C D : A) : ℝ :=
  2 * (dist A B + dist B C)

-- Main theorem statement
theorem perimeter_of_rectangle_abcdef_677 (A B C D P Q R S : A)
  (h_inscribed : inscribed_rhombus_in_rectangle A B C D P Q R S)
  (h_rect : ∃ m n, lowest_terms (perimeter_rectangle A B C D) m n ∧ m + n = 677) :
  (perimeter_rectangle A B C D) = 134.4 :=
begin
  sorry
end

/- 
Note:
dist denotes the distance function.
affine_add_subgroup.right_dist_eq_left_dist is an auxiliary definition used to describe the geometric properties.
lowest_terms is a hypothetical function denoting the numerator and denominator in lowest terms.
-/

end perimeter_of_rectangle_abcdef_677_l601_601110


namespace find_x_l601_601962

theorem find_x : ∃ x : ℝ, (1 / 8) * 2^36 = 4^x ∧ x = 16.5 :=
by {
  sorry,
}

end find_x_l601_601962


namespace ab_times_65_eq_48ab_l601_601957

theorem ab_times_65_eq_48ab (a b : ℕ) (h_ab : 0 ≤ a ∧ a < 10) (h_b : 0 ≤ b ∧ b < 10) :
  (10 * a + b) * 65 = 4800 + 10 * a + b ↔ 10 * a + b = 75 := by
sorry

end ab_times_65_eq_48ab_l601_601957


namespace ladder_base_distance_l601_601687

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601687


namespace find_b12_l601_601494

noncomputable def sequence_b : ℕ → ℤ
| 0 := 0
| 1 := 2
| (n + 2) := (sequence_b (n + 1)) + (sequence_b 1) + (n + 1)^2 + 1^2

theorem find_b12 : sequence_b 12 = 160 := by
  sorry

end find_b12_l601_601494


namespace collinear_points_sum_l601_601302

theorem collinear_points_sum (p q : ℝ) (h1 : 2 = p) (h2 : q = 4) : p + q = 6 :=
by 
  rw [h1, h2]
  sorry

end collinear_points_sum_l601_601302


namespace quadrilateral_area_l601_601579

-- Defining points P, Q, R, S and their respective distances
variable (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]

-- Defining specific distances
variables (PS SR PR PQ RQ : ℝ)
variables (h1 : PS = 3)
variables (h2 : SR = 4)
variables (h3 : PR = 5)
variables (h4 : PQ = 13)
variables (h5 : RQ = 12)

-- Defining right angles in triangles PSR and PRQ
variables (h6 : right_angle (angle P S R))
variables (h7 : right_angle (angle P R Q))

theorem quadrilateral_area :
  let area_triangle := fun a b c : ℝ => 0.5 * a * b in
  let area_PSR := area_triangle PS SR in
  let area_PRQ := area_triangle PR RQ in
  area_PSR + area_PRQ = 36 :=
by
  sorry

end quadrilateral_area_l601_601579


namespace solve_for_b_l601_601543

theorem solve_for_b (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x + 6 = 0 ↔ y = (2 / 3) * x - 2) → 
  (∀ x y, 4 * y + b * x + 3 = 0 ↔ y = -(b / 4) * x - 3 / 4) → 
  (∀ m1 m2, (m1 = (2 / 3)) → (m2 = -(b / 4)) → m1 * m2 = -1) → 
  b = 6 :=
sorry

end solve_for_b_l601_601543


namespace locus_of_M_l601_601577

/-- We draw a circle through the foci of an ellipse, which touches a tangent of the ellipse at an arbitrary point M. The locus of point M as the tangent continuously changes its position are the tangents to the ellipse passing through the vertices B and B'. -/
theorem locus_of_M (e : Ellipse) (t : Tangent e) (M : Point) 
  (h : M ∈ t ∧ t.IsTangentAt M) : 
  ∃ B B' : Point, is_tangent_through_vertices B B' ∧ M ∈ tangent_lines_through B B' :=
sorry

end locus_of_M_l601_601577


namespace find_number_eq_seven_point_five_l601_601203

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601203


namespace lunks_for_apples_l601_601416

theorem lunks_for_apples : 
  (∀ (a : ℕ) (b : ℕ) (k : ℕ), 3 * b * k = 5 * a → 15 * k = 9 * a ∧ 2 * a * 9 = 4 * b * 9 → 15 * 2 * a / 4 = 18) :=
by
  intro a b k h1 h2
  sorry

end lunks_for_apples_l601_601416


namespace PM_and_radius_correct_l601_601026

noncomputable def length_of_PM (PQ QR PR : ℝ) (h1 : PQ = 40) (h2 : QR = 40) (h3 : PR = 38) : ℝ :=
  sorry

noncomputable def radius_of_tangent_circle (PQ QR PR : ℝ) (h1 : PQ = 40) (h2 : QR = 40) (h3 : PR = 38) : ℝ :=
  sorry

theorem PM_and_radius_correct :
  let PQ := 40
  let QR := 40
  let PR := 38
  length_of_PM PQ QR PR (by rfl) (by rfl) (by rfl) = (3 / 2) * sqrt 498.67 ∧
  radius_of_tangent_circle PQ QR PR (by rfl) (by rfl) (by rfl) = (20 * sqrt 1239) / 59 :=
sorry

end PM_and_radius_correct_l601_601026


namespace perimeter_of_triangle_ABC_l601_601989

-- Definitions and Conditions
def is_isosceles (A B C : Type) [metric_space A] [metric_space B] [metric_space C] (AB AC BC : ℝ) : Prop :=
  AB = AC

variables (AB BC : ℝ)
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

-- Given Conditions
axiom h1 : is_isosceles A B C 6 6 8
axiom h2 : AB = 6
axiom h3 : BC = 8

-- Statement to prove
theorem perimeter_of_triangle_ABC : AB + BC + AB = 20 :=
by {
  rw [h2, h3],
  sorry
}

end perimeter_of_triangle_ABC_l601_601989


namespace speaking_orders_l601_601152

theorem speaking_orders (A : Type) (B : Type) (C : Type) (D : Type) (E : Type) (F : Type) :
  let positions := (1 : ℕ) :: (2 :: 3 :: 4 :: 5 :: 6 :: []) in
  let valid_positions := (2 :: 3 :: 4 :: 5 :: []) in
  -- number of positions for A to be placed
  let num_positions_for_A := valid_positions.length in
  -- number of permutations of the remaining 5 contestants
  let num_permutations_of_remaining := (Finset.univ : Finset (Fin 5)).card.factorial in
  -- total number of different speaking orders
  let total_orders := num_positions_for_A * num_permutations_of_remaining in
  total_orders = 480 :=
by
  sorry

end speaking_orders_l601_601152


namespace tap_B_fills_remaining_pool_l601_601778

theorem tap_B_fills_remaining_pool :
  ∀ (flow_A flow_B : ℝ) (t_A t_B : ℕ),
  flow_A = 7.5 / 100 →  -- A fills 7.5% of the pool per hour
  flow_B = 5 / 100 →    -- B fills 5% of the pool per hour
  t_A = 2 →             -- A is open for 2 hours during the second phase
  t_A * flow_A = 15 / 100 →  -- A fills 15% of the pool in 2 hours
  4 * (flow_A + flow_B) = 50 / 100 →  -- A and B together fill 50% of the pool in 4 hours
  (100 / 100 - 50 / 100 - 15 / 100) / flow_B = t_B →  -- remaining pool filled only by B
  t_B = 7 := sorry    -- Prove that t_B is 7

end tap_B_fills_remaining_pool_l601_601778


namespace quadratic_roots_inequality_l601_601943

theorem quadratic_roots_inequality (m : ℝ) :
  let f : ℝ → ℝ := λ x, x^2 + 2 * (m - 1) * x - 5 * m - 2 in
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 < 1 ∧ x2 > 1) → m > 1 :=
sorry

end quadratic_roots_inequality_l601_601943


namespace marshmallows_needed_l601_601788

def number_of_campers := 96
def fraction_of_boys := 2 / 3
def fraction_of_girls := 1 / 3
def percentage_boys_toast := 0.5
def percentage_girls_toast := 0.75

theorem marshmallows_needed : 
  let boys := fraction_of_boys * number_of_campers in
  let girls := fraction_of_girls * number_of_campers in
  let boys_toast := percentage_boys_toast * boys in
  let girls_toast := percentage_girls_toast * girls in
  boys_toast + girls_toast = 56 :=
by
  sorry

end marshmallows_needed_l601_601788


namespace ladder_distance_l601_601683

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601683


namespace chris_pennies_count_l601_601961

theorem chris_pennies_count (a c : ℤ) 
  (h1 : c + 2 = 4 * (a - 2)) 
  (h2 : c - 2 = 3 * (a + 2)) : 
  c = 62 := 
by 
  -- The actual proof is omitted
  sorry

end chris_pennies_count_l601_601961


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601181

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601181


namespace days_per_book_l601_601346

theorem days_per_book (total_books : ℕ) (total_days : ℕ)
  (h1 : total_books = 41)
  (h2 : total_days = 492) :
  total_days / total_books = 12 :=
by
  -- proof goes here
  sorry

end days_per_book_l601_601346


namespace smallest_natural_with_50_perfect_squares_l601_601878

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601878


namespace lifting_ratio_after_gain_l601_601794

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l601_601794


namespace pencils_per_box_l601_601309

theorem pencils_per_box (boxes : ℕ) (total_pencils : ℕ) (h1 : boxes = 3) (h2 : total_pencils = 27) : (total_pencils / boxes) = 9 := 
by
  sorry

end pencils_per_box_l601_601309


namespace truck_and_trailer_total_weight_l601_601260

def truck_weight : ℝ := 4800
def trailer_weight (truck_weight : ℝ) : ℝ := 0.5 * truck_weight - 200
def total_weight (truck_weight trailer_weight : ℝ) : ℝ := truck_weight + trailer_weight 

theorem truck_and_trailer_total_weight : 
  total_weight truck_weight (trailer_weight truck_weight) = 7000 :=
by 
  sorry

end truck_and_trailer_total_weight_l601_601260


namespace moles_of_ca_oh_2_l601_601325

-- Define the chemical reaction
def ca_o := 1
def h_2_o := 1
def ca_oh_2 := ca_o + h_2_o

-- Prove the result of the reaction
theorem moles_of_ca_oh_2 :
  ca_oh_2 = 1 := by sorry

end moles_of_ca_oh_2_l601_601325


namespace ladder_base_distance_l601_601609

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601609


namespace marshmallows_needed_l601_601787

def number_of_campers := 96
def fraction_of_boys := 2 / 3
def fraction_of_girls := 1 / 3
def percentage_boys_toast := 0.5
def percentage_girls_toast := 0.75

theorem marshmallows_needed : 
  let boys := fraction_of_boys * number_of_campers in
  let girls := fraction_of_girls * number_of_campers in
  let boys_toast := percentage_boys_toast * boys in
  let girls_toast := percentage_girls_toast * girls in
  boys_toast + girls_toast = 56 :=
by
  sorry

end marshmallows_needed_l601_601787


namespace abs_diff_60th_terms_arithmetic_sequences_l601_601575

theorem abs_diff_60th_terms_arithmetic_sequences :
  let C : (ℕ → ℤ) := λ n => 25 + 15 * (n - 1)
  let D : (ℕ → ℤ) := λ n => 40 - 15 * (n - 1)
  |C 60 - D 60| = 1755 :=
by
  sorry

end abs_diff_60th_terms_arithmetic_sequences_l601_601575


namespace distance_from_wall_l601_601627

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601627


namespace distance_from_wall_l601_601629

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601629


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601430

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601430


namespace num_students_earning_B_l601_601977

variables (nA nB nC nF : ℕ)

-- Conditions from the problem
def condition1 := nA = 6 * nB / 10
def condition2 := nC = 15 * nB / 10
def condition3 := nF = 4 * nB / 10
def condition4 := nA + nB + nC + nF = 50

-- The theorem to prove
theorem num_students_earning_B (nA nB nC nF : ℕ) : 
  condition1 nA nB → 
  condition2 nC nB → 
  condition3 nF nB → 
  condition4 nA nB nC nF → 
  nB = 14 :=
by
  sorry

end num_students_earning_B_l601_601977


namespace base_distance_l601_601620

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601620


namespace lattice_points_inside_region_l601_601247

theorem lattice_points_inside_region :
  ∃ (points : Finset (ℤ × ℤ)), (∀ p ∈ points, let (x, y) := p in (y < |x| ∧ y < -x^2 + 5) ∧ (y ≠ |x| ∧ y ≠ -x^2 + 5)) ∧ points.card = 4 :=
sorry

end lattice_points_inside_region_l601_601247


namespace max_value_f_l601_601549

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + x + 1)

theorem max_value_f : ∀ x : ℝ, f x ≤ 4 / 3 :=
sorry

end max_value_f_l601_601549


namespace fabric_needed_for_coats_l601_601347

variable (m d : ℝ)

def condition1 := 4 * m + 2 * d = 16
def condition2 := 2 * m + 6 * d = 18

theorem fabric_needed_for_coats (h1 : condition1 m d) (h2 : condition2 m d) :
  m = 3 ∧ d = 2 :=
by
  sorry

end fabric_needed_for_coats_l601_601347


namespace base_distance_l601_601624

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601624


namespace B_inv_3_correct_l601_601351

open Matrix

-- Define the given matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := ![
  [3, 7],
  [-2, -4]
]

-- Define the expected inverse of B^3
def B_inv_3_expected : Matrix (Fin 2) (Fin 2) ℤ := ![
  [11, 17],
  [-10, -18]
]

-- The theorem to prove
theorem B_inv_3_correct : (B_inv^3) = B_inv_3_expected := by
  sorry

end B_inv_3_correct_l601_601351


namespace cauchy_problem_solution_exists_l601_601331

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y' x - (1/x) * y x = - (2 / x^2) ∧ y 1 = 1

theorem cauchy_problem_solution_exists :
  ∃ y : ℝ → ℝ, solution y = λ x, 1 / x :=
sorry

end cauchy_problem_solution_exists_l601_601331


namespace smallest_natural_number_with_50_squares_in_interval_l601_601864

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601864


namespace solve_sqrt_equation_l601_601828

theorem solve_sqrt_equation : 
  ∀ z : ℝ, (sqrt (7 - 5 * z) = 10) ↔ (z = -93 / 5) :=
by
  intro z
  sorry

end solve_sqrt_equation_l601_601828


namespace equilateral_triangle_rotation_l601_601365

theorem equilateral_triangle_rotation (a p q r : ℝ) (h1 : a > 0) (h2 : p > 0) (h3 : q > 0) (h4 : r > 0)
  (h5 : ∃ (T : Type) [equilateral_triangle T] (P : T), distances_from_point P [p, q, r]) :
  ∃ (T' : Type) [equilateral_triangle T'] (Q : T'), distances_from_point Q [q, r, p] :=
sorry

end equilateral_triangle_rotation_l601_601365


namespace ratio_of_projections_l601_601086

variable (Point : Type)

variable [MetricSpace Point] [InnerProductSpace ℝ Point]

open_locale real_inner_product_space

variables (A B C D M E F : Point)

variable [Parallelogram : Parallelogram A B C D]

-- Orthogonal projections E and F
variable [ProjectionE : OrthogonalProjection M A B E]
variable [ProjectionF : OrthogonalProjection M A D F]

theorem ratio_of_projections (hParallelogram : Parallelogram A B C D)
    (hME : OrthogonalProjection M A B E)
    (hMF : OrthogonalProjection M A D F) :
  dist M E / dist M F = dist A D / dist A B := 
sorry

end ratio_of_projections_l601_601086


namespace baron_munchausen_not_deceiving_l601_601358

/-- Given 8 weights, each with distinct integer masses from 1 to 8, and an unknown weight-mass correspondence,
 prove that it is possible with one weighing to uniquely identify the mass of at least one weight. --/
theorem baron_munchausen_not_deceiving :
  ∃ a b c d e f g h : ℕ,
    {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8} ∧
    (∃ S T : finset ℕ, S ∪ T ∪ {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8} ∧
      S ≠ T ∧
      S.sum ≠ T.sum) :=
sorry

end baron_munchausen_not_deceiving_l601_601358


namespace ladder_distance_l601_601679

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601679


namespace inequality_amgm_l601_601005

theorem inequality_amgm (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^3 / (b^2 - 1)) + (b^3 / (c^2 - 1)) + (c^3 / (a^2 - 1)) ≥ (9 * Real.sqrt 3) / 2 :=
begin
  sorry,
end

end inequality_amgm_l601_601005


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601431

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601431


namespace ladder_distance_from_wall_l601_601715

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601715


namespace triangle_perimeter_eq_fourteen_l601_601373

theorem triangle_perimeter_eq_fourteen {m x : ℝ}
  (h_root : x^2 - 2 * m * x + 3 * m = 0)
  (h_isosceles : (x = 2) ∨ (x = 6)) :
  let a := if x = 2 then 6 else 2,
      b := x
  in a + a + b = 14 :=
by
  -- Given the conditions
  sorry

end triangle_perimeter_eq_fourteen_l601_601373


namespace kyle_gas_and_maintenance_expense_l601_601485

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l601_601485


namespace relationship_among_abc_l601_601360

noncomputable def f (x : ℝ) : ℝ := 2^|x| - 1

-- We define the specific values for a, b, and c
def a : ℝ := f (Real.log 3 / Real.log 0.5)
def b : ℝ := f (Real.log 5 / Real.log 2)
def c : ℝ := f (Real.log (1 / 4) / Real.log 2)

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l601_601360


namespace coin_count_l601_601561

theorem coin_count (x y : ℕ) 
  (h1 : x + y = 12) 
  (h2 : 5 * x + 10 * y = 90) :
  x = 6 ∧ y = 6 := 
sorry

end coin_count_l601_601561


namespace octagon_area_inscribed_in_circle_l601_601781

theorem octagon_area_inscribed_in_circle (r : ℝ) (oct_area : ℝ) :
  r = 3 → oct_area = 18 * real.sqrt 2 :=
by
  intro hr
  rw hr
  sorry

end octagon_area_inscribed_in_circle_l601_601781


namespace ladder_distance_l601_601685

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601685


namespace curve_implicit_equation_line_inclination_angle_distance_sum_PA_PB_l601_601055

noncomputable theory

open Real

def curve_parametric (α : ℝ) : ℝ × ℝ := ⟨3 * cos α, sin α⟩

theorem curve_implicit_equation :
    ∀ (α : ℝ),
    let ⟨x, y⟩ := curve_parametric α in
    (x^2) / 9 + y^2 = 1 := sorry

def polar_eq_line_l (ρ θ : ℝ) : Prop :=
    ρ * sin (θ - π / 4) = sqrt 2

theorem line_inclination_angle (ρ θ : ℝ) :
    polar_eq_line_l ρ θ → θ = π / 4 + arctan ((sqrt 2 / 2) / (sqrt 2 / 2)) := sorry

def point := ℝ × ℝ

def intersects (ℓ : point → Prop) (C : point → Prop) : set point :=
    {p | ℓ p ∧ C p}

def line_l (t : ℝ) : point := (sqrt 2 / 2 * t, 2 + sqrt 2 / 2 * t)

def curve_C (p : point) : Prop := (p.1^2) / 9 + p.2^2 = 1

theorem distance_sum_PA_PB :
    let P := (0, 2)
    let l := λ p : point, polar_eq_line_l (sqrt (p.1^2 + p.2^2)) (arctan (p.2 / p.1))
    let A_and_B := intersects l curve_C in
    ∃ A B : point, A ∈ A_and_B ∧ B ∈ A_and_B ∧ dist P A + dist P B = 18 * sqrt 2 / 5 := sorry

end curve_implicit_equation_line_inclination_angle_distance_sum_PA_PB_l601_601055


namespace relationship_between_vars_l601_601959

variable {α : Type*} [LinearOrderedAddCommGroup α]

theorem relationship_between_vars (a b : α) 
  (h1 : a + b < 0) 
  (h2 : b > 0) : a < -b ∧ -b < b ∧ b < -a :=
by
  sorry

end relationship_between_vars_l601_601959


namespace equal_prob_first_ace_l601_601038

theorem equal_prob_first_ace (deck : List ℕ) (players : Fin 4) (h_deck_size : deck.length = 32)
  (h_distinct : deck.nodup) (h_aces : ∀ _i, deck.filter (λ card, card = 1 ).length = 4)
  (h_shuffled : ∀ (card : ℕ), card ∈ deck → card ∈ (range 32)) :
  ∀ (player : Fin 4), let positions := List.range' (player + 1) (32 / 4) * 4 + player;
  (∀ (pos : ℕ), pos ∈ positions → deck.nth pos = some 1) →
  P(player) = 1 / 8 :=
by
  sorry

end equal_prob_first_ace_l601_601038


namespace equal_prob_first_ace_l601_601042

/-
  Define the problem:
  In a 4-player card game with a 32-card deck containing 4 aces,
  prove that the probability of each player drawing the first ace is 1/8.
-/

namespace CardGame

def deck : list ℕ := list.range 32

def is_ace (card : ℕ) : Prop := card % 8 = 0

def player_turn (turn : ℕ) : ℕ := turn % 4

def first_ace_turn (deck : list ℕ) : ℕ :=
deck.find_index is_ace

theorem equal_prob_first_ace :
  ∀ (deck : list ℕ) (h : deck.cardinality = 32) (h_ace : ∑ (card ∈ deck) (is_ace card) = 4),
  ∀ (player : ℕ), player < 4 → (∃ n < 32, first_ace_turn deck = some n ∧ player_turn n = player) →
  (deck.countp is_ace) / 32 = 1 / 8 :=
by sorry

end CardGame

end equal_prob_first_ace_l601_601042


namespace annual_increase_rate_l601_601312

theorem annual_increase_rate (PV FV : ℝ) (n : ℕ) (r : ℝ) :
  PV = 32000 ∧ FV = 40500 ∧ n = 2 ∧ FV = PV * (1 + r)^2 → r = 0.125 :=
by
  sorry

end annual_increase_rate_l601_601312


namespace slope_of_given_line_l601_601401

def slope_of_line (l : String) : Real :=
  -- Assuming that we have a function to parse the line equation
  -- and extract its slope. Normally, this would be a complex parsing function.
  1 -- Placeholder, as the slope calculation logic is trivial here.

theorem slope_of_given_line : slope_of_line "x - y - 1 = 0" = 1 := by
  sorry

end slope_of_given_line_l601_601401


namespace sum_of_numbers_in_row_l601_601101

theorem sum_of_numbers_in_row 
  (n : ℕ)
  (sum_eq : (n * (3 * n - 1)) / 2 = 20112) : 
  n = 1006 :=
sorry

end sum_of_numbers_in_row_l601_601101


namespace number_of_3number_passwords_number_of_9number_passwords_starting_at_5_l601_601157

-- Definition of the grid and movement constraints
def is_valid_path (path : List ℕ) : Prop :=
  --  Dummy definition, should define the actual path constraints
  (path.length = 3 ∨ path.length = 9) ∧ (∀ i, 0 < i → i < path.length → 
  (path.nth i).isSome ∧ (path.nth i).get < 10)

-- Definition of the problem context (setting up the 3x3 grid)
def grid : List (List ℕ) :=
  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

-- a) Number of 3-number passwords
theorem number_of_3number_passwords : 
  ∃ n, n = 44 ∧ 
  ∀ path, is_valid_path path → path.length = 3 → 
  -- Dummy condition to be replaced with specific path constraints.
  true := sorry

-- b) Number of passwords using all 9 numbers, starting at 5
theorem number_of_9number_passwords_starting_at_5 : 
  ∃ n, n = 8 ∧ 
  ∀ path, is_valid_path path → path.length = 9 → (path.nth 0).get_or_else 0 = 5 → 
  -- Dummy condition to be replaced with specific path constraints.
  true := sorry

end number_of_3number_passwords_number_of_9number_passwords_starting_at_5_l601_601157


namespace solve_for_x_l601_601367

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x, 2 * f x - 16 = f (x - 6) ∧ x = 1 := by
  exists 1
  sorry

end solve_for_x_l601_601367


namespace watermelon_cost_is_100_rubles_l601_601742

theorem watermelon_cost_is_100_rubles :
  (∀ (x y k m n : ℕ) (a b : ℝ),
    x + y = k →
    n * a = m * b →
    n * a + m * b = 24000 →
    n = 120 →
    m = 30 →
    k = 150 →
    a = 100) :=
by
  intros x y k m n a b
  intros h1 h2 h3 h4 h5 h6
  have h7 : 120 * a = 30 * b, from h2
  have h8 : 120 * a + 30 * b = 24000, from h3
  have h9 : 120 * a = 12000, from sorry
  have h10 : a = 100, from sorry
  exact h10

end watermelon_cost_is_100_rubles_l601_601742


namespace largest_possible_percent_error_is_32_25_l601_601159

noncomputable def actual_radius : ℝ := 10

def error_margin : ℝ := 0.15

noncomputable def actual_area : ℝ := real.pi * actual_radius^2

noncomputable def min_measured_area : ℝ := real.pi * (actual_radius * (1 - error_margin))^2

noncomputable def max_measured_area : ℝ := real.pi * (actual_radius * (1 + error_margin))^2

noncomputable def largest_percent_error : ℝ := 
  max ((actual_area - min_measured_area) / actual_area * 100)
      ((max_measured_area - actual_area) / actual_area * 100)

theorem largest_possible_percent_error_is_32_25 : largest_percent_error = 32.25 := by
  sorry

end largest_possible_percent_error_is_32_25_l601_601159


namespace possible_values_of_a_l601_601992

theorem possible_values_of_a (a : ℝ) : (2 < a ∧ a < 3 ∨ 3 < a ∧ a < 5) → (a = 5/2 ∨ a = 4) := 
by
  sorry

end possible_values_of_a_l601_601992


namespace problem_statement_l601_601934

variables {α : Type*} [linear_ordered_field α]
variable (f : α → α)
variable (h_even : ∀ x, f x = f (-x))
variable (h_monotone_dec : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → b ≤ 2 → f(a - 2) ≥ f(b - 2))

theorem problem_statement :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
by {
  sorry
}

end problem_statement_l601_601934


namespace ladder_distance_from_wall_l601_601721

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601721


namespace space_is_volume_stuff_is_capacity_film_is_surface_area_l601_601146

-- Let's define the properties based on the conditions
def size_of_space (box : Type) : Type := 
  sorry -- This will be volume later

def stuff_can_hold (box : Type) : Type :=
  sorry -- This will be capacity later

def film_needed_to_cover (box : Type) : Type :=
  sorry -- This will be surface area later

-- Now prove the correspondences
theorem space_is_volume (box : Type) :
  size_of_space box = volume := 
by 
  sorry

theorem stuff_is_capacity (box : Type) :
  stuff_can_hold box = capacity := 
by 
  sorry

theorem film_is_surface_area (box : Type) :
  film_needed_to_cover box = surface_area := 
by 
  sorry

end space_is_volume_stuff_is_capacity_film_is_surface_area_l601_601146


namespace ladder_base_distance_l601_601612

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601612


namespace base_distance_l601_601618

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601618


namespace volume_ABCD_l601_601586

open Real

noncomputable def volume_tetrahedron
    (AB AC AD BC BD CD : ℝ)
    (hAB : AB = 2) (hAC : AC = 4) (hAD : AD = 3)
    (hBC : BC = sqrt 17) (hBD : BD = sqrt 13) (hCD : CD = 5)
  : ℝ :=
  let a := AC, b := AD, c := CD
  let area_base := (1 / 2) * a * b   -- area of triangle ACD
  let height := sqrt ((AB^2 - ((a^2 + b^2 - c^2) / (2*a*b))^2) / 16)
  (1 / 3) * area_base * height

theorem volume_ABCD
    (AB AC AD BC BD CD : ℝ)
    (hAB : AB = 2) (hAC : AC = 4) (hAD : AD = 3)
    (hBC : BC = sqrt 17) (hBD : BD = sqrt 13) (hCD : CD = 5) :
    volume_tetrahedron AB AC AD BC BD CD hAB hAC hAD hBC hBD hCD = 6 * (sqrt 247) / 64 :=
sorry

end volume_ABCD_l601_601586


namespace ladder_base_distance_l601_601688

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601688


namespace a_8_is_256_l601_601473

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 2

axiom a_pq : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_is_256 : a 8 = 256 := by
  sorry

end a_8_is_256_l601_601473


namespace base_from_wall_l601_601641

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601641


namespace total_students_in_college_l601_601457

theorem total_students_in_college (ratio_b : ratio_g : ℕ) (num_girls : ℕ) (h_ratio : ratio_b = 8 ∧ ratio_g = 5) (h_girls : num_girls = 135) :
  let x := num_girls / ratio_g in
  let num_boys := ratio_b * x in
  let total_students := num_boys + num_girls in
  total_students = 351 :=
by
  sorry

end total_students_in_college_l601_601457


namespace equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601032

section
variable [decidable_eq ℕ]
variable (deck_size: ℕ := 32)
variable (num_aces: ℕ := 4)
variable (players: Π (i: fin 4), ℕ := λ i, 1)
variable [uniform_dist: Probability_Mass_Function (fin deck_size)] 

-- Part (a): Probabilities for each player to get the first Ace
noncomputable def player1_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player2_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player3_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player4_prob (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_of_first_ace :
  player1_prob deck = 1/8 ∧
  player2_prob deck = 1/8 ∧
  player3_prob deck = 1/8 ∧
  player4_prob deck = 1/8 :=
sorry

-- Part (b): Modify rules to deal until Ace of Spades
noncomputable def player_prob_ace_of_spades (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_with_ace_of_spades :
  ∀(p: fin 4), player_prob_ace_of_spades deck = 1/4 :=
sorry
end

end equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601032


namespace compare_sqrt_l601_601803

theorem compare_sqrt (a b : ℝ) (h1 : 2 * real.sqrt 2 = real.sqrt 8) (h2 : 3 = real.sqrt 9) (h3 : 8 < 9) : 2 * real.sqrt 2 - 3 < 0 :=
by {
  sorry
}

end compare_sqrt_l601_601803


namespace triangle_ARS_isosceles_l601_601500

open EuclideanGeometry

-- Definitions of the conditions
variables {A B C P R S : Point}
variables {Γ : Circle}

-- Conditions mentioned in a)
axiom hABC : Triangle A B C
axiom hAB_lt_AC : dist A B < dist A C
axiom hCircumcircle : Circumcircle Γ A B C
axiom hTangentP : Tangent Γ A P
axiom hBisector : ∃ M : Line, (IsBisector M ∠APB) ∧ Incides M R ∧ Incides M S

-- Statement of the problem
theorem triangle_ARS_isosceles :
  IsIsoscelesTriangle A R S :=
sorry

end triangle_ARS_isosceles_l601_601500


namespace base_distance_l601_601614

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601614


namespace calculate_expression_l601_601796

theorem calculate_expression : 7 * (12 + 2 / 5) - 3 = 83.8 :=
by
  sorry

end calculate_expression_l601_601796


namespace money_left_l601_601073

def initial_money : ℝ := 18
def spent_on_video_games : ℝ := 6
def spent_on_snack : ℝ := 3
def toy_original_cost : ℝ := 4
def toy_discount : ℝ := 0.25

theorem money_left (initial_money spent_on_video_games spent_on_snack toy_original_cost toy_discount : ℝ) :
  initial_money = 18 →
  spent_on_video_games = 6 →
  spent_on_snack = 3 →
  toy_original_cost = 4 →
  toy_discount = 0.25 →
  (initial_money - (spent_on_video_games + spent_on_snack + (toy_original_cost * (1 - toy_discount)))) = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end money_left_l601_601073


namespace find_all_good_numbers_l601_601760

def is_good (n : ℕ) : Prop :=
  ∀ (a : ℕ), (∃ (x : ℕ → ℤ), a = n^2 * (Finset.range n).sum (λ i, x i^2)) →
          ∃ (y : ℕ → ℤ), a = (Finset.range n).sum (λ i, y i^2) ∧ ∀ i, y i % n ≠ 0

theorem find_all_good_numbers : 
  { n : ℕ | is_good n } = {1} ∪ {n | 3 ≤ n ∧ n ≠ 4} := sorry

end find_all_good_numbers_l601_601760


namespace fraction_given_to_son_l601_601013

theorem fraction_given_to_son : 
  ∀ (blue_apples yellow_apples total_apples remaining_apples given_apples : ℕ),
    blue_apples = 5 →
    yellow_apples = 2 * blue_apples →
    total_apples = blue_apples + yellow_apples →
    remaining_apples = 12 →
    given_apples = total_apples - remaining_apples →
    (given_apples : ℚ) / total_apples = 1 / 5 :=
by
  intros
  sorry

end fraction_given_to_son_l601_601013


namespace solve_four_tuple_l601_601296

-- Define the problem conditions
theorem solve_four_tuple (a b c d : ℝ) : 
    (ab + c + d = 3) → 
    (bc + d + a = 5) → 
    (cd + a + b = 2) → 
    (da + b + c = 6) → 
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by
  intros h1 h2 h3 h4
  sorry

end solve_four_tuple_l601_601296


namespace smallest_a_with_50_perfect_squares_l601_601892

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601892


namespace find_number_divided_by_3_equals_subtracted_5_l601_601204

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601204


namespace probability_even_sum_l601_601161

/-- There are 20 balls numbered from 1 to 20. Ana and Ben each randomly remove one ball, 
and these balls have distinct numbers. Prove that the probability 
that the sum of the two numbers on the balls removed is even is 9/19. -/
theorem probability_even_sum (h_distinct: ∀ (a b : ℕ), a ≠ b):
  (∃ (a b : ℕ), a ≤ 20 ∧ 1 ≤ a ∧ b ≤ 20 ∧ 1 ≤ b ∧ a ≠ b ∧
   even (a + b) ∧ (∑ i in range(20), (ite (even i) 1 0)) /
   (∑ i in range(20), (ite (odd i) 1 0)) = 9/19) :=
by {
  sorry,
}

end probability_even_sum_l601_601161


namespace equation_solution_count_l601_601343

theorem equation_solution_count :
  (∃ (c : ℕ), c ∈ finset.range (2001) ∧ 
  ∀ x : ℝ, (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊) = c) ↔ 
  501 :=
sorry

end equation_solution_count_l601_601343


namespace max_trig_expr_l601_601298

theorem max_trig_expr : 
  (∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x + 1 ≤ Real.sqrt 13 + 1) ∧ 
  (∃ x : ℝ, 2 * Real.cos x + 3 * Real.sin x + 1 = Real.sqrt 13 + 1) :=
begin
  sorry,
end

end max_trig_expr_l601_601298


namespace range_of_x_l601_601469

theorem range_of_x {x : ℝ} : 
  (∀ y, y = (sqrt (x + 2)) / (3 * x) → x ≥ -2 ∧ x ≠ 0) :=
begin
  sorry
end

end range_of_x_l601_601469


namespace system_solution_l601_601100

theorem system_solution (a x0 : ℝ) (h : a ≠ 0) 
  (h1 : 3 * x0 + 2 * x0 = 15 * a) 
  (h2 : 1 / a * x0 + x0 = 9) 
  : x0 = 6 ∧ a = 2 :=
by {
  sorry
}

end system_solution_l601_601100


namespace distance_from_wall_l601_601632

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601632


namespace convert_to_rectangular_form_l601_601294

theorem convert_to_rectangular_form (θ : ℝ) (r : ℝ) (hθ : θ = 13 * Real.pi / 6) (hr : r = Real.sqrt 3) :
  r * Real.exp (Complex.I * θ) = (3 / 2 : ℝ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end convert_to_rectangular_form_l601_601294


namespace mary_potatoes_l601_601094

/--
Mary had 8 potatoes in the garden. The rabbits ate 3 of the potatoes.
Prove that Mary has now 5 potatoes.
-/
theorem mary_potatoes :
  let original_potatoes := 8
  let eaten_by_rabbits := 3
  original_potatoes - eaten_by_rabbits = 5 :=
by
  let original_potatoes := 8
  let eaten_by_rabbits := 3
  exact rfl
  sorry

end mary_potatoes_l601_601094


namespace magnitude_expression_l601_601436

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601436


namespace NP_value_l601_601064

noncomputable def triangle_ABC : Type := sorry
noncomputable def point_M (AC : ℝ) : Type := sorry
noncomputable def point_N (AB : ℝ) (AC : ℝ) : Type := sorry
noncomputable def point_P (BN : Type) (CM : Type) : Type := sorry
noncomputable def point_K (CM : Type) (M : Type) (AK : ℝ) : Type := sorry

theorem NP_value :
  ∀ (A B C M N P K : Type)
    (AC BC AM MC : ℝ)
    (AK : ℝ),
    AC = 540 →
    BC = 360 →
    AM = MC →
    AK = 240 →
    M = midpoint A C →
    N = point_of angle_bisector B (angle_bisector_ratio A B M C) →
    P = P_intersection CM BN →
    K = point_on_line CM with midpoint M PK, AK →
    NP = 480 :=
by
  sorry

end NP_value_l601_601064


namespace eq_squares_diff_l601_601317

theorem eq_squares_diff {x y z : ℝ} :
  x = (y - z)^2 ∧ y = (x - z)^2 ∧ z = (x - y)^2 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end eq_squares_diff_l601_601317


namespace find_a_value_l601_601350

theorem find_a_value (a : ℝ) (A : Set ℝ := {a^2, a + 1, -3}) (B : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}) 
    (h : ({a^2, a + 1, -3} ∩ {a - 3, 3 * a - 1, a^2 + 1} = {-3})) : 
    a = -2/3 :=
by
  sorry

end find_a_value_l601_601350


namespace convert_to_rectangular_form_l601_601292

theorem convert_to_rectangular_form (θ : ℝ) (r : ℝ) (hθ : θ = 13 * Real.pi / 6) (hr : r = Real.sqrt 3) :
  r * Real.exp (Complex.I * θ) = (3 / 2 : ℝ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end convert_to_rectangular_form_l601_601292


namespace watermelon_cost_l601_601745

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l601_601745


namespace smallest_sum_of_squares_l601_601330

theorem smallest_sum_of_squares :
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 ≥ 36 ∧ y^2 ≥ 36 ∧ x^2 + y^2 = 625 :=
by
  sorry

end smallest_sum_of_squares_l601_601330


namespace semiperimeter_eq_diagonal_l601_601765

-- The problem definition and given conditions
variable (a b c : ℝ) (h1 : b ≠ c) (h2 : b^2 + c^2 = 2 * a^2) (h3 : b + c = a * sqrt 2)

-- The theorem statement proving the desired result
theorem semiperimeter_eq_diagonal (h4 : sqrt (b^2 + c^2) = a * sqrt 2) : 
  (b + c) / 2 = a * sqrt 2 := 
by
  sorry

end semiperimeter_eq_diagonal_l601_601765


namespace find_line_eqn_from_bisected_chord_l601_601930

noncomputable def line_eqn_from_bisected_chord (x y : ℝ) : Prop :=
  2 * x + y - 3 = 0

theorem find_line_eqn_from_bisected_chord (
  A B : ℝ × ℝ) 
  (hA :  (A.1^2) / 2 + (A.2^2) / 4 = 1)
  (hB :  (B.1^2) / 2 + (B.2^2) / 4 = 1)
  (h_mid : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  line_eqn_from_bisected_chord 1 1 :=
by 
  sorry

end find_line_eqn_from_bisected_chord_l601_601930


namespace find_number_l601_601187

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601187


namespace determine_C_for_identity_l601_601015

theorem determine_C_for_identity :
  (∀ (x : ℝ), (1/2 * (Real.sin x)^2 + C = -1/4 * Real.cos (2 * x))) → C = -1/4 :=
by
  sorry

end determine_C_for_identity_l601_601015


namespace remainder_seven_power_twenty_seven_l601_601173

theorem remainder_seven_power_twenty_seven :
  (7^27) % 1000 = 543 := 
sorry

end remainder_seven_power_twenty_seven_l601_601173


namespace percent_students_70_to_79_correct_l601_601243

-- Given conditions as definitions
def total_students : ℕ := 5 + 7 + 9 + 7 + 3
def students_70_to_79 : ℕ := 9

-- Main statement to be proved
theorem percent_students_70_to_79_correct :
  (students_70_to_79 : ℚ) / (total_students : ℚ) * 100 ≈ 29.03 := 
sorry

end percent_students_70_to_79_correct_l601_601243


namespace votes_ratio_l601_601979

theorem votes_ratio (joey_votes barry_votes marcy_votes : ℕ) 
  (h1 : joey_votes = 8) 
  (h2 : barry_votes = 2 * (joey_votes + 3)) 
  (h3 : marcy_votes = 66) : 
  (marcy_votes : ℚ) / barry_votes = 3 / 1 := 
by
  sorry

end votes_ratio_l601_601979


namespace ladder_base_distance_l601_601663

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601663


namespace smallest_a_has_50_perfect_squares_l601_601857

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601857


namespace lily_correct_percentage_l601_601092

-- Define the conditions
def percent_first_test := 85 / 100
def percent_second_test := 75 / 100
def percent_third_test := 65 / 100
def num_problems_first_test := 20
def num_problems_second_test := 50
def num_problems_third_test := 15

-- Calculate the number of correct answers
def correct_first_test := percent_first_test * num_problems_first_test
def correct_second_test := percent_second_test * num_problems_second_test
def correct_third_test := percent_third_test * num_problems_third_test

-- Round the number of correct answers
def rounded_correct_first_test := Int.ofNat (Nat.ceil correct_first_test.toReal)
def rounded_correct_second_test := Int.ofNat (Nat.ceil correct_second_test.toReal)
def rounded_correct_third_test := Int.ofNat (Nat.ceil correct_third_test.toReal)

-- Calculate the total number of correct answers and the total number of problems
def total_correct_answers := rounded_correct_first_test + rounded_correct_second_test + rounded_correct_third_test
def total_problems := num_problems_first_test + num_problems_second_test + num_problems_third_test

-- Calculate the overall percentage
def overall_percentage := (total_correct_answers.toReal / total_problems.toReal) * 100

-- The proof statement
theorem lily_correct_percentage : overall_percentage = 76 := by
  sorry

end lily_correct_percentage_l601_601092


namespace max_positive_n_l601_601391

def a (n : ℕ) : ℤ := 19 - 2 * n

theorem max_positive_n (n : ℕ) (h : a n > 0) : n ≤ 9 :=
by
  sorry

end max_positive_n_l601_601391


namespace distance_from_wall_l601_601636

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601636


namespace range_of_function_on_interval_minimum_of_function_on_interval_l601_601399

section Problem1

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x - 1

theorem range_of_function_on_interval {a : ℝ} (h : a = 1) :
  ∃ (R : Set ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x a ∈ R) ∧ R = Set.Icc (-1 : ℝ) 14 := by
  sorry

end Problem1

section Problem2

theorem minimum_of_function_on_interval {a : ℝ} (h : a < 0) :
  ( -3 < a ∧  a < 0 → ∃ m, ∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≥ m ∧ m = -1 - a^2) ∧   
  (−∞ < a ∧ a ≤ -3 → ∃ m, ∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≥ m ∧ m = 8 + 6 * a) := by
  sorry

end Problem2

end range_of_function_on_interval_minimum_of_function_on_interval_l601_601399


namespace smallest_a_with_50_perfect_squares_l601_601901

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601901


namespace smallest_a_has_50_perfect_squares_l601_601860

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601860


namespace solution_set_l601_601933

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Condition 2: f is increasing in (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Condition 3: f(2) = 0
def f_at_2 (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The main theorem to prove
theorem solution_set (odd_f : odd_function f) (increasing_f : increasing_on_pos f) (f_2 : f_at_2 f) : 
  {x : ℝ | (f x - f (-x)) / x < 0} = set.Ioo (-2) 0 ∪ set.Ioo 0 2 := 
by
  sorry

end solution_set_l601_601933


namespace total_paper_clips_l601_601257

/-
Given:
- The number of cartons: c = 3
- The number of boxes: b = 4
- The number of bags: p = 2
- The number of paper clips in each carton: paper_clips_per_carton = 300
- The number of paper clips in each box: paper_clips_per_box = 550
- The number of paper clips in each bag: paper_clips_per_bag = 1200

Prove that the total number of paper clips is 5500.
-/

theorem total_paper_clips :
  let c := 3
  let paper_clips_per_carton := 300
  let b := 4
  let paper_clips_per_box := 550
  let p := 2
  let paper_clips_per_bag := 1200
  (c * paper_clips_per_carton + b * paper_clips_per_box + p * paper_clips_per_bag) = 5500 :=
by
  sorry

end total_paper_clips_l601_601257


namespace smallest_a_with_50_perfect_squares_l601_601899

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601899


namespace ladder_base_distance_l601_601659

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601659


namespace right_triangle_leg_squared_l601_601544

variable (a b c : ℝ)

theorem right_triangle_leg_squared (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_leg_squared_l601_601544


namespace part1_part2_l601_601277

-- Definitions of the points and their properties
variables (A B C P M N : Type) [inner_product_space ℝ A B C P M N] [nontrivial ℝ A B C P M N]
variable (circumcircle : A → B → C → Type)

-- Given conditions
axiom condition1 : P ∈ A.B ∧ (dist A B = 4 * dist A P)
axiom condition2 : M ∈ circumcircle A B C ∧ N ∈ circumcircle A B C ∧ line_through P M N
axiom condition3 : is_midpoint_arc A M N

-- Prove the following:
theorem part1 : ∼ (similar (triangle A B N) (triangle A N P)) := sorry
theorem part2 : (dist B M + dist B N = 2 * dist M N) := sorry

end part1_part2_l601_601277


namespace ladder_base_distance_l601_601610

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601610


namespace probability_same_color_is_correct_l601_601029

/- Given that there are 5 balls in total, where 3 are white and 2 are black, and two balls are drawn randomly from the bag, we need to prove that the probability of drawing two balls of the same color is 2/5. -/

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def total_ways (n r : ℕ) : ℕ := n.choose r
def white_ways : ℕ := total_ways white_balls 2
def black_ways : ℕ := total_ways black_balls 2
def same_color_ways : ℕ := white_ways + black_ways
def total_draws : ℕ := total_ways total_balls 2

def probability_same_color := ((same_color_ways : ℚ) / total_draws)
def expected_probability := (2 : ℚ) / 5

theorem probability_same_color_is_correct :
  probability_same_color = expected_probability :=
by
  sorry

end probability_same_color_is_correct_l601_601029


namespace ellipse_eq_l601_601148

noncomputable def standardEquationOfEllipse (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (y^2 / a^2 + x^2 / b^2 = 1)

theorem ellipse_eq (a b : ℝ) :
  ∀ (F1 P : ℝ × ℝ), 
  (F1 = (0, 1)) → 
  (P = (3/2, 1)) → 
  (a^2 - b^2 = 1) → 
  (P.1^2 / b^2 + P.2^2 / a^2 = 1) → 
  (standardEquationOfEllipse 2 sqrt(3)) :=
by 
  intros F1 P F1_def P_def ab_eq p_eq_pass
  sorry

end ellipse_eq_l601_601148


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601180

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601180


namespace base_distance_l601_601615

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601615


namespace solve_sqrt_equation_l601_601316

theorem solve_sqrt_equation (x : ℝ) :
  sqrt ((2 + sqrt 3) ^ x) + sqrt ((2 - sqrt 3) ^ x) = 6 ↔ x = -2 :=
by {
  sorry
}

end solve_sqrt_equation_l601_601316


namespace watermelon_cost_l601_601744

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l601_601744


namespace sum_of_perimeters_geq_4400_l601_601767

theorem sum_of_perimeters_geq_4400 (side original_side : ℕ) 
  (h_side_le_10 : ∀ s, s ≤ side → s ≤ 10) 
  (h_original_square : original_side = 100) 
  (h_cut_condition : side ≤ 10) : 
  ∃ (small_squares : ℕ → ℕ × ℕ), (original_side / side = n) → 4 * n * side ≥ 4400 :=
by
  sorry

end sum_of_perimeters_geq_4400_l601_601767


namespace find_f_log3_2_l601_601941

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 2 then f (x + 1) else 3 ^ (-x)

theorem find_f_log3_2 : f (Real.log 2 / Real.log 3) = 1 / 18 := by
  sorry

end find_f_log3_2_l601_601941


namespace find_n_solution_l601_601837

theorem find_n_solution : ∃ n : ℤ, (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n : ℝ) / (n + 1 : ℝ) = 3) :=
by
  use 0
  sorry

end find_n_solution_l601_601837


namespace find_smallest_a_l601_601853

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601853


namespace length_of_arc_SP_l601_601466

/-- Given that the measure of angle SIP is 45 degrees and the radius OS is 15 cm,
    prove that the length of the arc SP is 7.5π cm. -/
theorem length_of_arc_SP (angle_SIP : ℝ) (radius_OS : ℝ) (h1 : angle_SIP = 45) (h2 : radius_OS = 15) :
  arc_length_SP = 7.5 * π := by
  -- Define the arc length function.
  let arc_length_SP := (angle_SIP / 360) * (2 * radius_OS * π)
  -- Use the given conditions to transform the proof.
  have h3 : arc_length_SP = (45 / 360) * (2 * 15 * π) := by rw [h1, h2]
  -- Simplify to the final result.
  calc
    arc_length_SP = (45 / 360) * (2 * 15 * π) : by sql[h3]
    ... = 7.5 * π : by norm_num
  sorry

end length_of_arc_SP_l601_601466


namespace ahmed_goats_correct_l601_601263

-- Definitions based on the conditions given in the problem.
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 5 + 2 * adam_goats
def ahmed_goats : ℕ := andrew_goats - 6

-- The theorem statement that needs to be proven.
theorem ahmed_goats_correct : ahmed_goats = 13 := by
    sorry

end ahmed_goats_correct_l601_601263


namespace smallest_a_with_50_perfect_squares_l601_601890

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601890


namespace circle_radius_intersection_l601_601053

noncomputable def r : ℝ := Real.sqrt 10

theorem circle_radius_intersection :
  ∃ A B C : ℝ × ℝ,
    (∃ r : ℝ, r > 0 ∧ 
      A.1^2 + A.2^2 = r^2 ∧
      B.1^2 + B.2^2 = r^2 ∧
      C.1^2 + C.2^2 = r^2 ∧
      C = (5 / 4) • A + (3 / 4) • B ∧
      B.2 = -B.1 + 2 ∧ -- Point B lies on the line y = -x + 2
      A.2 = -A.1 + 2) 
  → r = Real.sqrt 10 :=
begin
  sorry
end

end circle_radius_intersection_l601_601053


namespace ladder_distance_l601_601705

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601705


namespace ladder_base_distance_l601_601725

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601725


namespace valid_sentence_count_l601_601531

def words : List String := ["splargh", "glumph", "amr", "florp"]

def isValidSentence (sentence : List String) : Prop :=
  match sentence with
  | [_, "glumph", "florp"] => False
  | ["glumph", "florp", _] => False
  | _ => True

def countValidSentences : Nat :=
  let allSentences := List.replicate 3 words |>.foldr (List.bind · List.map (fun w => [w])) [[]]
  let validSentences := allSentences.filter isValidSentence
  validSentences.length

theorem valid_sentence_count : countValidSentences = 56 := by
  sorry

end valid_sentence_count_l601_601531


namespace base_from_wall_l601_601638

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l601_601638


namespace magnitude_expression_l601_601435

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601435


namespace Spot_dog_reachable_area_l601_601527

noncomputable def Spot_reachable_area (side_length tether_length : ℝ) : ℝ := 
  -- Note here we compute using the areas described in the problem
  6 * Real.pi * (tether_length^2) / 3 - Real.pi * (side_length^2)

theorem Spot_dog_reachable_area (side_length tether_length : ℝ)
  (H1 : side_length = 2) (H2 : tether_length = 3) :
    Spot_reachable_area side_length tether_length = (22 * Real.pi) / 3 := by
  sorry

end Spot_dog_reachable_area_l601_601527


namespace no_trisection_120_div_n_l601_601382

theorem no_trisection_120_div_n (n : ℕ) (hn : n > 0) : 
  ¬constructible_with_ruler_and_compass (120 / n / 3)
  → ¬constructible_with_ruler_and_compass (60 / 3) :=
sorry

end no_trisection_120_div_n_l601_601382


namespace limit_sequence_sqrt2_l601_601068

open Real

theorem limit_sequence_sqrt2 :
  ∀ (k : ℕ), let n := ⌊(k + sqrt 2)^3⌋ in
  let m := k^3 in
  HasLimit (fun (k : ℕ) => (n^(1 / 3) - m^(1 / 3))) ∞ (sqrt 2) :=
sorry

end limit_sequence_sqrt2_l601_601068


namespace distance_from_circle_to_line_l601_601993

def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def polar_line (θ : ℝ) : Prop := θ = Real.pi / 6

theorem distance_from_circle_to_line : 
  ∃ d : ℝ, polar_circle ρ θ ∧ polar_line θ → d = Real.sqrt 3 := 
by
  sorry

end distance_from_circle_to_line_l601_601993


namespace add_num_denom_fraction_l601_601587

theorem add_num_denom_fraction (n : ℚ) : (2 + n) / (7 + n) = 3 / 5 ↔ n = 11 / 2 := 
by
  sorry

end add_num_denom_fraction_l601_601587


namespace magnitude_expression_l601_601432

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601432


namespace ladder_base_distance_l601_601608

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601608


namespace problem1_problem2_l601_601348

open Real

def vector_a (x : ℝ) : ℝ × ℝ := (2 + sin x, 1)
def vector_b : ℝ × ℝ := (2, -2)
def vector_c (x : ℝ) : ℝ × ℝ := (sin x - 3, 1)
def vector_d (k : ℝ) : ℝ × ℝ := (1, k)
def vector_b_plus_c (x : ℝ) : ℝ × ℝ := (sin x - 1, -1)
def f (x : ℝ) := (2 + sin x) * 2 - 2

theorem problem1 (x : ℝ) (h : x ∈ Icc (-π/2) (π/2)) :
  vector_a x = (2 + sin x, 1) → 
  is_parallel (vector_a x) (vector_b_plus_c x) → 
  x = -π/6 :=
sorry

theorem problem2 (x : ℝ) : 
  ∀ x, f x = (2 * sin x + 2) → 
  ∃ y, y ∈ Icc (-π/2) (π/2) ∧ ∀ z ∈ Icc (-π/2) (π/2), f y ≤ f z ∧ f y = 0 :=
sorry

end problem1_problem2_l601_601348


namespace customers_served_total_l601_601783

theorem customers_served_total :
  let Ann_hours := 8
  let Ann_rate := 7
  let Becky_hours := 7
  let Becky_rate := 8
  let Julia_hours := 6
  let Julia_rate := 6
  let lunch_break := 0.5
  let Ann_customers := (Ann_hours - lunch_break) * Ann_rate
  let Becky_customers := (Becky_hours - lunch_break) * Becky_rate
  let Julia_customers := (Julia_hours - lunch_break) * Julia_rate
  Ann_customers + Becky_customers + Julia_customers = 137 := by
  sorry

end customers_served_total_l601_601783


namespace intersection_product_OP_OQ_l601_601986

noncomputable def curve_C1_polar_eq := 
  fun (ρ θ : ℝ) => ρ^2 - 2 * real.sqrt 3 * ρ * real.cos θ - 4 * ρ * real.sin θ + 3 = 0

noncomputable def line_C2_polar_eq := π / 6

theorem intersection_product_OP_OQ :
  (∀ (α : ℝ), ∃ (x y : ℝ), 
  x = real.sqrt 3 + 2 * real.cos α ∧ 
  y = 2 + 2 * real.sin α ∧ 
  y = (real.sqrt 3 / 3) * x) →
  let ρ1 := some (x, y), θ1 := π / 6 in
  let ρ2 := some (x, y), θ2 := π / 6 in
  |ρ1| * |ρ2| = 3 :=
by
  intro h
  sorry

end intersection_product_OP_OQ_l601_601986


namespace ladder_base_distance_l601_601689

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601689


namespace total_fireworks_l601_601770

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l601_601770


namespace supplement_of_complementary_angle_of_35_deg_l601_601965

theorem supplement_of_complementary_angle_of_35_deg :
  let A := 35
  let C := 90 - A
  let S := 180 - C
  S = 125 :=
by
  let A := 35
  let C := 90 - A
  let S := 180 - C
  -- we need to prove S = 125
  sorry

end supplement_of_complementary_angle_of_35_deg_l601_601965


namespace b_horses_pasture_l601_601594

theorem b_horses_pasture (H : ℕ) : (9 * H / (96 + 9 * H + 108)) * 870 = 360 → H = 6 :=
by
  -- Here we state the problem and skip the proof
  sorry

end b_horses_pasture_l601_601594


namespace find_Matrix_M_l601_601322

open Matrix

noncomputable def M : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[-8, -3, 15],
     [3, -1, 5],
     [0, 0, 1]]

def matrix_A : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[-2, 3, 0],
     [6, -8, 5],
     [0, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  !![[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

theorem find_Matrix_M : M * matrix_A = I := 
  by
  sorry

end find_Matrix_M_l601_601322


namespace ladder_distance_from_wall_l601_601720

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601720


namespace find_number_divided_by_3_equals_subtracted_5_l601_601212

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601212


namespace ladder_base_distance_l601_601604

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601604


namespace cos_sin_identity_l601_601514

theorem cos_sin_identity (x : ℝ) (n : ℕ) (hx : 0 < x ∧ x < π / 2) :
  (complex.of_real (Real.cos x) + complex.of_real (Real.sin x) * complex.I) ^ n =
  complex.of_real (Real.cos (n * x)) + complex.of_real (Real.sin (n * x)) * complex.I :=
sorry

end cos_sin_identity_l601_601514


namespace sum_x_coordinates_Q3_l601_601806

theorem sum_x_coordinates_Q3 (x_coords : Fin 45 → ℝ) (h_sum : ∑ i, x_coords i = 135) : 
  let Q2_coords := λ i, (x_coords i + x_coords (if i.1 < 44 then i.1 + 1 else 0)) / 2 in
  let Q3_coords := λ i, (Q2_coords i + Q2_coords (if i.1 < 44 then i.1 + 1 else 0)) / 2 in
  ∑ i, Q3_coords i = 135 := sorry

end sum_x_coordinates_Q3_l601_601806


namespace complex_roots_of_polynomial_l601_601501

theorem complex_roots_of_polynomial 
  (a b c : ℂ)
  (h1 : a + b + c = 1)
  (h2 : a * b + a * c + b * c = 1)
  (h3 : a * b * c = -1) :
  {a, b, c} = {1, - 1 / 2 + Complex.i * Real.sqrt 3 / 2, - 1 / 2 - Complex.i * Real.sqrt 3 / 2} :=
sorry

end complex_roots_of_polynomial_l601_601501


namespace ladder_distance_l601_601702

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601702


namespace kathleen_allowance_l601_601076

theorem kathleen_allowance (x : ℝ) (h1 : Kathleen_middleschool_allowance = x + 2)
(h2 : Kathleen_senior_allowance = 5 + 2 * (x + 2))
(h3 : Kathleen_senior_allowance = 2.5 * Kathleen_middleschool_allowance) :
x = 8 :=
by sorry

end kathleen_allowance_l601_601076


namespace average_age_of_large_family_is_correct_l601_601246

def average_age_of_family 
  (num_grandparents : ℕ) (avg_age_grandparents : ℕ) 
  (num_parents : ℕ) (avg_age_parents : ℕ) 
  (num_children : ℕ) (avg_age_children : ℕ) 
  (num_siblings : ℕ) (avg_age_siblings : ℕ)
  (num_cousins : ℕ) (avg_age_cousins : ℕ)
  (num_aunts : ℕ) (avg_age_aunts : ℕ) : ℕ := 
  let total_age := num_grandparents * avg_age_grandparents + 
                   num_parents * avg_age_parents + 
                   num_children * avg_age_children + 
                   num_siblings * avg_age_siblings + 
                   num_cousins * avg_age_cousins + 
                   num_aunts * avg_age_aunts
  let total_family_members := num_grandparents + num_parents + num_children + num_siblings + num_cousins + num_aunts
  (total_age : ℕ) / total_family_members

theorem average_age_of_large_family_is_correct :
  average_age_of_family 4 67 3 41 5 8 2 35 3 22 2 45 = 35 := 
by 
  sorry

end average_age_of_large_family_is_correct_l601_601246


namespace points_form_two_parallel_lines_l601_601167

noncomputable def set_of_points {P Q : ℝ × ℝ} : set (ℝ × ℝ) :=
{X | let PQ : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2) in
     let h : ℝ := 18 / PQ in
     let d := abs ((Q.2 - P.2) * X.1 - (Q.1 - P.1) * X.2 + Q.1 * P.2 - Q.2 * P.1) / PQ in
     d = h}

theorem points_form_two_parallel_lines (P Q : ℝ × ℝ) :
  set_of_points P Q = { X : ℝ × ℝ | 
    let PQ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2) in
    let h := 18 / PQ in
    let line1 := { Y : ℝ × ℝ | abs ((Q.2 - P.2) * Y.1 - (Q.1 - P.1) * Y.2 + Q.1 * P.2 - Q.2 * P.1) / PQ = h } in
    let line2 := { Y : ℝ × ℝ | abs ((Q.2 - P.2) * Y.1 - (Q.1 - P.1) * Y.2 + Q.1 * P.2 - Q.2 * P.1) / PQ = -h } in
    X ∈ line1 ∪ line2 } :=
sorry

end points_form_two_parallel_lines_l601_601167


namespace sum_first_six_terms_arithmetic_seq_l601_601539

theorem sum_first_six_terms_arithmetic_seq :
  ∃ a_1 d : ℤ, (a_1 + 3 * d = 7) ∧ (a_1 + 4 * d = 12) ∧ (a_1 + 5 * d = 17) ∧ 
  (6 * (2 * a_1 + 5 * d) / 2 = 27) :=
by
  sorry

end sum_first_six_terms_arithmetic_seq_l601_601539


namespace initial_speed_100kmph_l601_601249

theorem initial_speed_100kmph (v x : ℝ) (h1 : 0 < v) (h2 : 100 - x = v / 2) 
  (h3 : (80 - x) / (v - 10) - 20 / (v - 20) = 1 / 12) : v = 100 :=
by 
  sorry

end initial_speed_100kmph_l601_601249


namespace handshaking_problem_l601_601980

-- Define the handshaking conditions
def handshaking (n : ℕ) (k : ℕ) : Prop := ∃ (g : ℕ → list ℕ), 
  (∀ i, g i).length = k ∧
  (∀ i, ∀ j ∈ g i, i ≠ j) ∧
  (∀ i, ∀ j ∈ g i, ∃ l ∈ g j, l = i) 

theorem handshaking_problem (N : ℕ) :
  (∃ g : ℕ → list ℕ, handshaking 11 3) →
  N = 1814400 / 2 →
  N % 1000 = 400 :=
by
  intro h
  sorry

end handshaking_problem_l601_601980


namespace units_digit_sum_of_powers_l601_601810

theorem units_digit_sum_of_powers (a : ℕ) :
  (∀ a, (a - 1) * (a + 1) = a^2 - 1 ∧
        (a - 1) * (a^2 + a + 1) = a^3 - 1 ∧
        (a - 1) * (a^3 + a^2 + a + 1) = a^4 - 1) → 
  (∃ d : ℕ, d = (2^2023 + 2^2022 + ... + 2^2 + 2 + 1) % 10 ∧ d = 5) :=
by { sorry }

end units_digit_sum_of_powers_l601_601810


namespace insphere_radius_l601_601471

theorem insphere_radius (V S : ℝ) (hV : V > 0) (hS : S > 0) : 
  ∃ r : ℝ, r = 3 * V / S := by
  sorry

end insphere_radius_l601_601471


namespace expression_evaluation_l601_601755

theorem expression_evaluation : 
  (1 : ℝ)^(6 * z - 3) / (7⁻¹ + 4⁻¹) = 28 / 11 :=
by
  sorry

end expression_evaluation_l601_601755


namespace obtuse_angle_vectors_x_range_l601_601008

theorem obtuse_angle_vectors_x_range (x : ℝ) 
  (a : ℝ × ℝ := (x, 2 * x)) 
  (b : ℝ × ℝ := (-3 * x, 2)) 
  (h : -(3 * x^2) + 4 * x < 0) : 
  x ∈ set.Ioo (-∞ : ℝ) (-1 / 3) ∪ set.Ioo (-1 / 3) 0 ∪ set.Ioo (4 / 3) (∞ : ℝ) :=
sorry

end obtuse_angle_vectors_x_range_l601_601008


namespace ladder_base_distance_l601_601696

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601696


namespace ladder_base_distance_l601_601611

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601611


namespace net_gain_is_48799_21_l601_601754

def property1_price := 512456
def property1_gain_perc := 0.25

def property2_price := 634890
def property2_loss_perc := 0.30

def property3_price := 707869
def property3_gain_perc := 0.35

def property4_price := 432100
def property4_loss_perc := 0.40

def property5_price := 815372
def property5_gain_perc := 0.15

def property6_price := 391217
def property6_loss_perc := 0.22

def total_gain := property1_price * property1_gain_perc + property3_price * property3_gain_perc + property5_price * property5_gain_perc
def total_loss := property2_price * property2_loss_perc + property4_price * property4_loss_perc + property6_price * property6_loss_perc

def net_gain_or_loss := total_gain - total_loss

theorem net_gain_is_48799_21 : net_gain_or_loss = 48799.21 := by
  sorry

end net_gain_is_48799_21_l601_601754


namespace calculate_percentage_increase_l601_601276

variable (fish_first_round : ℕ) (fish_second_round : ℕ) (fish_total : ℕ) (fish_last_round : ℕ) (increase : ℚ) (percentage_increase : ℚ)

theorem calculate_percentage_increase
  (h1 : fish_first_round = 8)
  (h2 : fish_second_round = fish_first_round + 12)
  (h3 : fish_total = 60)
  (h4 : fish_last_round = fish_total - (fish_first_round + fish_second_round))
  (h5 : increase = fish_last_round - fish_second_round)
  (h6 : percentage_increase = (increase / fish_second_round) * 100) :
  percentage_increase = 60 := by
  sorry

end calculate_percentage_increase_l601_601276


namespace total_soccer_balls_donated_l601_601763

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end total_soccer_balls_donated_l601_601763


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601178

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601178


namespace find_c_l601_601016

theorem find_c (x : ℝ) (c : ℝ) (h : x = 0.3)
  (equ : (10 * x + 2) / c - (3 * x - 6) / 18 = (2 * x + 4) / 3) :
  c = 4 :=
by
  sorry

end find_c_l601_601016


namespace smallest_natural_number_with_50_squares_in_interval_l601_601869

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601869


namespace range_t_necessary_not_sufficient_negation_condition_l601_601357

theorem range_t_necessary_not_sufficient_negation_condition 
  (p : ∀ a x : ℝ, (a - 1) ^ x > 1 → x < 0)
  (q : ∀ a t : ℝ, a^2 - 2*t*a + t^2 - 1 < 0 → t - 1 < a ∧ a < t + 1) :
  (∃ A B : set ℝ, A = {a | a ≤ 1 ∨ a ≥ 2} ∧ B = {a | a ≤ t - 1 ∨ a ≥ t + 1} ∧ B ⊂ A) → 1 ≤ t ∧ t ≤ 2 :=
by
  sorry

end range_t_necessary_not_sufficient_negation_condition_l601_601357


namespace number_without_daughters_l601_601505

-- Given conditions
def Marilyn_daughters : Nat := 10
def total_women : Nat := 40
def daughters_with_daughters_women_have_each : Nat := 5

-- Helper definition representing the computation of granddaughters
def Marilyn_granddaughters : Nat := total_women - Marilyn_daughters

-- Proving the main statement
theorem number_without_daughters : 
  (Marilyn_daughters - (Marilyn_granddaughters / daughters_with_daughters_women_have_each)) + Marilyn_granddaughters = 34 := by
  sorry

end number_without_daughters_l601_601505


namespace smallest_a_with_50_perfect_squares_l601_601898

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601898


namespace ladder_base_distance_l601_601661

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601661


namespace midpoint_product_l601_601079

noncomputable def point := ℝ × ℝ

def midpoint (A B: point) : point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_product :
  ∀ (A B : point), midpoint A B = (4, 7) → A = (2, 10) → ∃ x y, B = (x, y) ∧ x * y = 24 :=
by
  intros A B h1 h2
  have h3 : A = (2, 10) := h2
  cases h1
  cases B with x y
  use x, y
  sorry

end midpoint_product_l601_601079


namespace ladder_distance_l601_601684

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601684


namespace simplify_expression_l601_601518

variable {R : Type*} [CommRing R]
variables (a b c : R)

theorem simplify_expression : (6 * a^2 * b * c) / (3 * a * b) = 2 * a * c :=
by sorry

end simplify_expression_l601_601518


namespace marshmallows_needed_l601_601786

theorem marshmallows_needed (total_campers : ℕ) (fraction_boys : ℚ) (fraction_girls : ℚ)
  (boys_toast_fraction : ℚ) (girls_toast_fraction : ℚ) :
  total_campers = 96 →
  fraction_boys = 2 / 3 →
  fraction_girls = 1 / 3 →
  boys_toast_fraction = 1 / 2 →
  girls_toast_fraction = 3 / 4 →
  let boys := (fraction_boys * total_campers).natAbs,
      girls := (fraction_girls * total_campers).natAbs,
      boys_toast := (boys_toast_fraction * boys).natAbs,
      girls_toast := (girls_toast_fraction * girls).natAbs in
  boys_toast + girls_toast = 56 := by
sorrry

end marshmallows_needed_l601_601786


namespace find_r_and_k_l601_601548

-- Define the line equation
def line (x : ℝ) : ℝ := 5 * x - 7

-- Define the parameterization
def param (t r k : ℝ) : ℝ × ℝ := 
  (r + 3 * t, 2 + k * t)

-- Theorem stating that (r, k) = (9/5, 15) satisfies the given conditions
theorem find_r_and_k 
  (r k : ℝ)
  (H1 : param 0 r k = (r, 2))
  (H2 : line r = 2)
  (H3 : param 1 r k = (r + 3, 2 + k))
  (H4 : line (r + 3) = 2 + k)
  : (r, k) = (9/5, 15) :=
sorry

end find_r_and_k_l601_601548


namespace sum_of_25x25_subgrid_l601_601051

theorem sum_of_25x25_subgrid (grid : Fin 50 → Fin 50 → ℤ) (h : (∀ i j, grid i j = 1 ∨ grid i j = -1) ∧ (|∑ i, ∑ j, grid i j| ≤ 100)) :
  ∃ (x y : Fin 26), |∑ i in range x, ∑ j in range y, grid i j| ≤ 25 :=
by sorry

end sum_of_25x25_subgrid_l601_601051


namespace find_f_neg_1_l601_601130

variables {α : Type} [AddGroup α] [HasNeg α] [HasSub α] [HasAdd α] [DecidableEq α]

-- Define an even function
def even_function (f : α → α) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_line (f : α → α) (a : α) : Prop :=
  ∀ x, f (a + x) = f (a - x)

variables {f : ℝ → ℝ}

-- Assumptions
variable (even_f : even_function f)
variable (symmetric_f : symmetric_about_line f 2)
variable (f_at_3 : f 3 = 3)

-- Theorem proving f(-1) = 3
theorem find_f_neg_1 : f (-1) = 3 :=
by
  sorry

end find_f_neg_1_l601_601130


namespace p_plus_q_is_701_l601_601282

theorem p_plus_q_is_701 {p q : ℕ} (h_coprime : Nat.coprime p q) (h_pos_p : 0 < p) (h_pos_q : 0 < q) :
  (∃x : ℝ, ∑ x in {x | ⌊x⌋ * (x - ⌊x⌋) = (p : ℝ) / (q : ℝ) * x ^ 2} = 360) → p + q = 701 := 
sorry

end p_plus_q_is_701_l601_601282


namespace number_of_elements_l601_601565

theorem number_of_elements (n S : ℕ) 
  (avg_all: S = 60 * n) 
  (avg_first_6: ∑ i in (finset.range 6).erase 5, i = 468) 
  (sixth_number: (finset.range 6).nth 5 = some 258) 
  (avg_last_6: ∑ i in (finset.range (n - 5) \ (finset.range 6).erase 5), i = 450) :
  n = 11 :=
by sorry

end number_of_elements_l601_601565


namespace quadratic_problem_l601_601004

theorem quadratic_problem (α β p q : ℚ) (h1 : 3*α^2 + 4*α + 2 = 0) (h2 : 3*β^2 + 4*β + 2 = 0) 
                          (hαβ_sum : α + β = -4/3) (hαβ_prod : α * β = 2/3) :
    let new_root_sum := 2*α + 2*β + 2 in
    let p := 4*(-new_root_sum) in
    p = 8/3 :=
by
  let new_root_sum := 2 * α + 2 * β + 2
  let p := 4 * (-new_root_sum)
  have : 2 * (α + β) + 2 = -8/3 + 2 := by sorry
  have : p = 4 * (-(-2/3)) := by sorry
  exact sorry

end quadratic_problem_l601_601004


namespace no_real_solutions_l601_601138

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 → ¬(3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2) :=
by
  sorry

end no_real_solutions_l601_601138


namespace inequality_transitive_l601_601353

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  a + c > b + d :=
by {
  sorry
}

end inequality_transitive_l601_601353


namespace magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601426

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_iz_plus_3conjugate_z_eq_2sqrt2 :
  | complex.I * z + 3 * (conj z) | = 2 * real.sqrt 2 := 
sorry

end magnitude_iz_plus_3conjugate_z_eq_2sqrt2_l601_601426


namespace variance_of_data_set_l601_601562

theorem variance_of_data_set :
  let data_set := [9, 7, 8, 6, 5]
  let n := data_set.length
  let mean := (data_set.sum) / n.toReal
  let variance := (1 / n.toReal) * (data_set.map (λ x, (x.toReal - mean) ^ 2)).sum
  variance = 2 :=
by
  let data_set := [9, 7, 8, 6, 5]
  let n := data_set.length
  let mean := (data_set.sum) / n.toReal
  let variance := (1 / n.toReal) * (data_set.map (λ x, (x.toReal - mean) ^ 2)).sum
  -- prove that variance = 2
  sorry

end variance_of_data_set_l601_601562


namespace identify_direct_proportionality_l601_601269

def direct_proportionality (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def option_A (x : ℝ) : ℝ := x + 1
def option_B (x : ℝ) : ℝ := -x^2
def option_C (x : ℝ) : ℝ := x / 5
def option_D (x : ℝ) : ℝ := 5 / x

theorem identify_direct_proportionality :
  direct_proportionality option_C ∧
  ¬ direct_proportionality option_A ∧
  ¬ direct_proportionality option_B ∧
  ¬ direct_proportionality option_D :=
by
  sorry

end identify_direct_proportionality_l601_601269


namespace line_perpendicular_to_plane_implies_parallel_l601_601914

-- Definitions for lines and planes in space
axiom Line : Type
axiom Plane : Type

-- Relation of perpendicularity between a line and a plane
axiom perp : Line → Plane → Prop

-- Relation of parallelism between two lines
axiom parallel : Line → Line → Prop

-- The theorem to be proved
theorem line_perpendicular_to_plane_implies_parallel (x y : Line) (z : Plane) :
  perp x z → perp y z → parallel x y :=
by sorry

end line_perpendicular_to_plane_implies_parallel_l601_601914


namespace kyle_gas_and_maintenance_expense_l601_601484

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end kyle_gas_and_maintenance_expense_l601_601484


namespace gmat_exam_analysis_l601_601736

theorem gmat_exam_analysis (S B N : ℝ) (hS : S = 70) (hB : B = 60) (hN : N = 5) : 
  ∃ F : ℝ, F = 85 :=
by
  let F := 100 - S + B - N
  have hF : F = 85 := by
    calc
      F = 100 - S + B - N : by rfl
      ... = 100 - 70 + 60 - 5 : by rw [hS, hB, hN]
      ... = 30 + 60 - 5 : by simp
      ... = 90 - 5 : by simp
      ... = 85 : by simp
  use F
  exact hF

end gmat_exam_analysis_l601_601736


namespace probability_first_ace_equal_l601_601036

theorem probability_first_ace_equal (num_cards : ℕ) (num_aces : ℕ) (num_players : ℕ)
  (h1 : num_cards = 32) (h2 : num_aces = 4) (h3 : num_players = 4) :
  ∀ player : ℕ, player ∈ {1, 2, 3, 4} → (∃ positions : list ℕ, (∀ n ∈ positions, n % num_players = player - 1)) → 
  (positions.length = 8) →
  let P := 1 / 8 in
  P = 1 / num_players :=
begin
  sorry
end

end probability_first_ace_equal_l601_601036


namespace determine_professions_l601_601569

-- Definition of professions
inductive Profession
| cook
| carpenter
| painter

open Profession

-- Definitions of people
constant Victor : Profession
constant Andrey : Profession
constant Peter : Profession

-- Conditions
axiom Victor_statement : Victor = cook
axiom Andrey_statement : Andrey ≠ cook
axiom Peter_statement : Peter ≠ carpenter
axiom One_mistake : (Andrey_statement = false) ∨ (Peter_statement = false)

-- Conclusion
theorem determine_professions :
  Victor = cook ∧
  Peter = carpenter ∧
  Andrey = painter :=
sorry

end determine_professions_l601_601569


namespace ladder_base_distance_l601_601672

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601672


namespace smallest_a_with_50_perfect_squares_l601_601895

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601895


namespace cube_volume_in_pyramid_l601_601250

noncomputable def rect_pyramid_base_length_a : ℝ := 2
noncomputable def rect_pyramid_base_length_b : ℝ := 3
noncomputable def pyramid_lateral_faces_isosceles : Prop := true

-- Midpoints condition for vertices touching pyramid's lateral faces
noncomputable def cube_position_condition : Prop :=
  true

-- To prove
theorem cube_volume_in_pyramid :
  let s := sqrt(39) / 3 in
  (s^3 = 39 * sqrt(39) / 27) :=
by
  sorry

end cube_volume_in_pyramid_l601_601250


namespace find_number_eq_seven_point_five_l601_601202

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601202


namespace interval_of_monotonic_increase_l601_601395

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem interval_of_monotonic_increase :
  ∃ a b : ℝ, (a < b) ∧ (∀ x : ℝ, a < x ∧ x < b -> (1 - Real.log x) / x^2 > 0) ∧ a = 0 ∧ b = Real.exp 1 :=
begin
  use [0, Real.exp 1],
  split,
  { linarith [(Real.exp_pos (1)).le] },
  split,
  { intros x hx,
    sorry },
  refl
end

end interval_of_monotonic_increase_l601_601395


namespace ladder_base_distance_l601_601692

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601692


namespace average_is_seven_over_six_l601_601972

-- Definitions for conditions in the problem
def mode (s : List ℤ) : ℤ := 3 -- Given that 3 is the mode

def data_set : List ℤ := [3, -1, 0, 3, -1, 3]

-- Function to calculate the average
def average (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

-- The proof problem: Prove the average of the data set is 7/6 given the mode condition
theorem average_is_seven_over_six (h: mode data_set = 3) :
  average data_set = 7 / 6 :=
sorry

end average_is_seven_over_six_l601_601972


namespace zero_of_f_in_interval_l601_601371

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 2 / Real.log 3

def f (x : ℝ) := a^x + x - b

theorem zero_of_f_in_interval :
  2^a = 3 ∧ 3^b = 2 → ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 := by
  sorry

end zero_of_f_in_interval_l601_601371


namespace teacher_can_win_l601_601366

theorem teacher_can_win : 
  ∀ (infinite_grid_plane : Type) (players : Fin 31), 
  (∀ (move : ℕ → finite_grid_segment infinite_grid_plane), 
   (∀ (n : ℕ), 
     (teacher_move n ∧ student_moves n) → 
     (∃ (rect : unit_rectangle infinite_grid_plane), 
       boundary_painted rect ∧ 
       ¬inside_painted rect)) :=
sorry

end teacher_can_win_l601_601366


namespace equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601031

section
variable [decidable_eq ℕ]
variable (deck_size: ℕ := 32)
variable (num_aces: ℕ := 4)
variable (players: Π (i: fin 4), ℕ := λ i, 1)
variable [uniform_dist: Probability_Mass_Function (fin deck_size)] 

-- Part (a): Probabilities for each player to get the first Ace
noncomputable def player1_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player2_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player3_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player4_prob (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_of_first_ace :
  player1_prob deck = 1/8 ∧
  player2_prob deck = 1/8 ∧
  player3_prob deck = 1/8 ∧
  player4_prob deck = 1/8 :=
sorry

-- Part (b): Modify rules to deal until Ace of Spades
noncomputable def player_prob_ace_of_spades (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_with_ace_of_spades :
  ∀(p: fin 4), player_prob_ace_of_spades deck = 1/4 :=
sorry
end

end equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l601_601031


namespace ladder_base_distance_l601_601690

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601690


namespace smallest_a_with_50_perfect_squares_l601_601893

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601893


namespace ladder_base_distance_l601_601662

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601662


namespace ladder_base_distance_l601_601660

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601660


namespace hiring_strategy_probabilities_l601_601238

theorem hiring_strategy_probabilities
  (abilities : Fin 10 → ℕ)
  (h_abilities : Function.Injective abilities)
  (A : Fin 10 → ℕ)
  (h_top3 : (A 0 + A 1 + A 2) > 7 * 10!)
  (h_bottom3 : (A 7 = A 8 ∧ A 8 = A 9 ∧ A 8 * 3 = 1 * 10!)) :
  A 0 > A 1 ∧ A 1 > A 2 ∧ A 2 > A 3 ∧ A 3 > A 4 ∧ A 4 > A 5 ∧ A 5 > A 6 ∧ A 6 > A 7 ∧
  (A 7 = A 8 ∧ A 8 = A 9) ∧
  (A 0 + A 1 + A 2) > 7 * 10! ∧
  (A 7 + A 8 + A 9) = 10! / 10 :=
sorry

end hiring_strategy_probabilities_l601_601238


namespace parallelepiped_properties_l601_601511

open Real

structure Parallelepiped :=
  (BC B1C1 AA1 : ℝ)
  (BM CM : ℝ)
  (radius volume : ℝ)
  (BC_eq : BC = BM + CM)

theorem parallelepiped_properties (p : Parallelepiped)
    (H_BM : p.BM = 1)
    (H_CM : p.CM = 8)
    (H_BC_eq : p.BC_eq = 1 + 8) :
    p.AA1 = 10 ∧ p.radius = 3 ∧ p.volume = 162 :=
by
  sorry

end parallelepiped_properties_l601_601511


namespace Suresh_meeting_time_l601_601131

theorem Suresh_meeting_time :
  let C := 726
  let v1 := 75
  let v2 := 62.5
  C / (v1 + v2) = 5.28 := by
  sorry

end Suresh_meeting_time_l601_601131


namespace least_number_to_add_divisible_l601_601225

theorem least_number_to_add_divisible (n d : ℕ) (h1 : n = 929) (h2 : d = 30) : 
  ∃ x, (n + x) % d = 0 ∧ x = 1 := 
by 
  sorry

end least_number_to_add_divisible_l601_601225


namespace KA_bisects_BC_l601_601738

-- Define the points A, B, C, M, N, K, and D
variables {A B C M N K D : Type}

-- Define the properties and conditions
variables (circle : Set (Point))
variables (is_circumscribed : Triangle → Set (Point) → Prop)
variables (tangent : Set (Point) → Point → Line)
variables (is_midpoint : Point → Point → Point → Prop)
variables (is_parallel : Line → Line → Prop)
variables (bisects : Point → Line → Prop)
variables (intersection : Line → Line → Point)
variables (is_line : Point → Point → Line)
variables (belongs_to : Point → Set (Point) → Prop)

-- The given triangle ABC
variables (ΔABC : Triangle)

-- Conditions
axiom circumscribed_around_triangle_ABC : is_circumscribed ΔABC circle
axiom M_on_circle : belongs_to M circle
axiom AM_parallel_BC : is_parallel (is_line A M) (is_line B C)
axiom N_intersection_tangents : N = intersection (tangent circle B) (tangent circle C)
axiom K_intersection_MN_circle : K = intersection (is_line M N) circle

-- The proof statement
theorem KA_bisects_BC
  (A B C M N K D : Point)
  (circle : Set (Point))
  (is_circumscribed : Triangle → Set (Point) → Prop)
  (tangent : Set (Point) → Point → Line)
  (is_midpoint : Point → Point → Point → Prop)
  (is_parallel : Line → Line → Prop)
  (bisects : Point → Line → Prop)
  (intersection : Line → Line → Point)
  (is_line : Point → Point → Line)
  (belongs_to : Point → Set (Point) → Prop)
  (ΔABC : Triangle)
  (circumscribed_around_triangle_ABC : is_circumscribed ΔABC circle)
  (M_on_circle : belongs_to M circle)
  (AM_parallel_BC : is_parallel (is_line A M) (is_line B C))
  (N_intersection_tangents : N = intersection (tangent circle B) (tangent circle C))
  (K_intersection_MN_circle : K = intersection (is_line M N) circle)
  : bisects (is_line K A) (is_line B C) :=
sorry

end KA_bisects_BC_l601_601738


namespace james_lifting_ratio_correct_l601_601791

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l601_601791


namespace strongest_correlation_l601_601133

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end strongest_correlation_l601_601133


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601182

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601182


namespace largest_tan_D_l601_601067

variable (D E : Point) (DEF : Triangle D E F)
variable (DE EF : ℝ)
hypothesis hDE : DE = 24
hypothesis hEF : EF = 18

theorem largest_tan_D (DEF : Triangle D E F) (hDE : DE = 24) (hEF : EF = 18) :
  tan DEF.1 = 3 * sqrt 7 / 7 :=
sorry

end largest_tan_D_l601_601067


namespace largest_divisor_of_expression_l601_601339

theorem largest_divisor_of_expression (n : ℤ) (h_composite : ¬nat.prime n ∧ 1 < n) (h_multiple_of_4 : ∃ k : ℤ, n = 4 * k) : 
  ∃ d : ℤ, (∀ n : ℤ, (¬nat.prime n ∧ 1 < n) → (∃ k : ℤ, n = 4 * k) → (n^3 - n^2) % d = 0) ∧ d = 16 :=
  sorry

end largest_divisor_of_expression_l601_601339


namespace caramel_candy_boxes_l601_601280

theorem caramel_candy_boxes (total_chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ) (total_chocolate_boxes = 7) (pieces_per_box = 8) (total_pieces = 80) : 
  let total_chocolate_pieces := total_chocolate_boxes * pieces_per_box in
  let total_caramel_pieces := total_pieces - total_chocolate_pieces in
  let caramel_boxes := total_caramel_pieces / pieces_per_box in
  caramel_boxes = 3 :=
by
  have total_chocolate_pieces_eq : total_chocolate_boxes * pieces_per_box = 56 := by sorry
  have total_caramel_pieces_eq : total_pieces - total_chocolate_boxes * pieces_per_box = 24 := by sorry
  have caramel_boxes_eq : total_caramel_pieces / pieces_per_box = 3 := by sorry
  exact caramel_boxes_eq

end caramel_candy_boxes_l601_601280


namespace num_true_propositions_l601_601966
noncomputable theory

theorem num_true_propositions : 
  let P := ∀ x, (sin x = 0 → cos x = 1)
  let Q := ∀ x, (cos x ≠ 1 → sin x ≠ 0) -- contrapositive
  let R := ∀ x, (cos x = 1 → sin x = 0) -- converse
  let S := ∀ x, (cos x ≠ 1 → sin x ≠ 0) -- inverse
  (¬P → ¬Q) ∧ (R → ¬(¬S)) →

  -- Number of true propositions
  let number_of_true := 2
  in
  number_of_true = 2 :=
by
  sorry

end num_true_propositions_l601_601966


namespace ladder_distance_l601_601675

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601675


namespace derivative_y1_derivative_y2_l601_601318

-- Define the first function
def y1 (x : ℝ) : ℝ := 2 * x ^ 5 - 3 * x ^ 2 - 4

-- Define its derivative and the theorem stating the result
noncomputable def y1_deriv (x : ℝ) : ℝ := 10 * x ^ 4 - 6 * x

theorem derivative_y1 (x : ℝ) : deriv y1 x = y1_deriv x := by
  sorry

-- Define the second function
def y2 (x : ℝ) : ℝ := exp x / sin x

-- Define its derivative and the theorem stating the result
noncomputable def y2_deriv (x : ℝ) : ℝ := (exp x * sin x - exp x * cos x) / (sin x ^ 2)

theorem derivative_y2 (x : ℝ) : deriv y2 x = y2_deriv x := by
  sorry

end derivative_y1_derivative_y2_l601_601318


namespace ladder_distance_l601_601699

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601699


namespace smallest_b_factors_l601_601833

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l601_601833


namespace monotonic_decreasing_interval_of_f_l601_601386

noncomputable def f (α β x : ℝ) : ℝ := (x + α) * (x + β)

theorem monotonic_decreasing_interval_of_f :
  (∃ α β : ℝ, (α + 5 + log α = 0) ∧ (β + 5 + exp β = 0)) →
  ∃ I : set ℝ, (∀ x ∈ I, ∃ α β : ℝ, f α β x < 0) ∧ I = set.Iic (5 / 2) :=
by
  sorry

end monotonic_decreasing_interval_of_f_l601_601386


namespace seq_a_form_l601_601354

def seq_a : ℕ+ → ℚ
| 1       := 3
| (n + 1) := 3 * seq_a n / (seq_a n + 3)

theorem seq_a_form (n : ℕ+) : seq_a n = 3 / (n : ℚ) :=
sorry

end seq_a_form_l601_601354


namespace ladder_base_distance_l601_601697

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601697


namespace ladder_base_distance_l601_601650

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l601_601650


namespace negation_equiv_exists_l601_601090

theorem negation_equiv_exists : 
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 0 := 
by 
  sorry

end negation_equiv_exists_l601_601090


namespace seating_arrangement_count_l601_601049

/-- In a row of 9 seats, three distinct people, A, B, and C, need to be seated such that each has empty seats to their left and right, with A seated between B and C. Prove the total number of different seating arrangements is 20. -/
theorem seating_arrangement_count :
  let seats : Finset ℕ := Finset.range 9 in
  let possible_positions (x : ℕ) := Finset.filter (λ n, n > 0 ∧ n < 8) seats in
  let count_positions (x : ℕ) := (possible_positions x).card in
  let arrangement_count :=
    (possible_positions A).sum (λ a, (Finset.filter (λ n, n ≠ a ∧ n ≠ a + 1 ∧ n ≠ a - 1) seats).card * 2) in
  arrangement_count = 20 :=
by
  sorry

end seating_arrangement_count_l601_601049


namespace no_prime_satisfies_condition_l601_601096

def original_prime_condition (P : ℝ) : Prop :=
  100 * P = P + 138.6

theorem no_prime_satisfies_condition :
  ¬ ∃ P : ℝ, prime P ∧ original_prime_condition P :=
by
  sorry

end no_prime_satisfies_condition_l601_601096


namespace area_of_park_correct_area_of_park_l601_601132

variable (w l : ℕ) (P A : ℕ)

axiom length_condition : l = 4 * w + 15
axiom perimeter_condition : P = 780
axiom perimeter_formula : 2 * (l + w) = P
axiom width_value : w = 75
axiom length_value : l = 315

theorem area_of_park : A = w * l := by
  have w_value : w = 75 := width_value
  have l_value : l = 315 := length_value
  rw [w_value, l_value]
  exact rfl

theorem correct_area_of_park : A = 23,625 :=
  have : w = 75 := width_value
  have : l = 315 := length_value
  have : A = 75 * 315 := area_of_park
  calc A = 75 * 315 := this
     ... = 23625 := by norm_num

#eval correct_area_of_park

end area_of_park_correct_area_of_park_l601_601132


namespace geometry_problem_l601_601987

theorem geometry_problem
  (AB : ℝ) (hAB : AB = 24)
  (angle_ADB : ℝ) (hangle_ADB : angle_ADB = 90)
  (sinA : ℝ) (hsinA : sinA = 2 / 3)
  (sinC : ℝ) (hsinC : sinC = 1 / 3) :
  ∃ DC : ℝ, DC = 32 * real.sqrt 2 :=
by
  sorry

end geometry_problem_l601_601987


namespace find_f_l601_601448

-- Definitions for the inverse function and conditions
def is_inverse (f g : ℝ → ℝ) := ∀ x, f (g x) = x ∧ g (f x) = x
def f (x : ℝ) := if x = 3 then 1 else 0 -- Temporary definition, to be corrected by the problem

-- Given conditions
axiom a_gt_zero : ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ is_inverse f (λ x, a * x) ∧ f 3 = 1

-- The theorem to prove
theorem find_f :
  ∃ (f : ℝ → ℝ), is_inverse f (λ x, 3 * x) ∧ f 3 = 1 ∧ (∀ x, f x = Real.log x / Real.log 3) :=
by
  sorry

end find_f_l601_601448


namespace basis_condition_l601_601408

variable (e1 e2 : Type) [AddGroup e1] [AddAction ℝ e1] [Module ℝ e1]
variable (a b c : e1)

def vector_a : e1 := (2 : ℝ) • e1 - e2
def vector_b : e1 := e1 + (2 : ℝ) • e2
def vector_c : e1 := (1 / 2 : ℝ) • e1 - (3 / 2 : ℝ) • e2

theorem basis_condition (hne : ¬ collinear e1 e2) : 
¬ is_basis ℝ (λ i : Fin 2, if i = 0 then vector_a - vector_b else vector_c) := by sorry

end basis_condition_l601_601408


namespace find_cost_price_range_of_x_max_profit_l601_601098

-- Define the conditions
def cost_price_a (a : ℕ) : Prop := ∃ b : ℕ, a - 1 = b
def shop_spends_1400_on_a (y : ℕ) : Prop := y = 1400
def shop_spends_630_on_b (z : ℕ) : Prop := z = 630
def cost_double_condition (a b : ℕ) : Prop := 1400 / a = 2 * (630 / b)
def total_items_equal_600 (x y : ℕ) : Prop := x + y = 600
def quantity_b_not_less_390 (y : ℕ) : Prop := y ≥ 390
def quantity_b_not_exceeds_4_times_a (x y : ℕ) : Prop := y ≤ 4 * x
constant sale_price : ℕ := 15
constant discount_threshold : ℕ := 150
constant discount_rate : ℝ := 0.4

-- Prove the cost price per item of type A and B
theorem find_cost_price (a b : ℕ) :
  cost_price_a a →
  shop_spends_1400_on_a 1400 →
  shop_spends_630_on_b 630 →
  cost_double_condition a b →
  a = 10 ∧ b = 9 :=
sorry

-- Prove the range for quantity of type A
theorem range_of_x (x y : ℕ) :
  total_items_equal_600 x y →
  quantity_b_not_less_390 y →
  quantity_b_not_exceeds_4_times_a x y →
  120 ≤ x ∧ x ≤ 210 :=
sorry

-- Prove the maximum profit calculation
theorem max_profit (x y : ℕ) :
  total_items_equal_600 x y →
  quantity_b_not_less_390 y →
  quantity_b_not_exceeds_4_times_a x y →
  x = 210 → y = 390 →
  let profit := 15 * 600 - (if x > discount_threshold then 10 * discount_threshold + 0.6 * 10 * (x - discount_threshold) else 10 * x) - 9 * y in
  profit = 3630 :=
sorry

end find_cost_price_range_of_x_max_profit_l601_601098


namespace escalator_time_l601_601271

theorem escalator_time (escalator_speed person_speed length : ℕ) 
    (h1 : escalator_speed = 12) 
    (h2 : person_speed = 2) 
    (h3 : length = 196) : 
    (length / (escalator_speed + person_speed) = 14) :=
by
  sorry

end escalator_time_l601_601271


namespace card_problem_l601_601217

variable (S1 S2 S3 S4 S5 : Prop)

def card_statements
  (h1 : S1 ↔ count_true [S2, S3, S4, S5] = 4)
  (h2 : S2 ↔ count_true [S1, S3, S4, S5] = 3)
  (h3 : S3 ↔ count_true [S1, S2, S4, S5] = 2)
  (h4 : S4 ↔ count_true [S1, S2, S3, S5] = 1)
  (h5 : S5 ↔ count_true [S1, S2, S3, S4] = 0) 
  : Prop :=
  ∃ (n : ℕ), n = 4 ∧ count_false [S1, S2, S3, S4, S5] = n

noncomputable def count_false (l : list Prop) : ℕ :=
  l.countp (λ b, b = false)

noncomputable def count_true (l : list Prop) : ℕ :=
  l.countp (λ b, b = true)

theorem card_problem : 
  card_statements
    (S1 ↔ count_true [S2, S3, S4, S5] = 4)
    (S2 ↔ count_true [S1, S3, S4, S5] = 3)
    (S3 ↔ count_true [S1, S2, S4, S5] = 2)
    (S4 ↔ count_true [S1, S2, S3, S5] = 1)
    (S5 ↔ count_true [S1, S2, S3, S4] = 0) :=
  sorry

end card_problem_l601_601217


namespace find_rate_percent_l601_601583

theorem find_rate_percent (SI P T : ℝ) (h : SI = (P * R * T) / 100) (H_SI : SI = 250) 
  (H_P : P = 1500) (H_T : T = 5) : R = 250 / 75 := by
  sorry

end find_rate_percent_l601_601583


namespace probability_blue_is_approx_50_42_l601_601799

noncomputable def probability_blue_second_pick : ℚ :=
  let yellow := 30
  let green := yellow / 3
  let red := 2 * green
  let total_marbles := 120
  let blue := total_marbles - (yellow + green + red)
  let total_after_first_pick := total_marbles - 1
  let blue_probability := (blue : ℚ) / total_after_first_pick
  blue_probability * 100

theorem probability_blue_is_approx_50_42 :
  abs (probability_blue_second_pick - 50.42) < 0.005 := -- Approximately checking for equality due to possible floating-point precision issues
sorry

end probability_blue_is_approx_50_42_l601_601799


namespace watermelon_cost_100_l601_601748

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l601_601748


namespace sum_of_logs_l601_601372

theorem sum_of_logs (a b : ℝ) (h1 : 10^a = 5) (h2 : 10^b = 2) : a + b = 1 := 
by {
  sorry
}

end sum_of_logs_l601_601372


namespace distance_from_wall_l601_601631

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601631


namespace complex_series_convergence_l601_601268

variable {α : Type*} [IsROrC α]

theorem complex_series_convergence (z : ℕ → α) 
  (h1 : ∀ n, z n ≠ 0) 
  (h2 : ∀ m n, m ≠ n → ‖z m - z n‖ > 1) : 
  Summable (λ n, 1 / (z n) ^ (3 : ℝ)) :=
by sorry

end complex_series_convergence_l601_601268


namespace smallest_munificence_of_monic_cubic_polynomial_l601_601338

def munificence (p : ℝ → ℝ) (a b : ℝ) : ℝ := 
  sup (abs ∘ p '' set.Icc a b)

def cubic_polynomial (b c d : ℝ) : ℝ → ℝ := λ x, x ^ 3 + b * x ^ 2 + c * x + d

theorem smallest_munificence_of_monic_cubic_polynomial : 
  ∃ b c d : ℝ, munificence (cubic_polynomial b c d) (-1) 1 = 3 / 2 := 
sorry

end smallest_munificence_of_monic_cubic_polynomial_l601_601338


namespace find_constant_l601_601023

theorem find_constant :
  ∃ (c : ℝ), (∀ t : ℝ, (x y : ℝ), x = c - 4 * t → y = 2 * t - 2 → (t = 0.5 → x = y)) → c = 1 :=
by
  use 1
  sorry

end find_constant_l601_601023


namespace intersection_complement_N_l601_601387

def is_universal_set (R : Set ℝ) : Prop := ∀ x : ℝ, x ∈ R

def is_complement (U S C : Set ℝ) : Prop := 
  ∀ x : ℝ, x ∈ C ↔ x ∈ U ∧ x ∉ S

theorem intersection_complement_N 
  (U M N C : Set ℝ)
  (h_universal : is_universal_set U)
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1})
  (h_compl : is_complement U M C) :
  (C ∩ N) = {x : ℝ | x < -2} := 
by 
  sorry

end intersection_complement_N_l601_601387


namespace unit_vectors_have_equal_norm_sq_l601_601929

variables {α : Type*} [inner_product_space ℝ α]

variables (a b : α)

-- Condition specifying that a and b are unit vectors
axiom unit_vector_a : ∥a∥ = 1
axiom unit_vector_b : ∥b∥ = 1

-- Proof statement that needs to be proven
theorem unit_vectors_have_equal_norm_sq : (∥a∥^2 = ∥b∥^2) :=
by 
-- The proof is skipped with sorry
sorry

end unit_vectors_have_equal_norm_sq_l601_601929


namespace ladder_base_distance_l601_601731

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601731


namespace smallest_a_with_50_perfect_squares_l601_601894

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l601_601894


namespace proof_of_distance_equality_l601_601363

noncomputable theory

-- Let's define the geometry elements required by the conditions
structure Triangle :=
(A B C : Point)

def incircle_tangency_points (T : Triangle) :=
let (A, B, C) := (T.A, T.B, T.C) in
-- Assume existence of points X, Y, Z as tangency points, we don't define coordinates
{ X : Point // true, Y : Point // true, Z : Point // true }

def intersection_points (B Y C Z : Point) :=
-- Assume existence of intersection point G
{ G : Point // true }

def parallelogram_points (B Z R C Y S : Point) :=
-- Assume existence of points R and S forming parallelograms
{ R : Point // true, S : Point // true }

-- Define the final proof problem given all necessary conditions
theorem proof_of_distance_equality (T : Triangle)
  (Pts : incircle_tangency_points T)
  (Ints : intersection_points T.B Pts.val.2 T.C Pts.val.1)
  (Parall : parallelogram_points T.B Pts.val.3 Parall.val.1 T.C Pts.val.1 Parall.val.2) :
  distance Ints.val Parall.val.1 = distance Ints.val Parall.val.2 :=
by sorry


end proof_of_distance_equality_l601_601363


namespace find_k_value_l601_601446

noncomputable def find_k (k : ℝ) : Prop :=
  (∃ a d : ℝ, (a - d)^2 = 49 + k ∧ a^2 = 361 + k ∧ (a + d)^2 = 676 + k)

theorem find_k_value : ∃ k : ℝ, find_k k ∧ k ≈ 260.33 :=
by
  sorry

end find_k_value_l601_601446


namespace find_range_of_x_l601_601916

def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem find_range_of_x :
  ∀ x : ℝ, g (2 * x - 1) < g (3) ↔ -1 < x ∧ x < 2 :=
by
  intro x
  unfold g
  sorry

end find_range_of_x_l601_601916


namespace observation_count_l601_601567

theorem observation_count (n : ℤ) (mean_initial : ℝ) (erroneous_value correct_value : ℝ) (mean_corrected : ℝ) :
  mean_initial = 36 →
  erroneous_value = 20 →
  correct_value = 34 →
  mean_corrected = 36.45 →
  n ≥ 0 →
  ∃ n : ℤ, (n * mean_initial + (correct_value - erroneous_value) = n * mean_corrected) ∧ (n = 31) :=
by
  intros h1 h2 h3 h4 h5
  use 31
  sorry

end observation_count_l601_601567


namespace find_number_l601_601194

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601194


namespace find_number_divided_by_3_equals_subtracted_5_l601_601209

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601209


namespace magnitude_eq_2sqrt2_l601_601425

noncomputable def z : ℂ := 1 + complex.i

def zConjugate : ℂ := complex.conj z

theorem magnitude_eq_2sqrt2 : complex.abs (complex.i * z + 3 * zConjugate) = 2 * real.sqrt 2 :=
by sorry

end magnitude_eq_2sqrt2_l601_601425


namespace reciprocal_of_2023_l601_601556

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l601_601556


namespace max_intersection_points_l601_601171

theorem max_intersection_points (C : Set Point) (R : Set Point) (cond1 : ∀ L : Line, (L ∩ C).card ≤ 2) (cond2 : ∀ r : Rectangle, r.sides = 4) :
  ∃ n : ℕ, n = 8 ∧ (intersection_points (circle, rectangle) = n) := sorry

end max_intersection_points_l601_601171


namespace symmetric_table_diagonal_l601_601256

noncomputable def appears_on_diagonal (n : ℕ) (matrix : ℕ → ℕ → ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, 1 ≤ i ∧ i ≤ n ∧ matrix i i = k

theorem symmetric_table_diagonal (n : ℕ) (matrix : ℕ → ℕ → ℕ)
  (h_odd : odd n)
  (h_symmetric : ∀ i j, matrix i j = matrix j i)
  (h_rows : ∀ i, ∀ k (hk : 1 ≤ k ∧ k ≤ n), ∃ j, 1 ≤ j ∧ j ≤ n ∧ matrix i j = k)
  (h_cols : ∀ j, ∀ k (hk : 1 ≤ k ∧ k ≤ n), ∃ i, 1 ≤ i ∧ i ≤ n ∧ matrix i j = k)
  : appears_on_diagonal n matrix :=
by
  sorry

end symmetric_table_diagonal_l601_601256


namespace equal_prob_first_ace_l601_601041

theorem equal_prob_first_ace (deck : List ℕ) (players : Fin 4) (h_deck_size : deck.length = 32)
  (h_distinct : deck.nodup) (h_aces : ∀ _i, deck.filter (λ card, card = 1 ).length = 4)
  (h_shuffled : ∀ (card : ℕ), card ∈ deck → card ∈ (range 32)) :
  ∀ (player : Fin 4), let positions := List.range' (player + 1) (32 / 4) * 4 + player;
  (∀ (pos : ℕ), pos ∈ positions → deck.nth pos = some 1) →
  P(player) = 1 / 8 :=
by
  sorry

end equal_prob_first_ace_l601_601041


namespace product_of_segments_eq_radius_product_l601_601984

variable (r1 r2 : ℝ)
variables (O1 O2 A B C : Point)
variables (circle1 : Circle O1 r1) (circle2 : Circle O2 r2)
variables (tangent1 tangent2 : Line)
variables (internal_tangent : Line)

-- Conditions
axiom circles_outside_each_other : ∥O1 - O2∥ > r1 + r2
axiom external_tangents : tangents tangent1 tangent2 circle1 circle2
axiom internal_tangent_axiom : is_tangent internal_tangent circle1
axiom internal_tangent_intersection1 : intersects internal_tangent tangent1 A
axiom internal_tangent_intersection2 : intersects internal_tangent tangent2 B
axiom internal_tangent_touch : touches internal_tangent circle1 C

theorem product_of_segments_eq_radius_product :
  segment_length A C * segment_length B C = r1 * r2 := sorry

end product_of_segments_eq_radius_product_l601_601984


namespace school_pupils_l601_601982

def girls : ℕ := 868
def difference : ℕ := 281
def boys (g b : ℕ) : Prop := g = b + difference
def total_pupils (g b t : ℕ) : Prop := t = g + b

theorem school_pupils : 
  ∃ b t, boys girls b ∧ total_pupils girls b t ∧ t = 1455 :=
by
  sorry

end school_pupils_l601_601982


namespace ladder_base_distance_l601_601733

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601733


namespace main_theorem_l601_601384

-- Definitions
variables {Point Line Plane : Type}
variables {α β : Plane} {l m : Line}

-- Conditions
variable (l_perp_α : l ⊥ α)
variable (m_in_β : ∀ p : Point, p ∈ m → p ∈ β)

-- Statements to prove
lemma problem_statement (α_parallel_β : α ∥ β) : l ⊥ m :=
sorry

lemma problem_statement_2 (l_parallel_m : l ∥ m) : α ⊦ β :=
sorry

-- The final goal is to assert the correctness of the answers.
theorem main_theorem :
  (α ∥ β → l ⊥ m) ∧ (l ∥ m → α ⊦ β) :=
begin
  split,
  { intro h,
    exact problem_statement l_perp_α m_in_β h },
  { intro h,
    exact problem_statement_2 l_perp_α h }
end

end main_theorem_l601_601384


namespace parallel_lines_a_value_l601_601449

theorem parallel_lines_a_value (a : ℝ) : 
  (a = 0 ∨ a = 1/2) ↔ (∀ x y, x + 2*a*y - 1 = (a - 1)*x - a*y - 1) :=
begin
  sorry
end

end parallel_lines_a_value_l601_601449


namespace ladder_distance_l601_601680

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601680


namespace sin_product_solution_l601_601114

theorem sin_product_solution (n k m : ℕ) (hn : 1 ≤ n ∧ n ≤ 5) (hk : 1 ≤ k ∧ k ≤ 5) (hm : 1 ≤ m ∧ m ≤ 5) :
  (sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔
  ((n = 2 ∧ k = 2 ∧ m = 2) ∨
   (n = 1 ∧ k = 2 ∧ m = 5) ∨
   (n = 1 ∧ k = 5 ∧ m = 2) ∨
   (n = 2 ∧ k = 1 ∧ m = 5) ∨
   (n = 2 ∧ k = 5 ∧ m = 1) ∨
   (n = 5 ∧ k = 1 ∧ m = 2) ∨
   (n = 5 ∧ k = 2 ∧ m = 1)) :=
by sorry

end sin_product_solution_l601_601114


namespace area_of_given_pentagon_is_13_l601_601170

def Point where
  x : ℤ
  y : ℤ

def pentagon_vertices := [
  Point.mk 2 1,
  Point.mk 4 3,
  Point.mk 6 1,
  Point.mk 5 (-2),
  Point.mk 3 (-2)
]

def area_of_pentagon (vertices : List Point) : ℤ := sorry

theorem area_of_given_pentagon_is_13 : 
  area_of_pentagon pentagon_vertices = 13 := 
sorry

end area_of_given_pentagon_is_13_l601_601170


namespace circle_problem_equation_and_intersection_conditions_l601_601383

theorem circle_problem_equation_and_intersection_conditions :
  (∃ (C : ℝ → ℝ → Prop),
     C 1 1 ∧ C -2 -2 ∧
     (∃ (m : ℝ → ℝ → Prop), (∀ (x y : ℝ), m x y ↔ 2 * x - y = 4) ∧ 
       (C (1 + x)/2, C (1 + y)/2 → m x y)) ∧ 
     (∃ (x₁ y₁ x₂ y₂ : ℝ),
       C x₁ y₁ ∧ C x₂ y₂ ∧ 
       (OA x₁ ∧ OB y₁ ∧ OA y₁ ∧ OB x₁ ∧
       ((x₁, x₂) = (-a-1) ∧ x₁ * x₂ = (a^2 + 4a - 4)/2) ∧ 
       2 * x₁ * x₂ + a * (x₁ + x₂) + a^2 = 0 ∧ (a = -4 ∨ a = 1))      
  :=
  ∃ (C : ℝ → ℝ → Prop), 
    (C 1 1 ∧ C -2 -2 ∧
         (∃ (m : ℝ → ℝ → Prop), (∀ x y, m x y ↔ 2 * x - y = 4) ∧
           ∀ (A B : ℝ × ℝ), C A.1 A.2 → C B.1 B.2 → m (A.1 + B.1)/2 (A.2 + B.2)/2) ∧ 
       (∃ x₁ y₁ x₂ y₂, 
         C x₁ y₁ ∧ C x₂ y₂ ∧
         OA x₁ ∧ OB y₁ ∧ OA y₁ ∧ OB x₁ ∧
         ((x₁, x₂) = (-a-1) ∧ x₁ * x₂ = (a^2 + 4a - 4)/2) ∧ 
         2 * x₁ * x₂ + a * (x₁ + x₂) + a^2 = 0 ∧ (a = -4 ∨ a = 1))) :=
begin
  sorry
end

end circle_problem_equation_and_intersection_conditions_l601_601383


namespace lower_limit_for_a_l601_601964

noncomputable def a_lower_limit (a b : ℤ) := 
  a > some_value ∧ a < 15 ∧ b > 6 ∧ b < 21 ∧ (a / (b : ℚ)).natAbs = 1.55

theorem lower_limit_for_a : ∃ a : ℤ, ∀ b : ℤ, a_lower_limit a b → a = 17 :=
begin
  sorry
end

end lower_limit_for_a_l601_601964


namespace Polly_tweets_when_happy_l601_601103

theorem Polly_tweets_when_happy (
  H : ℕ,
  happy_time : ℕ := 20,
  hungry_time : ℕ := 20,
  mirror_time : ℕ := 20,
  hungry_rate : ℕ := 4,
  mirror_rate : ℕ := 45,
  total_tweets : ℕ := 1340
) : H = 18 :=
by
  have happy_tweets : ℕ := happy_time * H
  have hungry_tweets : ℕ := hungry_time * hungry_rate
  have mirror_tweets : ℕ := mirror_time * mirror_rate
  have total : ℕ := happy_tweets + hungry_tweets + mirror_tweets
  assert_eq : total = total_tweets := by 
    rw [<-eq, happy_tweets, hungry_tweets, mirror_tweets]
    exact 1340
  have total_eq : total_tweets = happy_time * H + hungry_time * hungry_rate + mirror_time * mirror_rate := by sorry
  simp at total_eq
  sorry

end Polly_tweets_when_happy_l601_601103


namespace range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l601_601503

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def setB : Set ℝ := {x : ℝ | x < -1 ∨ x > 2}

-- Question (1): Proof statement for A ∩ B = ∅ implying 0 ≤ a ≤ 1
theorem range_of_a_if_intersection_empty (a : ℝ) :
  (setA a ∩ setB = ∅) → (0 ≤ a ∧ a ≤ 1) := 
sorry

-- Question (2): Proof statement for A ∪ B = B implying a ≤ -2 or a ≥ 3
theorem range_of_a_if_union_equal_B (a : ℝ) :
  (setA a ∪ setB = setB) → (a ≤ -2 ∨ 3 ≤ a) := 
sorry

end range_of_a_if_intersection_empty_range_of_a_if_union_equal_B_l601_601503


namespace count_isosceles_points_l601_601232

def Point : Type := ℤ × ℤ

def D : Point := (1, 3)
def E : Point := (5, 3)
def isOnGrid (p : Point) : Prop := 
  ∃ i j : ℤ, 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ p = (i, j)
def midpoint (p1 p2 : Point) : Point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : Point) : ℤ := 
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^(1/2 : ℝ).toReal.toInt

def isIsosceles (A B C : Point) : Prop :=
  distance A B = distance A C ∨ distance A B = distance B C ∨ distance A C = distance B C

def points : List Point := 
  [ (i, j) | i ← [1,2,3,4,5], j ← [1,2,3,4,5], not ((i = 1 ∧ j = 3) ∨ (i = 5 ∧ j = 3)) ]

theorem count_isosceles_points : 
  ∃ F : Point, F ∈ points ∧ isIsosceles D E F :=
sorry

end count_isosceles_points_l601_601232


namespace unique_solution_a_eq_1_l601_601838

noncomputable def unique_solution (a : ℝ) := 
  ∃! x ∈ Set.Ico 0 Real.pi, sin (4*x) * sin (2*x) - sin x * sin (3*x) = a

theorem unique_solution_a_eq_1 : unique_solution 1 :=
sorry

end unique_solution_a_eq_1_l601_601838


namespace smallest_a_with_50_perfect_squares_l601_601888

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l601_601888


namespace smallest_x_multiple_of_1024_l601_601584

theorem smallest_x_multiple_of_1024 (x : ℕ) (hx : 900 * x % 1024 = 0) : x = 256 :=
sorry

end smallest_x_multiple_of_1024_l601_601584


namespace remaining_to_be_paid_l601_601224

-- Given conditions and question
def deposit : ℝ := 55
def deposit_percent : ℝ := 0.10

-- The total cost can be derived from the deposit and its percentage
def total_cost : ℝ := deposit / deposit_percent

-- The remaining amount to be paid is the total cost minus the deposit
def remaining_amount : ℝ := total_cost - deposit

-- The final theorem to be proven
theorem remaining_to_be_paid : remaining_amount = 495 := by
  sorry

end remaining_to_be_paid_l601_601224


namespace sales_exceedance_greatest_percentage_in_May_l601_601784

def sales_data : List (Nat × Nat × Nat) :=
  [(5, 4, 6), (6, 5, 6), (6, 6, 6), (7, 5, 8), (3, 5, 4)]

def percentage_difference (D B F : Nat) : Float :=
  if D = 0 ∨ B = 0 ∨ F = 0 then 0
  else Float.ofNat (Nat.max D (Nat.max B F) - Nat.min D (Nat.min B F)) / 
       Float.ofNat (Nat.min D (Nat.min B F)) * 100

def month_with_highest_percentage_difference : Nat :=
  let percentages := sales_data.map (fun (D, B, F) => percentage_difference D B F)
  let max_diff := List.maximum percentages
  List.indexOf max_diff percentages + 1 -- month index (Jan is 1, Feb is 2, ...) 

theorem sales_exceedance_greatest_percentage_in_May :
  month_with_highest_percentage_difference = 5 :=
sorry

end sales_exceedance_greatest_percentage_in_May_l601_601784


namespace math_proof_problem_l601_601963

noncomputable def p := 1
noncomputable def q := 6
noncomputable def r : ℤ := 8

theorem math_proof_problem : (p + q) * r = 56 := 
by {
  have p_def := show p = 1, by refl,
  have q_def := show q = 6, by refl,
  have r_def := show r = 8, by refl,
  rw [p_def, q_def, r_def],
  norm_num
}

end math_proof_problem_l601_601963


namespace tangent_lines_exist_and_monotonicity_l601_601368

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem tangent_lines_exist_and_monotonicity :
  (∃ m : ℝ, ∀ x : ℝ, f x = x + m ∧ g x = x + m) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x - 1 ∧ g x = k * x - 1) ∧
  (∀ x y : ℝ, (2 : ℝ) / 3 < x → x < y → y < ∞ → g x - f x < g y - f y) :=
  sorry

end tangent_lines_exist_and_monotonicity_l601_601368


namespace floor_property_implies_integer_l601_601843

theorem floor_property_implies_integer (r : ℝ) (h1 : r ≥ 0)
  (h2 : ∀ m n : ℤ, m ∣ n → ⌊m * r⌋ ∣ ⌊n * r⌋) : r ∈ ℤ := 
  sorry

end floor_property_implies_integer_l601_601843


namespace caterpillar_length_difference_l601_601099

-- Define the lengths of the caterpillars
def green_caterpillar_length : ℝ := 3
def orange_caterpillar_length : ℝ := 1.17

-- State the theorem we need to prove
theorem caterpillar_length_difference :
  green_caterpillar_length - orange_caterpillar_length = 1.83 :=
by
  sorry

end caterpillar_length_difference_l601_601099


namespace smallest_natural_with_50_perfect_squares_l601_601879

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601879


namespace Phillip_correct_answers_l601_601102

def C (x : ℕ) := 2 * x
def L (y : ℕ) := 3 * y + 10
def H (z : ℕ) := z^2 + 5 * z

def R (x : ℕ) := 70 + 0.5 * x
def S (y : ℕ) := 90 - y^2
def T (z : ℕ) := 100 - z * (5 - z)

theorem Phillip_correct_answers :
  let x := 5 in
  let y := 3 in
  let z := 4 in
  let Cx := C x in
  let Ly := L y in
  let Hz := H z in
  let Rx := R x / 100 in
  let Sy := S y / 100 in
  let Tz := T z / 100 in
  (Float.round (Cx * Rx) + Float.round (Ly * Sy) + Float.round (Hz * Tz)) = 57 := 
by
  sorry

end Phillip_correct_answers_l601_601102


namespace ineq_x4_y4_l601_601108

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l601_601108


namespace ladder_distance_l601_601678

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601678


namespace floor_smallest_positive_root_of_g_eq_two_l601_601084

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 4 * (1 / Real.cos x)

theorem floor_smallest_positive_root_of_g_eq_two :
  let s := Inf {x : ℝ | 0 < x ∧ g x = 0} in
  ⌊s⌋ = 2 := 
begin
  sorry
end

end floor_smallest_positive_root_of_g_eq_two_l601_601084


namespace angle_between_median_and_bisector_l601_601048

theorem angle_between_median_and_bisector (α : ℝ) :
  ∀ (A B C D E : Type) [triangleABC : ∀ (A B C : Type), A ∧ B ∧ C],
  -- Given conditions
  angle A = 90° ∧ angle ABC = α ∧ is_median BE ∧ is_bisector BD 
  →
  -- Prove the angle between median and bisector
  angle_between_median_and_bisector BD BE = arctan (tan α / 2) - α / 2 :=
by
  sorry

end angle_between_median_and_bisector_l601_601048


namespace ratio_of_area_of_small_triangle_to_square_l601_601571

theorem ratio_of_area_of_small_triangle_to_square
  (n : ℕ)
  (square_area : ℝ)
  (A1 : square_area > 0)
  (ADF_area : ℝ)
  (H1 : ADF_area = n * square_area)
  (FEC_area : ℝ)
  (H2 : FEC_area = 1 / (4 * n)) :
  FEC_area / square_area = 1 / (4 * n) :=
by
  sorry

end ratio_of_area_of_small_triangle_to_square_l601_601571


namespace distance_from_wall_l601_601633

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601633


namespace ladder_base_distance_l601_601728

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601728


namespace prove_correct_statement_l601_601136

def correlation_coefficients (r1 r2 r3 r4 : ℝ) := (r1, r2, r3, r4)

def correct_statement (r1 r2 r3 r4 : ℝ) : Prop :=
  let corrs := correlation_coefficients r1 r2 r3 r4 in
  r1 = 0 ∧ r2 = -0.95 ∧ |r3| = 0.89 ∧ r4 = 0.75 →
  (¬ all_points_on_same_line r1) ∧
  (has_strongest_correlation r2 corrs) ∧
  (¬ has_strongest_correlation r3 corrs) ∧
  (¬ has_weakest_correlation r4 corrs)

noncomputable def all_points_on_same_line (r : ℝ) : Prop := r = 1 ∨ r = -1

noncomputable def has_strongest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≥ |r'|

noncomputable def has_weakest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≤ |r'|

theorem prove_correct_statement :
  ∃ (r1 r2 r3 r4 : ℝ), correct_statement r1 r2 r3 r4 :=
by 
  use 0, -0.95, 0.89, 0.75
  sorry

end prove_correct_statement_l601_601136


namespace problem_proof_l601_601911

open Set

variable {α : Type*}

def A : Set ℤ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℕ := {x | x < 10}
def C : Set ℤ := {x | x < 5}

theorem problem_proof : 
  (A ∩ (B : Set ℤ) = {3, 4, 5, 6} ∧ (compl A ∪ C) = {x | x < 5 ∨ 7 ≤ x}) :=
by
  sorry

end problem_proof_l601_601911


namespace hexagon_properties_l601_601061

-- Given the conditions of the equiangular hexagon and known segment lengths
variables (α : ℝ) (AB BC CD DE EF FA : ℝ)

-- Conditions:
#check 4 = AB
#check 5 = BC
#check 2 = CD
#check 3 = DE

-- Prove:
theorem hexagon_properties :
  α = 120 ∧ EF = 6 ∧ FA = 1 ∧
  let area := (65 * Real.sqrt 3) / 4 in -- represents area of the hexagon
  True :=
by
  sorry

end hexagon_properties_l601_601061


namespace angle_AEC_is_right_angle_l601_601809

-- Define the necessary parameters and conditions
variables {A B C E : Type} [EuclideanGeometry A]
variable {dist : Point → Point → ℝ}

-- Given conditions
def is_isosceles_triangle (A B C : Point) :=
  dist C A = 15 ∧ dist C B = 15 ∧ dist A B = 24

def circle_center_radius (C : Point) (radius : ℝ) (p : Point) :=
  dist C p = radius

def point_on_line_extension (A B E : Point) :=
  ∃ t : ℝ, E = B + t • (B - A)

def point_on_circle (C : Point) (r : ℝ) (E : Point) :=
  dist C E = r

-- The proof problem
theorem angle_AEC_is_right_angle {A B C E : Point}
  (h_iso : is_isosceles_triangle A B C)
  (h_circle : circle_center_radius C 15 A ∧ circle_center_radius C 15 B)
  (h_extension : point_on_line_extension A B E)
  (h_on_circle : point_on_circle C 15 E)
  : ∠ A E C = 90 := sorry

end angle_AEC_is_right_angle_l601_601809


namespace sum_pqrst_is_neg_15_over_2_l601_601496

variable (p q r s t x : ℝ)
variable (h1 : p + 2 = x)
variable (h2 : q + 3 = x)
variable (h3 : r + 4 = x)
variable (h4 : s + 5 = x)
variable (h5 : t + 6 = x)
variable (h6 : p + q + r + s + t + 10 = x)

theorem sum_pqrst_is_neg_15_over_2 : p + q + r + s + t = -15 / 2 := by
  sorry

end sum_pqrst_is_neg_15_over_2_l601_601496


namespace ladder_distance_l601_601700

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601700


namespace measure_of_RPS_l601_601988

-- Assume the elements of the problem
variables {Q R P S : Type}

-- Angles in degrees
def angle_PQS := 35
def angle_QPR := 80
def angle_PSQ := 40

-- Define the angles and the straight line condition
def QRS_straight_line : Prop := true  -- This definition is trivial for a straight line

-- Measure of angle QPS using sum of angles in triangle
noncomputable def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Measure of angle RPS derived from the previous steps
noncomputable def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The statement of the problem in Lean
theorem measure_of_RPS : angle_RPS = 25 := by
  sorry

end measure_of_RPS_l601_601988


namespace min_b_for_factorization_l601_601835

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l601_601835


namespace ab_plus_cd_value_l601_601926

theorem ab_plus_cd_value (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = 1)
  (h3 : a + c + d = 12)
  (h4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := 
sorry

end ab_plus_cd_value_l601_601926


namespace sandy_receives_correct_change_l601_601798

-- Define the costs of each item
def cost_cappuccino : ℕ := 2
def cost_iced_tea : ℕ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℕ := 1

-- Define the quantities ordered
def qty_cappuccino : ℕ := 3
def qty_iced_tea : ℕ := 2
def qty_cafe_latte : ℕ := 2
def qty_espresso : ℕ := 2

-- Calculate the total cost
def total_cost : ℝ := (qty_cappuccino * cost_cappuccino) + 
                      (qty_iced_tea * cost_iced_tea) + 
                      (qty_cafe_latte * cost_cafe_latte) + 
                      (qty_espresso * cost_espresso)

-- Define the amount paid
def amount_paid : ℝ := 20

-- Calculate the change
def change : ℝ := amount_paid - total_cost

theorem sandy_receives_correct_change : change = 3 := by
  -- Detailed steps would go here
  sorry

end sandy_receives_correct_change_l601_601798


namespace volleyball_problem_correct_l601_601261

noncomputable def volleyball_problem : Nat :=
  let total_players := 16
  let triplets : Finset String := {"Alicia", "Amanda", "Anna"}
  let twins : Finset String := {"Beth", "Brenda"}
  let remaining_players := total_players - triplets.card - twins.card
  let no_triplets_no_twins := Nat.choose remaining_players 6
  let one_triplet_no_twins := triplets.card * Nat.choose remaining_players 5
  let no_triplets_one_twin := twins.card * Nat.choose remaining_players 5
  no_triplets_no_twins + one_triplet_no_twins + no_triplets_one_twin

theorem volleyball_problem_correct : volleyball_problem = 2772 := by
  sorry

end volleyball_problem_correct_l601_601261


namespace range_f_l601_601832

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (2 * x - 4)

theorem range_f : set.range f = set.Iio (3 / 2) ∪ set.Ioi (3 / 2) :=
by
  sorry

end range_f_l601_601832


namespace find_smallest_a_l601_601850

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601850


namespace james_lifting_ratio_correct_l601_601792

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l601_601792


namespace reliability_is_correct_l601_601991

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct_l601_601991


namespace trajectory_of_M_l601_601018

-- Definitions of fixed points and conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨3, 0⟩
def B : Point := ⟨-3, 0⟩

-- Definition of the condition on the perimeter
def perimeter (M A B : Point) : ℝ := 
  (Real.sqrt ((M.x - A.x) ^ 2 + (M.y - A.y) ^ 2)) + 
  (Real.sqrt ((M.x - B.x) ^ 2 + (M.y - B.y) ^ 2)) + 
  (Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2))

-- Lean 4 statement of the problem
theorem trajectory_of_M (M : Point) (h : perimeter M A B = 16) : 
  (Real.sqrt ((M.x - A.x) ^ 2 + (M.y - A.y) ^ 2)) + 
  (Real.sqrt ((M.x - B.x) ^ 2 + (M.y - B.y) ^ 2)) = 10
    ↔ (M.x^2 / 25 + M.y^2 / 16 = 1 ∧ -5 < M.x ∧ M.x < 5) := sorry

end trajectory_of_M_l601_601018


namespace choir_avg_age_l601_601978

-- Define the conditions
def females : ℕ := 12
def males : ℕ := 13
def avg_age_females : ℕ := 32
def avg_age_males : ℕ := 33

-- Define the required average age of the choir
def required_avg_age : ℕ := 33

-- Proof statement of the average age of members of the choir
theorem choir_avg_age :
  let total_people := females + males in
  let total_age := (avg_age_females * females) + (avg_age_males * males) in
  let avg_age := total_age / total_people in
  avg_age = required_avg_age := sorry

end choir_avg_age_l601_601978


namespace geometric_sequence_sum_relation_l601_601560

variable {n : ℕ}
variable {A B C q : ℝ}
variable (S : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def geom_seq_property_1 (S : ℕ → ℝ) (A B q : ℝ) :=
  S n = A ∧ S (2 * n) = B ∧ (S (2 * n) - S n) / S n = q^n

def geom_seq_property_2 (S : ℕ → ℝ) (B C q : ℝ) :=
  S (3 * n) = C ∧ (S (3 * n) - S (2 * n)) / (S (2 * n) - S n) = q^n

-- Proof Statement
theorem geometric_sequence_sum_relation (h1 : geom_seq_property_1 S A B q) (h2 : geom_seq_property_2 S B C q) :
  A^2 + B^2 = A * (B + C) := 
sorry

end geometric_sequence_sum_relation_l601_601560


namespace train_crossing_pole_time_l601_601774

-- Train length in meters
def train_length : ℝ := 800

-- Train speed in km/h
def train_speed_kmh : ℝ := 288

-- Conversion factor from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := (speed_kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Calculate time to cross the pole
def time_to_cross (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_crossing_pole_time :
  time_to_cross train_length train_speed_ms = 10 :=
by
  sorry

end train_crossing_pole_time_l601_601774


namespace find_number_divided_by_3_equals_subtracted_5_l601_601211

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601211


namespace second_person_days_l601_601230

theorem second_person_days (h1 : 2 * (1 : ℝ) / 8 = 1) 
                           (h2 : 1 / 24 + x / 24 = 1 / 8) : x = 1 / 12 :=
sorry

end second_person_days_l601_601230


namespace man_fraction_ownership_l601_601248

theorem man_fraction_ownership :
  ∀ (F : ℚ), (3 / 5 * F = 15000) → (75000 = 75000) → (F / 75000 = 1 / 3) :=
by
  intros F h1 h2
  sorry

end man_fraction_ownership_l601_601248


namespace part1_smallest_positive_period_part2_range_of_f_part3_range_of_alpha_l601_601356

def f (x : ℝ) : ℝ := 
  cos x * (2 * real.sqrt 3 * sin x + cos x) + sin (x + 3 * real.pi / 4) * cos (x - real.pi / 4) + 1 / 2

def g (x : ℝ) : ℝ := f (2 * x + real.pi / 6)

theorem part1_smallest_positive_period :
  ∃ T > 0, T = real.pi ∧ ∀ x, f (x + T) = f x := sorry

theorem part2_range_of_f :
  ∀ x, (real.pi / 12 < x ∧ x < real.pi / 2) →
  0 < f x ∧ f x ≤ 3 := sorry

theorem part3_range_of_alpha :
  ∃ α, (-2 * real.pi / 3 ≤ α ∧ α < -real.pi / 2) ∧ ∀ x, (α < x ∧ x < real.pi) → (∃! z, g z = 0) := sorry

end part1_smallest_positive_period_part2_range_of_f_part3_range_of_alpha_l601_601356


namespace number_of_paths_from_0_0_to_6_2_l601_601802

-- Definitions for the problem scenario
def valid_move (p1 p2: ℕ × ℕ) : Prop :=
  -- Move must be right (x increases by 1) or up/down (y changes by 1)
  (p2.1 = p1.1 + 1 ∧ (p2.2 = p1.2 + 1 ∨ p2.2 = p1.2 ∨ p2.2 = p1.2 - 1)) ∧
  -- y-coordinate must be in range [0, 2]
  (0 ≤ p2.2 ∧ p2.2 ≤ 2) ∧
  -- Cannot visit the same point twice
  (p1 ≠ p2)

def valid_path (path : List (ℕ × ℕ)) : Prop :=
  -- Starting point must be (0, 0)
  (path.head = some (0, 0)) ∧
  -- Ending point must be (6, 2)
  (path.reverse.head = some (6, 2)) ∧
  -- Path must consist of valid moves
  ∀ (p1 p2 : ℕ × ℕ), (p1, p2) ∈ path.zip path.tail → valid_move p1 p2

-- The property we want to prove
theorem number_of_paths_from_0_0_to_6_2 : 
  ∃ (paths : List (List (ℕ × ℕ))), 
    (∀ path ∈ paths, valid_path path) ∧ 
    paths.length = 729 := sorry

end number_of_paths_from_0_0_to_6_2_l601_601802


namespace number_of_blue_eyed_students_in_k_class_l601_601097

-- Definitions based on the given conditions
def total_students := 40
def blond_hair_to_blue_eyes_ratio := 2.5
def students_with_both := 8
def students_with_neither := 5

-- We need to prove that the number of blue-eyed students is 10
theorem number_of_blue_eyed_students_in_k_class 
  (x : ℕ)  -- number of blue-eyed students
  (H1 : total_students = 40)
  (H2 : ∀ x, blond_hair_to_blue_eyes_ratio * x = number_of_blond_students)
  (H3 : students_with_both = 8)
  (H4 : students_with_neither = 5)
  : x = 10 :=
sorry

end number_of_blue_eyed_students_in_k_class_l601_601097


namespace ladder_base_distance_l601_601603

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601603


namespace complement_U_M_correct_l601_601006

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 4 * x + 3 = 0}
def complement_U_M : Set ℕ := U \ M

theorem complement_U_M_correct : complement_U_M = {2, 4} :=
by
  -- Proof will be provided here
  sorry

end complement_U_M_correct_l601_601006


namespace solve_inequality_l601_601558

theorem solve_inequality : {x : ℝ | |x - 2| * (x - 1) < 2} = {x : ℝ | x < 3} :=
by
  sorry

end solve_inequality_l601_601558


namespace integer_count_in_interval_l601_601010

theorem integer_count_in_interval : 
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  upper_bound - lower_bound + 1 = 61 :=
by
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  have : upper_bound - lower_bound + 1 = 61 := sorry
  exact this

end integer_count_in_interval_l601_601010


namespace Mike_arcade_playtime_l601_601095

theorem Mike_arcade_playtime :
  (let weekly_pay := 100 in
  let arcade_budget := weekly_pay / 2 in
  let food_expense := 10 in
  let remaining_for_tokens := arcade_budget - food_expense in
  let play_time_per_8_dollars := 300 in
  let sets := remaining_for_tokens / 8 in
  let total_playtime := sets * (play_time_per_8_dollars / sets) in
  total_playtime = 300) := sorry

end Mike_arcade_playtime_l601_601095


namespace mnCardCondition_l601_601459

def distinct_remainders (m n : ℕ) (cards : Fin (m * n) → ℕ) : Prop :=
  ∀ i j k l : Fin (m * n), (i ≠ k ∨ j ≠ l) → (cards i + cards j) % (m * n) ≠ (cards k + cards l) % (m * n)

theorem mnCardCondition (m n : ℕ) (cards : Fin (m * n) → ℕ) : 
  distinct_remainders m n cards ↔ (n = 1 ∨ m = 1) :=
begin
  sorry
end

end mnCardCondition_l601_601459


namespace range_of_expression_positive_range_of_expression_negative_l601_601450

theorem range_of_expression_positive (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 > 0) ↔ (x < -3/2 ∨ x > 4) :=
sorry

theorem range_of_expression_negative (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 < 0) ↔ ( -3/2 < x ∧ x < 4) :=
sorry

end range_of_expression_positive_range_of_expression_negative_l601_601450


namespace dice_sum_probability_l601_601166

theorem dice_sum_probability :
  let total_outcomes := 36
  let sum_le_8_outcomes := 13
  (sum_le_8_outcomes : ℕ) / (total_outcomes : ℕ) = (13 / 18 : ℝ) :=
by
  sorry

end dice_sum_probability_l601_601166


namespace eval_expression_eq_13_div_6_l601_601825

-- Define the given constants
def expr := (Real.sqrt (9 / 4)) + (Real.sqrt (4 / 9))

-- State the theorem to be proven
theorem eval_expression_eq_13_div_6 : expr = 13 / 6 := by
  sorry

end eval_expression_eq_13_div_6_l601_601825


namespace second_projection_at_given_distance_l601_601154

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (point : Point)
  (direction : Point) -- Assume direction is given as a vector

def is_parallel (line1 line2 : Line) : Prop :=
  -- Function to check if two lines are parallel
  sorry

def distance (point1 point2 : Point) : ℝ := 
  -- Function to compute the distance between two points
  sorry

def first_projection_exists (M : Point) (a : Line) : Prop :=
  -- Check the projection outside the line a
  sorry

noncomputable def second_projection
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  Point :=
  sorry

theorem second_projection_at_given_distance
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  distance (second_projection M a d h_parallel h_projection) a.point = d :=
  sorry

end second_projection_at_given_distance_l601_601154


namespace line_equation_l601_601537

theorem line_equation (P : ℝ × ℝ) (θ : ℝ) (hP : P = (-1, 2)) (hθ : θ = π / 4) :
  ∃ a b c : ℝ, a * (-1) + b * 2 + c = 0 ∧ θ = arctan (b / a) ∧ (a, b, c) = (1, -1, 3) :=
by
  have h1 : tan (π / 4) = 1 := Real.tan_pi_div_four
  use [1, -1, 3]
  split
  · calc
      1 * (-1) + (-1) * 2 + 3 = -1 - 2 + 3 := by ring
      ... = 0 := by ring
  split
  · rw [←hθ, h1]
    exact div_one 1
  · rfl

end line_equation_l601_601537


namespace price_of_5_pound_bag_l601_601244

-- Definitions based on conditions
def price_10_pound_bag : ℝ := 20.42
def price_25_pound_bag : ℝ := 32.25
def min_pounds : ℝ := 65
def max_pounds : ℝ := 80
def total_min_cost : ℝ := 98.77

-- Define the sought price of the 5-pound bag in the hypothesis
variable {price_5_pound_bag : ℝ}

-- The theorem to prove based on the given conditions
theorem price_of_5_pound_bag
  (h₁ : price_10_pound_bag = 20.42)
  (h₂ : price_25_pound_bag = 32.25)
  (h₃ : min_pounds = 65)
  (h₄ : max_pounds = 80)
  (h₅ : total_min_cost = 98.77) :
  price_5_pound_bag = 2.02 :=
sorry

end price_of_5_pound_bag_l601_601244


namespace find_number_l601_601189

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l601_601189


namespace coeff_x2_in_x_minus_1_pow_4_l601_601058

theorem coeff_x2_in_x_minus_1_pow_4 :
  ∀ (x : ℝ), (∃ (p : ℕ), (x - 1) ^ 4 = p * x^2 + (other_terms) ∧ p = 6) :=
by sorry

end coeff_x2_in_x_minus_1_pow_4_l601_601058


namespace total_drink_ounces_l601_601070

def total_ounces_entire_drink (coke_parts sprite_parts md_parts coke_ounces : ℕ) : ℕ :=
  let total_parts := coke_parts + sprite_parts + md_parts
  let ounces_per_part := coke_ounces / coke_parts
  total_parts * ounces_per_part

theorem total_drink_ounces (coke_parts sprite_parts md_parts coke_ounces : ℕ) (coke_cond : coke_ounces = 8) (parts_cond : coke_parts = 4 ∧ sprite_parts = 2 ∧ md_parts = 5) : 
  total_ounces_entire_drink coke_parts sprite_parts md_parts coke_ounces = 22 :=
by
  sorry

end total_drink_ounces_l601_601070


namespace range_of_a_l601_601400

noncomputable def f (a x : ℝ) : ℝ := 3*a*x^2 + 2*a*x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 1) → a ∈ Ioc (-3) 0 :=
begin
  sorry -- proof to be done
end

end range_of_a_l601_601400


namespace inequality_holds_for_gt_sqrt2_l601_601105

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l601_601105


namespace unique_solution_m_n_eq_l601_601815

theorem unique_solution_m_n_eq (m n : ℕ) (h : m^2 = (10 * n + 1) * n + 2) : (m, n) = (11, 7) := by
  sorry

end unique_solution_m_n_eq_l601_601815


namespace smallest_a_has_50_perfect_squares_l601_601870

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l601_601870


namespace find_smallest_a_l601_601846

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l601_601846


namespace candidate_x_wins_by_38_33_percent_l601_601454

noncomputable def candidate_x_winning_percentage (n : ℕ) : ℝ :=
  let total_voters : ℕ := 6 * n in
  let voters_r : ℕ := 3 * n in
  let voters_d : ℕ := 2 * n in
  let voters_i : ℕ := n in
  let votes_x_r := 0.85 * (voters_r : ℝ) in
  let votes_x_d := 0.60 * (voters_d : ℝ) in
  let votes_x_i := 0.40 * (voters_i : ℝ) in
  let total_votes_x := votes_x_r + votes_x_d + votes_x_i in
  let votes_y_r := 0.15 * (voters_r : ℝ) in
  let votes_y_d := 0.40 * (voters_d : ℝ) in
  let votes_y_i := 0.60 * (voters_i : ℝ) in
  let total_votes_y := votes_y_r + votes_y_d + votes_y_i in
  ((total_votes_x - total_votes_y) / (total_voters : ℝ)) * 100

theorem candidate_x_wins_by_38_33_percent (n : ℕ) :
  candidate_x_winning_percentage n = 38.33 := by
  sorry

end candidate_x_wins_by_38_33_percent_l601_601454


namespace range_of_a_l601_601949

def A := {x : ℝ | x ≤ 1}
def B (a : ℝ) := {x : ℝ | x ≥ a}

theorem range_of_a (a : ℝ) (h : A ∪ B(a) = set.univ) : a ≤ 1 :=
sorry

end range_of_a_l601_601949


namespace ladder_base_distance_l601_601669

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601669


namespace segment_BC_length_l601_601990

noncomputable def length_segment_BC (AD CD AC : ℝ) (ABC_right : RightTriangle A B C) (ABD_right : RightTriangle A B D) : ℝ :=
  let AB := Real.sqrt (AD ^ 2 - (CD + AC) ^ 2)
  Real.sqrt (AB ^ 2 + AC ^ 2)

theorem segment_BC_length (AD CD AC : ℝ) (ABC_right : RightTriangle A B C) (ABD_right : RightTriangle A B D)
  (h1 : AD = 53) (h2 : CD = 25) (h3 : AC = 20) :
  length_segment_BC AD CD AC ABC_right ABD_right = 4 * Real.sqrt 74 := 
  by
  sorry

end segment_BC_length_l601_601990


namespace cn_dn_fraction_l601_601839

-- Conditions:
def c_n (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1 ᐸ 1), (k + 2 : ℚ) / (Nat.choose (n + 1) k)

def d_n (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1 ᐸ 1), (n + 1 - k : ℚ) / (Nat.choose (n + 1) k)

-- To prove:
theorem cn_dn_fraction (n : ℕ) (h : 0 < n) : (c_n n) / (d_n n) = (n + 1) / (n - 1) :=
by
  sorry

end cn_dn_fraction_l601_601839


namespace arith_seq_general_term_sum_b_n_l601_601921

-- Definitions and conditions
structure ArithSeq (f : ℕ → ℕ) :=
  (d : ℕ)
  (d_ne_zero : d ≠ 0)
  (Sn : ℕ → ℕ)
  (a3_plus_S5 : f 3 + Sn 5 = 42)
  (geom_seq : (f 4)^2 = (f 1) * (f 13))

-- Given the definitions and conditions, prove the general term formula of the sequence
theorem arith_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℕ) 
  (d_ne_zero : d ≠ 0) (a3_plus_S5 : a_n 3 + S_n 5 = 42)
  (geom_seq : (a_n 4)^2 = (a_n 1) * (a_n 13)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_b_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ) (n : ℕ):
  b_n n = 1 / (a_n (n - 1) * a_n n) →
  T_n n = (1 / 2) * (1 - (1 / (2 * n - 1))) →
  T_n n = (n - 1) / (2 * n - 1) :=
sorry

end arith_seq_general_term_sum_b_n_l601_601921


namespace field_length_proof_l601_601546

noncomputable def field_width (w : ℝ) : Prop := w > 0

def pond_side_length : ℝ := 7

def pond_area : ℝ := pond_side_length * pond_side_length

def field_length (w l : ℝ) : Prop := l = 2 * w

def field_area (w l : ℝ) : ℝ := l * w

def pond_area_condition (w l : ℝ) : Prop :=
  pond_area = (1 / 8) * field_area w l

theorem field_length_proof {w l : ℝ} (hw : field_width w)
                           (hl : field_length w l)
                           (hpond : pond_area_condition w l) :
  l = 28 := by
  sorry

end field_length_proof_l601_601546


namespace distance_from_wall_l601_601634

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l601_601634


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601185

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l601_601185


namespace only_six_distinct_sum_to_22_l601_601592

theorem only_six_distinct_sum_to_22 (a b c d e f : ℕ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d)
  (h₃ : a ≠ e) (h₄ : a ≠ f) (h₅ : b ≠ c) (h₆ : b ≠ d) (h₇ : b ≠ e) (h₈ : b ≠ f)
  (h₉ : c ≠ d) (h₁₀ : c ≠ e) (h₁₁ : c ≠ f) (h₁₂ : d ≠ e) (h₁₃ : d ≠ f) (h₁₄ : e ≠ f)
  (h_sum : a + b + c + d + e + f = 22) :
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 ∧ e = 5 ∧ f = 7) ∨
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 ∧ e = 7 ∧ f = 5) ∨
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5 ∧ f = 4) ∨
  -- All permutations need to be considered
  sorry

end only_six_distinct_sum_to_22_l601_601592


namespace ladder_base_distance_l601_601727

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l601_601727


namespace equal_prob_first_ace_l601_601043

/-
  Define the problem:
  In a 4-player card game with a 32-card deck containing 4 aces,
  prove that the probability of each player drawing the first ace is 1/8.
-/

namespace CardGame

def deck : list ℕ := list.range 32

def is_ace (card : ℕ) : Prop := card % 8 = 0

def player_turn (turn : ℕ) : ℕ := turn % 4

def first_ace_turn (deck : list ℕ) : ℕ :=
deck.find_index is_ace

theorem equal_prob_first_ace :
  ∀ (deck : list ℕ) (h : deck.cardinality = 32) (h_ace : ∑ (card ∈ deck) (is_ace card) = 4),
  ∀ (player : ℕ), player < 4 → (∃ n < 32, first_ace_turn deck = some n ∧ player_turn n = player) →
  (deck.countp is_ace) / 32 = 1 / 8 :=
by sorry

end CardGame

end equal_prob_first_ace_l601_601043


namespace stratified_sampling_female_students_l601_601046

-- Definitions from conditions
def male_students : ℕ := 800
def female_students : ℕ := 600
def drawn_male_students : ℕ := 40
def total_students : ℕ := 1400

-- Proof statement
theorem stratified_sampling_female_students : 
  (female_students * drawn_male_students) / male_students = 30 :=
by
  -- substitute and simplify
  sorry

end stratified_sampling_female_students_l601_601046


namespace western_rattlesnake_segments_l601_601820

theorem western_rattlesnake_segments (W : ℕ) (h1 : 6 : ℕ) (h2 : (W - 6) / W = 0.25) : W = 8 := by
  sorry

end western_rattlesnake_segments_l601_601820


namespace largest_ordered_pair_exists_l601_601808

-- Define the condition for ordered pairs (a, b)
def ordered_pair_condition (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ ∃ (k : ℤ), (a + b) * (a + b + 1) = k * a * b

-- Define the specific ordered pair to be checked
def specific_pair (a b : ℤ) : Prop :=
  a = 35 ∧ b = 90

-- The main statement to be proven
theorem largest_ordered_pair_exists : specific_pair 35 90 ∧ ordered_pair_condition 35 90 :=
by
  sorry

end largest_ordered_pair_exists_l601_601808


namespace locus_of_points_in_angle_l601_601601

-- Defining the conditions for the problem
variables (α : ℝ) (inside : Set ℝ → Prop)

-- α > 60 degrees and α ≠ 90 degrees
def angle_condition (α : ℝ) : Prop :=
  α > 60 ∧ α ≠ 90

-- Points inside the angle for which the sum of distances to the sides equals
-- the distance to the vertex, resulting in two rays
noncomputable def geometric_place_two_rays (α : ℝ) : Prop :=
  angle_condition α → inside { p : ℝ | dist p.side1 + dist p.side2 = dist p.vertex }

-- Changes when angles are 60 degrees, < 60 degrees, and 90 degrees
noncomputable def geometric_place_conditions : Prop :=
  (α = 60 → inside { ray }) ∧
  (α < 60 → ¬ inside { p : ℝ | dist p.side1 + dist p.side2 = dist p.vertex }) ∧
  (α = 90 → ¬ inside { p : ℝ | dist p.side1 + dist p.side2 = dist p.vertex })

-- Main theorem statement combining all conditions and results
theorem locus_of_points_in_angle :
  ∀ α, angle_condition α → geometric_place_two_rays α ∧ geometric_place_conditions α := 
sorry

end locus_of_points_in_angle_l601_601601


namespace length_of_train_is_100_meters_l601_601259

-- Define the conditions
def trainSpeed_km_hr : ℝ := 72
def tunnelLength_km : ℝ := 3.5
def passingTime_min : ℝ := 3
def speed_km_per_min := trainSpeed_km_hr / 60
def distance_traveled := speed_km_per_min * passingTime_min

-- The target we need to prove
def lengthOfTrain_km := distance_traveled - tunnelLength_km
def lengthOfTrain_m := lengthOfTrain_km * 1000

-- Lean 4 statement
theorem length_of_train_is_100_meters :
  lengthOfTrain_m = 100 :=
by
  -- Proof is omitted
  sorry

end length_of_train_is_100_meters_l601_601259


namespace ladder_base_distance_l601_601671

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601671


namespace ladder_distance_l601_601682

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l601_601682


namespace original_number_of_coins_in_first_pile_l601_601564

noncomputable def originalCoinsInFirstPile (x y z : ℕ) : ℕ :=
  if h : (2 * (x - y) = 16) ∧ (2 * y - z = 16) ∧ (2 * z - (x + y) = 16) then x else 0

theorem original_number_of_coins_in_first_pile (x y z : ℕ) (h1 : 2 * (x - y) = 16) 
                                              (h2 : 2 * y - z = 16) 
                                              (h3 : 2 * z - (x + y) = 16) : x = 22 :=
by sorry

end original_number_of_coins_in_first_pile_l601_601564


namespace magnitude_sum_vecs_l601_601409

open Real -- Utilizes the math library for sqrt, vectors, etc.

def vec_a : ℝ × ℝ := (1, Real.sqrt 3)
def vec_b : ℝ × ℝ := (-2, 0)

def sum_vecs (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_sum_vecs :
  magnitude (sum_vecs vec_a vec_b) = 2 :=
by
  -- The detailed proof steps would go here
  sorry

end magnitude_sum_vecs_l601_601409


namespace base_distance_l601_601625

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l601_601625


namespace distinct_meals_l601_601278

def num_entrees : ℕ := 4
def num_drinks : ℕ := 2
def num_desserts : ℕ := 2

theorem distinct_meals : num_entrees * num_drinks * num_desserts = 16 := by
  sorry

end distinct_meals_l601_601278


namespace smallest_a_with_50_squares_l601_601905


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601905


namespace strongest_correlation_l601_601134

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end strongest_correlation_l601_601134


namespace ladder_base_distance_l601_601686

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601686


namespace probability_first_ace_equal_l601_601037

theorem probability_first_ace_equal (num_cards : ℕ) (num_aces : ℕ) (num_players : ℕ)
  (h1 : num_cards = 32) (h2 : num_aces = 4) (h3 : num_players = 4) :
  ∀ player : ℕ, player ∈ {1, 2, 3, 4} → (∃ positions : list ℕ, (∀ n ∈ positions, n % num_players = player - 1)) → 
  (positions.length = 8) →
  let P := 1 / 8 in
  P = 1 / num_players :=
begin
  sorry
end

end probability_first_ace_equal_l601_601037


namespace sum_of_possible_x_values_l601_601522

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l601_601522


namespace LCM_divisibility_problem_l601_601080

open Nat

def P := lcm 15 (lcm 16 (lcm 17 (lcm 18 (lcm 19 (lcm 20 (lcm 21 (lcm 22 (lcm 23 (lcm 24 25)))))))))

def Q := lcm P (lcm 26 (lcm 27 (lcm 28 (lcm 29 (lcm 30 31)))))

theorem LCM_divisibility_problem : Q / P = 899 :=
by
  sorry -- Proof to be filled in later.

end LCM_divisibility_problem_l601_601080


namespace event_complement_inter_l601_601104

variables {Ω : Type} {p : ProbabilitySpace Ω}
variables {A B : Event Ω}

theorem event_complement_inter (A B : Event Ω) : 
  Event.complement (A ∪ B) = (Event.complement A) ∩ (Event.complement B) :=
sorry

end event_complement_inter_l601_601104


namespace find_number_eq_seven_point_five_l601_601196

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601196


namespace complex_number_quadrant_l601_601143

theorem complex_number_quadrant (a : ℝ) : 
  (a^2 - 2 = 3 * a - 4) ∧ (a^2 - 2 < 0 ∧ 3 * a - 4 < 0) → a = 1 :=
by
  sorry

end complex_number_quadrant_l601_601143


namespace max_value_f_inequality_ab_l601_601001

-- Define the absolute value function
def abs_val (x : ℝ) : ℝ := if x >= 0 then x else -x

-- Define the function f(x) given in the problem statement
def f (x : ℝ) : ℝ := abs_val (x + 1) - abs_val (x - 2)

-- Problem: Prove that the maximum value of f(x) is 3 given the function definition.
theorem max_value_f : ∃ t, t = 3 ∧ ∀ x : ℝ, f x ≤ t :=
by
  sorry

-- Problem: Given a² + 2b = 1, prove that 2a² + b² ≥ 1/4.
theorem inequality_ab (a b : ℝ) (h : a ^ 2 + 2 * b = 1) : 2 * a ^ 2 + b ^ 2 ≥ 1 / 4 :=
by
  sorry

end max_value_f_inequality_ab_l601_601001


namespace find_pq_l601_601071

-- Define the constants function for the given equation and form
noncomputable def quadratic_eq (p q r : ℤ) : (ℤ × ℤ × ℤ) :=
(2*p*q, p^2 + 2*p*q + q^2 + r, q*q + r)

-- Define the theorem we want to prove
theorem find_pq (p q r: ℤ) (h : quadratic_eq 2 q r = (8, -24, -56)) : pq = -12 :=
by sorry

end find_pq_l601_601071


namespace prove_correct_statement_l601_601135

def correlation_coefficients (r1 r2 r3 r4 : ℝ) := (r1, r2, r3, r4)

def correct_statement (r1 r2 r3 r4 : ℝ) : Prop :=
  let corrs := correlation_coefficients r1 r2 r3 r4 in
  r1 = 0 ∧ r2 = -0.95 ∧ |r3| = 0.89 ∧ r4 = 0.75 →
  (¬ all_points_on_same_line r1) ∧
  (has_strongest_correlation r2 corrs) ∧
  (¬ has_strongest_correlation r3 corrs) ∧
  (¬ has_weakest_correlation r4 corrs)

noncomputable def all_points_on_same_line (r : ℝ) : Prop := r = 1 ∨ r = -1

noncomputable def has_strongest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≥ |r'|

noncomputable def has_weakest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≤ |r'|

theorem prove_correct_statement :
  ∃ (r1 r2 r3 r4 : ℝ), correct_statement r1 r2 r3 r4 :=
by 
  use 0, -0.95, 0.89, 0.75
  sorry

end prove_correct_statement_l601_601135


namespace smallest_a_has_50_perfect_squares_l601_601856

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l601_601856


namespace intersection_of_A_and_B_l601_601927

open Set

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l601_601927


namespace ladder_base_distance_l601_601613

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l601_601613


namespace cube_edge_sum_leq_quarter_l601_601821

theorem cube_edge_sum_leq_quarter (a : Fin 8 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_sum_one : (Finset.univ.sum a) = 1) : 
  (∑ e in (Finset.powersetLen 2 Finset.univ), (a e.1) * (a e.2)) ≤ 1/4 := 
sorry

end cube_edge_sum_leq_quarter_l601_601821


namespace min_value_of_M_l601_601378

noncomputable def min_val (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :=
  max (1/(a*c) + b) (max (1/a + b*c) (a/b + c))

theorem min_value_of_M (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (min_val a b c h1 h2 h3) >= 2 :=
sorry

end min_value_of_M_l601_601378


namespace probability_is_correct_l601_601233

variables (total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items : ℕ)

-- Setting up the problem according to the given conditions
def conditions := (total_items = 10) ∧ 
                  (truckA_first_class = 2) ∧ (truckA_second_class = 2) ∧ 
                  (truckB_first_class = 4) ∧ (truckB_second_class = 2) ∧ 
                  (brokenA = 1) ∧ (brokenB = 1) ∧
                  (remaining_items = 8)

-- Calculating the probability of selecting a first-class item from the remaining items
def probability_of_first_class : ℚ :=
  1/3 * 1/2 + 1/6 * 5/8 + 1/3 * 5/8 + 1/6 * 3/4

-- The theorem to be proved
theorem probability_is_correct : 
  conditions total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items →
  probability_of_first_class = 29/48 :=
sorry

end probability_is_correct_l601_601233


namespace watermelon_cost_l601_601746

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l601_601746


namespace total_combined_cost_l601_601757

theorem total_combined_cost :
  let cost_A := 10 * 6
  let discount_A := 0.10 * cost_A
  let discounted_A := cost_A - discount_A
  let tax_A := 0.08 * cost_A
  let total_A := discounted_A + tax_A

  let cost_B := 12 * 9
  let discount_B := 0.15 * cost_B
  let discounted_B := cost_B - discount_B
  let tax_B := 0.08 * cost_B
  let total_B := discounted_B + tax_B

  total_A + total_B = 159.24 := by
  let cost_A := 10 * 6
  let discount_A := 0.10 * cost_A
  let discounted_A := cost_A - discount_A
  let tax_A := 0.08 * cost_A
  let total_A := discounted_A + tax_A

  let cost_B := 12 * 9
  let discount_B := 0.15 * cost_B
  let discounted_B := cost_B - discount_B
  let tax_B := 0.08 * cost_B
  let total_B := discounted_B + tax_B

  calc
  total_A + total_B
    = discounted_A + tax_A + discounted_B + tax_B : by rfl
    = (cost_A - discount_A) + (0.08 * cost_A) + (cost_B - discount_B) + (0.08 * cost_B) : by rfl
    = 54 + 4.8 + 91.8 + 8.64 : by sorry
    = 159.24 : by rfl

end total_combined_cost_l601_601757


namespace red_balls_count_l601_601976

theorem red_balls_count (R W N_1 N_2 : ℕ) 
  (h1 : R - 2 * N_1 = 18) 
  (h2 : W = 3 * N_1) 
  (h3 : R - 5 * N_2 = 0) 
  (h4 : W - 3 * N_2 = 18)
  : R = 50 :=
sorry

end red_balls_count_l601_601976


namespace equal_prob_first_ace_l601_601044

/-
  Define the problem:
  In a 4-player card game with a 32-card deck containing 4 aces,
  prove that the probability of each player drawing the first ace is 1/8.
-/

namespace CardGame

def deck : list ℕ := list.range 32

def is_ace (card : ℕ) : Prop := card % 8 = 0

def player_turn (turn : ℕ) : ℕ := turn % 4

def first_ace_turn (deck : list ℕ) : ℕ :=
deck.find_index is_ace

theorem equal_prob_first_ace :
  ∀ (deck : list ℕ) (h : deck.cardinality = 32) (h_ace : ∑ (card ∈ deck) (is_ace card) = 4),
  ∀ (player : ℕ), player < 4 → (∃ n < 32, first_ace_turn deck = some n ∧ player_turn n = player) →
  (deck.countp is_ace) / 32 = 1 / 8 :=
by sorry

end CardGame

end equal_prob_first_ace_l601_601044


namespace sum_of_powers_of_i_l601_601151

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end sum_of_powers_of_i_l601_601151


namespace count_valid_triples_l601_601252

def is_similar_prism (a b c : ℕ) : Prop :=
  ∃ x y z, (x ≤ y ∧ y ≤ z) ∧
           (x < a ∧ y < b ∧ z < c) ∧
           a * c = (2023 * 2023) ∧
           y = a ∧ z = b

theorem count_valid_triples :
  ∃ n : ℕ, n = 7 ∧
    ∀ a c, (a ≤ 2023 ∧ 2023 ≤ c) ∧ is_similar_prism a 2023 c →
    (∃! x, a * c = (2023 * 2023) ∧ a ≤ c) :=
begin
  sorry
end

end count_valid_triples_l601_601252


namespace ladder_base_distance_l601_601693

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l601_601693


namespace validate_l601_601117

variables (Line Plane : Type) [Distinct : ∀ (l1 l2 : Line), l1 ≠ l2]
variables [DistinctPlanes : ∀ (p1 p2 p3 : Plane), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3]

-- Define relationships and properties
variables (m n : Line) (α β γ : Plane)

-- Assume basic relationships in space
variables (parallel : Plane → Plane → Prop)
variables (subset  : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)

-- Propositions to be tested
def prop1 := parallel α β ∧ subset m α ∧ subset n β → parallel m n
def prop2 := perpendicular m α ∧ parallel m β → perpendicular α β
def prop3 := perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β
def prop4 := perpendicular α γ ∧ perpendicular β γ ∧ perpendicular m α → perpendicular m β

-- The results of validation
theorem validate : ¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 :=
by {
  -- Proof not included, placeholder sorry used for demonstration purposes
  sorry
}

end validate_l601_601117


namespace smallest_natural_with_50_perfect_squares_l601_601884

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601884


namespace bus_stoppage_time_l601_601313

theorem bus_stoppage_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (reduction_in_speed : speed_excluding_stoppages - speed_including_stoppages = 8) :
  ∃ t : ℝ, t = 9.6 := 
sorry

end bus_stoppage_time_l601_601313


namespace tournament_property_l601_601540

noncomputable def P (n : ℕ) : Prop :=
n ≥ 2 → ∃ p : Fin (n+1), ∀ q : Fin (n+1), p ≠ q → 
  ((∃ b : Fin (n+1), (b ≠ q) ∧ (b ≠ p) ∧ 
    (beaten_by b q ∨
    (∃ c : Fin (n+1), beaten_by c q ∧ beaten_by b c)))

def beaten_by (p q : Fin (n+1)) : Prop := sorry

-- The theorem stating the problem
theorem tournament_property (n : ℕ) : P n := 
sorry

end tournament_property_l601_601540


namespace total_soccer_balls_donated_l601_601761

-- Defining the conditions
def elementary_classes_per_school := 4
def middle_classes_per_school := 5
def schools := 2
def soccer_balls_per_class := 5

-- Proving the total number of soccer balls donated
theorem total_soccer_balls_donated : 
  (soccer_balls_per_class * (schools * (elementary_classes_per_school + middle_classes_per_school))) = 90 := 
by
  -- Using the given conditions to compute the numbers
  let total_classes_per_school := elementary_classes_per_school + middle_classes_per_school
  let total_classes := total_classes_per_school * schools
  let total_soccer_balls := soccer_balls_per_class * total_classes
  -- Prove the equivalence
  show total_soccer_balls = 90
  from sorry

end total_soccer_balls_donated_l601_601761


namespace projection_matrix_ordered_pair_l601_601137

theorem projection_matrix_ordered_pair (a c : ℚ)
  (P : Matrix (Fin 2) (Fin 2) ℚ) 
  (P := ![![a, 15 / 34], ![c, 25 / 34]]) :
  P * P = P ->
  (a, c) = (9 / 34, 15 / 34) :=
by
  sorry

end projection_matrix_ordered_pair_l601_601137


namespace magnitude_expression_l601_601433

noncomputable def z : ℂ := 1 + complex.I

theorem magnitude_expression : |complex.I * z + 3 * complex.conj z| = 2 * real.sqrt 2 :=
by
  sorry

end magnitude_expression_l601_601433


namespace parking_arrangements_count_l601_601462

-- Define the number of parking spaces.
def num_spaces : ℕ := 7

-- Define the number of trucks.
def num_trucks : ℕ := 2

-- Define the number of buses.
def num_buses : ℕ := 2

-- A theorem stating the total number of ways to arrange the trucks and buses given the conditions.
theorem parking_arrangements_count : 
  (∃ (p : Finset (Fin 7)) (t : Finset (Fin 7)) (b : Finset (Fin 7)),
    p.card = num_spaces ∧
    t.card = num_trucks ∧
    disjoint p t ∧ 
    (∀ {a b}, a ∈ t → b ∈ t → a ≠ b → b ≠ a.succ ∧ b ≠ a.pred) ∧
    (∀ {a b}, a ∈ b → b ∈ b → a ≠ b → b ≠ a.succ ∧ b ≠ a.pred))
  → 840 :=
by
  sorry

end parking_arrangements_count_l601_601462


namespace train_speed_kmph_l601_601773

noncomputable def train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) : ℝ :=
  let total_distance := length_train + length_platform
  let speed_mps := total_distance / time_seconds
  speed_mps * 3.6

theorem train_speed_kmph : train_speed 120 213.36 20 = 60.0048 := by
  unfold train_speed
  have h1 : 120 + 213.36 = 333.36 := by norm_num
  have h2 : 333.36 / 20 = 16.668 := by norm_num
  have h3 : 16.668 * 3.6 = 60.0048 := by norm_num
  rw [h1, h2, h3]
  rfl

end train_speed_kmph_l601_601773


namespace sequence_an_eq_n_l601_601377

theorem sequence_an_eq_n (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, n ≥ 1 → a n > 0) 
  (h₁ : ∀ n, n ≥ 1 → a n + 1 / 2 = Real.sqrt (2 * S n + 1 / 4)) : 
  ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end sequence_an_eq_n_l601_601377


namespace solve_phi_l601_601326

theorem solve_phi (n : ℕ) : 
  (∃ (x y z : ℕ), 5 * x + 2 * y + z = 10 * n) → 
  (∃ (φ : ℕ), φ = 5 * n^2 + 4 * n + 1) :=
by 
  sorry

end solve_phi_l601_601326


namespace PQ_equals_half_perimeter_l601_601227

variables {A B C D P Q M N : Type}
variable [DecidableEq P]
variable [DecidableEq Q]
variables (AB BC AD CD : ℕ)

-- Definition of trapezoid and base
structure is_trapezoid (A B C D : Type) :=
(base : A ↔ D)

def midpoint (x y : Type) : Type := sorry

-- Given conditions as Lean definitions
def exterior_angle_bisectors_intersect (A B C D P Q : Type) : Prop := sorry

-- Aim of the proof
theorem PQ_equals_half_perimeter (h : is_trapezoid A B C D) 
  (h1 : exterior_angle_bisectors_intersect A B P)
  (h2 : exterior_angle_bisectors_intersect C D Q)
  (M := midpoint A B)
  (N := midpoint C D) :
  PQ = (AB + BC + AD + CD) / 2 :=
sorry

end PQ_equals_half_perimeter_l601_601227


namespace median_eq_altitude_eq_perp_bisector_eq_l601_601370

open Real

def point := ℝ × ℝ

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

-- Median on BC
theorem median_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = ((1 + (-1))/2, (1 + 0)/2) → x = 1 :=
by
  intros x y h
  sorry

-- Altitude on BC
theorem altitude_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x - 1) / (y - 3) = -4 → 4*x + y - 7 = 0 :=
by
  intros x y h
  sorry

-- Perpendicular bisector of BC
theorem perp_bisector_eq : ∀ (x y : ℝ), (x = 1 ∧ y = 1/2) ∨ (x - 1) / (y - 1/2) = -4 
                          → 8*x + 2*y - 9 = 0 :=
by
  intros x y h
  sorry

end median_eq_altitude_eq_perp_bisector_eq_l601_601370


namespace closest_integer_power_of_eight_l601_601529

theorem closest_integer_power_of_eight : 
  ∃ n : ℤ, 2^(3 * n / 5) ≈ 100 ∧ n = 11 :=
by
  sorry

end closest_integer_power_of_eight_l601_601529


namespace janet_total_miles_run_l601_601072

/-- Janet was practicing for a marathon. She practiced for 9 days, running 8 miles each day.
Prove that Janet ran 72 miles in total. -/
theorem janet_total_miles_run (days_practiced : ℕ) (miles_per_day : ℕ) (total_miles : ℕ) 
  (h1 : days_practiced = 9) (h2 : miles_per_day = 8) : total_miles = 72 := by
  sorry

end janet_total_miles_run_l601_601072


namespace p_lies_on_locus_l601_601359

noncomputable def locus_of_points 
  (A B C D P : Type) [convex_quadrilateral A B C D] 
  (SAPB SAPC SABC SPAB SPCD SPBC SPDA : ℝ) : Prop :=
  SABC * (SPAB * SPCD) = (SPBC * SPDA)

theorem p_lies_on_locus 
  (A B C D P : Type) [convex_quadrilateral A B C D]
  (hyp : locus_of_points A B C D P SAPB SAPC SABC SPAB SPCD SPBC SPDA)
  : P ∈ (conic_through A B C D ∪ line_segment A C ∪ line_segment B D) :=
begin
  sorry
end

end p_lies_on_locus_l601_601359


namespace ladder_base_distance_l601_601673

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l601_601673


namespace triangle_AC_eq_4_l601_601974

open EuclideanGeometry

theorem triangle_AC_eq_4 (A B C D F : Point)
  (h1: Triangle ABC)
  (h2: D ∈ LineSegment AC)
  (h3: F ∈ LineSegment BC)
  (h4: Perpendicular AB AC)
  (h5: Perpendicular AF BC)
  (h6: Midpoint D AC)
  (h7: Distance BD = 2)
  (h8: Distance DC = 2)
  (h9: Distance FC = 2) :
  Distance AC = 4 := 
  sorry

end triangle_AC_eq_4_l601_601974


namespace smallest_natural_with_50_perfect_squares_l601_601881

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l601_601881


namespace area_of_regular_inscribed_octagon_l601_601780

-- Definitions from conditions
def radius : ℝ := 3

-- Problem statement to prove
theorem area_of_regular_inscribed_octagon 
  (h : ∀ (n : ℕ), n = 8 → (radius : ℝ) = 3) :
  ∃ (A : ℝ), A = 18 * √2 := sorry

end area_of_regular_inscribed_octagon_l601_601780


namespace angle_PCQ_inequality_l601_601995

theorem angle_PCQ_inequality (A B C P Q : Type) 
  [Add Angle]
  (a b γ : ℝ)
  (ha : a = segment A C)
  (hb : b = segment B C)
  (hγ : γ = ∠ACB)
  (hγ_gt_90 : γ > 90)
  (P Q : Point)
  (hP : P is a point on segment A C where ⊥ bisector to B exists)
  (hQ : Q is a point on segment B C where ⊥ bisector to A exists)
  (ϕ : ℝ)
  (hϕ_def : ϕ = ∠PCQ) :
  (ϕ = γ - (180 - γ)) <-> (γ <= 135) := 
  sorry

end angle_PCQ_inequality_l601_601995


namespace ladder_distance_from_wall_l601_601713

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l601_601713


namespace find_number_divided_by_3_equals_subtracted_5_l601_601205

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l601_601205


namespace smallest_natural_number_with_50_squares_in_interval_l601_601865

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l601_601865


namespace correct_statements_l601_601063

section SmokingAndLungDisease

-- Define the conditions
def confidence_level : ℝ := 0.99
def smoking_related_to_lung_disease : Prop := sorry  -- Placeholder
def statement1 : Prop := ∀ smokers : ℕ, smokers = 100 → ∃ lung_disease : ℕ, lung_disease ≥ 99
def statement2 : Prop := ∀ person : Type, smoking_related_to_lung_disease → (99 / 100 : ℝ)
def statement3 : Prop := ∀ smokers : ℕ, smokers = 100 → ∃ lung_disease : ℕ, lung_disease ≥ 1
def statement4 : Prop := ∀ smokers : ℕ, smokers = 100 → (∀ lung_disease : ℕ, lung_disease = 0)

-- The proof problem: We want to prove statements 2 and 4 are correct
theorem correct_statements : statement2 ∧ statement4 :=
by
  -- We skip the actual proof as per instructions
  sorry

end SmokingAndLungDisease

end correct_statements_l601_601063


namespace not_irrational_l601_601216

theorem not_irrational :
    ¬ irrational (0.202002) := by
sorry

end not_irrational_l601_601216


namespace smallest_a_with_50_squares_l601_601907


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l601_601907


namespace find_number_eq_seven_point_five_l601_601195

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l601_601195


namespace cube_root_of_neg_eight_squared_is_neg_four_l601_601536

-- Define the value of -8^2
def neg_eight_squared : ℤ := -8^2

-- Define what it means for a number to be the cube root of another number
def is_cube_root (a b : ℤ) : Prop := a^3 = b

-- The desired proof statement
theorem cube_root_of_neg_eight_squared_is_neg_four :
  neg_eight_squared = -64 → is_cube_root (-4) neg_eight_squared :=
by
  sorry

end cube_root_of_neg_eight_squared_is_neg_four_l601_601536


namespace correct_statements_count_l601_601552

def algorithm (s: Prop) (c: Prop) (l: Prop) (i: Prop) : Prop :=
  s ∧ ¬c ∧ l ∧ ¬i

theorem correct_statements_count : ∀ (s: Prop) (c: Prop) (l: Prop) (i: Prop), 
  (algorithm s c l i -> s = true ∧ c = false ∧ l = true ∧ i = false) → 
  (s ∧ l) ∧ ¬(c ∨ i) :=
by
  intros s c l i h
  rcases h (and.intro (and.intro (and.intro trivial
  sorry  -- skipping the actual proof steps


end correct_statements_count_l601_601552


namespace sum_of_ages_l601_601790

/--
Given:
- Beckett's age is 12.
- Olaf is 3 years older than Beckett.
- Shannen is 2 years younger than Olaf.
- Jack is 5 more than twice as old as Shannen.

Prove that the sum of the ages of Beckett, Olaf, Shannen, and Jack is 71 years.
-/
theorem sum_of_ages :
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  Beckett + Olaf + Shannen + Jack = 71 :=
by
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  show Beckett + Olaf + Shannen + Jack = 71
  sorry

end sum_of_ages_l601_601790


namespace hyperbola_eccentricity_range_l601_601002

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  let e := (a^2 + b^2) / a^2
  e ≥ 4

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : b / a ≥ √3) : 
  (a^2 + b^2) / a^2 ≥ 4 :=
sorry

end hyperbola_eccentricity_range_l601_601002


namespace car_fuel_efficiency_l601_601235

theorem car_fuel_efficiency 
  (H C T : ℤ)
  (h₁ : 900 = H * T)
  (h₂ : 600 = C * T)
  (h₃ : C = H - 5) :
  C = 10 := by
  sorry

end car_fuel_efficiency_l601_601235


namespace minimum_swaps_to_order_l601_601155

-- Define the condition of having a disordered 100-volume collection.
variable (volumes : List ℕ)
-- Assume the volumes are uniquely numbered from 1 to 100
-- and initially in a disordered state.
hypothesis (h_volumes_size : volumes.length = 100)
hypothesis (h_unique : volumes.nodup)

-- Define the concept of a swap operation involving elements with different parity.
def parity (n : ℕ) : ℕ := n % 2

-- Defining a swap function to swap elements with different parity.
def swap_parity (l : List ℕ) (i j : ℕ) (h_i : i < l.length) (h_j : j < l.length)
  (h_parity : parity (l.nth_le i h_i) ≠ parity (l.nth_le j h_j)) : List ℕ :=
l.modify_nth i (fun _ => l.nth_le j h_j) |>.modify_nth j (fun _ => l.nth_le i h_i)

-- The main theorem stating the minimum number of swaps to sort the list.
theorem minimum_swaps_to_order :
  ∃ k : ℕ, k = 124 ∧
  ∃ swaps : List (ℕ × ℕ),  
  (∀ (i j : ℕ), (i, j) ∈ swaps → parity (volumes.nth_le i sorry) ≠ parity (volumes.nth_le j sorry)) ∧
  List.foldl (fun l swap => swap_parity l swap.1 swap.2 sorry sorry sorry) volumes swaps = (List.range 100).map (λ n => n + 1)
:= by
sorry

end minimum_swaps_to_order_l601_601155


namespace inequality_generalization_l601_601379

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : n > 0) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2) (h2 : x + 4 / (x ^ 2) = (x / 2) + (x / 2) + 4 / (x ^ 2) ∧ (x / 2) + (x / 2) + 4 / (x ^ 2) ≥ 3) : 
  x + n^n / x^n ≥ n + 1 := 
sorry

end inequality_generalization_l601_601379


namespace lunks_needed_for_20_apples_l601_601960

-- Define the conditions as given in the problem
def lunks_to_kunks (lunks : ℤ) : ℤ := (4 * lunks) / 7
def kunks_to_apples (kunks : ℤ) : ℤ := (5 * kunks) / 3

-- Define the target function to calculate the number of lunks needed for given apples
def apples_to_lunks (apples : ℤ) : ℤ := 
  let kunks := (3 * apples) / 5
  let lunks := (7 * kunks) / 4
  lunks

-- Prove the given problem
theorem lunks_needed_for_20_apples : apples_to_lunks 20 = 21 := by
  sorry

end lunks_needed_for_20_apples_l601_601960


namespace ineq_x4_y4_l601_601107

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l601_601107


namespace midpoint_trajectory_fixed_point_N_l601_601946

-- Conditions
variable (x y : ℝ)
variable (x1 y1 x2 y2 k : ℝ)

-- Problem (I) The trajectory equation of the midpoint C
theorem midpoint_trajectory (B : ℝ × ℝ) (hB : B = (4, 0)) (A : ℝ × ℝ) (hA : (A.1 + 4)^2 + A.2^2 = 16) :
  (2*(A.1 + 4)/2 - 4/2)^2 + (2*A.2/2)^2 = 4 := sorry

-- Problem (II) Fixed point N on the positive half of the x-axis
theorem fixed_point_N (h_line_eq : y = k * (x - 1)) (h_circle_eq : x^2 + y^2 = 4)
  (hyp1 : x1 + x2 = 2 * k^2 / (k^2 + 1)) (hyp2 : x1 * x2 = (k^2 - 4) / (k^2 + 1)) :
  ∃ t : ℝ, t > 0 ∧ (AN_symmetric : (y1 / (x1 - t)) + (y2 / (x2 - t)) = 0) ↔ t = 4 := sorry

end midpoint_trajectory_fixed_point_N_l601_601946


namespace problem_E_is_true_l601_601845

variables (a b c d e f : ℝ)
hypothesis h1 : a > 0
hypothesis h2 : c > 0
hypothesis h3 : b > 0
hypothesis h4 : d > 0
hypothesis h5 : e < 0
hypothesis h6 : f < 0

theorem problem_E_is_true : a > c :=
by sorry

end problem_E_is_true_l601_601845


namespace find_m_l601_601447

theorem find_m (m : ℤ) (h : (-2)^(2*m) = 2^(18 - m)) : m = 6 :=
sorry

end find_m_l601_601447


namespace watermelon_cost_100_l601_601747

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l601_601747


namespace ladder_distance_l601_601708

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l601_601708
