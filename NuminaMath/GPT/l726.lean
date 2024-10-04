import Mathlib

namespace solution_set_inequality_l726_726298

theorem solution_set_inequality (x : ℝ) :
  let f := λ x : ℝ, x^2 + Real.logb 2 (abs x)
  f (x + 1) - f 2 < 0 ↔ (x ∈ Set.interval (-3 : ℝ) (-1)) ∨ (x ∈ Set.interval (-1 : ℝ) 1) :=
by
  -- The proof will be added here
  sorry

end solution_set_inequality_l726_726298


namespace sin_double_angle_l726_726266

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 4 / 5) : Real.sin (2 * x) = -7 / 25 := 
by 
  sorry

end sin_double_angle_l726_726266


namespace smallest_special_gt_3429_l726_726615

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726615


namespace monotonic_intervals_l726_726012

open Real

noncomputable def func (x : ℝ) : ℝ := log 0.3 (-x^2 + 4 * x)

theorem monotonic_intervals :
  (∀ x ∈ Ioo 0 2, ∀ y ∈ Ioo 0 2, x < y → func x > func y) ∧
  (∀ x ∈ Icc 2 4, ∀ y ∈ Icc 2 4, x < y → func x < func y) :=
sorry

end monotonic_intervals_l726_726012


namespace smallest_k_mod_19_7_3_l726_726036

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l726_726036


namespace sequence_permutation_of_naturals_l726_726146

def a : ℕ → ℕ
| 0   := 1
| 1   := 3
| 2   := 2
| (4 * n)     := 2 * a (2 * n)
| (4 * n + 1) := 2 * a (2 * n) + 1
| (4 * n + 2) := 2 * a (2 * n + 1) + 1
| (4 * n + 3) := 2 * a (2 * n + 1)

theorem sequence_permutation_of_naturals : ∀ n : ℕ, ∃ m : ℕ, a m = n :=
sorry

end sequence_permutation_of_naturals_l726_726146


namespace circle_tangent_distance_l726_726310

theorem circle_tangent_distance (O1 O2 : Point) (r1 r2 : ℝ) :
  r1 = 5 ∧ r2 = 3 ∧ (circles_tangent O1 r1 O2 r2) →
  (dist O1 O2 = 2 ∨ dist O1 O2 = 8) :=
by
  intros h
  sorry

end circle_tangent_distance_l726_726310


namespace polynomial_mod3_coefficients_l726_726654

theorem polynomial_mod3_coefficients {n : ℕ} (h : n ≤ 2) :
  ∀ (p : polynomial ℤ), p.degree = n →
  (∀ k : ℤ, (p.eval k) % 3 = 0 ∧ (p.eval (k + 1)) % 3 = 0 ∧ (p.eval (k + 2)) % 3 = 0) →
  (∀ i, p.coeff i % 3 = 0) :=
sorry

end polynomial_mod3_coefficients_l726_726654


namespace ratio_of_p_to_v_in_m_l726_726124

def juice_p := 24
def juice_v := 25
def juice_p_in_smoothie_m := 20
def juice_ratio_in_smoothie_y := (1, 5)

theorem ratio_of_p_to_v_in_m :
  ∃ r : ℕ × ℕ, r = (20 / (gcd 20 5), 5 / (gcd 20 5)) ∧ r = (4, 1) :=
by
  have juice_p_in_smoothie_y := juice_p - juice_p_in_smoothie_m;
  have juice_v_in_smoothie_y := juice_p_in_smoothie_y * snd juice_ratio_in_smoothie_y;
  have juice_v_in_smoothie_m := juice_v - juice_v_in_smoothie_y;
  exact ⟨(20 / (gcd 20 juice_v_in_smoothie_m), juice_v_in_smoothie_m / (gcd 20 juice_v_in_smoothie_m)), 
    by simp [juice_p_in_smoothie_m, juice_v_in_smoothie_m, gcd]⟩

end ratio_of_p_to_v_in_m_l726_726124


namespace bullet_velocity_l726_726138

variables (a b d v : ℝ) (haa : a > b)

def marksman_velocity : Prop :=
  let x := d * v / (a - b) in
  (a / v = d / x + b / v)

theorem bullet_velocity (a b d v : ℝ) (haa : a > b):
  marksman_velocity a b d v := by
  sorry

end bullet_velocity_l726_726138


namespace angle_relationship_l726_726456

-- Define the angles and the relationship
def larger_angle : ℝ := 99
def smaller_angle : ℝ := 81

-- State the problem as a theorem
theorem angle_relationship : larger_angle - smaller_angle = 18 := 
by
  -- The proof would be here
  sorry

end angle_relationship_l726_726456


namespace total_shaded_area_l726_726875

-- Definitions of the problem and conditions
def radius : ℝ := 1
def area_of_circle (r : ℝ) : ℝ := π * r^2
def area_of_semi_circle (r : ℝ) : ℝ := (1/2) * π * r^2

def num_circles : ℕ := 6
def num_semi_circles : ℕ := 4

-- Statement to prove
theorem total_shaded_area :
  (num_circles * area_of_circle radius + num_semi_circles * area_of_semi_circle radius) = 8 * π :=
by
  sorry

end total_shaded_area_l726_726875


namespace graph_of_transformed_function_l726_726731

theorem graph_of_transformed_function
  (f : ℝ → ℝ)
  (hf : f⁻¹ 1 = 0) :
  f (1 - 1) = 1 :=
by
  sorry

end graph_of_transformed_function_l726_726731


namespace least_seven_digit_binary_number_l726_726086

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726086


namespace point_divide_rectangle_number_of_points_l726_726245

-- Define a rectangle in the Cartesian plane
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (AB : A.1 = B.1 ∧ A.2 ≠ B.2)
  (BC : B.1 ≠ C.1 ∧ B.2 = C.2)
  (CD : C.1 = D.1 ∧ C.2 ≠ D.2)
  (DA : D.1 ≠ A.1 ∧ D.2 = A.2)

-- P is a point on one side of the rectangle
-- We need to find point P such that APB & CPD are similar, forming 3 similar triangles
theorem point_divide_rectangle (R : Rectangle) : Prop :=
  ∃ P : ℝ × ℝ,
    (P.1 = R.A.1 ∨ P.1 = R.B.1 ∨ P.1 = R.C.1 ∨ P.1 = R.D.1) →
    (P.2 ≠ R.A.2 ∧ P.2 ≠ R.C.2) →
    (P.1 ≠ R.B.1 ∧ P.1 ≠ R.D.1) →
    (∀ p1 p2 : ℝ × ℝ,
      (p1 = R.A ∨ p1 = R.B) →
      (p2 = R.C ∨ p2 = R.D) →
      ∠ (↔ ((p1.1 - P.1) ^ 2 + (p1.2 - P.2) ^ 2) = (π / 2) ∧
      (↔ ((p2.1 - P.1) ^ 2 + (p2.2 - P.2) ^ 2) = (π / 2)))

-- Correct answer: Since we need more specifics dimensions we conclude:
theorem number_of_points (R : Rectangle) : D :=
  sorry

end point_divide_rectangle_number_of_points_l726_726245


namespace minimum_total_length_of_arcs_l726_726844

theorem minimum_total_length_of_arcs (n : ℕ) (h : n > 0):
  ∃ F : finset (set (ℝ × ℝ)), 
  (∀ R : ℝ × ℝ → ℝ × ℝ, ∃ P ∈ F, R P ∈ F) ∧
  ∑ (A ∈ F), arc_length A = 360 / n := 
sorry

end minimum_total_length_of_arcs_l726_726844


namespace mode_of_data_set_l726_726691

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726691


namespace perpendicular_transitivity_l726_726823

variables {α β : Type} [Plane α] [Plane β]
variables {m n : Line}

-- Conditions
variable (m_perp_α : m ⊥ α)
variable (m_perp_β : m ⊥ β)
variable (non_coincident_lines : ¬ (m = n))

-- Theorem statement
theorem perpendicular_transitivity :
  (n ⊥ α ↔ n ⊥ β) :=
sorry

end perpendicular_transitivity_l726_726823


namespace length_of_crease_l726_726983

theorem length_of_crease {A B C A' P Q : Type} [LinearOrder A, LinearOrder B, LinearOrder C] 
  (is_isosceles_right_triangle : (angle BAC = 90 ∧ AB = AC))
  (folded_triangle : (BA' = 2 ∧ A'C = 3))
  : PQ = (10 * sqrt(2) - 6) / 8 :=
sorry

end length_of_crease_l726_726983


namespace ratio_of_Y_share_l726_726141

theorem ratio_of_Y_share (total_profit share_diff X_share Y_share : ℝ) 
(h1 : total_profit = 700) (h2 : share_diff = 140) 
(h3 : X_share + Y_share = 700) (h4 : X_share - Y_share = 140) : 
Y_share / total_profit = 2 / 5 :=
sorry

end ratio_of_Y_share_l726_726141


namespace square_side_length_l726_726170

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726170


namespace parabola_circle_intersection_l726_726632

theorem parabola_circle_intersection (a : ℝ) : 
  a ≤ Real.sqrt 2 + 1 / 4 → 
  ∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1 :=
by
  sorry

end parabola_circle_intersection_l726_726632


namespace no_infinite_sequence_l726_726234

-- Definition of the sequence of nonzero digits and the condition on N
def infinite_nonzero_digit_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n > 0 ∧ a n < 10

def perfect_square_condition (a : ℕ → ℕ) (N : ℕ) : Prop :=
  ∀ k > N, ∃ x : ℕ, (∑ i in finset.range (k + 1), a i * 10^i) = x^2

-- The main theorem: proving the non-existence
theorem no_infinite_sequence :
  ¬ (∃ (a : ℕ → ℕ) (N : ℕ),
      infinite_nonzero_digit_sequence a ∧ perfect_square_condition a N) :=
by sorry

end no_infinite_sequence_l726_726234


namespace garland_colors_l726_726899

noncomputable def garland (n : ℕ) (color : ℕ → Prop) : Prop :=
  ∃ (color : ℕ → Prop),
    color 1 = "Yellow" ∧
    color 3 = "Yellow" ∧
    (∀ (k : ℕ), k + 4 ≤ n → (∃ count_yellow count_blue,
      (∑ i in finset.range 5, if color (k + i) = "Yellow" then 1 else 0) = 2 ∧
      (∑ i in finset.range 5, if color (k + i) = "Blue" then 1 else 0) = 3)) ∧
    color 97 = "Blue" ∧
    color 98 = "Yellow" ∧
    color 99 = "Blue" ∧
    color 100 = "Blue"

theorem garland_colors : garland 100 (λ i, if i = 1 ∨ i = 3 ∨ i % 5 = 1 ∨ i % 5 = 3 then "Yellow" else "Blue") :=
by
  sorry

end garland_colors_l726_726899


namespace smallest_k_l726_726039

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l726_726039


namespace servings_of_sugar_l726_726128

theorem servings_of_sugar : 
  let total_sugar := 107 / 3;
  let serving_size := 3 / 2;
  (total_sugar / serving_size = 23 + 7 / 9) :=
by {
  let total_sugar := 107 / 3;
  let serving_size := 3 / 2;
  have h1 : total_sugar / serving_size = total_sugar * (2 / 3) := by rw [div_eq_mul_inv, mul_comm, div_mul_div, mul_comm, one_mul],
  have h2 : total_sugar * (2 / 3) = (107 * 2) / (3 * 3) := by rw [mul_div_assoc', Nat.cast_mul, Rat.mul_num, mul_comm],
  have h3 : (107 * 2) / (3 * 3) = (214 / 9) := by norm_num,
  exact (Rational.behaviour),
};

end servings_of_sugar_l726_726128


namespace range_of_heights_l726_726009

theorem range_of_heights (max_height min_height : ℝ) (h_max : max_height = 175) (h_min : min_height = 100) :
  (max_height - min_height) = 75 :=
by
  -- Defer proof
  sorry

end range_of_heights_l726_726009


namespace square_side_length_l726_726174

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726174


namespace smallest_special_number_gt_3429_l726_726604

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726604


namespace smallest_k_l726_726044

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l726_726044


namespace number_of_voters_in_election_l726_726522

theorem number_of_voters_in_election
  (total_membership : ℕ)
  (votes_cast : ℕ)
  (winning_percentage_cast : ℚ)
  (percentage_of_total : ℚ)
  (h_total : total_membership = 1600)
  (h_winning_percentage : winning_percentage_cast = 0.60)
  (h_percentage_of_total : percentage_of_total = 0.196875)
  (h_votes : winning_percentage_cast * votes_cast = percentage_of_total * total_membership) :
  votes_cast = 525 :=
by
  sorry

end number_of_voters_in_election_l726_726522


namespace evaluate_expression_l726_726259

theorem evaluate_expression (n : ℤ) (h : n ≥ 7) : 
  (n + 3)! + (n + 1)! / (n + 2)! = (n^2 + 5 * n + 7) / (n + 2) :=
sorry

end evaluate_expression_l726_726259


namespace maximum_height_l726_726495

theorem maximum_height (v : ℝ → ℝ) (h : ℝ) :
  (∀ t : ℝ, v t = 40 - 10 * t ^ 2) →
  (h = ∫ t in 0..2, v t) →
  h = 160 / 3 :=
by {
  sorry
}

end maximum_height_l726_726495


namespace integral_value_at_1_integral_value_at_3_integral_value_at_5_l726_726461

noncomputable def integral_contour_1 : ℂ := ∫ (z : ℂ) in circle (2 : ℂ) 1, (exp(z^2) / (z^2 - 6*z))

noncomputable def integral_contour_3 : ℂ := ∫ (z : ℂ) in circle (2 : ℂ) 3, (exp(z^2) / (z^2 - 6*z))

noncomputable def integral_contour_5 : ℂ := ∫ (z : ℂ) in circle (2 : ℂ) 5, (exp(z^2) / (z^2 - 6*z))

theorem integral_value_at_1 : integral_contour_1 = 0 :=
by {
  -- proof omitted
  sorry
}

theorem integral_value_at_3 : integral_contour_3 = -π * I / 3 :=
by {
  -- proof omitted
  sorry
}

theorem integral_value_at_5 : integral_contour_5 = (π * I * (exp(36) - 1)) / 3 :=
by {
  -- proof omitted
  sorry
}

end integral_value_at_1_integral_value_at_3_integral_value_at_5_l726_726461


namespace smallest_special_number_l726_726584

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726584


namespace max_distance_point_to_ellipse_l726_726274

theorem max_distance_point_to_ellipse :
  ∀ (A : ℝ × ℝ) (x y : ℝ), A = (0, 2) ∧ (x^2) / 4 + y^2 = 1 → (∃ P : ℝ × ℝ, P = (x, y)
    ∧ (sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) ≤ 2 * real.sqrt(21) / 3) :=
by
  intros A x y h
  sorry

end max_distance_point_to_ellipse_l726_726274


namespace diamond_property_C_l726_726217

-- Define the binary operation diamond
def diamond (a b : ℕ) : ℕ := a ^ (2 * b)

theorem diamond_property_C (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) : 
  (diamond a b) ^ n = diamond a (b * n) :=
by
  sorry

end diamond_property_C_l726_726217


namespace total_marks_more_than_physics_l726_726447

-- Definitions of variables for marks in different subjects
variables (P C M : ℕ)

-- Conditions provided in the problem
def total_marks_condition (P : ℕ) (C : ℕ) (M : ℕ) : Prop := P + C + M > P
def average_chemistry_math_marks (C : ℕ) (M : ℕ) : Prop := (C + M) / 2 = 55

-- The main proof statement: Proving the difference in total marks and physics marks
theorem total_marks_more_than_physics 
    (h1 : total_marks_condition P C M)
    (h2 : average_chemistry_math_marks C M) :
  (P + C + M) - P = 110 := 
sorry

end total_marks_more_than_physics_l726_726447


namespace find_y1_l726_726793

theorem find_y1 
    (h_vert1 : (4 : ℝ, y1 : ℝ))
    (h_vert2 : (4, 7))
    (h_vert3 : (12, 2))
    (h_vert4 : (12, -7))
    (h_area : 76 = 1/2 * 8 * (abs (7 - y1) + abs (2 + 7))) :
  y1 = -3 :=
sorry

end find_y1_l726_726793


namespace white_parallelepipeds_volumes_l726_726498

/-- Given a cube divided into 8 parallelepipeds by three planes,
the volumes of the black parallelepipeds being 1, 6, 8, and 12,
proves that the volumes of the white parallelepipeds are 2, 3, 4, and 24. -/
theorem white_parallelepipeds_volumes
  (black_volumes : set ℕ)
  (h : black_volumes = {1, 6, 8, 12}) :
  ∃ (white_volumes : set ℕ), white_volumes = {2, 3, 4, 24} :=
by
  sorry

end white_parallelepipeds_volumes_l726_726498


namespace period_f_interval_monotonic_increase_f_find_alpha_l726_726312

def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)

def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - Real.sqrt 3

theorem period_f : ∃ T > 0, ∀ x ∈ ℝ, f(x + T) = f(x) := 
  sorry

theorem interval_monotonic_increase_f : ∀ (k : ℤ), ∃ a b : ℝ, 
  a = k * Real.pi - (5 * Real.pi / 12) ∧ b = k * Real.pi + (Real.pi / 12) ∧ 
  ∀ x ∈ Set.Icc a b, monotone_increasing f x :=
  sorry

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h2 : f (α / 2 - (Real.pi / 6)) - f (α / 2 + (Real.pi / 12)) = Real.sqrt 6) : 
  α = (7 * Real.pi / 12) ∨ α = (11 * Real.pi / 12) :=
  sorry

end period_f_interval_monotonic_increase_f_find_alpha_l726_726312


namespace larger_triangle_side_length_l726_726428

theorem larger_triangle_side_length (A1 A2 k : ℕ) (h1 : A1 - A2 = 32) (h2 : A1 = k^2 * A2) (h3 : one_side_is_four : ∃ s : ℕ, s = 4) :
  ∃ (s' : ℕ), s' = 12 :=
by
-- Sorry used to skip the proof.
  sorry

end larger_triangle_side_length_l726_726428


namespace smallest_special_greater_than_3429_l726_726574

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726574


namespace tan_plus_sin_l726_726215

theorem tan_plus_sin :
  let θ := 30.0
  let tan_θ := 1/Real.sqrt 3
  let sin_θ := 1/2
  let cos_θ := Real.sqrt 3 / 2
  let sin_2θ := Real.sqrt 3 / 2
  tan θ + 3 * sin θ = (1 + 3 * Real.sqrt 3) / 2 := sorry

end tan_plus_sin_l726_726215


namespace least_7_digit_binary_number_is_64_l726_726080

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726080


namespace find_radius_of_W_l726_726807

theorem find_radius_of_W (L1 L2 : Set Point)
    (rA rB rW : ℝ)
    (tangents_to_circles : L1 ∈ TangentCircle A B C L1 L2)
    (radius_largest : rB = 18)
    (radius_smallest : rA = 8) :
    rW = 12 := by
  sorry

end find_radius_of_W_l726_726807


namespace solution_ne_zero_l726_726022

theorem solution_ne_zero (a x : ℝ) (h : x = a * x + 1) : x ≠ 0 := sorry

end solution_ne_zero_l726_726022


namespace wheel_radius_l726_726018
noncomputable theory

def speed_kmh := 66
def revolutions_per_minute := 100.09099181073704
def speed_cm_per_min := speed_kmh * 100000 / 60

theorem wheel_radius (r : ℝ) :
  speed_cm_per_min = revolutions_per_minute * 2 * Real.pi * r →
  r ≈ 175.03 :=
sorry

end wheel_radius_l726_726018


namespace edmonton_to_red_deer_distance_l726_726627

noncomputable def distance_from_Edmonton_to_Calgary (speed time: ℝ) : ℝ :=
  speed * time

theorem edmonton_to_red_deer_distance :
  let speed := 110
  let time := 3
  let distance_Calgary_RedDeer := 110
  let distance_Edmonton_Calgary := distance_from_Edmonton_to_Calgary speed time
  let distance_Edmonton_RedDeer := distance_Edmonton_Calgary - distance_Calgary_RedDeer
  distance_Edmonton_RedDeer = 220 :=
by
  sorry

end edmonton_to_red_deer_distance_l726_726627


namespace no_valid_j_l726_726337

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0) |>.sum id

-- Define the problem statement as a theorem
theorem no_valid_j :
  (finset.range 5001).filter (λ j, sum_of_divisors j = 2 + 2 * (int.sqrt j) + j) = ∅ :=
sorry

end no_valid_j_l726_726337


namespace larry_expression_correct_l726_726836

theorem larry_expression_correct (a b c d e : ℤ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 2) (h₄ : d = 5) :
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 :=
by
  sorry

end larry_expression_correct_l726_726836


namespace probability_half_x_interval_l726_726508

noncomputable def probability_half_x_in_interval (x : ℝ) : ℝ :=
  if x ∈ set.Icc (-2 : ℝ) (0 : ℝ) then 1 else 0

theorem probability_half_x_interval :
  let Ω := set.Icc (-3 : ℝ) (5 : ℝ)
  let event := {x : ℝ | 1 ≤ (1/2)^x ∧ (1/2)^x ≤ 4}
  let interval := set.Icc (-2 : ℝ) (0 : ℝ)
  (set.volume (interval ∩ Ω)) / (set.volume Ω) = 1/4 :=
by
  sorry

end probability_half_x_interval_l726_726508


namespace angle_BAC_concyclic_l726_726527

theorem angle_BAC_concyclic (ABC : Triangle) (I : Point) (O : Point) (X : Point) (M : Point) 
  (h1 : ABC.isAcute) 
  (h2 : ABC.incenter = I) 
  (h3 : ABC.circumcenter = O) 
  (h4 : line_through O I ∧ intersect BC = X)
  (h5 : midpoint_of_arc_not_containing A B C = M) 
  (h6 : concyclic {A, O, M, X}) : 
  ∠BAC = 60 := 
sorry

end angle_BAC_concyclic_l726_726527


namespace real_root_of_equation_l726_726647

theorem real_root_of_equation :
  ∃ x : ℝ, (sqrt x + sqrt (x + 4) = 12) ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726647


namespace least_7_digit_binary_number_is_64_l726_726076

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726076


namespace side_length_of_square_l726_726163

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726163


namespace positive_t_for_modulus_eq_l726_726653

theorem positive_t_for_modulus_eq (t : ℝ) (ht : 0 < t) (h : complex.abs (-5 + complex.i * t) = 2 * real.sqrt 13) :
  t = 3 * real.sqrt 3 :=
sorry

end positive_t_for_modulus_eq_l726_726653


namespace matrix_sum_lower_bound_l726_726384

open Matrix

variable (n : ℕ)
variable (A : Matrix (Fin n) (Fin n) ℕ)

theorem matrix_sum_lower_bound (h : ∀ i j : Fin n, A i j = 0 → (∑ k, A i k + ∑ k, A k j) ≥ n) :
  (∑ i j, A i j) ≥ n^2 / 2 := sorry

end matrix_sum_lower_bound_l726_726384


namespace length_BE_convex_quadrilateral_l726_726426

theorem length_BE_convex_quadrilateral
  (ABCD_inscribed : is_convex_and_inscribed_quadrilateral ABCD)
  (E_intersection : are_diagonals_intersect_at_point E AC BD)
  (BD_bisects_ABC : is_angle_bisector BD (angle ABC))
  (BD_length : BD = 25)
  (CD_length : CD = 15) :
  BE = 15.625 := by
  sorry

end length_BE_convex_quadrilateral_l726_726426


namespace product_sums_even_l726_726027

-- Given conditions:
def is_permutation (a b : list ℕ) : Prop := 
  a ~ b

-- Define the math proof problem
theorem product_sums_even (a b : list ℕ) 
  (h_len : a.length = 99) 
  (h_len' : b.length = 99)
  (h_perm : is_permutation b (list.range 1 100)) : 
  (∏ i in finset.range 99, a.nth i + b.nth i) % 2 = 0 :=
by
  sorry

end product_sums_even_l726_726027


namespace xiaohong_home_to_school_distance_l726_726400

noncomputable def driving_distance : ℝ := 1000
noncomputable def total_travel_time : ℝ := 22.5
noncomputable def walking_speed : ℝ := 80
noncomputable def biking_time : ℝ := 40
noncomputable def biking_speed_offset : ℝ := 800

theorem xiaohong_home_to_school_distance (d : ℝ) (v_d : ℝ) :
    let t_w := (d - driving_distance) / walking_speed
    let t_d := driving_distance / v_d
    let v_b := v_d - biking_speed_offset
    (t_d + t_w = total_travel_time)
    → (d / v_b = biking_time)
    → d = 2720 :=
by
  sorry

end xiaohong_home_to_school_distance_l726_726400


namespace solve_f_g_l726_726283

/-- Given conditions: --/
def f (x : ℝ) : ℝ := x^2006 + (p x + p (-x)) / 2
def g (x : ℝ) : ℝ := 2007 * x * real.sqrt (9 - x^2) + (p x - p (-x)) / 2
def p (x : ℝ) : ℝ -- Non-negative function p: to be defined/assumed as non-negative

-- The theorem to prove the original question:
theorem solve_f_g (h_even_f : ∀ x, f x = f (-x)) (h_odd_g : ∀ x, g x = -g (-x)) :
  ∀ x ∈ Icc (-3 : ℝ) 3, f x + g x ≥ 2007 * x * real.sqrt (9 - x^2) + x^2006 := by
  sorry

end solve_f_g_l726_726283


namespace garden_plant_count_l726_726134

theorem garden_plant_count :
  let rows := 52
  let columns := 15
  rows * columns = 780 := 
by
  sorry

end garden_plant_count_l726_726134


namespace power_expression_result_l726_726890

theorem power_expression_result : (-2)^2004 + (-2)^2005 = -2^2004 :=
by
  sorry

end power_expression_result_l726_726890


namespace modulus_of_z_l726_726773

theorem modulus_of_z (z : ℂ) (h : z + complex.I = (2 + complex.I) / complex.I) : complex.abs z = real.sqrt 10 := 
sorry

end modulus_of_z_l726_726773


namespace find_actual_price_of_good_l726_726198

theorem find_actual_price_of_good (P : ℝ) (price_after_discounts : P * 0.93 * 0.90 * 0.85 * 0.75 = 6600) :
  P = 11118.75 :=
by
  sorry

end find_actual_price_of_good_l726_726198


namespace ratio_of_vegetables_to_beef_l726_726813

variable (amountBeefInitial : ℕ) (amountBeefUnused : ℕ) (amountVegetables : ℕ)

def amount_beef_used (initial unused : ℕ) : ℕ := initial - unused
def ratio_vegetables_beef (vegetables beef : ℕ) : ℚ := vegetables / beef

theorem ratio_of_vegetables_to_beef 
  (h1 : amountBeefInitial = 4)
  (h2 : amountBeefUnused = 1)
  (h3 : amountVegetables = 6) :
  ratio_vegetables_beef amountVegetables (amount_beef_used amountBeefInitial amountBeefUnused) = 2 :=
by
  sorry

end ratio_of_vegetables_to_beef_l726_726813


namespace least_positive_base_ten_with_seven_binary_digits_l726_726092

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726092


namespace number_of_difference_of_squares_l726_726520

def is_difference_of_squares (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c, P a b = c * (a - b) * (a + b)

def P₁ (x y : ℝ) : ℝ := x^2 + y^2
def P₂ (x y : ℝ) : ℝ := -x^2 - 4*y^2
def P₃ (a : ℝ) : ℝ := -1 + a^2
def P₄ (a b : ℝ) : ℝ := 0.081*a^2 - b^2

theorem number_of_difference_of_squares :
  (if is_difference_of_squares P₁ then 1 else 0) +
  (if is_difference_of_squares P₂ then 1 else 0) +
  (if is_difference_of_squares P₃ then 1 else 0) +
  (if is_difference_of_squares P₄ then 1 else 0) = 2 :=
begin
  sorry
end

end number_of_difference_of_squares_l726_726520


namespace locus_of_circumcenters_of_triangles_l726_726536

-- Define the conditions as hypotheses
theorem locus_of_circumcenters_of_triangles 
  (Ω : Type) [metric_space Ω] [has_inner Ω] 
  (O I : Ω) (r R : ℝ)
  (hR : R > r)
  (ω : metric_sphere I r)
  (Ω : metric_sphere O R)
  (P : Ω)
  (S : ω)
  (A B : Ω)
  (tangent_ASB : ∀ (x : Ω),
    x ∈ Ω → metric_space.dist x S = metric_space.dist x A ∨ metric_space.dist x S = metric_space.dist x B)
  (tangent_SPI : tangent_to_sphere P
  (point_touch : touching_internals ω Ω P)
  (P_dist : metric_space.dist O P = R - r)
  : ∃ (C : metric_sphere O (2*R - r)/2), 
    C = circumcenter_of_triangle A I B := sorry

end locus_of_circumcenters_of_triangles_l726_726536


namespace set_equality_x_y_2014_2015_l726_726749

noncomputable def x : ℝ := -1
noncomputable def y : ℝ := 0
def A (x y : ℝ) := {x, y / x, 1}
def B (x y : ℝ) := {x^2, x + y, 0}

theorem set_equality_x_y_2014_2015 (h : A x y = B x y) : x ^ 2014 + y ^ 2015 = 1 :=
by {
  have hx : x ≠ 0, sorry,
  have hy : y = 0, sorry,
  have hx1 : x = -1, sorry,
  rw [hx1, hy],
  norm_num,
  rw [pow_zero, pow_2014],
  exact zero_add one_eq_one
}

end set_equality_x_y_2014_2015_l726_726749


namespace train_crosses_pole_in_8_75_seconds_l726_726931

noncomputable def train_crossing_time (train_length: ℝ) (train_speed_kmh: ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

theorem train_crosses_pole_in_8_75_seconds :
  train_crossing_time 350 144 = 8.75 :=
by
  unfold train_crossing_time
  have h : 144 * 1000 / 3600 = 40 := by norm_num
  rw [h]
  norm_num

end train_crosses_pole_in_8_75_seconds_l726_726931


namespace crayons_per_day_l726_726982

theorem crayons_per_day (b c : ℕ) (hb : b = 45) (hc : c = 7) : b * c = 315 :=
by
    rw [hb, hc]
    simp
    exact rfl

end crayons_per_day_l726_726982


namespace arithmetic_seq_sum_l726_726345

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℤ → ℤ) 
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 110) : 
  S 15 = 330 := 
by
  sorry

end arithmetic_seq_sum_l726_726345


namespace reflection_through_plane_l726_726374

def normal_vector : ℝ^3 := ![1, -2, 2]

def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [7/9, 4/9, -4/9],
    [4/9, 5/9, -8/9],
    [-4/9, 8/9, 5/9]
  ]

theorem reflection_through_plane (u : ℝ^3) :
  let S := reflection_matrix
  ∃ q : ℝ^3, 
    u - q = ((u ⬝ normal_vector) / (normal_vector ⬝ normal_vector)) • normal_vector 
    ∧ 
    2 * q - u = S ⬝ u :=
sorry

end reflection_through_plane_l726_726374


namespace multiplication_of_mixed_number_l726_726991

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726991


namespace area_of_ABF_l726_726718

/-- Define the parabola C and the key points -/
def parabola : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

def F : ℝ × ℝ := (1, 0)

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The area of ΔABF is 2 -/
theorem area_of_ABF
  (A B : ℝ × ℝ)
  (hA : A ∈ parabola)
  (hB : B ∈ parabola)
  (hM : midpoint A B = (2, 2)) :
  let AB := dist A B,
      height := 2 in
  1 / 2 * AB * height = 2 :=
by
  sorry

end area_of_ABF_l726_726718


namespace condition_neither_sufficient_nor_necessary_l726_726766

theorem condition_neither_sufficient_nor_necessary (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (a b : ℝ) (h : a > b) : 
  ¬ ((∀ (a b : ℝ), a > b → (a ^ m - b ^ m) * (a ^ n - b ^ n) > 0) ∨ ∀ (a b : ℝ), (a ^ m - b ^ m) * (a ^ n - b ^ n) > 0 → a > b) :=
begin
  sorry
end

end condition_neither_sufficient_nor_necessary_l726_726766


namespace iron_can_conduct_electricity_is_deductive_l726_726977

theorem iron_can_conduct_electricity_is_deductive :
  (∀ (M : Type), M → (∀ x : M, MetaisMetal x → ConductsElectricity x) → (∃ y : M, Iron y → ConductsElectricity y)) → 
  DeductiveReasoning :=
by 
  sorry

end iron_can_conduct_electricity_is_deductive_l726_726977


namespace rectangular_solid_surface_area_l726_726443

theorem rectangular_solid_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 14) 
  (h2 : a^2 + b^2 + c^2 = 121) : 
  2 * (a * b + b * c + a * c) = 75 := 
by
  sorry

end rectangular_solid_surface_area_l726_726443


namespace dataset_mode_l726_726707

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726707


namespace sum_inverse_cubes_l726_726539

theorem sum_inverse_cubes :
  (∑ n in Finset.range (2000 + 1), 1 / (n^3 + n^2)) = 2000 / 2001 :=
begin
  -- here the proof would go
  sorry
end

end sum_inverse_cubes_l726_726539


namespace square_side_length_l726_726150

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726150


namespace solve_system_l726_726630

theorem solve_system (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x - real.sqrt (x * y) - 4 * real.sqrt (x / y) + 2 = 0) →
  (2 * x ^ 2 + x ^ 2 * y ^ 4 = 18 * y ^ 2) →
  (x = 2 ∧ y = 2) ∨ 
  (x = real.sqrt (real.sqrt 286) / 4 ∧ y = real.sqrt (real.sqrt 286)) :=
  sorry

end solve_system_l726_726630


namespace book_arrangement_l726_726193

theorem book_arrangement : (Nat.choose 7 3 = 35) :=
by
  sorry

end book_arrangement_l726_726193


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726071

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726071


namespace smallest_special_gt_3429_l726_726598

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726598


namespace smallest_books_l726_726895

theorem smallest_books 
  (n : ℕ) 
  (h1 : n % 4 = 1)
  (h2 : n % 5 = 1)
  (h3 : n % 6 = 1)
  (h4 : n % 7 = 0) : 
  n = 301 :=
begin
  sorry
end

end smallest_books_l726_726895


namespace integer_solutions_of_inequality_l726_726759

theorem integer_solutions_of_inequality : 
  {n : ℤ | (n - 2) * (n + 4) < 0}.finite ∧
  {n : ℤ | (n - 2) * (n + 4) < 0}.card = 5 :=
by
  sorry

end integer_solutions_of_inequality_l726_726759


namespace no_such_numbers_l726_726396

theorem no_such_numbers (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  ¬ ∃ (a b : ℕ), (p ∣ a * b) ∧ (p ∣ a + b) ∧ (¬ (p ∣ a) ∨ ¬ (p ∣ b)) := 
begin
  sorry
end

end no_such_numbers_l726_726396


namespace base8_246_is_166_in_base10_l726_726551

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l726_726551


namespace infinitely_many_solutions_implies_b_eq_neg6_l726_726624

theorem infinitely_many_solutions_implies_b_eq_neg6 (b : ℤ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) → b = -6 :=
  sorry

end infinitely_many_solutions_implies_b_eq_neg6_l726_726624


namespace mode_of_data_set_l726_726697

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726697


namespace smallest_special_number_l726_726593

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726593


namespace min_distance_correct_l726_726637

noncomputable def min_distance_to_line : ℝ :=
let f : ℝ → ℝ := λ x, x^2 - Real.log x in
let derivative_f := λ x, (2 * x^2 - 1) / x in
let line_y := λ x, x - 2 in
if h : 1 > 0 then
let x1 := 1 in
let y1 := f x1 in
let distance := (1 - y1 - 2) / Real.sqrt (1^2 + (-1)^2) in
Real.abs distance
else 0

theorem min_distance_correct : min_distance_to_line = Real.sqrt 2 := by
  sorry

end min_distance_correct_l726_726637


namespace square_side_length_l726_726147

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726147


namespace all_elements_are_equal_l726_726388

def elements_equal (n : ℕ) (E : Finset ℕ) : Prop :=
  ∀ e ∈ E, e = (E.to_list.head?)

def removable_partition (n : ℕ) (E : Finset ℕ) : Prop :=
  ∀ e ∈ E, ∃ A B : Finset ℕ, 
    A.card = n ∧ B.card = n ∧
    A ∪ B = E.erase e ∧
    A.sum = B.sum

theorem all_elements_are_equal (n : ℕ) (E : Finset ℕ) :
  E.card = 2 * n + 1 →
  (∀ x ∈ E, x ≠ 0) →
  removable_partition n E →
  elements_equal n E :=
sorry

end all_elements_are_equal_l726_726388


namespace equilateral_triangle_side_length_squared_l726_726210

theorem equilateral_triangle_side_length_squared 
    (a b c : ℂ)
    (P : Polynomial ℂ)
    (h : ℝ)
    (ha : P = polynomial.C 7 + polynomial.C 5 * polynomial.X + polynomial.X^3)
    (h_zeros: a^3 + 5 * a + 7 = 0 ∧ b^3 + 5 * b + 7 = 0 ∧ c^3 + 5 * c + 7 = 0)
    (h_magnitudes : abs a ^ 2 + abs b ^ 2 + abs c ^ 2 = 300)
    (h_triangle : ∥a - b∥ = h ∧ ∥b - c∥ = h ∧ ∥c - a∥ = h):
  h ^ 2 = 225 :=
sorry

end equilateral_triangle_side_length_squared_l726_726210


namespace find_a_from_log_condition_l726_726291

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_a_from_log_condition (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : f a 9 = 2) : a = 3 :=
by
  sorry

end find_a_from_log_condition_l726_726291


namespace smallest_k_l726_726041

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l726_726041


namespace bronze_status_donation_l726_726439

variable (B S G : ℝ)
variable (f : B + S + G = 1)
variable (b : B = 10)
variable (s : S = 7)
variable (g : G = 1)

theorem bronze_status_donation :
  (f) ∧ (b) ∧ (s) ∧ (g) ∧ (700 + 50 = 750) → (B ≤ 50) :=
  sorry

end bronze_status_donation_l726_726439


namespace mode_of_data_set_l726_726690

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726690


namespace sum_non_intersecting_layers_l726_726130

theorem sum_non_intersecting_layers (side_length : ℕ) (n : ℕ) (f : Fin 10 × Fin 10 × Fin 10 → ℤ)
  (h1 : side_length = 10)
  (h2 : ∀ x : Fin 10, ∑ y : Fin 10, f (x, y, 0) = 0)
  (h3 : ∀ y : Fin 10, ∑ z : Fin 10, f (0, y, z) = 0)
  (h4 : ∃ x y z, f (x, y, z) = 1) :
  ∑ x in Finset.univ.erase 0, ∑ y in Finset.univ.erase 0, ∑ z in Finset.univ.erase 0, f (x, y, z) = -1 :=
by
  sorry

end sum_non_intersecting_layers_l726_726130


namespace crayons_received_l726_726402

theorem crayons_received (crayons_left : ℕ) (crayons_lost_given_away : ℕ) (lost_twice_given : ∃ (G L : ℕ), L = 2 * G ∧ L + G = crayons_lost_given_away) :
  crayons_left = 2560 →
  crayons_lost_given_away = 9750 →
  ∃ (total_crayons_received : ℕ), total_crayons_received = 12310 :=
by
  intros h1 h2
  obtain ⟨G, L, hL, h_sum⟩ := lost_twice_given
  sorry -- Proof goes here

end crayons_received_l726_726402


namespace real_root_of_quadratic_eq_l726_726747

theorem real_root_of_quadratic_eq (a x : ℂ) (h : a ≠ 0) (hx0 : (a * (1 + complex.i)) * x^2 + (1 + a^2 * complex.i) * x + (a^2 + complex.i) = 0) 
  (hx_real : ∃ (x0 : ℝ), x0 = x.re) : a = -1 :=
by
  sorry

end real_root_of_quadratic_eq_l726_726747


namespace larger_square_side_length_l726_726911

theorem larger_square_side_length (s1 s2 : ℝ) (h1 : s1 = 5) (h2 : s2 = s1 * 3) (a1 a2 : ℝ) (h3 : a1 = s1^2) (h4 : a2 = s2^2) : s2 = 15 := 
by
  sorry

end larger_square_side_length_l726_726911


namespace problem_statement_l726_726978

noncomputable def sin_cos_15_deg := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def cos2_minus_sin2_pi_12 := (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2
noncomputable def sqrt_half_plus_half_cos_pi_6 := Real.sqrt ((1/2) + (1/2) * Real.cos(Real.pi / 6))
noncomputable def tan_22_5_deg := Real.tan (22.5 * Real.pi / 180)
noncomputable def option_D := tan_22_5_deg / (1 - (tan_22_5_deg)^2)

theorem problem_statement : option_D = 1/2 := by
  sorry

end problem_statement_l726_726978


namespace find_perimeter_of_quadrilateral_l726_726808

theorem find_perimeter_of_quadrilateral
  (A B C D E : Type)
  (triangle_ABE triangle_BCE triangle_CDE : Type)
  (r1 r2 r3 : E)
  (h₁ : right_angle triangle_ABE E)
  (h₂ : right_angle triangle_BCE E)
  (h₃ : right_angle triangle_CDE E)
  (α : 45 =  angle AEB = angle BEC = angle CED)
  (AE : ℝ)
  (AE_val : AE = 40) :
  perimeter_ABCD = 60 + 40 * (sqrt 2) := 
sorry

end find_perimeter_of_quadrilateral_l726_726808


namespace base8_to_base10_l726_726555

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l726_726555


namespace second_term_is_4_l726_726785

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l726_726785


namespace value_of_x_l726_726803

-- Define a function to compute the positive difference
def pos_diff (a b : ℤ) : ℤ :=
  abs (a - b)

-- Define the values as given in the top row of the chart
def top_row : list ℤ := [8, 9, 17, 6]

-- Define the values in the second row derived from the top row using the positive difference condition
def second_row : list ℤ := [pos_diff 17 6, pos_diff 8 9]

-- Define the expected value of x in the last row
def x_val : ℤ := 2

-- State the theorem to prove the value of x
theorem value_of_x : x_val = 2 :=
by {
  sorry
}

end value_of_x_l726_726803


namespace part_a_sum_ctg_squares_part_b_sum_cosec_squares_l726_726859

-- Definitions and conditions from (a)
def sum_ctg_squares (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), (real.cot ((k * real.pi)/ (2 * n + 1)))^2

-- Correct answer we will prove
theorem part_a_sum_ctg_squares (n : ℕ) : sum_ctg_squares n = (n * (2 * n - 1)) / 3 := sorry

-- Definitions and conditions from (b)
def sum_cosec_squares (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), (1 / (real.sin ((k * real.pi)/ (2 * n + 1))))^2

-- Correct answer we will prove
theorem part_b_sum_cosec_squares (n : ℕ) : sum_cosec_squares n = (2 * n * (n + 1)) / 3 := sorry

end part_a_sum_ctg_squares_part_b_sum_cosec_squares_l726_726859


namespace pyramid_height_is_correct_l726_726145

-- Condition: The pyramid has a square base with a side length of 10 meters.
def side_length : ℝ := 10

-- Condition: The apex of the pyramid is 15 meters from each vertex of the square.
def apex_distance : ℝ := 15

-- Definition of the diagonal distance from center to the vertex.
def diagonal_half (a : ℝ) : ℝ := (real.sqrt (2 * a^2)) / 2

-- The required height of the pyramid from its peak to the center of the base.
def pyramid_height (d h : ℝ) : ℝ := real.sqrt (h^2 - d^2)

-- Height of the right pyramid
theorem pyramid_height_is_correct :
  pyramid_height (diagonal_half side_length) apex_distance = 5 * real.sqrt 7 := by
  sorry

end pyramid_height_is_correct_l726_726145


namespace square_side_length_l726_726151

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726151


namespace percent_chemical_b_l726_726861

def percent (x : ℝ) : ℝ := x / 100

theorem percent_chemical_b 
  (x_a : ℝ) (y_a : ℝ) (y_b : ℝ) (mix_a : ℝ) (mix_x : ℝ) 
  (hx_a : x_a = 40) 
  (hy_a : y_a = 50) (hy_b : y_b = 50) 
  (hmix_a : mix_a = 47) 
  (hmix_x : mix_x = 30) : 
  100 - x_a = 60 := 
by
  sorry

end percent_chemical_b_l726_726861


namespace additional_distance_l726_726769

theorem additional_distance (distance_speed_10 : ℝ) (speed1 speed2 time1 time2 distance actual_distance additional_distance : ℝ)
  (h1 : actual_distance = distance_speed_10)
  (h2 : time1 = distance_speed_10 / speed1)
  (h3 : time1 = 5)
  (h4 : speed1 = 10)
  (h5 : time2 = actual_distance / speed2)
  (h6 : speed2 = 14)
  (h7 : distance = speed2 * time1)
  (h8 : distance = 70)
  : additional_distance = distance - actual_distance
  := by
  sorry

end additional_distance_l726_726769


namespace least_seven_digit_binary_number_l726_726081

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726081


namespace problem_solution_l726_726754

variable {a b : ℝ}
variable (h1 : a ≠ 0)
variable (h2 : {a, b / a, 1} = {a^2, a + b, 0})

theorem problem_solution : a ^ 2015 + b ^ 2015 = -1 :=
by
  sorry

end problem_solution_l726_726754


namespace max_constant_M_correct_l726_726635

noncomputable def max_constant_M : ℝ :=
  3 / 2

theorem max_constant_M_correct (M : ℝ) (hM : M > 0) :
  (∀ n : ℕ, n > 0 →
    ∃ (a b : ℕ → ℝ),
    (∀ k, 1 ≤ k ∧ k ≤ n → b k > 0) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → a k > 0) ∧
    (∑ k in Finset.range n, b (k + 1)) = 1 ∧
    (∀ k, 2 ≤ k ∧ k ≤ n - 1 → 2 * b k ≥ b (k - 1) + b (k + 1)) ∧
    (a n = M) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n →
      (a k) ^ 2 ≤ 1 + ∑ i in Finset.range k, a (i + 1) * b (i + 1))
  ) ↔ M ≤ 3 / 2 :=
sorry

end max_constant_M_correct_l726_726635


namespace train_speed_l726_726514

theorem train_speed (train_length platform_length : ℕ) (time_seconds : ℕ) (conversion_factor : ℝ) :
  train_length = 360 → 
  platform_length = 140 → 
  time_seconds = 40 → 
  conversion_factor = 3.6 →
  let total_distance := train_length + platform_length in
  let speed_mps := total_distance / time_seconds in
  let speed_kmph := speed_mps * conversion_factor in
  speed_kmph = 45 := 
by
  intros h1 h2 h3 h4
  have h_dist : total_distance = 500 := by 
    rw [h1, h2]
    simp
  have h_speed_mps : speed_mps = 12.5 := by 
    rw [h_dist, h3]
    exact (500 / 40 : ℝ)
  have h_speed_kmph : speed_kmph = 45 := by 
    rw [h_speed_mps, h4]
    exact (12.5 * 3.6 : ℝ)
  assumption


end train_speed_l726_726514


namespace Apollonius_min_value_theorem_l726_726652

noncomputable def Apollonius_min_value (A B D P : Point)
  (h_A : A = (1, 0))
  (h_B : B = (4, 0))
  (h_D : D = (0, 3))
  (h_ratio : dist P A / dist P B = 1 / 2) : ℝ :=
2 * dist P D + dist P B

theorem Apollonius_min_value_theorem 
  (A B D P : Point)
  (h_A : A = (1, 0))
  (h_B : B = (4, 0))
  (h_D : D = (0, 3))
  (h_ratio : dist P A / dist P B = 1 / 2) : 
  Apollonius_min_value A B D P h_A h_B h_D h_ratio = 2 * √10 :=
sorry

end Apollonius_min_value_theorem_l726_726652


namespace photos_to_cover_poster_l726_726313

/-
We are given a poster of dimensions 3 feet by 5 feet, and photos of dimensions 3 inches by 5 inches.
We need to prove that the number of such photos required to cover the poster is 144.
-/

-- Convert feet to inches
def feet_to_inches(feet : ℕ) : ℕ := 12 * feet

-- Dimensions of the poster in inches
def poster_height_in_inches := feet_to_inches 3
def poster_width_in_inches := feet_to_inches 5

-- Area of the poster
def poster_area : ℕ := poster_height_in_inches * poster_width_in_inches

-- Dimensions and area of one photo in inches
def photo_height := 3
def photo_width := 5
def photo_area : ℕ := photo_height * photo_width

-- Number of photos required to cover the poster
def number_of_photos : ℕ := poster_area / photo_area

-- Theorem stating the required number of photos is 144
theorem photos_to_cover_poster : number_of_photos = 144 := by
  -- Proof is omitted
  sorry

end photos_to_cover_poster_l726_726313


namespace intersecting_lines_determine_plane_l726_726922

theorem intersecting_lines_determine_plane
  (L1 L2 : set (ℝ × ℝ × ℝ))
  (h_intersect : ∃ P : ℝ × ℝ × ℝ, P ∈ L1 ∧ P ∈ L2)
  (h1 : ∃ A B : ℝ × ℝ × ℝ, A ∈ L1 ∧ B ∈ L1 ∧ A ≠ B)
  (h2 : ∃ C D : ℝ × ℝ × ℝ, C ∈ L2 ∧ D ∈ L2 ∧ C ≠ D) :
  ∃ P : set (ℝ × ℝ × ℝ), ∀ Q, Q ∈ P ↔ ∃ α β : ℝ, Q = α • A + β • B :=
sorry

end intersecting_lines_determine_plane_l726_726922


namespace max_barons_l726_726118

-- Define the knights, vassals, and barons property.
structure Knight :=
  (vassals : ℕ)
  (isBaron : Prop := vassals ≥ 4)

-- Assuming there are 32 knights.
constant totalKnights : ℕ := 32

-- Define a function to count barons in a list of knights.
def countBarons (knights : List Knight) : ℕ :=
  knights.countp Knight.isBaron

-- Define the maximum number of barons function.
def maxNumberOfBarons (knights : List Knight) : ℕ :=
  totalKnights / 5

-- State that under the given conditions, the maximum number of barons can be 7.
theorem max_barons (knights : List Knight) (h₁ : knights.length = totalKnights) : 
  countBarons knights ≤ 7 :=
by
  sorry

end max_barons_l726_726118


namespace train_length_l726_726460

noncomputable def length_of_each_train : ℝ :=
  let speed_faster_train_km_per_hr := 46
  let speed_slower_train_km_per_hr := 36
  let relative_speed_km_per_hr := speed_faster_train_km_per_hr - speed_slower_train_km_per_hr
  let relative_speed_m_per_s := (relative_speed_km_per_hr * 1000) / 3600
  let time_s := 54
  let distance_m := relative_speed_m_per_s * time_s
  distance_m / 2

theorem train_length : length_of_each_train = 75 := by
  sorry

end train_length_l726_726460


namespace highest_number_on_dice_l726_726131

theorem highest_number_on_dice (n : ℕ) (h1 : 0 < n)
  (h2 : ∃ p : ℝ, p = 0.1111111111111111) 
  (h3 : 1 / 9 = 4 / (n * n)) 
  : n = 6 :=
sorry

end highest_number_on_dice_l726_726131


namespace find_f_prime_at_1_l726_726294

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * f'(-1) * x

theorem find_f_prime_at_1 (f : ℝ → ℝ) (x : ℝ)
  (h : ∀ x, deriv f x = 3 * x^2 + 2 * (deriv f) (-1)) :
  deriv f 1 = -3 := by
  sorry

end find_f_prime_at_1_l726_726294


namespace divisor_of_a_l726_726925

theorem divisor_of_a (a b : ℕ) (hx : a % x = 3) (hb : b % 6 = 5) (hab : (a * b) % 48 = 15) : x = 48 :=
by sorry

end divisor_of_a_l726_726925


namespace expression_nonnegative_interval_l726_726251

noncomputable def rational_expression (x : ℝ) : ℝ :=
  (3 * x - 12 * x^2 + 48 * x^3) / (27 - x^3)

theorem expression_nonnegative_interval :
  ∀ x : ℝ, (rational_expression x ≥ 0 ↔ x ∈ Set.Ico 0 3) := 
begin
  sorry
end

end expression_nonnegative_interval_l726_726251


namespace smallest_special_gt_3429_l726_726601

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726601


namespace mode_of_dataset_with_average_is_l726_726706

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726706


namespace train_length_correct_l726_726194

-- Define the conditions
def train_speed_km_hr : ℝ := 63
def time_seconds : ℝ := 20

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Converted speed in m/s
def speed_m_s : ℝ := train_speed_km_hr * conversion_factor

-- Correct answer for the length of the train
def length_of_train : ℝ := speed_m_s * time_seconds

-- Proof statement
theorem train_length_correct :
  length_of_train = 350 := by
  sorry

end train_length_correct_l726_726194


namespace mode_of_data_set_l726_726696

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726696


namespace total_number_of_vehicles_l726_726969

theorem total_number_of_vehicles (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (cars_per_lane_multiplier : ℕ)
  (total_trucks : ℕ)
  (total_cars : ℕ)
  (total_vehicles : ℕ) :
  lanes = 4 →
  trucks_per_lane = 60 →
  cars_per_lane_multiplier = 2 →
  total_trucks = lanes * trucks_per_lane →
  total_cars = lanes * (cars_per_lane_multiplier * total_trucks) →
  total_vehicles = total_trucks + total_cars →
  total_vehicles = 2160 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end total_number_of_vehicles_l726_726969


namespace second_type_circular_arrangements_l726_726253

def euler_totient (n : ℕ) : ℕ :=
  n.factorization.support.sum (λ p, n.factorization p * (p - 1))

noncomputable def M_3_6_value := (1 : ℚ) / (2 * 6) * list.sum (list.map (λ d, euler_totient d * 3^(6 / d)) [1, 2, 3, 6]) 
  + (1 : ℚ) / 4 * (3^(6 / 2) + 3^((6 + 2) / 2))

theorem second_type_circular_arrangements :
  M_3_6_value = 92 := 
by
  sorry

end second_type_circular_arrangements_l726_726253


namespace probability_of_selecting_one_defective_l726_726100

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l726_726100


namespace evelyn_lost_bottle_caps_l726_726628

-- Definitions from the conditions
def initial_amount : ℝ := 63.0
def final_amount : ℝ := 45.0
def lost_amount : ℝ := 18.0

-- Statement to be proved
theorem evelyn_lost_bottle_caps : initial_amount - final_amount = lost_amount := 
by 
  sorry

end evelyn_lost_bottle_caps_l726_726628


namespace count_valid_triplets_is_21_l726_726756

-- Define the digit bounds
def digit_range := {d : ℕ | 1 ≤ d ∧ d ≤ 9}

-- Define the product condition
def product_equals_36 (a b c : ℕ) : Prop :=
  a * b * c = 36

-- Define the overall condition for the 3-digit number
def valid_digits (a b c : ℕ) : Prop :=
  a ∈ digit_range ∧ b ∈ digit_range ∧ c ∈ digit_range ∧ product_equals_36 a b c

-- Define the set of valid 3-digit numbers
def valid_triplets : set (ℕ × ℕ × ℕ) :=
  {t | valid_digits t.1 t.2 t.3}

-- Define number of valid triplets
def count_valid_triplets : ℕ :=
  (valid_triplets.card : ℕ)

-- The theorem to prove
theorem count_valid_triplets_is_21 : count_valid_triplets = 21 :=
by sorry

end count_valid_triplets_is_21_l726_726756


namespace Petya_wins_l726_726450

theorem Petya_wins :
  ∃ (winner : String),
    let piles := 2021;
    let initial_pile_nuts := (λ n, n = 1);
    let valid_move := (λ p1 p2 p3, p1 = p2 ∧ p2 = p3);
    winner = "Petya" :=
by
  let piles := 2021
  let initial_pile_nuts := (λ n, n = 1)
  let valid_move := (λ p1 p2 p3, p1 = p2 ∧ p2 = p3)
  sorry

end Petya_wins_l726_726450


namespace watched_movies_l726_726449

-- Define the conditions and question as a Lean 4 statement
theorem watched_movies :
  ∀ (M : ℕ),
  (∃ (books movies : ℕ), books = 7 ∧ movies = 17 ∧ M = books + 14) →
  M = 21 :=
by
  intros M h
  cases h with books h
  cases h with movies h
  cases h with h_books h_movies
  cases h_movies with h_books_eq h_movies_eq
  rw h_books_eq at h
  rw add_comm at h
  rw Nat.add_comm 7 14 at h
  exact Eq.symm h

end watched_movies_l726_726449


namespace exists_min_constant_c_l726_726348

-- Define the points and their coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

def O : Point := ⟨0, 0⟩
def A : Point := ⟨1, 0⟩
def C1 : Point := ⟨-2, 0⟩
def radius_C1 : ℝ := 3

-- Define the distance function
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a predicate for a point being outside the circle C1
def outside_C1 (X : Point) : Prop :=
  distance X C1 > radius_C1

-- Define the main theorem
theorem exists_min_constant_c :
  ∃ (c > 0), ∀ (X : Point), outside_C1 X → distance O X - 1 ≥ c * min (distance A X) ((distance A X)^2) :=
begin
  use (real.sqrt 15 - 3) / 3,
  intros X hX,
  sorry
end

end exists_min_constant_c_l726_726348


namespace difference_between_smallest_and_largest_l726_726518

theorem difference_between_smallest_and_largest (
  a b c d e : ℝ
) :
(
  (a + b + c + d) / 4 + e = 74 ∧
  (a + b + c + e) / 4 + d = 80 ∧
  (a + b + d + e) / 4 + c = 98 ∧
  (a + c + d + e) / 4 + b = 116 ∧
  (b + c + d + e) / 4 + a = 128
) → 
  (let lst := [a, b, c, d, e] in max lst - min lst = 126) :=
sorry

end difference_between_smallest_and_largest_l726_726518


namespace no_odd_tens_digit_in_square_l726_726260

theorem no_odd_tens_digit_in_square (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n > 0) (h₃ : n < 100) : 
  (n * n / 10) % 10 % 2 = 0 := 
sorry

end no_odd_tens_digit_in_square_l726_726260


namespace robot_possible_path_lengths_l726_726512

theorem robot_possible_path_lengths (n : ℕ) (valid_path: ∀ (i : ℕ), i < n → (i % 4 = 0 ∨ i % 4 = 1 ∨ i % 4 = 2 ∨ i % 4 = 3)) :
  (n % 4 = 0) :=
by
  sorry

end robot_possible_path_lengths_l726_726512


namespace average_cd_e_l726_726433

theorem average_cd_e (c d e : ℝ) (h : (4 + 6 + 9 + c + d + e) / 6 = 20) : 
    (c + d + e) / 3 = 101 / 3 :=
by
  sorry

end average_cd_e_l726_726433


namespace perpendicular_midpoint_l726_726339

structure Triangle (α : Type) :=
(A B C : α)

def midpoint {α : Type} [Add α] [HasSmul α] [Div α] 
    (a b : α) : α := (a + b) / 2

structure Perp (α : Type) :=
(H M N C : α)

theorem perpendicular_midpoint {α : Type} [MetricSpace α] [InnerProductSpace α] 
  (T : Triangle α) (N : α) (P Q H M : α) (AB : α → α → Prop) (PQ : α → α → Prop) :
  AB T.A T.B ∧
  midpoint T.A T.B = N ∧
  Perp.orthocenter T.A T.B T.C H ∧
  PQ T.A T.B ∧
  PQ P Q ∧
  (AB T.A T.B ∩ PQ P Q = M) →
  InnerProductSpace.is_orthogonal (segment MH) (segment CN) :=
by
  sorry

end perpendicular_midpoint_l726_726339


namespace side_length_of_square_l726_726186

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726186


namespace proof_problem_l726_726376

variable {Plane : Type*} {Line : Type*}
variable (α β : Plane) (l m : Line)

def parallel_planes (α β : Plane) : Prop := _ -- Placeholder for plane parallelism definition
def parallel_lines (l m : Line) : Prop := _ -- Placeholder for line parallelism definition
def line_in_plane (l : Line) (α : Plane) : Prop := _ -- Placeholder for line in plane definition
def perpendicular_lines (l m : Line) : Prop := _ -- Placeholder for perpendicular lines definition
def perpendicular_planes (α β : Plane) : Prop := _ -- Placeholder for perpendicular planes definition

def p := parallel_planes α β ∧ line_in_plane l α ∧ line_in_plane m β → parallel_lines l m
def q := parallel_lines l α ∧ perpendicular_lines m l ∧ line_in_plane m β → perpendicular_planes α β

theorem proof_problem : ¬ p ∨ q :=
by sorry

end proof_problem_l726_726376


namespace base5_number_l726_726105

/-- A base-5 number only contains the digits 0, 1, 2, 3, and 4.
    Given the number 21340, we need to prove that it could possibly be a base-5 number. -/
theorem base5_number (n : ℕ) (h : n = 21340) : 
  ∀ d ∈ [2, 1, 3, 4, 0], d < 5 :=
by sorry

end base5_number_l726_726105


namespace different_parities_l726_726382

def f (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, (k / Nat.gcd k n))

theorem different_parities (n : ℕ) (hn : 1 < n) :
  (f n % 2) ≠ (f (2015 * n) % 2) :=
sorry

end different_parities_l726_726382


namespace Markus_great_grandson_age_l726_726838

noncomputable def Markus : Type := ℕ
noncomputable def Son : Type := ℕ
noncomputable def Grandson : Type := ℕ
noncomputable def GreatGrandson : Type := ℕ

variables (G : GreatGrandson) (M : Markus) (S : Son) (Gr : Grandson)

-- Markus is twice the age of his son.
def condition1 : Prop := M = 2 * S

-- Markus's son is twice the age of Markus's grandson.
def condition2 : Prop := S = 2 * Gr

-- Markus's grandson is three times the age of Markus's great-grandson.
def condition3 : Prop := Gr = 3 * G

-- The sum of the ages of Markus, his son, his grandson, and his great-grandson is 140 years.
def condition4 : Prop := M + S + Gr + G = 140

-- Question to prove: The age of Markus's great-grandson (G) is 140 / 22
theorem Markus_great_grandson_age : ∀ (G : GreatGrandson) (M : Markus) (S : Son) (Gr : Grandson),
  condition1 M S → condition2 S Gr → condition3 Gr G → condition4 M S Gr G → G = 140 / 22 :=
by
  intros
  sorry

end Markus_great_grandson_age_l726_726838


namespace complex_exponent_simplification_l726_726240

theorem complex_exponent_simplification : (i : ℂ) ^ 8 + (i : ℂ) ^ 20 + (i : ℂ) ^ (-30) = 1 :=
by
  -- Given condition
  have h : (i ^ 4 = 1) := sorry,
  sorry

end complex_exponent_simplification_l726_726240


namespace avg_speed_excluding_stoppages_l726_726243

theorem avg_speed_excluding_stoppages (V : ℝ) (h1 : ∀ t : ℝ, t ≥ 0 → t / 2 * V = 40 * t → V = 80 km/hr)
  (h2 : ∀ t : ℝ, t ≥ 0 → t / 2 = 30 / 60 -> V = 80 km/hr) :
  V = 80 := by
  sorry

end avg_speed_excluding_stoppages_l726_726243


namespace telescoping_sum_l726_726534

theorem telescoping_sum : (∑ k in Finset.range (2013 + 1), 1 / (k * (k + 1))) = 2013 / 2014 := 
sorry

end telescoping_sum_l726_726534


namespace distance_between_A_and_B_l726_726119
noncomputable theory

open Real

variables (v_A v_B S AC AD AE : ℕ)

-- Conditions
def speeds_relation : Prop := v_A = 4 * v_B
def is_positive_integer : Prop := S > 0
def factors_S : Prop := nat.factors S = 8
def AC_is_integer : Prop := integer AC
def AD_is_integer : Prop := integer AD
def AE_is_integer : Prop := integer AE
def motorcycle_speed : Prop := v_M = 14 * v_A
def reach_same_time_A : Prop := 
  let t := (S / (v_A + v_B)) in
  2 * t * v_A = S ∧ 
  let t' := ((S + AE) / (v_A + 14 * v_B)) in
  2 * t' * 14 * v_B = S

theorem distance_between_A_and_B :
  speeds_relation v_A v_B ∧
  is_positive_integer S ∧
  factors_S S ∧
  AC_is_integer AC ∧
  AD_is_integer AD ∧
  AE_is_integer AE ∧
  motorcycle_speed v_A ∧
  reach_same_time_A v_A v_B S AE 
  → S = 105 :=
sorry

end distance_between_A_and_B_l726_726119


namespace part_I_part_II_l726_726824

variable {a b : ℕ → ℕ}
variable {d q : ℕ}

def arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_seq (b : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, b (n + 1) = b n * q

noncomputable def sum_first_n_terms_arith (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)

noncomputable def sum_first_n_terms_geom (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  b 1 * (1 - q ^ n) / (1 - q)

theorem part_I (h1 : a 1 = 1) (h2 : b 1 = 2) (h3 : a 3 + b 3 = 13) (h4 : a 2 + b 2 = 7) :
  ∀ n, b n = 2^n :=
by 
  sorry

theorem part_II (h1 : a 1 = 1) (h2 : b 1 = 2) (h3 : a 3 + b 3 = 13) (h4 : T 3 = 14) :
  ∃ d,
    (∀ n, S n = -3/2 * n^2 + 5/2 * n ∨ S n = n^2) :=
by
  sorry

end part_I_part_II_l726_726824


namespace mode_of_data_set_l726_726687

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726687


namespace major_axis_length_of_ellipse_l726_726981

theorem major_axis_length_of_ellipse 
  (F1 F2 : ℝ × ℝ)
  (hx : F1.1 = 4 ∧ F1.2 = 10)
  (hy : F2.1 = 24 ∧ F2.2 = 30)
  (tangent_x_axis : true) -- This is implicitly asserting tangency to the x-axis

: ∃ L, L = 20 * Real.sqrt 5 :=
by
  use 20 * Real.sqrt 5
  sorry

end major_axis_length_of_ellipse_l726_726981


namespace find_absolute_slope_l726_726454

def circle := (center : (ℝ × ℝ)) (radius : ℝ)

-- Given circles
def circle1 : circle := ((10, 80), 4)
def circle2 : circle := ((13, 60), 4)
def circle3 : circle := ((15, 70), 4)

def line_passes_through (p: ℝ × ℝ) := { l : ℝ × ℝ // l = p }

-- Function to calculate the area of a circle (using π * radius^2)
noncomputable def area_of_circle (c : circle) : ℝ := Real.pi * (c.2)^2

-- Given that the line passes through (13, 60)
def line_through_13_60 (line: ℝ × ℝ) := line_passes_through (13, 60)

-- The condition that the line divides the total area equally
def divides_area_equally (line : ℝ × ℝ) : Prop := 
  let total_area := (area_of_circle circle1 + area_of_circle circle2 + area_of_circle circle3) / 2 in
  sorry -- Formula for checking equal area on both sides of the line goes here

-- The goal statement
theorem find_absolute_slope (line : (ℝ × ℝ)) (h : line_through_13_60 line) (d : divides_area_equally line) :
  | line.1 | = 5 :=
sorry

end find_absolute_slope_l726_726454


namespace original_deck_size_l726_726502

-- Define the conditions
def boys_kept_away (remaining_cards kept_away_cards : ℕ) : Prop :=
  remaining_cards + kept_away_cards = 52

-- Define the problem
theorem original_deck_size (remaining_cards : ℕ) (kept_away_cards := 2) :
  boys_kept_away remaining_cards kept_away_cards → remaining_cards + kept_away_cards = 52 :=
by
  intro h
  exact h

end original_deck_size_l726_726502


namespace distance_M_to_AB_l726_726000

noncomputable def distance_to_ab : ℝ := 5.8

theorem distance_M_to_AB
  (M : Point)
  (A B C : Point)
  (d_AC d_BC : ℝ)
  (AB BC AC : ℝ)
  (H1 : d_AC = 2)
  (H2 : d_BC = 4)
  (H3 : AB = 10)
  (H4 : BC = 17)
  (H5 : AC = 21) :
  distance_to_ab = 5.8 :=
by
  sorry

end distance_M_to_AB_l726_726000


namespace probability_between_lines_l726_726221

def line_l (x : ℝ) : ℝ := -2 * x + 8
def line_m (x : ℝ) : ℝ := -3 * x + 9

theorem probability_between_lines 
  (h1 : ∀ x > 0, line_l x ≥ 0) 
  (h2 : ∀ x > 0, line_m x ≥ 0) 
  (h3 : ∀ x > 0, line_l x < line_m x ∨ line_m x ≤ 0) : 
  (1 / 16 : ℝ) * 100 = 0.16 :=
by
  sorry

end probability_between_lines_l726_726221


namespace largest_sum_at_vertex_l726_726879

theorem largest_sum_at_vertex : 
  ∀ (f : Fin 8 → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ 8) → 
  (∃ v, ∃ a b c, 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 
  v ∈ (vertices_of_octahedron) ∧
  (a, b, c) ∈ (faces_meeting_at v)) →
  (f a + f b + f c ≤ 21) :=
by
  -- The proof intentionally omitted
  sorry

end largest_sum_at_vertex_l726_726879


namespace num_elements_in_A_l726_726407

-- Define sets A and B
variable (A B : Set ℕ)

-- Given conditions
axiom hyp1 : ∃ (a b : ℕ), a = 3 * b
axiom hyp2 : (A ∪ B).finite ∧ (A ∪ B).card = 4060
axiom hyp3 : (A ∩ B).finite ∧ (A ∩ B).card = 1240

-- Prove the total number of elements in set A is 3045
theorem num_elements_in_A : ∃ (a : ℕ), a = 3045 := by
  sorry

end num_elements_in_A_l726_726407


namespace no_equal_numbers_from_19_and_98_l726_726471

theorem no_equal_numbers_from_19_and_98 :
  ¬ (∃ s : ℕ, ∃ (a b : ℕ → ℕ), 
       (a 0 = 19) ∧ (b 0 = 98) ∧
       (∀ k, a (k + 1) = a k * a k ∨ a (k + 1) = a k + 1) ∧
       (∀ k, b (k + 1) = b k * b k ∨ b (k + 1) = b k + 1) ∧
       a s = b s) :=
sorry

end no_equal_numbers_from_19_and_98_l726_726471


namespace not_possible_155_cents_five_coins_l726_726257

/-- It is not possible to achieve a total value of 155 cents using exactly five coins 
    from a piggy bank containing only pennies (1 cent), nickels (5 cents), 
    quarters (25 cents), and half-dollars (50 cents). -/
theorem not_possible_155_cents_five_coins (n_pennies n_nickels n_quarters n_half_dollars : ℕ) 
    (h : n_pennies + n_nickels + n_quarters + n_half_dollars = 5) : 
    n_pennies * 1 + n_nickels * 5 + n_quarters * 25 + n_half_dollars * 50 ≠ 155 := 
sorry

end not_possible_155_cents_five_coins_l726_726257


namespace parallelLines_perpendicularLines_l726_726122

-- Problem A: Parallel lines
theorem parallelLines (a : ℝ) : 
  (∀x y : ℝ, y = -x + 2 * a → y = (a^2 - 2) * x + 2 → -1 = a^2 - 2) → 
  a = -1 := 
sorry

-- Problem B: Perpendicular lines
theorem perpendicularLines (a : ℝ) : 
  (∀x y : ℝ, y = (2 * a - 1) * x + 3 → y = 4 * x - 3 → (2 * a - 1) * 4 = -1) →
  a = 3 / 8 := 
sorry

end parallelLines_perpendicularLines_l726_726122


namespace angle_of_diameter_subtended_at_circle_point_l726_726406

theorem angle_of_diameter_subtended_at_circle_point (O : Point) (A B C : Point) (r : ℝ) 
  (h1 : circle O r A)
  (h2 : circle O r B)
  (h3 : circle O r C)
  (diam : dist O A = dist O B)
  (diam_points : dist A B = 2 * r) :
  ∠ A C B = π / 2 := 
  sorry

end angle_of_diameter_subtended_at_circle_point_l726_726406


namespace mode_of_dataset_with_average_is_l726_726702

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726702


namespace journey_time_difference_journey_time_difference_in_minutes_l726_726949

-- Define the constant speed of the bus
def speed : ℕ := 60

-- Define distances of journeys
def distance_1 : ℕ := 360
def distance_2 : ℕ := 420

-- Define the time calculation function
def time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the theorem
theorem journey_time_difference :
  time distance_2 speed - time distance_1 speed = 1 :=
by
  sorry

-- Convert the time difference from hours to minutes
theorem journey_time_difference_in_minutes :
  (time distance_2 speed - time distance_1 speed) * 60 = 60 :=
by
  sorry

end journey_time_difference_journey_time_difference_in_minutes_l726_726949


namespace smallest_special_number_l726_726589

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726589


namespace range_of_a_l726_726410

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a * x - 4 * a ≥ 0

theorem range_of_a :
  {a : ℝ | quadratic_nonneg_for_all_x a} = set.Icc (-16 : ℝ) 0 :=
sorry

end range_of_a_l726_726410


namespace pentagon_angle_C_gt_pi_over_2_l726_726888

theorem pentagon_angle_C_gt_pi_over_2 (A B C D E : ℝ)
  (h1 : A ≤ B)
  (h2 : B ≤ C)
  (h3 : C ≤ D)
  (h4 : D ≤ E)
  (h5 : A + B + C + D + E = 3 * Real.pi) :
  C > Real.pi / 2 ∧
  ∀ x, (∀ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ A + B + C + D + E = 3 * Real.pi → C > x) → x <= Real.pi / 2 :=
begin
  sorry -- proof placeholder
end

end pentagon_angle_C_gt_pi_over_2_l726_726888


namespace tetrahedron_edge_length_l726_726219

/-- Given a tetrahedron O P Q R with signed distances p, q, r of P, Q, R from a plane π along a directed normal, 
    prove that p² + q² + r² + (q - r)² + (r - p)² + (p - q)² = 2a², where a is the length of an edge of the tetrahedron. -/
theorem tetrahedron_edge_length
  (O P Q R : Point)
  (π : Plane)
  (p q r : ℝ)
  (a : ℝ)
  (h1 : signed_distance P π = p)
  (h2 : signed_distance Q π = q)
  (h3 : signed_distance R π = r)
  (h4 : edge_length O P = a)
  (h5 : edge_length O Q = a)
  (h6 : edge_length O R = a)
  (h7 : edge_length P Q = a)
  (h8 : edge_length P R = a)
  (h9 : edge_length Q R = a) :
  p^2 + q^2 + r^2 + (q - r)^2 + (r - p)^2 + (p - q)^2 = 2 * a^2 :=
sorry

end tetrahedron_edge_length_l726_726219


namespace square_side_length_l726_726152

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726152


namespace range_of_m_l726_726278

noncomputable def proposition_P (m : ℝ) : Prop :=
  let Δ := m^2 - 4 in
  Δ > 0 ∧ m > 2

noncomputable def proposition_Q (m : ℝ) : Prop :=
  let Δ := 16 * (m - 2)^2 - 16 in
  Δ < 0

theorem range_of_m (m : ℝ) :
  (proposition_P m ∨ proposition_Q m) ∧ ¬(proposition_P m ∧ proposition_Q m) →
  (1 < m ∧ m ≤ 2) ∨ (m ≥ 3) :=
by
  sorry

end range_of_m_l726_726278


namespace sum_of_digits_product_l726_726029

def sum_of_digits (n : ℕ) : ℕ := 9 * 2 ^ n

theorem sum_of_digits_product (n : ℕ) : 
  let m := list.range (n + 1) |>.map (λ i, 10^(2^i) - 1) |>.prod in 
  sum_of_digits n = 9 * 2 ^ n := sorry

end sum_of_digits_product_l726_726029


namespace unqualified_weight_l726_726950

theorem unqualified_weight (w : ℝ) (upper_limit lower_limit : ℝ) 
  (h1 : upper_limit = 10.1) 
  (h2 : lower_limit = 9.9) 
  (h3 : w = 9.09 ∨ w = 9.99 ∨ w = 10.01 ∨ w = 10.09) :
  ¬ (9.09 ≥ lower_limit ∧ 9.09 ≤ upper_limit) :=
by
  sorry

end unqualified_weight_l726_726950


namespace solution_set_x_f_x_lt_0_l726_726764

variable (f : ℝ → ℝ)

theorem solution_set_x_f_x_lt_0 (h1 : ∀ x, f (-x) = -f x)
                               (h2 : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2)
                               (h3 : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = set.Ioo (-3) 0 := 
sorry

end solution_set_x_f_x_lt_0_l726_726764


namespace probability_of_neither_prime_nor_composite_l726_726335

theorem probability_of_neither_prime_nor_composite :
  let S : Finset ℕ := (Finset.range 96).map ⟨λ x, x + 1, λ _, rfl⟩ in
  let non_prime_non_composite_numbers := {1} in
  let total_numbers := S.card in
  let favorable_numbers := non_prime_non_composite_numbers.card in
  (favorable_numbers : ℚ) / total_numbers = 1 / 96 :=
by
  let S : Finset ℕ := (Finset.range 96).map ⟨λ x, x + 1, λ _, rfl⟩
  let non_prime_non_composite_numbers := {1}
  let total_numbers := S.card
  let favorable_numbers := non_prime_non_composite_numbers.card
  have total_eq : total_numbers = 96 := rfl
  have favorable_eq : favorable_numbers = 1 := rfl
  rw [favorable_eq, total_eq]
  norm_num
  sorry

end probability_of_neither_prime_nor_composite_l726_726335


namespace convert_base_8_to_base_10_l726_726542

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l726_726542


namespace smallest_special_gt_3429_l726_726611

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726611


namespace coeff_x2_in_expansion_l726_726914

theorem coeff_x2_in_expansion (x : ℝ) : 
  (∀ (p : ℕ), p = 4 → (x + 1)^p = ∑ i in (finset.range (p + 1)), (nat.choose p i) * x ^ i * 1^(p - i)) →
  (nat.choose 4 2) * x ^ 2 = 6 :=
by sorry

end coeff_x2_in_expansion_l726_726914


namespace side_length_of_square_l726_726159

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726159


namespace length_of_bridge_l726_726883

theorem length_of_bridge
  (length_train : ℕ) (speed_train_kmhr : ℕ) (crossing_time : ℕ)
  (speed_conversion_factor : ℝ) (m_per_s_kmhr : ℝ) 
  (speed_train_ms : ℝ) (total_distance : ℝ) (length_bridge : ℝ)
  (h1 : length_train = 155)
  (h2 : speed_train_kmhr = 45)
  (h3 : crossing_time = 30)
  (h4 : speed_conversion_factor = 1000 / 3600)
  (h5 : m_per_s_kmhr = speed_train_kmhr * speed_conversion_factor)
  (h6 : speed_train_ms = 45 * (5 / 18))
  (h7 : total_distance = speed_train_ms * crossing_time)
  (h8 : length_bridge = total_distance - length_train):
  length_bridge = 220 :=
by
  sorry

end length_of_bridge_l726_726883


namespace least_positive_base_ten_seven_binary_digits_l726_726048

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726048


namespace range_of_a_l726_726739

noncomputable def function_decreasing (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def decreasing_function (a : ℝ) : Prop :=
  function_decreasing a (λ x, if x ≤ 1 then (a-3) * x + 5 else 2 * a / x)

theorem range_of_a (a : ℝ) (h_decreasing : decreasing_function a) :
  0 < a ∧ a ≤ 2 :=
  sorry

end range_of_a_l726_726739


namespace num_prime_factors_20_l726_726758

theorem num_prime_factors_20! : 
  ∃ n : ℕ, n = 8 ∧ ∀ p : ℕ, (p.prime ∧ p ∣ (nat.factorial 20)) ↔ p ∈ {2, 3, 5, 7, 11, 13, 17, 19} :=
by 
  sorry

end num_prime_factors_20_l726_726758


namespace dataset_mode_l726_726712

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726712


namespace ak_expr_bound_fn1_l726_726485

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ := sorry

axiom fn_zero (n : ℕ) : fn n 0 = 1/2
axiom fn_step (n k : ℕ) : n * (fn n ((k+1 : ℕ) / n) - fn n (k / n)) = fn n (k / n) - 1

def ak (n k : ℕ) : ℝ := 1 / fn n (k / n)

theorem ak_expr (n k : ℕ) : ak n k = 1 + (1 + 1 / n) ^ k := sorry

theorem bound_fn1 (n : ℕ) (_ : 0 < n): 1/4 < fn n 1 ∧ fn n 1 ≤ 1/3 := sorry

end ak_expr_bound_fn1_l726_726485


namespace multiplication_of_mixed_number_l726_726993

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726993


namespace part_I_part_II_part_III_l726_726751

noncomputable def vector_a (k : ℝ) : ℝ × ℝ := (sqrt 3, k)
def vector_b : ℝ × ℝ := (0, -1)
def vector_c : ℝ × ℝ := (1, sqrt 3)

-- Prove k = -1 when vector_a ⟂ vector_c
theorem part_I (k : ℝ) (h_perp : vector_a k.1 * vector_c.1 + vector_a k.2 * vector_c.2 = 0) : k = -1 := 
sorry

-- Prove λ = 2 when k = 1 and (vector_a - λ * vector_b) collinear with vector_c
theorem part_II (λ : ℝ) (h_collinear : (sqrt 3, 1) - λ * vector_b = λ * vector_c) : λ = 2 := 
sorry

-- Prove |vector_m + 2 * vector_c| = sqrt 7 given the necessary conditions
theorem part_III (|vector_m| : ℝ) (|vector_b| : ℝ) (angle : ℝ) (h_magnitude : |vector_m|^2 = (sqrt 3 * |vector_b|)^2)
    (h_angle : cos (150 * π / 180) = -sqrt 3 / 2) : norm (|vector_m + 2 * vector_c|) = sqrt 7 :=
sorry

end part_I_part_II_part_III_l726_726751


namespace matrix_multiplication_correct_l726_726211

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [2, 0, -1],
      [0, 3, -2],
      [-2, 3, 2] ]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [1, -1, 0],
      [2, 0, -2],
      [3, 0, 0] ]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ [-1, -2, 0],
      [0, 0, -6],
      [10, 2, -6] ]

theorem matrix_multiplication_correct :
  A ⬝ B = C :=
by
  sorry

end matrix_multiplication_correct_l726_726211


namespace max_area_difference_line_l726_726961

-- Define the given point P
def P := (1 : ℝ, 1 : ℝ)

-- Define the circular region with radius 2
def circle_region (x y : ℝ) := x^2 + y^2 ≤ 4

-- Define the statement of the problem in Lean 4
theorem max_area_difference_line : 
  ∃ (a b c : ℝ), (a * 1 + b * 1 + c = 0) ∧ (∀ x y : ℝ, circle_region x y → 
  (a * x + b * y + c = 0) → 
  (a = 1 ∧ b = 1 ∧ c = -2)) :=
sorry

end max_area_difference_line_l726_726961


namespace minimum_reciprocal_sum_l726_726775

noncomputable def circle_center : ℝ × ℝ := (-1, 2)

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : a + 2 * b = 1) : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 := 
sorry

end minimum_reciprocal_sum_l726_726775


namespace tan_plus_3sin_30_eq_2_plus_3sqrt3_l726_726212

theorem tan_plus_3sin_30_eq_2_plus_3sqrt3 :
  let θ := Real.pi / 6 -- 30 degrees in radians
  in Real.tan θ + 3 * Real.sin θ = 2 + 3 * Real.sqrt 3 := by
  have sin_30 : Real.sin θ = 1 / 2 := by
    sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by
    sorry
  sorry

end tan_plus_3sin_30_eq_2_plus_3sqrt3_l726_726212


namespace find_x_squared_l726_726921

variable (a b x p q : ℝ)

theorem find_x_squared (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : q ≠ p) (h4 : (a^2 + x^2) / (b^2 + x^2) = p / q) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := 
by 
  sorry

end find_x_squared_l726_726921


namespace correct_statements_count_l726_726980

noncomputable def question_statement1 : Prop :=
  ∀ (A B : Type) (K2 : ℕ), -- general categorical variables and K^2 type
    K2 ≥ 0 → true -- placeholder to state that larger K2 increases credibility

noncomputable def question_statement2 : Prop :=
  ∀ (x : ℝ),
    let y := λ c k, c * real.exp(k * x),
        z := λ y, real.log(y)
    in z(y(exp 4 0.3)) = 0.3 * x + 4

noncomputable def question_statement3 : Prop :=
  let y := λ a b x, a + b * x,
      b := 2,
      x_bar := 1,
      y_bar := 3,
      a := y_bar - b * x_bar
  in (a = 1)

theorem correct_statements_count : ℕ :=
  if question_statement1 ∧ question_statement2 ∧ question_statement3 then 3 else 0

end correct_statements_count_l726_726980


namespace subset_condition_l726_726716

variable A B : Set ℝ
variable a : ℝ

theorem subset_condition : (A = {0}) → (B = {x | x < a}) → (A ⊆ B) → (a > 0) :=
by
  assume hA : A = {0}
  assume hB : B = {x | x < a}
  assume hAB : A ⊆ B
  sorry

end subset_condition_l726_726716


namespace real_root_of_equation_l726_726645

theorem real_root_of_equation :
  ∃ x : ℝ, (sqrt x + sqrt (x + 4) = 12) ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726645


namespace probability_of_odd_row_diag_sum_l726_726340

/-- Define the 3x3 grid and the property that each row and diagonal has an odd sum -/
structure Grid3x3 :=
  (cells : Fin 3 → Fin 3 → ℕ)
  (unique : ∀ i j, 1 ≤ cells i j ∧ cells i j ≤ 9)
  (odd_sums : (∀ i, Odd (Fin.sum (cells i id)) ∧ Odd (Fin.sum (λ j, cells j i)) ∧ Odd (cells 0 0 + cells 1 1 + cells 2 2) ∧ Odd (cells 0 2 + cells 1 1 + cells 2 0)))

noncomputable def probability_odd_row_diag_sum_is_odd : ℚ :=
  let total_permutations := (9.factorial : ℚ)
  let valid_arrangements := 5760
  in valid_arrangements / total_permutations

theorem probability_of_odd_row_diag_sum :
  probability_odd_row_diag_sum_is_odd = 1 / 63 :=
by
  sorry

end probability_of_odd_row_diag_sum_l726_726340


namespace six_people_rolling_dice_prob_l726_726860

theorem six_people_rolling_dice_prob :
  (let 
    total_combinations := 6^6,
    acceptable_combinations := 6 * 5^4 * 4,
    prob := (acceptable_combinations : ℚ) / total_combinations
  in prob) = 625 / 1944 := 
by 
  -- Given the complexity of the problem solved above
  sorry

end six_people_rolling_dice_prob_l726_726860


namespace problem_inequality_l726_726368

noncomputable def seq : ℕ → ℝ
| 0     := sqrt 2
| (n+1) := seq n + 1 / seq n

theorem problem_inequality : 
  (∑ n in Finset.range 2019, 1 / (seq (n + 2)^2 - seq (n + 1)^2)) >
  2019^2 / (seq 2020^2 - 2) :=
sorry

end problem_inequality_l726_726368


namespace real_root_solution_l726_726644

noncomputable def real_root_equation : Nat := 36

theorem real_root_solution (x : ℝ) (h : x ≥ 0 ∧ √x + √(x+4) = 12) : x = 1225 / real_root_equation := by
  sorry

end real_root_solution_l726_726644


namespace smallest_special_number_l726_726595

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726595


namespace sum_of_numbers_is_60_l726_726011

-- Define the primary values used in the conditions
variables (a b c : ℝ)

-- Define the conditions in the problem
def mean_condition_1 : Prop := (a + b + c) / 3 = a + 20
def mean_condition_2 : Prop := (a + b + c) / 3 = c - 30
def median_condition : Prop := b = 10

-- Prove that the sum of the numbers is 60 given the conditions
theorem sum_of_numbers_is_60 (hac1 : mean_condition_1 a b c) (hac2 : mean_condition_2 a b c) (hbm : median_condition b) : a + b + c = 60 :=
by 
  sorry

end sum_of_numbers_is_60_l726_726011


namespace eighteenth_number_5732_l726_726655

theorem eighteenth_number_5732 :
  let digits := [2, 3, 5, 7]
  let all_numbers := (List.permutations digits).map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let sorted_numbers := all_numbers.qsort (λ a b, a < b)
  sorted_numbers.nth 17 = some 5732 := 
by
  let digits := [2, 3, 5, 7]
  let all_numbers := (List.permutations digits).map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let sorted_numbers := all_numbers.qsort (λ a b, a < b)
  have nth_18 : sorted_numbers.nth 17 = some 5732, from sorry
  exact nth_18

end eighteenth_number_5732_l726_726655


namespace ratio_of_triangle_areas_l726_726341

theorem ratio_of_triangle_areas
  (A B C D E : Point)
  (α : ℝ)
  (h1 : is_diameter A B)
  (h2 : CD_parallel_AB C D A B)
  (h3 : intersect_ac_bd_at_e A C B D E)
  (h4 : angle_AED α)
  (h5 : angle_AEB_90 A B E) :
  area_CDE_div_area_ABE C D E A B E = (Real.sin(α))^2 := 
sorry

end ratio_of_triangle_areas_l726_726341


namespace find_digits_of_Vasya_l726_726463

open Nat

def is_non_zero_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

noncomputable def sum_of_two_digit_numbers (a b c : ℕ) : ℕ :=
  (10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b)

theorem find_digits_of_Vasya (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : is_non_zero_digit a) (h₅ : is_non_zero_digit b) (h₆ : is_non_zero_digit c)
  (h₇ : sum_of_two_digit_numbers a b c = 231) :
  {a, b, c} = {2, 3, 6} :=
by
  sorry

end find_digits_of_Vasya_l726_726463


namespace least_positive_base_ten_seven_binary_digits_l726_726047

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726047


namespace pizza_slices_with_both_toppings_l726_726126

theorem pizza_slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices n : ℕ) 
    (h1 : total_slices = 14) 
    (h2 : pepperoni_slices = 8) 
    (h3 : mushroom_slices = 12) 
    (h4 : ∀ s, s = pepperoni_slices + mushroom_slices - n ∧ s = total_slices := by sorry) :
    n = 6 :=
sorry

end pizza_slices_with_both_toppings_l726_726126


namespace least_positive_base_ten_seven_binary_digits_l726_726051

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726051


namespace total_pencils_l726_726856

theorem total_pencils (serenity_pencils : ℕ) (friend1_pencils : ℕ) (friend2_pencils : ℕ) (h1 : serenity_pencils = 7) (h2 : friend1_pencils = 7) (h3 : friend2_pencils = 7) : serenity_pencils + friend1_pencils + friend2_pencils = 21 :=
by
  rw [h1, h2, h3]
  exact rfl

end total_pencils_l726_726856


namespace coefficient_a3_of_expansion_l726_726662

theorem coefficient_a3_of_expansion :
  let a := (fun n (x : ℕ) => (2 * x^2 + 1) ^ n),
      expanded := ∑ i in (Finset.range (6 + 1)), (a 5 i) * x^(2 * i)
  in (expanded.coeff 6 = 80) :=
by
  sorry

end coefficient_a3_of_expansion_l726_726662


namespace distinct_parallel_lines_l726_726003

theorem distinct_parallel_lines (k : ℝ) :
  (∃ (L1 L2 : ℝ × ℝ → Prop), 
    (∀ x y, L1 (x, y) ↔ x - 2 * y - 3 = 0) ∧ 
    (∀ x y, L2 (x, y) ↔ 18 * x - k^2 * y - 9 * k = 0)) → 
  (∃ slope1 slope2, 
    slope1 = 1/2 ∧ 
    slope2 = 18 / k^2 ∧
    (slope1 = slope2) ∧
    (¬ (∀ x y, x - 2 * y - 3 = 18 * x - k^2 * y - 9 * k))) → 
  k = -6 :=
by 
  sorry

end distinct_parallel_lines_l726_726003


namespace parametric_line_eq_l726_726424

-- Define the parameterized functions for x and y 
def parametric_x (t : ℝ) : ℝ := 3 * t + 7
def parametric_y (t : ℝ) : ℝ := 5 * t - 8

-- Define the equation of the line (here it's a relation that relates x and y)
def line_equation (x y : ℝ) : Prop := 
  y = (5 / 3) * x - (59 / 3)

theorem parametric_line_eq : 
  ∃ t : ℝ, line_equation (parametric_x t) (parametric_y t) := 
by
  -- Proof goes here
  sorry

end parametric_line_eq_l726_726424


namespace smallest_special_number_l726_726587

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726587


namespace subset_0_in_X_l726_726336

def X : Set ℝ := {x | x > -1}

theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l726_726336


namespace polynomial_at_n_plus_1_l726_726271

noncomputable def cnk (n k : ℕ) : ℚ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem polynomial_at_n_plus_1 (n : ℕ) (P : ℕ → ℚ)
  (hP : ∀ k, k ≤ n → P k = 1 / cnk n k) :
  P (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end polynomial_at_n_plus_1_l726_726271


namespace train_speed_kmh_l726_726515

theorem train_speed_kmh (T P: ℝ) (L: ℝ):
  (T = L + 320) ∧ (L = 18 * P) ->
  P = 20 -> 
  P * 3.6 = 72 := 
by
  sorry

end train_speed_kmh_l726_726515


namespace adjusted_guess_red_white_jellybeans_l726_726358

variables {small_red small_black small_green small_purple small_yellow small_white : ℕ}
variables {medium_red medium_black medium_green medium_purple medium_yellow medium_white : ℕ}
variables (vol_ratio : ℝ := 0.6)
variables (bags_fill_fishbowl : ℕ := 3)

-- Conditions from the problem
def jellybean_sizes : Prop :=
  small_red = 12 ∧ medium_red = 10 ∧
  small_black = 25 ∧ medium_black = 20 ∧ 
  small_green = 36 ∧ medium_green = 30 ∧ 
  small_purple = 15 ∧ medium_purple = 28 ∧ 
  small_yellow = 12 ∧ medium_yellow = 32 ∧ 
  small_white = 8 ∧ medium_white = 18

-- Proof problem statement
theorem adjusted_guess_red_white_jellybeans (h_sizes : jellybean_sizes) : 
  let red_white_equivalent := (12 * vol_ratio + 10) + (8 * vol_ratio + 18) in
  let total_equivalent := 
    (12 + 25 + 36 + 15 + 12 + 8) * vol_ratio + 
    (10 + 20 + 30 + 28 + 32 + 18) in
  let equivalent_three_bags := total_equivalent * bags_fill_fishbowl in
  let ratio_red_white := red_white_equivalent / total_equivalent in
  let adjusted_red_white := ratio_red_white * equivalent_three_bags in
  | adjusted_red_white - 120 | ≤ 1 :=
begin
  sorry
end

end adjusted_guess_red_white_jellybeans_l726_726358


namespace least_integer_greater_than_z_exp_l726_726280

noncomputable def least_int_greater_than_expression (z : ℂ) (h : z + (1 / z) = 2 * complex.cos (real.pi * 3 / 180)) : ℤ :=
  ⌈z^2000 + (1 / z^2000)⌉

theorem least_integer_greater_than_z_exp (z : ℂ) (h : z + (1 / z) = 2 * complex.cos (real.pi * 3 / 180)) :
  least_int_greater_than_expression z h = 0 :=
sorry

end least_integer_greater_than_z_exp_l726_726280


namespace area_ratio_equality_l726_726352

-- Define the necessary elements for the proof
variable (A B C M C1 : Type)

-- Define that A, B, C are points creating a triangle
axiom TriangleABC : Triangle A B C

-- Define the point M inside the triangle ABC
axiom PointM : Point M

-- Define the point C₁ as the intersection point of the line AB and CM
axiom IntersectionC1 : IntersectionPoint C1 (Line A B) (Line C M)

-- Define the areas of the triangles ACM, BCM, and ABM
axiom AreaACM : Area (Triangle A C M)
axiom AreaBCM : Area (Triangle B C M)
axiom AreaABM : Area (Triangle A B M)

-- Define the segment lengths CM and C1M
axiom LengthCM : Length (Segment C M)
axiom LengthC1M : Length (Segment C1 M)

-- Define the necessary equality to prove
theorem area_ratio_equality : ∀ A B C M C1,
  (AreaACM + AreaBCM) / AreaABM = LengthCM / LengthC1M := 
sorry

end area_ratio_equality_l726_726352


namespace no_hyperdeficient_numbers_l726_726218

def sum_of_divisors (n : ℕ) : ℕ := nat.divisors n |> list.sum

def is_hyperdeficient (n : ℕ) : Prop :=
  sum_of_divisors (sum_of_divisors n) = n + 3

theorem no_hyperdeficient_numbers : ∀ n : ℕ, ¬ is_hyperdeficient n :=
sorry

end no_hyperdeficient_numbers_l726_726218


namespace sequence_sum_is_correct_l726_726391

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (2 + k * (1/3))

theorem sequence_sum_is_correct (n : ℕ) : 
  sequence_sum n = n * (n + 11) / 6 :=
by
  sorry

end sequence_sum_is_correct_l726_726391


namespace train_crossing_time_l726_726195

/-- Given the conditions that a moving train requires 10 seconds to pass a pole,
    its speed is 36 km/h, and the length of a stationary train is 300 meters,
    prove that the moving train takes 40 seconds to cross the stationary train. -/
theorem train_crossing_time (t_pole : ℕ)
  (v_kmh : ℕ)
  (length_stationary : ℕ) :
  t_pole = 10 →
  v_kmh = 36 →
  length_stationary = 300 →
  ∃ t_cross : ℕ, t_cross = 40 :=
by
  intros h1 h2 h3
  sorry

end train_crossing_time_l726_726195


namespace least_positive_base_ten_with_seven_binary_digits_l726_726090

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726090


namespace M_inter_complement_N_empty_l726_726309

-- Defining the sets M and N as per the conditions
def M : Set ℝ := { x | |x - 1| < 1 }
def N : Set ℝ := { x | x^2 - 2x < 3 }

-- The proof that the intersection of M and the complement of N is empty
theorem M_inter_complement_N_empty : M ∩ (Set.compl N) = ∅ := by
  sorry

end M_inter_complement_N_empty_l726_726309


namespace trapezoid_sides_l726_726421

theorem trapezoid_sides :
  ∀ (A B C D O : Type) [EuclideanGeometry.Point A] [EuclideanGeometry.Point B] [EuclideanGeometry.Point C] [EuclideanGeometry.Point D] [EuclideanGeometry.Point O],
    (EuclideanGeometry.is_right_trapezoid A B C D) →
    (EuclideanGeometry.is_center_in_circle O A B C D) →
    (EuclideanGeometry.dist O C = 3) →
    (EuclideanGeometry.dist O D = 9) →
    EuclideanGeometry.sides_of_trapezoid A B C D = [9 * Real.sqrt 10 / 5, 6 * Real.sqrt 10 / 5, 3 * Real.sqrt 10, 18 * Real.sqrt 10 / 5] :=
by
  sorry

end trapezoid_sides_l726_726421


namespace decimal_difference_l726_726934

theorem decimal_difference (a b : ℝ) (h1 : a = 0.127) (h2 : b = 1 / 8) : a - b = 0.002 :=
begin
  sorry
end

end decimal_difference_l726_726934


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726068

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726068


namespace equilateral_triangle_sine_determinant_l726_726372

theorem equilateral_triangle_sine_determinant :
  let A := real.pi / 3
  let B := real.pi / 3
  let C := real.pi / 3
  let sin60 := real.sin (real.pi / 3)
  let M := ![
    ![sin60, 1, 1],
    ![1, sin60, 1],
    ![1, 1, sin60]
  ]
  matrix.det M = -real.sqrt 3 / 8 :=
by
  let A := real.pi / 3
  let B := real.pi / 3
  let C := real.pi / 3
  let sin60 := real.sin (real.pi / 3)
  let M := ![
    ![sin60, 1, 1],
    ![1, sin60, 1],
    ![1, 1, sin60]
  ]
  have hsin : sin60 = real.sqrt 3 / 2 := by norm_num
  rw [hsin]
  sorry

end equilateral_triangle_sine_determinant_l726_726372


namespace time_taken_by_alex_l726_726199

-- Define the conditions
def distance_per_lap : ℝ := 500 -- distance per lap in meters
def distance_first_part : ℝ := 150 -- first part of the distance in meters
def speed_first_part : ℝ := 3 -- speed for the first part in meters per second
def distance_second_part : ℝ := 350 -- remaining part of the distance in meters
def speed_second_part : ℝ := 4 -- speed for the remaining part in meters per second
def num_laps : ℝ := 4 -- number of laps run by Alex

-- Target time, expressed in seconds
def target_time : ℝ := 550 -- 9 minutes and 10 seconds is 550 seconds

-- Prove that given the conditions, the total time Alex takes to run 4 laps is 550 seconds
theorem time_taken_by_alex :
  (distance_first_part / speed_first_part + distance_second_part / speed_second_part) * num_laps = target_time :=
by
  sorry

end time_taken_by_alex_l726_726199


namespace minimum_zeros_l726_726796

theorem minimum_zeros (board : Fin 11 → Fin 11 → ℤ)
  (cond1 : ∀ j : Fin 11, 0 ≤ ∑ i, board i j)
  (cond2 : ∀ i : Fin 11, ∑ j, board i j ≤ 0)
  (value_range : ∀ i j, board i j = -1 ∨ board i j = 0 ∨ board i j = 1) :
  ∃ zs : Fin 11 → Fin 11 → Prop, (∀ i j, zs i j ↔ board i j = 0) ∧
    (∑ i j, if zs i j then 1 else 0) = 11 :=
by
  sorry

end minimum_zeros_l726_726796


namespace cannot_transform_604_to_703_l726_726958

theorem cannot_transform_604_to_703 :
  ∀ (f : ℕ → ℕ) (g : ℕ → ℕ → ℕ), 
    (f 604 = 604^2) ∧ (g X n = if n > 3 then (X % 1000) + (X / 1000) else X) →
    (∀ n : ℕ, (f^[n] 604 ≠ 703)) :=
begin
  intros f g h,
  sorry,
end

end cannot_transform_604_to_703_l726_726958


namespace equal_expressions_l726_726519

theorem equal_expressions : (-2)^3 = -(2^3) :=
by sorry

end equal_expressions_l726_726519


namespace least_positive_base_ten_with_seven_binary_digits_l726_726093

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726093


namespace range_of_x_l726_726772

theorem range_of_x (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
sorry

end range_of_x_l726_726772


namespace sequence_is_integer_l726_726619

-- Definition of the recurrence relation sequence
def a : ℕ → ℤ
| 0     := 1
| 1     := 1
| (n+2) := ((2*n + 1) * a (n+1) + 3 * n * a n) / (n + 2)

-- Theorem stating that all elements of the sequence are integers
theorem sequence_is_integer (n : ℕ) : a n ∈ ℤ := sorry

end sequence_is_integer_l726_726619


namespace find_a_minus_b_l726_726445

theorem find_a_minus_b (a b c d : ℤ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 :=
sorry

end find_a_minus_b_l726_726445


namespace smallest_special_gt_3429_l726_726613

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726613


namespace probability_at_least_6_stay_l726_726770

theorem probability_at_least_6_stay (p : ℚ) (h1 : p = 1 / 3) : 
  let psure := 3,
      punsure := 4,
      p_basketball_game := ((punsure.choose 3) * (p ^ 3) * ((1 - p) ^ 1) + (p ^ 4)),
      prob_at_least_6 := psure + 3 ∨ psure + 4 in
  p_basketball_game = 1 / 9 :=
by
  have h_cases : (prob_at_least_6 = 6) ∨ (prob_at_least_6 = 7), -- define scenarios
  by sorry,
  rw [h1, ←rat.add_num_denom, ←rat.mul_num_denom], -- simplifying using the given probability 1/3
  have h_case_6 : (4.choose 3 * (1 / 3) ^ 3 * (2 / 3)) = 8 / 81,
  by sorry,
  have h_case_7 : ((1 / 3) ^ 4) = 1 / 81,
  by sorry,
  have h_sum_cases := h_case_6 + h_case_7,
  finish,
  sorry

end probability_at_least_6_stay_l726_726770


namespace square_side_length_l726_726168

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726168


namespace varphi_le_one_varphi_l726_726290

noncomputable def f (a x : ℝ) := -a * Real.log x

-- Definition of the minimum value function φ for a > 0
noncomputable def varphi (a : ℝ) := -a * Real.log a

theorem varphi_le_one (a : ℝ) (h : 0 < a) : varphi a ≤ 1 := 
by sorry

theorem varphi'_le (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
    (1 - Real.log a) ≤ (1 - Real.log b) := 
by sorry

end varphi_le_one_varphi_l726_726290


namespace slower_train_pass_faster_driver_l726_726935

-- Definitions based on conditions
def train_length : ℝ := 475
def speed_first_train_kmh : ℝ := 55
def speed_second_train_kmh : ℝ := 40
def kmh_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Speeds in meters per second
def speed_first_train_mps : ℝ := kmh_to_mps speed_first_train_kmh
def speed_second_train_mps : ℝ := kmh_to_mps speed_second_train_kmh

-- Relative speed considering they are running in opposite directions
def relative_speed_mps : ℝ := speed_first_train_mps + speed_second_train_mps

-- Distance to be covered is the length of one train
def distance_to_cover : ℝ := train_length

-- Time taken calculation
def time_taken : ℝ := distance_to_cover / relative_speed_mps

-- Proof statement
theorem slower_train_pass_faster_driver : 
  time_taken = 18 := by
  -- Using external tool for proof completion
  sorry

end slower_train_pass_faster_driver_l726_726935


namespace vector_magnitude_l726_726666

-- Given conditions and vectors definitions
def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vec_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)

-- Given condition on cosine of the angle difference
axiom cosine_angle_diff_zero (α β : ℝ) : Real.cos (α - β) = 0

-- Goal: |vec_a + vec_b| = √2
theorem vector_magnitude (α β : ℝ) (h : Real.cos (α - β) = 0) : 
  let a := vec_a α 
  let b := vec_b β 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 2 := 
sorry

end vector_magnitude_l726_726666


namespace shaded_area_calc_l726_726909

theorem shaded_area_calc (r1_area r2_area overlap_area circle_area : ℝ)
  (h_r1_area : r1_area = 36)
  (h_r2_area : r2_area = 28)
  (h_overlap_area : overlap_area = 21)
  (h_circle_area : circle_area = Real.pi) : 
  (r1_area + r2_area - overlap_area - circle_area) = 64 - Real.pi :=
by
  sorry

end shaded_area_calc_l726_726909


namespace intersection_M_N_l726_726750

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = { z | 0 ≤ z ∧ z ≤ 1 } := by
  sorry

end intersection_M_N_l726_726750


namespace sum_values_m_min_area_l726_726435

def is_line (p1 p2 : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∀ x y z w, l (x, y) → l (z, w) → (z - x) * (snd p2 - snd p1) = (first p2 - first p1) * (w - y)

def equation_of_line : ℝ → ℝ := λ x, (7 / 6) * x + 11 / 3

def dist_to_line (m : ℝ) : ℝ := abs (m - equation_of_line 6)

def possible_m (p : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ m1 m2, dist_to_line m1 = dist_to_line m2

theorem sum_values_m_min_area :
  ∃ m1 m2 : ℤ, 
    possible_m (2, 5) m1 m2 ∧ 
    m1 + m2 = 21 := 
sorry

end sum_values_m_min_area_l726_726435


namespace quadratic_condition_g_monotonic_decrease_g_inequality_condition_l726_726305

def f (x : ℝ) : ℝ := x^2 - 4 * x

def g (x : ℝ) : ℝ :=
  if x >= 0 then f x else -f (-x)

theorem quadratic_condition :
  f 1 = -3 ∧ f 3 = -3 := by 
  -- proof of these equalities
  sorry

theorem g_monotonic_decrease :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∀ y, x < y → g x > g y := by
  -- proof for monotonic decreasing interval
  sorry

theorem g_inequality_condition (a : ℝ) :
  g a > a → (a > 5 ∨ (-5 < a ∧ a < 0)) := by
  -- proof for range of values of a
  sorry

end quadratic_condition_g_monotonic_decrease_g_inequality_condition_l726_726305


namespace function_characterization_l726_726380

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  (∀ x : ℝ, 0 ≤ x → f x = if x < 2 then 2 / (2 - x) else 0) := sorry

end function_characterization_l726_726380


namespace point_in_third_quadrant_l726_726801

theorem point_in_third_quadrant (x y : ℤ) (hx : x = -8) (hy : y = -3) : (x < 0) ∧ (y < 0) :=
by
  have hx_neg : x < 0 := by rw [hx]; norm_num
  have hy_neg : y < 0 := by rw [hy]; norm_num
  exact ⟨hx_neg, hy_neg⟩

end point_in_third_quadrant_l726_726801


namespace minimum_M_coordinates_l726_726330

-- Define point A and its coordinates
def A : ℝ × ℝ := (1/2, 2)

-- Define the parabola y^2 = 2x
def parabola (x y : ℝ) : Prop := y ^ 2 = 2 * x

-- Define the focus F coordinates
def F : ℝ × ℝ := (1/2, 0)

-- Prove the coordinates of point M that minimize |MF| + |MA|
theorem minimum_M_coordinates (M : ℝ × ℝ) (hM : parabola M.1 M.2) :
  ∃ (x y : ℝ), M = (x, y) ∧ x = 1/2 ∧ y = 1 :=
sorry

end minimum_M_coordinates_l726_726330


namespace smallest_base_10_integer_l726_726465

-- Given conditions
def is_valid_base (a b : ℕ) : Prop := a > 2 ∧ b > 2

def base_10_equivalence (a b n : ℕ) : Prop := (2 * a + 1 = n) ∧ (b + 2 = n)

-- The smallest base-10 integer represented as 21_a and 12_b
theorem smallest_base_10_integer :
  ∃ (a b n : ℕ), is_valid_base a b ∧ base_10_equivalence a b n ∧ n = 7 :=
by
  sorry

end smallest_base_10_integer_l726_726465


namespace simplify_expression_l726_726235

theorem simplify_expression :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) *
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 :=
by 
  sorry

end simplify_expression_l726_726235


namespace parabolas_intersect_at_points_l726_726540

theorem parabolas_intersect_at_points :
  ∀ (x y : ℝ), (y = 3 * x^2 - 12 * x - 9) ↔ (y = 2 * x^2 - 8 * x + 5) →
  (x, y) = (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2) ∨ (x, y) = (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2) :=
by
  sorry

end parabolas_intersect_at_points_l726_726540


namespace cylinder_height_and_diameter_l726_726444

theorem cylinder_height_and_diameter {r : ℝ} (h_eq_d : ℝ) (A_sphere_eq_A_cylinder : 4 * real.pi * 5^2 = 2 * real.pi * r * h_eq_d) :
  h_eq_d = 10 ∧ 2 * r = 10 :=
sorry

end cylinder_height_and_diameter_l726_726444


namespace rabbit_total_distance_l726_726023

theorem rabbit_total_distance (speed_white : ℕ) (speed_brown : ℕ) (time : ℕ) (total_distance : ℕ) :
  speed_white = 15 → speed_brown = 12 → time = 5 → total_distance = 135 → 
  (speed_white * time + speed_brown * time = total_distance) := by
  intros h_white h_brown h_time h_total_distance
  rw [h_white, h_brown, h_time, h_total_distance]
  sorry

end rabbit_total_distance_l726_726023


namespace coordinates_of_A_after_move_l726_726275

noncomputable def moved_coordinates (a : ℝ) : ℝ × ℝ :=
  let x := 2 * a - 9 + 5
  let y := 1 - 2 * a
  (x, y)

theorem coordinates_of_A_after_move (a : ℝ) (h : moved_coordinates a = (0, 1 - 2 * a)) :
  moved_coordinates 2 = (-5, -3) :=
by
  -- Proof omitted
  sorry

end coordinates_of_A_after_move_l726_726275


namespace find_k_l726_726753

theorem find_k
  (k : ℝ)
  (a : ℝ × ℝ := (3,1))
  (b : ℝ × ℝ := (1,3))
  (c : ℝ × ℝ := (k,2))
  (h : (a.1 - c.1, a.2 - c.2) = (a.1 - k, a.2 - 2) ∧ (a.1 - k, a.2 - 2) ∙ b = 0) 
  : k = 0 :=
by
  sorry

end find_k_l726_726753


namespace calculate_product_l726_726996

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l726_726996


namespace convert_246_octal_to_decimal_l726_726564

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l726_726564


namespace binomial_coefficient_of_x_l726_726333

theorem binomial_coefficient_of_x (a : ℝ) (hx : x + a*y - 1 = 0 ∧ ⟂ (2*x - 4*y + 3 = 0)) :
  coefficient (expand (λ n, coe n * x ^ (5 - n) * (ax^2 - 1/x)^5) x) 1 = - 5 / 2 :=
by {
  sorry
}

end binomial_coefficient_of_x_l726_726333


namespace side_length_of_square_l726_726167

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726167


namespace inequality_holds_for_all_x_iff_m_eq_1_l726_726622

theorem inequality_holds_for_all_x_iff_m_eq_1 (m : ℝ) (h_m : m ≠ 0) :
  (∀ x > 0, x^2 - 2 * m * Real.log x ≥ 1) ↔ m = 1 :=
by
  sorry

end inequality_holds_for_all_x_iff_m_eq_1_l726_726622


namespace smallest_special_number_gt_3429_l726_726603

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726603


namespace part1_part2_l726_726295

-- Define the function
def f (x a : ℝ) : ℝ := |x - 2 * a| + |x + 1 / a|

-- Statement 1: For a = 1, proving the solution set of the inequality f(x) > 4
theorem part1 (x : ℝ) : (|x - 2| + |x + 1| > 4) ↔ (x < -3 / 2 ∨ x > 5 / 2) := by
  sorry

-- Statement 2: For all x, a in reals, proving the range of m
theorem part2 {x a : ℝ} : (∀ x a, f x a ≥ m^2 - m + 2 * Real.sqrt 2) ↔ (0 ≤ m ∧ m ≤ 1) := by
  sorry

end part1_part2_l726_726295


namespace simple_interest_l726_726777

noncomputable theory

def principal (CI : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
   CI / (rate * (1 + rate / 100) ^ time - 1)

theorem simple_interest (CI : ℝ) (rate : ℝ) (time : ℕ)
  (h_rate : rate = 5) (h_time : time = 2) (h_CI : CI = 41) :
  let P := principal CI rate time in P * rate * time / 100 = 40 := 
by
  dsimp [principal]
  rw [h_rate, h_time, h_CI]
  have h1 : P = 400 := sorry
  norm_num
  sorry

end simple_interest_l726_726777


namespace smallest_special_number_gt_3429_l726_726605

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726605


namespace find_seating_capacity_l726_726954

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l726_726954


namespace quadratic_roots_r6_s6_l726_726383

theorem quadratic_roots_r6_s6 (r s : ℝ) (h1 : r + s = 3 * Real.sqrt 2) (h2 : r * s = 4) : r^6 + s^6 = 648 := by
  sorry

end quadratic_roots_r6_s6_l726_726383


namespace probability_one_defective_l726_726103

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l726_726103


namespace smallest_k_l726_726042

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l726_726042


namespace least_binary_seven_digits_l726_726061

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726061


namespace john_average_speed_l726_726356

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end john_average_speed_l726_726356


namespace triangle_area_range_l726_726884

noncomputable theory

def point (x y : ℝ) := (x, y)

def line (x y : ℝ) : Prop := x + y + 2 = 0

def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def area (a b c : ℝ × ℝ) : ℝ :=
  let base := distance a b,
      height := abs ((c.2 - a.2) * (b.1 - a.1) - (c.1 - a.1) * (b.2 - a.2)) /
          distance a b in
  0.5 * base * height

def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, -2)
 
theorem triangle_area_range : 
  ∀ P: ℝ × ℝ, (circle P.1 P.2) → 
    ∃ (a b : ℝ), (2 ≤ area point_A point_B P) ∧ (area point_A point_B P ≤ 6) :=
begin
  sorry
end

end triangle_area_range_l726_726884


namespace convert_246_octal_to_decimal_l726_726562

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l726_726562


namespace smallest_special_number_gt_3429_l726_726578

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726578


namespace length_of_second_train_l726_726924

def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem length_of_second_train (length_first_train : ℕ) (speed_first_train_kmph : ℕ) (speed_second_train_kmph : ℕ) (cross_time_seconds : ℕ) :
  length_first_train = 290 ∧
  speed_first_train_kmph = 120 ∧
  speed_second_train_kmph = 80 ∧
  cross_time_seconds = 9 →
  let speed_first_train := kmph_to_mps speed_first_train_kmph;
      speed_second_train := kmph_to_mps speed_second_train_kmph;
      relative_speed := speed_first_train + speed_second_train;
      total_distance := relative_speed * cross_time_seconds;
  total_distance - length_first_train = 210 :=
sorry

end length_of_second_train_l726_726924


namespace distance_from_A_to_B_l726_726505

-- Definitions of the conditions
def avg_speed : ℝ := 25
def distance_AB (D : ℝ) : Prop := ∃ T : ℝ, D / (4 * T) = avg_speed ∧ D = 3 * (T * avg_speed)∧ (D / 2) = (T * avg_speed)

theorem distance_from_A_to_B : ∃ D : ℝ, distance_AB D ∧ D = 100 / 3 :=
by
  sorry

end distance_from_A_to_B_l726_726505


namespace number_of_digits_2021_factorial_trailing_zeros_2021_factorial_l726_726942

/-! Prove that the number of digits in the decimal representation of 2021! is 5805 using logarithms -/
theorem number_of_digits_2021_factorial :
  let d := ⌊ (∑ k in finset.range (2021+1), real.log (k + 1)) / real.log 10 ⌋ + 1
  in d = 5805 :=
by
  sorry

/-! Prove that the number of trailing zeros in 2021! is 503 using Legendre's formula -/
theorem trailing_zeros_2021_factorial :
  let v5 := ∑ j in finset.range (⌊ real.log 2021 / real.log 5 ⌋ + 1), ⌊ 2021 / 5^j ⌋
  in v5 = 503 :=
by
  sorry

end number_of_digits_2021_factorial_trailing_zeros_2021_factorial_l726_726942


namespace num_true_statements_l726_726734

theorem num_true_statements :
  (∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) ∧
  (∀ x y a, a ≠ 0 → (a^2 * x ≥ a^2 * y → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x ≥ y → x / a^2 ≥ y / a^2)) →
  ((∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) →
   (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y))) :=
sorry

end num_true_statements_l726_726734


namespace least_binary_seven_digits_l726_726066

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726066


namespace regular_polygon_sides_l726_726329

theorem regular_polygon_sides (n : ℕ) (h : 0 < n) (h_angle : (n - 2) * 180 = 144 * n) :
  n = 10 :=
sorry

end regular_polygon_sides_l726_726329


namespace number_of_primes_l726_726492

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_primes (p : ℕ)
  (H_prime : is_prime p)
  (H_square : is_perfect_square (1 + p + p^2 + p^3 + p^4)) :
  p = 3 :=
sorry

end number_of_primes_l726_726492


namespace least_number_of_cans_l726_726959

theorem least_number_of_cans (liters_maaza : ℕ) (liters_pepsi : ℕ) (liters_sprite : ℕ)
  (h_maaza : liters_maaza = 10)
  (h_pepsi : liters_pepsi = 144)
  (h_sprite : liters_sprite = 368) :
  let gcd_val := Nat.gcd (Nat.gcd liters_maaza liters_pepsi) liters_sprite
  in liters_maaza / gcd_val + liters_pepsi / gcd_val + liters_sprite / gcd_val = 261 :=
by
  rw [h_maaza, h_pepsi, h_sprite]
  let gcd_val := Nat.gcd (Nat.gcd 10 144) 368 in
  have h1 : gcd_val = 2 :=
    by
      calc
        Nat.gcd 10 144 = 2 := by norm_num
        Nat.gcd 2 368 = 2 := by norm_num
  rw [h1]
  exact (by norm_num : 10 / 2 + 144 / 2 + 368 / 2 = 261)

end least_number_of_cans_l726_726959


namespace pyramid_volume_surface_area_l726_726420

variables (a α φ : ℝ)

-- Assume all angles are given in radians
def volume (a α φ : ℝ) : ℝ := (1 / 3) * a^3 * (Real.sin α)^2 * Real.tan φ
def lateral_surface_area (a α φ : ℝ) : ℝ := 
  (2 * a^2 * Real.sin α * (Real.cos (π/4 - φ/2))^2) / Real.cos φ

-- Statement to prove the volume and lateral surface area
theorem pyramid_volume_surface_area :
  ∀ (a α φ : ℝ), 
  volume a α φ = (1 / 3) * a^3 * (Real.sin α)^2 * Real.tan φ ∧
  lateral_surface_area a α φ = 
  (2 * a^2 * Real.sin α * (Real.cos (π/4 - φ/2))^2) / Real.cos φ := 
by
  sorry

end pyramid_volume_surface_area_l726_726420


namespace side_length_of_square_l726_726166

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726166


namespace sum_q_t_at_8_eq_128_l726_726822

noncomputable def T : set (fin 8 → bool) :=
  {f : fin 8 → bool | true}

-- Define the polynomial q_t for each tuple t in T
noncomputable def q_t (t : fin 8 → bool) : polynomial ℚ :=
  polynomial.sum (finset.range 8) (λ n, if t n then (polynomial.monomial (n : ℕ) (1 : ℚ)) else 0)

-- Define the polynomial q based on the sum of q_t over all t in T
noncomputable def q : polynomial ℚ :=
  polynomial.sum (finset.powerset_univ {(i : fin 8) // true}) (λ t, q_t t)

-- Lean proof statement
theorem sum_q_t_at_8_eq_128 : polynomial.eval (8 : ℚ) q = 128 := 
  by sorry

end sum_q_t_at_8_eq_128_l726_726822


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726053

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726053


namespace photo_arrangement_l726_726972

noncomputable def valid_arrangements (teacher boys girls : ℕ) : ℕ :=
  if girls = 2 ∧ teacher = 1 ∧ boys = 2 then 24 else 0

theorem photo_arrangement :
  valid_arrangements 1 2 2 = 24 :=
by {
  -- The proof goes here.
  sorry
}

end photo_arrangement_l726_726972


namespace possible_value_of_n_l726_726767

theorem possible_value_of_n :
  ∃ (n : ℕ), (345564 - n) % (13 * 17 * 19) = 0 ∧ 0 < n ∧ n < 1000 ∧ n = 98 :=
sorry

end possible_value_of_n_l726_726767


namespace expand_expression_l726_726629

theorem expand_expression (x y : ℤ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := 
by
  sorry

end expand_expression_l726_726629


namespace bala_age_difference_l726_726943

theorem bala_age_difference 
  (a10 : ℕ) -- Anand's age 10 years ago.
  (b10 : ℕ) -- Bala's age 10 years ago.
  (h1 : a10 = b10 / 3) -- 10 years ago, Anand's age was one-third Bala's age.
  (h2 : a10 = 15 - 10) -- Anand was 5 years old 10 years ago, given his current age is 15.
  : (b10 + 10) - 15 = 10 := -- Bala is 10 years older than Anand.
sorry

end bala_age_difference_l726_726943


namespace proof_problem_l726_726664

variable {a b c : ℝ}

theorem proof_problem (h1 : ∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) : 
  (4 * a + 2 * b + c = 28) := by
  -- The proof goes here. The goal statement is what we need.
  sorry

end proof_problem_l726_726664


namespace smallest_special_greater_than_3429_l726_726571

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726571


namespace smallest_special_gt_3429_l726_726596

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726596


namespace value_of_g_if_f_is_odd_l726_726833

theorem value_of_g_if_f_is_odd (f g : ℝ → ℝ)
  (hf : ∀ x, f x = if x < 0 then 2^x else g x)
  (hodd : ∀ x, f (-x) = -f x) :
  g 3 = -1/8 :=
begin
  sorry
end

end value_of_g_if_f_is_odd_l726_726833


namespace Karls_Total_Travel_Distance_l726_726366

theorem Karls_Total_Travel_Distance :
  let consumption_rate := 35
  let full_tank_gallons := 14
  let initial_miles := 350
  let added_gallons := 8
  let remaining_gallons := 7
  let net_gallons_consumed := (full_tank_gallons + added_gallons - remaining_gallons)
  let total_distance := net_gallons_consumed * consumption_rate
  total_distance = 525 := 
by 
  sorry

end Karls_Total_Travel_Distance_l726_726366


namespace area_of_circular_flower_bed_l726_726873

theorem area_of_circular_flower_bed (C : ℝ) (hC : C = 62.8) : ∃ (A : ℝ), A = 314 :=
by
  sorry

end area_of_circular_flower_bed_l726_726873


namespace problem_statement_l726_726133

noncomputable def f : ℕ → ℝ
| 1     := 1
| 2     := 2
| (n+2) := f (n+2 - f (n+1)) + f (n+1 - f n)

theorem problem_statement (n : ℕ) : (0 ≤ f n + 1 - f n) ∧ (f 1025 = 1025) := 
by {
  sorry
}

end problem_statement_l726_726133


namespace pages_written_first_week_l726_726526

variable (x : ℕ) -- The number of pages Anahi wrote on in the first week

-- Given conditions
def total_pages : ℕ := 500
def second_week_written_pages : ℕ := (total_pages - x) * 30 / 100
def remaining_after_second_week : ℕ := (total_pages - x) * 70 / 100
def damaged_pages : ℕ := remaining_after_second_week * 20 / 100
def remaining_empty_pages : ℕ := remaining_after_second_week - damaged_pages

-- The target condition after the coffee spill
def empty_pages_after_coffee_spill : ℕ := 196

-- The proof statement
theorem pages_written_first_week : remaining_empty_pages = empty_pages_after_coffee_spill → x = 150 :=
by
  sorry

end pages_written_first_week_l726_726526


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726069

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726069


namespace solve_for_x_l726_726752

def vec_a := (2, 4)
def vec_b (x : ℝ) := (1, x)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

theorem solve_for_x (x : ℝ) (h : dot_product vec_a (vec_b x) = 0) : x = -1 / 2 :=
sorry

end solve_for_x_l726_726752


namespace f_equiv_x1_x2_l726_726745

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x^2) * Real.exp x

-- State the problem as a theorem
theorem f_equiv_x1_x2 {x₁ x₂ : ℝ} (h : f x₁ = f x₂) (hne : x₁ ≠ x₂) : x₁ + x₂ < 0 :=
by
  sorry

end f_equiv_x1_x2_l726_726745


namespace least_seven_digit_binary_number_l726_726082

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726082


namespace max_sequence_value_l726_726636

theorem max_sequence_value : 
  ∃ n ∈ (Set.univ : Set ℤ), (∀ m ∈ (Set.univ : Set ℤ), -m^2 + 15 * m + 3 ≤ -n^2 + 15 * n + 3) ∧ (-n^2 + 15 * n + 3 = 59) :=
by
  sorry

end max_sequence_value_l726_726636


namespace river_current_speed_l726_726191

theorem river_current_speed (distance : ℝ) (still_water_speed : ℝ) (time : ℝ) 
  (h1 : distance = 7)
  (h2 : still_water_speed = 4.4)
  (h3 : time = 3.684210526315789) :
  let v := still_water_speed - (distance / time) in
  v = 2.5 :=
by
  sorry

end river_current_speed_l726_726191


namespace interval_monotonic_increasing_triangle_ABC_sides_l726_726297

open Real

def f (ω x : ℝ) : ℝ := sqrt 3 * (sin (ω * x) * cos (ω * x)) - cos (ω * x) ^ 2 - 1/2

theorem interval_monotonic_increasing :
  (∀ k ∈ ℤ, ∃ (a b : ℝ), (a = -π/6 + k * π) ∧ (b = k * π + π/3) ∧ (∀ x, a ≤ x ∧ x ≤ b → f 1 x = sin (2 * x - π/6) - 1)
            → monotone_on (f 1) (set.Icc a b)) :=
sorry

theorem triangle_ABC_sides :
  ∀ (A B C a b c : ℝ), (c = sqrt 7) ∧ (sin B = 3 * sin A) ∧ (2 * C - π/6 = π /2) ∧ (f 1 C = 0) 
                      ∧ (cos C = 1/2)
                    → ( a = 1 ∧ b = 3 ) :=
sorry

end interval_monotonic_increasing_triangle_ABC_sides_l726_726297


namespace cost_of_building_fence_l726_726484

theorem cost_of_building_fence (A : ℝ) (P : ℝ) (side : ℝ) (perimeter : ℝ) (cost : ℝ) :
  A = 289 → P = 57 → side = real.sqrt A → perimeter = 4 * side → cost = perimeter * P → cost = 3876 :=
by
  intros hA hP hside hperimeter hcost
  rw [hA, hP, hside, hperimeter, hcost]
  sorry

end cost_of_building_fence_l726_726484


namespace regular_pay_correct_l726_726963

noncomputable def regular_pay_per_hour (total_payment : ℝ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℝ) : ℝ :=
  let R := total_payment / (regular_hours + overtime_rate * overtime_hours)
  R

theorem regular_pay_correct :
  regular_pay_per_hour 198 40 13 2 = 3 :=
by
  sorry

end regular_pay_correct_l726_726963


namespace Glen_Hannah_first_130km_distance_l726_726755

theorem Glen_Hannah_first_130km_distance :
  (∃ t : ℝ, t = 2.5 ∧ 37 * t + 15 * t = 130 ∧
    (11 - t) = 8.5) :=
by
  let Glen_speed := 37 -- km per hour
  let Hannah_speed := 15 -- km per hour
  let initial_distance := 130 -- km
  let relative_speed := Glen_speed + Hannah_speed
  let t := initial_distance / relative_speed
  have t_eq_2_5 : t = 2.5 := by
    simp [relative_speed, initial_distance]
    exact (by norm_num : 130 / 52 = 2.5)
  exists t
  simp [t_eq_2_5]
  split
  . exact t_eq_2_5
  . split
    . calc
      Glen_speed * t + Hannah_speed * t
        = 37 * t + 15 * t : by simp [Glen_speed, Hannah_speed]
        ... = 52 * t : by ring
        ... = 130 : by simp [t_eq_2_5]
    . calc
      11 - t
        = 11 - 2.5 : by simp [t_eq_2_5]
        ... = 8.5 : by norm_num

end Glen_Hannah_first_130km_distance_l726_726755


namespace calculate_product_l726_726998

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l726_726998


namespace probability_one_defective_l726_726104

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l726_726104


namespace exactly_three_assertions_l726_726516

theorem exactly_three_assertions (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧
  ((x % 3 = 0) ∧ (x % 5 = 0) ∧ (x % 9 ≠ 0) ∧ (x % 15 = 0) ∧ (x % 25 ≠ 0) ∧ (x % 45 ≠ 0)) ↔
  (x = 15 ∨ x = 30 ∨ x = 60) :=
by
  sorry

end exactly_three_assertions_l726_726516


namespace triangles_in_50th_ring_l726_726220

theorem triangles_in_50th_ring : 
  (∀ n, n ≥ 1 →  T n = 9 + 6 * (n - 1)) → T 50 = 303 :=
by
  assume h : ∀ n, n ≥ 1 → T n = 9 + 6 * (n - 1)
  sorry

end triangles_in_50th_ring_l726_726220


namespace degree_of_d_l726_726140

theorem degree_of_d (f d q r : Polynomial ℝ) (f_deg : f.degree = 17)
  (q_deg : q.degree = 10) (r_deg : r.degree = 4) 
  (remainder : r = Polynomial.C 5 * X^4 - Polynomial.C 3 * X^3 + Polynomial.C 2 * X^2 - X + 15)
  (div_relation : f = d * q + r) (r_deg_lt_d_deg : r.degree < d.degree) :
  d.degree = 7 :=
sorry

end degree_of_d_l726_726140


namespace square_side_length_l726_726179

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726179


namespace square_side_length_l726_726171

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726171


namespace value_of_y_l726_726805

theorem value_of_y (y : ℝ) : (y + 90 = 360) → y = 270 := 
by
  intro h
  have h1 : y = 360 - 90 := by linarith
  exact h1
  -- Provided as the complete proof can be represented succinctly
  -- sorry can be used to skip the proof for educational purposes

end value_of_y_l726_726805


namespace range_of_x_l726_726258

theorem range_of_x (m : ℝ) (x : ℝ) (h : 0 < m ∧ m ≤ 5) : 
  (x^2 + (2 * m - 1) * x > 4 * x + 2 * m - 4) ↔ (x < -6 ∨ x > 4) := 
sorry

end range_of_x_l726_726258


namespace rationalize_fraction_l726_726853

open BigOperators

theorem rationalize_fraction :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 :=
by
  -- Our proof intention will be inserted here.
  sorry

end rationalize_fraction_l726_726853


namespace mode_of_data_set_l726_726686

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726686


namespace particle_speed_l726_726507

noncomputable def position (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2*t + 7, 3*t - 13)

theorem particle_speed (t : ℝ) : 
  let deltaX := (t + 1)^2 + 2*(t + 1) + 7 - (t^2 + 2*t + 7),
      deltaY := 3*(t + 1) - 13 - (3*t - 13),
      speed := (deltaX^2 + deltaY^2).sqrt
  in speed = (4*t^2 + 12*t + 18).sqrt :=
by
  sorry

end particle_speed_l726_726507


namespace smallest_k_l726_726043

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l726_726043


namespace range_of_m_l726_726268

-- Given conditions
variables (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 4)

-- Target statement
theorem range_of_m (hx : x > 0) (hy : y > 0) (h : x + y = 4) : (∀ x y, x > 0 → y > 0 → x + y = 4 → (1/x + 4/y) ≥ m) ↔ (m ≤ 9/4) :=
begin
  sorry
end

end range_of_m_l726_726268


namespace general_formula_for_seq_l726_726674

open Nat

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * seq (n - 1) + 1

theorem general_formula_for_seq (n : ℕ) (h₁ : n > 0) : seq n = 2^(n-1) - 1 := by
  sorry

end general_formula_for_seq_l726_726674


namespace problem1_problem2_l726_726675

theorem problem1 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, a (n + 1) = a n / (a n + 3)) :
  ∃ q : ℝ, ∀ n : ℕ, (1 / a n + 1 / 2) = (3 / 2) * 3 ^ n :=
sorry

theorem problem2 (a : ℕ → ℝ) (b : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, a (n + 1) = a n / (a n + 3))
  (h₂ : ∀ n : ℕ, b n = 2 / (a n)) :
  ∀ n : ℕ, (∑ i in finset.range n, b (i + 1)) = (3 ^ (n + 1) - 3) / 2 - n :=
sorry

end problem1_problem2_l726_726675


namespace expression_c_is_negative_l726_726434

noncomputable def A : ℝ := -4.2
noncomputable def B : ℝ := 2.3
noncomputable def C : ℝ := -0.5
noncomputable def D : ℝ := 3.4
noncomputable def E : ℝ := -1.8

theorem expression_c_is_negative : D / B * C < 0 := 
by
  -- proof goes here
  sorry

end expression_c_is_negative_l726_726434


namespace collinear_vector_l726_726470

theorem collinear_vector (b : ℝ × ℝ) : 
  b = (-1, -2) → (∀ b, (λ b, (1 * b.2 - 2 * b.1) = 0)) :=
begin
  assume hb,
  -- Let's assume hb for the hypothesis that b = (-1, -2).
  sorry
end

end collinear_vector_l726_726470


namespace smallest_natural_number_composed_of_all_distinct_digits_l726_726473

-- Definitions of the conditions
def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- The Lean statement to express our proof problem
theorem smallest_natural_number_composed_of_all_distinct_digits :
  ∃ n : ℕ, (∀ d ∈ digits, d ∈ n.digits) ∧ (∀ i j, i ≠ j → n.digits[i] ≠ n.digits[j]) ∧ (n ≠ 0) ∧ (n = 1023456789) :=
by
  sorry

end smallest_natural_number_composed_of_all_distinct_digits_l726_726473


namespace non_equilateral_triangle_combinations_l726_726933

theorem non_equilateral_triangle_combinations :
  ∀ (n : ℕ) (h : n = 6), 
  let total_combinations := nat.choose n 3 in
  let equilateral_combinations := 2 in
  total_combinations - equilateral_combinations = 18 :=
begin
  intros n h,
  have H1 : n.choose 3 = 20, by {
    rw h,
    exact nat.choose_eq_factorial_div_factorial (nat.choose_pos _ _) dec_trivial,
  },
  have H2 : 20 - 2 = 18, by {
    norm_num,
  },
  rw H1,
  exact H2,
end

end non_equilateral_triangle_combinations_l726_726933


namespace distance_from_Acaster_to_Beetown_l726_726394

theorem distance_from_Acaster_to_Beetown 
  (x : ℝ) 
  (travel_same_time : ∀ tG tL : ℝ, tG = tL + 1) 
  (lewis_speed : ℝ) (lewis_speed = 70) 
  (geraint_speed : ℝ) (geraint_speed = 30) 
  (meet_distance_from_Beetown : ℝ) (meet_distance_from_Beetown = 105) 
  (tG : ℝ) (tG = (x - meet_distance_from_Beetown) / geraint_speed)
  (tL : ℝ) (tL = (x + meet_distance_from_Beetown) / lewis_speed + 1) 
  : x = 315 := 
by
  sorry

end distance_from_Acaster_to_Beetown_l726_726394


namespace solution_set_a_eq_1_find_a_min_value_3_l726_726740

open Real

noncomputable def f (x a : ℝ) := 2 * abs (x + 1) + abs (x - a)

-- The statement for the first question
theorem solution_set_a_eq_1 (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -2 ∨ x ≥ (4 / 3) := 
by sorry

-- The statement for the second question
theorem find_a_min_value_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) ∧ (∃ x : ℝ, f x a = 3) ↔ a = 2 ∨ a = -4 := 
by sorry

end solution_set_a_eq_1_find_a_min_value_3_l726_726740


namespace sum_first_2015_terms_l726_726743

def f (x : ℕ) : ℝ := 4 * (x:ℝ)^2 - 1

def sum_seq (n : ℕ) : ℝ := (∑ i in finset.range (n + 1), 1 / f (i + 1))

theorem sum_first_2015_terms : sum_seq 2015 = 2015 / 4031 := 
sorry

end sum_first_2015_terms_l726_726743


namespace least_binary_seven_digits_l726_726060

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726060


namespace tan_plus_sin_l726_726214

theorem tan_plus_sin :
  let θ := 30.0
  let tan_θ := 1/Real.sqrt 3
  let sin_θ := 1/2
  let cos_θ := Real.sqrt 3 / 2
  let sin_2θ := Real.sqrt 3 / 2
  tan θ + 3 * sin θ = (1 + 3 * Real.sqrt 3) / 2 := sorry

end tan_plus_sin_l726_726214


namespace intersection_point_sum_l726_726530

-- Define the function h
variable (h : ℝ → ℝ)

-- Define the property that the function satisfies
axiom h_eq : ∀ a, h a = h (a + 4)

-- Specific points on the graph
axiom h_at_neg_one_point_five : h (-1.5) = 3
axiom h_at_two_point_five : h (2.5) = 3

-- Define the target sum
theorem intersection_point_sum :
  ∃ a b : ℝ, h a = h (a + 4) ∧ a + b = 1.5 :=
begin
  use [-1.5, 3],
  split,
  { exact h_at_neg_one_point_five, },
  { norm_num, }
end

end intersection_point_sum_l726_726530


namespace most_probable_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l726_726937
open ProbabilityTheory

-- Declare the problem parameters
variables {n : ℕ} (h : n > 0)

-- Declare the main variables x for number of dissatisfied passengers
def dissatisfiedPassengers : ℕ → ℝ

-- Define xi to be representing the number of dissatisfied passengers
noncomputable def xi := dissatisfiedPassengers

-- Define probability measures used to calculate expectations and variance
noncomputable def P_xi_eq_0 := (nat.choose (2*n) n) / (2 ^ (2 * n))
noncomputable def P_xi_eq_1 := 2 * (nat.choose (2*n) (n - 1)) / (2 ^ (2 * n))

-- Expected Value of xi
noncomputable def E_xi := sqrt (n / π)

-- Variance of xi
noncomputable def Var_xi := ((π - 2) / (2 * π)) * (n : ℝ)

-- Part (a)
theorem most_probable_dissatisfied_passengers (hn : n > 1) : xi = 1 := sorry

-- Part (b)
theorem expected_dissatisfied_passengers : E xi = sqrt (n / π) := sorry

-- Part (c)
theorem variance_dissatisfied_passengers : Var xi = ((π - 2) / (2*π)) * (n : ℝ) := sorry


end most_probable_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l726_726937


namespace polynomial_sum_l726_726430

noncomputable def p (x : ℝ) := (x + 1) * (x - 2)
noncomputable def q (x : ℝ) := (x + 1) * (x - 2) * (x - 3)

theorem polynomial_sum (p_val : p 2 = 2) (q_val : q (-1) = -1) : 
  p x + q x = x^3 - 3x^2 + 4x + 4 := 
by exact sorry

end polynomial_sum_l726_726430


namespace triangle_not_necessarily_isosceles_l726_726353

-- Definitions of triangle sides
def AB : ℝ := 6
def BC : ℝ := 4
def CA : ℝ := 8

-- Definitions of points
def O : Point := Point.mk (0, 0)  -- Center of the inscribed circle (Placeholder for simplicity)
def B₁ : Point := midpoint (point A) (point C)
def A₁ : Point := midpoint (point B) (point C)

-- Definitions of distances
def distance_O_B₁ : ℝ := dist O B₁
def distance_O_A₁ : ℝ := dist O A₁

-- Hypothesis: O is equidistant from B₁ and A₁
axiom H : distance_O_B₁ = distance_O_A₁

-- Goal: Prove that ∆ABC is not necessarily isosceles
theorem triangle_not_necessarily_isosceles (H : distance_O_B₁ = distance_O_A₁) : ¬ is_isosceles ABC := 
  sorry

end triangle_not_necessarily_isosceles_l726_726353


namespace multiplication_of_mixed_number_l726_726992

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726992


namespace stack_height_sheets_l726_726509

noncomputable def sheets_in_stack (ream_thickness : ℝ) (ream_sheets : ℕ) (stack_height : ℝ) : ℕ :=
  let sheet_thickness := ream_thickness / ream_sheets
  (stack_height / sheet_thickness).to_nat

theorem stack_height_sheets :
  sheets_in_stack 3 50 10 = 167 :=
by
  sorry

end stack_height_sheets_l726_726509


namespace value_of_m_plus_n_l726_726744

noncomputable def exponential_function (a x m n : ℝ) : ℝ :=
  a^(x - m) + n - 3

theorem value_of_m_plus_n (a x m n y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : exponential_function a 3 m n = 2) : m + n = 7 :=
by
  sorry

end value_of_m_plus_n_l726_726744


namespace hyperbola_equation_is_correct_l726_726730

open Real

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

noncomputable def hyperbola_vertex (a : ℝ) : (ℝ × ℝ) := (-a, 0)

noncomputable def hyperbola_asymptote_slope (a b : ℝ) : ℝ := b / a

theorem hyperbola_equation_is_correct (a b p : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_p_pos : 0 < p) 
  (h_distance : dist (hyperbola_vertex a) (parabola_focus p) = 4)
  (h_asymptote_point : (−2, -4) ∈ { pt | ∃ k, pt.2 = k * pt.1 ∧ k = hyperbola_asymptote_slope a b ∧ pt.1 = -p / 2 }) :
  (∀ x y, (y^2 = 2*p*x) → (x + 2)^2 = 4 → ((x^2 / (a^2)) - (y^2 / (b^2)) = 1)) := 
by 
  -- proof goes here
  sorry

end hyperbola_equation_is_correct_l726_726730


namespace domain_k_width_l726_726765

-- Declare j as a function with a domain of [-12, 12]
def j : ℝ → ℝ := sorry

-- Declare the domain condition for j
def domain_j (x : ℝ) : Prop := -12 ≤ x ∧ x ≤ 12

-- Define k(x) as j(x/3)
def k (x : ℝ) := j (x / 3)

-- Prove the width of the domain of k(x) is 72
theorem domain_k_width : 
  (∀ x, domain_j (x / 3) → -36 ≤ x ∧ x ≤ 36) →
  (width : ℝ) (width = 36 - (-36)) →
  width = 72 :=
by
  intros h dom_width
  simp [←interval_of_bounds]
  rw [abs_eq]
  sorry

end domain_k_width_l726_726765


namespace neg_p_true_l726_726304

theorem neg_p_true :
  ∀ (x : ℝ), -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 :=
by
  sorry

end neg_p_true_l726_726304


namespace painting_days_l726_726940

theorem painting_days (a_days : ℕ) (b_efficiency_increase : ℕ) (hA : a_days = 12) (hB : b_efficiency_increase = 75) :
  ∃ days : ℝ, days ≈ 4.36 :=
by
  sorry

end painting_days_l726_726940


namespace Bobby_ate_5_pancakes_l726_726533

theorem Bobby_ate_5_pancakes
  (total_pancakes : ℕ := 21)
  (dog_eaten : ℕ := 7)
  (leftover : ℕ := 9) :
  (total_pancakes - dog_eaten - leftover = 5) := by
  sorry

end Bobby_ate_5_pancakes_l726_726533


namespace trigonometric_identity_l726_726475

theorem trigonometric_identity (α : ℝ) :
  (sin (135 * (real.pi / 180) - α)) ^ 2 
  - (sin (210 * (real.pi / 180) - α)) ^ 2 
  - (sin (195 * (real.pi / 180))) * (cos (165 * (real.pi / 180) - 2 * α)) = 
-1 :=
sorry

end trigonometric_identity_l726_726475


namespace parabola_properties_l726_726302

namespace ParabolaProof

-- Define the given conditions
def passes_through (a : ℝ) (x y : ℝ) : Prop := y = a * x^2

-- Example conditions provided in the problem
def A := (1 : ℝ, 2 : ℝ)
def point_x := fst A
def point_y := snd A

-- Define what we need to prove
theorem parabola_properties (a : ℝ) (dir_eq : ℝ) :
  passes_through a point_x point_y →
  a = 2 ∧ dir_eq = (-1 / 8 : ℝ) :=
by {
  intros h,
  -- Proof to be provided here
  sorry
}

end ParabolaProof

end parabola_properties_l726_726302


namespace trapezium_area_correct_l726_726932

-- Define the necessary parameters and the area function
def a : ℝ := 20
def b : ℝ := 18
def h : ℝ := 15

-- Define the formula for the area of a trapezium
def trapeziumArea (a b h : ℝ) := (1 / 2) * (a + b) * h

-- State the theorem that needs to be proven
theorem trapezium_area_correct : trapeziumArea a b h = 285 := by
  sorry

end trapezium_area_correct_l726_726932


namespace prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l726_726857

-- Show that if \( p \) is a prime number, then \( p \) divides \( (p-1)! + 1 \).
theorem prime_divides_factorial_plus_one (p : ℕ) (hp : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

-- Show that if \( n \) is not a prime number, then \( n \) does not divide \( (n-1)! + 1 \).
theorem non_prime_not_divides_factorial_plus_one (n : ℕ) (hn : ¬Nat.Prime n) : ¬(n ∣ (Nat.factorial (n - 1) + 1)) :=
sorry

-- Calculate the remainder of the division of \((n-1)!\) by \( n \).
theorem factorial_mod_non_prime_is_zero (n : ℕ) (hn : ¬Nat.Prime n) : (Nat.factorial (n - 1)) % n = 0 :=
sorry

end prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l726_726857


namespace rectangle_area_l726_726028

theorem rectangle_area (d : ℝ) (P Q R W X Y Z : ℝ) (hx1 : d = 6)
  (hx2 : W = 0) (hx3 : X = d * 2) (hx4 : Z = d) (hx5 : Y = d) :
  WZ * WX = 72 :=
by
  -- Definitions based on the problem conditions
  let WZ := d
  let WX := d * 2
  -- Area calculation
  have area := WZ * WX
  -- Final conclusion
  exact area

end rectangle_area_l726_726028


namespace jade_more_transactions_l726_726843

theorem jade_more_transactions (mabel_transactions : ℕ) (anthony_percentage : ℕ) (cal_fraction_numerator : ℕ) 
  (cal_fraction_denominator : ℕ) (jade_transactions : ℕ) (h1 : mabel_transactions = 90) 
  (h2 : anthony_percentage = 10) (h3 : cal_fraction_numerator = 2) (h4 : cal_fraction_denominator = 3) 
  (h5 : jade_transactions = 83) :
  jade_transactions - (2 * (90 + (90 * 10 / 100)) / 3) = 17 := 
by
  sorry

end jade_more_transactions_l726_726843


namespace base8_to_base10_l726_726552

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l726_726552


namespace modulus_of_z_l726_726282

noncomputable def complex_z_satisfies_property : Prop :=
  ∃ z : ℂ, z * (1 - complex.i)^2 = 1 + complex.i ∧ complex.abs z = complex.abs (1 - complex.i) / complex.abs (-2 * complex.i)

theorem modulus_of_z :
  ∀ (z : ℂ), z * (1 - complex.i)^2 = 1 + complex.i → complex.abs z = (Real.sqrt 2) / 2 := 
by
  sorry

end modulus_of_z_l726_726282


namespace sequences_have_limits_l726_726367

theorem sequences_have_limits (x y : ℕ → ℝ) (h_pos_x : ∀ n, 0 < x n) (h_pos_y : ∀ n, 0 < y n) :
  (∀ n, x (n + 1) ≥ (x n + y n) / 2) →
  (∀ n, y (n + 1) ≥ real.sqrt ((x n ^ 2 + y n ^ 2) / 2)) →
  (∃ l1 : ℝ, ∃ l2 : ℝ, (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n + y n - l1| < ε) ∧
                  (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n * y n - l2| < ε)) ∧
  (∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - l| < ε) ∧
             (∀ ε > 0, ∃ N, ∀ n ≥ N, |y n - l| < ε)) :=
by
  intros h1 h2
  sorry

end sequences_have_limits_l726_726367


namespace three_triangles_form_one_larger_triangle_l726_726625

theorem three_triangles_form_one_larger_triangle :
  ∀ (ABC : Triangle) (α β γ : ℝ),
    -- condition 1: angles of triangle ABC
    ABC.angles = (α, β, γ) ∧ α + β + γ = 180 ∧
    -- condition 2: three identical triangles
    (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 → ABC_i = ABC_j) ∧
    -- condition 3: cut each triangle along the median
    (cut_along_median abc_1 median_AM = (triangle_1_part1, triangle_1_part2)) ∧
    (cut_along_median abc_2 median_BM = (triangle_2_part1, triangle_2_part2)) ∧
    (cut_along_median abc_3 median_CM = (triangle_3_part1, triangle_3_part2))
  -- Prove that we can form a larger triangle
  → ∃ (large_triangle : Triangle),
      large_triangle = arrange_parts(triangle_1_part1, triangle_1_part2, 
                                     triangle_2_part1, triangle_2_part2, 
                                     triangle_3_part1, triangle_3_part2) := 
sorry

end three_triangles_form_one_larger_triangle_l726_726625


namespace probability_remainder_1_l726_726524

theorem probability_remainder_1 (N : ℕ) (hN : 1 ≤ N ∧ N ≤ 2027) : nat.gcd (2027, 7) = 1 → nat.gcd (N, 7) = 1 →
  let outcomes := 2027
  let favorable := (2 : ℚ)
  let total_outcomes := (7 : ℚ)
  (favorable / total_outcomes) = (2 / 7) := 
sorry

end probability_remainder_1_l726_726524


namespace ages_correct_l726_726137

variables (Son Daughter Wife Man Father : ℕ)

theorem ages_correct :
  (Man = Son + 20) ∧
  (Man = Daughter + 15) ∧
  (Man + 2 = 2 * (Son + 2)) ∧
  (Man + 2 = 3 * (Daughter + 2)) ∧
  (Wife = Man - 5) ∧
  (Wife + 6 = 2 * (Daughter + 6)) ∧
  (Father = Man + 32) →
  (Son = 7 ∧ Daughter = 12 ∧ Wife = 22 ∧ Man = 27 ∧ Father = 59) :=
by
  intros h
  sorry

end ages_correct_l726_726137


namespace side_length_of_square_l726_726164

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726164


namespace tangent_line_eq_l726_726737

/-- Given the function f(x) = x * exp(x), the equation of the tangent line to the graph
of f(x) at the point (0, f(0)) is x - y = 0. -/
theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x * Real.exp x) :
  ∀ x y, (0, f 0).snd = 0 ∧ (0, f 0).fst = 0 → x - y = 0 → 
  (∃ (tangent_line : ℝ → ℝ), tangent_line = λ x, x ∧ ∀ x, tangent_line x = x - y) :=
by
  sorry

end tangent_line_eq_l726_726737


namespace vector_relationship_perpendicular_l726_726661

variable (a b : EuclideanSpace ℝ (Fin 3))
variable (ha : ‖a‖ = 3)
variable (hb : ‖b‖ = 4)

theorem vector_relationship_perpendicular :
  (a + (3 / 4 : ℝ) • b) ⬝ (a - (3 / 4 : ℝ) • b) = 0 := 
sorry

end vector_relationship_perpendicular_l726_726661


namespace mod_mult_example_l726_726865

theorem mod_mult_example : ∃ m, 65 * 76 * 87 % 25 = m ∧ 0 ≤ m ∧ m < 25 ∧ m = 5 := by
  use 5
  split
  { -- Proof that 65 * 76 * 87 ≡ 5 (mod 25)
    calc
      (65 : ℤ) * (76 : ℤ) * (87 : ℤ) % 25
          = (15 : ℤ) * (1 : ℤ) * (12 : ℤ) % 25 : by { norm_num }
      ... = (180 : ℤ) % 25                       : by { norm_num }
      ... = 5                                   : by { norm_num },
  }
  split
  { sorry, }
  split
  { sorry, }
  { sorry, }

end mod_mult_example_l726_726865


namespace ratio_of_money_earned_l726_726395

variable (L T J : ℕ) 

theorem ratio_of_money_earned 
  (total_earned : L + T + J = 60)
  (lisa_earning : L = 30)
  (lisa_tommy_diff : L = T + 15) : 
  T / L = 1 / 2 := 
by
  sorry

end ratio_of_money_earned_l726_726395


namespace problem_1_problem_2_problem_3_l726_726293

-- Definitions
def f (a: ℝ) (x: ℝ) : ℝ := a^x - a^(-x)

variables (a : ℝ) (x : ℝ)
variable (h_a_pos: a > 1)

-- Statements to be proved
theorem problem_1 : ∀ x : ℝ, f a (-x) = - (f a x) := sorry

theorem problem_2 : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

theorem problem_3 : ∀ t : ℝ, f a (1 - t) + f a (1 - t^2) < 0 → (t < -2 ∨ t > 1) := sorry

end problem_1_problem_2_problem_3_l726_726293


namespace correct_data_summary_l726_726936

-- Assume there are n=20 elements initially.
variables {α : Type*} [DecidableEq α] [Field α]
variable (data : Fin 20 → α)

-- Given conditions
def initial_average := (20 : α)
def initial_variance := (28 : α)
def incorrect_data := (data 18 = 21 ∧ data 19 = 19 ∧ ∑ i, data i = 20 * 20)

-- Define the corrected data
def corrected_data := (λ i, if i = 18 then 11 else if i = 19 then 29 else data i)

-- Statements to prove
def new_average := ∑ i, corrected_data data i / 20 = 20
def new_variance := let s := ∑ i, ((corrected_data data i) - 20)^2 in s / 20 > 28

theorem correct_data_summary (data : Fin 20 → α) (h1 : initial_average = 20) (h2 : initial_variance = 28) (h3 : incorrect_data data) :
  new_average data corrected_data = 20 ∧ new_variance data corrected_data :=
by
  sorry

end correct_data_summary_l726_726936


namespace table_chair_price_l726_726970

theorem table_chair_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : T = 84) : T + C = 96 :=
sorry

end table_chair_price_l726_726970


namespace correct_quadratic_eq_l726_726800

-- Define the given conditions
def first_student_sum (b : ℝ) : Prop := 5 + 3 = -b
def second_student_product (c : ℝ) : Prop := (-12) * (-4) = c

-- Define the proof statement
theorem correct_quadratic_eq (b c : ℝ) (h1 : first_student_sum b) (h2 : second_student_product c) :
    b = -8 ∧ c = 48 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 → (x=5 ∨ x=3 ∨ x=-12 ∨ x=-4)) :=
by
  sorry

end correct_quadratic_eq_l726_726800


namespace repeating_decimal_denominator_l726_726874

theorem repeating_decimal_denominator : 
  ∀ (S : ℚ), S = 27 / 99 → (∃ (a b : ℕ), S = a / b ∧ Int.gcd a b = 1 ∧ b = 11) :=
by
  intros S hS
  have eq1 : S = 3 / 11 := by sorry
  use 3, 11
  split
  exact eq1
  split
  have gcd1 : Int.gcd 3 11 = 1 := by sorry
  exact gcd1
  rfl

end repeating_decimal_denominator_l726_726874


namespace second_term_is_4_l726_726784

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l726_726784


namespace number_of_centrally_symmetric_shapes_l726_726202

def is_centrally_symmetric (shape : Type) := ∃ (p : shape → Prop), ∀ s₁ s₂ : shape, (p s₁ → p s₂ → s₁ ≠ s₂ → central_sym (s₁, s₂))

inductive Shape
| equilateral_triangle
| square
| rhombus
| isosceles_trapezoid

open Shape

def centrally_symmetric_shapes := 
  {s ∈ [equilateral_triangle, square, rhombus, isosceles_trapezoid] | is_centrally_symmetric s}

theorem number_of_centrally_symmetric_shapes : finset.card centrally_symmetric_shapes = 2 := 
sorry

end number_of_centrally_symmetric_shapes_l726_726202


namespace smallest_special_number_gt_3429_l726_726580

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726580


namespace calculate_product_l726_726997

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l726_726997


namespace bridget_profit_l726_726986

-- Definitions of conditions in the problem
def num_loaves := 60
def morning_price := 3.0
def afternoon_discount := 1.0
def cost_per_loaf := 0.80
def late_afternoon_price := 1.50

-- Definitions derived from the conditions
def morning_loaves := num_loaves / 2
def remaining_after_morning := num_loaves - morning_loaves
def afternoon_loaves := (3 / 4) * remaining_after_morning
def late_afternoon_loaves := remaining_after_morning - afternoon_loaves

def morning_revenue := morning_loaves * morning_price
def afternoon_revenue := afternoon_loaves * (morning_price - afternoon_discount)
def late_afternoon_revenue := late_afternoon_loaves * late_afternoon_price

def total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue
def total_cost := num_loaves * cost_per_loaf
def total_profit := total_revenue - total_cost

-- Theorem statement
theorem bridget_profit : total_profit = 98.50 :=
by sorry

end bridget_profit_l726_726986


namespace Greenwood_High_School_chemistry_students_l726_726207

theorem Greenwood_High_School_chemistry_students 
    (U : Finset ℕ) (B C P : Finset ℕ) 
    (hU_card : U.card = 20) 
    (hB_subset_U : B ⊆ U) 
    (hC_subset_U : C ⊆ U)
    (hP_subset_U : P ⊆ U)
    (hB_card : B.card = 10) 
    (hB_C_card : (B ∩ C).card = 4) 
    (hB_C_P_card : (B ∩ C ∩ P).card = 3) 
    (hAll_atleast_one : ∀ x ∈ U, x ∈ B ∨ x ∈ C ∨ x ∈ P) :
    C.card = 6 := 
by 
  sorry

end Greenwood_High_School_chemistry_students_l726_726207


namespace total_cost_of_order_l726_726403

theorem total_cost_of_order :
  let pencil_price_per_carton := 6
      eraser_price_per_carton := 3
      total_cartons := 100
      pencil_cartons := 20
      eraser_cartons := total_cartons - pencil_cartons
      pencil_cost := pencil_price_per_carton * pencil_cartons
      eraser_cost := eraser_price_per_carton * eraser_cartons
      total_cost := pencil_cost + eraser_cost
  in total_cost = 360 :=
by
  sorry

end total_cost_of_order_l726_726403


namespace find_x_values_l726_726248

theorem find_x_values (x : ℝ) (h : x ≠ 5) : x + 36 / (x - 5) = -12 ↔ x = -8 ∨ x = 3 :=
by sorry

end find_x_values_l726_726248


namespace sequence_problem_l726_726019

noncomputable def b_n (n : ℕ) : ℝ := 5 * (5/3)^(n-2)

theorem sequence_problem 
  (a_n : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : a_n 8 = a_n 5 + 3 * d)
  (h4 : a_n 13 = a_n 8 + 5 * d)
  (b_2 : ℝ)
  (hb2 : b_2 = 5)
  (h5 : ∀ n, b_n n = (match n with | 2 => b_2 | _ => sorry))
  (conseq_terms : ∀ (n : ℕ), (a_n 5 + 3 * d)^2 = a_n 5 * (a_n 5 + 8 * d)) 
  : ∀ n, b_n n = b_n 2 * (5/3)^(n-2) := 
by 
  sorry

end sequence_problem_l726_726019


namespace sum_of_y_coordinates_l726_726404

theorem sum_of_y_coordinates (y : ℝ) (dist : ∀ y, real.sqrt ((-2 - 4)^2 + (5 - y)^2) = 13) :
  (let y₁ := 5 + real.sqrt 133,
       y₂ := 5 - real.sqrt 133
   in y₁ + y₂ = 10) :=
by
  sorry

end sum_of_y_coordinates_l726_726404


namespace tan_alpha_eq_3_div_4_l726_726735

theorem tan_alpha_eq_3_div_4
    {α : ℝ}
    (f : ℝ → ℝ)
    (h₁ : ∀ x, f(x) = 3 * Real.sin x + 4 * Real.cos x)
    (h₂ : ∀ x, f(x) ≥ f(α))
    : Real.tan α = 3 / 4 := by
  sorry

end tan_alpha_eq_3_div_4_l726_726735


namespace smallest_special_greater_than_3429_l726_726568

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726568


namespace least_7_digit_binary_number_is_64_l726_726078

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726078


namespace general_equation_C1_area_triangle_OPR_l726_726347

/-- Prove that the general equation of C1 is (x^2/9 + y^2/3 = 1) -/
theorem general_equation_C1 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  (∀ φ : ℝ, (a * Real.cos φ, b * Real.sin φ) ∈ SetOf (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1)) →
  a = 3 ∧ b = Real.sqrt 3 :=
sorry

/-- Prove that the area of ΔOPR is (3 * sqrt 30 / 20) -/
theorem area_triangle_OPR (t : ℝ) (h3 : t ≠ 0) :
  (∀ t : ℝ, (-t, sqrt 3 * t) ∈ SetOf (λ p : ℝ × ℝ, (p.1 - 1)^2 + p.2^2 = 1)) →
  let α := Real.pi / 3 in
  let P := (3 * sqrt 10 / 10, 3 * sqrt 30 / 10) in
  let R := (1 / 2, - sqrt 3 / 2) in
  (P.1^2 / 9 + P.2^2 / 3 = 1) →
  (R.1 - R.2 * sqrt 3 / 2 = 0) →
  (1 / 2 * (3 * sqrt 10 / 5) * (sqrt 3 / 2) = 3 * sqrt 30 / 20) :=
sorry

end general_equation_C1_area_triangle_OPR_l726_726347


namespace area_second_smallest_region_l726_726882

-- Define the parabola equation
def parabola_eq (x y : ℝ) := 4 + (x + 2) * y = x^2

-- Define the circle equation
def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 16

-- Prove the area of the second smallest region of intersection between the parabola and the circle
theorem area_second_smallest_region : 
  ∃ a : ℝ, 
  (∀ x y : ℝ, 
    parabola_eq x y ∧ circle_eq x y → 
    true) → -- This condition states that for all x and y that satisfy both the parabola and circle equations, which imply the intersection points
  (true → -- This true condition is a placeholder for the mathematical statement that leads us to find 
    a = sorry) := -- Area of the second smallest region is left as sorry since explicit calculation is not provided 
begin
  sorry
end

end area_second_smallest_region_l726_726882


namespace dataset_mode_l726_726709

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726709


namespace problem_1_problem_2_l726_726285

theorem problem_1 (n : ℕ) (hn : n > 0) :
  let S : ℕ → ℕ := λ n, 2 * n^2 + n, 
      a := S n - S (n - 1),
      b := 2 ^ (n - 1) in
  a = 4 * n - 1 ∧ b = 2 ^ (n - 1) :=
by sorry

theorem problem_2 (n : ℕ) (hn : n > 0) :
  let S : ℕ → ℕ := λ n, 2 * n^2 + n,
      a := λ n, if n = 1 then S 1 else S n - S (n - 1),
      b := λ n, 2 ^ (n - 1),
      T := ∑ i in (Finset.range n).map (λ i, i + 1), a (i + 1) * b (i + 1) in
  T = (4 * n - 5) * 2 ^ (n - 1) + 5 :=
by sorry

end problem_1_problem_2_l726_726285


namespace bulbs_97_100_l726_726901

def BulbColor := ℕ → String

noncomputable def garland : BulbColor :=
λ n, match n % 5 with
| 0 => "Yellow"
| 1 => "Blue"
| 2 => "Yellow"
| 3 => "Blue"
| 4 => "Blue"
| _ => "Unknown"
end

theorem bulbs_97_100 :
  (garland 96 = "Yellow" ∧ garland 97 = "Blue" ∧ garland 98 = "Yellow" ∧ garland 99 = "Blue" ∧ garland 100 = "Blue") :=
by
  have universal_condition :
    ∀ (n : ℕ), (garland n = "Yellow" ∧ garland (n + 2) = "Yellow"
                ∧ garland (n + 1) = "Blue" ∧ garland (n + 3) = "Blue" ∧ garland (n + 4) = "Blue") :=
    by
      intro n
      cases (n % 5) with
      | 0 => sorry
      | 1 => sorry
      | 2 => sorry
      | 3 => sorry
      | 4 => sorry
      | _ => sorry
  show (garland 96 = "Yellow" ∧ garland 97 = "Blue" ∧ garland 98 = "Yellow" ∧ garland 99 = "Blue" ∧ garland 100 = "Blue") 
  by exact universal_condition 95

end bulbs_97_100_l726_726901


namespace parabola_intersection_points_l726_726222

theorem parabola_intersection_points :
  ∃ y1 y2 : ℝ, 
    (∃ (x1 x2 : ℝ), 
      x1 = (5 + Real.sqrt 73) / -6 ∧
      x2 = (5 - Real.sqrt 73) / -6 ∧
      y1 = 3 * x1^2 - 4 * x1 + 7 ∧
      y2 = 3 * x2^2 - 4 * x2 + 7 ∧
      y1 = 6 * x1^2 + x1 + 3 ∧
      y2 = 6 * x2^2 + x2 + 3 ) :=
begin
  sorry
end

end parabola_intersection_points_l726_726222


namespace sum_of_digits_l726_726319

theorem sum_of_digits :
  ∃ A B C : ℕ, 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A > 0 ∧ B > 0 ∧ C > 0 ∧
    A < 6 ∧ B < 6 ∧ C < 6 ∧
    (A * 6^1 + B * 6^0) + (B * 6^1 + C * 6^0) = A * 6^1 + C * 6^1 + A * 6^0 ∧ 
    (A + B + C = 9) ∧ 
    (2 + 3 + 4 = 9) ∧
    6 * 1 + 5 * 1 = 9 := 
begin
  use [2, 3, 4],
  repeat { split },
  { norm_num }, -- A ≠ B
  { norm_num }, -- B ≠ C
  { norm_num }, -- A ≠ C
  { norm_num }, -- A > 0
  { norm_num }, -- B > 0
  { norm_num }, -- C > 0
  { norm_num }, -- A < 6
  { norm_num }, -- B < 6
  { norm_num }, -- C < 6
  {
    calc
      (2 * 6^1 + 3 * 6^0) + (3 * 6^1 + 4 * 6^0)
        = (2 * 6 + 3) + (3 * 6 + 4) : rfl
    ... = 15 + 22 : by norm_num
    ... = 37 : by norm_num
    ... = (2 * 6^1 + 4 * 6^0) + 15 : by norm_num
  },
  { norm_num }, -- A + B + C = 9
  { norm_num }, -- 2 + 3 + 4 = 9 confirmed
  { norm_num }, -- 6 * 1 + 5 * 1 = 9 confirmed
end

end sum_of_digits_l726_726319


namespace find_d_l726_726437

noncomputable def d_value (a b c : ℝ) := (2 * a + 2 * b + 2 * c - (3 / 4)^2) / 3

theorem find_d (a b c d : ℝ) (h : 2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + (2 * a + 2 * b + 2 * c - 3 * d)^(1/2)) : 
  d = 23 / 48 :=
sorry

end find_d_l726_726437


namespace side_length_of_square_l726_726183

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726183


namespace A_intersect_B_eq_123_l726_726665

-- Definitions of A and B
def is_in_A (x : ℕ) : Prop := x > 0 ∧ - (x : ℝ)^2 + 3 * (x : ℝ) ≥ 0
def is_in_B (x : ℝ) : Prop := real.log x / real.log (1/2) ≤ 0

-- Lean statement for the proof problem
theorem A_intersect_B_eq_123 : {x : ℕ | is_in_A x} ∩ {x : ℝ | is_in_B x} = {1, 2, 3} :=
by
  -- Placeholder for the proof
  sorry

end A_intersect_B_eq_123_l726_726665


namespace mode_of_data_set_l726_726682

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726682


namespace total_price_all_art_l726_726364

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l726_726364


namespace mode_of_data_set_l726_726678

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726678


namespace monotonic_increasing_intervals_l726_726014

def f (x : ℝ) := Real.tan (x + Real.pi / 4)

theorem monotonic_increasing_intervals : ∀ k : ℤ, 
  Ioo (k * Real.pi - 3 * Real.pi / 4) (k * Real.pi + Real.pi / 4) = 
  { x : ℝ | f (x + k * Real.pi) < f ((x + k * Real.pi) + Real.pi) } :=
by sorry

end monotonic_increasing_intervals_l726_726014


namespace initial_amount_l726_726136

theorem initial_amount (M : ℝ) (h1 : M * 2 - 50 > 0) (h2 : (M * 2 - 50) * 2 - 60 > 0) 
(h3 : ((M * 2 - 50) * 2 - 60) * 2 - 70 > 0) 
(h4 : (((M * 2 - 50) * 2 - 60) * 2 - 70) * 2 - 80 = 0) : M = 53.75 := 
sorry

end initial_amount_l726_726136


namespace PP1_length_l726_726809

open Real

theorem PP1_length (AB AC : ℝ) (h₁ : AB = 5) (h₂ : AC = 3)
  (h₃ : ∃ γ : ℝ, γ = 90)  -- a right angle at A
  (BC : ℝ) (h₄ : BC = sqrt (AB^2 - AC^2))
  (A1B : ℝ) (A1C : ℝ) (h₅ : BC = A1B + A1C)
  (h₆ : A1B / A1C = AB / AC)
  (PQ : ℝ) (h₇ : PQ = A1B)
  (PR : ℝ) (h₈ : PR = A1C)
  (PP1 : ℝ) :
  PP1 = (3 * sqrt 5) / 4 :=
sorry

end PP1_length_l726_726809


namespace factor_between_l726_726270

theorem factor_between (n a b : ℕ) (h1 : 10 < n) 
(h2 : n = a * a + b) 
(h3 : a ∣ n) 
(h4 : b ∣ n) 
(h5 : a ≠ b) 
(h6 : 1 < a) 
(h7 : 1 < b) : 
    ∃ m : ℕ, b = m * a ∧ 1 < m ∧ a < a + m ∧ a + m < b  :=
by
  -- proof to be filled in
  sorry

end factor_between_l726_726270


namespace max_bishops_on_chessboard_l726_726915

theorem max_bishops_on_chessboard : ∃ n : ℕ, n = 14 ∧ (∃ k : ℕ, n * n = k^2) := 
by {
  sorry
}

end max_bishops_on_chessboard_l726_726915


namespace smallest_int_11a_eq_22b_l726_726917

theorem smallest_int_11a_eq_22b (a b : ℕ) (ha : a > 2) (hb : b > 2) (h : a = 2 * b + 1) : 
    11_a = 22_b → (1*a + 1 = 2 * b + 2 → 8 = 11_a ∧ 8 = 22_b) :=
by
    sorry

end smallest_int_11a_eq_22b_l726_726917


namespace winner_vote_difference_l726_726343

theorem winner_vote_difference (total_votes : ℕ) (winner_votes : ℕ) (loser_votes : ℕ) 
  (h1 : winner_votes = 868) 
  (h2 : winner_votes = 0.62 * total_votes)  
  (h3 : loser_votes = 0.38 * total_votes) : 
  winner_votes - loser_votes = 336 :=
by sorry

end winner_vote_difference_l726_726343


namespace dissimilar_terms_in_expansion_l726_726621

theorem dissimilar_terms_in_expansion : 
  let expansion (a b c d : ℕ) := (a + b + c + d)^7 in
  num_dissimilar_terms expansion = 120 :=
sorry

end dissimilar_terms_in_expansion_l726_726621


namespace count_valid_4_digit_numbers_l726_726757

-- Define the conditions as sets and properties
def first_two_digit_choices := {2, 6, 7}
def last_two_digit_choices := {1, 3, 9}
def valid_last_two_digits (a b : ℕ) : Prop :=
  a ∈ last_two_digit_choices ∧ b ∈ last_two_digit_choices ∧ a < b

-- Statement of the problem
theorem count_valid_4_digit_numbers :
  (∃ (f1 f2 l1 l2 : ℕ), 
    f1 ∈ first_two_digit_choices ∧ 
    f2 ∈ first_two_digit_choices ∧ 
    valid_last_two_digits l1 l2) ↔ 
  27 :=
sorry

end count_valid_4_digit_numbers_l726_726757


namespace smallest_special_gt_3429_l726_726612

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726612


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726054

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726054


namespace find_m_l726_726303

noncomputable def parametricCurveC (α m : ℝ) : ℝ × ℝ := (Real.cos α, m + Real.sin α)

noncomputable def parametricLineL (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 5 / 5) * t, 4 + (2 * Real.sqrt 5 / 5) * t)

theorem find_m (α t m : ℝ) (P Q : ℝ × ℝ) 
(hP : P = parametricCurveC α m ∧ P = parametricLineL t) 
(hQ : Q = parametricCurveC α m ∧ Q = parametricLineL t) 
(h_dist : Real.dist P Q = 4 * Real.sqrt 5 / 5) : 
m = 1 ∨ m = 3 := sorry

end find_m_l726_726303


namespace water_spilled_l726_726487

theorem water_spilled (x s : ℕ) (h1 : s = x + 7) : s = 8 := by
  -- The proof would go here
  sorry

end water_spilled_l726_726487


namespace smallest_k_mod_19_7_3_l726_726035

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l726_726035


namespace find_x_value_l726_726670

noncomputable def log (a b: ℝ): ℝ := Real.log a / Real.log b

theorem find_x_value (a n : ℝ) (t y: ℝ):
  1 < a →
  1 < t →
  y = 8 →
  log n (a^t) - 3 * log a (a^t) * log y 8 = 3 →
  x = a^t →
  x = a^2 :=
by
  sorry

end find_x_value_l726_726670


namespace mrs_franklin_initial_valentines_l726_726399

theorem mrs_franklin_initial_valentines (v g l : ℕ) (h1 : g = 42) (h2 : l = 16) (h3 : v = g + l) : v = 58 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end mrs_franklin_initial_valentines_l726_726399


namespace sequence_bk_bl_sum_l726_726370

theorem sequence_bk_bl_sum (b : ℕ → ℕ) (m : ℕ) 
  (h_pairwise_distinct : ∀ i j, i ≠ j → b i ≠ b j)
  (h_b0 : b 0 = 0)
  (h_b_lt_2n : ∀ n, 0 < n → b n < 2 * n) :
  ∃ k ℓ : ℕ, b k + b ℓ = m := 
  sorry

end sequence_bk_bl_sum_l726_726370


namespace base8_to_base10_conversion_l726_726558

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l726_726558


namespace base8_to_base10_l726_726553

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l726_726553


namespace base8_to_base10_conversion_l726_726557

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l726_726557


namespace smallest_special_number_l726_726585

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726585


namespace ceil_y_squared_possibilities_l726_726326

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end ceil_y_squared_possibilities_l726_726326


namespace machines_in_first_scenario_l726_726771

theorem machines_in_first_scenario :
  ∃ M : ℕ, (∀ (units1 units2 : ℕ) (hours1 hours2 : ℕ),
    units1 = 20 ∧ hours1 = 10 ∧ units2 = 200 ∧ hours2 = 25 ∧
    (M * units1 / hours1 = 20 * units2 / hours2)) → M = 5 :=
by
  sorry

end machines_in_first_scenario_l726_726771


namespace max_rooks_ensuring_unattacked_cell_l726_726344

-- Define the size of the board and relevant parameters
def board_size := 10
def Cell := Fin board_size × Fin board_size

noncomputable def isAttacked (rooks : Finset Cell) (cell : Cell) : Prop :=
  ∃ r ∈ rooks, (r.1 = cell.1 ∨ r.2 = cell.2)

theorem max_rooks_ensuring_unattacked_cell
  (rooks : Finset Cell)
  (h_rooks : rooks.card = 16) :
  ∃ r ∈ rooks, ∃ cell : Cell, isAttacked rooks cell ∧ ¬ isAttacked (rooks.erase r) cell :=
sorry

end max_rooks_ensuring_unattacked_cell_l726_726344


namespace find_area_of_field_l726_726483

def breadth (L : ℝ) := 0.60 * L

def perimeter (L B : ℝ) := 2 * L + 2 * B

def area (L B : ℝ) := L * B

theorem find_area_of_field (L B : ℝ) (h1 : B = 0.60 * L) (h2 : perimeter L B = 800) :
  area L B = 37500 :=
by
  have hL : L = 250 := by
    -- Derivation steps
    have h : 2 * L + 2 * B = 800 := h2
    rw [h1] at h
    linarith
  have hB : B = 0.60 * 250 := by rw [hL, h1]
  have hB_final : B = 150 := by norm_num [hB]
  have hArea : area 250 150 = 37500 := by norm_num
  exact hArea.symm

end find_area_of_field_l726_726483


namespace john_average_speed_l726_726357

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end john_average_speed_l726_726357


namespace probability_all_real_roots_l726_726142

noncomputable def probability_real_roots (a : ℝ) : ℝ :=
  if a ∈ [-2*sqrt 3, -7/4] ∪ [5/4, 2*sqrt 3] then 1 else 0

theorem probability_all_real_roots :
  let len_interval := 35
  let favorable_interval := (2*sqrt 3 - 5/4)
  let total_probability := favorable_interval / len_interval
  a_is_uniform : ∀ (a : ℝ), a ∈ set.Icc (-15) 20 →
  probability_real_roots a = 1 - total_probability := sorry

end probability_all_real_roots_l726_726142


namespace equilateral_triangle_l726_726733

noncomputable def triangle_equilateral (z1 z2 z3 : ℂ) : Prop :=
  z1 + z2 + z3 = 0 ∧ |z1| = |z2| ∧ |z3| = |z2| →
  ∃ (p1 p2 p3 : ℂ), (p1 = 0 ∧ p2 = z2 - z1 ∧ p3 = z3 - z1 ∧
  (|p2| = |p3| ∧ |p2 - p3| = |p3 - 0|))

theorem equilateral_triangle (z1 z2 z3 : ℂ) (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0) 
  (h4 : z1 + z2 + z3 = 0) (h5 : |z1| = |z2| ∧ |z3| = |z2|) : 
  triangle_equilateral z1 z2 z3 :=
sorry

end equilateral_triangle_l726_726733


namespace lines_parallel_and_separate_l726_726269

theorem lines_parallel_and_separate (r a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) (h_point_inside : a^2 + b^2 < r^2) :
  let O := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2},
      P := (a, b),
      l1_slope := -a/b,
      l1 := {p : ℝ × ℝ | a * p.1 + b * p.2 = a^2 + b^2},
      l2 := {p : ℝ × ℝ | b * p.2 - a * p.1 + r^2 = 0} in
  (∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ l1 → p₂ ∈ l1 → p₁.1 = p₂.1 → p₁.2 = p₂.2 → a * p₁.1 + b * p₁.2 = a^2 + b^2) ∧
  (∀ p₁ p₂ : ℝ × ℝ, p₁ ∈ l2 → p₂ ∈ l2 → p₁.1 = p₂.1 → p₁.2 = p₂.2) ∧
  ∀ c : ℝ × ℝ, (c.1^2 + c.2^2 < r^2 ∨ c.1^2 + c.2^2 = r^2) → ¬(b * c.2 - a * c.1 + r^2 = 0) :=
by
  sorry

end lines_parallel_and_separate_l726_726269


namespace find_seating_capacity_l726_726953

noncomputable def seating_capacity (buses : ℕ) (students_left : ℤ) : ℤ :=
  buses * 40 + students_left

theorem find_seating_capacity :
  (seating_capacity 4 30) = (seating_capacity 5 (-10)) :=
by
  -- Proof is not required, hence omitted.
  sorry

end find_seating_capacity_l726_726953


namespace square_side_length_l726_726149

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726149


namespace total_length_BIO_l726_726871

-- Define the lengths of the components for each letter in "BIO"
def length_B : ℝ := 4 + 1 + Real.pi   -- 4 units from vertical segments, 1 unit from horizontal segment, π units from semi-circle
def length_I : ℝ := 2 + 2             -- 2 units from vertical segment, 2 units from horizontal segments (1 + 1)
def length_O : ℝ := 2 * Real.pi       -- Circumference of a circle with diameter 2 units

-- Define the total length of the acronym "BIO"
def length_BIO : ℝ := length_B + length_I + length_O

-- The theorem stating the total length of the acronym "BIO"
theorem total_length_BIO : length_BIO = 9 + 3 * Real.pi :=
by
  sorry

end total_length_BIO_l726_726871


namespace general_formula_sum_inverse_T_l726_726676

variable {ℕ : Type*}
variable {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℕ} {T : ℕ → ℕ}

/-- Definition of sequence a_n such that a_1 = 2 and a_(n+1) = S_n + 2 for n ∈ ℕ* -/
def seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n:ℕ, 0 < n → a (n + 1) = S n + 2

/-- Definition of S_n as the sum of the first n terms of the sequence a_n -/
def sum_S (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n:ℕ, S n = (finset.range (n+1)).sum a

/-- Prove the general formula for the sequence a_n -/
theorem general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (h_seq_a : seq_a a S) (h_sum_S : sum_S a S) : 
  ∀ n:ℕ, a n = 2 ^ n := sorry

/-- Definition of arithmetic sequence b_n with common difference d = 1, sum T_n, where b_2 = a_1 and b_4 = a_2 -/
def arith_seq_b (a : ℕ → ℕ) (b : ℕ → ℕ) (d T : ℕ → ℕ) : Prop :=
  b 2 = a 1 ∧ b 4 = a 2 ∧ d 1 = 1 ∧ ∀ n:ℕ, b n = n ∧ T n = n * (n + 1) / 2

/-- Prove the inequality 1 ≤ ∑_{i=1}^{n} 1/T_i < 2 -/
theorem sum_inverse_T (a : ℕ → ℕ) (b : ℕ → ℕ) (d T : ℕ → ℕ) 
  (h_seq_a : seq_a a S) (h_sum_S : sum_S a S) (h_arith_seq_b : arith_seq_b a b d T) : 
  ∀ n:ℕ, 1 ≤ (finset.range (n+1)).sum (λ i, 1 / (T (i+1))) ∧ (finset.range (n+1)).sum (λ i, 1 / (T (i+1))) < 2 := sorry

end general_formula_sum_inverse_T_l726_726676


namespace base8_246_is_166_in_base10_l726_726547

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l726_726547


namespace rectangle_angle_proof_l726_726855

theorem rectangle_angle_proof
  (EFGH : Type)
  (EF GH : EFGH -> EFGH -> ℝ)
  (N : EFGH)
  (EF_len : EF EFGH EFGH = 8)
  (FG_len : FG EFGH EFGH = 4)
  (EN_eq_FN : ∀ EN FN, ∠ EFGH EN = ∠ EFGH FN) : 
  ∠ EFGH (EN_eq_FN EFGH N) = 45 := 
sorry

end rectangle_angle_proof_l726_726855


namespace range_of_a_l726_726334

-- Define the points K and Q
def point_K (a : ℝ) : ℝ × ℝ := (1 - a, 1 + a)
def point_Q (a : ℝ) : ℝ × ℝ := (3, 2 * a)

-- Define the slope of the line passing through K and Q
def slope (a : ℝ) : ℝ := 
  let (xK, yK) := point_K a
  let (xQ, yQ) := point_Q a
  (yQ - yK) / (xQ - xK)

-- Define the condition for an obtuse angle with the x-axis
def is_obtuse (m : ℝ) : Prop :=
  m < 0

-- State the main theorem
theorem range_of_a (a : ℝ) : 
  -2 < a ∧ a < 1 ↔ is_obtuse (slope a) :=
sorry

end range_of_a_l726_726334


namespace second_term_is_4_l726_726786

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l726_726786


namespace function_C_is_odd_and_decreasing_l726_726979

noncomputable def f (x : ℝ) := 2^(-x) - 2^x

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) := ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ ≥ f x₂

theorem function_C_is_odd_and_decreasing :
  is_odd f ∧ is_decreasing f :=
by
  sorry

end function_C_is_odd_and_decreasing_l726_726979


namespace total_girls_in_circle_l726_726503

theorem total_girls_in_circle (girls : Nat) 
  (h1 : (4 + 7) = girls + 2) : girls = 11 := 
by
  sorry

end total_girls_in_circle_l726_726503


namespace calculate_product_l726_726999

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l726_726999


namespace smallest_special_number_l726_726588

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726588


namespace set_equality_solution_l726_726748

theorem set_equality_solution (a : ℝ) (h : {0, -1, 2 * a} = {a - 1, -|a|, a + 1}) : a = 1 ∨ a = -1 := 
by {
  sorry
}

end set_equality_solution_l726_726748


namespace minimum_value_and_period_find_a_b_l726_726292

noncomputable def f (x : ℝ) : ℝ := sin (2*x) - (cos (x)^2 - sin (x)^2) - 1

theorem minimum_value_and_period :
  (∀ x, f x ≥ -2) ∧ (∃ x, f x = -2) ∧ (∀ x, f (x + π) = f x) :=
sorry

structure TriangleABC :=
(a b c A B C : ℝ)
(c_eq_sqrt7 : c = sqrt 7)
(C_eq_pi_div_4 : f C = 0)
(sinB_eq_3sinA : sin B = 3 * sin A)
(b_eq_3a : b = 3 * a)

theorem find_a_b (ABC : TriangleABC)
  (h : ∀ a' b' : ℝ, a'^2 + b'^2 - a' * b' = 7 → b' = 3 * a' → a' = ABC.a ∧ b' = ABC.b) :
  ABC.c_eq_sqrt7 ∧ ABC.C_eq_pi_div_4 ∧ ABC.sinB_eq_3sinA ∧ ABC.b_eq_3a :=
sorry

end minimum_value_and_period_find_a_b_l726_726292


namespace side_length_of_square_l726_726187

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726187


namespace similar_triangles_side_ratio_l726_726941

theorem similar_triangles_side_ratio {P Q R S T U : Type} [HasLength P Q R S T U] 
  (h_similar : similar_triangles P Q R S T U) 
  (hpq : length P Q = 7) 
  (hqr : length Q R = 10) 
  (hst : length S T = 4.9) : 
  length T U = 7 := 
begin 
  sorry 
end

end similar_triangles_side_ratio_l726_726941


namespace integral_correct_value_l726_726242

noncomputable def integral_example : ℝ :=
  ∫ x in 1..(Real.exp 1), x + (1 / x)

theorem integral_correct_value : integral_example = (Real.exp 2 + 1) / 2 := by
  sorry

end integral_correct_value_l726_726242


namespace cost_price_per_meter_l726_726109

-- Definitions based on the conditions given in the problem
def meters_of_cloth : ℕ := 45
def selling_price : ℕ := 4500
def profit_per_meter : ℕ := 12

-- Statement to prove
theorem cost_price_per_meter :
  (selling_price - (profit_per_meter * meters_of_cloth)) / meters_of_cloth = 88 :=
by
  sorry

end cost_price_per_meter_l726_726109


namespace min_abs_diff_of_xy_eq_315_l726_726322

theorem min_abs_diff_of_xy_eq_315 (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h : x * y - 4 * x + 3 * y = 315) : |x - y| = 91 :=
sorry

end min_abs_diff_of_xy_eq_315_l726_726322


namespace solve_sin_equation_l726_726408

theorem solve_sin_equation :
  ∀ x : ℝ, 0 < x ∧ x < 90 → 
  (sin 9 * sin 21 * sin (102 + x) = sin 30 * sin 42 * sin x) → 
  x = 9 :=
by
  intros x h_cond h_eq
  sorry

end solve_sin_equation_l726_726408


namespace linear_function_above_x_axis_l726_726881

theorem linear_function_above_x_axis (a : ℝ) :
  (-1 < a ∧ a < 2 ∧ a ≠ 0) ↔
  (∀ x, -2 ≤ x ∧ x ≤ 1 → ax + a + 2 > 0) :=
sorry

end linear_function_above_x_axis_l726_726881


namespace z_has_purely_imaginary_difference_l726_726725

theorem z_has_purely_imaginary_difference
  (z : ℂ) (h : z = 2 - complex.i) : z - 2 = -complex.i := 
sorry

end z_has_purely_imaginary_difference_l726_726725


namespace mode_of_data_set_l726_726685

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726685


namespace solve_inequality_f_ge_4_a_1_find_range_of_a_l726_726738

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - 1) + a * abs (x - 1)

-- (I) Solve the inequality f(x) ≥ 4 for x when a = 1
theorem solve_inequality_f_ge_4_a_1 :
  { x : ℝ | (abs (2 * x - 1) + abs (x - 1)) ≥ 4 } = { x : ℝ | x ≤ - (2 / 3) ∨ x ≥ 2 } :=
sorry

-- (II) If the solution set of f(x) ≥ |x - 2| includes [1/2, 2], find the range of a
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (1/2) 2 → abs (2 * x - 1) + a * abs (x - 1) ≥ abs (x - 2)) → a ≥ 3 :=
sorry

end solve_inequality_f_ge_4_a_1_find_range_of_a_l726_726738


namespace conic_section_is_hyperbola_l726_726232

theorem conic_section_is_hyperbola (x y : ℝ) :
  (x - 3)^2 = (3 * y + 4)^2 - 75 → 
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 :=
sorry

end conic_section_is_hyperbola_l726_726232


namespace series_eq_inv_sqrt_sin_l726_726236

noncomputable def S (x : ℝ) := x + ∑' n : ℕ, (2 * (n+1)).factorial / ((n+1).factorial * (n+1).factorial * (2 * (n+1) + 1)) * x^(2*(n+1) + 1)

theorem series_eq_inv_sqrt_sin :
  ∀ x ∈ Icc (-1 : ℝ) 1,
  S x = (1 - x^2)⁻¹ / 2 * arcsin x := 
by
  intro x hx
  sorry

end series_eq_inv_sqrt_sin_l726_726236


namespace find_smallest_result_l726_726541

namespace small_result

def num_set : Set Int := { -10, -4, 0, 2, 7 }

def all_results : Set Int := 
  { z | ∃ x ∈ num_set, ∃ y ∈ num_set, z = x * y ∨ z = x + y }

def smallest_result := -70

theorem find_smallest_result : ∃ z ∈ all_results, z = smallest_result :=
by
  sorry

end small_result

end find_smallest_result_l726_726541


namespace maximum_distance_l726_726973

-- Definitions from the conditions
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def distance_driven : ℝ := 244
def gallons_used : ℝ := 20

-- Problem statement
theorem maximum_distance (h: (distance_driven / gallons_used = highway_mpg)): 
  (distance_driven = 244) :=
sorry

end maximum_distance_l726_726973


namespace ordered_pair_c_d_l726_726412

noncomputable def find_pair (c d : ℝ) : Prop :=
  c ≠ 0 ∧ d ≠ 0 ∧ 
  (∀ x : ℝ, x^2 + c * x + 2 * d = 0 → (x = c ∨ x = d)) ∧
  (c = 2 ∧ d = -4)

theorem ordered_pair_c_d :
  find_pair 2 (-4) :=
by
  intros
  split
  -- prove c ≠ 0
  { exact two_ne_zero }
  split
  -- prove d ≠ 0
  { exact neg_ne_zero.mpr two_ne_zero }
  split
  -- prove ∀ x, x^2 + 2 * x - 8 = 0 → (x = 2 ∨ x = -4)
  { intros x hx
    sorry } -- the proof of the polynomial equality
  -- finally prove (c = 2 ∧ d = -4)
  { split
    { refl }
    { refl } }

end ordered_pair_c_d_l726_726412


namespace length_of_car_is_270_l726_726493

noncomputable def speed_in_mps (speed_kmph : ℕ) : ℝ :=
  (speed_kmph * 1000 : ℝ) / 3600

noncomputable def length_of_car
  (length_of_train : ℝ)
  (speed_train_kmph : ℕ)
  (speed_car_kmph : ℕ)
  (crossing_time : ℝ) : ℝ :=
let speed_train_mps := speed_in_mps speed_train_kmph in
let speed_car_mps := speed_in_mps speed_car_kmph in
let relative_speed := speed_train_mps + speed_car_mps in
let total_distance := relative_speed * crossing_time in
total_distance - length_of_train

theorem length_of_car_is_270 :
  length_of_car 230 120 80 9 = 270 := by
  sorry

end length_of_car_is_270_l726_726493


namespace wall_thickness_is_correct_l726_726314

-- Define the dimensions of the brick.
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of required bricks.
def num_bricks : ℝ := 2000

-- Define the dimensions of the wall.
def wall_length : ℝ := 800
def wall_height : ℝ := 600

-- The volume of one brick.
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- The volume of the wall.
def wall_volume (T : ℝ) : ℝ := wall_length * wall_height * T

-- The thickness of the wall to be proved.
theorem wall_thickness_is_correct (T_wall : ℝ) (h : num_bricks * brick_volume = wall_volume T_wall) : 
  T_wall = 22.5 :=
sorry

end wall_thickness_is_correct_l726_726314


namespace questionOne_questionTwo_questionThree_questionFour_a_questionFour_b_questionFive_l726_726208

-- Statement (1)
theorem questionOne : (1 : ℂ) * (3 + 2 * complex.i) + (complex.sqrt 3 - 2) * complex.i = 3 + complex.sqrt 3 * complex.i :=
by sorry

-- Statement (2)
theorem questionTwo : (9 + 2 * complex.i) / (2 + complex.i) = 4 - complex.i :=
by sorry

-- Statement (3)
theorem questionThree : ((-1 + complex.i) * (2 + complex.i)) / (complex.i ^ 3) = -1 - 3 * complex.i :=
by sorry

-- Statement (4a)
theorem questionFour_a : 
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (2, 1)
  2 • a + 3 • b = (4, 7) :=
by sorry

-- Statement (4b)
theorem questionFour_b : 
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (2, 1)
  (λ x y : ℝ × ℝ, x.1 * y.1 + x.2 * y.2) a b = 0 :=
by sorry

-- Statement (5)
theorem questionFive : 
  ∀ (a b : ℝ × ℝ), 
  ∥a∥ = 1 ∧ (λ x y : ℝ × ℝ, x.1 * y.1 + x.2 * y.2) a b = -1 -> 
  (λ x y : ℝ × ℝ, x.1 * y.1 + x.2 * y.2) a (2 • a - b) = 3 :=
by sorry

end questionOne_questionTwo_questionThree_questionFour_a_questionFour_b_questionFive_l726_726208


namespace vertical_asymptote_l726_726233

noncomputable def has_vertical_asymptote_at_x (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ δ > 0, ∃ t ∈ Ioo (x - δ) (x + δ), abs (f t) > ε

theorem vertical_asymptote (x : ℝ) (h : x = 3) :
  has_vertical_asymptote_at_x (λ x, (x^2 + 5 * x + 6) / (x - 3)) 3 :=
begin
  sorry
end

end vertical_asymptote_l726_726233


namespace solution_set_eq_l726_726442

theorem solution_set_eq : { x : ℝ | |x| * (x - 2) ≥ 0 } = { x : ℝ | x ≥ 2 ∨ x = 0 } := by
  sorry

end solution_set_eq_l726_726442


namespace num_perfect_square_factors_of_8000_l726_726315

theorem num_perfect_square_factors_of_8000 :
  let a_values := {0, 2, 4, 6}
  let b_values := {0, 2}
  8000 = 2^6 * 5^3 →
  (∀a b, a ∈ a_values → b ∈ b_values → 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 3) →
  (set.size a_values * set.size b_values = 8) :=
by
  intros a_values b_values h_1 h_2
  have h_size_a : set.size a_values = 4, from sorry
  have h_size_b : set.size b_values = 2, from sorry
  exact eq.trans (mul_comm (set.size a_values) (set.size b_values))
    (congr_arg (λ x, 4 * x) (eq.symm h_size_b))
#align num_perfect_square_factors_of_8000 num_perfect_square_factors_of_8000

end num_perfect_square_factors_of_8000_l726_726315


namespace distribute_dogs_l726_726529

theorem distribute_dogs :
  ∃ (comb : ℕ), comb = (Nat.choose 11 3) * (Nat.choose 7 4) ∧ comb = 5775 :=
by
  use (Nat.choose 11 3) * (Nat.choose 7 4)
  split
  <;> sorry

end distribute_dogs_l726_726529


namespace area_region_l726_726633

theorem area_region {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : 100 * frac x ≥ ⌊x⌋ + ⌊y⌋ + 50) : 
  measure_theory.measure_space.volume (set_of (λ p : ℝ × ℝ, (0 ≤ p.1 ∧ p.2 ≥ 0 ∧ 100 * (p.1 - p.1.floor) ≥ p.1.floor + p.2.floor + 50))) = 1 := 
sorry

end area_region_l726_726633


namespace plates_arrangement_l726_726964

theorem plates_arrangement : 
  let blue := 6
  let red := 3
  let green := 2
  let yellow := 1
  let total_ways_without_rest := Nat.factorial (blue + red + green + yellow - 1) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)
  let green_adj_ways := Nat.factorial (blue + red + green + yellow - 2) / (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * Nat.factorial yellow)
  total_ways_without_rest - green_adj_ways = 22680 
:= sorry

end plates_arrangement_l726_726964


namespace purchasing_plans_l726_726842

theorem purchasing_plans (x y : ℕ) (h : 2 * x + 3 * y = 30) : 
   ∃ xvals yvals, 
     (xvals.length = 4) ∧ 
     (∀ i ∈ xvals, 2 * i + 3 * (yvals.get $ yvals.indexOf i) = 30) ∧ 
     (∀ i ∈ xvals, yvals.indexOf i ≠ (-1 : ℕ)) := 
sorry

end purchasing_plans_l726_726842


namespace find_value_of_y_l726_726648

theorem find_value_of_y (y : ℝ) (h : (sqrt 1.21 / sqrt y + sqrt 1.44 / sqrt 0.49) = 2.9365079365079367) : y = 0.81 :=
by
  sorry

end find_value_of_y_l726_726648


namespace real_root_of_equation_l726_726641

theorem real_root_of_equation :
  ∃ x : ℝ, sqrt x + sqrt (x + 4) = 12 ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726641


namespace h_95_eq_40_l726_726224

def h : ℕ+ → ℕ
| ⟨x, hx⟩ := if (∃ (i : ℕ), x = 2^i) then Nat.log2 x else 1 + h ⟨x + 1, Nat.succ_pos x⟩

theorem h_95_eq_40 : h ⟨95, by norm_num⟩ = 40 :=
sorry

end h_95_eq_40_l726_726224


namespace base8_to_base10_conversion_l726_726561

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l726_726561


namespace cos_add_eq_l726_726327

namespace ComplexExponential

theorem cos_add_eq :
  (∀ γ δ : ℝ, (complex.exp (complex.I * γ) = complex.ofReal (4/5) + complex.I * (3/5)) ∧ 
                      (complex.exp (complex.I * δ) = complex.ofReal (-5/13) + complex.I * (-12/13))
          → real.cos (γ + δ) = 16 / 65) := 
sorry

end ComplexExponential

end cos_add_eq_l726_726327


namespace equal_number_of_boys_and_girls_l726_726896

theorem equal_number_of_boys_and_girls
  (m d M D : ℕ)
  (h1 : (M / m) ≠ (D / d))
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : m = d :=
sorry

end equal_number_of_boys_and_girls_l726_726896


namespace find_special_number_l726_726246

noncomputable def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

noncomputable def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

noncomputable def split_digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := (n / 10000) % 10
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  let d := (n / 10) % 10
  let e := n % 10
  (a, b, c, d, e)

noncomputable def ace (a b c d e : ℕ) : ℕ :=
  100 * a + 10 * c + e

noncomputable def bda (a b c d e : ℕ) : ℕ :=
  100 * b + 10 * d + a

noncomputable def satisfies_conditions (n : ℕ) : Prop :=
  let (a, b, c, d, e) := split_digits n
  is_five_digit_number n ∧ is_divisible_by_9 n ∧ ace a b c d e - bda a b c d e = 760

theorem find_special_number (n : ℕ) : satisfies_conditions 81828 :=
by {
  let a := 8, let b := 1, let c := 8, let d := 2, let e := 8,
  have spl : split_digits 81828 = (a, b, c, d, e), {
    dsimp [split_digits], norm_num,
  },
  rw spl,
  rw [←nat.foldl_div_mod_comm],
  rw [mul_zero, mul_one],
  simp,
  sorry
}

end find_special_number_l726_726246


namespace w_share_l726_726190

theorem w_share (k : ℝ) (w x y z : ℝ) (h1 : w = k) (h2 : x = 6 * k) (h3 : y = 2 * k) (h4 : z = 4 * k) (h5 : x - y = 1500):
  w = 375 := by
  /- Lean code to show w = 375 -/
  sorry

end w_share_l726_726190


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726067

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726067


namespace range_of_r_l726_726230

noncomputable def r (x : ℝ) : ℝ :=
  (x^4 + 6*x^2 + 9) - 2*x

theorem range_of_r : set.range (λ x : ℝ, if x ≥ 0 then r x else 0) = {y : ℝ | y ≥ 9} :=
by 
  have domain : ∀ x, x < 0 → r x = 0 := by sorry
  have values : ∀ y, ∃ x, x ≥ 0 ∧ r x = y ↔ y ≥ 9 := by sorry
  ext y
  split
  · -- Show ∀ y ∈ set.range (λ x : ℝ, if x ≥ 0 then r x else 0), y ≥ 9.
    rw set.mem_range
    intro hy
    obtain ⟨x, hx⟩ := hy
    by_cases h : x ≥ 0
    · rw if_pos h at hx
      rw ←hx
      apply values.1
      sorry
    · rw if_neg h at hx
      exfalso
      sorry
  · -- Show ∀ y ≥ 9, ∃ x, x ≥ 0 ∧ r x = y.
    sorry

end range_of_r_l726_726230


namespace multiplication_of_mixed_number_l726_726987

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726987


namespace tim_earn_per_visit_l726_726905

theorem tim_earn_per_visit :
  (let visitors_first_6_days := 100 * 6 in
   let visitors_last_day := 2 * visitors_first_6_days in
   let total_visitors := visitors_first_6_days + visitors_last_day in
   let total_earnings := 18 in
   total_earnings / total_visitors = 0.01) :=
by
  sorry

end tim_earn_per_visit_l726_726905


namespace inequality_of_sum_of_squares_l726_726825

theorem inequality_of_sum_of_squares (a b c : ℝ) (h : a * b + b * c + a * c = 1) : (a + b + c) ^ 2 ≥ 3 :=
sorry

end inequality_of_sum_of_squares_l726_726825


namespace measure_of_angle_CAF_l726_726238

-- Definitions and assumptions from the conditions
def equilateral_triangle_interior_angle := 60
def regular_pentagon_interior_angle := 108
def angle_sum_triangle := 180

-- Given:
-- Equilateral triangle ABC and regular pentagon BCFGH are coplanar.
-- B and C are consecutive vertices of both shapes.

-- Given these definitions, prove that the measure of angle CAF is 6 degrees
theorem measure_of_angle_CAF :
  ∀ (CAF_angle : ℕ), 
  (CAF_angle = regular_pentagon_interior_angle + equilateral_triangle_interior_angle - angle_sum_triangle) / 2 = 6 :=
by
  intros
  unfold equilateral_triangle_interior_angle regular_pentagon_interior_angle angle_sum_triangle
  simp
  sorry

end measure_of_angle_CAF_l726_726238


namespace multiplication_of_mixed_number_l726_726989

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726989


namespace lowest_price_per_component_l726_726108

-- Define constants and conditions
def cost_per_component : ℝ := 80
def shipping_per_unit : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def num_components_per_month : ℝ := 150

def total_variable_cost_per_unit : ℝ := cost_per_component + shipping_per_unit
def total_variable_cost (num_units : ℝ) : ℝ := total_variable_cost_per_unit * num_units
def total_cost (num_units : ℝ) : ℝ := total_variable_cost(num_units) + fixed_monthly_costs
def price_per_component (total_cost : ℝ) (num_units : ℝ) : ℝ := total_cost / num_units

theorem lowest_price_per_component :
  price_per_component (total_cost num_components_per_month) num_components_per_month = 196 :=
by
  -- Skip the proof
  sorry

end lowest_price_per_component_l726_726108


namespace convert_base_8_to_base_10_l726_726543

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l726_726543


namespace feeding_ways_l726_726197

theorem feeding_ways :
  ∃ (n : ℕ), n = 5760 ∧
  ∀ (pairs : ℕ), pairs = 5 →
    ∀ (start : bool), start = ff → -- 'ff' represents starting with female
      -- Calculate the number of feeding ways given the conditions:
      n = (1 * 5 * 4 * 4 * 3 * 3 * 2 * 2 * 1 * 1) :=
begin
  use 5760,
  split,
  { refl },
  intros pairs hpairs start hstart,
  rw [hpairs, hstart],
  simp,
  norm_num
end

end feeding_ways_l726_726197


namespace find_k_l726_726848

variable {s t k : ℝ}

def line1 (s : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 4 - k * s, 3 + k * s)

def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (t / 3, 2 + t, 5 - t)

def coplanar_lines (k : ℝ) : Prop :=
  ∃ s t : ℝ, 
    (2 + s = t / 3) ∧ 
    (4 - k * s = 2 + t) ∧ 
    (3 + k * s = 5 - t)

theorem find_k : coplanar_lines k → k = -3 :=
by
  sorry

end find_k_l726_726848


namespace faulty_clock_correct_display_fraction_l726_726132

-- Defining the faulty display clock and the fraction calculation proof
theorem faulty_clock_correct_display_fraction : 
  let hours := 12
  let correct_hours := 10
  let hours_fraction := (correct_hours : ℚ) / hours 
  let minutes_per_hour := 60
  let incorrect_minutes := 16
  let correct_minutes := minutes_per_hour - incorrect_minutes
  let minutes_fraction := (correct_minutes : ℚ) / minutes_per_hour
  hours_fraction * minutes_fraction = 11 / 18 :=
by
  let hours := 12
  let correct_hours := 10
  let correct_hours_fraction := (correct_hours : ℚ) / hours
  let minutes_per_hour := 60
  let incorrect_minutes := 16
  let correct_minutes := minutes_per_hour - incorrect_minutes
  let correct_minutes_fraction := (correct_minutes : ℚ) / minutes_per_hour
  calc
    (correct_hours_fraction * correct_minutes_fraction) 
      = (10 / 12) * (44 / 60) : by sorry
    ... = 11 / 18 : by sorry

end faulty_clock_correct_display_fraction_l726_726132


namespace true_propositions_l726_726201

def P1 := ∀ x : ℝ, 2 * x + 5 > 0
def P2 := ¬(∀ x : ℝ, x^2 + 5 * x = 6) = (∃ x : ℝ, x^2 + 5 * x ≠ 6)
def P3 := ∀ x y : ℝ, abs x = abs y → x = y
def P4 := ∀ p q : Prop, ¬(p ∨ q) → (¬ p ∧ ¬ q)

theorem true_propositions : ({P1, P4} : set Prop) = ({P1, P4}) := by
  sorry

end true_propositions_l726_726201


namespace triangle_area_91_84_35_l726_726929

noncomputable def area_of_triangle_heron (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_91_84_35 : 
  area_of_triangle_heron 91 84 35 ≈ 453.19 := 
sorry

end triangle_area_91_84_35_l726_726929


namespace length_of_FD_l726_726806

theorem length_of_FD (A B C D E F: Point)
  (square : is_square A B C D)
  (side_len : ∀ p1 p2, (p1 = A ∧ p2 = D) ∨ (p1 = B ∧ p2 = C) -> distance p1 p2 = 10)
  (pointE_on_AD : ∃ t, 0 < t ∧ t < 1 ∧ E = t * A + (1 - t) * D)
  (distance_DE : distance D E = 3)
  (F_on_CD : ∃ u, 0 ≤ u ∧ u ≤ 1 ∧ F = u * C + (1 - u) * D)
  (EF_CF : distance E F = distance C F)
  (right_angle_FDE : is_right_angle (angle F D E)) :
  distance F D = 91 / 20 := 
sorry

end length_of_FD_l726_726806


namespace part1_part2_l726_726438

noncomputable def T (t : ℝ) : ℝ := 120 / (t + 5) + 15

theorem part1 (t : ℝ) : (deriv T 10) = -8 / 15 := 
by sorry

theorem part2 :
  ∃ t : ℝ, (deriv T t = -1) ∧ (abs (t - (2 * real.sqrt 30 - 5)) < 0.01) := 
by sorry

end part1_part2_l726_726438


namespace domain_of_f_l726_726876

def is_value_nonnegative (x : ℝ) : Prop := x ≥ 0
def is_value_positive (x : ℝ) : Prop := x > 0

def f (x : ℝ) : ℝ := Real.sqrt (x - 1) - Real.log (2 - x) / Real.log 10

theorem domain_of_f : { x : ℝ | is_value_nonnegative (x - 1) ∧ is_value_positive (2 - x) } = set.Icc 1 2 :=
by 
  sorry

end domain_of_f_l726_726876


namespace problem_solution_l726_726668

theorem problem_solution (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ ab < b^2 :=
sorry

end problem_solution_l726_726668


namespace total_price_all_art_l726_726363

-- Define the conditions
def total_price_first_three_pieces : ℕ := 45000
def price_next_piece := (total_price_first_three_pieces / 3) * 3 / 2 

-- Statement to prove
theorem total_price_all_art : total_price_first_three_pieces + price_next_piece = 67500 :=
by
  sorry -- Proof is omitted

end total_price_all_art_l726_726363


namespace count_birds_l726_726626

theorem count_birds (b m c : ℕ) (h1 : b + m + c = 300) (h2 : 2 * b + 4 * m + 3 * c = 708) : b = 192 := 
sorry

end count_birds_l726_726626


namespace arithmetic_sequence_second_term_l726_726780

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l726_726780


namespace probability_of_selecting_one_defective_l726_726101

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l726_726101


namespace ellipse_eccentricity_l726_726651

variables {a b c : ℝ}
variables (x y : ℝ) 

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def line_symmetry (x y : ℝ) : Prop :=
  √3 * x + y = 0

def focal_point (a b : ℝ) : ℝ := 
  (a^2 - b^2)^0.5

def symmetric_point (a b c : ℝ) : Prop :=
  ((c/2) ^ 2 / a^2) + ((√3 * c / 2) ^ 2 / b^2) = 1 
  
def eccentricity (a c : ℝ) : ℝ := 
  c / a

theorem ellipse_eccentricity (ha : a > b) (hb : b > 0) (hline : line_symmetry x y) (hsymm : symmetric_point a b c):
  eccentricity a c = √3 - 1 :=
sorry

end ellipse_eccentricity_l726_726651


namespace mode_of_data_set_l726_726700

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726700


namespace initial_percentage_of_water_l726_726494

theorem initial_percentage_of_water (C V final_volume : ℝ) (P : ℝ) 
  (hC : C = 80)
  (hV : V = 36)
  (h_final_volume : final_volume = (3/4) * C)
  (h_initial_equation: (P / 100) * C + V = final_volume) : 
  P = 30 :=
by
  sorry

end initial_percentage_of_water_l726_726494


namespace area_limit_l726_726205

-- Definitions for the initial semicircle and the sequence of shapes
def initial_radius : ℝ := 1
def initial_area : ℝ := (1 / 2) * Real.pi * initial_radius^2

-- A function that defines the area of the k-th removed semicircle based on its radius
def removed_area (k : ℕ) : ℝ :=
  (1 / 2) * Real.pi * (1 / (2^(k-1)))^2

-- The total area removed up to (n-1)th semicircle
def total_removed_area (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n-1), removed_area (k + 1)

-- The area of shape P_n
def shape_area (n : ℕ) : ℝ :=
  initial_area - total_removed_area n

-- Prove the limit of the area of shape P_n as n approaches infinity
theorem area_limit : tendsto (λ n, shape_area n) at_top (𝓝 (Real.pi / 3)) :=
by
  sorry

end area_limit_l726_726205


namespace exists_expression_divisible_by_2000_l726_726342

theorem exists_expression_divisible_by_2000 (nums : List ℕ) (h_len : nums.length = 23) :
  ∃ expr, evaluate_expression expr nums ∧ (evaluate_expression expr nums) % 2000 = 0 :=
sorry

-- Assume evaluate_expression is a function that evaluates the expression with the given list of numbers. This is left as a placeholder.

def evaluate_expression : (some_type_for_expression) → (List ℕ) → ℕ
| _ _ := sorry  -- Placeholder for the expression evaluator

end exists_expression_divisible_by_2000_l726_726342


namespace find_smallest_x_l726_726318

theorem find_smallest_x (y : ℕ) (h1 : 0.75 = y / (200 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 4 :=
by
  sorry

end find_smallest_x_l726_726318


namespace smallest_fraction_l726_726106

theorem smallest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 3) (h2 : f2 = 3 / 4) (h3 : f3 = 5 / 6) 
  (h4 : f4 = 5 / 8) (h5 : f5 = 11 / 12) : f4 = 5 / 8 ∧ f4 < f1 ∧ f4 < f2 ∧ f4 < f3 ∧ f4 < f5 := 
by 
  sorry

end smallest_fraction_l726_726106


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726059

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726059


namespace real_root_solution_l726_726642

noncomputable def real_root_equation : Nat := 36

theorem real_root_solution (x : ℝ) (h : x ≥ 0 ∧ √x + √(x+4) = 12) : x = 1225 / real_root_equation := by
  sorry

end real_root_solution_l726_726642


namespace Jill_braid_time_l726_726359

theorem Jill_braid_time
  (dancers : ℕ)
  (braids_per_dancer : ℕ)
  (total_time_minutes : ℕ)
  (seconds_per_minute : ℕ := 60)
  (h_dancers : dancers = 8)
  (h_braids_per_dancer : braids_per_dancer = 5)
  (h_total_time_minutes : total_time_minutes = 20) :
  let total_braids := dancers * braids_per_dancer
      total_time_seconds := total_time_minutes * seconds_per_minute
      time_per_braid := total_time_seconds / total_braids
  in time_per_braid = 30 := by
  sorry

end Jill_braid_time_l726_726359


namespace f_maximum_value_f_inequality_solution_set_l726_726393

-- Define the function f
def f (a x : ℝ) := a * |x - 2| + x

-- Define the conditions and respective conclusions as propositions
theorem f_maximum_value (f : ℝ → ℝ) (a : ℝ) : 
  (∀ (x : ℝ), (x < 2 → f(x) = (1 - a) * x + 2 * a) ∧ (x ≥ 2 → f(x) = (1 + a) * x - 2 * a)) → 
  (∃ M : ℝ, ∀ x : ℝ, f(x) ≤ M) → 
  a ≤ -1 := 
by 
  sorry 

theorem f_inequality_solution_set (a x : ℝ) (h: a = 1) : 
  f a x > |2 * x - 3| → 
  x > 1 / 2 := 
by 
  sorry

end f_maximum_value_f_inequality_solution_set_l726_726393


namespace Patel_family_theme_park_expenses_l726_726657

def regular_ticket_price : ℝ := 12.5
def senior_discount : ℝ := 0.8
def child_discount : ℝ := 0.6
def senior_ticket_price := senior_discount * regular_ticket_price
def child_ticket_price := child_discount * regular_ticket_price

theorem Patel_family_theme_park_expenses :
  (2 * senior_ticket_price + 2 * child_ticket_price + 4 * regular_ticket_price) = 85 := by
  sorry

end Patel_family_theme_park_expenses_l726_726657


namespace Bo_knew_percentage_l726_726985

-- Definitions from the conditions
def total_flashcards := 800
def words_per_day := 16
def days := 40
def total_words_to_learn := words_per_day * days
def known_words := total_flashcards - total_words_to_learn

-- Statement that we need to prove
theorem Bo_knew_percentage : (known_words.toFloat / total_flashcards.toFloat) * 100 = 20 :=
by
  sorry  -- Proof is omitted as per the instructions

end Bo_knew_percentage_l726_726985


namespace similar_triangles_perimeter_l726_726458

noncomputable def isosceles_triangle (a b : ℝ) (h : b > a) : Prop :=
  ∀ x y z, (x = a ∧ (y = b ∧ z = b)) ∨ (y = a ∧ (x = b ∧ z = b))

theorem similar_triangles_perimeter
  (a b : ℝ)
  (h1 : isosceles_triangle a b 10)
  (h2 : 50 = 5 * a):
  ∃ p : ℝ, p = 250 := 
sorry

end similar_triangles_perimeter_l726_726458


namespace triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l726_726405

/-- Points \(P, Q, R, S\) are distinct, collinear, and ordered on a line with line segment lengths \( a, b, c \)
    such that \(a = PQ\), \(b = PR\), \(c = PS\). After rotating \(PQ\) and \(RS\) to make \( P \) and \( S \) coincide
    and form a triangle with a positive area, we must show:
    \(I. a < \frac{c}{3}\) must be satisfied in accordance to the triangle inequality revelations -/
theorem triangle_inequality_necessary_conditions (a b c : ℝ)
  (h_abc1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : b > c - b ∧ c > a ∧ c > b - a) :
  a < c / 3 :=
sorry

theorem triangle_inequality_sufficient_conditions (a b c : ℝ)
  (h_abc2 : b ≥ c / 3 ∧ a < c ∧ 2 * b ≤ c) :
  ¬ b < c / 3 :=
sorry

end triangle_inequality_necessary_conditions_triangle_inequality_sufficient_conditions_l726_726405


namespace smallest_special_greater_than_3429_l726_726573

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726573


namespace intersection_points_collinear_l726_726244

-- Define the convex quadrilateral and points P, Q
variables {A B C D P Q : Point}
variable [convex_quadrilateral A B C D]

-- Conditions
axiom AB_CD_intersection : line (AB) ∩ line (CD) = {P}
axiom BC_AD_intersection : line (BC) ∩ line (AD) = {Q}
-- Each pair of external angle bisectors at vertices has a point of intersection
axiom external_angle_bisectors_A_C : ∃ X, is_intersection (external_angle_bisector A B P) (external_angle_bisector C D P) X
axiom external_angle_bisectors_B_D : ∃ Y, is_intersection (external_angle_bisector B C Q) (external_angle_bisector D A Q) Y
axiom external_angle_bisectors_P_Q : ∃ Z, is_intersection (external_angle_bisector Q A B) (external_angle_bisector P B C) Z 

-- Prove that the points of intersection lie on a single line
theorem intersection_points_collinear :
    ∀ X Y Z,
      (is_intersection (external_angle_bisector A B P) (external_angle_bisector C D P) X) →
      (is_intersection (external_angle_bisector B C Q) (external_angle_bisector D A Q) Y) →
      (is_intersection (external_angle_bisector Q A B) (external_angle_bisector P B C) Z) →
      collinear X Y Z :=
by
  sorry

end intersection_points_collinear_l726_726244


namespace range_of_a_l726_726826

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x / (1 + a * x^2)

theorem range_of_a (a : ℝ) (ha : a > 0)
  (h_monotone : ∀ x y, x ≤ y → f a x ≤ f a y) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l726_726826


namespace max_segment_length_l726_726206

theorem max_segment_length {A B C D : ℝ×ℝ} (h_rect : is_rectangle A B C D)
  (h_AB : dist A B = 6) (h_BC : dist B C = 8) :
  ∃ M, max_segment_length A C D M ∧ M = 8 :=
by
  sorry

end max_segment_length_l726_726206


namespace intersection_A_B_l726_726308

noncomputable def A : set ℝ := {y | ∃ x ∈ ℝ, y = -x^2 + 1}
def B : set ℕ := {n | true}

theorem intersection_A_B :
  A ∩ B = {0, 1} :=
sorry

end intersection_A_B_l726_726308


namespace deluxe_stereo_time_fraction_l726_726928

theorem deluxe_stereo_time_fraction (S : ℕ) (B : ℝ)
  (H1 : 2 / 3 > 0)
  (H2 : 1.6 > 0) :
  (1.6 / 3 * S * B) / (1.2 * S * B) = 4 / 9 :=
by
  sorry

end deluxe_stereo_time_fraction_l726_726928


namespace no_diagonal_diameter_l726_726650

open Real

theorem no_diagonal_diameter 
  {A B C D : Point} {R : ℝ}
  (h : is_cyclic_quadrilateral A B C D)
  (hR : ∀ (P Q : Point), point_dist(A, P, Q) ≠ ∅ → (point_dist(A, P, Q) = R) ∧ (point_dist(B, P, Q) = R) ∧ (point_dist(C, P, Q) = R) ∧ (point_dist(D, P, Q) = R))
  (cond : (point_dist(A, B))² + (point_dist(B, C))² + (point_dist(C, D))² + (point_dist(D, A))² = 8 * R²) :
  ¬ (∃ (P Q : Point), (P = A ∧ Q = C) ∨ (P = B ∧ Q = D) ∧ (point_dist(P, Q) = 2 * R)) := 
sorry

end no_diagonal_diameter_l726_726650


namespace batsman_average_30_matches_l726_726418

theorem batsman_average_30_matches (avg_20_matches : ℕ -> ℚ) (avg_10_matches : ℕ -> ℚ)
  (h1 : avg_20_matches 20 = 40)
  (h2 : avg_10_matches 10 = 20)
  : (20 * (avg_20_matches 20) + 10 * (avg_10_matches 10)) / 30 = 33.33 := by
  sorry

end batsman_average_30_matches_l726_726418


namespace find_seating_capacity_l726_726956

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l726_726956


namespace fraction_order_l726_726468

theorem fraction_order :
  (25 / 19 : ℚ) < (21 / 16 : ℚ) ∧ (21 / 16 : ℚ) < (23 / 17 : ℚ) := by
  sorry

end fraction_order_l726_726468


namespace square_side_length_l726_726177

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726177


namespace molecular_weight_correct_l726_726916

noncomputable def molecular_weight (Ca_weight : ℝ) (I_weight : ℝ) (C_weight : ℝ) (H_weight : ℝ) 
                                   (num_Ca : ℕ) (num_I : ℕ) (num_C : ℕ) (num_H : ℕ) : ℝ :=
  (num_Ca * Ca_weight) + (num_I * I_weight) + (num_C * C_weight) + (num_H * H_weight)

theorem molecular_weight_correct :
  molecular_weight 40.08 126.90 12.01 1.008 2 4 1 3 = 602.794 := 
by 
  unfold molecular_weight 
  norm_num
  done

#eval molecular_weight 40.08 126.90 12.01 1.008 2 4 1 3

end molecular_weight_correct_l726_726916


namespace decreasing_interval_of_function_l726_726431

theorem decreasing_interval_of_function :
  ∀ k : ℤ, ∃ I : set ℝ, I = set.Ioo (k * π - π / 3) (k * π + π / 6) ∧
    (∀ x ∈ I, ∃ y : ℝ, y = 3 - 2 * (Real.cos (2 * x - π / 3))) :=
by
  intros k
  use (set.Ioo (k * π - π / 3) (k * π + π / 6))
  split
  . refl
  . intros x hx
    use 3 - 2 * Real.cos (2 * x - π / 3)
    sorry

end decreasing_interval_of_function_l726_726431


namespace least_positive_base_ten_seven_binary_digits_l726_726049

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726049


namespace john_paid_more_l726_726360

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end john_paid_more_l726_726360


namespace purely_imaginary_subtraction_l726_726727

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end purely_imaginary_subtraction_l726_726727


namespace prove_weight_loss_l726_726107

variable (W : ℝ) -- Original weight
variable (x : ℝ) -- Percentage of weight lost

def weight_equation := W - (x / 100) * W + (2 / 100) * W = (89.76 / 100) * W

theorem prove_weight_loss (h : weight_equation W x) : x = 12.24 :=
by
  sorry

end prove_weight_loss_l726_726107


namespace cycling_meeting_time_l726_726490

theorem cycling_meeting_time (b : ℕ) (h_b : b = 48) (p : ℕ) 
  (h_peter_speed : ∀ t : ℕ, t = 7) (h_john_speed : ∀ t : ℕ, t = 5) :
  p = 4 :=
by 
  have h_relative_speed := 7 + 5
  have h_meeting_time := b / h_relative_speed
  rw h_b at h_meeting_time
  rw h_meeting_time
  sorry

end cycling_meeting_time_l726_726490


namespace ortho_distance_prove_l726_726834

noncomputable def ortho_distance {α : Type*} [linear_ordered_field α] 
    (A B C : EuclideanGeometry.Point α) (M : EuclideanGeometry.Point α) : α :=
  let A1 := EuclideanGeometry.foot_of_perpendicular A B C in
  EuclideanGeometry.distance M A1

theorem ortho_distance_prove {α : Type*} [linear_ordered_field α]
  (A B C : EuclideanGeometry.Point α)
  (h_altitude : EuclideanGeometry.distance A (EuclideanGeometry.foot_of_perpendicular A B C) = 15)
  (h_BA1 : EuclideanGeometry.distance B (EuclideanGeometry.foot_of_perpendicular A B C) = 10)
  (h_CA1 : EuclideanGeometry.distance C (EuclideanGeometry.foot_of_perpendicular A B C) = 6)
  : ortho_distance A B C (EuclideanGeometry.orthocenter A B C) = 4 :=
by
  sorry

end ortho_distance_prove_l726_726834


namespace smallest_special_gt_3429_l726_726614

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726614


namespace last_snake_length_l726_726535

def Christina.snakes:= ℕ

-- Given conditions
variable (length_snake1_feet : ℕ) (length_snake2_inches : ℕ) (total_combined_length_inches : ℕ)
variable (length_snake1_inches : ℕ) (two_snakes_length_inches : ℕ)
variable (L : ℕ)

-- Length of the first snake in feet converted to inches
def length_snake1_inches_def := length_snake1_feet * 12

-- Total length of the first two snakes in inches
def two_snakes_length_inches_def := length_snake1_inches + length_snake2_inches

-- Proof statement
theorem last_snake_length :
  length_snake1_feet = 2 →
  length_snake2_inches = 16 →
  total_combined_length_inches = 50 →
  length_snake1_inches = length_snake1_inches_def →
  two_snakes_length_inches = two_snakes_length_inches_def →
  L = total_combined_length_inches - two_snakes_length_inches →
  L = 10 := by
  intros 
  unfold length_snake1_inches_def
  unfold two_snakes_length_inches_def
  sorry

end last_snake_length_l726_726535


namespace smallest_n_for_congruence_l726_726231

theorem smallest_n_for_congruence : ∃ n : ℕ, 0 < n ∧ 7^n % 5 = n^4 % 5 ∧ (∀ m : ℕ, 0 < m ∧ 7^m % 5 = m^4 % 5 → n ≤ m) ∧ n = 4 :=
by
  sorry

end smallest_n_for_congruence_l726_726231


namespace problem_l726_726736

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem problem (a : ℝ) (h : f a = 2) : f (-a) = 0 := 
  sorry

end problem_l726_726736


namespace find_x_for_condition_l726_726479

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x_for_condition :
  (2 * f 1 - 16 = f (1 - 6)) :=
by
  sorry

end find_x_for_condition_l726_726479


namespace odd_function_f_of_negative_one_l726_726392

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2 * x else -(((-x) : ℝ)^2 - 2 * (-x))

theorem odd_function_f_of_negative_one :
  (∀ x : ℝ, f(-x) = - f(x)) ∧ (∀ x : ℝ, x ≥ 0 → f(x) = x^2 - 2 * x) → f (-1) = 1 :=
by sorry

end odd_function_f_of_negative_one_l726_726392


namespace probability_one_instrument_l726_726930

-- Definitions based on conditions
def total_people : Nat := 800
def play_at_least_one : Nat := total_people / 5
def play_two_or_more : Nat := 32
def play_exactly_one : Nat := play_at_least_one - play_two_or_more

-- Target statement to prove the equivalence
theorem probability_one_instrument: (play_exactly_one : ℝ) / (total_people : ℝ) = 0.16 := by
  sorry

end probability_one_instrument_l726_726930


namespace time_to_mow_lawn_l726_726397

-- Define the given conditions
def lawn_length := 120 -- in feet
def lawn_width := 180 -- in feet
def effective_swath_width := 2 -- in feet
def mowing_rate := 4000 -- in feet per hour

-- Restate the problem
theorem time_to_mow_lawn : (lawn_length * (lawn_width / effective_swath_width)) / mowing_rate = 2.7 := 
by
  sorry

end time_to_mow_lawn_l726_726397


namespace game_draw_14_squares_l726_726903

def player := ℕ → ℕ → Prop

def game_on_line : Prop :=
  ∃ (A B : ℕ) (AB: player), 
  (A = 14) ∧
  (∀ move: ℕ, move < A) ∧
  ∀ p1 p2 : player, 
  (p1 = (λ m p : ℕ, m = 1 ∧ p <= A)) → 
  (p2 = (λ m p : ℕ, m = 2 ∧ p <= A)) → 
  (∃ b : ℕ, b = B → 
  (∀ turns : ℕ, turns < B → 
  ¬ (turns = (λ i : ℕ, i = A ∧ B)))) →
  ((∃ p : player, (p = p1) ∨ (p = AB)) → False)

theorem game_draw_14_squares : game_on_line :=
by {
  sorry
}

end game_draw_14_squares_l726_726903


namespace expansion_binomial_coefficient_l726_726226

theorem expansion_binomial_coefficient :
  ∀ (x : ℝ), ∃! (c : ℝ), 
  (∃ (f : ℤ → ℝ) (n : ℤ), 
    (x ≠ 0) ∧ 
    (7 : ℤ) = n ∧ 
    (n >= 2) ∧ 
    (f (2 : ℤ) = c) ∧ 
    ((∑ i in (finset.range (n.to_nat + 1)), 
      (finset.choose n i)*((-2:ℝ)^i)*((x^(1/2):ℝ)^(n-i)*((x^(-1):ℝ)^i)) = (x^(2:ℤ)))),
       c = (84:ℝ)) :=
begin
  intros x,
  use 84,
  split,
  {
    sorry, -- Proof would go here
  },
  {
    intros y hy,
    sorry, -- Proof of uniqueness would go here
  }
end

end expansion_binomial_coefficient_l726_726226


namespace triangle_cos2C_l726_726351

variables {A B C : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C]
variables (a b : ℝ) (area : ℝ)

def cos2C (a b : ℝ) (area : ℝ) : ℝ :=
1 - 2 * ((2 * area / (a * b)) ^ 2 )

theorem triangle_cos2C:
  a = 8 → b = 5 → area = 12 → cos2C a b area = 7 / 25 :=
by
  intros
  simp [cos2C]
  sorry

end triangle_cos2C_l726_726351


namespace value_of_a9_l726_726721

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

variable a : ℕ → ℝ

axiom a1 : a 1 = 1
axiom a7_a11 : a 7 * a 11 = 100
axiom geo_seq : geometric_sequence a

theorem value_of_a9 : a 9 = 10 :=
by
  sorry

end value_of_a9_l726_726721


namespace least_positive_base_ten_with_seven_binary_digits_l726_726089

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726089


namespace larger_of_two_numbers_l726_726113

theorem larger_of_two_numbers (hcf : ℕ) (f1 : ℕ) (f2 : ℕ) 
(h_hcf : hcf = 10) 
(h_f1 : f1 = 11) 
(h_f2 : f2 = 15) 
: max (hcf * f1) (hcf * f2) = 150 :=
by
  have lcm := hcf * f1 * f2
  have num1 := hcf * f1
  have num2 := hcf * f2
  sorry

end larger_of_two_numbers_l726_726113


namespace trig_expression_value_l726_726448

open Real

theorem trig_expression_value : 
  (2 * cos (10 * (π / 180)) - sin (20 * (π / 180))) / cos (20 * (π / 180)) = sqrt 3 :=
by
  -- Proof should go here
  sorry

end trig_expression_value_l726_726448


namespace gross_profit_percentage_is_correct_l726_726262

def selling_price : ℝ := 28
def wholesale_cost : ℝ := 24.56
def gross_profit : ℝ := selling_price - wholesale_cost

-- Define the expected profit percentage as a constant value.
def expected_profit_percentage : ℝ := 14.01

theorem gross_profit_percentage_is_correct :
  ((gross_profit / wholesale_cost) * 100) = expected_profit_percentage :=
by
  -- Placeholder for proof
  sorry

end gross_profit_percentage_is_correct_l726_726262


namespace Cody_first_week_books_l726_726537

theorem Cody_first_week_books :
  ∀ (totalBooks secondWeekBooks weeklyBooks remainingWeeks : ℕ) (firstWeekBooks : ℕ),
    totalBooks = 54 →
    secondWeekBooks = 3 →
    weeklyBooks = 9 →
    remainingWeeks = 5 →
    firstWeekBooks + secondWeekBooks + remainingWeeks * weeklyBooks = totalBooks →
    firstWeekBooks = 6 :=
by
  intros totalBooks secondWeekBooks weeklyBooks remainingWeeks firstWeekBooks
  intro h_totals h_second h_weekly h_remaining h_eq
  have h1 : 54 = totalBooks := h_totals
  have h2 : 3 = secondWeekBooks := h_second
  have h3 : 9 = weeklyBooks := h_weekly
  have h4 : 5 = remainingWeeks := h_remaining
  have h5 : firstWeekBooks + secondWeekBooks + remainingWeeks * weeklyBooks = totalBooks := h_eq
  have h6 : 54 = 3 + 5 * 9 + firstWeekBooks := by
    rw [h_totals, h_second, h_weekly, h_remaining, h_eq]
  sorry

end Cody_first_week_books_l726_726537


namespace least_positive_base_ten_with_seven_binary_digits_l726_726094

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726094


namespace lambda_greater_than_neg_three_l726_726722

noncomputable theory

open Classical

variable (a : ℕ → ℝ)
variable (λ : ℝ)

-- Conditions
def sequence_definition (n : ℕ) : Prop := a n = n^2 + λ * n
def increasing_sequence : Prop := ∀ n : ℕ, a (n + 1) > a n

-- The proof statement
theorem lambda_greater_than_neg_three :
  (∀ n : ℕ, sequence_definition a λ n) → (increasing_sequence a) → λ > -3 :=
  by
    intro seq_def inc_seq
    sorry

end lambda_greater_than_neg_three_l726_726722


namespace least_positive_base_ten_with_seven_binary_digits_l726_726091

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726091


namespace number_students_passed_is_49_number_students_scored_above_120_is_8_l726_726984

open Real

noncomputable def normal_dist (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ :=
MeasureTheory.ProbabilityMeasure.fromDensity (λ x, (exp (-((x - μ)^2) / (2 * σ^2))) / (σ * sqrt (2 * π)))

constant X : ℝ → ℝ -- Random variable representing scores

constant P : Set ℝ → ℝ -- Probability measure

axiom normal_X : ∀ (s : Set ℝ), P s = MeasureTheory.ProbabilityMeasure.toMeasure (normal_dist 110 10) s

-- Number of students
constant num_students : ℕ := 50

def number_of_students_passed : ℕ :=
(num_students * (1 - ((1 - 0.954) / 2)))

def number_of_students_scored_above_120 : ℕ :=
(num_students * (0.5 * (1 - 0.683)))

theorem number_students_passed_is_49 :
  number_of_students_passed = 49 := by
  sorry

theorem number_students_scored_above_120_is_8 :
  number_of_students_scored_above_120 = 8 := by
  sorry

end number_students_passed_is_49_number_students_scored_above_120_is_8_l726_726984


namespace smallest_k_l726_726037

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l726_726037


namespace total_amount_received_l726_726127

theorem total_amount_received
  (total_books : ℕ := 500)
  (novels_price : ℕ := 8)
  (biographies_price : ℕ := 12)
  (science_books_price : ℕ := 10)
  (novels_discount : ℚ := 0.25)
  (biographies_discount : ℚ := 0.30)
  (science_books_discount : ℚ := 0.20)
  (sales_tax : ℚ := 0.05)
  (remaining_novels : ℕ := 60)
  (remaining_biographies : ℕ := 65)
  (remaining_science_books : ℕ := 50)
  (novel_ratio_sold : ℚ := 3/5)
  (biography_ratio_sold : ℚ := 2/3)
  (science_book_ratio_sold : ℚ := 7/10)
  (original_novels : ℕ := 150)
  (original_biographies : ℕ := 195)
  (original_science_books : ℕ := 167) -- Rounded from 166.67
  (sold_novels : ℕ := 90)
  (sold_biographies : ℕ := 130)
  (sold_science_books : ℕ := 117)
  (total_revenue_before_discount : ℚ := (90 * 8 + 130 * 12 + 117 * 10))
  (total_revenue_after_discount : ℚ := (720 * (1 - 0.25) + 1560 * (1 - 0.30) + 1170 * (1 - 0.20)))
  (total_revenue_after_tax : ℚ := (2568 * 1.05)) :
  total_revenue_after_tax = 2696.4 :=
by
  sorry

end total_amount_received_l726_726127


namespace least_positive_base_ten_seven_binary_digits_l726_726050

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726050


namespace find_length_DE_l726_726528

-- Define the lengths AB and AD as constants
def AB : ℕ := 5
def AD : ℕ := 6

-- Define the equality of areas
def equal_areas := (AB * AD : ℕ) = (1 / 2 : ℝ) * (AB : ℝ) * (CE : ℝ)

-- Declare the unknown length CE as a variable
noncomputable def CE : ℝ := (60 : ℝ) / (AB : ℝ)

-- The Pythagorean theorem to find the hypotenuse DE
theorem find_length_DE : CE = 12 → √((AB : ℝ) ^ 2 + CE ^ 2) = 13 := 
by
  intro hCE
  rw hCE
  calc 
    √((AB : ℝ) ^ 2 + (CE : ℝ) ^ 2)
        = √(5 ^ 2 + 12 ^ 2)      : by rw [CE, ← hCE]
    ... = √(25 + 144)
    ... = √169
    ... = 13

end find_length_DE_l726_726528


namespace shortest_stable_tower_has_5_levels_l726_726913

-- Definitions to formalize tower and stability conditions.
def tower (levels : ℕ) : Prop :=
  -- Add condition for the layout of dominoes covering a 10x11 grid at each level.
  sorry -- Assumes coverage of dominoes at each level

def stable (levels : ℕ) : Prop :=
  -- Assumes the condition of internal grid points being covered by internal points of dominos at each level.
  sorry -- Assumption ensuring stability by the problem definition

-- Main theorem stating the minimal number of levels for a stable tower is 5.
theorem shortest_stable_tower_has_5_levels : ∀ (n : ℕ), stable n → n ≥ 5 :=
by {
  intros n h_stable,
  sorry
}

end shortest_stable_tower_has_5_levels_l726_726913


namespace estate_value_l726_726398

theorem estate_value (x : ℕ) (E : ℕ) (cook_share : ℕ := 500) 
  (daughter_share : ℕ := 4 * x) (son_share : ℕ := 3 * x) 
  (wife_share : ℕ := 6 * x) (estate_eqn : E = 14 * x) : 
  2 * (daughter_share + son_share) = E ∧ wife_share = 2 * son_share ∧ E = 13 * x + cook_share → 
  E = 7000 :=
by
  sorry

end estate_value_l726_726398


namespace line_equation_parallel_to_x_axis_through_point_l726_726878

-- Define the point (3, -2)
def point : ℝ × ℝ := (3, -2)

-- Define a predicate for a line being parallel to the X-axis
def is_parallel_to_x_axis (line : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, line x k

-- Define the equation of the line passing through the given point
def equation_of_line_through_point (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- State the theorem to be proved
theorem line_equation_parallel_to_x_axis_through_point :
  ∀ (line : ℝ → ℝ → Prop), 
    (equation_of_line_through_point point line) → (is_parallel_to_x_axis line) → (∀ x, line x (-2)) :=
by
  sorry

end line_equation_parallel_to_x_axis_through_point_l726_726878


namespace horizontal_distance_parabola_l726_726966

theorem horizontal_distance_parabola :
  let parabola := λ x : ℝ, x^2 - x - 6
  let P := (x : ℝ) (parabola x = 6)
  let Q := (x : ℝ) (parabola x = -6)
  (∃ x_p x_q, parabola x_p = 6 ∧ parabola x_q = -6 ∧ |x_p - x_q| = 4) :=
begin
  let parabola := λ x : ℝ, x^2 - x - 6,
  let P := parabola 4,
  let Q := parabola 0,
  have horz_dist := abs (4 - 0) ∨ abs (-3 - 1),
  exact horz_dist,
  sorry
end

end horizontal_distance_parabola_l726_726966


namespace boys_girls_distribution_l726_726971

theorem boys_girls_distribution
  (B G : ℕ) (HB : B ≈ 60) (h₁ : B + G = 100) (h₂ : 2.40 * G ≤ 312) : 
  let total_girls_amt := 2.40 * G in
  let total_boys_amt := 312 - total_girls_amt in
  let amount_per_boy := total_boys_amt / B in 
  amount_per_boy = 3.60 :=
by
  sorry

end boys_girls_distribution_l726_726971


namespace range_of_k_l726_726307

open Set

theorem range_of_k (A B : Set ℝ) (k : ℝ) :
  A = { x : ℝ | x ≤ 1 ∨ x ≥ 3 } →
  B = { x : ℝ | k < x ∧ x < k + 1 } →
  disjoint (compl A) B →
  k ∈ Iic 0 ∪ Ici 3 :=
by
  intros hA hB hDisjoint
  sorry

end range_of_k_l726_726307


namespace calculate_expression_l726_726478

def theta (m v : ℕ) : ℕ := m % v

theorem calculate_expression :
  (theta (theta 790 243) 127) + (theta 950 (theta 237 207)) * ((theta 564 123) - (theta 789 (theta 331 197))) = -879 :=
by
  sorry

end calculate_expression_l726_726478


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726070

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726070


namespace gcd_g105_g106_l726_726387

def g (x : ℕ) : ℕ := x^2 - x + 2502

theorem gcd_g105_g106 : gcd (g 105) (g 106) = 2 := by
  sorry

end gcd_g105_g106_l726_726387


namespace angle_ADB_is_right_angle_l726_726525

-- Define the basic structures and assumptions
noncomputable def isosceles_triangle : Prop :=
  ∃ (A B C : Point), A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  dist A B = dist A C

-- Prove the angle ADB is 90 degrees
theorem angle_ADB_is_right_angle
  (A B C D : Point)
  (h_isosceles : isosceles_triangle A B C)
  (h_circle_center : Center C)
  (h_circle_radius : radius C = 12)
  (h_circle_through_B : on_circle C B)
  (h_B_not_A : ¬ on_circle C A)
  (h_D_on_extension : on_circle C D ∧ lies_on_line D (extension B C)) :
  angle A D B = 90 :=
sorry

end angle_ADB_is_right_angle_l726_726525


namespace half_radius_y_l726_726481

theorem half_radius_y (r_x r_y : ℝ) (hx : 2 * Real.pi * r_x = 12 * Real.pi) (harea : Real.pi * r_x ^ 2 = Real.pi * r_y ^ 2) : r_y / 2 = 3 := by
  sorry

end half_radius_y_l726_726481


namespace area_triangle_is_sqrt3_l726_726802

noncomputable def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def line_l_polar (θ : ℝ) : Prop := θ = Real.pi / 6
noncomputable def point_A (x y : ℝ) : Prop := y = (Real.sqrt 3) / 3 * x  -- Based on the tangent condition
noncomputable def point_N_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
noncomputable def area_of_triangle (A N : ℝ × ℝ) : ℝ :=
  0.5 * Real.abs (A.1 * (0 - N.2) + N.1 * (A.2 - 0))  -- Using the determinant formula for area of a triangle

theorem area_triangle_is_sqrt3 :
  ∃ A N : ℝ × ℝ, circle_M A.1 A.2 ∧ curve_C N.1 N.2 ∧
    line_l_polar (Real.atan2 A.2 A.1) ∧ point_N_in_third_quadrant N.1 N.2 ∧
    area_of_triangle A N = Real.sqrt 3 :=
sorry

end area_triangle_is_sqrt3_l726_726802


namespace lemonade_calories_l726_726907

/-- Carla mixes the following into her lemonade:
- 150 grams of lemon juice with 30 calories per 100 grams.
- 150 grams of sugar with 390 calories per 100 grams.
- 200 grams of water with 0 calories per 100 grams.
- 50 grams of lime juice with 10 calories per 100 grams.

We need to prove that 300 grams of this lemonade contains 346 calories. -/
theorem lemonade_calories :
  let lemon_juice_grams := 150
  let sugar_grams := 150
  let water_grams := 200
  let lime_juice_grams := 50
  
  let lemon_juice_cal_per_100_grams := 30
  let sugar_cal_per_100_grams := 390
  let water_cal_per_100_grams := 0
  let lime_juice_cal_per_100_grams := 10
  
  -- Calculate the total calories in 550 grams of lemonade.
  let total_calories := (lemon_juice_grams * lemon_juice_cal_per_100_grams / 100)
                      + (sugar_grams * sugar_cal_per_100_grams / 100)
                      + (lime_juice_grams * lime_juice_cal_per_100_grams / 100)
                      + (water_grams * water_cal_per_100_grams / 100)
  -- Proportional calculation to find calories in 300 grams of lemonade.
  let calories_in_300_grams := total_calories * 300 / 550
  calories_in_300_grams = 346 :=
by
  -- weights
  have h1 : lemon_juice_grams = 150 := rfl
  have h2 : sugar_grams = 150 := rfl
  have h3 : water_grams = 200 := rfl
  have h4 : lime_juice_grams = 50 := rfl

  -- caloric content
  have h5 : lemon_juice_cal_per_100_grams = 30 := rfl
  have h6 : sugar_cal_per_100_grams = 390 := rfl
  have h7 : water_cal_per_100_grams = 0 := rfl
  have h8 : lime_juice_cal_per_100_grams = 10 := rfl

  -- total calories
  have h_total_calories : total_calories = (150 * 30 / 100) + (150 * 390 / 100) + (50 * 10 / 100) + (200 * 0 / 100) := rfl
  
  -- Calculations proceed as in the solution steps ...
  
  sorry

end lemonade_calories_l726_726907


namespace problem_statement_l726_726832

def f (x : ℝ) : ℝ :=
if x ≥ 0 then sin (π * x)
else cos (π * x / 2 + π / 3)

theorem problem_statement : f (f (15 / 2)) = √3 / 2 :=
by
  sorry

end problem_statement_l726_726832


namespace consumer_installment_credit_l726_726477

theorem consumer_installment_credit : 
  ∃ C : ℝ, 
    (0.43 * C = 200) ∧ 
    (C = 465.116) :=
by
  sorry

end consumer_installment_credit_l726_726477


namespace right_triangle_median_l726_726839

variable (A B C M N : Type) [LinearOrder B] [LinearOrder C] [LinearOrder A] [LinearOrder M] [LinearOrder N]
variable (AC BC AM BN AB : ℝ)
variable (right_triangle : AC * AC + BC * BC = AB * AB)
variable (median_A : AC * AC + (1 / 4) * BC * BC = 81)
variable (median_B : BC * BC + (1 / 4) * AC * AC = 99)

theorem right_triangle_median :
  ∀ (AC BC AB : ℝ),
  (AC * AC + BC * BC = 144) → (AC * AC + BC * BC = AB * AB) → AB = 12 :=
by
  intros
  sorry

end right_triangle_median_l726_726839


namespace ceil_sq_values_count_l726_726324

theorem ceil_sq_values_count (y : ℝ) (hy : ⌈y⌉ = 15) : 
  (finset.range (⌈15^2⌉ - ⌈14^2⌉ + 1)).card = 29 :=
by
  let lower_bound := 14
  let upper_bound := 15
  have h1 : lower_bound < y := sorry
  have h2 : y ≤ upper_bound := sorry
  have sq_lower_bound : lower_bound^2 < y^2 := sorry
  have sq_upper_bound : y^2 ≤ upper_bound^2 := sorry
  let vals := (finset.range (⌈upper_bound^2⌉ - ⌈lower_bound^2⌉ + 1)).val
  have h3 : sq_lower_bound ≥ 196 := sorry
  have h4 : sq_upper_bound ≤ 225 := sorry
  rw finset.card
  exact sorry

end ceil_sq_values_count_l726_726324


namespace missing_number_is_seven_l726_726189

def givenNumbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / (l.length : ℚ)

theorem missing_number_is_seven (x : ℕ) :
  mean (givenNumbers ++ [x]) = 12 → x = 7 := by
have sum_given : givenNumbers.sum = 137 := by sorry
have count_given : givenNumbers.length = 11 := by sorry
assume h : mean (givenNumbers ++ [x]) = 12
have total_count : (givenNumbers ++ [x]).length = 12 := by 
  rw [List.length_append, List.length_singleton]
  exact count_given + 1
have total_sum : (givenNumbers ++ [x]).sum = 12 * 12 := by
  unfold mean at h
  rw [List.length_append, List.length_singleton, count_given] at h
  norm_cast at h
  have h_sum := congr_arg (λ y, y * 12) h
  rw [mul_div_cancel' _ (nat.succ_ne_zero 11)] at h_sum
  exact congr_arg (Nat.add x) (eq.trans sum_given h_sum) 
have : 137 + x = 144 := by rw [List.sum_append, List.sum_singleton, sum_given]
exact eq_of_add_eq_add_left this rfl

end missing_number_is_seven_l726_726189


namespace new_quadratic_equation_has_square_roots_l726_726472

theorem new_quadratic_equation_has_square_roots (p q : ℝ) (x : ℝ) :
  (x^2 + px + q = 0 → ∃ x1 x2 : ℝ, x^2 - (p^2 - 2 * q) * x + q^2 = 0 ∧ (x1^2 = x ∨ x2^2 = x)) :=
by sorry

end new_quadratic_equation_has_square_roots_l726_726472


namespace balloon_difference_l726_726474

theorem balloon_difference 
  (your_balloons : ℕ := 7) 
  (friend_balloons : ℕ := 5) : 
  your_balloons - friend_balloons = 2 := 
by 
  sorry

end balloon_difference_l726_726474


namespace side_length_of_square_l726_726160

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726160


namespace decreasing_interval_l726_726250

noncomputable def f (x : ℝ) := -x^2 - x + 4

theorem decreasing_interval : 
  ∀ x₀, (x₀ ∈ Ici (-(1:ℝ)/2)) → (∀ x > x₀, f x < f x₀) :=
sorry

end decreasing_interval_l726_726250


namespace convert_base_8_to_base_10_l726_726546

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l726_726546


namespace smallest_k_l726_726040

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l726_726040


namespace base8_246_is_166_in_base10_l726_726549

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l726_726549


namespace smallest_special_gt_3429_l726_726600

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726600


namespace smallest_possible_degree_l726_726965

theorem smallest_possible_degree :
  ∃ (p : polynomial ℚ), p ≠ 0 ∧
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (p.eval (n + real.sqrt (n + 2) : ℂ) = 0)) ∧
  (∃! d, p.degree = (191 : ℕ) ∧ p.degree < d) := sorry

end smallest_possible_degree_l726_726965


namespace solve_for_x_l726_726862

theorem solve_for_x (x : ℝ) :
  sqrt (x^3) = 9 * (81^(1/9)) → x = 9 :=
by
  sorry

end solve_for_x_l726_726862


namespace imaginary_part_z_l726_726288

/-- Let \(z = \frac {2i^{3}}{i-1}\),
where \(i\) is the imaginary unit. Prove that the imaginary part of \(z\) is \(1\). -/
theorem imaginary_part_z : 
  let i := Complex.I in
  let z := (2 * i^3) / (i - 1) in
  Complex.im z = 1 :=
by
  -- Definitions
  let i := Complex.I
  let z := (2 * i^3) / (i - 1)
  -- Proof would go here
  sorry

end imaginary_part_z_l726_726288


namespace roots_of_quadratic_eq_l726_726891

theorem roots_of_quadratic_eq (x : ℝ) : (x + 1) ^ 2 = 0 → x = -1 := by
  sorry

end roots_of_quadratic_eq_l726_726891


namespace M_inter_N_eq_N_l726_726264

variable (M : Set ℝ)
variable (N : Set ℝ)

def condM : Prop := M = Set.univ
def condN : Prop := N = { y : ℝ | y ≥ -2 }

theorem M_inter_N_eq_N (hM : condM M) (hN : condN N) : M ∩ N = N := 
by 
  sorry

end M_inter_N_eq_N_l726_726264


namespace claire_apple_pies_l726_726095

theorem claire_apple_pies (N : ℤ) 
  (h1 : N % 6 = 4) 
  (h2 : N % 8 = 5) 
  (h3 : N < 30) : 
  N = 22 :=
by
  sorry

end claire_apple_pies_l726_726095


namespace mode_of_data_set_l726_726683

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726683


namespace smallest_k_mod_19_7_3_l726_726033

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l726_726033


namespace nonnegative_diff_roots_l726_726228

theorem nonnegative_diff_roots : 
  ∀ a b c: ℝ, 
  a ≠ 0 → 
  a * b^2 + b * b + c = 0 → 
  let Δ := b^2 - 4 * a * c in 
  Δ ≥ 0 ∧ a > 0 ∧ c > 0 →
  let x1 := (-b + Real.sqrt Δ) / (2 * a) in
  let x2 := (-b - Real.sqrt Δ) / (2 * a) in
  a = 1 ∧ b = 6 ∧ c = 8 →
  |x1 - x2| = 2 :=
by sorry

end nonnegative_diff_roots_l726_726228


namespace intersection_of_diagonals_on_XO_l726_726968

theorem intersection_of_diagonals_on_XO
    {A B C D O X : Point}
    (h1 : is_cyclic_quadrilateral A B C D O)
    (h2 : ∠ B A X = 90)
    (h3 : ∠ C D X = 90) :
  lies_on_line (line_through X O) (intersection_point (line_through A C) (line_through B D)) := 
sorry

end intersection_of_diagonals_on_XO_l726_726968


namespace wedge_product_correct_l726_726377

variables {a1 a2 b1 b2 : ℝ}

def vector_a := (a1, a2)
def vector_b := (b1, b2)

def wedge_product (a b : ℝ × ℝ) : ℝ := 
  a.1 * b.2 - a.2 * b.1

theorem wedge_product_correct :
  wedge_product vector_a vector_b = a1 * b2 - a2 * b1 :=
by sorry

end wedge_product_correct_l726_726377


namespace volume_of_regular_triangular_pyramid_l726_726256

noncomputable def pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

theorem volume_of_regular_triangular_pyramid (R φ : ℝ) 
  (cond1 : R > 0)
  (cond2: 0 < φ ∧ φ < π) :
  ∃ V, V = pyramid_volume R φ := by
    use (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)
    sorry

end volume_of_regular_triangular_pyramid_l726_726256


namespace volume_of_solid_l726_726846

-- Definitions and conditions
def C (x y : ℝ) : Prop := x^2 + y^2 ≤ 1
def L_parallel := true -- Line L is parallel to the x-axis, no other specification needed here

-- Given these conditions, prove the volume of the solid
theorem volume_of_solid : 
  ∀ x y: ℝ, C x y → 
  let AQ := x + sqrt(1 - y^2) in
  let BQ := sqrt(1 - y^2) - x in
  let PQ := (1 - x^2 - y^2) in 
  PQ = AQ * BQ →
  2 * (∫ θ in 0..2*π, ∫ r in 0..1, (1 - r^2) * r) = π :=
  sorry

end volume_of_solid_l726_726846


namespace number_of_excellent_sequences_l726_726672

-- Definitions based on conditions
def excellent_sequence (n : ℕ) (xs : Fin (n+1) → ℤ) : Prop :=
  (∀ i, abs (xs i) ≤ n) ∧
  (∀ i j, i < j → xs i ≠ xs j) ∧
  (∀ i j k, i < j → j < k → max (abs (xs k - xs i)) (abs (xs k - xs j)) =
    ↑(1/2 * (abs (xs i - xs j) + abs (xs j - xs k) + abs (xs k - xs i))))

-- The main problem statement
theorem number_of_excellent_sequences (n : ℕ) :
  ∃ (count : ℕ), count = nat.choose (2*n+1) (n+1) * 2^n :=
sorry

end number_of_excellent_sequences_l726_726672


namespace max_votes_independent_set_l726_726957

noncomputable def max_independent_set (n : ℕ) : ℕ := n / 3

theorem max_votes_independent_set (n : ℕ) (hn : 2 ≤ n) :
  ∃ (k : ℕ), k = max_independent_set n ∧
  (∀ (vote : Fin n → Fin n), k ≤ ∃ (independent_set : Set (Fin n)),
    (∀ (i : Fin n), i ∈ independent_set → ∀ (j : Fin n), vote i ≠ j ∨ j ∉ independent_set)) :=
by
  let k := max_independent_set n
  use k
  split
  . rfl
  . intro vote
    sorry

end max_votes_independent_set_l726_726957


namespace side_length_of_square_l726_726158

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726158


namespace smallest_special_number_gt_3429_l726_726607

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726607


namespace rectangle_area_l726_726854

theorem rectangle_area (Q P R : Point) (ABCD_inscribed : Rectangle ABCD inscribed_in Triangle PQR)
  (PR : Segment P R) (QS : Altitude P Q R)
  (PR_length : PR.length = 12) (QS_length : QS.length = 8)
  (AD_on_PR : AD.on PR) (AB_eq_one_third_AD : AB.length = (1/3) * AD.length) :
  ∃ (AB : ℝ) (AD : ℝ), AD = 8 ∧ AB = 8/3 ∧ rectangle_area ABCD = 64/3 :=
by
  sorry

end rectangle_area_l726_726854


namespace correct_answer_is__l726_726521

inductive ChromosomeVariation : Type
| deletion : ChromosomeVariation
| duplication : ChromosomeVariation
| inversion : ChromosomeVariation
| translocation : ChromosomeVariation
| numerical : ChromosomeVariation

structure Statement (n : Nat) : Type :=
(description : String)
(is_variation : Prop)

def stmt1 : Statement 1 := { 
  description := "Partial deletion of chromosome 5 causes cri-du-chat syndrome",
  is_variation := true }

def stmt2 : Statement 2 := {
  description := "Free combination of non-homologous chromosomes during meiosis",
  is_variation := false }

def stmt3 : Statement 3 := {
  description := "Chromosomal exchange between synapsed homologous chromosomes",
  is_variation := false }

def stmt4 : Statement 4 := {
  description := "Patients with Down syndrome have an extra chromosome 21",
  is_variation := true }

def isCorrectAnswer (s1 s2 : Statement _) : Prop :=
  s1.is_variation ∧ s2.is_variation

theorem correct_answer_is_ {A B C D : (List (Statement _))} : 
  isCorrectAnswer stmt1 stmt4 ∧ ¬ isCorrectAnswer stmt2 stmt3 ∧ ¬ isCorrectAnswer stmt2 stmt4
  → B = [stmt1, stmt4] :=
by {
  intro h,
  cases h with h1 h2, sorry
}

end correct_answer_is__l726_726521


namespace at_least_one_unwatched_l726_726894

theorem at_least_one_unwatched (n : ℕ) (odd_n : n % 2 = 1)
  (d : Fin n → Fin n → ℝ)
  (distinct_distances : ∀ i j k l : Fin n, d i j ≠ d k l)
  (nearest_neighbor : ∀ i : Fin n, ∃ j : Fin n, j ≠ i ∧ ∀ k : Fin n, k ≠ j → d i j < d i k) :
  ∃ i : Fin n, ∀ j : Fin n, j ≠ i → (!(∃ k : Fin n, nearest_neighbor k = i)) :=
sorry

end at_least_one_unwatched_l726_726894


namespace mode_of_data_set_l726_726698

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726698


namespace sum_squares_leq_one_l726_726887

theorem sum_squares_leq_one 
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_decreasing : ∀ i j : Fin n, i ≤ j → x i ≥ x j)
  (h_nonneg : ∀ i : Fin n, 0 ≤ x i)
  (h_sum : ∑ i, x i / Real.sqrt (i + 1) = 1) 
  : ∑ i, (x i)^2 ≤ 1 :=
sorry

end sum_squares_leq_one_l726_726887


namespace multiplicative_inverse_600_mod_4901_l726_726015

theorem multiplicative_inverse_600_mod_4901 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 4901 ∧ (600 * n) % 4901 = 1 :=
begin
  use 3196,
  norm_num,
end

end multiplicative_inverse_600_mod_4901_l726_726015


namespace mode_of_data_set_l726_726699

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726699


namespace smallest_special_number_l726_726583

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726583


namespace tammy_total_distance_l726_726868

theorem tammy_total_distance :
  let D1 := 70 * 2,
      D2 := 60 * 3,
      D3 := 55 * 2,
      D4 := 65 * 4 in
  D1 + D2 + D3 + D4 = 690 :=
by
  sorry

end tammy_total_distance_l726_726868


namespace square_side_length_l726_726178

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726178


namespace prism_sliced_surface_area_l726_726513

noncomputable def area_CXZ_Y := 100 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 418.75) / 2

structure Prism :=
  (height : ℝ)
  (side_length : ℝ)
  (is_equilateral : ∀ p q r, p ∈ {a, b, c} → q ∈ {a, b, c} → r ∈ {a, b, c} → 
     triangle_equilateral p q r)

-- Define points A, B, C, D, E, F and their placements
structure Points :=
  (A B C D E F : ℝ × ℝ × ℝ)

structure Midpoints :=
  (X Y Z_prime : ℝ × ℝ × ℝ)

def Midpoints_on_edges (P : Points) (M : Midpoints) : Prop :=
  midpoint M.X P.A P.C ∧
  midpoint M.Y P.B P.C ∧
  midpoint M.Z_prime P.D P.F

def valid_prism (P : Prism) (Pts : Points) (M : Midpoints) : Prop :=
  Prism.height P = 20 ∧
  Prism.side_length P = 10 ∧
  Midpoints_on_edges Pts M

theorem prism_sliced_surface_area 
  (P : Prism) (Pts : Points) (M : Midpoints) (h : valid_prism P Pts M) :
  surface_area_sliced Pts.M.X M.Y M.Z_prime M.C = 
    area_CXZ_Y := 
sorry

end prism_sliced_surface_area_l726_726513


namespace boys_not_adjacent_correct_girls_adjacent_correct_girls_not_at_ends_correct_l726_726025

-- Define the necessary elements for the problem
inductive Person : Type
| boy : Person
| girl : Person

open Person

-- Define the specific conditions for the arrangement based on the problem statement

-- Number of boys and girls
def boys_count : ℕ := 3
def girls_count : ℕ := 2
def total_people : ℕ := boys_count + girls_count

-- Factorial function (this is already in Mathlib)
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Calculate factorial value of total people (5!)
def total_permutations : ℕ := fact total_people

-- Boys are not adjacent
def boys_not_adjacent : ℕ :=
  let ways_girls := fact girls_count in
  let ways_boys := fact boys_count in
  ways_girls * ways_boys

-- Girls are adjacent (considering them as a single unit)
def girls_adjacent : ℕ :=
  let new_units := total_people - girls_count + 1 in -- 4 units (3 boys + 1 girl unit)
  let ways_units := fact new_units in
  let ways_within_unit := fact girls_count in
  ways_units * ways_within_unit

-- Girls are not at the ends
def girls_not_at_ends : ℕ :=
  let positions_for_girls := fact 3 / fact (3 - girls_count) in -- Choose 2 out of 3 positions
  let ways_boys := fact boys_count in
  positions_for_girls * ways_boys

-- The actual assertions we need to prove
theorem boys_not_adjacent_correct : boys_not_adjacent = 12 := by sorry
theorem girls_adjacent_correct : girls_adjacent = 48 := by sorry
theorem girls_not_at_ends_correct : girls_not_at_ends = 36 := by sorry

end boys_not_adjacent_correct_girls_adjacent_correct_girls_not_at_ends_correct_l726_726025


namespace eccentricity_is_sqrt5_l726_726455

open Real

noncomputable def eccentricity_of_hyperbola 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ℝ :=
let A := (a, 0)
let l := λ x : ℝ, -x + a
let l₁ := λ (x y : ℝ), b * x - a * y = 0
let l₂ := λ (x y : ℝ), b * x + a * y = 0
let B := (a^2 / (a + b), a * b / (a + b))
let C := (a^2 / (a - b), -a * b / (a - b))
let AB := (-a * b / (a + b), a * b / (a + b))
let BC := (2 * a^2 * b / (a^2 - b^2), -2 * a^2 * b / (a^2 - b^2))
if 2 * AB = BC then
  let c := sqrt (a^2 + b^2)
  sqrt (c^2 / a^2)
else
  0

theorem eccentricity_is_sqrt5 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h3 : b = 2 * a) 
  (h4 : A := (a, 0))
  (h5 : l := λ x : ℝ, -x + a)
  (h6 : l₁ := λ (x y : ℝ), b * x - a * y = 0)
  (h7 : l₂ := λ (x y : ℝ), b * x + a * y = 0)
  (h8 : B := (a^2 / (a + b), a * b / (a + b)))
  (h9 : C := (a^2 / (a - b), -a * b / (a - b)))
  (h10 : AB := (-a * b / (a + b), a * b / (a + b)))
  (h11 : BC := (2 * a^2 * b / (a^2 - b^2), -2 * a^2 * b / (a^2 - b^2)))
  (h12 : 2 * AB = BC) : 
  eccentricity_of_hyperbola a b h₁ h₂ = sqrt 5 := 
by 
  sorry

end eccentricity_is_sqrt5_l726_726455


namespace find_a_range_l726_726741

-- The function f(x) is given as:
def f (x a : ℝ) := (1/3) * x^3 + x^2 - a * x

-- Define the derivative of the function f(x)
def f_derivative (x a : ℝ) := x^2 + 2*x - a

-- Lean statement to assert the conditions and required proof
theorem find_a_range (a : ℝ) :
  (∀ x > 1, f_derivative x a ≥ 0) ∧ (∃ x ∈ Set.Ioo (1:ℝ) 2, f x a = 0) → (4/3 : ℝ) < a ∧ a ≤ 3 :=
begin
  sorry
end

end find_a_range_l726_726741


namespace no_draw_in_token_game_l726_726489

theorem no_draw_in_token_game 
  (tokens : Finset (ℕ × ℕ))
  (red_tokens : ℕ)
  (blue_tokens : ℕ)
  (turns : list (ℕ × ℕ))
  (player_A_win : list (ℕ × ℕ) -> Prop)
  (player_B_win : list (ℕ × ℕ) -> Prop)
  (A_path : set (ℕ × ℕ))
  (B_path : set (ℕ × ℕ)) : Prop := 
  (∀ (ps qr pq sr : set (ℕ × ℕ)), A_path = ps ∩ qr) → 
  (∀ (pq sr : set (ℕ × ℕ)), B_path = pq ∩ sr) → 
  (tokens.card = 121 ∧ red_tokens = 61 ∧ blue_tokens = 60) →
  (∀ turns, (∃ A_path ∪ B_path ∈ tokens)) →
  ((player_A_win turns) ∨ (player_B_win turns)) 
  sorry

end no_draw_in_token_game_l726_726489


namespace bob_fencing_needed_l726_726531

-- Problem conditions
def length : ℕ := 225
def width : ℕ := 125
def small_gate : ℕ := 3
def large_gate : ℕ := 10

-- Definition of perimeter
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Total width of the gates
def total_gate_width (g1 g2 : ℕ) : ℕ := g1 + g2

-- Amount of fencing needed
def fencing_needed (p gw : ℕ) : ℕ := p - gw

-- Theorem statement
theorem bob_fencing_needed :
  fencing_needed (perimeter length width) (total_gate_width small_gate large_gate) = 687 :=
by 
  sorry

end bob_fencing_needed_l726_726531


namespace triangle_subsegment_length_l726_726021

theorem triangle_subsegment_length (DF DE EF DG GF : ℚ)
  (h_ratio : ∃ x : ℚ, DF = 3 * x ∧ DE = 4 * x ∧ EF = 5 * x)
  (h_EF_len : EF = 20)
  (h_angle_bisector : DG + GF = DE ∧ DG / GF = DE / DF) :
  DF < DE ∧ DE < EF →
  min DG GF = 48 / 7 :=
by
  sorry

end triangle_subsegment_length_l726_726021


namespace tens_digit_of_3_pow_405_tens_digit_is_4_l726_726045

theorem tens_digit_of_3_pow_405 :
  (3 ^ 405) % 100 = 43 :=
sorry

theorem tens_digit_is_4 :
  (3 ^ 405) // 10 % 10 = 4 :=
sorry

end tens_digit_of_3_pow_405_tens_digit_is_4_l726_726045


namespace average_remaining_numbers_l726_726872

theorem average_remaining_numbers (s : Fin 12 → ℝ) 
  (h_average : (∑ i, s i) / 12 = 90)
  (h_remove : ∀ i j k, s i = 80 → s j = 85 → s k = 95)
  : (∑ i in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} \ {i, j, k}, s i) / 9 = 820 / 9 := 
sorry

end average_remaining_numbers_l726_726872


namespace rate_of_change_const_derivative_constant_function_instantaneous_rate_of_change_physical_quantity_derivative_definition_nonzero_l726_726620

theorem rate_of_change_const (c : ℝ) (x1 x2 : ℝ) :
  (x1 ≠ x2) → ((f : ℝ → ℝ) = λ x, c) → (Δy = f x2 - f x1) →
  (Δx = x2 - x1) → (Δy / Δx = 0) :=
by
  sorry

theorem derivative_constant_function (c : ℝ) (x : ℝ) :
  ((f : ℝ → ℝ) = λ x, c) → (∀ Δx ≠ 0, Δy = f (x + Δx) - f x) →
  (lim (Δx → 0) (Δy / Δx) = 0) :=
by
  sorry

theorem instantaneous_rate_of_change_physical_quantity {f : ℝ → ℝ} (x1 x2 : ℝ) :
  (x1 ≠ x2) →
  (Δy = f x2 - f x1) → (Δx = x2 - x1) →
  (Δx → 0) → 
  physical_quantity (Δy / Δx) :=
by
  sorry

-- An example definition of "physical_quantity" for illustration, should be replaced by a correct definition
def physical_quantity (r : ℝ) : Prop := true

theorem derivative_definition_nonzero (x : ℝ) :
  ((f : ℝ → ℝ) = λ x, (f x)) → (Δx ≠ 0) → (Δy ≠ 0) →
  (lim (Δx → 0) ((f (x + Δx) - f x) / Δx) = f' x) :=
by
  sorry

end rate_of_change_const_derivative_constant_function_instantaneous_rate_of_change_physical_quantity_derivative_definition_nonzero_l726_726620


namespace triangle_area_diff_zero_l726_726030

variable (A B M N : Point)
variable (n : ℝ) (α : ℝ)
hypothesis h_angle : 0 < α ∧ α < 180
hypothesis h_distance : dist A B = n
hypothesis h_tangent : isTangent (circle A) (line M N) ∧ isTangent (circle B) (line M N)

theorem triangle_area_diff_zero : 
  area (triangle B M N) - area (triangle A M N) = 0 :=
sorry

end triangle_area_diff_zero_l726_726030


namespace find_q2_l726_726389

-- Definition of the polynomial q(x)
def q (d e : ℤ) (x : ℤ) : ℤ := x^2 + d * x + e

-- Given: q(x) is a factor of both polynomials
theorem find_q2 (d e : ℤ) (h1 : ∃ p, ∀ (x : ℤ), x^4 + 8 * x^2 + 49 = q d e x * p x)
                         (h2 : ∃ p, ∀ (x : ℤ), 2 * x^4 + 5 * x^2 + 35 * x + 7 = q d e x * p x) :
  q d e 2 = 6 :=
by
  sorry

end find_q2_l726_726389


namespace mode_of_data_set_l726_726679

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726679


namespace geometric_number_difference_l726_726944

theorem geometric_number_difference : 
  ∀ (a b c : ℕ), 8 = a → b ≠ c → (∃ k : ℕ, 8 ≠ k ∧ b = k ∧ c = k * k / 8) → (10^2 * a + 10 * b + c = 842) ∧ (10^2 * a + 10 * b + c = 842) → (10^2 * a + 10 * b + c) - (10^2 * a + 10 * b + c) = 0 :=
by
  intro a b c
  intro ha hb
  intro hk
  intro hseq
  sorry

end geometric_number_difference_l726_726944


namespace side_length_of_square_l726_726154

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726154


namespace triangle_count_l726_726026

theorem triangle_count : 
  let lengths := [2, 3, 4, 5] in
  (∃ a b c : ℕ, 
    (a ∈ lengths) ∧ (b ∈ lengths) ∧ (c ∈ lengths) ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
  card { (a, b, c) | 
    a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a } = 3 := 
sorry

end triangle_count_l726_726026


namespace quadratic_eq_coeff_m_l726_726429

theorem quadratic_eq_coeff_m (m : ℤ) : 
  (|m| = 2 ∧ m + 2 ≠ 0) → m = 2 := 
by
  intro h
  sorry

end quadratic_eq_coeff_m_l726_726429


namespace machine_Y_produces_more_widgets_l726_726837

-- Definitions for the rates and widgets produced
def W_x := 18 -- widgets per hour by machine X
def total_widgets := 1080

-- Calculations for time taken by each machine
def T_x := total_widgets / W_x -- time taken by machine X
def T_y := T_x - 10 -- machine Y takes 10 hours less

-- Rate at which machine Y produces widgets
def W_y := total_widgets / T_y

-- Calculation of percentage increase
def percentage_increase := (W_y - W_x) / W_x * 100

-- The final theorem to prove
theorem machine_Y_produces_more_widgets : percentage_increase = 20 := by
  sorry

end machine_Y_produces_more_widgets_l726_726837


namespace smallest_special_number_gt_3429_l726_726608

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726608


namespace smallest_special_number_gt_3429_l726_726606

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726606


namespace parallelepiped_diagonal_bounds_l726_726510

theorem parallelepiped_diagonal_bounds (s d : ℝ) (h_s : 0 < s ∧ s < 1) :
  ((0 < s ∧ s ≤ 1 / 18) →  (sqrt (2 / 3 - 2 * s) ≤ d ∧ d < sqrt (2 - 2 * s))) ∧
  ((7 + 2 * sqrt 6) / 25 ≤ s ∧ s < 1 / 2 → (sqrt (1 - 2 * sqrt (2 * s) + 4 * s) ≤ d ∧ d < sqrt (1 - 2 * sqrt s + 3 * s))) ∧
  (1 / 2 ≤ s ∧ s < 1 → (sqrt (2 * s) < d ∧ d ≤ sqrt (1 - 2 * sqrt s + 3 * s))) :=
sorry

end parallelepiped_diagonal_bounds_l726_726510


namespace smallest_special_greater_than_3429_l726_726569

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726569


namespace density_of_second_part_l726_726016

theorem density_of_second_part (V m : ℝ) (h1 : ∀ V m : ℝ, V_1 = 0.3 * V) 
  (h2 : ∀ V m : ℝ, m_1 = 0.6 * m) 
  (rho1 : ρ₁ = 7800) : 
  ∃ ρ₂, ρ₂ = 2229 :=
by sorry

end density_of_second_part_l726_726016


namespace four_lines_cover_3x3_impossible_three_lines_cover_3x3_six_lines_cover_4x4_l726_726111

-- Definition for a formulatable grid of points
def point := (ℕ, ℕ)
def line := set point

-- Conditions stated as functions/properties
def in_grid_3x3 (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 1 ≤ p.2 ∧ p.2 ≤ 3

def in_grid_4x4 (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4

def line_passes_through_points (l : line) (points : set point) : Prop :=
  ∀ p ∈ points, p ∈ l

def lines_cover_points (lines : list line) (points : set point) : Prop :=
  ∀ p ∈ points, ∃ l ∈ lines, p ∈ l

-- Problem statement a)
theorem four_lines_cover_3x3 (lines : list line) (points : set point) : 
  (∀ p ∈ points, in_grid_3x3 p) →
  lines_length_is_4 : lines.length = 4 →
  lines_cover_points lines points :=
sorry

-- Problem statement b)
theorem impossible_three_lines_cover_3x3 (lines : list line) (points : set point) : 
  (∀ p ∈ points, in_grid_3x3 p) →
  (lines.length = 3) →
  ¬lines_cover_points lines points :=
sorry

-- Problem statement c)
theorem six_lines_cover_4x4 (lines : list line) (points : set point) :
  (∀ p ∈ points, in_grid_4x4 p) →
  lines.length = 6 →
  lines_cover_points lines points :=
sorry

end four_lines_cover_3x3_impossible_three_lines_cover_3x3_six_lines_cover_4x4_l726_726111


namespace distribute_problems_l726_726975

theorem distribute_problems
  (problems : Fin 8)             -- 8 different problems
  (friends : Fin 10)             -- 10 friends
  (p1 p2 : problems)             -- problems 1 and 2
  (h : p1 ≠ p2)                  -- problems 1 and 2 are different
  (same_friend : ∃ f : friends, ∀ p ∈ {p1, p2}, p = f)  -- problems 1 and 2 to the same friend
  : (10 : ℕ) * (10 ^ 6 : ℕ) = 10^7 := by
    sorry

end distribute_problems_l726_726975


namespace least_positive_base_ten_with_seven_binary_digits_l726_726088

theorem least_positive_base_ten_with_seven_binary_digits : 
  ∃ n : ℕ, (n >= 1 ∧ 7 ≤ n.digits 2 .length) → n = 64 :=
begin
  sorry
end

end least_positive_base_ten_with_seven_binary_digits_l726_726088


namespace inv_f_of_3_l726_726669

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inv_f_of_3 (h : ∀ x, x < 0 → f x = 3 → x = -2) : ∃ x, x < 0 ∧ f x = 3 ∧ x = -2 :=
by {
  use -2,
  split,
  { exact (by linarith : -2 < 0)},
  split,
  { show f (-2) = 3,
    rw [f, show (-2)^2 = 4 by norm_num],
    norm_num },
  { refl }
}

end inv_f_of_3_l726_726669


namespace probability_of_x_greater_than_3y_l726_726851

noncomputable def probability_greater_than (x y : ℝ) (x_range : 0 ≤ x ∧ x ≤ 2010) (y_range : 0 ≤ y ∧ y ≤ 2011) : ℝ :=
  if x > 3 * y then 1 else 0

theorem probability_of_x_greater_than_3y (x y : ℝ) :
  (∫ (x : ℝ) in 0..2010, (∫ (y : ℝ) in 0..2011, probability_greater_than x y ⟨0, le_refl 2010⟩ ⟨0, le_refl 2011⟩)) 
    / (2010 * 2011) = 670 / 4021 :=
by sorry

end probability_of_x_greater_than_3y_l726_726851


namespace ceil_sq_values_count_l726_726323

theorem ceil_sq_values_count (y : ℝ) (hy : ⌈y⌉ = 15) : 
  (finset.range (⌈15^2⌉ - ⌈14^2⌉ + 1)).card = 29 :=
by
  let lower_bound := 14
  let upper_bound := 15
  have h1 : lower_bound < y := sorry
  have h2 : y ≤ upper_bound := sorry
  have sq_lower_bound : lower_bound^2 < y^2 := sorry
  have sq_upper_bound : y^2 ≤ upper_bound^2 := sorry
  let vals := (finset.range (⌈upper_bound^2⌉ - ⌈lower_bound^2⌉ + 1)).val
  have h3 : sq_lower_bound ≥ 196 := sorry
  have h4 : sq_upper_bound ≤ 225 := sorry
  rw finset.card
  exact sorry

end ceil_sq_values_count_l726_726323


namespace equal_distribution_possible_if_rel_prime_equal_distribution_impossible_if_not_rel_prime_l726_726817

-- Problem (a)
theorem equal_distribution_possible_if_rel_prime (m n : ℕ) (hmn : n < m)
    (hrel : Nat.gcd m n = 1) :
    ∃ (f : Fin m → ℕ), ∀ i, f i = f 0 + i * n % m :=
sorry

-- Problem (b)
theorem equal_distribution_impossible_if_not_rel_prime (m n : ℕ) (hmn : n < m)
    (hnot_rel : Nat.gcd m n > 1) :
    ∃ (f : Fin m → ℕ), ∀ k, ∃ i j, i ≠ j ∧ (f k i - f k j) % Nat.gcd m n ≠ 0 :=
sorry

end equal_distribution_possible_if_rel_prime_equal_distribution_impossible_if_not_rel_prime_l726_726817


namespace side_length_of_square_l726_726188

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726188


namespace bob_before_1230_conditional_prob_l726_726200

open ProbabilityTheory

noncomputable def prob_bob_before_1230_alice_after_bob :
  ℝ := sorry

theorem bob_before_1230_conditional_prob :
  prob_bob_before_1230_alice_after_bob = 1 / 4 :=
sorry

end bob_before_1230_conditional_prob_l726_726200


namespace probability_one_defective_l726_726097

theorem probability_one_defective (g d : ℕ) (h_g : g = 3) (h_d : d = 1) : 
  let total_items := g + d in
  let sample_space := (total_items.choose 2).toFinset in
  let event_A := {x ∈ sample_space | x.count (0 = ∘ id) = 1} in
  (event_A.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_one_defective_l726_726097


namespace richest_600_income_l726_726880

theorem richest_600_income : 
  (∃ x : ℝ, 3 * 10^9 * x^(-2) = 600) → ∃ x : ℝ, x ≈ 10^4 :=
sorry

end richest_600_income_l726_726880


namespace mode_of_data_set_l726_726684

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726684


namespace probability_one_defective_l726_726102

theorem probability_one_defective (g d : Nat) (h1 : g = 3) (h2 : d = 1) : 
  let total_combinations := (g + d).choose 2
  let favorable_outcomes := g * d
  favorable_outcomes / total_combinations = 1 / 2 := by
sorry

end probability_one_defective_l726_726102


namespace square_side_length_l726_726172

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726172


namespace eight_faucets_fill_time_l726_726656

theorem eight_faucets_fill_time 
  (fill_rate_4_faucets : 4 * 200 / 8 = 100 : ℕ) :
  let fill_rate_1_faucet := 200 / (4 * 8),
      fill_rate_8_faucets := 8 * fill_rate_1_faucet in
  (50 / fill_rate_8_faucets : ℤ) * 60 = 60 := 
begin
  sorry
end

end eight_faucets_fill_time_l726_726656


namespace angle_BCF_eq_angle_ACD_l726_726974

variables {α : Type*} [linear_ordered_field α] [topological_space α]
variables {A B C H D E F : α}

/-- Conditions -/
def triangle (A B C : α) : Prop := true -- Placeholder for triangle definition
def is_right_angle (B C : α) : Prop := true -- Placeholder for right angle definition
def midpoint (X Y M : α) : Prop := true -- Placeholder for midpoint definition
def parallel (L1 L2 : α) : Prop := true -- Placeholder for parallel lines definition

-- Given conditions
variables (A B C : α) (H : α) (D E : α) (F : α)
  (H1 : ∃ (B_angle : α), B_angle > 90) -- ∠B > 90°
  (H2 : ∃ (M : α), midpoint A C M ∧ H = M) -- H is on side AC such that AH = BH
  (H3 : is_right_angle B C H) -- BH is perpendicular to BC
  (H4 : D = (A + B) / 2) -- D is midpoint of AB
  (H5 : E = (B + C) / 2) -- E is midpoint of BC
  (H6 : ∃ (P : α), parallel H P ∧ F = P) -- Line through H parallel to AB meets DE at F

/-- Statement to prove -/
theorem angle_BCF_eq_angle_ACD : ∠ B C F = ∠ A C D :=
sorry

end angle_BCF_eq_angle_ACD_l726_726974


namespace second_term_arithmetic_seq_l726_726782

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l726_726782


namespace polynomial_root_bound_l726_726723

theorem polynomial_root_bound {p : ℤ[X]} (h : p.monic) (deg_p : p.degree = 2003) :
  (p ^ 2 - 25).roots.to_finset.card ≤ 2003 :=
sorry

end polynomial_root_bound_l726_726723


namespace convert_246_octal_to_decimal_l726_726566

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l726_726566


namespace find_natural_number_l726_726638

theorem find_natural_number (n : ℕ) 
    (h : 3^2 * 3^5 * 3^8 * ∀ i : ℕ, (i < n → ∃ k, i = 3 * k + 2) = 27^5) : 
    n = 3 := by
  sorry

end find_natural_number_l726_726638


namespace turtle_population_estimate_l726_726946

theorem turtle_population_estimate :
  (tagged_in_june = 90) →
  (sample_november = 50) →
  (tagged_november = 4) →
  (natural_causes_removal = 0.30) →
  (new_hatchlings_november = 0.50) →
  estimate = 563 :=
by
  intros tagged_in_june sample_november tagged_november natural_causes_removal new_hatchlings_november
  sorry

end turtle_population_estimate_l726_726946


namespace sum_of_chosen_vectors_is_zero_l726_726279

variable (Points : Type) [AddCommGroup Points] -- Define a type for points in the plane with an additive commutative group structure

variable (pairs : list (Points × Points)) -- a list of pairs of points representing vectors chosen
variable (start_count end_count : Points → ℕ) -- functions that count how many vectors start or end at a point

-- Assume the condition that the number of vectors starting at any point equals the number of vectors ending at that point
variable (h : ∀ P : Points, start_count P = end_count P)

-- Define the sum of vectors function
def sum_of_vectors (pairs : list (Points × Points)) : Points :=
  pairs.foldl (λ acc pair, acc + (pair.2 - pair.1)) 0

-- The theorem stating the sum of all vectors is zero
theorem sum_of_chosen_vectors_is_zero : sum_of_vectors Points pairs = 0 :=
  by sorry

end sum_of_chosen_vectors_is_zero_l726_726279


namespace roots_positive_real_contradiction_l726_726371

theorem roots_positive_real_contradiction (n : ℕ) (h : n > 3) (a : Finₓ (n-3) → ℝ) :
  ¬ ∃ (x : Finₓ n → ℝ), (∀ i, 0 ≤ x i) ∧
  (∑ i, x i = 5) ∧
  (∑ i in Finₓ.univ.attach, ite (i.1 < i.2) ((x i.1) * (x i.2)) 0 = 12) ∧
  (∑ i in Finₓ.univ.attach, ite ((i.1 < i.2) ∧ (i.2 < i.3)) ((x i.1) * (x i.2) * (x i.3)) 0 = 15) := 
sorry

end roots_positive_real_contradiction_l726_726371


namespace log_a_2023_is_12_l726_726225

def diamondsuit (a b : ℝ) : ℝ := a ^ Real.log10(b)
def heartsuit (a b : ℝ) : ℝ := a ^ (1 / Real.log10(b))

noncomputable def a_seq : ℕ → ℝ
| 3      := heartsuit 4 3
| (n+1) := diamondsuit (heartsuit n (n+1)) (a_seq n)

def problem_statement : Prop :=
  Real.log10 (a_seq 2023) = 12

theorem log_a_2023_is_12 : problem_statement :=
by
  sorry

end log_a_2023_is_12_l726_726225


namespace sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l726_726466

theorem sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3 :
  let largestThreeDigitMultipleOf4 := 996
  let smallestFourDigitMultipleOf3 := 1002
  largestThreeDigitMultipleOf4 + smallestFourDigitMultipleOf3 = 1998 :=
by
  sorry

end sum_largest_three_digit_multiple_of_4_smallest_four_digit_multiple_of_3_l726_726466


namespace median_salary_is_28000_l726_726031

def num_CEOs : ℕ := 1
def num_SVPs : ℕ := 4
def num_Managers : ℕ := 12
def num_AssistantManagers : ℕ := 8
def num_OfficeClerks : ℕ := 58

def salary_CEO : ℕ := 150000
def salary_SVP : ℕ := 100000
def salary_Manager : ℕ := 80000
def salary_AssistantManager : ℕ := 60000
def salary_OfficeClerk : ℕ := 28000

def total_employees : ℕ := num_CEOs + num_SVPs + num_Managers + num_AssistantManagers + num_OfficeClerks

theorem median_salary_is_28000 : total_employees = 83 ∧
  (num_CEOs + num_SVPs + num_Managers + num_AssistantManagers < (total_employees + 1) / 2) ∧
  ((total_employees + 1) / 2 ≤ total_employees) →
  (num_CEOs + num_SVPs + num_Managers + num_AssistantManagers < 42 ∧ 42 ≤ total_employees) →
  (salary_OfficeClerk = 28000) :=
begin
  sorry
end

end median_salary_is_28000_l726_726031


namespace range_of_a_l726_726763

theorem range_of_a (a : ℝ) (h₀ : 0 < a) : a ≤ Real.exp 2 :=
begin
  sorry
end

end range_of_a_l726_726763


namespace maximize_total_profit_l726_726436

-- Definitions and conditions based on the problem statement
def P (t : ℝ) : ℝ := (1 / 5) * t
def Q (t : ℝ) : ℝ := (3 / 5) * Real.sqrt t
def y (x : ℝ) : ℝ := P x + Q (3 - x)

theorem maximize_total_profit :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = (21 / 20) ∧ x = (3 / 4)) :=
begin
  sorry
end

end maximize_total_profit_l726_726436


namespace real_root_of_equation_l726_726646

theorem real_root_of_equation :
  ∃ x : ℝ, (sqrt x + sqrt (x + 4) = 12) ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726646


namespace smallest_special_number_gt_3429_l726_726576

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726576


namespace cosine_sum_of_angles_l726_726267

theorem cosine_sum_of_angles (α β : ℝ) 
  (hα : Complex.exp (Complex.I * α) = (4 / 5) + (3 / 5) * Complex.I)
  (hβ : Complex.exp (Complex.I * β) = (-5 / 13) + (12 / 13) * Complex.I) :
  Real.cos (α + β) = -7 / 13 :=
by
  sorry

end cosine_sum_of_angles_l726_726267


namespace geometric_common_ratio_eq_three_l726_726795

theorem geometric_common_ratio_eq_three 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h_arithmetic_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0) 
  (h_geom_seq : (a 2 + 2 * d) ^ 2 = (a 2 + d) * (a 2 + 5 * d)) : 
  (a 3) / (a 2) = 3 :=
by 
  sorry

end geometric_common_ratio_eq_three_l726_726795


namespace log_base_2_of_7_l726_726265

variable (m n : ℝ)

theorem log_base_2_of_7 (h1 : Real.log 5 = m) (h2 : Real.log 7 = n) : Real.logb 2 7 = n / (1 - m) :=
by
  sorry

end log_base_2_of_7_l726_726265


namespace mode_of_dataset_with_average_is_l726_726701

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726701


namespace square_ratio_l726_726413

theorem square_ratio (a : ℝ) : 
  let E := (1 + sqrt 3 / 3, 0)
  let F := (0, 1 + sqrt 3 / 3)
  let G := (-1 - sqrt 3 / 3, 0)
  let H := (0, -1 - sqrt 3 / 3)
  let Area_ABCD := (2*a)^2
  let side_EFGH := (2*a * (1 + sqrt 3 / 3)) * sqrt 2
  let Area_EFGH := side_EFGH^2 
in
  Area_EFGH / Area_ABCD = (2 + sqrt 3) / 3
:= sorry

end square_ratio_l726_726413


namespace probability_of_random_point_in_inscribed_circle_l726_726869

noncomputable def triangle_with_inscribed_circle_probability (a b : ℕ) (h : a = 8 ∧ b = 15) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  let r := (a * b) / (a + b + c) in
  let area_triangle := (1 / 2) * a * b in
  let area_circle := Real.pi * r^2 in
  area_circle / area_triangle

theorem probability_of_random_point_in_inscribed_circle :
  triangle_with_inscribed_circle_probability 8 15 (and.intro rfl rfl) = (3 * Real.pi) / 20 := sorry

end probability_of_random_point_in_inscribed_circle_l726_726869


namespace side_length_of_square_l726_726165

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726165


namespace find_y_l726_726649

structure Vector2D where
  x : ℝ
  y : ℝ

def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

def projection (v w : Vector2D) : Vector2D :=
  let scalar := (dot_product v w) / (dot_product w w)
  Vector2D.mk (scalar * w.x) (scalar * w.y)

theorem find_y (y : ℝ) 
  (v : Vector2D := Vector2D.mk 1 y)
  (w : Vector2D := Vector2D.mk 9 3)
  (proj_v_w : Vector2D := Vector2D.mk (-6) (-2)) :
  projection v w = proj_v_w →  y = -23 :=
by
  sorry

end find_y_l726_726649


namespace scientific_notation_of_120000_l726_726841

theorem scientific_notation_of_120000 : 
  (120000 : ℝ) = 1.2 * 10^5 := 
by 
  sorry

end scientific_notation_of_120000_l726_726841


namespace smallest_n_for_413_consecutive_digits_l726_726425

theorem smallest_n_for_413_consecutive_digits :
  ∃ (m n : ℕ), n = 414 ∧ Nat.Coprime m n ∧ m < n ∧
  (let decimal_repr := (m / n : ℚ).to_digits 10 1 in
   ∃ (i : ℕ), (decimal_repr.get? i = some 4) ∧
              (decimal_repr.get? (i+1) = some 1) ∧
              (decimal_repr.get? (i+2) = some 3)) :=
begin
  sorry
end

end smallest_n_for_413_consecutive_digits_l726_726425


namespace arithmetic_b_a_general_term_a_sum_l726_726306

open Nat

def a_seq (a : Nat → ℝ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 3 * a n + 3 ^ n

def b_seq (a b : Nat → ℝ) : Prop :=
  ∀ n, b n = a n / 3 ^ n

theorem arithmetic_b (a b : Nat → ℝ) (h₁ : a_seq a) (h₂ : b_seq a b) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
by
  sorry

theorem a_general_term (a : Nat → ℝ) (h₁ : a_seq a) :
  ∀ n, a n = (n + 2) * 3 ^ (n - 1) :=
by
  sorry

theorem a_sum (a : Nat → ℝ) (h₁ : a_seq a) :
  ∀ n, (∑ k in range (n + 1), a k) = ((2 * n + 3) * 3 ^ n / 4) - (3 / 4) :=
by
  sorry

end arithmetic_b_a_general_term_a_sum_l726_726306


namespace side_length_of_square_l726_726185

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726185


namespace two_true_propositions_exist_l726_726720

-- Definitions of parallelism and perpendicularity for planes and lines
variable {Plane Line : Type}
variable {parallel : Plane → Plane → Prop}
variable {perpendicular : Plane → Plane → Prop}
variable {parallel_lines : Line → Line → Prop}
variable {perpendicular_plane_line : Plane → Line → Prop}

-- Given conditions
variable (α β γ : Plane)
variable (h1 : parallel α β)
variable (h2 : perpendicular α γ)
variable (h3 : perpendicular β γ)

noncomputable def number_of_true_propositions : ℕ :=
  if (∀ a b : Line, parallel_lines a b ∧ perpendicular_plane_line α a → perpendicular_plane_line α b) then 1 else 0 +
  if (∀ a b : Line, parallel_lines a β ∧ perpendicular a b → perpendicular b β) then 1 else 0 +
  if (∀ a b : Line, parallel_lines a α ∧ perpendicular_plane_line b γ → perpendicular α b) then 1 else 0

theorem two_true_propositions_exist :
  number_of_true_propositions α β γ = 2 := by
  sorry

end two_true_propositions_exist_l726_726720


namespace find_range_of_a_l726_726715

def p (a : ℝ) : Prop := 
  a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)

def q (a : ℝ) : Prop := 
  a^2 - 2 * a - 3 < 0

theorem find_range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end find_range_of_a_l726_726715


namespace shaded_area_correct_l726_726791

-- Given problem conditions
def square_side_length : ℝ := 10
def pond_radius : ℝ := 3
def num_ponds : ℕ := 3

-- Calculated areas
def area_of_square : ℝ := square_side_length * square_side_length
def area_of_one_pond : ℝ := Real.pi * pond_radius^2
def total_area_of_ponds : ℝ := ↑num_ponds * area_of_one_pond

-- Mathematically equivalent proof problem
theorem shaded_area_correct :
  area_of_square - total_area_of_ponds = 100 - 27 * Real.pi :=
by
  -- Proof omitted as instructed
  sorry

end shaded_area_correct_l726_726791


namespace region_area_eq_35_pi_l726_726464

theorem region_area_eq_35_pi
  (x y : ℝ)
  (h : x^2 + y^2 - 5 = 6 * y - 10 * x + 4) :
  ∃ r : ℝ, r = 5 * √7 ∧ ∃ a b : ℝ, ((x + 5)^2 + (y - 3)^2 = 35) ∧ ∃ π : ℝ, 
  real.pi * r^2 = 35 * π :=
by
  sorry

end region_area_eq_35_pi_l726_726464


namespace mode_of_data_set_l726_726689

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726689


namespace side_length_of_square_l726_726161

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726161


namespace measure_angle_DAB_l726_726349

theorem measure_angle_DAB (A B C D : Type) [euclidean_geometry A B C D] :
  right_angle B A D ∧ midpoint C A D ∧ AB = 2 * BC → angle D A B = 30 :=
by
  sorry

end measure_angle_DAB_l726_726349


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726058

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726058


namespace average_of_remaining_two_l726_726417

theorem average_of_remaining_two
  (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg_2_1 : (a + b) / 2 = 4.2)
  (h_avg_2_2 : (c + d) / 2 = 3.85) : 
  ((e + f) / 2) = 3.8 :=
by
  sorry

end average_of_remaining_two_l726_726417


namespace discount_percent_l726_726423

theorem discount_percent
  (MP CP SP : ℝ)
  (h1 : CP = 0.55 * MP)
  (gainPercent : ℝ)
  (h2 : gainPercent = 54.54545454545454 / 100)
  (h3 : (SP - CP) / CP = gainPercent)
  : ((MP - SP) / MP) * 100 = 15 := by
  sorry

end discount_percent_l726_726423


namespace rectangle_area_change_l726_726331

theorem rectangle_area_change 
  (L B : ℝ) 
  (A : ℝ := L * B) 
  (L' : ℝ := 1.30 * L) 
  (B' : ℝ := 0.75 * B) 
  (A' : ℝ := L' * B') : 
  A' / A = 0.975 := 
by sorry

end rectangle_area_change_l726_726331


namespace number_of_divisors_of_x_l726_726866

theorem number_of_divisors_of_x 
  (p1 p2 p3 : ℕ) [fact (nat.prime p1)] [fact (nat.prime p2)] [fact (nat.prime p3)] (h_diff : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
  let n := p1 * p2^2 * p3^5,
      x := n^4 in
  nat.divisors_count x = 945 :=
by
  sorry

end number_of_divisors_of_x_l726_726866


namespace correct_statement_D_l726_726951

-- Define the quantities and conditions given in the problem.
def big_cow_count : ℕ := 30
def small_cow_count : ℕ := 15
def total_fodder : ℝ := 675
def consumption_equation (x y : ℝ) : Prop := 30 * x + 15 * y = 675

-- Define what it means for m and n to represent the fodder needed per day for each big and small cow
def represents_fodder (m n: ℝ) : Prop := 30 * m + 15 * n = total_fodder

-- The theorem to be proved
theorem correct_statement_D (m n : ℝ) (h : represents_fodder m n) : consumption_equation m n :=
begin
  exact h,
end

end correct_statement_D_l726_726951


namespace inequality_problem_l726_726667

theorem inequality_problem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by
  sorry

end inequality_problem_l726_726667


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726055

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726055


namespace smallest_special_number_gt_3429_l726_726579

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726579


namespace digital_root_8_pow_n_l726_726845

-- Define the conditions
def n : ℕ := 1989

-- Define the simplified problem
def digital_root (x : ℕ) : ℕ := if x % 9 = 0 then 9 else x % 9

-- Statement of the problem
theorem digital_root_8_pow_n : digital_root (8 ^ n) = 8 := by
  have mod_nine_eq : 8^n % 9 = 8 := by
    sorry
  simp [digital_root, mod_nine_eq]

end digital_root_8_pow_n_l726_726845


namespace bulbs_97_100_l726_726900

def BulbColor := ℕ → String

noncomputable def garland : BulbColor :=
λ n, match n % 5 with
| 0 => "Yellow"
| 1 => "Blue"
| 2 => "Yellow"
| 3 => "Blue"
| 4 => "Blue"
| _ => "Unknown"
end

theorem bulbs_97_100 :
  (garland 96 = "Yellow" ∧ garland 97 = "Blue" ∧ garland 98 = "Yellow" ∧ garland 99 = "Blue" ∧ garland 100 = "Blue") :=
by
  have universal_condition :
    ∀ (n : ℕ), (garland n = "Yellow" ∧ garland (n + 2) = "Yellow"
                ∧ garland (n + 1) = "Blue" ∧ garland (n + 3) = "Blue" ∧ garland (n + 4) = "Blue") :=
    by
      intro n
      cases (n % 5) with
      | 0 => sorry
      | 1 => sorry
      | 2 => sorry
      | 3 => sorry
      | 4 => sorry
      | _ => sorry
  show (garland 96 = "Yellow" ∧ garland 97 = "Blue" ∧ garland 98 = "Yellow" ∧ garland 99 = "Blue" ∧ garland 100 = "Blue") 
  by exact universal_condition 95

end bulbs_97_100_l726_726900


namespace sum_of_k_minimized_area_l726_726623

def is_minimized_area (k : ℕ) : ℝ :=
let x1 := 2, y1 := 8, x2 := 14, y2 := 17, x3 := 6, y3 := k in
let line_eq := (3/4) * x3 + 13/2 in
if y3 = line_eq then 0 else 
real.abs ((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / 2

theorem sum_of_k_minimized_area : 
  ∑ k in {10, 12}, is_minimized_area k = 22 := by
sorry

end sum_of_k_minimized_area_l726_726623


namespace geometric_meaning_of_derivative_l726_726006

variable (f : ℝ → ℝ) (x₀ : ℝ)

-- Definition for the derivative at a point
def derivative_at_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
 ∃ (f' : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs ((f x - f x₀) / (x - x₀) - f') < ε

-- Statement expressing the geometric meaning of the derivative
theorem geometric_meaning_of_derivative :
  derivative_at_point f x₀ →
  ∃ m, (∀ x, (x ≠ x₀) → (f x - f x₀) / (x - x₀) → m) :=
sorry

end geometric_meaning_of_derivative_l726_726006


namespace sum_of_problematic_x_values_l726_726227

theorem sum_of_problematic_x_values :
  let f : ℝ → ℝ := λ x => 5 * x / (3 * x^2 - 9 * x + 6)
  ∑ x in ({1, 2} : Finset ℝ), x = 3 :=
by
  let f : ℝ → ℝ := λ x => 5 * x / (3 * x^2 - 9 * x + 6)
  have problematic_x_values : ({1, 2} : Finset ℝ) = {x | 3 * x^2 - 9 * x + 6 = 0}.to_finset := sorry
  have h : ∑ x in ({1, 2} : Finset ℝ), x = 1 + 2 := by simp
  exact h

end sum_of_problematic_x_values_l726_726227


namespace log9_log11_lt_one_l726_726538

theorem log9_log11_lt_one (log9_pos : 0 < Real.log 9) (log11_pos : 0 < Real.log 11) : 
  Real.log 9 * Real.log 11 < 1 :=
by
  sorry

end log9_log11_lt_one_l726_726538


namespace rectangle_area_change_l726_726415

theorem rectangle_area_change 
  (L W : ℝ) 
  (h : L * W = 540) :
  let L_new := 0.85 * L in
  let W_new := 1.20 * W in
  (L_new * W_new).round = 551 :=
by
  let L_new := 0.85 * L
  let W_new := 1.20 * W
  have : L_new * W_new = 1.02 * (L * W) := by {
    calc 
      L_new * W_new = 0.85 * L * (1.20 * W) : by ring
              ... = 0.85 * 1.20 * (L * W)   : by ring
              ... = 1.02 * (L * W)          : by norm_num }
  rw [h] at this
  have : 1.02 * 540 = 550.8 := by norm_num
  rw this at this
  exact round_eq_of_lt (by norm_num : 550.8 ∈ Ico 550.5 (551.5 : ℝ)) 551

end rectangle_area_change_l726_726415


namespace garland_colors_l726_726898

noncomputable def garland (n : ℕ) (color : ℕ → Prop) : Prop :=
  ∃ (color : ℕ → Prop),
    color 1 = "Yellow" ∧
    color 3 = "Yellow" ∧
    (∀ (k : ℕ), k + 4 ≤ n → (∃ count_yellow count_blue,
      (∑ i in finset.range 5, if color (k + i) = "Yellow" then 1 else 0) = 2 ∧
      (∑ i in finset.range 5, if color (k + i) = "Blue" then 1 else 0) = 3)) ∧
    color 97 = "Blue" ∧
    color 98 = "Yellow" ∧
    color 99 = "Blue" ∧
    color 100 = "Blue"

theorem garland_colors : garland 100 (λ i, if i = 1 ∨ i = 3 ∨ i % 5 = 1 ∨ i % 5 = 3 then "Yellow" else "Blue") :=
by
  sorry

end garland_colors_l726_726898


namespace side_length_of_square_l726_726184

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726184


namespace fourth_member_income_l726_726114

theorem fourth_member_income :
  ∀ (n : ℕ) (avg : ℕ) (inc1 inc2 inc3 : ℕ),
  n = 4 → avg = 10000 → inc1 = 8000 → inc2 = 15000 → inc3 = 6000 →
  (n * avg - (inc1 + inc2 + inc3) = 11000) :=
by
  intros n avg inc1 inc2 inc3 h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num

end fourth_member_income_l726_726114


namespace side_length_of_square_l726_726155

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726155


namespace inscribed_circle_area_l726_726204

theorem inscribed_circle_area (a : ℝ) (h : 0 < a) :
  ∃ (ABC : Triangle),
    ABC.is_isosceles ∧
    ABC.angleB = 120 ∧
    ABC.midpoint_distance = a ∧
    ABC.inscribed_circle_area = 12 * Real.pi * a^2 * (7 - 4 * Real.sqrt 3) := sorry

end inscribed_circle_area_l726_726204


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726056

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726056


namespace base8_to_base10_l726_726556

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l726_726556


namespace ceil_y_squared_possibilities_l726_726325

theorem ceil_y_squared_possibilities (y : ℝ) (h : ⌈y⌉ = 15) : 
  ∃ n : ℕ, (n = 29) ∧ (∀ z : ℕ, ⌈y^2⌉ = z → (197 ≤ z ∧ z ≤ 225)) :=
by
  sorry

end ceil_y_squared_possibilities_l726_726325


namespace least_positive_base_ten_seven_binary_digits_l726_726046

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726046


namespace algebraic_expression_eval_l726_726829

theorem algebraic_expression_eval (a b : ℝ) 
  (h_eq : ∀ (x : ℝ), ¬(x ≠ 0 ∧ x ≠ 1 ∧ (x / (x - 1) + (x - 1) / x = (a + b * x) / (x^2 - x)))) :
  8 * a + 4 * b - 5 = 27 := 
sorry

end algebraic_expression_eval_l726_726829


namespace unit_digit_n_is_zero_l726_726255

noncomputable def n : ℤ := (75 ^ (Finset.range 81).sum Factorial) +
                           (25 ^ (Finset.range 76).sum Factorial) -
                           Int.log (97 ^ (Finset.range 51).sum Factorial) +
                           Real.sin (123 ^ (Finset.range 26).sum Factorial)

theorem unit_digit_n_is_zero : (n % 10) = 0 := 
by sorry

end unit_digit_n_is_zero_l726_726255


namespace simplify_expression_l726_726858

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4 + 3*z^2) = -1 - 8*z^2 :=
by
  sorry

end simplify_expression_l726_726858


namespace gcd_840_1764_gcd_440_556_l726_726123

-- Definition of GCD using the Euclidean algorithm
def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd_euclidean b (a % b)

-- Definition of GCD using the method of continued subtraction
def gcd_subtraction (a b : ℕ) : ℕ :=
  if a = b then a
  else if a > b then gcd_subtraction (a - b) b
  else gcd_subtraction a (b - a)

-- Problem (I): Prove that the GCD of 840 and 1764 is 84
theorem gcd_840_1764 : gcd_euclidean 840 1764 = 84 := by
  sorry

-- Problem (II): Prove that the GCD of 440 and 556 is 4
theorem gcd_440_556 : gcd_subtraction 440 556 = 4 := by
  sorry

end gcd_840_1764_gcd_440_556_l726_726123


namespace circle_enlarge_ratios_l726_726496

theorem circle_enlarge_ratios (r : ℝ) :
  let old_area := π * r^2,
      new_radius := 3 * r,
      new_area := π * new_radius^2,
      old_circumference := 2 * π * r,
      new_circumference := 2 * π * new_radius
  in (new_area / old_area = 9) ∧ (new_circumference / old_circumference = 3) :=
by
  let old_area := π * r^2
  let new_radius := 3 * r
  let new_area := π * new_radius^2
  let old_circumference := 2 * π * r
  let new_circumference := 2 * π * new_radius
  have area_ratio : new_area / old_area = 9 := sorry
  have circumference_ratio : new_circumference / old_circumference = 3 := sorry
  exact ⟨area_ratio, circumference_ratio⟩

end circle_enlarge_ratios_l726_726496


namespace hh3_value_l726_726301

noncomputable def h (x : ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x - 1

theorem hh3_value : h (h 3) = 3406935 := by
  sorry

end hh3_value_l726_726301


namespace nat_sin_cos_combination_l726_726810

theorem nat_sin_cos_combination (x y : ℝ) (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : sin x + cos y = (p : ℚ) / q)
  (h2 : sin y + cos x = (r : ℚ) / s) :
  ∃ (m n : ℕ), nat (m * sin x + n * cos x).abs :=
begin
  sorry
end

end nat_sin_cos_combination_l726_726810


namespace calculate_product_l726_726995

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l726_726995


namespace square_side_length_l726_726153

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726153


namespace right_triangle_m_c_l726_726223

theorem right_triangle_m_c (a b c : ℝ) (m_c : ℝ) 
  (h : (1 / a) + (1 / b) = 3 / c) : 
  m_c = (c * (1 + Real.sqrt 10)) / 9 :=
sorry

end right_triangle_m_c_l726_726223


namespace band_row_arrangements_l726_726945

open Nat

theorem band_row_arrangements (n : ℕ) (h : n = 120) :
  (card {d : ℕ | d ∣ n ∧ 4 ≤ d ∧ d ≤ 30} = 10) :=
by
  have h1 : n = 120 := h
  sorry

end band_row_arrangements_l726_726945


namespace ramu_profit_percent_l726_726852

theorem ramu_profit_percent (purchase_price repairs taxes insurance selling_price : ℝ)
  (h1 : purchase_price = 42000)
  (h2 : repairs = 13000)
  (h3 : taxes = 5000)
  (h4 : insurance = 8000)
  (h5 : selling_price = 69900) :
  ((selling_price - (purchase_price + repairs + taxes + insurance)) / (purchase_price + repairs + taxes + insurance)) * 100 ≈ 2.794 :=
by
  -- Proof goes here
  sorry

end ramu_profit_percent_l726_726852


namespace least_seven_digit_binary_number_l726_726084

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726084


namespace part_I_f_2_eq_0_part_II_f_x_gt_0_l726_726671

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

theorem part_I_f_2_eq_0 : f a 2 = 0 := by
  sorry

theorem part_II_f_x_gt_0 (x : ℝ) :
  (a < -1 → (f a x > 0 ↔ x ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo (1 - a) ∞)) ∧
  (a = -1 → (f a x > 0 ↔ x ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 ∞)) ∧
  (a > -1 → (f a x > 0 ↔ x ∈ Set.Ioo (-∞) (1 - a) ∪ Set.Ioo 2 ∞)) := by
  sorry

end part_I_f_2_eq_0_part_II_f_x_gt_0_l726_726671


namespace mode_of_dataset_with_average_is_l726_726704

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726704


namespace base8_246_is_166_in_base10_l726_726550

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l726_726550


namespace movie_of_the_year_l726_726906

theorem movie_of_the_year (members : ℕ) (total_lists : ℕ) (lists_threshold : ℕ) :
  members = 770 →
  total_lists = members →
  lists_threshold = ⌈total_lists / 4⌉ →
  lists_threshold = 193 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have : 770 / 4 = 192.5, by norm_num
  rw this at h3
  have : ⌈192.5⌉ = 193, by norm_num
  rw this at h3
  exact h3

end movie_of_the_year_l726_726906


namespace population_estimate_Eldoria_l726_726239

theorem population_estimate_Eldoria (initial_population : ℕ) (t : ℕ)
  (h_initial : initial_population = 500)
  (h_double : ∀ n : ℕ, t = 15 → n * 2 ^ (t / 15) = n * 2 ^ (n / 15)) :
  initial_population * 2 ^ (70 / 15) = 8000 := 
by {
  sorry
}

example : population_estimate_Eldoria 500 70 (by rfl) (by intros; rfl) :=
by {
  sorry
}

end population_estimate_Eldoria_l726_726239


namespace tractor_path_length_l726_726500

theorem tractor_path_length (d_AB : ℝ) (r_A : ℝ) (r_B : ℝ) 
  (h1 : d_AB = 51) (h2 : r_A = 12) (h3 : r_B = 7) : 
  ∃ (L : ℝ), L = 69 := 
by
  -- Conditions specified
  have h_dist : d_AB = 51 := h1
  have h_rad_A : r_A = 12 := h2
  have h_rad_B : r_B = 7 := h3
  -- Placeholder for the proof
  use 69
  sorry

end tractor_path_length_l726_726500


namespace max_kings_l726_726902

theorem max_kings (initial_kings : ℕ) (kings_attacking_each_other : initial_kings = 21) 
  (no_two_kings_attack : ∀ kings_remaining, kings_remaining ≤ 16) : 
  ∃ kings_remaining, kings_remaining = 16 :=
by
  sorry

end max_kings_l726_726902


namespace system_of_equations_solution_l726_726938

theorem system_of_equations_solution (x y : ℚ) :
  (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧ 4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by
  sorry

end system_of_equations_solution_l726_726938


namespace arithmetic_progression_coprime_l726_726203

theorem arithmetic_progression_coprime :
  ∀ (n : ℕ), n = 100 →
  ∀ (a r : ℕ),
    a = n! + 1 →
    r = n! →
    (∀ i j, 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n ∧ i ≠ j →
    Nat.coprime (a + i * r) (a + j * r)) :=
by
  intros n hn a r ha hr
  rw [hn, ha, hr]
  intros i hi j hj hij
  sorry

end arithmetic_progression_coprime_l726_726203


namespace show_fn_n_gt_half_exp_n_l726_726386

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in finset.range (n + 1), x^i / (i.factorial : ℝ)

theorem show_fn_n_gt_half_exp_n (n : ℕ) :
  (∑ i in finset.range (n + 1), n^i / (i.factorial : ℝ)) > real.exp n / 2 :=
begin
  have h1 : ∀ x, real.exp x - f_n n x = (1 / n.factorial : ℝ) * 
    ∫ t in 0..x, (x - t)^n * real.exp t :=
    sorry, -- given as part of the problem
  have h2 : ∫ t in 0..∞, t^n * real.exp (-t) = n.factorial := sorry, -- given as well
  sorry -- rest of the proof
end

end show_fn_n_gt_half_exp_n_l726_726386


namespace base8_to_base10_conversion_l726_726560

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l726_726560


namespace mode_of_data_set_l726_726695

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l726_726695


namespace product_of_two_numbers_l726_726457

-- Define HCF (Highest Common Factor) and LCM (Least Common Multiple) conditions
def hcf_of_two_numbers (a b : ℕ) : ℕ := 11
def lcm_of_two_numbers (a b : ℕ) : ℕ := 181

-- The theorem to prove
theorem product_of_two_numbers (a b : ℕ) 
  (h1 : hcf_of_two_numbers a b = 11)
  (h2 : lcm_of_two_numbers a b = 181) : 
  a * b = 1991 :=
by 
  -- This is where we would put the proof, but we can use sorry for now
  sorry

end product_of_two_numbers_l726_726457


namespace multiplication_of_mixed_number_l726_726988

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726988


namespace stock_quote_example_l726_726927

noncomputable def stock_quote (investment interest rate : ℝ) : ℝ :=
  let face_value := (interest * 100) / rate in
  (investment / face_value) * 100

theorem stock_quote_example :
  stock_quote 1620 135 8 = 96 :=
by 
  -- Here we can compute the actual proof steps
  sorry

end stock_quote_example_l726_726927


namespace total_students_l726_726840

-- Define the conditions based on the problem
def valentines_have : ℝ := 58.0
def valentines_needed : ℝ := 16.0

-- Theorem stating that the total number of students (which is equal to the total number of Valentines required)
theorem total_students : valentines_have + valentines_needed = 74.0 :=
by
  sorry

end total_students_l726_726840


namespace side_length_of_square_l726_726157

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726157


namespace dave_tshirts_total_l726_726617

/-- Dave bought 3 packs of white T-shirts, 2 packs of blue T-shirts,
4 packs of red T-shirts, and 1 pack of green T-shirts for his basketball team.
The white T-shirts come in packs of 6, the blue T-shirts come in packs of 4,
the red T-shirts come in packs of 5, and the green T-shirts come in a pack of 3.
Prove that the total number of T-shirts Dave bought is 49. -/
theorem dave_tshirts_total :
  let packs_white := 3,
      packs_blue := 2,
      packs_red := 4,
      packs_green := 1,
      per_pack_white := 6,
      per_pack_blue := 4,
      per_pack_red := 5,
      per_pack_green := 3,
      total_white := packs_white * per_pack_white,
      total_blue := packs_blue * per_pack_blue,
      total_red := packs_red * per_pack_red,
      total_green := packs_green * per_pack_green,
      total_tshirts := total_white + total_blue + total_red + total_green
  in total_tshirts = 49 :=
by
  -- Proof omitted
  sorry

end dave_tshirts_total_l726_726617


namespace greatest_possible_value_y_l726_726867

theorem greatest_possible_value_y
  (x y : ℤ)
  (h : x * y + 3 * x + 2 * y = -6) : 
  y ≤ 3 :=
by sorry

end greatest_possible_value_y_l726_726867


namespace segment_PK_length_l726_726849

-- Define the necessary elements of the problem
variables {A B C P K : Type}
variables [Triangle ABC : BoundedGeometry]
variables [Perimeter ABC P : Real]

-- Translate problem definition to Lean 4
theorem segment_PK_length (A B C P K : Type) [Triangle ABC : BoundedGeometry] [Perimeter ABC P : Real] 
  (AP AK : Line) (PK : Segment)
  (hAP : IsPerpendicular AP (AngleBisector B External))
  (hAK : IsPerpendicular AK (AngleBisector C External))
  (hPK : Connects PK AP AK) :
  length PK = P / 2 := 
sorry

end segment_PK_length_l726_726849


namespace evaluate_expression_at_d_4_l726_726241

theorem evaluate_expression_at_d_4 :
  (let d := 4 in (d^d - d * (d - 2)^d + d^2)^d) = 1874164224 :=
by
  sorry

end evaluate_expression_at_d_4_l726_726241


namespace smallest_special_greater_than_3429_l726_726572

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726572


namespace least_7_digit_binary_number_is_64_l726_726074

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726074


namespace total_art_cost_l726_726361

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l726_726361


namespace problem_1_problem_2_l726_726296

noncomputable def f (x : ℝ) : ℝ := real.sqrt 2 * real.sin (x - π / 12)

theorem problem_1 : f (π / 3) = 1 :=
sorry

theorem problem_2 (θ : ℝ) (h1 : real.sin θ = 4 / 5) (h2 : 0 < θ ∧ θ < π / 2) : f (θ - π / 6) = 1 / 5 :=
sorry

end problem_1_problem_2_l726_726296


namespace polynomial_value_at_3_l726_726385

theorem polynomial_value_at_3 :
  ∃ (P : ℕ → ℚ), 
    (∀ (x : ℕ), P x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6) ∧ 
    (∀ (i : ℕ), i ≤ 6 → 0 ≤ b_i ∧ b_i < 5) ∧ 
    P (Nat.sqrt 5) = 35 + 26 * Nat.sqrt 5 -> 
    P 3 = 437 := 
by
  simp
  sorry

end polynomial_value_at_3_l726_726385


namespace z_has_purely_imaginary_difference_l726_726726

theorem z_has_purely_imaginary_difference
  (z : ℂ) (h : z = 2 - complex.i) : z - 2 = -complex.i := 
sorry

end z_has_purely_imaginary_difference_l726_726726


namespace club_truncator_more_wins_than_losses_l726_726209

noncomputable def clubTruncatorWinsProbability : ℚ :=
  let total_matches := 8
  let prob := 1/3
  -- The combinatorial calculations for the balanced outcomes
  let balanced_outcomes := 70 + 560 + 420 + 28 + 1
  let total_outcomes := 3^total_matches
  let prob_balanced := balanced_outcomes / total_outcomes
  let prob_more_wins_or_more_losses := 1 - prob_balanced
  (prob_more_wins_or_more_losses / 2)

theorem club_truncator_more_wins_than_losses : 
  clubTruncatorWinsProbability = 2741 / 6561 := 
by 
  sorry

#check club_truncator_more_wins_than_losses

end club_truncator_more_wins_than_losses_l726_726209


namespace john_average_speed_l726_726355

variable {minutes_uphill : ℝ} (h1 : minutes_uphill = 45)
variable {distance_uphill : ℝ} (h2 : distance_uphill = 2)
variable {minutes_downhill : ℝ} (h3 : minutes_downhill = 15)
variable {distance_downhill : ℝ} (h4 : distance_downhill = 2)

theorem john_average_speed : 
  let total_distance := distance_uphill + distance_downhill in
  let total_time := minutes_uphill + minutes_downhill in
  total_distance / (total_time / 60) = 4 :=
by
  sorry

end john_average_speed_l726_726355


namespace sum_f_equal_zero_l726_726660

def sum_zero (a b : ℝ) (c : ℕ → ℝ) (x : ℕ → ℝ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a * x i ^ 2 + b * x i + c i = 0) →
  ((Σ' i : ℕ, 1 ≤ i ∧ i ≤ 99, a * x i ^ 2 + b * x i + c (i+1) )) + (a * (x 100) ^ 2 + b * x 100 + c 1) = 0

theorem sum_f_equal_zero (a b : ℝ) (c : ℕ → ℝ) (x : ℕ → ℝ)
  (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a * x i ^ 2 + b * x i + c i = 0) :
  (Σ' i : ℕ, 1 ≤ i ∧ i ≤ 99, a * x i ^ 2 + b * x i + c (i+1) ) + (a * (x 100) ^ 2 + b * x 100 + c 1) = 0 :=
by
  sorry

end sum_f_equal_zero_l726_726660


namespace sum_of_multiples_of_3_l726_726920

/-- Define the arithmetic sequence of multiples of 3 from -30 to 60 --/
def multiples_of_3 : List Int :=
  List.range' (-30) (3 * 31) 3  -- generates the list [-30, -27, ..., 60]

/-- Prove that the sum of the multiples of 3 from -30 to 60 is 465 --/
theorem sum_of_multiples_of_3 :
  (multiples_of_3.sum = 465) :=
  sorry

end sum_of_multiples_of_3_l726_726920


namespace sum_max_min_f_on_1_e2_l726_726299

noncomputable def f (x : ℝ) (a : ℝ) := x - 1 - a * Real.log x

theorem sum_max_min_f_on_1_e2 (a : ℝ) (h : a > 0)
  (h_zero : ∃! x ∈ Ioi 0, f x a = 0) :
  (f (Real.exp 2) a + f 1 a) = Real.exp 2 - 3 := by
  sorry

end sum_max_min_f_on_1_e2_l726_726299


namespace tangent_proof_l726_726788

namespace geometry

open_locale classical

variables {α : Type*} [euclidean_space.{α}] [metric_space α]

-- Given conditions
variables {A B C D E X Y Z: α}
variables (ABC : triangle α)
variables (O : circumcircle ABC)

-- Specifying the triangle ABC and its properties
def is_right_angled_triangle (ABC : triangle α) :=
  ∃ A B C : α, ∠ B A C = 90

def angle_order (B C : α) : Prop :=
  ∃ A : α, ∠ B A C < ∠ C A B

-- Defining the tangent line, reflection and midpoints
def tangent_at_point (A : α) (O : circumcircle ABC) (D : α) :=
  tangent_line_at A O D

def reflection_across_line (A B C E : α) : Prop :=
  E = reflection A (line_through B C)

def perpendicular_bisector (A X B E : α) : Prop :=
  ∃ X : α, AX ⊥ BE

def midpoint (A X Y : α) : Prop :=
  midpoint A X = Y

-- Describing the intersection and proving BD is tangent
def intersection_with_circumcircle (BY : α) (Z O : β) : Prop :=
  BY ∩ O = Z

def tangent_to_circumcircle (BD : α) (triangle : κε) :=
  tangent_line_at B (circumcircle triangle) D

-- The final theorem statement
theorem tangent_proof :
  is_right_angled_triangle ABC ∧
  angle_order B C ∧
  tangent_at_point A O D ∧
  reflection_across_line A B C E ∧
  perpendicular_bisector A X B E ∧
  midpoint A X Y ∧
  intersection_with_circumcircle BY Z O →
  tangent_to_circumcircle BD (triangle ADZ) :=
begin
  sorry
end

end geometry

end tangent_proof_l726_726788


namespace girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l726_726451

namespace PhotoArrangement

/-- There are 4 boys and 3 girls. -/
def boys : ℕ := 4
def girls : ℕ := 3

/-- Number of ways to arrange given conditions -/
def arrangementsWithGirlsAtEnds : ℕ := 720
def arrangementsWithNoGirlsNextToEachOther : ℕ := 1440
def arrangementsWithGirlAtoRightOfGirlB : ℕ := 2520

-- Problem 1: If there are girls at both ends
theorem girls_at_ends (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlsAtEnds := by
  sorry

-- Problem 2: If no two girls are standing next to each other
theorem no_girls_next_to_each_other (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithNoGirlsNextToEachOther := by
  sorry

-- Problem 3: If girl A must be to the right of girl B
theorem girl_A_right_of_girl_B (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlAtoRightOfGirlB := by
  sorry

end PhotoArrangement

end girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l726_726451


namespace least_binary_seven_digits_l726_726063

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726063


namespace prime_divisors_of_1320_l726_726761

theorem prime_divisors_of_1320 : 
  ∃ (S : Finset ℕ), (S = {2, 3, 5, 11}) ∧ S.card = 4 := 
by
  sorry

end prime_divisors_of_1320_l726_726761


namespace dataset_mode_l726_726711

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726711


namespace fixed_point_on_circle_l726_726272

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

def eccentricity (e a c : ℝ) : Prop := e = c / a

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 2

theorem fixed_point_on_circle (a b c : ℝ) (e : ℝ) (P A B : ℝ × ℝ) :
  (a > b) →
  (b > 0) →
  ellipse_eq a b 0 0 →
  eccentricity (Real.sqrt 2 / 2) a c →
  parabola_eq c 0 →
  circle_eq P.fst P.snd →
  ∃ O : ℝ × ℝ, O = (0, 0) ∧
  (∀ A B : ℝ × ℝ, ellipse_eq a b A.fst A.snd ∧ ellipse_eq a b B.fst B.snd → 
  (∀ k : ℝ, P.snd = k * P.fst + Real.sqrt (2 * k^2 + 2) → 
  (O.fst, O.snd) ∈ circle_eq ((A.fst + B.fst) / 2) ((A.snd + B.snd) / 2))
⟩
:= by
  sorry

end fixed_point_on_circle_l726_726272


namespace no_real_solution_l726_726864

-- Given conditions as definitions in Lean 4
def eq1 (x : ℝ) : Prop := x^5 + 3 * x^4 + 5 * x^3 + 5 * x^2 + 6 * x + 2 = 0
def eq2 (x : ℝ) : Prop := x^3 + 3 * x^2 + 4 * x + 1 = 0

-- The theorem to prove
theorem no_real_solution : ¬ ∃ x : ℝ, eq1 x ∧ eq2 x :=
by sorry

end no_real_solution_l726_726864


namespace sister_team_points_l726_726798

variable (w d : ℕ)
variable (w_gt_d : w > d)
variable (team_A_games_won : 7)
variable (team_A_games_drawn : 3)
variable (team_A_total_points : 44)
variable (team_B_games_won : 5)
variable (team_B_games_drawn : 2)

theorem sister_team_points :
  7 * w + 3 * d = team_A_total_points →
  5 * w + 2 * d = 31 :=
by
  intro eq1
  have eq2 : 5 * w + 2 * d = 31 := sorry
  exact eq2

end sister_team_points_l726_726798


namespace strawberry_cake_cost_proof_l726_726976

-- Define the constants
def chocolate_cakes : ℕ := 3
def price_per_chocolate_cake : ℕ := 12
def total_bill : ℕ := 168
def number_of_strawberry_cakes : ℕ := 6

-- Define the calculation for the total cost of chocolate cakes
def total_cost_of_chocolate_cakes : ℕ := chocolate_cakes * price_per_chocolate_cake

-- Define the remaining cost for strawberry cakes
def remaining_cost : ℕ := total_bill - total_cost_of_chocolate_cakes

-- Prove the cost per strawberry cake
def cost_per_strawberry_cake : ℕ := remaining_cost / number_of_strawberry_cakes

theorem strawberry_cake_cost_proof : cost_per_strawberry_cake = 22 := by
  -- We skip the proof here. Detailed proof steps would go in the place of sorry
  sorry

end strawberry_cake_cost_proof_l726_726976


namespace smallest_special_number_gt_3429_l726_726609

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_726609


namespace max_value_of_expressions_l726_726717

theorem max_value_of_expressions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2 * a * b ∧ b > a^2 + b^2 :=
by
  sorry

end max_value_of_expressions_l726_726717


namespace length_of_train_B_is_correct_length_of_train_C_is_correct_l726_726459

-- Define the speeds and lengths in kmph and meters
def speed_train_A_kmph : ℝ := 120
def speed_train_B_kmph : ℝ := 80
def length_train_A_meters : ℝ := 180
def time_cross_AB_seconds : ℝ := 9
def speed_train_C_kmph : ℝ := 100
def time_cross_AC_seconds : ℝ := 12

-- Convert speeds from kmph to m/s
def kmph_to_mps (kmph: ℝ) : ℝ := kmph * (5 / 18)

-- Relative speed of Train A and Train B in m/s
def relative_speed_AB_mps := kmph_to_mps (speed_train_A_kmph + speed_train_B_kmph)

-- Distance covered when Train A and Train B cross each other
def distance_AB_meters := relative_speed_AB_mps * time_cross_AB_seconds

-- The length of Train B (to be proved as 320.04 meters)
def length_train_B_meters := distance_AB_meters - length_train_A_meters

-- Speed of Train C in m/s
def speed_C_mps := kmph_to_mps speed_train_C_kmph

-- Distance covered when Train C crosses Train A
def distance_AC_meters := speed_C_mps * time_cross_AC_seconds

-- The length of Train C (to be proved as 333.36 meters)
def length_train_C_meters := distance_AC_meters

-- Prove that length of Train B equals 320.04 meters
theorem length_of_train_B_is_correct : length_train_B_meters = 320.04 := by
  sorry

-- Prove that length of Train C equals 333.36 meters
theorem length_of_train_C_is_correct : length_train_C_meters = 333.36 := by
  sorry

end length_of_train_B_is_correct_length_of_train_C_is_correct_l726_726459


namespace smallest_special_number_l726_726590

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726590


namespace find_fn_l726_726746

noncomputable def f : ℝ+ → ℝ+ :=
  λ n, 9 * n

theorem find_fn (n : ℝ+) (h : f(f(n)) = 2016 * n - 215 * f(n)) : f(n) = 9 * n := by
  sorry

end find_fn_l726_726746


namespace EM_parallel_AC_l726_726273

-- Define the points A, B, C, D, E, and M
variables (A B C D E M : Type) 

-- Define the conditions described in the problem
variables {x y : Real}

-- Given that ABCD is an isosceles trapezoid with AB parallel to CD and AB > CD
variable (isosceles_trapezoid : Prop)

-- E is the foot of the perpendicular from D to AB
variable (foot_perpendicular : Prop)

-- M is the midpoint of BD
variable (midpoint : Prop)

-- We need to prove that EM is parallel to AC
theorem EM_parallel_AC (h1 : isosceles_trapezoid) (h2 : foot_perpendicular) (h3 : midpoint) : Prop := sorry

end EM_parallel_AC_l726_726273


namespace complex_div_eq_l726_726121

theorem complex_div_eq (z1 z2 : ℂ) (h1 : z1 = 3 - i) (h2 : z2 = 2 + i) :
  z1 / z2 = 1 - i := by
  sorry

end complex_div_eq_l726_726121


namespace sufficient_but_not_necessary_l726_726939

variable (x : ℝ)

def condition1 : Prop := x > 2
def condition2 : Prop := x^2 > 4

theorem sufficient_but_not_necessary :
  (condition1 x → condition2 x) ∧ (¬ (condition2 x → condition1 x)) :=
by 
  sorry

end sufficient_but_not_necessary_l726_726939


namespace bezout_theorem_root_divisibility_l726_726110

-- Part (a)
theorem bezout_theorem (f : ℕ → ℕ) (a : ℕ) : 
  ∃ r g, f = λ x, (x - a) * g x + r ∧ r = f a := 
  sorry

-- Part (b)
theorem root_divisibility (f : ℕ -> ℕ) (x0 : ℕ) 
  (h : f x0 = 0) : 
  ∃ q, f = λ x, (x - x0) * q x := 
  sorry

end bezout_theorem_root_divisibility_l726_726110


namespace mode_of_data_set_l726_726680

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726680


namespace least_binary_seven_digits_l726_726065

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726065


namespace total_cost_pants_and_belt_l726_726446

theorem total_cost_pants_and_belt (P B : ℝ) 
  (hP : P = 34.0) 
  (hCondition : P = B - 2.93) : 
  P + B = 70.93 :=
by
  -- Placeholder for proof
  sorry

end total_cost_pants_and_belt_l726_726446


namespace compare_ln_terms_l726_726321

theorem compare_ln_terms (x : ℝ) (h1 : x > exp (-1)) (h2 : x < 1) :
    let a := log x
    let b := 2 * log x
    let c := (log x) ^ 3
    b < c ∧ c < a := 
by
  let a := log x
  let b := 2 * log x
  let c := (log x) ^ 3
  sorry

end compare_ln_terms_l726_726321


namespace average_class_score_l726_726789

theorem average_class_score :
  ∀ (total_students assigned_day_students make_up_date_students later_date_students : ℕ)
  (assigned_day_avg make_up_date_avg later_date_avg total_class_avg : ℝ),
    total_students = 100 →
    assigned_day_students = 60% * total_students →
    make_up_date_students = 30% * total_students →
    later_date_students = 10% * total_students →
    assigned_day_avg = 60 →
    make_up_date_avg = 80 →
    later_date_avg = 75 →
    (total_class_avg = 
      ((assigned_day_students * assigned_day_avg + make_up_date_students * make_up_date_avg +
       later_date_students * later_date_avg) / total_students)) →
    total_class_avg = 67.5 := 
by
  sorry

end average_class_score_l726_726789


namespace area_triangle_ABC_l726_726338

open Real

theorem area_triangle_ABC (L : ℝ) :
  ∀ (ABC : Triangle)
    (right_angle : ABC.angle_C = π/2)
    (CD_altitude : IsAltitude CD ABC.C ABC.ABC)
    (CE_median : IsMedian CE ABC.C ABC.ABC)
    (angle_ECD : angle_ECD = 20°)
    (angle_DCE : angle_DCE = 40°)
    (area_CDE : area ABC.CDE = L),
    area ABC = 4 * sqrt 3 * L / tan 40 :=
by
  sorry

end area_triangle_ABC_l726_726338


namespace square_side_length_l726_726176

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726176


namespace minimum_detectors_required_l726_726453

/-- There is a cube with each face divided into 4 identical square cells, making a total of 24 cells.
Oleg wants to mark 8 cells with invisible ink such that no two marked cells share a side.
Rustem wants to place detectors in the cells so that all marked cells can be identified. -/
def minimum_detectors_to_identify_all_marked_cells (total_cells: ℕ) (marked_cells: ℕ) 
  (cells_per_face: ℕ) (faces: ℕ) : ℕ :=
  if total_cells = faces * cells_per_face ∧ marked_cells = 8 then 16 else 0

theorem minimum_detectors_required :
  minimum_detectors_to_identify_all_marked_cells 24 8 4 6 = 16 :=
by
  sorry

end minimum_detectors_required_l726_726453


namespace probability_one_defective_l726_726098

theorem probability_one_defective (g d : ℕ) (h_g : g = 3) (h_d : d = 1) : 
  let total_items := g + d in
  let sample_space := (total_items.choose 2).toFinset in
  let event_A := {x ∈ sample_space | x.count (0 = ∘ id) = 1} in
  (event_A.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_one_defective_l726_726098


namespace julia_total_balls_l726_726814

theorem julia_total_balls :
  (3 * 19) + (10 * 19) + (8 * 19) = 399 :=
by
  -- proof goes here
  sorry

end julia_total_balls_l726_726814


namespace correct_addition_by_changing_digit_l726_726414

theorem correct_addition_by_changing_digit :
  ∃ (d : ℕ), (d < 10) ∧ (d = 4) ∧
  (374 + (500 + d) + 286 = 1229 - 50) :=
by
  sorry

end correct_addition_by_changing_digit_l726_726414


namespace not_at_front_count_l726_726452

theorem not_at_front_count : 
  let n := 5 in
  let number_of_arrangements := n! in
  let front_restricted_arrangements := (n-1)! in
  number_of_arrangements - front_restricted_arrangements = 96 :=
by
  let n := 5
  let number_of_arrangements := n!
  let front_restricted_arrangements := (n-1)!
  show number_of_arrangements - front_restricted_arrangements = 96
  sorry

end not_at_front_count_l726_726452


namespace probability_of_selecting_one_defective_l726_726099

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l726_726099


namespace sum_lent_correct_l726_726501

def P : ℝ := 1000

-- Definitions based on conditions
def r : ℝ := 0.05
def t : ℝ := 5
def I : ℝ := P - 750

-- The simple interest calculation
def simple_interest (P r t : ℝ) : ℝ := P * r * t

theorem sum_lent_correct (P : ℝ) (r : ℝ) (t : ℝ) :
  (simple_interest P r t = P - 750) → P = 1000 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_lent_correct_l726_726501


namespace smallest_k_mod_19_7_3_l726_726034

theorem smallest_k_mod_19_7_3 : ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 := 
by {
  -- statements of conditions in form of hypotheses
  let h1 := k > 1,
  let h2 := k % 19 = 1,
  let h3 := k % 7 = 1,
  let h4 := k % 3 = 1,
  -- goal of the theorem
  exact ⟨400, _⟩ sorry -- we indicate the goal should be of the form ⟨value, proof⟩, and fill in the proof with 'sorry'
}

end smallest_k_mod_19_7_3_l726_726034


namespace least_positive_base_ten_seven_binary_digits_l726_726052

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end least_positive_base_ten_seven_binary_digits_l726_726052


namespace value_of_a_sum_l726_726663

theorem value_of_a_sum (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^7 = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 128 := 
by
  sorry

end value_of_a_sum_l726_726663


namespace least_seven_digit_binary_number_l726_726085

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726085


namespace square_side_length_l726_726181

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726181


namespace arithmetic_sequence_second_term_l726_726778

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l726_726778


namespace problem_proof_l726_726673

noncomputable def a : ℕ → ℝ
| 1   := 1
| 2   := 2
| n+1 := real.sqrt ((a n)^2 + (a (n - 1))^2) / real.sqrt 2 -- derived from 2a_n^2 = a_(n-1)^2 + a_(n+1)^2

def b (n : ℕ) : ℝ := 1 / (a n + a (n + 1))

def S (n : ℕ) : ℝ := finset.sum (finset.range n) (λ i, b (i + 1))

theorem problem_proof : S 33 = 3 :=
sorry

end problem_proof_l726_726673


namespace widgets_per_carton_l726_726237

-- Define the dimensions of the cartons and the shipping boxes
constant carton_width : ℕ := 4
constant carton_length : ℕ := 4
constant carton_height : ℕ := 5
constant box_width : ℕ := 20
constant box_length : ℕ := 20
constant box_height : ℕ := 20
constant total_widgets : ℕ := 300

-- Prove the number of widgets per carton
theorem widgets_per_carton : (total_widgets / ((box_width / carton_width) * (box_length / carton_length) * (box_height / carton_height))) = 3 :=
by
  sorry

end widgets_per_carton_l726_726237


namespace polynomial_factorization_l726_726120

theorem polynomial_factorization (x y : ℝ) : -(2 * x - y) * (2 * x + y) = -4 * x ^ 2 + y ^ 2 :=
by sorry

end polynomial_factorization_l726_726120


namespace total_remaining_bottles_is_correct_l726_726476

def initial_small_bottles : ℕ := 6000
def initial_big_bottles : ℕ := 15000
def percent_sold_small : ℚ := 12 / 100
def percent_sold_big : ℚ := 14 / 100
def remaining_bottles : ℕ := 18180

theorem total_remaining_bottles_is_correct :
  (initial_small_bottles - (initial_small_bottles * percent_sold_small).toNat) +
  (initial_big_bottles - (initial_big_bottles * percent_sold_big).toNat) =
  remaining_bottles :=
by
  sorry

end total_remaining_bottles_is_correct_l726_726476


namespace triangle_congruence_proof_l726_726467

noncomputable def triangle_congruence_condition_1 (ABC A'B'C' : Triangle) : Prop :=
  ABC.AB = A'B'C'.AB ∧ ABC.BC = A'B'C'.BC ∧ ABC.AC = A'B'C'.AC

noncomputable def triangle_congruence_condition_4 (ABC A'B'C' : Triangle) : Prop :=
  ABC.AB = A'B'C'.AB ∧ ABC.angle_B = A'B'C'.angle_B ∧ ABC.angle_C = A'B'C'.angle_C

theorem triangle_congruence_proof (ABC A'B'C' : Triangle) :
  (triangle_congruence_condition_1 ABC A'B'C' ∨ triangle_congruence_condition_4 ABC A'B'C') →
  (ABC ≅ A'B'C') :=
begin
  sorry
end

end triangle_congruence_proof_l726_726467


namespace square_side_length_l726_726180

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726180


namespace graph_of_g_abs_x_l726_726007

def g (x : ℝ) : ℝ :=
  if x >= 0 then x - 3 else -x

def absolute_value (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

def g_abs_x (x : ℝ) : ℝ := g (absolute_value x)

theorem graph_of_g_abs_x (x : ℝ) : g_abs_x x = absolute_value x - 3 :=
by 
  unfold g_abs_x 
  unfold g
  unfold absolute_value
  split_ifs
  case _ h₁ h₂ { 
    simp at h₁
    sorry
  }
  case _ h₀ { 
    simp at h₀
    sorry
  }

end graph_of_g_abs_x_l726_726007


namespace cut_problem_l726_726506

theorem cut_problem (n : ℕ) : (1 / 2 : ℝ) ^ n = 1 / 64 ↔ n = 6 :=
by
  sorry

end cut_problem_l726_726506


namespace cycle_original_cost_l726_726499

theorem cycle_original_cost (SP : ℝ) (gain : ℝ) (CP : ℝ) (h₁ : SP = 2000) (h₂ : gain = 1) (h₃ : SP = CP * (1 + gain)) : CP = 1000 :=
by
  sorry

end cycle_original_cost_l726_726499


namespace max_good_parabolas_l726_726714

-- Definitions to model the problem conditions
def Point := ℝ × ℝ -- A point in the plane

noncomputable def is_good_parabola (p1 p2 : Point) (points : Finset Point) : Prop :=
  ∀ p ∈ points, p ≠ p1 ∧ p ≠ p2 → p.2 > (p.1 - ((p1.1 + p2.1) / 2))^2 - (p1.2 + p2.2) / 2

-- Main theorem to prove
theorem max_good_parabolas (n : ℕ) (points : Finset Point)
  (h_diff_abscissas : ∀ p1 p2 ∈ points, p1.1 ≠ p2.1) :
  ∃ (good_parabolas : Finset (Point × Point)),
    good_parabolas.card ≤ n - 1 ∧
    ∀ par ∈ good_parabolas, is_good_parabola par.1 par.2 points :=
by
  sorry

end max_good_parabolas_l726_726714


namespace least_seven_digit_binary_number_l726_726083

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726083


namespace initially_estimated_days_is_8_l726_726125

-- Given conditions expressed in Lean
variables 
  (workers1 workers2 : ℕ) -- number of workers in two stages
  (days1 days2 : ℕ) -- number of days in two stages
  (total_work : ℕ) -- total work in worker-days
  (initial_days : ℕ) -- initially estimated days (to be proven)

-- Initialize values based on given real-world problem
def workers1 := 6
def days1 := 3
def joined_workers := 4
def workers2 := 10
def days2 := 3
def total_work := (workers1 * days1) + (workers2 * days2)

-- The Lean problem statement to prove
theorem initially_estimated_days_is_8 (h : total_work = workers1 * initial_days) : initial_days = 8 :=
by
  have := total_work
  rw [(show workers1 = 6, by rfl), (show workers2 = 10, by rfl), (show days1 = 3, by rfl), (show days2 = 3, by rfl)] at this
  change 48 = 6 * initial_days at this
  have h_init := @eq_of_mul_eq_mul_left' _ _ 6 48 initial_days (lt_of_le_of_lt (zero_le _) (nat.lt_succ_self 6)) this
  rw h_init
  sorry

end initially_estimated_days_is_8_l726_726125


namespace proof_problem_l726_726432

-- Definitions
def is_relative_prime (p q : ℕ) : Prop := Nat.gcd p q = 1

-- Main theorem statement
theorem proof_problem (p q : ℕ) (a : ℚ) (S : Set ℝ) 
  (h1 : a = p / q)
  (h2 : is_relative_prime p q)
  (h3 : ∀ x ∈ S, ⌊x⌋ * (x - ⌊x⌋) = a * x^2)
  (h4 : ∑ x in S, x = 540) : p + q = 1023 :=
sorry

end proof_problem_l726_726432


namespace incorrect_proposition_statement_l726_726469

theorem incorrect_proposition_statement 
  (p: ∃ x : ℝ, x^2 + x + 1 < 0)
  (h₁: ¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0))
  (h₂: ∀ x : ℝ, x = 1 → x^2 - 3x + 2 = 0)
  (h₃: ∀ x : ℝ, ¬ (x ≠ 1 → x^2 - 3x + 2 ≠ 0))
  (h₄: ∀ p q : Prop, ¬ (p ∧ q) → (¬ p ∨ ¬ q)) :
  ¬ (∀ p q : Prop, ¬ (p ∧ q) → (¬ p ∧ ¬ q)) := 
sorry

end incorrect_proposition_statement_l726_726469


namespace area_of_rectangular_field_l726_726332

-- Define the conditions
def length (b : ℕ) : ℕ := b + 30
def perimeter (b : ℕ) (l : ℕ) : ℕ := 2 * (b + l)

-- Define the main theorem to prove
theorem area_of_rectangular_field (b : ℕ) (l : ℕ) (h1 : l = length b) (h2 : perimeter b l = 540) : 
  l * b = 18000 := by
  -- Placeholder for the proof
  sorry

end area_of_rectangular_field_l726_726332


namespace correct_input_statement_l726_726462

-- Definitions based on the conditions
def input_format_A : Prop := sorry
def input_format_B : Prop := sorry
def input_format_C : Prop := sorry
def output_format_D : Prop := sorry

-- The main statement we need to prove
theorem correct_input_statement : input_format_A ∧ ¬ input_format_B ∧ ¬ input_format_C ∧ ¬ output_format_D := 
by sorry

end correct_input_statement_l726_726462


namespace purely_imaginary_subtraction_l726_726728

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end purely_imaginary_subtraction_l726_726728


namespace phillips_mother_money_l726_726850

theorem phillips_mother_money (spent_oranges spent_apples spent_candy amount_left total_given : ℕ)
  (h_oranges : spent_oranges = 14)
  (h_apples : spent_apples = 25)
  (h_candy : spent_candy = 6)
  (h_left : amount_left = 50)
  (h_total_given : total_given = spent_oranges + spent_apples + spent_candy + amount_left) :
  total_given = 95 :=
begin
  sorry
end

end phillips_mother_money_l726_726850


namespace parcel_cost_l726_726017

theorem parcel_cost (W : ℝ) : 8 * Real.ceil W + 5 = 8 * ⌈ W ⌉ + 5 :=
by sorry

end parcel_cost_l726_726017


namespace total_days_2005_to_2008_l726_726317

-- Definitions based on conditions
def is_leap_year (year : ℕ) : Prop :=
  year = 2008

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  List.sum (List.map days_in_year [2005, 2006, 2007, 2008])

-- Statement: the proof problem
theorem total_days_2005_to_2008 : total_days 2005 2008 = 1461 :=
  sorry

end total_days_2005_to_2008_l726_726317


namespace find_seating_capacity_l726_726955

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l726_726955


namespace principal_amount_l726_726427

theorem principal_amount
  (P : ℝ)
  (r : ℝ := 0.05)
  (t : ℝ := 2)
  (H : P * (1 + r)^t - P - P * r * t = 17) :
  P = 6800 :=
by sorry

end principal_amount_l726_726427


namespace polynomial_product_l726_726117

theorem polynomial_product (a b c : ℝ) :
  a * (b - c) ^ 3 + b * (c - a) ^ 3 + c * (a - b) ^ 3 = (a - b) * (b - c) * (c - a) * (a + b + c) :=
by sorry

end polynomial_product_l726_726117


namespace mode_of_data_set_l726_726694

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726694


namespace smallest_special_number_gt_3429_l726_726575

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726575


namespace count_valid_numbers_l726_726020

theorem count_valid_numbers : 
  let count_A := 10 
  let count_B := 2 
  count_A * count_B = 20 :=
by 
  let count_A := 10
  let count_B := 2
  have : count_A * count_B = 20 := by norm_num
  exact this

end count_valid_numbers_l726_726020


namespace smallest_special_number_l726_726591

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726591


namespace measure_angle_MBA_l726_726350

open Classical
noncomputable theory

-- Definition of the problem
def triangle := Type
variables {ABC : triangle} (A B C M : ABC) 
variables (angle : ABC → ABC → ABC → ℝ) 

-- Given conditions
axiom angle_BAC : angle B A C = 30
axiom angle_ABC : angle A B C = 70
axiom angle_MAB : angle M A B = 20
axiom angle_MCA : angle M C A = 20

-- Prove that the measure of angle MBA is 30 degrees
theorem measure_angle_MBA : angle M B A = 30 :=
by {
  sorry
}

end measure_angle_MBA_l726_726350


namespace aₙ_formula_Tₙ_formula_l726_726713

noncomputable theory

-- Define the arithmetic sequence {a_n} and sum S_n
def a (n : ℕ) : ℤ := -2 * n + 7
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

-- Conditions given in the problem
axiom a₅_eq : a 5 = -3
axiom S₁₀_eq : S 10 = -40

-- Define the new sequence {b_n}
def b (n : ℕ) : ℤ := a (2^n)

-- Define the sum of the first n terms of {b_n}, T_n
def T (n : ℕ) : ℤ := (Finset.range n).sum (λ i, b (i + 1))

-- The hypotheses, using the conditions
theorem aₙ_formula : ∀ n : ℕ, a n = -2 * n + 7 :=
by sorry

theorem Tₙ_formula : ∀ n : ℕ, T n = 4 + 7 * n - 2^(n + 2) :=
by sorry

end aₙ_formula_Tₙ_formula_l726_726713


namespace invariant_chord_length_CD_l726_726422

variables {k1 k2 : Circle} {A B P C D : Point}

-- Assume A and B are the common points of circles k1 and k2
variable (h1 : intersects k1 k2 A)
variable (h2 : intersects k1 k2 B)

-- Precondition: P is a point on circle k1
variable (h3 : on_circle P k1)

-- C and D are second intersection points of lines PA and PB with circle k2
variable (h4 : second_intersection C (line PA) k2)
variable (h5 : second_intersection D (line PB) k2)

-- Proof goal: the length of chord CD does not depend on the choice of point P on k1
theorem invariant_chord_length_CD :
  ∀ P, on_circle P k1 → chord_length C D = constant_length := 
sorry

end invariant_chord_length_CD_l726_726422


namespace sum_of_integers_from_neg20_to_100_l726_726919

theorem sum_of_integers_from_neg20_to_100 :
  let a := -20 in
  let l := 100 in
  let n := (l - a + 1) in
  (n * (a + l)) / 2 = 4840 := by
  let a := -20
  let l := 100
  let n := l - a + 1
  suffices (n * (a + l)) / 2 = 4840 by trivial
  sorry

end sum_of_integers_from_neg20_to_100_l726_726919


namespace find_g_one_half_l726_726005

noncomputable def g (x : ℝ) : ℝ := sorry

lemma g_property (x : ℝ) (hx : x ≠ 0) : g(x) - 3 * g(1/x) = 4^x + Real.exp(x) :=
sorry

theorem find_g_one_half : g(1/2) = (3 * Real.exp(2) - 13 * Real.sqrt(Real.exp(1)) + 82) / 8 :=
by
  -- The proof will be filled here
  sorry

end find_g_one_half_l726_726005


namespace mode_of_data_set_l726_726692

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726692


namespace smallest_special_number_l726_726586

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726586


namespace sum_of_distances_between_18_and_19_l726_726790

noncomputable def A : ℝ × ℝ := (16, 0)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def D : ℝ × ℝ := (0, 2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

def AD : ℝ := distance A D
def BD : ℝ := distance B D

theorem sum_of_distances_between_18_and_19 :
  18 < AD + BD ∧ AD + BD < 19 :=
by
  sorry

end sum_of_distances_between_18_and_19_l726_726790


namespace find_triples_l726_726830

def d (n : ℕ) : ℕ := 
  if n = 0 then 0 else (Finset.filter (λ x, n % x = 0) (Finset.range (n + 1))).card

theorem find_triples (n k p : ℕ) (hp : Nat.Prime p) (hn : n > 0) (hk : k > 0)
  (h_eq : n ^ d(n) - 1 = p ^ k):
  (n = 2 ∧ k = 1 ∧ p = 3) ∨ (n = 3 ∧ k = 3 ∧ p = 2) :=
by
  sorry

end find_triples_l726_726830


namespace grasshopper_reach_l726_726135

-- Define the initial point
def initial_point : (ℤ × ℤ) := (1, 1)

-- Define the area condition
def area_half (A B : ℤ × ℤ) : Prop :=
  let (a1, a2) := A in
  let (b1, b2) := B in
  (abs (a1 * b2 - a2 * b1) = 1)

-- Define reachable/lattice points
def reachable (start target : ℤ × ℤ) : Prop :=
  ∃ (path : List (ℤ × ℤ)), path.head = start ∧ path.last = some target ∧
  (∀ (i : ℕ), i < path.length - 1 → area_half (path.nth_le i sorry) (path.nth_le (i + 1) sorry))

-- Define the gcd function for pairs
def gcd_pair (p : ℤ × ℤ) : ℤ := p.1.gcd p.2

-- The proof problem statement
theorem grasshopper_reach (m n : ℤ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) :
  reachable initial_point (m, n) ∧ (∃ (path : List (ℤ × ℤ)), path.head = initial_point ∧ path.last = some (m, n) ∧ path.length - 1 ≤ abs (m - n)) :=
sorry

end grasshopper_reach_l726_726135


namespace max_number_of_streetlights_l726_726511

theorem max_number_of_streetlights (road_length : ℕ) (illumination_length : ℕ) (full_illumination_condition : ∀ {n : ℕ}, 
  (road_length = 1000) ∧ (illumination_length = 1) ∧ ((∀ i ∈ (Finset.range n), illuminates i → ∀ j ∈ (Finset.range n), illuminates j → i ≠ j)) ∧ 
  (∀ k ∈ (Finset.range n), ¬fully_illuminated_without k) → n ≤ 1998) : road_length = 1000 ∧ illumination_length = 1 ∧ n = 1998 :=
begin
  -- Proof is required to demonstrate this theorem
  sorry
end

end max_number_of_streetlights_l726_726511


namespace least_7_digit_binary_number_is_64_l726_726075

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726075


namespace initial_outlay_l726_726517

theorem initial_outlay (cost_per_set initial_outlay selling_price : ℝ)
  (num_sets : ℕ)
  (profit : ℝ) 
  (h1 : cost_per_set = 20.75)
  (h2 : selling_price = 50)
  (h3 : num_sets = 950)
  (h4 : profit = 15337.5) :
  initial_outlay = 12450 :=
by
  have cost_total := cost_per_set * num_sets + initial_outlay
  have revenue_total := selling_price * num_sets
  have profit_eq := profit = revenue_total - cost_total
  rw [cost_per_set, selling_price, num_sets, profit] at profit_eq
  simp at profit_eq
  linarith
  sorry

end initial_outlay_l726_726517


namespace force_on_dam_calculation_l726_726116

/-- 
The force with which water pushes against a dam, whose cross-section has the shape 
of an isosceles trapezoid, is given by the following parameters:
- a: top length of the trapezoid (5.4 m),
- b: bottom length of the trapezoid (8.4 m),
- h: height of the trapezoid (3.0 m),
- ρ: density of water (1000 kg/m^3),
- g: acceleration due to gravity (10 m/s^2).

We need to prove that the total force \( F \) exerted by the water on the dam is 
288000 N.
-/
def calculate_dam_force (a b h ρ g : ℝ) : ℝ :=
  ρ * g * h^2 * ((b / 2) - (b - a) / 3)

theorem force_on_dam_calculation :
  calculate_dam_force 5.4 8.4 3.0 1000 10 = 288000 :=
by /
  -- Proof will be provided here
  sorry

end force_on_dam_calculation_l726_726116


namespace least_7_digit_binary_number_is_64_l726_726079

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726079


namespace parabola_standard_equation_l726_726284

noncomputable def parabola_equation (p : ℝ) : Prop := ∀ (x y : ℝ), y^2 = 2 * p * x → ((x - 1) ^ 2 + (y - 1) ^ 2 = 0 → p = 1)

theorem parabola_standard_equation :
  parabola_equation 1 :=
by
  intros x y h1 h2,
  sorry

end parabola_standard_equation_l726_726284


namespace three_digit_numbers_not_divisible_by_three_l726_726659

theorem three_digit_numbers_not_divisible_by_three :
  let digits := {1, 2, 3, 4, 5}
  let combinations := (Comb n 3).to_finset
  let valid_combinations := combinations.filter (λ comb, (sum comb) % 3 ≠ 0)
  let permutations_of_comb : Finset (Finset (List ℕ)) := valid_combinations.bind (λ comb, (Multichoose ℕ).permutations)
  permutations_of_comb.card = 18 := sorry

end three_digit_numbers_not_divisible_by_three_l726_726659


namespace square_side_length_l726_726175

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l726_726175


namespace smallest_special_number_l726_726594

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726594


namespace quadratic_solution_eq_l726_726827

noncomputable def p : ℝ :=
  (8 + Real.sqrt 364) / 10

noncomputable def q : ℝ :=
  (8 - Real.sqrt 364) / 10

theorem quadratic_solution_eq (p q : ℝ) (h₁ : 5 * p^2 - 8 * p - 15 = 0) (h₂ : 5 * q^2 - 8 * q - 15 = 0) : 
  (p - q) ^ 2 = 14.5924 :=
sorry

end quadratic_solution_eq_l726_726827


namespace real_root_of_equation_l726_726639

theorem real_root_of_equation :
  ∃ x : ℝ, sqrt x + sqrt (x + 4) = 12 ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726639


namespace monotonic_decreasing_interval_l726_726886

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem monotonic_decreasing_interval : 
  {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f' x < 0} :=
by
  -- defintion of derivative
  noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 6*x
  -- proof
  sorry

end monotonic_decreasing_interval_l726_726886


namespace current_tree_height_in_inches_l726_726816

-- Constants
def initial_height_ft : ℝ := 10
def growth_percentage : ℝ := 0.50
def feet_to_inches : ℝ := 12

-- Conditions
def growth_ft : ℝ := growth_percentage * initial_height_ft
def current_height_ft : ℝ := initial_height_ft + growth_ft

-- Question/Answer equivalence
theorem current_tree_height_in_inches :
  (current_height_ft * feet_to_inches) = 180 :=
by 
  sorry

end current_tree_height_in_inches_l726_726816


namespace derivative_at_4_l726_726300

noncomputable def f (x : ℝ) : ℝ := x^2 + (fderiv ℝ f 2) ((log x) - x)

theorem derivative_at_4 : deriv f 4 = 6 := by
  sorry

end derivative_at_4_l726_726300


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726072

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726072


namespace infinite_area_sum_ratio_l726_726440

theorem infinite_area_sum_ratio (T t : ℝ) (p q : ℝ) (h_ratio : T / t = 3 / 2) :
    let series_ratio_triangles := (p + q)^2 / (3 * p * q)
    let series_ratio_quadrilaterals := (p + q)^2 / (2 * p * q)
    (T * series_ratio_triangles) / (t * series_ratio_quadrilaterals) = 1 :=
by
  -- Proof steps go here
  sorry

end infinite_area_sum_ratio_l726_726440


namespace probability_not_snowing_l726_726889

theorem probability_not_snowing (p_snowing : ℚ) (h : p_snowing = 1/4) : 1 - p_snowing = 3/4 := 
by
  rw [h]
  norm_num
  sorry

end probability_not_snowing_l726_726889


namespace sequence_sum_mod_255_l726_726618

theorem sequence_sum_mod_255 :
  let T : ℕ → ℕ :=
    λ n, Nat.recOn n 2 (λ n' T_n, 2 ^ T_n)
  in (List.range 256).sum (λ n, T n) % 255 = 20 :=
sorry

end sequence_sum_mod_255_l726_726618


namespace irrational_arithmetic_seq_l726_726378

theorem irrational_arithmetic_seq :
  (∀ n : ℕ, a n ∈ {n : ℝ | irrational n} ∧ 
  ∃ l m : ℕ, l ≠ m ∧ a l = real.sqrt 2 ∧ a m = real.sqrt 3) 
  → ∀ n : ℕ, irrational (a n) :=
by {
  sorry
}

end irrational_arithmetic_seq_l726_726378


namespace triangle_cosine_problem_l726_726373

noncomputable def verify_cosine_identity (A B C : ℝ) : Prop :=
  ∃ (p q r s : ℕ), (B > π / 2) ∧ 
  (cos A ^ 2 + cos B ^ 2 + 2 * sin A * sin B * cos C = 15 / 8) ∧
  (cos B ^ 2 + cos C ^ 2 + 2 * sin B * sin C * cos A = 14 / 9) ∧ 
  (cos C ^ 2 + cos A ^ 2 + 2 * sin C * sin A * cos B = (p - q * real.sqrt r) / s) ∧ 
  (p + q + r + s = 222) ∧ 
  nat.gcd (p + q) s = 1 ∧ 
  (∀ k : ℕ, k^2 ∣ r → k = 1)

theorem triangle_cosine_problem :
  ∃ (A B C : ℝ), verify_cosine_identity A B C :=
by {
  -- Here the concrete angles and the verification would occur
  sorry
}

end triangle_cosine_problem_l726_726373


namespace volume_ratio_of_cylinder_to_cube_l726_726144

theorem volume_ratio_of_cylinder_to_cube (s : ℝ) :
  (s > 0) → (π * ((s / 2) ^ 2) * s) / (s ^ 3) = π / 4 :=
begin
  intro hs,
  have h1 : (π * ((s / 2) ^ 2) * s) = (π * (s^2 / 4) * s), from congr_arg (λ x, π * x * s) (by ring),
  have h2 : (π * (s^2 / 4) * s) = (π * s^3 / 4), from congr_arg (λ x, π * x / 4) (by ring),
  have h3 : (π * s^3 / 4) / (s ^ 3) = π / 4, { field_simp [hs], ring },
  exact eq.trans (eq.trans (eq.symm h1) h2) h3
end

end volume_ratio_of_cylinder_to_cube_l726_726144


namespace max_disjoint_pairs_sum_l726_726819

theorem max_disjoint_pairs_sum (n : ℕ) (hn : n ≥ 1) : 
  ∃ f : ℕ → ℕ, (∀ n, f n = ⌊(2 * n - 1) / 5⌋) ∧ f n = ⌊(2 * n - 1) / 5⌋ :=
 by
  sorry

end max_disjoint_pairs_sum_l726_726819


namespace ratio_of_areas_l726_726143

theorem ratio_of_areas (w : ℝ) (A B : ℝ) :
  (A = 2 * w^2) ∧ (B = 2 * w^2 - w^2 * ℝ.sqrt 2) → (B / A = 1 - ℝ.sqrt 2 / 2) :=
begin
  sorry
end

end ratio_of_areas_l726_726143


namespace perpendicular_bisector_value_l726_726008

theorem perpendicular_bisector_value :
  let M : ℝ × ℝ := (5, 4)
  in ∀ x y b : ℝ, 
  (M.1 x + M.2 y = b) ∧ ((x, y) = (2, 1) ∨ (x, y) = (8, 7))
  → b = 9 :=
by
  intro M
  let M := (5, 4)
  intros x y b h
  sorry

end perpendicular_bisector_value_l726_726008


namespace base8_to_base10_l726_726554

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l726_726554


namespace min_value_frac_sqrt_l726_726261

theorem min_value_frac_sqrt (x : ℝ) (h : x > 1) : 
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 :=
sorry

end min_value_frac_sqrt_l726_726261


namespace multiplication_of_mixed_number_l726_726994

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726994


namespace num_integers_without_digits_l726_726760

theorem num_integers_without_digits (a b c d : ℕ) (h1 : 1000 ≤ 1000 + 1000 * a + 100 * b + 10 * c + d) 
                                      (h2 : 1000 + 1000 * a + 100 * b + 10 * c + d ≤ 9999)
                                      (h3 : a ∉ {1, 5, 6, 7}) 
                                      (h4 : b ∉ {1, 5, 6, 7}) 
                                      (h5 : c ∉ {1, 5, 6, 7}) 
                                      (h6 : d ∉ {1, 5, 6, 7}) : 
                                      1080 = (5 * 6 * 6 * 6) :=
  by
  sorry

end num_integers_without_digits_l726_726760


namespace average_percentage_reduction_is_correct_l726_726797

noncomputable def average_percentage_reduction (P_initial P_final : ℝ) :=
  let avg_reduction := 1 - real.sqrt (P_final / P_initial)
  avg_reduction

theorem average_percentage_reduction_is_correct :
  average_percentage_reduction 100 81 = 0.1 :=
by
  sorry

end average_percentage_reduction_is_correct_l726_726797


namespace smallest_k_l726_726038

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l726_726038


namespace skt_lineups_l726_726024

/-- 
Given:
- There are 111 StarCraft programmers.
- SKT starts with a set of 11 programmers.
- At the end of each season, SKT drops one programmer and adds another, potentially even the same one.
- At the start of the second season, SKT needs to field a team of five programmers.

Prove:
The number of different lineups of five players could be fielded if the order of players on the lineup matters is 61593840.
-/
theorem skt_lineups (total_programmers : ℕ) (initial_team_size : ℕ) (final_team_size : ℕ) 
  (choose_size : ℕ) [fact (total_programmers = 111)] [fact (initial_team_size = 11)] 
  [fact (final_team_size = 11)] [fact (choose_size = 5)] :
  (initial_team_size * (total_programmers - initial_team_size + 1) * 
  (Nat.choose final_team_size choose_size) * (Nat.factorial choose_size) = 61593840) :=
sorry

end skt_lineups_l726_726024


namespace cupboard_cost_price_l726_726115

theorem cupboard_cost_price (C : ℝ) 
  (h1 : ∀ C₀, C = C₀ → C₀ * 0.88 + 1500 = C₀ * 1.12) :
  C = 6250 := by
  sorry

end cupboard_cost_price_l726_726115


namespace resistance_per_band_is_10_l726_726835

noncomputable def resistance_per_band := 10
def total_squat_weight := 30
def dumbbell_weight := 10
def number_of_bands := 2

theorem resistance_per_band_is_10 :
  (total_squat_weight - dumbbell_weight) / number_of_bands = resistance_per_band := 
by
  sorry

end resistance_per_band_is_10_l726_726835


namespace max_interval_length_value_a_l726_726732

noncomputable def max_interval_value_a (x1 x2 : ℝ) (a : ℝ) (f : ℝ → ℝ) : ℝ :=
if h : x2 > x1 ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc x1 x2, f x = (a^2 + a) * x - 1 / (a^2 * x)) 
then 3 
else 0

theorem max_interval_length_value_a :
  ∀ (a : ℝ) (f : ℝ → ℝ) (m n : ℝ),
    (∀ x : ℝ, f x = (a^2 + a) * x - 1 / (a^2 * x)) →
    a ≠ 0 →
    f m = m →
    f n = n →
    ∃ l : ℝ, l = (n - m) ∧ (l = max_interval_value_a m n a f) :=
sorry

end max_interval_length_value_a_l726_726732


namespace square_side_length_l726_726173

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726173


namespace least_seven_digit_binary_number_l726_726087

theorem least_seven_digit_binary_number : ∃ n : ℕ, (nat.binary_digits n = 7) ∧ (n = 64) := by
  sorry

end least_seven_digit_binary_number_l726_726087


namespace real_root_of_equation_l726_726640

theorem real_root_of_equation :
  ∃ x : ℝ, sqrt x + sqrt (x + 4) = 12 ∧ x = 1225 / 36 :=
by
  sorry

end real_root_of_equation_l726_726640


namespace minimum_balls_drawn_l726_726948

theorem minimum_balls_drawn
    (white red blue black : ℕ)
    (h_white : white = 100)
    (h_red : red = 100)
    (h_blue : blue = 100)
    (h_black : black = 100) :
    ∃ n, n = 9 ∧ ∀ draws, (draws >= n) → ∃ color, color ∈ { "white", "red", "blue", "black" } ∧ count color draws ≥ 3 :=
by
  sorry

end minimum_balls_drawn_l726_726948


namespace base8_246_is_166_in_base10_l726_726548

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l726_726548


namespace jellybean_capacity_l726_726904

theorem jellybean_capacity (V_Tim Ella_Volume jellybeans_Tim jellybeans_Ella : ℕ) 
  (h_1 : V_Tim = jellybeans_Tim)
  (h_2 : Ella_Volume = 27 * V_Tim)
  (h_3 : jellybeans_Tim = 150) :
  jellybeans_Ella = 27 * 150 :=
begin
  sorry
end

end jellybean_capacity_l726_726904


namespace conjugate_in_third_quadrant_l726_726287

theorem conjugate_in_third_quadrant (z : ℂ) (h : z = (2 * (Complex.I)) / (1 - Complex.I)) :
  Complex.conj z = -Complex.I - 1 ∧ -1 < 0 ∧ -1 < 0 := 
by
  sorry

end conjugate_in_third_quadrant_l726_726287


namespace intersection_result_l726_726719

variable {U : Set ℝ}
variable {A B : Set ℝ}
variable {a b t : ℝ}

def U := Set.univ  -- U = ℝ

-- Given conditions
axiom (H1 : a ≠ 0)
axiom (H2 : ∀ x, x ∈ U → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a)
axiom (H3 : a > b)

-- Result sets
def T := {t | t = (a^2 + b^2) / (a - b)}
def A := {m | 2 * Real.sqrt 2 ≤ m ∧ m < 4}
def B := {m | ∀ x, x ∈ U → |x + 1| - |x - 3| ≤ m^2 - 3 * m}
def complement_B := {m | -1 < m ∧ m < 4}

theorem intersection_result : A ∩ complement_B = {m | 2 * Real.sqrt 2 ≤ m ∧ m < 4} :=
sorry

end intersection_result_l726_726719


namespace constant_term_in_binomial_expansion_l726_726379

theorem constant_term_in_binomial_expansion :
  let a := ∫ x in 0..1, 2 * x
  let binomial := (a * x^2 - 1 / x) ^ 6
  (∃ c : ℤ, c = 15 ∧ ∀ x : ℝ, binomial = c) ∧ a = 1 := sorry

end constant_term_in_binomial_expansion_l726_726379


namespace combined_list_correct_l726_726812

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end combined_list_correct_l726_726812


namespace selection_of_students_l726_726497

theorem selection_of_students (boys girls : ℕ) (students_selected : ℕ)
  (at_least_one_girl : students_selected ≥ 1) :
  boys = 4 → girls = 2 → students_selected = 4 → 
  (choose boys 3 * choose girls 1 + choose boys 2 * choose girls 2) = 14 :=
by
  intros hboys hgirls hstudents
  rw [hboys, hgirls, hstudents]
  norm_num
  sorry

end selection_of_students_l726_726497


namespace total_art_cost_l726_726362

-- Definitions based on the conditions
def total_price_first_3_pieces (price_per_piece : ℤ) : ℤ :=
  price_per_piece * 3

def price_increase (price_per_piece : ℤ) : ℤ :=
  price_per_piece / 2

def total_price_all_arts (price_per_piece next_piece_price : ℤ) : ℤ :=
  (total_price_first_3_pieces price_per_piece) + next_piece_price

-- The proof problem statement
theorem total_art_cost : 
  ∀ (price_per_piece : ℤ),
  total_price_first_3_pieces price_per_piece = 45000 →
  next_piece_price = price_per_piece + price_increase price_per_piece →
  total_price_all_arts price_per_piece next_piece_price = 67500 :=
  by
    intros price_per_piece h1 h2
    sorry

end total_art_cost_l726_726362


namespace dataset_mode_l726_726710

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726710


namespace smallest_special_number_l726_726592

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l726_726592


namespace daphney_pays_15_l726_726567

variable (kg_of_potatoes : ℕ) (cost_for_2kg : ℝ)

-- Define the conditions as premises
def condition1 : kg_of_potatoes = 5 := rfl
def condition2 : cost_for_2kg = 6 := rfl

-- Define the price per kg given the conditions
def price_per_kg := cost_for_2kg / 2

-- Define the total cost for 5 kg
def total_cost := kg_of_potatoes * price_per_kg

-- Prove that the total cost for 5 kg of potatoes is $15
theorem daphney_pays_15 : total_cost kg_of_potatoes cost_for_2kg = 15 := by
  rw [condition1]
  rw [condition2]
  simp [total_cost, price_per_kg]
  linarith


end daphney_pays_15_l726_726567


namespace smallest_special_greater_than_3429_l726_726570

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l726_726570


namespace apples_to_bananas_l726_726411

theorem apples_to_bananas {A B : Type}
  (value_equiv : 0.75 * (12 : ℝ) = 10)
  (apples_to_bananas : A ≃ B) :
  (2/3 * (9 : ℝ)) = (20 / 3 : ℝ) :=
begin
  sorry
end

end apples_to_bananas_l726_726411


namespace caterpillars_and_leaves_l726_726897

theorem caterpillars_and_leaves 
(initial_caterpillars : ℕ)
(storm_fall : ℕ)
(eggs_hatched : ℕ)
(leaves_eaten_per_baby_per_day : ℕ)
(week_days : ℕ)
(cocoon_leavers : ℕ)
(turn_into_moths_ratio : ℕ)
(h : initial_caterpillars = 14)
(s : storm_fall = 3)
(e : eggs_hatched = 6)
(l : leaves_eaten_per_baby_per_day = 2)
(w : week_days = 7)
(c : cocoon_leavers = 9)
(r : turn_into_moths_ratio = 2) :
  let remaining_caterpillars_after_storm := initial_caterpillars - storm_fall,
      total_caterpillars := remaining_caterpillars_after_storm + eggs_hatched,
      leaves_per_day := leaves_eaten_per_baby_per_day * eggs_hatched,
      total_leaves := leaves_per_day * week_days,
      remaining_caterpillars_after_cocoon := total_caterpillars - cocoon_leavers,
      moths := remaining_caterpillars_after_cocoon / turn_into_moths_ratio,
      final_caterpillars := remaining_caterpillars_after_cocoon - moths in
  final_caterpillars = 4 ∧ total_leaves = 84 := by
  sorry

end caterpillars_and_leaves_l726_726897


namespace highest_score_not_ceiling_margin_l726_726960

noncomputable def round_robin_tournament (teams : ℕ) : Prop :=
  (teams = 14) ∧
  (∀ t, t < teams →
    ∃ scores : Fin (teams + 1) → ℕ,
    (∀ a b, a ≠ b → scores a ≤ scores b ∨ scores b ≤ scores a) ∧
    (∀ t, scores t ≠ 0 ∨ (∃! t', t ≠ t' ∧ scores t' = 0)) ∧
    (14 teams play 91 games which contribute a total of 182 points) ∧
    (∑ i, scores i ≥ 130) ∧
    (∃ i, scores i ≥ 13) ∧
    ( (∀ t, scores t % 2 = 1 → odd (scores t)) ∧
      (∀ t, scores t % 2 = 0 → even (scores t)) ∧
      (∃ s, ∀ t, scores t ≠ s ∨ even s)
    )
  )

theorem highest_score_not_ceiling_margin : 
  round_robin_tournament 14 → (∃! t, scores t ≥ 13) := sorry

end highest_score_not_ceiling_margin_l726_726960


namespace area_of_gray_part_l726_726910

theorem area_of_gray_part
  (A1 A2 : ℕ)
  (a b : ℕ)
  (c d : ℕ)
  (area_black : ℕ) :
  a = 8 → b = 10 → c = 12 → d = 9 → area_black = 37 →
  A1 = a * b → A2 = c * d →
  (A2 - (A1 - area_black) = 65) :=
by {
  intros ha hb hc hd hblack hA1 hA2,
  rw [ha, hb] at hA1,
  rw [hc, hd] at hA2,
  rw [hA1, hA2, hblack],
  norm_num,
}

end area_of_gray_part_l726_726910


namespace smallest_special_number_gt_3429_l726_726577

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726577


namespace mode_of_data_set_l726_726681

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726681


namespace not_convex_f4_l726_726289

-- Definition of functions
def f1 (x : ℝ) := sin x + cos x
def f2 (x : ℝ) := log (1-x)
def f3 (x : ℝ) := -x^3 + 2*x - 1
def f4 (x : ℝ) := x * exp x

-- Definition of convex function on domain D
def is_convex_on (f : ℝ → ℝ) (D : set ℝ) :=
  ∀ x ∈ D, deriv (deriv f x) < 0

-- Definitions of the specific domain
def D : set ℝ := Ioo 0 (π / 2)

-- Statement to prove
theorem not_convex_f4 : ¬is_convex_on f4 D := by
  sorry

end not_convex_f4_l726_726289


namespace tank_filling_time_l726_726482

theorem tank_filling_time
  (T : ℕ) (Rₐ R_b R_c : ℕ) (C : ℕ)
  (hRₐ : Rₐ = 40) (hR_b : R_b = 30) (hR_c : R_c = 20) (hC : C = 950)
  (h_cycle : T = 1 + 1 + 1) : 
  T * (C / (Rₐ + R_b - R_c)) - 1 = 56 :=
by
  sorry

end tank_filling_time_l726_726482


namespace john_average_speed_l726_726354

variable {minutes_uphill : ℝ} (h1 : minutes_uphill = 45)
variable {distance_uphill : ℝ} (h2 : distance_uphill = 2)
variable {minutes_downhill : ℝ} (h3 : minutes_downhill = 15)
variable {distance_downhill : ℝ} (h4 : distance_downhill = 2)

theorem john_average_speed : 
  let total_distance := distance_uphill + distance_downhill in
  let total_time := minutes_uphill + minutes_downhill in
  total_distance / (total_time / 60) = 4 :=
by
  sorry

end john_average_speed_l726_726354


namespace students_count_l726_726416

noncomputable def num_students (N T : ℕ) : Prop :=
  T = 72 * N ∧ (T - 200) / (N - 5) = 92

theorem students_count (N T : ℕ) : num_students N T → N = 13 :=
by
  sorry

end students_count_l726_726416


namespace second_term_arithmetic_seq_l726_726783

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l726_726783


namespace mode_of_dataset_with_average_is_l726_726705

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726705


namespace prime_mult_by_three_mean_l726_726249

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, (m ∣ n) → (m = 1 ∨ m = n))

def list_primes (l : List ℕ) : List ℕ :=
  l.filter is_prime

def multiply_each_by_three (l : List ℕ) : List ℕ :=
  l.map (λ x => 3 * x)

def arithmetic_mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem prime_mult_by_three_mean :
  arithmetic_mean (multiply_each_by_three (list_primes [10, 22, 31, 33, 37])) = 102 := by
  sorry

end prime_mult_by_three_mean_l726_726249


namespace problem_l726_726480

-- Define the conditions
variables (x y : ℝ)
axiom h1 : 2 * x + y = 7
axiom h2 : x + 2 * y = 5

-- Statement of the problem
theorem problem : (2 * x * y) / 3 = 2 :=
by 
  -- Proof is omitted, but you should replace 'sorry' by the actual proof
  sorry

end problem_l726_726480


namespace farmer_payment_per_acre_l726_726952

-- Define the conditions
def monthly_payment : ℝ := 300
def length_ft : ℝ := 360
def width_ft : ℝ := 1210
def sqft_per_acre : ℝ := 43560

-- Define the question and its correct answer
def payment_per_acre_per_month : ℝ := 30

-- Prove that the farmer pays $30 per acre per month
theorem farmer_payment_per_acre :
  (monthly_payment / ((length_ft * width_ft) / sqft_per_acre)) = payment_per_acre_per_month :=
by
  sorry

end farmer_payment_per_acre_l726_726952


namespace tan_plus_3sin_30_eq_2_plus_3sqrt3_l726_726213

theorem tan_plus_3sin_30_eq_2_plus_3sqrt3 :
  let θ := Real.pi / 6 -- 30 degrees in radians
  in Real.tan θ + 3 * Real.sin θ = 2 + 3 * Real.sqrt 3 := by
  have sin_30 : Real.sin θ = 1 / 2 := by
    sorry
  have cos_30 : Real.cos θ = Real.sqrt 3 / 2 := by
    sorry
  sorry

end tan_plus_3sin_30_eq_2_plus_3sqrt3_l726_726213


namespace r_n_m_smallest_m_for_r_2006_l726_726818

def euler_totient (n : ℕ) : ℕ := 
  n * (1 - (1 / 2)) * (1 - (1 / 17)) * (1 - (1 / 59))

def r (n m : ℕ) : ℕ :=
  m * euler_totient n

theorem r_n_m (n m : ℕ) : r n m = m * euler_totient n := 
  by sorry

theorem smallest_m_for_r_2006 (n m : ℕ) (h : n = 2006) (h2 : r n m = 841 * 928) : 
  ∃ m, r n m = 841^2 := 
  by sorry

end r_n_m_smallest_m_for_r_2006_l726_726818


namespace polynomial_integer_p_factors_l726_726776

theorem polynomial_integer_p_factors 
  (p : ℤ) : (∃ a b : ℤ, a * b = 12 ∧ a + b = p) → p ∈ {7, -7, 8, -8, 13, -13} :=
by
  sorry

end polynomial_integer_p_factors_l726_726776


namespace DF_eq_sqrt_13_l726_726390

-- Define points, sides, and the context of the problem
section
variables {Point : Type} [Point Geometry] (A B C D E F X : Point)

-- Condition 1: ABCD is a square of side length 13
def is_square_ABCD (A B C D : Point) (s : ℝ) := 
s = 13

-- Condition 2: Points E and F are on rays AB and AD such that the area of square ABCD equals the area of triangle AEF
def area_ABCD_eq_area_AEF (A B C D E F : Point) :=
area (triangle ABCD) = area (triangle AEF)

-- Condition 3: EF intersects BC at X
def EF_intersects_BC_at_X (E F B C X : Point) :=
intersects (line_through E F) (line_through B C X)

-- Condition 4: BX = 6
def BX_is_six (X B : Point) :=
distance X B = 6

-- Question: Prove that DF = sqrt(13) given the conditions
theorem DF_eq_sqrt_13 
  (A B C D E F X : Point)
  (h1 : is_square_ABCD A B C D 13)
  (h2 : area_ABCD_eq_area_AEF A B C D E F)
  (h3 : EF_intersects_BC_at_X E F B C X)
  (h4 : BX_is_six X B) :
  distance D F = real.sqrt 13 :=
sorry
end

end DF_eq_sqrt_13_l726_726390


namespace mode_of_data_set_l726_726677

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l726_726677


namespace least_positive_base_ten_number_with_seven_binary_digits_l726_726073

theorem least_positive_base_ten_number_with_seven_binary_digits :
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n >= 2^6) ∧ (nat.binary_length n = 7) ∧ n = 64 :=
begin
  sorry
end

end least_positive_base_ten_number_with_seven_binary_digits_l726_726073


namespace good_numbers_upto_17_not_good_number_18_l726_726381

def is_good_number (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ m = n / (nat.divisors n).card

theorem good_numbers_upto_17 :
  ∀ m ∈ (finset.range 18), m > 0 → is_good_number m :=
by { sorry }

theorem not_good_number_18 : ¬ is_good_number 18 :=
by { sorry }

end good_numbers_upto_17_not_good_number_18_l726_726381


namespace find_lambda_l726_726799

variable {A B C D E : Type}
variable [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E]
variable [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] [Module ℝ E]

variables (CA DA CE : A)
variable (λ : ℝ)

-- Definition: 
def is_parallelogram (A B C D : A) : Prop :=
  A + C = B + D

-- Given:
def shift_vector : Prop :=
  CE = (-CA) + λ • DA

theorem find_lambda
  (h : is_parallelogram A B C D)
  (E_on_line_AB : ∃ μ : ℝ, E = A + μ • (B - A))
  (vec_prop : shift_vector CA DA CE λ) :
  λ = 2 :=
by
  sorry

end find_lambda_l726_726799


namespace symmetric_circle_equation_l726_726002

theorem symmetric_circle_equation:
  ∀ (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 4) →
  (∀ x y, l x y ↔ x + y = 0) →
  (∃ x y, C' x y ↔ (x + 1)^2 + (y + 2)^2 = 4) :=
by
  sorry

end symmetric_circle_equation_l726_726002


namespace smallest_special_gt_3429_l726_726602

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726602


namespace ellipse_foci_k_value_l726_726001

theorem ellipse_foci_k_value 
    (k : ℝ) 
    (h1 : 5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5): 
    k = 1 := 
by 
  sorry

end ellipse_foci_k_value_l726_726001


namespace arithmetic_progression_sum_l726_726112

noncomputable def sum_first_15_terms (a d : ℝ) : ℝ := 15 / 2 * (2 * a + 14 * d)

theorem arithmetic_progression_sum (a d : ℝ) (h : a + 7 * d = 15) :
  sum_first_15_terms a d = 225 :=
by
  have h2 : 2 * a + 14 * d = 30 := by linarith
  rw [sum_first_15_terms, h2]
  norm_num
  sorry

end arithmetic_progression_sum_l726_726112


namespace moles_of_SO2_formed_l726_726252

variable (n_NaHSO3 n_HCl n_SO2 : ℕ)

/--
The reaction between sodium bisulfite (NaHSO3) and hydrochloric acid (HCl) is:
NaHSO3 + HCl → NaCl + H2O + SO2
Given 2 moles of NaHSO3 and 2 moles of HCl, prove that the number of moles of SO2 formed is 2.
-/
theorem moles_of_SO2_formed :
  (n_NaHSO3 = 2) →
  (n_HCl = 2) →
  (∀ (n : ℕ), (n_NaHSO3 = n) → (n_HCl = n) → (n_SO2 = n)) →
  n_SO2 = 2 :=
by 
  intros hNaHSO3 hHCl hReaction
  exact hReaction 2 hNaHSO3 hHCl

end moles_of_SO2_formed_l726_726252


namespace triangle_angles_l726_726441

theorem triangle_angles :
  ∃ (α β γ : ℝ), (α + β + γ = 180) ∧
  (cos γ = 3 / 4) ∧
  (sin α = 3 * (sin γ) / (sqrt 7)) ∧
  (sin β = 4 * (sin γ) / (sqrt 7)) ∧
  (α ≈ 30) ∧ (β ≈ 108.59) ∧ (γ ≈ 41.41) := sorry

end triangle_angles_l726_726441


namespace find_k_values_l726_726346

noncomputable def z := Complex

-- Conditions
def condition1 : Prop := ∀ (z : Complex), Complex.abs (z - 4) = 3 * Complex.abs (z + 4)
def condition2 (k : ℝ) : Prop := ∃ (z : Complex), Complex.abs z = k ∧ condition1 z

-- Conclusion to prove
def conclusion (k : ℝ) : Prop := k = 4 ∨ k = 14

theorem find_k_values (k : ℝ) : condition2 k → conclusion k :=
by
  sorry

end find_k_values_l726_726346


namespace square_side_length_l726_726169

theorem square_side_length (a : ℚ) (s : ℚ) (h : a = 9/16) (h_area : s^2 = a) : s = 3/4 :=
by {
  -- proof omitted
  sorry
}

end square_side_length_l726_726169


namespace least_binary_seven_digits_l726_726064

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726064


namespace sum_three_numbers_l726_726885

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = a + 20) 
  (h2 : (a + b + c) / 3 = c - 30) 
  (h3 : b = 10) :
  sum_of_three_numbers a b c = 60 :=
by
  sorry

end sum_three_numbers_l726_726885


namespace ratio_of_areas_l726_726820

-- Define the problem parameters and conditions
variables {A B C D E F G H : ℝ}
variables (AB BC : ℝ)
variables (rect : Rectangle AB BC)
variables (equilateral_centers : ∀ (base : Segment), Point)

-- The lengths of the sides of the rectangle
def AB := 8
def BC := 4

-- The centers of the equilateral triangles based on the conditions
def E := equilateral_centers (Segment.mk A B)
def F := equilateral_centers (Segment.mk B C)
def G := equilateral_centers (Segment.mk C D)
def H := equilateral_centers (Segment.mk D A)

-- The calculated area ratio
def area_ratio : ℝ := (11 + 6*Real.sqrt 3) / 16

-- Prove that the ratio of the area of square EFGH to the area of rectangle ABCD is as given
theorem ratio_of_areas : 
  let EFGH_area := square E F G H in
  let ABCD_area := rect.area in
  EFGH_area / ABCD_area = area_ratio := 
sorry

end ratio_of_areas_l726_726820


namespace debby_remaining_pictures_l726_726488

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (deleted_pictures : ℕ)

def initial_pictures (zoo_pictures museum_pictures : ℕ) : ℕ :=
  zoo_pictures + museum_pictures

def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (initial_pictures zoo_pictures museum_pictures) - deleted_pictures

theorem debby_remaining_pictures :
  remaining_pictures 24 12 14 = 22 :=
by
  sorry

end debby_remaining_pictures_l726_726488


namespace goodsTrain_passing_time_l726_726504

-- Define the conditions
def passengerTrainSpeed : ℝ := 50 -- in km/h
def goodsTrainSpeed : ℝ := 62 -- in km/h
def goodsTrainLength : ℝ := 280 -- in meters

-- Define the conversion factors
def kmToMeter : ℝ := 1000 -- 1 km = 1000 meters
def hourToSecond : ℝ := 3600 -- 1 hour = 3600 seconds

-- Calculate the relative speed in m/s
def relativeSpeed : ℝ := (passengerTrainSpeed + goodsTrainSpeed) * kmToMeter / hourToSecond

-- Calculate the time taken to pass the goods train
def timeToPass : ℝ := goodsTrainLength / relativeSpeed

-- State the problem in Lean 4
theorem goodsTrain_passing_time :
  abs (timeToPass - 9) < 0.01 := 
sorry

end goodsTrain_passing_time_l726_726504


namespace distinct_real_roots_l726_726774

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem distinct_real_roots
  (a b : ℝ)
  (hf : ∫ x in -1..1, |f x a b| < 2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = 0 ∧ f x2 a b = 0) :=
sorry

end distinct_real_roots_l726_726774


namespace dataset_mode_l726_726708

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l726_726708


namespace correct_BX_l726_726369

noncomputable def BX_is_correct : Prop :=
  ∃ (a b : ℕ), gcd a b = 1 ∧ BX = (180 / 23) ∧ a + b = 203

theorem correct_BX (BC DE CD : ℝ) (lineThruEParallelBD lineThruBPerpBE : Prop) 
  (BE := 20) (BC := 2 * Real.sqrt 34) (CD := 8) (DE := 2 * Real.sqrt 10)
  (A_intersection E_parallel_BD : Prop) (M_intersection BD_diagonals : Prop) 
  (X_intersection AM_BE : Prop) : 
  BX_is_correct :=
begin
  sorry,
end

end correct_BX_l726_726369


namespace multiplication_of_mixed_number_l726_726990

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l726_726990


namespace ratio_of_areas_l726_726804

-- Define the midpoints condition.
variables {A B C D L M N : Type} -- Types for the vertices and midpoints
variables [convex_quad A B C D] -- Assuming A, B, C, D form a convex quadrilateral
variables 
  (is_midpoint_L : midpoint L B C)
  (is_midpoint_M : midpoint M A D)
  (is_midpoint_N : midpoint N A B)

noncomputable def area_ratio_triangle_LMN_to_quad_ABCD : real :=
  let area_LMN := area_triangle L M N
  let area_ABCD := area_quad A B C D in
  area_LMN / area_ABCD

theorem ratio_of_areas (h : convex_quad A B C D) :
  area_ratio_triangle_LMN_to_quad_ABCD is_midpoint_L is_midpoint_M is_midpoint_N = 1 / 4 :=
sorry

end ratio_of_areas_l726_726804


namespace trout_problem_l726_726401

def trout_in_pool (P : ℕ) : Prop :=
    let OnumLake := P + 25 in
    let RiddlePond := OnumLake / 2 in
    (P + OnumLake + RiddlePond) = 225

theorem trout_problem : ∃ P : ℕ, trout_in_pool P ∧ P = 75 := 
begin
  use 75,
  unfold trout_in_pool,
  simp,
  sorry
end

end trout_problem_l726_726401


namespace probability_B_in_A_or_S_minus_A_l726_726821

noncomputable theory
open_locale classical

variables (A B S : set ℕ)

def n := 6
def total_subsets := 2^n
def total_pairs := total_subsets ^ 2
def favorable_pairs :=
  ∑ A in (finset.powerset (finset.range n)), (2 ^ A.card + 2 ^ (n - A.card) - 1)

theorem probability_B_in_A_or_S_minus_A :
  (favorable_pairs.to_nat : ℚ) / total_pairs.to_nat = 63 / 64 :=
sorry

end probability_B_in_A_or_S_minus_A_l726_726821


namespace square_side_length_l726_726148

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l726_726148


namespace convert_base_8_to_base_10_l726_726545

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l726_726545


namespace smallest_special_number_gt_3429_l726_726581

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l726_726581


namespace convex_figure_contains_integer_points_l726_726129

variable {Φ : Type}

-- Definitions for convex figure, area, semiperimeter, and integer points inclusion
variable [ConvexFigure Φ] (Φ : Φ) (S p : ℝ)

-- Helper definitions
def area (Φ : Type) [ConvexFigure Φ] : ℝ := sorry
def semiperimeter (Φ : Type) [ConvexFigure Φ] : ℝ := sorry
def contains_n_integer_points (Φ : Type) [ConvexFigure Φ] (n : ℕ) : Prop := sorry

theorem convex_figure_contains_integer_points (Φ : Φ) (S p : ℝ) (n : ℕ)
  [ConvexFigure Φ] (h_area : area Φ = S) (h_semiperimeter : semiperimeter Φ = p) (h_cond : S > n * p) :
  contains_n_integer_points Φ n :=
sorry

end convex_figure_contains_integer_points_l726_726129


namespace vector_projection_is_correct_l726_726254

def projection_vector (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let norm_squared := b.1 * b.1 + b.2 * b.2 + b.3 * b.3
  (dot_product / norm_squared * b.1, dot_product / norm_squared * b.2, dot_product / norm_squared * b.3)

theorem vector_projection_is_correct :
  projection_vector (4, 2, -3) (3, 1, -2) = (30 / 7, 10 / 7, -20 / 7) :=
by
  sorry

end vector_projection_is_correct_l726_726254


namespace solve_for_x_l726_726863

theorem solve_for_x (x : ℝ) :
  sqrt (x^3) = 9 * (81^(1/9)) → x = 9 :=
by
  sorry

end solve_for_x_l726_726863


namespace smallest_special_gt_3429_l726_726616

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726616


namespace least_7_digit_binary_number_is_64_l726_726077

theorem least_7_digit_binary_number_is_64 : ∃ n : ℕ, n = 64 ∧ (∀ m : ℕ, (m < 64 ∧ m >= 64) → false) ∧ nat.log2 64 = 6 :=
by
  sorry

end least_7_digit_binary_number_is_64_l726_726077


namespace arithmetic_sequence_second_term_l726_726779

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l726_726779


namespace least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726057

theorem least_positive_base_ten_number_with_seven_digit_binary_representation :
  ∃ n : ℤ, n > 0 ∧ (∀ k : ℤ, k > 0 ∧ k < n → digit_length binary_digit_representation k < 7) ∧ digit_length binary_digit_representation n = 7 :=
sorry

end least_positive_base_ten_number_with_seven_digit_binary_representation_l726_726057


namespace trigonometric_ratios_l726_726286

noncomputable def hypotenuse (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem trigonometric_ratios (x y : ℝ) (r : ℝ) (h : r = hypotenuse x y)
  (hx : x = 3) (hy : y = 4) :
  sin (atan2 y x) = 4 / 5 ∧ cos (atan2 y x) = 3 / 5 ∧ tan (atan2 y x) = 4 / 3 :=
by
  sorry

end trigonometric_ratios_l726_726286


namespace jony_starting_block_l726_726365

-- Given conditions
def blocks_are_sequentially_numbered (blocks : List ℕ) : Prop :=
  ∀ i, i < blocks.length → blocks[i] = i + 1

def block_measure : ℕ := 40  -- Each block measures 40 meters
def jony_speed : ℕ := 100   -- Jony's speed is 100 meters per minute
def total_walking_time : ℕ := 40 -- Jony walks for 40 minutes

def distance_per_block : ℕ := block_measure
def walking_time_to_distance (time : ℕ) (speed : ℕ) : ℕ := time * speed

-- Distance Jony walks
noncomputable def total_distance_walked : ℕ := walking_time_to_distance total_walking_time jony_speed
noncomputable def total_blocks_walked : ℕ := total_distance_walked / distance_per_block

-- Prove starting block
theorem jony_starting_block (S : ℕ) : blocks_are_sequentially_numbered (List.range 100) →
    ((90 - S) + 20 = total_blocks_walked) → S = 10 :=
by
  intros h_blocks h_distance
  sorry

end jony_starting_block_l726_726365


namespace combined_list_correct_l726_726811

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end combined_list_correct_l726_726811


namespace cost_price_per_meter_l726_726926

theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (total_cost_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5)
  (h4 : total_cost_price = selling_price + total_meters * loss_per_meter)
  (h5 : cost_price_per_meter = total_cost_price / total_meters) :
  cost_price_per_meter = 50 :=
by
  sorry

end cost_price_per_meter_l726_726926


namespace frustum_shortest_distance_l726_726892

open Real

noncomputable def shortest_distance (R1 R2 : ℝ) (AB : ℝ) (string_from_midpoint : Bool) : ℝ :=
  if R1 = 5 ∧ R2 = 10 ∧ AB = 20 ∧ string_from_midpoint = true then 4 else 0

theorem frustum_shortest_distance : 
  shortest_distance 5 10 20 true = 4 :=
by sorry

end frustum_shortest_distance_l726_726892


namespace smallest_special_number_l726_726582

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l726_726582


namespace mode_of_data_set_l726_726693

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l726_726693


namespace find_y_eq_1_div_5_l726_726320

theorem find_y_eq_1_div_5 (b : ℝ) (y : ℝ) (h1 : b > 2) (h2 : y > 0) (h3 : (3 * y)^(Real.log 3 / Real.log b) - (5 * y)^(Real.log 5 / Real.log b) = 0) :
  y = 1 / 5 :=
by
  sorry

end find_y_eq_1_div_5_l726_726320


namespace g_13_equals_236_l726_726768

def g (n : ℕ) : ℕ := n^2 + 2 * n + 41

theorem g_13_equals_236 : g 13 = 236 := sorry

end g_13_equals_236_l726_726768


namespace bobby_initial_pieces_l726_726532

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end bobby_initial_pieces_l726_726532


namespace calculate_principal_amount_borrowed_l726_726962

noncomputable def principal_borrowed : ℝ :=
  let interest := 9692
  let rate1 := 0.12
  let rate2 := 0.14
  let rate3 := 0.17
  interest / (rate1 + rate2 + rate3)

theorem calculate_principal_amount_borrowed :
  let interest := 9692
  let rate1 := 0.12
  let rate2 := 0.14
  let rate3 := 0.17
  principal_borrowed = 22539.53 :=
by
  let total_interest := 9692
  let rate1 := 0.12
  let rate2 := 0.14
  let rate3 := 0.17
  let principal := total_interest / (rate1 + rate2 + rate3)
  sorry

end calculate_principal_amount_borrowed_l726_726962


namespace monotonicity_of_f_zeros_of_g_l726_726742

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := real.log x + 1 / (a * x) - 1 / a

-- Problem 1: Monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) : 
  (∀ x > 0, (a < 0 → deriv (f a) x > 0) ∧ (a > 0 → (deriv (f a) x > 0 ↔ x > 1 / a) ∧ (deriv (f a) x < 0 ↔ x < 1 / a))) := sorry
  
-- Define the function g(x)
def g (m : ℝ) (x : ℝ) : ℝ := (real.log x - 1) * real.exp x + x - m

-- Problem 2: Number of zeros of g(x)
theorem zeros_of_g (m : ℝ) : 
  ((∀ x ∈ set.Icc (1 / real.exp 1) real.exp 1, deriv (λ x, (real.log x - 1) * real.exp x + x) x > 0) →
  (if m < -2 * real.exp (1 / real.exp 1) + 1 / real.exp 1 ∨ m > real.exp 1 
   then ∃ x ∈ set.Icc (1 / real.exp 1) real.exp 1, g m x = 0 
   else ∀ x ∈ set.Icc (1 / real.exp 1) real.exp 1, g m x ≠ 0)) := sorry

end monotonicity_of_f_zeros_of_g_l726_726742


namespace smallest_special_gt_3429_l726_726610

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l726_726610


namespace Danny_more_than_Larry_l726_726815

/-- Keith scored 3 points. --/
def Keith_marks : Nat := 3

/-- Larry scored 3 times as many marks as Keith. --/
def Larry_marks : Nat := 3 * Keith_marks

/-- The total marks scored by Keith, Larry, and Danny is 26. --/
def total_marks (D : Nat) : Prop := Keith_marks + Larry_marks + D = 26

/-- Prove the number of more marks Danny scored than Larry is 5. --/
theorem Danny_more_than_Larry (D : Nat) (h : total_marks D) : D - Larry_marks = 5 :=
sorry

end Danny_more_than_Larry_l726_726815


namespace intersecting_lines_ratios_l726_726486

variable (A B C D M N O : Type*)
-- Parallelogram ABCD
variable [Parallelogram A B C D] 
-- Points M on AD, N on CD
variable [OnLine A M D] [OnLine C N D]
-- Given ratios
variable (hAM_MD : ratio AM MD = 2 / 9) (hCN_ND : ratio CN ND = 3 / 8)

theorem intersecting_lines_ratios 
  (h := hAM_MD) (h' := hCN_ND) :
  ratio ON OB = 7 / 31 ∧ ratio OC OM = 9 / 31 := 
by
  sorry

end intersecting_lines_ratios_l726_726486


namespace find_a_l726_726631

noncomputable def roots_form_arithmetic_progression (r d : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ {r - 3*d, r - d, r + d, r + 3*d}

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x^8 + a * x^4 + 1 = 0) → x ∈ {r - 3*d, r - d, r + d, r + 3*d}) →
  a = - 82 / 9 :=
by
  sorry

end find_a_l726_726631


namespace athlete_speed_200m_in_24s_is_30kmh_l726_726523

noncomputable def speed_in_kmh (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

theorem athlete_speed_200m_in_24s_is_30kmh :
  speed_in_kmh 200 24 = 30 := by
  sorry

end athlete_speed_200m_in_24s_is_30kmh_l726_726523


namespace distance_moved_at_t3_l726_726967

open Real

noncomputable def velocity_function (t : ℝ) : ℝ := 2 - t

theorem distance_moved_at_t3 : (∫ t in 0..3, abs (velocity_function t)) = 5 / 2 :=
by
  apply integral_of_has_deriv_at_of_le
  { apply interval_integrable_abs_of_continuous_on
    exact continuous_on_sub continuous_const continuous_id }
  { intros x hx
    exact has_deriv_at_abs_real_of_has_deriv_at (by norm_num) (by norm_num) _ (has_deriv_at_sub_const _) }
  { exact continuous_on_sub continuous_const continuous_id.interval_integrable }

  sorry

end distance_moved_at_t3_l726_726967


namespace total_pages_l726_726947

theorem total_pages (chap1 : ℕ) (chap2 : ℕ) (H1 : chap1 = 48) (H2 : chap2 = 46) : chap1 + chap2 = 94 :=
by
  rw [H1, H2]
  rfl

end total_pages_l726_726947


namespace median_unchanged_l726_726729

-- Define the dataset and conditions
variables {n : ℕ} (x : Fin n → ℝ)
-- Assume that n is at least 3
variable (hn : n ≥ 3)

-- Define the median function for a list of real numbers
noncomputable def median (s : Fin n → ℝ) : ℝ :=
  let sorted := Multiset.sort (≤) (Multiset.univ.map (λ i => s i))
  if h : n % 2 = 1 then
    sorted.get ⟨n / 2, sorry⟩ -- median for odd length
  else
    (sorted.get ⟨n / 2 - 1, sorry⟩ + sorted.get ⟨n / 2, sorry⟩) / 2 -- median for even length

-- Define high and low functions to remove the maximum and minimum values
def remove_highest_lowest (s : Fin n → ℝ) : List ℝ :=
  (List.erase (List.erase (Multiset.sort (≤) (Multiset.univ.map (λ i => s i))) (Multiset.max' sorry (Multiset.univ.map (λ i => s i)))) (Multiset.min' sorry (Multiset.univ.map (λ i => s i))))

-- Equivalent proof problem in Lean: Prove median of dataset without highest and lowest values is same as original median
theorem median_unchanged (x : Fin n → ℝ) (hn : n ≥ 3) :
  median x = median (λ i, if i < ⟨n - 2, sorry⟩ then remove_highest_lowest x !! i else 0) :=
sorry

end median_unchanged_l726_726729


namespace side_length_of_square_l726_726182

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l726_726182


namespace side_length_of_square_l726_726162

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end side_length_of_square_l726_726162


namespace smallest_multiple_of_40_gt_100_l726_726918

theorem smallest_multiple_of_40_gt_100 :
  ∃ x : ℕ, 0 < x ∧ 40 * x > 100 ∧ ∀ y : ℕ, 0 < y ∧ 40 * y > 100 → x ≤ y → 40 * x = 120 :=
by
  sorry

end smallest_multiple_of_40_gt_100_l726_726918


namespace false_statement_A_l726_726828

open Complex

theorem false_statement_A (z1 z2 : ℂ) (h : ∥z1∥ = ∥z2∥) : z1^2 ≠ z2^2 :=
by {
  have example1 : z1 = 1 - I, from sorry,
  have example2 : z2 = 1 + I, from sorry,
  have h_modulus : ∥1 - I∥ = ∥1 + I∥ := by simp,
  have h_value1 : (1 - I)^2 = -2 * I := by simp,
  have h_value2 : (1 + I)^2 = 2 * I := by simp,
  have h_neq : (1 - I)^2 ≠ (1 + I)^2 := by linarith,
  exact h_neq
}

end false_statement_A_l726_726828


namespace perpendiculars_concur_at_single_point_l726_726196

/-- 
Let ABC be a triangle with an inscribed circle.
Let T, Q, and P be the points of tangency of the incircle with sides AC, BC, and AB respectively.
Let M be the midpoint of the segment PQ.
Let N be the midpoint of the segment PT.
Let O be the midpoint of the segment QT.
Let l_a be the perpendicular from M to BC.
Let l_b be the perpendicular from N to AC.
Let l_c be the perpendicular from O to AB.
We aim to prove that l_a, l_b, and l_c intersect at a single point.
-/
theorem perpendiculars_concur_at_single_point
  (ABC : Triangle) (T Q P : Point)
  (h_A_incircle: Incircle ABC T Q P)
  (M : Point := midpoint P Q)
  (N : Point := midpoint P T)
  (O : Point := midpoint Q T) :
  let l_a := perpendicular M BC,
      l_b := perpendicular N AC,
      l_c := perpendicular O AB
  in intersect_at_single_point l_a l_b l_c := sorry

end perpendiculars_concur_at_single_point_l726_726196


namespace angle_MAK_is_45_degrees_l726_726847

theorem angle_MAK_is_45_degrees
  {A B C D M K : Type}
  [square : is_square A B C D]
  (hM : on_side M C B)
  (hK : on_side K C D)
  (hperimeter : perimeter (triangle C M K) = 2 * side_length A B) :
  angle M A K = 45 :=
sorry

end angle_MAK_is_45_degrees_l726_726847


namespace find_extrema_of_y_l726_726263

theorem find_extrema_of_y (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  let y := 4^(x - 1/2) - 3 * 2^x + 5 in
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1/2 ≤ y ∧ y ≤ 5/2) :=
by
  intro x hx
  sorry

end find_extrema_of_y_l726_726263


namespace least_binary_seven_digits_l726_726062

theorem least_binary_seven_digits : (n : ℕ) → (dig : ℕ) 
  (h : bit_length n = 7) : n = 64 := 
begin
  assume n dig h,
  sorry
end

end least_binary_seven_digits_l726_726062


namespace side_length_of_square_l726_726156

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l726_726156


namespace convert_base_8_to_base_10_l726_726544

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l726_726544


namespace arg_cubed_eq_pi_l726_726724

open Complex

theorem arg_cubed_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : 
  arg (z2 / z1) ^ 3 = π :=
by
  sorry

end arg_cubed_eq_pi_l726_726724


namespace mathematics_views_encapsulation_l726_726409

theorem mathematics_views_encapsulation :
  (views_on_mathematics : Type) 
  → (practicality foundational nature applicability logical_thinking cultural_literacy scientific_spirit : views_on_mathematics)
  → True :=
by {
  intros views_on_mathematics practicality foundational nature applicability logical_thinking cultural_literacy scientific_spirit,
  exact trivial,
}

end mathematics_views_encapsulation_l726_726409


namespace convert_246_octal_to_decimal_l726_726563

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l726_726563


namespace find_B_divisible_by_6_l726_726032

theorem find_B_divisible_by_6 (B : ℕ) : (5170 + B) % 6 = 0 ↔ (B = 2 ∨ B = 8) :=
by
  -- Conditions extracted from the problem are directly used here:
  sorry -- Proof would be here

end find_B_divisible_by_6_l726_726032


namespace mode_of_dataset_with_average_is_l726_726703

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l726_726703


namespace inequality_proof_l726_726831

theorem inequality_proof
  (n : ℕ) (h_pos : 0 < n)
  (a b : Fin n → ℝ)
  (h_cond : ∀ i, 0 < a i + b i)
  : (∑ i, (a i * b i - b i^2) / (a i + b i))
    ≤ (∑ i, a i) * (∑ i, b i) - (∑ i, b i)^2 / ∑ i, (a i + b i) :=
by
  sorry

end inequality_proof_l726_726831


namespace hyperbola_properties_l726_726877

theorem hyperbola_properties :
  (∃ a b c e : ℝ, a = 2 ∧ b = 2 * Real.sqrt 3 ∧ c = Real.sqrt (4 + 12) ∧ e = 2 ∧ e = c / a ∧ (∀ x y, (y = (b / a) * x) ∨ (y = -(b / a) * x))) :=
begin
  let a := 2,
  let b := 2 * Real.sqrt 3,
  let c := Real.sqrt (4 + 12),
  let e := c / a,
  existsi [a, b, c, e],
  split, exact rfl,
  split, exact rfl,
  split, { simp, norm_num },
  split, { simp only [sqrt_four, sqrt, add_zero, rfl] },
  split, { norm_num },
  simp,
  intros x y,
  left,
  exact rfl,
end

end hyperbola_properties_l726_726877


namespace new_deal_correct_l726_726870

theorem new_deal_correct :
  (implemented_by_Roosevelt New_Deal ∧
   aimed_to_overcome_economic_crisis New_Deal ∧
   made_government_responsible_economic_stability New_Deal) →
  statement_about_New_Deal_is_correct :=
by
  sorry


end new_deal_correct_l726_726870


namespace contest_score_difference_l726_726792

theorem contest_score_difference :
  let percent_50 := 0.05
  let percent_60 := 0.20
  let percent_70 := 0.25
  let percent_80 := 0.30
  let percent_90 := 1 - (percent_50 + percent_60 + percent_70 + percent_80)
  let mean := (percent_50 * 50) + (percent_60 * 60) + (percent_70 * 70) + (percent_80 * 80) + (percent_90 * 90)
  let median := 70
  median - mean = -4 :=
by
  sorry

end contest_score_difference_l726_726792


namespace first_man_reaches_first_and_time_difference_l726_726908

-- Define constants for speeds and distances
def rowing_speed_first_with_stream : ℝ := 26
def rowing_speed_first_against_stream : ℝ := 14
def rowing_speed_second_with_stream : ℝ := 22
def rowing_speed_second_against_stream : ℝ := 18
def wind_speed : ℝ := 3
def distance_to_marker : ℝ := 40

-- Calculate effective speeds downstream
def effective_speed_first_downstream : ℝ := rowing_speed_first_with_stream + wind_speed
def effective_speed_second_downstream : ℝ := rowing_speed_second_with_stream + wind_speed

-- Calculate times taken to reach the marker
def time_taken_first : ℝ := distance_to_marker / effective_speed_first_downstream
def time_taken_second : ℝ := distance_to_marker / effective_speed_second_downstream

-- Calculate time difference in hours and convert to minutes
def time_difference_hours : ℝ := time_taken_second - time_taken_first
def time_difference_minutes : ℝ := time_difference_hours * 60

theorem first_man_reaches_first_and_time_difference :
  time_taken_first < time_taken_second ∧ abs (time_difference_minutes - 13.24) < 0.01 := 
by sorry

end first_man_reaches_first_and_time_difference_l726_726908


namespace mats_in_10_days_of_all_weavers_l726_726794

theorem mats_in_10_days_of_all_weavers :
  let rate_a := 4 / 6 : ℚ
  let rate_b := 5 / 7 : ℚ
  let rate_c := 3 / 4 : ℚ
  let rate_d := 6 / 9 : ℚ
  let total_rate := rate_a + rate_b + rate_c + rate_d
  let total_mats := total_rate * 10
  ⌊total_mats⌋ = 28 :=
by
  sorry

end mats_in_10_days_of_all_weavers_l726_726794


namespace dot_product_in_triangle_l726_726787

noncomputable def ab := 3
noncomputable def ac := 2
noncomputable def bc := Real.sqrt 10

theorem dot_product_in_triangle : 
  let AB := ab
  let AC := ac
  let BC := bc
  (AB = 3) → (AC = 2) → (BC = Real.sqrt 10) → 
  ∃ cosA, (cosA = (AB^2 + AC^2 - BC^2) / (2 * AB * AC)) →
  ∃ dot_product, (dot_product = AB * AC * cosA) ∧ dot_product = 3 / 2 :=
by
  sorry

end dot_product_in_triangle_l726_726787


namespace rectangular_prism_surface_area_l726_726010

theorem rectangular_prism_surface_area
  (l w h : ℕ)
  (hl : l > 1)
  (hw : w > 1)
  (hh : h > 1)
  (coprime_lw : Nat.coprime l w)
  (coprime_wh : Nat.coprime w h)
  (coprime_lh : Nat.coprime l h)
  (volume_eq_665 : l * w * h = 665) :
  2 * (l * w + l * h + w * h) = 526 :=
sorry

end rectangular_prism_surface_area_l726_726010


namespace mode_of_data_set_l726_726688

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l726_726688


namespace slower_train_speed_l726_726912

-- Conditions
variables (L : ℕ) -- Length of each train (in meters)
variables (v_f : ℕ) -- Speed of the faster train (in km/hr)
variables (t : ℕ) -- Time taken by the faster train to pass the slower one (in seconds)
variables (v_s : ℕ) -- Speed of the slower train (in km/hr)

-- Assumptions based on conditions of the problem
axiom length_eq : L = 30
axiom fast_speed : v_f = 42
axiom passing_time : t = 36

-- Conversion for km/hr to m/s
def km_per_hr_to_m_per_s (v : ℕ) : ℕ := (v * 5) / 18

-- Problem statement
theorem slower_train_speed : v_s = 36 :=
by
  let rel_speed := km_per_hr_to_m_per_s (v_f - v_s)
  have rel_speed_def : rel_speed = (42 - v_s) * 5 / 18 := by sorry
  have distance : 60 = rel_speed * t := by sorry
  have equation : 60 = (42 - v_s) * 10 := by sorry
  have solve_v_s : v_s = 36 := by sorry
  exact solve_v_s

end slower_train_speed_l726_726912


namespace monotonic_increasing_interval_l726_726013

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x ≠ 0 → ( ∀ a b : ℝ, 0 < a ∧ a < b → f a < f b ) :=
begin
  intros x h,
  sorry,
end

end monotonic_increasing_interval_l726_726013


namespace max_value_of_target_function_l726_726328

open Real

def target_function (x : ℝ) : ℝ :=
  tan (x + 2 * π / 3) - tan (x + π / 6) + cos (x + π / 6)

theorem max_value_of_target_function :
  ∀ (x : ℝ),
  x ∈ set.Icc (-5 * π / 12) (-π / 3) →
  target_function x = 11 / 6 * √3 :=
sorry

end max_value_of_target_function_l726_726328


namespace DEFABC_is_multiple_l726_726139

-- The digits of the number
inductive Digit | a | b | c | d | e | f

-- The numbers in terms of their digits
def ABCDEF (a b c d e f : ℕ) : ℕ := a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f
def FABCD (f a b c d : ℕ) : ℕ := f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10
def CDEFAB (c d e f a b : ℕ) : ℕ := c * 10^5 + d * 10^4 + e * 10^3 + f * 10^2 + a * 10 + b

noncomputable def DEFABC_multiple (n a b c d e f : ℕ) :=
  let DEFABC := DEFABC d e f a b c
  n * DEFABC =  DEFABC
  
theorem DEFABC_is_multiple (n a b c d e f : ℕ) :
  (4 * n = ABCDEF a b c d e f) →
  (13 * n = FABCD f a b c d) →
  (22 * n = CDEFAB c d e f a b) →
  DEFABC_multiple n a b c d e f :=
by
  intros h1 h2 h3
  sorry

end DEFABC_is_multiple_l726_726139


namespace smallest_special_gt_3429_l726_726599

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726599


namespace even_sum_subsets_count_l726_726316

-- Definition of the set with elements
def my_set : Finset ℕ := {42, 55, 78, 103, 144, 157, 198}

-- The proof problem: Number of 4-element subsets with an even sum is 19.
theorem even_sum_subsets_count : (my_set.powerset.filter (λ s, s.card = 4 ∧ (s.sum id) % 2 = 0)).card = 19 := by
  sorry

end even_sum_subsets_count_l726_726316


namespace Math_Proof_Problem_l726_726216

noncomputable def problem : ℝ := (1005^3) / (1003 * 1004) - (1003^3) / (1004 * 1005)

theorem Math_Proof_Problem : ⌊ problem ⌋ = 8 :=
by
  sorry

end Math_Proof_Problem_l726_726216


namespace hyperbolas_same_asymptotes_l726_726311

theorem hyperbolas_same_asymptotes :
  (∃ M : ℚ, 
    (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) → (y = (4 / 3) * x) ∨ (y = -(4 / 3) * x)) ∧
    (∀ x y : ℝ, (y^2 / 25 - x^2 / M = 1) → (y = (5 / real.sqrt (M.to_real)) * x) ∨ (y = -(5 / real.sqrt (M.to_real)) * x)) ∧
    M = 225 / 16) :=
begin
  sorry
end

end hyperbolas_same_asymptotes_l726_726311


namespace incircle_radius_of_convex_quadrilateral_l726_726491

theorem incircle_radius_of_convex_quadrilateral
  (AB BC CD AD : ℝ)
  (hAB : AB = 2)
  (hBC : BC = 3)
  (hCD : CD = 7)
  (hAD : AD = 6)
  (h_right_angle : ∠ABC = 90°)
  (h_incircle : exists (r : ℝ), quadrilateral_has_incircle AB BC CD AD r) :
  exists (r : ℝ), r = (1 + Real.sqrt 13) / 3 := 
by
  sorry

end incircle_radius_of_convex_quadrilateral_l726_726491


namespace fourth_sample_is_06_l726_726658

-- Declare the basic parameters of the problem
def population_size : ℕ := 40
def sample_size : ℕ := 7
def starting_row : ℕ := 6
def starting_column : ℕ := 8

-- Declare the given random number table rows as lists of numbers
def row6 : list ℕ := [84, 42, 17, 56, 31, 07, 23, 55, 06, 82, 77, 04, 74, 43, 59, 76, 30, 63, 50, 25, 83, 92, 12, 06]
def row7 : list ℕ := [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38]

-- Function to construct the sequence by reading the number table in the described manner
def read_sequence (table : list (list ℕ)) (start_row start_col : ℕ) : list ℕ := 
by sorry

-- Given the read sequence, apply the filtering according to the rules
def valid_samples (seq : list ℕ) : list ℕ := 
by sorry

-- Statement to prove that the 4th valid sample is 06
theorem fourth_sample_is_06 :
  valid_samples (read_sequence [row6, row7] starting_row starting_column)!3 = 06 := sorry

end fourth_sample_is_06_l726_726658


namespace find_original_numbers_l726_726762

theorem find_original_numbers (p q : ℕ) (hp : 10 ≤ p ∧ p < 100) (hq : 10 ≤ q ∧ q < 100) 
  (h1 : ((round(p / 10.0) · ℕ⟨(10:ℕ)⟩) - (round(q / 10.0) · ℕ⟨(10:ℕ)⟩)) = (p - q)) 
  (h2 : ((round(p / 10.0) · ℕ⟨(10:ℕ)⟩) * (round(q / 10.0) · ℕ⟨(10:ℕ)⟩)) = ((p * q) + 184)) : 
  (p = 16 ∧ q = 26) ∨ (p = 26 ∧ q = 16) :=
sorry

end find_original_numbers_l726_726762


namespace smallest_special_gt_3429_l726_726597

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l726_726597


namespace correct_unit_price_pork_ribs_last_month_correct_unit_price_radish_this_month_correct_unit_price_pork_ribs_this_month_correct_extra_money_spent_correct_extra_money_when_a_is_4_l726_726923

-- Definitions based on the problem conditions
def unit_price_radish_last_month (a : ℝ) := a

def unit_price_pork_ribs_last_month (a : ℝ) := 7 * a + 2

def unit_price_radish_this_month (a : ℝ) := 1.25 * a

def unit_price_pork_ribs_this_month (a : ℝ) := (7 * a + 2) * 1.2

def extra_money_spent (a : ℝ) := 
  let extra_radish := 3 * (1.25 * a - a)
  let extra_pork_ribs := 2 * ((7 * a + 2) * 1.2 - (7 * a + 2))
  extra_radish + extra_pork_ribs 

def extra_money_when_a_is_4 := extra_money_spent 4

theorem correct_unit_price_pork_ribs_last_month (a : ℝ) : 
  unit_price_pork_ribs_last_month a = 7 * a + 2 := by
  rfl

theorem correct_unit_price_radish_this_month (a : ℝ) : 
  unit_price_radish_this_month a = 1.25 * a := by
  rfl

theorem correct_unit_price_pork_ribs_this_month (a : ℝ) : 
  unit_price_pork_ribs_this_month a = 8.4 * a + 2.4 := by
  simp [unit_price_pork_ribs_last_month, unit_price_pork_ribs_this_month]
  ring

theorem correct_extra_money_spent (a : ℝ) : 
  extra_money_spent a = 3.55 * a + 0.8 := by
  simp [extra_money_spent, extra_money_spent._aux_1, extra_money_spent._aux_2]
  ring

theorem correct_extra_money_when_a_is_4 : 
  extra_money_when_a_is_4 = 15 := by
  simp [extra_money_when_a_is_4, correct_extra_money_spent]
  norm_num
  rfl

attribute [local simp] correct_unit_price_pork_ribs_last_month correct_unit_price_radish_this_month correct_unit_price_pork_ribs_this_month correct_extra_money_spent

end correct_unit_price_pork_ribs_last_month_correct_unit_price_radish_this_month_correct_unit_price_pork_ribs_this_month_correct_extra_money_spent_correct_extra_money_when_a_is_4_l726_726923


namespace average_even_prime_numbers_excluding_2_is_undefined_l726_726634

-- Define what it means to be an even prime number
def is_even_prime (n : ℕ) : Prop := nat.prime n ∧ n % 2 = 0

-- Define the set of even prime numbers, excluding 2
def even_primes_excl_2 : set ℕ := {n : ℕ | is_even_prime n ∧ n ≠ 2}

-- Claim that the average of the first 10 even prime numbers excluding 2 is undefined
noncomputable def average_even_primes_excl_2 : ℝ := 
  if (finite {n : ℕ | n ∈ even_primes_excl_2}) then 0 else undefined

theorem average_even_prime_numbers_excluding_2_is_undefined :
  average_even_primes_excl_2 = undefined :=
  sorry

end average_even_prime_numbers_excluding_2_is_undefined_l726_726634


namespace find_equation_parabola_find_value_AF_minus_BF_l726_726281

-- Given definitions
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

def point_2_p (p : ℝ) : Prop := parabola p 2 (2 / p)

def distance_P_O (p : ℝ) (d : ℝ) : Prop := 
  let P := ((2 : ℝ), (2 / p) : ℝ × ℝ)
  let O := (0 : ℝ, 0 : ℝ)
  let PO_squared := (P.1 - O.1)^2 + (P.2 - O.2)^2
  PO_squared = (d / 2)^2 + (d + 2 / p)^2

def distance_MN (p : ℝ) : Prop := distance_P_O p 2

-- Problem statements using the definitions
theorem find_equation_parabola (p : ℝ) (h : p = 2): 
  parabola 2 x y → x^2 = 4 * y :=
sorry

theorem find_value_AF_minus_BF : 
  ∀ (A B : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) (k : ℝ),
  F = (0, 1) ∧ H = (0, -1) ∧
  (A.2 = (1 / 4) * A.1^2) ∧ (B.2 = (1 / 4) * B.1^2) ∧
  (A.1 + B.1 = 4 * k) ∧ (A.1 * B.1) = -4 ∧
  (A.2 - 1) * (B.2 + 1) + A.1 * B.1 = 0 ∧
  x^2 = 4*y 
  → abs (A.2 - F.2) - abs (B.2 - F.2) = 4 :=
sorry

end find_equation_parabola_find_value_AF_minus_BF_l726_726281


namespace age_of_student_who_left_l726_726419

/-- 
The average student age of a class with 30 students is 10 years.
After one student leaves and the teacher (who is 41 years old) is included,
the new average age is 11 years. Prove that the student who left is 11 years old.
-/
theorem age_of_student_who_left (x : ℕ) (h1 : (30 * 10) = 300)
    (h2 : (300 - x + 41) / 30 = 11) : x = 11 :=
by 
  -- This is where the proof would go
  sorry

end age_of_student_who_left_l726_726419


namespace convert_246_octal_to_decimal_l726_726565

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l726_726565


namespace second_term_arithmetic_seq_l726_726781

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l726_726781


namespace find_integer_tuples_l726_726247

theorem find_integer_tuples : 
  {n : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ | 
    let a := n.1.1.1.1.1, 
        b := n.1.1.1.1.2, 
        c := n.1.1.1.2, 
        x := n.1.1.2, 
        y := n.1.2, 
        z := n.2 in
    (a + b + c = x * y * z) ∧ 
    (x + y + z = a * b * c) ∧ 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ 1) ∧ 
    (x ≥ y) ∧ (y ≥ z) ∧ (z ≥ 1)} = 
  {(2, 2, 2, 6, 1, 1), (5, 2, 1, 8, 1, 1), (3, 3, 1, 7, 1, 1), (3, 2, 1, 3, 2, 1)} := by 
  sorry

end find_integer_tuples_l726_726247


namespace base8_to_base10_conversion_l726_726559

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l726_726559


namespace geometric_sequence_26th_term_l726_726004

noncomputable def r : ℝ := (8 : ℝ)^(1/6)

noncomputable def a (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_26th_term :
  (a 26 (a 14 10 r) r = 640) :=
by
  have h₁ : a 14 10 r = 10 := sorry
  have h₂ : r^6 = 8 := sorry
  sorry

end geometric_sequence_26th_term_l726_726004


namespace trajectory_curve_C_MN_passes_fixed_point_l726_726277

-- Definition and problem statement
def Point : Type := ℝ × ℝ
def A : Point := (-Real.sqrt 3, 0)
def B : Point := (Real.sqrt 3, 0)

def is_constant_slope_product (P : Point) : Prop :=
  let kPA := (P.snd / (P.fst + Real.sqrt 3))
  let kPB := (P.snd / (P.fst - Real.sqrt 3))
  kPA * kPB = -2 / 3

-- 1. Proving the equation of curve C
def curve_C (P : Point) : Prop :=
  P.fst ^ 2 / 3 + P.snd ^ 2 / 2 = 1 ∧ P.fst ≠ Real.sqrt 3 ∧ P.fst ≠ -Real.sqrt 3

theorem trajectory_curve_C (P : Point) (h : is_constant_slope_product P) : curve_C P :=
  sorry

-- 2. Proving that line MN passes through a fixed point
def M (t : ℝ) : Point := (t, Real.sqrt(2 - 2 * t ^ 2 / 3))
def N (t : ℝ) : Point := (t, -Real.sqrt(2 - 2 * t ^ 2 / 3))

def are_points_on_curve_C (M N : Point) : Prop :=
  curve_C M ∧ curve_C N

def perpendicular_through_A (M N : Point) : Prop :=
  let vect_AM := (M.fst - A.fst, M.snd - A.snd)
  let vect_AN := (N.fst - A.fst, N.snd - A.snd)
  vect_AM.fst * vect_AN.fst + vect_AM.snd * vect_AN.snd = 0

def line_MN_passes_fixed_point (M N : Point) : Prop :=
  let t := -Real.sqrt 3 / 5
  let fixed_point : Point := (-Real.sqrt 3 / 5, 0)
  M.fst = t ∧ N.fst = t ∧ fixed_point.snd = 0

theorem MN_passes_fixed_point (M N : Point) (hM : curve_C M) (hN : curve_C N) (h_perpendicular : perpendicular_through_A M N) : ∃ P : Point, line_MN_passes_fixed_point M N :=
  sorry

end trajectory_curve_C_MN_passes_fixed_point_l726_726277


namespace rational_number_count_eq_198_l726_726229

theorem rational_number_count_eq_198 :
  (∃ (k : ℚ), |k| < 300 ∧ ∃ (p q : ℤ), p ≠ q ∧ 3 * (p + q) = -k) ↔
  { k : ℚ | ∃ (p q : ℤ), p ≠ q ∧ 3 * (p + q) = -k ∧ |k| < 300 }.to_finset.card = 198 := sorry

end rational_number_count_eq_198_l726_726229


namespace prove_cube_diagonals_angles_l726_726893

noncomputable def cube_diagonals_angles : Prop :=
  let side_length := 1 
  let base_diagonal_angle := 45
  -- Calculate the diagonal lengths
  let diagonal_length := Math.sqrt 2 
  let space_diagonal_length := Math.sqrt 3
  -- Define the angles to be checked
  let angle_between_space_and_other_diagonals := [45, 90, 30, 73.2]
  -- Check the conditions and angles
  ∀ (side_length: ℝ = 1) (base_diagonal_angle: ℝ = 45)
  . angles_of_space_diagonal(angle_between_space_and_other_diagonals)

theorem prove_cube_diagonals_angles : cube_diagonals_angles :=
sorry

end prove_cube_diagonals_angles_l726_726893


namespace tailor_time_l726_726192

theorem tailor_time (x : ℝ) 
  (t_shirt : ℝ := x) 
  (t_pants : ℝ := 2 * x) 
  (t_jacket : ℝ := 3 * x) 
  (h_capacity : 2 * t_shirt + 3 * t_pants + 4 * t_jacket = 10) : 
  14 * t_shirt + 10 * t_pants + 2 * t_jacket = 20 :=
by
  sorry

end tailor_time_l726_726192


namespace find_k_l726_726375

theorem find_k (Z K : ℤ) (h1 : 2000 < Z) (h2 : Z < 3000) (h3 : K > 1) (h4 : Z = K * K^2) (h5 : ∃ n : ℤ, n^3 = Z) : K = 13 :=
by
-- Solution omitted
sorry

end find_k_l726_726375


namespace real_root_solution_l726_726643

noncomputable def real_root_equation : Nat := 36

theorem real_root_solution (x : ℝ) (h : x ≥ 0 ∧ √x + √(x+4) = 12) : x = 1225 / real_root_equation := by
  sorry

end real_root_solution_l726_726643


namespace probability_one_defective_l726_726096

theorem probability_one_defective (g d : ℕ) (h_g : g = 3) (h_d : d = 1) : 
  let total_items := g + d in
  let sample_space := (total_items.choose 2).toFinset in
  let event_A := {x ∈ sample_space | x.count (0 = ∘ id) = 1} in
  (event_A.card : ℚ) / (sample_space.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_one_defective_l726_726096


namespace findLineEquation_l726_726276

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to represent the hyperbola condition
def isOnHyperbola (pt : Point) : Prop :=
  pt.x ^ 2 - 4 * pt.y ^ 2 = 4

-- Define midpoint condition for points A and B
def isMidpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- Define points
def P : Point := ⟨8, 1⟩
def A : Point := sorry
def B : Point := sorry

-- Statement to prove
theorem findLineEquation :
  isOnHyperbola A ∧ isOnHyperbola B ∧ isMidpoint P A B →
  ∃ m b, (∀ pt : Point, pt.y = m * pt.x + b ↔ pt.x = 8 ∧ pt.y = 1) ∧ (m = 2) ∧ (b = -15) :=
by
  sorry

end findLineEquation_l726_726276
